use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaEvent, CudaStream, DeviceBuffer, Sm12xFp4DeviceGemmVector, Sm12xFp4DeviceGemmWeight,
    Sm12xFp4GemmVector, Sm12xFp4GemmWeight, Sm12xKvAttentionWorkspace, Sm12xKvCache,
    cached_gqa_attention_f32_into_on_stream, cached_gqa_attention_nvfp4_into_on_stream,
    device_weight_gemv_native_vector_on_stream, device_weight_gemv_on_stream,
    quantize_dynamic_vector_on_stream, quantize_nvfp4_simple_scales_f32_into_on_stream,
    softmax_f32_in_place_on_stream,
};
use std::time::Duration;

const Q_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;

struct KvAttentionBench<const CACHE_LEN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<f32>,
    key_f32: DeviceBuffer<f32>,
    value_f32: DeviceBuffer<f32>,
    key_nvfp4: DeviceBuffer<u8>,
    key_scales: DeviceBuffer<u8>,
    value_nvfp4: DeviceBuffer<u8>,
    value_scales: DeviceBuffer<u8>,
    f32_output: DeviceBuffer<f32>,
    nvfp4_output: DeviceBuffer<f32>,
}

impl<const CACHE_LEN: usize> BenchContext for KvAttentionBench<CACHE_LEN> {
    fn prepare(_num_chunks: usize) -> Self {
        let cache_values = CACHE_LEN * KV_HEADS * HEAD_DIM;
        let values = |factor| {
            (0..cache_values)
                .map(|index| ((index * factor % 509) as f32 - 254.0) / 256.0)
                .collect::<Vec<_>>()
        };
        let stream = CudaStream::new_non_blocking().expect("stream");
        let key_f32 = DeviceBuffer::from_host(&values(29)).expect("key cache");
        let value_f32 = DeviceBuffer::from_host(&values(43)).expect("value cache");
        let mut key_nvfp4 = DeviceBuffer::zeroed(cache_values.div_ceil(2)).expect("packed key");
        let mut key_scales = DeviceBuffer::zeroed(cache_values.div_ceil(16)).expect("key scales");
        let mut value_nvfp4 = DeviceBuffer::zeroed(cache_values.div_ceil(2)).expect("packed value");
        let mut value_scales =
            DeviceBuffer::zeroed(cache_values.div_ceil(16)).expect("value scales");
        quantize_nvfp4_simple_scales_f32_into_on_stream(
            &key_f32,
            &mut key_nvfp4,
            &mut key_scales,
            &stream,
        )
        .expect("quantize key");
        quantize_nvfp4_simple_scales_f32_into_on_stream(
            &value_f32,
            &mut value_nvfp4,
            &mut value_scales,
            &stream,
        )
        .expect("quantize value");
        stream.synchronize().expect("quantize sync");
        let mut bench = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            query: DeviceBuffer::from_host(
                &(0..Q_HEADS * HEAD_DIM)
                    .map(|index| ((index * 17 % 251) as f32 - 125.0) / 128.0)
                    .collect::<Vec<_>>(),
            )
            .expect("query"),
            key_f32,
            value_f32,
            key_nvfp4,
            key_scales,
            value_nvfp4,
            value_scales,
            f32_output: DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("f32 output"),
            nvfp4_output: DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("nvfp4 output"),
        };
        bench.assert_approximate_correctness();
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl<const CACHE_LEN: usize> KvAttentionBench<CACHE_LEN> {
    fn assert_approximate_correctness(&mut self) {
        cached_gqa_attention_f32_into_on_stream(
            &self.query,
            &self.key_f32,
            &self.value_f32,
            self.f32_output.output(),
            CACHE_LEN,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &self.stream,
        )
        .expect("f32 attention");
        cached_gqa_attention_nvfp4_into_on_stream(
            &self.query,
            &self.key_nvfp4,
            &self.key_scales,
            &self.value_nvfp4,
            &self.value_scales,
            self.nvfp4_output.output(),
            CACHE_LEN,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &self.stream,
        )
        .expect("NVFP4 attention");
        self.stream.synchronize().expect("attention sync");
        let reference = self
            .f32_output
            .copy_to_host(&self.stream)
            .expect("copy f32 output");
        let actual = self
            .nvfp4_output
            .copy_to_host(&self.stream)
            .expect("copy NVFP4 output");
        let max_abs = reference
            .iter()
            .zip(actual.iter())
            .map(|(reference, actual)| (reference - actual).abs())
            .fold(0.0f32, f32::max);
        let actual_max = actual
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.20,
            "NVFP4 KV attention error too large at {CACHE_LEN} tokens: max_abs={max_abs} actual_max={actual_max}"
        );
    }
}

fn f32_sample<const CACHE_LEN: usize>(
    ctx: &mut KvAttentionBench<CACHE_LEN>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk {
        cached_gqa_attention_f32_into_on_stream(
            &ctx.query,
            &ctx.key_f32,
            &ctx.value_f32,
            ctx.f32_output.output(),
            CACHE_LEN,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &ctx.stream,
        )
        .expect("f32 attention");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    black_box(ctx.f32_output.as_const_ptr());
    BenchSampleResult::operations(chunk as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk as f64,
        "ms",
    ))
}

fn nvfp4_sample<const CACHE_LEN: usize>(
    ctx: &mut KvAttentionBench<CACHE_LEN>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk {
        cached_gqa_attention_nvfp4_into_on_stream(
            &ctx.query,
            &ctx.key_nvfp4,
            &ctx.key_scales,
            &ctx.value_nvfp4,
            &ctx.value_scales,
            ctx.nvfp4_output.output(),
            CACHE_LEN,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &ctx.stream,
        )
        .expect("NVFP4 attention");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    black_box(ctx.nvfp4_output.as_const_ptr());
    BenchSampleResult::operations(chunk as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk as f64,
        "ms",
    ))
}

/// Persistent native-tile K cache times an already quantized decode query.
///
/// This is deliberately QK-only: it establishes the useful CUDA-MMA shape
/// before a full softmax/PV implementation adds another layout and reduction.
struct TileQkBench<const CACHE_LEN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    key: Sm12xFp4DeviceGemmWeight,
    query: Sm12xFp4DeviceGemmVector,
    output: DeviceBuffer<f32>,
}

impl<const CACHE_LEN: usize> BenchContext for TileQkBench<CACHE_LEN> {
    fn prepare(_num_chunks: usize) -> Self {
        let key_values = (0..CACHE_LEN * HEAD_DIM)
            .map(|index| ((index * 29 % 509) as f32 - 254.0) / 256.0)
            .collect::<Vec<_>>();
        let query_values = (0..HEAD_DIM)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) / 128.0)
            .collect::<Vec<_>>();
        let key =
            Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(CACHE_LEN, HEAD_DIM, &key_values)
                .expect("pack K cache");
        let query =
            Sm12xFp4GemmVector::quantize_f32_k16(HEAD_DIM, &query_values).expect("pack query");
        let expected = key
            .dequantized_row_major
            .chunks_exact(HEAD_DIM)
            .map(|row| {
                row.iter()
                    .zip(&query.dequantized)
                    .map(|(key, query)| key * query)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut bench = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            key: key.weight.to_device().expect("upload K cache"),
            query: query.vector.to_device().expect("upload query"),
            output: DeviceBuffer::zeroed(CACHE_LEN).expect("output"),
        };
        device_weight_gemv_on_stream(
            &bench.key,
            &bench.query,
            bench.output.output(),
            &bench.stream,
        )
        .expect("CUDA MMA QK");
        let actual = bench
            .output
            .copy_to_host(&bench.stream)
            .expect("copy QK output");
        let max_abs = expected
            .iter()
            .zip(actual.iter())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.01,
            "SM12x tiled QK differs from packed reference at {CACHE_LEN} rows: max_abs={max_abs}"
        );
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn tile_qk_sample<const CACHE_LEN: usize>(
    ctx: &mut TileQkBench<CACHE_LEN>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk {
        device_weight_gemv_on_stream(&ctx.key, &ctx.query, ctx.output.output(), &ctx.stream)
            .expect("CUDA MMA QK");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    black_box(ctx.output.as_const_ptr());
    BenchSampleResult::operations(chunk as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk as f64,
        "ms",
    ))
}

/// One-head FP4 attention with K in token-major QK tiles and V transposed into
/// dimension-major PV tiles. Cache packing happens in `prepare`; timed work is
/// QK MMA, f32 softmax, probability quantization, and PV MMA.
struct TileAttentionBench<const CACHE_LEN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    key: Sm12xFp4DeviceGemmWeight,
    value: Sm12xFp4DeviceGemmWeight,
    query: Sm12xFp4DeviceGemmVector,
    scores: DeviceBuffer<f32>,
    probability_tiles: DeviceBuffer<u8>,
    probability_scales: DeviceBuffer<u32>,
    output: DeviceBuffer<f32>,
}

impl<const CACHE_LEN: usize> TileAttentionBench<CACHE_LEN> {
    fn run_once(&mut self) {
        device_weight_gemv_on_stream(&self.key, &self.query, self.scores.output(), &self.stream)
            .expect("CUDA MMA QK");
        softmax_f32_in_place_on_stream(&mut self.scores, &self.stream).expect("softmax");
        quantize_dynamic_vector_on_stream(
            &self.scores,
            &mut self.probability_tiles,
            &mut self.probability_scales,
            &self.stream,
        )
        .expect("quantize probabilities");
        device_weight_gemv_native_vector_on_stream(
            &self.value,
            &self.probability_tiles,
            &self.probability_scales,
            self.output.output(),
            &self.stream,
        )
        .expect("CUDA MMA PV");
    }
}

impl<const CACHE_LEN: usize> BenchContext for TileAttentionBench<CACHE_LEN> {
    fn prepare(_num_chunks: usize) -> Self {
        let query = (0..HEAD_DIM)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) / (128.0 * (HEAD_DIM as f32).sqrt()))
            .collect::<Vec<_>>();
        let key_values = (0..CACHE_LEN * HEAD_DIM)
            .map(|index| ((index * 29 % 509) as f32 - 254.0) / 256.0)
            .collect::<Vec<_>>();
        let value_values = (0..CACHE_LEN * HEAD_DIM)
            .map(|index| ((index * 43 % 509) as f32 - 254.0) / 256.0)
            .collect::<Vec<_>>();
        let key =
            Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(CACHE_LEN, HEAD_DIM, &key_values)
                .expect("pack K cache");
        let mut transposed_value = vec![0.0f32; CACHE_LEN * HEAD_DIM];
        for token in 0..CACHE_LEN {
            for dim in 0..HEAD_DIM {
                transposed_value[dim * CACHE_LEN + token] = value_values[token * HEAD_DIM + dim];
            }
        }
        let value = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
            HEAD_DIM,
            CACHE_LEN,
            &transposed_value,
        )
        .expect("pack transposed V cache");
        let query_pack =
            Sm12xFp4GemmVector::quantize_f32_k16(HEAD_DIM, &query).expect("pack query");
        let mut reference_scores = key
            .dequantized_row_major
            .chunks_exact(HEAD_DIM)
            .map(|row| {
                row.iter()
                    .zip(&query_pack.dequantized)
                    .map(|(key, query)| key * query)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let maximum = reference_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum = reference_scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f32>();
        for score in &mut reference_scores {
            *score /= sum;
        }
        let reference = (0..HEAD_DIM)
            .map(|dim| {
                reference_scores
                    .iter()
                    .enumerate()
                    .map(|(token, probability)| {
                        probability * value.dequantized_row_major[dim * CACHE_LEN + token]
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut bench = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            key: key.weight.to_device().expect("upload K"),
            value: value.weight.to_device().expect("upload V"),
            query: query_pack.vector.to_device().expect("upload Q"),
            scores: DeviceBuffer::zeroed(CACHE_LEN).expect("scores"),
            probability_tiles: DeviceBuffer::zeroed(CACHE_LEN / 64 * 512)
                .expect("probability tiles"),
            probability_scales: DeviceBuffer::zeroed(CACHE_LEN / 64).expect("probability scales"),
            output: DeviceBuffer::zeroed(HEAD_DIM).expect("output"),
        };
        bench.run_once();
        let actual = bench
            .output
            .copy_to_host(&bench.stream)
            .expect("copy output");
        let max_abs = reference
            .iter()
            .zip(actual.iter())
            .map(|(reference, actual)| (reference - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.05,
            "SM12x tiled attention differs from packed reference at {CACHE_LEN}: max_abs={max_abs}"
        );
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn tile_attention_sample<const CACHE_LEN: usize>(
    ctx: &mut TileAttentionBench<CACHE_LEN>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk {
        ctx.run_once();
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    black_box(ctx.output.as_const_ptr());
    BenchSampleResult::operations(chunk as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk as f64,
        "ms",
    ))
}

const PREFILL_PREFIX: usize = 2_048;
const PREFILL_ROWS: usize = 128;

struct CausalPrefillBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    serial_cache: Sm12xKvCache,
    serial_workspace: Sm12xKvAttentionWorkspace,
    serial_output: DeviceBuffer<f32>,
    batched_cache: Sm12xKvCache,
    batched_workspace: Sm12xKvAttentionWorkspace,
    batched_output: DeviceBuffer<f32>,
}

impl BenchContext for CausalPrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        let max_tokens = PREFILL_PREFIX + PREFILL_ROWS;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let key = DeviceBuffer::from_host(
            &(0..max_tokens * kv_width)
                .map(|index| ((index * 29 % 509) as f32 - 254.0) / 384.0)
                .collect::<Vec<_>>(),
        )
        .expect("prefill key");
        let value = DeviceBuffer::from_host(
            &(0..max_tokens * kv_width)
                .map(|index| ((index * 43 % 509) as f32 - 254.0) / 448.0)
                .collect::<Vec<_>>(),
        )
        .expect("prefill value");
        let query = DeviceBuffer::from_host(
            &(0..max_tokens * q_width)
                .map(|index| ((index * 17 % 251) as f32 - 125.0) / 512.0)
                .collect::<Vec<_>>(),
        )
        .expect("prefill query");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut serial_cache =
            Sm12xKvCache::new(max_tokens, KV_HEADS, HEAD_DIM).expect("serial cache");
        serial_cache
            .append_rows_at_offset_on_stream(&key, &value, 0, PREFILL_PREFIX, &stream)
            .expect("serial prefix");
        let mut batched_cache =
            Sm12xKvCache::new(max_tokens, KV_HEADS, HEAD_DIM).expect("batched cache");
        batched_cache
            .append_rows_at_offset_on_stream(&key, &value, 0, PREFILL_PREFIX, &stream)
            .expect("batched prefix");
        stream.synchronize().expect("prefix sync");
        Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            key,
            value,
            query,
            serial_cache,
            serial_workspace: Sm12xKvAttentionWorkspace::new_gqa_batched(
                max_tokens, Q_HEADS, KV_HEADS, HEAD_DIM, 1,
            )
            .expect("serial workspace"),
            serial_output: DeviceBuffer::zeroed(max_tokens * q_width).expect("serial output"),
            batched_cache,
            batched_workspace: Sm12xKvAttentionWorkspace::new_gqa_batched(
                max_tokens, Q_HEADS, KV_HEADS, HEAD_DIM, 8,
            )
            .expect("batched workspace"),
            batched_output: DeviceBuffer::zeroed(max_tokens * q_width).expect("batched output"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl CausalPrefillBench {
    fn run_serial(&mut self) {
        self.serial_cache
            .truncate(PREFILL_PREFIX)
            .expect("truncate serial cache");
        self.serial_workspace
            .append_causal_rows_at_offset_into_on_stream(
                &mut self.serial_cache,
                &self.query,
                &self.key,
                &self.value,
                PREFILL_PREFIX,
                PREFILL_ROWS,
                None,
                self.serial_output.output(),
                &self.stream,
            )
            .expect("serial causal prefill");
    }

    fn run_batched(&mut self) {
        self.batched_cache
            .truncate(PREFILL_PREFIX)
            .expect("truncate batched cache");
        self.batched_workspace
            .append_causal_rows_at_offset_into_on_stream(
                &mut self.batched_cache,
                &self.query,
                &self.key,
                &self.value,
                PREFILL_PREFIX,
                PREFILL_ROWS,
                None,
                self.batched_output.output(),
                &self.stream,
            )
            .expect("batched causal prefill");
    }
}

fn causal_prefill_serial_sample(
    ctx: &mut CausalPrefillBench,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk {
        ctx.run_serial();
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    black_box(ctx.serial_output.as_const_ptr());
    BenchSampleResult::operations((chunk * PREFILL_ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk as f64,
        "ms/chunk",
    ))
}

fn causal_prefill_batched_sample(
    ctx: &mut CausalPrefillBench,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk {
        ctx.run_batched();
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    black_box(ctx.batched_output.as_const_ptr());
    BenchSampleResult::operations((chunk * PREFILL_ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk as f64,
        "ms/chunk",
    ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nvfp4-kv-attention".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<CausalPrefillBench>("SM12x compact causal prefill 2K+128", |group| {
                group.bench_sample("one_row_per_launch", causal_prefill_serial_sample);
                group.bench_sample("eight_rows_per_launch", causal_prefill_batched_sample);
            });
            runner.group::<KvAttentionBench<4_096>>("Qwen3.6 GQA 4K", |group| {
                group.bench_sample("f32_cache", f32_sample::<4_096>);
                group.bench_sample("nvfp4_cache_fused_decode", nvfp4_sample::<4_096>);
            });
            runner.group::<KvAttentionBench<32_768>>("Qwen3.6 GQA 32K", |group| {
                group.bench_sample("f32_cache", f32_sample::<32_768>);
                group.bench_sample("nvfp4_cache_fused_decode", nvfp4_sample::<32_768>);
            });
            runner.group::<KvAttentionBench<131_072>>("Qwen3.6 GQA 128K", |group| {
                group.bench_sample("f32_cache", f32_sample::<131_072>);
                group.bench_sample("nvfp4_cache_fused_decode", nvfp4_sample::<131_072>);
            });
            runner.group::<TileQkBench<4_096>>("SM12x FP4 QK tile 4K", |group| {
                group.bench_sample("m16n8k64_cuda_mma", tile_qk_sample::<4_096>);
            });
            runner.group::<TileQkBench<32_768>>("SM12x FP4 QK tile 32K", |group| {
                group.bench_sample("m16n8k64_cuda_mma", tile_qk_sample::<32_768>);
            });
            runner.group::<TileQkBench<131_072>>("SM12x FP4 QK tile 128K", |group| {
                group.bench_sample("m16n8k64_cuda_mma", tile_qk_sample::<131_072>);
            });
            runner.group::<TileAttentionBench<4_096>>("SM12x FP4 attention 4K", |group| {
                group.bench_sample("qk_softmax_pv_cuda_mma", tile_attention_sample::<4_096>);
            });
            runner.group::<TileAttentionBench<32_768>>("SM12x FP4 attention 32K", |group| {
                group.bench_sample("qk_softmax_pv_cuda_mma", tile_attention_sample::<32_768>);
            });
            runner.group::<TileAttentionBench<131_072>>("SM12x FP4 attention 128K", |group| {
                group.bench_sample("qk_softmax_pv_cuda_mma", tile_attention_sample::<131_072>);
            });
        },
    );
}
