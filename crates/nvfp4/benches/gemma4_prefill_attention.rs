use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{
    Bf16TnMatmulPlan, CublasLt, CudaEvent, CudaStream, DeviceBuffer, GemmShape,
    Gemma4LocalPrefillAttention, Nvfp4Matrix, Sm12xKvCache,
    causal_window_softmax_f32_to_bf16_on_stream, pack_token_heads_bf16_into_on_stream,
    unpack_heads_f32_into_on_stream,
    unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream,
};
use std::time::Duration;

const TOKENS: usize = 2_613;
const Q_HEADS: usize = 16;
const WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;

struct GemmaLocalBoundary<const PREFIX: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    bf16_cache: Sm12xKvCache,
    compact_cache: Sm12xKvCache,
    packed_query: DeviceBuffer<u16>,
    packed_key: DeviceBuffer<u16>,
    packed_value: DeviceBuffer<u16>,
    bf16_output: DeviceBuffer<u16>,
    compact_output: DeviceBuffer<u16>,
    bf16_quantized: Nvfp4Matrix,
    compact_quantized: Nvfp4Matrix,
    local: Gemma4LocalPrefillAttention,
}

impl<const PREFIX: usize, const ROWS: usize> BenchContext for GemmaLocalBoundary<PREFIX, ROWS> {
    fn prepare(_num_chunks: usize) -> Self {
        const KV_WIDTH: usize = 8 * 256;
        const Q_WIDTH: usize = Q_HEADS * 256;
        let cache_tokens = PREFIX + ROWS;
        let values = |len: usize, factor: usize| {
            DeviceBuffer::from_host(
                &(0..len)
                    .map(|index| ((index * factor % 509) as f32 - 254.0) / 1_024.0)
                    .collect::<Vec<_>>(),
            )
        };
        let query = values(ROWS * Q_WIDTH, 17).expect("query");
        let key = values(cache_tokens * KV_WIDTH, 29).expect("key");
        let value = values(cache_tokens * KV_WIDTH, 43).expect("value");
        let mut bf16_cache = Sm12xKvCache::new(cache_tokens, 8, 256).expect("BF16 cache");
        let mut compact_cache = Sm12xKvCache::new(cache_tokens, 8, 256).expect("compact cache");
        let stream = CudaStream::new_non_blocking().expect("stream");
        if PREFIX != 0 {
            bf16_cache
                .append_rows_at_offset_on_stream(&key, &value, 0, PREFIX, &stream)
                .expect("BF16 prefix");
            compact_cache
                .append_rows_at_offset_on_stream(&key, &value, 0, PREFIX, &stream)
                .expect("compact prefix");
            stream.synchronize().expect("prefix synchronize");
        }
        Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            query,
            key,
            value,
            bf16_cache,
            compact_cache,
            packed_query: DeviceBuffer::zeroed(ROWS * Q_WIDTH).expect("packed query"),
            packed_key: DeviceBuffer::zeroed(cache_tokens * KV_WIDTH).expect("packed key"),
            packed_value: DeviceBuffer::zeroed(cache_tokens * KV_WIDTH).expect("packed value"),
            bf16_output: DeviceBuffer::zeroed(ROWS * Q_WIDTH).expect("BF16 output"),
            compact_output: DeviceBuffer::zeroed(ROWS * Q_WIDTH).expect("compact output"),
            bf16_quantized: Nvfp4Matrix::zeroed_col_major(Q_WIDTH, ROWS)
                .expect("BF16 quantized output"),
            compact_quantized: Nvfp4Matrix::zeroed_col_major(Q_WIDTH, ROWS)
                .expect("compact quantized output"),
            local: Gemma4LocalPrefillAttention::new().expect("local attention"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl<const PREFIX: usize, const ROWS: usize> GemmaLocalBoundary<PREFIX, ROWS> {
    fn run_bf16(&mut self) {
        self.bf16_cache
            .truncate(PREFIX)
            .expect("truncate BF16 cache");
        pack_token_heads_bf16_into_on_stream(
            &self.query,
            self.packed_query.output(),
            ROWS,
            Q_HEADS,
            256,
            &self.stream,
        )
        .expect("pack query");
        if PREFIX == 0 {
            self.bf16_cache
                .append_initial_rows_and_stage_bf16_on_stream(
                    &self.key,
                    &self.value,
                    0,
                    ROWS,
                    self.packed_key.output(),
                    self.packed_value.output(),
                    &self.stream,
                )
                .expect("append and stage cache");
        } else {
            self.bf16_cache
                .append_rows_at_offset_on_stream(&self.key, &self.value, PREFIX, ROWS, &self.stream)
                .expect("append cache");
            self.bf16_cache
                .unpack_bf16_on_stream(
                    self.packed_key.output(),
                    self.packed_value.output(),
                    &self.stream,
                )
                .expect("unpack cache");
        }
        self.local
            .run_on_stream(
                &self.packed_query,
                &self.packed_key,
                &self.packed_value,
                self.bf16_output.output(),
                ROWS,
                PREFIX + ROWS,
                PREFIX,
                &self.stream,
            )
            .expect("BF16 attention");
        unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
            &self.bf16_output,
            &mut self.bf16_quantized,
            ROWS,
            Q_HEADS,
            256,
            0,
            1.0,
            &self.stream,
        )
        .expect("quantize BF16 output");
    }

    fn run_compact(&mut self) {
        self.compact_cache
            .truncate(PREFIX)
            .expect("truncate compact cache");
        pack_token_heads_bf16_into_on_stream(
            &self.query,
            self.packed_query.output(),
            ROWS,
            Q_HEADS,
            256,
            &self.stream,
        )
        .expect("pack query");
        self.compact_cache
            .append_rows_at_offset_on_stream(&self.key, &self.value, PREFIX, ROWS, &self.stream)
            .expect("append compact cache");
        self.local
            .run_compact_on_stream(
                &self.packed_query,
                &self.compact_cache,
                self.compact_output.output(),
                ROWS,
                PREFIX,
                &self.stream,
            )
            .expect("compact attention");
        unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
            &self.compact_output,
            &mut self.compact_quantized,
            ROWS,
            Q_HEADS,
            256,
            0,
            1.0,
            &self.stream,
        )
        .expect("quantize compact output");
    }
}

fn boundary_bf16_sample<const PREFIX: usize, const ROWS: usize>(
    context: &mut GemmaLocalBoundary<PREFIX, ROWS>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context.run_bf16();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn boundary_compact_sample<const PREFIX: usize, const ROWS: usize>(
    context: &mut GemmaLocalBoundary<PREFIX, ROWS>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context.run_compact();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

struct GemmaPrefillAttention<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize> {
    lt: CublasLt,
    qk: Bf16TnMatmulPlan,
    pv: Bf16TnMatmulPlan,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    cache: Sm12xKvCache,
    packed_query: DeviceBuffer<u16>,
    packed_key: DeviceBuffer<u16>,
    packed_value: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    packed_probabilities: DeviceBuffer<u16>,
    packed_output: DeviceBuffer<f32>,
    fused_output: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
    local: Option<Gemma4LocalPrefillAttention>,
}

impl<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize> BenchContext
    for GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>
{
    fn prepare(_num_chunks: usize) -> Self {
        let q_len = TOKENS * Q_HEADS * HEAD_DIM;
        let kv_len = TOKENS * KV_HEADS * HEAD_DIM;
        let values = |len: usize, factor: usize| {
            (0..len)
                .map(|index| ((index * factor % 509) as f32 - 254.0) / 1024.0)
                .collect::<Vec<_>>()
        };
        let lt = CublasLt::new().expect("cuBLASLt");
        let queries_per_kv = Q_HEADS / KV_HEADS;
        let qk = Bf16TnMatmulPlan::new_strided_batch(
            &lt,
            GemmShape::new(TOKENS, TOKENS * queries_per_kv, HEAD_DIM),
            KV_HEADS,
            TOKENS * HEAD_DIM,
            queries_per_kv * TOKENS * HEAD_DIM,
            queries_per_kv * TOKENS * TOKENS,
            WORKSPACE_LIMIT,
        )
        .expect("QK plan");
        let pv = Bf16TnMatmulPlan::new_strided_batch(
            &lt,
            GemmShape::new(HEAD_DIM, TOKENS * queries_per_kv, TOKENS),
            KV_HEADS,
            HEAD_DIM * TOKENS,
            queries_per_kv * TOKENS * TOKENS,
            queries_per_kv * TOKENS * HEAD_DIM,
            WORKSPACE_LIMIT,
        )
        .expect("PV plan");
        Self {
            lt,
            qk,
            pv,
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            query: DeviceBuffer::from_host(&values(q_len, 17)).expect("query"),
            key: DeviceBuffer::from_host(&values(kv_len, 29)).expect("key"),
            value: DeviceBuffer::from_host(&values(kv_len, 43)).expect("value"),
            cache: Sm12xKvCache::new(TOKENS, KV_HEADS, HEAD_DIM).expect("cache"),
            packed_query: DeviceBuffer::zeroed(q_len).expect("packed query"),
            packed_key: DeviceBuffer::zeroed(kv_len).expect("packed key"),
            packed_value: DeviceBuffer::zeroed(kv_len).expect("packed value"),
            scores: DeviceBuffer::zeroed(Q_HEADS * TOKENS * TOKENS).expect("scores"),
            packed_probabilities: DeviceBuffer::zeroed(Q_HEADS * TOKENS * TOKENS)
                .expect("packed probabilities"),
            packed_output: DeviceBuffer::zeroed(q_len).expect("packed output"),
            fused_output: DeviceBuffer::zeroed(q_len).expect("fused output"),
            output: DeviceBuffer::zeroed(q_len).expect("output"),
            local: (KV_HEADS == 8 && HEAD_DIM == 256 && WINDOW == 1_024)
                .then(Gemma4LocalPrefillAttention::new)
                .transpose()
                .expect("local attention"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn fused_local_sample(
    context: &mut GemmaPrefillAttention<8, 256, 1_024>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    pack_token_heads_bf16_into_on_stream(
        &context.query,
        context.packed_query.output(),
        TOKENS,
        Q_HEADS,
        256,
        &context.stream,
    )
    .expect("pack query");
    context.cache.truncate(0).expect("truncate cache");
    context
        .cache
        .append_rows_at_offset_on_stream(&context.key, &context.value, 0, TOKENS, &context.stream)
        .expect("append cache");
    context
        .cache
        .unpack_bf16_on_stream(
            context.packed_key.output(),
            context.packed_value.output(),
            &context.stream,
        )
        .expect("unpack cache");
    context.stream.synchronize().expect("prepare synchronize");

    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context
            .local
            .as_ref()
            .expect("local attention")
            .run_on_stream(
                &context.packed_query,
                &context.packed_key,
                &context.packed_value,
                context.fused_output.output(),
                TOKENS,
                TOKENS,
                0,
                &context.stream,
            )
            .expect("fused attention");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    black_box(context.fused_output.as_const_ptr());
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn compact_local_sample(
    context: &mut GemmaPrefillAttention<8, 256, 1_024>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    pack_token_heads_bf16_into_on_stream(
        &context.query,
        context.packed_query.output(),
        TOKENS,
        Q_HEADS,
        256,
        &context.stream,
    )
    .expect("pack query");
    context.cache.truncate(0).expect("truncate cache");
    context
        .cache
        .append_rows_at_offset_on_stream(&context.key, &context.value, 0, TOKENS, &context.stream)
        .expect("append cache");
    context.stream.synchronize().expect("prepare synchronize");

    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context
            .local
            .as_ref()
            .expect("local attention")
            .run_compact_on_stream(
                &context.packed_query,
                &context.cache,
                context.fused_output.output(),
                TOKENS,
                0,
                &context.stream,
            )
            .expect("compact attention");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    black_box(context.fused_output.as_const_ptr());
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

impl<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>
    GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>
{
    fn run(&mut self) {
        pack_token_heads_bf16_into_on_stream(
            &self.query,
            self.packed_query.output(),
            TOKENS,
            Q_HEADS,
            HEAD_DIM,
            &self.stream,
        )
        .expect("pack query");
        self.cache.truncate(0).expect("truncate cache");
        self.cache
            .append_rows_at_offset_on_stream(&self.key, &self.value, 0, TOKENS, &self.stream)
            .expect("append cache");
        self.cache
            .unpack_bf16_on_stream(
                self.packed_key.output(),
                self.packed_value.output(),
                &self.stream,
            )
            .expect("unpack cache");

        self.qk
            .run_on_stream(
                &self.lt,
                &self.packed_key,
                &self.packed_query,
                self.scores.output(),
                &self.stream,
            )
            .expect("QK");
        causal_window_softmax_f32_to_bf16_on_stream(
            &self.scores,
            self.packed_probabilities.output(),
            TOKENS,
            TOKENS,
            0,
            Q_HEADS,
            HEAD_DIM,
            (WINDOW != 0).then_some(WINDOW),
            &self.stream,
        )
        .expect("softmax");
        self.pv
            .run_on_stream(
                &self.lt,
                &self.packed_value,
                &self.packed_probabilities,
                self.packed_output.output(),
                &self.stream,
            )
            .expect("PV");
        unpack_heads_f32_into_on_stream(
            &self.packed_output,
            self.output.output(),
            TOKENS,
            Q_HEADS,
            HEAD_DIM,
            &self.stream,
        )
        .expect("unpack output");
    }
}

fn sample<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>(
    context: &mut GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context.run();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    black_box(context.output.as_const_ptr());
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn append_sample<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>(
    context: &mut GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context.cache.truncate(0).expect("truncate cache");
        context
            .cache
            .append_rows_at_offset_on_stream(
                &context.key,
                &context.value,
                0,
                TOKENS,
                &context.stream,
            )
            .expect("append cache");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn unpack_sample<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>(
    context: &mut GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context.cache.truncate(0).expect("truncate cache");
    context
        .cache
        .append_rows_at_offset_on_stream(&context.key, &context.value, 0, TOKENS, &context.stream)
        .expect("append cache");
    context.stream.synchronize().expect("append synchronize");
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context
            .cache
            .unpack_bf16_on_stream(
                context.packed_key.output(),
                context.packed_value.output(),
                &context.stream,
            )
            .expect("unpack cache");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn qk_sample<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>(
    context: &mut GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context
            .qk
            .run_on_stream(
                &context.lt,
                &context.packed_key,
                &context.packed_query,
                context.scores.output(),
                &context.stream,
            )
            .expect("QK");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn softmax_sample<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>(
    context: &mut GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        causal_window_softmax_f32_to_bf16_on_stream(
            &context.scores,
            context.packed_probabilities.output(),
            TOKENS,
            TOKENS,
            0,
            Q_HEADS,
            HEAD_DIM,
            (WINDOW != 0).then_some(WINDOW),
            &context.stream,
        )
        .expect("softmax");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn pv_sample<const KV_HEADS: usize, const HEAD_DIM: usize, const WINDOW: usize>(
    context: &mut GemmaPrefillAttention<KV_HEADS, HEAD_DIM, WINDOW>,
    chunk: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk {
        context
            .pv
            .run_on_stream(
                &context.lt,
                &context.packed_value,
                &context.packed_probabilities,
                context.packed_output.output(),
                &context.stream,
            )
            .expect("PV");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((chunk * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk as f64,
        "ms/chunk",
    ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("gemma4-prefill-attention".to_string()),
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
            runner.group::<GemmaLocalBoundary<0, 2_613>>(
                "Gemma 4 local boundary cold 2613",
                |group| {
                    group.bench_sample("BF16 staged boundary", boundary_bf16_sample::<0, 2_613>);
                    group.bench_sample(
                        "compact direct boundary",
                        boundary_compact_sample::<0, 2_613>,
                    );
                },
            );
            runner.group::<GemmaLocalBoundary<32_768, 512>>(
                "Gemma 4 local boundary 32K+512",
                |group| {
                    group.bench_sample("BF16 staged boundary", boundary_bf16_sample::<32_768, 512>);
                    group.bench_sample(
                        "compact direct boundary",
                        boundary_compact_sample::<32_768, 512>,
                    );
                },
            );
            runner.group::<GemmaLocalBoundary<8_192, 512>>(
                "Gemma 4 local boundary 8K+512",
                |group| {
                    group.bench_sample("BF16 staged boundary", boundary_bf16_sample::<8_192, 512>);
                    group.bench_sample(
                        "compact direct boundary",
                        boundary_compact_sample::<8_192, 512>,
                    );
                },
            );
            runner.group::<GemmaLocalBoundary<4_096, 512>>(
                "Gemma 4 local boundary 4K+512",
                |group| {
                    group.bench_sample("BF16 staged boundary", boundary_bf16_sample::<4_096, 512>);
                    group.bench_sample(
                        "compact direct boundary",
                        boundary_compact_sample::<4_096, 512>,
                    );
                },
            );
            runner.group::<GemmaLocalBoundary<16_384, 512>>(
                "Gemma 4 local boundary 16K+512",
                |group| {
                    group.bench_sample("BF16 staged boundary", boundary_bf16_sample::<16_384, 512>);
                    group.bench_sample(
                        "compact direct boundary",
                        boundary_compact_sample::<16_384, 512>,
                    );
                },
            );
            runner.group::<GemmaPrefillAttention<8, 256, 1_024>>(
                "Gemma 4 local attention 2613",
                |group| {
                    group.bench_sample("fused local attention", fused_local_sample);
                    group.bench_sample("compact fused local attention", compact_local_sample);
                    group.bench_sample("full-score BF16 staged", sample::<8, 256, 1_024>);
                    group.bench_sample("QK", qk_sample::<8, 256, 1_024>);
                    group.bench_sample("softmax", softmax_sample::<8, 256, 1_024>);
                    group.bench_sample("PV", pv_sample::<8, 256, 1_024>);
                    group.bench_sample("compact KV append", append_sample::<8, 256, 1_024>);
                    group.bench_sample("compact KV unpack", unpack_sample::<8, 256, 1_024>);
                },
            );
            runner.group::<GemmaPrefillAttention<2, 512, 0>>(
                "Gemma 4 global attention 2613",
                |group| {
                    group.bench_sample("full-score BF16 staged", sample::<2, 512, 0>);
                    group.bench_sample("QK", qk_sample::<2, 512, 0>);
                    group.bench_sample("softmax", softmax_sample::<2, 512, 0>);
                    group.bench_sample("PV", pv_sample::<2, 512, 0>);
                    group.bench_sample("compact KV append", append_sample::<2, 512, 0>);
                    group.bench_sample("compact KV unpack", unpack_sample::<2, 512, 0>);
                },
            );
        },
    );
}
