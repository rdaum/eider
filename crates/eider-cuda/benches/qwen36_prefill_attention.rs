use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaEvent, CudaStream, DeviceBuffer, GemmShape, Sm12xKvCache,
    causal_window_softmax_f32_to_bf16_on_stream, pack_token_heads_bf16_at_offset_into_on_stream,
    pack_token_heads_bf16_into_on_stream, prefill_gqa_attention_f32_into, synchronize_device,
    unpack_heads_f32_at_offset_into_on_stream, unpack_heads_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const TOKENS: usize = 2_048;
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const QUERY_TILE_ROWS: usize = 256;
const WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;

struct AttentionTile {
    query_offset: usize,
    rows: usize,
    key_tokens: usize,
    qk: Bf16TnMatmulPlan,
    pv: Bf16TnMatmulPlan,
}

struct Qwen36PrefillAttentionBench {
    lt: CublasLt,
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
    probabilities: DeviceBuffer<u16>,
    packed_output: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    tiles: Vec<AttentionTile>,
}

impl Qwen36PrefillAttentionBench {
    fn new() -> Self {
        let q_width = Q_HEADS * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        let values = |len: usize, factor: usize| {
            (0..len)
                .map(|index| ((index * factor % 509) as f32 - 254.0) / 1_024.0)
                .collect::<Vec<_>>()
        };
        let lt = CublasLt::new().expect("cuBLASLt");
        let queries_per_kv = Q_HEADS / KV_HEADS;
        let mut tiles = Vec::new();
        let mut query_offset = 0;
        while query_offset < TOKENS {
            let rows = (TOKENS - query_offset).min(QUERY_TILE_ROWS);
            let key_tokens = query_offset + rows;
            let qk = Bf16TnMatmulPlan::new_strided_batch(
                &lt,
                GemmShape::new(key_tokens, rows * queries_per_kv, HEAD_DIM),
                KV_HEADS,
                TOKENS * HEAD_DIM,
                queries_per_kv * rows * HEAD_DIM,
                queries_per_kv * rows * key_tokens,
                WORKSPACE_LIMIT,
            )
            .expect("QK plan");
            let pv = Bf16TnMatmulPlan::new_strided_batch_with_a_leading_dimension(
                &lt,
                GemmShape::new(HEAD_DIM, rows * queries_per_kv, key_tokens),
                TOKENS,
                KV_HEADS,
                HEAD_DIM * TOKENS,
                queries_per_kv * rows * key_tokens,
                queries_per_kv * rows * HEAD_DIM,
                WORKSPACE_LIMIT,
            )
            .expect("PV plan");
            tiles.push(AttentionTile {
                query_offset,
                rows,
                key_tokens,
                qk,
                pv,
            });
            query_offset += rows;
        }
        Self {
            lt,
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            query: DeviceBuffer::from_host(&values(TOKENS * q_width, 17)).expect("query"),
            key: DeviceBuffer::from_host(&values(TOKENS * kv_width, 29)).expect("key"),
            value: DeviceBuffer::from_host(&values(TOKENS * kv_width, 43)).expect("value"),
            cache: Sm12xKvCache::new(TOKENS, KV_HEADS, HEAD_DIM).expect("cache"),
            packed_query: DeviceBuffer::zeroed(QUERY_TILE_ROWS * q_width).expect("packed query"),
            packed_key: DeviceBuffer::zeroed(TOKENS * kv_width).expect("packed key"),
            packed_value: DeviceBuffer::zeroed(TOKENS * kv_width).expect("packed value"),
            scores: DeviceBuffer::zeroed(QUERY_TILE_ROWS * Q_HEADS * TOKENS).expect("scores"),
            probabilities: DeviceBuffer::zeroed(QUERY_TILE_ROWS * Q_HEADS * TOKENS)
                .expect("probabilities"),
            packed_output: DeviceBuffer::zeroed(QUERY_TILE_ROWS * q_width).expect("packed output"),
            output: DeviceBuffer::zeroed(TOKENS * q_width).expect("output"),
            tiles,
        }
    }

    fn enqueue(&mut self) {
        self.cache.truncate(0).expect("truncate cache");
        self.cache
            .append_initial_rows_and_stage_bf16_on_stream(
                &self.key,
                &self.value,
                0,
                TOKENS,
                self.packed_key.output(),
                self.packed_value.output(),
                &self.stream,
            )
            .expect("append and stage cache");
        for tile in &self.tiles {
            pack_token_heads_bf16_at_offset_into_on_stream(
                &self.query,
                self.packed_query.output(),
                tile.rows,
                Q_HEADS,
                HEAD_DIM,
                tile.query_offset,
                &self.stream,
            )
            .expect("pack query tile");
            tile.qk
                .run_offsets_on_stream(
                    &self.lt,
                    &self.packed_key,
                    0,
                    &self.packed_query,
                    0,
                    self.scores.output(),
                    0,
                    &self.stream,
                )
                .expect("QK");
            causal_window_softmax_f32_to_bf16_on_stream(
                &self.scores,
                self.probabilities.output(),
                tile.rows,
                tile.key_tokens,
                tile.query_offset,
                Q_HEADS,
                HEAD_DIM,
                None,
                &self.stream,
            )
            .expect("softmax");
            tile.pv
                .run_offsets_on_stream(
                    &self.lt,
                    &self.packed_value,
                    0,
                    &self.probabilities,
                    0,
                    self.packed_output.output(),
                    0,
                    &self.stream,
                )
                .expect("PV");
            unpack_heads_f32_at_offset_into_on_stream(
                &self.packed_output,
                self.output.output(),
                tile.rows,
                Q_HEADS,
                HEAD_DIM,
                tile.query_offset,
                &self.stream,
            )
            .expect("unpack output tile");
        }
    }
}

impl BenchContext for Qwen36PrefillAttentionBench {
    fn prepare(_num_chunks: usize) -> Self {
        validate_tensor_core_attention();
        let mut context = Self::new();
        context.enqueue();
        context.stream.synchronize().expect("prepare synchronize");
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn validate_tensor_core_attention() {
    const VALIDATION_TOKENS: usize = 32;
    let q_width = Q_HEADS * HEAD_DIM;
    let kv_width = KV_HEADS * HEAD_DIM;
    let values = |len: usize, factor: usize| {
        (0..len)
            .map(|index| ((index * factor % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>()
    };
    let query = DeviceBuffer::from_host(&values(VALIDATION_TOKENS * q_width, 17)).expect("query");
    let key = DeviceBuffer::from_host(&values(VALIDATION_TOKENS * kv_width, 29)).expect("key");
    let value = DeviceBuffer::from_host(&values(VALIDATION_TOKENS * kv_width, 43)).expect("value");
    let mut reference = DeviceBuffer::zeroed(VALIDATION_TOKENS * q_width).expect("reference");
    prefill_gqa_attention_f32_into(
        &query,
        &key,
        &value,
        reference.output(),
        VALIDATION_TOKENS,
        0,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
    )
    .expect("reference attention");
    synchronize_device().expect("reference synchronize");

    let stream = CudaStream::new_non_blocking().expect("validation stream");
    let mut packed_query = DeviceBuffer::zeroed(VALIDATION_TOKENS * q_width).expect("packed query");
    let mut packed_key = DeviceBuffer::zeroed(VALIDATION_TOKENS * kv_width).expect("packed key");
    let mut packed_value =
        DeviceBuffer::zeroed(VALIDATION_TOKENS * kv_width).expect("packed value");
    pack_token_heads_bf16_into_on_stream(
        &query,
        packed_query.output(),
        VALIDATION_TOKENS,
        Q_HEADS,
        HEAD_DIM,
        &stream,
    )
    .expect("pack query");
    let mut cache =
        Sm12xKvCache::new(VALIDATION_TOKENS, KV_HEADS, HEAD_DIM).expect("validation cache");
    cache
        .append_initial_rows_and_stage_bf16_on_stream(
            &key,
            &value,
            0,
            VALIDATION_TOKENS,
            packed_key.output(),
            packed_value.output(),
            &stream,
        )
        .expect("stage validation cache");
    let queries_per_kv = Q_HEADS / KV_HEADS;
    let lt = CublasLt::new().expect("validation cuBLASLt");
    let qk = Bf16TnMatmulPlan::new_strided_batch(
        &lt,
        GemmShape::new(
            VALIDATION_TOKENS,
            VALIDATION_TOKENS * queries_per_kv,
            HEAD_DIM,
        ),
        KV_HEADS,
        VALIDATION_TOKENS * HEAD_DIM,
        queries_per_kv * VALIDATION_TOKENS * HEAD_DIM,
        queries_per_kv * VALIDATION_TOKENS * VALIDATION_TOKENS,
        WORKSPACE_LIMIT,
    )
    .expect("validation QK plan");
    let mut scores =
        DeviceBuffer::zeroed(Q_HEADS * VALIDATION_TOKENS * VALIDATION_TOKENS).expect("scores");
    qk.run_on_stream(&lt, &packed_key, &packed_query, scores.output(), &stream)
        .expect("validation QK");
    let mut probabilities = DeviceBuffer::zeroed(Q_HEADS * VALIDATION_TOKENS * VALIDATION_TOKENS)
        .expect("probabilities");
    causal_window_softmax_f32_to_bf16_on_stream(
        &scores,
        probabilities.output(),
        VALIDATION_TOKENS,
        VALIDATION_TOKENS,
        0,
        Q_HEADS,
        HEAD_DIM,
        None,
        &stream,
    )
    .expect("validation softmax");
    let pv = Bf16TnMatmulPlan::new_strided_batch(
        &lt,
        GemmShape::new(
            HEAD_DIM,
            VALIDATION_TOKENS * queries_per_kv,
            VALIDATION_TOKENS,
        ),
        KV_HEADS,
        HEAD_DIM * VALIDATION_TOKENS,
        queries_per_kv * VALIDATION_TOKENS * VALIDATION_TOKENS,
        queries_per_kv * VALIDATION_TOKENS * HEAD_DIM,
        WORKSPACE_LIMIT,
    )
    .expect("validation PV plan");
    let mut packed_output =
        DeviceBuffer::zeroed(VALIDATION_TOKENS * q_width).expect("packed output");
    pv.run_on_stream(
        &lt,
        &packed_value,
        &probabilities,
        packed_output.output(),
        &stream,
    )
    .expect("validation PV");
    let mut output = DeviceBuffer::zeroed(VALIDATION_TOKENS * q_width).expect("output");
    unpack_heads_f32_into_on_stream(
        &packed_output,
        output.output(),
        VALIDATION_TOKENS,
        Q_HEADS,
        HEAD_DIM,
        &stream,
    )
    .expect("validation unpack");
    let reference = reference.copy_to_host(&stream).expect("reference download");
    let output = output.copy_to_host(&stream).expect("output download");
    let (max_index, reference_value, output_value, max_error) = reference
        .iter()
        .zip(output.iter())
        .enumerate()
        .map(|(index, (reference, output))| {
            (index, *reference, *output, (reference - output).abs())
        })
        .max_by(|left, right| left.3.total_cmp(&right.3))
        .expect("attention output");
    assert!(
        max_error < 0.20,
        "attention max error {max_error} at {max_index}: reference={reference_value} output={output_value}"
    );
}

fn prefill_attention_sample(
    context: &mut Qwen36PrefillAttentionBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    black_box(context.output.cuda_address());
    BenchSampleResult::operations((chunk_size * TOKENS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / chunk_size as f64,
        "ms/chunk",
    ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen36-prefill-attention".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Qwen36PrefillAttentionBench>(
                "Qwen3.6 tensor-core prefill attention 2K",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(1, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("pipeline", prefill_attention_sample);
                },
            );
        },
    );
}
