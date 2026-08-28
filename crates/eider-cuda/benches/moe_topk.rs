use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, moe_topk_f32_batch_into_on_stream,
    moe_topk_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, black_box, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const PREFILL_ROWS: usize = 2_048;
const QWEN38_EXPERTS: usize = 512;
const QWEN38_TOP_K: usize = 10;
const QWEN38_ROWS: usize = 2;

struct MoeTopkBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

struct MoeTopkBatchBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

struct Qwen38MoeTopkBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

impl BenchContext for Qwen38MoeTopkBench {
    fn prepare(_num_chunks: usize) -> Self {
        let logits = (0..QWEN38_ROWS * QWEN38_EXPERTS)
            .map(|index| {
                let row = index / QWEN38_EXPERTS;
                let expert = index % QWEN38_EXPERTS;
                (((expert * 73 + row * 37) % 1009) as f32 - 504.0) * 0.03125
            })
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut context = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            logits: DeviceBuffer::from_host(&logits).expect("logits"),
            indices: DeviceBuffer::zeroed(QWEN38_ROWS * QWEN38_TOP_K).expect("indices"),
            weights: DeviceBuffer::zeroed(QWEN38_ROWS * QWEN38_TOP_K).expect("weights"),
        };
        context.enqueue();
        let actual_indices = context
            .indices
            .copy_to_host(&context.stream)
            .expect("indices");
        let actual_weights = context
            .weights
            .copy_to_host(&context.stream)
            .expect("weights");
        for row in 0..QWEN38_ROWS {
            let row_logits = &logits[row * QWEN38_EXPERTS..(row + 1) * QWEN38_EXPERTS];
            let mut expected_indices = (0..QWEN38_EXPERTS).collect::<Vec<_>>();
            expected_indices.sort_unstable_by(|&left, &right| {
                row_logits[right]
                    .total_cmp(&row_logits[left])
                    .then_with(|| left.cmp(&right))
            });
            let expected_indices = &expected_indices[..QWEN38_TOP_K];
            let selected_max = row_logits[expected_indices[0]];
            let selected_sum = expected_indices
                .iter()
                .map(|&expert| (row_logits[expert] - selected_max).exp())
                .sum::<f32>();
            for (slot, &expert) in expected_indices.iter().enumerate() {
                assert_eq!(actual_indices[row * QWEN38_TOP_K + slot], expert as u32);
                let expected = (row_logits[expert] - selected_max).exp() / selected_sum;
                let actual = actual_weights[row * QWEN38_TOP_K + slot];
                assert!((actual - expected).abs() < 1.0e-6);
            }
        }
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(5_000)
    }
}

impl Qwen38MoeTopkBench {
    fn enqueue(&mut self) {
        moe_topk_f32_batch_into_on_stream(
            &self.logits,
            self.indices.output(),
            self.weights.output(),
            QWEN38_ROWS,
            QWEN38_EXPERTS,
            QWEN38_TOP_K,
            true,
            &self.stream,
        )
        .expect("Qwen3.8 top-k");
    }
}

impl BenchContext for MoeTopkBatchBench {
    fn prepare(_num_chunks: usize) -> Self {
        let logits = (0..PREFILL_ROWS * EXPERTS)
            .map(|index| {
                let row = index / EXPERTS;
                let expert = index % EXPERTS;
                (((expert * 73 + row * 37) % 509) as f32 - 254.0) * 0.03125
            })
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut context = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            logits: DeviceBuffer::from_host(&logits).expect("logits"),
            indices: DeviceBuffer::zeroed(PREFILL_ROWS * TOP_K).expect("indices"),
            weights: DeviceBuffer::zeroed(PREFILL_ROWS * TOP_K).expect("weights"),
        };
        context.validate(&logits);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl MoeTopkBatchBench {
    fn enqueue(&mut self) {
        moe_topk_f32_batch_into_on_stream(
            &self.logits,
            self.indices.output(),
            self.weights.output(),
            PREFILL_ROWS,
            EXPERTS,
            TOP_K,
            true,
            &self.stream,
        )
        .expect("batch top-k");
    }

    fn validate(&mut self, logits: &[f32]) {
        self.enqueue();
        let actual_indices = self
            .indices
            .copy_to_host(&self.stream)
            .expect("batch indices");
        let actual_weights = self
            .weights
            .copy_to_host(&self.stream)
            .expect("batch weights");
        for row in [0, PREFILL_ROWS / 2, PREFILL_ROWS - 1] {
            let row_logits = DeviceBuffer::from_host(&logits[row * EXPERTS..(row + 1) * EXPERTS])
                .expect("row logits");
            let mut row_indices = DeviceBuffer::zeroed(TOP_K).expect("row indices");
            let mut row_weights = DeviceBuffer::zeroed(TOP_K).expect("row weights");
            moe_topk_f32_into_on_stream(
                &row_logits,
                row_indices.output(),
                row_weights.output(),
                TOP_K,
                true,
                &self.stream,
            )
            .expect("row top-k");
            assert_eq!(
                &actual_indices[row * TOP_K..(row + 1) * TOP_K],
                &*row_indices.copy_to_host(&self.stream).expect("row indices")
            );
            assert_eq!(
                &actual_weights[row * TOP_K..(row + 1) * TOP_K],
                &*row_weights.copy_to_host(&self.stream).expect("row weights")
            );
        }
    }
}

impl BenchContext for MoeTopkBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            logits: DeviceBuffer::from_host(
                &(0..EXPERTS)
                    .map(|idx| (((idx * 73) % 257) as f32 - 128.0) * 0.03125)
                    .collect::<Vec<_>>(),
            )
            .expect("logits"),
            indices: DeviceBuffer::zeroed(TOP_K).expect("indices"),
            weights: DeviceBuffer::zeroed(TOP_K).expect("weights"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(100)
    }
}

fn topk_sample(ctx: &mut MoeTopkBench, chunk_size: usize, _chunk_num: usize) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        moe_topk_f32_into_on_stream(
            &ctx.logits,
            ctx.indices.output(),
            ctx.weights.output(),
            TOP_K,
            true,
            &ctx.stream,
        )
        .expect("top-k");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.indices.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn topk_batch_sample(
    ctx: &mut MoeTopkBatchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.enqueue();
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.indices.as_const_ptr());
    BenchSampleResult::operations((chunk_size * PREFILL_ROWS) as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms/chunk")
            .with_display_name("CUDA event"),
    )
}

fn qwen38_topk_sample(
    ctx: &mut Qwen38MoeTopkBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.enqueue();
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.indices.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-moe-topk".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<MoeTopkBench>("MoE top-k", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample("qwen36_experts256_top8_normalized", topk_sample);
        });
        runner.group::<MoeTopkBatchBench>("MoE top-k batch", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample(
                "qwen36_rows2048_experts256_top8_normalized",
                topk_batch_sample,
            );
        });
        runner.group::<Qwen38MoeTopkBench>("Qwen3.8 MoE top-k", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample("rows2_experts512_top10_normalized", qwen38_topk_sample);
        });
    });
}
