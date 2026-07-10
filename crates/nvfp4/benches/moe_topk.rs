use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{CudaEvent, CudaStream, DeviceBuffer, moe_topk_f32_into_on_stream};
use std::time::Duration;

const EXPERTS: usize = 256;
const TOP_K: usize = 8;

struct MoeTopkBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
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

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-moe-topk".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
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
            group.bench_sample("qwen36_experts256_top8_normalized", topk_sample);
        });
    });
}
