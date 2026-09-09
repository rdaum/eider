//! Measures the complete MTP MoE block from a Qwen3.8 Flash Next checkpoint.

use eider_cuda::CudaEvent;
use eider_inference::qwen38_flash_next::benchmark::Qwen38MtpMoeMicrobench;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

struct MtpMoeBench {
    moe: Qwen38MtpMoeMicrobench,
    start: CudaEvent,
    stop: CudaEvent,
}

impl BenchContext for MtpMoeBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut moe = Qwen38MtpMoeMicrobench::open(model_dir(), artifact_dir())
            .expect("load Qwen3.8 MTP MoE");
        moe.validate().expect("validate Qwen3.8 MTP MoE");
        Self {
            moe,
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
        }
    }
}

fn moe_sample(
    context: &mut MtpMoeBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(context.moe.stream())
        .expect("record MTP MoE start");
    for _ in 0..chunk_size {
        black_box(context.moe.enqueue().expect("run MTP MoE"));
    }
    context
        .stop
        .record_on_stream(context.moe.stream())
        .expect("record MTP MoE stop");
    context.stop.synchronize().expect("MTP MoE synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("MTP MoE elapsed") as f64;
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms/block")
            .with_display_name("complete MTP MoE"),
    )
}

fn model_dir() -> PathBuf {
    std::env::var_os("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set EIDER_QWEN38_FLASH_NEXT_MODEL_DIR")
}

fn artifact_dir() -> PathBuf {
    std::env::var_os("EIDER_QWEN38_FLASH_NEXT_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("eider-qwen38-mtp-moe-bench"))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-flash-next-mtp-moe".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(100),
                benchmark_duration: Duration::from_millis(500),
                min_samples: 4,
                max_samples: 6,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<MtpMoeBench>("Qwen3.8 Flash Next MTP MoE", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "blocks"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("complete block", moe_sample);
            });
        },
    );
}
