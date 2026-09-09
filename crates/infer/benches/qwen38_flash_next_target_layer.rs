//! Measures one complete Qwen3.8 Flash Next target layer at decode shape.

use eider_cuda::CudaEvent;
use eider_inference::qwen38_flash_next::benchmark::{
    Qwen38TargetLayerMicrobench, Qwen38TargetLayerProfile,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

struct TargetLayerBench {
    layer: Qwen38TargetLayerMicrobench,
    profile: Qwen38TargetLayerProfile,
    start: CudaEvent,
    stop: CudaEvent,
}

impl BenchContext for TargetLayerBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut layer =
            Qwen38TargetLayerMicrobench::open(model_dir(), artifact_dir(), layer_index())
                .expect("load Qwen3.8 target layer");
        layer.validate().expect("validate Qwen3.8 target layer");
        let profile = layer.profile().expect("profile Qwen3.8 target layer");
        Self {
            layer,
            profile,
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
        }
    }
}

fn layer_sample(
    context: &mut TargetLayerBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(context.layer.stream())
        .expect("record layer start");
    for _ in 0..chunk_size {
        context.layer.enqueue().expect("run target layer");
    }
    context
        .stop
        .record_on_stream(context.layer.stream())
        .expect("record layer stop");
    context.stop.synchronize().expect("layer synchronize");
    black_box(context.layer.output_address());
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("layer elapsed") as f64;
    let mut sample = BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms/layer")
            .with_display_name("complete layer"),
    );
    for (name, value) in [
        ("repeat_streams_ms", context.profile.repeat_streams_ms),
        ("attention_mix_ms", context.profile.attention_mix_ms),
        ("attention_ms", context.profile.attention_ms),
        ("attention_combine_ms", context.profile.attention_combine_ms),
        ("mlp_mix_ms", context.profile.mlp_mix_ms),
        ("moe_ms", context.profile.moe_ms),
        ("mlp_combine_ms", context.profile.mlp_combine_ms),
    ] {
        sample = sample.push_metric(MetricValue::new(name, value as f64, "ms"));
    }
    sample
}

fn model_dir() -> PathBuf {
    std::env::var_os("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set EIDER_QWEN38_FLASH_NEXT_MODEL_DIR")
}

fn artifact_dir() -> PathBuf {
    std::env::var_os("EIDER_QWEN38_FLASH_NEXT_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("eider-qwen38-target-layer-bench"))
}

fn layer_index() -> usize {
    std::env::var("EIDER_QWEN38_FLASH_NEXT_LAYER")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("EIDER_QWEN38_FLASH_NEXT_LAYER is an integer")
        })
        .unwrap_or(0)
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-flash-next-target-layer".to_string()),
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
            runner.group::<TargetLayerBench>("Qwen3.8 Flash Next target layer", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "layers"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("complete target layer", layer_sample);
            });
        },
    );
}
