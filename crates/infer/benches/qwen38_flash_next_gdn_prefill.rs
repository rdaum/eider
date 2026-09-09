//! Measures one Qwen3.8 Flash Next GDN layer at prompt-prefill shape.

use eider_cuda::CudaEvent;
use eider_inference::qwen38_flash_next::benchmark::{
    Qwen38GdnPrefillLayerProfile, Qwen38GdnPrefillMicrobench,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

struct GdnPrefillBench {
    gdn: Qwen38GdnPrefillMicrobench,
    profile: Qwen38GdnPrefillLayerProfile,
    start: CudaEvent,
    stop: CudaEvent,
}

impl BenchContext for GdnPrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut gdn = Qwen38GdnPrefillMicrobench::open(
            model_dir(),
            artifact_dir(),
            layer_index(),
            bench_tokens(),
        )
        .expect("load Qwen3.8 GDN layer");
        let quality = gdn.validate().expect("validate vectorized GDN prefill");
        assert!(
            quality.cosine >= 0.98 && quality.relative_rmse <= 0.20,
            "vectorized GDN quality: {quality:?}"
        );
        gdn.validate_layer().expect("validate complete GDN layer");
        let profile = gdn.profile().expect("profile complete GDN layer");
        Self {
            gdn,
            profile,
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample(
    context: &mut GdnPrefillBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context.gdn.reset_input().expect("reset GDN input");
    context
        .start
        .record_on_stream(context.gdn.stream())
        .expect("record GDN start");
    for _ in 0..chunk_size {
        context.gdn.enqueue_layer().expect("run complete GDN layer");
    }
    context
        .stop
        .record_on_stream(context.gdn.stream())
        .expect("record GDN stop");
    context.stop.synchronize().expect("GDN synchronize");
    black_box(context.gdn.output_address());
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("GDN elapsed") as f64;
    let mut sample = BenchSampleResult::operations((bench_tokens() * chunk_size) as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms/chunk")
                .with_display_name("complete layer"),
        );
    for (name, value) in [
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
        .unwrap_or_else(|| std::env::temp_dir().join("eider-qwen38-gdn-prefill-bench"))
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

fn bench_tokens() -> usize {
    std::env::var("QWEN38_PREFILL_BENCH_TOKENS")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("QWEN38_PREFILL_BENCH_TOKENS is an integer")
        })
        .unwrap_or(512)
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-flash-next-gdn-prefill".to_string()),
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
            runner.group::<GdnPrefillBench>("Qwen3.8 Flash Next GDN prefill", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("vectorized", sample);
            });
        },
    );
}
