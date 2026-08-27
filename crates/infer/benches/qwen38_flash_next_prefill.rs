use infer::qwen38_flash_next::benchmark::{
    Qwen38HyperPrefillMicrobench, Qwen38QsaPrefillMicrobench,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::CudaEvent;
use std::path::PathBuf;
use std::time::Duration;

struct QsaPrefillBench {
    qsa: Qwen38QsaPrefillMicrobench,
    start: CudaEvent,
    stop: CudaEvent,
}

struct HyperPrefillBench {
    hyper: Qwen38HyperPrefillMicrobench,
    start: CudaEvent,
    stop: CudaEvent,
}

impl BenchContext for HyperPrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut hyper = Qwen38HyperPrefillMicrobench::open(model_dir(), bench_tokens())
            .expect("load Qwen3.8 hyperconnection");
        let quality = hyper
            .validate()
            .expect("validate tensor-core Qwen3.8 hyperconnection");
        assert!(
            quality.max_abs_error <= 0.01
                && quality.cosine >= 0.999
                && quality.relative_rmse <= 0.01,
            "tensor-core Qwen3.8 hyperconnection quality: {quality:?}"
        );
        Self {
            hyper,
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl BenchContext for QsaPrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut qsa = Qwen38QsaPrefillMicrobench::open(model_dir(), bench_tokens())
            .expect("load Qwen3.8 QSA layer");
        let quality = qsa.validate().expect("validate batched QSA prefill");
        assert!(
            quality.max_abs_error <= 0.005
                && quality.cosine >= 0.999
                && quality.relative_rmse <= 0.01,
            "batched QSA quality: {quality:?}"
        );
        Self {
            qsa,
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn serial_sample(
    context: &mut QsaPrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(context.qsa.stream())
        .expect("record serial start");
    for _ in 0..chunk_size {
        context.qsa.enqueue_serial().expect("serial QSA prefill");
    }
    context
        .stop
        .record_on_stream(context.qsa.stream())
        .expect("record serial stop");
    context.stop.synchronize().expect("serial synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("serial elapsed") as f64;
    black_box(context.qsa.serial_output_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn batched_sample(
    context: &mut QsaPrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(context.qsa.stream())
        .expect("record batched start");
    for _ in 0..chunk_size {
        context.qsa.enqueue_batched().expect("batched QSA prefill");
    }
    context
        .stop
        .record_on_stream(context.qsa.stream())
        .expect("record batched stop");
    context.stop.synchronize().expect("batched synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("batched elapsed") as f64;
    black_box(context.qsa.serial_output_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn serial_hyper_sample(
    context: &mut HyperPrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(context.hyper.stream())
        .expect("record serial hyperconnection start");
    for _ in 0..chunk_size {
        context
            .hyper
            .enqueue_serial()
            .expect("serial Qwen3.8 hyperconnection");
    }
    context
        .stop
        .record_on_stream(context.hyper.stream())
        .expect("record serial hyperconnection stop");
    context
        .stop
        .synchronize()
        .expect("serial hyperconnection synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("serial hyperconnection elapsed") as f64;
    black_box(context.hyper.serial_output_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn tensor_hyper_sample(
    context: &mut HyperPrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(context.hyper.stream())
        .expect("record tensor hyperconnection start");
    for _ in 0..chunk_size {
        context
            .hyper
            .enqueue_tensor()
            .expect("tensor Qwen3.8 hyperconnection");
    }
    context
        .stop
        .record_on_stream(context.hyper.stream())
        .expect("record tensor hyperconnection stop");
    context
        .stop
        .synchronize()
        .expect("tensor hyperconnection synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("tensor hyperconnection elapsed") as f64;
    black_box(context.hyper.tensor_output_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn model_dir() -> PathBuf {
    std::env::var_os("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set EIDER_QWEN38_FLASH_NEXT_MODEL_DIR to the released checkpoint")
}

fn bench_tokens() -> usize {
    std::env::var("QWEN38_PREFILL_BENCH_TOKENS")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("QWEN38_PREFILL_BENCH_TOKENS is an integer")
        })
        .unwrap_or(64)
}

fn main() {
    let tokens = bench_tokens();
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-flash-next-prefill".to_string()),
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
            runner.group::<QsaPrefillBench>("Qwen3.8 Flash Next QSA prefill", |group| {
                let group = group
                    .throughput(Throughput::per_operation(tokens as u64, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("serial_rows", serial_sample);
                group.bench_sample("batched_projections", batched_sample);
            });
            runner.group::<HyperPrefillBench>(
                "Qwen3.8 Flash Next hyperconnection prefill",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(tokens as u64, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("f32_activation_warps", serial_hyper_sample);
                    group.bench_sample("bf16_tensor_cores", tensor_hyper_sample);
                },
            );
        },
    );
}
