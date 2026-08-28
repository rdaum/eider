mod support;

use eider_cuda::{
    Fp4TnMatmul, GemmShape, GpuCounterCollector, InferenceGemm, Result, format, synchronize_device,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, DiagnosticError, DiagnosticResult, MeasurementDomain, MetricValue,
    Throughput, run_benchmark_main,
};
use std::time::Duration;
use support::CudaEventBackend;

const fn bytes_per_gemm(m: usize, n: usize, k: usize) -> u64 {
    ((k * m) / 2
        + (k * n) / 2
        + format::ue4m3_scale_layout_len_const(m, k)
        + format::ue4m3_scale_layout_len_const(n, k)
        + (m * n * 2)) as u64
}

const fn flops_per_gemm(m: usize, n: usize, k: usize) -> u64 {
    (2 * m * n * k) as u64
}

const fn chunk_size_for_shape(shape: usize) -> usize {
    if shape <= 128 {
        3500
    } else if shape <= 256 {
        2500
    } else if shape <= 512 {
        3000
    } else if shape <= 1024 {
        2000
    } else if shape <= 2048 {
        600
    } else if shape <= 4096 {
        160
    } else if shape <= 8192 {
        80
    } else {
        40
    }
}

struct Fp4Bench<const M: usize, const N: usize, const K: usize> {
    matmul: Fp4TnMatmul,
}

const GPU_COUNTER_METRICS: &[&str] = &[
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
];

impl<const M: usize, const N: usize, const K: usize> BenchContext for Fp4Bench<M, N, K> {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare FP4 cuBLASLt benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(chunk_size_for_shape(M.max(N).max(K)))
    }
}

impl<const M: usize, const N: usize, const K: usize> Fp4Bench<M, N, K> {
    fn new() -> Result<Self> {
        Ok(Self {
            matmul: Fp4TnMatmul::ones(GemmShape::new(M, N, K), 4 * 1024 * 1024)?,
        })
    }

    fn run_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.matmul
                .run_on_default_stream()
                .expect("FP4 benchmark matmul");
        }
        synchronize_device().expect("FP4 benchmark synchronize");
    }
}

fn fp4_gemm<const M: usize, const N: usize, const K: usize>(
    ctx: &mut Fp4Bench<M, N, K>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_chunk(chunk_size);
    let metadata = ctx.matmul.metadata();

    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("workspace_bytes", metadata.workspace_bytes as f64, "bytes")
                .with_display_name("Workspace"),
        )
        .push_metric(
            MetricValue::new("algo_word0", metadata.algorithm_data[0] as f64, "")
                .with_display_name("Algorithm word 0"),
        )
}

fn fp4_gemm_diagnostic<const M: usize, const N: usize, const K: usize>(
    ctx: &mut Fp4Bench<M, N, K>,
    chunk_size: usize,
    _chunk_num: usize,
) -> std::result::Result<DiagnosticResult, DiagnosticError> {
    let mut collector = match GpuCounterCollector::new(GPU_COUNTER_METRICS, "fp4_cublaslt") {
        Ok(collector) => collector,
        Err(error) => return Ok(gpu_counter_error_result(&error.to_string())),
    };
    let mut passes = 0;
    loop {
        passes += 1;
        if let Err(error) = collector.begin() {
            return Ok(gpu_counter_error_result(&error.to_string()));
        }
        ctx.run_chunk(chunk_size);
        let done = match collector.end() {
            Ok(done) => done,
            Err(error) => return Ok(gpu_counter_error_result(&error.to_string())),
        };
        if done || passes >= 8 {
            break;
        }
    }

    let mut result = DiagnosticResult::new("gpu counters").push_metric(
        MetricValue::integer("gpu_counter_replay_passes", passes, "passes")
            .with_display_name("Replay passes"),
    );
    let metrics = match collector.decode() {
        Ok(metrics) => metrics,
        Err(error) => return Ok(gpu_counter_error_result(&error.to_string())),
    };
    for metric in metrics {
        result = result.push_metric(gpu_counter_metric(&metric.name, metric.value));
    }
    Ok(result)
}

fn gpu_counter_error_result(error: &str) -> DiagnosticResult {
    let lower = error.to_ascii_lowercase();
    let metric = if error.contains("ERR_NVGPUCTRPERM")
        || error.contains("CUPTI_ERROR_INSUFFICIENT_PRIVILEGES")
        || lower.contains("permission")
        || lower.contains("privilege")
    {
        MetricValue::integer("gpu_counter_permission_error", 1, "errors")
            .with_display_name("Counter permission error")
    } else {
        MetricValue::integer(gpu_counter_error_name(&lower), 1, "errors")
            .with_display_name(gpu_counter_error_label(&lower))
    };
    DiagnosticResult::new("gpu counters").push_metric(metric)
}

fn gpu_counter_error_name(error: &str) -> &'static str {
    if error.contains("configaddmetrics") || error.contains("metric") {
        "gpu_counter_metric_error"
    } else if error.contains("setconfig") {
        "gpu_counter_config_error"
    } else if error.contains("start") || error.contains("pushrange") {
        "gpu_counter_start_error"
    } else if error.contains("stop") || error.contains("poprange") {
        "gpu_counter_stop_error"
    } else if error.contains("decode") || error.contains("evaluate") {
        "gpu_counter_decode_error"
    } else {
        "gpu_counter_collection_error"
    }
}

fn gpu_counter_error_label(error: &str) -> &'static str {
    if error.contains("configaddmetrics") || error.contains("metric") {
        "Counter metric error"
    } else if error.contains("setconfig") {
        "Counter config error"
    } else if error.contains("start") || error.contains("pushrange") {
        "Counter start error"
    } else if error.contains("stop") || error.contains("poprange") {
        "Counter stop error"
    } else if error.contains("decode") || error.contains("evaluate") {
        "Counter decode error"
    } else {
        "Counter collection error"
    }
}

fn gpu_counter_metric(name: &str, value: f64) -> MetricValue {
    match name {
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" => {
            MetricValue::new("memory_pct_of_peak", value, "%").with_display_name("Memory peak")
        }
        "lts__throughput.avg.pct_of_peak_sustained_elapsed" => {
            MetricValue::new("l2_pct_of_peak", value, "%").with_display_name("L2 peak")
        }
        "sm__throughput.avg.pct_of_peak_sustained_elapsed" => {
            MetricValue::new("sm_pct_of_peak", value, "%").with_display_name("SM peak")
        }
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active" => {
            MetricValue::new("tensor_active_pct", value, "%").with_display_name("Tensor active")
        }
        _ => MetricValue::new("gpu_counter", value, "").with_display_name("GPU counter"),
    }
}

fn register_shape<const M: usize, const N: usize, const K: usize>(
    runner: &micromeasure::BenchmarkRunner,
    group_name: &'static str,
    bench_name: &'static str,
) {
    runner.group::<Fp4Bench<M, N, K>>(group_name, |g| {
        g.throughput(Throughput::bytes(bytes_per_gemm(M, N, K)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(|| {
                Box::new(CudaEventBackend::new(
                    bytes_per_gemm(M, N, K),
                    flops_per_gemm(M, N, K),
                ))
            })
            .diagnostic_samples(2)
            .diagnostic_pass(fp4_gemm_diagnostic::<M, N, K>)
            .bench_sample(bench_name, fp4_gemm::<M, N, K>);
    });
}

fn register_inference_shape<const M: usize, const N: usize, const K: usize>(
    runner: &micromeasure::BenchmarkRunner,
    group_name: &'static str,
    bench_name: &'static str,
    inference_shape: InferenceGemm,
) {
    assert_eq!(inference_shape.gemm_shape(), GemmShape::new(M, N, K));
    register_shape::<M, N, K>(runner, group_name, bench_name);
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-gpu".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(250),
            benchmark_duration: Duration::from_secs(1),
            min_samples: 10,
            max_samples: 20,
        },
        ..BenchmarkMainOptions::default()
    };

    run_benchmark_main(options, |runner| {
        register_shape::<128, 128, 128>(runner, "cuBLASLt FP4 square", "fp4_128x128x128");
        register_shape::<256, 256, 256>(runner, "cuBLASLt FP4 square", "fp4_256x256x256");
        register_shape::<512, 512, 512>(runner, "cuBLASLt FP4 square", "fp4_512x512x512");
        register_shape::<1024, 1024, 1024>(runner, "cuBLASLt FP4 square", "fp4_1024x1024x1024");
        register_shape::<2048, 2048, 2048>(runner, "cuBLASLt FP4 square", "fp4_2048x2048x2048");

        register_inference_shape::<4096, 1, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B decode",
            "q_proj_decode_m4096_n1_k2560",
            InferenceGemm::qwen3_4b_q_projection(1),
        );
        register_inference_shape::<1024, 1, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B decode",
            "kv_proj_decode_m1024_n1_k2560",
            InferenceGemm::qwen3_4b_kv_projection(1),
        );
        register_inference_shape::<2560, 1, 4096>(
            runner,
            "cuBLASLt FP4 Qwen3-4B decode",
            "o_proj_decode_m2560_n1_k4096",
            InferenceGemm::qwen3_4b_o_projection(1),
        );
        register_inference_shape::<19456, 1, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B decode",
            "ffn_gate_up_decode_m19456_n1_k2560",
            InferenceGemm::qwen3_4b_ffn_gate_up(1),
        );
        register_inference_shape::<2560, 1, 9728>(
            runner,
            "cuBLASLt FP4 Qwen3-4B decode",
            "ffn_down_decode_m2560_n1_k9728",
            InferenceGemm::qwen3_4b_ffn_down(1),
        );

        register_shape::<6144, 1, 4096>(
            runner,
            "cuBLASLt FP4 Qwen3-8B decode",
            "qkv_decode_m6144_n1_k4096",
        );
        register_shape::<24576, 1, 4096>(
            runner,
            "cuBLASLt FP4 Qwen3-8B decode",
            "ffn_gate_up_decode_m24576_n1_k4096",
        );
        register_shape::<4096, 1, 12288>(
            runner,
            "cuBLASLt FP4 Qwen3-8B decode",
            "ffn_down_decode_m4096_n1_k12288",
        );

        register_inference_shape::<4096, 8, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B small batch",
            "q_proj_batch8_m4096_n8_k2560",
            InferenceGemm::qwen3_4b_q_projection(8),
        );
        register_inference_shape::<1024, 8, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B small batch",
            "kv_proj_batch8_m1024_n8_k2560",
            InferenceGemm::qwen3_4b_kv_projection(8),
        );
        register_inference_shape::<19456, 8, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B small batch",
            "ffn_gate_up_batch8_m19456_n8_k2560",
            InferenceGemm::qwen3_4b_ffn_gate_up(8),
        );
        register_inference_shape::<2560, 8, 9728>(
            runner,
            "cuBLASLt FP4 Qwen3-4B small batch",
            "ffn_down_batch8_m2560_n8_k9728",
            InferenceGemm::qwen3_4b_ffn_down(8),
        );

        register_inference_shape::<4096, 128, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B prefill",
            "q_proj_prefill128_m4096_n128_k2560",
            InferenceGemm::qwen3_4b_q_projection(128),
        );
        register_inference_shape::<19456, 128, 2560>(
            runner,
            "cuBLASLt FP4 Qwen3-4B prefill",
            "ffn_gate_up_prefill128_m19456_n128_k2560",
            InferenceGemm::qwen3_4b_ffn_gate_up(128),
        );
        register_shape::<4096, 7, 4096>(
            runner,
            "cuBLASLt FP4 tile pressure",
            "legacy_misaligned_batch_m4096_n7_k4096",
        );
    });
}
