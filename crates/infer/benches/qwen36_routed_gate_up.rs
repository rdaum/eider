use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, GpuCounterCollector, Result,
    round_f32_to_bf16_into_on_stream,
};
use eider_inference::qwen3::qwen36::{Qwen36LayerBlock, Qwen36LayerBlockWorkspace, Qwen36Model};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, DiagnosticError, DiagnosticResult, MeasurementBackend, MeasurementDomain,
    MetricValue, Throughput, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const GPU_COUNTER_METRICS: &[&str] = &[
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
];

struct Qwen36RoutedGateUpBench {
    block: Qwen36LayerBlock,
    workspace: Qwen36LayerBlockWorkspace,
    ffn_norm: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    rounded_reference: DeviceBuffer<f32>,
    stream: CudaStream,
    route_indices: Vec<usize>,
    route_weights: Vec<f32>,
    hidden: usize,
    experts: usize,
    top_k: usize,
    expert_intermediate: usize,
}

struct CudaEventBackend {
    start: CudaEvent,
    stop: CudaEvent,
    host_start: Option<Instant>,
    host_elapsed: Duration,
    device_ms: f64,
}

impl CudaEventBackend {
    fn new() -> Self {
        Self {
            start: CudaEvent::new().expect("create CUDA start event"),
            stop: CudaEvent::new().expect("create CUDA stop event"),
            host_start: None,
            host_elapsed: Duration::ZERO,
            device_ms: 0.0,
        }
    }
}

impl MeasurementBackend for CudaEventBackend {
    fn begin(&mut self) {
        self.host_start = Some(Instant::now());
        self.start
            .record_default_stream()
            .expect("record CUDA start event");
    }

    fn end(&mut self) {
        self.stop
            .record_default_stream()
            .expect("record CUDA stop event");
        self.stop
            .synchronize()
            .expect("synchronize CUDA stop event");
        self.device_ms = self
            .start
            .elapsed_ms_until(&self.stop)
            .expect("compute CUDA event elapsed time") as f64;
        self.host_elapsed = self
            .host_start
            .take()
            .expect("CUDA event backend begin before end")
            .elapsed();
    }

    fn collect(
        &mut self,
        _host_elapsed: Duration,
        ops: u64,
        _chunk_index: usize,
        results: &mut micromeasure::bench::Results,
        metrics: &mut Vec<MetricValue>,
    ) {
        let device_duration = Duration::from_secs_f64(self.device_ms / 1_000.0);
        let host_overhead = self.host_elapsed.saturating_sub(device_duration);
        results.duration = device_duration;
        results.iterations = ops;
        results.chunks_executed = 1;
        metrics.push(
            MetricValue::duration_ms("cuda_event_ms", device_duration)
                .with_display_name("CUDA event"),
        );
        metrics.push(
            MetricValue::duration_ms("host_overhead_ms", host_overhead)
                .with_display_name("Host overhead"),
        );
    }

    fn measurement_label(&self) -> &'static str {
        "timing + CUDA events"
    }

    fn emits_cpu_diagnostics(&self) -> bool {
        false
    }
}

impl BenchContext for Qwen36RoutedGateUpBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Qwen3.6 routed gate/up benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

impl Qwen36RoutedGateUpBench {
    fn new() -> Result<Self> {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36Model::open(&model_dir)?;
        let manifest = model.manifest();
        let block = Qwen36LayerBlock::load(&model, 0)?;
        let mut workspace = block.workspace(&model, 8)?;
        let stream = CudaStream::new_blocking()?;
        let ffn_norm_host = (0..manifest.hidden)
            .map(|idx| ((idx % 251) as f32 - 125.0) / 125.0)
            .collect::<Vec<_>>();
        let ffn_norm = DeviceBuffer::from_host(&ffn_norm_host)?;
        let residual_host = (0..manifest.hidden)
            .map(|idx| ((idx % 199) as f32 - 99.0) / 99.0)
            .collect::<Vec<_>>();
        let residual = DeviceBuffer::from_host(&residual_host)?;
        block
            .moe
            .prepare_routed_gate_up(&mut workspace.moe, manifest, &ffn_norm, &stream)?;
        block
            .moe
            .run_routed_gate_up_only(&mut workspace.moe, &ffn_norm, &stream)?;
        block
            .moe
            .prepare_grouped_down(&mut workspace.moe, &stream)?;
        block
            .moe
            .run_router_linear_only(&mut workspace.moe, &ffn_norm, &stream)?;
        block
            .moe
            .run_grouped_down_gather_only(&mut workspace.moe, &stream)?;
        block
            .moe
            .run_grouped_down_gemv_only(&mut workspace.moe, &stream)?;
        block
            .moe
            .run_grouped_down_accum_only(&mut workspace.moe, &stream)?;
        block
            .moe
            .run_shared_gate_up_only(&mut workspace.moe, &ffn_norm, &stream)?;
        block
            .moe
            .run_shared_silu_only(&mut workspace.moe, &stream)?;
        block
            .moe
            .run_shared_down_only(&mut workspace.moe, &stream)?;
        block
            .moe
            .run_shared_gate_only(&mut workspace.moe, &ffn_norm, &stream)?;
        block
            .moe
            .run_ffn_combine_only(&mut workspace.moe, &residual, &stream)?;
        let mut rounded_reference = DeviceBuffer::zeroed(manifest.hidden)?;
        round_f32_to_bf16_into_on_stream(
            &workspace.moe.ffn_residual,
            rounded_reference.output(),
            &stream,
        )?;
        block
            .moe
            .run_shared_gate_linear_only(&mut workspace.moe, &ffn_norm, &stream)?;
        block
            .moe
            .run_ffn_finalize_routed_only(&mut workspace.moe, &residual, &stream)?;
        let reference = rounded_reference.copy_to_host(&stream)?;
        let candidate = workspace.moe.ffn_residual.copy_to_host(&stream)?;
        assert_eq!(
            candidate.into_vec(),
            reference.into_vec(),
            "fused routed FFN finalization changed BF16-rounded output"
        );
        let route_indices = workspace
            .moe
            .route
            .indices
            .copy_to_host(&stream)?
            .into_vec()
            .into_iter()
            .map(|idx| idx as usize)
            .collect::<Vec<_>>();
        let route_weights = workspace
            .moe
            .route
            .weights
            .copy_to_host(&stream)?
            .into_vec();
        stream.synchronize()?;
        let (experts, top_k, expert_intermediate) = block.moe.shape();
        Ok(Self {
            block,
            workspace,
            ffn_norm,
            residual,
            rounded_reference,
            stream,
            route_indices,
            route_weights,
            hidden: manifest.hidden,
            experts,
            top_k,
            expert_intermediate,
        })
    }

    fn run_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_routed_gate_up_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run selected routed gate/up");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_router_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_router_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run routed router/top-k");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_router_linear_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_router_linear_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run router projection");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_topk_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_topk_only(&mut self.workspace.moe, &self.stream)
                .expect("run router top-k");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_down_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_grouped_down_only(&mut self.workspace.moe, &self.stream)
                .expect("run grouped routed down");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_w4a16_gate_up_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_w4a16_gate_up_slots_only(
                    &mut self.workspace.moe,
                    &self.ffn_norm,
                    &self.route_indices,
                    &self.stream,
                )
                .expect("run W4A16 routed gate/up");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_w4a16_down_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_w4a16_down_slots_only(
                    &mut self.workspace.moe,
                    &self.route_indices,
                    &self.stream,
                )
                .expect("run W4A16 routed down");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_w4a16_moe_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_w4a16_moe_slots_only(
                    &mut self.workspace.moe,
                    &self.ffn_norm,
                    &self.route_indices,
                    &self.route_weights,
                    &self.stream,
                )
                .expect("run W4A16 routed MoE");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_silu_quantize_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .prepare_grouped_down(&mut self.workspace.moe, &self.stream)
                .expect("run routed SiLU quantize");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_down_gather_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_grouped_down_gather_only(&mut self.workspace.moe, &self.stream)
                .expect("run routed down gather");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_down_gemv_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_grouped_down_gemv_only(&mut self.workspace.moe, &self.stream)
                .expect("run routed down GEMV");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_down_accum_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_grouped_down_accum_only(&mut self.workspace.moe, &self.stream)
                .expect("run routed down accumulate");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_shared_gate_up_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_shared_gate_up_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run shared gate/up");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_shared_silu_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_shared_silu_only(&mut self.workspace.moe, &self.stream)
                .expect("run shared SiLU");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_shared_down_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_shared_down_only(&mut self.workspace.moe, &self.stream)
                .expect("run shared down");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_shared_gate_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_shared_gate_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run shared gate");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_ffn_combine_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_ffn_combine_only(&mut self.workspace.moe, &self.residual, &self.stream)
                .expect("run FFN combine");
        }
        self.stream
            .synchronize()
            .expect("synchronize benchmark stream");
    }

    fn run_ffn_finalize_reference_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_grouped_down_accum_only(&mut self.workspace.moe, &self.stream)
                .expect("run routed accumulation");
            self.block
                .moe
                .run_shared_gate_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run shared gate");
            self.block
                .moe
                .run_ffn_combine_only(&mut self.workspace.moe, &self.residual, &self.stream)
                .expect("run FFN combine");
            round_f32_to_bf16_into_on_stream(
                &self.workspace.moe.ffn_residual,
                self.rounded_reference.output(),
                &self.stream,
            )
            .expect("round FFN output to BF16");
        }
        self.stream
            .synchronize()
            .expect("synchronize reference FFN finalization");
    }

    fn run_ffn_finalize_fused_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.block
                .moe
                .run_shared_gate_linear_only(&mut self.workspace.moe, &self.ffn_norm, &self.stream)
                .expect("run shared gate projection");
            self.block
                .moe
                .run_ffn_finalize_routed_only(&mut self.workspace.moe, &self.residual, &self.stream)
                .expect("run fused routed FFN finalization");
        }
        self.stream
            .synchronize()
            .expect("synchronize fused FFN finalization");
    }
}

fn routed_gate_up_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_chunk(chunk_size);
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::integer(
            "hidden",
            ctx.hidden as i64,
            "elements",
        ))
        .push_metric(MetricValue::integer(
            "experts",
            ctx.experts as i64,
            "experts",
        ))
        .push_metric(MetricValue::integer("top_k", ctx.top_k as i64, "experts"))
        .push_metric(MetricValue::integer(
            "expert_intermediate",
            ctx.expert_intermediate as i64,
            "elements",
        ))
}

fn router_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_router_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn router_linear_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_router_linear_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn topk_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_topk_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn routed_down_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_down_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn w4a16_gate_up_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_w4a16_gate_up_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn w4a16_down_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_w4a16_down_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn w4a16_moe_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_w4a16_moe_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn silu_quantize_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_silu_quantize_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn routed_down_gather_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_down_gather_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn routed_down_gemv_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_down_gemv_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn routed_down_accum_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_down_accum_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn shared_gate_up_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_shared_gate_up_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn shared_silu_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_shared_silu_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn shared_down_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_shared_down_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn shared_gate_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_shared_gate_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn ffn_combine_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_ffn_combine_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn ffn_finalize_reference_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_ffn_finalize_reference_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn ffn_finalize_fused_sample(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_ffn_finalize_fused_chunk(chunk_size);
    common_sample_metrics(ctx, chunk_size)
}

fn common_sample_metrics(ctx: &Qwen36RoutedGateUpBench, chunk_size: usize) -> BenchSampleResult {
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::integer(
            "hidden",
            ctx.hidden as i64,
            "elements",
        ))
        .push_metric(MetricValue::integer(
            "experts",
            ctx.experts as i64,
            "experts",
        ))
        .push_metric(MetricValue::integer("top_k", ctx.top_k as i64, "experts"))
        .push_metric(MetricValue::integer(
            "expert_intermediate",
            ctx.expert_intermediate as i64,
            "elements",
        ))
}

fn routed_gate_up_diagnostic(
    ctx: &mut Qwen36RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> std::result::Result<DiagnosticResult, DiagnosticError> {
    let mut collector = match GpuCounterCollector::new(GPU_COUNTER_METRICS, "qwen36_routed_gate_up")
    {
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
        MetricValue::integer("gpu_counter_collection_error", 1, "errors")
            .with_display_name("Counter collection error")
    };
    DiagnosticResult::new("gpu counters").push_metric(metric)
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

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("infer-gpu".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(250),
            benchmark_duration: Duration::from_secs(1),
            min_samples: 5,
            max_samples: 10,
        },
        ..BenchmarkMainOptions::default()
    };

    run_benchmark_main(options, |runner| {
        runner.group::<Qwen36RoutedGateUpBench>("Qwen3.6 routed MoE", |g| {
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("router_topk_experts256_top8_hidden2048", router_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("router_linear_experts256_hidden2048", router_linear_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("router_topk_only_experts256_top8", topk_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .diagnostic_samples(2)
                .diagnostic_pass(routed_gate_up_diagnostic)
                .bench_sample(
                    "routed_gate_up_top8_hidden2048_intermediate512",
                    routed_gate_up_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "routed_silu_quantize_top8_intermediate512",
                    silu_quantize_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "grouped_down_top8_hidden2048_intermediate512",
                    routed_down_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "w4a16_gate_up_top8_hidden2048_intermediate512",
                    w4a16_gate_up_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "w4a16_down_top8_hidden2048_intermediate512",
                    w4a16_down_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "w4a16_moe_top8_hidden2048_intermediate512",
                    w4a16_moe_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("grouped_down_gather_top8", routed_down_gather_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "grouped_down_gemv_top8_hidden2048_intermediate512",
                    routed_down_gemv_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "grouped_down_accum_top8_hidden2048",
                    routed_down_accum_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "shared_gate_up_hidden2048_intermediate512",
                    shared_gate_up_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("shared_silu_intermediate512", shared_silu_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("shared_down_hidden2048_intermediate512", shared_down_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("shared_gate_hidden2048", shared_gate_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("ffn_combine_hidden2048", ffn_combine_sample);
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample(
                    "ffn_finalize_reference_hidden2048",
                    ffn_finalize_reference_sample,
                );
            g.throughput(Throughput::ops())
                .measurement_domain(MeasurementDomain::Gpu)
                .backend(|| Box::new(CudaEventBackend::new()))
                .bench_sample("ffn_finalize_fused_hidden2048", ffn_finalize_fused_sample);
        });
    });
}
