use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Result, Sm121W4A16Linear,
    nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream,
    nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

const LAYERS: usize = 40;
const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 512;
const GATE_UP: usize = INTERMEDIATE * 2;

struct SharedProjection {
    gate_up_plan: Sm121W4A16Linear,
    down_plan: Sm121W4A16Linear,
    gate_up_weight: DeviceBuffer<u8>,
    gate_up_scale: DeviceBuffer<u8>,
    gate_up_scale_2: f32,
    down_weight: DeviceBuffer<u8>,
    down_scale: DeviceBuffer<u8>,
    down_scale_2: f32,
}

struct Sm121W4A16SharedExpertBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    hidden_input: DeviceBuffer<f32>,
    intermediate_input: DeviceBuffer<f32>,
    gate_up_output: DeviceBuffer<f32>,
    down_output: DeviceBuffer<f32>,
    projections: Vec<SharedProjection>,
}

impl BenchContext for Sm121W4A16SharedExpertBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare SM121 W4A16 shared-expert benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl Sm121W4A16SharedExpertBench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let mut projections = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let prefix = format!("model.language_model.layers.{layer}.mlp.shared_expert");
            let gate = checkpoint.load_nvfp4_linear(&format!("{prefix}.gate_proj"))?;
            let up = checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?;
            let gate_up = ModelOptNvfp4Linear::concat_out_features(
                format!("{prefix}.gate_up_proj"),
                &gate,
                &up,
            )?;
            let down = checkpoint.load_nvfp4_linear(&format!("{prefix}.down_proj"))?;
            projections.push(SharedProjection {
                gate_up_plan: Sm121W4A16Linear::new(&gate_up)?,
                down_plan: Sm121W4A16Linear::new(&down)?,
                gate_up_weight: DeviceBuffer::from_host(&gate_up.packed_weight)?,
                gate_up_scale: DeviceBuffer::from_host(&gate_up.weight_scale)?,
                gate_up_scale_2: gate_up.weight_scale_2,
                down_weight: DeviceBuffer::from_host(&down.packed_weight)?,
                down_scale: DeviceBuffer::from_host(&down.weight_scale)?,
                down_scale_2: down.weight_scale_2,
            });
        }
        let stream = CudaStream::new_blocking()?;
        let hidden_input = DeviceBuffer::from_host(&host_input(HIDDEN))?;
        let intermediate_input = DeviceBuffer::from_host(&host_input(INTERMEDIATE))?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            hidden_input,
            intermediate_input,
            gate_up_output: DeviceBuffer::zeroed(GATE_UP)?,
            down_output: DeviceBuffer::zeroed(HIDDEN)?,
            projections,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn validate(&mut self) -> Result<()> {
        let projection = &self.projections[0];
        let mut gate_up_reference = DeviceBuffer::zeroed(GATE_UP)?;
        let mut down_reference = DeviceBuffer::zeroed(HIDDEN)?;
        nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream(
            &self.hidden_input,
            &projection.gate_up_weight,
            &projection.gate_up_scale,
            gate_up_reference.output(),
            GATE_UP,
            HIDDEN,
            projection.gate_up_scale_2,
            &self.stream,
        )?;
        nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream(
            &self.intermediate_input,
            &projection.down_weight,
            &projection.down_scale,
            down_reference.output(),
            HIDDEN,
            INTERMEDIATE,
            projection.down_scale_2,
            &self.stream,
        )?;
        nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
            &self.hidden_input,
            &projection.gate_up_weight,
            &projection.gate_up_scale,
            self.gate_up_output.output(),
            GATE_UP,
            HIDDEN,
            projection.gate_up_scale_2,
            8,
            &self.stream,
        )?;
        nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
            &self.intermediate_input,
            &projection.down_weight,
            &projection.down_scale,
            self.down_output.output(),
            HIDDEN,
            INTERMEDIATE,
            projection.down_scale_2,
            8,
            &self.stream,
        )?;
        let gate_up_reference = gate_up_reference.copy_to_host(&self.stream)?;
        let gate_up_warp_rows = self.gate_up_output.copy_to_host(&self.stream)?;
        let down_reference = down_reference.copy_to_host(&self.stream)?;
        let down_warp_rows = self.down_output.copy_to_host(&self.stream)?;
        check_close(
            "warp-row shared gate/up",
            &gate_up_warp_rows,
            &gate_up_reference,
        )?;
        check_close("warp-row shared down", &down_warp_rows, &down_reference)?;

        projection.gate_up_plan.run_on_stream(
            &self.hidden_input,
            self.gate_up_output.output(),
            &self.stream,
        )?;
        projection.down_plan.run_on_stream(
            &self.intermediate_input,
            self.down_output.output(),
            &self.stream,
        )?;
        let gate_up_actual = self.gate_up_output.copy_to_host(&self.stream)?;
        let down_actual = self.down_output.copy_to_host(&self.stream)?;
        check_close("shared gate/up", &gate_up_actual, &gate_up_reference)?;
        check_close("shared down", &down_actual, &down_reference)
    }
}

fn check_close(label: &'static str, actual: &[f32], expected: &[f32]) -> Result<()> {
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let allowed = 0.01 + 0.01 * expected.abs();
        if error > allowed {
            return Err(eider_cuda::Error::Format {
                label,
                detail: format!(
                    "index={idx} actual={actual} expected={expected} error={error} allowed={allowed}"
                ),
            });
        }
    }
    Ok(())
}

fn host_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx * 13) % 31) as f32 - 15.0) * 0.03125)
        .collect()
}

fn model_dir() -> PathBuf {
    std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.6-35b-a3-nvfp4")
        })
}

fn scalar_sample(
    ctx: &mut Sm121W4A16SharedExpertBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.gate_up_weight,
                &projection.gate_up_scale,
                ctx.gate_up_output.output(),
                GATE_UP,
                HIDDEN,
                projection.gate_up_scale_2,
                &ctx.stream,
            )
            .expect("scalar shared gate/up");
            nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream(
                &ctx.intermediate_input,
                &projection.down_weight,
                &projection.down_scale,
                ctx.down_output.output(),
                HIDDEN,
                INTERMEDIATE,
                projection.down_scale_2,
                &ctx.stream,
            )
            .expect("scalar shared down");
        }
    }
    finish_sample(ctx, chunk_size)
}

fn sm121_w4a16_sample(
    ctx: &mut Sm121W4A16SharedExpertBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            projection
                .gate_up_plan
                .run_on_stream(&ctx.hidden_input, ctx.gate_up_output.output(), &ctx.stream)
                .expect("SM121 W4A16 shared gate/up");
            projection
                .down_plan
                .run_on_stream(
                    &ctx.intermediate_input,
                    ctx.down_output.output(),
                    &ctx.stream,
                )
                .expect("SM121 W4A16 shared down");
        }
    }
    finish_sample(ctx, chunk_size)
}

fn warp_rows_sample<const WARPS: usize>(
    ctx: &mut Sm121W4A16SharedExpertBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.gate_up_weight,
                &projection.gate_up_scale,
                ctx.gate_up_output.output(),
                GATE_UP,
                HIDDEN,
                projection.gate_up_scale_2,
                WARPS,
                &ctx.stream,
            )
            .expect("warp-row shared gate/up");
            nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
                &ctx.intermediate_input,
                &projection.down_weight,
                &projection.down_scale,
                ctx.down_output.output(),
                HIDDEN,
                INTERMEDIATE,
                projection.down_scale_2,
                WARPS,
                &ctx.stream,
            )
            .expect("warp-row shared down");
        }
    }
    finish_sample(ctx, chunk_size)
}

fn finish_sample(ctx: &mut Sm121W4A16SharedExpertBench, chunk_size: usize) -> BenchSampleResult {
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.down_output.cuda_address());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-w4a16-shared-expert".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(500),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Sm121W4A16SharedExpertBench>("NVFP4 shared expert", |group| {
            group.bench_sample("qwen36_40_layers_scalar", scalar_sample);
            group.bench_sample("qwen36_40_layers_warp_rows_4", warp_rows_sample::<4>);
            group.bench_sample("qwen36_40_layers_warp_rows_8", warp_rows_sample::<8>);
            group.bench_sample("qwen36_40_layers_warp_rows_16", warp_rows_sample::<16>);
            group.bench_sample("qwen36_40_layers_warp_rows_32", warp_rows_sample::<32>);
            group.bench_sample("qwen36_40_layers_sm121_w4a16", sm121_w4a16_sample);
        });
    });
}
