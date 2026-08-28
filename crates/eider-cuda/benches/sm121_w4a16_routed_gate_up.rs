use eider_cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, ModelOptCheckpoint, ModelOptNvfp4Linear,
    Result, Sm121W4A16GateUp, nvfp4_w4a16_matvec_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

const HIDDEN: usize = 2048;
const GATE_UP: usize = 1024;
const TOP_K: usize = 8;
const EXPERTS: usize = 8;
const ROUTES: [u32; TOP_K] = [7, 2, 7, 0, 5, 1, 6, 3];

struct Sm121W4A16RoutedGateUpBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    plan: Sm121W4A16GateUp,
    weights: Vec<ModelOptNvfp4Linear>,
    indices: DeviceBuffer<u32>,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    graph: CudaGraphExec,
}

impl BenchContext for Sm121W4A16RoutedGateUpBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare SM121 W4A16 routed gate/up benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

impl Sm121W4A16RoutedGateUpBench {
    fn new() -> Result<Self> {
        let weights = load_weights()?;
        let plan = Sm121W4A16GateUp::new(&weights)?;
        let stream = CudaStream::new_blocking()?;
        let input = DeviceBuffer::from_host(
            &(0..HIDDEN)
                .map(|idx| (((idx * 7) % 17) as f32 - 8.0) * 0.03125)
                .collect::<Vec<_>>(),
        )?;
        let indices = DeviceBuffer::from_host(&ROUTES)?;
        let mut output = DeviceBuffer::zeroed(TOP_K * GATE_UP)?;
        let graph = stream
            .capture(|stream| plan.run_on_stream(&indices, &input, output.output(), stream))?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            plan,
            weights,
            indices,
            input,
            output,
            graph,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn validate(&mut self) -> Result<()> {
        self.plan.run_on_stream(
            &self.indices,
            &self.input,
            self.output.output(),
            &self.stream,
        )?;
        let actual = self.output.copy_to_host(&self.stream)?;
        let output_bf16 = self
            .plan
            .run_bf16_on_stream(&self.indices, &self.input, &self.stream)?
            .copy_to_host(&self.stream)?;
        let bf16_expanded = output_bf16
            .iter()
            .copied()
            .map(eider_cuda::format::bf16_to_f32)
            .collect::<Vec<_>>();
        if actual.as_ref() != bf16_expanded.as_slice() {
            return Err(eider_cuda::Error::Format {
                label: "SM121 W4A16 BF16 routed gate/up",
                detail: "BF16-only output differs from the expanded F32 path".to_string(),
            });
        }
        self.graph.launch(&self.stream)?;
        let graph_actual = self.output.copy_to_host(&self.stream)?;
        if actual.as_ref() != graph_actual.as_ref() {
            let (index, direct, replay) = actual
                .iter()
                .zip(graph_actual.iter())
                .enumerate()
                .find_map(|(index, (&direct, &replay))| {
                    (direct.to_bits() != replay.to_bits()).then_some((index, direct, replay))
                })
                .expect("different outputs have a mismatch");
            return Err(eider_cuda::Error::Format {
                label: "SM121 W4A16 routed gate/up graph replay",
                detail: format!(
                    "index={index} direct={direct} replay={replay} error={}",
                    (direct - replay).abs()
                ),
            });
        }
        let mut reference = DeviceBuffer::zeroed(GATE_UP)?;
        let mut worst = (0.0f32, 0.0f32, 0.0f32, 0usize, 0usize);
        let mut expected_outputs = Vec::with_capacity(TOP_K);
        for (slot, &expert) in ROUTES.iter().enumerate() {
            let weight = &self.weights[expert as usize];
            let packed_weight = DeviceBuffer::from_host(&weight.packed_weight)?;
            let weight_scale = DeviceBuffer::from_host(&weight.weight_scale)?;
            nvfp4_w4a16_matvec_f32_into_on_stream(
                &self.input,
                &packed_weight,
                &weight_scale,
                reference.output(),
                GATE_UP,
                HIDDEN,
                weight.weight_scale_2,
                &self.stream,
            )?;
            let expected = reference.copy_to_host(&self.stream)?;
            for row in 0..GATE_UP {
                let actual = actual[slot * GATE_UP + row];
                let expected = expected[row];
                let error = (actual - expected).abs();
                if error > worst.0 {
                    worst = (error, actual, expected, slot, row);
                }
            }
            expected_outputs.push(expected.into_vec());
        }
        let allowed = 0.01 + 0.01 * worst.2.abs();
        if worst.0 > allowed {
            let nearest = expected_outputs
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    (left[worst.4] - worst.1)
                        .abs()
                        .total_cmp(&(right[worst.4] - worst.1).abs())
                })
                .map(|(slot, values)| (slot, ROUTES[slot], values[worst.4]));
            return Err(eider_cuda::Error::Format {
                label: "SM121 W4A16 routed gate/up versus W4A16",
                detail: format!(
                    "slot={} expert={} row={} actual={} expected={} error={} allowed={allowed} nearest={nearest:?}",
                    worst.3, ROUTES[worst.3], worst.4, worst.1, worst.2, worst.0
                ),
            });
        }
        Ok(())
    }
}

fn load_weights() -> Result<Vec<ModelOptNvfp4Linear>> {
    let checkpoint = ModelOptCheckpoint::open(model_dir())?;
    let prefix = "model.language_model.layers.0.mlp";
    let mut weights = Vec::with_capacity(EXPERTS);
    for expert in 0..EXPERTS {
        let expert_prefix = format!("{prefix}.experts.{expert}");
        let gate = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.gate_proj"))?;
        let up = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.up_proj"))?;
        weights.push(ModelOptNvfp4Linear::concat_out_features(
            format!("{expert_prefix}.gate_up_proj"),
            &gate,
            &up,
        )?);
    }
    Ok(weights)
}

fn model_dir() -> PathBuf {
    std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.6-35b-a3-nvfp4")
        })
}

fn sm121_w4a16_sample(
    ctx: &mut Sm121W4A16RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start
        .record_on_stream(&ctx.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        ctx.plan
            .run_on_stream(&ctx.indices, &ctx.input, ctx.output.output(), &ctx.stream)
            .expect("SM121 W4A16 routed gate/up");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop event");
    ctx.stop.synchronize().expect("sync stop event");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::new(
            "cuda_event_ms",
            total_ms / chunk_size as f64,
            "ms",
        ))
        .push_metric(MetricValue::integer("slots", TOP_K as i64, "slots"))
}

fn graph_sample(
    ctx: &mut Sm121W4A16RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start
        .record_on_stream(&ctx.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        ctx.graph
            .launch(&ctx.stream)
            .expect("SM121 W4A16 graph replay");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop event");
    ctx.stop.synchronize().expect("sync stop event");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::new(
            "cuda_event_ms",
            total_ms / chunk_size as f64,
            "ms",
        ))
        .push_metric(MetricValue::integer("slots", TOP_K as i64, "slots"))
}

fn sm121_w4a16_bf16_sample(
    ctx: &mut Sm121W4A16RoutedGateUpBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start
        .record_on_stream(&ctx.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        ctx.plan
            .run_bf16_on_stream(&ctx.indices, &ctx.input, &ctx.stream)
            .expect("SM121 W4A16 BF16 routed gate/up");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop event");
    ctx.stop.synchronize().expect("sync stop event");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.plan.output_bf16().as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::new(
            "cuda_event_ms",
            total_ms / chunk_size as f64,
            "ms",
        ))
        .push_metric(MetricValue::integer("slots", TOP_K as i64, "slots"))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-w4a16-routed-gate-up".to_string()),
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
        runner.group::<Sm121W4A16RoutedGateUpBench>("NVFP4 SM121 W4A16 routed gate/up", |group| {
            group.bench_sample("qwen36_layer0_top8_m1024_k2048", sm121_w4a16_sample);
            group.bench_sample(
                "qwen36_layer0_top8_m1024_k2048_bf16_output",
                sm121_w4a16_bf16_sample,
            );
            group.bench_sample("qwen36_layer0_top8_m1024_k2048_graph", graph_sample);
        });
    });
}
