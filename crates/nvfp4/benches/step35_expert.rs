use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, F32Matrix, MarlinNvfp4GateUp,
    ModelOptCheckpoint, ModelOptNvfp4Linear, Result, Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmWeight,
    indexed_grouped_gemv_on_stream, moe_silu_quantize_bf16_slots_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream,
};
use std::path::PathBuf;
use std::time::Duration;

const LAYER: usize = 3;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 1280;
const GATE_UP: usize = INTERMEDIATE * 2;
const TOP_K: usize = 8;

struct ExpertHost {
    gate: ModelOptNvfp4Linear,
    up: ModelOptNvfp4Linear,
    down: ModelOptNvfp4Linear,
}

struct Step35ExpertBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    gate_up: MarlinNvfp4GateUp,
    down_weights: Vec<Sm12xFp4DeviceGemmWeight>,
    down_tiles: DeviceBuffer<*const u8>,
    down_scales: DeviceBuffer<*const u32>,
    down_outputs: Vec<F32Matrix>,
    down_input_ptrs: DeviceBuffer<*const f32>,
    down_output_ptrs: DeviceBuffer<*mut f32>,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    input: DeviceBuffer<f32>,
    down_input_scales: DeviceBuffer<f32>,
    gate_up_unity_alphas: DeviceBuffer<f32>,
    down_alphas: DeviceBuffer<f32>,
    down_input_tiles: DeviceBuffer<u8>,
    down_input_scale_words: DeviceBuffer<u32>,
    aggregate: DeviceBuffer<f32>,
    graph: Option<CudaGraphExec>,
}

impl BenchContext for Step35ExpertBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Step-3.5 layer-3 expert benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(10)
    }
}

impl Step35ExpertBench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let experts = load_experts(&checkpoint)?;
        let input_host = (0..HIDDEN)
            .map(|idx| (((idx * 29) % 127) as f32 - 63.0) * 0.00390625)
            .collect::<Vec<_>>();
        let route_weights_host = (1..=TOP_K)
            .map(|value| value as f32 / 36.0)
            .collect::<Vec<_>>();

        let gate_up_weights = experts
            .iter()
            .enumerate()
            .map(|(expert, weights)| {
                ModelOptNvfp4Linear::concat_out_features(
                    format!("step35.layer{LAYER}.expert{expert}.gate_up"),
                    &weights.gate,
                    &weights.up,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let gate_up = MarlinNvfp4GateUp::new(&gate_up_weights)?;
        if gate_up.shape() != (GATE_UP, HIDDEN) {
            return Err(nvfp4::Error::Shape {
                label: "Step-3.5 Marlin gate/up plan",
                expected: format!("out={GATE_UP} in={HIDDEN}"),
                actual: format!("out={} in={}", gate_up.shape().0, gate_up.shape().1),
            });
        }

        let down_weights = experts
            .iter()
            .map(|expert| prepare_down_weight(&expert.down))
            .collect::<Result<Vec<_>>>()?;
        let down_tiles = DeviceBuffer::from_host(
            &down_weights
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::tiles_ptr)
                .collect::<Vec<_>>(),
        )?;
        let down_scales = DeviceBuffer::from_host(
            &down_weights
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::scales_ptr)
                .collect::<Vec<_>>(),
        )?;
        let mut down_outputs = Vec::with_capacity(TOP_K);
        let mut down_input_ptrs_host = Vec::with_capacity(TOP_K);
        let mut down_output_ptrs_host = Vec::with_capacity(TOP_K);
        for _ in 0..TOP_K {
            let mut output = F32Matrix::zeroed(HIDDEN, 1)?;
            down_input_ptrs_host.push(output.data_ptr());
            down_output_ptrs_host.push(output.data_mut_ptr());
            down_outputs.push(output);
        }

        let indices = DeviceBuffer::from_host(&(0..TOP_K as u32).collect::<Vec<_>>())?;
        let route_weights = DeviceBuffer::from_host(&route_weights_host)?;
        let input = DeviceBuffer::from_host(&input_host)?;
        let down_input_scales = DeviceBuffer::from_host(
            &experts
                .iter()
                .map(|expert| expert.down.input_scale)
                .collect::<Vec<_>>(),
        )?;
        let gate_up_unity_alphas = DeviceBuffer::from_host(&[1.0; TOP_K])?;
        let down_alphas = DeviceBuffer::from_host(
            &experts
                .iter()
                .map(|expert| expert.down.weight_scale_2 * expert.down.input_scale)
                .collect::<Vec<_>>(),
        )?;

        let mut bench = Self {
            stream: CudaStream::new_non_blocking()?,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            gate_up,
            down_weights,
            down_tiles,
            down_scales,
            down_outputs,
            down_input_ptrs: DeviceBuffer::from_host(&down_input_ptrs_host)?,
            down_output_ptrs: DeviceBuffer::from_host(&down_output_ptrs_host)?,
            indices,
            route_weights,
            input,
            down_input_scales,
            gate_up_unity_alphas,
            down_alphas,
            down_input_tiles: DeviceBuffer::zeroed(TOP_K * (INTERMEDIATE / 64) * 512)?,
            down_input_scale_words: DeviceBuffer::zeroed(TOP_K * (INTERMEDIATE / 64))?,
            aggregate: DeviceBuffer::zeroed(HIDDEN)?,
            graph: None,
        };
        bench.capture_graph()?;
        bench.validate(&experts, &input_host, &route_weights_host)?;
        Ok(bench)
    }

    fn capture_graph(&mut self) -> Result<()> {
        let graph = self.stream.capture(|stream| {
            self.gate_up
                .run_bf16_on_stream(&self.indices, &self.input, stream)?;
            moe_silu_quantize_bf16_slots_on_stream(
                &self.indices,
                self.gate_up.output_bf16(),
                &mut self.down_input_tiles,
                &mut self.down_input_scale_words,
                &self.down_input_scales,
                &self.gate_up_unity_alphas,
                INTERMEDIATE,
                TOP_K,
                stream,
            )?;
            indexed_grouped_gemv_on_stream(
                &self.indices,
                &self.down_tiles,
                &self.down_scales,
                TOP_K,
                &self.down_input_tiles,
                &self.down_input_scale_words,
                &self.down_output_ptrs,
                HIDDEN / 16,
                INTERMEDIATE / 64,
                TOP_K,
                stream,
            )?;
            moe_weighted_accumulate_slots_f32_on_stream(
                &self.indices,
                &self.route_weights,
                &self.down_input_ptrs,
                &self.down_alphas,
                self.aggregate.inout(),
                stream,
            )
        })?;
        self.graph = Some(graph);
        Ok(())
    }

    fn validate(
        &mut self,
        experts: &[ExpertHost],
        input: &[f32],
        route_weights: &[f32],
    ) -> Result<()> {
        self.graph
            .as_ref()
            .expect("captured graph")
            .launch(&self.stream)?;
        let actual = self.aggregate.copy_to_host(&self.stream)?;
        let gate_up_actual = self.gate_up.output_bf16().copy_to_host(&self.stream)?;
        let (expected, first_gate_up) = cpu_reference(experts, input, route_weights);

        let gate_up_actual = gate_up_actual[..GATE_UP]
            .iter()
            .copied()
            .map(nvfp4::format::bf16_to_f32)
            .collect::<Vec<_>>();
        require_similarity(
            "Step-3.5 layer-3 expert-0 Marlin gate/up",
            &gate_up_actual,
            &first_gate_up,
            0.998,
            0.08,
        )?;
        require_similarity(
            "Step-3.5 layer-3 top-8 expert chain",
            actual.as_ref(),
            &expected,
            0.97,
            0.30,
        )
    }
}

fn load_experts(checkpoint: &ModelOptCheckpoint) -> Result<Vec<ExpertHost>> {
    (0..TOP_K)
        .map(|expert| {
            let prefix = format!("model.layers.{LAYER}.moe.experts.{expert}");
            Ok(ExpertHost {
                gate: checkpoint.load_nvfp4_linear(&format!("{prefix}.gate_proj"))?,
                up: checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?,
                down: checkpoint.load_nvfp4_linear(&format!("{prefix}.down_proj"))?,
            })
        })
        .collect()
}

fn prepare_down_weight(linear: &ModelOptNvfp4Linear) -> Result<Sm12xFp4DeviceGemmWeight> {
    let row_major = linear.dequantize_to_f32_col_major();
    Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
        linear.out_features,
        linear.in_features,
        &row_major,
    )?
    .weight
    .to_device()
}

fn cpu_reference(
    experts: &[ExpertHost],
    input: &[f32],
    route_weights: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let input = input
        .iter()
        .copied()
        .map(|value| nvfp4::format::bf16_to_f32(nvfp4::format::f32_to_bf16(value)))
        .collect::<Vec<_>>();
    let mut aggregate = vec![0.0f32; HIDDEN];
    let mut first_gate_up = Vec::new();
    for (slot, expert) in experts.iter().enumerate() {
        let gate = cpu_linear(&expert.gate, &input, true);
        let up = cpu_linear(&expert.up, &input, true);
        if slot == 0 {
            first_gate_up.extend_from_slice(&gate);
            first_gate_up.extend_from_slice(&up);
        }
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| gate * (1.0 / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>();
        let down = cpu_linear(&expert.down, &activated, false);
        for (output, value) in aggregate.iter_mut().zip(down) {
            *output += value * route_weights[slot];
        }
    }
    (aggregate, first_gate_up)
}

fn cpu_linear(weight: &ModelOptNvfp4Linear, input: &[f32], round_bf16: bool) -> Vec<f32> {
    let values = weight.dequantize_to_f32_col_major();
    values
        .chunks_exact(weight.in_features)
        .map(|row| {
            let value = row
                .iter()
                .zip(input)
                .map(|(&weight, &input)| weight * input)
                .sum::<f32>()
                * weight.weight_scale_2;
            if round_bf16 {
                nvfp4::format::bf16_to_f32(nvfp4::format::f32_to_bf16(value))
            } else {
                value
            }
        })
        .collect()
}

fn require_similarity(
    label: &'static str,
    actual: &[f32],
    expected: &[f32],
    minimum_cosine: f64,
    maximum_nrmse: f64,
) -> Result<()> {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        dot += actual as f64 * expected as f64;
        actual_norm += (actual as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        squared_error += ((actual - expected) as f64).powi(2);
        let error = (actual - expected).abs();
        if error > worst.0 {
            worst = (error, index, actual, expected);
        }
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    if !cosine.is_finite() || !nrmse.is_finite() || cosine < minimum_cosine || nrmse > maximum_nrmse
    {
        return Err(nvfp4::Error::Format {
            label,
            detail: format!(
                "cosine={cosine:.6} required>={minimum_cosine:.6} nrmse={nrmse:.6} required<={maximum_nrmse:.6} worst_index={} actual={} expected={} abs_error={}",
                worst.1, worst.2, worst.3, worst.0
            ),
        });
    }
    eprintln!(
        "validated {label}: cosine={cosine:.6}, nrmse={nrmse:.6}, worst_abs={:.6}",
        worst.0
    );
    Ok(())
}

fn model_dir() -> PathBuf {
    std::env::var_os("STEP35_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.5-flash-nvfp4")
        })
}

fn expert_chain_sample(
    context: &mut Step35ExpertBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        context
            .graph
            .as_ref()
            .expect("captured graph")
            .launch(&context.stream)
            .expect("Step-3.5 expert graph");
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop event");
    context.stop.synchronize().expect("synchronize stop event");
    let total_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64;
    black_box(context.aggregate.as_const_ptr());
    black_box(context.down_weights[0].tiles_ptr());
    black_box(context.down_outputs[0].data_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
                .with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer("experts", TOP_K as i64, "experts"))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-step35-expert".to_string()),
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
        runner.group::<Step35ExpertBench>("Step-3.5 layer-3 routed experts", |group| {
            group.bench_sample("top8_marlin_gate_up_swiglu_sm12x_down", expert_chain_sample);
        });
    });
}
