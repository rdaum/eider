use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaEvent, CudaStream, CutlassFp4GroupedGemmPlan, CutlassFp4GroupedGemvF32Plan, DeviceBuffer,
    F32Matrix, ModelOptCublasLtWeight, ModelOptNvfp4Linear, MoeSortedNvfp4Rows, MoeSortedRoutes,
    Nvfp4Matrix, Result, Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmWeight, Sm121W4A16GateUp,
    indexed_grouped_gemv_on_stream, moe_silu_quantize_bf16_slots_on_stream,
    moe_silu_quantize_slots_on_stream, moe_weighted_accumulate_slots_f32_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
};
use std::time::Duration;

const HIDDEN: usize = 2_048;
const INTERMEDIATE: usize = 512;
const GATE_UP: usize = INTERMEDIATE * 2;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const ROUTES: [u32; TOP_K] = [7, 2, 199, 0, 155, 1, 206, 33];

struct DownWorkspace {
    b_tiles: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u32>,
    outputs: Vec<F32Matrix>,
    output_mut_table: DeviceBuffer<*mut f32>,
    output_table: DeviceBuffer<*const f32>,
    reduced: DeviceBuffer<f32>,
}

struct GroupedW4A4Workspace {
    sorted_routes: MoeSortedRoutes,
    gate_up_input: MoeSortedNvfp4Rows,
    down_input: MoeSortedNvfp4Rows,
    gate_up_plan: CutlassFp4GroupedGemmPlan,
    down_plan: CutlassFp4GroupedGemmPlan,
    gate_up: DeviceBuffer<u16>,
    down: DeviceBuffer<u16>,
    gate_up_output_table: DeviceBuffer<*mut u16>,
    down_output_table: DeviceBuffer<*mut u16>,
    reduced: DeviceBuffer<f32>,
}

impl GroupedW4A4Workspace {
    fn new() -> Result<Self> {
        Ok(Self {
            sorted_routes: MoeSortedRoutes::new(TOP_K, EXPERTS)?,
            gate_up_input: MoeSortedNvfp4Rows::new(1, TOP_K, EXPERTS, HIDDEN)?,
            down_input: MoeSortedNvfp4Rows::new(1, TOP_K, EXPERTS, INTERMEDIATE)?,
            gate_up_plan: CutlassFp4GroupedGemmPlan::new(GATE_UP, TOP_K, HIDDEN, EXPERTS)?,
            down_plan: CutlassFp4GroupedGemmPlan::new(HIDDEN, TOP_K, INTERMEDIATE, EXPERTS)?,
            gate_up: DeviceBuffer::zeroed(TOP_K * GATE_UP)?,
            down: DeviceBuffer::zeroed(TOP_K * HIDDEN)?,
            gate_up_output_table: DeviceBuffer::zeroed(EXPERTS)?,
            down_output_table: DeviceBuffer::zeroed(EXPERTS)?,
            reduced: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }
}

impl DownWorkspace {
    fn new() -> Result<Self> {
        let mut outputs = (0..TOP_K)
            .map(|_| F32Matrix::zeroed(HIDDEN, 1))
            .collect::<Result<Vec<_>>>()?;
        let output_mut_table = DeviceBuffer::from_host(
            &outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )?;
        let output_table = DeviceBuffer::from_host(
            &outputs
                .iter()
                .map(|output| output.data_ptr())
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            b_tiles: DeviceBuffer::zeroed(TOP_K * (INTERMEDIATE / 64) * 512)?,
            b_scales: DeviceBuffer::zeroed(TOP_K * (INTERMEDIATE / 64))?,
            outputs,
            output_mut_table,
            output_table,
            reduced: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }
}

struct Qwen36RoutedMoeDecodeBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    input: DeviceBuffer<f32>,
    w4a16_gate_up: Sm121W4A16GateUp,
    w4a4_gate_up_weights: Vec<ModelOptCublasLtWeight>,
    w4a4_gate_up_values: DeviceBuffer<*const u8>,
    w4a4_gate_up_scales: DeviceBuffer<*const u8>,
    w4a4_gate_up_alphas: DeviceBuffer<f32>,
    w4a4_gate_up_input: Nvfp4Matrix,
    w4a4_gate_up_plan: CutlassFp4GroupedGemvF32Plan,
    w4a4_gate_up_zero: F32Matrix,
    w4a4_gate_up_outputs: Vec<F32Matrix>,
    w4a4_gate_up_output_mut_table: DeviceBuffer<*mut f32>,
    w4a4_gate_up_output_table: DeviceBuffer<*const f32>,
    grouped_gate_up_alpha_values: DeviceBuffer<f32>,
    grouped_gate_up_alpha_table: DeviceBuffer<*mut f32>,
    grouped_down_weights: Vec<ModelOptCublasLtWeight>,
    grouped_down_values: DeviceBuffer<*const u8>,
    grouped_down_scales: DeviceBuffer<*const u8>,
    grouped_down_alpha_values: DeviceBuffer<f32>,
    grouped_down_alpha_table: DeviceBuffer<*mut f32>,
    grouped_w4a4: GroupedW4A4Workspace,
    down_weights: Vec<Sm12xFp4DeviceGemmWeight>,
    down_tiles: DeviceBuffer<*const u8>,
    down_scales: DeviceBuffer<*const u32>,
    down_input_scales: DeviceBuffer<f32>,
    down_alphas: DeviceBuffer<f32>,
    unity_gate_up_alphas: DeviceBuffer<f32>,
    w4a16_down: DownWorkspace,
    w4a4_down: DownWorkspace,
    gate_up_nrmse: f64,
    layer_nrmse: f64,
    layer_max_abs_error: f32,
}

impl BenchContext for Qwen36RoutedMoeDecodeBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Qwen3.6 routed MoE decode benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(10)
    }
}

impl Qwen36RoutedMoeDecodeBench {
    fn new() -> Result<Self> {
        let gate_up_host = synthetic_gate_up_weights();
        let w4a16_gate_up = Sm121W4A16GateUp::new_with_top_k(&gate_up_host, TOP_K)?;
        let w4a4_gate_up_weights = gate_up_host
            .iter()
            .map(ModelOptNvfp4Linear::as_cublaslt_weight)
            .collect::<Result<Vec<_>>>()?;
        let w4a4_gate_up_values = DeviceBuffer::from_host(
            &w4a4_gate_up_weights
                .iter()
                .map(|weight| weight.matrix().values_ptr())
                .collect::<Vec<_>>(),
        )?;
        let w4a4_gate_up_scales = DeviceBuffer::from_host(
            &w4a4_gate_up_weights
                .iter()
                .map(|weight| weight.matrix().scales_ptr())
                .collect::<Vec<_>>(),
        )?;
        let w4a4_gate_up_alphas = DeviceBuffer::from_host(
            &w4a4_gate_up_weights
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let mut w4a4_gate_up_outputs = (0..TOP_K)
            .map(|_| F32Matrix::zeroed(GATE_UP, 1))
            .collect::<Result<Vec<_>>>()?;
        let w4a4_gate_up_output_mut_table = DeviceBuffer::from_host(
            &w4a4_gate_up_outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )?;
        let w4a4_gate_up_output_table = DeviceBuffer::from_host(
            &w4a4_gate_up_outputs
                .iter()
                .map(|output| output.data_ptr())
                .collect::<Vec<_>>(),
        )?;
        let mut grouped_gate_up_alpha_values = DeviceBuffer::from_host(
            &w4a4_gate_up_weights
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let grouped_gate_up_alpha_table = scalar_pointer_table(&mut grouped_gate_up_alpha_values)?;

        let down_host = synthetic_down_weights();
        let down_weights = down_host
            .iter()
            .map(|weight| {
                Sm12xFp4GemmWeight::from_modelopt_row_major(
                    HIDDEN,
                    INTERMEDIATE,
                    &weight.packed_weight,
                    &weight.weight_scale,
                )?
                .to_device()
            })
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
        let grouped_down_weights = down_host
            .iter()
            .map(ModelOptNvfp4Linear::as_cublaslt_weight)
            .collect::<Result<Vec<_>>>()?;
        let grouped_down_values = DeviceBuffer::from_host(
            &grouped_down_weights
                .iter()
                .map(|weight| weight.matrix().values_ptr())
                .collect::<Vec<_>>(),
        )?;
        let grouped_down_scales = DeviceBuffer::from_host(
            &grouped_down_weights
                .iter()
                .map(|weight| weight.matrix().scales_ptr())
                .collect::<Vec<_>>(),
        )?;
        let mut grouped_down_alpha_values = DeviceBuffer::from_host(
            &grouped_down_weights
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let grouped_down_alpha_table = scalar_pointer_table(&mut grouped_down_alpha_values)?;
        let stream = CudaStream::new_non_blocking()?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            indices: DeviceBuffer::from_host(&ROUTES)?,
            route_weights: DeviceBuffer::from_host(&[1.0 / TOP_K as f32; TOP_K])?,
            input: DeviceBuffer::from_host(
                &(0..HIDDEN)
                    .map(|index| ((index * 17 % 257) as f32 + 1.0) / 512.0)
                    .collect::<Vec<_>>(),
            )?,
            w4a16_gate_up,
            w4a4_gate_up_weights,
            w4a4_gate_up_values,
            w4a4_gate_up_scales,
            w4a4_gate_up_alphas,
            w4a4_gate_up_input: Nvfp4Matrix::zeroed_col_major(HIDDEN, 1)?,
            w4a4_gate_up_plan: CutlassFp4GroupedGemvF32Plan::new(GATE_UP, HIDDEN, TOP_K)?,
            w4a4_gate_up_zero: F32Matrix::zeroed(GATE_UP, 1)?,
            w4a4_gate_up_outputs,
            w4a4_gate_up_output_mut_table,
            w4a4_gate_up_output_table,
            grouped_gate_up_alpha_values,
            grouped_gate_up_alpha_table,
            grouped_down_weights,
            grouped_down_values,
            grouped_down_scales,
            grouped_down_alpha_values,
            grouped_down_alpha_table,
            grouped_w4a4: GroupedW4A4Workspace::new()?,
            down_weights,
            down_tiles,
            down_scales,
            down_input_scales: DeviceBuffer::from_host(&[1.0; EXPERTS])?,
            down_alphas: DeviceBuffer::from_host(&vec![1.0 / INTERMEDIATE as f32; EXPERTS])?,
            unity_gate_up_alphas: DeviceBuffer::from_host(&[1.0; EXPERTS])?,
            w4a16_down: DownWorkspace::new()?,
            w4a4_down: DownWorkspace::new()?,
            gate_up_nrmse: 0.0,
            layer_nrmse: 0.0,
            layer_max_abs_error: 0.0,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn enqueue_w4a16(&mut self) -> Result<()> {
        self.w4a16_gate_up
            .run_bf16_on_stream(&self.indices, &self.input, &self.stream)?;
        moe_silu_quantize_bf16_slots_on_stream(
            &self.indices,
            self.w4a16_gate_up.output_bf16(),
            &mut self.w4a16_down.b_tiles,
            &mut self.w4a16_down.b_scales,
            &self.down_input_scales,
            &self.unity_gate_up_alphas,
            INTERMEDIATE,
            TOP_K,
            &self.stream,
        )?;
        indexed_grouped_gemv_on_stream(
            &self.indices,
            &self.down_tiles,
            &self.down_scales,
            EXPERTS,
            &self.w4a16_down.b_tiles,
            &self.w4a16_down.b_scales,
            &self.w4a16_down.output_mut_table,
            HIDDEN / 16,
            INTERMEDIATE / 64,
            TOP_K,
            &self.stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            &self.indices,
            &self.route_weights,
            &self.w4a16_down.output_table,
            &self.down_alphas,
            self.w4a16_down.reduced.inout(),
            &self.stream,
        )
    }

    fn enqueue_w4a4(&mut self) -> Result<()> {
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            HIDDEN,
            1,
            &self.input,
            &mut self.w4a4_gate_up_input,
            1.0,
            &self.stream,
        )?;
        self.w4a4_gate_up_plan
            .run_indexed_a_tiled_scales_on_stream(
                &self.indices,
                &self.w4a4_gate_up_values,
                &self.w4a4_gate_up_scales,
                &self.w4a4_gate_up_alphas,
                &self.w4a4_gate_up_input,
                &self.w4a4_gate_up_zero,
                &self.w4a4_gate_up_output_mut_table,
                &self.stream,
            )?;
        moe_silu_quantize_slots_on_stream(
            &self.indices,
            &self.w4a4_gate_up_output_table,
            &mut self.w4a4_down.b_tiles,
            &mut self.w4a4_down.b_scales,
            &self.down_input_scales,
            &self.unity_gate_up_alphas,
            INTERMEDIATE,
            TOP_K,
            &self.stream,
        )?;
        indexed_grouped_gemv_on_stream(
            &self.indices,
            &self.down_tiles,
            &self.down_scales,
            EXPERTS,
            &self.w4a4_down.b_tiles,
            &self.w4a4_down.b_scales,
            &self.w4a4_down.output_mut_table,
            HIDDEN / 16,
            INTERMEDIATE / 64,
            TOP_K,
            &self.stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            &self.indices,
            &self.route_weights,
            &self.w4a4_down.output_table,
            &self.down_alphas,
            self.w4a4_down.reduced.inout(),
            &self.stream,
        )
    }

    fn enqueue_grouped_w4a4(&mut self) -> Result<()> {
        let grouped = &mut self.grouped_w4a4;
        grouped
            .sorted_routes
            .sort_on_stream(&self.indices, &self.stream)?;
        grouped.gate_up_input.gather_quantize_on_stream(
            &self.input,
            &grouped.sorted_routes,
            &self.stream,
        )?;
        grouped.gate_up_input.build_pointer_tables_on_stream(
            &grouped.sorted_routes,
            &mut grouped.gate_up,
            &mut grouped.gate_up_output_table,
            GATE_UP,
            &self.stream,
        )?;
        grouped.gate_up_plan.run_on_stream(
            &self.w4a4_gate_up_values,
            &self.w4a4_gate_up_scales,
            grouped.gate_up_input.packed_table(),
            grouped.gate_up_input.scale_table(),
            &grouped.gate_up_output_table,
            &self.grouped_gate_up_alpha_table,
            grouped.sorted_routes.expert_counts(),
            &self.stream,
        )?;
        grouped
            .down_input
            .silu_mul_halves_quantize_sorted_on_stream(
                &grouped.gate_up,
                &grouped.sorted_routes,
                &self.stream,
            )?;
        grouped.down_input.build_pointer_tables_on_stream(
            &grouped.sorted_routes,
            &mut grouped.down,
            &mut grouped.down_output_table,
            HIDDEN,
            &self.stream,
        )?;
        grouped.down_plan.run_on_stream(
            &self.grouped_down_values,
            &self.grouped_down_scales,
            grouped.down_input.packed_table(),
            grouped.down_input.scale_table(),
            &grouped.down_output_table,
            &self.grouped_down_alpha_table,
            grouped.sorted_routes.expert_counts(),
            &self.stream,
        )?;
        moe_weighted_accumulate_sorted_bf16_batch_on_stream(
            &grouped.sorted_routes,
            &self.route_weights,
            &grouped.down,
            grouped.reduced.output(),
            1,
            TOP_K,
            HIDDEN,
            &self.stream,
        )
    }

    fn validate(&mut self) -> Result<()> {
        self.enqueue_w4a16()?;
        self.enqueue_w4a4()?;
        self.enqueue_grouped_w4a4()?;

        let w4a16_gate_up = self
            .w4a16_gate_up
            .output_bf16()
            .copy_to_host(&self.stream)?;
        let w4a4_gate_up = self
            .w4a4_gate_up_outputs
            .iter()
            .map(|output| output.data().copy_to_host(&self.stream))
            .collect::<Result<Vec<_>>>()?;
        let mut gate_error = 0.0f64;
        let mut gate_reference = 0.0f64;
        for route in 0..TOP_K {
            for feature in 0..GATE_UP {
                let reference =
                    nvfp4::format::bf16_to_f32(w4a16_gate_up[route * GATE_UP + feature]) as f64;
                let actual = w4a4_gate_up[route][feature] as f64;
                let error = actual - reference;
                gate_error += error * error;
                gate_reference += reference * reference;
            }
        }
        self.gate_up_nrmse = (gate_error / gate_reference.max(f64::MIN_POSITIVE)).sqrt();
        if !self.gate_up_nrmse.is_finite() || self.gate_up_nrmse > 0.08 {
            return Err(nvfp4::Error::Format {
                label: "Qwen3.6 W4A4 gate/up correctness",
                detail: format!("W4A4 versus W4A16 nrmse={:.6}", self.gate_up_nrmse),
            });
        }

        let w4a16 = self.w4a16_down.reduced.copy_to_host(&self.stream)?;
        let w4a4 = self.w4a4_down.reduced.copy_to_host(&self.stream)?;
        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_abs_error = 0.0f32;
        for (&reference, &actual) in w4a16.iter().zip(w4a4.iter()) {
            let error = actual as f64 - reference as f64;
            squared_error += error * error;
            reference_norm += (reference as f64) * (reference as f64);
            max_abs_error = max_abs_error.max((actual - reference).abs());
        }
        self.layer_nrmse = (squared_error / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
        self.layer_max_abs_error = max_abs_error;
        if !self.layer_nrmse.is_finite() || self.layer_nrmse > 0.10 {
            return Err(nvfp4::Error::Format {
                label: "Qwen3.6 W4A4 routed layer correctness",
                detail: format!(
                    "W4A4 versus W4A16 nrmse={:.6} max_abs_error={max_abs_error:.6}",
                    self.layer_nrmse
                ),
            });
        }
        let grouped = self.grouped_w4a4.reduced.copy_to_host(&self.stream)?;
        let grouped_nrmse = nrmse(&w4a4, &grouped);
        if !grouped_nrmse.is_finite() || grouped_nrmse > 0.10 {
            return Err(nvfp4::Error::Format {
                label: "Qwen3.6 grouped W4A4 routed layer correctness",
                detail: format!("grouped versus indexed W4A4 nrmse={grouped_nrmse:.6}"),
            });
        }
        Ok(())
    }

    fn measure(
        &mut self,
        chunk_size: usize,
        enqueue: fn(&mut Self) -> Result<()>,
    ) -> BenchSampleResult {
        self.start
            .record_on_stream(&self.stream)
            .expect("record start event");
        for _ in 0..chunk_size {
            enqueue(self).expect("enqueue Qwen3.6 routed MoE");
        }
        self.stop
            .record_on_stream(&self.stream)
            .expect("record stop event");
        self.stop.synchronize().expect("synchronize stop event");
        let elapsed_ms = self
            .start
            .elapsed_ms_until(&self.stop)
            .expect("elapsed time") as f64
            / chunk_size as f64;
        black_box(self.w4a16_down.reduced.as_const_ptr());
        black_box(self.w4a4_down.reduced.as_const_ptr());
        black_box(self.grouped_w4a4.reduced.as_const_ptr());
        black_box(self.w4a4_gate_up_weights.len());
        black_box(self.grouped_gate_up_alpha_values.len());
        black_box(self.grouped_down_weights.len());
        black_box(self.grouped_down_alpha_values.len());
        black_box(self.down_weights.len());
        black_box(self.w4a16_down.outputs.len());
        BenchSampleResult::operations(chunk_size as u64)
            .push_metric(MetricValue::new("cuda_event_ms", elapsed_ms, "ms"))
            .push_metric(MetricValue::new(
                "gate_up_w4a4_vs_w4a16_nrmse",
                self.gate_up_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "layer_w4a4_vs_w4a16_nrmse",
                self.layer_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "layer_w4a4_vs_w4a16_max_abs_error",
                self.layer_max_abs_error as f64,
                "value",
            ))
    }
}

fn synthetic_gate_up_weights() -> Vec<ModelOptNvfp4Linear> {
    (0..EXPERTS)
        .map(|expert| ModelOptNvfp4Linear {
            prefix: format!("qwen36.experts.{expert}.gate_up_proj"),
            out_features: GATE_UP,
            in_features: HIDDEN,
            packed_weight: vec![0x22 + (expert % 2) as u8; GATE_UP * HIDDEN / 2],
            weight_scale: vec![0x38; GATE_UP * HIDDEN / 16],
            weight_scale_2: (1.0 + expert as f32 / (EXPERTS * 8) as f32) / HIDDEN as f32,
            input_scale: 1.0,
        })
        .collect()
}

fn synthetic_down_weights() -> Vec<ModelOptNvfp4Linear> {
    (0..EXPERTS)
        .map(|expert| ModelOptNvfp4Linear {
            prefix: format!("qwen36.experts.{expert}.down_proj"),
            out_features: HIDDEN,
            in_features: INTERMEDIATE,
            packed_weight: vec![0x22 + (expert % 2) as u8; HIDDEN * INTERMEDIATE / 2],
            weight_scale: vec![0x38; HIDDEN * INTERMEDIATE / 16],
            weight_scale_2: 1.0 / INTERMEDIATE as f32,
            input_scale: 1.0,
        })
        .collect()
}

fn nrmse(reference: &[f32], actual: &[f32]) -> f64 {
    let (error, norm) = reference.iter().zip(actual).fold(
        (0.0f64, 0.0f64),
        |(error, norm), (&reference, &actual)| {
            let delta = actual as f64 - reference as f64;
            (
                error + delta * delta,
                norm + reference as f64 * reference as f64,
            )
        },
    );
    (error / norm.max(f64::MIN_POSITIVE)).sqrt()
}

fn scalar_pointer_table(values: &mut DeviceBuffer<f32>) -> Result<DeviceBuffer<*mut f32>> {
    let base = values.as_const_ptr().cast::<f32>().cast_mut();
    DeviceBuffer::from_host(
        &(0..values.len())
            .map(|index| unsafe { base.add(index) })
            .collect::<Vec<_>>(),
    )
}

fn w4a16_sample(
    context: &mut Qwen36RoutedMoeDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen36RoutedMoeDecodeBench::enqueue_w4a16)
}

fn w4a4_sample(
    context: &mut Qwen36RoutedMoeDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen36RoutedMoeDecodeBench::enqueue_w4a4)
}

fn grouped_w4a4_sample(
    context: &mut Qwen36RoutedMoeDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen36RoutedMoeDecodeBench::enqueue_grouped_w4a4)
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen36-routed-moe-decode".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(100),
                benchmark_duration: Duration::from_millis(500),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Qwen36RoutedMoeDecodeBench>("Qwen3.6 routed MoE decode", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("sm121_w4a16_full_layer", w4a16_sample);
                group.bench_sample("cutlass_w4a4_full_layer", w4a4_sample);
                group.bench_sample("grouped_w4a4_full_layer", grouped_w4a4_sample);
            });
        },
    );
}
