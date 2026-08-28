use eider_cuda::{
    CublasLt, CudaEvent, CudaStream, CutlassFp4GroupedGemmPlan, CutlassFp4GroupedGemvF32Plan,
    DeviceBuffer, F32Matrix, Fp4TnMatmulPlan, GemmShape, ModelOptCublasLtWeight,
    MoeSortedNvfp4Rows, MoeSortedRoutes, Nvfp4Matrix, Nvfp4TnInputs, Result, Sm121W4A16GateUp,
    Sm121W4A16GateUpBatchWorkspace, format, moe_silu_quantize_bf16_expert_sorted_slots_on_stream,
    moe_silu_quantize_bf16_sorted_slots_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    quantize_nvfp4_vector_simple_scales_f32_into_on_stream,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const ROWS: usize = 1_024;
const HIDDEN: usize = 3_072;
const INTERMEDIATE: usize = 1_024;
const GATE_UP: usize = INTERMEDIATE * 2;
const EXPERTS: usize = 256;
const TOP_K: usize = 10;
const ROUTES: usize = ROWS * TOP_K;

struct LagunaRoutedGateUpBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    sorted_routes: MoeSortedRoutes,
    w4a16: Sm121W4A16GateUp,
    w4a16_workspace: Sm121W4A16GateUpBatchWorkspace,
    grouped_weights: Vec<ModelOptCublasLtWeight>,
    grouped_weight_values: DeviceBuffer<*const u8>,
    grouped_weight_scales: DeviceBuffer<*const u8>,
    grouped_alphas: DeviceBuffer<f32>,
    grouped_alpha_table: DeviceBuffer<*mut f32>,
    grouped_input: MoeSortedNvfp4Rows,
    grouped_plan: CutlassFp4GroupedGemmPlan,
    grouped_output: DeviceBuffer<u16>,
    grouped_output_table: DeviceBuffer<*mut u16>,
    decode_input: DeviceBuffer<f32>,
    decode_indices: DeviceBuffer<u32>,
    decode_input_nvfp4: Nvfp4Matrix,
    decode_input_scales: DeviceBuffer<u8>,
    decode_plan: CutlassFp4GroupedGemvF32Plan,
    decode_outputs: Vec<F32Matrix>,
    decode_output_table: DeviceBuffer<*mut f32>,
    decode_cutlass_plan: Fp4TnMatmulPlan,
    decode_cutlass_c: F32Matrix,
    decode_cutlass_outputs: Vec<F32Matrix>,
    decode_cutlass_output_table: DeviceBuffer<*mut f32>,
    validation_nrmse: f64,
    validation_max_abs_error: f32,
    decode_validation_nrmse: f64,
    decode_validation_max_abs_error: f32,
    decode_cutlass_validation_nrmse: f64,
    decode_cutlass_validation_max_abs_error: f32,
    decode_indexed_cutlass_validation_nrmse: f64,
    decode_indexed_cutlass_validation_max_abs_error: f32,
    w4a16_reference_nrmse: f64,
    w4a16_reference_max_abs_error: f32,
}

impl BenchContext for LagunaRoutedGateUpBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Laguna routed gate/up benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl LagunaRoutedGateUpBench {
    fn new() -> Result<Self> {
        let weights = synthetic_weights();
        let w4a16 = Sm121W4A16GateUp::new_with_top_k(&weights, TOP_K)?;
        let w4a16_workspace = w4a16.new_batch_workspace(ROWS)?;
        let grouped_weights = weights
            .iter()
            .map(ModelOptCublasLtWeight::from_modelopt)
            .collect::<Result<Vec<_>>>()?;
        let grouped_weight_values = DeviceBuffer::from_host(
            &grouped_weights
                .iter()
                .map(|weight| weight.matrix().values_ptr())
                .collect::<Vec<_>>(),
        )?;
        let grouped_weight_scales = DeviceBuffer::from_host(
            &grouped_weights
                .iter()
                .map(|weight| weight.matrix().scales_ptr())
                .collect::<Vec<_>>(),
        )?;
        let mut grouped_alphas = DeviceBuffer::from_host(
            &grouped_weights
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let grouped_alpha_table = scalar_pointer_table(&mut grouped_alphas)?;
        let input = DeviceBuffer::from_host(
            &(0..ROWS * HIDDEN)
                .map(|index| {
                    // Keep the synthetic dot products away from zero so the
                    // relative-error gate measures quantization fidelity
                    // instead of cancellation noise.
                    let value = ((index * 17 + index / HIDDEN * 13) % 257) as f32 + 1.0;
                    value / 512.0
                })
                .collect::<Vec<_>>(),
        )?;
        let indices = DeviceBuffer::from_host(
            &(0..ROUTES)
                .map(|route| (route % EXPERTS) as u32)
                .collect::<Vec<_>>(),
        )?;
        let decode_input = DeviceBuffer::from_host(
            &(0..HIDDEN)
                .map(|index| ((index * 17 % 257) as f32 + 1.0) / 512.0)
                .collect::<Vec<_>>(),
        )?;
        let decode_indices =
            DeviceBuffer::from_host(&(0..TOP_K).map(|route| route as u32).collect::<Vec<_>>())?;
        let decode_input_nvfp4 = Nvfp4Matrix::zeroed_col_major(HIDDEN, 1)?;
        let mut decode_outputs = (0..TOP_K)
            .map(|_| F32Matrix::zeroed(GATE_UP, 1))
            .collect::<Result<Vec<_>>>()?;
        let decode_output_table = DeviceBuffer::from_host(
            &decode_outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr().cast())
                .collect::<Vec<_>>(),
        )?;
        let decode_cutlass_c = F32Matrix::zeroed(GATE_UP, 1)?;
        let mut decode_cutlass_outputs = (0..TOP_K)
            .map(|_| F32Matrix::zeroed(GATE_UP, 1))
            .collect::<Result<Vec<_>>>()?;
        let decode_cutlass_output_table = DeviceBuffer::from_host(
            &decode_cutlass_outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr().cast())
                .collect::<Vec<_>>(),
        )?;
        let lt = CublasLt::new()?;
        let decode_cutlass_plan = Fp4TnMatmulPlan::new_f32_output(
            &lt,
            GemmShape::new(GATE_UP, 1, HIDDEN),
            Nvfp4TnInputs::new(grouped_weights[0].matrix(), &decode_input_nvfp4),
            &decode_cutlass_c,
            4 * 1024 * 1024,
        )?;
        let stream = CudaStream::new_non_blocking()?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            input,
            indices,
            sorted_routes: MoeSortedRoutes::new(ROUTES, EXPERTS)?,
            w4a16,
            w4a16_workspace,
            grouped_weights,
            grouped_weight_values,
            grouped_weight_scales,
            grouped_alphas,
            grouped_alpha_table,
            grouped_input: MoeSortedNvfp4Rows::new(ROWS, TOP_K, EXPERTS, HIDDEN)?,
            grouped_plan: CutlassFp4GroupedGemmPlan::new(GATE_UP, ROUTES, HIDDEN, EXPERTS)?,
            grouped_output: DeviceBuffer::zeroed(ROUTES * GATE_UP)?,
            grouped_output_table: DeviceBuffer::zeroed(EXPERTS)?,
            decode_input,
            decode_indices,
            decode_input_nvfp4,
            decode_input_scales: DeviceBuffer::zeroed(HIDDEN.div_ceil(16))?,
            decode_plan: CutlassFp4GroupedGemvF32Plan::new(GATE_UP, HIDDEN, TOP_K)?,
            decode_outputs,
            decode_output_table,
            decode_cutlass_plan,
            decode_cutlass_c,
            decode_cutlass_outputs,
            decode_cutlass_output_table,
            validation_nrmse: 0.0,
            validation_max_abs_error: 0.0,
            decode_validation_nrmse: 0.0,
            decode_validation_max_abs_error: 0.0,
            decode_cutlass_validation_nrmse: 0.0,
            decode_cutlass_validation_max_abs_error: 0.0,
            decode_indexed_cutlass_validation_nrmse: 0.0,
            decode_indexed_cutlass_validation_max_abs_error: 0.0,
            w4a16_reference_nrmse: 0.0,
            w4a16_reference_max_abs_error: 0.0,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn enqueue_w4a16_pipeline(&mut self) -> Result<()> {
        self.w4a16.run_batch_bf16_on_stream(
            &self.w4a16_workspace,
            &self.indices,
            &self.input,
            &self.stream,
        )?;
        self.sorted_routes
            .sort_on_stream(&self.indices, &self.stream)
    }

    fn enqueue_grouped_prepare(&mut self) -> Result<()> {
        self.sorted_routes
            .sort_on_stream(&self.indices, &self.stream)?;
        self.grouped_input.gather_quantize_on_stream(
            &self.input,
            &self.sorted_routes,
            &self.stream,
        )?;
        self.grouped_input.build_pointer_tables_on_stream(
            &self.sorted_routes,
            &mut self.grouped_output,
            &mut self.grouped_output_table,
            GATE_UP,
            &self.stream,
        )
    }

    fn enqueue_grouped_gemm(&mut self) -> Result<()> {
        self.grouped_plan.run_on_stream(
            &self.grouped_weight_values,
            &self.grouped_weight_scales,
            self.grouped_input.packed_table(),
            self.grouped_input.scale_table(),
            &self.grouped_output_table,
            &self.grouped_alpha_table,
            self.sorted_routes.expert_counts(),
            &self.stream,
        )
    }

    fn enqueue_grouped_pipeline(&mut self) -> Result<()> {
        self.enqueue_grouped_prepare()?;
        self.enqueue_grouped_gemm()
    }

    fn enqueue_w4a16_decode(&mut self) -> Result<()> {
        self.w4a16
            .run_bf16_on_stream(&self.decode_indices, &self.decode_input, &self.stream)
            .map(|_| ())
    }

    fn enqueue_grouped_decode(&mut self) -> Result<()> {
        self.enqueue_grouped_decode_quantize()?;
        self.enqueue_grouped_decode_gemv()
    }

    fn enqueue_grouped_decode_quantize(&mut self) -> Result<()> {
        quantize_nvfp4_vector_simple_scales_f32_into_on_stream(
            HIDDEN,
            &self.decode_input,
            &mut self.decode_input_nvfp4,
            &mut self.decode_input_scales,
            1.0,
            &self.stream,
        )
    }

    fn enqueue_grouped_decode_gemv(&mut self) -> Result<()> {
        unsafe {
            self.decode_plan.run_indexed_a_on_stream(
                &self.decode_indices,
                &self.grouped_weight_values,
                &self.grouped_weight_scales,
                EXPERTS,
                self.decode_input_nvfp4.values_ptr(),
                self.decode_input_scales.as_const_ptr().cast(),
                &self.decode_output_table,
                1.0 / HIDDEN as f32,
                &self.stream,
            )
        }
    }

    fn enqueue_cutlass_decode(&mut self) -> Result<()> {
        self.enqueue_cutlass_decode_quantize()?;
        self.enqueue_cutlass_decode_gemv()
    }

    fn enqueue_cutlass_decode_quantize(&mut self) -> Result<()> {
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            HIDDEN,
            1,
            &self.decode_input,
            &mut self.decode_input_nvfp4,
            1.0,
            &self.stream,
        )
    }

    fn enqueue_cutlass_decode_gemv(&mut self) -> Result<()> {
        for route in 0..TOP_K {
            let weight = &self.grouped_weights[route];
            self.decode_cutlass_plan
                .run_cutlass_fp4_gemv_f32_on_stream(
                    Nvfp4TnInputs::new(weight.matrix(), &self.decode_input_nvfp4),
                    &self.decode_cutlass_c,
                    &mut self.decode_cutlass_outputs[route],
                    weight.matmul_alpha(),
                    &self.stream,
                )?;
        }
        Ok(())
    }

    fn enqueue_indexed_cutlass_decode(&mut self) -> Result<()> {
        self.enqueue_cutlass_decode_quantize()?;
        self.enqueue_indexed_cutlass_decode_gemv()
    }

    fn enqueue_indexed_cutlass_decode_gemv(&mut self) -> Result<()> {
        self.decode_plan.run_indexed_a_tiled_scales_on_stream(
            &self.decode_indices,
            &self.grouped_weight_values,
            &self.grouped_weight_scales,
            &self.grouped_alphas,
            &self.decode_input_nvfp4,
            &self.decode_cutlass_c,
            &self.decode_cutlass_output_table,
            &self.stream,
        )
    }

    fn validate(&mut self) -> Result<()> {
        self.enqueue_w4a16_pipeline()?;
        let w4a16 = self
            .w4a16_workspace
            .output_bf16()
            .copy_to_host(&self.stream)?
            .into_vec();
        self.enqueue_grouped_pipeline()?;
        let grouped = self.grouped_output.copy_to_host(&self.stream)?.into_vec();
        let sorted_routes = self
            .sorted_routes
            .sorted_routes()
            .copy_to_host(&self.stream)?
            .into_vec();
        let counts = self
            .sorted_routes
            .expert_counts()
            .copy_to_host(&self.stream)?
            .into_vec();
        if counts.iter().map(|&count| count as usize).sum::<usize>() != ROUTES {
            return Err(eider_cuda::Error::Format {
                label: "Laguna W4A4 gate/up route counts",
                detail: "expert route counts do not sum to the active route count".to_string(),
            });
        }

        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_error = 0.0f32;
        for (sorted, &original) in sorted_routes.iter().enumerate() {
            let original = original as usize;
            for feature in 0..GATE_UP {
                let reference = format::bf16_to_f32(w4a16[original * GATE_UP + feature]) as f64;
                let actual = format::bf16_to_f32(grouped[sorted * GATE_UP + feature]) as f64;
                let error = actual - reference;
                squared_error += error * error;
                reference_norm += reference * reference;
                max_error = max_error.max(error.abs() as f32);
            }
        }
        let nrmse = (squared_error / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
        if !nrmse.is_finite() || nrmse > 0.08 {
            return Err(eider_cuda::Error::Format {
                label: "Laguna W4A4 gate/up correctness",
                detail: format!("W4A4 versus W4A16 nrmse={nrmse:.6} max_abs_error={max_error:.6}"),
            });
        }
        self.validation_nrmse = nrmse;
        self.validation_max_abs_error = max_error;

        let mut w4a16_sorted = vec![0u16; ROUTES * GATE_UP];
        for (sorted, &original) in sorted_routes.iter().enumerate() {
            let original = original as usize;
            w4a16_sorted[sorted * GATE_UP..(sorted + 1) * GATE_UP]
                .copy_from_slice(&w4a16[original * GATE_UP..(original + 1) * GATE_UP]);
        }
        let w4a16_sorted = DeviceBuffer::from_host(&w4a16_sorted)?;
        let input_scales = DeviceBuffer::from_host(&vec![1.0; EXPERTS])?;
        let gate_up_alphas = DeviceBuffer::from_host(&vec![1.0; EXPERTS])?;
        let tile_len = ROUTES * (INTERMEDIATE / 64) * 512;
        let scale_len = ROUTES * (INTERMEDIATE / 64);
        let mut gathered_tiles = DeviceBuffer::zeroed(tile_len)?;
        let mut gathered_scales = DeviceBuffer::zeroed(scale_len)?;
        let mut sorted_tiles = DeviceBuffer::zeroed(tile_len)?;
        let mut sorted_scales = DeviceBuffer::zeroed(scale_len)?;
        moe_silu_quantize_bf16_sorted_slots_on_stream(
            &self.sorted_routes,
            &self.indices,
            self.w4a16_workspace.output_bf16(),
            &mut gathered_tiles,
            &mut gathered_scales,
            &input_scales,
            &gate_up_alphas,
            INTERMEDIATE,
            ROUTES,
            &self.stream,
        )?;
        moe_silu_quantize_bf16_expert_sorted_slots_on_stream(
            self.sorted_routes.sorted_experts(),
            &w4a16_sorted,
            &mut sorted_tiles,
            &mut sorted_scales,
            &input_scales,
            &gate_up_alphas,
            INTERMEDIATE,
            ROUTES,
            &self.stream,
        )?;
        let gathered_tiles = gathered_tiles.copy_to_host(&self.stream)?.into_vec();
        let sorted_tiles = sorted_tiles.copy_to_host(&self.stream)?.into_vec();
        let gathered_scales = gathered_scales.copy_to_host(&self.stream)?.into_vec();
        let sorted_scales = sorted_scales.copy_to_host(&self.stream)?.into_vec();
        if gathered_tiles != sorted_tiles || gathered_scales != sorted_scales {
            return Err(eider_cuda::Error::Format {
                label: "Laguna expert-sorted SiLU quantization correctness",
                detail: "already-sorted output differs from gathered sorted output".to_string(),
            });
        }

        self.enqueue_w4a16_decode()?;
        let w4a16 = self
            .w4a16
            .output_bf16()
            .copy_to_host(&self.stream)?
            .into_vec();
        let expected = (0..HIDDEN)
            .map(|index| ((index * 17 % 257) as f32 + 1.0) / 512.0)
            .map(format::f32_to_bf16)
            .map(format::bf16_to_f32)
            .sum::<f32>()
            / HIDDEN as f32;
        let expected = format::bf16_to_f32(format::f32_to_bf16(expected));
        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_error = 0.0f32;
        for &actual in &w4a16 {
            let reference = expected as f64;
            let actual = format::bf16_to_f32(actual) as f64;
            let error = actual - reference;
            squared_error += error * error;
            reference_norm += reference * reference;
            max_error = max_error.max(error.abs() as f32);
        }
        let nrmse = (squared_error / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
        if !nrmse.is_finite() || nrmse > 0.005 {
            return Err(eider_cuda::Error::Format {
                label: "SM121 W4A16 decode gate/up correctness",
                detail: format!(
                    "native versus analytic reference nrmse={nrmse:.6} \
                     max_abs_error={max_error:.6}"
                ),
            });
        }
        self.w4a16_reference_nrmse = nrmse;
        self.w4a16_reference_max_abs_error = max_error;

        self.enqueue_grouped_decode()?;
        let grouped = self
            .decode_outputs
            .iter()
            .map(|output| {
                output
                    .data()
                    .copy_to_host(&self.stream)
                    .map(|values| values.into_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_error = 0.0f32;
        for route in 0..TOP_K {
            for feature in 0..GATE_UP {
                let reference = format::bf16_to_f32(w4a16[route * GATE_UP + feature]) as f64;
                let actual = grouped[route][feature] as f64;
                let error = actual - reference;
                squared_error += error * error;
                reference_norm += reference * reference;
                max_error = max_error.max(error.abs() as f32);
            }
        }
        let nrmse = (squared_error / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
        if !nrmse.is_finite() || nrmse > 0.08 {
            return Err(eider_cuda::Error::Format {
                label: "Laguna W4A4 decode gate/up correctness",
                detail: format!("W4A4 versus W4A16 nrmse={nrmse:.6} max_abs_error={max_error:.6}"),
            });
        }
        self.decode_validation_nrmse = nrmse;
        self.decode_validation_max_abs_error = max_error;

        self.enqueue_cutlass_decode()?;
        let cutlass = self
            .decode_cutlass_outputs
            .iter()
            .map(|output| {
                output
                    .data()
                    .copy_to_host(&self.stream)
                    .map(|values| values.into_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_error = 0.0f32;
        for route in 0..TOP_K {
            for feature in 0..GATE_UP {
                let reference = format::bf16_to_f32(w4a16[route * GATE_UP + feature]) as f64;
                let actual = cutlass[route][feature] as f64;
                let error = actual - reference;
                squared_error += error * error;
                reference_norm += reference * reference;
                max_error = max_error.max(error.abs() as f32);
            }
        }
        let nrmse = (squared_error / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
        if !nrmse.is_finite() || nrmse > 0.08 {
            return Err(eider_cuda::Error::Format {
                label: "Laguna CUTLASS W4A4 decode gate/up correctness",
                detail: format!("W4A4 versus W4A16 nrmse={nrmse:.6} max_abs_error={max_error:.6}"),
            });
        }
        self.decode_cutlass_validation_nrmse = nrmse;
        self.decode_cutlass_validation_max_abs_error = max_error;

        self.enqueue_indexed_cutlass_decode()?;
        let indexed_cutlass = self
            .decode_cutlass_outputs
            .iter()
            .map(|output| {
                output
                    .data()
                    .copy_to_host(&self.stream)
                    .map(|values| values.into_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut squared_error = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut max_error = 0.0f32;
        for route in 0..TOP_K {
            for feature in 0..GATE_UP {
                let reference = format::bf16_to_f32(w4a16[route * GATE_UP + feature]) as f64;
                let actual = indexed_cutlass[route][feature] as f64;
                let error = actual - reference;
                squared_error += error * error;
                reference_norm += reference * reference;
                max_error = max_error.max(error.abs() as f32);
            }
        }
        let nrmse = (squared_error / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
        if !nrmse.is_finite() || nrmse > 0.08 {
            return Err(eider_cuda::Error::Format {
                label: "Laguna indexed CUTLASS W4A4 decode gate/up correctness",
                detail: format!("W4A4 versus W4A16 nrmse={nrmse:.6} max_abs_error={max_error:.6}"),
            });
        }
        self.decode_indexed_cutlass_validation_nrmse = nrmse;
        self.decode_indexed_cutlass_validation_max_abs_error = max_error;
        Ok(())
    }

    fn measure(
        &mut self,
        operations: u64,
        enqueue: impl FnOnce(&mut Self) -> Result<()>,
    ) -> BenchSampleResult {
        self.start
            .record_on_stream(&self.stream)
            .expect("record start event");
        enqueue(self).expect("enqueue Laguna gate/up path");
        self.stop
            .record_on_stream(&self.stream)
            .expect("record stop event");
        self.stop.synchronize().expect("synchronize stop event");
        black_box(self.w4a16_workspace.output_bf16().as_const_ptr());
        black_box(self.w4a16.output_bf16().as_const_ptr());
        black_box(self.grouped_output.as_const_ptr());
        black_box(self.decode_output_table.as_const_ptr());
        black_box(self.decode_cutlass_outputs[0].data().as_const_ptr());
        black_box(self.decode_cutlass_output_table.as_const_ptr());
        black_box(self.grouped_weights.len());
        black_box(self.grouped_alphas.as_const_ptr());
        BenchSampleResult::operations(operations)
            .push_metric(MetricValue::new(
                "cuda_event_ms",
                self.start
                    .elapsed_ms_until(&self.stop)
                    .expect("elapsed time") as f64,
                "ms/chunk",
            ))
            .push_metric(MetricValue::new(
                "w4a4_vs_w4a16_nrmse",
                self.validation_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "w4a4_vs_w4a16_max_abs_error",
                self.validation_max_abs_error as f64,
                "value",
            ))
            .push_metric(MetricValue::new(
                "decode_w4a4_vs_w4a16_nrmse",
                self.decode_validation_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "decode_w4a4_vs_w4a16_max_abs_error",
                self.decode_validation_max_abs_error as f64,
                "value",
            ))
            .push_metric(MetricValue::new(
                "decode_cutlass_w4a4_vs_w4a16_nrmse",
                self.decode_cutlass_validation_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "decode_cutlass_w4a4_vs_w4a16_max_abs_error",
                self.decode_cutlass_validation_max_abs_error as f64,
                "value",
            ))
            .push_metric(MetricValue::new(
                "decode_indexed_cutlass_w4a4_vs_w4a16_nrmse",
                self.decode_indexed_cutlass_validation_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "decode_indexed_cutlass_w4a4_vs_w4a16_max_abs_error",
                self.decode_indexed_cutlass_validation_max_abs_error as f64,
                "value",
            ))
            .push_metric(MetricValue::new(
                "w4a16_reference_nrmse",
                self.w4a16_reference_nrmse,
                "ratio",
            ))
            .push_metric(MetricValue::new(
                "w4a16_reference_max_abs_error",
                self.w4a16_reference_max_abs_error as f64,
                "value",
            ))
    }
}

fn synthetic_weights() -> Vec<ModelOptNvfp4Linear> {
    (0..EXPERTS)
        .map(|expert| ModelOptNvfp4Linear {
            prefix: format!("laguna.experts.{expert}.gate_up_proj"),
            out_features: GATE_UP,
            in_features: HIDDEN,
            packed_weight: vec![0x22; GATE_UP * HIDDEN / 2],
            weight_scale: vec![0x38; GATE_UP * HIDDEN / 16],
            weight_scale_2: 1.0 / HIDDEN as f32,
            input_scale: 1.0,
        })
        .collect()
}

fn scalar_pointer_table(values: &mut DeviceBuffer<f32>) -> Result<DeviceBuffer<*mut f32>> {
    let base = values.as_const_ptr().cast::<f32>().cast_mut();
    DeviceBuffer::from_host(
        &(0..values.len())
            .map(|index| unsafe { base.add(index) })
            .collect::<Vec<_>>(),
    )
}

fn w4a16_pipeline_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(ROWS as u64, LagunaRoutedGateUpBench::enqueue_w4a16_pipeline)
}

fn grouped_prepare_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(
        ROWS as u64,
        LagunaRoutedGateUpBench::enqueue_grouped_prepare,
    )
}

fn grouped_gemm_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(ROWS as u64, LagunaRoutedGateUpBench::enqueue_grouped_gemm)
}

fn grouped_pipeline_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(
        ROWS as u64,
        LagunaRoutedGateUpBench::enqueue_grouped_pipeline,
    )
}

fn w4a16_decode_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_w4a16_decode)
}

fn grouped_decode_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_grouped_decode)
}

fn grouped_decode_quantize_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_grouped_decode_quantize)
}

fn grouped_decode_gemv_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_grouped_decode_gemv)
}

fn cutlass_decode_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_cutlass_decode)
}

fn cutlass_decode_gemv_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_cutlass_decode_gemv)
}

fn indexed_cutlass_decode_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(1, LagunaRoutedGateUpBench::enqueue_indexed_cutlass_decode)
}

fn indexed_cutlass_decode_gemv_sample(
    context: &mut LagunaRoutedGateUpBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.measure(
        1,
        LagunaRoutedGateUpBench::enqueue_indexed_cutlass_decode_gemv,
    )
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("laguna-routed-gate-up".to_string()),
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
            runner.group::<LagunaRoutedGateUpBench>("Laguna routed gate/up 1K", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("sm121_w4a16_pipeline", w4a16_pipeline_sample);
                group.bench_sample("grouped_w4a4_pipeline", grouped_pipeline_sample);
                group.bench_sample("grouped_w4a4_prepare", grouped_prepare_sample);
                group.bench_sample("grouped_w4a4_gemm", grouped_gemm_sample);
            });
            runner.group::<LagunaRoutedGateUpBench>("Laguna routed gate/up decode", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("sm121_w4a16_decode", w4a16_decode_sample);
                group.bench_sample("grouped_w4a4_decode", grouped_decode_sample);
                group.bench_sample(
                    "grouped_w4a4_decode_quantize",
                    grouped_decode_quantize_sample,
                );
                group.bench_sample("grouped_w4a4_decode_gemv", grouped_decode_gemv_sample);
                group.bench_sample("cutlass_w4a4_decode", cutlass_decode_sample);
                group.bench_sample("cutlass_w4a4_decode_gemv", cutlass_decode_gemv_sample);
                group.bench_sample("indexed_cutlass_w4a4_decode", indexed_cutlass_decode_sample);
                group.bench_sample(
                    "indexed_cutlass_w4a4_decode_gemv",
                    indexed_cutlass_decode_gemv_sample,
                );
            });
        },
    );
}
