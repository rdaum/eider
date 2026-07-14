mod support;

use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaStream, CutlassFp4GroupedGemvF32Plan, DeviceBuffer, F32Matrix, ModelOptCheckpoint,
    ModelOptNvfp4Linear, MoeSiluQuantizeSlotBuffers, Result, Sm12xFp4DeviceGemmWeight,
    Sm12xFp4GemmWeight, format, indexed_gemv_on_stream, indexed_grouped_gemv_on_stream,
    moe_silu_quantize_slots_nvfp4_simple_scales_on_stream, moe_silu_quantize_slots_on_stream,
    moe_silu_quantize_slots_reference_on_stream, moe_weighted_accumulate_slots_f32_on_stream,
    quantize_fixed_scale_vector_on_stream,
};
use std::path::PathBuf;
use std::time::Duration;
use support::CudaEventBackend;

const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 512;
const TOP_K: usize = 8;
const EXPERTS: usize = 8;
const GATE_UP_OUT: usize = INTERMEDIATE * 2;

struct Nvfp4RoutedMoeShapeBench<const BATCH: usize> {
    stream: CudaStream,
    gate_up: GroupedOp,
    down: GroupedOp,
    sm12x_gate_up: Sm12xOp,
    sm12x_down: Sm12xOp,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    sm12x_input: DeviceBuffer<f32>,
    input_scale_table: DeviceBuffer<f32>,
    gate_up_alpha_table: DeviceBuffer<f32>,
    down_alpha_table: DeviceBuffer<f32>,
    sm12x_reference_tiles: DeviceBuffer<u8>,
    sm12x_reference_scales: DeviceBuffer<u32>,
    reduced: DeviceBuffer<f32>,
}

struct GroupedOp {
    plan: CutlassFp4GroupedGemvF32Plan,
    a_values: DeviceBuffer<*const u8>,
    a_scales: DeviceBuffer<*const u8>,
    b_values: DeviceBuffer<*const u8>,
    b_scales: DeviceBuffer<*const u8>,
    b_values_mut: DeviceBuffer<*mut u8>,
    b_scales_mut: DeviceBuffer<*mut u8>,
    c: DeviceBuffer<*const f32>,
    d: DeviceBuffer<*mut f32>,
    owned_a_values: Vec<DeviceBuffer<u8>>,
    owned_a_scales: Vec<DeviceBuffer<u8>>,
    owned_b_values: Vec<DeviceBuffer<u8>>,
    owned_b_scales: Vec<DeviceBuffer<u8>>,
    contiguous_b_values: DeviceBuffer<u8>,
    contiguous_b_scales: DeviceBuffer<u8>,
    contiguous_output: DeviceBuffer<f32>,
    outputs: Vec<F32Matrix>,
}

struct Sm12xOp {
    m: usize,
    k: usize,
    slots: usize,
    weights: Vec<Sm12xFp4DeviceGemmWeight>,
    a_tiles: DeviceBuffer<*const u8>,
    a_scales: DeviceBuffer<*const u32>,
    b_tiles: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u32>,
    c: DeviceBuffer<*const f32>,
    d: DeviceBuffer<*mut f32>,
    outputs: Vec<F32Matrix>,
}

impl<const BATCH: usize> BenchContext for Nvfp4RoutedMoeShapeBench<BATCH> {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare NVFP4 routed MoE shape benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

impl<const BATCH: usize> Nvfp4RoutedMoeShapeBench<BATCH> {
    fn new() -> Result<Self> {
        let slots = BATCH * TOP_K;
        let weights = QwenLayerWeights::load()?;
        let indices_host = (0..slots)
            .map(|slot| (slot % EXPERTS) as u32)
            .collect::<Vec<_>>();
        let route_weight = 1.0 / TOP_K as f32;
        let mut sm12x_gate_up = Sm12xOp::load_gate_up(GATE_UP_OUT, HIDDEN, slots)?;
        for (slot, output) in sm12x_gate_up.outputs.iter_mut().enumerate() {
            let values = (0..GATE_UP_OUT)
                .map(|idx| (((idx * 17 + slot * 13) % 257) as f32 - 128.0) / 32.0)
                .collect::<Vec<_>>();
            output.data_mut().copy_from_host(&values)?;
        }
        let sm12x_down = Sm12xOp::load_down(HIDDEN, INTERMEDIATE, slots)?;
        let mut bench = Self {
            stream: CudaStream::new_blocking()?,
            gate_up: GroupedOp::new(GATE_UP_OUT, HIDDEN, slots, weights.gate_up)?,
            down: GroupedOp::new(HIDDEN, INTERMEDIATE, slots, weights.down)?,
            sm12x_gate_up,
            sm12x_reference_tiles: DeviceBuffer::zeroed(sm12x_down.b_tiles.len())?,
            sm12x_reference_scales: DeviceBuffer::zeroed(sm12x_down.b_scales.len())?,
            sm12x_down,
            indices: DeviceBuffer::from_host(&indices_host)?,
            route_weights: DeviceBuffer::from_host(&vec![route_weight; slots])?,
            sm12x_input: DeviceBuffer::from_host(
                &(0..HIDDEN)
                    .map(|idx| (((idx * 7) % 17) as f32 - 8.0) * 0.03125)
                    .collect::<Vec<_>>(),
            )?,
            input_scale_table: DeviceBuffer::from_host(&[1.0f32; EXPERTS])?,
            gate_up_alpha_table: DeviceBuffer::from_host(&[1.0f32; EXPERTS])?,
            down_alpha_table: DeviceBuffer::from_host(&[1.0f32; EXPERTS])?,
            reduced: DeviceBuffer::zeroed(HIDDEN)?,
        };
        bench.verify_sm12x_silu_quantizers()?;
        Ok(bench)
    }

    fn run_gate_up_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.gate_up
                .run(&self.stream)
                .expect("gate/up grouped GEMV");
        }
        self.stream.synchronize().expect("sync gate/up bench");
        black_box(self.gate_up.outputs[0].data_ptr());
        black_box(self.gate_up.owned_a_values[0].as_const_ptr());
        black_box(self.gate_up.owned_a_scales[0].as_const_ptr());
    }

    fn run_down_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.down.run(&self.stream).expect("down grouped GEMV");
        }
        self.stream.synchronize().expect("sync down bench");
        black_box(self.down.outputs[0].data_ptr());
        black_box(self.down.owned_b_values[0].as_const_ptr());
        black_box(self.down.owned_b_scales[0].as_const_ptr());
    }

    fn run_gate_up_down_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.gate_up
                .run(&self.stream)
                .expect("gate/up grouped GEMV");
            self.down.run(&self.stream).expect("down grouped GEMV");
        }
        self.stream.synchronize().expect("sync routed bench");
        black_box(self.gate_up.outputs[0].data_ptr());
        black_box(self.down.outputs[0].data_ptr());
    }

    fn run_gate_up_contiguous_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.gate_up
                .run_contiguous(&self.stream)
                .expect("gate/up contiguous grouped GEMV");
        }
        self.stream
            .synchronize()
            .expect("sync gate/up contiguous bench");
        black_box(self.gate_up.contiguous_output.as_const_ptr());
    }

    fn run_down_contiguous_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.down
                .run_contiguous(&self.stream)
                .expect("down contiguous grouped GEMV");
        }
        self.stream
            .synchronize()
            .expect("sync down contiguous bench");
        black_box(self.down.contiguous_output.as_const_ptr());
    }

    fn run_gate_up_down_contiguous_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.gate_up
                .run_contiguous(&self.stream)
                .expect("gate/up contiguous grouped GEMV");
            self.down
                .run_contiguous(&self.stream)
                .expect("down contiguous grouped GEMV");
        }
        self.stream
            .synchronize()
            .expect("sync gate/up down contiguous bench");
        black_box(self.gate_up.contiguous_output.as_const_ptr());
        black_box(self.down.contiguous_output.as_const_ptr());
    }

    fn run_routed_core_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            self.gate_up
                .run(&self.stream)
                .expect("gate/up grouped GEMV");
            moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
                MoeSiluQuantizeSlotBuffers {
                    indices: &self.indices,
                    gate_up_table: &self.gate_up.c,
                    packed_table: self.down.b_values_mut.output(),
                    scales_table: self.down.b_scales_mut.output(),
                    input_scale_table: &self.input_scale_table,
                    gate_up_alpha_table: &self.gate_up_alpha_table,
                },
                INTERMEDIATE,
                &self.stream,
            )
            .expect("SiLU quantize slots");
            self.down.run(&self.stream).expect("down grouped GEMV");
            moe_weighted_accumulate_slots_f32_on_stream(
                &self.indices,
                &self.route_weights,
                &self.down.c,
                &self.down_alpha_table,
                self.reduced.inout(),
                &self.stream,
            )
            .expect("weighted reduce slots");
        }
        self.stream.synchronize().expect("sync routed core bench");
        black_box(self.reduced.as_const_ptr());
    }

    fn run_sm12x_gate_up_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            quantize_fixed_scale_vector_on_stream(
                &self.sm12x_input,
                0.25,
                &mut self.sm12x_gate_up.b_tiles,
                &mut self.sm12x_gate_up.b_scales,
                &self.stream,
            )
            .expect("SM12x quantize gate/up input");
            self.sm12x_gate_up
                .run_shared_input(&self.indices, &self.stream)
                .expect("SM12x gate/up GEMV");
        }
        self.stream.synchronize().expect("sync SM12x gate/up bench");
        black_box(self.sm12x_gate_up.outputs[0].data_ptr());
        black_box(self.sm12x_gate_up.weights[0].tiles_ptr());
    }

    fn run_sm12x_down_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            moe_silu_quantize_slots_on_stream(
                &self.indices,
                &self.sm12x_gate_up.c,
                &mut self.sm12x_down.b_tiles,
                &mut self.sm12x_down.b_scales,
                &self.input_scale_table,
                &self.gate_up_alpha_table,
                INTERMEDIATE,
                self.sm12x_down.slots,
                &self.stream,
            )
            .expect("SM12x SiLU quantize slots");
            self.sm12x_down
                .run_grouped_input(&self.indices, &self.stream)
                .expect("SM12x down GEMV");
        }
        self.stream.synchronize().expect("sync SM12x down bench");
        black_box(self.sm12x_down.outputs[0].data_ptr());
        black_box(self.sm12x_down.weights[0].tiles_ptr());
    }

    fn run_sm12x_silu_quantize_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            moe_silu_quantize_slots_on_stream(
                &self.indices,
                &self.sm12x_gate_up.c,
                &mut self.sm12x_down.b_tiles,
                &mut self.sm12x_down.b_scales,
                &self.input_scale_table,
                &self.gate_up_alpha_table,
                INTERMEDIATE,
                self.sm12x_down.slots,
                &self.stream,
            )
            .expect("parallel SM12x SiLU quantize slots");
        }
        self.stream
            .synchronize()
            .expect("sync parallel SM12x SiLU quantize bench");
        black_box(self.sm12x_down.b_tiles.as_const_ptr());
        black_box(self.sm12x_down.b_scales.as_const_ptr());
    }

    fn run_sm12x_silu_quantize_reference_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            moe_silu_quantize_slots_reference_on_stream(
                &self.indices,
                &self.sm12x_gate_up.c,
                &mut self.sm12x_reference_tiles,
                &mut self.sm12x_reference_scales,
                &self.input_scale_table,
                &self.gate_up_alpha_table,
                INTERMEDIATE,
                self.sm12x_down.slots,
                &self.stream,
            )
            .expect("reference SM12x SiLU quantize slots");
        }
        self.stream
            .synchronize()
            .expect("sync reference SM12x SiLU quantize bench");
        black_box(self.sm12x_reference_tiles.as_const_ptr());
        black_box(self.sm12x_reference_scales.as_const_ptr());
    }

    fn verify_sm12x_silu_quantizers(&mut self) -> Result<()> {
        moe_silu_quantize_slots_reference_on_stream(
            &self.indices,
            &self.sm12x_gate_up.c,
            &mut self.sm12x_reference_tiles,
            &mut self.sm12x_reference_scales,
            &self.input_scale_table,
            &self.gate_up_alpha_table,
            INTERMEDIATE,
            self.sm12x_down.slots,
            &self.stream,
        )?;
        moe_silu_quantize_slots_on_stream(
            &self.indices,
            &self.sm12x_gate_up.c,
            &mut self.sm12x_down.b_tiles,
            &mut self.sm12x_down.b_scales,
            &self.input_scale_table,
            &self.gate_up_alpha_table,
            INTERMEDIATE,
            self.sm12x_down.slots,
            &self.stream,
        )?;
        let reference_tiles = self.sm12x_reference_tiles.copy_to_host(&self.stream)?;
        let candidate_tiles = self.sm12x_down.b_tiles.copy_to_host(&self.stream)?;
        assert_eq!(
            candidate_tiles.into_vec(),
            reference_tiles.into_vec(),
            "parallel SM12x quantizer changed native tile bytes"
        );
        let reference_scales = self.sm12x_reference_scales.copy_to_host(&self.stream)?;
        let candidate_scales = self.sm12x_down.b_scales.copy_to_host(&self.stream)?;
        assert_eq!(
            candidate_scales.into_vec(),
            reference_scales.into_vec(),
            "parallel SM12x quantizer changed scale words"
        );
        Ok(())
    }

    fn run_sm12x_routed_core_chunk(&mut self, chunk_size: usize) {
        for _ in 0..chunk_size {
            quantize_fixed_scale_vector_on_stream(
                &self.sm12x_input,
                0.25,
                &mut self.sm12x_gate_up.b_tiles,
                &mut self.sm12x_gate_up.b_scales,
                &self.stream,
            )
            .expect("SM12x quantize gate/up input");
            self.sm12x_gate_up
                .run_shared_input(&self.indices, &self.stream)
                .expect("SM12x gate/up GEMV");
            moe_silu_quantize_slots_on_stream(
                &self.indices,
                &self.sm12x_gate_up.c,
                &mut self.sm12x_down.b_tiles,
                &mut self.sm12x_down.b_scales,
                &self.input_scale_table,
                &self.gate_up_alpha_table,
                INTERMEDIATE,
                self.sm12x_down.slots,
                &self.stream,
            )
            .expect("SM12x SiLU quantize slots");
            self.sm12x_down
                .run_grouped_input(&self.indices, &self.stream)
                .expect("SM12x down GEMV");
            moe_weighted_accumulate_slots_f32_on_stream(
                &self.indices,
                &self.route_weights,
                &self.sm12x_down.c,
                &self.down_alpha_table,
                self.reduced.inout(),
                &self.stream,
            )
            .expect("weighted reduce SM12x slots");
        }
        self.stream
            .synchronize()
            .expect("sync SM12x routed core bench");
        black_box(self.reduced.as_const_ptr());
    }
}

impl GroupedOp {
    fn new(
        m: usize,
        k: usize,
        slots: usize,
        weights: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
    ) -> Result<Self> {
        let plan = CutlassFp4GroupedGemvF32Plan::new(m, k, slots)?;
        let mut owned_a_values = Vec::with_capacity(weights.len());
        let mut owned_a_scales = Vec::with_capacity(weights.len());
        for (values, scales) in weights {
            owned_a_values.push(values);
            owned_a_scales.push(scales);
        }

        let mut owned_b_values = Vec::with_capacity(slots);
        let mut owned_b_scales = Vec::with_capacity(slots);
        let mut contiguous_b_values_host = Vec::with_capacity(slots * (k / 2));
        let mut contiguous_b_scales_host = Vec::with_capacity(slots * (k / 16));
        for slot in 0..slots {
            let (values, scales) = synthetic_vector(k, slot)?;
            contiguous_b_values_host
                .extend_from_slice(&values.copy_to_host(&CudaStream::new_blocking()?)?);
            contiguous_b_scales_host
                .extend_from_slice(&scales.copy_to_host(&CudaStream::new_blocking()?)?);
            owned_b_values.push(values);
            owned_b_scales.push(scales);
        }

        let a_value_ptrs = (0..slots)
            .map(|slot| {
                owned_a_values[slot % owned_a_values.len()]
                    .as_const_ptr()
                    .cast::<u8>()
            })
            .collect::<Vec<_>>();
        let a_scale_ptrs = (0..slots)
            .map(|slot| {
                owned_a_scales[slot % owned_a_scales.len()]
                    .as_const_ptr()
                    .cast::<u8>()
            })
            .collect::<Vec<_>>();
        let b_value_ptrs = owned_b_values
            .iter()
            .map(|buffer| buffer.as_const_ptr().cast::<u8>())
            .collect::<Vec<_>>();
        let b_scale_ptrs = owned_b_scales
            .iter()
            .map(|buffer| buffer.as_const_ptr().cast::<u8>())
            .collect::<Vec<_>>();
        let b_value_mut_ptrs = owned_b_values
            .iter()
            .map(|buffer| buffer.as_const_ptr().cast::<u8>().cast_mut())
            .collect::<Vec<_>>();
        let b_scale_mut_ptrs = owned_b_scales
            .iter()
            .map(|buffer| buffer.as_const_ptr().cast::<u8>().cast_mut())
            .collect::<Vec<_>>();

        let mut outputs = (0..slots)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()?;
        let c_ptrs = outputs
            .iter()
            .map(|output| output.data_ptr())
            .collect::<Vec<_>>();
        let mut d_ptrs = Vec::with_capacity(slots);
        for output in &mut outputs {
            d_ptrs.push(output.data_mut_ptr().cast());
        }

        Ok(Self {
            plan,
            a_values: DeviceBuffer::from_host(&a_value_ptrs)?,
            a_scales: DeviceBuffer::from_host(&a_scale_ptrs)?,
            b_values: DeviceBuffer::from_host(&b_value_ptrs)?,
            b_scales: DeviceBuffer::from_host(&b_scale_ptrs)?,
            b_values_mut: DeviceBuffer::from_host(&b_value_mut_ptrs)?,
            b_scales_mut: DeviceBuffer::from_host(&b_scale_mut_ptrs)?,
            c: DeviceBuffer::from_host(&c_ptrs)?,
            d: DeviceBuffer::from_host(&d_ptrs)?,
            owned_a_values,
            owned_a_scales,
            owned_b_values,
            owned_b_scales,
            contiguous_b_values: DeviceBuffer::from_host(&contiguous_b_values_host)?,
            contiguous_b_scales: DeviceBuffer::from_host(&contiguous_b_scales_host)?,
            contiguous_output: DeviceBuffer::zeroed(slots * m)?,
            outputs,
        })
    }

    fn run(&self, stream: &CudaStream) -> Result<()> {
        self.plan.run_on_stream(
            &self.a_values,
            &self.a_scales,
            &self.b_values,
            &self.b_scales,
            &self.c,
            &self.d,
            1.0,
            0.0,
            stream,
        )
    }

    fn run_contiguous(&mut self, stream: &CudaStream) -> Result<()> {
        self.plan.run_contiguous_b_on_stream(
            &self.a_values,
            &self.a_scales,
            &self.contiguous_b_values,
            &self.contiguous_b_scales,
            &mut self.contiguous_output,
            1.0,
            stream,
        )
    }
}

impl Sm12xOp {
    fn load_gate_up(m: usize, k: usize, slots: usize) -> Result<Self> {
        Self::load(m, k, slots, true)
    }

    fn load_down(m: usize, k: usize, slots: usize) -> Result<Self> {
        Self::load(m, k, slots, false)
    }

    fn load(m: usize, k: usize, slots: usize, gate_up: bool) -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let prefix = "model.language_model.layers.0.mlp";
        let mut weights = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let expert_prefix = format!("{prefix}.experts.{expert}");
            let linear = if gate_up {
                let gate = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.gate_proj"))?;
                let up = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.up_proj"))?;
                ModelOptNvfp4Linear::concat_out_features(
                    format!("{expert_prefix}.gate_up_proj"),
                    &gate,
                    &up,
                )?
            } else {
                checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.down_proj"))?
            };
            weights.push(sm12x_requantized_device_weight(&linear)?);
        }
        let a_tiles = DeviceBuffer::from_host(
            &weights
                .iter()
                .map(|weight| weight.tiles_ptr())
                .collect::<Vec<_>>(),
        )?;
        let a_scales = DeviceBuffer::from_host(
            &weights
                .iter()
                .map(|weight| weight.scales_ptr())
                .collect::<Vec<_>>(),
        )?;
        let mut outputs = (0..slots)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()?;
        let c_ptrs = outputs
            .iter()
            .map(|output| output.data_ptr())
            .collect::<Vec<_>>();
        let d_ptrs = outputs
            .iter_mut()
            .map(|output| output.data_mut_ptr())
            .collect::<Vec<_>>();
        let k_tiles = k / 64;
        let b_groups = if gate_up { 1 } else { slots };
        Ok(Self {
            m,
            k,
            slots,
            weights,
            a_tiles,
            a_scales,
            b_tiles: DeviceBuffer::zeroed(b_groups * k_tiles * 512)?,
            b_scales: DeviceBuffer::zeroed(b_groups * k_tiles)?,
            c: DeviceBuffer::from_host(&c_ptrs)?,
            d: DeviceBuffer::from_host(&d_ptrs)?,
            outputs,
        })
    }

    fn run_shared_input(&self, indices: &DeviceBuffer<u32>, stream: &CudaStream) -> Result<()> {
        indexed_gemv_on_stream(
            indices,
            &self.a_tiles,
            &self.a_scales,
            self.weights.len(),
            &self.b_tiles,
            &self.b_scales,
            &self.d,
            self.m / 16,
            self.k / 64,
            self.slots,
            stream,
        )
    }

    fn run_grouped_input(&self, indices: &DeviceBuffer<u32>, stream: &CudaStream) -> Result<()> {
        indexed_grouped_gemv_on_stream(
            indices,
            &self.a_tiles,
            &self.a_scales,
            self.weights.len(),
            &self.b_tiles,
            &self.b_scales,
            &self.d,
            self.m / 16,
            self.k / 64,
            self.slots,
            stream,
        )
    }
}

fn sm12x_requantized_device_weight(
    linear: &ModelOptNvfp4Linear,
) -> Result<Sm12xFp4DeviceGemmWeight> {
    let dequant = linear.dequantize_to_f32_col_major();
    let mut row_major = vec![0.0f32; linear.out_features * linear.in_features];
    for row in 0..linear.out_features {
        for col in 0..linear.in_features {
            row_major[row * linear.in_features + col] = dequant[col + row * linear.in_features];
        }
    }
    Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
        linear.out_features,
        linear.in_features,
        &row_major,
    )?
    .weight
    .to_device()
}

struct QwenLayerWeights {
    gate_up: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
    down: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
}

impl QwenLayerWeights {
    fn load() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let prefix = "model.language_model.layers.0.mlp";
        let mut gate_up = Vec::with_capacity(EXPERTS);
        let mut down = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let expert_prefix = format!("{prefix}.experts.{expert}");
            let gate = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.gate_proj"))?;
            let up = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.up_proj"))?;
            let gate_up_host = ModelOptNvfp4Linear::concat_out_features(
                format!("{expert_prefix}.gate_up_proj"),
                &gate,
                &up,
            )?;
            gate_up.push(gate_up_host.to_grouped_device()?);
            down.push(
                checkpoint
                    .load_nvfp4_linear(&format!("{expert_prefix}.down_proj"))?
                    .to_grouped_device()?,
            );
        }
        Ok(Self { gate_up, down })
    }
}

fn model_dir() -> PathBuf {
    std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.6-35b-a3-nvfp4")
        })
}

fn synthetic_vector(k: usize, slot: usize) -> Result<(DeviceBuffer<u8>, DeviceBuffer<u8>)> {
    let values = (0..k)
        .map(|idx| if (idx + slot) & 1 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    Ok((
        DeviceBuffer::from_host(&format::pack_e2m1(&values))?,
        DeviceBuffer::from_host(&vec![0x38; k / 16])?,
    ))
}

fn bytes_per_gate_up(slots: usize) -> u64 {
    (slots
        * (GATE_UP_OUT * HIDDEN / 2
            + HIDDEN / 2
            + GATE_UP_OUT * (HIDDEN / 16)
            + HIDDEN / 16
            + GATE_UP_OUT * 4)) as u64
}

fn bytes_per_down(slots: usize) -> u64 {
    (slots
        * (HIDDEN * INTERMEDIATE / 2
            + INTERMEDIATE / 2
            + HIDDEN * (INTERMEDIATE / 16)
            + INTERMEDIATE / 16
            + HIDDEN * 4)) as u64
}

fn bytes_per_silu_quantize(slots: usize) -> u64 {
    (slots * (GATE_UP_OUT * 4 + INTERMEDIATE / 2 + INTERMEDIATE / 16)) as u64
}

fn flops_per_gate_up(slots: usize) -> u64 {
    (2 * slots * GATE_UP_OUT * HIDDEN) as u64
}

fn flops_per_down(slots: usize) -> u64 {
    (2 * slots * HIDDEN * INTERMEDIATE) as u64
}

fn sample_metrics<const BATCH: usize>(chunk_size: usize) -> BenchSampleResult {
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::integer("batch", BATCH as i64, "tokens"))
        .push_metric(MetricValue::integer(
            "slots",
            (BATCH * TOP_K) as i64,
            "slots",
        ))
        .push_metric(MetricValue::integer("top_k", TOP_K as i64, "experts"))
}

fn gate_up_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_gate_up_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn gate_up_contiguous_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_gate_up_contiguous_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn down_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_down_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn down_contiguous_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_down_contiguous_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn gate_up_down_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_gate_up_down_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn gate_up_down_contiguous_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_gate_up_down_contiguous_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn routed_core_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_routed_core_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn sm12x_gate_up_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_sm12x_gate_up_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn sm12x_down_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_sm12x_down_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn sm12x_silu_quantize_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_sm12x_silu_quantize_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn sm12x_silu_quantize_reference_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_sm12x_silu_quantize_reference_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn sm12x_routed_core_sample<const BATCH: usize>(
    ctx: &mut Nvfp4RoutedMoeShapeBench<BATCH>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.run_sm12x_routed_core_chunk(chunk_size);
    sample_metrics::<BATCH>(chunk_size)
}

fn register_batch<const BATCH: usize>(runner: &micromeasure::BenchmarkRunner) {
    let slots = BATCH * TOP_K;
    runner.group::<Nvfp4RoutedMoeShapeBench<BATCH>>("NVFP4 routed MoE vLLM shape", |g| {
        g.throughput(Throughput::bytes(bytes_per_gate_up(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || {
                Box::new(CudaEventBackend::new(
                    bytes_per_gate_up(slots),
                    flops_per_gate_up(slots),
                ))
            })
            .bench_sample(
                &format!("gate_up_batch{BATCH}_slots{slots}_m1024_k2048"),
                gate_up_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(bytes_per_gate_up(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || {
                Box::new(CudaEventBackend::new(
                    bytes_per_gate_up(slots),
                    flops_per_gate_up(slots),
                ))
            })
            .bench_sample(
                &format!("gate_up_contiguous_batch{BATCH}_slots{slots}_m1024_k2048"),
                gate_up_contiguous_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(bytes_per_down(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || {
                Box::new(CudaEventBackend::new(
                    bytes_per_down(slots),
                    flops_per_down(slots),
                ))
            })
            .bench_sample(
                &format!("down_batch{BATCH}_slots{slots}_m2048_k512"),
                down_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(bytes_per_down(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || {
                Box::new(CudaEventBackend::new(
                    bytes_per_down(slots),
                    flops_per_down(slots),
                ))
            })
            .bench_sample(
                &format!("down_contiguous_batch{BATCH}_slots{slots}_m2048_k512"),
                down_contiguous_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(
            bytes_per_gate_up(slots) + bytes_per_down(slots),
        ))
        .measurement_domain(MeasurementDomain::Gpu)
        .backend(move || {
            Box::new(CudaEventBackend::new(
                bytes_per_gate_up(slots) + bytes_per_down(slots),
                flops_per_gate_up(slots) + flops_per_down(slots),
            ))
        })
        .bench_sample(
            &format!("gate_up_down_batch{BATCH}_slots{slots}"),
            gate_up_down_sample::<BATCH>,
        );
        g.throughput(Throughput::bytes(
            bytes_per_gate_up(slots) + bytes_per_down(slots),
        ))
        .measurement_domain(MeasurementDomain::Gpu)
        .backend(move || {
            Box::new(CudaEventBackend::new(
                bytes_per_gate_up(slots) + bytes_per_down(slots),
                flops_per_gate_up(slots) + flops_per_down(slots),
            ))
        })
        .bench_sample(
            &format!("gate_up_down_contiguous_batch{BATCH}_slots{slots}"),
            gate_up_down_contiguous_sample::<BATCH>,
        );
        g.throughput(Throughput::bytes(
            bytes_per_gate_up(slots) + bytes_per_down(slots),
        ))
        .measurement_domain(MeasurementDomain::Gpu)
        .backend(move || {
            Box::new(CudaEventBackend::new(
                bytes_per_gate_up(slots) + bytes_per_down(slots),
                flops_per_gate_up(slots) + flops_per_down(slots),
            ))
        })
        .bench_sample(
            &format!("routed_core_batch{BATCH}_slots{slots}"),
            routed_core_sample::<BATCH>,
        );
        g.throughput(Throughput::bytes(bytes_per_gate_up(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || {
                Box::new(CudaEventBackend::new(
                    bytes_per_gate_up(slots),
                    flops_per_gate_up(slots),
                ))
            })
            .bench_sample(
                &format!("sm12x_gate_up_batch{BATCH}_slots{slots}_m1024_k2048"),
                sm12x_gate_up_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(bytes_per_down(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || {
                Box::new(CudaEventBackend::new(
                    bytes_per_down(slots),
                    flops_per_down(slots),
                ))
            })
            .bench_sample(
                &format!("sm12x_down_batch{BATCH}_slots{slots}_m2048_k512"),
                sm12x_down_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(bytes_per_silu_quantize(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || Box::new(CudaEventBackend::new(bytes_per_silu_quantize(slots), 0)))
            .bench_sample(
                &format!("sm12x_silu_quantize_parallel_batch{BATCH}_slots{slots}_k512"),
                sm12x_silu_quantize_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(bytes_per_silu_quantize(slots)))
            .measurement_domain(MeasurementDomain::Gpu)
            .backend(move || Box::new(CudaEventBackend::new(bytes_per_silu_quantize(slots), 0)))
            .bench_sample(
                &format!("sm12x_silu_quantize_reference_batch{BATCH}_slots{slots}_k512"),
                sm12x_silu_quantize_reference_sample::<BATCH>,
            );
        g.throughput(Throughput::bytes(
            bytes_per_gate_up(slots) + bytes_per_down(slots),
        ))
        .measurement_domain(MeasurementDomain::Gpu)
        .backend(move || {
            Box::new(CudaEventBackend::new(
                bytes_per_gate_up(slots) + bytes_per_down(slots),
                flops_per_gate_up(slots) + flops_per_down(slots),
            ))
        })
        .bench_sample(
            &format!("sm12x_routed_core_batch{BATCH}_slots{slots}"),
            sm12x_routed_core_sample::<BATCH>,
        );
    });
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-routed-moe-shape".to_string()),
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
        register_batch::<1>(runner);
        register_batch::<2>(runner);
        register_batch::<4>(runner);
        register_batch::<8>(runner);
        register_batch::<16>(runner);
    });
}
