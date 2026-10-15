//! LFM2 projections, gated short convolution, routing, and head-64 attention.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::{
    Bf16TnMatmulPlan, CublasLt, DeviceAddress, Error, Fp8TnMatmulPlan, GemmShape, MoeSortedRoutes,
    Result,
};
use std::collections::HashMap;
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_expand_dense_bf16_on_stream(
        weight: *const u8,
        scales: *const u8,
        output: *mut u16,
        elements: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_quantized_dense_gemm_f32_on_stream(
        input: *const f32,
        weight: *const u8,
        block_scales: *const u8,
        channel_scales: *const f32,
        global_scale: f32,
        output: *mut f32,
        rows: u32,
        out_features: u32,
        in_features: u32,
        format: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_split_bf16_on_stream(
        input: *const f32,
        parts: *mut u16,
        count: u32,
        stride: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_sum_parts_on_stream(
        parts: *const f32,
        output: *mut f32,
        count: u32,
        stride: u32,
        channels: *const f32,
        scale: f32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Projects F32 rows through compressed FP8 or NVFP4 weights using BF16 tensor
/// cores. NVFP4 supplies K16 block scales; FP8 supplies optional channel scales.
/// Three BF16 components retain the F32 input mantissa without requantization.
#[allow(clippy::too_many_arguments)]
pub fn quantized_dense_gemm_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    block_scales: Option<&DeviceBuffer<u8>>,
    channel_scales: Option<&DeviceBuffer<f32>>,
    global_scale: f32,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let elements = out_features.checked_mul(in_features);
    let divisor = if block_scales.is_some() { 2 } else { 1 };
    if rows == 0
        || rows > 65535 * 8
        || out_features == 0
        || out_features > u32::MAX as usize - 63
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
        || elements.is_none_or(|n| n / divisor != weight.len())
        || block_scales.is_some_and(|s| elements.is_none_or(|n| n / 16 != s.len()))
        || channel_scales.is_some_and(|s| s.len() != out_features || block_scales.is_some())
        || rows
            .checked_mul(in_features)
            .is_none_or(|n| n > input.len())
        || rows
            .checked_mul(out_features)
            .is_none_or(|n| n > output.len())
    {
        return Err(Error::Format {
            label: "quantized F32 dense GEMM",
            detail: "expected matching FP8 or NVFP4 buffers and positive dimensions with K divisible by 16".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_quantized_dense_gemm_f32_on_stream",
            infer_quantized_dense_gemm_f32_on_stream(
                input.ptr,
                weight.ptr,
                block_scales.map_or(std::ptr::null(), |s| s.ptr),
                channel_scales.map_or(std::ptr::null(), |s| s.ptr),
                global_scale,
                output.buffer_mut().ptr,
                rows as u32,
                out_features as u32,
                in_features as u32,
                if block_scales.is_some() { 2 } else { 1 },
                stream.as_raw(),
            ),
        )
    }
}

/// Reusable projections from F32 rows, with BF16 decomposition or calibrated FP8 activations.
pub struct F32DenseGemm {
    lt: CublasLt,
    plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    fp8_plans: HashMap<(usize, usize, usize), Fp8TnMatmulPlan>,
    fp8_input: DeviceBuffer<u8>,
    input: DeviceBuffer<u16>,
    products: DeviceBuffer<f32>,
    expanded_weight: DeviceBuffer<u16>,
    capacity: usize,
    features: usize,
}

enum DenseGemmWeights<'a> {
    Bf16(&'a DeviceBuffer<u16>),
    Quantized(&'a DeviceBuffer<u8>),
}

impl F32DenseGemm {
    /// Allocates scratch for the largest input/output width and prompt chunk.
    /// Set max_quantized_elements to zero for BF16-only projections.
    pub fn new(capacity: usize, features: usize, max_quantized_elements: usize) -> Result<Self> {
        let count = capacity
            .checked_next_power_of_two()
            .and_then(|rows| rows.checked_mul(features))
            .and_then(|n| n.checked_mul(3));
        if capacity == 0
            || features == 0
            || max_quantized_elements > u32::MAX as usize
            || count.is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Format {
                label: "F32 dense GEMM",
                detail: "invalid workspace dimensions".into(),
            });
        }
        Ok(Self {
            lt: CublasLt::new()?,
            plans: HashMap::new(),
            fp8_plans: HashMap::new(),
            fp8_input: DeviceBuffer::zeroed(count.unwrap() / 3)?,
            input: DeviceBuffer::zeroed(count.unwrap())?,
            products: DeviceBuffer::zeroed(count.unwrap())?,
            expanded_weight: DeviceBuffer::zeroed(max_quantized_elements)?,
            capacity,
            features,
        })
    }

    /// Uses a checkpoint's calibrated FP8 activation scale and resident FP8 weights.
    /// Plans use the exact row count; callers retain one workspace per batch shape.
    #[allow(clippy::too_many_arguments)]
    pub fn run_calibrated_fp8_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        weight_scale: f32,
        input_scale: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0
            || rows > self.capacity
            || out == 0
            || cols == 0
            || out > self.features
            || cols > self.features
            || input.len() < rows * cols
            || output.len() < rows * out
            || weight.len() != out * cols
            || !input_scale.is_finite()
            || input_scale <= 0.
            || !weight_scale.is_finite()
        {
            return Err(Error::Format {
                label: "calibrated FP8 dense GEMM",
                detail: "invalid shape, buffers, or scales".into(),
            });
        }
        let key = (out, cols, rows);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.fp8_plans.entry(key) {
            entry.insert(Fp8TnMatmulPlan::new(
                &self.lt,
                GemmShape {
                    m: out,
                    n: rows,
                    k: cols,
                },
                1 << 20,
            )?);
        }
        unsafe {
            check_cuda(
                "infer_quantize_fp8_e4m3_f32_on_stream",
                crate::ffi::infer_quantize_fp8_e4m3_f32_on_stream(
                    input.ptr,
                    self.fp8_input.ptr,
                    (rows * cols) as u32,
                    input_scale,
                    stream.as_raw(),
                ),
            )?;
        }
        self.fp8_plans[&key].run_with_alpha_on_stream(
            &self.lt,
            weight,
            &self.fp8_input,
            output,
            weight_scale * input_scale,
            stream,
        )
    }

    /// Projects active rows, preserving F32 activation residuals in three products.
    /// Plans are cached by matrix shape and power-of-two row count. The first
    /// call for each shape creates its plan outside CUDA graph capture.
    #[allow(clippy::too_many_arguments)]
    pub fn run_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_impl(
            input,
            DenseGemmWeights::Bf16(weight),
            None,
            None,
            1.0,
            output,
            rows,
            out,
            cols,
            stream,
        )
    }

    /// Expands compressed weights into shared scratch, then projects F32 rows.
    /// Weight expansion, activation splitting, and scaling run on the same stream.
    #[allow(clippy::too_many_arguments)]
    pub fn run_quantized_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        block_scales: Option<&DeviceBuffer<u8>>,
        channel_scales: Option<&DeviceBuffer<f32>>,
        global_scale: f32,
        output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let elements = out.checked_mul(cols);
        let divisor = if block_scales.is_some() { 2 } else { 1 };
        if !cols.is_multiple_of(16)
            || elements
                .is_none_or(|n| n > self.expanded_weight.len() || n / divisor != weight.len())
            || block_scales.is_some_and(|s| elements.is_none_or(|n| n / 16 != s.len()))
            || channel_scales.is_some_and(|s| s.len() != out || block_scales.is_some())
            || !global_scale.is_finite()
        {
            return Err(Error::Format {
                label: "F32 dense GEMM",
                detail: "invalid compressed weight dimensions or scales".into(),
            });
        }
        self.run_impl(
            input,
            DenseGemmWeights::Quantized(weight),
            block_scales,
            channel_scales,
            global_scale,
            output,
            rows,
            out,
            cols,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_impl(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: DenseGemmWeights<'_>,
        block_scales: Option<&DeviceBuffer<u8>>,
        channel_scales: Option<&DeviceBuffer<f32>>,
        global_scale: f32,
        mut output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0
            || rows > self.capacity
            || out == 0
            || cols == 0
            || out > self.features
            || cols > self.features
            || input.len() < rows * cols
            || output.len() < rows * out
            || matches!(weight, DenseGemmWeights::Bf16(w) if out.checked_mul(cols) != Some(w.len()))
        {
            return Err(Error::Format {
                label: "F32 dense GEMM",
                detail: "invalid projection dimensions or buffers".into(),
            });
        }
        // Power-of-two row buckets bound the plan cache across prompt lengths.
        let padded_rows = rows.next_power_of_two();
        let shape = (out, cols, padded_rows);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.plans.entry(shape) {
            entry.insert(Bf16TnMatmulPlan::new(
                &self.lt,
                GemmShape {
                    m: out,
                    n: 3 * padded_rows,
                    k: cols,
                },
                1 << 20,
            )?);
        }
        let plan = &self.plans[&shape];
        unsafe {
            check_cuda(
                "infer_lfm2_split_bf16_on_stream",
                infer_lfm2_split_bf16_on_stream(
                    input.ptr,
                    self.input.ptr,
                    (rows * cols) as u32,
                    (padded_rows * cols) as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        let weight = match weight {
            DenseGemmWeights::Bf16(weight) => weight,
            DenseGemmWeights::Quantized(weight) => {
                unsafe {
                    check_cuda(
                        "infer_expand_dense_bf16_on_stream",
                        infer_expand_dense_bf16_on_stream(
                            weight.ptr,
                            block_scales.map_or(std::ptr::null(), |s| s.ptr),
                            self.expanded_weight.ptr,
                            (out * cols) as u32,
                            stream.as_raw(),
                        ),
                    )?;
                }
                &self.expanded_weight
            }
        };
        plan.run_on_stream(
            &self.lt,
            weight,
            &self.input,
            self.products.output(),
            stream,
        )?;

        unsafe {
            check_cuda(
                "infer_lfm2_sum_parts_on_stream",
                infer_lfm2_sum_parts_on_stream(
                    self.products.ptr,
                    output.buffer_mut().ptr,
                    (rows * out) as u32,
                    (padded_rows * out) as u32,
                    channel_scales.map_or(std::ptr::null(), |s| s.ptr),
                    global_scale,
                    out as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
    /// Returns scratch and cached plan workspace bytes on the device.
    pub fn device_bytes(&self) -> usize {
        self.input.device_bytes()
            + self.fp8_input.device_bytes()
            + self
                .fp8_plans
                .values()
                .map(Fp8TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
            + self.products.device_bytes()
            + self.expanded_weight.device_bytes()
            + self
                .plans
                .values()
                .map(Bf16TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
    }
}

unsafe extern "C" {
    fn infer_lfm2_short_conv_on_stream(
        bcx: *const f32,
        weight: *const f32,
        state: *mut f32,
        output: *mut f32,
        hidden: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_router_on_stream(
        logits: *const f32,
        bias: *const f32,
        indices: *mut u32,
        weights: *mut f32,
        experts: u32,
        top_k: u32,
        normalize: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Applies `C * causal_conv(B * x)` and advances channel-major history.
/// The projection stores contiguous B, C, and x vectors; weights are `[H,L]`.
pub fn lfm2_short_conv_into_on_stream(
    bcx: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hidden = output.len();
    if hidden == 0
        || hidden > u32::MAX as usize
        || bcx.len() > u32::MAX as usize
        || weight.len() > u32::MAX as usize
        || width == 0
        || width > u32::MAX as usize
        || hidden.checked_mul(3) != Some(bcx.len())
        || hidden.checked_mul(width) != Some(weight.len())
        || hidden.checked_mul(width - 1) != Some(state.len())
    {
        return Err(Error::Format {
            label: "LFM2 short convolution",
            detail: "expected B/C/x [3H], weight [H,L], state [H,L-1], output [H]".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_short_conv_on_stream",
            infer_lfm2_short_conv_on_stream(
                bcx.ptr,
                weight.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                hidden as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Routes by sigmoid plus bias, returning unbiased, optionally normalized weights.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_router_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut indices: DeviceOutput<'_, u32>,
    mut weights: DeviceOutput<'_, f32>,
    normalize: bool,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let experts = logits.len();
    let top_k = indices.len();
    if experts == 0
        || experts > 512
        || bias.len() != experts
        || top_k == 0
        || top_k > experts
        || weights.len() != top_k
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Format {
            label: "LFM2 router",
            detail: "invalid expert tables, top-k or routed scale".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_router_on_stream",
            infer_lfm2_router_on_stream(
                logits.ptr,
                bias.ptr,
                indices.buffer_mut().ptr,
                weights.buffer_mut().ptr,
                experts as u32,
                top_k as u32,
                i32::from(normalize),
                scale,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_short_conv_batch_on_stream(
        bcx: *const f32,
        weight: *const f32,
        state: *mut f32,
        output: *mut f32,
        hidden: u32,
        width: u32,
        rows: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_router_batch_on_stream(
        logits: *const f32,
        bias: *const f32,
        indices: *mut u32,
        weights: *mut f32,
        experts: u32,
        top_k: u32,
        rows: u32,
        normalize: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_decode_inputs_on_stream(
        token: u32,
        position: u32,
        token_id: *mut u32,
        start: *mut u32,
        length: *mut u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Applies a chunk of short convolution and commits its final history.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_short_conv_batch_into_on_stream(
    bcx: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    hidden: usize,
    width: usize,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    if hidden == 0
        || width == 0
        || rows == 0
        || rows > 65535
        || hidden.checked_mul(width) != Some(weight.len())
        || hidden.checked_mul(width - 1) != Some(state.len())
        || rows
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(3))
            .is_none_or(|n| n > bcx.len())
        || rows.checked_mul(hidden).is_none_or(|n| n > output.len())
        || weight.len() > u32::MAX as usize
        || hidden > u32::MAX as usize / 3
    {
        return Err(Error::Format {
            label: "LFM2 batched convolution",
            detail: "invalid dimensions or buffers".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_short_conv_batch_on_stream",
            infer_lfm2_short_conv_batch_on_stream(
                bcx.ptr,
                weight.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                hidden as u32,
                width as u32,
                rows as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Routes each row independently, retaining unbiased sigmoid weights.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_router_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut indices: DeviceOutput<'_, u32>,
    mut weights: DeviceOutput<'_, f32>,
    rows: usize,
    top_k: usize,
    normalize: bool,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let experts = bias.len();
    if experts == 0
        || experts > 512
        || rows == 0
        || rows > u32::MAX as usize
        || top_k == 0
        || top_k > experts
        || !scale.is_finite()
        || scale <= 0.0
        || rows.checked_mul(experts).is_none_or(|n| n > logits.len())
        || rows
            .checked_mul(top_k)
            .is_none_or(|n| n > indices.len() || n > weights.len())
    {
        return Err(Error::Format {
            label: "LFM2 batched router",
            detail: "invalid dimensions or buffers".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_router_batch_on_stream",
            infer_lfm2_router_batch_on_stream(
                logits.ptr,
                bias.ptr,
                indices.buffer_mut().ptr,
                weights.buffer_mut().ptr,
                experts as u32,
                top_k as u32,
                rows as u32,
                i32::from(normalize),
                scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Updates graph inputs on the execution stream without a host-memory transfer.
pub fn lfm2_decode_inputs_on_stream(
    token: u32,
    position: u32,
    mut token_id: DeviceOutput<'_, u32>,
    mut start: DeviceOutput<'_, u32>,
    mut length: DeviceOutput<'_, u32>,
    stream: &CudaStream,
) -> Result<()> {
    if token_id.len() != 1 || start.len() != 1 || length.len() != 1 || position == u32::MAX {
        return Err(Error::Format {
            label: "LFM2 decode inputs",
            detail: "expected scalar token, position and length".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_decode_inputs_on_stream",
            infer_lfm2_decode_inputs_on_stream(
                token,
                position,
                token_id.buffer_mut().ptr,
                start.buffer_mut().ptr,
                length.buffer_mut().ptr,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_expert_gemm_on_stream(
        sorted_routes: *const u32,
        offsets: *const u32,
        inputs: *const *const f32,
        weights: *const *const u8,
        scales: *const *const u8,
        globals: *const f32,
        outputs: *const *mut f32,
        routes: u32,
        experts: u32,
        out_features: u32,
        in_features: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_expert_gemm_pipelined_on_stream(
        sorted_routes: *const u32,
        offsets: *const u32,
        inputs: *const *const f32,
        weights: *const *const u8,
        scales: *const *const u8,
        globals: *const f32,
        outputs: *const *mut f32,
        routes: u32,
        experts: u32,
        out_features: u32,
        in_features: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Groups sixteen routes per tensor-core tile, retaining F32 activation precision.
/// Weight tables contain row-major ModelOpt NVFP4 and K16 UE4M3 scales. Each
/// route's input and output addresses must cover the declared feature counts.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_expert_gemm_into_on_stream(
    routes: &MoeSortedRoutes,
    inputs: &DeviceBuffer<DeviceAddress<f32>>,
    weights: &DeviceBuffer<DeviceAddress<u8>>,
    scales: &DeviceBuffer<DeviceAddress<u8>>,
    globals: &DeviceBuffer<f32>,
    outputs: &DeviceBuffer<DeviceAddress<f32>>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let experts = weights.len();
    if experts == 0
        || experts > 1024
        || scales.len() != experts
        || globals.len() != experts
        || routes.expert_offsets().len() != experts + 1
        || inputs.len() < routes.active_routes()
        || outputs.len() < routes.active_routes()
        || out_features == 0
        || out_features > u32::MAX as usize - 63
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
    {
        return Err(Error::Format {
            label: "LFM2 grouped expert GEMM",
            detail: "expected matching route and expert tables, positive dimensions and K divisible by 16".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_expert_gemm_on_stream",
            infer_lfm2_expert_gemm_on_stream(
                routes.sorted_routes().as_const_ptr().cast(),
                routes.expert_offsets().as_const_ptr().cast(),
                inputs.ptr.cast(),
                weights.ptr.cast(),
                scales.ptr.cast(),
                globals.ptr,
                outputs.ptr.cast(),
                routes.active_routes() as u32,
                experts as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Pipelines compressed weight tiles for sixteen routes, retaining F32 activations.
/// Source vectors must be 16-byte aligned when K is divisible by 64.
/// Weight tables contain row-major ModelOpt NVFP4 and K16 UE4M3 scales. Each
/// route's input and output addresses must cover the declared feature counts.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_expert_gemm_pipelined_into_on_stream(
    routes: &MoeSortedRoutes,
    inputs: &DeviceBuffer<DeviceAddress<f32>>,
    weights: &DeviceBuffer<DeviceAddress<u8>>,
    scales: &DeviceBuffer<DeviceAddress<u8>>,
    globals: &DeviceBuffer<f32>,
    outputs: &DeviceBuffer<DeviceAddress<f32>>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let experts = weights.len();
    if experts == 0
        || experts > 1024
        || scales.len() != experts
        || globals.len() != experts
        || routes.expert_offsets().len() != experts + 1
        || inputs.len() < routes.active_routes()
        || outputs.len() < routes.active_routes()
        || out_features == 0
        || out_features > u32::MAX as usize - 63
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
    {
        return Err(Error::Format {
            label: "LFM2 grouped expert GEMM",
            detail: "expected matching route and expert tables, positive dimensions and K divisible by 16".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_expert_gemm_pipelined_on_stream",
            infer_lfm2_expert_gemm_pipelined_on_stream(
                routes.sorted_routes().as_const_ptr().cast(),
                routes.expert_offsets().as_const_ptr().cast(),
                inputs.ptr.cast(),
                weights.ptr.cast(),
                scales.ptr.cast(),
                globals.ptr,
                outputs.ptr.cast(),
                routes.active_routes() as u32,
                experts as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_dense_gemm_on_stream(
        input: *const f32,
        weight: *const u16,
        output: *mut f32,
        rows: u32,
        out_features: u32,
        in_features: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Multiplies BF16 weights by a row-major F32 batch using three BF16 components.
pub fn lfm2_dense_gemm_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || rows > 65535 * 8
        || out_features == 0
        || out_features > u32::MAX as usize - 63
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
        || rows
            .checked_mul(in_features)
            .is_none_or(|n| n > input.len())
        || out_features.checked_mul(in_features) != Some(weight.len())
        || rows
            .checked_mul(out_features)
            .is_none_or(|n| n > output.len())
    {
        return Err(Error::Format {
            label: "F32 dense GEMM",
            detail: "expected matching positive batch dimensions and K divisible by 16".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_dense_gemm_on_stream",
            infer_lfm2_dense_gemm_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_decode_attention_on_stream(
        query: *const f32,
        key: *const f32,
        value: *const f32,
        length: *const u32,
        partial: *mut f32,
        output: *mut f32,
        q_heads: u32,
        kv_heads: u32,
        partitions: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Reusable partial softmax storage for F32 head-64 decode attention.
pub struct Head64DecodeAttention {
    partial: DeviceBuffer<f32>,
    capacity: usize,
    q_heads: usize,
    kv_heads: usize,
    partitions: usize,
}
impl Head64DecodeAttention {
    /// Allocates F32 softmax partials for a fixed cache capacity and head layout.
    pub fn new(capacity: usize, q_heads: usize, kv_heads: usize) -> Result<Self> {
        if capacity == 0
            || capacity > u32::MAX as usize - 63
            || q_heads == 0
            || q_heads > 65535
            || kv_heads == 0
            || !q_heads.is_multiple_of(kv_heads)
        {
            return Err(Error::Format {
                label: "head-64 decode attention",
                detail: "expected positive capacity and query heads divisible by KV heads".into(),
            });
        }
        let partitions = capacity.div_ceil(256).min(32);
        Ok(Self {
            partial: DeviceBuffer::zeroed(q_heads * partitions * 66)?,
            capacity,
            q_heads,
            kv_heads,
            partitions,
        })
    }
    /// Returns the bytes owned by the partial-reduction buffer.
    pub fn device_bytes(&self) -> usize {
        self.partial.device_bytes()
    }
    /// The device length must be in `1..=capacity` at execution time.
    pub fn run_on_stream(
        &mut self,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        length: &DeviceBuffer<u32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if query.len() != self.q_heads * 64
            || output.len() != query.len()
            || key.len() < self.capacity * self.kv_heads * 64
            || value.len() < key.len()
            || length.len() != 1
        {
            return Err(Error::Format {
                label: "head-64 decode attention",
                detail: "query, cache, length or output does not match the workspace".into(),
            });
        }
        unsafe {
            check_cuda(
                "infer_lfm2_decode_attention_on_stream",
                infer_lfm2_decode_attention_on_stream(
                    query.ptr,
                    key.ptr,
                    value.ptr,
                    length.ptr,
                    self.partial.ptr,
                    output.buffer_mut().ptr,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.partitions as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn lfm2_expert_all_finite_scales_and_packed_codes() {
        let stream = CudaStream::new_non_blocking().unwrap();
        for cols in [16, 64, 128] {
            let rows = 127 * 256 + 1;
            let mut packed = vec![0u8; rows * cols / 2];
            let scales = (0..rows * cols / 16)
                .map(|i| ((i / (cols / 16) / 256).min(126)) as u8)
                .collect::<Vec<_>>();
            for row in 0..rows {
                packed[row * cols / 2] = (row % 256) as u8;
            }
            let weight = DeviceBuffer::from_host(&packed).unwrap();
            let scale = DeviceBuffer::from_host(&scales).unwrap();
            let weights = DeviceBuffer::from_host(&[weight.cuda_address()]).unwrap();
            let scale_table = DeviceBuffer::from_host(&[scale.cuda_address()]).unwrap();
            let global = DeviceBuffer::from_host(&[1.0f32]).unwrap();
            let indices = DeviceBuffer::from_host(&[0u32]).unwrap();
            let mut routes = MoeSortedRoutes::new(1, 1).unwrap();
            routes.sort_on_stream(&indices, &stream).unwrap();
            let output = DeviceBuffer::<f32>::zeroed(rows).unwrap();
            let outputs = DeviceBuffer::from_host(&[output.cuda_address()]).unwrap();
            let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            for nibble in 0..2 {
                let mut input = vec![0.0f32; cols];
                input[nibble] = 1.0;
                let input = DeviceBuffer::from_host(&input).unwrap();
                let inputs = DeviceBuffer::from_host(&[input.cuda_address()]).unwrap();
                for kernel in [
                    lfm2_expert_gemm_into_on_stream,
                    lfm2_expert_gemm_pipelined_into_on_stream,
                ] {
                    kernel(
                        &routes,
                        &inputs,
                        &weights,
                        &scale_table,
                        &global,
                        &outputs,
                        rows,
                        cols,
                        &stream,
                    )
                    .unwrap();
                    let actual = output.copy_to_host(&stream).unwrap();
                    for (row, &value) in actual.iter().enumerate() {
                        let code = (row % 256) >> (nibble * 4) & 15;
                        let scale = (row / 256).min(126);
                        let exponent = (scale >> 3) as i32;
                        let mantissa = (scale & 7) as f32;
                        let scale = if exponent == 0 {
                            mantissa * 2.0f32.powi(-9)
                        } else {
                            (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
                        };
                        let sign = if code & 8 == 0 { 1.0 } else { -1.0 };
                        assert_eq!(
                            value,
                            sign * values[code & 7] * scale,
                            "packed byte {}, scale {}, nibble {nibble}",
                            row % 256,
                            row / 256
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn lfm2_short_conv_preserves_history_and_linear_gates() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let weights = [0.5, -0.25, 2.0, -1.0, 0.75, 0.25];
        let weight = DeviceBuffer::from_host(&weights).unwrap();
        let mut state = DeviceBuffer::zeroed(4).unwrap();
        let mut output = DeviceBuffer::zeroed(2).unwrap();
        let mut history = [[0.0f32; 2]; 2];
        for row in [
            [2.0, -3.0, -2.0, 0.5, 1.5, 2.0],
            [-1.0, 4.0, 3.0, -2.0, 0.5, -1.0],
            [0.0, 1.0, -1.0, 2.0, 3.0, 0.5],
            [1.0; 6],
        ] {
            let bcx = DeviceBuffer::from_host(&row).unwrap();
            lfm2_short_conv_into_on_stream(
                &bcx,
                &weight,
                state.inout(),
                output.output(),
                3,
                &stream,
            )
            .unwrap();
            let actual = output.copy_to_host(&stream).unwrap();
            for c in 0..2 {
                let bx = row[c] * row[4 + c];
                let expected = row[2 + c]
                    * (history[c][0] * weights[c * 3]
                        + history[c][1] * weights[c * 3 + 1]
                        + bx * weights[c * 3 + 2]);
                assert!((actual[c] - expected).abs() < 1e-6);
                history[c] = [history[c][1], bx];
            }
        }
        let final_state = state.copy_to_host(&stream).unwrap();
        assert_eq!(&final_state[..], &history.concat());
    }

    #[test]
    #[serial_test::serial]
    fn lfm2_routing_bias_does_not_leak_into_weights() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let logits = DeviceBuffer::from_host(&[0.0f32, 2.0, -2.0, 1.0]).unwrap();
        let bias = DeviceBuffer::from_host(&[0.0f32, -2.0, 2.0, 0.0]).unwrap();
        let mut indices = DeviceBuffer::zeroed(2).unwrap();
        let mut weights = DeviceBuffer::zeroed(2).unwrap();
        for normalize in [false, true] {
            lfm2_router_into_on_stream(
                &logits,
                &bias,
                indices.output(),
                weights.output(),
                normalize,
                1.7,
                &stream,
            )
            .unwrap();
            assert_eq!(&indices.copy_to_host(&stream).unwrap()[..], &[2, 3]);
            let expected = [1.0 / (1.0 + 2.0f32.exp()), 1.0 / (1.0 + (-1.0f32).exp())];
            let denominator = if normalize {
                expected.iter().sum::<f32>() + 1e-6
            } else {
                1.0
            };
            let actual = weights.copy_to_host(&stream).unwrap();
            for i in 0..2 {
                assert!((actual[i] - expected[i] * 1.7 / denominator).abs() < 1e-6);
            }
        }
        // The epsilon matters for tiny probabilities, where a clamp differs.
        let logits = DeviceBuffer::from_host(&[-20.0f32; 4]).unwrap();
        lfm2_router_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            true,
            1.0,
            &stream,
        )
        .unwrap();
        let p = 1.0 / (1.0 + 20.0f32.exp());
        for value in weights.copy_to_host(&stream).unwrap().iter() {
            assert!((*value - p / (2.0 * p + 1e-6)).abs() < 1e-8);
        }
    }
    #[test]
    #[serial_test::serial]
    fn lfm2_batched_convolution_matches_decode_across_short_chunks() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let h = 5;
        let width = 4;
        let host: Vec<f32> = (0..9 * 3 * h)
            .map(|i| (i % 17) as f32 * 0.25 - 2.0)
            .collect();
        let weight = DeviceBuffer::from_host(
            &(0..h * width)
                .map(|i| i as f32 * 0.125 - 1.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut serial = DeviceBuffer::zeroed(h * (width - 1)).unwrap();
        let mut batch = DeviceBuffer::zeroed(h * (width - 1)).unwrap();
        let mut one = DeviceBuffer::zeroed(h).unwrap();
        let mut expected = Vec::new();
        for row in host.chunks_exact(3 * h) {
            let input = DeviceBuffer::from_host(row).unwrap();
            lfm2_short_conv_into_on_stream(
                &input,
                &weight,
                serial.inout(),
                one.output(),
                width,
                &stream,
            )
            .unwrap();
            expected.extend_from_slice(&one.copy_to_host(&stream).unwrap());
        }
        let mut start = 0;
        for rows in [1, 2, 1, 5] {
            let input =
                DeviceBuffer::from_host(&host[start * 3 * h..(start + rows) * 3 * h]).unwrap();
            let mut output = DeviceBuffer::zeroed(rows * h).unwrap();
            lfm2_short_conv_batch_into_on_stream(
                &input,
                &weight,
                batch.inout(),
                output.output(),
                h,
                width,
                rows,
                &stream,
            )
            .unwrap();
            assert_eq!(
                &output.copy_to_host(&stream).unwrap()[..],
                &expected[start * h..(start + rows) * h]
            );
            start += rows;
        }
        assert_eq!(
            &serial.copy_to_host(&stream).unwrap()[..],
            &batch.copy_to_host(&stream).unwrap()[..]
        );
    }

    #[test]
    #[serial_test::serial]
    fn lfm2_batched_router_keeps_rows_independent() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let logits =
            DeviceBuffer::from_host(&[3.0f32, 1.0, -2.0, -3.0, -4.0, -1.0, 2.0, 3.0]).unwrap();
        let bias = DeviceBuffer::zeroed(4).unwrap();
        let mut indices = DeviceBuffer::zeroed(4).unwrap();
        let mut weights = DeviceBuffer::zeroed(4).unwrap();
        lfm2_router_batch_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            2,
            2,
            true,
            1.0,
            &stream,
        )
        .unwrap();
        assert_eq!(&indices.copy_to_host(&stream).unwrap()[..], &[0, 1, 3, 2]);
        let values = weights.copy_to_host(&stream).unwrap();
        for row in values.as_chunks::<2>().0 {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
}
