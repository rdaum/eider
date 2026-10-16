//! LFM2 projections, gated short convolution, routing, and head-64 attention.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::{
    Bf16TnMatmulPlan, CublasLt, DeviceAddress, Error, Fp8TnMatmulPlan, GemmShape, MoeSortedRoutes,
    Result,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

unsafe extern "C" {
    fn infer_fp8_bf16_rows_on_stream(
        input: *const f32,
        weight: *const u8,
        channel_scales: *const f32,
        scale: f32,
        output: *mut f32,
        rows: u32,
        out: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_split_f32_fp8_parts_on_stream(
        input: *const f32,
        parts: *mut u8,
        powers: *mut f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_sum_f32_fp8_parts_on_stream(
        parts: *const f32,
        powers: *const f32,
        output: *mut f32,
        rows: u32,
        width: u32,
        weight_scale: f32,
        stream: *mut c_void,
    ) -> i32;
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
        parts_count: u32,
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
        parts_count: u32,
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

/// Arithmetic used for dense and packed tensor-core projections.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GemmPrecision {
    /// Retain F32 inputs with multiple tensor-core products.
    #[default]
    F32,
    /// Round inputs and dequantized weights to BF16, accumulating in F32.
    Bf16,
}

/// Reusable projections from F32 rows, with BF16 decomposition or calibrated FP8 activations.
pub struct F32DenseGemm {
    ggml: Option<crate::GgmlGemmWorkspace>,
    precision: GemmPrecision,
    row_segments: Vec<usize>,
    lt: CublasLt,
    plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    plan_workspace: Rc<DeviceBuffer<u8>>,
    fp8_plans: HashMap<(usize, usize, usize), Fp8TnMatmulPlan>,
    fp8_input: DeviceBuffer<u8>,
    fp8_parts: Option<Fp8Parts>,
    input: DeviceBuffer<u16>,
    products: DeviceBuffer<f32>,
    expanded_weight: DeviceBuffer<u16>,
    capacity: usize,
    features: usize,
}

struct Fp8Parts {
    input: DeviceBuffer<u8>,
    products: DeviceBuffer<f32>,
    powers: DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
enum DenseGemmWeights<'a> {
    Bf16(&'a DeviceBuffer<u16>),
    Quantized(&'a DeviceBuffer<u8>),
}

impl F32DenseGemm {
    /// Selected matrix arithmetic.
    pub fn precision(&self) -> GemmPrecision {
        self.precision
    }

    /// Selects matrix arithmetic before workspace warm-up and graph capture.
    pub fn set_precision(&mut self, precision: GemmPrecision) {
        self.precision = precision;
    }

    /// Reuses one activation workspace across packed dense and expert projections.
    pub fn ggml_workspace(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<&mut crate::GgmlGemmWorkspace> {
        if let Some(ws) = &mut self.ggml {
            ws.reserve(rows, cols)?;
        } else {
            self.ggml = Some(crate::GgmlGemmWorkspace::new(rows, cols)?);
        }
        let ws = self.ggml.as_mut().unwrap();
        ws.set_precision(self.precision);
        Ok(ws)
    }

    /// Allocates scratch for the largest input/output width and prompt chunk.
    /// Set max_quantized_elements to zero for BF16-only projections.
    pub fn new(capacity: usize, features: usize, max_quantized_elements: usize) -> Result<Self> {
        let count = capacity
            .max(8)
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
        // Preparation and GEMM overwrite every active element before it is read.
        Ok(Self {
            ggml: None,
            precision: GemmPrecision::F32,
            row_segments: Vec::new(),
            lt: CublasLt::new()?,
            plans: HashMap::new(),
            plan_workspace: Rc::new(DeviceBuffer::uninitialized(1 << 20)?),
            fp8_plans: HashMap::new(),
            fp8_input: DeviceBuffer::uninitialized(count.unwrap() / 3)?,
            fp8_parts: None,
            input: DeviceBuffer::uninitialized(count.unwrap())?,
            products: DeviceBuffer::uninitialized(count.unwrap())?,
            expanded_weight: DeviceBuffer::uninitialized(max_quantized_elements)?,
            capacity,
            features,
        })
    }

    /// Preserves each sequence's reduction shape before activation quantization.
    /// An empty slice treats all rows as one matrix.
    pub fn set_row_segments(&mut self, lengths: &[u32]) -> Result<()> {
        if lengths.contains(&0)
            || lengths.iter().map(|&n| n as usize).sum::<usize>() > self.capacity
        {
            return Err(Error::Format {
                label: "dense GEMM row segments",
                detail: "invalid row lengths or capacity".into(),
            });
        }
        self.row_segments.clear();
        self.row_segments
            .extend(lengths.iter().map(|&n| n as usize));
        Ok(())
    }

    /// Projects up to eight F32 rows as six scaled E4M3 residual components.
    /// Plans and component scratch are prepared on the first call before graph capture.
    #[allow(clippy::too_many_arguments)]
    pub fn run_fp8_f32_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        mut output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        weight_scale: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0
            || rows > 8
            || rows > self.capacity
            || out == 0
            || cols == 0
            || out > self.features
            || cols > self.features
            || !out.is_multiple_of(16)
            || !cols.is_multiple_of(16)
            || input.len() < rows * cols
            || output.len() < rows * out
            || weight.len() != out * cols
            || !weight_scale.is_finite()
            || (!self.row_segments.is_empty() && self.row_segments.iter().sum::<usize>() != rows)
        {
            return Err(Error::Format {
                label: "F32 FP8 component GEMM",
                detail: "invalid dimensions, buffers, or row segments".into(),
            });
        }
        if self.fp8_parts.is_none() {
            let count = 6 * self.capacity.min(8) * self.features;
            self.fp8_parts = Some(Fp8Parts {
                input: DeviceBuffer::zeroed(count)?,
                products: DeviceBuffer::zeroed(count)?,
                powers: DeviceBuffer::zeroed(self.capacity.min(8))?,
            });
        }
        let mut offset = 0;
        for index in 0..self.row_segments.len().max(1) {
            let count = self.row_segments.get(index).copied().unwrap_or(rows);
            let key = (out, cols, 6 * count);
            if let std::collections::hash_map::Entry::Vacant(entry) = self.fp8_plans.entry(key) {
                entry.insert(Fp8TnMatmulPlan::new(
                    &self.lt,
                    GemmShape::new(out, 6 * count, cols),
                    4 << 20,
                )?);
            }
            let parts = self.fp8_parts.as_mut().unwrap();
            unsafe {
                check_cuda(
                    "infer_split_f32_fp8_parts_on_stream",
                    infer_split_f32_fp8_parts_on_stream(
                        input.ptr.add(offset * cols),
                        parts.input.ptr,
                        parts.powers.ptr,
                        count as u32,
                        cols as u32,
                        stream.as_raw(),
                    ),
                )?;
            }
            self.fp8_plans[&key].run_with_alpha_on_stream(
                &self.lt,
                weight,
                &parts.input,
                parts.products.output(),
                1.0,
                stream,
            )?;
            unsafe {
                check_cuda(
                    "infer_sum_f32_fp8_parts_on_stream",
                    infer_sum_f32_fp8_parts_on_stream(
                        parts.products.ptr,
                        parts.powers.ptr,
                        output.buffer_mut().ptr.add(offset * out),
                        count as u32,
                        out as u32,
                        weight_scale,
                        stream.as_raw(),
                    ),
                )
            }?;
            offset += count;
        }
        Ok(())
    }

    /// Uses a checkpoint's calibrated FP8 activation scale and resident FP8 weights.
    /// Plans use the exact row count; callers retain one workspace per batch shape.
    #[allow(clippy::too_many_arguments)]
    pub fn run_calibrated_fp8_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u8>,
        mut output: DeviceOutput<'_, f32>,
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
        if !self.row_segments.is_empty() && self.row_segments.iter().sum::<usize>() != rows {
            return Err(Error::Format {
                label: "dense GEMM row segments",
                detail: "row count differs from configured segments".into(),
            });
        }
        let mut offset = 0;
        for index in 0..self.row_segments.len().max(1) {
            let count = self.row_segments.get(index).copied().unwrap_or(rows);
            let key = (out, cols, count);
            if let std::collections::hash_map::Entry::Vacant(entry) = self.fp8_plans.entry(key) {
                entry.insert(Fp8TnMatmulPlan::new(
                    &self.lt,
                    GemmShape {
                        m: out,
                        n: count,
                        k: cols,
                    },
                    if count < 8 { 4 << 20 } else { 1 << 20 },
                )?);
            }
            unsafe {
                check_cuda(
                    "infer_quantize_fp8_e4m3_f32_on_stream",
                    crate::ffi::infer_quantize_fp8_e4m3_f32_on_stream(
                        input.ptr.add(offset * cols),
                        self.fp8_input.ptr,
                        (count * cols) as u32,
                        input_scale,
                        stream.as_raw(),
                    ),
                )?;
            }
            if count == rows {
                self.fp8_plans[&key].run_with_alpha_on_stream(
                    &self.lt,
                    weight,
                    &self.fp8_input,
                    output,
                    weight_scale * input_scale,
                    stream,
                )?;
                return Ok(());
            }
            self.fp8_plans[&key].run_with_alpha_on_stream(
                &self.lt,
                weight,
                &self.fp8_input,
                self.products.output(),
                weight_scale * input_scale,
                stream,
            )?;
            output.buffer_mut().copy_range_from_device_on_stream(
                offset * out,
                &self.products,
                0,
                count * out,
                stream,
            )?;
            offset += count;
        }
        Ok(())
    }

    /// Projects active rows with the selected matrix precision.
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
        self.run_with_precision_on_stream(
            input,
            weight,
            output,
            rows,
            out,
            cols,
            self.precision,
            stream,
        )
    }

    /// Overrides arithmetic for one projection without changing the workspace default.
    /// Sensitive projections such as MoE routing can retain F32 inputs while
    /// sharing scratch and cached plans with BF16 matrix operations.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_precision_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        precision: GemmPrecision,
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
            precision,
            stream,
        )
    }

    /// Prepares exact BF16 E2M1-by-UE4M3 products once for repeated dense batches.
    /// The tensor-wide scale remains separate and is applied after projection.
    pub fn prepare_nvfp4_weights(
        weight: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        stream: &CudaStream,
    ) -> Result<DeviceBuffer<u16>> {
        let elements = weight
            .len()
            .checked_mul(2)
            .filter(|&n| n > 0 && n <= u32::MAX as usize && n.is_multiple_of(16))
            .ok_or_else(|| Error::Format {
                label: "prepared NVFP4 dense weight",
                detail: "invalid packed weight size".into(),
            })?;
        if scales.len() != elements / 16 {
            return Err(Error::Format {
                label: "prepared NVFP4 dense weight",
                detail: "invalid K16 scales".into(),
            });
        }
        let output = DeviceBuffer::zeroed(elements)?;
        unsafe {
            check_cuda(
                "infer_expand_dense_bf16_on_stream",
                infer_expand_dense_bf16_on_stream(
                    weight.ptr,
                    scales.ptr,
                    output.ptr,
                    elements as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        Ok(output)
    }

    /// Projects prepared quantized weights without repeating their expansion.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prepared_quantized_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: DeviceOutput<'_, f32>,
        rows: usize,
        out: usize,
        cols: usize,
        global_scale: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        if !global_scale.is_finite() {
            return Err(Error::Format {
                label: "prepared quantized dense projection",
                detail: "non-finite global scale".into(),
            });
        }
        self.run_impl(
            input,
            DenseGemmWeights::Bf16(weight),
            None,
            None,
            global_scale,
            output,
            rows,
            out,
            cols,
            self.precision,
            stream,
        )
    }

    /// Projects compressed weights, using them directly for small BF16 FP8 batches.
    /// Larger batches expand weights into shared scratch on the same stream.
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
        let packed = self.precision == GemmPrecision::Bf16 && rows <= 8 && block_scales.is_none();
        if !cols.is_multiple_of(16)
            || elements.is_none_or(|n| {
                (!packed && n > self.expanded_weight.len()) || n / divisor != weight.len()
            })
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
            self.precision,
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
        precision: GemmPrecision,
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
        let mut offset = 0;
        if self.row_segments.is_empty() {
            return self.run_segment(
                input,
                weight,
                block_scales,
                channel_scales,
                global_scale,
                output,
                rows,
                out,
                cols,
                0,
                precision,
                stream,
            );
        }
        if self.row_segments.iter().sum::<usize>() != rows {
            return Err(Error::Format {
                label: "dense GEMM row segments",
                detail: "row count differs from configured segments".into(),
            });
        }
        for index in 0..self.row_segments.len() {
            let count = self.row_segments[index];
            self.run_segment(
                input,
                weight,
                block_scales,
                channel_scales,
                global_scale,
                output.buffer_mut().output(),
                count,
                out,
                cols,
                offset,
                precision,
                stream,
            )?;
            offset += count;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_segment(
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
        offset: usize,
        precision: GemmPrecision,
        stream: &CudaStream,
    ) -> Result<()> {
        if precision == GemmPrecision::Bf16
            && rows <= 8
            && block_scales.is_none()
            && let DenseGemmWeights::Quantized(weight) = weight
        {
            return unsafe {
                check_cuda(
                    "infer_fp8_bf16_rows_on_stream",
                    infer_fp8_bf16_rows_on_stream(
                        input.ptr.add(offset * cols),
                        weight.ptr,
                        channel_scales.map_or(std::ptr::null(), |s| s.ptr),
                        global_scale,
                        output.buffer_mut().ptr.add(offset * out),
                        rows as u32,
                        out as u32,
                        cols as u32,
                        stream.as_raw(),
                    ),
                )
            };
        }
        // Power-of-two row buckets bound the plan cache across prompt lengths.
        // Share the BF16 reduction shape across serial and speculative decode.
        let padded_rows = if precision == GemmPrecision::Bf16 {
            rows.max(8)
        } else {
            rows
        }
        .next_power_of_two();
        let parts = if precision == GemmPrecision::Bf16 {
            1
        } else {
            3
        };
        let shape = (out, cols, parts * padded_rows);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.plans.entry(shape) {
            entry.insert(Bf16TnMatmulPlan::new_with_workspace(
                &self.lt,
                GemmShape {
                    m: out,
                    n: parts * padded_rows,
                    k: cols,
                },
                Rc::clone(&self.plan_workspace),
            )?);
        }
        let plan = &self.plans[&shape];
        unsafe {
            check_cuda(
                "infer_lfm2_split_bf16_on_stream",
                infer_lfm2_split_bf16_on_stream(
                    input.ptr.add(offset * cols),
                    self.input.ptr,
                    (rows * cols) as u32,
                    (padded_rows * cols) as u32,
                    parts as u32,
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
        if parts == 1 && rows == padded_rows && channel_scales.is_none() {
            return plan.run_scaled_offsets_on_stream(
                &self.lt,
                weight,
                0,
                &self.input,
                0,
                output,
                offset * out,
                global_scale,
                stream,
            );
        }
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
                    output.buffer_mut().ptr.add(offset * out),
                    (rows * out) as u32,
                    (padded_rows * out) as u32,
                    channel_scales.map_or(std::ptr::null(), |s| s.ptr),
                    global_scale,
                    out as u32,
                    parts as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
    /// Returns scratch and cached plan workspace bytes on the device.
    pub fn device_bytes(&self) -> usize {
        self.input.device_bytes()
            + self
                .ggml
                .as_ref()
                .map_or(0, crate::GgmlGemmWorkspace::device_bytes)
            + self.fp8_parts.as_ref().map_or(0, |p| {
                p.input.device_bytes() + p.products.device_bytes() + p.powers.device_bytes()
            })
            + self.fp8_input.device_bytes()
            + self
                .fp8_plans
                .values()
                .map(Fp8TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
            + self.products.device_bytes()
            + self.expanded_weight.device_bytes()
            + self.plan_workspace.device_bytes()
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
    fn infer_lfm2_expert_gemm_f16_parts_on_stream(
        sorted_routes: *const u32,
        offsets: *const u32,
        inputs: *const *const f32,
        prepared: *mut u16,
        input_powers: *mut f32,
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

/// Projects F32 activations through two scaled FP16 components and separate
/// accumulators. Scratch needs two elements per active input element and one
/// power-of-two scale per route. K16 tails retain the BF16 decomposition kernel.
/// Weight and address tables follow `lfm2_expert_gemm_pipelined_into_on_stream`.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_expert_gemm_f16_parts_into_on_stream(
    routes: &MoeSortedRoutes,
    inputs: &DeviceBuffer<DeviceAddress<f32>>,
    prepared: &mut DeviceBuffer<u16>,
    input_powers: &mut DeviceBuffer<f32>,
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
        || routes
            .active_routes()
            .checked_mul(in_features)
            .and_then(|n| n.checked_mul(2))
            .is_none_or(|n| n > prepared.len())
        || input_powers.len() < routes.active_routes()
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
    {
        return Err(Error::Format {
            label: "LFM2 grouped expert GEMM",
            detail: "expected matching route and expert tables, positive dimensions and K divisible by 16".into(),
        });
    }
    if !in_features.is_multiple_of(64) {
        return lfm2_expert_gemm_pipelined_into_on_stream(
            routes,
            inputs,
            weights,
            scales,
            globals,
            outputs,
            out_features,
            in_features,
            stream,
        );
    }
    unsafe {
        check_cuda(
            "infer_lfm2_expert_gemm_f16_parts_on_stream",
            infer_lfm2_expert_gemm_f16_parts_on_stream(
                routes.sorted_routes().as_const_ptr().cast(),
                routes.expert_offsets().as_const_ptr().cast(),
                inputs.ptr.cast(),
                prepared.ptr,
                input_powers.ptr,
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
    fn packed_fp8_bf16_rows_match_cpu_and_serial() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let (rows, out, cols) = (8, 65, 4096);
        let rounded = |x| crate::format::bf16_to_f32(crate::format::f32_to_bf16(x));
        let host: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 31 % 1009) as f32 - 504.) / 137.137)
            .collect();
        let weights: Vec<_> = (0..out * cols)
            .map(|i| ((i * 31 % 96) | ((i % 3 == 0) as usize * 128)) as u8)
            .collect();
        let channels: Vec<_> = (0..out).map(|i| (i % 19 + 1) as f32 / 137.).collect();
        let input = DeviceBuffer::from_host(&host).unwrap();
        let weight = DeviceBuffer::from_host(&weights).unwrap();
        let scales = DeviceBuffer::from_host(&channels).unwrap();
        let mut output = DeviceBuffer::zeroed(rows * out).unwrap();
        let mut gemm = F32DenseGemm::new(rows, cols, 0).unwrap();
        gemm.set_precision(GemmPrecision::Bf16);
        for channel_scaled in [false, true] {
            let mut reference = Vec::new();
            for row in 0..rows {
                let source = DeviceBuffer::from_host(&host[row * cols..(row + 1) * cols]).unwrap();
                gemm.run_quantized_on_stream(
                    &source,
                    &weight,
                    None,
                    channel_scaled.then_some(&scales),
                    0.125,
                    output.output(),
                    1,
                    out,
                    cols,
                    &stream,
                )
                .unwrap();
                reference.extend_from_slice(&output.copy_to_host(&stream).unwrap()[..out]);
            }
            for count in [1, 2, 3, 4, 5, 7, 8] {
                gemm.run_quantized_on_stream(
                    &input,
                    &weight,
                    None,
                    channel_scaled.then_some(&scales),
                    0.125,
                    output.output(),
                    count,
                    out,
                    cols,
                    &stream,
                )
                .unwrap();
                let actual = output.copy_to_host(&stream).unwrap();
                assert_eq!(&actual[..count * out], &reference[..count * out]);
            }
            gemm.set_row_segments(&[2, 1, 5]).unwrap();
            gemm.run_quantized_on_stream(
                &input,
                &weight,
                None,
                channel_scaled.then_some(&scales),
                0.125,
                output.output(),
                rows,
                out,
                cols,
                &stream,
            )
            .unwrap();
            assert_eq!(&*output.copy_to_host(&stream).unwrap(), &reference);
            gemm.set_row_segments(&[]).unwrap();
            for row in 0..rows {
                for channel in 0..out {
                    let expected = (0..cols)
                        .map(|k| {
                            f64::from(rounded(host[row * cols + k]))
                                * f64::from(crate::format::e4m3_value(weights[channel * cols + k]))
                        })
                        .sum::<f64>() as f32;
                    let expected = expected
                        * if channel_scaled {
                            channels[channel]
                        } else {
                            0.125
                        };
                    let actual = reference[row * out + channel];
                    assert!(
                        (actual - expected).abs() < 2e-4 + expected.abs() * 2e-5,
                        "GPU {actual}, CPU {expected}"
                    );
                }
            }
        }
        assert!(
            gemm.run_quantized_on_stream(
                &input,
                &weight,
                None,
                None,
                1.,
                output.output(),
                0,
                out,
                cols,
                &stream,
            )
            .is_err()
        );
        gemm.set_row_segments(&[1]).unwrap();
        assert!(
            gemm.run_quantized_on_stream(
                &input,
                &weight,
                None,
                None,
                1.,
                output.output(),
                rows,
                out,
                cols,
                &stream,
            )
            .is_err()
        );
    }

    #[test]
    #[serial_test::serial]
    fn bf16_dense_precision_matches_rounded_cpu() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let (rows, out, cols) = (33, 65, 1024);
        let rounded = |x: f32| {
            f32::from_bits((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
        };
        let host: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 31 % 1009) as f32 - 504.) / 137.137)
            .collect();
        let weights: Vec<_> = (0..out * cols)
            .map(|i| (rounded(((i * 71 % 1021) as f32 - 510.) / 1024.).to_bits() >> 16) as u16)
            .collect();
        let input = DeviceBuffer::from_host(&host).unwrap();
        let weight = DeviceBuffer::from_host(&weights).unwrap();
        let mut output = DeviceBuffer::zeroed(rows * out).unwrap();
        let mut gemm = F32DenseGemm::new(rows, cols, 0).unwrap();
        gemm.set_row_segments(&[16, 17]).unwrap();
        for (default, precision) in [
            (GemmPrecision::F32, GemmPrecision::F32),
            (GemmPrecision::Bf16, GemmPrecision::Bf16),
            (GemmPrecision::Bf16, GemmPrecision::F32),
            (GemmPrecision::F32, GemmPrecision::F32),
        ] {
            gemm.set_precision(default);
            gemm.run_with_precision_on_stream(
                &input,
                &weight,
                output.output(),
                rows,
                out,
                cols,
                precision,
                &stream,
            )
            .unwrap();
            assert_eq!(gemm.precision(), default);
            let actual = output.copy_to_host(&stream).unwrap();
            for row in 0..rows {
                for channel in 0..out {
                    let expected = (0..cols)
                        .map(|k| {
                            let x = host[row * cols + k];
                            let x = if precision == GemmPrecision::Bf16 {
                                rounded(x)
                            } else {
                                x
                            };
                            f64::from(x)
                                * f64::from(f32::from_bits(
                                    u32::from(weights[channel * cols + k]) << 16,
                                ))
                        })
                        .sum::<f64>() as f32;
                    let value = actual[row * out + channel];
                    assert!(
                        (value - expected).abs() < 2e-4 + expected.abs() * 2e-5,
                        "{precision:?}: GPU {value}, CPU {expected}"
                    );
                }
            }
        }
        assert_eq!(
            gemm.plans.len(),
            4,
            "reuse both precisions for both row buckets"
        );
    }

    #[test]
    #[serial_test::serial]
    fn segmented_dense_rows_match_independent_bf16_and_calibrated_fp8() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let (cols, out, rows) = (4096, 512, 145);
        let input_host = (0..rows * cols)
            .map(|i| ((i * 31 % 1009) as f32 / 137.137).sin())
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input_host).unwrap();
        let bf16 = DeviceBuffer::from_host(
            &(0..out * cols)
                .map(|i| ((((i * 71 % 1021) as f32 - 510.) / 1024.).to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let fp8 = DeviceBuffer::from_host(
            &(0..out * cols)
                .map(|i| ((i * 31 % 96) | ((i % 3 == 0) as usize * 128)) as u8)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut packed = F32DenseGemm::new(rows, cols, 0).unwrap();
        let mut actual = DeviceBuffer::zeroed(rows * out).unwrap();
        for lengths in [[41, 42, 62], [62, 42, 41]] {
            packed.set_row_segments(&lengths).unwrap();
            for calibrated in [false, true] {
                if calibrated {
                    packed
                        .run_calibrated_fp8_on_stream(
                            &input,
                            &fp8,
                            actual.output(),
                            rows,
                            out,
                            cols,
                            0.125,
                            0.03125,
                            &stream,
                        )
                        .unwrap();
                } else {
                    packed
                        .run_on_stream(&input, &bf16, actual.output(), rows, out, cols, &stream)
                        .unwrap();
                }
                let actual = actual.copy_to_host(&stream).unwrap().into_vec();
                let mut offset = 0;
                for count in lengths {
                    let count = count as usize;
                    let source = DeviceBuffer::from_host(
                        &input_host[offset * cols..(offset + count) * cols],
                    )
                    .unwrap();
                    let mut expected = DeviceBuffer::zeroed(count * out).unwrap();
                    let mut isolated = F32DenseGemm::new(count, cols, 0).unwrap();
                    if calibrated {
                        isolated
                            .run_calibrated_fp8_on_stream(
                                &source,
                                &fp8,
                                expected.output(),
                                count,
                                out,
                                cols,
                                0.125,
                                0.03125,
                                &stream,
                            )
                            .unwrap();
                    } else {
                        isolated
                            .run_on_stream(
                                &source,
                                &bf16,
                                expected.output(),
                                count,
                                out,
                                cols,
                                &stream,
                            )
                            .unwrap();
                    }
                    assert_eq!(
                        &actual[offset * out..(offset + count) * out],
                        expected.copy_to_host(&stream).unwrap().as_ref(),
                        "calibrated={calibrated} offset={offset}"
                    );
                    offset += count;
                }
            }
        }
        assert!(packed.set_row_segments(&[0]).is_err());
        assert!(packed.set_row_segments(&[146]).is_err());
        packed.set_row_segments(&[41]).unwrap();
        assert!(
            packed
                .run_on_stream(&input, &bf16, actual.output(), rows, out, cols, &stream)
                .is_err()
        );
    }

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
            let mut prepared = DeviceBuffer::zeroed(cols * 2).unwrap();
            let mut powers = DeviceBuffer::zeroed(1).unwrap();
            let indices = DeviceBuffer::from_host(&[0u32]).unwrap();
            let mut routes = MoeSortedRoutes::new(1, 1).unwrap();
            routes.sort_on_stream(&indices, &stream).unwrap();
            let output = DeviceBuffer::<f32>::zeroed(rows).unwrap();
            let outputs = DeviceBuffer::from_host(&[output.cuda_address()]).unwrap();
            let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            for magnitude in [0.0, 1.0, 2.0f32.powi(80), 2.0f32.powi(-80)] {
                for nibble in 0..2 {
                    let mut input = vec![0.0f32; cols];
                    input[nibble] = magnitude;
                    let input = DeviceBuffer::from_host(&input).unwrap();
                    let inputs = DeviceBuffer::from_host(&[input.cuda_address()]).unwrap();
                    for mode in 0..3 {
                        if mode == 2 {
                            lfm2_expert_gemm_f16_parts_into_on_stream(
                                &routes,
                                &inputs,
                                &mut prepared,
                                &mut powers,
                                &weights,
                                &scale_table,
                                &global,
                                &outputs,
                                rows,
                                cols,
                                &stream,
                            )
                            .unwrap();
                        } else {
                            let kernel = [
                                lfm2_expert_gemm_into_on_stream,
                                lfm2_expert_gemm_pipelined_into_on_stream,
                            ][mode];
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
                        }
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
                                sign * values[code & 7] * scale * magnitude,
                                "packed byte {}, scale {}, nibble {nibble}",
                                row % 256,
                                row / 256
                            );
                        }
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
