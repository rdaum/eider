//! Group-scaled ternary weights used by the mainline Bonsai `Q2_0_g64` GGUFs.
//!
//! Each 64-value block stores one little-endian FP16 scale followed by sixteen
//! bytes containing four consecutive two-bit weights apiece. Codes 0, 1, and 2
//! represent `-scale`, zero, and `+scale`; code 3 is rejected. Runtime
//! activations use a matching symmetric INT8 scale per 64-value group so the
//! CUDA projection can accumulate exact integer dot products with `dp4a`.

use crate::cublaslt::{Bf16TnMatmulPlan, CublasLt, Fp4TnMatmulPlan, GemmShape, Nvfp4TnInputs};
use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceInput, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::kernels::non_gemm::f32_to_bf16_into_on_stream;
use crate::matrix::Nvfp4Matrix;
use crate::modelopt_device::{ModelOptCublasLtWeight, ModelOptNvfp4Linear};

/// Number of input values represented by one GGUF scale block.
pub const TERNARY_G64_GROUP_SIZE: usize = 64;
const PACKED_BYTES_PER_GROUP: usize = TERNARY_G64_GROUP_SIZE / 4;
const GGUF_BYTES_PER_GROUP: usize = 2 + PACKED_BYTES_PER_GROUP;

/// Host-side row-major packed group-scaled ternary linear.
#[derive(Clone, Debug)]
pub struct TernaryG64PackedLinear {
    /// Logical tensor name used in diagnostics.
    pub name: String,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
    /// Four K-consecutive ternary codes per byte, row-major.
    pub packed_weight: Vec<u8>,
    /// One decoded FP16 checkpoint scale per output row and 64-value K group.
    pub group_scales: Vec<f32>,
}

impl TernaryG64PackedLinear {
    /// Imports raw mainline GGUF `Q2_0_g64` blocks without dequantizing weights.
    pub fn from_gguf_q2_0_g64(
        name: impl Into<String>,
        out_features: usize,
        in_features: usize,
        raw: &[u8],
    ) -> Result<Self> {
        validate_shape(out_features, in_features)?;
        let groups_per_row = in_features / TERNARY_G64_GROUP_SIZE;
        let groups = out_features
            .checked_mul(groups_per_row)
            .ok_or_else(|| shape_overflow("GGUF group count"))?;
        let expected = groups
            .checked_mul(GGUF_BYTES_PER_GROUP)
            .ok_or_else(|| shape_overflow("GGUF byte count"))?;
        if raw.len() != expected {
            return Err(Error::Shape {
                label: "ternary g64 GGUF payload",
                expected: format!("{expected} bytes"),
                actual: format!("{} bytes", raw.len()),
            });
        }

        let mut packed_weight = Vec::with_capacity(groups * PACKED_BYTES_PER_GROUP);
        let mut group_scales = Vec::with_capacity(groups);
        for (group, block) in raw.chunks_exact(GGUF_BYTES_PER_GROUP).enumerate() {
            let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
            if !scale.is_finite() || scale < 0.0 {
                return Err(Error::Format {
                    label: "ternary g64 scale",
                    detail: format!("group {group} has invalid FP16 scale {scale}"),
                });
            }
            for (packed_index, &packed) in block[2..].iter().enumerate() {
                if packed & (packed >> 1) & 0x55 != 0 {
                    for within in 0..4 {
                        let code = (packed >> (within * 2)) & 0x03;
                        if code != 3 {
                            continue;
                        }
                        let element = group * TERNARY_G64_GROUP_SIZE + packed_index * 4 + within;
                        return Err(Error::Format {
                            label: "ternary g64 weight",
                            detail: format!("reserved code 3 at flattened element {element}"),
                        });
                    }
                }
            }
            group_scales.push(scale);
            packed_weight.extend_from_slice(&block[2..]);
        }

        Ok(Self {
            name: name.into(),
            out_features,
            in_features,
            packed_weight,
            group_scales,
        })
    }

    /// Concatenates matrices with a shared input width along the output axis.
    pub fn concat_rows(name: impl Into<String>, matrices: &[Self]) -> Result<Self> {
        let Some(first) = matrices.first() else {
            return Err(Error::Shape {
                label: "ternary g64 row concatenation",
                expected: "at least one matrix".to_string(),
                actual: "zero matrices".to_string(),
            });
        };
        if matrices
            .iter()
            .any(|matrix| matrix.in_features != first.in_features)
        {
            return Err(Error::Shape {
                label: "ternary g64 row concatenation",
                expected: format!("all input widths equal {}", first.in_features),
                actual: matrices
                    .iter()
                    .map(|matrix| matrix.in_features.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        let out_features = matrices.iter().try_fold(0usize, |rows, matrix| {
            rows.checked_add(matrix.out_features)
                .ok_or_else(|| shape_overflow("concatenated output rows"))
        })?;
        let packed_bytes = matrices
            .iter()
            .map(|matrix| matrix.packed_weight.len())
            .sum();
        let scale_count = matrices
            .iter()
            .map(|matrix| matrix.group_scales.len())
            .sum();
        let mut packed_weight = Vec::with_capacity(packed_bytes);
        let mut group_scales = Vec::with_capacity(scale_count);
        for matrix in matrices {
            packed_weight.extend_from_slice(&matrix.packed_weight);
            group_scales.extend_from_slice(&matrix.group_scales);
        }
        Ok(Self {
            name: name.into(),
            out_features,
            in_features: first.in_features,
            packed_weight,
            group_scales,
        })
    }

    /// Returns the represented checkpoint weight at `(row, col)`.
    pub fn weight(&self, row: usize, col: usize) -> Result<f32> {
        if row >= self.out_features || col >= self.in_features {
            return Err(Error::Shape {
                label: "ternary g64 weight index",
                expected: format!("row < {} and col < {}", self.out_features, self.in_features),
                actual: format!("row={row} col={col}"),
            });
        }
        Ok(self.weight_unchecked(row, col))
    }

    fn weight_unchecked(&self, row: usize, col: usize) -> f32 {
        let groups_per_row = self.in_features / TERNARY_G64_GROUP_SIZE;
        let group = row * groups_per_row + col / TERNARY_G64_GROUP_SIZE;
        let row_byte = row * (self.in_features / 4) + col / 4;
        let code = (self.packed_weight[row_byte] >> ((col % 4) * 2)) & 0x03;
        (i32::from(code) - 1) as f32 * self.group_scales[group]
    }

    fn quantize_nvfp4(&self) -> Result<ModelOptNvfp4Linear> {
        ModelOptNvfp4Linear::quantize_values(
            format!("{}.nvfp4_prefill", self.name),
            self.out_features,
            self.in_features,
            |index| {
                let row = index / self.in_features;
                let col = index % self.in_features;
                self.weight_unchecked(row, col)
            },
        )
        .map_err(|error| Error::Format {
            label: "ternary NVFP4 preparation",
            detail: error.to_string(),
        })
    }

    /// Computes the same per-group W2A8 operation as the CUDA path.
    pub fn reference_w2a8(&self, input: &[f32], batch_rows: usize) -> Result<Vec<f32>> {
        if batch_rows == 0 || input.len() != batch_rows * self.in_features {
            return Err(Error::Shape {
                label: "ternary g64 reference input",
                expected: format!("{} values", batch_rows * self.in_features),
                actual: format!("{} values", input.len()),
            });
        }
        let groups_per_row = self.in_features / TERNARY_G64_GROUP_SIZE;
        let mut output = vec![0.0f32; batch_rows * self.out_features];
        for batch in 0..batch_rows {
            let input_row = &input[batch * self.in_features..(batch + 1) * self.in_features];
            let mut quantized = vec![0i32; self.in_features];
            let mut input_scales = vec![0.0f32; groups_per_row];
            for (group, input_scale) in input_scales.iter_mut().enumerate() {
                let start = group * TERNARY_G64_GROUP_SIZE;
                let values = &input_row[start..start + TERNARY_G64_GROUP_SIZE];
                let maximum = values
                    .iter()
                    .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                *input_scale = maximum / 127.0;
                let quantize_scale = if maximum == 0.0 { 0.0 } else { 127.0 / maximum };
                for (offset, value) in values.iter().enumerate() {
                    quantized[start + offset] = if maximum == 0.0 {
                        0
                    } else {
                        (value * quantize_scale)
                            .round_ties_even()
                            .clamp(-127.0, 127.0) as i32
                    };
                }
            }
            for row in 0..self.out_features {
                let mut sum = 0.0f32;
                for (group, &input_scale) in input_scales.iter().enumerate() {
                    let mut integer_sum = 0i32;
                    let start = group * TERNARY_G64_GROUP_SIZE;
                    for (within_group, &quantized_value) in quantized
                        [start..start + TERNARY_G64_GROUP_SIZE]
                        .iter()
                        .enumerate()
                    {
                        let col = start + within_group;
                        let byte = self.packed_weight[row * (self.in_features / 4) + col / 4];
                        let code = (byte >> ((col % 4) * 2)) & 0x03;
                        integer_sum += (i32::from(code) - 1) * quantized_value;
                    }
                    sum += integer_sum as f32
                        * input_scale
                        * self.group_scales[row * groups_per_row + group];
                }
                output[batch * self.out_features + row] = sum;
            }
        }
        Ok(output)
    }
}

/// Device-resident group-scaled ternary linear.
pub struct TernaryG64Matrix {
    rows: usize,
    cols: usize,
    packed_weight: DeviceBuffer<u8>,
    group_scales: DeviceBuffer<f32>,
    bf16_prefill_weight: Option<DeviceBuffer<u16>>,
    nvfp4_prefill_weight: Option<ModelOptCublasLtWeight>,
}

impl TernaryG64Matrix {
    /// Uploads one imported group-scaled ternary linear.
    pub fn from_packed(linear: &TernaryG64PackedLinear) -> Result<Self> {
        Self::from_packed_impl(linear, false, false)
    }

    /// Uploads packed decode weights and materializes a BF16 tensor-core prefill copy.
    pub fn from_packed_with_bf16_prefill(linear: &TernaryG64PackedLinear) -> Result<Self> {
        Self::from_packed_impl(linear, true, false)
    }

    /// Uploads packed decode weights and materializes an NVFP4 tensor-core prefill copy.
    pub fn from_packed_with_nvfp4_prefill(linear: &TernaryG64PackedLinear) -> Result<Self> {
        Self::from_packed_impl(linear, false, true)
    }

    fn from_packed_impl(
        linear: &TernaryG64PackedLinear,
        bf16_prefill: bool,
        nvfp4_prefill: bool,
    ) -> Result<Self> {
        validate_shape(linear.out_features, linear.in_features)?;
        let groups = linear.out_features * (linear.in_features / TERNARY_G64_GROUP_SIZE);
        if linear.packed_weight.len() != linear.out_features * linear.in_features / 4
            || linear.group_scales.len() != groups
        {
            return Err(Error::Shape {
                label: "ternary g64 host matrix",
                expected: format!(
                    "{} packed bytes and {groups} scales",
                    linear.out_features * linear.in_features / 4
                ),
                actual: format!(
                    "{} packed bytes and {} scales",
                    linear.packed_weight.len(),
                    linear.group_scales.len()
                ),
            });
        }
        let mut matrix = Self {
            rows: linear.out_features,
            cols: linear.in_features,
            packed_weight: DeviceBuffer::from_host(&linear.packed_weight)?,
            group_scales: DeviceBuffer::from_host(&linear.group_scales)?,
            bf16_prefill_weight: None,
            nvfp4_prefill_weight: None,
        };
        if bf16_prefill {
            let weight = DeviceBuffer::zeroed(matrix.rows * matrix.cols)?;
            let stream = CudaStream::new_non_blocking()?;
            let rows = u32::try_from(matrix.rows).map_err(dimension_overflow("rows"))?;
            let cols = u32::try_from(matrix.cols).map_err(dimension_overflow("columns"))?;
            unsafe {
                check_cuda(
                    "infer_ternary_g64_expand_bf16_on_stream",
                    ffi::infer_ternary_g64_expand_bf16_on_stream(
                        matrix.packed_weight.ptr,
                        matrix.group_scales.ptr,
                        weight.ptr,
                        rows,
                        cols,
                        stream.as_raw(),
                    ),
                )?;
            }
            stream.synchronize()?;
            matrix.bf16_prefill_weight = Some(weight);
        }
        if nvfp4_prefill {
            let host = linear.quantize_nvfp4()?;
            matrix.nvfp4_prefill_weight = Some(ModelOptCublasLtWeight::from_modelopt(&host)?);
        }
        Ok(matrix)
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Device bytes occupied by packed weights and group scales.
    pub fn device_bytes(&self) -> usize {
        self.packed_weight.device_bytes()
            + self.group_scales.device_bytes()
            + self
                .bf16_prefill_weight
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes)
            + self
                .nvfp4_prefill_weight
                .as_ref()
                .map_or(0, ModelOptCublasLtWeight::device_bytes)
    }

    /// Creates a reusable F32-output W4A4 prefill plan for this matrix.
    pub fn new_f32_batch_nvfp4_plan(
        &self,
        lt: &CublasLt,
        activation: &Nvfp4Matrix,
        batch_rows: usize,
        workspace_limit: u64,
    ) -> Result<Fp4TnMatmulPlan> {
        let Some(weight) = self.nvfp4_prefill_weight.as_ref() else {
            return Err(Error::Format {
                label: "ternary g64 NVFP4 prefill linear",
                detail: "matrix has no NVFP4 prefill weights".to_string(),
            });
        };
        Fp4TnMatmulPlan::new_f32_output_for_shape(
            lt,
            GemmShape::new(self.rows, batch_rows, self.cols),
            Nvfp4TnInputs::new(weight.matrix(), activation),
            workspace_limit,
        )
    }

    /// Quantizes activations and enqueues the tensor-core W4A4 prefill projection.
    #[allow(clippy::too_many_arguments)]
    pub fn run_f32_batch_nvfp4_into_on_stream(
        &self,
        lt: &CublasLt,
        plan: &Fp4TnMatmulPlan,
        input: DeviceInput<'_, f32>,
        activation: &mut Nvfp4Matrix,
        output: DeviceInOut<'_, f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let expected_shape = GemmShape::new(self.rows, batch_rows, self.cols);
        if batch_rows == 0
            || input.len() != batch_rows * self.cols
            || activation.shape().rows != self.cols
            || activation.shape().cols != batch_rows
            || output.len() < batch_rows * self.rows
            || plan.shape() != expected_shape
        {
            return Err(Error::Shape {
                label: "ternary g64 NVFP4 prefill linear",
                expected: format!(
                    "shape={expected_shape:?}, input={} values, activation={}x{}, output >= {} values",
                    batch_rows * self.cols,
                    self.cols,
                    batch_rows,
                    batch_rows * self.rows
                ),
                actual: format!(
                    "shape={:?} input={} activation={:?} output={}",
                    plan.shape(),
                    input.len(),
                    activation.shape(),
                    output.len()
                ),
            });
        }
        let Some(weight) = self.nvfp4_prefill_weight.as_ref() else {
            return Err(Error::Format {
                label: "ternary g64 NVFP4 prefill linear",
                detail: "matrix has no NVFP4 prefill weights".to_string(),
            });
        };
        weight.quantize_activation_device_col_major_f32_into_on_stream(
            self.cols,
            batch_rows,
            input.buffer(),
            activation,
            stream,
        )?;
        plan.run_with_alpha_beta_f32_inout_buffer_on_stream(
            lt,
            Nvfp4TnInputs::new(weight.matrix(), activation),
            output,
            weight.matmul_alpha(),
            0.0,
            stream,
        )
    }

    /// Converts activations to BF16 and enqueues the tensor-core prefill projection.
    #[allow(clippy::too_many_arguments)]
    pub fn run_f32_batch_bf16_into_on_stream(
        &self,
        lt: &CublasLt,
        plan: &Bf16TnMatmulPlan,
        input: DeviceInput<'_, f32>,
        input_bf16: &mut DeviceBuffer<u16>,
        output: DeviceOutput<'_, f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let expected_shape = GemmShape::new(self.rows, batch_rows, self.cols);
        if batch_rows == 0
            || input.len() != batch_rows * self.cols
            || input_bf16.len() < input.len()
            || output.len() != batch_rows * self.rows
            || plan.gemm_shape() != expected_shape
        {
            return Err(Error::Shape {
                label: "ternary g64 BF16 prefill linear",
                expected: format!(
                    "shape={expected_shape:?}, input={} values, BF16 input >= {}, output={} values",
                    batch_rows * self.cols,
                    batch_rows * self.cols,
                    batch_rows * self.rows
                ),
                actual: format!(
                    "shape={:?} input={} BF16 input={} output={}",
                    plan.gemm_shape(),
                    input.len(),
                    input_bf16.len(),
                    output.len()
                ),
            });
        }
        let Some(weight) = self.bf16_prefill_weight.as_ref() else {
            return Err(Error::Format {
                label: "ternary g64 BF16 prefill linear",
                detail: "matrix has no BF16 prefill weights".to_string(),
            });
        };
        f32_to_bf16_into_on_stream(input.buffer(), input_bf16.output(), stream)?;
        plan.run_on_stream(lt, weight, input_bf16, output, stream)
    }

    /// Quantizes `input` per group and enqueues the direct W2A8 projection.
    pub fn run_f32_batch_into_on_stream(
        &self,
        input: DeviceInput<'_, f32>,
        mut output: DeviceOutput<'_, f32>,
        batch_rows: usize,
        workspace: &mut TernaryG64ActivationWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if batch_rows == 0
            || input.len() != batch_rows * self.cols
            || output.len() != batch_rows * self.rows
        {
            return Err(Error::Shape {
                label: "ternary g64 W2A8 linear",
                expected: format!(
                    "batch_rows > 0, input={} values, output={} values",
                    batch_rows * self.cols,
                    batch_rows * self.rows
                ),
                actual: format!(
                    "batch_rows={batch_rows} input={} output={}",
                    input.len(),
                    output.len()
                ),
            });
        }
        workspace.validate(batch_rows, self.cols)?;
        let batch_rows = u32::try_from(batch_rows).map_err(dimension_overflow("batch rows"))?;
        let rows = u32::try_from(self.rows).map_err(dimension_overflow("rows"))?;
        let cols = u32::try_from(self.cols).map_err(dimension_overflow("columns"))?;
        unsafe {
            check_cuda(
                "infer_ternary_g64_quantize_i8_f32_on_stream",
                ffi::infer_ternary_g64_quantize_i8_f32_on_stream(
                    input.as_const_ptr().cast(),
                    workspace.quantized.ptr,
                    workspace.dequant_scales.ptr,
                    batch_rows,
                    cols,
                    stream.as_raw(),
                ),
            )?;
            check_cuda(
                "infer_ternary_g64_w2a8_linear_f32_on_stream",
                ffi::infer_ternary_g64_w2a8_linear_f32_on_stream(
                    workspace.quantized.ptr,
                    workspace.dequant_scales.ptr,
                    self.packed_weight.ptr,
                    self.group_scales.ptr,
                    output.as_mut_ptr().cast(),
                    batch_rows,
                    rows,
                    cols,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Expands selected matrix rows into dense F32 outputs on the device.
    pub fn lookup_rows_f32_into_on_stream(
        &self,
        row_indices: &DeviceBuffer<u32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let batch_rows = row_indices.len();
        if batch_rows == 0 || output.len() != batch_rows * self.cols {
            return Err(Error::Shape {
                label: "ternary g64 row lookup",
                expected: format!(
                    "positive row count and {} output values",
                    batch_rows * self.cols
                ),
                actual: format!("rows={batch_rows} output={}", output.len()),
            });
        }
        let batch_rows = u32::try_from(batch_rows).map_err(dimension_overflow("batch rows"))?;
        let rows = u32::try_from(self.rows).map_err(dimension_overflow("rows"))?;
        let cols = u32::try_from(self.cols).map_err(dimension_overflow("columns"))?;
        unsafe {
            check_cuda(
                "infer_ternary_g64_lookup_rows_f32_on_stream",
                ffi::infer_ternary_g64_lookup_rows_f32_on_stream(
                    self.packed_weight.ptr,
                    self.group_scales.ptr,
                    row_indices.ptr,
                    output.as_mut_ptr().cast(),
                    batch_rows,
                    rows,
                    cols,
                    stream.as_raw(),
                ),
            )
        }
    }
}

/// Reusable per-group INT8 activation storage.
pub struct TernaryG64ActivationWorkspace {
    batch_rows: usize,
    cols: usize,
    quantized: DeviceBuffer<i8>,
    dequant_scales: DeviceBuffer<f32>,
}

impl TernaryG64ActivationWorkspace {
    /// Allocates workspace for `batch_rows` vectors of width `cols`.
    pub fn new(batch_rows: usize, cols: usize) -> Result<Self> {
        if batch_rows == 0 || cols == 0 || !cols.is_multiple_of(TERNARY_G64_GROUP_SIZE) {
            return Err(Error::Shape {
                label: "ternary g64 activation workspace",
                expected: "positive batch rows and K divisible by 64".to_string(),
                actual: format!("batch_rows={batch_rows} cols={cols}"),
            });
        }
        Ok(Self {
            batch_rows,
            cols,
            quantized: DeviceBuffer::zeroed(batch_rows * cols)?,
            dequant_scales: DeviceBuffer::zeroed(batch_rows * (cols / TERNARY_G64_GROUP_SIZE))?,
        })
    }

    /// Device bytes occupied by quantized values and group scales.
    pub fn device_bytes(&self) -> usize {
        self.quantized.device_bytes() + self.dequant_scales.device_bytes()
    }

    fn validate(&self, batch_rows: usize, cols: usize) -> Result<()> {
        if self.batch_rows != batch_rows || self.cols != cols {
            return Err(Error::Shape {
                label: "ternary g64 activation workspace",
                expected: format!("batch_rows={} cols={}", self.batch_rows, self.cols),
                actual: format!("batch_rows={batch_rows} cols={cols}"),
            });
        }
        Ok(())
    }
}

fn validate_shape(out_features: usize, in_features: usize) -> Result<()> {
    if out_features == 0 || in_features == 0 || !in_features.is_multiple_of(TERNARY_G64_GROUP_SIZE)
    {
        return Err(Error::Shape {
            label: "ternary g64 linear",
            expected: "positive output size and input size divisible by 64".to_string(),
            actual: format!("out_features={out_features} in_features={in_features}"),
        });
    }
    Ok(())
}

fn shape_overflow(detail: &'static str) -> Error {
    Error::Shape {
        label: "ternary g64 shape",
        expected: format!("{detail} without overflow"),
        actual: "dimension overflow".to_string(),
    }
}

fn dimension_overflow(label: &'static str) -> impl Fn(std::num::TryFromIntError) -> Error {
    move |_| Error::Shape {
        label: "ternary g64 CUDA dimensions",
        expected: format!("{label} fitting u32"),
        actual: "dimension overflow".to_string(),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    match exponent {
        0 if mantissa == 0 => sign * 0.0,
        0 => sign * (mantissa as f32) * 2.0f32.powi(-24),
        0x1f if mantissa == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + mantissa as f32 / 1024.0) * 2.0f32.powi(exponent as i32 - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_gguf(rows: usize, cols: usize) -> Vec<u8> {
        let groups = rows * (cols / TERNARY_G64_GROUP_SIZE);
        let mut raw = Vec::with_capacity(groups * GGUF_BYTES_PER_GROUP);
        for group in 0..groups {
            let scale_bits = [0x3800u16, 0x3400, 0x3c00][group % 3];
            raw.extend_from_slice(&scale_bits.to_le_bytes());
            for packed in 0..PACKED_BYTES_PER_GROUP {
                let mut byte = 0u8;
                for within in 0..4 {
                    let code = ((group * 7 + packed * 3 + within) % 3) as u8;
                    byte |= code << (within * 2);
                }
                raw.push(byte);
            }
        }
        raw
    }

    #[test]
    fn imports_mainline_q2_0_g64_blocks_without_dequantizing_codes() {
        let linear =
            TernaryG64PackedLinear::from_gguf_q2_0_g64("test", 2, 128, &synthetic_gguf(2, 128))
                .expect("import");
        assert_eq!(linear.group_scales, [0.5, 0.25, 1.0, 0.5]);
        assert_eq!(linear.packed_weight.len(), 64);
        for row in 0..2 {
            for col in 0..128 {
                let group = row * 2 + col / TERNARY_G64_GROUP_SIZE;
                let packed = (col % TERNARY_G64_GROUP_SIZE) / 4;
                let within = col % 4;
                let code = ((group * 7 + packed * 3 + within) % 3) as i32;
                let expected = (code - 1) as f32 * linear.group_scales[group];
                assert_eq!(linear.weight(row, col).expect("weight"), expected);
            }
        }
    }

    #[test]
    fn rejects_nonternary_code_three() {
        let mut raw = synthetic_gguf(1, 64);
        raw[2] = 3;
        let error = TernaryG64PackedLinear::from_gguf_q2_0_g64("test", 1, 64, &raw)
            .expect_err("reserved code");
        assert!(error.to_string().contains("reserved code 3"));
    }

    #[test]
    fn concatenates_rows_without_repacking_groups() {
        let left =
            TernaryG64PackedLinear::from_gguf_q2_0_g64("left", 2, 64, &synthetic_gguf(2, 64))
                .expect("left");
        let right =
            TernaryG64PackedLinear::from_gguf_q2_0_g64("right", 3, 64, &synthetic_gguf(3, 64))
                .expect("right");
        let joined = TernaryG64PackedLinear::concat_rows("joined", &[left.clone(), right.clone()])
            .expect("join");
        assert_eq!(joined.out_features, 5);
        assert_eq!(joined.in_features, 64);
        assert_eq!(
            &joined.packed_weight[..left.packed_weight.len()],
            &left.packed_weight
        );
        assert_eq!(
            &joined.packed_weight[left.packed_weight.len()..],
            &right.packed_weight
        );
    }

    #[test]
    fn gpu_row_lookup_matches_imported_weights() {
        const ROWS: usize = 7;
        const COLS: usize = 128;
        let packed = TernaryG64PackedLinear::from_gguf_q2_0_g64(
            "lookup",
            ROWS,
            COLS,
            &synthetic_gguf(ROWS, COLS),
        )
        .expect("import");
        let matrix = TernaryG64Matrix::from_packed(&packed).expect("upload");
        let indices = DeviceBuffer::from_host(&[6u32, 1, 4]).expect("indices");
        let mut output = DeviceBuffer::zeroed(3 * COLS).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        matrix
            .lookup_rows_f32_into_on_stream(&indices, output.output(), &stream)
            .expect("lookup");
        let actual = output.copy_to_host(&stream).expect("download");
        for (batch, row) in [6usize, 1, 4].into_iter().enumerate() {
            for col in 0..COLS {
                assert_eq!(actual[batch * COLS + col], packed.weight(row, col).unwrap());
            }
        }
    }

    #[test]
    fn gpu_matches_group_scaled_integer_reference_at_bonsai_hidden_shape() {
        const ROWS: usize = 4_096;
        const COLS: usize = 4_096;
        const BATCH: usize = 2;
        let packed = TernaryG64PackedLinear::from_gguf_q2_0_g64(
            "bonsai.hidden",
            ROWS,
            COLS,
            &synthetic_gguf(ROWS, COLS),
        )
        .expect("import");
        let input_host = (0..BATCH * COLS)
            .map(|index| ((index * 37 % 509) as f32 - 254.0) / 73.0)
            .collect::<Vec<_>>();
        let expected = packed
            .reference_w2a8(&input_host, BATCH)
            .expect("reference");
        let matrix = TernaryG64Matrix::from_packed(&packed).expect("upload");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut output = DeviceBuffer::zeroed(BATCH * ROWS).expect("output");
        let mut workspace = TernaryG64ActivationWorkspace::new(BATCH, COLS).expect("workspace");
        let stream = CudaStream::new_non_blocking().expect("stream");
        matrix
            .run_f32_batch_into_on_stream(
                input.input(),
                output.output(),
                BATCH,
                &mut workspace,
                &stream,
            )
            .expect("projection");
        let quantized = workspace
            .quantized
            .copy_to_host(&stream)
            .expect("quantized download");
        let scales = workspace
            .dequant_scales
            .copy_to_host(&stream)
            .expect("scale download");
        for batch in 0..BATCH {
            for group in 0..COLS / TERNARY_G64_GROUP_SIZE {
                let start = batch * COLS + group * TERNARY_G64_GROUP_SIZE;
                let values = &input_host[start..start + TERNARY_G64_GROUP_SIZE];
                let maximum = values
                    .iter()
                    .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                assert_eq!(
                    scales[batch * (COLS / TERNARY_G64_GROUP_SIZE) + group],
                    maximum / 127.0
                );
                let quantize_scale = if maximum == 0.0 { 0.0 } else { 127.0 / maximum };
                for (offset, value) in values.iter().enumerate() {
                    let expected = (value * quantize_scale)
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i8;
                    assert_eq!(quantized[start + offset], expected);
                }
            }
        }
        let actual = output.copy_to_host(&stream).expect("download");
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let rmse = (actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).powi(2) as f64)
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let scale = (expected
            .iter()
            .map(|value| value.powi(2) as f64)
            .sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        let relative_rmse = rmse / scale.max(f64::EPSILON);
        assert!(
            relative_rmse <= 2.0e-5,
            "Bonsai g64 W2A8 max_abs={max_abs} relative_rmse={relative_rmse} scale={scale}"
        );
    }

    #[test]
    fn tiled_gpu_projection_matches_reference_with_batch_and_row_tails() {
        const ROWS: usize = 37;
        const COLS: usize = 192;
        const BATCH: usize = 11;
        let packed = TernaryG64PackedLinear::from_gguf_q2_0_g64(
            "tiled.tails",
            ROWS,
            COLS,
            &synthetic_gguf(ROWS, COLS),
        )
        .expect("import");
        let input_host = (0..BATCH * COLS)
            .map(|index| ((index * 53 % 613) as f32 - 306.0) / 91.0)
            .collect::<Vec<_>>();
        let expected = packed
            .reference_w2a8(&input_host, BATCH)
            .expect("reference");
        let matrix = TernaryG64Matrix::from_packed(&packed).expect("upload");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut output = DeviceBuffer::zeroed(BATCH * ROWS).expect("output");
        let mut workspace = TernaryG64ActivationWorkspace::new(BATCH, COLS).expect("workspace");
        let stream = CudaStream::new_non_blocking().expect("stream");
        matrix
            .run_f32_batch_into_on_stream(
                input.input(),
                output.output(),
                BATCH,
                &mut workspace,
                &stream,
            )
            .expect("projection");
        let actual = output.copy_to_host(&stream).expect("download");
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let rmse = (actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).powi(2) as f64)
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let scale = (expected
            .iter()
            .map(|value| value.powi(2) as f64)
            .sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        let relative_rmse = rmse / scale.max(f64::EPSILON);
        assert!(
            relative_rmse <= 2.0e-5,
            "tiled g64 W2A8 max_abs={max_abs} relative_rmse={relative_rmse} scale={scale}"
        );
    }

    #[test]
    fn bf16_tensor_prefill_matches_rounded_dense_reference() {
        const ROWS: usize = 96;
        const COLS: usize = 128;
        const BATCH: usize = 7;
        let packed = TernaryG64PackedLinear::from_gguf_q2_0_g64(
            "bf16.prefill",
            ROWS,
            COLS,
            &synthetic_gguf(ROWS, COLS),
        )
        .expect("import");
        let input_host = (0..BATCH * COLS)
            .map(|index| ((index * 43 % 557) as f32 - 278.0) / 83.0)
            .collect::<Vec<_>>();
        let expected = (0..BATCH)
            .flat_map(|batch| {
                let packed = &packed;
                let input_host = &input_host;
                (0..ROWS).map(move |row| {
                    (0..COLS)
                        .map(|col| {
                            let input = crate::format::bf16_to_f32(crate::format::f32_to_bf16(
                                input_host[batch * COLS + col],
                            ));
                            let weight = crate::format::bf16_to_f32(crate::format::f32_to_bf16(
                                packed.weight(row, col).unwrap(),
                            ));
                            input * weight
                        })
                        .sum::<f32>()
                })
            })
            .collect::<Vec<_>>();
        let matrix = TernaryG64Matrix::from_packed_with_bf16_prefill(&packed).expect("upload");
        let lt = CublasLt::new().expect("cuBLASLt");
        let plan = Bf16TnMatmulPlan::new(&lt, GemmShape::new(ROWS, BATCH, COLS), 8 * 1024 * 1024)
            .expect("plan");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut input_bf16 = DeviceBuffer::zeroed(BATCH * COLS).expect("BF16 input");
        let mut output = DeviceBuffer::zeroed(BATCH * ROWS).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        matrix
            .run_f32_batch_bf16_into_on_stream(
                &lt,
                &plan,
                input.input(),
                &mut input_bf16,
                output.output(),
                BATCH,
                &stream,
            )
            .expect("projection");
        let actual = output.copy_to_host(&stream).expect("download");
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 0.002 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "BF16 prefill value {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
        let packed_bytes = ROWS * COLS / 4 + ROWS * (COLS / 64) * std::mem::size_of::<f32>();
        assert_eq!(matrix.device_bytes(), packed_bytes + ROWS * COLS * 2);
    }

    #[test]
    fn nvfp4_tensor_prefill_tracks_ternary_dense_reference_for_unaligned_batch() {
        const ROWS: usize = 128;
        const COLS: usize = 128;
        const BATCH: usize = 7;
        let packed = TernaryG64PackedLinear::from_gguf_q2_0_g64(
            "nvfp4.prefill",
            ROWS,
            COLS,
            &synthetic_gguf(ROWS, COLS),
        )
        .expect("import");
        let input_host = (0..BATCH * COLS)
            .map(|index| ((index * 43 % 557) as f32 - 278.0) / 83.0)
            .collect::<Vec<_>>();
        let expected = (0..BATCH)
            .flat_map(|batch| {
                let packed = &packed;
                let input_host = &input_host;
                (0..ROWS).map(move |row| {
                    (0..COLS)
                        .map(|col| {
                            input_host[batch * COLS + col] * packed.weight(row, col).unwrap()
                        })
                        .sum::<f32>()
                })
            })
            .collect::<Vec<_>>();
        let matrix = TernaryG64Matrix::from_packed_with_nvfp4_prefill(&packed).expect("upload");
        let lt = CublasLt::new().expect("cuBLASLt");
        let mut activation = Nvfp4Matrix::zeroed_col_major(COLS, BATCH).expect("NVFP4 activation");
        let plan = matrix
            .new_f32_batch_nvfp4_plan(&lt, &activation, BATCH, 8 * 1024 * 1024)
            .expect("plan");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut output = DeviceBuffer::zeroed(BATCH * ROWS).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        matrix
            .run_f32_batch_nvfp4_into_on_stream(
                &lt,
                &plan,
                input.input(),
                &mut activation,
                output.inout(),
                BATCH,
                &stream,
            )
            .expect("projection");
        let actual = output.copy_to_host(&stream).expect("download");
        let (mut dot, mut expected_squared, mut actual_squared, mut error_squared) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&actual, &expected) in actual.iter().zip(&expected) {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            dot += actual * expected;
            expected_squared += expected * expected;
            actual_squared += actual * actual;
            error_squared += (actual - expected).powi(2);
        }
        let cosine = dot / (expected_squared.sqrt() * actual_squared.sqrt());
        let nrmse = (error_squared / expected_squared).sqrt();
        assert!(
            cosine >= 0.97 && nrmse <= 0.35,
            "NVFP4 prefill cosine={cosine} nrmse={nrmse}"
        );

        let packed_bytes = ROWS * COLS / 4 + ROWS * (COLS / 64) * size_of::<f32>();
        let nvfp4_bytes = ROWS * COLS / 2 + ROWS * (COLS / 16);
        assert_eq!(matrix.device_bytes(), packed_bytes + nvfp4_bytes);
    }
}
