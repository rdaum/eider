//! CUDA preparation for host-side ModelOpt records.
//!
//! `eider-format` owns ModelOpt checkpoint decoding and host layouts. This
//! module performs the explicit conversion into CUDA allocations and
//! cuBLASLt-compatible layouts.

use crate::error::{Error, Result};
use crate::format;
use crate::kernels::non_gemm::{
    quantize_nvfp4_col_major_f32_device, quantize_nvfp4_col_major_f32_device_into_on_stream,
};
use crate::matrix::Nvfp4Matrix;
use crate::{CudaStream, DeviceBuffer};

pub use eider_format::{
    ModelOptBlockScaledFp8Linear, ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear,
    modelopt_scales_to_cublaslt,
};

/// cuBLASLt-ready CUDA preparation of one ModelOpt NVFP4 weight.
pub struct ModelOptCublasLtWeight {
    matrix: Nvfp4Matrix,
    weight_scale_2: f32,
    input_scale: f32,
}

/// CUDA-resident NVFP4 activation operand prepared for one ModelOpt linear.
pub struct ModelOptNvfp4Activation {
    matrix: Nvfp4Matrix,
    input_scale: f32,
    dequantized_scaled_values: Vec<f32>,
}

impl ModelOptNvfp4Activation {
    /// Quantizes a column-major host activation into a CUDA NVFP4 matrix.
    pub fn quantize_col_major_f32(
        rows: usize,
        cols: usize,
        values: &[f32],
        input_scale: f32,
    ) -> Result<Self> {
        if !input_scale.is_finite() || input_scale <= 0.0 {
            return Err(Error::Format {
                label: "ModelOpt activation input_scale",
                detail: format!("expected positive finite scale, got {input_scale}"),
            });
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
            label: "ModelOpt activation",
            expected: "rows * columns without overflow".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
        if values.len() != expected {
            return Err(Error::Shape {
                label: "ModelOpt activation",
                expected: format!("{expected} values"),
                actual: format!("{} values", values.len()),
            });
        }
        let scaled = values
            .iter()
            .map(|value| value / input_scale)
            .collect::<Vec<_>>();
        let quantized = format::quantize_nvfp4_col_major(rows, cols, &scaled);
        Ok(Self {
            matrix: Nvfp4Matrix::from_packed_col_major_parts(
                rows,
                cols,
                &quantized.packed_values,
                &quantized.scales,
            )?,
            input_scale,
            dequantized_scaled_values: quantized.dequantized_values,
        })
    }

    /// Quantizes a device-resident column-major activation.
    pub fn quantize_device_col_major_f32(
        rows: usize,
        cols: usize,
        values: &DeviceBuffer<f32>,
        input_scale: f32,
    ) -> Result<Self> {
        Ok(Self {
            matrix: quantize_nvfp4_col_major_f32_device(rows, cols, values, input_scale)?,
            input_scale,
            dequantized_scaled_values: Vec::new(),
        })
    }

    /// Returns the prepared CUDA matrix.
    pub fn matrix(&self) -> &Nvfp4Matrix {
        &self.matrix
    }

    /// Returns the static ModelOpt activation scale.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Returns CPU reference values for host-originated quantization.
    pub fn dequantized_scaled_values(&self) -> &[f32] {
        &self.dequantized_scaled_values
    }

    /// Returns CUDA allocation bytes owned by this activation.
    pub fn device_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }
}

impl ModelOptCublasLtWeight {
    /// Builds a prepared ModelOpt weight from an already-uploaded matrix.
    pub fn from_matrix(matrix: Nvfp4Matrix, weight_scale_2: f32, input_scale: f32) -> Result<Self> {
        if !weight_scale_2.is_finite() || !input_scale.is_finite() || input_scale <= 0.0 {
            return Err(Error::Format {
                label: "ModelOpt cuBLASLt scalar scales",
                detail: format!("weight_scale_2={weight_scale_2} input_scale={input_scale}"),
            });
        }
        Ok(Self {
            matrix,
            weight_scale_2,
            input_scale,
        })
    }

    /// Uploads and prepares one host ModelOpt NVFP4 record.
    pub fn from_modelopt(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        let (rows, cols) = weight.tn_weight_shape();
        let matrix = Nvfp4Matrix::from_packed_col_major_parts(
            rows,
            cols,
            &weight.packed_weight,
            &weight.cublaslt_scales(),
        )?;
        Self::from_matrix(matrix, weight.weight_scale_2, weight.input_scale)
    }

    /// Returns the cuBLASLt-compatible CUDA matrix.
    pub fn matrix(&self) -> &Nvfp4Matrix {
        &self.matrix
    }

    /// Returns ModelOpt's tensor-wide weight scale.
    pub fn weight_scale_2(&self) -> f32 {
        self.weight_scale_2
    }

    /// Returns ModelOpt's static activation scale.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Returns the required ModelOpt GEMM alpha.
    pub fn matmul_alpha(&self) -> f32 {
        self.weight_scale_2 * self.input_scale
    }

    /// Quantizes a host activation for this weight.
    pub fn quantize_activation_col_major_f32(
        &self,
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> Result<ModelOptNvfp4Activation> {
        ModelOptNvfp4Activation::quantize_col_major_f32(rows, cols, values, self.input_scale)
    }

    /// Quantizes a CUDA activation for this weight.
    pub fn quantize_activation_device_col_major_f32(
        &self,
        rows: usize,
        cols: usize,
        values: &DeviceBuffer<f32>,
    ) -> Result<ModelOptNvfp4Activation> {
        ModelOptNvfp4Activation::quantize_device_col_major_f32(rows, cols, values, self.input_scale)
    }

    /// Enqueues CUDA activation quantization into preallocated storage.
    pub fn quantize_activation_device_col_major_f32_into_on_stream(
        &self,
        rows: usize,
        cols: usize,
        values: &DeviceBuffer<f32>,
        output: &mut Nvfp4Matrix,
        stream: &CudaStream,
    ) -> Result<()> {
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            rows,
            cols,
            values,
            output,
            self.input_scale,
            stream,
        )
    }

    /// Returns CUDA allocation bytes owned by this prepared weight.
    pub fn device_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }
}

/// Uploads a host record in the grouped layout used by the SIMT NVFP4 GEMV kernel.
pub fn upload_grouped_nvfp4(
    weight: &ModelOptNvfp4Linear,
) -> Result<(DeviceBuffer<u8>, DeviceBuffer<u8>)> {
    let (rows, cols) = weight.tn_weight_shape();
    let grouped =
        format::quantize_nvfp4_4row_groups(rows, cols, &weight.dequantize_to_f32_col_major());
    Ok((
        DeviceBuffer::from_host(&grouped.packed_values)?,
        DeviceBuffer::from_host(&grouped.group_scales)?,
    ))
}
