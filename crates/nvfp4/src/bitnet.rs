//! Checkpoint-exact ternary weights and W2A8 execution for BitNet.
//!
//! Hugging Face BitNet checkpoints pack four output rows into each byte of a
//! `[out / 4, in]` U8 tensor. Eider transcodes that layout once into rows with
//! four consecutive input-channel weights per byte. Runtime activations use
//! per-row symmetric INT8 quantization and the CUDA kernel accumulates exact
//! integer dot products with `dp4a` before applying both scales.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInput, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::modelopt::ModelOptCheckpoint;

/// Host-side row-major packed ternary linear.
#[derive(Clone, Debug)]
pub struct BitNetPackedLinear {
    /// Tensor prefix, for example `model.layers.0.self_attn.q_proj`.
    pub prefix: String,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
    /// Four K-consecutive ternary weights per byte, row-major.
    pub packed_weight: Vec<u8>,
    /// One checkpoint weight scale per output row.
    pub row_scales: Vec<f32>,
}

impl BitNetPackedLinear {
    /// Loads and transcodes one offline-packed BitNet linear.
    pub fn from_checkpoint(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_shard = checkpoint.open_shard_for_tensor(&weight_name)?;
        let scale_shard = checkpoint.open_shard_for_tensor(&scale_name)?;
        let weight_info = weight_shard.require_tensor(&weight_name)?;
        if weight_info.dtype != "U8" || weight_info.shape != [out_features / 4, in_features] {
            return Err(Error::Shape {
                label: "BitNet packed weight",
                expected: format!(
                    "{weight_name} dtype=U8 shape=[{}, {in_features}]",
                    out_features / 4
                ),
                actual: format!("dtype={} shape={:?}", weight_info.dtype, weight_info.shape),
            });
        }
        let scale_info = scale_shard.require_tensor(&scale_name)?;
        let scales = scale_shard.read_float_tensor_as_f32(&scale_name)?;
        if scales.len() != 1 {
            return Err(Error::Shape {
                label: "BitNet weight scale",
                expected: format!("{scale_name} with one F32 or BF16 value"),
                actual: format!("dtype={} shape={:?}", scale_info.dtype, scale_info.shape),
            });
        }
        let scale = scales[0];
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Format {
                label: "BitNet weight scale",
                detail: format!("{scale_name} must be positive and finite, got {scale}"),
            });
        }
        Self::from_hf_packed(
            prefix,
            out_features,
            in_features,
            &weight_shard.read_tensor_bytes(&weight_name)?,
            scale,
        )
    }

    /// Transcodes Hugging Face output-packed bytes into Eider's K-packed rows.
    pub fn from_hf_packed(
        prefix: impl Into<String>,
        out_features: usize,
        in_features: usize,
        hf_packed: &[u8],
        weight_scale: f32,
    ) -> Result<Self> {
        validate_shape(out_features, in_features)?;
        let expected = out_features
            .checked_mul(in_features)
            .and_then(|values| values.checked_div(4))
            .ok_or_else(|| Error::Shape {
                label: "BitNet packed weight",
                expected: "out_features * in_features / 4 without overflow".to_string(),
                actual: format!("out_features={out_features} in_features={in_features}"),
            })?;
        if hf_packed.len() != expected || !weight_scale.is_finite() || weight_scale <= 0.0 {
            return Err(Error::Shape {
                label: "BitNet packed weight",
                expected: format!("{expected} bytes and a positive finite scale"),
                actual: format!("{} bytes, scale={weight_scale}", hf_packed.len()),
            });
        }

        let packed_rows = out_features / 4;
        let row_bytes = in_features / 4;
        let mut packed_weight = vec![0u8; expected];
        for row in 0..out_features {
            let source_row = row % packed_rows;
            let pair = row / packed_rows;
            for packed_col in 0..row_bytes {
                let mut output = 0u8;
                for within in 0..4 {
                    let source = hf_packed[source_row * in_features + packed_col * 4 + within];
                    let code = (source >> (pair * 2)) & 0x03;
                    if code == 3 {
                        return Err(Error::Format {
                            label: "BitNet packed weight",
                            detail: format!(
                                "reserved ternary code 3 at output row {row}, input column {}",
                                packed_col * 4 + within
                            ),
                        });
                    }
                    output |= code << (within * 2);
                }
                packed_weight[row * row_bytes + packed_col] = output;
            }
        }
        Ok(Self {
            prefix: prefix.into(),
            out_features,
            in_features,
            packed_weight,
            row_scales: vec![weight_scale; out_features],
        })
    }

    /// Concatenates equal-width linears along the output dimension.
    pub fn concat_rows(prefix: impl Into<String>, linears: &[Self]) -> Result<Self> {
        let Some(first) = linears.first() else {
            return Err(Error::Shape {
                label: "BitNet linear concatenation",
                expected: "at least one linear".to_string(),
                actual: "zero linears".to_string(),
            });
        };
        if linears
            .iter()
            .any(|linear| linear.in_features != first.in_features)
        {
            return Err(Error::Shape {
                label: "BitNet linear concatenation",
                expected: format!("all inputs have {} columns", first.in_features),
                actual: linears
                    .iter()
                    .map(|linear| linear.in_features.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        let out_features = linears.iter().map(|linear| linear.out_features).sum();
        let mut packed_weight = Vec::new();
        let mut row_scales = Vec::with_capacity(out_features);
        for linear in linears {
            packed_weight.extend_from_slice(&linear.packed_weight);
            row_scales.extend_from_slice(&linear.row_scales);
        }
        Ok(Self {
            prefix: prefix.into(),
            out_features,
            in_features: first.in_features,
            packed_weight,
            row_scales,
        })
    }

    /// Returns the represented ternary weight at `(row, col)`.
    pub fn weight(&self, row: usize, col: usize) -> Result<f32> {
        if row >= self.out_features || col >= self.in_features {
            return Err(Error::Shape {
                label: "BitNet weight index",
                expected: format!("row < {} and col < {}", self.out_features, self.in_features),
                actual: format!("row={row} col={col}"),
            });
        }
        let byte = self.packed_weight[row * (self.in_features / 4) + col / 4];
        let code = (byte >> ((col % 4) * 2)) & 0x03;
        Ok((i32::from(code) - 1) as f32 * self.row_scales[row])
    }

    /// Computes the checkpoint-exact W2A8 operation on the CPU.
    pub fn reference_f32(&self, input: &[f32], batch_rows: usize) -> Result<Vec<f32>> {
        if batch_rows == 0 || input.len() != batch_rows * self.in_features {
            return Err(Error::Shape {
                label: "BitNet reference input",
                expected: format!("{} values", batch_rows * self.in_features),
                actual: format!("{} values", input.len()),
            });
        }
        let mut output = vec![0.0f32; batch_rows * self.out_features];
        for batch in 0..batch_rows {
            let row_input = &input[batch * self.in_features..(batch + 1) * self.in_features];
            let max_abs = row_input
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let dequant_scale = max_abs / 127.0;
            let quantized = row_input
                .iter()
                .map(|value| {
                    if max_abs == 0.0 {
                        0i32
                    } else {
                        (value * 127.0 / max_abs).round().clamp(-127.0, 127.0) as i32
                    }
                })
                .collect::<Vec<_>>();
            for row in 0..self.out_features {
                let mut sum = 0i32;
                for (col, &activation) in quantized.iter().enumerate() {
                    let byte = self.packed_weight[row * (self.in_features / 4) + col / 4];
                    let code = (byte >> ((col % 4) * 2)) & 0x03;
                    sum += (i32::from(code) - 1) * activation;
                }
                output[batch * self.out_features + row] =
                    sum as f32 * dequant_scale * self.row_scales[row];
            }
        }
        Ok(output)
    }
}

/// Device-resident checkpoint-exact BitNet linear.
pub struct BitNetMatrix {
    rows: usize,
    cols: usize,
    packed_weight: DeviceBuffer<u8>,
    row_scales: DeviceBuffer<f32>,
}

impl BitNetMatrix {
    /// Uploads an imported BitNet linear.
    pub fn from_packed(linear: &BitNetPackedLinear) -> Result<Self> {
        validate_shape(linear.out_features, linear.in_features)?;
        Ok(Self {
            rows: linear.out_features,
            cols: linear.in_features,
            packed_weight: DeviceBuffer::from_host(&linear.packed_weight)?,
            row_scales: DeviceBuffer::from_host(&linear.row_scales)?,
        })
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Device bytes occupied by weights and scales.
    pub fn device_bytes(&self) -> usize {
        self.packed_weight.device_bytes() + self.row_scales.device_bytes()
    }

    /// Quantizes `input` into `workspace` and computes the W2A8 batch linear.
    pub fn run_f32_batch_into_on_stream(
        &self,
        input: DeviceInput<'_, f32>,
        mut output: DeviceOutput<'_, f32>,
        batch_rows: usize,
        workspace: &mut BitNetActivationWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if batch_rows == 0
            || input.len() != batch_rows * self.cols
            || output.len() != batch_rows * self.rows
        {
            return Err(Error::Shape {
                label: "BitNet W2A8 linear",
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
                "infer_bitnet_quantize_i8_f32_on_stream",
                ffi::infer_bitnet_quantize_i8_f32_on_stream(
                    input.as_const_ptr().cast(),
                    workspace.quantized.ptr,
                    workspace.dequant_scales.ptr,
                    batch_rows,
                    cols,
                    stream.as_raw(),
                ),
            )?;
            check_cuda(
                "infer_bitnet_w2a8_linear_f32_on_stream",
                ffi::infer_bitnet_w2a8_linear_f32_on_stream(
                    workspace.quantized.ptr,
                    workspace.dequant_scales.ptr,
                    self.packed_weight.ptr,
                    self.row_scales.ptr,
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

/// Reusable per-token INT8 activation storage for a BitNet linear.
pub struct BitNetActivationWorkspace {
    batch_rows: usize,
    cols: usize,
    quantized: DeviceBuffer<i8>,
    dequant_scales: DeviceBuffer<f32>,
}

/// Applies BitNet's `ReLU(gate)^2 * up` activation to fused row-major pairs.
pub fn relu_squared_mul_halves_f32_batch_into_on_stream(
    input: DeviceInput<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    if batch_rows == 0
        || cols == 0
        || input.len() != batch_rows * cols * 2
        || output.len() != batch_rows * cols
    {
        return Err(Error::Shape {
            label: "BitNet ReLU squared gate/up",
            expected: format!(
                "batch_rows > 0, cols > 0, input={} values, output={} values",
                batch_rows * cols * 2,
                batch_rows * cols
            ),
            actual: format!(
                "batch_rows={batch_rows} cols={cols} input={} output={}",
                input.len(),
                output.len()
            ),
        });
    }
    let rows = u32::try_from(batch_rows).map_err(dimension_overflow("batch rows"))?;
    let cols = u32::try_from(cols).map_err(dimension_overflow("columns"))?;
    unsafe {
        check_cuda(
            "infer_bitnet_relu_squared_mul_halves_f32_on_stream",
            ffi::infer_bitnet_relu_squared_mul_halves_f32_on_stream(
                input.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows,
                cols,
                stream.as_raw(),
            ),
        )
    }
}

impl BitNetActivationWorkspace {
    /// Allocates workspace for `batch_rows` vectors of width `cols`.
    pub fn new(batch_rows: usize, cols: usize) -> Result<Self> {
        if batch_rows == 0 || cols == 0 || !cols.is_multiple_of(4) {
            return Err(Error::Shape {
                label: "BitNet activation workspace",
                expected: "positive batch rows and a positive K divisible by 4".to_string(),
                actual: format!("batch_rows={batch_rows} cols={cols}"),
            });
        }
        Ok(Self {
            batch_rows,
            cols,
            quantized: DeviceBuffer::zeroed(batch_rows * cols)?,
            dequant_scales: DeviceBuffer::zeroed(batch_rows)?,
        })
    }

    /// Device bytes occupied by quantized values and row scales.
    pub fn device_bytes(&self) -> usize {
        self.quantized.device_bytes() + self.dequant_scales.device_bytes()
    }

    fn validate(&self, batch_rows: usize, cols: usize) -> Result<()> {
        if self.batch_rows != batch_rows || self.cols != cols {
            return Err(Error::Shape {
                label: "BitNet activation workspace",
                expected: format!("batch_rows={} cols={}", self.batch_rows, self.cols),
                actual: format!("batch_rows={batch_rows} cols={cols}"),
            });
        }
        Ok(())
    }
}

fn validate_shape(out_features: usize, in_features: usize) -> Result<()> {
    if out_features == 0
        || in_features == 0
        || !out_features.is_multiple_of(4)
        || !in_features.is_multiple_of(4)
    {
        return Err(Error::Shape {
            label: "BitNet linear",
            expected: "positive output and input dimensions divisible by 4".to_string(),
            actual: format!("out_features={out_features} in_features={in_features}"),
        });
    }
    Ok(())
}

fn dimension_overflow(label: &'static str) -> impl Fn(std::num::TryFromIntError) -> Error {
    move |_| Error::Shape {
        label: "BitNet CUDA dimensions",
        expected: format!("{label} fitting u32"),
        actual: "dimension overflow".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn hf_pack(weights: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let packed_rows = rows / 4;
        let mut output = vec![0u8; rows * cols / 4];
        for packed_row in 0..packed_rows {
            for col in 0..cols {
                let mut byte = 0u8;
                for pair in 0..4 {
                    let row = packed_row + pair * packed_rows;
                    let code = (weights[row * cols + col] + 1) as u8;
                    byte |= code << (pair * 2);
                }
                output[packed_row * cols + col] = byte;
            }
        }
        output
    }

    #[test]
    fn transcodes_hf_output_packing_to_k_packed_rows() {
        let weights = [
            -1, 0, 1, -1, 0, 1, -1, 0, // row 0
            1, 1, 0, 0, -1, -1, 1, 0, // row 1
            0, -1, 0, 1, 1, 0, -1, 1, // row 2
            -1, -1, -1, -1, 1, 1, 1, 1, // row 3
        ];
        let linear =
            BitNetPackedLinear::from_hf_packed("test", 4, 8, &hf_pack(&weights, 4, 8), 0.25)
                .expect("transcode");
        for row in 0..4 {
            for col in 0..8 {
                assert_eq!(
                    linear.weight(row, col).expect("weight"),
                    weights[row * 8 + col] as f32 * 0.25
                );
            }
        }
    }

    #[test]
    fn reference_uses_per_token_absmax_and_integer_dot_product() {
        let weights = [
            -1, 0, 1, -1, 0, 1, -1, 0, // row 0
            1, 1, 0, 0, -1, -1, 1, 0, // row 1
            0, -1, 0, 1, 1, 0, -1, 1, // row 2
            -1, -1, -1, -1, 1, 1, 1, 1, // row 3
        ];
        let linear =
            BitNetPackedLinear::from_hf_packed("test", 4, 8, &hf_pack(&weights, 4, 8), 0.5)
                .expect("transcode");
        let input = [1.0, -0.5, 0.25, 0.0, -1.0, 0.5, 0.0, 0.25];
        let output = linear.reference_f32(&input, 1).expect("reference");
        let scale = 1.0 / 127.0;
        let quantized = [127, -64, 32, 0, -127, 64, 0, 32];
        for row in 0..4 {
            let sum = (0..8)
                .map(|col| weights[row * 8 + col] as i32 * quantized[col])
                .sum::<i32>();
            assert!((output[row] - sum as f32 * scale * 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_reserved_ternary_code() {
        let error = BitNetPackedLinear::from_hf_packed("test", 4, 4, &[3, 0, 0, 0], 1.0)
            .expect_err("reserved code must fail");
        assert!(error.to_string().contains("reserved ternary code 3"));
    }

    #[test]
    fn loads_single_file_checkpoint_with_bf16_scale() {
        let root = std::env::temp_dir().join(format!(
            "eider-bitnet-safetensors-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create checkpoint directory");
        let weights = [
            -1, 0, 1, -1, // row 0
            1, 1, 0, 0, // row 1
            0, -1, 0, 1, // row 2
            -1, -1, -1, -1, // row 3
        ];
        let packed = hf_pack(&weights, 4, 4);
        let scale = crate::format::f32_to_bf16(0.5).to_le_bytes();
        let mut header = serde_json::to_vec(&json!({
            "linear.weight": {
                "dtype": "U8",
                "shape": [1, 4],
                "data_offsets": [0, 4]
            },
            "linear.weight_scale": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [4, 6]
            }
        }))
        .expect("serialize safetensors header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + 6);
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&packed);
        file.extend_from_slice(&scale);
        fs::write(root.join("model.safetensors"), file).expect("write checkpoint");

        let checkpoint = ModelOptCheckpoint::open(&root).expect("open single-file checkpoint");
        let linear = BitNetPackedLinear::from_checkpoint(&checkpoint, "linear", 4, 4)
            .expect("load BitNet linear");
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    linear.weight(row, col).expect("weight"),
                    weights[row * 4 + col] as f32 * 0.5
                );
            }
        }
        fs::remove_dir_all(root).expect("remove checkpoint directory");
    }
}
