//! Import helpers for NVIDIA ModelOpt NVFP4 safetensors checkpoints.
//!
//! ModelOpt stores each quantized linear as a small group of tensors:
//!
//! - `<prefix>.weight`: packed FP4 E2M1 bytes with shape `[out, in / 2]`;
//! - `<prefix>.weight_scale`: E4M3 scale bytes with shape `[out, in / 16]`;
//! - `<prefix>.weight_scale_2`: scalar F32 tensor-wide scale;
//! - `<prefix>.input_scale`: scalar F32 calibrated activation scale.
//!
//! The packed value layout is compatible with the current cuBLASLt TN weight
//! convention when interpreted as a column-major `K x M` matrix with `K=in`
//! and `M=out`: one ModelOpt output row is one cuBLASLt weight column. The
//! scale tensor is not in cuBLASLt's tiled `VEC16_UE4M3` layout, so it must be
//! repacked before use with [`crate::Nvfp4Matrix`].

use crate::error::{Error, Result};
use crate::format;
use crate::kernels::non_gemm::{
    quantize_nvfp4_col_major_f32_device, quantize_nvfp4_col_major_f32_device_into_on_stream,
};
use crate::matrix::Nvfp4Matrix;
use crate::safetensors::{SafeTensorInfo, SafeTensorShard};
use crate::{CudaStream, DeviceBuffer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Metadata and raw host bytes for one ModelOpt NVFP4 linear weight.
#[derive(Clone, Debug)]
pub struct ModelOptNvfp4Linear {
    /// Tensor name prefix, for example `model.layers.0.self_attn.q_proj`.
    pub prefix: String,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
    /// Packed E2M1 bytes from `<prefix>.weight`.
    pub packed_weight: Vec<u8>,
    /// ModelOpt linear E4M3 scale bytes from `<prefix>.weight_scale`.
    pub weight_scale: Vec<u8>,
    /// Tensor-wide ModelOpt weight scale.
    pub weight_scale_2: f32,
    /// Static calibrated activation scale.
    pub input_scale: f32,
}

/// Metadata and raw host bytes for one ModelOpt FP8 linear weight.
#[derive(Clone, Debug)]
pub struct ModelOptFp8Linear {
    /// Tensor name prefix, for example `model.layers.0.self_attn.q_proj`.
    pub prefix: String,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
    /// E4M3 bytes from `<prefix>.weight` with shape `[out, in]`.
    pub weight: Vec<u8>,
    /// Tensor-wide ModelOpt weight scale.
    pub weight_scale: f32,
    /// Optional per-output-channel compressed-tensors weight scales.
    pub channel_weight_scale: Option<Vec<f32>>,
    /// Static calibrated activation scale.
    pub input_scale: Option<f32>,
}

/// cuBLASLt-ready imported ModelOpt NVFP4 weight plus scalar scale metadata.
///
/// This type deliberately keeps `weight_scale_2` and `input_scale` next to the
/// uploaded matrix. A ModelOpt import is not just an [`Nvfp4Matrix`]: the matrix
/// contains packed values and repacked micro-scales, while the scalar scales
/// must still be accounted for at the operation boundary.
pub struct ModelOptCublasLtWeight {
    matrix: Nvfp4Matrix,
    weight_scale_2: f32,
    input_scale: f32,
}

/// cuBLASLt-ready ModelOpt W4A4 activation operand.
///
/// ModelOpt stores `input_scale` as a tensor-wide activation scale. vLLM's
/// W4A4 path quantizes the runtime activation after multiplying by
/// `1 / input_scale`, then uses `input_scale * weight_scale_2` as the GEMM
/// alpha. This type makes that convention explicit for the B operand.
pub struct ModelOptNvfp4Activation {
    matrix: Nvfp4Matrix,
    input_scale: f32,
    dequantized_scaled_values: Vec<f32>,
}

impl ModelOptNvfp4Activation {
    /// Quantizes a column-major activation operand for a ModelOpt NVFP4 linear.
    ///
    /// `values` are the unscaled runtime activation values. The packed operand
    /// represents `values / input_scale`; callers should use
    /// [`ModelOptCublasLtWeight::matmul_alpha`] for the corresponding GEMM
    /// alpha.
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
        if values.len() != rows * cols {
            return Err(Error::Shape {
                label: "ModelOpt activation",
                expected: format!("{} values", rows * cols),
                actual: format!("{} values", values.len()),
            });
        }

        let scaled_values = values
            .iter()
            .map(|value| value / input_scale)
            .collect::<Vec<_>>();
        let quantized = format::quantize_nvfp4_col_major(rows, cols, &scaled_values);
        let matrix = Nvfp4Matrix::from_packed_col_major_parts(
            rows,
            cols,
            &quantized.packed_values,
            &quantized.scales,
        )?;
        Ok(Self {
            matrix,
            input_scale,
            dequantized_scaled_values: quantized.dequantized_values,
        })
    }

    /// Quantizes a device-resident column-major activation operand.
    ///
    /// This follows the same `values / input_scale` convention as
    /// [`ModelOptNvfp4Activation::quantize_col_major_f32`], but avoids copying
    /// the activation values back to the host. The CPU-reference values are not
    /// retained, so [`ModelOptNvfp4Activation::dequantized_scaled_values`]
    /// returns an empty slice for operands created through this path.
    pub fn quantize_device_col_major_f32(
        rows: usize,
        cols: usize,
        values: &DeviceBuffer<f32>,
        input_scale: f32,
    ) -> Result<Self> {
        let matrix = quantize_nvfp4_col_major_f32_device(rows, cols, values, input_scale)?;
        Ok(Self {
            matrix,
            input_scale,
            dequantized_scaled_values: Vec::new(),
        })
    }

    /// Returns the cuBLASLt-compatible B operand matrix.
    pub fn matrix(&self) -> &Nvfp4Matrix {
        &self.matrix
    }

    /// Returns the ModelOpt activation scale used to form this operand.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Returns host dequantized values represented by this operand.
    ///
    /// These values are still in the `x / input_scale` domain, matching what
    /// cuBLASLt sees before GEMM alpha is applied.
    pub fn dequantized_scaled_values(&self) -> &[f32] {
        &self.dequantized_scaled_values
    }

    /// Returns total device bytes owned by the uploaded operand.
    pub fn device_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }
}

impl ModelOptCublasLtWeight {
    #[allow(missing_docs)]
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

    /// Returns the cuBLASLt-compatible weight matrix.
    pub fn matrix(&self) -> &Nvfp4Matrix {
        &self.matrix
    }

    /// Returns ModelOpt's tensor-wide weight scale.
    pub fn weight_scale_2(&self) -> f32 {
        self.weight_scale_2
    }

    /// Returns ModelOpt's static activation scale for this linear.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Returns the GEMM alpha for ModelOpt W4A4 execution.
    pub fn matmul_alpha(&self) -> f32 {
        self.weight_scale_2 * self.input_scale
    }

    /// Quantizes a column-major activation operand for this weight.
    pub fn quantize_activation_col_major_f32(
        &self,
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> Result<ModelOptNvfp4Activation> {
        ModelOptNvfp4Activation::quantize_col_major_f32(rows, cols, values, self.input_scale)
    }

    /// Quantizes a device-resident column-major activation operand.
    pub fn quantize_activation_device_col_major_f32(
        &self,
        rows: usize,
        cols: usize,
        values: &DeviceBuffer<f32>,
    ) -> Result<ModelOptNvfp4Activation> {
        ModelOptNvfp4Activation::quantize_device_col_major_f32(rows, cols, values, self.input_scale)
    }

    /// Enqueues quantization of a device-resident activation operand into
    /// preallocated cuBLASLt storage on `stream`.
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

    /// Returns total device bytes owned by the uploaded matrix.
    pub fn device_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }
}

impl ModelOptNvfp4Linear {
    /// Imports a ModelOpt NVFP4 linear from one safetensors shard.
    pub fn from_shard(shard: &SafeTensorShard, prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_scale_2_name = format!("{prefix}.weight_scale_2");
        let input_scale_name = format!("{prefix}.input_scale");

        let weight_info = shard.require_tensor(&weight_name)?;
        let scale_info = shard.require_tensor(&scale_name)?;
        let (out_features, in_features) = validate_modelopt_weight(weight_info)?;
        validate_modelopt_scale(scale_info, out_features, in_features)?;

        Ok(Self {
            prefix,
            out_features,
            in_features,
            packed_weight: shard.read_tensor_bytes(&weight_name)?,
            weight_scale: shard.read_tensor_bytes(&scale_name)?,
            weight_scale_2: shard.read_scalar_f32(&weight_scale_2_name)?,
            input_scale: shard.read_scalar_f32(&input_scale_name)?,
        })
    }

    /// Dequantizes the packed E2M1 weight with per-row ModelOpt scales to f32
    /// column-major [K, M] values.
    ///
    /// `weight_scale` is `[out, in/16]` row-major. The dequantized value at
    /// (m, k) is `e2m1_value(nibble) * e4m3_value(weight_scale[m * (in/16) + k/16])`.
    pub fn dequantize_to_f32_col_major(&self) -> Vec<f32> {
        let k = self.in_features;
        let m = self.out_features;
        let in_blocks = k / 16;
        let mut values = vec![0.0f32; k * m];
        for mi in 0..m {
            for ki in 0..k {
                let flat = mi * k + ki;
                let nibble = if flat & 1 == 0 {
                    self.packed_weight[flat / 2] & 0x0F
                } else {
                    (self.packed_weight[flat / 2] >> 4) & 0x0F
                };
                let e2m1 = format::e2m1_value(nibble);
                let scale_code = self.weight_scale[mi * in_blocks + ki / 16];
                let scale = format::e4m3_value(scale_code);
                values[ki + mi * k] = e2m1 * scale;
            }
        }
        values
    }

    /// Re-quantizes the weight with 4-row-grouped scales for the SIMT grouped
    /// GEMV kernel. Returns the re-quantized packed E2M1 values (column-major
    /// [K, M]) and `[M/4, K/16]` row-major group scales as device buffers.
    pub fn to_grouped_device(&self) -> Result<(DeviceBuffer<u8>, DeviceBuffer<u8>)> {
        let k = self.in_features;
        let m = self.out_features;
        let dequant = self.dequantize_to_f32_col_major();
        let grouped = format::quantize_nvfp4_4row_groups(k, m, &dequant);
        Ok((
            DeviceBuffer::from_host(&grouped.packed_values)?,
            DeviceBuffer::from_host(&grouped.group_scales)?,
        ))
    }

    /// Returns the logical cuBLASLt TN matrix dimensions, `K x M`.
    pub fn tn_weight_shape(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }

    /// Returns expected packed weight bytes.
    pub fn expected_weight_bytes(&self) -> usize {
        self.out_features * self.in_features / 2
    }

    /// Returns expected ModelOpt scale bytes.
    pub fn expected_scale_bytes(&self) -> usize {
        self.out_features * self.in_features / 16
    }

    /// Repackages ModelOpt linear scale bytes into cuBLASLt's tiled scale
    /// layout.
    pub fn cublaslt_scales(&self) -> Vec<u8> {
        modelopt_scales_to_cublaslt(self.out_features, self.in_features, &self.weight_scale)
    }

    /// Uploads this imported weight as a cuBLASLt-compatible NVFP4 matrix.
    ///
    /// This imports only the FP4 values and micro-scales. `weight_scale_2` and
    /// `input_scale` are retained on this struct because the correct full
    /// ModelOpt numerical path must account for them at the operation boundary.
    pub fn to_nvfp4_matrix(&self) -> Result<Nvfp4Matrix> {
        let (rows, cols) = self.tn_weight_shape();
        Nvfp4Matrix::from_packed_col_major_parts(
            rows,
            cols,
            &self.packed_weight,
            &self.cublaslt_scales(),
        )
    }

    /// Uploads this linear as an explicit ModelOpt/cuBLASLt weight object.
    pub fn as_cublaslt_weight(&self) -> Result<ModelOptCublasLtWeight> {
        Ok(ModelOptCublasLtWeight {
            matrix: self.to_nvfp4_matrix()?,
            weight_scale_2: self.weight_scale_2,
            input_scale: self.input_scale,
        })
    }

    #[allow(missing_docs)]
    pub fn concat_out_features(
        prefix: impl Into<String>,
        first: &Self,
        second: &Self,
    ) -> Result<Self> {
        if first.in_features != second.in_features {
            return Err(Error::Shape {
                label: "ModelOpt concat in_features",
                expected: format!("{}", first.in_features),
                actual: format!("{}", second.in_features),
            });
        }
        if first.input_scale != second.input_scale || first.weight_scale_2 != second.weight_scale_2
        {
            return Err(Error::Format {
                label: "ModelOpt concat scalar scales",
                detail: format!(
                    "input_scale {} vs {}, weight_scale_2 {} vs {}",
                    first.input_scale,
                    second.input_scale,
                    first.weight_scale_2,
                    second.weight_scale_2
                ),
            });
        }

        let mut packed_weight =
            Vec::with_capacity(first.packed_weight.len() + second.packed_weight.len());
        packed_weight.extend_from_slice(&first.packed_weight);
        packed_weight.extend_from_slice(&second.packed_weight);
        let mut weight_scale =
            Vec::with_capacity(first.weight_scale.len() + second.weight_scale.len());
        weight_scale.extend_from_slice(&first.weight_scale);
        weight_scale.extend_from_slice(&second.weight_scale);

        Ok(Self {
            prefix: prefix.into(),
            out_features: first.out_features + second.out_features,
            in_features: first.in_features,
            packed_weight,
            weight_scale,
            weight_scale_2: first.weight_scale_2,
            input_scale: first.input_scale,
        })
    }
}

impl ModelOptFp8Linear {
    /// Imports a ModelOpt FP8 linear from one safetensors shard.
    pub fn from_shard(shard: &SafeTensorShard, prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let input_scale_name = format!("{prefix}.input_scale");

        let weight_info = shard.require_tensor(&weight_name)?;
        let (out_features, in_features) = validate_modelopt_fp8_weight(weight_info)?;

        Ok(Self {
            prefix,
            out_features,
            in_features,
            weight: shard.read_tensor_bytes(&weight_name)?,
            weight_scale: shard.read_scalar_f32(&scale_name)?,
            channel_weight_scale: None,
            input_scale: Some(shard.read_scalar_f32(&input_scale_name)?),
        })
    }

    /// Uploads FP8 weight bytes to device memory.
    pub fn weight_device(&self) -> Result<DeviceBuffer<u8>> {
        DeviceBuffer::from_host(&self.weight)
    }

    /// Returns expected FP8 weight bytes.
    pub fn expected_weight_bytes(&self) -> usize {
        self.out_features * self.in_features
    }
}

/// Sharded ModelOpt safetensors checkpoint index.
#[derive(Clone, Debug)]
pub struct ModelOptCheckpoint {
    root: PathBuf,
    weight_map: Arc<BTreeMap<String, String>>,
    shards: Arc<Mutex<BTreeMap<String, Arc<SafeTensorShard>>>>,
}

impl ModelOptCheckpoint {
    /// Opens a Hugging Face safetensors checkpoint directory.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let index_path = root.join("model.safetensors.index.json");
        let index = fs::read_to_string(&index_path).map_err(|err| Error::Format {
            label: "safetensors index",
            detail: format!("{}: {err}", index_path.display()),
        })?;
        let json: Value = serde_json::from_str(&index).map_err(|err| Error::Format {
            label: "safetensors index json",
            detail: err.to_string(),
        })?;
        let map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Format {
                label: "safetensors index weight_map",
                detail: "missing weight_map object".to_string(),
            })?;
        let mut weight_map = BTreeMap::new();
        for (tensor, shard) in map {
            let shard = shard.as_str().ok_or_else(|| Error::Format {
                label: "safetensors index weight_map",
                detail: format!("{tensor} shard is not a string"),
            })?;
            weight_map.insert(tensor.clone(), shard.to_string());
        }
        Ok(Self {
            root,
            weight_map: Arc::new(weight_map),
            shards: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    /// Returns the checkpoint root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the shard filename for a named tensor.
    pub fn shard_name_for_tensor(&self, tensor: &str) -> Result<&str> {
        self.weight_map
            .get(tensor)
            .map(String::as_str)
            .ok_or_else(|| Error::Format {
                label: "safetensors index lookup",
                detail: format!("missing tensor {tensor}"),
            })
    }

    /// Returns true when the checkpoint index contains `tensor`.
    pub fn contains_tensor(&self, tensor: &str) -> bool {
        self.weight_map.contains_key(tensor)
    }

    /// Returns safetensors metadata for `tensor`.
    pub fn tensor_info(&self, tensor: &str) -> Result<SafeTensorInfo> {
        let shard = self.open_shard_for_tensor(tensor)?;
        Ok(shard.require_tensor(tensor)?.clone())
    }

    /// Opens the safetensors shard containing `tensor`.
    pub fn open_shard_for_tensor(&self, tensor: &str) -> Result<Arc<SafeTensorShard>> {
        let shard = self.shard_name_for_tensor(tensor)?;
        if let Some(cached) = self
            .shards
            .lock()
            .expect("safetensors shard cache mutex poisoned")
            .get(shard)
            .cloned()
        {
            return Ok(cached);
        }
        let opened = Arc::new(SafeTensorShard::open(self.root.join(shard))?);
        self.shards
            .lock()
            .expect("safetensors shard cache mutex poisoned")
            .insert(shard.to_string(), opened.clone());
        Ok(opened)
    }

    /// Imports a ModelOpt NVFP4 linear by tensor prefix.
    pub fn load_nvfp4_linear(&self, prefix: &str) -> Result<ModelOptNvfp4Linear> {
        if self.contains_tensor(&format!("{prefix}.weight_packed")) {
            return self.load_compressed_tensors_nvfp4_linear(prefix);
        }
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_scale_2_name = format!("{prefix}.weight_scale_2");
        let input_scale_name = format!("{prefix}.input_scale");

        let weight_shard = self.open_shard_for_tensor(&weight_name)?;
        let scale_shard = self.open_shard_for_tensor(&scale_name)?;
        let weight_scale_2_shard = self.open_shard_for_tensor(&weight_scale_2_name)?;
        let input_scale_shard = self.open_shard_for_tensor(&input_scale_name)?;

        let weight_info = weight_shard.require_tensor(&weight_name)?;
        let scale_info = scale_shard.require_tensor(&scale_name)?;
        let (out_features, in_features) = validate_modelopt_weight(weight_info)?;
        validate_modelopt_scale(scale_info, out_features, in_features)?;

        Ok(ModelOptNvfp4Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            packed_weight: weight_shard.read_tensor_bytes(&weight_name)?,
            weight_scale: scale_shard.read_tensor_bytes(&scale_name)?,
            weight_scale_2: weight_scale_2_shard.read_scalar_f32(&weight_scale_2_name)?,
            input_scale: input_scale_shard.read_scalar_f32(&input_scale_name)?,
        })
    }

    /// Reads only the global weight and input scales for an NVFP4 linear.
    ///
    /// This avoids faulting the packed weight and block-scale payloads when a
    /// caller needs scalar metadata before loading weights on demand.
    pub fn load_nvfp4_scales(&self, prefix: &str) -> Result<(f32, f32)> {
        if self.contains_tensor(&format!("{prefix}.weight_packed")) {
            let weight_name = format!("{prefix}.weight_global_scale");
            let input_name = format!("{prefix}.input_global_scale");
            let weight_shard = self.open_shard_for_tensor(&weight_name)?;
            let input_shard = self.open_shard_for_tensor(&input_name)?;
            let weight_divisor = read_single_f32(
                &weight_shard,
                &weight_name,
                "compressed-tensors NVFP4 weight_global_scale",
            )?;
            let input_divisor = read_single_f32(
                &input_shard,
                &input_name,
                "compressed-tensors NVFP4 input_global_scale",
            )?;
            return Ok((
                reciprocal_scale(
                    weight_divisor,
                    "compressed-tensors NVFP4 weight_global_scale",
                )?,
                reciprocal_scale(input_divisor, "compressed-tensors NVFP4 input_global_scale")?,
            ));
        }

        let weight_name = format!("{prefix}.weight_scale_2");
        let input_name = format!("{prefix}.input_scale");
        let weight_shard = self.open_shard_for_tensor(&weight_name)?;
        let input_shard = self.open_shard_for_tensor(&input_name)?;
        Ok((
            weight_shard.read_scalar_f32(&weight_name)?,
            input_shard.read_scalar_f32(&input_name)?,
        ))
    }

    /// Imports a ModelOpt FP8 linear by tensor prefix.
    pub fn load_fp8_linear(&self, prefix: &str) -> Result<ModelOptFp8Linear> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_shard = self.open_shard_for_tensor(&weight_name)?;
        let scale_shard = self.open_shard_for_tensor(&scale_name)?;

        let weight_info = weight_shard.require_tensor(&weight_name)?;
        let (out_features, in_features) = validate_modelopt_fp8_weight(weight_info)?;

        let input_scale_name = format!("{prefix}.input_scale");
        let (weight_scale, channel_weight_scale, input_scale) =
            if self.contains_tensor(&input_scale_name) {
                let input_scale_shard = self.open_shard_for_tensor(&input_scale_name)?;
                (
                    scale_shard.read_scalar_f32(&scale_name)?,
                    None,
                    Some(input_scale_shard.read_scalar_f32(&input_scale_name)?),
                )
            } else {
                let scales = scale_shard.read_float_tensor_as_f32(&scale_name)?;
                if scales.len() != out_features {
                    return Err(Error::Shape {
                        label: "compressed-tensors FP8 weight_scale",
                        expected: format!("{out_features} per-output-channel scales"),
                        actual: format!("{} values", scales.len()),
                    });
                }
                if scales.iter().any(|scale| !scale.is_finite()) {
                    return Err(Error::Format {
                        label: "compressed-tensors FP8 weight_scale",
                        detail: "all channel scales must be finite".to_string(),
                    });
                }
                (1.0, Some(scales), None)
            };

        Ok(ModelOptFp8Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            weight: weight_shard.read_tensor_bytes(&weight_name)?,
            weight_scale,
            channel_weight_scale,
            input_scale,
        })
    }

    fn load_compressed_tensors_nvfp4_linear(&self, prefix: &str) -> Result<ModelOptNvfp4Linear> {
        let weight_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_global_scale_name = format!("{prefix}.weight_global_scale");
        let input_global_scale_name = format!("{prefix}.input_global_scale");

        let weight_shard = self.open_shard_for_tensor(&weight_name)?;
        let scale_shard = self.open_shard_for_tensor(&scale_name)?;
        let weight_global_scale_shard = self.open_shard_for_tensor(&weight_global_scale_name)?;
        let input_global_scale_shard = self.open_shard_for_tensor(&input_global_scale_name)?;
        let weight_info = weight_shard.require_tensor(&weight_name)?;
        let scale_info = scale_shard.require_tensor(&scale_name)?;
        let (out_features, in_features) = validate_modelopt_weight(weight_info)?;
        validate_modelopt_scale(scale_info, out_features, in_features)?;

        let weight_divisor = read_single_f32(
            &weight_global_scale_shard,
            &weight_global_scale_name,
            "compressed-tensors NVFP4 weight_global_scale",
        )?;
        let input_divisor = read_single_f32(
            &input_global_scale_shard,
            &input_global_scale_name,
            "compressed-tensors NVFP4 input_global_scale",
        )?;

        Ok(ModelOptNvfp4Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            packed_weight: weight_shard.read_tensor_bytes(&weight_name)?,
            weight_scale: scale_shard.read_tensor_bytes(&scale_name)?,
            weight_scale_2: reciprocal_scale(
                weight_divisor,
                "compressed-tensors NVFP4 weight_global_scale",
            )?,
            input_scale: reciprocal_scale(
                input_divisor,
                "compressed-tensors NVFP4 input_global_scale",
            )?,
        })
    }
}

fn read_single_f32(shard: &SafeTensorShard, name: &str, label: &'static str) -> Result<f32> {
    let values = shard.read_float_tensor_as_f32(name)?;
    if values.len() != 1 {
        return Err(Error::Shape {
            label,
            expected: "one F32 value".to_string(),
            actual: format!("{} values", values.len()),
        });
    }
    Ok(values[0])
}

fn reciprocal_scale(value: f32, label: &'static str) -> Result<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::Format {
            label,
            detail: format!("expected positive finite divisor, got {value}"),
        });
    }
    Ok(value.recip())
}

/// Converts ModelOpt's linear `[out, in / 16]` E4M3 scale matrix into
/// cuBLASLt's tiled `VEC16_UE4M3` layout for a `K x M` TN weight matrix.
pub fn modelopt_scales_to_cublaslt(
    out_features: usize,
    in_features: usize,
    modelopt_scales: &[u8],
) -> Vec<u8> {
    assert_eq!(in_features % 16, 0);
    let in_blocks = in_features / 16;
    assert_eq!(modelopt_scales.len(), out_features * in_blocks);

    let mut cublaslt = vec![0u8; format::ue4m3_scale_layout_len(out_features, in_features)];
    for out in 0..out_features {
        for block in 0..in_blocks {
            let src = out * in_blocks + block;
            let dst = format::ue4m3_tiled_scale_offset(out, block, in_features);
            cublaslt[dst] = modelopt_scales[src];
        }
    }
    cublaslt
}

fn validate_modelopt_weight(info: &SafeTensorInfo) -> Result<(usize, usize)> {
    if info.dtype != "U8" || info.shape.len() != 2 {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight",
            expected: "dtype=U8 shape=[out,in/2]".to_string(),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let out_features = info.shape[0];
    let packed_in = info.shape[1];
    let in_features = packed_in * 2;
    let expected_bytes = out_features * packed_in;
    if info.byte_len() != expected_bytes as u64 {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight bytes",
            expected: format!("{expected_bytes}"),
            actual: format!("{}", info.byte_len()),
        });
    }
    Ok((out_features, in_features))
}

fn validate_modelopt_fp8_weight(info: &SafeTensorInfo) -> Result<(usize, usize)> {
    if info.dtype != "F8_E4M3" || info.shape.len() != 2 {
        return Err(Error::Shape {
            label: "ModelOpt FP8 weight",
            expected: "dtype=F8_E4M3 shape=[out,in]".to_string(),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let out_features = info.shape[0];
    let in_features = info.shape[1];
    if out_features == 0
        || in_features == 0
        || info.byte_len() != (out_features * in_features) as u64
    {
        return Err(Error::Shape {
            label: "ModelOpt FP8 weight bytes",
            expected: format!("{}", out_features * in_features),
            actual: format!("{}", info.byte_len()),
        });
    }
    Ok((out_features, in_features))
}

fn validate_modelopt_scale(
    info: &SafeTensorInfo,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    if info.dtype != "F8_E4M3" || info.shape != [out_features, in_features / 16] {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight_scale",
            expected: format!("dtype=F8_E4M3 shape=[{out_features},{}]", in_features / 16),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let expected_bytes = out_features * in_features / 16;
    if info.byte_len() != expected_bytes as u64 {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight_scale bytes",
            expected: format!("{expected_bytes}"),
            actual: format!("{}", info.byte_len()),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modelopt_scale_repack_matches_cublaslt_offsets() {
        let out = 130;
        let input = 80;
        let blocks = input / 16;
        let source = (0..out * blocks)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>();
        let repacked = modelopt_scales_to_cublaslt(out, input, &source);

        for o in 0..out {
            for b in 0..blocks {
                let src = o * blocks + b;
                let dst = format::ue4m3_tiled_scale_offset(o, b, input);
                assert_eq!(repacked[dst], source[src]);
            }
        }
    }
}
