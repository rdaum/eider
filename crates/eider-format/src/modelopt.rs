//! Host records and checkpoint decoding for NVIDIA ModelOpt weights.
//!
//! These types preserve checkpoint byte layouts and calibration metadata. They
//! contain no CUDA allocation, stream, or prepared kernel representation.

use crate::{Error, Result, SafeTensorCheckpoint, SafeTensorInfo, SafeTensorShard};

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
    /// ModelOpt row-major E4M3 K16 scale bytes from `<prefix>.weight_scale`.
    pub weight_scale: Vec<u8>,
    /// Tensor-wide ModelOpt weight scale.
    pub weight_scale_2: f32,
    /// Static calibrated activation scale, or `1.0` for W4A16 weights.
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

/// Raw block-scaled FP8 linear used by dense ModelOpt projections.
///
/// Weights are E4M3 in row-major `[out, in]` order. One unsigned E8M0 scale
/// applies to each 128 by 128 weight block.
#[derive(Clone, Debug)]
pub struct ModelOptBlockScaledFp8Linear {
    /// Tensor name prefix.
    pub prefix: String,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
    /// E4M3 weight bytes from `<prefix>.weight`.
    pub weight: Vec<u8>,
    /// E8M0 block-scale bytes.
    pub weight_scale: Vec<u8>,
}

/// Model-family interpretation of a sharded ModelOpt safetensors checkpoint.
///
/// This wrapper owns no device resources. It resolves ModelOpt tensor groups
/// into host records; inference must explicitly prepare those records for a
/// CUDA plan.
#[derive(Clone, Debug)]
pub struct ModelOptCheckpoint {
    checkpoint: SafeTensorCheckpoint,
}

impl ModelOptCheckpoint {
    /// Opens a ModelOpt checkpoint directory.
    pub fn open(root: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            checkpoint: SafeTensorCheckpoint::open(root)?,
        })
    }

    /// Returns the checkpoint directory.
    pub fn root(&self) -> &std::path::Path {
        self.checkpoint.root()
    }

    /// Returns whether `tensor` is indexed by this checkpoint.
    pub fn contains_tensor(&self, tensor: &str) -> bool {
        self.checkpoint.contains_tensor(tensor)
    }

    /// Returns the shard filename containing `tensor`.
    pub fn shard_name_for_tensor(&self, tensor: &str) -> Result<&str> {
        self.checkpoint.shard_name_for_tensor(tensor)
    }

    /// Returns host metadata for `tensor`.
    pub fn tensor_info(&self, tensor: &str) -> Result<SafeTensorInfo> {
        self.checkpoint.tensor_info(tensor)
    }

    /// Opens the host shard containing `tensor`.
    pub fn open_shard_for_tensor(&self, tensor: &str) -> Result<std::sync::Arc<SafeTensorShard>> {
        self.checkpoint.open_shard_for_tensor(tensor)
    }

    /// Imports an NVFP4 linear by tensor prefix.
    pub fn load_nvfp4_linear(&self, prefix: &str) -> Result<ModelOptNvfp4Linear> {
        if self.contains_tensor(&format!("{prefix}.weight_packed")) {
            return self.load_compressed_tensors_nvfp4_linear(prefix);
        }
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let global_name = format!("{prefix}.weight_scale_2");
        let input_name = format!("{prefix}.input_scale");
        let weight = self.open_shard_for_tensor(&weight_name)?;
        let scales = self.open_shard_for_tensor(&scale_name)?;
        let global = self.open_shard_for_tensor(&global_name)?;
        let (out_features, in_features) =
            validate_nvfp4_weight(weight.require_tensor(&weight_name)?)?;
        validate_nvfp4_scales(
            scales.require_tensor(&scale_name)?,
            out_features,
            in_features,
        )?;
        Ok(ModelOptNvfp4Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            packed_weight: weight.read_tensor_bytes(&weight_name)?,
            weight_scale: scales.read_tensor_bytes(&scale_name)?,
            weight_scale_2: global.read_scalar_f32(&global_name)?,
            input_scale: self.input_scale_or_unity(&input_name)?,
        })
    }

    /// Imports MXFP4 K32 scales and expands them to ModelOpt's K16 records.
    pub fn load_mxfp4_linear(&self, prefix: &str) -> Result<ModelOptNvfp4Linear> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.scale");
        let weight = self.open_shard_for_tensor(&weight_name)?;
        let scales = self.open_shard_for_tensor(&scale_name)?;
        let weight_info = weight.require_tensor(&weight_name)?;
        let scale_info = scales.require_tensor(&scale_name)?;
        if weight_info.dtype != "I8" || weight_info.shape.len() != 2 {
            return Err(Error::Shape {
                label: "MXFP4 weight",
                expected: "dtype=I8 shape=[out,in/2]".to_string(),
                actual: format!("dtype={} shape={:?}", weight_info.dtype, weight_info.shape),
            });
        }
        let out_features = weight_info.shape[0];
        let in_features = weight_info.shape[1]
            .checked_mul(2)
            .ok_or_else(|| Error::Shape {
                label: "MXFP4 weight",
                expected: "input width without overflow".to_string(),
                actual: weight_info.shape[1].to_string(),
            })?;
        if !in_features.is_multiple_of(32) {
            return Err(Error::Shape {
                label: "MXFP4 weight",
                expected: "input width divisible by 32".to_string(),
                actual: in_features.to_string(),
            });
        }
        let expected_shape = [out_features, in_features / 32];
        if scale_info.dtype != "F8_E8M0"
            || scale_info.shape != expected_shape
            || scale_info.byte_len() != (out_features * in_features / 32) as u64
        {
            return Err(Error::Shape {
                label: "MXFP4 scale",
                expected: format!("dtype=F8_E8M0 shape={expected_shape:?}"),
                actual: format!("dtype={} shape={:?}", scale_info.dtype, scale_info.shape),
            });
        }
        let weight_scale = scales
            .read_tensor_bytes(&scale_name)?
            .into_iter()
            .map(|code| {
                let value = e8m0_value(code);
                if !value.is_finite() {
                    return Err(Error::Format {
                        label: "MXFP4 scale",
                        detail: format!("{scale_name} contains a NaN scale code"),
                    });
                }
                let converted = ue4m3_code(value);
                if e4m3_value(converted) != value {
                    return Err(Error::Format {
                        label: "MXFP4 scale",
                        detail: format!(
                            "{scale_name} contains E8M0 scale {value} outside exact UE4M3 range"
                        ),
                    });
                }
                Ok(converted)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|scale| [scale, scale])
            .collect();
        Ok(ModelOptNvfp4Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            packed_weight: weight.read_tensor_bytes(&weight_name)?,
            weight_scale,
            weight_scale_2: 1.0,
            input_scale: 1.0,
        })
    }

    /// Imports one expert from a stacked NVFP4 ModelOpt linear.
    pub fn load_nvfp4_expert_linear(
        &self,
        prefix: &str,
        expert: usize,
    ) -> Result<ModelOptNvfp4Linear> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let global_name = format!("{prefix}.weight_scale_2");
        let input_name = format!("{prefix}.input_scale");
        let weight = self.open_shard_for_tensor(&weight_name)?;
        let scales = self.open_shard_for_tensor(&scale_name)?;
        let global = self.open_shard_for_tensor(&global_name)?;
        let (experts, out_features, in_features) =
            validate_nvfp4_expert_weight(weight.require_tensor(&weight_name)?)?;
        validate_nvfp4_expert_scales(
            scales.require_tensor(&scale_name)?,
            experts,
            out_features,
            in_features,
        )?;
        if expert >= experts {
            return Err(Error::Shape {
                label: "ModelOpt NVFP4 expert index",
                expected: format!("expert < {experts}"),
                actual: expert.to_string(),
            });
        }
        let weight_bytes = out_features * in_features / 2;
        let scale_bytes = out_features * in_features / 16;
        let weight_scale_2 = global.read_float_tensor_as_f32(&global_name)?;
        let input_scale = if self.contains_tensor(&input_name) {
            self.open_shard_for_tensor(&input_name)?
                .read_float_tensor_as_f32(&input_name)?
        } else {
            vec![1.0; experts]
        };
        if weight_scale_2.len() != experts || input_scale.len() != experts {
            return Err(Error::Shape {
                label: "ModelOpt NVFP4 expert scalar scales",
                expected: format!("{experts} weight and input scales"),
                actual: format!(
                    "weight_scale_2={} input_scale={}",
                    weight_scale_2.len(),
                    input_scale.len()
                ),
            });
        }
        Ok(ModelOptNvfp4Linear {
            prefix: format!("{prefix}[{expert}]"),
            out_features,
            in_features,
            packed_weight: weight.read_tensor_byte_range(
                &weight_name,
                (expert * weight_bytes) as u64,
                weight_bytes,
            )?,
            weight_scale: scales.read_tensor_byte_range(
                &scale_name,
                (expert * scale_bytes) as u64,
                scale_bytes,
            )?,
            weight_scale_2: weight_scale_2[expert],
            input_scale: input_scale[expert],
        })
    }

    /// Reads only the global weight and input scales for an NVFP4 linear.
    pub fn load_nvfp4_scales(&self, prefix: &str) -> Result<(f32, f32)> {
        if self.contains_tensor(&format!("{prefix}.weight_packed")) {
            let weight_name = format!("{prefix}.weight_global_scale");
            let input_name = format!("{prefix}.input_global_scale");
            let weight = self.open_shard_for_tensor(&weight_name)?;
            let input = self.open_shard_for_tensor(&input_name)?;
            return Ok((
                reciprocal_scale(
                    read_single_f32(
                        &weight,
                        &weight_name,
                        "compressed-tensors NVFP4 weight_global_scale",
                    )?,
                    "compressed-tensors NVFP4 weight_global_scale",
                )?,
                reciprocal_scale(
                    read_single_f32(
                        &input,
                        &input_name,
                        "compressed-tensors NVFP4 input_global_scale",
                    )?,
                    "compressed-tensors NVFP4 input_global_scale",
                )?,
            ));
        }
        let weight_name = format!("{prefix}.weight_scale_2");
        Ok((
            self.open_shard_for_tensor(&weight_name)?
                .read_scalar_f32(&weight_name)?,
            self.input_scale_or_unity(&format!("{prefix}.input_scale"))?,
        ))
    }

    /// Imports an FP8 linear by tensor prefix.
    pub fn load_fp8_linear(&self, prefix: &str) -> Result<ModelOptFp8Linear> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight = self.open_shard_for_tensor(&weight_name)?;
        let scales = self.open_shard_for_tensor(&scale_name)?;
        let (out_features, in_features) =
            validate_fp8_weight(weight.require_tensor(&weight_name)?)?;
        let input_name = format!("{prefix}.input_scale");
        let (weight_scale, channel_weight_scale, input_scale) = if self.contains_tensor(&input_name)
        {
            let input = self.open_shard_for_tensor(&input_name)?;
            (
                scales.read_scalar_f32(&scale_name)?,
                None,
                Some(input.read_scalar_f32(&input_name)?),
            )
        } else {
            let channel = scales.read_float_tensor_as_f32(&scale_name)?;
            if channel.len() != out_features || channel.iter().any(|value| !value.is_finite()) {
                return Err(Error::Shape {
                    label: "compressed-tensors FP8 weight_scale",
                    expected: format!("{out_features} finite per-output-channel scales"),
                    actual: format!("{} values", channel.len()),
                });
            }
            (1.0, Some(channel), None)
        };
        Ok(ModelOptFp8Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            weight: weight.read_tensor_bytes(&weight_name)?,
            weight_scale,
            channel_weight_scale,
            input_scale,
        })
    }

    /// Imports a block-scaled E4M3 linear with E8M0 scales.
    pub fn load_block_scaled_fp8_linear(
        &self,
        prefix: &str,
    ) -> Result<ModelOptBlockScaledFp8Linear> {
        self.load_block_scaled_fp8_linear_with_scale(prefix, "scale", &["F8_E8M0"])
    }

    /// Imports a block-scaled E4M3 linear with inverse weight scales.
    pub fn load_weight_scale_inv_block_fp8_linear(
        &self,
        prefix: &str,
    ) -> Result<ModelOptBlockScaledFp8Linear> {
        self.load_block_scaled_fp8_linear_with_scale(
            prefix,
            "weight_scale_inv",
            &["F8_E8M0FNU", "F8_E8M0"],
        )
    }

    fn input_scale_or_unity(&self, name: &str) -> Result<f32> {
        if self.contains_tensor(name) {
            self.open_shard_for_tensor(name)?.read_scalar_f32(name)
        } else {
            Ok(1.0)
        }
    }

    fn load_block_scaled_fp8_linear_with_scale(
        &self,
        prefix: &str,
        suffix: &str,
        dtypes: &[&str],
    ) -> Result<ModelOptBlockScaledFp8Linear> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.{suffix}");
        let weight = self.open_shard_for_tensor(&weight_name)?;
        let scales = self.open_shard_for_tensor(&scale_name)?;
        let (out_features, in_features) =
            validate_fp8_weight(weight.require_tensor(&weight_name)?)?;
        validate_block_scaled_fp8_scales(
            scales.require_tensor(&scale_name)?,
            out_features,
            in_features,
            dtypes,
        )?;
        Ok(ModelOptBlockScaledFp8Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            weight: weight.read_tensor_bytes(&weight_name)?,
            weight_scale: scales.read_tensor_bytes(&scale_name)?,
        })
    }

    fn load_compressed_tensors_nvfp4_linear(&self, prefix: &str) -> Result<ModelOptNvfp4Linear> {
        let weight_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_global_name = format!("{prefix}.weight_global_scale");
        let input_global_name = format!("{prefix}.input_global_scale");
        let weight = self.open_shard_for_tensor(&weight_name)?;
        let scales = self.open_shard_for_tensor(&scale_name)?;
        let weight_global = self.open_shard_for_tensor(&weight_global_name)?;
        let input_global = self.open_shard_for_tensor(&input_global_name)?;
        let (out_features, in_features) =
            validate_nvfp4_weight(weight.require_tensor(&weight_name)?)?;
        validate_nvfp4_scales(
            scales.require_tensor(&scale_name)?,
            out_features,
            in_features,
        )?;
        Ok(ModelOptNvfp4Linear {
            prefix: prefix.to_string(),
            out_features,
            in_features,
            packed_weight: weight.read_tensor_bytes(&weight_name)?,
            weight_scale: scales.read_tensor_bytes(&scale_name)?,
            weight_scale_2: reciprocal_scale(
                read_single_f32(
                    &weight_global,
                    &weight_global_name,
                    "compressed-tensors NVFP4 weight_global_scale",
                )?,
                "compressed-tensors NVFP4 weight_global_scale",
            )?,
            input_scale: reciprocal_scale(
                read_single_f32(
                    &input_global,
                    &input_global_name,
                    "compressed-tensors NVFP4 input_global_scale",
                )?,
                "compressed-tensors NVFP4 input_global_scale",
            )?,
        })
    }
}

impl ModelOptNvfp4Linear {
    /// Quantizes a row-major BF16 weight to ModelOpt-compatible NVFP4 storage.
    pub fn quantize_bf16(
        prefix: impl Into<String>,
        out_features: usize,
        in_features: usize,
        values: &[u16],
    ) -> Result<Self> {
        let expected = checked_elements("BF16-to-NVFP4 weight", out_features, in_features)?;
        validate_k16_shape("BF16-to-NVFP4 weight", out_features, in_features)?;
        if values.len() != expected {
            return Err(Error::Shape {
                label: "BF16-to-NVFP4 weight",
                expected: format!("{expected} values"),
                actual: format!("{} values", values.len()),
            });
        }
        Self::quantize_values(prefix, out_features, in_features, |index| {
            bf16_to_f32(values[index])
        })
    }

    /// Requantizes a row-major ModelOpt FP8 weight to NVFP4 storage.
    pub fn quantize_fp8(weight: &ModelOptFp8Linear) -> Result<Self> {
        let expected = checked_elements(
            "FP8-to-NVFP4 weight",
            weight.out_features,
            weight.in_features,
        )?;
        if weight.weight.len() != expected {
            return Err(Error::Shape {
                label: "FP8-to-NVFP4 weight",
                expected: format!("{expected} values"),
                actual: format!("{} values", weight.weight.len()),
            });
        }
        if !weight.weight_scale.is_finite() {
            return Err(Error::Format {
                label: "FP8-to-NVFP4 weight scale",
                detail: format!("expected finite scalar scale, got {}", weight.weight_scale),
            });
        }
        if let Some(scales) = &weight.channel_weight_scale
            && (scales.len() != weight.out_features
                || scales.iter().any(|scale| !scale.is_finite()))
        {
            return Err(Error::Shape {
                label: "FP8-to-NVFP4 channel scales",
                expected: format!("{} finite scales", weight.out_features),
                actual: format!("{} scales", scales.len()),
            });
        }
        Self::quantize_values(
            weight.prefix.clone(),
            weight.out_features,
            weight.in_features,
            |index| {
                let row = index / weight.in_features;
                let scale = weight
                    .channel_weight_scale
                    .as_ref()
                    .map_or(weight.weight_scale, |scales| scales[row]);
                e4m3_value(weight.weight[index]) * scale
            },
        )
    }

    /// Requantizes a 128 by 128 block-scaled FP8 weight to NVFP4 storage.
    pub fn quantize_block_scaled_fp8(weight: &ModelOptBlockScaledFp8Linear) -> Result<Self> {
        let expected_values = checked_elements(
            "block-FP8-to-NVFP4 weight",
            weight.out_features,
            weight.in_features,
        )?;
        let expected_scales = (weight.out_features / 128)
            .checked_mul(weight.in_features / 128)
            .ok_or_else(|| Error::Shape {
                label: "block-FP8-to-NVFP4 scales",
                expected: "block rows * block columns without overflow".to_string(),
                actual: format!(
                    "out_features={} in_features={}",
                    weight.out_features, weight.in_features
                ),
            })?;
        if weight.out_features == 0
            || weight.in_features == 0
            || !weight.out_features.is_multiple_of(128)
            || !weight.in_features.is_multiple_of(128)
            || weight.weight.len() != expected_values
            || weight.weight_scale.len() != expected_scales
        {
            return Err(Error::Shape {
                label: "block-FP8-to-NVFP4 weight",
                expected: format!(
                    "non-zero 128-aligned dimensions, {expected_values} values, {expected_scales} scales"
                ),
                actual: format!(
                    "out_features={} in_features={} values={} scales={}",
                    weight.out_features,
                    weight.in_features,
                    weight.weight.len(),
                    weight.weight_scale.len()
                ),
            });
        }
        if weight.weight_scale.contains(&u8::MAX) {
            return Err(Error::Format {
                label: "block-FP8-to-NVFP4 scale",
                detail: "E8M0 scale tensor contains a NaN code".to_string(),
            });
        }
        let scale_cols = weight.in_features / 128;
        Self::quantize_values(
            weight.prefix.clone(),
            weight.out_features,
            weight.in_features,
            |index| {
                let row = index / weight.in_features;
                let col = index % weight.in_features;
                let scale = weight.weight_scale[(row / 128) * scale_cols + col / 128];
                e4m3_value(weight.weight[index]) * e8m0_value(scale)
            },
        )
    }

    /// Quantizes values supplied in row-major `[out, in]` order into a host
    /// ModelOpt NVFP4 record.
    pub fn quantize_values(
        prefix: impl Into<String>,
        out_features: usize,
        in_features: usize,
        value_at: impl Fn(usize) -> f32 + Sync,
    ) -> Result<Self> {
        validate_k16_shape("NVFP4 weight quantization", out_features, in_features)?;
        let blocks_per_row = in_features / 16;
        let mut packed_weight = vec![0u8; out_features * in_features / 2];
        let mut weight_scale = vec![0u8; out_features * blocks_per_row];
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(out_features);
        let rows_per_chunk = out_features.div_ceil(workers);
        let packed_per_chunk = rows_per_chunk * in_features / 2;
        let scales_per_chunk = rows_per_chunk * blocks_per_row;
        std::thread::scope(|scope| {
            for (chunk, (packed, scales)) in packed_weight
                .chunks_mut(packed_per_chunk)
                .zip(weight_scale.chunks_mut(scales_per_chunk))
                .enumerate()
            {
                let value_at = &value_at;
                scope.spawn(move || {
                    let rows = scales.len() / blocks_per_row;
                    for row in 0..rows {
                        let global_row = chunk * rows_per_chunk + row;
                        for block in 0..blocks_per_row {
                            let global_start = global_row * in_features + block * 16;
                            let local_start = row * in_features + block * 16;
                            let max_abs = (global_start..global_start + 16)
                                .map(value_at)
                                .filter(|value| value.is_finite())
                                .map(f32::abs)
                                .fold(0.0f32, f32::max);
                            let scale_code = if max_abs != 0.0 {
                                ue4m3_code(max_abs / 6.0)
                            } else {
                                0
                            };
                            let scale = e4m3_value(scale_code);
                            scales[row * blocks_per_row + block] = scale_code;
                            for offset in 0..16 {
                                let flat = local_start + offset;
                                let value = value_at(global_start + offset);
                                let code =
                                    e2m1_code(if scale == 0.0 { 0.0 } else { value / scale });
                                if flat & 1 == 0 {
                                    packed[flat / 2] = code;
                                } else {
                                    packed[flat / 2] |= code << 4;
                                }
                            }
                        }
                    }
                });
            }
        });
        Ok(Self {
            prefix: prefix.into(),
            out_features,
            in_features,
            packed_weight,
            weight_scale,
            weight_scale_2: 1.0,
            input_scale: 1.0,
        })
    }

    /// Imports an NVFP4 linear from one safetensors shard.
    pub fn from_shard(shard: &SafeTensorShard, prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_scale_2_name = format!("{prefix}.weight_scale_2");
        let input_scale_name = format!("{prefix}.input_scale");
        let weight_info = shard.require_tensor(&weight_name)?;
        let scale_info = shard.require_tensor(&scale_name)?;
        let (out_features, in_features) = validate_nvfp4_weight(weight_info)?;
        validate_nvfp4_scales(scale_info, out_features, in_features)?;
        Ok(Self {
            prefix,
            out_features,
            in_features,
            packed_weight: shard.read_tensor_bytes(&weight_name)?,
            weight_scale: shard.read_tensor_bytes(&scale_name)?,
            weight_scale_2: shard.read_scalar_f32(&weight_scale_2_name)?,
            input_scale: input_scale_or_unity(shard, &input_scale_name)?,
        })
    }

    /// Dequantizes this ModelOpt record to column-major `[K, M]` f32 values.
    pub fn dequantize_to_f32_col_major(&self) -> Vec<f32> {
        let blocks = self.in_features / 16;
        let mut values = vec![0.0; self.in_features * self.out_features];
        for out in 0..self.out_features {
            for input in 0..self.in_features {
                let flat = out * self.in_features + input;
                let nibble = if flat & 1 == 0 {
                    self.packed_weight[flat / 2] & 0x0f
                } else {
                    self.packed_weight[flat / 2] >> 4
                };
                values[input + out * self.in_features] =
                    e2m1_value(nibble) * e4m3_value(self.weight_scale[out * blocks + input / 16]);
            }
        }
        values
    }

    /// Returns the logical cuBLASLt TN matrix dimensions, `K x M`.
    pub fn tn_weight_shape(&self) -> (usize, usize) {
        (self.in_features, self.out_features)
    }

    /// Returns expected packed value bytes.
    pub fn expected_weight_bytes(&self) -> usize {
        self.out_features * self.in_features / 2
    }

    /// Returns expected ModelOpt K16 scale bytes.
    pub fn expected_scale_bytes(&self) -> usize {
        self.out_features * self.in_features / 16
    }

    /// Repackages ModelOpt scales into cuBLASLt's tiled VEC16 layout.
    pub fn cublaslt_scales(&self) -> Vec<u8> {
        modelopt_scales_to_cublaslt(self.out_features, self.in_features, &self.weight_scale)
    }

    /// Concatenates two records along their output-feature dimension.
    pub fn concat_out_features(
        prefix: impl Into<String>,
        first: &Self,
        second: &Self,
    ) -> Result<Self> {
        if first.in_features != second.in_features {
            return Err(Error::Shape {
                label: "ModelOpt concat in_features",
                expected: first.in_features.to_string(),
                actual: second.in_features.to_string(),
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
        let mut packed_weight = first.packed_weight.clone();
        packed_weight.extend_from_slice(&second.packed_weight);
        let mut weight_scale = first.weight_scale.clone();
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
    /// Imports an FP8 linear from one safetensors shard.
    pub fn from_shard(shard: &SafeTensorShard, prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let input_scale_name = format!("{prefix}.input_scale");
        let weight_info = shard.require_tensor(&weight_name)?;
        let (out_features, in_features) = validate_fp8_weight(weight_info)?;
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

    /// Returns expected FP8 weight bytes.
    pub fn expected_weight_bytes(&self) -> usize {
        self.out_features * self.in_features
    }

    /// Concatenates two records along their output-feature dimension.
    pub fn concat_out_features(
        prefix: impl Into<String>,
        first: &Self,
        second: &Self,
    ) -> Result<Self> {
        if first.in_features != second.in_features {
            return Err(Error::Shape {
                label: "FP8 concat in_features",
                expected: first.in_features.to_string(),
                actual: second.in_features.to_string(),
            });
        }
        if first.weight_scale.to_bits() != second.weight_scale.to_bits()
            || first.input_scale.map(f32::to_bits) != second.input_scale.map(f32::to_bits)
            || first.channel_weight_scale.is_some() != second.channel_weight_scale.is_some()
        {
            return Err(Error::Format {
                label: "FP8 concat scales",
                detail: "projections use incompatible tensor, channel, or activation scales"
                    .to_string(),
            });
        }
        let mut weight = first.weight.clone();
        weight.extend_from_slice(&second.weight);
        let channel_weight_scale = match (&first.channel_weight_scale, &second.channel_weight_scale)
        {
            (Some(first), Some(second)) => {
                let mut scales = first.clone();
                scales.extend_from_slice(second);
                Some(scales)
            }
            (None, None) => None,
            _ => unreachable!("scale presence was validated"),
        };
        Ok(Self {
            prefix: prefix.into(),
            out_features: first.out_features + second.out_features,
            in_features: first.in_features,
            weight,
            weight_scale: first.weight_scale,
            channel_weight_scale,
            input_scale: first.input_scale,
        })
    }
}

fn checked_elements(label: &'static str, out_features: usize, in_features: usize) -> Result<usize> {
    out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::Shape {
            label,
            expected: "out_features * in_features without overflow".to_string(),
            actual: format!("out_features={out_features} in_features={in_features}"),
        })
}

fn validate_k16_shape(label: &'static str, out_features: usize, in_features: usize) -> Result<()> {
    if out_features == 0 || in_features == 0 || !in_features.is_multiple_of(16) {
        return Err(Error::Shape {
            label,
            expected: "non-zero dimensions and in_features divisible by 16".to_string(),
            actual: format!("out_features={out_features} in_features={in_features}"),
        });
    }
    Ok(())
}

fn input_scale_or_unity(shard: &SafeTensorShard, name: &str) -> Result<f32> {
    if shard.tensor(name).is_none() {
        Ok(1.0)
    } else {
        shard.read_scalar_f32(name)
    }
}

fn validate_nvfp4_weight(info: &SafeTensorInfo) -> Result<(usize, usize)> {
    if info.dtype != "U8" || info.shape.len() != 2 {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight",
            expected: "dtype=U8 shape=[out,in/2]".to_string(),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let out_features = info.shape[0];
    let in_features = info.shape[1].checked_mul(2).ok_or_else(|| Error::Shape {
        label: "ModelOpt NVFP4 weight",
        expected: "input width without overflow".to_string(),
        actual: info.shape[1].to_string(),
    })?;
    validate_k16_shape("ModelOpt NVFP4 weight", out_features, in_features)?;
    if info.byte_len() != (out_features * in_features / 2) as u64 {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight bytes",
            expected: (out_features * in_features / 2).to_string(),
            actual: info.byte_len().to_string(),
        });
    }
    Ok((out_features, in_features))
}

fn validate_nvfp4_scales(
    info: &SafeTensorInfo,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    let expected_shape = [out_features, in_features / 16];
    let expected_bytes = out_features * in_features / 16;
    if info.dtype != "F8_E4M3"
        || info.shape != expected_shape
        || info.byte_len() != expected_bytes as u64
    {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 weight_scale",
            expected: format!("dtype=F8_E4M3 shape={expected_shape:?} bytes={expected_bytes}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    Ok(())
}

fn validate_fp8_weight(info: &SafeTensorInfo) -> Result<(usize, usize)> {
    if info.dtype != "F8_E4M3"
        || info.shape.len() != 2
        || info.shape.contains(&0)
        || info.byte_len() != (info.shape[0] * info.shape[1]) as u64
    {
        return Err(Error::Shape {
            label: "ModelOpt FP8 weight",
            expected: "dtype=F8_E4M3 non-empty shape=[out,in] with matching byte count".to_string(),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    Ok((info.shape[0], info.shape[1]))
}

fn validate_nvfp4_expert_weight(info: &SafeTensorInfo) -> Result<(usize, usize, usize)> {
    if info.dtype != "U8" || info.shape.len() != 3 {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 expert weight",
            expected: "dtype=U8 shape=[experts,out,in/2]".to_string(),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let experts = info.shape[0];
    let out_features = info.shape[1];
    let in_features = info.shape[2].checked_mul(2).ok_or_else(|| Error::Shape {
        label: "ModelOpt NVFP4 expert weight",
        expected: "input width without overflow".to_string(),
        actual: info.shape[2].to_string(),
    })?;
    let expected = experts
        .checked_mul(out_features)
        .and_then(|value| value.checked_mul(in_features))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| Error::Shape {
            label: "ModelOpt NVFP4 expert weight",
            expected: "byte count without overflow".to_string(),
            actual: format!("experts={experts} out={out_features} in={in_features}"),
        })?;
    if experts == 0
        || out_features == 0
        || in_features == 0
        || !in_features.is_multiple_of(16)
        || info.byte_len() != expected as u64
    {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 expert weight bytes",
            expected: expected.to_string(),
            actual: info.byte_len().to_string(),
        });
    }
    Ok((experts, out_features, in_features))
}

fn validate_nvfp4_expert_scales(
    info: &SafeTensorInfo,
    experts: usize,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    let expected_shape = [experts, out_features, in_features / 16];
    let expected = experts * out_features * in_features / 16;
    if info.dtype != "F8_E4M3" || info.shape != expected_shape || info.byte_len() != expected as u64
    {
        return Err(Error::Shape {
            label: "ModelOpt NVFP4 expert weight_scale",
            expected: format!("dtype=F8_E4M3 shape={expected_shape:?} bytes={expected}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    Ok(())
}

fn validate_block_scaled_fp8_scales(
    info: &SafeTensorInfo,
    out_features: usize,
    in_features: usize,
    dtypes: &[&str],
) -> Result<()> {
    if !out_features.is_multiple_of(128) || !in_features.is_multiple_of(128) {
        return Err(Error::Shape {
            label: "ModelOpt block-scaled FP8 dimensions",
            expected: "out and in divisible by 128".to_string(),
            actual: format!("out={out_features} in={in_features}"),
        });
    }
    let expected_shape = [out_features / 128, in_features / 128];
    let expected = expected_shape[0] * expected_shape[1];
    if !dtypes.contains(&info.dtype.as_str())
        || info.shape != expected_shape
        || info.byte_len() != expected as u64
    {
        return Err(Error::Shape {
            label: "ModelOpt block-scaled FP8 scale",
            expected: format!("dtype in {dtypes:?} shape={expected_shape:?} bytes={expected}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    Ok(())
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

/// Repackages row-major ModelOpt K16 scales into cuBLASLt VEC16 storage.
pub fn modelopt_scales_to_cublaslt(
    out_features: usize,
    in_features: usize,
    scales: &[u8],
) -> Vec<u8> {
    let blocks = in_features / 16;
    let mut packed = vec![0; cublaslt_scale_layout_len(out_features, in_features)];
    for out in 0..out_features {
        for block in 0..blocks {
            packed[cublaslt_scale_offset(out, block, in_features)] = scales[out * blocks + block];
        }
    }
    packed
}

fn e2m1_code(value: f32) -> u8 {
    const VALUES: [(u8, f32); 8] = [
        (0, 0.0),
        (1, 0.5),
        (2, 1.0),
        (3, 1.5),
        (4, 2.0),
        (5, 3.0),
        (6, 4.0),
        (7, 6.0),
    ];
    if value.is_nan() {
        return 7;
    }
    let sign = if value.is_sign_negative() { 8 } else { 0 };
    let absolute = value.abs();
    if absolute > 6.0 {
        return sign | 7;
    }
    let mut best = 0;
    let mut error = f32::INFINITY;
    for (code, candidate) in VALUES {
        let candidate_error = (absolute - candidate).abs();
        if candidate_error < error
            || (candidate_error == error && (code & 1) == 0 && (best & 1) == 1)
        {
            best = code;
            error = candidate_error;
        }
    }
    sign | best
}

fn e2m1_value(code: u8) -> f32 {
    let magnitude = match code & 7 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if code & 8 == 0 { magnitude } else { -magnitude }
}

fn e4m3_value(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 7;
    if exponent == 0 {
        sign * mantissa as f32 * 2.0f32.powi(-9)
    } else if exponent == 0x0f && mantissa == 7 {
        f32::NAN
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }
}

fn e8m0_value(code: u8) -> f32 {
    if code == u8::MAX {
        f32::NAN
    } else {
        2.0f32.powi(i32::from(code) - 127)
    }
}

fn ue4m3_code(value: f32) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let mut best = 0;
    let mut error = value.abs();
    for code in 1..=0x7e {
        let candidate_error = (value - e4m3_value(code)).abs();
        if candidate_error < error
            || (candidate_error == error && (code & 1) == 0 && (best & 1) == 1)
        {
            best = code;
            error = candidate_error;
        }
    }
    best
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn cublaslt_scale_layout_len(outer: usize, inner: usize) -> usize {
    let blocks = inner.div_ceil(16);
    let inner = blocks.div_ceil(4) * 4;
    outer.div_ceil(128) * 128 * inner
}

fn cublaslt_scale_offset(outer: usize, block: usize, inner: usize) -> usize {
    let blocks = inner.div_ceil(16);
    let padded_blocks = blocks.div_ceil(4) * 4;
    let tile_base = ((block / 4) * 4 + (outer / 128) * padded_blocks) * 128;
    tile_base + (outer % 32) * 16 + (outer % 128 / 32) * 4 + block % 4
}

#[cfg(test)]
mod tests {
    use super::{ModelOptFp8Linear, ModelOptNvfp4Linear, modelopt_scales_to_cublaslt};

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        (rounded >> 16) as u16
    }

    #[test]
    fn bf16_quantization_preserves_modelopt_k16_encoding() {
        let row = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let values = row.into_iter().map(bf16).collect::<Vec<_>>();
        let weight = ModelOptNvfp4Linear::quantize_bf16("test", 1, 16, &values)
            .expect("quantize ModelOpt record");
        assert_eq!(weight.weight_scale, [0x38]);
        assert_eq!(
            weight.packed_weight,
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]
        );
        assert_eq!(weight.dequantize_to_f32_col_major(), row);
    }

    #[test]
    fn fp8_record_concat_preserves_channel_scales() {
        let first = ModelOptFp8Linear {
            prefix: "gate".to_string(),
            out_features: 1,
            in_features: 2,
            weight: vec![1, 2],
            weight_scale: 1.0,
            channel_weight_scale: Some(vec![0.5]),
            input_scale: None,
        };
        let second = ModelOptFp8Linear {
            prefix: "up".to_string(),
            out_features: 1,
            in_features: 2,
            weight: vec![3, 4],
            weight_scale: 1.0,
            channel_weight_scale: Some(vec![2.0]),
            input_scale: None,
        };
        let joined = ModelOptFp8Linear::concat_out_features("gate_up", &first, &second)
            .expect("concatenate records");
        assert_eq!(joined.weight, [1, 2, 3, 4]);
        assert_eq!(joined.channel_weight_scale, Some(vec![0.5, 2.0]));
    }

    #[test]
    fn cublaslt_scale_repack_uses_documented_tile_layout() {
        let source = (0..130 * 5).map(|value| value as u8).collect::<Vec<_>>();
        let packed = modelopt_scales_to_cublaslt(130, 80, &source);
        assert_eq!(packed.len(), 2048);
        assert_eq!(packed[0], source[0]);
        assert_eq!(packed[16], source[5]);
        assert_eq!(packed[4], source[32 * 5]);
    }
}
