#![allow(clippy::too_many_arguments)]

//! Iterative Qwen3 decode path over the full NVFP4 checkpoint.

use crate::kv_cache::KvCache;
use nvfp4::{
    ArgmaxResult, CublasLt, CudaEvent, CudaGraphExec, CudaStream, CutlassFp4GroupedGemvF32Plan,
    DeviceBuffer, Error, F32Matrix, Fp4TnMatmulPlan, GemmShape, GroupedGemvPointerBuffers,
    GroupedGemvPointerTableBuffers, ModelOptCheckpoint, ModelOptCublasLtWeight,
    ModelOptNvfp4Activation, ModelOptNvfp4Linear, MoeSiluQuantizeSlotBuffers, Nvfp4Matrix,
    Nvfp4TnInputs, Result, SafeTensorInfo, add_f32_into_on_stream, append_rows_f32_into_on_stream,
    argmax_f32_into_on_stream, bf16_linear_argmax_f32, bf16_linear_logits_f32_into_on_stream,
    copy_bf16_row_to_f32_indexed_into_on_stream, copy_row_f32_into_on_stream,
    fill_f32_into_on_stream, format, gather_nvfp4_grouped_gemv_ptr_tables_on_stream,
    gather_nvfp4_grouped_gemv_ptrs_on_stream,
    moe_silu_quantize_slots_nvfp4_simple_scales_on_stream, moe_topk_f32_into_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream, rms_norm_f32_into_on_stream,
    rms_norm_rope_neox_f32_indexed_into_on_stream, rope_neox_sequence_f32_into_on_stream,
    scaled_add_f32_into_on_stream, silu_mul_f32_into_on_stream, silu_mul_halves_f32_into_on_stream,
    silu_mul_halves_quantize_nvfp4_col_major_f32_into_on_stream, split_qkv_f32_into_on_stream,
    synchronize_device,
};
use serde_json::Value;
use std::cell::{Ref, RefCell};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct QwenModelConfig {
    hidden: usize,
    q_width: usize,
    kv_width: usize,
    intermediate: usize,
    layers: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_theta: f32,
    vocab: usize,
    ffn: QwenFfnConfig,
}

#[derive(Clone, Copy, Debug)]
pub enum QwenFfnConfig {
    Dense,
    Moe {
        experts: usize,
        experts_per_token: usize,
        expert_intermediate: usize,
        norm_topk_prob: bool,
    },
}

/// Model-family shape discovered from a Qwen Hugging Face config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenArchitecture {
    /// Existing dense/MoE Qwen3 decoder path with full attention in every layer.
    Qwen3,
    /// Hybrid Qwen3.5/Qwen3.6 text architecture with Gated Delta Net layers.
    Qwen35Moe,
}

/// Per-layer attention implementation used by the text stack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenLayerKind {
    /// Standard causal attention with KV cache.
    FullAttention,
    /// Qwen3.5/3.6 Gated Delta Net recurrent layer.
    LinearAttention,
}

/// Compact model manifest used to validate checkpoint support before loading.
#[derive(Clone, Debug)]
pub struct QwenModelManifest {
    /// Model architecture family.
    pub architecture: QwenArchitecture,
    /// Tensor prefix for the text model, e.g. `model` or `model.language_model`.
    pub tensor_prefix: String,
    /// Hidden width.
    pub hidden: usize,
    /// Number of text layers.
    pub layers: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Dense intermediate width, or the config default used beside MoE metadata.
    pub intermediate: usize,
    /// Query heads for full attention.
    pub q_heads: usize,
    /// KV heads for full attention.
    pub kv_heads: usize,
    /// Full-attention head dimension.
    pub head_dim: usize,
    /// Number of full-attention head channels that receive RoPE.
    pub rotary_dim: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE theta.
    pub rope_theta: f32,
    /// MRoPE/IMRoPE section sizes `[v0,v1,v2,v3]` (t,h,w,extra), when present.
    /// When `None`, full-attention layers use standard partial Neox RoPE.
    pub mrope_sections: Option<[usize; 4]>,
    /// FFN/MoE shape.
    pub ffn: QwenFfnConfig,
    /// Per-layer attention schedule.
    pub layer_kinds: Vec<QwenLayerKind>,
    /// Qwen3.5/3.6 linear-attention shape, when present.
    pub linear_attention: Option<QwenLinearAttentionConfig>,
    /// Shared expert intermediate size, when present.
    pub shared_expert_intermediate: Option<usize>,
    /// Number of MTP/next-token prediction layers stored in the checkpoint.
    pub mtp_layers: usize,
}

/// Qwen3.5/3.6 Gated Delta Net dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenLinearAttentionConfig {
    /// Depthwise convolution kernel size.
    pub conv_kernel: usize,
    /// Key head count before repeating to value heads.
    pub key_heads: usize,
    /// Value/recurrent head count.
    pub value_heads: usize,
    /// Key/query state dimension.
    pub key_head_dim: usize,
    /// Value/recurrent state dimension.
    pub value_head_dim: usize,
}

/// Tensor-level support summary for a Qwen checkpoint.
#[derive(Clone, Debug)]
pub struct QwenModelInspection {
    /// Parsed model manifest.
    pub manifest: QwenModelManifest,
    /// Tensor checks for representative required weights.
    pub tensors: Vec<QwenTensorCheck>,
}

/// One tensor metadata check from checkpoint inspection.
#[derive(Clone, Debug)]
pub struct QwenTensorCheck {
    /// Tensor name.
    pub name: String,
    /// Whether the checkpoint index contains this tensor.
    pub present: bool,
    /// Safetensors dtype if present.
    pub dtype: Option<String>,
    /// Safetensors shape if present.
    pub shape: Option<Vec<usize>>,
}

impl QwenModelConfig {
    fn ffn_label(self) -> String {
        match self.ffn {
            QwenFfnConfig::Dense => format!("dense intermediate={}", self.intermediate),
            QwenFfnConfig::Moe {
                experts,
                experts_per_token,
                expert_intermediate,
                ..
            } => format!(
                "moe experts={experts} top_k={experts_per_token} expert_intermediate={expert_intermediate}"
            ),
        }
    }
}

impl QwenModelConfig {
    fn load(model_dir: &Path) -> Result<Self> {
        let manifest = QwenModelManifest::load(model_dir)?;
        if manifest.architecture != QwenArchitecture::Qwen3
            || manifest.tensor_prefix != "model"
            || manifest
                .layer_kinds
                .iter()
                .any(|kind| *kind != QwenLayerKind::FullAttention)
        {
            return Err(Error::Format {
                label: "Qwen runtime architecture",
                detail: format!(
                    "unsupported architecture {:?}; use qwen-inspect-model until the hybrid decode path is implemented",
                    manifest.architecture
                ),
            });
        }
        let hidden = manifest.hidden;
        let intermediate = manifest.intermediate;
        let layers = manifest.layers;
        let q_heads = manifest.q_heads;
        let kv_heads = manifest.kv_heads;
        let head_dim = manifest.head_dim;
        let vocab = manifest.vocab;
        let rms_eps = manifest.rms_eps;
        let rope_theta = manifest.rope_theta;
        let ffn = manifest.ffn;
        let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
            label: "Qwen config KV width",
            expected: "num_key_value_heads * head_dim without overflow".to_string(),
            actual: format!("num_key_value_heads={kv_heads} head_dim={head_dim}"),
        })?;
        let q_width = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
            label: "Qwen config Q width",
            expected: "num_attention_heads * head_dim without overflow".to_string(),
            actual: format!("num_attention_heads={q_heads} head_dim={head_dim}"),
        })?;
        Ok(Self {
            hidden,
            q_width,
            kv_width,
            intermediate,
            layers,
            q_heads,
            kv_heads,
            head_dim,
            rms_eps,
            rope_theta,
            vocab,
            ffn,
        })
    }
}

impl QwenModelManifest {
    /// Parses a Qwen config without loading tensor payloads.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let json = read_config_json(model_dir)?;
        let root_model_type = json
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (architecture, text, tensor_prefix) = if root_model_type == "qwen3_5_moe" {
            let text = json.get("text_config").ok_or_else(|| Error::Format {
                label: "Qwen config",
                detail: "qwen3_5_moe config missing text_config".to_string(),
            })?;
            (
                QwenArchitecture::Qwen35Moe,
                text,
                "model.language_model".to_string(),
            )
        } else {
            (QwenArchitecture::Qwen3, &json, "model".to_string())
        };

        let hidden = required_usize(text, "hidden_size")?;
        let layers = required_usize(text, "num_hidden_layers")?;
        let q_heads = required_usize(text, "num_attention_heads")?;
        let kv_heads = required_usize(text, "num_key_value_heads")?;
        let head_dim = required_usize(text, "head_dim")?;
        let partial_rotary_factor = optional_f32(text, "partial_rotary_factor")?
            .or_else(|| {
                text.get("rope_parameters")
                    .and_then(|rope| rope.get("partial_rotary_factor"))
                    .and_then(Value::as_f64)
                    .map(|value| value as f32)
            })
            .unwrap_or(1.0);
        if !partial_rotary_factor.is_finite()
            || partial_rotary_factor <= 0.0
            || partial_rotary_factor > 1.0
        {
            return Err(Error::Format {
                label: "Qwen config",
                detail: format!(
                    "invalid partial_rotary_factor {partial_rotary_factor}; expected 0 < value <= 1"
                ),
            });
        }
        let rotary_dim = ((head_dim as f32) * partial_rotary_factor) as usize;
        if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
            return Err(Error::Shape {
                label: "Qwen rotary dimension",
                expected: "non-zero even rotary_dim <= head_dim".to_string(),
                actual: format!(
                    "head_dim={head_dim} partial_rotary_factor={partial_rotary_factor} rotary_dim={rotary_dim}"
                ),
            });
        }
        let vocab = required_usize(text, "vocab_size")?;
        let rms_eps = required_f32(text, "rms_norm_eps")?;
        let rope_theta = optional_f32(text, "rope_theta")?
            .or_else(|| {
                text.get("rope_parameters")
                    .and_then(|rope| rope.get("rope_theta"))
                    .and_then(Value::as_f64)
                    .map(|value| value as f32)
            })
            .ok_or_else(|| Error::Format {
                label: "Qwen config",
                detail: "missing rope_theta or rope_parameters.rope_theta".to_string(),
            })?;
        let mrope_sections = text
            .get("rope_parameters")
            .and_then(|rope| rope.get("mrope_section"))
            .and_then(Value::as_array)
            .map(|values| {
                if values.is_empty() || values.len() > 4 {
                    return Err(Error::Format {
                        label: "Qwen config mrope_section",
                        detail: format!("expected 1..=4 entries, got {}", values.len()),
                    });
                }
                let mut sections = [0usize; 4];
                for (idx, value) in values.iter().enumerate() {
                    sections[idx] = value.as_u64().ok_or_else(|| Error::Format {
                        label: "Qwen config mrope_section",
                        detail: format!("section {idx} is not an integer"),
                    })? as usize;
                }
                Ok(sections)
            })
            .transpose()?;
        let experts = optional_usize(text, "num_experts")?.unwrap_or(0);
        let experts_per_token = optional_usize(text, "num_experts_per_tok")?
            .or(optional_usize(text, "num_experts_used")?)
            .unwrap_or(0);
        let intermediate = optional_usize(text, "intermediate_size")?.unwrap_or(0);
        let expert_intermediate = optional_usize(text, "moe_intermediate_size")?
            .or(optional_usize(text, "expert_feed_forward_length")?)
            .unwrap_or(intermediate);
        let norm_topk_prob = optional_bool(text, "norm_topk_prob")?.unwrap_or(true);
        let ffn = if experts > 0 {
            if experts_per_token == 0 || experts_per_token > experts {
                return Err(Error::Shape {
                    label: "Qwen MoE config",
                    expected: "0 < num_experts_per_tok <= num_experts".to_string(),
                    actual: format!(
                        "num_experts={experts} num_experts_per_tok={experts_per_token}"
                    ),
                });
            }
            QwenFfnConfig::Moe {
                experts,
                experts_per_token,
                expert_intermediate,
                norm_topk_prob,
            }
        } else {
            if intermediate == 0 {
                return Err(Error::Format {
                    label: "Qwen config",
                    detail: "missing intermediate_size for dense FFN".to_string(),
                });
            }
            QwenFfnConfig::Dense
        };
        let layer_kinds = parse_layer_kinds(text, layers, architecture)?;
        let linear_attention = if layer_kinds.contains(&QwenLayerKind::LinearAttention) {
            Some(QwenLinearAttentionConfig {
                conv_kernel: required_usize(text, "linear_conv_kernel_dim")?,
                key_heads: required_usize(text, "linear_num_key_heads")?,
                value_heads: required_usize(text, "linear_num_value_heads")?,
                key_head_dim: required_usize(text, "linear_key_head_dim")?,
                value_head_dim: required_usize(text, "linear_value_head_dim")?,
            })
        } else {
            None
        };
        Ok(Self {
            architecture,
            tensor_prefix,
            hidden,
            layers,
            vocab,
            intermediate,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rms_eps,
            rope_theta,
            mrope_sections,
            ffn,
            layer_kinds,
            linear_attention,
            shared_expert_intermediate: optional_usize(text, "shared_expert_intermediate_size")?,
            mtp_layers: optional_usize(text, "mtp_num_hidden_layers")?.unwrap_or(0),
        })
    }

    /// Parses config and checks representative tensors in the safetensors index.
    pub fn inspect(model_dir: &Path) -> Result<QwenModelInspection> {
        let manifest = Self::load(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let mut tensors = Vec::new();
        let prefix = manifest.tensor_prefix.as_str();
        push_tensor_check(
            &checkpoint,
            &mut tensors,
            format!("{prefix}.embed_tokens.weight"),
        )?;
        push_tensor_check(&checkpoint, &mut tensors, format!("{prefix}.norm.weight"))?;
        push_tensor_check(&checkpoint, &mut tensors, "lm_head.weight".to_string())?;
        push_tensor_check(
            &checkpoint,
            &mut tensors,
            format!("{prefix}.layers.0.input_layernorm.weight"),
        )?;
        push_tensor_check(
            &checkpoint,
            &mut tensors,
            format!("{prefix}.layers.0.post_attention_layernorm.weight"),
        )?;
        match manifest.layer_kinds.first().copied() {
            Some(QwenLayerKind::LinearAttention) => {
                for suffix in [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.conv1d.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.out_proj.weight",
                ] {
                    push_tensor_check(
                        &checkpoint,
                        &mut tensors,
                        format!("{prefix}.layers.0.{suffix}"),
                    )?;
                }
            }
            Some(QwenLayerKind::FullAttention) => {
                for suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                ] {
                    push_tensor_check(
                        &checkpoint,
                        &mut tensors,
                        format!("{prefix}.layers.0.{suffix}"),
                    )?;
                }
            }
            None => {}
        }
        if let Some(full_layer) = manifest
            .layer_kinds
            .iter()
            .position(|kind| *kind == QwenLayerKind::FullAttention)
            .filter(|layer| *layer != 0)
        {
            for suffix in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ] {
                push_tensor_check(
                    &checkpoint,
                    &mut tensors,
                    format!("{prefix}.layers.{full_layer}.{suffix}"),
                )?;
            }
        }
        if let QwenFfnConfig::Moe { .. } = manifest.ffn {
            for suffix in [
                "mlp.gate.weight",
                "mlp.experts.0.gate_proj.weight",
                "mlp.experts.0.up_proj.weight",
                "mlp.experts.0.down_proj.weight",
            ] {
                push_tensor_check(
                    &checkpoint,
                    &mut tensors,
                    format!("{prefix}.layers.0.{suffix}"),
                )?;
            }
            if manifest.shared_expert_intermediate.is_some() {
                for suffix in [
                    "mlp.shared_expert.gate_proj.weight",
                    "mlp.shared_expert.up_proj.weight",
                    "mlp.shared_expert.down_proj.weight",
                    "mlp.shared_expert_gate.weight",
                ] {
                    push_tensor_check(
                        &checkpoint,
                        &mut tensors,
                        format!("{prefix}.layers.0.{suffix}"),
                    )?;
                }
            }
        }
        if manifest.mtp_layers > 0 {
            push_tensor_check(&checkpoint, &mut tensors, "mtp.fc.weight".to_string())?;
        }
        Ok(QwenModelInspection { manifest, tensors })
    }
}

fn read_config_json(model_dir: &Path) -> Result<Value> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|error| Error::Format {
        label: "Qwen config",
        detail: format!("failed to read {}: {error}", path.display()),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| Error::Format {
        label: "Qwen config",
        detail: format!("failed to parse {}: {error}", path.display()),
    })
}

fn parse_layer_kinds(
    text: &Value,
    layers: usize,
    architecture: QwenArchitecture,
) -> Result<Vec<QwenLayerKind>> {
    if let Some(values) = text.get("layer_types") {
        let values = values.as_array().ok_or_else(|| Error::Format {
            label: "Qwen config layer_types",
            detail: "layer_types is not an array".to_string(),
        })?;
        if values.len() != layers {
            return Err(Error::Shape {
                label: "Qwen config layer_types",
                expected: format!("{layers} entries"),
                actual: values.len().to_string(),
            });
        }
        return values
            .iter()
            .map(|value| match value.as_str() {
                Some("full_attention") => Ok(QwenLayerKind::FullAttention),
                Some("linear_attention") => Ok(QwenLayerKind::LinearAttention),
                Some(other) => Err(Error::Format {
                    label: "Qwen config layer_types",
                    detail: format!("unsupported layer type {other}"),
                }),
                None => Err(Error::Format {
                    label: "Qwen config layer_types",
                    detail: "layer type is not a string".to_string(),
                }),
            })
            .collect();
    }
    if architecture == QwenArchitecture::Qwen35Moe {
        let interval = optional_usize(text, "full_attention_interval")?.unwrap_or(4);
        Ok((0..layers)
            .map(|idx| {
                if (idx + 1) % interval == 0 {
                    QwenLayerKind::FullAttention
                } else {
                    QwenLayerKind::LinearAttention
                }
            })
            .collect())
    } else {
        Ok(vec![QwenLayerKind::FullAttention; layers])
    }
}

fn push_tensor_check(
    checkpoint: &ModelOptCheckpoint,
    tensors: &mut Vec<QwenTensorCheck>,
    name: String,
) -> Result<()> {
    let info: Option<SafeTensorInfo> = if checkpoint.contains_tensor(&name) {
        Some(checkpoint.tensor_info(&name)?)
    } else {
        None
    };
    tensors.push(QwenTensorCheck {
        name,
        present: info.is_some(),
        dtype: info.as_ref().map(|info| info.dtype.clone()),
        shape: info.map(|info| info.shape),
    });
    Ok(())
}

/// Returns a permutation from HF grouped V-head order to llama.cpp/ggml tiled order.
///
/// HF stores value-side linear-attention heads grouped by key head:
/// `[k0_v0, k0_v1, k1_v0, k1_v1, ...]`. The fused delta-net path wants tiled
/// order for cheap broadcast against repeated Q/K heads:
/// `[k0_v0, k1_v0, k0_v1, k1_v1, ...]`.
pub fn qwen35_v_head_tiled_permutation(
    num_key_heads: usize,
    num_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<usize>> {
    if num_key_heads == 0
        || num_value_heads == 0
        || head_dim == 0
        || !num_value_heads.is_multiple_of(num_key_heads)
    {
        return Err(Error::Shape {
            label: "Qwen3.5 V-head permutation",
            expected: "non-zero heads with value_heads divisible by key_heads".to_string(),
            actual: format!(
                "key_heads={num_key_heads} value_heads={num_value_heads} head_dim={head_dim}"
            ),
        });
    }
    let values_per_key = num_value_heads / num_key_heads;
    let mut perm = Vec::with_capacity(num_value_heads * head_dim);
    for value_group in 0..values_per_key {
        for key_head in 0..num_key_heads {
            let head = key_head * values_per_key + value_group;
            for dim in 0..head_dim {
                perm.push(head * head_dim + dim);
            }
        }
    }
    Ok(perm)
}

/// Reorders rows from HF grouped V-head order to tiled order.
pub fn qwen35_reorder_rows_grouped_to_tiled<T: Copy>(
    values: &[T],
    row_width: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<T>> {
    let rows = num_value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.5 V-head reorder rows",
            expected: "value_heads * head_dim without overflow".to_string(),
            actual: format!("value_heads={num_value_heads} head_dim={head_dim}"),
        })?;
    if row_width == 0 || values.len() != rows * row_width {
        return Err(Error::Shape {
            label: "Qwen3.5 V-head reorder rows",
            expected: format!("{} values", rows * row_width),
            actual: values.len().to_string(),
        });
    }
    let perm = qwen35_v_head_tiled_permutation(num_key_heads, num_value_heads, head_dim)?;
    let mut out = Vec::with_capacity(values.len());
    for row in perm {
        let start = row * row_width;
        out.extend_from_slice(&values[start..start + row_width]);
    }
    Ok(out)
}

/// Reorders the V rows of a concatenated `[Q; K; V]` linear-attention projection.
pub fn qwen35_reorder_qkv_rows_grouped_to_tiled<T: Copy>(
    values: &[T],
    row_width: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
) -> Result<Vec<T>> {
    let qk_rows = num_key_heads
        .checked_mul(key_head_dim)
        .and_then(|rows| rows.checked_mul(2))
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.5 QKV reorder rows",
            expected: "2 * key_heads * key_head_dim without overflow".to_string(),
            actual: format!("key_heads={num_key_heads} key_head_dim={key_head_dim}"),
        })?;
    let v_rows = num_value_heads
        .checked_mul(value_head_dim)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.5 QKV reorder rows",
            expected: "value_heads * value_head_dim without overflow".to_string(),
            actual: format!("value_heads={num_value_heads} value_head_dim={value_head_dim}"),
        })?;
    if row_width == 0 || values.len() != (qk_rows + v_rows) * row_width {
        return Err(Error::Shape {
            label: "Qwen3.5 QKV reorder rows",
            expected: format!("{} values", (qk_rows + v_rows) * row_width),
            actual: values.len().to_string(),
        });
    }
    let split = qk_rows * row_width;
    let mut out = Vec::with_capacity(values.len());
    out.extend_from_slice(&values[..split]);
    out.extend(qwen35_reorder_rows_grouped_to_tiled(
        &values[split..],
        row_width,
        num_key_heads,
        num_value_heads,
        value_head_dim,
    )?);
    Ok(out)
}

fn required_usize(json: &Value, key: &'static str) -> Result<usize> {
    json.get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Format {
            label: "Qwen config",
            detail: format!("missing or invalid integer field {key}"),
        })
}

fn required_f32(json: &Value, key: &'static str) -> Result<f32> {
    json.get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .ok_or_else(|| Error::Format {
            label: "Qwen config",
            detail: format!("missing or invalid float field {key}"),
        })
}

fn optional_f32(json: &Value, key: &'static str) -> Result<Option<f32>> {
    match json.get(key) {
        Some(value) => value
            .as_f64()
            .map(|value| Some(value as f32))
            .ok_or_else(|| Error::Format {
                label: "Qwen config",
                detail: format!("invalid float field {key}"),
            }),
        None => Ok(None),
    }
}

fn optional_usize(json: &Value, key: &'static str) -> Result<Option<usize>> {
    match json.get(key) {
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| Error::Format {
                label: "Qwen config",
                detail: format!("invalid integer field {key}"),
            }),
        None => Ok(None),
    }
}

fn optional_bool(json: &Value, key: &'static str) -> Result<Option<bool>> {
    match json.get(key) {
        Some(value) => value.as_bool().map(Some).ok_or_else(|| Error::Format {
            label: "Qwen config",
            detail: format!("invalid bool field {key}"),
        }),
        None => Ok(None),
    }
}

static RUNTIME_COUNTERS: RuntimeAtomicCounters = RuntimeAtomicCounters::new();

struct RuntimeAtomicCounters {
    fp4_gemm_calls: AtomicU64,
    fp4_gemm_m_total: AtomicU64,
    fp4_gemm_n_total: AtomicU64,
    fp4_gemm_k_total: AtomicU64,
    quantize_calls: AtomicU64,
    rms_norm_calls: AtomicU64,
    rope_calls: AtomicU64,
    attention_calls: AtomicU64,
    silu_calls: AtomicU64,
    add_calls: AtomicU64,
    bf16_to_f32_calls: AtomicU64,
    lm_head_argmax_calls: AtomicU64,
    lm_head_logits_calls: AtomicU64,
    host_logits_bytes: AtomicU64,
}

impl RuntimeAtomicCounters {
    const fn new() -> Self {
        Self {
            fp4_gemm_calls: AtomicU64::new(0),
            fp4_gemm_m_total: AtomicU64::new(0),
            fp4_gemm_n_total: AtomicU64::new(0),
            fp4_gemm_k_total: AtomicU64::new(0),
            quantize_calls: AtomicU64::new(0),
            rms_norm_calls: AtomicU64::new(0),
            rope_calls: AtomicU64::new(0),
            attention_calls: AtomicU64::new(0),
            silu_calls: AtomicU64::new(0),
            add_calls: AtomicU64::new(0),
            bf16_to_f32_calls: AtomicU64::new(0),
            lm_head_argmax_calls: AtomicU64::new(0),
            lm_head_logits_calls: AtomicU64::new(0),
            host_logits_bytes: AtomicU64::new(0),
        }
    }
}

/// Snapshot of cumulative Qwen runtime operation counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct QwenRuntimeCounters {
    /// Number of FP4 GEMM launches.
    pub fp4_gemm_calls: u64,
    /// Sum of FP4 GEMM M dimensions.
    pub fp4_gemm_m_total: u64,
    /// Sum of FP4 GEMM N dimensions.
    pub fp4_gemm_n_total: u64,
    /// Sum of FP4 GEMM K dimensions.
    pub fp4_gemm_k_total: u64,
    /// Number of activation quantization calls.
    pub quantize_calls: u64,
    /// Number of RMSNorm calls.
    pub rms_norm_calls: u64,
    /// Number of RoPE calls.
    pub rope_calls: u64,
    /// Number of attention calls.
    pub attention_calls: u64,
    /// Number of SiLU multiply calls.
    pub silu_calls: u64,
    /// Number of residual add calls.
    pub add_calls: u64,
    /// Number of BF16-to-F32 conversion calls.
    pub bf16_to_f32_calls: u64,
    /// Number of lm-head GPU argmax calls.
    pub lm_head_argmax_calls: u64,
    /// Number of lm-head logits calls.
    pub lm_head_logits_calls: u64,
    /// Bytes copied from GPU logits to host.
    pub host_logits_bytes: u64,
}

impl QwenRuntimeCounters {
    /// Returns the saturating difference between two cumulative snapshots.
    pub fn saturating_sub(self, earlier: Self) -> Self {
        Self {
            fp4_gemm_calls: self.fp4_gemm_calls.saturating_sub(earlier.fp4_gemm_calls),
            fp4_gemm_m_total: self
                .fp4_gemm_m_total
                .saturating_sub(earlier.fp4_gemm_m_total),
            fp4_gemm_n_total: self
                .fp4_gemm_n_total
                .saturating_sub(earlier.fp4_gemm_n_total),
            fp4_gemm_k_total: self
                .fp4_gemm_k_total
                .saturating_sub(earlier.fp4_gemm_k_total),
            quantize_calls: self.quantize_calls.saturating_sub(earlier.quantize_calls),
            rms_norm_calls: self.rms_norm_calls.saturating_sub(earlier.rms_norm_calls),
            rope_calls: self.rope_calls.saturating_sub(earlier.rope_calls),
            attention_calls: self.attention_calls.saturating_sub(earlier.attention_calls),
            silu_calls: self.silu_calls.saturating_sub(earlier.silu_calls),
            add_calls: self.add_calls.saturating_sub(earlier.add_calls),
            bf16_to_f32_calls: self
                .bf16_to_f32_calls
                .saturating_sub(earlier.bf16_to_f32_calls),
            lm_head_argmax_calls: self
                .lm_head_argmax_calls
                .saturating_sub(earlier.lm_head_argmax_calls),
            lm_head_logits_calls: self
                .lm_head_logits_calls
                .saturating_sub(earlier.lm_head_logits_calls),
            host_logits_bytes: self
                .host_logits_bytes
                .saturating_sub(earlier.host_logits_bytes),
        }
    }
}

/// Returns cumulative operation counters for Qwen runtime calls.
pub fn runtime_counters() -> QwenRuntimeCounters {
    QwenRuntimeCounters {
        fp4_gemm_calls: load_counter(&RUNTIME_COUNTERS.fp4_gemm_calls),
        fp4_gemm_m_total: load_counter(&RUNTIME_COUNTERS.fp4_gemm_m_total),
        fp4_gemm_n_total: load_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total),
        fp4_gemm_k_total: load_counter(&RUNTIME_COUNTERS.fp4_gemm_k_total),
        quantize_calls: load_counter(&RUNTIME_COUNTERS.quantize_calls),
        rms_norm_calls: load_counter(&RUNTIME_COUNTERS.rms_norm_calls),
        rope_calls: load_counter(&RUNTIME_COUNTERS.rope_calls),
        attention_calls: load_counter(&RUNTIME_COUNTERS.attention_calls),
        silu_calls: load_counter(&RUNTIME_COUNTERS.silu_calls),
        add_calls: load_counter(&RUNTIME_COUNTERS.add_calls),
        bf16_to_f32_calls: load_counter(&RUNTIME_COUNTERS.bf16_to_f32_calls),
        lm_head_argmax_calls: load_counter(&RUNTIME_COUNTERS.lm_head_argmax_calls),
        lm_head_logits_calls: load_counter(&RUNTIME_COUNTERS.lm_head_logits_calls),
        host_logits_bytes: load_counter(&RUNTIME_COUNTERS.host_logits_bytes),
    }
}

fn load_counter(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

fn inc_counter(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

fn add_counter(counter: &AtomicU64, value: u64) {
    counter.fetch_add(value, Ordering::Relaxed);
}

fn time_stage<F>(stream: &CudaStream, enqueue: F) -> Result<f64>
where
    F: FnOnce() -> Result<()>,
{
    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record_on_stream(stream)?;
    enqueue()?;
    end.record_on_stream(stream)?;
    start.synchronize()?;
    end.synchronize()?;
    Ok(start.elapsed_ms_until(&end)? as f64)
}

fn record_one_token_decode_counters(config: QwenModelConfig) {
    add_counter(&RUNTIME_COUNTERS.fp4_gemm_calls, (config.layers * 4) as u64);
    let gemm_m_per_layer = (config.hidden
        + config.q_width
        + config.kv_width
        + config.kv_width
        + config.intermediate * 2
        + config.hidden) as u64;
    let gemm_k_per_layer =
        (config.hidden + config.q_width + config.hidden + config.intermediate) as u64;
    add_counter(
        &RUNTIME_COUNTERS.fp4_gemm_m_total,
        gemm_m_per_layer * config.layers as u64,
    );
    add_counter(
        &RUNTIME_COUNTERS.fp4_gemm_n_total,
        (config.layers * 4) as u64,
    );
    add_counter(
        &RUNTIME_COUNTERS.fp4_gemm_k_total,
        gemm_k_per_layer * config.layers as u64,
    );
    add_counter(&RUNTIME_COUNTERS.quantize_calls, (config.layers * 4) as u64);
    add_counter(
        &RUNTIME_COUNTERS.rms_norm_calls,
        (config.layers * 4 + 1) as u64,
    );
    add_counter(&RUNTIME_COUNTERS.rope_calls, (config.layers * 2) as u64);
    add_counter(&RUNTIME_COUNTERS.attention_calls, config.layers as u64);
    add_counter(&RUNTIME_COUNTERS.silu_calls, config.layers as u64);
    add_counter(&RUNTIME_COUNTERS.add_calls, (config.layers * 2) as u64);
    inc_counter(&RUNTIME_COUNTERS.lm_head_argmax_calls);
}

/// Result of a one-token full-model decode pass.
#[derive(Clone, Copy, Debug)]
pub struct NextToken {
    /// Input token id used for the decode step.
    pub input_token: u32,
    /// Argmax next token id.
    pub token: u32,
    /// Logit for `token`.
    pub logit: f32,
}

/// CPU-visible logits for the next token after a decode or prefill step.
#[derive(Clone, Debug)]
pub struct NextTokenLogits {
    /// Input token id at the position that produced these logits.
    pub input_token: u32,
    /// Full vocabulary logits copied from the GPU lm-head output.
    pub logits: Vec<f32>,
}

/// CUDA-event decode timing accumulated over one or more generated tokens.
#[derive(Clone, Copy, Debug, Default)]
pub struct QwenDecodeProfile {
    /// Number of profiled decode tokens.
    pub tokens: u64,
    /// Device time spent reading the current token embedding.
    pub embedding_ms: f64,
    /// Device time spent in per-layer input RMSNorm.
    pub input_norm_ms: f64,
    /// Device time spent quantizing Q/K/V activations.
    pub qkv_quantize_ms: f64,
    /// Device time spent in Q/K/V FP4 GEMMs.
    pub qkv_gemm_ms: f64,
    /// Device time spent in Q/K RMSNorm.
    pub qk_norm_ms: f64,
    /// Device time spent applying RoPE to Q/K.
    pub rope_ms: f64,
    /// Device time spent appending K/V rows to cache.
    pub kv_append_ms: f64,
    /// Device time spent in cached decode attention.
    pub attention_ms: f64,
    /// Device time spent quantizing attention output for O projection.
    pub o_quantize_ms: f64,
    /// Device time spent in O projection FP4 GEMM.
    pub o_gemm_ms: f64,
    /// Device time spent adding the attention residual.
    pub attn_residual_ms: f64,
    /// Device time spent in post-attention RMSNorm before the MLP.
    pub ffn_norm_ms: f64,
    /// Device time spent quantizing gate/up/down MLP activations.
    pub ffn_quantize_ms: f64,
    /// Device time spent in gate/up/down MLP FP4 GEMMs.
    pub ffn_gemm_ms: f64,
    /// Host wall time spent in the MoE FFN block when profiled.
    pub ffn_wall_ms: f64,
    /// Host wall time spent synchronizing/copying/selecting MoE route.
    pub moe_route_wall_ms: f64,
    /// Device time spent in the fused gate/up MLP FP4 GEMM.
    pub ffn_gate_up_gemm_ms: f64,
    /// Device time spent in the down-projection MLP FP4 GEMM.
    pub ffn_down_gemm_ms: f64,
    /// Device time spent in SiLU multiply.
    pub silu_ms: f64,
    /// Device time spent adding the MLP residual.
    pub ffn_residual_ms: f64,
    /// Device time spent in final RMSNorm.
    pub final_norm_ms: f64,
    /// Device time spent computing lm-head logits and GPU argmax.
    pub lm_head_argmax_ms: f64,
    /// Device time spent in Qwen3.6 router and top-k selection.
    pub qwen36_router_ms: f64,
    /// Device time spent in Qwen3.6 router projection.
    pub qwen36_router_linear_ms: f64,
    /// Device time spent in Qwen3.6 router top-k selection.
    pub qwen36_router_topk_ms: f64,
    /// Device time spent in Qwen3.6 routed gate/up expert work.
    pub qwen36_routed_gate_up_ms: f64,
    /// Device time spent in Qwen3.6 routed SiLU and down-input quantization.
    pub qwen36_routed_silu_quantize_ms: f64,
    /// Device time spent in Qwen3.6 routed down expert work and accumulation.
    pub qwen36_routed_down_ms: f64,
    /// Device time spent gathering Qwen3.6 routed down pointer tables.
    pub qwen36_routed_down_gather_ms: f64,
    /// Device time spent in Qwen3.6 routed down grouped GEMV.
    pub qwen36_routed_down_gemv_ms: f64,
    /// Device time spent weighted-accumulating Qwen3.6 routed down outputs.
    pub qwen36_routed_down_accum_ms: f64,
    /// Device time spent in Qwen3.6 shared gate/up projection.
    pub qwen36_shared_gate_up_ms: f64,
    /// Device time spent in Qwen3.6 shared SiLU.
    pub qwen36_shared_silu_ms: f64,
    /// Device time spent in Qwen3.6 shared down projection.
    pub qwen36_shared_down_ms: f64,
    /// Device time spent in Qwen3.6 shared gate projection and multiply.
    pub qwen36_shared_gate_ms: f64,
    /// Device time spent combining Qwen3.6 routed/shared FFN outputs and residual.
    pub qwen36_ffn_combine_ms: f64,
    /// Device time spent in Qwen3.6 Gated Delta Net linear-attention layers.
    pub qwen36_linear_attention_ms: f64,
    /// Device time spent in Qwen3.6 full-attention layers.
    pub qwen36_full_attention_ms: f64,
    /// Device time spent in Qwen3.6 linear-attention qkv projection.
    pub qwen36_linear_qkv_ms: f64,
    /// Device time spent in Qwen3.6 linear-attention z projection.
    pub qwen36_linear_z_ms: f64,
    /// Device time spent in Qwen3.6 linear-attention alpha/beta projections.
    pub qwen36_linear_alpha_beta_ms: f64,
    /// Device time spent preparing Q/K/V and convolution state for Qwen3.6 GDN.
    pub qwen36_linear_gdn_prep_ms: f64,
    /// Device time spent preparing GDN gate/beta for Qwen3.6.
    pub qwen36_linear_gdn_gate_ms: f64,
    /// Device time spent in Qwen3.6 Gated Delta Net recurrence.
    pub qwen36_linear_gdn_ms: f64,
    /// Device time spent in Qwen3.6 GDN gated RMSNorm.
    pub qwen36_linear_norm_ms: f64,
    /// Device time spent in Qwen3.6 linear-attention output projection.
    pub qwen36_linear_out_ms: f64,
}

impl QwenDecodeProfile {
    /// Total profiled CUDA-event time in milliseconds.
    pub fn total_ms(self) -> f64 {
        self.embedding_ms
            + self.input_norm_ms
            + self.qkv_quantize_ms
            + self.qkv_gemm_ms
            + self.qk_norm_ms
            + self.rope_ms
            + self.kv_append_ms
            + self.attention_ms
            + self.o_quantize_ms
            + self.o_gemm_ms
            + self.attn_residual_ms
            + self.ffn_norm_ms
            + self.ffn_quantize_ms
            + self.ffn_gemm_ms
            + self.silu_ms
            + self.ffn_residual_ms
            + self.final_norm_ms
            + self.lm_head_argmax_ms
    }
}

/// Loaded Qwen3-8B NVFP4 model state.
///
/// Weights are uploaded once and reused across decode steps. The current
/// implementation is single-sequence decode only.
pub struct Qwen3Model {
    config: QwenModelConfig,
    checkpoint: ModelOptCheckpoint,
    lt: CublasLt,
    embeddings: DeviceBuffer<u16>,
    layers: Vec<QwenLayerWeights>,
    final_norm_weight: RmsNormWeight,
    lm_head: DeviceBuffer<u16>,
    lm_head_fp4: LayerLinear,
}

/// Mutable single-sequence decode state.
pub struct DecodeState {
    /// Device-resident per-layer K/V cache.
    pub kv_cache: KvCache,
    /// Absolute token position for the next decode step.
    pub position: usize,
    /// Last token emitted by `decode_one`, if any.
    pub last_token: Option<u32>,
    current_token_device: DeviceBuffer<u32>,
    workspace: DecodeWorkspace,
}

struct QwenLayerWeights {
    q_proj: LayerLinear,
    k_proj: LayerLinear,
    v_proj: LayerLinear,
    qkv_proj: LayerLinear,
    o_proj: LayerLinear,
    ffn: QwenFfnWeights,
    input_norm_weight: RmsNormWeight,
    q_norm_weight: RmsNormWeight,
    k_norm_weight: RmsNormWeight,
    post_attn_norm_weight: RmsNormWeight,
}

enum QwenFfnWeights {
    Dense {
        gate_proj: LayerLinear,
        up_proj: LayerLinear,
        gate_up_proj: LayerLinear,
        down_proj: LayerLinear,
    },
    Moe {
        router: Bf16Linear,
        experts: Vec<LazyMoeExpertWeights>,
    },
}

struct Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

struct MoeExpertWeights {
    gate_up_proj: LayerLinear,
    down_proj: LayerLinear,
}

pub(crate) struct LazyMoeExpertWeights {
    checkpoint: ModelOptCheckpoint,
    prefix: String,
    weights: RefCell<Option<MoeExpertWeights>>,
}

struct LayerLinear {
    device: ModelOptCublasLtWeight,
}

struct RmsNormWeight {
    device: DeviceBuffer<f32>,
}

struct DecodeWorkspace {
    config: QwenModelConfig,
    stream: CudaStream,
    hidden: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    lm_index: DeviceBuffer<u32>,
    lm_value: DeviceBuffer<f32>,
    lm_head_fp4: LinearDecodeOp,
    position_device: DeviceBuffer<u32>,
    cache_len_device: DeviceBuffer<u32>,
    layers: Vec<LayerDecodeWorkspace>,
    graph: Option<CudaGraphExec>,
}

struct LayerDecodeWorkspace {
    config: QwenModelConfig,
    normed_hidden: DeviceBuffer<f32>,
    q: LinearDecodeOp,
    k: LinearDecodeOp,
    v: LinearDecodeOp,
    qkv: LinearDecodeOp,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    o: LinearDecodeOp,
    attn_residual: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    ffn: LayerFfnDecodeWorkspace,
}

enum LayerFfnDecodeWorkspace {
    Dense {
        gate_up: LinearDecodeOp,
        ffn_activated: DeviceBuffer<f32>,
        down: LinearDecodeOp,
    },
    Moe {
        experts: Vec<MoeExpertDecodeWorkspace>,
        expert_streams: Vec<CudaStream>,
        expert_ready_events: Vec<CudaEvent>,
        route_ready_event: CudaEvent,
        router_logits: DeviceBuffer<f32>,
        route: MoeRouteWorkspace,
        gate_up_input: Nvfp4Matrix,
        expert_ptrs: MoeExpertPointerTables,
        grouped_gate_up: Option<GroupedGemvWorkspace>,
        grouped_down: Option<MoeGroupedDownWorkspace>,
        ffn_out: DeviceBuffer<f32>,
    },
}

struct MoeExpertDecodeWorkspace {
    gate_up: Option<LinearDecodeOp>,
    down: Option<LinearDecodeOp>,
}

pub struct MoeRouteWorkspace {
    pub indices: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
}

pub struct MoeExpertPointerTables {
    pub gate_up_values: DeviceBuffer<*const u8>,
    pub gate_up_scales: DeviceBuffer<*const u8>,
    pub gate_up_grouped_values: DeviceBuffer<*const u8>,
    pub gate_up_grouped_scales: DeviceBuffer<*const u8>,
    #[allow(dead_code)]
    pub down_values: DeviceBuffer<*const u8>,
    #[allow(dead_code)]
    pub down_scales: DeviceBuffer<*const u8>,
    pub down_grouped_values: DeviceBuffer<*const u8>,
    pub down_grouped_scales: DeviceBuffer<*const u8>,
    pub down_input_scales: DeviceBuffer<f32>,
    pub down_alphas: DeviceBuffer<f32>,
    pub shared_gate_up_input_scale: Option<f32>,
    pub gate_up_alphas: DeviceBuffer<f32>,
}

pub struct MoeGroupedDownWorkspace {
    pub gemv: GroupedGemvWorkspace,
    pub inputs: Vec<Nvfp4Matrix>,
    pub input_simple_scales: Vec<DeviceBuffer<u8>>,
    pub input_values: DeviceBuffer<*const u8>,
    pub input_scales: DeviceBuffer<*const u8>,
    pub input_values_mut: DeviceBuffer<*mut u8>,
    pub input_scales_mut: DeviceBuffer<*mut u8>,
}

pub struct GroupedGemvWorkspace {
    pub plan: CutlassFp4GroupedGemvF32Plan,
    pub a_values: DeviceBuffer<*const u8>,
    pub a_scales: DeviceBuffer<*const u8>,
    pub b_values: DeviceBuffer<*const u8>,
    pub b_scales: DeviceBuffer<*const u8>,
    pub c: DeviceBuffer<*const f32>,
    pub d: DeviceBuffer<*mut f32>,
    pub outputs: Vec<F32Matrix>,
}

struct LinearDecodeOp {
    input: Nvfp4Matrix,
    c: F32Matrix,
    d: F32Matrix,
    plan: Fp4TnMatmulPlan,
    use_cutlass_gemv: bool,
}

impl Qwen3Model {
    /// Vocabulary size expected by this Qwen3 checkpoint.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab
    }

    /// Loads the full Qwen3-8B NVFP4 checkpoint and uploads decode weights.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = QwenModelConfig::load(model_dir)?;
        println!(
            "loading Qwen model: layers={} hidden={} q_heads={} kv_heads={} ffn={}",
            config.layers,
            config.hidden,
            config.q_heads,
            config.kv_heads,
            config.ffn_label()
        );
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let lt = CublasLt::new()?;
        let mut layers = Vec::with_capacity(config.layers);
        for layer_idx in 0..config.layers {
            layers.push(QwenLayerWeights::load(&checkpoint, config, layer_idx)?);
        }
        let embeddings = read_bf16_matrix_device(
            &checkpoint,
            "model.embed_tokens.weight",
            config.vocab,
            config.hidden,
        )?;
        let final_norm_weight = RmsNormWeight::load(&checkpoint, "model.norm.weight")?;
        let lm_head =
            read_bf16_matrix_device(&checkpoint, "lm_head.weight", config.vocab, config.hidden)?;
        let lm_head_fp4 = LayerLinear::load_bf16_fp4_lm_head(&checkpoint, config)?;
        println!("Qwen model loaded");
        Ok(Self {
            config,
            checkpoint,
            lt,
            embeddings,
            layers,
            final_norm_weight,
            lm_head,
            lm_head_fp4,
        })
    }

    /// Allocates a decode state capable of storing `max_tokens` positions.
    pub fn new_decode_state(&self, max_tokens: usize) -> Result<DecodeState> {
        DecodeState::new(self, max_tokens)
    }

    /// Decodes one token id, appends K/V for the current position, and returns GPU argmax.
    pub fn decode_one(&self, state: &mut DecodeState, token_id: u32) -> Result<NextToken> {
        self.validate_decode_token(state, token_id)?;
        state.current_token_device.copy_from_host(&[token_id])?;
        let ArgmaxResult { index, value } = state.workspace.replay_graph(
            &self.lt,
            &self.embeddings,
            &state.current_token_device,
            &self.layers,
            &self.final_norm_weight,
            &self.lm_head,
            &self.lm_head_fp4,
            &mut state.kv_cache,
            state.position,
        )?;
        state.position += 1;
        state.last_token = Some(index);
        Ok(NextToken {
            input_token: token_id,
            token: index,
            logit: value,
        })
    }

    /// Decodes one token with CUDA-event stage profiling.
    ///
    /// This intentionally bypasses CUDA graph replay so each stage can be
    /// timed independently. Use [`decode_one`](Self::decode_one) for normal
    /// throughput measurements.
    pub fn decode_one_profiled(
        &mut self,
        state: &mut DecodeState,
        token_id: u32,
        profile: &mut QwenDecodeProfile,
    ) -> Result<NextToken> {
        self.validate_decode_token(state, token_id)?;
        state.current_token_device.copy_from_host(&[token_id])?;
        let ArgmaxResult { index, value } = state.workspace.run_profiled(
            &self.lt,
            &self.embeddings,
            &state.current_token_device,
            &self.layers,
            &self.final_norm_weight,
            &self.lm_head,
            &self.lm_head_fp4,
            &mut state.kv_cache,
            state.position,
            profile,
        )?;
        state.position += 1;
        state.last_token = Some(index);
        profile.tokens += 1;
        Ok(NextToken {
            input_token: token_id,
            token: index,
            logit: value,
        })
    }

    /// Decodes one token id and returns the full lm-head logits on CPU.
    pub fn decode_one_logits(
        &self,
        state: &mut DecodeState,
        token_id: u32,
    ) -> Result<NextTokenLogits> {
        let (input_token, final_hidden_device) = self.decode_one_hidden(state, token_id)?;
        let logits = self.lm_head_logits(final_hidden_device)?;
        Ok(NextTokenLogits {
            input_token,
            logits,
        })
    }

    fn decode_one_hidden<'a>(
        &self,
        state: &'a mut DecodeState,
        token_id: u32,
    ) -> Result<(u32, &'a DeviceBuffer<f32>)> {
        self.validate_decode_token(state, token_id)?;
        state.current_token_device.copy_from_host(&[token_id])?;
        state.workspace.run(
            &self.lt,
            &self.embeddings,
            &state.current_token_device,
            &self.layers,
            &self.final_norm_weight,
            &mut state.kv_cache,
            state.position,
        )?;
        state.position += 1;
        Ok((token_id, &state.workspace.final_hidden))
    }

    fn validate_decode_token(&self, state: &DecodeState, token_id: u32) -> Result<()> {
        if token_id as usize >= self.config.vocab {
            return Err(Error::Shape {
                label: "input token id",
                expected: format!("token < {}", self.config.vocab),
                actual: token_id.to_string(),
            });
        }
        if state.position >= state.kv_cache.layer(0)?.max_tokens() {
            return Err(Error::Shape {
                label: "decode position",
                expected: format!("position < {}", state.kv_cache.layer(0)?.max_tokens()),
                actual: state.position.to_string(),
            });
        }
        if state.position > u32::MAX as usize - 1 {
            return Err(Error::Shape {
                label: "decode position",
                expected: "position <= u32::MAX - 1".to_string(),
                actual: state.position.to_string(),
            });
        }
        Ok(())
    }

    /// Prefills the decode state from a contiguous prompt chunk.
    ///
    /// This runs sequence-shaped projection and MLP matmuls for the prompt
    /// chunk, appends all K/V rows into the cache, and returns the next-token
    /// argmax from the final prompt position.
    pub fn prefill(&self, state: &mut DecodeState, token_ids: &[u32]) -> Result<NextToken> {
        let (input_token, last_hidden) = self.prefill_hidden(state, token_ids)?;
        let ArgmaxResult { index, value } = self.lm_head_argmax(&last_hidden)?;
        state.last_token = Some(index);
        Ok(NextToken {
            input_token,
            token: index,
            logit: value,
        })
    }

    /// Prefills the decode state from a prompt chunk and returns full logits.
    pub fn prefill_logits(
        &self,
        state: &mut DecodeState,
        token_ids: &[u32],
    ) -> Result<NextTokenLogits> {
        let (input_token, last_hidden) = self.prefill_hidden(state, token_ids)?;
        let logits = self.lm_head_logits(&last_hidden)?;
        Ok(NextTokenLogits {
            input_token,
            logits,
        })
    }

    fn prefill_hidden(
        &self,
        state: &mut DecodeState,
        token_ids: &[u32],
    ) -> Result<(u32, DeviceBuffer<f32>)> {
        if token_ids.is_empty() {
            return Err(Error::Shape {
                label: "prefill token ids",
                expected: "at least one token".to_string(),
                actual: "0 tokens".to_string(),
            });
        }
        if state.position + token_ids.len() > state.kv_cache.layer(0)?.max_tokens() {
            return Err(Error::Shape {
                label: "prefill capacity",
                expected: format!(
                    "at most {} remaining positions",
                    state.kv_cache.layer(0)?.max_tokens() - state.position
                ),
                actual: format!("{} tokens", token_ids.len()),
            });
        }
        for &token_id in token_ids {
            if token_id as usize >= self.config.vocab {
                return Err(Error::Shape {
                    label: "prefill token id",
                    expected: format!("token < {}", self.config.vocab),
                    actual: token_id.to_string(),
                });
            }
        }

        let tokens = token_ids.len();
        let start_position = state.position;
        let mut hidden = read_bf16_rows_device(
            &self.checkpoint,
            "model.embed_tokens.weight",
            token_ids,
            self.config.hidden,
        )?;
        for (layer_idx, weights) in self.layers.iter().enumerate() {
            hidden = run_layer_prefill(
                self.config,
                &self.lt,
                layer_idx,
                weights,
                hidden,
                &mut state.kv_cache,
                start_position,
                tokens,
            )?;
        }

        inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        let stream = CudaStream::new_blocking()?;
        let mut final_hidden = DeviceBuffer::zeroed(tokens * self.config.hidden)?;
        rms_norm_f32_into_on_stream(
            tokens,
            self.config.hidden,
            &hidden,
            &self.final_norm_weight.device,
            final_hidden.output(),
            self.config.rms_eps,
            &stream,
        )?;
        let mut last_hidden = DeviceBuffer::zeroed(self.config.hidden)?;
        copy_row_f32_into_on_stream(
            tokens,
            self.config.hidden,
            tokens - 1,
            &final_hidden,
            last_hidden.output(),
            &stream,
        )?;
        stream.synchronize()?;

        state.position += tokens;
        Ok((token_ids[tokens - 1], last_hidden))
    }

    fn lm_head_argmax(&self, hidden: &DeviceBuffer<f32>) -> Result<ArgmaxResult> {
        inc_counter(&RUNTIME_COUNTERS.lm_head_argmax_calls);
        let result =
            bf16_linear_argmax_f32(hidden, &self.lm_head, self.config.vocab, self.config.hidden)?;
        synchronize_device()?;
        Ok(result)
    }

    fn lm_head_logits(&self, hidden: &DeviceBuffer<f32>) -> Result<Vec<f32>> {
        inc_counter(&RUNTIME_COUNTERS.lm_head_logits_calls);
        add_counter(
            &RUNTIME_COUNTERS.host_logits_bytes,
            (self.config.vocab * std::mem::size_of::<f32>()) as u64,
        );
        let stream = CudaStream::new_non_blocking()?;
        let mut logits_device = DeviceBuffer::<f32>::zeroed(self.config.vocab)?;
        bf16_linear_logits_f32_into_on_stream(
            hidden,
            &self.lm_head,
            logits_device.output(),
            self.config.vocab,
            self.config.hidden,
            &stream,
        )?;
        Ok(logits_device.copy_to_host(&stream)?.into_vec())
    }
}

impl DecodeState {
    /// Allocates an empty single-sequence decode state.
    fn new(model: &Qwen3Model, max_tokens: usize) -> Result<Self> {
        Ok(Self {
            kv_cache: KvCache::new(
                model.config.layers,
                max_tokens,
                model.config.kv_heads,
                model.config.head_dim,
            )?,
            position: 0,
            last_token: None,
            current_token_device: DeviceBuffer::from_host(&[0])?,
            workspace: DecodeWorkspace::new(
                model.config,
                &model.lt,
                &model.layers,
                &model.lm_head_fp4,
            )?,
        })
    }
}

impl QwenLayerWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: QwenModelConfig,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer}");
        let q_host = checkpoint.load_nvfp4_linear(&format!("{prefix}.self_attn.q_proj"))?;
        let k_host = checkpoint.load_nvfp4_linear(&format!("{prefix}.self_attn.k_proj"))?;
        let v_host = checkpoint.load_nvfp4_linear(&format!("{prefix}.self_attn.v_proj"))?;
        let qk_host = ModelOptNvfp4Linear::concat_out_features(
            format!("{prefix}.self_attn.qk_proj"),
            &q_host,
            &k_host,
        )?;
        let qkv_host = ModelOptNvfp4Linear::concat_out_features(
            format!("{prefix}.self_attn.qkv_proj"),
            &qk_host,
            &v_host,
        )?;
        Ok(Self {
            q_proj: LayerLinear::from_host(&q_host)?,
            k_proj: LayerLinear::from_host(&k_host)?,
            v_proj: LayerLinear::from_host(&v_host)?,
            qkv_proj: LayerLinear::from_host(&qkv_host)?,
            o_proj: LayerLinear::load(checkpoint, &format!("{prefix}.self_attn.o_proj"))?,
            ffn: QwenFfnWeights::load(checkpoint, config, &prefix)?,
            input_norm_weight: RmsNormWeight::load(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
            )?,
            q_norm_weight: RmsNormWeight::load(
                checkpoint,
                &format!("{prefix}.self_attn.q_norm.weight"),
            )?,
            k_norm_weight: RmsNormWeight::load(
                checkpoint,
                &format!("{prefix}.self_attn.k_norm.weight"),
            )?,
            post_attn_norm_weight: RmsNormWeight::load(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
        })
    }
}

impl QwenFfnWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: QwenModelConfig,
        prefix: &str,
    ) -> Result<Self> {
        match config.ffn {
            QwenFfnConfig::Dense => {
                let gate_host = checkpoint.load_nvfp4_linear(&format!("{prefix}.mlp.gate_proj"))?;
                let up_host = checkpoint.load_nvfp4_linear(&format!("{prefix}.mlp.up_proj"))?;
                let gate_up_host = ModelOptNvfp4Linear::concat_out_features(
                    format!("{prefix}.mlp.gate_up_proj"),
                    &gate_host,
                    &up_host,
                )?;
                Ok(Self::Dense {
                    gate_proj: LayerLinear::from_host(&gate_host)?,
                    up_proj: LayerLinear::from_host(&up_host)?,
                    gate_up_proj: LayerLinear::from_host(&gate_up_host)?,
                    down_proj: LayerLinear::load(checkpoint, &format!("{prefix}.mlp.down_proj"))?,
                })
            }
            QwenFfnConfig::Moe { experts, .. } => {
                let router = Bf16Linear::load(
                    checkpoint,
                    &format!("{prefix}.mlp.gate.weight"),
                    experts,
                    config.hidden,
                )?;
                let mut expert_weights = Vec::with_capacity(experts);
                for expert_idx in 0..experts {
                    let expert_prefix = format!("{prefix}.mlp.experts.{expert_idx}");
                    expert_weights
                        .push(LazyMoeExpertWeights::new(checkpoint.clone(), expert_prefix));
                }
                Ok(Self::Moe {
                    router,
                    experts: expert_weights,
                })
            }
        }
    }
}

impl LayerLinear {
    fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let host: ModelOptNvfp4Linear = checkpoint.load_nvfp4_linear(prefix)?;
        Self::from_host(&host)
    }

    fn load_bf16_fp4_lm_head(
        checkpoint: &ModelOptCheckpoint,
        config: QwenModelConfig,
    ) -> Result<Self> {
        let values =
            read_bf16_matrix_f32(checkpoint, "lm_head.weight", config.vocab, config.hidden)?;
        let matrix = Nvfp4Matrix::quantize_col_major_f32(config.hidden, config.vocab, &values)?;
        Ok(Self {
            device: ModelOptCublasLtWeight::from_matrix(matrix, 1.0, 1.0)?,
        })
    }

    fn from_host(host: &ModelOptNvfp4Linear) -> Result<Self> {
        Ok(Self {
            device: host.as_cublaslt_weight()?,
        })
    }
}

impl Bf16Linear {
    fn load(checkpoint: &ModelOptCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weight: read_bf16_matrix_device(checkpoint, name, rows, cols)?,
            rows,
            cols,
        })
    }

    fn run_logits_into_on_stream(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
    }
}

impl LazyMoeExpertWeights {
    fn new(checkpoint: ModelOptCheckpoint, prefix: String) -> Self {
        Self {
            checkpoint,
            prefix,
            weights: RefCell::new(None),
        }
    }

    fn get(&self) -> Result<Ref<'_, MoeExpertWeights>> {
        if self.weights.borrow().is_none() {
            let gate_host = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.gate_proj", self.prefix))?;
            let up_host = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.up_proj", self.prefix))?;
            let gate_up_host = ModelOptNvfp4Linear::concat_out_features(
                format!("{}.gate_up_proj", self.prefix),
                &gate_host,
                &up_host,
            )?;
            let weights = MoeExpertWeights {
                gate_up_proj: LayerLinear::from_host(&gate_up_host)?,
                down_proj: LayerLinear::load(
                    &self.checkpoint,
                    &format!("{}.down_proj", self.prefix),
                )?,
            };
            *self.weights.borrow_mut() = Some(weights);
        }
        Ok(Ref::map(self.weights.borrow(), |weights| {
            weights.as_ref().expect("MoE expert loaded")
        }))
    }
}

impl RmsNormWeight {
    fn load(checkpoint: &ModelOptCheckpoint, name: &str) -> Result<Self> {
        let host = read_bf16_vector(checkpoint, name)?;
        Ok(Self {
            device: DeviceBuffer::from_host(&host)?,
        })
    }
}

impl DecodeWorkspace {
    fn new(
        config: QwenModelConfig,
        lt: &CublasLt,
        layers: &[QwenLayerWeights],
        lm_head_fp4: &LayerLinear,
    ) -> Result<Self> {
        let mut layer_workspaces = Vec::with_capacity(layers.len());
        for weights in layers {
            layer_workspaces.push(LayerDecodeWorkspace::new(config, lt, weights)?);
        }
        Ok(Self {
            config,
            stream: CudaStream::new_non_blocking()?,
            hidden: DeviceBuffer::zeroed(config.hidden)?,
            final_hidden: DeviceBuffer::zeroed(config.hidden)?,
            lm_index: DeviceBuffer::zeroed(1)?,
            lm_value: DeviceBuffer::zeroed(1)?,
            lm_head_fp4: LinearDecodeOp::new(lt, lm_head_fp4, config.vocab, config.hidden)?,
            position_device: DeviceBuffer::from_host(&[0])?,
            cache_len_device: DeviceBuffer::from_host(&[1])?,
            layers: layer_workspaces,
            graph: None,
        })
    }

    fn run(
        &mut self,
        lt: &CublasLt,
        embeddings: &DeviceBuffer<u16>,
        token_id: &DeviceBuffer<u32>,
        layers: &[QwenLayerWeights],
        final_norm_weight: &RmsNormWeight,
        kv_cache: &mut KvCache,
        position: usize,
    ) -> Result<()> {
        self.prepare_decode_scalars(position)?;
        self.enqueue_transformer(
            lt,
            embeddings,
            token_id,
            layers,
            final_norm_weight,
            kv_cache,
            position,
            true,
        )?;
        self.stream.synchronize()?;
        kv_cache.advance_all(1)
    }

    fn run_profiled(
        &mut self,
        lt: &CublasLt,
        embeddings: &DeviceBuffer<u16>,
        token_id: &DeviceBuffer<u32>,
        layers: &[QwenLayerWeights],
        final_norm_weight: &RmsNormWeight,
        _lm_head: &DeviceBuffer<u16>,
        lm_head_fp4: &LayerLinear,
        kv_cache: &mut KvCache,
        position: usize,
        profile: &mut QwenDecodeProfile,
    ) -> Result<ArgmaxResult> {
        self.prepare_decode_scalars(position)?;

        profile.embedding_ms += time_stage(&self.stream, || {
            copy_bf16_row_to_f32_indexed_into_on_stream(
                self.config.vocab,
                self.config.hidden,
                embeddings,
                token_id,
                self.hidden.output(),
                &self.stream,
            )
        })?;

        for (layer_idx, (weights, workspace)) in
            layers.iter().zip(self.layers.iter_mut()).enumerate()
        {
            workspace.run_profiled(
                lt,
                layer_idx,
                weights,
                &mut self.hidden,
                kv_cache,
                position,
                &self.position_device,
                &self.cache_len_device,
                &self.stream,
                profile,
            )?;
        }

        inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        profile.final_norm_ms += time_stage(&self.stream, || {
            rms_norm_f32_into_on_stream(
                1,
                self.config.hidden,
                &self.hidden,
                &final_norm_weight.device,
                self.final_hidden.output(),
                self.config.rms_eps,
                &self.stream,
            )
        })?;

        inc_counter(&RUNTIME_COUNTERS.lm_head_argmax_calls);
        let start = CudaEvent::new()?;
        let end = CudaEvent::new()?;
        start.record_on_stream(&self.stream)?;
        self.enqueue_lm_head_argmax(lt, lm_head_fp4, false)?;
        end.record_on_stream(&self.stream)?;
        start.synchronize()?;
        end.synchronize()?;
        profile.lm_head_argmax_ms += start.elapsed_ms_until(&end)? as f64;

        kv_cache.advance_all(1)?;
        let index = self.lm_index.copy_to_host(&self.stream)?[0];
        let value = self.lm_value.copy_to_host(&self.stream)?[0];
        Ok(ArgmaxResult { index, value })
    }

    fn replay_graph(
        &mut self,
        lt: &CublasLt,
        embeddings: &DeviceBuffer<u16>,
        token_id: &DeviceBuffer<u32>,
        layers: &[QwenLayerWeights],
        final_norm_weight: &RmsNormWeight,
        _lm_head: &DeviceBuffer<u16>,
        lm_head_fp4: &LayerLinear,
        kv_cache: &mut KvCache,
        position: usize,
    ) -> Result<ArgmaxResult> {
        self.prepare_decode_scalars(position)?;
        if self.graph.is_none() {
            self.capture_graph(
                lt,
                embeddings,
                token_id,
                layers,
                final_norm_weight,
                _lm_head,
                lm_head_fp4,
                kv_cache,
                position,
            )?;
        }

        self.graph
            .as_ref()
            .expect("decode graph captured")
            .launch(&self.stream)?;
        kv_cache.advance_all(1)?;
        record_one_token_decode_counters(self.config);
        let index = self.lm_index.copy_to_host(&self.stream)?[0];
        let value = self.lm_value.copy_to_host(&self.stream)?[0];
        Ok(ArgmaxResult { index, value })
    }

    fn capture_graph(
        &mut self,
        lt: &CublasLt,
        embeddings: &DeviceBuffer<u16>,
        token_id: &DeviceBuffer<u32>,
        layers: &[QwenLayerWeights],
        final_norm_weight: &RmsNormWeight,
        _lm_head: &DeviceBuffer<u16>,
        lm_head_fp4: &LayerLinear,
        kv_cache: &mut KvCache,
        position: usize,
    ) -> Result<()> {
        self.stream.begin_capture()?;
        let enqueue_result = (|| {
            self.enqueue_transformer(
                lt,
                embeddings,
                token_id,
                layers,
                final_norm_weight,
                kv_cache,
                position,
                false,
            )?;
            self.enqueue_lm_head_argmax(lt, lm_head_fp4, false)
        })();
        let graph_result = self.stream.end_capture();
        enqueue_result?;
        self.graph = Some(graph_result?);
        Ok(())
    }

    fn prepare_decode_scalars(&mut self, position: usize) -> Result<()> {
        self.position_device.copy_from_host(&[position as u32])?;
        self.cache_len_device
            .copy_from_host(&[(position + 1) as u32])
    }

    fn enqueue_transformer(
        &mut self,
        lt: &CublasLt,
        embeddings: &DeviceBuffer<u16>,
        token_id: &DeviceBuffer<u32>,
        layers: &[QwenLayerWeights],
        final_norm_weight: &RmsNormWeight,
        kv_cache: &mut KvCache,
        position: usize,
        count: bool,
    ) -> Result<()> {
        copy_bf16_row_to_f32_indexed_into_on_stream(
            self.config.vocab,
            self.config.hidden,
            embeddings,
            token_id,
            self.hidden.output(),
            &self.stream,
        )?;
        for (layer_idx, (weights, workspace)) in
            layers.iter().zip(self.layers.iter_mut()).enumerate()
        {
            workspace.run(
                lt,
                layer_idx,
                weights,
                &mut self.hidden,
                kv_cache,
                position,
                &self.position_device,
                &self.cache_len_device,
                &self.stream,
                count,
            )?;
        }

        if count {
            inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        }
        rms_norm_f32_into_on_stream(
            1,
            self.config.hidden,
            &self.hidden,
            &final_norm_weight.device,
            self.final_hidden.output(),
            self.config.rms_eps,
            &self.stream,
        )
    }

    fn enqueue_lm_head_argmax(
        &mut self,
        lt: &CublasLt,
        lm_head_fp4: &LayerLinear,
        count: bool,
    ) -> Result<()> {
        if count {
            inc_counter(&RUNTIME_COUNTERS.lm_head_argmax_calls);
        }
        self.lm_head_fp4.run(
            lt,
            lm_head_fp4,
            self.config.hidden,
            &self.final_hidden,
            &self.stream,
            false,
        )?;
        argmax_f32_into_on_stream(
            self.lm_head_fp4.output(),
            self.lm_index.output(),
            self.lm_value.output(),
            &self.stream,
        )
    }
}

impl LayerDecodeWorkspace {
    fn new(config: QwenModelConfig, lt: &CublasLt, weights: &QwenLayerWeights) -> Result<Self> {
        Ok(Self {
            config,
            normed_hidden: DeviceBuffer::zeroed(config.hidden)?,
            q: LinearDecodeOp::new(lt, &weights.q_proj, config.q_width, config.hidden)?,
            k: LinearDecodeOp::new(lt, &weights.k_proj, config.kv_width, config.hidden)?,
            v: LinearDecodeOp::new(lt, &weights.v_proj, config.kv_width, config.hidden)?,
            qkv: LinearDecodeOp::new(
                lt,
                &weights.qkv_proj,
                config.q_width + config.kv_width + config.kv_width,
                config.hidden,
            )?,
            q_rope: DeviceBuffer::zeroed(config.q_width)?,
            k_rope: DeviceBuffer::zeroed(config.kv_width)?,
            attn: DeviceBuffer::zeroed(config.q_width)?,
            o: LinearDecodeOp::new(lt, &weights.o_proj, config.hidden, config.q_width)?,
            attn_residual: DeviceBuffer::zeroed(config.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(config.hidden)?,
            ffn: LayerFfnDecodeWorkspace::new(config, lt, &weights.ffn)?,
        })
    }

    fn run(
        &mut self,
        lt: &CublasLt,
        layer_idx: usize,
        weights: &QwenLayerWeights,
        hidden: &mut DeviceBuffer<f32>,
        kv_cache: &mut KvCache,
        position: usize,
        position_device: &DeviceBuffer<u32>,
        cache_len_device: &DeviceBuffer<u32>,
        stream: &CudaStream,
        count: bool,
    ) -> Result<()> {
        if count {
            inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        }
        rms_norm_f32_into_on_stream(
            1,
            self.config.hidden,
            hidden,
            &weights.input_norm_weight.device,
            self.normed_hidden.output(),
            self.config.rms_eps,
            stream,
        )?;

        if count {
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        }
        self.qkv.run(
            lt,
            &weights.qkv_proj,
            self.config.hidden,
            &self.normed_hidden,
            stream,
            count,
        )?;
        split_qkv_f32_into_on_stream(
            self.qkv.output(),
            self.q.output_mut().output(),
            self.k.output_mut().output(),
            self.v.output_mut().output(),
            stream,
        )?;

        if count {
            add_counter(&RUNTIME_COUNTERS.rms_norm_calls, 2);
            add_counter(&RUNTIME_COUNTERS.rope_calls, 2);
        }
        rms_norm_rope_neox_f32_indexed_into_on_stream(
            self.config.q_heads,
            self.config.head_dim,
            self.q.output(),
            &weights.q_norm_weight.device,
            self.q_rope.output(),
            position_device,
            self.config.rope_theta,
            self.config.rms_eps,
            stream,
        )?;
        rms_norm_rope_neox_f32_indexed_into_on_stream(
            self.config.kv_heads,
            self.config.head_dim,
            self.k.output(),
            &weights.k_norm_weight.device,
            self.k_rope.output(),
            position_device,
            self.config.rope_theta,
            self.config.rms_eps,
            stream,
        )?;

        kv_cache.layer_mut(layer_idx)?.append_indexed_on_stream(
            &self.k_rope,
            self.v.output(),
            position_device,
            position,
            stream,
        )?;
        if count {
            inc_counter(&RUNTIME_COUNTERS.attention_calls);
        }
        kv_cache
            .layer(layer_idx)?
            .decode_attention_indexed_into_on_stream(
                &self.q_rope,
                self.attn.output(),
                cache_len_device,
                position + 1,
                self.config.q_heads,
                stream,
            )?;

        if count {
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        }
        self.o.run(
            lt,
            &weights.o_proj,
            self.config.q_width,
            &self.attn,
            stream,
            count,
        )?;
        if count {
            inc_counter(&RUNTIME_COUNTERS.add_calls);
        }
        add_f32_into_on_stream(hidden, self.o.output(), self.attn_residual.output(), stream)?;

        if count {
            inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        }
        rms_norm_f32_into_on_stream(
            1,
            self.config.hidden,
            &self.attn_residual,
            &weights.post_attn_norm_weight.device,
            self.ffn_norm.output(),
            self.config.rms_eps,
            stream,
        )?;

        self.ffn.run(
            self.config,
            lt,
            &weights.ffn,
            &self.ffn_norm,
            &self.attn_residual,
            hidden,
            stream,
            count,
            None,
        )
    }

    fn run_profiled(
        &mut self,
        lt: &CublasLt,
        layer_idx: usize,
        weights: &QwenLayerWeights,
        hidden: &mut DeviceBuffer<f32>,
        kv_cache: &mut KvCache,
        position: usize,
        position_device: &DeviceBuffer<u32>,
        cache_len_device: &DeviceBuffer<u32>,
        stream: &CudaStream,
        profile: &mut QwenDecodeProfile,
    ) -> Result<()> {
        inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        profile.input_norm_ms += time_stage(stream, || {
            rms_norm_f32_into_on_stream(
                1,
                self.config.hidden,
                hidden,
                &weights.input_norm_weight.device,
                self.normed_hidden.output(),
                self.config.rms_eps,
                stream,
            )
        })?;

        inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        profile.qkv_quantize_ms += time_stage(stream, || {
            self.qkv.quantize(
                &weights.qkv_proj,
                self.config.hidden,
                &self.normed_hidden,
                stream,
            )
        })?;

        profile.qkv_gemm_ms += time_stage(stream, || {
            self.qkv
                .run_quantized(lt, &weights.qkv_proj, self.config.hidden, stream, true)?;
            split_qkv_f32_into_on_stream(
                self.qkv.output(),
                self.q.output_mut().output(),
                self.k.output_mut().output(),
                self.v.output_mut().output(),
                stream,
            )
        })?;

        add_counter(&RUNTIME_COUNTERS.rms_norm_calls, 2);
        add_counter(&RUNTIME_COUNTERS.rope_calls, 2);
        profile.qk_norm_ms += time_stage(stream, || {
            rms_norm_rope_neox_f32_indexed_into_on_stream(
                self.config.q_heads,
                self.config.head_dim,
                self.q.output(),
                &weights.q_norm_weight.device,
                self.q_rope.output(),
                position_device,
                self.config.rope_theta,
                self.config.rms_eps,
                stream,
            )?;
            rms_norm_rope_neox_f32_indexed_into_on_stream(
                self.config.kv_heads,
                self.config.head_dim,
                self.k.output(),
                &weights.k_norm_weight.device,
                self.k_rope.output(),
                position_device,
                self.config.rope_theta,
                self.config.rms_eps,
                stream,
            )
        })?;

        profile.kv_append_ms += time_stage(stream, || {
            kv_cache.layer_mut(layer_idx)?.append_indexed_on_stream(
                &self.k_rope,
                self.v.output(),
                position_device,
                position,
                stream,
            )
        })?;

        inc_counter(&RUNTIME_COUNTERS.attention_calls);
        profile.attention_ms += time_stage(stream, || {
            kv_cache
                .layer(layer_idx)?
                .decode_attention_indexed_into_on_stream(
                    &self.q_rope,
                    self.attn.output(),
                    cache_len_device,
                    position + 1,
                    self.config.q_heads,
                    stream,
                )
        })?;

        inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        profile.o_quantize_ms += time_stage(stream, || {
            self.o
                .quantize(&weights.o_proj, self.config.q_width, &self.attn, stream)
        })?;

        profile.o_gemm_ms += time_stage(stream, || {
            self.o
                .run_quantized(lt, &weights.o_proj, self.config.q_width, stream, true)
        })?;

        inc_counter(&RUNTIME_COUNTERS.add_calls);
        profile.attn_residual_ms += time_stage(stream, || {
            add_f32_into_on_stream(hidden, self.o.output(), self.attn_residual.output(), stream)
        })?;

        inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
        profile.ffn_norm_ms += time_stage(stream, || {
            rms_norm_f32_into_on_stream(
                1,
                self.config.hidden,
                &self.attn_residual,
                &weights.post_attn_norm_weight.device,
                self.ffn_norm.output(),
                self.config.rms_eps,
                stream,
            )
        })?;

        self.ffn.run_profiled(
            self.config,
            lt,
            &weights.ffn,
            &self.ffn_norm,
            &self.attn_residual,
            hidden,
            stream,
            profile,
        )?;

        Ok(())
    }
}

impl LayerFfnDecodeWorkspace {
    fn new(config: QwenModelConfig, lt: &CublasLt, weights: &QwenFfnWeights) -> Result<Self> {
        match (config.ffn, weights) {
            (
                QwenFfnConfig::Dense,
                QwenFfnWeights::Dense {
                    gate_up_proj,
                    down_proj,
                    ..
                },
            ) => Ok(Self::Dense {
                gate_up: LinearDecodeOp::new(
                    lt,
                    gate_up_proj,
                    config.intermediate * 2,
                    config.hidden,
                )?,
                ffn_activated: DeviceBuffer::zeroed(config.intermediate)?,
                down: LinearDecodeOp::new(lt, down_proj, config.hidden, config.intermediate)?,
            }),
            (
                QwenFfnConfig::Moe {
                    experts,
                    expert_intermediate,
                    experts_per_token,
                    ..
                },
                QwenFfnWeights::Moe {
                    experts: expert_weights,
                    ..
                },
            ) => {
                if expert_weights.len() != experts {
                    return Err(Error::Shape {
                        label: "MoE experts",
                        expected: format!("{experts} experts"),
                        actual: format!("{} experts", expert_weights.len()),
                    });
                }
                let mut expert_workspaces = Vec::with_capacity(experts);
                for _ in expert_weights {
                    expert_workspaces.push(MoeExpertDecodeWorkspace::new());
                }
                let mut expert_streams = Vec::with_capacity(experts_per_token);
                let mut expert_ready_events = Vec::with_capacity(experts_per_token);
                for _ in 0..experts_per_token {
                    expert_streams.push(CudaStream::new_non_blocking()?);
                    expert_ready_events.push(CudaEvent::new()?);
                }
                let expert_ptrs = MoeExpertPointerTables::new(expert_weights)?;
                Ok(Self::Moe {
                    experts: expert_workspaces,
                    expert_streams,
                    expert_ready_events,
                    route_ready_event: CudaEvent::new()?,
                    router_logits: DeviceBuffer::zeroed(experts)?,
                    route: MoeRouteWorkspace::new(experts_per_token)?,
                    gate_up_input: Nvfp4Matrix::zeroed_col_major(config.hidden, 1)?,
                    expert_ptrs,
                    grouped_gate_up: GroupedGemvWorkspace::new(
                        expert_intermediate * 2,
                        config.hidden,
                        experts_per_token,
                    )?,
                    grouped_down: MoeGroupedDownWorkspace::new(
                        config.hidden,
                        expert_intermediate,
                        experts_per_token,
                    )?,
                    ffn_out: DeviceBuffer::zeroed(config.hidden)?,
                })
            }
            _ => Err(Error::Format {
                label: "Qwen FFN",
                detail: "config and weight FFN variants do not match".to_string(),
            }),
        }
    }

    fn run(
        &mut self,
        config: QwenModelConfig,
        lt: &CublasLt,
        weights: &QwenFfnWeights,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
        count: bool,
        profile: Option<&mut QwenDecodeProfile>,
    ) -> Result<()> {
        match (&mut *self, config.ffn, weights) {
            (
                Self::Dense {
                    gate_up,
                    ffn_activated,
                    down,
                },
                QwenFfnConfig::Dense,
                QwenFfnWeights::Dense {
                    gate_up_proj,
                    down_proj,
                    ..
                },
            ) => run_dense_decode_ffn(
                config,
                lt,
                gate_up,
                ffn_activated,
                down,
                gate_up_proj,
                down_proj,
                ffn_norm,
                residual,
                output,
                stream,
                count,
            ),
            (
                Self::Moe {
                    experts,
                    expert_streams,
                    expert_ready_events,
                    route_ready_event,
                    router_logits,
                    route,
                    gate_up_input,
                    expert_ptrs,
                    grouped_gate_up,
                    grouped_down,
                    ffn_out,
                },
                QwenFfnConfig::Moe {
                    experts_per_token,
                    expert_intermediate,
                    norm_topk_prob,
                    ..
                },
                QwenFfnWeights::Moe {
                    router,
                    experts: expert_weights,
                },
            ) => run_moe_decode_ffn(
                config,
                lt,
                router,
                experts,
                expert_streams,
                expert_ready_events,
                route_ready_event,
                router_logits,
                route,
                gate_up_input,
                expert_ptrs,
                grouped_gate_up.as_mut(),
                grouped_down.as_mut(),
                expert_weights,
                expert_intermediate,
                experts_per_token,
                norm_topk_prob,
                ffn_norm,
                residual,
                ffn_out,
                output,
                stream,
                count,
                profile,
            ),
            _ => Err(Error::Format {
                label: "Qwen FFN",
                detail: "config, workspace, and weight FFN variants do not match".to_string(),
            }),
        }
    }

    fn run_profiled(
        &mut self,
        config: QwenModelConfig,
        lt: &CublasLt,
        weights: &QwenFfnWeights,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
        profile: &mut QwenDecodeProfile,
    ) -> Result<()> {
        match (self, config.ffn, weights) {
            (
                Self::Dense {
                    gate_up,
                    ffn_activated,
                    down,
                },
                QwenFfnConfig::Dense,
                QwenFfnWeights::Dense {
                    gate_up_proj,
                    down_proj,
                    ..
                },
            ) => {
                inc_counter(&RUNTIME_COUNTERS.quantize_calls);
                profile.ffn_quantize_ms += time_stage(stream, || {
                    gate_up.quantize(gate_up_proj, config.hidden, ffn_norm, stream)
                })?;

                let gate_up_ms = time_stage(stream, || {
                    gate_up.run_quantized(lt, gate_up_proj, config.hidden, stream, true)
                })?;
                profile.ffn_gemm_ms += gate_up_ms;
                profile.ffn_gate_up_gemm_ms += gate_up_ms;

                inc_counter(&RUNTIME_COUNTERS.silu_calls);
                profile.silu_ms += time_stage(stream, || {
                    silu_mul_halves_f32_into_on_stream(
                        gate_up.output(),
                        ffn_activated.output(),
                        config.intermediate,
                        stream,
                    )
                })?;

                inc_counter(&RUNTIME_COUNTERS.quantize_calls);
                profile.ffn_quantize_ms += time_stage(stream, || {
                    down.quantize(down_proj, config.intermediate, ffn_activated, stream)
                })?;

                let down_ms = time_stage(stream, || {
                    down.run_quantized(lt, down_proj, config.intermediate, stream, true)
                })?;
                profile.ffn_gemm_ms += down_ms;
                profile.ffn_down_gemm_ms += down_ms;

                inc_counter(&RUNTIME_COUNTERS.add_calls);
                profile.ffn_residual_ms += time_stage(stream, || {
                    add_f32_into_on_stream(residual, down.output(), output.output(), stream)
                })?;
                Ok(())
            }
            (workspace, _, _) => {
                let start = CudaEvent::new()?;
                let end = CudaEvent::new()?;
                let wall_start = Instant::now();
                start.record_on_stream(stream)?;
                workspace.run(
                    config,
                    lt,
                    weights,
                    ffn_norm,
                    residual,
                    output,
                    stream,
                    true,
                    Some(profile),
                )?;
                end.record_on_stream(stream)?;
                start.synchronize()?;
                end.synchronize()?;
                profile.ffn_gemm_ms += start.elapsed_ms_until(&end)? as f64;
                profile.ffn_wall_ms += wall_start.elapsed().as_secs_f64() * 1_000.0;
                Ok(())
            }
        }
    }
}

impl MoeExpertDecodeWorkspace {
    fn new() -> Self {
        Self {
            gate_up: None,
            down: None,
        }
    }

    fn ensure(
        &mut self,
        config: QwenModelConfig,
        lt: &CublasLt,
        weights: &MoeExpertWeights,
        expert_intermediate: usize,
    ) -> Result<()> {
        if self.gate_up.is_none() {
            self.gate_up = Some(LinearDecodeOp::new_with_cutlass(
                lt,
                &weights.gate_up_proj,
                expert_intermediate * 2,
                config.hidden,
                true,
            )?);
        }
        if self.down.is_none() {
            self.down = Some(LinearDecodeOp::new_with_cutlass(
                lt,
                &weights.down_proj,
                config.hidden,
                expert_intermediate,
                false,
            )?);
        }
        Ok(())
    }

    fn down_mut(&mut self) -> &mut LinearDecodeOp {
        self.down.as_mut().expect("MoE down plan initialized")
    }
}

impl MoeRouteWorkspace {
    pub fn new(experts_per_token: usize) -> Result<Self> {
        Ok(Self {
            indices: DeviceBuffer::zeroed(experts_per_token)?,
            weights: DeviceBuffer::zeroed(experts_per_token)?,
        })
    }

    pub fn run_topk(
        &mut self,
        router_logits: &DeviceBuffer<f32>,
        norm_topk_prob: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        let k = self.indices.len();
        moe_topk_f32_into_on_stream(
            router_logits,
            self.indices.output(),
            self.weights.output(),
            k,
            norm_topk_prob,
            stream,
        )
    }

    fn copy_to_host(&self, stream: &CudaStream) -> Result<Vec<(usize, f32)>> {
        let indices = self.indices.copy_to_host(stream)?;
        let weights = self.weights.copy_to_host(stream)?;
        Ok(indices
            .iter()
            .zip(weights.iter())
            .map(|(&idx, &weight)| (idx as usize, weight))
            .collect())
    }
}

impl MoeExpertPointerTables {
    pub(crate) fn new(expert_weights: &[LazyMoeExpertWeights]) -> Result<Self> {
        let mut gate_up_values = Vec::with_capacity(expert_weights.len());
        let mut gate_up_scales = Vec::with_capacity(expert_weights.len());
        let mut down_values = Vec::with_capacity(expert_weights.len());
        let mut down_scales = Vec::with_capacity(expert_weights.len());
        let mut down_input_scales = Vec::with_capacity(expert_weights.len());
        let mut down_alphas = Vec::with_capacity(expert_weights.len());
        let mut shared_gate_up_input_scale = None;
        let mut gate_up_alphas = Vec::with_capacity(expert_weights.len());

        for expert in expert_weights {
            let expert = expert.get()?;
            let gate_up = &expert.gate_up_proj.device;
            let gate_up_matrix = gate_up.matrix();
            gate_up_values.push(gate_up_matrix.values_ptr());
            gate_up_scales.push(gate_up_matrix.scales_ptr());
            match shared_gate_up_input_scale {
                None => shared_gate_up_input_scale = Some(gate_up.input_scale()),
                Some(first) if first == gate_up.input_scale() => {}
                Some(_) => shared_gate_up_input_scale = Some(f32::NAN),
            }
            gate_up_alphas.push(gate_up.matmul_alpha());

            let down = &expert.down_proj.device;
            let down_matrix = down.matrix();
            down_values.push(down_matrix.values_ptr());
            down_scales.push(down_matrix.scales_ptr());
            down_input_scales.push(down.input_scale());
            down_alphas.push(down.matmul_alpha());
        }

        Ok(Self {
            gate_up_values: DeviceBuffer::from_host(&gate_up_values)?,
            gate_up_scales: DeviceBuffer::from_host(&gate_up_scales)?,
            gate_up_grouped_values: DeviceBuffer::from_host(&vec![
                std::ptr::null();
                expert_weights.len()
            ])?,
            gate_up_grouped_scales: DeviceBuffer::from_host(&vec![
                std::ptr::null();
                expert_weights.len()
            ])?,
            down_values: DeviceBuffer::from_host(&down_values)?,
            down_scales: DeviceBuffer::from_host(&down_scales)?,
            down_grouped_values: DeviceBuffer::from_host(&vec![
                std::ptr::null();
                expert_weights.len()
            ])?,
            down_grouped_scales: DeviceBuffer::from_host(&vec![
                std::ptr::null();
                expert_weights.len()
            ])?,
            down_input_scales: DeviceBuffer::from_host(&down_input_scales)?,
            down_alphas: DeviceBuffer::from_host(&down_alphas)?,
            shared_gate_up_input_scale: shared_gate_up_input_scale.filter(|value| !value.is_nan()),
            gate_up_alphas: DeviceBuffer::from_host(&gate_up_alphas)?,
        })
    }
}

impl MoeGroupedDownWorkspace {
    pub fn new(out_features: usize, in_features: usize, groups: usize) -> Result<Option<Self>> {
        let Some(gemv) = GroupedGemvWorkspace::new(out_features, in_features, groups)? else {
            return Ok(None);
        };
        let mut inputs = Vec::with_capacity(groups);
        let mut input_simple_scales = Vec::with_capacity(groups);
        let mut input_values = Vec::with_capacity(groups);
        let mut input_scales = Vec::with_capacity(groups);
        let mut input_values_mut = Vec::with_capacity(groups);
        let mut input_scales_mut = Vec::with_capacity(groups);
        for _ in 0..groups {
            let mut input = Nvfp4Matrix::zeroed_col_major(in_features, 1)?;
            let simple_scales = DeviceBuffer::zeroed(in_features.div_ceil(16))?;
            input_values.push(input.values_ptr());
            input_scales.push(simple_scales.as_const_ptr().cast::<u8>());
            input_values_mut.push(input.values_mut_ptr().cast());
            input_scales_mut.push(simple_scales.as_const_ptr().cast_mut().cast::<u8>());
            inputs.push(input);
            input_simple_scales.push(simple_scales);
        }
        Ok(Some(Self {
            gemv,
            inputs,
            input_simple_scales,
            input_values: DeviceBuffer::from_host(&input_values)?,
            input_scales: DeviceBuffer::from_host(&input_scales)?,
            input_values_mut: DeviceBuffer::from_host(&input_values_mut)?,
            input_scales_mut: DeviceBuffer::from_host(&input_scales_mut)?,
        }))
    }

    pub fn run_device_route(
        &mut self,
        route: &MoeRouteWorkspace,
        expert_ptrs: &MoeExpertPointerTables,
        gate_up_outputs: &DeviceBuffer<*const f32>,
        ffn_out: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<bool> {
        let groups = self.inputs.len();
        if route.indices.len() != groups || self.gemv.outputs.len() != groups {
            return Ok(false);
        }
        moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
            MoeSiluQuantizeSlotBuffers {
                indices: &route.indices,
                gate_up_table: gate_up_outputs,
                packed_table: self.input_values_mut.output(),
                scales_table: self.input_scales_mut.output(),
                input_scale_table: &expert_ptrs.down_input_scales,
                gate_up_alpha_table: &expert_ptrs.gate_up_alphas,
            },
            self.inputs[0].rows,
            stream,
        )?;
        gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
            GroupedGemvPointerTableBuffers {
                indices: &route.indices,
                a_values_table: &expert_ptrs.down_grouped_values,
                a_scales_table: &expert_ptrs.down_grouped_scales,
                b_values_table: &self.input_values,
                b_scales_table: &self.input_scales,
                c_table: self.gemv.c.inout(),
                d_table: self.gemv.d.inout(),
                out_a_values: self.gemv.a_values.output(),
                out_a_scales: self.gemv.a_scales.output(),
                out_b_values: self.gemv.b_values.output(),
                out_b_scales: self.gemv.b_scales.output(),
            },
            stream,
        )?;
        self.gemv.plan.run_on_stream(
            &self.gemv.a_values,
            &self.gemv.a_scales,
            &self.gemv.b_values,
            &self.gemv.b_scales,
            &self.gemv.c,
            &self.gemv.d,
            1.0,
            0.0,
            stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            &route.indices,
            &route.weights,
            &self.gemv.c,
            &expert_ptrs.down_alphas,
            ffn_out.inout(),
            stream,
        )?;
        Ok(true)
    }

    pub fn run_prequantized_device_route(
        &mut self,
        route: &MoeRouteWorkspace,
        expert_ptrs: &MoeExpertPointerTables,
        ffn_out: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<bool> {
        let groups = self.inputs.len();
        if route.indices.len() != groups || self.gemv.outputs.len() != groups {
            return Ok(false);
        }
        gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
            GroupedGemvPointerTableBuffers {
                indices: &route.indices,
                a_values_table: &expert_ptrs.down_grouped_values,
                a_scales_table: &expert_ptrs.down_grouped_scales,
                b_values_table: &self.input_values,
                b_scales_table: &self.input_scales,
                c_table: self.gemv.c.inout(),
                d_table: self.gemv.d.inout(),
                out_a_values: self.gemv.a_values.output(),
                out_a_scales: self.gemv.a_scales.output(),
                out_b_values: self.gemv.b_values.output(),
                out_b_scales: self.gemv.b_scales.output(),
            },
            stream,
        )?;
        self.gemv.plan.run_on_stream(
            &self.gemv.a_values,
            &self.gemv.a_scales,
            &self.gemv.b_values,
            &self.gemv.b_scales,
            &self.gemv.c,
            &self.gemv.d,
            1.0,
            0.0,
            stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            &route.indices,
            &route.weights,
            &self.gemv.c,
            &expert_ptrs.down_alphas,
            ffn_out.inout(),
            stream,
        )?;
        Ok(true)
    }
}

impl GroupedGemvWorkspace {
    pub fn new(out_features: usize, in_features: usize, groups: usize) -> Result<Option<Self>> {
        if !CutlassFp4GroupedGemvF32Plan::supported(out_features, in_features, groups) {
            return Ok(None);
        }
        let plan = CutlassFp4GroupedGemvF32Plan::new(out_features, in_features, groups)?;
        let mut outputs = Vec::with_capacity(groups);
        let mut c_ptrs = Vec::with_capacity(groups);
        let mut d_ptrs = Vec::with_capacity(groups);
        for _ in 0..groups {
            let mut output = F32Matrix::zeroed(out_features, 1)?;
            c_ptrs.push(output.data_ptr());
            d_ptrs.push(output.data_mut_ptr().cast());
            outputs.push(output);
        }
        Ok(Some(Self {
            plan,
            a_values: DeviceBuffer::from_host(&vec![std::ptr::null(); groups])?,
            a_scales: DeviceBuffer::from_host(&vec![std::ptr::null(); groups])?,
            b_values: DeviceBuffer::from_host(&vec![std::ptr::null(); groups])?,
            b_scales: DeviceBuffer::from_host(&vec![std::ptr::null(); groups])?,
            c: DeviceBuffer::from_host(&c_ptrs)?,
            d: DeviceBuffer::from_host(&d_ptrs)?,
            outputs,
        }))
    }

    pub fn run_gate_up_device_route(
        &mut self,
        route: &MoeRouteWorkspace,
        expert_ptrs: &MoeExpertPointerTables,
        gate_up_input: &Nvfp4Matrix,
        gate_up_input_scales: *const u8,
        stream: &CudaStream,
    ) -> Result<bool> {
        let groups = self.outputs.len();
        if route.indices.len() != groups {
            return Ok(false);
        }
        gather_nvfp4_grouped_gemv_ptrs_on_stream(
            GroupedGemvPointerBuffers {
                indices: &route.indices,
                a_values_table: &expert_ptrs.gate_up_grouped_values,
                a_scales_table: &expert_ptrs.gate_up_grouped_scales,
                b_values: gate_up_input.values_ptr(),
                b_scales: gate_up_input_scales,
                c_table: self.c.inout(),
                d_table: self.d.inout(),
                out_a_values: self.a_values.output(),
                out_a_scales: self.a_scales.output(),
                out_b_values: self.b_values.output(),
                out_b_scales: self.b_scales.output(),
            },
            stream,
        )?;
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
        )?;
        Ok(true)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_dense_decode_ffn(
    config: QwenModelConfig,
    lt: &CublasLt,
    gate_up: &mut LinearDecodeOp,
    ffn_activated: &mut DeviceBuffer<f32>,
    down: &mut LinearDecodeOp,
    gate_up_proj: &LayerLinear,
    down_proj: &LayerLinear,
    ffn_norm: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
    count: bool,
) -> Result<()> {
    if count {
        inc_counter(&RUNTIME_COUNTERS.quantize_calls);
    }
    gate_up.run(lt, gate_up_proj, config.hidden, ffn_norm, stream, count)?;
    if count {
        inc_counter(&RUNTIME_COUNTERS.silu_calls);
    }
    silu_mul_halves_f32_into_on_stream(
        gate_up.output(),
        ffn_activated.output(),
        config.intermediate,
        stream,
    )?;
    if count {
        inc_counter(&RUNTIME_COUNTERS.quantize_calls);
    }
    down.run(
        lt,
        down_proj,
        config.intermediate,
        ffn_activated,
        stream,
        count,
    )?;
    if count {
        inc_counter(&RUNTIME_COUNTERS.add_calls);
    }
    add_f32_into_on_stream(residual, down.output(), output.output(), stream)
}

#[allow(clippy::too_many_arguments)]
fn run_moe_decode_ffn(
    config: QwenModelConfig,
    lt: &CublasLt,
    router: &Bf16Linear,
    experts: &mut [MoeExpertDecodeWorkspace],
    expert_streams: &[CudaStream],
    expert_ready_events: &[CudaEvent],
    route_ready_event: &CudaEvent,
    router_logits: &mut DeviceBuffer<f32>,
    route_workspace: &mut MoeRouteWorkspace,
    gate_up_input: &mut Nvfp4Matrix,
    expert_ptrs: &MoeExpertPointerTables,
    mut grouped_gate_up: Option<&mut GroupedGemvWorkspace>,
    mut grouped_down: Option<&mut MoeGroupedDownWorkspace>,
    expert_weights: &[LazyMoeExpertWeights],
    expert_intermediate: usize,
    experts_per_token: usize,
    norm_topk_prob: bool,
    ffn_norm: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    ffn_out: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
    count: bool,
    mut profile: Option<&mut QwenDecodeProfile>,
) -> Result<()> {
    if expert_streams.len() < experts_per_token || expert_ready_events.len() < experts_per_token {
        return Err(Error::Shape {
            label: "MoE expert streams",
            expected: format!("{experts_per_token} streams/events"),
            actual: format!(
                "{} streams, {} events",
                expert_streams.len(),
                expert_ready_events.len()
            ),
        });
    }
    let route_wall_start = profile.as_ref().map(|_| Instant::now());
    router.run_logits_into_on_stream(ffn_norm, router_logits, stream)?;
    route_workspace.run_topk(router_logits, norm_topk_prob, stream)?;
    let shared_gate_up_input_scale = expert_ptrs.shared_gate_up_input_scale;
    if let Some(gate_up_input_scale) = shared_gate_up_input_scale
        && let (Some(grouped_gate_up), Some(grouped_down)) =
            (grouped_gate_up.as_mut(), grouped_down.as_mut())
    {
        if count {
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        }
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            config.hidden,
            1,
            ffn_norm,
            gate_up_input,
            gate_up_input_scale,
            stream,
        )?;
        if count {
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_calls, experts_per_token as u64);
            add_counter(
                &RUNTIME_COUNTERS.fp4_gemm_m_total,
                (expert_intermediate * 2 * experts_per_token) as u64,
            );
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total, experts_per_token as u64);
            add_counter(
                &RUNTIME_COUNTERS.fp4_gemm_k_total,
                (config.hidden * experts_per_token) as u64,
            );
        }
        grouped_gate_up.run_gate_up_device_route(
            route_workspace,
            expert_ptrs,
            gate_up_input,
            gate_up_input.scales_ptr(),
            stream,
        )?;
        if count {
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_calls, experts_per_token as u64);
            add_counter(
                &RUNTIME_COUNTERS.fp4_gemm_m_total,
                (config.hidden * experts_per_token) as u64,
            );
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total, experts_per_token as u64);
            add_counter(
                &RUNTIME_COUNTERS.fp4_gemm_k_total,
                (expert_intermediate * experts_per_token) as u64,
            );
            add_counter(&RUNTIME_COUNTERS.silu_calls, experts_per_token as u64);
            add_counter(&RUNTIME_COUNTERS.quantize_calls, experts_per_token as u64);
        }
        grouped_down.run_device_route(
            route_workspace,
            expert_ptrs,
            &grouped_gate_up.c,
            ffn_out,
            stream,
        )?;
        if let Some(start) = route_wall_start
            && let Some(profile) = profile.as_deref_mut()
        {
            profile.moe_route_wall_ms += start.elapsed().as_secs_f64() * 1_000.0;
        }
        if count {
            inc_counter(&RUNTIME_COUNTERS.add_calls);
        }
        return add_f32_into_on_stream(residual, ffn_out, output.output(), stream);
    }

    let route = route_workspace.copy_to_host(stream)?;
    if let Some(start) = route_wall_start
        && let Some(profile) = profile
    {
        profile.moe_route_wall_ms += start.elapsed().as_secs_f64() * 1_000.0;
    }
    if let Some((first_expert_idx, _)) = route
        .first()
        .filter(|_| shared_gate_up_input_scale.is_some())
    {
        let expert_weight = expert_weights
            .get(*first_expert_idx)
            .ok_or_else(|| Error::Shape {
                label: "MoE expert weight index",
                expected: format!("expert < {}", expert_weights.len()),
                actual: first_expert_idx.to_string(),
            })?
            .get()?;
        if count {
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        }
        expert_weight
            .gate_up_proj
            .device
            .quantize_activation_device_col_major_f32_into_on_stream(
                config.hidden,
                1,
                ffn_norm,
                gate_up_input,
                stream,
            )?;
    }
    fill_f32_into_on_stream(ffn_out.output(), 0.0, stream)?;
    route_ready_event.record_on_stream(stream)?;
    for (slot, (expert_idx, _)) in route.iter().copied().enumerate() {
        let expert_stream = &expert_streams[slot];
        expert_stream.wait_event(route_ready_event)?;
        let expert_count = experts.len();
        let expert = experts.get_mut(expert_idx).ok_or_else(|| Error::Shape {
            label: "MoE expert index",
            expected: format!("expert < {expert_count}"),
            actual: expert_idx.to_string(),
        })?;
        let expert_weight = expert_weights
            .get(expert_idx)
            .ok_or_else(|| Error::Shape {
                label: "MoE expert weight index",
                expected: format!("expert < {}", expert_weights.len()),
                actual: expert_idx.to_string(),
            })?
            .get()?;
        expert.ensure(config, lt, &expert_weight, expert_intermediate)?;
        let (Some(gate_up), Some(down)) = (&mut expert.gate_up, &mut expert.down) else {
            unreachable!("MoE expert plans initialized");
        };
        let gate_up_output = if shared_gate_up_input_scale.is_some() {
            gate_up.run_with_quantized_input(
                lt,
                &expert_weight.gate_up_proj,
                config.hidden,
                gate_up_input,
                expert_stream,
                count,
            )?;
            gate_up.output()
        } else {
            if count {
                inc_counter(&RUNTIME_COUNTERS.quantize_calls);
            }
            gate_up.run(
                lt,
                &expert_weight.gate_up_proj,
                config.hidden,
                ffn_norm,
                expert_stream,
                count,
            )?;
            gate_up.output()
        };
        if count {
            inc_counter(&RUNTIME_COUNTERS.silu_calls);
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
        }
        down.quantize_silu_halves(&expert_weight.down_proj, gate_up_output, expert_stream)?;
        expert_ready_events[slot].record_on_stream(expert_stream)?;
    }
    for (slot, (expert_idx, weight)) in route.into_iter().enumerate() {
        stream.wait_event(&expert_ready_events[slot])?;
        let expert_count = experts.len();
        let expert = experts.get_mut(expert_idx).ok_or_else(|| Error::Shape {
            label: "MoE expert index",
            expected: format!("expert < {expert_count}"),
            actual: expert_idx.to_string(),
        })?;
        let expert_weight = expert_weights
            .get(expert_idx)
            .ok_or_else(|| Error::Shape {
                label: "MoE expert weight index",
                expected: format!("expert < {}", expert_weights.len()),
                actual: expert_idx.to_string(),
            })?
            .get()?;
        expert.down_mut().run_quantized_accumulate_into(
            lt,
            &expert_weight.down_proj,
            expert_intermediate,
            ffn_out,
            weight,
            stream,
            count,
        )?;
    }
    if count {
        inc_counter(&RUNTIME_COUNTERS.add_calls);
    }
    add_f32_into_on_stream(residual, ffn_out, output.output(), stream)
}

fn select_moe_experts(logits: &[f32], k: usize, norm_topk_prob: bool) -> Result<Vec<(usize, f32)>> {
    if logits.is_empty() || k == 0 || k > logits.len() {
        return Err(Error::Shape {
            label: "MoE router logits",
            expected: "0 < k <= experts".to_string(),
            actual: format!("experts={} k={k}", logits.len()),
        });
    }
    let sanitized = logits
        .iter()
        .map(|&value| {
            if value.is_nan() {
                f32::NEG_INFINITY
            } else if value == f32::INFINITY {
                f32::MAX
            } else if value == f32::NEG_INFINITY {
                f32::NEG_INFINITY
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let max = sanitized.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(Error::Shape {
            label: "MoE router logits",
            expected: "at least one finite router logit".to_string(),
            actual: "all logits were non-finite".to_string(),
        });
    }
    let mut probs = logits
        .iter()
        .enumerate()
        .map(|(idx, _)| (idx, (sanitized[idx] - max).exp()))
        .collect::<Vec<_>>();
    let sum = probs.iter().map(|(_, prob)| *prob).sum::<f32>();
    for (_, prob) in &mut probs {
        *prob /= sum;
    }
    probs.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    probs.truncate(k);
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if norm_topk_prob {
        let selected_sum = probs.iter().map(|(_, prob)| *prob).sum::<f32>();
        for (_, prob) in &mut probs {
            *prob /= selected_sum;
        }
    }
    Ok(probs)
}

impl LinearDecodeOp {
    fn new(
        lt: &CublasLt,
        linear: &LayerLinear,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        Self::new_with_cutlass(lt, linear, out_features, in_features, true)
    }

    fn new_with_cutlass(
        lt: &CublasLt,
        linear: &LayerLinear,
        out_features: usize,
        in_features: usize,
        allow_cutlass_gemv: bool,
    ) -> Result<Self> {
        let input = Nvfp4Matrix::zeroed_col_major(in_features, 1)?;
        let c = F32Matrix::zeroed(out_features, 1)?;
        let d = F32Matrix::zeroed(out_features, 1)?;
        let shape = GemmShape::new(out_features, 1, in_features);
        let plan = Fp4TnMatmulPlan::new_f32_output(
            lt,
            shape,
            Nvfp4TnInputs::new(linear.device.matrix(), &input),
            &c,
            WORKSPACE_LIMIT,
        )?;
        let use_cutlass_gemv = allow_cutlass_gemv && plan.cutlass_fp4_gemv_f32_supported();
        Ok(Self {
            input,
            c,
            d,
            plan,
            use_cutlass_gemv,
        })
    }

    fn run(
        &mut self,
        lt: &CublasLt,
        linear: &LayerLinear,
        input_rows: usize,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
        count: bool,
    ) -> Result<()> {
        linear
            .device
            .quantize_activation_device_col_major_f32_into_on_stream(
                input_rows,
                1,
                input,
                &mut self.input,
                stream,
            )?;
        run_linear_decode_on_stream(lt, linear, self, input_rows, stream, count)
    }

    fn quantize(
        &mut self,
        linear: &LayerLinear,
        input_rows: usize,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        linear
            .device
            .quantize_activation_device_col_major_f32_into_on_stream(
                input_rows,
                1,
                input,
                &mut self.input,
                stream,
            )
    }

    fn quantize_silu_halves(
        &mut self,
        linear: &LayerLinear,
        gate_up: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        silu_mul_halves_quantize_nvfp4_col_major_f32_into_on_stream(
            gate_up,
            &mut self.input,
            linear.device.input_scale(),
            stream,
        )
    }

    fn run_quantized(
        &mut self,
        lt: &CublasLt,
        linear: &LayerLinear,
        input_rows: usize,
        stream: &CudaStream,
        count: bool,
    ) -> Result<()> {
        run_linear_decode_on_stream(lt, linear, self, input_rows, stream, count)
    }

    fn run_quantized_accumulate_into(
        &mut self,
        lt: &CublasLt,
        linear: &LayerLinear,
        input_rows: usize,
        output: &mut DeviceBuffer<f32>,
        scale: f32,
        stream: &CudaStream,
        count: bool,
    ) -> Result<()> {
        if count {
            inc_counter(&RUNTIME_COUNTERS.fp4_gemm_calls);
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_m_total, self.d.rows as u64);
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total, 1);
            add_counter(&RUNTIME_COUNTERS.fp4_gemm_k_total, input_rows as u64);
        }
        let inputs = Nvfp4TnInputs::new(linear.device.matrix(), &self.input);
        self.plan.run_with_alpha_beta_f32_inout_buffer_on_stream(
            lt,
            inputs,
            output.inout(),
            linear.device.matmul_alpha() * scale,
            1.0,
            stream,
        )
    }

    fn run_with_quantized_input(
        &mut self,
        lt: &CublasLt,
        linear: &LayerLinear,
        input_rows: usize,
        input: &Nvfp4Matrix,
        stream: &CudaStream,
        count: bool,
    ) -> Result<()> {
        run_linear_decode_with_input_on_stream(lt, linear, self, input_rows, input, stream, count)
    }

    fn output(&self) -> &DeviceBuffer<f32> {
        self.d.data()
    }

    fn output_mut(&mut self) -> &mut DeviceBuffer<f32> {
        self.d.data_mut()
    }
}

fn run_linear_decode_on_stream(
    lt: &CublasLt,
    linear: &LayerLinear,
    op: &mut LinearDecodeOp,
    in_features: usize,
    stream: &CudaStream,
    count: bool,
) -> Result<()> {
    if count {
        inc_counter(&RUNTIME_COUNTERS.fp4_gemm_calls);
        add_counter(&RUNTIME_COUNTERS.fp4_gemm_m_total, op.d.rows as u64);
        add_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total, 1);
        add_counter(&RUNTIME_COUNTERS.fp4_gemm_k_total, in_features as u64);
    }
    let inputs = Nvfp4TnInputs::new(linear.device.matrix(), &op.input);
    let alpha = linear.device.matmul_alpha();
    if op.use_cutlass_gemv {
        op.plan
            .run_cutlass_fp4_gemv_f32_on_stream(inputs, &op.c, &mut op.d, alpha, stream)
    } else {
        op.plan
            .run_with_alpha_f32_output_on_stream(lt, inputs, &op.c, &mut op.d, alpha, stream)
    }
}

fn run_linear_decode_with_input_on_stream(
    lt: &CublasLt,
    linear: &LayerLinear,
    op: &mut LinearDecodeOp,
    in_features: usize,
    input: &Nvfp4Matrix,
    stream: &CudaStream,
    count: bool,
) -> Result<()> {
    if count {
        inc_counter(&RUNTIME_COUNTERS.fp4_gemm_calls);
        add_counter(&RUNTIME_COUNTERS.fp4_gemm_m_total, op.d.rows as u64);
        add_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total, 1);
        add_counter(&RUNTIME_COUNTERS.fp4_gemm_k_total, in_features as u64);
    }
    let inputs = Nvfp4TnInputs::new(linear.device.matrix(), input);
    let alpha = linear.device.matmul_alpha();
    if op.use_cutlass_gemv {
        op.plan
            .run_cutlass_fp4_gemv_f32_on_stream(inputs, &op.c, &mut op.d, alpha, stream)
    } else {
        op.plan
            .run_with_alpha_f32_output_on_stream(lt, inputs, &op.c, &mut op.d, alpha, stream)
    }
}

fn run_layer_prefill(
    config: QwenModelConfig,
    lt: &CublasLt,
    layer_idx: usize,
    weights: &QwenLayerWeights,
    hidden: DeviceBuffer<f32>,
    kv_cache: &mut KvCache,
    start_position: usize,
    tokens: usize,
) -> Result<DeviceBuffer<f32>> {
    inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
    let stream = CudaStream::new_blocking()?;
    let mut normed_hidden = DeviceBuffer::zeroed(tokens * config.hidden)?;
    rms_norm_f32_into_on_stream(
        tokens,
        config.hidden,
        &hidden,
        &weights.input_norm_weight.device,
        normed_hidden.output(),
        config.rms_eps,
        &stream,
    )?;
    add_counter(&RUNTIME_COUNTERS.quantize_calls, 3);
    let q_input = weights
        .q_proj
        .device
        .quantize_activation_device_col_major_f32(config.hidden, tokens, &normed_hidden)?;
    let k_input = weights
        .k_proj
        .device
        .quantize_activation_device_col_major_f32(config.hidden, tokens, &normed_hidden)?;
    let v_input = weights
        .v_proj
        .device
        .quantize_activation_device_col_major_f32(config.hidden, tokens, &normed_hidden)?;

    let q = run_linear_device(
        lt,
        &weights.q_proj,
        &q_input,
        config.q_width,
        tokens,
        config.hidden,
    )?;
    let k = run_linear_device(
        lt,
        &weights.k_proj,
        &k_input,
        config.kv_width,
        tokens,
        config.hidden,
    )?;
    let v = run_linear_device(
        lt,
        &weights.v_proj,
        &v_input,
        config.kv_width,
        tokens,
        config.hidden,
    )?;

    add_counter(&RUNTIME_COUNTERS.rms_norm_calls, 2);
    let mut q_normed = DeviceBuffer::zeroed(tokens * config.q_width)?;
    rms_norm_f32_into_on_stream(
        tokens * config.q_heads,
        config.head_dim,
        &q,
        &weights.q_norm_weight.device,
        q_normed.output(),
        config.rms_eps,
        &stream,
    )?;
    let mut k_normed = DeviceBuffer::zeroed(tokens * config.kv_width)?;
    rms_norm_f32_into_on_stream(
        tokens * config.kv_heads,
        config.head_dim,
        &k,
        &weights.k_norm_weight.device,
        k_normed.output(),
        config.rms_eps,
        &stream,
    )?;
    add_counter(&RUNTIME_COUNTERS.rope_calls, 2);
    let mut q_rope = DeviceBuffer::zeroed(tokens * config.q_width)?;
    rope_neox_sequence_f32_into_on_stream(
        tokens,
        config.q_heads,
        config.head_dim,
        &q_normed,
        q_rope.output(),
        start_position,
        config.rope_theta,
        &stream,
    )?;
    let mut k_rope = DeviceBuffer::zeroed(tokens * config.kv_width)?;
    rope_neox_sequence_f32_into_on_stream(
        tokens,
        config.kv_heads,
        config.head_dim,
        &k_normed,
        k_rope.output(),
        start_position,
        config.rope_theta,
        &stream,
    )?;

    stream.synchronize()?;
    kv_cache.layer_mut(layer_idx)?.append(&k_rope, &v)?;
    inc_counter(&RUNTIME_COUNTERS.attention_calls);
    let mut attn = DeviceBuffer::zeroed(tokens * config.q_width)?;
    kv_cache.layer(layer_idx)?.prefill_attention_into(
        &q_rope,
        &mut attn,
        tokens,
        start_position,
        config.q_heads,
    )?;
    inc_counter(&RUNTIME_COUNTERS.quantize_calls);
    let attn_input = weights
        .o_proj
        .device
        .quantize_activation_device_col_major_f32(config.q_width, tokens, &attn)?;
    let o = run_linear_device(
        lt,
        &weights.o_proj,
        &attn_input,
        config.hidden,
        tokens,
        config.q_width,
    )?;
    inc_counter(&RUNTIME_COUNTERS.add_calls);
    let mut attn_residual = DeviceBuffer::zeroed(tokens * config.hidden)?;
    add_f32_into_on_stream(&hidden, &o, attn_residual.output(), &stream)?;

    inc_counter(&RUNTIME_COUNTERS.rms_norm_calls);
    let mut ffn_norm = DeviceBuffer::zeroed(tokens * config.hidden)?;
    rms_norm_f32_into_on_stream(
        tokens,
        config.hidden,
        &attn_residual,
        &weights.post_attn_norm_weight.device,
        ffn_norm.output(),
        config.rms_eps,
        &stream,
    )?;
    stream.synchronize()?;
    let down = run_prefill_ffn(config, lt, &weights.ffn, &ffn_norm, tokens)?;
    inc_counter(&RUNTIME_COUNTERS.add_calls);
    let mut output = DeviceBuffer::zeroed(tokens * config.hidden)?;
    add_f32_into_on_stream(&attn_residual, &down, output.output(), &stream)?;
    stream.synchronize()?;
    Ok(output)
}

fn run_prefill_ffn(
    config: QwenModelConfig,
    lt: &CublasLt,
    weights: &QwenFfnWeights,
    ffn_norm: &DeviceBuffer<f32>,
    tokens: usize,
) -> Result<DeviceBuffer<f32>> {
    match (config.ffn, weights) {
        (
            QwenFfnConfig::Dense,
            QwenFfnWeights::Dense {
                gate_proj,
                up_proj,
                down_proj,
                ..
            },
        ) => run_prefill_dense_ffn(config, lt, gate_proj, up_proj, down_proj, ffn_norm, tokens),
        (
            QwenFfnConfig::Moe {
                experts_per_token,
                expert_intermediate,
                norm_topk_prob,
                ..
            },
            QwenFfnWeights::Moe { router, experts },
        ) => run_prefill_moe_ffn(
            config,
            lt,
            router,
            experts,
            expert_intermediate,
            experts_per_token,
            norm_topk_prob,
            ffn_norm,
            tokens,
        ),
        _ => Err(Error::Format {
            label: "Qwen FFN prefill",
            detail: "config and weight FFN variants do not match".to_string(),
        }),
    }
}

fn run_prefill_dense_ffn(
    config: QwenModelConfig,
    lt: &CublasLt,
    gate_proj: &LayerLinear,
    up_proj: &LayerLinear,
    down_proj: &LayerLinear,
    ffn_norm: &DeviceBuffer<f32>,
    tokens: usize,
) -> Result<DeviceBuffer<f32>> {
    add_counter(&RUNTIME_COUNTERS.quantize_calls, 2);
    let gate_input = gate_proj.device.quantize_activation_device_col_major_f32(
        config.hidden,
        tokens,
        ffn_norm,
    )?;
    let up_input =
        up_proj
            .device
            .quantize_activation_device_col_major_f32(config.hidden, tokens, ffn_norm)?;
    let gate = run_linear_device(
        lt,
        gate_proj,
        &gate_input,
        config.intermediate,
        tokens,
        config.hidden,
    )?;
    let up = run_linear_device(
        lt,
        up_proj,
        &up_input,
        config.intermediate,
        tokens,
        config.hidden,
    )?;
    inc_counter(&RUNTIME_COUNTERS.silu_calls);
    let mut ffn_activated = DeviceBuffer::zeroed(tokens * config.intermediate)?;
    let stream = CudaStream::new_blocking()?;
    silu_mul_f32_into_on_stream(&gate, &up, ffn_activated.output(), &stream)?;
    stream.synchronize()?;
    inc_counter(&RUNTIME_COUNTERS.quantize_calls);
    let ffn_input = down_proj.device.quantize_activation_device_col_major_f32(
        config.intermediate,
        tokens,
        &ffn_activated,
    )?;
    run_linear_device(
        lt,
        down_proj,
        &ffn_input,
        config.hidden,
        tokens,
        config.intermediate,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_moe_ffn(
    config: QwenModelConfig,
    lt: &CublasLt,
    router: &Bf16Linear,
    experts: &[LazyMoeExpertWeights],
    expert_intermediate: usize,
    experts_per_token: usize,
    norm_topk_prob: bool,
    ffn_norm: &DeviceBuffer<f32>,
    tokens: usize,
) -> Result<DeviceBuffer<f32>> {
    let stream = CudaStream::new_non_blocking()?;
    let mut output = DeviceBuffer::zeroed(tokens * config.hidden)?;
    fill_f32_into_on_stream(output.output(), 0.0, &stream)?;
    let mut token_norm = DeviceBuffer::zeroed(config.hidden)?;
    let mut token_out = DeviceBuffer::zeroed(config.hidden)?;
    let mut router_logits = DeviceBuffer::zeroed(router.rows)?;
    let mut activated = DeviceBuffer::zeroed(expert_intermediate)?;
    for token_idx in 0..tokens {
        copy_row_f32_into_on_stream(
            tokens,
            config.hidden,
            token_idx,
            ffn_norm,
            token_norm.output(),
            &stream,
        )?;
        fill_f32_into_on_stream(token_out.output(), 0.0, &stream)?;
        router.run_logits_into_on_stream(&token_norm, &mut router_logits, &stream)?;
        let route = select_moe_experts(
            &router_logits.copy_to_host(&stream)?,
            experts_per_token,
            norm_topk_prob,
        )?;
        for (expert_idx, weight) in route {
            let expert = experts[expert_idx].get()?;
            stream.synchronize()?;
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
            let gate_up_input = expert
                .gate_up_proj
                .device
                .quantize_activation_device_col_major_f32(config.hidden, 1, &token_norm)?;
            let gate_up = run_linear_device(
                lt,
                &expert.gate_up_proj,
                &gate_up_input,
                expert_intermediate * 2,
                1,
                config.hidden,
            )?;
            inc_counter(&RUNTIME_COUNTERS.silu_calls);
            silu_mul_halves_f32_into_on_stream(
                &gate_up,
                activated.output(),
                expert_intermediate,
                &stream,
            )?;
            stream.synchronize()?;
            inc_counter(&RUNTIME_COUNTERS.quantize_calls);
            let down_input = expert
                .down_proj
                .device
                .quantize_activation_device_col_major_f32(expert_intermediate, 1, &activated)?;
            let down = run_linear_device(
                lt,
                &expert.down_proj,
                &down_input,
                config.hidden,
                1,
                expert_intermediate,
            )?;
            synchronize_device()?;
            scaled_add_f32_into_on_stream(&down, token_out.inout(), weight, &stream)?;
        }
        append_rows_f32_into_on_stream(
            &token_out,
            output.output(),
            token_idx,
            1,
            config.hidden,
            &stream,
        )?;
    }
    stream.synchronize()?;
    Ok(output)
}

fn run_linear_device(
    lt: &CublasLt,
    linear: &LayerLinear,
    input: &ModelOptNvfp4Activation,
    out_features: usize,
    cols: usize,
    in_features: usize,
) -> Result<DeviceBuffer<f32>> {
    inc_counter(&RUNTIME_COUNTERS.fp4_gemm_calls);
    add_counter(&RUNTIME_COUNTERS.fp4_gemm_m_total, out_features as u64);
    add_counter(&RUNTIME_COUNTERS.fp4_gemm_n_total, cols as u64);
    add_counter(&RUNTIME_COUNTERS.fp4_gemm_k_total, in_features as u64);
    let shape = GemmShape::new(out_features, cols, in_features);
    let c = F32Matrix::zeroed(out_features, cols)?;
    let mut d = F32Matrix::zeroed(out_features, cols)?;
    let plan = Fp4TnMatmulPlan::new_f32_output(
        lt,
        shape,
        Nvfp4TnInputs::new(linear.device.matrix(), input.matrix()),
        &c,
        WORKSPACE_LIMIT,
    )?;
    plan.run_with_alpha_f32_output_on_default_stream(
        lt,
        Nvfp4TnInputs::new(linear.device.matrix(), input.matrix()),
        &c,
        &mut d,
        linear.device.matmul_alpha(),
    )?;
    Ok(d.into_data())
}

fn read_bf16_rows_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    rows: &[u32],
    cols: usize,
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape.len() != 2 || info.shape[1] != cols {
        return Err(Error::Shape {
            label: "BF16 rows tensor",
            expected: format!("dtype=BF16 shape=[*,{cols}]"),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let row_bytes = cols * 2;
    let mut values = Vec::with_capacity(rows.len() * cols);
    for &row in rows {
        let row = row as usize;
        if row >= info.shape[0] {
            return Err(Error::Shape {
                label: "BF16 rows index",
                expected: format!("row < {}", info.shape[0]),
                actual: row.to_string(),
            });
        }
        let bytes = shard.read_tensor_byte_range(name, (row * row_bytes) as u64, row_bytes)?;
        values.extend(
            bytes
                .chunks_exact(2)
                .map(|chunk| format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))),
        );
    }
    DeviceBuffer::from_host(&values)
}

fn read_bf16_vector(checkpoint: &ModelOptCheckpoint, name: &str) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    let expected_bytes = info.shape.iter().product::<usize>() * 2;
    if info.dtype != "BF16" || info.byte_len() != expected_bytes as u64 {
        return Err(Error::Shape {
            label: "BF16 vector",
            expected: format!("dtype=BF16 bytes={expected_bytes}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn read_bf16_matrix_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    let expected_bytes = rows * cols * 2;
    if info.dtype != "BF16"
        || info.shape != [rows, cols]
        || info.byte_len() != expected_bytes as u64
    {
        return Err(Error::Shape {
            label: "BF16 matrix f32",
            expected: format!("dtype=BF16 shape=[{rows},{cols}] bytes={expected_bytes}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn read_bf16_matrix_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    let expected_bytes = rows * cols * 2;
    if info.dtype != "BF16"
        || info.shape != [rows, cols]
        || info.byte_len() != expected_bytes as u64
    {
        return Err(Error::Shape {
            label: "BF16 matrix",
            expected: format!("dtype=BF16 shape=[{rows},{cols}] bytes={expected_bytes}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    let bytes = shard.read_tensor_bytes(name)?;
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(&values)
}

#[cfg(test)]
mod tests {
    use super::{
        qwen35_reorder_qkv_rows_grouped_to_tiled, qwen35_reorder_rows_grouped_to_tiled,
        qwen35_v_head_tiled_permutation, select_moe_experts,
    };

    #[test]
    fn moe_topk_normalizes_selected_probabilities() {
        let selected = select_moe_experts(&[0.0, 3.0, 1.0, 2.0], 2, true).unwrap();
        assert_eq!(selected[0].0, 1);
        assert_eq!(selected[1].0, 3);
        let sum = selected.iter().map(|(_, weight)| *weight).sum::<f32>();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(selected[0].1 > selected[1].1);
    }

    #[test]
    fn qwen35_v_head_permutation_moves_grouped_heads_to_tiled_order() {
        let perm = qwen35_v_head_tiled_permutation(2, 4, 2).unwrap();
        assert_eq!(perm, vec![0, 1, 4, 5, 2, 3, 6, 7]);
    }

    #[test]
    fn qwen35_reorder_rows_moves_grouped_v_heads_to_tiled_order() {
        let rows = (0u32..8)
            .flat_map(|row| [row * 10, row * 10 + 1])
            .collect::<Vec<_>>();
        let reordered = qwen35_reorder_rows_grouped_to_tiled(&rows, 2, 2, 4, 2).unwrap();
        assert_eq!(
            reordered,
            vec![0, 1, 10, 11, 40, 41, 50, 51, 20, 21, 30, 31, 60, 61, 70, 71]
        );
    }

    #[test]
    fn qwen35_reorder_qkv_preserves_qk_and_reorders_only_v() {
        let rows = (0u32..8).collect::<Vec<_>>();
        let reordered = qwen35_reorder_qkv_rows_grouped_to_tiled(&rows, 1, 2, 4, 1, 1).unwrap();
        assert_eq!(reordered, vec![0, 1, 2, 3, 4, 6, 5, 7]);
    }
}
