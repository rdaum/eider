//! Qwen3.8 Flash Next checkpoint configuration parsing and validation.

use crate::qwen3::infer::{
    QwenArchitecture, QwenFfnConfig, QwenLayerKind, QwenLinearAttentionConfig, QwenModelManifest,
};
use eider_cuda::{Error, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Parsed text configuration for the released Qwen3.8 Flash Next checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(missing_docs)]
pub struct Qwen38FlashNextConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub layer_types: Vec<Qwen38LayerType>,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel: usize,
    pub experts: usize,
    pub experts_per_token: usize,
    pub expert_intermediate: usize,
    pub shared_expert_intermediate: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub ple_layer: usize,
    pub ple_embedding_dim: usize,
    pub ple_conv_kernel: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_shards: usize,
    pub ngram_vocab_base: usize,
    pub ngram_vocab_alignment: usize,
    pub indexer_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_compress_ratio: usize,
    pub indexer_budget: usize,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub rms_eps_bits: u32,
    pub rope_theta_bits: u32,
    pub mtp_layers: usize,
}

/// Per-layer attention type in the released text stack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38LayerType {
    /// Gated Delta Net recurrent attention.
    LinearAttention,
    /// Qwen Sparse Attention over the ordinary KV cache.
    FullAttention,
}

impl Qwen38FlashNextConfig {
    /// Reads and validates `config.json` without loading checkpoint tensors.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let bytes = fs::read(&path).map_err(|error| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: format!("{}: {error}", path.display()),
        })?;
        let root: Value = serde_json::from_slice(&bytes).map_err(|error| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: error.to_string(),
        })?;
        Self::from_value(&root)
    }

    pub(crate) fn from_value(root: &Value) -> Result<Self> {
        let model_type = required_str(root, "model_type")?;
        if model_type != "qwen3_8_flash_next" {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next config",
                detail: format!("unsupported model_type {model_type}"),
            });
        }
        let text = root.get("text_config").ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: "missing text_config".to_string(),
        })?;
        if required_str(text, "model_type")? != "qwen3_8_flash_next_text" {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next text config",
                detail: "unsupported text model_type".to_string(),
            });
        }

        let hidden = required_usize(text, "hidden_size")?;
        let q_heads = required_usize(text, "num_attention_heads")?;
        let head_dim = required_usize(text, "head_dim")?;
        let partial_rotary = required_f64(text, "partial_rotary_factor")?;
        let rotary_dim = (head_dim as f64 * partial_rotary).round() as usize;
        let layer_types = required_array(text, "layer_types")?
            .iter()
            .map(|value| match value.as_str() {
                Some("linear_attention") => Ok(Qwen38LayerType::LinearAttention),
                Some("full_attention") => Ok(Qwen38LayerType::FullAttention),
                other => Err(Error::Format {
                    label: "Qwen3.8 Flash Next layer type",
                    detail: format!("unsupported value {other:?}"),
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        let layers = required_usize(text, "num_hidden_layers")?;
        if layer_types.len() != layers {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next layer schedule",
                expected: format!("{layers} entries"),
                actual: layer_types.len().to_string(),
            });
        }
        let ple_layers = required_array(text, "ple_layer_ids")?;
        if ple_layers.len() != 1 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next PLE layers",
                expected: "one one-based layer ID".to_string(),
                actual: format!("{ple_layers:?}"),
            });
        }
        let one_based_ple = ple_layers[0].as_u64().ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next PLE layer",
            detail: "layer ID is not unsigned".to_string(),
        })? as usize;
        let ple_layer = one_based_ple.checked_sub(1).ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next PLE layer",
            expected: "positive one-based layer ID".to_string(),
            actual: one_based_ple.to_string(),
        })?;
        let output_gate = required_str(text, "output_gate_type")?;
        if output_gate != "sigmoid" {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next output gate",
                detail: format!("unsupported gate {output_gate}"),
            });
        }
        let eos = required_usize(text, "eos_token_id")?;
        let eos_token_id = u32::try_from(eos).map_err(|_| Error::Shape {
            label: "Qwen3.8 Flash Next EOS token",
            expected: "token fitting u32".to_string(),
            actual: eos.to_string(),
        })?;
        let rms_eps = required_f64(text, "rms_norm_eps")? as f32;
        let rope = text.get("rope_parameters").ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next RoPE",
            detail: "missing rope_parameters".to_string(),
        })?;
        let rope_theta = required_f64(rope, "rope_theta")? as f32;
        Ok(Self {
            hidden,
            vocab: required_usize(text, "vocab_size")?,
            layers,
            layer_types,
            q_heads,
            kv_heads: required_usize(text, "num_key_value_heads")?,
            head_dim,
            rotary_dim,
            linear_key_heads: required_usize(text, "linear_num_key_heads")?,
            linear_value_heads: required_usize(text, "linear_num_value_heads")?,
            linear_key_head_dim: required_usize(text, "linear_key_head_dim")?,
            linear_value_head_dim: required_usize(text, "linear_value_head_dim")?,
            linear_conv_kernel: required_usize(text, "linear_conv_kernel_dim")?,
            experts: required_usize(text, "num_experts")?,
            experts_per_token: required_usize(text, "num_experts_per_tok")?,
            expert_intermediate: required_usize(text, "moe_intermediate_size")?,
            shared_expert_intermediate: required_usize(text, "shared_expert_intermediate_size")?,
            hc_count: required_usize(text, "hc_count")?,
            hc_lowrank: required_usize(text, "hc_lowrank")?,
            ple_layer,
            ple_embedding_dim: required_usize(text, "ple_embed_dim")?,
            ple_conv_kernel: required_usize(text, "ple_conv_kernel_size")?,
            ngram_size: required_usize(text, "ngram_size")?,
            heads_per_ngram: required_usize(text, "heads_per_ngram")?,
            ngram_shards: required_usize(text, "split_ngram_parts")?,
            ngram_vocab_base: required_usize(text, "ngram_vocab_size_base")?,
            ngram_vocab_alignment: required_usize(text, "make_ngram_vocab_size_divisible_by")?,
            indexer_heads: required_usize(text, "indexer_n_heads")?,
            indexer_kv_heads: required_usize(text, "indexer_kv_heads")?,
            indexer_head_dim: required_usize(text, "indexer_head_dim")?,
            indexer_compress_ratio: required_usize(text, "indexer_compress_ratio")?,
            indexer_budget: required_usize(text, "indexer_budget")?,
            max_position_embeddings: required_usize(text, "max_position_embeddings")?,
            eos_token_id,
            rms_eps_bits: rms_eps.to_bits(),
            rope_theta_bits: rope_theta.to_bits(),
            mtp_layers: required_usize(text, "mtp_num_hidden_layers")?,
        })
    }

    /// Number of independent PLE hash heads selected for each token.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Width of one selected PLE embedding row.
    pub fn ngram_head_dim(&self) -> usize {
        self.ple_embedding_dim / self.ngram_heads()
    }

    /// RMSNorm epsilon represented by the checkpoint JSON.
    pub fn rms_eps(&self) -> f32 {
        f32::from_bits(self.rms_eps_bits)
    }

    /// RoPE theta represented by the checkpoint JSON.
    pub fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    pub(crate) fn qwen_manifest(&self) -> QwenModelManifest {
        QwenModelManifest {
            architecture: QwenArchitecture::Qwen38FlashNext,
            tensor_prefix: "model.language_model".to_string(),
            hidden: self.hidden,
            layers: self.layers,
            vocab: self.vocab,
            intermediate: self.expert_intermediate,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            rms_eps: self.rms_eps(),
            rope_theta: self.rope_theta(),
            mrope_sections: None,
            ffn: QwenFfnConfig::Moe {
                experts: self.experts,
                experts_per_token: self.experts_per_token,
                expert_intermediate: self.expert_intermediate,
                norm_topk_prob: true,
            },
            layer_kinds: self
                .layer_types
                .iter()
                .map(|kind| match kind {
                    Qwen38LayerType::LinearAttention => QwenLayerKind::LinearAttention,
                    Qwen38LayerType::FullAttention => QwenLayerKind::FullAttention,
                })
                .collect(),
            linear_attention: Some(QwenLinearAttentionConfig {
                conv_kernel: self.linear_conv_kernel,
                key_heads: self.linear_key_heads,
                value_heads: self.linear_value_heads,
                key_head_dim: self.linear_key_head_dim,
                value_head_dim: self.linear_value_head_dim,
            }),
            shared_expert_intermediate: Some(self.shared_expert_intermediate),
            mtp_layers: self.mtp_layers,
        }
    }
}

fn required_usize(value: &Value, name: &str) -> Result<usize> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: format!("missing unsigned integer {name}"),
        })
}

fn required_f64(value: &Value, name: &str) -> Result<f64> {
    value
        .get(name)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: format!("missing finite number {name}"),
        })
}

fn required_str<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: format!("missing string {name}"),
        })
}

fn required_array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value]> {
    value
        .get(name)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next config",
            detail: format!("missing array {name}"),
        })
}

#[cfg(test)]
mod tests {
    use super::{Qwen38FlashNextConfig, Qwen38LayerType};
    use serde_json::json;

    #[test]
    fn parses_released_checkpoint_config() {
        let layer_types = (0..48)
            .map(|layer| {
                if layer % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        let value = json!({
            "model_type": "qwen3_8_flash_next",
            "text_config": {
                "model_type": "qwen3_8_flash_next_text",
                "hidden_size": 2560,
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "layer_types": layer_types,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "split_ngram_parts": 128,
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_compress_ratio": 4,
                "indexer_budget": 2048,
                "max_position_embeddings": 262144,
                "eos_token_id": 248044,
                "rms_norm_eps": 0.000001,
                "rope_parameters": {"rope_theta": 10000000.0},
                "output_gate_type": "sigmoid",
                "mtp_num_hidden_layers": 1
            }
        });
        let config = Qwen38FlashNextConfig::from_value(&value).expect("Flash Next config");
        assert_eq!(config.hidden, 2560);
        assert_eq!(config.layers, 48);
        assert_eq!(config.layer_types[3], Qwen38LayerType::FullAttention);
        assert_eq!(config.experts, 512);
        assert_eq!(config.experts_per_token, 10);
        assert_eq!(config.ple_layer, 1);
        assert_eq!(config.ngram_heads(), 16);
        assert_eq!(config.ngram_head_dim(), 160);
        assert_eq!(config.indexer_budget, 2048);
    }
}
