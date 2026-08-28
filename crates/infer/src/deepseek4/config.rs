//! DeepSeek V4 checkpoint configuration parsing and validation.

use eider_cuda::{Error, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

const MODEL_TYPE: &str = "deepseek_v4";
const ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
const SCORING_FUNCTION: &str = "sqrtsoftplus";
const TOPK_METHOD: &str = "noaux_tc";
const ROPE_TYPE: &str = "yarn";
const HYPER_STREAMS: usize = 4;

/// Attention implementation selected by one DeepSeek V4 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Deepseek4AttentionKind {
    Sliding,
    CompressedSparse,
    HeavilyCompressed,
}

/// Validated DeepSeek V4 model architecture.
#[derive(Clone, Debug, PartialEq)]
pub struct Deepseek4ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub sliding_window: usize,
    pub compress_rope_theta: f32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_original_max_positions: usize,
    pub max_position_embeddings: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub expert_intermediate: usize,
    pub shared_experts: usize,
    pub hash_layers: usize,
    pub routed_scaling_factor: f32,
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub nextn_predict_layers: usize,
    layer_attention: Vec<Deepseek4AttentionKind>,
}

impl Deepseek4ModelConfig {
    /// Reads and validates a Hugging Face DeepSeek V4 `config.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let bytes = fs::read(&path).map_err(|error| Error::Format {
            label: "DeepSeek V4 config",
            detail: format!("failed to read {}: {error}", path.display()),
        })?;
        Self::from_json(&bytes).map_err(|error| match error {
            Error::Format { label, detail } => Error::Format {
                label,
                detail: format!("{}: {detail}", path.display()),
            },
            other => other,
        })
    }

    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw: RawDeepseek4Config =
            serde_json::from_slice(bytes).map_err(|error| Error::Format {
                label: "DeepSeek V4 config",
                detail: format!("invalid JSON: {error}"),
            })?;
        raw.validate()
    }

    /// Returns the attention implementation for `layer`.
    pub fn attention_kind(&self, layer: usize) -> Result<Deepseek4AttentionKind> {
        self.layer_attention
            .get(layer)
            .copied()
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 layer",
                expected: format!("layer < {}", self.layer_attention.len()),
                actual: layer.to_string(),
            })
    }

    /// Returns the checkpoint compression ratio for `layer`.
    pub fn compression_ratio(&self, layer: usize) -> Result<usize> {
        Ok(match self.attention_kind(layer)? {
            Deepseek4AttentionKind::Sliding => 0,
            Deepseek4AttentionKind::CompressedSparse => 4,
            Deepseek4AttentionKind::HeavilyCompressed => 128,
        })
    }

    /// Returns the inverse frequencies used by ordinary sliding-window RoPE.
    pub fn sliding_rope_inv_freq(&self) -> Vec<f32> {
        standard_rope_inv_freq(self.rope_theta, self.qk_rope_head_dim)
    }

    /// Returns the exact YaRN inverse frequencies used by compressed attention.
    pub fn compressed_rope_inv_freq(&self) -> Vec<f32> {
        yarn_rope_inv_freq(
            self.compress_rope_theta,
            self.qk_rope_head_dim,
            self.rope_factor,
            self.rope_original_max_positions,
            32.0,
            1.0,
        )
    }
}

#[derive(Deserialize)]
struct RawDeepseek4Config {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    q_lora_rank: usize,
    qk_rope_head_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
    sliding_window: usize,
    compress_ratios: Vec<usize>,
    compress_rope_theta: f32,
    rope_theta: f32,
    rope_scaling: RawRopeScaling,
    max_position_embeddings: usize,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    n_shared_experts: usize,
    num_hash_layers: usize,
    routed_scaling_factor: f32,
    scoring_func: String,
    topk_method: String,
    swiglu_limit: f32,
    rms_norm_eps: f32,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    num_nextn_predict_layers: usize,
}

#[derive(Deserialize)]
struct RawRopeScaling {
    #[serde(rename = "type")]
    kind: String,
    factor: f32,
    original_max_position_embeddings: usize,
}

impl RawDeepseek4Config {
    fn validate(self) -> Result<Deepseek4ModelConfig> {
        let expected_ratios = self
            .num_hidden_layers
            .checked_add(self.num_nextn_predict_layers)
            .ok_or_else(|| config_error("layer count overflow"))?;
        let scalars_valid = [
            self.compress_rope_theta,
            self.rope_theta,
            self.rope_scaling.factor,
            self.routed_scaling_factor,
            self.swiglu_limit,
            self.rms_norm_eps,
            self.hc_eps,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0);
        if self.model_type != MODEL_TYPE
            || !self
                .architectures
                .iter()
                .any(|architecture| architecture == ARCHITECTURE)
            || self.scoring_func != SCORING_FUNCTION
            || self.topk_method != TOPK_METHOD
            || self.rope_scaling.kind != ROPE_TYPE
            || !scalars_valid
        {
            return Err(config_error(format!(
                "unsupported identity or numerical policy: model_type={} architectures={:?} scoring={} topk={} rope={}",
                self.model_type,
                self.architectures,
                self.scoring_func,
                self.topk_method,
                self.rope_scaling.kind
            )));
        }
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads != 1
            || self.head_dim == 0
            || self.q_lora_rank == 0
            || self.qk_rope_head_dim == 0
            || self.qk_rope_head_dim > self.head_dim
            || !self.qk_rope_head_dim.is_multiple_of(2)
            || self.o_groups == 0
            || !self.num_attention_heads.is_multiple_of(self.o_groups)
            || self.o_lora_rank == 0
            || self.sliding_window == 0
            || self.max_position_embeddings == 0
            || self.rope_scaling.original_max_position_embeddings == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.index_topk == 0
            || self.n_routed_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.moe_intermediate_size == 0
            || self.n_shared_experts != 1
            || self.num_hash_layers > self.num_hidden_layers
            || self.hc_mult != HYPER_STREAMS
            || self.hc_sinkhorn_iters == 0
            || self.num_nextn_predict_layers > 1
            || self.compress_ratios.len() != expected_ratios
        {
            return Err(config_error(format!(
                "invalid dimensions: vocab={} hidden={} layers={} heads={} kv_heads={} head_dim={} top_k={} experts={} ratios={}/{}",
                self.vocab_size,
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.num_experts_per_tok,
                self.n_routed_experts,
                self.compress_ratios.len(),
                expected_ratios
            )));
        }

        let layer_attention = self
            .compress_ratios
            .iter()
            .enumerate()
            .map(|(layer, &ratio)| match ratio {
                0 => Ok(Deepseek4AttentionKind::Sliding),
                4 => Ok(Deepseek4AttentionKind::CompressedSparse),
                128 => Ok(Deepseek4AttentionKind::HeavilyCompressed),
                other => Err(config_error(format!(
                    "unsupported compression ratio {other} at layer {layer}"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Deepseek4ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            q_lora_rank: self.q_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,
            o_groups: self.o_groups,
            o_lora_rank: self.o_lora_rank,
            sliding_window: self.sliding_window,
            compress_rope_theta: self.compress_rope_theta,
            rope_theta: self.rope_theta,
            rope_factor: self.rope_scaling.factor,
            rope_original_max_positions: self.rope_scaling.original_max_position_embeddings,
            max_position_embeddings: self.max_position_embeddings,
            index_heads: self.index_n_heads,
            index_head_dim: self.index_head_dim,
            index_topk: self.index_topk,
            routed_experts: self.n_routed_experts,
            experts_per_token: self.num_experts_per_tok,
            expert_intermediate: self.moe_intermediate_size,
            shared_experts: self.n_shared_experts,
            hash_layers: self.num_hash_layers,
            routed_scaling_factor: self.routed_scaling_factor,
            swiglu_limit: self.swiglu_limit,
            rms_norm_eps: self.rms_norm_eps,
            hc_mult: self.hc_mult,
            hc_sinkhorn_iters: self.hc_sinkhorn_iters,
            hc_eps: self.hc_eps,
            nextn_predict_layers: self.num_nextn_predict_layers,
            layer_attention,
        })
    }
}

fn standard_rope_inv_freq(theta: f32, dim: usize) -> Vec<f32> {
    (0..dim / 2)
        .map(|index| 1.0 / theta.powf((2 * index) as f32 / dim as f32))
        .collect()
}

fn yarn_rope_inv_freq(
    theta: f32,
    dim: usize,
    factor: f32,
    original_max_positions: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    fn correction_dim(rotations: f32, dim: usize, theta: f32, positions: usize) -> f32 {
        dim as f32 * (positions as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * theta.ln())
    }

    let low = correction_dim(beta_fast, dim, theta, original_max_positions)
        .floor()
        .max(0.0);
    let high = correction_dim(beta_slow, dim, theta, original_max_positions)
        .ceil()
        .min((dim - 1) as f32);
    standard_rope_inv_freq(theta, dim)
        .into_iter()
        .enumerate()
        .map(|(index, extrapolated)| {
            let ramp = ((index as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            let extrapolation_factor = 1.0 - ramp;
            let interpolated = extrapolated / factor;
            interpolated * (1.0 - extrapolation_factor) + extrapolated * extrapolation_factor
        })
        .collect()
}

fn config_error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "DeepSeek V4 config",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Deepseek4AttentionKind, Deepseek4ModelConfig};

    const FLASH_CONFIG: &str = r#"{
        "architectures":["DeepseekV4ForCausalLM"],
        "model_type":"deepseek_v4",
        "vocab_size":129280,
        "hidden_size":4096,
        "num_hidden_layers":4,
        "num_attention_heads":64,
        "num_key_value_heads":1,
        "head_dim":512,
        "q_lora_rank":1024,
        "qk_rope_head_dim":64,
        "o_groups":8,
        "o_lora_rank":1024,
        "sliding_window":128,
        "compress_ratios":[0,0,4,128,0],
        "compress_rope_theta":160000,
        "rope_theta":10000,
        "rope_scaling":{"type":"yarn","factor":16,"original_max_position_embeddings":65536},
        "max_position_embeddings":1048576,
        "index_n_heads":64,
        "index_head_dim":128,
        "index_topk":512,
        "n_routed_experts":256,
        "num_experts_per_tok":6,
        "moe_intermediate_size":2048,
        "n_shared_experts":1,
        "num_hash_layers":3,
        "routed_scaling_factor":1.5,
        "scoring_func":"sqrtsoftplus",
        "topk_method":"noaux_tc",
        "swiglu_limit":10.0,
        "rms_norm_eps":1e-6,
        "hc_mult":4,
        "hc_sinkhorn_iters":20,
        "hc_eps":1e-6,
        "num_nextn_predict_layers":1
    }"#;

    #[test]
    fn validates_flash_layer_pattern() {
        let config =
            Deepseek4ModelConfig::from_json(FLASH_CONFIG.as_bytes()).expect("valid config");
        assert_eq!(
            config.attention_kind(0).expect("layer 0"),
            Deepseek4AttentionKind::Sliding
        );
        assert_eq!(
            config.attention_kind(2).expect("layer 2"),
            Deepseek4AttentionKind::CompressedSparse
        );
        assert_eq!(
            config.attention_kind(3).expect("layer 3"),
            Deepseek4AttentionKind::HeavilyCompressed
        );
        assert_eq!(config.compression_ratio(3).expect("ratio"), 128);
        assert_eq!(
            config.attention_kind(4).expect("MTP layer"),
            Deepseek4AttentionKind::Sliding
        );
    }

    #[test]
    fn rejects_unknown_compression_ratio() {
        let json = FLASH_CONFIG.replace("[0,0,4,128,0]", "[0,0,8,128,0]");
        let error = Deepseek4ModelConfig::from_json(json.as_bytes()).expect_err("invalid ratio");
        assert!(error.to_string().contains("compression ratio 8"));
    }

    #[test]
    fn rejects_missing_mtp_ratio() {
        let json = FLASH_CONFIG.replace("[0,0,4,128,0]", "[0,0,4,128]");
        let error = Deepseek4ModelConfig::from_json(json.as_bytes()).expect_err("missing ratio");
        assert!(error.to_string().contains("ratios=4/5"));
    }

    #[test]
    fn matches_deepseek_yarn_correction_band() {
        let config =
            Deepseek4ModelConfig::from_json(FLASH_CONFIG.as_bytes()).expect("valid config");
        let frequencies = config.compressed_rope_inv_freq();
        assert_eq!(frequencies.len(), 32);
        assert!((frequencies[0] - 1.0).abs() < 1.0e-7);
        assert!((frequencies[15] - 0.0036355386).abs() < 1.0e-8);
        assert!((frequencies[20] - 0.00029697778).abs() < 1.0e-8);
        assert!((frequencies[25] - 0.000005372313).abs() < 1.0e-9);
        assert!((frequencies[31] - 0.0000005680529).abs() < 1.0e-10);
    }

    #[test]
    fn rejects_non_flash_hyper_stream_count() {
        let json = FLASH_CONFIG.replace("\"hc_mult\":4", "\"hc_mult\":2");
        let error =
            Deepseek4ModelConfig::from_json(json.as_bytes()).expect_err("invalid stream count");
        assert!(error.to_string().contains("invalid dimensions"));
    }
}
