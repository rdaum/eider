//! NVIDIA Nemotron 3 hybrid-model configuration and checkpoint topology.

use nvfp4::{Error, ModelOptCheckpoint, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

mod attention;
mod linear;
mod mamba;
mod model;
mod moe;
mod router;

pub use attention::{Nemotron3AttentionLayer, Nemotron3AttentionWorkspace};
pub use linear::{Nemotron3Bf16Storage, Nemotron3Fp8Storage, Nemotron3StorageConfig};
pub use mamba::{Nemotron3MambaLayer, Nemotron3MambaState, Nemotron3MambaWorkspace};
pub use model::{Nemotron3DecodeState, Nemotron3Model};
pub use moe::{Nemotron3MoeLayer, Nemotron3MoeWorkspace};
pub use router::{Nemotron3Router, Nemotron3RouterWorkspace};

/// Mixer used by one Nemotron 3 backbone layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nemotron3LayerKind {
    /// Mamba-2 selective state-space layer.
    Mamba,
    /// Sparse latent mixture-of-experts layer.
    Moe,
    /// Grouped-query causal-attention layer.
    Attention,
}

impl Nemotron3LayerKind {
    fn from_name(name: &str) -> Result<Self> {
        match name {
            "mamba" => Ok(Self::Mamba),
            "moe" => Ok(Self::Moe),
            "attention" => Ok(Self::Attention),
            other => Err(Error::Format {
                label: "Nemotron 3 layer type",
                detail: format!("unsupported layer type {other:?}"),
            }),
        }
    }

    fn from_pattern(character: char) -> Result<Self> {
        match character {
            'M' => Ok(Self::Mamba),
            'E' => Ok(Self::Moe),
            '*' => Ok(Self::Attention),
            other => Err(Error::Format {
                label: "Nemotron 3 hybrid pattern",
                detail: format!("unsupported layer marker {other:?}"),
            }),
        }
    }

    /// Returns the checkpoint/config spelling for this layer type.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mamba => "mamba",
            Self::Moe => "moe",
            Self::Attention => "attention",
        }
    }
}

/// Validated architecture description shared by Nemotron 3 checkpoints.
#[derive(Clone, Debug, PartialEq)]
pub struct Nemotron3Manifest {
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Backbone hidden width.
    pub hidden_size: usize,
    /// Ordered backbone mixer types.
    pub layers: Vec<Nemotron3LayerKind>,
    /// Attention query-head count.
    pub attention_heads: usize,
    /// Attention key/value-head count.
    pub kv_heads: usize,
    /// Attention head width.
    pub attention_head_dim: usize,
    /// Maximum configured context length.
    pub max_position_embeddings: usize,
    /// Mamba recurrent head count.
    pub mamba_heads: usize,
    /// Width of one Mamba head.
    pub mamba_head_dim: usize,
    /// Mamba B/C group count.
    pub mamba_groups: usize,
    /// Mamba recurrent state width.
    pub mamba_state_size: usize,
    /// Mamba depthwise convolution width.
    pub mamba_conv_kernel: usize,
    /// Mamba scan chunk size.
    pub mamba_chunk_size: usize,
    /// Routed-expert count in each MoE layer.
    pub routed_experts: usize,
    /// Experts selected per token.
    pub experts_per_token: usize,
    /// Routed expert hidden width.
    pub moe_intermediate_size: usize,
    /// Latent expert input/output width, when enabled.
    pub moe_latent_size: Option<usize>,
    /// Shared expert hidden width.
    pub shared_expert_intermediate_size: usize,
    /// Router output multiplier.
    pub routed_scaling_factor: f32,
    /// Number of expert groups considered by the router.
    pub expert_groups: usize,
    /// Expert groups retained before expert-level top-k selection.
    pub topk_groups: usize,
    /// Whether selected sigmoid probabilities are normalized.
    pub normalize_topk_probabilities: bool,
    /// Number of MTP prediction blocks.
    pub mtp_prediction_layers: usize,
    /// Ordered MTP mixer types.
    pub mtp_layers: Vec<Nemotron3LayerKind>,
    /// RMS-normalization epsilon.
    pub norm_epsilon: f32,
}

#[derive(Debug, Deserialize)]
struct Nemotron3Config {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    #[serde(default)]
    layers_block_type: Option<Vec<String>>,
    #[serde(default)]
    hybrid_override_pattern: Option<String>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    #[serde(alias = "mamba_n_groups")]
    n_groups: usize,
    #[serde(alias = "mamba_d_state")]
    ssm_state_size: usize,
    #[serde(alias = "mamba_d_conv")]
    conv_kernel: usize,
    #[serde(alias = "mamba_chunk_size")]
    chunk_size: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    #[serde(default)]
    moe_latent_size: Option<usize>,
    moe_shared_expert_intermediate_size: usize,
    routed_scaling_factor: f32,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    #[serde(default)]
    num_nextn_predict_layers: usize,
    #[serde(default)]
    mtp_layers_block_type: Option<Vec<String>>,
    #[serde(default)]
    mtp_hybrid_override_pattern: Option<String>,
    #[serde(default)]
    layer_norm_epsilon: Option<f32>,
    #[serde(default)]
    norm_eps: Option<f32>,
}

impl Nemotron3Manifest {
    /// Reads and validates `config.json` from a Hugging Face checkpoint.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let contents = fs::read_to_string(&path).map_err(|error| Error::Format {
            label: "Nemotron 3 config",
            detail: format!("{}: {error}", path.display()),
        })?;
        Self::from_config_str(&contents)
    }

    /// Parses and validates a Nemotron 3 Hugging Face configuration.
    pub fn from_config_str(contents: &str) -> Result<Self> {
        let config: Nemotron3Config =
            serde_json::from_str(contents).map_err(|error| Error::Format {
                label: "Nemotron 3 config json",
                detail: error.to_string(),
            })?;
        if config.model_type != "nemotron_h" {
            return Err(Error::Format {
                label: "Nemotron 3 architecture",
                detail: format!(
                    "expected model_type nemotron_h, got {:?}",
                    config.model_type
                ),
            });
        }

        let layers = parse_layers(
            config.layers_block_type.as_deref(),
            config.hybrid_override_pattern.as_deref(),
            "Nemotron 3 backbone",
        )?;
        if let Some(expected) = config.num_hidden_layers
            && layers.len() != expected
        {
            return Err(Error::Shape {
                label: "Nemotron 3 backbone layers",
                expected: expected.to_string(),
                actual: layers.len().to_string(),
            });
        }

        let mtp_layers = if config.num_nextn_predict_layers == 0 {
            Vec::new()
        } else {
            parse_layers(
                config.mtp_layers_block_type.as_deref(),
                config.mtp_hybrid_override_pattern.as_deref(),
                "Nemotron 3 MTP",
            )?
        };

        let norm_epsilon = compatible_value(
            config.layer_norm_epsilon,
            config.norm_eps,
            "Nemotron 3 norm epsilon",
        )?;
        let manifest = Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            layers,
            attention_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            attention_head_dim: config.head_dim,
            max_position_embeddings: config.max_position_embeddings,
            mamba_heads: config.mamba_num_heads,
            mamba_head_dim: config.mamba_head_dim,
            mamba_groups: config.n_groups,
            mamba_state_size: config.ssm_state_size,
            mamba_conv_kernel: config.conv_kernel,
            mamba_chunk_size: config.chunk_size,
            routed_experts: config.n_routed_experts,
            experts_per_token: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            moe_latent_size: config.moe_latent_size,
            shared_expert_intermediate_size: config.moe_shared_expert_intermediate_size,
            routed_scaling_factor: config.routed_scaling_factor,
            expert_groups: config.n_group,
            topk_groups: config.topk_group,
            normalize_topk_probabilities: config.norm_topk_prob,
            mtp_prediction_layers: config.num_nextn_predict_layers,
            mtp_layers,
            norm_epsilon,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Returns the Mamba projected-state width before the output projection.
    pub fn mamba_intermediate_size(&self) -> usize {
        self.mamba_heads * self.mamba_head_dim
    }

    /// Returns the depthwise convolution channel count in each Mamba layer.
    pub fn mamba_conv_channels(&self) -> usize {
        self.mamba_intermediate_size() + 2 * self.mamba_groups * self.mamba_state_size
    }

    /// Returns the output width of a Mamba input projection.
    pub fn mamba_projection_size(&self) -> usize {
        self.mamba_intermediate_size() + self.mamba_conv_channels() + self.mamba_heads
    }

    /// Returns per-sequence recurrent-state bytes when stored as FP32.
    pub fn mamba_state_bytes_fp32(&self) -> usize {
        let layer_state = self.mamba_heads * self.mamba_head_dim * self.mamba_state_size;
        let convolution = self.mamba_conv_channels() * self.mamba_conv_kernel;
        let layers = self
            .layers
            .iter()
            .filter(|&&kind| kind == Nemotron3LayerKind::Mamba)
            .count();
        layers * (layer_state + convolution) * size_of::<f32>()
    }

    /// Checks that the checkpoint index contains every tensor required by the backbone.
    ///
    /// Quantization companions are deliberately not required here because Nemotron 3
    /// is published in BF16, FP8, and NVFP4 variants with the same model topology.
    pub fn validate_checkpoint_index(&self, checkpoint: &ModelOptCheckpoint) -> Result<()> {
        for tensor in [
            "backbone.embeddings.weight",
            "backbone.norm_f.weight",
            "lm_head.weight",
        ] {
            require_tensor(checkpoint, tensor)?;
        }
        for (layer, kind) in self.layers.iter().copied().enumerate() {
            require_tensor(checkpoint, &format!("backbone.layers.{layer}.norm.weight"))?;
            match kind {
                Nemotron3LayerKind::Mamba => self.validate_mamba_layer(checkpoint, layer)?,
                Nemotron3LayerKind::Attention => {
                    self.validate_attention_layer(checkpoint, layer)?
                }
                Nemotron3LayerKind::Moe => self.validate_moe_layer(checkpoint, layer)?,
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("attention_heads", self.attention_heads),
            ("kv_heads", self.kv_heads),
            ("attention_head_dim", self.attention_head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("mamba_heads", self.mamba_heads),
            ("mamba_head_dim", self.mamba_head_dim),
            ("mamba_groups", self.mamba_groups),
            ("mamba_state_size", self.mamba_state_size),
            ("mamba_conv_kernel", self.mamba_conv_kernel),
            ("mamba_chunk_size", self.mamba_chunk_size),
            ("routed_experts", self.routed_experts),
            ("experts_per_token", self.experts_per_token),
            ("expert_groups", self.expert_groups),
            ("topk_groups", self.topk_groups),
            ("moe_intermediate_size", self.moe_intermediate_size),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
        ] {
            if value == 0 {
                return Err(Error::Shape {
                    label: "Nemotron 3 config dimension",
                    expected: format!("non-zero {name}"),
                    actual: "0".to_string(),
                });
            }
        }
        if self.layers.is_empty() {
            return Err(Error::Shape {
                label: "Nemotron 3 backbone layers",
                expected: "at least one layer".to_string(),
                actual: "0".to_string(),
            });
        }
        if !self.attention_heads.is_multiple_of(self.kv_heads) {
            return Err(Error::Shape {
                label: "Nemotron 3 attention heads",
                expected: "query heads divisible by key/value heads".to_string(),
                actual: format!("{} / {}", self.attention_heads, self.kv_heads),
            });
        }
        if !self.mamba_heads.is_multiple_of(self.mamba_groups) {
            return Err(Error::Shape {
                label: "Nemotron 3 Mamba groups",
                expected: "Mamba heads divisible by groups".to_string(),
                actual: format!("{} / {}", self.mamba_heads, self.mamba_groups),
            });
        }
        if self.experts_per_token > self.routed_experts {
            return Err(Error::Shape {
                label: "Nemotron 3 routed experts",
                expected: format!("top-k <= {}", self.routed_experts),
                actual: self.experts_per_token.to_string(),
            });
        }
        if !self.routed_experts.is_multiple_of(self.expert_groups)
            || self.topk_groups > self.expert_groups
        {
            return Err(Error::Shape {
                label: "Nemotron 3 expert groups",
                expected: "experts divisible by groups and topk_groups <= groups".to_string(),
                actual: format!(
                    "experts={} groups={} topk_groups={}",
                    self.routed_experts, self.expert_groups, self.topk_groups
                ),
            });
        }
        if self.mtp_prediction_layers != 0 && self.mtp_layers.is_empty() {
            return Err(Error::Shape {
                label: "Nemotron 3 MTP layers",
                expected: "non-empty MTP layer pattern".to_string(),
                actual: "empty".to_string(),
            });
        }
        if !self.norm_epsilon.is_finite() || self.norm_epsilon <= 0.0 {
            return Err(Error::Format {
                label: "Nemotron 3 norm epsilon",
                detail: format!("expected positive finite value, got {}", self.norm_epsilon),
            });
        }
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            return Err(Error::Format {
                label: "Nemotron 3 routed scaling factor",
                detail: format!(
                    "expected positive finite value, got {}",
                    self.routed_scaling_factor
                ),
            });
        }
        Ok(())
    }

    fn validate_mamba_layer(&self, checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<()> {
        let prefix = format!("backbone.layers.{layer}.mixer");
        for suffix in [
            "A_log",
            "D",
            "conv1d.bias",
            "conv1d.weight",
            "dt_bias",
            "in_proj.weight",
            "norm.weight",
            "out_proj.weight",
        ] {
            require_tensor(checkpoint, &format!("{prefix}.{suffix}"))?;
        }
        Ok(())
    }

    fn validate_attention_layer(
        &self,
        checkpoint: &ModelOptCheckpoint,
        layer: usize,
    ) -> Result<()> {
        let prefix = format!("backbone.layers.{layer}.mixer");
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            require_tensor(checkpoint, &format!("{prefix}.{projection}.weight"))?;
        }
        Ok(())
    }

    fn validate_moe_layer(&self, checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<()> {
        let prefix = format!("backbone.layers.{layer}.mixer");
        for suffix in [
            "gate.weight",
            "gate.e_score_correction_bias",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        ] {
            require_tensor(checkpoint, &format!("{prefix}.{suffix}"))?;
        }
        if self.moe_latent_size.is_some() {
            require_tensor(checkpoint, &format!("{prefix}.fc1_latent_proj.weight"))?;
            require_tensor(checkpoint, &format!("{prefix}.fc2_latent_proj.weight"))?;
        }
        for expert in 0..self.routed_experts {
            require_tensor(
                checkpoint,
                &format!("{prefix}.experts.{expert}.up_proj.weight"),
            )?;
            require_tensor(
                checkpoint,
                &format!("{prefix}.experts.{expert}.down_proj.weight"),
            )?;
        }
        Ok(())
    }
}

fn parse_layers(
    explicit: Option<&[String]>,
    pattern: Option<&str>,
    label: &'static str,
) -> Result<Vec<Nemotron3LayerKind>> {
    if let Some(explicit) = explicit {
        return explicit
            .iter()
            .map(|name| Nemotron3LayerKind::from_name(name))
            .collect();
    }
    let pattern = pattern.ok_or_else(|| Error::Format {
        label,
        detail: "missing layers_block_type and hybrid_override_pattern".to_string(),
    })?;
    pattern
        .chars()
        .map(Nemotron3LayerKind::from_pattern)
        .collect()
}

fn require_tensor(checkpoint: &ModelOptCheckpoint, tensor: &str) -> Result<()> {
    if checkpoint.contains_tensor(tensor) {
        return Ok(());
    }
    Err(Error::Format {
        label: "Nemotron 3 checkpoint",
        detail: format!("missing tensor {tensor}"),
    })
}

fn compatible_value(current: Option<f32>, legacy: Option<f32>, label: &'static str) -> Result<f32> {
    match (current, legacy) {
        (Some(current), Some(legacy)) if current != legacy => Err(Error::Format {
            label,
            detail: format!("conflicting values {current} and {legacy}"),
        }),
        (Some(value), _) | (_, Some(value)) => Ok(value),
        (None, None) => Err(Error::Format {
            label,
            detail: "missing value".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{Nemotron3LayerKind, Nemotron3Manifest};

    const SUPER_CONFIG: &str = r#"{
        "model_type": "nemotron_h",
        "vocab_size": 131072,
        "hidden_size": 4096,
        "hybrid_override_pattern": "MEM*E",
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 262144,
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "ssm_state_size": 128,
        "conv_kernel": 4,
        "chunk_size": 128,
        "n_routed_experts": 512,
        "num_experts_per_tok": 22,
        "moe_intermediate_size": 2688,
        "moe_latent_size": 1024,
        "moe_shared_expert_intermediate_size": 5376,
        "routed_scaling_factor": 5.0,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": true,
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*E",
        "layer_norm_epsilon": 0.00001
    }"#;

    #[test]
    fn parses_super_hybrid_configuration() {
        let manifest = Nemotron3Manifest::from_config_str(SUPER_CONFIG).expect("manifest");
        assert_eq!(
            manifest.layers,
            [
                Nemotron3LayerKind::Mamba,
                Nemotron3LayerKind::Moe,
                Nemotron3LayerKind::Mamba,
                Nemotron3LayerKind::Attention,
                Nemotron3LayerKind::Moe,
            ]
        );
        assert_eq!(manifest.mtp_layers.len(), 2);
        assert_eq!(manifest.mamba_intermediate_size(), 8192);
        assert_eq!(manifest.mamba_conv_channels(), 10_240);
        assert_eq!(manifest.mamba_projection_size(), 18_560);
    }

    #[test]
    fn explicit_layer_list_takes_precedence_over_legacy_pattern() {
        let config = SUPER_CONFIG.replace(
            "\"hybrid_override_pattern\": \"MEM*E\"",
            "\"layers_block_type\": [\"attention\", \"moe\", \"mamba\", \"moe\", \"mamba\"], \"hybrid_override_pattern\": \"MEM*E\"",
        );
        let manifest = Nemotron3Manifest::from_config_str(&config).expect("manifest");
        assert_eq!(manifest.layers[0], Nemotron3LayerKind::Attention);
    }

    #[test]
    fn rejects_invalid_expert_top_k() {
        let config = SUPER_CONFIG.replace(
            "\"num_experts_per_tok\": 22",
            "\"num_experts_per_tok\": 513",
        );
        let error = Nemotron3Manifest::from_config_str(&config).expect_err("invalid top-k");
        assert!(error.to_string().contains("top-k <= 512"));
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let config = SUPER_CONFIG.replace("\"num_hidden_layers\": 5", "\"num_hidden_layers\": 6");
        let error = Nemotron3Manifest::from_config_str(&config).expect_err("layer mismatch");
        assert!(error.to_string().contains("backbone layers"));
    }
}
