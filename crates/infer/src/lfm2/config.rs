//! Checkpoint dimensions and supported LFM2 MoE semantics.

use eider_cuda::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Lfm2Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_dense_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub use_expert_bias: bool,
    pub tie_word_embeddings: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_width: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParameters,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}

impl Lfm2Config {
    pub fn load(root: &Path) -> Result<Self> {
        let bytes = std::fs::read(root.join("config.json")).map_err(|e| invalid(e.to_string()))?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "lfm2_moe"
            || self.conv_bias
            || !self.tie_word_embeddings
            || self.rope_parameters.rope_type != "default"
        {
            return Err(invalid(
                "requires lfm2_moe, tied embeddings, bias-free convolution and default RoPE",
            ));
        }
        if [
            self.hidden_size,
            self.intermediate_size,
            self.moe_intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_experts,
            self.num_experts_per_tok,
            self.vocab_size,
            self.max_position_embeddings,
            self.conv_width,
        ]
        .contains(&0)
            || self.num_hidden_layers > 256
            || self.max_position_embeddings >= u32::MAX as usize
            || self.num_dense_layers > self.num_hidden_layers
            || self.num_experts > 512
            || self.num_experts_per_tok > self.num_experts
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !self.hidden_size.is_multiple_of(16)
            || !self.moe_intermediate_size.is_multiple_of(16)
            || !self.head_dim().is_multiple_of(2)
            || self.head_dim() > 256
            || self.layer_types.len() != self.num_hidden_layers
            || self
                .layer_types
                .iter()
                .any(|s| s != "conv" && s != "full_attention")
        {
            return Err(invalid("invalid layer schedule or model dimensions"));
        }
        for scalar in [
            self.norm_eps,
            self.routed_scaling_factor,
            self.rope_parameters.rope_theta,
        ] {
            if !scalar.is_finite() || scalar <= 0.0 {
                return Err(invalid(
                    "normalization, routed scale and RoPE base must be positive and finite",
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "LFM2 MoE",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Lfm2Config {
        Lfm2Config {
            model_type: "lfm2_moe".into(),
            hidden_size: 2048,
            intermediate_size: 11776,
            moe_intermediate_size: 1536,
            num_hidden_layers: 40,
            num_dense_layers: 2,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_experts: 64,
            num_experts_per_tok: 4,
            vocab_size: 128000,
            max_position_embeddings: 131072,
            norm_eps: 1e-5,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            use_expert_bias: true,
            tie_word_embeddings: true,
            conv_width: 3,
            conv_bias: false,
            layer_types: (0..40)
                .map(|i| if i % 4 == 2 { "full_attention" } else { "conv" }.into())
                .collect(),
            rope_parameters: RopeParameters {
                rope_theta: 1e7,
                rope_type: "default".into(),
            },
        }
    }

    #[test]
    fn lfm25_dimensions_and_schedule() {
        let c = config();
        c.validate().unwrap();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(
            c.layer_types
                .iter()
                .filter(|s| *s == "full_attention")
                .count(),
            10
        );
    }

    #[test]
    fn rejects_invalid_or_unsupported_semantics_without_panicking() {
        let mutations: &[fn(&mut Lfm2Config)] = &[
            |c| c.num_attention_heads = 0,
            |c| c.num_key_value_heads = 0,
            |c| c.conv_width = 0,
            |c| c.conv_bias = true,
            |c| c.layer_types.pop().map(|_| ()).unwrap(),
            |c| c.layer_types[0] = "mamba".into(),
            |c| c.num_experts_per_tok = 65,
            |c| c.num_dense_layers = 41,
            |c| c.moe_intermediate_size = 1537,
            |c| c.norm_eps = f32::NAN,
            |c| c.rope_parameters.rope_type = "yarn".into(),
            |c| c.tie_word_embeddings = false,
        ];
        for mutation in mutations {
            let mut c = config();
            mutation(&mut c);
            assert!(c.validate().is_err());
        }
    }
}
