//! LFM3 checkpoint dimensions and layer-specific rotary frequencies.

use eider_cuda::{Error, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct Lfm3Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    pub tie_word_embeddings: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_width: usize,
    pub conv_bias: bool,
    pub use_sandwich_norm: bool,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: BTreeMap<String, RopeParameters>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
    #[serde(default = "one")]
    pub partial_rotary_factor: f32,
}

fn one() -> f32 {
    1.0
}

impl Lfm3Config {
    pub fn load(root: &Path) -> Result<Self> {
        let bytes = std::fs::read(root.join("config.json")).map_err(|e| invalid(e.to_string()))?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "lfm3" || self.conv_bias || !self.tie_word_embeddings {
            return Err(invalid(
                "requires lfm3, tied embeddings and bias-free convolution",
            ));
        }
        if [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
            self.sliding_window,
        ]
        .contains(&0)
            || self.conv_width < 2
            || self.conv_width > 32
            || self.num_hidden_layers > 256
            || self.head_dim > 256
            || !self.head_dim.is_multiple_of(2)
            || !self.hidden_size.is_multiple_of(64)
            || !self.intermediate_size.is_multiple_of(64)
            || self.num_attention_heads.checked_mul(self.head_dim) != Some(self.hidden_size)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.max_position_embeddings >= u32::MAX as usize
            || self.layer_types.len() != self.num_hidden_layers
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
        {
            return Err(invalid("invalid model dimensions or normalization epsilon"));
        }
        for kind in &self.layer_types {
            if kind == "conv" {
                continue;
            }
            if kind != "full_attention" && kind != "sliding_attention" {
                return Err(invalid(format!("unsupported layer type {kind}")));
            }
            let r = self
                .rope_parameters
                .get(kind)
                .ok_or_else(|| invalid(format!("missing RoPE for {kind}")))?;
            if r.rope_type != "default"
                || !r.rope_theta.is_finite()
                || r.rope_theta <= 0.0
                || !r.partial_rotary_factor.is_finite()
                || r.partial_rotary_factor <= 0.0
                || r.partial_rotary_factor > 1.0
                || (r.partial_rotary_factor * self.head_dim as f32) as usize / 2 == 0
            {
                return Err(invalid(
                    "requires default RoPE with a positive base and partial factor in (0, 1]",
                ));
            }
        }
        Ok(())
    }

    /// LFM3 pairs across the full head and zeroes unused frequencies.
    pub fn inverse_frequencies(&self, kind: &str) -> Vec<f32> {
        let r = &self.rope_parameters[kind];
        let angles = (r.partial_rotary_factor * self.head_dim as f32) as usize / 2;
        (0..self.head_dim / 2)
            .map(|i| {
                if i < angles {
                    r.rope_theta.powf(-((2 * i) as f32) / self.head_dim as f32)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

pub(super) fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "LFM3",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_schedule_and_zero_tail_rope() {
        let c: Lfm3Config = serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
        c.validate().unwrap();
        assert_eq!(c.layer_types.iter().filter(|s| *s == "conv").count(), 13);
        assert_eq!(
            c.layer_types
                .iter()
                .filter(|s| *s == "full_attention")
                .count(),
            6
        );
        let freq = c.inverse_frequencies("full_attention");
        assert_eq!(freq.len(), 64);
        assert!((freq[31] - 1e6_f32.powf(-62.0 / 128.0)).abs() < 1e-7);
        assert!(freq[32..].iter().all(|&x| x == 0.0));
        assert!(
            c.inverse_frequencies("sliding_attention")
                .iter()
                .all(|&x| x > 0.0)
        );
        for change in [
            (|c: &mut Lfm3Config| c.num_attention_heads = 0) as fn(&mut Lfm3Config),
            |c| c.num_key_value_heads = 0,
            |c| c.head_dim = 64,
            |c| c.conv_width = 0,
            |c| c.layer_types[0] = "unknown".into(),
            |c| {
                c.rope_parameters
                    .get_mut("full_attention")
                    .unwrap()
                    .partial_rotary_factor = f32::NAN
            },
            |c| {
                c.rope_parameters
                    .remove("sliding_attention")
                    .map(|_| ())
                    .unwrap()
            },
        ] {
            let mut bad = c.clone();
            change(&mut bad);
            assert!(bad.validate().is_err());
        }
    }
}
