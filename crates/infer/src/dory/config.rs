//! Dory loop schedule and checkpoint semantics.
use eider_cuda::{Error, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};
#[derive(Clone, Debug, Deserialize)]
pub struct DoryConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub n_input_layers: usize,
    pub n_recurrent_layers: usize,
    pub n_output_layers: usize,
    pub n_recurrent_loops: usize,
    pub layer_types: Vec<String>,
    pub sliding_window: usize,
    pub initializer_range: f32,
    pub ngpt_alpha_init: f32,
    pub attention_output_gate_logit_scale_init: Option<f32>,
    pub rope_parameters: BTreeMap<String, Rope>,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub hidden_act: String,
    #[serde(default)]
    pub recurrent_kv_cache_mode: Option<String>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Rope {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}
impl DoryConfig {
    pub fn load(root: &Path, loops: Option<usize>) -> Result<Self> {
        let mut c: Self = serde_json::from_slice(
            &std::fs::read(root.join("config.json")).map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        if let Some(loops) = loops {
            c.n_recurrent_loops = loops;
        }
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "dory"
            || self.tie_word_embeddings
            || self.attention_bias
            || self.mlp_bias
            || self.hidden_act != "silu"
            || self
                .recurrent_kv_cache_mode
                .as_deref()
                .is_some_and(|m| m != "per_loop")
        {
            return Err(invalid(
                "requires bias-free Dory SwiGLU with separate head and per-loop KV",
            ));
        }
        if self.head_dim != 256
            || self.num_key_value_heads == 0
            || self.num_attention_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.num_attention_heads > 32
            || !(1..=4).contains(&self.n_recurrent_loops)
            || self.sliding_window == 0
            || self.max_position_embeddings == 0
            || self.max_position_embeddings > 262144
            || self.vocab_size == 0
            || self.vocab_size > u32::MAX as usize
            || self.n_recurrent_layers == 0
            || self
                .n_input_layers
                .checked_add(self.n_recurrent_layers)
                .and_then(|n| n.checked_add(self.n_output_layers))
                != Some(self.num_hidden_layers)
            || self.layer_types.len() != self.num_hidden_layers
            || [self.hidden_size, self.intermediate_size]
                .iter()
                .any(|&n| n == 0 || n > 8192 || !n.is_multiple_of(64))
            || !self.initializer_range.is_finite()
            || self.initializer_range <= 0.0
            || !self.ngpt_alpha_init.is_finite()
            || self.ngpt_alpha_init <= 0.0
            || self
                .attention_output_gate_logit_scale_init
                .is_some_and(|x| !x.is_finite() || x <= 0.0)
        {
            return Err(invalid(
                "unsupported dimensions, loop count, or normalization scales",
            ));
        }
        for kind in &self.layer_types {
            if kind == "mlp" {
                continue;
            }
            if !matches!(kind.as_str(), "full_attention" | "sliding_attention") {
                return Err(invalid("unknown layer type"));
            }
            let rope = self
                .rope_parameters
                .get(kind)
                .ok_or_else(|| invalid("missing RoPE parameters"))?;
            if rope.rope_type != "default"
                || rope.partial_rotary_factor != 1.0
                || !rope.rope_theta.is_finite()
                || rope.rope_theta <= 0.0
            {
                return Err(invalid("requires full-head default RoPE"));
            }
        }
        Ok(())
    }
    /// Physical weight index for each executed sublayer. Repeated weights share no KV.
    pub fn schedule(&self) -> Vec<usize> {
        (0..self.n_input_layers)
            .chain(
                (0..self.n_recurrent_loops).flat_map(|_| {
                    self.n_input_layers..self.n_input_layers + self.n_recurrent_layers
                }),
            )
            .chain(self.n_input_layers + self.n_recurrent_layers..self.num_hidden_layers)
            .collect()
    }
}
pub(super) fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Dory",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> DoryConfig {
        serde_json::from_str(include_str!("config-fixture.json")).unwrap()
    }
    #[test]
    fn dory_schedule_repeats_only_the_recurrent_weights() {
        let mut c = fixture();
        c.validate().unwrap();
        assert_eq!(c.schedule().len(), 112);
        assert_eq!(&c.schedule()[16..56], &c.schedule()[56..96]);
        assert_eq!(&c.schedule()[96..], &(56..72).collect::<Vec<_>>());
        assert_eq!(
            c.schedule()
                .iter()
                .filter(|&&i| c.layer_types[i] != "mlp")
                .count(),
            56
        );
        c.n_recurrent_loops = 1;
        assert_eq!(c.schedule(), (0..72).collect::<Vec<_>>());
        c.n_recurrent_loops = 4;
        c.validate().unwrap();
        assert_eq!(c.schedule().len(), 192);
    }
    #[test]
    fn dory_rejects_unsupported_checkpoint_semantics() {
        for count in [0, 5] {
            let mut c = fixture();
            c.n_recurrent_loops = count;
            assert!(c.validate().is_err());
        }
        let mut c = fixture();
        c.recurrent_kv_cache_mode = Some("last_loop".into());
        assert!(c.validate().is_err());
        let mut c = fixture();
        c.n_output_layers += 1;
        assert!(c.validate().is_err());
        let mut c = fixture();
        c.layer_types[0] = "unknown".into();
        assert!(c.validate().is_err());
        let mut c = fixture();
        c.initializer_range = 0.;
        assert!(c.validate().is_err());
        let mut c = fixture();
        c.attention_bias = true;
        assert!(c.validate().is_err());
    }
}
