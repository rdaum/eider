//! Inkling 2 Mini text configuration and physical-to-virtual layer schedule.

use eider_cuda::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Inkling2Config {
    pub model_type: String,
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    /// The DSpark draft stored in `mtp.safetensors`.
    pub mtp_config: Option<DraftConfig>,
}

/// DSpark draft geometry. Draft layers attend bidirectionally over a block of
/// one anchor and `block_size - 1` masks and over K/V derived from target
/// features captured at `target_layer_ids` (schedule visits, the last being
/// the final-norm output).
#[derive(Clone, Debug, Deserialize)]
pub struct DraftConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub model_max_length: usize,
    pub rms_norm_eps: f32,
    pub use_value_norm_no_gamma: bool,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub sliding_window_size: usize,
    pub n_layers_per_attn_res_block: usize,
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    pub use_gate_bias: bool,
    pub final_logit_softcapping: f32,
    pub loop_config: LoopConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub is_varloop: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Visit {
    pub physical_layer: usize,
    pub cache_layer: usize,
    pub pass: usize,
    pub local: bool,
}

impl Inkling2Config {
    pub fn load(root: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(root.join("config.json")).map_err(|error| invalid(error.to_string()))?;
        let config: Self =
            serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let c = &self.text_config;
        let l = &c.loop_config;
        let dimensions = self.model_type == "inkling2_loop"
            && c.hidden_size == 2304
            && c.num_hidden_layers == 24
            && c.vocab_size == 201024
            && c.unpadded_vocab_size == 200058
            && c.num_attention_heads == 24
            && c.num_key_value_heads == 8
            && c.head_dim == 128
            && (
                c.swa_head_dim,
                c.swa_num_attention_heads,
                c.swa_num_key_value_heads,
            ) == (c.head_dim, c.num_attention_heads, c.num_key_value_heads)
            && c.d_rel == 16
            && c.rel_extent == 1024
            && c.model_max_length == 1_048_576
            && c.n_routed_experts == 128
            && c.num_experts_per_tok == 8
            && c.n_shared_experts == 0
            && c.intermediate_size == 1152
            && c.sconv_kernel_size == 4
            && c.sliding_window_size == 1024
            && c.n_layers_per_attn_res_block == 6
            && c.use_value_norm_no_gamma
            && c.attn_res_normalize_values
            && c.use_gate_bias;
        let loop_valid = l.core_start == 6
            && l.core_end == 18
            && l.num_passes == 3
            && l.gate_rank == 64
            && !l.is_varloop;
        let scalars = [
            c.rms_norm_eps,
            c.qk_l2_clip_threshold,
            c.rel_l2_clip_threshold,
            c.log_scaling_alpha,
            c.mlp_gate_softcap,
            c.mlp_up_softcap,
            c.route_scale,
            c.final_logit_softcapping,
        ];
        if !dimensions
            || !loop_valid
            || c.log_scaling_n_floor == 0
            || scalars.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(invalid(
                "requires the fixed-pass Inkling 2 Mini text checkpoint with its released geometry",
            ));
        }
        let expected_local = (0..c.num_hidden_layers)
            .filter(|layer| layer % 6 != 5)
            .collect::<Vec<_>>();
        if c.local_layer_ids != expected_local {
            return Err(invalid("unsupported local/global attention schedule"));
        }
        Ok(())
    }

    /// Validates the DSpark draft geometry the draft kernels support.
    pub fn validate_draft(&self) -> Result<&DraftConfig> {
        let c = &self.text_config;
        let d = self
            .mtp_config
            .as_ref()
            .ok_or_else(|| invalid("the checkpoint has no DSpark draft configuration"))?;
        let visits = self.schedule().len();
        let taps = &d.target_layer_ids;
        if d.hidden_size != c.hidden_size
            || d.num_key_value_heads * d.head_dim != c.num_key_value_heads * c.head_dim
            || d.head_dim != 128
            || !d.num_attention_heads.is_multiple_of(d.num_key_value_heads)
            || d.num_attention_heads / d.num_key_value_heads > 3
            || d.block_size < 2
            || d.block_size > 8
            || d.markov_head_type != "rnn_hidden"
            || d.local_layer_ids
                .iter()
                .any(|&layer| layer >= d.num_hidden_layers)
            || taps.len() < 2
            || taps.windows(2).any(|pair| pair[0] >= pair[1])
            || taps.last() != Some(&visits)
            || !d.rms_norm_eps.is_finite()
            || d.rms_norm_eps <= 0.0
            || !d.rope_theta.is_finite()
            || d.rope_theta <= 0.0
        {
            return Err(invalid("unsupported DSpark draft geometry"));
        }
        Ok(d)
    }

    /// Maps the 24 stored layers onto 48 independent attention/convolution visits.
    pub fn schedule(&self) -> Vec<Visit> {
        let c = &self.text_config;
        let l = &c.loop_config;
        (0..l.num_passes + 2)
            .flat_map(|part| {
                let (range, pass) = match part {
                    0 => (0..l.core_start, 0),
                    p if p <= l.num_passes => (l.core_start..l.core_end, p - 1),
                    _ => (l.core_end..c.num_hidden_layers, 0),
                };
                range.map(move |physical_layer| Visit {
                    physical_layer,
                    cache_layer: if pass == 0 {
                        physical_layer
                    } else {
                        c.num_hidden_layers
                            + (pass - 1) * (l.core_end - l.core_start)
                            + physical_layer
                            - l.core_start
                    },
                    pass,
                    local: c.local_layer_ids.contains(&physical_layer),
                })
            })
            .collect()
    }
}

pub(super) fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Inkling2",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_schedule_has_independent_loop_state() {
        let c: Inkling2Config = serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
        c.validate().unwrap();
        let schedule = c.schedule();
        assert_eq!(schedule.len(), 48);
        assert_eq!(
            schedule
                .iter()
                .map(|v| v.physical_layer)
                .collect::<Vec<_>>(),
            (0..6)
                .chain((0..3).flat_map(|_| 6..18))
                .chain(18..24)
                .collect::<Vec<_>>()
        );
        assert_eq!(schedule[6].cache_layer, 6);
        assert_eq!(schedule[18].cache_layer, 24);
        assert_eq!(schedule[30].cache_layer, 36);
        assert_eq!(schedule[42].cache_layer, 18);
        assert!(schedule[18].local);
        assert!(!schedule[17].local);
        assert_eq!(schedule.iter().filter(|v| v.local).count(), 40);
        assert_eq!(
            schedule
                .iter()
                .map(|v| v.cache_layer)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            48
        );
    }

    #[test]
    fn variable_loop_and_wrong_layout_are_rejected() {
        let mut c: Inkling2Config =
            serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
        c.text_config.loop_config.is_varloop = true;
        assert!(c.validate().is_err());
        let mut c: Inkling2Config =
            serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
        c.text_config.local_layer_ids.swap(0, 5);
        assert!(c.validate().is_err());
    }
}
