//! Released W4A16 companions and their target feature and candidate-head contracts.

use crate::nemotron3::Nemotron3Manifest;
use eider_cuda::{Error, Result};
use eider_format::ModelOptCheckpoint;
use serde::Deserialize;
use std::{fs, path::Path};

/// Geometry of the supported Nemotron 3.5 LiLiCorr companion.
#[derive(Clone, Debug, Deserialize)]
pub struct LiLiCorrConfig {
    architectures: Vec<String>,
    model_type: String,
    /// Anchor plus speculative positions.
    pub block_size: usize,
    pub(crate) dflash_config: DraftConfig,
    /// Width of both the target residual stream and the draft backbone.
    pub hidden_size: usize,
    /// Number of draft transformer layers.
    pub num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) intermediate_size: usize,
    hidden_act: String,
    pub(crate) rms_norm_eps: f32,
    /// Full target vocabulary.
    pub vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    attention_bias: bool,
    attention_dropout: f32,
    pub(crate) rope_theta: f32,
    rope_scaling: Option<serde_json::Value>,
    tie_word_embeddings: bool,
    torch_dtype: String,
    num_target_layers: usize,
    layer_types: Vec<String>,
    pub(crate) sliding_window: usize,
    use_sliding_window: bool,
    #[serde(default)]
    pub(crate) has_own_lm_head: bool,
    draft_vocab_size: Option<usize>,
    quantization_config: Quantization,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DraftConfig {
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    causal: bool,
    #[serde(default)]
    pub attention_sink_bias: bool,
    projector_type: String,
    lilicorr_enabled: bool,
    pub lilicorr_candidate_topk: usize,
    pub lilicorr_hidden_size: usize,
    pub lilicorr_num_layers: usize,
    pub lilicorr_num_heads: usize,
    lilicorr_mlp_ratio: f32,
    pub lilicorr_factor_dim: usize,
    pub lilicorr_vector_eps: f32,
    pub lilicorr_logit_scale: f32,
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
    use_swa: bool,
    swa_window_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct Quantization {
    quant_algo: String,
    quant_method: String,
    group_size: usize,
    kv_cache_quant_algo: Option<String>,
}

impl LiLiCorrConfig {
    /// Target residual streams used to project the draft context.
    pub fn target_layers(&self) -> &[usize] {
        &self.dflash_config.target_layer_ids
    }

    /// Largest proposal supported by this checkpoint.
    pub const fn draft_tokens(&self) -> usize {
        self.block_size - 1
    }

    /// Checks the shared vocabulary, embedding width, and target layer count.
    pub fn validate_target(&self, target: &Nemotron3Manifest) -> Result<()> {
        if target.hidden_size != self.hidden_size
            || target.vocab_size != self.vocab_size
            || target.layers.len() != 88
            || self
                .target_layers()
                .iter()
                .any(|&layer| layer >= target.layers.len())
        {
            return Err(invalid(format!(
                "target must have hidden={}, vocab={}, layers={}; got {}/{}/{}",
                self.hidden_size,
                self.vocab_size,
                88,
                target.hidden_size,
                target.vocab_size,
                target.layers.len(),
            )));
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        let d = &self.dflash_config;
        let q = &self.quantization_config;
        if self.architectures != ["LiLiCorrDraftModel"]
            || self.model_type != "qwen3"
            || self.torch_dtype != "bfloat16"
            || self.hidden_act != "silu"
            || self.attention_bias
            || self.attention_dropout != 0.0
            || self.rope_scaling.is_some()
            || self.tie_word_embeddings
            || self.draft_vocab_size.is_some_and(|v| v != self.vocab_size)
            || d.causal
            || !d.lilicorr_enabled
            || d.projector_type != "lilicorr"
            || q.quant_algo != "W4A16_NVFP4"
            || q.quant_method != "modelopt_fp4"
            || q.group_size != 16
            || q.kv_cache_quant_algo.is_some()
        {
            return Err(invalid(
                "requires a non-causal LiLiCorrDraftModel W4A16_NVFP4 checkpoint",
            ));
        }
        // The Bolt release leaves the target layer count at zero and shares its
        // target LM head. Keep both released profiles explicit.
        let bolt = d.target_layer_ids == [1, 18, 35, 51, 68, 85];
        let expected_window = if bolt { 1024 } else { 8192 };
        if self.has_own_lm_head == bolt
            || d.attention_sink_bias != bolt
            || self.num_target_layers != if bolt { 0 } else { 88 }
            || (!bolt && d.target_layer_ids != [1, 22, 43, 64, 85])
        {
            return Err(invalid("unsupported Nemotron 3.5 LiLiCorr target profile"));
        }
        if self.block_size != 8
            || self.hidden_size != 4096
            || self.vocab_size != 131072
            || self.num_hidden_layers != 5
            || self.num_attention_heads != 32
            || self.num_key_value_heads != if bolt { 2 } else { 8 }
            || self.head_dim != 128
            || self.intermediate_size != 12288
            || d.lilicorr_candidate_topk != 8
            || d.lilicorr_hidden_size != 1024
            || d.lilicorr_num_layers != 2
            || d.lilicorr_num_heads != 8
            || d.lilicorr_mlp_ratio != 2.0
            || d.lilicorr_factor_dim != 1024
            || d.conv_kernel_size != 2
            || d.conv_group_size != 16
            || d.mask_token_id as usize >= self.vocab_size
            || self.sliding_window != expected_window
            || !self.use_sliding_window
            || !d.use_swa
            || d.swa_window_size != self.sliding_window
            || self.max_position_embeddings < self.sliding_window
            || self.layer_types != vec!["sliding_attention"; self.num_hidden_layers]
            || !positive(self.rope_theta)
            || !positive(self.rms_norm_eps)
            || !positive(d.lilicorr_vector_eps)
            || !positive(d.lilicorr_logit_scale)
        {
            return Err(invalid("unsupported Nemotron 3.5 LiLiCorr geometry"));
        }
        Ok(())
    }
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}
fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "LiLiCorr config",
        detail: detail.into(),
    }
}

/// Reads and validates companion metadata without allocating device memory.
pub fn inspect_lilicorr_config(root: impl AsRef<Path>) -> Result<LiLiCorrConfig> {
    let path = root.as_ref().join("config.json");
    let bytes = fs::read(&path).map_err(|e| invalid(format!("read {}: {e}", path.display())))?;
    let config: LiLiCorrConfig = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("parse {}: {e}", path.display())))?;
    config.validate()?;
    Ok(config)
}

/// Checks every consumed tensor, including biases and NVFP4 scale records.
pub fn validate_lilicorr_checkpoint(root: impl AsRef<Path>, config: &LiLiCorrConfig) -> Result<()> {
    config.validate()?;
    let checkpoint = ModelOptCheckpoint::open(root)?;
    for tensor in expected_tensors(config) {
        let info = checkpoint.tensor_info(&tensor.name)?;
        tensor.validate(&info.dtype, &info.shape)?;
    }
    Ok(())
}

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

impl Tensor {
    fn validate(&self, dtype: &str, shape: &[usize]) -> Result<()> {
        if dtype != self.dtype || shape != self.shape {
            return Err(Error::Shape {
                label: "LiLiCorr tensor",
                expected: format!("{}: {} {:?}", self.name, self.dtype, self.shape),
                actual: format!("{}: {dtype} {shape:?}", self.name),
            });
        }
        Ok(())
    }
}

fn push(tensors: &mut Vec<Tensor>, name: impl Into<String>, dtype: &'static str, shape: &[usize]) {
    tensors.push(Tensor {
        name: name.into(),
        dtype,
        shape: shape.into(),
    });
}

fn linear(
    tensors: &mut Vec<Tensor>,
    prefix: &str,
    out: usize,
    input: usize,
    fp4: bool,
    bias: bool,
) {
    if fp4 {
        push(tensors, format!("{prefix}.weight"), "U8", &[out, input / 2]);
        push(
            tensors,
            format!("{prefix}.weight_scale"),
            "F8_E4M3",
            &[out, input / 16],
        );
        push(tensors, format!("{prefix}.weight_scale_2"), "F32", &[]);
    } else {
        push(tensors, format!("{prefix}.weight"), "BF16", &[out, input]);
    }
    if bias {
        push(tensors, format!("{prefix}.bias"), "BF16", &[out]);
    }
}

fn expected_tensors(c: &LiLiCorrConfig) -> Vec<Tensor> {
    let d = &c.dflash_config;
    let h = c.hidden_size;
    let l = d.lilicorr_hidden_size;
    let mut t = Vec::new();
    linear(&mut t, "fc", h, h * d.target_layer_ids.len(), true, false);
    if c.has_own_lm_head {
        linear(&mut t, "lm_head", c.vocab_size, h, true, false);
    }
    for name in ["hidden_norm", "norm"] {
        push(&mut t, format!("{name}.weight"), "BF16", &[h]);
    }
    for i in 0..c.num_hidden_layers {
        let p = format!("layers.{i}");
        for side in ["attention_conv", "mlp_conv"] {
            push(
                &mut t,
                format!("{p}.{side}.base_kernel"),
                "BF16",
                &[2, d.conv_kernel_size, h],
            );
            linear(
                &mut t,
                &format!("{p}.{side}.kernel_projection"),
                2 * d.conv_kernel_size * h / d.conv_group_size,
                h,
                false,
                false,
            );
        }
        for n in ["input_layernorm", "post_attention_layernorm"] {
            push(&mut t, format!("{p}.{n}.weight"), "BF16", &[h]);
        }
        for n in ["q_norm", "k_norm"] {
            push(
                &mut t,
                format!("{p}.self_attn.{n}.weight"),
                "BF16",
                &[c.head_dim],
            );
        }
        if d.attention_sink_bias {
            push(
                &mut t,
                format!("{p}.self_attn.attention_sink_bias"),
                "BF16",
                &[c.num_attention_heads],
            );
        }
        for (n, out, input) in [
            ("q_proj", c.num_attention_heads * c.head_dim, h),
            ("k_proj", c.num_key_value_heads * c.head_dim, h),
            ("v_proj", c.num_key_value_heads * c.head_dim, h),
            ("o_proj", h, c.num_attention_heads * c.head_dim),
        ] {
            linear(
                &mut t,
                &format!("{p}.self_attn.{n}"),
                out,
                input,
                false,
                false,
            );
        }
        for n in ["gate_proj", "up_proj"] {
            linear(
                &mut t,
                &format!("{p}.mlp.{n}"),
                c.intermediate_size,
                h,
                true,
                false,
            );
        }
        linear(
            &mut t,
            &format!("{p}.mlp.down_proj"),
            h,
            c.intermediate_size,
            true,
            false,
        );
    }
    for n in ["token_proj", "pass_hidden_proj", "context_proj"] {
        linear(&mut t, &format!("lilicorr.{n}"), l, h, true, true);
    }
    for n in ["weight", "bias"] {
        push(&mut t, format!("lilicorr.feature_norm.{n}"), "BF16", &[5]);
    }
    linear(&mut t, "lilicorr.feature_mlp.up_proj", l, 5, false, true);
    linear(&mut t, "lilicorr.feature_mlp.down_proj", l, l, false, true);
    push(
        &mut t,
        "lilicorr.slot_embedding",
        "BF16",
        &[1, 1, c.draft_tokens(), 1, l],
    );
    push(
        &mut t,
        "lilicorr.rank_embedding",
        "BF16",
        &[1, 1, 1, d.lilicorr_candidate_topk, l],
    );
    push(
        &mut t,
        "lilicorr.relative_slot_bias",
        "BF16",
        &[d.lilicorr_num_heads, 2 * c.block_size - 1],
    );
    push(
        &mut t,
        "lilicorr.same_slot_bias",
        "BF16",
        &[d.lilicorr_num_heads],
    );
    for i in 0..d.lilicorr_num_layers {
        let p = format!("lilicorr.layers.{i}");
        for n in ["attn_norm", "mlp_norm"] {
            push(&mut t, format!("{p}.{n}.weight"), "BF16", &[l]);
        }
        push(
            &mut t,
            format!("{p}.attn.in_proj_weight"),
            "BF16",
            &[3 * l, l],
        );
        push(&mut t, format!("{p}.attn.in_proj_bias"), "BF16", &[3 * l]);
        linear(&mut t, &format!("{p}.attn.out_proj"), l, l, true, true);
        linear(&mut t, &format!("{p}.mlp.up_proj"), 2 * l, l, true, true);
        linear(&mut t, &format!("{p}.mlp.down_proj"), l, 2 * l, true, true);
    }
    for n in ["output_norm", "anchor_norm"] {
        push(&mut t, format!("lilicorr.{n}.weight"), "BF16", &[l]);
    }
    linear(&mut t, "lilicorr.factor_input_proj", l, 3 * l, false, true);
    for n in ["out_head", "in_head", "anchor_out_head"] {
        linear(
            &mut t,
            &format!("lilicorr.{n}"),
            d.lilicorr_factor_dim,
            l,
            false,
            true,
        );
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    fn official() -> LiLiCorrConfig {
        serde_json::from_str(include_str!("fixtures/config.json")).unwrap()
    }

    #[test]
    fn released_header_matches_complete_tensor_contract() {
        // Header from pinned HF revision 7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2.
        let header: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/tensors.json")).unwrap();
        let config = official();
        config.validate().unwrap();
        let expected = expected_tensors(&config);
        assert_eq!(expected.len(), header.as_object().unwrap().len());
        for t in expected {
            let info = &header[&t.name];
            let shape: Vec<usize> = serde_json::from_value(info["shape"].clone()).unwrap();
            t.validate(info["dtype"].as_str().unwrap(), &shape).unwrap();
        }
    }

    #[test]
    fn bolt_header_matches_shared_head_and_attention_sinks() {
        let config: LiLiCorrConfig =
            serde_json::from_str(include_str!("fixtures/bolt-config.json")).unwrap();
        config.validate().unwrap();
        let header: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/bolt-tensors.json")).unwrap();
        let expected = expected_tensors(&config);
        assert_eq!(expected.len(), header.as_object().unwrap().len());
        for t in expected {
            let info = &header[&t.name];
            let shape: Vec<usize> = serde_json::from_value(info["shape"].clone()).unwrap();
            t.validate(info["dtype"].as_str().unwrap(), &shape).unwrap();
        }
        let mut invalid = config;
        invalid.dflash_config.attention_sink_bias = false;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn rejects_other_drafters_and_activation_quantization() {
        let mut config = official();
        config.architectures = vec!["DFlash2DraftModel".into()];
        assert!(config.validate().is_err());
        config = official();
        config.quantization_config.quant_algo = "NVFP4".into();
        assert!(config.validate().is_err());
        config = official();
        config.dflash_config.target_layer_ids[4] = 88;
        assert!(config.validate().is_err());
        config = official();
        config.has_own_lm_head = false;
        assert!(config.validate().is_err());
    }
}
