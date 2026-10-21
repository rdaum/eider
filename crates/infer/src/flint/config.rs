//! Flint architecture and tensor contract from the NVFP4 GGUF checkpoint.

use eider_cuda::{Error, Result};
use eider_format::{GGML_TYPE_NVFP4, GgufIndex, GgufValue};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct FlintConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub expert_intermediate: usize,
    pub shared_intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub dense_layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub max_context: usize,
    pub window: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub sliding: Vec<bool>,
    pub rope: Vec<bool>,
}

impl FlintConfig {
    /// Opens a GGUF and validates its architecture and complete tensor contract.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        Self::from_index(&GgufIndex::open(path)?)
    }
    pub fn from_index(index: &GgufIndex) -> Result<Self> {
        let meta = index.metadata();
        if meta.get("general.architecture").and_then(GgufValue::as_str) != Some("flint") {
            return Err(invalid("expected general.architecture=flint"));
        }
        let integer = |name: &str| -> Result<usize> {
            meta.get(&format!("flint.{name}"))
                .and_then(GgufValue::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid(format!("missing integer flint.{name}")))
        };
        let float = |name: &str| -> Result<f32> {
            meta.get(&format!("flint.{name}"))
                .and_then(GgufValue::as_f64)
                .map(|v| v as f32)
                .ok_or_else(|| invalid(format!("missing number flint.{name}")))
        };
        let pattern = |name: &str| -> Result<Vec<bool>> {
            meta.get(&format!("flint.attention.{name}_pattern"))
                .and_then(GgufValue::as_bool_slice)
                .map(<[bool]>::to_vec)
                .ok_or_else(|| invalid(format!("missing {name} pattern")))
        };
        let vocab = match index.tensor("token_embd.weight")?.dimensions.as_slice() {
            &[_, vocab] => usize::try_from(vocab).map_err(|_| invalid("vocabulary overflow"))?,
            _ => return Err(invalid("embedding must have rank two")),
        };
        let config = Self {
            layers: integer("block_count")?,
            hidden: integer("embedding_length")?,
            intermediate: integer("feed_forward_length")?,
            expert_intermediate: integer("expert_feed_forward_length")?,
            shared_intermediate: integer("expert_shared_feed_forward_length")?,
            experts: integer("expert_count")?,
            top_k: integer("expert_used_count")?,
            dense_layers: integer("leading_dense_block_count")?,
            q_heads: integer("attention.head_count")?,
            kv_heads: integer("attention.head_count_kv")?,
            head_dim: integer("attention.key_length")?,
            vocab,
            max_context: integer("context_length")?,
            window: integer("attention.sliding_window")?,
            rms_eps: float("attention.layer_norm_rms_epsilon")?,
            rope_theta: float("rope.freq_base")?,
            sliding: pattern("sliding_window")?,
            rope: pattern("rope")?,
        };
        config.validate()?;
        if integer("attention.value_length")? != config.head_dim
            || integer("expert_gating_func")? != 2
            || meta.get("flint.expert_weights_norm") != Some(&GgufValue::Bool(true))
            || float("expert_weights_scale")? != 1.0
        {
            return Err(invalid(
                "expected matching KV dimensions and normalized sigmoid routing with scale 1",
            ));
        }
        config.validate_tensors(index)?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.layers == 0
            || self.layers > 256
            || self.dense_layers >= self.layers
            || self.q_heads == 0
            || self.kv_heads == 0
            || !self.q_heads.is_multiple_of(self.kv_heads)
            || self.head_dim != 128
            || self.vocab == 0
            || self.max_context == 0
            || self.max_context > u32::MAX as usize
            || self.window == 0
            || self.window > self.max_context
            || self.experts == 0
            || self.experts > 512
            || self.top_k == 0
            || self.top_k > self.experts
            || [
                self.hidden,
                self.intermediate,
                self.expert_intermediate,
                self.shared_intermediate,
            ]
            .iter()
            .any(|&n| n == 0 || !n.is_multiple_of(64))
            || self.sliding.len() != self.layers
            || self.rope.len() != self.layers
            || !self.rms_eps.is_finite()
            || self.rms_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(invalid(format!("unsupported configuration: {self:?}")));
        }
        Ok(())
    }

    fn validate_tensors(&self, index: &GgufIndex) -> Result<()> {
        let mut expected = BTreeSet::new();
        let mut quantized = BTreeSet::new();
        let mut require = |name: String, shape: Vec<usize>, kind: u32| -> Result<()> {
            let t = index.tensor(&name)?;
            if t.kind != kind || t.dimensions != shape.iter().map(|&v| v as u64).collect::<Vec<_>>()
            {
                return Err(invalid(format!(
                    "{name}: expected type {kind} {shape:?}, got type {} {:?}",
                    t.kind, t.dimensions
                )));
            }
            expected.insert(name.clone());
            if kind == GGML_TYPE_NVFP4 {
                quantized.insert(name.clone());
                let scale_name = format!("{}.scale", name.strip_suffix(".weight").unwrap());
                let scale = index.tensor(&scale_name)?;
                if scale.kind != 0
                    || scale.dimensions != [shape.get(2).copied().unwrap_or(1) as u64]
                {
                    return Err(invalid(format!("invalid scale tensor {scale_name}")));
                }
                expected.insert(scale_name);
            }
            Ok(())
        };
        let h = self.hidden;
        require("token_embd.weight".into(), vec![h, self.vocab], 30)?;
        require("output.weight".into(), vec![h, self.vocab], 40)?;
        require("output_norm.weight".into(), vec![h], 0)?;
        for layer in 0..self.layers {
            let p = format!("blk.{layer}");
            for name in ["attn_norm", "ffn_norm"] {
                require(format!("{p}.{name}.weight"), vec![h], 0)?;
            }
            for name in ["attn_q_norm", "attn_k_norm"] {
                require(format!("{p}.{name}.weight"), vec![self.head_dim], 0)?;
            }
            require(format!("{p}.attn_sinks.weight"), vec![self.q_heads], 0)?;
            for (name, rows, cols) in [
                ("attn_q", self.q_heads * self.head_dim, h),
                ("attn_k", self.kv_heads * self.head_dim, h),
                ("attn_v", self.kv_heads * self.head_dim, h),
                ("attn_output", h, self.q_heads * self.head_dim),
            ] {
                require(format!("{p}.{name}.weight"), vec![cols, rows], 40)?;
            }
            let (width, suffix) = if layer < self.dense_layers {
                (self.intermediate, "")
            } else {
                (self.shared_intermediate, "_shexp")
            };
            for name in ["gate", "up"] {
                require(format!("{p}.ffn_{name}{suffix}.weight"), vec![h, width], 40)?;
            }
            require(format!("{p}.ffn_down{suffix}.weight"), vec![width, h], 40)?;
            if layer >= self.dense_layers {
                require(
                    format!("{p}.ffn_gate_inp.weight"),
                    vec![h, self.experts],
                    30,
                )?;
                for name in ["gate", "up"] {
                    require(
                        format!("{p}.ffn_{name}_exps.weight"),
                        vec![h, self.expert_intermediate, self.experts],
                        40,
                    )?;
                }
                require(
                    format!("{p}.ffn_down_exps.weight"),
                    vec![self.expert_intermediate, h, self.experts],
                    40,
                )?;
            }
        }
        if index.tensors().keys().any(|name| !expected.contains(name)) {
            return Err(invalid("unrecognized tensor in Flint checkpoint"));
        }
        let Some(GgufValue::StringArray(names)) = index.metadata().get("general.tensor_extra.name")
        else {
            return Err(invalid("missing per-tensor precision names"));
        };
        let Some(GgufValue::BoolArray(a4)) = index.metadata().get("general.tensor_extra.prec_a4")
        else {
            return Err(invalid("missing per-tensor activation precision"));
        };
        if names.len() != a4.len()
            || a4.iter().any(|&v| v)
            || names.iter().cloned().collect::<BTreeSet<_>>() != quantized
            || names.len() != quantized.len()
        {
            return Err(invalid(
                "expected explicit W4A16 precision for every NVFP4 tensor",
            ));
        }
        Ok(())
    }
}

pub(super) fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Flint",
        detail: detail.into(),
    }
}
