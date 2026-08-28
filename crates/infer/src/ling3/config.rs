//! Ling 3 checkpoint configuration parsing and validation.

use eider_cuda::{Error, Result};
use eider_format::ModelOptCheckpoint;
use serde::Deserialize;
use std::fs;
use std::path::Path;

const MODEL_TYPE: &str = "bailing_hybrid";
const ARCHITECTURE: &str = "BailingMoeV3ForCausalLM";

/// Attention implementation used by one Ling 3 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ling3AttentionKind {
    /// Kimi Delta Attention with persistent convolution and matrix state.
    Kda,
    /// Gated Multi-head Latent Attention with a causal KV cache.
    Mla,
}

/// Feed-forward implementation used by one Ling 3 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ling3FfnKind {
    Dense,
    Moe,
}

/// Published block-FP8 storage metadata, when present.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ling3Fp8Config {
    pub weight_block_size: [usize; 2],
    pub scale_format: Option<String>,
}

/// Validated Ling 3 checkpoint architecture.
#[derive(Clone, Debug, PartialEq)]
pub struct Ling3Manifest {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub qk_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub dense_intermediate_size: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub expert_groups: usize,
    pub selected_expert_groups: usize,
    pub first_moe_layer: usize,
    pub layer_group_size: usize,
    pub conv_kernel_size: usize,
    pub kda_lower_bound: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub nextn_predict_layers: usize,
    pub fp8: Option<Ling3Fp8Config>,
    layer_attention: Vec<Ling3AttentionKind>,
    layer_ffn: Vec<Ling3FfnKind>,
}

/// Representative checkpoint tensor metadata collected without loading payloads.
#[derive(Clone, Debug, PartialEq)]
pub struct Ling3TensorCheck {
    pub name: String,
    pub present: bool,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub expected_shape: Vec<usize>,
}

impl Ling3TensorCheck {
    pub fn shape_matches(&self) -> bool {
        self.present && self.shape.as_ref() == Some(&self.expected_shape)
    }
}

/// Tensor-level support summary for a Ling 3 checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Ling3ModelInspection {
    pub manifest: Ling3Manifest,
    pub tensors: Vec<Ling3TensorCheck>,
}

impl Ling3Manifest {
    /// Reads and validates a Hugging Face Ling 3 `config.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let bytes = fs::read(&path).map_err(|error| Error::Format {
            label: "Ling 3 config",
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
        let raw: RawLing3Config = serde_json::from_slice(bytes).map_err(|error| Error::Format {
            label: "Ling 3 config",
            detail: format!("invalid JSON: {error}"),
        })?;
        raw.validate()
    }

    pub fn attention_kind(&self, layer: usize) -> Result<Ling3AttentionKind> {
        self.layer_attention
            .get(layer)
            .copied()
            .ok_or_else(|| layer_error(layer, self.num_hidden_layers))
    }

    pub fn ffn_kind(&self, layer: usize) -> Result<Ling3FfnKind> {
        self.layer_ffn
            .get(layer)
            .copied()
            .ok_or_else(|| layer_error(layer, self.num_hidden_layers))
    }

    pub fn kda_layers(&self) -> usize {
        self.layer_attention
            .iter()
            .filter(|&&kind| kind == Ling3AttentionKind::Kda)
            .count()
    }

    pub fn mla_layers(&self) -> usize {
        self.num_hidden_layers - self.kda_layers()
    }

    pub fn recurrent_state_values_per_kda_layer(&self) -> usize {
        self.attention_heads * self.head_dim * self.v_head_dim
    }

    pub fn conv_state_values_per_kda_layer(&self) -> usize {
        3 * self.attention_heads * self.head_dim * (self.conv_kernel_size - 1)
    }

    /// Inspects representative tensors from every distinct Ling layer boundary.
    pub fn inspect(model_dir: impl AsRef<Path>) -> Result<Ling3ModelInspection> {
        let manifest = Self::load(&model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let expected = manifest.representative_tensors();
        let mut tensors = Vec::with_capacity(expected.len());
        for (name, expected_shape) in expected {
            if !checkpoint.contains_tensor(&name) {
                tensors.push(Ling3TensorCheck {
                    name,
                    present: false,
                    dtype: None,
                    shape: None,
                    expected_shape,
                });
                continue;
            }
            let info = checkpoint.tensor_info(&name)?;
            let fp8_scale = (info.dtype == "F8_E4M3" && info.shape.len() == 2).then(|| {
                (
                    format!("{name}_scale_inv"),
                    vec![info.shape[0].div_ceil(128), info.shape[1].div_ceil(128)],
                )
            });
            tensors.push(Ling3TensorCheck {
                name,
                present: true,
                dtype: Some(info.dtype),
                shape: Some(info.shape),
                expected_shape,
            });
            if let Some((scale_name, scale_shape)) = fp8_scale {
                if checkpoint.contains_tensor(&scale_name) {
                    let scale_info = checkpoint.tensor_info(&scale_name)?;
                    tensors.push(Ling3TensorCheck {
                        name: scale_name,
                        present: true,
                        dtype: Some(scale_info.dtype),
                        shape: Some(scale_info.shape),
                        expected_shape: scale_shape,
                    });
                } else {
                    tensors.push(Ling3TensorCheck {
                        name: scale_name,
                        present: false,
                        dtype: None,
                        shape: None,
                        expected_shape: scale_shape,
                    });
                }
            }
        }
        Ok(Ling3ModelInspection { manifest, tensors })
    }

    fn representative_tensors(&self) -> Vec<(String, Vec<usize>)> {
        let hidden = self.hidden_size;
        let projection = self.attention_heads * self.head_dim;
        let mut tensors = vec![
            (
                "model.word_embeddings.weight".to_string(),
                vec![self.vocab_size, hidden],
            ),
            ("model.norm.weight".to_string(), vec![hidden]),
            ("lm_head.weight".to_string(), vec![self.vocab_size, hidden]),
        ];

        let kda = self
            .layer_attention
            .iter()
            .position(|&kind| kind == Ling3AttentionKind::Kda)
            .expect("validated Ling checkpoint has KDA layers");
        let kda_prefix = format!("model.layers.{kda}.attention");
        for projection_name in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"] {
            tensors.push((
                format!("{kda_prefix}.{projection_name}.weight"),
                vec![projection, hidden],
            ));
        }
        tensors.push((
            format!("{kda_prefix}.o_proj.weight"),
            vec![hidden, projection],
        ));
        tensors.push((
            format!("{kda_prefix}.b_proj.weight"),
            vec![self.attention_heads, hidden],
        ));
        for conv in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            tensors.push((
                format!("{kda_prefix}.{conv}.weight"),
                vec![projection, 1, self.conv_kernel_size],
            ));
        }
        tensors.push((format!("{kda_prefix}.A_log"), vec![self.attention_heads]));
        tensors.push((format!("{kda_prefix}.dt_bias"), vec![projection]));
        tensors.push((format!("{kda_prefix}.o_norm.weight"), vec![self.head_dim]));

        let mla = self
            .layer_attention
            .iter()
            .position(|&kind| kind == Ling3AttentionKind::Mla)
            .expect("validated Ling checkpoint has MLA layers");
        let mla_prefix = format!("model.layers.{mla}.attention");
        if let Some(rank) = self.q_lora_rank {
            tensors.push((format!("{mla_prefix}.q_a_proj.weight"), vec![rank, hidden]));
            tensors.push((format!("{mla_prefix}.q_a_layernorm.weight"), vec![rank]));
            tensors.push((
                format!("{mla_prefix}.q_b_proj.weight"),
                vec![self.attention_heads * self.qk_head_dim, rank],
            ));
        } else {
            tensors.push((
                format!("{mla_prefix}.q_proj.weight"),
                vec![self.attention_heads * self.qk_head_dim, hidden],
            ));
        }
        tensors.extend([
            (
                format!("{mla_prefix}.kv_a_proj_with_mqa.weight"),
                vec![self.kv_lora_rank + self.qk_rope_head_dim, hidden],
            ),
            (
                format!("{mla_prefix}.kv_a_layernorm.weight"),
                vec![self.kv_lora_rank],
            ),
            (
                format!("{mla_prefix}.kv_b_proj.weight"),
                vec![
                    self.attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
                    self.kv_lora_rank,
                ],
            ),
            (
                format!("{mla_prefix}.g_proj.weight"),
                vec![self.attention_heads, hidden],
            ),
            (
                format!("{mla_prefix}.dense.weight"),
                vec![hidden, self.attention_heads * self.v_head_dim],
            ),
        ]);

        let dense = self
            .layer_ffn
            .iter()
            .position(|&kind| kind == Ling3FfnKind::Dense)
            .expect("validated Ling checkpoint has a dense layer");
        let dense_prefix = format!("model.layers.{dense}.mlp");
        tensors.extend([
            (
                format!("{dense_prefix}.gate_proj.weight"),
                vec![self.dense_intermediate_size, hidden],
            ),
            (
                format!("{dense_prefix}.up_proj.weight"),
                vec![self.dense_intermediate_size, hidden],
            ),
            (
                format!("{dense_prefix}.down_proj.weight"),
                vec![hidden, self.dense_intermediate_size],
            ),
        ]);

        let moe = self
            .layer_ffn
            .iter()
            .position(|&kind| kind == Ling3FfnKind::Moe)
            .expect("validated Ling checkpoint has MoE layers");
        let moe_prefix = format!("model.layers.{moe}.mlp");
        tensors.extend([
            (
                format!("{moe_prefix}.gate.weight"),
                vec![self.routed_experts, hidden],
            ),
            (
                format!("{moe_prefix}.gate.expert_bias"),
                vec![self.routed_experts],
            ),
        ]);
        for expert in [0, self.routed_experts - 1] {
            let expert_prefix = format!("{moe_prefix}.experts.{expert}");
            tensors.extend([
                (
                    format!("{expert_prefix}.gate_proj.weight"),
                    vec![self.expert_intermediate_size, hidden],
                ),
                (
                    format!("{expert_prefix}.up_proj.weight"),
                    vec![self.expert_intermediate_size, hidden],
                ),
                (
                    format!("{expert_prefix}.down_proj.weight"),
                    vec![hidden, self.expert_intermediate_size],
                ),
            ]);
        }
        let shared = format!("{moe_prefix}.shared_experts");
        tensors.extend([
            (
                format!("{shared}.gate_proj.weight"),
                vec![self.shared_expert_intermediate_size, hidden],
            ),
            (
                format!("{shared}.up_proj.weight"),
                vec![self.shared_expert_intermediate_size, hidden],
            ),
            (
                format!("{shared}.down_proj.weight"),
                vec![hidden, self.shared_expert_intermediate_size],
            ),
        ]);
        tensors
    }
}

#[derive(Deserialize)]
struct RawLing3Config {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    qk_head_dim: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    q_lora_rank: Option<usize>,
    kv_lora_rank: usize,
    intermediate_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    moe_shared_expert_intermediate_size: usize,
    num_shared_experts: usize,
    routed_scaling_factor: f32,
    n_group: usize,
    topk_group: usize,
    first_k_dense_replace: usize,
    layer_group_size: usize,
    short_conv_kernel_size: usize,
    kda_lower_bound: f32,
    kda_safe_gate: bool,
    no_kda_lora: bool,
    use_qk_norm: bool,
    linear_silu: bool,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_interleave: bool,
    gated_attention_proj_granularity_type: String,
    hidden_act: String,
    score_function: String,
    scoring_func: String,
    topk_method: String,
    norm_topk_prob: bool,
    router_dtype: String,
    num_nextn_predict_layers: usize,
    quantization_config: Option<RawLing3QuantizationConfig>,
}

#[derive(Deserialize)]
struct RawLing3QuantizationConfig {
    quant_method: String,
    fmt: String,
    activation_scheme: String,
    weight_block_size: Vec<usize>,
    scale_fmt: Option<String>,
}

impl RawLing3Config {
    fn validate(self) -> Result<Ling3Manifest> {
        let identity_valid = self.model_type == MODEL_TYPE
            && self.architectures.iter().any(|value| value == ARCHITECTURE)
            && self.hidden_act == "silu"
            && self.score_function == "sigmoid"
            && self.scoring_func == "sigmoid"
            && self.topk_method == "noaux_tc"
            && self.norm_topk_prob
            && self.router_dtype == "fp32"
            && self.gated_attention_proj_granularity_type == "head_wise"
            && self.kda_safe_gate
            && self.no_kda_lora
            && self.use_qk_norm
            && self.linear_silu
            && self.rope_interleave;
        if !identity_valid {
            return Err(config_error(format!(
                "unsupported identity or numerical policy: model_type={} architectures={:?} score={}/{} topk={} router={} attention_gate={}",
                self.model_type,
                self.architectures,
                self.score_function,
                self.scoring_func,
                self.topk_method,
                self.router_dtype,
                self.gated_attention_proj_granularity_type,
            )));
        }
        let finite_positive = [
            self.routed_scaling_factor,
            self.rms_norm_eps,
            self.rope_theta,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0);
        let dimensions_valid = self.vocab_size > 0
            && self.hidden_size > 0
            && self.num_hidden_layers > 0
            && self.max_position_embeddings > 0
            && self.num_attention_heads > 0
            && self.num_key_value_heads == self.num_attention_heads
            && self.head_dim > 0
            && self.rotary_dim > 0
            && self.rotary_dim.is_multiple_of(2)
            && self.qk_head_dim == self.qk_nope_head_dim + self.qk_rope_head_dim
            && self.qk_rope_head_dim == self.rotary_dim
            && self.v_head_dim > 0
            && self.q_lora_rank.is_none_or(|rank| rank > 0)
            && self.kv_lora_rank > 0
            && self.intermediate_size > 0
            && self.num_experts > 0
            && self.num_experts_per_tok > 0
            && self.num_experts_per_tok <= self.num_experts
            && self.moe_intermediate_size > 0
            && self.moe_shared_expert_intermediate_size > 0
            && self.num_shared_experts == 1
            && self.n_group > 0
            && self.num_experts.is_multiple_of(self.n_group)
            && self.topk_group > 0
            && self.topk_group <= self.n_group
            && self.first_k_dense_replace > 0
            && self.first_k_dense_replace < self.num_hidden_layers
            && self.layer_group_size > 1
            && self.layer_group_size <= self.num_hidden_layers
            && self.short_conv_kernel_size > 1
            && self.kda_lower_bound.is_finite()
            && self.kda_lower_bound < 0.0
            && self.num_nextn_predict_layers <= 1
            && finite_positive;
        if !dimensions_valid {
            return Err(config_error(format!(
                "invalid dimensions: vocab={} hidden={} layers={} heads={} kv_heads={} head_dim={} qk={}/{}+{} experts={} top_k={} groups={}/{}",
                self.vocab_size,
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.qk_head_dim,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.num_experts,
                self.num_experts_per_tok,
                self.topk_group,
                self.n_group,
            )));
        }

        let fp8 = self
            .quantization_config
            .map(RawLing3QuantizationConfig::validate)
            .transpose()?;
        let layer_attention = (0..self.num_hidden_layers)
            .map(|layer| {
                if (layer + 1).is_multiple_of(self.layer_group_size)
                    || layer
                        >= self.num_hidden_layers / self.layer_group_size * self.layer_group_size
                {
                    Ling3AttentionKind::Mla
                } else {
                    Ling3AttentionKind::Kda
                }
            })
            .collect::<Vec<_>>();
        if !layer_attention.contains(&Ling3AttentionKind::Kda)
            || !layer_attention.contains(&Ling3AttentionKind::Mla)
        {
            return Err(config_error("layer schedule must contain both KDA and MLA"));
        }
        let layer_ffn = (0..self.num_hidden_layers)
            .map(|layer| {
                if layer < self.first_k_dense_replace {
                    Ling3FfnKind::Dense
                } else {
                    Ling3FfnKind::Moe
                }
            })
            .collect();

        Ok(Ling3Manifest {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            max_position_embeddings: self.max_position_embeddings,
            attention_heads: self.num_attention_heads,
            kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            qk_head_dim: self.qk_head_dim,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            q_lora_rank: self.q_lora_rank,
            kv_lora_rank: self.kv_lora_rank,
            dense_intermediate_size: self.intermediate_size,
            routed_experts: self.num_experts,
            experts_per_token: self.num_experts_per_tok,
            expert_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.moe_shared_expert_intermediate_size,
            routed_scaling_factor: self.routed_scaling_factor,
            expert_groups: self.n_group,
            selected_expert_groups: self.topk_group,
            first_moe_layer: self.first_k_dense_replace,
            layer_group_size: self.layer_group_size,
            conv_kernel_size: self.short_conv_kernel_size,
            kda_lower_bound: self.kda_lower_bound,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            nextn_predict_layers: self.num_nextn_predict_layers,
            fp8,
            layer_attention,
            layer_ffn,
        })
    }
}

impl RawLing3QuantizationConfig {
    fn validate(self) -> Result<Ling3Fp8Config> {
        if self.quant_method != "fp8"
            || self.fmt != "e4m3"
            || self.activation_scheme != "dynamic"
            || self.weight_block_size.as_slice() != [128, 128]
            || self
                .scale_fmt
                .as_deref()
                .is_some_and(|format| format != "ue8m0")
        {
            return Err(config_error(format!(
                "unsupported quantization: method={} format={} activation={} block={:?} scale={:?}",
                self.quant_method,
                self.fmt,
                self.activation_scheme,
                self.weight_block_size,
                self.scale_fmt,
            )));
        }
        Ok(Ling3Fp8Config {
            weight_block_size: [128, 128],
            scale_format: self.scale_fmt,
        })
    }
}

fn layer_error(layer: usize, layers: usize) -> Error {
    Error::Shape {
        label: "Ling 3 layer",
        expected: format!("layer < {layers}"),
        actual: layer.to_string(),
    }
}

fn config_error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Ling 3 config",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Ling3AttentionKind, Ling3FfnKind, Ling3Manifest};

    const TINY_CONFIG: &str = r#"{
      "architectures":["BailingMoeV3ForCausalLM"],
      "model_type":"bailing_hybrid",
      "vocab_size":157184,
      "hidden_size":1536,
      "num_hidden_layers":24,
      "max_position_embeddings":131072,
      "num_attention_heads":16,
      "num_key_value_heads":16,
      "head_dim":128,
      "rotary_dim":64,
      "qk_head_dim":192,
      "qk_nope_head_dim":128,
      "qk_rope_head_dim":64,
      "v_head_dim":128,
      "q_lora_rank":256,
      "kv_lora_rank":512,
      "intermediate_size":4608,
      "num_experts":128,
      "num_experts_per_tok":8,
      "moe_intermediate_size":512,
      "moe_shared_expert_intermediate_size":512,
      "num_shared_experts":1,
      "routed_scaling_factor":2.5,
      "n_group":8,
      "topk_group":4,
      "first_k_dense_replace":1,
      "layer_group_size":4,
      "short_conv_kernel_size":4,
      "kda_lower_bound":-5,
      "kda_safe_gate":true,
      "no_kda_lora":true,
      "use_qk_norm":true,
      "linear_silu":true,
      "rms_norm_eps":1e-6,
      "rope_theta":6000000,
      "rope_interleave":true,
      "gated_attention_proj_granularity_type":"head_wise",
      "hidden_act":"silu",
      "score_function":"sigmoid",
      "scoring_func":"sigmoid",
      "topk_method":"noaux_tc",
      "norm_topk_prob":true,
      "router_dtype":"fp32",
      "num_nextn_predict_layers":0
    }"#;

    #[test]
    fn tiny_schedule_and_state_shape_are_exact() {
        let manifest = Ling3Manifest::from_json(TINY_CONFIG.as_bytes()).expect("Tiny config");
        assert_eq!(manifest.kda_layers(), 18);
        assert_eq!(manifest.mla_layers(), 6);
        assert_eq!(manifest.attention_kind(0).unwrap(), Ling3AttentionKind::Kda);
        assert_eq!(manifest.attention_kind(3).unwrap(), Ling3AttentionKind::Mla);
        assert_eq!(
            manifest.attention_kind(23).unwrap(),
            Ling3AttentionKind::Mla
        );
        assert_eq!(manifest.ffn_kind(0).unwrap(), Ling3FfnKind::Dense);
        assert_eq!(manifest.ffn_kind(1).unwrap(), Ling3FfnKind::Moe);
        assert_eq!(
            manifest.recurrent_state_values_per_kda_layer(),
            16 * 128 * 128
        );
        assert_eq!(manifest.conv_state_values_per_kda_layer(), 3 * 16 * 128 * 3);
    }

    #[test]
    fn tiny_fp8_policy_is_recorded() {
        let json = TINY_CONFIG.replace(
            "\"num_nextn_predict_layers\":0",
            "\"num_nextn_predict_layers\":0,\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"activation_scheme\":\"dynamic\",\"scale_fmt\":\"ue8m0\",\"weight_block_size\":[128,128]}",
        );
        let manifest = Ling3Manifest::from_json(json.as_bytes()).expect("Tiny FP8 config");
        let fp8 = manifest.fp8.expect("FP8 metadata");
        assert_eq!(fp8.weight_block_size, [128, 128]);
        assert_eq!(fp8.scale_format.as_deref(), Some("ue8m0"));
    }
}
