//! Gemma 4 text-model loading and inference.
//!
//! Dense and expert linears are resident in ModelOpt NVFP4 form. BF16 source
//! tensors are converted during loading without materializing a whole expert
//! stack on the host.

use crate::sm12x_cache::Sm12xCacheContext;
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptCublasLtWeight,
    ModelOptNvfp4Linear, Result, Sm12xKvAttentionWorkspace, Sm12xKvPagePool,
    add_f32_into_on_stream, bf16_linear_argmax_f32_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, gather_indexed_mul_f32_into_on_stream,
    gelu_tanh_mul_f32_into_on_stream, lm_head_top1_f32_into_on_stream, moe_topk_f32_into_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_f32_into_on_stream,
    nvfp4_w4a16_matrix_matvec_f32_batch_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_f32_into_on_stream, rope_neox_proportional_f32_into_on_stream,
    round_f32_to_bf16_in_place_on_stream, scale_channel_f32_device_scalar_in_place_on_stream,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

mod batch;
mod sequence;
pub use batch::{Gemma4PrefillBatchWorkspace, Gemma4PrefillOutput, Gemma4PrefillRow};
pub(crate) use sequence::{
    Gemma4Append, gemma4_cache_error, new_gemma4_sequence_cache_with_budget,
};
pub use sequence::{Gemma4Sequence, Gemma4SequenceCache, new_gemma4_sequence_cache};

fn default_rms_norm_eps() -> f32 {
    1.0e-6
}

fn default_attention_k_eq_v() -> bool {
    true
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_final_logit_softcapping() -> f32 {
    30.0
}

fn default_local_rope() -> Gemma4RopeConfig {
    Gemma4RopeConfig {
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
    }
}

fn default_full_rope() -> Gemma4RopeConfig {
    Gemma4RopeConfig {
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 0.25,
    }
}

/// Text-model dimensions and attention layout required to load Gemma 4.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub final_logit_softcapping: f32,
    pub rms_norm_eps: f32,
    pub attention_k_eq_v: bool,
    pub local_rope_theta: f32,
    pub full_rope_theta: f32,
    pub full_partial_rotary_factor: f32,
    pub layer_types: Vec<String>,
}

#[derive(Deserialize)]
struct FileConfig {
    model_type: String,
    text_config: TextConfig,
}

#[derive(Deserialize)]
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_global_key_value_heads: usize,
    head_dim: usize,
    global_head_dim: usize,
    num_experts: usize,
    top_k_experts: usize,
    sliding_window: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default = "default_final_logit_softcapping")]
    final_logit_softcapping: f32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_attention_k_eq_v")]
    attention_k_eq_v: bool,
    #[serde(default)]
    rope_parameters: Gemma4RopeParameters,
    layer_types: Vec<String>,
}

#[derive(Deserialize)]
struct Gemma4RopeParameters {
    #[serde(default = "default_local_rope")]
    sliding_attention: Gemma4RopeConfig,
    #[serde(default = "default_full_rope")]
    full_attention: Gemma4RopeConfig,
}

#[derive(Deserialize)]
struct Gemma4RopeConfig {
    rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f32,
}

impl Default for Gemma4RopeParameters {
    fn default() -> Self {
        Self {
            sliding_attention: default_local_rope(),
            full_attention: default_full_rope(),
        }
    }
}

impl Gemma4Config {
    /// Reads and validates `config.json` from a Gemma 4 checkpoint directory.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text = fs::read_to_string(&path).map_err(|err| Error::Format {
            label: "Gemma 4 config",
            detail: format!("{}: {err}", path.display()),
        })?;
        Self::from_json(&text)
    }

    fn from_json(text: &str) -> Result<Self> {
        let file: FileConfig = serde_json::from_str(text).map_err(|err| Error::Format {
            label: "Gemma 4 config JSON",
            detail: err.to_string(),
        })?;
        if file.model_type != "gemma4" {
            return Err(Error::Format {
                label: "Gemma 4 config",
                detail: format!("expected model_type=gemma4, got {}", file.model_type),
            });
        }
        let config = file.text_config;
        if config.num_hidden_layers == 0
            || config.hidden_size == 0
            || config.num_experts == 0
            || config.top_k_experts == 0
            || config.top_k_experts > config.num_experts
            || !config.rms_norm_eps.is_finite()
            || config.rms_norm_eps <= 0.0
            || !config.final_logit_softcapping.is_finite()
            || config.final_logit_softcapping <= 0.0
            || !config
                .rope_parameters
                .sliding_attention
                .rope_theta
                .is_finite()
            || config.rope_parameters.sliding_attention.rope_theta <= 0.0
            || !config.rope_parameters.full_attention.rope_theta.is_finite()
            || config.rope_parameters.full_attention.rope_theta <= 0.0
            || !config
                .rope_parameters
                .full_attention
                .partial_rotary_factor
                .is_finite()
            || config.rope_parameters.full_attention.partial_rotary_factor <= 0.0
            || config.rope_parameters.full_attention.partial_rotary_factor > 1.0
            || config.layer_types.len() != config.num_hidden_layers
        {
            return Err(Error::Shape {
                label: "Gemma 4 config",
                expected: "nonzero dimensions, 0 < top_k_experts <= num_experts, and one layer type per layer".to_string(),
                actual: format!(
                    "layers={} layer_types={} hidden={} experts={} top_k={}",
                    config.num_hidden_layers,
                    config.layer_types.len(),
                    config.hidden_size,
                    config.num_experts,
                    config.top_k_experts,
                ),
            });
        }
        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            num_global_key_value_heads: config.num_global_key_value_heads,
            head_dim: config.head_dim,
            global_head_dim: config.global_head_dim,
            num_experts: config.num_experts,
            top_k_experts: config.top_k_experts,
            sliding_window: config.sliding_window,
            max_position_embeddings: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            final_logit_softcapping: config.final_logit_softcapping,
            rms_norm_eps: config.rms_norm_eps,
            attention_k_eq_v: config.attention_k_eq_v,
            local_rope_theta: config.rope_parameters.sliding_attention.rope_theta,
            full_rope_theta: config.rope_parameters.full_attention.rope_theta,
            full_partial_rotary_factor: config.rope_parameters.full_attention.partial_rotary_factor,
            layer_types: config.layer_types,
        })
    }

    /// Returns whether `layer` is one of Gemma 4's full-attention layers.
    pub fn is_full_attention_layer(&self, layer: usize) -> Result<bool> {
        self.layer_types
            .get(layer)
            .map(|kind| kind == "full_attention")
            .ok_or_else(|| Error::Shape {
                label: "Gemma 4 layer index",
                expected: format!("layer < {}", self.num_hidden_layers),
                actual: layer.to_string(),
            })
    }
}

/// A Gemma 4 checkpoint with native-ModelOpt and BF16 source support.
#[derive(Clone, Debug)]
pub struct Gemma4Checkpoint {
    config: Gemma4Config,
    checkpoint: ModelOptCheckpoint,
}

/// Device-resident Gemma linear with decode and tensor-core scale layouts.
pub struct Gemma4Linear {
    storage: Gemma4LinearStorage,
    out_features: usize,
    in_features: usize,
}

enum Gemma4LinearStorage {
    Nvfp4 {
        weight: ModelOptCublasLtWeight,
        weight_scale: DeviceBuffer<u8>,
    },
}

/// Gemma's dense gated-GELU feed-forward network.
pub struct Gemma4Mlp {
    gate: Gemma4Linear,
    up: Gemma4Linear,
    down: Gemma4Linear,
    intermediate_size: usize,
}

/// Reusable device buffers for [`Gemma4Mlp`].
pub struct Gemma4MlpWorkspace {
    rows: usize,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

/// Gemma's router normalization, expert selection, and calibrated route weights.
pub struct Gemma4Router {
    input_norm_weight: DeviceBuffer<f32>,
    input_norm_scalar: DeviceBuffer<f32>,
    input_norm_scalar_value: f32,
    router_scale: DeviceBuffer<f32>,
    projection: Gemma4Linear,
    per_expert_scale: DeviceBuffer<f32>,
    rms_norm_eps: f32,
    top_k: usize,
}

/// Reusable device workspace for one Gemma router invocation.
pub struct Gemma4RouterWorkspace {
    normalized: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    normalized_weights: DeviceBuffer<f32>,
    route_weights: DeviceBuffer<f32>,
}

/// One resident Gemma MoE expert, stored as two NVFP4 linear projections.
pub struct Gemma4Expert {
    gate: Gemma4Linear,
    up: Gemma4Linear,
    down: Gemma4Linear,
}

/// A Gemma MoE layer with resident experts and a calibrated router.
pub struct Gemma4Moe {
    router: Gemma4Router,
    // Owns the allocations referenced by the expert pointer tables below.
    _experts: Vec<Gemma4Expert>,
    gate_packed_table: DeviceBuffer<*const u8>,
    gate_scale_table: DeviceBuffer<*const u8>,
    gate_tiled_scale_table: DeviceBuffer<*const u8>,
    gate_scale_2: DeviceBuffer<f32>,
    gate_alpha_table: DeviceBuffer<*mut f32>,
    up_packed_table: DeviceBuffer<*const u8>,
    up_scale_table: DeviceBuffer<*const u8>,
    up_tiled_scale_table: DeviceBuffer<*const u8>,
    up_scale_2: DeviceBuffer<f32>,
    up_alpha_table: DeviceBuffer<*mut f32>,
    down_packed_table: DeviceBuffer<*const u8>,
    down_scale_table: DeviceBuffer<*const u8>,
    down_tiled_scale_table: DeviceBuffer<*const u8>,
    down_scale_2: DeviceBuffer<f32>,
    down_alpha_table: DeviceBuffer<*mut f32>,
    expert_alpha: DeviceBuffer<f32>,
    intermediate_size: usize,
    hidden_size: usize,
}

/// Device-routed scratch for one Gemma MoE token.
pub struct Gemma4MoeWorkspace {
    router: Gemma4RouterWorkspace,
    gate: Vec<DeviceBuffer<f32>>,
    up: Vec<DeviceBuffer<f32>>,
    activated: Vec<DeviceBuffer<f32>>,
    down: Vec<DeviceBuffer<f32>>,
    gate_output_table: DeviceBuffer<*mut f32>,
    up_output_table: DeviceBuffer<*mut f32>,
    down_input_table: DeviceBuffer<*const f32>,
    down_output_table: DeviceBuffer<*mut f32>,
    down_result_table: DeviceBuffer<*const f32>,
    output: DeviceBuffer<f32>,
}

/// A learned Gemma RMSNorm weight.
pub struct Gemma4RmsNorm {
    weight: DeviceBuffer<f32>,
    eps: f32,
}

/// One Gemma local or global grouped-query attention layer.
pub struct Gemma4Attention {
    q: Gemma4Linear,
    k: Gemma4Linear,
    v: Option<Gemma4Linear>,
    output: Gemma4Linear,
    q_norm: Gemma4RmsNorm,
    k_norm: Gemma4RmsNorm,
    value_norm_weight: DeviceBuffer<f32>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    window: Option<usize>,
}

/// Reusable one-token scratch for [`Gemma4Attention`].
pub struct Gemma4AttentionWorkspace {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    v_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

/// One complete Gemma decoder layer.
pub struct Gemma4DecoderLayer {
    input_norm: Gemma4RmsNorm,
    attention: Gemma4Attention,
    post_attention_norm: Gemma4RmsNorm,
    dense_input_norm: Gemma4RmsNorm,
    dense: Gemma4Mlp,
    dense_post_norm: Gemma4RmsNorm,
    moe_input_norm: Gemma4RmsNorm,
    moe: Gemma4Moe,
    moe_post_norm: Gemma4RmsNorm,
    post_feedforward_norm: Gemma4RmsNorm,
    layer_scale_channels: DeviceBuffer<f32>,
    layer_scalar: DeviceBuffer<f32>,
    layer_scalar_value: f32,
}

/// One-token working buffers for a [`Gemma4DecoderLayer`].
pub struct Gemma4DecoderLayerWorkspace {
    attention: Gemma4AttentionWorkspace,
    dense: Gemma4MlpWorkspace,
    moe: Gemma4MoeWorkspace,
    normalized: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    dense_input: DeviceBuffer<f32>,
    dense_output: DeviceBuffer<f32>,
    moe_input: DeviceBuffer<f32>,
    moe_output: DeviceBuffer<f32>,
    combined: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

/// Complete Gemma 4 text backbone with tied BF16 embedding/LM-head storage.
pub struct Gemma4Model {
    config: Gemma4Config,
    embedding: DeviceBuffer<u16>,
    embedding_channel_scale: DeviceBuffer<f32>,
    embedding_scalar: DeviceBuffer<f32>,
    embedding_scalar_value: f32,
    layers: Vec<Gemma4DecoderLayer>,
    final_norm: Gemma4RmsNorm,
}

/// Persistent decode state with independently shaped cache per layer.
pub struct Gemma4DecodeState {
    hidden: DeviceBuffer<f32>,
    layers: Vec<Gemma4DecoderLayerWorkspace>,
    compact_attention: Gemma4CompactAttentionWorkspaces,
    lm_logits: DeviceBuffer<f32>,
    lm_top1_scratch_index: DeviceBuffer<u32>,
    lm_argmax: DeviceBuffer<u32>,
    lm_argmax_value: DeviceBuffer<f32>,
    pub(crate) position: usize,
    max_tokens: usize,
}

struct Gemma4CompactAttentionWorkspaces {
    local: Option<Sm12xKvAttentionWorkspace>,
    global: Option<Sm12xKvAttentionWorkspace>,
}

impl Gemma4CompactAttentionWorkspaces {
    fn new(layers: &[Gemma4DecoderLayer], max_tokens: usize) -> Result<Self> {
        let local = layers
            .iter()
            .find(|layer| layer.attention.window.is_some())
            .map(|layer| layer.attention.new_compact_attention_workspace(max_tokens))
            .transpose()?;
        let global = layers
            .iter()
            .find(|layer| layer.attention.window.is_none())
            .map(|layer| layer.attention.new_compact_attention_workspace(max_tokens))
            .transpose()?;
        Ok(Self { local, global })
    }

    fn for_layer_mut(&mut self, local: bool) -> Result<&mut Sm12xKvAttentionWorkspace> {
        let workspace = if local {
            self.local.as_mut()
        } else {
            self.global.as_mut()
        };
        workspace.ok_or_else(|| Error::Format {
            label: "Gemma 4 compact attention workspace",
            detail: format!(
                "missing {} workspace for decoder layer",
                if local { "local" } else { "global" }
            ),
        })
    }

    fn device_bytes(&self) -> usize {
        self.local
            .as_ref()
            .map_or(0, Sm12xKvAttentionWorkspace::device_bytes)
            + self
                .global
                .as_ref()
                .map_or(0, Sm12xKvAttentionWorkspace::device_bytes)
    }
}

struct Gemma4LayerCache<'a> {
    pool: &'a mut Sm12xKvPagePool,
    page_slot: usize,
    page_offset: usize,
    page_table: &'a DeviceBuffer<u32>,
    attention: &'a mut Sm12xKvAttentionWorkspace,
}

/// The argmax result for one Gemma input token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4NextToken {
    /// Token that was decoded through the model.
    pub input_token: u32,
    /// Highest-logit next token.
    pub token: u32,
    /// Logit of [`Self::token`].
    pub logit: f32,
}

impl Gemma4Linear {
    /// Loads a Gemma linear into resident ModelOpt NVFP4 storage.
    pub fn load(checkpoint: &Gemma4Checkpoint, tensor: &str) -> Result<Self> {
        Self::from_modelopt(checkpoint.load_linear_nvfp4(tensor)?)
    }

    /// Uploads a Gemma expert projection, converting only the selected expert.
    pub fn load_expert(checkpoint: &Gemma4Checkpoint, tensor: &str, expert: usize) -> Result<Self> {
        Self::from_modelopt(checkpoint.load_expert_linear_nvfp4(tensor, expert)?)
    }

    /// Uploads ModelOpt-compatible weight data into the SM12x row-scale layout.
    pub fn from_modelopt(weight: ModelOptNvfp4Linear) -> Result<Self> {
        if !weight.out_features.is_multiple_of(16) || !weight.in_features.is_multiple_of(64) {
            return Err(Error::Shape {
                label: "Gemma 4 NVFP4 linear",
                expected: "out_features divisible by 16 and in_features divisible by 64"
                    .to_string(),
                actual: format!(
                    "out_features={} in_features={}",
                    weight.out_features, weight.in_features
                ),
            });
        }
        let weight_scale = DeviceBuffer::from_host(&weight.weight_scale)?;
        let out_features = weight.out_features;
        let in_features = weight.in_features;
        let weight = ModelOptCublasLtWeight::from_modelopt(&weight)?;
        Ok(Self {
            storage: Gemma4LinearStorage::Nvfp4 {
                weight,
                weight_scale,
            },
            out_features,
            in_features,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// Executes one or more contiguous f32 rows without host synchronization.
    pub fn run_rows_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0
            || input.len() != rows * self.in_features
            || output.len() != rows * self.out_features
        {
            return Err(Error::Shape {
                label: "Gemma 4 linear buffers",
                expected: format!(
                    "rows={rows} input={} output={}",
                    rows * self.in_features,
                    rows * self.out_features
                ),
                actual: format!("rows={rows} input={} output={}", input.len(), output.len()),
            });
        }
        match &self.storage {
            Gemma4LinearStorage::Nvfp4 {
                weight,
                weight_scale,
            } => nvfp4_w4a16_matrix_matvec_f32_batch_into_on_stream(
                input,
                weight.matrix(),
                weight_scale,
                output.output(),
                rows,
                self.out_features,
                self.in_features,
                weight.weight_scale_2(),
                stream,
            ),
        }
    }

    /// Executes a linear for one or more contiguous rows.
    pub fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_rows_into(input, output, rows, stream)
    }

    fn nvfp4_parts(&self) -> Result<(*const u8, *const u8, f32)> {
        let Gemma4LinearStorage::Nvfp4 {
            weight,
            weight_scale,
        } = &self.storage;
        Ok((
            weight.matrix().values_ptr(),
            weight_scale.as_const_ptr().cast::<u8>(),
            weight.weight_scale_2(),
        ))
    }

    fn cublaslt_weight(&self) -> &ModelOptCublasLtWeight {
        let Gemma4LinearStorage::Nvfp4 { weight, .. } = &self.storage;
        weight
    }

    fn grouped_gemm_parts(&self) -> (*const u8, *const u8, f32) {
        let weight = self.cublaslt_weight();
        (
            weight.matrix().values_ptr(),
            weight.matrix().scales_ptr(),
            weight.weight_scale_2(),
        )
    }

    pub fn device_bytes(&self) -> usize {
        let Gemma4LinearStorage::Nvfp4 {
            weight,
            weight_scale,
        } = &self.storage;
        weight.device_bytes() + weight_scale.device_bytes()
    }
}

impl Gemma4Mlp {
    /// Loads the dense gated-GELU MLP for `layer`.
    pub fn load(checkpoint: &Gemma4Checkpoint, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let gate = Gemma4Linear::load(checkpoint, &format!("{prefix}.gate_proj.weight"))?;
        let up = Gemma4Linear::load(checkpoint, &format!("{prefix}.up_proj.weight"))?;
        let down = Gemma4Linear::load(checkpoint, &format!("{prefix}.down_proj.weight"))?;
        let (intermediate_size, hidden_size) = gate.shape();
        if up.shape() != (intermediate_size, hidden_size)
            || down.shape() != (hidden_size, intermediate_size)
            || hidden_size != checkpoint.config.hidden_size
            || intermediate_size != checkpoint.config.intermediate_size
        {
            return Err(Error::Shape {
                label: "Gemma 4 dense MLP",
                expected: format!(
                    "gate/up={}x{}, down={}x{}",
                    checkpoint.config.intermediate_size,
                    checkpoint.config.hidden_size,
                    checkpoint.config.hidden_size,
                    checkpoint.config.intermediate_size,
                ),
                actual: format!(
                    "gate={:?}, up={:?}, down={:?}",
                    gate.shape(),
                    up.shape(),
                    down.shape(),
                ),
            });
        }
        Ok(Self {
            gate,
            up,
            down,
            intermediate_size,
        })
    }

    /// Allocates reusable workspace for `rows` independent tokens.
    pub fn new_workspace(&self, rows: usize) -> Result<Gemma4MlpWorkspace> {
        let hidden_size = self.gate.shape().1;
        let values = rows
            .checked_mul(self.intermediate_size)
            .ok_or_else(|| Error::Shape {
                label: "Gemma 4 dense MLP workspace",
                expected: "rows * intermediate size without overflow".to_string(),
                actual: format!("rows={rows} intermediate={}", self.intermediate_size),
            })?;
        let output_values = rows.checked_mul(hidden_size).ok_or_else(|| Error::Shape {
            label: "Gemma 4 dense MLP workspace",
            expected: "rows * hidden size without overflow".to_string(),
            actual: format!("rows={rows} hidden={hidden_size}"),
        })?;
        Ok(Gemma4MlpWorkspace {
            rows,
            gate: DeviceBuffer::zeroed(values)?,
            up: DeviceBuffer::zeroed(values)?,
            activated: DeviceBuffer::zeroed(values)?,
            output: DeviceBuffer::zeroed(output_values)?,
        })
    }

    /// Applies `down(gelu_tanh(gate(x)) * up(x))` for every input row.
    pub fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Gemma4MlpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = workspace.rows;
        let hidden_size = self.gate.shape().1;
        if input.len() != rows * hidden_size {
            return Err(Error::Shape {
                label: "Gemma 4 dense MLP input",
                expected: format!("{} values", rows * hidden_size),
                actual: input.len().to_string(),
            });
        }
        self.gate
            .run_rows_into(input, &mut workspace.gate, rows, stream)?;
        self.up
            .run_rows_into(input, &mut workspace.up, rows, stream)?;
        gelu_tanh_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.up,
            workspace.activated.output(),
            stream,
        )?;
        self.down
            .run_rows_into(&workspace.activated, &mut workspace.output, rows, stream)
    }

    /// Returns the most recent MLP output in `workspace`.
    pub fn output<'a>(&self, workspace: &'a Gemma4MlpWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

impl Gemma4Router {
    /// Loads the calibrated MoE router for `layer`.
    pub fn load(checkpoint: &Gemma4Checkpoint, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.router");
        let projection = Gemma4Linear::load(checkpoint, &format!("{prefix}.proj.weight"))?;
        if projection.shape() != (checkpoint.config.num_experts, checkpoint.config.hidden_size) {
            return Err(Error::Shape {
                label: "Gemma 4 router projection",
                expected: format!(
                    "{}x{}",
                    checkpoint.config.num_experts, checkpoint.config.hidden_size
                ),
                actual: format!("{:?}", projection.shape()),
            });
        }
        let router_scale = DeviceBuffer::from_host(
            &checkpoint
                .load_bf16_vector_f32(&format!("{prefix}.scale"), checkpoint.config.hidden_size)?,
        )?;
        let per_expert_scale = DeviceBuffer::from_host(&checkpoint.load_bf16_vector_f32(
            &format!("{prefix}.per_expert_scale"),
            checkpoint.config.num_experts,
        )?)?;
        let input_norm_scalar_value = 1.0 / (checkpoint.config.hidden_size as f32).sqrt();
        Ok(Self {
            input_norm_weight: DeviceBuffer::from_host(&vec![1.0; checkpoint.config.hidden_size])?,
            input_norm_scalar: DeviceBuffer::from_host(&[input_norm_scalar_value])?,
            input_norm_scalar_value,
            router_scale,
            projection,
            per_expert_scale,
            rms_norm_eps: checkpoint.config.rms_norm_eps,
            top_k: checkpoint.config.top_k_experts,
        })
    }

    /// Allocates device buffers for one route selection.
    pub fn new_workspace(&self) -> Result<Gemma4RouterWorkspace> {
        let (experts, hidden_size) = self.projection.shape();
        Ok(Gemma4RouterWorkspace {
            normalized: DeviceBuffer::zeroed(hidden_size)?,
            logits: DeviceBuffer::zeroed(experts)?,
            indices: DeviceBuffer::zeroed(self.top_k)?,
            normalized_weights: DeviceBuffer::zeroed(self.top_k)?,
            route_weights: DeviceBuffer::zeroed(self.top_k)?,
        })
    }

    /// Computes Gemma's calibrated top-k route entirely on the device.
    pub fn run(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Gemma4RouterWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let (_, hidden_size) = self.projection.shape();
        if input.len() != hidden_size
            || workspace.normalized.len() != hidden_size
            || workspace.logits.len() != self.per_expert_scale.len()
            || workspace.indices.len() != self.top_k
            || workspace.normalized_weights.len() != self.top_k
            || workspace.route_weights.len() != self.top_k
        {
            return Err(Error::Shape {
                label: "Gemma 4 router workspace",
                expected: format!(
                    "input/normalized={hidden_size} logits={} route buffers={}",
                    self.per_expert_scale.len(),
                    self.top_k
                ),
                actual: format!(
                    "input={} normalized={} logits={} indices={} normalized_weights={} route_weights={}",
                    input.len(),
                    workspace.normalized.len(),
                    workspace.logits.len(),
                    workspace.indices.len(),
                    workspace.normalized_weights.len(),
                    workspace.route_weights.len(),
                ),
            });
        }
        rms_norm_f32_into_on_stream(
            1,
            hidden_size,
            input,
            &self.input_norm_weight,
            workspace.normalized.output(),
            self.rms_norm_eps,
            stream,
        )?;
        scale_channel_f32_device_scalar_in_place_on_stream(
            workspace.normalized.inout(),
            &self.router_scale,
            &self.input_norm_scalar,
            stream,
        )?;
        self.projection
            .run_rows_into(&workspace.normalized, &mut workspace.logits, 1, stream)?;
        moe_topk_f32_into_on_stream(
            &workspace.logits,
            workspace.indices.output(),
            workspace.normalized_weights.output(),
            self.top_k,
            true,
            stream,
        )?;
        gather_indexed_mul_f32_into_on_stream(
            &self.per_expert_scale,
            &workspace.indices,
            &workspace.normalized_weights,
            workspace.route_weights.output(),
            stream,
        )
    }

    /// Selected expert indices for the last [`Self::run`] call.
    pub fn indices<'a>(&self, workspace: &'a Gemma4RouterWorkspace) -> &'a DeviceBuffer<u32> {
        &workspace.indices
    }

    /// Calibrated selected expert weights for the last [`Self::run`] call.
    pub fn route_weights<'a>(&self, workspace: &'a Gemma4RouterWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.route_weights
    }

    fn device_bytes(&self) -> usize {
        self.input_norm_weight.device_bytes()
            + self.input_norm_scalar.device_bytes()
            + self.router_scale.device_bytes()
            + self.projection.device_bytes()
            + self.per_expert_scale.device_bytes()
    }
}

impl Gemma4Expert {
    /// Loads one expert from stacked BF16 or per-expert ModelOpt tensors.
    pub fn load(checkpoint: &Gemma4Checkpoint, layer: usize, expert: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.experts");
        let native_gate = format!("{prefix}.{expert}.gate_proj.weight");
        let (gate, up, down) = if checkpoint.checkpoint.contains_tensor(&native_gate) {
            let gate = checkpoint
                .checkpoint
                .load_nvfp4_linear(&format!("{prefix}.{expert}.gate_proj"))?;
            let up = checkpoint
                .checkpoint
                .load_nvfp4_linear(&format!("{prefix}.{expert}.up_proj"))?;
            (
                Gemma4Linear::from_modelopt(gate)?,
                Gemma4Linear::from_modelopt(up)?,
                Gemma4Linear::load(checkpoint, &format!("{prefix}.{expert}.down_proj.weight"))?,
            )
        } else {
            let gate_up =
                checkpoint.load_expert_linear_nvfp4(&format!("{prefix}.gate_up_proj"), expert)?;
            let (gate, up) = split_modelopt_out_features(
                gate_up,
                checkpoint.config.moe_intermediate_size,
                format!("{prefix}.gate_proj[{expert}]"),
                format!("{prefix}.up_proj[{expert}]"),
            )?;
            (
                Gemma4Linear::from_modelopt(gate)?,
                Gemma4Linear::from_modelopt(up)?,
                Gemma4Linear::load_expert(checkpoint, &format!("{prefix}.down_proj"), expert)?,
            )
        };
        let (intermediate_size, hidden_size) = gate.shape();
        if hidden_size != checkpoint.config.hidden_size
            || intermediate_size != checkpoint.config.moe_intermediate_size
            || up.shape() != (intermediate_size, hidden_size)
            || down.shape() != (hidden_size, intermediate_size)
        {
            return Err(Error::Shape {
                label: "Gemma 4 expert projections",
                expected: format!(
                    "gate/up={}x{}, down={}x{}",
                    checkpoint.config.moe_intermediate_size,
                    checkpoint.config.hidden_size,
                    checkpoint.config.hidden_size,
                    checkpoint.config.moe_intermediate_size,
                ),
                actual: format!(
                    "gate={:?} up={:?} down={:?}",
                    gate.shape(),
                    up.shape(),
                    down.shape()
                ),
            });
        }
        Ok(Self { gate, up, down })
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

impl Gemma4Moe {
    /// Loads the router and every resident expert for `layer`.
    pub fn load(checkpoint: &Gemma4Checkpoint, layer: usize) -> Result<Self> {
        let router = Gemma4Router::load(checkpoint, layer)?;
        let mut experts = Vec::with_capacity(checkpoint.config.num_experts);
        for expert in 0..checkpoint.config.num_experts {
            experts.push(Gemma4Expert::load(checkpoint, layer, expert)?);
        }
        let mut gate_packed = Vec::with_capacity(experts.len());
        let mut gate_scales = Vec::with_capacity(experts.len());
        let mut gate_tiled_scales = Vec::with_capacity(experts.len());
        let mut gate_scale_2 = Vec::with_capacity(experts.len());
        let mut up_packed = Vec::with_capacity(experts.len());
        let mut up_scales = Vec::with_capacity(experts.len());
        let mut up_tiled_scales = Vec::with_capacity(experts.len());
        let mut up_scale_2 = Vec::with_capacity(experts.len());
        let mut down_packed = Vec::with_capacity(experts.len());
        let mut down_scales = Vec::with_capacity(experts.len());
        let mut down_tiled_scales = Vec::with_capacity(experts.len());
        let mut down_scale_2 = Vec::with_capacity(experts.len());
        for expert in &experts {
            let (packed, scales, scale_2) = expert.gate.nvfp4_parts()?;
            gate_packed.push(packed);
            gate_scales.push(scales);
            gate_scale_2.push(scale_2);
            let (_, scales, _) = expert.gate.grouped_gemm_parts();
            gate_tiled_scales.push(scales);
            let (packed, scales, scale_2) = expert.up.nvfp4_parts()?;
            up_packed.push(packed);
            up_scales.push(scales);
            up_scale_2.push(scale_2);
            let (_, scales, _) = expert.up.grouped_gemm_parts();
            up_tiled_scales.push(scales);
            let (packed, scales, scale_2) = expert.down.nvfp4_parts()?;
            down_packed.push(packed);
            down_scales.push(scales);
            down_scale_2.push(scale_2);
            let (_, scales, _) = expert.down.grouped_gemm_parts();
            down_tiled_scales.push(scales);
        }
        let intermediate_size = checkpoint.config.moe_intermediate_size;
        let hidden_size = checkpoint.config.hidden_size;
        let mut gate_scale_2 = DeviceBuffer::from_host(&gate_scale_2)?;
        let gate_alpha_table = scalar_pointer_table(&mut gate_scale_2)?;
        let mut up_scale_2 = DeviceBuffer::from_host(&up_scale_2)?;
        let up_alpha_table = scalar_pointer_table(&mut up_scale_2)?;
        let mut down_scale_2 = DeviceBuffer::from_host(&down_scale_2)?;
        let down_alpha_table = scalar_pointer_table(&mut down_scale_2)?;
        Ok(Self {
            router,
            gate_packed_table: DeviceBuffer::from_host(&gate_packed)?,
            gate_scale_table: DeviceBuffer::from_host(&gate_scales)?,
            gate_tiled_scale_table: DeviceBuffer::from_host(&gate_tiled_scales)?,
            gate_scale_2,
            gate_alpha_table,
            up_packed_table: DeviceBuffer::from_host(&up_packed)?,
            up_scale_table: DeviceBuffer::from_host(&up_scales)?,
            up_tiled_scale_table: DeviceBuffer::from_host(&up_tiled_scales)?,
            up_scale_2,
            up_alpha_table,
            down_packed_table: DeviceBuffer::from_host(&down_packed)?,
            down_scale_table: DeviceBuffer::from_host(&down_scales)?,
            down_tiled_scale_table: DeviceBuffer::from_host(&down_tiled_scales)?,
            down_scale_2,
            down_alpha_table,
            expert_alpha: DeviceBuffer::from_host(&vec![1.0; experts.len()])?,
            _experts: experts,
            intermediate_size,
            hidden_size,
        })
    }

    /// Allocates one-token MoE workspace.
    pub fn new_workspace(&self) -> Result<Gemma4MoeWorkspace> {
        let routes = self.router.top_k;
        let gate = (0..routes)
            .map(|_| DeviceBuffer::zeroed(self.intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        let up = (0..routes)
            .map(|_| DeviceBuffer::zeroed(self.intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        let activated = (0..routes)
            .map(|_| DeviceBuffer::zeroed(self.intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        let down = (0..routes)
            .map(|_| DeviceBuffer::zeroed(self.hidden_size))
            .collect::<Result<Vec<_>>>()?;
        Ok(Gemma4MoeWorkspace {
            router: self.router.new_workspace()?,
            gate_output_table: mutable_f32_pointer_table(&gate)?,
            up_output_table: mutable_f32_pointer_table(&up)?,
            down_input_table: const_f32_pointer_table(&activated)?,
            down_output_table: mutable_f32_pointer_table(&down)?,
            down_result_table: const_f32_pointer_table(&down)?,
            gate,
            up,
            activated,
            down,
            output: DeviceBuffer::zeroed(self.hidden_size)?,
        })
    }

    /// Runs one token through the selected experts without host synchronization.
    pub fn run_into(
        &self,
        router_input: &DeviceBuffer<f32>,
        expert_input: &DeviceBuffer<f32>,
        workspace: &mut Gemma4MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.router
            .run(router_input, &mut workspace.router, stream)?;
        if expert_input.len() != self.hidden_size
            || workspace.gate.len() != self.router.top_k
            || workspace.up.len() != self.router.top_k
            || workspace.activated.len() != self.router.top_k
            || workspace.down.len() != self.router.top_k
        {
            return Err(Error::Shape {
                label: "Gemma 4 grouped MoE workspace",
                expected: format!(
                    "input={} and {} gate/up, activation, and down slots",
                    self.hidden_size, self.router.top_k
                ),
                actual: format!(
                    "input={} gate={} up={} activated={} down={}",
                    expert_input.len(),
                    workspace.gate.len(),
                    workspace.up.len(),
                    workspace.activated.len(),
                    workspace.down.len()
                ),
            });
        }
        let indices = self.router.indices(&workspace.router);
        nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
            indices,
            expert_input,
            &self.gate_packed_table,
            &self.gate_scale_table,
            &self.gate_scale_2,
            &workspace.gate_output_table,
            self.intermediate_size,
            self.hidden_size,
            stream,
        )?;
        nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
            indices,
            expert_input,
            &self.up_packed_table,
            &self.up_scale_table,
            &self.up_scale_2,
            &workspace.up_output_table,
            self.intermediate_size,
            self.hidden_size,
            stream,
        )?;
        for ((gate, up), activated) in workspace
            .gate
            .iter()
            .zip(&workspace.up)
            .zip(&mut workspace.activated)
        {
            gelu_tanh_mul_f32_into_on_stream(gate, up, activated.output(), stream)?;
        }
        nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
            indices,
            &workspace.down_input_table,
            &self.down_packed_table,
            &self.down_scale_table,
            &self.down_scale_2,
            &workspace.down_output_table,
            self.hidden_size,
            self.intermediate_size,
            stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            indices,
            self.router.route_weights(&workspace.router),
            &workspace.down_result_table,
            &self.expert_alpha,
            workspace.output.inout(),
            stream,
        )
    }

    /// Returns the most recent MoE output in `workspace`.
    pub fn output<'a>(&self, workspace: &'a Gemma4MoeWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self
                ._experts
                .iter()
                .map(Gemma4Expert::device_bytes)
                .sum::<usize>()
            + self.gate_packed_table.device_bytes()
            + self.gate_scale_table.device_bytes()
            + self.gate_tiled_scale_table.device_bytes()
            + self.gate_scale_2.device_bytes()
            + self.gate_alpha_table.device_bytes()
            + self.up_packed_table.device_bytes()
            + self.up_scale_table.device_bytes()
            + self.up_tiled_scale_table.device_bytes()
            + self.up_scale_2.device_bytes()
            + self.up_alpha_table.device_bytes()
            + self.down_packed_table.device_bytes()
            + self.down_scale_table.device_bytes()
            + self.down_tiled_scale_table.device_bytes()
            + self.down_scale_2.device_bytes()
            + self.down_alpha_table.device_bytes()
            + self.expert_alpha.device_bytes()
    }
}

fn const_f32_pointer_table(buffers: &[DeviceBuffer<f32>]) -> Result<DeviceBuffer<*const f32>> {
    DeviceBuffer::from_host(
        &buffers
            .iter()
            .map(|buffer| buffer.as_const_ptr().cast::<f32>())
            .collect::<Vec<_>>(),
    )
}

fn mutable_f32_pointer_table(buffers: &[DeviceBuffer<f32>]) -> Result<DeviceBuffer<*mut f32>> {
    DeviceBuffer::from_host(
        &buffers
            .iter()
            .map(|buffer| buffer.as_const_ptr().cast_mut().cast::<f32>())
            .collect::<Vec<_>>(),
    )
}

fn scalar_pointer_table(buffer: &mut DeviceBuffer<f32>) -> Result<DeviceBuffer<*mut f32>> {
    let base = buffer.as_const_ptr().cast::<f32>().cast_mut();
    DeviceBuffer::from_host(
        &(0..buffer.len())
            .map(|index| unsafe { base.add(index) })
            .collect::<Vec<_>>(),
    )
}

fn split_modelopt_out_features(
    mut weight: ModelOptNvfp4Linear,
    first_out_features: usize,
    first_prefix: String,
    second_prefix: String,
) -> Result<(ModelOptNvfp4Linear, ModelOptNvfp4Linear)> {
    if first_out_features == 0 || first_out_features >= weight.out_features {
        return Err(Error::Shape {
            label: "Gemma 4 gate/up split",
            expected: format!("0 < first rows < {}", weight.out_features),
            actual: first_out_features.to_string(),
        });
    }
    let packed_per_row = weight.in_features / 2;
    let scales_per_row = weight.in_features / 16;
    let packed_split = first_out_features * packed_per_row;
    let scale_split = first_out_features * scales_per_row;
    if weight.packed_weight.len() != weight.out_features * packed_per_row
        || weight.weight_scale.len() != weight.out_features * scales_per_row
    {
        return Err(Error::Shape {
            label: "Gemma 4 gate/up ModelOpt storage",
            expected: format!(
                "packed={} scales={}",
                weight.out_features * packed_per_row,
                weight.out_features * scales_per_row
            ),
            actual: format!(
                "packed={} scales={}",
                weight.packed_weight.len(),
                weight.weight_scale.len()
            ),
        });
    }
    let second_packed = weight.packed_weight.split_off(packed_split);
    let second_scales = weight.weight_scale.split_off(scale_split);
    let second_out_features = weight.out_features - first_out_features;
    let second = ModelOptNvfp4Linear {
        prefix: second_prefix,
        out_features: second_out_features,
        in_features: weight.in_features,
        packed_weight: second_packed,
        weight_scale: second_scales,
        weight_scale_2: weight.weight_scale_2,
        input_scale: weight.input_scale,
    };
    weight.prefix = first_prefix;
    weight.out_features = first_out_features;
    Ok((weight, second))
}

impl Gemma4MlpWorkspace {
    fn device_bytes(&self) -> usize {
        self.gate.device_bytes()
            + self.up.device_bytes()
            + self.activated.device_bytes()
            + self.output.device_bytes()
    }
}

impl Gemma4RouterWorkspace {
    fn device_bytes(&self) -> usize {
        self.normalized.device_bytes()
            + self.logits.device_bytes()
            + self.indices.device_bytes()
            + self.normalized_weights.device_bytes()
            + self.route_weights.device_bytes()
    }
}

impl Gemma4MoeWorkspace {
    fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self
                .gate
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
            + self
                .up
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
            + self
                .activated
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
            + self
                .down
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
            + self.gate_output_table.device_bytes()
            + self.up_output_table.device_bytes()
            + self.down_input_table.device_bytes()
            + self.down_output_table.device_bytes()
            + self.down_result_table.device_bytes()
            + self.output.device_bytes()
    }
}

impl Gemma4AttentionWorkspace {
    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_normed.device_bytes()
            + self.k_normed.device_bytes()
            + self.v_normed.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attended.device_bytes()
            + self.output.device_bytes()
    }
}

impl Gemma4DecoderLayerWorkspace {
    fn device_bytes(&self) -> usize {
        self.attention.device_bytes()
            + self.dense.device_bytes()
            + self.moe.device_bytes()
            + self.normalized.device_bytes()
            + self.residual.device_bytes()
            + self.dense_input.device_bytes()
            + self.dense_output.device_bytes()
            + self.moe_input.device_bytes()
            + self.moe_output.device_bytes()
            + self.combined.device_bytes()
            + self.output.device_bytes()
    }
}

impl Gemma4RmsNorm {
    /// Loads one learned Gemma RMSNorm vector.
    pub fn load(checkpoint: &Gemma4Checkpoint, tensor: &str, width: usize) -> Result<Self> {
        Self::load_scaled(checkpoint, tensor, width, 1.0)
    }

    fn load_scaled(
        checkpoint: &Gemma4Checkpoint,
        tensor: &str,
        width: usize,
        scale: f32,
    ) -> Result<Self> {
        let mut weight = checkpoint.load_bf16_vector_f32(tensor, width)?;
        for value in &mut weight {
            *value *= scale;
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
            eps: checkpoint.config.rms_norm_eps,
        })
    }

    /// Normalizes `rows` contiguous vectors of `width` values.
    pub fn run_into(
        &self,
        rows: usize,
        width: usize,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.weight.len() != width {
            return Err(Error::Shape {
                label: "Gemma 4 RMSNorm weight",
                expected: format!("{width} values"),
                actual: self.weight.len().to_string(),
            });
        }
        rms_norm_f32_into_on_stream(
            rows,
            width,
            input,
            &self.weight,
            output.output(),
            self.eps,
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

impl Gemma4Attention {
    /// Loads local or global GQA weights for `layer`.
    pub fn load(checkpoint: &Gemma4Checkpoint, layer: usize) -> Result<Self> {
        let config = checkpoint.config();
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let global = config.is_full_attention_layer(layer)?;
        let kv_heads = if global {
            config.num_global_key_value_heads
        } else {
            config.num_key_value_heads
        };
        let head_dim = if global {
            config.global_head_dim
        } else {
            config.head_dim
        };
        let q = Gemma4Linear::load(checkpoint, &format!("{prefix}.q_proj.weight"))?;
        let k = Gemma4Linear::load(checkpoint, &format!("{prefix}.k_proj.weight"))?;
        let v_name = format!("{prefix}.v_proj.weight");
        let v = checkpoint
            .checkpoint
            .contains_tensor(&v_name)
            .then(|| Gemma4Linear::load(checkpoint, &v_name))
            .transpose()?;
        let output = Gemma4Linear::load(checkpoint, &format!("{prefix}.o_proj.weight"))?;
        let q_width = config.num_attention_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        if q.shape() != (q_width, config.hidden_size)
            || k.shape() != (kv_width, config.hidden_size)
            || output.shape() != (config.hidden_size, q_width)
            || (!global
                && v.as_ref().map(Gemma4Linear::shape) != Some((kv_width, config.hidden_size)))
            || (global && config.attention_k_eq_v && v.is_some())
        {
            return Err(Error::Shape {
                label: "Gemma 4 attention projections",
                expected: format!(
                    "q={}x{}, k/v={}x{}, o={}x{}{}",
                    q_width,
                    config.hidden_size,
                    kv_width,
                    config.hidden_size,
                    config.hidden_size,
                    q_width,
                    if global && config.attention_k_eq_v {
                        ", no v projection"
                    } else {
                        ""
                    },
                ),
                actual: format!(
                    "q={:?} k={:?} v={:?} o={:?}",
                    q.shape(),
                    k.shape(),
                    v.as_ref().map(Gemma4Linear::shape),
                    output.shape(),
                ),
            });
        }
        let rotary_dim = if global {
            (head_dim as f32 * config.full_partial_rotary_factor) as usize
        } else {
            head_dim
        };
        if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
            return Err(Error::Shape {
                label: "Gemma 4 attention rotary dimension",
                expected: "an even nonzero rotary dimension no larger than the head".to_string(),
                actual: format!("rotary_dim={rotary_dim} head_dim={head_dim}"),
            });
        }
        Ok(Self {
            q,
            k,
            v,
            output,
            // Compact attention applies 1/sqrt(head_dim), while Gemma 4's
            // normalized queries use an attention scale of 1.0.
            q_norm: Gemma4RmsNorm::load_scaled(
                checkpoint,
                &format!("{prefix}.q_norm.weight"),
                head_dim,
                (head_dim as f32).sqrt(),
            )?,
            k_norm: Gemma4RmsNorm::load(checkpoint, &format!("{prefix}.k_norm.weight"), head_dim)?,
            value_norm_weight: DeviceBuffer::from_host(&vec![1.0; head_dim])?,
            q_heads: config.num_attention_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta: if global {
                config.full_rope_theta
            } else {
                config.local_rope_theta
            },
            window: (!global).then_some(config.sliding_window),
        })
    }

    /// Allocates compact-cache attention workspace for this layer.
    pub fn new_compact_attention_workspace(
        &self,
        max_tokens: usize,
    ) -> Result<Sm12xKvAttentionWorkspace> {
        Sm12xKvAttentionWorkspace::new_gqa(max_tokens, self.q_heads, self.kv_heads, self.head_dim)
    }

    /// Allocates reusable one-token attention buffers.
    pub fn new_workspace(&self) -> Result<Gemma4AttentionWorkspace> {
        let q_width = self.q_heads * self.head_dim;
        let kv_width = self.kv_heads * self.head_dim;
        Ok(Gemma4AttentionWorkspace {
            q: DeviceBuffer::zeroed(q_width)?,
            k: DeviceBuffer::zeroed(kv_width)?,
            v: DeviceBuffer::zeroed(kv_width)?,
            q_normed: DeviceBuffer::zeroed(q_width)?,
            k_normed: DeviceBuffer::zeroed(kv_width)?,
            v_normed: DeviceBuffer::zeroed(kv_width)?,
            q_rope: DeviceBuffer::zeroed(q_width)?,
            k_rope: DeviceBuffer::zeroed(kv_width)?,
            attended: DeviceBuffer::zeroed(q_width)?,
            output: DeviceBuffer::zeroed(self.output.shape().0)?,
        })
    }

    /// Runs one token, appending this layer's K/V and reading its own cache.
    #[allow(clippy::too_many_arguments)]
    fn run_decode_into(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Gemma4AttentionWorkspace,
        cache: Gemma4LayerCache<'_>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.q.shape().1 {
            return Err(Error::Shape {
                label: "Gemma 4 decode attention inputs",
                expected: format!("input={} and cache position={position}", self.q.shape().1),
                actual: format!("input={}", input.len()),
            });
        }
        self.q.run_rows_into(input, &mut workspace.q, 1, stream)?;
        self.k.run_rows_into(input, &mut workspace.k, 1, stream)?;
        if let Some(v) = &self.v {
            v.run_rows_into(input, &mut workspace.v, 1, stream)?;
        }
        self.q_norm.run_into(
            self.q_heads,
            self.head_dim,
            &workspace.q,
            &mut workspace.q_normed,
            stream,
        )?;
        self.k_norm.run_into(
            self.kv_heads,
            self.head_dim,
            &workspace.k,
            &mut workspace.k_normed,
            stream,
        )?;
        let value_input = self.v.as_ref().map_or(&workspace.k, |_| &workspace.v);
        rms_norm_f32_into_on_stream(
            self.kv_heads,
            self.head_dim,
            value_input,
            &self.value_norm_weight,
            workspace.v_normed.output(),
            self.q_norm.eps,
            stream,
        )?;
        self.apply_rope(
            self.q_heads,
            &workspace.q_normed,
            workspace.q_rope.output(),
            position,
            stream,
        )?;
        self.apply_rope(
            self.kv_heads,
            &workspace.k_normed,
            workspace.k_rope.output(),
            position,
            stream,
        )?;
        cache.pool.append_at_offsets_on_stream(
            cache.page_slot,
            cache.page_offset,
            &workspace.k_rope,
            0,
            &workspace.v_normed,
            0,
            stream,
        )?;
        let cache_len = position + 1;
        let window_start = self
            .window
            .map_or(0, |window| cache_len.saturating_sub(window));
        cache
            .attention
            .attention_paged_window_offsets_into_on_stream(
                cache.pool,
                cache.page_table,
                cache_len,
                &workspace.q_rope,
                0,
                workspace.attended.output(),
                0,
                window_start,
                stream,
            )?;
        self.output
            .run_rows_into(&workspace.attended, &mut workspace.output, 1, stream)
    }

    fn apply_rope(
        &self,
        rows: usize,
        input: &DeviceBuffer<f32>,
        output: eider_cuda::DeviceOutput<'_, f32>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.rotary_dim == self.head_dim {
            rope_neox_f32_into_on_stream(
                rows,
                self.head_dim,
                input,
                output,
                position,
                self.rope_theta,
                stream,
            )
        } else {
            rope_neox_proportional_f32_into_on_stream(
                rows,
                self.head_dim,
                self.rotary_dim,
                input,
                output,
                position,
                self.rope_theta,
                stream,
            )
        }
    }

    /// Returns the latest projected attention output in `workspace`.
    pub fn output<'a>(&self, workspace: &'a Gemma4AttentionWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.as_ref().map_or(0, Gemma4Linear::device_bytes)
            + self.output.device_bytes()
            + self.q_norm.device_bytes()
            + self.k_norm.device_bytes()
            + self.value_norm_weight.device_bytes()
    }
}

impl Gemma4DecoderLayer {
    /// Loads all attention, dense, and MoE weights for `layer`.
    pub fn load(checkpoint: &Gemma4Checkpoint, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}");
        let hidden_size = checkpoint.config.hidden_size;
        let layer_scalar = checkpoint.load_bf16_vector_f32(&format!("{prefix}.layer_scalar"), 1)?;
        Ok(Self {
            input_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                hidden_size,
            )?,
            attention: Gemma4Attention::load(checkpoint, layer)?,
            post_attention_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden_size,
            )?,
            dense_input_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
                hidden_size,
            )?,
            dense: Gemma4Mlp::load(checkpoint, layer)?,
            dense_post_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm_1.weight"),
                hidden_size,
            )?,
            moe_input_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.pre_feedforward_layernorm_2.weight"),
                hidden_size,
            )?,
            moe: Gemma4Moe::load(checkpoint, layer)?,
            moe_post_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm_2.weight"),
                hidden_size,
            )?,
            post_feedforward_norm: Gemma4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                hidden_size,
            )?,
            layer_scale_channels: DeviceBuffer::from_host(&vec![1.0; hidden_size])?,
            layer_scalar: DeviceBuffer::from_host(&layer_scalar)?,
            layer_scalar_value: layer_scalar[0],
        })
    }

    /// Allocates one-token layer state with no per-token allocations.
    pub fn new_workspace(&self) -> Result<Gemma4DecoderLayerWorkspace> {
        let hidden_size = self.attention.q.shape().1;
        Ok(Gemma4DecoderLayerWorkspace {
            attention: self.attention.new_workspace()?,
            dense: self.dense.new_workspace(1)?,
            moe: self.moe.new_workspace()?,
            normalized: DeviceBuffer::zeroed(hidden_size)?,
            residual: DeviceBuffer::zeroed(hidden_size)?,
            dense_input: DeviceBuffer::zeroed(hidden_size)?,
            dense_output: DeviceBuffer::zeroed(hidden_size)?,
            moe_input: DeviceBuffer::zeroed(hidden_size)?,
            moe_output: DeviceBuffer::zeroed(hidden_size)?,
            combined: DeviceBuffer::zeroed(hidden_size)?,
            output: DeviceBuffer::zeroed(hidden_size)?,
        })
    }

    /// Runs one decoder layer at `position` using its own heterogeneous cache.
    #[allow(clippy::too_many_arguments)]
    fn run_decode_into(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Gemma4DecoderLayerWorkspace,
        cache: Gemma4LayerCache<'_>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden_size = self.attention.q.shape().1;
        if input.len() != hidden_size {
            return Err(Error::Shape {
                label: "Gemma 4 decoder layer input",
                expected: format!("{hidden_size} values"),
                actual: input.len().to_string(),
            });
        }
        self.input_norm
            .run_into(1, hidden_size, input, &mut workspace.normalized, stream)?;
        self.attention.run_decode_into(
            &workspace.normalized,
            &mut workspace.attention,
            cache,
            position,
            stream,
        )?;
        self.post_attention_norm.run_into(
            1,
            hidden_size,
            self.attention.output(&workspace.attention),
            &mut workspace.normalized,
            stream,
        )?;
        add_f32_into_on_stream(
            input,
            &workspace.normalized,
            workspace.residual.output(),
            stream,
        )?;

        self.dense_input_norm.run_into(
            1,
            hidden_size,
            &workspace.residual,
            &mut workspace.dense_input,
            stream,
        )?;
        self.dense
            .run_into(&workspace.dense_input, &mut workspace.dense, stream)?;
        self.dense_post_norm.run_into(
            1,
            hidden_size,
            self.dense.output(&workspace.dense),
            &mut workspace.dense_output,
            stream,
        )?;

        self.moe_input_norm.run_into(
            1,
            hidden_size,
            &workspace.residual,
            &mut workspace.moe_input,
            stream,
        )?;
        self.moe.run_into(
            &workspace.residual,
            &workspace.moe_input,
            &mut workspace.moe,
            stream,
        )?;
        self.moe_post_norm.run_into(
            1,
            hidden_size,
            self.moe.output(&workspace.moe),
            &mut workspace.moe_output,
            stream,
        )?;
        add_f32_into_on_stream(
            &workspace.dense_output,
            &workspace.moe_output,
            workspace.combined.output(),
            stream,
        )?;
        self.post_feedforward_norm.run_into(
            1,
            hidden_size,
            &workspace.combined,
            &mut workspace.normalized,
            stream,
        )?;
        add_f32_into_on_stream(
            &workspace.residual,
            &workspace.normalized,
            workspace.output.output(),
            stream,
        )?;
        scale_channel_f32_device_scalar_in_place_on_stream(
            workspace.output.inout(),
            &self.layer_scale_channels,
            &self.layer_scalar,
            stream,
        )
    }

    /// Returns this layer's most recent output.
    pub fn output<'a>(&self, workspace: &'a Gemma4DecoderLayerWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    fn device_bytes(&self) -> usize {
        self.input_norm.device_bytes()
            + self.attention.device_bytes()
            + self.post_attention_norm.device_bytes()
            + self.dense_input_norm.device_bytes()
            + self.dense.device_bytes()
            + self.dense_post_norm.device_bytes()
            + self.moe_input_norm.device_bytes()
            + self.moe.device_bytes()
            + self.moe_post_norm.device_bytes()
            + self.post_feedforward_norm.device_bytes()
            + self.layer_scale_channels.device_bytes()
            + self.layer_scalar.device_bytes()
    }
}

impl Gemma4Model {
    /// Loads every Gemma 4 text layer and its tied BF16 embedding/LM head.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = Gemma4Checkpoint::open(model_dir)?;
        let config = checkpoint.config.clone();
        if config.vocab_size == 0 {
            return Err(Error::Shape {
                label: "Gemma 4 vocabulary",
                expected: "a nonzero vocabulary size".to_string(),
                actual: config.vocab_size.to_string(),
            });
        }
        let embedding = checkpoint.load_bf16_matrix_device(
            "model.language_model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(Gemma4DecoderLayer::load(&checkpoint, layer)?);
        }
        let embedding_scalar_value = eider_cuda::format::bf16_to_f32(
            eider_cuda::format::f32_to_bf16((config.hidden_size as f32).sqrt()),
        );
        Ok(Self {
            embedding,
            embedding_channel_scale: DeviceBuffer::from_host(&vec![1.0; config.hidden_size])?,
            embedding_scalar: DeviceBuffer::from_host(&[embedding_scalar_value])?,
            embedding_scalar_value,
            final_norm: Gemma4RmsNorm::load(
                &checkpoint,
                "model.language_model.norm.weight",
                config.hidden_size,
            )?,
            config,
            layers,
        })
    }

    /// Returns the token vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Returns bytes retained by model weights and constant device buffers.
    pub fn device_bytes(&self) -> usize {
        self.embedding.device_bytes()
            + self.embedding_channel_scale.device_bytes()
            + self.embedding_scalar.device_bytes()
            + self
                .layers
                .iter()
                .map(Gemma4DecoderLayer::device_bytes)
                .sum::<usize>()
            + self.final_norm.device_bytes()
    }

    /// Allocates request-private execution state for one sequence.
    pub fn new_sequence_state(&self, max_tokens: usize) -> Result<Gemma4DecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_position_embeddings {
            return Err(Error::Shape {
                label: "Gemma 4 decode capacity",
                expected: format!("1..={}", self.config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(layer.new_workspace()?);
        }
        let compact_attention = Gemma4CompactAttentionWorkspaces::new(&self.layers, max_tokens)?;
        Ok(Gemma4DecodeState {
            hidden: DeviceBuffer::zeroed(self.config.hidden_size)?,
            layers,
            compact_attention,
            lm_logits: DeviceBuffer::zeroed(self.config.vocab_size)?,
            lm_top1_scratch_index: DeviceBuffer::zeroed(self.config.vocab_size)?,
            lm_argmax: DeviceBuffer::zeroed(1)?,
            lm_argmax_value: DeviceBuffer::zeroed(1)?,
            position: 0,
            max_tokens,
        })
    }

    pub(crate) fn sequence_layer_geometries(
        &self,
    ) -> impl Iterator<Item = Option<(usize, usize)>> + '_ {
        self.layers
            .iter()
            .map(|layer| Some((layer.attention.kv_heads, layer.attention.head_dim)))
    }

    /// Enqueues prompt tokens without materializing vocabulary logits.
    pub fn prefill_tokens(
        &self,
        sequence: &mut Gemma4Sequence,
        tokens: &[u32],
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
    ) -> Result<()> {
        for &token in tokens {
            self.forward_token(sequence, token, Gemma4PrefillOutput::None, stream, cache)?;
        }
        Ok(())
    }

    /// Runs one token and leaves tied-language-head logits in `state`.
    pub fn forward_one(
        &self,
        sequence: &mut Gemma4Sequence,
        token: u32,
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
    ) -> Result<()> {
        self.forward_token(
            sequence,
            token,
            Gemma4PrefillOutput::FullLogits,
            stream,
            cache,
        )
    }

    pub(crate) fn forward_one_top1(
        &self,
        sequence: &mut Gemma4Sequence,
        token: u32,
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
    ) -> Result<()> {
        self.forward_token(sequence, token, Gemma4PrefillOutput::Top1, stream, cache)
    }

    fn forward_token(
        &self,
        sequence: &mut Gemma4Sequence,
        token: u32,
        output: Gemma4PrefillOutput,
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
    ) -> Result<()> {
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                1,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(gemma4_cache_error)?;
        let result =
            self.forward_token_uncommitted(sequence, token, output, stream, cache, &reservation);
        if let Err(error) = result {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(gemma4_cache_error)?;
            return Err(error);
        }
        cache
            .commit_append(
                reservation,
                1,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(gemma4_cache_error)?;
        sequence.state.position += 1;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_token_uncommitted(
        &self,
        sequence: &mut Gemma4Sequence,
        token: u32,
        output: Gemma4PrefillOutput,
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
        reservation: &seqcache::AppendReservation,
    ) -> Result<()> {
        let state = &mut sequence.state;
        if token as usize >= self.config.vocab_size {
            return Err(Error::Shape {
                label: "Gemma 4 input token",
                expected: format!("token < {}", self.config.vocab_size),
                actual: token.to_string(),
            });
        }
        if state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "Gemma 4 decode position",
                expected: format!("position < {}", state.max_tokens),
                actual: state.position.to_string(),
            });
        }
        copy_bf16_row_to_f32_into_on_stream(
            self.config.vocab_size,
            self.config.hidden_size,
            token as usize,
            &self.embedding,
            state.hidden.output(),
            stream,
        )?;
        scale_channel_f32_device_scalar_in_place_on_stream(
            state.hidden.inout(),
            &self.embedding_channel_scale,
            &self.embedding_scalar,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(state.hidden.inout(), stream)?;
        for layer_index in 0..self.layers.len() {
            let local_attention = self.layers[layer_index].attention.window.is_some();
            let (previous, current) = state.layers.split_at_mut(layer_index);
            let input = if layer_index == 0 {
                &state.hidden
            } else {
                self.layers[layer_index - 1].output(&previous[layer_index - 1])
            };
            cache
                .with_append_pages(reservation, |backend, pages| {
                    let page = pages.iter().next().expect("one decode append page");
                    let segment = page.segment();
                    self.layers[layer_index].run_decode_into(
                        input,
                        &mut current[0],
                        Gemma4LayerCache {
                            pool: backend.pool_mut(layer_index)?,
                            page_slot: page.page().slot(),
                            page_offset: segment.page_offset(),
                            page_table: sequence.page_table.device(),
                            attention: state.compact_attention.for_layer_mut(local_attention)?,
                        },
                        state.position,
                        stream,
                    )
                })
                .map_err(gemma4_cache_error)?;
        }
        if output != Gemma4PrefillOutput::None {
            let final_input = self
                .layers
                .last()
                .expect("Gemma 4 model contains at least one layer")
                .output(state.layers.last().expect("Gemma 4 state has every layer"));
            self.final_norm.run_into(
                1,
                self.config.hidden_size,
                final_input,
                &mut state.hidden,
                stream,
            )?;
            match output {
                Gemma4PrefillOutput::None => unreachable!(),
                Gemma4PrefillOutput::FullLogits => bf16_linear_argmax_f32_into_on_stream(
                    &state.hidden,
                    &self.embedding,
                    state.lm_logits.output(),
                    state.lm_argmax.output(),
                    state.lm_argmax_value.output(),
                    self.config.vocab_size,
                    self.config.hidden_size,
                    stream,
                )?,
                Gemma4PrefillOutput::Top1 => lm_head_top1_f32_into_on_stream(
                    &state.hidden,
                    &self.embedding,
                    &state.lm_logits,
                    &state.lm_top1_scratch_index,
                    &state.lm_argmax,
                    &state.lm_argmax_value,
                    self.config.vocab_size,
                    self.config.hidden_size,
                    stream,
                )?,
            }
        }
        Ok(())
    }

    /// Returns the latest greedy token and its soft-capped logit.
    pub fn argmax_with_logit(
        &self,
        state: &Gemma4DecodeState,
        stream: &CudaStream,
    ) -> Result<(u32, f32)> {
        let token = state.lm_argmax.copy_to_host(stream)?[0];
        let logit = state.lm_argmax_value.copy_to_host(stream)?[0];
        Ok((token, self.softcap_logit(logit)))
    }

    /// Copies the latest soft-capped vocabulary logits to the host.
    pub fn logits_to_host(
        &self,
        state: &Gemma4DecodeState,
        stream: &CudaStream,
    ) -> Result<Vec<f32>> {
        let mut logits = state.lm_logits.copy_to_host(stream)?.to_vec();
        for logit in &mut logits {
            *logit = self.softcap_logit(*logit);
        }
        Ok(logits)
    }

    /// Decodes one token through all layers and returns the tied-LM-head argmax.
    pub fn decode_one(
        &self,
        sequence: &mut Gemma4Sequence,
        token: u32,
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
    ) -> Result<Gemma4NextToken> {
        self.forward_one(sequence, token, stream, cache)?;
        let (next_token, logit) = self.argmax_with_logit(&sequence.state, stream)?;
        Ok(Gemma4NextToken {
            input_token: token,
            token: next_token,
            logit,
        })
    }

    fn softcap_logit(&self, logit: f32) -> f32 {
        let cap = self.config.final_logit_softcapping;
        (logit / cap).tanh() * cap
    }
}

impl Gemma4DecodeState {
    /// Returns the number of tokens already processed into the K/V cache.
    pub fn len(&self) -> usize {
        self.position
    }

    /// Returns whether the sequence contains no processed tokens.
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Returns the allocated context capacity.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns bytes owned by this active sequence and its execution scratch.
    pub fn device_bytes(&self) -> usize {
        self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Gemma4DecoderLayerWorkspace::device_bytes)
                .sum::<usize>()
            + self.compact_attention.device_bytes()
            + self.lm_logits.device_bytes()
            + self.lm_top1_scratch_index.device_bytes()
            + self.lm_argmax.device_bytes()
            + self.lm_argmax_value.device_bytes()
    }
}

impl Gemma4Checkpoint {
    /// Opens the checkpoint index and model configuration.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        Ok(Self {
            config: Gemma4Config::open(model_dir)?,
            checkpoint: ModelOptCheckpoint::open(model_dir)?,
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    /// Loads one non-expert linear as ModelOpt-compatible NVFP4.
    ///
    /// `tensor` is the checkpoint tensor name, for example
    /// `model.language_model.layers.0.self_attn.q_proj.weight`.
    pub fn load_linear_nvfp4(&self, tensor: &str) -> Result<ModelOptNvfp4Linear> {
        if self.checkpoint.contains_tensor(tensor) {
            let info = self.checkpoint.tensor_info(tensor)?;
            if info.dtype == "BF16" {
                let [out_features, in_features] = matrix_shape(&info.shape, tensor)?;
                let values = self.read_bf16_tensor(tensor, out_features * in_features)?;
                return Ok(ModelOptNvfp4Linear::quantize_bf16(
                    tensor,
                    out_features,
                    in_features,
                    &values,
                )?);
            }
        }
        let prefix = self.native_prefix(tensor)?;
        Ok(self.checkpoint.load_nvfp4_linear(&prefix)?)
    }

    /// Loads one expert projection from a stacked Gemma MoE tensor as NVFP4.
    ///
    /// This reads and converts only the selected `[out, in]` slice. `tensor`
    /// is typically `model.language_model.layers.N.experts.gate_up_proj` or
    /// `model.language_model.layers.N.experts.down_proj`.
    pub fn load_expert_linear_nvfp4(
        &self,
        tensor: &str,
        expert: usize,
    ) -> Result<ModelOptNvfp4Linear> {
        if self.checkpoint.contains_tensor(tensor) {
            let info = self.checkpoint.tensor_info(tensor)?;
            if info.dtype == "BF16" {
                let [experts, out_features, in_features] = expert_shape(&info.shape, tensor)?;
                if expert >= experts {
                    return Err(Error::Shape {
                        label: "Gemma 4 expert index",
                        expected: format!("expert < {experts}"),
                        actual: expert.to_string(),
                    });
                }
                let values_per_expert = out_features * in_features;
                let values = self.read_bf16_tensor_range(
                    tensor,
                    expert * values_per_expert,
                    values_per_expert,
                )?;
                return Ok(ModelOptNvfp4Linear::quantize_bf16(
                    format!("{tensor}[{expert}]"),
                    out_features,
                    in_features,
                    &values,
                )?);
            }
        }
        let prefix = self.native_prefix(tensor)?;
        Ok(self.checkpoint.load_nvfp4_expert_linear(&prefix, expert)?)
    }

    /// Loads a BF16 vector as host F32 values, for normalization and routing.
    pub fn load_bf16_vector_f32(&self, tensor: &str, expected_len: usize) -> Result<Vec<f32>> {
        let info = self.checkpoint.tensor_info(tensor)?;
        if info.dtype != "BF16" || info.shape.as_slice() != [expected_len] {
            return Err(Error::Shape {
                label: "Gemma 4 BF16 vector",
                expected: format!("BF16 shape=[{expected_len}]"),
                actual: format!("dtype={} shape={:?} for {tensor}", info.dtype, info.shape),
            });
        }
        Ok(self
            .read_bf16_tensor(tensor, expected_len)?
            .into_iter()
            .map(eider_cuda::format::bf16_to_f32)
            .collect())
    }

    /// Uploads a BF16 matrix without converting it, for embeddings or the LM head.
    pub fn load_bf16_matrix_device(
        &self,
        tensor: &str,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceBuffer<u16>> {
        let info = self.checkpoint.tensor_info(tensor)?;
        if info.dtype != "BF16" || info.shape.as_slice() != [rows, cols] {
            return Err(Error::Shape {
                label: "Gemma 4 BF16 matrix",
                expected: format!("BF16 shape=[{rows},{cols}]"),
                actual: format!("dtype={} shape={:?} for {tensor}", info.dtype, info.shape),
            });
        }
        let values = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
            label: "Gemma 4 BF16 matrix",
            expected: "rows * cols without overflow".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
        DeviceBuffer::from_host(&self.read_bf16_tensor(tensor, values)?)
    }

    fn native_prefix(&self, tensor: &str) -> Result<String> {
        let without_weight = tensor.strip_suffix(".weight").unwrap_or(tensor);
        if self
            .checkpoint
            .contains_tensor(&format!("{without_weight}.weight_packed"))
            || self
                .checkpoint
                .contains_tensor(&format!("{without_weight}.weight_scale"))
        {
            return Ok(without_weight.to_string());
        }
        Err(Error::Format {
            label: "Gemma 4 NVFP4 linear",
            detail: format!("{tensor} is neither BF16 nor a supported ModelOpt NVFP4 prefix"),
        })
    }

    fn read_bf16_tensor(&self, tensor: &str, values: usize) -> Result<Vec<u16>> {
        self.read_bf16_tensor_range(tensor, 0, values)
    }

    fn read_bf16_tensor_range(
        &self,
        tensor: &str,
        value_offset: usize,
        values: usize,
    ) -> Result<Vec<u16>> {
        let bytes = values.checked_mul(2).ok_or_else(|| Error::Shape {
            label: "Gemma 4 BF16 tensor",
            expected: "value count * 2 without overflow".to_string(),
            actual: values.to_string(),
        })?;
        let offset = value_offset.checked_mul(2).ok_or_else(|| Error::Shape {
            label: "Gemma 4 BF16 tensor",
            expected: "value offset * 2 without overflow".to_string(),
            actual: value_offset.to_string(),
        })?;
        let shard = self.checkpoint.open_shard_for_tensor(tensor)?;
        let raw = shard.read_tensor_byte_range(tensor, offset as u64, bytes)?;
        if raw.len() != bytes {
            return Err(Error::Shape {
                label: "Gemma 4 BF16 tensor",
                expected: format!("{bytes} bytes"),
                actual: format!("{} bytes for {tensor}", raw.len()),
            });
        }
        Ok(raw
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect())
    }
}

fn matrix_shape(shape: &[usize], tensor: &str) -> Result<[usize; 2]> {
    let &[out_features, in_features] = shape else {
        return Err(Error::Shape {
            label: "Gemma 4 BF16 linear",
            expected: "shape=[out,in]".to_string(),
            actual: format!("shape={shape:?} for {tensor}"),
        });
    };
    Ok([out_features, in_features])
}

fn expert_shape(shape: &[usize], tensor: &str) -> Result<[usize; 3]> {
    let &[experts, out_features, in_features] = shape else {
        return Err(Error::Shape {
            label: "Gemma 4 BF16 expert linear",
            expected: "shape=[experts,out,in]".to_string(),
            actual: format!("shape={shape:?} for {tensor}"),
        });
    };
    Ok([experts, out_features, in_features])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(1);

    fn fixture_directory() -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "eider-gemma4-loader-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&directory).expect("create fixture directory");
        directory
    }

    fn write_fixture(directory: &Path) {
        let dense = [0x3f80_u16; 32];
        let expert_zero = [0x3f80_u16; 32];
        let expert_one = [0x4000_u16; 32];
        let mut payload = Vec::new();
        for value in dense.into_iter().chain(expert_zero).chain(expert_one) {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let header = json!({
            "model.language_model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16", "shape": [2, 16], "data_offsets": [0, 64]
            },
            "model.language_model.layers.0.experts.gate_up_proj": {
                "dtype": "BF16", "shape": [2, 2, 16], "data_offsets": [64, 192]
            }
        });
        let header = serde_json::to_vec(&header).expect("serialize safetensors header");
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header);
        shard.extend_from_slice(&payload);
        fs::write(directory.join("model.safetensors"), shard).expect("write fixture shard");
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": {
                "model.language_model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                "model.language_model.layers.0.experts.gate_up_proj": "model.safetensors"
            }}))
            .expect("serialize fixture index"),
        )
        .expect("write fixture index");
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({"model_type": "gemma4", "text_config": {
                "hidden_size": 16, "intermediate_size": 16, "moe_intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 16,
                "num_experts": 2, "top_k_experts": 1, "sliding_window": 16,
                "max_position_embeddings": 32, "layer_types": ["sliding_attention"]
            }}))
            .expect("serialize fixture config"),
        )
        .expect("write fixture config");
    }

    fn write_native_nvfp4_fixture(directory: &Path) {
        let prefix = "model.language_model.layers.0.self_attn.q_proj";
        let packed_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_global_name = format!("{prefix}.weight_global_scale");
        let input_global_name = format!("{prefix}.input_global_scale");
        let header = json!({
            &packed_name: {"dtype": "U8", "shape": [2, 8], "data_offsets": [0, 16]},
            &scale_name: {"dtype": "F8_E4M3", "shape": [2, 1], "data_offsets": [16, 18]},
            &weight_global_name: {"dtype": "F32", "shape": [1], "data_offsets": [18, 22]},
            &input_global_name: {"dtype": "F32", "shape": [1], "data_offsets": [22, 26]}
        });
        let header = serde_json::to_vec(&header).expect("serialize safetensors header");
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header);
        shard.extend_from_slice(&[0x11; 16]);
        shard.extend_from_slice(&[0x77; 2]);
        shard.extend_from_slice(&2.0_f32.to_le_bytes());
        shard.extend_from_slice(&4.0_f32.to_le_bytes());
        fs::write(directory.join("model.safetensors"), shard).expect("write native fixture shard");
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": {
                packed_name: "model.safetensors",
                scale_name: "model.safetensors",
                weight_global_name: "model.safetensors",
                input_global_name: "model.safetensors"
            }}))
            .expect("serialize native fixture index"),
        )
        .expect("write native fixture index");
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({"model_type": "gemma4", "text_config": {
                "hidden_size": 16, "intermediate_size": 16, "moe_intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 16,
                "num_experts": 2, "top_k_experts": 1, "sliding_window": 16,
                "max_position_embeddings": 32, "layer_types": ["sliding_attention"]
            }}))
            .expect("serialize native fixture config"),
        )
        .expect("write native fixture config");
    }

    #[test]
    fn bf16_dense_and_individual_experts_convert_without_materializing_the_stack() {
        let directory = fixture_directory();
        write_fixture(&directory);
        let checkpoint = Gemma4Checkpoint::open(&directory).expect("open fixture");

        let dense = checkpoint
            .load_linear_nvfp4("model.language_model.layers.0.self_attn.q_proj.weight")
            .expect("convert dense linear");
        let zero = checkpoint
            .load_expert_linear_nvfp4("model.language_model.layers.0.experts.gate_up_proj", 0)
            .expect("convert expert zero");
        let one = checkpoint
            .load_expert_linear_nvfp4("model.language_model.layers.0.experts.gate_up_proj", 1)
            .expect("convert expert one");

        assert_eq!((dense.out_features, dense.in_features), (2, 16));
        assert_eq!((zero.out_features, zero.in_features), (2, 16));
        assert_eq!(
            zero.prefix,
            "model.language_model.layers.0.experts.gate_up_proj[0]"
        );
        assert_ne!(zero.weight_scale, one.weight_scale);
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn stacked_gate_up_split_preserves_modelopt_rows_and_scales() {
        let weight = ModelOptNvfp4Linear {
            prefix: "gate_up".to_string(),
            out_features: 4,
            in_features: 16,
            packed_weight: (0..32).collect(),
            weight_scale: vec![11, 22, 33, 44],
            weight_scale_2: 0.125,
            input_scale: 0.25,
        };
        let (gate, up) =
            split_modelopt_out_features(weight, 2, "gate".to_string(), "up".to_string())
                .expect("split gate/up");

        assert_eq!((gate.out_features, gate.in_features), (2, 16));
        assert_eq!((up.out_features, up.in_features), (2, 16));
        assert_eq!(gate.packed_weight, (0..16).collect::<Vec<_>>());
        assert_eq!(up.packed_weight, (16..32).collect::<Vec<_>>());
        assert_eq!(gate.weight_scale, [11, 22]);
        assert_eq!(up.weight_scale, [33, 44]);
        assert_eq!((gate.weight_scale_2, up.weight_scale_2), (0.125, 0.125));
        assert_eq!((gate.input_scale, up.input_scale), (0.25, 0.25));
    }

    #[test]
    fn config_rejects_incomplete_layer_types() {
        let error = Gemma4Config::from_json(
            r#"{"model_type":"gemma4","text_config":{"hidden_size":16,"intermediate_size":16,"moe_intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":16,"global_head_dim":16,"num_experts":2,"top_k_experts":1,"sliding_window":16,"max_position_embeddings":32,"layer_types":["sliding_attention"]}}"#,
        )
        .expect_err("incomplete layer types must fail");
        assert!(matches!(error, Error::Shape { .. }));
    }

    #[test]
    fn native_compressed_tensors_nvfp4_is_used_without_a_bf16_tensor() {
        let directory = fixture_directory();
        write_native_nvfp4_fixture(&directory);
        let checkpoint = Gemma4Checkpoint::open(&directory).expect("open native fixture");
        let weight = checkpoint
            .load_linear_nvfp4("model.language_model.layers.0.self_attn.q_proj.weight")
            .expect("load native NVFP4 linear");

        assert_eq!((weight.out_features, weight.in_features), (2, 16));
        assert_eq!(weight.weight_scale_2, 0.5);
        assert_eq!(weight.input_scale, 0.25);
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn resident_linear_executes_the_modelopt_nvfp4_weight() {
        let values = (0..(16 * 64))
            .map(|index| match index % 5 {
                0 => 0x3f80_u16,
                1 => 0xbf80_u16,
                2 => 0x3f00_u16,
                3 => 0x4000_u16,
                _ => 0xbe80_u16,
            })
            .collect::<Vec<_>>();
        let weight =
            ModelOptNvfp4Linear::quantize_bf16("test", 16, 64, &values).expect("quantize weight");
        let expected_weight = weight.dequantize_to_f32_col_major();
        let linear = Gemma4Linear::from_modelopt(weight).expect("upload weight");
        let input = (0..64)
            .map(|index| (index as f32 - 31.0) / 31.0)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_blocking().expect("create stream");
        let input_device = DeviceBuffer::from_host(&input).expect("upload input");
        let mut output = DeviceBuffer::zeroed(16).expect("allocate output");
        linear
            .run_into(&input_device, &mut output, 1, &stream)
            .expect("run linear");
        let actual = output.copy_to_host(&stream).expect("download output");

        for output_index in 0..16 {
            let expected = (0..64)
                .map(|input_index| {
                    expected_weight[input_index + output_index * 64] * input[input_index]
                })
                .sum::<f32>();
            let tolerance = 0.3 + expected.abs() * 0.2;
            assert!(
                (actual[output_index] - expected).abs() <= tolerance,
                "output={output_index} expected={expected} actual={}",
                actual[output_index]
            );
        }
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_batched_prefill_logits_stay_within_serial_tolerance() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let model = Gemma4Model::load(model_dir).expect("load Gemma 4");
        let prompt = [2, 2364, 107, 496, 603, 563, 506, 236881];
        let stream = CudaStream::new_blocking().expect("stream");
        let mut cache = crate::gemma4::new_gemma4_sequence_cache(&model, 2, prompt.len() + 1)
            .expect("sequence cache");
        let mut serial = Gemma4Sequence::admit(&model, &mut cache, prompt.len() + 1, &stream)
            .expect("serial sequence");
        let mut batched = Gemma4Sequence::admit(&model, &mut cache, prompt.len() + 1, &stream)
            .expect("batched sequence");
        let mut serial_workspace = model
            .new_prefill_batch_workspace(1, 1, prompt.len() + 1)
            .expect("serial W4A4 workspace");
        for token in &prompt[..prompt.len() - 1] {
            model
                .prefill_batch(
                    &mut serial_workspace,
                    &mut [Gemma4PrefillRow {
                        token_ids: std::slice::from_ref(token),
                        sequence: &mut serial,
                        output: Gemma4PrefillOutput::None,
                    }],
                    &stream,
                    &mut cache,
                )
                .expect("serial W4A4 prefill");
        }
        model
            .forward_one(&mut serial, prompt[prompt.len() - 1], &stream, &mut cache)
            .expect("serial prompt logits");
        let mut workspace = model
            .new_prefill_batch_workspace(1, prompt.len(), prompt.len() + 1)
            .expect("batch workspace");
        model
            .prefill_batch(
                &mut workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &prompt,
                    sequence: &mut batched,
                    output: Gemma4PrefillOutput::FullLogits,
                }],
                &stream,
                &mut cache,
            )
            .expect("batch prefill");
        let serial_logits = model
            .logits_to_host(&serial.state, &stream)
            .expect("serial prompt logits");
        let batched_logits = model
            .logits_to_host(&batched.state, &stream)
            .expect("batched prompt logits");
        let error_rms = (serial_logits
            .iter()
            .zip(&batched_logits)
            .map(|(serial, batched)| (serial - batched).powi(2))
            .sum::<f32>()
            / serial_logits.len() as f32)
            .sqrt();
        let reference_rms = (serial_logits.iter().map(|value| value.powi(2)).sum::<f32>()
            / serial_logits.len() as f32)
            .sqrt();
        assert!(
            error_rms <= reference_rms * 0.2,
            "error_rms={error_rms} reference_rms={reference_rms}"
        );
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_batched_prefill_matches_token_serial_cache() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let model = Gemma4Model::load(model_dir).expect("load Gemma 4");
        let prompt = [2, 2364, 107, 496, 603, 563, 506, 236881];
        let next_input = 107;
        let stream = CudaStream::new_blocking().expect("stream");
        let mut cache = crate::gemma4::new_gemma4_sequence_cache(&model, 2, prompt.len() + 1)
            .expect("sequence cache");
        let mut serial = Gemma4Sequence::admit(&model, &mut cache, prompt.len() + 1, &stream)
            .expect("serial sequence");
        let mut batched = Gemma4Sequence::admit(&model, &mut cache, prompt.len() + 1, &stream)
            .expect("batched sequence");
        let mut serial_workspace = model
            .new_prefill_batch_workspace(1, 1, prompt.len() + 1)
            .expect("serial W4A4 workspace");
        for token in &prompt {
            model
                .prefill_batch(
                    &mut serial_workspace,
                    &mut [Gemma4PrefillRow {
                        token_ids: std::slice::from_ref(token),
                        sequence: &mut serial,
                        output: Gemma4PrefillOutput::None,
                    }],
                    &stream,
                    &mut cache,
                )
                .expect("serial W4A4 prefill");
        }
        let mut workspace = model
            .new_prefill_batch_workspace(1, prompt.len(), prompt.len() + 1)
            .expect("batch workspace");
        model
            .prefill_batch(
                &mut workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &prompt,
                    sequence: &mut batched,
                    output: Gemma4PrefillOutput::None,
                }],
                &stream,
                &mut cache,
            )
            .expect("batch prefill");
        let serial_next = model
            .decode_one(&mut serial, next_input, &stream, &mut cache)
            .expect("serial next token");
        let batched_next = model
            .decode_one(&mut batched, next_input, &stream, &mut cache)
            .expect("batched next token");
        assert_eq!(batched.position(), serial.position());
        assert_eq!(batched_next.token, serial_next.token);
        assert!(
            (batched_next.logit - serial_next.logit).abs()
                <= serial_next.logit.abs().mul_add(0.2, 0.25),
            "serial={serial_next:?} batched={batched_next:?}"
        );
    }
}
