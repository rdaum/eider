//! Poolside Laguna text-model loading and inference.
//!
//! Laguna combines heterogeneous grouped-query attention with resident NVFP4
//! routed experts. The checkpoint's dense, attention, shared-expert, embedding,
//! and LM-head weights remain BF16.

use crate::runtime::laguna_sequence_cache::{
    LagunaSequence, LagunaSequenceCache, laguna_cache_error,
};
use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;
use nvfp4::{
    CudaStream, CutlassFp4GroupedGemvF32Plan, DeviceBuffer, Error, F32Matrix, GpuSampledToken,
    GpuSamplingRow, GpuTokenSampler, ModelOptCheckpoint, ModelOptCublasLtWeight,
    ModelOptNvfp4Linear, Nvfp4Matrix, Result, Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmWeight,
    Sm12xKvAttentionWorkspace, Sm12xKvPagePool, add_f32_into_on_stream, argmax_f32_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, bf16_linear_pair_logits_f32_into_on_stream,
    copy_bf16_row_to_f32_indexed_into_on_stream, fill_f32_into_on_stream,
    indexed_grouped_gemv_on_stream, moe_silu_quantize_slots_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream, nemotron3_sigmoid_topk_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_inv_freq_scaled_sequence_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream, silu_mul_f32_into_on_stream,
    softplus_scale_heads_f32_into_on_stream,
};
use serde::Deserialize;
use std::f32::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tracing::info;

mod batch;
pub use batch::{LagunaPrefillBatchWorkspace, LagunaPrefillRow};

const HIDDEN: usize = 3_072;
pub(crate) const LAYERS: usize = 48;
pub(crate) const KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;
const EXPERTS: usize = 256;
const TOP_K: usize = 10;
const EXPERT_INTERMEDIATE: usize = 1_024;
const SHARED_INTERMEDIATE: usize = 1_024;
const DENSE_INTERMEDIATE: usize = 12_288;
const VOCAB: usize = 100_352;
const SLIDING_WINDOW: usize = 512;
const RMS_EPS: f32 = 1.0e-6;
const ROUTED_SCALE: f32 = 2.5;
const FULL_ROPE_SCALE: f32 = 1.346_573_6;

static NEXT_LAGUNA_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// Dimensions and per-layer attention layout for a Laguna checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct LagunaConfig {
    /// Hidden width.
    pub hidden_size: usize,
    /// Dense layer intermediate width.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Key/value head count.
    pub num_key_value_heads: usize,
    /// Attention head width.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of routed experts per sparse layer.
    pub num_experts: usize,
    /// Experts selected per token.
    pub num_experts_per_tok: usize,
    /// Routed expert intermediate width.
    pub moe_intermediate_size: usize,
    /// Shared expert intermediate width.
    pub shared_expert_intermediate_size: usize,
    /// Sliding-attention window.
    pub sliding_window: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// Per-layer attention kinds.
    pub layer_types: Vec<String>,
    /// Per-layer query head counts.
    pub num_attention_heads_per_layer: Vec<usize>,
    /// Per-layer MLP kinds.
    pub mlp_layer_types: Vec<String>,
}

#[derive(Deserialize)]
struct FileConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    norm_topk_prob: bool,
    moe_apply_router_weight_on_input: bool,
    moe_routed_scaling_factor: f32,
    sliding_window: usize,
    gating: String,
    layer_types: Vec<String>,
    num_attention_heads_per_layer: Vec<usize>,
    mlp_layer_types: Vec<String>,
    rope_parameters: RopeParameters,
}

#[derive(Deserialize)]
struct RopeParameters {
    full_attention: FullRopeParameters,
    sliding_attention: SlidingRopeParameters,
}

#[derive(Deserialize)]
struct FullRopeParameters {
    rope_type: String,
    rope_theta: f32,
    factor: f32,
    original_max_position_embeddings: usize,
    beta_slow: f32,
    beta_fast: f32,
    attention_factor: f32,
    partial_rotary_factor: f32,
}

#[derive(Deserialize)]
struct SlidingRopeParameters {
    rope_type: String,
    rope_theta: f32,
    partial_rotary_factor: f32,
}

impl LagunaConfig {
    /// Reads and validates the supported Laguna configuration.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text = fs::read_to_string(&path).map_err(|error| Error::Format {
            label: "Laguna config",
            detail: format!("{}: {error}", path.display()),
        })?;
        Self::from_json(&text)
    }

    fn from_json(text: &str) -> Result<Self> {
        let config: FileConfig = serde_json::from_str(text).map_err(|error| Error::Format {
            label: "Laguna config JSON",
            detail: error.to_string(),
        })?;
        let supported = config.model_type == "laguna"
            && config.hidden_size == HIDDEN
            && config.intermediate_size == DENSE_INTERMEDIATE
            && config.num_hidden_layers == LAYERS
            && config.num_key_value_heads == KV_HEADS
            && config.head_dim == HEAD_DIM
            && config.vocab_size == VOCAB
            && config.num_experts == EXPERTS
            && config.num_experts_per_tok == TOP_K
            && config.moe_intermediate_size == EXPERT_INTERMEDIATE
            && config.shared_expert_intermediate_size == SHARED_INTERMEDIATE
            && config.sliding_window == SLIDING_WINDOW
            && config.rms_norm_eps.to_bits() == RMS_EPS.to_bits()
            && config.norm_topk_prob
            && !config.moe_apply_router_weight_on_input
            && config.moe_routed_scaling_factor.to_bits() == ROUTED_SCALE.to_bits()
            && config.gating == "per-head"
            && config.layer_types.len() == LAYERS
            && config.num_attention_heads_per_layer.len() == LAYERS
            && config.mlp_layer_types.len() == LAYERS
            && config
                .mlp_layer_types
                .first()
                .is_some_and(|kind| kind == "dense")
            && config
                .mlp_layer_types
                .iter()
                .skip(1)
                .all(|kind| kind == "sparse")
            && config.rope_parameters.full_attention.rope_type == "yarn"
            && config.rope_parameters.full_attention.rope_theta.to_bits() == 500_000.0f32.to_bits()
            && config.rope_parameters.full_attention.factor.to_bits() == 32.0f32.to_bits()
            && config
                .rope_parameters
                .full_attention
                .original_max_position_embeddings
                == 8_192
            && config.rope_parameters.full_attention.beta_slow.to_bits() == 1.0f32.to_bits()
            && config.rope_parameters.full_attention.beta_fast.to_bits() == 32.0f32.to_bits()
            && config
                .rope_parameters
                .full_attention
                .attention_factor
                .to_bits()
                == FULL_ROPE_SCALE.to_bits()
            && config
                .rope_parameters
                .full_attention
                .partial_rotary_factor
                .to_bits()
                == 0.5f32.to_bits()
            && config.rope_parameters.sliding_attention.rope_type == "default"
            && config
                .rope_parameters
                .sliding_attention
                .rope_theta
                .to_bits()
                == 10_000.0f32.to_bits()
            && config
                .rope_parameters
                .sliding_attention
                .partial_rotary_factor
                .to_bits()
                == 1.0f32.to_bits();
        if !supported {
            return Err(Error::Format {
                label: "Laguna config",
                detail: "checkpoint does not match the supported Laguna-S-2.1 layout".to_string(),
            });
        }
        for (layer, ((kind, &heads), mlp)) in config
            .layer_types
            .iter()
            .zip(&config.num_attention_heads_per_layer)
            .zip(&config.mlp_layer_types)
            .enumerate()
        {
            let full = layer.is_multiple_of(4);
            if kind
                != if full {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
                || heads != if full { 48 } else { 72 }
                || mlp != if layer == 0 { "dense" } else { "sparse" }
            {
                return Err(Error::Format {
                    label: "Laguna layer layout",
                    detail: format!(
                        "unexpected layer {layer}: attention={kind} heads={heads} mlp={mlp}"
                    ),
                });
            }
        }
        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            vocab_size: config.vocab_size,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
            sliding_window: config.sliding_window,
            max_position_embeddings: config.max_position_embeddings,
            layer_types: config.layer_types,
            num_attention_heads_per_layer: config.num_attention_heads_per_layer,
            mlp_layer_types: config.mlp_layer_types,
        })
    }
}

struct Bf16Linear {
    name: String,
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Bf16Linear {
    fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let info = checkpoint.tensor_info(&name)?;
        if info.dtype != "BF16" || info.shape.len() != 2 {
            return Err(Error::Shape {
                label: "Laguna BF16 linear",
                expected: "BF16 [rows, cols]".to_string(),
                actual: format!("{} {:?} for {name}", info.dtype, info.shape),
            });
        }
        let rows = info.shape[0];
        let cols = info.shape[1];
        Ok(Self {
            name: prefix.to_string(),
            weight: read_bf16_device(checkpoint, &name, &[rows, cols])?,
            rows,
            cols,
        })
    }

    fn require_shape(&self, rows: usize, cols: usize, label: &'static str) -> Result<()> {
        if self.rows != rows || self.cols != cols {
            return Err(Error::Shape {
                label,
                expected: format!("[{rows}, {cols}]"),
                actual: format!("[{}, {}]", self.rows, self.cols),
            });
        }
        Ok(())
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.cols || output.len() != self.rows {
            return Err(Error::Shape {
                label: "Laguna BF16 linear buffers",
                expected: format!("input={} output={}", self.cols, self.rows),
                actual: format!("input={} output={}", input.len(), output.len()),
            });
        }
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
        .map_err(|error| Error::Format {
            label: "Laguna BF16 linear execution",
            detail: format!("{} [{}, {}]: {error}", self.name, self.rows, self.cols),
        })?;
        round_f32_to_bf16_in_place_on_stream(output.inout(), stream)
    }

    fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

struct LagunaRmsNorm {
    weight: DeviceBuffer<f32>,
    cols: usize,
}

impl LagunaRmsNorm {
    fn load(checkpoint: &ModelOptCheckpoint, name: &str, cols: usize) -> Result<Self> {
        let weight = read_float_device(checkpoint, name, &[cols])?;
        Ok(Self { weight, cols })
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            rows,
            self.cols,
            input,
            &self.weight,
            output.output(),
            RMS_EPS,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(output.inout(), rows * self.cols, stream)
    }

    fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

fn read_bf16_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<u16>> {
    let info = checkpoint.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(Error::Shape {
            label: "Laguna BF16 tensor",
            expected: format!("BF16 {shape:?}"),
            actual: format!("{} {:?} for {name}", info.dtype, info.shape),
        });
    }
    let bytes = checkpoint
        .open_shard_for_tensor(name)?
        .read_tensor_bytes(name)?;
    let expected = shape.iter().product::<usize>() * 2;
    if bytes.len() != expected {
        return Err(Error::Shape {
            label: "Laguna BF16 tensor bytes",
            expected: format!("{expected} bytes"),
            actual: format!("{} bytes for {name}", bytes.len()),
        });
    }
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>(),
    )
}

fn read_float_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let info = checkpoint.tensor_info(name)?;
    if info.shape != shape {
        return Err(Error::Shape {
            label: "Laguna float tensor",
            expected: format!("{shape:?}"),
            actual: format!("{:?} for {name}", info.shape),
        });
    }
    let values = checkpoint
        .open_shard_for_tensor(name)?
        .read_float_tensor_as_f32(name)?;
    DeviceBuffer::from_host(&values)
}

fn default_inverse_frequencies(rotary_dim: usize, theta: f32) -> Vec<f32> {
    (0..rotary_dim / 2)
        .map(|index| 1.0 / theta.powf(2.0 * index as f32 / rotary_dim as f32))
        .collect()
}

fn yarn_inverse_frequencies() -> Vec<f32> {
    let dim = HEAD_DIM / 2;
    let base = 500_000.0f32;
    let factor = 32.0f32;
    let original_context = 8_192.0f32;
    let correction = |rotations: f32| {
        dim as f32 * (original_context / (rotations * 2.0 * PI)).ln() / (2.0 * base.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min((dim - 1) as f32);
    (0..dim / 2)
        .map(|index| {
            let position_frequency = base.powf(2.0 * index as f32 / dim as f32);
            let extrapolated = position_frequency.recip();
            let interpolated = (factor * position_frequency).recip();
            let ramp = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation_factor = 1.0 - ramp;
            interpolated * (1.0 - extrapolation_factor) + extrapolated * extrapolation_factor
        })
        .collect()
}

struct LagunaMlp {
    gate: Bf16Linear,
    up: Bf16Linear,
    down: Bf16Linear,
    intermediate: usize,
}

struct LagunaMlpWorkspace {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl LagunaMlpWorkspace {
    fn device_bytes(&self) -> usize {
        self.gate.device_bytes()
            + self.up.device_bytes()
            + self.activated.device_bytes()
            + self.output.device_bytes()
    }
}

impl LagunaMlp {
    fn load(checkpoint: &ModelOptCheckpoint, prefix: &str, intermediate: usize) -> Result<Self> {
        let gate = Bf16Linear::load(checkpoint, &format!("{prefix}.gate_proj"))?;
        let up = Bf16Linear::load(checkpoint, &format!("{prefix}.up_proj"))?;
        let down = Bf16Linear::load(checkpoint, &format!("{prefix}.down_proj"))?;
        gate.require_shape(intermediate, HIDDEN, "Laguna MLP gate")?;
        up.require_shape(intermediate, HIDDEN, "Laguna MLP up")?;
        down.require_shape(HIDDEN, intermediate, "Laguna MLP down")?;
        Ok(Self {
            gate,
            up,
            down,
            intermediate,
        })
    }

    fn new_workspace(&self) -> Result<LagunaMlpWorkspace> {
        Ok(LagunaMlpWorkspace {
            gate: DeviceBuffer::zeroed(self.intermediate)?,
            up: DeviceBuffer::zeroed(self.intermediate)?,
            activated: DeviceBuffer::zeroed(self.intermediate)?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    fn run<'a>(
        &self,
        workspace: &'a mut LagunaMlpWorkspace,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        bf16_linear_pair_logits_f32_into_on_stream(
            input,
            &self.gate.weight,
            &self.up.weight,
            workspace.gate.output(),
            workspace.up.output(),
            self.intermediate,
            self.intermediate,
            HIDDEN,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.gate.inout(), stream)?;
        round_f32_to_bf16_in_place_on_stream(workspace.up.inout(), stream)?;
        silu_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.up,
            workspace.activated.output(),
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.activated.inout(), stream)?;
        self.down
            .run_into(&workspace.activated, &mut workspace.output, stream)?;
        Ok(&workspace.output)
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

struct LagunaAttention {
    q: Bf16Linear,
    k: Bf16Linear,
    v: Bf16Linear,
    o: Bf16Linear,
    gate: Bf16Linear,
    q_norm: LagunaRmsNorm,
    k_norm: LagunaRmsNorm,
    inv_freq: DeviceBuffer<f32>,
    q_heads: usize,
    rotary_dim: usize,
    rope_scale: f32,
    window: Option<usize>,
}

struct LagunaAttentionWorkspace {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl LagunaAttentionWorkspace {
    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_normed.device_bytes()
            + self.k_normed.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attended.device_bytes()
            + self.gate.device_bytes()
            + self.gated.device_bytes()
            + self.output.device_bytes()
    }
}

impl LagunaAttention {
    fn load(checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<Self> {
        let prefix = format!("model.layers.{layer}.self_attn");
        let full = layer.is_multiple_of(4);
        let q_heads = if full { 48 } else { 72 };
        let q_width = q_heads * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q = Bf16Linear::load(checkpoint, &format!("{prefix}.q_proj"))?;
        let k = Bf16Linear::load(checkpoint, &format!("{prefix}.k_proj"))?;
        let v = Bf16Linear::load(checkpoint, &format!("{prefix}.v_proj"))?;
        let o = Bf16Linear::load(checkpoint, &format!("{prefix}.o_proj"))?;
        let gate = Bf16Linear::load(checkpoint, &format!("{prefix}.g_proj"))?;
        q.require_shape(q_width, HIDDEN, "Laguna query projection")?;
        k.require_shape(kv_width, HIDDEN, "Laguna key projection")?;
        v.require_shape(kv_width, HIDDEN, "Laguna value projection")?;
        o.require_shape(HIDDEN, q_width, "Laguna attention output projection")?;
        gate.require_shape(q_heads, HIDDEN, "Laguna attention gate")?;
        let rotary_dim = if full { HEAD_DIM / 2 } else { HEAD_DIM };
        Ok(Self {
            q,
            k,
            v,
            o,
            gate,
            q_norm: LagunaRmsNorm::load(checkpoint, &format!("{prefix}.q_norm.weight"), HEAD_DIM)?,
            k_norm: LagunaRmsNorm::load(checkpoint, &format!("{prefix}.k_norm.weight"), HEAD_DIM)?,
            inv_freq: DeviceBuffer::from_host(&if full {
                yarn_inverse_frequencies()
            } else {
                default_inverse_frequencies(HEAD_DIM, 10_000.0)
            })?,
            q_heads,
            rotary_dim,
            rope_scale: if full { FULL_ROPE_SCALE } else { 1.0 },
            window: (!full).then_some(SLIDING_WINDOW),
        })
    }

    fn new_workspace(&self) -> Result<LagunaAttentionWorkspace> {
        let q_width = self.q_heads * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        Ok(LagunaAttentionWorkspace {
            q: DeviceBuffer::zeroed(q_width)?,
            k: DeviceBuffer::zeroed(kv_width)?,
            v: DeviceBuffer::zeroed(kv_width)?,
            q_normed: DeviceBuffer::zeroed(q_width)?,
            k_normed: DeviceBuffer::zeroed(kv_width)?,
            q_rope: DeviceBuffer::zeroed(q_width)?,
            k_rope: DeviceBuffer::zeroed(kv_width)?,
            attended: DeviceBuffer::zeroed(q_width)?,
            gate: DeviceBuffer::zeroed(self.q_heads)?,
            gated: DeviceBuffer::zeroed(q_width)?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_decode<'a>(
        &self,
        workspace: &'a mut LagunaAttentionWorkspace,
        input: &DeviceBuffer<f32>,
        cache: LagunaLayerCache<'_>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.q.run_into(input, &mut workspace.q, stream)?;
        self.k.run_into(input, &mut workspace.k, stream)?;
        self.v.run_into(input, &mut workspace.v, stream)?;
        self.q_norm
            .run_into(&workspace.q, &mut workspace.q_normed, self.q_heads, stream)?;
        self.k_norm
            .run_into(&workspace.k, &mut workspace.k_normed, KV_HEADS, stream)?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            1,
            self.q_heads,
            HEAD_DIM,
            self.rotary_dim,
            &workspace.q_normed,
            &self.inv_freq,
            workspace.q_rope.output(),
            position,
            self.rope_scale,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.q_rope.inout(), stream)?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            1,
            KV_HEADS,
            HEAD_DIM,
            self.rotary_dim,
            &workspace.k_normed,
            &self.inv_freq,
            workspace.k_rope.output(),
            position,
            self.rope_scale,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.k_rope.inout(), stream)?;
        cache.pool.append_at_offsets_on_stream(
            cache.page_slot,
            cache.page_offset,
            &workspace.k_rope,
            0,
            &workspace.v,
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
        self.gate.run_into(input, &mut workspace.gate, stream)?;
        softplus_scale_heads_f32_into_on_stream(
            &workspace.gate,
            &workspace.attended,
            workspace.gated.output(),
            HEAD_DIM,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.gated.inout(), stream)?;
        self.o
            .run_into(&workspace.gated, &mut workspace.output, stream)?;
        Ok(&workspace.output)
    }

    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.o.device_bytes()
            + self.gate.device_bytes()
            + self.q_norm.device_bytes()
            + self.k_norm.device_bytes()
            + self.inv_freq.device_bytes()
    }
}

struct LagunaMoe {
    router: Bf16Linear,
    correction_bias: DeviceBuffer<f32>,
    _gate_up: Vec<ModelOptCublasLtWeight>,
    gate_up_values: DeviceBuffer<*const u8>,
    gate_up_scales: DeviceBuffer<*const u8>,
    gate_up_alphas: DeviceBuffer<f32>,
    gate_up_alpha_table: DeviceBuffer<*mut f32>,
    _down: Vec<Sm12xFp4DeviceGemmWeight>,
    down_tiles: DeviceBuffer<*const u8>,
    down_scales: DeviceBuffer<*const u32>,
    down_input_scales: DeviceBuffer<f32>,
    down_alphas: DeviceBuffer<f32>,
    gate_up_unity_alphas: DeviceBuffer<f32>,
    shared: LagunaMlp,
}

struct LagunaMoeWorkspace {
    router_logits: DeviceBuffer<f32>,
    route_indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    gate_up_input: Nvfp4Matrix,
    gate_up_c: F32Matrix,
    gate_up_plan: CutlassFp4GroupedGemvF32Plan,
    gate_up_output: DeviceBuffer<f32>,
    gate_up_table: DeviceBuffer<*const f32>,
    gate_up_output_table: DeviceBuffer<*mut f32>,
    down_tiles: DeviceBuffer<u8>,
    down_scales: DeviceBuffer<u32>,
    _down_outputs: Vec<F32Matrix>,
    down_inputs: DeviceBuffer<*const f32>,
    down_outputs: DeviceBuffer<*mut f32>,
    routed: DeviceBuffer<f32>,
    shared: LagunaMlpWorkspace,
    output: DeviceBuffer<f32>,
}

impl LagunaMoeWorkspace {
    fn device_bytes(&self) -> usize {
        self.router_logits.device_bytes()
            + self.route_indices.device_bytes()
            + self.route_weights.device_bytes()
            + self.gate_up_input.device_bytes()
            + self.gate_up_c.device_bytes()
            + self.gate_up_output.device_bytes()
            + self.gate_up_table.device_bytes()
            + self.gate_up_output_table.device_bytes()
            + self.down_tiles.device_bytes()
            + self.down_scales.device_bytes()
            + self
                ._down_outputs
                .iter()
                .map(F32Matrix::device_bytes)
                .sum::<usize>()
            + self.down_inputs.device_bytes()
            + self.down_outputs.device_bytes()
            + self.routed.device_bytes()
            + self.shared.device_bytes()
            + self.output.device_bytes()
    }
}

impl LagunaMoe {
    fn load(checkpoint: &ModelOptCheckpoint, artifact_dir: &Path, layer: usize) -> Result<Self> {
        let prefix = format!("model.layers.{layer}.mlp");
        let layer_artifacts = artifact_dir.join(format!("layer-{layer:02}"));
        fs::create_dir_all(&layer_artifacts).map_err(|error| Error::Format {
            label: "Laguna expert artifacts",
            detail: format!("{}: {error}", layer_artifacts.display()),
        })?;
        ensure_down_artifacts(checkpoint, &layer_artifacts, layer)?;
        let router = Bf16Linear::load(checkpoint, &format!("{prefix}.gate"))?;
        router.require_shape(EXPERTS, HIDDEN, "Laguna router")?;
        let correction_bias = read_float_device(
            checkpoint,
            &format!("{prefix}.experts.e_score_correction_bias"),
            &[EXPERTS],
        )?;
        let gate_up = load_gate_up(checkpoint, &prefix)?;
        if let Some((expert, weight)) = gate_up.iter().enumerate().find(|(_, weight)| {
            let shape = weight.matrix().shape();
            (shape.rows, shape.cols) != (HIDDEN, EXPERT_INTERMEDIATE * 2)
        }) {
            let shape = weight.matrix().shape();
            return Err(Error::Shape {
                label: "Laguna gate/up expert TN storage",
                expected: format!("{}x{}", HIDDEN, EXPERT_INTERMEDIATE * 2),
                actual: format!("expert {expert}: {}x{}", shape.rows, shape.cols),
            });
        }
        let gate_up_values = gate_up
            .iter()
            .map(|weight| weight.matrix().values_ptr())
            .collect::<Vec<_>>();
        let gate_up_scales = gate_up
            .iter()
            .map(|weight| weight.matrix().scales_ptr())
            .collect::<Vec<_>>();
        let mut gate_up_alphas = DeviceBuffer::from_host(
            &gate_up
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let gate_up_alpha_table = scalar_pointer_table(&mut gate_up_alphas)?;
        let mut down = Vec::with_capacity(EXPERTS);
        let mut down_tiles = Vec::with_capacity(EXPERTS);
        let mut down_scales = Vec::with_capacity(EXPERTS);
        let mut down_input_scales = Vec::with_capacity(EXPERTS);
        let mut down_alphas = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let expert_prefix = format!("{prefix}.experts.{expert}");
            let (weight_scale, input_scale) =
                checkpoint.load_nvfp4_scales(&format!("{expert_prefix}.down_proj"))?;
            down_input_scales.push(input_scale);
            down_alphas.push(input_scale * weight_scale);
            let down_path = layer_artifacts.join(format!("expert-{expert:03}-down.sm12x"));
            let native = Sm12xFp4GemmWeight::read_cache_file(&down_path)?;
            let device = native.to_device()?;
            down_tiles.push(device.tiles_ptr());
            down_scales.push(device.scales_ptr());
            down.push(device);
        }
        Ok(Self {
            router,
            correction_bias,
            _gate_up: gate_up,
            gate_up_values: DeviceBuffer::from_host(&gate_up_values)?,
            gate_up_scales: DeviceBuffer::from_host(&gate_up_scales)?,
            gate_up_alphas,
            gate_up_alpha_table,
            _down: down,
            down_tiles: DeviceBuffer::from_host(&down_tiles)?,
            down_scales: DeviceBuffer::from_host(&down_scales)?,
            down_input_scales: DeviceBuffer::from_host(&down_input_scales)?,
            down_alphas: DeviceBuffer::from_host(&down_alphas)?,
            gate_up_unity_alphas: DeviceBuffer::from_host(&vec![1.0; EXPERTS])?,
            shared: LagunaMlp::load(
                checkpoint,
                &format!("{prefix}.shared_expert"),
                SHARED_INTERMEDIATE,
            )?,
        })
    }

    fn new_workspace(&self) -> Result<LagunaMoeWorkspace> {
        let gate_up_width = EXPERT_INTERMEDIATE * 2;
        let gate_up_output = DeviceBuffer::zeroed(TOP_K * gate_up_width)?;
        let gate_up_base = gate_up_output.as_const_ptr().cast::<f32>();
        let gate_up_table = DeviceBuffer::from_host(
            &(0..TOP_K)
                .map(|slot| unsafe { gate_up_base.add(slot * gate_up_width) })
                .collect::<Vec<_>>(),
        )?;
        let gate_up_output_table = DeviceBuffer::from_host(
            &(0..TOP_K)
                .map(|slot| unsafe { gate_up_base.cast_mut().add(slot * gate_up_width) })
                .collect::<Vec<_>>(),
        )?;
        let mut down_outputs = Vec::with_capacity(TOP_K);
        let mut down_inputs = Vec::with_capacity(TOP_K);
        let mut down_output_ptrs = Vec::with_capacity(TOP_K);
        for _ in 0..TOP_K {
            let mut output = F32Matrix::zeroed(HIDDEN, 1)?;
            down_inputs.push(output.data_ptr());
            down_output_ptrs.push(output.data_mut_ptr());
            down_outputs.push(output);
        }
        Ok(LagunaMoeWorkspace {
            router_logits: DeviceBuffer::zeroed(EXPERTS)?,
            route_indices: DeviceBuffer::zeroed(TOP_K)?,
            route_weights: DeviceBuffer::zeroed(TOP_K)?,
            gate_up_input: Nvfp4Matrix::zeroed_col_major(HIDDEN, 1)?,
            gate_up_c: F32Matrix::zeroed(gate_up_width, 1)?,
            gate_up_plan: CutlassFp4GroupedGemvF32Plan::new(gate_up_width, HIDDEN, TOP_K)?,
            gate_up_output,
            gate_up_table,
            gate_up_output_table,
            down_tiles: DeviceBuffer::zeroed(TOP_K * (EXPERT_INTERMEDIATE / 64) * 512)?,
            down_scales: DeviceBuffer::zeroed(TOP_K * (EXPERT_INTERMEDIATE / 64))?,
            _down_outputs: down_outputs,
            down_inputs: DeviceBuffer::from_host(&down_inputs)?,
            down_outputs: DeviceBuffer::from_host(&down_output_ptrs)?,
            routed: DeviceBuffer::zeroed(HIDDEN)?,
            shared: self.shared.new_workspace()?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    fn run<'a>(
        &self,
        workspace: &'a mut LagunaMoeWorkspace,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.router
            .run_into(input, &mut workspace.router_logits, stream)?;
        nemotron3_sigmoid_topk_f32_into_on_stream(
            &workspace.router_logits,
            &self.correction_bias,
            workspace.route_indices.output(),
            workspace.route_weights.output(),
            TOP_K,
            1,
            1,
            true,
            ROUTED_SCALE,
            stream,
        )?;
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            HIDDEN,
            1,
            input,
            &mut workspace.gate_up_input,
            1.0,
            stream,
        )?;
        workspace
            .gate_up_plan
            .run_indexed_a_tiled_scales_on_stream(
                &workspace.route_indices,
                &self.gate_up_values,
                &self.gate_up_scales,
                &self.gate_up_alphas,
                &workspace.gate_up_input,
                &workspace.gate_up_c,
                &workspace.gate_up_output_table,
                stream,
            )?;
        moe_silu_quantize_slots_on_stream(
            &workspace.route_indices,
            &workspace.gate_up_table,
            &mut workspace.down_tiles,
            &mut workspace.down_scales,
            &self.down_input_scales,
            &self.gate_up_unity_alphas,
            EXPERT_INTERMEDIATE,
            TOP_K,
            stream,
        )?;
        indexed_grouped_gemv_on_stream(
            &workspace.route_indices,
            &self.down_tiles,
            &self.down_scales,
            EXPERTS,
            &workspace.down_tiles,
            &workspace.down_scales,
            &workspace.down_outputs,
            HIDDEN / 16,
            EXPERT_INTERMEDIATE / 64,
            TOP_K,
            stream,
        )?;
        fill_f32_into_on_stream(workspace.routed.output(), 0.0, stream)?;
        moe_weighted_accumulate_slots_f32_on_stream(
            &workspace.route_indices,
            &workspace.route_weights,
            &workspace.down_inputs,
            &self.down_alphas,
            workspace.routed.inout(),
            stream,
        )?;
        let shared = self.shared.run(&mut workspace.shared, input, stream)?;
        add_f32_into_on_stream(&workspace.routed, shared, workspace.output.output(), stream)?;
        Ok(&workspace.output)
    }

    fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.correction_bias.device_bytes()
            + self
                ._gate_up
                .iter()
                .map(ModelOptCublasLtWeight::device_bytes)
                .sum::<usize>()
            + self.gate_up_values.device_bytes()
            + self.gate_up_scales.device_bytes()
            + self.gate_up_alphas.device_bytes()
            + self.gate_up_alpha_table.device_bytes()
            + self
                ._down
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::device_bytes)
                .sum::<usize>()
            + self.down_tiles.device_bytes()
            + self.down_scales.device_bytes()
            + self.down_input_scales.device_bytes()
            + self.down_alphas.device_bytes()
            + self.gate_up_unity_alphas.device_bytes()
            + self.shared.device_bytes()
    }
}

fn load_gate_up(
    checkpoint: &ModelOptCheckpoint,
    layer_prefix: &str,
) -> Result<Vec<ModelOptCublasLtWeight>> {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8);
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let sender = sender.clone();
            let next = &next;
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let expert = next.fetch_add(1, Ordering::Relaxed);
                    if expert >= EXPERTS {
                        break;
                    }
                    let prefix = format!("{layer_prefix}.experts.{expert}");
                    let gate = checkpoint.load_nvfp4_linear(&format!("{prefix}.gate_proj"))?;
                    let up = checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?;
                    let gate_up = ModelOptNvfp4Linear::concat_out_features(
                        format!("{prefix}.gate_up_proj"),
                        &gate,
                        &up,
                    )?;
                    sender
                        .send((expert, gate_up))
                        .map_err(|error| Error::Format {
                            label: "Laguna gate/up loading",
                            detail: error.to_string(),
                        })?;
                }
                Ok(())
            }));
        }
        drop(sender);
        let mut prepared = receiver.into_iter().collect::<Vec<_>>();
        for handle in handles {
            handle.join().map_err(|_| Error::Format {
                label: "Laguna gate/up loading",
                detail: "worker panicked".to_string(),
            })??;
        }
        prepared.sort_unstable_by_key(|(expert, _)| *expert);
        if prepared.len() != EXPERTS {
            return Err(Error::Format {
                label: "Laguna gate/up loading",
                detail: format!("loaded {} of {EXPERTS} experts", prepared.len()),
            });
        }
        prepared
            .into_iter()
            .map(|(_, weight)| weight.as_cublaslt_weight())
            .collect()
    })
}

fn scalar_pointer_table(values: &mut DeviceBuffer<f32>) -> Result<DeviceBuffer<*mut f32>> {
    let base = values.as_const_ptr().cast::<f32>().cast_mut();
    DeviceBuffer::from_host(
        &(0..values.len())
            .map(|index| unsafe { base.add(index) })
            .collect::<Vec<_>>(),
    )
}

fn ensure_down_artifacts(
    checkpoint: &ModelOptCheckpoint,
    layer_artifacts: &Path,
    layer: usize,
) -> Result<()> {
    let missing = (0..EXPERTS)
        .filter(|expert| {
            !Sm12xFp4GemmWeight::cache_file_matches(
                layer_artifacts.join(format!("expert-{expert:03}-down.sm12x")),
                HIDDEN,
                EXPERT_INTERMEDIATE,
            )
        })
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8)
        .min(missing.len());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(|| -> Result<()> {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&expert) = missing.get(index) else {
                        break;
                    };
                    let prefix = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj");
                    let host = checkpoint.load_nvfp4_linear(&prefix)?;
                    let row_major = host.dequantize_to_f32_col_major();
                    let native = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
                        HIDDEN,
                        EXPERT_INTERMEDIATE,
                        &row_major,
                    )?
                    .weight;
                    native.write_cache_file(
                        layer_artifacts.join(format!("expert-{expert:03}-down.sm12x")),
                    )?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Format {
                label: "Laguna expert artifacts",
                detail: format!("layer {layer} cache worker panicked"),
            })??;
        }
        Ok(())
    })
}

enum LagunaFfn {
    Dense(LagunaMlp),
    Moe(Box<LagunaMoe>),
}

enum LagunaFfnWorkspace {
    Dense(LagunaMlpWorkspace),
    Moe(Box<LagunaMoeWorkspace>),
}

impl LagunaFfnWorkspace {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Dense(workspace) => workspace.device_bytes(),
            Self::Moe(workspace) => workspace.device_bytes(),
        }
    }
}

struct LagunaLayer {
    input_norm: LagunaRmsNorm,
    attention: LagunaAttention,
    post_attention_norm: LagunaRmsNorm,
    ffn: LagunaFfn,
}

struct LagunaLayerWorkspace {
    normalized: DeviceBuffer<f32>,
    attention: LagunaAttentionWorkspace,
    attention_residual: DeviceBuffer<f32>,
    ffn_normalized: DeviceBuffer<f32>,
    ffn: LagunaFfnWorkspace,
    output: DeviceBuffer<f32>,
}

impl LagunaLayerWorkspace {
    fn device_bytes(&self) -> usize {
        self.normalized.device_bytes()
            + self.attention.device_bytes()
            + self.attention_residual.device_bytes()
            + self.ffn_normalized.device_bytes()
            + self.ffn.device_bytes()
            + self.output.device_bytes()
    }
}

impl LagunaLayer {
    fn load(checkpoint: &ModelOptCheckpoint, artifact_dir: &Path, layer: usize) -> Result<Self> {
        let prefix = format!("model.layers.{layer}");
        Ok(Self {
            input_norm: LagunaRmsNorm::load(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                HIDDEN,
            )?,
            attention: LagunaAttention::load(checkpoint, layer)?,
            post_attention_norm: LagunaRmsNorm::load(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                HIDDEN,
            )?,
            ffn: if layer == 0 {
                LagunaFfn::Dense(LagunaMlp::load(
                    checkpoint,
                    &format!("{prefix}.mlp"),
                    DENSE_INTERMEDIATE,
                )?)
            } else {
                LagunaFfn::Moe(Box::new(LagunaMoe::load(checkpoint, artifact_dir, layer)?))
            },
        })
    }

    fn new_workspace(&self) -> Result<LagunaLayerWorkspace> {
        Ok(LagunaLayerWorkspace {
            normalized: DeviceBuffer::zeroed(HIDDEN)?,
            attention: self.attention.new_workspace()?,
            attention_residual: DeviceBuffer::zeroed(HIDDEN)?,
            ffn_normalized: DeviceBuffer::zeroed(HIDDEN)?,
            ffn: match &self.ffn {
                LagunaFfn::Dense(mlp) => LagunaFfnWorkspace::Dense(mlp.new_workspace()?),
                LagunaFfn::Moe(moe) => LagunaFfnWorkspace::Moe(Box::new(moe.new_workspace()?)),
            },
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_one(
        &self,
        workspace: &mut LagunaLayerWorkspace,
        input: &DeviceBuffer<f32>,
        cache: LagunaLayerCache<'_>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.input_norm
            .run_into(input, &mut workspace.normalized, 1, stream)?;
        let attention = self.attention.run_decode(
            &mut workspace.attention,
            &workspace.normalized,
            cache,
            position,
            stream,
        )?;
        add_f32_into_on_stream(
            input,
            attention,
            workspace.attention_residual.output(),
            stream,
        )?;
        self.post_attention_norm.run_into(
            &workspace.attention_residual,
            &mut workspace.ffn_normalized,
            1,
            stream,
        )?;
        let ffn = match (&self.ffn, &mut workspace.ffn) {
            (LagunaFfn::Dense(mlp), LagunaFfnWorkspace::Dense(scratch)) => {
                mlp.run(scratch, &workspace.ffn_normalized, stream)?
            }
            (LagunaFfn::Moe(moe), LagunaFfnWorkspace::Moe(scratch)) => {
                moe.run(scratch, &workspace.ffn_normalized, stream)?
            }
            _ => {
                return Err(Error::Format {
                    label: "Laguna layer workspace",
                    detail: "FFN workspace does not match its weights".to_string(),
                });
            }
        };
        add_f32_into_on_stream(
            &workspace.attention_residual,
            ffn,
            workspace.output.output(),
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.input_norm.device_bytes()
            + self.attention.device_bytes()
            + self.post_attention_norm.device_bytes()
            + match &self.ffn {
                LagunaFfn::Dense(mlp) => mlp.device_bytes(),
                LagunaFfn::Moe(moe) => moe.device_bytes(),
            }
    }
}

struct LagunaCompactAttention {
    full: Sm12xKvAttentionWorkspace,
    sliding: Sm12xKvAttentionWorkspace,
}

impl LagunaCompactAttention {
    fn new(max_tokens: usize) -> Result<Self> {
        Ok(Self {
            full: Sm12xKvAttentionWorkspace::new_gqa(max_tokens, 48, KV_HEADS, HEAD_DIM)?,
            sliding: Sm12xKvAttentionWorkspace::new_gqa(max_tokens, 72, KV_HEADS, HEAD_DIM)?,
        })
    }

    fn for_layer(&mut self, layer: usize) -> &mut Sm12xKvAttentionWorkspace {
        if layer.is_multiple_of(4) {
            &mut self.full
        } else {
            &mut self.sliding
        }
    }

    fn device_bytes(&self) -> usize {
        self.full.device_bytes() + self.sliding.device_bytes()
    }
}

/// Fully loaded Laguna-S-2.1 text model.
pub struct LagunaModel {
    model_id: u64,
    config: LagunaConfig,
    embedding: DeviceBuffer<u16>,
    layers: Vec<LagunaLayer>,
    final_norm: LagunaRmsNorm,
    lm_head: Bf16Linear,
    stream: CudaStream,
}

/// Mutable execution and compact K/V state for one Laguna sequence.
pub struct LagunaDecodeState {
    model_id: u64,
    token: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layers: Vec<LagunaLayerWorkspace>,
    compact_attention: LagunaCompactAttention,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_index: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    sampler: GpuTokenSampler,
    pub(crate) position: usize,
    max_tokens: usize,
}

struct LagunaLayerCache<'a> {
    pool: &'a mut Sm12xKvPagePool,
    page_slot: usize,
    page_offset: usize,
    page_table: &'a DeviceBuffer<u32>,
    attention: &'a mut Sm12xKvAttentionWorkspace,
}

/// One greedy Laguna next-token result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagunaNextToken {
    /// Highest-logit token.
    pub token: u32,
    /// Logit associated with [`Self::token`].
    pub logit: f32,
}

impl LagunaModel {
    /// Loads the supported Laguna checkpoint into resident device storage.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        Self::load_with_artifact_dir(model_dir, default_artifact_dir(model_dir)?)
    }

    /// Loads Laguna with an explicit writable directory for derived native weights.
    pub fn load_with_artifact_dir(
        model_dir: impl AsRef<Path>,
        artifact_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let artifact_dir = artifact_dir.as_ref();
        let config = LagunaConfig::open(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let embedding =
            read_bf16_device(&checkpoint, "model.embed_tokens.weight", &[VOCAB, HIDDEN])?;
        let final_norm = LagunaRmsNorm::load(&checkpoint, "model.norm.weight", HIDDEN)?;
        let lm_head = Bf16Linear::load(&checkpoint, "lm_head")?;
        lm_head.require_shape(VOCAB, HIDDEN, "Laguna LM head")?;
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            layers.push(LagunaLayer::load(&checkpoint, artifact_dir, layer)?);
            let device_bytes = embedding.device_bytes()
                + final_norm.device_bytes()
                + lm_head.device_bytes()
                + layers.iter().map(LagunaLayer::device_bytes).sum::<usize>();
            info!(
                layer,
                device_weight_gib = device_bytes as f64 / (1u64 << 30) as f64,
                "loaded Laguna layer"
            );
        }
        Ok(Self {
            model_id: NEXT_LAGUNA_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            stream: CudaStream::new_non_blocking()?,
        })
    }

    /// Returns the validated model configuration.
    pub fn config(&self) -> &LagunaConfig {
        &self.config
    }

    /// Returns the checkpoint vocabulary size.
    pub fn vocab(&self) -> usize {
        self.config.vocab_size
    }

    /// Waits for all work submitted by this model instance.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Allocates request-private execution state for one sequence.
    pub fn new_sequence_state(&self, max_tokens: usize) -> Result<LagunaDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_position_embeddings {
            return Err(Error::Shape {
                label: "Laguna decode capacity",
                expected: format!("1..={}", self.config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        Ok(LagunaDecodeState {
            model_id: self.model_id,
            token: DeviceBuffer::zeroed(1)?,
            hidden: DeviceBuffer::zeroed(HIDDEN)?,
            layers: self
                .layers
                .iter()
                .map(LagunaLayer::new_workspace)
                .collect::<Result<Vec<_>>>()?,
            compact_attention: LagunaCompactAttention::new(max_tokens)?,
            final_hidden: DeviceBuffer::zeroed(HIDDEN)?,
            logits: DeviceBuffer::zeroed(VOCAB)?,
            next_index: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            sampler: GpuTokenSampler::new(1, VOCAB)?,
            position: 0,
            max_tokens,
        })
    }

    /// Advances one input token without selecting a result.
    pub fn consume_one(
        &self,
        sequence: &mut LagunaSequence,
        token: u32,
        cache: &mut LagunaSequenceCache,
    ) -> Result<()> {
        let target = self.reserve_token(sequence, cache)?;
        let result = self.forward_hidden_uncommitted(sequence, token, cache, target);
        self.complete_token(sequence, cache, target, result)
    }

    /// Advances one token and returns all output logits.
    pub fn logits_one(
        &self,
        sequence: &mut LagunaSequence,
        token: u32,
        cache: &mut LagunaSequenceCache,
    ) -> Result<Vec<f32>> {
        self.forward_one(sequence, token, cache)?;
        Ok(sequence.state.logits.copy_to_host(&self.stream)?.into_vec())
    }

    /// Advances one token and performs greedy selection.
    pub fn decode_one(
        &self,
        sequence: &mut LagunaSequence,
        token: u32,
        cache: &mut LagunaSequenceCache,
    ) -> Result<LagunaNextToken> {
        self.forward_one(sequence, token, cache)?;
        let state = &mut sequence.state;
        argmax_f32_into_on_stream(
            &state.logits,
            state.next_index.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        Ok(LagunaNextToken {
            token: state.next_index.copy_to_host(&self.stream)?[0],
            logit: state.next_value.copy_to_host(&self.stream)?[0],
        })
    }

    /// Advances one token and samples from its device-resident logits.
    pub fn sample_one(
        &self,
        sequence: &mut LagunaSequence,
        token: u32,
        sampling: &mut GpuSamplingRow<'_>,
        cache: &mut LagunaSequenceCache,
    ) -> Result<GpuSampledToken> {
        self.forward_one(sequence, token, cache)?;
        sequence
            .state
            .sampler
            .sample(
                &sequence.state.logits,
                std::slice::from_mut(sampling),
                VOCAB,
                &self.stream,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Format {
                label: "Laguna GPU sampling",
                detail: "sampler returned no token".to_string(),
            })
    }

    fn reserve_token(
        &self,
        sequence: &mut LagunaSequence,
        cache: &mut LagunaSequenceCache,
    ) -> Result<sequence_cache::AppendTarget> {
        cache
            .reserve_append(
                sequence.cache_id,
                1,
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(laguna_cache_error)
    }

    fn complete_token(
        &self,
        sequence: &mut LagunaSequence,
        cache: &mut LagunaSequenceCache,
        target: sequence_cache::AppendTarget,
        result: Result<()>,
    ) -> Result<()> {
        if let Err(error) = result {
            cache.abort_append(target).map_err(laguna_cache_error)?;
            return Err(error);
        }
        cache
            .commit_append(
                target,
                1,
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(laguna_cache_error)?;
        sequence.state.position += 1;
        Ok(())
    }

    fn forward_hidden_uncommitted(
        &self,
        sequence: &mut LagunaSequence,
        token: u32,
        cache: &mut LagunaSequenceCache,
        target: sequence_cache::AppendTarget,
    ) -> Result<()> {
        let state = &mut sequence.state;
        if state.model_id != self.model_id {
            return Err(Error::Format {
                label: "Laguna decode state",
                detail: "state belongs to a different model instance".to_string(),
            });
        }
        if token as usize >= VOCAB || state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "Laguna decode token",
                expected: format!("token < {VOCAB} and position < {}", state.max_tokens),
                actual: format!("token={token} position={}", state.position),
            });
        }
        state.token.copy_from_host(&[token])?;
        copy_bf16_row_to_f32_indexed_into_on_stream(
            VOCAB,
            HIDDEN,
            &self.embedding,
            &state.token,
            state.hidden.output(),
            &self.stream,
        )?;
        for layer in 0..LAYERS {
            let (previous, current) = state.layers.split_at_mut(layer);
            let input = if layer == 0 {
                &state.hidden
            } else {
                &previous[layer - 1].output
            };
            cache
                .with_append_page(target, |backend, page| {
                    self.layers[layer].run_one(
                        &mut current[0],
                        input,
                        LagunaLayerCache {
                            pool: backend.pool_mut(layer)?,
                            page_slot: page.slot(),
                            page_offset: target.page_offset(),
                            page_table: sequence.page_table.device(),
                            attention: state.compact_attention.for_layer(layer),
                        },
                        state.position,
                        &self.stream,
                    )
                })
                .map_err(laguna_cache_error)?;
        }
        Ok(())
    }

    fn forward_one(
        &self,
        sequence: &mut LagunaSequence,
        token: u32,
        cache: &mut LagunaSequenceCache,
    ) -> Result<()> {
        let target = self.reserve_token(sequence, cache)?;
        let hidden = self.forward_hidden_uncommitted(sequence, token, cache, target);
        if let Err(error) = hidden {
            return self.complete_token(sequence, cache, target, Err(error));
        }
        let state = &mut sequence.state;
        let last = &state
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Laguna model",
                detail: "model has no decoder layers".to_string(),
            })?
            .output;
        self.final_norm
            .run_into(last, &mut state.final_hidden, 1, &self.stream)?;
        let result = self
            .lm_head
            .run_into(&state.final_hidden, &mut state.logits, &self.stream);
        self.complete_token(sequence, cache, target, result)
    }
}

fn default_artifact_dir(model_dir: &Path) -> Result<PathBuf> {
    let cache_home = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .ok_or_else(|| Error::Format {
            label: "Laguna artifact directory",
            detail: "neither XDG_CACHE_HOME nor HOME is set".to_string(),
        })?;
    let revision = model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("local");
    Ok(cache_home
        .join("eider")
        .join("models")
        .join("poolside--Laguna-S-2.1-NVFP4")
        .join(revision)
        .join("laguna-experts-v1"))
}

impl LagunaDecodeState {
    /// Returns the number of tokens already represented in the K/V cache.
    pub fn len(&self) -> usize {
        self.position
    }

    /// Returns whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Returns the allocated sequence capacity.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns bytes owned by this active sequence.
    pub fn device_bytes(&self) -> usize {
        self.token.device_bytes()
            + self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(LagunaLayerWorkspace::device_bytes)
                .sum::<usize>()
            + self.compact_attention.device_bytes()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_index.device_bytes()
            + self.next_value.device_bytes()
            + self.sampler.device_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_checkpoint() -> Option<(PathBuf, PathBuf)> {
        let model_dir = std::env::var_os("LAGUNA_MODEL").map(PathBuf::from)?;
        let artifact_dir = std::env::var_os("LAGUNA_ARTIFACT_DIR").map(PathBuf::from)?;
        Some((model_dir, artifact_dir))
    }

    #[test]
    fn official_config_is_accepted() {
        let model_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/laguna-s-2.1-nvfp4");
        if !model_dir.join("config.json").is_file() {
            return;
        }
        let config = LagunaConfig::open(model_dir).expect("official config");
        assert_eq!(config.num_hidden_layers, LAYERS);
        assert_eq!(config.num_experts_per_tok, TOP_K);
    }

    #[test]
    fn yarn_frequency_endpoints_match_transformers_formula() {
        let frequencies = yarn_inverse_frequencies();
        assert_eq!(frequencies.len(), 32);
        assert!((frequencies[0] - 1.0).abs() < 1.0e-7);
        assert!(frequencies.windows(2).all(|pair| pair[0] > pair[1]));
        assert!((frequencies[31] - 9.418_306_5e-8).abs() < 1.0e-12);
    }

    #[test]
    #[ignore = "requires LAGUNA_MODEL and LAGUNA_ARTIFACT_DIR"]
    fn local_batched_prefill_is_stable_across_chunks() {
        let (model_dir, artifact_dir) =
            local_checkpoint().expect("set LAGUNA_MODEL and LAGUNA_ARTIFACT_DIR");
        let model =
            LagunaModel::load_with_artifact_dir(model_dir, artifact_dir).expect("load Laguna");
        let prompt = [9707, 3710, 9707, 3710, 9707, 3710, 9707, 3710];
        let final_token = 9707;
        let cache_stream = CudaStream::new_blocking().expect("cache stream");
        let mut cache =
            crate::runtime::laguna_sequence_cache::new_laguna_sequence_cache(&model, 2, 96)
                .expect("sequence cache");
        let mut batched = LagunaSequence::admit(&model, &mut cache, 32, &cache_stream)
            .expect("whole validation sequence");
        let mut workspace = model
            .new_prefill_batch_workspace(1, 128, 32)
            .expect("batch validation workspace");
        model
            .prefill_batch(
                &mut workspace,
                &mut [LagunaPrefillRow {
                    token_ids: &prompt,
                    sequence: &mut batched,
                }],
                &mut cache,
            )
            .expect("whole validation prefill");
        let whole = model
            .logits_one(&mut batched, final_token, &mut cache)
            .expect("whole validation logits");
        batched
            .finish(&mut cache, &cache_stream)
            .expect("finish whole validation sequence");

        let mut split = LagunaSequence::admit(&model, &mut cache, 32, &cache_stream)
            .expect("split validation sequence");
        for chunk in [&prompt[..3], &prompt[3..]] {
            model
                .prefill_batch(
                    &mut workspace,
                    &mut [LagunaPrefillRow {
                        token_ids: chunk,
                        sequence: &mut split,
                    }],
                    &mut cache,
                )
                .expect("split validation prefill");
        }
        let split_logits = model
            .logits_one(&mut split, final_token, &mut cache)
            .expect("split validation logits");
        let top = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .expect("non-empty logits")
                .0
        };
        let squared_error = split_logits
            .iter()
            .zip(&whole)
            .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
            .sum::<f64>();
        let expected_norm = whole
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
        assert_eq!(
            top(&split_logits),
            top(&whole),
            "split batched prefill selected a different token; nrmse={nrmse:.6}"
        );
        assert!(nrmse <= 0.12, "split batched prefill nrmse={nrmse:.6}");
        split
            .finish(&mut cache, &cache_stream)
            .expect("finish split validation sequence");

        let prompt = (0usize..64)
            .map(|index| if index.is_multiple_of(2) { 9707 } else { 3710 })
            .collect::<Vec<_>>();
        let mut long_prompt_workspace = model
            .new_prefill_batch_workspace(1, 128, 96)
            .expect("long-prompt validation workspace");
        let mut whole_state = LagunaSequence::admit(&model, &mut cache, 96, &cache_stream)
            .expect("whole long-prompt sequence");
        model
            .prefill_batch(
                &mut long_prompt_workspace,
                &mut [LagunaPrefillRow {
                    token_ids: &prompt,
                    sequence: &mut whole_state,
                }],
                &mut cache,
            )
            .expect("whole long-prompt prefill");
        let whole = model
            .logits_one(&mut whole_state, final_token, &mut cache)
            .expect("whole long-prompt logits");
        whole_state
            .finish(&mut cache, &cache_stream)
            .expect("finish whole long-prompt sequence");

        let mut split_state = LagunaSequence::admit(&model, &mut cache, 96, &cache_stream)
            .expect("split long-prompt sequence");
        for chunk in prompt.chunks(32) {
            model
                .prefill_batch(
                    &mut long_prompt_workspace,
                    &mut [LagunaPrefillRow {
                        token_ids: chunk,
                        sequence: &mut split_state,
                    }],
                    &mut cache,
                )
                .expect("split long-prompt prefill");
        }
        let split = model
            .logits_one(&mut split_state, final_token, &mut cache)
            .expect("split long-prompt logits");
        let squared_error = split
            .iter()
            .zip(&whole)
            .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
            .sum::<f64>();
        let expected_norm = whole
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
        assert_eq!(
            top(&split),
            top(&whole),
            "split long-prompt prefill selected a different token; nrmse={nrmse:.6}"
        );
        assert!(nrmse <= 0.12, "split long-prompt prefill nrmse={nrmse:.6}");
        split_state
            .finish(&mut cache, &cache_stream)
            .expect("finish split long-prompt sequence");
    }
}
