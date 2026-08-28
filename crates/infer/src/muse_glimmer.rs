//! Muse Glimmer dense text-model loading and inference.
//!
//! The text-only runtime consumes the released ModelOpt NVFP4 projections and
//! converts the checkpoint's BF16 attention gates and language head to NVFP4
//! during loading. Embeddings and normalization vectors remain BF16.

use crate::sm12x_cache::Sm12xCacheContext;
use eider_cuda::{
    CublasLt, CudaStream, DeviceBuffer, Error, Fp4TnMatmulPlan, GemmShape, ModelOptCheckpoint,
    ModelOptCublasLtWeight, ModelOptNvfp4Linear, Nvfp4Matrix, Nvfp4TnInputs, Result,
    Sm12xKvAttentionWorkspace, Sm12xKvPagePool, add_f32_into_on_stream, argmax_f32_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, copy_f32_rows_into_columns_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    sigmoid_mul_f32_into_on_stream, silu_mul_f32_into_on_stream,
};
use seqcache::RetainedSnapshot;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

mod batch;
mod dflash;
mod sequence;

pub use dflash::{DFlashConfig, DFlashModel, MuseGlimmerDFlashCycle};
pub(crate) use sequence::{
    MuseGlimmerAppend, muse_glimmer_cache_error, new_muse_glimmer_sequence_cache_with_budget,
};
pub use sequence::{
    MuseGlimmerSequence, MuseGlimmerSequenceCache, new_muse_glimmer_sequence_cache,
};

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// Validated language-model dimensions and execution parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub post_norm_eps: f32,
    pub qk_scale_factor: f32,
    pub output_multiplier: f32,
    pub final_logit_softcapping: f32,
    pub layer_types: Vec<String>,
    pub layer_rope_theta: Vec<f32>,
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
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    post_norm_eps: f32,
    qk_scale_factor: f32,
    output_multiplier: f32,
    final_logit_softcapping: f32,
    layer_types: Vec<String>,
    layer_rope_theta: Vec<f32>,
}

impl MuseGlimmerConfig {
    /// Reads and validates `config.json` from a Muse Glimmer checkpoint.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text = fs::read_to_string(&path).map_err(|error| Error::Format {
            label: "Muse Glimmer config",
            detail: format!("{}: {error}", path.display()),
        })?;
        Self::from_json(&text)
    }

    fn from_json(text: &str) -> Result<Self> {
        let file: FileConfig = serde_json::from_str(text).map_err(|error| Error::Format {
            label: "Muse Glimmer config JSON",
            detail: error.to_string(),
        })?;
        if file.model_type != "muse_glimmer" {
            return Err(Error::Format {
                label: "Muse Glimmer config",
                detail: format!("expected model_type=muse_glimmer, got {}", file.model_type),
            });
        }
        let config = file.text_config;
        let dimensions_valid = config.hidden_size != 0
            && config.intermediate_size != 0
            && config.num_hidden_layers != 0
            && config.num_attention_heads != 0
            && config.num_key_value_heads != 0
            && config.head_dim != 0
            && config.sliding_window != 0
            && config.max_position_embeddings != 0
            && config.vocab_size != 0
            && config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
            && config.hidden_size.is_multiple_of(64)
            && config.intermediate_size.is_multiple_of(64)
            && (config.num_attention_heads * config.head_dim).is_multiple_of(64)
            && (config.num_key_value_heads * config.head_dim).is_multiple_of(64);
        let scalars_valid = [
            config.rms_norm_eps,
            config.post_norm_eps,
            config.qk_scale_factor,
            config.output_multiplier,
            config.final_logit_softcapping,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0);
        let layout_valid = config.layer_types.len() == config.num_hidden_layers
            && config.layer_rope_theta.len() == config.num_hidden_layers
            && config
                .layer_types
                .iter()
                .all(|kind| matches!(kind.as_str(), "sliding_attention" | "full_attention"))
            && config
                .layer_types
                .iter()
                .zip(&config.layer_rope_theta)
                .all(|(kind, theta)| {
                    theta.is_finite()
                        && ((*theta > 0.0 && kind == "sliding_attention")
                            || (*theta == 0.0 && kind == "full_attention"))
                });
        if !dimensions_valid || !scalars_valid || !layout_valid {
            return Err(Error::Shape {
                label: "Muse Glimmer config",
                expected: "valid SM121 dimensions, positive finite scales, and matching local-RoPE/global-NoPE layer metadata".to_string(),
                actual: format!(
                    "layers={} types={} rope={} hidden={} intermediate={} q_heads={} kv_heads={} head_dim={}",
                    config.num_hidden_layers,
                    config.layer_types.len(),
                    config.layer_rope_theta.len(),
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                ),
            });
        }
        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            sliding_window: config.sliding_window,
            max_position_embeddings: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            rms_norm_eps: config.rms_norm_eps,
            post_norm_eps: config.post_norm_eps,
            qk_scale_factor: config.qk_scale_factor,
            output_multiplier: config.output_multiplier,
            final_logit_softcapping: config.final_logit_softcapping,
            layer_types: config.layer_types,
            layer_rope_theta: config.layer_rope_theta,
        })
    }

    fn is_local_layer(&self, layer: usize) -> Result<bool> {
        self.layer_types
            .get(layer)
            .map(|kind| kind == "sliding_attention")
            .ok_or_else(|| Error::Shape {
                label: "Muse Glimmer layer index",
                expected: format!("layer < {}", self.num_hidden_layers),
                actual: layer.to_string(),
            })
    }
}

#[derive(Clone, Debug)]
struct MuseGlimmerCheckpoint {
    config: MuseGlimmerConfig,
    checkpoint: ModelOptCheckpoint,
}

impl MuseGlimmerCheckpoint {
    fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        Ok(Self {
            config: MuseGlimmerConfig::open(model_dir)?,
            checkpoint: ModelOptCheckpoint::open(model_dir)?,
        })
    }

    fn load_nvfp4_linear(&self, prefix: &str) -> Result<MuseNvfp4Linear> {
        MuseNvfp4Linear::from_modelopt(prefix, self.checkpoint.load_nvfp4_linear(prefix)?)
    }

    fn load_bf16_nvfp4_linear(&self, prefix: &str) -> Result<MuseNvfp4Linear> {
        let tensor = format!("{prefix}.weight");
        let info = self.checkpoint.tensor_info(&tensor)?;
        if info.dtype != "BF16" || info.shape.len() != 2 {
            return Err(Error::Shape {
                label: "Muse Glimmer BF16-to-NVFP4 linear",
                expected: "BF16 [rows, cols]".to_string(),
                actual: format!("{} {:?} for {tensor}", info.dtype, info.shape),
            });
        }
        let bytes = self
            .checkpoint
            .open_shard_for_tensor(&tensor)?
            .read_tensor_bytes(&tensor)?;
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        MuseNvfp4Linear::from_modelopt(
            prefix,
            ModelOptNvfp4Linear::quantize_bf16(prefix, info.shape[0], info.shape[1], &values)?,
        )
    }

    fn load_bf16_vector(&self, tensor: &str, width: usize) -> Result<Vec<f32>> {
        let info = self.checkpoint.tensor_info(tensor)?;
        if info.dtype != "BF16" || info.shape.as_slice() != [width] {
            return Err(Error::Shape {
                label: "Muse Glimmer BF16 vector",
                expected: format!("BF16 [{width}]"),
                actual: format!("{} {:?} for {tensor}", info.dtype, info.shape),
            });
        }
        let shard = self.checkpoint.open_shard_for_tensor(tensor)?;
        Ok(shard
            .read_tensor_bytes(tensor)?
            .chunks_exact(2)
            .map(|bytes| eider_cuda::format::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect())
    }

    fn load_bf16_matrix(
        &self,
        tensor: &str,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceBuffer<u16>> {
        let info = self.checkpoint.tensor_info(tensor)?;
        if info.dtype != "BF16" || info.shape.as_slice() != [rows, cols] {
            return Err(Error::Shape {
                label: "Muse Glimmer BF16 matrix",
                expected: format!("BF16 [{rows}, {cols}]"),
                actual: format!("{} {:?} for {tensor}", info.dtype, info.shape),
            });
        }
        let bytes = self
            .checkpoint
            .open_shard_for_tensor(tensor)?
            .read_tensor_bytes(tensor)?;
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        DeviceBuffer::from_host(&values)
    }
}

struct MuseNvfp4Linear {
    name: String,
    weight: ModelOptCublasLtWeight,
    out_features: usize,
    in_features: usize,
}

struct MuseNvfp4LinearWorkspace {
    activation: Nvfp4Matrix,
    output: DeviceBuffer<f32>,
}

impl MuseNvfp4LinearWorkspace {
    fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    fn device_bytes(&self) -> usize {
        self.activation.device_bytes() + self.output.device_bytes()
    }
}

impl MuseNvfp4Linear {
    fn from_modelopt(name: &str, weight: ModelOptNvfp4Linear) -> Result<Self> {
        if !weight.out_features.is_multiple_of(16) || !weight.in_features.is_multiple_of(64) {
            return Err(Error::Shape {
                label: "Muse Glimmer NVFP4 linear",
                expected: "out_features divisible by 16 and in_features divisible by 64"
                    .to_string(),
                actual: format!(
                    "out_features={} in_features={}",
                    weight.out_features, weight.in_features
                ),
            });
        }
        let out_features = weight.out_features;
        let in_features = weight.in_features;
        Ok(Self {
            name: name.to_string(),
            weight: weight.as_cublaslt_weight()?,
            out_features,
            in_features,
        })
    }

    fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    fn new_workspace(&self) -> Result<MuseNvfp4LinearWorkspace> {
        Ok(MuseNvfp4LinearWorkspace {
            activation: Nvfp4Matrix::zeroed_col_major(self.in_features, 1)?,
            output: DeviceBuffer::zeroed(self.out_features)?,
        })
    }

    fn run_into(
        &self,
        lt: &CublasLt,
        plans: &MuseLinearPlans,
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseNvfp4LinearWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.in_features || workspace.output().len() != self.out_features {
            return Err(Error::Shape {
                label: "Muse Glimmer NVFP4 linear buffers",
                expected: format!("input={} output={}", self.in_features, self.out_features),
                actual: format!("input={} output={}", input.len(), workspace.output().len()),
            });
        }
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            self.in_features,
            1,
            input,
            &mut workspace.activation,
            self.weight.input_scale(),
            stream,
        )?;
        plans
            .for_linear(self)?
            .run_with_alpha_beta_f32_inout_buffer_on_stream(
                lt,
                Nvfp4TnInputs::new(self.weight.matrix(), &workspace.activation),
                workspace.output.inout(),
                self.weight.matmul_alpha(),
                0.0,
                stream,
            )
            .map_err(|error| Error::Format {
                label: "Muse Glimmer NVFP4 linear execution",
                detail: format!(
                    "{} [{}x{}]: {error}",
                    self.name, self.out_features, self.in_features
                ),
            })
    }

    fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

struct MuseRmsNorm {
    weight: DeviceBuffer<f32>,
    eps: f32,
}

impl MuseRmsNorm {
    fn load(
        checkpoint: &MuseGlimmerCheckpoint,
        tensor: &str,
        width: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&checkpoint.load_bf16_vector(tensor, width)?)?,
            eps,
        })
    }

    fn load_centered(
        checkpoint: &MuseGlimmerCheckpoint,
        tensor: &str,
        width: usize,
        eps: f32,
    ) -> Result<Self> {
        let mut weight = checkpoint.load_bf16_vector(tensor, width)?;
        for value in &mut weight {
            *value += 1.0;
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
            eps,
        })
    }

    fn constant(width: usize, scale: f32, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&vec![scale; width])?,
            eps,
        })
    }

    fn run_into(
        &self,
        rows: usize,
        width: usize,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.weight.len() != width {
            return Err(Error::Shape {
                label: "Muse Glimmer RMSNorm weight",
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

struct MuseMlp {
    gate: MuseNvfp4Linear,
    up: MuseNvfp4Linear,
    down: MuseNvfp4Linear,
    intermediate_size: usize,
}

struct MuseMlpWorkspace {
    gate: MuseNvfp4LinearWorkspace,
    up: MuseNvfp4LinearWorkspace,
    activated: DeviceBuffer<f32>,
    down: MuseNvfp4LinearWorkspace,
}

impl MuseMlpWorkspace {
    fn output(&self) -> &DeviceBuffer<f32> {
        self.down.output()
    }
}

impl MuseMlp {
    fn load(checkpoint: &MuseGlimmerCheckpoint, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let gate = checkpoint.load_nvfp4_linear(&format!("{prefix}.gate_proj"))?;
        let up = checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?;
        let down = checkpoint.load_nvfp4_linear(&format!("{prefix}.down_proj"))?;
        let config = &checkpoint.config;
        if gate.shape() != (config.intermediate_size, config.hidden_size)
            || up.shape() != (config.intermediate_size, config.hidden_size)
            || down.shape() != (config.hidden_size, config.intermediate_size)
        {
            return Err(Error::Shape {
                label: "Muse Glimmer MLP projections",
                expected: format!(
                    "gate/up={}x{} down={}x{}",
                    config.intermediate_size,
                    config.hidden_size,
                    config.hidden_size,
                    config.intermediate_size
                ),
                actual: format!(
                    "gate={:?} up={:?} down={:?}",
                    gate.shape(),
                    up.shape(),
                    down.shape()
                ),
            });
        }
        Ok(Self {
            gate,
            up,
            down,
            intermediate_size: config.intermediate_size,
        })
    }

    fn new_workspace(&self) -> Result<MuseMlpWorkspace> {
        Ok(MuseMlpWorkspace {
            gate: self.gate.new_workspace()?,
            up: self.up.new_workspace()?,
            activated: DeviceBuffer::zeroed(self.intermediate_size)?,
            down: self.down.new_workspace()?,
        })
    }

    fn run_into(
        &self,
        lt: &CublasLt,
        plans: &MuseLinearPlans,
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseMlpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.gate
            .run_into(lt, plans, input, &mut workspace.gate, stream)?;
        self.up
            .run_into(lt, plans, input, &mut workspace.up, stream)?;
        silu_mul_f32_into_on_stream(
            workspace.gate.output(),
            workspace.up.output(),
            workspace.activated.output(),
            stream,
        )?;
        self.down
            .run_into(lt, plans, &workspace.activated, &mut workspace.down, stream)
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

struct MuseAttention {
    q: MuseNvfp4Linear,
    k: MuseNvfp4Linear,
    v: MuseNvfp4Linear,
    gate: MuseNvfp4Linear,
    output: MuseNvfp4Linear,
    q_norm: MuseRmsNorm,
    k_norm: MuseRmsNorm,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_theta: Option<f32>,
    window: Option<usize>,
}

struct MuseAttentionWorkspace {
    q: MuseNvfp4LinearWorkspace,
    k: MuseNvfp4LinearWorkspace,
    v: MuseNvfp4LinearWorkspace,
    gate: MuseNvfp4LinearWorkspace,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_positioned: DeviceBuffer<f32>,
    k_positioned: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    output: MuseNvfp4LinearWorkspace,
}

struct MuseAttentionCache<'a> {
    pool: &'a mut Sm12xKvPagePool,
    page_slot: usize,
    page_offset: usize,
    page_table: &'a DeviceBuffer<u32>,
}

impl MuseAttentionWorkspace {
    fn output(&self) -> &DeviceBuffer<f32> {
        self.output.output()
    }
}

impl MuseAttention {
    fn load(checkpoint: &MuseGlimmerCheckpoint, layer: usize) -> Result<Self> {
        let config = &checkpoint.config;
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let q = checkpoint.load_nvfp4_linear(&format!("{prefix}.q_proj"))?;
        let k = checkpoint.load_nvfp4_linear(&format!("{prefix}.k_proj"))?;
        let v = checkpoint.load_nvfp4_linear(&format!("{prefix}.v_proj"))?;
        let gate = checkpoint.load_bf16_nvfp4_linear(&format!("{prefix}.gate_proj"))?;
        let output = checkpoint.load_nvfp4_linear(&format!("{prefix}.o_proj"))?;
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        if q.shape() != (q_width, config.hidden_size)
            || k.shape() != (kv_width, config.hidden_size)
            || v.shape() != (kv_width, config.hidden_size)
            || output.shape() != (config.hidden_size, q_width)
        {
            return Err(Error::Shape {
                label: "Muse Glimmer attention projections",
                expected: format!(
                    "q={q_width}x{}, k/v={kv_width}x{}, o={}x{q_width}",
                    config.hidden_size, config.hidden_size, config.hidden_size
                ),
                actual: format!(
                    "q={:?} k={:?} v={:?} o={:?}",
                    q.shape(),
                    k.shape(),
                    v.shape(),
                    output.shape()
                ),
            });
        }
        if gate.shape() != (q_width, config.hidden_size) {
            return Err(Error::Shape {
                label: "Muse Glimmer attention gate",
                expected: format!("[{q_width}, {}]", config.hidden_size),
                actual: format!("{:?}", gate.shape()),
            });
        }
        let local = config.is_local_layer(layer)?;
        let theta = config.layer_rope_theta[layer];
        Ok(Self {
            q,
            k,
            v,
            gate,
            output,
            q_norm: MuseRmsNorm::constant(
                config.head_dim,
                config.qk_scale_factor,
                config.rms_norm_eps,
            )?,
            k_norm: MuseRmsNorm::constant(config.head_dim, 1.0, config.rms_norm_eps)?,
            q_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: local.then_some(theta),
            window: local.then_some(config.sliding_window),
        })
    }

    fn new_workspace(&self) -> Result<MuseAttentionWorkspace> {
        let q_width = self.q_heads * self.head_dim;
        let kv_width = self.kv_heads * self.head_dim;
        Ok(MuseAttentionWorkspace {
            q: self.q.new_workspace()?,
            k: self.k.new_workspace()?,
            v: self.v.new_workspace()?,
            gate: self.gate.new_workspace()?,
            q_normed: DeviceBuffer::zeroed(q_width)?,
            k_normed: DeviceBuffer::zeroed(kv_width)?,
            q_positioned: DeviceBuffer::zeroed(q_width)?,
            k_positioned: DeviceBuffer::zeroed(kv_width)?,
            attended: DeviceBuffer::zeroed(q_width)?,
            gated: DeviceBuffer::zeroed(q_width)?,
            output: self.output.new_workspace()?,
        })
    }

    fn new_compact_attention_workspace(
        &self,
        max_tokens: usize,
    ) -> Result<Sm12xKvAttentionWorkspace> {
        let attention_capacity = max_tokens.div_ceil(eider_cuda::SM12X_KV_PAGE_TOKENS)
            * eider_cuda::SM12X_KV_PAGE_TOKENS;
        Sm12xKvAttentionWorkspace::new_gqa(
            attention_capacity,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_into(
        &self,
        lt: &CublasLt,
        plans: &MuseLinearPlans,
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseAttentionWorkspace,
        cache: MuseAttentionCache<'_>,
        compact_attention: &mut Sm12xKvAttentionWorkspace,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.q
            .run_into(lt, plans, input, &mut workspace.q, stream)?;
        self.k
            .run_into(lt, plans, input, &mut workspace.k, stream)?;
        self.v
            .run_into(lt, plans, input, &mut workspace.v, stream)?;
        self.gate
            .run_into(lt, plans, input, &mut workspace.gate, stream)?;
        round_f32_to_bf16_in_place_on_stream(workspace.gate.output.inout(), stream)?;
        self.q_norm.run_into(
            self.q_heads,
            self.head_dim,
            workspace.q.output(),
            &mut workspace.q_normed,
            stream,
        )?;
        self.k_norm.run_into(
            self.kv_heads,
            self.head_dim,
            workspace.k.output(),
            &mut workspace.k_normed,
            stream,
        )?;
        if let Some(theta) = self.rope_theta {
            rope_neox_f32_into_on_stream(
                self.q_heads,
                self.head_dim,
                &workspace.q_normed,
                workspace.q_positioned.output(),
                position,
                theta,
                stream,
            )?;
            rope_neox_f32_into_on_stream(
                self.kv_heads,
                self.head_dim,
                &workspace.k_normed,
                workspace.k_positioned.output(),
                position,
                theta,
                stream,
            )?;
        } else {
            workspace.q_positioned.copy_prefix_from_device_on_stream(
                &workspace.q_normed,
                workspace.q_normed.len(),
                stream,
            )?;
            workspace.k_positioned.copy_prefix_from_device_on_stream(
                &workspace.k_normed,
                workspace.k_normed.len(),
                stream,
            )?;
        }
        cache.pool.append_at_offsets_on_stream(
            cache.page_slot,
            cache.page_offset,
            &workspace.k_positioned,
            0,
            workspace.v.output(),
            0,
            stream,
        )?;
        compact_attention.attention_paged_window_offsets_into_on_stream(
            cache.pool,
            cache.page_table,
            position + 1,
            &workspace.q_positioned,
            0,
            workspace.attended.output(),
            0,
            self.window
                .map_or(0, |window| (position + 1).saturating_sub(window)),
            stream,
        )?;
        sigmoid_mul_f32_into_on_stream(
            workspace.gate.output(),
            &workspace.attended,
            workspace.gated.output(),
            stream,
        )?;
        self.output
            .run_into(lt, plans, &workspace.gated, &mut workspace.output, stream)
    }

    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.gate.device_bytes()
            + self.output.device_bytes()
            + self.q_norm.device_bytes()
            + self.k_norm.device_bytes()
    }
}

struct MuseDecoderLayer {
    input_norm: MuseRmsNorm,
    attention: MuseAttention,
    post_attention_norm: MuseRmsNorm,
    pre_feedforward_norm: MuseRmsNorm,
    mlp: MuseMlp,
    post_feedforward_norm: MuseRmsNorm,
}

struct MuseDecoderLayerWorkspace {
    attention: MuseAttentionWorkspace,
    mlp: MuseMlpWorkspace,
    normalized: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    feedforward_input: DeviceBuffer<f32>,
    feedforward_output: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl MuseDecoderLayer {
    fn load(checkpoint: &MuseGlimmerCheckpoint, layer: usize) -> Result<Self> {
        let config = &checkpoint.config;
        let prefix = format!("model.language_model.layers.{layer}");
        Ok(Self {
            input_norm: MuseRmsNorm::load_centered(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            attention: MuseAttention::load(checkpoint, layer)?,
            post_attention_norm: MuseRmsNorm::load_centered(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
                config.post_norm_eps,
            )?,
            pre_feedforward_norm: MuseRmsNorm::load_centered(
                checkpoint,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            mlp: MuseMlp::load(checkpoint, layer)?,
            post_feedforward_norm: MuseRmsNorm::load_centered(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                config.hidden_size,
                config.post_norm_eps,
            )?,
        })
    }

    fn new_workspace(&self) -> Result<MuseDecoderLayerWorkspace> {
        let hidden = self.attention.q.shape().1;
        Ok(MuseDecoderLayerWorkspace {
            attention: self.attention.new_workspace()?,
            mlp: self.mlp.new_workspace()?,
            normalized: DeviceBuffer::zeroed(hidden)?,
            residual: DeviceBuffer::zeroed(hidden)?,
            feedforward_input: DeviceBuffer::zeroed(hidden)?,
            feedforward_output: DeviceBuffer::zeroed(hidden)?,
            output: DeviceBuffer::zeroed(hidden)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_into(
        &self,
        lt: &CublasLt,
        plans: &MuseLinearPlans,
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseDecoderLayerWorkspace,
        cache: MuseAttentionCache<'_>,
        compact_attention: &mut Sm12xKvAttentionWorkspace,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden = self.attention.q.shape().1;
        self.input_norm
            .run_into(1, hidden, input, &mut workspace.normalized, stream)?;
        self.attention.run_into(
            lt,
            plans,
            &workspace.normalized,
            &mut workspace.attention,
            cache,
            compact_attention,
            position,
            stream,
        )?;
        self.post_attention_norm.run_into(
            1,
            hidden,
            workspace.attention.output(),
            &mut workspace.normalized,
            stream,
        )?;
        add_f32_into_on_stream(
            input,
            &workspace.normalized,
            workspace.residual.output(),
            stream,
        )?;
        self.pre_feedforward_norm.run_into(
            1,
            hidden,
            &workspace.residual,
            &mut workspace.feedforward_input,
            stream,
        )?;
        self.mlp.run_into(
            lt,
            plans,
            &workspace.feedforward_input,
            &mut workspace.mlp,
            stream,
        )?;
        self.post_feedforward_norm.run_into(
            1,
            hidden,
            workspace.mlp.output(),
            &mut workspace.feedforward_output,
            stream,
        )?;
        add_f32_into_on_stream(
            &workspace.residual,
            &workspace.feedforward_output,
            workspace.output.output(),
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.input_norm.device_bytes()
            + self.attention.device_bytes()
            + self.post_attention_norm.device_bytes()
            + self.pre_feedforward_norm.device_bytes()
            + self.mlp.device_bytes()
            + self.post_feedforward_norm.device_bytes()
    }
}

struct MuseCompactAttentionWorkspaces {
    local: Option<Sm12xKvAttentionWorkspace>,
    global: Option<Sm12xKvAttentionWorkspace>,
}

impl MuseCompactAttentionWorkspaces {
    fn new(layers: &[MuseDecoderLayer], max_tokens: usize) -> Result<Self> {
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
            label: "Muse Glimmer compact attention workspace",
            detail: format!(
                "missing {} workspace",
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

struct MuseLinearPlan {
    out_features: usize,
    in_features: usize,
    plan: Fp4TnMatmulPlan,
}

struct MuseLinearPlans {
    plans: Vec<MuseLinearPlan>,
}

impl MuseLinearPlans {
    fn new(lt: &CublasLt, layers: &[MuseDecoderLayer], lm_head: &MuseNvfp4Linear) -> Result<Self> {
        let first = layers.first().ok_or_else(|| Error::Format {
            label: "Muse Glimmer linear plans",
            detail: "model has no decoder layers".to_string(),
        })?;
        let representatives = [
            &first.attention.q,
            &first.attention.k,
            &first.attention.output,
            &first.mlp.gate,
            &first.mlp.down,
            lm_head,
        ];
        let mut plans = Vec::with_capacity(representatives.len());
        for linear in representatives {
            let activation = Nvfp4Matrix::zeroed_col_major(linear.in_features, 1)?;
            let shape = GemmShape::new(linear.out_features, 1, linear.in_features);
            plans.push(MuseLinearPlan {
                out_features: linear.out_features,
                in_features: linear.in_features,
                plan: Fp4TnMatmulPlan::new_f32_output_for_shape(
                    lt,
                    shape,
                    Nvfp4TnInputs::new(linear.weight.matrix(), &activation),
                    8 << 20,
                )?,
            });
        }
        Ok(Self { plans })
    }

    fn for_linear(&self, linear: &MuseNvfp4Linear) -> Result<&Fp4TnMatmulPlan> {
        self.plans
            .iter()
            .find(|plan| {
                (plan.out_features, plan.in_features) == (linear.out_features, linear.in_features)
            })
            .map(|plan| &plan.plan)
            .ok_or_else(|| Error::Shape {
                label: "Muse Glimmer linear plan",
                expected: "one of the model's dense projection shapes".to_string(),
                actual: format!("{}x{}", linear.out_features, linear.in_features),
            })
    }

    fn device_bytes(&self) -> usize {
        self.plans
            .iter()
            .map(|plan| plan.plan.workspace_bytes())
            .sum()
    }
}

/// Complete resident Muse Glimmer text model.
pub struct MuseGlimmerModel {
    model_id: u64,
    config: MuseGlimmerConfig,
    embedding: DeviceBuffer<u16>,
    embedding_norm: MuseRmsNorm,
    layers: Vec<MuseDecoderLayer>,
    lt: CublasLt,
    linear_plans: MuseLinearPlans,
    final_norm: MuseRmsNorm,
    lm_head: MuseNvfp4Linear,
    dflash: Option<DFlashModel>,
    stream: CudaStream,
}

/// Mutable execution and compact K/V state for one text sequence.
pub struct MuseGlimmerDecodeState {
    model_id: u64,
    hidden: DeviceBuffer<f32>,
    embedding_output: DeviceBuffer<f32>,
    layers: Vec<MuseDecoderLayerWorkspace>,
    compact_attention: MuseCompactAttentionWorkspaces,
    final_hidden: DeviceBuffer<f32>,
    lm_head: MuseNvfp4LinearWorkspace,
    next_index: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    verification: Option<Box<batch::MuseTargetBatchWorkspace>>,
    dflash_state: Option<Box<dflash::DFlashSequenceState>>,
    batch_logits_row: Option<usize>,
    position: usize,
    max_tokens: usize,
}

/// Immutable aligned Muse Glimmer prompt-prefix state.
pub struct MuseGlimmerSequenceSnapshot {
    model_id: u64,
    position: usize,
    dflash: Option<dflash::DFlashSequenceCheckpoint>,
}

/// One greedy next-token result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuseGlimmerNextToken {
    pub token: u32,
    pub logit: f32,
}

impl MuseGlimmerModel {
    /// Loads the checkpoint's complete text backbone into resident storage.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = MuseGlimmerCheckpoint::open(model_dir)?;
        let config = checkpoint.config.clone();
        let lt = CublasLt::new()?;
        let embedding = checkpoint.load_bf16_matrix(
            "model.language_model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;
        let embedding_norm = MuseRmsNorm::constant(config.hidden_size, 1.0, config.rms_norm_eps)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(MuseDecoderLayer::load(&checkpoint, layer)?);
            let bytes = embedding.device_bytes()
                + embedding_norm.device_bytes()
                + layers
                    .iter()
                    .map(MuseDecoderLayer::device_bytes)
                    .sum::<usize>();
            info!(
                layer,
                device_weight_gib = bytes as f64 / (1u64 << 30) as f64,
                "loaded Muse Glimmer layer"
            );
        }
        let final_norm = MuseRmsNorm::load(
            &checkpoint,
            "model.language_model.norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let lm_head = checkpoint.load_bf16_nvfp4_linear("lm_head")?;
        if lm_head.shape() != (config.vocab_size, config.hidden_size) {
            return Err(Error::Shape {
                label: "Muse Glimmer language head",
                expected: format!("[{}, {}]", config.vocab_size, config.hidden_size),
                actual: format!("{:?}", lm_head.shape()),
            });
        }
        let linear_plans = MuseLinearPlans::new(&lt, &layers, &lm_head)?;
        Ok(Self {
            model_id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            embedding,
            embedding_norm,
            layers,
            lt,
            linear_plans,
            final_norm,
            lm_head,
            dflash: None,
            stream: CudaStream::new_non_blocking()?,
        })
    }

    /// Loads Muse Glimmer together with Meta's official DFlash companion.
    pub fn load_with_dflash(
        model_dir: impl AsRef<Path>,
        dflash_gguf: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut model = Self::load(model_dir)?;
        model.dflash = Some(DFlashModel::load(dflash_gguf, &model.config)?);
        Ok(model)
    }

    /// Returns the validated checkpoint configuration.
    pub fn config(&self) -> &MuseGlimmerConfig {
        &self.config
    }

    /// Returns bytes retained by model weights and constant buffers.
    pub fn device_bytes(&self) -> usize {
        self.embedding.device_bytes()
            + self.embedding_norm.device_bytes()
            + self
                .layers
                .iter()
                .map(MuseDecoderLayer::device_bytes)
                .sum::<usize>()
            + self.linear_plans.device_bytes()
            + self.final_norm.device_bytes()
            + self.lm_head.device_bytes()
            + self.dflash.as_ref().map_or(0, DFlashModel::device_bytes)
    }

    /// Waits for work submitted by this model instance.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Allocates execution scratch and private state for one sequence.
    pub fn new_sequence_state(&self, max_tokens: usize) -> Result<MuseGlimmerDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_position_embeddings {
            return Err(Error::Shape {
                label: "Muse Glimmer decode capacity",
                expected: format!("1..={}", self.config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        let verification = self
            .dflash
            .as_ref()
            .map(|_| batch::MuseTargetBatchWorkspace::new(self, max_tokens).map(Box::new))
            .transpose()?;
        let dflash_state = self
            .dflash
            .as_ref()
            .map(|dflash| {
                dflash
                    .new_sequence_state(max_tokens, self.config.vocab_size)
                    .map(Box::new)
            })
            .transpose()?;
        Ok(MuseGlimmerDecodeState {
            model_id: self.model_id,
            hidden: DeviceBuffer::zeroed(self.config.hidden_size)?,
            embedding_output: DeviceBuffer::zeroed(self.config.hidden_size)?,
            layers: self
                .layers
                .iter()
                .map(MuseDecoderLayer::new_workspace)
                .collect::<Result<Vec<_>>>()?,
            compact_attention: MuseCompactAttentionWorkspaces::new(&self.layers, max_tokens)?,
            final_hidden: DeviceBuffer::zeroed(self.config.hidden_size)?,
            lm_head: self.lm_head.new_workspace()?,
            next_index: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            verification,
            dflash_state,
            batch_logits_row: None,
            position: 0,
            max_tokens,
        })
    }

    /// Advances one token without materializing vocabulary logits.
    pub fn consume_one(
        &self,
        sequence: &mut MuseGlimmerSequence,
        token: u32,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<()> {
        self.forward_hidden(sequence, token, cache)
    }

    /// Advances one token and copies transformed vocabulary logits to the host.
    pub fn logits_one(
        &self,
        sequence: &mut MuseGlimmerSequence,
        token: u32,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<Vec<f32>> {
        self.forward_one(sequence, token, cache)?;
        self.logits_to_host(sequence)
    }

    /// Advances one token and performs greedy selection.
    pub fn decode_one(
        &self,
        sequence: &mut MuseGlimmerSequence,
        token: u32,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<MuseGlimmerNextToken> {
        self.forward_one(sequence, token, cache)?;
        let (token, logit) = self.argmax_with_logit(sequence)?;
        Ok(MuseGlimmerNextToken { token, logit })
    }

    /// Copies the most recent transformed vocabulary logits to the host.
    pub fn logits_to_host(&self, sequence: &MuseGlimmerSequence) -> Result<Vec<f32>> {
        let state = &sequence.state;
        if state.position == 0 {
            return Err(Error::Format {
                label: "Muse Glimmer logits",
                detail: "no token has been evaluated".to_string(),
            });
        }
        let mut logits = state
            .lm_head
            .output()
            .copy_to_host(&self.stream)?
            .into_vec();
        for logit in &mut logits {
            *logit = self.transform_logit(*logit);
        }
        Ok(logits)
    }

    /// Returns the greedy token and transformed logit without copying the vocabulary.
    pub fn argmax_with_logit(&self, sequence: &mut MuseGlimmerSequence) -> Result<(u32, f32)> {
        let state = &mut sequence.state;
        if state.position == 0 {
            return Err(Error::Format {
                label: "Muse Glimmer logits",
                detail: "no token has been evaluated".to_string(),
            });
        }
        if let Some(row) = state.batch_logits_row.take() {
            let verification = state
                .verification
                .as_ref()
                .expect("DFlash verification workspace");
            let tokens = verification.argmax_indices.copy_to_host(&self.stream)?;
            let values = verification.argmax_values.copy_to_host(&self.stream)?;
            return Ok((tokens[row], self.transform_logit(values[row])));
        }
        argmax_f32_into_on_stream(
            state.lm_head.output(),
            state.next_index.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        let token = state.next_index.copy_to_host(&self.stream)?[0];
        let logit = self.transform_logit(state.next_value.copy_to_host(&self.stream)?[0]);
        Ok((token, logit))
    }

    fn forward_hidden(
        &self,
        sequence: &mut MuseGlimmerSequence,
        token: u32,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<()> {
        let state = &mut sequence.state;
        state.batch_logits_row = None;
        if state.model_id != self.model_id {
            return Err(Error::Format {
                label: "Muse Glimmer decode state",
                detail: "state belongs to a different model instance".to_string(),
            });
        }
        if token as usize >= self.config.vocab_size || state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "Muse Glimmer decode token",
                expected: format!(
                    "token < {} and position < {}",
                    self.config.vocab_size, state.max_tokens
                ),
                actual: format!("token={token} position={}", state.position),
            });
        }
        let position = state.position;
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                1,
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(muse_glimmer_cache_error)?;
        let result = (|| {
            copy_bf16_row_to_f32_into_on_stream(
                self.config.vocab_size,
                self.config.hidden_size,
                token as usize,
                &self.embedding,
                state.hidden.output(),
                &self.stream,
            )?;
            self.embedding_norm.run_into(
                1,
                self.config.hidden_size,
                &state.hidden,
                &mut state.embedding_output,
                &self.stream,
            )?;
            for layer_index in 0..self.layers.len() {
                let local = self.layers[layer_index].attention.window.is_some();
                let (previous, current) = state.layers.split_at_mut(layer_index);
                let input = if layer_index == 0 {
                    &state.embedding_output
                } else {
                    &previous[layer_index - 1].output
                };
                if let Some(dflash) = &self.dflash
                    && let Some(extract_index) = dflash
                        .config
                        .target_layers
                        .iter()
                        .position(|&extract| extract == layer_index)
                {
                    copy_f32_rows_into_columns_on_stream(
                        1,
                        self.config.hidden_size,
                        dflash.config.target_layers.len() * self.config.hidden_size,
                        extract_index * self.config.hidden_size,
                        input,
                        state
                            .verification
                            .as_mut()
                            .expect("DFlash verification workspace")
                            .features
                            .output(),
                        &self.stream,
                    )?;
                }
                cache
                    .with_append_pages(&reservation, |backend, pages| {
                        let page = pages.iter().next().expect("one Muse append page");
                        let segment = page.segment();
                        self.layers[layer_index].run_into(
                            &self.lt,
                            &self.linear_plans,
                            input,
                            &mut current[0],
                            MuseAttentionCache {
                                pool: backend.pool_mut(layer_index)?,
                                page_slot: page.page().slot(),
                                page_offset: segment.page_offset(),
                                page_table: sequence.page_table.device(),
                            },
                            state.compact_attention.for_layer_mut(local)?,
                            position,
                            &self.stream,
                        )
                    })
                    .map_err(muse_glimmer_cache_error)?;
            }
            if let Some(dflash) = &self.dflash {
                let MuseGlimmerDecodeState {
                    verification,
                    dflash_state,
                    ..
                } = state;
                dflash.inject_features(
                    &verification
                        .as_ref()
                        .expect("DFlash verification workspace")
                        .features,
                    dflash_state.as_mut().expect("DFlash sequence state"),
                    1,
                    position,
                    &self.stream,
                )?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &self.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(muse_glimmer_cache_error)?;
            return Err(error);
        }
        cache
            .commit_append(
                reservation,
                1,
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(muse_glimmer_cache_error)?;
        state.position = position + 1;
        Ok(())
    }

    /// Advances one token and leaves its vocabulary logits resident on the device.
    pub fn forward_one(
        &self,
        sequence: &mut MuseGlimmerSequence,
        token: u32,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<()> {
        self.forward_hidden(sequence, token, cache)?;
        let state = &mut sequence.state;
        let last = &state
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Muse Glimmer model",
                detail: "model has no decoder layers".to_string(),
            })?
            .output;
        self.final_norm.run_into(
            1,
            self.config.hidden_size,
            last,
            &mut state.final_hidden,
            &self.stream,
        )?;
        self.lm_head.run_into(
            &self.lt,
            &self.linear_plans,
            &state.final_hidden,
            &mut state.lm_head,
            &self.stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(state.lm_head.output.inout(), &self.stream)
    }

    fn transform_logit(&self, logit: f32) -> f32 {
        let cap = self.config.final_logit_softcapping;
        (logit * self.config.output_multiplier / cap).tanh() * cap
    }

    /// Returns private snapshot storage required for an aligned shared prefix.
    pub fn snapshot_sequence_device_bytes(
        &self,
        state: &MuseGlimmerDecodeState,
        prefix_tokens: usize,
    ) -> Result<usize> {
        if prefix_tokens == 0
            || !prefix_tokens.is_multiple_of(128)
            || prefix_tokens > state.position
        {
            return Err(Error::Shape {
                label: "Muse Glimmer sequence snapshot byte estimate",
                expected: format!(
                    "a nonzero 128-token-aligned prefix at most {} tokens",
                    state.position
                ),
                actual: prefix_tokens.to_string(),
            });
        }
        Ok(match (&self.dflash, &state.dflash_state) {
            (Some(model), Some(state)) => {
                model.checkpoint_sequence_device_bytes(state, prefix_tokens)?
            }
            (None, None) => 0,
            _ => {
                return Err(Error::Shape {
                    label: "Muse Glimmer DFlash checkpoint state",
                    expected: "model and sequence DFlash state to match".to_string(),
                    actual: format!(
                        "model={} sequence={}",
                        self.dflash.is_some(),
                        state.dflash_state.is_some()
                    ),
                });
            }
        })
    }

    /// Captures private DFlash state for an aligned shared KV-page prefix.
    pub fn snapshot_sequence(
        &self,
        state: &MuseGlimmerDecodeState,
        prefix_tokens: usize,
    ) -> Result<MuseGlimmerSequenceSnapshot> {
        self.snapshot_sequence_device_bytes(state, prefix_tokens)?;
        let dflash = match (&self.dflash, &state.dflash_state) {
            (Some(model), Some(state)) => {
                Some(model.checkpoint_sequence(state, prefix_tokens, &self.stream)?)
            }
            (None, None) => None,
            _ => {
                return Err(Error::Shape {
                    label: "Muse Glimmer DFlash checkpoint state",
                    expected: "model and sequence DFlash state to match".to_string(),
                    actual: format!(
                        "model={} sequence={}",
                        self.dflash.is_some(),
                        state.dflash_state.is_some()
                    ),
                });
            }
        };
        self.stream.synchronize()?;
        Ok(MuseGlimmerSequenceSnapshot {
            model_id: self.model_id,
            position: prefix_tokens,
            dflash,
        })
    }

    /// Restores private state associated with shared prefix pages.
    pub fn restore_sequence_snapshot(
        &self,
        snapshot: &MuseGlimmerSequenceSnapshot,
        state: &mut MuseGlimmerDecodeState,
        position: usize,
    ) -> Result<()> {
        if snapshot.model_id != self.model_id
            || snapshot.position != position
            || position > state.max_tokens
            || snapshot.dflash.is_some() != self.dflash.is_some()
        {
            return Err(Error::Shape {
                label: "Muse Glimmer sequence snapshot restore",
                expected: format!(
                    "matching model, position {position}, sufficient capacity, and matching DFlash state"
                ),
                actual: format!(
                    "snapshot_position={} capacity={} dflash={}",
                    snapshot.position,
                    state.max_tokens,
                    snapshot.dflash.is_some()
                ),
            });
        }
        if let (Some(model), Some(dflash_checkpoint), Some(dflash_state)) =
            (&self.dflash, &snapshot.dflash, state.dflash_state.as_mut())
        {
            model.restore_sequence_checkpoint(
                dflash_checkpoint,
                dflash_state,
                position,
                &self.stream,
            )?;
        }
        self.stream.synchronize()?;
        state.position = position;
        Ok(())
    }
}

impl MuseGlimmerDecodeState {
    pub fn len(&self) -> usize {
        self.position
    }

    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn device_bytes(&self) -> usize {
        self.hidden.device_bytes()
            + self.embedding_output.device_bytes()
            + self
                .layers
                .iter()
                .map(|layer| {
                    layer.attention.q.device_bytes()
                        + layer.attention.k.device_bytes()
                        + layer.attention.v.device_bytes()
                        + layer.attention.gate.device_bytes()
                        + layer.attention.q_normed.device_bytes()
                        + layer.attention.k_normed.device_bytes()
                        + layer.attention.q_positioned.device_bytes()
                        + layer.attention.k_positioned.device_bytes()
                        + layer.attention.attended.device_bytes()
                        + layer.attention.gated.device_bytes()
                        + layer.attention.output.device_bytes()
                        + layer.mlp.gate.device_bytes()
                        + layer.mlp.up.device_bytes()
                        + layer.mlp.activated.device_bytes()
                        + layer.mlp.down.device_bytes()
                        + layer.normalized.device_bytes()
                        + layer.residual.device_bytes()
                        + layer.feedforward_input.device_bytes()
                        + layer.feedforward_output.device_bytes()
                        + layer.output.device_bytes()
                })
                .sum::<usize>()
            + self.compact_attention.device_bytes()
            + self.final_hidden.device_bytes()
            + self.lm_head.device_bytes()
            + self.next_index.device_bytes()
            + self.next_value.device_bytes()
            + self
                .verification
                .as_ref()
                .map_or(0, |workspace| workspace.device_bytes())
            + self
                .dflash_state
                .as_ref()
                .map_or(0, |state| state.device_bytes())
    }
}

impl MuseGlimmerSequenceSnapshot {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn device_bytes(&self) -> usize {
        self.dflash
            .as_ref()
            .map_or(0, dflash::DFlashSequenceCheckpoint::device_bytes)
    }
}

impl RetainedSnapshot for MuseGlimmerSequenceSnapshot {
    fn retained_bytes(&self) -> usize {
        self.device_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json() -> String {
        serde_json::json!({
            "model_type": "muse_glimmer",
            "text_config": {
                "hidden_size": 6656,
                "intermediate_size": 19968,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "sliding_window": 2048,
                "max_position_embeddings": 131072,
                "vocab_size": 202048,
                "rms_norm_eps": 1e-5,
                "post_norm_eps": 1e-8,
                "qk_scale_factor": 3.87,
                "output_multiplier": 0.19611613513818404,
                "final_logit_softcapping": 20.0,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0]
            }
        })
        .to_string()
    }

    #[test]
    fn validates_released_text_layout() {
        let config = MuseGlimmerConfig::from_json(&config_json()).expect("config");
        assert_eq!(config.hidden_size, 6656);
        assert_eq!(config.num_key_value_heads, 2);
        assert!(config.is_local_layer(2).expect("local"));
        assert!(!config.is_local_layer(3).expect("global"));
    }

    #[test]
    fn rejects_rope_on_global_layer() {
        let text = config_json().replace(
            "500000.0,500000.0,500000.0,0.0",
            "500000.0,500000.0,500000.0,500000.0",
        );
        assert!(MuseGlimmerConfig::from_json(&text).is_err());
    }

    #[test]
    #[ignore = "requires the local Muse Glimmer NVFP4 checkpoint and an SM121 GPU"]
    fn local_checkpoint_greedy_continuation_is_stable() {
        use crate::muse_glimmer::{MuseGlimmerSequence, new_muse_glimmer_sequence_cache};

        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/muse-glimmer-30b-nvfp4");
        let model = MuseGlimmerModel::load(model_dir).expect("load local Muse Glimmer model");
        let mut cache = new_muse_glimmer_sequence_cache(&model, 1, 16).expect("sequence cache");
        let mut sequence =
            MuseGlimmerSequence::admit(&model, &mut cache, 16).expect("sequence admission");
        model
            .consume_one(&mut sequence, 200_000, &mut cache)
            .expect("consume BOS");
        let mut token = 19_873;
        let mut generated = Vec::new();
        for _ in 0..8 {
            let next = model
                .decode_one(&mut sequence, token, &mut cache)
                .expect("greedy decode");
            token = next.token;
            generated.push(token);
        }
        assert_eq!(generated, [24, 372, 1_045, 10_016, 328, 2_885, 262, 5_091]);
    }

    #[test]
    #[ignore = "requires the local Muse Glimmer NVFP4 and DFlash checkpoints and an SM121 GPU"]
    fn dflash_checkpoint_restore_preserves_the_next_speculative_cycle() {
        use crate::muse_glimmer::{
            MuseGlimmerSequence, new_muse_glimmer_sequence_cache_with_budget,
        };
        use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageTable};
        use seqcache::{AdmissionOutcome, AdmissionRequest};

        let model_dir = std::env::var_os("MUSE_GLIMMER_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/muse-glimmer-30b-nvfp4")
            });
        let dflash = std::env::var_os("MUSE_GLIMMER_DFLASH")
            .map(std::path::PathBuf::from)
            .expect("set MUSE_GLIMMER_DFLASH to Meta's dflash-kquant.gguf");
        let model =
            MuseGlimmerModel::load_with_dflash(model_dir, dflash).expect("load Muse with DFlash");
        let prefix = (0..128).map(|token| token as u32 + 1).collect::<Vec<_>>();
        let suffix = [129, 130, 131];
        let mut cache = new_muse_glimmer_sequence_cache_with_budget(
            &model,
            2,
            160,
            Some(4 * 1024 * 1024 * 1024),
        )
        .expect("sequence cache");
        let mut direct =
            MuseGlimmerSequence::admit(&model, &mut cache, 160).expect("direct sequence");
        for chunk in prefix.chunks(16) {
            model
                .dflash_prefill_chunk(&mut direct, chunk, false, &mut cache)
                .expect("prefill checkpoint prefix");
        }
        let estimated = model
            .snapshot_sequence_device_bytes(&direct.state, prefix.len())
            .expect("estimate snapshot bytes");
        let snapshot = model
            .snapshot_sequence(&direct.state, prefix.len())
            .expect("snapshot DFlash state");
        assert_eq!(snapshot.device_bytes(), estimated);
        cache
            .retain_prefix(
                direct.cache_id,
                &prefix,
                snapshot,
                &mut Sm12xCacheContext {
                    stream: model.stream(),
                    page_table: &mut direct.page_table,
                },
            )
            .expect("retain shared target pages");
        let matched = cache
            .lookup_prefix(&[prefix.as_slice(), &[999]].concat())
            .expect("lookup retained prefix");
        let mut restored_state = model.new_sequence_state(160).expect("restored state");
        let mut restored_table = Sm12xPageTable::new(160).expect("restored table");
        let outcome = cache
            .admit(
                Some(matched),
                AdmissionRequest {
                    max_position: 160,
                    private_state_bytes: restored_state.device_bytes(),
                    page_table_bytes: restored_table.managed_bytes(),
                    allow_emergency: false,
                },
                &mut Sm12xCacheContext {
                    stream: model.stream(),
                    page_table: &mut restored_table,
                },
                |snapshot, position| {
                    model.restore_sequence_snapshot(
                        snapshot.expect("retained snapshot"),
                        &mut restored_state,
                        position,
                    )
                },
            )
            .expect("restore admission");
        let AdmissionOutcome::Admitted(restored_id) = outcome else {
            panic!("restored admission would block");
        };
        let mut restored =
            MuseGlimmerSequence::from_admission(restored_id, restored_table, restored_state);
        assert_eq!(restored.position(), prefix.len());

        model
            .dflash_prefill_chunk(&mut direct, &suffix, true, &mut cache)
            .expect("continue direct state");
        model
            .dflash_prefill_chunk(&mut restored, &suffix, true, &mut cache)
            .expect("continue restored state");
        let direct_anchor = model
            .argmax_with_logit(&mut direct)
            .expect("direct anchor")
            .0;
        let restored_anchor = model
            .argmax_with_logit(&mut restored)
            .expect("restored anchor")
            .0;
        assert_eq!(restored_anchor, direct_anchor);
        let direct_cycle = model
            .dflash_cycle(&mut direct, direct_anchor, &mut cache)
            .expect("direct DFlash cycle");
        let restored_cycle = model
            .dflash_cycle(&mut restored, restored_anchor, &mut cache)
            .expect("restored DFlash cycle");
        assert_eq!(restored_cycle, direct_cycle);
    }
}
