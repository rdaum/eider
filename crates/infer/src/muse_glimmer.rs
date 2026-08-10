//! Muse Glimmer dense text-model loading and inference.
//!
//! The initial runtime is text-only. It consumes the released ModelOpt NVFP4
//! language projections while retaining the checkpoint's attention gates,
//! embeddings, normalization vectors, and language head in BF16.

use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptCublasLtWeight,
    ModelOptNvfp4Linear, Result, Sm12xKvAttentionWorkspace, Sm12xKvCache, add_f32_into_on_stream,
    argmax_f32_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, nvfp4_w4a16_matrix_matvec_f32_batch_into_on_stream,
    rms_norm_f32_into_on_stream, rope_neox_f32_into_on_stream,
    round_f32_to_bf16_in_place_on_stream, sigmoid_mul_f32_into_on_stream,
    silu_mul_f32_into_on_stream,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

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

    fn load_bf16_linear(&self, prefix: &str) -> Result<MuseBf16Linear> {
        MuseBf16Linear::load(&self.checkpoint, prefix)
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
            .map(|bytes| nvfp4::format::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
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
    weight_scale: DeviceBuffer<u8>,
    out_features: usize,
    in_features: usize,
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
        let weight_scale = DeviceBuffer::from_host(&weight.weight_scale)?;
        let out_features = weight.out_features;
        let in_features = weight.in_features;
        Ok(Self {
            name: name.to_string(),
            weight: weight.as_cublaslt_weight()?,
            weight_scale,
            out_features,
            in_features,
        })
    }

    fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err(Error::Shape {
                label: "Muse Glimmer NVFP4 linear buffers",
                expected: format!("input={} output={}", self.in_features, self.out_features),
                actual: format!("input={} output={}", input.len(), output.len()),
            });
        }
        nvfp4_w4a16_matrix_matvec_f32_batch_into_on_stream(
            input,
            self.weight.matrix(),
            &self.weight_scale,
            output.output(),
            1,
            self.out_features,
            self.in_features,
            self.weight.weight_scale_2(),
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
        self.weight.device_bytes() + self.weight_scale.device_bytes()
    }
}

struct MuseBf16Linear {
    name: String,
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl MuseBf16Linear {
    fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let info = checkpoint.tensor_info(&name)?;
        if info.dtype != "BF16" || info.shape.len() != 2 {
            return Err(Error::Shape {
                label: "Muse Glimmer BF16 linear",
                expected: "BF16 [rows, cols]".to_string(),
                actual: format!("{} {:?} for {name}", info.dtype, info.shape),
            });
        }
        let rows = info.shape[0];
        let cols = info.shape[1];
        let bytes = checkpoint
            .open_shard_for_tensor(&name)?
            .read_tensor_bytes(&name)?;
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        Ok(Self {
            name: prefix.to_string(),
            weight: DeviceBuffer::from_host(&values)?,
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
                label: "Muse Glimmer BF16 linear buffers",
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
            label: "Muse Glimmer BF16 linear execution",
            detail: format!("{} [{}, {}]: {error}", self.name, self.rows, self.cols),
        })?;
        round_f32_to_bf16_in_place_on_stream(output.inout(), stream)
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
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
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
            gate: DeviceBuffer::zeroed(self.intermediate_size)?,
            up: DeviceBuffer::zeroed(self.intermediate_size)?,
            activated: DeviceBuffer::zeroed(self.intermediate_size)?,
            output: DeviceBuffer::zeroed(self.down.shape().0)?,
        })
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseMlpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.gate.run_into(input, &mut workspace.gate, stream)?;
        self.up.run_into(input, &mut workspace.up, stream)?;
        silu_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.up,
            workspace.activated.output(),
            stream,
        )?;
        self.down
            .run_into(&workspace.activated, &mut workspace.output, stream)
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

struct MuseAttention {
    q: MuseNvfp4Linear,
    k: MuseNvfp4Linear,
    v: MuseNvfp4Linear,
    gate: MuseBf16Linear,
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
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_positioned: DeviceBuffer<f32>,
    k_positioned: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl MuseAttention {
    fn load(checkpoint: &MuseGlimmerCheckpoint, layer: usize) -> Result<Self> {
        let config = &checkpoint.config;
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let q = checkpoint.load_nvfp4_linear(&format!("{prefix}.q_proj"))?;
        let k = checkpoint.load_nvfp4_linear(&format!("{prefix}.k_proj"))?;
        let v = checkpoint.load_nvfp4_linear(&format!("{prefix}.v_proj"))?;
        let gate = checkpoint.load_bf16_linear(&format!("{prefix}.gate_proj"))?;
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
        gate.require_shape(q_width, config.hidden_size, "Muse Glimmer attention gate")?;
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
            q: DeviceBuffer::zeroed(q_width)?,
            k: DeviceBuffer::zeroed(kv_width)?,
            v: DeviceBuffer::zeroed(kv_width)?,
            gate: DeviceBuffer::zeroed(q_width)?,
            q_normed: DeviceBuffer::zeroed(q_width)?,
            k_normed: DeviceBuffer::zeroed(kv_width)?,
            q_positioned: DeviceBuffer::zeroed(q_width)?,
            k_positioned: DeviceBuffer::zeroed(kv_width)?,
            attended: DeviceBuffer::zeroed(q_width)?,
            gated: DeviceBuffer::zeroed(q_width)?,
            output: DeviceBuffer::zeroed(self.output.shape().0)?,
        })
    }

    fn new_kv_cache(&self, max_tokens: usize) -> Result<Sm12xKvCache> {
        Sm12xKvCache::new(max_tokens, self.kv_heads, self.head_dim)
    }

    fn new_compact_attention_workspace(
        &self,
        max_tokens: usize,
    ) -> Result<Sm12xKvAttentionWorkspace> {
        Sm12xKvAttentionWorkspace::new_gqa(max_tokens, self.q_heads, self.kv_heads, self.head_dim)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseAttentionWorkspace,
        cache: &mut Sm12xKvCache,
        compact_attention: &mut Sm12xKvAttentionWorkspace,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len() != position {
            return Err(Error::Shape {
                label: "Muse Glimmer attention cache position",
                expected: position.to_string(),
                actual: cache.len().to_string(),
            });
        }
        self.q.run_into(input, &mut workspace.q, stream)?;
        self.k.run_into(input, &mut workspace.k, stream)?;
        self.v.run_into(input, &mut workspace.v, stream)?;
        self.gate.run_into(input, &mut workspace.gate, stream)?;
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
        cache.append_at_on_stream(&workspace.k_positioned, &workspace.v, position, stream)?;
        if let Some(window) = self.window {
            compact_attention.attention_window_into_on_stream(
                cache,
                &workspace.q_positioned,
                workspace.attended.output(),
                cache.len().saturating_sub(window),
                stream,
            )?;
        } else {
            compact_attention.attention_into_on_stream(
                cache,
                &workspace.q_positioned,
                workspace.attended.output(),
                stream,
            )?;
        }
        sigmoid_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.attended,
            workspace.gated.output(),
            stream,
        )?;
        self.output
            .run_into(&workspace.gated, &mut workspace.output, stream)
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
        input: &DeviceBuffer<f32>,
        workspace: &mut MuseDecoderLayerWorkspace,
        cache: &mut Sm12xKvCache,
        compact_attention: &mut Sm12xKvAttentionWorkspace,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden = self.attention.q.shape().1;
        self.input_norm
            .run_into(1, hidden, input, &mut workspace.normalized, stream)?;
        self.attention.run_into(
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
            &workspace.attention.output,
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
        self.mlp
            .run_into(&workspace.feedforward_input, &mut workspace.mlp, stream)?;
        self.post_feedforward_norm.run_into(
            1,
            hidden,
            &workspace.mlp.output,
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

/// Complete resident Muse Glimmer text model.
pub struct MuseGlimmerModel {
    model_id: u64,
    config: MuseGlimmerConfig,
    embedding: DeviceBuffer<u16>,
    embedding_norm: MuseRmsNorm,
    layers: Vec<MuseDecoderLayer>,
    final_norm: MuseRmsNorm,
    lm_head: MuseBf16Linear,
    stream: CudaStream,
}

/// Mutable execution and compact K/V state for one text sequence.
pub struct MuseGlimmerDecodeState {
    model_id: u64,
    hidden: DeviceBuffer<f32>,
    embedding_output: DeviceBuffer<f32>,
    layers: Vec<MuseDecoderLayerWorkspace>,
    kv_caches: Vec<Sm12xKvCache>,
    compact_attention: MuseCompactAttentionWorkspaces,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_index: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    position: usize,
    max_tokens: usize,
}

/// Immutable aligned Muse Glimmer prompt-prefix state.
pub struct MuseGlimmerSequenceCheckpoint {
    model_id: u64,
    position: usize,
    kv_caches: Vec<Sm12xKvCache>,
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
        let lm_head = checkpoint.load_bf16_linear("lm_head")?;
        lm_head.require_shape(
            config.vocab_size,
            config.hidden_size,
            "Muse Glimmer language head",
        )?;
        Ok(Self {
            model_id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            embedding,
            embedding_norm,
            layers,
            final_norm,
            lm_head,
            stream: CudaStream::new_non_blocking()?,
        })
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
            + self.final_norm.device_bytes()
            + self.lm_head.device_bytes()
    }

    /// Waits for work submitted by this model instance.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Allocates execution scratch and compact K/V storage for one sequence.
    pub fn new_decode_state(&self, max_tokens: usize) -> Result<MuseGlimmerDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_position_embeddings {
            return Err(Error::Shape {
                label: "Muse Glimmer decode capacity",
                expected: format!("1..={}", self.config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        Ok(MuseGlimmerDecodeState {
            model_id: self.model_id,
            hidden: DeviceBuffer::zeroed(self.config.hidden_size)?,
            embedding_output: DeviceBuffer::zeroed(self.config.hidden_size)?,
            layers: self
                .layers
                .iter()
                .map(MuseDecoderLayer::new_workspace)
                .collect::<Result<Vec<_>>>()?,
            kv_caches: self
                .layers
                .iter()
                .map(|layer| layer.attention.new_kv_cache(max_tokens))
                .collect::<Result<Vec<_>>>()?,
            compact_attention: MuseCompactAttentionWorkspaces::new(&self.layers, max_tokens)?,
            final_hidden: DeviceBuffer::zeroed(self.config.hidden_size)?,
            logits: DeviceBuffer::zeroed(self.config.vocab_size)?,
            next_index: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            position: 0,
            max_tokens,
        })
    }

    /// Advances one token without materializing vocabulary logits.
    pub fn consume_one(&self, state: &mut MuseGlimmerDecodeState, token: u32) -> Result<()> {
        self.forward_hidden(state, token)
    }

    /// Advances one token and copies transformed vocabulary logits to the host.
    pub fn logits_one(&self, state: &mut MuseGlimmerDecodeState, token: u32) -> Result<Vec<f32>> {
        self.forward_one(state, token)?;
        self.logits_to_host(state)
    }

    /// Advances one token and performs greedy selection.
    pub fn decode_one(
        &self,
        state: &mut MuseGlimmerDecodeState,
        token: u32,
    ) -> Result<MuseGlimmerNextToken> {
        self.forward_one(state, token)?;
        let (token, logit) = self.argmax_with_logit(state)?;
        Ok(MuseGlimmerNextToken { token, logit })
    }

    /// Copies the most recent transformed vocabulary logits to the host.
    pub fn logits_to_host(&self, state: &MuseGlimmerDecodeState) -> Result<Vec<f32>> {
        if state.position == 0 {
            return Err(Error::Format {
                label: "Muse Glimmer logits",
                detail: "no token has been evaluated".to_string(),
            });
        }
        let mut logits = state.logits.copy_to_host(&self.stream)?.into_vec();
        for logit in &mut logits {
            *logit = self.transform_logit(*logit);
        }
        Ok(logits)
    }

    /// Returns the greedy token and transformed logit without copying the vocabulary.
    pub fn argmax_with_logit(&self, state: &mut MuseGlimmerDecodeState) -> Result<(u32, f32)> {
        if state.position == 0 {
            return Err(Error::Format {
                label: "Muse Glimmer logits",
                detail: "no token has been evaluated".to_string(),
            });
        }
        argmax_f32_into_on_stream(
            &state.logits,
            state.next_index.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        let token = state.next_index.copy_to_host(&self.stream)?[0];
        let logit = self.transform_logit(state.next_value.copy_to_host(&self.stream)?[0]);
        Ok((token, logit))
    }

    fn forward_hidden(&self, state: &mut MuseGlimmerDecodeState, token: u32) -> Result<()> {
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
            self.layers[layer_index].run_into(
                input,
                &mut current[0],
                &mut state.kv_caches[layer_index],
                state.compact_attention.for_layer_mut(local)?,
                state.position,
                &self.stream,
            )?;
        }
        state.position += 1;
        Ok(())
    }

    /// Advances one token and leaves its vocabulary logits resident on the device.
    pub fn forward_one(&self, state: &mut MuseGlimmerDecodeState, token: u32) -> Result<()> {
        self.forward_hidden(state, token)?;
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
        self.lm_head
            .run_into(&state.final_hidden, &mut state.logits, &self.stream)
    }

    fn transform_logit(&self, logit: f32) -> f32 {
        let cap = self.config.final_logit_softcapping;
        (logit * self.config.output_multiplier / cap).tanh() * cap
    }

    /// Returns compact storage required for an aligned prefix checkpoint.
    pub fn checkpoint_sequence_device_bytes(
        &self,
        state: &MuseGlimmerDecodeState,
        prefix_tokens: usize,
    ) -> Result<usize> {
        if prefix_tokens == 0
            || !prefix_tokens.is_multiple_of(128)
            || prefix_tokens > state.position
        {
            return Err(Error::Shape {
                label: "Muse Glimmer sequence checkpoint byte estimate",
                expected: format!(
                    "a nonzero 128-token-aligned prefix at most {} tokens",
                    state.position
                ),
                actual: prefix_tokens.to_string(),
            });
        }
        state.kv_caches.iter().try_fold(0usize, |total, cache| {
            let bytes = cache.device_bytes_for_capacity(prefix_tokens)?;
            total.checked_add(bytes).ok_or_else(|| Error::Shape {
                label: "Muse Glimmer checkpoint byte estimate",
                expected: "device-byte total without overflow".to_string(),
                actual: prefix_tokens.to_string(),
            })
        })
    }

    /// Copies an aligned K/V prefix into immutable compact storage.
    pub fn checkpoint_sequence(
        &self,
        state: &MuseGlimmerDecodeState,
        prefix_tokens: usize,
    ) -> Result<MuseGlimmerSequenceCheckpoint> {
        self.checkpoint_sequence_device_bytes(state, prefix_tokens)?;
        let mut kv_caches = self
            .layers
            .iter()
            .map(|layer| layer.attention.new_kv_cache(prefix_tokens))
            .collect::<Result<Vec<_>>>()?;
        for (destination, source) in kv_caches.iter_mut().zip(&state.kv_caches) {
            destination.copy_aligned_prefix_from_on_stream(source, prefix_tokens, &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok(MuseGlimmerSequenceCheckpoint {
            model_id: self.model_id,
            position: prefix_tokens,
            kv_caches,
        })
    }

    /// Restores a compact prompt checkpoint into a new active state.
    pub fn restore_sequence_checkpoint(
        &self,
        checkpoint: &MuseGlimmerSequenceCheckpoint,
        max_tokens: usize,
    ) -> Result<MuseGlimmerDecodeState> {
        if checkpoint.model_id != self.model_id
            || checkpoint.position > max_tokens
            || checkpoint.kv_caches.len() != self.layers.len()
        {
            return Err(Error::Shape {
                label: "Muse Glimmer sequence checkpoint restore",
                expected: format!(
                    "matching model, capacity >= {}, and {} layer caches",
                    checkpoint.position,
                    self.layers.len()
                ),
                actual: format!(
                    "capacity={max_tokens} layer_caches={}",
                    checkpoint.kv_caches.len()
                ),
            });
        }
        let mut state = self.new_decode_state(max_tokens)?;
        for (destination, source) in state.kv_caches.iter_mut().zip(&checkpoint.kv_caches) {
            destination.copy_aligned_prefix_from_on_stream(
                source,
                checkpoint.position,
                &self.stream,
            )?;
        }
        self.stream.synchronize()?;
        state.position = checkpoint.position;
        Ok(state)
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
                        + layer.mlp.output.device_bytes()
                        + layer.normalized.device_bytes()
                        + layer.residual.device_bytes()
                        + layer.feedforward_input.device_bytes()
                        + layer.feedforward_output.device_bytes()
                        + layer.output.device_bytes()
                })
                .sum::<usize>()
            + self
                .kv_caches
                .iter()
                .map(Sm12xKvCache::device_bytes)
                .sum::<usize>()
            + self.compact_attention.device_bytes()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_index.device_bytes()
            + self.next_value.device_bytes()
    }
}

impl MuseGlimmerSequenceCheckpoint {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn device_bytes(&self) -> usize {
        self.kv_caches.iter().map(Sm12xKvCache::device_bytes).sum()
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
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/muse-glimmer-30b-nvfp4");
        let model = MuseGlimmerModel::load(model_dir).expect("load local Muse Glimmer model");
        let mut state = model.new_decode_state(16).expect("decode state");
        model.consume_one(&mut state, 200_000).expect("consume BOS");
        let mut token = 19_873;
        let mut generated = Vec::new();
        for _ in 0..8 {
            let next = model.decode_one(&mut state, token).expect("greedy decode");
            token = next.token;
            generated.push(token);
        }
        assert_eq!(generated, [24, 372, 1_045, 10_016, 328, 2_885, 262, 5_091]);
    }
}
