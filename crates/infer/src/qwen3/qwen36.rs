//! Qwen3.6 / Qwen3.5-MoE hybrid execution pieces.

use crate::nvfp4::{
    CublasLt, CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, Error, F32Matrix,
    Fp8TnMatmulPlan, GemmShape, GpuCounterCollector, GroupedGemvPointerTableBuffers,
    MarlinNvfp4GateUp, ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear,
    MoeSiluQuantizeSlotBuffers, MropeSections, Nvfp4Matrix, Result, SafeTensorInfo,
    Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmVector, Sm12xFp4GemmWeight, add_f32_into_on_stream,
    append_rows_f32_indexed_into_on_stream, append_rows_f32_into_on_stream,
    argmax_f32_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    cached_gqa_attention_f32_indexed_into_on_stream, cached_gqa_attention_f32_into_on_stream,
    copy_bf16_row_to_f32_indexed_into_on_stream, device_weight_gemv_on_stream,
    fill_f32_into_on_stream, fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream,
    fp8_linear_configured_f32_into_on_stream, fp8_linear_f32_into_on_stream,
    fp8_linear_w8a8_f32_into_on_stream, fp8_moe_grouped_down_f32_into_on_stream,
    fp8_moe_grouped_gate_up_f32_into_on_stream, gated_delta_net_128_f32_into_on_stream,
    gated_rms_norm_f32_into_on_stream, gather_nvfp4_grouped_gemv_ptr_tables_on_stream,
    indexed_grouped_gemv_on_stream, moe_silu_quantize_bf16_slots_on_stream,
    moe_silu_quantize_fp8_slots_f32_into_on_stream, moe_silu_quantize_slots_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream, nvfp4_w4a16_grouped_matvec_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_into_on_stream, nvfp4_w4a16_top1_f32_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_into_on_stream,
    quantize_nvfp4_vector_simple_scales_f32_into_on_stream, qwen36_ffn_finalize_f32_into_on_stream,
    qwen36_ffn_finalize_routed_f32_into_on_stream, qwen36_full_attn_prep_f32_into_on_stream,
    qwen36_gdn_gate_into_on_stream, qwen36_gdn_prep_into_on_stream, rms_norm_f32_into_on_stream,
    rope_imrope_f32_indexed_into_on_stream, rope_imrope_f32_into_on_stream,
    round_f32_to_bf16_in_place_on_stream, scale_channel_f32_device_scalar_in_place_on_stream,
    scaled_add_f32_into_on_stream, sigmoid_mul_f32_into_on_stream,
    sigmoid_scale_scalar_f32_into_on_stream, silu_mul_halves_f32_into_on_stream,
};

use super::infer::{
    GroupedGemvWorkspace, MoeExpertPointerTables, MoeGroupedDownWorkspace, MoeRouteWorkspace,
    QwenArchitecture, QwenDecodeProfile, QwenFfnConfig, QwenLayerKind, QwenLinearAttentionConfig,
    QwenModelManifest,
};
use super::qwen36_cache::{ensure_layer_cache, ensure_model_cache, prepared_layer_dir};

use std::rc::Rc;
use std::time::Instant;

/// Loader scaffold for the Qwen3.6/Qwen3.5-MoE hybrid text stack.
pub struct Qwen36Model {
    manifest: QwenModelManifest,
    checkpoint: ModelOptCheckpoint,
}

/// Device-ready weights for one Qwen3.6 text layer.
pub enum Qwen36LayerWeights {
    /// Gated Delta Net recurrent layer.
    LinearAttention(Qwen36LinearAttentionWeights),
    /// Standard full-attention layer.
    FullAttention(Qwen36FullAttentionWeights),
}

/// Device-ready Qwen3.6 full-attention weights.
pub struct Qwen36FullAttentionWeights {
    q: Fp8Linear,
    k: Fp8Linear,
    v: Fp8Linear,
    o: Fp8Linear,
    q_norm_weight: DeviceBuffer<f32>,
    k_norm_weight: DeviceBuffer<f32>,
}

/// Mutable one-token decode workspace for a Qwen3.6 full-attention layer.
pub struct Qwen36FullAttentionWorkspace {
    fp8_dynamic_input: DeviceBuffer<u8>,
    fp8_dynamic_input_scale: DeviceBuffer<f32>,
    pub q_proj_output: DeviceBuffer<f32>,
    pub q_normed: DeviceBuffer<f32>,
    pub gate: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub k_normed: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub q_rope: DeviceBuffer<f32>,
    pub k_rope: DeviceBuffer<f32>,
    pub key_cache: DeviceBuffer<f32>,
    pub value_cache: DeviceBuffer<f32>,
    pub attn: DeviceBuffer<f32>,
    pub gated_attn: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f32>,
    cache_capacity: usize,
}

/// Borrowed outputs from one full-attention step.
pub struct Qwen36FullAttentionStep<'a> {
    /// Raw Q projection containing `[query, gate]`.
    pub q_proj_output: &'a DeviceBuffer<f32>,
    /// RoPE'd query used for attention.
    pub q_rope: &'a DeviceBuffer<f32>,
    /// Attention output before sigmoid gate.
    pub attn: &'a DeviceBuffer<f32>,
    /// Attention output after sigmoid gate.
    pub gated_attn: &'a DeviceBuffer<f32>,
    /// Final layer output after output projection.
    pub output: &'a DeviceBuffer<f32>,
}

/// Device-ready Qwen3.6 Gated Delta Net layer weights.
pub struct Qwen36LinearAttentionWeights {
    fp8: Rc<Qwen36LinearFp8Execution>,
    qkv: Fp8Linear,
    z: Fp8Linear,
    alpha: Bf16Linear,
    beta: Bf16Linear,
    conv_weight: DeviceBuffer<u16>,
    a_log: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
    norm_weight: DeviceBuffer<f32>,
    out: Fp8Linear,
}

/// Mutable one-token decode workspace for a Qwen3.6 Gated Delta Net layer.
pub struct Qwen36LinearAttentionWorkspace {
    linear: QwenLinearAttentionConfig,
    fp8_dynamic_input: DeviceBuffer<u8>,
    fp8_dynamic_input_scale: DeviceBuffer<f32>,
    fp8_value_input: DeviceBuffer<u8>,
    fp8_value_input_scale: DeviceBuffer<f32>,
    pub qkv_output: DeviceBuffer<f32>,
    pub z_output: DeviceBuffer<f32>,
    pub alpha: DeviceBuffer<f32>,
    beta_input: DeviceBuffer<f32>,
    pub gate: DeviceBuffer<f32>,
    pub beta: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    /// Conv recurrent state, laid out as `[conv_channel][kernel - 1]`.
    pub conv_state: DeviceBuffer<f32>,
    /// GDN recurrent state, laid out as `[value_head][col][row]`.
    pub recurrent_state: DeviceBuffer<f32>,
    pub gdn_output: DeviceBuffer<f32>,
    pub normed: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

/// Borrowed outputs from one linear-attention step.
pub struct Qwen36LinearAttentionStep<'a> {
    /// Raw pre-conv QKV projection.
    pub qkv_output: &'a DeviceBuffer<f32>,
    /// Raw Z projection.
    pub z_output: &'a DeviceBuffer<f32>,
    /// Gated Delta Net output before gated RMSNorm.
    pub gdn_output: &'a DeviceBuffer<f32>,
    /// Final layer output after output projection.
    pub output: &'a DeviceBuffer<f32>,
}

struct Fp8Linear {
    weight: DeviceBuffer<u8>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    channel_weight_scale: Option<DeviceBuffer<f32>>,
    input_scale: Option<f32>,
}

struct Qwen36LinearFp8Plans {
    qkv: Fp8TnMatmulPlan,
    z: Fp8TnMatmulPlan,
    out: Fp8TnMatmulPlan,
}

struct Qwen36LinearFp8Execution {
    lt: CublasLt,
    plans: Option<Qwen36LinearFp8Plans>,
}

impl Qwen36LinearFp8Execution {
    fn new(checkpoint: &ModelOptCheckpoint, manifest: &QwenModelManifest) -> Result<Self> {
        let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
            label: "Qwen3.6 linear FP8 execution",
            detail: "manifest has no linear-attention config".to_string(),
        })?;
        let first_linear_layer = manifest
            .layer_kinds
            .iter()
            .position(|kind| *kind == QwenLayerKind::LinearAttention)
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 linear FP8 execution",
                detail: "model has no linear-attention layer".to_string(),
            })?;
        let prefix = format!(
            "{}.layers.{first_linear_layer}.linear_attn.in_proj_qkv",
            manifest.tensor_prefix
        );
        let lt = CublasLt::new()?;
        let plans = if checkpoint.contains_tensor(&format!("{prefix}.input_scale")) {
            None
        } else {
            let key_dim = linear.key_heads * linear.key_head_dim;
            let value_dim = linear.value_heads * linear.value_head_dim;
            let qkv_dim = key_dim * 2 + value_dim;
            const WORKSPACE_LIMIT: u64 = 8 << 20;
            Some(Qwen36LinearFp8Plans {
                qkv: Fp8TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(qkv_dim, 1, manifest.hidden),
                    WORKSPACE_LIMIT,
                )?,
                z: Fp8TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(value_dim, 1, manifest.hidden),
                    WORKSPACE_LIMIT,
                )?,
                out: Fp8TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(manifest.hidden, 1, value_dim),
                    WORKSPACE_LIMIT,
                )?,
            })
        };
        Ok(Self { lt, plans })
    }
}

struct Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Qwen36Model {
    /// Opens a Qwen3.6/Qwen3.5-MoE checkpoint and validates its hybrid schedule.
    pub fn open(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let manifest = QwenModelManifest::load(model_dir.as_ref())?;
        if manifest.architecture != QwenArchitecture::Qwen35Moe {
            return Err(Error::Format {
                label: "Qwen3.6 model",
                detail: format!(
                    "expected qwen3_5_moe architecture, got {:?}",
                    manifest.architecture
                ),
            });
        }
        if manifest.layer_kinds.len() != manifest.layers {
            return Err(Error::Shape {
                label: "Qwen3.6 layer schedule",
                expected: format!("{} layer entries", manifest.layers),
                actual: format!("{} layer entries", manifest.layer_kinds.len()),
            });
        }
        if manifest.linear_attention.is_none()
            || !manifest
                .layer_kinds
                .contains(&QwenLayerKind::LinearAttention)
        {
            return Err(Error::Format {
                label: "Qwen3.6 model",
                detail: "missing linear-attention schedule/config".to_string(),
            });
        }
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        Ok(Self {
            manifest,
            checkpoint,
        })
    }

    /// Returns the parsed model manifest.
    pub fn manifest(&self) -> &QwenModelManifest {
        &self.manifest
    }

    /// Returns the underlying ModelOpt checkpoint handle.
    pub fn checkpoint(&self) -> &ModelOptCheckpoint {
        &self.checkpoint
    }

    /// Returns the layer kind for `layer`.
    pub fn layer_kind(&self, layer: usize) -> Result<QwenLayerKind> {
        self.manifest
            .layer_kinds
            .get(layer)
            .copied()
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 layer index",
                expected: format!("layer < {}", self.manifest.layers),
                actual: layer.to_string(),
            })
    }

    /// Loads one layer according to the hybrid layer schedule.
    pub fn load_layer(&self, layer: usize) -> Result<Qwen36LayerWeights> {
        match self.layer_kind(layer)? {
            QwenLayerKind::LinearAttention => Ok(Qwen36LayerWeights::LinearAttention(
                Qwen36LinearAttentionWeights::load(&self.checkpoint, &self.manifest, layer)?,
            )),
            QwenLayerKind::FullAttention => Ok(Qwen36LayerWeights::FullAttention(
                Qwen36FullAttentionWeights::load(&self.checkpoint, &self.manifest, layer)?,
            )),
        }
    }

    /// Loads the first linear-attention layer in the schedule.
    pub fn load_first_linear_attention_layer(
        &self,
    ) -> Result<(usize, Qwen36LinearAttentionWeights)> {
        let layer = self
            .manifest
            .layer_kinds
            .iter()
            .position(|kind| *kind == QwenLayerKind::LinearAttention)
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 layer schedule",
                detail: "no linear-attention layers".to_string(),
            })?;
        match self.load_layer(layer)? {
            Qwen36LayerWeights::LinearAttention(weights) => Ok((layer, weights)),
            Qwen36LayerWeights::FullAttention(_) => unreachable!(),
        }
    }

    /// Loads the first full-attention layer in the schedule.
    pub fn load_first_full_attention_layer(&self) -> Result<(usize, Qwen36FullAttentionWeights)> {
        let layer = self
            .manifest
            .layer_kinds
            .iter()
            .position(|kind| *kind == QwenLayerKind::FullAttention)
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 layer schedule",
                detail: "no full-attention layers".to_string(),
            })?;
        match self.load_layer(layer)? {
            Qwen36LayerWeights::FullAttention(weights) => Ok((layer, weights)),
            Qwen36LayerWeights::LinearAttention(_) => unreachable!(),
        }
    }

    /// Allocates workspace for a loaded linear-attention layer.
    pub fn linear_attention_workspace(
        &self,
        weights: &Qwen36LinearAttentionWeights,
    ) -> Result<Qwen36LinearAttentionWorkspace> {
        let linear = self
            .manifest
            .linear_attention
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 linear attention",
                detail: "manifest has no linear-attention config".to_string(),
            })?;
        Qwen36LinearAttentionWorkspace::new(&self.manifest, linear, weights)
    }

    /// Allocates workspace for a loaded full-attention layer.
    pub fn full_attention_workspace(
        &self,
        weights: &Qwen36FullAttentionWeights,
        cache_capacity: usize,
    ) -> Result<Qwen36FullAttentionWorkspace> {
        Qwen36FullAttentionWorkspace::new(&self.manifest, weights, cache_capacity)
    }

    /// Loads the MoE + shared-expert FFN for `layer`.
    pub fn load_moe(&self, layer: usize) -> Result<Qwen36MoeWeights> {
        if layer >= self.manifest.layers {
            return Err(Error::Shape {
                label: "Qwen3.6 MoE layer index",
                expected: format!("layer < {}", self.manifest.layers),
                actual: layer.to_string(),
            });
        }
        Qwen36MoeWeights::load(&self.checkpoint, &self.manifest, layer, false)
    }

    fn load_moe_from_prepared_cache(&self, layer: usize) -> Result<Qwen36MoeWeights> {
        Qwen36MoeWeights::load(&self.checkpoint, &self.manifest, layer, true)
    }

    /// Allocates workspace for a loaded MoE + shared-expert FFN.
    pub fn moe_workspace(&self) -> Result<Qwen36MoeWorkspace> {
        Qwen36MoeWorkspace::new(&self.manifest)
    }

    /// Loads the input RMSNorm weight for `layer` (`input_layernorm.weight`).
    pub fn load_input_norm(&self, layer: usize) -> Result<DeviceBuffer<f32>> {
        let name = format!(
            "{}.layers.{layer}.input_layernorm.weight",
            self.manifest.tensor_prefix
        );
        read_bf16_vector_delta_as_f32_device(&self.checkpoint, &name, self.manifest.hidden)
    }

    /// Loads the post-attention RMSNorm weight for `layer`
    /// (`post_attention_layernorm.weight`).
    pub fn load_post_attn_norm(&self, layer: usize) -> Result<DeviceBuffer<f32>> {
        let name = format!(
            "{}.layers.{layer}.post_attention_layernorm.weight",
            self.manifest.tensor_prefix
        );
        read_bf16_vector_delta_as_f32_device(&self.checkpoint, &name, self.manifest.hidden)
    }
}

impl Qwen36FullAttentionWeights {
    /// Loads a full-attention layer by layer index from the Qwen3.6 text stack.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("{}.layers.{layer}.self_attn", manifest.tensor_prefix);
        let q = Fp8Linear::from_host(&checkpoint.load_fp8_linear(&format!("{prefix}.q_proj"))?)?;
        let k = Fp8Linear::from_host(&checkpoint.load_fp8_linear(&format!("{prefix}.k_proj"))?)?;
        let v = Fp8Linear::from_host(&checkpoint.load_fp8_linear(&format!("{prefix}.v_proj"))?)?;
        let o = Fp8Linear::from_host(&checkpoint.load_fp8_linear(&format!("{prefix}.o_proj"))?)?;

        let expected_q_rows = manifest
            .q_heads
            .checked_mul(manifest.head_dim)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 full-attention q_proj",
                expected: "2 * q_heads * head_dim without overflow".to_string(),
                actual: format!(
                    "q_heads={} head_dim={}",
                    manifest.q_heads, manifest.head_dim
                ),
            })?;
        let expected_kv_rows = manifest
            .kv_heads
            .checked_mul(manifest.head_dim)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 full-attention kv_proj",
                expected: "kv_heads * head_dim without overflow".to_string(),
                actual: format!(
                    "kv_heads={} head_dim={}",
                    manifest.kv_heads, manifest.head_dim
                ),
            })?;
        q.require_shape(expected_q_rows, manifest.hidden, "Qwen3.6 q_proj")?;
        k.require_shape(expected_kv_rows, manifest.hidden, "Qwen3.6 k_proj")?;
        v.require_shape(expected_kv_rows, manifest.hidden, "Qwen3.6 v_proj")?;
        o.require_shape(
            manifest.hidden,
            manifest.q_heads * manifest.head_dim,
            "Qwen3.6 o_proj",
        )?;

        Ok(Self {
            q,
            k,
            v,
            o,
            q_norm_weight: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{prefix}.q_norm.weight"),
                manifest.head_dim,
            )?,
            k_norm_weight: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{prefix}.k_norm.weight"),
                manifest.head_dim,
            )?,
        })
    }

    /// Returns `(q_rows, k_rows, v_rows, o_rows)` for inspection/probes.
    pub fn projection_rows(&self) -> (usize, usize, usize, usize) {
        (self.q.rows, self.k.rows, self.v.rows, self.o.rows)
    }

    /// Returns `(q_norm_len, k_norm_len)`.
    pub fn norm_lens(&self) -> (usize, usize) {
        (self.q_norm_weight.len(), self.k_norm_weight.len())
    }

    /// Returns output width.
    pub fn output_width(&self) -> usize {
        self.o.rows
    }

    /// Runs one token through this full-attention layer.
    pub fn run_one_token<'a>(
        &'a self,
        workspace: &'a mut Qwen36FullAttentionWorkspace,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<Qwen36FullAttentionStep<'a>> {
        if position >= workspace.cache_capacity {
            return Err(Error::Shape {
                label: "Qwen3.6 full-attention cache",
                expected: format!("position < {}", workspace.cache_capacity),
                actual: position.to_string(),
            });
        }

        self.q.run_into(
            hidden,
            &mut workspace.q_proj_output,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        self.k.run_into(
            hidden,
            &mut workspace.k,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        self.v.run_into(
            hidden,
            &mut workspace.v,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        qwen36_full_attn_prep_f32_into_on_stream(
            &workspace.q_proj_output,
            &workspace.k,
            &self.q_norm_weight,
            &self.k_norm_weight,
            workspace.q_normed.output(),
            workspace.gate.output(),
            workspace.k_normed.output(),
            manifest.q_heads,
            manifest.kv_heads,
            manifest.head_dim,
            manifest.rms_eps,
            stream,
        )?;
        apply_rope(
            manifest,
            manifest.q_heads,
            &workspace.q_normed,
            &mut workspace.q_rope,
            position,
            stream,
        )?;
        apply_rope(
            manifest,
            manifest.kv_heads,
            &workspace.k_normed,
            &mut workspace.k_rope,
            position,
            stream,
        )?;
        append_rows_f32_into_on_stream(
            &workspace.k_rope,
            workspace.key_cache.output(),
            position,
            1,
            manifest.kv_heads * manifest.head_dim,
            stream,
        )?;
        append_rows_f32_into_on_stream(
            &workspace.v,
            workspace.value_cache.output(),
            position,
            1,
            manifest.kv_heads * manifest.head_dim,
            stream,
        )?;
        cached_gqa_attention_f32_into_on_stream(
            &workspace.q_rope,
            &workspace.key_cache,
            &workspace.value_cache,
            workspace.attn.output(),
            position + 1,
            manifest.q_heads,
            manifest.kv_heads,
            manifest.head_dim,
            stream,
        )?;
        sigmoid_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.attn,
            workspace.gated_attn.output(),
            stream,
        )?;
        self.o.run_into(
            &workspace.gated_attn,
            &mut workspace.output,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        Ok(Qwen36FullAttentionStep {
            q_proj_output: &workspace.q_proj_output,
            q_rope: &workspace.q_rope,
            attn: &workspace.attn,
            gated_attn: &workspace.gated_attn,
            output: &workspace.output,
        })
    }

    fn run_one_token_indexed<'a>(
        &'a self,
        workspace: &'a mut Qwen36FullAttentionWorkspace,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: &DeviceBuffer<u32>,
        cache_len: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<Qwen36FullAttentionStep<'a>> {
        self.q.run_into(
            hidden,
            &mut workspace.q_proj_output,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        self.k.run_into(
            hidden,
            &mut workspace.k,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        self.v.run_into(
            hidden,
            &mut workspace.v,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        qwen36_full_attn_prep_f32_into_on_stream(
            &workspace.q_proj_output,
            &workspace.k,
            &self.q_norm_weight,
            &self.k_norm_weight,
            workspace.q_normed.output(),
            workspace.gate.output(),
            workspace.k_normed.output(),
            manifest.q_heads,
            manifest.kv_heads,
            manifest.head_dim,
            manifest.rms_eps,
            stream,
        )?;
        apply_rope_indexed(
            manifest,
            manifest.q_heads,
            &workspace.q_normed,
            &mut workspace.q_rope,
            position,
            stream,
        )?;
        apply_rope_indexed(
            manifest,
            manifest.kv_heads,
            &workspace.k_normed,
            &mut workspace.k_rope,
            position,
            stream,
        )?;
        append_rows_f32_indexed_into_on_stream(
            &workspace.k_rope,
            workspace.key_cache.output(),
            position,
            workspace.cache_capacity - 1,
            1,
            manifest.kv_heads * manifest.head_dim,
            stream,
        )?;
        append_rows_f32_indexed_into_on_stream(
            &workspace.v,
            workspace.value_cache.output(),
            position,
            workspace.cache_capacity - 1,
            1,
            manifest.kv_heads * manifest.head_dim,
            stream,
        )?;
        cached_gqa_attention_f32_indexed_into_on_stream(
            &workspace.q_rope,
            &workspace.key_cache,
            &workspace.value_cache,
            workspace.attn.output(),
            cache_len,
            workspace.cache_capacity,
            manifest.q_heads,
            manifest.kv_heads,
            manifest.head_dim,
            stream,
        )?;
        sigmoid_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.attn,
            workspace.gated_attn.output(),
            stream,
        )?;
        self.o.run_into(
            &workspace.gated_attn,
            &mut workspace.output,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        Ok(Qwen36FullAttentionStep {
            q_proj_output: &workspace.q_proj_output,
            q_rope: &workspace.q_rope,
            attn: &workspace.attn,
            gated_attn: &workspace.gated_attn,
            output: &workspace.output,
        })
    }
}

impl Qwen36LinearAttentionWeights {
    /// Loads a linear-attention layer by layer index from the Qwen3.6 text stack.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
    ) -> Result<Self> {
        let fp8 = Rc::new(Qwen36LinearFp8Execution::new(checkpoint, manifest)?);
        Self::load_with_fp8(checkpoint, manifest, layer, fp8)
    }

    fn load_with_fp8(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
        fp8: Rc<Qwen36LinearFp8Execution>,
    ) -> Result<Self> {
        let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
            label: "Qwen3.6 linear attention",
            detail: "manifest has no linear-attention config".to_string(),
        })?;
        let prefix = format!("{}.layers.{layer}.linear_attn", manifest.tensor_prefix);
        let key_heads = linear.key_heads;
        let value_heads = linear.value_heads;
        let head_v_dim = linear.value_head_dim;
        let qkv_host = checkpoint.load_fp8_linear(&format!("{prefix}.in_proj_qkv"))?;
        let z_host = checkpoint.load_fp8_linear(&format!("{prefix}.in_proj_z"))?;
        let out_host = checkpoint.load_fp8_linear(&format!("{prefix}.out_proj"))?;

        // Reorder V heads from grouped-by-K to tiled order for tensors consumed after GDN prep.
        // qkv/conv stay in checkpoint order; qwen36_gdn_prep reorders V after depthwise conv.
        let z_host = reorder_fp8_v_rows(z_host, key_heads, value_heads, head_v_dim);
        let out_host = reorder_fp8_v_cols(out_host, key_heads, value_heads, head_v_dim);

        // Alpha/beta: BF16 [value_heads, hidden] — reorder rows
        let alpha_host = reorder_bf16_rows(
            read_bf16_matrix_host(
                checkpoint,
                &format!("{prefix}.in_proj_a.weight"),
                value_heads,
                manifest.hidden,
            )?,
            key_heads,
            value_heads,
        );
        let beta_host = reorder_bf16_rows(
            read_bf16_matrix_host(
                checkpoint,
                &format!("{prefix}.in_proj_b.weight"),
                value_heads,
                manifest.hidden,
            )?,
            key_heads,
            value_heads,
        );

        // Conv1d: BF16 [conv_dim, kernel]
        let conv_host = read_bf16_flat_host(
            checkpoint,
            &format!("{prefix}.conv1d.weight"),
            qkv_host.out_features * linear.conv_kernel,
        )?;

        // A_log / dt_bias: BF16 [value_heads] — reorder elements
        let a_log_host = read_bf16_flat_host(checkpoint, &format!("{prefix}.A_log"), value_heads)?;
        let dt_bias_host =
            read_bf16_flat_host(checkpoint, &format!("{prefix}.dt_bias"), value_heads)?;
        let a_log_host = reorder_v_heads_1d(a_log_host, key_heads, value_heads);
        let dt_bias_host = reorder_v_heads_1d(dt_bias_host, key_heads, value_heads);

        Ok(Self {
            fp8,
            qkv: Fp8Linear::from_host(&qkv_host)?,
            z: Fp8Linear::from_host(&z_host)?,
            alpha: Bf16Linear::from_host(&alpha_host, value_heads, manifest.hidden)?,
            beta: Bf16Linear::from_host(&beta_host, value_heads, manifest.hidden)?,
            conv_weight: DeviceBuffer::from_host(&conv_host)?,
            a_log: DeviceBuffer::from_host(&a_log_host)?,
            dt_bias: DeviceBuffer::from_host(&dt_bias_host)?,
            norm_weight: read_bf16_vector_as_f32_device(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                linear.value_head_dim,
            )?,
            out: Fp8Linear::from_host(&out_host)?,
        })
    }

    fn run_qkv(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(plans) = self.fp8.plans.as_ref() else {
            return self.qkv.run_into(
                hidden,
                &mut workspace.qkv_output,
                &mut workspace.fp8_dynamic_input,
                &mut workspace.fp8_dynamic_input_scale,
                stream,
            );
        };
        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            hidden,
            &mut workspace.fp8_dynamic_input,
            &mut workspace.fp8_dynamic_input_scale,
            stream,
        )?;
        self.qkv.run_prequantized_channel_scaled_with_plan_into(
            &self.fp8,
            &plans.qkv,
            &workspace.fp8_dynamic_input,
            &workspace.fp8_dynamic_input_scale,
            &mut workspace.qkv_output,
            stream,
        )
    }

    fn run_z(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(plans) = self.fp8.plans.as_ref() else {
            return self.z.run_into(
                hidden,
                &mut workspace.z_output,
                &mut workspace.fp8_dynamic_input,
                &mut workspace.fp8_dynamic_input_scale,
                stream,
            );
        };
        self.z.run_prequantized_channel_scaled_with_plan_into(
            &self.fp8,
            &plans.z,
            &workspace.fp8_dynamic_input,
            &workspace.fp8_dynamic_input_scale,
            &mut workspace.z_output,
            stream,
        )
    }

    fn run_output_projection(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(plans) = self.fp8.plans.as_ref() else {
            return self.out.run_into(
                &workspace.normed,
                &mut workspace.output,
                &mut workspace.fp8_dynamic_input,
                &mut workspace.fp8_dynamic_input_scale,
                stream,
            );
        };
        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            &workspace.normed,
            &mut workspace.fp8_value_input,
            &mut workspace.fp8_value_input_scale,
            stream,
        )?;
        self.out.run_prequantized_channel_scaled_with_plan_into(
            &self.fp8,
            &plans.out,
            &workspace.fp8_value_input,
            &workspace.fp8_value_input_scale,
            &mut workspace.output,
            stream,
        )
    }

    fn enqueue_pre_gdn(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_qkv(workspace, hidden, stream)?;
        self.run_z(workspace, hidden, stream)?;
        self.alpha.run_into(hidden, &mut workspace.alpha, stream)?;
        self.beta
            .run_into(hidden, &mut workspace.beta_input, stream)?;
        qwen36_gdn_prep_into_on_stream(
            &workspace.qkv_output,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            workspace.conv_state.inout(),
            workspace.linear.key_heads,
            workspace.linear.value_heads,
            workspace.linear.value_head_dim,
            stream,
        )?;
        qwen36_gdn_gate_into_on_stream(
            &workspace.alpha,
            &workspace.beta_input,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            workspace.linear.value_heads,
            stream,
        )
    }

    fn enqueue_gdn(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        gated_delta_net_128_f32_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            workspace.recurrent_state.inout(),
            workspace.gdn_output.output(),
            workspace.linear.value_heads,
            stream,
        )
    }

    fn enqueue_post_gdn(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        rms_eps: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        gated_rms_norm_f32_into_on_stream(
            &workspace.gdn_output,
            &workspace.z_output,
            &self.norm_weight,
            workspace.normed.output(),
            workspace.linear.value_heads,
            workspace.linear.value_head_dim,
            rms_eps,
            stream,
        )?;
        self.run_output_projection(workspace, stream)
    }

    /// Runs one token through this linear-attention layer.
    #[allow(clippy::needless_option_as_deref)]
    pub fn run_one_token<'a>(
        &'a self,
        workspace: &'a mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        rms_eps: f32,
        stream: &CudaStream,
        mut profile: Option<&mut QwenDecodeProfile>,
    ) -> Result<Qwen36LinearAttentionStep<'a>> {
        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || self.run_qkv(workspace, hidden, stream))?;
            profile.qwen36_linear_qkv_ms += ms;
            let (_, ms) = timed_cuda(stream, || self.run_z(workspace, hidden, stream))?;
            profile.qwen36_linear_z_ms += ms;
            let (_, ms) = timed_cuda(stream, || {
                self.alpha.run_into(hidden, &mut workspace.alpha, stream)?;
                self.beta
                    .run_into(hidden, &mut workspace.beta_input, stream)
            })?;
            profile.qwen36_linear_alpha_beta_ms += ms;
            let (_, ms) = timed_cuda(stream, || {
                qwen36_gdn_prep_into_on_stream(
                    &workspace.qkv_output,
                    &self.conv_weight,
                    workspace.q.output(),
                    workspace.k.output(),
                    workspace.v.output(),
                    workspace.conv_state.inout(),
                    workspace.linear.key_heads,
                    workspace.linear.value_heads,
                    workspace.linear.value_head_dim,
                    stream,
                )
            })?;
            profile.qwen36_linear_gdn_prep_ms += ms;
            let (_, ms) = timed_cuda(stream, || {
                qwen36_gdn_gate_into_on_stream(
                    &workspace.alpha,
                    &workspace.beta_input,
                    &self.a_log,
                    &self.dt_bias,
                    workspace.gate.output(),
                    workspace.beta.output(),
                    workspace.linear.value_heads,
                    stream,
                )
            })?;
            profile.qwen36_linear_gdn_gate_ms += ms;
            let (_, ms) = timed_cuda(stream, || {
                gated_delta_net_128_f32_into_on_stream(
                    &workspace.q,
                    &workspace.k,
                    &workspace.v,
                    &workspace.gate,
                    &workspace.beta,
                    workspace.recurrent_state.inout(),
                    workspace.gdn_output.output(),
                    workspace.linear.value_heads,
                    stream,
                )
            })?;
            profile.qwen36_linear_gdn_ms += ms;
            let (_, ms) = timed_cuda(stream, || {
                gated_rms_norm_f32_into_on_stream(
                    &workspace.gdn_output,
                    &workspace.z_output,
                    &self.norm_weight,
                    workspace.normed.output(),
                    workspace.linear.value_heads,
                    workspace.linear.value_head_dim,
                    rms_eps,
                    stream,
                )
            })?;
            profile.qwen36_linear_norm_ms += ms;
            let (_, ms) = timed_cuda(stream, || self.run_output_projection(workspace, stream))?;
            profile.qwen36_linear_out_ms += ms;
        } else {
            self.enqueue_pre_gdn(workspace, hidden, stream)?;
            self.enqueue_gdn(workspace, stream)?;
            self.enqueue_post_gdn(workspace, rms_eps, stream)?;
        }
        Ok(Qwen36LinearAttentionStep {
            qkv_output: &workspace.qkv_output,
            z_output: &workspace.z_output,
            gdn_output: &workspace.gdn_output,
            output: &workspace.output,
        })
    }

    /// Returns output width.
    pub fn output_width(&self) -> usize {
        self.out.rows
    }
}

impl Qwen36LinearAttentionWorkspace {
    /// Allocates one-token workspace for a Qwen3.6 linear-attention layer.
    pub fn new(
        manifest: &QwenModelManifest,
        linear: QwenLinearAttentionConfig,
        weights: &Qwen36LinearAttentionWeights,
    ) -> Result<Self> {
        let value_dim = linear.value_heads * linear.value_head_dim;
        Ok(Self {
            linear,
            fp8_dynamic_input: DeviceBuffer::zeroed(manifest.hidden)?,
            fp8_dynamic_input_scale: DeviceBuffer::zeroed(1)?,
            fp8_value_input: DeviceBuffer::zeroed(value_dim)?,
            fp8_value_input_scale: DeviceBuffer::zeroed(1)?,
            qkv_output: DeviceBuffer::zeroed(weights.qkv.rows)?,
            z_output: DeviceBuffer::zeroed(weights.z.rows)?,
            alpha: DeviceBuffer::zeroed(linear.value_heads)?,
            beta_input: DeviceBuffer::zeroed(linear.value_heads)?,
            gate: DeviceBuffer::zeroed(linear.value_heads)?,
            beta: DeviceBuffer::zeroed(linear.value_heads)?,
            q: DeviceBuffer::zeroed(value_dim)?,
            k: DeviceBuffer::zeroed(value_dim)?,
            v: DeviceBuffer::zeroed(value_dim)?,
            conv_state: DeviceBuffer::zeroed(weights.qkv.rows * (linear.conv_kernel - 1))?,
            recurrent_state: DeviceBuffer::zeroed(
                linear.value_heads * linear.value_head_dim * linear.value_head_dim,
            )?,
            gdn_output: DeviceBuffer::zeroed(value_dim)?,
            normed: DeviceBuffer::zeroed(value_dim)?,
            output: DeviceBuffer::zeroed(manifest.hidden)?,
        })
    }
}

impl Qwen36FullAttentionWorkspace {
    /// Allocates one-token workspace and K/V cache for a Qwen3.6 full-attention layer.
    pub fn new(
        manifest: &QwenModelManifest,
        weights: &Qwen36FullAttentionWeights,
        cache_capacity: usize,
    ) -> Result<Self> {
        if cache_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 full-attention cache",
                expected: "non-zero cache capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let q_width = manifest.q_heads * manifest.head_dim;
        let kv_width = manifest.kv_heads * manifest.head_dim;
        Ok(Self {
            fp8_dynamic_input: DeviceBuffer::zeroed(manifest.hidden.max(q_width))?,
            fp8_dynamic_input_scale: DeviceBuffer::zeroed(1)?,
            q_proj_output: DeviceBuffer::zeroed(weights.q.rows)?,
            q_normed: DeviceBuffer::zeroed(q_width)?,
            gate: DeviceBuffer::zeroed(q_width)?,
            k: DeviceBuffer::zeroed(kv_width)?,
            k_normed: DeviceBuffer::zeroed(kv_width)?,
            v: DeviceBuffer::zeroed(kv_width)?,
            q_rope: DeviceBuffer::zeroed(q_width)?,
            k_rope: DeviceBuffer::zeroed(kv_width)?,
            key_cache: DeviceBuffer::zeroed(cache_capacity * kv_width)?,
            value_cache: DeviceBuffer::zeroed(cache_capacity * kv_width)?,
            attn: DeviceBuffer::zeroed(q_width)?,
            gated_attn: DeviceBuffer::zeroed(q_width)?,
            output: DeviceBuffer::zeroed(manifest.hidden)?,
            cache_capacity,
        })
    }
}

impl Fp8Linear {
    fn from_host(host: &ModelOptFp8Linear) -> Result<Self> {
        Self::from_reordered_host(host, host.weight.clone())
    }

    fn from_reordered_host(host: &ModelOptFp8Linear, weight: Vec<u8>) -> Result<Self> {
        if weight.len() != host.expected_weight_bytes() {
            return Err(Error::Shape {
                label: "Qwen3.6 FP8 reordered weight",
                expected: format!("{} bytes", host.expected_weight_bytes()),
                actual: format!("{} bytes", weight.len()),
            });
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
            rows: host.out_features,
            cols: host.in_features,
            weight_scale: host.weight_scale,
            channel_weight_scale: host
                .channel_weight_scale
                .as_deref()
                .map(DeviceBuffer::from_host)
                .transpose()?,
            input_scale: host.input_scale,
        })
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        dynamic_input: &mut DeviceBuffer<u8>,
        dynamic_input_scale: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let result = if let Some(channel_weight_scale) = &self.channel_weight_scale {
            if std::env::var_os("QWEN36_FP8_W8A8").is_some() {
                return Err(Error::Format {
                    label: "Qwen3.6 compressed-tensors FP8",
                    detail: "dynamic per-token W8A8 activation quantization is not implemented"
                        .to_string(),
                });
            }
            fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream(
                input,
                dynamic_input,
                &self.weight,
                channel_weight_scale,
                dynamic_input_scale,
                output.output(),
                self.rows,
                self.cols,
                stream,
            )
        } else if std::env::var_os("QWEN36_FP8_W8A8").is_some() {
            let input_scale = self.input_scale.ok_or_else(|| Error::Format {
                label: "Qwen3.6 FP8 activation scale",
                detail: "checkpoint does not contain a static input scale".to_string(),
            })?;
            fp8_linear_w8a8_f32_into_on_stream(
                input,
                &self.weight,
                output.output(),
                self.rows,
                self.cols,
                self.weight_scale,
                input_scale,
                stream,
            )
        } else {
            if (self.rows, self.cols) == (8192, 2048) {
                fp8_linear_configured_f32_into_on_stream(
                    input,
                    &self.weight,
                    output.output(),
                    self.rows,
                    self.cols,
                    self.weight_scale,
                    128,
                    stream,
                )
            } else {
                fp8_linear_f32_into_on_stream(
                    input,
                    &self.weight,
                    output.output(),
                    self.rows,
                    self.cols,
                    self.weight_scale,
                    stream,
                )
            }
        };
        result?;
        maybe_round_device_f32_to_bf16(output, stream)
    }

    fn run_prequantized_channel_scaled_with_plan_into(
        &self,
        execution: &Qwen36LinearFp8Execution,
        plan: &Fp8TnMatmulPlan,
        input: &DeviceBuffer<u8>,
        input_scale: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let channel_scale = self
            .channel_weight_scale
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 channel-scaled FP8 plan",
                detail: "linear does not have per-output-channel scales".to_string(),
            })?;
        plan.run_with_alpha_on_stream(
            &execution.lt,
            &self.weight,
            input,
            output.output(),
            1.0,
            stream,
        )?;
        scale_channel_f32_device_scalar_in_place_on_stream(
            output.inout(),
            channel_scale,
            input_scale,
            stream,
        )?;
        maybe_round_device_f32_to_bf16(output, stream)
    }

    fn require_shape(&self, rows: usize, cols: usize, label: &'static str) -> Result<()> {
        if self.rows != rows || self.cols != cols {
            return Err(Error::Shape {
                label,
                expected: format!("rows={rows} cols={cols}"),
                actual: format!("rows={} cols={}", self.rows, self.cols),
            });
        }
        Ok(())
    }
}

impl Bf16Linear {
    fn load(checkpoint: &ModelOptCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weight: read_bf16_matrix_device(checkpoint, name, rows, cols)?,
            rows,
            cols,
        })
    }

    fn from_host(weight: &[u16], rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(weight)?,
            rows,
            cols,
        })
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
    }
}

fn read_bf16_matrix_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer<u16>> {
    let bytes = read_checked_bf16_bytes(checkpoint, name, &[rows, cols])?;
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>(),
    )
}

fn read_bf16_vector_as_f32_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    len: usize,
) -> Result<DeviceBuffer<f32>> {
    let bytes = read_checked_bf16_bytes(checkpoint, name, &[len])?;
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|chunk| {
                crate::nvfp4::format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
            })
            .collect::<Vec<_>>(),
    )
}

fn read_bf16_vector_delta_as_f32_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    len: usize,
) -> Result<DeviceBuffer<f32>> {
    let bytes = read_checked_bf16_bytes(checkpoint, name, &[len])?;
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|chunk| {
                1.0 + crate::nvfp4::format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
            })
            .collect::<Vec<_>>(),
    )
}

fn read_bf16_flat_host(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    len: usize,
) -> Result<Vec<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape.iter().product::<usize>() != len {
        return Err(shape_error(
            "BF16 flat tensor",
            info,
            format!("{len} BF16 values"),
        ));
    }
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_bf16_matrix_host(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<u16>> {
    let bytes = read_checked_bf16_bytes(checkpoint, name, &[rows, cols])?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

/// Reorders V heads from grouped-by-K `[K0_V0..V{r-1}, K1_V0..V{r-1}, ...]`
/// to tiled `[K0_V0, K1_V0, ..., K0_V1, K1_V1, ...]` for a 1D tensor of `value_heads` elements.
fn reorder_v_heads_1d(data: Vec<u16>, key_heads: usize, value_heads: usize) -> Vec<u16> {
    if key_heads == value_heads {
        return data;
    }
    let v_per_k = value_heads / key_heads;
    let mut out = vec![0u16; value_heads];
    for (v_k_head, value) in data.iter().copied().enumerate() {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        out[v_sub * key_heads + k_head] = value;
    }
    out
}

/// Reorders V head rows in a BF16 matrix `[value_heads, cols]`.
fn reorder_bf16_rows(data: Vec<u16>, key_heads: usize, value_heads: usize) -> Vec<u16> {
    if key_heads == value_heads {
        return data;
    }
    let cols = data.len() / value_heads;
    let v_per_k = value_heads / key_heads;
    let mut out = vec![0u16; data.len()];
    for (v_k_head, row) in data.chunks(cols).enumerate() {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let dst = (v_sub * key_heads + k_head) * cols;
        out[dst..dst + cols].copy_from_slice(row);
    }
    out
}

/// Reorders V rows in an FP8 ModelOpt linear weight.
fn reorder_fp8_v_rows(
    mut host: ModelOptFp8Linear,
    key_heads: usize,
    value_heads: usize,
    head_v_dim: usize,
) -> ModelOptFp8Linear {
    if key_heads == value_heads || host.out_features != value_heads * head_v_dim {
        return host;
    }
    let v_per_k = value_heads / key_heads;
    let mut reordered = vec![0u8; host.weight.len()];
    let row_bytes = head_v_dim * host.in_features;
    for (v_k_head, src_row) in host
        .weight
        .chunks_exact(row_bytes)
        .take(value_heads)
        .enumerate()
    {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let dst = (v_sub * key_heads + k_head) * row_bytes;
        reordered[dst..dst + row_bytes].copy_from_slice(src_row);
    }
    host.weight = reordered;
    if let Some(scales) = host.channel_weight_scale.take() {
        let mut reordered_scales = vec![0.0; scales.len()];
        for (v_k_head, src) in scales.chunks_exact(head_v_dim).enumerate() {
            let k_head = v_k_head / v_per_k;
            let v_sub = v_k_head % v_per_k;
            let dst = (v_sub * key_heads + k_head) * head_v_dim;
            reordered_scales[dst..dst + head_v_dim].copy_from_slice(src);
        }
        host.channel_weight_scale = Some(reordered_scales);
    }
    host
}

/// Reorders V columns in the out_proj FP8 weight `[hidden, value_dim]`.
fn reorder_fp8_v_cols(
    mut host: ModelOptFp8Linear,
    key_heads: usize,
    value_heads: usize,
    head_v_dim: usize,
) -> ModelOptFp8Linear {
    if key_heads == value_heads || host.in_features != value_heads * head_v_dim {
        return host;
    }
    let v_per_k = value_heads / key_heads;
    let mut reordered = vec![0u8; host.weight.len()];
    for v_k_head in 0..value_heads {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let src_col_start = v_k_head * head_v_dim;
        let dst_col_start = (v_sub * key_heads + k_head) * head_v_dim;
        for row in 0..host.out_features {
            let src = row * host.in_features + src_col_start;
            let dst = row * host.in_features + dst_col_start;
            reordered[dst..dst + head_v_dim].copy_from_slice(&host.weight[src..src + head_v_dim]);
        }
    }
    host.weight = reordered;
    host
}

fn read_checked_bf16_bytes(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u8>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(shape_error(
            "BF16 tensor",
            info,
            format!("dtype=BF16 shape={shape:?}"),
        ));
    }
    shard.read_tensor_bytes(name)
}

fn shape_error(label: &'static str, info: &SafeTensorInfo, expected: String) -> Error {
    Error::Shape {
        label,
        expected,
        actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
    }
}

/// Applies the configured RoPE variant to one head-batched row-major tensor.
///
/// Qwen3.5/3.6 MoE uses IMRoPE (interleaved MRoPE) with `mrope_section=[11,11,10]`
/// unconditionally — including 1D text-only decode. llama.cpp's `qwen35moe` model
/// passes `LLAMA_ROPE_TYPE_IMROPE` and `sections` to `ggml_rope_multi` for all
/// attention layers. For 1D text, positions are `[pos, pos, pos, 0]` (T/H/W
/// identical, extra=0), but the section-based frequency assignment still differs
/// from standard Neox: groups of 3 consecutive pairs share the same frequency.
fn apply_rope(
    manifest: &QwenModelManifest,
    rows: usize,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    position: usize,
    stream: &CudaStream,
) -> Result<()> {
    let sections = manifest.mrope_sections.ok_or_else(|| Error::Format {
        label: "Qwen3.6 IMRoPE",
        detail: "mrope_sections not set in manifest".to_string(),
    })?;
    rope_imrope_f32_into_on_stream(
        rows,
        manifest.head_dim,
        manifest.rotary_dim,
        MropeSections {
            v0: sections[0],
            v1: sections[1],
            v2: sections[2],
            v3: sections[3],
        },
        [position as u32, position as u32, position as u32, 0],
        input,
        output.output(),
        manifest.rope_theta,
        stream,
    )
}

fn apply_rope_indexed(
    manifest: &QwenModelManifest,
    rows: usize,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    position: &DeviceBuffer<u32>,
    stream: &CudaStream,
) -> Result<()> {
    let sections = manifest.mrope_sections.ok_or_else(|| Error::Format {
        label: "Qwen3.6 indexed IMRoPE",
        detail: "mrope_sections not set in manifest".to_string(),
    })?;
    rope_imrope_f32_indexed_into_on_stream(
        rows,
        manifest.head_dim,
        manifest.rotary_dim,
        MropeSections {
            v0: sections[0],
            v1: sections[1],
            v2: sections[2],
            v3: sections[3],
        },
        position,
        input,
        output.output(),
        manifest.rope_theta,
        stream,
    )
}

// ---------------------------------------------------------------------------
// MoE + shared expert FFN
// ---------------------------------------------------------------------------

/// Device-ready weights for the Qwen3.6 MoE + shared-expert FFN block.
///
/// Every Qwen3.6 text layer carries a BF16 router over 256 routed experts
/// (top-8), a quantized shared expert, and a BF16 scalar shared-expert gate.
/// NVFP4 layers use Marlin/SM12x; mixed-precision layers keep channel-scaled
/// FP8 expert tables device-resident for device-routed W8A8 execution.
pub struct Qwen36MoeWeights {
    router: Bf16Linear,
    experts: Vec<LazyQwen36Expert>,
    expert_ptrs: super::infer::MoeExpertPointerTables,
    gate_up_w4a16_weight_scale_2: DeviceBuffer<f32>,
    gate_up_w4a16_unity_alphas: DeviceBuffer<f32>,
    storage_plan: Qwen36MoeStoragePlan,
    gate_up_storage: Qwen36GateUpStorage,
    fp8_experts: Option<Qwen36Fp8Experts>,
    shared: Qwen36SharedExpertStorage,
    shared_gate: Bf16Linear,
    _sm12x_down: Vec<Sm12xFp4DeviceGemmWeight>,
    sm12x_down_tiles: Option<DeviceBuffer<*const u8>>,
    sm12x_down_scales: Option<DeviceBuffer<*const u32>>,
    sm12x_down_m_tiles: usize,
    sm12x_down_k_tiles: usize,
    num_experts: usize,
    experts_per_token: usize,
    expert_intermediate: usize,
    norm_topk_prob: bool,
}

enum Qwen36GateUpStorage {
    Marlin(MarlinNvfp4GateUp),
    Grouped { _weights: Vec<Nvfp4DeviceLinear> },
    Fp8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen36DownStorage {
    Legacy,
    Sm12x,
    Fp8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen36MoeStoragePlan {
    down: Qwen36DownStorage,
}

impl Qwen36MoeStoragePlan {
    fn select(request_sm12x_down: bool, sm12x_down_cache_complete: bool) -> Self {
        let down = if request_sm12x_down && sm12x_down_cache_complete {
            Qwen36DownStorage::Sm12x
        } else {
            Qwen36DownStorage::Legacy
        };
        Self { down }
    }
}

struct LazyQwen36Expert {
    checkpoint: ModelOptCheckpoint,
    prefix: String,
    gate_up_w4a16: std::cell::RefCell<Option<Nvfp4DeviceLinear>>,
    down_w4a16: std::cell::RefCell<Option<Nvfp4DeviceLinear>>,
    gate_up_sm12x: std::cell::RefCell<Option<Sm12xDeviceLinear>>,
    down_sm12x: std::cell::RefCell<Option<Sm12xDeviceLinear>>,
}

struct Qwen36SharedExpert {
    gate_up: Nvfp4DeviceLinear,
    down: Nvfp4DeviceLinear,
}

struct Qwen36Fp8ExpertTable {
    _weights: DeviceBuffer<u8>,
    _scales: DeviceBuffer<f32>,
    weights: DeviceBuffer<*const u8>,
    scales: DeviceBuffer<*const f32>,
}

struct Qwen36Fp8Experts {
    gate: Qwen36Fp8ExpertTable,
    up: Qwen36Fp8ExpertTable,
    down: Qwen36Fp8ExpertTable,
}

enum Qwen36SharedExpertStorage {
    Nvfp4(Qwen36SharedExpert),
    Fp8 { gate_up: Fp8Linear, down: Fp8Linear },
}

/// A device-resident NVFP4 linear weight for W4A16 execution.
///
/// Stores the raw ModelOpt packed E2M1 weight and UE4M3 per-block scales
/// (not cuBLASLt-repacked), plus the scalar `weight_scale_2`. For W4A16,
/// activations stay f32; the GEMM dequantizes weights on the fly.
struct Nvfp4DeviceLinear {
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    weight_scale_2: f32,
    input_scale: f32,
    out_features: usize,
    in_features: usize,
}

struct Sm12xDeviceLinear {
    weight: Sm12xFp4DeviceGemmWeight,
    weight_scale_2: f32,
    out_features: usize,
    in_features: usize,
}

/// Mutable one-token decode workspace for the Qwen3.6 MoE + shared-expert FFN.
pub struct Qwen36MoeWorkspace {
    pub router_logits: DeviceBuffer<f32>,
    pub route: MoeRouteWorkspace,
    pub gate_up_input: Nvfp4Matrix,
    pub gate_up_input_simple_scales: DeviceBuffer<u8>,
    pub grouped_gate_up: Option<GroupedGemvWorkspace>,
    marlin_gate_up_output: DeviceBuffer<f32>,
    marlin_gate_up_table: DeviceBuffer<*const f32>,
    fp8_hidden_input: DeviceBuffer<u8>,
    fp8_hidden_input_scale: DeviceBuffer<f32>,
    fp8_down_input: DeviceBuffer<u8>,
    fp8_down_input_scales: DeviceBuffer<f32>,
    fp8_shared_input: DeviceBuffer<u8>,
    fp8_shared_input_scale: DeviceBuffer<f32>,
    sm12x_down: Sm12xGateUpWorkspace,
    pub grouped_down: Option<MoeGroupedDownWorkspace>,
    pub fallback_gate_up_out: DeviceBuffer<f32>,
    pub fallback_down_input: DeviceBuffer<f32>,
    pub fallback_down_out: DeviceBuffer<f32>,
    pub shared_gate_up_output: DeviceBuffer<f32>,
    pub shared_activated: DeviceBuffer<f32>,
    pub shared_output: DeviceBuffer<f32>,
    pub shared_gate_logits: DeviceBuffer<f32>,
    pub shared_gated: DeviceBuffer<f32>,
    pub moe_out: DeviceBuffer<f32>,
    pub ffn_out: DeviceBuffer<f32>,
    pub ffn_residual: DeviceBuffer<f32>,
}

struct Sm12xGateUpWorkspace {
    b_tiles: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u32>,
    _outputs: Vec<F32Matrix>,
    c: DeviceBuffer<*const f32>,
    d: DeviceBuffer<*mut f32>,
    groups: usize,
}

impl Sm12xGateUpWorkspace {
    fn new(
        out_features: usize,
        in_features: usize,
        groups: usize,
        b_groups: usize,
    ) -> Result<Self> {
        if !out_features.is_multiple_of(16) || !in_features.is_multiple_of(64) {
            return Err(Error::Shape {
                label: "Qwen3.6 SM12x gate/up workspace",
                expected: "out_features multiple of 16 and in_features multiple of 64".to_string(),
                actual: format!("out_features={out_features} in_features={in_features}"),
            });
        }
        let mut outputs = Vec::with_capacity(groups);
        let mut c_ptrs = Vec::with_capacity(groups);
        let mut d_ptrs = Vec::with_capacity(groups);
        for _ in 0..groups {
            let mut output = F32Matrix::zeroed(out_features, 1)?;
            c_ptrs.push(output.data_ptr());
            d_ptrs.push(output.data_mut_ptr());
            outputs.push(output);
        }
        Ok(Self {
            b_tiles: DeviceBuffer::zeroed(b_groups * (in_features / 64) * 512)?,
            b_scales: DeviceBuffer::zeroed(b_groups * (in_features / 64))?,
            _outputs: outputs,
            c: DeviceBuffer::from_host(&c_ptrs)?,
            d: DeviceBuffer::from_host(&d_ptrs)?,
            groups,
        })
    }
}

/// Borrowed outputs from one MoE/shared-expert FFN step.
pub struct Qwen36MoeStep<'a> {
    /// Router top-k indices (host-visible via copy).
    pub route_indices: &'a DeviceBuffer<u32>,
    /// Router top-k weights.
    pub route_weights: &'a DeviceBuffer<f32>,
    /// Final residual FFN output rounded to BF16 precision in F32 storage.
    pub ffn_out: &'a DeviceBuffer<f32>,
}

impl Qwen36MoeWeights {
    /// Loads the MoE + shared-expert FFN for layer `layer`.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
        cache_prepared: bool,
    ) -> Result<Self> {
        let (experts, experts_per_token, expert_intermediate, norm_topk_prob) = match manifest.ffn {
            QwenFfnConfig::Moe {
                experts,
                experts_per_token,
                expert_intermediate,
                norm_topk_prob,
            } => (
                experts,
                experts_per_token,
                expert_intermediate,
                norm_topk_prob,
            ),
            QwenFfnConfig::Dense => {
                return Err(Error::Format {
                    label: "Qwen3.6 MoE FFN",
                    detail: "expected MoE config, got Dense".to_string(),
                });
            }
        };
        let prefix = format!("{}.layers.{layer}.mlp", manifest.tensor_prefix);
        let router = Bf16Linear::load(
            checkpoint,
            &format!("{prefix}.gate.weight"),
            experts,
            manifest.hidden,
        )?;
        let first_gate = format!("{prefix}.experts.0.gate_proj");
        let uses_nvfp4 = checkpoint.contains_tensor(&format!("{first_gate}.weight_scale_2"))
            || checkpoint.contains_tensor(&format!("{first_gate}.weight_global_scale"));
        if !uses_nvfp4 {
            return Self::load_fp8(
                checkpoint,
                manifest,
                prefix,
                router,
                experts,
                experts_per_token,
                expert_intermediate,
                norm_topk_prob,
            );
        }
        let sm12x_cache_dir = if cache_prepared {
            prepared_layer_dir(checkpoint, layer)
        } else {
            ensure_layer_cache(checkpoint, manifest, layer)?
        };

        let mut lazy_experts = Vec::with_capacity(experts);
        for expert_idx in 0..experts {
            lazy_experts.push(LazyQwen36Expert {
                checkpoint: checkpoint.clone(),
                prefix: format!("{prefix}.experts.{expert_idx}"),
                gate_up_w4a16: std::cell::RefCell::new(None),
                down_w4a16: std::cell::RefCell::new(None),
                gate_up_sm12x: std::cell::RefCell::new(None),
                down_sm12x: std::cell::RefCell::new(None),
            });
        }

        // Compressed-tensors checkpoints use W4A4 for the shared expert and
        // provide the scale metadata needed by the grouped routed fallback.
        let compressed_tensors =
            checkpoint.contains_tensor(&format!("{prefix}.experts.0.gate_proj.weight_packed"));
        let request_sm12x_down = true;
        let sm12x_down_cache_complete = request_sm12x_down
            && (0..experts).all(|expert_idx| {
                sm12x_cache_dir
                    .join(format!("expert-{expert_idx:03}.down.s12x"))
                    .is_file()
            });
        let storage_plan =
            Qwen36MoeStoragePlan::select(request_sm12x_down, sm12x_down_cache_complete);

        // Pointer table fields which are irrelevant to the selected path remain
        // null. Their allocations are tiny and preserve the shared table ABI.
        let gate_up_ptrs = vec![std::ptr::null(); experts];
        let gate_up_scale_ptrs = vec![std::ptr::null(); experts];
        let mut gate_up_grouped_value_ptrs = vec![std::ptr::null(); experts];
        let mut gate_up_grouped_scale_ptrs = vec![std::ptr::null(); experts];
        let down_ptrs = vec![std::ptr::null(); experts];
        let down_scale_ptrs = vec![std::ptr::null(); experts];
        let mut down_grouped_value_ptrs = vec![std::ptr::null(); experts];
        let mut down_grouped_scale_ptrs = vec![std::ptr::null(); experts];
        let mut down_input_scales = Vec::with_capacity(experts);
        let mut down_alphas = Vec::with_capacity(experts);
        let mut shared_gate_up_input_scale: Option<f32> = None;
        let mut gate_up_alphas = Vec::with_capacity(experts);
        let mut gate_up_w4a16_weight_scale_2 = Vec::with_capacity(experts);
        let mut marlin_gate_up_weights = Vec::with_capacity(experts);
        let mut grouped_gate_up_weights = Vec::with_capacity(experts);
        let mut sm12x_down = Vec::with_capacity(experts);
        let mut sm12x_down_tile_ptrs = Vec::with_capacity(experts);
        let mut sm12x_down_scale_ptrs = Vec::with_capacity(experts);
        let mut sm12x_down_m_tiles = 0usize;
        let mut sm12x_down_k_tiles = 0usize;
        for (expert_idx, expert) in lazy_experts.iter().enumerate() {
            let gate = checkpoint.load_nvfp4_linear(&format!("{}.gate_proj", expert.prefix))?;
            let up = checkpoint.load_nvfp4_linear(&format!("{}.up_proj", expert.prefix))?;
            let weight = ModelOptNvfp4Linear::concat_out_features(
                format!("{}.gate_up_proj", expert.prefix),
                &gate,
                &up,
            )?;
            if compressed_tensors {
                match shared_gate_up_input_scale {
                    None => shared_gate_up_input_scale = Some(weight.input_scale),
                    Some(first) if first.to_bits() == weight.input_scale.to_bits() => {}
                    Some(_) => shared_gate_up_input_scale = Some(f32::NAN),
                }
                gate_up_alphas.push(weight.weight_scale_2 * weight.input_scale);
            }
            gate_up_w4a16_weight_scale_2.push(weight.weight_scale_2);
            marlin_gate_up_weights.push(weight);

            match storage_plan.down {
                Qwen36DownStorage::Legacy => {
                    let weight = expert.get_down_w4a16()?;
                    down_input_scales.push(weight.input_scale);
                    down_alphas.push(weight.weight_scale_2 * weight.input_scale);
                    down_grouped_value_ptrs[expert_idx] =
                        weight.packed_weight.as_const_ptr().cast();
                    down_grouped_scale_ptrs[expert_idx] = weight.weight_scale.as_const_ptr().cast();
                }
                Qwen36DownStorage::Sm12x => {
                    let weight =
                        checkpoint.load_nvfp4_linear(&format!("{}.down_proj", expert.prefix))?;
                    down_input_scales.push(weight.input_scale);
                    down_alphas.push(weight.weight_scale_2 * weight.input_scale);
                }
                Qwen36DownStorage::Fp8 => {
                    unreachable!("NVFP4 loader cannot select FP8 down storage")
                }
            }

            if storage_plan.down == Qwen36DownStorage::Sm12x {
                let path = sm12x_cache_dir.join(format!("expert-{expert_idx:03}.down.s12x"));
                let weight = Sm12xFp4GemmWeight::read_cache_file(&path)?.to_device()?;
                sm12x_down_m_tiles = weight.m_tiles();
                sm12x_down_k_tiles = weight.k_tiles();
                sm12x_down_tile_ptrs.push(weight.tiles_ptr());
                sm12x_down_scale_ptrs.push(weight.scales_ptr());
                sm12x_down.push(weight);
            }
        }

        // The Marlin tensor-core path is substantially faster than the SIMT
        // grouped W4A4 kernel for this batch-one top-8 shape. Keep grouped W4A4
        // as the device-support fallback without retaining both weight layouts.
        let gate_up_storage = match MarlinNvfp4GateUp::new(&marlin_gate_up_weights) {
            Ok(marlin) => Qwen36GateUpStorage::Marlin(marlin),
            Err(_error) if compressed_tensors => {
                if shared_gate_up_input_scale.is_none_or(f32::is_nan) {
                    return Err(Error::Format {
                        label: "Qwen3.6 grouped gate/up",
                        detail: "expert gate/up input scales are not shared".to_string(),
                    });
                }
                for (expert_idx, weight) in marlin_gate_up_weights.iter().enumerate() {
                    let device = Nvfp4DeviceLinear::from_host(weight)?;
                    gate_up_grouped_value_ptrs[expert_idx] =
                        device.packed_weight.as_const_ptr().cast();
                    gate_up_grouped_scale_ptrs[expert_idx] =
                        device.weight_scale.as_const_ptr().cast();
                    grouped_gate_up_weights.push(device);
                }
                Qwen36GateUpStorage::Grouped {
                    _weights: grouped_gate_up_weights,
                }
            }
            Err(error) => return Err(error),
        };
        let expert_ptrs = MoeExpertPointerTables {
            gate_up_values: DeviceBuffer::from_host(&gate_up_ptrs)?,
            gate_up_scales: DeviceBuffer::from_host(&gate_up_scale_ptrs)?,
            gate_up_grouped_values: DeviceBuffer::from_host(&gate_up_grouped_value_ptrs)?,
            gate_up_grouped_scales: DeviceBuffer::from_host(&gate_up_grouped_scale_ptrs)?,
            down_values: DeviceBuffer::from_host(&down_ptrs)?,
            down_scales: DeviceBuffer::from_host(&down_scale_ptrs)?,
            down_grouped_values: DeviceBuffer::from_host(&down_grouped_value_ptrs)?,
            down_grouped_scales: DeviceBuffer::from_host(&down_grouped_scale_ptrs)?,
            down_input_scales: DeviceBuffer::from_host(&down_input_scales)?,
            down_alphas: DeviceBuffer::from_host(&down_alphas)?,
            shared_gate_up_input_scale: shared_gate_up_input_scale.filter(|v| !v.is_nan()),
            gate_up_alphas: DeviceBuffer::from_host(&gate_up_alphas)?,
        };

        let shared_gate_up = load_concat_gate_up(
            checkpoint,
            &format!("{prefix}.shared_expert.gate_proj"),
            &format!("{prefix}.shared_expert.up_proj"),
            "Qwen3.6 shared expert gate/up",
        )?;
        let shared_down =
            checkpoint.load_nvfp4_linear(&format!("{prefix}.shared_expert.down_proj"))?;
        let shared_intermediate = shared_gate_up.out_features / 2;
        if shared_gate_up.in_features != manifest.hidden
            || shared_down.in_features != shared_intermediate
            || shared_down.out_features != manifest.hidden
        {
            return Err(Error::Shape {
                label: "Qwen3.6 shared expert",
                expected: format!(
                    "gate_up in={} out=2*{} down in={} out={}",
                    manifest.hidden, shared_intermediate, shared_intermediate, manifest.hidden
                ),
                actual: format!(
                    "gate_up in={} out={} down in={} out={}",
                    shared_gate_up.in_features,
                    shared_gate_up.out_features,
                    shared_down.in_features,
                    shared_down.out_features
                ),
            });
        }
        let shared = Qwen36SharedExpertStorage::Nvfp4(Qwen36SharedExpert {
            gate_up: Nvfp4DeviceLinear::from_host(&shared_gate_up)?,
            down: Nvfp4DeviceLinear::from_host(&shared_down)?,
        });

        let shared_gate = Bf16Linear::load(
            checkpoint,
            &format!("{prefix}.shared_expert_gate.weight"),
            1,
            manifest.hidden,
        )?;

        Ok(Self {
            router,
            experts: lazy_experts,
            expert_ptrs,
            gate_up_w4a16_weight_scale_2: DeviceBuffer::from_host(&gate_up_w4a16_weight_scale_2)?,
            gate_up_w4a16_unity_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
            storage_plan,
            gate_up_storage,
            fp8_experts: None,
            shared,
            shared_gate,
            _sm12x_down: sm12x_down,
            sm12x_down_tiles: if storage_plan.down == Qwen36DownStorage::Sm12x {
                Some(DeviceBuffer::from_host(&sm12x_down_tile_ptrs)?)
            } else {
                None
            },
            sm12x_down_scales: if storage_plan.down == Qwen36DownStorage::Sm12x {
                Some(DeviceBuffer::from_host(&sm12x_down_scale_ptrs)?)
            } else {
                None
            },
            sm12x_down_m_tiles,
            sm12x_down_k_tiles,
            num_experts: experts,
            experts_per_token,
            expert_intermediate,
            norm_topk_prob,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_fp8(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        prefix: String,
        router: Bf16Linear,
        experts: usize,
        experts_per_token: usize,
        expert_intermediate: usize,
        norm_topk_prob: bool,
    ) -> Result<Self> {
        let expert_prefix =
            |expert: usize, projection: &str| format!("{prefix}.experts.{expert}.{projection}");
        let fp8_experts = Qwen36Fp8Experts {
            gate: Qwen36Fp8ExpertTable::load(
                checkpoint,
                experts,
                expert_intermediate,
                manifest.hidden,
                |expert| expert_prefix(expert, "gate_proj"),
            )?,
            up: Qwen36Fp8ExpertTable::load(
                checkpoint,
                experts,
                expert_intermediate,
                manifest.hidden,
                |expert| expert_prefix(expert, "up_proj"),
            )?,
            down: Qwen36Fp8ExpertTable::load(
                checkpoint,
                experts,
                manifest.hidden,
                expert_intermediate,
                |expert| expert_prefix(expert, "down_proj"),
            )?,
        };
        let shared_gate =
            checkpoint.load_fp8_linear(&format!("{prefix}.shared_expert.gate_proj"))?;
        let shared_up = checkpoint.load_fp8_linear(&format!("{prefix}.shared_expert.up_proj"))?;
        let shared_gate_up =
            concat_fp8_out_features(shared_gate, shared_up, "Qwen3.6 FP8 shared expert gate/up")?;
        let shared_down =
            checkpoint.load_fp8_linear(&format!("{prefix}.shared_expert.down_proj"))?;
        let shared = Qwen36SharedExpertStorage::Fp8 {
            gate_up: Fp8Linear::from_host(&shared_gate_up)?,
            down: Fp8Linear::from_host(&shared_down)?,
        };
        let shared_gate = Bf16Linear::load(
            checkpoint,
            &format!("{prefix}.shared_expert_gate.weight"),
            1,
            manifest.hidden,
        )?;
        let null_u8 = vec![std::ptr::null(); experts];
        let expert_ptrs = MoeExpertPointerTables {
            gate_up_values: DeviceBuffer::from_host(&null_u8)?,
            gate_up_scales: DeviceBuffer::from_host(&null_u8)?,
            gate_up_grouped_values: DeviceBuffer::from_host(&null_u8)?,
            gate_up_grouped_scales: DeviceBuffer::from_host(&null_u8)?,
            down_values: DeviceBuffer::from_host(&null_u8)?,
            down_scales: DeviceBuffer::from_host(&null_u8)?,
            down_grouped_values: DeviceBuffer::from_host(&null_u8)?,
            down_grouped_scales: DeviceBuffer::from_host(&null_u8)?,
            down_input_scales: DeviceBuffer::from_host(&vec![1.0; experts])?,
            down_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
            shared_gate_up_input_scale: None,
            gate_up_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
        };
        Ok(Self {
            router,
            experts: Vec::new(),
            expert_ptrs,
            gate_up_w4a16_weight_scale_2: DeviceBuffer::from_host(&vec![1.0; experts])?,
            gate_up_w4a16_unity_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
            storage_plan: Qwen36MoeStoragePlan {
                down: Qwen36DownStorage::Fp8,
            },
            gate_up_storage: Qwen36GateUpStorage::Fp8,
            fp8_experts: Some(fp8_experts),
            shared,
            shared_gate,
            _sm12x_down: Vec::new(),
            sm12x_down_tiles: None,
            sm12x_down_scales: None,
            sm12x_down_m_tiles: 0,
            sm12x_down_k_tiles: 0,
            num_experts: experts,
            experts_per_token,
            expert_intermediate,
            norm_topk_prob,
        })
    }

    /// Returns `(experts, top_k, expert_intermediate)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.num_experts,
            self.experts_per_token,
            self.expert_intermediate,
        )
    }

    fn workspace(&self, manifest: &QwenModelManifest) -> Result<Qwen36MoeWorkspace> {
        let enable_grouped = true;
        Qwen36MoeWorkspace::new_for_paths(
            manifest,
            enable_grouped,
            self.storage_plan.down == Qwen36DownStorage::Sm12x,
        )
    }

    /// Prepares routing and any activation state required by the selected gate/up path.
    pub fn prepare_routed_gate_up(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.router
            .run_into(ffn_norm, &mut workspace.router_logits, stream)?;
        workspace
            .route
            .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)?;
        if matches!(self.gate_up_storage, Qwen36GateUpStorage::Grouped { .. }) {
            let input_scale = self
                .expert_ptrs
                .shared_gate_up_input_scale
                .expect("grouped W4A4 storage requires a shared input scale");
            quantize_nvfp4_vector_simple_scales_f32_into_on_stream(
                manifest.hidden,
                ffn_norm,
                &mut workspace.gate_up_input,
                &mut workspace.gate_up_input_simple_scales,
                input_scale,
                stream,
            )?;
        }
        Ok(())
    }

    /// Runs the selected routed gate/up kernel using prepared route state.
    pub fn run_routed_gate_up_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.gate_up_storage {
            Qwen36GateUpStorage::Grouped { .. } => self.run_grouped_gate_up_only(workspace, stream),
            Qwen36GateUpStorage::Marlin(marlin) => marlin.run_on_stream(
                &workspace.route.indices,
                ffn_norm,
                workspace.marlin_gate_up_output.output(),
                stream,
            ),
            Qwen36GateUpStorage::Fp8 => {
                let fp8 = self.fp8_experts.as_ref().ok_or_else(|| Error::Format {
                    label: "Qwen3.6 FP8 routed gate/up",
                    detail: "FP8 expert tables are unavailable".to_string(),
                })?;
                quantize_fp8_e4m3_dynamic_f32_into_on_stream(
                    ffn_norm,
                    &mut workspace.fp8_hidden_input,
                    &mut workspace.fp8_hidden_input_scale,
                    stream,
                )?;
                fp8_moe_grouped_gate_up_f32_into_on_stream(
                    &workspace.route.indices,
                    &workspace.fp8_hidden_input,
                    &workspace.fp8_hidden_input_scale,
                    &fp8.gate.weights,
                    &fp8.gate.scales,
                    &fp8.up.weights,
                    &fp8.up.scales,
                    workspace.marlin_gate_up_output.output(),
                    self.expert_intermediate,
                    ffn_norm.len(),
                    self.experts_per_token,
                    stream,
                )
            }
        }
    }

    /// Prepares route indices and the quantized gate/up input for grouped gate/up benchmarking.
    pub fn prepare_grouped_gate_up(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let input_scale =
            self.expert_ptrs
                .shared_gate_up_input_scale
                .ok_or_else(|| Error::Format {
                    label: "Qwen3.6 grouped gate/up",
                    detail: "expert gate/up input scales are not shared".to_string(),
                })?;
        self.router
            .run_into(ffn_norm, &mut workspace.router_logits, stream)?;
        workspace
            .route
            .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)?;
        quantize_nvfp4_vector_simple_scales_f32_into_on_stream(
            manifest.hidden,
            ffn_norm,
            &mut workspace.gate_up_input,
            &mut workspace.gate_up_input_simple_scales,
            input_scale,
            stream,
        )
    }

    /// Runs only the routed grouped gate/up stage using already-prepared route and input.
    pub fn run_grouped_gate_up_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let grouped_gate_up = workspace
            .grouped_gate_up
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped gate/up",
                detail: "grouped gate/up workspace is unavailable".to_string(),
            })?;
        grouped_gate_up.run_gate_up_device_route(
            &workspace.route,
            &self.expert_ptrs,
            &workspace.gate_up_input,
            workspace.gate_up_input_simple_scales.as_const_ptr().cast(),
            stream,
        )?;
        Ok(())
    }

    /// Runs only device-routed grouped W4A16 gate/up using raw ModelOpt weights.
    pub fn run_grouped_w4a16_gate_up_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let grouped_gate_up = workspace
            .grouped_gate_up
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped W4A16 gate/up",
                detail: "grouped gate/up workspace is unavailable".to_string(),
            })?;
        nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
            &workspace.route.indices,
            ffn_norm,
            &self.expert_ptrs.gate_up_grouped_values,
            &self.expert_ptrs.gate_up_grouped_scales,
            &self.gate_up_w4a16_weight_scale_2,
            &grouped_gate_up.d,
            self.expert_intermediate * 2,
            ffn_norm.len(),
            stream,
        )
    }

    /// Runs only the router and top-k stage.
    pub fn run_router_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.router
            .run_into(ffn_norm, &mut workspace.router_logits, stream)?;
        workspace
            .route
            .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)
    }

    /// Runs only the router projection stage.
    pub fn run_router_linear_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.router
            .run_into(ffn_norm, &mut workspace.router_logits, stream)
    }

    /// Runs only top-k using already-computed router logits.
    pub fn run_topk_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace
            .route
            .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)
    }

    /// Prepares routed down inputs from already-computed gate/up outputs.
    pub fn prepare_grouped_down(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let gate_up_table = match &self.gate_up_storage {
            Qwen36GateUpStorage::Grouped { .. } => {
                &workspace
                    .grouped_gate_up
                    .as_ref()
                    .ok_or_else(|| Error::Format {
                        label: "Qwen3.6 grouped down",
                        detail: "grouped gate/up workspace is unavailable".to_string(),
                    })?
                    .c
            }
            Qwen36GateUpStorage::Marlin(_) | Qwen36GateUpStorage::Fp8 => {
                &workspace.marlin_gate_up_table
            }
        };
        let grouped_down = workspace
            .grouped_down
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped down",
                detail: "grouped down workspace is unavailable".to_string(),
            })?;
        let enable_sm12x = self.storage_plan.down == Qwen36DownStorage::Sm12x;
        if enable_sm12x && self.sm12x_down_tiles.is_some() && self.sm12x_down_scales.is_some() {
            let gate_up_alpha_table =
                if matches!(&self.gate_up_storage, Qwen36GateUpStorage::Grouped { .. }) {
                    &self.expert_ptrs.gate_up_alphas
                } else {
                    &self.gate_up_w4a16_unity_alphas
                };
            return moe_silu_quantize_slots_on_stream(
                &workspace.route.indices,
                gate_up_table,
                &mut workspace.sm12x_down.b_tiles,
                &mut workspace.sm12x_down.b_scales,
                &self.expert_ptrs.down_input_scales,
                gate_up_alpha_table,
                grouped_down.inputs[0].rows,
                workspace.sm12x_down.groups,
                stream,
            );
        }
        crate::nvfp4::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
            MoeSiluQuantizeSlotBuffers {
                indices: &workspace.route.indices,
                gate_up_table,
                packed_table: grouped_down.input_values_mut.output(),
                scales_table: grouped_down.input_scales_mut.output(),
                input_scale_table: &self.expert_ptrs.down_input_scales,
                gate_up_alpha_table: &self.expert_ptrs.gate_up_alphas,
            },
            grouped_down.inputs[0].rows,
            stream,
        )
    }

    /// Runs only the routed grouped down stage using already-quantized down inputs.
    pub fn run_grouped_down_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let grouped_down = workspace
            .grouped_down
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped down",
                detail: "grouped down workspace is unavailable".to_string(),
            })?;
        let enable_sm12x = self.storage_plan.down == Qwen36DownStorage::Sm12x;
        if enable_sm12x {
            let (Some(sm12x_down_tiles), Some(sm12x_down_scales)) =
                (&self.sm12x_down_tiles, &self.sm12x_down_scales)
            else {
                return Ok(());
            };
            indexed_grouped_gemv_on_stream(
                &workspace.route.indices,
                sm12x_down_tiles,
                sm12x_down_scales,
                self.num_experts,
                &workspace.sm12x_down.b_tiles,
                &workspace.sm12x_down.b_scales,
                &workspace.sm12x_down.d,
                self.sm12x_down_m_tiles,
                self.sm12x_down_k_tiles,
                workspace.sm12x_down.groups,
                stream,
            )?;
            return moe_weighted_accumulate_slots_f32_on_stream(
                &workspace.route.indices,
                &workspace.route.weights,
                &workspace.sm12x_down.c,
                &self.expert_ptrs.down_alphas,
                workspace.moe_out.inout(),
                stream,
            );
        }
        grouped_down.run_prequantized_device_route(
            &workspace.route,
            &self.expert_ptrs,
            &mut workspace.moe_out,
            stream,
        )?;
        Ok(())
    }

    pub fn run_w4a16_gate_up_slots_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        indices: &[usize],
        stream: &CudaStream,
    ) -> Result<()> {
        for &expert_idx in indices {
            if expert_idx >= self.num_experts {
                return Err(Error::Shape {
                    label: "Qwen3.6 MoE route index",
                    expected: format!("expert < {}", self.num_experts),
                    actual: expert_idx.to_string(),
                });
            }
            let expert = self.experts[expert_idx].get_gate_up_w4a16()?;
            expert.run_f32_into(ffn_norm, &mut workspace.fallback_gate_up_out, stream)?;
        }
        Ok(())
    }

    pub fn run_w4a16_down_slots_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        indices: &[usize],
        stream: &CudaStream,
    ) -> Result<()> {
        for &expert_idx in indices {
            if expert_idx >= self.num_experts {
                return Err(Error::Shape {
                    label: "Qwen3.6 MoE route index",
                    expected: format!("expert < {}", self.num_experts),
                    actual: expert_idx.to_string(),
                });
            }
            let expert = self.experts[expert_idx].get_down_w4a16()?;
            expert.run_f32_into(
                &workspace.fallback_down_input,
                &mut workspace.fallback_down_out,
                stream,
            )?;
        }
        Ok(())
    }

    pub fn run_w4a16_moe_slots_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        indices: &[usize],
        weights: &[f32],
        stream: &CudaStream,
    ) -> Result<()> {
        if indices.len() != weights.len() {
            return Err(Error::Shape {
                label: "Qwen3.6 MoE route buffers",
                expected: format!("matching route index/weight lengths, got {}", indices.len()),
                actual: weights.len().to_string(),
            });
        }
        fill_f32_into_on_stream(workspace.moe_out.output(), 0.0, stream)?;
        for (&expert_idx, &weight) in indices.iter().zip(weights.iter()) {
            if expert_idx >= self.num_experts {
                return Err(Error::Shape {
                    label: "Qwen3.6 MoE route index",
                    expected: format!("expert < {}", self.num_experts),
                    actual: expert_idx.to_string(),
                });
            }
            let gate_up = self.experts[expert_idx].get_gate_up_w4a16()?;
            gate_up.run_f32_into(ffn_norm, &mut workspace.fallback_gate_up_out, stream)?;
            silu_mul_halves_f32_into_on_stream(
                &workspace.fallback_gate_up_out,
                workspace.fallback_down_input.output(),
                self.expert_intermediate,
                stream,
            )?;
            let down = self.experts[expert_idx].get_down_w4a16()?;
            down.run_f32_into(
                &workspace.fallback_down_input,
                &mut workspace.fallback_down_out,
                stream,
            )?;
            scaled_add_f32_into_on_stream(
                &workspace.fallback_down_out,
                workspace.moe_out.inout(),
                weight,
                stream,
            )?;
        }
        Ok(())
    }

    /// Runs only the routed down pointer-table gather.
    pub fn run_grouped_down_gather_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let grouped_down = workspace
            .grouped_down
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped down",
                detail: "grouped down workspace is unavailable".to_string(),
            })?;
        gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
            GroupedGemvPointerTableBuffers {
                indices: &workspace.route.indices,
                a_values_table: &self.expert_ptrs.down_grouped_values,
                a_scales_table: &self.expert_ptrs.down_grouped_scales,
                b_values_table: &grouped_down.input_values,
                b_scales_table: &grouped_down.input_scales,
                c_table: grouped_down.gemv.c.inout(),
                d_table: grouped_down.gemv.d.inout(),
                out_a_values: grouped_down.gemv.a_values.output(),
                out_a_scales: grouped_down.gemv.a_scales.output(),
                out_b_values: grouped_down.gemv.b_values.output(),
                out_b_scales: grouped_down.gemv.b_scales.output(),
            },
            stream,
        )
    }

    /// Runs only the routed down grouped GEMV using prepared pointer tables.
    pub fn run_grouped_down_gemv_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let grouped_down = workspace
            .grouped_down
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped down",
                detail: "grouped down workspace is unavailable".to_string(),
            })?;
        let enable_sm12x = self.storage_plan.down == Qwen36DownStorage::Sm12x;
        if let (true, Some(sm12x_down_tiles), Some(sm12x_down_scales)) = (
            enable_sm12x,
            &self.sm12x_down_tiles,
            &self.sm12x_down_scales,
        ) {
            return indexed_grouped_gemv_on_stream(
                &workspace.route.indices,
                sm12x_down_tiles,
                sm12x_down_scales,
                self.num_experts,
                &workspace.sm12x_down.b_tiles,
                &workspace.sm12x_down.b_scales,
                &workspace.sm12x_down.d,
                self.sm12x_down_m_tiles,
                self.sm12x_down_k_tiles,
                workspace.sm12x_down.groups,
                stream,
            );
        }
        grouped_down.gemv.plan.run_on_stream(
            &grouped_down.gemv.a_values,
            &grouped_down.gemv.a_scales,
            &grouped_down.gemv.b_values,
            &grouped_down.gemv.b_scales,
            &grouped_down.gemv.c,
            &grouped_down.gemv.d,
            1.0,
            0.0,
            stream,
        )
    }

    /// Runs only the routed down weighted accumulation.
    pub fn run_grouped_down_accum_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let grouped_down = workspace
            .grouped_down
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 grouped down",
                detail: "grouped down workspace is unavailable".to_string(),
            })?;
        let enable_sm12x = self.storage_plan.down == Qwen36DownStorage::Sm12x;
        let inputs = if enable_sm12x
            && self.sm12x_down_tiles.is_some()
            && self.sm12x_down_scales.is_some()
        {
            &workspace.sm12x_down.c
        } else {
            &grouped_down.gemv.c
        };
        moe_weighted_accumulate_slots_f32_on_stream(
            &workspace.route.indices,
            &workspace.route.weights,
            inputs,
            &self.expert_ptrs.down_alphas,
            workspace.moe_out.inout(),
            stream,
        )
    }

    fn run_shared_gate_up(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.shared {
            Qwen36SharedExpertStorage::Nvfp4(shared) => {
                shared
                    .gate_up
                    .run_f32_into(ffn_norm, &mut workspace.shared_gate_up_output, stream)
            }
            Qwen36SharedExpertStorage::Fp8 { gate_up, .. } => gate_up.run_into(
                ffn_norm,
                &mut workspace.shared_gate_up_output,
                &mut workspace.fp8_shared_input,
                &mut workspace.fp8_shared_input_scale,
                stream,
            ),
        }
    }

    fn run_shared_down(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.shared {
            Qwen36SharedExpertStorage::Nvfp4(shared) => shared.down.run_f32_into(
                &workspace.shared_activated,
                &mut workspace.shared_output,
                stream,
            ),
            Qwen36SharedExpertStorage::Fp8 { down, .. } => down.run_into(
                &workspace.shared_activated,
                &mut workspace.shared_output,
                &mut workspace.fp8_shared_input,
                &mut workspace.fp8_shared_input_scale,
                stream,
            ),
        }
    }

    /// Runs only shared expert gate/up projection.
    pub fn run_shared_gate_up_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_shared_gate_up(workspace, ffn_norm, stream)
    }

    /// Runs only shared expert SiLU activation.
    pub fn run_shared_silu_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        silu_mul_halves_f32_into_on_stream(
            &workspace.shared_gate_up_output,
            workspace.shared_activated.output(),
            self.expert_intermediate,
            stream,
        )
    }

    /// Runs only shared expert down projection.
    pub fn run_shared_down_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_shared_down(workspace, stream)
    }

    /// Runs only shared expert gate projection and scaling.
    pub fn run_shared_gate_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.shared_gate
            .run_into(ffn_norm, &mut workspace.shared_gate_logits, stream)?;
        sigmoid_scale_scalar_f32_into_on_stream(
            &workspace.shared_gate_logits,
            &workspace.shared_output,
            workspace.shared_gated.output(),
            stream,
        )
    }

    /// Runs only the shared expert gate projection.
    pub fn run_shared_gate_linear_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.shared_gate
            .run_into(ffn_norm, &mut workspace.shared_gate_logits, stream)
    }

    /// Runs the fused routed accumulation, shared gate, residual, and BF16
    /// finalization used by the SM12x routed path.
    pub fn run_ffn_finalize_routed_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        qwen36_ffn_finalize_routed_f32_into_on_stream(
            &workspace.route.indices,
            &workspace.route.weights,
            &workspace.sm12x_down.c,
            &self.expert_ptrs.down_alphas,
            &workspace.shared_gate_logits,
            &workspace.shared_output,
            residual,
            workspace.ffn_residual.output(),
            stream,
        )
    }

    /// Runs only final FFN routed/shared combine and residual add.
    pub fn run_ffn_combine_only(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        add_f32_into_on_stream(
            &workspace.moe_out,
            &workspace.shared_gated,
            workspace.ffn_out.output(),
            stream,
        )?;
        add_f32_into_on_stream(
            residual,
            &workspace.ffn_out,
            workspace.ffn_residual.output(),
            stream,
        )
    }

    /// Runs one token through the MoE + shared-expert FFN.
    ///
    /// `ffn_norm` is the post-attention-norm hidden vector; `residual` is the
    /// pre-FFN residual (post-attention output). The output is written to
    /// `workspace.ffn_out` and equals the BF16-rounded value of
    /// `residual + (routed_moe + gated_shared)`.
    #[allow(clippy::needless_option_as_deref, clippy::too_many_arguments)]
    pub fn run_one_token<'a>(
        &'a self,
        _lt: &CublasLt,
        workspace: &'a mut Qwen36MoeWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
        mut profile: Option<&mut QwenDecodeProfile>,
        mut gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
    ) -> Result<Qwen36MoeStep<'a>> {
        if ffn_norm.len() != manifest.hidden || residual.len() != manifest.hidden {
            return Err(Error::Shape {
                label: "Qwen3.6 MoE FFN inputs",
                expected: format!("hidden={}", manifest.hidden),
                actual: format!("ffn_norm={} residual={}", ffn_norm.len(), residual.len()),
            });
        }

        // Router + topk — route stays device-resident, no host readback.
        if let Some(profile) = profile.as_deref_mut() {
            let (_, linear_ms) = timed_cuda(stream, || {
                self.router
                    .run_into(ffn_norm, &mut workspace.router_logits, stream)
            })?;
            profile.qwen36_router_linear_ms += linear_ms;
            profile.qwen36_router_ms += linear_ms;
        } else {
            self.router
                .run_into(ffn_norm, &mut workspace.router_logits, stream)?;
        }

        // Routed experts via device-resident grouped GEMV (no sync, no readback)
        // when supported; falls back to host-loop dispatch otherwise.
        let use_sm12x_down = self.storage_plan.down == Qwen36DownStorage::Sm12x
            && self.sm12x_down_tiles.is_some()
            && self.sm12x_down_scales.is_some();
        let use_device_route = workspace.grouped_down.is_some();
        let used_grouped = if let Some(fp8) = &self.fp8_experts {
            if let Some(profile) = profile.as_deref_mut() {
                let (_, topk_ms) = timed_cuda(stream, || {
                    workspace
                        .route
                        .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)
                })?;
                profile.qwen36_router_topk_ms += topk_ms;
                profile.qwen36_router_ms += topk_ms;
            } else {
                workspace
                    .route
                    .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)?;
            }
            let mut run_gate_up = || {
                quantize_fp8_e4m3_dynamic_f32_into_on_stream(
                    ffn_norm,
                    &mut workspace.fp8_hidden_input,
                    &mut workspace.fp8_hidden_input_scale,
                    stream,
                )?;
                fp8_moe_grouped_gate_up_f32_into_on_stream(
                    &workspace.route.indices,
                    &workspace.fp8_hidden_input,
                    &workspace.fp8_hidden_input_scale,
                    &fp8.gate.weights,
                    &fp8.gate.scales,
                    &fp8.up.weights,
                    &fp8.up.scales,
                    workspace.marlin_gate_up_output.output(),
                    self.expert_intermediate,
                    manifest.hidden,
                    self.experts_per_token,
                    stream,
                )
            };
            if let Some(profile) = profile.as_deref_mut() {
                let (_, ms) = timed_cuda(stream, run_gate_up)?;
                profile.qwen36_routed_gate_up_ms += ms;
            } else {
                run_gate_up()?;
            }
            let mut run_silu_quantize = || {
                moe_silu_quantize_fp8_slots_f32_into_on_stream(
                    &workspace.marlin_gate_up_output,
                    &mut workspace.fp8_down_input,
                    &mut workspace.fp8_down_input_scales,
                    self.expert_intermediate,
                    self.experts_per_token,
                    stream,
                )
            };
            if let Some(profile) = profile.as_deref_mut() {
                let (_, ms) = timed_cuda(stream, run_silu_quantize)?;
                profile.qwen36_routed_silu_quantize_ms += ms;
            } else {
                run_silu_quantize()?;
            }
            if !use_sm12x_down {
                fill_f32_into_on_stream(workspace.moe_out.output(), 0.0, stream)?;
            }
            let sm12x_down = &workspace.sm12x_down;
            if let Some(profile) = profile.as_deref_mut() {
                let (_, gemv_ms) = timed_cuda(stream, || {
                    fp8_moe_grouped_down_f32_into_on_stream(
                        &workspace.route.indices,
                        &workspace.fp8_down_input,
                        &workspace.fp8_down_input_scales,
                        &fp8.down.weights,
                        &fp8.down.scales,
                        &sm12x_down.d,
                        manifest.hidden,
                        self.expert_intermediate,
                        self.experts_per_token,
                        stream,
                    )
                })?;
                profile.qwen36_routed_down_gemv_ms += gemv_ms;
                let (_, accum_ms) = timed_cuda(stream, || {
                    moe_weighted_accumulate_slots_f32_on_stream(
                        &workspace.route.indices,
                        &workspace.route.weights,
                        &sm12x_down.c,
                        &self.expert_ptrs.down_alphas,
                        workspace.moe_out.inout(),
                        stream,
                    )
                })?;
                profile.qwen36_routed_down_accum_ms += accum_ms;
                profile.qwen36_routed_down_ms += gemv_ms + accum_ms;
            } else {
                fp8_moe_grouped_down_f32_into_on_stream(
                    &workspace.route.indices,
                    &workspace.fp8_down_input,
                    &workspace.fp8_down_input_scales,
                    &fp8.down.weights,
                    &fp8.down.scales,
                    &sm12x_down.d,
                    manifest.hidden,
                    self.expert_intermediate,
                    self.experts_per_token,
                    stream,
                )?;
                moe_weighted_accumulate_slots_f32_on_stream(
                    &workspace.route.indices,
                    &workspace.route.weights,
                    &sm12x_down.c,
                    &self.expert_ptrs.down_alphas,
                    workspace.moe_out.inout(),
                    stream,
                )?;
            }
            true
        } else if use_device_route {
            let grouped_down = workspace
                .grouped_down
                .as_mut()
                .expect("device route requires grouped down workspace");
            if let Some(profile) = profile.as_deref_mut() {
                let (_, topk_ms) = timed_cuda(stream, || {
                    workspace
                        .route
                        .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)
                })?;
                profile.qwen36_router_topk_ms += topk_ms;
                profile.qwen36_router_ms += topk_ms;
            } else {
                workspace
                    .route
                    .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)?;
            }

            let use_grouped_w4a4 =
                matches!(self.gate_up_storage, Qwen36GateUpStorage::Grouped { .. });
            if use_grouped_w4a4 {
                let input_scale = self
                    .expert_ptrs
                    .shared_gate_up_input_scale
                    .expect("grouped W4A4 storage requires a shared input scale");
                quantize_nvfp4_vector_simple_scales_f32_into_on_stream(
                    manifest.hidden,
                    ffn_norm,
                    &mut workspace.gate_up_input,
                    &mut workspace.gate_up_input_simple_scales,
                    input_scale,
                    stream,
                )?;
                let grouped_gate_up = workspace
                    .grouped_gate_up
                    .as_mut()
                    .expect("device route requires grouped gate/up workspace");
                let mut run_grouped = || {
                    grouped_gate_up.run_gate_up_device_route(
                        &workspace.route,
                        &self.expert_ptrs,
                        &workspace.gate_up_input,
                        workspace.gate_up_input_simple_scales.as_const_ptr().cast(),
                        stream,
                    )?;
                    Ok(())
                };
                if gpu_probe
                    .as_ref()
                    .is_some_and(|probe| probe.should_capture(Qwen36GpuCounterStage::RoutedGateUp))
                {
                    gpu_probe
                        .as_deref_mut()
                        .expect("probe present")
                        .capture(run_grouped)?;
                } else if let Some(profile) = profile.as_deref_mut() {
                    let (_, ms) = timed_cuda(stream, run_grouped)?;
                    profile.qwen36_routed_gate_up_ms += ms;
                } else {
                    run_grouped()?;
                }
            } else {
                let mut run_marlin = || {
                    let Qwen36GateUpStorage::Marlin(marlin) = &self.gate_up_storage else {
                        unreachable!("checked Marlin gate/up storage")
                    };
                    if use_sm12x_down {
                        marlin
                            .run_bf16_on_stream(&workspace.route.indices, ffn_norm, stream)
                            .map(|_| ())
                    } else {
                        marlin.run_on_stream(
                            &workspace.route.indices,
                            ffn_norm,
                            workspace.marlin_gate_up_output.output(),
                            stream,
                        )
                    }
                };
                if gpu_probe
                    .as_ref()
                    .is_some_and(|probe| probe.should_capture(Qwen36GpuCounterStage::RoutedGateUp))
                {
                    gpu_probe
                        .as_deref_mut()
                        .expect("probe present")
                        .capture(run_marlin)?;
                } else if let Some(profile) = profile.as_deref_mut() {
                    let (_, ms) = timed_cuda(stream, run_marlin)?;
                    profile.qwen36_routed_gate_up_ms += ms;
                } else {
                    run_marlin()?;
                }
            }
            let gate_up_table = if use_grouped_w4a4 {
                &workspace
                    .grouped_gate_up
                    .as_ref()
                    .expect("grouped W4A4 workspace")
                    .c
            } else {
                &workspace.marlin_gate_up_table
            };
            let gate_up_alpha_table = if use_grouped_w4a4 {
                &self.expert_ptrs.gate_up_alphas
            } else {
                &self.gate_up_w4a16_unity_alphas
            };
            if use_sm12x_down {
                let sm12x_down = &mut workspace.sm12x_down;
                let mut run_silu_quantize = || {
                    if use_grouped_w4a4 {
                        moe_silu_quantize_slots_on_stream(
                            &workspace.route.indices,
                            gate_up_table,
                            &mut sm12x_down.b_tiles,
                            &mut sm12x_down.b_scales,
                            &self.expert_ptrs.down_input_scales,
                            gate_up_alpha_table,
                            grouped_down.inputs[0].rows,
                            sm12x_down.groups,
                            stream,
                        )
                    } else {
                        let Qwen36GateUpStorage::Marlin(marlin) = &self.gate_up_storage else {
                            unreachable!("checked Marlin gate/up storage")
                        };
                        moe_silu_quantize_bf16_slots_on_stream(
                            &workspace.route.indices,
                            marlin.output_bf16(),
                            &mut sm12x_down.b_tiles,
                            &mut sm12x_down.b_scales,
                            &self.expert_ptrs.down_input_scales,
                            gate_up_alpha_table,
                            grouped_down.inputs[0].rows,
                            sm12x_down.groups,
                            stream,
                        )
                    }
                };
                if let Some(profile) = profile.as_deref_mut() {
                    let (_, ms) = timed_cuda(stream, run_silu_quantize)?;
                    profile.qwen36_routed_silu_quantize_ms += ms;
                } else {
                    run_silu_quantize()?;
                }
            } else if let Some(profile) = profile.as_deref_mut() {
                let (_, ms) = timed_cuda(stream, || {
                    crate::nvfp4::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
                        MoeSiluQuantizeSlotBuffers {
                            indices: &workspace.route.indices,
                            gate_up_table,
                            packed_table: grouped_down.input_values_mut.output(),
                            scales_table: grouped_down.input_scales_mut.output(),
                            input_scale_table: &self.expert_ptrs.down_input_scales,
                            gate_up_alpha_table,
                        },
                        grouped_down.inputs[0].rows,
                        stream,
                    )
                })?;
                profile.qwen36_routed_silu_quantize_ms += ms;
            } else {
                crate::nvfp4::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
                    MoeSiluQuantizeSlotBuffers {
                        indices: &workspace.route.indices,
                        gate_up_table,
                        packed_table: grouped_down.input_values_mut.output(),
                        scales_table: grouped_down.input_scales_mut.output(),
                        input_scale_table: &self.expert_ptrs.down_input_scales,
                        gate_up_alpha_table,
                    },
                    grouped_down.inputs[0].rows,
                    stream,
                )?;
            }

            // Down grouped GEMV + weighted accumulate into moe_out.
            fill_f32_into_on_stream(workspace.moe_out.output(), 0.0, stream)?;
            if use_sm12x_down {
                let (sm12x_down_tiles, sm12x_down_scales) = (
                    self.sm12x_down_tiles
                        .as_ref()
                        .expect("SM12x down tiles are required"),
                    self.sm12x_down_scales
                        .as_ref()
                        .expect("SM12x down scales are required"),
                );
                let sm12x_down = &mut workspace.sm12x_down;
                if let Some(profile) = profile.as_deref_mut() {
                    let (_, gemv_ms) = timed_cuda(stream, || {
                        indexed_grouped_gemv_on_stream(
                            &workspace.route.indices,
                            sm12x_down_tiles,
                            sm12x_down_scales,
                            self.num_experts,
                            &sm12x_down.b_tiles,
                            &sm12x_down.b_scales,
                            &sm12x_down.d,
                            self.sm12x_down_m_tiles,
                            self.sm12x_down_k_tiles,
                            sm12x_down.groups,
                            stream,
                        )
                    })?;
                    profile.qwen36_routed_down_gemv_ms += gemv_ms;
                    profile.qwen36_routed_down_ms += gemv_ms;
                } else {
                    indexed_grouped_gemv_on_stream(
                        &workspace.route.indices,
                        sm12x_down_tiles,
                        sm12x_down_scales,
                        self.num_experts,
                        &sm12x_down.b_tiles,
                        &sm12x_down.b_scales,
                        &sm12x_down.d,
                        self.sm12x_down_m_tiles,
                        self.sm12x_down_k_tiles,
                        sm12x_down.groups,
                        stream,
                    )?;
                }
            } else if let Some(profile) = profile.as_deref_mut() {
                let (_, gather_ms) = timed_cuda(stream, || {
                    gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
                        GroupedGemvPointerTableBuffers {
                            indices: &workspace.route.indices,
                            a_values_table: &self.expert_ptrs.down_grouped_values,
                            a_scales_table: &self.expert_ptrs.down_grouped_scales,
                            b_values_table: &grouped_down.input_values,
                            b_scales_table: &grouped_down.input_scales,
                            c_table: grouped_down.gemv.c.inout(),
                            d_table: grouped_down.gemv.d.inout(),
                            out_a_values: grouped_down.gemv.a_values.output(),
                            out_a_scales: grouped_down.gemv.a_scales.output(),
                            out_b_values: grouped_down.gemv.b_values.output(),
                            out_b_scales: grouped_down.gemv.b_scales.output(),
                        },
                        stream,
                    )
                })?;
                profile.qwen36_routed_down_gather_ms += gather_ms;
                let (_, gemv_ms) = timed_cuda(stream, || {
                    grouped_down.gemv.plan.run_on_stream(
                        &grouped_down.gemv.a_values,
                        &grouped_down.gemv.a_scales,
                        &grouped_down.gemv.b_values,
                        &grouped_down.gemv.b_scales,
                        &grouped_down.gemv.c,
                        &grouped_down.gemv.d,
                        1.0,
                        0.0,
                        stream,
                    )
                })?;
                profile.qwen36_routed_down_gemv_ms += gemv_ms;
                let (_, accum_ms) = timed_cuda(stream, || {
                    moe_weighted_accumulate_slots_f32_on_stream(
                        &workspace.route.indices,
                        &workspace.route.weights,
                        &grouped_down.gemv.c,
                        &self.expert_ptrs.down_alphas,
                        workspace.moe_out.inout(),
                        stream,
                    )
                })?;
                profile.qwen36_routed_down_accum_ms += accum_ms;
                profile.qwen36_routed_down_ms += gather_ms + gemv_ms + accum_ms;
            } else {
                grouped_down.run_prequantized_device_route(
                    &workspace.route,
                    &self.expert_ptrs,
                    &mut workspace.moe_out,
                    stream,
                )?;
            }
            true
        } else {
            // Fallback: host-loop expert dispatch with host readback.
            if let Some(profile) = profile.as_deref_mut() {
                let (_, topk_ms) = timed_cuda(stream, || {
                    workspace
                        .route
                        .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)
                })?;
                profile.qwen36_router_topk_ms += topk_ms;
                profile.qwen36_router_ms += topk_ms;
            } else {
                workspace
                    .route
                    .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)?;
            }
            let indices = workspace.route.indices.copy_to_host(stream)?;
            let weights = workspace.route.weights.copy_to_host(stream)?;
            let use_sm12x_native = std::env::var_os("QWEN36_SM12X_NATIVE_MOE").is_some();
            let ffn_norm_host = if use_sm12x_native {
                Some(ffn_norm.copy_to_host(stream)?.into_vec())
            } else {
                None
            };

            fill_f32_into_on_stream(workspace.moe_out.output(), 0.0, stream)?;
            for slot in 0..self.experts_per_token {
                let expert_idx = indices[slot] as usize;
                let weight = weights[slot];
                if expert_idx >= self.num_experts {
                    return Err(Error::Shape {
                        label: "Qwen3.6 MoE route index",
                        expected: format!("expert < {}", self.num_experts),
                        actual: expert_idx.to_string(),
                    });
                }
                let lazy_expert = self
                    .experts
                    .get(expert_idx)
                    .expect("expert index validated");
                let down_input = &mut workspace.fallback_down_input;

                if let Some(ffn_norm_host) = &ffn_norm_host {
                    let native = lazy_expert.get_gate_up_sm12x()?;
                    native.run_host_vector_into(
                        ffn_norm_host,
                        &mut workspace.fallback_gate_up_out,
                        stream,
                    )?;
                } else {
                    let expert = lazy_expert.get_gate_up_w4a16()?;
                    crate::nvfp4::nvfp4_w4a16_matvec_f32_into_on_stream(
                        ffn_norm,
                        &expert.packed_weight,
                        &expert.weight_scale,
                        workspace.fallback_gate_up_out.output(),
                        expert.out_features,
                        expert.in_features,
                        expert.weight_scale_2,
                        stream,
                    )?;
                }
                silu_mul_halves_f32_into_on_stream(
                    &workspace.fallback_gate_up_out,
                    down_input.output(),
                    self.expert_intermediate,
                    stream,
                )?;
                if use_sm12x_native {
                    let down_input_host = down_input.copy_to_host(stream)?;
                    let native = lazy_expert.get_down_sm12x()?;
                    native.run_host_vector_into(
                        &down_input_host,
                        &mut workspace.fallback_down_out,
                        stream,
                    )?;
                } else {
                    let expert = lazy_expert.get_down_w4a16()?;
                    crate::nvfp4::nvfp4_w4a16_matvec_f32_into_on_stream(
                        down_input,
                        &expert.packed_weight,
                        &expert.weight_scale,
                        workspace.fallback_down_out.output(),
                        expert.out_features,
                        expert.in_features,
                        expert.weight_scale_2,
                        stream,
                    )?;
                }
                scaled_add_f32_into_on_stream(
                    &workspace.fallback_down_out,
                    workspace.moe_out.inout(),
                    weight,
                    stream,
                )?;
            }
            false
        };
        let _ = used_grouped;

        // Shared experts follow the layer's checkpoint format: NVFP4 uses the
        // established W4A16 path, while mixed layers use dynamic W8A8.
        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                self.run_shared_gate_up(workspace, ffn_norm, stream)
            })?;
            profile.qwen36_shared_gate_up_ms += ms;
        } else {
            self.run_shared_gate_up(workspace, ffn_norm, stream)?;
        }
        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                silu_mul_halves_f32_into_on_stream(
                    &workspace.shared_gate_up_output,
                    workspace.shared_activated.output(),
                    self.expert_intermediate,
                    stream,
                )
            })?;
            profile.qwen36_shared_silu_ms += ms;
        } else {
            silu_mul_halves_f32_into_on_stream(
                &workspace.shared_gate_up_output,
                workspace.shared_activated.output(),
                self.expert_intermediate,
                stream,
            )?;
        }
        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || self.run_shared_down(workspace, stream))?;
            profile.qwen36_shared_down_ms += ms;
        } else {
            self.run_shared_down(workspace, stream)?;
        }

        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                self.shared_gate
                    .run_into(ffn_norm, &mut workspace.shared_gate_logits, stream)
            })?;
            profile.qwen36_shared_gate_ms += ms;
        } else {
            self.shared_gate
                .run_into(ffn_norm, &mut workspace.shared_gate_logits, stream)?;
        }

        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                if use_device_route && use_sm12x_down {
                    qwen36_ffn_finalize_routed_f32_into_on_stream(
                        &workspace.route.indices,
                        &workspace.route.weights,
                        &workspace.sm12x_down.c,
                        &self.expert_ptrs.down_alphas,
                        &workspace.shared_gate_logits,
                        &workspace.shared_output,
                        residual,
                        workspace.ffn_residual.output(),
                        stream,
                    )
                } else {
                    qwen36_ffn_finalize_f32_into_on_stream(
                        &workspace.moe_out,
                        &workspace.shared_gate_logits,
                        &workspace.shared_output,
                        residual,
                        workspace.ffn_residual.output(),
                        stream,
                    )
                }
            })?;
            profile.qwen36_ffn_combine_ms += ms;
        } else if use_device_route && use_sm12x_down {
            qwen36_ffn_finalize_routed_f32_into_on_stream(
                &workspace.route.indices,
                &workspace.route.weights,
                &workspace.sm12x_down.c,
                &self.expert_ptrs.down_alphas,
                &workspace.shared_gate_logits,
                &workspace.shared_output,
                residual,
                workspace.ffn_residual.output(),
                stream,
            )?;
        } else {
            qwen36_ffn_finalize_f32_into_on_stream(
                &workspace.moe_out,
                &workspace.shared_gate_logits,
                &workspace.shared_output,
                residual,
                workspace.ffn_residual.output(),
                stream,
            )?;
        }
        std::mem::swap(&mut workspace.ffn_out, &mut workspace.ffn_residual);

        Ok(Qwen36MoeStep {
            route_indices: &workspace.route.indices,
            route_weights: &workspace.route.weights,
            ffn_out: &workspace.ffn_out,
        })
    }
}

impl LazyQwen36Expert {
    fn get_gate_up_w4a16(&self) -> Result<std::cell::Ref<'_, Nvfp4DeviceLinear>> {
        if self.gate_up_w4a16.borrow().is_none() {
            let gate = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.gate_proj", self.prefix))?;
            let up = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.up_proj", self.prefix))?;
            let gate_up = ModelOptNvfp4Linear::concat_out_features(
                format!("{}.gate_up_proj", self.prefix),
                &gate,
                &up,
            )?;
            *self.gate_up_w4a16.borrow_mut() = Some(Nvfp4DeviceLinear::from_host(&gate_up)?);
            crate::nvfp4::synchronize_device()?;
        }
        Ok(std::cell::Ref::map(self.gate_up_w4a16.borrow(), |weight| {
            weight.as_ref().expect("Qwen3.6 gate/up loaded")
        }))
    }

    fn get_down_w4a16(&self) -> Result<std::cell::Ref<'_, Nvfp4DeviceLinear>> {
        if self.down_w4a16.borrow().is_none() {
            let down = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.down_proj", self.prefix))?;
            *self.down_w4a16.borrow_mut() = Some(Nvfp4DeviceLinear::from_host(&down)?);
            crate::nvfp4::synchronize_device()?;
        }
        Ok(std::cell::Ref::map(self.down_w4a16.borrow(), |weight| {
            weight.as_ref().expect("Qwen3.6 down loaded")
        }))
    }

    fn get_gate_up_sm12x(&self) -> Result<std::cell::Ref<'_, Sm12xDeviceLinear>> {
        if self.gate_up_sm12x.borrow().is_none() {
            let gate = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.gate_proj", self.prefix))?;
            let up = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.up_proj", self.prefix))?;
            let gate_up = ModelOptNvfp4Linear::concat_out_features(
                format!("{}.gate_up_proj", self.prefix),
                &gate,
                &up,
            )?;
            *self.gate_up_sm12x.borrow_mut() = Some(Sm12xDeviceLinear::from_host(&gate_up)?);
            crate::nvfp4::synchronize_device()?;
        }
        Ok(std::cell::Ref::map(self.gate_up_sm12x.borrow(), |weight| {
            weight.as_ref().expect("Qwen3.6 SM12x gate/up loaded")
        }))
    }

    fn get_down_sm12x(&self) -> Result<std::cell::Ref<'_, Sm12xDeviceLinear>> {
        if self.down_sm12x.borrow().is_none() {
            let down = self
                .checkpoint
                .load_nvfp4_linear(&format!("{}.down_proj", self.prefix))?;
            *self.down_sm12x.borrow_mut() = Some(Sm12xDeviceLinear::from_host(&down)?);
            crate::nvfp4::synchronize_device()?;
        }
        Ok(std::cell::Ref::map(self.down_sm12x.borrow(), |weight| {
            weight.as_ref().expect("Qwen3.6 SM12x down loaded")
        }))
    }
}

impl Nvfp4DeviceLinear {
    fn from_host(host: &ModelOptNvfp4Linear) -> Result<Self> {
        Ok(Self {
            packed_weight: DeviceBuffer::from_host(&host.packed_weight)?,
            weight_scale: DeviceBuffer::from_host(&host.weight_scale)?,
            weight_scale_2: host.weight_scale_2,
            input_scale: host.input_scale,
            out_features: host.out_features,
            in_features: host.in_features,
        })
    }

    fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let host = checkpoint.load_nvfp4_linear(prefix)?;
        Self::from_host(&host)
    }

    /// W4A16 matvec: f32 input × dequantized NVFP4 weight → f32 output.
    fn run_f32_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_matvec_f32_into_on_stream(
            input,
            &self.packed_weight,
            &self.weight_scale,
            output.output(),
            self.out_features,
            self.in_features,
            self.weight_scale_2,
            stream,
        )?;
        maybe_round_device_f32_to_bf16(output, stream)
    }
}

impl Sm12xDeviceLinear {
    fn from_host(host: &ModelOptNvfp4Linear) -> Result<Self> {
        let dequant_col_major = host.dequantize_to_f32_col_major();
        let mut row_major = vec![0.0f32; host.out_features * host.in_features];
        for row in 0..host.out_features {
            for col in 0..host.in_features {
                row_major[row * host.in_features + col] =
                    dequant_col_major[col + row * host.in_features];
            }
        }
        let quantized = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
            host.out_features,
            host.in_features,
            &row_major,
        )?;
        Ok(Self {
            weight: quantized.weight.to_device()?,
            weight_scale_2: host.weight_scale_2,
            out_features: host.out_features,
            in_features: host.in_features,
        })
    }

    fn run_host_vector_into(
        &self,
        input_host: &[f32],
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if input_host.len() != self.in_features || output.len() != self.out_features {
            return Err(Error::Shape {
                label: "Qwen3.6 SM12x linear",
                expected: format!("input={} output={}", self.in_features, self.out_features),
                actual: format!("input={} output={}", input_host.len(), output.len()),
            });
        }
        let vector = Sm12xFp4GemmVector::quantize_f32_k16(self.in_features, input_host)?;
        let vector = vector.vector.to_device()?;
        device_weight_gemv_on_stream(&self.weight, &vector, output.output(), stream)?;
        let mut host = output.copy_to_host(stream)?.into_vec();
        for value in &mut host {
            *value *= self.weight_scale_2;
        }
        output.copy_from_host(&host)
    }
}

fn maybe_round_device_f32_to_bf16(
    output: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if std::env::var_os("QWEN36_ROUND_LINEAR_OUTPUTS_BF16").is_none() {
        return Ok(());
    }
    round_f32_to_bf16_in_place_on_stream(output.inout(), stream)
}

impl Qwen36Fp8ExpertTable {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        experts: usize,
        rows: usize,
        cols: usize,
        prefix: impl Fn(usize) -> String,
    ) -> Result<Self> {
        let matrix_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
            label: "Qwen3.6 FP8 expert table",
            expected: "rows * cols fits usize".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
        let mut host_weights = Vec::with_capacity(experts * matrix_len);
        let mut host_scales = Vec::with_capacity(experts * rows);
        for expert in 0..experts {
            let weight = checkpoint.load_fp8_linear(&prefix(expert))?;
            let scales = weight.channel_weight_scale.ok_or_else(|| Error::Format {
                label: "Qwen3.6 FP8 expert table",
                detail: format!("expert {expert} lacks per-channel weight scales"),
            })?;
            if weight.out_features != rows
                || weight.in_features != cols
                || weight.weight.len() != matrix_len
                || scales.len() != rows
                || weight.input_scale.is_some()
            {
                return Err(Error::Shape {
                    label: "Qwen3.6 FP8 expert table",
                    expected: format!(
                        "{rows}x{cols} channel-scaled weight with dynamic input activation"
                    ),
                    actual: format!(
                        "expert={expert} shape={}x{} weight={} scales={} input_scale={:?}",
                        weight.out_features,
                        weight.in_features,
                        weight.weight.len(),
                        scales.len(),
                        weight.input_scale
                    ),
                });
            }
            host_weights.extend_from_slice(&weight.weight);
            host_scales.extend_from_slice(&scales);
        }
        let weights = DeviceBuffer::from_host(&host_weights)?;
        let scales = DeviceBuffer::from_host(&host_scales)?;
        let weight_base = weights.as_const_ptr().cast::<u8>();
        let scale_base = scales.as_const_ptr().cast::<f32>();
        let weight_ptrs = (0..experts)
            .map(|expert| unsafe { weight_base.add(expert * matrix_len) })
            .collect::<Vec<_>>();
        let scale_ptrs = (0..experts)
            .map(|expert| unsafe { scale_base.add(expert * rows) })
            .collect::<Vec<_>>();
        Ok(Self {
            _weights: weights,
            _scales: scales,
            weights: DeviceBuffer::from_host(&weight_ptrs)?,
            scales: DeviceBuffer::from_host(&scale_ptrs)?,
        })
    }
}

impl Qwen36MoeWorkspace {
    /// Allocates one-token workspace for the Qwen3.6 MoE + shared-expert FFN.
    pub fn new(manifest: &QwenModelManifest) -> Result<Self> {
        let enable_grouped = true;
        let enable_sm12x_down = true;
        Self::new_for_paths(manifest, enable_grouped, enable_sm12x_down)
    }

    fn new_for_paths(
        manifest: &QwenModelManifest,
        enable_grouped: bool,
        enable_sm12x_down: bool,
    ) -> Result<Self> {
        let (experts, experts_per_token, expert_intermediate) = match manifest.ffn {
            QwenFfnConfig::Moe {
                experts,
                experts_per_token,
                expert_intermediate,
                ..
            } => (experts, experts_per_token, expert_intermediate),
            QwenFfnConfig::Dense => {
                return Err(Error::Format {
                    label: "Qwen3.6 MoE workspace",
                    detail: "manifest is not MoE".to_string(),
                });
            }
        };
        let shared_intermediate =
            manifest
                .shared_expert_intermediate
                .ok_or_else(|| Error::Format {
                    label: "Qwen3.6 MoE workspace",
                    detail: "manifest missing shared_expert_intermediate".to_string(),
                })?;
        if shared_intermediate != expert_intermediate {
            return Err(Error::Shape {
                label: "Qwen3.6 MoE workspace shared intermediate",
                expected: format!("shared_intermediate={expert_intermediate}"),
                actual: format!("shared_intermediate={shared_intermediate}"),
            });
        }
        let hidden = manifest.hidden;
        let gate_up_out_features = expert_intermediate * 2;
        let grouped_gate_up = if enable_grouped {
            GroupedGemvWorkspace::new(gate_up_out_features, hidden, experts_per_token)?
        } else {
            None
        };
        let grouped_down = if enable_grouped || enable_sm12x_down {
            MoeGroupedDownWorkspace::new(hidden, expert_intermediate, experts_per_token)?
        } else {
            None
        };
        let marlin_gate_up_output = DeviceBuffer::zeroed(experts_per_token * gate_up_out_features)?;
        let marlin_base = marlin_gate_up_output.as_const_ptr().cast::<f32>();
        let marlin_gate_up_ptrs = (0..experts_per_token)
            .map(|slot| unsafe { marlin_base.add(slot * gate_up_out_features) })
            .collect::<Vec<_>>();
        Ok(Self {
            router_logits: DeviceBuffer::zeroed(experts)?,
            route: MoeRouteWorkspace::new(experts_per_token)?,
            gate_up_input: Nvfp4Matrix::zeroed_col_major(hidden, 1)?,
            gate_up_input_simple_scales: DeviceBuffer::zeroed(hidden.div_ceil(16))?,
            grouped_gate_up,
            marlin_gate_up_output,
            marlin_gate_up_table: DeviceBuffer::from_host(&marlin_gate_up_ptrs)?,
            fp8_hidden_input: DeviceBuffer::zeroed(hidden)?,
            fp8_hidden_input_scale: DeviceBuffer::zeroed(1)?,
            fp8_down_input: DeviceBuffer::zeroed(experts_per_token * expert_intermediate)?,
            fp8_down_input_scales: DeviceBuffer::zeroed(experts_per_token)?,
            fp8_shared_input: DeviceBuffer::zeroed(hidden.max(expert_intermediate))?,
            fp8_shared_input_scale: DeviceBuffer::zeroed(1)?,
            sm12x_down: Sm12xGateUpWorkspace::new(
                hidden,
                expert_intermediate,
                experts_per_token,
                experts_per_token,
            )?,
            grouped_down,
            fallback_gate_up_out: DeviceBuffer::zeroed(gate_up_out_features)?,
            fallback_down_input: DeviceBuffer::zeroed(expert_intermediate)?,
            fallback_down_out: DeviceBuffer::zeroed(hidden)?,
            shared_gate_up_output: DeviceBuffer::zeroed(gate_up_out_features)?,
            shared_activated: DeviceBuffer::zeroed(expert_intermediate)?,
            shared_output: DeviceBuffer::zeroed(hidden)?,
            shared_gate_logits: DeviceBuffer::zeroed(1)?,
            shared_gated: DeviceBuffer::zeroed(hidden)?,
            moe_out: DeviceBuffer::zeroed(hidden)?,
            ffn_out: DeviceBuffer::zeroed(hidden)?,
            ffn_residual: DeviceBuffer::zeroed(hidden)?,
        })
    }
}

fn load_concat_gate_up(
    checkpoint: &ModelOptCheckpoint,
    gate_prefix: &str,
    up_prefix: &str,
    label: &'static str,
) -> Result<ModelOptNvfp4Linear> {
    let gate = checkpoint.load_nvfp4_linear(gate_prefix)?;
    let up = checkpoint.load_nvfp4_linear(up_prefix)?;
    if gate.in_features != up.in_features {
        return Err(Error::Shape {
            label,
            expected: format!("matching in_features={}", gate.in_features),
            actual: format!("gate={} up={}", gate.in_features, up.in_features),
        });
    }
    ModelOptNvfp4Linear::concat_out_features(format!("{gate_prefix}.gate_up_proj"), &gate, &up)
}

fn concat_fp8_out_features(
    first: ModelOptFp8Linear,
    second: ModelOptFp8Linear,
    label: &'static str,
) -> Result<ModelOptFp8Linear> {
    if first.in_features != second.in_features
        || first.input_scale.map(f32::to_bits) != second.input_scale.map(f32::to_bits)
    {
        return Err(Error::Shape {
            label,
            expected: "matching input shape and activation scale".to_string(),
            actual: format!(
                "first={}x{} input_scale={:?} second={}x{} input_scale={:?}",
                first.out_features,
                first.in_features,
                first.input_scale,
                second.out_features,
                second.in_features,
                second.input_scale
            ),
        });
    }
    let first_scales = first.channel_weight_scale.ok_or_else(|| Error::Format {
        label,
        detail: "first projection lacks per-channel scales".to_string(),
    })?;
    let second_scales = second.channel_weight_scale.ok_or_else(|| Error::Format {
        label,
        detail: "second projection lacks per-channel scales".to_string(),
    })?;
    let mut weight = first.weight;
    weight.extend_from_slice(&second.weight);
    let mut channel_weight_scale = first_scales;
    channel_weight_scale.extend_from_slice(&second_scales);
    Ok(ModelOptFp8Linear {
        prefix: format!("{}+{}", first.prefix, second.prefix),
        out_features: first.out_features + second.out_features,
        in_features: first.in_features,
        weight,
        weight_scale: 1.0,
        channel_weight_scale: Some(channel_weight_scale),
        input_scale: first.input_scale,
    })
}

// ---------------------------------------------------------------------------
// Layer block: attention + MoE + norms + residuals
// ---------------------------------------------------------------------------

/// Device-ready weights for one Qwen3.6 text layer block.
///
/// A block owns its input/post-attention RMSNorm weights, the scheduled
/// attention weights (linear or full), and the shared MoE FFN.
pub struct Qwen36LayerBlock {
    pub layer: usize,
    pub kind: QwenLayerKind,
    pub input_norm: DeviceBuffer<f32>,
    pub post_attn_norm: DeviceBuffer<f32>,
    pub attention: Qwen36Attention,
    pub moe: Qwen36MoeWeights,
}

/// Attention variant held by a layer block.
pub enum Qwen36Attention {
    LinearAttention(Qwen36LinearAttentionWeights),
    FullAttention(Qwen36FullAttentionWeights),
}

/// Mutable one-token workspace for one Qwen3.6 text layer block.
pub struct Qwen36LayerBlockWorkspace {
    pub kind: QwenLayerKind,
    pub normed_hidden: DeviceBuffer<f32>,
    pub attn_residual: DeviceBuffer<f32>,
    pub ffn_norm: DeviceBuffer<f32>,
    pub attention: Qwen36AttentionWorkspace,
    pub moe: Qwen36MoeWorkspace,
}

/// Attention workspace variant held by a layer block.
pub enum Qwen36AttentionWorkspace {
    LinearAttention(Qwen36LinearAttentionWorkspace),
    FullAttention(Qwen36FullAttentionWorkspace),
}

/// Borrowed outputs from one layer-block step.
pub struct Qwen36LayerBlockStep<'a> {
    /// Final block output (already includes the second residual add).
    pub output: &'a DeviceBuffer<f32>,
}

impl Qwen36LayerBlock {
    /// Loads the full layer block (norms + attention + MoE) for `layer`.
    pub fn load(model: &Qwen36Model, layer: usize) -> Result<Self> {
        let fp8 = Rc::new(Qwen36LinearFp8Execution::new(
            model.checkpoint(),
            model.manifest(),
        )?);
        Self::load_inner(model, layer, false, fp8)
    }

    fn load_from_prepared_cache(
        model: &Qwen36Model,
        layer: usize,
        fp8: Rc<Qwen36LinearFp8Execution>,
    ) -> Result<Self> {
        Self::load_inner(model, layer, true, fp8)
    }

    fn load_inner(
        model: &Qwen36Model,
        layer: usize,
        cache_prepared: bool,
        fp8: Rc<Qwen36LinearFp8Execution>,
    ) -> Result<Self> {
        let kind = model.layer_kind(layer)?;
        let input_norm = model.load_input_norm(layer)?;
        let post_attn_norm = model.load_post_attn_norm(layer)?;
        let attention = match kind {
            QwenLayerKind::LinearAttention => {
                Qwen36Attention::LinearAttention(Qwen36LinearAttentionWeights::load_with_fp8(
                    &model.checkpoint,
                    &model.manifest,
                    layer,
                    fp8,
                )?)
            }
            QwenLayerKind::FullAttention => Qwen36Attention::FullAttention(
                Qwen36FullAttentionWeights::load(&model.checkpoint, &model.manifest, layer)?,
            ),
        };
        let moe = if cache_prepared {
            model.load_moe_from_prepared_cache(layer)?
        } else {
            model.load_moe(layer)?
        };
        Ok(Self {
            layer,
            kind,
            input_norm,
            post_attn_norm,
            attention,
            moe,
        })
    }

    /// Allocates workspace for this layer block.
    ///
    /// `cache_capacity` is the KV-cache capacity for full-attention layers;
    /// linear-attention layers ignore it (they carry conv/GDN state instead).
    pub fn workspace(
        &self,
        model: &Qwen36Model,
        cache_capacity: usize,
    ) -> Result<Qwen36LayerBlockWorkspace> {
        let manifest = model.manifest();
        let attention = match &self.attention {
            Qwen36Attention::LinearAttention(weights) => Qwen36AttentionWorkspace::LinearAttention(
                model.linear_attention_workspace(weights)?,
            ),
            Qwen36Attention::FullAttention(weights) => Qwen36AttentionWorkspace::FullAttention(
                model.full_attention_workspace(weights, cache_capacity)?,
            ),
        };
        Ok(Qwen36LayerBlockWorkspace {
            kind: self.kind,
            normed_hidden: DeviceBuffer::zeroed(manifest.hidden)?,
            attn_residual: DeviceBuffer::zeroed(manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(manifest.hidden)?,
            attention,
            moe: self.moe.workspace(manifest)?,
        })
    }

    fn enqueue_linear_pre_gdn(
        &self,
        workspace: &mut Qwen36LayerBlockWorkspace,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            hidden,
            &self.input_norm,
            workspace.normed_hidden.output(),
            manifest.rms_eps,
            stream,
        )?;
        match (&self.attention, &mut workspace.attention) {
            (
                Qwen36Attention::LinearAttention(weights),
                Qwen36AttentionWorkspace::LinearAttention(attention_workspace),
            ) => weights.enqueue_pre_gdn(attention_workspace, &workspace.normed_hidden, stream),
            _ => Err(Error::Format {
                label: "Qwen3.6 segmented graph",
                detail: "pre-GDN segment requires a linear-attention layer".to_string(),
            }),
        }
    }

    fn enqueue_linear_gdn(
        &self,
        workspace: &mut Qwen36LayerBlockWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        match (&self.attention, &mut workspace.attention) {
            (
                Qwen36Attention::LinearAttention(weights),
                Qwen36AttentionWorkspace::LinearAttention(attention_workspace),
            ) => weights.enqueue_gdn(attention_workspace, stream),
            _ => Err(Error::Format {
                label: "Qwen3.6 segmented graph",
                detail: "direct GDN update requires a linear-attention layer".to_string(),
            }),
        }
    }

    fn enqueue_linear_post_gdn(
        &self,
        lt: &CublasLt,
        workspace: &mut Qwen36LayerBlockWorkspace,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let attn_output = match (&self.attention, &mut workspace.attention) {
            (
                Qwen36Attention::LinearAttention(weights),
                Qwen36AttentionWorkspace::LinearAttention(attention_workspace),
            ) => {
                weights.enqueue_post_gdn(attention_workspace, manifest.rms_eps, stream)?;
                &attention_workspace.output
            }
            _ => {
                return Err(Error::Format {
                    label: "Qwen3.6 segmented graph",
                    detail: "post-GDN segment requires a linear-attention layer".to_string(),
                });
            }
        };
        add_f32_into_on_stream(
            hidden,
            attn_output,
            workspace.attn_residual.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            &workspace.attn_residual,
            &self.post_attn_norm,
            workspace.ffn_norm.output(),
            manifest.rms_eps,
            stream,
        )?;
        self.moe.run_one_token(
            lt,
            &mut workspace.moe,
            manifest,
            &workspace.ffn_norm,
            &workspace.attn_residual,
            stream,
            None,
            None,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_full_layer_indexed(
        &self,
        lt: &CublasLt,
        workspace: &mut Qwen36LayerBlockWorkspace,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: &DeviceBuffer<u32>,
        cache_len: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            hidden,
            &self.input_norm,
            workspace.normed_hidden.output(),
            manifest.rms_eps,
            stream,
        )?;
        let attn_output = match (&self.attention, &mut workspace.attention) {
            (
                Qwen36Attention::FullAttention(weights),
                Qwen36AttentionWorkspace::FullAttention(attention_workspace),
            ) => {
                weights.run_one_token_indexed(
                    attention_workspace,
                    manifest,
                    &workspace.normed_hidden,
                    position,
                    cache_len,
                    stream,
                )?;
                &attention_workspace.output
            }
            _ => {
                return Err(Error::Format {
                    label: "Qwen3.6 segmented graph",
                    detail: "indexed full-layer segment requires a full-attention layer"
                        .to_string(),
                });
            }
        };
        add_f32_into_on_stream(
            hidden,
            attn_output,
            workspace.attn_residual.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            &workspace.attn_residual,
            &self.post_attn_norm,
            workspace.ffn_norm.output(),
            manifest.rms_eps,
            stream,
        )?;
        self.moe.run_one_token(
            lt,
            &mut workspace.moe,
            manifest,
            &workspace.ffn_norm,
            &workspace.attn_residual,
            stream,
            None,
            None,
        )?;
        Ok(())
    }

    /// Runs one token through the full layer block.
    ///
    /// `hidden` is the input hidden vector; the block writes its output into
    /// `workspace.ffn_norm`-adjacent storage and returns a borrow of the
    /// final buffer (the MoE `ffn_out`, which already includes the residual).
    #[allow(clippy::needless_option_as_deref, clippy::too_many_arguments)]
    pub fn run_one_token<'a>(
        &'a self,
        lt: &CublasLt,
        workspace: &'a mut Qwen36LayerBlockWorkspace,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: usize,
        stream: &CudaStream,
        mut profile: Option<&mut QwenDecodeProfile>,
        mut gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
    ) -> Result<Qwen36LayerBlockStep<'a>> {
        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                rms_norm_f32_into_on_stream(
                    1,
                    manifest.hidden,
                    hidden,
                    &self.input_norm,
                    workspace.normed_hidden.output(),
                    manifest.rms_eps,
                    stream,
                )
            })?;
            profile.input_norm_ms += ms;
        } else {
            rms_norm_f32_into_on_stream(
                1,
                manifest.hidden,
                hidden,
                &self.input_norm,
                workspace.normed_hidden.output(),
                manifest.rms_eps,
                stream,
            )?;
        }

        let attn_output: &DeviceBuffer<f32> = if let Some(profile) = profile.as_deref_mut() {
            let (output, ms) = timed_cuda(stream, || {
                run_qwen36_attention(
                    &self.attention,
                    &mut workspace.attention,
                    manifest,
                    &workspace.normed_hidden,
                    position,
                    stream,
                    Some(&mut *profile),
                )
            })?;
            profile.attention_ms += ms;
            match self.attention {
                Qwen36Attention::LinearAttention(_) => profile.qwen36_linear_attention_ms += ms,
                Qwen36Attention::FullAttention(_) => profile.qwen36_full_attention_ms += ms,
            }
            output
        } else {
            run_qwen36_attention(
                &self.attention,
                &mut workspace.attention,
                manifest,
                &workspace.normed_hidden,
                position,
                stream,
                None,
            )?
        };

        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                add_f32_into_on_stream(
                    hidden,
                    attn_output,
                    workspace.attn_residual.output(),
                    stream,
                )
            })?;
            profile.attn_residual_ms += ms;
        } else {
            add_f32_into_on_stream(
                hidden,
                attn_output,
                workspace.attn_residual.output(),
                stream,
            )?;
        }

        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                rms_norm_f32_into_on_stream(
                    1,
                    manifest.hidden,
                    &workspace.attn_residual,
                    &self.post_attn_norm,
                    workspace.ffn_norm.output(),
                    manifest.rms_eps,
                    stream,
                )
            })?;
            profile.ffn_norm_ms += ms;
        } else {
            rms_norm_f32_into_on_stream(
                1,
                manifest.hidden,
                &workspace.attn_residual,
                &self.post_attn_norm,
                workspace.ffn_norm.output(),
                manifest.rms_eps,
                stream,
            )?;
        }

        let moe_step = if let Some(profile) = profile.as_deref_mut() {
            let wall_start = Instant::now();
            let (step, ms) = timed_cuda(stream, || {
                self.moe.run_one_token(
                    lt,
                    &mut workspace.moe,
                    manifest,
                    &workspace.ffn_norm,
                    &workspace.attn_residual,
                    stream,
                    Some(&mut *profile),
                    gpu_probe.as_deref_mut(),
                )
            })?;
            profile.ffn_gemm_ms += ms;
            profile.ffn_wall_ms += wall_start.elapsed().as_secs_f64() * 1_000.0;
            step
        } else {
            self.moe.run_one_token(
                lt,
                &mut workspace.moe,
                manifest,
                &workspace.ffn_norm,
                &workspace.attn_residual,
                stream,
                None,
                gpu_probe.as_deref_mut(),
            )?
        };
        Ok(Qwen36LayerBlockStep {
            output: moe_step.ffn_out,
        })
    }
}

fn run_qwen36_attention<'a>(
    attention: &'a Qwen36Attention,
    workspace: &'a mut Qwen36AttentionWorkspace,
    manifest: &QwenModelManifest,
    normed_hidden: &DeviceBuffer<f32>,
    position: usize,
    stream: &CudaStream,
    profile: Option<&mut QwenDecodeProfile>,
) -> Result<&'a DeviceBuffer<f32>> {
    match (attention, workspace) {
        (Qwen36Attention::LinearAttention(w), Qwen36AttentionWorkspace::LinearAttention(ws)) => {
            let step = w.run_one_token(ws, normed_hidden, manifest.rms_eps, stream, profile)?;
            Ok(step.output)
        }
        (Qwen36Attention::FullAttention(w), Qwen36AttentionWorkspace::FullAttention(ws)) => {
            let step = w.run_one_token(ws, manifest, normed_hidden, position, stream)?;
            Ok(step.output)
        }
        _ => Err(Error::Format {
            label: "Qwen3.6 layer block",
            detail: "attention weight/workspace variant mismatch".to_string(),
        }),
    }
}

fn timed_cuda<T>(stream: &CudaStream, f: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record_on_stream(stream)?;
    let value = f()?;
    end.record_on_stream(stream)?;
    end.synchronize()?;
    Ok((value, start.elapsed_ms_until(&end)? as f64))
}

// ---------------------------------------------------------------------------
// Full text model: embedding + 40 layer blocks + final norm + lm_head
// ---------------------------------------------------------------------------

/// Fully loaded Qwen3.6 text model ready for one-token-at-a-time decode.
///
/// Holds all layer block weights, the BF16 embedding table, the final RMSNorm
/// weight, and the quantized lm_head. Routed-expert NVFP4 weights are loaded
/// lazily on first use.
pub struct Qwen36TextModel {
    manifest: QwenModelManifest,
    checkpoint: ModelOptCheckpoint,
    lt: CublasLt,
    layers: Vec<Qwen36LayerBlock>,
    embedding: DeviceBuffer<u16>,
    final_norm: DeviceBuffer<f32>,
    lm_head: Qwen36LmHead,
}

enum Qwen36LmHead {
    Nvfp4(Nvfp4DeviceLinear),
    Fp8 {
        linear: Fp8Linear,
        plan: Box<Fp8TnMatmulPlan>,
    },
}

impl Qwen36LmHead {
    fn load(checkpoint: &ModelOptCheckpoint, lt: &CublasLt) -> Result<Self> {
        if checkpoint.contains_tensor("lm_head.weight_scale_2") {
            Ok(Self::Nvfp4(Nvfp4DeviceLinear::load(checkpoint, "lm_head")?))
        } else {
            let linear = Fp8Linear::from_host(&checkpoint.load_fp8_linear("lm_head")?)?;
            let plan =
                Fp8TnMatmulPlan::new(lt, GemmShape::new(linear.rows, 1, linear.cols), 8 << 20)?;
            Ok(Self::Fp8 {
                linear,
                plan: Box::new(plan),
            })
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Nvfp4(linear) => (linear.out_features, linear.in_features),
            Self::Fp8 { linear, .. } => (linear.rows, linear.cols),
        }
    }

    fn run_logits(
        &self,
        lt: &CublasLt,
        input: &DeviceBuffer<f32>,
        workspace: &mut Qwen36LmHeadWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Nvfp4(linear) => linear.run_f32_into(input, &mut workspace.logits, stream),
            Self::Fp8 { linear, plan } => {
                Self::run_fp8_logits(lt, linear, plan, input, workspace, stream)
            }
        }
    }

    fn run_top1(
        &self,
        lt: &CublasLt,
        input: &DeviceBuffer<f32>,
        workspace: &mut Qwen36LmHeadWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Nvfp4(linear) => nvfp4_w4a16_top1_f32_into_on_stream(
                input,
                &linear.packed_weight,
                &linear.weight_scale,
                &workspace.scratch_value,
                &workspace.scratch_index,
                &workspace.next_index,
                &workspace.next_value,
                linear.out_features,
                linear.in_features,
                linear.weight_scale_2,
                stream,
            ),
            Self::Fp8 { linear, plan } => {
                Self::run_fp8_logits(lt, linear, plan, input, workspace, stream)?;
                argmax_f32_into_on_stream(
                    &workspace.logits,
                    workspace.next_index.output(),
                    workspace.next_value.output(),
                    stream,
                )
            }
        }
    }

    fn run_fp8_logits(
        lt: &CublasLt,
        linear: &Fp8Linear,
        plan: &Fp8TnMatmulPlan,
        input: &DeviceBuffer<f32>,
        workspace: &mut Qwen36LmHeadWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(channel_scale) = linear.channel_weight_scale.as_ref() else {
            return linear.run_into(
                input,
                &mut workspace.logits,
                &mut workspace.dynamic_input,
                &mut workspace.dynamic_input_scale,
                stream,
            );
        };
        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            input,
            &mut workspace.dynamic_input,
            &mut workspace.dynamic_input_scale,
            stream,
        )?;
        plan.run_with_alpha_on_stream(
            lt,
            &linear.weight,
            &workspace.dynamic_input,
            workspace.logits.output(),
            1.0,
            stream,
        )?;
        scale_channel_f32_device_scalar_in_place_on_stream(
            workspace.logits.inout(),
            channel_scale,
            &workspace.dynamic_input_scale,
            stream,
        )
    }
}

struct Qwen36LmHeadWorkspace {
    logits: DeviceBuffer<f32>,
    dynamic_input: DeviceBuffer<u8>,
    dynamic_input_scale: DeviceBuffer<f32>,
    scratch_value: DeviceBuffer<f32>,
    scratch_index: DeviceBuffer<u32>,
    next_index: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
}

struct Qwen36LinearLayerGraphs {
    pre_gdn: CudaGraphExec,
    post_gdn: CudaGraphExec,
}

enum Qwen36LayerGraphs {
    Linear(Qwen36LinearLayerGraphs),
    Full(CudaGraphExec),
}

/// Mutable decode state for [`Qwen36TextModel`].
pub struct Qwen36DecodeState {
    stream: CudaStream,
    token_id_device: DeviceBuffer<u32>,
    position_device: DeviceBuffer<u32>,
    cache_len_device: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layer_workspaces: Vec<Qwen36LayerBlockWorkspace>,
    final_hidden: DeviceBuffer<f32>,
    lm_head: Qwen36LmHeadWorkspace,
    segmented_graphs: Option<Vec<Qwen36LayerGraphs>>,
    position: usize,
    max_tokens: usize,
}

/// One decoded next-token result.
pub struct Qwen36NextToken {
    /// Argmax token id.
    pub id: u32,
    /// Winning logit value.
    pub value: f32,
}

/// CPU-visible lm-head logits produced by one Qwen3.6 decode step.
pub struct Qwen36NextTokenLogits {
    /// One logit for every vocabulary entry.
    pub logits: Vec<f32>,
}

/// Qwen3.6 decode stage that can be wrapped by GPU counter collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36GpuCounterStage {
    /// Routed expert grouped gate/up stage.
    RoutedGateUp,
}

/// One-shot GPU counter probe for a Qwen3.6 decode stage.
pub struct Qwen36GpuCounterProbe<'a> {
    collector: &'a mut GpuCounterCollector,
    stage: Qwen36GpuCounterStage,
    captured: bool,
    done: bool,
}

impl<'a> Qwen36GpuCounterProbe<'a> {
    /// Creates a one-shot probe around `stage` using `collector`.
    pub fn new(collector: &'a mut GpuCounterCollector, stage: Qwen36GpuCounterStage) -> Self {
        Self {
            collector,
            stage,
            captured: false,
            done: false,
        }
    }

    /// Returns true when this pass captured the requested stage.
    pub fn captured(&self) -> bool {
        self.captured
    }

    /// Returns true when all replay passes have been submitted.
    pub fn done(&self) -> bool {
        self.done
    }

    fn should_capture(&self, stage: Qwen36GpuCounterStage) -> bool {
        !self.captured && self.stage == stage
    }

    fn capture<T>(&mut self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.collector.begin()?;
        let value = f()?;
        self.done = self.collector.end()?;
        self.captured = true;
        Ok(value)
    }
}

impl Qwen36TextModel {
    /// Loads the full Qwen3.6 text model from `model_dir`.
    pub fn open(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let model = Qwen36Model::open(model_dir)?;
        Self::from_qwen36_model(model)
    }

    /// Builds the full text model from an already-opened [`Qwen36Model`].
    pub fn from_qwen36_model(model: Qwen36Model) -> Result<Self> {
        let manifest = model.manifest().clone();
        let checkpoint = model.checkpoint().clone();
        ensure_model_cache(&checkpoint, &manifest)?;
        let lt = CublasLt::new()?;
        let linear_fp8 = Rc::new(Qwen36LinearFp8Execution::new(&checkpoint, &manifest)?);
        let mut layers = Vec::with_capacity(manifest.layers);
        for layer in 0..manifest.layers {
            layers.push(Qwen36LayerBlock::load_from_prepared_cache(
                &model,
                layer,
                Rc::clone(&linear_fp8),
            )?);
        }
        let embedding = read_bf16_matrix_device(
            &checkpoint,
            &format!("{}.embed_tokens.weight", manifest.tensor_prefix),
            manifest.vocab,
            manifest.hidden,
        )?;
        let final_norm = read_bf16_vector_delta_as_f32_device(
            &checkpoint,
            &format!("{}.norm.weight", manifest.tensor_prefix),
            manifest.hidden,
        )?;
        let lm_head = Qwen36LmHead::load(&checkpoint, &lt)?;
        let lm_head_shape = lm_head.shape();
        if lm_head_shape != (manifest.vocab, manifest.hidden) {
            return Err(Error::Shape {
                label: "Qwen3.6 lm_head",
                expected: format!("[{}, {}]", manifest.vocab, manifest.hidden),
                actual: format!("[{}, {}]", lm_head_shape.0, lm_head_shape.1),
            });
        }
        Ok(Self {
            manifest,
            checkpoint,
            lt,
            layers,
            embedding,
            final_norm,
            lm_head,
        })
    }

    /// Returns the parsed manifest.
    pub fn manifest(&self) -> &QwenModelManifest {
        &self.manifest
    }

    /// Returns the BF16 embedding table `[vocab, hidden]`.
    pub fn embedding(&self) -> &DeviceBuffer<u16> {
        &self.embedding
    }

    /// Allocates a decode state capable of storing `max_tokens` positions.
    pub fn new_decode_state(&self, max_tokens: usize) -> Result<Qwen36DecodeState> {
        if max_tokens == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 decode state",
                expected: "max_tokens > 0".to_string(),
                actual: "0".to_string(),
            });
        }
        let stream = CudaStream::new_blocking()?;
        let mut layer_workspaces = Vec::with_capacity(self.layers.len());
        let model = Qwen36Model {
            manifest: self.manifest.clone(),
            checkpoint: self.checkpoint.clone(),
        };
        for block in &self.layers {
            layer_workspaces.push(block.workspace(&model, max_tokens)?);
        }
        let mut state = Qwen36DecodeState {
            stream,
            token_id_device: DeviceBuffer::zeroed(1)?,
            position_device: DeviceBuffer::zeroed(1)?,
            cache_len_device: DeviceBuffer::zeroed(1)?,
            hidden: DeviceBuffer::zeroed(self.manifest.hidden)?,
            layer_workspaces,
            final_hidden: DeviceBuffer::zeroed(self.manifest.hidden)?,
            lm_head: Qwen36LmHeadWorkspace {
                logits: DeviceBuffer::zeroed(self.manifest.vocab)?,
                dynamic_input: DeviceBuffer::zeroed(self.manifest.hidden)?,
                dynamic_input_scale: DeviceBuffer::zeroed(1)?,
                scratch_value: DeviceBuffer::zeroed(self.manifest.vocab.div_ceil(8))?,
                scratch_index: DeviceBuffer::zeroed(self.manifest.vocab.div_ceil(8))?,
                next_index: DeviceBuffer::zeroed(1)?,
                next_value: DeviceBuffer::zeroed(1)?,
            },
            segmented_graphs: None,
            position: 0,
            max_tokens,
        };
        let enable_segmented_graphs = !std::env::var("EIDER_DISABLE_DECODE_GRAPHS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
        if enable_segmented_graphs {
            state.segmented_graphs = Some(self.capture_segmented_graphs(&mut state)?);
        }
        Ok(state)
    }

    fn capture_segmented_graphs(
        &self,
        state: &mut Qwen36DecodeState,
    ) -> Result<Vec<Qwen36LayerGraphs>> {
        let mut graphs = Vec::with_capacity(self.layers.len());
        for (layer_idx, block) in self.layers.iter().enumerate() {
            let (previous, current) = state.layer_workspaces.split_at_mut(layer_idx);
            let hidden = if layer_idx == 0 {
                &state.hidden
            } else {
                &previous[layer_idx - 1].moe.ffn_out
            };
            let workspace = &mut current[0];
            match &block.attention {
                Qwen36Attention::LinearAttention(_) => {
                    let pre_gdn = state.stream.capture(|stream| {
                        block.enqueue_linear_pre_gdn(workspace, &self.manifest, hidden, stream)
                    })?;
                    let post_gdn = state.stream.capture(|stream| {
                        block.enqueue_linear_post_gdn(
                            &self.lt,
                            workspace,
                            &self.manifest,
                            hidden,
                            stream,
                        )
                    })?;
                    graphs.push(Qwen36LayerGraphs::Linear(Qwen36LinearLayerGraphs {
                        pre_gdn,
                        post_gdn,
                    }));
                }
                Qwen36Attention::FullAttention(_) => {
                    let graph = state.stream.capture(|stream| {
                        block.enqueue_full_layer_indexed(
                            &self.lt,
                            workspace,
                            &self.manifest,
                            hidden,
                            &state.position_device,
                            &state.cache_len_device,
                            stream,
                        )
                    })?;
                    graphs.push(Qwen36LayerGraphs::Full(graph));
                }
            }
        }
        Ok(graphs)
    }

    /// Decodes one token: embedding lookup -> 40 layer blocks -> final norm ->
    /// lm_head -> argmax. Advances the decode position by one.
    pub fn decode_one_token(
        &self,
        state: &mut Qwen36DecodeState,
        token_id: u32,
    ) -> Result<Qwen36NextToken> {
        self.decode_one_token_impl(state, token_id, None, None, false)
            .map(|(next, _)| next)
    }

    /// Decodes one token and returns the complete lm-head logits on the CPU.
    ///
    /// This is intended for sampling and is slower than the GPU top-1 path
    /// because it copies one `vocab`-sized vector to the host per token.
    pub fn decode_one_token_logits(
        &self,
        state: &mut Qwen36DecodeState,
        token_id: u32,
    ) -> Result<Qwen36NextTokenLogits> {
        let (_, logits) = self.decode_one_token_impl(state, token_id, None, None, true)?;
        Ok(Qwen36NextTokenLogits {
            logits: logits.expect("full-logit decode requested"),
        })
    }

    /// Decodes one token and accumulates coarse CUDA-event timings.
    pub fn decode_one_token_profiled(
        &self,
        state: &mut Qwen36DecodeState,
        token_id: u32,
        profile: &mut QwenDecodeProfile,
    ) -> Result<Qwen36NextToken> {
        self.decode_one_token_impl(state, token_id, Some(profile), None, false)
            .map(|(next, _)| next)
    }

    /// Decodes one token while wrapping one selected stage in a GPU counter range.
    pub fn decode_one_token_with_gpu_counter_probe(
        &self,
        state: &mut Qwen36DecodeState,
        token_id: u32,
        probe: &mut Qwen36GpuCounterProbe<'_>,
    ) -> Result<Qwen36NextToken> {
        self.decode_one_token_impl(state, token_id, None, Some(probe), false)
            .map(|(next, _)| next)
    }

    #[allow(clippy::needless_option_as_deref)]
    fn decode_one_token_impl(
        &self,
        state: &mut Qwen36DecodeState,
        token_id: u32,
        mut profile: Option<&mut QwenDecodeProfile>,
        mut gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
        return_logits: bool,
    ) -> Result<(Qwen36NextToken, Option<Vec<f32>>)> {
        if state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "Qwen3.6 decode position",
                expected: format!("position < {}", state.max_tokens),
                actual: state.position.to_string(),
            });
        }
        if (token_id as usize) >= self.manifest.vocab {
            return Err(Error::Shape {
                label: "Qwen3.6 token id",
                expected: format!("token < {}", self.manifest.vocab),
                actual: token_id.to_string(),
            });
        }
        state.token_id_device.copy_from_host(&[token_id])?;
        let stream = &state.stream;
        if let Some(profile) = profile.as_deref_mut() {
            profile.tokens += 1;
            let (_, ms) = timed_cuda(stream, || {
                copy_bf16_row_to_f32_indexed_into_on_stream(
                    self.manifest.vocab,
                    self.manifest.hidden,
                    &self.embedding,
                    &state.token_id_device,
                    state.hidden.output(),
                    stream,
                )
            })?;
            profile.embedding_ms += ms;
        } else {
            copy_bf16_row_to_f32_indexed_into_on_stream(
                self.manifest.vocab,
                self.manifest.hidden,
                &self.embedding,
                &state.token_id_device,
                state.hidden.output(),
                stream,
            )?;
        }

        let use_segmented_graphs =
            profile.is_none() && gpu_probe.is_none() && state.segmented_graphs.is_some();
        if !use_segmented_graphs {
            state.segmented_graphs = None;
        }

        if let Some(graphs) = state.segmented_graphs.as_ref() {
            state
                .position_device
                .copy_from_host(&[state.position as u32])?;
            state
                .cache_len_device
                .copy_from_host(&[(state.position + 1) as u32])?;
            for ((block, workspace), graph) in self
                .layers
                .iter()
                .zip(state.layer_workspaces.iter_mut())
                .zip(graphs.iter())
            {
                match graph {
                    Qwen36LayerGraphs::Linear(graph) => {
                        graph.pre_gdn.launch(stream)?;
                        block.enqueue_linear_gdn(workspace, stream)?;
                        graph.post_gdn.launch(stream)?;
                    }
                    Qwen36LayerGraphs::Full(graph) => graph.launch(stream)?,
                }
            }
        } else {
            for (layer_idx, block) in self.layers.iter().enumerate() {
                let (previous, current) = state.layer_workspaces.split_at_mut(layer_idx);
                let hidden = if layer_idx == 0 {
                    &state.hidden
                } else {
                    &previous[layer_idx - 1].moe.ffn_out
                };
                block.run_one_token(
                    &self.lt,
                    &mut current[0],
                    &self.manifest,
                    hidden,
                    state.position,
                    stream,
                    profile.as_deref_mut(),
                    gpu_probe.as_deref_mut(),
                )?;
            }
        }
        let hidden = &state
            .layer_workspaces
            .last()
            .expect("Qwen3.6 has at least one layer")
            .moe
            .ffn_out;

        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                rms_norm_f32_into_on_stream(
                    1,
                    self.manifest.hidden,
                    hidden,
                    &self.final_norm,
                    state.final_hidden.output(),
                    self.manifest.rms_eps,
                    stream,
                )
            })?;
            profile.final_norm_ms += ms;
        } else {
            rms_norm_f32_into_on_stream(
                1,
                self.manifest.hidden,
                hidden,
                &self.final_norm,
                state.final_hidden.output(),
                self.manifest.rms_eps,
                stream,
            )?;
        }

        round_f32_to_bf16_in_place_on_stream(state.final_hidden.inout(), stream)?;

        let (id, value, logits) = if return_logits {
            self.lm_head
                .run_logits(&self.lt, &state.final_hidden, &mut state.lm_head, stream)?;
            let logits = state.lm_head.logits.copy_to_host(stream)?.into_vec();
            let (id, value) = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(id, value)| (id as u32, value))
                .expect("Qwen3.6 vocabulary is non-empty");
            (id, value, Some(logits))
        } else if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                self.lm_head
                    .run_top1(&self.lt, &state.final_hidden, &mut state.lm_head, stream)
            })?;
            profile.lm_head_argmax_ms += ms;
            let id = state.lm_head.next_index.copy_to_host(stream)?[0];
            let value = state.lm_head.next_value.copy_to_host(stream)?[0];
            (id, value, None)
        } else {
            self.lm_head
                .run_top1(&self.lt, &state.final_hidden, &mut state.lm_head, stream)?;
            let id = state.lm_head.next_index.copy_to_host(stream)?[0];
            let value = state.lm_head.next_value.copy_to_host(stream)?[0];
            (id, value, None)
        };
        state.position += 1;
        Ok((Qwen36NextToken { id, value }, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::reorder_fp8_v_rows;
    use crate::nvfp4::ModelOptFp8Linear;

    #[test]
    fn reorder_fp8_v_rows_keeps_channel_scales_with_weights() {
        let host = ModelOptFp8Linear {
            prefix: "z".to_string(),
            out_features: 8,
            in_features: 1,
            weight: (0..8).collect(),
            weight_scale: 1.0,
            channel_weight_scale: Some((100..108).map(|value| value as f32).collect()),
            input_scale: None,
        };

        let reordered = reorder_fp8_v_rows(host, 2, 4, 2);
        assert_eq!(reordered.weight, vec![0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(
            reordered.channel_weight_scale,
            Some(vec![100.0, 101.0, 104.0, 105.0, 102.0, 103.0, 106.0, 107.0])
        );
    }
}
