//! Qwen3.6 / Qwen3.5-MoE hybrid execution pieces.

mod batch;
mod dflash2;
mod execution;
mod mtp;
mod sequence;

pub(crate) use batch::{
    BatchFullAttentionWorkspace, Qwen36BatchModelView, Qwen36HybridPrefillWorkspace,
};
pub use batch::{
    Qwen36DecodeBatchTrace, Qwen36DecodeBatchWorkspace, Qwen36DecodeLayerTrace, Qwen36DecodeRow,
    Qwen36DecodedBatch, Qwen36PrefillBatchWorkspace, Qwen36PrefillRow,
    Qwen36SpeculativeCycleOutcome, Qwen36SpeculativeCycleWorkspace, Qwen36SpeculativeFrontier,
};
pub use dflash2::{DFlash2Config, inspect_dflash2_config, validate_dflash2_checkpoint};
pub(crate) use dflash2::{
    Qwen38DFlash2PrefixCache, Qwen38DFlash2SequenceState, Qwen38DFlash2Workspace,
};
#[cfg(test)]
pub(crate) use execution::decode_capacity_classes;
pub(crate) use execution::{Qwen36ExecutionConfig, Qwen36ExecutionState, Qwen36SequenceId};
pub use mtp::{Qwen36MtpDraftWorkspace, Qwen36MtpSequenceState, Qwen36MtpWeights};
pub use sequence::{Qwen36Sequence, Qwen36SequenceCache, new_qwen36_sequence_cache};

pub(crate) use sequence::{Qwen36Append, qwen36_cache_error};

use crate::metrics::ExpertPagingMetricHandle;
use eider_cuda::{
    CublasLt, CudaEvent, CudaGraphExec, CudaStream, DeviceAddress, DeviceBuffer, Error, F32Matrix,
    Fp8TnMatmulPlan, GemmShape, GpuCounterCollector, GroupedGemvPointerTableBuffers,
    ModelOptCublasLtWeight, MoeSiluQuantizeSlotBuffers, MropeSections, Nvfp4Matrix,
    PinnedHostBuffer, Result, Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmVector, Sm12xFp4GemmWeight,
    Sm12xKvAttentionWorkspace, Sm12xKvCache, Sm12xKvPagePool, Sm121W4A16GateUp,
    Sm121W4A16HostWeight, add_f32_into_on_stream, argmax_f32_batch_into_on_stream,
    argmax_f32_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, bf16_linear_pair_logits_f32_into_on_stream,
    bf16_linear_two_rows_f32_into_on_stream, copy_bf16_rows_to_f32_indexed_prefix_into_on_stream,
    copy_fp8_rows_to_f32_indexed_prefix_into_on_stream, device_weight_gemv_on_stream,
    fill_f32_into_on_stream, fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream,
    fp8_linear_channel_scaled_f32_into_on_stream, fp8_linear_configured_f32_into_on_stream,
    fp8_linear_f32_into_on_stream, fp8_linear_pair_configured_f32_into_on_stream,
    fp8_linear_triple_configured_f32_into_on_stream, fp8_linear_w8a8_f32_into_on_stream,
    fp8_moe_grouped_down_addressed_f32_into_on_stream,
    fp8_moe_grouped_gate_up_addressed_f32_into_on_stream, gated_delta_net_128_f32_into_on_stream,
    gated_rms_norm_f32_into_on_stream, gather_nvfp4_grouped_gemv_ptr_tables_on_stream,
    indexed_grouped_gemv_addresses_on_stream as indexed_grouped_gemv_on_stream,
    ling3_sigmoid_gated_rms_norm_f32_into_on_stream, lm_head_top1_f32_batch_into_on_stream,
    moe_silu_quantize_fp8_slots_f32_into_on_stream, moe_silu_quantize_slots_on_stream,
    moe_topk_f32_batch_into_on_stream, moe_weighted_accumulate_slots_f32_on_stream,
    nvfp4_w4a16_matvec_f32_into_on_stream, nvfp4_w4a16_top1_f32_into_on_stream,
    quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream, qwen36_ffn_finalize_f32_into_on_stream,
    qwen36_ffn_finalize_routed_f32_into_on_stream, qwen36_full_attn_prep_f32_into_on_stream,
    qwen36_gdn_gate_into_on_stream, qwen36_gdn_prep_into_on_stream, rms_norm_f32_into_on_stream,
    rope_imrope_f32_indexed_into_on_stream, rope_imrope_f32_into_on_stream,
    rope_neox_partial_f32_indexed_into_on_stream, rope_neox_partial_f32_into_on_stream,
    round_f32_to_bf16_in_place_on_stream, scale_channel_f32_device_scalar_in_place_on_stream,
    scaled_add_f32_into_on_stream, sigmoid_mul_f32_into_on_stream,
    sigmoid_scale_scalar_f32_into_on_stream, silu_mul_halves_f32_into_on_stream,
};
use eider_format::SafeTensorInfo;
use eider_format::{ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear};

use super::infer::{
    GroupedGemvWorkspace, MoeExpertPointerTables, MoeGroupedDownWorkspace, MoeRouteWorkspace,
    QwenArchitecture, QwenDecodeProfile, QwenFfnConfig, QwenLayerKind, QwenLinearAttentionConfig,
    QwenModelManifest,
};
use super::qwen36_cache::{
    Qwen36Fp8Nvfp4Cache, down_path, ensure_layer_cache, ensure_model_cache, gate_up_path,
    prepared_layer_dir,
};
use crate::runtime::expert_cache::{
    ExpertRecordSource, ExpertSlotCache, ExpertUploadCoordinator, read_expert_misses,
};

use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static NEXT_QWEN36_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// Loader scaffold for the Qwen3.6/Qwen3.5-MoE hybrid text stack.
pub struct Qwen36Model {
    manifest: QwenModelManifest,
    checkpoint: ModelOptCheckpoint,
    artifact_dir: PathBuf,
    bf16_storage: Qwen36Bf16StorageConfig,
    fp8_attention_storage: Qwen36Fp8Storage,
    fp8_dense_mlp_storage: Qwen36Fp8Storage,
    fp8_lm_head_storage: Qwen36Fp8Storage,
    fp8_nvfp4_cache: Qwen36Fp8Nvfp4Cache,
}

/// Runtime storage for checkpoint BF16 projection weights.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Qwen36Bf16Storage {
    /// Retain the checkpoint's BF16 weights.
    Bf16,
    /// Convert weights to per-output-channel E4M3.
    Fp8,
    /// Convert weights to E2M1 with K16 UE4M3 scales.
    #[default]
    Nvfp4,
}

/// Independent storage choices for BF16 attention projections and the LM head.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen36Bf16StorageConfig {
    /// Storage used for BF16 attention projection weights.
    pub attention: Qwen36Bf16Storage,
    /// Storage used for the BF16 LM-head weight.
    pub lm_head: Qwen36Bf16Storage,
}

impl Qwen36Bf16StorageConfig {
    /// Creates an independent BF16 weight-storage configuration.
    pub const fn new(attention: Qwen36Bf16Storage, lm_head: Qwen36Bf16Storage) -> Self {
        Self { attention, lm_head }
    }
}

/// Runtime storage for checkpoint FP8 projection weights.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Qwen36Fp8Storage {
    /// Retain the checkpoint's E4M3 weights and scales.
    Fp8,
    /// Requantize the dequantized weights to E2M1 with K16 UE4M3 scales.
    #[default]
    Nvfp4,
}

#[derive(Clone, Copy)]
struct Qwen36AttentionStorage {
    bf16: Qwen36Bf16Storage,
    fp8: Qwen36Fp8Storage,
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
    q: Qwen36Linear,
    k: Qwen36Linear,
    v: Qwen36Linear,
    o: Qwen36Linear,
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
    compact_attention: Sm12xKvAttentionWorkspace,
    pub attn: DeviceBuffer<f32>,
    pub gated_attn: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f32>,
}

/// Persistent full-attention state owned by one generated sequence.
pub struct Qwen36FullAttentionState {
    compact_cache: Sm12xKvCache,
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
    qkv: Qwen36Linear,
    z: Qwen36Linear,
    alpha: Bf16Linear,
    beta: Bf16Linear,
    alpha_beta: Bf16Linear,
    conv_weight: DeviceBuffer<u16>,
    a_log: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
    norm_weight: DeviceBuffer<f32>,
    out: Qwen36Linear,
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
    pub gdn_output: DeviceBuffer<f32>,
    pub normed: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

/// Persistent Gated Delta Net state owned by one generated sequence.
pub struct Qwen36LinearAttentionState {
    /// Conv recurrent state, laid out as `[conv_channel][kernel - 1]`.
    pub conv_state: DeviceBuffer<f32>,
    /// GDN recurrent state, laid out as `[value_head][col][row]`.
    pub recurrent_state: DeviceBuffer<f32>,
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
    weight_only: bool,
}

enum Qwen36Linear {
    Nvfp4(Nvfp4DeviceLinear),
    Fp8(Fp8Linear),
    Bf16(Bf16Linear),
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
    fn new(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
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
        let checkpoint_dtype = &checkpoint.tensor_info(&format!("{prefix}.weight"))?.dtype;
        let uses_fp8 = if checkpoint_dtype == "BF16" {
            bf16_storage.attention == Qwen36Bf16Storage::Fp8
        } else {
            fp8_attention_storage == Qwen36Fp8Storage::Fp8
        };
        let plans = if !uses_fp8 || checkpoint.contains_tensor(&format!("{prefix}.input_scale")) {
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

pub(crate) struct Bf16Linear {
    pub(crate) weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Qwen36Model {
    /// Opens a Qwen3.6/Qwen3.5-MoE checkpoint and validates its hybrid schedule.
    pub fn open(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::open_with_storage(
            model_dir,
            Qwen36Bf16StorageConfig::default(),
            Qwen36Fp8Storage::default(),
        )
    }

    /// Opens a checkpoint with explicit runtime storage for BF16 weights.
    pub fn open_with_bf16_storage(
        model_dir: impl AsRef<std::path::Path>,
        bf16_storage: Qwen36Bf16StorageConfig,
    ) -> Result<Self> {
        Self::open_with_storage(model_dir, bf16_storage, Qwen36Fp8Storage::default())
    }

    /// Opens a checkpoint with explicit runtime storage for BF16 and FP8 weights.
    pub fn open_with_storage(
        model_dir: impl AsRef<std::path::Path>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        let artifact_dir = model_dir.as_ref().join(".eider-cache/qwen36-experts-v1");
        Self::open_with_storage_and_artifact_dir(
            model_dir,
            artifact_dir,
            bf16_storage,
            fp8_attention_storage,
        )
    }

    /// Opens a checkpoint with a writable root for reconstructed expert data.
    pub fn open_with_storage_and_artifact_dir(
        model_dir: impl AsRef<std::path::Path>,
        artifact_dir: impl Into<PathBuf>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        Self::open_with_fp8_storage_and_artifact_dir(
            model_dir,
            artifact_dir,
            bf16_storage,
            fp8_attention_storage,
            Qwen36Fp8Storage::default(),
            Qwen36Fp8Storage::default(),
        )
    }

    /// Opens a checkpoint with independent runtime storage for native FP8 weights.
    pub fn open_with_fp8_storage_and_artifact_dir(
        model_dir: impl AsRef<std::path::Path>,
        artifact_dir: impl Into<PathBuf>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
        fp8_dense_mlp_storage: Qwen36Fp8Storage,
        fp8_lm_head_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        let manifest = QwenModelManifest::load(model_dir.as_ref())?;
        if manifest.architecture != QwenArchitecture::Qwen35Hybrid {
            return Err(Error::Format {
                label: "Qwen3.5 hybrid model",
                detail: format!(
                    "expected qwen3_5 or qwen3_5_moe architecture, got {:?}",
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
        let artifact_dir = artifact_dir.into();
        let fp8_nvfp4_cache = Qwen36Fp8Nvfp4Cache::new(&checkpoint, &artifact_dir)?;
        Ok(Self {
            manifest,
            checkpoint,
            artifact_dir,
            bf16_storage,
            fp8_attention_storage,
            fp8_dense_mlp_storage,
            fp8_lm_head_storage,
            fp8_nvfp4_cache,
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
                Qwen36LinearAttentionWeights::load(
                    &self.checkpoint,
                    &self.manifest,
                    layer,
                    self.bf16_storage.attention,
                    self.fp8_attention_storage,
                    &self.fp8_nvfp4_cache,
                )?,
            )),
            QwenLayerKind::FullAttention => Ok(Qwen36LayerWeights::FullAttention(
                Qwen36FullAttentionWeights::load(
                    &self.checkpoint,
                    &self.manifest,
                    layer,
                    self.bf16_storage.attention,
                    self.fp8_attention_storage,
                    &self.fp8_nvfp4_cache,
                )?,
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
        Qwen36MoeWeights::load(
            &self.checkpoint,
            &self.manifest,
            &self.artifact_dir,
            layer,
            false,
        )
    }

    fn load_moe_from_prepared_cache(&self, layer: usize) -> Result<Qwen36MoeWeights> {
        Qwen36MoeWeights::load(
            &self.checkpoint,
            &self.manifest,
            &self.artifact_dir,
            layer,
            true,
        )
    }

    fn load_moe_from_prepared_cache_paged(
        &self,
        layer: usize,
        capacity: usize,
    ) -> Result<Qwen36MoeWeights> {
        Qwen36MoeWeights::load_paged(
            &self.checkpoint,
            &self.manifest,
            &self.artifact_dir,
            layer,
            true,
            capacity,
        )
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
    pub(crate) fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
        bf16_storage: Qwen36Bf16Storage,
        fp8_storage: Qwen36Fp8Storage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let prefix = format!("{}.layers.{layer}.self_attn", manifest.tensor_prefix);
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
        let q = Qwen36Linear::load(
            checkpoint,
            &format!("{prefix}.q_proj"),
            expected_q_rows,
            manifest.hidden,
            bf16_storage,
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
        let k = Qwen36Linear::load(
            checkpoint,
            &format!("{prefix}.k_proj"),
            expected_kv_rows,
            manifest.hidden,
            bf16_storage,
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
        let v = Qwen36Linear::load(
            checkpoint,
            &format!("{prefix}.v_proj"),
            expected_kv_rows,
            manifest.hidden,
            bf16_storage,
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
        let o = Qwen36Linear::load(
            checkpoint,
            &format!("{prefix}.o_proj"),
            manifest.hidden,
            manifest.q_heads * manifest.head_dim,
            bf16_storage,
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
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
        (self.q.rows(), self.k.rows(), self.v.rows(), self.o.rows())
    }

    /// Returns `(q_norm_len, k_norm_len)`.
    pub fn norm_lens(&self) -> (usize, usize) {
        (self.q_norm_weight.len(), self.k_norm_weight.len())
    }

    /// Returns output width.
    pub fn output_width(&self) -> usize {
        self.o.rows()
    }

    fn run_qkv_projections(
        &self,
        workspace: &mut Qwen36FullAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if let (Qwen36Linear::Fp8(q), Qwen36Linear::Fp8(k), Qwen36Linear::Fp8(v)) =
            (&self.q, &self.k, &self.v)
            && q.channel_weight_scale.is_none()
            && k.channel_weight_scale.is_none()
            && v.channel_weight_scale.is_none()
            && std::env::var_os("QWEN36_FP8_W8A8").is_none()
        {
            fp8_linear_triple_configured_f32_into_on_stream(
                hidden,
                &q.weight,
                &k.weight,
                &v.weight,
                workspace.q_proj_output.output(),
                workspace.k.output(),
                workspace.v.output(),
                q.rows,
                k.rows,
                v.rows,
                q.cols,
                q.weight_scale,
                k.weight_scale,
                v.weight_scale,
                128,
                stream,
            )?;
            maybe_round_device_f32_to_bf16(&mut workspace.q_proj_output, stream)?;
            maybe_round_device_f32_to_bf16(&mut workspace.k, stream)?;
            return maybe_round_device_f32_to_bf16(&mut workspace.v, stream);
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
        )
    }

    /// Prepares one token's RoPE'd Q/K and V without appending or running attention.
    pub fn prepare_qkv_one_token(
        &self,
        workspace: &mut Qwen36FullAttentionWorkspace,
        state: &Qwen36FullAttentionState,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if position >= state.cache_capacity {
            return Err(Error::Shape {
                label: "Qwen3.6 full-attention preparation",
                expected: format!("position < {}", state.cache_capacity),
                actual: position.to_string(),
            });
        }
        self.run_qkv_projections(workspace, hidden, stream)?;
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
        )
    }

    /// Runs one token through this full-attention layer.
    pub fn run_one_token<'a>(
        &'a self,
        workspace: &'a mut Qwen36FullAttentionWorkspace,
        state: &mut Qwen36FullAttentionState,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<Qwen36FullAttentionStep<'a>> {
        if position >= state.cache_capacity {
            return Err(Error::Shape {
                label: "Qwen3.6 full-attention cache",
                expected: format!("position < {}", state.cache_capacity),
                actual: position.to_string(),
            });
        }

        self.prepare_qkv_one_token(workspace, state, manifest, hidden, position, stream)?;
        state.compact_cache.append_at_on_stream(
            &workspace.k_rope,
            &workspace.v,
            position,
            stream,
        )?;
        workspace.compact_attention.attention_into_on_stream(
            &state.compact_cache,
            &workspace.q_rope,
            workspace.attn.output(),
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

    /// Runs one token against a page-table cache restricted by a sparse token mask.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_one_token_paged_sparse<'a>(
        &'a self,
        workspace: &'a mut Qwen36FullAttentionWorkspace,
        pool: &mut Sm12xKvPagePool,
        page_table: &DeviceBuffer<u32>,
        selected_blocks: &DeviceBuffer<u8>,
        selected_tiles: &DeviceBuffer<u8>,
        selected_tokens: usize,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: usize,
        slot: usize,
        page_offset: usize,
        stream: &CudaStream,
    ) -> Result<Qwen36FullAttentionStep<'a>> {
        let capacity = page_table
            .len()
            .checked_mul(eider_cuda::SM12X_KV_PAGE_TOKENS)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 paged sparse attention capacity",
                expected: "page table capacity without overflow".to_string(),
                actual: page_table.len().to_string(),
            })?;
        if position >= capacity {
            return Err(Error::Shape {
                label: "Qwen3.6 paged sparse attention position",
                expected: format!("position < {capacity}"),
                actual: position.to_string(),
            });
        }
        self.run_qkv_projections(workspace, hidden, stream)?;
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
        pool.append_at_offsets_on_stream(
            slot,
            page_offset,
            &workspace.k_rope,
            0,
            &workspace.v,
            0,
            stream,
        )?;
        workspace
            .compact_attention
            .attention_paged_sparse_offsets_into_on_stream(
                pool,
                page_table,
                position + 1,
                selected_blocks,
                selected_tiles,
                selected_tokens,
                &workspace.q_rope,
                0,
                workspace.attn.output(),
                0,
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

    #[allow(clippy::too_many_arguments)]
    fn run_one_token_indexed<'a>(
        &'a self,
        workspace: &'a mut Qwen36FullAttentionWorkspace,
        state: &mut Qwen36FullAttentionState,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: &DeviceBuffer<u32>,
        cache_len: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<Qwen36FullAttentionStep<'a>> {
        self.run_qkv_projections(workspace, hidden, stream)?;
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
        state.compact_cache.append_indexed_on_stream(
            &workspace.k_rope,
            &workspace.v,
            position,
            stream,
        )?;
        workspace
            .compact_attention
            .attention_indexed_into_on_stream(
                &state.compact_cache,
                &workspace.q_rope,
                cache_len,
                workspace.attn.output(),
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
    pub(crate) fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
        bf16_storage: Qwen36Bf16Storage,
        fp8_storage: Qwen36Fp8Storage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let fp8 = Rc::new(Qwen36LinearFp8Execution::new(
            checkpoint,
            manifest,
            Qwen36Bf16StorageConfig::new(bf16_storage, Qwen36Bf16Storage::Bf16),
            fp8_storage,
        )?);
        Self::load_with_fp8(
            checkpoint,
            manifest,
            layer,
            fp8,
            bf16_storage,
            fp8_storage,
            fp8_nvfp4_cache,
        )
    }

    fn load_with_fp8(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
        fp8: Rc<Qwen36LinearFp8Execution>,
        bf16_storage: Qwen36Bf16Storage,
        fp8_storage: Qwen36Fp8Storage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
            label: "Qwen3.6 linear attention",
            detail: "manifest has no linear-attention config".to_string(),
        })?;
        let prefix = format!("{}.layers.{layer}.linear_attn", manifest.tensor_prefix);
        let key_heads = linear.key_heads;
        let value_heads = linear.value_heads;
        let head_v_dim = linear.value_head_dim;
        let key_dim = key_heads * linear.key_head_dim;
        let value_dim = value_heads * head_v_dim;
        let qkv_rows = key_dim * 2 + value_dim;
        let storage = Qwen36AttentionStorage {
            bf16: bf16_storage,
            fp8: fp8_storage,
        };
        let qkv = Qwen36Linear::load(
            checkpoint,
            &format!("{prefix}.in_proj_qkv"),
            qkv_rows,
            manifest.hidden,
            bf16_storage,
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
        let z = Qwen36Linear::load_reordered_v_rows(
            checkpoint,
            &format!("{prefix}.in_proj_z"),
            value_heads,
            head_v_dim,
            manifest.hidden,
            key_heads,
            storage,
            fp8_nvfp4_cache,
        )?;
        let out = Qwen36Linear::load_reordered_v_cols(
            checkpoint,
            &format!("{prefix}.out_proj"),
            manifest.hidden,
            value_heads,
            head_v_dim,
            key_heads,
            storage,
            fp8_nvfp4_cache,
        )?;

        // Reorder V heads from grouped-by-K to tiled order for tensors consumed after GDN prep.
        // qkv/conv stay in checkpoint order; qwen36_gdn_prep reorders V after depthwise conv.

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
        let mut alpha_beta_host = alpha_host.clone();
        alpha_beta_host.extend_from_slice(&beta_host);

        // Conv1d: BF16 [conv_dim, kernel]
        let conv_host = read_bf16_flat_host(
            checkpoint,
            &format!("{prefix}.conv1d.weight"),
            qkv_rows * linear.conv_kernel,
        )?;

        // A_log / dt_bias: BF16 [value_heads] — reorder elements
        let a_log_host = read_bf16_flat_host(checkpoint, &format!("{prefix}.A_log"), value_heads)?;
        let dt_bias_host =
            read_bf16_flat_host(checkpoint, &format!("{prefix}.dt_bias"), value_heads)?;
        let a_log_host = reorder_v_heads_1d(a_log_host, key_heads, value_heads);
        let dt_bias_host = reorder_v_heads_1d(dt_bias_host, key_heads, value_heads);

        Ok(Self {
            fp8,
            qkv,
            z,
            alpha: Bf16Linear::from_host(&alpha_host, value_heads, manifest.hidden)?,
            beta: Bf16Linear::from_host(&beta_host, value_heads, manifest.hidden)?,
            alpha_beta: Bf16Linear::from_host(&alpha_beta_host, value_heads * 2, manifest.hidden)?,
            conv_weight: DeviceBuffer::from_host(&conv_host)?,
            a_log: DeviceBuffer::from_host(&a_log_host)?,
            dt_bias: DeviceBuffer::from_host(&dt_bias_host)?,
            norm_weight: read_bf16_vector_as_f32_device(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                linear.value_head_dim,
            )?,
            out,
        })
    }

    fn run_qkv(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let Qwen36Linear::Fp8(qkv) = &self.qkv else {
            return self.qkv.run_into(
                hidden,
                &mut workspace.qkv_output,
                &mut workspace.fp8_dynamic_input,
                &mut workspace.fp8_dynamic_input_scale,
                stream,
            );
        };
        let Some(plans) = self.fp8.plans.as_ref() else {
            return qkv.run_into(
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
        qkv.run_prequantized_channel_scaled_with_plan_into(
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
        let Qwen36Linear::Fp8(z) = &self.z else {
            return self.z.run_into(
                hidden,
                &mut workspace.z_output,
                &mut workspace.fp8_dynamic_input,
                &mut workspace.fp8_dynamic_input_scale,
                stream,
            );
        };
        let Some(plans) = self.fp8.plans.as_ref() else {
            return z.run_into(
                hidden,
                &mut workspace.z_output,
                &mut workspace.fp8_dynamic_input,
                &mut workspace.fp8_dynamic_input_scale,
                stream,
            );
        };
        z.run_prequantized_channel_scaled_with_plan_into(
            &self.fp8,
            &plans.z,
            &workspace.fp8_dynamic_input,
            &workspace.fp8_dynamic_input_scale,
            &mut workspace.z_output,
            stream,
        )
    }

    fn run_qkv_z(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if let (Qwen36Linear::Bf16(qkv), Qwen36Linear::Bf16(z)) = (&self.qkv, &self.z) {
            return bf16_linear_pair_logits_f32_into_on_stream(
                hidden,
                &qkv.weight,
                &z.weight,
                workspace.qkv_output.output(),
                workspace.z_output.output(),
                qkv.rows,
                z.rows,
                qkv.cols,
                stream,
            );
        }
        if let (Qwen36Linear::Fp8(qkv), Qwen36Linear::Fp8(z)) = (&self.qkv, &self.z)
            && self.fp8.plans.is_none()
            && qkv.channel_weight_scale.is_none()
            && z.channel_weight_scale.is_none()
            && std::env::var_os("QWEN36_FP8_W8A8").is_none()
        {
            fp8_linear_pair_configured_f32_into_on_stream(
                hidden,
                &qkv.weight,
                &z.weight,
                workspace.qkv_output.output(),
                workspace.z_output.output(),
                qkv.rows,
                z.rows,
                qkv.cols,
                qkv.weight_scale,
                z.weight_scale,
                128,
                stream,
            )?;
            maybe_round_device_f32_to_bf16(&mut workspace.qkv_output, stream)?;
            return maybe_round_device_f32_to_bf16(&mut workspace.z_output, stream);
        }

        self.run_qkv(workspace, hidden, stream)?;
        self.run_z(workspace, hidden, stream)
    }

    fn run_alpha_beta(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_pair_logits_f32_into_on_stream(
            hidden,
            &self.alpha.weight,
            &self.beta.weight,
            workspace.alpha.output(),
            workspace.beta_input.output(),
            self.alpha.rows,
            self.beta.rows,
            self.alpha.cols,
            stream,
        )
    }

    fn run_output_projection(
        &self,
        workspace: &mut Qwen36LinearAttentionWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let Qwen36Linear::Fp8(out) = &self.out else {
            return self.out.run_into(
                &workspace.normed,
                &mut workspace.output,
                &mut workspace.fp8_value_input,
                &mut workspace.fp8_value_input_scale,
                stream,
            );
        };
        let Some(plans) = self.fp8.plans.as_ref() else {
            return out.run_into(
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
        out.run_prequantized_channel_scaled_with_plan_into(
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
        state: &mut Qwen36LinearAttentionState,
        hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_qkv_z(workspace, hidden, stream)?;
        self.run_alpha_beta(workspace, hidden, stream)?;
        qwen36_gdn_prep_into_on_stream(
            &workspace.qkv_output,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            state.conv_state.inout(),
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
        state: &mut Qwen36LinearAttentionState,
        stream: &CudaStream,
    ) -> Result<()> {
        gated_delta_net_128_f32_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            state.recurrent_state.inout(),
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
        state: &mut Qwen36LinearAttentionState,
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
            let (_, ms) = timed_cuda(stream, || self.run_alpha_beta(workspace, hidden, stream))?;
            profile.qwen36_linear_alpha_beta_ms += ms;
            let (_, ms) = timed_cuda(stream, || {
                qwen36_gdn_prep_into_on_stream(
                    &workspace.qkv_output,
                    &self.conv_weight,
                    workspace.q.output(),
                    workspace.k.output(),
                    workspace.v.output(),
                    state.conv_state.inout(),
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
                    state.recurrent_state.inout(),
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
            self.enqueue_pre_gdn(workspace, state, hidden, stream)?;
            self.enqueue_gdn(workspace, state, stream)?;
            self.enqueue_post_gdn(workspace, rms_eps, stream)?;
        }
        Ok(Qwen36LinearAttentionStep {
            qkv_output: &workspace.qkv_output,
            z_output: &workspace.z_output,
            gdn_output: &workspace.gdn_output,
            output: &workspace.output,
        })
    }

    /// Runs one token with the sigmoid output gate used by Qwen4-Exp.
    pub(crate) fn run_one_token_sigmoid_output_gate<'a>(
        &'a self,
        workspace: &'a mut Qwen36LinearAttentionWorkspace,
        state: &mut Qwen36LinearAttentionState,
        hidden: &DeviceBuffer<f32>,
        rms_eps: f32,
        stream: &CudaStream,
    ) -> Result<Qwen36LinearAttentionStep<'a>> {
        self.enqueue_pre_gdn(workspace, state, hidden, stream)?;
        self.enqueue_gdn(workspace, state, stream)?;
        ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
            &workspace.gdn_output,
            &workspace.z_output,
            &self.norm_weight,
            workspace.normed.output(),
            workspace.linear.value_heads,
            workspace.linear.value_head_dim,
            rms_eps,
            stream,
        )?;
        self.run_output_projection(workspace, stream)?;
        Ok(Qwen36LinearAttentionStep {
            qkv_output: &workspace.qkv_output,
            z_output: &workspace.z_output,
            gdn_output: &workspace.gdn_output,
            output: &workspace.output,
        })
    }

    /// Returns output width.
    pub fn output_width(&self) -> usize {
        self.out.rows()
    }
}

/// Loads the reusable GDN portion of a Qwen hybrid layer for another text runtime.
pub(crate) fn load_hybrid_linear_attention(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    artifact_dir: &std::path::Path,
    layer: usize,
) -> Result<Qwen36LinearAttentionWeights> {
    let cache = Qwen36Fp8Nvfp4Cache::new(checkpoint, artifact_dir)?;
    Qwen36LinearAttentionWeights::load(
        checkpoint,
        manifest,
        layer,
        Qwen36Bf16Storage::Bf16,
        Qwen36Fp8Storage::Nvfp4,
        &cache,
    )
}

/// Loads the dense-oracle full-attention portion of a Qwen hybrid layer.
pub(crate) fn load_hybrid_full_attention(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    artifact_dir: &std::path::Path,
    layer: usize,
) -> Result<Qwen36FullAttentionWeights> {
    let cache = Qwen36Fp8Nvfp4Cache::new(checkpoint, artifact_dir)?;
    Qwen36FullAttentionWeights::load(
        checkpoint,
        manifest,
        layer,
        Qwen36Bf16Storage::Bf16,
        Qwen36Fp8Storage::Nvfp4,
        &cache,
    )
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
            qkv_output: DeviceBuffer::zeroed(weights.qkv.rows())?,
            z_output: DeviceBuffer::zeroed(weights.z.rows())?,
            alpha: DeviceBuffer::zeroed(linear.value_heads)?,
            beta_input: DeviceBuffer::zeroed(linear.value_heads)?,
            gate: DeviceBuffer::zeroed(linear.value_heads)?,
            beta: DeviceBuffer::zeroed(linear.value_heads)?,
            q: DeviceBuffer::zeroed(value_dim)?,
            k: DeviceBuffer::zeroed(value_dim)?,
            v: DeviceBuffer::zeroed(value_dim)?,
            gdn_output: DeviceBuffer::zeroed(value_dim)?,
            normed: DeviceBuffer::zeroed(value_dim)?,
            output: DeviceBuffer::zeroed(manifest.hidden)?,
        })
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.fp8_dynamic_input.device_bytes()
            + self.fp8_dynamic_input_scale.device_bytes()
            + self.fp8_value_input.device_bytes()
            + self.fp8_value_input_scale.device_bytes()
            + self.qkv_output.device_bytes()
            + self.z_output.device_bytes()
            + self.alpha.device_bytes()
            + self.beta_input.device_bytes()
            + self.gate.device_bytes()
            + self.beta.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.gdn_output.device_bytes()
            + self.normed.device_bytes()
            + self.output.device_bytes()
    }
}

impl Qwen36LinearAttentionState {
    /// Allocates empty recurrent state for one generated sequence.
    pub fn new(
        linear: QwenLinearAttentionConfig,
        weights: &Qwen36LinearAttentionWeights,
    ) -> Result<Self> {
        Ok(Self {
            conv_state: DeviceBuffer::zeroed(weights.qkv.rows() * (linear.conv_kernel - 1))?,
            recurrent_state: DeviceBuffer::zeroed(
                linear.value_heads * linear.value_head_dim * linear.value_head_dim,
            )?,
        })
    }

    pub(crate) fn copy_from_on_stream(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        if self.conv_state.len() != source.conv_state.len()
            || self.recurrent_state.len() != source.recurrent_state.len()
        {
            return Err(Error::Shape {
                label: "Qwen linear-attention state copy",
                expected: format!(
                    "conv={} recurrent={}",
                    self.conv_state.len(),
                    self.recurrent_state.len()
                ),
                actual: format!(
                    "conv={} recurrent={}",
                    source.conv_state.len(),
                    source.recurrent_state.len()
                ),
            });
        }
        self.conv_state.copy_prefix_from_device_on_stream(
            &source.conv_state,
            source.conv_state.len(),
            stream,
        )?;
        self.recurrent_state.copy_prefix_from_device_on_stream(
            &source.recurrent_state,
            source.recurrent_state.len(),
            stream,
        )
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.conv_state.device_bytes() + self.recurrent_state.device_bytes()
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
            q_proj_output: DeviceBuffer::zeroed(weights.q.rows())?,
            q_normed: DeviceBuffer::zeroed(q_width)?,
            gate: DeviceBuffer::zeroed(q_width)?,
            k: DeviceBuffer::zeroed(kv_width)?,
            k_normed: DeviceBuffer::zeroed(kv_width)?,
            v: DeviceBuffer::zeroed(kv_width)?,
            q_rope: DeviceBuffer::zeroed(q_width)?,
            k_rope: DeviceBuffer::zeroed(kv_width)?,
            compact_attention: Sm12xKvAttentionWorkspace::new_gqa(
                cache_capacity,
                manifest.q_heads,
                manifest.kv_heads,
                manifest.head_dim,
            )?,
            attn: DeviceBuffer::zeroed(q_width)?,
            gated_attn: DeviceBuffer::zeroed(q_width)?,
            output: DeviceBuffer::zeroed(manifest.hidden)?,
        })
    }

    /// Returns the exact device bytes owned by the full-attention workspace.
    pub fn device_bytes(&self) -> usize {
        self.fp8_dynamic_input.device_bytes()
            + self.fp8_dynamic_input_scale.device_bytes()
            + self.q_proj_output.device_bytes()
            + self.q_normed.device_bytes()
            + self.gate.device_bytes()
            + self.k.device_bytes()
            + self.k_normed.device_bytes()
            + self.v.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.compact_attention.device_bytes()
            + self.attn.device_bytes()
            + self.gated_attn.device_bytes()
            + self.output.device_bytes()
    }
}

impl Qwen36FullAttentionState {
    /// Allocates an empty compact KV cache for one generated sequence.
    pub fn new(manifest: &QwenModelManifest, cache_capacity: usize) -> Result<Self> {
        if cache_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 full-attention state",
                expected: "non-zero cache capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        Ok(Self {
            compact_cache: Sm12xKvCache::new(cache_capacity, manifest.kv_heads, manifest.head_dim)?,
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
            weight_only: false,
        })
    }

    fn from_bf16_host(weight: &[u16], rows: usize, cols: usize) -> Result<Self> {
        let expected = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
            label: "Qwen3.5 BF16-to-FP8 projection",
            expected: "rows * cols without overflow".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
        if weight.len() != expected {
            return Err(Error::Shape {
                label: "Qwen3.5 BF16-to-FP8 projection",
                expected: format!("{expected} weights"),
                actual: format!("{} weights", weight.len()),
            });
        }
        let source = DeviceBuffer::from_host(weight)?;
        let mut quantized = DeviceBuffer::zeroed(weight.len())?;
        let stream = CudaStream::new_blocking()?;
        let scales = weight
            .chunks_exact(cols)
            .map(|row| {
                let max_abs = row
                    .iter()
                    .map(|&value| eider_cuda::format::bf16_to_f32(value).abs())
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, f32::max);
                if max_abs == 0.0 { 1.0 } else { max_abs / 448.0 }
            })
            .collect::<Vec<_>>();
        let scales = DeviceBuffer::from_host(&scales)?;
        quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream(
            &source,
            &scales,
            quantized.output(),
            rows,
            cols,
            &stream,
        )?;
        stream.synchronize()?;
        Ok(Self {
            weight: quantized,
            rows,
            cols,
            weight_scale: 1.0,
            channel_weight_scale: Some(scales),
            input_scale: None,
            weight_only: true,
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
        let result = if let Some(channel_weight_scale) = &self.channel_weight_scale
            && self.weight_only
        {
            fp8_linear_channel_scaled_f32_into_on_stream(
                input,
                &self.weight,
                channel_weight_scale,
                output.output(),
                self.rows,
                self.cols,
                128,
                stream,
            )
        } else if let Some(channel_weight_scale) = &self.channel_weight_scale {
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
}

impl Qwen36Linear {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
        bf16_storage: Qwen36Bf16Storage,
        fp8_storage: Qwen36Fp8Storage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let linear = if checkpoint.contains_tensor(&format!("{prefix}.weight_scale_2"))
            || checkpoint.contains_tensor(&format!("{prefix}.weight_global_scale"))
        {
            Self::Nvfp4(Nvfp4DeviceLinear::load(checkpoint, prefix)?)
        } else if checkpoint.tensor_info(&weight_name)?.dtype == "BF16" {
            match bf16_storage {
                Qwen36Bf16Storage::Bf16 => {
                    Self::Bf16(Bf16Linear::load(checkpoint, &weight_name, rows, cols)?)
                }
                Qwen36Bf16Storage::Fp8 => {
                    let host = read_bf16_matrix_host(checkpoint, &weight_name, rows, cols)?;
                    Self::Fp8(Fp8Linear::from_bf16_host(&host, rows, cols)?)
                }
                Qwen36Bf16Storage::Nvfp4 => {
                    let host = read_bf16_matrix_host(checkpoint, &weight_name, rows, cols)?;
                    Self::Nvfp4(Nvfp4DeviceLinear::from_bf16_host(
                        prefix, &host, rows, cols,
                    )?)
                }
            }
        } else {
            match fp8_storage {
                Qwen36Fp8Storage::Fp8 => {
                    let host = checkpoint.load_fp8_linear(prefix)?;
                    Self::Fp8(Fp8Linear::from_host(&host)?)
                }
                Qwen36Fp8Storage::Nvfp4 => Self::Nvfp4(Nvfp4DeviceLinear::from_host(
                    &fp8_nvfp4_cache.load_or_quantize(checkpoint, prefix)?,
                )?),
            }
        };
        linear.require_shape(rows, cols, "Qwen3.5 projection")?;
        Ok(linear)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_reordered_v_rows(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        value_heads: usize,
        head_dim: usize,
        cols: usize,
        key_heads: usize,
        storage: Qwen36AttentionStorage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let rows = value_heads * head_dim;
        let weight_name = format!("{prefix}.weight");
        let info = checkpoint.tensor_info(&weight_name)?;
        if checkpoint.contains_tensor(&format!("{prefix}.weight_scale_2"))
            || checkpoint.contains_tensor(&format!("{prefix}.weight_global_scale"))
        {
            let host = reorder_nvfp4_v_rows(
                checkpoint.load_nvfp4_linear(prefix)?,
                key_heads,
                value_heads,
                head_dim,
            );
            return Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_host(&host)?));
        }
        if info.dtype == "BF16" {
            let host = read_bf16_matrix_host(checkpoint, &weight_name, rows, cols)?;
            let host = reorder_bf16_v_rows(host, key_heads, value_heads, head_dim);
            return match storage.bf16 {
                Qwen36Bf16Storage::Bf16 => {
                    Ok(Self::Bf16(Bf16Linear::from_host(&host, rows, cols)?))
                }
                Qwen36Bf16Storage::Fp8 => {
                    Ok(Self::Fp8(Fp8Linear::from_bf16_host(&host, rows, cols)?))
                }
                Qwen36Bf16Storage::Nvfp4 => Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_bf16_host(
                    prefix, &host, rows, cols,
                )?)),
            };
        }
        match storage.fp8 {
            Qwen36Fp8Storage::Fp8 => {
                let host = reorder_fp8_v_rows(
                    checkpoint.load_fp8_linear(prefix)?,
                    key_heads,
                    value_heads,
                    head_dim,
                );
                Ok(Self::Fp8(Fp8Linear::from_host(&host)?))
            }
            Qwen36Fp8Storage::Nvfp4 => {
                let host = reorder_nvfp4_v_rows(
                    fp8_nvfp4_cache.load_or_quantize(checkpoint, prefix)?,
                    key_heads,
                    value_heads,
                    head_dim,
                );
                Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_host(&host)?))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn load_reordered_v_cols(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        value_heads: usize,
        head_dim: usize,
        key_heads: usize,
        storage: Qwen36AttentionStorage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let cols = value_heads * head_dim;
        let weight_name = format!("{prefix}.weight");
        let info = checkpoint.tensor_info(&weight_name)?;
        if checkpoint.contains_tensor(&format!("{prefix}.weight_scale_2"))
            || checkpoint.contains_tensor(&format!("{prefix}.weight_global_scale"))
        {
            let host = reorder_nvfp4_v_cols(
                checkpoint.load_nvfp4_linear(prefix)?,
                key_heads,
                value_heads,
                head_dim,
            );
            return Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_host(&host)?));
        }
        if info.dtype == "BF16" {
            let host = read_bf16_matrix_host(checkpoint, &weight_name, rows, cols)?;
            let host = reorder_bf16_v_cols(host, rows, key_heads, value_heads, head_dim);
            return match storage.bf16 {
                Qwen36Bf16Storage::Bf16 => {
                    Ok(Self::Bf16(Bf16Linear::from_host(&host, rows, cols)?))
                }
                Qwen36Bf16Storage::Fp8 => {
                    Ok(Self::Fp8(Fp8Linear::from_bf16_host(&host, rows, cols)?))
                }
                Qwen36Bf16Storage::Nvfp4 => Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_bf16_host(
                    prefix, &host, rows, cols,
                )?)),
            };
        }
        match storage.fp8 {
            Qwen36Fp8Storage::Fp8 => {
                let host = reorder_fp8_v_cols(
                    checkpoint.load_fp8_linear(prefix)?,
                    key_heads,
                    value_heads,
                    head_dim,
                );
                Ok(Self::Fp8(Fp8Linear::from_host(&host)?))
            }
            Qwen36Fp8Storage::Nvfp4 => {
                let host = reorder_nvfp4_v_cols(
                    fp8_nvfp4_cache.load_or_quantize(checkpoint, prefix)?,
                    key_heads,
                    value_heads,
                    head_dim,
                );
                Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_host(&host)?))
            }
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Nvfp4(linear) => linear.out_features,
            Self::Fp8(linear) => linear.rows,
            Self::Bf16(linear) => linear.rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Nvfp4(linear) => linear.in_features,
            Self::Fp8(linear) => linear.cols,
            Self::Bf16(linear) => linear.cols,
        }
    }

    fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        dynamic_input: &mut DeviceBuffer<u8>,
        dynamic_input_scale: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Nvfp4(linear) => linear.run_f32_into(input, output, stream),
            Self::Fp8(linear) => {
                linear.run_into(input, output, dynamic_input, dynamic_input_scale, stream)
            }
            Self::Bf16(linear) => linear.run_into(input, output, stream),
        }
    }

    fn require_shape(&self, rows: usize, cols: usize, label: &'static str) -> Result<()> {
        if self.rows() != rows || self.cols() != cols {
            return Err(Error::Shape {
                label,
                expected: format!("rows={rows} cols={cols}"),
                actual: format!("rows={} cols={}", self.rows(), self.cols()),
            });
        }
        Ok(())
    }
}

impl Bf16Linear {
    pub(crate) fn load(
        checkpoint: &ModelOptCheckpoint,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
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

    pub(crate) fn run_into(
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

    pub(crate) fn run_batch_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_size: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_batch_into_on_stream(
            input,
            &self.weight,
            output.output(),
            batch_size,
            self.rows,
            self.cols,
            stream,
        )
    }

    pub(crate) fn run_exact_two_rows_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_two_rows_f32_into_on_stream(
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

pub(crate) fn read_bf16_vector_as_f32_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    len: usize,
) -> Result<DeviceBuffer<f32>> {
    let bytes = read_checked_bf16_bytes(checkpoint, name, &[len])?;
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|chunk| eider_cuda::format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn read_bf16_vector_delta_as_f32_device(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    len: usize,
) -> Result<DeviceBuffer<f32>> {
    let bytes = read_checked_bf16_bytes(checkpoint, name, &[len])?;
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|chunk| {
                1.0 + eider_cuda::format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
            })
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn read_bf16_flat_host(
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

/// Reorders grouped V-head blocks in a BF16 matrix `[value_heads * head_dim, cols]`.
fn reorder_bf16_v_rows(
    data: Vec<u16>,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
) -> Vec<u16> {
    if key_heads == value_heads {
        return data;
    }
    let cols = data.len() / (value_heads * head_dim);
    let rows_per_head = head_dim * cols;
    let v_per_k = value_heads / key_heads;
    let mut out = vec![0u16; data.len()];
    for v_k_head in 0..value_heads {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let src = v_k_head * rows_per_head;
        let dst = (v_sub * key_heads + k_head) * rows_per_head;
        out[dst..dst + rows_per_head].copy_from_slice(&data[src..src + rows_per_head]);
    }
    out
}

/// Reorders grouped V-head blocks in a BF16 matrix `[rows, value_heads * head_dim]`.
fn reorder_bf16_v_cols(
    data: Vec<u16>,
    rows: usize,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
) -> Vec<u16> {
    if key_heads == value_heads {
        return data;
    }
    let cols = value_heads * head_dim;
    let v_per_k = value_heads / key_heads;
    let mut out = vec![0u16; data.len()];
    for v_k_head in 0..value_heads {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let src_col = v_k_head * head_dim;
        let dst_col = (v_sub * key_heads + k_head) * head_dim;
        for row in 0..rows {
            let src = row * cols + src_col;
            let dst = row * cols + dst_col;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
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

/// Reorders V-head output rows in a ModelOpt NVFP4 weight.
fn reorder_nvfp4_v_rows(
    mut host: ModelOptNvfp4Linear,
    key_heads: usize,
    value_heads: usize,
    head_v_dim: usize,
) -> ModelOptNvfp4Linear {
    if key_heads == value_heads || host.out_features != value_heads * head_v_dim {
        return host;
    }
    let v_per_k = value_heads / key_heads;
    let packed_row_bytes = host.in_features / 2;
    let scale_row_bytes = host.in_features / 16;
    let mut packed = vec![0u8; host.packed_weight.len()];
    let mut scales = vec![0u8; host.weight_scale.len()];
    for v_k_head in 0..value_heads {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let dst_head = v_sub * key_heads + k_head;
        let src_packed = v_k_head * head_v_dim * packed_row_bytes;
        let dst_packed = dst_head * head_v_dim * packed_row_bytes;
        let packed_len = head_v_dim * packed_row_bytes;
        packed[dst_packed..dst_packed + packed_len]
            .copy_from_slice(&host.packed_weight[src_packed..src_packed + packed_len]);
        let src_scale = v_k_head * head_v_dim * scale_row_bytes;
        let dst_scale = dst_head * head_v_dim * scale_row_bytes;
        let scale_len = head_v_dim * scale_row_bytes;
        scales[dst_scale..dst_scale + scale_len]
            .copy_from_slice(&host.weight_scale[src_scale..src_scale + scale_len]);
    }
    host.packed_weight = packed;
    host.weight_scale = scales;
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

/// Reorders V-head input columns in a ModelOpt NVFP4 output projection.
fn reorder_nvfp4_v_cols(
    mut host: ModelOptNvfp4Linear,
    key_heads: usize,
    value_heads: usize,
    head_v_dim: usize,
) -> ModelOptNvfp4Linear {
    if key_heads == value_heads || host.in_features != value_heads * head_v_dim {
        return host;
    }
    let v_per_k = value_heads / key_heads;
    let packed_row_bytes = host.in_features / 2;
    let scale_row_bytes = host.in_features / 16;
    let packed_head_bytes = head_v_dim / 2;
    let scale_head_bytes = head_v_dim / 16;
    let mut packed = vec![0u8; host.packed_weight.len()];
    let mut scales = vec![0u8; host.weight_scale.len()];
    for v_k_head in 0..value_heads {
        let k_head = v_k_head / v_per_k;
        let v_sub = v_k_head % v_per_k;
        let dst_head = v_sub * key_heads + k_head;
        for row in 0..host.out_features {
            let src = row * packed_row_bytes + v_k_head * packed_head_bytes;
            let dst = row * packed_row_bytes + dst_head * packed_head_bytes;
            packed[dst..dst + packed_head_bytes]
                .copy_from_slice(&host.packed_weight[src..src + packed_head_bytes]);
            let src = row * scale_row_bytes + v_k_head * scale_head_bytes;
            let dst = row * scale_row_bytes + dst_head * scale_head_bytes;
            scales[dst..dst + scale_head_bytes]
                .copy_from_slice(&host.weight_scale[src..src + scale_head_bytes]);
        }
    }
    host.packed_weight = packed;
    host.weight_scale = scales;
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
    Ok(shard.read_tensor_bytes(name)?)
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
    let Some(sections) = manifest.mrope_sections else {
        return rope_neox_partial_f32_into_on_stream(
            rows,
            manifest.head_dim,
            manifest.rotary_dim,
            input,
            output.output(),
            position,
            manifest.rope_theta,
            stream,
        );
    };
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
    let Some(sections) = manifest.mrope_sections else {
        return rope_neox_partial_f32_indexed_into_on_stream(
            rows,
            manifest.head_dim,
            manifest.rotary_dim,
            input,
            output.output(),
            position,
            manifest.rope_theta,
            stream,
        );
    };
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
/// Resident NVFP4 layers use native W4A4 gate/up and SM12x down execution;
/// mixed-precision layers keep channel-scaled FP8 expert tables device-resident.
pub struct Qwen36MoeWeights {
    router: Bf16Linear,
    experts: Vec<LazyQwen36Expert>,
    expert_ptrs: super::infer::MoeExpertPointerTables,
    gate_up_unity_alphas: DeviceBuffer<f32>,
    storage_plan: Qwen36MoeStoragePlan,
    gate_up_storage: Qwen36GateUpStorage,
    grouped: Option<Qwen36GroupedMoeWeights>,
    fp8_experts: Option<Qwen36Fp8Experts>,
    shared: Qwen36SharedExpertStorage,
    shared_gate: Bf16Linear,
    _sm12x_down: Vec<Sm12xFp4DeviceGemmWeight>,
    sm12x_down_tiles: Option<DeviceBuffer<DeviceAddress<u8>>>,
    sm12x_down_scales: Option<DeviceBuffer<DeviceAddress<u32>>>,
    sm12x_down_m_tiles: usize,
    sm12x_down_k_tiles: usize,
    num_experts: usize,
    experts_per_token: usize,
    expert_intermediate: usize,
    norm_topk_prob: bool,
    expert_pager: std::cell::RefCell<Option<Qwen36ExpertPager>>,
}

/// Result of resolving original expert IDs into a bounded resident slot table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen36PageResolution {
    /// Slot IDs corresponding one-for-one with the requested expert IDs.
    pub slots: Vec<u32>,
    /// Requested experts that were already resident.
    pub hits: usize,
    /// Requested experts loaded into a slot.
    pub misses: usize,
    /// Prepared cache bytes read for misses.
    pub bytes_read: usize,
}

/// Cumulative expert-cache activity across paged layers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen36PagingStats {
    /// Expert route lookups served by an already-resident slot.
    pub hits: u64,
    /// Expert route lookups that loaded a prepared record.
    pub misses: u64,
    /// Prepared cache bytes read for misses.
    pub bytes_read: u64,
}

/// Bounded expert cache for the Qwen3.6 NVFP4 MoE path.
pub struct Qwen36ExpertPager {
    cache_dir: PathBuf,
    gate_up: Sm121W4A16GateUp,
    down: Vec<Sm12xFp4DeviceGemmWeight>,
    down_tiles: DeviceBuffer<DeviceAddress<u8>>,
    down_scales: DeviceBuffer<DeviceAddress<u32>>,
    down_input_scales: DeviceBuffer<f32>,
    down_alphas: DeviceBuffer<f32>,
    gate_up_unity_alphas: DeviceBuffer<f32>,
    slots: ExpertSlotCache,
    down_input_scales_host: Vec<f32>,
    down_alphas_host: Vec<f32>,
    hidden: usize,
    intermediate: usize,
    top_k: usize,
    stats: Qwen36PagingStats,
    uploads: ExpertUploadCoordinator,
    staging_pool: Vec<Qwen36ExpertStaging>,
    paging_metrics: ExpertPagingMetricHandle,
}

struct Qwen36ExpertStaging {
    slot: usize,
    gate_weight: PinnedHostBuffer<u8>,
    gate_scale: PinnedHostBuffer<u8>,
    gate_global_scale: PinnedHostBuffer<f32>,
    down_tiles: PinnedHostBuffer<u8>,
    down_scales: PinnedHostBuffer<u32>,
    down_input_scale: PinnedHostBuffer<f32>,
    down_alpha: PinnedHostBuffer<f32>,
}

struct Qwen36PreparedExpertRecord {
    gate: Sm121W4A16HostWeight,
    down: Sm12xFp4GemmWeight,
    bytes: usize,
}

struct Qwen36ExpertRecordSource<'a> {
    cache_dir: &'a std::path::Path,
}

impl ExpertRecordSource for Qwen36ExpertRecordSource<'_> {
    type Record = Qwen36PreparedExpertRecord;

    fn read_record(&self, expert: usize) -> Result<Self::Record> {
        let gate_path = gate_up_path(self.cache_dir, expert);
        let down_path = down_path(self.cache_dir, expert);
        let gate = Sm121W4A16HostWeight::read_cache_file(&gate_path)?;
        let down = Sm12xFp4GemmWeight::read_cache_file(&down_path)?;
        let bytes = std::fs::metadata(&gate_path)
            .map_err(|error| Error::Format {
                label: "Qwen3.6 expert pager",
                detail: format!("failed to inspect {}: {error}", gate_path.display()),
            })?
            .len() as usize
            + std::fs::metadata(&down_path)
                .map_err(|error| Error::Format {
                    label: "Qwen3.6 expert pager",
                    detail: format!("failed to inspect {}: {error}", down_path.display()),
                })?
                .len() as usize;
        Ok(Qwen36PreparedExpertRecord { gate, down, bytes })
    }
}

impl Qwen36ExpertStaging {
    fn new(gate_up: usize, hidden: usize, intermediate: usize) -> Result<Self> {
        let down_tiles = (hidden / 16) * (intermediate / 64) * 512;
        let down_scales = (hidden / 16) * (intermediate / 64);
        Ok(Self {
            slot: 0,
            gate_weight: PinnedHostBuffer::zeroed(gate_up * hidden / 2)?,
            gate_scale: PinnedHostBuffer::zeroed(gate_up * hidden / 16)?,
            gate_global_scale: PinnedHostBuffer::zeroed(1)?,
            down_tiles: PinnedHostBuffer::zeroed(down_tiles)?,
            down_scales: PinnedHostBuffer::zeroed(down_scales)?,
            down_input_scale: PinnedHostBuffer::zeroed(1)?,
            down_alpha: PinnedHostBuffer::zeroed(1)?,
        })
    }
}

enum Qwen36GateUpStorage {
    CutlassW4A4,
    Paged,
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
    weights: DeviceBuffer<DeviceAddress<u8>>,
    scales: DeviceBuffer<DeviceAddress<f32>>,
}

struct Qwen36Fp8Experts {
    gate: Qwen36Fp8ExpertTable,
    up: Qwen36Fp8ExpertTable,
    down: Qwen36Fp8ExpertTable,
}

enum Qwen36SharedExpertStorage {
    Nvfp4(Qwen36SharedExpert),
    Fp8 {
        gate_up: Fp8Linear,
        down: Fp8Linear,
    },
    Bf16 {
        gate_up: Bf16Linear,
        down: Bf16Linear,
    },
}

struct Qwen36GroupedMoeWeights {
    _gate_up: Vec<ModelOptCublasLtWeight>,
    _down: Vec<ModelOptCublasLtWeight>,
    gate_up_values: DeviceBuffer<DeviceAddress<u8>>,
    gate_up_scales: DeviceBuffer<DeviceAddress<u8>>,
    _gate_up_alphas: DeviceBuffer<f32>,
    gate_up_alpha_table: DeviceBuffer<DeviceAddress<f32>>,
    down_values: DeviceBuffer<DeviceAddress<u8>>,
    down_scales: DeviceBuffer<DeviceAddress<u8>>,
    _down_alphas: DeviceBuffer<f32>,
    down_alpha_table: DeviceBuffer<DeviceAddress<f32>>,
}

impl Qwen36GroupedMoeWeights {
    fn new(
        gate_up: Vec<ModelOptCublasLtWeight>,
        down: Vec<ModelOptCublasLtWeight>,
    ) -> Result<Self> {
        let gate_up_values = gate_up
            .iter()
            .map(|weight| weight.matrix().values_address())
            .collect::<Vec<_>>();
        let gate_up_scales = gate_up
            .iter()
            .map(|weight| weight.matrix().scales_address())
            .collect::<Vec<_>>();
        let down_values = down
            .iter()
            .map(|weight| weight.matrix().values_address())
            .collect::<Vec<_>>();
        let down_scales = down
            .iter()
            .map(|weight| weight.matrix().scales_address())
            .collect::<Vec<_>>();
        let gate_up_alphas = DeviceBuffer::from_host(
            &gate_up
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let gate_up_alpha_table = scalar_pointer_table(&gate_up_alphas)?;
        let down_alphas = DeviceBuffer::from_host(
            &down
                .iter()
                .map(ModelOptCublasLtWeight::weight_scale_2)
                .collect::<Vec<_>>(),
        )?;
        let down_alpha_table = scalar_pointer_table(&down_alphas)?;
        Ok(Self {
            gate_up_values: DeviceBuffer::from_host(&gate_up_values)?,
            gate_up_scales: DeviceBuffer::from_host(&gate_up_scales)?,
            _gate_up_alphas: gate_up_alphas,
            gate_up_alpha_table,
            down_values: DeviceBuffer::from_host(&down_values)?,
            down_scales: DeviceBuffer::from_host(&down_scales)?,
            _down_alphas: down_alphas,
            down_alpha_table,
            _gate_up: gate_up,
            _down: down,
        })
    }
}

fn scalar_pointer_table(values: &DeviceBuffer<f32>) -> Result<DeviceBuffer<DeviceAddress<f32>>> {
    let base = values.cuda_address();
    let pointers = (0..values.len())
        .map(|index| base.offset(index))
        .collect::<Result<Vec<_>>>()?;
    DeviceBuffer::from_host(&pointers)
}

/// A device-resident NVFP4 linear weight for W4A16 execution.
///
/// Stores the raw ModelOpt packed E2M1 weight and UE4M3 per-block scales
/// (not cuBLASLt-repacked), plus the scalar `weight_scale_2`. For W4A16,
/// activations stay f32; the GEMM dequantizes weights on the fly.
struct Nvfp4DeviceLinear {
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    cublaslt_weight: ModelOptCublasLtWeight,
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
    w4a16_gate_up_output: DeviceBuffer<f32>,
    w4a16_gate_up_table: DeviceBuffer<*const f32>,
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

/// Two-row MoE verifier workspace that retains canonical per-row expert math.
pub(crate) struct Qwen36ExactMoePairWorkspace {
    inputs: Vec<DeviceBuffer<f32>>,
    rows: Vec<Qwen36MoeWorkspace>,
    router_logits: DeviceBuffer<f32>,
    route_indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    zero_hidden: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

pub(crate) struct Qwen36MoeProbeSnapshot {
    pub(crate) router_logits: Vec<f32>,
    pub(crate) route_indices: Vec<u32>,
    pub(crate) route_weights: Vec<f32>,
    pub(crate) gate_up_input_values: Vec<u8>,
    pub(crate) gate_up_input_scales: Vec<u8>,
    pub(crate) routed_output: Vec<f32>,
    pub(crate) routed_gate_up: Vec<f32>,
    pub(crate) repeated_routed_gate_up: Option<Vec<f32>>,
    pub(crate) oracle_routed_gate_up: Option<Vec<f32>>,
    pub(crate) routed_down_slots: Vec<f32>,
    pub(crate) shared_gate_logits: Vec<f32>,
    pub(crate) shared_output: Vec<f32>,
    pub(crate) final_output: Vec<f32>,
}

/// Device-ready dense SwiGLU feed-forward weights used by Qwen3.8.
pub struct Qwen36DenseMlpWeights {
    gate_up: Qwen36Linear,
    down: Qwen36Linear,
}

/// Mutable one-token workspace for a dense SwiGLU feed-forward block.
pub struct Qwen36DenseMlpWorkspace {
    gate_up: DeviceBuffer<f32>,
    gate_up_fp8_input: DeviceBuffer<u8>,
    gate_up_fp8_input_scale: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    down_fp8_input: DeviceBuffer<u8>,
    down_fp8_input_scale: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

/// Feed-forward weights for a Qwen3.5-family layer.
pub enum Qwen36LayerFfnWeights {
    /// Routed and shared experts used by Qwen3.5/3.6 MoE checkpoints.
    Moe(Box<Qwen36MoeWeights>),
    /// Dense SwiGLU used by Qwen3.8 dense checkpoints.
    Dense(Box<Qwen36DenseMlpWeights>),
}

/// Mutable one-token feed-forward workspace for a Qwen3.5-family layer.
pub enum Qwen36LayerFfnWorkspace {
    /// Routed and shared-expert workspace.
    Moe(Box<Qwen36MoeWorkspace>),
    /// Dense SwiGLU workspace.
    Dense(Qwen36DenseMlpWorkspace),
}

impl Deref for Qwen36LayerFfnWeights {
    type Target = Qwen36MoeWeights;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Moe(weights) => weights,
            Self::Dense(_) => panic!("MoE diagnostics are unavailable for a dense Qwen FFN"),
        }
    }
}

impl DerefMut for Qwen36LayerFfnWeights {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Moe(weights) => weights,
            Self::Dense(_) => panic!("MoE diagnostics are unavailable for a dense Qwen FFN"),
        }
    }
}

impl Deref for Qwen36LayerFfnWorkspace {
    type Target = Qwen36MoeWorkspace;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Moe(workspace) => workspace,
            Self::Dense(_) => panic!("MoE diagnostics are unavailable for a dense Qwen FFN"),
        }
    }
}

impl DerefMut for Qwen36LayerFfnWorkspace {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Moe(workspace) => workspace,
            Self::Dense(_) => panic!("MoE diagnostics are unavailable for a dense Qwen FFN"),
        }
    }
}

impl Qwen36LayerFfnWorkspace {
    fn output(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Moe(workspace) => &workspace.ffn_out,
            Self::Dense(workspace) => &workspace.output,
        }
    }
}

struct Sm12xGateUpWorkspace {
    b_tiles: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u32>,
    _outputs: Vec<F32Matrix>,
    c: DeviceBuffer<*const f32>,
    d: DeviceBuffer<*mut f32>,
    indexed_d: DeviceBuffer<DeviceAddress<f32>>,
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
        let mut indexed_d = Vec::with_capacity(groups);
        for _ in 0..groups {
            let mut output = F32Matrix::zeroed(out_features, 1)?;
            c_ptrs.push(output.data_ptr());
            d_ptrs.push(output.data_mut_ptr());
            indexed_d.push(output.data_address());
            outputs.push(output);
        }
        Ok(Self {
            b_tiles: DeviceBuffer::zeroed(b_groups * (in_features / 64) * 512)?,
            b_scales: DeviceBuffer::zeroed(b_groups * (in_features / 64))?,
            _outputs: outputs,
            c: DeviceBuffer::from_host(&c_ptrs)?,
            d: DeviceBuffer::from_host(&d_ptrs)?,
            indexed_d: DeviceBuffer::from_host(&indexed_d)?,
            groups,
        })
    }

    fn device_bytes(&self) -> usize {
        self.b_tiles.device_bytes()
            + self.b_scales.device_bytes()
            + self
                ._outputs
                .iter()
                .map(F32Matrix::device_bytes)
                .sum::<usize>()
            + self.c.device_bytes()
            + self.d.device_bytes()
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

#[derive(Clone, Copy)]
struct Qwen36ParallelMoe<'a> {
    shared_stream: &'a CudaStream,
    fork: &'a CudaEvent,
    join: &'a CudaEvent,
}

impl Qwen36ExpertPager {
    /// Allocates `capacity` resident slots and retains only per-expert scalar metadata.
    pub fn load(model: &Qwen36Model, layer: usize, capacity: usize) -> Result<Self> {
        Self::load_from_checkpoint(
            &model.checkpoint,
            &model.manifest,
            &model.artifact_dir,
            layer,
            capacity,
            false,
        )
    }

    fn load_from_checkpoint(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        artifact_root: &std::path::Path,
        layer: usize,
        capacity: usize,
        cache_prepared: bool,
    ) -> Result<Self> {
        if layer >= manifest.layers {
            return Err(Error::Shape {
                label: "Qwen3.6 expert pager layer",
                expected: format!("layer < {}", manifest.layers),
                actual: layer.to_string(),
            });
        }
        let (experts, top_k, intermediate) = match manifest.ffn {
            QwenFfnConfig::Moe {
                experts,
                experts_per_token,
                expert_intermediate,
                ..
            } => (experts, experts_per_token, expert_intermediate),
            QwenFfnConfig::Dense => {
                return Err(Error::Format {
                    label: "Qwen3.6 expert pager",
                    detail: "expected MoE model".to_string(),
                });
            }
        };
        if capacity < top_k || capacity > experts {
            return Err(Error::Shape {
                label: "Qwen3.6 expert pager capacity",
                expected: format!("{top_k}..={experts} slots"),
                actual: capacity.to_string(),
            });
        }

        let cache_dir = if cache_prepared {
            prepared_layer_dir(artifact_root, layer)
        } else {
            ensure_layer_cache(checkpoint, manifest, artifact_root, layer)?
        };
        let prefix = format!("{}.layers.{layer}.mlp.experts", manifest.tensor_prefix);
        let mut down_input_scales_host = Vec::with_capacity(experts);
        let mut down_alphas_host = Vec::with_capacity(experts);
        for expert in 0..experts {
            let (weight_scale, input_scale) =
                checkpoint.load_nvfp4_scales(&format!("{prefix}.{expert}.down_proj"))?;
            down_input_scales_host.push(input_scale);
            down_alphas_host.push(weight_scale * input_scale);
        }

        let gate_up =
            Sm121W4A16GateUp::new_empty_slots(capacity, intermediate * 2, manifest.hidden)?;
        let mut down = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            down.push(Sm12xFp4DeviceGemmWeight::zeroed(
                manifest.hidden,
                intermediate,
            )?);
        }
        let down_tiles = DeviceBuffer::from_host(
            &down
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::tiles_address)
                .collect::<Vec<_>>(),
        )?;
        let down_scales = DeviceBuffer::from_host(
            &down
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::scales_address)
                .collect::<Vec<_>>(),
        )?;
        let down_input_scales = DeviceBuffer::zeroed(capacity)?;
        let down_alphas = DeviceBuffer::zeroed(capacity)?;
        let expert_device_bytes = gate_up.expert_device_bytes()
            + down
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::device_bytes)
                .sum::<usize>()
            + down_input_scales.device_bytes()
            + down_alphas.device_bytes();
        let slots = ExpertSlotCache::new(experts, capacity, top_k)?;
        let uploads = ExpertUploadCoordinator::new()?;
        let staging_pool = (0..top_k)
            .map(|_| Qwen36ExpertStaging::new(intermediate * 2, manifest.hidden, intermediate))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cache_dir,
            gate_up,
            down,
            down_tiles,
            down_scales,
            down_input_scales,
            down_alphas,
            gate_up_unity_alphas: DeviceBuffer::from_host(&vec![1.0; capacity])?,
            slots,
            down_input_scales_host,
            down_alphas_host,
            hidden: manifest.hidden,
            intermediate,
            top_k,
            stats: Qwen36PagingStats::default(),
            uploads,
            staging_pool,
            paging_metrics: ExpertPagingMetricHandle::new(capacity, expert_device_bytes),
        })
    }

    /// Resolves one token's routed experts and enqueues any miss uploads.
    pub fn resolve(
        &mut self,
        expert_ids: &[u32],
        device_expert_ids: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<Qwen36PageResolution> {
        if device_expert_ids.len() != self.top_k {
            return Err(Error::Shape {
                label: "Qwen3.6 expert pager route",
                expected: format!("{} device expert IDs", self.top_k),
                actual: format!("{} device expert IDs", device_expert_ids.len()),
            });
        }
        if let Some(timing) = self.uploads.wait_for_staging_reuse()? {
            self.paging_metrics.record_page_upload(timing.upload);
            self.paging_metrics.record_staging_wait(timing.staging_wait);
        }
        let plan = self.slots.plan(expert_ids)?;
        let slots = plan.slots;
        let hits = plan.hits;
        let evictions = plan.evictions;
        let resident_slots = plan.resident_slots;
        let pending = plan.misses;
        let misses = pending.len();

        let mut bytes_read = 0;
        let resolve_started = (!pending.is_empty()).then(Instant::now);
        if !pending.is_empty() {
            self.uploads.begin(stream)?;

            let source = Qwen36ExpertRecordSource {
                cache_dir: &self.cache_dir,
            };
            let read_started = Instant::now();
            let loaded = read_expert_misses(&source, &pending)?;
            self.paging_metrics.record_page_read(read_started.elapsed());

            for (staged, loaded) in self.staging_pool.iter_mut().zip(loaded) {
                let expert = loaded.expert;
                let Qwen36PreparedExpertRecord { gate, down, bytes } = loaded.record;
                bytes_read += bytes;
                let down_tiles = down.tile_bytes();
                staged.slot = loaded.slot;
                staged.gate_weight.copy_from_slice(&gate.packed_weight)?;
                staged.gate_scale.copy_from_slice(&gate.weight_scale)?;
                staged
                    .gate_global_scale
                    .copy_from_slice(&[gate.global_scale])?;
                staged.down_tiles.copy_from_slice(&down_tiles)?;
                staged.down_scales.copy_from_slice(down.scale_words())?;
                staged
                    .down_input_scale
                    .copy_from_slice(&[self.down_input_scales_host[expert]])?;
                staged
                    .down_alpha
                    .copy_from_slice(&[self.down_alphas_host[expert]])?;
            }
            for staged in self.staging_pool.iter().take(misses) {
                self.gate_up.load_slot_from_pinned_on_stream(
                    staged.slot,
                    &staged.gate_weight,
                    &staged.gate_scale,
                    &staged.gate_global_scale,
                    self.uploads.stream(),
                )?;
                self.down[staged.slot].copy_from_pinned_on_stream(
                    &staged.down_tiles,
                    &staged.down_scales,
                    self.uploads.stream(),
                )?;
                self.down_input_scales.copy_range_from_pinned_on_stream(
                    staged.slot,
                    &staged.down_input_scale,
                    self.uploads.stream(),
                )?;
                self.down_alphas.copy_range_from_pinned_on_stream(
                    staged.slot,
                    &staged.down_alpha,
                    self.uploads.stream(),
                )?;
            }
            self.slots.enqueue_mapping_upload(self.uploads.stream())?;
            self.uploads.finish(stream)?;
        }

        self.slots.remap_on_stream(device_expert_ids, stream)?;
        self.stats.hits += hits as u64;
        self.stats.misses += misses as u64;
        self.stats.bytes_read += bytes_read as u64;
        self.paging_metrics.record_cache_activity(
            hits,
            misses,
            evictions,
            bytes_read,
            resident_slots,
        );
        if let Some(started) = resolve_started {
            self.paging_metrics.record_page_resolve(started.elapsed());
        }
        Ok(Qwen36PageResolution {
            slots,
            hits,
            misses,
            bytes_read,
        })
    }

    /// Runs the routed expert computation using the most recently resolved slots.
    pub fn run_routed<'a>(
        &self,
        workspace: &'a mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if ffn_norm.len() != self.hidden || workspace.route.weights.len() != self.top_k {
            return Err(Error::Shape {
                label: "Qwen3.6 paged routed MoE inputs",
                expected: format!("hidden={} route_weights={}", self.hidden, self.top_k),
                actual: format!(
                    "hidden={} route_weights={}",
                    ffn_norm.len(),
                    workspace.route.weights.len()
                ),
            });
        }
        let slot_indices = self.slots.slot_indices();
        self.gate_up.run_on_stream(
            slot_indices,
            ffn_norm,
            workspace.w4a16_gate_up_output.output(),
            stream,
        )?;
        moe_silu_quantize_slots_on_stream(
            slot_indices,
            &workspace.w4a16_gate_up_table,
            &mut workspace.sm12x_down.b_tiles,
            &mut workspace.sm12x_down.b_scales,
            &self.down_input_scales,
            &self.gate_up_unity_alphas,
            self.intermediate,
            self.top_k,
            stream,
        )?;
        indexed_grouped_gemv_on_stream(
            slot_indices,
            &self.down_tiles,
            &self.down_scales,
            self.slots.capacity(),
            &workspace.sm12x_down.b_tiles,
            &workspace.sm12x_down.b_scales,
            &workspace.sm12x_down.indexed_d,
            self.hidden / 16,
            self.intermediate / 64,
            self.top_k,
            stream,
        )?;
        fill_f32_into_on_stream(workspace.moe_out.output(), 0.0, stream)?;
        moe_weighted_accumulate_slots_f32_on_stream(
            slot_indices,
            &workspace.route.weights,
            &workspace.sm12x_down.c,
            &self.down_alphas,
            workspace.moe_out.inout(),
            stream,
        )?;
        Ok(&workspace.moe_out)
    }

    /// Returns device bytes retained by the resident expert slots.
    pub fn expert_device_bytes(&self) -> usize {
        self.gate_up.expert_device_bytes()
            + self
                .down
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::device_bytes)
                .sum::<usize>()
            + self.down_input_scales.device_bytes()
            + self.down_alphas.device_bytes()
    }

    /// Returns cumulative lookup and miss-I/O counters.
    pub fn stats(&self) -> Qwen36PagingStats {
        self.stats
    }
}

impl Qwen36MoeWeights {
    /// Loads the MoE + shared-expert FFN for layer `layer`.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        artifact_root: &std::path::Path,
        layer: usize,
        cache_prepared: bool,
    ) -> Result<Self> {
        Self::load_with_down_storage(
            checkpoint,
            manifest,
            artifact_root,
            layer,
            cache_prepared,
            true,
        )
    }

    /// Loads resident experts once in checkpoint layout without an SM12x down duplicate.
    pub(crate) fn load_checkpoint_layout(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        artifact_root: &std::path::Path,
        layer: usize,
    ) -> Result<Self> {
        Self::load_with_down_storage(checkpoint, manifest, artifact_root, layer, false, false)
    }

    fn load_with_down_storage(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        artifact_root: &std::path::Path,
        layer: usize,
        cache_prepared: bool,
        request_sm12x_down: bool,
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
        let sm12x_cache_dir = if request_sm12x_down {
            if cache_prepared {
                prepared_layer_dir(artifact_root, layer)
            } else {
                ensure_layer_cache(checkpoint, manifest, artifact_root, layer)?
            }
        } else {
            PathBuf::new()
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

        let sm12x_down_cache_complete = request_sm12x_down
            && (0..experts).all(|expert_idx| down_path(&sm12x_cache_dir, expert_idx).is_file());
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
        let mut gate_up_alphas = Vec::with_capacity(experts);
        let mut grouped_gate_up = Vec::with_capacity(experts);
        let mut grouped_down = Vec::with_capacity(experts);
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
            gate_up_alphas.push(weight.weight_scale_2);
            grouped_gate_up.push(ModelOptCublasLtWeight::from_modelopt(&weight)?);

            match storage_plan.down {
                Qwen36DownStorage::Legacy => {
                    let weight =
                        checkpoint.load_nvfp4_linear(&format!("{}.down_proj", expert.prefix))?;
                    down_input_scales.push(weight.input_scale);
                    down_alphas.push(weight.weight_scale_2 * weight.input_scale);
                    grouped_down.push(ModelOptCublasLtWeight::from_modelopt(&weight)?);
                }
                Qwen36DownStorage::Sm12x => {
                    let weight =
                        checkpoint.load_nvfp4_linear(&format!("{}.down_proj", expert.prefix))?;
                    down_input_scales.push(weight.input_scale);
                    down_alphas.push(weight.weight_scale_2 * weight.input_scale);
                    grouped_down.push(ModelOptCublasLtWeight::from_modelopt(&weight)?);
                }
                Qwen36DownStorage::Fp8 => {
                    unreachable!("NVFP4 loader cannot select FP8 down storage")
                }
            }

            if storage_plan.down == Qwen36DownStorage::Sm12x {
                let path = down_path(&sm12x_cache_dir, expert_idx);
                let weight = Sm12xFp4GemmWeight::read_cache_file(&path)?.to_device()?;
                sm12x_down_m_tiles = weight.m_tiles();
                sm12x_down_k_tiles = weight.k_tiles();
                sm12x_down_tile_ptrs.push(weight.tiles_address());
                sm12x_down_scale_ptrs.push(weight.scales_address());
                sm12x_down.push(weight);
            }
        }

        for (expert_idx, weight) in grouped_gate_up.iter().enumerate() {
            gate_up_grouped_value_ptrs[expert_idx] = weight.matrix().values_ptr();
            gate_up_grouped_scale_ptrs[expert_idx] = weight.matrix().scales_ptr();
        }
        for (expert_idx, weight) in grouped_down.iter().enumerate() {
            down_grouped_value_ptrs[expert_idx] = weight.matrix().values_ptr();
            down_grouped_scale_ptrs[expert_idx] = weight.matrix().scales_ptr();
        }
        let gate_up_storage = Qwen36GateUpStorage::CutlassW4A4;
        let grouped = if grouped_gate_up.len() == experts && grouped_down.len() == experts {
            Some(Qwen36GroupedMoeWeights::new(grouped_gate_up, grouped_down)?)
        } else {
            None
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
            shared_gate_up_input_scale: None,
            gate_up_alphas: DeviceBuffer::from_host(&gate_up_alphas)?,
        };

        let (shared, _) = load_shared_expert(checkpoint, &prefix, manifest.hidden)?;

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
            gate_up_unity_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
            storage_plan,
            gate_up_storage,
            grouped,
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
            expert_pager: std::cell::RefCell::new(None),
        })
    }

    fn load_paged(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        artifact_root: &std::path::Path,
        layer: usize,
        cache_prepared: bool,
        capacity: usize,
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
                    label: "Qwen3.6 paged MoE FFN",
                    detail: "expected MoE config, got Dense".to_string(),
                });
            }
        };
        let prefix = format!("{}.layers.{layer}.mlp", manifest.tensor_prefix);
        let first_gate = format!("{prefix}.experts.0.gate_proj");
        let uses_nvfp4 = checkpoint.contains_tensor(&format!("{first_gate}.weight_scale_2"))
            || checkpoint.contains_tensor(&format!("{first_gate}.weight_global_scale"));
        if !uses_nvfp4 {
            return Err(Error::Format {
                label: "Qwen3.6 paged MoE FFN",
                detail: format!("layer {layer} uses FP8 routed experts"),
            });
        }

        let router = Bf16Linear::load(
            checkpoint,
            &format!("{prefix}.gate.weight"),
            experts,
            manifest.hidden,
        )?;
        let pager = Qwen36ExpertPager::load_from_checkpoint(
            checkpoint,
            manifest,
            artifact_root,
            layer,
            capacity,
            cache_prepared,
        )?;

        let (shared, _) = load_shared_expert(checkpoint, &prefix, manifest.hidden)?;
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
            gate_up_unity_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
            storage_plan: Qwen36MoeStoragePlan {
                down: Qwen36DownStorage::Sm12x,
            },
            gate_up_storage: Qwen36GateUpStorage::Paged,
            grouped: None,
            fp8_experts: None,
            shared,
            shared_gate,
            _sm12x_down: Vec::new(),
            sm12x_down_tiles: None,
            sm12x_down_scales: None,
            sm12x_down_m_tiles: manifest.hidden / 16,
            sm12x_down_k_tiles: expert_intermediate / 64,
            num_experts: experts,
            experts_per_token,
            expert_intermediate,
            norm_topk_prob,
            expert_pager: std::cell::RefCell::new(Some(pager)),
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
            gate_up_unity_alphas: DeviceBuffer::from_host(&vec![1.0; experts])?,
            storage_plan: Qwen36MoeStoragePlan {
                down: Qwen36DownStorage::Fp8,
            },
            gate_up_storage: Qwen36GateUpStorage::Fp8,
            grouped: None,
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
            expert_pager: std::cell::RefCell::new(None),
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

    /// Installs a bounded expert pager for this layer's routed path.
    pub fn enable_expert_paging(
        &mut self,
        model: &Qwen36Model,
        layer: usize,
        capacity: usize,
    ) -> Result<()> {
        *self.expert_pager.get_mut() = Some(Qwen36ExpertPager::load(model, layer, capacity)?);
        Ok(())
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
        if matches!(self.gate_up_storage, Qwen36GateUpStorage::CutlassW4A4) {
            quantize_nvfp4_col_major_f32_device_into_on_stream(
                manifest.hidden,
                1,
                ffn_norm,
                &mut workspace.gate_up_input,
                1.0,
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
            Qwen36GateUpStorage::CutlassW4A4 => self.run_grouped_gate_up_only(workspace, stream),
            Qwen36GateUpStorage::Paged => Err(Error::Format {
                label: "Qwen3.6 paged routed gate/up",
                detail: "resolve resident expert slots before routed execution".to_string(),
            }),
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
                fp8_moe_grouped_gate_up_addressed_f32_into_on_stream(
                    &workspace.route.indices,
                    &workspace.fp8_hidden_input,
                    &workspace.fp8_hidden_input_scale,
                    &fp8.gate.weights,
                    &fp8.gate.scales,
                    &fp8.up.weights,
                    &fp8.up.scales,
                    workspace.w4a16_gate_up_output.output(),
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
        self.router
            .run_into(ffn_norm, &mut workspace.router_logits, stream)?;
        workspace
            .route
            .run_topk(&workspace.router_logits, self.norm_topk_prob, stream)?;
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            manifest.hidden,
            1,
            ffn_norm,
            &mut workspace.gate_up_input,
            1.0,
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
        grouped_gate_up.run_indexed_gate_up_device_route(
            &workspace.route,
            &self.expert_ptrs,
            &workspace.gate_up_input,
            stream,
        )?;
        Ok(())
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
            Qwen36GateUpStorage::CutlassW4A4 => {
                &workspace
                    .grouped_gate_up
                    .as_ref()
                    .ok_or_else(|| Error::Format {
                        label: "Qwen3.6 grouped down",
                        detail: "grouped gate/up workspace is unavailable".to_string(),
                    })?
                    .c
            }
            Qwen36GateUpStorage::Fp8 => &workspace.w4a16_gate_up_table,
            Qwen36GateUpStorage::Paged => {
                return Err(Error::Format {
                    label: "Qwen3.6 paged grouped down",
                    detail: "use the expert pager routed path".to_string(),
                });
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
            let gate_up_alpha_table = &self.gate_up_unity_alphas;
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
        eider_cuda::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
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
                &workspace.sm12x_down.indexed_d,
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
                &workspace.sm12x_down.indexed_d,
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
            Qwen36SharedExpertStorage::Bf16 { gate_up, .. } => {
                gate_up.run_into(ffn_norm, &mut workspace.shared_gate_up_output, stream)
            }
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
            Qwen36SharedExpertStorage::Bf16 { down, .. } => down.run_into(
                &workspace.shared_activated,
                &mut workspace.shared_output,
                stream,
            ),
        }
    }

    fn enqueue_shared_branch(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_shared_gate_up(workspace, ffn_norm, stream)?;
        silu_mul_halves_f32_into_on_stream(
            &workspace.shared_gate_up_output,
            workspace.shared_activated.output(),
            self.expert_intermediate,
            stream,
        )?;
        self.run_shared_down(workspace, stream)?;
        self.shared_gate
            .run_into(ffn_norm, &mut workspace.shared_gate_logits, stream)
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

    /// Runs two rows with one exact router/top-k batch and canonical per-row
    /// routed and shared expert kernels.
    pub(crate) fn run_exact_pair<'a>(
        &self,
        workspace: &'a mut Qwen36ExactMoePairWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if ffn_norm.len() < 2 * manifest.hidden
            || self.storage_plan.down != Qwen36DownStorage::Legacy
            || !matches!(self.gate_up_storage, Qwen36GateUpStorage::CutlassW4A4)
            || self.grouped.is_none()
            || self.expert_pager.borrow().is_some()
        {
            return Err(Error::Format {
                label: "Qwen exact MoE pair",
                detail: "requires two rows of resident W4A4 experts with legacy down storage"
                    .to_string(),
            });
        }

        bf16_linear_two_rows_f32_into_on_stream(
            ffn_norm,
            &self.router.weight,
            workspace.router_logits.output(),
            self.num_experts,
            manifest.hidden,
            stream,
        )?;
        moe_topk_f32_batch_into_on_stream(
            &workspace.router_logits,
            workspace.route_indices.output(),
            workspace.route_weights.output(),
            2,
            self.num_experts,
            self.experts_per_token,
            self.norm_topk_prob,
            stream,
        )?;

        for row in 0..2 {
            let row_workspace = &mut workspace.rows[row];
            workspace.inputs[row].copy_range_from_device_on_stream(
                0,
                ffn_norm,
                row * manifest.hidden,
                manifest.hidden,
                stream,
            )?;
            row_workspace
                .route
                .indices
                .copy_range_from_device_on_stream(
                    0,
                    &workspace.route_indices,
                    row * self.experts_per_token,
                    self.experts_per_token,
                    stream,
                )?;
            row_workspace
                .route
                .weights
                .copy_range_from_device_on_stream(
                    0,
                    &workspace.route_weights,
                    row * self.experts_per_token,
                    self.experts_per_token,
                    stream,
                )?;
            quantize_nvfp4_col_major_f32_device_into_on_stream(
                manifest.hidden,
                1,
                &workspace.inputs[row],
                &mut row_workspace.gate_up_input,
                1.0,
                stream,
            )?;
            self.run_grouped_gate_up_only(row_workspace, stream)?;
            let gate_up_table = &row_workspace
                .grouped_gate_up
                .as_ref()
                .expect("resident W4A4 gate/up workspace")
                .c;
            let grouped_down = row_workspace
                .grouped_down
                .as_mut()
                .expect("resident W4A4 down workspace");
            eider_cuda::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
                MoeSiluQuantizeSlotBuffers {
                    indices: &row_workspace.route.indices,
                    gate_up_table,
                    packed_table: grouped_down.input_values_mut.output(),
                    scales_table: grouped_down.input_scales_mut.output(),
                    input_scale_table: &self.expert_ptrs.down_input_scales,
                    gate_up_alpha_table: &self.gate_up_unity_alphas,
                },
                grouped_down.inputs[0].rows,
                stream,
            )?;
            fill_f32_into_on_stream(row_workspace.moe_out.output(), 0.0, stream)?;
            if !grouped_down.run_prequantized_device_route(
                &row_workspace.route,
                &self.expert_ptrs,
                &mut row_workspace.moe_out,
                stream,
            )? {
                return Err(Error::Format {
                    label: "Qwen exact MoE pair",
                    detail: "grouped down rejected a canonical route".to_string(),
                });
            }
            self.run_shared_gate_up(row_workspace, &workspace.inputs[row], stream)?;
            silu_mul_halves_f32_into_on_stream(
                &row_workspace.shared_gate_up_output,
                row_workspace.shared_activated.output(),
                self.expert_intermediate,
                stream,
            )?;
            self.run_shared_down(row_workspace, stream)?;
            self.shared_gate.run_into(
                &workspace.inputs[row],
                &mut row_workspace.shared_gate_logits,
                stream,
            )?;
            qwen36_ffn_finalize_f32_into_on_stream(
                &row_workspace.moe_out,
                &row_workspace.shared_gate_logits,
                &row_workspace.shared_output,
                &workspace.zero_hidden,
                row_workspace.ffn_residual.output(),
                stream,
            )?;
            std::mem::swap(&mut row_workspace.ffn_out, &mut row_workspace.ffn_residual);
            workspace.output.copy_range_from_device_on_stream(
                row * manifest.hidden,
                &row_workspace.ffn_out,
                0,
                manifest.hidden,
                stream,
            )?;
        }
        Ok(&workspace.output)
    }

    pub(crate) fn probe_repeat_exact_pair_gate_up(
        &self,
        workspace: &mut Qwen36ExactMoePairWorkspace,
        stream: &CudaStream,
    ) -> Result<Vec<Qwen36MoeProbeSnapshot>> {
        let mut snapshots = workspace.probe_snapshots(
            self.num_experts,
            self.experts_per_token,
            workspace.inputs[0].len(),
            stream,
        )?;
        for (row, snapshot) in snapshots.iter_mut().enumerate() {
            self.run_grouped_gate_up_only(&mut workspace.rows[row], stream)?;
            snapshot.repeated_routed_gate_up = Some(
                workspace.rows[row]
                    .grouped_gate_up
                    .as_ref()
                    .expect("exact pair grouped gate/up")
                    .copy_outputs_to_host(stream)?,
            );
        }
        Ok(snapshots)
    }

    pub(crate) fn probe_repeat_workspace_gate_up(
        &self,
        workspace: &mut Qwen36MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<Qwen36MoeProbeSnapshot> {
        let mut snapshot = workspace.probe_snapshot(stream)?;
        self.run_grouped_gate_up_only(workspace, stream)?;
        snapshot.repeated_routed_gate_up = Some(
            workspace
                .grouped_gate_up
                .as_ref()
                .expect("grouped gate/up workspace")
                .copy_outputs_to_host(stream)?,
        );
        Ok(snapshot)
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
        lt: &CublasLt,
        workspace: &'a mut Qwen36MoeWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
        profile: Option<&mut QwenDecodeProfile>,
        gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
    ) -> Result<Qwen36MoeStep<'a>> {
        self.run_one_token_impl(
            lt, workspace, manifest, ffn_norm, residual, stream, None, profile, gpu_probe,
        )
    }

    #[allow(clippy::needless_option_as_deref, clippy::too_many_arguments)]
    fn run_one_token_impl<'a>(
        &'a self,
        _lt: &CublasLt,
        workspace: &'a mut Qwen36MoeWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
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

        if let Some(parallel) = parallel_moe {
            parallel.fork.record_on_stream(stream)?;
            parallel.shared_stream.wait_event(parallel.fork)?;
            self.enqueue_shared_branch(workspace, ffn_norm, parallel.shared_stream)?;
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
        let used_pager = self.expert_pager.borrow().is_some();
        let used_grouped = if used_pager {
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
            let indices = workspace.route.indices.copy_to_host(stream)?.into_vec();
            let mut pager = self.expert_pager.borrow_mut();
            let pager = pager.as_mut().expect("expert pager checked above");
            pager.resolve(&indices, &workspace.route.indices, stream)?;
            pager.run_routed(workspace, ffn_norm, stream)?;
            true
        } else if let Some(fp8) = &self.fp8_experts {
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
                fp8_moe_grouped_gate_up_addressed_f32_into_on_stream(
                    &workspace.route.indices,
                    &workspace.fp8_hidden_input,
                    &workspace.fp8_hidden_input_scale,
                    &fp8.gate.weights,
                    &fp8.gate.scales,
                    &fp8.up.weights,
                    &fp8.up.scales,
                    workspace.w4a16_gate_up_output.output(),
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
                    &workspace.w4a16_gate_up_output,
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
                    fp8_moe_grouped_down_addressed_f32_into_on_stream(
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
                fp8_moe_grouped_down_addressed_f32_into_on_stream(
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

            if !matches!(self.gate_up_storage, Qwen36GateUpStorage::CutlassW4A4) {
                return Err(Error::Format {
                    label: "Qwen3.6 resident routed gate/up",
                    detail: "device-routed NVFP4 execution requires CUTLASS W4A4 storage"
                        .to_string(),
                });
            }
            quantize_nvfp4_col_major_f32_device_into_on_stream(
                manifest.hidden,
                1,
                ffn_norm,
                &mut workspace.gate_up_input,
                1.0,
                stream,
            )?;
            let grouped_gate_up = workspace
                .grouped_gate_up
                .as_mut()
                .expect("device route requires grouped gate/up workspace");
            let run_grouped = || {
                grouped_gate_up.run_indexed_gate_up_device_route(
                    &workspace.route,
                    &self.expert_ptrs,
                    &workspace.gate_up_input,
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
            let gate_up_table = &workspace
                .grouped_gate_up
                .as_ref()
                .expect("grouped W4A4 workspace")
                .c;
            let gate_up_alpha_table = &self.gate_up_unity_alphas;
            if use_sm12x_down {
                let sm12x_down = &mut workspace.sm12x_down;
                let mut run_silu_quantize = || {
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
                };
                if let Some(profile) = profile.as_deref_mut() {
                    let (_, ms) = timed_cuda(stream, run_silu_quantize)?;
                    profile.qwen36_routed_silu_quantize_ms += ms;
                } else {
                    run_silu_quantize()?;
                }
            } else if let Some(profile) = profile.as_deref_mut() {
                let (_, ms) = timed_cuda(stream, || {
                    eider_cuda::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
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
                eider_cuda::moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
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
                            &sm12x_down.indexed_d,
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
                        &sm12x_down.indexed_d,
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
                    eider_cuda::nvfp4_w4a16_matvec_f32_into_on_stream(
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
                    eider_cuda::nvfp4_w4a16_matvec_f32_into_on_stream(
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

        if let Some(parallel) = parallel_moe {
            parallel.join.record_on_stream(parallel.shared_stream)?;
            stream.wait_event(parallel.join)?;
        } else {
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
        }

        if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                if use_device_route && use_sm12x_down && !used_pager {
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
        } else if use_device_route && use_sm12x_down && !used_pager {
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
            eider_cuda::synchronize_device()?;
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
            eider_cuda::synchronize_device()?;
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
            eider_cuda::synchronize_device()?;
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
            eider_cuda::synchronize_device()?;
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
            cublaslt_weight: ModelOptCublasLtWeight::from_modelopt(host)?,
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

    fn from_bf16_host(prefix: &str, values: &[u16], rows: usize, cols: usize) -> Result<Self> {
        let host = ModelOptNvfp4Linear::quantize_bf16(prefix, rows, cols, values)?;
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

    fn run_f32_batch_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        eider_cuda::nvfp4_w4a16_matvec_f32_batch_into_on_stream(
            input,
            &self.packed_weight,
            &self.weight_scale,
            output.output(),
            rows,
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
        let weight_addresses = (0..experts)
            .map(|expert| weights.address_at(expert * matrix_len))
            .collect::<Result<Vec<_>>>()?;
        let scale_addresses = (0..experts)
            .map(|expert| scales.address_at(expert * rows))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            _weights: weights,
            _scales: scales,
            weights: DeviceBuffer::from_host(&weight_addresses)?,
            scales: DeviceBuffer::from_host(&scale_addresses)?,
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

    pub(crate) fn device_bytes(&self) -> usize {
        let grouped_down_bytes = self.grouped_down.as_ref().map_or(0, |workspace| {
            workspace.gemv.device_bytes()
                + workspace
                    .inputs
                    .iter()
                    .map(Nvfp4Matrix::device_bytes)
                    .sum::<usize>()
                + workspace
                    .input_simple_scales
                    .iter()
                    .map(DeviceBuffer::device_bytes)
                    .sum::<usize>()
                + workspace.input_values.device_bytes()
                + workspace.input_scales.device_bytes()
                + workspace.input_values_mut.device_bytes()
                + workspace.input_scales_mut.device_bytes()
        });
        self.router_logits.device_bytes()
            + self.route.indices.device_bytes()
            + self.route.weights.device_bytes()
            + self.gate_up_input.device_bytes()
            + self.gate_up_input_simple_scales.device_bytes()
            + self
                .grouped_gate_up
                .as_ref()
                .map_or(0, GroupedGemvWorkspace::device_bytes)
            + self.w4a16_gate_up_output.device_bytes()
            + self.w4a16_gate_up_table.device_bytes()
            + self.fp8_hidden_input.device_bytes()
            + self.fp8_hidden_input_scale.device_bytes()
            + self.fp8_down_input.device_bytes()
            + self.fp8_down_input_scales.device_bytes()
            + self.fp8_shared_input.device_bytes()
            + self.fp8_shared_input_scale.device_bytes()
            + self.sm12x_down.device_bytes()
            + grouped_down_bytes
            + self.fallback_gate_up_out.device_bytes()
            + self.fallback_down_input.device_bytes()
            + self.fallback_down_out.device_bytes()
            + self.shared_gate_up_output.device_bytes()
            + self.shared_activated.device_bytes()
            + self.shared_output.device_bytes()
            + self.shared_gate_logits.device_bytes()
            + self.shared_gated.device_bytes()
            + self.moe_out.device_bytes()
            + self.ffn_out.device_bytes()
            + self.ffn_residual.device_bytes()
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
        let w4a16_gate_up_output = DeviceBuffer::zeroed(experts_per_token * gate_up_out_features)?;
        let w4a16_gate_up_offsets = (0..experts_per_token)
            .map(|slot| slot * gate_up_out_features)
            .collect::<Vec<_>>();
        let w4a16_gate_up_table =
            w4a16_gate_up_output.legacy_const_pointer_table(&w4a16_gate_up_offsets)?;
        Ok(Self {
            router_logits: DeviceBuffer::zeroed(experts)?,
            route: MoeRouteWorkspace::new(experts_per_token)?,
            gate_up_input: Nvfp4Matrix::zeroed_col_major(hidden, 1)?,
            gate_up_input_simple_scales: DeviceBuffer::zeroed(hidden.div_ceil(16))?,
            grouped_gate_up,
            w4a16_gate_up_output,
            w4a16_gate_up_table,
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

impl Qwen36ExactMoePairWorkspace {
    pub(crate) fn new(manifest: &QwenModelManifest) -> Result<Self> {
        let (experts, routes) = match manifest.ffn {
            QwenFfnConfig::Moe {
                experts,
                experts_per_token,
                ..
            } => (experts, experts_per_token),
            QwenFfnConfig::Dense => {
                return Err(Error::Format {
                    label: "Qwen exact MoE pair workspace",
                    detail: "manifest is not MoE".to_string(),
                });
            }
        };
        Ok(Self {
            inputs: (0..2)
                .map(|_| DeviceBuffer::zeroed(manifest.hidden))
                .collect::<Result<Vec<_>>>()?,
            rows: (0..2)
                .map(|_| Qwen36MoeWorkspace::new(manifest))
                .collect::<Result<Vec<_>>>()?,
            router_logits: DeviceBuffer::zeroed(2 * experts)?,
            route_indices: DeviceBuffer::zeroed(2 * routes)?,
            route_weights: DeviceBuffer::zeroed(2 * routes)?,
            zero_hidden: DeviceBuffer::zeroed(manifest.hidden)?,
            output: DeviceBuffer::zeroed(2 * manifest.hidden)?,
        })
    }

    pub(crate) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    pub(crate) fn probe_snapshots(
        &self,
        experts: usize,
        routes: usize,
        hidden: usize,
        stream: &CudaStream,
    ) -> Result<Vec<Qwen36MoeProbeSnapshot>> {
        let router_logits = self.router_logits.copy_to_host(stream)?;
        let route_indices = self.route_indices.copy_to_host(stream)?;
        let route_weights = self.route_weights.copy_to_host(stream)?;
        let output = self.output.copy_to_host(stream)?;
        (0..2)
            .map(|row| {
                let mut snapshot = self.rows[row].probe_snapshot(stream)?;
                snapshot.router_logits = router_logits[row * experts..(row + 1) * experts].to_vec();
                snapshot.route_indices = route_indices[row * routes..(row + 1) * routes].to_vec();
                snapshot.route_weights = route_weights[row * routes..(row + 1) * routes].to_vec();
                snapshot.final_output = output[row * hidden..(row + 1) * hidden].to_vec();
                Ok(snapshot)
            })
            .collect()
    }
}

impl Qwen36MoeWorkspace {
    pub(crate) fn probe_snapshot(&self, stream: &CudaStream) -> Result<Qwen36MoeProbeSnapshot> {
        Ok(Qwen36MoeProbeSnapshot {
            router_logits: self.router_logits.copy_to_host(stream)?.into_vec(),
            route_indices: self.route.indices.copy_to_host(stream)?.into_vec(),
            route_weights: self.route.weights.copy_to_host(stream)?.into_vec(),
            gate_up_input_values: self.gate_up_input.copy_values_to_host(stream)?.into_vec(),
            gate_up_input_scales: self.gate_up_input.copy_scales_to_host(stream)?.into_vec(),
            routed_output: self.moe_out.copy_to_host(stream)?.into_vec(),
            routed_gate_up: self
                .grouped_gate_up
                .as_ref()
                .map(|workspace| workspace.copy_outputs_to_host(stream))
                .transpose()?
                .unwrap_or_default(),
            repeated_routed_gate_up: None,
            oracle_routed_gate_up: None,
            routed_down_slots: self
                .grouped_down
                .as_ref()
                .map(|workspace| workspace.gemv.copy_outputs_to_host(stream))
                .transpose()?
                .unwrap_or_default(),
            shared_gate_logits: self.shared_gate_logits.copy_to_host(stream)?.into_vec(),
            shared_output: self.shared_output.copy_to_host(stream)?.into_vec(),
            final_output: self.ffn_out.copy_to_host(stream)?.into_vec(),
        })
    }
}

fn load_shared_expert(
    checkpoint: &ModelOptCheckpoint,
    mlp_prefix: &str,
    hidden: usize,
) -> Result<(Qwen36SharedExpertStorage, usize)> {
    let gate_prefix = format!("{mlp_prefix}.shared_expert.gate_proj");
    let up_prefix = format!("{mlp_prefix}.shared_expert.up_proj");
    let down_prefix = format!("{mlp_prefix}.shared_expert.down_proj");
    let uses_nvfp4 = checkpoint.contains_tensor(&format!("{gate_prefix}.weight_scale_2"))
        || checkpoint.contains_tensor(&format!("{gate_prefix}.weight_global_scale"));
    if uses_nvfp4 {
        let gate_up = load_concat_gate_up(
            checkpoint,
            &gate_prefix,
            &up_prefix,
            "Qwen shared expert gate/up",
        )?;
        let down = checkpoint.load_nvfp4_linear(&down_prefix)?;
        let intermediate = gate_up.out_features / 2;
        if gate_up.in_features != hidden
            || !gate_up.out_features.is_multiple_of(2)
            || down.in_features != intermediate
            || down.out_features != hidden
        {
            return Err(Error::Shape {
                label: "Qwen shared NVFP4 expert",
                expected: format!(
                    "gate_up in={hidden} out=2*intermediate, down in=intermediate out={hidden}"
                ),
                actual: format!(
                    "gate_up in={} out={} down in={} out={}",
                    gate_up.in_features, gate_up.out_features, down.in_features, down.out_features
                ),
            });
        }
        return Ok((
            Qwen36SharedExpertStorage::Nvfp4(Qwen36SharedExpert {
                gate_up: Nvfp4DeviceLinear::from_host(&gate_up)?,
                down: Nvfp4DeviceLinear::from_host(&down)?,
            }),
            intermediate,
        ));
    }

    let gate_name = format!("{gate_prefix}.weight");
    let up_name = format!("{up_prefix}.weight");
    let down_name = format!("{down_prefix}.weight");
    let gate_info = checkpoint.tensor_info(&gate_name)?;
    let up_info = checkpoint.tensor_info(&up_name)?;
    let down_info = checkpoint.tensor_info(&down_name)?;
    let [intermediate, gate_hidden] = gate_info.shape.as_slice() else {
        return Err(shape_error(
            "Qwen shared BF16 gate",
            &gate_info,
            "two-dimensional weight".to_string(),
        ));
    };
    if gate_info.dtype != "BF16"
        || up_info.dtype != "BF16"
        || down_info.dtype != "BF16"
        || up_info.shape != [*intermediate, *gate_hidden]
        || down_info.shape != [hidden, *intermediate]
        || *gate_hidden != hidden
    {
        return Err(Error::Shape {
            label: "Qwen shared BF16 expert",
            expected: format!(
                "gate/up dtype=BF16 shape=[intermediate,{hidden}], down dtype=BF16 shape=[{hidden},intermediate]"
            ),
            actual: format!(
                "gate dtype={} shape={:?}, up dtype={} shape={:?}, down dtype={} shape={:?}",
                gate_info.dtype,
                gate_info.shape,
                up_info.dtype,
                up_info.shape,
                down_info.dtype,
                down_info.shape
            ),
        });
    }
    let mut gate_up = read_bf16_matrix_host(checkpoint, &gate_name, *intermediate, hidden)?;
    gate_up.extend(read_bf16_matrix_host(
        checkpoint,
        &up_name,
        *intermediate,
        hidden,
    )?);
    Ok((
        Qwen36SharedExpertStorage::Bf16 {
            gate_up: Bf16Linear::from_host(&gate_up, 2 * intermediate, hidden)?,
            down: Bf16Linear::load(checkpoint, &down_name, hidden, *intermediate)?,
        },
        *intermediate,
    ))
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
    Ok(ModelOptNvfp4Linear::concat_out_features(
        format!("{gate_prefix}.gate_up_proj"),
        &gate,
        &up,
    )?)
}

fn load_dense_gate_up(
    checkpoint: &ModelOptCheckpoint,
    gate_prefix: &str,
    up_prefix: &str,
    fp8_storage: Qwen36Fp8Storage,
    fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
) -> Result<Qwen36Linear> {
    let gate_nvfp4 = checkpoint.contains_tensor(&format!("{gate_prefix}.weight_scale_2"))
        || checkpoint.contains_tensor(&format!("{gate_prefix}.weight_global_scale"));
    let up_nvfp4 = checkpoint.contains_tensor(&format!("{up_prefix}.weight_scale_2"))
        || checkpoint.contains_tensor(&format!("{up_prefix}.weight_global_scale"));
    if gate_nvfp4 || up_nvfp4 {
        if !gate_nvfp4 || !up_nvfp4 {
            return Err(Error::Format {
                label: "Qwen dense gate/up storage",
                detail: "gate and up projections use different quantization formats".to_string(),
            });
        }
        let joined = load_concat_gate_up(checkpoint, gate_prefix, up_prefix, "Qwen dense gate/up")?;
        return Nvfp4DeviceLinear::from_host(&joined).map(Qwen36Linear::Nvfp4);
    }

    let gate_name = format!("{gate_prefix}.weight");
    let up_name = format!("{up_prefix}.weight");
    let gate_info = checkpoint.tensor_info(&gate_name)?;
    let up_info = checkpoint.tensor_info(&up_name)?;
    match (gate_info.dtype.as_str(), up_info.dtype.as_str()) {
        ("BF16", "BF16") => {
            let [gate_rows, gate_cols] = gate_info.shape.as_slice() else {
                return Err(shape_error(
                    "Qwen dense BF16 gate",
                    &gate_info,
                    "two-dimensional weight".to_string(),
                ));
            };
            let [up_rows, up_cols] = up_info.shape.as_slice() else {
                return Err(shape_error(
                    "Qwen dense BF16 up",
                    &up_info,
                    "two-dimensional weight".to_string(),
                ));
            };
            if gate_cols != up_cols {
                return Err(Error::Shape {
                    label: "Qwen dense BF16 gate/up",
                    expected: format!("matching in_features={gate_cols}"),
                    actual: format!("gate={gate_cols} up={up_cols}"),
                });
            }
            let mut weight = read_bf16_matrix_host(checkpoint, &gate_name, *gate_rows, *gate_cols)?;
            weight.extend(read_bf16_matrix_host(
                checkpoint, &up_name, *up_rows, *up_cols,
            )?);
            Bf16Linear::from_host(&weight, gate_rows + up_rows, *gate_cols).map(Qwen36Linear::Bf16)
        }
        ("BF16", _) | (_, "BF16") => Err(Error::Format {
            label: "Qwen dense gate/up storage",
            detail: "gate and up projections use different quantization formats".to_string(),
        }),
        _ => match fp8_storage {
            Qwen36Fp8Storage::Fp8 => {
                let gate = checkpoint.load_fp8_linear(gate_prefix)?;
                let up = checkpoint.load_fp8_linear(up_prefix)?;
                let joined = ModelOptFp8Linear::concat_out_features(
                    format!("{gate_prefix}.gate_up_proj"),
                    &gate,
                    &up,
                )?;
                Fp8Linear::from_host(&joined).map(Qwen36Linear::Fp8)
            }
            Qwen36Fp8Storage::Nvfp4 => {
                let gate = fp8_nvfp4_cache.load_or_quantize(checkpoint, gate_prefix)?;
                let up = fp8_nvfp4_cache.load_or_quantize(checkpoint, up_prefix)?;
                let joined = ModelOptNvfp4Linear::concat_out_features(
                    format!("{gate_prefix}.gate_up_proj"),
                    &gate,
                    &up,
                )?;
                Nvfp4DeviceLinear::from_host(&joined).map(Qwen36Linear::Nvfp4)
            }
        },
    }
}

impl Qwen36DenseMlpWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        layer: usize,
        fp8_storage: Qwen36Fp8Storage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let prefix = format!("{}.layers.{layer}.mlp", manifest.tensor_prefix);
        let gate_up = load_dense_gate_up(
            checkpoint,
            &format!("{prefix}.gate_proj"),
            &format!("{prefix}.up_proj"),
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
        let down = Qwen36Linear::load(
            checkpoint,
            &format!("{prefix}.down_proj"),
            manifest.hidden,
            manifest.intermediate,
            Qwen36Bf16Storage::Bf16,
            fp8_storage,
            fp8_nvfp4_cache,
        )?;
        if gate_up.rows() != manifest.intermediate * 2
            || gate_up.cols() != manifest.hidden
            || down.rows() != manifest.hidden
            || down.cols() != manifest.intermediate
        {
            return Err(Error::Shape {
                label: "Qwen dense FFN",
                expected: format!(
                    "gate_up=[{}, {}] down=[{}, {}]",
                    manifest.intermediate * 2,
                    manifest.hidden,
                    manifest.hidden,
                    manifest.intermediate
                ),
                actual: format!(
                    "gate_up=[{}, {}] down=[{}, {}]",
                    gate_up.rows(),
                    gate_up.cols(),
                    down.rows(),
                    down.cols()
                ),
            });
        }
        Ok(Self { gate_up, down })
    }

    fn workspace(&self) -> Result<Qwen36DenseMlpWorkspace> {
        Ok(Qwen36DenseMlpWorkspace {
            gate_up: DeviceBuffer::zeroed(self.gate_up.rows())?,
            gate_up_fp8_input: DeviceBuffer::zeroed(self.gate_up.cols())?,
            gate_up_fp8_input_scale: DeviceBuffer::zeroed(1)?,
            activated: DeviceBuffer::zeroed(self.down.cols())?,
            down: DeviceBuffer::zeroed(self.down.rows())?,
            down_fp8_input: DeviceBuffer::zeroed(self.down.cols())?,
            down_fp8_input_scale: DeviceBuffer::zeroed(1)?,
            output: DeviceBuffer::zeroed(self.down.rows())?,
        })
    }

    fn run_one_token<'a>(
        &'a self,
        workspace: &'a mut Qwen36DenseMlpWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if ffn_norm.len() != self.gate_up.cols() || residual.len() != self.down.rows() {
            return Err(Error::Shape {
                label: "Qwen dense FFN inputs",
                expected: format!(
                    "ffn_norm={} residual={}",
                    self.gate_up.cols(),
                    self.down.rows()
                ),
                actual: format!("ffn_norm={} residual={}", ffn_norm.len(), residual.len()),
            });
        }
        self.gate_up.run_into(
            ffn_norm,
            &mut workspace.gate_up,
            &mut workspace.gate_up_fp8_input,
            &mut workspace.gate_up_fp8_input_scale,
            stream,
        )?;
        silu_mul_halves_f32_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            self.down.cols(),
            stream,
        )?;
        self.down.run_into(
            &workspace.activated,
            &mut workspace.down,
            &mut workspace.down_fp8_input,
            &mut workspace.down_fp8_input_scale,
            stream,
        )?;
        add_f32_into_on_stream(residual, &workspace.down, workspace.output.output(), stream)?;
        round_f32_to_bf16_in_place_on_stream(workspace.output.inout(), stream)?;
        Ok(&workspace.output)
    }
}

impl Qwen36LayerFfnWeights {
    fn workspace(&self, manifest: &QwenModelManifest) -> Result<Qwen36LayerFfnWorkspace> {
        match self {
            Self::Moe(weights) => weights
                .workspace(manifest)
                .map(Box::new)
                .map(Qwen36LayerFfnWorkspace::Moe),
            Self::Dense(weights) => weights.workspace().map(Qwen36LayerFfnWorkspace::Dense),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_one_token<'a>(
        &'a self,
        lt: &CublasLt,
        workspace: &'a mut Qwen36LayerFfnWorkspace,
        manifest: &QwenModelManifest,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
        profile: Option<&mut QwenDecodeProfile>,
        gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
    ) -> Result<&'a DeviceBuffer<f32>> {
        match (self, workspace) {
            (Self::Moe(weights), Qwen36LayerFfnWorkspace::Moe(workspace)) => weights
                .run_one_token_impl(
                    lt,
                    workspace,
                    manifest,
                    ffn_norm,
                    residual,
                    stream,
                    parallel_moe,
                    profile,
                    gpu_probe,
                )
                .map(|step| step.ffn_out),
            (Self::Dense(weights), Qwen36LayerFfnWorkspace::Dense(workspace)) => {
                weights.run_one_token(workspace, ffn_norm, residual, stream)
            }
            _ => Err(Error::Format {
                label: "Qwen feed-forward workspace",
                detail: "weights and workspace variants do not match".to_string(),
            }),
        }
    }
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
/// attention weights (linear or full), and the configured feed-forward block.
pub struct Qwen36LayerBlock {
    pub layer: usize,
    pub kind: QwenLayerKind,
    pub input_norm: DeviceBuffer<f32>,
    pub post_attn_norm: DeviceBuffer<f32>,
    pub attention: Qwen36Attention,
    pub moe: Qwen36LayerFfnWeights,
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
    pub moe: Qwen36LayerFfnWorkspace,
}

/// Attention workspace variant held by a layer block.
pub enum Qwen36AttentionWorkspace {
    LinearAttention(Qwen36LinearAttentionWorkspace),
    FullAttention(Qwen36FullAttentionWorkspace),
}

/// Persistent attention state for one layer of one generated sequence.
pub enum Qwen36AttentionState {
    LinearAttention(Qwen36LinearAttentionState),
    FullAttention(Qwen36FullAttentionState),
}

/// Persistent state for one model layer of one generated sequence.
pub struct Qwen36LayerSequenceState {
    pub kind: QwenLayerKind,
    pub attention: Qwen36AttentionState,
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
            model.bf16_storage,
            model.fp8_attention_storage,
        )?);
        Self::load_inner(model, layer, false, None, fp8)
    }

    fn load_from_prepared_cache(
        model: &Qwen36Model,
        layer: usize,
        fp8: Rc<Qwen36LinearFp8Execution>,
    ) -> Result<Self> {
        Self::load_inner(model, layer, true, None, fp8)
    }

    fn load_from_prepared_cache_paged(
        model: &Qwen36Model,
        layer: usize,
        capacity: usize,
        fp8: Rc<Qwen36LinearFp8Execution>,
    ) -> Result<Self> {
        Self::load_inner(model, layer, true, Some(capacity), fp8)
    }

    fn load_inner(
        model: &Qwen36Model,
        layer: usize,
        cache_prepared: bool,
        expert_cache_capacity: Option<usize>,
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
                    model.bf16_storage.attention,
                    model.fp8_attention_storage,
                    &model.fp8_nvfp4_cache,
                )?)
            }
            QwenLayerKind::FullAttention => {
                Qwen36Attention::FullAttention(Qwen36FullAttentionWeights::load(
                    &model.checkpoint,
                    &model.manifest,
                    layer,
                    model.bf16_storage.attention,
                    model.fp8_attention_storage,
                    &model.fp8_nvfp4_cache,
                )?)
            }
        };
        let moe = match model.manifest.ffn {
            QwenFfnConfig::Moe { .. } => {
                let weights = if let Some(capacity) = expert_cache_capacity {
                    model.load_moe_from_prepared_cache_paged(layer, capacity)?
                } else if cache_prepared {
                    model.load_moe_from_prepared_cache(layer)?
                } else {
                    model.load_moe(layer)?
                };
                Qwen36LayerFfnWeights::Moe(Box::new(weights))
            }
            QwenFfnConfig::Dense => {
                Qwen36LayerFfnWeights::Dense(Box::new(Qwen36DenseMlpWeights::load(
                    &model.checkpoint,
                    &model.manifest,
                    layer,
                    model.fp8_dense_mlp_storage,
                    &model.fp8_nvfp4_cache,
                )?))
            }
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

    /// Allocates persistent state for one layer of one generated sequence.
    pub fn sequence_state(
        &self,
        model: &Qwen36Model,
        cache_capacity: usize,
    ) -> Result<Qwen36LayerSequenceState> {
        let manifest = model.manifest();
        let attention = match &self.attention {
            Qwen36Attention::LinearAttention(weights) => {
                let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
                    label: "Qwen3.6 linear-attention state",
                    detail: "manifest has no linear-attention configuration".to_string(),
                })?;
                Qwen36AttentionState::LinearAttention(Qwen36LinearAttentionState::new(
                    linear, weights,
                )?)
            }
            Qwen36Attention::FullAttention(_) => Qwen36AttentionState::FullAttention(
                Qwen36FullAttentionState::new(manifest, cache_capacity)?,
            ),
        };
        Ok(Qwen36LayerSequenceState {
            kind: self.kind,
            attention,
        })
    }

    fn enqueue_linear_pre_gdn(
        &self,
        workspace: &mut Qwen36LayerBlockWorkspace,
        state: &mut Qwen36LayerSequenceState,
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
        match (
            &self.attention,
            &mut workspace.attention,
            &mut state.attention,
        ) {
            (
                Qwen36Attention::LinearAttention(weights),
                Qwen36AttentionWorkspace::LinearAttention(attention_workspace),
                Qwen36AttentionState::LinearAttention(attention_state),
            ) => weights.enqueue_pre_gdn(
                attention_workspace,
                attention_state,
                &workspace.normed_hidden,
                stream,
            ),
            _ => Err(Error::Format {
                label: "Qwen3.6 segmented graph",
                detail: "pre-GDN segment requires a linear-attention layer".to_string(),
            }),
        }
    }

    fn enqueue_linear_gdn(
        &self,
        workspace: &mut Qwen36LayerBlockWorkspace,
        state: &mut Qwen36LayerSequenceState,
        stream: &CudaStream,
    ) -> Result<()> {
        match (
            &self.attention,
            &mut workspace.attention,
            &mut state.attention,
        ) {
            (
                Qwen36Attention::LinearAttention(weights),
                Qwen36AttentionWorkspace::LinearAttention(attention_workspace),
                Qwen36AttentionState::LinearAttention(attention_state),
            ) => weights.enqueue_gdn(attention_workspace, attention_state, stream),
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
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
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
            parallel_moe,
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
        state: &mut Qwen36LayerSequenceState,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        position: &DeviceBuffer<u32>,
        cache_len: &DeviceBuffer<u32>,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
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
        let attn_output = match (
            &self.attention,
            &mut workspace.attention,
            &mut state.attention,
        ) {
            (
                Qwen36Attention::FullAttention(weights),
                Qwen36AttentionWorkspace::FullAttention(attention_workspace),
                Qwen36AttentionState::FullAttention(attention_state),
            ) => {
                weights.run_one_token_indexed(
                    attention_workspace,
                    attention_state,
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
            parallel_moe,
            None,
            None,
        )?;
        Ok(())
    }

    /// Runs one token through the full layer block.
    ///
    /// `hidden` is the input hidden vector; the block writes its output into
    /// `workspace.ffn_norm`-adjacent storage and returns a borrow of the
    /// final feed-forward buffer, which already includes the residual.
    #[allow(clippy::needless_option_as_deref, clippy::too_many_arguments)]
    pub fn run_one_token<'a>(
        &'a self,
        lt: &CublasLt,
        workspace: &'a mut Qwen36LayerBlockWorkspace,
        state: &mut Qwen36LayerSequenceState,
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
                    &mut state.attention,
                    manifest,
                    &workspace.normed_hidden,
                    position,
                    stream,
                    Some(&mut *profile),
                    gpu_probe.as_deref_mut(),
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
                &mut state.attention,
                manifest,
                &workspace.normed_hidden,
                position,
                stream,
                None,
                gpu_probe.as_deref_mut(),
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

        let ffn_output = if let Some(profile) = profile.as_deref_mut() {
            let wall_start = Instant::now();
            let (step, ms) = timed_cuda(stream, || {
                self.moe.run_one_token(
                    lt,
                    &mut workspace.moe,
                    manifest,
                    &workspace.ffn_norm,
                    &workspace.attn_residual,
                    stream,
                    None,
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
                None,
                gpu_probe.as_deref_mut(),
            )?
        };
        Ok(Qwen36LayerBlockStep { output: ffn_output })
    }
}

#[allow(clippy::needless_option_as_deref, clippy::too_many_arguments)]
fn run_qwen36_attention<'a>(
    attention: &'a Qwen36Attention,
    workspace: &'a mut Qwen36AttentionWorkspace,
    state: &mut Qwen36AttentionState,
    manifest: &QwenModelManifest,
    normed_hidden: &DeviceBuffer<f32>,
    position: usize,
    stream: &CudaStream,
    profile: Option<&mut QwenDecodeProfile>,
    mut gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
) -> Result<&'a DeviceBuffer<f32>> {
    match (attention, workspace, state) {
        (
            Qwen36Attention::LinearAttention(w),
            Qwen36AttentionWorkspace::LinearAttention(ws),
            Qwen36AttentionState::LinearAttention(sequence),
        ) => {
            let step = if gpu_probe
                .as_ref()
                .is_some_and(|probe| probe.should_capture(Qwen36GpuCounterStage::LinearAttention))
            {
                gpu_probe
                    .as_deref_mut()
                    .expect("probe present")
                    .capture(|| {
                        w.run_one_token(ws, sequence, normed_hidden, manifest.rms_eps, stream, None)
                    })?
            } else {
                w.run_one_token(
                    ws,
                    sequence,
                    normed_hidden,
                    manifest.rms_eps,
                    stream,
                    profile,
                )?
            };
            Ok(step.output)
        }
        (
            Qwen36Attention::FullAttention(w),
            Qwen36AttentionWorkspace::FullAttention(ws),
            Qwen36AttentionState::FullAttention(sequence),
        ) => {
            let step = if gpu_probe
                .as_ref()
                .is_some_and(|probe| probe.should_capture(Qwen36GpuCounterStage::FullAttention))
            {
                gpu_probe
                    .as_deref_mut()
                    .expect("probe present")
                    .capture(|| {
                        w.run_one_token(ws, sequence, manifest, normed_hidden, position, stream)
                    })?
            } else {
                w.run_one_token(ws, sequence, manifest, normed_hidden, position, stream)?
            };
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

/// Fully loaded Qwen3.6 text model ready for stateful batched decode.
///
/// Holds all layer block weights, the embedding table, the final RMSNorm
/// weight, and the quantized lm_head. Routed-expert NVFP4 weights are loaded
/// lazily on first use.
pub struct Qwen36TextModel {
    model_id: u64,
    manifest: QwenModelManifest,
    checkpoint: ModelOptCheckpoint,
    artifact_dir: PathBuf,
    lt: CublasLt,
    layers: Vec<Qwen36LayerBlock>,
    embedding: Qwen36Embedding,
    final_norm: DeviceBuffer<f32>,
    lm_head: Qwen36LmHead,
    mtp: Option<Qwen36MtpWeights>,
    dflash2: Option<dflash2::Qwen38DFlash2>,
    expert_paging: bool,
    bf16_storage: Qwen36Bf16StorageConfig,
    fp8_attention_storage: Qwen36Fp8Storage,
    fp8_dense_mlp_storage: Qwen36Fp8Storage,
    fp8_lm_head_storage: Qwen36Fp8Storage,
    fp8_nvfp4_cache: Qwen36Fp8Nvfp4Cache,
}

pub(crate) enum Qwen36Embedding {
    Bf16(DeviceBuffer<u16>),
    Fp8 {
        weight: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
    },
}

impl Qwen36Embedding {
    pub(crate) fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        if checkpoint.tensor_info(&weight_name)?.dtype == "BF16" {
            return read_bf16_matrix_device(checkpoint, &weight_name, rows, cols).map(Self::Bf16);
        }

        let host = checkpoint.load_fp8_linear(prefix)?;
        if host.out_features != rows || host.in_features != cols {
            return Err(Error::Shape {
                label: "Qwen embedding",
                expected: format!("[{rows}, {cols}]"),
                actual: format!("[{}, {}]", host.out_features, host.in_features),
            });
        }
        let row_scales = host
            .channel_weight_scale
            .unwrap_or_else(|| vec![host.weight_scale; rows]);
        Ok(Self::Fp8 {
            weight: DeviceBuffer::from_host(&host.weight)?,
            row_scales: DeviceBuffer::from_host(&row_scales)?,
        })
    }

    pub(crate) fn gather_prefix(
        &self,
        vocab: usize,
        hidden: usize,
        token_ids: &DeviceBuffer<u32>,
        output: eider_cuda::DeviceOutput<'_, f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Bf16(weight) => copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
                vocab, hidden, weight, token_ids, output, rows, stream,
            ),
            Self::Fp8 { weight, row_scales } => copy_fp8_rows_to_f32_indexed_prefix_into_on_stream(
                vocab, hidden, weight, row_scales, token_ids, output, rows, stream,
            ),
        }
    }
}

#[allow(private_interfaces)]
pub(crate) enum Qwen36LmHead {
    Nvfp4(Nvfp4DeviceLinear),
    Bf16(Bf16Linear),
    Fp8 {
        linear: Fp8Linear,
        plan: Option<Box<Fp8TnMatmulPlan>>,
    },
}

impl Qwen36LmHead {
    pub(crate) fn load_bf16(
        checkpoint: &ModelOptCheckpoint,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Bf16Linear::load(checkpoint, "lm_head.weight", rows, cols).map(Self::Bf16)
    }

    fn load(
        checkpoint: &ModelOptCheckpoint,
        lt: &CublasLt,
        bf16_storage: Qwen36Bf16Storage,
        fp8_storage: Qwen36Fp8Storage,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        if checkpoint.contains_tensor("lm_head.weight_scale_2")
            || checkpoint.contains_tensor("lm_head.weight_global_scale")
        {
            Ok(Self::Nvfp4(Nvfp4DeviceLinear::load(checkpoint, "lm_head")?))
        } else if checkpoint.tensor_info("lm_head.weight")?.dtype == "BF16" {
            let info = checkpoint.tensor_info("lm_head.weight")?;
            let [rows, cols] = info.shape.as_slice() else {
                return Err(Error::Shape {
                    label: "Qwen3.5 BF16 lm_head",
                    expected: "two-dimensional weight".to_string(),
                    actual: format!("shape={:?}", info.shape),
                });
            };
            match bf16_storage {
                Qwen36Bf16Storage::Bf16 => Ok(Self::Bf16(Bf16Linear::load(
                    checkpoint,
                    "lm_head.weight",
                    *rows,
                    *cols,
                )?)),
                Qwen36Bf16Storage::Fp8 => {
                    let host = read_bf16_matrix_host(checkpoint, "lm_head.weight", *rows, *cols)?;
                    let linear = Fp8Linear::from_bf16_host(&host, *rows, *cols)?;
                    Ok(Self::Fp8 { linear, plan: None })
                }
                Qwen36Bf16Storage::Nvfp4 => {
                    let host = read_bf16_matrix_host(checkpoint, "lm_head.weight", *rows, *cols)?;
                    Ok(Self::Nvfp4(Nvfp4DeviceLinear::from_bf16_host(
                        "lm_head", &host, *rows, *cols,
                    )?))
                }
            }
        } else {
            match fp8_storage {
                Qwen36Fp8Storage::Fp8 => {
                    let host = checkpoint.load_fp8_linear("lm_head")?;
                    let linear = Fp8Linear::from_host(&host)?;
                    let plan = Fp8TnMatmulPlan::new(
                        lt,
                        GemmShape::new(linear.rows, 1, linear.cols),
                        8 << 20,
                    )?;
                    Ok(Self::Fp8 {
                        linear,
                        plan: Some(Box::new(plan)),
                    })
                }
                Qwen36Fp8Storage::Nvfp4 => Nvfp4DeviceLinear::from_host(
                    &fp8_nvfp4_cache.load_or_quantize(checkpoint, "lm_head")?,
                )
                .map(Self::Nvfp4),
            }
        }
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        match self {
            Self::Nvfp4(linear) => (linear.out_features, linear.in_features),
            Self::Bf16(linear) => (linear.rows, linear.cols),
            Self::Fp8 { linear, .. } => (linear.rows, linear.cols),
        }
    }

    pub(crate) fn run_top1(
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
            Self::Bf16(linear) => {
                linear.run_into(input, &mut workspace.logits, stream)?;
                argmax_f32_into_on_stream(
                    &workspace.logits,
                    workspace.next_index.output(),
                    workspace.next_value.output(),
                    stream,
                )
            }
            Self::Fp8 { linear, plan } => {
                Self::run_fp8_logits(lt, linear, plan.as_deref(), input, workspace, stream)?;
                argmax_f32_into_on_stream(
                    &workspace.logits,
                    workspace.next_index.output(),
                    workspace.next_value.output(),
                    stream,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_bf16_top1_batch(
        &self,
        input: &DeviceBuffer<f32>,
        logits: &mut DeviceBuffer<f32>,
        scratch_indices: &DeviceBuffer<u32>,
        output_indices: &mut DeviceBuffer<u32>,
        output_values: &mut DeviceBuffer<f32>,
        rows: usize,
        row_capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let Self::Bf16(linear) = self else {
            return Err(Error::Format {
                label: "Qwen BF16 batched lm_head",
                detail: "the loaded vocabulary head is not BF16".to_string(),
            });
        };
        lm_head_top1_f32_batch_into_on_stream(
            input,
            &linear.weight,
            logits,
            scratch_indices,
            output_indices,
            output_values,
            rows,
            linear.rows,
            linear.cols,
            stream,
        )?;
        if output_indices.len() < row_capacity || output_values.len() < row_capacity {
            return Err(Error::Shape {
                label: "Qwen BF16 batched lm_head outputs",
                expected: format!("at least {row_capacity} rows"),
                actual: format!(
                    "indices={} values={}",
                    output_indices.len(),
                    output_values.len()
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn run_bf16_exact_two_rows_top1(
        &self,
        input: &DeviceBuffer<f32>,
        logits: &mut DeviceBuffer<f32>,
        output_indices: &mut DeviceBuffer<u32>,
        output_values: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let Self::Bf16(linear) = self else {
            return Err(Error::Format {
                label: "Qwen exact two-row lm_head",
                detail: "the loaded vocabulary head is not BF16".to_string(),
            });
        };
        bf16_linear_two_rows_f32_into_on_stream(
            input,
            &linear.weight,
            logits.output(),
            linear.rows,
            linear.cols,
            stream,
        )?;
        argmax_f32_batch_into_on_stream(
            logits,
            output_indices.output(),
            output_values.output(),
            2,
            linear.rows,
            stream,
        )
    }

    pub(crate) fn run_logits(
        &self,
        lt: &CublasLt,
        input: &DeviceBuffer<f32>,
        workspace: &mut Qwen36LmHeadWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Nvfp4(linear) => linear.run_f32_into(input, &mut workspace.logits, stream),
            Self::Bf16(linear) => linear.run_into(input, &mut workspace.logits, stream),
            Self::Fp8 { linear, plan } => {
                Self::run_fp8_logits(lt, linear, plan.as_deref(), input, workspace, stream)
            }
        }
    }

    fn run_fp8_logits(
        lt: &CublasLt,
        linear: &Fp8Linear,
        plan: Option<&Fp8TnMatmulPlan>,
        input: &DeviceBuffer<f32>,
        workspace: &mut Qwen36LmHeadWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if linear.weight_only {
            return linear.run_into(
                input,
                &mut workspace.logits,
                &mut workspace.dynamic_input,
                &mut workspace.dynamic_input_scale,
                stream,
            );
        }
        let Some(channel_scale) = linear.channel_weight_scale.as_ref() else {
            return linear.run_into(
                input,
                &mut workspace.logits,
                &mut workspace.dynamic_input,
                &mut workspace.dynamic_input_scale,
                stream,
            );
        };
        let plan = plan.ok_or_else(|| Error::Format {
            label: "Qwen3.6 FP8 lm_head",
            detail: "channel-scaled checkpoint weight has no cuBLASLt plan".to_string(),
        })?;
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

pub(crate) struct Qwen36LmHeadWorkspace {
    logits: DeviceBuffer<f32>,
    dynamic_input: DeviceBuffer<u8>,
    dynamic_input_scale: DeviceBuffer<f32>,
    scratch_value: DeviceBuffer<f32>,
    scratch_index: DeviceBuffer<u32>,
    next_index: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
}

impl Qwen36LmHeadWorkspace {
    pub(crate) fn new(vocab: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            logits: DeviceBuffer::zeroed(vocab)?,
            dynamic_input: DeviceBuffer::zeroed(hidden)?,
            dynamic_input_scale: DeviceBuffer::zeroed(1)?,
            scratch_value: DeviceBuffer::zeroed(vocab.div_ceil(8))?,
            scratch_index: DeviceBuffer::zeroed(vocab.div_ceil(8))?,
            next_index: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
        })
    }

    pub(crate) fn read_top1(&self, stream: &CudaStream) -> Result<(u32, f32)> {
        let index = self.next_index.copy_to_host(stream)?;
        let value = self.next_value.copy_to_host(stream)?;
        Ok((index[0], value[0]))
    }

    pub(crate) fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    pub(crate) fn read_logits(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        Ok(self.logits.copy_to_host(stream)?.into_vec())
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.logits.device_bytes()
            + self.dynamic_input.device_bytes()
            + self.dynamic_input_scale.device_bytes()
            + self.scratch_value.device_bytes()
            + self.scratch_index.device_bytes()
            + self.next_index.device_bytes()
            + self.next_value.device_bytes()
    }
}

struct Qwen36LinearLayerGraphs {
    layer: CudaGraphExec,
}

enum Qwen36LayerGraphs {
    Linear(Qwen36LinearLayerGraphs),
    Full(CudaGraphExec),
}

struct Qwen36MoeGraphSync {
    fork: CudaEvent,
    join: CudaEvent,
}

/// Mutable decode state for [`Qwen36TextModel`].
pub(crate) struct Qwen36SequenceState {
    model_id: u64,
    linear_states: Vec<Option<Qwen36LinearAttentionState>>,
    rollback_linear_states: Vec<Option<Qwen36LinearAttentionState>>,
    rollback_position: usize,
    append_pending: bool,
    position: usize,
    max_tokens: usize,
}

/// Immutable page-aligned snapshot of Qwen3.6's non-pageable recurrent state.
///
/// Full-attention KV pages are intentionally absent and remain owned by the
/// shared sequence cache.
pub struct Qwen36SequenceSnapshot {
    model_id: u64,
    position: usize,
    linear_states: Vec<Option<Qwen36LinearAttentionState>>,
    device_bytes: usize,
}

impl seqcache::RetainedSnapshot for Qwen36SequenceSnapshot {
    fn retained_bytes(&self) -> usize {
        self.device_bytes
    }
}

impl Qwen36SequenceState {
    /// Returns the next position that will be written by decode.
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Returns the allocated context capacity.
    pub(crate) fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns the number of device bytes owned by this sequence state.
    pub(crate) fn device_bytes(&self) -> usize {
        self.linear_states
            .iter()
            .chain(&self.rollback_linear_states)
            .flatten()
            .map(Qwen36LinearAttentionState::device_bytes)
            .sum()
    }

    pub(crate) fn begin_append(&mut self, stream: &CudaStream) -> Result<()> {
        if self.append_pending {
            return Err(Error::Format {
                label: "Qwen3.6 recurrent transaction",
                detail: "an append transaction is already pending".to_string(),
            });
        }
        for (source, destination) in self
            .linear_states
            .iter()
            .zip(&mut self.rollback_linear_states)
        {
            match (source, destination) {
                (Some(source), Some(destination)) => {
                    destination.conv_state.copy_prefix_from_device_on_stream(
                        &source.conv_state,
                        source.conv_state.len(),
                        stream,
                    )?;
                    destination
                        .recurrent_state
                        .copy_prefix_from_device_on_stream(
                            &source.recurrent_state,
                            source.recurrent_state.len(),
                            stream,
                        )?;
                }
                (None, None) => {}
                _ => unreachable!("Qwen recurrent rollback topology matches active state"),
            }
        }
        self.rollback_position = self.position;
        self.append_pending = true;
        Ok(())
    }

    pub(crate) fn commit_append(&mut self, rows: usize) {
        assert!(self.append_pending, "Qwen recurrent append is pending");
        self.position = self.rollback_position + rows;
        self.append_pending = false;
    }

    pub(crate) fn abort_append(&mut self, stream: &CudaStream) -> Result<()> {
        if !self.append_pending {
            return Err(Error::Format {
                label: "Qwen3.6 recurrent transaction",
                detail: "no append transaction is pending".to_string(),
            });
        }
        for (source, destination) in self
            .rollback_linear_states
            .iter()
            .zip(&mut self.linear_states)
        {
            match (source, destination) {
                (Some(source), Some(destination)) => {
                    destination.conv_state.copy_prefix_from_device_on_stream(
                        &source.conv_state,
                        source.conv_state.len(),
                        stream,
                    )?;
                    destination
                        .recurrent_state
                        .copy_prefix_from_device_on_stream(
                            &source.recurrent_state,
                            source.recurrent_state.len(),
                            stream,
                        )?;
                }
                (None, None) => {}
                _ => unreachable!("Qwen recurrent rollback topology matches active state"),
            }
        }
        self.position = self.rollback_position;
        self.append_pending = false;
        Ok(())
    }
}

impl Qwen36SequenceSnapshot {
    /// Returns the page-aligned position represented by this snapshot.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns exact device bytes retained outside shared KV pages.
    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }
}

struct Qwen36ReferenceSequenceState {
    layer_states: Vec<Qwen36LayerSequenceState>,
    position: usize,
    max_tokens: usize,
}

/// Single-row reference execution state for profiling and GPU diagnostics.
///
/// Serving and ordinary generation use the shared paged batch API instead.
pub struct Qwen36ReferenceDecodeState {
    stream: CudaStream,
    parallel_moe_stream: Option<CudaStream>,
    parallel_moe_sync: Vec<Qwen36MoeGraphSync>,
    token_id_device: DeviceBuffer<u32>,
    position_device: DeviceBuffer<u32>,
    cache_len_device: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    sequence: Qwen36ReferenceSequenceState,
    layer_workspaces: Vec<Qwen36LayerBlockWorkspace>,
    final_hidden: DeviceBuffer<f32>,
    lm_head: Qwen36LmHeadWorkspace,
    segmented_graphs: Option<Vec<Qwen36LayerGraphs>>,
}

/// One decoded next-token result.
pub struct Qwen36NextToken {
    /// Argmax token id.
    pub id: u32,
    /// Winning logit value.
    pub value: f32,
}

/// Qwen3.6 decode stage that can be wrapped by GPU counter collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36GpuCounterStage {
    /// Routed expert grouped gate/up stage.
    RoutedGateUp,
    /// One complete full-attention layer, including projections and KV attention.
    FullAttention,
    /// One complete linear-attention layer.
    LinearAttention,
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

    /// Loads the model with an explicit writable root for reconstructed experts.
    pub fn open_with_storage_and_artifact_dir(
        model_dir: impl AsRef<std::path::Path>,
        artifact_dir: impl Into<PathBuf>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        let model = Qwen36Model::open_with_storage_and_artifact_dir(
            model_dir,
            artifact_dir,
            bf16_storage,
            fp8_attention_storage,
        )?;
        Self::from_qwen36_model(model)
    }

    /// Loads the model with independent runtime storage for native FP8 weights.
    pub fn open_with_fp8_storage_and_artifact_dir(
        model_dir: impl AsRef<std::path::Path>,
        artifact_dir: impl Into<PathBuf>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
        fp8_dense_mlp_storage: Qwen36Fp8Storage,
        fp8_lm_head_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        let model = Qwen36Model::open_with_fp8_storage_and_artifact_dir(
            model_dir,
            artifact_dir,
            bf16_storage,
            fp8_attention_storage,
            fp8_dense_mlp_storage,
            fp8_lm_head_storage,
        )?;
        Self::from_qwen36_model(model)
    }

    /// Loads the model with explicit runtime storage for BF16 weights.
    pub fn open_with_bf16_storage(
        model_dir: impl AsRef<std::path::Path>,
        bf16_storage: Qwen36Bf16StorageConfig,
    ) -> Result<Self> {
        let model = Qwen36Model::open_with_bf16_storage(model_dir, bf16_storage)?;
        Self::from_qwen36_model(model)
    }

    /// Loads the model with explicit runtime storage for BF16 and FP8 weights.
    pub fn open_with_storage(
        model_dir: impl AsRef<std::path::Path>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        let model = Qwen36Model::open_with_storage(model_dir, bf16_storage, fp8_attention_storage)?;
        Self::from_qwen36_model(model)
    }

    /// Loads the model with only `capacity_per_layer` routed experts resident.
    pub fn open_with_expert_cache_capacity(
        model_dir: impl AsRef<std::path::Path>,
        capacity_per_layer: usize,
    ) -> Result<Self> {
        let model = Qwen36Model::open(model_dir)?;
        Self::from_qwen36_model_with_expert_cache_capacity(model, capacity_per_layer)
    }

    /// Builds the full text model from an already-opened [`Qwen36Model`].
    pub fn from_qwen36_model(model: Qwen36Model) -> Result<Self> {
        Self::from_qwen36_model_inner(model, None)
    }

    /// Builds a text model whose routed experts are backed only by bounded slots.
    pub fn from_qwen36_model_with_expert_cache_capacity(
        model: Qwen36Model,
        capacity_per_layer: usize,
    ) -> Result<Self> {
        Self::from_qwen36_model_inner(model, Some(capacity_per_layer))
    }

    fn from_qwen36_model_inner(
        model: Qwen36Model,
        capacity_per_layer: Option<usize>,
    ) -> Result<Self> {
        let manifest = model.manifest().clone();
        let checkpoint = model.checkpoint().clone();
        let artifact_dir = model.artifact_dir.clone();
        let bf16_storage = model.bf16_storage;
        let fp8_attention_storage = model.fp8_attention_storage;
        let fp8_dense_mlp_storage = model.fp8_dense_mlp_storage;
        let fp8_lm_head_storage = model.fp8_lm_head_storage;
        let fp8_nvfp4_cache = model.fp8_nvfp4_cache.clone();
        let is_moe = matches!(manifest.ffn, QwenFfnConfig::Moe { .. });
        if is_moe {
            ensure_model_cache(&checkpoint, &manifest, &artifact_dir)?;
        }
        let lt = CublasLt::new()?;
        let linear_fp8 = Rc::new(Qwen36LinearFp8Execution::new(
            &checkpoint,
            &manifest,
            bf16_storage,
            fp8_attention_storage,
        )?);
        let mut layers = Vec::with_capacity(manifest.layers);
        for layer in 0..manifest.layers {
            let block = if let Some(capacity) = capacity_per_layer {
                Qwen36LayerBlock::load_from_prepared_cache_paged(
                    &model,
                    layer,
                    capacity,
                    Rc::clone(&linear_fp8),
                )?
            } else {
                Qwen36LayerBlock::load_from_prepared_cache(&model, layer, Rc::clone(&linear_fp8))?
            };
            layers.push(block);
        }
        let embedding = Qwen36Embedding::load(
            &checkpoint,
            &format!("{}.embed_tokens", manifest.tensor_prefix),
            manifest.vocab,
            manifest.hidden,
        )?;
        let final_norm = read_bf16_vector_delta_as_f32_device(
            &checkpoint,
            &format!("{}.norm.weight", manifest.tensor_prefix),
            manifest.hidden,
        )?;
        let lm_head = Qwen36LmHead::load(
            &checkpoint,
            &lt,
            bf16_storage.lm_head,
            fp8_lm_head_storage,
            &fp8_nvfp4_cache,
        )?;
        let lm_head_shape = lm_head.shape();
        if lm_head_shape != (manifest.vocab, manifest.hidden) {
            return Err(Error::Shape {
                label: "Qwen3.6 lm_head",
                expected: format!("[{}, {}]", manifest.vocab, manifest.hidden),
                actual: format!("[{}, {}]", lm_head_shape.0, lm_head_shape.1),
            });
        }
        let has_mtp = manifest.mtp_layers > 0 && checkpoint.contains_tensor("mtp.fc.weight");
        let has_dense_mtp = checkpoint.contains_tensor("mtp.layers.0.mlp.gate_proj.weight");
        let mtp = if has_mtp && has_dense_mtp {
            Some(Qwen36MtpWeights::load(
                &checkpoint,
                &manifest,
                &fp8_nvfp4_cache,
            )?)
        } else {
            if has_mtp {
                tracing::info!("checkpoint MTP block is not a supported dense draft; MTP disabled");
            }
            None
        };
        let (cache_hits, cache_prepared) = fp8_nvfp4_cache.stats();
        if cache_hits + cache_prepared > 0 {
            tracing::info!(
                cache_hits,
                cache_prepared,
                "loaded Qwen FP8-to-NVFP4 weight cache"
            );
        }
        Ok(Self {
            model_id: NEXT_QWEN36_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            manifest,
            checkpoint,
            artifact_dir,
            lt,
            layers,
            embedding,
            final_norm,
            lm_head,
            mtp,
            dflash2: None,
            expert_paging: is_moe && capacity_per_layer.is_some(),
            bf16_storage,
            fp8_attention_storage,
            fp8_dense_mlp_storage,
            fp8_lm_head_storage,
            fp8_nvfp4_cache,
        })
    }

    /// Enables the synchronous bounded expert-cache experiment for every NVFP4 layer.
    ///
    /// The resident weights remain allocated so this mode can be compared
    /// directly with the established path. Decode graphs are disabled because
    /// route readback and slot replacement are host-controlled.
    pub fn enable_expert_paging(&mut self, capacity_per_layer: usize) -> Result<()> {
        if !matches!(self.manifest.ffn, QwenFfnConfig::Moe { .. }) {
            return Err(Error::Format {
                label: "Qwen expert paging",
                detail: "expert paging requires a MoE checkpoint".to_string(),
            });
        }
        let model = Qwen36Model {
            manifest: self.manifest.clone(),
            checkpoint: self.checkpoint.clone(),
            artifact_dir: self.artifact_dir.clone(),
            bf16_storage: self.bf16_storage,
            fp8_attention_storage: self.fp8_attention_storage,
            fp8_dense_mlp_storage: self.fp8_dense_mlp_storage,
            fp8_lm_head_storage: self.fp8_lm_head_storage,
            fp8_nvfp4_cache: self.fp8_nvfp4_cache.clone(),
        };
        for block in &mut self.layers {
            block
                .moe
                .enable_expert_paging(&model, block.layer, capacity_per_layer)?;
        }
        self.expert_paging = true;
        Ok(())
    }

    /// Returns cumulative paging activity across all layers, when enabled.
    pub fn expert_paging_stats(&self) -> Option<Qwen36PagingStats> {
        self.expert_paging.then(|| {
            self.layers
                .iter()
                .filter_map(|block| {
                    block
                        .moe
                        .expert_pager
                        .borrow()
                        .as_ref()
                        .map(Qwen36ExpertPager::stats)
                })
                .fold(Qwen36PagingStats::default(), |mut total, layer| {
                    total.hits += layer.hits;
                    total.misses += layer.misses;
                    total.bytes_read += layer.bytes_read;
                    total
                })
        })
    }

    /// Returns the parsed manifest.
    pub fn manifest(&self) -> &QwenModelManifest {
        &self.manifest
    }

    /// Gathers one embedding row selected by a device-resident token ID.
    pub fn gather_embedding(
        &self,
        token_id: &DeviceBuffer<u32>,
        output: eider_cuda::DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.embedding.gather_prefix(
            self.manifest.vocab,
            self.manifest.hidden,
            token_id,
            output,
            1,
            stream,
        )
    }

    /// Allocates request-private recurrent state for `max_tokens` positions.
    ///
    /// Full-attention K/V is owned by the shared sequence cache.
    pub(crate) fn new_sequence_state(&self, max_tokens: usize) -> Result<Qwen36SequenceState> {
        if max_tokens == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 sequence state",
                expected: "max_tokens > 0".to_string(),
                actual: "0".to_string(),
            });
        }
        let mut linear_states = Vec::with_capacity(self.layers.len());
        let mut rollback_linear_states = Vec::with_capacity(self.layers.len());
        for block in &self.layers {
            let (state, rollback) = match &block.attention {
                Qwen36Attention::LinearAttention(weights) => {
                    let linear = self
                        .manifest
                        .linear_attention
                        .ok_or_else(|| Error::Format {
                            label: "Qwen3.6 linear-attention state",
                            detail: "manifest has no linear-attention configuration".to_string(),
                        })?;
                    (
                        Some(Qwen36LinearAttentionState::new(linear, weights)?),
                        Some(Qwen36LinearAttentionState::new(linear, weights)?),
                    )
                }
                Qwen36Attention::FullAttention(_) => (None, None),
            };
            linear_states.push(state);
            rollback_linear_states.push(rollback);
        }
        Ok(Qwen36SequenceState {
            model_id: self.model_id,
            linear_states,
            rollback_linear_states,
            rollback_position: 0,
            append_pending: false,
            position: 0,
            max_tokens,
        })
    }

    /// Copies only Qwen3.6's fixed-size recurrent state for prefix retention.
    pub(crate) fn snapshot_sequence(
        &self,
        source: &Qwen36SequenceState,
    ) -> Result<Qwen36SequenceSnapshot> {
        if source.model_id != self.model_id
            || source.position == 0
            || !source.position.is_multiple_of(128)
        {
            return Err(Error::Shape {
                label: "Qwen3.6 sequence snapshot",
                expected: "matching model and nonzero 128-token-aligned position".to_string(),
                actual: format!(
                    "model={} expected_model={} position={}",
                    source.model_id, self.model_id, source.position
                ),
            });
        }
        let stream = CudaStream::new_non_blocking()?;
        let mut linear_states = Vec::with_capacity(source.linear_states.len());
        let mut device_bytes = 0usize;
        for state in &source.linear_states {
            match state {
                Some(source) => {
                    let mut destination = Qwen36LinearAttentionState {
                        conv_state: DeviceBuffer::zeroed(source.conv_state.len())?,
                        recurrent_state: DeviceBuffer::zeroed(source.recurrent_state.len())?,
                    };
                    destination.conv_state.copy_prefix_from_device_on_stream(
                        &source.conv_state,
                        source.conv_state.len(),
                        &stream,
                    )?;
                    destination
                        .recurrent_state
                        .copy_prefix_from_device_on_stream(
                            &source.recurrent_state,
                            source.recurrent_state.len(),
                            &stream,
                        )?;
                    device_bytes = device_bytes
                        .checked_add(destination.device_bytes())
                        .ok_or_else(|| Error::Shape {
                            label: "Qwen3.6 snapshot bytes",
                            expected: "byte total without overflow".to_string(),
                            actual: source.device_bytes().to_string(),
                        })?;
                    linear_states.push(Some(destination));
                }
                None => linear_states.push(None),
            }
        }
        stream.synchronize()?;
        Ok(Qwen36SequenceSnapshot {
            model_id: self.model_id,
            position: source.position,
            linear_states,
            device_bytes,
        })
    }

    /// Restores a retained recurrent snapshot into an empty sequence.
    pub(crate) fn restore_sequence_snapshot(
        &self,
        snapshot: &Qwen36SequenceSnapshot,
        destination: &mut Qwen36SequenceState,
    ) -> Result<()> {
        if snapshot.model_id != self.model_id
            || destination.model_id != self.model_id
            || destination.position != 0
            || snapshot.position > destination.max_tokens
            || snapshot.linear_states.len() != destination.linear_states.len()
        {
            return Err(Error::Format {
                label: "Qwen3.6 sequence snapshot restore",
                detail: "snapshot and empty destination are incompatible".to_string(),
            });
        }
        let stream = CudaStream::new_non_blocking()?;
        for (snapshot, destination) in snapshot
            .linear_states
            .iter()
            .zip(&mut destination.linear_states)
        {
            match (snapshot, destination) {
                (Some(source), Some(destination)) => {
                    destination.conv_state.copy_prefix_from_device_on_stream(
                        &source.conv_state,
                        source.conv_state.len(),
                        &stream,
                    )?;
                    destination
                        .recurrent_state
                        .copy_prefix_from_device_on_stream(
                            &source.recurrent_state,
                            source.recurrent_state.len(),
                            &stream,
                        )?;
                }
                (None, None) => {}
                _ => {
                    return Err(Error::Format {
                        label: "Qwen3.6 sequence snapshot restore",
                        detail: "snapshot layer kinds differ from destination".to_string(),
                    });
                }
            }
        }
        stream.synchronize()?;
        destination.position = snapshot.position;
        Ok(())
    }

    /// Allocates single-row reference state for profiling and GPU diagnostics.
    pub fn new_reference_decode_state(
        &self,
        max_tokens: usize,
    ) -> Result<Qwen36ReferenceDecodeState> {
        let stream = CudaStream::new_blocking()?;
        let mut layer_workspaces = Vec::with_capacity(self.layers.len());
        let mut layer_states = Vec::with_capacity(self.layers.len());
        let model = Qwen36Model {
            manifest: self.manifest.clone(),
            checkpoint: self.checkpoint.clone(),
            artifact_dir: self.artifact_dir.clone(),
            bf16_storage: self.bf16_storage,
            fp8_attention_storage: self.fp8_attention_storage,
            fp8_dense_mlp_storage: self.fp8_dense_mlp_storage,
            fp8_lm_head_storage: self.fp8_lm_head_storage,
            fp8_nvfp4_cache: self.fp8_nvfp4_cache.clone(),
        };
        for block in &self.layers {
            layer_workspaces.push(block.workspace(&model, max_tokens)?);
            layer_states.push(block.sequence_state(&model, max_tokens)?);
        }
        let sequence = Qwen36ReferenceSequenceState {
            layer_states,
            position: 0,
            max_tokens,
        };
        let enable_segmented_graphs = !self.expert_paging
            && !std::env::var("EIDER_DISABLE_DECODE_GRAPHS")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
        let enable_parallel_moe = enable_segmented_graphs;
        let parallel_moe_stream = enable_parallel_moe
            .then(CudaStream::new_non_blocking)
            .transpose()?;
        let mut parallel_moe_sync = Vec::with_capacity(self.layers.len());
        if enable_parallel_moe {
            for _ in &self.layers {
                parallel_moe_sync.push(Qwen36MoeGraphSync {
                    fork: CudaEvent::new_sync()?,
                    join: CudaEvent::new_sync()?,
                });
            }
        }
        let mut state = Qwen36ReferenceDecodeState {
            stream,
            parallel_moe_stream,
            parallel_moe_sync,
            token_id_device: DeviceBuffer::zeroed(1)?,
            position_device: DeviceBuffer::zeroed(1)?,
            cache_len_device: DeviceBuffer::zeroed(1)?,
            hidden: DeviceBuffer::zeroed(self.manifest.hidden)?,
            sequence,
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
        };
        if enable_segmented_graphs {
            state.segmented_graphs = Some(self.capture_segmented_graphs(&mut state)?);
        }
        Ok(state)
    }

    fn capture_segmented_graphs(
        &self,
        state: &mut Qwen36ReferenceDecodeState,
    ) -> Result<Vec<Qwen36LayerGraphs>> {
        let mut graphs = Vec::with_capacity(self.layers.len());
        for (layer_idx, block) in self.layers.iter().enumerate() {
            let parallel_moe = state
                .parallel_moe_stream
                .as_ref()
                .zip(state.parallel_moe_sync.get(layer_idx))
                .map(|(shared_stream, sync)| Qwen36ParallelMoe {
                    shared_stream,
                    fork: &sync.fork,
                    join: &sync.join,
                });
            let (previous, current) = state.layer_workspaces.split_at_mut(layer_idx);
            let (_, current_state) = state.sequence.layer_states.split_at_mut(layer_idx);
            let hidden = if layer_idx == 0 {
                &state.hidden
            } else {
                previous[layer_idx - 1].moe.output()
            };
            let workspace = &mut current[0];
            let sequence = &mut current_state[0];
            match &block.attention {
                Qwen36Attention::LinearAttention(_) => {
                    let layer = state.stream.capture(|stream| {
                        block.enqueue_linear_pre_gdn(
                            workspace,
                            sequence,
                            &self.manifest,
                            hidden,
                            stream,
                        )?;
                        block.enqueue_linear_gdn(workspace, sequence, stream)?;
                        block.enqueue_linear_post_gdn(
                            &self.lt,
                            workspace,
                            &self.manifest,
                            hidden,
                            stream,
                            parallel_moe,
                        )
                    })?;
                    graphs.push(Qwen36LayerGraphs::Linear(Qwen36LinearLayerGraphs { layer }));
                }
                Qwen36Attention::FullAttention(_) => {
                    let graph = state.stream.capture(|stream| {
                        block.enqueue_full_layer_indexed(
                            &self.lt,
                            workspace,
                            sequence,
                            &self.manifest,
                            hidden,
                            &state.position_device,
                            &state.cache_len_device,
                            stream,
                            parallel_moe,
                        )
                    })?;
                    graphs.push(Qwen36LayerGraphs::Full(graph));
                }
            }
        }
        Ok(graphs)
    }

    /// Runs one single-row reference token for profiling and diagnostics.
    pub fn decode_reference_token(
        &self,
        state: &mut Qwen36ReferenceDecodeState,
        token_id: u32,
    ) -> Result<Qwen36NextToken> {
        self.decode_reference_token_impl(state, token_id, None, None)
    }

    /// Runs one single-row reference token with coarse CUDA-event timings.
    pub fn decode_reference_token_profiled(
        &self,
        state: &mut Qwen36ReferenceDecodeState,
        token_id: u32,
        profile: &mut QwenDecodeProfile,
    ) -> Result<Qwen36NextToken> {
        self.decode_reference_token_impl(state, token_id, Some(profile), None)
    }

    /// Runs one reference token while wrapping a selected GPU counter range.
    pub fn decode_reference_token_with_gpu_counter_probe(
        &self,
        state: &mut Qwen36ReferenceDecodeState,
        token_id: u32,
        probe: &mut Qwen36GpuCounterProbe<'_>,
    ) -> Result<Qwen36NextToken> {
        self.decode_reference_token_impl(state, token_id, None, Some(probe))
    }

    #[allow(clippy::needless_option_as_deref)]
    fn decode_reference_token_impl(
        &self,
        state: &mut Qwen36ReferenceDecodeState,
        token_id: u32,
        mut profile: Option<&mut QwenDecodeProfile>,
        mut gpu_probe: Option<&mut Qwen36GpuCounterProbe<'_>>,
    ) -> Result<Qwen36NextToken> {
        if state.sequence.position >= state.sequence.max_tokens {
            return Err(Error::Shape {
                label: "Qwen3.6 decode position",
                expected: format!("position < {}", state.sequence.max_tokens),
                actual: state.sequence.position.to_string(),
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
                self.gather_embedding(&state.token_id_device, state.hidden.output(), stream)
            })?;
            profile.embedding_ms += ms;
        } else {
            self.gather_embedding(&state.token_id_device, state.hidden.output(), stream)?;
        }

        let use_segmented_graphs =
            profile.is_none() && gpu_probe.is_none() && state.segmented_graphs.is_some();
        if !use_segmented_graphs {
            state.segmented_graphs = None;
        }

        if let Some(graphs) = state.segmented_graphs.as_ref() {
            state
                .position_device
                .copy_from_host(&[state.sequence.position as u32])?;
            state
                .cache_len_device
                .copy_from_host(&[(state.sequence.position + 1) as u32])?;
            for graph in graphs {
                match graph {
                    Qwen36LayerGraphs::Linear(graph) => {
                        graph.layer.launch(stream)?;
                    }
                    Qwen36LayerGraphs::Full(graph) => graph.launch(stream)?,
                }
            }
        } else {
            for (layer_idx, block) in self.layers.iter().enumerate() {
                let (previous, current) = state.layer_workspaces.split_at_mut(layer_idx);
                let (_, current_state) = state.sequence.layer_states.split_at_mut(layer_idx);
                let hidden = if layer_idx == 0 {
                    &state.hidden
                } else {
                    previous[layer_idx - 1].moe.output()
                };
                block.run_one_token(
                    &self.lt,
                    &mut current[0],
                    &mut current_state[0],
                    &self.manifest,
                    hidden,
                    state.sequence.position,
                    stream,
                    profile.as_deref_mut(),
                    gpu_probe.as_deref_mut(),
                )?;
            }
        }
        let hidden = state
            .layer_workspaces
            .last()
            .expect("Qwen3.6 has at least one layer")
            .moe
            .output();

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

        let (id, value) = if let Some(profile) = profile.as_deref_mut() {
            let (_, ms) = timed_cuda(stream, || {
                self.lm_head
                    .run_top1(&self.lt, &state.final_hidden, &mut state.lm_head, stream)
            })?;
            profile.lm_head_argmax_ms += ms;
            let id = state.lm_head.next_index.copy_to_host(stream)?[0];
            let value = state.lm_head.next_value.copy_to_host(stream)?[0];
            (id, value)
        } else {
            self.lm_head
                .run_top1(&self.lt, &state.final_hidden, &mut state.lm_head, stream)?;
            let id = state.lm_head.next_index.copy_to_host(stream)?[0];
            let value = state.lm_head.next_value.copy_to_host(stream)?[0];
            (id, value)
        };
        state.sequence.position += 1;
        Ok(Qwen36NextToken { id, value })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen36LinearAttentionState, Qwen36SequenceState, reorder_bf16_v_cols, reorder_bf16_v_rows,
        reorder_fp8_v_cols, reorder_fp8_v_rows, reorder_nvfp4_v_cols, reorder_nvfp4_v_rows,
    };
    use eider_cuda::{CudaStream, DeviceBuffer, PinnedHostBuffer};
    use eider_format::{ModelOptFp8Linear, ModelOptNvfp4Linear};

    #[test]
    fn recurrent_append_transaction_restores_and_commits_explicitly() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut state = Qwen36SequenceState {
            model_id: 1,
            linear_states: vec![Some(Qwen36LinearAttentionState {
                conv_state: DeviceBuffer::from_host(&[1.0, 2.0]).expect("conv"),
                recurrent_state: DeviceBuffer::from_host(&[3.0, 4.0]).expect("recurrent"),
            })],
            rollback_linear_states: vec![Some(Qwen36LinearAttentionState {
                conv_state: DeviceBuffer::zeroed(2).expect("rollback conv"),
                recurrent_state: DeviceBuffer::zeroed(2).expect("rollback recurrent"),
            })],
            rollback_position: 0,
            append_pending: false,
            position: 0,
            max_tokens: 8,
        };
        state.begin_append(&stream).expect("begin append");
        let active = state.linear_states[0].as_mut().expect("active state");
        let mutated_conv = PinnedHostBuffer::from_slice(&[9.0, 10.0]).expect("mutated conv");
        let mutated_recurrent =
            PinnedHostBuffer::from_slice(&[11.0, 12.0]).expect("mutated recurrent");
        active
            .conv_state
            .copy_range_from_pinned_on_stream(0, &mutated_conv, &stream)
            .expect("mutate conv");
        active
            .recurrent_state
            .copy_range_from_pinned_on_stream(0, &mutated_recurrent, &stream)
            .expect("mutate recurrent");
        state.abort_append(&stream).expect("abort append");
        let active = state.linear_states[0].as_ref().expect("active state");
        assert_eq!(
            active
                .conv_state
                .copy_to_host(&stream)
                .expect("conv read")
                .as_ref(),
            &[1.0, 2.0]
        );
        assert_eq!(
            active
                .recurrent_state
                .copy_to_host(&stream)
                .expect("recurrent read")
                .as_ref(),
            &[3.0, 4.0]
        );

        state.begin_append(&stream).expect("begin retry");
        let committed_conv = PinnedHostBuffer::from_slice(&[9.0, 10.0]).expect("committed conv");
        state.linear_states[0]
            .as_mut()
            .expect("active state")
            .conv_state
            .copy_range_from_pinned_on_stream(0, &committed_conv, &stream)
            .expect("mutate retry");
        state.commit_append(2);
        assert_eq!(state.position(), 2);
        assert_eq!(
            state.linear_states[0]
                .as_ref()
                .expect("active state")
                .conv_state
                .copy_to_host(&stream)
                .expect("committed read")
                .as_ref(),
            &[9.0, 10.0]
        );
    }

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

    #[test]
    fn reorder_nvfp4_v_rows_moves_packed_values_and_scales_together() {
        let mut packed_weight = Vec::new();
        let mut weight_scale = Vec::new();
        for head in 0..4u8 {
            for _ in 0..16 {
                packed_weight.extend_from_slice(&[head; 8]);
                weight_scale.push(head + 10);
            }
        }
        let reordered = reorder_nvfp4_v_rows(
            ModelOptNvfp4Linear {
                prefix: "z".to_string(),
                out_features: 64,
                in_features: 16,
                packed_weight,
                weight_scale,
                weight_scale_2: 1.0,
                input_scale: 1.0,
            },
            2,
            4,
            16,
        );
        assert_eq!(
            [0, 16, 32, 48].map(|row| reordered.packed_weight[row * 8]),
            [0, 2, 1, 3]
        );
        assert_eq!(
            [0, 16, 32, 48].map(|row| reordered.weight_scale[row]),
            [10, 12, 11, 13]
        );
    }

    #[test]
    fn reorder_nvfp4_v_cols_moves_packed_values_and_scales_together() {
        let reordered = reorder_nvfp4_v_cols(
            ModelOptNvfp4Linear {
                prefix: "out".to_string(),
                out_features: 1,
                in_features: 64,
                packed_weight: [vec![0; 8], vec![1; 8], vec![2; 8], vec![3; 8]].concat(),
                weight_scale: vec![10, 11, 12, 13],
                weight_scale_2: 1.0,
                input_scale: 1.0,
            },
            2,
            4,
            16,
        );
        assert_eq!(
            [0, 8, 16, 24].map(|offset| reordered.packed_weight[offset]),
            [0, 2, 1, 3]
        );
        assert_eq!(reordered.weight_scale, [10, 12, 11, 13]);
    }

    #[test]
    fn fp8_to_nvfp4_conversion_commutes_with_v_row_reordering() {
        let source = ModelOptFp8Linear {
            prefix: "z".to_string(),
            out_features: 64,
            in_features: 16,
            weight: (0..64 * 16).map(|index| (index % 239) as u8).collect(),
            weight_scale: 0.5,
            channel_weight_scale: Some((0..64).map(|row| 0.25 + row as f32 / 64.0).collect()),
            input_scale: None,
        };
        let converted_after =
            ModelOptNvfp4Linear::quantize_fp8(&reorder_fp8_v_rows(source.clone(), 2, 4, 16))
                .expect("convert reordered FP8");
        let reordered_after = reorder_nvfp4_v_rows(
            ModelOptNvfp4Linear::quantize_fp8(&source).expect("convert FP8"),
            2,
            4,
            16,
        );

        assert_eq!(reordered_after.packed_weight, converted_after.packed_weight);
        assert_eq!(reordered_after.weight_scale, converted_after.weight_scale);
    }

    #[test]
    fn fp8_to_nvfp4_conversion_commutes_with_v_column_reordering() {
        let source = ModelOptFp8Linear {
            prefix: "out".to_string(),
            out_features: 2,
            in_features: 64,
            weight: (0..2 * 64).map(|index| (index % 239) as u8).collect(),
            weight_scale: 0.75,
            channel_weight_scale: Some(vec![0.5, 1.25]),
            input_scale: None,
        };
        let converted_after =
            ModelOptNvfp4Linear::quantize_fp8(&reorder_fp8_v_cols(source.clone(), 2, 4, 16))
                .expect("convert reordered FP8");
        let reordered_after = reorder_nvfp4_v_cols(
            ModelOptNvfp4Linear::quantize_fp8(&source).expect("convert FP8"),
            2,
            4,
            16,
        );

        assert_eq!(reordered_after.packed_weight, converted_after.packed_weight);
        assert_eq!(reordered_after.weight_scale, converted_after.weight_scale);
    }

    #[test]
    fn reorder_bf16_v_rows_moves_complete_head_blocks() {
        let reordered = reorder_bf16_v_rows((0..16).collect(), 2, 4, 2);
        assert_eq!(
            reordered,
            vec![0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        );
    }

    #[test]
    fn reorder_bf16_v_cols_moves_complete_head_blocks_per_row() {
        let reordered = reorder_bf16_v_cols(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17],
            2,
            2,
            4,
            2,
        );
        assert_eq!(
            reordered,
            vec![0, 1, 4, 5, 2, 3, 6, 7, 10, 11, 14, 15, 12, 13, 16, 17]
        );
    }
}
