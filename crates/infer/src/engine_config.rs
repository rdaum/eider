//! Inference-owned configuration for loading a model engine.

use crate::nemotron3::Nemotron3StorageConfig;
use crate::qwen3::qwen36::{Qwen36Bf16StorageConfig, Qwen36Fp8Storage};
use crate::step37::Step37Bf16StorageConfig;
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::scheduler::SchedulerConfig;
use std::path::PathBuf;

/// Deployment configuration consumed while loading one model engine.
#[derive(Clone, Debug)]
pub struct InferenceEngineConfig {
    /// Immutable checkpoint snapshot directory.
    pub model_dir: PathBuf,
    /// Directory for derived model artifacts.
    pub artifact_dir: PathBuf,
    /// Optional Muse Glimmer DFlash GGUF.
    pub dflash_gguf: Option<PathBuf>,
    /// Optional Qwen DFlash2 directory.
    pub dflash2_dir: Option<PathBuf>,
    /// Request scheduling limits.
    pub scheduler: SchedulerConfig,
    /// Logical prefix-retention policy.
    pub sequence_cache: SequenceCacheConfig,
    /// Qwen BF16 storage policy.
    pub qwen_bf16_storage: Qwen36Bf16StorageConfig,
    /// Qwen FP8 attention storage policy.
    pub qwen_fp8_attention_storage: Qwen36Fp8Storage,
    /// Qwen FP8 dense-MLP storage policy.
    pub qwen_fp8_dense_mlp_storage: Qwen36Fp8Storage,
    /// Qwen FP8 LM-head storage policy.
    pub qwen_fp8_lm_head_storage: Qwen36Fp8Storage,
    /// Step routed-expert residency capacity.
    pub step_expert_capacity: usize,
    /// DeepSeek routed-expert residency capacity.
    pub deepseek_expert_capacity: usize,
    /// Step BF16 storage policy.
    pub step_bf16_storage: Step37Bf16StorageConfig,
    /// Nemotron storage policy.
    pub nemotron_storage: Nemotron3StorageConfig,
    /// Bounded API event-channel capacity.
    pub event_capacity: usize,
}

impl InferenceEngineConfig {
    /// Creates a configuration with the standard deployment defaults.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        Self {
            artifact_dir: model_dir.join(".eider-cache"),
            dflash_gguf: None,
            dflash2_dir: None,
            model_dir,
            scheduler: SchedulerConfig::default(),
            sequence_cache: SequenceCacheConfig::default(),
            qwen_bf16_storage: Qwen36Bf16StorageConfig::default(),
            qwen_fp8_attention_storage: Qwen36Fp8Storage::default(),
            qwen_fp8_dense_mlp_storage: Qwen36Fp8Storage::default(),
            qwen_fp8_lm_head_storage: Qwen36Fp8Storage::default(),
            step_expert_capacity: 240,
            deepseek_expert_capacity: 8,
            step_bf16_storage: Step37Bf16StorageConfig::default(),
            nemotron_storage: Nemotron3StorageConfig::default(),
            event_capacity: 256,
        }
    }
}
