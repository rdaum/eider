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
    pub model_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub dflash_gguf: Option<PathBuf>,
    pub dflash2_dir: Option<PathBuf>,
    pub scheduler: SchedulerConfig,
    pub sequence_cache: SequenceCacheConfig,
    pub qwen_bf16_storage: Qwen36Bf16StorageConfig,
    pub qwen_fp8_attention_storage: Qwen36Fp8Storage,
    pub qwen_fp8_dense_mlp_storage: Qwen36Fp8Storage,
    pub qwen_fp8_lm_head_storage: Qwen36Fp8Storage,
    pub step_expert_capacity: usize,
    pub deepseek_expert_capacity: usize,
    pub step_bf16_storage: Step37Bf16StorageConfig,
    pub nemotron_storage: Nemotron3StorageConfig,
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
