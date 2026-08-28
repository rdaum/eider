//! Persistent Flash Next execution resources.
//!
//! This state owns the loaded model and the CUDA-backed resources reused over
//! service ticks. Request admission and output policy stay in the runtime.

use super::{
    Qwen38FlashNextCacheConfig, Qwen38FlashNextModel, Qwen38FlashNextMtpSequenceCache,
    Qwen38FlashNextMtpWorkspace, Qwen38FlashNextPrefillWorkspace, Qwen38FlashNextSequenceCache,
    Qwen38FlashNextSpeculativeWorkspace, new_qwen38_flash_next_mtp_sequence_cache,
    new_qwen38_flash_next_sequence_cache_with_config,
};
use crate::qwen3::infer::QwenLayerKind;
use eider_cuda::{GpuTokenSampler, Result};

/// Capacity and retention limits for one Flash Next execution state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen38FlashNextExecutionConfig {
    pub(crate) max_active_sequences: usize,
    pub(crate) max_context_tokens: usize,
    pub(crate) prefill_token_capacity: usize,
    pub(crate) speculative_drafts: usize,
    pub(crate) retained_prefix_bytes: usize,
}

/// CUDA-backed Flash Next state retained across runtime service ticks.
pub(crate) struct Qwen38FlashNextExecutionState {
    pub(crate) model: Qwen38FlashNextModel,
    pub(crate) prefill_workspace: Qwen38FlashNextPrefillWorkspace,
    pub(crate) sequence_cache: Qwen38FlashNextSequenceCache,
    pub(crate) mtp_sequence_cache: Option<Qwen38FlashNextMtpSequenceCache>,
    pub(crate) mtp_workspace: Option<Qwen38FlashNextMtpWorkspace>,
    pub(crate) speculative_workspace: Option<Qwen38FlashNextSpeculativeWorkspace>,
    pub(crate) gpu_sampler: GpuTokenSampler,
}

impl Qwen38FlashNextExecutionState {
    /// Allocates the persistent model, cache, and workspace resources.
    pub(crate) fn new(
        model: Qwen38FlashNextModel,
        config: Qwen38FlashNextExecutionConfig,
    ) -> Result<Self> {
        let prefill_workspace = model.new_prefill_workspace(config.prefill_token_capacity)?;
        let sequence_cache = new_qwen38_flash_next_sequence_cache_with_config(
            &model,
            config.max_active_sequences,
            config.max_context_tokens,
            Qwen38FlashNextCacheConfig {
                max_retained_bytes: config.retained_prefix_bytes,
            },
        )?;
        let gpu_sampler = GpuTokenSampler::new(1, model.config().vocab)?;
        let mtp_sequence_cache = (config.speculative_drafts == 1)
            .then(|| {
                let target_qsa_layers = model
                    .manifest()
                    .layer_kinds
                    .iter()
                    .filter(|&&kind| kind == QwenLayerKind::FullAttention)
                    .count()
                    .max(1);
                new_qwen38_flash_next_mtp_sequence_cache(
                    &model,
                    config.max_active_sequences,
                    config.max_context_tokens,
                    config.retained_prefix_bytes / target_qsa_layers,
                )
            })
            .transpose()?;
        let mtp_workspace = (config.speculative_drafts == 1)
            .then(|| {
                model.new_mtp_workspace(config.max_context_tokens, config.prefill_token_capacity)
            })
            .transpose()?;
        let speculative_workspace = (config.speculative_drafts == 1)
            .then(|| model.new_speculative_workspace(1))
            .transpose()?;

        Ok(Self {
            model,
            prefill_workspace,
            sequence_cache,
            mtp_sequence_cache,
            mtp_workspace,
            speculative_workspace,
            gpu_sampler,
        })
    }
}
