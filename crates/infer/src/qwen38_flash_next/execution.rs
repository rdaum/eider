//! Persistent Flash Next execution resources.
//!
//! This state owns the loaded model and the CUDA-backed resources reused over
//! service ticks. Request admission and output policy stay in the runtime.

use super::{
    Qwen38FlashNextCacheConfig, Qwen38FlashNextModel, Qwen38FlashNextMtpSequenceCache,
    Qwen38FlashNextMtpSequenceState, Qwen38FlashNextMtpWorkspace, Qwen38FlashNextPrefillWorkspace,
    Qwen38FlashNextSequence, Qwen38FlashNextSequenceCache, Qwen38FlashNextSpeculativeFrontier,
    Qwen38FlashNextSpeculativeWorkspace, new_qwen38_flash_next_mtp_sequence_cache,
    new_qwen38_flash_next_sequence_cache_with_config,
};
use crate::qwen3::infer::QwenLayerKind;
use eider_cuda::{DeviceBuffer, Error, GpuTokenSampler, Result};
use std::collections::BTreeMap;

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
    pub(crate) sequences: Qwen38FlashNextSequencePool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Qwen38FlashNextSequenceId(u64);

pub(crate) struct Qwen38FlashNextExecutionSequence {
    pub(crate) sequence: Box<Qwen38FlashNextSequence>,
    pub(crate) mtp_sequence: Option<Qwen38FlashNextMtpSequenceState>,
    pub(crate) speculative_frontier: Option<Qwen38FlashNextSpeculativeFrontier>,
    pub(crate) device_token_counts: Option<DeviceBuffer<u32>>,
    device_bytes: usize,
}

pub(crate) struct Qwen38FlashNextSequencePool {
    sequences: BTreeMap<Qwen38FlashNextSequenceId, Qwen38FlashNextExecutionSequence>,
    next_id: u64,
}

pub(crate) struct Qwen38FlashNextSequenceLease<'a> {
    pool: &'a mut Qwen38FlashNextSequencePool,
    id: Qwen38FlashNextSequenceId,
    sequence: Option<Qwen38FlashNextExecutionSequence>,
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
            sequences: Qwen38FlashNextSequencePool::new(),
        })
    }
}

impl Qwen38FlashNextExecutionSequence {
    pub(crate) fn new(
        sequence: Qwen38FlashNextSequence,
        mtp_sequence: Option<Qwen38FlashNextMtpSequenceState>,
        speculative_frontier: Option<Qwen38FlashNextSpeculativeFrontier>,
        device_token_counts: Option<DeviceBuffer<u32>>,
        device_bytes: usize,
    ) -> Self {
        Self {
            sequence: Box::new(sequence),
            mtp_sequence,
            speculative_frontier,
            device_token_counts,
            device_bytes,
        }
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub(crate) fn finish(
        mut self,
        sequence_cache: &mut Qwen38FlashNextSequenceCache,
        mtp_sequence_cache: Option<&mut Qwen38FlashNextMtpSequenceCache>,
    ) -> Result<()> {
        if let Some(mtp_sequence) = self.mtp_sequence.take() {
            let mtp_sequence_cache = mtp_sequence_cache.expect("MTP sequence has its cache");
            mtp_sequence.finish(mtp_sequence_cache, self.sequence.state.stream())?;
        }
        self.sequence.finish(sequence_cache)
    }
}

impl Qwen38FlashNextSequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub(crate) fn insert(
        &mut self,
        sequence: Qwen38FlashNextExecutionSequence,
    ) -> Result<Qwen38FlashNextSequenceId> {
        let id = Qwen38FlashNextSequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }

    pub(crate) fn release(
        &mut self,
        id: Qwen38FlashNextSequenceId,
    ) -> Result<Qwen38FlashNextExecutionSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }

    pub(crate) fn lease(
        &mut self,
        id: Qwen38FlashNextSequenceId,
    ) -> Result<Qwen38FlashNextSequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(Qwen38FlashNextSequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl Qwen38FlashNextSequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut Qwen38FlashNextExecutionSequence {
        self.sequence.as_mut().expect("active lease")
    }
}

impl Drop for Qwen38FlashNextSequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}
