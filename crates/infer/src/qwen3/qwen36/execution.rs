//! Persistent Qwen execution resources.
//!
//! This module owns the CUDA-backed state needed to execute Qwen batches. The
//! scheduler selects requests and passes rows to it, but does not define or
//! allocate model execution resources.

use super::{
    Qwen36DecodeBatchWorkspace, Qwen36MtpDraftWorkspace, Qwen36PrefillBatchWorkspace,
    Qwen36Sequence, Qwen36SequenceCache, Qwen36SpeculativeCycleWorkspace, Qwen36TextModel,
    Qwen38DFlash2PrefixCache, Qwen38DFlash2Workspace,
};
use crate::sm12x_cache::{Sm12xPageBackend, Sm12xPageTable};
use eider_cuda::{CudaStream, DeviceBuffer, Error, Result, SM12X_KV_PAGE_TOKENS};
use seqcache::PageBackend;
use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

/// Opaque identity for CUDA-backed Qwen sequence state.
///
/// IDs are never reused by one execution state. A request retaining an ID
/// after release therefore cannot resolve another sequence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Qwen36SequenceId(u64);

/// CUDA-backed state retained for an admitted Qwen sequence.
pub(crate) struct Qwen36ExecutionSequence {
    pub(crate) sequence: Qwen36Sequence,
    pub(crate) device_token_counts: Option<DeviceBuffer<u32>>,
    pub(crate) device_bytes: usize,
}

/// Model-owned storage for all live Qwen sequences.
pub(crate) struct Qwen36SequencePool {
    sequences: BTreeMap<Qwen36SequenceId, Qwen36ExecutionSequence>,
    next_id: u64,
}

/// Temporary exclusive access to one batch of engine-owned sequences.
///
/// Dropping a lease restores every entry, including while unwinding an error.
pub(crate) struct Qwen36SequenceBatch<'a> {
    pool: &'a mut Qwen36SequencePool,
    entries: Vec<(Qwen36SequenceId, Qwen36ExecutionSequence)>,
}

impl Qwen36SequencePool {
    fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Inserts newly admitted state and returns its non-reusable identity.
    pub(crate) fn insert(
        &mut self,
        sequence: Qwen36Sequence,
        device_token_counts: Option<DeviceBuffer<u32>>,
    ) -> Result<(Qwen36SequenceId, usize)> {
        let device_bytes = sequence
            .device_bytes()
            .checked_add(
                device_token_counts
                    .as_ref()
                    .map_or(0, DeviceBuffer::device_bytes),
            )
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 admitted sequence bytes",
                expected: "sequence state and sampling bytes without overflow".to_string(),
                actual: "byte count overflow".to_string(),
            })?;
        let id = Qwen36SequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Qwen3.6 sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        let previous = self.sequences.insert(
            id,
            Qwen36ExecutionSequence {
                sequence,
                device_token_counts,
                device_bytes,
            },
        );
        debug_assert!(previous.is_none());
        Ok((id, device_bytes))
    }

    /// Permanently removes one sequence after it has finished or been cancelled.
    pub(crate) fn release(&mut self, id: Qwen36SequenceId) -> Result<Qwen36ExecutionSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Qwen3.6 execution sequence",
            detail: format!("unknown or released sequence {}", id.0),
        })
    }

    /// Exclusively leases all requested entries for one model submission.
    pub(crate) fn lease_many(
        &mut self,
        ids: &[Qwen36SequenceId],
    ) -> Result<Qwen36SequenceBatch<'_>> {
        let unique_ids = ids.iter().copied().collect::<BTreeSet<_>>();
        if unique_ids.len() != ids.len() {
            return Err(Error::Format {
                label: "Qwen3.6 execution sequence lease",
                detail: "duplicate sequence ID in one batch".to_string(),
            });
        }
        let mut entries = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(entry) = self.sequences.remove(&id) else {
                for (restored_id, restored) in entries.drain(..) {
                    let previous = self.sequences.insert(restored_id, restored);
                    debug_assert!(previous.is_none());
                }
                return Err(Error::Format {
                    label: "Qwen3.6 execution sequence lease",
                    detail: format!("unknown or released sequence {}", id.0),
                });
            };
            entries.push((id, entry));
        }
        Ok(Qwen36SequenceBatch {
            pool: self,
            entries,
        })
    }

    /// Returns the number of live CUDA-backed sequences.
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Returns the device allocation retained for one live sequence.
    pub(crate) fn device_bytes(&self, id: Qwen36SequenceId) -> Option<usize> {
        self.sequences.get(&id).map(|entry| entry.device_bytes)
    }

    /// Reports whether an identity still resolves to live execution state.
    #[cfg(test)]
    pub(crate) fn contains(&self, id: Qwen36SequenceId) -> bool {
        self.sequences.contains_key(&id)
    }
}

impl Qwen36SequenceBatch<'_> {
    /// Returns engine sequence state in the same order as the leased IDs.
    pub(crate) fn entries_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut Qwen36ExecutionSequence> {
        self.entries.iter_mut().map(|(_, entry)| entry)
    }

    /// Returns one engine sequence state by its batch row.
    pub(crate) fn entry_mut(&mut self, row: usize) -> &mut Qwen36ExecutionSequence {
        &mut self.entries[row].1
    }
}

impl Drop for Qwen36SequenceBatch<'_> {
    fn drop(&mut self) {
        for (id, entry) in self.entries.drain(..) {
            let previous = self.pool.sequences.insert(id, entry);
            debug_assert!(previous.is_none());
        }
    }
}

/// Capacity and retention limits used to build one Qwen execution state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36ExecutionConfig {
    pub(crate) decode_capacity: usize,
    pub(crate) prefill_sequence_capacity: usize,
    pub(crate) prefill_token_capacity: usize,
    pub(crate) max_active_sequences: usize,
    pub(crate) max_context_tokens: usize,
    pub(crate) speculative_drafts: usize,
    pub(crate) retained_prefix_bytes: usize,
}

/// CUDA-backed Qwen state retained across scheduler ticks.
pub(crate) struct Qwen36ExecutionState<'model> {
    pub(crate) model: &'model Qwen36TextModel,
    pub(crate) decode_workspaces: Vec<Qwen36DecodeBatchWorkspace>,
    pub(crate) prefill_workspace: Qwen36PrefillBatchWorkspace,
    pub(crate) sequence_cache: Qwen36SequenceCache,
    pub(crate) cache_stream: CudaStream,
    pub(crate) spec_workspace: Option<Qwen36SpeculativeCycleWorkspace>,
    pub(crate) mtp_workspace: Option<Qwen36MtpDraftWorkspace>,
    pub(crate) mtp_hidden_scratch: Option<DeviceBuffer<f32>>,
    pub(crate) dflash2_workspace: Option<Qwen38DFlash2Workspace>,
    pub(crate) dflash2_prefix_cache: Qwen38DFlash2PrefixCache,
    pub(crate) sequences: Qwen36SequencePool,
}

impl<'model> Qwen36ExecutionState<'model> {
    /// Allocates the model, cache, and workspace state for a scheduler.
    pub(crate) fn new(
        model: &'model Qwen36TextModel,
        config: Qwen36ExecutionConfig,
    ) -> Result<Self> {
        let mut decode_workspaces = decode_capacity_classes(config.decode_capacity)
            .into_iter()
            .map(|capacity| model.new_decode_batch_workspace(capacity, config.max_context_tokens))
            .collect::<Result<Vec<_>>>()?;
        let mut prefill_workspace = model.new_prefill_batch_workspace(
            config.prefill_sequence_capacity,
            config.prefill_token_capacity,
            config.max_context_tokens,
        )?;
        if model.dflash2_enabled() && config.speculative_drafts > 0 {
            for workspace in &mut decode_workspaces {
                model.enable_dflash2_decode_capture(workspace)?;
            }
            model.enable_dflash2_prefill_capture(&mut prefill_workspace)?;
        }

        let probe_backend = Sm12xPageBackend::new(
            model
                .manifest()
                .layer_kinds
                .iter()
                .map(|kind| *kind == crate::qwen3::infer::QwenLayerKind::FullAttention),
            1,
            model.manifest().kv_heads,
            model.manifest().head_dim,
        )?;
        let page_bytes = probe_backend.page_bytes();
        let private_state_bytes = model.new_sequence_state(1)?.device_bytes();
        let page_table_bytes = Sm12xPageTable::new(config.max_context_tokens)?.managed_bytes();
        let sampling_bytes = model
            .manifest()
            .vocab
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache sampling bytes",
                expected: "vocabulary byte count without overflow".to_string(),
                actual: model.manifest().vocab.to_string(),
            })?;
        let fixed_per_sequence = private_state_bytes
            .checked_add(page_table_bytes)
            .and_then(|bytes| bytes.checked_add(sampling_bytes))
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache fixed bytes",
                expected: "per-sequence byte count without overflow".to_string(),
                actual: format!(
                    "private={private_state_bytes} table={page_table_bytes} sampling={sampling_bytes}"
                ),
            })?;
        let fixed_capacity = fixed_per_sequence
            .checked_mul(config.max_active_sequences)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache fixed capacity",
                expected: "active fixed-state byte count without overflow".to_string(),
                actual: format!(
                    "per_sequence={fixed_per_sequence} active={}",
                    config.max_active_sequences
                ),
            })?;
        let eager_pages = config
            .max_context_tokens
            .div_ceil(SM12X_KV_PAGE_TOKENS)
            .checked_mul(config.max_active_sequences)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 active sequence-cache pages",
                expected: "page count without overflow".to_string(),
                actual: format!(
                    "context={} active={}",
                    config.max_context_tokens, config.max_active_sequences
                ),
            })?;
        let active_page_bytes =
            eager_pages
                .checked_mul(page_bytes)
                .ok_or_else(|| Error::Shape {
                    label: "Qwen3.6 active sequence-cache page bytes",
                    expected: "page byte count without overflow".to_string(),
                    actual: format!("pages={eager_pages} page_bytes={page_bytes}"),
                })?;
        let active_capacity = fixed_capacity
            .checked_add(active_page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 active sequence-cache capacity",
                expected: "managed byte count without overflow".to_string(),
                actual: format!("fixed={fixed_capacity} pages={active_page_bytes}"),
            })?;
        let dflash2_retained_bytes = if model.dflash2_enabled() && config.speculative_drafts > 0 {
            config.retained_prefix_bytes / 8
        } else {
            0
        };
        let target_retained_bytes = config.retained_prefix_bytes - dflash2_retained_bytes;
        let snapshot_capacity = target_retained_bytes / 4;
        let managed_bytes = active_capacity
            .checked_add(target_retained_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache capacity",
                expected: "active and retained byte count without overflow".to_string(),
                actual: format!("active={active_capacity} retained={target_retained_bytes}"),
            })?;
        let page_slots = eager_pages
            .checked_add(target_retained_bytes.saturating_sub(snapshot_capacity) / page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache page slots",
                expected: "page count without overflow".to_string(),
                actual: format!("active={eager_pages} retained={target_retained_bytes}"),
            })?;
        if page_slots == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 sequence-cache capacity",
                expected: format!(
                    "budget greater than fixed active capacity {fixed_capacity}, snapshot reserve {snapshot_capacity}, and one {page_bytes}-byte page"
                ),
                actual: managed_bytes.to_string(),
            });
        }
        let backend = Sm12xPageBackend::new(
            model
                .manifest()
                .layer_kinds
                .iter()
                .map(|kind| *kind == crate::qwen3::infer::QwenLayerKind::FullAttention),
            page_slots,
            model.manifest().kv_heads,
            model.manifest().head_dim,
        )?;
        let sequence_cache = Qwen36SequenceCache::new(
            seqcache::CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: managed_bytes,
                max_snapshot_bytes: snapshot_capacity,
                max_prefix_entries: (target_retained_bytes == 0).then_some(0),
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(|error| Error::Format {
            label: "Qwen3.6 sequence-cache configuration",
            detail: error.to_string(),
        })?;

        Ok(Self {
            model,
            decode_workspaces,
            prefill_workspace,
            sequence_cache,
            cache_stream: CudaStream::new_blocking()?,
            spec_workspace: None,
            mtp_workspace: None,
            mtp_hidden_scratch: None,
            dflash2_workspace: None,
            dflash2_prefix_cache: Qwen38DFlash2PrefixCache::new(dflash2_retained_bytes),
            sequences: Qwen36SequencePool::new(),
        })
    }
}

pub(crate) fn decode_capacity_classes(capacity: usize) -> Vec<usize> {
    let mut classes = Vec::new();
    let mut value = 1;
    while value < capacity {
        classes.push(value);
        value = value.saturating_mul(2);
    }
    classes.push(capacity);
    classes
}
