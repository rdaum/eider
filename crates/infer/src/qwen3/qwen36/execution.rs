//! Persistent Qwen execution resources.
//!
//! This module owns the CUDA-backed state needed to execute Qwen batches. The
//! scheduler selects requests and passes rows to it, but does not define or
//! allocate model execution resources.

use super::{
    Qwen36DecodeBatchWorkspace, Qwen36MtpDraftWorkspace, Qwen36PrefillBatchWorkspace,
    Qwen36SequenceCache, Qwen36SpeculativeCycleWorkspace, Qwen36TextModel,
    Qwen38DFlash2PrefixCache, Qwen38DFlash2Workspace,
};
use crate::sm12x_cache::{Sm12xPageBackend, Sm12xPageTable};
use eider_cuda::{CudaStream, DeviceBuffer, Error, Result, SM12X_KV_PAGE_TOKENS};
use seqcache::PageBackend;
use std::mem::size_of;

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
