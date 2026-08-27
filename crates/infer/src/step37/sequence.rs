//! Shared paged sequence storage for Step-3.7.

use super::{HEAD_DIM, KV_HEADS, Step37DecodeState, Step37TextModel};
use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use nvfp4::{CudaStream, Error, Result};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, AppendReservation, CacheError, SequenceCache, SequenceId,
};

/// Scheduler-owned Step-3.7 shared KV manager.
pub type Step37SequenceCache = SequenceCache<Sm12xPageBackend, ()>;

/// Per-row append capability and stable page table passed into model execution.
pub(crate) struct Step37Append<'a> {
    pub(crate) reservation: &'a AppendReservation,
    pub(crate) page_table: &'a nvfp4::DeviceBuffer<u32>,
}

/// One admitted Step-3.7 sequence and all request-private execution state.
pub struct Step37Sequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: Step37DecodeState,
}

impl Step37Sequence {
    /// Admits an empty sequence into a non-retaining cache.
    pub fn admit(
        model: &Step37TextModel,
        cache: &mut Step37SequenceCache,
        max_tokens: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        let state = model.new_sequence_state(max_tokens)?;
        let mut page_table = Sm12xPageTable::new(max_tokens)?;
        let outcome = cache
            .admit(
                None,
                AdmissionRequest {
                    max_position: max_tokens,
                    private_state_bytes: state.device_bytes(),
                    page_table_bytes: page_table.managed_bytes(),
                    allow_emergency: false,
                },
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut page_table,
                },
                |snapshot, position| {
                    debug_assert!(snapshot.is_none());
                    debug_assert_eq!(position, 0);
                    Ok(())
                },
            )
            .map_err(step37_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "Step-3.7 sequence admission",
                detail: "configured cache has insufficient capacity".to_string(),
            });
        };
        stream.synchronize()?;
        Ok(Self {
            cache_id,
            page_table,
            state,
        })
    }

    pub(crate) fn from_admission(
        cache_id: SequenceId,
        page_table: Sm12xPageTable,
        state: Step37DecodeState,
    ) -> Self {
        Self {
            cache_id,
            page_table,
            state,
        }
    }

    pub fn position(&self) -> usize {
        self.state.len()
    }

    pub fn max_tokens(&self) -> usize {
        self.state.max_tokens()
    }

    pub fn device_bytes(&self) -> usize {
        self.state.device_bytes() + self.page_table.managed_bytes()
    }

    pub fn finish(self, cache: &mut Step37SequenceCache, stream: &CudaStream) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut page_table,
                },
            )
            .map_err(step37_cache_error)
    }
}

pub fn new_step37_sequence_cache(
    model: &Step37TextModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<Step37SequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "Step-3.7 sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let pages_per_sequence = max_context_tokens.div_ceil(nvfp4::SM12X_KV_PAGE_TOKENS);
    let page_slots = sequence_capacity
        .checked_mul(pages_per_sequence)
        .ok_or_else(|| Error::Shape {
            label: "Step-3.7 sequence cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!(
                "sequences={sequence_capacity} pages_per_sequence={pages_per_sequence}"
            ),
        })?;
    let backend = Sm12xPageBackend::new(
        std::iter::repeat_n(true, model.layer_count()),
        page_slots,
        KV_HEADS,
        HEAD_DIM,
    )?;
    let page_bytes = seqcache::PageBackend::page_bytes(&backend);
    let private_bytes = model.new_sequence_state(max_context_tokens)?.device_bytes();
    let table_bytes = Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let fixed_bytes = private_bytes
        .checked_add(table_bytes)
        .and_then(|bytes| bytes.checked_mul(sequence_capacity))
        .ok_or_else(|| Error::Shape {
            label: "Step-3.7 sequence cache private bytes",
            expected: "private byte count without overflow".to_string(),
            actual: format!(
                "private={private_bytes} table={table_bytes} sequences={sequence_capacity}"
            ),
        })?;
    let managed_bytes = page_bytes
        .checked_mul(page_slots)
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or_else(|| Error::Shape {
            label: "Step-3.7 sequence cache managed bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("page_bytes={page_bytes} page_slots={page_slots}"),
        })?;
    Step37SequenceCache::new(
        seqcache::CacheConfig {
            page_tokens: nvfp4::SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: 0,
            max_prefix_entries: Some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(step37_cache_error)
}

pub(crate) fn step37_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Step-3.7 sequence cache",
        detail: error.to_string(),
    }
}
