//! Shared paged sequence storage for Gemma 4.

use super::{Gemma4DecodeState, Gemma4Model};
use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use eider_cuda::{CudaStream, Error, Result};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, AppendReservation, CacheError, SequenceCache, SequenceId,
};

pub type Gemma4SequenceCache = SequenceCache<Sm12xPageBackend, ()>;

pub(crate) struct Gemma4Append<'a> {
    pub(crate) reservation: &'a AppendReservation,
    pub(crate) page_table: &'a eider_cuda::DeviceBuffer<u32>,
}

pub struct Gemma4Sequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: Gemma4DecodeState,
}

impl Gemma4Sequence {
    pub fn admit(
        model: &Gemma4Model,
        cache: &mut Gemma4SequenceCache,
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
            .map_err(gemma4_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "Gemma 4 sequence admission",
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
        state: Gemma4DecodeState,
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

    pub fn finish(self, cache: &mut Gemma4SequenceCache, stream: &CudaStream) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut page_table,
                },
            )
            .map_err(gemma4_cache_error)
    }
}

pub fn new_gemma4_sequence_cache(
    model: &Gemma4Model,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<Gemma4SequenceCache> {
    new_gemma4_sequence_cache_with_budget(model, sequence_capacity, max_context_tokens, None)
}

pub(crate) fn new_gemma4_sequence_cache_with_budget(
    model: &Gemma4Model,
    sequence_capacity: usize,
    max_context_tokens: usize,
    retained_budget_bytes: Option<usize>,
) -> Result<Gemma4SequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "Gemma 4 sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let pages_per_sequence = max_context_tokens.div_ceil(eider_cuda::SM12X_KV_PAGE_TOKENS);
    let eager_page_slots = sequence_capacity
        .checked_mul(pages_per_sequence)
        .ok_or_else(|| Error::Shape {
            label: "Gemma 4 sequence cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!(
                "sequences={sequence_capacity} pages_per_sequence={pages_per_sequence}"
            ),
        })?;
    let probe = Sm12xPageBackend::new_heterogeneous(model.sequence_layer_geometries(), 1)?;
    let page_bytes = seqcache::PageBackend::page_bytes(&probe);
    let private_bytes = model.new_sequence_state(max_context_tokens)?.device_bytes();
    let table_bytes = Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let fixed_bytes = private_bytes
        .checked_add(table_bytes)
        .and_then(|bytes| bytes.checked_mul(sequence_capacity))
        .ok_or_else(|| Error::Shape {
            label: "Gemma 4 sequence cache private bytes",
            expected: "private byte count without overflow".to_string(),
            actual: format!(
                "private={private_bytes} table={table_bytes} sequences={sequence_capacity}"
            ),
        })?;
    let eager_managed_bytes = page_bytes
        .checked_mul(eager_page_slots)
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or_else(|| Error::Shape {
            label: "Gemma 4 sequence cache bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("page_bytes={page_bytes} page_slots={eager_page_slots}"),
        })?;
    let retained_bytes = retained_budget_bytes.unwrap_or(0);
    let managed_bytes = eager_managed_bytes
        .checked_add(retained_bytes)
        .ok_or_else(|| Error::Shape {
            label: "Gemma 4 sequence cache budget",
            expected: "active and retained budgets without overflow".to_string(),
            actual: format!("active={eager_managed_bytes} retained={retained_bytes}"),
        })?;
    let retained_pages = retained_bytes / page_bytes;
    let page_slots = eager_page_slots
        .checked_add(retained_pages)
        .ok_or_else(|| Error::Shape {
            label: "Gemma 4 sequence cache pages",
            expected: "active and retained page counts without overflow".to_string(),
            actual: format!("active={eager_page_slots} retained={retained_pages}"),
        })?;
    let backend =
        Sm12xPageBackend::new_heterogeneous(model.sequence_layer_geometries(), page_slots)?;
    Gemma4SequenceCache::new(
        seqcache::CacheConfig {
            page_tokens: eider_cuda::SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: 0,
            max_prefix_entries: retained_budget_bytes.is_none().then_some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(gemma4_cache_error)
}

pub(crate) fn gemma4_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Gemma 4 sequence cache",
        detail: error.to_string(),
    }
}
