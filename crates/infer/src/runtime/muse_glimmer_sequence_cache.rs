//! Shared paged sequence storage for Muse Glimmer.

use super::sm12x_sequence_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use crate::muse_glimmer::{MuseGlimmerDecodeState, MuseGlimmerModel, MuseGlimmerSequenceSnapshot};
use nvfp4::{Error, Result, SM12X_KV_PAGE_TOKENS};
use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, CacheConfig, CacheError, PageBackend, SequenceCache,
    SequenceId,
};

pub type MuseGlimmerSequenceCache = SequenceCache<Sm12xPageBackend, MuseGlimmerSequenceSnapshot>;

pub(crate) struct MuseGlimmerAppend<'a> {
    pub(crate) reservation: &'a sequence_cache::AppendReservation,
    pub(crate) page_table: &'a nvfp4::DeviceBuffer<u32>,
}

pub struct MuseGlimmerSequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: MuseGlimmerDecodeState,
}

impl MuseGlimmerSequence {
    pub fn admit(
        model: &MuseGlimmerModel,
        cache: &mut MuseGlimmerSequenceCache,
        max_tokens: usize,
    ) -> Result<Self> {
        let state = model.new_sequence_state(max_tokens)?;
        let mut page_table = Sm12xPageTable::new(max_tokens)?;
        let stream = model.stream();
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
            .map_err(muse_glimmer_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "Muse Glimmer sequence admission",
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
        state: MuseGlimmerDecodeState,
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

    pub fn finish(
        self,
        model: &MuseGlimmerModel,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Sm12xCacheContext {
                    stream: model.stream(),
                    page_table: &mut page_table,
                },
            )
            .map_err(muse_glimmer_cache_error)
    }
}

pub fn new_muse_glimmer_sequence_cache(
    model: &MuseGlimmerModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<MuseGlimmerSequenceCache> {
    new_muse_glimmer_sequence_cache_with_budget(model, sequence_capacity, max_context_tokens, None)
}

pub(crate) fn new_muse_glimmer_sequence_cache_with_budget(
    model: &MuseGlimmerModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
    retained_budget_bytes: Option<usize>,
) -> Result<MuseGlimmerSequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "Muse Glimmer sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let config = model.config();
    let probe = Sm12xPageBackend::new(
        std::iter::repeat_n(true, config.num_hidden_layers),
        1,
        config.num_key_value_heads,
        config.head_dim,
    )?;
    let page_bytes = probe.page_bytes();
    let private_bytes = model.new_sequence_state(max_context_tokens)?.device_bytes();
    let table_bytes = Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let fixed_bytes = private_bytes
        .checked_add(table_bytes)
        .and_then(|bytes| bytes.checked_mul(sequence_capacity))
        .ok_or_else(|| Error::Shape {
            label: "Muse Glimmer sequence cache private bytes",
            expected: "private byte count without overflow".to_string(),
            actual: format!(
                "private={private_bytes} table={table_bytes} sequences={sequence_capacity}"
            ),
        })?;
    let eager_pages = sequence_capacity
        .checked_mul(max_context_tokens.div_ceil(SM12X_KV_PAGE_TOKENS))
        .ok_or_else(|| Error::Shape {
            label: "Muse Glimmer sequence cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        })?;
    let eager_bytes = eager_pages
        .checked_mul(page_bytes)
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or_else(|| Error::Shape {
            label: "Muse Glimmer sequence cache bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("page_bytes={page_bytes} pages={eager_pages}"),
        })?;
    let retained_bytes = retained_budget_bytes.unwrap_or(0);
    let managed_bytes = eager_bytes
        .checked_add(retained_bytes)
        .ok_or_else(|| Error::Shape {
            label: "Muse Glimmer sequence cache budget",
            expected: "active and retained budgets without overflow".to_string(),
            actual: format!("active={eager_bytes} retained={retained_bytes}"),
        })?;
    let snapshot_bytes = if retained_budget_bytes.is_some() && model.has_dflash() {
        retained_bytes / 4
    } else {
        0
    };
    let retained_pages = retained_bytes.saturating_sub(snapshot_bytes) / page_bytes;
    let page_slots = eager_pages
        .checked_add(retained_pages)
        .ok_or_else(|| Error::Shape {
            label: "Muse Glimmer sequence cache pages",
            expected: "active and retained page counts without overflow".to_string(),
            actual: format!("active={eager_pages} retained={retained_pages}"),
        })?;
    let backend = Sm12xPageBackend::new(
        std::iter::repeat_n(true, config.num_hidden_layers),
        page_slots,
        config.num_key_value_heads,
        config.head_dim,
    )?;
    MuseGlimmerSequenceCache::new(
        CacheConfig {
            page_tokens: SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: snapshot_bytes,
            max_prefix_entries: retained_budget_bytes.is_none().then_some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(muse_glimmer_cache_error)
}

pub(crate) fn muse_glimmer_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Muse Glimmer sequence cache",
        detail: error.to_string(),
    }
}
