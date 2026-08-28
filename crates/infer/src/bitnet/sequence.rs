//! Shared paged sequence storage for BitNet.

use super::{BitNetDecodeState, BitNetModel};
use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use eider_cuda::{Error, Result};
use seqcache::{AdmissionOutcome, AdmissionRequest, CacheError, SequenceCache, SequenceId};

pub type BitNetSequenceCache = SequenceCache<Sm12xPageBackend, ()>;

pub struct BitNetSequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: BitNetDecodeState,
}

impl BitNetSequence {
    pub fn admit(
        model: &BitNetModel,
        cache: &mut BitNetSequenceCache,
        max_tokens: usize,
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
                    stream: state.stream(),
                    page_table: &mut page_table,
                },
                |snapshot, position| {
                    debug_assert!(snapshot.is_none());
                    debug_assert_eq!(position, 0);
                    Ok(())
                },
            )
            .map_err(bitnet_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "BitNet sequence admission",
                detail: "configured cache has insufficient capacity".to_string(),
            });
        };
        state.stream().synchronize()?;
        Ok(Self {
            cache_id,
            page_table,
            state,
        })
    }

    pub fn position(&self) -> usize {
        self.state.len()
    }

    pub fn device_bytes(&self) -> usize {
        self.state.device_bytes() + self.page_table.managed_bytes()
    }

    pub fn finish(self, cache: &mut BitNetSequenceCache) -> Result<()> {
        let Self {
            cache_id,
            mut page_table,
            state,
        } = self;
        cache
            .finish(
                cache_id,
                &mut Sm12xCacheContext {
                    stream: state.stream(),
                    page_table: &mut page_table,
                },
            )
            .map_err(bitnet_cache_error)
    }
}

pub fn new_bitnet_sequence_cache(
    model: &BitNetModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<BitNetSequenceCache> {
    let config = model.config();
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "BitNet sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let page_slots = sequence_capacity
        .checked_mul(max_context_tokens.div_ceil(eider_cuda::SM12X_KV_PAGE_TOKENS))
        .ok_or_else(|| Error::Shape {
            label: "BitNet sequence cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        })?;
    let backend = Sm12xPageBackend::new(
        std::iter::repeat_n(true, config.layers),
        page_slots,
        config.kv_heads,
        config.head_dim,
    )?;
    let page_bytes = seqcache::PageBackend::page_bytes(&backend);
    let fixed = model
        .new_sequence_state(max_context_tokens)?
        .device_bytes()
        .checked_add(Sm12xPageTable::new(max_context_tokens)?.managed_bytes())
        .and_then(|bytes| bytes.checked_mul(sequence_capacity))
        .ok_or_else(|| Error::Shape {
            label: "BitNet sequence cache bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        })?;
    let managed_bytes = page_bytes
        .checked_mul(page_slots)
        .and_then(|bytes| bytes.checked_add(fixed))
        .ok_or_else(|| Error::Shape {
            label: "BitNet sequence cache bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("page_bytes={page_bytes} pages={page_slots}"),
        })?;
    BitNetSequenceCache::new(
        seqcache::CacheConfig {
            page_tokens: eider_cuda::SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: 0,
            max_prefix_entries: Some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(bitnet_cache_error)
}

pub(crate) fn bitnet_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "BitNet sequence cache",
        detail: error.to_string(),
    }
}
