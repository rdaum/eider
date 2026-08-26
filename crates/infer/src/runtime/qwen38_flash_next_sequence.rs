//! Qwen3.8 Flash Next sequence state backed by shared QSA pages.

use super::sm12x_sequence_cache::{
    Sm12xAppendTransaction, Sm12xCacheContext, Sm12xPage, Sm12xPageBackend, Sm12xPageTable,
};
use crate::nvfp4::{CudaStream, Error, Qwen38QsaIndexPool, Result, Sm12xKvPagePool};
use crate::qwen3::infer::QwenLayerKind;
use crate::qwen38_flash_next::{
    Qwen38FlashNextDecodeState, Qwen38FlashNextModel, Qwen38FlashNextSequenceSnapshot,
    Qwen38NextToken,
};
use crate::runtime::cache_config::SequenceCacheConfig;
use seqcache::{
    AdmissionOutcome, AdmissionRequest, BackendAppendCommit, BackendAppendPage, CacheConfig,
    CacheError, PageAllocation, PageBackend, RetainedSnapshot, RetireError, RetireOutcome,
    SequenceCache, SequenceId,
};

impl RetainedSnapshot for Qwen38FlashNextSequenceSnapshot {
    fn retained_bytes(&self) -> usize {
        self.device_bytes()
    }
}

/// Shared QSA page manager for active Flash Next sequences.
pub type Qwen38FlashNextSequenceCache =
    SequenceCache<Qwen38FlashNextPageBackend, Qwen38FlashNextSequenceSnapshot>;

/// One admitted Flash Next sequence and its stable device page table.
pub struct Qwen38FlashNextSequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: Qwen38FlashNextDecodeState,
}

impl Qwen38FlashNextSequence {
    /// Admits a fresh sequence against the cache's bounded page budget.
    pub fn admit(
        model: &Qwen38FlashNextModel,
        cache: &mut Qwen38FlashNextSequenceCache,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::admit_with_prefix(model, cache, max_tokens, &[])
    }

    /// Admits a sequence, restoring the longest retained prompt prefix on a hit.
    pub fn admit_with_prefix(
        model: &Qwen38FlashNextModel,
        cache: &mut Qwen38FlashNextSequenceCache,
        max_tokens: usize,
        prompt_tokens: &[u32],
    ) -> Result<Self> {
        let prefix = cache.lookup_prefix(prompt_tokens);
        let mut page_table = Sm12xPageTable::new(max_tokens)?;
        let logical_capacity = page_table
            .page_capacity()
            .checked_mul(crate::nvfp4::SM12X_KV_PAGE_TOKENS)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 Flash Next sequence capacity",
                expected: "page-aligned capacity without overflow".to_string(),
                actual: max_tokens.to_string(),
            })?;
        let mut state = model.new_decode_state(logical_capacity)?;
        let cache_stream = CudaStream::new_blocking()?;
        let private_state_bytes = state.device_bytes();
        let outcome = cache
            .admit(
                prefix,
                AdmissionRequest {
                    max_position: max_tokens,
                    private_state_bytes,
                    page_table_bytes: page_table.managed_bytes(),
                    allow_emergency: false,
                },
                &mut Sm12xCacheContext {
                    stream: &cache_stream,
                    page_table: &mut page_table,
                },
                |snapshot, position| {
                    if let Some(snapshot) = snapshot {
                        model.restore_sequence_snapshot(snapshot, &mut state)?;
                    } else if position != 0 {
                        return Err(Error::Format {
                            label: "Qwen3.8 Flash Next sequence-cache restore",
                            detail: "nonzero prefix has no recurrent snapshot".to_string(),
                        });
                    }
                    Ok(())
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next sequence admission",
                detail: "configured cache has insufficient capacity".to_string(),
            });
        };
        cache_stream.synchronize()?;
        Ok(Self {
            cache_id,
            page_table,
            state,
        })
    }

    /// Returns the number of committed tokens.
    pub fn position(&self) -> usize {
        self.state.position()
    }

    /// Returns private state plus the stable device page table.
    pub fn device_bytes(&self) -> usize {
        self.state.device_bytes() + self.page_table.managed_bytes()
    }

    /// Evaluates and transactionally commits one token.
    pub fn decode_token(
        &mut self,
        model: &mut Qwen38FlashNextModel,
        cache: &mut Qwen38FlashNextSequenceCache,
        token: u32,
    ) -> Result<Qwen38NextToken> {
        model.decode_token(
            &mut self.state,
            cache,
            self.cache_id,
            &mut self.page_table,
            token,
        )
    }

    /// Finishes the active sequence and releases its page ownership.
    pub fn finish(self, cache: &mut Qwen38FlashNextSequenceCache) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Sm12xCacheContext {
                    stream: self.state.stream(),
                    page_table: &mut page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)
    }
}

/// Allocates the fixed QSA page budget for active sequences.
pub fn new_qwen38_flash_next_sequence_cache(
    model: &Qwen38FlashNextModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<Qwen38FlashNextSequenceCache> {
    new_qwen38_flash_next_sequence_cache_with_config(
        model,
        sequence_capacity,
        max_context_tokens,
        SequenceCacheConfig {
            max_retained_bytes: 0,
        },
    )
}

/// Allocates active QSA pages plus the configured retained-prefix budget.
pub fn new_qwen38_flash_next_sequence_cache_with_config(
    model: &Qwen38FlashNextModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
    cache_config: SequenceCacheConfig,
) -> Result<Qwen38FlashNextSequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let pages_per_sequence = max_context_tokens.div_ceil(crate::nvfp4::SM12X_KV_PAGE_TOKENS);
    let max_sequence_tokens = pages_per_sequence
        .checked_mul(crate::nvfp4::SM12X_KV_PAGE_TOKENS)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next sequence capacity",
            expected: "page-aligned context without overflow".to_string(),
            actual: max_context_tokens.to_string(),
        })?;
    let private_state_bytes = model.new_decode_state(max_sequence_tokens)?.device_bytes();
    let active_page_slots = sequence_capacity
        .checked_mul(pages_per_sequence)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next sequence-cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!(
                "sequences={sequence_capacity} pages_per_sequence={pages_per_sequence}"
            ),
        })?;
    let qsa_layers = model
        .manifest()
        .layer_kinds
        .iter()
        .map(|kind| *kind == QwenLayerKind::FullAttention)
        .collect::<Vec<_>>();
    let probe_backend = Qwen38FlashNextPageBackend::new(
        qsa_layers.iter().copied(),
        1,
        model.manifest().kv_heads,
        model.manifest().head_dim,
        model.config().indexer_head_dim,
    )?;
    let page_bytes = probe_backend.page_bytes();
    let table_bytes = Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let table_budget = table_bytes
        .checked_mul(sequence_capacity)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next sequence-cache page-table bytes",
            expected: "page-table byte count without overflow".to_string(),
            actual: format!("table_bytes={table_bytes} sequences={sequence_capacity}"),
        })?;
    let private_budget = private_state_bytes
        .checked_mul(sequence_capacity)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next private-state bytes",
            expected: "private-state byte count without overflow".to_string(),
            actual: format!("state_bytes={private_state_bytes} sequences={sequence_capacity}"),
        })?;
    let active_page_bytes =
        page_bytes
            .checked_mul(active_page_slots)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 Flash Next active page bytes",
                expected: "page byte count without overflow".to_string(),
                actual: format!("page_bytes={page_bytes} page_slots={active_page_slots}"),
            })?;
    let active_managed_bytes = active_page_bytes
        .checked_add(table_budget)
        .and_then(|bytes| bytes.checked_add(private_budget))
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next sequence-cache bytes",
            expected: "page, table, and private bytes without overflow".to_string(),
            actual: format!(
                "page_bytes={page_bytes} page_slots={active_page_slots} table_bytes={table_bytes} private_bytes={private_state_bytes}"
            ),
        })?;
    let retained_bytes = cache_config.max_retained_bytes;
    let snapshot_capacity = retained_bytes / 4;
    let page_slots = active_page_slots
        .checked_add(retained_bytes.saturating_sub(snapshot_capacity) / page_bytes)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next retained page slots",
            expected: "page count without overflow".to_string(),
            actual: format!("active={active_page_slots} retained={retained_bytes}"),
        })?;
    let managed_bytes = active_managed_bytes
        .checked_add(retained_bytes)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next retained cache bytes",
            expected: "active and retained byte count without overflow".to_string(),
            actual: format!("active={active_managed_bytes} retained={retained_bytes}"),
        })?;
    let backend = Qwen38FlashNextPageBackend::new(
        qsa_layers,
        page_slots,
        model.manifest().kv_heads,
        model.manifest().head_dim,
        model.config().indexer_head_dim,
    )?;
    Qwen38FlashNextSequenceCache::new(
        CacheConfig {
            page_tokens: crate::nvfp4::SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: snapshot_capacity,
            max_prefix_entries: (retained_bytes == 0).then_some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(qwen38_flash_next_cache_error)
}

pub(crate) fn qwen38_flash_next_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Qwen3.8 Flash Next sequence cache",
        detail: error.to_string(),
    }
}

/// Physical page backend bundling compact K/V and raw QSA index keys.
pub struct Qwen38FlashNextPageBackend {
    kv: Sm12xPageBackend,
    index_keys: Vec<Option<Qwen38QsaIndexPool>>,
    page_bytes: usize,
}

impl Qwen38FlashNextPageBackend {
    /// Allocates one stable-slot pool for each QSA layer.
    pub fn new(
        qsa_layers: impl IntoIterator<Item = bool>,
        page_slots: usize,
        kv_heads: usize,
        kv_head_dim: usize,
        index_head_dim: usize,
    ) -> Result<Self> {
        let qsa_layers = qsa_layers.into_iter().collect::<Vec<_>>();
        let kv = Sm12xPageBackend::new(
            qsa_layers.iter().copied(),
            page_slots,
            kv_heads,
            kv_head_dim,
        )?;
        let mut page_bytes = kv.page_bytes();
        let mut index_keys = Vec::with_capacity(qsa_layers.len());
        for qsa in qsa_layers {
            let pool = if qsa {
                let pool = Qwen38QsaIndexPool::new(page_slots, index_head_dim)?;
                page_bytes =
                    page_bytes
                        .checked_add(pool.page_bytes())
                        .ok_or_else(|| Error::Shape {
                            label: "Qwen3.8 Flash Next page bytes",
                            expected: "KV and index-key byte sum without overflow".to_string(),
                            actual: format!("current={page_bytes} next={}", pool.page_bytes()),
                        })?;
                Some(pool)
            } else {
                None
            };
            index_keys.push(pool);
        }
        Ok(Self {
            kv,
            index_keys,
            page_bytes,
        })
    }

    /// Returns one QSA layer's compact KV pool.
    pub fn kv_pool_mut(&mut self, layer: usize) -> Result<&mut Sm12xKvPagePool> {
        self.kv.pool_mut(layer)
    }

    /// Returns one QSA layer's raw index-key pool.
    pub fn index_key_pool_mut(&mut self, layer: usize) -> Result<&mut Qwen38QsaIndexPool> {
        self.index_keys
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 QSA index-key layer",
                expected: "a valid QSA layer".to_string(),
                actual: layer.to_string(),
            })
    }

    /// Borrows both page pools belonging to one QSA layer.
    pub fn qsa_pools_mut(
        &mut self,
        layer: usize,
    ) -> Result<(&mut Sm12xKvPagePool, &mut Qwen38QsaIndexPool)> {
        let kv = &mut self.kv;
        let index_keys = &mut self.index_keys;
        let kv_pool = kv.pool_mut(layer)?;
        let index_pool = index_keys
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 QSA index-key layer",
                expected: "a valid QSA layer".to_string(),
                actual: layer.to_string(),
            })?;
        Ok((kv_pool, index_pool))
    }
}

impl PageBackend for Qwen38FlashNextPageBackend {
    type Page = Sm12xPage;
    type Context<'a> = Sm12xCacheContext<'a>;
    type AppendTransaction = Sm12xAppendTransaction;
    type Error = Error;

    fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn page_capacity(&self) -> Option<usize> {
        self.kv.page_capacity()
    }

    fn allocate_page(
        &mut self,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>> {
        self.kv.allocate_page(context)
    }

    fn rollback_page(&mut self, page: Self::Page, context: &mut Self::Context<'_>) {
        self.kv.rollback_page(page, context);
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        start_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction> {
        self.kv.prepare_append(pages, start_position, context)
    }

    fn abort_append(
        &mut self,
        transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        self.kv.abort_append(
            transaction,
            restored_pages,
            released_pages,
            restored_position,
            context,
        )
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>> {
        let allocation = self.kv.copy_partial_page(source, valid_tokens, context)?;
        for pool in self.index_keys.iter_mut().filter_map(Option::as_mut) {
            if let Err(error) =
                pool.copy_page_on_stream(source.slot(), allocation.page.slot(), context.stream)
            {
                self.kv.rollback_page(allocation.page, context);
                return Err(error);
            }
        }
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        self.kv.commit_append(transaction, commit, context)
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        self.kv.update_page_table(pages, position, context)
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        self.kv.retire_pages(pages, context)
    }

    fn retirement_is_immediate(&self) -> bool {
        self.kv.retirement_is_immediate()
    }

    fn poll_reclaimed(&mut self, context: &mut Self::Context<'_>) -> Result<usize> {
        self.kv.poll_reclaimed(context)
    }
}
