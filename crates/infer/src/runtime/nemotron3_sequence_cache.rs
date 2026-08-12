//! Shared paged backbone-attention storage for Nemotron 3.

use super::sm12x_sequence_cache::Sm12xPageTable;
use crate::nemotron3::{
    Nemotron3DecodeState, Nemotron3KvCacheStorage, Nemotron3LayerKind, Nemotron3Model,
    Nemotron3SequenceSnapshot,
};
use nvfp4::{CudaStream, DeviceBuffer, Error, Result, SM12X_KV_PAGE_TOKENS, Sm12xKvPagePool};
use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, CacheConfig, CacheError, PageAllocation, PageBackend,
    RetireError, RetireOutcome, SequenceCache, SequenceId,
};

pub type Nemotron3SequenceCache = SequenceCache<Nemotron3PageBackend, Nemotron3SequenceSnapshot>;

pub struct Nemotron3Sequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: Nemotron3DecodeState,
}

impl Nemotron3Sequence {
    pub fn admit(
        model: &Nemotron3Model,
        cache: &mut Nemotron3SequenceCache,
        max_tokens: usize,
    ) -> Result<Self> {
        let state = model.sequence_state(max_tokens)?;
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
                &mut Nemotron3CacheContext {
                    stream,
                    page_table: &mut page_table,
                },
                |snapshot, position| {
                    debug_assert!(snapshot.is_none());
                    debug_assert_eq!(position, 0);
                    Ok(())
                },
            )
            .map_err(nemotron3_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "Nemotron 3 sequence admission",
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
        state: Nemotron3DecodeState,
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
        self.state.max_tokens
    }

    pub fn device_bytes(&self) -> usize {
        self.state.device_bytes() + self.page_table.managed_bytes()
    }

    pub fn finish(self, model: &Nemotron3Model, cache: &mut Nemotron3SequenceCache) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Nemotron3CacheContext {
                    stream: model.stream(),
                    page_table: &mut page_table,
                },
            )
            .map_err(nemotron3_cache_error)
    }
}

pub fn new_nemotron3_sequence_cache(
    model: &Nemotron3Model,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<Nemotron3SequenceCache> {
    new_nemotron3_sequence_cache_with_budget(model, sequence_capacity, max_context_tokens, None)
}

pub(crate) fn new_nemotron3_sequence_cache_with_budget(
    model: &Nemotron3Model,
    sequence_capacity: usize,
    max_context_tokens: usize,
    prefix_budget_bytes: Option<usize>,
) -> Result<Nemotron3SequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "Nemotron 3 sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let manifest = model.manifest();
    let storage = model.kv_cache_storage();
    let geometry = || {
        manifest
            .layers
            .iter()
            .map(|kind| *kind == Nemotron3LayerKind::Attention)
    };
    let probe = Nemotron3PageBackend::new(
        geometry(),
        1,
        manifest.kv_heads,
        manifest.attention_head_dim,
        storage,
    )?;
    let page_bytes = probe.page_bytes();
    let private_bytes = model.sequence_state(max_context_tokens)?.device_bytes();
    let table_bytes = Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let fixed_bytes = private_bytes
        .checked_add(table_bytes)
        .and_then(|bytes| bytes.checked_mul(sequence_capacity))
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 sequence-cache private bytes",
            expected: "private byte count without overflow".to_string(),
            actual: format!(
                "private={private_bytes} table={table_bytes} sequences={sequence_capacity}"
            ),
        })?;
    let eager_pages = sequence_capacity
        .checked_mul(max_context_tokens.div_ceil(SM12X_KV_PAGE_TOKENS))
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 sequence-cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        })?;
    let eager_bytes = eager_pages
        .checked_mul(page_bytes)
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 sequence-cache bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("page_bytes={page_bytes} pages={eager_pages}"),
        })?;
    let prefix_bytes = prefix_budget_bytes.unwrap_or(0);
    let managed_bytes = eager_bytes
        .checked_add(prefix_bytes)
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 sequence-cache managed budget",
            expected: "active and prefix budgets without overflow".to_string(),
            actual: format!("active={eager_bytes} prefix={prefix_bytes}"),
        })?;
    let snapshot_bytes = prefix_bytes / 4;
    let extra_pages = prefix_bytes.saturating_sub(snapshot_bytes) / page_bytes;
    let page_slots = eager_pages.saturating_add(extra_pages);
    if page_slots == 0 {
        return Err(Error::Shape {
            label: "Nemotron 3 sequence-cache capacity",
            expected: format!(
                "budget greater than fixed capacity {fixed_bytes}, snapshot reserve {snapshot_bytes}, and one {page_bytes}-byte page"
            ),
            actual: managed_bytes.to_string(),
        });
    }
    let backend = Nemotron3PageBackend::new(
        geometry(),
        page_slots,
        manifest.kv_heads,
        manifest.attention_head_dim,
        storage,
    )?;
    Nemotron3SequenceCache::new(
        CacheConfig {
            page_tokens: SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: snapshot_bytes,
            max_prefix_entries: prefix_budget_bytes.is_none().then_some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(nemotron3_cache_error)
}

pub(crate) fn nemotron3_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Nemotron 3 sequence cache",
        detail: error.to_string(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nemotron3Page {
    slot: u32,
}

impl Nemotron3Page {
    pub(crate) fn slot(self) -> usize {
        self.slot as usize
    }
}

pub struct Nemotron3CacheContext<'a> {
    pub stream: &'a CudaStream,
    pub page_table: &'a mut Sm12xPageTable,
}

pub(crate) struct F32KvPagePool {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    page_slots: usize,
    width: usize,
}

impl F32KvPagePool {
    fn new(page_slots: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        let width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
            label: "F32 KV page width",
            expected: "head geometry without overflow".to_string(),
            actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
        })?;
        let values = page_slots
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .and_then(|rows| rows.checked_mul(width))
            .ok_or_else(|| Error::Shape {
                label: "F32 KV page pool",
                expected: "pool shape without overflow".to_string(),
                actual: format!("slots={page_slots} width={width}"),
            })?;
        Ok(Self {
            key: DeviceBuffer::zeroed(values)?,
            value: DeviceBuffer::zeroed(values)?,
            page_slots,
            width,
        })
    }

    pub(crate) fn buffers(&self) -> (&DeviceBuffer<f32>, &DeviceBuffer<f32>) {
        (&self.key, &self.value)
    }

    pub(crate) fn buffers_mut(&mut self) -> (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>) {
        (&mut self.key, &mut self.value)
    }

    fn page_bytes(&self) -> usize {
        2 * SM12X_KV_PAGE_TOKENS * self.width * size_of::<f32>()
    }

    fn copy_page(&mut self, source: usize, destination: usize, stream: &CudaStream) -> Result<()> {
        if source >= self.page_slots || destination >= self.page_slots {
            return Err(Error::Shape {
                label: "F32 KV page copy",
                expected: format!("slots below {}", self.page_slots),
                actual: format!("source={source} destination={destination}"),
            });
        }
        let values = SM12X_KV_PAGE_TOKENS * self.width;
        let source_offset = source * values;
        let destination_offset = destination * values;
        self.key
            .copy_within_on_stream(source_offset, destination_offset, values, stream)?;
        self.value
            .copy_within_on_stream(source_offset, destination_offset, values, stream)
    }
}

enum Nemotron3LayerPool {
    F32(F32KvPagePool),
    Nvfp4(Sm12xKvPagePool),
}

pub struct Nemotron3PageBackend {
    pools: Vec<Option<Nemotron3LayerPool>>,
    free_slots: Vec<u32>,
    used_slots: Vec<bool>,
    ever_used_slots: Vec<bool>,
    page_bytes: usize,
    storage: Nemotron3KvCacheStorage,
}

impl Nemotron3PageBackend {
    pub fn new(
        attention_layers: impl IntoIterator<Item = bool>,
        page_slots: usize,
        kv_heads: usize,
        head_dim: usize,
        storage: Nemotron3KvCacheStorage,
    ) -> Result<Self> {
        if page_slots == 0 || page_slots > u32::MAX as usize {
            return Err(Error::Shape {
                label: "Nemotron 3 KV page slots",
                expected: format!("1..={}", u32::MAX),
                actual: page_slots.to_string(),
            });
        }
        let mut pools = Vec::new();
        let mut page_bytes = 0usize;
        for attention in attention_layers {
            let pool = if attention {
                let pool = match storage {
                    Nemotron3KvCacheStorage::F32 => {
                        Nemotron3LayerPool::F32(F32KvPagePool::new(page_slots, kv_heads, head_dim)?)
                    }
                    Nemotron3KvCacheStorage::Nvfp4 => Nemotron3LayerPool::Nvfp4(
                        Sm12xKvPagePool::new(page_slots, kv_heads, head_dim)?,
                    ),
                };
                page_bytes = page_bytes
                    .checked_add(match &pool {
                        Nemotron3LayerPool::F32(pool) => pool.page_bytes(),
                        Nemotron3LayerPool::Nvfp4(pool) => pool.page_bytes(),
                    })
                    .ok_or_else(|| Error::Shape {
                        label: "Nemotron 3 page bundle bytes",
                        expected: "layer page-byte sum without overflow".to_string(),
                        actual: format!("layers={}", pools.len() + 1),
                    })?;
                Some(pool)
            } else {
                None
            };
            pools.push(pool);
        }
        Ok(Self {
            pools,
            free_slots: (0..page_slots as u32).rev().collect(),
            used_slots: vec![false; page_slots],
            ever_used_slots: vec![false; page_slots],
            page_bytes,
            storage,
        })
    }

    pub fn storage(&self) -> Nemotron3KvCacheStorage {
        self.storage
    }

    pub(crate) fn f32_pool_mut(&mut self, layer: usize) -> Result<&mut F32KvPagePool> {
        match self.pools.get_mut(layer).and_then(Option::as_mut) {
            Some(Nemotron3LayerPool::F32(pool)) => Ok(pool),
            _ => Err(Error::Shape {
                label: "Nemotron 3 F32 page pool",
                expected: "an F32 attention-layer pool".to_string(),
                actual: layer.to_string(),
            }),
        }
    }

    pub(crate) fn nvfp4_pool_mut(&mut self, layer: usize) -> Result<&mut Sm12xKvPagePool> {
        match self.pools.get_mut(layer).and_then(Option::as_mut) {
            Some(Nemotron3LayerPool::Nvfp4(pool)) => Ok(pool),
            _ => Err(Error::Shape {
                label: "Nemotron 3 NVFP4 page pool",
                expected: "an NVFP4 attention-layer pool".to_string(),
                actual: layer.to_string(),
            }),
        }
    }

    fn validate_page(&self, page: Nemotron3Page) -> Result<()> {
        let slot = page.slot();
        if slot >= self.used_slots.len() || !self.used_slots[slot] {
            return Err(Error::Shape {
                label: "Nemotron 3 physical page",
                expected: "an allocated pool slot".to_string(),
                actual: slot.to_string(),
            });
        }
        Ok(())
    }
}

impl PageBackend for Nemotron3PageBackend {
    type Page = Nemotron3Page;
    type Context<'a> = Nemotron3CacheContext<'a>;
    type Error = Error;

    fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn page_capacity(&self) -> Option<usize> {
        Some(self.used_slots.len())
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>> {
        let slot = self.free_slots.pop().ok_or_else(|| Error::Shape {
            label: "Nemotron 3 physical page allocation",
            expected: "a free preallocated slot".to_string(),
            actual: "pool exhausted".to_string(),
        })?;
        let index = slot as usize;
        let recycled = self.ever_used_slots[index];
        self.used_slots[index] = true;
        self.ever_used_slots[index] = true;
        Ok(PageAllocation {
            page: Nemotron3Page { slot },
            recycled,
        })
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.used_slots[page.slot()] = false;
        self.free_slots.push(page.slot);
    }

    fn abort_append(
        &mut self,
        pages: &[&Self::Page],
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in pages {
            self.validate_page(**page)?;
        }
        context.stream.synchronize()?;
        for page in pages {
            self.used_slots[page.slot()] = false;
            self.free_slots.push(page.slot);
        }
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>> {
        self.validate_page(*source)?;
        if valid_tokens == 0 || valid_tokens >= SM12X_KV_PAGE_TOKENS {
            return Err(Error::Shape {
                label: "Nemotron 3 partial page copy",
                expected: format!("valid tokens in 1..{SM12X_KV_PAGE_TOKENS}"),
                actual: valid_tokens.to_string(),
            });
        }
        let allocation = self.allocate_page(context)?;
        for pool in self.pools.iter_mut().filter_map(Option::as_mut) {
            let result = match pool {
                Nemotron3LayerPool::F32(pool) => {
                    pool.copy_page(source.slot(), allocation.page.slot(), context.stream)
                }
                Nemotron3LayerPool::Nvfp4(pool) => {
                    pool.copy_page_on_stream(source.slot(), allocation.page.slot(), context.stream)
                }
            };
            if let Err(error) = result {
                context.stream.synchronize()?;
                self.rollback_page(allocation.page, context);
                return Err(error);
            }
        }
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        committed_pages: &[&Self::Page],
        sealed_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        new_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in committed_pages
            .iter()
            .chain(sealed_pages)
            .chain(released_pages)
        {
            self.validate_page(**page)?;
        }
        context.page_table.update_slots(
            committed_pages.iter().map(|page| page.slot),
            committed_pages.len(),
            new_position,
            context.stream,
        )?;
        if !released_pages.is_empty() {
            context.stream.synchronize()?;
            for page in released_pages {
                self.used_slots[page.slot()] = false;
                self.free_slots.push(page.slot);
            }
        }
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in pages {
            self.validate_page(**page)?;
        }
        context.page_table.update_slots(
            pages.iter().map(|page| page.slot),
            pages.len(),
            position,
            context.stream,
        )
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> core::result::Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        if let Some(error) = pages
            .iter()
            .find_map(|page| self.validate_page(*page).err())
        {
            return Err(RetireError { error, pages });
        }
        for page in &pages {
            self.used_slots[page.slot()] = false;
            self.free_slots.push(page.slot);
        }
        Ok(RetireOutcome::default())
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admitted(outcome: AdmissionOutcome) -> SequenceId {
        match outcome {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("test admission unexpectedly blocked"),
        }
    }

    #[test]
    fn f32_backend_copies_partial_tail_and_reserves_across_pages() -> Result<()> {
        let stream = CudaStream::new_non_blocking()?;
        let backend = Nemotron3PageBackend::new([true], 4, 1, 2, Nemotron3KvCacheStorage::F32)?;
        let page_bytes = backend.page_bytes();
        let mut source_table = Sm12xPageTable::new(256)?;
        let mut branch_table = Sm12xPageTable::new(256)?;
        let table_bytes = source_table.managed_bytes();
        let mut cache = SequenceCache::<_, ()>::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: 4 * page_bytes + 2 * table_bytes,
                max_snapshot_bytes: 0,
                max_prefix_entries: Some(0),
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(nemotron3_cache_error)?;
        let request = AdmissionRequest {
            max_position: 256,
            private_state_bytes: 0,
            page_table_bytes: table_bytes,
            allow_emergency: false,
        };
        let source = admitted(
            cache
                .admit(
                    None,
                    request,
                    &mut Nemotron3CacheContext {
                        stream: &stream,
                        page_table: &mut source_table,
                    },
                    |snapshot, position| {
                        assert!(snapshot.is_none());
                        assert_eq!(position, 0);
                        Ok(())
                    },
                )
                .map_err(nemotron3_cache_error)?,
        );

        let source_append = cache
            .reserve_append(
                source,
                64,
                &mut Nemotron3CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .map_err(nemotron3_cache_error)?;
        cache
            .with_append_pages(&source_append, |backend, pages| {
                let slot = pages
                    .iter()
                    .next()
                    .expect("one reserved page")
                    .page()
                    .slot();
                let values_per_page = SM12X_KV_PAGE_TOKENS * 2;
                let values = (0..values_per_page)
                    .map(|index| index as f32 + 0.5)
                    .collect::<Vec<_>>();
                let (key, value) = backend.f32_pool_mut(0)?.buffers_mut();
                key.copy_range_from_host(slot * values_per_page, &values)?;
                value.copy_range_from_host(slot * values_per_page, &values)?;
                Ok(())
            })
            .map_err(nemotron3_cache_error)?;
        cache
            .commit_append(
                source_append,
                64,
                &mut Nemotron3CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .map_err(nemotron3_cache_error)?;

        let branch = admitted(
            cache
                .branch(
                    source,
                    request,
                    &mut Nemotron3CacheContext {
                        stream: &stream,
                        page_table: &mut branch_table,
                    },
                )
                .map_err(nemotron3_cache_error)?,
        );
        let source_page = cache.page_table(source).unwrap().pages()[0];
        let branch_page = cache.page_table(branch).unwrap().pages()[0];
        let source_slot = cache.page(source_page).unwrap().slot();
        let branch_slot = cache.page(branch_page).unwrap().slot();
        assert_ne!(source_slot, branch_slot);
        let values_per_page = SM12X_KV_PAGE_TOKENS * 2;
        let all_keys = cache.with_backend(|backend| {
            backend
                .f32_pool_mut(0)?
                .buffers()
                .0
                .copy_to_host(&stream)
                .map(|values| values.into_vec())
        })?;
        assert_eq!(
            &all_keys[source_slot * values_per_page..(source_slot + 1) * values_per_page],
            &all_keys[branch_slot * values_per_page..(branch_slot + 1) * values_per_page]
        );

        let append = cache
            .reserve_append(
                branch,
                192,
                &mut Nemotron3CacheContext {
                    stream: &stream,
                    page_table: &mut branch_table,
                },
            )
            .map_err(nemotron3_cache_error)?;
        assert_eq!(
            append
                .segments()
                .iter()
                .map(|segment| (
                    segment.page_offset(),
                    segment.input_offset(),
                    segment.rows()
                ))
                .collect::<Vec<_>>(),
            [(64, 0, 64), (0, 64, 128)]
        );
        cache
            .commit_append(
                append,
                192,
                &mut Nemotron3CacheContext {
                    stream: &stream,
                    page_table: &mut branch_table,
                },
            )
            .map_err(nemotron3_cache_error)?;
        let branch_pages = cache.page_table(branch).unwrap();
        assert_eq!(branch_pages.position(), 256);
        assert_eq!(branch_pages.pages().len(), 2);
        for page in branch_pages.pages() {
            let page = cache.page(*page).unwrap();
            assert!(cache.backend().used_slots[page.slot()]);
        }
        cache.validate().map_err(nemotron3_cache_error)?;
        Ok(())
    }
}
