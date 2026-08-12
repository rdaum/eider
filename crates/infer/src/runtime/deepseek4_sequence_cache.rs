//! Shared paged sliding-attention storage for DeepSeek V4 sequences.

use super::sm12x_sequence_cache::Sm12xPageTable;
use crate::deepseek4::{Deepseek4SequenceCheckpoint, Deepseek4SequenceState, Deepseek4TextModel};
use nvfp4::{CudaStream, DeviceBuffer, Error, Result, SM12X_KV_PAGE_TOKENS};
use seqcache::{
    BackendAppendCommit, BackendAppendPage, CacheConfig, CacheError, PageAllocation, PageBackend,
    RetireError, RetireOutcome, SequenceCache, SequenceId,
};
use std::mem::size_of;

pub type Deepseek4SequenceCache = SequenceCache<Deepseek4PageBackend, Deepseek4SequenceCheckpoint>;

/// One admitted DeepSeek sequence and its private compressed-attention state.
pub struct Deepseek4Sequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: Deepseek4SequenceState,
}

impl Deepseek4Sequence {
    pub(crate) fn from_admission(
        cache_id: SequenceId,
        page_table: Sm12xPageTable,
        state: Deepseek4SequenceState,
    ) -> Self {
        Self {
            cache_id,
            page_table,
            state,
        }
    }

    pub fn position(&self) -> usize {
        self.state.position()
    }

    pub fn max_tokens(&self) -> usize {
        self.state.max_tokens()
    }

    pub fn device_bytes(&self) -> usize {
        self.state.device_bytes() + self.page_table.managed_bytes()
    }

    pub fn finish(self, stream: &CudaStream, cache: &mut Deepseek4SequenceCache) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Deepseek4CacheContext {
                    stream,
                    page_table: &mut page_table,
                },
            )
            .map_err(deepseek4_cache_error)
    }
}

pub fn new_deepseek4_sequence_cache(
    model: &Deepseek4TextModel,
    sequence_capacity: usize,
    max_context_tokens: usize,
    prefix_budget_bytes: Option<usize>,
) -> Result<Deepseek4SequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "DeepSeek V4 sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let config = &model.weights.config;
    let probe = Deepseek4PageBackend::new(config.num_hidden_layers, config.head_dim, 1)?;
    let page_bytes = probe.page_bytes();
    let state_bytes = model.new_sequence_state(max_context_tokens)?.device_bytes();
    let table_bytes = Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let fixed_bytes = state_bytes
        .checked_add(table_bytes)
        .and_then(|bytes| bytes.checked_mul(sequence_capacity))
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 sequence-cache private bytes",
            expected: "private byte count without overflow".to_string(),
            actual: format!(
                "state={state_bytes} table={table_bytes} sequences={sequence_capacity}"
            ),
        })?;
    let eager_pages = sequence_capacity
        .checked_mul(max_context_tokens.div_ceil(SM12X_KV_PAGE_TOKENS))
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 sequence-cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        })?;
    let eager_bytes = eager_pages
        .checked_mul(page_bytes)
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 sequence-cache bytes",
            expected: "managed byte count without overflow".to_string(),
            actual: format!("page_bytes={page_bytes} pages={eager_pages}"),
        })?;
    let prefix_bytes = prefix_budget_bytes.unwrap_or(0);
    let snapshot_bytes = prefix_bytes / 2;
    let extra_pages = prefix_bytes.saturating_sub(snapshot_bytes) / page_bytes;
    let page_slots = eager_pages
        .checked_add(extra_pages)
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 page slots",
            expected: "page count without overflow".to_string(),
            actual: format!("active={eager_pages} prefix={extra_pages}"),
        })?;
    let managed_bytes = eager_bytes
        .checked_add(prefix_bytes)
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 sequence-cache budget",
            expected: "active and prefix budgets without overflow".to_string(),
            actual: format!("active={eager_bytes} prefix={prefix_bytes}"),
        })?;
    let backend = Deepseek4PageBackend::new(config.num_hidden_layers, config.head_dim, page_slots)?;
    Deepseek4SequenceCache::new(
        CacheConfig {
            page_tokens: SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed_bytes,
            max_snapshot_bytes: snapshot_bytes,
            max_prefix_entries: prefix_budget_bytes.is_none().then_some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(deepseek4_cache_error)
}

pub(crate) fn deepseek4_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "DeepSeek V4 sequence cache",
        detail: error.to_string(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Deepseek4Page {
    slot: u32,
}

impl Deepseek4Page {
    pub(crate) fn slot(self) -> usize {
        self.slot as usize
    }
}

pub struct Deepseek4CacheContext<'a> {
    pub stream: &'a CudaStream,
    pub page_table: &'a mut Sm12xPageTable,
}

pub(crate) struct Deepseek4PagePool {
    values: DeviceBuffer<f32>,
    slots: usize,
    width: usize,
}

impl Deepseek4PagePool {
    fn new(slots: usize, width: usize) -> Result<Self> {
        let values = slots
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .and_then(|rows| rows.checked_mul(width))
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 page pool",
                expected: "pool shape without overflow".to_string(),
                actual: format!("slots={slots} width={width}"),
            })?;
        Ok(Self {
            values: DeviceBuffer::zeroed(values)?,
            slots,
            width,
        })
    }

    pub(crate) fn values(&self) -> &DeviceBuffer<f32> {
        &self.values
    }

    pub(crate) fn append_segment(
        &mut self,
        slot: usize,
        page_offset: usize,
        source: &DeviceBuffer<f32>,
        source_row: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if slot >= self.slots
            || page_offset.saturating_add(rows) > SM12X_KV_PAGE_TOKENS
            || source_row
                .checked_add(rows)
                .and_then(|end| end.checked_mul(self.width))
                .is_none_or(|end| end > source.len())
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 paged append",
                expected: format!(
                    "slot < {}, page rows <= {}, and source rows within {} values",
                    self.slots,
                    SM12X_KV_PAGE_TOKENS,
                    source.len()
                ),
                actual: format!(
                    "slot={slot} page_offset={page_offset} source_row={source_row} rows={rows}"
                ),
            });
        }
        self.values.copy_range_from_device_on_stream(
            (slot * SM12X_KV_PAGE_TOKENS + page_offset) * self.width,
            source,
            source_row * self.width,
            rows * self.width,
            stream,
        )
    }

    fn page_bytes(&self) -> usize {
        SM12X_KV_PAGE_TOKENS * self.width * size_of::<f32>()
    }

    fn copy_page(&mut self, source: usize, destination: usize, stream: &CudaStream) -> Result<()> {
        if source >= self.slots || destination >= self.slots {
            return Err(Error::Shape {
                label: "DeepSeek V4 page copy",
                expected: format!("slots below {}", self.slots),
                actual: format!("source={source} destination={destination}"),
            });
        }
        let values = SM12X_KV_PAGE_TOKENS * self.width;
        self.values
            .copy_within_on_stream(source * values, destination * values, values, stream)
    }
}

pub struct Deepseek4PageBackend {
    pools: Vec<Deepseek4PagePool>,
    free_slots: Vec<u32>,
    used_slots: Vec<bool>,
    ever_used_slots: Vec<bool>,
    page_bytes: usize,
}

impl Deepseek4PageBackend {
    pub(crate) fn new(layers: usize, width: usize, slots: usize) -> Result<Self> {
        if slots == 0 || slots > u32::MAX as usize {
            return Err(Error::Shape {
                label: "DeepSeek V4 page slots",
                expected: format!("1..={}", u32::MAX),
                actual: slots.to_string(),
            });
        }
        if layers == 0 || width == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 page geometry",
                expected: "positive layer count and width".to_string(),
                actual: format!("layers={layers} width={width}"),
            });
        }
        let mut pools = Vec::with_capacity(layers);
        let mut page_bytes = 0usize;
        for _ in 0..layers {
            let pool = Deepseek4PagePool::new(slots, width)?;
            page_bytes = page_bytes
                .checked_add(pool.page_bytes())
                .ok_or_else(|| Error::Shape {
                    label: "DeepSeek V4 page bundle bytes",
                    expected: "layer page-byte sum without overflow".to_string(),
                    actual: format!("layers={}", pools.len() + 1),
                })?;
            pools.push(pool);
        }
        Ok(Self {
            pools,
            free_slots: (0..slots as u32).rev().collect(),
            used_slots: vec![false; slots],
            ever_used_slots: vec![false; slots],
            page_bytes,
        })
    }

    pub(crate) fn pool_mut(&mut self, layer: usize) -> Result<&mut Deepseek4PagePool> {
        let layers = self.pools.len();
        self.pools.get_mut(layer).ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 page-pool layer",
            expected: format!("layer < {layers}"),
            actual: layer.to_string(),
        })
    }

    fn validate(&self, page: Deepseek4Page) -> Result<()> {
        if page.slot() >= self.used_slots.len() || !self.used_slots[page.slot()] {
            return Err(Error::Format {
                label: "DeepSeek V4 physical page",
                detail: format!("slot {} is not allocated", page.slot()),
            });
        }
        Ok(())
    }
}

impl PageBackend for Deepseek4PageBackend {
    type Page = Deepseek4Page;
    type Context<'a> = Deepseek4CacheContext<'a>;
    type AppendTransaction = ();
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
        let slot = self.free_slots.pop().ok_or_else(|| Error::Format {
            label: "DeepSeek V4 physical page allocation",
            detail: "preallocated page pool exhausted".to_string(),
        })?;
        let index = slot as usize;
        let recycled = self.ever_used_slots[index];
        self.used_slots[index] = true;
        self.ever_used_slots[index] = true;
        Ok(PageAllocation {
            page: Deepseek4Page { slot },
            recycled,
        })
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.used_slots[page.slot()] = false;
        self.free_slots.push(page.slot);
    }

    fn prepare_append(
        &mut self,
        _pages: &[BackendAppendPage<'_, Self::Page>],
        _start_position: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction> {
        Ok(())
    }

    fn abort_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        restored: &[&Self::Page],
        released: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in restored.iter().chain(released) {
            self.validate(**page)?;
        }
        context.stream.synchronize()?;
        context.page_table.update_slots(
            restored.iter().map(|page| page.slot),
            restored.len(),
            restored_position,
            context.stream,
        )?;
        for page in released {
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
        self.validate(*source)?;
        if valid_tokens == 0 || valid_tokens >= SM12X_KV_PAGE_TOKENS {
            return Err(Error::Shape {
                label: "DeepSeek V4 partial page copy",
                expected: format!("valid tokens in 1..{SM12X_KV_PAGE_TOKENS}"),
                actual: valid_tokens.to_string(),
            });
        }
        let allocation = self.allocate_page(context)?;
        for pool in &mut self.pools {
            if let Err(error) =
                pool.copy_page(source.slot(), allocation.page.slot(), context.stream)
            {
                context.stream.synchronize()?;
                self.rollback_page(allocation.page, context);
                return Err(error);
            }
        }
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in commit
            .committed_pages()
            .iter()
            .chain(commit.sealed_pages())
            .chain(commit.released_pages())
        {
            self.validate(**page)?;
        }
        if !commit.released_pages().is_empty() {
            context.stream.synchronize()?;
        }
        context.page_table.update_slots(
            commit.committed_pages().iter().map(|page| page.slot),
            commit.committed_pages().len(),
            commit.position(),
            context.stream,
        )?;
        if !commit.released_pages().is_empty() {
            for page in commit.released_pages() {
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
            self.validate(**page)?;
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
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        for page in &pages {
            if let Err(error) = self.validate(*page) {
                return Err(RetireError { error, pages });
            }
        }
        if let Err(error) = context.stream.synchronize() {
            return Err(RetireError { error, pages });
        }
        for page in pages {
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
    use seqcache::{AdmissionOutcome, AdmissionRequest};

    #[test]
    fn backend_reserves_and_commits_a_multi_page_prefill() -> Result<()> {
        let stream = CudaStream::new_non_blocking()?;
        let backend = Deepseek4PageBackend::new(2, 8, 3)?;
        let page_bytes = backend.page_bytes();
        let mut table = Sm12xPageTable::new(384)?;
        let table_bytes = table.managed_bytes();
        let mut cache = SequenceCache::<_, ()>::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: 3 * page_bytes + table_bytes,
                max_snapshot_bytes: 0,
                max_prefix_entries: Some(0),
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(deepseek4_cache_error)?;
        let sequence = match cache
            .admit(
                None,
                AdmissionRequest {
                    max_position: 384,
                    private_state_bytes: 0,
                    page_table_bytes: table_bytes,
                    allow_emergency: false,
                },
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
                |_, _| Ok(()),
            )
            .map_err(deepseek4_cache_error)?
        {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("dedicated cache must admit"),
        };
        let reservation = cache
            .reserve_append(
                sequence,
                257,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
            )
            .map_err(deepseek4_cache_error)?;
        assert_eq!(reservation.segments().len(), 3);
        cache
            .commit_append(
                reservation,
                257,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
            )
            .map_err(deepseek4_cache_error)?;
        assert_eq!(cache.page_table(sequence).unwrap().position(), 257);
        cache.validate().map_err(deepseek4_cache_error)
    }

    #[test]
    fn retained_pages_are_shared_and_branch_append_stays_private() -> Result<()> {
        let stream = CudaStream::new_non_blocking()?;
        let backend = Deepseek4PageBackend::new(1, 2, 4)?;
        let page_bytes = backend.page_bytes();
        let mut source_table = Sm12xPageTable::new(256)?;
        let mut branch_table = Sm12xPageTable::new(256)?;
        let table_bytes = source_table.managed_bytes();
        let mut cache = SequenceCache::<_, ()>::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: 4 * page_bytes + 2 * table_bytes,
                max_snapshot_bytes: 0,
                max_prefix_entries: None,
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(deepseek4_cache_error)?;
        let request = AdmissionRequest {
            max_position: 256,
            private_state_bytes: 0,
            page_table_bytes: table_bytes,
            allow_emergency: false,
        };
        let source = match cache
            .admit(
                None,
                request,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
                |_, _| Ok(()),
            )
            .map_err(deepseek4_cache_error)?
        {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("source must admit"),
        };
        let reservation = cache
            .reserve_append(
                source,
                SM12X_KV_PAGE_TOKENS,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .map_err(deepseek4_cache_error)?;
        cache
            .commit_append(
                reservation,
                SM12X_KV_PAGE_TOKENS,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .map_err(deepseek4_cache_error)?;
        let tokens = (0..=SM12X_KV_PAGE_TOKENS as u32).collect::<Vec<_>>();
        cache
            .retain_prefix(
                source,
                &tokens,
                (),
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .map_err(deepseek4_cache_error)?;

        let prefix = cache.lookup_prefix(&tokens);
        let branch = match cache
            .admit(
                prefix,
                request,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut branch_table,
                },
                |snapshot, position| {
                    assert!(snapshot.is_some());
                    assert_eq!(position, SM12X_KV_PAGE_TOKENS);
                    Ok(())
                },
            )
            .map_err(deepseek4_cache_error)?
        {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("branch must admit"),
        };
        assert_eq!(
            cache.page_table(source).unwrap().pages()[0],
            cache.page_table(branch).unwrap().pages()[0]
        );
        let append = cache
            .reserve_append(
                branch,
                1,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut branch_table,
                },
            )
            .map_err(deepseek4_cache_error)?;
        cache
            .commit_append(
                append,
                1,
                &mut Deepseek4CacheContext {
                    stream: &stream,
                    page_table: &mut branch_table,
                },
            )
            .map_err(deepseek4_cache_error)?;
        assert_eq!(cache.page_table(source).unwrap().position(), 128);
        assert_eq!(cache.page_table(branch).unwrap().position(), 129);
        assert_eq!(cache.page_table(source).unwrap().pages().len(), 1);
        assert_eq!(cache.page_table(branch).unwrap().pages().len(), 2);
        cache.validate().map_err(deepseek4_cache_error)
    }
}
