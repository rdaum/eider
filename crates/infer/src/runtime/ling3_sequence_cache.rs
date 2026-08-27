//! Shared paged MLA storage for Ling 3 sequences.

use crate::ling3::{Ling3Model, Ling3ModelState, Ling3ModelWorkspace};
use crate::sm12x_cache::Sm12xPageTable;
use nvfp4::{CudaStream, DeviceBuffer, Error, Result, SM12X_KV_PAGE_TOKENS};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, BackendAppendCommit, BackendAppendPage, CacheConfig,
    CacheError, PageAllocation, PageBackend, RetireError, RetireOutcome, SequenceCache, SequenceId,
};
use std::mem::size_of;

pub type Ling3SequenceCache = SequenceCache<Ling3PageBackend, ()>;

pub struct Ling3Sequence {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    pub(crate) state: Ling3ModelState,
    pub(crate) workspace: Ling3ModelWorkspace,
}

impl Ling3Sequence {
    pub fn position(&self) -> usize {
        self.state.position()
    }

    pub fn device_bytes(&self) -> usize {
        self.state.device_bytes() + self.workspace.device_bytes() + self.page_table.managed_bytes()
    }

    pub fn finish(self, stream: &CudaStream, cache: &mut Ling3SequenceCache) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Ling3CacheContext {
                    stream,
                    page_table: &mut page_table,
                },
            )
            .map_err(ling3_cache_error)
    }
}

pub fn new_ling3_sequence_cache(
    model: &Ling3Model,
    sequence_capacity: usize,
    max_context_tokens: usize,
) -> Result<Ling3SequenceCache> {
    if sequence_capacity == 0 || max_context_tokens == 0 {
        return Err(Error::Shape {
            label: "Ling 3 sequence cache",
            expected: "positive sequence and context capacities".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        });
    }
    let page_slots = sequence_capacity
        .checked_mul(max_context_tokens.div_ceil(SM12X_KV_PAGE_TOKENS))
        .ok_or_else(|| Error::Shape {
            label: "Ling 3 sequence-cache pages",
            expected: "page count without overflow".to_string(),
            actual: format!("sequences={sequence_capacity} context={max_context_tokens}"),
        })?;
    let backend = Ling3PageBackend::new(model.mla_page_layouts(), page_slots)?;
    let private = model.new_state(max_context_tokens)?.device_bytes()
        + model.new_workspace()?.device_bytes()
        + Sm12xPageTable::new(max_context_tokens)?.managed_bytes();
    let private_total = private
        .checked_mul(sequence_capacity)
        .ok_or_else(|| Error::Shape {
            label: "Ling 3 private sequence bytes",
            expected: "private bytes without overflow".to_string(),
            actual: format!("private={private} sequences={sequence_capacity}"),
        })?;
    let managed = page_slots
        .checked_mul(backend.page_bytes())
        .and_then(|bytes| bytes.checked_add(private_total))
        .ok_or_else(|| Error::Shape {
            label: "Ling 3 sequence-cache bytes",
            expected: "managed bytes without overflow".to_string(),
            actual: format!("pages={page_slots} private={private}"),
        })?;
    Ling3SequenceCache::new(
        CacheConfig {
            page_tokens: SM12X_KV_PAGE_TOKENS,
            max_managed_bytes: managed,
            max_snapshot_bytes: 0,
            max_prefix_entries: Some(0),
            emergency_bytes: 0,
        },
        backend,
    )
    .map_err(ling3_cache_error)
}

pub fn admit_ling3_sequence(
    model: &Ling3Model,
    cache: &mut Ling3SequenceCache,
    max_tokens: usize,
    stream: &CudaStream,
) -> Result<Option<Ling3Sequence>> {
    let mut state = model.new_state(max_tokens)?;
    let mut workspace = model.new_workspace()?;
    model.prepare_decode_graphs(&mut state, &mut workspace, stream)?;
    let mut page_table = Sm12xPageTable::new(max_tokens)?;
    let outcome = cache
        .admit(
            None,
            AdmissionRequest {
                max_position: max_tokens,
                private_state_bytes: state.device_bytes() + workspace.device_bytes(),
                page_table_bytes: page_table.managed_bytes(),
                allow_emergency: false,
            },
            &mut Ling3CacheContext {
                stream,
                page_table: &mut page_table,
            },
            |snapshot, position| {
                debug_assert!(snapshot.is_none());
                debug_assert_eq!(position, 0);
                Ok(())
            },
        )
        .map_err(ling3_cache_error)?;
    let AdmissionOutcome::Admitted(cache_id) = outcome else {
        return Ok(None);
    };
    Ok(Some(Ling3Sequence {
        cache_id,
        page_table,
        state,
        workspace,
    }))
}

pub(crate) fn ling3_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Ling 3 sequence cache",
        detail: error.to_string(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ling3Page {
    slot: u32,
}

impl Ling3Page {
    pub(crate) fn slot(self) -> usize {
        self.slot as usize
    }
}

pub struct Ling3CacheContext<'a> {
    pub stream: &'a CudaStream,
    pub page_table: &'a mut Sm12xPageTable,
}

pub(crate) struct Ling3MlaPagePool {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    key_width: usize,
    value_width: usize,
    slots: usize,
}

impl Ling3MlaPagePool {
    fn new(slots: usize, key_width: usize, value_width: usize) -> Result<Self> {
        let key_values = slots
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .and_then(|rows| rows.checked_mul(key_width))
            .ok_or_else(|| Error::Shape {
                label: "Ling 3 MLA key pool",
                expected: "pool shape without overflow".to_string(),
                actual: format!("slots={slots} width={key_width}"),
            })?;
        let value_values = slots
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .and_then(|rows| rows.checked_mul(value_width))
            .ok_or_else(|| Error::Shape {
                label: "Ling 3 MLA value pool",
                expected: "pool shape without overflow".to_string(),
                actual: format!("slots={slots} width={value_width}"),
            })?;
        Ok(Self {
            key: DeviceBuffer::zeroed(key_values)?,
            value: DeviceBuffer::zeroed(value_values)?,
            key_width,
            value_width,
            slots,
        })
    }

    pub(crate) fn buffers_mut(&mut self) -> (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>) {
        (&mut self.key, &mut self.value)
    }

    pub(crate) fn buffers(&self) -> (&DeviceBuffer<f32>, &DeviceBuffer<f32>) {
        (&self.key, &self.value)
    }

    fn page_bytes(&self) -> usize {
        SM12X_KV_PAGE_TOKENS * (self.key_width + self.value_width) * size_of::<f32>()
    }

    fn copy_page(&mut self, source: usize, destination: usize, stream: &CudaStream) -> Result<()> {
        if source >= self.slots || destination >= self.slots {
            return Err(Error::Shape {
                label: "Ling 3 MLA page copy",
                expected: format!("slots below {}", self.slots),
                actual: format!("source={source} destination={destination}"),
            });
        }
        let key_values = SM12X_KV_PAGE_TOKENS * self.key_width;
        let value_values = SM12X_KV_PAGE_TOKENS * self.value_width;
        self.key.copy_within_on_stream(
            source * key_values,
            destination * key_values,
            key_values,
            stream,
        )?;
        self.value.copy_within_on_stream(
            source * value_values,
            destination * value_values,
            value_values,
            stream,
        )
    }
}

pub struct Ling3PageBackend {
    pools: Vec<Option<Ling3MlaPagePool>>,
    free_slots: Vec<u32>,
    used_slots: Vec<bool>,
    ever_used_slots: Vec<bool>,
    page_bytes: usize,
}

impl Ling3PageBackend {
    fn new(layouts: Vec<Option<(usize, usize)>>, slots: usize) -> Result<Self> {
        if slots == 0 || slots > u32::MAX as usize {
            return Err(Error::Shape {
                label: "Ling 3 page slots",
                expected: format!("1..={}", u32::MAX),
                actual: slots.to_string(),
            });
        }
        let mut page_bytes = 0usize;
        let mut pools = Vec::with_capacity(layouts.len());
        for layout in layouts {
            let pool = layout
                .map(|(key_width, value_width)| {
                    Ling3MlaPagePool::new(slots, key_width, value_width)
                })
                .transpose()?;
            if let Some(pool) = &pool {
                page_bytes =
                    page_bytes
                        .checked_add(pool.page_bytes())
                        .ok_or_else(|| Error::Shape {
                            label: "Ling 3 page bytes",
                            expected: "page bytes without overflow".to_string(),
                            actual: format!("accumulated={page_bytes} next={}", pool.page_bytes()),
                        })?;
            }
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

    pub(crate) fn pool_mut(&mut self, layer: usize) -> Result<&mut Ling3MlaPagePool> {
        self.pools
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Format {
                label: "Ling 3 MLA page pool",
                detail: format!("layer {layer} has no MLA storage"),
            })
    }

    fn validate(&self, page: Ling3Page) -> Result<()> {
        if page.slot() >= self.used_slots.len() || !self.used_slots[page.slot()] {
            return Err(Error::Format {
                label: "Ling 3 physical page",
                detail: format!("slot {} is not allocated", page.slot()),
            });
        }
        Ok(())
    }
}

impl PageBackend for Ling3PageBackend {
    type Page = Ling3Page;
    type Context<'a> = Ling3CacheContext<'a>;
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
            label: "Ling 3 physical page allocation",
            detail: "preallocated page pool exhausted".to_string(),
        })?;
        let index = slot as usize;
        let recycled = self.ever_used_slots[index];
        self.used_slots[index] = true;
        self.ever_used_slots[index] = true;
        Ok(PageAllocation {
            page: Ling3Page { slot },
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
                label: "Ling 3 partial page copy",
                expected: format!("valid tokens in 1..{SM12X_KV_PAGE_TOKENS}"),
                actual: valid_tokens.to_string(),
            });
        }
        let allocation = self.allocate_page(context)?;
        for pool in self.pools.iter_mut().filter_map(Option::as_mut) {
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

    #[test]
    fn backend_reserves_and_commits_a_multi_page_prefill() -> Result<()> {
        let stream = CudaStream::new_non_blocking()?;
        let backend = Ling3PageBackend::new(vec![Some((10, 8)), None], 3)?;
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
        .map_err(ling3_cache_error)?;
        let sequence = match cache
            .admit(
                None,
                AdmissionRequest {
                    max_position: 384,
                    private_state_bytes: 0,
                    page_table_bytes: table_bytes,
                    allow_emergency: false,
                },
                &mut Ling3CacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
                |_, _| Ok(()),
            )
            .map_err(ling3_cache_error)?
        {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("dedicated cache must admit"),
        };
        let reservation = cache
            .reserve_append(
                sequence,
                257,
                &mut Ling3CacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
            )
            .map_err(ling3_cache_error)?;
        assert_eq!(
            reservation
                .segments()
                .iter()
                .map(|segment| (
                    segment.page_offset(),
                    segment.input_offset(),
                    segment.rows()
                ))
                .collect::<Vec<_>>(),
            [(0, 0, 128), (0, 128, 128), (0, 256, 1)]
        );
        cache
            .commit_append(
                reservation,
                257,
                &mut Ling3CacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
            )
            .map_err(ling3_cache_error)?;
        assert_eq!(cache.page_table(sequence).unwrap().position(), 257);
        assert_eq!(cache.page_table(sequence).unwrap().pages().len(), 3);
        cache.validate().map_err(ling3_cache_error)
    }
}
