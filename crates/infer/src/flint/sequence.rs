//! Shared global BF16 pages and bounded local-attention snapshots.

use super::config::invalid;
use super::{FlintModel, FlintSequence};
use eider_cuda::{CudaStream, DeviceAddress, DeviceBuffer, Error, Result, SinkAttentionCache};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, BackendAppendCommit, BackendAppendPage, CacheConfig,
    CacheError, PageAllocation, PageBackend, RetainedSnapshot, RetireError, RetireOutcome,
    SequenceCache,
};
use std::sync::Arc;
use std::time::Instant;

pub const PAGE_TOKENS: usize = 128;
pub type FlintSequenceCache = SequenceCache<FlintPageBackend, FlintSnapshot>;

pub struct FlintSnapshot {
    local: Vec<Option<SinkAttentionCache>>,
}
impl RetainedSnapshot for FlintSnapshot {
    fn retained_bytes(&self) -> usize {
        self.local
            .iter()
            .flatten()
            .map(SinkAttentionCache::device_bytes)
            .sum()
    }
}

pub struct FlintPageBackend {
    namespace: Arc<()>,
    identity: Arc<()>,
    elements: usize,
}
pub struct FlintPage {
    data: DeviceBuffer<u16>,
}
pub struct FlintCacheContext<'a> {
    pub stream: &'a CudaStream,
    pub table: &'a mut FlintPageTable,
}
pub struct FlintPageTable {
    pub(crate) device: DeviceBuffer<DeviceAddress<u16>>,
    addresses: Vec<DeviceAddress<u16>>,
}
impl FlintPageTable {
    pub(crate) fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            device: DeviceBuffer::zeroed(capacity.div_ceil(PAGE_TOKENS))?,
            addresses: Vec::with_capacity(capacity.div_ceil(PAGE_TOKENS)),
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        self.device.device_bytes() + self.addresses.capacity() * size_of::<DeviceAddress<u16>>()
    }
    fn update(&mut self, pages: &[&FlintPage], stream: &CudaStream) -> Result<()> {
        if pages.len() > self.device.len() {
            return Err(invalid("page table capacity exceeded"));
        }
        if self.addresses.len() == pages.len()
            && self
                .addresses
                .iter()
                .zip(pages)
                .all(|(a, p)| *a == p.data.cuda_address())
        {
            return Ok(());
        }
        // Table addresses change only at page boundaries or admission/retirement.
        // Finish the old users before a synchronous table upload or page release.
        stream.synchronize()?;
        let addresses: Vec<_> = pages.iter().map(|p| p.data.cuda_address()).collect();
        self.device.copy_range_from_host(0, &addresses)?;
        self.addresses.clear();
        self.addresses.extend(addresses);
        Ok(())
    }
}

impl PageBackend for FlintPageBackend {
    type Page = FlintPage;
    type Context<'a> = FlintCacheContext<'a>;
    type AppendTransaction = ();
    type Error = Error;
    fn page_bytes(&self) -> usize {
        self.elements * size_of::<u16>()
    }
    fn allocate_page(&mut self, _: &mut Self::Context<'_>) -> Result<PageAllocation<FlintPage>> {
        Ok(PageAllocation {
            page: FlintPage {
                data: DeviceBuffer::zeroed(self.elements)?,
            },
            recycled: false,
        })
    }
    fn rollback_page(&mut self, _: FlintPage, _: &mut Self::Context<'_>) {}
    fn prepare_append(
        &mut self,
        _: &[BackendAppendPage<'_, FlintPage>],
        _: usize,
        _: &mut Self::Context<'_>,
    ) -> Result<()> {
        Ok(())
    }
    fn abort_append(
        &mut self,
        _: &mut (),
        pages: &[&FlintPage],
        _: &[&FlintPage],
        _: usize,
        ctx: &mut Self::Context<'_>,
    ) -> Result<()> {
        ctx.stream.synchronize()?;
        ctx.table.update(pages, ctx.stream)
    }
    fn copy_partial_page(
        &mut self,
        source: &FlintPage,
        valid: usize,
        ctx: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<FlintPage>> {
        if valid == 0 || valid >= PAGE_TOKENS {
            return Err(invalid("invalid partial page length"));
        }
        let mut page = self.allocate_page(ctx)?;
        page.page.data.copy_prefix_from_device_on_stream(
            &source.data,
            self.elements,
            ctx.stream,
        )?;
        ctx.stream.synchronize()?;
        Ok(page)
    }
    fn commit_append(
        &mut self,
        _: &mut (),
        commit: BackendAppendCommit<'_, FlintPage>,
        ctx: &mut Self::Context<'_>,
    ) -> Result<()> {
        if !commit.released_pages().is_empty() {
            ctx.stream.synchronize()?;
        }
        ctx.table.update(commit.committed_pages(), ctx.stream)
    }
    fn update_page_table(
        &mut self,
        pages: &[&FlintPage],
        _: usize,
        ctx: &mut Self::Context<'_>,
    ) -> Result<()> {
        ctx.table.update(pages, ctx.stream)
    }
    fn retire_pages(
        &mut self,
        pages: Vec<FlintPage>,
        ctx: &mut Self::Context<'_>,
    ) -> std::result::Result<RetireOutcome, RetireError<Error, FlintPage>> {
        if let Err(error) = ctx.stream.synchronize() {
            return Err(RetireError { error, pages });
        }
        drop(pages);
        Ok(RetireOutcome::default())
    }
    fn retirement_is_immediate(&self) -> bool {
        true
    }
    fn poll_reclaimed(&mut self, _: &mut Self::Context<'_>) -> Result<usize> {
        Ok(0)
    }
}

pub(crate) fn cache_error(e: CacheError<Error>) -> Error {
    invalid(format!("sequence cache: {e}"))
}

impl FlintModel {
    /// Reserves worst-case live capacity plus the configured retained-prefix budget.
    /// Physical global pages are allocated only as requests append tokens.
    pub fn new_sequence_cache(
        &self,
        sequences: usize,
        context: usize,
        retained: usize,
    ) -> Result<FlintSequenceCache> {
        if sequences == 0 || context == 0 || context > self.config().max_context {
            return Err(invalid("invalid sequence cache capacity"));
        }
        let c = self.config();
        let elements =
            c.sliding.iter().filter(|&&local| !local).count() * 2 * PAGE_TOKENS * c.kv_heads * 128;
        let local =
            c.sliding.iter().filter(|&&local| local).count() * 2 * c.window * c.kv_heads * 128 * 2;
        let table = context.div_ceil(PAGE_TOKENS) * 2 * size_of::<DeviceAddress<u16>>();
        let active = context
            .div_ceil(PAGE_TOKENS)
            .checked_mul(elements * 2)
            .and_then(|v| v.checked_add(local + table + c.vocab * 8 + 12))
            .and_then(|v| v.checked_mul(sequences))
            .and_then(|v| v.checked_add(retained))
            .ok_or_else(|| invalid("cache budget overflow"))?;
        SequenceCache::new(
            CacheConfig {
                page_tokens: PAGE_TOKENS,
                max_managed_bytes: active,
                max_snapshot_bytes: retained,
                max_prefix_entries: (retained == 0).then_some(0),
                emergency_bytes: 0,
            },
            FlintPageBackend {
                elements,
                namespace: Arc::clone(&self.cache_namespace),
                identity: Arc::new(()),
            },
        )
        .map_err(cache_error)
    }
    /// Admits a request and restores its longest retained prompt prefix.
    pub fn admit<'m>(
        &'m self,
        cache: &mut FlintSequenceCache,
        capacity: usize,
        prompt: &[u32],
    ) -> Result<Option<FlintSequence<'m>>> {
        let identity = cache.with_backend(|backend| {
            if !Arc::ptr_eq(&backend.namespace, &self.cache_namespace) {
                return Err(invalid("sequence cache belongs to another model"));
            }
            Ok(Arc::clone(&backend.identity))
        })?;
        let mut state = self.new_sequence_state(capacity)?;
        state.cache_identity = Some(identity);
        let prefix = cache.lookup_prefix(prompt);
        let outcome = cache
            .admit(
                prefix,
                AdmissionRequest {
                    max_position: capacity,
                    private_state_bytes: state.private_bytes(),
                    page_table_bytes: state.table.bytes(),
                    allow_emergency: false,
                },
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
                |snapshot, position| {
                    if let Some(snapshot) = snapshot {
                        let start = Instant::now();
                        for (dest, src) in state.caches.iter_mut().zip(&snapshot.local) {
                            if let (Some(dest), Some(src)) = (dest, src) {
                                dest.restore(src, &self.stream)?;
                            }
                        }
                        self.stream.synchronize()?;
                        state.restore_duration = start.elapsed();
                    }
                    state.position = position;
                    Ok(())
                },
            )
            .map_err(cache_error)?;
        let AdmissionOutcome::Admitted(id) = outcome else {
            return Ok(None);
        };
        state.cache_id = Some(id);
        self.stream.synchronize()?;
        Ok(Some(state))
    }
    /// Retains shared global pages and a bounded local ring at an aligned position.
    pub fn retain_prefix(
        &self,
        cache: &mut FlintSequenceCache,
        state: &mut FlintSequence,
        tokens: &[u32],
    ) -> Result<()> {
        state.validate_cache(cache)?;
        if !state.valid || state.position == 0 || !state.position.is_multiple_of(PAGE_TOKENS) {
            return Err(invalid("prefix must be valid and page aligned"));
        }
        if cache.contains_prefix(tokens, state.position) {
            return Ok(());
        }
        let snapshot_bytes: usize = state
            .caches
            .iter()
            .flatten()
            .map(SinkAttentionCache::device_bytes)
            .sum();
        if snapshot_bytes > cache.config().max_snapshot_bytes {
            return Err(invalid(
                "prefix budget cannot hold the local-attention snapshot",
            ));
        }
        let local = state
            .caches
            .iter()
            .map(|c| c.as_ref().map(|c| c.snapshot(&self.stream)).transpose())
            .collect::<Result<Vec<_>>>()?;
        self.stream.synchronize()?;
        cache
            .retain_prefix(
                state
                    .cache_id
                    .ok_or_else(|| invalid("unadmitted sequence"))?,
                tokens,
                FlintSnapshot { local },
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            )
            .map_err(cache_error)?;
        Ok(())
    }
    /// Releases active ownership; retained prefixes keep their shared pages alive.
    pub fn finish(&self, cache: &mut FlintSequenceCache, mut state: FlintSequence) -> Result<()> {
        state.validate_cache(cache)?;
        cache
            .finish(
                state
                    .cache_id
                    .take()
                    .ok_or_else(|| invalid("unadmitted sequence"))?,
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            )
            .map_err(cache_error)
    }
}

impl FlintSequence<'_> {
    pub(super) fn validate_cache(&self, cache: &mut FlintSequenceCache) -> Result<()> {
        cache.with_backend(|backend| {
            if self
                .cache_identity
                .as_ref()
                .is_some_and(|identity| Arc::ptr_eq(identity, &backend.identity))
            {
                Ok(())
            } else {
                Err(invalid("sequence belongs to another cache"))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paged_backend_rolls_back_reservations_and_preserves_shared_readers() -> Result<()> {
        let stream = CudaStream::new_non_blocking()?;
        let mut table = FlintPageTable::new(256)?;
        let mut other_table = FlintPageTable::new(256)?;
        let mut backend = FlintPageBackend {
            elements: 512,
            namespace: Arc::new(()),
            identity: Arc::new(()),
        };
        let page_bytes = backend.page_bytes();
        let source = FlintPage {
            data: DeviceBuffer::from_host(&(0..512).map(|i| i as u16).collect::<Vec<_>>())?,
        };
        let copied = backend.copy_partial_page(
            &source,
            17,
            &mut FlintCacheContext {
                stream: &stream,
                table: &mut table,
            },
        )?;
        assert_ne!(source.data.cuda_address(), copied.page.data.cuda_address());
        assert_eq!(
            source.data.copy_to_host(&stream)?.as_slice(),
            copied.page.data.copy_to_host(&stream)?.as_slice()
        );
        drop(copied);
        drop(source);
        let request = AdmissionRequest {
            max_position: 256,
            private_state_bytes: 0,
            page_table_bytes: table.bytes(),
            allow_emergency: false,
        };
        let mut cache = SequenceCache::<_, ()>::new(
            CacheConfig {
                page_tokens: PAGE_TOKENS,
                max_managed_bytes: 2 * page_bytes + 2 * table.bytes(),
                max_snapshot_bytes: 0,
                max_prefix_entries: None,
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(cache_error)?;
        let AdmissionOutcome::Admitted(id) = cache
            .admit(
                None,
                request,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
                |_, _| Ok(()),
            )
            .map_err(cache_error)?
        else {
            panic!("first admission blocked");
        };
        assert_eq!(cache.stats().resident_pages, 0);
        assert!(matches!(
            cache
                .admit(
                    None,
                    request,
                    &mut FlintCacheContext {
                        stream: &stream,
                        table: &mut other_table
                    },
                    |_, _| Ok(())
                )
                .map_err(cache_error)?,
            AdmissionOutcome::WouldBlock
        ));
        let reservation = cache
            .reserve_append(
                id,
                192,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
            )
            .map_err(cache_error)?;
        assert_eq!(cache.stats().resident_pages, 2);
        assert_eq!(table.addresses.len(), 2);
        cache
            .abort_append(
                reservation,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
            )
            .map_err(cache_error)?;
        assert_eq!(cache.stats().resident_pages, 0);
        assert_eq!(cache.stats().reserved_pages, 2);
        assert!(table.addresses.is_empty());
        let reservation = cache
            .reserve_append(
                id,
                192,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
            )
            .map_err(cache_error)?;
        cache
            .commit_append(
                reservation,
                128,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
            )
            .map_err(cache_error)?;
        assert_eq!(cache.stats().resident_pages, 1);
        assert_eq!(table.addresses.len(), 1);
        let tokens: Vec<_> = (0..129).collect();
        cache
            .retain_prefix(
                id,
                &tokens,
                (),
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
            )
            .map_err(cache_error)?;
        let prefix = cache.lookup_prefix(&tokens);
        let AdmissionOutcome::Admitted(reader) = cache
            .admit(
                prefix,
                AdmissionRequest {
                    max_position: 128,
                    ..request
                },
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut other_table,
                },
                |_, position| {
                    assert_eq!(position, 128);
                    Ok(())
                },
            )
            .map_err(cache_error)?
        else {
            panic!("shared reader admission blocked");
        };
        assert_eq!(cache.stats().resident_pages, 1);
        assert_eq!(table.addresses, other_table.addresses);
        cache
            .finish(
                id,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut table,
                },
            )
            .map_err(cache_error)?;
        cache
            .finish(
                reader,
                &mut FlintCacheContext {
                    stream: &stream,
                    table: &mut other_table,
                },
            )
            .map_err(cache_error)?;
        assert_eq!(cache.stats().active_sequences, 0);
        assert_eq!(cache.stats().reserved_pages, 0);
        assert_eq!(cache.stats().page_table_bytes, 0);
        assert_eq!(cache.stats().unique_resident_page_bytes, page_bytes);
        Ok(())
    }
}
