//! Shared global-visit KV pages and retained local and convolution snapshots.
//!
//! The eight global visits append BF16 K/V to shared 128-token page bundles laid
//! out as `[global visit][K/V][page token][KV head][dimension]`; with a DSpark
//! draft a final slot holds its global layer's context K/V. Every other piece
//! of per-visit history (local KV rings, all short-convolution inputs, and the
//! draft's local rings) is request-private and retained by copying it into a
//! snapshot.

use super::config::invalid;
use super::draft::DraftState;
use super::model::{Inkling2Model, Inkling2Sequence};
use eider_cuda::{CudaStream, DeviceAddress, DeviceBuffer, Error, Result};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, BackendAppendCommit, BackendAppendPage, CacheConfig,
    CacheError, PageAllocation, PageBackend, RetainedSnapshot, RetireError, RetireOutcome,
    SequenceCache,
};
use std::sync::Arc;
use std::time::Instant;

pub const PAGE_TOKENS: usize = 128;
pub type Inkling2SequenceCache = SequenceCache<Inkling2PageBackend, Inkling2Snapshot>;

/// Request-private visit history at a retained prompt position.
pub struct Inkling2Snapshot {
    pub(super) visits: Vec<super::model::VisitState>,
    pub(super) draft: Option<DraftState>,
}

impl RetainedSnapshot for Inkling2Snapshot {
    fn retained_bytes(&self) -> usize {
        self.visits
            .iter()
            .map(super::model::VisitState::device_bytes)
            .sum::<usize>()
            + self.draft.as_ref().map_or(0, DraftState::device_bytes)
    }
}

pub struct Inkling2PageBackend {
    namespace: Arc<()>,
    identity: Arc<()>,
    elements: usize,
}

pub struct Inkling2Page {
    data: DeviceBuffer<u16>,
}

pub struct Inkling2CacheContext<'a> {
    pub stream: &'a CudaStream,
    pub table: &'a mut Inkling2PageTable,
}

/// Device page addresses of one sequence, uploaded only when pages change.
pub struct Inkling2PageTable {
    pub(super) device: DeviceBuffer<DeviceAddress<u16>>,
    addresses: Vec<DeviceAddress<u16>>,
}

impl Inkling2PageTable {
    pub(super) fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            device: DeviceBuffer::zeroed(capacity.div_ceil(PAGE_TOKENS).max(1))?,
            addresses: Vec::with_capacity(capacity.div_ceil(PAGE_TOKENS)),
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.device.device_bytes() + self.addresses.capacity() * size_of::<DeviceAddress<u16>>()
    }

    fn update(&mut self, pages: &[&Inkling2Page], stream: &CudaStream) -> Result<()> {
        if pages.len() > self.device.len() {
            return Err(invalid("page table capacity exceeded"));
        }
        if self.addresses.len() == pages.len()
            && self
                .addresses
                .iter()
                .zip(pages)
                .all(|(address, page)| *address == page.data.cuda_address())
        {
            return Ok(());
        }
        // Addresses change only at page boundaries, admission, or retirement.
        // Finish earlier readers before the synchronous upload or a release.
        stream.synchronize()?;
        let addresses = pages
            .iter()
            .map(|page| page.data.cuda_address())
            .collect::<Vec<_>>();
        self.device.copy_range_from_host(0, &addresses)?;
        self.addresses.clear();
        self.addresses.extend(addresses);
        Ok(())
    }
}

impl PageBackend for Inkling2PageBackend {
    type Page = Inkling2Page;
    type Context<'a> = Inkling2CacheContext<'a>;
    type AppendTransaction = ();
    type Error = Error;

    fn page_bytes(&self) -> usize {
        self.elements * size_of::<u16>()
    }

    fn allocate_page(&mut self, _: &mut Self::Context<'_>) -> Result<PageAllocation<Inkling2Page>> {
        Ok(PageAllocation {
            page: Inkling2Page {
                data: DeviceBuffer::zeroed(self.elements)?,
            },
            recycled: false,
        })
    }

    fn rollback_page(&mut self, _: Inkling2Page, _: &mut Self::Context<'_>) {}

    fn prepare_append(
        &mut self,
        _: &[BackendAppendPage<'_, Inkling2Page>],
        _: usize,
        _: &mut Self::Context<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn abort_append(
        &mut self,
        _: &mut (),
        pages: &[&Inkling2Page],
        _: &[&Inkling2Page],
        _: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        context.stream.synchronize()?;
        context.table.update(pages, context.stream)
    }

    fn copy_partial_page(
        &mut self,
        source: &Inkling2Page,
        valid: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Inkling2Page>> {
        if valid == 0 || valid >= PAGE_TOKENS {
            return Err(invalid("invalid partial page length"));
        }
        let mut page = self.allocate_page(context)?;
        page.page.data.copy_prefix_from_device_on_stream(
            &source.data,
            self.elements,
            context.stream,
        )?;
        context.stream.synchronize()?;
        Ok(page)
    }

    fn commit_append(
        &mut self,
        _: &mut (),
        commit: BackendAppendCommit<'_, Inkling2Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        if !commit.released_pages().is_empty() {
            context.stream.synchronize()?;
        }
        context
            .table
            .update(commit.committed_pages(), context.stream)
    }

    fn update_page_table(
        &mut self,
        pages: &[&Inkling2Page],
        _: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        context.table.update(pages, context.stream)
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Inkling2Page>,
        context: &mut Self::Context<'_>,
    ) -> std::result::Result<RetireOutcome, RetireError<Error, Inkling2Page>> {
        if let Err(error) = context.stream.synchronize() {
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

pub(super) fn cache_error(error: CacheError<Error>) -> Error {
    invalid(format!("sequence cache: {error}"))
}

impl Inkling2Model {
    /// Reserves worst-case live capacity plus the retained-prefix budget.
    /// Physical global pages are allocated only as requests append tokens.
    pub fn new_sequence_cache(
        &self,
        sequences: usize,
        context: usize,
        retained: usize,
    ) -> Result<Inkling2SequenceCache> {
        let c = &self.config().text_config;
        if sequences == 0 || context == 0 || context > c.model_max_length {
            return Err(invalid("invalid sequence cache capacity"));
        }
        let slots = self.global_visits() + usize::from(self.draft.is_some());
        let elements = slots * 2 * PAGE_TOKENS * c.num_key_value_heads * c.head_dim;
        let private = self.private_sequence_bytes();
        let table = Inkling2PageTable::new(context)?.bytes();
        let active = context
            .div_ceil(PAGE_TOKENS)
            .checked_mul(elements * size_of::<u16>())
            .and_then(|bytes| bytes.checked_add(private + table))
            .and_then(|bytes| bytes.checked_mul(sequences))
            .and_then(|bytes| bytes.checked_add(retained))
            .ok_or_else(|| invalid("cache budget overflow"))?;
        SequenceCache::new(
            CacheConfig {
                page_tokens: PAGE_TOKENS,
                max_managed_bytes: active,
                max_snapshot_bytes: retained,
                max_prefix_entries: (retained == 0).then_some(0),
                emergency_bytes: 0,
            },
            Inkling2PageBackend {
                elements,
                namespace: Arc::clone(&self.cache_namespace),
                identity: Arc::new(()),
            },
        )
        .map_err(cache_error)
    }

    /// Admits a request and restores its longest retained prompt prefix.
    /// Returns `None` while the cache cannot hold another sequence.
    pub fn admit<'m>(
        &'m self,
        cache: &mut Inkling2SequenceCache,
        capacity: usize,
        prompt: &[u32],
    ) -> Result<Option<Inkling2Sequence<'m>>> {
        let identity = cache.with_backend(|backend| {
            if !Arc::ptr_eq(&backend.namespace, &self.cache_namespace) {
                return Err(invalid("sequence cache belongs to another model"));
            }
            Ok(Arc::clone(&backend.identity))
        })?;
        let mut state = self.new_sequence(capacity)?;
        state.cache_identity = Some(identity);
        let prefix = cache.lookup_prefix(prompt);
        let private_state_bytes = state.private_bytes();
        let page_table_bytes = state.table.bytes();
        let outcome = cache
            .admit(
                prefix,
                AdmissionRequest {
                    max_position: capacity,
                    private_state_bytes,
                    page_table_bytes,
                    allow_emergency: false,
                },
                &mut Inkling2CacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
                |snapshot, position| {
                    if let Some(snapshot) = snapshot {
                        let start = Instant::now();
                        for (visit, source) in state.visits.iter_mut().zip(&snapshot.visits) {
                            visit.restore(source, &self.stream)?;
                        }
                        match (state.draft.as_mut(), snapshot.draft.as_ref()) {
                            (Some(draft), Some(source)) => draft.restore(source, &self.stream)?,
                            (None, None) => {}
                            _ => return Err(invalid("draft snapshot presence mismatch")),
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

    /// Retains the shared global pages and a copy of the private visit history
    /// at a page-aligned prompt position.
    pub fn retain_prefix(
        &self,
        cache: &mut Inkling2SequenceCache,
        state: &mut Inkling2Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        state.validate_cache(cache)?;
        if !state.valid || state.position == 0 || !state.position.is_multiple_of(PAGE_TOKENS) {
            return Err(invalid("prefix must be valid and page aligned"));
        }
        if cache.contains_prefix(tokens, state.position) {
            return Ok(());
        }
        let bytes = state
            .visits
            .iter()
            .map(super::model::VisitState::device_bytes)
            .sum::<usize>()
            + state.draft.as_ref().map_or(0, DraftState::device_bytes);
        if bytes > cache.config().max_snapshot_bytes {
            return Err(invalid(
                "prefix budget cannot hold the visit-history snapshot",
            ));
        }
        let visits = state
            .visits
            .iter()
            .map(|visit| visit.snapshot(&self.stream))
            .collect::<Result<Vec<_>>>()?;
        let draft = state
            .draft
            .as_ref()
            .map(|draft| draft.snapshot(&self.stream))
            .transpose()?;
        self.stream.synchronize()?;
        cache
            .retain_prefix(
                state
                    .cache_id
                    .ok_or_else(|| invalid("unadmitted sequence"))?,
                tokens,
                Inkling2Snapshot { visits, draft },
                &mut Inkling2CacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            )
            .map_err(cache_error)?;
        Ok(())
    }

    /// Releases active ownership; retained prefixes keep their shared pages.
    pub fn finish(
        &self,
        cache: &mut Inkling2SequenceCache,
        mut state: Inkling2Sequence,
    ) -> Result<()> {
        state.validate_cache(cache)?;
        cache
            .finish(
                state
                    .cache_id
                    .take()
                    .ok_or_else(|| invalid("unadmitted sequence"))?,
                &mut Inkling2CacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            )
            .map_err(cache_error)
    }
}

impl Inkling2Sequence<'_> {
    pub(super) fn validate_cache(&self, cache: &mut Inkling2SequenceCache) -> Result<()> {
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
