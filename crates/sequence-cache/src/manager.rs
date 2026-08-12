//! Sequence ownership, admission control, and prefix retention.
//!
//! This module holds the cache's handle types, configuration, exact stats, and
//! the [`SequenceCache`] state machine. Storage operations are delegated to the
//! configured backend; index bookkeeping lives in the crate's index module.

use crate::RetainedSnapshot;
use crate::backend::PageBackend;
use crate::error::{CacheError, ConfigError, Result};
use crate::index::{PrefixIndex, PrefixKey};
use crate::metrics::CacheMetrics;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::time::Instant;

/// Declares a copyable arena handle which rejects reuse after removal.
///
/// The handle pairs a slot index with the slot's current generation, so a stale
/// copy fails validation once the slot is recycled for a new owner.
macro_rules! generational_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name {
            slot: u32,
            generation: u32,
        }

        impl $name {
            fn new(slot: usize, generation: u32) -> Self {
                Self {
                    slot: slot as u32,
                    generation,
                }
            }

            fn slot(self) -> usize {
                self.slot as usize
            }
        }
    };
}

generational_id!(
    /// Handle for one manager-owned logical page.
    ///
    /// Resolved through [`SequenceCache::page`] and invalidated when the page is
    /// retired. Stale handles fail with [`CacheError::StalePage`].
    PageId
);
generational_id!(
    /// Handle for one admitted sequence.
    ///
    /// Obtained from [`SequenceCache::admit`] or [`SequenceCache::branch`] and
    /// invalidated by [`SequenceCache::finish`]. Stale handles fail with
    /// [`CacheError::StaleSequence`].
    SequenceId
);

/// Stable, never-reused identity for one retained prefix entry.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixEntryId(u64);

/// Stable, never-reused identity for one interned page-sized token block.
///
/// Token blocks are the key material of the content-based prefix index: two
/// prefixes sharing a leading run of identical page-sized token runs resolve to
/// the same block IDs. Identities are assigned monotonically and survive only
/// as long as some prefix entry references the block.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenBlockId(u64);

impl TokenBlockId {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

/// Immutable manager geometry and byte limits.
///
/// The byte budget covers unique resident pages, outstanding reservations,
/// per-sequence private state and page tables, and retained snapshots. Physical
/// page size is reported by the backend, so all accounting is exact once the
/// cache is constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheConfig {
    /// Tokens stored per page; every shared or sealed page covers exactly this
    /// many positions.
    pub page_tokens: usize,
    /// Total bytes the cache may commit across pages, reservations, private
    /// state, page tables, and snapshots.
    pub max_managed_bytes: usize,
    /// Total bytes retained snapshots may occupy within the managed budget.
    pub max_snapshot_bytes: usize,
    /// Optional hard cap on retained prefix entries, in addition to the byte
    /// limits.
    pub max_prefix_entries: Option<usize>,
    /// Capacity unavailable to ordinary admissions but usable by runtime policy.
    ///
    /// Admissions with `allow_emergency` set may draw on this margin; all other
    /// admissions must fit below `max_managed_bytes - emergency_bytes`.
    pub emergency_bytes: usize,
}

/// Per-request strict admission requirements.
///
/// A request either fits completely — including the reserved pages required to
/// reach `max_position` — or is declined with
/// [`AdmissionOutcome::WouldBlock`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionRequest {
    /// Maximum sequence length in tokens the admission must provision for.
    pub max_position: usize,
    /// Exact bytes of per-sequence private (non-paged) state.
    pub private_state_bytes: usize,
    /// Exact bytes of the per-sequence backend page table.
    pub page_table_bytes: usize,
    /// Whether this request may consume the configured emergency margin.
    pub allow_emergency: bool,
}

/// A successful prefix lookup. The handle becomes stale if the entry is evicted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixMatch {
    entry: PrefixEntryId,
    position: usize,
    page_count: usize,
}

impl PrefixMatch {
    /// Identity of the matched prefix entry.
    pub fn entry_id(self) -> PrefixEntryId {
        self.entry
    }

    /// Page-aligned token position where the matched prefix ends.
    pub fn position(self) -> usize {
        self.position
    }

    /// Number of pages the matched prefix occupies.
    pub fn page_count(self) -> usize {
        self.page_count
    }
}

/// Result of an admission request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionOutcome {
    /// The request fit within capacity; carries the new sequence handle.
    Admitted(SequenceId),
    /// The request cannot currently fit, even after evicting evictable
    /// prefixes. The caller should retry after other work finishes.
    ///
    /// Pressure is reported here rather than as an error because it is part of
    /// normal scheduler behaviour.
    WouldBlock,
}

/// Result of retaining a sequence's aligned prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainOutcome {
    /// The prefix was newly retained; carries the entry identity.
    Inserted(PrefixEntryId),
    /// The exact prefix was already retained; carries the existing entry
    /// identity. No additional ownership is taken.
    Duplicate(PrefixEntryId),
}

/// One ordered writable segment of an append reservation.
///
/// Segments cover the reservation without gaps. `input_offset` addresses the
/// first row in the caller's compute chunk; `page_offset` addresses the first
/// row in the physical page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendSegment {
    page: PageId,
    page_offset: usize,
    input_offset: usize,
    rows: usize,
}

impl AppendSegment {
    /// Logical page receiving this segment.
    pub fn page(self) -> PageId {
        self.page
    }

    /// First writable row in the physical page.
    pub fn page_offset(self) -> usize {
        self.page_offset
    }

    /// First source row in the caller's compute chunk.
    pub fn input_offset(self) -> usize {
        self.input_offset
    }

    /// Number of consecutive rows in this segment.
    pub fn rows(self) -> usize {
        self.rows
    }
}

/// Capability for one exact, possibly multi-page append reservation.
///
/// Obtained from [`SequenceCache::reserve_append`] and consumed by exactly one
/// [`SequenceCache::commit_append`] or [`SequenceCache::abort_append`]. The
/// embedded nonce makes copied or replayed reservations fail with
/// [`CacheError::AppendReservationMismatch`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppendReservation {
    sequence: SequenceId,
    start_position: usize,
    rows: usize,
    segments: Box<[AppendSegment]>,
    nonce: u64,
}

impl AppendReservation {
    /// Sequence the append belongs to.
    pub fn sequence(&self) -> SequenceId {
        self.sequence
    }

    /// Committed sequence position immediately before this append.
    pub fn start_position(&self) -> usize {
        self.start_position
    }

    /// Exact number of writable rows reserved. Commit may retain a nonempty
    /// prefix and releases the unused suffix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Ordered physical-page segments covering the complete append.
    pub fn segments(&self) -> &[AppendSegment] {
        &self.segments
    }
}

/// One physical page borrowed for a reservation segment.
pub struct AppendPage<'a, P> {
    segment: AppendSegment,
    page: &'a P,
}

/// Allocation-free ordered view of every page in an append reservation.
pub struct AppendPages<'a, P> {
    segments: &'a [AppendSegment],
    slots: &'a [Slot<PageRecord<P>>],
}

impl<'a, P> AppendPages<'a, P> {
    /// Number of writable physical segments in this reservation.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the reservation contains no writable segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Iterates over writable pages in logical token order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AppendPage<'a, P>> + '_ {
        self.segments.iter().map(|segment| {
            let page = self
                .slots
                .get(segment.page.slot())
                .filter(|slot| slot.generation == segment.page.generation)
                .and_then(|slot| slot.value.as_ref())
                .and_then(|record| record.physical.as_ref())
                .expect("validated append page remains present");
            AppendPage {
                segment: *segment,
                page,
            }
        })
    }
}

impl<'a, P> AppendPage<'a, P> {
    /// Logical and row-range description for this physical page.
    pub fn segment(&self) -> AppendSegment {
        self.segment
    }

    /// Backend page backing the segment.
    pub fn page(&self) -> &'a P {
        self.page
    }
}

/// Borrowed logical page ordering for an attention operation.
///
/// Pages are listed in token order; [`PageTableView::position`] gives the total
/// valid tokens across them. Only the final page may be partially valid, and
/// only while it is the sequence's private writable tail.
pub struct PageTableView<'a> {
    pages: &'a [PageId],
    position: usize,
    page_tokens: usize,
}

impl PageTableView<'_> {
    /// Logical pages in token order.
    pub fn pages(&self) -> &[PageId] {
        self.pages
    }

    /// Total valid token position published with this table.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Tokens per page for interpreting the table.
    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }
}

/// Exact synchronous state owned by one manager.
///
/// Recomputed from ownership records after every mutation
/// ([`SequenceCache::stats`]) and mirrored to the exported gauges, so optimizing
/// telemetry against ownership is checked by [`SequenceCache::validate`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    /// Live sequences admitted but not yet finished.
    pub active_sequences: usize,
    /// Currently retained prefix entries.
    pub retained_prefix_entries: usize,
    /// Interned page-sized token blocks currently referenced by prefix entries.
    pub interned_token_blocks: usize,
    /// Physical pages occupying storage, including deferred retirements.
    pub resident_pages: usize,
    /// Pages a subsequent reserve could still commit, subject to the byte
    /// budget and any backend slot cap.
    pub free_pages: usize,
    /// Future pages promised to admitted sequences but not yet allocated.
    pub reserved_pages: usize,
    /// Retired pages awaiting asynchronous backend reclamation.
    pub deferred_retirement_pages: usize,
    /// Bytes occupied by `resident_pages`; shared pages are counted once.
    pub unique_resident_page_bytes: usize,
    /// Bytes promised to admitted sequences for pages not yet allocated.
    pub outstanding_reservation_bytes: usize,
    /// Per-sequence private state bytes across active sequences.
    pub active_private_state_bytes: usize,
    /// Snapshot bytes held by retained prefix entries.
    pub retained_snapshot_bytes: usize,
    /// Page-table bytes across active sequences.
    pub page_table_bytes: usize,
    /// Bytes held solely by prefix entries and immediately reclaimable by
    /// eviction.
    pub reclaimable_prefix_only_bytes: usize,
    /// Sum of every managed byte class; admission keeps this within budget.
    pub total_managed_bytes: usize,
}

/// One arena slot: the value currently occupying it plus the generation the
/// next handle minted from this slot must carry.
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Ownership state of one logical page.
struct PageRecord<P> {
    /// Backend-owned storage. `None` marks a record being retired
    /// transactionally; readers treat it as stale.
    physical: Option<P>,
    /// Live sequences whose page tables include this page.
    active_refs: usize,
    /// Retained prefix entries which share this page.
    prefix_refs: usize,
    /// Committed tokens in this page; equals `page_tokens` once sealed.
    valid_tokens: usize,
    /// Whether the page is complete and immutable, hence safe to share.
    sealed: bool,
}

/// In-flight exact append spanning one or more pages.
#[derive(Clone)]
struct PendingAppend {
    start_position: usize,
    rows: usize,
    segments: Box<[AppendSegment]>,
    new_pages: Box<[PageId]>,
    nonce: u64,
}

/// Ownership state of one admitted sequence.
struct SequenceRecord {
    /// Logical pages in token order.
    pages: Vec<PageId>,
    /// Committed token position.
    position: usize,
    /// Admitted maximum token position.
    max_position: usize,
    /// Pages still promised for growth toward `max_position`.
    reserved_pages: usize,
    /// Exact declared private-state bytes.
    private_state_bytes: usize,
    /// Exact declared page-table bytes.
    page_table_bytes: usize,
    /// Pending append, if any.
    pending: Option<PendingAppend>,
}

/// One retained prefix checkpoint.
struct PrefixEntry<S> {
    /// ART key over the interned token block IDs.
    key: PrefixKey,
    /// Blocks contributing to the key, held for refcounted release.
    blocks: Vec<TokenBlockId>,
    /// Sealed shared pages backing the entry, in token order.
    pages: Vec<PageId>,
    /// Page-aligned token position covered by the entry.
    position: usize,
    /// Runtime-defined immutable snapshot stored at the prefix end.
    snapshot: S,
    /// Managed bytes attributed to the snapshot.
    snapshot_bytes: usize,
    /// Logical clock of the last lookup or retention, for LRU eviction.
    last_used: u64,
}

/// Single-owner logical sequence and reusable-prefix manager.
///
/// The cache owns page lifetimes, prefix indexing, admission reservations, and
/// exact byte accounting for one model runtime. It is generic over:
///
/// - `B` — a [`PageBackend`] supplying transactional physical storage; and
/// - `S` — a [`RetainedSnapshot`] stored at retained prefix endpoints.
///
/// All mutation is serialized through `&mut self`, so no interior locking is
/// required on either the manager or the backend. The cache is deliberately
/// `!Sync`; callers which share it across tasks must serialize access
/// themselves.
pub struct SequenceCache<B: PageBackend, S: RetainedSnapshot> {
    config: CacheConfig,
    page_bytes: usize,
    page_capacity: usize,
    backend: B,
    metrics: CacheMetrics,
    index: PrefixIndex,
    pages: Vec<Slot<PageRecord<B::Page>>>,
    free_page_slots: Vec<usize>,
    sequences: Vec<Slot<SequenceRecord>>,
    free_sequence_slots: Vec<usize>,
    prefixes: BTreeMap<PrefixEntryId, PrefixEntry<S>>,
    next_prefix_id: u64,
    clock: u64,
    append_nonce: u64,
    stats: CacheStats,
    deferred_pages: usize,
    not_sync: PhantomData<Cell<()>>,
}

impl<B: PageBackend, S: RetainedSnapshot> SequenceCache<B, S> {
    /// Create a cache over `backend` with default single-shard metrics.
    pub fn new(config: CacheConfig, backend: B) -> Result<Self, B::Error> {
        Self::with_metrics(config, backend, CacheMetrics::default())
    }

    /// Create a cache exporting into a pre-built [`CacheMetrics`] set.
    ///
    /// Supplying the metric set lets a runtime choose the shard count and reuse
    /// one registry across several caches.
    pub fn with_metrics(
        config: CacheConfig,
        backend: B,
        metrics: CacheMetrics,
    ) -> Result<Self, B::Error> {
        let page_bytes = backend.page_bytes();
        let page_capacity = backend.page_capacity().unwrap_or(usize::MAX);
        validate_config(config, page_bytes)?;
        let mut cache = Self {
            config,
            page_bytes,
            page_capacity,
            backend,
            metrics,
            index: PrefixIndex::new(),
            pages: Vec::new(),
            free_page_slots: Vec::new(),
            sequences: Vec::new(),
            free_sequence_slots: Vec::new(),
            prefixes: BTreeMap::new(),
            next_prefix_id: 0,
            clock: 0,
            append_nonce: 0,
            stats: CacheStats::default(),
            deferred_pages: 0,
            not_sync: PhantomData,
        };
        cache.refresh_derived_stats()?;
        Ok(cache)
    }

    /// Immutable geometry and byte limits the cache was built with.
    pub fn config(&self) -> CacheConfig {
        self.config
    }

    /// The storage backend, for read-only runtime queries.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Perform a short backend configuration or diagnostic operation.
    ///
    /// The closure runs synchronously while no other cache operation is in
    /// flight. It must not mutate page storage or tables the manager currently
    /// owns.
    pub fn with_backend<R>(&mut self, operation: impl FnOnce(&mut B) -> R) -> R {
        operation(&mut self.backend)
    }

    /// Exported counters, gauges, and latency histograms for this cache.
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    /// Exact ownership snapshot; identical to the exported gauges.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Default checkpoint position, preserving the final prompt token for decode.
    ///
    /// The result is the largest page-aligned position strictly below
    /// `prompt_tokens`, since the final prompt token has not been processed when
    /// a prompt is checkpointed before decode.
    pub fn cacheable_prefix_tokens(&self, prompt_tokens: usize) -> usize {
        prompt_tokens.saturating_sub(1) / self.config.page_tokens * self.config.page_tokens
    }

    /// Find the longest retained prefix covering the start of `tokens`.
    ///
    /// The query is truncated to [`SequenceCache::cacheable_prefix_tokens`]
    /// before matching. A miss never interns token blocks, so speculative
    /// lookups cannot grow the index. A hit refreshes the entry's LRU clock.
    pub fn lookup_prefix(&mut self, tokens: &[u32]) -> Option<PrefixMatch> {
        let started = Instant::now();
        self.metrics.prefix_lookups.inc();
        let position = self.cacheable_prefix_tokens(tokens.len());
        let found = if position == 0 {
            None
        } else {
            self.index
                .lookup_key(&tokens[..position], self.config.page_tokens)
                .and_then(|key| self.index.longest(&key))
        };

        let result = found.and_then(|entry_id| {
            let clock = self.tick();
            let entry = self.prefixes.get_mut(&entry_id)?;
            entry.last_used = clock;
            Some(PrefixMatch {
                entry: entry_id,
                position: entry.position,
                page_count: entry.pages.len(),
            })
        });
        if let Some(hit) = result {
            self.metrics.prefix_hits.inc();
            self.metrics
                .prefix_restored_tokens
                .add(hit.position.min(isize::MAX as usize) as isize);
        } else {
            self.metrics.prefix_misses.inc();
        }
        self.metrics.lookup_us.record(elapsed_us(started));
        result
    }

    /// Returns whether an exact aligned token prefix is already retained.
    pub fn contains_prefix(&self, tokens: &[u32], position: usize) -> bool {
        position != 0
            && position.is_multiple_of(self.config.page_tokens)
            && tokens.len() >= position
            && self
                .index
                .exact(&tokens[..position], self.config.page_tokens)
                .is_some()
    }

    /// Admit a sequence, optionally sharing a previously matched aligned prefix.
    ///
    /// Admission is strict: the request reserves every page needed to reach
    /// [`AdmissionRequest::max_position`], so an admitted sequence can always
    /// grow to its full length. When a [`PrefixMatch`] is supplied, its sealed
    /// pages are shared without copying and the sequence starts past the prefix.
    ///
    /// The `restore` callback receives the retained immutable snapshot and the
    /// aligned position before any manager ownership is committed, letting the
    /// runtime rebuild non-paged state. Its failure leaves cache metadata
    /// unchanged. When capacity cannot be guaranteed the call returns
    /// [`AdmissionOutcome::WouldBlock`] rather than an error.
    pub fn admit<F>(
        &mut self,
        prefix: Option<PrefixMatch>,
        request: AdmissionRequest,
        context: &mut B::Context<'_>,
        restore: F,
    ) -> Result<AdmissionOutcome, B::Error>
    where
        F: FnOnce(Option<&S>, usize) -> core::result::Result<(), B::Error>,
    {
        let started = Instant::now();
        self.reclaim_deferred(context)?;
        let (prefix_id, position, shared_pages) = if let Some(prefix_match) = prefix {
            let entry = self
                .prefixes
                .get(&prefix_match.entry)
                .ok_or(CacheError::StalePrefix)?;
            if entry.position != prefix_match.position
                || entry.pages.len() != prefix_match.page_count
            {
                return Err(CacheError::StalePrefix);
            }
            (
                Some(prefix_match.entry),
                entry.position,
                entry.pages.clone(),
            )
        } else {
            (None, 0, Vec::new())
        };
        if request.max_position < position {
            return Err(CacheError::InvalidPosition);
        }
        let total_pages = div_ceil(request.max_position, self.config.page_tokens)?;
        let reserved_pages = total_pages
            .checked_sub(shared_pages.len())
            .ok_or(CacheError::Invariant("prefix exceeds maximum position"))?;
        let extra = self.admission_bytes(reserved_pages, request)?;
        let limit = self.admission_limit(request.allow_emergency)?;
        let Some(evictions) =
            self.plan_evictions(extra, reserved_pages, limit, prefix_id, None, None)?
        else {
            self.metrics.admission_would_block.inc();
            self.metrics.admission_us.record(elapsed_us(started));
            return Ok(AdmissionOutcome::WouldBlock);
        };
        self.prepare_sequence_slot()?;

        let snapshot = prefix_id.map(|id| &self.prefixes[&id].snapshot);
        let restore_started = Instant::now();
        if let Err(error) = restore(snapshot, position) {
            self.metrics.backend_failures.inc();
            return Err(CacheError::Backend(error));
        }
        self.metrics.restore_us.record(elapsed_us(restore_started));

        let (backend, pages) = (&mut self.backend, &self.pages);
        let page_refs = physical_refs_from::<B>(pages, &shared_pages)?;
        if let Err(error) = backend.update_page_table(&page_refs, position, context) {
            self.metrics.backend_failures.inc();
            return Err(CacheError::Backend(error));
        }

        self.commit_evictions(&evictions, context)?;
        for page in &shared_pages {
            self.page_record_mut(*page)?.active_refs = self
                .page_record(*page)?
                .active_refs
                .checked_add(1)
                .ok_or(CacheError::ArithmeticOverflow)?;
        }
        let id = self.insert_sequence(SequenceRecord {
            pages: shared_pages,
            position,
            max_position: request.max_position,
            reserved_pages,
            private_state_bytes: request.private_state_bytes,
            page_table_bytes: request.page_table_bytes,
            pending: None,
        })?;
        self.metrics.admission_successes.inc();
        self.refresh_stats()?;
        self.metrics.admission_us.record(elapsed_us(started));
        Ok(AdmissionOutcome::Admitted(id))
    }

    /// Reserve an exact append, allocating every physical page it spans.
    ///
    /// The returned reservation covers the requested rows without regard to
    /// page boundaries. Its complete page table is published before this
    /// method returns, allowing one model operation to write and attend through
    /// every segment. Only one append may be pending per sequence.
    pub fn reserve_append(
        &mut self,
        sequence: SequenceId,
        rows: usize,
        context: &mut B::Context<'_>,
    ) -> Result<AppendReservation, B::Error> {
        if rows == 0 {
            return Err(CacheError::InvalidPosition);
        }
        let (position, max_position, old_page_count, tail, reserved_pages, pending) = {
            let record = self.sequence_record(sequence)?;
            (
                record.position,
                record.max_position,
                record.pages.len(),
                record.pages.last().copied(),
                record.reserved_pages,
                record.pending.is_some(),
            )
        };
        if pending {
            return Err(CacheError::AppendPending);
        }
        let new_position = position
            .checked_add(rows)
            .ok_or(CacheError::ArithmeticOverflow)?;
        if new_position > max_position {
            return Err(CacheError::InvalidPosition);
        }
        if let Some(tail) = tail.filter(|_| !position.is_multiple_of(self.config.page_tokens)) {
            let record = self.page_record(tail)?;
            if record.sealed || record.active_refs != 1 || record.prefix_refs != 0 {
                return Err(CacheError::Invariant("writable tail is not private"));
            }
        }

        let required_pages = div_ceil(new_position, self.config.page_tokens)?;
        let new_page_count = required_pages
            .checked_sub(old_page_count)
            .ok_or(CacheError::Invariant("append page count moved backwards"))?;
        if new_page_count > reserved_pages {
            return Err(CacheError::Invariant("append exceeds admitted reservation"));
        }
        self.prepare_page_slots(new_page_count)?;
        let nonce = self.next_append_nonce()?;

        let mut allocations = Vec::with_capacity(new_page_count);
        for _ in 0..new_page_count {
            match self.backend.allocate_page(context) {
                Ok(allocation) => allocations.push(allocation),
                Err(error) => {
                    for allocation in allocations.drain(..).rev() {
                        self.backend.rollback_page(allocation.page, context);
                    }
                    self.metrics.backend_failures.inc();
                    return Err(CacheError::Backend(error));
                }
            }
        }
        if !allocations.is_empty() {
            let old_pages = self.sequence_record(sequence)?.pages.clone();
            let (backend, page_slots) = (&mut self.backend, &self.pages);
            let mut table = physical_refs_from::<B>(page_slots, &old_pages)?;
            table.extend(allocations.iter().map(|allocation| &allocation.page));
            if let Err(error) = backend.update_page_table(&table, position, context) {
                for allocation in allocations.drain(..).rev() {
                    backend.rollback_page(allocation.page, context);
                }
                self.metrics.backend_failures.inc();
                return Err(CacheError::Backend(error));
            }
        }

        let mut new_pages = Vec::with_capacity(new_page_count);
        for allocation in allocations {
            let page = self.insert_page_prepared(PageRecord {
                physical: Some(allocation.page),
                active_refs: 1,
                prefix_refs: 0,
                valid_tokens: 0,
                sealed: false,
            });
            new_pages.push(page);
            if allocation.recycled {
                self.metrics.pages_recycled.inc();
            } else {
                self.metrics.pages_allocated.inc();
            }
        }
        let mut segments =
            Vec::with_capacity(rows.div_ceil(self.config.page_tokens).saturating_add(1));
        let mut input_offset = 0;
        let mut logical_position = position;
        while input_offset < rows {
            let logical_page = logical_position / self.config.page_tokens;
            let page_offset = logical_position % self.config.page_tokens;
            let segment_rows = (rows - input_offset).min(self.config.page_tokens - page_offset);
            let page = if logical_page < old_page_count {
                *self
                    .sequence_record(sequence)?
                    .pages
                    .get(logical_page)
                    .ok_or(CacheError::Invariant(
                        "existing page missing from append span",
                    ))?
            } else {
                *new_pages
                    .get(logical_page - old_page_count)
                    .ok_or(CacheError::Invariant("new page missing from append span"))?
            };
            segments.push(AppendSegment {
                page,
                page_offset,
                input_offset,
                rows: segment_rows,
            });
            input_offset += segment_rows;
            logical_position += segment_rows;
        }
        let segments = segments.into_boxed_slice();
        let pending = PendingAppend {
            start_position: position,
            rows,
            segments: segments.clone(),
            new_pages: new_pages.clone().into_boxed_slice(),
            nonce,
        };
        let record = self.sequence_record_mut(sequence)?;
        record.reserved_pages -= new_page_count;
        record.pending = Some(pending);
        self.refresh_stats()?;
        Ok(AppendReservation {
            sequence,
            start_position: position,
            rows,
            segments,
            nonce,
        })
    }

    /// Borrow every physical page covered by a pending reservation.
    ///
    /// The pages are ordered exactly like the reservation segments. Runtime
    /// kernels may scatter one compute chunk directly into these pages.
    pub fn with_append_pages<R, F>(
        &mut self,
        reservation: &AppendReservation,
        operation: F,
    ) -> Result<R, B::Error>
    where
        F: FnOnce(&mut B, AppendPages<'_, B::Page>) -> core::result::Result<R, B::Error>,
    {
        self.validate_reservation(reservation)?;
        let (backend, slots) = (&mut self.backend, &self.pages);
        let pages = AppendPages {
            segments: &reservation.segments,
            slots,
        };
        operation(backend, pages).map_err(|error| {
            self.metrics.backend_failures.inc();
            CacheError::Backend(error)
        })
    }

    /// Commit a nonempty prefix of the span described by a reservation.
    ///
    /// Pages used only by the uncommitted suffix are returned to the backend
    /// and restored to the sequence's admission reservation. Rows beyond the
    /// committed prefix in its final kept page remain invalid and writable.
    /// A backend failure leaves the prior logical position intact and the
    /// reservation pending.
    pub fn commit_append(
        &mut self,
        reservation: AppendReservation,
        committed_rows: usize,
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        self.validate_reservation(&reservation)?;
        if committed_rows == 0 || committed_rows > reservation.rows {
            return Err(CacheError::InvalidPosition);
        }
        let (old_pages, new_pages) = {
            let sequence = self.sequence_record(reservation.sequence)?;
            let pending = sequence
                .pending
                .as_ref()
                .ok_or(CacheError::NoAppendPending)?;
            (sequence.pages.clone(), pending.new_pages.to_vec())
        };
        let new_position = reservation
            .start_position
            .checked_add(committed_rows)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let committed_page_count = div_ceil(new_position, self.config.page_tokens)?;
        let kept_new_page_count =
            committed_page_count
                .checked_sub(old_pages.len())
                .ok_or(CacheError::Invariant(
                    "partial commit page count moved backwards",
                ))?;
        let (kept_new_pages, released_pages) = new_pages.split_at(kept_new_page_count);
        let mut committed_pages = old_pages;
        committed_pages.extend_from_slice(kept_new_pages);
        let committed_segments = reservation
            .segments
            .iter()
            .copied()
            .take_while(|segment| segment.input_offset < committed_rows)
            .map(|mut segment| {
                segment.rows = segment.rows.min(committed_rows - segment.input_offset);
                segment
            })
            .collect::<Vec<_>>();
        let mut sealed_ids = Vec::new();
        for segment in committed_segments
            .iter()
            .filter(|segment| segment.page_offset + segment.rows == self.config.page_tokens)
        {
            if !self.page_record(segment.page)?.sealed {
                sealed_ids.push(segment.page);
            }
        }
        {
            let (backend, slots) = (&mut self.backend, &self.pages);
            let committed = physical_refs_from::<B>(slots, &committed_pages)?;
            let sealed = physical_refs_from::<B>(slots, &sealed_ids)?;
            let released = physical_refs_from::<B>(slots, released_pages)?;
            if let Err(error) =
                backend.commit_append(&committed, &sealed, &released, new_position, context)
            {
                self.metrics.backend_failures.inc();
                return Err(CacheError::Backend(error));
            }
        }
        let page_tokens = self.config.page_tokens;
        for segment in committed_segments {
            let record = self.page_record_mut(segment.page)?;
            record.valid_tokens = segment.page_offset + segment.rows;
            record.sealed = record.valid_tokens == page_tokens;
        }
        for page in released_pages {
            self.page_record_mut(*page)?
                .physical
                .take()
                .ok_or(CacheError::StalePage)?;
            self.remove_page(*page)?;
        }
        let sequence = self.sequence_record_mut(reservation.sequence)?;
        sequence.pages.extend_from_slice(kept_new_pages);
        sequence.reserved_pages = sequence
            .reserved_pages
            .checked_add(released_pages.len())
            .ok_or(CacheError::ArithmeticOverflow)?;
        sequence.position = new_position;
        sequence.pending = None;
        self.metrics
            .pages_sealed
            .add(sealed_ids.len().min(isize::MAX as usize) as isize);
        self.refresh_stats()?;
        Ok(())
    }

    /// Abort a reservation and restore the exact prior page table.
    ///
    /// If restoring the backend table fails, the reservation remains pending
    /// and may be retried. On success every newly allocated page is returned.
    pub fn abort_append(
        &mut self,
        reservation: AppendReservation,
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        self.validate_reservation(&reservation)?;
        let (old_pages, new_pages, old_position) = {
            let sequence = self.sequence_record(reservation.sequence)?;
            let pending = sequence
                .pending
                .as_ref()
                .ok_or(CacheError::NoAppendPending)?;
            (
                sequence.pages.clone(),
                pending.new_pages.to_vec(),
                sequence.position,
            )
        };
        if !new_pages.is_empty() {
            let (backend, slots) = (&mut self.backend, &self.pages);
            let table = physical_refs_from::<B>(slots, &old_pages)?;
            if let Err(error) = backend.update_page_table(&table, old_position, context) {
                self.metrics.backend_failures.inc();
                return Err(CacheError::Backend(error));
            }
        }
        {
            let (backend, slots) = (&mut self.backend, &self.pages);
            let physical_pages = physical_refs_from::<B>(slots, &new_pages)?;
            if let Err(error) = backend.abort_append(&physical_pages, context) {
                self.metrics.backend_failures.inc();
                return Err(CacheError::Backend(error));
            }
        }
        for page in &new_pages {
            self.page_record_mut(*page)?
                .physical
                .take()
                .ok_or(CacheError::StalePage)?;
            self.remove_page(*page)?;
        }
        let sequence = self.sequence_record_mut(reservation.sequence)?;
        sequence.reserved_pages = sequence
            .reserved_pages
            .checked_add(new_pages.len())
            .ok_or(CacheError::ArithmeticOverflow)?;
        sequence.pending = None;
        self.refresh_stats()?;
        Ok(())
    }

    /// Borrow the sequence's logical page table and committed position.
    pub fn page_table(&self, sequence: SequenceId) -> Result<PageTableView<'_>, B::Error> {
        let sequence = self.sequence_record(sequence)?;
        Ok(PageTableView {
            pages: &sequence.pages,
            position: sequence.position,
            page_tokens: self.config.page_tokens,
        })
    }

    /// Resolve a page handle to its backend storage.
    pub fn page(&self, page: PageId) -> Result<&B::Page, B::Error> {
        self.page_record(page)?
            .physical
            .as_ref()
            .ok_or(CacheError::StalePage)
    }

    /// Retain the sequence's current aligned pages without copying KV storage.
    ///
    /// The sequence must end exactly on a page boundary with all pages sealed;
    /// the retained entry shares those pages, its interned token blocks become
    /// the index key, and `snapshot` records any extra model state a future
    /// admission needs to resume past the prefix. Already-retained prefixes
    /// report [`RetainOutcome::Duplicate`] and refresh the existing entry's LRU
    /// clock instead of evicting a competitor.
    ///
    /// When capacity is tight the manager first evicts colder prefix entries,
    /// including ones sharing pages with active sequences; only entry metadata
    /// and unshared pages are reclaimed.
    pub fn retain_prefix(
        &mut self,
        sequence: SequenceId,
        tokens: &[u32],
        snapshot: S,
        context: &mut B::Context<'_>,
    ) -> Result<RetainOutcome, B::Error> {
        let started = Instant::now();
        let (position, pages) = {
            let sequence = self.sequence_record(sequence)?;
            (sequence.position, sequence.pages.clone())
        };
        if position == 0
            || !position.is_multiple_of(self.config.page_tokens)
            || tokens.len() < position
        {
            return Err(CacheError::InvalidTokenPrefix);
        }
        for page in &pages {
            let page = self.page_record(*page)?;
            if !page.sealed || page.valid_tokens != self.config.page_tokens {
                return Err(CacheError::Invariant("prefix contains an unsealed page"));
            }
        }
        if let Some(existing) = self
            .index
            .exact(&tokens[..position], self.config.page_tokens)
        {
            let clock = self.tick();
            self.prefixes
                .get_mut(&existing)
                .ok_or(CacheError::StalePrefix)?
                .last_used = clock;
            self.metrics.prefix_duplicate_insertions.inc();
            self.metrics.insertion_us.record(elapsed_us(started));
            return Ok(RetainOutcome::Duplicate(existing));
        }

        let snapshot_bytes = snapshot.retained_bytes();
        if snapshot_bytes > self.config.max_snapshot_bytes {
            return Err(CacheError::SnapshotCapacity);
        }
        let next_snapshot_bytes = self
            .stats
            .retained_snapshot_bytes
            .checked_add(snapshot_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let entry_count = self
            .prefixes
            .len()
            .checked_add(1)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let prepared = self
            .index
            .prepare_key(&tokens[..position], self.config.page_tokens)?;
        let extra = snapshot_bytes;
        let evictions = match self.plan_evictions(
            extra,
            0,
            self.config.max_managed_bytes,
            None,
            Some(next_snapshot_bytes),
            Some(entry_count),
        )? {
            Some(plan) => plan,
            None => {
                self.index.rollback_key(prepared);
                return if next_snapshot_bytes > self.config.max_snapshot_bytes {
                    Err(CacheError::SnapshotCapacity)
                } else {
                    Err(CacheError::PrefixCapacity)
                };
            }
        };
        self.prepare_prefix_id()?;
        if let Err(error) = self.commit_evictions(&evictions, context) {
            self.index.rollback_key(prepared);
            return Err(error);
        }

        let id = PrefixEntryId(self.next_prefix_id);
        self.next_prefix_id = self
            .next_prefix_id
            .checked_add(1)
            .ok_or(CacheError::IdExhausted("prefix entry"))?;
        for page in &pages {
            let refs = self.page_record(*page)?.prefix_refs;
            self.page_record_mut(*page)?.prefix_refs =
                refs.checked_add(1).ok_or(CacheError::ArithmeticOverflow)?;
        }
        self.index.commit_key(&prepared, id);
        let clock = self.tick();
        self.prefixes.insert(
            id,
            PrefixEntry {
                key: prepared.key,
                blocks: prepared.blocks,
                pages,
                position,
                snapshot,
                snapshot_bytes,
                last_used: clock,
            },
        );
        self.metrics.prefix_insertions.inc();
        self.refresh_stats()?;
        self.metrics.insertion_us.record(elapsed_us(started));
        Ok(RetainOutcome::Inserted(id))
    }

    /// Drop one retained prefix entry.
    ///
    /// Pages also referenced by live sequences stay resident; pages owned only
    /// by the entry are retired through the backend. Any outstanding
    /// [`PrefixMatch`] against the entry becomes stale.
    pub fn evict_prefix(
        &mut self,
        entry: PrefixEntryId,
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        if !self.prefixes.contains_key(&entry) {
            return Err(CacheError::StalePrefix);
        }
        self.commit_evictions(&[entry], context)?;
        self.refresh_stats()?;
        Ok(())
    }

    /// Branch an unaligned live sequence, sharing sealed pages and copying one tail.
    ///
    /// The new sequence starts at the source's current position. Its complete
    /// pages are shared with the source; its writable tail is a private
    /// copy-on-write duplicate, after which the branch is independent. The
    /// source must not have a pending append.
    pub fn branch(
        &mut self,
        source: SequenceId,
        request: AdmissionRequest,
        context: &mut B::Context<'_>,
    ) -> Result<AdmissionOutcome, B::Error> {
        let (position, source_pages) = {
            let source = self.sequence_record(source)?;
            if source.pending.is_some() {
                return Err(CacheError::AppendPending);
            }
            (source.position, source.pages.clone())
        };
        if position == 0 || position.is_multiple_of(self.config.page_tokens) {
            return Err(CacheError::InvalidPosition);
        }
        if request.max_position < position {
            return Err(CacheError::InvalidPosition);
        }
        let complete_count = position / self.config.page_tokens;
        let shared_pages = source_pages[..complete_count].to_vec();
        let source_tail = *source_pages
            .get(complete_count)
            .ok_or(CacheError::Invariant("unaligned source has no tail"))?;
        let total_pages = div_ceil(request.max_position, self.config.page_tokens)?;
        let reserved_pages = total_pages
            .checked_sub(complete_count + 1)
            .ok_or(CacheError::Invariant("branch exceeds admitted pages"))?;
        let page_commitment = reserved_pages
            .checked_add(1)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let extra = self.admission_bytes(page_commitment, request)?;
        let limit = self.admission_limit(request.allow_emergency)?;
        let Some(evictions) =
            self.plan_evictions(extra, page_commitment, limit, None, None, None)?
        else {
            self.metrics.admission_would_block.inc();
            return Ok(AdmissionOutcome::WouldBlock);
        };
        self.prepare_sequence_slot()?;
        self.prepare_page_slot()?;
        let copied_id = self.peek_page_id()?;
        let (backend, page_slots) = (&mut self.backend, &self.pages);
        let source_physical = page_record_from::<B>(page_slots, source_tail)?
            .physical
            .as_ref()
            .ok_or(CacheError::StalePage)?;
        let allocation = match backend.copy_partial_page(
            source_physical,
            position % self.config.page_tokens,
            context,
        ) {
            Ok(allocation) => allocation,
            Err(error) => {
                self.metrics.backend_failures.inc();
                return Err(CacheError::Backend(error));
            }
        };
        let copied = allocation.page;
        let mut table = physical_refs_from::<B>(page_slots, &shared_pages)?;
        table.push(&copied);
        if let Err(error) = backend.update_page_table(&table, position, context) {
            backend.rollback_page(copied, context);
            self.metrics.backend_failures.inc();
            return Err(CacheError::Backend(error));
        }
        self.commit_evictions(&evictions, context)?;
        for page in &shared_pages {
            let refs = self.page_record(*page)?.active_refs;
            self.page_record_mut(*page)?.active_refs =
                refs.checked_add(1).ok_or(CacheError::ArithmeticOverflow)?;
        }
        let copied_id_actual = self.insert_page(PageRecord {
            physical: Some(copied),
            active_refs: 1,
            prefix_refs: 0,
            valid_tokens: position % self.config.page_tokens,
            sealed: false,
        })?;
        debug_assert_eq!(copied_id, copied_id_actual);
        let mut pages = shared_pages;
        pages.push(copied_id_actual);
        let id = self.insert_sequence(SequenceRecord {
            pages,
            position,
            max_position: request.max_position,
            reserved_pages,
            private_state_bytes: request.private_state_bytes,
            page_table_bytes: request.page_table_bytes,
            pending: None,
        })?;
        if allocation.recycled {
            self.metrics.pages_recycled.inc();
        } else {
            self.metrics.pages_allocated.inc();
        }
        self.metrics.pages_copied_on_write.inc();
        self.metrics.admission_successes.inc();
        self.refresh_stats()?;
        Ok(AdmissionOutcome::Admitted(id))
    }

    /// Finish or cancel a sequence and release reservations and active page refs.
    ///
    /// Pages the sequence shares with retained prefixes or other live sequences
    /// remain resident; unshared pages and unused reservations are freed. A
    /// sequence with a pending append must finish that append first, either by
    /// commit or abort.
    pub fn finish(
        &mut self,
        sequence: SequenceId,
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        let record = self.sequence_record(sequence)?;
        if record.pending.is_some() {
            return Err(CacheError::AppendPending);
        }
        let pages = record.pages.clone();
        self.prepare_remove_sequence(sequence)?;
        let mut retire_ids = Vec::new();
        for page in &pages {
            let record = self.page_record(*page)?;
            if record.active_refs == 1 && record.prefix_refs == 0 {
                retire_ids.push(*page);
            }
        }
        self.retire_page_ids(&retire_ids, context)?;
        for page in &pages {
            if retire_ids.contains(page) {
                continue;
            }
            let refs = self.page_record(*page)?.active_refs;
            self.page_record_mut(*page)?.active_refs = refs
                .checked_sub(1)
                .ok_or(CacheError::Invariant("missing active page reference"))?;
        }
        self.remove_sequence(sequence)?;
        self.refresh_stats()?;
        Ok(())
    }

    /// Poll backend synchronization and release completed deferred retirements.
    ///
    /// Deferred pages remain charged against capacity until reclaimed, so
    /// runtimes with asynchronous retirement should call this periodically —
    /// admission does so automatically. Returns the number of pages reclaimed.
    pub fn reclaim_deferred(&mut self, context: &mut B::Context<'_>) -> Result<usize, B::Error> {
        let reclaimed = self.backend.poll_reclaimed(context).map_err(|error| {
            self.metrics.backend_failures.inc();
            CacheError::Backend(error)
        })?;
        if reclaimed > self.deferred_pages {
            return Err(CacheError::Invariant(
                "backend reclaimed more pages than were deferred",
            ));
        }
        self.deferred_pages -= reclaimed;
        self.refresh_stats()?;
        Ok(reclaimed)
    }

    /// Recompute references and byte totals from first principles.
    ///
    /// Checks every ownership invariant — reference counts, alignment, sealing,
    /// exact stats, and gauge agreement — without mutating state. Intended for
    /// tests and debug-enabled health checks, not the hot path.
    pub fn validate(&self) -> Result<(), B::Error> {
        let mut active_refs: HashMap<PageId, usize> = HashMap::new();
        for slot in &self.sequences {
            let Some(sequence) = &slot.value else {
                continue;
            };
            if sequence.position > sequence.max_position {
                return Err(CacheError::Invariant("sequence exceeds maximum position"));
            }
            let expected_pages = div_ceil(sequence.position, self.config.page_tokens)?;
            let has_preallocated_tail = sequence.position.is_multiple_of(self.config.page_tokens)
                && sequence.pages.len() == expected_pages + 1
                && sequence
                    .pages
                    .last()
                    .map(|page| {
                        self.page_record(*page)
                            .map(|record| record.valid_tokens == 0 && !record.sealed)
                            .unwrap_or(false)
                    })
                    .unwrap_or(false);
            if sequence.pages.len() != expected_pages && !has_preallocated_tail {
                return Err(CacheError::Invariant(
                    "sequence position disagrees with pages",
                ));
            }
            for page in &sequence.pages {
                *active_refs.entry(*page).or_default() += 1;
            }
            if let Some(pending) = &sequence.pending {
                for page in pending.new_pages.iter().copied() {
                    *active_refs.entry(page).or_default() += 1;
                    let record = self.page_record(page)?;
                    if record.valid_tokens != 0
                        || record.sealed
                        || record.active_refs != 1
                        || record.prefix_refs != 0
                    {
                        return Err(CacheError::Invariant(
                            "pending append page is not private and empty",
                        ));
                    }
                }
            }
            if let Some(tail) = sequence.pages.last().filter(|_| !has_preallocated_tail) {
                let tail = self.page_record(*tail)?;
                let expected = if sequence.position.is_multiple_of(self.config.page_tokens) {
                    self.config.page_tokens
                } else {
                    sequence.position % self.config.page_tokens
                };
                if tail.valid_tokens != expected {
                    return Err(CacheError::Invariant(
                        "tail valid rows disagree with position",
                    ));
                }
                if expected < self.config.page_tokens
                    && (tail.sealed || tail.active_refs != 1 || tail.prefix_refs != 0)
                {
                    return Err(CacheError::Invariant("writable tail is shared"));
                }
            }
            let max_pages = div_ceil(sequence.max_position, self.config.page_tokens)?;
            let pending_pages = sequence
                .pending
                .as_ref()
                .map_or(0, |pending| pending.new_pages.len());
            if sequence.pages.len() + pending_pages + sequence.reserved_pages != max_pages {
                return Err(CacheError::Invariant(
                    "sequence reservation disagrees with maximum",
                ));
            }
        }
        let mut prefix_refs: HashMap<PageId, usize> = HashMap::new();
        for entry in self.prefixes.values() {
            if entry.position == 0
                || !entry.position.is_multiple_of(self.config.page_tokens)
                || entry.pages.len() * self.config.page_tokens != entry.position
            {
                return Err(CacheError::Invariant("prefix is not page aligned"));
            }
            for page in &entry.pages {
                let record = self.page_record(*page)?;
                if !record.sealed || record.valid_tokens != self.config.page_tokens {
                    return Err(CacheError::Invariant("prefix references writable page"));
                }
                *prefix_refs.entry(*page).or_default() += 1;
            }
        }
        for (slot_index, slot) in self.pages.iter().enumerate() {
            let Some(page) = &slot.value else {
                continue;
            };
            let id = PageId::new(slot_index, slot.generation);
            if page.active_refs != active_refs.get(&id).copied().unwrap_or(0)
                || page.prefix_refs != prefix_refs.get(&id).copied().unwrap_or(0)
            {
                return Err(CacheError::Invariant("page reference count mismatch"));
            }
            if page.active_refs == 0 && page.prefix_refs == 0 {
                return Err(CacheError::Invariant("unowned resident page"));
            }
        }
        let recomputed = self.compute_stats()?;
        if recomputed != self.stats {
            return Err(CacheError::Invariant(
                "cached accounting differs from ownership",
            ));
        }
        if self.metrics.active_sequences.get() != self.stats.active_sequences as i64
            || self.metrics.retained_prefix_entries.get()
                != self.stats.retained_prefix_entries as i64
            || self.metrics.interned_token_blocks.get() != self.stats.interned_token_blocks as i64
            || self.metrics.resident_pages.get() != self.stats.resident_pages as i64
            || self.metrics.free_pages.get() != self.stats.free_pages as i64
            || self.metrics.reserved_pages.get() != self.stats.reserved_pages as i64
            || self.metrics.deferred_retirement_pages.get()
                != self.stats.deferred_retirement_pages as i64
            || self.metrics.unique_resident_page_bytes.get()
                != self.stats.unique_resident_page_bytes as i64
            || self.metrics.outstanding_reservation_bytes.get()
                != self.stats.outstanding_reservation_bytes as i64
            || self.metrics.active_private_state_bytes.get()
                != self.stats.active_private_state_bytes as i64
            || self.metrics.retained_snapshot_bytes.get()
                != self.stats.retained_snapshot_bytes as i64
            || self.metrics.page_table_bytes.get() != self.stats.page_table_bytes as i64
            || self.metrics.reclaimable_prefix_only_bytes.get()
                != self.stats.reclaimable_prefix_only_bytes as i64
            || self.metrics.total_managed_bytes.get() != self.stats.total_managed_bytes as i64
        {
            return Err(CacheError::Invariant(
                "exported gauges differ from exact cache state",
            ));
        }
        Ok(())
    }

    fn admission_bytes(&self, pages: usize, request: AdmissionRequest) -> Result<usize, B::Error> {
        pages
            .checked_mul(self.page_bytes)
            .and_then(|bytes| bytes.checked_add(request.private_state_bytes))
            .and_then(|bytes| bytes.checked_add(request.page_table_bytes))
            .ok_or(CacheError::ArithmeticOverflow)
    }

    fn admission_limit(&self, allow_emergency: bool) -> Result<usize, B::Error> {
        if allow_emergency {
            Ok(self.config.max_managed_bytes)
        } else {
            self.config
                .max_managed_bytes
                .checked_sub(self.config.emergency_bytes)
                .ok_or(CacheError::ArithmeticOverflow)
        }
    }

    fn plan_evictions(
        &self,
        extra_bytes: usize,
        extra_pages: usize,
        byte_limit: usize,
        protected: Option<PrefixEntryId>,
        target_snapshot_bytes: Option<usize>,
        target_entry_count: Option<usize>,
    ) -> Result<Option<Vec<PrefixEntryId>>, B::Error> {
        let target_total = self
            .stats
            .total_managed_bytes
            .checked_add(extra_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let mut total = target_total;
        let mut pages = self
            .stats
            .resident_pages
            .checked_add(self.stats.reserved_pages)
            .and_then(|pages| pages.checked_add(extra_pages))
            .ok_or(CacheError::ArithmeticOverflow)?;
        let mut snapshots = target_snapshot_bytes.unwrap_or(self.stats.retained_snapshot_bytes);
        let mut entries = target_entry_count.unwrap_or(self.prefixes.len());
        let entry_limit = self.config.max_prefix_entries.unwrap_or(usize::MAX);
        if total <= byte_limit
            && pages <= self.page_capacity
            && snapshots <= self.config.max_snapshot_bytes
            && entries <= entry_limit
        {
            return Ok(Some(Vec::new()));
        }

        let mut candidates = self
            .prefixes
            .iter()
            .filter(|(id, _)| Some(**id) != protected)
            .map(|(id, entry)| (*id, entry.last_used))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(id, last_used)| (*last_used, *id));
        let mut removed_page_refs: HashMap<PageId, usize> = HashMap::new();
        let mut plan = Vec::new();
        for (id, _) in candidates {
            let entry = &self.prefixes[&id];
            snapshots = snapshots
                .checked_sub(entry.snapshot_bytes)
                .ok_or(CacheError::ArithmeticOverflow)?;
            total = total
                .checked_sub(entry.snapshot_bytes)
                .ok_or(CacheError::ArithmeticOverflow)?;
            entries -= 1;
            for page_id in &entry.pages {
                let removed = removed_page_refs.entry(*page_id).or_default();
                *removed += 1;
                let page = self.page_record(*page_id)?;
                if self.backend.retirement_is_immediate()
                    && page.active_refs == 0
                    && *removed == page.prefix_refs
                {
                    pages = pages.checked_sub(1).ok_or(CacheError::ArithmeticOverflow)?;
                    total = total
                        .checked_sub(self.page_bytes)
                        .ok_or(CacheError::ArithmeticOverflow)?;
                }
            }
            plan.push(id);
            if total <= byte_limit
                && pages <= self.page_capacity
                && snapshots <= self.config.max_snapshot_bytes
                && entries <= entry_limit
            {
                return Ok(Some(plan));
            }
        }
        Ok(None)
    }

    fn commit_evictions(
        &mut self,
        entries: &[PrefixEntryId],
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        if entries.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let mut removed_refs: HashMap<PageId, usize> = HashMap::new();
        for id in entries {
            let entry = self.prefixes.get(id).ok_or(CacheError::StalePrefix)?;
            for page in &entry.pages {
                *removed_refs.entry(*page).or_default() += 1;
            }
        }
        let retire_ids = removed_refs
            .iter()
            .filter_map(|(id, removed)| {
                let page = self.page_record(*id).ok()?;
                (page.active_refs == 0 && page.prefix_refs == *removed).then_some(*id)
            })
            .collect::<Vec<_>>();
        let reclaimable = retire_ids
            .len()
            .checked_mul(self.page_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        self.retire_page_ids(&retire_ids, context)?;

        for id in entries {
            let entry = self.prefixes.remove(id).ok_or(CacheError::StalePrefix)?;
            self.index.remove(&entry.key, &entry.blocks);
            for page in entry.pages {
                if retire_ids.contains(&page) {
                    continue;
                }
                let refs = self.page_record(page)?.prefix_refs;
                self.page_record_mut(page)?.prefix_refs = refs
                    .checked_sub(1)
                    .ok_or(CacheError::Invariant("missing prefix page reference"))?;
            }
            self.metrics.prefix_evictions.inc();
        }
        self.metrics
            .bytes_made_reclaimable
            .add(reclaimable.min(isize::MAX as usize) as isize);
        self.metrics.eviction_us.record(elapsed_us(started));
        self.refresh_stats()?;
        Ok(())
    }

    fn retire_page_ids(
        &mut self,
        ids: &[PageId],
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        if ids.is_empty() {
            return Ok(());
        }
        for id in ids {
            let slot = self
                .pages
                .get(id.slot())
                .filter(|slot| slot.generation == id.generation)
                .ok_or(CacheError::StalePage)?;
            if slot.generation == u32::MAX {
                return Err(CacheError::IdExhausted("page generation"));
            }
            if slot
                .value
                .as_ref()
                .and_then(|record| record.physical.as_ref())
                .is_none()
            {
                return Err(CacheError::StalePage);
            }
        }
        let mut physical = Vec::with_capacity(ids.len());
        for id in ids {
            physical.push(
                self.page_record_mut(*id)?
                    .physical
                    .take()
                    .ok_or(CacheError::StalePage)?,
            );
        }
        let outcome = match self.backend.retire_pages(physical, context) {
            Ok(outcome) => outcome,
            Err(failure) => {
                for (id, page) in ids.iter().zip(failure.pages) {
                    self.page_record_mut(*id)?.physical = Some(page);
                }
                self.metrics.backend_failures.inc();
                return Err(CacheError::Backend(failure.error));
            }
        };
        if outcome.deferred_pages > ids.len() {
            return Err(CacheError::Invariant(
                "backend deferred more pages than were retired",
            ));
        }
        self.deferred_pages = self
            .deferred_pages
            .checked_add(outcome.deferred_pages)
            .ok_or(CacheError::ArithmeticOverflow)?;
        for id in ids {
            self.remove_page(*id)?;
        }
        self.metrics
            .pages_retired
            .add(ids.len().min(isize::MAX as usize) as isize);
        Ok(())
    }

    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            let mut order = self
                .prefixes
                .iter()
                .map(|(id, entry)| (*id, entry.last_used))
                .collect::<Vec<_>>();
            order.sort_by_key(|(id, timestamp)| (*timestamp, *id));
            for (index, (id, _)) in order.into_iter().enumerate() {
                self.prefixes
                    .get_mut(&id)
                    .expect("retained entry")
                    .last_used = index as u64 + 1;
            }
            self.clock = self.prefixes.len() as u64;
        }
        self.clock += 1;
        self.clock
    }

    fn next_append_nonce(&mut self) -> Result<u64, B::Error> {
        let nonce = self.append_nonce;
        self.append_nonce = self
            .append_nonce
            .checked_add(1)
            .ok_or(CacheError::IdExhausted("append"))?;
        Ok(nonce)
    }

    fn validate_reservation(&self, reservation: &AppendReservation) -> Result<(), B::Error> {
        let pending = self
            .sequence_record(reservation.sequence)?
            .pending
            .as_ref()
            .ok_or(CacheError::NoAppendPending)?;
        if pending.start_position != reservation.start_position
            || pending.rows != reservation.rows
            || pending.segments.as_ref() != reservation.segments.as_ref()
            || pending.nonce != reservation.nonce
        {
            return Err(CacheError::AppendReservationMismatch);
        }
        Ok(())
    }

    fn page_record(&self, id: PageId) -> Result<&PageRecord<B::Page>, B::Error> {
        self.pages
            .get(id.slot())
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_ref())
            .ok_or(CacheError::StalePage)
    }

    fn page_record_mut(&mut self, id: PageId) -> Result<&mut PageRecord<B::Page>, B::Error> {
        self.pages
            .get_mut(id.slot())
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_mut())
            .ok_or(CacheError::StalePage)
    }

    fn sequence_record(&self, id: SequenceId) -> Result<&SequenceRecord, B::Error> {
        self.sequences
            .get(id.slot())
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_ref())
            .ok_or(CacheError::StaleSequence)
    }

    fn sequence_record_mut(&mut self, id: SequenceId) -> Result<&mut SequenceRecord, B::Error> {
        self.sequences
            .get_mut(id.slot())
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_mut())
            .ok_or(CacheError::StaleSequence)
    }

    fn prepare_page_slot(&self) -> Result<(), B::Error> {
        if self.free_page_slots.is_empty() && self.pages.len() > u32::MAX as usize {
            return Err(CacheError::IdExhausted("page slot"));
        }
        Ok(())
    }

    fn prepare_page_slots(&self, count: usize) -> Result<(), B::Error> {
        let new_slots = count.saturating_sub(self.free_page_slots.len());
        let final_len = self
            .pages
            .len()
            .checked_add(new_slots)
            .ok_or(CacheError::IdExhausted("page slot"))?;
        if final_len > u32::MAX as usize + 1 {
            return Err(CacheError::IdExhausted("page slot"));
        }
        Ok(())
    }

    fn peek_page_id(&self) -> Result<PageId, B::Error> {
        if let Some(slot) = self.free_page_slots.last().copied() {
            Ok(PageId::new(slot, self.pages[slot].generation))
        } else {
            self.prepare_page_slot()?;
            Ok(PageId::new(self.pages.len(), 0))
        }
    }

    fn insert_page(&mut self, value: PageRecord<B::Page>) -> Result<PageId, B::Error> {
        if let Some(slot) = self.free_page_slots.pop() {
            let id = PageId::new(slot, self.pages[slot].generation);
            self.pages[slot].value = Some(value);
            Ok(id)
        } else {
            self.prepare_page_slot()?;
            let slot = self.pages.len();
            self.pages.push(Slot {
                generation: 0,
                value: Some(value),
            });
            Ok(PageId::new(slot, 0))
        }
    }

    fn insert_page_prepared(&mut self, value: PageRecord<B::Page>) -> PageId {
        if let Some(slot) = self.free_page_slots.pop() {
            let id = PageId::new(slot, self.pages[slot].generation);
            self.pages[slot].value = Some(value);
            id
        } else {
            let slot = self.pages.len();
            debug_assert!(slot <= u32::MAX as usize);
            self.pages.push(Slot {
                generation: 0,
                value: Some(value),
            });
            PageId::new(slot, 0)
        }
    }

    fn remove_page(&mut self, id: PageId) -> Result<(), B::Error> {
        let slot = self
            .pages
            .get_mut(id.slot())
            .filter(|slot| slot.generation == id.generation)
            .ok_or(CacheError::StalePage)?;
        slot.value.take().ok_or(CacheError::StalePage)?;
        slot.generation = slot
            .generation
            .checked_add(1)
            .ok_or(CacheError::IdExhausted("page generation"))?;
        self.free_page_slots.push(id.slot());
        Ok(())
    }

    fn prepare_sequence_slot(&self) -> Result<(), B::Error> {
        if self.free_sequence_slots.is_empty() && self.sequences.len() > u32::MAX as usize {
            return Err(CacheError::IdExhausted("sequence slot"));
        }
        Ok(())
    }

    fn insert_sequence(&mut self, value: SequenceRecord) -> Result<SequenceId, B::Error> {
        if let Some(slot) = self.free_sequence_slots.pop() {
            let id = SequenceId::new(slot, self.sequences[slot].generation);
            self.sequences[slot].value = Some(value);
            Ok(id)
        } else {
            self.prepare_sequence_slot()?;
            let slot = self.sequences.len();
            self.sequences.push(Slot {
                generation: 0,
                value: Some(value),
            });
            Ok(SequenceId::new(slot, 0))
        }
    }

    fn remove_sequence(&mut self, id: SequenceId) -> Result<(), B::Error> {
        let slot = self
            .sequences
            .get_mut(id.slot())
            .filter(|slot| slot.generation == id.generation)
            .ok_or(CacheError::StaleSequence)?;
        slot.value.take().ok_or(CacheError::StaleSequence)?;
        slot.generation = slot
            .generation
            .checked_add(1)
            .ok_or(CacheError::IdExhausted("sequence generation"))?;
        self.free_sequence_slots.push(id.slot());
        Ok(())
    }

    fn prepare_remove_sequence(&self, id: SequenceId) -> Result<(), B::Error> {
        let slot = self
            .sequences
            .get(id.slot())
            .filter(|slot| slot.generation == id.generation && slot.value.is_some())
            .ok_or(CacheError::StaleSequence)?;
        if slot.generation == u32::MAX {
            Err(CacheError::IdExhausted("sequence generation"))
        } else {
            Ok(())
        }
    }

    fn prepare_prefix_id(&self) -> Result<(), B::Error> {
        if self.next_prefix_id == u64::MAX {
            Err(CacheError::IdExhausted("prefix entry"))
        } else {
            Ok(())
        }
    }

    fn compute_stats(&self) -> Result<CacheStats, B::Error> {
        let active_sequences = self
            .sequences
            .iter()
            .filter(|slot| slot.value.is_some())
            .count();
        let owned_resident_pages = self
            .pages
            .iter()
            .filter(|slot| slot.value.is_some())
            .count();
        let resident_pages = owned_resident_pages
            .checked_add(self.deferred_pages)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let reserved_pages = self
            .sequences
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .try_fold(0usize, |total, sequence| {
                total
                    .checked_add(sequence.reserved_pages)
                    .ok_or(CacheError::ArithmeticOverflow)
            })?;
        let active_private_state_bytes = self
            .sequences
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .try_fold(0usize, |total, sequence| {
                total
                    .checked_add(sequence.private_state_bytes)
                    .ok_or(CacheError::ArithmeticOverflow)
            })?;
        let page_table_bytes = self
            .sequences
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .try_fold(0usize, |total, sequence| {
                total
                    .checked_add(sequence.page_table_bytes)
                    .ok_or(CacheError::ArithmeticOverflow)
            })?;
        let retained_snapshot_bytes = self.prefixes.values().try_fold(0usize, |total, entry| {
            total
                .checked_add(entry.snapshot_bytes)
                .ok_or(CacheError::ArithmeticOverflow)
        })?;
        let unique_resident_page_bytes = resident_pages
            .checked_mul(self.page_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let outstanding_reservation_bytes = reserved_pages
            .checked_mul(self.page_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let reclaimable_pages = self
            .pages
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .filter(|page| page.active_refs == 0 && page.prefix_refs != 0)
            .count();
        let reclaimable_prefix_only_bytes = reclaimable_pages
            .checked_mul(self.page_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let total_managed_bytes = unique_resident_page_bytes
            .checked_add(outstanding_reservation_bytes)
            .and_then(|total| total.checked_add(active_private_state_bytes))
            .and_then(|total| total.checked_add(retained_snapshot_bytes))
            .and_then(|total| total.checked_add(page_table_bytes))
            .ok_or(CacheError::ArithmeticOverflow)?;
        let uncommitted_bytes = self
            .config
            .max_managed_bytes
            .saturating_sub(total_managed_bytes);
        let free_page_slots = self
            .page_capacity
            .saturating_sub(resident_pages.saturating_add(reserved_pages));
        Ok(CacheStats {
            active_sequences,
            retained_prefix_entries: self.prefixes.len(),
            interned_token_blocks: self.index.block_count(),
            resident_pages,
            free_pages: (uncommitted_bytes / self.page_bytes).min(free_page_slots),
            reserved_pages,
            deferred_retirement_pages: self.deferred_pages,
            unique_resident_page_bytes,
            outstanding_reservation_bytes,
            active_private_state_bytes,
            retained_snapshot_bytes,
            page_table_bytes,
            reclaimable_prefix_only_bytes,
            total_managed_bytes,
        })
    }

    fn refresh_stats(&mut self) -> Result<(), B::Error> {
        self.stats = self.compute_stats()?;
        self.publish_gauges();
        Ok(())
    }

    fn refresh_derived_stats(&mut self) -> Result<(), B::Error> {
        self.refresh_stats()
    }

    fn publish_gauges(&self) {
        let stats = self.stats;
        self.metrics
            .active_sequences
            .set(stats.active_sequences as i64);
        self.metrics
            .retained_prefix_entries
            .set(stats.retained_prefix_entries as i64);
        self.metrics
            .interned_token_blocks
            .set(stats.interned_token_blocks as i64);
        self.metrics.resident_pages.set(stats.resident_pages as i64);
        self.metrics.free_pages.set(stats.free_pages as i64);
        self.metrics.reserved_pages.set(stats.reserved_pages as i64);
        self.metrics
            .deferred_retirement_pages
            .set(stats.deferred_retirement_pages as i64);
        self.metrics
            .unique_resident_page_bytes
            .set(stats.unique_resident_page_bytes as i64);
        self.metrics
            .outstanding_reservation_bytes
            .set(stats.outstanding_reservation_bytes as i64);
        self.metrics
            .active_private_state_bytes
            .set(stats.active_private_state_bytes as i64);
        self.metrics
            .retained_snapshot_bytes
            .set(stats.retained_snapshot_bytes as i64);
        self.metrics
            .page_table_bytes
            .set(stats.page_table_bytes as i64);
        self.metrics
            .reclaimable_prefix_only_bytes
            .set(stats.reclaimable_prefix_only_bytes as i64);
        self.metrics
            .total_managed_bytes
            .set(stats.total_managed_bytes as i64);
    }
}

fn validate_config<E>(config: CacheConfig, page_bytes: usize) -> Result<(), E> {
    if config.page_tokens == 0 {
        return Err(ConfigError::ZeroPageTokens.into());
    }
    if page_bytes == 0 {
        return Err(ConfigError::ZeroPageBytes.into());
    }
    if config.max_managed_bytes == 0 {
        return Err(ConfigError::ZeroManagedBytes.into());
    }
    if page_bytes > config.max_managed_bytes {
        return Err(ConfigError::PageExceedsManagedBytes.into());
    }
    if config.emergency_bytes > config.max_managed_bytes {
        return Err(ConfigError::EmergencyCapacityExceedsManagedBytes.into());
    }
    if config.max_snapshot_bytes > config.max_managed_bytes {
        return Err(ConfigError::SnapshotLimitExceedsManagedBytes.into());
    }
    if config.max_managed_bytes > i64::MAX as usize {
        return Err(ConfigError::ManagedBytesExceedMetricRange.into());
    }
    let max_pages = config.max_managed_bytes / page_bytes;
    max_pages
        .checked_mul(page_bytes)
        .ok_or(ConfigError::CapacityOverflow)?;
    Ok(())
}

fn page_record_from<B: PageBackend>(
    pages: &[Slot<PageRecord<B::Page>>],
    id: PageId,
) -> Result<&PageRecord<B::Page>, B::Error> {
    pages
        .get(id.slot())
        .filter(|slot| slot.generation == id.generation)
        .and_then(|slot| slot.value.as_ref())
        .ok_or(CacheError::StalePage)
}

fn physical_refs_from<'a, B: PageBackend>(
    pages: &'a [Slot<PageRecord<B::Page>>],
    ids: &[PageId],
) -> Result<Vec<&'a B::Page>, B::Error> {
    ids.iter()
        .map(|id| {
            page_record_from::<B>(pages, *id)?
                .physical
                .as_ref()
                .ok_or(CacheError::StalePage)
        })
        .collect()
}

fn div_ceil<E>(value: usize, divisor: usize) -> Result<usize, E> {
    value
        .checked_add(divisor - 1)
        .map(|sum| sum / divisor)
        .ok_or(CacheError::ArithmeticOverflow)
}

fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PageAllocation, RetireError, RetireOutcome};
    use std::convert::Infallible;

    struct NoopBackend;

    impl PageBackend for NoopBackend {
        type Page = ();
        type Context<'a> = ();
        type Error = Infallible;

        fn page_bytes(&self) -> usize {
            1
        }

        fn allocate_page(
            &mut self,
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<PageAllocation<Self::Page>, Self::Error> {
            Ok(PageAllocation {
                page: (),
                recycled: false,
            })
        }

        fn rollback_page(&mut self, _page: Self::Page, _context: &mut Self::Context<'_>) {}

        fn abort_append(
            &mut self,
            _pages: &[&Self::Page],
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<(), Self::Error> {
            Ok(())
        }

        fn copy_partial_page(
            &mut self,
            _source: &Self::Page,
            _valid_tokens: usize,
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<PageAllocation<Self::Page>, Self::Error> {
            Ok(PageAllocation {
                page: (),
                recycled: false,
            })
        }

        fn commit_append(
            &mut self,
            _committed_pages: &[&Self::Page],
            _sealed_pages: &[&Self::Page],
            _released_pages: &[&Self::Page],
            _new_position: usize,
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<(), Self::Error> {
            Ok(())
        }

        fn update_page_table(
            &mut self,
            _pages: &[&Self::Page],
            _position: usize,
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<(), Self::Error> {
            Ok(())
        }

        fn retire_pages(
            &mut self,
            _pages: Vec<Self::Page>,
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
            Ok(RetireOutcome::default())
        }

        fn poll_reclaimed(
            &mut self,
            _context: &mut Self::Context<'_>,
        ) -> core::result::Result<usize, Self::Error> {
            Ok(0)
        }
    }

    #[test]
    fn clock_renormalization_preserves_lru_order() {
        let mut cache = SequenceCache::new(
            CacheConfig {
                page_tokens: 4,
                max_managed_bytes: 16,
                max_snapshot_bytes: 0,
                max_prefix_entries: None,
                emergency_bytes: 0,
            },
            NoopBackend,
        )
        .expect("cache");
        let older = PrefixEntryId(0);
        let newer = PrefixEntryId(1);
        cache.prefixes.insert(
            older,
            PrefixEntry {
                key: PrefixKey::new_from_array([0]),
                blocks: Vec::new(),
                pages: Vec::new(),
                position: 4,
                snapshot: (),
                snapshot_bytes: 0,
                last_used: 100,
            },
        );
        cache.prefixes.insert(
            newer,
            PrefixEntry {
                key: PrefixKey::new_from_array([1]),
                blocks: Vec::new(),
                pages: Vec::new(),
                position: 4,
                snapshot: (),
                snapshot_bytes: 0,
                last_used: 200,
            },
        );
        cache.clock = u64::MAX;

        assert_eq!(cache.tick(), 3);
        assert_eq!(cache.prefixes[&older].last_used, 1);
        assert_eq!(cache.prefixes[&newer].last_used, 2);
    }
}
