use crate::RetainedSnapshot;
use crate::backend::PageBackend;
use crate::error::{CacheError, ConfigError, Result};
use crate::index::PrefixIndex;
use crate::metrics::CacheMetrics;
use rart::VectorKey;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::time::Instant;

macro_rules! generational_id {
    ($name:ident) => {
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

generational_id!(PageId);
generational_id!(SequenceId);

/// Stable, never-reused identity for one retained prefix entry.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixEntryId(u64);

/// Stable identity for one interned page-sized token block.
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheConfig {
    pub page_tokens: usize,
    pub max_managed_bytes: usize,
    pub max_snapshot_bytes: usize,
    pub max_prefix_entries: Option<usize>,
    /// Capacity unavailable to ordinary admissions but usable by runtime policy.
    pub emergency_bytes: usize,
}

/// Per-request strict admission requirements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionRequest {
    pub max_position: usize,
    pub private_state_bytes: usize,
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
    pub fn entry_id(self) -> PrefixEntryId {
        self.entry
    }

    pub fn position(self) -> usize {
        self.position
    }

    pub fn page_count(self) -> usize {
        self.page_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionOutcome {
    Admitted(SequenceId),
    WouldBlock,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainOutcome {
    Inserted(PrefixEntryId),
    Duplicate(PrefixEntryId),
}

/// Capability returned for one pending append.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendTarget {
    sequence: SequenceId,
    page: PageId,
    start: usize,
    max_rows: usize,
    nonce: u64,
}

impl AppendTarget {
    pub fn sequence(self) -> SequenceId {
        self.sequence
    }

    pub fn page(self) -> PageId {
        self.page
    }

    pub fn page_offset(self) -> usize {
        self.start
    }

    pub fn max_rows(self) -> usize {
        self.max_rows
    }
}

/// Borrowed logical page ordering for an attention operation.
pub struct PageTableView<'a> {
    pages: &'a [PageId],
    position: usize,
    page_tokens: usize,
}

impl PageTableView<'_> {
    pub fn pages(&self) -> &[PageId] {
        self.pages
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }
}

/// Exact synchronous state owned by one manager.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub active_sequences: usize,
    pub retained_prefix_entries: usize,
    pub interned_token_blocks: usize,
    pub resident_pages: usize,
    pub free_pages: usize,
    pub reserved_pages: usize,
    pub deferred_retirement_pages: usize,
    pub unique_resident_page_bytes: usize,
    pub outstanding_reservation_bytes: usize,
    pub active_private_state_bytes: usize,
    pub retained_snapshot_bytes: usize,
    pub page_table_bytes: usize,
    pub reclaimable_prefix_only_bytes: usize,
    pub total_managed_bytes: usize,
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

struct PageRecord<P> {
    physical: Option<P>,
    active_refs: usize,
    prefix_refs: usize,
    valid_tokens: usize,
    sealed: bool,
}

#[derive(Clone, Copy)]
struct PendingAppend {
    page: PageId,
    start: usize,
    max_rows: usize,
    nonce: u64,
}

struct SequenceRecord {
    pages: Vec<PageId>,
    position: usize,
    max_position: usize,
    reserved_pages: usize,
    private_state_bytes: usize,
    page_table_bytes: usize,
    pending: Option<PendingAppend>,
}

struct PrefixEntry<S> {
    key: VectorKey,
    blocks: Vec<TokenBlockId>,
    pages: Vec<PageId>,
    position: usize,
    snapshot: S,
    snapshot_bytes: usize,
    last_used: u64,
}

/// Single-owner logical sequence and reusable-prefix manager.
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
    pub fn new(config: CacheConfig, backend: B) -> Result<Self, B::Error> {
        Self::with_metrics(config, backend, CacheMetrics::default())
    }

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

    pub fn config(&self) -> CacheConfig {
        self.config
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Perform a short backend configuration or diagnostic operation.
    pub fn with_backend<R>(&mut self, operation: impl FnOnce(&mut B) -> R) -> R {
        operation(&mut self.backend)
    }

    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Default checkpoint position, preserving the final prompt token for decode.
    pub fn cacheable_prefix_tokens(&self, prompt_tokens: usize) -> usize {
        prompt_tokens.saturating_sub(1) / self.config.page_tokens * self.config.page_tokens
    }

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
    /// `restore` receives the retained immutable snapshot before any manager
    /// ownership is committed. A failure leaves cache metadata unchanged.
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

    /// Begin an append, allocating exactly one page when crossing a boundary.
    pub fn reserve_append(
        &mut self,
        sequence: SequenceId,
        requested_rows: usize,
        context: &mut B::Context<'_>,
    ) -> Result<AppendTarget, B::Error> {
        if requested_rows == 0 {
            return Err(CacheError::InvalidPosition);
        }
        let (position, max_position, old_pages, reserved_pages, pending) = {
            let record = self.sequence_record(sequence)?;
            (
                record.position,
                record.max_position,
                record.pages.clone(),
                record.reserved_pages,
                record.pending,
            )
        };
        if pending.is_some() {
            return Err(CacheError::AppendPending);
        }
        if position >= max_position {
            return Err(CacheError::InvalidPosition);
        }
        let mut page_ids = old_pages.clone();
        let offset = position % self.config.page_tokens;
        let page = if offset == 0 {
            let reusable_empty_tail = old_pages.last().copied().filter(|page| {
                self.page_record(*page)
                    .map(|record| record.valid_tokens == 0 && !record.sealed)
                    .unwrap_or(false)
            });
            if let Some(page) = reusable_empty_tail {
                page
            } else {
                if reserved_pages == 0 {
                    return Err(CacheError::Invariant("append has no admitted reservation"));
                }
                self.prepare_page_slot()?;
                let allocation = match self.backend.allocate_page(context) {
                    Ok(allocation) => allocation,
                    Err(error) => {
                        self.metrics.backend_failures.inc();
                        return Err(CacheError::Backend(error));
                    }
                };
                let physical = allocation.page;
                page_ids.push(self.peek_page_id()?);
                let (backend, pages) = (&mut self.backend, &self.pages);
                let mut refs = physical_refs_from::<B>(pages, &old_pages)?;
                refs.push(&physical);
                if let Err(error) = backend.update_page_table(&refs, position, context) {
                    backend.rollback_page(physical, context);
                    self.metrics.backend_failures.inc();
                    return Err(CacheError::Backend(error));
                }
                let page = self.insert_page(PageRecord {
                    physical: Some(physical),
                    active_refs: 1,
                    prefix_refs: 0,
                    valid_tokens: 0,
                    sealed: false,
                })?;
                debug_assert_eq!(page_ids.last().copied(), Some(page));
                let sequence_record = self.sequence_record_mut(sequence)?;
                sequence_record.pages.push(page);
                sequence_record.reserved_pages -= 1;
                if allocation.recycled {
                    self.metrics.pages_recycled.inc();
                } else {
                    self.metrics.pages_allocated.inc();
                }
                page
            }
        } else {
            *old_pages
                .last()
                .ok_or(CacheError::Invariant("non-zero position has no tail page"))?
        };
        let page_record = self.page_record(page)?;
        if page_record.sealed || page_record.active_refs != 1 || page_record.prefix_refs != 0 {
            return Err(CacheError::Invariant("writable tail is not private"));
        }
        let available = (self.config.page_tokens - offset).min(max_position - position);
        let max_rows = requested_rows.min(available);
        let nonce = self.next_append_nonce()?;
        let pending = PendingAppend {
            page,
            start: offset,
            max_rows,
            nonce,
        };
        self.sequence_record_mut(sequence)?.pending = Some(pending);
        self.refresh_stats()?;
        Ok(AppendTarget {
            sequence,
            page,
            start: offset,
            max_rows,
            nonce,
        })
    }

    /// Borrow the backend and pending writable page for a runtime append enqueue.
    pub fn with_append_page<R, F>(
        &mut self,
        target: AppendTarget,
        operation: F,
    ) -> Result<R, B::Error>
    where
        F: FnOnce(&mut B, &mut B::Page) -> core::result::Result<R, B::Error>,
    {
        self.validate_target(target)?;
        let slot = target.page.slot();
        let (backend, pages) = (&mut self.backend, &mut self.pages);
        let record = pages
            .get_mut(slot)
            .and_then(|slot| (slot.generation == target.page.generation).then_some(slot))
            .and_then(|slot| slot.value.as_mut())
            .ok_or(CacheError::StalePage)?;
        let page = record.physical.as_mut().ok_or(CacheError::StalePage)?;
        operation(backend, page).map_err(|error| {
            self.metrics.backend_failures.inc();
            CacheError::Backend(error)
        })
    }

    /// Commit rows previously written through an [`AppendTarget`].
    pub fn commit_append(
        &mut self,
        target: AppendTarget,
        rows: usize,
        context: &mut B::Context<'_>,
    ) -> Result<(), B::Error> {
        self.validate_target(target)?;
        if rows == 0 || rows > target.max_rows {
            return Err(CacheError::InvalidPosition);
        }
        let (page_ids, old_position) = {
            let sequence = self.sequence_record(target.sequence)?;
            (sequence.pages.clone(), sequence.position)
        };
        let new_position = old_position
            .checked_add(rows)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let new_valid = target
            .start
            .checked_add(rows)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let seal = new_valid == self.config.page_tokens;

        let target_slot = target.page.slot();
        let (backend, pages) = (&mut self.backend, &mut self.pages);
        let (before, rest) = pages.split_at_mut(target_slot);
        let (target_slot_ref, after) = rest.split_first_mut().ok_or(CacheError::StalePage)?;
        if target_slot_ref.generation != target.page.generation {
            return Err(CacheError::StalePage);
        }
        let target_record = target_slot_ref
            .value
            .as_mut()
            .ok_or(CacheError::StalePage)?;
        let target_physical = target_record
            .physical
            .as_mut()
            .ok_or(CacheError::StalePage)?;
        let target_logical = page_ids
            .iter()
            .position(|page| *page == target.page)
            .ok_or(CacheError::Invariant(
                "append page missing from sequence table",
            ))?;
        let mut table_before = Vec::with_capacity(target_logical);
        let mut table_after = Vec::with_capacity(page_ids.len().saturating_sub(target_logical + 1));
        for (logical, id) in page_ids.iter().enumerate() {
            if logical == target_logical {
                debug_assert_eq!(*id, target.page);
                continue;
            }
            let slot = if id.slot() < target_slot {
                before.get(id.slot())
            } else {
                after.get(id.slot() - target_slot - 1)
            }
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.value.as_ref())
            .and_then(|record| record.physical.as_ref())
            .ok_or(CacheError::StalePage)?;
            if logical < target_logical {
                table_before.push(slot);
            } else {
                table_after.push(slot);
            }
        }
        if let Err(error) = backend.commit_append(
            target_physical,
            &table_before,
            &table_after,
            new_position,
            seal,
            context,
        ) {
            self.metrics.backend_failures.inc();
            return Err(CacheError::Backend(error));
        }
        target_record.valid_tokens = new_valid;
        target_record.sealed = seal;
        let sequence = self.sequence_record_mut(target.sequence)?;
        sequence.position = new_position;
        sequence.pending = None;
        if seal {
            self.metrics.pages_sealed.inc();
        }
        self.refresh_stats()?;
        Ok(())
    }

    /// Cancel a pending append without changing logical length.
    pub fn abort_append(&mut self, target: AppendTarget) -> Result<(), B::Error> {
        self.validate_target(target)?;
        self.sequence_record_mut(target.sequence)?.pending = None;
        Ok(())
    }

    pub fn page_table(&self, sequence: SequenceId) -> Result<PageTableView<'_>, B::Error> {
        let sequence = self.sequence_record(sequence)?;
        Ok(PageTableView {
            pages: &sequence.pages,
            position: sequence.position,
            page_tokens: self.config.page_tokens,
        })
    }

    pub fn page(&self, page: PageId) -> Result<&B::Page, B::Error> {
        self.page_record(page)?
            .physical
            .as_ref()
            .ok_or(CacheError::StalePage)
    }

    /// Retain the sequence's current aligned pages without copying KV storage.
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
            if sequence.pages.len() + sequence.reserved_pages != max_pages {
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

    fn validate_target(&self, target: AppendTarget) -> Result<(), B::Error> {
        let pending = self
            .sequence_record(target.sequence)?
            .pending
            .ok_or(CacheError::NoAppendPending)?;
        if pending.page != target.page
            || pending.start != target.start
            || pending.max_rows != target.max_rows
            || pending.nonce != target.nonce
        {
            return Err(CacheError::AppendTargetMismatch);
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
            _page: &mut Self::Page,
            _pages_before: &[&Self::Page],
            _pages_after: &[&Self::Page],
            _new_position: usize,
            _seal: bool,
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
                key: VectorKey::new_from_vec(vec![0]),
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
                key: VectorKey::new_from_vec(vec![1]),
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
