use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, CacheConfig, CacheError, PageAllocation, PageBackend,
    RetainOutcome, RetainedSnapshot, RetireError, RetireOutcome, SequenceCache, SequenceId,
};
use std::fmt;

#[derive(Clone, Debug)]
struct FakePage {
    id: u64,
    rows: Vec<u32>,
    sealed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Allocate,
    Copy,
    Update,
    Commit,
    Retire,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FakeError(Operation);

impl fmt::Display for FakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injected {:?} failure", self.0)
    }
}

impl std::error::Error for FakeError {}

#[derive(Default)]
struct FakeContext {
    table: Vec<u64>,
    position: usize,
}

struct FakeBackend {
    page_bytes: usize,
    page_capacity: Option<usize>,
    next_id: u64,
    allocations: usize,
    copies: usize,
    retirements: usize,
    rollbacks: usize,
    fail_next: Option<Operation>,
    immediate_retirement: bool,
    deferred_pages: usize,
    complete_deferred: bool,
}

impl FakeBackend {
    fn new(page_bytes: usize) -> Self {
        Self {
            page_bytes,
            page_capacity: None,
            next_id: 0,
            allocations: 0,
            copies: 0,
            retirements: 0,
            rollbacks: 0,
            fail_next: None,
            immediate_retirement: true,
            deferred_pages: 0,
            complete_deferred: false,
        }
    }

    fn deferred(page_bytes: usize) -> Self {
        Self {
            immediate_retirement: false,
            ..Self::new(page_bytes)
        }
    }

    fn fail(&mut self, operation: Operation) {
        self.fail_next = Some(operation);
    }

    fn with_page_capacity(mut self, page_capacity: usize) -> Self {
        self.page_capacity = Some(page_capacity);
        self
    }

    fn take_failure(&mut self, operation: Operation) -> Result<(), FakeError> {
        if self.fail_next == Some(operation) {
            self.fail_next = None;
            Err(FakeError(operation))
        } else {
            Ok(())
        }
    }
}

impl PageBackend for FakeBackend {
    type Page = FakePage;
    type Context<'a> = FakeContext;
    type Error = FakeError;

    fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn page_capacity(&self) -> Option<usize> {
        self.page_capacity
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.take_failure(Operation::Allocate)?;
        let page = FakePage {
            id: self.next_id,
            rows: Vec::new(),
            sealed: false,
        };
        self.next_id += 1;
        self.allocations += 1;
        Ok(PageAllocation {
            page,
            recycled: false,
        })
    }

    fn rollback_page(&mut self, _page: Self::Page, _context: &mut Self::Context<'_>) {
        self.rollbacks += 1;
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.take_failure(Operation::Copy)?;
        let page = FakePage {
            id: self.next_id,
            rows: source.rows[..valid_tokens].to_vec(),
            sealed: false,
        };
        self.next_id += 1;
        self.copies += 1;
        Ok(PageAllocation {
            page,
            recycled: false,
        })
    }

    fn commit_append(
        &mut self,
        page: &mut Self::Page,
        pages_before: &[&Self::Page],
        pages_after: &[&Self::Page],
        new_position: usize,
        seal: bool,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.take_failure(Operation::Commit)?;
        let mut table = pages_before.iter().map(|page| page.id).collect::<Vec<_>>();
        table.push(page.id);
        table.extend(pages_after.iter().map(|page| page.id));
        context.table = table;
        context.position = new_position;
        page.sealed = seal;
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.take_failure(Operation::Update)?;
        context.table = pages.iter().map(|page| page.id).collect();
        context.position = position;
        Ok(())
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        if let Err(error) = self.take_failure(Operation::Retire) {
            return Err(RetireError { error, pages });
        }
        self.retirements += pages.len();
        let deferred_pages = if self.immediate_retirement {
            0
        } else {
            self.deferred_pages += pages.len();
            pages.len()
        };
        Ok(RetireOutcome { deferred_pages })
    }

    fn retirement_is_immediate(&self) -> bool {
        self.immediate_retirement
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize, Self::Error> {
        if self.complete_deferred {
            self.complete_deferred = false;
            Ok(std::mem::take(&mut self.deferred_pages))
        } else {
            Ok(0)
        }
    }
}

#[derive(Clone, Debug)]
struct Snapshot(usize);

impl RetainedSnapshot for Snapshot {
    fn retained_bytes(&self) -> usize {
        self.0
    }
}

fn config(max_bytes: usize) -> CacheConfig {
    CacheConfig {
        page_tokens: 4,
        max_managed_bytes: max_bytes,
        max_snapshot_bytes: max_bytes / 2,
        max_prefix_entries: None,
        emergency_bytes: 0,
    }
}

fn request(max_position: usize) -> AdmissionRequest {
    AdmissionRequest {
        max_position,
        private_state_bytes: 0,
        page_table_bytes: 0,
        allow_emergency: false,
    }
}

fn cache(max_bytes: usize) -> SequenceCache<FakeBackend, Snapshot> {
    SequenceCache::new(config(max_bytes), FakeBackend::new(100)).expect("valid cache")
}

fn admit(
    cache: &mut SequenceCache<FakeBackend, Snapshot>,
    max_position: usize,
    context: &mut FakeContext,
) -> SequenceId {
    match cache
        .admit(
            None,
            request(max_position),
            context,
            |snapshot, position| {
                assert!(snapshot.is_none());
                assert_eq!(position, 0);
                Ok(())
            },
        )
        .expect("admit")
    {
        AdmissionOutcome::Admitted(id) => id,
        AdmissionOutcome::WouldBlock => panic!("unexpected admission pressure"),
    }
}

fn append(
    cache: &mut SequenceCache<FakeBackend, Snapshot>,
    sequence: SequenceId,
    rows: &[u32],
    context: &mut FakeContext,
) {
    let mut consumed = 0;
    while consumed < rows.len() {
        let target = cache
            .reserve_append(sequence, rows.len() - consumed, context)
            .expect("reserve append");
        let count = target.max_rows();
        cache
            .with_append_page(target, |_backend, page| {
                assert_eq!(page.rows.len(), target.page_offset());
                page.rows
                    .extend_from_slice(&rows[consumed..consumed + count]);
                Ok(())
            })
            .expect("write append rows");
        cache
            .commit_append(target, count, context)
            .expect("commit append");
        consumed += count;
    }
}

fn make_retained_prefix(
    cache: &mut SequenceCache<FakeBackend, Snapshot>,
    tokens: &[u32],
    max_position: usize,
    snapshot_bytes: usize,
    context: &mut FakeContext,
) -> (SequenceId, sequence_cache::PrefixEntryId) {
    assert!(tokens.len().is_multiple_of(4));
    let sequence = admit(cache, max_position, context);
    append(cache, sequence, tokens, context);
    let entry = match cache
        .retain_prefix(sequence, tokens, Snapshot(snapshot_bytes), context)
        .expect("retain prefix")
    {
        RetainOutcome::Inserted(id) => id,
        RetainOutcome::Duplicate(_) => panic!("unexpected duplicate"),
    };
    (sequence, entry)
}

#[test]
fn cacheable_positions_cover_empty_short_aligned_and_multi_page_prompts() {
    let cache = cache(1_000);
    assert_eq!(cache.cacheable_prefix_tokens(0), 0);
    assert_eq!(cache.cacheable_prefix_tokens(1), 0);
    assert_eq!(cache.cacheable_prefix_tokens(4), 0);
    assert_eq!(cache.cacheable_prefix_tokens(5), 4);
    assert_eq!(cache.cacheable_prefix_tokens(8), 4);
    assert_eq!(cache.cacheable_prefix_tokens(9), 8);
}

#[test]
fn longest_prefix_selects_nested_entry_and_rejects_divergence() {
    let mut cache = cache(2_000);
    let mut context = FakeContext::default();
    let sequence = admit(&mut cache, 8, &mut context);
    append(&mut cache, sequence, &[0, 1, 2, 3], &mut context);
    cache
        .retain_prefix(sequence, &[0, 1, 2, 3], Snapshot(0), &mut context)
        .expect("first prefix");
    append(&mut cache, sequence, &[4, 5, 6, 7], &mut context);
    cache
        .retain_prefix(
            sequence,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            Snapshot(0),
            &mut context,
        )
        .expect("second prefix");

    assert_eq!(
        cache
            .lookup_prefix(&[0, 1, 2, 3, 4, 5, 6, 7, 99])
            .expect("longest hit")
            .position(),
        8
    );
    assert_eq!(
        cache
            .lookup_prefix(&[0, 1, 2, 3, 40, 50, 60, 70, 99])
            .expect("shorter hit")
            .position(),
        4
    );
    assert!(cache.lookup_prefix(&[9, 1, 2, 3, 99]).is_none());
    cache.validate().expect("valid nested prefix state");
}

#[test]
fn lookup_miss_does_not_intern_and_eviction_collects_blocks() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    assert!(cache.lookup_prefix(&[1, 2, 3, 4, 5]).is_none());
    assert_eq!(cache.stats().interned_token_blocks, 0);
    let (sequence, entry) = make_retained_prefix(&mut cache, &[1, 2, 3, 4], 4, 0, &mut context);
    assert_eq!(cache.stats().interned_token_blocks, 1);
    cache.finish(sequence, &mut context).expect("finish");
    cache.evict_prefix(entry, &mut context).expect("evict");
    assert_eq!(cache.stats().interned_token_blocks, 0);
    cache.validate().expect("valid after collection");
}

#[test]
fn duplicate_insertion_does_not_evict_another_prefix() {
    let mut cfg = config(1_000);
    cfg.max_prefix_entries = Some(2);
    let mut cache = SequenceCache::new(cfg, FakeBackend::new(100)).expect("cache");
    let mut context = FakeContext::default();
    let (first, _) = make_retained_prefix(&mut cache, &[1, 2, 3, 4], 4, 0, &mut context);
    let (second, _) = make_retained_prefix(&mut cache, &[5, 6, 7, 8], 4, 0, &mut context);
    let allocations = cache.backend().allocations;
    let duplicate = cache
        .retain_prefix(first, &[1, 2, 3, 4], Snapshot(0), &mut context)
        .expect("duplicate");
    assert!(matches!(duplicate, RetainOutcome::Duplicate(_)));
    assert_eq!(cache.stats().retained_prefix_entries, 2);
    assert_eq!(cache.backend().allocations, allocations);
    assert!(cache.lookup_prefix(&[5, 6, 7, 8, 9]).is_some());
    cache.finish(first, &mut context).expect("finish first");
    cache.finish(second, &mut context).expect("finish second");
}

#[test]
fn insertion_and_aligned_restore_share_pages_without_copy_or_allocation() {
    let mut cache = cache(2_000);
    let mut context = FakeContext::default();
    let (original, _) =
        make_retained_prefix(&mut cache, &[1, 2, 3, 4, 5, 6, 7, 8], 12, 20, &mut context);
    let allocations = cache.backend().allocations;
    let copies = cache.backend().copies;
    let matched = cache
        .lookup_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9])
        .expect("prefix hit");
    let restored = match cache
        .admit(
            Some(matched),
            request(16),
            &mut context,
            |snapshot, position| {
                assert_eq!(snapshot.expect("snapshot").0, 20);
                assert_eq!(position, 8);
                Ok(())
            },
        )
        .expect("restore admission")
    {
        AdmissionOutcome::Admitted(id) => id,
        AdmissionOutcome::WouldBlock => panic!("restore should fit"),
    };
    assert_eq!(cache.backend().allocations, allocations);
    assert_eq!(cache.backend().copies, copies);
    assert_eq!(cache.stats().resident_pages, 2);
    assert_eq!(cache.stats().reserved_pages, 3);
    assert_eq!(
        cache.page_table(original).expect("original table").pages(),
        cache.page_table(restored).expect("restored table").pages()
    );
    cache.validate().expect("valid shared restore");
}

#[test]
fn page_table_order_is_logical_after_manager_slot_reuse() {
    let mut cache = cache(2_000);
    let mut temporary_context = FakeContext::default();
    let temporary = admit(&mut cache, 4, &mut temporary_context);
    append(
        &mut cache,
        temporary,
        &[90, 91, 92, 93],
        &mut temporary_context,
    );

    let mut context = FakeContext::default();
    let sequence = admit(&mut cache, 8, &mut context);
    append(&mut cache, sequence, &[1, 2, 3, 4], &mut context);
    assert_eq!(context.table, vec![1]);

    // Releasing the first arena slot makes the next logical page reuse a
    // lower metadata slot than the sequence's existing first page.
    cache
        .finish(temporary, &mut temporary_context)
        .expect("finish temporary sequence");
    append(&mut cache, sequence, &[5, 6, 7, 8], &mut context);
    assert_eq!(context.table, vec![1, 2]);
    cache.validate().expect("valid after page-slot reuse");
}

#[test]
fn unaligned_branch_shares_complete_pages_and_copies_one_tail() {
    let mut cache = cache(2_000);
    let mut context = FakeContext::default();
    let source = admit(&mut cache, 12, &mut context);
    append(&mut cache, source, &[1, 2, 3, 4, 5, 6], &mut context);
    let source_pages = cache
        .page_table(source)
        .expect("source table")
        .pages()
        .to_vec();
    let branch = match cache
        .branch(source, request(12), &mut context)
        .expect("branch")
    {
        AdmissionOutcome::Admitted(id) => id,
        AdmissionOutcome::WouldBlock => panic!("branch should fit"),
    };
    let branch_pages = cache
        .page_table(branch)
        .expect("branch table")
        .pages()
        .to_vec();
    assert_eq!(branch_pages[0], source_pages[0]);
    assert_ne!(branch_pages[1], source_pages[1]);
    assert_eq!(cache.backend().copies, 1);
    assert_eq!(cache.stats().reserved_pages, 2);
    cache.validate().expect("valid branch");
}

#[test]
fn eviction_preserves_live_pages_and_last_finish_retires_them() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    let (sequence, entry) = make_retained_prefix(&mut cache, &[1, 2, 3, 4], 4, 0, &mut context);
    cache.evict_prefix(entry, &mut context).expect("evict");
    assert_eq!(cache.stats().resident_pages, 1);
    assert_eq!(cache.backend().retirements, 0);
    cache.finish(sequence, &mut context).expect("finish");
    assert_eq!(cache.stats().resident_pages, 0);
    assert_eq!(cache.backend().retirements, 1);
}

#[test]
fn nested_prefixes_account_shared_physical_pages_once() {
    let mut cache = cache(2_000);
    let mut context = FakeContext::default();
    let sequence = admit(&mut cache, 8, &mut context);
    append(&mut cache, sequence, &[1, 2, 3, 4], &mut context);
    cache
        .retain_prefix(sequence, &[1, 2, 3, 4], Snapshot(0), &mut context)
        .expect("short prefix");
    append(&mut cache, sequence, &[5, 6, 7, 8], &mut context);
    cache
        .retain_prefix(
            sequence,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            Snapshot(0),
            &mut context,
        )
        .expect("long prefix");
    assert_eq!(cache.stats().resident_pages, 2);
    assert_eq!(cache.stats().unique_resident_page_bytes, 200);
    cache.finish(sequence, &mut context).expect("finish");
    assert_eq!(cache.stats().reclaimable_prefix_only_bytes, 200);
    cache.validate().expect("valid physical accounting");
}

#[test]
fn strict_admission_reserves_future_pages_and_cancellation_releases_them() {
    let mut cache = cache(300);
    let mut context = FakeContext::default();
    let first = admit(&mut cache, 12, &mut context);
    assert_eq!(cache.stats().reserved_pages, 3);
    let blocked = cache
        .admit(
            None,
            request(4),
            &mut context,
            |_snapshot, _position| Ok(()),
        )
        .expect("pressure is not an error");
    assert_eq!(blocked, AdmissionOutcome::WouldBlock);
    append(&mut cache, first, &[1, 2], &mut context);
    assert_eq!(cache.stats().resident_pages, 1);
    assert_eq!(cache.stats().reserved_pages, 2);
    assert_eq!(cache.stats().total_managed_bytes, 300);
    cache.finish(first, &mut context).expect("cancel");
    assert_eq!(cache.stats().total_managed_bytes, 0);
    let second = admit(&mut cache, 4, &mut context);
    cache.finish(second, &mut context).expect("finish second");
}

#[test]
fn strict_admission_respects_a_preallocated_backend_page_limit() {
    let backend = FakeBackend::new(100).with_page_capacity(2);
    let mut cache = SequenceCache::<_, Snapshot>::new(config(10_000), backend).expect("cache");
    let mut context = FakeContext::default();
    assert_eq!(
        cache
            .admit(None, request(12), &mut context, |_, _| Ok(()))
            .expect("capacity decision"),
        AdmissionOutcome::WouldBlock
    );
    assert_eq!(cache.stats().free_pages, 2);
    cache.validate().expect("valid after capped rejection");
}

#[test]
fn deferred_retirement_stays_charged_until_backend_reclaims_it() {
    let mut cache = SequenceCache::new(config(200), FakeBackend::deferred(100)).expect("cache");
    let mut context = FakeContext::default();
    let first = admit(&mut cache, 4, &mut context);
    append(&mut cache, first, &[1], &mut context);
    cache.finish(first, &mut context).expect("deferred finish");
    assert_eq!(cache.stats().resident_pages, 1);
    assert_eq!(cache.stats().deferred_retirement_pages, 1);
    assert_eq!(cache.stats().total_managed_bytes, 100);

    assert_eq!(
        cache
            .admit(
                None,
                request(8),
                &mut context,
                |_snapshot, _position| Ok(())
            )
            .expect("admission pressure"),
        AdmissionOutcome::WouldBlock
    );
    cache.with_backend(|backend| backend.complete_deferred = true);
    let second = match cache
        .admit(
            None,
            request(8),
            &mut context,
            |_snapshot, _position| Ok(()),
        )
        .expect("admission after reclaim")
    {
        AdmissionOutcome::Admitted(id) => id,
        AdmissionOutcome::WouldBlock => panic!("reclaimed pages should release capacity"),
    };
    assert_eq!(cache.stats().deferred_retirement_pages, 0);
    assert_eq!(cache.stats().reserved_pages, 2);
    cache.finish(second, &mut context).expect("finish second");
}

#[test]
fn backend_failures_leave_accounting_valid() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    let sequence = admit(&mut cache, 8, &mut context);

    cache.with_backend(|backend| backend.fail(Operation::Allocate));
    let error = cache
        .reserve_append(sequence, 1, &mut context)
        .expect_err("allocation failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Allocate))
    ));
    assert_eq!(cache.stats().resident_pages, 0);
    assert_eq!(cache.stats().reserved_pages, 2);
    cache.validate().expect("valid after allocation failure");

    cache.with_backend(|backend| backend.fail(Operation::Update));
    let error = cache
        .reserve_append(sequence, 1, &mut context)
        .expect_err("table failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Update))
    ));
    assert_eq!(cache.backend().rollbacks, 1);
    assert_eq!(cache.stats().resident_pages, 0);
    cache.validate().expect("valid after update failure");

    append(&mut cache, sequence, &[1, 2, 3], &mut context);
    let branch_stats = cache.stats();
    cache.with_backend(|backend| backend.fail(Operation::Copy));
    let error = cache
        .branch(sequence, request(8), &mut context)
        .expect_err("copy failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Copy))
    ));
    assert_eq!(cache.stats(), branch_stats);

    let target = cache
        .reserve_append(sequence, 1, &mut context)
        .expect("pending append");
    cache
        .with_append_page(target, |_backend, page| {
            page.rows.push(4);
            Ok(())
        })
        .expect("write row");
    cache.with_backend(|backend| backend.fail(Operation::Commit));
    let error = cache
        .commit_append(target, 1, &mut context)
        .expect_err("commit failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Commit))
    ));
    assert_eq!(cache.page_table(sequence).expect("table").position(), 3);
    cache.abort_append(target).expect("abort failed append");
    cache.validate().expect("valid after commit failure");
}

#[test]
fn retirement_failure_restores_prefix_and_page_ownership() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    let (sequence, entry) = make_retained_prefix(&mut cache, &[1, 2, 3, 4], 4, 0, &mut context);
    cache
        .finish(sequence, &mut context)
        .expect("finish active owner");
    let before = cache.stats();
    cache.with_backend(|backend| backend.fail(Operation::Retire));
    let error = cache
        .evict_prefix(entry, &mut context)
        .expect_err("retirement failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Retire))
    ));
    assert_eq!(cache.stats(), before);
    assert!(cache.lookup_prefix(&[1, 2, 3, 4, 9]).is_some());
    cache.validate().expect("valid retirement rollback");
}

#[test]
fn restore_and_admission_table_failures_do_not_acquire_ownership() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    let (original, _) = make_retained_prefix(&mut cache, &[1, 2, 3, 4], 4, 10, &mut context);
    cache
        .finish(original, &mut context)
        .expect("finish original");
    let matched = cache.lookup_prefix(&[1, 2, 3, 4, 9]).expect("prefix match");
    let before = cache.stats();

    let error = cache
        .admit(
            Some(matched),
            request(8),
            &mut context,
            |_snapshot, _position| Err(FakeError(Operation::Copy)),
        )
        .expect_err("snapshot restore failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Copy))
    ));
    assert_eq!(cache.stats(), before);
    cache.validate().expect("valid after restore failure");

    cache.with_backend(|backend| backend.fail(Operation::Update));
    let error = cache
        .admit(
            Some(matched),
            request(8),
            &mut context,
            |_snapshot, _position| Ok(()),
        )
        .expect_err("admission table failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Update))
    ));
    assert_eq!(cache.stats(), before);
    assert!(cache.lookup_prefix(&[1, 2, 3, 4, 9]).is_some());
    cache
        .validate()
        .expect("valid after admission table failure");
}

#[test]
fn failed_replacement_keeps_old_prefix_and_rolls_back_new_token_blocks() {
    let mut cfg = config(1_000);
    cfg.max_prefix_entries = Some(1);
    let mut cache = SequenceCache::new(cfg, FakeBackend::new(100)).expect("cache");
    let mut context = FakeContext::default();
    let (first_sequence, _) = make_retained_prefix(&mut cache, &[1, 2, 3, 4], 4, 0, &mut context);
    cache
        .finish(first_sequence, &mut context)
        .expect("make first prefix reclaimable");
    let second = admit(&mut cache, 4, &mut context);
    append(&mut cache, second, &[5, 6, 7, 8], &mut context);
    cache.with_backend(|backend| backend.fail(Operation::Retire));
    let error = cache
        .retain_prefix(second, &[5, 6, 7, 8], Snapshot(0), &mut context)
        .expect_err("replacement retirement failure");
    assert!(matches!(
        error,
        CacheError::Backend(FakeError(Operation::Retire))
    ));
    assert_eq!(cache.stats().retained_prefix_entries, 1);
    assert_eq!(cache.stats().interned_token_blocks, 1);
    assert!(cache.lookup_prefix(&[1, 2, 3, 4, 9]).is_some());
    assert!(cache.lookup_prefix(&[5, 6, 7, 8, 9]).is_none());
    cache.validate().expect("valid failed replacement rollback");
}

#[test]
fn metrics_gauges_match_exact_stats_across_lifecycle() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    let sequence = admit(&mut cache, 8, &mut context);
    append(&mut cache, sequence, &[1, 2, 3, 4], &mut context);
    cache
        .retain_prefix(sequence, &[1, 2, 3, 4], Snapshot(7), &mut context)
        .expect("retain");
    assert!(cache.lookup_prefix(&[1, 2, 3, 4, 9]).is_some());
    let stats = cache.stats();
    let metrics = cache.metrics();
    assert_eq!(
        metrics.active_sequences.get(),
        stats.active_sequences as i64
    );
    assert_eq!(
        metrics.retained_prefix_entries.get(),
        stats.retained_prefix_entries as i64
    );
    assert_eq!(
        metrics.interned_token_blocks.get(),
        stats.interned_token_blocks as i64
    );
    assert_eq!(metrics.resident_pages.get(), stats.resident_pages as i64);
    assert_eq!(metrics.free_pages.get(), stats.free_pages as i64);
    assert_eq!(metrics.reserved_pages.get(), stats.reserved_pages as i64);
    assert_eq!(
        metrics.unique_resident_page_bytes.get(),
        stats.unique_resident_page_bytes as i64
    );
    assert_eq!(
        metrics.outstanding_reservation_bytes.get(),
        stats.outstanding_reservation_bytes as i64
    );
    assert_eq!(
        metrics.retained_snapshot_bytes.get(),
        stats.retained_snapshot_bytes as i64
    );
    assert_eq!(
        metrics.total_managed_bytes.get(),
        stats.total_managed_bytes as i64
    );
    assert_eq!(metrics.admission_successes.sum(), 1);
    assert_eq!(metrics.pages_allocated.sum(), 1);
    assert_eq!(metrics.pages_sealed.sum(), 1);
    assert_eq!(metrics.prefix_insertions.sum(), 1);
    assert_eq!(metrics.prefix_hits.sum(), 1);
    cache.validate().expect("metrics agree with ownership");
}

#[derive(Clone)]
struct ActiveSequence {
    id: SequenceId,
    tokens: Vec<u32>,
    max_position: usize,
}

#[test]
fn deterministic_state_machine_recomputes_invariants_after_every_operation() {
    let mut cfg = config(4_000);
    cfg.max_prefix_entries = Some(4);
    let mut cache = SequenceCache::new(cfg, FakeBackend::new(100)).expect("cache");
    let mut context = FakeContext::default();
    let mut active: Vec<ActiveSequence> = Vec::new();
    let mut seed = 0x9e37_79b9_u32;
    let mut next_token = 1_u32;

    for _step in 0..500 {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let choice = seed % 6;
        match choice {
            0 if active.len() < 6 => {
                let max_position = ((seed as usize >> 8) % 4 + 1) * 4;
                if let AdmissionOutcome::Admitted(id) = cache
                    .admit(
                        None,
                        request(max_position),
                        &mut context,
                        |_snapshot, _position| Ok(()),
                    )
                    .expect("state-machine admission")
                {
                    active.push(ActiveSequence {
                        id,
                        tokens: Vec::new(),
                        max_position,
                    });
                }
            }
            1 if !active.is_empty() => {
                let index = (seed as usize >> 8) % active.len();
                let sequence = &mut active[index];
                let remaining = sequence.max_position - sequence.tokens.len();
                if remaining != 0 {
                    let count = remaining.min(((seed as usize >> 16) % 3) + 1);
                    let rows = (0..count)
                        .map(|_| {
                            let token = next_token;
                            next_token = next_token.wrapping_add(1);
                            token
                        })
                        .collect::<Vec<_>>();
                    append(&mut cache, sequence.id, &rows, &mut context);
                    sequence.tokens.extend(rows);
                }
            }
            2 if !active.is_empty() => {
                let index = (seed as usize >> 8) % active.len();
                let sequence = &active[index];
                if !sequence.tokens.is_empty() && sequence.tokens.len().is_multiple_of(4) {
                    cache
                        .retain_prefix(
                            sequence.id,
                            &sequence.tokens,
                            Snapshot((seed as usize >> 24) % 8),
                            &mut context,
                        )
                        .expect("state-machine retain");
                }
            }
            3 if !active.is_empty() => {
                let index = (seed as usize >> 8) % active.len();
                let sequence = active.swap_remove(index);
                cache
                    .finish(sequence.id, &mut context)
                    .expect("state-machine finish");
            }
            4 if !active.is_empty() => {
                let index = (seed as usize >> 8) % active.len();
                let mut query = active[index].tokens.clone();
                query.push(u32::MAX);
                cache.lookup_prefix(&query);
            }
            5 if active.len() < 6 && !active.is_empty() => {
                let index = (seed as usize >> 8) % active.len();
                let source = active[index].clone();
                if !source.tokens.is_empty()
                    && !source.tokens.len().is_multiple_of(4)
                    && let AdmissionOutcome::Admitted(id) = cache
                        .branch(source.id, request(source.max_position), &mut context)
                        .expect("state-machine branch")
                {
                    active.push(ActiveSequence {
                        id,
                        tokens: source.tokens,
                        max_position: source.max_position,
                    });
                }
            }
            _ => {}
        }
        cache.validate().expect("state-machine ownership invariant");
        assert!(cache.stats().total_managed_bytes <= cache.config().max_managed_bytes);
    }

    for sequence in active {
        cache
            .finish(sequence.id, &mut context)
            .expect("final state-machine finish");
        cache.validate().expect("valid state-machine cleanup");
    }
}

#[test]
fn stale_sequence_and_page_ids_are_rejected_after_slot_reuse() {
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    let first = admit(&mut cache, 4, &mut context);
    append(&mut cache, first, &[1], &mut context);
    let stale_page = cache.page_table(first).expect("table").pages()[0];
    cache.finish(first, &mut context).expect("finish first");
    let second = admit(&mut cache, 4, &mut context);
    append(&mut cache, second, &[2], &mut context);
    assert!(matches!(
        cache.page_table(first),
        Err(CacheError::StaleSequence)
    ));
    assert!(matches!(cache.page(stale_page), Err(CacheError::StalePage)));
    cache.finish(second, &mut context).expect("finish second");
}

#[test]
fn checked_configuration_and_accounting_reject_impossible_values() {
    let mut cfg = config(1_000);
    cfg.page_tokens = 0;
    assert!(matches!(
        SequenceCache::<FakeBackend, Snapshot>::new(cfg, FakeBackend::new(100)),
        Err(CacheError::Config(_))
    ));
    let cfg = config(100);
    assert!(matches!(
        SequenceCache::<FakeBackend, Snapshot>::new(cfg, FakeBackend::new(101)),
        Err(CacheError::Config(_))
    ));
    let mut cache = cache(1_000);
    let mut context = FakeContext::default();
    assert!(matches!(
        cache.admit(
            None,
            AdmissionRequest {
                max_position: usize::MAX,
                private_state_bytes: 0,
                page_table_bytes: 0,
                allow_emergency: false,
            },
            &mut context,
            |_snapshot, _position| Ok(())
        ),
        Err(CacheError::ArithmeticOverflow)
    ));
}
