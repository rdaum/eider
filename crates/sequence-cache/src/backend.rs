//! Physical page backend contract.

/// An all-or-nothing retirement failure.
///
/// The backend returns every page unchanged when a retirement batch fails. This
/// lets the manager restore its exact prior ownership state.
#[derive(Debug)]
pub struct RetireError<E, P> {
    /// Backend-specific failure.
    pub error: E,
    /// All pages passed to the failed operation, in their original order.
    pub pages: Vec<P>,
}

/// One backend page allocation and whether it reused an existing physical slot.
#[derive(Debug)]
pub struct PageAllocation<P> {
    /// The newly writable page.
    pub page: P,
    /// Whether the page came from a recycled slot rather than fresh storage.
    ///
    /// This is informational only; the manager uses it to distinguish
    /// allocation from reclamation in its exported counters.
    pub recycled: bool,
}

/// One physical page segment covered by a pending append transaction.
///
/// The manager supplies these in logical input order before publishing the
/// reservation's complete page table. Backends may use the geometry to prepare
/// rollback state for storage which already contains committed rows.
pub struct BackendAppendPage<'a, P> {
    page: &'a P,
    page_offset: usize,
    input_offset: usize,
    rows: usize,
    existing: bool,
}

/// Backend-visible result of committing a reserved append prefix.
pub struct BackendAppendCommit<'a, P> {
    committed_pages: &'a [&'a P],
    sealed_pages: &'a [&'a P],
    released_pages: &'a [&'a P],
    rows: usize,
    position: usize,
}

impl<'a, P> BackendAppendCommit<'a, P> {
    pub(crate) fn new(
        committed_pages: &'a [&'a P],
        sealed_pages: &'a [&'a P],
        released_pages: &'a [&'a P],
        rows: usize,
        position: usize,
    ) -> Self {
        Self {
            committed_pages,
            sealed_pages,
            released_pages,
            rows,
            position,
        }
    }

    /// Complete logical page table after commit.
    pub fn committed_pages(&self) -> &[&P] {
        self.committed_pages
    }

    /// Ordered committed pages which became full.
    pub fn sealed_pages(&self) -> &[&P] {
        self.sealed_pages
    }

    /// Uncommitted reserved suffix pages to reclaim.
    pub fn released_pages(&self) -> &[&P] {
        self.released_pages
    }

    /// Number of source rows committed from the reservation.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// New logical sequence position.
    pub fn position(&self) -> usize {
        self.position
    }
}

impl<'a, P> BackendAppendPage<'a, P> {
    pub(crate) fn new(
        page: &'a P,
        page_offset: usize,
        input_offset: usize,
        rows: usize,
        existing: bool,
    ) -> Self {
        Self {
            page,
            page_offset,
            input_offset,
            rows,
            existing,
        }
    }

    /// Physical page receiving this segment.
    pub fn page(&self) -> &P {
        self.page
    }

    /// First writable row in the physical page.
    pub fn page_offset(&self) -> usize {
        self.page_offset
    }

    /// First source row in the caller's compute span.
    pub fn input_offset(&self) -> usize {
        self.input_offset
    }

    /// Number of consecutive rows in this segment.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Whether this page was already part of the committed sequence.
    pub fn existed_before_reservation(&self) -> bool {
        self.existing
    }
}

/// Successful ownership transfer for a retirement batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetireOutcome {
    /// Pages still resident and unavailable pending asynchronous reclamation.
    pub deferred_pages: usize,
}

/// Runtime-owned physical storage and synchronization operations.
///
/// The manager owns logical page lifetimes, reference counts, and accounting;
/// the backend owns the physical storage those pages alias (for example device
/// KV slabs) plus the page table each sequence reads through.
///
/// Metadata mutation is serialized by [`crate::SequenceCache`], so
/// implementations do not need interior locking. Implementations must not
/// partially succeed: allocation and copy errors return no page, page table
/// updates leave the previous table intact on failure, and retirement returns
/// all input pages in [`RetireError`] on failure. Pages handed to
/// [`PageBackend::retire_pages`] become backend-owned again; until
/// [`PageBackend::poll_reclaimed`] reports them reusable they remain charged
/// against the cache's capacity.
pub trait PageBackend {
    /// One physical page bundle.
    ///
    /// A bundle groups the per-layer storage addressed by one shared physical
    /// slot so that the manager can account for it as a single page.
    type Page;
    /// Explicit runtime synchronization or executor context.
    ///
    /// The caller threads one context through every operation of a higher-level
    /// cache call so the backend can enrol page updates and retirement into its
    /// own synchronization discipline.
    type Context<'a>;
    /// Backend-owned rollback state for one pending append.
    type AppendTransaction;
    /// Backend-specific error.
    type Error;

    /// Exact bytes occupied by every page bundle.
    fn page_bytes(&self) -> usize;

    /// Hard number of physical page slots, when the backend is preallocated.
    ///
    /// Returning a value lets admission prove that reservations cannot exhaust
    /// the backend even when non-page bytes leave unused room in the byte
    /// budget.
    fn page_capacity(&self) -> Option<usize> {
        None
    }

    /// Allocate one writable page bundle.
    fn allocate_page(
        &mut self,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<PageAllocation<Self::Page>, Self::Error>;

    /// Return an unpublished allocation after a later transactional step fails.
    ///
    /// The page has never been visible to an attention operation, so this must
    /// be infallible and need not follow normal asynchronous retirement.
    fn rollback_page(&mut self, page: Self::Page, context: &mut Self::Context<'_>);

    /// Prepare rollback state before publishing an append span.
    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        start_position: usize,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<Self::AppendTransaction, Self::Error>;

    /// Restore backend-owned contents, republish the old table, and reclaim all
    /// pages from an aborted reservation as one transaction.
    ///
    /// Implementations must ensure that earlier work using these pages has
    /// completed before making them reusable. On failure every page must
    /// remain allocated and unchanged.
    fn abort_append(
        &mut self,
        transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<(), Self::Error>;

    /// Copy the valid prefix of a writable tail into a new private page.
    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<PageAllocation<Self::Page>, Self::Error>;

    /// Commit a prefix of a logical append spanning one or more physical pages.
    ///
    /// The complete logical page table was already published by reservation.
    /// `committed_pages` is the complete table after commit, `sealed_pages` is
    /// its ordered subset which became full, and `released_pages` is the
    /// uncommitted suffix to reclaim. Publishing the shorter table and
    /// `new_position`, sealing pages, and reclaiming the suffix must be atomic:
    /// on error the reservation remains writable and may be retried or aborted.
    fn commit_append(
        &mut self,
        transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<(), Self::Error>;

    /// Atomically replace a sequence's backend-native page table.
    ///
    /// On failure both the previously published page ordering and position
    /// must remain unchanged.
    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<(), Self::Error>;

    /// Retire a batch atomically, or return every page unchanged on failure.
    ///
    /// Asynchronous implementations may enqueue retirement here. Such pages
    /// remain backend-owned; the configured capacity must include any deferred
    /// pool slots.
    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<RetireOutcome, RetireError<Self::Error, Self::Page>>;

    /// Whether a successful retirement can be counted as immediately reusable
    /// while planning admission. Returning false is the conservative default.
    fn retirement_is_immediate(&self) -> bool {
        false
    }

    /// Report pages from earlier deferred retirements which are now reusable.
    fn poll_reclaimed(
        &mut self,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<usize, Self::Error>;
}
