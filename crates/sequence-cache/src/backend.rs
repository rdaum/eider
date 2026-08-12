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

    /// Reclaim all pages from an aborted reservation after its old table is
    /// republished.
    ///
    /// Implementations must ensure that earlier work using these pages has
    /// completed before making them reusable. On failure every page must
    /// remain allocated and unchanged.
    fn abort_append(
        &mut self,
        pages: &[&Self::Page],
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
        committed_pages: &[&Self::Page],
        sealed_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        new_position: usize,
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
