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
    pub page: P,
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
/// Metadata mutation is serialized by [`crate::SequenceCache`]. Implementations
/// must not partially succeed: allocation and copy errors return no page, page
/// table updates leave the previous table intact on failure, and retirement
/// returns all input pages in [`RetireError`] on failure.
pub trait PageBackend {
    /// One physical page bundle.
    type Page;
    /// Explicit runtime synchronization or executor context.
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

    /// Copy the valid prefix of a writable tail into a new private page.
    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<PageAllocation<Self::Page>, Self::Error>;

    /// Commit a logical append and optionally seal its now-complete page.
    ///
    /// The page-table length update and sealing transition are one fallible
    /// operation: on error the prior table and writable-page state remain valid.
    fn commit_append(
        &mut self,
        page: &mut Self::Page,
        pages_before: &[&Self::Page],
        pages_after: &[&Self::Page],
        new_position: usize,
        seal: bool,
        context: &mut Self::Context<'_>,
    ) -> core::result::Result<(), Self::Error>;

    /// Atomically replace a sequence's backend-native page table.
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
