//! Backend-independent ownership for paged sequence state and reusable prefixes.
//!
//! The crate owns logical page lifetimes, prefix indexing, admission reservations,
//! and exact accounting. Physical page storage and synchronization remain the
//! responsibility of a runtime-provided [`PageBackend`].
//!
//! # Concepts
//!
//! - A **sequence** is a live run of tokens. It owns an ordered list of logical
//!   **pages** of fixed token width, allocates one new page per boundary
//!   crossing, and writes through a private unsealed tail page.
//! - Sealed pages are immutable and shareable. **Prefix entries** retain a
//!   page-aligned run of sealed pages plus a runtime-defined
//!   [`RetainedSnapshot`]; later sequences restore them without copying
//!   storage. A **branch** copies only the unaligned tail, giving
//!   copy-on-write sequences.
//! - The prefix index is content-addressed: page-sized token runs are interned
//!   as **token blocks**, and an adaptive radix tree finds the longest retained
//!   prefix for a query. Lookups never intern blocks, so speculative probing
//!   cannot grow the index.
//! - **Admission** is strict. A request reserves every page it could need up
//!   to its maximum position, and is declined with
//!   [`AdmissionOutcome::WouldBlock`] rather than admitted into an overcommitted
//!   cache. Under pressure the manager evicts least-recently-used prefix
//!   entries, which never disturbs live sequences.
//! - Handles are copyable values which detect staleness: [`SequenceId`] and
//!   [`PageId`] are generational arena handles which become invalid when their
//!   slot is recycled, while [`PrefixEntryId`] and [`TokenBlockId`] are
//!   never-reused identities which become invalid on eviction or collection.
//!
//! # Lifecycle
//!
//! 1. [`SequenceCache::admit`] a sequence, optionally with a [`PrefixMatch`]
//!    from [`SequenceCache::lookup_prefix`].
//! 2. Append tokens with [`SequenceCache::reserve_append`],
//!    [`SequenceCache::with_append_pages`], and [`SequenceCache::commit_append`]
//!    (or [`SequenceCache::abort_append`]).
//! 3. Reach useful page boundaries, then [`SequenceCache::retain_prefix`] to
//!    make the prefix reusable.
//! 4. [`SequenceCache::finish`] the sequence when done.
//! 5. Poll [`SequenceCache::reclaim_deferred`] so asynchronous backend
//!    retirements release capacity (admission polls automatically).
//!
//! Every mutation republishes exact [`CacheStats`] and exported
//! [`CacheMetrics`]; [`SequenceCache::validate`] recomputes all ownership
//! invariants from first principles for tests and health checks.

mod backend;
mod error;
mod index;
mod manager;
mod metrics;

pub use backend::{PageAllocation, PageBackend, RetireError, RetireOutcome};
pub use error::{CacheError, ConfigError, Result};
pub use manager::{
    AdmissionOutcome, AdmissionRequest, AppendPage, AppendPages, AppendReservation, AppendSegment,
    CacheConfig, CacheStats, PageId, PageTableView, PrefixEntryId, PrefixMatch, RetainOutcome,
    SequenceCache, SequenceId, TokenBlockId,
};
pub use metrics::CacheMetrics;

/// Immutable model-specific state retained at a page-aligned prefix.
///
/// Snapshots capture whatever non-paged state a runtime needs to resume past a
/// retained prefix — for example recurrent state or auxiliary tensors. The
/// declared bytes count toward the configured snapshot budget, so
/// implementations must report them exactly.
pub trait RetainedSnapshot {
    /// Exact managed bytes owned by this snapshot.
    fn retained_bytes(&self) -> usize;
}

/// A runtime without auxiliary state can instantiate the cache over `()`.
impl RetainedSnapshot for () {
    fn retained_bytes(&self) -> usize {
        0
    }
}
