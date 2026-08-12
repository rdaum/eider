//! Backend-independent ownership for paged sequence state and reusable prefixes.
//!
//! The crate owns logical page lifetimes, prefix indexing, admission reservations,
//! and exact accounting. Physical page storage and synchronization remain the
//! responsibility of a runtime-provided [`PageBackend`].

mod backend;
mod error;
mod index;
mod manager;
mod metrics;

pub use backend::{PageAllocation, PageBackend, RetireError, RetireOutcome};
pub use error::{CacheError, ConfigError, Result};
pub use manager::{
    AdmissionOutcome, AdmissionRequest, AppendTarget, CacheConfig, CacheStats, PageId,
    PageTableView, PrefixEntryId, PrefixMatch, RetainOutcome, SequenceCache, SequenceId,
    TokenBlockId,
};
pub use metrics::CacheMetrics;

/// Immutable model-specific state retained at a page-aligned prefix.
pub trait RetainedSnapshot {
    /// Exact managed bytes owned by this snapshot.
    fn retained_bytes(&self) -> usize;
}

impl RetainedSnapshot for () {
    fn retained_bytes(&self) -> usize {
        0
    }
}
