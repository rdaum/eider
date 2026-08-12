//! Error types for configuration and cache operations.

use core::fmt;

/// Invalid immutable cache geometry or capacity.
///
/// Returned by [`crate::SequenceCache::new`] before any state is created. The
/// byte limits are checked against the backend's reported page size, so some
/// variants (for example [`ConfigError::PageExceedsManagedBytes`]) can only
/// surface once a backend is attached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    /// `page_tokens` was zero.
    ZeroPageTokens,
    /// The backend reported a zero-byte page.
    ZeroPageBytes,
    /// `max_managed_bytes` was zero.
    ZeroManagedBytes,
    /// One backend page is larger than the whole managed byte budget.
    PageExceedsManagedBytes,
    /// `emergency_bytes` exceeds `max_managed_bytes`.
    EmergencyCapacityExceedsManagedBytes,
    /// `max_snapshot_bytes` exceeds `max_managed_bytes`.
    SnapshotLimitExceedsManagedBytes,
    /// `max_managed_bytes` cannot be represented exactly by the metric gauges.
    ManagedBytesExceedMetricRange,
    /// Deriving the managed page capacity overflowed.
    CapacityOverflow,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ZeroPageTokens => "tokens per page must be non-zero",
            Self::ZeroPageBytes => "backend page bytes must be non-zero",
            Self::ZeroManagedBytes => "managed byte capacity must be non-zero",
            Self::PageExceedsManagedBytes => "one page exceeds managed byte capacity",
            Self::EmergencyCapacityExceedsManagedBytes => {
                "emergency capacity exceeds managed byte capacity"
            }
            Self::SnapshotLimitExceedsManagedBytes => {
                "snapshot byte limit exceeds managed byte capacity"
            }
            Self::ManagedBytesExceedMetricRange => {
                "managed byte capacity exceeds the exact metric range"
            }
            Self::CapacityOverflow => "configured capacity arithmetic overflowed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ConfigError {}

/// Sequence-cache operation failure.
///
/// `E` is the associated error type of the configured
/// [`crate::PageBackend`]. Every variant other than [`CacheError::Backend`]
/// leaves cache ownership and accounting unchanged; the backend contract makes
/// the same guarantee for failed storage operations.
#[derive(Debug)]
pub enum CacheError<E> {
    /// The supplied [`crate::CacheConfig`] failed validation.
    Config(ConfigError),
    /// The sequence handle no longer refers to a live sequence.
    StaleSequence,
    /// The page handle no longer refers to a live page.
    StalePage,
    /// The prefix entry handle no longer refers to a retained entry.
    StalePrefix,
    /// The requested position is impossible for the target sequence.
    InvalidPosition,
    /// The token prefix is empty, unaligned, or longer than the sequence.
    InvalidTokenPrefix,
    /// The retained snapshot would exceed the configured snapshot byte limit.
    SnapshotCapacity,
    /// The prefix entry would exceed the configured entry or byte limits.
    PrefixCapacity,
    /// Internal accounting arithmetic overflowed.
    ArithmeticOverflow,
    /// An internal ID space was exhausted; the payload names the ID kind.
    IdExhausted(&'static str),
    /// An internal ownership invariant was violated; the payload describes it.
    Invariant(&'static str),
    /// The sequence already has a pending append.
    AppendPending,
    /// The sequence has no pending append matching the request.
    NoAppendPending,
    /// The append reservation is stale or does not match the pending append.
    AppendReservationMismatch,
    /// The page backend reported a failure.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for CacheError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(f, "invalid cache configuration: {error}"),
            Self::StaleSequence => f.write_str("stale sequence ID"),
            Self::StalePage => f.write_str("stale page ID"),
            Self::StalePrefix => f.write_str("stale prefix entry ID"),
            Self::InvalidPosition => f.write_str("invalid sequence position"),
            Self::InvalidTokenPrefix => f.write_str("invalid token prefix"),
            Self::SnapshotCapacity => f.write_str("prefix snapshot capacity exceeded"),
            Self::PrefixCapacity => f.write_str("prefix entry capacity exceeded"),
            Self::ArithmeticOverflow => f.write_str("cache accounting arithmetic overflowed"),
            Self::IdExhausted(kind) => write!(f, "{kind} ID space exhausted"),
            Self::Invariant(detail) => write!(f, "cache invariant failed: {detail}"),
            Self::AppendPending => f.write_str("sequence already has a pending append"),
            Self::NoAppendPending => f.write_str("sequence has no pending append"),
            Self::AppendReservationMismatch => {
                f.write_str("append reservation is stale or mismatched")
            }
            Self::Backend(error) => write!(f, "page backend operation failed: {error}"),
        }
    }
}

impl<E> std::error::Error for CacheError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}

impl<E> From<ConfigError> for CacheError<E> {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

/// Result type for cache operations parameterised by the backend error.
pub type Result<T, E> = core::result::Result<T, CacheError<E>>;
