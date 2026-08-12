use core::fmt;

/// Invalid immutable cache geometry or capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    ZeroPageTokens,
    ZeroPageBytes,
    ZeroManagedBytes,
    PageExceedsManagedBytes,
    EmergencyCapacityExceedsManagedBytes,
    SnapshotLimitExceedsManagedBytes,
    ManagedBytesExceedMetricRange,
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
#[derive(Debug)]
pub enum CacheError<E> {
    Config(ConfigError),
    StaleSequence,
    StalePage,
    StalePrefix,
    InvalidPosition,
    InvalidTokenPrefix,
    SnapshotCapacity,
    PrefixCapacity,
    ArithmeticOverflow,
    IdExhausted(&'static str),
    Invariant(&'static str),
    AppendPending,
    NoAppendPending,
    AppendTargetMismatch,
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
            Self::AppendTargetMismatch => f.write_str("append target is stale or mismatched"),
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

pub type Result<T, E> = core::result::Result<T, CacheError<E>>;
