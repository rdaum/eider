//! Errors from host checkpoint and artifact handling.

/// Convenient result alias for Eider format operations.
pub type Result<T> = std::result::Result<T, Error>;

/// An invalid file representation or host-side shape.
#[derive(Debug)]
pub enum Error {
    /// A shape or size relationship did not hold.
    Shape {
        /// Check name.
        label: &'static str,
        /// Required relationship.
        expected: String,
        /// Observed value.
        actual: String,
    },
    /// A checkpoint or artifact did not meet its format contract.
    Format {
        /// Check name.
        label: &'static str,
        /// Human-readable detail.
        detail: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape {
                label,
                expected,
                actual,
            } => write!(
                formatter,
                "{label} shape mismatch: expected {expected}, got {actual}"
            ),
            Self::Format { label, detail } => {
                write!(formatter, "{label} format error: {detail}")
            }
        }
    }
}

impl std::error::Error for Error {}
