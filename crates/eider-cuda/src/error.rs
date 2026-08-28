//! Error and result types used by the crate.

/// Convenient result alias for CUDA/cuBLASLt operations in this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by CUDA calls, cuBLASLt calls, and local validation checks.
#[derive(Debug)]
pub enum Error {
    /// A CUDA runtime call failed.
    Cuda(&'static str, i32),
    /// A cuBLASLt call failed.
    Cublas(&'static str, i32),
    /// A deterministic smoke-test result did not match its CPU reference.
    Mismatch {
        /// Expected values.
        expected: Vec<f32>,
        /// Actual values.
        actual: Vec<f32>,
    },
    /// cuBLASLt did not return a usable algorithm for the requested operation.
    EmptyHeuristic(&'static str),
    /// Matrix dimensions did not match the requested operation shape.
    Shape {
        /// Check name.
        label: &'static str,
        /// Expected dimensions or relationship.
        expected: String,
        /// Actual dimensions.
        actual: String,
    },
    /// A checkpoint or metadata file did not match the expected format.
    Format {
        /// Check name.
        label: &'static str,
        /// Human-readable detail.
        detail: String,
    },
    /// A numerical check exceeded its allowed tolerance.
    Tolerance {
        /// Check name.
        label: &'static str,
        /// Largest absolute error observed.
        max_abs_error: f32,
        /// Allowed largest absolute error.
        tolerance: f32,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cuda(call, code) => write!(f, "{call} failed with CUDA status {code}"),
            Error::Cublas(call, code) => write!(f, "{call} failed with cuBLAS status {code}"),
            Error::Mismatch { expected, actual } => {
                write!(f, "result mismatch: expected {expected:?}, got {actual:?}")
            }
            Error::EmptyHeuristic(label) => write!(f, "{label} returned no algorithms"),
            Error::Shape {
                label,
                expected,
                actual,
            } => write!(
                f,
                "{label} shape mismatch: expected {expected}, got {actual}"
            ),
            Error::Format { label, detail } => write!(f, "{label} format error: {detail}"),
            Error::Tolerance {
                label,
                max_abs_error,
                tolerance,
            } => write!(
                f,
                "{label} exceeded tolerance: max_abs_error={max_abs_error}, tolerance={tolerance}"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<eider_format::Error> for Error {
    fn from(error: eider_format::Error) -> Self {
        match error {
            eider_format::Error::Shape {
                label,
                expected,
                actual,
            } => Self::Shape {
                label,
                expected,
                actual,
            },
            eider_format::Error::Format { label, detail } => Self::Format { label, detail },
        }
    }
}
