//! Errors that cross the inference-engine boundary.

/// Convenient result alias for inference-engine operations.
pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

/// Failure reported by an inference engine to its caller.
#[derive(Debug)]
pub enum InferenceError {
    /// A CUDA operation or device-side validation failed during inference.
    Cuda(eider_cuda::Error),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(error) => Some(error),
        }
    }
}

impl From<eider_cuda::Error> for InferenceError {
    fn from(error: eider_cuda::Error) -> Self {
        Self::Cuda(error)
    }
}
