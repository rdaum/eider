//! Errors that cross the inference-engine boundary.

/// Convenient result alias for inference-engine operations.
pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

/// Failure reported by an inference engine to its caller.
#[derive(Debug)]
pub enum InferenceError {
    /// A CUDA operation or device-side validation failed during inference.
    Cuda(eider_cuda::Error),
    /// A checkpoint or derived host artifact was malformed.
    Format(eider_format::Error),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(error) => Some(error),
            Self::Format(error) => Some(error),
        }
    }
}

impl From<eider_cuda::Error> for InferenceError {
    fn from(error: eider_cuda::Error) -> Self {
        Self::Cuda(error)
    }
}

impl From<eider_format::Error> for InferenceError {
    fn from(error: eider_format::Error) -> Self {
        Self::Format(error)
    }
}

#[cfg(test)]
mod tests {
    use super::InferenceError;
    use std::error::Error as _;

    #[test]
    fn preserves_format_error_identity() {
        let error = InferenceError::from(eider_format::Error::Shape {
            label: "test record",
            expected: "K16 dimensions".to_string(),
            actual: "K15 dimensions".to_string(),
        });
        assert!(matches!(error, InferenceError::Format(_)));
        assert!(error.source().is_some());
    }
}
