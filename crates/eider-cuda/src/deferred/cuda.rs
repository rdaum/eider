//! CUDA stream implementation of deferred execution ownership.

use crate::cuda::CudaStream;
use crate::error::{Error, Result};

use super::{CompletionStatus, DeferredBackend};

/// Deferred backend over one ordered CUDA stream.
///
/// CUDA launches are eager: recording through [`CudaPass`] immediately
/// enqueues work. Discarding a recording therefore returns a fence instead of
/// making its resources immediately available.
#[derive(Clone, Copy)]
pub struct CudaBackend<'stream> {
    stream: &'stream CudaStream,
}

impl<'stream> CudaBackend<'stream> {
    /// Creates a backend for one ordered CUDA stream.
    pub const fn new(stream: &'stream CudaStream) -> Self {
        Self { stream }
    }
}

/// Exclusive recording capability for one CUDA stream.
pub struct CudaPass<'stream> {
    stream: &'stream CudaStream,
}

impl CudaPass<'_> {
    /// Borrows the CUDA stream for one launch.
    ///
    /// The returned borrow is tied to the pass borrow and cannot outlive the
    /// surrounding recording call.
    pub const fn stream(&self) -> &CudaStream {
        self.stream
    }
}

/// Conservative completion fence for a CUDA stream submission.
///
/// This zero-allocation fence queries the stream. Work enqueued after the
/// logical submission can delay completion observation, but cannot cause
/// resources to be reclaimed early.
pub struct CudaFence<'stream> {
    stream: &'stream CudaStream,
}

// Every CUDA pass enqueues on one ordered stream. Stream query/synchronisation
// conservatively cover all work that could still access retained resources.
unsafe impl<'stream> DeferredBackend for CudaBackend<'stream> {
    type Encoder = CudaPass<'stream>;
    type Fence = CudaFence<'stream>;
    type Error = Error;

    fn begin(&self, _label: &'static str) -> Result<Self::Encoder> {
        Ok(CudaPass {
            stream: self.stream,
        })
    }

    fn submit(&self, _encoder: Self::Encoder) -> Self::Fence {
        CudaFence {
            stream: self.stream,
        }
    }

    fn poll(&self, fence: &mut Self::Fence) -> Result<CompletionStatus> {
        fence.stream.query().map(|complete| {
            if complete {
                CompletionStatus::Complete
            } else {
                CompletionStatus::Pending
            }
        })
    }

    fn wait(&self, fence: &mut Self::Fence) -> Result<()> {
        fence.stream.synchronize()
    }

    fn discard(&self, encoder: Self::Encoder) -> Option<Self::Fence> {
        Some(self.submit(encoder))
    }
}
