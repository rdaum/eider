//! Deferred command submission with resource ownership through completion.
//!
//! A [`Recording`] owns every resource referenced by its backend encoder.
//! Submission moves the encoder and resources into [`InFlight`], which does
//! not expose the resources until the backend observes completion. This makes
//! asynchronous resource reuse a Rust ownership transition rather than a
//! caller convention.

use core::convert::Infallible;
use core::num::NonZeroUsize;

mod cuda;
mod slots;

pub use cuda::{CudaBackend, CudaFence, CudaPass};
pub use slots::{BoundedExecutionSlots, ExecutionSlotStatus, SlotSubmitError};

/// Completion state reported by a deferred backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionStatus {
    /// Submitted work has not completed.
    Pending,
    /// Submitted work has completed and its resources can be reclaimed.
    Complete,
}

/// Limit for grouping ordered recording units into one submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionGroupPolicy {
    max_groups: NonZeroUsize,
}

impl SubmissionGroupPolicy {
    /// Creates a policy that submits after `max_groups` recording units.
    pub const fn new(max_groups: NonZeroUsize) -> Self {
        Self { max_groups }
    }

    /// Returns the maximum number of recording units per submission.
    pub const fn max_groups(self) -> usize {
        self.max_groups.get()
    }

    /// Creates a recording cursor for this policy.
    pub const fn grouping(self) -> SubmissionGrouping {
        SubmissionGrouping {
            policy: self,
            pending_groups: 0,
        }
    }
}

/// Per-recording cursor for a [`SubmissionGroupPolicy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionGrouping {
    policy: SubmissionGroupPolicy,
    pending_groups: usize,
}

impl SubmissionGrouping {
    /// Records one unit and reports whether the submission boundary was met.
    pub fn record_group(&mut self) -> bool {
        self.pending_groups += 1;
        if self.pending_groups == self.policy.max_groups() {
            self.pending_groups = 0;
            true
        } else {
            false
        }
    }

    /// Consumes and reports a non-empty tail below the normal boundary.
    pub fn flush(&mut self) -> bool {
        core::mem::take(&mut self.pending_groups) != 0
    }

    /// Returns the number of recording units in the current tail.
    pub const fn pending_groups(&self) -> usize {
        self.pending_groups
    }
}

/// Recording, submission, polling, and wait operations for one ordered domain.
///
/// Backends whose encoders retain commands can discard an unsubmitted encoder
/// with the default [`Self::discard`] implementation. An eager backend such as
/// a CUDA stream must override `discard` and return a fence covering work that
/// the encoder already enqueued.
///
/// Submissions from one backend value must be ordered. A later fence must cover
/// every earlier submission in that execution domain.
///
/// # Safety
///
/// A fence returned by `submit` or `discard` must cover every backend access
/// to the recording's resources. `Complete` from `poll` and success from
/// `wait` must mean that no covered work can access those resources again.
/// Backend clones must refer to the same ordered execution domain.
pub unsafe trait DeferredBackend: Clone {
    /// Backend command encoder or eager execution pass.
    type Encoder;
    /// Completion fence for submitted commands.
    type Fence;
    /// Backend lifecycle error.
    type Error;

    /// Begins one recording.
    fn begin(&self, label: &'static str) -> Result<Self::Encoder, Self::Error>;

    /// Submits one completed recording and returns its completion fence.
    fn submit(&self, encoder: Self::Encoder) -> Self::Fence;

    /// Polls one submitted fence without blocking.
    fn poll(&self, fence: &mut Self::Fence) -> Result<CompletionStatus, Self::Error>;

    /// Waits for one submitted fence.
    fn wait(&self, fence: &mut Self::Fence) -> Result<(), Self::Error>;

    /// Discards a recording and returns a fence when work can still execute.
    fn discard(&self, encoder: Self::Encoder) -> Option<Self::Fence> {
        drop(encoder);
        None
    }
}

/// Resources paired with a backend encoder before submission.
///
/// `resources` must own or borrow every allocation referenced by commands in
/// `encoder`. Dropping a recording safely discards deferred commands. For an
/// eager backend, drop waits until already-enqueued work completes.
pub struct Recording<B: DeferredBackend, R> {
    backend: B,
    encoder: Option<B::Encoder>,
    resources: Option<R>,
    latest_segment: Option<B::Fence>,
}

impl<B: DeferredBackend, R> Recording<B, R> {
    /// Begins a recording without losing `resources` if construction fails.
    pub fn try_new(backend: B, resources: R, label: &'static str) -> Result<Self, (B::Error, R)> {
        let encoder = match backend.begin(label) {
            Ok(encoder) => encoder,
            Err(error) => return Err((error, resources)),
        };
        Ok(Self {
            backend,
            encoder: Some(encoder),
            resources: Some(resources),
            latest_segment: None,
        })
    }

    /// Begins a recording.
    pub fn new(backend: B, resources: R, label: &'static str) -> Result<Self, B::Error> {
        Self::try_new(backend, resources, label).map_err(|(error, _resources)| error)
    }

    /// Borrows the encoder and its retained resources together.
    pub fn parts_mut(&mut self) -> (&mut B::Encoder, &mut R) {
        (
            self.encoder
                .as_mut()
                .expect("active recording owns its encoder"),
            self.resources
                .as_mut()
                .expect("active recording owns its resources"),
        )
    }

    /// Records work while borrowing the encoder and resources together.
    pub fn record<T>(&mut self, record: impl FnOnce(&mut B::Encoder, &mut R) -> T) -> T {
        let (encoder, resources) = self.parts_mut();
        record(encoder, resources)
    }

    /// Returns the retained resources without exposing mutable access.
    pub fn resources(&self) -> &R {
        self.resources
            .as_ref()
            .expect("active recording owns its resources")
    }

    /// Submits the current segment and begins the next ordered segment.
    ///
    /// Encoder creation happens before submission. A construction failure
    /// therefore leaves the current recording intact.
    pub fn submit_segment(&mut self, next_label: &'static str) -> Result<&B::Fence, B::Error> {
        let next = self.backend.begin(next_label)?;
        let current = self
            .encoder
            .replace(next)
            .expect("active recording owns its encoder");
        self.latest_segment = Some(self.backend.submit(current));
        Ok(self
            .latest_segment
            .as_ref()
            .expect("successful segment submission retains its fence"))
    }

    /// Returns the latest submitted segment fence, when one exists.
    pub fn latest_segment(&self) -> Option<&B::Fence> {
        self.latest_segment.as_ref()
    }

    /// Discards the current encoder and preserves resources behind any fence.
    pub fn discard(mut self) -> DiscardedRecording<B, R> {
        let encoder = self
            .encoder
            .take()
            .expect("active recording owns its encoder");
        let resources = self
            .resources
            .take()
            .expect("active recording owns its resources");
        let fence = self
            .backend
            .discard(encoder)
            .or_else(|| self.latest_segment.take());
        match fence {
            Some(fence) => {
                DiscardedRecording::InFlight(InFlight::new(self.backend.clone(), fence, resources))
            }
            None => DiscardedRecording::Ready(resources),
        }
    }

    /// Submits this recording and transfers its resources into an owner.
    pub fn submit(mut self) -> InFlight<B, R> {
        let encoder = self
            .encoder
            .take()
            .expect("active recording owns its encoder");
        let resources = self
            .resources
            .take()
            .expect("active recording owns its resources");
        let fence = self.backend.submit(encoder);
        InFlight::new(self.backend.clone(), fence, resources)
    }
}

impl<B: DeferredBackend, R> Drop for Recording<B, R> {
    fn drop(&mut self) {
        let (Some(encoder), Some(resources)) = (self.encoder.take(), self.resources.take()) else {
            return;
        };
        let fence = self
            .backend
            .discard(encoder)
            .or_else(|| self.latest_segment.take());
        if let Some(mut fence) = fence
            && self.backend.wait(&mut fence).is_err()
        {
            // Releasing resources after an unknown completion state can create
            // a host/device race. There is no safe recovery during Drop.
            std::process::abort();
        }
        drop(resources);
    }
}

/// Resources recovered from a discarded recording.
pub enum DiscardedRecording<B: DeferredBackend, R> {
    /// No recorded work can execute, so resources are immediately available.
    Ready(R),
    /// Work can still execute, so resources remain behind a completion fence.
    InFlight(InFlight<B, R>),
}

/// Submitted resources that cannot be reused before completion.
pub struct InFlight<B: DeferredBackend, R> {
    backend: B,
    fence: B::Fence,
    resources: Option<R>,
    complete: bool,
}

impl<B: DeferredBackend, R> InFlight<B, R> {
    fn new(backend: B, fence: B::Fence, resources: R) -> Self {
        Self {
            backend,
            fence,
            resources: Some(resources),
            complete: false,
        }
    }

    /// Polls the backend and remembers an observed completion.
    pub fn poll(&mut self) -> Result<CompletionStatus, B::Error> {
        if self.complete {
            return Ok(CompletionStatus::Complete);
        }
        let status = self.backend.poll(&mut self.fence)?;
        self.complete = status == CompletionStatus::Complete;
        Ok(status)
    }

    /// Waits for completion while retaining resources if the wait fails.
    pub fn wait(&mut self) -> Result<(), B::Error> {
        if !self.complete {
            self.backend.wait(&mut self.fence)?;
            self.complete = true;
        }
        Ok(())
    }

    /// Reports whether completion has been observed.
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// Recovers resources only after completion was observed.
    pub fn try_reclaim(mut self) -> Result<R, Self> {
        if !self.complete {
            return Err(self);
        }
        Ok(self
            .resources
            .take()
            .expect("in-flight owner retains resources until reclamation"))
    }
}

impl<B: DeferredBackend, R> Drop for InFlight<B, R> {
    fn drop(&mut self) {
        if !self.complete && self.backend.wait(&mut self.fence).is_err() {
            // The retained resources must not be destroyed while backend work
            // can still access them.
            std::process::abort();
        }
        self.complete = true;
    }
}

/// Inline backend whose work completes while it is recorded.
#[derive(Clone, Copy, Debug, Default)]
pub struct InlineBackend;

// Inline work finishes during the exclusive recording borrow.
unsafe impl DeferredBackend for InlineBackend {
    type Encoder = ();
    type Fence = ();
    type Error = Infallible;

    fn begin(&self, _label: &'static str) -> Result<Self::Encoder, Self::Error> {
        Ok(())
    }

    fn submit(&self, _encoder: Self::Encoder) -> Self::Fence {}

    fn poll(&self, _fence: &mut Self::Fence) -> Result<CompletionStatus, Self::Error> {
        Ok(CompletionStatus::Complete)
    }

    fn wait(&self, _fence: &mut Self::Fence) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Clone, Default)]
    struct EagerDropBackend {
        waits: Arc<AtomicUsize>,
    }

    // The manual fence is completed by wait and covers the empty encoder.
    unsafe impl DeferredBackend for EagerDropBackend {
        type Encoder = ();
        type Fence = ();
        type Error = Infallible;

        fn begin(&self, _label: &'static str) -> Result<Self::Encoder, Self::Error> {
            Ok(())
        }

        fn submit(&self, _encoder: Self::Encoder) -> Self::Fence {}

        fn poll(&self, _fence: &mut Self::Fence) -> Result<CompletionStatus, Self::Error> {
            Ok(CompletionStatus::Pending)
        }

        fn wait(&self, _fence: &mut Self::Fence) -> Result<(), Self::Error> {
            self.waits.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn discard(&self, encoder: Self::Encoder) -> Option<Self::Fence> {
            Some(self.submit(encoder))
        }
    }

    #[test]
    fn inline_submission_retains_resources_until_reclaim() {
        let mut recording = Recording::new(InlineBackend, vec![1_u32, 2, 3], "inline").unwrap();
        recording.record(|(), resources| resources[1] = 7);

        let mut submission = recording.submit();
        assert!(!submission.is_complete());
        assert_eq!(submission.poll().unwrap(), CompletionStatus::Complete);
        assert_eq!(submission.try_reclaim().ok().unwrap(), [1, 7, 3]);
    }

    #[test]
    fn submission_grouping_flushes_only_non_empty_tails() {
        let policy = SubmissionGroupPolicy::new(NonZeroUsize::new(2).unwrap());
        let mut grouping = policy.grouping();
        assert!(!grouping.record_group());
        assert!(grouping.flush());
        assert!(!grouping.flush());
        assert!(!grouping.record_group());
        assert!(grouping.record_group());
        assert_eq!(grouping.pending_groups(), 0);
    }

    #[test]
    fn segmented_recording_retains_resources_behind_latest_fence() {
        let mut recording = Recording::new(InlineBackend, vec![1_u32], "first").unwrap();
        recording.record(|(), resources| resources.push(2));
        recording.submit_segment("second").unwrap();
        recording.record(|(), resources| resources.push(3));

        let mut submission = recording.submit();
        submission.wait().unwrap();
        assert_eq!(submission.try_reclaim().ok().unwrap(), [1, 2, 3]);
    }

    #[test]
    fn dropping_eager_owners_waits_before_releasing_resources() {
        let backend = EagerDropBackend::default();
        drop(Recording::new(backend.clone(), 3_u32, "recording").unwrap());
        assert_eq!(backend.waits.load(Ordering::Relaxed), 1);

        let submission = Recording::new(backend.clone(), 5_u32, "submission")
            .unwrap()
            .submit();
        drop(submission);
        assert_eq!(backend.waits.load(Ordering::Relaxed), 2);
    }
}
