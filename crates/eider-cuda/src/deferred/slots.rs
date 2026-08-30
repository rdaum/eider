//! Bounded reusable resource slots for deferred submissions.

use core::fmt;

use super::{CompletionStatus, DeferredBackend, DiscardedRecording, InFlight, Recording};

/// Observable lifecycle of one bounded execution slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionSlotStatus {
    /// Resources are available for a new recording.
    Ready,
    /// Resources are retained by submitted or eagerly enqueued work.
    InFlight,
    /// Completion was observed and the completion callback has not run.
    Completed,
}

enum SlotState<B: DeferredBackend, R> {
    Ready(R),
    InFlight(InFlight<B, R>),
    Completed(R),
    Transition,
}

struct RecordingRestore<'a, B: DeferredBackend, R> {
    state: &'a mut SlotState<B, R>,
    recording: Option<Recording<B, R>>,
}

impl<B: DeferredBackend, R> Drop for RecordingRestore<'_, B, R> {
    fn drop(&mut self) {
        let Some(recording) = self.recording.take() else {
            return;
        };
        *self.state = match recording.discard() {
            DiscardedRecording::Ready(resources) => SlotState::Ready(resources),
            DiscardedRecording::InFlight(submission) => SlotState::InFlight(submission),
        };
    }
}

/// Failure while beginning or populating a bounded execution slot.
#[derive(Debug)]
pub enum SlotSubmitError<BackendError, RecordError> {
    /// The backend could not begin a recording.
    Begin(BackendError),
    /// The recording closure rejected the operation.
    Record(RecordError),
}

impl<BackendError, RecordError> fmt::Display for SlotSubmitError<BackendError, RecordError>
where
    BackendError: fmt::Display,
    RecordError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Begin(error) => write!(formatter, "execution recording failed to begin: {error}"),
            Self::Record(error) => write!(formatter, "execution recording failed: {error}"),
        }
    }
}

impl<BackendError, RecordError> std::error::Error for SlotSubmitError<BackendError, RecordError>
where
    BackendError: std::error::Error + 'static,
    RecordError: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Begin(error) => Some(error),
            Self::Record(error) => Some(error),
        }
    }
}

/// Fixed-capacity ownership state machine for reusable submission resources.
///
/// Recording failure and unwinding restore deferred encoders immediately. For
/// eager backends, already-enqueued work remains in flight until completion.
pub struct BoundedExecutionSlots<B: DeferredBackend, R> {
    backend: B,
    slots: Vec<SlotState<B, R>>,
}

impl<B: DeferredBackend, R> BoundedExecutionSlots<B, R> {
    /// Creates a non-empty slot set from reusable resources.
    pub fn new(backend: B, resources: impl IntoIterator<Item = R>) -> Result<Self, &'static str> {
        let slots = resources
            .into_iter()
            .map(SlotState::Ready)
            .collect::<Vec<_>>();
        if slots.is_empty() {
            return Err("bounded execution slots require at least one resource set");
        }
        Ok(Self { backend, slots })
    }

    /// Returns the fixed slot count.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Reports whether the slot set is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Returns one slot's current lifecycle state.
    pub fn status(&self, index: usize) -> Option<ExecutionSlotStatus> {
        self.slots.get(index).map(|state| match state {
            SlotState::Ready(_) => ExecutionSlotStatus::Ready,
            SlotState::InFlight(_) => ExecutionSlotStatus::InFlight,
            SlotState::Completed(_) => ExecutionSlotStatus::Completed,
            SlotState::Transition => unreachable!("slot transition cannot escape a method"),
        })
    }

    /// Returns the number of ready slots.
    pub fn ready_count(&self) -> usize {
        self.count(ExecutionSlotStatus::Ready)
    }

    /// Returns the number of in-flight slots.
    pub fn in_flight_count(&self) -> usize {
        self.count(ExecutionSlotStatus::InFlight)
    }

    /// Returns the number of completed slots awaiting recycling.
    pub fn completed_count(&self) -> usize {
        self.count(ExecutionSlotStatus::Completed)
    }

    /// Records and submits into the first ready slot.
    ///
    /// `Ok(None)` is bounded backpressure: no slot can be reused yet.
    pub fn try_submit<RecordError>(
        &mut self,
        label: &'static str,
        record: impl FnOnce(&mut B::Encoder, &mut R) -> Result<(), RecordError>,
    ) -> Result<Option<usize>, SlotSubmitError<B::Error, RecordError>> {
        let Some(index) = self
            .slots
            .iter()
            .position(|state| matches!(state, SlotState::Ready(_)))
        else {
            return Ok(None);
        };
        let state = &mut self.slots[index];
        let SlotState::Ready(resources) = core::mem::replace(state, SlotState::Transition) else {
            unreachable!("selected slot was ready")
        };
        let recording = match Recording::try_new(self.backend.clone(), resources, label) {
            Ok(recording) => recording,
            Err((error, resources)) => {
                *state = SlotState::Ready(resources);
                return Err(SlotSubmitError::Begin(error));
            }
        };
        let mut restore = RecordingRestore {
            state,
            recording: Some(recording),
        };
        let result = restore
            .recording
            .as_mut()
            .expect("recording restore owns the recording")
            .record(record);
        if let Err(error) = result {
            return Err(SlotSubmitError::Record(error));
        }
        let recording = restore
            .recording
            .take()
            .expect("successful recording remains owned");
        *restore.state = SlotState::InFlight(recording.submit());
        Ok(Some(index))
    }

    /// Polls every in-flight slot once and marks newly completed slots.
    pub fn poll(&mut self) -> Result<usize, B::Error> {
        let mut completed = 0;
        for index in 0..self.slots.len() {
            let is_complete = match &mut self.slots[index] {
                SlotState::InFlight(submission) => submission.poll()? == CompletionStatus::Complete,
                _ => false,
            };
            if !is_complete {
                continue;
            }
            let SlotState::InFlight(submission) =
                core::mem::replace(&mut self.slots[index], SlotState::Transition)
            else {
                unreachable!("completed slot was in flight")
            };
            let resources = submission
                .try_reclaim()
                .ok()
                .expect("completion was observed before reclamation");
            self.slots[index] = SlotState::Completed(resources);
            completed += 1;
        }
        Ok(completed)
    }

    /// Waits for one in-flight slot and marks it completed.
    ///
    /// Returns `false` when the index is absent or not in flight.
    pub fn wait(&mut self, index: usize) -> Result<bool, B::Error> {
        let Some(state) = self.slots.get_mut(index) else {
            return Ok(false);
        };
        let SlotState::InFlight(submission) = state else {
            return Ok(false);
        };
        submission.wait()?;
        let SlotState::InFlight(submission) = core::mem::replace(state, SlotState::Transition)
        else {
            unreachable!("waited slot remained in flight")
        };
        let resources = submission
            .try_reclaim()
            .ok()
            .expect("wait observed completion before reclamation");
        *state = SlotState::Completed(resources);
        Ok(true)
    }

    /// Handles and recycles one completed slot.
    ///
    /// The slot becomes ready before `complete` runs, so unwinding cannot
    /// strand its resource owner.
    pub fn recycle_completed<T>(
        &mut self,
        index: usize,
        complete: impl FnOnce(&mut R) -> T,
    ) -> Option<T> {
        let state = self.slots.get_mut(index)?;
        let resources = match core::mem::replace(state, SlotState::Transition) {
            SlotState::Completed(resources) => resources,
            other => {
                *state = other;
                return None;
            }
        };
        *state = SlotState::Ready(resources);
        let SlotState::Ready(resources) = state else {
            unreachable!("completed slot was recycled to ready")
        };
        Some(complete(resources))
    }

    fn count(&self, status: ExecutionSlotStatus) -> usize {
        self.slots
            .iter()
            .filter(|state| {
                matches!(
                    (status, state),
                    (ExecutionSlotStatus::Ready, SlotState::Ready(_))
                        | (ExecutionSlotStatus::InFlight, SlotState::InFlight(_))
                        | (ExecutionSlotStatus::Completed, SlotState::Completed(_))
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::deferred::InlineBackend;

    #[derive(Clone, Default)]
    struct ManualBackend {
        fences: Arc<Mutex<Vec<Arc<AtomicBool>>>>,
        fail_begin: Arc<AtomicBool>,
    }

    // Submitted test encoders are covered by their explicit atomic fence.
    unsafe impl DeferredBackend for ManualBackend {
        type Encoder = Vec<u32>;
        type Fence = Arc<AtomicBool>;
        type Error = &'static str;

        fn begin(&self, _label: &'static str) -> Result<Self::Encoder, Self::Error> {
            if self.fail_begin.swap(false, Ordering::AcqRel) {
                Err("injected begin failure")
            } else {
                Ok(Vec::new())
            }
        }

        fn submit(&self, _encoder: Self::Encoder) -> Self::Fence {
            let fence = Arc::new(AtomicBool::new(false));
            self.fences.lock().unwrap().push(Arc::clone(&fence));
            fence
        }

        fn poll(&self, fence: &mut Self::Fence) -> Result<CompletionStatus, Self::Error> {
            Ok(if fence.load(Ordering::Acquire) {
                CompletionStatus::Complete
            } else {
                CompletionStatus::Pending
            })
        }

        fn wait(&self, fence: &mut Self::Fence) -> Result<(), Self::Error> {
            fence.store(true, Ordering::Release);
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct ManualEagerBackend(ManualBackend);

    // Discard submits the eager encoder behind the same explicit fence.
    unsafe impl DeferredBackend for ManualEagerBackend {
        type Encoder = Vec<u32>;
        type Fence = Arc<AtomicBool>;
        type Error = &'static str;

        fn begin(&self, label: &'static str) -> Result<Self::Encoder, Self::Error> {
            self.0.begin(label)
        }

        fn submit(&self, encoder: Self::Encoder) -> Self::Fence {
            self.0.submit(encoder)
        }

        fn poll(&self, fence: &mut Self::Fence) -> Result<CompletionStatus, Self::Error> {
            self.0.poll(fence)
        }

        fn wait(&self, fence: &mut Self::Fence) -> Result<(), Self::Error> {
            self.0.wait(fence)
        }

        fn discard(&self, encoder: Self::Encoder) -> Option<Self::Fence> {
            Some(self.submit(encoder))
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Resources {
        identity: usize,
        generation: usize,
    }

    #[test]
    fn slots_apply_backpressure_and_recycle_observed_completion() {
        let backend = ManualBackend::default();
        let mut slots = BoundedExecutionSlots::new(
            backend.clone(),
            [
                Resources {
                    identity: 0,
                    generation: 0,
                },
                Resources {
                    identity: 1,
                    generation: 0,
                },
            ],
        )
        .unwrap();
        for expected in 0..2 {
            assert_eq!(
                slots
                    .try_submit("manual", |encoder, resources| {
                        assert_eq!(resources.identity, expected);
                        resources.generation += 1;
                        encoder.push(resources.identity as u32);
                        Ok::<_, Infallible>(())
                    })
                    .unwrap(),
                Some(expected)
            );
        }
        assert_eq!(
            slots
                .try_submit("full", |_encoder, _resources| Ok::<_, Infallible>(()))
                .unwrap(),
            None
        );
        backend.fences.lock().unwrap()[1].store(true, Ordering::Release);
        assert_eq!(slots.poll().unwrap(), 1);
        assert_eq!(slots.status(1), Some(ExecutionSlotStatus::Completed));
        assert_eq!(
            slots.recycle_completed(1, |resources| *resources),
            Some(Resources {
                identity: 1,
                generation: 1,
            })
        );
        assert!(slots.wait(0).unwrap());
        assert_eq!(slots.status(0), Some(ExecutionSlotStatus::Completed));
    }

    #[test]
    fn failures_restore_deferred_resources() {
        let backend = ManualBackend::default();
        let mut slots = BoundedExecutionSlots::new(
            backend.clone(),
            [Resources {
                identity: 7,
                generation: 0,
            }],
        )
        .unwrap();
        backend.fail_begin.store(true, Ordering::Release);
        assert!(matches!(
            slots.try_submit("begin", |_encoder, _resources| Ok::<_, Infallible>(())),
            Err(SlotSubmitError::Begin("injected begin failure"))
        ));
        assert_eq!(slots.status(0), Some(ExecutionSlotStatus::Ready));
        assert!(matches!(
            slots.try_submit("record", |_encoder, _resources| Err::<(), _>(
                "record failure"
            )),
            Err(SlotSubmitError::Record("record failure"))
        ));
        assert_eq!(slots.status(0), Some(ExecutionSlotStatus::Ready));
    }

    #[test]
    fn eager_recording_failure_retains_resources_until_completion() {
        let backend = ManualEagerBackend::default();
        let mut slots = BoundedExecutionSlots::new(
            backend.clone(),
            [Resources {
                identity: 11,
                generation: 0,
            }],
        )
        .unwrap();
        assert!(matches!(
            slots.try_submit("eager", |encoder, resources| {
                encoder.push(resources.identity as u32);
                resources.generation += 1;
                Err::<(), _>("record failure")
            }),
            Err(SlotSubmitError::Record("record failure"))
        ));
        assert_eq!(slots.status(0), Some(ExecutionSlotStatus::InFlight));
        backend.0.fences.lock().unwrap()[0].store(true, Ordering::Release);
        assert_eq!(slots.poll().unwrap(), 1);
        assert_eq!(
            slots.recycle_completed(0, |resources| *resources),
            Some(Resources {
                identity: 11,
                generation: 1,
            })
        );
    }

    #[test]
    fn inline_backend_uses_the_same_slot_lifecycle() {
        let mut slots = BoundedExecutionSlots::new(InlineBackend, [3_u32]).unwrap();
        assert_eq!(
            slots
                .try_submit("inline", |(), resource| {
                    *resource += 1;
                    Ok::<_, Infallible>(())
                })
                .unwrap(),
            Some(0)
        );
        assert_eq!(slots.poll().unwrap(), 1);
        assert_eq!(slots.recycle_completed(0, |resource| *resource), Some(4));
    }
}
