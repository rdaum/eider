//! Model-neutral records exchanged between a serving actor and an inference engine.
//!
//! These records describe request lifecycle and compact output only. They do
//! not expose model buffers, streams, logits, or model-specific sequence IDs.

use crate::chat_output::ChatOutputEvent;
use crate::request::{ChatFinishReason, ChatUsage};
use std::time::Duration;

/// Request metadata known once an inference engine accepts a request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineAdmission {
    /// Engine-local request identity used for subsequent lifecycle operations.
    pub request_id: u64,
    /// Rendered prompt token count.
    pub prompt_tokens: usize,
    /// Requested completion-token limit.
    pub max_output_tokens: usize,
}

/// Persistent sequence state allocated during a scheduler tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineAdmissionProgress {
    /// Engine-local request identity.
    pub request_id: u64,
    /// Device bytes retained for this request's sequence state.
    pub sequence_device_bytes: usize,
    /// Prompt tokens restored from a retained prefix.
    pub cached_prompt_tokens: usize,
    /// Time spent allocating the active sequence.
    pub allocation_duration: Duration,
    /// Time spent restoring a retained prefix checkpoint.
    pub checkpoint_copy_duration: Duration,
    /// Elapsed scheduler-tick time when admission completed.
    pub admitted_after_tick_start: Duration,
}

/// Lifecycle transition emitted by an inference engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineLifecycleEvent {
    /// Persistent sequence state is ready for a request.
    Admitted(EngineAdmissionProgress),
    /// The request's next prompt chunk is about to enter model execution.
    PrefillStarted(u64),
}

/// Prompt work completed during one engine tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnginePrefillProgress {
    /// Engine-local request identity.
    pub request_id: u64,
    /// Total prompt position after this tick.
    pub prompt_position: usize,
}

/// One compact output event emitted by an engine.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineDelta {
    /// Engine-local request identity.
    pub request_id: u64,
    /// Decoded reasoning, text, or tool-call event.
    pub event: ChatOutputEvent,
}

/// Terminal request metadata emitted by an engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineFinished {
    /// Engine-local request identity.
    pub request_id: u64,
    /// Why generation completed.
    pub finish_reason: ChatFinishReason,
    /// Final token accounting.
    pub usage: ChatUsage,
    /// Sequence-specific bytes released at completion.
    pub released_sequence_device_bytes: usize,
}

/// Cumulative draft-and-verify telemetry for one request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineDraftStats {
    /// Completed draft-and-verify cycles.
    pub cycles: usize,
    /// Draft tokens proposed across all cycles.
    pub drafted_tokens: usize,
    /// Draft tokens accepted by the target.
    pub accepted_drafts: usize,
    /// Target-approved tokens emitted to the request.
    pub emitted_tokens: usize,
    /// Time spent in draft-and-verify cycles.
    pub cycle_duration: Duration,
    /// Latest retained target-model position.
    pub target_position: usize,
    /// Latest retained draft-model position.
    pub draft_position: usize,
}

/// Updated cumulative draft-and-verify telemetry from one engine tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineDraftProgress {
    /// Engine-local request identity.
    pub request_id: u64,
    /// Cumulative telemetry after the latest cycle.
    pub stats: EngineDraftStats,
}

/// Speculative verification progress from one engine tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineSpeculativeProgress {
    /// Engine-local request identity.
    pub request_id: u64,
    /// Verification cycles completed in this tick.
    pub cycles: usize,
    /// Draft tokens accepted in this tick.
    pub accepted_drafts: usize,
}

/// Compact engine work completed in one scheduler tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineTick {
    /// Prompt positions advanced during this tick.
    pub prefilled: Vec<EnginePrefillProgress>,
    /// Requests that emitted one generated token during this tick.
    pub generated: Vec<u64>,
    /// Speculative verifier progress.
    pub speculative: Vec<EngineSpeculativeProgress>,
    /// Draft-and-verify telemetry.
    pub dflash: Vec<EngineDraftProgress>,
    /// Decoded output events.
    pub output: Vec<EngineDelta>,
    /// Completed requests.
    pub finished: Vec<EngineFinished>,
    /// Number of live model sequences after the tick.
    pub active_sequences: usize,
}

/// Result of a cancellation request sent to an inference engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineCancelOutcome {
    /// The engine released sequence-specific state.
    Cancelled {
        /// Bytes released with the sequence state.
        released_sequence_device_bytes: usize,
    },
    /// The request had already completed.
    AlreadyFinished,
    /// The engine did not retain the request identity.
    NotFound,
}
