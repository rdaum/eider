//! Model-independent request scheduling policy.

use crate::sampling::SamplingConfig;
use eider_format::{Error, Result};
use std::collections::BTreeSet;

/// Request lifecycle visible to a serving frontend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    /// Accepted on the CPU but not yet allocated persistent inference state.
    Waiting,
    /// Consuming all but the final prompt token into persistent model state.
    Prefilling,
    /// Producing completion tokens.
    Decoding,
    /// Reached EOS or the requested completion length.
    Finished,
}

/// Execution and admission limits for one scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    /// Maximum independent rows in one latency-sensitive decode batch.
    pub decode_capacity: usize,
    /// Maximum independent prompt chunks in one prefill batch.
    pub prefill_sequence_capacity: usize,
    /// Maximum total prompt tokens consumed by one prefill batch.
    pub prefill_token_capacity: usize,
    /// Maximum requests with allocated persistent inference state.
    pub max_active_sequences: usize,
    /// Maximum prompt plus completion tokens for any request.
    pub max_context_tokens: usize,
    /// Draft tokens verified per speculative cycle; zero disables speculation.
    pub speculative_drafts: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            decode_capacity: 8,
            prefill_sequence_capacity: 8,
            prefill_token_capacity: 2_048,
            max_active_sequences: 8,
            max_context_tokens: 32_768,
            speculative_drafts: 0,
        }
    }
}

impl SchedulerConfig {
    /// Validates model-independent execution and admission limits.
    pub fn validate(self) -> Result<()> {
        if self.decode_capacity == 0
            || self.prefill_sequence_capacity == 0
            || self.prefill_token_capacity == 0
            || self.max_active_sequences == 0
            || self.max_context_tokens == 0
        {
            return Err(Error::Shape {
                label: "scheduler configuration",
                expected: "all capacities greater than zero".to_string(),
                actual: format!(
                    "decode={} prefill_sequences={} prefill_tokens={} active={} context={}",
                    self.decode_capacity,
                    self.prefill_sequence_capacity,
                    self.prefill_token_capacity,
                    self.max_active_sequences,
                    self.max_context_tokens
                ),
            });
        }
        Ok(())
    }
}

/// Token-level generation policy for a scheduled request.
#[derive(Clone, Debug)]
pub struct RequestConfig {
    /// Token selection policy.
    pub sampling: SamplingConfig,
    /// Maximum number of completion tokens.
    pub max_new_tokens: usize,
    /// Model token IDs that terminate generation.
    pub eos_token_ids: BTreeSet<u32>,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            max_new_tokens: 64,
            eos_token_ids: BTreeSet::new(),
        }
    }
}

impl RequestConfig {
    /// Validates token-selection parameters.
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()
    }
}

/// Why a tokenized scheduled request completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestFinishReason {
    /// The model selected a configured EOS token.
    Eos,
    /// The request reached its completion-token limit.
    Length,
    /// A request-scoped tool grammar completed a function call.
    ToolCalls,
}

/// A lifecycle event emitted at the point where scheduler work begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestLifecycleEvent<RequestId, AdmissionProgress> {
    /// Persistent inference state is ready for the request.
    Admitted(AdmissionProgress),
    /// The request's next prompt chunk is about to enter the model.
    PrefillStarted(RequestId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_limits_reject_zero_capacity() {
        assert!(SchedulerConfig::default().validate().is_ok());
        assert!(
            SchedulerConfig {
                decode_capacity: 0,
                ..SchedulerConfig::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn request_policy_delegates_sampling_validation() {
        assert!(RequestConfig::default().validate().is_ok());
        assert!(
            RequestConfig {
                sampling: SamplingConfig {
                    top_p: 0.0,
                    ..SamplingConfig::default()
                },
                ..RequestConfig::default()
            }
            .validate()
            .is_err()
        );
    }
}
