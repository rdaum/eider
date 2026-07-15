//! Tokenized Qwen3.6 scheduling over chunked prefill and batched decode.

use super::sampling::{SampledToken, Sampler, SamplingConfig, TokenHistory};
use crate::qwen3::qwen36::{
    Qwen36DecodeBatchWorkspace, Qwen36DecodeRow, Qwen36NextToken, Qwen36PrefillBatchWorkspace,
    Qwen36PrefillRow, Qwen36SequenceState, Qwen36TextModel,
};
use nvfp4::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Stable scheduler identity for one request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Qwen36RequestId(u64);

impl Qwen36RequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request lifecycle visible to a serving frontend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36RequestState {
    /// Accepted on the CPU but not yet allocated persistent GPU state.
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
pub struct Qwen36SchedulerConfig {
    /// Maximum independent rows in one latency-sensitive decode batch.
    pub decode_capacity: usize,
    /// Maximum independent prompt chunks in one prefill batch.
    pub prefill_sequence_capacity: usize,
    /// Maximum total prompt tokens consumed by one prefill batch.
    pub prefill_token_capacity: usize,
    /// Maximum requests with allocated persistent GPU state.
    pub max_active_sequences: usize,
    /// Maximum prompt plus completion tokens for any request.
    pub max_context_tokens: usize,
}

impl Default for Qwen36SchedulerConfig {
    fn default() -> Self {
        Self {
            decode_capacity: 8,
            prefill_sequence_capacity: 8,
            prefill_token_capacity: 128,
            max_active_sequences: 8,
            max_context_tokens: 32_768,
        }
    }
}

impl Qwen36SchedulerConfig {
    fn validate(self) -> Result<()> {
        if self.decode_capacity == 0
            || self.prefill_sequence_capacity == 0
            || self.prefill_token_capacity == 0
            || self.max_active_sequences == 0
            || self.max_context_tokens == 0
        {
            return Err(Error::Shape {
                label: "Qwen3.6 scheduler configuration",
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
pub struct Qwen36RequestConfig {
    /// Token selection policy.
    pub sampling: SamplingConfig,
    /// Maximum number of completion tokens.
    pub max_new_tokens: usize,
    /// Model token IDs that terminate generation.
    pub eos_token_ids: BTreeSet<u32>,
}

impl Default for Qwen36RequestConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            max_new_tokens: 64,
            eos_token_ids: BTreeSet::new(),
        }
    }
}

impl Qwen36RequestConfig {
    /// Validates token-selection parameters.
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()
    }
}

/// Why a tokenized scheduled request completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen36RequestFinishReason {
    /// The model selected a configured EOS token.
    Eos,
    /// The request reached its completion-token limit.
    Length,
}

/// One completion token produced by a scheduler tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36ScheduledToken {
    /// Request that owns the token.
    pub request_id: Qwen36RequestId,
    /// Selected vocabulary ID.
    pub id: u32,
    /// Original model logit for the selected ID.
    pub logit: f32,
    /// Present when this token finishes the request.
    pub finish_reason: Option<Qwen36RequestFinishReason>,
}

/// Prompt progress made for one request during a scheduler tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen36PrefillProgress {
    /// Request whose prompt state advanced.
    pub request_id: Qwen36RequestId,
    /// Prompt tokens consumed in this tick.
    pub tokens: usize,
    /// Total prompt tokens consumed after this tick.
    pub prompt_position: usize,
}

/// Observable result of one scheduler tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen36SchedulerTick {
    /// Requests selected for model work, with decode rows before prefill rows.
    pub scheduled: Vec<Qwen36RequestId>,
    /// Prompt progress made by the prefill batch.
    pub prefilled: Vec<Qwen36PrefillProgress>,
    /// Completion tokens produced after prompt consumption.
    pub generated: Vec<Qwen36ScheduledToken>,
    /// Requests that finished during this tick.
    pub finished: Vec<Qwen36RequestId>,
}

/// Result removed from the scheduler after a request finishes.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36FinishedRequest {
    /// Stable request identity.
    pub id: Qwen36RequestId,
    /// Original prompt tokens.
    pub prompt_tokens: Vec<u32>,
    /// Generated completion tokens and logits.
    pub generated_tokens: Vec<Qwen36ScheduledToken>,
    /// Final completion reason.
    pub finish_reason: Qwen36RequestFinishReason,
}

/// Request data returned when active or waiting work is cancelled.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36CancelledRequest {
    /// Stable request identity.
    pub id: Qwen36RequestId,
    /// Original prompt tokens.
    pub prompt_tokens: Vec<u32>,
    /// Completion tokens produced before cancellation.
    pub generated_tokens: Vec<Qwen36ScheduledToken>,
}

/// Outcome of a scheduler cancellation request.
#[derive(Clone, Debug, PartialEq)]
pub enum Qwen36CancelOutcome {
    /// Waiting or active work was removed and any GPU state was released.
    Cancelled(Qwen36CancelledRequest),
    /// The request had already reached a normal terminal state.
    AlreadyFinished,
    /// No retained request has this identity.
    NotFound,
}

struct Qwen36Request {
    id: Qwen36RequestId,
    lifecycle: Qwen36RequestState,
    config: Qwen36RequestConfig,
    prompt_tokens: Vec<u32>,
    prompt_position: usize,
    sequence: Option<Box<Qwen36SequenceState>>,
    sampler: Sampler,
    history: TokenHistory,
    last_token: Option<u32>,
    generated_tokens: Vec<Qwen36ScheduledToken>,
    finish_reason: Option<Qwen36RequestFinishReason>,
}

impl Qwen36Request {
    fn max_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.config.max_new_tokens
    }

    fn remaining_prompt_tokens(&self) -> usize {
        self.prompt_tokens.len() - self.prompt_position
    }

    fn prefillable_tokens(&self) -> usize {
        self.remaining_prompt_tokens().saturating_sub(1)
    }

    fn decode_input_token(&self) -> Result<u32> {
        if self.remaining_prompt_tokens() == 1 {
            return Ok(self.prompt_tokens[self.prompt_position]);
        }
        self.last_token.ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduled request",
            detail: format!("request {} has no decode input token", self.id.get()),
        })
    }

    fn apply_sample(&mut self, sampled: SampledToken) -> Qwen36ScheduledToken {
        if self.remaining_prompt_tokens() == 1 {
            self.prompt_position += 1;
        }
        self.last_token = Some(sampled.id);
        self.history.push(sampled.id);
        let generated_count = self.generated_tokens.len() + 1;
        let finish_reason = if self.config.eos_token_ids.contains(&sampled.id) {
            Some(Qwen36RequestFinishReason::Eos)
        } else if generated_count == self.config.max_new_tokens {
            Some(Qwen36RequestFinishReason::Length)
        } else {
            None
        };
        self.lifecycle = if finish_reason.is_some() {
            Qwen36RequestState::Finished
        } else {
            Qwen36RequestState::Decoding
        };
        self.finish_reason = finish_reason;
        let token = Qwen36ScheduledToken {
            request_id: self.id,
            id: sampled.id,
            logit: sampled.logit,
            finish_reason,
        };
        self.generated_tokens.push(token);
        token
    }
}

/// Decode-first continuous scheduler with deferred GPU admission.
pub struct Qwen36Scheduler<'model> {
    model: &'model Qwen36TextModel,
    config: Qwen36SchedulerConfig,
    decode_workspace: Qwen36DecodeBatchWorkspace,
    prefill_workspace: Qwen36PrefillBatchWorkspace,
    requests: BTreeMap<Qwen36RequestId, Box<Qwen36Request>>,
    waiting: VecDeque<Qwen36RequestId>,
    prefilling: VecDeque<Qwen36RequestId>,
    decoding: VecDeque<Qwen36RequestId>,
    next_id: u64,
}

impl<'model> Qwen36Scheduler<'model> {
    /// Creates a scheduler with explicit execution and admission limits.
    pub fn new(model: &'model Qwen36TextModel, config: Qwen36SchedulerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            model,
            config,
            decode_workspace: model
                .new_decode_batch_workspace(config.decode_capacity, config.max_context_tokens)?,
            prefill_workspace: model.new_prefill_batch_workspace(
                config.prefill_sequence_capacity,
                config.prefill_token_capacity,
                config.max_context_tokens,
            )?,
            requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            decoding: VecDeque::new(),
            next_id: 0,
        })
    }

    /// Adds an already-tokenized request to the CPU-only waiting queue.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: Qwen36RequestConfig,
    ) -> Result<Qwen36RequestId> {
        config.validate()?;
        if prompt_tokens.is_empty() {
            return Err(Error::Format {
                label: "Qwen3.6 scheduler prompt",
                detail: "prompt must contain at least one token".to_string(),
            });
        }
        if let Some(&token) = prompt_tokens
            .iter()
            .find(|&&token| token as usize >= self.model.manifest().vocab)
        {
            return Err(Error::Shape {
                label: "Qwen3.6 scheduler prompt token",
                expected: format!("token < {}", self.model.manifest().vocab),
                actual: token.to_string(),
            });
        }
        let max_tokens = prompt_tokens
            .len()
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 scheduler request capacity",
                expected: "prompt + completion length without overflow".to_string(),
                actual: format!("{} + {}", prompt_tokens.len(), config.max_new_tokens),
            })?;
        if max_tokens > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Qwen3.6 scheduler request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: max_tokens.to_string(),
            });
        }
        let id = Qwen36RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduler request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let lifecycle = if config.max_new_tokens == 0 {
            Qwen36RequestState::Finished
        } else {
            Qwen36RequestState::Waiting
        };
        let finish_reason =
            (config.max_new_tokens == 0).then_some(Qwen36RequestFinishReason::Length);
        let sampler = Sampler::new(config.sampling)?;
        let history = TokenHistory::from_tokens(prompt_tokens.iter().copied());
        self.requests.insert(
            id,
            Box::new(Qwen36Request {
                id,
                lifecycle,
                config,
                prompt_tokens,
                prompt_position: 0,
                sequence: None,
                sampler,
                history,
                last_token: None,
                generated_tokens: Vec::new(),
                finish_reason,
            }),
        );
        if lifecycle == Qwen36RequestState::Waiting {
            self.waiting.push_back(id);
        }
        Ok(id)
    }

    /// Runs one decode-first scheduling iteration followed by bounded prefill.
    pub fn tick(&mut self) -> Result<Qwen36SchedulerTick> {
        self.admit_waiting()?;
        let mut tick = Qwen36SchedulerTick::default();
        self.run_decode_phase(&mut tick)?;
        self.run_prefill_phase(&mut tick)?;
        Ok(tick)
    }

    fn admit_waiting(&mut self) -> Result<()> {
        while self.active_sequence_count() < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self
                .requests
                .get_mut(&id)
                .expect("waiting request is retained");
            let sequence = match self.model.new_sequence_state(request.max_tokens().max(1)) {
                Ok(sequence) => sequence,
                Err(error) => {
                    self.waiting.push_front(id);
                    return Err(error);
                }
            };
            request.sequence = Some(Box::new(sequence));
            request.lifecycle = Qwen36RequestState::Prefilling;
            self.prefilling.push_back(id);
        }
        Ok(())
    }

    fn run_decode_phase(&mut self, tick: &mut Qwen36SchedulerTick) -> Result<()> {
        let mut selected = Vec::with_capacity(self.config.decode_capacity);
        while selected.len() < self.config.decode_capacity {
            let Some(id) = self.decoding.pop_front() else {
                break;
            };
            selected.push(
                self.requests
                    .remove(&id)
                    .expect("decoding request is retained"),
            );
        }
        let prefill_scan = self.prefilling.len();
        for _ in 0..prefill_scan {
            let Some(id) = self.prefilling.pop_front() else {
                break;
            };
            if selected.len() < self.config.decode_capacity
                && self.requests[&id].remaining_prompt_tokens() == 1
            {
                selected.push(
                    self.requests
                        .remove(&id)
                        .expect("prefilling request is retained"),
                );
            } else {
                self.prefilling.push_back(id);
            }
        }
        if selected.is_empty() {
            return Ok(());
        }
        tick.scheduled
            .extend(selected.iter().map(|request| request.id));
        let samples = match self.execute_decode(&mut selected) {
            Ok(samples) => samples,
            Err(error) => {
                for request in selected.into_iter().rev() {
                    let queue = if request.lifecycle == Qwen36RequestState::Decoding {
                        &mut self.decoding
                    } else {
                        &mut self.prefilling
                    };
                    queue.push_front(request.id);
                    self.requests.insert(request.id, request);
                }
                return Err(error);
            }
        };
        for (mut request, sample) in selected.into_iter().zip(samples) {
            let token = request.apply_sample(sample);
            tick.generated.push(token);
            if request.lifecycle == Qwen36RequestState::Finished {
                request.sequence.take();
                tick.finished.push(request.id);
            } else {
                self.decoding.push_back(request.id);
            }
            self.requests.insert(request.id, request);
        }
        Ok(())
    }

    fn execute_decode(&mut self, selected: &mut [Box<Qwen36Request>]) -> Result<Vec<SampledToken>> {
        let needs_host_logits = selected
            .iter()
            .any(|request| !request.sampler.config().uses_fast_argmax());
        let mut rows = selected
            .iter_mut()
            .map(|request| {
                let token_id = request.decode_input_token()?;
                let state = request
                    .sequence
                    .as_deref_mut()
                    .ok_or_else(|| Error::Format {
                        label: "Qwen3.6 scheduled decode",
                        detail: format!(
                            "request {} has no admitted sequence state",
                            request.id.get()
                        ),
                    })?;
                Ok(Qwen36DecodeRow { token_id, state })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut decoded = self
            .model
            .decode_batch(&mut self.decode_workspace, &mut rows)?;
        if !needs_host_logits {
            return decoded
                .top1()
                .map(|tokens| tokens.into_iter().map(sampled_top1).collect());
        }
        let vocab = decoded.vocab();
        let logits = decoded.copy_logits()?;
        selected
            .iter_mut()
            .enumerate()
            .map(|(row, request)| {
                let row_logits = &logits[row * vocab..(row + 1) * vocab];
                if request.sampler.config().uses_fast_argmax() {
                    return argmax_logits(row_logits);
                }
                request.sampler.sample(row_logits, &request.history)
            })
            .collect()
    }

    fn run_prefill_phase(&mut self, tick: &mut Qwen36SchedulerTick) -> Result<()> {
        let eligible = self
            .prefilling
            .iter()
            .filter(|id| self.requests[id].prefillable_tokens() > 0)
            .take(self.config.prefill_sequence_capacity)
            .count();
        if eligible == 0 {
            return Ok(());
        }
        let mut selected = Vec::with_capacity(eligible);
        let mut chunk_lengths = Vec::with_capacity(eligible);
        let mut token_budget = self.config.prefill_token_capacity;
        let mut slots_remaining = eligible;
        let scan = self.prefilling.len();
        for _ in 0..scan {
            if selected.len() == eligible || token_budget == 0 {
                break;
            }
            let id = self
                .prefilling
                .pop_front()
                .expect("prefilling scan has a request");
            let available = self.requests[&id].prefillable_tokens();
            if available == 0 {
                self.prefilling.push_back(id);
                continue;
            }
            let fair_share = token_budget.div_ceil(slots_remaining);
            let chunk = available.min(fair_share);
            token_budget -= chunk;
            slots_remaining -= 1;
            chunk_lengths.push(chunk);
            selected.push(
                self.requests
                    .remove(&id)
                    .expect("prefilling request is retained"),
            );
        }
        if selected.is_empty() {
            return Ok(());
        }
        tick.scheduled
            .extend(selected.iter().map(|request| request.id));
        let prefill_result = {
            let mut rows = selected
                .iter_mut()
                .zip(chunk_lengths.iter().copied())
                .map(|(request, chunk)| {
                    let start = request.prompt_position;
                    let end = start + chunk;
                    let token_ids = &request.prompt_tokens[start..end];
                    let state = request
                        .sequence
                        .as_deref_mut()
                        .expect("prefilling request has admitted sequence state");
                    Qwen36PrefillRow { token_ids, state }
                })
                .collect::<Vec<_>>();
            self.model
                .prefill_batch(&mut self.prefill_workspace, &mut rows)
        };
        if let Err(error) = prefill_result {
            for request in selected.into_iter().rev() {
                self.prefilling.push_front(request.id);
                self.requests.insert(request.id, request);
            }
            return Err(error);
        }
        for (mut request, chunk) in selected.into_iter().zip(chunk_lengths) {
            request.prompt_position += chunk;
            tick.prefilled.push(Qwen36PrefillProgress {
                request_id: request.id,
                tokens: chunk,
                prompt_position: request.prompt_position,
            });
            self.prefilling.push_back(request.id);
            self.requests.insert(request.id, request);
        }
        Ok(())
    }

    /// Cancels waiting or active work between model submissions.
    pub fn cancel_request(&mut self, id: Qwen36RequestId) -> Qwen36CancelOutcome {
        let Some(request) = self.requests.get(&id) else {
            return Qwen36CancelOutcome::NotFound;
        };
        if request.lifecycle == Qwen36RequestState::Finished {
            return Qwen36CancelOutcome::AlreadyFinished;
        }
        self.waiting.retain(|queued| *queued != id);
        self.prefilling.retain(|queued| *queued != id);
        self.decoding.retain(|queued| *queued != id);
        let request = self
            .requests
            .remove(&id)
            .expect("cancellation target remains retained");
        Qwen36CancelOutcome::Cancelled(Qwen36CancelledRequest {
            id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
        })
    }

    /// Returns the configured scheduler limits.
    pub fn config(&self) -> Qwen36SchedulerConfig {
        self.config
    }

    /// Returns the maximum rows in one decode batch.
    pub fn capacity(&self) -> usize {
        self.config.decode_capacity
    }

    /// Returns exact shared prefill and decode workspace device bytes.
    pub fn workspace_device_bytes(&self) -> usize {
        self.decode_workspace.device_bytes() + self.prefill_workspace.device_bytes()
    }

    /// Returns the number of requests retained by the scheduler.
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns the number of CPU-only requests awaiting admission.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Returns the number of admitted prefill and decode requests.
    pub fn active_sequence_count(&self) -> usize {
        self.prefilling.len() + self.decoding.len()
    }

    /// Returns the number of admitted requests eligible for model work.
    pub fn runnable_count(&self) -> usize {
        self.active_sequence_count()
    }

    /// Returns a request's current lifecycle state.
    pub fn request_state(&self, id: Qwen36RequestId) -> Option<Qwen36RequestState> {
        self.requests.get(&id).map(|request| request.lifecycle)
    }

    /// Returns generated completion tokens retained for a request.
    pub fn generated_tokens(&self, id: Qwen36RequestId) -> Option<&[Qwen36ScheduledToken]> {
        self.requests
            .get(&id)
            .map(|request| request.generated_tokens.as_slice())
    }

    /// Returns exact device bytes owned by an admitted request's sequence state.
    pub fn request_device_bytes(&self, id: Qwen36RequestId) -> Option<usize> {
        self.requests.get(&id).map(|request| {
            request
                .sequence
                .as_deref()
                .map_or(0, Qwen36SequenceState::device_bytes)
        })
    }

    /// Removes and returns a finished request.
    pub fn remove_finished(&mut self, id: Qwen36RequestId) -> Option<Qwen36FinishedRequest> {
        if self.request_state(id) != Some(Qwen36RequestState::Finished) {
            return None;
        }
        let request = self.requests.remove(&id)?;
        Some(Qwen36FinishedRequest {
            id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            finish_reason: request
                .finish_reason
                .expect("finished request has a completion reason"),
        })
    }
}

fn sampled_top1(token: Qwen36NextToken) -> SampledToken {
    SampledToken {
        id: token.id,
        logit: token.value,
        adjusted_logit: token.value,
    }
}

fn argmax_logits(logits: &[f32]) -> Result<SampledToken> {
    let (id, logit) = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduler logits",
            detail: "no finite logits".to_string(),
        })?;
    Ok(SampledToken {
        id: id as u32,
        logit,
        adjusted_logit: logit,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen36CancelOutcome, Qwen36RequestConfig, Qwen36RequestFinishReason, Qwen36RequestState,
        Qwen36Scheduler, Qwen36SchedulerConfig, argmax_logits,
    };
    use crate::qwen3::qwen36::Qwen36TextModel;
    use crate::runtime::sampling::SamplingConfig;
    use std::path::PathBuf;

    #[test]
    fn argmax_prefers_the_lowest_token_on_a_tie() {
        let token = argmax_logits(&[1.0, 3.0, 3.0, f32::NAN]).expect("argmax");
        assert_eq!(token.id, 1);
        assert_eq!(token.logit, 3.0);
    }

    #[test]
    fn lifecycle_finish_and_cancellation_are_distinct_public_states() {
        assert_ne!(Qwen36RequestState::Waiting, Qwen36RequestState::Prefilling);
        assert_ne!(Qwen36RequestState::Prefilling, Qwen36RequestState::Decoding);
        assert_ne!(Qwen36RequestState::Decoding, Qwen36RequestState::Finished);
        assert_ne!(
            Qwen36RequestFinishReason::Eos,
            Qwen36RequestFinishReason::Length
        );
        assert_eq!(Qwen36CancelOutcome::NotFound, Qwen36CancelOutcome::NotFound);
    }

    #[test]
    #[ignore = "loads the full local Qwen3.6 checkpoint"]
    fn real_model_prefills_admits_rotates_and_cancels_without_moving_states() {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36TextModel::open(model_dir).expect("load Qwen3.6 model");
        let mut scheduler = Qwen36Scheduler::new(
            &model,
            Qwen36SchedulerConfig {
                decode_capacity: 2,
                prefill_sequence_capacity: 2,
                prefill_token_capacity: 4,
                max_active_sequences: 2,
                max_context_tokens: 8,
            },
        )
        .expect("scheduler");
        let config = |max_new_tokens| Qwen36RequestConfig {
            sampling: SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            },
            max_new_tokens,
            ..Qwen36RequestConfig::default()
        };
        let first = scheduler
            .add_request(vec![9707], config(2))
            .expect("first request");
        let second = scheduler
            .add_request(vec![3710, 9707, 3710], config(2))
            .expect("second request");
        let third = scheduler
            .add_request(vec![9707, 3710], config(1))
            .expect("third request");
        let cancelled_waiting = scheduler
            .add_request(vec![3710], config(1))
            .expect("waiting cancellation request");
        assert!(matches!(
            scheduler.cancel_request(cancelled_waiting),
            Qwen36CancelOutcome::Cancelled(_)
        ));
        assert_eq!(scheduler.request_device_bytes(first), Some(0));
        assert_eq!(scheduler.waiting_count(), 3);

        let tick = scheduler.tick().expect("first tick");
        assert_eq!(tick.generated.len(), 1);
        assert_eq!(tick.generated[0].request_id, first);
        assert_eq!(tick.prefilled.len(), 1);
        assert_eq!(tick.prefilled[0].request_id, second);
        assert_eq!(tick.prefilled[0].tokens, 2);
        assert_eq!(scheduler.waiting_count(), 1);
        assert!(scheduler.request_device_bytes(second).unwrap() > 0);

        let second_sequence = scheduler.requests[&second]
            .sequence
            .as_deref()
            .expect("second admitted state") as *const _;
        let tick = scheduler.tick().expect("second tick");
        assert_eq!(tick.generated.len(), 2);
        assert_eq!(tick.finished, [first]);
        assert_eq!(scheduler.request_device_bytes(first), Some(0));
        assert_eq!(
            scheduler.requests[&second]
                .sequence
                .as_deref()
                .expect("second retained state") as *const _,
            second_sequence
        );

        let tick = scheduler.tick().expect("third tick");
        assert!(tick.scheduled.contains(&third));
        let outcome = scheduler.cancel_request(third);
        assert!(matches!(outcome, Qwen36CancelOutcome::Cancelled(_)));
        assert_eq!(scheduler.request_state(third), None);
        assert_eq!(
            scheduler.cancel_request(third),
            Qwen36CancelOutcome::NotFound
        );

        while scheduler.request_state(second) != Some(Qwen36RequestState::Finished) {
            scheduler.tick().expect("finish second");
        }
        assert_eq!(
            scheduler.cancel_request(second),
            Qwen36CancelOutcome::AlreadyFinished
        );
        let finished = scheduler.remove_finished(second).expect("remove second");
        assert_eq!(finished.generated_tokens.len(), 2);

        let cancelled_decode = scheduler
            .add_request(vec![9707], config(2))
            .expect("decode cancellation request");
        scheduler.tick().expect("start decode cancellation request");
        assert_eq!(
            scheduler.request_state(cancelled_decode),
            Some(Qwen36RequestState::Decoding)
        );
        assert!(matches!(
            scheduler.cancel_request(cancelled_decode),
            Qwen36CancelOutcome::Cancelled(_)
        ));
        assert_eq!(scheduler.request_state(cancelled_decode), None);
    }
}
