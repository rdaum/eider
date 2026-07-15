//! Tokenized Qwen3.6 request scheduling over the canonical batch decoder.

use super::sampling::{SampledToken, Sampler, SamplingConfig, TokenHistory};
use crate::qwen3::qwen36::{
    Qwen36DecodeBatchWorkspace, Qwen36DecodeRow, Qwen36NextToken, Qwen36SequenceState,
    Qwen36TextModel,
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
    /// Accepted but not yet selected for model work.
    Waiting,
    /// Consuming prompt tokens into persistent model state.
    Prefilling,
    /// Producing completion tokens.
    Decoding,
    /// Reached EOS or the requested completion length.
    Finished,
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

/// Observable result of one scheduler tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen36SchedulerTick {
    /// Requests selected for this batch, in model row order.
    pub scheduled: Vec<Qwen36RequestId>,
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

struct Qwen36Request {
    id: Qwen36RequestId,
    lifecycle: Qwen36RequestState,
    config: Qwen36RequestConfig,
    prompt_tokens: Vec<u32>,
    prompt_position: usize,
    sequence: Box<Qwen36SequenceState>,
    sampler: Sampler,
    history: TokenHistory,
    last_token: Option<u32>,
    generated_tokens: Vec<Qwen36ScheduledToken>,
    finish_reason: Option<Qwen36RequestFinishReason>,
}

impl Qwen36Request {
    fn input_token(&self) -> Result<u32> {
        if let Some(&token) = self.prompt_tokens.get(self.prompt_position) {
            return Ok(token);
        }
        self.last_token.ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduled request",
            detail: format!("request {} has no decode input token", self.id.get()),
        })
    }

    fn needs_output(&self) -> bool {
        self.prompt_position + 1 >= self.prompt_tokens.len()
    }

    fn apply_output(&mut self, sampled: Option<SampledToken>) -> Option<Qwen36ScheduledToken> {
        if self.prompt_position < self.prompt_tokens.len() {
            self.prompt_position += 1;
            if self.prompt_position < self.prompt_tokens.len() {
                self.lifecycle = Qwen36RequestState::Prefilling;
                return None;
            }
        }

        let sampled = sampled.expect("request needing output has a selected token");
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
        Some(token)
    }
}

/// Round-robin token scheduler owning one shared Qwen3.6 batch workspace.
pub struct Qwen36Scheduler<'model> {
    model: &'model Qwen36TextModel,
    workspace: Qwen36DecodeBatchWorkspace,
    requests: BTreeMap<Qwen36RequestId, Box<Qwen36Request>>,
    runnable: VecDeque<Qwen36RequestId>,
    next_id: u64,
}

impl<'model> Qwen36Scheduler<'model> {
    /// Creates a scheduler with a fixed execution capacity and context ceiling.
    pub fn new(
        model: &'model Qwen36TextModel,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        Ok(Self {
            model,
            workspace: model.new_decode_batch_workspace(capacity, max_context_tokens)?,
            requests: BTreeMap::new(),
            runnable: VecDeque::new(),
            next_id: 0,
        })
    }

    /// Adds an already-tokenized request to the waiting queue.
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
        if max_tokens > self.workspace.max_context_tokens() {
            return Err(Error::Shape {
                label: "Qwen3.6 scheduler request capacity",
                expected: format!("at most {} tokens", self.workspace.max_context_tokens()),
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
        let sequence = Box::new(self.model.new_sequence_state(max_tokens.max(1))?);
        let history = TokenHistory::from_tokens(prompt_tokens.iter().copied());
        let request = Box::new(Qwen36Request {
            id,
            lifecycle,
            config,
            prompt_tokens,
            prompt_position: 0,
            sequence,
            sampler,
            history,
            last_token: None,
            generated_tokens: Vec::new(),
            finish_reason,
        });
        self.requests.insert(id, request);
        if lifecycle != Qwen36RequestState::Finished {
            self.runnable.push_back(id);
        }
        Ok(id)
    }

    /// Executes at most one model row per selected runnable request.
    pub fn tick(&mut self) -> Result<Qwen36SchedulerTick> {
        let selected_count = self.workspace.capacity().min(self.runnable.len());
        if selected_count == 0 {
            return Ok(Qwen36SchedulerTick::default());
        }

        let mut selected = Vec::with_capacity(selected_count);
        for _ in 0..selected_count {
            let id = self
                .runnable
                .pop_front()
                .expect("selected runnable request");
            let mut request = self
                .requests
                .remove(&id)
                .expect("runnable request is present");
            if request.lifecycle == Qwen36RequestState::Waiting {
                request.lifecycle = Qwen36RequestState::Prefilling;
            }
            selected.push(request);
        }

        let scheduled = selected.iter().map(|request| request.id).collect();
        let outputs = self.execute_selected(&mut selected);
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(error) => {
                for request in selected.into_iter().rev() {
                    self.runnable.push_front(request.id);
                    self.requests.insert(request.id, request);
                }
                return Err(error);
            }
        };

        let mut tick = Qwen36SchedulerTick {
            scheduled,
            ..Qwen36SchedulerTick::default()
        };
        for (mut request, output) in selected.into_iter().zip(outputs) {
            if let Some(token) = request.apply_output(output) {
                tick.generated.push(token);
            }
            if request.lifecycle == Qwen36RequestState::Finished {
                tick.finished.push(request.id);
            } else {
                self.runnable.push_back(request.id);
            }
            self.requests.insert(request.id, request);
        }
        Ok(tick)
    }

    fn execute_selected(
        &mut self,
        selected: &mut [Box<Qwen36Request>],
    ) -> Result<Vec<Option<SampledToken>>> {
        let needs_host_logits = selected
            .iter()
            .any(|request| request.needs_output() && !request.sampler.config().uses_fast_argmax());
        let mut rows = selected
            .iter_mut()
            .map(|request| {
                let token_id = request.input_token()?;
                Ok(Qwen36DecodeRow {
                    token_id,
                    state: request.sequence.as_mut(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut decoded = self.model.decode_batch(&mut self.workspace, &mut rows)?;

        if !needs_host_logits {
            let top1 = decoded.top1()?;
            return Ok(selected
                .iter()
                .zip(top1)
                .map(|(request, token)| request.needs_output().then(|| sampled_top1(token)))
                .collect());
        }

        let vocab = decoded.vocab();
        let logits = decoded.copy_logits()?;
        selected
            .iter_mut()
            .enumerate()
            .map(|(row, request)| {
                if !request.needs_output() {
                    return Ok(None);
                }
                let row_logits = &logits[row * vocab..(row + 1) * vocab];
                if request.sampler.config().uses_fast_argmax() {
                    return Ok(Some(argmax_logits(row_logits)?));
                }
                request
                    .sampler
                    .sample(row_logits, &request.history)
                    .map(Some)
            })
            .collect()
    }

    /// Returns the configured maximum rows per tick.
    pub fn capacity(&self) -> usize {
        self.workspace.capacity()
    }

    /// Returns exact shared-workspace device bytes.
    pub fn workspace_device_bytes(&self) -> usize {
        self.workspace.device_bytes()
    }

    /// Returns the number of requests retained by the scheduler.
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns the number of runnable requests.
    pub fn runnable_count(&self) -> usize {
        self.runnable.len()
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

    /// Returns exact device bytes owned by a request's persistent sequence state.
    pub fn request_device_bytes(&self, id: Qwen36RequestId) -> Option<usize> {
        self.requests
            .get(&id)
            .map(|request| request.sequence.device_bytes())
    }

    /// Removes and returns a finished request without relocating other request state.
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
        Qwen36RequestConfig, Qwen36RequestFinishReason, Qwen36RequestState, Qwen36Scheduler,
        argmax_logits,
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
    fn lifecycle_and_finish_reason_are_distinct_public_states() {
        assert_ne!(Qwen36RequestState::Waiting, Qwen36RequestState::Prefilling);
        assert_ne!(Qwen36RequestState::Prefilling, Qwen36RequestState::Decoding);
        assert_ne!(Qwen36RequestState::Decoding, Qwen36RequestState::Finished);
        assert_ne!(
            Qwen36RequestFinishReason::Eos,
            Qwen36RequestFinishReason::Length
        );
    }

    #[test]
    #[ignore = "loads the full local Qwen3.6 checkpoint"]
    fn real_model_rotates_mixed_rows_and_keeps_sequence_allocations_stable() {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36TextModel::open(model_dir).expect("load Qwen3.6 model");
        let mut scheduler = Qwen36Scheduler::new(&model, 2, 8).expect("scheduler");
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
        let second_sequence = scheduler.requests[&second].sequence.as_ref() as *const _;

        let tick = scheduler.tick().expect("first tick");
        assert_eq!(tick.scheduled, [first, second]);
        assert_eq!(tick.generated.len(), 1);
        assert_eq!(
            scheduler.request_state(first),
            Some(Qwen36RequestState::Decoding)
        );
        assert_eq!(
            scheduler.request_state(second),
            Some(Qwen36RequestState::Prefilling)
        );

        let tick = scheduler.tick().expect("second tick");
        assert_eq!(tick.scheduled, [third, first]);
        assert_eq!(tick.finished, [first]);
        scheduler.remove_finished(first).expect("remove first");
        assert_eq!(
            scheduler.requests[&second].sequence.as_ref() as *const _,
            second_sequence
        );

        let tick = scheduler.tick().expect("third tick");
        assert_eq!(tick.scheduled, [second, third]);
        assert_eq!(tick.finished, [third]);
        scheduler.remove_finished(third).expect("remove third");
        assert_eq!(
            scheduler.requests[&second].sequence.as_ref() as *const _,
            second_sequence
        );

        let tick = scheduler.tick().expect("fourth tick");
        assert_eq!(tick.scheduled, [second]);
        assert_eq!(
            scheduler.request_state(second),
            Some(Qwen36RequestState::Decoding)
        );
        let tick = scheduler.tick().expect("fifth tick");
        assert_eq!(tick.finished, [second]);
        let finished = scheduler.remove_finished(second).expect("remove second");
        assert_eq!(finished.generated_tokens.len(), 2);
        assert_eq!(scheduler.request_count(), 0);
        assert_eq!(scheduler.runnable_count(), 0);
    }
}
