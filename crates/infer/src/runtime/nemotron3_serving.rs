//! Structured multi-session chat serving for Nemotron 3.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::sampling::{Sampler, TokenHistory};
use super::scheduler::{RequestConfig, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::nemotron3::{Nemotron3DecodeState, Nemotron3Model};
use nvfp4::{Error, Result};
use std::collections::{BTreeMap, VecDeque};

/// Stable request identity assigned by a Nemotron 3 chat service.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Nemotron3RequestId(u64);

impl Nemotron3RequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after prompt rendering and tokenization.
pub struct Nemotron3Admission {
    pub request_id: Nemotron3RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

/// Device-state allocation completed during a tick.
pub struct Nemotron3AdmissionProgress {
    pub request_id: Nemotron3RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
}

/// Prompt progress completed during a tick.
pub struct Nemotron3PrefillProgress {
    pub request_id: Nemotron3RequestId,
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct Nemotron3ChatDelta {
    pub request_id: Nemotron3RequestId,
    pub event: ChatOutputEvent,
}

/// Terminal request metadata.
pub struct Nemotron3Finished {
    pub request_id: Nemotron3RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

/// Observable work and output from one service iteration.
#[derive(Default)]
pub struct Nemotron3Tick {
    pub admitted: Vec<Nemotron3AdmissionProgress>,
    pub prefilled: Vec<Nemotron3PrefillProgress>,
    pub generated: Vec<Nemotron3RequestId>,
    pub output: Vec<Nemotron3ChatDelta>,
    pub finished: Vec<Nemotron3Finished>,
    pub active_sequences: usize,
}

/// Outcome of cancelling a waiting or active request.
pub enum Nemotron3CancelOutcome {
    Cancelled {
        released_sequence_device_bytes: usize,
    },
    NotFound,
}

struct ActiveRequest<'tokenizer> {
    prompt: Vec<u32>,
    prompt_position: usize,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    state: Option<Nemotron3DecodeState>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint prompt rendering and round-robin multi-session Nemotron execution.
pub struct Nemotron3ChatService<'model, 'template> {
    model: &'model Nemotron3Model,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<Nemotron3RequestId>,
    requests: BTreeMap<Nemotron3RequestId, ActiveRequest<'template>>,
    active_sequences: usize,
}

impl<'model, 'template> Nemotron3ChatService<'model, 'template> {
    /// Creates a multi-session service with explicit scheduling limits.
    pub fn new(
        model: &'model Nemotron3Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
        })
    }

    /// Renders, tokenizes, and queues one request without allocating GPU state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<Nemotron3Admission> {
        request.generation.validate()?;
        if request.stop_sequences.iter().any(String::is_empty) {
            return Err(Error::Format {
                label: "chat stop sequences",
                detail: "stop sequences must not be empty".to_string(),
            });
        }
        let prompt = self.template.render_and_tokenize(
            &request.messages,
            &request.tools,
            request.template,
        )?;
        if prompt.token_ids.is_empty() {
            return Err(Error::Format {
                label: "Nemotron 3 chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Nemotron 3 request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Nemotron 3 request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Nemotron3RequestId(self.next_id);
        self.next_id += 1;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let active = ActiveRequest {
            prompt: prompt.token_ids.clone(),
            prompt_position: 0,
            generation: request.generation.clone(),
            generated_tokens: 0,
            last_token: None,
            state: None,
            sampler: Sampler::new(request.generation.sampling)?,
            history: TokenHistory::from_tokens(prompt.token_ids.iter().copied()),
            output: ChatOutputCodec::new(
                self.template.tokenizer(),
                &request.tools,
                starts_in_reasoning,
            )?,
            filter: ResponseFilter::new(request.stop_sequences),
            usage: ChatUsage {
                prompt_tokens: prompt.token_ids.len(),
                ..ChatUsage::default()
            },
        };
        self.waiting.push_back(id);
        self.requests.insert(id, active);
        Ok(Nemotron3Admission {
            request_id: id,
            prompt_tokens: prompt.token_ids.len(),
            max_output_tokens: request.generation.max_new_tokens,
        })
    }

    /// Runs one decode-first round-robin scheduling iteration.
    pub fn tick(&mut self) -> Result<Nemotron3Tick> {
        let mut tick = Nemotron3Tick::default();
        self.admit(&mut tick)?;
        let mut terminal = BTreeMap::new();

        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.state.is_some()
                    && request.prompt_position + 1 >= request.prompt.len()
                    && request.generated_tokens < request.generation.max_new_tokens
            })
            .map(|(&id, _)| id)
            .take(self.config.decode_capacity)
            .collect::<Vec<_>>();
        for id in decode_ids {
            if let Some(reason) = self.generate_one(id, &mut tick)? {
                terminal.insert(id, reason);
            }
        }

        let prefill_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.state.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position + 1 < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        let mut token_budget = self.config.prefill_token_capacity;
        let mut prefill_progress = BTreeMap::new();
        while token_budget != 0 {
            let mut progressed = false;
            for &id in &prefill_ids {
                if token_budget == 0 {
                    break;
                }
                let request = self.requests.get_mut(&id).expect("prefill request exists");
                if request.prompt_position + 1 >= request.prompt.len() {
                    continue;
                }
                let token = request.prompt[request.prompt_position];
                self.model.forward_one(
                    request.state.as_mut().expect("admitted request has state"),
                    token,
                )?;
                request.prompt_position += 1;
                token_budget -= 1;
                progressed = true;
                prefill_progress.insert(id, request.prompt_position);
            }
            if !progressed {
                break;
            }
        }
        for (id, position) in prefill_progress {
            tick.prefilled.push(Nemotron3PrefillProgress {
                request_id: id,
                prompt_position: position,
            });
        }

        for (&id, request) in &self.requests {
            if request.state.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        tick.active_sequences = self.active_sequences;
        Ok(tick)
    }

    /// Cancels a waiting or active request.
    pub fn cancel_request(&mut self, id: Nemotron3RequestId) -> Nemotron3CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Nemotron3CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.state.map_or(0, |state| state.device_bytes());
        if released != 0 {
            self.active_sequences -= 1;
        }
        Nemotron3CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns the number of requests currently owning device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(&mut self, tick: &mut Nemotron3Tick) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let state = self.model.sequence_state(capacity.max(1))?;
            let bytes = state.device_bytes();
            request.state = Some(state);
            self.active_sequences += 1;
            tick.admitted.push(Nemotron3AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens: 0,
            });
        }
        Ok(())
    }

    fn generate_one(
        &mut self,
        id: Nemotron3RequestId,
        tick: &mut Nemotron3Tick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let input = request
            .last_token
            .unwrap_or(request.prompt[request.prompt.len() - 1]);
        let state = request.state.as_mut().expect("admitted request has state");
        self.model.forward_one(state, input)?;
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(state)?;
            super::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            let logits = self.model.logits_to_host(state)?;
            request.sampler.sample(&logits, &request.history)?
        };
        request.generated_tokens += 1;
        request.last_token = Some(sampled.id);
        request.history.push(sampled.id);
        request.usage.completion_tokens += 1;
        if request.output.is_reasoning() {
            request.usage.reasoning_tokens += 1;
        }
        tick.generated.push(id);
        let events = request.output.push_token(sampled.id)?;
        if let Some(reason) = request.filter.apply(id, events, &mut tick.output) {
            return Ok(Some(reason));
        }
        if request.generation.eos_token_ids.contains(&sampled.id) {
            return Ok(Some(ChatFinishReason::Eos));
        }
        if request.generated_tokens == request.generation.max_new_tokens {
            return Ok(Some(ChatFinishReason::Length));
        }
        Ok(None)
    }

    fn finish_request(
        &mut self,
        id: Nemotron3RequestId,
        mut reason: ChatFinishReason,
        tick: &mut Nemotron3Tick,
    ) -> Result<()> {
        let request = self.requests.get_mut(&id).expect("terminal request exists");
        if matches!(reason, ChatFinishReason::Eos | ChatFinishReason::Length) {
            let events = if matches!(reason, ChatFinishReason::Length) {
                request.output.finish_truncated()?
            } else {
                request.output.finish()?
            };
            if let Some(protocol_reason) = request.filter.apply(id, events, &mut tick.output) {
                reason = protocol_reason;
            } else if request.filter.saw_tool_calls {
                reason = ChatFinishReason::ToolCalls;
            } else {
                request.filter.flush(id, &mut tick.output);
            }
        }
        let mut request = self.requests.remove(&id).expect("terminal request remains");
        let state = request.state.take().expect("terminal request is admitted");
        let released = state.device_bytes();
        self.active_sequences -= 1;
        tick.finished.push(Nemotron3Finished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

struct ResponseFilter {
    stop: StopBuffer,
    saw_tool_calls: bool,
}

impl ResponseFilter {
    fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            stop: StopBuffer::new(stop_sequences),
            saw_tool_calls: false,
        }
    }

    fn apply(
        &mut self,
        request_id: Nemotron3RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Nemotron3ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(Nemotron3ChatDelta { request_id, event })
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Nemotron3ChatDelta {
                            request_id,
                            event: ChatOutputEvent::Text(stopped.text),
                        });
                    }
                    if let Some(sequence) = stopped.matched {
                        return Some(ChatFinishReason::Stop(sequence));
                    }
                }
                ChatOutputEvent::ToolCall(_) => {
                    self.flush(request_id, output);
                    output.push(Nemotron3ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: Nemotron3RequestId, output: &mut Vec<Nemotron3ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Nemotron3ChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
