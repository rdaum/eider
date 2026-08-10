//! Multi-session chat serving for Muse Glimmer.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::sampling::{Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::muse_glimmer::{MuseGlimmerDecodeState, MuseGlimmerModel};
use nvfp4::{Error, Result};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

/// Stable identity assigned to a Muse Glimmer request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MuseGlimmerRequestId(u64);

impl MuseGlimmerRequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after rendering and tokenization.
pub struct MuseGlimmerAdmission {
    /// Assigned request identity.
    pub request_id: MuseGlimmerRequestId,
    /// Rendered prompt token count.
    pub prompt_tokens: usize,
    /// Requested completion-token limit.
    pub max_output_tokens: usize,
}

/// Device allocation completed during a service tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MuseGlimmerAdmissionProgress {
    /// Admitted request.
    pub request_id: MuseGlimmerRequestId,
    /// Sequence-specific device bytes.
    pub sequence_device_bytes: usize,
    /// Prefix-cache hits; currently always zero for Muse Glimmer.
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time at admission.
    pub admitted_after_tick_start: Duration,
}

/// Prompt progress completed during one tick.
pub struct MuseGlimmerPrefillProgress {
    /// Request whose prompt advanced.
    pub request_id: MuseGlimmerRequestId,
    /// Total prompt position after this tick.
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct MuseGlimmerChatDelta {
    /// Request owning this delta.
    pub request_id: MuseGlimmerRequestId,
    /// Reasoning, visible text, or tool-call output.
    pub event: ChatOutputEvent,
}

/// Terminal request metadata.
pub struct MuseGlimmerFinished {
    /// Finished request.
    pub request_id: MuseGlimmerRequestId,
    /// API-facing finish reason.
    pub finish_reason: ChatFinishReason,
    /// Final token usage.
    pub usage: ChatUsage,
    /// Sequence device bytes released at completion.
    pub released_sequence_device_bytes: usize,
}

/// Work and output from one service iteration.
#[derive(Default)]
pub struct MuseGlimmerTick {
    /// Requests allocated during this tick.
    pub admitted: Vec<MuseGlimmerAdmissionProgress>,
    /// Prompt progress during this tick.
    pub prefilled: Vec<MuseGlimmerPrefillProgress>,
    /// Requests producing a token during this tick.
    pub generated: Vec<MuseGlimmerRequestId>,
    /// Structured streaming deltas.
    pub output: Vec<MuseGlimmerChatDelta>,
    /// Requests completing during this tick.
    pub finished: Vec<MuseGlimmerFinished>,
    /// Device-resident sequences remaining after the tick.
    pub active_sequences: usize,
}

/// Outcome of cancelling a queued or active request.
pub enum MuseGlimmerCancelOutcome {
    /// The request was removed and these device bytes were released.
    Cancelled {
        /// Sequence-specific allocation released, or zero while queued.
        released_sequence_device_bytes: usize,
    },
    /// No retained request had this identity.
    NotFound,
}

struct ActiveRequest<'tokenizer> {
    prompt: Vec<u32>,
    prompt_position: usize,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    dflash_enabled: bool,
    pending_dflash_token: Option<u32>,
    prompt_logits_ready: bool,
    state: Option<MuseGlimmerDecodeState>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint rendering and decode-first Muse Glimmer execution.
pub struct MuseGlimmerChatService<'model, 'template> {
    model: &'model MuseGlimmerModel,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<MuseGlimmerRequestId>,
    requests: BTreeMap<MuseGlimmerRequestId, ActiveRequest<'template>>,
    active_sequences: usize,
}

impl<'model, 'template> MuseGlimmerChatService<'model, 'template> {
    /// Creates a service with explicit scheduling limits.
    pub fn new(
        model: &'model MuseGlimmerModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_position_embeddings {
            return Err(Error::Shape {
                label: "Muse Glimmer scheduler context",
                expected: format!("at most {} tokens", model.config().max_position_embeddings),
                actual: format!("{} tokens", config.max_context_tokens),
            });
        }
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

    /// Renders, tokenizes, and queues a request without allocating GPU state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<MuseGlimmerAdmission> {
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
                label: "Muse Glimmer chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Muse Glimmer request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Muse Glimmer request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = MuseGlimmerRequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Muse Glimmer request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        let dflash_enabled = self.model.has_dflash()
            && request.generation.sampling.uses_fast_argmax()
            && total
                .checked_add(15)
                .is_some_and(|capacity| capacity <= self.model.config().max_position_embeddings);
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids.clone(),
                prompt_position: 0,
                generation: request.generation.clone(),
                generated_tokens: 0,
                last_token: None,
                dflash_enabled,
                pending_dflash_token: None,
                prompt_logits_ready: false,
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
                    prompt_tokens,
                    ..ChatUsage::default()
                },
            },
        );
        self.waiting.push_back(id);
        Ok(MuseGlimmerAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one decode-first scheduling iteration.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<MuseGlimmerRequestId, MuseGlimmerAdmissionProgress>,
        ),
    ) -> Result<MuseGlimmerTick> {
        let started = Instant::now();
        let mut tick = MuseGlimmerTick::default();
        self.admit(&mut tick, started, on_lifecycle)?;
        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.state.is_some()
                    && request.prompt_position == request.prompt.len()
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
                    && request.prompt_position < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick, on_lifecycle)?;
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

    /// Cancels a queued or active request.
    pub fn cancel_request(&mut self, id: MuseGlimmerRequestId) -> MuseGlimmerCancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return MuseGlimmerCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.state.map_or(0, |state| state.device_bytes());
        if released != 0 {
            self.active_sequences -= 1;
        }
        MuseGlimmerCancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns the number of requests with device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut MuseGlimmerTick,
        started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<MuseGlimmerRequestId, MuseGlimmerAdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len()
                + request.generation.max_new_tokens
                + usize::from(request.dflash_enabled) * 15;
            let state = self.model.new_decode_state(capacity.max(1))?;
            let progress = MuseGlimmerAdmissionProgress {
                request_id: id,
                sequence_device_bytes: state.device_bytes(),
                cached_prompt_tokens: 0,
                admitted_after_tick_start: started.elapsed(),
            };
            request.state = Some(state);
            self.active_sequences += 1;
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        ids: &[MuseGlimmerRequestId],
        tick: &mut MuseGlimmerTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<MuseGlimmerRequestId, MuseGlimmerAdmissionProgress>,
        ),
    ) -> Result<()> {
        let mut budget = self.config.prefill_token_capacity;
        for (index, &id) in ids.iter().enumerate() {
            let request = self.requests.get_mut(&id).expect("prefill request exists");
            let available = request.prompt.len() - request.prompt_position;
            let remaining = ids.len() - index;
            let chunk = available.min(budget.div_ceil(remaining));
            if chunk == 0 {
                continue;
            }
            budget -= chunk;
            let start = request.prompt_position;
            let end = start + chunk;
            on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            let state = request.state.as_mut().expect("request is admitted");
            if request.dflash_enabled {
                let mut chunk_start = start;
                while chunk_start < end {
                    let chunk_end = (chunk_start + 16).min(end);
                    self.model.dflash_prefill_chunk(
                        state,
                        &request.prompt[chunk_start..chunk_end],
                        chunk_end == request.prompt.len(),
                    )?;
                    chunk_start = chunk_end;
                }
            } else {
                for (offset, &token) in request.prompt[start..end].iter().enumerate() {
                    if start + offset + 1 == request.prompt.len() {
                        self.model.forward_one(state, token)?;
                    } else {
                        self.model.consume_one(state, token)?;
                    }
                }
            }
            request.prompt_position = end;
            request.prompt_logits_ready = end == request.prompt.len();
            tick.prefilled.push(MuseGlimmerPrefillProgress {
                request_id: id,
                prompt_position: end,
            });
        }
        Ok(())
    }

    fn generate_one(
        &mut self,
        id: MuseGlimmerRequestId,
        tick: &mut MuseGlimmerTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        if request.dflash_enabled {
            return Self::generate_dflash(self.model, id, request, tick);
        }
        let state = request.state.as_mut().expect("decode request is admitted");
        if request.prompt_logits_ready {
            request.prompt_logits_ready = false;
        } else {
            self.model.forward_one(
                state,
                request
                    .last_token
                    .expect("generated token exists after prompt logits"),
            )?;
        }
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(state)?;
            super::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            request
                .sampler
                .sample(&self.model.logits_to_host(state)?, &request.history)?
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

    fn generate_dflash(
        model: &MuseGlimmerModel,
        id: MuseGlimmerRequestId,
        request: &mut ActiveRequest<'template>,
        tick: &mut MuseGlimmerTick,
    ) -> Result<Option<ChatFinishReason>> {
        let anchor = if let Some(token) = request.pending_dflash_token.take() {
            token
        } else {
            if !request.prompt_logits_ready {
                return Err(Error::Format {
                    label: "Muse Glimmer DFlash serving",
                    detail: "missing prompt logits or pending target token".to_string(),
                });
            }
            request.prompt_logits_ready = false;
            model
                .argmax_with_logit(request.state.as_mut().expect("request is admitted"))?
                .0
        };
        let cycle =
            model.dflash_cycle(request.state.as_mut().expect("request is admitted"), anchor)?;
        request.pending_dflash_token = Some(cycle.next_token);
        for token in cycle.tokens {
            request.generated_tokens += 1;
            request.last_token = Some(token);
            request.history.push(token);
            request.usage.completion_tokens += 1;
            if request.output.is_reasoning() {
                request.usage.reasoning_tokens += 1;
            }
            tick.generated.push(id);
            let events = request.output.push_token(token)?;
            if let Some(reason) = request.filter.apply(id, events, &mut tick.output) {
                return Ok(Some(reason));
            }
            if request.generation.eos_token_ids.contains(&token) {
                return Ok(Some(ChatFinishReason::Eos));
            }
            if request.generated_tokens == request.generation.max_new_tokens {
                return Ok(Some(ChatFinishReason::Length));
            }
        }
        Ok(None)
    }

    fn finish_request(
        &mut self,
        id: MuseGlimmerRequestId,
        mut reason: ChatFinishReason,
        tick: &mut MuseGlimmerTick,
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
        let released = request
            .state
            .take()
            .expect("terminal request is admitted")
            .device_bytes();
        self.active_sequences -= 1;
        tick.finished.push(MuseGlimmerFinished {
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
        request_id: MuseGlimmerRequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<MuseGlimmerChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(MuseGlimmerChatDelta { request_id, event })
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(MuseGlimmerChatDelta {
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
                    output.push(MuseGlimmerChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: MuseGlimmerRequestId, output: &mut Vec<MuseGlimmerChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(MuseGlimmerChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
