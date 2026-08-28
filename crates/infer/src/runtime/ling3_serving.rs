//! Decode-first multi-session chat serving for Ling 3.

use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use crate::ling3::{
    Ling3Model, Ling3PrefillWorkspace, Ling3Sequence, Ling3SequenceCache, admit_ling3_sequence,
    new_ling3_sequence_cache,
};
use eider_cuda::{CudaStream, Error, Result};
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::sampling::{Sampler, TokenHistory};
use eider_runtime::stop::StopBuffer;
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

/// Stable identity assigned to a Ling 3 request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Ling3RequestId(u64);

impl Ling3RequestId {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after rendering and tokenisation.
pub struct Ling3Admission {
    pub request_id: Ling3RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

/// Device allocation completed during a service tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ling3AdmissionProgress {
    pub request_id: Ling3RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    pub admitted_after_tick_start: Duration,
}

/// Prompt progress completed during one tick.
pub struct Ling3PrefillProgress {
    pub request_id: Ling3RequestId,
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct Ling3ChatDelta {
    pub request_id: Ling3RequestId,
    pub event: ChatOutputEvent,
}

/// Terminal request metadata.
pub struct Ling3Finished {
    pub request_id: Ling3RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

/// Work and output from one service iteration.
#[derive(Default)]
pub struct Ling3Tick {
    pub admitted: Vec<Ling3AdmissionProgress>,
    pub prefilled: Vec<Ling3PrefillProgress>,
    pub generated: Vec<Ling3RequestId>,
    pub output: Vec<Ling3ChatDelta>,
    pub finished: Vec<Ling3Finished>,
    pub active_sequences: usize,
}

/// Outcome of cancelling a queued or active request.
pub enum Ling3CancelOutcome {
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
    prompt_logits_ready: bool,
    sequence: Option<Ling3Sequence>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint rendering and correctness-first Ling 3 execution.
pub struct Ling3ChatService<'model, 'template> {
    model: &'model Ling3Model,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    stream: CudaStream,
    sequence_cache: Ling3SequenceCache,
    prefill_workspace: Ling3PrefillWorkspace,
    next_id: u64,
    waiting: VecDeque<Ling3RequestId>,
    requests: BTreeMap<Ling3RequestId, ActiveRequest<'template>>,
    active_sequences: usize,
}

impl<'model, 'template> Ling3ChatService<'model, 'template> {
    pub fn new(
        model: &'model Ling3Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.max_context_tokens() {
            return Err(Error::Shape {
                label: "Ling 3 scheduler context",
                expected: format!("at most {} tokens", model.max_context_tokens()),
                actual: format!("{} tokens", config.max_context_tokens),
            });
        }
        let stream = CudaStream::new_non_blocking()?;
        let sequence_cache = new_ling3_sequence_cache(
            model,
            config.max_active_sequences,
            config.max_context_tokens,
        )?;
        let prefill_workspace = model.new_prefill_workspace(config.prefill_token_capacity)?;
        Ok(Self {
            model,
            template,
            config,
            stream,
            sequence_cache,
            prefill_workspace,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
        })
    }

    /// Renders, tokenises, and queues a request without allocating device state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<Ling3Admission> {
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
                label: "Ling 3 chat prompt",
                detail: "prompt tokenised to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Ling 3 request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Ling 3 request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Ling3RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Ling 3 request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids.clone(),
                prompt_position: 0,
                generation: request.generation.clone(),
                generated_tokens: 0,
                last_token: None,
                prompt_logits_ready: false,
                sequence: None,
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
        Ok(Ling3Admission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one decode-first scheduling iteration.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(RequestLifecycleEvent<Ling3RequestId, Ling3AdmissionProgress>),
    ) -> Result<Ling3Tick> {
        let started = Instant::now();
        let mut tick = Ling3Tick::default();
        self.admit(&mut tick, started, on_lifecycle)?;
        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.sequence.is_some()
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
                request.sequence.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick, on_lifecycle)?;
        for (&id, request) in &self.requests {
            if request.sequence.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        tick.active_sequences = self.active_sequences;
        Ok(tick)
    }

    pub fn cancel_request(&mut self, id: Ling3RequestId) -> Ling3CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Ling3CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request
            .sequence
            .as_ref()
            .map_or(0, Ling3Sequence::device_bytes);
        if let Some(sequence) = request.sequence {
            if let Err(error) = sequence.finish(&self.stream, &mut self.sequence_cache) {
                tracing::warn!(%error, request_id = id.get(), "failed to release cancelled Ling sequence");
            }
            self.active_sequences -= 1;
        }
        Ling3CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut Ling3Tick,
        started: Instant,
        on_lifecycle: &mut dyn FnMut(RequestLifecycleEvent<Ling3RequestId, Ling3AdmissionProgress>),
    ) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let Some(sequence) = admit_ling3_sequence(
                self.model,
                &mut self.sequence_cache,
                capacity.max(1),
                &self.stream,
            )?
            else {
                self.waiting.push_front(id);
                break;
            };
            let progress = Ling3AdmissionProgress {
                request_id: id,
                sequence_device_bytes: sequence.device_bytes(),
                cached_prompt_tokens: 0,
                admitted_after_tick_start: started.elapsed(),
            };
            request.sequence = Some(sequence);
            self.active_sequences += 1;
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        ids: &[Ling3RequestId],
        tick: &mut Ling3Tick,
        on_lifecycle: &mut dyn FnMut(RequestLifecycleEvent<Ling3RequestId, Ling3AdmissionProgress>),
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
            let sequence = request.sequence.as_mut().expect("request is admitted");
            self.model.prefill(
                &mut self.prefill_workspace,
                sequence,
                &mut self.sequence_cache,
                &request.prompt[start..end],
                &self.stream,
            )?;
            request.prompt_position = end;
            request.prompt_logits_ready = end == request.prompt.len();
            tick.prefilled.push(Ling3PrefillProgress {
                request_id: id,
                prompt_position: end,
            });
        }
        Ok(())
    }

    fn generate_one(
        &mut self,
        id: Ling3RequestId,
        tick: &mut Ling3Tick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let sequence = request
            .sequence
            .as_mut()
            .expect("decode request is admitted");
        if request.prompt_logits_ready {
            request.prompt_logits_ready = false;
        } else {
            self.model.decode_token(
                request
                    .last_token
                    .expect("generated token exists after prompt logits"),
                sequence,
                &mut self.sequence_cache,
                &self.stream,
            )?;
        }
        let logits = self
            .model
            .logits(&sequence.workspace)
            .copy_to_host(&self.stream)?;
        let sampled = request.sampler.sample(&logits, &request.history)?;
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
        id: Ling3RequestId,
        mut reason: ChatFinishReason,
        tick: &mut Ling3Tick,
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
        let sequence = request
            .sequence
            .take()
            .expect("terminal request is admitted");
        let released = sequence.device_bytes();
        sequence.finish(&self.stream, &mut self.sequence_cache)?;
        self.active_sequences -= 1;
        tick.finished.push(Ling3Finished {
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
        request_id: Ling3RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Ling3ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(Ling3ChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Ling3ChatDelta {
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
                    output.push(Ling3ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: Ling3RequestId, output: &mut Vec<Ling3ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Ling3ChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
