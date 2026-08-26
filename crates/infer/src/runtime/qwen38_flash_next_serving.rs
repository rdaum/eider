//! Serial chat serving for the Qwen3.8 Flash Next native QSA path.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::nvfp4::{Error, Result};
use crate::qwen38_flash_next::{Qwen38FlashNextModel, Qwen38NextToken};
use crate::runtime::qwen38_flash_next_sequence::{
    Qwen38FlashNextSequence, Qwen38FlashNextSequenceCache, new_qwen38_flash_next_sequence_cache,
};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Qwen38FlashNextRequestId(u64);

impl Qwen38FlashNextRequestId {
    pub fn get(self) -> u64 {
        self.0
    }
}

pub struct Qwen38FlashNextAdmission {
    pub request_id: Qwen38FlashNextRequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextAdmissionProgress {
    pub request_id: Qwen38FlashNextRequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    pub allocation_duration: Duration,
    pub admitted_after_tick_start: Duration,
}

pub struct Qwen38FlashNextPrefillProgress {
    pub request_id: Qwen38FlashNextRequestId,
    pub prompt_position: usize,
}

pub struct Qwen38FlashNextChatDelta {
    pub request_id: Qwen38FlashNextRequestId,
    pub event: ChatOutputEvent,
}

pub struct Qwen38FlashNextFinished {
    pub request_id: Qwen38FlashNextRequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

#[derive(Default)]
pub struct Qwen38FlashNextTick {
    pub admitted: Vec<Qwen38FlashNextAdmissionProgress>,
    pub prefilled: Vec<Qwen38FlashNextPrefillProgress>,
    pub generated: Vec<Qwen38FlashNextRequestId>,
    pub output: Vec<Qwen38FlashNextChatDelta>,
    pub finished: Vec<Qwen38FlashNextFinished>,
    pub active_sequences: usize,
}

pub enum Qwen38FlashNextCancelOutcome {
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
    pending_next: Option<Qwen38NextToken>,
    sequence: Option<Box<Qwen38FlashNextSequence>>,
    sequence_device_bytes: usize,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Decode-first, single-sequence service for the native QSA runtime.
pub struct Qwen38FlashNextChatService<'template> {
    model: Qwen38FlashNextModel,
    sequence_cache: Qwen38FlashNextSequenceCache,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<Qwen38FlashNextRequestId>,
    requests: BTreeMap<Qwen38FlashNextRequestId, ActiveRequest<'template>>,
    active_sequences: usize,
}

impl<'template> Qwen38FlashNextChatService<'template> {
    pub fn new(
        model: Qwen38FlashNextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_position_embeddings {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next server context",
                expected: format!("at most {} tokens", model.config().max_position_embeddings),
                actual: format!("{} tokens", config.max_context_tokens),
            });
        }
        if config.max_active_sequences != 1
            || config.decode_capacity != 1
            || config.prefill_sequence_capacity != 1
        {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next scheduler",
                expected: "one active, decode, and prefill sequence".to_string(),
                actual: format!(
                    "active={} decode={} prefill={}",
                    config.max_active_sequences,
                    config.decode_capacity,
                    config.prefill_sequence_capacity
                ),
            });
        }
        if config.speculative_drafts != 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next speculative decoding",
                expected: "zero drafts until MTP is implemented".to_string(),
                actual: config.speculative_drafts.to_string(),
            });
        }
        let sequence_cache = new_qwen38_flash_next_sequence_cache(
            &model,
            config.max_active_sequences,
            config.max_context_tokens,
        )?;
        Ok(Self {
            model,
            sequence_cache,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
        })
    }

    pub fn add_request(&mut self, request: ChatRequest) -> Result<Qwen38FlashNextAdmission> {
        request.generation.validate()?;
        if !request.generation.sampling.uses_fast_argmax() {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next sampling",
                detail: "the native QSA server currently supports greedy decoding only".to_string(),
            });
        }
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
                label: "Qwen3.8 Flash Next chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 Flash Next request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Qwen38FlashNextRequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids,
                prompt_position: 0,
                generation: request.generation,
                generated_tokens: 0,
                last_token: None,
                pending_next: None,
                sequence: None,
                sequence_device_bytes: 0,
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
        Ok(Qwen38FlashNextAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen38FlashNextRequestId, Qwen38FlashNextAdmissionProgress>,
        ),
    ) -> Result<Qwen38FlashNextTick> {
        let started = Instant::now();
        let mut tick = Qwen38FlashNextTick::default();
        self.admit(&mut tick, started, on_lifecycle)?;
        let mut terminal = BTreeMap::new();

        let decode_id = self.requests.iter().find_map(|(&id, request)| {
            (request.sequence.is_some()
                && request.prompt_position == request.prompt.len()
                && request.generated_tokens < request.generation.max_new_tokens)
                .then_some(id)
        });
        if let Some(id) = decode_id
            && let Some(reason) = self.generate_one(id, &mut tick)?
        {
            terminal.insert(id, reason);
        }

        let prefill_id = self.requests.iter().find_map(|(&id, request)| {
            (request.sequence.is_some()
                && request.generation.max_new_tokens != 0
                && request.prompt_position < request.prompt.len())
            .then_some(id)
        });
        if let Some(id) = prefill_id {
            self.prefill(id, &mut tick, on_lifecycle)?;
        }
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

    pub fn cancel_request(&mut self, id: Qwen38FlashNextRequestId) -> Qwen38FlashNextCancelOutcome {
        let Some(mut request) = self.requests.remove(&id) else {
            return Qwen38FlashNextCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        if let Some(sequence) = request.sequence.take() {
            let _ = sequence.finish(&mut self.sequence_cache);
            self.active_sequences -= 1;
        }
        Qwen38FlashNextCancelOutcome::Cancelled {
            released_sequence_device_bytes: request.sequence_device_bytes,
        }
    }

    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut Qwen38FlashNextTick,
        started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen38FlashNextRequestId, Qwen38FlashNextAdmissionProgress>,
        ),
    ) -> Result<()> {
        if self.active_sequences != 0 {
            return Ok(());
        }
        let Some(id) = self.waiting.pop_front() else {
            return Ok(());
        };
        let allocation_started = Instant::now();
        let request = self.requests.get_mut(&id).expect("waiting request exists");
        let capacity = request.prompt.len() + request.generation.max_new_tokens;
        let sequence =
            Qwen38FlashNextSequence::admit(&self.model, &mut self.sequence_cache, capacity.max(1))?;
        let sequence_device_bytes = sequence.device_bytes();
        let progress = Qwen38FlashNextAdmissionProgress {
            request_id: id,
            sequence_device_bytes,
            cached_prompt_tokens: 0,
            allocation_duration: allocation_started.elapsed(),
            admitted_after_tick_start: started.elapsed(),
        };
        request.sequence = Some(Box::new(sequence));
        request.sequence_device_bytes = sequence_device_bytes;
        self.active_sequences = 1;
        on_lifecycle(RequestLifecycleEvent::Admitted(progress));
        tick.admitted.push(progress);
        Ok(())
    }

    fn prefill(
        &mut self,
        id: Qwen38FlashNextRequestId,
        tick: &mut Qwen38FlashNextTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen38FlashNextRequestId, Qwen38FlashNextAdmissionProgress>,
        ),
    ) -> Result<()> {
        let request = self.requests.get_mut(&id).expect("prefill request exists");
        let end = (request.prompt_position + self.config.prefill_token_capacity)
            .min(request.prompt.len());
        if end == request.prompt_position {
            return Ok(());
        }
        on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
        let sequence = request
            .sequence
            .as_deref_mut()
            .expect("request is admitted");
        for &token in &request.prompt[request.prompt_position..end] {
            request.pending_next =
                Some(sequence.decode_token(&mut self.model, &mut self.sequence_cache, token)?);
        }
        request.prompt_position = end;
        tick.prefilled.push(Qwen38FlashNextPrefillProgress {
            request_id: id,
            prompt_position: end,
        });
        Ok(())
    }

    fn generate_one(
        &mut self,
        id: Qwen38FlashNextRequestId,
        tick: &mut Qwen38FlashNextTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let next = match request.pending_next.take() {
            Some(next) => next,
            None => {
                let token = request
                    .last_token
                    .expect("generated token exists after prompt logits");
                request
                    .sequence
                    .as_deref_mut()
                    .expect("request is admitted")
                    .decode_token(&mut self.model, &mut self.sequence_cache, token)?
            }
        };
        request.generated_tokens += 1;
        request.last_token = Some(next.id);
        request.usage.completion_tokens += 1;
        if request.output.is_reasoning() {
            request.usage.reasoning_tokens += 1;
        }
        tick.generated.push(id);
        let events = request.output.push_token(next.id)?;
        if let Some(reason) = request.filter.apply(id, events, &mut tick.output) {
            return Ok(Some(reason));
        }
        if request.generation.eos_token_ids.contains(&next.id) {
            return Ok(Some(ChatFinishReason::Eos));
        }
        if request.generated_tokens == request.generation.max_new_tokens {
            return Ok(Some(ChatFinishReason::Length));
        }
        Ok(None)
    }

    fn finish_request(
        &mut self,
        id: Qwen38FlashNextRequestId,
        mut reason: ChatFinishReason,
        tick: &mut Qwen38FlashNextTick,
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
        if let Some(sequence) = request.sequence.take() {
            sequence.finish(&mut self.sequence_cache)?;
        }
        self.active_sequences -= 1;
        tick.finished.push(Qwen38FlashNextFinished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: request.sequence_device_bytes,
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
        request_id: Qwen38FlashNextRequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Qwen38FlashNextChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(Qwen38FlashNextChatDelta { request_id, event })
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Qwen38FlashNextChatDelta {
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
                    output.push(Qwen38FlashNextChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(
        &mut self,
        request_id: Qwen38FlashNextRequestId,
        output: &mut Vec<Qwen38FlashNextChatDelta>,
    ) {
        let text = self.stop.finish();
        if !text.is_empty() && !self.saw_tool_calls {
            output.push(Qwen38FlashNextChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
