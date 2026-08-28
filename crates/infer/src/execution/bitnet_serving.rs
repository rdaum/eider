//! Multi-session chat serving for BitNet.

use crate::bitnet::{BitNetModel, BitNetPrefillWorkspace, BitNetSequenceId, BitNetSequencePool};
use crate::bitnet::{BitNetSequence, BitNetSequenceCache, new_bitnet_sequence_cache};
use crate::{InferenceError, InferenceResult};
use eider_cuda::{Error, Result};
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineDelta, EngineFinished,
    EngineLifecycleEvent, EnginePrefillProgress, EngineService, EngineTick,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::sampling::{Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Stable identity assigned to a BitNet request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct BitNetRequestId(u64);

impl BitNetRequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after rendering and tokenization.
pub struct BitNetAdmission {
    /// Assigned request identity.
    pub request_id: BitNetRequestId,
    /// Rendered prompt token count.
    pub prompt_tokens: usize,
    /// Requested completion-token limit.
    pub max_output_tokens: usize,
}

/// Device allocation completed during a service tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BitNetAdmissionProgress {
    /// Admitted request.
    pub request_id: BitNetRequestId,
    /// Sequence-specific device bytes.
    pub sequence_device_bytes: usize,
    /// Tokens restored from a retained prefix; currently always zero for BitNet.
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time at admission.
    pub admitted_after_tick_start: Duration,
}

/// Prompt progress completed during one tick.
pub struct BitNetPrefillProgress {
    /// Request whose prompt advanced.
    pub request_id: BitNetRequestId,
    /// Total prompt position after this tick.
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct BitNetChatDelta {
    /// Request owning this delta.
    pub request_id: BitNetRequestId,
    /// Reasoning, visible text, or tool-call output.
    pub event: ChatOutputEvent,
}

/// Terminal request metadata.
pub struct BitNetFinished {
    /// Finished request.
    pub request_id: BitNetRequestId,
    /// API-facing finish reason.
    pub finish_reason: ChatFinishReason,
    /// Final token usage.
    pub usage: ChatUsage,
    /// Sequence device bytes released at completion.
    pub released_sequence_device_bytes: usize,
}

/// Work and output from one service iteration.
#[derive(Default)]
pub struct BitNetTick {
    /// Requests allocated during this tick.
    pub admitted: Vec<BitNetAdmissionProgress>,
    /// Prompt progress during this tick.
    pub prefilled: Vec<BitNetPrefillProgress>,
    /// Requests producing a token during this tick.
    pub generated: Vec<BitNetRequestId>,
    /// Structured streaming deltas.
    pub output: Vec<BitNetChatDelta>,
    /// Requests completing during this tick.
    pub finished: Vec<BitNetFinished>,
    /// Device-resident sequences remaining after the tick.
    pub active_sequences: usize,
}

/// Outcome of cancelling a queued or active request.
pub enum BitNetCancelOutcome {
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
    prompt_logits_ready: bool,
    sequence_id: Option<BitNetSequenceId>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint rendering and decode-first BitNet execution.
pub struct BitNetChatService<'model, 'template> {
    model: &'model BitNetModel,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<BitNetRequestId>,
    requests: BTreeMap<BitNetRequestId, ActiveRequest<'template>>,
    sequences: BitNetSequencePool,
    sequence_cache: BitNetSequenceCache,
    prefill_workspace: BitNetPrefillWorkspace,
}

impl<'model, 'template> BitNetChatService<'model, 'template> {
    /// Creates a service with explicit scheduling limits.
    pub fn new(
        model: &'model BitNetModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_context {
            return Err(Error::Shape {
                label: "BitNet scheduler context",
                expected: format!("at most {} tokens", model.config().max_context),
                actual: format!("{} tokens", config.max_context_tokens),
            });
        }
        let warmup_started = Instant::now();
        let warmup_rows = config.prefill_token_capacity.min(config.max_context_tokens);
        let mut sequence_cache = new_bitnet_sequence_cache(
            model,
            config.max_active_sequences,
            config.max_context_tokens,
        )?;
        let mut prefill_workspace =
            model.new_prefill_workspace(warmup_rows, config.max_context_tokens)?;
        let mut warmup_sequence = BitNetSequence::admit(model, &mut sequence_cache, warmup_rows)?;
        model.prefill(
            &mut prefill_workspace,
            &mut warmup_sequence,
            &vec![0; warmup_rows],
            &mut sequence_cache,
        )?;
        warmup_sequence.finish(&mut sequence_cache)?;
        info!(
            tokens = warmup_rows,
            elapsed_ms = warmup_started.elapsed().as_secs_f64() * 1000.0,
            "warmed BitNet prefill path"
        );
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            sequences: BitNetSequencePool::new(),
            sequence_cache,
            prefill_workspace,
        })
    }

    /// Renders, tokenizes, and queues a request without allocating GPU state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<BitNetAdmission> {
        request.generation.validate()?;
        if !request.tools.is_empty() {
            return Err(Error::Format {
                label: "BitNet chat tools",
                detail: "the official BitNet checkpoint template has no tool-call protocol"
                    .to_string(),
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
                label: "BitNet chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "BitNet request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "BitNet request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = BitNetRequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "BitNet request ID",
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
                sequence_id: None,
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
        Ok(BitNetAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one decode-first scheduling iteration.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<BitNetRequestId, BitNetAdmissionProgress>,
        ),
    ) -> Result<BitNetTick> {
        let started = Instant::now();
        let mut tick = BitNetTick::default();
        self.admit(&mut tick, started, on_lifecycle)?;
        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.sequence_id.is_some()
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
                request.sequence_id.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick, on_lifecycle)?;
        for (&id, request) in &self.requests {
            if request.sequence_id.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        tick.active_sequences = self.sequences.len();
        Ok(tick)
    }

    /// Cancels a queued or active request.
    pub fn cancel_request(&mut self, id: BitNetRequestId) -> BitNetCancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return BitNetCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.sequence_id.and_then(|sequence_id| self.sequences.release(sequence_id).ok()).map_or(0, |sequence| { let bytes = sequence.device_bytes(); if let Err(error) = sequence.finish(&mut self.sequence_cache) { warn!(%error, request_id = id.get(), "failed to release cancelled BitNet sequence"); } bytes });
        BitNetCancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns the number of requests with device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.sequences.len()
    }

    fn admit(
        &mut self,
        tick: &mut BitNetTick,
        started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<BitNetRequestId, BitNetAdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.sequences.len() < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let sequence =
                BitNetSequence::admit(self.model, &mut self.sequence_cache, capacity.max(1))?;
            let progress = BitNetAdmissionProgress {
                request_id: id,
                sequence_device_bytes: sequence.device_bytes(),
                cached_prompt_tokens: 0,
                admitted_after_tick_start: started.elapsed(),
            };
            request.sequence_id = Some(self.sequences.insert(sequence)?);
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        ids: &[BitNetRequestId],
        tick: &mut BitNetTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<BitNetRequestId, BitNetAdmissionProgress>,
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
            let sequence_id = request.sequence_id.expect("request is admitted");
            let mut sequence = self.sequences.lease(sequence_id)?;
            self.model.prefill(
                &mut self.prefill_workspace,
                sequence.sequence_mut(),
                &request.prompt[start..end],
                &mut self.sequence_cache,
            )?;
            request.prompt_position = end;
            request.prompt_logits_ready = end == request.prompt.len();
            tick.prefilled.push(BitNetPrefillProgress {
                request_id: id,
                prompt_position: end,
            });
        }
        Ok(())
    }

    fn generate_one(
        &mut self,
        id: BitNetRequestId,
        tick: &mut BitNetTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let mut sequence = self
            .sequences
            .lease(request.sequence_id.expect("decode request is admitted"))?;
        let sequence = sequence.sequence_mut();
        if request.prompt_logits_ready {
            request.prompt_logits_ready = false;
        } else {
            self.model.forward_one(
                sequence,
                request
                    .last_token
                    .expect("generated token exists after prompt logits"),
                &mut self.sequence_cache,
            )?;
        }
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(sequence)?;
            eider_runtime::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            request
                .sampler
                .sample(&self.model.logits_to_host(sequence)?, &request.history)?
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
        id: BitNetRequestId,
        mut reason: ChatFinishReason,
        tick: &mut BitNetTick,
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
        let sequence = self.sequences.release(
            request
                .sequence_id
                .take()
                .expect("terminal request is admitted"),
        )?;
        let released = sequence.device_bytes();
        sequence.finish(&mut self.sequence_cache)?;
        tick.finished.push(BitNetFinished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

impl EngineService for BitNetChatService<'_, '_> {
    type Error = InferenceError;

    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = BitNetChatService::add_request(self, request)?;
        let id = admission.request_id.get();
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<BitNetRequestId, BitNetAdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                        request_id: progress.request_id.get(),
                        sequence_device_bytes: progress.sequence_device_bytes,
                        cached_prompt_tokens: progress.cached_prompt_tokens,
                        allocation_duration: Duration::ZERO,
                        checkpoint_copy_duration: Duration::ZERO,
                        admitted_after_tick_start: progress.admitted_after_tick_start,
                    }));
                }
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = BitNetChatService::tick_with_lifecycle(self, &mut observer)?;
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(BitNetRequestId::get)
                .collect(),
            speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        match BitNetChatService::cancel_request(self, BitNetRequestId(id)) {
            BitNetCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            BitNetCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        BitNetChatService::active_sequence_count(self)
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
        request_id: BitNetRequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<BitNetChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(BitNetChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(BitNetChatDelta {
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
                    output.push(BitNetChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: BitNetRequestId, output: &mut Vec<BitNetChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(BitNetChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
