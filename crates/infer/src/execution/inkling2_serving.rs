//! Inkling 2 text and tool serving with shared global KV pages and retained prompt prefixes.

use crate::inkling2::{Inkling2Model, Inkling2Sequence, Inkling2SequenceCache, Inkling2Workspace};
use eider_cuda::{Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::chat::{CheckpointChatTemplate, Inkling2PromptRenderer};
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineDelta, EngineError,
    EngineFinished, EngineLifecycleEvent, EnginePrefillProgress, EngineRequestId, EngineResult,
    EngineService, EngineTick,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::sampling::{Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

struct Request<'m, 't> {
    prompt: Vec<u32>,
    position: usize,
    sequence: Option<Inkling2Sequence<'m>>,
    generation: RequestConfig,
    last_token: Option<u32>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'t>,
    stop: StopBuffer,
    usage: ChatUsage,
    called_tools: bool,
    /// Page-aligned prompt position whose history is retained, or zero.
    prefix_target: usize,
    prefix_retained: bool,
}

pub(crate) struct Inkling2ChatService<'m, 't> {
    model: &'m Inkling2Model,
    template: &'t CheckpointChatTemplate,
    renderer: Inkling2PromptRenderer,
    config: SchedulerConfig,
    prefill: Inkling2Workspace<'m>,
    cache: Inkling2SequenceCache,
    retain_prefixes: bool,
    next_id: u64,
    waiting: VecDeque<EngineRequestId>,
    requests: BTreeMap<EngineRequestId, Request<'m, 't>>,
}

impl<'m, 't> Inkling2ChatService<'m, 't> {
    pub(crate) fn new(
        model: &'m Inkling2Model,
        template: &'t CheckpointChatTemplate,
        config: SchedulerConfig,
        cache: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().text_config.model_max_length {
            return Err(error("scheduler context exceeds checkpoint context"));
        }
        Ok(Self {
            model,
            template,
            renderer: Inkling2PromptRenderer::new(template.tokenizer())?,
            prefill: model.new_workspace(config.prefill_token_capacity)?,
            cache: model.new_sequence_cache(
                config.max_active_sequences,
                config.max_context_tokens,
                cache.max_retained_bytes,
            )?,
            retain_prefixes: cache.max_retained_bytes != 0,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
        })
    }

    fn add(&mut self, request: ChatRequest) -> Result<EngineAdmission> {
        request.generation.validate()?;
        if request.stop_sequences.iter().any(String::is_empty) {
            return Err(error("empty stop sequence"));
        }
        // The reference frontend renders Inkling natively rather than through
        // the checkpoint's Jinja template.
        let prompt = self.renderer.render(
            &request.messages,
            &request.tools,
            request.template,
            request.parallel_tool_calls,
        )?;
        let total = prompt
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| error("prompt plus completion length overflow"))?;
        if prompt.is_empty() || total > self.config.max_context_tokens {
            return Err(error(format!(
                "request requires {total} tokens; context limit is {}",
                self.config.max_context_tokens
            )));
        }
        let id = EngineRequestId::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| error("request IDs exhausted"))?;
        let admission = EngineAdmission {
            request_id: id,
            prompt_tokens: prompt.len(),
            max_output_tokens: request.generation.max_new_tokens,
        };
        // Inkling samples its own message_model header and content-type markers.
        let output = ChatOutputCodec::new(self.template.tokenizer(), &request.tools, false)?;
        let history = TokenHistory::from_tokens(prompt.iter().copied());
        let sampler = Sampler::new(request.generation.sampling)?;
        self.requests.insert(
            id,
            Request {
                prompt,
                position: 0,
                sequence: None,
                generation: request.generation,
                last_token: None,
                sampler,
                history,
                output,
                stop: StopBuffer::new(request.stop_sequences),
                usage: ChatUsage {
                    prompt_tokens: admission.prompt_tokens,
                    ..ChatUsage::default()
                },
                called_tools: false,
                prefix_target: if self.retain_prefixes {
                    self.cache.cacheable_prefix_tokens(admission.prompt_tokens)
                } else {
                    0
                },
                prefix_retained: false,
            },
        );
        self.waiting.push_back(id);
        Ok(admission)
    }

    fn run_tick(&mut self, lifecycle: &mut dyn FnMut(EngineLifecycleEvent)) -> Result<EngineTick> {
        let started = Instant::now();
        let mut tick = EngineTick::default();
        while self.active_sequence_count() < self.config.max_active_sequences {
            let Some(&id) = self.waiting.front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("queued request");
            let allocation = Instant::now();
            // The cache reserves worst-case pages; admission waits when it is full.
            let Some(sequence) = self.model.admit(
                &mut self.cache,
                (request.prompt.len() + request.generation.max_new_tokens).max(1),
                &request.prompt,
            )?
            else {
                break;
            };
            let cached = sequence.position();
            request.position = cached;
            request.usage.cached_prompt_tokens = cached;
            request.prefix_retained = cached == request.prefix_target;
            let bytes = sequence.device_bytes();
            let restore = sequence.restore_duration();
            request.sequence = Some(sequence);
            self.waiting.pop_front();
            lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens: cached,
                allocation_duration: allocation.elapsed().saturating_sub(restore),
                checkpoint_copy_duration: restore,
                admitted_after_tick_start: started.elapsed(),
            }));
        }
        let mut done = Vec::new();
        let mut decode_budget = self.config.decode_capacity;
        for (&id, request) in &mut self.requests {
            let Some(sequence) = request.sequence.as_mut() else {
                continue;
            };
            if request.generation.max_new_tokens == 0 {
                done.push((id, ChatFinishReason::Length));
                continue;
            }
            if request.position != request.prompt.len() || decode_budget == 0 {
                continue;
            }
            decode_budget -= 1;
            if let Some(token) = request.last_token {
                self.model
                    .forward(&mut self.prefill, sequence, &mut self.cache, &[token])?;
            }
            let token = if request.sampler.config().uses_fast_argmax() {
                self.model.argmax_with_logit(sequence)?.0
            } else {
                request
                    .sampler
                    .sample(&self.model.logits_to_host(sequence)?, &request.history)?
                    .id
            };
            request.last_token = Some(token);
            request.history.push(token);
            request.usage.completion_tokens += 1;
            request.usage.reasoning_tokens += usize::from(request.output.is_reasoning());
            tick.generated.push(id);
            let events = request.output.push_token(token)?;
            let called_tools = &mut request.called_tools;
            let reason = emit(
                id,
                &mut request.stop,
                events,
                &mut tick.output,
                called_tools,
            )
            .or_else(|| {
                request
                    .generation
                    .eos_token_ids
                    .contains(&token)
                    .then_some(ChatFinishReason::Eos)
            })
            .or_else(|| {
                (request.usage.completion_tokens >= request.generation.max_new_tokens)
                    .then_some(ChatFinishReason::Length)
            });
            if let Some(reason) = reason {
                done.push((id, reason));
            }
        }
        let mut token_budget = self.config.prefill_token_capacity;
        let mut sequence_budget = self.config.prefill_sequence_capacity;
        for (&id, request) in &mut self.requests {
            let Some(sequence) = request.sequence.as_mut() else {
                continue;
            };
            if token_budget == 0 || sequence_budget == 0 {
                break;
            }
            if request.position == request.prompt.len() || request.generation.max_new_tokens == 0 {
                continue;
            }
            lifecycle(EngineLifecycleEvent::PrefillStarted(id));
            // Stop at the retained-prefix boundary so its history can be copied.
            let limit = if request.prefix_retained || request.prefix_target == 0 {
                request.prompt.len()
            } else {
                request.prefix_target
            };
            let end = (request.position + token_budget.min(self.prefill.capacity())).min(limit);
            self.model.forward(
                &mut self.prefill,
                sequence,
                &mut self.cache,
                &request.prompt[request.position..end],
            )?;
            if !request.prefix_retained && end == request.prefix_target {
                if let Err(error) =
                    self.model
                        .retain_prefix(&mut self.cache, sequence, &request.prompt)
                {
                    tracing::warn!(%error, "could not retain Inkling 2 prompt prefix");
                }
                request.prefix_retained = true;
            }
            token_budget -= end - request.position;
            sequence_budget -= 1;
            request.position = end;
            tick.prefilled.push(EnginePrefillProgress {
                request_id: id,
                prompt_position: end,
            });
        }
        self.model.synchronize()?;
        for (id, mut reason) in done {
            let mut request = self.requests.remove(&id).expect("finished request");
            let events = if matches!(reason, ChatFinishReason::Length | ChatFinishReason::Stop(_)) {
                request.output.finish_truncated()?
            } else {
                request.output.finish()?
            };
            if !matches!(reason, ChatFinishReason::Stop(_)) {
                if let Some(stop) = emit(
                    id,
                    &mut request.stop,
                    events,
                    &mut tick.output,
                    &mut request.called_tools,
                ) {
                    reason = stop;
                }
                let tail = request.stop.finish();
                if !tail.is_empty() {
                    tick.output.push(EngineDelta {
                        request_id: id,
                        event: ChatOutputEvent::Text(tail),
                    });
                }
            }
            if reason == ChatFinishReason::Eos && request.called_tools {
                reason = ChatFinishReason::ToolCalls;
            }
            let released = request
                .sequence
                .as_ref()
                .map_or(0, Inkling2Sequence::device_bytes);
            if let Some(sequence) = request.sequence.take() {
                self.model.finish(&mut self.cache, sequence)?;
            }
            tick.finished.push(EngineFinished {
                request_id: id,
                finish_reason: reason,
                usage: request.usage,
                released_sequence_device_bytes: released,
            });
        }
        Ok(tick)
    }
}

impl EngineService for Inkling2ChatService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> EngineResult<EngineAdmission> {
        self.add(request).map_err(EngineError::new)
    }

    fn tick(
        &mut self,
        lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> EngineResult<EngineTick> {
        self.run_tick(lifecycle).map_err(EngineError::new)
    }

    fn cancel_request(&mut self, id: EngineRequestId) -> EngineCancelOutcome {
        let Some(mut request) = self.requests.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request
            .sequence
            .as_ref()
            .map_or(0, Inkling2Sequence::device_bytes);
        if let Some(sequence) = request.sequence.take()
            && let Err(error) = self.model.finish(&mut self.cache, sequence)
        {
            tracing::error!(%error, "failed to release cancelled Inkling 2 sequence");
        }
        EngineCancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.sequence.is_some())
            .count()
    }
}

/// Forwards codec events. Inkling writes every parallel call as its own block
/// and may follow calls with text, so a tool call does not end the turn: the
/// model's end-of-sampling token does. Only a stop sequence ends it early.
fn emit(
    id: EngineRequestId,
    stop: &mut StopBuffer,
    events: Vec<ChatOutputEvent>,
    output: &mut Vec<EngineDelta>,
    called_tools: &mut bool,
) -> Option<ChatFinishReason> {
    for event in events {
        match event {
            ChatOutputEvent::Text(text) => {
                let chunk = stop.push(&text);
                if !chunk.text.is_empty() {
                    output.push(EngineDelta {
                        request_id: id,
                        event: ChatOutputEvent::Text(chunk.text),
                    });
                }
                if let Some(matched) = chunk.matched {
                    return Some(ChatFinishReason::Stop(matched));
                }
            }
            event @ ChatOutputEvent::ToolCall(_) => {
                // Release held-back text so it stays ordered before the call.
                let tail = stop.finish();
                if !tail.is_empty() {
                    output.push(EngineDelta {
                        request_id: id,
                        event: ChatOutputEvent::Text(tail),
                    });
                }
                output.push(EngineDelta {
                    request_id: id,
                    event,
                });
                *called_tools = true;
            }
            event @ ChatOutputEvent::Reasoning(_) => output.push(EngineDelta {
                request_id: id,
                event,
            }),
        }
    }
    None
}

fn error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Inkling 2 serving",
        detail: detail.into(),
    }
}
