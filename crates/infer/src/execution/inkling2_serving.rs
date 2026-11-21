//! Text-only Inkling 2 serving with complete per-request loop and convolution state.

use super::lfm2_serving::emit;
use crate::inkling2::{Inkling2Model, Inkling2Sequence, Inkling2Workspace};
use eider_cuda::{Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::chat::CheckpointChatTemplate;
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
}

pub(crate) struct Inkling2ChatService<'m, 't> {
    model: &'m Inkling2Model,
    template: &'t CheckpointChatTemplate,
    config: SchedulerConfig,
    prefill: Inkling2Workspace<'m>,
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
        if cache.max_retained_bytes != 0 {
            tracing::warn!(
                "Inkling 2 prefix retention is not implemented; all requests will execute their full prompts"
            );
        }
        Ok(Self {
            model,
            template,
            prefill: model.new_workspace(config.prefill_token_capacity)?,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
        })
    }

    fn add(&mut self, request: ChatRequest) -> Result<EngineAdmission> {
        request.generation.validate()?;
        if !request.tools.is_empty()
            || request.messages.iter().any(|message| {
                !message.tool_calls.is_empty()
                    || message.role == eider_runtime::chat::ChatRole::Tool
            })
        {
            return Err(error(
                "the text-only Inkling 2 adapter does not yet support tool history or definitions",
            ));
        }
        if request.stop_sequences.iter().any(String::is_empty) {
            return Err(error("empty stop sequence"));
        }
        let prompt = self.template.render_and_tokenize(
            &request.messages,
            &request.tools,
            request.template,
        )?;
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| error("prompt plus completion length overflow"))?;
        if prompt.token_ids.is_empty() || total > self.config.max_context_tokens {
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
            prompt_tokens: prompt.token_ids.len(),
            max_output_tokens: request.generation.max_new_tokens,
        };
        // Inkling samples its own message_model header and content-type markers.
        let output = ChatOutputCodec::new(self.template.tokenizer(), &[], false)?;
        let history = TokenHistory::from_tokens(prompt.token_ids.iter().copied());
        let sampler = Sampler::new(request.generation.sampling)?;
        self.requests.insert(
            id,
            Request {
                prompt: prompt.token_ids,
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
            let sequence = self
                .model
                .new_sequence((request.prompt.len() + request.generation.max_new_tokens).max(1))?;
            let bytes = sequence.device_bytes();
            request.sequence = Some(sequence);
            self.waiting.pop_front();
            lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens: 0,
                allocation_duration: allocation.elapsed(),
                checkpoint_copy_duration: Default::default(),
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
                self.model.forward(&mut self.prefill, sequence, &[token])?;
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
            let reason = emit(id, &mut request.stop, events, &mut tick.output)
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
            let end = (request.position + token_budget.min(self.prefill.capacity()))
                .min(request.prompt.len());
            self.model.forward(
                &mut self.prefill,
                sequence,
                &request.prompt[request.position..end],
            )?;
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
                if let Some(protocol) = emit(id, &mut request.stop, events, &mut tick.output) {
                    reason = protocol;
                }
                let tail = request.stop.finish();
                if !tail.is_empty() {
                    tick.output.push(EngineDelta {
                        request_id: id,
                        event: ChatOutputEvent::Text(tail),
                    });
                }
            }
            tick.finished.push(EngineFinished {
                request_id: id,
                finish_reason: reason,
                usage: request.usage,
                released_sequence_device_bytes: request
                    .sequence
                    .as_ref()
                    .map_or(0, Inkling2Sequence::device_bytes),
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
        let Some(request) = self.requests.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        EngineCancelOutcome::Cancelled {
            released_sequence_device_bytes: request
                .sequence
                .as_ref()
                .map_or(0, Inkling2Sequence::device_bytes),
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.sequence.is_some())
            .count()
    }
}

fn error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Inkling 2 serving",
        detail: detail.into(),
    }
}
