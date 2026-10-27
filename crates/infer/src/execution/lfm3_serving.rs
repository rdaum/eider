//! Decode-first LFM3 serving with request-owned convolution and KV state.

use super::lfm2_serving::emit;
use crate::lfm3::{Lfm3Model, Lfm3PrefixCache, Lfm3Sequence, Lfm3Workspace};
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

struct Request<'m, 'a> {
    prompt: Vec<u32>,
    position: usize,
    sequence: Option<Lfm3Sequence<'m>>,
    generation: RequestConfig,
    last_token: Option<u32>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'a>,
    stop: StopBuffer,
    usage: ChatUsage,
}

pub(crate) struct Lfm3ChatService<'m, 't> {
    model: &'m Lfm3Model,
    template: &'t CheckpointChatTemplate,
    config: SchedulerConfig,
    prefill: Lfm3Workspace<'m>,
    prefixes: Lfm3PrefixCache,
    next_id: u64,
    waiting: VecDeque<EngineRequestId>,
    requests: BTreeMap<EngineRequestId, Request<'m, 't>>,
}

impl<'m, 't> Lfm3ChatService<'m, 't> {
    pub(crate) fn new(
        model: &'m Lfm3Model,
        template: &'t CheckpointChatTemplate,
        config: SchedulerConfig,
        cache: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_position_embeddings {
            return Err(error("scheduler context exceeds checkpoint context"));
        }
        Ok(Self {
            model,
            template,
            config,
            prefill: model.new_workspace(config.prefill_token_capacity)?,
            prefixes: Lfm3PrefixCache::new(cache.max_retained_bytes),
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
        // The rendered prefix selects the initial reasoning state.
        let starts_in_reasoning = prompt.text.ends_with("<think>");
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
        let output = ChatOutputCodec::new(
            self.template.tokenizer(),
            &request.tools,
            starts_in_reasoning,
        )?;
        let history = TokenHistory::from_tokens(prompt.token_ids.iter().copied());
        self.requests.insert(
            id,
            Request {
                prompt: prompt.token_ids,
                position: 0,
                sequence: None,
                sampler: Sampler::new(request.generation.sampling)?,
                generation: request.generation,
                last_token: None,
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
            let r = self.requests.get_mut(&id).expect("queued request");
            let allocation = Instant::now();
            let mut sequence = self
                .model
                .new_sequence((r.prompt.len() + r.generation.max_new_tokens).max(1))?;
            let allocation_duration = allocation.elapsed();
            let restore = Instant::now();
            let cached = self
                .prefixes
                .restore(self.model, &mut sequence, &r.prompt)?;
            let checkpoint_copy_duration = restore.elapsed();
            r.position = cached;
            r.usage.cached_prompt_tokens = cached;
            let bytes = sequence.device_bytes();
            r.sequence = Some(sequence);
            self.waiting.pop_front();
            lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens: cached,
                allocation_duration,
                checkpoint_copy_duration,
                admitted_after_tick_start: started.elapsed(),
            }));
        }
        let mut done = Vec::new();
        let mut decode_budget = self.config.decode_capacity;
        for (&id, r) in &mut self.requests {
            let Some(s) = r.sequence.as_mut() else {
                continue;
            };
            if r.generation.max_new_tokens == 0 {
                done.push((id, ChatFinishReason::Length));
                continue;
            }
            if r.position != r.prompt.len() || decode_budget == 0 {
                continue;
            }
            decode_budget -= 1;
            if let Some(token) = r.last_token {
                self.model.forward(&mut self.prefill, s, &[token])?;
            }
            let token = if r.sampler.config().uses_fast_argmax() {
                self.model.argmax_with_logit(s)?.0
            } else {
                r.sampler
                    .sample(&self.model.logits_to_host(s)?, &r.history)?
                    .id
            };
            r.last_token = Some(token);
            r.history.push(token);
            r.usage.completion_tokens += 1;
            r.usage.reasoning_tokens += usize::from(r.output.is_reasoning());
            tick.generated.push(id);
            let events = r.output.push_token(token)?;
            let reason = emit(id, &mut r.stop, events, &mut tick.output)
                .or_else(|| {
                    r.generation
                        .eos_token_ids
                        .contains(&token)
                        .then_some(ChatFinishReason::Eos)
                })
                .or_else(|| {
                    (r.usage.completion_tokens >= r.generation.max_new_tokens)
                        .then_some(ChatFinishReason::Length)
                });
            if let Some(reason) = reason {
                done.push((id, reason));
            }
        }
        let mut token_budget = self.config.prefill_token_capacity;
        let prefix_interval = self.config.prefill_token_capacity;
        let mut sequence_budget = self.config.prefill_sequence_capacity;
        for (&id, r) in &mut self.requests {
            let Some(s) = r.sequence.as_mut() else {
                continue;
            };
            if token_budget == 0 || sequence_budget == 0 {
                break;
            }
            if r.position == r.prompt.len() || r.generation.max_new_tokens == 0 {
                continue;
            }
            lifecycle(EngineLifecycleEvent::PrefillStarted(id));
            let target = r.prompt.len().saturating_sub(1);
            let boundary = if self.prefixes.enabled() && r.position < target {
                // Hybrid state can resume only at a retained endpoint. Keep
                // regular checkpoints as well as the final prompt prefix.
                ((r.position / prefix_interval + 1) * prefix_interval).min(target)
            } else {
                r.prompt.len()
            };
            let end = boundary.min(r.position + token_budget);
            self.model
                .forward(&mut self.prefill, s, &r.prompt[r.position..end])?;
            if self.prefixes.enabled() && (end == target || end.is_multiple_of(prefix_interval)) {
                self.prefixes.retain(self.model, s, &r.prompt[..end])?;
            }
            token_budget -= end - r.position;
            sequence_budget -= 1;
            r.position = end;
            tick.prefilled.push(EnginePrefillProgress {
                request_id: id,
                prompt_position: end,
            });
        }
        // Finish queued work before any request state can be dropped or cancelled.
        self.model.synchronize()?;
        for (id, mut reason) in done {
            let mut r = self.requests.remove(&id).expect("finished request");
            let events = if matches!(reason, ChatFinishReason::Length | ChatFinishReason::Stop(_)) {
                r.output.finish_truncated()?
            } else {
                r.output.finish()?
            };
            if !matches!(reason, ChatFinishReason::Stop(_)) {
                if let Some(protocol) = emit(id, &mut r.stop, events, &mut tick.output) {
                    reason = protocol;
                }
                let tail = r.stop.finish();
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
                usage: r.usage,
                released_sequence_device_bytes: r
                    .sequence
                    .as_ref()
                    .map_or(0, Lfm3Sequence::device_bytes),
            });
        }
        Ok(tick)
    }
}

impl EngineService for Lfm3ChatService<'_, '_> {
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
        let Some(r) = self.requests.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        EngineCancelOutcome::Cancelled {
            released_sequence_device_bytes: r
                .sequence
                .as_ref()
                .map_or(0, Lfm3Sequence::device_bytes),
        }
    }
    fn active_sequence_count(&self) -> usize {
        self.requests
            .values()
            .filter(|r| r.sequence.is_some())
            .count()
    }
}

fn error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "LFM3 serving",
        detail: detail.into(),
    }
}
