//! Dory serving with request-owned per-loop KV and bounded prompt retention.

use crate::dory::{
    DoryDecisionReadout, DoryDecisionSequence, DoryModel, DoryPrefillWorkspace, DoryPrefixCache,
    DorySequence,
};
use eider_cuda::{Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::decision::{DecisionRequest, validated_decision_labels};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineCapabilities,
    EngineDecisionAdmission, EngineDecisionFailed, EngineDelta, EngineError, EngineFinished,
    EngineLifecycleEvent, EnginePrefillProgress, EngineRequestId, EngineResult, EngineService,
    EngineTick,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::sampling::{Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

mod decision;
use decision::DecisionGroup;

struct Request<'m, 'a> {
    prompt: Vec<u32>,
    position: usize,
    sequence: Option<DorySequence<'m>>,
    generation: RequestConfig,
    last_token: Option<u32>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'a>,
    stop: StopBuffer,
    usage: ChatUsage,
}

pub(crate) struct DoryChatService<'m, 't> {
    model: &'m DoryModel,
    template: &'t CheckpointChatTemplate,
    config: SchedulerConfig,
    prefill: DoryPrefillWorkspace,
    prefixes: DoryPrefixCache,
    idle_sequence: Option<DorySequence<'m>>,
    idle_decision: Option<DoryDecisionSequence<'m>>,
    next_id: u64,
    waiting: VecDeque<EngineRequestId>,
    requests: BTreeMap<EngineRequestId, Request<'m, 't>>,
    decision_readout: DoryDecisionReadout<'m>,
    decisions: BTreeMap<EngineRequestId, DecisionGroup<'m>>,
    decision_ready: VecDeque<EngineRequestId>,
}

impl<'m, 't> DoryChatService<'m, 't> {
    pub(crate) fn new(
        model: &'m DoryModel,
        template: &'t CheckpointChatTemplate,
        config: SchedulerConfig,
        cache: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_position_embeddings {
            return Err(error("scheduler context exceeds checkpoint context"));
        }
        let labels = validated_decision_labels(template.tokenizer())
            .map_err(|err| error(err.to_string()))?
            .into_iter()
            .map(|(_, id)| id)
            .collect::<Vec<_>>();
        let mut prefill = model.new_prefill_workspace(config.prefill_token_capacity)?;
        let idle_sequence = if config.max_active_sequences == 1 {
            // Reserve the measured chat working set before accepting requests.
            // Larger contexts grow the allocation; the configured limit is unchanged.
            let capacity = config.max_context_tokens.min(32_768);
            let mut sequence = model.new_sequence(capacity)?;
            let rows = config.prefill_token_capacity.min(32).min(capacity);
            model.prefill_hidden(&mut prefill, &mut sequence, &vec![0; rows])?;
            if rows < capacity {
                model.decode(&mut sequence, 0)?;
            }
            sequence.reset()?;
            tracing::info!(
                reserved_tokens = capacity,
                reserved_bytes = sequence.device_bytes(),
                "prepared Dory decode workspace"
            );
            Some(sequence)
        } else {
            None
        };
        Ok(Self {
            model,
            template,
            config,
            prefill,
            prefixes: DoryPrefixCache::new(cache.max_retained_bytes),
            idle_sequence,
            idle_decision: None,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            decision_readout: model.new_decision_readout(&labels)?,
            decisions: BTreeMap::new(),
            decision_ready: VecDeque::new(),
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
        let starts_in_reasoning = prompt.text.trim_end().ends_with("<think>");
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
            if self.decisions.contains_key(&id) {
                self.waiting.pop_front();
                if let Err(err) = self.admit_decision_sequence(id) {
                    self.model.synchronize()?;
                    self.decisions.remove(&id);
                    tick.decisions_failed.push(EngineDecisionFailed {
                        request_id: id,
                        message: err.to_string(),
                        released_sequence_device_bytes: 0,
                    });
                }
                continue;
            }
            let r = self.requests.get_mut(&id).expect("queued request");
            let allocation = Instant::now();
            let required = (r.prompt.len() + r.generation.max_new_tokens).max(1);
            let mut sequence = if let Some(mut sequence) = self
                .idle_sequence
                .take()
                .filter(|s| s.capacity() >= required)
            {
                sequence.reset()?;
                sequence
            } else {
                // One idle allocation is retained for sequential chat. Bound its
                // growth to twice the requested capacity and the context limit.
                let capacity = if self.config.max_active_sequences == 1 {
                    (if required < 8192 {
                        required.next_power_of_two()
                    } else {
                        required.next_multiple_of(8192)
                    })
                    .min(self.config.max_context_tokens)
                } else {
                    required
                };
                self.model.new_sequence(capacity)?
            };
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
                self.model.decode(s, token)?;
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
            // Snapshots can restore a partial prefix, so the final prompt token
            // stays in the batch instead of paying for an extra backbone pass.
            let end = (r.position + token_budget).min(r.prompt.len());
            let tokens = &r.prompt[r.position..end];
            if end == r.prompt.len() {
                self.model.prefill(&mut self.prefill, s, tokens)?;
            } else {
                self.model.prefill_hidden(&mut self.prefill, s, tokens)?;
            }
            if self.prefixes.enabled() && end == r.prompt.len() {
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
        self.run_decisions(&mut tick, &mut token_budget, &mut sequence_budget)?;
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
                    .map_or(0, DorySequence::device_bytes),
            });
            if self.config.max_active_sequences == 1 {
                self.idle_sequence = r.sequence.take();
            }
        }
        Ok(tick)
    }
}

impl EngineService for DoryChatService<'_, '_> {
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities { decisions: true }
    }

    fn add_decision(&mut self, request: DecisionRequest) -> EngineResult<EngineDecisionAdmission> {
        self.queue_decision(request).map_err(EngineError::new)
    }

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
        if let Some(mut group) = self.decisions.remove(&id) {
            self.waiting.retain(|&waiting| waiting != id);
            self.decision_ready.retain(|&ready| ready != id);
            let bytes = group.device_bytes();
            if self.config.max_active_sequences == 1 && group.sequence.is_some() {
                self.idle_decision = group.sequence.take();
            }
            return EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: bytes,
            };
        }
        let Some(mut r) = self.requests.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let bytes = r.sequence.as_ref().map_or(0, DorySequence::device_bytes);
        if self.config.max_active_sequences == 1 && r.sequence.is_some() {
            self.idle_sequence = r.sequence.take();
        }
        EngineCancelOutcome::Cancelled {
            released_sequence_device_bytes: bytes,
        }
    }
    fn active_sequence_count(&self) -> usize {
        self.requests
            .values()
            .filter(|r| r.sequence.is_some())
            .count()
            + self
                .decisions
                .values()
                .filter(|group| group.sequence.is_some())
                .count()
    }
}

fn emit(
    id: EngineRequestId,
    stop: &mut StopBuffer,
    events: Vec<ChatOutputEvent>,
    output: &mut Vec<EngineDelta>,
) -> Option<ChatFinishReason> {
    let mut tools = false;
    for event in events {
        match event {
            ChatOutputEvent::Text(text) if !tools => {
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
            ChatOutputEvent::ToolCall(_) => {
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
                tools = true;
            }
            ChatOutputEvent::Reasoning(_) if !tools => output.push(EngineDelta {
                request_id: id,
                event,
            }),
            _ => {}
        }
    }
    tools.then_some(ChatFinishReason::ToolCalls)
}

fn error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Dory serving",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::chat::{ChatFunctionCall, ChatToolCall};

    #[test]
    fn emits_all_parallel_tool_calls_before_finishing() {
        let id = EngineRequestId::new(1);
        let calls = ["first", "second"].map(|name| {
            ChatOutputEvent::ToolCall(ChatToolCall {
                id: name.into(),
                function: ChatFunctionCall {
                    name: name.into(),
                    arguments: BTreeMap::new(),
                },
            })
        });
        let mut output = Vec::new();
        let mut stop = StopBuffer::new(Vec::new());
        assert_eq!(
            emit(id, &mut stop, calls.to_vec(), &mut output),
            Some(ChatFinishReason::ToolCalls)
        );
        assert_eq!(output.len(), 2);
        assert_eq!(output[1].event, calls[1]);
    }

    #[test]
    fn stop_sequence_spanning_output_events_is_not_emitted() {
        let id = EngineRequestId::new(1);
        let mut output = Vec::new();
        let mut stop = StopBuffer::new(vec!["STOP".into()]);
        assert_eq!(
            emit(
                id,
                &mut stop,
                vec![ChatOutputEvent::Text("hello ST".into())],
                &mut output
            ),
            None
        );
        assert_eq!(
            emit(
                id,
                &mut stop,
                vec![ChatOutputEvent::Text("OP ignored".into())],
                &mut output
            ),
            Some(ChatFinishReason::Stop("STOP".into()))
        );
        assert_eq!(output[0].event, ChatOutputEvent::Text("hello ".into()));
        assert_eq!(output.len(), 1);
    }
}
