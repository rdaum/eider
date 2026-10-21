//! Decode-first Flint serving through the common engine protocol.

use crate::flint::{FlintModel, FlintSequence, FlintSequenceCache, FlintWorkspace};
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
    prefix_target: usize,
    prefix_retained: bool,
    state: Option<FlintSequence<'m>>,
    generation: RequestConfig,
    last: Option<u32>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'t>,
    stop: StopBuffer,
    saw_tools: bool,
    parallel_tool_calls: bool,
    usage: ChatUsage,
}

pub(crate) struct FlintChatService<'m, 't> {
    model: &'m FlintModel,
    template: &'t CheckpointChatTemplate,
    config: SchedulerConfig,
    workspace: FlintWorkspace<'m>,
    cache: FlintSequenceCache,
    retain_prefixes: bool,
    decode_cursor: Option<EngineRequestId>,
    prefill_cursor: Option<EngineRequestId>,
    next_id: u64,
    waiting: VecDeque<EngineRequestId>,
    requests: BTreeMap<EngineRequestId, Request<'m, 't>>,
}

impl<'m, 't> FlintChatService<'m, 't> {
    pub(crate) fn new(
        model: &'m FlintModel,
        template: &'t CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.speculative_drafts != 0 {
            return Err(invalid("Flint does not support speculative decoding"));
        }
        if config.max_context_tokens > model.config().max_context {
            return Err(invalid("context exceeds checkpoint limit"));
        }
        for (token, id) in [
            ("<|end_of_text|>", 150001),
            ("<|user|>", 150003),
            ("<|assistant|>", 150004),
            ("<|observation|>", 150005),
            ("<think>", 150006),
            ("</think>", 150007),
            ("<tool_call>", 150010),
            ("</tool_call>", 150011),
            ("<arg_key>", 150012),
            ("</arg_key>", 150013),
            ("<arg_value>", 150014),
            ("</arg_value>", 150015),
            ("<|developer|>", 150018),
        ] {
            if template.tokenizer().token_to_id(token) != Some(id) {
                return Err(invalid(format!("tokenizer must map {token} to {id}")));
            }
        }
        let workspace = model.new_workspace(
            config.prefill_token_capacity.max(config.decode_capacity),
            config.max_context_tokens,
        )?;
        let cache = model.new_sequence_cache(
            config.max_active_sequences,
            config.max_context_tokens,
            cache_config.max_retained_bytes,
        )?;
        Ok(Self {
            cache,
            retain_prefixes: cache_config.max_retained_bytes != 0,
            model,
            template,
            config,
            workspace,
            decode_cursor: None,
            prefill_cursor: None,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
        })
    }
    fn enqueue(&mut self, request: ChatRequest) -> Result<EngineAdmission> {
        request.generation.validate()?;
        if request.stop_sequences.iter().any(String::is_empty) {
            return Err(invalid("empty stop sequence"));
        }
        let prompt = self.template.render_and_tokenize(
            &request.messages,
            &request.tools,
            request.template,
        )?;
        let capacity = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| invalid("request length overflow"))?;
        if prompt.token_ids.is_empty() || capacity > self.config.max_context_tokens {
            return Err(invalid(
                "request exceeds context capacity or has an empty prompt",
            ));
        }
        let id = EngineRequestId::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| invalid("request ID overflow"))?;
        let admission = EngineAdmission {
            request_id: id,
            prompt_tokens: prompt.token_ids.len(),
            max_output_tokens: request.generation.max_new_tokens,
        };
        self.requests.insert(
            id,
            Request {
                position: 0,
                prefix_target: if self.retain_prefixes {
                    self.cache.cacheable_prefix_tokens(prompt.token_ids.len())
                } else {
                    0
                },
                prefix_retained: false,
                state: None,
                generation: request.generation.clone(),
                last: None,
                sampler: Sampler::new(request.generation.sampling)?,
                history: TokenHistory::from_tokens(prompt.token_ids.iter().copied()),
                output: ChatOutputCodec::new(self.template.tokenizer(), &request.tools, false)?,
                stop: StopBuffer::new(request.stop_sequences),
                saw_tools: false,
                parallel_tool_calls: request.parallel_tool_calls,
                usage: ChatUsage {
                    prompt_tokens: prompt.token_ids.len(),
                    ..ChatUsage::default()
                },
                prompt: prompt.token_ids,
            },
        );
        self.waiting.push_back(id);
        Ok(admission)
    }
    fn step(&mut self, observer: &mut dyn FnMut(EngineLifecycleEvent)) -> Result<EngineTick> {
        let start = Instant::now();
        let mut tick = EngineTick::default();
        while self.active_sequence_count() < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("queued request");
            let allocation = Instant::now();
            let Some(mut state) = self.model.admit(
                &mut self.cache,
                (request.prompt.len() + request.generation.max_new_tokens).max(1),
                &request.prompt,
            )?
            else {
                self.waiting.push_front(id);
                break;
            };
            if request.generation.sampling.uses_history_penalties() {
                self.model
                    .initialize_sampling(&mut state, &request.history)?;
            }
            request.position = state.position();
            request.usage.cached_prompt_tokens = state.position();
            request.prefix_retained = state.position() == request.prefix_target;

            observer(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                request_id: id,
                sequence_device_bytes: state.device_bytes(),
                cached_prompt_tokens: state.position(),
                allocation_duration: allocation
                    .elapsed()
                    .saturating_sub(state.restore_duration()),
                checkpoint_copy_duration: state.restore_duration(),
                admitted_after_tick_start: start.elapsed(),
            }));
            request.state = Some(state);
        }
        let mut terminal = BTreeMap::new();
        let mut decode = self
            .requests
            .iter()
            .filter(|(_, r)| {
                r.state.is_some() && r.position == r.prompt.len() && r.generation.max_new_tokens > 0
            })
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        rotate_requests(
            &mut decode,
            &mut self.decode_cursor,
            self.config.decode_capacity,
        );
        let mut batch = self
            .requests
            .iter_mut()
            .filter(|(id, request)| decode.contains(id) && request.last.is_some())
            .map(|(_, request)| {
                (
                    request.state.as_mut().expect("decode state"),
                    request.last.expect("decode token"),
                )
            })
            .collect::<Vec<_>>();
        let packed = batch.len() > 1;
        if packed {
            self.model
                .forward_decode_batch(&mut self.workspace, &mut self.cache, &mut batch)?;
        }
        drop(batch);
        for id in decode {
            let request = self.requests.get_mut(&id).expect("active request");
            let state = request.state.as_mut().expect("active state");
            if let Some(token) = request.last
                && !packed
            {
                self.model
                    .forward(&mut self.workspace, state, &mut self.cache, &[token])?;
            }
            let token = self.model.sample(
                &mut self.workspace,
                state,
                &mut request.sampler,
                &request.history,
            )?;
            request.last = Some(token);
            request.history.push(token);
            request.usage.completion_tokens += 1;
            if request.output.is_reasoning() {
                request.usage.reasoning_tokens += 1;
            }
            tick.generated.push(id);
            // Stop role markers before they enter the user-visible decoder.
            if request.generation.eos_token_ids.contains(&token)
                || [150001, 150003, 150005, 150018].contains(&token)
            {
                terminal.insert(id, ChatFinishReason::Eos);
            } else {
                let events = request.output.push_token(token)?;
                if let Some(reason) = apply_events(request, id, events, &mut tick.output) {
                    terminal.insert(id, reason);
                } else if request.usage.completion_tokens >= request.generation.max_new_tokens {
                    terminal.insert(id, ChatFinishReason::Length);
                }
            }
        }
        let mut prefill = self
            .requests
            .iter()
            .filter(|(_, r)| {
                r.state.is_some() && r.position < r.prompt.len() && r.generation.max_new_tokens > 0
            })
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        rotate_requests(
            &mut prefill,
            &mut self.prefill_cursor,
            self.config.prefill_sequence_capacity,
        );
        let mut budget = self.config.prefill_token_capacity;
        for (i, id) in prefill.iter().copied().enumerate() {
            let request = self.requests.get_mut(&id).expect("active request");
            let until_prefix = if request.prefix_retained || request.prefix_target == 0 {
                usize::MAX
            } else {
                request.prefix_target - request.position
            };
            let rows = (request.prompt.len() - request.position)
                .min(budget.div_ceil(prefill.len() - i))
                .min(until_prefix);
            if rows == 0 {
                continue;
            }
            budget -= rows;
            observer(EngineLifecycleEvent::PrefillStarted(id));
            self.model.forward(
                &mut self.workspace,
                request.state.as_mut().expect("active state"),
                &mut self.cache,
                &request.prompt[request.position..request.position + rows],
            )?;
            request.position += rows;
            if !request.prefix_retained && request.position == request.prefix_target {
                if let Err(error) = self.model.retain_prefix(
                    &mut self.cache,
                    request.state.as_mut().expect("active state"),
                    &request.prompt,
                ) {
                    tracing::warn!(%error, "could not retain Flint prompt prefix");
                }
                request.prefix_retained = true;
            }
            tick.prefilled.push(EnginePrefillProgress {
                request_id: id,
                prompt_position: request.position,
            });
        }
        for (&id, r) in &self.requests {
            if r.state.is_some() && r.generation.max_new_tokens == 0 {
                terminal.insert(id, ChatFinishReason::Length);
            }
        }
        for (id, mut reason) in terminal {
            let mut r = self.requests.remove(&id).expect("terminal request");
            let events = if matches!(reason, ChatFinishReason::Length) {
                r.output.finish_truncated()?
            } else {
                r.output.finish()?
            };
            if let Some(end) = apply_events(&mut r, id, events, &mut tick.output) {
                reason = end;
            }
            let text = r.stop.finish();
            if !text.is_empty() && !r.saw_tools {
                tick.output.push(EngineDelta {
                    request_id: id,
                    event: ChatOutputEvent::Text(text),
                });
            }
            if r.saw_tools && matches!(reason, ChatFinishReason::Eos) {
                reason = ChatFinishReason::ToolCalls;
            }
            let released = r.state.as_ref().map_or(0, FlintSequence::device_bytes);
            if let Some(state) = r.state.take() {
                self.model.finish(&mut self.cache, state)?;
            }
            tick.finished.push(EngineFinished {
                request_id: id,
                finish_reason: reason,
                usage: r.usage,
                released_sequence_device_bytes: released,
            });
        }
        Ok(tick)
    }
}

impl EngineService for FlintChatService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> EngineResult<EngineAdmission> {
        self.enqueue(request).map_err(EngineError::new)
    }
    fn tick(&mut self, observer: &mut dyn FnMut(EngineLifecycleEvent)) -> EngineResult<EngineTick> {
        self.step(observer).map_err(EngineError::new)
    }
    fn cancel_request(&mut self, id: EngineRequestId) -> EngineCancelOutcome {
        self.waiting.retain(|&queued| queued != id);
        match self.requests.remove(&id) {
            Some(mut r) => {
                let released = r.state.as_ref().map_or(0, FlintSequence::device_bytes);
                if let Some(state) = r.state.take()
                    && let Err(error) = self.model.finish(&mut self.cache, state)
                {
                    tracing::error!(%error, "failed to release cancelled Flint sequence");
                }
                EngineCancelOutcome::Cancelled {
                    released_sequence_device_bytes: released,
                }
            }
            None => EngineCancelOutcome::NotFound,
        }
    }
    fn active_sequence_count(&self) -> usize {
        self.requests.values().filter(|r| r.state.is_some()).count()
    }
}

fn apply_events(
    request: &mut Request,
    id: EngineRequestId,
    events: Vec<ChatOutputEvent>,
    out: &mut Vec<EngineDelta>,
) -> Option<ChatFinishReason> {
    for event in events {
        match event {
            ChatOutputEvent::Text(text) if !request.saw_tools => {
                let stopped = request.stop.push(&text);
                if !stopped.text.is_empty() {
                    out.push(EngineDelta {
                        request_id: id,
                        event: ChatOutputEvent::Text(stopped.text),
                    });
                }
                if let Some(sequence) = stopped.matched {
                    return Some(ChatFinishReason::Stop(sequence));
                }
            }
            ChatOutputEvent::ToolCall(call) => {
                let text = request.stop.finish();
                if !text.is_empty() {
                    out.push(EngineDelta {
                        request_id: id,
                        event: ChatOutputEvent::Text(text),
                    });
                }
                request.saw_tools = true;
                out.push(EngineDelta {
                    request_id: id,
                    event: ChatOutputEvent::ToolCall(call),
                });
                if !request.parallel_tool_calls {
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
            ChatOutputEvent::Reasoning(text) if !request.saw_tools => out.push(EngineDelta {
                request_id: id,
                event: ChatOutputEvent::Reasoning(text),
            }),
            _ => {}
        }
    }
    None
}
fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Flint serving",
        detail: detail.into(),
    }
}

fn rotate_requests(
    ids: &mut Vec<EngineRequestId>,
    cursor: &mut Option<EngineRequestId>,
    capacity: usize,
) {
    if let Some(previous) = cursor {
        let offset = ids.partition_point(|id| id <= previous);
        ids.rotate_left(offset);
    }
    ids.truncate(capacity);
    if let Some(last) = ids.last() {
        *cursor = Some(*last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_rotation_does_not_starve_later_sessions() {
        let mut cursor = None;
        let mut selected = Vec::new();
        for _ in 0..6 {
            let mut ids = (1..=3).map(EngineRequestId::new).collect();
            rotate_requests(&mut ids, &mut cursor, 1);
            selected.extend(ids);
        }
        assert_eq!(selected, [1, 2, 3, 1, 2, 3].map(EngineRequestId::new));
        let mut ids = vec![EngineRequestId::new(1), EngineRequestId::new(4)];
        rotate_requests(&mut ids, &mut cursor, 1);
        assert_eq!(ids, [EngineRequestId::new(4)]);
    }
}
