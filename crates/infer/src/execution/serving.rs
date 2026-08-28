//! Structured chat serving over the Qwen3.6 continuous scheduler.

use super::scheduler::{
    Qwen36AdmissionProgress, Qwen36CancelOutcome, Qwen36PrefillProgress, Qwen36RequestId,
    Qwen36Scheduler, Qwen38SpeculativeProgress,
};
use crate::qwen3::qwen36::Qwen36TextModel;
use eider_cuda::{Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
#[cfg(test)]
use eider_runtime::chat::ChatMessage;
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineDelta, EngineError,
    EngineFinished, EngineLifecycleEvent, EnginePrefillProgress, EngineRequestId, EngineResult,
    EngineService, EngineTick, EngineVerificationProgress,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
#[cfg(test)]
use eider_runtime::scheduler::RequestConfig;
use eider_runtime::scheduler::{
    RequestFinishReason, RequestLifecycleEvent, RequestState, SchedulerConfig,
};
use eider_runtime::stop::StopBuffer;
use eider_runtime::tool_grammar::QwenXmlGrammarFactory;
use std::collections::BTreeMap;
use std::time::Duration;

/// One request-scoped structured output delta.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36ChatDelta {
    /// Scheduler request that owns the output.
    pub request_id: Qwen36RequestId,
    /// Reasoning, visible text, or a completed tool call.
    pub event: ChatOutputEvent,
}

/// Request metadata known after rendering and CPU queueing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen36ChatAdmission {
    /// Scheduler identity assigned to the queued request.
    pub request_id: Qwen36RequestId,
    /// Tokens in the rendered prompt.
    pub prompt_tokens: usize,
    /// Maximum completion tokens requested.
    pub max_output_tokens: usize,
}

/// Terminal request metadata emitted exactly once by the serving bridge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen36ChatFinished {
    /// Finished scheduler request.
    pub request_id: Qwen36RequestId,
    /// API-facing reason for stopping generation.
    pub finish_reason: ChatFinishReason,
    /// Final prompt and completion token counts.
    pub usage: ChatUsage,
    /// Persistent sequence-state bytes released at termination.
    pub released_sequence_device_bytes: usize,
}

/// Observable work and output from one decode-first scheduler iteration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen36ChatTick {
    /// Requests receiving persistent device sequence state during the tick.
    /// Requests selected for model work, preserving scheduler order.
    pub scheduled: Vec<Qwen36RequestId>,
    /// Chunked prompt progress made during the tick.
    pub prefilled: Vec<Qwen36PrefillProgress>,
    /// One entry for each completion token selected during the tick.
    pub generated: Vec<Qwen36RequestId>,
    /// Qwen3.8 draft acceptance observed during the tick.
    pub speculative: Vec<Qwen38SpeculativeProgress>,
    /// Structured output safe to stream to API clients.
    pub output: Vec<Qwen36ChatDelta>,
    /// Requests reaching a serving-level terminal state.
    pub finished: Vec<Qwen36ChatFinished>,
}

struct ActiveChatRequest<'tokenizer> {
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint prompt rendering, continuous scheduling, and streaming output lifecycle.
pub struct Qwen36ChatService<'model, 'template> {
    template: &'template CheckpointChatTemplate,
    scheduler: Qwen36Scheduler<'model>,
    tool_grammar: QwenXmlGrammarFactory,
    requests: BTreeMap<Qwen36RequestId, ActiveChatRequest<'template>>,
}

impl<'model, 'template> Qwen36ChatService<'model, 'template> {
    #[cfg(test)]
    fn new(
        model: &'model Qwen36TextModel,
        template: &'template CheckpointChatTemplate,
        scheduler: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_cache_config(model, template, scheduler, SequenceCacheConfig::default())
    }

    #[cfg(test)]
    fn tick(&mut self) -> Result<Qwen36ChatTick> {
        self.tick_with_lifecycle(&mut |_| {})
    }

    /// Creates a serving bridge with explicit scheduler and cache limits.
    pub fn new_with_cache_config(
        model: &'model Qwen36TextModel,
        template: &'template CheckpointChatTemplate,
        scheduler: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        let tool_grammar =
            QwenXmlGrammarFactory::new(template.tokenizer(), model.manifest().vocab)?;
        Ok(Self {
            template,
            scheduler: Qwen36Scheduler::new_with_cache_config(model, scheduler, cache_config)?,
            tool_grammar,
            requests: BTreeMap::new(),
        })
    }

    /// Renders, tokenizes, and admits a structured request to the CPU waiting queue.
    pub fn add_request(&mut self, mut request: ChatRequest) -> Result<Qwen36ChatAdmission> {
        validate_stop_sequences(&request.stop_sequences)?;
        let prompt = self.template.render_and_tokenize(
            &request.messages,
            &request.tools,
            request.template,
        )?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let output = ChatOutputCodec::new(
            self.template.tokenizer(),
            &request.tools,
            starts_in_reasoning,
        )?;
        let filter = ResponseFilter::new(request.stop_sequences);
        let prompt_tokens = prompt.token_ids.len();
        request.generation.max_new_tokens = completion_tokens_within_context(
            prompt_tokens,
            request.generation.max_new_tokens,
            self.scheduler.config().max_context_tokens,
        )?;
        let max_output_tokens = request.generation.max_new_tokens;
        let grammar = self.tool_grammar.build(&request.tools)?;
        let id = self.scheduler.add_request_with_grammar(
            prompt.token_ids,
            request.generation,
            grammar,
        )?;
        let previous = self.requests.insert(
            id,
            ActiveChatRequest {
                output,
                filter,
                usage: ChatUsage {
                    prompt_tokens,
                    cached_prompt_tokens: 0,
                    completion_tokens: 0,
                    reasoning_tokens: 0,
                },
            },
        );
        debug_assert!(previous.is_none());
        Ok(Qwen36ChatAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one scheduler iteration and reports admission and prefill events
    /// when they occur.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen36RequestId, Qwen36AdmissionProgress>,
        ),
    ) -> Result<Qwen36ChatTick> {
        let scheduled = self.scheduler.tick_with_lifecycle(on_lifecycle)?;
        for admission in &scheduled.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted chat request is retained")
                .usage
                .cached_prompt_tokens = admission.cached_prompt_tokens;
        }
        let mut tick = Qwen36ChatTick {
            scheduled: scheduled.scheduled,
            prefilled: scheduled.prefilled,
            speculative: scheduled.speculative,
            ..Qwen36ChatTick::default()
        };
        let mut terminal = BTreeMap::new();

        for token in scheduled.generated {
            tick.generated.push(token.request_id);
            let request =
                self.requests
                    .get_mut(&token.request_id)
                    .ok_or_else(|| Error::Format {
                        label: "Qwen3.6 chat service",
                        detail: format!(
                            "scheduler emitted token for unknown request {}",
                            token.request_id.get()
                        ),
                    })?;
            request.usage.completion_tokens += 1;
            if request.output.is_reasoning() {
                request.usage.reasoning_tokens += 1;
            }
            let events = request.output.push_token(token.id)?;
            if let Some(reason) = request
                .filter
                .apply(token.request_id, events, &mut tick.output)
            {
                terminal.insert(token.request_id, reason);
            } else if let Some(reason) = token.finish_reason {
                terminal.insert(token.request_id, map_scheduler_finish(reason));
            }
        }

        for &id in self.requests.keys() {
            if self.scheduler.request_state(id) == Some(RequestState::Finished) {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }

        for (id, mut reason) in terminal {
            let request = self
                .requests
                .get_mut(&id)
                .expect("terminal chat request is retained");
            if matches!(
                reason,
                ChatFinishReason::Eos | ChatFinishReason::Length | ChatFinishReason::ToolCalls
            ) {
                let events = if matches!(reason, ChatFinishReason::Length) {
                    request.output.finish_truncated()?
                } else {
                    request.output.finish()?
                };
                if let Some(protocol_reason) = request.filter.apply(id, events, &mut tick.output) {
                    reason = protocol_reason;
                } else if request.filter.saw_tool_calls() {
                    reason = ChatFinishReason::ToolCalls;
                } else {
                    request.filter.flush(id, &mut tick.output);
                }
            }

            let released_sequence_device_bytes = self.release_scheduler_request(id)?;
            let request = self
                .requests
                .remove(&id)
                .expect("terminal chat request remains retained");
            tick.finished.push(Qwen36ChatFinished {
                request_id: id,
                finish_reason: reason,
                usage: request.usage,
                released_sequence_device_bytes,
            });
        }
        Ok(tick)
    }

    /// Cancels a waiting or active request, such as after a client disconnect.
    pub fn cancel_request(&mut self, id: Qwen36RequestId) -> Qwen36CancelOutcome {
        let outcome = self.scheduler.cancel_request(id);
        match &outcome {
            Qwen36CancelOutcome::Cancelled(_) => {
                self.requests.remove(&id);
            }
            Qwen36CancelOutcome::AlreadyFinished => {
                self.scheduler.remove_finished(id);
                self.requests.remove(&id);
            }
            Qwen36CancelOutcome::NotFound => {
                self.requests.remove(&id);
            }
        }
        outcome
    }

    /// Returns the number of requests retained by the serving bridge.
    #[cfg(test)]
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns the number of requests currently owning device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.scheduler.active_sequence_count()
    }

    /// Returns a request's scheduler lifecycle state.
    #[cfg(test)]
    pub fn request_state(&self, id: Qwen36RequestId) -> Option<RequestState> {
        self.scheduler.request_state(id)
    }

    fn release_scheduler_request(&mut self, id: Qwen36RequestId) -> Result<usize> {
        match self.scheduler.request_state(id) {
            Some(RequestState::Finished) => {
                let finished = self
                    .scheduler
                    .remove_finished(id)
                    .expect("finished scheduler request is removable");
                Ok(finished.released_sequence_device_bytes)
            }
            Some(_) => {
                let Qwen36CancelOutcome::Cancelled(cancelled) = self.scheduler.cancel_request(id)
                else {
                    return Err(Error::Format {
                        label: "Qwen3.6 chat service",
                        detail: format!("failed to stop request {}", id.get()),
                    });
                };
                Ok(cancelled.released_sequence_device_bytes)
            }
            None => Err(Error::Format {
                label: "Qwen3.6 chat service",
                detail: format!("terminal request {} is absent from scheduler", id.get()),
            }),
        }
    }
}

impl EngineService for Qwen36ChatService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> EngineResult<EngineAdmission> {
        let admission = Qwen36ChatService::add_request(self, request).map_err(EngineError::new)?;
        let id = admission.request_id.get();
        Ok(EngineAdmission {
            request_id: EngineRequestId::new(id),
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }
    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> EngineResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<Qwen36RequestId, Qwen36AdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                        request_id: EngineRequestId::new(progress.request_id.get()),
                        sequence_device_bytes: progress.sequence_device_bytes,
                        cached_prompt_tokens: progress.cached_prompt_tokens,
                        allocation_duration: Duration::ZERO,
                        checkpoint_copy_duration: Duration::ZERO,
                        admitted_after_tick_start: progress.admitted_after_tick_start,
                    }))
                }
                RequestLifecycleEvent::PrefillStarted(id) => on_lifecycle(
                    EngineLifecycleEvent::PrefillStarted(EngineRequestId::new(id.get())),
                ),
            };
        let tick = Qwen36ChatService::tick_with_lifecycle(self, &mut observer)
            .map_err(EngineError::new)?;
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: EngineRequestId::new(progress.request_id.get()),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(|id| EngineRequestId::new(id.get()))
                .collect(),
            verification: tick
                .speculative
                .into_iter()
                .map(|progress| EngineVerificationProgress {
                    request_id: EngineRequestId::new(progress.request_id.get()),
                    cycles: progress.cycles,
                    accepted_drafts: progress.accepted_drafts,
                })
                .collect(),
            draft_progress: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: EngineRequestId::new(delta.request_id.get()),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: EngineRequestId::new(finished.request_id.get()),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
        };
        Ok(converted)
    }
    fn cancel_request(&mut self, id: EngineRequestId) -> EngineCancelOutcome {
        match Qwen36ChatService::cancel_request(self, Qwen36RequestId::from_u64(id.get())) {
            Qwen36CancelOutcome::Cancelled(cancelled) => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: cancelled.released_sequence_device_bytes,
            },
            Qwen36CancelOutcome::AlreadyFinished => EngineCancelOutcome::AlreadyFinished,
            Qwen36CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }
    fn active_sequence_count(&self) -> usize {
        Qwen36ChatService::active_sequence_count(self)
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
        request_id: Qwen36RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Qwen36ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(Qwen36ChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Qwen36ChatDelta {
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
                    output.push(Qwen36ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn saw_tool_calls(&self) -> bool {
        self.saw_tool_calls
    }

    fn flush(&mut self, request_id: Qwen36RequestId, output: &mut Vec<Qwen36ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Qwen36ChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}

fn validate_stop_sequences(stop_sequences: &[String]) -> Result<()> {
    if stop_sequences.iter().any(String::is_empty) {
        return Err(Error::Format {
            label: "chat stop sequences",
            detail: "stop sequences must not be empty".to_string(),
        });
    }
    Ok(())
}

fn completion_tokens_within_context(
    prompt_tokens: usize,
    requested_completion_tokens: usize,
    max_context_tokens: usize,
) -> Result<usize> {
    let Some(remaining_tokens) = max_context_tokens.checked_sub(prompt_tokens) else {
        return Err(Error::Shape {
            label: "Qwen3.6 chat prompt capacity",
            expected: format!("at most {max_context_tokens} tokens"),
            actual: format!("{prompt_tokens} tokens"),
        });
    };
    Ok(requested_completion_tokens.min(remaining_tokens))
}

fn map_scheduler_finish(reason: RequestFinishReason) -> ChatFinishReason {
    match reason {
        RequestFinishReason::Eos => ChatFinishReason::Eos,
        RequestFinishReason::Length => ChatFinishReason::Length,
        RequestFinishReason::ToolCalls => ChatFinishReason::ToolCalls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::chat::ChatToolCall;
    use eider_runtime::generation::GenerationConfig;
    use eider_runtime::sampling::SamplingConfig;
    use serde_json::json;
    use std::path::PathBuf;

    fn request_id() -> Qwen36RequestId {
        Qwen36RequestId::for_test(7)
    }

    #[test]
    fn empty_stop_sequences_are_rejected() {
        assert!(validate_stop_sequences(&[String::new()]).is_err());
        assert!(validate_stop_sequences(&["END".to_string()]).is_ok());
    }

    #[test]
    fn completion_limit_is_capped_to_remaining_context() {
        assert_eq!(
            completion_tokens_within_context(29_064, 4_096, 32_768).expect("prompt fits context"),
            3_704
        );
        assert_eq!(
            completion_tokens_within_context(32_768, 4_096, 32_768)
                .expect("full prompt fits context"),
            0
        );
        assert!(completion_tokens_within_context(32_769, 1, 32_768).is_err());
    }

    #[test]
    fn scheduler_finish_reasons_map_to_serving_reasons() {
        assert_eq!(
            map_scheduler_finish(RequestFinishReason::Eos),
            ChatFinishReason::Eos
        );
        assert_eq!(
            map_scheduler_finish(RequestFinishReason::Length),
            ChatFinishReason::Length
        );
        assert_eq!(
            map_scheduler_finish(RequestFinishReason::ToolCalls),
            ChatFinishReason::ToolCalls
        );
    }

    #[test]
    #[ignore = "loads the full local Qwen3.6 checkpoint"]
    fn real_model_renders_prefills_streams_finishes_and_cancels() {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36TextModel::open(&model_dir).expect("load Qwen3.6 model");
        let template = CheckpointChatTemplate::from_model_dir(&model_dir).expect("chat template");
        let mut service = Qwen36ChatService::new(
            &model,
            &template,
            SchedulerConfig {
                decode_capacity: 2,
                prefill_sequence_capacity: 2,
                prefill_token_capacity: 8,
                max_active_sequences: 2,
                max_context_tokens: 128,
                speculative_drafts: 0,
            },
        )
        .expect("chat service");
        let defaults = GenerationConfig::from_model_dir(&model_dir).expect("generation defaults");
        let generation = RequestConfig {
            sampling: SamplingConfig {
                temperature: 0.0,
                ..defaults.sampling
            },
            max_new_tokens: 4,
            eos_token_ids: defaults.eos_token_ids,
        };
        let id = service
            .add_request(ChatRequest::new(
                vec![ChatMessage::user("Reply with one short greeting.")],
                generation.clone(),
            ))
            .expect("chat request")
            .request_id;
        assert_eq!(service.request_state(id), Some(RequestState::Waiting));

        let mut finished = None;
        let mut saw_prefill = false;
        let mut generated_tokens = 0;
        let mut output = Vec::new();
        for _ in 0..64 {
            let tick = service.tick().expect("chat tick");
            saw_prefill |= !tick.prefilled.is_empty();
            generated_tokens += tick.generated.len();
            output.extend(tick.output);
            if let Some(done) = tick.finished.into_iter().next() {
                finished = Some(done);
                break;
            }
        }
        let finished = finished.expect("request finished");
        assert!(saw_prefill);
        assert_eq!(finished.request_id, id);
        assert!(finished.usage.prompt_tokens > 1);
        assert_eq!(finished.usage.completion_tokens, 4);
        assert_eq!(generated_tokens, 4);
        assert_eq!(finished.finish_reason, ChatFinishReason::Length);
        assert!(!output.is_empty());
        assert_eq!(service.request_count(), 0);
        assert_eq!(service.request_state(id), None);

        let cancelled = service
            .add_request(ChatRequest::new(
                vec![ChatMessage::user("This request will be cancelled.")],
                generation,
            ))
            .expect("cancellation request")
            .request_id;
        assert!(matches!(
            service.cancel_request(cancelled),
            Qwen36CancelOutcome::Cancelled(_)
        ));
        assert_eq!(service.request_count(), 0);
    }

    #[test]
    fn stop_filter_holds_split_prefix_and_maps_stop_reason() {
        let id = request_id();
        let mut filter = ResponseFilter::new(vec!["END".to_string()]);
        let mut output = Vec::new();
        assert_eq!(
            filter.apply(
                id,
                vec![ChatOutputEvent::Text("before E".to_string())],
                &mut output,
            ),
            None
        );
        assert_eq!(
            output,
            [Qwen36ChatDelta {
                request_id: id,
                event: ChatOutputEvent::Text("before ".to_string())
            }]
        );
        assert_eq!(
            filter.apply(
                id,
                vec![ChatOutputEvent::Text("ND ignored".to_string())],
                &mut output,
            ),
            Some(ChatFinishReason::Stop("END".to_string()))
        );
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn first_complete_tool_call_flushes_text_and_terminates_generation() {
        let id = request_id();
        let mut filter = ResponseFilter::new(vec!["END".to_string()]);
        let mut output = Vec::new();
        let call = ChatOutputEvent::ToolCall(ChatToolCall {
            id: "call_1".to_string(),
            function: eider_runtime::chat::ChatFunctionCall {
                name: "read".to_string(),
                arguments: BTreeMap::from([("path".to_string(), json!("README.md"))]),
            },
        });
        assert_eq!(
            filter.apply(
                id,
                vec![
                    ChatOutputEvent::Text("safe E".to_string()),
                    call.clone(),
                    ChatOutputEvent::Text("ignored".to_string()),
                ],
                &mut output,
            ),
            Some(ChatFinishReason::ToolCalls)
        );
        assert!(filter.saw_tool_calls());
        assert_eq!(
            output,
            [
                Qwen36ChatDelta {
                    request_id: id,
                    event: ChatOutputEvent::Text("safe ".to_string())
                },
                Qwen36ChatDelta {
                    request_id: id,
                    event: ChatOutputEvent::Text("E".to_string())
                },
                Qwen36ChatDelta {
                    request_id: id,
                    event: call
                }
            ]
        );
    }
}
