//! Structured chat serving over the multi-session Step-3.7 scheduler.

use super::step37_scheduler::{
    Step37AdmissionProgress, Step37CancelOutcome, Step37PrefillProgress, Step37RequestId,
    Step37Scheduler,
};
use crate::step37::Step37TextModel;
use eider_cuda::{Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::scheduler::{
    RequestFinishReason, RequestLifecycleEvent, RequestState, SchedulerConfig,
};
use eider_runtime::stop::StopBuffer;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Step37ChatDelta {
    pub request_id: Step37RequestId,
    pub event: ChatOutputEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Step37ChatAdmission {
    pub request_id: Step37RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Step37ChatFinished {
    pub request_id: Step37RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step37ChatTick {
    pub admitted: Vec<Step37AdmissionProgress>,
    pub scheduled: Vec<Step37RequestId>,
    pub prefilled: Vec<Step37PrefillProgress>,
    pub generated: Vec<Step37RequestId>,
    pub output: Vec<Step37ChatDelta>,
    pub finished: Vec<Step37ChatFinished>,
    pub active_sequences: usize,
}

struct ActiveChatRequest<'tokenizer> {
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

pub struct Step37ChatService<'template> {
    template: &'template CheckpointChatTemplate,
    scheduler: Step37Scheduler,
    requests: BTreeMap<Step37RequestId, ActiveChatRequest<'template>>,
}

impl<'template> Step37ChatService<'template> {
    pub fn new(
        model: Step37TextModel,
        template: &'template CheckpointChatTemplate,
        scheduler: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_cache_config(model, template, scheduler, SequenceCacheConfig::default())
    }

    pub fn new_with_cache_config(
        model: Step37TextModel,
        template: &'template CheckpointChatTemplate,
        scheduler: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        Ok(Self {
            template,
            scheduler: Step37Scheduler::new_with_cache_config(model, scheduler, cache_config)?,
            requests: BTreeMap::new(),
        })
    }

    pub fn add_request(&mut self, request: ChatRequest) -> Result<Step37ChatAdmission> {
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
        let max_output_tokens = request.generation.max_new_tokens;
        let id = self
            .scheduler
            .add_request(prompt.token_ids, request.generation)?;
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
        Ok(Step37ChatAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    pub fn tick(&mut self) -> Result<Step37ChatTick> {
        self.tick_with_lifecycle(&mut |_| {})
    }

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Step37RequestId, Step37AdmissionProgress>,
        ),
    ) -> Result<Step37ChatTick> {
        let scheduled = self.scheduler.tick_with_lifecycle(on_lifecycle)?;
        for admission in &scheduled.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted chat request is retained")
                .usage
                .cached_prompt_tokens = admission.cached_prompt_tokens;
        }
        let mut tick = Step37ChatTick {
            admitted: scheduled.admitted,
            scheduled: scheduled.scheduled,
            prefilled: scheduled.prefilled,
            ..Step37ChatTick::default()
        };
        let mut terminal = BTreeMap::new();

        for token in scheduled.generated {
            tick.generated.push(token.request_id);
            let request =
                self.requests
                    .get_mut(&token.request_id)
                    .ok_or_else(|| Error::Format {
                        label: "Step-3.7 chat service",
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
                .expect("terminal request retained");
            if matches!(reason, ChatFinishReason::Eos | ChatFinishReason::Length) {
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
                .expect("terminal request retained");
            tick.finished.push(Step37ChatFinished {
                request_id: id,
                finish_reason: reason,
                usage: request.usage,
                released_sequence_device_bytes,
            });
        }
        tick.active_sequences = self.scheduler.active_sequence_count();
        Ok(tick)
    }

    pub fn cancel_request(&mut self, id: Step37RequestId) -> Step37CancelOutcome {
        let outcome = self.scheduler.cancel_request(id);
        match &outcome {
            Step37CancelOutcome::Cancelled(_) => {
                self.requests.remove(&id);
            }
            Step37CancelOutcome::AlreadyFinished => {
                self.scheduler.remove_finished(id);
                self.requests.remove(&id);
            }
            Step37CancelOutcome::NotFound => {
                self.requests.remove(&id);
            }
        }
        outcome
    }

    pub fn active_sequence_count(&self) -> usize {
        self.scheduler.active_sequence_count()
    }

    fn release_scheduler_request(&mut self, id: Step37RequestId) -> Result<usize> {
        match self.scheduler.request_state(id) {
            Some(RequestState::Finished) => Ok(self
                .scheduler
                .remove_finished(id)
                .expect("finished scheduler request is removable")
                .released_sequence_device_bytes),
            Some(_) => {
                let Step37CancelOutcome::Cancelled(cancelled) = self.scheduler.cancel_request(id)
                else {
                    return Err(Error::Format {
                        label: "Step-3.7 chat service",
                        detail: format!("failed to stop request {}", id.get()),
                    });
                };
                Ok(cancelled.released_sequence_device_bytes)
            }
            None => Err(Error::Format {
                label: "Step-3.7 chat service",
                detail: format!("terminal request {} is absent from scheduler", id.get()),
            }),
        }
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
        request_id: Step37RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Step37ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(Step37ChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Step37ChatDelta {
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
                    output.push(Step37ChatDelta { request_id, event });
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

    fn flush(&mut self, request_id: Step37RequestId, output: &mut Vec<Step37ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Step37ChatDelta {
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
    use eider_runtime::chat::{ChatFunctionCall, ChatToolCall};
    use serde_json::json;

    #[test]
    fn first_complete_tool_call_terminates_generation() {
        let request_id = Step37RequestId::for_test(7);
        let call = ChatOutputEvent::ToolCall(ChatToolCall {
            id: "call_1".to_string(),
            function: ChatFunctionCall {
                name: "read".to_string(),
                arguments: BTreeMap::from([("path".to_string(), json!("README.md"))]),
            },
        });
        let mut filter = ResponseFilter::new(Vec::new());
        let mut output = Vec::new();

        assert_eq!(
            filter.apply(
                request_id,
                vec![call.clone(), ChatOutputEvent::Text("ignored".to_string())],
                &mut output,
            ),
            Some(ChatFinishReason::ToolCalls)
        );
        assert!(filter.saw_tool_calls());
        assert_eq!(
            output,
            [Step37ChatDelta {
                request_id,
                event: call,
            }]
        );
    }
}
