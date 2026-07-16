//! Structured chat serving over the multi-session Step-3.7 scheduler.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::scheduler::{RequestFinishReason, RequestState, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::step35_scheduler::{
    Step35AdmissionProgress, Step35CancelOutcome, Step35PrefillProgress, Step35RequestId,
    Step35Scheduler,
};
use super::stop::StopBuffer;
use crate::step35::Step35TextModel;
use nvfp4::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Step35ChatDelta {
    pub request_id: Step35RequestId,
    pub event: ChatOutputEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Step35ChatAdmission {
    pub request_id: Step35RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Step35ChatFinished {
    pub request_id: Step35RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step35ChatTick {
    pub admitted: Vec<Step35AdmissionProgress>,
    pub scheduled: Vec<Step35RequestId>,
    pub prefilled: Vec<Step35PrefillProgress>,
    pub generated: Vec<Step35RequestId>,
    pub output: Vec<Step35ChatDelta>,
    pub finished: Vec<Step35ChatFinished>,
    pub active_sequences: usize,
}

struct ActiveChatRequest<'tokenizer> {
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

pub struct Step35ChatService<'template> {
    template: &'template CheckpointChatTemplate,
    scheduler: Step35Scheduler,
    requests: BTreeMap<Step35RequestId, ActiveChatRequest<'template>>,
}

impl<'template> Step35ChatService<'template> {
    pub fn new(
        model: Step35TextModel,
        template: &'template CheckpointChatTemplate,
        scheduler: SchedulerConfig,
    ) -> Result<Self> {
        Ok(Self {
            template,
            scheduler: Step35Scheduler::new(model, scheduler)?,
            requests: BTreeMap::new(),
        })
    }

    pub fn add_request(&mut self, request: ChatRequest) -> Result<Step35ChatAdmission> {
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
                    completion_tokens: 0,
                },
            },
        );
        debug_assert!(previous.is_none());
        Ok(Step35ChatAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    pub fn tick(&mut self) -> Result<Step35ChatTick> {
        let scheduled = self.scheduler.tick()?;
        let mut tick = Step35ChatTick {
            admitted: scheduled.admitted,
            scheduled: scheduled.scheduled,
            prefilled: scheduled.prefilled,
            ..Step35ChatTick::default()
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
                let events = request.output.finish()?;
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
            tick.finished.push(Step35ChatFinished {
                request_id: id,
                finish_reason: reason,
                usage: request.usage,
                released_sequence_device_bytes,
            });
        }
        tick.active_sequences = self.scheduler.active_sequence_count();
        Ok(tick)
    }

    pub fn cancel_request(&mut self, id: Step35RequestId) -> Step35CancelOutcome {
        let outcome = self.scheduler.cancel_request(id);
        match &outcome {
            Step35CancelOutcome::Cancelled(_) => {
                self.requests.remove(&id);
            }
            Step35CancelOutcome::AlreadyFinished => {
                self.scheduler.remove_finished(id);
                self.requests.remove(&id);
            }
            Step35CancelOutcome::NotFound => {
                self.requests.remove(&id);
            }
        }
        outcome
    }

    pub fn active_sequence_count(&self) -> usize {
        self.scheduler.active_sequence_count()
    }

    fn release_scheduler_request(&mut self, id: Step35RequestId) -> Result<usize> {
        match self.scheduler.request_state(id) {
            Some(RequestState::Finished) => Ok(self
                .scheduler
                .remove_finished(id)
                .expect("finished scheduler request is removable")
                .released_sequence_device_bytes),
            Some(_) => {
                let Step35CancelOutcome::Cancelled(cancelled) = self.scheduler.cancel_request(id)
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
        request_id: Step35RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Step35ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(Step35ChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Step35ChatDelta {
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
                    output.push(Step35ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                }
            }
        }
        None
    }

    fn saw_tool_calls(&self) -> bool {
        self.saw_tool_calls
    }

    fn flush(&mut self, request_id: Step35RequestId, output: &mut Vec<Step35ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Step35ChatDelta {
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
    }
}
