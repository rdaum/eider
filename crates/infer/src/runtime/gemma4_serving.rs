//! Multi-session chat serving for Gemma 4.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::prefix_cache::{
    PrefixCache, PrefixCacheConfig, PrefixCacheKey, cacheable_prompt_prefix_tokens,
};
use super::sampling::{Sampler, TokenHistory};
use super::scheduler::{RequestConfig, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::gemma4::{
    Gemma4DecodeState, Gemma4Model, Gemma4PrefillBatchWorkspace, Gemma4PrefillRow,
    Gemma4SequenceCheckpoint,
};
use nvfp4::{CudaStream, Error, Result};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

const TAIL_PREFILL_TOKEN_CAPACITY: usize = 512;

/// Stable request identity assigned by a Gemma 4 chat service.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Gemma4RequestId(u64);

impl Gemma4RequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after rendering and tokenization.
pub struct Gemma4Admission {
    pub request_id: Gemma4RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

/// Device-state allocation completed during a tick.
pub struct Gemma4AdmissionProgress {
    pub request_id: Gemma4RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time when admission completed.
    pub admitted_after_tick_start: Duration,
}

/// Prompt progress completed during a tick.
pub struct Gemma4PrefillProgress {
    pub request_id: Gemma4RequestId,
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct Gemma4ChatDelta {
    pub request_id: Gemma4RequestId,
    pub event: ChatOutputEvent,
}

/// Terminal request metadata.
pub struct Gemma4Finished {
    pub request_id: Gemma4RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

/// Observable work and output from one service iteration.
#[derive(Default)]
pub struct Gemma4Tick {
    pub admitted: Vec<Gemma4AdmissionProgress>,
    pub prefilled: Vec<Gemma4PrefillProgress>,
    pub generated: Vec<Gemma4RequestId>,
    pub output: Vec<Gemma4ChatDelta>,
    pub finished: Vec<Gemma4Finished>,
    pub active_sequences: usize,
}

/// Outcome of cancelling a waiting or active request.
pub enum Gemma4CancelOutcome {
    Cancelled {
        released_sequence_device_bytes: usize,
    },
    NotFound,
}

struct ActiveRequest<'tokenizer> {
    prompt: Vec<u32>,
    prompt_position: usize,
    prefix_cache_key: Option<PrefixCacheKey>,
    prefix_cache_target: usize,
    prefix_cache_checkpointed: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    state: Option<Gemma4DecodeState>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint rendering and decode-first, round-robin Gemma 4 execution.
pub struct Gemma4ChatService<'model, 'template> {
    model: &'model Gemma4Model,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    stream: CudaStream,
    prefill_workspace: Gemma4PrefillBatchWorkspace,
    tail_prefill_workspace: Option<Gemma4PrefillBatchWorkspace>,
    next_id: u64,
    waiting: VecDeque<Gemma4RequestId>,
    requests: BTreeMap<Gemma4RequestId, ActiveRequest<'template>>,
    active_sequences: usize,
    prefix_cache: Option<PrefixCache<Gemma4SequenceCheckpoint>>,
}

impl<'model, 'template> Gemma4ChatService<'model, 'template> {
    /// Creates a multi-session service with explicit scheduling limits.
    pub fn new(
        model: &'model Gemma4Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_prefix_cache(model, template, config, PrefixCacheConfig::default())
    }

    /// Creates a multi-session service with ART-backed prompt prefixes.
    pub fn new_with_prefix_cache(
        model: &'model Gemma4Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        prefix_cache: PrefixCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let prefill_workspace = model.new_prefill_batch_workspace(
            config.prefill_sequence_capacity,
            config.prefill_token_capacity,
            config.max_context_tokens,
        )?;
        let tail_prefill_workspace = (config.prefill_token_capacity > TAIL_PREFILL_TOKEN_CAPACITY)
            .then(|| {
                model.new_prefill_batch_workspace(
                    config.prefill_sequence_capacity,
                    TAIL_PREFILL_TOKEN_CAPACITY,
                    config.max_context_tokens,
                )
            })
            .transpose()?;
        Ok(Self {
            model,
            template,
            config,
            stream: CudaStream::new_non_blocking()?,
            prefill_workspace,
            tail_prefill_workspace,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
            prefix_cache: (prefix_cache.max_device_bytes != 0)
                .then(|| PrefixCache::new(prefix_cache.max_device_bytes)),
        })
    }

    /// Renders, tokenizes, and queues a request without allocating GPU state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<Gemma4Admission> {
        request.generation.validate()?;
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
                label: "Gemma 4 chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Gemma 4 request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Gemma 4 request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Gemma4RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Gemma 4 request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prefix_cache_target = cacheable_prompt_prefix_tokens(prompt.token_ids.len());
        let prefix_cache_key = if prefix_cache_target == 0 {
            None
        } else {
            self.prefix_cache
                .as_mut()
                .map(|cache| cache.prompt_key(&prompt.token_ids, prefix_cache_target))
                .transpose()?
        };
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids.clone(),
                prompt_position: 0,
                prefix_cache_key,
                prefix_cache_target,
                prefix_cache_checkpointed: false,
                generation: request.generation.clone(),
                generated_tokens: 0,
                last_token: None,
                state: None,
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
        Ok(Gemma4Admission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one decode-first scheduling iteration across active requests.
    pub fn tick(&mut self) -> Result<Gemma4Tick> {
        let tick_started = Instant::now();
        let mut tick = Gemma4Tick::default();
        self.admit(&mut tick, tick_started)?;
        for admission in &tick.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted Gemma 4 request is retained")
                .usage
                .cached_prompt_tokens = admission.cached_prompt_tokens;
        }
        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.state.is_some()
                    && request.prompt_position + 1 >= request.prompt.len()
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
                request.state.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position + 1 < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick)?;

        for (&id, request) in &self.requests {
            if request.state.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        tick.active_sequences = self.active_sequences;
        Ok(tick)
    }

    /// Cancels a waiting or active request.
    pub fn cancel_request(&mut self, id: Gemma4RequestId) -> Gemma4CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Gemma4CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.state.map_or(0, |state| state.device_bytes());
        if released != 0 {
            self.active_sequences -= 1;
        }
        Gemma4CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns requests currently owning device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(&mut self, tick: &mut Gemma4Tick, tick_started: Instant) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let restored = match (&mut self.prefix_cache, request.prefix_cache_key.as_ref()) {
                (Some(cache), Some(key)) => {
                    cache.restore(key, Gemma4SequenceCheckpoint::position, |checkpoint| {
                        self.model.restore_sequence_checkpoint(
                            checkpoint,
                            capacity.max(1),
                            &self.stream,
                        )
                    })?
                }
                _ => None,
            };
            let cached_prompt_tokens = restored.as_ref().map_or(0, Gemma4DecodeState::len);
            let state = restored.unwrap_or(self.model.new_decode_state(capacity.max(1))?);
            let bytes = state.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_cache_checkpointed =
                cached_prompt_tokens == request.prefix_cache_target && cached_prompt_tokens != 0;
            request.state = Some(state);
            self.active_sequences += 1;
            tick.admitted.push(Gemma4AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                admitted_after_tick_start: tick_started.elapsed(),
            });
        }
        Ok(())
    }

    fn prefill(&mut self, ids: &[Gemma4RequestId], tick: &mut Gemma4Tick) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut budget = self.config.prefill_token_capacity;
        let mut selected = Vec::with_capacity(ids.len());
        for (index, &id) in ids.iter().enumerate() {
            let request = self.requests.get(&id).expect("prefill request exists");
            let available = request
                .prompt
                .len()
                .saturating_sub(request.prompt_position + 1);
            let remaining_sequences = ids.len() - index;
            let chunk = available.min(budget.div_ceil(remaining_sequences));
            if chunk == 0 {
                continue;
            }
            budget -= chunk;
            selected.push((id, chunk));
        }
        if selected.is_empty() {
            return Ok(());
        }
        let mut requests = selected
            .iter()
            .map(|(id, _)| self.requests.remove(id).expect("prefill request exists"))
            .collect::<Vec<_>>();
        let result = {
            let selected_tokens = selected.iter().map(|(_, chunk)| *chunk).sum::<usize>();
            let workspace = self
                .tail_prefill_workspace
                .as_mut()
                .filter(|_| selected_tokens <= TAIL_PREFILL_TOKEN_CAPACITY)
                .unwrap_or(&mut self.prefill_workspace);
            let mut rows = requests
                .iter_mut()
                .zip(selected.iter().map(|(_, chunk)| *chunk))
                .map(|(request, chunk)| {
                    let start = request.prompt_position;
                    let end = start + chunk;
                    Gemma4PrefillRow {
                        token_ids: &request.prompt[start..end],
                        state: request.state.as_mut().expect("prefill request is admitted"),
                    }
                })
                .collect::<Vec<_>>();
            self.model
                .prefill_batch(workspace, &mut rows, &self.stream)
                .and_then(|()| self.stream.synchronize())
        };
        if let Err(error) = result {
            for (request, (id, _)) in requests.into_iter().zip(&selected) {
                self.requests.insert(*id, request);
            }
            return Err(error);
        }
        for (mut request, (id, chunk)) in requests.into_iter().zip(selected) {
            request.prompt_position += chunk;
            if checkpoint_ready(
                request.prompt_position,
                request.prefix_cache_target,
                request.prefix_cache_checkpointed,
            ) {
                Self::retain_request_checkpoint(
                    self.model,
                    &self.stream,
                    &mut self.prefix_cache,
                    &mut request,
                );
            }
            tick.prefilled.push(Gemma4PrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn retain_request_checkpoint(
        model: &Gemma4Model,
        stream: &CudaStream,
        prefix_cache: &mut Option<PrefixCache<Gemma4SequenceCheckpoint>>,
        request: &mut ActiveRequest<'template>,
    ) {
        if request.prefix_cache_checkpointed || request.prefix_cache_target == 0 {
            return;
        }
        let (Some(cache), Some(key), Some(state)) = (
            prefix_cache.as_mut(),
            request.prefix_cache_key.as_ref(),
            request.state.as_ref(),
        ) else {
            request.prefix_cache_checkpointed = true;
            return;
        };
        if state.len() < request.prefix_cache_target {
            return;
        }
        if !cache.contains(key) {
            let Ok(estimated_bytes) =
                model.checkpoint_sequence_device_bytes(state, request.prefix_cache_target)
            else {
                request.prefix_cache_checkpointed = true;
                return;
            };
            if cache.prepare_insert(estimated_bytes) {
                let started = Instant::now();
                match model.checkpoint_sequence(state, request.prefix_cache_target, stream) {
                    Ok(checkpoint) => {
                        cache.record_checkpoint(started);
                        let bytes = checkpoint.device_bytes();
                        if let Err(error) = cache.insert(key.clone(), checkpoint, bytes) {
                            warn!(%error, "failed to retain Gemma 4 prompt prefix checkpoint");
                        }
                    }
                    Err(error) => warn!(%error, "failed to checkpoint Gemma 4 prompt prefix"),
                }
            }
        }
        request.prefix_cache_checkpointed = true;
    }

    fn generate_one(
        &mut self,
        id: Gemma4RequestId,
        tick: &mut Gemma4Tick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let input = request
            .last_token
            .unwrap_or(request.prompt[request.prompt.len() - 1]);
        let state = request.state.as_mut().expect("decode request is admitted");
        self.model.forward_one(state, input, &self.stream)?;
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(state, &self.stream)?;
            super::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            let logits = self.model.logits_to_host(state, &self.stream)?;
            request.sampler.sample(&logits, &request.history)?
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
        id: Gemma4RequestId,
        mut reason: ChatFinishReason,
        tick: &mut Gemma4Tick,
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
        let state = request.state.take().expect("terminal request is admitted");
        let released = state.device_bytes();
        self.active_sequences -= 1;
        tick.finished.push(Gemma4Finished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

fn checkpoint_ready(
    prompt_position: usize,
    prefix_cache_target: usize,
    prefix_cache_checkpointed: bool,
) -> bool {
    !prefix_cache_checkpointed && prefix_cache_target != 0 && prompt_position >= prefix_cache_target
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
        request_id: Gemma4RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Gemma4ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(Gemma4ChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Gemma4ChatDelta {
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
                    output.push(Gemma4ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: Gemma4RequestId, output: &mut Vec<Gemma4ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Gemma4ChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::checkpoint_ready;

    #[test]
    fn checkpoint_is_ready_after_crossing_the_aligned_prefix() {
        assert!(checkpoint_ready(384, 256, false));
        assert!(checkpoint_ready(256, 256, false));
        assert!(!checkpoint_ready(128, 256, false));
    }

    #[test]
    fn disabled_or_completed_checkpoint_is_not_ready() {
        assert!(!checkpoint_ready(256, 0, false));
        assert!(!checkpoint_ready(256, 256, true));
    }
}
