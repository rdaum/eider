//! Multi-session chat serving for DeepSeek V4.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::prefix_cache::{
    PrefixCache, PrefixCacheConfig, PrefixCacheKey, cacheable_prompt_prefix_tokens,
};
use super::sampling::{SampledToken, Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::deepseek4::{
    Deepseek4BatchRow, Deepseek4BatchWorkspace, Deepseek4SequenceCheckpoint,
    Deepseek4SequenceState, Deepseek4TextModel,
};
use nvfp4::{Error, Result};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

const MAX_CONTINUATION_PREFILL_TOKENS: usize = 1_024;

fn prefill_chunk_capacity(prompt_position: usize, fair_share: usize) -> usize {
    if prompt_position == 0 {
        fair_share
    } else {
        fair_share.min(MAX_CONTINUATION_PREFILL_TOKENS)
    }
}

fn checkpoint_bounded_chunk(
    chunk: usize,
    prompt_position: usize,
    prefix_cache_target: usize,
    prefix_cache_key_present: bool,
    prefix_cache_checkpointed: bool,
) -> usize {
    if prefix_cache_checkpointed
        || !prefix_cache_key_present
        || prompt_position >= prefix_cache_target
    {
        chunk
    } else {
        chunk.min(prefix_cache_target - prompt_position)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Deepseek4RequestId(u64);

impl Deepseek4RequestId {
    pub fn get(self) -> u64 {
        self.0
    }
}

pub struct Deepseek4Admission {
    pub request_id: Deepseek4RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Deepseek4AdmissionProgress {
    pub request_id: Deepseek4RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    pub allocation_duration: Duration,
    pub checkpoint_copy_duration: Duration,
    pub admitted_after_tick_start: Duration,
}

pub struct Deepseek4PrefillProgress {
    pub request_id: Deepseek4RequestId,
    pub prompt_position: usize,
}

pub struct Deepseek4ChatDelta {
    pub request_id: Deepseek4RequestId,
    pub event: ChatOutputEvent,
}

pub struct Deepseek4Finished {
    pub request_id: Deepseek4RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

#[derive(Default)]
pub struct Deepseek4Tick {
    pub admitted: Vec<Deepseek4AdmissionProgress>,
    pub prefilled: Vec<Deepseek4PrefillProgress>,
    pub generated: Vec<Deepseek4RequestId>,
    pub output: Vec<Deepseek4ChatDelta>,
    pub finished: Vec<Deepseek4Finished>,
    pub active_sequences: usize,
}

pub enum Deepseek4CancelOutcome {
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
    pending_sample: Option<SampledToken>,
    state: Option<Deepseek4SequenceState>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

pub struct Deepseek4ChatService<'template> {
    model: Deepseek4TextModel,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<Deepseek4RequestId>,
    requests: BTreeMap<Deepseek4RequestId, ActiveRequest<'template>>,
    active_sequences: usize,
    prefix_cache: Option<PrefixCache<Deepseek4SequenceCheckpoint>>,
    workspace: Deepseek4BatchWorkspace,
}

impl<'template> Deepseek4ChatService<'template> {
    pub fn new(
        model: Deepseek4TextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_prefix_cache(model, template, config, PrefixCacheConfig::default())
    }

    pub fn new_with_prefix_cache(
        model: Deepseek4TextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        prefix_cache: PrefixCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let workspace = model.new_batch_workspace(
            config.decode_capacity.max(config.prefill_sequence_capacity),
            config.prefill_token_capacity.max(config.decode_capacity),
            config.max_context_tokens,
        )?;
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
            prefix_cache: (prefix_cache.max_device_bytes != 0)
                .then(|| PrefixCache::new(prefix_cache.max_device_bytes)),
            workspace,
        })
    }

    pub fn add_request(&mut self, request: ChatRequest) -> Result<Deepseek4Admission> {
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
                label: "DeepSeek V4 chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "DeepSeek V4 request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Deepseek4RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "DeepSeek V4 request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let prefix_cache_target = cacheable_prompt_prefix_tokens(prompt.token_ids.len());
        let prefix_cache_key = if prefix_cache_target == 0 {
            None
        } else {
            self.prefix_cache
                .as_mut()
                .map(|cache| cache.prompt_key(&prompt.token_ids, prefix_cache_target))
                .transpose()?
        };
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
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
                pending_sample: None,
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
        Ok(Deepseek4Admission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    pub fn tick(&mut self) -> Result<Deepseek4Tick> {
        self.tick_with_lifecycle(&mut |_| {})
    }

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Deepseek4RequestId, Deepseek4AdmissionProgress>,
        ),
    ) -> Result<Deepseek4Tick> {
        let tick_started = Instant::now();
        let mut tick = Deepseek4Tick::default();
        self.admit(&mut tick, tick_started, on_lifecycle)?;
        for admission in &tick.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted DeepSeek V4 request is retained")
                .usage
                .cached_prompt_tokens = admission.cached_prompt_tokens;
        }

        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.state.is_some()
                    && request.prompt_position >= request.prompt.len()
                    && request.generated_tokens < request.generation.max_new_tokens
            })
            .map(|(&id, _)| id)
            .take(self.config.decode_capacity)
            .collect::<Vec<_>>();
        for (id, reason) in self.generate(&decode_ids, &mut tick)? {
            terminal.insert(id, reason);
        }

        let prefill_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.state.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick, on_lifecycle)?;

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

    pub fn cancel_request(&mut self, id: Deepseek4RequestId) -> Deepseek4CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Deepseek4CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.state.map_or(0, |state| state.device_bytes());
        if released != 0 {
            self.active_sequences -= 1;
        }
        Deepseek4CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut Deepseek4Tick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Deepseek4RequestId, Deepseek4AdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let mut allocation_duration = Duration::ZERO;
            let mut checkpoint_copy_duration = Duration::ZERO;
            let restored = match (&mut self.prefix_cache, request.prefix_cache_key.as_ref()) {
                (Some(cache), Some(key)) => {
                    cache.restore(key, Deepseek4SequenceCheckpoint::position, |checkpoint| {
                        let started = Instant::now();
                        let state = self.model.restore_sequence_checkpoint(
                            checkpoint,
                            capacity.max(1),
                            &self.workspace,
                        )?;
                        checkpoint_copy_duration = started.elapsed();
                        Ok(state)
                    })?
                }
                _ => None,
            };
            let cached_prompt_tokens = restored
                .as_ref()
                .map_or(0, Deepseek4SequenceState::position);
            let state = if let Some(restored) = restored {
                restored
            } else {
                let started = Instant::now();
                let state = self.model.new_sequence_state(capacity.max(1))?;
                allocation_duration = started.elapsed();
                state
            };
            let bytes = state.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_cache_checkpointed =
                cached_prompt_tokens == request.prefix_cache_target && cached_prompt_tokens != 0;
            request.state = Some(state);
            self.active_sequences += 1;
            let progress = Deepseek4AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                allocation_duration,
                checkpoint_copy_duration,
                admitted_after_tick_start: tick_started.elapsed(),
            };
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        ids: &[Deepseek4RequestId],
        tick: &mut Deepseek4Tick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Deepseek4RequestId, Deepseek4AdmissionProgress>,
        ),
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut budget = self.config.prefill_token_capacity;
        let mut selected = Vec::with_capacity(ids.len());
        for (index, &id) in ids.iter().enumerate() {
            if budget == 0 {
                break;
            }
            let request = self.requests.get(&id).expect("prefill request exists");
            let available = request.prompt.len() - request.prompt_position;
            let batchable = available.saturating_sub(1);
            let remaining_sequences = ids.len() - index;
            let fair_share = budget.div_ceil(remaining_sequences);
            let mut chunk =
                batchable.min(prefill_chunk_capacity(request.prompt_position, fair_share));
            chunk = checkpoint_bounded_chunk(
                chunk,
                request.prompt_position,
                request.prefix_cache_target,
                request.prefix_cache_key.is_some(),
                request.prefix_cache_checkpointed,
            );
            if chunk == 0 {
                continue;
            }
            budget -= chunk;
            selected.push((id, chunk));
        }
        if !selected.is_empty() {
            let mut requests = selected
                .iter()
                .map(|(id, _)| self.requests.remove(id).expect("prefill request exists"))
                .collect::<Vec<_>>();
            let result = {
                let mut rows = requests
                    .iter_mut()
                    .zip(selected.iter().map(|(_, chunk)| *chunk))
                    .map(|(request, chunk)| {
                        let start = request.prompt_position;
                        Deepseek4BatchRow {
                            token_ids: &request.prompt[start..start + chunk],
                            state: request.state.as_mut().expect("prefill request is admitted"),
                        }
                    })
                    .collect::<Vec<_>>();
                for &(id, _) in &selected {
                    on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
                }
                self.model.prefill_batch(&mut self.workspace, &mut rows)
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
                        &self.model,
                        &self.workspace,
                        &mut self.prefix_cache,
                        &mut request,
                    );
                }
                tick.prefilled.push(Deepseek4PrefillProgress {
                    request_id: id,
                    prompt_position: request.prompt_position,
                });
                self.requests.insert(id, request);
            }
            return Ok(());
        }

        let tail_ids = ids
            .iter()
            .copied()
            .filter(|id| {
                let request = self.requests.get(id).expect("prefill request exists");
                request.prompt_position + 1 == request.prompt.len()
            })
            .take(self.config.prefill_token_capacity)
            .collect::<Vec<_>>();
        if tail_ids.is_empty() {
            return Ok(());
        }
        let mut requests = tail_ids
            .iter()
            .map(|id| self.requests.remove(id).expect("prefill request exists"))
            .collect::<Vec<_>>();
        let result = {
            let mut rows = requests
                .iter_mut()
                .map(|request| Deepseek4BatchRow {
                    token_ids: &request.prompt[request.prompt_position..],
                    state: request.state.as_mut().expect("prefill request is admitted"),
                })
                .collect::<Vec<_>>();
            for &id in &tail_ids {
                on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            }
            self.model
                .forward_batch(&mut self.workspace, &mut rows)
                .and_then(|logits| logits.copy_to_host())
        };
        let logits = match result {
            Ok(logits) => logits,
            Err(error) => {
                for (request, id) in requests.into_iter().zip(&tail_ids) {
                    self.requests.insert(*id, request);
                }
                return Err(error);
            }
        };
        let vocab = self.model.weights.config.vocab_size;
        for ((mut request, id), row_logits) in requests
            .into_iter()
            .zip(tail_ids)
            .zip(logits.chunks_exact(vocab))
        {
            request.prompt_position += 1;
            request.pending_sample = Some(request.sampler.sample(row_logits, &request.history)?);
            if checkpoint_ready(
                request.prompt_position,
                request.prefix_cache_target,
                request.prefix_cache_checkpointed,
            ) {
                Self::retain_request_checkpoint(
                    &self.model,
                    &self.workspace,
                    &mut self.prefix_cache,
                    &mut request,
                );
            }
            tick.prefilled.push(Deepseek4PrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn retain_request_checkpoint(
        model: &Deepseek4TextModel,
        workspace: &Deepseek4BatchWorkspace,
        prefix_cache: &mut Option<PrefixCache<Deepseek4SequenceCheckpoint>>,
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
        if state.position() != request.prefix_cache_target {
            return;
        }
        let Ok(estimated_bytes) =
            model.checkpoint_sequence_device_bytes(request.prefix_cache_target)
        else {
            request.prefix_cache_checkpointed = true;
            return;
        };
        if !cache.contains(key) && cache.prepare_insert(estimated_bytes) {
            let started = Instant::now();
            match model.checkpoint_sequence(state, workspace) {
                Ok(checkpoint) => {
                    cache.record_checkpoint(started);
                    let bytes = checkpoint.device_bytes();
                    if let Err(error) = cache.insert(key.clone(), checkpoint, bytes) {
                        warn!(%error, "failed to retain DeepSeek V4 prompt prefix checkpoint");
                    }
                }
                Err(error) => warn!(%error, "failed to checkpoint DeepSeek V4 prompt prefix"),
            }
        }
        request.prefix_cache_checkpointed = true;
    }

    fn generate(
        &mut self,
        ids: &[Deepseek4RequestId],
        tick: &mut Deepseek4Tick,
    ) -> Result<Vec<(Deepseek4RequestId, ChatFinishReason)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut requests = ids
            .iter()
            .map(|id| self.requests.remove(id).expect("decode request exists"))
            .collect::<Vec<_>>();
        let model_count = requests
            .iter()
            .filter(|request| request.pending_sample.is_none())
            .count();
        let logits = if model_count == 0 {
            Vec::new()
        } else {
            let result = {
                let mut rows = requests
                    .iter_mut()
                    .filter_map(|request| {
                        if request.pending_sample.is_some() {
                            return None;
                        }
                        let token = request
                            .last_token
                            .as_ref()
                            .expect("generated token exists after prompt logits");
                        Some(Deepseek4BatchRow {
                            token_ids: std::slice::from_ref(token),
                            state: request.state.as_mut().expect("decode request is admitted"),
                        })
                    })
                    .collect::<Vec<_>>();
                self.model
                    .forward_batch(&mut self.workspace, &mut rows)
                    .and_then(|logits| logits.copy_to_host())
            };
            match result {
                Ok(logits) => logits,
                Err(error) => {
                    for (request, id) in requests.into_iter().zip(ids) {
                        self.requests.insert(*id, request);
                    }
                    return Err(error);
                }
            }
        };

        let vocab = self.model.weights.config.vocab_size;
        let mut logits_rows = logits.chunks_exact(vocab);
        let mut terminal = Vec::new();
        for (mut request, &id) in requests.into_iter().zip(ids) {
            let sampled = if let Some(sampled) = request.pending_sample.take() {
                sampled
            } else {
                request.sampler.sample(
                    logits_rows
                        .next()
                        .expect("one logits row exists per forwarded request"),
                    &request.history,
                )?
            };
            if let Some(reason) = apply_sample(&mut request, id, sampled, tick)? {
                terminal.push((id, reason));
            }
            self.requests.insert(id, request);
        }
        debug_assert!(logits_rows.next().is_none());
        Ok(terminal)
    }

    fn finish_request(
        &mut self,
        id: Deepseek4RequestId,
        mut reason: ChatFinishReason,
        tick: &mut Deepseek4Tick,
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
        tick.finished.push(Deepseek4Finished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

fn apply_sample(
    request: &mut ActiveRequest<'_>,
    id: Deepseek4RequestId,
    sampled: SampledToken,
    tick: &mut Deepseek4Tick,
) -> Result<Option<ChatFinishReason>> {
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
        request_id: Deepseek4RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Deepseek4ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(Deepseek4ChatDelta { request_id, event });
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Deepseek4ChatDelta {
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
                    output.push(Deepseek4ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: Deepseek4RequestId, output: &mut Vec<Deepseek4ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Deepseek4ChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CONTINUATION_PREFILL_TOKENS, checkpoint_bounded_chunk, checkpoint_ready,
        prefill_chunk_capacity,
    };

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

    #[test]
    fn initial_prefill_uses_the_full_scheduler_share() {
        assert_eq!(prefill_chunk_capacity(0, 4_096), 4_096);
        assert_eq!(prefill_chunk_capacity(0, 512), 512);
    }

    #[test]
    fn continuation_prefill_is_bounded_for_actor_responsiveness() {
        assert_eq!(
            prefill_chunk_capacity(4_096, 4_096),
            MAX_CONTINUATION_PREFILL_TOKENS
        );
        assert_eq!(prefill_chunk_capacity(4_096, 512), 512);
    }

    #[test]
    fn prefill_stops_exactly_at_a_pending_prefix_checkpoint() {
        assert_eq!(checkpoint_bounded_chunk(1_024, 128, 512, true, false), 384);
        assert_eq!(
            checkpoint_bounded_chunk(1_024, 512, 512, true, false),
            1_024
        );
        assert_eq!(
            checkpoint_bounded_chunk(1_024, 128, 512, false, false),
            1_024
        );
        assert_eq!(checkpoint_bounded_chunk(1_024, 128, 512, true, true), 1_024);
    }
}
