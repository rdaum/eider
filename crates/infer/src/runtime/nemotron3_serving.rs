//! Structured multi-session chat serving for Nemotron 3.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::prefix_cache::{
    PrefixCache, PrefixCacheConfig, PrefixCacheKey, cacheable_prompt_prefix_tokens,
};
use super::sampling::{Sampler, TokenHistory};
use super::scheduler::{RequestConfig, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::nemotron3::{
    Nemotron3BlockWorkspace, Nemotron3DecodeState, Nemotron3Model, Nemotron3MtpWorkspace,
    Nemotron3SequenceCheckpoint, Nemotron3SpeculativeCycleWorkspace,
};
use nvfp4::{DeviceBuffer, Error, Result};
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;
use tracing::warn;

/// Stable request identity assigned by a Nemotron 3 chat service.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Nemotron3RequestId(u64);

impl Nemotron3RequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after prompt rendering and tokenization.
pub struct Nemotron3Admission {
    pub request_id: Nemotron3RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

/// Device-state allocation completed during a tick.
pub struct Nemotron3AdmissionProgress {
    pub request_id: Nemotron3RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
}

/// Prompt progress completed during a tick.
pub struct Nemotron3PrefillProgress {
    pub request_id: Nemotron3RequestId,
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct Nemotron3ChatDelta {
    pub request_id: Nemotron3RequestId,
    pub event: ChatOutputEvent,
}

/// Terminal request metadata.
pub struct Nemotron3Finished {
    pub request_id: Nemotron3RequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

/// Observable work and output from one service iteration.
#[derive(Default)]
pub struct Nemotron3Tick {
    pub admitted: Vec<Nemotron3AdmissionProgress>,
    pub prefilled: Vec<Nemotron3PrefillProgress>,
    pub generated: Vec<Nemotron3RequestId>,
    pub output: Vec<Nemotron3ChatDelta>,
    pub finished: Vec<Nemotron3Finished>,
    pub active_sequences: usize,
}

/// Outcome of cancelling a waiting or active request.
pub enum Nemotron3CancelOutcome {
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
    state: Option<Nemotron3DecodeState>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

struct Nemotron3PrefillWorkspace {
    target: Nemotron3BlockWorkspace,
    previous_hidden: DeviceBuffer<f32>,
    mtp: Option<Nemotron3MtpWorkspace>,
    mtp_hidden: Option<DeviceBuffer<f32>>,
}

impl Nemotron3PrefillWorkspace {
    fn new(
        model: &Nemotron3Model,
        sequence_count: usize,
        rows: usize,
        mtp_sequence_count: usize,
        mtp_rows: usize,
    ) -> Result<Self> {
        let hidden = model.manifest().hidden_size;
        let mtp = (mtp_rows != 0)
            .then(|| model.mtp_workspace(mtp_sequence_count, mtp_rows))
            .transpose()?;
        Ok(Self {
            target: model.block_workspace(sequence_count, rows)?,
            previous_hidden: DeviceBuffer::zeroed(sequence_count * hidden)?,
            mtp,
            mtp_hidden: (mtp_rows != 0)
                .then(|| DeviceBuffer::zeroed(mtp_rows * hidden))
                .transpose()?,
        })
    }
}

/// Checkpoint prompt rendering and round-robin multi-session Nemotron execution.
pub struct Nemotron3ChatService<'model, 'template> {
    model: &'model Nemotron3Model,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<Nemotron3RequestId>,
    requests: BTreeMap<Nemotron3RequestId, ActiveRequest<'template>>,
    active_sequences: usize,
    prefix_cache: Option<PrefixCache<Nemotron3SequenceCheckpoint>>,
    mtp_token_workspace: Option<Nemotron3MtpWorkspace>,
    prefill_workspaces: BTreeMap<(usize, usize, usize, usize), Nemotron3PrefillWorkspace>,
    speculative_workspaces: BTreeMap<usize, Nemotron3SpeculativeCycleWorkspace>,
}

impl<'model, 'template> Nemotron3ChatService<'model, 'template> {
    /// Creates a multi-session service with explicit scheduling limits.
    pub fn new(
        model: &'model Nemotron3Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_prefix_cache(model, template, config, PrefixCacheConfig::default())
    }

    /// Creates a multi-session service with ART-backed reusable prompt prefixes.
    pub fn new_with_prefix_cache(
        model: &'model Nemotron3Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        prefix_cache: PrefixCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
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
            mtp_token_workspace: model
                .has_mtp()
                .then(|| model.mtp_workspace(1, 1))
                .transpose()?,
            prefill_workspaces: BTreeMap::new(),
            speculative_workspaces: BTreeMap::new(),
        })
    }

    /// Renders, tokenizes, and queues one request without allocating GPU state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<Nemotron3Admission> {
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
                label: "Nemotron 3 chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Nemotron 3 request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Nemotron 3 request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Nemotron3RequestId(self.next_id);
        self.next_id += 1;
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
        let active = ActiveRequest {
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
                prompt_tokens: prompt.token_ids.len(),
                ..ChatUsage::default()
            },
        };
        self.waiting.push_back(id);
        self.requests.insert(id, active);
        Ok(Nemotron3Admission {
            request_id: id,
            prompt_tokens: prompt.token_ids.len(),
            max_output_tokens: request.generation.max_new_tokens,
        })
    }

    /// Runs one decode-first round-robin scheduling iteration.
    pub fn tick(&mut self) -> Result<Nemotron3Tick> {
        let mut tick = Nemotron3Tick::default();
        self.admit(&mut tick)?;
        for admission in &tick.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted Nemotron request is retained")
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
        let mut speculative_ids = Vec::new();
        for id in decode_ids {
            let request = self.requests.get(&id).expect("decode request exists");
            let remaining = request
                .generation
                .max_new_tokens
                .saturating_sub(request.generated_tokens);
            if self.model.has_mtp()
                && request.sampler.config().uses_fast_argmax()
                && request.last_token.is_some()
                && remaining >= 4
            {
                speculative_ids.push(id);
                continue;
            }
            if let Some(reason) = self.generate_one(id, &mut tick)? {
                terminal.insert(id, reason);
            }
        }
        self.generate_speculative(&speculative_ids, &mut tick, &mut terminal)?;

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
        self.prefill_blocks(&prefill_ids, &mut tick)?;

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
    pub fn cancel_request(&mut self, id: Nemotron3RequestId) -> Nemotron3CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Nemotron3CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.state.map_or(0, |state| state.device_bytes());
        if released != 0 {
            self.active_sequences -= 1;
        }
        Nemotron3CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns the number of requests currently owning device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(&mut self, tick: &mut Nemotron3Tick) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let restored = match (&mut self.prefix_cache, request.prefix_cache_key.as_ref()) {
                (Some(cache), Some(key)) => {
                    cache.restore(key, Nemotron3SequenceCheckpoint::position, |checkpoint| {
                        self.model
                            .restore_sequence_checkpoint(checkpoint, capacity.max(1))
                    })?
                }
                _ => None,
            };
            let cached_prompt_tokens = restored.as_ref().map_or(0, Nemotron3DecodeState::len);
            let state = restored.unwrap_or(self.model.sequence_state(capacity.max(1))?);
            let bytes = state.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_cache_checkpointed =
                cached_prompt_tokens == request.prefix_cache_target && cached_prompt_tokens != 0;
            request.state = Some(state);
            self.active_sequences += 1;
            tick.admitted.push(Nemotron3AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
            });
        }
        Ok(())
    }

    fn prefill_blocks(
        &mut self,
        ids: &[Nemotron3RequestId],
        tick: &mut Nemotron3Tick,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut remaining_budget = self.config.prefill_token_capacity;
        let mut selected = Vec::new();
        for (index, &id) in ids.iter().enumerate() {
            let request = self.requests.get(&id).expect("prefill request exists");
            let remaining_sequences = ids.len() - index;
            let available = request
                .prompt
                .len()
                .saturating_sub(request.prompt_position + 1);
            let before_checkpoint = prefill_rows_before_checkpoint(
                available,
                request.prompt_position,
                request.prefix_cache_target,
                request.prefix_cache_checkpointed,
            );
            let rows = available
                .min(before_checkpoint)
                .min(remaining_budget.div_ceil(remaining_sequences));
            if rows == 0 {
                continue;
            }
            selected.push((
                id,
                request.prompt_position,
                request.prompt[request.prompt_position..request.prompt_position + rows].to_vec(),
            ));
            remaining_budget -= rows;
        }
        if selected.is_empty() {
            return Ok(());
        }

        let sequence_count = selected.len();
        let rows = selected
            .iter()
            .map(|(_, _, chunk)| chunk.len())
            .sum::<usize>();
        let mtp_rows = selected
            .iter()
            .map(|(_, start, chunk)| chunk.len().saturating_sub(usize::from(*start == 0)))
            .sum::<usize>();
        let mtp_sequence_count = selected
            .iter()
            .filter(|(_, start, chunk)| chunk.len() > usize::from(*start == 0))
            .count();
        let key = (sequence_count, rows, mtp_sequence_count, mtp_rows);
        if !self.prefill_workspaces.contains_key(&key) {
            self.prefill_workspaces.insert(
                key,
                Nemotron3PrefillWorkspace::new(
                    self.model,
                    sequence_count,
                    rows,
                    mtp_sequence_count,
                    mtp_rows,
                )?,
            );
        }
        let workspace = self
            .prefill_workspaces
            .get_mut(&key)
            .expect("prefill workspace was inserted");
        let mut requests = selected
            .iter()
            .map(|(id, _, _)| {
                (
                    *id,
                    self.requests.remove(id).expect("prefill request exists"),
                )
            })
            .collect::<Vec<_>>();
        let chunks = selected
            .iter()
            .map(|(_, _, chunk)| chunk.as_slice())
            .collect::<Vec<_>>();
        let starts = selected
            .iter()
            .map(|(_, start, _)| *start)
            .collect::<Vec<_>>();
        let mut states = requests
            .iter_mut()
            .map(|(_, request)| request.state.as_mut().expect("prefill request has state"))
            .collect::<Vec<_>>();
        self.model
            .capture_final_hidden_rows(&states, &mut workspace.previous_hidden)?;
        self.model
            .forward_block(&mut states, &chunks, &mut workspace.target)?;
        if let (Some(mtp), Some(mtp_hidden)) = (&mut workspace.mtp, &mut workspace.mtp_hidden) {
            let offsets = selected
                .iter()
                .scan(0usize, |offset, (_, _, chunk)| {
                    let current = *offset;
                    *offset += chunk.len();
                    Some(u32::try_from(current).expect("prefill row offsets fit u32"))
                })
                .collect::<Vec<_>>();
            self.model.append_mtp_prompt_block(
                &mut states,
                &chunks,
                &starts,
                &offsets,
                &workspace.previous_hidden,
                workspace.target.final_hidden(),
                mtp_hidden,
                mtp,
            )?;
        }
        drop(states);
        for ((id, mut request), (_, _, chunk)) in requests.into_iter().zip(&selected) {
            request.prompt_position += chunk.len();
            let checkpoint = request.prompt_position == request.prefix_cache_target;
            if checkpoint {
                self.retain_request_checkpoint(&mut request);
            }
            tick.prefilled.push(Nemotron3PrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn retain_request_checkpoint(&mut self, request: &mut ActiveRequest<'template>) {
        if request.prefix_cache_checkpointed || request.prefix_cache_target == 0 {
            return;
        }
        let (Some(cache), Some(key), Some(state)) = (
            self.prefix_cache.as_mut(),
            request.prefix_cache_key.as_ref(),
            request.state.as_ref(),
        ) else {
            return;
        };
        if state.len() != request.prefix_cache_target {
            return;
        }
        if !cache.contains(key) {
            let estimated_bytes = self.model.checkpoint_sequence_device_bytes(state);
            if cache.prepare_insert(estimated_bytes) {
                let started = Instant::now();
                match self.model.checkpoint_sequence(state) {
                    Ok(checkpoint) => {
                        cache.record_checkpoint(started);
                        let device_bytes = checkpoint.device_bytes();
                        if let Err(error) = cache.insert(key.clone(), checkpoint, device_bytes) {
                            warn!(%error, "failed to retain Nemotron prompt prefix checkpoint");
                        }
                    }
                    Err(error) => warn!(%error, "failed to checkpoint Nemotron prompt prefix"),
                }
            }
        }
        request.prefix_cache_checkpointed = true;
    }

    fn generate_one(
        &mut self,
        id: Nemotron3RequestId,
        tick: &mut Nemotron3Tick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let input = request
            .last_token
            .unwrap_or(request.prompt[request.prompt.len() - 1]);
        let state = request.state.as_mut().expect("admitted request has state");
        if let Some(workspace) = self.mtp_token_workspace.as_mut() {
            self.model
                .append_mtp_prompt_token(state, input, workspace)?;
        }
        self.model.forward_one(state, input)?;
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(state)?;
            super::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            let logits = self.model.logits_to_host(state)?;
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

    fn generate_speculative(
        &mut self,
        ids: &[Nemotron3RequestId],
        tick: &mut Nemotron3Tick,
        terminal: &mut BTreeMap<Nemotron3RequestId, ChatFinishReason>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut selected = ids
            .iter()
            .map(|id| {
                (
                    *id,
                    self.requests
                        .remove(id)
                        .expect("selected speculative request exists"),
                )
            })
            .collect::<Vec<_>>();
        let inputs = selected
            .iter()
            .map(|(_, request)| {
                request
                    .last_token
                    .expect("speculative request has input token")
            })
            .collect::<Vec<_>>();
        let result = {
            if !self.speculative_workspaces.contains_key(&selected.len()) {
                let workspace = self.model.speculative_cycle_workspace(selected.len())?;
                self.speculative_workspaces
                    .insert(selected.len(), workspace);
            }
            let workspace = self
                .speculative_workspaces
                .get_mut(&selected.len())
                .expect("speculative workspace was inserted");
            let mut states = selected
                .iter_mut()
                .map(|(_, request)| request.state.as_mut().expect("selected request has state"))
                .collect::<Vec<_>>();
            self.model
                .speculative_cycle_argmax(&mut states, &inputs, workspace)
        };
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                for (id, request) in selected {
                    self.requests.insert(id, request);
                }
                return Err(error);
            }
        };
        for (sequence, (id, mut request)) in selected.into_iter().enumerate() {
            let emitted = result.emitted_tokens(sequence)?;
            request.last_token = emitted.last().copied();
            for token in emitted {
                request.generated_tokens += 1;
                request.history.push(token);
                request.usage.completion_tokens += 1;
                if request.output.is_reasoning() {
                    request.usage.reasoning_tokens += 1;
                }
                tick.generated.push(id);
                let events = request.output.push_token(token)?;
                if let Some(reason) = request.filter.apply(id, events, &mut tick.output) {
                    terminal.insert(id, reason);
                    break;
                }
                if request.generation.eos_token_ids.contains(&token) {
                    terminal.insert(id, ChatFinishReason::Eos);
                    break;
                }
                if request.generated_tokens == request.generation.max_new_tokens {
                    terminal.insert(id, ChatFinishReason::Length);
                    break;
                }
            }
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn finish_request(
        &mut self,
        id: Nemotron3RequestId,
        mut reason: ChatFinishReason,
        tick: &mut Nemotron3Tick,
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
        tick.finished.push(Nemotron3Finished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

fn prefill_rows_before_checkpoint(
    available: usize,
    prompt_position: usize,
    prefix_cache_target: usize,
    prefix_cache_checkpointed: bool,
) -> usize {
    if prefix_cache_checkpointed || prefix_cache_target == 0 {
        available
    } else {
        prefix_cache_target.saturating_sub(prompt_position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_prompts_are_not_blocked_by_an_absent_prefix_checkpoint() {
        assert_eq!(prefill_rows_before_checkpoint(21, 0, 0, false), 21);
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
        request_id: Nemotron3RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Nemotron3ChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(Nemotron3ChatDelta { request_id, event })
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Nemotron3ChatDelta {
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
                    output.push(Nemotron3ChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: Nemotron3RequestId, output: &mut Vec<Nemotron3ChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(Nemotron3ChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
