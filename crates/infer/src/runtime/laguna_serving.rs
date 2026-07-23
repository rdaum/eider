//! Multi-session chat serving for Laguna-S-2.1.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::prefix_cache::{
    PrefixCache, PrefixCacheConfig, PrefixCacheKey, cacheable_prompt_prefix_tokens,
};
use super::sampling::{SampledToken, Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::laguna::{LagunaDecodeState, LagunaModel, LagunaNextToken, LagunaSequenceCheckpoint};
use nvfp4::{Error, Result};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LagunaRequestId(u64);

impl LagunaRequestId {
    pub fn get(self) -> u64 {
        self.0
    }
}

pub struct LagunaAdmission {
    pub request_id: LagunaRequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaAdmissionProgress {
    pub request_id: LagunaRequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    pub allocation_duration: Duration,
    pub checkpoint_copy_duration: Duration,
    pub admitted_after_tick_start: Duration,
}

pub struct LagunaPrefillProgress {
    pub request_id: LagunaRequestId,
    pub prompt_position: usize,
}

pub struct LagunaChatDelta {
    pub request_id: LagunaRequestId,
    pub event: ChatOutputEvent,
}

pub struct LagunaFinished {
    pub request_id: LagunaRequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

#[derive(Default)]
pub struct LagunaTick {
    pub admitted: Vec<LagunaAdmissionProgress>,
    pub prefilled: Vec<LagunaPrefillProgress>,
    pub generated: Vec<LagunaRequestId>,
    pub output: Vec<LagunaChatDelta>,
    pub finished: Vec<LagunaFinished>,
    pub active_sequences: usize,
}

pub enum LagunaCancelOutcome {
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
    state: Option<LagunaDecodeState>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

pub struct LagunaChatService<'model, 'template> {
    model: &'model LagunaModel,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<LagunaRequestId>,
    requests: BTreeMap<LagunaRequestId, ActiveRequest<'template>>,
    active_sequences: usize,
    prefix_cache: Option<PrefixCache<LagunaSequenceCheckpoint>>,
}

impl<'model, 'template> LagunaChatService<'model, 'template> {
    pub fn new(
        model: &'model LagunaModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_prefix_cache(model, template, config, PrefixCacheConfig::default())
    }

    pub fn new_with_prefix_cache(
        model: &'model LagunaModel,
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
        })
    }

    pub fn add_request(&mut self, request: ChatRequest) -> Result<LagunaAdmission> {
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
                label: "Laguna chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Laguna request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Laguna request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = LagunaRequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Laguna request ID",
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
        Ok(LagunaAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    pub fn tick(&mut self) -> Result<LagunaTick> {
        self.tick_with_lifecycle(&mut |_| {})
    }

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>,
        ),
    ) -> Result<LagunaTick> {
        let tick_started = Instant::now();
        let mut tick = LagunaTick::default();
        self.admit(&mut tick, tick_started, on_lifecycle)?;
        for admission in &tick.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted Laguna request is retained")
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

    pub fn cancel_request(&mut self, id: LagunaRequestId) -> LagunaCancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return LagunaCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request.state.map_or(0, |state| state.device_bytes());
        if released != 0 {
            self.active_sequences -= 1;
        }
        LagunaCancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut LagunaTick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>,
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
                    cache.restore(key, LagunaSequenceCheckpoint::position, |checkpoint| {
                        let restore_started = Instant::now();
                        let state = self
                            .model
                            .restore_sequence_checkpoint(checkpoint, capacity.max(1))?;
                        checkpoint_copy_duration = restore_started.elapsed();
                        Ok(state)
                    })?
                }
                _ => None,
            };
            let cached_prompt_tokens = restored.as_ref().map_or(0, LagunaDecodeState::len);
            let state = if let Some(restored) = restored {
                restored
            } else {
                let allocation_started = Instant::now();
                let state = self.model.new_decode_state(capacity.max(1))?;
                allocation_duration = allocation_started.elapsed();
                state
            };
            let bytes = state.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_cache_checkpointed =
                cached_prompt_tokens == request.prefix_cache_target && cached_prompt_tokens != 0;
            request.state = Some(state);
            self.active_sequences += 1;
            let progress = LagunaAdmissionProgress {
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
        ids: &[LagunaRequestId],
        tick: &mut LagunaTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>,
        ),
    ) -> Result<()> {
        let mut budget = self.config.prefill_token_capacity;
        for (index, &id) in ids.iter().enumerate() {
            if budget == 0 {
                break;
            }
            let remaining_sequences = ids.len() - index;
            let request = self.requests.get_mut(&id).expect("prefill request exists");
            let available = request.prompt.len() - request.prompt_position;
            let chunk = available.min(budget.div_ceil(remaining_sequences));
            if chunk == 0 {
                continue;
            }
            budget -= chunk;
            on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            let end = request.prompt_position + chunk;
            while request.prompt_position < end {
                let token = request.prompt[request.prompt_position];
                request.prompt_position += 1;
                if request.prompt_position == request.prompt.len() {
                    let sampled = if request.sampler.config().uses_fast_argmax() {
                        sampled_from_top1(self.model.decode_one(
                            request.state.as_mut().expect("prefill request is admitted"),
                            token,
                        )?)
                    } else {
                        let logits = self.model.logits_one(
                            request.state.as_mut().expect("prefill request is admitted"),
                            token,
                        )?;
                        request.sampler.sample(&logits, &request.history)?
                    };
                    request.pending_sample = Some(sampled);
                } else {
                    self.model.consume_one(
                        request.state.as_mut().expect("prefill request is admitted"),
                        token,
                    )?;
                }
            }
            self.model.synchronize()?;
            if checkpoint_ready(
                request.prompt_position,
                request.prefix_cache_target,
                request.prefix_cache_checkpointed,
            ) {
                Self::retain_request_checkpoint(self.model, &mut self.prefix_cache, request);
            }
            tick.prefilled.push(LagunaPrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
        }
        Ok(())
    }

    fn retain_request_checkpoint(
        model: &LagunaModel,
        prefix_cache: &mut Option<PrefixCache<LagunaSequenceCheckpoint>>,
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
                match model.checkpoint_sequence(state, request.prefix_cache_target) {
                    Ok(checkpoint) => {
                        cache.record_checkpoint(started);
                        let bytes = checkpoint.device_bytes();
                        if let Err(error) = cache.insert(key.clone(), checkpoint, bytes) {
                            warn!(%error, "failed to retain Laguna prompt prefix checkpoint");
                        }
                    }
                    Err(error) => warn!(%error, "failed to checkpoint Laguna prompt prefix"),
                }
            }
        }
        request.prefix_cache_checkpointed = true;
    }

    fn generate_one(
        &mut self,
        id: LagunaRequestId,
        tick: &mut LagunaTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let sampled = if let Some(sampled) = request.pending_sample.take() {
            sampled
        } else {
            let token = request
                .last_token
                .expect("generated Laguna token exists after prompt logits");
            if request.sampler.config().uses_fast_argmax() {
                sampled_from_top1(self.model.decode_one(
                    request.state.as_mut().expect("decode request is admitted"),
                    token,
                )?)
            } else {
                let logits = self.model.logits_one(
                    request.state.as_mut().expect("decode request is admitted"),
                    token,
                )?;
                request.sampler.sample(&logits, &request.history)?
            }
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
        id: LagunaRequestId,
        mut reason: ChatFinishReason,
        tick: &mut LagunaTick,
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
        tick.finished.push(LagunaFinished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

fn sampled_from_top1(token: LagunaNextToken) -> SampledToken {
    SampledToken {
        id: token.token,
        logit: token.logit,
        adjusted_logit: token.logit,
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
        request_id: LagunaRequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<LagunaChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(LagunaChatDelta { request_id, event }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(LagunaChatDelta {
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
                    output.push(LagunaChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: LagunaRequestId, output: &mut Vec<LagunaChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(LagunaChatDelta {
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
