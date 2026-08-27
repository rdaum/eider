//! Multi-session chat serving for the Qwen3.8 Flash Next native QSA path.

use super::cache_config::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::sampling::{SampledToken, Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::stop::StopBuffer;
use crate::nvfp4::{DeviceBuffer, Error, GpuSamplingRow, GpuTokenSampler, Result};
use crate::qwen3::infer::QwenLayerKind;
use crate::qwen38_flash_next::{
    Qwen38FlashNextModel, Qwen38FlashNextMtpSequenceState, Qwen38FlashNextMtpWorkspace,
    Qwen38FlashNextPrefillWorkspace, Qwen38FlashNextSpeculativeFrontier,
    Qwen38FlashNextSpeculativeWorkspace, Qwen38LogitsMode, Qwen38NextToken,
};
use crate::runtime::qwen38_flash_next_sequence::{
    Qwen38FlashNextMtpSequenceCache, Qwen38FlashNextMtpSnapshot, Qwen38FlashNextSequence,
    Qwen38FlashNextSequenceCache, new_qwen38_flash_next_mtp_sequence_cache,
    new_qwen38_flash_next_sequence_cache_with_config, qwen38_flash_next_cache_error,
};
use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Qwen38FlashNextRequestId(u64);

impl Qwen38FlashNextRequestId {
    pub fn get(self) -> u64 {
        self.0
    }
}

pub struct Qwen38FlashNextAdmission {
    pub request_id: Qwen38FlashNextRequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38FlashNextAdmissionProgress {
    pub request_id: Qwen38FlashNextRequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    pub allocation_duration: Duration,
    pub admitted_after_tick_start: Duration,
}

pub struct Qwen38FlashNextPrefillProgress {
    pub request_id: Qwen38FlashNextRequestId,
    pub prompt_position: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38FlashNextSpeculativeProgress {
    pub request_id: Qwen38FlashNextRequestId,
    pub cycles: usize,
    pub accepted_drafts: usize,
}

pub struct Qwen38FlashNextChatDelta {
    pub request_id: Qwen38FlashNextRequestId,
    pub event: ChatOutputEvent,
}

pub struct Qwen38FlashNextFinished {
    pub request_id: Qwen38FlashNextRequestId,
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
    pub released_sequence_device_bytes: usize,
}

#[derive(Default)]
pub struct Qwen38FlashNextTick {
    pub admitted: Vec<Qwen38FlashNextAdmissionProgress>,
    pub prefilled: Vec<Qwen38FlashNextPrefillProgress>,
    pub generated: Vec<Qwen38FlashNextRequestId>,
    pub speculative: Vec<Qwen38FlashNextSpeculativeProgress>,
    pub output: Vec<Qwen38FlashNextChatDelta>,
    pub finished: Vec<Qwen38FlashNextFinished>,
    pub active_sequences: usize,
}

pub enum Qwen38FlashNextCancelOutcome {
    Cancelled {
        released_sequence_device_bytes: usize,
    },
    NotFound,
}

struct ActiveRequest<'tokenizer> {
    prompt: Vec<u32>,
    prompt_position: usize,
    prefix_target: usize,
    prefix_retained: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    prompt_logits_ready: bool,
    sequence: Option<Box<Qwen38FlashNextSequence>>,
    mtp_sequence: Option<Qwen38FlashNextMtpSequenceState>,
    speculative_frontier: Option<Qwen38FlashNextSpeculativeFrontier>,
    sampler: Sampler,
    history: TokenHistory,
    device_token_counts: Option<DeviceBuffer<u32>>,
    sequence_device_bytes: usize,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Decode-first multi-session service for the native QSA runtime.
pub struct Qwen38FlashNextChatService<'template> {
    model: Qwen38FlashNextModel,
    prefill_workspace: Qwen38FlashNextPrefillWorkspace,
    sequence_cache: Qwen38FlashNextSequenceCache,
    mtp_sequence_cache: Option<Qwen38FlashNextMtpSequenceCache>,
    mtp_workspace: Option<Qwen38FlashNextMtpWorkspace>,
    speculative_workspace: Option<Qwen38FlashNextSpeculativeWorkspace>,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<Qwen38FlashNextRequestId>,
    prefilling: VecDeque<Qwen38FlashNextRequestId>,
    decoding: VecDeque<Qwen38FlashNextRequestId>,
    requests: BTreeMap<Qwen38FlashNextRequestId, ActiveRequest<'template>>,
    active_sequences: usize,
    gpu_sampler: GpuTokenSampler,
}

impl<'template> Qwen38FlashNextChatService<'template> {
    pub fn new(
        model: Qwen38FlashNextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_cache_config(model, template, config, SequenceCacheConfig::default())
    }

    pub fn new_with_cache_config(
        model: Qwen38FlashNextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_position_embeddings {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next server context",
                expected: format!("at most {} tokens", model.config().max_position_embeddings),
                actual: format!("{} tokens", config.max_context_tokens),
            });
        }
        if config.speculative_drafts > 1 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next speculative decoding",
                expected: "zero or one native MTP draft".to_string(),
                actual: config.speculative_drafts.to_string(),
            });
        }
        if config.speculative_drafts == 1 && !model.mtp_enabled() {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next speculative decoding",
                detail: "native MTP weights were not enabled while loading the model".to_string(),
            });
        }
        let prefill_workspace = model.new_prefill_workspace(config.prefill_token_capacity)?;
        let sequence_cache = new_qwen38_flash_next_sequence_cache_with_config(
            &model,
            config.max_active_sequences,
            config.max_context_tokens,
            cache_config,
        )?;
        let gpu_sampler = GpuTokenSampler::new(1, model.config().vocab)?;
        let mtp_sequence_cache = (config.speculative_drafts == 1)
            .then(|| {
                let target_qsa_layers = model
                    .manifest()
                    .layer_kinds
                    .iter()
                    .filter(|&&kind| kind == QwenLayerKind::FullAttention)
                    .count()
                    .max(1);
                new_qwen38_flash_next_mtp_sequence_cache(
                    &model,
                    config.max_active_sequences,
                    config.max_context_tokens,
                    cache_config.max_retained_bytes / target_qsa_layers,
                )
            })
            .transpose()?;
        let mtp_workspace = (config.speculative_drafts == 1)
            .then(|| {
                model.new_mtp_workspace(config.max_context_tokens, config.prefill_token_capacity)
            })
            .transpose()?;
        let speculative_workspace = (config.speculative_drafts == 1)
            .then(|| model.new_speculative_workspace(1))
            .transpose()?;
        Ok(Self {
            model,
            prefill_workspace,
            sequence_cache,
            mtp_sequence_cache,
            mtp_workspace,
            speculative_workspace,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            decoding: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
            gpu_sampler,
        })
    }

    pub fn add_request(&mut self, request: ChatRequest) -> Result<Qwen38FlashNextAdmission> {
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
                label: "Qwen3.8 Flash Next chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 Flash Next request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = Qwen38FlashNextRequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prompt_tokens = prompt.token_ids.len();
        let prefix_target = retained_prompt_prefix_tokens(prompt_tokens);
        let max_output_tokens = request.generation.max_new_tokens;
        let sampler = Sampler::new(request.generation.sampling)?;
        let history = TokenHistory::from_tokens(prompt.token_ids.iter().copied());
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids,
                prompt_position: 0,
                prefix_target,
                prefix_retained: false,
                generation: request.generation,
                generated_tokens: 0,
                last_token: None,
                prompt_logits_ready: false,
                sequence: None,
                mtp_sequence: None,
                speculative_frontier: None,
                sampler,
                history,
                device_token_counts: None,
                sequence_device_bytes: 0,
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
        Ok(Qwen38FlashNextAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen38FlashNextRequestId, Qwen38FlashNextAdmissionProgress>,
        ),
    ) -> Result<Qwen38FlashNextTick> {
        let started = Instant::now();
        let mut tick = Qwen38FlashNextTick::default();
        self.admit(&mut tick, started, on_lifecycle)?;
        let mut terminal = BTreeMap::new();

        let decode_count = self.decoding.len().min(self.config.decode_capacity);
        for _ in 0..decode_count {
            let id = self.decoding.pop_front().expect("decode request exists");
            if let Some(reason) = self.generate_one(id, &mut tick)? {
                terminal.insert(id, reason);
            } else {
                self.decoding.push_back(id);
            }
        }

        let prefill_count = self
            .prefilling
            .len()
            .min(self.config.prefill_sequence_capacity);
        let mut token_budget = self.config.prefill_token_capacity;
        for slot in 0..prefill_count {
            if token_budget == 0 {
                break;
            }
            let id = self.prefilling.pop_front().expect("prefill request exists");
            let slots_left = prefill_count - slot;
            let fair_share = token_budget.div_ceil(slots_left);
            let consumed = self.prefill(id, fair_share, &mut tick, on_lifecycle)?;
            token_budget -= consumed;
            let request = self.requests.get(&id).expect("prefill request remains");
            if request.prompt_position == request.prompt.len() {
                self.decoding.push_back(id);
            } else {
                self.prefilling.push_back(id);
            }
        }
        for (&id, request) in &self.requests {
            if request.sequence.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        tick.active_sequences = self.active_sequences;
        Ok(tick)
    }

    pub fn cancel_request(&mut self, id: Qwen38FlashNextRequestId) -> Qwen38FlashNextCancelOutcome {
        let Some(mut request) = self.requests.remove(&id) else {
            return Qwen38FlashNextCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        self.prefilling.retain(|&prefilling| prefilling != id);
        self.decoding.retain(|&decoding| decoding != id);
        if let Some(mtp_sequence) = request.mtp_sequence.take()
            && let Some(sequence) = request.sequence.as_deref()
            && let Some(cache) = self.mtp_sequence_cache.as_mut()
        {
            let _ = mtp_sequence.finish(cache, sequence.state.stream());
        }
        if let Some(sequence) = request.sequence.take() {
            let _ = sequence.finish(&mut self.sequence_cache);
            self.active_sequences -= 1;
        }
        Qwen38FlashNextCancelOutcome::Cancelled {
            released_sequence_device_bytes: request.sequence_device_bytes,
        }
    }

    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut Qwen38FlashNextTick,
        started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen38FlashNextRequestId, Qwen38FlashNextAdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let allocation_started = Instant::now();
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let device_token_counts = if request.generation.sampling.supports_gpu_sampling()
                && request.generation.sampling.uses_history_penalties()
            {
                Some(DeviceBuffer::from_host(
                    &request.history.dense_counts(self.model.config().vocab),
                )?)
            } else {
                None
            };
            let sampling_bytes = device_token_counts
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes);
            let sequence = Qwen38FlashNextSequence::admit_with_prefix_and_private_bytes(
                &self.model,
                &mut self.sequence_cache,
                capacity.max(1),
                &request.prompt,
                sampling_bytes,
            )?;
            let cached_prompt_tokens = sequence.position();
            let (mtp_sequence, speculative_frontier) = if self.config.speculative_drafts == 1
                && request.generation.sampling.uses_fast_argmax()
                && request.generation.max_new_tokens != 0
            {
                let mtp_cache = self
                    .mtp_sequence_cache
                    .as_mut()
                    .expect("MTP cache was allocated");
                let mtp_sequence = self.model.new_mtp_sequence_state(
                    mtp_cache,
                    capacity.max(1),
                    &request.prompt,
                    sequence.state.stream(),
                )?;
                let mut frontier = Qwen38FlashNextSpeculativeFrontier {
                    token: 0,
                    logit: 0.0,
                    previous_streams: DeviceBuffer::zeroed(
                        self.model.config().hidden * self.model.config().hc_count,
                    )?,
                };
                if mtp_sequence.position() == cached_prompt_tokens {
                    if cached_prompt_tokens != 0 {
                        self.model.copy_decode_target_streams(
                            &sequence.state,
                            &mut frontier.previous_streams,
                        )?;
                    }
                    (Some(mtp_sequence), Some(frontier))
                } else {
                    mtp_sequence.finish(mtp_cache, sequence.state.stream())?;
                    (None, None)
                }
            } else {
                (None, None)
            };
            let mtp_bytes = mtp_sequence
                .as_ref()
                .map_or(0, Qwen38FlashNextMtpSequenceState::device_bytes)
                .checked_add(
                    speculative_frontier
                        .as_ref()
                        .map_or(0, |frontier| frontier.previous_streams.device_bytes()),
                )
                .ok_or_else(|| Error::Shape {
                    label: "Qwen3.8 Flash Next MTP admitted bytes",
                    expected: "MTP state bytes without overflow".to_string(),
                    actual: "overflow".to_string(),
                })?;
            let sequence_device_bytes = sequence
                .device_bytes()
                .checked_add(sampling_bytes)
                .and_then(|bytes| bytes.checked_add(mtp_bytes))
                .ok_or_else(|| Error::Shape {
                    label: "Qwen3.8 Flash Next admitted bytes",
                    expected: "sequence and sampling bytes without overflow".to_string(),
                    actual: format!(
                        "sequence={} sampling={sampling_bytes}",
                        sequence.device_bytes()
                    ),
                })?;
            let progress = Qwen38FlashNextAdmissionProgress {
                request_id: id,
                sequence_device_bytes,
                cached_prompt_tokens,
                allocation_duration: allocation_started.elapsed(),
                admitted_after_tick_start: started.elapsed(),
            };
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence = Some(Box::new(sequence));
            request.mtp_sequence = mtp_sequence;
            request.speculative_frontier = speculative_frontier;
            request.device_token_counts = device_token_counts;
            request.sequence_device_bytes = sequence_device_bytes;
            request.usage.cached_prompt_tokens = cached_prompt_tokens;
            self.active_sequences += 1;
            if request.generation.max_new_tokens != 0 {
                self.prefilling.push_back(id);
            }
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        id: Qwen38FlashNextRequestId,
        token_capacity: usize,
        tick: &mut Qwen38FlashNextTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen38FlashNextRequestId, Qwen38FlashNextAdmissionProgress>,
        ),
    ) -> Result<usize> {
        let request = self.requests.get_mut(&id).expect("prefill request exists");
        let mut end = (request.prompt_position + token_capacity).min(request.prompt.len());
        if !request.prefix_retained
            && request.prompt_position < request.prefix_target
            && end > request.prefix_target
        {
            end = request.prefix_target;
        }
        if end == request.prompt_position {
            return Ok(0);
        }
        on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
        let sequence = request
            .sequence
            .as_deref_mut()
            .expect("request is admitted");
        let start = request.prompt_position;
        let final_prompt_chunk = end == request.prompt.len();
        let logits = if !final_prompt_chunk {
            Qwen38LogitsMode::None
        } else if request.sampler.config().uses_fast_argmax() {
            Qwen38LogitsMode::Top1
        } else {
            Qwen38LogitsMode::Full
        };
        sequence.forward_tokens(
            &mut self.model,
            &mut self.prefill_workspace,
            &mut self.sequence_cache,
            &request.prompt[start..end],
            logits,
        )?;
        if let (Some(mtp_sequence), Some(frontier)) = (
            request.mtp_sequence.as_mut(),
            request.speculative_frontier.as_mut(),
        ) {
            let mtp_cache = self
                .mtp_sequence_cache
                .as_mut()
                .expect("MTP cache was allocated");
            let mtp_workspace = self
                .mtp_workspace
                .as_mut()
                .expect("MTP workspace was allocated");
            self.model.mtp_prefill_tokens(
                mtp_sequence,
                mtp_workspace,
                mtp_cache,
                &request.prompt[start..end],
                &self.prefill_workspace,
                &frontier.previous_streams,
                sequence.state.stream(),
            )?;
            self.model.copy_prefill_target_streams(
                &self.prefill_workspace,
                end - start - 1,
                &mut frontier.previous_streams,
                sequence.state.stream(),
            )?;
            if final_prompt_chunk {
                let next = self.model.read_top1(&sequence.state)?;
                frontier.token = next.id;
                frontier.logit = next.value;
            }
        }
        request.prompt_position = end;
        request.prompt_logits_ready =
            end == request.prompt.len() && request.speculative_frontier.is_none();
        Self::retain_request_checkpoint(
            &self.model,
            &mut self.sequence_cache,
            self.mtp_sequence_cache.as_mut(),
            request,
        );
        tick.prefilled.push(Qwen38FlashNextPrefillProgress {
            request_id: id,
            prompt_position: end,
        });
        Ok(end - start)
    }

    fn retain_request_checkpoint(
        model: &Qwen38FlashNextModel,
        sequence_cache: &mut Qwen38FlashNextSequenceCache,
        mut mtp_sequence_cache: Option<&mut Qwen38FlashNextMtpSequenceCache>,
        request: &mut ActiveRequest<'template>,
    ) {
        if request.prefix_retained || request.prefix_target == 0 {
            return;
        }
        if sequence_cache.config().max_prefix_entries == Some(0) {
            request.prefix_retained = true;
            return;
        }
        let Some(sequence) = request.sequence.as_deref_mut() else {
            return;
        };
        if sequence.position() != request.prefix_target {
            return;
        }
        if sequence_cache.contains_prefix(&request.prompt, request.prefix_target) {
            request.prefix_retained = true;
            return;
        }
        let snapshot = match model.snapshot_sequence(&sequence.state) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                warn!(%error, "failed to copy Qwen3.8 recurrent prompt-prefix snapshot");
                request.prefix_retained = true;
                return;
            }
        };
        if let Err(error) = sequence_cache.retain_prefix(
            sequence.cache_id,
            &request.prompt,
            snapshot,
            &mut Sm12xCacheContext {
                stream: sequence.state.stream(),
                page_table: &mut sequence.page_table,
            },
        ) {
            let error = qwen38_flash_next_cache_error(error);
            warn!(%error, "failed to retain shared Qwen3.8 prompt prefix");
        }
        if let (Some(cache), Some(mtp_sequence)) =
            (mtp_sequence_cache.as_mut(), request.mtp_sequence.as_mut())
            && mtp_sequence.position() == request.prefix_target
            && !cache.contains_prefix(&request.prompt, request.prefix_target)
            && let Err(error) = cache.retain_prefix(
                mtp_sequence.cache_id,
                &request.prompt,
                Qwen38FlashNextMtpSnapshot,
                &mut Sm12xCacheContext {
                    stream: sequence.state.stream(),
                    page_table: &mut mtp_sequence.page_table,
                },
            )
        {
            let error = qwen38_flash_next_cache_error(error);
            warn!(%error, "failed to retain Qwen3.8 MTP prompt prefix");
        }
        request.prefix_retained = true;
    }

    fn generate_one(
        &mut self,
        id: Qwen38FlashNextRequestId,
        tick: &mut Qwen38FlashNextTick,
    ) -> Result<Option<ChatFinishReason>> {
        if self.requests[&id].speculative_frontier.is_some() {
            return self.generate_speculative(id, tick);
        }
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let sampling = request.sampler.config();
        if request.prompt_logits_ready {
            request.prompt_logits_ready = false;
        } else {
            let token = request
                .last_token
                .expect("generated token exists after prompt logits");
            let logits = if sampling.uses_fast_argmax() {
                Qwen38LogitsMode::Top1
            } else {
                Qwen38LogitsMode::Full
            };
            request
                .sequence
                .as_deref_mut()
                .expect("request is admitted")
                .forward_token(&mut self.model, &mut self.sequence_cache, token, logits)?;
        }
        let sequence = request
            .sequence
            .as_deref()
            .expect("decode request is admitted");
        let sampled = if sampling.uses_fast_argmax() {
            let next = self.model.read_top1(&sequence.state)?;
            SampledToken {
                id: next.id,
                logit: next.value,
                adjusted_logit: next.value,
            }
        } else if sampling.supports_gpu_sampling() {
            let draw = if sampling.temperature == 0.0 || sampling.top_k == 1 {
                0.0
            } else {
                request.sampler.next_gpu_draw()
            };
            let mut row = GpuSamplingRow {
                temperature: sampling.temperature,
                top_k: sampling.top_k,
                top_p: sampling.top_p,
                presence_penalty: sampling.presence_penalty,
                frequency_penalty: sampling.frequency_penalty,
                draw,
                token_counts: request.device_token_counts.as_mut(),
            };
            let sampled =
                self.model
                    .sample_logits_gpu(&sequence.state, &mut self.gpu_sampler, &mut row)?;
            SampledToken {
                id: sampled.id,
                logit: sampled.logit,
                adjusted_logit: sampled.adjusted_logit,
            }
        } else {
            let logits = self.model.logits_to_host(&sequence.state)?;
            request.sampler.sample(&logits, &request.history)?
        };
        Self::record_generated(request, id, sampled, tick)
    }

    fn generate_speculative(
        &mut self,
        id: Qwen38FlashNextRequestId,
        tick: &mut Qwen38FlashNextTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let remaining = request
            .generation
            .max_new_tokens
            .saturating_sub(request.generated_tokens);
        if remaining == 0 {
            return Ok(Some(ChatFinishReason::Length));
        }
        let committed = if remaining == 1 {
            let frontier = request
                .speculative_frontier
                .as_ref()
                .expect("speculative request has a frontier");
            vec![Qwen38NextToken {
                id: frontier.token,
                value: frontier.logit,
            }]
        } else {
            let sequence = request
                .sequence
                .as_deref_mut()
                .expect("request is admitted");
            let outcome = self.model.speculative_cycle_argmax(
                self.speculative_workspace
                    .as_mut()
                    .expect("speculative workspace was allocated"),
                request
                    .speculative_frontier
                    .as_mut()
                    .expect("speculative request has a frontier"),
                &mut sequence.state,
                &mut self.sequence_cache,
                sequence.cache_id,
                &mut sequence.page_table,
                request
                    .mtp_sequence
                    .as_mut()
                    .expect("speculative request has MTP state"),
                self.mtp_workspace
                    .as_mut()
                    .expect("MTP workspace was allocated"),
                self.mtp_sequence_cache
                    .as_mut()
                    .expect("MTP cache was allocated"),
            )?;
            tick.speculative.push(Qwen38FlashNextSpeculativeProgress {
                request_id: id,
                cycles: 1,
                accepted_drafts: outcome.accepted_drafts,
            });
            outcome.committed
        };
        for next in committed.into_iter().take(remaining) {
            let sampled = SampledToken {
                id: next.id,
                logit: next.value,
                adjusted_logit: next.value,
            };
            if let Some(reason) = Self::record_generated(request, id, sampled, tick)? {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }

    fn record_generated(
        request: &mut ActiveRequest<'template>,
        id: Qwen38FlashNextRequestId,
        sampled: SampledToken,
        tick: &mut Qwen38FlashNextTick,
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

    fn finish_request(
        &mut self,
        id: Qwen38FlashNextRequestId,
        mut reason: ChatFinishReason,
        tick: &mut Qwen38FlashNextTick,
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
        self.prefilling.retain(|&prefilling| prefilling != id);
        self.decoding.retain(|&decoding| decoding != id);
        if let Some(mtp_sequence) = request.mtp_sequence.take()
            && let Some(sequence) = request.sequence.as_deref()
            && let Some(cache) = self.mtp_sequence_cache.as_mut()
        {
            mtp_sequence.finish(cache, sequence.state.stream())?;
        }
        if let Some(sequence) = request.sequence.take() {
            sequence.finish(&mut self.sequence_cache)?;
        }
        self.active_sequences -= 1;
        tick.finished.push(Qwen38FlashNextFinished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: request.sequence_device_bytes,
        });
        Ok(())
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
        request_id: Qwen38FlashNextRequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<Qwen38FlashNextChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(Qwen38FlashNextChatDelta { request_id, event })
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(Qwen38FlashNextChatDelta {
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
                    output.push(Qwen38FlashNextChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(
        &mut self,
        request_id: Qwen38FlashNextRequestId,
        output: &mut Vec<Qwen38FlashNextChatDelta>,
    ) {
        let text = self.stop.finish();
        if !text.is_empty() && !self.saw_tool_calls {
            output.push(Qwen38FlashNextChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}
