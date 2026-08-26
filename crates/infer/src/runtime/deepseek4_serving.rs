//! Multi-session chat serving for DeepSeek V4.

use super::cache_config::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::deepseek4_sequence_cache::{
    Deepseek4CacheContext, Deepseek4MtpSequence, Deepseek4MtpSequenceCache, Deepseek4Sequence,
    Deepseek4SequenceCache, deepseek4_cache_error, new_deepseek4_mtp_sequence_cache,
    new_deepseek4_sequence_cache,
};
use super::sampling::{SampledToken, Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::sm12x_sequence_cache::Sm12xPageTable;
use super::stop::StopBuffer;
use crate::deepseek4::{
    Deepseek4BatchRow, Deepseek4BatchWorkspace, Deepseek4LayerSequenceState, Deepseek4MtpBatchRow,
    Deepseek4MtpWorkspace, Deepseek4TextModel,
};
use nvfp4::{Error, Result};
use seqcache::{AdmissionOutcome, AdmissionRequest};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::Path;
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

fn retention_bounded_chunk(
    chunk: usize,
    prompt_position: usize,
    prefix_target: usize,
    retention_enabled: bool,
    prefix_retained: bool,
) -> usize {
    if prefix_retained || !retention_enabled || prompt_position >= prefix_target {
        chunk
    } else {
        chunk.min(prefix_target - prompt_position)
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

pub struct Deepseek4SpeculativeProgress {
    pub request_id: Deepseek4RequestId,
    pub cycles: usize,
    pub accepted_drafts: usize,
}

#[derive(Default)]
pub struct Deepseek4Tick {
    pub admitted: Vec<Deepseek4AdmissionProgress>,
    pub prefilled: Vec<Deepseek4PrefillProgress>,
    pub generated: Vec<Deepseek4RequestId>,
    pub speculative: Vec<Deepseek4SpeculativeProgress>,
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
    prefix_target: usize,
    prefix_retained: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    pending_sample: Option<SampledToken>,
    sequence: Option<Deepseek4Sequence>,
    mtp_sequence: Option<Deepseek4MtpSequence>,
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
    sequence_cache: Deepseek4SequenceCache,
    mtp_sequence_cache: Option<Deepseek4MtpSequenceCache>,
    retain_prefixes: bool,
    workspace: Deepseek4BatchWorkspace,
    mtp_workspace: Option<Deepseek4MtpWorkspace>,
}

impl<'template> Deepseek4ChatService<'template> {
    pub fn new(
        model: Deepseek4TextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_cache_config(model, template, config, SequenceCacheConfig::default())
    }

    pub fn new_with_cache_config(
        model: Deepseek4TextModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.speculative_drafts > 1 {
            return Err(Error::Shape {
                label: "DeepSeek V4 speculative drafts",
                expected: "zero or one native MTP draft".to_string(),
                actual: config.speculative_drafts.to_string(),
            });
        }
        let speculative = config.speculative_drafts == 1;
        if speculative && !model.mtp_enabled() {
            return Err(Error::Format {
                label: "DeepSeek V4 speculative decoding",
                detail: "model was not loaded with native MTP weights".to_string(),
            });
        }
        let workspace = model.new_batch_workspace(
            config.decode_capacity.max(config.prefill_sequence_capacity),
            config.prefill_token_capacity.max(config.decode_capacity),
            config.max_context_tokens,
        )?;
        let retained_bytes = if speculative {
            None
        } else {
            (cache_config.max_retained_bytes != 0).then_some(cache_config.max_retained_bytes)
        };
        let sequence_cache = new_deepseek4_sequence_cache(
            &model,
            config.max_active_sequences,
            config.max_context_tokens,
            retained_bytes,
        )?;
        let mtp_sequence_cache = speculative
            .then(|| {
                new_deepseek4_mtp_sequence_cache(
                    &model,
                    config.max_active_sequences,
                    config.max_context_tokens,
                )
            })
            .transpose()?;
        let mtp_workspace = speculative
            .then(|| {
                model.new_mtp_workspace(
                    config.decode_capacity.max(config.prefill_sequence_capacity),
                    config
                        .prefill_token_capacity
                        .max(config.decode_capacity * 2),
                )
            })
            .transpose()?;
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
            sequence_cache,
            mtp_sequence_cache,
            retain_prefixes: retained_bytes.is_some(),
            workspace,
            mtp_workspace,
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
        let prefix_target = retained_prompt_prefix_tokens(prompt.token_ids.len());
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids.clone(),
                prompt_position: 0,
                prefix_target,
                prefix_retained: false,
                generation: request.generation.clone(),
                generated_tokens: 0,
                last_token: None,
                pending_sample: None,
                sequence: None,
                mtp_sequence: None,
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

    /// Atomically writes cumulative routing observations for hot-cache preparation.
    pub fn write_hotset_plan(
        &self,
        path: impl AsRef<Path>,
        capacity_per_layer: usize,
    ) -> Result<usize> {
        if self.active_sequences != 0 {
            return Err(Error::Format {
                label: "DeepSeek V4 hotset plan",
                detail: "cannot snapshot routing while sequences are active".to_string(),
            });
        }
        let path = path.as_ref();
        let plan = self
            .model
            .hotset_plan(capacity_per_layer, &self.workspace)?;
        let bytes = serde_json::to_vec_pretty(&plan).map_err(|error| Error::Format {
            label: "DeepSeek V4 hotset plan",
            detail: format!("failed to encode plan: {error}"),
        })?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|error| Error::Format {
                label: "DeepSeek V4 hotset plan",
                detail: format!("failed to create {}: {error}", parent.display()),
            })?;
        }
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, bytes).map_err(|error| Error::Format {
            label: "DeepSeek V4 hotset plan",
            detail: format!("failed to write {}: {error}", temporary.display()),
        })?;
        fs::rename(&temporary, path).map_err(|error| Error::Format {
            label: "DeepSeek V4 hotset plan",
            detail: format!(
                "failed to publish {} as {}: {error}",
                temporary.display(),
                path.display()
            ),
        })?;
        Ok(plan.values().map(Vec::len).sum())
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
                request.sequence.is_some()
                    && request.prompt_position >= request.prompt.len()
                    && request.generated_tokens < request.generation.max_new_tokens
            })
            .map(|(&id, _)| id)
            .take(self.config.decode_capacity)
            .collect::<Vec<_>>();
        let mut ordinary_ids = Vec::new();
        let mut speculative_ids = Vec::new();
        for id in decode_ids {
            let request = self.requests.get(&id).expect("decode request exists");
            let remaining = request
                .generation
                .max_new_tokens
                .saturating_sub(request.generated_tokens);
            if self.mtp_workspace.is_some()
                && request.pending_sample.is_none()
                && request.last_token.is_some()
                && request.sampler.config().uses_fast_argmax()
                && remaining >= 2
            {
                speculative_ids.push(id);
            } else {
                ordinary_ids.push(id);
            }
        }
        for (id, reason) in self.generate(&ordinary_ids, &mut tick)? {
            terminal.insert(id, reason);
        }
        self.generate_speculative(&speculative_ids, &mut tick, &mut terminal)?;

        let prefill_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.sequence.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick, on_lifecycle)?;

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

    pub fn cancel_request(&mut self, id: Deepseek4RequestId) -> Deepseek4CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Deepseek4CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request
            .sequence
            .as_ref()
            .map_or(0, Deepseek4Sequence::device_bytes)
            .saturating_add(
                request
                    .mtp_sequence
                    .as_ref()
                    .map_or(0, Deepseek4MtpSequence::device_bytes),
            );
        if let Some(sequence) = request.sequence {
            if let Err(error) = sequence.finish(self.workspace.stream(), &mut self.sequence_cache) {
                warn!(%error, request_id = id.get(), "failed to release cancelled DeepSeek V4 sequence");
            }
            self.active_sequences -= 1;
        }
        if let Some(sequence) = request.mtp_sequence
            && let Some(cache) = self.mtp_sequence_cache.as_mut()
            && let Err(error) = sequence.finish(self.workspace.stream(), cache)
        {
            warn!(%error, request_id = id.get(), "failed to release cancelled DeepSeek V4 MTP sequence");
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
            let allocation_started = Instant::now();
            let mut checkpoint_copy_duration = Duration::ZERO;
            let prefix = self.sequence_cache.lookup_prefix(&request.prompt);
            let mut state = Some(self.model.new_sequence_state(capacity.max(1))?);
            let mut page_table = Sm12xPageTable::new(capacity.max(1))?;
            let outcome = self
                .sequence_cache
                .admit(
                    prefix,
                    AdmissionRequest {
                        max_position: capacity.max(1),
                        private_state_bytes: state
                            .as_ref()
                            .expect("state allocated")
                            .device_bytes(),
                        page_table_bytes: page_table.managed_bytes(),
                        allow_emergency: false,
                    },
                    &mut Deepseek4CacheContext {
                        stream: self.workspace.stream(),
                        page_table: &mut page_table,
                    },
                    |snapshot, position| {
                        if let Some(snapshot) = snapshot {
                            let started = Instant::now();
                            state = Some(self.model.restore_sequence_checkpoint(
                                snapshot,
                                capacity.max(1),
                                &self.workspace,
                            )?);
                            checkpoint_copy_duration = started.elapsed();
                        }
                        debug_assert_eq!(
                            state.as_ref().map_or(0, |state| state.position()),
                            position
                        );
                        Ok(())
                    },
                )
                .map_err(deepseek4_cache_error)?;
            let AdmissionOutcome::Admitted(cache_id) = outcome else {
                self.waiting.push_front(id);
                break;
            };
            let state = state.take().expect("admitted state retained");
            let cached_prompt_tokens = state.position();
            let sequence = Deepseek4Sequence::from_admission(cache_id, page_table, state);
            let mtp_sequence = if let Some(cache) = self.mtp_sequence_cache.as_mut() {
                debug_assert_eq!(cached_prompt_tokens, 0);
                let mut page_table = Sm12xPageTable::new(capacity.max(1))?;
                let state = Deepseek4LayerSequenceState::new(
                    &self.model.weights.config,
                    self.model.weights.config.num_hidden_layers,
                    capacity.max(1),
                )?;
                let private_state_bytes = state.device_bytes().saturating_add(
                    self.model.weights.config.hc_mult
                        * self.model.weights.config.hidden_size
                        * 2
                        * std::mem::size_of::<f32>(),
                );
                let outcome = cache
                    .admit(
                        None,
                        AdmissionRequest {
                            max_position: capacity.max(1),
                            private_state_bytes,
                            page_table_bytes: page_table.managed_bytes(),
                            allow_emergency: false,
                        },
                        &mut Deepseek4CacheContext {
                            stream: self.workspace.stream(),
                            page_table: &mut page_table,
                        },
                        |_, position| {
                            debug_assert_eq!(position, 0);
                            Ok(())
                        },
                    )
                    .map_err(deepseek4_cache_error)?;
                let AdmissionOutcome::Admitted(cache_id) = outcome else {
                    return Err(Error::Format {
                        label: "DeepSeek V4 MTP admission",
                        detail: "preallocated MTP cache refused an admitted target sequence"
                            .to_string(),
                    });
                };
                Some(Deepseek4MtpSequence::from_admission(
                    cache_id,
                    page_table,
                    state,
                    self.model.weights.config.hc_mult * self.model.weights.config.hidden_size,
                )?)
            } else {
                None
            };
            let bytes = sequence.device_bytes().saturating_add(
                mtp_sequence
                    .as_ref()
                    .map_or(0, Deepseek4MtpSequence::device_bytes),
            );
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence = Some(sequence);
            request.mtp_sequence = mtp_sequence;
            self.active_sequences += 1;
            let progress = Deepseek4AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                allocation_duration: allocation_started
                    .elapsed()
                    .saturating_sub(checkpoint_copy_duration),
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
            chunk = retention_bounded_chunk(
                chunk,
                request.prompt_position,
                request.prefix_target,
                self.retain_prefixes,
                request.prefix_retained,
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
                for &(id, _) in &selected {
                    on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
                }
                if let (Some(mtp_workspace), Some(mtp_cache)) = (
                    self.mtp_workspace.as_mut(),
                    self.mtp_sequence_cache.as_mut(),
                ) {
                    let mut rows = Vec::with_capacity(requests.len());
                    let mut mtp_rows = Vec::with_capacity(requests.len());
                    for (request, chunk) in requests
                        .iter_mut()
                        .zip(selected.iter().map(|(_, chunk)| *chunk))
                    {
                        let ActiveRequest {
                            prompt,
                            prompt_position,
                            sequence,
                            mtp_sequence,
                            ..
                        } = request;
                        let tokens = &prompt[*prompt_position..*prompt_position + chunk];
                        rows.push(Deepseek4BatchRow {
                            token_ids: tokens,
                            sequence: sequence.as_mut().expect("prefill request is admitted"),
                        });
                        mtp_rows.push(Deepseek4MtpBatchRow {
                            token_ids: tokens,
                            sequence: mtp_sequence
                                .as_mut()
                                .expect("speculative request has an MTP sequence"),
                        });
                    }
                    self.model.prefill_batch_with_mtp(
                        &mut self.workspace,
                        mtp_workspace,
                        &mut rows,
                        &mut mtp_rows,
                        &mut self.sequence_cache,
                        mtp_cache,
                    )
                } else {
                    let mut rows = requests
                        .iter_mut()
                        .zip(selected.iter().map(|(_, chunk)| *chunk))
                        .map(|(request, chunk)| {
                            let start = request.prompt_position;
                            Deepseek4BatchRow {
                                token_ids: &request.prompt[start..start + chunk],
                                sequence: request
                                    .sequence
                                    .as_mut()
                                    .expect("prefill request is admitted"),
                            }
                        })
                        .collect::<Vec<_>>();
                    self.model.prefill_batch(
                        &mut self.workspace,
                        &mut rows,
                        &mut self.sequence_cache,
                    )
                }
            };
            if let Err(error) = result {
                for (request, (id, _)) in requests.into_iter().zip(&selected) {
                    self.requests.insert(*id, request);
                }
                return Err(error);
            }
            for (mut request, (id, chunk)) in requests.into_iter().zip(selected) {
                request.prompt_position += chunk;
                if retained_prefix_ready(
                    request.prompt_position,
                    request.prefix_target,
                    request.prefix_retained,
                ) {
                    self.retain_request_prefix(&mut request);
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
            for &id in &tail_ids {
                on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            }
            if let (Some(mtp_workspace), Some(mtp_cache)) = (
                self.mtp_workspace.as_mut(),
                self.mtp_sequence_cache.as_mut(),
            ) {
                let mut rows = Vec::with_capacity(requests.len());
                let mut mtp_rows = Vec::with_capacity(requests.len());
                for request in &mut requests {
                    let ActiveRequest {
                        prompt,
                        prompt_position,
                        sequence,
                        mtp_sequence,
                        ..
                    } = request;
                    let tokens = &prompt[*prompt_position..];
                    rows.push(Deepseek4BatchRow {
                        token_ids: tokens,
                        sequence: sequence.as_mut().expect("prefill request is admitted"),
                    });
                    mtp_rows.push(Deepseek4MtpBatchRow {
                        token_ids: tokens,
                        sequence: mtp_sequence
                            .as_mut()
                            .expect("speculative request has an MTP sequence"),
                    });
                }
                self.model.forward_batch_with_mtp(
                    &mut self.workspace,
                    mtp_workspace,
                    &mut rows,
                    &mut mtp_rows,
                    &mut self.sequence_cache,
                    mtp_cache,
                )
            } else {
                let mut rows = requests
                    .iter_mut()
                    .map(|request| Deepseek4BatchRow {
                        token_ids: &request.prompt[request.prompt_position..],
                        sequence: request
                            .sequence
                            .as_mut()
                            .expect("prefill request is admitted"),
                    })
                    .collect::<Vec<_>>();
                self.model
                    .forward_batch(&mut self.workspace, &mut rows, &mut self.sequence_cache)
                    .and_then(|logits| logits.copy_to_host())
            }
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
        let sampled = requests
            .iter_mut()
            .zip(logits.chunks_exact(vocab))
            .map(|(request, row_logits)| request.sampler.sample(row_logits, &request.history))
            .collect::<Result<Vec<_>>>();
        let sampled = match sampled {
            Ok(sampled) => sampled,
            Err(error) => {
                for (request, id) in requests.into_iter().zip(&tail_ids) {
                    self.requests.insert(*id, request);
                }
                return Err(error);
            }
        };
        for ((mut request, id), sampled) in requests.into_iter().zip(tail_ids).zip(sampled) {
            request.prompt_position += 1;
            request.pending_sample = Some(sampled);
            if retained_prefix_ready(
                request.prompt_position,
                request.prefix_target,
                request.prefix_retained,
            ) {
                self.retain_request_prefix(&mut request);
            }
            tick.prefilled.push(Deepseek4PrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn retain_request_prefix(&mut self, request: &mut ActiveRequest<'template>) {
        if request.prefix_retained || request.prefix_target == 0 || !self.retain_prefixes {
            return;
        }
        let Some(sequence) = request.sequence.as_mut() else {
            return;
        };
        if sequence.position() != request.prefix_target {
            return;
        }
        if !self
            .sequence_cache
            .contains_prefix(&request.prompt, request.prefix_target)
        {
            match self
                .model
                .checkpoint_sequence(&sequence.state, &self.workspace)
            {
                Ok(snapshot) => {
                    if let Err(error) = self.sequence_cache.retain_prefix(
                        sequence.cache_id,
                        &request.prompt,
                        snapshot,
                        &mut Deepseek4CacheContext {
                            stream: self.workspace.stream(),
                            page_table: &mut sequence.page_table,
                        },
                    ) {
                        warn!(error = %deepseek4_cache_error(error), "failed to retain DeepSeek V4 prompt prefix");
                    }
                }
                Err(error) => warn!(%error, "failed to snapshot DeepSeek V4 prompt prefix"),
            }
        }
        request.prefix_retained = true;
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
                            sequence: request
                                .sequence
                                .as_mut()
                                .expect("decode request is admitted"),
                        })
                    })
                    .collect::<Vec<_>>();
                self.model
                    .forward_batch(&mut self.workspace, &mut rows, &mut self.sequence_cache)
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
        let result = (|| {
            let mut terminal = Vec::new();
            for (request, &id) in requests.iter_mut().zip(ids) {
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
                if let Some(reason) = apply_sample(request, id, sampled, tick)? {
                    terminal.push((id, reason));
                }
            }
            debug_assert!(logits_rows.next().is_none());
            Ok(terminal)
        })();
        for (request, &id) in requests.into_iter().zip(ids) {
            self.requests.insert(id, request);
        }
        result
    }

    fn generate_speculative(
        &mut self,
        ids: &[Deepseek4RequestId],
        tick: &mut Deepseek4Tick,
        terminal: &mut BTreeMap<Deepseek4RequestId, ChatFinishReason>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut requests = ids
            .iter()
            .map(|id| self.requests.remove(id).expect("decode request exists"))
            .collect::<Vec<_>>();
        let inputs = requests
            .iter()
            .map(|request| request.last_token.expect("speculative input exists"))
            .collect::<Vec<_>>();
        let result = {
            let mut target_sequences = Vec::with_capacity(requests.len());
            let mut mtp_sequences = Vec::with_capacity(requests.len());
            for request in &mut requests {
                let ActiveRequest {
                    sequence,
                    mtp_sequence,
                    ..
                } = request;
                target_sequences.push(
                    sequence
                        .as_mut()
                        .expect("speculative target sequence exists"),
                );
                mtp_sequences.push(
                    mtp_sequence
                        .as_mut()
                        .expect("speculative MTP sequence exists"),
                );
            }
            self.model.speculative_cycle_argmax(
                &mut self.workspace,
                self.mtp_workspace
                    .as_mut()
                    .expect("speculative workspace exists"),
                &mut target_sequences,
                &mut mtp_sequences,
                &inputs,
                &mut self.sequence_cache,
                self.mtp_sequence_cache
                    .as_mut()
                    .expect("speculative cache exists"),
            )
        };
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                for (request, &id) in requests.into_iter().zip(ids) {
                    self.requests.insert(id, request);
                }
                return Err(error);
            }
        };
        let accepted = result.accepted_counts().collect::<Vec<_>>();
        for (sequence, ((mut request, &id), accepted)) in
            requests.into_iter().zip(ids).zip(accepted).enumerate()
        {
            tick.speculative.push(Deepseek4SpeculativeProgress {
                request_id: id,
                cycles: 1,
                accepted_drafts: accepted as usize,
            });
            for token in result.emitted_tokens(sequence)? {
                let sampled = SampledToken {
                    id: token,
                    logit: 0.0,
                    adjusted_logit: 0.0,
                };
                if let Some(reason) = apply_sample(&mut request, id, sampled, tick)? {
                    terminal.insert(id, reason);
                    break;
                }
            }
            self.requests.insert(id, request);
        }
        Ok(())
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
        let sequence = request
            .sequence
            .take()
            .expect("terminal request is admitted");
        let mtp_sequence = request.mtp_sequence.take();
        let released = sequence.device_bytes().saturating_add(
            mtp_sequence
                .as_ref()
                .map_or(0, Deepseek4MtpSequence::device_bytes),
        );
        sequence.finish(self.workspace.stream(), &mut self.sequence_cache)?;
        if let Some(sequence) = mtp_sequence {
            sequence.finish(
                self.workspace.stream(),
                self.mtp_sequence_cache
                    .as_mut()
                    .expect("MTP sequence has an MTP cache"),
            )?;
        }
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

fn retained_prefix_ready(
    prompt_position: usize,
    prefix_target: usize,
    prefix_retained: bool,
) -> bool {
    !prefix_retained && prefix_target != 0 && prompt_position >= prefix_target
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
        let mut emitted_tool_call = false;
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
                    emitted_tool_call = true;
                }
            }
        }
        emitted_tool_call.then_some(ChatFinishReason::ToolCalls)
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
        Deepseek4RequestId, MAX_CONTINUATION_PREFILL_TOKENS, ResponseFilter,
        prefill_chunk_capacity, retained_prefix_ready, retention_bounded_chunk,
    };
    use crate::runtime::chat::{ChatFunctionCall, ChatToolCall};
    use crate::runtime::chat_output::ChatOutputEvent;
    use crate::runtime::serving::ChatFinishReason;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn checkpoint_is_ready_after_crossing_the_aligned_prefix() {
        assert!(retained_prefix_ready(384, 256, false));
        assert!(retained_prefix_ready(256, 256, false));
        assert!(!retained_prefix_ready(128, 256, false));
    }

    #[test]
    fn disabled_or_completed_checkpoint_is_not_ready() {
        assert!(!retained_prefix_ready(256, 0, false));
        assert!(!retained_prefix_ready(256, 256, true));
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
        assert_eq!(retention_bounded_chunk(1_024, 128, 512, true, false), 384);
        assert_eq!(retention_bounded_chunk(1_024, 512, 512, true, false), 1_024);
        assert_eq!(
            retention_bounded_chunk(1_024, 128, 512, false, false),
            1_024
        );
        assert_eq!(retention_bounded_chunk(1_024, 128, 512, true, true), 1_024);
    }

    #[test]
    fn response_filter_preserves_every_tool_call_in_one_dsml_block() {
        let events = ["read_file", "write_file"]
            .into_iter()
            .map(|name| {
                ChatOutputEvent::ToolCall(ChatToolCall {
                    id: format!("call_{name}"),
                    function: ChatFunctionCall {
                        name: name.to_string(),
                        arguments: BTreeMap::from([("path".to_string(), json!("README.md"))]),
                    },
                })
            })
            .collect();
        let mut filter = ResponseFilter::new(Vec::new());
        let mut output = Vec::new();
        assert_eq!(
            filter.apply(Deepseek4RequestId(1), events, &mut output),
            Some(ChatFinishReason::ToolCalls)
        );
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|delta| {
            matches!(delta.event, ChatOutputEvent::ToolCall(_))
                && delta.request_id == Deepseek4RequestId(1)
        }));
    }
}
