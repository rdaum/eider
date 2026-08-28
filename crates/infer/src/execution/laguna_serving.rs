//! Multi-session chat serving for Laguna-S-2.1.

use crate::laguna::{
    HEAD_DIM, KV_HEADS, LAYERS, LagunaModel, LagunaNextToken, LagunaPrefillBatchWorkspace,
    LagunaPrefillRow, LagunaSequenceId, LagunaSequencePool,
};
use crate::laguna::{LagunaSequence, LagunaSequenceCache, laguna_cache_error};
use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use eider_cuda::{CudaStream, Error, Result, SM12X_KV_PAGE_TOKENS};
use eider_runtime::cache::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use eider_runtime::chat::{ChatReasoningEffort, CheckpointChatTemplate};
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineDelta, EngineError,
    EngineFinished, EngineLifecycleEvent, EnginePrefillProgress, EngineRequestId, EngineResult,
    EngineService, EngineTick,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::sampling::{SampledToken, Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
use seqcache::{AdmissionOutcome, AdmissionRequest, CacheConfig, PageBackend};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

const TAIL_PREFILL_TOKEN_CAPACITY: usize = 64;
const MAX_CONTINUATION_PREFILL_TOKENS: usize = 1_024;
const REASONING_END: &str = "</think>";

fn prefill_chunk_capacity(prompt_position: usize, fair_share: usize) -> usize {
    if prompt_position == 0 {
        fair_share
    } else {
        fair_share.min(MAX_CONTINUATION_PREFILL_TOKENS)
    }
}

fn reasoning_token_budget(
    effort: Option<ChatReasoningEffort>,
    max_output_tokens: usize,
) -> Option<usize> {
    // This is deliberately a runtime workaround, not a fix for Laguna's
    // reasoning termination. Remove it when a checkpoint resolves the model's
    // tendency to reopen completed reasoning:
    // https://huggingface.co/poolside/Laguna-S-2.1-FP8/discussions/1
    let numerator = match effort? {
        ChatReasoningEffort::Low => 1,
        ChatReasoningEffort::Medium => 2,
        ChatReasoningEffort::High => 3,
        ChatReasoningEffort::XHigh => 4,
    };
    if max_output_tokens < 2 {
        return None;
    }
    Some(
        max_output_tokens
            .saturating_mul(numerator)
            .checked_div(4)
            .unwrap_or(0)
            .max(1)
            .min(max_output_tokens - 1),
    )
}

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
    pub prefilled: Vec<LagunaPrefillProgress>,
    pub generated: Vec<LagunaRequestId>,
    pub output: Vec<LagunaChatDelta>,
    pub finished: Vec<LagunaFinished>,
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
    prefix_target: usize,
    prefix_retained: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    reasoning_token_budget: Option<usize>,
    last_token: Option<u32>,
    pending_sample: Option<SampledToken>,
    sequence_id: Option<LagunaSequenceId>,
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
    sequences: LagunaSequencePool,
    sequence_cache: LagunaSequenceCache,
    cache_stream: CudaStream,
    prefill_workspace: LagunaPrefillBatchWorkspace,
    tail_prefill_workspace: Option<LagunaPrefillBatchWorkspace>,
    reasoning_end_token: u32,
}

impl<'model, 'template> LagunaChatService<'model, 'template> {
    pub fn new_with_cache_config(
        model: &'model LagunaModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
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
        let reasoning_end_token =
            template
                .tokenizer()
                .token_to_id(REASONING_END)
                .ok_or_else(|| Error::Format {
                    label: "Laguna reasoning token",
                    detail: format!("tokenizer does not define {REASONING_END:?}"),
                })?;
        let probe_backend =
            Sm12xPageBackend::new(std::iter::repeat_n(true, LAYERS), 1, KV_HEADS, HEAD_DIM)?;
        let page_bytes = probe_backend.page_bytes();
        let private_state_bytes = model
            .new_sequence_state(config.max_context_tokens)?
            .device_bytes();
        let page_table_bytes = Sm12xPageTable::new(config.max_context_tokens)?.managed_bytes();
        let fixed_per_sequence = private_state_bytes
            .checked_add(page_table_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Laguna sequence-cache fixed bytes",
                expected: "per-sequence byte count without overflow".to_string(),
                actual: format!("private={private_state_bytes} table={page_table_bytes}"),
            })?;
        let fixed_capacity = fixed_per_sequence
            .checked_mul(config.max_active_sequences)
            .ok_or_else(|| Error::Shape {
                label: "Laguna sequence-cache fixed capacity",
                expected: "active fixed-state byte count without overflow".to_string(),
                actual: format!(
                    "per_sequence={fixed_per_sequence} active={}",
                    config.max_active_sequences
                ),
            })?;
        let eager_pages = config
            .max_context_tokens
            .div_ceil(SM12X_KV_PAGE_TOKENS)
            .checked_mul(config.max_active_sequences)
            .ok_or_else(|| Error::Shape {
                label: "Laguna active sequence-cache pages",
                expected: "page count without overflow".to_string(),
                actual: format!(
                    "context={} active={}",
                    config.max_context_tokens, config.max_active_sequences
                ),
            })?;
        let active_page_bytes =
            eager_pages
                .checked_mul(page_bytes)
                .ok_or_else(|| Error::Shape {
                    label: "Laguna active sequence-cache page bytes",
                    expected: "page byte count without overflow".to_string(),
                    actual: format!("pages={eager_pages} page_bytes={page_bytes}"),
                })?;
        let active_capacity = fixed_capacity
            .checked_add(active_page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Laguna active sequence-cache capacity",
                expected: "managed byte count without overflow".to_string(),
                actual: format!("fixed={fixed_capacity} pages={active_page_bytes}"),
            })?;
        let retained_bytes = cache_config.max_retained_bytes;
        let managed_bytes =
            active_capacity
                .checked_add(retained_bytes)
                .ok_or_else(|| Error::Shape {
                    label: "Laguna sequence-cache capacity",
                    expected: "active and retained byte count without overflow".to_string(),
                    actual: format!("active={active_capacity} retained={retained_bytes}"),
                })?;
        let page_slots = eager_pages
            .checked_add(retained_bytes / page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Laguna sequence-cache page slots",
                expected: "page count without overflow".to_string(),
                actual: format!("active={eager_pages} retained={retained_bytes}"),
            })?;
        if page_slots == 0 {
            return Err(Error::Shape {
                label: "Laguna sequence-cache capacity",
                expected: format!(
                    "budget greater than fixed active capacity {fixed_capacity} and one {page_bytes}-byte page"
                ),
                actual: managed_bytes.to_string(),
            });
        }
        let backend = Sm12xPageBackend::new(
            std::iter::repeat_n(true, LAYERS),
            page_slots,
            KV_HEADS,
            HEAD_DIM,
        )?;
        let sequence_cache = LagunaSequenceCache::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: managed_bytes,
                max_snapshot_bytes: 0,
                max_prefix_entries: (cache_config.max_retained_bytes == 0).then_some(0),
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(laguna_cache_error)?;
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            sequences: LagunaSequencePool::new(),
            sequence_cache,
            cache_stream: CudaStream::new_blocking()?,
            prefill_workspace,
            tail_prefill_workspace,
            reasoning_end_token,
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
        let prefix_target =
            retained_prompt_prefix_tokens(prompt.token_ids.len(), SM12X_KV_PAGE_TOKENS);
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        let reasoning_token_budget = if request.template.enable_thinking {
            reasoning_token_budget(
                request.template.reasoning_effort,
                request.generation.max_new_tokens,
            )
        } else {
            None
        };
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids.clone(),
                prompt_position: 0,
                prefix_target,
                prefix_retained: false,
                generation: request.generation.clone(),
                generated_tokens: 0,
                reasoning_token_budget,
                last_token: None,
                pending_sample: None,
                sequence_id: None,
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

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>,
        ),
    ) -> Result<LagunaTick> {
        let tick_started = Instant::now();
        let mut tick = LagunaTick::default();
        self.admit(&mut tick, tick_started, on_lifecycle)?;

        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.sequence_id.is_some()
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
                request.sequence_id.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill(&prefill_ids, &mut tick, on_lifecycle)?;

        for (&id, request) in &self.requests {
            if request.sequence_id.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        Ok(tick)
    }

    pub fn cancel_request(&mut self, id: LagunaRequestId) -> LagunaCancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return LagunaCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = if let Some(sequence_id) = request.sequence_id {
            match self.sequences.release(sequence_id) {
                Ok(sequence) => {
                    let bytes = sequence.device_bytes();
                    if let Err(error) =
                        sequence.finish(&mut self.sequence_cache, &self.cache_stream)
                    {
                        warn!(%error, "failed to release cancelled Laguna sequence");
                    }
                    bytes
                }
                Err(error) => {
                    warn!(%error, "missing cancelled Laguna sequence");
                    0
                }
            }
        } else {
            0
        };
        LagunaCancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    pub fn active_sequence_count(&self) -> usize {
        self.sequences.len()
    }

    fn admit(
        &mut self,
        _tick: &mut LagunaTick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.sequences.len() < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let allocation_started = Instant::now();
            let prefix = self.sequence_cache.lookup_prefix(&request.prompt);
            let mut state = self.model.new_sequence_state(capacity.max(1))?;
            let mut page_table = Sm12xPageTable::new(capacity.max(1))?;
            let outcome = self
                .sequence_cache
                .admit(
                    prefix,
                    AdmissionRequest {
                        max_position: capacity.max(1),
                        private_state_bytes: state.device_bytes(),
                        page_table_bytes: page_table.managed_bytes(),
                        allow_emergency: false,
                    },
                    &mut Sm12xCacheContext {
                        stream: &self.cache_stream,
                        page_table: &mut page_table,
                    },
                    |_snapshot, position| {
                        state.position = position;
                        Ok(())
                    },
                )
                .map_err(laguna_cache_error)?;
            let cache_id = match outcome {
                AdmissionOutcome::Admitted(id) => id,
                AdmissionOutcome::WouldBlock => {
                    self.waiting.push_front(id);
                    break;
                }
            };
            self.cache_stream.synchronize()?;
            let cached_prompt_tokens = state.len();
            let sequence = LagunaSequence::from_admission(cache_id, page_table, state);
            let bytes = sequence.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence_id = Some(self.sequences.insert(sequence)?);
            let progress = LagunaAdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                allocation_duration: allocation_started.elapsed(),
                checkpoint_copy_duration: Duration::ZERO,
                admitted_after_tick_start: tick_started.elapsed(),
            };
            request.usage.cached_prompt_tokens = cached_prompt_tokens;
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
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
            let chunk = batchable
                .min(prefill_chunk_capacity(request.prompt_position, fair_share))
                .min(
                    if !request.prefix_retained && request.prompt_position < request.prefix_target {
                        request.prefix_target - request.prompt_position
                    } else {
                        usize::MAX
                    },
                );
            if chunk == 0 {
                continue;
            }
            budget -= chunk;
            selected.push((id, chunk));
        }
        if !selected.is_empty() {
            let sequence_ids = selected
                .iter()
                .map(|(id, _)| {
                    self.requests[id].sequence_id.ok_or_else(|| Error::Format {
                        label: "Laguna scheduled prefill",
                        detail: format!("request {} has no admitted sequence", id.get()),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut sequences = self.sequences.lease_many(&sequence_ids)?;
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
                    .zip(sequences.sequences_mut())
                    .zip(selected.iter().map(|(_, chunk)| *chunk))
                    .map(|((request, sequence), chunk)| {
                        let start = request.prompt_position;
                        LagunaPrefillRow {
                            token_ids: &request.prompt[start..start + chunk],
                            sequence,
                        }
                    })
                    .collect::<Vec<_>>();
                for &(id, _) in &selected {
                    on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
                }
                self.model
                    .prefill_batch(workspace, &mut rows, &mut self.sequence_cache)
                    .and_then(|()| self.model.synchronize())
            };
            drop(sequences);
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
                    request.prefix_target,
                    request.prefix_retained,
                ) {
                    Self::retain_request_checkpoint(
                        &mut self.sequence_cache,
                        &self.cache_stream,
                        &mut self.sequences,
                        &mut request,
                    );
                }
                tick.prefilled.push(LagunaPrefillProgress {
                    request_id: id,
                    prompt_position: request.prompt_position,
                });
                self.requests.insert(id, request);
            }
            return Ok(());
        }

        for &id in ids {
            let request = self.requests.get_mut(&id).expect("prefill request exists");
            if request.prompt_position + 1 != request.prompt.len() {
                continue;
            }
            on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            let token = request.prompt[request.prompt_position];
            request.prompt_position += 1;
            let sequence_id = request.sequence_id.expect("prefill request is admitted");
            let mut sequences = self.sequences.lease_many(&[sequence_id])?;
            let sequence = sequences.sequence_mut(0);
            let sampled = if request.sampler.config().uses_fast_argmax() {
                sampled_from_top1(self.model.decode_one(
                    sequence,
                    token,
                    &mut self.sequence_cache,
                )?)
            } else {
                let logits = self
                    .model
                    .logits_one(sequence, token, &mut self.sequence_cache)?;
                request.sampler.sample(&logits, &request.history)?
            };
            drop(sequences);
            request.pending_sample = Some(sampled);
            self.model.synchronize()?;
            if checkpoint_ready(
                request.prompt_position,
                request.prefix_target,
                request.prefix_retained,
            ) {
                Self::retain_request_checkpoint(
                    &mut self.sequence_cache,
                    &self.cache_stream,
                    &mut self.sequences,
                    request,
                );
            }
            tick.prefilled.push(LagunaPrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
        }
        Ok(())
    }

    fn retain_request_checkpoint(
        sequence_cache: &mut LagunaSequenceCache,
        cache_stream: &CudaStream,
        sequences: &mut LagunaSequencePool,
        request: &mut ActiveRequest<'template>,
    ) {
        if request.prefix_retained || request.prefix_target == 0 {
            return;
        }
        if sequence_cache.config().max_prefix_entries == Some(0) {
            request.prefix_retained = true;
            return;
        }
        let Some(sequence_id) = request.sequence_id else {
            return;
        };
        let Ok(mut leased) = sequences.lease_many(&[sequence_id]) else {
            warn!("missing sequence while retaining Laguna prompt prefix");
            return;
        };
        let sequence = leased.sequence_mut(0);
        if sequence.position() != request.prefix_target {
            return;
        }
        if sequence_cache.contains_prefix(&request.prompt, request.prefix_target) {
            request.prefix_retained = true;
            return;
        }
        if let Err(error) = sequence_cache.retain_prefix(
            sequence.cache_id,
            &request.prompt,
            (),
            &mut Sm12xCacheContext {
                stream: cache_stream,
                page_table: &mut sequence.page_table,
            },
        ) {
            warn!(%error, "failed to retain shared Laguna prompt prefix");
        }
        request.prefix_retained = true;
    }

    fn generate_one(
        &mut self,
        id: LagunaRequestId,
        tick: &mut LagunaTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let sequence_id = request.sequence_id.expect("decode request is admitted");
        let mut sequences = self.sequences.lease_many(&[sequence_id])?;
        let sequence = sequences.sequence_mut(0);
        let reasoning_budget_reached = request.output.is_reasoning()
            && request
                .reasoning_token_budget
                .is_some_and(|budget| request.usage.reasoning_tokens >= budget);
        let sampled = if reasoning_budget_reached {
            let token = request
                .last_token
                .expect("budgeted Laguna request has generated reasoning");
            self.model
                .consume_one(sequence, token, &mut self.sequence_cache)?;
            SampledToken {
                id: self.reasoning_end_token,
                logit: 0.0,
                adjusted_logit: 0.0,
            }
        } else if let Some(sampled) = request.pending_sample.take() {
            sampled
        } else {
            let token = request
                .last_token
                .expect("generated Laguna token exists after prompt logits");
            if request.sampler.config().uses_fast_argmax() {
                sampled_from_top1(self.model.decode_one(
                    sequence,
                    token,
                    &mut self.sequence_cache,
                )?)
            } else {
                let logits = self
                    .model
                    .logits_one(sequence, token, &mut self.sequence_cache)?;
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
        let sequence_id = request
            .sequence_id
            .take()
            .expect("terminal request is admitted");
        let sequence = self.sequences.release(sequence_id)?;
        let released = sequence.device_bytes();
        sequence.finish(&mut self.sequence_cache, &self.cache_stream)?;
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

fn checkpoint_ready(prompt_position: usize, prefix_target: usize, prefix_retained: bool) -> bool {
    !prefix_retained && prefix_target != 0 && prompt_position >= prefix_target
}

impl EngineService for LagunaChatService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> EngineResult<EngineAdmission> {
        let admission = LagunaChatService::add_request(self, request).map_err(EngineError::new)?;
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
            |event: RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                        request_id: EngineRequestId::new(progress.request_id.get()),
                        sequence_device_bytes: progress.sequence_device_bytes,
                        cached_prompt_tokens: progress.cached_prompt_tokens,
                        allocation_duration: progress.allocation_duration,
                        checkpoint_copy_duration: progress.checkpoint_copy_duration,
                        admitted_after_tick_start: progress.admitted_after_tick_start,
                    }))
                }
                RequestLifecycleEvent::PrefillStarted(id) => on_lifecycle(
                    EngineLifecycleEvent::PrefillStarted(EngineRequestId::new(id.get())),
                ),
            };
        let tick = LagunaChatService::tick_with_lifecycle(self, &mut observer)
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
            verification: Vec::new(),
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
        match LagunaChatService::cancel_request(self, LagunaRequestId(id.get())) {
            LagunaCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            LagunaCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        LagunaChatService::active_sequence_count(self)
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
    use super::{
        ChatReasoningEffort, MAX_CONTINUATION_PREFILL_TOKENS, checkpoint_ready,
        prefill_chunk_capacity, reasoning_token_budget,
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
    fn reasoning_effort_reserves_output_capacity_for_the_answer() {
        assert_eq!(
            reasoning_token_budget(Some(ChatReasoningEffort::Low), 4_096),
            Some(1_024)
        );
        assert_eq!(
            reasoning_token_budget(Some(ChatReasoningEffort::Medium), 4_096),
            Some(2_048)
        );
        assert_eq!(
            reasoning_token_budget(Some(ChatReasoningEffort::High), 4_096),
            Some(3_072)
        );
        assert_eq!(reasoning_token_budget(None, 4_096), None);
        assert_eq!(
            reasoning_token_budget(Some(ChatReasoningEffort::Low), 2),
            Some(1)
        );
        assert_eq!(
            reasoning_token_budget(Some(ChatReasoningEffort::Low), 1),
            None
        );
    }
}
