//! Tokenized Qwen3.6 scheduling over chunked prefill and batched decode.

use super::cache_config::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use super::qwen36_sequence::{Qwen36Sequence, Qwen36SequenceCache};
use super::sampling::{SampledToken, Sampler, SamplingConfig, TokenHistory};
use super::sm12x_sequence_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use crate::metrics::metrics;
use crate::qwen3::qwen36::{
    Qwen36DecodeBatchWorkspace, Qwen36DecodeRow, Qwen36MtpDraftWorkspace, Qwen36MtpSequenceState,
    Qwen36NextToken, Qwen36PrefillBatchWorkspace, Qwen36PrefillRow,
    Qwen36SpeculativeCycleWorkspace, Qwen36SpeculativeFrontier, Qwen36TextModel,
    Qwen38DFlash2SequenceState, Qwen38DFlash2Workspace,
};
use nvfp4::{CudaStream, DeviceBuffer, Error, GpuSamplingRow, Result, SM12X_KV_PAGE_TOKENS};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, CacheConfig, CacheError, CacheStats, PageBackend,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::mem::size_of;
use std::time::{Duration, Instant};
use tracing::warn;

const MAX_SPECULATIVE_DRAFTS: usize = 7;

/// Stable scheduler identity for one request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Qwen36RequestId(u64);

impl Qwen36RequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(value: u64) -> Self {
        Self(value)
    }
}

/// Request lifecycle visible to a serving frontend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    /// Accepted on the CPU but not yet allocated persistent GPU state.
    Waiting,
    /// Consuming all but the final prompt token into persistent model state.
    Prefilling,
    /// Producing completion tokens.
    Decoding,
    /// Reached EOS or the requested completion length.
    Finished,
}

/// Execution and admission limits for one scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    /// Maximum independent rows in one latency-sensitive decode batch.
    pub decode_capacity: usize,
    /// Maximum independent prompt chunks in one prefill batch.
    pub prefill_sequence_capacity: usize,
    /// Maximum total prompt tokens consumed by one prefill batch.
    pub prefill_token_capacity: usize,
    /// Maximum requests with allocated persistent GPU state.
    pub max_active_sequences: usize,
    /// Maximum prompt plus completion tokens for any request.
    pub max_context_tokens: usize,
    /// Draft tokens verified per speculative cycle; zero disables speculation.
    pub speculative_drafts: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            decode_capacity: 8,
            prefill_sequence_capacity: 8,
            prefill_token_capacity: 2_048,
            max_active_sequences: 8,
            max_context_tokens: 32_768,
            speculative_drafts: 0,
        }
    }
}

impl SchedulerConfig {
    pub(crate) fn validate(self) -> Result<()> {
        if self.decode_capacity == 0
            || self.prefill_sequence_capacity == 0
            || self.prefill_token_capacity == 0
            || self.max_active_sequences == 0
            || self.max_context_tokens == 0
        {
            return Err(Error::Shape {
                label: "scheduler configuration",
                expected: "all capacities greater than zero".to_string(),
                actual: format!(
                    "decode={} prefill_sequences={} prefill_tokens={} active={} context={}",
                    self.decode_capacity,
                    self.prefill_sequence_capacity,
                    self.prefill_token_capacity,
                    self.max_active_sequences,
                    self.max_context_tokens
                ),
            });
        }
        if self.speculative_drafts > MAX_SPECULATIVE_DRAFTS {
            return Err(Error::Shape {
                label: "scheduler speculative drafts",
                expected: format!("at most {MAX_SPECULATIVE_DRAFTS} drafts"),
                actual: format!("{} drafts", self.speculative_drafts),
            });
        }
        Ok(())
    }
}

/// Token-level generation policy for a scheduled request.
#[derive(Clone, Debug)]
pub struct RequestConfig {
    /// Token selection policy.
    pub sampling: SamplingConfig,
    /// Maximum number of completion tokens.
    pub max_new_tokens: usize,
    /// Model token IDs that terminate generation.
    pub eos_token_ids: BTreeSet<u32>,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            max_new_tokens: 64,
            eos_token_ids: BTreeSet::new(),
        }
    }
}

impl RequestConfig {
    /// Validates token-selection parameters.
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()
    }
}

/// Why a tokenized scheduled request completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestFinishReason {
    /// The model selected a configured EOS token.
    Eos,
    /// The request reached its completion-token limit.
    Length,
}

/// One completion token produced by a scheduler tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen36ScheduledToken {
    /// Request that owns the token.
    pub request_id: Qwen36RequestId,
    /// Selected vocabulary ID.
    pub id: u32,
    /// Original model logit for the selected ID.
    pub logit: f32,
    /// Present when this token finishes the request.
    pub finish_reason: Option<RequestFinishReason>,
}

/// Prompt progress made for one request during a scheduler tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen36PrefillProgress {
    /// Request whose prompt state advanced.
    pub request_id: Qwen36RequestId,
    /// Prompt tokens consumed in this tick.
    pub tokens: usize,
    /// Total prompt tokens consumed after this tick.
    pub prompt_position: usize,
}

/// Persistent sequence state allocated for a newly admitted request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen36AdmissionProgress {
    /// Request receiving device-resident sequence state.
    pub request_id: Qwen36RequestId,
    /// Exact bytes owned by the newly allocated sequence state.
    pub sequence_device_bytes: usize,
    /// Prompt tokens restored from a reusable hybrid-state checkpoint.
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time when admission completed.
    pub admitted_after_tick_start: Duration,
}

/// A lifecycle event emitted at the point where scheduler work begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestLifecycleEvent<RequestId, AdmissionProgress> {
    /// Device-resident sequence state is ready for the request.
    Admitted(AdmissionProgress),
    /// The request's next prompt chunk is about to enter the model.
    PrefillStarted(RequestId),
}

/// Observable result of one scheduler tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen36SchedulerTick {
    /// Requests moved from the CPU waiting queue into device-resident state.
    pub admitted: Vec<Qwen36AdmissionProgress>,
    /// Requests selected for model work, with decode rows before prefill rows.
    pub scheduled: Vec<Qwen36RequestId>,
    /// Prompt progress made by the prefill batch.
    pub prefilled: Vec<Qwen36PrefillProgress>,
    /// Completion tokens produced after prompt consumption.
    pub generated: Vec<Qwen36ScheduledToken>,
    /// Qwen3.8 speculation completed during the tick.
    pub speculative: Vec<Qwen38SpeculativeProgress>,
    /// Requests that finished during this tick.
    pub finished: Vec<Qwen36RequestId>,
    /// Device-resident sequences remaining after the tick.
    pub active_sequences: usize,
}

/// Request-scoped Qwen3.8 draft acceptance observed during one scheduler tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38SpeculativeProgress {
    /// Request whose target model verified the drafts.
    pub request_id: Qwen36RequestId,
    /// Number of completed target-verification cycles.
    pub cycles: usize,
    /// Draft tokens accepted by the target model.
    pub accepted_drafts: usize,
}

/// Result removed from the scheduler after a request finishes.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36FinishedRequest {
    /// Stable request identity.
    pub id: Qwen36RequestId,
    /// Original prompt tokens.
    pub prompt_tokens: Vec<u32>,
    /// Generated completion tokens and logits.
    pub generated_tokens: Vec<Qwen36ScheduledToken>,
    /// Final completion reason.
    pub finish_reason: RequestFinishReason,
    /// Device bytes released when the request reached its terminal state.
    pub released_sequence_device_bytes: usize,
}

/// Request data returned when active or waiting work is cancelled.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36CancelledRequest {
    /// Stable request identity.
    pub id: Qwen36RequestId,
    /// Original prompt tokens.
    pub prompt_tokens: Vec<u32>,
    /// Completion tokens produced before cancellation.
    pub generated_tokens: Vec<Qwen36ScheduledToken>,
    /// Device bytes released by cancellation, or zero for a waiting request.
    pub released_sequence_device_bytes: usize,
}

/// Outcome of a scheduler cancellation request.
#[derive(Clone, Debug, PartialEq)]
pub enum Qwen36CancelOutcome {
    /// Waiting or active work was removed and any GPU state was released.
    Cancelled(Qwen36CancelledRequest),
    /// The request had already reached a normal terminal state.
    AlreadyFinished,
    /// No retained request has this identity.
    NotFound,
}

struct Qwen36Request {
    id: Qwen36RequestId,
    lifecycle: RequestState,
    config: RequestConfig,
    prompt_tokens: Vec<u32>,
    prompt_position: usize,
    prefix_target: usize,
    prefix_retained: bool,
    sequence: Option<Box<Qwen36Sequence>>,
    device_token_counts: Option<DeviceBuffer<u32>>,
    sequence_device_bytes: usize,
    sampler: Sampler,
    history: TokenHistory,
    last_token: Option<u32>,
    generated_tokens: Vec<Qwen36ScheduledToken>,
    finish_reason: Option<RequestFinishReason>,
    mtp_state: Option<Qwen36MtpSequenceState>,
    dflash2_state: Option<Qwen38DFlash2SequenceState>,
    spec_frontier: Option<Qwen36SpeculativeFrontier>,
    spec_ready: bool,
    spec_started: bool,
}

impl Qwen36Request {
    fn max_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.config.max_new_tokens
    }

    fn remaining_prompt_tokens(&self) -> usize {
        self.prompt_tokens.len() - self.prompt_position
    }

    fn prefillable_tokens(&self) -> usize {
        self.remaining_prompt_tokens().saturating_sub(1)
    }

    fn decode_input_token(&self) -> Result<u32> {
        if self.remaining_prompt_tokens() == 1 {
            return Ok(self.prompt_tokens[self.prompt_position]);
        }
        if self.spec_ready {
            return self
                .spec_frontier
                .as_ref()
                .map(|frontier| frontier.token)
                .ok_or_else(|| Error::Format {
                    label: "Qwen3.8 speculative request",
                    detail: format!("request {} is ready without a frontier", self.id.get()),
                });
        }
        self.last_token.ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduled request",
            detail: format!("request {} has no decode input token", self.id.get()),
        })
    }

    fn active_speculative_drafts(&self, configured: usize) -> usize {
        speculative_draft_count(
            configured,
            self.config.max_new_tokens,
            self.generated_tokens.len(),
            self.spec_started,
        )
    }

    fn apply_sample(&mut self, sampled: SampledToken) -> Qwen36ScheduledToken {
        if self.remaining_prompt_tokens() == 1 {
            self.prompt_position += 1;
        }
        self.last_token = Some(sampled.id);
        self.history.push(sampled.id);
        let generated_count = self.generated_tokens.len() + 1;
        let finish_reason = if self.config.eos_token_ids.contains(&sampled.id) {
            Some(RequestFinishReason::Eos)
        } else if generated_count == self.config.max_new_tokens {
            Some(RequestFinishReason::Length)
        } else {
            None
        };
        self.lifecycle = if finish_reason.is_some() {
            RequestState::Finished
        } else {
            RequestState::Decoding
        };
        self.finish_reason = finish_reason;
        let token = Qwen36ScheduledToken {
            request_id: self.id,
            id: sampled.id,
            logit: sampled.logit,
            finish_reason,
        };
        self.generated_tokens.push(token);
        token
    }
}

/// Decode-first continuous scheduler with deferred GPU admission.
pub struct Qwen36Scheduler<'model> {
    model: &'model Qwen36TextModel,
    config: SchedulerConfig,
    decode_workspaces: Vec<Qwen36DecodeBatchWorkspace>,
    prefill_workspace: Qwen36PrefillBatchWorkspace,
    requests: BTreeMap<Qwen36RequestId, Box<Qwen36Request>>,
    waiting: VecDeque<Qwen36RequestId>,
    prefilling: VecDeque<Qwen36RequestId>,
    decoding: VecDeque<Qwen36RequestId>,
    sequence_cache: Qwen36SequenceCache,
    cache_stream: CudaStream,
    spec_workspace: Option<Qwen36SpeculativeCycleWorkspace>,
    mtp_workspace: Option<Qwen36MtpDraftWorkspace>,
    mtp_hidden_scratch: Option<DeviceBuffer<f32>>,
    dflash2_workspace: Option<Qwen38DFlash2Workspace>,
    next_id: u64,
}

impl<'model> Qwen36Scheduler<'model> {
    /// Creates a scheduler with explicit execution and admission limits.
    pub fn new(model: &'model Qwen36TextModel, config: SchedulerConfig) -> Result<Self> {
        Self::new_with_cache_config(model, config, SequenceCacheConfig::default())
    }

    /// Creates a scheduler with explicit execution, admission, and cache limits.
    pub fn new_with_cache_config(
        model: &'model Qwen36TextModel,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let mut decode_workspaces = decode_capacity_classes(config.decode_capacity)
            .into_iter()
            .map(|capacity| model.new_decode_batch_workspace(capacity, config.max_context_tokens))
            .collect::<Result<Vec<_>>>()?;
        let mut prefill_workspace = model.new_prefill_batch_workspace(
            config.prefill_sequence_capacity,
            config.prefill_token_capacity,
            config.max_context_tokens,
        )?;
        if model.dflash2_enabled() && config.speculative_drafts > 0 {
            for workspace in &mut decode_workspaces {
                model.enable_dflash2_decode_capture(workspace)?;
            }
            model.enable_dflash2_prefill_capture(&mut prefill_workspace)?;
        }
        let probe_backend = Sm12xPageBackend::new(
            model
                .manifest()
                .layer_kinds
                .iter()
                .map(|kind| *kind == crate::qwen3::infer::QwenLayerKind::FullAttention),
            1,
            model.manifest().kv_heads,
            model.manifest().head_dim,
        )?;
        let page_bytes = probe_backend.page_bytes();
        let private_state_bytes = model.new_sequence_state(1)?.device_bytes();
        let page_table_bytes = Sm12xPageTable::new(config.max_context_tokens)?.managed_bytes();
        let sampling_bytes = model
            .manifest()
            .vocab
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache sampling bytes",
                expected: "vocabulary byte count without overflow".to_string(),
                actual: model.manifest().vocab.to_string(),
            })?;
        let fixed_per_sequence = private_state_bytes
            .checked_add(page_table_bytes)
            .and_then(|bytes| bytes.checked_add(sampling_bytes))
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache fixed bytes",
                expected: "per-sequence byte count without overflow".to_string(),
                actual: format!(
                    "private={private_state_bytes} table={page_table_bytes} sampling={sampling_bytes}"
                ),
            })?;
        let fixed_capacity = fixed_per_sequence
            .checked_mul(config.max_active_sequences)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache fixed capacity",
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
                label: "Qwen3.6 active sequence-cache pages",
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
                    label: "Qwen3.6 active sequence-cache page bytes",
                    expected: "page byte count without overflow".to_string(),
                    actual: format!("pages={eager_pages} page_bytes={page_bytes}"),
                })?;
        let active_capacity = fixed_capacity
            .checked_add(active_page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 active sequence-cache capacity",
                expected: "managed byte count without overflow".to_string(),
                actual: format!("fixed={fixed_capacity} pages={active_page_bytes}"),
            })?;
        let retained_bytes = cache_config.max_retained_bytes;
        let snapshot_capacity = retained_bytes / 4;
        let managed_bytes =
            active_capacity
                .checked_add(retained_bytes)
                .ok_or_else(|| Error::Shape {
                    label: "Qwen3.6 sequence-cache capacity",
                    expected: "active and retained byte count without overflow".to_string(),
                    actual: format!("active={active_capacity} retained={retained_bytes}"),
                })?;
        let page_slots = eager_pages
            .checked_add(retained_bytes.saturating_sub(snapshot_capacity) / page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 sequence-cache page slots",
                expected: "page count without overflow".to_string(),
                actual: format!("active={eager_pages} retained={retained_bytes}"),
            })?;
        if page_slots == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 sequence-cache capacity",
                expected: format!(
                    "budget greater than fixed active capacity {fixed_capacity}, snapshot reserve {snapshot_capacity}, and one {page_bytes}-byte page"
                ),
                actual: managed_bytes.to_string(),
            });
        }
        let backend = Sm12xPageBackend::new(
            model
                .manifest()
                .layer_kinds
                .iter()
                .map(|kind| *kind == crate::qwen3::infer::QwenLayerKind::FullAttention),
            page_slots,
            model.manifest().kv_heads,
            model.manifest().head_dim,
        )?;
        let sequence_cache = Qwen36SequenceCache::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: managed_bytes,
                max_snapshot_bytes: snapshot_capacity,
                max_prefix_entries: (retained_bytes == 0).then_some(0),
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(|error| Error::Format {
            label: "Qwen3.6 sequence-cache configuration",
            detail: error.to_string(),
        })?;
        Ok(Self {
            model,
            config,
            decode_workspaces,
            prefill_workspace,
            requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            decoding: VecDeque::new(),
            sequence_cache,
            cache_stream: CudaStream::new_blocking()?,
            spec_workspace: None,
            mtp_workspace: None,
            mtp_hidden_scratch: None,
            dflash2_workspace: None,
            next_id: 0,
        })
    }

    /// Adds an already-tokenized request to the CPU-only waiting queue.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: RequestConfig,
    ) -> Result<Qwen36RequestId> {
        config.validate()?;
        if prompt_tokens.is_empty() {
            return Err(Error::Format {
                label: "Qwen3.6 scheduler prompt",
                detail: "prompt must contain at least one token".to_string(),
            });
        }
        if let Some(&token) = prompt_tokens
            .iter()
            .find(|&&token| token as usize >= self.model.manifest().vocab)
        {
            return Err(Error::Shape {
                label: "Qwen3.6 scheduler prompt token",
                expected: format!("token < {}", self.model.manifest().vocab),
                actual: token.to_string(),
            });
        }
        let max_tokens = prompt_tokens
            .len()
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 scheduler request capacity",
                expected: "prompt + completion length without overflow".to_string(),
                actual: format!("{} + {}", prompt_tokens.len(), config.max_new_tokens),
            })?;
        if max_tokens > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Qwen3.6 scheduler request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: max_tokens.to_string(),
            });
        }
        let id = Qwen36RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduler request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let lifecycle = if config.max_new_tokens == 0 {
            RequestState::Finished
        } else {
            RequestState::Waiting
        };
        let finish_reason = (config.max_new_tokens == 0).then_some(RequestFinishReason::Length);
        let sampler = Sampler::new(config.sampling)?;
        let history = TokenHistory::from_tokens(prompt_tokens.iter().copied());
        let prefix_target = retained_prompt_prefix_tokens(prompt_tokens.len());
        self.requests.insert(
            id,
            Box::new(Qwen36Request {
                id,
                lifecycle,
                config,
                prompt_tokens,
                prompt_position: 0,
                prefix_target,
                prefix_retained: false,
                sequence: None,
                device_token_counts: None,
                sequence_device_bytes: 0,
                sampler,
                history,
                last_token: None,
                generated_tokens: Vec::new(),
                finish_reason,
                mtp_state: None,
                dflash2_state: None,
                spec_frontier: None,
                spec_ready: false,
                spec_started: false,
            }),
        );
        if lifecycle == RequestState::Waiting {
            self.waiting.push_back(id);
        }
        Ok(id)
    }

    /// Runs one decode-first scheduling iteration followed by bounded prefill.
    pub fn tick(&mut self) -> Result<Qwen36SchedulerTick> {
        self.tick_with_lifecycle(&mut |_| {})
    }

    /// Runs one scheduler iteration and reports admission and prefill events
    /// when they occur.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen36RequestId, Qwen36AdmissionProgress>,
        ),
    ) -> Result<Qwen36SchedulerTick> {
        let tick_started = Instant::now();
        let mut tick = Qwen36SchedulerTick::default();
        self.admit_waiting(&mut tick, tick_started, on_lifecycle)?;
        self.run_decode_phase(&mut tick)?;
        self.run_prefill_phase(&mut tick, on_lifecycle)?;
        tick.active_sequences = self.active_sequence_count();
        Ok(tick)
    }

    fn admit_waiting(
        &mut self,
        tick: &mut Qwen36SchedulerTick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen36RequestId, Qwen36AdmissionProgress>,
        ),
    ) -> Result<()> {
        let model = self.model;
        while self.active_sequence_count() < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let prefix = self
                .sequence_cache
                .lookup_prefix(&self.requests[&id].prompt_tokens);
            let request = self
                .requests
                .get_mut(&id)
                .expect("waiting request is retained");
            let mut sequence = model.new_sequence_state(request.max_tokens().max(1))?;
            let mut page_table = Sm12xPageTable::new(request.max_tokens().max(1))?;
            let device_token_counts = if request.config.sampling.supports_gpu_sampling()
                && request.config.sampling.uses_history_penalties()
            {
                Some(DeviceBuffer::from_host(
                    &request.history.dense_counts(self.model.manifest().vocab),
                )?)
            } else {
                None
            };
            let private_state_bytes = sequence.device_bytes()
                + device_token_counts
                    .as_ref()
                    .map_or(0, DeviceBuffer::device_bytes);
            let page_table_bytes = page_table.managed_bytes();
            let outcome = self
                .sequence_cache
                .admit(
                    prefix,
                    AdmissionRequest {
                        max_position: request.max_tokens().max(1),
                        private_state_bytes,
                        page_table_bytes,
                        allow_emergency: false,
                    },
                    &mut Sm12xCacheContext {
                        stream: &self.cache_stream,
                        page_table: &mut page_table,
                    },
                    |snapshot, position| {
                        if let Some(snapshot) = snapshot {
                            model.restore_sequence_snapshot(snapshot, &mut sequence)?;
                        } else if position != 0 {
                            return Err(Error::Format {
                                label: "Qwen3.6 sequence-cache restore",
                                detail: "nonzero prefix has no recurrent snapshot".to_string(),
                            });
                        }
                        Ok(())
                    },
                )
                .map_err(sequence_cache_error)?;
            let cache_sequence = match outcome {
                AdmissionOutcome::Admitted(id) => id,
                AdmissionOutcome::WouldBlock => {
                    self.waiting.push_front(id);
                    break;
                }
            };
            self.cache_stream.synchronize()?;
            let cached_prompt_tokens = sequence.position();
            let sequence = Qwen36Sequence::from_admission(cache_sequence, page_table, sequence);
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence_device_bytes = sequence.device_bytes()
                + device_token_counts
                    .as_ref()
                    .map_or(0, DeviceBuffer::device_bytes);
            request.sequence = Some(Box::new(sequence));
            request.device_token_counts = device_token_counts;
            if (self.model.dflash2_enabled() || self.model.mtp_weights().is_some())
                && self.config.speculative_drafts > 0
                && request.sampler.config().uses_fast_argmax()
                && cached_prompt_tokens == 0
            {
                let speculative_state = (|| {
                    let dflash2_state = self
                        .model
                        .dflash2_enabled()
                        .then(|| self.model.new_dflash2_sequence_state())
                        .transpose()?;
                    let mtp_state = if dflash2_state.is_none() {
                        self.model
                            .new_mtp_sequence_state(request.max_tokens())
                            .map(Some)?
                    } else {
                        None
                    };
                    Ok::<_, Error>((
                        dflash2_state,
                        mtp_state,
                        Qwen36SpeculativeFrontier {
                            token: 0,
                            logit: 0.0,
                            prev_hidden: DeviceBuffer::zeroed(self.model.manifest().hidden)?,
                        },
                    ))
                })();
                match speculative_state {
                    Ok((dflash2_state, mtp_state, frontier)) => {
                        request.dflash2_state = dflash2_state;
                        request.mtp_state = mtp_state;
                        request.spec_frontier = Some(frontier);
                    }
                    Err(error) => warn!(
                        request = request.id.get(),
                        %error,
                        "continuing without Qwen3.8 speculation after state allocation failed"
                    ),
                }
            }
            request.lifecycle = RequestState::Prefilling;
            self.prefilling.push_back(id);
            let progress = Qwen36AdmissionProgress {
                request_id: id,
                sequence_device_bytes: request.sequence_device_bytes,
                cached_prompt_tokens,
                admitted_after_tick_start: tick_started.elapsed(),
            };
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn run_decode_phase(&mut self, tick: &mut Qwen36SchedulerTick) -> Result<()> {
        let mut selected = Vec::with_capacity(self.config.decode_capacity);
        while selected.len() < self.config.decode_capacity {
            let Some(id) = self.decoding.pop_front() else {
                break;
            };
            selected.push(
                self.requests
                    .remove(&id)
                    .expect("decoding request is retained"),
            );
        }
        let prefill_scan = self.prefilling.len();
        for _ in 0..prefill_scan {
            let Some(id) = self.prefilling.pop_front() else {
                break;
            };
            if selected.len() < self.config.decode_capacity
                && self.requests[&id].remaining_prompt_tokens() == 1
            {
                selected.push(
                    self.requests
                        .remove(&id)
                        .expect("prefilling request is retained"),
                );
            } else {
                self.prefilling.push_back(id);
            }
        }
        if selected.is_empty() {
            return Ok(());
        }
        tick.scheduled
            .extend(selected.iter().map(|request| request.id));
        let samples_per_request = if selected.len() == 1 && self.spec_eligible(&selected[0]) {
            let request_id = selected[0].id;
            match self.execute_speculative(&mut selected[0]) {
                Ok((samples, accepted_drafts)) => {
                    tick.speculative.push(Qwen38SpeculativeProgress {
                        request_id,
                        cycles: 1,
                        accepted_drafts,
                    });
                    vec![samples]
                }
                Err(error) => {
                    self.requeue_selected(selected);
                    return Err(error);
                }
            }
        } else {
            match self.execute_decode(&mut selected) {
                Ok(samples) => samples.into_iter().map(|sample| vec![sample]).collect(),
                Err(error) => {
                    self.requeue_selected(selected);
                    return Err(error);
                }
            }
        };
        for (mut request, samples) in selected.into_iter().zip(samples_per_request) {
            for sample in samples {
                let token = request.apply_sample(sample);
                tick.generated.push(token);
                if request.lifecycle == RequestState::Finished {
                    break;
                }
            }
            if request.lifecycle == RequestState::Finished {
                let sequence = request
                    .sequence
                    .take()
                    .expect("finished admitted request has a sequence");
                (*sequence).finish(&mut self.sequence_cache, &self.cache_stream)?;
                request.device_token_counts.take();
                tick.finished.push(request.id);
            } else {
                self.decoding.push_back(request.id);
            }
            self.requests.insert(request.id, request);
        }
        Ok(())
    }

    #[allow(clippy::vec_box)]
    fn requeue_selected(&mut self, selected: Vec<Box<Qwen36Request>>) {
        for request in selected.into_iter().rev() {
            let queue = if request.lifecycle == RequestState::Decoding {
                &mut self.decoding
            } else {
                &mut self.prefilling
            };
            queue.push_front(request.id);
            self.requests.insert(request.id, request);
        }
    }

    fn spec_eligible(&self, request: &Qwen36Request) -> bool {
        (request.dflash2_state.is_some() || self.model.mtp_weights().is_some())
            && self.config.speculative_drafts > 0
            && request.spec_ready
            && request.spec_frontier.is_some()
            && (request.dflash2_state.is_some() || request.mtp_state.is_some())
            && request.sampler.config().uses_fast_argmax()
            && request.active_speculative_drafts(self.config.speculative_drafts) > 0
    }

    fn execute_speculative(
        &mut self,
        request: &mut Qwen36Request,
    ) -> Result<(Vec<SampledToken>, usize)> {
        let drafts = request.active_speculative_drafts(self.config.speculative_drafts);
        if self.spec_workspace.is_none() {
            let mut workspace = if self.model.dflash2_enabled() {
                self.model.new_external_speculative_cycle_workspace(
                    self.config.speculative_drafts,
                    self.config.max_context_tokens,
                )?
            } else {
                self.model.new_speculative_cycle_workspace(
                    self.config.speculative_drafts,
                    self.config.max_context_tokens,
                )?
            };
            if self.model.dflash2_enabled() {
                self.model
                    .enable_dflash2_speculative_capture(&mut workspace)?;
            }
            self.spec_workspace = Some(workspace);
        }
        let workspace = self
            .spec_workspace
            .as_mut()
            .expect("speculative workspace was allocated");
        let mut frontier = request
            .spec_frontier
            .take()
            .expect("speculative request has a frontier");
        let sequence = request
            .sequence
            .as_deref_mut()
            .expect("speculative request has a sequence");
        let mut disable_dflash2 = false;
        let outcome = if let Some(dflash2_state) = request.dflash2_state.as_mut() {
            if self.dflash2_workspace.is_none() {
                self.dflash2_workspace = Some(self.model.new_dflash2_workspace()?);
            }
            let dflash2_workspace = self
                .dflash2_workspace
                .as_mut()
                .expect("DFlash2 workspace was allocated");
            (|| {
                let proposals = self.model.dflash2_propose(
                    dflash2_state,
                    frontier.token,
                    drafts,
                    dflash2_workspace,
                    workspace.stream(),
                )?;
                let outcome = self.model.verify_external_speculative_argmax(
                    workspace,
                    &proposals,
                    &mut frontier,
                    sequence,
                    &mut self.sequence_cache,
                )?;
                if let Err(error) = self.model.dflash2_append_speculative(
                    dflash2_state,
                    workspace,
                    0,
                    outcome.accepted_drafts + 1,
                    dflash2_workspace,
                ) {
                    warn!(
                        request = request.id.get(),
                        %error,
                        "disabling DFlash2 after accepted-context append failed"
                    );
                    disable_dflash2 = true;
                } else {
                    debug_assert_eq!(dflash2_state.position(), sequence.position());
                }
                Ok(outcome)
            })()
        } else {
            let mtp_state = request
                .mtp_state
                .as_mut()
                .expect("speculative request has MTP state");
            self.model.speculative_cycle_argmax(
                workspace,
                drafts,
                &mut frontier,
                sequence,
                mtp_state,
                &mut self.sequence_cache,
            )
        };
        request.spec_frontier = Some(frontier);
        if disable_dflash2 || outcome.is_err() && request.dflash2_state.is_some() {
            request.dflash2_state = None;
        }
        let outcome = outcome?;
        if !outcome.speculation_ready {
            warn!(
                request = request.id.get(),
                "continuing with ordinary decode after Qwen3.8 MTP catch-up failed"
            );
            request.mtp_state = None;
            request.spec_ready = true;
        }
        let skip = if request.spec_started { 0 } else { 1 };
        request.spec_started = true;
        let infer = metrics();
        infer.qwen38_speculative_cycles.add(1);
        infer
            .qwen38_speculative_accepted_drafts
            .add(outcome.accepted_drafts as isize);
        let samples = outcome
            .committed
            .iter()
            .skip(skip)
            .zip(outcome.committed_logits.iter().skip(skip))
            .map(|(&id, &logit)| SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            })
            .collect();
        Ok((samples, outcome.accepted_drafts))
    }

    fn execute_decode(&mut self, selected: &mut [Box<Qwen36Request>]) -> Result<Vec<SampledToken>> {
        let needs_host_logits = selected
            .iter()
            .any(|request| !request.sampler.config().supports_gpu_sampling());
        let all_fast_argmax = selected
            .iter()
            .all(|request| request.sampler.config().uses_fast_argmax());
        let tracked_frontier_rows = selected
            .iter()
            .enumerate()
            .filter(|(_, request)| {
                request.spec_frontier.is_some()
                    && (request.dflash2_state.is_some()
                        || request.mtp_state.is_some()
                        || request.spec_ready)
                    && request.sampler.config().uses_fast_argmax()
            })
            .map(|(row, _)| row)
            .collect::<Vec<_>>();
        let needs_mtp_catchup = tracked_frontier_rows
            .iter()
            .any(|&row| selected[row].mtp_state.is_some());
        if needs_mtp_catchup && self.mtp_workspace.is_none() {
            self.mtp_workspace = Some(
                self.model
                    .new_mtp_draft_workspace(self.config.max_context_tokens)?,
            );
        }
        let needs_dflash2_append = selected
            .iter()
            .any(|request| request.dflash2_state.is_some());
        if needs_dflash2_append && self.dflash2_workspace.is_none() {
            self.dflash2_workspace = Some(self.model.new_dflash2_workspace()?);
        }
        let workspace = self
            .decode_workspaces
            .iter_mut()
            .find(|workspace| workspace.capacity() >= selected.len())
            .expect("decode capacity classes cover the configured maximum");
        let mut rows = Vec::with_capacity(selected.len());
        let mut input_tokens = Vec::with_capacity(selected.len());
        for request in selected.iter_mut() {
            let token_id = request.decode_input_token()?;
            input_tokens.push(token_id);
            let sequence = request
                .sequence
                .as_deref_mut()
                .ok_or_else(|| Error::Format {
                    label: "Qwen3.6 scheduled decode",
                    detail: format!("request {} has no admitted sequence", request.id.get()),
                })?;
            rows.push(Qwen36DecodeRow { token_id, sequence });
        }
        let decoded = self
            .model
            .decode_batch(workspace, &mut rows, &mut self.sequence_cache);
        drop(rows);
        let samples = {
            let mut decoded = decoded?;
            let hidden = self.model.manifest().hidden;
            let mut samples = if all_fast_argmax {
                decoded
                    .top1()?
                    .into_iter()
                    .map(sampled_top1)
                    .collect::<Vec<_>>()
            } else if !needs_host_logits {
                let mut sampling_rows = selected
                    .iter_mut()
                    .map(|request| {
                        let config = request.sampler.config();
                        let draw = if config.temperature == 0.0 || config.top_k == 1 {
                            0.0
                        } else {
                            request.sampler.next_gpu_draw()
                        };
                        GpuSamplingRow {
                            temperature: config.temperature,
                            top_k: config.top_k,
                            top_p: config.top_p,
                            presence_penalty: config.presence_penalty,
                            frequency_penalty: config.frequency_penalty,
                            draw,
                            token_counts: request.device_token_counts.as_mut(),
                        }
                    })
                    .collect::<Vec<_>>();
                decoded
                    .sample_topk_topp(&mut sampling_rows)?
                    .into_iter()
                    .map(|sample| SampledToken {
                        id: sample.id,
                        logit: sample.logit,
                        adjusted_logit: sample.adjusted_logit,
                    })
                    .collect::<Vec<_>>()
            } else {
                let vocab = decoded.vocab();
                let logits = decoded.copy_logits()?;
                let samples = selected
                    .iter_mut()
                    .enumerate()
                    .map(|(row, request)| {
                        let row_logits = &logits[row * vocab..(row + 1) * vocab];
                        if request.sampler.config().uses_fast_argmax() {
                            return argmax_logits(row_logits);
                        }
                        request.sampler.sample(row_logits, &request.history)
                    })
                    .collect::<Result<Vec<_>>>()?;
                for (request, sample) in selected.iter_mut().zip(&samples) {
                    let Some(counts) = request.device_token_counts.as_mut() else {
                        continue;
                    };
                    let mut dense = request.history.dense_counts(vocab);
                    dense[sample.id as usize] += 1;
                    counts.copy_from_host(&dense)?;
                }
                samples
            };
            let mut copied_hidden = false;
            for &row in &tracked_frontier_rows {
                let request = &mut selected[row];
                let mut frontier = request
                    .spec_frontier
                    .take()
                    .expect("speculative request has a frontier");
                let mut mtp_state = request.mtp_state.take();
                let emit_frontier = request.spec_ready && request.spec_started;
                let emitted_frontier = SampledToken {
                    id: frontier.token,
                    logit: frontier.logit,
                    adjusted_logit: frontier.logit,
                };
                let canonical_sample = samples[row];
                samples[row] =
                    ordinary_decode_emission(emit_frontier, emitted_frontier, canonical_sample);
                if let Some(state) = mtp_state.as_mut() {
                    let catchup = self.model.mtp_append_kv(
                        state,
                        self.mtp_workspace
                            .as_mut()
                            .expect("speculative scheduler has an MTP workspace"),
                        input_tokens[row],
                        &frontier.prev_hidden,
                        decoded.stream(),
                    );
                    let catchup = catchup.and_then(|()| {
                        frontier.prev_hidden.copy_range_from_device_on_stream(
                            0,
                            decoded.hidden(),
                            row * hidden,
                            hidden,
                            decoded.stream(),
                        )
                    });
                    if let Err(error) = catchup {
                        warn!(
                            request = request.id.get(),
                            %error,
                            "disabling Qwen3.8 speculation after MTP catch-up failed"
                        );
                        mtp_state = None;
                    } else {
                        copied_hidden = true;
                    }
                }
                frontier.token = canonical_sample.id;
                frontier.logit = canonical_sample.logit;
                request.spec_frontier = Some(frontier);
                request.mtp_state = mtp_state;
                request.spec_ready = true;
            }
            if copied_hidden {
                decoded.stream().synchronize()?;
            }
            samples
        };
        if needs_dflash2_append {
            let dflash2_workspace = self
                .dflash2_workspace
                .as_mut()
                .expect("DFlash2 workspace was allocated");
            for (row, request) in selected.iter_mut().enumerate() {
                let Some(state) = request.dflash2_state.as_mut() else {
                    continue;
                };
                if let Err(error) =
                    self.model
                        .dflash2_append_decode(state, workspace, row, 1, dflash2_workspace)
                {
                    warn!(
                        request = request.id.get(),
                        %error,
                        "disabling DFlash2 after decode-context append failed"
                    );
                    request.dflash2_state = None;
                }
            }
            workspace.stream().synchronize()?;
        }
        Ok(samples)
    }

    fn retain_request_checkpoint(&mut self, request: &mut Qwen36Request) {
        if request.prefix_retained || request.prefix_target == 0 {
            return;
        }
        if self.sequence_cache.config().max_prefix_entries == Some(0) {
            request.prefix_retained = true;
            return;
        }
        let Some(sequence) = request.sequence.as_deref_mut() else {
            return;
        };
        if sequence.position() != request.prefix_target {
            return;
        }
        if self
            .sequence_cache
            .contains_prefix(&request.prompt_tokens, request.prefix_target)
        {
            request.prefix_retained = true;
            return;
        }
        let snapshot = match self.model.snapshot_sequence(&sequence.state) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                warn!(
                    request = request.id.get(),
                    %error,
                    "failed to copy recurrent prompt-prefix snapshot"
                );
                request.prefix_retained = true;
                return;
            }
        };
        if let Err(error) = self.sequence_cache.retain_prefix(
            sequence.cache_id,
            &request.prompt_tokens,
            snapshot,
            &mut Sm12xCacheContext {
                stream: &self.cache_stream,
                page_table: &mut sequence.page_table,
            },
        ) {
            warn!(
                request = request.id.get(),
                %error,
                "failed to retain shared prompt prefix"
            );
        }
        request.prefix_retained = true;
    }

    fn run_prefill_phase(
        &mut self,
        tick: &mut Qwen36SchedulerTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Qwen36RequestId, Qwen36AdmissionProgress>,
        ),
    ) -> Result<()> {
        let eligible = self
            .prefilling
            .iter()
            .filter(|id| self.requests[id].prefillable_tokens() > 0)
            .take(self.config.prefill_sequence_capacity)
            .count();
        if eligible == 0 {
            return Ok(());
        }
        let mut selected = Vec::with_capacity(eligible);
        let mut chunk_lengths = Vec::with_capacity(eligible);
        let mut token_budget = self.config.prefill_token_capacity;
        let mut slots_remaining = eligible;
        let scan = self.prefilling.len();
        for _ in 0..scan {
            if selected.len() == eligible || token_budget == 0 {
                break;
            }
            let id = self
                .prefilling
                .pop_front()
                .expect("prefilling scan has a request");
            let available = self.requests[&id].prefillable_tokens();
            if available == 0 {
                self.prefilling.push_back(id);
                continue;
            }
            let fair_share = token_budget.div_ceil(slots_remaining);
            let request = &self.requests[&id];
            let chunk = prefill_chunk_tokens(
                available,
                fair_share,
                request.prompt_position,
                request.prefix_target,
                request.prefix_retained,
            );
            token_budget -= chunk;
            slots_remaining -= 1;
            chunk_lengths.push(chunk);
            selected.push(
                self.requests
                    .remove(&id)
                    .expect("prefilling request is retained"),
            );
        }
        if selected.is_empty() {
            return Ok(());
        }
        tick.scheduled
            .extend(selected.iter().map(|request| request.id));
        let hidden = self.model.manifest().hidden;
        let mut needs_mtp_warmup = selected
            .iter()
            .any(|request| request.mtp_state.is_some() && request.spec_frontier.is_some());
        if needs_mtp_warmup {
            let allocation = (|| -> Result<()> {
                if self.mtp_workspace.is_none() {
                    self.mtp_workspace = Some(
                        self.model
                            .new_mtp_draft_workspace(self.config.max_context_tokens)?,
                    );
                }
                if self.mtp_hidden_scratch.is_none() {
                    self.mtp_hidden_scratch = Some(DeviceBuffer::zeroed(hidden)?);
                }
                Ok(())
            })();
            if let Err(error) = allocation {
                warn!(
                    %error,
                    "continuing without Qwen3.8 speculation after warmup allocation failed"
                );
                for request in &mut selected {
                    request.mtp_state = None;
                    request.spec_frontier = None;
                    request.spec_ready = false;
                }
                needs_mtp_warmup = false;
            }
        }
        let needs_dflash2_warmup = selected
            .iter()
            .any(|request| request.dflash2_state.is_some());
        if needs_dflash2_warmup && self.dflash2_workspace.is_none() {
            self.dflash2_workspace = Some(self.model.new_dflash2_workspace()?);
        }
        let prefill_ids = selected
            .iter()
            .map(|request| request.id)
            .collect::<Vec<_>>();
        let prefill_result = {
            let mut rows = Vec::with_capacity(selected.len());
            for (request, chunk) in selected.iter_mut().zip(chunk_lengths.iter().copied()) {
                let start = request.prompt_position;
                let end = start + chunk;
                let (prompt_tokens, sequence) = (&request.prompt_tokens, &mut request.sequence);
                rows.push(Qwen36PrefillRow {
                    token_ids: &prompt_tokens[start..end],
                    sequence: sequence
                        .as_deref_mut()
                        .expect("prefilling request has admitted sequence"),
                });
            }
            for id in prefill_ids {
                on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            }
            self.model.prefill_batch(
                &mut self.prefill_workspace,
                &mut rows,
                &mut self.sequence_cache,
            )
        };
        if let Err(error) = prefill_result {
            for request in selected.into_iter().rev() {
                self.prefilling.push_front(request.id);
                self.requests.insert(request.id, request);
            }
            return Err(error);
        }
        let mut row_offset = 0usize;
        for (request, &chunk) in selected.iter_mut().zip(&chunk_lengths) {
            let start = request.prompt_position;
            let end = start + chunk;
            if let Some(state) = request.dflash2_state.as_mut()
                && let Err(error) = self.model.dflash2_append_prefill(
                    state,
                    &self.prefill_workspace,
                    row_offset,
                    chunk,
                    self.dflash2_workspace
                        .as_mut()
                        .expect("DFlash2 workspace was allocated"),
                )
            {
                warn!(
                    request = request.id.get(),
                    %error,
                    "disabling DFlash2 after prompt-context append failed"
                );
                request.dflash2_state = None;
                request.spec_frontier = None;
                request.spec_ready = false;
            }
            if let (Some(mtp_state), Some(frontier)) =
                (&mut request.mtp_state, &mut request.spec_frontier)
            {
                let mtp_workspace = self
                    .mtp_workspace
                    .as_mut()
                    .expect("speculative scheduler has an MTP workspace");
                let mtp_hidden_scratch = self
                    .mtp_hidden_scratch
                    .as_mut()
                    .expect("speculative scheduler has MTP hidden scratch");
                let warmup = self.model.mtp_warmup_kv(
                    mtp_state,
                    mtp_workspace,
                    mtp_hidden_scratch,
                    &request.prompt_tokens[start..end],
                    self.prefill_workspace.prompt_hidden(),
                    row_offset * hidden,
                    &frontier.prev_hidden,
                    self.prefill_workspace.stream(),
                );
                let warmup = warmup.and_then(|()| {
                    frontier.prev_hidden.copy_range_from_device_on_stream(
                        0,
                        self.prefill_workspace.prompt_hidden(),
                        (row_offset + chunk - 1) * hidden,
                        hidden,
                        self.prefill_workspace.stream(),
                    )
                });
                if let Err(error) = warmup {
                    warn!(
                        request = request.id.get(),
                        %error,
                        "disabling Qwen3.8 speculation after MTP prompt warmup failed"
                    );
                    request.mtp_state = None;
                    request.spec_frontier = None;
                    request.spec_ready = false;
                }
            }
            row_offset += chunk;
        }
        if needs_mtp_warmup || needs_dflash2_warmup {
            self.prefill_workspace.stream().synchronize()?;
        }
        for (mut request, chunk) in selected.into_iter().zip(chunk_lengths) {
            request.prompt_position += chunk;
            self.retain_request_checkpoint(&mut request);
            tick.prefilled.push(Qwen36PrefillProgress {
                request_id: request.id,
                tokens: chunk,
                prompt_position: request.prompt_position,
            });
            self.prefilling.push_back(request.id);
            self.requests.insert(request.id, request);
        }
        Ok(())
    }

    /// Cancels waiting or active work between model submissions.
    pub fn cancel_request(&mut self, id: Qwen36RequestId) -> Qwen36CancelOutcome {
        let Some(request) = self.requests.get(&id) else {
            return Qwen36CancelOutcome::NotFound;
        };
        if request.lifecycle == RequestState::Finished {
            return Qwen36CancelOutcome::AlreadyFinished;
        }
        self.waiting.retain(|queued| *queued != id);
        self.prefilling.retain(|queued| *queued != id);
        self.decoding.retain(|queued| *queued != id);
        let mut request = self
            .requests
            .remove(&id)
            .expect("cancellation target remains retained");
        if let Some(sequence) = request.sequence.take()
            && let Err(error) = (*sequence).finish(&mut self.sequence_cache, &self.cache_stream)
        {
            warn!(request = id.get(), %error, "failed to release cancelled sequence cache state");
        }
        Qwen36CancelOutcome::Cancelled(Qwen36CancelledRequest {
            id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            released_sequence_device_bytes: request.sequence_device_bytes,
        })
    }

    /// Returns the configured scheduler limits.
    pub fn config(&self) -> SchedulerConfig {
        self.config
    }

    /// Returns the maximum rows in one decode batch.
    pub fn capacity(&self) -> usize {
        self.config.decode_capacity
    }

    /// Returns exact shared prefill and decode workspace device bytes.
    pub fn workspace_device_bytes(&self) -> usize {
        self.decode_workspaces
            .iter()
            .map(Qwen36DecodeBatchWorkspace::device_bytes)
            .sum::<usize>()
            + self.prefill_workspace.device_bytes()
            + self
                .spec_workspace
                .as_ref()
                .map_or(0, Qwen36SpeculativeCycleWorkspace::device_bytes)
            + self
                .mtp_workspace
                .as_ref()
                .map_or(0, Qwen36MtpDraftWorkspace::device_bytes)
            + self
                .mtp_hidden_scratch
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes)
    }

    /// Returns exact logical ownership and reservation state for shared KV.
    pub fn sequence_cache_stats(&self) -> CacheStats {
        self.sequence_cache.stats()
    }

    /// Returns bytes physically preallocated in the per-layer CUDA page slabs.
    pub fn sequence_cache_pool_device_bytes(&self) -> usize {
        self.sequence_cache.backend().device_bytes()
    }

    /// Returns the number of requests retained by the scheduler.
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns the number of CPU-only requests awaiting admission.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Returns the number of admitted prefill and decode requests.
    pub fn active_sequence_count(&self) -> usize {
        self.prefilling.len() + self.decoding.len()
    }

    /// Returns the number of admitted requests eligible for model work.
    pub fn runnable_count(&self) -> usize {
        self.active_sequence_count()
    }

    /// Returns a request's current lifecycle state.
    pub fn request_state(&self, id: Qwen36RequestId) -> Option<RequestState> {
        self.requests.get(&id).map(|request| request.lifecycle)
    }

    /// Returns generated completion tokens retained for a request.
    pub fn generated_tokens(&self, id: Qwen36RequestId) -> Option<&[Qwen36ScheduledToken]> {
        self.requests
            .get(&id)
            .map(|request| request.generated_tokens.as_slice())
    }

    /// Returns exact device bytes owned by an admitted request's sequence state.
    pub fn request_device_bytes(&self, id: Qwen36RequestId) -> Option<usize> {
        self.requests.get(&id).map(|request| {
            request
                .sequence
                .as_ref()
                .map_or(0, |_| request.sequence_device_bytes)
        })
    }

    /// Removes and returns a finished request.
    pub fn remove_finished(&mut self, id: Qwen36RequestId) -> Option<Qwen36FinishedRequest> {
        if self.request_state(id) != Some(RequestState::Finished) {
            return None;
        }
        let request = self.requests.remove(&id)?;
        Some(Qwen36FinishedRequest {
            id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            finish_reason: request
                .finish_reason
                .expect("finished request has a completion reason"),
            released_sequence_device_bytes: request.sequence_device_bytes,
        })
    }
}

fn decode_capacity_classes(max_capacity: usize) -> Vec<usize> {
    let mut classes = Vec::new();
    let mut capacity = 1;
    while capacity < max_capacity {
        classes.push(capacity);
        capacity = capacity.saturating_mul(2).min(max_capacity);
    }
    classes.push(max_capacity);
    classes
}

fn speculative_draft_count(
    configured: usize,
    max_new_tokens: usize,
    generated_tokens: usize,
    spec_started: bool,
) -> usize {
    let remaining = max_new_tokens.saturating_sub(generated_tokens);
    let draft_budget = if spec_started {
        remaining.saturating_sub(1)
    } else {
        remaining
    };
    configured.min(draft_budget)
}

fn ordinary_decode_emission(
    emit_frontier: bool,
    frontier: SampledToken,
    target_sample: SampledToken,
) -> SampledToken {
    if emit_frontier {
        frontier
    } else {
        target_sample
    }
}

fn sampled_top1(token: Qwen36NextToken) -> SampledToken {
    SampledToken {
        id: token.id,
        logit: token.value,
        adjusted_logit: token.value,
    }
}

fn argmax_logits(logits: &[f32]) -> Result<SampledToken> {
    let (id, logit) = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .ok_or_else(|| Error::Format {
            label: "Qwen3.6 scheduler logits",
            detail: "no finite logits".to_string(),
        })?;
    Ok(SampledToken {
        id: id as u32,
        logit,
        adjusted_logit: logit,
    })
}

fn sequence_cache_error(error: CacheError<Error>) -> Error {
    Error::Format {
        label: "Qwen3.6 sequence cache",
        detail: error.to_string(),
    }
}

fn prefill_chunk_tokens(
    available: usize,
    fair_share: usize,
    prompt_position: usize,
    checkpoint_target: usize,
    checkpointed: bool,
) -> usize {
    let chunk = available.min(fair_share);
    if !checkpointed
        && prompt_position < checkpoint_target
        && prompt_position + chunk > checkpoint_target
    {
        checkpoint_target - prompt_position
    } else {
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_SPECULATIVE_DRAFTS, Qwen36CancelOutcome, Qwen36Scheduler, RequestConfig,
        RequestFinishReason, RequestState, SchedulerConfig, argmax_logits, decode_capacity_classes,
        ordinary_decode_emission, prefill_chunk_tokens, speculative_draft_count,
    };
    use crate::qwen3::qwen36::{Qwen36DecodeBatchWorkspace, Qwen36TextModel};
    use crate::runtime::cache_config::SequenceCacheConfig;
    use crate::runtime::sampling::{SampledToken, SamplingConfig};
    use std::path::PathBuf;

    #[test]
    fn argmax_prefers_the_lowest_token_on_a_tie() {
        let token = argmax_logits(&[1.0, 3.0, 3.0, f32::NAN]).expect("argmax");
        assert_eq!(token.id, 1);
        assert_eq!(token.logit, 3.0);
    }

    #[test]
    fn lifecycle_finish_and_cancellation_are_distinct_public_states() {
        assert_ne!(RequestState::Waiting, RequestState::Prefilling);
        assert_ne!(RequestState::Prefilling, RequestState::Decoding);
        assert_ne!(RequestState::Decoding, RequestState::Finished);
        assert_ne!(RequestFinishReason::Eos, RequestFinishReason::Length);
        assert_eq!(Qwen36CancelOutcome::NotFound, Qwen36CancelOutcome::NotFound);
    }

    #[test]
    fn prefill_chunks_follow_compute_capacity_not_cache_page_boundaries() {
        assert_eq!(prefill_chunk_tokens(2_048, 2_048, 0, 0, true), 2_048);
        assert_eq!(prefill_chunk_tokens(2_048, 2_048, 64, 0, true), 2_048);
        assert_eq!(prefill_chunk_tokens(2_048, 2_048, 64, 384, false), 320);
    }

    #[test]
    fn speculative_draft_budget_reserves_an_unemitted_frontier() {
        assert_eq!(speculative_draft_count(2, 8, 0, false), 2);
        assert_eq!(speculative_draft_count(2, 8, 6, false), 2);
        assert_eq!(speculative_draft_count(2, 8, 6, true), 1);
        assert_eq!(speculative_draft_count(2, 8, 7, true), 0);
        assert_eq!(speculative_draft_count(2, 8, 8, true), 0);
    }

    #[test]
    fn ordinary_decode_emits_pending_frontier_after_speculation() {
        let frontier = SampledToken {
            id: 11,
            logit: 1.5,
            adjusted_logit: 1.5,
        };
        let target_sample = SampledToken {
            id: 12,
            logit: 2.5,
            adjusted_logit: 2.5,
        };
        assert_eq!(
            ordinary_decode_emission(false, frontier, target_sample),
            target_sample
        );
        assert_eq!(
            ordinary_decode_emission(true, frontier, target_sample),
            frontier
        );
    }

    #[test]
    fn scheduler_rejects_unbounded_speculative_workspaces() {
        let config = SchedulerConfig {
            speculative_drafts: MAX_SPECULATIVE_DRAFTS + 1,
            ..SchedulerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 checkpoint"]
    fn qwen38_scheduler_speculation_matches_greedy_decode_through_tail() {
        let model_dir = std::env::var_os("QWEN38_MODEL")
            .map(PathBuf::from)
            .expect("set QWEN38_MODEL to a Qwen3.8 checkpoint");
        let mut model = Qwen36TextModel::open(model_dir).expect("load Qwen3.8 model");
        if let Some(dflash2_dir) = std::env::var_os("QWEN38_DFLASH2").map(PathBuf::from) {
            model
                .enable_dflash2(dflash2_dir)
                .expect("load Qwen3.8 DFlash2 companion");
        }
        let run = |speculative_drafts| {
            let mut scheduler = Qwen36Scheduler::new_with_cache_config(
                &model,
                SchedulerConfig {
                    decode_capacity: 1,
                    prefill_sequence_capacity: 1,
                    prefill_token_capacity: 16,
                    max_active_sequences: 1,
                    max_context_tokens: crate::nvfp4::SM12X_KV_PAGE_TOKENS,
                    speculative_drafts,
                },
                SequenceCacheConfig {
                    max_retained_bytes: 0,
                },
            )
            .expect("scheduler");
            let request = scheduler
                .add_request(
                    vec![1, 2, 3, 4],
                    RequestConfig {
                        sampling: SamplingConfig {
                            temperature: 0.0,
                            ..SamplingConfig::default()
                        },
                        max_new_tokens: 7,
                        ..RequestConfig::default()
                    },
                )
                .expect("request");
            for _ in 0..32 {
                if scheduler.request_state(request) == Some(RequestState::Finished) {
                    break;
                }
                scheduler.tick().expect("scheduler tick");
            }
            scheduler
                .remove_finished(request)
                .expect("request finished")
                .generated_tokens
                .into_iter()
                .map(|token| token.id)
                .collect::<Vec<_>>()
        };

        assert_eq!(run(2), run(0));
    }

    #[test]
    fn decode_capacity_classes_bound_padding_and_include_the_maximum() {
        assert_eq!(decode_capacity_classes(1), [1]);
        assert_eq!(decode_capacity_classes(8), [1, 2, 4, 8]);
        assert_eq!(decode_capacity_classes(6), [1, 2, 4, 6]);
        for max_capacity in 1..=64 {
            let classes = decode_capacity_classes(max_capacity);
            assert_eq!(classes.last(), Some(&max_capacity));
            assert!(classes.windows(2).all(|pair| pair[0] < pair[1]));
            for active_rows in 1..=max_capacity {
                let selected = classes
                    .iter()
                    .copied()
                    .find(|capacity| *capacity >= active_rows)
                    .expect("maximum capacity covers every active row count");
                assert!(selected < active_rows.saturating_mul(2));
            }
        }
    }

    #[test]
    #[ignore = "loads the full local Qwen3.6 checkpoint"]
    fn real_model_prefills_multiple_cache_pages_in_one_operation() {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36TextModel::open(model_dir).expect("load Qwen3.6 model");
        let mut scheduler = Qwen36Scheduler::new(
            &model,
            SchedulerConfig {
                decode_capacity: 1,
                prefill_sequence_capacity: 1,
                prefill_token_capacity: 384,
                max_active_sequences: 1,
                max_context_tokens: 386,
                speculative_drafts: 0,
            },
        )
        .expect("scheduler");
        scheduler
            .add_request(
                vec![9707; 385],
                RequestConfig {
                    sampling: SamplingConfig {
                        temperature: 0.0,
                        ..SamplingConfig::default()
                    },
                    max_new_tokens: 1,
                    ..RequestConfig::default()
                },
            )
            .expect("request");

        let tick = scheduler.tick().expect("multi-page prefill tick");
        assert_eq!(tick.prefilled.len(), 1);
        assert_eq!(tick.prefilled[0].tokens, 384);
    }

    #[test]
    #[ignore = "loads the full local Qwen3.6 checkpoint"]
    fn real_model_prefills_admits_rotates_and_cancels_without_moving_states() {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36TextModel::open(model_dir).expect("load Qwen3.6 model");
        let mut scheduler = Qwen36Scheduler::new(
            &model,
            SchedulerConfig {
                decode_capacity: 2,
                prefill_sequence_capacity: 2,
                prefill_token_capacity: 4,
                max_active_sequences: 2,
                max_context_tokens: 8,
                speculative_drafts: 0,
            },
        )
        .expect("scheduler");
        assert_eq!(
            scheduler
                .decode_workspaces
                .iter()
                .map(Qwen36DecodeBatchWorkspace::capacity)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        let config = |max_new_tokens| RequestConfig {
            sampling: SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            },
            max_new_tokens,
            ..RequestConfig::default()
        };
        let first = scheduler
            .add_request(vec![9707], config(2))
            .expect("first request");
        let second = scheduler
            .add_request(vec![3710, 9707, 3710], config(2))
            .expect("second request");
        let third = scheduler
            .add_request(vec![9707, 3710], config(1))
            .expect("third request");
        let cancelled_waiting = scheduler
            .add_request(vec![3710], config(1))
            .expect("waiting cancellation request");
        assert!(matches!(
            scheduler.cancel_request(cancelled_waiting),
            Qwen36CancelOutcome::Cancelled(_)
        ));
        assert_eq!(scheduler.request_device_bytes(first), Some(0));
        assert_eq!(scheduler.waiting_count(), 3);

        let tick = scheduler.tick().expect("first tick");
        assert_eq!(tick.generated.len(), 1);
        assert_eq!(tick.generated[0].request_id, first);
        assert_eq!(tick.prefilled.len(), 1);
        assert_eq!(tick.prefilled[0].request_id, second);
        assert_eq!(tick.prefilled[0].tokens, 2);
        assert_eq!(scheduler.waiting_count(), 1);
        assert!(scheduler.request_device_bytes(second).unwrap() > 0);

        let second_sequence = scheduler.requests[&second]
            .sequence
            .as_deref()
            .expect("second admitted state") as *const _;
        let tick = scheduler.tick().expect("second tick");
        assert_eq!(tick.generated.len(), 2);
        assert_eq!(tick.finished, [first]);
        assert_eq!(scheduler.request_device_bytes(first), Some(0));
        assert_eq!(
            scheduler.requests[&second]
                .sequence
                .as_deref()
                .expect("second retained state") as *const _,
            second_sequence
        );

        let tick = scheduler.tick().expect("third tick");
        assert!(tick.scheduled.contains(&third));
        let outcome = scheduler.cancel_request(third);
        assert!(matches!(outcome, Qwen36CancelOutcome::Cancelled(_)));
        assert_eq!(scheduler.request_state(third), None);
        assert_eq!(
            scheduler.cancel_request(third),
            Qwen36CancelOutcome::NotFound
        );

        while scheduler.request_state(second) != Some(RequestState::Finished) {
            scheduler.tick().expect("finish second");
        }
        assert_eq!(
            scheduler.cancel_request(second),
            Qwen36CancelOutcome::AlreadyFinished
        );
        let finished = scheduler.remove_finished(second).expect("remove second");
        assert_eq!(finished.generated_tokens.len(), 2);

        let cancelled_decode = scheduler
            .add_request(vec![9707], config(2))
            .expect("decode cancellation request");
        scheduler.tick().expect("start decode cancellation request");
        assert_eq!(
            scheduler.request_state(cancelled_decode),
            Some(RequestState::Decoding)
        );
        assert!(matches!(
            scheduler.cancel_request(cancelled_decode),
            Qwen36CancelOutcome::Cancelled(_)
        ));
        assert_eq!(scheduler.request_state(cancelled_decode), None);
    }

    #[test]
    #[ignore = "loads the full local Qwen3.6 checkpoint"]
    fn real_model_prefix_checkpoint_restores_into_concurrent_requests() {
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/qwen3.6-35b-a3-nvfp4")
            });
        let model = Qwen36TextModel::open(model_dir).expect("load Qwen3.6 model");
        let mut scheduler = Qwen36Scheduler::new(
            &model,
            SchedulerConfig {
                decode_capacity: 2,
                prefill_sequence_capacity: 2,
                prefill_token_capacity: 128,
                max_active_sequences: 2,
                max_context_tokens: 257,
                speculative_drafts: 0,
            },
        )
        .expect("scheduler");
        let config = RequestConfig {
            sampling: SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            },
            max_new_tokens: 1,
            ..RequestConfig::default()
        };
        let prompt = vec![9707; 256];

        let first = scheduler
            .add_request(prompt.clone(), config.clone())
            .expect("first request");
        while scheduler.request_state(first) != Some(RequestState::Finished) {
            scheduler.tick().expect("run first request");
        }
        let first_token = scheduler
            .remove_finished(first)
            .expect("first result")
            .generated_tokens[0]
            .id;

        let second = scheduler
            .add_request(prompt.clone(), config.clone())
            .expect("second request");
        let third = scheduler
            .add_request(prompt, config)
            .expect("third request");
        let tick = scheduler.tick().expect("restore concurrent requests");
        assert_eq!(tick.admitted.len(), 2);
        assert!(
            tick.admitted
                .iter()
                .all(|admission| admission.cached_prompt_tokens == 128)
        );
        while scheduler.request_state(second) != Some(RequestState::Finished)
            || scheduler.request_state(third) != Some(RequestState::Finished)
        {
            scheduler.tick().expect("finish concurrent requests");
        }
        assert_eq!(
            scheduler.request_state(second),
            Some(RequestState::Finished)
        );
        assert_eq!(scheduler.request_state(third), Some(RequestState::Finished));
        let second_token = scheduler
            .remove_finished(second)
            .expect("second result")
            .generated_tokens[0]
            .id;
        let third_token = scheduler
            .remove_finished(third)
            .expect("third result")
            .generated_tokens[0]
            .id;
        assert_eq!(second_token, first_token);
        assert_eq!(third_token, first_token);
    }
}
