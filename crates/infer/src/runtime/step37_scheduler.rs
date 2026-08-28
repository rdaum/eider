//! Multi-session scheduling for the paged Step-3.7 runtime.

use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageBackend, Sm12xPageTable};
use crate::step37::{
    HEAD_DIM, KV_HEADS, Step37PrefillBatchWorkspace, Step37PrefillRow, Step37TextModel,
};
use crate::step37::{Step37Sequence, Step37SequenceCache, step37_cache_error};
use eider_cuda::{CudaStream, DeviceBuffer, Error, GpuSamplingRow, Result, SM12X_KV_PAGE_TOKENS};
use eider_runtime::cache::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use eider_runtime::sampling::{SampledToken, Sampler, TokenHistory};
use eider_runtime::scheduler::{
    RequestConfig, RequestFinishReason, RequestLifecycleEvent, RequestState, SchedulerConfig,
};
use seqcache::{AdmissionOutcome, AdmissionRequest, CacheConfig, PageBackend};
use std::collections::{BTreeMap, VecDeque};
use std::mem::size_of;
use std::time::{Duration, Instant};
use tracing::warn;

/// Stable scheduler identity for one Step request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Step37RequestId(u64);

impl Step37RequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(value: u64) -> Self {
        Self(value)
    }
}

/// One completion token produced by a Step scheduler tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step37ScheduledToken {
    pub request_id: Step37RequestId,
    pub id: u32,
    pub logit: f32,
    pub finish_reason: Option<RequestFinishReason>,
}

/// Prompt progress made for one Step request during a scheduler tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Step37PrefillProgress {
    pub request_id: Step37RequestId,
    pub tokens: usize,
    pub prompt_position: usize,
}

/// Persistent sequence state allocated for a newly admitted Step request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Step37AdmissionProgress {
    pub request_id: Step37RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time when admission completed.
    pub admitted_after_tick_start: Duration,
}

/// Observable result of one multi-session Step scheduler iteration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step37SchedulerTick {
    pub admitted: Vec<Step37AdmissionProgress>,
    pub scheduled: Vec<Step37RequestId>,
    pub prefilled: Vec<Step37PrefillProgress>,
    pub generated: Vec<Step37ScheduledToken>,
    pub finished: Vec<Step37RequestId>,
    pub active_sequences: usize,
}

/// Result removed after a Step request finishes.
#[derive(Clone, Debug, PartialEq)]
pub struct Step37FinishedRequest {
    pub id: Step37RequestId,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<Step37ScheduledToken>,
    pub finish_reason: RequestFinishReason,
    pub released_sequence_device_bytes: usize,
}

/// Request data returned when active or waiting Step work is cancelled.
#[derive(Clone, Debug, PartialEq)]
pub struct Step37CancelledRequest {
    pub id: Step37RequestId,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<Step37ScheduledToken>,
    pub released_sequence_device_bytes: usize,
}

/// Outcome of cancelling a Step request.
#[derive(Clone, Debug, PartialEq)]
pub enum Step37CancelOutcome {
    Cancelled(Step37CancelledRequest),
    AlreadyFinished,
    NotFound,
}

struct Step37Request {
    id: Step37RequestId,
    lifecycle: RequestState,
    config: RequestConfig,
    prompt_tokens: Vec<u32>,
    prompt_position: usize,
    prefix_target: usize,
    prefix_retained: bool,
    sequence: Option<Box<Step37Sequence>>,
    device_token_counts: Option<DeviceBuffer<u32>>,
    sequence_device_bytes: usize,
    sampler: Sampler,
    history: TokenHistory,
    last_token: Option<u32>,
    generated_tokens: Vec<Step37ScheduledToken>,
    finish_reason: Option<RequestFinishReason>,
}

impl Step37Request {
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
        self.last_token.ok_or_else(|| Error::Format {
            label: "Step-3.7 scheduled request",
            detail: format!("request {} has no decode input token", self.id.get()),
        })
    }

    fn apply_sample(&mut self, sampled: SampledToken) -> Step37ScheduledToken {
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
        let token = Step37ScheduledToken {
            request_id: self.id,
            id: sampled.id,
            logit: sampled.logit,
            finish_reason,
        };
        self.generated_tokens.push(token);
        token
    }
}

/// Decode-first scheduler sharing one paged expert cache across independent sequences.
pub struct Step37Scheduler {
    model: Step37TextModel,
    prefill_workspace: Step37PrefillBatchWorkspace,
    config: SchedulerConfig,
    requests: BTreeMap<Step37RequestId, Box<Step37Request>>,
    waiting: VecDeque<Step37RequestId>,
    prefilling: VecDeque<Step37RequestId>,
    decoding: VecDeque<Step37RequestId>,
    next_id: u64,
    sequence_cache: Step37SequenceCache,
    cache_stream: CudaStream,
}

impl Step37Scheduler {
    pub fn new(model: Step37TextModel, config: SchedulerConfig) -> Result<Self> {
        Self::new_with_cache_config(model, config, SequenceCacheConfig::default())
    }

    pub fn new_with_cache_config(
        model: Step37TextModel,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let prefill_workspace = model.new_prefill_batch_workspace(
            config.prefill_sequence_capacity,
            config.prefill_token_capacity,
            config.max_context_tokens,
        )?;
        let probe_backend = Sm12xPageBackend::new(
            std::iter::repeat_n(true, model.layer_count()),
            1,
            KV_HEADS,
            HEAD_DIM,
        )?;
        let page_bytes = probe_backend.page_bytes();
        let private_state_bytes = model
            .new_sequence_state(config.max_context_tokens)?
            .device_bytes();
        let page_table_bytes = Sm12xPageTable::new(config.max_context_tokens)?.managed_bytes();
        let sampling_bytes =
            model
                .vocab()
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| Error::Shape {
                    label: "Step-3.7 sequence-cache sampling bytes",
                    expected: "vocabulary byte count without overflow".to_string(),
                    actual: model.vocab().to_string(),
                })?;
        let fixed_per_sequence = private_state_bytes
            .checked_add(page_table_bytes)
            .and_then(|bytes| bytes.checked_add(sampling_bytes))
            .ok_or_else(|| Error::Shape {
                label: "Step-3.7 sequence-cache fixed bytes",
                expected: "per-sequence byte count without overflow".to_string(),
                actual: format!(
                    "private={private_state_bytes} table={page_table_bytes} sampling={sampling_bytes}"
                ),
            })?;
        let fixed_capacity = fixed_per_sequence
            .checked_mul(config.max_active_sequences)
            .ok_or_else(|| Error::Shape {
                label: "Step-3.7 sequence-cache fixed capacity",
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
                label: "Step-3.7 active sequence-cache pages",
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
                    label: "Step-3.7 active sequence-cache page bytes",
                    expected: "page byte count without overflow".to_string(),
                    actual: format!("pages={eager_pages} page_bytes={page_bytes}"),
                })?;
        let active_capacity = fixed_capacity
            .checked_add(active_page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Step-3.7 active sequence-cache capacity",
                expected: "managed byte count without overflow".to_string(),
                actual: format!("fixed={fixed_capacity} pages={active_page_bytes}"),
            })?;
        let retained_bytes = cache_config.max_retained_bytes;
        let managed_bytes =
            active_capacity
                .checked_add(retained_bytes)
                .ok_or_else(|| Error::Shape {
                    label: "Step-3.7 sequence-cache capacity",
                    expected: "active and retained byte count without overflow".to_string(),
                    actual: format!("active={active_capacity} retained={retained_bytes}"),
                })?;
        let page_slots = eager_pages
            .checked_add(retained_bytes / page_bytes)
            .ok_or_else(|| Error::Shape {
                label: "Step-3.7 sequence-cache page slots",
                expected: "page count without overflow".to_string(),
                actual: format!("active={eager_pages} retained={retained_bytes}"),
            })?;
        if page_slots == 0 {
            return Err(Error::Shape {
                label: "Step-3.7 sequence-cache capacity",
                expected: format!(
                    "budget greater than fixed active capacity {fixed_capacity} and one {page_bytes}-byte page"
                ),
                actual: managed_bytes.to_string(),
            });
        }
        let backend = Sm12xPageBackend::new(
            std::iter::repeat_n(true, model.layer_count()),
            page_slots,
            KV_HEADS,
            HEAD_DIM,
        )?;
        let sequence_cache = Step37SequenceCache::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: managed_bytes,
                max_snapshot_bytes: 0,
                max_prefix_entries: (retained_bytes == 0).then_some(0),
                emergency_bytes: 0,
            },
            backend,
        )
        .map_err(step37_cache_error)?;
        Ok(Self {
            model,
            prefill_workspace,
            config,
            requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            decoding: VecDeque::new(),
            next_id: 0,
            sequence_cache,
            cache_stream: CudaStream::new_blocking()?,
        })
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: RequestConfig,
    ) -> Result<Step37RequestId> {
        config.validate()?;
        if prompt_tokens.is_empty() {
            return Err(Error::Format {
                label: "Step-3.7 scheduler prompt",
                detail: "prompt must contain at least one token".to_string(),
            });
        }
        if let Some(&token) = prompt_tokens
            .iter()
            .find(|&&token| token as usize >= self.model.vocab())
        {
            return Err(Error::Shape {
                label: "Step-3.7 scheduler prompt token",
                expected: format!("token < {}", self.model.vocab()),
                actual: token.to_string(),
            });
        }
        let max_tokens = prompt_tokens
            .len()
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Step-3.7 scheduler request capacity",
                expected: "prompt + completion length without overflow".to_string(),
                actual: format!("{} + {}", prompt_tokens.len(), config.max_new_tokens),
            })?;
        if max_tokens > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Step-3.7 scheduler request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: max_tokens.to_string(),
            });
        }
        let id = Step37RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Step-3.7 scheduler request ID",
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
        let prefix_target =
            retained_prompt_prefix_tokens(prompt_tokens.len(), SM12X_KV_PAGE_TOKENS);
        self.requests.insert(
            id,
            Box::new(Step37Request {
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
            }),
        );
        if lifecycle == RequestState::Waiting {
            self.waiting.push_back(id);
        }
        Ok(id)
    }

    pub fn tick(&mut self) -> Result<Step37SchedulerTick> {
        self.tick_with_lifecycle(&mut |_| {})
    }

    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Step37RequestId, Step37AdmissionProgress>,
        ),
    ) -> Result<Step37SchedulerTick> {
        let tick_started = Instant::now();
        let mut tick = Step37SchedulerTick::default();
        self.admit_waiting(&mut tick, tick_started, on_lifecycle)?;
        self.run_decode_phase(&mut tick)?;
        self.run_prefill_phase(&mut tick, on_lifecycle)?;
        tick.active_sequences = self.active_sequence_count();
        Ok(tick)
    }

    fn admit_waiting(
        &mut self,
        tick: &mut Step37SchedulerTick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Step37RequestId, Step37AdmissionProgress>,
        ),
    ) -> Result<()> {
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
                .expect("waiting request retained");
            let mut state = self.model.new_sequence_state(request.max_tokens().max(1))?;
            let mut page_table = Sm12xPageTable::new(request.max_tokens().max(1))?;
            let device_token_counts = if request
                .config
                .sampling
                .supports_gpu_sampling(eider_cuda::GPU_SAMPLING_MAX_TOP_K)
                && request.config.sampling.uses_history_penalties()
            {
                Some(DeviceBuffer::from_host(
                    &request.history.dense_counts(self.model.vocab()),
                )?)
            } else {
                None
            };
            let private_state_bytes = state.device_bytes()
                + device_token_counts
                    .as_ref()
                    .map_or(0, DeviceBuffer::device_bytes);
            let outcome = self
                .sequence_cache
                .admit(
                    prefix,
                    AdmissionRequest {
                        max_position: request.max_tokens().max(1),
                        private_state_bytes,
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
                .map_err(step37_cache_error)?;
            let cache_id = match outcome {
                AdmissionOutcome::Admitted(id) => id,
                AdmissionOutcome::WouldBlock => {
                    self.waiting.push_front(id);
                    break;
                }
            };
            self.cache_stream.synchronize()?;
            let cached_prompt_tokens = state.len();
            let sequence = Step37Sequence::from_admission(cache_id, page_table, state);
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence_device_bytes = sequence.device_bytes()
                + device_token_counts
                    .as_ref()
                    .map_or(0, DeviceBuffer::device_bytes);
            request.sequence = Some(Box::new(sequence));
            request.device_token_counts = device_token_counts;
            request.lifecycle = RequestState::Prefilling;
            self.prefilling.push_back(id);
            let progress = Step37AdmissionProgress {
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

    fn run_decode_phase(&mut self, tick: &mut Step37SchedulerTick) -> Result<()> {
        let mut selected = Vec::with_capacity(self.config.decode_capacity);
        while selected.len() < self.config.decode_capacity {
            let Some(id) = self.decoding.pop_front() else {
                break;
            };
            selected.push(
                self.requests
                    .remove(&id)
                    .expect("decoding request retained"),
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
                        .expect("prefilling request retained"),
                );
            } else {
                self.prefilling.push_back(id);
            }
        }
        let mut selected = selected.into_iter();
        while let Some(mut request) = selected.next() {
            tick.scheduled.push(request.id);
            let result = self.execute_decode(&mut request);
            let sample = match result {
                Ok(sample) => sample,
                Err(error) => {
                    let mut restore = vec![request];
                    restore.extend(selected);
                    for request in restore.into_iter().rev() {
                        let queue = if request.lifecycle == RequestState::Decoding {
                            &mut self.decoding
                        } else {
                            &mut self.prefilling
                        };
                        queue.push_front(request.id);
                        self.requests.insert(request.id, request);
                    }
                    return Err(error);
                }
            };
            let token = request.apply_sample(sample);
            tick.generated.push(token);
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

    fn execute_decode(&mut self, request: &mut Step37Request) -> Result<SampledToken> {
        let token = request.decode_input_token()?;
        let state = request
            .sequence
            .as_deref_mut()
            .ok_or_else(|| Error::Format {
                label: "Step-3.7 scheduled decode",
                detail: format!(
                    "request {} has no admitted sequence state",
                    request.id.get()
                ),
            })?;
        if request
            .sampler
            .config()
            .supports_gpu_sampling(eider_cuda::GPU_SAMPLING_MAX_TOP_K)
        {
            let config = request.sampler.config();
            let draw = if config.temperature == 0.0 || config.top_k == 1 {
                0.0
            } else {
                request.sampler.next_gpu_draw()
            };
            let mut row = GpuSamplingRow {
                temperature: config.temperature,
                top_k: config.top_k,
                top_p: config.top_p,
                presence_penalty: config.presence_penalty,
                frequency_penalty: config.frequency_penalty,
                draw,
                token_counts: request.device_token_counts.as_mut(),
            };
            let sampled =
                self.model
                    .sample_one(state, token, &mut row, &mut self.sequence_cache)?;
            return Ok(SampledToken {
                id: sampled.id,
                logit: sampled.logit,
                adjusted_logit: sampled.adjusted_logit,
            });
        }
        let logits = self
            .model
            .logits_one(state, token, &mut self.sequence_cache)?;
        Ok(request.sampler.sample(&logits, &request.history)?)
    }

    fn retain_request_checkpoint(&mut self, request: &mut Step37Request) {
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
        if let Err(error) = self.sequence_cache.retain_prefix(
            sequence.cache_id,
            &request.prompt_tokens,
            (),
            &mut Sm12xCacheContext {
                stream: &self.cache_stream,
                page_table: &mut sequence.page_table,
            },
        ) {
            warn!(
                request = request.id.get(),
                %error,
                "failed to retain shared Step prompt prefix"
            );
        }
        request.prefix_retained = true;
    }

    fn run_prefill_phase(
        &mut self,
        tick: &mut Step37SchedulerTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Step37RequestId, Step37AdmissionProgress>,
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
        let mut token_budget = self.config.prefill_token_capacity;
        let mut slots_remaining = eligible;
        let scan = self.prefilling.len();
        let mut selected = Vec::with_capacity(eligible);
        for _ in 0..scan {
            if selected.len() == eligible || token_budget == 0 {
                break;
            }
            let id = self
                .prefilling
                .pop_front()
                .expect("prefilling request retained");
            let available = self.requests[&id].prefillable_tokens();
            if available == 0 {
                self.prefilling.push_back(id);
                continue;
            }
            let mut chunk = available.min(token_budget.div_ceil(slots_remaining));
            let request = &self.requests[&id];
            if !request.prefix_retained
                && request.prompt_position < request.prefix_target
                && request.prompt_position + chunk > request.prefix_target
            {
                chunk = request.prefix_target - request.prompt_position;
            }
            token_budget -= chunk;
            slots_remaining -= 1;
            selected.push((id, chunk));
        }
        if selected.is_empty() {
            return Ok(());
        }
        let mut requests = selected
            .iter()
            .map(|(id, _)| {
                self.requests
                    .remove(id)
                    .expect("prefilling request retained")
            })
            .collect::<Vec<_>>();
        tick.scheduled.extend(selected.iter().map(|(id, _)| *id));
        let result = {
            let mut rows = requests
                .iter_mut()
                .zip(selected.iter().map(|(_, chunk)| *chunk))
                .map(|(request, chunk)| {
                    let start = request.prompt_position;
                    let end = start + chunk;
                    Step37PrefillRow {
                        token_ids: &request.prompt_tokens[start..end],
                        sequence: request
                            .sequence
                            .as_deref_mut()
                            .expect("prefilling request has admitted sequence"),
                    }
                })
                .collect::<Vec<_>>();
            for &(id, _) in &selected {
                on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            }
            self.model.prefill_batch(
                &mut self.prefill_workspace,
                &mut rows,
                &mut self.sequence_cache,
            )
        };
        if let Err(error) = result {
            for request in requests.into_iter().rev() {
                self.prefilling.push_front(request.id);
                self.requests.insert(request.id, request);
            }
            return Err(error);
        }
        for (mut request, (_, chunk)) in requests.into_iter().zip(selected) {
            request.prompt_position += chunk;
            self.retain_request_checkpoint(&mut request);
            tick.prefilled.push(Step37PrefillProgress {
                request_id: request.id,
                tokens: chunk,
                prompt_position: request.prompt_position,
            });
            self.prefilling.push_back(request.id);
            self.requests.insert(request.id, request);
        }
        Ok(())
    }

    pub fn cancel_request(&mut self, id: Step37RequestId) -> Step37CancelOutcome {
        let Some(request) = self.requests.get(&id) else {
            return Step37CancelOutcome::NotFound;
        };
        if request.lifecycle == RequestState::Finished {
            return Step37CancelOutcome::AlreadyFinished;
        }
        self.waiting.retain(|queued| *queued != id);
        self.prefilling.retain(|queued| *queued != id);
        self.decoding.retain(|queued| *queued != id);
        let mut request = self
            .requests
            .remove(&id)
            .expect("cancellation target retained");
        if let Some(sequence) = request.sequence.take()
            && let Err(error) = (*sequence).finish(&mut self.sequence_cache, &self.cache_stream)
        {
            warn!(request = id.get(), %error, "failed to release cancelled Step sequence state");
        }
        Step37CancelOutcome::Cancelled(Step37CancelledRequest {
            id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            released_sequence_device_bytes: request.sequence_device_bytes,
        })
    }

    pub fn active_sequence_count(&self) -> usize {
        self.prefilling.len() + self.decoding.len()
    }

    pub fn request_state(&self, id: Step37RequestId) -> Option<RequestState> {
        self.requests.get(&id).map(|request| request.lifecycle)
    }

    pub fn remove_finished(&mut self, id: Step37RequestId) -> Option<Step37FinishedRequest> {
        if self.request_state(id) != Some(RequestState::Finished) {
            return None;
        }
        let request = self.requests.remove(&id)?;
        Some(Step37FinishedRequest {
            id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            finish_reason: request
                .finish_reason
                .expect("finished request has a reason"),
            released_sequence_device_bytes: request.sequence_device_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::sampling::SamplingConfig;
    use std::path::PathBuf;

    #[test]
    #[ignore = "loads the full local Step-3.7 checkpoint"]
    fn real_model_prefix_checkpoint_restores_repeated_prompt() {
        let model_dir = std::env::var_os("STEP37_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/step-3.7-flash-nvfp4")
            });
        let model = Step37TextModel::open(model_dir, 240).expect("load Step-3.7 model");
        let mut scheduler = Step37Scheduler::new(
            model,
            SchedulerConfig {
                decode_capacity: 1,
                prefill_sequence_capacity: 1,
                prefill_token_capacity: 128,
                max_active_sequences: 1,
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
            .add_request(prompt, config)
            .expect("second request");
        let tick = scheduler.tick().expect("restore repeated prompt");
        assert_eq!(tick.admitted.len(), 1);
        assert_eq!(tick.admitted[0].cached_prompt_tokens, 128);
        while scheduler.request_state(second) != Some(RequestState::Finished) {
            scheduler.tick().expect("finish second request");
        }
        let second_token = scheduler
            .remove_finished(second)
            .expect("second result")
            .generated_tokens[0]
            .id;
        assert_eq!(second_token, first_token);
    }
}
