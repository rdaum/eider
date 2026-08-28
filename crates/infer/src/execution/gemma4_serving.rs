//! Multi-session chat serving for Gemma 4.

use crate::gemma4::{
    Gemma4Model, Gemma4PrefillBatchWorkspace, Gemma4PrefillOutput, Gemma4PrefillRow,
    Gemma4SequenceId, Gemma4SequencePool,
};
use crate::gemma4::{
    Gemma4Sequence, Gemma4SequenceCache, gemma4_cache_error, new_gemma4_sequence_cache_with_budget,
};
use crate::metrics::{duration_us, metrics};
use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageTable};
use eider_cuda::{CudaStream, Error, Result, SM12X_KV_PAGE_TOKENS};
use eider_runtime::cache::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineDelta, EngineError,
    EngineFinished, EngineLifecycleEvent, EnginePrefillProgress, EngineRequestId, EngineResult,
    EngineService, EngineTick,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::sampling::{Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
use seqcache::{AdmissionOutcome, AdmissionRequest};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{info, warn};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4AdmissionProgress {
    pub request_id: Gemma4RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    /// Wall time spent allocating active sequence state.
    pub allocation_duration: Duration,
    /// Wall time spent copying a retained checkpoint into active KV storage.
    pub checkpoint_copy_duration: Duration,
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
    pub prefilled: Vec<Gemma4PrefillProgress>,
    pub generated: Vec<Gemma4RequestId>,
    pub output: Vec<Gemma4ChatDelta>,
    pub finished: Vec<Gemma4Finished>,
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
    prefix_target: usize,
    prefix_retained: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    prompt_logits_ready: bool,
    sequence_id: Option<Gemma4SequenceId>,
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
    sequences: Gemma4SequencePool,
    sequence_cache: Gemma4SequenceCache,
}

impl<'model, 'template> Gemma4ChatService<'model, 'template> {
    /// Creates a multi-session service with ART-backed prompt prefixes.
    pub fn new_with_cache_config(
        model: &'model Gemma4Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let stream = CudaStream::new_non_blocking()?;
        let mut prefill_workspace = model.new_prefill_batch_workspace(
            config.prefill_sequence_capacity,
            config.prefill_token_capacity,
            config.max_context_tokens,
        )?;
        let mut sequence_cache = new_gemma4_sequence_cache_with_budget(
            model,
            config.max_active_sequences,
            config.max_context_tokens,
            (cache_config.max_retained_bytes != 0).then_some(cache_config.max_retained_bytes),
        )?;
        let warmup_started = Instant::now();
        let warmup_tokens = vec![0; config.prefill_token_capacity];
        let mut warmup_sequence = Gemma4Sequence::admit(
            model,
            &mut sequence_cache,
            config.prefill_token_capacity,
            &stream,
        )?;
        model.prefill_batch(
            &mut prefill_workspace,
            &mut [Gemma4PrefillRow {
                token_ids: &warmup_tokens,
                sequence: &mut warmup_sequence,
                output: Gemma4PrefillOutput::None,
            }],
            &stream,
            &mut sequence_cache,
        )?;
        stream.synchronize()?;
        warmup_sequence.finish(&mut sequence_cache, &stream)?;
        info!(
            tokens = config.prefill_token_capacity,
            elapsed_ms = warmup_started.elapsed().as_secs_f64() * 1000.0,
            "warmed Gemma 4 prefill path"
        );
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
            stream,
            prefill_workspace,
            tail_prefill_workspace,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            sequences: Gemma4SequencePool::new(),
            sequence_cache,
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
        let prefix_target =
            retained_prompt_prefix_tokens(prompt.token_ids.len(), SM12X_KV_PAGE_TOKENS);
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
                prompt_logits_ready: false,
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
        Ok(Gemma4Admission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one scheduler iteration and reports admission and prefill events
    /// when they occur.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Gemma4RequestId, Gemma4AdmissionProgress>,
        ),
    ) -> Result<Gemma4Tick> {
        let tick_started = Instant::now();
        let mut tick = Gemma4Tick::default();
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

    /// Cancels a waiting or active request.
    pub fn cancel_request(&mut self, id: Gemma4RequestId) -> Gemma4CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Gemma4CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = if let Some(sequence_id) = request.sequence_id {
            match self.sequences.release(sequence_id) {
                Ok(sequence) => {
                    let bytes = sequence.device_bytes();
                    if let Err(error) = sequence.finish(&mut self.sequence_cache, &self.stream) {
                        warn!(%error, "failed to release cancelled Gemma 4 sequence");
                    }
                    bytes
                }
                Err(error) => {
                    warn!(%error, "missing cancelled Gemma 4 sequence");
                    0
                }
            }
        } else {
            0
        };
        Gemma4CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns requests currently owning device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.sequences.len()
    }

    fn admit(
        &mut self,
        _tick: &mut Gemma4Tick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Gemma4RequestId, Gemma4AdmissionProgress>,
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
                        stream: &self.stream,
                        page_table: &mut page_table,
                    },
                    |_snapshot, position| {
                        state.position = position;
                        Ok(())
                    },
                )
                .map_err(gemma4_cache_error)?;
            let cache_id = match outcome {
                AdmissionOutcome::Admitted(id) => id,
                AdmissionOutcome::WouldBlock => {
                    self.waiting.push_front(id);
                    break;
                }
            };
            self.stream.synchronize()?;
            let allocation_duration = allocation_started.elapsed();
            let checkpoint_copy_duration = Duration::ZERO;
            let cached_prompt_tokens = state.len();
            let sequence = Gemma4Sequence::from_admission(cache_id, page_table, state);
            metrics()
                .gemma4_sequence_allocation_us
                .record(duration_us(allocation_duration));
            if checkpoint_copy_duration != Duration::ZERO {
                metrics()
                    .gemma4_checkpoint_copy_us
                    .record(duration_us(checkpoint_copy_duration));
            }
            let bytes = sequence.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence_id = Some(self.sequences.insert(sequence)?);
            let progress = Gemma4AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                allocation_duration,
                checkpoint_copy_duration,
                admitted_after_tick_start: tick_started.elapsed(),
            };
            request.usage.cached_prompt_tokens = cached_prompt_tokens;
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        ids: &[Gemma4RequestId],
        tick: &mut Gemma4Tick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Gemma4RequestId, Gemma4AdmissionProgress>,
        ),
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut budget = self.config.prefill_token_capacity;
        let mut selected = Vec::with_capacity(ids.len());
        for (index, &id) in ids.iter().enumerate() {
            let request = self.requests.get(&id).expect("prefill request exists");
            let available = request.prompt.len().saturating_sub(request.prompt_position);
            let remaining_sequences = ids.len() - index;
            let chunk = available.min(budget.div_ceil(remaining_sequences)).min(
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
        if selected.is_empty() {
            return Ok(());
        }
        let sequence_ids = selected
            .iter()
            .map(|(id, _)| {
                self.requests[id].sequence_id.ok_or_else(|| Error::Format {
                    label: "Gemma 4 scheduled prefill",
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
                    let end = start + chunk;
                    Gemma4PrefillRow {
                        token_ids: &request.prompt[start..end],
                        sequence,
                        output: if end != request.prompt.len() {
                            Gemma4PrefillOutput::None
                        } else if request.sampler.config().uses_fast_argmax() {
                            Gemma4PrefillOutput::Top1
                        } else {
                            Gemma4PrefillOutput::FullLogits
                        },
                    }
                })
                .collect::<Vec<_>>();
            for &(id, _) in &selected {
                on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            }
            self.model
                .prefill_batch(workspace, &mut rows, &self.stream, &mut self.sequence_cache)
                .and_then(|()| self.stream.synchronize())
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
            request.prompt_logits_ready = request.prompt_position == request.prompt.len();
            if checkpoint_ready(
                request.prompt_position,
                request.prefix_target,
                request.prefix_retained,
            ) {
                Self::retain_request_checkpoint(
                    &mut self.sequence_cache,
                    &self.stream,
                    &mut self.sequences,
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
        sequence_cache: &mut Gemma4SequenceCache,
        stream: &CudaStream,
        sequences: &mut Gemma4SequencePool,
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
            warn!("missing sequence while retaining Gemma 4 prompt prefix");
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
                stream,
                page_table: &mut sequence.page_table,
            },
        ) {
            warn!(%error, "failed to retain shared Gemma 4 prompt prefix");
        }
        request.prefix_retained = true;
    }

    fn generate_one(
        &mut self,
        id: Gemma4RequestId,
        tick: &mut Gemma4Tick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let sequence_id = request.sequence_id.expect("decode request is admitted");
        let mut sequences = self.sequences.lease_many(&[sequence_id])?;
        let sequence = sequences.sequence_mut(0);
        if request.prompt_logits_ready {
            request.prompt_logits_ready = false;
        } else {
            let input = request
                .last_token
                .expect("generated Gemma 4 token exists after prompt logits");
            if request.sampler.config().uses_fast_argmax() {
                self.model.forward_one_top1(
                    sequence,
                    input,
                    &self.stream,
                    &mut self.sequence_cache,
                )?;
            } else {
                self.model
                    .forward_one(sequence, input, &self.stream, &mut self.sequence_cache)?;
            }
        }
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self
                .model
                .argmax_with_logit(&sequence.state, &self.stream)?;
            eider_runtime::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            let logits = self.model.logits_to_host(&sequence.state, &self.stream)?;
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
        let sequence_id = request
            .sequence_id
            .take()
            .expect("terminal request is admitted");
        let sequence = self.sequences.release(sequence_id)?;
        let released = sequence.device_bytes();
        sequence.finish(&mut self.sequence_cache, &self.stream)?;
        tick.finished.push(Gemma4Finished {
            request_id: id,
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

fn checkpoint_ready(prompt_position: usize, prefix_target: usize, prefix_retained: bool) -> bool {
    !prefix_retained && prefix_target != 0 && prompt_position >= prefix_target
}

impl EngineService for Gemma4ChatService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> EngineResult<EngineAdmission> {
        let admission = Gemma4ChatService::add_request(self, request).map_err(EngineError::new)?;
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
            |event: RequestLifecycleEvent<Gemma4RequestId, Gemma4AdmissionProgress>| match event {
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
        let tick = Gemma4ChatService::tick_with_lifecycle(self, &mut observer)
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
        match Gemma4ChatService::cancel_request(self, Gemma4RequestId(id.get())) {
            Gemma4CancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Gemma4CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        Gemma4ChatService::active_sequence_count(self)
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
