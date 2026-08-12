//! Multi-session chat serving for Muse Glimmer.

use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::muse_glimmer_sequence_cache::{
    MuseGlimmerSequence, MuseGlimmerSequenceCache, muse_glimmer_cache_error,
    new_muse_glimmer_sequence_cache_with_budget,
};
use super::prefix_cache::{PrefixCacheConfig, cacheable_prompt_prefix_tokens};
use super::sampling::{Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::sm12x_sequence_cache::{Sm12xCacheContext, Sm12xPageTable};
use super::stop::StopBuffer;
use crate::muse_glimmer::{MuseGlimmerDFlashCycle, MuseGlimmerModel};
use nvfp4::{Error, Result};
use sequence_cache::{AdmissionOutcome, AdmissionRequest};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

/// Stable identity assigned to a Muse Glimmer request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MuseGlimmerRequestId(u64);

impl MuseGlimmerRequestId {
    /// Returns the numeric request identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after rendering and tokenization.
pub struct MuseGlimmerAdmission {
    /// Assigned request identity.
    pub request_id: MuseGlimmerRequestId,
    /// Rendered prompt token count.
    pub prompt_tokens: usize,
    /// Requested completion-token limit.
    pub max_output_tokens: usize,
}

/// Device allocation completed during a service tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MuseGlimmerAdmissionProgress {
    /// Admitted request.
    pub request_id: MuseGlimmerRequestId,
    /// Sequence-specific device bytes.
    pub sequence_device_bytes: usize,
    /// Prompt tokens restored from a retained prefix checkpoint.
    pub cached_prompt_tokens: usize,
    /// Time spent allocating a fresh or restored active sequence.
    pub allocation_duration: Duration,
    /// Time spent copying a retained prefix into the active sequence.
    pub checkpoint_copy_duration: Duration,
    /// Elapsed scheduler-tick time at admission.
    pub admitted_after_tick_start: Duration,
}

/// Prompt progress completed during one tick.
pub struct MuseGlimmerPrefillProgress {
    /// Request whose prompt advanced.
    pub request_id: MuseGlimmerRequestId,
    /// Total prompt position after this tick.
    pub prompt_position: usize,
}

/// One structured output delta.
pub struct MuseGlimmerChatDelta {
    /// Request owning this delta.
    pub request_id: MuseGlimmerRequestId,
    /// Reasoning, visible text, or tool-call output.
    pub event: ChatOutputEvent,
}

/// Cumulative DFlash work retained for one request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MuseGlimmerDFlashStats {
    /// Completed draft-and-verify cycles.
    pub cycles: usize,
    /// DFlash predictions proposed across all cycles.
    pub drafted_tokens: usize,
    /// DFlash predictions accepted by the target.
    pub accepted_drafts: usize,
    /// Target-approved tokens emitted to the request.
    pub emitted_tokens: usize,
    /// Time spent inside draft-and-verify cycles.
    pub cycle_duration: Duration,
    /// Latest retained target-model position.
    pub target_position: usize,
    /// Latest retained DFlash position.
    pub dflash_position: usize,
}

/// Updated cumulative DFlash statistics produced by one service tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MuseGlimmerDFlashProgress {
    /// Request owning the speculative state.
    pub request_id: MuseGlimmerRequestId,
    /// Cumulative statistics after the latest cycle.
    pub stats: MuseGlimmerDFlashStats,
}

impl MuseGlimmerDFlashStats {
    fn record_cycle(
        &mut self,
        cycle: &MuseGlimmerDFlashCycle,
        emitted_tokens: usize,
        cycle_duration: Duration,
    ) {
        self.cycles += 1;
        self.drafted_tokens += cycle.drafted_tokens;
        self.accepted_drafts += cycle.accepted_drafts;
        self.emitted_tokens += emitted_tokens;
        self.cycle_duration += cycle_duration;
        self.target_position = cycle.target_position;
        self.dflash_position = cycle.dflash_position;
    }
}

/// Terminal request metadata.
pub struct MuseGlimmerFinished {
    /// Finished request.
    pub request_id: MuseGlimmerRequestId,
    /// API-facing finish reason.
    pub finish_reason: ChatFinishReason,
    /// Final token usage.
    pub usage: ChatUsage,
    /// Sequence device bytes released at completion.
    pub released_sequence_device_bytes: usize,
}

/// Work and output from one service iteration.
#[derive(Default)]
pub struct MuseGlimmerTick {
    /// Requests allocated during this tick.
    pub admitted: Vec<MuseGlimmerAdmissionProgress>,
    /// Prompt progress during this tick.
    pub prefilled: Vec<MuseGlimmerPrefillProgress>,
    /// Requests producing a token during this tick.
    pub generated: Vec<MuseGlimmerRequestId>,
    /// Requests completing a DFlash draft-and-verify cycle.
    pub dflash: Vec<MuseGlimmerDFlashProgress>,
    /// Structured streaming deltas.
    pub output: Vec<MuseGlimmerChatDelta>,
    /// Requests completing during this tick.
    pub finished: Vec<MuseGlimmerFinished>,
    /// Device-resident sequences remaining after the tick.
    pub active_sequences: usize,
}

/// Outcome of cancelling a queued or active request.
pub enum MuseGlimmerCancelOutcome {
    /// The request was removed and these device bytes were released.
    Cancelled {
        /// Sequence-specific allocation released, or zero while queued.
        released_sequence_device_bytes: usize,
    },
    /// No retained request had this identity.
    NotFound,
}

struct ActiveRequest<'tokenizer> {
    prompt: Vec<u32>,
    prompt_position: usize,
    prefix_cache_target: usize,
    prefix_cache_checkpointed: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    dflash_enabled: bool,
    pending_dflash_token: Option<u32>,
    dflash_stats: MuseGlimmerDFlashStats,
    prompt_logits_ready: bool,
    sequence: Option<Box<MuseGlimmerSequence>>,
    sampler: Sampler,
    history: TokenHistory,
    output: ChatOutputCodec<'tokenizer>,
    filter: ResponseFilter,
    usage: ChatUsage,
}

/// Checkpoint rendering and decode-first Muse Glimmer execution.
pub struct MuseGlimmerChatService<'model, 'template> {
    model: &'model MuseGlimmerModel,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<MuseGlimmerRequestId>,
    requests: BTreeMap<MuseGlimmerRequestId, ActiveRequest<'template>>,
    active_sequences: usize,
    sequence_cache: MuseGlimmerSequenceCache,
}

impl<'model, 'template> MuseGlimmerChatService<'model, 'template> {
    /// Creates a service with explicit scheduling limits.
    pub fn new(
        model: &'model MuseGlimmerModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
    ) -> Result<Self> {
        Self::new_with_cache_config(model, template, config, PrefixCacheConfig::default())
    }

    /// Creates a service with explicit scheduling and prompt-prefix limits.
    pub fn new_with_cache_config(
        model: &'model MuseGlimmerModel,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        prefix_cache: PrefixCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.max_context_tokens > model.config().max_position_embeddings {
            return Err(Error::Shape {
                label: "Muse Glimmer scheduler context",
                expected: format!("at most {} tokens", model.config().max_position_embeddings),
                actual: format!("{} tokens", config.max_context_tokens),
            });
        }
        let sequence_cache = new_muse_glimmer_sequence_cache_with_budget(
            model,
            config.max_active_sequences,
            config.max_context_tokens,
            (prefix_cache.max_device_bytes != 0).then_some(prefix_cache.max_device_bytes),
        )?;
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
            sequence_cache,
        })
    }

    /// Renders, tokenizes, and queues a request without allocating GPU state.
    pub fn add_request(&mut self, request: ChatRequest) -> Result<MuseGlimmerAdmission> {
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
                label: "Muse Glimmer chat prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        let total = prompt
            .token_ids
            .len()
            .checked_add(request.generation.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "Muse Glimmer request capacity",
                expected: "prompt + completion without overflow".to_string(),
                actual: format!(
                    "{} + {}",
                    prompt.token_ids.len(),
                    request.generation.max_new_tokens
                ),
            })?;
        if total > self.config.max_context_tokens {
            return Err(Error::Shape {
                label: "Muse Glimmer request capacity",
                expected: format!("at most {} tokens", self.config.max_context_tokens),
                actual: format!("{total} tokens"),
            });
        }
        let id = MuseGlimmerRequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Muse Glimmer request ID",
            detail: "request ID space exhausted".to_string(),
        })?;
        let starts_in_reasoning =
            request.template.add_generation_prompt && request.template.enable_thinking;
        let prefix_cache_target = cacheable_prompt_prefix_tokens(prompt.token_ids.len());
        let prompt_tokens = prompt.token_ids.len();
        let max_output_tokens = request.generation.max_new_tokens;
        let dflash_enabled = self.model.has_dflash()
            && request.generation.sampling.uses_fast_argmax()
            && total
                .checked_add(15)
                .is_some_and(|capacity| capacity <= self.model.config().max_position_embeddings);
        self.requests.insert(
            id,
            ActiveRequest {
                prompt: prompt.token_ids.clone(),
                prompt_position: 0,
                prefix_cache_target,
                prefix_cache_checkpointed: false,
                generation: request.generation.clone(),
                generated_tokens: 0,
                last_token: None,
                dflash_enabled,
                pending_dflash_token: None,
                dflash_stats: MuseGlimmerDFlashStats::default(),
                prompt_logits_ready: false,
                sequence: None,
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
        Ok(MuseGlimmerAdmission {
            request_id: id,
            prompt_tokens,
            max_output_tokens,
        })
    }

    /// Runs one decode-first scheduling iteration.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<MuseGlimmerRequestId, MuseGlimmerAdmissionProgress>,
        ),
    ) -> Result<MuseGlimmerTick> {
        let started = Instant::now();
        let mut tick = MuseGlimmerTick::default();
        self.admit(&mut tick, started, on_lifecycle)?;
        for admission in &tick.admitted {
            self.requests
                .get_mut(&admission.request_id)
                .expect("admitted Muse Glimmer request is retained")
                .usage
                .cached_prompt_tokens = admission.cached_prompt_tokens;
        }
        let mut terminal = BTreeMap::new();
        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.sequence.is_some()
                    && request.prompt_position == request.prompt.len()
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

    /// Cancels a queued or active request.
    pub fn cancel_request(&mut self, id: MuseGlimmerRequestId) -> MuseGlimmerCancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return MuseGlimmerCancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request
            .sequence
            .as_ref()
            .map_or(0, |sequence| sequence.device_bytes());
        if let Some(sequence) = request.sequence
            && let Err(error) = (*sequence).finish(self.model, &mut self.sequence_cache)
        {
            warn!(%error, request_id = id.get(), "failed to release cancelled Muse Glimmer sequence");
        }
        if released != 0 {
            self.active_sequences -= 1;
        }
        MuseGlimmerCancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns the number of requests with device sequence state.
    pub fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    fn admit(
        &mut self,
        tick: &mut MuseGlimmerTick,
        started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<MuseGlimmerRequestId, MuseGlimmerAdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len()
                + request.generation.max_new_tokens
                + usize::from(request.dflash_enabled) * 15;
            let allocation_started = Instant::now();
            let mut checkpoint_copy_duration = Duration::ZERO;
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
                        stream: self.model.stream(),
                        page_table: &mut page_table,
                    },
                    |snapshot, position| {
                        if position == 0 {
                            return Ok(());
                        }
                        let snapshot = snapshot.ok_or_else(|| Error::Format {
                            label: "Muse Glimmer sequence snapshot restore",
                            detail: "retained prefix has no private snapshot".to_string(),
                        })?;
                        let restore_started = Instant::now();
                        self.model
                            .restore_sequence_snapshot(snapshot, &mut state, position)?;
                        checkpoint_copy_duration = restore_started.elapsed();
                        Ok(())
                    },
                )
                .map_err(muse_glimmer_cache_error)?;
            let cache_id = match outcome {
                AdmissionOutcome::Admitted(id) => id,
                AdmissionOutcome::WouldBlock => {
                    self.waiting.push_front(id);
                    break;
                }
            };
            self.model.stream().synchronize()?;
            let allocation_duration = allocation_started.elapsed();
            let cached_prompt_tokens = state.len();
            let sequence = MuseGlimmerSequence::from_admission(cache_id, page_table, state);
            let sequence_device_bytes = sequence.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_cache_checkpointed =
                cached_prompt_tokens == request.prefix_cache_target && cached_prompt_tokens != 0;
            let progress = MuseGlimmerAdmissionProgress {
                request_id: id,
                sequence_device_bytes,
                cached_prompt_tokens,
                allocation_duration,
                checkpoint_copy_duration,
                admitted_after_tick_start: started.elapsed(),
            };
            request.sequence = Some(Box::new(sequence));
            self.active_sequences += 1;
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill(
        &mut self,
        ids: &[MuseGlimmerRequestId],
        tick: &mut MuseGlimmerTick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<MuseGlimmerRequestId, MuseGlimmerAdmissionProgress>,
        ),
    ) -> Result<()> {
        let mut budget = self.config.prefill_token_capacity;
        for (index, &id) in ids.iter().enumerate() {
            let request = self.requests.get_mut(&id).expect("prefill request exists");
            let available = request.prompt.len() - request.prompt_position;
            let remaining = ids.len() - index;
            let chunk = available.min(budget.div_ceil(remaining)).min(
                if !request.prefix_cache_checkpointed
                    && request.prompt_position < request.prefix_cache_target
                {
                    request.prefix_cache_target - request.prompt_position
                } else {
                    usize::MAX
                },
            );
            if chunk == 0 {
                continue;
            }
            budget -= chunk;
            let start = request.prompt_position;
            let end = start + chunk;
            on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
            let sequence = request
                .sequence
                .as_deref_mut()
                .expect("request is admitted");
            if request.dflash_enabled {
                let mut chunk_start = start;
                while chunk_start < end {
                    let chunk_end = (chunk_start + 16).min(end);
                    self.model.dflash_prefill_chunk(
                        sequence,
                        &request.prompt[chunk_start..chunk_end],
                        chunk_end == request.prompt.len(),
                        &mut self.sequence_cache,
                    )?;
                    chunk_start = chunk_end;
                }
            } else {
                for (offset, &token) in request.prompt[start..end].iter().enumerate() {
                    if start + offset + 1 == request.prompt.len() {
                        self.model
                            .forward_one(sequence, token, &mut self.sequence_cache)?;
                    } else {
                        self.model
                            .consume_one(sequence, token, &mut self.sequence_cache)?;
                    }
                }
            }
            request.prompt_position = end;
            request.prompt_logits_ready = end == request.prompt.len();
            if checkpoint_ready(
                request.prompt_position,
                request.prefix_cache_target,
                request.prefix_cache_checkpointed,
            ) {
                Self::retain_request_prefix(self.model, &mut self.sequence_cache, request);
            }
            tick.prefilled.push(MuseGlimmerPrefillProgress {
                request_id: id,
                prompt_position: end,
            });
        }
        Ok(())
    }

    fn retain_request_prefix(
        model: &MuseGlimmerModel,
        sequence_cache: &mut MuseGlimmerSequenceCache,
        request: &mut ActiveRequest<'template>,
    ) {
        if request.prefix_cache_checkpointed || request.prefix_cache_target == 0 {
            return;
        }
        if sequence_cache.config().max_prefix_entries == Some(0) {
            request.prefix_cache_checkpointed = true;
            return;
        }
        let Some(sequence) = request.sequence.as_deref_mut() else {
            return;
        };
        if sequence.position() != request.prefix_cache_target {
            return;
        }
        if sequence_cache.contains_prefix(&request.prompt, request.prefix_cache_target) {
            request.prefix_cache_checkpointed = true;
            return;
        }
        let snapshot = match model.snapshot_sequence(&sequence.state, request.prefix_cache_target) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                warn!(%error, "failed to snapshot Muse Glimmer prompt prefix");
                request.prefix_cache_checkpointed = true;
                return;
            }
        };
        if let Err(error) = sequence_cache.retain_prefix(
            sequence.cache_id,
            &request.prompt,
            snapshot,
            &mut Sm12xCacheContext {
                stream: model.stream(),
                page_table: &mut sequence.page_table,
            },
        ) {
            warn!(%error, "failed to retain shared Muse Glimmer prompt prefix");
        }
        request.prefix_cache_checkpointed = true;
    }

    fn generate_one(
        &mut self,
        id: MuseGlimmerRequestId,
        tick: &mut MuseGlimmerTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        if request.dflash_enabled {
            return Self::generate_dflash(self.model, &mut self.sequence_cache, id, request, tick);
        }
        let sequence = request
            .sequence
            .as_deref_mut()
            .expect("decode request is admitted");
        if request.prompt_logits_ready {
            request.prompt_logits_ready = false;
        } else {
            self.model.forward_one(
                sequence,
                request
                    .last_token
                    .expect("generated token exists after prompt logits"),
                &mut self.sequence_cache,
            )?;
        }
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(sequence)?;
            super::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            request
                .sampler
                .sample(&self.model.logits_to_host(sequence)?, &request.history)?
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

    fn generate_dflash(
        model: &MuseGlimmerModel,
        sequence_cache: &mut MuseGlimmerSequenceCache,
        id: MuseGlimmerRequestId,
        request: &mut ActiveRequest<'template>,
        tick: &mut MuseGlimmerTick,
    ) -> Result<Option<ChatFinishReason>> {
        let anchor = if let Some(token) = request.pending_dflash_token.take() {
            token
        } else {
            if !request.prompt_logits_ready {
                return Err(Error::Format {
                    label: "Muse Glimmer DFlash serving",
                    detail: "missing prompt logits or pending target token".to_string(),
                });
            }
            request.prompt_logits_ready = false;
            model
                .argmax_with_logit(
                    request
                        .sequence
                        .as_deref_mut()
                        .expect("request is admitted"),
                )?
                .0
        };
        let cycle_started = Instant::now();
        let cycle = model.dflash_cycle(
            request
                .sequence
                .as_deref_mut()
                .expect("request is admitted"),
            anchor,
            sequence_cache,
        )?;
        let cycle_duration = cycle_started.elapsed();
        request.pending_dflash_token = Some(cycle.next_token);
        let mut emitted_tokens = 0;
        let mut terminal = None;
        for &token in &cycle.tokens {
            request.generated_tokens += 1;
            emitted_tokens += 1;
            request.last_token = Some(token);
            request.history.push(token);
            request.usage.completion_tokens += 1;
            if request.output.is_reasoning() {
                request.usage.reasoning_tokens += 1;
            }
            tick.generated.push(id);
            let events = request.output.push_token(token)?;
            if let Some(reason) = request.filter.apply(id, events, &mut tick.output) {
                terminal = Some(reason);
                break;
            }
            if request.generation.eos_token_ids.contains(&token) {
                terminal = Some(ChatFinishReason::Eos);
                break;
            }
            if request.generated_tokens == request.generation.max_new_tokens {
                terminal = Some(ChatFinishReason::Length);
                break;
            }
        }
        request
            .dflash_stats
            .record_cycle(&cycle, emitted_tokens, cycle_duration);
        tick.dflash.push(MuseGlimmerDFlashProgress {
            request_id: id,
            stats: request.dflash_stats,
        });
        Ok(terminal)
    }

    fn finish_request(
        &mut self,
        id: MuseGlimmerRequestId,
        mut reason: ChatFinishReason,
        tick: &mut MuseGlimmerTick,
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
        let released = sequence.device_bytes();
        (*sequence).finish(self.model, &mut self.sequence_cache)?;
        self.active_sequences -= 1;
        tick.finished.push(MuseGlimmerFinished {
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
        request_id: MuseGlimmerRequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<MuseGlimmerChatDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => {
                    output.push(MuseGlimmerChatDelta { request_id, event })
                }
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(MuseGlimmerChatDelta {
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
                    output.push(MuseGlimmerChatDelta { request_id, event });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: MuseGlimmerRequestId, output: &mut Vec<MuseGlimmerChatDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(MuseGlimmerChatDelta {
                request_id,
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MuseGlimmerDFlashStats, checkpoint_ready};
    use crate::muse_glimmer::MuseGlimmerDFlashCycle;
    use std::time::Duration;

    #[test]
    fn checkpoint_is_ready_after_crossing_the_aligned_prefix() {
        assert!(checkpoint_ready(4_736, 4_736, false));
        assert!(checkpoint_ready(4_800, 4_736, false));
        assert!(!checkpoint_ready(4_672, 4_736, false));
    }

    #[test]
    fn disabled_or_completed_checkpoint_is_not_ready() {
        assert!(!checkpoint_ready(4_736, 0, false));
        assert!(!checkpoint_ready(4_736, 4_736, true));
    }

    #[test]
    fn dflash_stats_accumulate_acceptance_emissions_latency_and_positions() {
        let mut stats = MuseGlimmerDFlashStats::default();
        stats.record_cycle(
            &MuseGlimmerDFlashCycle {
                tokens: vec![10, 11, 12],
                next_token: 13,
                accepted_drafts: 2,
                drafted_tokens: 15,
                target_position: 4_739,
                dflash_position: 4_739,
            },
            2,
            Duration::from_millis(30),
        );
        stats.record_cycle(
            &MuseGlimmerDFlashCycle {
                tokens: vec![13],
                next_token: 14,
                accepted_drafts: 0,
                drafted_tokens: 15,
                target_position: 4_740,
                dflash_position: 4_740,
            },
            1,
            Duration::from_millis(20),
        );

        assert_eq!(stats.cycles, 2);
        assert_eq!(stats.drafted_tokens, 30);
        assert_eq!(stats.accepted_drafts, 2);
        assert_eq!(stats.emitted_tokens, 3);
        assert_eq!(stats.cycle_duration, Duration::from_millis(50));
        assert_eq!(stats.target_position, 4_740);
        assert_eq!(stats.dflash_position, 4_740);
    }
}
