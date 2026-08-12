//! Structured multi-session chat serving for Nemotron 3.

use super::cache_config::SequenceCacheConfig;
use super::chat::CheckpointChatTemplate;
use super::chat_output::{ChatOutputCodec, ChatOutputEvent};
use super::nemotron3_sequence_cache::{
    Nemotron3CacheContext, Nemotron3Sequence, Nemotron3SequenceCache, nemotron3_cache_error,
    new_nemotron3_sequence_cache_with_budget,
};
use super::sampling::{Sampler, TokenHistory};
use super::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use super::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use super::sm12x_sequence_cache::Sm12xPageTable;
use super::stop::StopBuffer;
use crate::nemotron3::{
    Nemotron3BlockWorkspace, Nemotron3Model, Nemotron3MtpWorkspace,
    Nemotron3SpeculativeCycleWorkspace,
};
use nvfp4::{DeviceBuffer, Error, Result};
use sequence_cache::{AdmissionOutcome, AdmissionRequest};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nemotron3AdmissionProgress {
    pub request_id: Nemotron3RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time when admission completed.
    pub admitted_after_tick_start: Duration,
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
    prefix_target: usize,
    prefix_retained: bool,
    generation: RequestConfig,
    generated_tokens: usize,
    last_token: Option<u32>,
    sequence: Option<Nemotron3Sequence>,
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
    sequence_cache: Nemotron3SequenceCache,
    retain_prefixes: bool,
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
        Self::new_with_cache_config(model, template, config, SequenceCacheConfig::default())
    }

    /// Creates a multi-session service with ART-backed reusable prompt prefixes.
    pub fn new_with_cache_config(
        model: &'model Nemotron3Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        let sequence_cache = new_nemotron3_sequence_cache_with_budget(
            model,
            config.max_active_sequences,
            config.max_context_tokens,
            (cache_config.max_retained_bytes != 0).then_some(cache_config.max_retained_bytes),
        )?;
        let retain_prefixes = cache_config.max_retained_bytes != 0;
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            active_sequences: 0,
            sequence_cache,
            retain_prefixes,
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
        let prefix_target = if self.retain_prefixes {
            self.sequence_cache
                .cacheable_prefix_tokens(prompt.token_ids.len())
        } else {
            0
        };
        let active = ActiveRequest {
            prompt: prompt.token_ids.clone(),
            prompt_position: 0,
            prefix_target,
            prefix_retained: false,
            generation: request.generation.clone(),
            generated_tokens: 0,
            last_token: None,
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
        self.tick_with_lifecycle(&mut |_| {})
    }

    /// Runs one scheduler iteration and reports admission and prefill events
    /// when they occur.
    pub fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Nemotron3RequestId, Nemotron3AdmissionProgress>,
        ),
    ) -> Result<Nemotron3Tick> {
        let tick_started = Instant::now();
        let mut tick = Nemotron3Tick::default();
        self.admit(&mut tick, tick_started, on_lifecycle)?;
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
                request.sequence.is_some()
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
                request.sequence.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position + 1 < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        self.prefill_blocks(&prefill_ids, &mut tick, on_lifecycle)?;

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

    /// Cancels a waiting or active request.
    pub fn cancel_request(&mut self, id: Nemotron3RequestId) -> Nemotron3CancelOutcome {
        let Some(request) = self.requests.remove(&id) else {
            return Nemotron3CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request
            .sequence
            .as_ref()
            .map_or(0, Nemotron3Sequence::device_bytes);
        if let Some(sequence) = request.sequence {
            if let Err(error) = sequence.finish(self.model, &mut self.sequence_cache) {
                warn!(%error, request_id = id.get(), "failed to release cancelled Nemotron sequence");
            }
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

    fn admit(
        &mut self,
        tick: &mut Nemotron3Tick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Nemotron3RequestId, Nemotron3AdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.active_sequences < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            let request = self.requests.get_mut(&id).expect("waiting request exists");
            let capacity = request.prompt.len() + request.generation.max_new_tokens;
            let prefix = self.sequence_cache.lookup_prefix(&request.prompt);
            let mut state = Some(self.model.sequence_state(capacity.max(1))?);
            let mut restored = false;
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
                    &mut Nemotron3CacheContext {
                        stream: self.model.stream(),
                        page_table: &mut page_table,
                    },
                    |snapshot, position| {
                        if let Some(snapshot) = snapshot {
                            self.model.restore_sequence_snapshot(
                                snapshot,
                                state.as_mut().expect("state allocated"),
                            )?;
                            restored = true;
                        }
                        debug_assert_eq!(state.as_ref().map_or(0, |state| state.len()), position);
                        Ok(())
                    },
                )
                .map_err(nemotron3_cache_error)?;
            let AdmissionOutcome::Admitted(cache_id) = outcome else {
                self.waiting.push_front(id);
                break;
            };
            let state = state.take().expect("new state retained");
            let cached_prompt_tokens = state.len();
            debug_assert_eq!(restored, cached_prompt_tokens != 0);
            let sequence = Nemotron3Sequence::from_admission(cache_id, page_table, state);
            let bytes = sequence.device_bytes();
            request.prompt_position = cached_prompt_tokens;
            request.prefix_retained =
                cached_prompt_tokens == request.prefix_target && cached_prompt_tokens != 0;
            request.sequence = Some(sequence);
            self.active_sequences += 1;
            let progress = Nemotron3AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                admitted_after_tick_start: tick_started.elapsed(),
            };
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
            tick.admitted.push(progress);
        }
        Ok(())
    }

    fn prefill_blocks(
        &mut self,
        ids: &[Nemotron3RequestId],
        tick: &mut Nemotron3Tick,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Nemotron3RequestId, Nemotron3AdmissionProgress>,
        ),
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
            let before_retained_prefix = prefill_rows_before_retained_prefix(
                available,
                request.prompt_position,
                request.prefix_target,
                request.prefix_retained,
            );
            let rows = available
                .min(before_retained_prefix)
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
            .map(|(_, request)| {
                request
                    .sequence
                    .as_mut()
                    .expect("prefill request has sequence")
            })
            .collect::<Vec<_>>();
        for &(id, _, _) in &selected {
            on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
        }
        self.model
            .capture_final_hidden_rows(&states, &mut workspace.previous_hidden)?;
        self.model.forward_block(
            &mut states,
            &chunks,
            &mut workspace.target,
            &mut self.sequence_cache,
        )?;
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
            let retain_prefix = request.prompt_position == request.prefix_target;
            if retain_prefix {
                self.retain_request_prefix(&mut request);
            }
            tick.prefilled.push(Nemotron3PrefillProgress {
                request_id: id,
                prompt_position: request.prompt_position,
            });
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn retain_request_prefix(&mut self, request: &mut ActiveRequest<'template>) {
        if request.prefix_retained || request.prefix_target == 0 {
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
            match self.model.snapshot_sequence(&sequence.state) {
                Ok(snapshot) => {
                    if let Err(error) = self.sequence_cache.retain_prefix(
                        sequence.cache_id,
                        &request.prompt,
                        snapshot,
                        &mut Nemotron3CacheContext {
                            stream: self.model.stream(),
                            page_table: &mut sequence.page_table,
                        },
                    ) {
                        warn!(error = %nemotron3_cache_error(error), "failed to retain Nemotron prompt prefix");
                    }
                }
                Err(error) => warn!(%error, "failed to snapshot Nemotron prompt prefix"),
            }
        }
        request.prefix_retained = true;
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
        let sequence = request
            .sequence
            .as_mut()
            .expect("admitted request has sequence");
        if let Some(workspace) = self.mtp_token_workspace.as_mut() {
            self.model
                .append_mtp_prompt_token(sequence, input, workspace)?;
        }
        self.model
            .forward_one(sequence, &mut self.sequence_cache, input)?;
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(sequence)?;
            super::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            let logits = self.model.logits_to_host(sequence)?;
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
                .map(|(_, request)| {
                    request
                        .sequence
                        .as_mut()
                        .expect("selected request has sequence")
                })
                .collect::<Vec<_>>();
            self.model.speculative_cycle_argmax(
                &mut states,
                &inputs,
                workspace,
                &mut self.sequence_cache,
            )
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
        let sequence = request
            .sequence
            .take()
            .expect("terminal request is admitted");
        let released = sequence.device_bytes();
        sequence.finish(self.model, &mut self.sequence_cache)?;
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

fn prefill_rows_before_retained_prefix(
    available: usize,
    prompt_position: usize,
    prefix_target: usize,
    prefix_retained: bool,
) -> usize {
    if prefix_retained || prefix_target == 0 {
        available
    } else {
        prefix_target.saturating_sub(prompt_position)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_prompts_are_not_blocked_by_an_absent_retained_prefix() {
        assert_eq!(prefill_rows_before_retained_prefix(21, 0, 0, false), 21);
    }
}
