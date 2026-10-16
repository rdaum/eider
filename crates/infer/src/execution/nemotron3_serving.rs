//! Structured multi-session chat serving for Nemotron 3.

use crate::nemotron3::lilicorr::LiLiCorrWorkspace;
use crate::nemotron3::{
    Nemotron3BlockWorkspace, Nemotron3CacheContext, Nemotron3Model, Nemotron3MtpWorkspace,
    Nemotron3Sequence, Nemotron3SequenceCache, Nemotron3SequenceId, Nemotron3SequencePool,
    Nemotron3SpeculativeCycleWorkspace, nemotron3_cache_error,
    new_nemotron3_sequence_cache_with_budget,
};
use crate::sm12x_cache::Sm12xPageTable;
use eider_cuda::{DeviceBuffer, Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::decision::{DecisionRequest, validated_decision_labels};
use eider_runtime::engine::{
    EngineAdmission, EngineAdmissionProgress, EngineCancelOutcome, EngineCapabilities,
    EngineDecisionAdmission, EngineDelta, EngineError, EngineFinished, EngineLifecycleEvent,
    EnginePrefillProgress, EngineRequestId, EngineResult, EngineService, EngineTick,
    EngineVerificationProgress,
};
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::sampling::{Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, RequestLifecycleEvent, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
use seqcache::{AdmissionOutcome, AdmissionRequest};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::warn;

mod decision;
use crate::nemotron3::Nemotron3DecisionReadout;
use decision::DecisionGroup;

/// Stable request identity assigned by a Nemotron 3 chat service.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Nemotron3RequestId(u64);

impl Nemotron3RequestId {
    /// Returns the numeric request identity.
    fn get(self) -> u64 {
        self.0
    }
}

/// Request metadata known after prompt rendering and tokenization.
struct Nemotron3Admission {
    pub request_id: Nemotron3RequestId,
    pub prompt_tokens: usize,
    pub max_output_tokens: usize,
}

/// Device-state allocation completed during a tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Nemotron3AdmissionProgress {
    pub request_id: Nemotron3RequestId,
    pub sequence_device_bytes: usize,
    pub cached_prompt_tokens: usize,
    /// Elapsed scheduler-tick time when admission completed.
    pub admitted_after_tick_start: Duration,
}

/// Outcome of cancelling a waiting or active request.
enum Nemotron3CancelOutcome {
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
    sequence_id: Option<Nemotron3SequenceId>,
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
    pub(crate) fn new(
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
            target: model.prefill_workspace(sequence_count, rows)?,
            previous_hidden: DeviceBuffer::zeroed(sequence_count * hidden)?,
            mtp,
            mtp_hidden: (mtp_rows != 0)
                .then(|| DeviceBuffer::zeroed(mtp_rows * hidden))
                .transpose()?,
        })
    }
}

/// Checkpoint prompt rendering and round-robin multi-session Nemotron execution.
pub(crate) struct Nemotron3ChatService<'model, 'template> {
    model: &'model Nemotron3Model,
    template: &'template CheckpointChatTemplate,
    config: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<Nemotron3RequestId>,
    requests: BTreeMap<Nemotron3RequestId, ActiveRequest<'template>>,
    sequences: Nemotron3SequencePool,
    sequence_cache: Nemotron3SequenceCache,
    retain_prefixes: bool,
    mtp_token_workspace: Option<Nemotron3MtpWorkspace>,
    lilicorr_workspace: Option<LiLiCorrWorkspace>,
    lilicorr_verification: BTreeMap<usize, Nemotron3BlockWorkspace>,
    prefill_workspaces: BTreeMap<(usize, usize, usize, usize), Nemotron3PrefillWorkspace>,
    speculative_workspaces: BTreeMap<usize, Nemotron3SpeculativeCycleWorkspace>,
    decision_readout: Nemotron3DecisionReadout,
    decisions: BTreeMap<Nemotron3RequestId, DecisionGroup>,
    decision_ready: VecDeque<Nemotron3RequestId>,
    decision_workspaces: BTreeMap<Vec<usize>, Nemotron3BlockWorkspace>,
}

impl<'model, 'template> Nemotron3ChatService<'model, 'template> {
    /// Creates a multi-session service with ART-backed reusable prompt prefixes.
    pub(crate) fn new_with_cache_config(
        model: &'model Nemotron3Model,
        template: &'template CheckpointChatTemplate,
        config: SchedulerConfig,
        cache_config: SequenceCacheConfig,
    ) -> Result<Self> {
        config.validate()?;
        if model.lilicorr().is_some() {
            validate_lilicorr_drafts(config.speculative_drafts)?;
        } else {
            validate_mtp_drafts(config.speculative_drafts)?;
        }
        let sequence_cache = new_nemotron3_sequence_cache_with_budget(
            model,
            config.max_active_sequences,
            config.max_context_tokens,
            (cache_config.max_retained_bytes != 0).then_some(cache_config.max_retained_bytes),
        )?;
        let retain_prefixes = cache_config.max_retained_bytes != 0;
        let labels = validated_decision_labels(template.tokenizer())
            .map_err(|err| decision::error(err.to_string()))?
            .into_iter()
            .map(|(_, id)| id)
            .collect::<Vec<_>>();
        Ok(Self {
            model,
            template,
            config,
            next_id: 1,
            waiting: VecDeque::new(),
            requests: BTreeMap::new(),
            sequences: Nemotron3SequencePool::new(),
            sequence_cache,
            retain_prefixes,
            mtp_token_workspace: (model.lilicorr().is_none()
                && model.has_mtp()
                && config.speculative_drafts != 0)
                .then(|| model.mtp_workspace(1, 1))
                .transpose()?,
            lilicorr_workspace: model
                .lilicorr()
                .map(|draft| draft.new_workspace())
                .transpose()?,
            lilicorr_verification: BTreeMap::new(),
            prefill_workspaces: BTreeMap::new(),
            speculative_workspaces: BTreeMap::new(),
            decision_readout: model.new_decision_readout(&labels)?,
            decisions: BTreeMap::new(),
            decision_ready: VecDeque::new(),
            decision_workspaces: BTreeMap::new(),
        })
    }

    /// Renders, tokenizes, and queues one request without allocating GPU state.
    fn add_request(&mut self, request: ChatRequest) -> Result<Nemotron3Admission> {
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

    /// Runs one scheduler iteration and reports admission and prefill events
    /// when they occur.
    fn tick_with_lifecycle(
        &mut self,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Nemotron3RequestId, Nemotron3AdmissionProgress>,
        ),
    ) -> Result<EngineTick> {
        let tick_started = Instant::now();
        let mut tick = EngineTick::default();
        self.admit(&mut tick, tick_started, on_lifecycle)?;

        let mut terminal = BTreeMap::new();

        let decode_ids = self
            .requests
            .iter()
            .filter(|(_, request)| {
                request.sequence_id.is_some()
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
            if self.lilicorr_workspace.is_some()
                && self.config.speculative_drafts != 0
                && request.last_token.is_some()
                && remaining >= 2
            {
                let drafts = self.config.speculative_drafts.min(remaining - 1);
                if let Some(reason) = self.generate_lilicorr(id, drafts, &mut tick)? {
                    terminal.insert(id, reason);
                }
                continue;
            }
            if self.mtp_token_workspace.is_some()
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
                request.sequence_id.is_some()
                    && request.generation.max_new_tokens != 0
                    && request.prompt_position + 1 < request.prompt.len()
            })
            .map(|(&id, _)| id)
            .take(self.config.prefill_sequence_capacity)
            .collect::<Vec<_>>();
        let before_prefill = prefill_ids
            .iter()
            .map(|id| self.requests[id].prompt_position)
            .sum::<usize>();
        self.prefill_blocks(&prefill_ids, &mut tick, on_lifecycle)?;
        let chat_rows = prefill_ids
            .iter()
            .map(|id| self.requests[id].prompt_position)
            .sum::<usize>()
            - before_prefill;

        for (&id, request) in &self.requests {
            if request.sequence_id.is_some() && request.generation.max_new_tokens == 0 {
                terminal.entry(id).or_insert(ChatFinishReason::Length);
            }
        }
        for (id, reason) in terminal {
            self.finish_request(id, reason, &mut tick)?;
        }
        self.run_decisions(&mut tick, chat_rows, prefill_ids.len())?;
        Ok(tick)
    }

    /// Cancels a waiting or active request.
    fn cancel_request(&mut self, id: Nemotron3RequestId) -> Nemotron3CancelOutcome {
        if let Some(mut group) = self.decisions.remove(&id) {
            self.waiting.retain(|&waiting| waiting != id);
            self.decision_ready.retain(|&ready| ready != id);
            let bytes = group.device_bytes();
            if let Err(error) = group.release(self.model) {
                warn!(%error, "failed to release cancelled Nemotron decision");
            }
            return Nemotron3CancelOutcome::Cancelled {
                released_sequence_device_bytes: bytes,
            };
        }
        let Some(request) = self.requests.remove(&id) else {
            return Nemotron3CancelOutcome::NotFound;
        };
        self.waiting.retain(|&waiting| waiting != id);
        let released = request
            .sequence_id
            .and_then(|sequence_id| self.sequences.release(sequence_id).ok())
            .map_or(0, |sequence| {
                let bytes = sequence.device_bytes();
                if let Err(error) = sequence.finish(self.model, &mut self.sequence_cache) {
                    warn!(%error, request_id = id.get(), "failed to release cancelled Nemotron sequence");
                }
                bytes
            });
        Nemotron3CancelOutcome::Cancelled {
            released_sequence_device_bytes: released,
        }
    }

    /// Returns all live chat, decision-parent, and decision-branch sequences.
    fn active_sequence_count(&self) -> usize {
        self.sequences.len()
            + self
                .decisions
                .values()
                .map(DecisionGroup::sequence_count)
                .sum::<usize>()
    }

    fn admitted_request_count(&self) -> usize {
        self.sequences.len()
            + self
                .decisions
                .values()
                .filter(|group| group.active())
                .count()
    }

    fn admit(
        &mut self,
        tick: &mut EngineTick,
        tick_started: Instant,
        on_lifecycle: &mut dyn FnMut(
            RequestLifecycleEvent<Nemotron3RequestId, Nemotron3AdmissionProgress>,
        ),
    ) -> Result<()> {
        while self.admitted_request_count() < self.config.max_active_sequences {
            let Some(id) = self.waiting.pop_front() else {
                break;
            };
            if self.decisions.contains_key(&id) {
                self.admit_decision(id, tick)?;
                continue;
            }
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
            request.sequence_id = Some(self.sequences.insert(sequence)?);
            let progress = Nemotron3AdmissionProgress {
                request_id: id,
                sequence_device_bytes: bytes,
                cached_prompt_tokens,
                admitted_after_tick_start: tick_started.elapsed(),
            };
            request.usage.cached_prompt_tokens = cached_prompt_tokens;
            on_lifecycle(RequestLifecycleEvent::Admitted(progress));
        }
        Ok(())
    }

    fn prefill_blocks(
        &mut self,
        ids: &[Nemotron3RequestId],
        tick: &mut EngineTick,
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
            .filter(|_| self.mtp_token_workspace.is_some())
            .map(|(_, start, chunk)| chunk.len().saturating_sub(usize::from(*start == 0)))
            .sum::<usize>();
        let mtp_sequence_count = selected
            .iter()
            .filter(|_| self.mtp_token_workspace.is_some())
            .filter(|(_, start, chunk)| chunk.len() > usize::from(*start == 0))
            .count();
        let key = (sequence_count, rows, mtp_sequence_count, mtp_rows);
        if !self.prefill_workspaces.contains_key(&key) {
            // Keep full chunks and the current tail. Each tail owns GEMM scratch,
            // so retaining every exact prompt remainder would grow without bound.
            self.prefill_workspaces
                .retain(|cached, _| cached.1 == self.config.prefill_token_capacity);
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
        let sequence_ids = requests
            .iter()
            .map(|(_, request)| request.sequence_id.expect("prefill request has sequence"))
            .collect::<Vec<_>>();
        for &(id, _, _) in &selected {
            on_lifecycle(RequestLifecycleEvent::PrefillStarted(id));
        }
        let result = (|| -> Result<()> {
            let mut sequences = self.sequences.lease_many(&sequence_ids)?;
            let mut states = sequences.sequences_mut().collect::<Vec<_>>();
            if workspace.mtp.is_some() {
                self.model
                    .capture_final_hidden_rows(&states, &mut workspace.previous_hidden)?;
            }
            self.model.forward_block(
                &mut states,
                &chunks,
                &mut workspace.target,
                &mut self.sequence_cache,
            )?;
            if let Some(draft_workspace) = &mut self.lilicorr_workspace {
                let mut offset = 0;
                for (sequence, chunk) in states.iter_mut().zip(&chunks) {
                    self.model.append_lilicorr_block_context(
                        sequence,
                        &workspace.target,
                        offset,
                        chunk.len(),
                        draft_workspace,
                    )?;
                    offset += chunk.len();
                }
            }
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
            for (index, ((_, request), (_, _, chunk))) in
                requests.iter_mut().zip(&selected).enumerate()
            {
                request.prompt_position += chunk.len();
                if request.prompt_position == request.prefix_target {
                    Self::retain_request_prefix(
                        self.model,
                        &mut self.sequence_cache,
                        request,
                        sequences.sequence_mut(index),
                    );
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for (id, request) in requests {
                self.requests.insert(id, request);
            }
            return Err(error);
        }
        for (id, request) in requests {
            tick.prefilled.push(EnginePrefillProgress {
                request_id: EngineRequestId::new(id.get()),
                prompt_position: request.prompt_position,
            });
            self.requests.insert(id, request);
        }
        Ok(())
    }

    fn retain_request_prefix(
        model: &Nemotron3Model,
        sequence_cache: &mut Nemotron3SequenceCache,
        request: &mut ActiveRequest<'template>,
        sequence: &mut Nemotron3Sequence,
    ) {
        if request.prefix_retained || request.prefix_target == 0 {
            return;
        }
        if sequence.position() != request.prefix_target {
            return;
        }
        if !sequence_cache.contains_prefix(&request.prompt, request.prefix_target) {
            match model.snapshot_sequence(&sequence.state) {
                Ok(snapshot) => {
                    if let Err(error) = sequence_cache.retain_prefix(
                        sequence.cache_id,
                        &request.prompt,
                        snapshot,
                        &mut Nemotron3CacheContext {
                            stream: model.stream(),
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
        tick: &mut EngineTick,
    ) -> Result<Option<ChatFinishReason>> {
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let input = request
            .last_token
            .unwrap_or(request.prompt[request.prompt.len() - 1]);
        let mut sequence = self
            .sequences
            .lease(request.sequence_id.expect("admitted request has sequence"))?;
        let sequence = sequence.sequence_mut();
        if let Some(workspace) = self.mtp_token_workspace.as_mut() {
            self.model
                .append_mtp_prompt_token(sequence, input, workspace)?;
        }
        self.model
            .forward_one(sequence, &mut self.sequence_cache, input)?;
        if let Some(workspace) = &mut self.lilicorr_workspace {
            self.model
                .append_lilicorr_decode_context(sequence, workspace)?;
        }
        let sampled = if request.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(sequence)?;
            eider_runtime::sampling::SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            }
        } else {
            let logits = self.model.logits_to_host(sequence)?;
            request.sampler.sample(&logits, &request.history)?
        };
        Self::emit_token(id, request, sampled.id, tick)
    }

    fn emit_token(
        id: Nemotron3RequestId,
        request: &mut ActiveRequest<'template>,
        token: u32,
        tick: &mut EngineTick,
    ) -> Result<Option<ChatFinishReason>> {
        request.generated_tokens += 1;
        request.last_token = Some(token);
        request.history.push(token);
        request.usage.completion_tokens += 1;
        if request.output.is_reasoning() {
            request.usage.reasoning_tokens += 1;
        }
        tick.generated.push(EngineRequestId::new(id.get()));
        let events = request.output.push_token(token)?;
        if let Some(reason) = request.filter.apply(id, events, &mut tick.output) {
            return Ok(Some(reason));
        }
        if request.generation.eos_token_ids.contains(&token) {
            return Ok(Some(ChatFinishReason::Eos));
        }
        if request.generated_tokens == request.generation.max_new_tokens {
            return Ok(Some(ChatFinishReason::Length));
        }
        Ok(None)
    }

    fn generate_lilicorr(
        &mut self,
        id: Nemotron3RequestId,
        drafts: usize,
        tick: &mut EngineTick,
    ) -> Result<Option<ChatFinishReason>> {
        if !self.lilicorr_verification.contains_key(&drafts) {
            // Retain the full block and one current tail; recurrent snapshots are large.
            self.lilicorr_verification
                .retain(|&size, _| size == self.config.speculative_drafts);
            self.lilicorr_verification
                .insert(drafts, self.model.speculative_workspace(1, drafts + 1)?);
        }
        let request = self.requests.get_mut(&id).expect("decode request exists");
        let mut sequence = self
            .sequences
            .lease(request.sequence_id.expect("admitted request has sequence"))?;
        let result = self.model.lilicorr_cycle(
            sequence.sequence_mut(),
            request.last_token.expect("speculative request has anchor"),
            drafts,
            self.lilicorr_workspace.as_mut().expect("companion enabled"),
            self.lilicorr_verification
                .get_mut(&drafts)
                .expect("verification workspace inserted"),
            &mut request.sampler,
            &request.history,
            &mut self.sequence_cache,
        )?;
        tick.verification.push(EngineVerificationProgress {
            request_id: EngineRequestId::new(id.get()),
            cycles: 1,
            accepted_drafts: result.accepted_counts()[0] as usize,
        });
        for token in result.emitted_tokens(0)? {
            if let Some(reason) = Self::emit_token(id, request, token, tick)? {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }

    fn generate_speculative(
        &mut self,
        ids: &[Nemotron3RequestId],
        tick: &mut EngineTick,
        terminal: &mut BTreeMap<Nemotron3RequestId, ChatFinishReason>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let selected = ids
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
        let sequence_ids = selected
            .iter()
            .map(|(_, request)| request.sequence_id.expect("selected request has sequence"))
            .collect::<Vec<_>>();
        let result = (|| -> Result<_> {
            if !self.speculative_workspaces.contains_key(&selected.len()) {
                let workspace = self.model.speculative_cycle_workspace(selected.len())?;
                self.speculative_workspaces
                    .insert(selected.len(), workspace);
            }
            let workspace = self
                .speculative_workspaces
                .get_mut(&selected.len())
                .expect("speculative workspace was inserted");
            let mut sequences = self.sequences.lease_many(&sequence_ids)?;
            let mut states = sequences.sequences_mut().collect::<Vec<_>>();
            self.model.speculative_cycle_argmax(
                &mut states,
                &inputs,
                workspace,
                &mut self.sequence_cache,
            )
        })();
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
            tick.verification.push(EngineVerificationProgress {
                request_id: EngineRequestId::new(id.get()),
                cycles: 1,
                accepted_drafts: result.accepted_counts()[sequence] as usize,
            });
            for token in emitted {
                if let Some(reason) = Self::emit_token(id, &mut request, token, tick)? {
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
        id: Nemotron3RequestId,
        mut reason: ChatFinishReason,
        tick: &mut EngineTick,
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
        let sequence = self.sequences.release(
            request
                .sequence_id
                .take()
                .expect("terminal request is admitted"),
        )?;
        let released = sequence.device_bytes();
        sequence.finish(self.model, &mut self.sequence_cache)?;
        tick.finished.push(EngineFinished {
            request_id: EngineRequestId::new(id.get()),
            finish_reason: reason,
            usage: request.usage,
            released_sequence_device_bytes: released,
        });
        Ok(())
    }
}

fn validate_mtp_drafts(drafts: usize) -> Result<()> {
    if matches!(drafts, 0 | 3) {
        return Ok(());
    }
    Err(eider_cuda::Error::Shape {
        label: "Nemotron 3 speculative drafts",
        expected: "0 (disabled) or 3 MTP drafts".to_string(),
        actual: drafts.to_string(),
    })
}

fn validate_lilicorr_drafts(drafts: usize) -> Result<()> {
    if drafts <= 7 {
        return Ok(());
    }
    Err(Error::Shape {
        label: "Nemotron LiLiCorr drafts",
        expected: "0 (disabled) or 1..=7 LiLiCorr drafts".into(),
        actual: drafts.to_string(),
    })
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

impl EngineService for Nemotron3ChatService<'_, '_> {
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities { decisions: true }
    }

    fn add_decision(&mut self, request: DecisionRequest) -> EngineResult<EngineDecisionAdmission> {
        self.queue_decision(request).map_err(EngineError::new)
    }

    fn add_request(&mut self, request: ChatRequest) -> EngineResult<EngineAdmission> {
        let admission =
            Nemotron3ChatService::add_request(self, request).map_err(EngineError::new)?;
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
        let mut observer = |event: RequestLifecycleEvent<
            Nemotron3RequestId,
            Nemotron3AdmissionProgress,
        >| match event {
            RequestLifecycleEvent::Admitted(progress) => {
                on_lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                    request_id: EngineRequestId::new(progress.request_id.get()),
                    sequence_device_bytes: progress.sequence_device_bytes,
                    cached_prompt_tokens: progress.cached_prompt_tokens,
                    allocation_duration: Duration::ZERO,
                    checkpoint_copy_duration: Duration::ZERO,
                    admitted_after_tick_start: progress.admitted_after_tick_start,
                }))
            }
            RequestLifecycleEvent::PrefillStarted(id) => on_lifecycle(
                EngineLifecycleEvent::PrefillStarted(EngineRequestId::new(id.get())),
            ),
        };
        Nemotron3ChatService::tick_with_lifecycle(self, &mut observer).map_err(EngineError::new)
    }
    fn cancel_request(&mut self, id: EngineRequestId) -> EngineCancelOutcome {
        match Nemotron3ChatService::cancel_request(self, Nemotron3RequestId(id.get())) {
            Nemotron3CancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Nemotron3CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }
    fn active_sequence_count(&self) -> usize {
        Nemotron3ChatService::active_sequence_count(self)
    }
}

struct ResponseFilter {
    stop: StopBuffer,
    saw_tool_calls: bool,
}

impl ResponseFilter {
    pub(crate) fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            stop: StopBuffer::new(stop_sequences),
            saw_tool_calls: false,
        }
    }

    fn apply(
        &mut self,
        request_id: Nemotron3RequestId,
        events: Vec<ChatOutputEvent>,
        output: &mut Vec<EngineDelta>,
    ) -> Option<ChatFinishReason> {
        for event in events {
            match event {
                ChatOutputEvent::Reasoning(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Reasoning(_) => output.push(EngineDelta {
                    request_id: EngineRequestId::new(request_id.get()),
                    event,
                }),
                ChatOutputEvent::Text(_) if self.saw_tool_calls => {}
                ChatOutputEvent::Text(text) => {
                    let stopped = self.stop.push(&text);
                    if !stopped.text.is_empty() {
                        output.push(EngineDelta {
                            request_id: EngineRequestId::new(request_id.get()),
                            event: ChatOutputEvent::Text(stopped.text),
                        });
                    }
                    if let Some(sequence) = stopped.matched {
                        return Some(ChatFinishReason::Stop(sequence));
                    }
                }
                ChatOutputEvent::ToolCall(_) => {
                    self.flush(request_id, output);
                    output.push(EngineDelta {
                        request_id: EngineRequestId::new(request_id.get()),
                        event,
                    });
                    self.saw_tool_calls = true;
                    return Some(ChatFinishReason::ToolCalls);
                }
            }
        }
        None
    }

    fn flush(&mut self, request_id: Nemotron3RequestId, output: &mut Vec<EngineDelta>) {
        let text = self.stop.finish();
        if !text.is_empty() {
            output.push(EngineDelta {
                request_id: EngineRequestId::new(request_id.get()),
                event: ChatOutputEvent::Text(text),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::chat::ChatMessage;
    use eider_runtime::sampling::SamplingConfig;

    fn run_checkpoint_request(
        service: &mut Nemotron3ChatService<'_, '_>,
        request: ChatRequest,
    ) -> (String, serde_json::Value) {
        let admission = service.add_request(request).unwrap();
        let started = Instant::now();
        let mut first = None;
        let mut cycles = 0;
        let mut accepted = 0;
        let mut output = String::new();
        loop {
            let tick = service.tick_with_lifecycle(&mut |_| {}).unwrap();
            if !tick.generated.is_empty() && first.is_none() {
                first = Some(Instant::now());
            }
            for progress in tick.verification {
                cycles += progress.cycles;
                accepted += progress.accepted_drafts;
            }
            for delta in tick.output {
                output.push_str(&format!("{:?}\n", delta.event));
            }
            if let Some(finished) = tick.finished.first() {
                let first = first.expect("request generated output");
                let decode_seconds = first.elapsed().as_secs_f64();
                assert_eq!(service.active_sequence_count(), 0);
                return (
                    output,
                    serde_json::json!({
                        "prompt_tokens": admission.prompt_tokens,
                        "cached_tokens": finished.usage.cached_prompt_tokens,
                        "output_tokens": finished.usage.completion_tokens,
                        "ttft_ms": first.duration_since(started).as_secs_f64() * 1000.,
                        "total_ms": started.elapsed().as_secs_f64() * 1000.,
                        "decode_tok_s": finished.usage.completion_tokens.saturating_sub(1) as f64 / decode_seconds,
                        "cycles": cycles, "accepted_drafts": accepted,
                    }),
                );
            }
        }
    }

    #[test]
    #[ignore = "loads the full target and companion; requires NEMOTRON_MODEL_DIR and LILICORR_DIR"]
    fn lilicorr_serving_restores_prefix_and_preserves_request_policies() {
        let root = std::env::var("NEMOTRON_MODEL_DIR").unwrap();
        let template = CheckpointChatTemplate::from_model_dir(&root).unwrap();
        let content = std::env::var("LILICORR_PROMPT").ok()
            .map(|path| std::fs::read_to_string(path).unwrap())
            .unwrap_or_else(|| format!("Review this Rust code for correctness, then explain the three most important issues.\n{}", include_str!("nemotron3_serving.rs")));
        let request = ChatRequest::new(
            vec![ChatMessage::user(content)],
            RequestConfig {
                max_new_tokens: 128,
                sampling: SamplingConfig {
                    temperature: 0.,
                    ..SamplingConfig::default()
                },
                ..RequestConfig::default()
            },
        );
        let scheduler = SchedulerConfig {
            max_active_sequences: 1,
            decode_capacity: 1,
            prefill_sequence_capacity: 1,
            prefill_token_capacity: 1024,
            max_context_tokens: 32768,
            ..SchedulerConfig::default()
        };
        let mut model = Nemotron3Model::load(&root).unwrap();
        // Match the launcher: the generic 2 GiB default masked undersized
        // snapshot reserves when LiLiCorr's sliding context was included.
        let cache_config = SequenceCacheConfig {
            max_retained_bytes: 1024 * 1024 * 1024,
        };
        let mut baseline =
            Nemotron3ChatService::new_with_cache_config(&model, &template, scheduler, cache_config)
                .unwrap();
        let (_, serial_cold) = run_checkpoint_request(&mut baseline, request.clone());
        let (_, serial_cached) = run_checkpoint_request(&mut baseline, request.clone());
        drop(baseline);
        model
            .enable_lilicorr(std::env::var("LILICORR_DIR").unwrap())
            .unwrap();
        let speculative_drafts = std::env::var("LILICORR_TEST_DRAFTS")
            .map(|value| value.parse().unwrap())
            .unwrap_or(7);
        let mut service = Nemotron3ChatService::new_with_cache_config(
            &model,
            &template,
            SchedulerConfig {
                speculative_drafts,
                ..scheduler
            },
            cache_config,
        )
        .unwrap();
        let (cold_output, cold) = run_checkpoint_request(&mut service, request.clone());
        let (cached_output, cached) = run_checkpoint_request(&mut service, request.clone());
        assert_eq!(
            cold_output, cached_output,
            "restored target and draft context must reproduce the uncached request"
        );
        assert!(cached["cached_tokens"].as_u64().unwrap() > 0);
        assert!(cached["cycles"].as_u64().unwrap() > 0);
        let mut sampled = request.clone();
        sampled.generation.max_new_tokens = 17;
        sampled.generation.sampling = SamplingConfig {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            presence_penalty: 0.2,
            frequency_penalty: 0.1,
            seed: Some(72),
        };
        let (sampled_output, sampled_report) =
            run_checkpoint_request(&mut service, sampled.clone());
        let (repeated_output, _) = run_checkpoint_request(&mut service, sampled);
        assert_eq!(
            sampled_output, repeated_output,
            "seeded speculative sampling must be repeatable"
        );
        assert!(sampled_report["cycles"].as_u64().unwrap() > 0);
        assert!(sampled_report["output_tokens"].as_u64().unwrap() <= 17);
        let queued = service.add_request(request.clone()).unwrap();
        assert!(matches!(
            service.cancel_request(queued.request_id),
            Nemotron3CancelOutcome::Cancelled {
                released_sequence_device_bytes: 0
            }
        ));
        let active = service.add_request(request).unwrap();
        service.tick_with_lifecycle(&mut |_| {}).unwrap();
        assert!(
            matches!(service.cancel_request(active.request_id), Nemotron3CancelOutcome::Cancelled {released_sequence_device_bytes} if released_sequence_device_bytes > 0)
        );
        assert_eq!(service.active_sequence_count(), 0);
        if let Ok(path) = std::env::var("LILICORR_SERVING_REPORT") {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "drafts": speculative_drafts,
                    "serial_cold": serial_cold, "serial_cached": serial_cached,
                    "lilicorr_cold": cold, "lilicorr_cached": cached, "sampled": sampled_report,
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn mtp_requires_the_supported_draft_count() {
        assert!(validate_mtp_drafts(0).is_ok());
        assert!(validate_mtp_drafts(3).is_ok());
        for drafts in [1, 2, 4, usize::MAX] {
            assert!(validate_mtp_drafts(drafts).is_err());
        }
    }

    #[test]
    fn lilicorr_accepts_full_and_shortened_blocks() {
        for drafts in 0..=7 {
            assert!(validate_lilicorr_drafts(drafts).is_ok());
        }
        for drafts in [8, usize::MAX] {
            assert!(validate_lilicorr_drafts(drafts).is_err());
        }
    }

    #[test]
    fn short_prompts_are_not_blocked_by_an_absent_retained_prefix() {
        assert_eq!(prefill_rows_before_retained_prefix(21, 0, 0, false), 21);
    }
}
