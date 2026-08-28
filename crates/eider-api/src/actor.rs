//! Dedicated inference thread owning all CUDA and scheduler state.

use crate::metrics::{FinishReason, ServerEndpoint, metrics as server_metrics};
use crate::protocol::{ApiError, InferenceEvent, InferenceFinished};
use eider_inference::metrics::metrics as infer_metrics;
use eider_inference::{InferenceEngineConfig, with_loaded_engine};
use eider_runtime::engine::{
    EngineCancelOutcome, EngineDraftStats, EngineFinished, EngineLifecycleEvent, EngineRequestId,
    EngineService, EngineVerificationProgress,
};
use eider_runtime::generation::GenerationConfig;
use eider_runtime::request::{ChatFinishReason, ChatRequest};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

const SESSION_METRICS_INTERVAL: Duration = Duration::from_secs(10);

/// API-actor configuration around one inference engine.
#[derive(Clone, Debug)]
pub struct InferenceActorConfig {
    /// Model loading and execution configuration.
    pub engine: InferenceEngineConfig,
    /// Bounded event queue per API request.
    pub event_capacity: usize,
}

impl InferenceActorConfig {
    /// Creates actor and engine configuration with standard defaults.
    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            engine: InferenceEngineConfig::new(model_dir),
            event_capacity: 256,
        }
    }
}

/// Actor-local request identity used for cancellation from async clients.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ActorRequestId(u64);

/// Accepted request and its bounded event receiver.
pub struct ActorResponse {
    pub id: ActorRequestId,
    pub events: mpsc::Receiver<InferenceEvent>,
}

/// Cloneable async-side handle for the CUDA-owning inference thread.
#[derive(Clone)]
pub struct InferenceActor {
    inner: Arc<ActorInner>,
    defaults: GenerationConfig,
}

struct ActorInner {
    commands: mpsc::UnboundedSender<ActorCommand>,
    next_request_id: AtomicU64,
    event_capacity: usize,
    worker: Option<thread::JoinHandle<()>>,
}

enum ActorCommand {
    Submit {
        id: ActorRequestId,
        request: ChatRequest,
        events: mpsc::Sender<InferenceEvent>,
        submitted_at: Instant,
    },
    Cancel(ActorRequestId),
    Shutdown,
}

struct ActiveRequest {
    external_id: ActorRequestId,
    events: mpsc::Sender<InferenceEvent>,
    metrics: SessionMetrics,
}

struct SessionMetrics {
    submitted_at: Instant,
    admitted_at: Option<Instant>,
    prefill_started_at: Option<Instant>,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    sequence_device_bytes: usize,
    prefilled_tokens: usize,
    last_prefill_report_at: Instant,
    last_prefill_report_tokens: usize,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    last_report_at: Option<Instant>,
    last_report_tokens: usize,
    generated_tokens: usize,
    verification_cycles: usize,
    accepted_drafts: usize,
    draft_progress: Option<DraftSessionMetrics>,
}

struct PrefillMetricsSnapshot {
    prompt_position: usize,
    interval_tokens_per_second: f64,
    compute_tokens_per_second: f64,
    effective_tokens_per_second: f64,
}

struct SessionMetricsSnapshot {
    output_tokens: usize,
    interval_tokens_per_second: f64,
    decode_tokens_per_second: f64,
}

struct DraftSessionMetrics {
    cumulative: EngineDraftStats,
    last_report_at: Instant,
    last_report: EngineDraftStats,
}

struct DraftMetricsSnapshot {
    interval: EngineDraftStats,
    cumulative: EngineDraftStats,
}

impl InferenceActor {
    /// Starts the actor and waits until model loading and workspace allocation finish.
    pub fn spawn(config: InferenceActorConfig) -> Result<Self, ApiError> {
        if config.event_capacity == 0 {
            return Err(ApiError::server(
                "actor event capacity must be greater than zero",
            ));
        }
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let event_capacity = config.event_capacity;
        let worker = thread::Builder::new()
            .name("eider-inference".to_string())
            .spawn(move || actor_main(config, commands_rx, ready_tx))
            .map_err(|error| {
                ApiError::server(format!("failed to start inference actor: {error}"))
            })?;
        let defaults = ready_rx
            .recv()
            .map_err(|_| ApiError::server("inference actor exited during startup"))?
            .map_err(ApiError::server)?;
        Ok(Self {
            inner: Arc::new(ActorInner {
                commands: commands_tx,
                next_request_id: AtomicU64::new(1),
                event_capacity,
                worker: Some(worker),
            }),
            defaults,
        })
    }

    pub fn generation_defaults(&self) -> &GenerationConfig {
        &self.defaults
    }

    /// Queues a request without blocking an async executor on CUDA work.
    pub fn submit(&self, request: ChatRequest) -> Result<ActorResponse, ApiError> {
        let id = ActorRequestId(self.inner.next_request_id.fetch_add(1, Ordering::Relaxed));
        let (events_tx, events_rx) = mpsc::channel(self.inner.event_capacity);
        self.inner
            .commands
            .send(ActorCommand::Submit {
                id,
                request,
                events: events_tx,
                submitted_at: Instant::now(),
            })
            .map_err(|_| ApiError::server("inference actor is not running"))?;
        Ok(ActorResponse {
            id,
            events: events_rx,
        })
    }

    /// Requests cancellation. It is safe if the request has already finished.
    pub fn cancel(&self, id: ActorRequestId) {
        let _ = self.inner.commands.send(ActorCommand::Cancel(id));
    }

    /// Stops accepting inference work and cancels active requests.
    pub fn shutdown(&self) {
        let _ = self.inner.commands.send(ActorCommand::Shutdown);
    }
}

impl Drop for ActorInner {
    fn drop(&mut self) {
        let _ = self.commands.send(ActorCommand::Shutdown);
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            error!("inference actor panicked during shutdown");
        }
    }
}

fn actor_main(
    config: InferenceActorConfig,
    mut commands: mpsc::UnboundedReceiver<ActorCommand>,
    ready: std::sync::mpsc::SyncSender<Result<GenerationConfig, String>>,
) {
    let result = with_loaded_engine(config.engine, |service, defaults| {
        run_actor_loop(service, &mut commands, &ready, defaults);
    });
    if let Err(error) = result {
        let _ = ready.send(Err(error.to_string()));
    }
}

fn run_actor_loop(
    service: &mut dyn EngineService,
    commands: &mut mpsc::UnboundedReceiver<ActorCommand>,
    ready: &std::sync::mpsc::SyncSender<Result<GenerationConfig, String>>,
    defaults: GenerationConfig,
) {
    info!(
        temperature = %defaults.sampling.temperature,
        top_k = defaults.sampling.top_k,
        top_p = %defaults.sampling.top_p,
        seed = ?defaults.sampling.seed,
        presence_penalty = %defaults.sampling.presence_penalty,
        frequency_penalty = %defaults.sampling.frequency_penalty,
        "inference actor ready"
    );
    if ready.send(Ok(defaults)).is_err() {
        return;
    }

    let mut active = BTreeMap::<EngineRequestId, ActiveRequest>::new();
    let mut scheduler_by_external = BTreeMap::<ActorRequestId, EngineRequestId>::new();
    loop {
        if active.is_empty() {
            let Some(command) = commands.blocking_recv() else {
                break;
            };
            if !handle_command(command, service, &mut active, &mut scheduler_by_external) {
                break;
            }
        }

        while let Ok(command) = commands.try_recv() {
            if !handle_command(command, service, &mut active, &mut scheduler_by_external) {
                cancel_all(service, &mut active, &mut scheduler_by_external);
                shutdown_service(service);
                return;
            }
        }
        if active.is_empty() {
            continue;
        }

        let tick_start = Instant::now();
        let mut live_active_sequences = service.active_sequence_count();
        let tick_result = {
            let mut on_lifecycle = |event| match event {
                EngineLifecycleEvent::Admitted(admission) => {
                    let Some(request) = active.get_mut(&admission.request_id) else {
                        return;
                    };
                    let admitted_at = tick_start + admission.admitted_after_tick_start;
                    request.metrics.record_admission(
                        admitted_at,
                        admission.cached_prompt_tokens,
                        admission.sequence_device_bytes,
                    );
                    infer_metrics().requests_admitted.inc();
                    live_active_sequences += 1;
                    info!(
                        session = request.external_id.0,
                        state_bytes = admission.sequence_device_bytes,
                        cached_prompt_tokens = admission.cached_prompt_tokens,
                        admission_ms = request.metrics.admission_duration().as_secs_f64() * 1000.0,
                        allocation_ms = admission.allocation_duration.as_secs_f64() * 1000.0,
                        checkpoint_copy_ms =
                            admission.checkpoint_copy_duration.as_secs_f64() * 1000.0,
                        active_sequences = live_active_sequences,
                        "request admitted"
                    );
                }
                EngineLifecycleEvent::PrefillStarted(request_id) => {
                    let Some(request) = active.get_mut(&request_id) else {
                        return;
                    };
                    let now = Instant::now();
                    if request.metrics.record_prefill_start(now) {
                        info!(
                            session = request.external_id.0,
                            prompt_tokens = request.metrics.prompt_tokens,
                            queued_ms = now
                                .duration_since(request.metrics.submitted_at)
                                .as_secs_f64()
                                * 1000.0,
                            "prefill started"
                        );
                    }
                }
            };
            service.tick(&mut on_lifecycle)
        };
        let tick = match tick_result {
            Ok(tick) => tick,
            Err(error) => {
                let message = error.to_string();
                error!(error = %message, "inference scheduler failed");
                server_metrics()
                    .request_errors
                    .add(ServerEndpoint::Responses, active.len() as isize);
                for request in active.values() {
                    let _ = request
                        .events
                        .try_send(InferenceEvent::Error(message.clone()));
                }
                fail_all(service, &mut active, &mut scheduler_by_external, &message);
                continue;
            }
        };
        let now = Instant::now();
        let tick_us = duration_us(now.duration_since(tick_start));
        if !tick.prefilled.is_empty() {
            infer_metrics().prefill_tick_us.record(tick_us);
        }
        if !tick.generated.is_empty() {
            infer_metrics().decode_tick_us.record(tick_us);
        }
        for progress in &tick.prefilled {
            if let Some(request) = active.get_mut(&progress.request_id) {
                let prefill_delta = progress
                    .prompt_position
                    .saturating_sub(request.metrics.prefilled_tokens);
                infer_metrics().prefill_tokens.add(prefill_delta as isize);
                let snapshot = request
                    .metrics
                    .record_prefill(now, progress.prompt_position);
                if let Some(snapshot) = snapshot {
                    info!(
                        session = request.external_id.0,
                        prompt_position = snapshot.prompt_position,
                        prompt_tokens = request.metrics.prompt_tokens,
                        interval_tok_s = snapshot.interval_tokens_per_second,
                        prefill_compute_tok_s = snapshot.compute_tokens_per_second,
                        effective_prefill_tok_s = snapshot.effective_tokens_per_second,
                        "prefill progress"
                    );
                }
            }
        }
        for progress in &tick.verification {
            if let Some(request) = active.get_mut(&progress.request_id) {
                request.metrics.record_verification(progress);
            }
        }
        for request_id in &tick.generated {
            if let Some(request) = active.get_mut(request_id) {
                let starting = request.metrics.first_token_at.is_none();
                let snapshot = request.metrics.record_token(now);
                infer_metrics().generated_tokens.inc();
                if starting {
                    let ttft = now.duration_since(request.metrics.submitted_at);
                    let admission = request.metrics.admission_duration();
                    let prefill_compute = request.metrics.prefill_compute_duration(now);
                    let effective_prefill_tok_s =
                        request.metrics.effective_prefill_tokens_per_second(now);
                    let prefill_compute_tok_s =
                        request.metrics.prefill_compute_tokens_per_second(now);
                    infer_metrics().ttft_us.record(duration_us(ttft));
                    server_metrics()
                        .request_admission_duration_us
                        .record(duration_us(admission));
                    server_metrics()
                        .prefill_tokens_per_second
                        .record(effective_prefill_tok_s as u64);
                    server_metrics()
                        .prefill_compute_tokens_per_second
                        .record(prefill_compute_tok_s as u64);
                    info!(
                        session = request.external_id.0,
                        ttft_ms = ttft.as_secs_f64() * 1000.0,
                        admission_ms = admission.as_secs_f64() * 1000.0,
                        prefill_compute_ms = prefill_compute.as_secs_f64() * 1000.0,
                        prompt_tokens = request.metrics.prompt_tokens,
                        cached_prompt_tokens = request.metrics.cached_prompt_tokens,
                        prefill_compute_tok_s,
                        effective_prefill_tok_s,
                        "decoding started"
                    );
                }
                if let Some(snapshot) = snapshot {
                    info!(
                        session = request.external_id.0,
                        output_tokens = snapshot.output_tokens,
                        interval_tok_s = snapshot.interval_tokens_per_second,
                        decode_tok_s = snapshot.decode_tokens_per_second,
                        verification_cycles = request.metrics.verification_cycles,
                        accepted_drafts = request.metrics.accepted_drafts,
                        accepted_drafts_per_cycle = ratio(
                            request.metrics.accepted_drafts,
                            request.metrics.verification_cycles
                        ),
                        "decode progress"
                    );
                }
            }
        }
        for progress in &tick.draft_progress {
            if let Some(request) = active.get_mut(&progress.request_id)
                && let Some(snapshot) = request.metrics.record_draft_progress(now, progress.stats)
            {
                snapshot.log(request.external_id);
            }
        }
        let mut disconnected = Vec::new();
        for delta in tick.output {
            if let Some(request) = active.get(&delta.request_id)
                && request
                    .events
                    .try_send(InferenceEvent::Output(delta.event))
                    .is_err()
            {
                disconnected.push(delta.request_id);
            }
        }
        let active_sequences = service.active_sequence_count();
        for finished in tick.finished {
            if let Some(request) = active.remove(&finished.request_id) {
                scheduler_by_external.remove(&request.external_id);
                let active_requests = active.len();
                infer_metrics().requests_completed.inc();
                let reason = map_finish_reason(&finished.finish_reason);
                server_metrics().responses_completed.inc(reason);
                server_metrics()
                    .completion_tokens
                    .add(finished.usage.completion_tokens as isize);
                server_metrics()
                    .decode_tokens_per_second
                    .record(request.metrics.decode_tokens_per_second() as u64);
                request.metrics.log_finished(
                    request.external_id,
                    now,
                    &finished,
                    active_requests,
                    active_sequences,
                );
                let _ = request
                    .events
                    .try_send(InferenceEvent::Finished(InferenceFinished {
                        finish_reason: finished.finish_reason,
                        usage: finished.usage,
                    }));
            }
        }
        disconnected.sort_unstable();
        disconnected.dedup();
        for id in disconnected {
            cancel_scheduler_request(id, service, &mut active, &mut scheduler_by_external);
        }
        update_current_counts(service, &active);
    }
    cancel_all(service, &mut active, &mut scheduler_by_external);
    shutdown_service(service);
}

fn shutdown_service(service: &mut dyn EngineService) {
    if let Err(error) = service.shutdown() {
        error!(error = %error, "failed to shut down inference service");
    }
}

fn handle_command(
    command: ActorCommand,
    service: &mut dyn EngineService,
    active: &mut BTreeMap<EngineRequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, EngineRequestId>,
) -> bool {
    match command {
        ActorCommand::Submit {
            id,
            request,
            events,
            submitted_at,
        } => match service.add_request(request) {
            Ok(admission) => {
                active.insert(
                    admission.request_id,
                    ActiveRequest {
                        external_id: id,
                        events,
                        metrics: SessionMetrics::new(submitted_at, admission.prompt_tokens),
                    },
                );
                scheduler_by_external.insert(id, admission.request_id);
                server_metrics().active_requests.set(active.len() as i64);
                server_metrics()
                    .prompt_tokens
                    .add(admission.prompt_tokens as isize);
                info!(
                    session = id.0,
                    prompt_tokens = admission.prompt_tokens,
                    max_output_tokens = admission.max_output_tokens,
                    active_requests = active.len(),
                    "request queued"
                );
            }
            Err(error) => {
                warn!(session = id.0, error = %error, "failed to admit request");
                server_metrics().responses_admission_errors.inc();
                let _ = events.try_send(InferenceEvent::Error(error.to_string()));
            }
        },
        ActorCommand::Cancel(id) => {
            if let Some(scheduler_id) = scheduler_by_external.get(&id).copied() {
                cancel_scheduler_request(scheduler_id, service, active, scheduler_by_external);
            }
        }
        ActorCommand::Shutdown => return false,
    }
    true
}

fn cancel_scheduler_request(
    scheduler_id: EngineRequestId,
    service: &mut dyn EngineService,
    active: &mut BTreeMap<EngineRequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, EngineRequestId>,
) {
    let outcome = service.cancel_request(scheduler_id);
    let released_sequence_device_bytes = match outcome {
        EngineCancelOutcome::Cancelled {
            released_sequence_device_bytes,
        } => released_sequence_device_bytes,
        EngineCancelOutcome::AlreadyFinished | EngineCancelOutcome::NotFound => 0,
    };
    if let Some(request) = active.remove(&scheduler_id) {
        scheduler_by_external.remove(&request.external_id);
        server_metrics().active_requests.set(active.len() as i64);
        infer_metrics().requests_cancelled.inc();
        server_metrics()
            .responses_completed
            .inc(FinishReason::Cancelled);
        request.metrics.log_cancelled(
            request.external_id,
            Instant::now(),
            released_sequence_device_bytes,
            active.len(),
            service.active_sequence_count(),
        );
    }
    update_current_counts(service, active);
}

fn fail_all(
    service: &mut dyn EngineService,
    active: &mut BTreeMap<EngineRequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, EngineRequestId>,
    error: &str,
) {
    let ids = active.keys().copied().collect::<Vec<_>>();
    for id in ids {
        let outcome = service.cancel_request(id);
        let released_sequence_device_bytes = match outcome {
            EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => released_sequence_device_bytes,
            EngineCancelOutcome::AlreadyFinished | EngineCancelOutcome::NotFound => 0,
        };
        if let Some(request) = active.remove(&id) {
            scheduler_by_external.remove(&request.external_id);
            infer_metrics().requests_failed.inc();
            server_metrics()
                .responses_completed
                .inc(FinishReason::Error);
            request.metrics.log_failed(
                request.external_id,
                Instant::now(),
                released_sequence_device_bytes,
                active.len(),
                service.active_sequence_count(),
                error,
            );
        }
    }
    update_current_counts(service, active);
}

fn cancel_all(
    service: &mut dyn EngineService,
    active: &mut BTreeMap<EngineRequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, EngineRequestId>,
) {
    let ids = active.keys().copied().collect::<Vec<_>>();
    for id in ids {
        cancel_scheduler_request(id, service, active, scheduler_by_external);
    }
}

fn update_current_counts(
    service: &dyn EngineService,
    active: &BTreeMap<EngineRequestId, ActiveRequest>,
) {
    server_metrics().active_requests.set(active.len() as i64);
    infer_metrics()
        .active_sequences
        .set(service.active_sequence_count() as i64);
}

impl SessionMetrics {
    fn new(submitted_at: Instant, prompt_tokens: usize) -> Self {
        Self {
            submitted_at,
            admitted_at: None,
            prefill_started_at: None,
            prompt_tokens,
            cached_prompt_tokens: 0,
            sequence_device_bytes: 0,
            prefilled_tokens: 0,
            last_prefill_report_at: submitted_at,
            last_prefill_report_tokens: 0,
            first_token_at: None,
            last_token_at: None,
            last_report_at: None,
            last_report_tokens: 0,
            generated_tokens: 0,
            verification_cycles: 0,
            accepted_drafts: 0,
            draft_progress: None,
        }
    }

    fn record_admission(
        &mut self,
        now: Instant,
        cached_prompt_tokens: usize,
        sequence_device_bytes: usize,
    ) {
        self.admitted_at = Some(now);
        self.sequence_device_bytes = sequence_device_bytes;
        self.cached_prompt_tokens = cached_prompt_tokens;
        self.prefilled_tokens = cached_prompt_tokens;
        self.last_prefill_report_tokens = cached_prompt_tokens;
        if self.prefill_started_at.is_none() {
            self.last_prefill_report_at = now;
        }
    }

    fn record_prefill_start(&mut self, now: Instant) -> bool {
        if self.prefill_started_at.is_some() {
            return false;
        }
        self.prefill_started_at = Some(now);
        self.last_prefill_report_at = now;
        true
    }

    fn record_prefill(
        &mut self,
        now: Instant,
        prompt_position: usize,
    ) -> Option<PrefillMetricsSnapshot> {
        self.prefilled_tokens = prompt_position;
        let interval = now.duration_since(self.last_prefill_report_at);
        if interval < SESSION_METRICS_INTERVAL {
            return None;
        }
        let interval_tokens = prompt_position.saturating_sub(self.last_prefill_report_tokens);
        let snapshot = PrefillMetricsSnapshot {
            prompt_position,
            interval_tokens_per_second: rate(interval_tokens, interval),
            compute_tokens_per_second: self.prefill_compute_tokens_per_second(now),
            effective_tokens_per_second: self.effective_prefill_tokens_per_second(now),
        };
        self.last_prefill_report_at = now;
        self.last_prefill_report_tokens = prompt_position;
        Some(snapshot)
    }

    fn record_token(&mut self, now: Instant) -> Option<SessionMetricsSnapshot> {
        self.generated_tokens += 1;
        self.last_token_at = Some(now);
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
            self.last_report_at = Some(now);
            self.last_report_tokens = self.generated_tokens;
            return None;
        }
        let last_report_at = self
            .last_report_at
            .expect("first token starts report interval");
        let interval = now.duration_since(last_report_at);
        if interval < SESSION_METRICS_INTERVAL {
            return None;
        }
        let interval_tokens = self.generated_tokens - self.last_report_tokens;
        let snapshot = SessionMetricsSnapshot {
            output_tokens: self.generated_tokens,
            interval_tokens_per_second: rate(interval_tokens, interval),
            decode_tokens_per_second: self.decode_tokens_per_second(),
        };
        self.last_report_at = Some(now);
        self.last_report_tokens = self.generated_tokens;
        Some(snapshot)
    }

    fn record_verification(&mut self, progress: &EngineVerificationProgress) {
        self.verification_cycles += progress.cycles;
        self.accepted_drafts += progress.accepted_drafts;
    }

    fn record_draft_progress(
        &mut self,
        now: Instant,
        stats: EngineDraftStats,
    ) -> Option<DraftMetricsSnapshot> {
        let Some(draft_progress) = &mut self.draft_progress else {
            self.draft_progress = Some(DraftSessionMetrics {
                cumulative: stats,
                last_report_at: now,
                last_report: EngineDraftStats::default(),
            });
            return None;
        };
        draft_progress.cumulative = stats;
        if now.duration_since(draft_progress.last_report_at) < SESSION_METRICS_INTERVAL {
            return None;
        }
        let snapshot = DraftMetricsSnapshot {
            interval: draft_stats_delta(stats, draft_progress.last_report),
            cumulative: stats,
        };
        draft_progress.last_report_at = now;
        draft_progress.last_report = stats;
        Some(snapshot)
    }

    fn log_finished(
        &self,
        id: ActorRequestId,
        now: Instant,
        finished: &EngineFinished,
        active_requests: usize,
        active_sequences: usize,
    ) {
        debug_assert_eq!(self.generated_tokens, finished.usage.completion_tokens);
        debug_assert_eq!(
            self.cached_prompt_tokens,
            finished.usage.cached_prompt_tokens
        );
        let time_to_first_token = self.first_token_at.map_or(Duration::ZERO, |first| {
            first.duration_since(self.submitted_at)
        });
        let admission = self.admission_duration();
        let prefill_compute = self.prefill_compute_duration(now);
        info!(
            session = id.0,
            prompt_tokens = finished.usage.prompt_tokens,
            cached_prompt_tokens = finished.usage.cached_prompt_tokens,
            output_tokens = finished.usage.completion_tokens,
            reasoning_tokens = finished.usage.reasoning_tokens,
            ttft_ms = time_to_first_token.as_secs_f64() * 1000.0,
            admission_ms = admission.as_secs_f64() * 1000.0,
            prefill_compute_ms = prefill_compute.as_secs_f64() * 1000.0,
            prefill_compute_tok_s = self.prefill_compute_tokens_per_second(now),
            effective_prefill_tok_s = self.effective_prefill_tokens_per_second(now),
            decode_tok_s = self.decode_tokens_per_second(),
            verification_cycles = self.verification_cycles,
            accepted_drafts = self.accepted_drafts,
            accepted_drafts_per_cycle = ratio(
                self.accepted_drafts,
                self.verification_cycles
            ),
            total_tok_s = rate(
                finished.usage.completion_tokens,
                now.duration_since(self.submitted_at)
            ),
            finish_reason = ?finished.finish_reason,
            state_released_bytes = finished.released_sequence_device_bytes,
            active_requests,
            active_sequences,
            "session complete"
        );
        if let Some(draft_progress) = &self.draft_progress {
            log_draft_summary(id, draft_progress.cumulative);
        }
    }

    fn log_cancelled(
        &self,
        id: ActorRequestId,
        now: Instant,
        released_sequence_device_bytes: usize,
        active_requests: usize,
        active_sequences: usize,
    ) {
        info!(
            session = id.0,
            output_tokens = self.generated_tokens,
            elapsed_ms = now.duration_since(self.submitted_at).as_secs_f64() * 1000.0,
            decode_tok_s = self.decode_tokens_per_second(),
            state_released_bytes = released_sequence_device_bytes,
            active_requests,
            active_sequences,
            "session cancelled"
        );
    }

    fn log_failed(
        &self,
        id: ActorRequestId,
        now: Instant,
        released_sequence_device_bytes: usize,
        active_requests: usize,
        active_sequences: usize,
        error: &str,
    ) {
        warn!(
            session = id.0,
            output_tokens = self.generated_tokens,
            elapsed_ms = now.duration_since(self.submitted_at).as_secs_f64() * 1000.0,
            decode_tok_s = self.decode_tokens_per_second(),
            state_released_bytes = released_sequence_device_bytes,
            active_requests,
            active_sequences,
            error,
            "session failed"
        );
    }

    fn admission_duration(&self) -> Duration {
        self.admitted_at.map_or(Duration::ZERO, |admitted| {
            admitted.duration_since(self.submitted_at)
        })
    }

    fn prefill_compute_duration(&self, now: Instant) -> Duration {
        let Some(started) = self.prefill_started_at else {
            return Duration::ZERO;
        };
        self.first_token_at.unwrap_or(now).duration_since(started)
    }

    fn effective_prefill_tokens_per_second(&self, now: Instant) -> f64 {
        let finished = self.first_token_at.unwrap_or(now);
        rate(
            self.uncached_prefilled_tokens(),
            finished.duration_since(self.submitted_at),
        )
    }

    fn prefill_compute_tokens_per_second(&self, now: Instant) -> f64 {
        rate(
            self.uncached_prefilled_tokens(),
            self.prefill_compute_duration(now),
        )
    }

    fn uncached_prefilled_tokens(&self) -> usize {
        self.prefilled_tokens
            .saturating_sub(self.cached_prompt_tokens)
    }

    fn decode_tokens_per_second(&self) -> f64 {
        let (Some(first), Some(last)) = (self.first_token_at, self.last_token_at) else {
            return 0.0;
        };
        rate(
            self.generated_tokens.saturating_sub(1),
            last.duration_since(first),
        )
    }
}

impl DraftMetricsSnapshot {
    fn log(&self, id: ActorRequestId) {
        info!(
            session = id.0,
            interval_cycles = self.interval.cycles,
            interval_drafted_tokens = self.interval.drafted_tokens,
            interval_accepted_drafts = self.interval.accepted_drafts,
            interval_acceptance_pct =
                percentage(self.interval.accepted_drafts, self.interval.drafted_tokens),
            interval_emitted_tokens = self.interval.emitted_tokens,
            interval_tokens_per_cycle = ratio(self.interval.emitted_tokens, self.interval.cycles),
            interval_cycle_ms =
                average_duration_ms(self.interval.cycle_duration, self.interval.cycles),
            cycles = self.cumulative.cycles,
            drafted_tokens = self.cumulative.drafted_tokens,
            accepted_drafts = self.cumulative.accepted_drafts,
            acceptance_pct = percentage(
                self.cumulative.accepted_drafts,
                self.cumulative.drafted_tokens
            ),
            emitted_tokens = self.cumulative.emitted_tokens,
            tokens_per_cycle = ratio(self.cumulative.emitted_tokens, self.cumulative.cycles),
            cycle_ms = average_duration_ms(self.cumulative.cycle_duration, self.cumulative.cycles),
            target_position = self.cumulative.target_position,
            draft_position = self.cumulative.draft_position,
            "draft-and-verify progress"
        );
    }
}

fn log_draft_summary(id: ActorRequestId, stats: EngineDraftStats) {
    info!(
        session = id.0,
        cycles = stats.cycles,
        drafted_tokens = stats.drafted_tokens,
        accepted_drafts = stats.accepted_drafts,
        acceptance_pct = percentage(stats.accepted_drafts, stats.drafted_tokens),
        emitted_tokens = stats.emitted_tokens,
        tokens_per_cycle = ratio(stats.emitted_tokens, stats.cycles),
        cycle_ms = average_duration_ms(stats.cycle_duration, stats.cycles),
        target_position = stats.target_position,
        draft_position = stats.draft_position,
        "draft-and-verify session complete"
    );
}

fn draft_stats_delta(current: EngineDraftStats, previous: EngineDraftStats) -> EngineDraftStats {
    EngineDraftStats {
        cycles: current.cycles.saturating_sub(previous.cycles),
        drafted_tokens: current
            .drafted_tokens
            .saturating_sub(previous.drafted_tokens),
        accepted_drafts: current
            .accepted_drafts
            .saturating_sub(previous.accepted_drafts),
        emitted_tokens: current
            .emitted_tokens
            .saturating_sub(previous.emitted_tokens),
        cycle_duration: current
            .cycle_duration
            .saturating_sub(previous.cycle_duration),
        target_position: current.target_position,
        draft_position: current.draft_position,
    }
}

fn percentage(numerator: usize, denominator: usize) -> f64 {
    ratio(numerator, denominator) * 100.0
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

fn average_duration_ms(duration: Duration, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    duration.as_secs_f64() * 1000.0 / count as f64
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return 0.0;
    }
    tokens as f64 / elapsed.as_secs_f64()
}

fn duration_us(elapsed: Duration) -> u64 {
    elapsed.as_micros().min(u128::from(u64::MAX)) as u64
}

fn map_finish_reason(reason: &ChatFinishReason) -> FinishReason {
    match reason {
        ChatFinishReason::Eos => FinishReason::Eos,
        ChatFinishReason::Length => FinishReason::Length,
        ChatFinishReason::Stop(_) => FinishReason::Stop,
        ChatFinishReason::ToolCalls => FinishReason::ToolCalls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_inference::{CheckpointArchitecture, checkpoint_architecture};
    use eider_runtime::chat_output::ChatOutputEvent;
    use eider_runtime::engine::{
        EngineAdmission, EngineAdmissionProgress, EngineDelta, EnginePrefillProgress, EngineResult,
        EngineTick,
    };
    use eider_runtime::request::ChatUsage;
    use std::fs;

    struct FakeEngine {
        ticked: bool,
        cancelled: Vec<EngineRequestId>,
    }

    impl EngineService for FakeEngine {
        fn add_request(&mut self, _request: ChatRequest) -> EngineResult<EngineAdmission> {
            Ok(EngineAdmission {
                request_id: EngineRequestId::new(1),
                prompt_tokens: 3,
                max_output_tokens: 2,
            })
        }

        fn tick(
            &mut self,
            on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
        ) -> EngineResult<EngineTick> {
            assert!(!self.ticked, "actor must stop after the terminal tick");
            self.ticked = true;
            let request_id = EngineRequestId::new(1);
            on_lifecycle(EngineLifecycleEvent::Admitted(EngineAdmissionProgress {
                request_id,
                sequence_device_bytes: 64,
                cached_prompt_tokens: 0,
                allocation_duration: Duration::ZERO,
                checkpoint_copy_duration: Duration::ZERO,
                admitted_after_tick_start: Duration::ZERO,
            }));
            on_lifecycle(EngineLifecycleEvent::PrefillStarted(request_id));
            Ok(EngineTick {
                prefilled: vec![EnginePrefillProgress {
                    request_id,
                    prompt_position: 3,
                }],
                generated: vec![request_id],
                output: vec![EngineDelta {
                    request_id,
                    event: ChatOutputEvent::Text("ok".to_string()),
                }],
                finished: vec![EngineFinished {
                    request_id,
                    finish_reason: ChatFinishReason::Length,
                    usage: ChatUsage {
                        prompt_tokens: 3,
                        completion_tokens: 1,
                        ..ChatUsage::default()
                    },
                    released_sequence_device_bytes: 64,
                }],
                ..EngineTick::default()
            })
        }

        fn cancel_request(&mut self, id: EngineRequestId) -> EngineCancelOutcome {
            self.cancelled.push(id);
            EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: 0,
            }
        }

        fn active_sequence_count(&self) -> usize {
            usize::from(!self.ticked)
        }
    }

    #[test]
    fn actor_drives_a_model_neutral_engine_contract() {
        let (commands_tx, mut commands) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (events_tx, mut events) = mpsc::channel(4);
        commands_tx
            .send(ActorCommand::Submit {
                id: ActorRequestId(1),
                request: ChatRequest::new(Vec::new(), Default::default()),
                events: events_tx,
                submitted_at: Instant::now(),
            })
            .expect("actor command receiver is live");
        drop(commands_tx);

        let mut engine = FakeEngine {
            ticked: false,
            cancelled: Vec::new(),
        };
        run_actor_loop(
            &mut engine,
            &mut commands,
            &ready_tx,
            GenerationConfig::default(),
        );

        assert!(ready_rx.recv().expect("actor reports readiness").is_ok());
        assert!(matches!(
            events.try_recv(),
            Ok(InferenceEvent::Output(ChatOutputEvent::Text(text))) if text == "ok"
        ));
        assert!(matches!(
            events.try_recv(),
            Ok(InferenceEvent::Finished(InferenceFinished {
                finish_reason: ChatFinishReason::Length,
                ..
            }))
        ));
        assert!(engine.cancelled.is_empty());
    }

    #[test]
    fn session_metrics_report_exact_interval_and_decode_rates() {
        let submitted = Instant::now();
        let first = submitted + Duration::from_secs(1);
        let mut metrics = SessionMetrics::new(submitted, 100);
        assert!(metrics.record_token(first).is_none());
        for seconds in 2..11 {
            assert!(
                metrics
                    .record_token(submitted + Duration::from_secs(seconds))
                    .is_none()
            );
        }
        let snapshot = metrics
            .record_token(submitted + Duration::from_secs(11))
            .expect("ten-second report interval elapsed");
        assert_eq!(snapshot.output_tokens, 11);
        assert_eq!(snapshot.interval_tokens_per_second, 1.0);
        assert_eq!(snapshot.decode_tokens_per_second, 1.0);
    }

    #[test]
    fn session_metrics_accumulate_verification_acceptance() {
        let mut metrics = SessionMetrics::new(Instant::now(), 8);
        metrics.record_verification(&EngineVerificationProgress {
            request_id: EngineRequestId::new(7),
            cycles: 1,
            accepted_drafts: 2,
        });
        metrics.record_verification(&EngineVerificationProgress {
            request_id: EngineRequestId::new(7),
            cycles: 1,
            accepted_drafts: 1,
        });

        assert_eq!(metrics.verification_cycles, 2);
        assert_eq!(metrics.accepted_drafts, 3);
        assert_eq!(
            ratio(metrics.accepted_drafts, metrics.verification_cycles),
            1.5
        );
    }

    #[test]
    fn draft_metrics_report_interval_and_cumulative_acceptance() {
        let started = Instant::now();
        let mut metrics = SessionMetrics::new(started, 1_000);
        assert!(
            metrics
                .record_draft_progress(
                    started,
                    EngineDraftStats {
                        cycles: 1,
                        drafted_tokens: 15,
                        accepted_drafts: 3,
                        emitted_tokens: 4,
                        cycle_duration: Duration::from_millis(30),
                        target_position: 1_004,
                        draft_position: 1_004,
                    },
                )
                .is_none()
        );
        let snapshot = metrics
            .record_draft_progress(
                started + SESSION_METRICS_INTERVAL,
                EngineDraftStats {
                    cycles: 4,
                    drafted_tokens: 60,
                    accepted_drafts: 15,
                    emitted_tokens: 19,
                    cycle_duration: Duration::from_millis(120),
                    target_position: 1_019,
                    draft_position: 1_019,
                },
            )
            .expect("ten-second draft report interval elapsed");

        assert_eq!(snapshot.interval.cycles, 4);
        assert_eq!(snapshot.interval.drafted_tokens, 60);
        assert_eq!(snapshot.interval.accepted_drafts, 15);
        assert_eq!(snapshot.interval.emitted_tokens, 19);
        assert_eq!(snapshot.interval.cycle_duration, Duration::from_millis(120));
        assert_eq!(snapshot.cumulative.target_position, 1_019);
        assert_eq!(snapshot.cumulative.draft_position, 1_019);
        assert_eq!(percentage(15, 60), 25.0);
        assert_eq!(ratio(19, 4), 4.75);
        assert_eq!(average_duration_ms(Duration::from_millis(120), 4), 30.0);
    }

    #[test]
    fn session_metrics_report_prefill_progress_and_rates() {
        let submitted = Instant::now();
        let mut metrics = SessionMetrics::new(submitted, 1_000);
        metrics.record_admission(submitted, 0, 0);
        assert!(metrics.record_prefill_start(submitted));
        assert!(!metrics.record_prefill_start(submitted + Duration::from_secs(1)));
        assert!(
            metrics
                .record_prefill(submitted + Duration::from_secs(5), 100)
                .is_none()
        );

        let first = metrics
            .record_prefill(submitted + Duration::from_secs(10), 300)
            .expect("ten-second report interval elapsed");
        assert_eq!(first.prompt_position, 300);
        assert_eq!(first.interval_tokens_per_second, 30.0);
        assert_eq!(first.compute_tokens_per_second, 30.0);
        assert_eq!(first.effective_tokens_per_second, 30.0);

        let second = metrics
            .record_prefill(submitted + Duration::from_secs(20), 500)
            .expect("second report interval elapsed");
        assert_eq!(second.prompt_position, 500);
        assert_eq!(second.interval_tokens_per_second, 20.0);
        assert_eq!(second.compute_tokens_per_second, 25.0);
        assert_eq!(second.effective_tokens_per_second, 25.0);
    }

    #[test]
    fn session_metrics_exclude_cached_tokens_from_prefill_rates() {
        let submitted = Instant::now();
        let mut metrics = SessionMetrics::new(submitted, 1_000);
        metrics.record_admission(submitted, 256, 0);
        metrics.record_prefill_start(submitted);

        let snapshot = metrics
            .record_prefill(submitted + Duration::from_secs(10), 456)
            .expect("ten-second report interval elapsed");
        assert_eq!(snapshot.prompt_position, 456);
        assert_eq!(snapshot.interval_tokens_per_second, 20.0);
        assert_eq!(snapshot.compute_tokens_per_second, 20.0);
        assert_eq!(snapshot.effective_tokens_per_second, 20.0);
    }

    #[test]
    fn session_metrics_separate_admission_compute_and_effective_prefill() {
        let submitted = Instant::now();
        let admitted = submitted + Duration::from_secs(2);
        let prefill_started = submitted + Duration::from_secs(3);
        let first_token = submitted + Duration::from_secs(5);
        let mut metrics = SessionMetrics::new(submitted, 1_000);
        metrics.record_admission(admitted, 256, 123_456);
        metrics.record_prefill_start(prefill_started);
        metrics.prefilled_tokens = 456;
        metrics.record_token(first_token);

        assert_eq!(metrics.admission_duration(), Duration::from_secs(2));
        assert_eq!(
            metrics.prefill_compute_duration(first_token),
            Duration::from_secs(2)
        );
        assert_eq!(
            metrics.prefill_compute_tokens_per_second(first_token),
            100.0
        );
        assert_eq!(
            metrics.effective_prefill_tokens_per_second(first_token),
            40.0
        );
    }

    #[test]
    fn zero_duration_has_no_rate() {
        assert_eq!(rate(1, Duration::ZERO), 0.0);
    }

    #[test]
    fn checkpoint_architecture_selects_supported_model_families() {
        let directory = std::env::temp_dir().join(format!(
            "eider-actor-model-type-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory).expect("create checkpoint directory");
        fs::write(directory.join("config.json"), r#"{"model_type":"step3p7"}"#)
            .expect("write Step config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Step37
        );
        fs::write(directory.join("config.json"), r#"{"model_type":"bitnet"}"#)
            .expect("write BitNet config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::BitNet
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"bailing_hybrid"}"#,
        )
        .expect("write Ling config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Ling3
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"muse_glimmer"}"#,
        )
        .expect("write Muse Glimmer config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::MuseGlimmer
        );
        fs::write(directory.join("config.json"), r#"{"model_type":"bonsai"}"#)
            .expect("write Bonsai config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Bonsai
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"qwen3_5_moe"}"#,
        )
        .expect("write Qwen config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Qwen36
        );
        fs::write(directory.join("config.json"), r#"{"model_type":"qwen3_5"}"#)
            .expect("write dense Qwen config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Qwen36
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"qwen3_8_flash_next"}"#,
        )
        .expect("write Flash Next config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Qwen38FlashNext
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"nemotron_h"}"#,
        )
        .expect("write Nemotron config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Nemotron3
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"nemotron_h_puzzle"}"#,
        )
        .expect("write Puzzle config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Nemotron3
        );
        fs::write(directory.join("config.json"), r#"{"model_type":"gemma4"}"#)
            .expect("write Gemma config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Gemma4
        );
        fs::write(directory.join("config.json"), r#"{"model_type":"laguna"}"#)
            .expect("write Laguna config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Laguna
        );
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"deepseek_v4"}"#,
        )
        .expect("write DeepSeek V4 config");
        assert_eq!(
            checkpoint_architecture(&directory).unwrap(),
            CheckpointArchitecture::Deepseek4
        );
        fs::remove_dir_all(directory).expect("remove checkpoint directory");
    }
}
