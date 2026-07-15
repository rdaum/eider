//! Dedicated inference thread owning all CUDA and scheduler state.

use crate::protocol::{ApiError, InferenceEvent, InferenceFinished};
use infer::qwen3::qwen36::Qwen36TextModel;
use infer::runtime::chat::CheckpointChatTemplate;
use infer::runtime::generation::GenerationConfig;
use infer::runtime::scheduler::{Qwen36RequestId, Qwen36SchedulerConfig};
use infer::runtime::serving::{Qwen36ChatFinished, Qwen36ChatRequest, Qwen36ChatService};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const SESSION_METRICS_INTERVAL: Duration = Duration::from_secs(10);

/// Model and scheduler configuration loaded by the actor thread.
#[derive(Clone, Debug)]
pub struct InferenceActorConfig {
    pub model_dir: PathBuf,
    pub scheduler: Qwen36SchedulerConfig,
    pub event_capacity: usize,
}

impl InferenceActorConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            scheduler: Qwen36SchedulerConfig::default(),
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
}

enum ActorCommand {
    Submit {
        id: ActorRequestId,
        request: Qwen36ChatRequest,
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
    prompt_tokens: usize,
    sequence_device_bytes: usize,
    prefilled_tokens: usize,
    last_prefill_report_at: Instant,
    last_prefill_report_tokens: usize,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    last_report_at: Option<Instant>,
    last_report_tokens: usize,
    generated_tokens: usize,
}

struct PrefillMetricsSnapshot {
    prompt_position: usize,
    interval_tokens_per_second: f64,
    total_tokens_per_second: f64,
}

struct SessionMetricsSnapshot {
    output_tokens: usize,
    interval_tokens_per_second: f64,
    decode_tokens_per_second: f64,
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
        let model_dir = config.model_dir.clone();
        let scheduler = config.scheduler;
        thread::Builder::new()
            .name("eider-inference".to_string())
            .spawn(move || actor_main(model_dir, scheduler, commands_rx, ready_tx))
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
                event_capacity: config.event_capacity,
            }),
            defaults,
        })
    }

    pub fn generation_defaults(&self) -> &GenerationConfig {
        &self.defaults
    }

    /// Queues a request without blocking an async executor on CUDA work.
    pub fn submit(&self, request: Qwen36ChatRequest) -> Result<ActorResponse, ApiError> {
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
}

impl Drop for ActorInner {
    fn drop(&mut self) {
        let _ = self.commands.send(ActorCommand::Shutdown);
    }
}

fn actor_main(
    model_dir: PathBuf,
    scheduler: Qwen36SchedulerConfig,
    mut commands: mpsc::UnboundedReceiver<ActorCommand>,
    ready: std::sync::mpsc::SyncSender<Result<GenerationConfig, String>>,
) {
    eprintln!("loading Qwen3.6 model from {}", model_dir.display());
    let model = match Qwen36TextModel::open(&model_dir) {
        Ok(model) => model,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    eprintln!("model weights loaded; loading chat template and generation defaults");
    let template = match CheckpointChatTemplate::from_model_dir(&model_dir) {
        Ok(template) => template,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let defaults = match GenerationConfig::from_model_dir(&model_dir) {
        Ok(defaults) => defaults,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    eprintln!(
        "allocating scheduler workspaces: decode={} prefill_sequences={} prefill_tokens={} active={} context={}",
        scheduler.decode_capacity,
        scheduler.prefill_sequence_capacity,
        scheduler.prefill_token_capacity,
        scheduler.max_active_sequences,
        scheduler.max_context_tokens
    );
    let mut service = match Qwen36ChatService::new(&model, &template, scheduler) {
        Ok(service) => service,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    eprintln!("inference actor ready");
    if ready.send(Ok(defaults)).is_err() {
        return;
    }

    let mut active = BTreeMap::<Qwen36RequestId, ActiveRequest>::new();
    let mut scheduler_by_external = BTreeMap::<ActorRequestId, Qwen36RequestId>::new();
    loop {
        if active.is_empty() {
            let Some(command) = commands.blocking_recv() else {
                break;
            };
            if !handle_command(
                command,
                &mut service,
                &mut active,
                &mut scheduler_by_external,
            ) {
                break;
            }
        }

        while let Ok(command) = commands.try_recv() {
            if !handle_command(
                command,
                &mut service,
                &mut active,
                &mut scheduler_by_external,
            ) {
                cancel_all(&mut service, &mut active, &mut scheduler_by_external);
                return;
            }
        }
        if active.is_empty() {
            continue;
        }

        let tick = match service.tick() {
            Ok(tick) => tick,
            Err(error) => {
                let message = error.to_string();
                eprintln!("inference scheduler failed: {message}");
                for request in active.values() {
                    let _ = request
                        .events
                        .try_send(InferenceEvent::Error(message.clone()));
                }
                cancel_all(&mut service, &mut active, &mut scheduler_by_external);
                continue;
            }
        };
        let now = Instant::now();
        for admission in &tick.admitted {
            if let Some(request) = active.get_mut(&admission.request_id) {
                request.metrics.sequence_device_bytes = admission.sequence_device_bytes;
                eprintln!(
                    "session={} admitted state_bytes={} active_sequences={}",
                    request.external_id.0, admission.sequence_device_bytes, tick.active_sequences
                );
            }
        }
        for progress in &tick.prefilled {
            if let Some(request) = active.get_mut(&progress.request_id) {
                let starting = request.metrics.prefilled_tokens == 0;
                let snapshot = request
                    .metrics
                    .record_prefill(now, progress.prompt_position);
                if starting {
                    eprintln!(
                        "session={} prefill_started prompt_tokens={} state_bytes={}",
                        request.external_id.0,
                        request.metrics.prompt_tokens,
                        request.metrics.sequence_device_bytes
                    );
                }
                if let Some(snapshot) = snapshot {
                    eprintln!(
                        "session={} prefill prompt_tokens={}/{} interval_tok_s={:.2} prefill_tok_s={:.2}",
                        request.external_id.0,
                        snapshot.prompt_position,
                        request.metrics.prompt_tokens,
                        snapshot.interval_tokens_per_second,
                        snapshot.total_tokens_per_second
                    );
                }
            }
        }
        for request_id in &tick.generated {
            if let Some(request) = active.get_mut(request_id) {
                let starting = request.metrics.first_token_at.is_none();
                let snapshot = request.metrics.record_token(now);
                if starting {
                    eprintln!(
                        "session={} decoding ttft_ms={:.1} prompt_tokens={} prefill_tok_s={:.2}",
                        request.external_id.0,
                        now.duration_since(request.metrics.submitted_at)
                            .as_secs_f64()
                            * 1000.0,
                        request.metrics.prompt_tokens,
                        request.metrics.prefill_tokens_per_second(now)
                    );
                }
                if let Some(snapshot) = snapshot {
                    eprintln!(
                        "session={} progress output_tokens={} interval_tok_s={:.2} decode_tok_s={:.2}",
                        request.external_id.0,
                        snapshot.output_tokens,
                        snapshot.interval_tokens_per_second,
                        snapshot.decode_tokens_per_second
                    );
                }
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
        for finished in tick.finished {
            if let Some(request) = active.remove(&finished.request_id) {
                scheduler_by_external.remove(&request.external_id);
                let active_requests = active.len();
                request.metrics.log_finished(
                    request.external_id,
                    now,
                    &finished,
                    active_requests,
                    tick.active_sequences,
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
            cancel_scheduler_request(id, &mut service, &mut active, &mut scheduler_by_external);
        }
    }
    cancel_all(&mut service, &mut active, &mut scheduler_by_external);
}

fn handle_command(
    command: ActorCommand,
    service: &mut Qwen36ChatService<'_, '_>,
    active: &mut BTreeMap<Qwen36RequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, Qwen36RequestId>,
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
                eprintln!(
                    "session={} queued prompt_tokens={} max_output_tokens={} active_requests={}",
                    id.0,
                    admission.prompt_tokens,
                    admission.max_output_tokens,
                    active.len()
                );
            }
            Err(error) => {
                eprintln!("failed to admit request {}: {error}", id.0);
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
    scheduler_id: Qwen36RequestId,
    service: &mut Qwen36ChatService<'_, '_>,
    active: &mut BTreeMap<Qwen36RequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, Qwen36RequestId>,
) {
    let outcome = service.cancel_request(scheduler_id);
    let released_sequence_device_bytes = match outcome {
        infer::runtime::scheduler::Qwen36CancelOutcome::Cancelled(cancelled) => {
            cancelled.released_sequence_device_bytes
        }
        infer::runtime::scheduler::Qwen36CancelOutcome::AlreadyFinished
        | infer::runtime::scheduler::Qwen36CancelOutcome::NotFound => 0,
    };
    if let Some(request) = active.remove(&scheduler_id) {
        scheduler_by_external.remove(&request.external_id);
        request.metrics.log_cancelled(
            request.external_id,
            Instant::now(),
            released_sequence_device_bytes,
            active.len(),
            service.active_sequence_count(),
        );
    }
}

fn cancel_all(
    service: &mut Qwen36ChatService<'_, '_>,
    active: &mut BTreeMap<Qwen36RequestId, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, Qwen36RequestId>,
) {
    let ids = active.keys().copied().collect::<Vec<_>>();
    for id in ids {
        cancel_scheduler_request(id, service, active, scheduler_by_external);
    }
}

impl SessionMetrics {
    fn new(submitted_at: Instant, prompt_tokens: usize) -> Self {
        Self {
            submitted_at,
            prompt_tokens,
            sequence_device_bytes: 0,
            prefilled_tokens: 0,
            last_prefill_report_at: submitted_at,
            last_prefill_report_tokens: 0,
            first_token_at: None,
            last_token_at: None,
            last_report_at: None,
            last_report_tokens: 0,
            generated_tokens: 0,
        }
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
            total_tokens_per_second: self.prefill_tokens_per_second(now),
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

    fn log_finished(
        &self,
        id: ActorRequestId,
        now: Instant,
        finished: &Qwen36ChatFinished,
        active_requests: usize,
        active_sequences: usize,
    ) {
        debug_assert_eq!(self.generated_tokens, finished.usage.completion_tokens);
        let time_to_first_token = self.first_token_at.map_or(Duration::ZERO, |first| {
            first.duration_since(self.submitted_at)
        });
        eprintln!(
            "session={} complete prompt_tokens={} output_tokens={} ttft_ms={:.1} decode_tok_s={:.2} total_tok_s={:.2} reason={:?} state_released_bytes={} active_requests={} active_sequences={}",
            id.0,
            finished.usage.prompt_tokens,
            finished.usage.completion_tokens,
            time_to_first_token.as_secs_f64() * 1000.0,
            self.decode_tokens_per_second(),
            rate(
                finished.usage.completion_tokens,
                now.duration_since(self.submitted_at)
            ),
            finished.finish_reason,
            finished.released_sequence_device_bytes,
            active_requests,
            active_sequences
        );
    }

    fn log_cancelled(
        &self,
        id: ActorRequestId,
        now: Instant,
        released_sequence_device_bytes: usize,
        active_requests: usize,
        active_sequences: usize,
    ) {
        eprintln!(
            "session={} cancelled output_tokens={} elapsed_ms={:.1} decode_tok_s={:.2} state_released_bytes={} active_requests={} active_sequences={}",
            id.0,
            self.generated_tokens,
            now.duration_since(self.submitted_at).as_secs_f64() * 1000.0,
            self.decode_tokens_per_second(),
            released_sequence_device_bytes,
            active_requests,
            active_sequences
        );
    }

    fn prefill_tokens_per_second(&self, now: Instant) -> f64 {
        rate(self.prefilled_tokens, now.duration_since(self.submitted_at))
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

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return 0.0;
    }
    tokens as f64 / elapsed.as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn session_metrics_report_prefill_progress_and_rates() {
        let submitted = Instant::now();
        let mut metrics = SessionMetrics::new(submitted, 1_000);
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
        assert_eq!(first.total_tokens_per_second, 30.0);

        let second = metrics
            .record_prefill(submitted + Duration::from_secs(20), 500)
            .expect("second report interval elapsed");
        assert_eq!(second.prompt_position, 500);
        assert_eq!(second.interval_tokens_per_second, 20.0);
        assert_eq!(second.total_tokens_per_second, 25.0);
    }

    #[test]
    fn zero_duration_has_no_rate() {
        assert_eq!(rate(1, Duration::ZERO), 0.0);
    }
}
