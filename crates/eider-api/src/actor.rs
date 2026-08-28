//! Dedicated inference thread owning all CUDA and scheduler state.

use crate::metrics::{FinishReason, ServerEndpoint, metrics as server_metrics};
use crate::protocol::{ApiError, InferenceEvent, InferenceFinished};
use eider_inference::InferenceResult;
use eider_inference::bitnet::BitNetModel;
use eider_inference::bonsai::{BonsaiModel, load_chat_template as bonsai_chat_template};
use eider_inference::deepseek4::Deepseek4TextModel;
use eider_inference::execution::bitnet_serving::{
    BitNetAdmissionProgress, BitNetCancelOutcome, BitNetChatService, BitNetRequestId,
};
use eider_inference::execution::bonsai_serving::{
    BonsaiAdmissionProgress, BonsaiCancelOutcome, BonsaiChatService, BonsaiRequestId,
};
use eider_inference::execution::deepseek4_serving::{
    Deepseek4AdmissionProgress, Deepseek4CancelOutcome, Deepseek4ChatService, Deepseek4RequestId,
    Deepseek4SpeculativeProgress,
};
use eider_inference::execution::gemma4_serving::{
    Gemma4AdmissionProgress, Gemma4CancelOutcome, Gemma4ChatService, Gemma4RequestId,
};
use eider_inference::execution::laguna_serving::{
    LagunaAdmissionProgress, LagunaCancelOutcome, LagunaChatService, LagunaRequestId,
};
use eider_inference::execution::ling3_serving::{
    Ling3AdmissionProgress, Ling3CancelOutcome, Ling3ChatService, Ling3RequestId,
};
use eider_inference::execution::muse_glimmer_serving::{
    MuseGlimmerAdmissionProgress, MuseGlimmerCancelOutcome, MuseGlimmerChatService,
    MuseGlimmerDFlashProgress, MuseGlimmerDFlashStats, MuseGlimmerRequestId,
};
use eider_inference::execution::nemotron3_serving::{
    Nemotron3AdmissionProgress, Nemotron3CancelOutcome, Nemotron3ChatService, Nemotron3RequestId,
};
use eider_inference::execution::qwen38_flash_next_serving::{
    Qwen38FlashNextAdmissionProgress, Qwen38FlashNextCancelOutcome, Qwen38FlashNextChatService,
    Qwen38FlashNextRequestId,
};
use eider_inference::execution::scheduler::{
    Qwen36AdmissionProgress, Qwen36CancelOutcome, Qwen36RequestId, Qwen38SpeculativeProgress,
};
use eider_inference::execution::serving::Qwen36ChatService;
use eider_inference::execution::step37_scheduler::{
    Step37AdmissionProgress, Step37CancelOutcome, Step37RequestId,
};
use eider_inference::execution::step37_serving::Step37ChatService;
use eider_inference::gemma4::Gemma4Model;
use eider_inference::laguna::LagunaModel;
use eider_inference::ling3::Ling3Model;
use eider_inference::metrics::metrics as infer_metrics;
use eider_inference::muse_glimmer::MuseGlimmerModel;
use eider_inference::nemotron3::{Nemotron3Model, Nemotron3StorageConfig};
use eider_inference::qwen3::qwen36::{Qwen36Bf16StorageConfig, Qwen36Fp8Storage, Qwen36TextModel};
use eider_inference::qwen38_flash_next::Qwen38FlashNextModel;
use eider_inference::step37::{Step37Bf16StorageConfig, Step37TextModel};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::chat_output::ChatOutputEvent;
use eider_runtime::generation::GenerationConfig;
use eider_runtime::request::{ChatFinishReason, ChatRequest, ChatUsage};
use eider_runtime::scheduler::{RequestLifecycleEvent, SchedulerConfig};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

const SESSION_METRICS_INTERVAL: Duration = Duration::from_secs(10);

/// Model and scheduler configuration loaded by the actor thread.
#[derive(Clone, Debug)]
pub struct InferenceActorConfig {
    pub model_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub dflash_gguf: Option<PathBuf>,
    pub dflash2_dir: Option<PathBuf>,
    pub scheduler: SchedulerConfig,
    pub sequence_cache: SequenceCacheConfig,
    pub qwen_bf16_storage: Qwen36Bf16StorageConfig,
    pub qwen_fp8_attention_storage: Qwen36Fp8Storage,
    pub qwen_fp8_dense_mlp_storage: Qwen36Fp8Storage,
    pub qwen_fp8_lm_head_storage: Qwen36Fp8Storage,
    pub step_expert_capacity: usize,
    pub deepseek_expert_capacity: usize,
    pub step_bf16_storage: Step37Bf16StorageConfig,
    pub nemotron_storage: Nemotron3StorageConfig,
    pub event_capacity: usize,
}

impl InferenceActorConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        Self {
            artifact_dir: model_dir.join(".eider-cache"),
            dflash_gguf: None,
            dflash2_dir: None,
            model_dir,
            scheduler: SchedulerConfig::default(),
            sequence_cache: SequenceCacheConfig::default(),
            qwen_bf16_storage: Qwen36Bf16StorageConfig::default(),
            qwen_fp8_attention_storage: Qwen36Fp8Storage::default(),
            qwen_fp8_dense_mlp_storage: Qwen36Fp8Storage::default(),
            qwen_fp8_lm_head_storage: Qwen36Fp8Storage::default(),
            step_expert_capacity: 240,
            deepseek_expert_capacity: 8,
            step_bf16_storage: Step37Bf16StorageConfig::default(),
            nemotron_storage: Nemotron3StorageConfig::default(),
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
    qwen38_speculative_cycles: usize,
    qwen38_accepted_drafts: usize,
    dflash: Option<DFlashSessionMetrics>,
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

struct DFlashSessionMetrics {
    cumulative: MuseGlimmerDFlashStats,
    last_report_at: Instant,
    last_report: MuseGlimmerDFlashStats,
}

struct DFlashMetricsSnapshot {
    interval: MuseGlimmerDFlashStats,
    cumulative: MuseGlimmerDFlashStats,
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
    let InferenceActorConfig {
        model_dir,
        artifact_dir,
        dflash_gguf,
        dflash2_dir,
        scheduler,
        sequence_cache,
        qwen_bf16_storage,
        qwen_fp8_attention_storage,
        qwen_fp8_dense_mlp_storage,
        qwen_fp8_lm_head_storage,
        step_expert_capacity,
        deepseek_expert_capacity,
        step_bf16_storage,
        nemotron_storage,
        ..
    } = config;
    let architecture = match checkpoint_architecture(&model_dir) {
        Ok(architecture) => architecture,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let template: std::result::Result<_, String> = match architecture {
        CheckpointArchitecture::Bonsai => bonsai_chat_template(&model_dir),
        _ => CheckpointChatTemplate::from_model_dir(&model_dir).map_err(|error| error.to_string()),
    };
    let template = match template {
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
    info!(
        decode_capacity = scheduler.decode_capacity,
        prefill_sequence_capacity = scheduler.prefill_sequence_capacity,
        prefill_token_capacity = scheduler.prefill_token_capacity,
        max_active_sequences = scheduler.max_active_sequences,
        max_context_tokens = scheduler.max_context_tokens,
        speculative_drafts = scheduler.speculative_drafts,
        "allocating scheduler workspaces"
    );

    match architecture {
        CheckpointArchitecture::BitNet => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.0;
            info!(model_dir = %model_dir.display(), "loading BitNet model");
            let model = match BitNetModel::load(&model_dir) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let bitnet_scheduler = SchedulerConfig {
                max_context_tokens: scheduler.max_context_tokens.min(model.config().max_context),
                ..scheduler
            };
            let service = match BitNetChatService::new(&model, &template, bitnet_scheduler) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = BitNetActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Ling3 => {
            info!(model_dir = %model_dir.display(), "loading Ling 3 model");
            let model = match Ling3Model::load(&model_dir) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            info!(
                device_weights_gib = model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded Ling 3 text model"
            );
            let ling_scheduler = SchedulerConfig {
                max_context_tokens: scheduler.max_context_tokens.min(model.max_context_tokens()),
                ..scheduler
            };
            let service = match Ling3ChatService::new(&model, &template, ling_scheduler) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = Ling3ActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::MuseGlimmer => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.0;
            info!(
                model_dir = %model_dir.display(),
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                "loading Muse Glimmer model"
            );
            let model_result = match dflash_gguf {
                Some(path) => MuseGlimmerModel::load_with_dflash(&model_dir, path),
                None => MuseGlimmerModel::load(&model_dir),
            };
            let model = match model_result {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            info!(
                device_weights_gib = model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded Muse Glimmer text model"
            );
            let muse_scheduler = SchedulerConfig {
                max_context_tokens: scheduler
                    .max_context_tokens
                    .min(model.config().max_position_embeddings),
                ..scheduler
            };
            let service = match MuseGlimmerChatService::new_with_cache_config(
                &model,
                &template,
                muse_scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = MuseGlimmerActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Bonsai => {
            let gguf = bonsai_gguf_path(&model_dir);
            info!(gguf = %gguf.display(), "loading Ternary Bonsai model");
            let model = match BonsaiModel::load(&gguf) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let bonsai_scheduler = SchedulerConfig {
                max_context_tokens: scheduler.max_context_tokens.min(model.config().max_context),
                ..scheduler
            };
            let service = match BonsaiChatService::new(&model, &template, bonsai_scheduler) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = BonsaiActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Qwen36 => {
            let mut defaults = defaults;
            info!(
                model_dir = %model_dir.display(),
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                bf16_storage = ?qwen_bf16_storage,
                native_fp8_attention_storage = ?qwen_fp8_attention_storage,
                native_fp8_dense_mlp_storage = ?qwen_fp8_dense_mlp_storage,
                native_fp8_lm_head_storage = ?qwen_fp8_lm_head_storage,
                "loading Qwen hybrid model"
            );
            let mut model = match Qwen36TextModel::open_with_fp8_storage_and_artifact_dir(
                &model_dir,
                &artifact_dir,
                qwen_bf16_storage,
                qwen_fp8_attention_storage,
                qwen_fp8_dense_mlp_storage,
                qwen_fp8_lm_head_storage,
            ) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            if scheduler.speculative_drafts > 0
                && let Some(dflash2_dir) = dflash2_dir
                && let Err(error) = model.enable_dflash2(&dflash2_dir)
            {
                let _ = ready.send(Err(error.to_string()));
                return;
            }
            if (model.dflash2_enabled() || model.mtp_weights().is_some())
                && scheduler.speculative_drafts > 0
            {
                // Qwen3.8 speculative verification is exact only for greedy decoding.
                // Enabling speculative drafts opts omitted request sampling
                // into that path; explicit API sampling still takes priority.
                defaults.sampling.temperature = 0.0;
            }
            let service = match Qwen36ChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = QwenActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Qwen38FlashNext => {
            let mut defaults = defaults;
            info!(
                model_dir = %model_dir.display(),
                artifact_dir = %artifact_dir.display(),
                attention_backend = "native-qsa",
                max_context_tokens = scheduler.max_context_tokens,
                "loading Qwen3.8 Flash Next model"
            );
            let mut model = match Qwen38FlashNextModel::open(&model_dir, &artifact_dir) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            if scheduler.speculative_drafts > 0 {
                if let Err(error) = model.enable_mtp() {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
                defaults.sampling.temperature = 0.0;
            }
            let qsa_scheduler = SchedulerConfig {
                max_context_tokens: scheduler
                    .max_context_tokens
                    .min(model.config().max_position_embeddings),
                ..scheduler
            };
            let service = match Qwen38FlashNextChatService::new_with_cache_config(
                model,
                &template,
                qsa_scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            info!(
                attention_backend = "native-qsa",
                max_context_tokens = qsa_scheduler.max_context_tokens,
                "loaded Qwen3.8 Flash Next text model"
            );
            let mut service = Qwen38FlashNextActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Step37 => {
            info!(
                model_dir = %model_dir.display(),
                expert_capacity = step_expert_capacity,
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                bf16_storage = ?step_bf16_storage,
                "loading Step-3.7 model"
            );
            let model = match Step37TextModel::open_with_bf16_storage_and_artifact_dir(
                &model_dir,
                &artifact_dir,
                step_expert_capacity,
                step_bf16_storage,
            ) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let service = match Step37ChatService::new_with_cache_config(
                model,
                &template,
                scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = StepActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Nemotron3 => {
            let mut defaults = defaults;
            // The MTP path is exact only for greedy decoding. Checkpoint
            // sampling defaults describe an offline generation policy, while
            // interactive API serving should use the fast greedy path unless
            // the request explicitly overrides temperature or top-k.
            defaults.sampling.temperature = 0.0;
            info!(
                model_dir = %model_dir.display(),
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                storage = ?nemotron_storage,
                "loading Nemotron 3 model"
            );
            let model = match Nemotron3Model::load_with_storage(&model_dir, nemotron_storage) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let service = match Nemotron3ChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = NemotronActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Gemma4 => {
            let mut defaults = defaults;
            // Gemma's stochastic checkpoint defaults can fall into repetitive
            // reasoning during interactive tool use. Prefer greedy serving
            // unless the request explicitly supplies sampling parameters.
            defaults.sampling.temperature = 0.0;
            info!(
                model_dir = %model_dir.display(),
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                "loading Gemma 4 model"
            );
            let model = match Gemma4Model::load(&model_dir) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            info!(
                device_weights_gib = model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded Gemma 4 text model"
            );
            let service = match Gemma4ChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = GemmaActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Laguna => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.7;
            defaults.sampling.top_k = 20;
            defaults.sampling.top_p = 0.95;
            info!(
                model_dir = %model_dir.display(),
                artifact_dir = %artifact_dir.display(),
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                "loading Laguna-S-2.1 model"
            );
            let model = match LagunaModel::load_with_artifact_dir(&model_dir, &artifact_dir) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let service = match LagunaChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = LagunaActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
        CheckpointArchitecture::Deepseek4 => {
            let mut defaults = defaults;
            if scheduler.speculative_drafts > 0 {
                defaults.sampling.temperature = 0.0;
            }
            info!(
                model_dir = %model_dir.display(),
                expert_store_dir = %artifact_dir.display(),
                expert_capacity = deepseek_expert_capacity,
                retained_prefix_bytes = sequence_cache.max_retained_bytes,
                "loading DeepSeek V4 model"
            );
            let model = match if scheduler.speculative_drafts > 0 {
                Deepseek4TextModel::load_paged_nvfp4_with_mtp(
                    &model_dir,
                    &artifact_dir,
                    deepseek_expert_capacity,
                )
            } else {
                Deepseek4TextModel::load_paged_nvfp4(
                    &model_dir,
                    &artifact_dir,
                    deepseek_expert_capacity,
                )
            } {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            info!(
                device_weights_gib = model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded DeepSeek V4 text model"
            );
            let service = match Deepseek4ChatService::new_with_cache_config(
                model,
                &template,
                scheduler,
                sequence_cache,
            ) {
                Ok(service) => service,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut service = DeepseekActorService::new(service);
            run_actor_loop(&mut service, &mut commands, ready, defaults);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointArchitecture {
    BitNet,
    Ling3,
    MuseGlimmer,
    Bonsai,
    Qwen36,
    Qwen38FlashNext,
    Step37,
    Nemotron3,
    Gemma4,
    Laguna,
    Deepseek4,
}

#[derive(Deserialize)]
struct CheckpointConfig {
    model_type: String,
}

fn checkpoint_architecture(model_dir: &std::path::Path) -> Result<CheckpointArchitecture, String> {
    let path = model_dir.join("config.json");
    let contents = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let config: CheckpointConfig = serde_json::from_str(&contents)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    match config.model_type.as_str() {
        "bitnet" => Ok(CheckpointArchitecture::BitNet),
        "bailing_hybrid" => Ok(CheckpointArchitecture::Ling3),
        "muse_glimmer" => Ok(CheckpointArchitecture::MuseGlimmer),
        "bonsai" => Ok(CheckpointArchitecture::Bonsai),
        "qwen3_5" | "qwen3_5_moe" => Ok(CheckpointArchitecture::Qwen36),
        "qwen3_8_flash_next" => Ok(CheckpointArchitecture::Qwen38FlashNext),
        "step3p7" => Ok(CheckpointArchitecture::Step37),
        "nemotron_h" | "nemotron_h_puzzle" => Ok(CheckpointArchitecture::Nemotron3),
        "gemma4" => Ok(CheckpointArchitecture::Gemma4),
        "laguna" => Ok(CheckpointArchitecture::Laguna),
        "deepseek_v4" => Ok(CheckpointArchitecture::Deepseek4),
        other => Err(format!(
            "unsupported model_type {other:?} in {}",
            path.display()
        )),
    }
}

fn bonsai_gguf_path(model_dir: &std::path::Path) -> PathBuf {
    model_dir.join("Ternary-Bonsai-8B-Q2_0_g64.gguf")
}

struct EngineAdmission {
    request_id: u64,
    prompt_tokens: usize,
    max_output_tokens: usize,
}

struct EngineAdmissionProgress {
    request_id: u64,
    sequence_device_bytes: usize,
    cached_prompt_tokens: usize,
    allocation_duration: Duration,
    checkpoint_copy_duration: Duration,
    admitted_after_tick_start: Duration,
}

fn qwen_admission_progress(progress: Qwen36AdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: Duration::ZERO,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn qwen38_speculative_progress(
    progress: Qwen38SpeculativeProgress,
) -> EngineQwen38SpeculativeProgress {
    EngineQwen38SpeculativeProgress {
        request_id: progress.request_id.get(),
        cycles: progress.cycles,
        accepted_drafts: progress.accepted_drafts,
    }
}

fn deepseek4_speculative_progress(
    progress: Deepseek4SpeculativeProgress,
) -> EngineQwen38SpeculativeProgress {
    EngineQwen38SpeculativeProgress {
        request_id: progress.request_id.get(),
        cycles: progress.cycles,
        accepted_drafts: progress.accepted_drafts,
    }
}

fn bitnet_admission_progress(progress: BitNetAdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: Duration::ZERO,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn ling3_admission_progress(progress: Ling3AdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: Duration::ZERO,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn muse_admission_progress(progress: MuseGlimmerAdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: progress.allocation_duration,
        checkpoint_copy_duration: progress.checkpoint_copy_duration,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn muse_dflash_progress(progress: MuseGlimmerDFlashProgress) -> EngineDFlashProgress {
    EngineDFlashProgress {
        request_id: progress.request_id.get(),
        stats: progress.stats,
    }
}

fn bonsai_admission_progress(progress: BonsaiAdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: Duration::ZERO,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn step_admission_progress(progress: Step37AdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: Duration::ZERO,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn nemotron_admission_progress(progress: Nemotron3AdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: Duration::ZERO,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn gemma_admission_progress(progress: Gemma4AdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: progress.allocation_duration,
        checkpoint_copy_duration: progress.checkpoint_copy_duration,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn laguna_admission_progress(progress: LagunaAdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: progress.allocation_duration,
        checkpoint_copy_duration: progress.checkpoint_copy_duration,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn deepseek_admission_progress(progress: Deepseek4AdmissionProgress) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: progress.allocation_duration,
        checkpoint_copy_duration: progress.checkpoint_copy_duration,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

fn qwen38_flash_next_admission_progress(
    progress: Qwen38FlashNextAdmissionProgress,
) -> EngineAdmissionProgress {
    EngineAdmissionProgress {
        request_id: progress.request_id.get(),
        sequence_device_bytes: progress.sequence_device_bytes,
        cached_prompt_tokens: progress.cached_prompt_tokens,
        allocation_duration: progress.allocation_duration,
        checkpoint_copy_duration: Duration::ZERO,
        admitted_after_tick_start: progress.admitted_after_tick_start,
    }
}

enum EngineLifecycleEvent {
    Admitted(EngineAdmissionProgress),
    PrefillStarted(u64),
}

struct EnginePrefillProgress {
    request_id: u64,
    prompt_position: usize,
}

struct EngineDelta {
    request_id: u64,
    event: ChatOutputEvent,
}

struct EngineFinished {
    request_id: u64,
    finish_reason: ChatFinishReason,
    usage: ChatUsage,
    released_sequence_device_bytes: usize,
}

struct EngineDFlashProgress {
    request_id: u64,
    stats: MuseGlimmerDFlashStats,
}

struct EngineQwen38SpeculativeProgress {
    request_id: u64,
    cycles: usize,
    accepted_drafts: usize,
}

#[derive(Default)]
struct EngineTick {
    prefilled: Vec<EnginePrefillProgress>,
    generated: Vec<u64>,
    qwen38_speculative: Vec<EngineQwen38SpeculativeProgress>,
    dflash: Vec<EngineDFlashProgress>,
    output: Vec<EngineDelta>,
    finished: Vec<EngineFinished>,
    active_sequences: usize,
}

enum EngineCancelOutcome {
    Cancelled {
        released_sequence_device_bytes: usize,
    },
    AlreadyFinished,
    NotFound,
}

trait ActorService {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission>;
    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick>;
    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome;
    fn active_sequence_count(&self) -> usize;
    fn shutdown(&mut self) -> InferenceResult<()> {
        Ok(())
    }
}

struct QwenActorService<'model, 'template> {
    inner: Qwen36ChatService<'model, 'template>,
    ids: BTreeMap<u64, Qwen36RequestId>,
}

impl<'model, 'template> QwenActorService<'model, 'template> {
    fn new(inner: Qwen36ChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for QwenActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<Qwen36RequestId, Qwen36AdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(qwen_admission_progress(
                        progress,
                    )));
                }
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Qwen36RequestId::get)
                .collect(),
            qwen38_speculative: tick
                .speculative
                .into_iter()
                .map(qwen38_speculative_progress)
                .collect(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Qwen36CancelOutcome::Cancelled(cancelled) => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: cancelled.released_sequence_device_bytes,
            },
            Qwen36CancelOutcome::AlreadyFinished => EngineCancelOutcome::AlreadyFinished,
            Qwen36CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct Qwen38FlashNextActorService<'template> {
    inner: Qwen38FlashNextChatService<'template>,
    ids: BTreeMap<u64, Qwen38FlashNextRequestId>,
}

impl<'template> Qwen38FlashNextActorService<'template> {
    fn new(inner: Qwen38FlashNextChatService<'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for Qwen38FlashNextActorService<'_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer = |event: RequestLifecycleEvent<
            Qwen38FlashNextRequestId,
            Qwen38FlashNextAdmissionProgress,
        >| match event {
            RequestLifecycleEvent::Admitted(progress) => on_lifecycle(
                EngineLifecycleEvent::Admitted(qwen38_flash_next_admission_progress(progress)),
            ),
            RequestLifecycleEvent::PrefillStarted(id) => {
                on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
            }
        };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Qwen38FlashNextRequestId::get)
                .collect(),
            qwen38_speculative: tick
                .speculative
                .into_iter()
                .map(|progress| EngineQwen38SpeculativeProgress {
                    request_id: progress.request_id.get(),
                    cycles: progress.cycles,
                    accepted_drafts: progress.accepted_drafts,
                })
                .collect(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Qwen38FlashNextCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Qwen38FlashNextCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct StepActorService<'template> {
    inner: Step37ChatService<'template>,
    ids: BTreeMap<u64, Step37RequestId>,
}

impl<'template> StepActorService<'template> {
    fn new(inner: Step37ChatService<'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for StepActorService<'_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<Step37RequestId, Step37AdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(step_admission_progress(
                        progress,
                    )));
                }
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Step37RequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Step37CancelOutcome::Cancelled(cancelled) => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: cancelled.released_sequence_device_bytes,
            },
            Step37CancelOutcome::AlreadyFinished => EngineCancelOutcome::AlreadyFinished,
            Step37CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct NemotronActorService<'model, 'template> {
    inner: Nemotron3ChatService<'model, 'template>,
    ids: BTreeMap<u64, Nemotron3RequestId>,
}

impl<'model, 'template> NemotronActorService<'model, 'template> {
    fn new(inner: Nemotron3ChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for NemotronActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer = |event: RequestLifecycleEvent<
            Nemotron3RequestId,
            Nemotron3AdmissionProgress,
        >| match event {
            RequestLifecycleEvent::Admitted(progress) => {
                on_lifecycle(EngineLifecycleEvent::Admitted(nemotron_admission_progress(
                    progress,
                )));
            }
            RequestLifecycleEvent::PrefillStarted(id) => {
                on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
            }
        };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Nemotron3RequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Nemotron3CancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Nemotron3CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct GemmaActorService<'model, 'template> {
    inner: Gemma4ChatService<'model, 'template>,
    ids: BTreeMap<u64, Gemma4RequestId>,
}

impl<'model, 'template> GemmaActorService<'model, 'template> {
    fn new(inner: Gemma4ChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for GemmaActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<Gemma4RequestId, Gemma4AdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(gemma_admission_progress(
                        progress,
                    )));
                }
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Gemma4RequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Gemma4CancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Gemma4CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct BitNetActorService<'model, 'template> {
    inner: BitNetChatService<'model, 'template>,
    ids: BTreeMap<u64, BitNetRequestId>,
}

impl<'model, 'template> BitNetActorService<'model, 'template> {
    fn new(inner: BitNetChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for BitNetActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<BitNetRequestId, BitNetAdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => on_lifecycle(
                    EngineLifecycleEvent::Admitted(bitnet_admission_progress(progress)),
                ),
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(BitNetRequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            BitNetCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            BitNetCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct Ling3ActorService<'model, 'template> {
    inner: Ling3ChatService<'model, 'template>,
    ids: BTreeMap<u64, Ling3RequestId>,
}

impl<'model, 'template> Ling3ActorService<'model, 'template> {
    fn new(inner: Ling3ChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for Ling3ActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<Ling3RequestId, Ling3AdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => on_lifecycle(
                    EngineLifecycleEvent::Admitted(ling3_admission_progress(progress)),
                ),
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Ling3RequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Ling3CancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Ling3CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct MuseGlimmerActorService<'model, 'template> {
    inner: MuseGlimmerChatService<'model, 'template>,
    ids: BTreeMap<u64, MuseGlimmerRequestId>,
}

impl<'model, 'template> MuseGlimmerActorService<'model, 'template> {
    fn new(inner: MuseGlimmerChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for MuseGlimmerActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer = |event: RequestLifecycleEvent<
            MuseGlimmerRequestId,
            MuseGlimmerAdmissionProgress,
        >| match event {
            RequestLifecycleEvent::Admitted(progress) => on_lifecycle(
                EngineLifecycleEvent::Admitted(muse_admission_progress(progress)),
            ),
            RequestLifecycleEvent::PrefillStarted(id) => {
                on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
            }
        };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(MuseGlimmerRequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: tick.dflash.into_iter().map(muse_dflash_progress).collect(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            MuseGlimmerCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            MuseGlimmerCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct BonsaiActorService<'model, 'template> {
    inner: BonsaiChatService<'model, 'template>,
    ids: BTreeMap<u64, BonsaiRequestId>,
}

impl<'model, 'template> BonsaiActorService<'model, 'template> {
    fn new(inner: BonsaiChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for BonsaiActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<BonsaiRequestId, BonsaiAdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => on_lifecycle(
                    EngineLifecycleEvent::Admitted(bonsai_admission_progress(progress)),
                ),
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(BonsaiRequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            BonsaiCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            BonsaiCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct LagunaActorService<'model, 'template> {
    inner: LagunaChatService<'model, 'template>,
    ids: BTreeMap<u64, LagunaRequestId>,
}

impl<'model, 'template> LagunaActorService<'model, 'template> {
    fn new(inner: LagunaChatService<'model, 'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for LagunaActorService<'_, '_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer =
            |event: RequestLifecycleEvent<LagunaRequestId, LagunaAdmissionProgress>| match event {
                RequestLifecycleEvent::Admitted(progress) => {
                    on_lifecycle(EngineLifecycleEvent::Admitted(laguna_admission_progress(
                        progress,
                    )));
                }
                RequestLifecycleEvent::PrefillStarted(id) => {
                    on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
                }
            };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(LagunaRequestId::get)
                .collect(),
            qwen38_speculative: Vec::new(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            LagunaCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            LagunaCancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

struct DeepseekActorService<'template> {
    inner: Deepseek4ChatService<'template>,
    ids: BTreeMap<u64, Deepseek4RequestId>,
}

impl<'template> DeepseekActorService<'template> {
    fn new(inner: Deepseek4ChatService<'template>) -> Self {
        Self {
            inner,
            ids: BTreeMap::new(),
        }
    }
}

impl ActorService for DeepseekActorService<'_> {
    fn add_request(&mut self, request: ChatRequest) -> InferenceResult<EngineAdmission> {
        let admission = self.inner.add_request(request)?;
        let id = admission.request_id.get();
        self.ids.insert(id, admission.request_id);
        Ok(EngineAdmission {
            request_id: id,
            prompt_tokens: admission.prompt_tokens,
            max_output_tokens: admission.max_output_tokens,
        })
    }

    fn tick(
        &mut self,
        on_lifecycle: &mut dyn FnMut(EngineLifecycleEvent),
    ) -> InferenceResult<EngineTick> {
        let mut observer = |event: RequestLifecycleEvent<
            Deepseek4RequestId,
            Deepseek4AdmissionProgress,
        >| match event {
            RequestLifecycleEvent::Admitted(progress) => {
                on_lifecycle(EngineLifecycleEvent::Admitted(deepseek_admission_progress(
                    progress,
                )));
            }
            RequestLifecycleEvent::PrefillStarted(id) => {
                on_lifecycle(EngineLifecycleEvent::PrefillStarted(id.get()));
            }
        };
        let tick = self.inner.tick_with_lifecycle(&mut observer)?;
        let finished_ids = tick
            .finished
            .iter()
            .map(|finished| finished.request_id.get())
            .collect::<Vec<_>>();
        let converted = EngineTick {
            prefilled: tick
                .prefilled
                .into_iter()
                .map(|progress| EnginePrefillProgress {
                    request_id: progress.request_id.get(),
                    prompt_position: progress.prompt_position,
                })
                .collect(),
            generated: tick
                .generated
                .into_iter()
                .map(Deepseek4RequestId::get)
                .collect(),
            qwen38_speculative: tick
                .speculative
                .into_iter()
                .map(deepseek4_speculative_progress)
                .collect(),
            dflash: Vec::new(),
            output: tick
                .output
                .into_iter()
                .map(|delta| EngineDelta {
                    request_id: delta.request_id.get(),
                    event: delta.event,
                })
                .collect(),
            finished: tick
                .finished
                .into_iter()
                .map(|finished| EngineFinished {
                    request_id: finished.request_id.get(),
                    finish_reason: finished.finish_reason,
                    usage: finished.usage,
                    released_sequence_device_bytes: finished.released_sequence_device_bytes,
                })
                .collect(),
            active_sequences: tick.active_sequences,
        };
        for id in finished_ids {
            self.ids.remove(&id);
        }
        Ok(converted)
    }

    fn cancel_request(&mut self, id: u64) -> EngineCancelOutcome {
        let Some(inner_id) = self.ids.remove(&id) else {
            return EngineCancelOutcome::NotFound;
        };
        match self.inner.cancel_request(inner_id) {
            Deepseek4CancelOutcome::Cancelled {
                released_sequence_device_bytes,
            } => EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes,
            },
            Deepseek4CancelOutcome::NotFound => EngineCancelOutcome::NotFound,
        }
    }

    fn active_sequence_count(&self) -> usize {
        self.inner.active_sequence_count()
    }
}

fn run_actor_loop(
    service: &mut dyn ActorService,
    commands: &mut mpsc::UnboundedReceiver<ActorCommand>,
    ready: std::sync::mpsc::SyncSender<Result<GenerationConfig, String>>,
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

    let mut active = BTreeMap::<u64, ActiveRequest>::new();
    let mut scheduler_by_external = BTreeMap::<ActorRequestId, u64>::new();
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
        for progress in &tick.qwen38_speculative {
            if let Some(request) = active.get_mut(&progress.request_id) {
                request.metrics.record_qwen38_speculative(progress);
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
                        speculative_cycles = request.metrics.qwen38_speculative_cycles,
                        accepted_drafts = request.metrics.qwen38_accepted_drafts,
                        accepted_drafts_per_cycle = ratio(
                            request.metrics.qwen38_accepted_drafts,
                            request.metrics.qwen38_speculative_cycles
                        ),
                        "decode progress"
                    );
                }
            }
        }
        for progress in &tick.dflash {
            if let Some(request) = active.get_mut(&progress.request_id)
                && let Some(snapshot) = request.metrics.record_dflash(now, progress.stats)
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
            cancel_scheduler_request(id, service, &mut active, &mut scheduler_by_external);
        }
        update_current_counts(service, &active);
    }
    cancel_all(service, &mut active, &mut scheduler_by_external);
    shutdown_service(service);
}

fn shutdown_service(service: &mut dyn ActorService) {
    if let Err(error) = service.shutdown() {
        error!(error = %error, "failed to shut down inference service");
    }
}

fn handle_command(
    command: ActorCommand,
    service: &mut dyn ActorService,
    active: &mut BTreeMap<u64, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, u64>,
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
    scheduler_id: u64,
    service: &mut dyn ActorService,
    active: &mut BTreeMap<u64, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, u64>,
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
    service: &mut dyn ActorService,
    active: &mut BTreeMap<u64, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, u64>,
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
    service: &mut dyn ActorService,
    active: &mut BTreeMap<u64, ActiveRequest>,
    scheduler_by_external: &mut BTreeMap<ActorRequestId, u64>,
) {
    let ids = active.keys().copied().collect::<Vec<_>>();
    for id in ids {
        cancel_scheduler_request(id, service, active, scheduler_by_external);
    }
}

fn update_current_counts(service: &dyn ActorService, active: &BTreeMap<u64, ActiveRequest>) {
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
            qwen38_speculative_cycles: 0,
            qwen38_accepted_drafts: 0,
            dflash: None,
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

    fn record_qwen38_speculative(&mut self, progress: &EngineQwen38SpeculativeProgress) {
        self.qwen38_speculative_cycles += progress.cycles;
        self.qwen38_accepted_drafts += progress.accepted_drafts;
    }

    fn record_dflash(
        &mut self,
        now: Instant,
        stats: MuseGlimmerDFlashStats,
    ) -> Option<DFlashMetricsSnapshot> {
        let Some(dflash) = &mut self.dflash else {
            self.dflash = Some(DFlashSessionMetrics {
                cumulative: stats,
                last_report_at: now,
                last_report: MuseGlimmerDFlashStats::default(),
            });
            return None;
        };
        dflash.cumulative = stats;
        if now.duration_since(dflash.last_report_at) < SESSION_METRICS_INTERVAL {
            return None;
        }
        let snapshot = DFlashMetricsSnapshot {
            interval: dflash_stats_delta(stats, dflash.last_report),
            cumulative: stats,
        };
        dflash.last_report_at = now;
        dflash.last_report = stats;
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
            speculative_cycles = self.qwen38_speculative_cycles,
            accepted_drafts = self.qwen38_accepted_drafts,
            accepted_drafts_per_cycle = ratio(
                self.qwen38_accepted_drafts,
                self.qwen38_speculative_cycles
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
        if let Some(dflash) = &self.dflash {
            log_dflash_summary(id, dflash.cumulative);
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

impl DFlashMetricsSnapshot {
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
            dflash_position = self.cumulative.dflash_position,
            "DFlash progress"
        );
    }
}

fn log_dflash_summary(id: ActorRequestId, stats: MuseGlimmerDFlashStats) {
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
        dflash_position = stats.dflash_position,
        "DFlash session complete"
    );
}

fn dflash_stats_delta(
    current: MuseGlimmerDFlashStats,
    previous: MuseGlimmerDFlashStats,
) -> MuseGlimmerDFlashStats {
    MuseGlimmerDFlashStats {
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
        dflash_position: current.dflash_position,
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
    use std::fs;

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
    fn session_metrics_accumulate_qwen38_speculative_acceptance() {
        let mut metrics = SessionMetrics::new(Instant::now(), 8);
        metrics.record_qwen38_speculative(&EngineQwen38SpeculativeProgress {
            request_id: 7,
            cycles: 1,
            accepted_drafts: 2,
        });
        metrics.record_qwen38_speculative(&EngineQwen38SpeculativeProgress {
            request_id: 7,
            cycles: 1,
            accepted_drafts: 1,
        });

        assert_eq!(metrics.qwen38_speculative_cycles, 2);
        assert_eq!(metrics.qwen38_accepted_drafts, 3);
        assert_eq!(
            ratio(
                metrics.qwen38_accepted_drafts,
                metrics.qwen38_speculative_cycles
            ),
            1.5
        );
    }

    #[test]
    fn dflash_metrics_report_interval_and_cumulative_acceptance() {
        let started = Instant::now();
        let mut metrics = SessionMetrics::new(started, 1_000);
        assert!(
            metrics
                .record_dflash(
                    started,
                    MuseGlimmerDFlashStats {
                        cycles: 1,
                        drafted_tokens: 15,
                        accepted_drafts: 3,
                        emitted_tokens: 4,
                        cycle_duration: Duration::from_millis(30),
                        target_position: 1_004,
                        dflash_position: 1_004,
                    },
                )
                .is_none()
        );
        let snapshot = metrics
            .record_dflash(
                started + SESSION_METRICS_INTERVAL,
                MuseGlimmerDFlashStats {
                    cycles: 4,
                    drafted_tokens: 60,
                    accepted_drafts: 15,
                    emitted_tokens: 19,
                    cycle_duration: Duration::from_millis(120),
                    target_position: 1_019,
                    dflash_position: 1_019,
                },
            )
            .expect("ten-second DFlash report interval elapsed");

        assert_eq!(snapshot.interval.cycles, 4);
        assert_eq!(snapshot.interval.drafted_tokens, 60);
        assert_eq!(snapshot.interval.accepted_drafts, 15);
        assert_eq!(snapshot.interval.emitted_tokens, 19);
        assert_eq!(snapshot.interval.cycle_duration, Duration::from_millis(120));
        assert_eq!(snapshot.cumulative.target_position, 1_019);
        assert_eq!(snapshot.cumulative.dflash_position, 1_019);
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
