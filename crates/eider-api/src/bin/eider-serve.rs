use clap::{Parser, ValueEnum};
use eider_api::deployment::{ArtifactKind, resolve_catalogue_model, resolve_local_model};
use eider_api::metrics::{TokenRateSampler, metrics as server_metrics};
use eider_api::{ApiConfig, InferenceActor, InferenceActorConfig, serve_with_shutdown};
use fast_telemetry_export::dogstatsd::DogStatsDConfig;
use infer::metrics::metrics as infer_metrics;
use infer::nemotron3::{
    Nemotron3Bf16Storage, Nemotron3Fp8Storage, Nemotron3KvCacheStorage, Nemotron3StorageConfig,
};
use infer::qwen3::qwen36::{Qwen36Bf16Storage, Qwen36Bf16StorageConfig, Qwen36Fp8AttentionStorage};
use infer::runtime::scheduler::SchedulerConfig;
use infer::step37::{Step37Bf16Storage, Step37Bf16StorageConfig};
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::info;

const TOKEN_RATE_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const DEFAULT_LOG_FILTER: &str = "info,hf_xet=warn,xet_client=warn,xet_data=warn";

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum QwenBf16StorageArg {
    Bf16,
    Fp8,
    #[default]
    Nvfp4,
}

impl From<QwenBf16StorageArg> for Qwen36Bf16Storage {
    fn from(value: QwenBf16StorageArg) -> Self {
        match value {
            QwenBf16StorageArg::Bf16 => Self::Bf16,
            QwenBf16StorageArg::Fp8 => Self::Fp8,
            QwenBf16StorageArg::Nvfp4 => Self::Nvfp4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum QwenFp8AttentionStorageArg {
    #[default]
    Fp8,
    Nvfp4,
}

impl From<QwenFp8AttentionStorageArg> for Qwen36Fp8AttentionStorage {
    fn from(value: QwenFp8AttentionStorageArg) -> Self {
        match value {
            QwenFp8AttentionStorageArg::Fp8 => Self::Fp8,
            QwenFp8AttentionStorageArg::Nvfp4 => Self::Nvfp4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum StepBf16StorageArg {
    Bf16,
    #[default]
    Nvfp4,
}

impl From<StepBf16StorageArg> for Step37Bf16Storage {
    fn from(value: StepBf16StorageArg) -> Self {
        match value {
            StepBf16StorageArg::Bf16 => Self::Bf16,
            StepBf16StorageArg::Nvfp4 => Self::Nvfp4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum NemotronBf16StorageArg {
    Bf16,
    Fp8,
    #[default]
    Nvfp4,
}

impl From<NemotronBf16StorageArg> for Nemotron3Bf16Storage {
    fn from(value: NemotronBf16StorageArg) -> Self {
        match value {
            NemotronBf16StorageArg::Bf16 => Self::Bf16,
            NemotronBf16StorageArg::Fp8 => Self::Fp8,
            NemotronBf16StorageArg::Nvfp4 => Self::Nvfp4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum NemotronFp8StorageArg {
    Fp8,
    #[default]
    Nvfp4,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum NemotronKvCacheStorageArg {
    #[default]
    F32,
    Nvfp4,
}

impl From<NemotronKvCacheStorageArg> for Nemotron3KvCacheStorage {
    fn from(value: NemotronKvCacheStorageArg) -> Self {
        match value {
            NemotronKvCacheStorageArg::F32 => Self::F32,
            NemotronKvCacheStorageArg::Nvfp4 => Self::Nvfp4,
        }
    }
}

impl From<NemotronFp8StorageArg> for Nemotron3Fp8Storage {
    fn from(value: NemotronFp8StorageArg) -> Self {
        match value {
            NemotronFp8StorageArg::Fp8 => Self::Fp8,
            NemotronFp8StorageArg::Nvfp4 => Self::Nvfp4,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Serve Eider through the OpenAI Responses API")]
struct Args {
    /// Stable ID of a model in Eider's built-in catalogue.
    model: Option<String>,

    /// Explicit local checkpoint directory for development or conversion work.
    #[arg(long, conflicts_with = "model")]
    model_dir: Option<PathBuf>,

    /// Explicit derived-artifact directory for a local development checkpoint.
    #[arg(long, requires = "model_dir")]
    artifact_dir: Option<PathBuf>,

    /// Prohibit network access while resolving a catalogue model.
    #[arg(long)]
    offline: bool,

    /// Address exposed by the HTTP server.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    /// Model name accepted in Responses requests.
    #[arg(long)]
    served_model_name: Option<String>,

    /// Maximum simultaneous decode rows.
    #[arg(long, default_value_t = 8)]
    decode_capacity: usize,

    /// Maximum simultaneous prefill rows.
    #[arg(long, default_value_t = 8)]
    prefill_sequence_capacity: usize,

    /// Maximum total prompt tokens in one prefill iteration.
    #[arg(long)]
    prefill_token_capacity: Option<usize>,

    /// Maximum requests retaining device sequence state.
    #[arg(long, default_value_t = 8)]
    max_active_sequences: usize,

    /// Maximum prompt plus generated tokens per request.
    #[arg(long)]
    max_context_tokens: Option<usize>,

    /// Device-memory budget in GiB for prompt-prefix checkpoints; zero disables it.
    #[arg(long, default_value_t = 2)]
    prefix_cache_gib: usize,

    /// Runtime storage for BF16 Qwen attention projections.
    #[arg(long, value_enum, default_value_t = QwenBf16StorageArg::Nvfp4)]
    qwen_bf16_attention: QwenBf16StorageArg,

    /// Runtime storage for the BF16 Qwen LM head.
    #[arg(long, value_enum, default_value_t = QwenBf16StorageArg::Nvfp4)]
    qwen_bf16_lm_head: QwenBf16StorageArg,

    /// Runtime storage for native FP8 Qwen attention projections.
    #[arg(long, value_enum, default_value_t = QwenFp8AttentionStorageArg::Fp8)]
    qwen_fp8_attention: QwenFp8AttentionStorageArg,

    /// Resident expert slots per routed Step layer.
    #[arg(long)]
    step_expert_capacity: Option<usize>,

    /// Resident exact-NVFP4 expert slots per routed DeepSeek V4 layer.
    #[arg(long, default_value_t = 8)]
    deepseek_expert_capacity: usize,

    /// Runtime storage for BF16 Step attention projections.
    #[arg(long, value_enum, default_value_t = StepBf16StorageArg::Nvfp4)]
    step_bf16_attention: StepBf16StorageArg,

    /// Runtime storage for the BF16 Step dense MLPs.
    #[arg(long, value_enum, default_value_t = StepBf16StorageArg::Nvfp4)]
    step_bf16_dense_mlp: StepBf16StorageArg,

    /// Runtime storage for BF16 Step shared experts.
    #[arg(long, value_enum, default_value_t = StepBf16StorageArg::Nvfp4)]
    step_bf16_shared_expert: StepBf16StorageArg,

    /// Runtime storage for the BF16 Step LM head.
    #[arg(long, value_enum, default_value_t = StepBf16StorageArg::Nvfp4)]
    step_bf16_lm_head: StepBf16StorageArg,

    /// Runtime storage for BF16 Nemotron dense linears.
    #[arg(long, value_enum, default_value_t = NemotronBf16StorageArg::Nvfp4)]
    nemotron_bf16_storage: NemotronBf16StorageArg,

    /// Runtime storage for native FP8 Nemotron dense linears.
    #[arg(long, value_enum, default_value_t = NemotronFp8StorageArg::Nvfp4)]
    nemotron_fp8_storage: NemotronFp8StorageArg,

    /// Runtime storage for Nemotron attention key/value cache pages.
    #[arg(long, value_enum, default_value_t = NemotronKvCacheStorageArg::F32)]
    nemotron_kv_cache: NemotronKvCacheStorageArg,

    /// Environment variable containing an optional server bearer token.
    #[arg(long, default_value = "EIDER_API_KEY")]
    api_key_env: String,

    /// DogStatsD endpoint (host:port) for push-based metrics export.
    ///
    /// When set, metrics are exported via DogStatsD over UDP at the given
    /// interval. Prometheus remains available at /metrics regardless.
    #[arg(long, value_name = "ADDR")]
    dogstatsd_endpoint: Option<String>,

    /// DogStatsD export interval in seconds.
    #[arg(long, default_value_t = 1, value_name = "SECONDS")]
    dogstatsd_interval_secs: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER)),
        )
        .try_init();
    let startup_complete = Arc::new(AtomicBool::new(false));
    let shutdown = CancellationToken::new();
    let signal = install_shutdown_signal()?;
    let signal_task = tokio::spawn({
        let startup_complete = Arc::clone(&startup_complete);
        let shutdown = shutdown.clone();
        async move {
            let signal = signal.await;
            if !startup_complete.load(Ordering::Acquire) {
                info!(
                    signal = signal.name(),
                    "shutdown signal received during startup; exiting"
                );
                std::process::exit(signal.exit_code());
            }
            info!(signal = signal.name(), "shutdown signal received");
            shutdown.cancel();
        }
    });
    info!("Eider");
    let args = Args::parse();
    let mut resolved = match (args.model.as_deref(), args.model_dir) {
        (Some(id), None) => resolve_catalogue_model(id, args.offline).await?,
        (None, Some(path)) => resolve_local_model(path)?,
        (Some(_), Some(_)) => unreachable!("clap rejects model plus --model-dir"),
        (None, None) => return Err("provide a supported model ID or --model-dir PATH".into()),
    };
    if let Some(artifact_dir) = args.artifact_dir {
        resolved.artifact_dir = artifact_dir;
    }
    if resolved.preparation == ArtifactKind::Step37Experts {
        server_metrics().model_preparations.inc();
        info!(artifact_dir = %resolved.artifact_dir.display(), "preparing Step-3.7 expert artifacts");
        infer::step37::prepare_all_at(&resolved.checkpoint_dir, &resolved.artifact_dir)?;
    }
    let defaults = resolved.defaults;
    let served_model_name = args
        .served_model_name
        .unwrap_or_else(|| defaults.served_model_name.to_string());
    let max_context_tokens = args
        .max_context_tokens
        .unwrap_or(defaults.max_context_tokens);
    let prefill_token_capacity = args
        .prefill_token_capacity
        .unwrap_or(defaults.prefill_token_capacity);
    let step_expert_capacity = args
        .step_expert_capacity
        .unwrap_or(defaults.step_expert_capacity);
    let mut actor_config = InferenceActorConfig::new(&resolved.checkpoint_dir);
    actor_config.artifact_dir = resolved.artifact_dir.clone();
    actor_config.scheduler = SchedulerConfig {
        decode_capacity: args.decode_capacity,
        prefill_sequence_capacity: args.prefill_sequence_capacity,
        prefill_token_capacity,
        max_active_sequences: args.max_active_sequences,
        max_context_tokens,
    };
    actor_config.prefix_cache.max_device_bytes = args
        .prefix_cache_gib
        .checked_mul(1024 * 1024 * 1024)
        .ok_or("Qwen prefix-cache size exceeds usize")?;
    actor_config.qwen_bf16_storage = Qwen36Bf16StorageConfig::new(
        args.qwen_bf16_attention.into(),
        args.qwen_bf16_lm_head.into(),
    );
    actor_config.qwen_fp8_attention_storage = args.qwen_fp8_attention.into();
    actor_config.step_expert_capacity = step_expert_capacity;
    actor_config.deepseek_expert_capacity = args.deepseek_expert_capacity;
    actor_config.step_bf16_storage = Step37Bf16StorageConfig {
        attention: args.step_bf16_attention.into(),
        dense_mlp: args.step_bf16_dense_mlp.into(),
        shared_expert: args.step_bf16_shared_expert.into(),
        lm_head: args.step_bf16_lm_head.into(),
    };
    actor_config.nemotron_storage = Nemotron3StorageConfig {
        bf16: args.nemotron_bf16_storage.into(),
        fp8: args.nemotron_fp8_storage.into(),
        kv_cache: args.nemotron_kv_cache.into(),
    };
    let actor = InferenceActor::spawn(actor_config)
        .map_err(|error| format!("failed to initialise inference: {}", error.message))?;
    startup_complete.store(true, Ordering::Release);
    let config = ApiConfig {
        listen: args.listen,
        model: served_model_name,
        bearer_token: std::env::var(&args.api_key_env).ok(),
        context_window: max_context_tokens,
    };
    info!(
        model = %config.model,
        identity = %resolved.identity,
        checkpoint_dir = %resolved.checkpoint_dir.display(),
        artifact_dir = %resolved.artifact_dir.display(),
        listen = %config.listen,
        "serving model"
    );

    let metrics_task = match args.dogstatsd_endpoint {
        Some(endpoint) => start_dogstatsd_export(endpoint, args.dogstatsd_interval_secs),
        None => start_token_rate_sampler(),
    };

    let shutdown_actor = actor.clone();
    let shutdown_server = shutdown.clone();
    let serve_result = serve_with_shutdown(actor, config, async move {
        shutdown_server.cancelled().await;
        info!("stopping inference");
        shutdown_actor.shutdown();
    })
    .await;
    signal_task.abort();
    info!("stopping metrics sampler");
    metrics_task.cancel();
    serve_result?;
    info!("server stopped");
    Ok(())
}

#[derive(Clone, Copy)]
enum ShutdownSignal {
    Interrupt,
    Terminate,
}

impl ShutdownSignal {
    fn name(self) -> &'static str {
        match self {
            Self::Interrupt => "SIGINT",
            Self::Terminate => "SIGTERM",
        }
    }

    fn exit_code(self) -> i32 {
        match self {
            Self::Interrupt => 128 + 2,
            Self::Terminate => 128 + 15,
        }
    }
}

#[cfg(unix)]
fn install_shutdown_signal()
-> Result<impl Future<Output = ShutdownSignal> + Send + 'static, std::io::Error> {
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    Ok(async move {
        tokio::select! {
            _ = interrupt.recv() => ShutdownSignal::Interrupt,
            _ = terminate.recv() => ShutdownSignal::Terminate,
        }
    })
}

#[cfg(not(unix))]
fn install_shutdown_signal()
-> Result<impl Future<Output = ShutdownSignal> + Send + 'static, std::io::Error> {
    Ok(async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "failed to wait for interrupt signal");
            std::future::pending::<()>().await;
        }
        ShutdownSignal::Interrupt
    })
}

fn start_dogstatsd_export(endpoint: String, interval_secs: u64) -> CancellationToken {
    server_metrics().dogstatsd_configured.set(1);
    server_metrics().dogstatsd_exporters_started.inc();
    let cancel = CancellationToken::new();
    let config =
        DogStatsDConfig::new(endpoint).with_interval(Duration::from_secs(interval_secs.max(1)));
    let cancel_clone = cancel.clone();
    let mut token_rate_sampler = new_token_rate_sampler();
    tokio::spawn(async move {
        let mut server_state = eider_api::metrics::ServerMetricsDogStatsDState::new();
        let mut infer_state = infer::metrics::InferMetricsDogStatsDState::new();
        fast_telemetry_export::dogstatsd::run(config, cancel_clone, move |output| {
            sample_token_rates(&mut token_rate_sampler);
            server_metrics().export_dogstatsd_delta(output, &[], &mut server_state);
            infer_metrics().export_dogstatsd_delta(output, &[], &mut infer_state);
            server_metrics().dogstatsd_export_ticks.inc();
        })
        .await;
    });
    cancel
}

fn start_token_rate_sampler() -> CancellationToken {
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let mut sampler = new_token_rate_sampler();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(TOKEN_RATE_SAMPLE_INTERVAL);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            tokio::select! {
                _ = interval.tick() => sample_token_rates(&mut sampler),
                _ = cancel_clone.cancelled() => return,
            }
        }
    });
    cancel
}

fn new_token_rate_sampler() -> TokenRateSampler {
    let metrics = infer_metrics();
    TokenRateSampler::new(
        Instant::now(),
        metrics.prefill_tokens.sum(),
        metrics.generated_tokens.sum(),
    )
}

fn sample_token_rates(sampler: &mut TokenRateSampler) {
    let infer = infer_metrics();
    let rates = sampler.sample(
        Instant::now(),
        infer.prefill_tokens.sum(),
        infer.generated_tokens.sum(),
    );
    let server = server_metrics();
    server
        .current_prefill_tokens_per_second
        .set(rates.prefill_tokens_per_second);
    server
        .current_decode_tokens_per_second
        .set(rates.decode_tokens_per_second);
}
