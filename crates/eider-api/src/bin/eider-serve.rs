use clap::Parser;
use eider_api::metrics::{TokenRateSampler, metrics as server_metrics};
use eider_api::{ApiConfig, InferenceActor, InferenceActorConfig, serve};
use fast_telemetry_export::dogstatsd::DogStatsDConfig;
use infer::metrics::metrics as infer_metrics;
use infer::runtime::scheduler::SchedulerConfig;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::info;

const TOKEN_RATE_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Parser)]
#[command(about = "Serve Eider through the OpenAI Responses API")]
struct Args {
    /// Supported checkpoint directory.
    model_dir: PathBuf,

    /// Address exposed by the HTTP server.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    /// Model name accepted in Responses requests.
    #[arg(long, default_value = "eider-qwen3.6")]
    served_model_name: String,

    /// Maximum simultaneous decode rows.
    #[arg(long, default_value_t = 8)]
    decode_capacity: usize,

    /// Maximum simultaneous prefill rows.
    #[arg(long, default_value_t = 8)]
    prefill_sequence_capacity: usize,

    /// Maximum total prompt tokens in one prefill iteration.
    #[arg(long, default_value_t = 128)]
    prefill_token_capacity: usize,

    /// Maximum requests retaining device sequence state.
    #[arg(long, default_value_t = 8)]
    max_active_sequences: usize,

    /// Maximum prompt plus generated tokens per request.
    #[arg(long, default_value_t = 32_768)]
    max_context_tokens: usize,

    /// Resident expert slots per routed Step layer.
    #[arg(long, default_value_t = 240)]
    step_expert_capacity: usize,

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
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    info!("Eider");
    let args = Args::parse();
    let mut actor_config = InferenceActorConfig::new(&args.model_dir);
    actor_config.scheduler = SchedulerConfig {
        decode_capacity: args.decode_capacity,
        prefill_sequence_capacity: args.prefill_sequence_capacity,
        prefill_token_capacity: args.prefill_token_capacity,
        max_active_sequences: args.max_active_sequences,
        max_context_tokens: args.max_context_tokens,
    };
    actor_config.step_expert_capacity = args.step_expert_capacity;
    let actor = InferenceActor::spawn(actor_config)
        .map_err(|error| format!("failed to initialise inference: {}", error.message))?;
    let config = ApiConfig {
        listen: args.listen,
        model: args.served_model_name,
        bearer_token: std::env::var(&args.api_key_env).ok(),
        context_window: args.max_context_tokens,
    };
    info!(model = %config.model, listen = %config.listen, "serving model");

    let metrics_task = match args.dogstatsd_endpoint {
        Some(endpoint) => start_dogstatsd_export(endpoint, args.dogstatsd_interval_secs),
        None => start_token_rate_sampler(),
    };

    let serve_result = serve(actor, config).await;
    info!("stopping metrics sampler");
    metrics_task.cancel();
    serve_result?;
    Ok(())
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
