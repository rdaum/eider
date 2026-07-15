use clap::Parser;
use eider_api::{ApiConfig, InferenceActor, InferenceActorConfig, serve};
use infer::runtime::scheduler::Qwen36SchedulerConfig;
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::info;

#[derive(Debug, Parser)]
#[command(about = "Serve Eider through the OpenAI Responses API")]
struct Args {
    /// Qwen3.6 checkpoint directory.
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

    /// Environment variable containing an optional server bearer token.
    #[arg(long, default_value = "EIDER_API_KEY")]
    api_key_env: String,
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
    actor_config.scheduler = Qwen36SchedulerConfig {
        decode_capacity: args.decode_capacity,
        prefill_sequence_capacity: args.prefill_sequence_capacity,
        prefill_token_capacity: args.prefill_token_capacity,
        max_active_sequences: args.max_active_sequences,
        max_context_tokens: args.max_context_tokens,
    };
    let actor = InferenceActor::spawn(actor_config)
        .map_err(|error| format!("failed to initialise inference: {}", error.message))?;
    let config = ApiConfig {
        listen: args.listen,
        model: args.served_model_name,
        bearer_token: std::env::var(&args.api_key_env).ok(),
        context_window: args.max_context_tokens,
    };
    info!(model = %config.model, listen = %config.listen, "serving model");
    serve(actor, config).await?;
    Ok(())
}
