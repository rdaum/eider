use clap::{Parser, Subcommand};
use eider_api::deployment::{ArtifactKind, catalogue_models, resolve_catalogue_model};
use tracing::info;

const DEFAULT_LOG_FILTER: &str = "info,hf_xet=warn,xet_client=warn,xet_data=warn";

#[derive(Debug, Parser)]
#[command(about = "Manage Eider model deployments")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// List stable IDs in Eider's built-in catalogue.
    List,

    /// Populate the Hugging Face snapshot cache for a supported model.
    Fetch {
        /// Stable ID of a model in Eider's built-in catalogue.
        model: String,

        /// Prohibit network access and require a complete cached snapshot.
        #[arg(long)]
        offline: bool,

        /// Also construct Step-3.7's disk-backed expert records.
        #[arg(long)]
        prepare: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER)),
        )
        .try_init();
    match Args::parse().command {
        Command::Model {
            command: ModelCommand::List,
        } => {
            for model in catalogue_models() {
                println!("{}\t{}", model.id, model.repository);
            }
        }
        Command::Model {
            command:
                ModelCommand::Fetch {
                    model,
                    offline,
                    prepare,
                },
        } => {
            let resolved = resolve_catalogue_model(&model, offline).await?;
            if prepare {
                if resolved.preparation != ArtifactKind::Step37Experts {
                    return Err(format!(
                        "--prepare is currently required only for Step-3.7; {model} prepares at server startup when needed"
                    )
                    .into());
                }
                eider_api::metrics::metrics().model_preparations.inc();
                infer::step37::prepare_all_at(&resolved.checkpoint_dir, &resolved.artifact_dir)?;
            }
            info!(
                identity = %resolved.identity,
                checkpoint_dir = %resolved.checkpoint_dir.display(),
                artifact_dir = %resolved.artifact_dir.display(),
                "model fetch complete"
            );
        }
    }
    Ok(())
}
