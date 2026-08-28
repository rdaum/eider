use eider_cuda::{Error, Result};
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use infer::nemotron3::{
    Nemotron3Bf16Storage, Nemotron3Fp8Storage, Nemotron3Model, Nemotron3StorageConfig,
};
use infer::runtime::generation::{GenerationConfig, Nemotron3GenerationSession};
use std::env;
use std::io::Write;
use std::path::PathBuf;
use tracing::info;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nemotron3-generate".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!(
                "{program} <model-dir> [prompt] [max-new-tokens] [bf16|fp8|nvfp4] [fp8|nvfp4]"
            ),
        })?;
    let prompt = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "Hello!".to_string());
    let max_new_tokens = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| Error::Format {
            label: "max-new-tokens",
            detail: error.to_string(),
        })?
        .unwrap_or(64);
    let storage = Nemotron3StorageConfig {
        bf16: parse_bf16_storage(args.next())?,
        fp8: parse_fp8_storage(args.next())?,
        ..Nemotron3StorageConfig::default()
    };

    let mut generation = GenerationConfig::from_model_dir(&model_dir)?;
    generation.max_new_tokens = max_new_tokens;
    let template = CheckpointChatTemplate::from_model_dir(&model_dir)?;
    let rendered = template.render_and_tokenize(
        &[ChatMessage::user(prompt)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    let model = Nemotron3Model::load_with_storage(&model_dir, storage)?;
    let mut session = Nemotron3GenerationSession::new(
        &model,
        template.tokenizer(),
        &rendered.token_ids,
        generation,
    )?;
    while let Some(token) = session.next_token()? {
        print!("{}", token.text);
        std::io::stdout().flush().ok();
    }
    info!(
        generated_tokens = session.generated_token_count(),
        finish_reason = ?session.finish_reason(),
        "generation complete"
    );
    Ok(())
}

fn parse_bf16_storage(value: Option<std::ffi::OsString>) -> Result<Nemotron3Bf16Storage> {
    match value
        .as_deref()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("bf16")
    {
        "bf16" => Ok(Nemotron3Bf16Storage::Bf16),
        "fp8" => Ok(Nemotron3Bf16Storage::Fp8),
        "nvfp4" => Ok(Nemotron3Bf16Storage::Nvfp4),
        value => Err(Error::Format {
            label: "BF16 storage",
            detail: format!("expected bf16, fp8, or nvfp4, got {value:?}"),
        }),
    }
}

fn parse_fp8_storage(value: Option<std::ffi::OsString>) -> Result<Nemotron3Fp8Storage> {
    match value
        .as_deref()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("fp8")
    {
        "fp8" => Ok(Nemotron3Fp8Storage::Fp8),
        "nvfp4" => Ok(Nemotron3Fp8Storage::Nvfp4),
        value => Err(Error::Format {
            label: "FP8 storage",
            detail: format!("expected fp8 or nvfp4, got {value:?}"),
        }),
    }
}
