//! Compares production Flash Next prefill with serial GDN and QSA references.

use eider_cuda::{Error, Result};
use eider_inference::qwen38_flash_next::{Qwen38FlashNextModel, probe_prefill_against_reference};
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen38-flash-next-prefill-probe".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: usage(&program),
        })?;
    let prompt_file = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: usage(&program),
        })?;
    let chunk_tokens = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| Error::Format {
            label: "chunk-tokens",
            detail: error.to_string(),
        })?
        .unwrap_or(512);
    let artifact_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or(default_artifact_dir()?);
    let prompt = std::fs::read_to_string(&prompt_file).map_err(|error| Error::Format {
        label: "Qwen3.8 Flash Next prefill probe prompt",
        detail: format!("{}: {error}", prompt_file.display()),
    })?;
    let template = CheckpointChatTemplate::from_model_dir(&model_dir)?;
    let rendered = template.render_and_tokenize(
        &[ChatMessage::user(prompt)],
        &[],
        ChatTemplateOptions::default(),
    )?;

    let load_started = Instant::now();
    let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir)?;
    eprintln!(
        "model loaded in {:.2}s",
        load_started.elapsed().as_secs_f64()
    );
    let report = probe_prefill_against_reference(&mut model, &rendered.token_ids, chunk_tokens)?;
    println!(
        "prompt_tokens={} reference_token={} reference_seconds={:.3}",
        report.prompt_tokens,
        report.reference_frontier.id,
        report.reference_duration.as_secs_f64(),
    );
    println!(
        "production_token={} token_match={} production_seconds={:.3} max_abs={:.9} cosine={:.12} relative_rmse={:.12}",
        report.production_frontier.id,
        report.frontier_matches(),
        report.production_duration.as_secs_f64(),
        report.production_difference.maximum_absolute_error,
        report.production_difference.cosine_similarity,
        report.production_difference.relative_rmse,
    );
    if !report.frontier_matches() {
        return Err(Error::Format {
            label: "Qwen3.8 Flash Next prefill probe",
            detail: format!(
                "production token {} differs from reference token {}",
                report.production_frontier.id, report.reference_frontier.id
            ),
        });
    }
    Ok(())
}

fn usage(program: &str) -> String {
    format!("{program} <model-dir> <prompt-file> [chunk-tokens] [artifact-dir]")
}

fn default_artifact_dir() -> Result<PathBuf> {
    let root = if let Some(path) = env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(path)
    } else {
        let home = env::var_os("HOME").ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next artifact directory",
            detail: "HOME and XDG_CACHE_HOME are unset".to_string(),
        })?;
        PathBuf::from(home).join(".cache")
    };
    Ok(root.join("eider/qwen38-flash-next-native"))
}
