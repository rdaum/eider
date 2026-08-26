use infer::nvfp4::{Error, Result};
use infer::qwen38_flash_next::Qwen38FlashNextModel;
use infer::runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use infer::runtime::qwen38_flash_next_sequence::{
    Qwen38FlashNextSequence, new_qwen38_flash_next_sequence_cache,
};
use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen38-flash-next-generate".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir> [prompt] [max-new-tokens] [artifact-dir]"),
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
        .unwrap_or(16);
    let artifact_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or(default_artifact_dir()?);

    let template = CheckpointChatTemplate::from_model_dir(&model_dir)?;
    let rendered = template.render_and_tokenize(
        &[ChatMessage::user(prompt)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    if rendered.token_ids.is_empty() {
        return Err(Error::Format {
            label: "Qwen3.8 Flash Next prompt",
            detail: "chat template produced no tokens".to_string(),
        });
    }

    let load_started = Instant::now();
    let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir)?;
    eprintln!(
        "model loaded in {:.2}s",
        load_started.elapsed().as_secs_f64()
    );
    let capacity = rendered
        .token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next generation capacity",
            expected: "prompt + output without overflow".to_string(),
            actual: format!(
                "prompt={} output={max_new_tokens}",
                rendered.token_ids.len()
            ),
        })?
        .min(model.config().max_position_embeddings);
    let mut cache = new_qwen38_flash_next_sequence_cache(&model, 1, capacity)?;
    let mut sequence = Qwen38FlashNextSequence::admit(&model, &mut cache, capacity)?;

    let prefill_started = Instant::now();
    let mut next = None;
    for &token in &rendered.token_ids {
        next = Some(sequence.decode_token(&mut model, &mut cache, token)?);
    }
    eprintln!(
        "prefilled {} tokens in {:.2}s",
        rendered.token_ids.len(),
        prefill_started.elapsed().as_secs_f64()
    );

    let generation_started = Instant::now();
    for generated in 0..max_new_tokens {
        let token = next.expect("non-empty prompt produces a next token");
        if token.id == model.config().eos_token_id {
            break;
        }
        let text = template
            .tokenizer()
            .decode(&[token.id], false)
            .map_err(|error| Error::Format {
                label: "Qwen3.8 Flash Next token decode",
                detail: error.to_string(),
            })?;
        print!("{text}");
        std::io::stdout().flush().ok();
        if generated + 1 < max_new_tokens {
            next = Some(sequence.decode_token(&mut model, &mut cache, token.id)?);
        }
    }
    sequence.finish(&mut cache)?;
    println!();
    eprintln!(
        "generated up to {max_new_tokens} tokens in {:.2}s",
        generation_started.elapsed().as_secs_f64()
    );
    Ok(())
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
