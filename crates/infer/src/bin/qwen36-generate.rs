use infer::nvfp4::{Error, Result};
use infer::qwen3::qwen36::Qwen36TextModel;
use infer::runtime::generation::{GenerationConfig, Qwen36GenerationSession};
use std::env;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct GenerateArgs {
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut generation = GenerationConfig::from_model_dir(&args.model_dir)?;
    generation.max_new_tokens = args.max_new_tokens;
    generation.sampling.temperature = args.temperature.unwrap_or(generation.sampling.temperature);
    generation.sampling.top_k = args.top_k.unwrap_or(generation.sampling.top_k);
    generation.sampling.top_p = args.top_p.unwrap_or(generation.sampling.top_p);
    generation.sampling.seed = args.seed.or(generation.sampling.seed);
    generation.sampling.presence_penalty = args
        .presence_penalty
        .unwrap_or(generation.sampling.presence_penalty);
    generation.sampling.frequency_penalty = args
        .frequency_penalty
        .unwrap_or(generation.sampling.frequency_penalty);

    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json")).map_err(|err| {
        Error::Format {
            label: "tokenizer",
            detail: err.to_string(),
        }
    })?;

    let model = Qwen36TextModel::open(&args.model_dir)?;
    let prompt = render_prompt(&args.prompt);
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|err| Error::Format {
            label: "tokenizer encode",
            detail: err.to_string(),
        })?;
    let prompt_ids = encoding.get_ids();
    let (layers, hidden, vocab) = {
        let manifest = model.manifest();
        (manifest.layers, manifest.hidden, manifest.vocab)
    };
    let mut session = Qwen36GenerationSession::new(&model, &tokenizer, prompt_ids, generation)?;
    while let Some(token) = session.next_token()? {
        print!("{}", token.text);
        std::io::stdout().flush().ok();
    }

    eprintln!(
        "\n[generated {} tokens, {} layers, hidden={}, vocab={}, finish={:?}]",
        session.generated_token_count(),
        layers,
        hidden,
        vocab,
        session.finish_reason(),
    );

    Ok(())
}

fn render_prompt(prompt: &str) -> String {
    if prompt.contains("<|im_start|>") {
        return prompt.to_string();
    }
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

fn parse_args() -> Result<GenerateArgs> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-generate".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!(
                "{program} <model-dir> [prompt] [max-new-tokens] [temperature] [top-k] [top-p] [seed] [presence-penalty] [frequency-penalty]"
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
        .map_err(|err| Error::Format {
            label: "max-new-tokens",
            detail: err.to_string(),
        })?
        .unwrap_or(64);
    let temperature = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "temperature",
            detail: err.to_string(),
        })?;
    let top_k = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "top-k",
            detail: err.to_string(),
        })?;
    let top_p = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "top-p",
            detail: err.to_string(),
        })?;
    let seed = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<u64>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "seed",
            detail: err.to_string(),
        })?;
    let presence_penalty = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "presence-penalty",
            detail: err.to_string(),
        })?;
    let frequency_penalty = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "frequency-penalty",
            detail: err.to_string(),
        })?;
    Ok(GenerateArgs {
        model_dir,
        prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
        presence_penalty,
        frequency_penalty,
    })
}

#[cfg(test)]
mod tests {
    use super::render_prompt;

    #[test]
    fn plain_prompt_gets_qwen_chat_prefix() {
        assert_eq!(
            render_prompt("Hello"),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn templated_prompt_is_not_wrapped_again() {
        let prompt = "<|im_start|>user\nHello<|im_end|>\n";
        assert_eq!(render_prompt(prompt), prompt);
    }
}
