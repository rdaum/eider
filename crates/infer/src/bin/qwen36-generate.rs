use infer::nvfp4::{Error, Result};
use infer::qwen3::qwen36::Qwen36TextModel;
use rand::Rng;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Clone, Copy)]
struct SamplingConfig {
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

struct GenerateArgs {
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut sampling = load_generation_config(&args.model_dir)?;
    sampling.temperature = args.temperature.unwrap_or(sampling.temperature);
    sampling.top_k = args.top_k.unwrap_or(sampling.top_k);
    sampling.top_p = args.top_p.unwrap_or(sampling.top_p);

    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json")).map_err(|err| {
        Error::Format {
            label: "tokenizer",
            detail: err.to_string(),
        }
    })?;

    let model = Qwen36TextModel::open(&args.model_dir)?;
    let manifest = model.manifest();

    let prompt = render_prompt(&args.prompt);
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|err| Error::Format {
            label: "tokenizer encode",
            detail: err.to_string(),
        })?;
    let prompt_ids = encoding.get_ids();
    if prompt_ids.is_empty() {
        return Err(Error::Format {
            label: "prompt",
            detail: "prompt tokenized to zero tokens".to_string(),
        });
    }

    let total_tokens = prompt_ids.len() + args.max_new_tokens;
    let mut state = model.new_decode_state(total_tokens)?;

    let eos_ids = load_eos_token_ids(&args.model_dir)?;
    let mut rng = rand::rng();
    let mut decode_stream = tokenizer.decode_stream(true);

    let mut last_id: u32 = *prompt_ids.last().expect("non-empty prompt");
    let mut generated: Vec<u32> = Vec::with_capacity(args.max_new_tokens);

    for (i, &token_id) in prompt_ids.iter().enumerate() {
        let is_last_prompt_token = i + 1 == prompt_ids.len();
        let (next_id, next_value) = if is_last_prompt_token && sampling.temperature > 0.0 {
            let logits = model.decode_one_token_logits(&mut state, token_id)?.logits;
            let id = sample_next_token(&logits, sampling, &mut rng)?;
            (id, logits[id as usize])
        } else {
            let next = model.decode_one_token(&mut state, token_id)?;
            (next.id, next.value)
        };
        last_id = next_id;
        eprint!(
            "[pos {i} tok {} -> {} v={:.2}] ",
            token_id, next_id, next_value
        );
        if is_last_prompt_token {
            generated.push(last_id);
            print_stream_token(&mut decode_stream, last_id)?;
        }
    }
    eprintln!();

    for _ in 0..args.max_new_tokens.saturating_sub(1) {
        if eos_ids.contains(&last_id) {
            break;
        }
        last_id = if sampling.temperature > 0.0 {
            let logits = model.decode_one_token_logits(&mut state, last_id)?.logits;
            sample_next_token(&logits, sampling, &mut rng)?
        } else {
            model.decode_one_token(&mut state, last_id)?.id
        };
        generated.push(last_id);
        print_stream_token(&mut decode_stream, last_id)?;
    }

    eprintln!(
        "\n[generated {} tokens, {} layers, hidden={}, vocab={}]",
        generated.len(),
        manifest.layers,
        manifest.hidden,
        manifest.vocab,
    );

    Ok(())
}

fn print_stream_token<
    M: tokenizers::Model,
    N: tokenizers::Normalizer,
    PT: tokenizers::PreTokenizer,
    PP: tokenizers::PostProcessor,
    D: tokenizers::Decoder,
>(
    stream: &mut tokenizers::DecodeStream<'_, M, N, PT, PP, D>,
    id: u32,
) -> Result<()> {
    if let Some(decoded) = stream.step(id).map_err(|err| Error::Format {
        label: "tokenizer stream decode",
        detail: err.to_string(),
    })? {
        print!("{decoded}");
    } else {
        return Ok(());
    }
    std::io::stdout().flush().ok();
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
                "{program} <model-dir> [prompt] [max-new-tokens] [temperature] [top-k] [top-p]"
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
    Ok(GenerateArgs {
        model_dir,
        prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
    })
}

fn load_generation_config(model_dir: &std::path::Path) -> Result<SamplingConfig> {
    let path = model_dir.join("generation_config.json");
    let value: Value =
        serde_json::from_str(&fs::read_to_string(&path).map_err(|err| Error::Format {
            label: "generation_config.json",
            detail: format!("{}: {err}", path.display()),
        })?)
        .map_err(|err| Error::Format {
            label: "generation_config.json",
            detail: err.to_string(),
        })?;
    Ok(SamplingConfig {
        temperature: value["temperature"].as_f64().unwrap_or(1.0) as f32,
        top_k: value["top_k"].as_u64().unwrap_or(20) as usize,
        top_p: value["top_p"].as_f64().unwrap_or(0.95) as f32,
    })
}

fn load_eos_token_ids(model_dir: &std::path::Path) -> Result<BTreeSet<u32>> {
    let path = model_dir.join("generation_config.json");
    let value: Value =
        serde_json::from_str(&fs::read_to_string(&path).map_err(|err| Error::Format {
            label: "generation_config.json",
            detail: format!("{}: {err}", path.display()),
        })?)
        .map_err(|err| Error::Format {
            label: "generation_config.json",
            detail: err.to_string(),
        })?;
    let mut ids = BTreeSet::new();
    match &value["eos_token_id"] {
        Value::Number(number) => {
            if let Some(id) = number.as_u64() {
                ids.insert(id as u32);
            }
        }
        Value::Array(values) => {
            ids.extend(values.iter().filter_map(Value::as_u64).map(|id| id as u32));
        }
        _ => {}
    }
    Ok(ids)
}

fn sample_next_token<R: Rng + ?Sized>(
    logits: &[f32],
    config: SamplingConfig,
    rng: &mut R,
) -> Result<u32> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(Error::Format {
            label: "temperature",
            detail: "expected a finite non-negative value".to_string(),
        });
    }
    if !config.top_p.is_finite() || config.top_p <= 0.0 || config.top_p > 1.0 {
        return Err(Error::Format {
            label: "top-p",
            detail: "expected 0 < top-p <= 1".to_string(),
        });
    }
    if config.temperature == 0.0 || config.top_k == 1 {
        return Ok(argmax(logits).0 as u32);
    }

    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .collect::<Vec<_>>();
    candidates.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.truncate(config.top_k);
    }
    let max = candidates
        .first()
        .map(|(_, logit)| *logit)
        .ok_or_else(|| Error::Format {
            label: "sampling logits",
            detail: "no finite logits".to_string(),
        })?;
    let mut weighted = candidates
        .into_iter()
        .map(|(id, logit)| (id, ((logit - max) / config.temperature).exp()))
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    for (_, weight) in &mut weighted {
        *weight /= total;
    }
    let cutoff = weighted
        .iter()
        .scan(0.0, |sum, (_, probability)| {
            *sum += probability;
            Some(*sum)
        })
        .position(|sum| sum >= config.top_p)
        .map_or(weighted.len(), |index| index + 1);
    weighted.truncate(cutoff.max(1));

    let retained = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    let mut draw = rng.random::<f32>() * retained;
    for (id, weight) in weighted {
        if draw <= weight {
            return Ok(id as u32);
        }
        draw -= weight;
    }
    Ok(argmax(logits).0 as u32)
}

fn argmax(logits: &[f32]) -> (usize, f32) {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("Qwen3.6 vocabulary is non-empty")
}

#[cfg(test)]
mod tests {
    use super::{SamplingConfig, render_prompt, sample_next_token};
    use rand::{SeedableRng, rngs::StdRng};

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

    #[test]
    fn zero_temperature_selects_argmax() {
        let mut rng = StdRng::seed_from_u64(1);
        let token = sample_next_token(
            &[1.0, 3.0, 2.0],
            SamplingConfig {
                temperature: 0.0,
                top_k: 20,
                top_p: 0.95,
            },
            &mut rng,
        )
        .expect("sampling");
        assert_eq!(token, 1);
    }
}
