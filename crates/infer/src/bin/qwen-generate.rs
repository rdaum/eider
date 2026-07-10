use infer::nvfp4::{Error, Result};
use infer::qwen3::infer::Qwen3Model;
use infer::qwen3::layer0::DEFAULT_MODEL_DIR;
use rand::Rng;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

struct GenerateArgs {
    model_dir: PathBuf,
    prompt: PromptInput,
    max_new_tokens: usize,
    sampling: SamplingConfig,
}

enum PromptInput {
    Raw(String),
    Chat {
        system: Option<String>,
        message: String,
    },
}

#[derive(Clone, Copy, Debug)]
struct SamplingConfig {
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

fn main() -> Result<()> {
    let args = GenerateArgs::parse()?;
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let eos_token_ids = load_eos_token_ids(&args.model_dir)?;
    let prompt_text = args.prompt.render();
    let prompt_ids = encode_prompt(&tokenizer, &prompt_text)?;
    if prompt_ids.is_empty() {
        return Err(Error::Format {
            label: "prompt",
            detail: "tokenizer produced no token ids".to_string(),
        });
    }
    let model = Qwen3Model::load(&args.model_dir)?;
    validate_token_ids("prompt token id", &prompt_ids, model.vocab_size())?;

    println!("Qwen3 text generation");
    println!("  model dir: {}", args.model_dir.display());
    println!("  prompt tokens: {}", prompt_ids.len());
    println!("  max new tokens: {}", args.max_new_tokens);
    println!("  eos token ids: {:?}", eos_token_ids);
    println!(
        "  sampling: temperature={} top_k={} top_p={}",
        args.sampling.temperature, args.sampling.top_k, args.sampling.top_p
    );

    let mut state = model.new_decode_state(prompt_ids.len() + args.max_new_tokens)?;

    let mut generated_ids = Vec::with_capacity(args.max_new_tokens);
    let mut stopped_on_eos = None;
    let mut rng = rand::rng();
    let mut next_token = sample_next_token(
        &model.prefill_logits(&mut state, &prompt_ids)?.logits,
        args.sampling,
        &mut rng,
    )?;
    state.last_token = Some(next_token);

    for _ in 0..args.max_new_tokens {
        if eos_token_ids.contains(&next_token) {
            stopped_on_eos = Some(next_token);
            break;
        }
        generated_ids.push(next_token);
        next_token = sample_next_token(
            &model.decode_one_logits(&mut state, next_token)?.logits,
            args.sampling,
            &mut rng,
        )?;
        state.last_token = Some(next_token);
    }

    let generated_text = decode_tokens(&tokenizer, &generated_ids)?;
    let full_ids = prompt_ids
        .iter()
        .copied()
        .chain(generated_ids.iter().copied())
        .collect::<Vec<_>>();
    let full_text = decode_tokens(&tokenizer, &full_ids)?;

    println!("generated token ids: {:?}", generated_ids);
    if let Some(token_id) = stopped_on_eos {
        println!("stopped on eos token: {token_id}");
    }
    println!("generated text:\n{generated_text}");
    println!("full text:\n{full_text}");

    Ok(())
}

impl GenerateArgs {
    fn parse() -> Result<Self> {
        let mut model_dir = PathBuf::from(DEFAULT_MODEL_DIR);
        let mut prompt = None;
        let mut chat_message = None;
        let mut system = None;
        let mut max_new_tokens = 64;
        let mut sampling = SamplingConfig::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model_dir = PathBuf::from(args.next().ok_or_else(|| Error::Format {
                        label: "--model",
                        detail: "expected model directory".to_string(),
                    })?);
                }
                "--prompt" => {
                    prompt = Some(args.next().ok_or_else(|| Error::Format {
                        label: "--prompt",
                        detail: "expected prompt text".to_string(),
                    })?);
                }
                "--chat-message" => {
                    chat_message = Some(args.next().ok_or_else(|| Error::Format {
                        label: "--chat-message",
                        detail: "expected user message text".to_string(),
                    })?);
                }
                "--system" => {
                    system = Some(args.next().ok_or_else(|| Error::Format {
                        label: "--system",
                        detail: "expected system message text".to_string(),
                    })?);
                }
                "--max-new-tokens" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--max-new-tokens",
                        detail: "expected token count".to_string(),
                    })?;
                    max_new_tokens = value.parse::<usize>().map_err(|err| Error::Format {
                        label: "--max-new-tokens",
                        detail: err.to_string(),
                    })?;
                }
                "--temperature" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--temperature",
                        detail: "expected temperature".to_string(),
                    })?;
                    sampling.temperature = value.parse::<f32>().map_err(|err| Error::Format {
                        label: "--temperature",
                        detail: err.to_string(),
                    })?;
                }
                "--top-k" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--top-k",
                        detail: "expected token count".to_string(),
                    })?;
                    sampling.top_k = value.parse::<usize>().map_err(|err| Error::Format {
                        label: "--top-k",
                        detail: err.to_string(),
                    })?;
                }
                "--top-p" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--top-p",
                        detail: "expected probability mass".to_string(),
                    })?;
                    sampling.top_p = value.parse::<f32>().map_err(|err| Error::Format {
                        label: "--top-p",
                        detail: err.to_string(),
                    })?;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::Format {
                        label: "argument",
                        detail: format!("unknown argument {other:?}"),
                    });
                }
            }
        }

        let prompt = match (prompt, chat_message) {
            (Some(prompt), None) => PromptInput::Raw(prompt),
            (None, Some(message)) => PromptInput::Chat { system, message },
            (None, None) => {
                return Err(Error::Format {
                    label: "prompt",
                    detail: "--prompt or --chat-message is required".to_string(),
                });
            }
            (Some(_), Some(_)) => {
                return Err(Error::Format {
                    label: "prompt",
                    detail: "use only one of --prompt or --chat-message".to_string(),
                });
            }
        };
        Ok(Self {
            model_dir,
            prompt,
            max_new_tokens,
            sampling,
        })
    }
}

impl PromptInput {
    fn render(&self) -> String {
        match self {
            Self::Raw(prompt) => prompt.clone(),
            Self::Chat { system, message } => {
                let mut prompt = String::new();
                if let Some(system) = system {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str(system);
                    prompt.push_str("<|im_end|>\n");
                }
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(message);
                prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
                prompt
            }
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
        }
    }
}

fn print_usage() {
    println!(
        "usage: qwen-generate --model models/qwen3-8b-nvfp4 (--prompt TEXT | --chat-message TEXT [--system TEXT]) [--max-new-tokens N] [--temperature T] [--top-k K] [--top-p P]"
    );
}

fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&tokenizer_path).map_err(|err| Error::Format {
        label: "tokenizer.json",
        detail: format!("{}: {err}", tokenizer_path.display()),
    })
}

fn load_eos_token_ids(model_dir: &Path) -> Result<BTreeSet<u32>> {
    let path = model_dir.join("generation_config.json");
    let json = fs::read_to_string(&path).map_err(|err| Error::Format {
        label: "generation_config.json",
        detail: format!("{}: {err}", path.display()),
    })?;
    let value = serde_json::from_str::<Value>(&json).map_err(|err| Error::Format {
        label: "generation_config.json",
        detail: err.to_string(),
    })?;
    let eos = value.get("eos_token_id").ok_or_else(|| Error::Format {
        label: "generation_config.json",
        detail: "missing eos_token_id".to_string(),
    })?;

    let mut ids = BTreeSet::new();
    match eos {
        Value::Number(number) => {
            ids.insert(json_u32("eos_token_id", number)?);
        }
        Value::Array(values) => {
            for value in values {
                match value {
                    Value::Number(number) => {
                        ids.insert(json_u32("eos_token_id", number)?);
                    }
                    other => {
                        return Err(Error::Format {
                            label: "generation_config.json",
                            detail: format!("expected numeric eos_token_id, got {other}"),
                        });
                    }
                }
            }
        }
        other => {
            return Err(Error::Format {
                label: "generation_config.json",
                detail: format!("expected number or array for eos_token_id, got {other}"),
            });
        }
    }
    Ok(ids)
}

fn json_u32(label: &'static str, number: &serde_json::Number) -> Result<u32> {
    let value = number.as_u64().ok_or_else(|| Error::Format {
        label,
        detail: format!("expected unsigned integer, got {number}"),
    })?;
    u32::try_from(value).map_err(|err| Error::Format {
        label,
        detail: err.to_string(),
    })
}

fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(prompt, true)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|err| Error::Format {
            label: "prompt encode",
            detail: err.to_string(),
        })
}

fn decode_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(token_ids, true)
        .map_err(|err| Error::Format {
            label: "token decode",
            detail: err.to_string(),
        })
}

fn validate_token_ids(label: &'static str, token_ids: &[u32], vocab_size: usize) -> Result<()> {
    for &token_id in token_ids {
        if token_id as usize >= vocab_size {
            return Err(Error::Shape {
                label,
                expected: format!("token < {vocab_size}"),
                actual: token_id.to_string(),
            });
        }
    }
    Ok(())
}

fn sample_next_token<R: Rng + ?Sized>(
    logits: &[f32],
    config: SamplingConfig,
    rng: &mut R,
) -> Result<u32> {
    if logits.is_empty() {
        return Err(Error::Shape {
            label: "sampling logits",
            expected: "at least one logit".to_string(),
            actual: "0 logits".to_string(),
        });
    }
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(Error::Format {
            label: "--temperature",
            detail: format!(
                "expected finite non-negative temperature, got {}",
                config.temperature
            ),
        });
    }
    if !config.top_p.is_finite() || config.top_p <= 0.0 || config.top_p > 1.0 {
        return Err(Error::Format {
            label: "--top-p",
            detail: format!("expected 0 < top_p <= 1, got {}", config.top_p),
        });
    }

    if config.temperature == 0.0 || config.top_k == 1 {
        return Ok(argmax(logits).0);
    }

    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(Error::Format {
            label: "sampling logits",
            detail: "no finite logits".to_string(),
        });
    }

    candidates.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.truncate(config.top_k);
    }

    let max_logit = candidates[0].1;
    let mut weighted = candidates
        .into_iter()
        .map(|(idx, logit)| (idx, ((logit - max_logit) / config.temperature).exp()))
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Ok(argmax(logits).0);
    }
    for (_, weight) in &mut weighted {
        *weight /= total;
    }

    weighted.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut cutoff = weighted.len();
    for (idx, (_, probability)) in weighted.iter().enumerate() {
        cumulative += *probability;
        if cumulative >= config.top_p {
            cutoff = idx + 1;
            break;
        }
    }
    weighted.truncate(cutoff.max(1));

    let renormalized_total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    let mut draw = rng.random::<f32>() * renormalized_total;
    for (idx, weight) in weighted {
        if draw <= weight {
            return Ok(idx as u32);
        }
        draw -= weight;
    }
    Ok(argmax(logits).0)
}

fn argmax(logits: &[f32]) -> (u32, f32) {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(idx, value)| (idx as u32, value))
        .unwrap_or((0, f32::NEG_INFINITY))
}
