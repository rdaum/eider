use eider_cuda::{Error, Result};
use eider_runtime::sampling::{Sampler, SamplingConfig, TokenHistory};
use infer::qwen3::infer::Qwen3Model;
use infer::qwen3::layer0::DEFAULT_MODEL_DIR;
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
        "  sampling: temperature={} top_k={} top_p={} seed={:?} presence_penalty={} frequency_penalty={}",
        args.sampling.temperature,
        args.sampling.top_k,
        args.sampling.top_p,
        args.sampling.seed,
        args.sampling.presence_penalty,
        args.sampling.frequency_penalty,
    );

    let mut state = model.new_decode_state(prompt_ids.len() + args.max_new_tokens)?;

    let mut generated_ids = Vec::with_capacity(args.max_new_tokens);
    let mut stopped_on_eos = None;
    let mut history = TokenHistory::from_tokens(prompt_ids.iter().copied());
    let mut sampler = Sampler::new(args.sampling)?;
    let mut next_token = sampler
        .sample(
            &model.prefill_logits(&mut state, &prompt_ids)?.logits,
            &history,
        )?
        .id;
    state.last_token = Some(next_token);

    for _ in 0..args.max_new_tokens {
        if eos_token_ids.contains(&next_token) {
            stopped_on_eos = Some(next_token);
            break;
        }
        generated_ids.push(next_token);
        history.push(next_token);
        next_token = sampler
            .sample(
                &model.decode_one_logits(&mut state, next_token)?.logits,
                &history,
            )?
            .id;
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
        let mut sampling = SamplingConfig {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            ..SamplingConfig::default()
        };
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
                "--seed" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--seed",
                        detail: "expected unsigned integer".to_string(),
                    })?;
                    sampling.seed = Some(value.parse::<u64>().map_err(|err| Error::Format {
                        label: "--seed",
                        detail: err.to_string(),
                    })?);
                }
                "--presence-penalty" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--presence-penalty",
                        detail: "expected penalty".to_string(),
                    })?;
                    sampling.presence_penalty =
                        value.parse::<f32>().map_err(|err| Error::Format {
                            label: "--presence-penalty",
                            detail: err.to_string(),
                        })?;
                }
                "--frequency-penalty" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--frequency-penalty",
                        detail: "expected penalty".to_string(),
                    })?;
                    sampling.frequency_penalty =
                        value.parse::<f32>().map_err(|err| Error::Format {
                            label: "--frequency-penalty",
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

fn print_usage() {
    println!(
        "usage: qwen-generate --model models/qwen3-8b-nvfp4 (--prompt TEXT | --chat-message TEXT [--system TEXT]) [--max-new-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed N] [--presence-penalty P] [--frequency-penalty P]"
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
