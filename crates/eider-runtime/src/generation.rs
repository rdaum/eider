//! Checkpoint-derived generation policy.

use crate::chat::{ChatReasoningEffort, ChatTemplateOptions};
use crate::sampling::SamplingConfig;
use eider_format::{Error, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use tokenizers::Tokenizer;

/// Request-level generation policy independent of an HTTP API schema.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Token selection policy.
    pub sampling: SamplingConfig,
    /// Maximum number of completion tokens.
    pub max_new_tokens: usize,
    /// Model-specific token IDs that terminate generation.
    pub eos_token_ids: BTreeSet<u32>,
    /// Text sequences that terminate generation without being emitted.
    pub stop_sequences: Vec<String>,
    /// Checkpoint-native chat-template defaults.
    pub chat_template: ChatTemplateOptions,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            max_new_tokens: 64,
            eos_token_ids: BTreeSet::new(),
            stop_sequences: Vec::new(),
            chat_template: ChatTemplateOptions {
                enable_thinking: false,
                ..ChatTemplateOptions::default()
            },
        }
    }
}

impl GenerationConfig {
    /// Loads checkpoint sampling defaults and EOS IDs.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let path = model_dir.join("generation_config.json");
        let contents = std::fs::read_to_string(&path).map_err(|error| Error::Format {
            label: "generation_config.json",
            detail: format!("{}: {error}", path.display()),
        })?;
        let value: Value = serde_json::from_str(&contents).map_err(|error| Error::Format {
            label: "generation_config.json",
            detail: error.to_string(),
        })?;
        let defaults = SamplingConfig::default();
        let eos_token_ids = parse_eos_token_ids(&value)?;
        let eos_token_ids = if eos_token_ids.is_empty() {
            let config_path = model_dir.join("config.json");
            let contents =
                std::fs::read_to_string(&config_path).map_err(|error| Error::Format {
                    label: "config.json",
                    detail: format!("{}: {error}", config_path.display()),
                })?;
            let model: Value = serde_json::from_str(&contents).map_err(|error| Error::Format {
                label: "config.json",
                detail: error.to_string(),
            })?;
            parse_eos_token_ids(&model)?
        } else {
            eos_token_ids
        };
        let mut eos_token_ids = eos_token_ids;
        add_tokenizer_eos_ids(model_dir, &mut eos_token_ids)?;
        let do_sample = value["do_sample"].as_bool().unwrap_or(false);
        let config = Self {
            sampling: SamplingConfig {
                temperature: if do_sample {
                    value["temperature"]
                        .as_f64()
                        .map_or(defaults.temperature, |value| value as f32)
                } else {
                    0.0
                },
                top_k: value["top_k"]
                    .as_u64()
                    .map_or(defaults.top_k, |value| value as usize),
                top_p: value["top_p"]
                    .as_f64()
                    .map_or(defaults.top_p, |value| value as f32),
                seed: value["seed"].as_u64(),
                presence_penalty: value["presence_penalty"]
                    .as_f64()
                    .map_or(0.0, |value| value as f32),
                frequency_penalty: value["frequency_penalty"]
                    .as_f64()
                    .map_or(0.0, |value| value as f32),
            },
            eos_token_ids,
            chat_template: checkpoint_chat_template_defaults(model_dir)?,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates request parameters before allocating decode state.
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if let Some(stop) = self.stop_sequences.iter().find(|stop| stop.is_empty()) {
            return Err(Error::Format {
                label: "stop sequence",
                detail: format!("stop sequences must not be empty: {stop:?}"),
            });
        }
        Ok(())
    }
}

fn checkpoint_chat_template_defaults(model_dir: &Path) -> Result<ChatTemplateOptions> {
    let path = model_dir.join("config.json");
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ChatTemplateOptions {
                enable_thinking: false,
                ..ChatTemplateOptions::default()
            });
        }
        Err(error) => {
            return Err(Error::Format {
                label: "config.json",
                detail: format!("{}: {error}", path.display()),
            });
        }
    };
    let config: Value = serde_json::from_str(&contents).map_err(|error| Error::Format {
        label: "config.json",
        detail: error.to_string(),
    })?;
    if config["model_type"].as_str() == Some("qwen3_5") {
        Ok(ChatTemplateOptions {
            enable_thinking: true,
            reasoning_effort: Some(ChatReasoningEffort::XHigh),
            ..ChatTemplateOptions::default()
        })
    } else {
        Ok(ChatTemplateOptions {
            enable_thinking: false,
            ..ChatTemplateOptions::default()
        })
    }
}

fn add_tokenizer_eos_ids(model_dir: &Path, ids: &mut BTreeSet<u32>) -> Result<()> {
    let config_path = model_dir.join("tokenizer_config.json");
    let contents = match std::fs::read_to_string(&config_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(Error::Format {
                label: "tokenizer_config.json",
                detail: format!("{}: {error}", config_path.display()),
            });
        }
    };
    let config: Value = serde_json::from_str(&contents).map_err(|error| Error::Format {
        label: "tokenizer_config.json",
        detail: error.to_string(),
    })?;
    let eos_tokens = tokenizer_eos_tokens(&config["eos_token"])?;
    if eos_tokens.is_empty() {
        return Ok(());
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| Error::Format {
        label: "tokenizer.json",
        detail: format!("{}: {error}", tokenizer_path.display()),
    })?;
    for token in eos_tokens {
        let id = tokenizer.token_to_id(&token).ok_or_else(|| Error::Format {
            label: "tokenizer_config.json",
            detail: format!("EOS token {token:?} is absent from tokenizer.json"),
        })?;
        ids.insert(id);
    }
    Ok(())
}

fn tokenizer_eos_tokens(value: &Value) -> Result<Vec<String>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::String(token) => Ok(vec![token.clone()]),
        Value::Object(object) => object
            .get("content")
            .and_then(Value::as_str)
            .map(|token| vec![token.to_string()])
            .ok_or_else(|| Error::Format {
                label: "tokenizer_config.json",
                detail: format!("expected eos_token object with string content, got {value}"),
            }),
        Value::Array(values) => {
            let mut tokens = Vec::with_capacity(values.len());
            for value in values {
                tokens.extend(tokenizer_eos_tokens(value)?);
            }
            Ok(tokens)
        }
        other => Err(Error::Format {
            label: "tokenizer_config.json",
            detail: format!("expected string, object, or array eos_token, got {other}"),
        }),
    }
}

fn parse_eos_token_ids(value: &Value) -> Result<BTreeSet<u32>> {
    let mut ids = BTreeSet::new();
    match &value["eos_token_id"] {
        Value::Null => {}
        Value::Number(number) => {
            ids.insert(json_token_id(number.as_u64(), "eos_token_id")?);
        }
        Value::Array(values) => {
            for value in values {
                ids.insert(json_token_id(value.as_u64(), "eos_token_id")?);
            }
        }
        other => {
            return Err(Error::Format {
                label: "generation_config.json",
                detail: format!("expected numeric eos_token_id or array, got {other}"),
            });
        }
    }
    Ok(ids)
}

fn json_token_id(value: Option<u64>, name: &'static str) -> Result<u32> {
    value
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| Error::Format {
            label: "generation_config.json",
            detail: format!("expected {name} in the u32 range"),
        })
}
