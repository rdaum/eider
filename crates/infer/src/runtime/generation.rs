//! Reusable request-scoped generation sessions.

use super::sampling::{SampledToken, Sampler, SamplingConfig, TokenHistory};
use super::stop::StopBuffer;
use crate::qwen3::qwen36::{Qwen36DecodeState, Qwen36TextModel};
use nvfp4::{Error, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{DecodeStream, Tokenizer};

type TokenizerDecodeStream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

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
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingConfig::default(),
            max_new_tokens: 64,
            eos_token_ids: BTreeSet::new(),
            stop_sequences: Vec::new(),
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

/// Why a generation session stopped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationFinishReason {
    /// The model selected a configured EOS token.
    Eos,
    /// Generated text matched a configured stop sequence.
    StopSequence(String),
    /// The request reached `max_new_tokens`.
    Length,
}

/// One generated token and the text now safe to stream to a client.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedToken {
    /// Selected vocabulary ID.
    pub id: u32,
    /// Original model logit for `id`.
    pub logit: f32,
    /// Text that cannot be part of a future stop-sequence match.
    pub text: String,
    /// Present on the final token in the session.
    pub finish_reason: Option<GenerationFinishReason>,
}

/// One Qwen3.6 generation request with isolated decode and sampling state.
pub struct Qwen36GenerationSession<'a> {
    model: &'a Qwen36TextModel,
    state: Qwen36DecodeState,
    sampler: Sampler,
    decode_stream: TokenizerDecodeStream<'a>,
    config: GenerationConfig,
    prompt_tokens: Vec<u32>,
    history: TokenHistory,
    stop_buffer: StopBuffer,
    prefilled: bool,
    last_token: Option<u32>,
    generated_tokens: usize,
    finish_reason: Option<GenerationFinishReason>,
}

impl<'a> Qwen36GenerationSession<'a> {
    /// Creates a request session for an already-tokenized non-empty prompt.
    pub fn new(
        model: &'a Qwen36TextModel,
        tokenizer: &'a Tokenizer,
        prompt_tokens: &[u32],
        config: GenerationConfig,
    ) -> Result<Self> {
        config.validate()?;
        if prompt_tokens.is_empty() {
            return Err(Error::Format {
                label: "generation prompt",
                detail: "prompt tokenized to zero tokens".to_string(),
            });
        }
        if let Some(&token) = prompt_tokens
            .iter()
            .find(|&&token| token as usize >= model.manifest().vocab)
        {
            return Err(Error::Shape {
                label: "generation prompt token",
                expected: format!("token < {}", model.manifest().vocab),
                actual: token.to_string(),
            });
        }
        let max_tokens = prompt_tokens
            .len()
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| Error::Shape {
                label: "generation capacity",
                expected: "prompt + completion length without overflow".to_string(),
                actual: format!("{} + {}", prompt_tokens.len(), config.max_new_tokens),
            })?
            .max(1);
        let state = model.new_decode_state(max_tokens)?;
        let sampler = Sampler::new(config.sampling)?;
        let finish_reason = (config.max_new_tokens == 0).then_some(GenerationFinishReason::Length);
        Ok(Self {
            model,
            state,
            sampler,
            decode_stream: tokenizer.decode_stream(true),
            prompt_tokens: prompt_tokens.to_vec(),
            history: TokenHistory::from_tokens(prompt_tokens.iter().copied()),
            stop_buffer: StopBuffer::new(config.stop_sequences.clone()),
            config,
            prefilled: false,
            last_token: None,
            generated_tokens: 0,
            finish_reason,
        })
    }

    /// Generates the next token, or returns `None` after the session finishes.
    pub fn next_token(&mut self) -> Result<Option<GeneratedToken>> {
        if self.finish_reason.is_some() {
            return Ok(None);
        }
        let input = self.next_input_token()?;
        let sampled = self.decode_and_select(input)?;
        self.generated_tokens += 1;
        self.last_token = Some(sampled.id);
        self.history.push(sampled.id);

        let mut finish_reason = None;
        let mut text = String::new();
        if self.config.eos_token_ids.contains(&sampled.id) {
            text.push_str(&self.stop_buffer.finish());
            finish_reason = Some(GenerationFinishReason::Eos);
        } else if let Some(chunk) =
            self.decode_stream
                .step(sampled.id)
                .map_err(|error| Error::Format {
                    label: "tokenizer decode stream",
                    detail: error.to_string(),
                })?
        {
            let output = self.stop_buffer.push(&chunk);
            text.push_str(&output.text);
            finish_reason = output.matched.map(GenerationFinishReason::StopSequence);
        }

        if finish_reason.is_none() && self.generated_tokens == self.config.max_new_tokens {
            text.push_str(&self.stop_buffer.finish());
            finish_reason = Some(GenerationFinishReason::Length);
        }
        if let Some(reason) = &finish_reason {
            self.finish_reason = Some(reason.clone());
        }
        Ok(Some(GeneratedToken {
            id: sampled.id,
            logit: sampled.logit,
            text,
            finish_reason,
        }))
    }

    /// Returns the final reason after the session has stopped.
    pub fn finish_reason(&self) -> Option<&GenerationFinishReason> {
        self.finish_reason.as_ref()
    }

    /// Returns the number of completion tokens selected so far.
    pub fn generated_token_count(&self) -> usize {
        self.generated_tokens
    }

    /// Returns prompt and generated token history used by penalties.
    pub fn history(&self) -> &TokenHistory {
        &self.history
    }

    fn next_input_token(&mut self) -> Result<u32> {
        if self.prefilled {
            return self.last_token.ok_or_else(|| Error::Format {
                label: "generation session",
                detail: "prefilled session has no generated token".to_string(),
            });
        }
        for index in 0..self.prompt_tokens.len() - 1 {
            self.model
                .decode_one_token(&mut self.state, self.prompt_tokens[index])?;
        }
        self.prefilled = true;
        Ok(*self.prompt_tokens.last().expect("non-empty prompt"))
    }

    fn decode_and_select(&mut self, input: u32) -> Result<SampledToken> {
        if self.sampler.config().uses_fast_argmax() {
            let token = self.model.decode_one_token(&mut self.state, input)?;
            return Ok(SampledToken {
                id: token.id,
                logit: token.value,
                adjusted_logit: token.value,
            });
        }
        let logits = self
            .model
            .decode_one_token_logits(&mut self.state, input)?
            .logits;
        self.sampler.sample(&logits, &self.history)
    }
}

#[cfg(test)]
mod tests {
    use super::GenerationConfig;
    use std::fs;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{AddedToken, Tokenizer};

    #[test]
    fn model_generation_config_loads_sampling_and_eos_defaults() {
        let directory = std::env::temp_dir().join(format!(
            "eider-generation-config-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory).expect("create config directory");
        fs::write(
            directory.join("generation_config.json"),
            r#"{"do_sample":true,"temperature":0.7,"top_k":40,"top_p":0.8,"eos_token_id":[1,2]}"#,
        )
        .expect("write generation config");

        let config = GenerationConfig::from_model_dir(&directory).expect("generation config");
        assert_eq!(config.sampling.temperature, 0.7);
        assert_eq!(config.sampling.top_k, 40);
        assert_eq!(config.sampling.top_p, 0.8);
        assert_eq!(config.eos_token_ids.into_iter().collect::<Vec<_>>(), [1, 2]);

        fs::remove_dir_all(directory).expect("remove config directory");
    }

    #[test]
    fn model_config_supplies_eos_when_generation_config_omits_it() {
        let directory = std::env::temp_dir().join(format!(
            "eider-generation-model-config-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory).expect("create config directory");
        fs::write(
            directory.join("generation_config.json"),
            r#"{"temperature":0.6,"top_p":0.95}"#,
        )
        .expect("write generation config");
        fs::write(
            directory.join("config.json"),
            r#"{"eos_token_id":[1,2,128007]}"#,
        )
        .expect("write model config");

        let config = GenerationConfig::from_model_dir(&directory).expect("generation config");
        assert_eq!(config.sampling.temperature, 0.0);
        assert_eq!(
            config.eos_token_ids.into_iter().collect::<Vec<_>>(),
            [1, 2, 128007]
        );

        fs::remove_dir_all(directory).expect("remove config directory");
    }

    #[test]
    fn tokenizer_declared_eos_is_added_to_stale_generation_ids() {
        let directory = std::env::temp_dir().join(format!(
            "eider-tokenizer-eos-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory).expect("create config directory");
        fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id":1}"#,
        )
        .expect("write generation config");
        fs::write(
            directory.join("tokenizer_config.json"),
            r#"{"eos_token":"<|im_end|>"}"#,
        )
        .expect("write tokenizer config");
        let model = WordLevel::builder()
            .vocab(
                [("[UNK]".to_string(), 0), ("ordinary".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .expect("word-level model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
        tokenizer
            .save(directory.join("tokenizer.json"), false)
            .expect("write tokenizer");

        let config = GenerationConfig::from_model_dir(&directory).expect("generation config");
        assert_eq!(config.eos_token_ids.into_iter().collect::<Vec<_>>(), [1, 2]);

        fs::remove_dir_all(directory).expect("remove config directory");
    }
}
