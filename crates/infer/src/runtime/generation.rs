//! Reusable request-scoped generation sessions.

use super::sampling::{SampledToken, Sampler, SamplingConfig, TokenHistory};
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
        let path = model_dir.as_ref().join("generation_config.json");
        let contents = std::fs::read_to_string(&path).map_err(|error| Error::Format {
            label: "generation_config.json",
            detail: format!("{}: {error}", path.display()),
        })?;
        let value: Value = serde_json::from_str(&contents).map_err(|error| Error::Format {
            label: "generation_config.json",
            detail: error.to_string(),
        })?;
        let defaults = SamplingConfig::default();
        let config = Self {
            sampling: SamplingConfig {
                temperature: value["temperature"]
                    .as_f64()
                    .map_or(defaults.temperature, |value| value as f32),
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
            eos_token_ids: parse_eos_token_ids(&value)?,
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

struct StopOutput {
    text: String,
    matched: Option<String>,
}

struct StopBuffer {
    sequences: Vec<String>,
    pending: String,
}

impl StopBuffer {
    fn new(sequences: Vec<String>) -> Self {
        Self {
            sequences,
            pending: String::new(),
        }
    }

    fn push(&mut self, chunk: &str) -> StopOutput {
        self.pending.push_str(chunk);
        if let Some((index, sequence)) = self.earliest_match() {
            let text = self.pending[..index].to_string();
            self.pending.clear();
            return StopOutput {
                text,
                matched: Some(sequence),
            };
        }

        let holdback = self
            .pending
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(self.pending.len()))
            .filter_map(|index| {
                let suffix = &self.pending[index..];
                self.sequences
                    .iter()
                    .any(|sequence| sequence.starts_with(suffix))
                    .then_some(suffix.len())
            })
            .max()
            .unwrap_or(0);
        let emit_len = self.pending.len() - holdback;
        let text = self.pending[..emit_len].to_string();
        self.pending.drain(..emit_len);
        StopOutput {
            text,
            matched: None,
        }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn earliest_match(&self) -> Option<(usize, String)> {
        self.sequences
            .iter()
            .filter_map(|sequence| {
                self.pending
                    .find(sequence)
                    .map(|index| (index, sequence.clone()))
            })
            .min_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.len().cmp(&left.1.len()))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, StopBuffer};
    use std::fs;

    #[test]
    fn split_stop_sequence_is_never_emitted() {
        let mut buffer = StopBuffer::new(vec!["END".to_string()]);
        let first = buffer.push("hello E");
        assert_eq!(first.text, "hello ");
        assert_eq!(first.matched, None);
        let second = buffer.push("ND ignored");
        assert_eq!(second.text, "");
        assert_eq!(second.matched.as_deref(), Some("END"));
    }

    #[test]
    fn unmatched_stop_prefix_flushes_at_length_limit() {
        let mut buffer = StopBuffer::new(vec!["END".to_string()]);
        let output = buffer.push("hello E");
        assert_eq!(output.text, "hello ");
        assert_eq!(buffer.finish(), "E");
    }

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
            r#"{"temperature":0.7,"top_k":40,"top_p":0.8,"eos_token_id":[1,2]}"#,
        )
        .expect("write generation config");

        let config = GenerationConfig::from_model_dir(&directory).expect("generation config");
        assert_eq!(config.sampling.temperature, 0.7);
        assert_eq!(config.sampling.top_k, 40);
        assert_eq!(config.sampling.top_p, 0.8);
        assert_eq!(config.eos_token_ids.into_iter().collect::<Vec<_>>(), [1, 2]);

        fs::remove_dir_all(directory).expect("remove config directory");
    }
}
