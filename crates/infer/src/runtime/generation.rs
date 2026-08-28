//! Reusable request-scoped generation sessions.

use super::scheduler::{Qwen36RequestId, Qwen36Scheduler};
use crate::nemotron3::Nemotron3Model;
use crate::nemotron3::{Nemotron3Sequence, Nemotron3SequenceCache, new_nemotron3_sequence_cache};
use crate::qwen3::qwen36::Qwen36TextModel;
use eider_cuda::{Error, Result};
use eider_runtime::cache::SequenceCacheConfig;
use eider_runtime::generation::GenerationConfig;
use eider_runtime::sampling::{SampledToken, Sampler, TokenHistory};
use eider_runtime::scheduler::{RequestConfig, RequestFinishReason, SchedulerConfig};
use eider_runtime::stop::StopBuffer;
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

/// Why a generation session stopped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationFinishReason {
    /// The model selected a configured EOS token.
    Eos,
    /// Generated text matched a configured stop sequence.
    StopSequence(String),
    /// The request reached `max_new_tokens`.
    Length,
    /// A request-scoped tool grammar completed a function call.
    ToolCalls,
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
    scheduler: Qwen36Scheduler<'a>,
    request_id: Qwen36RequestId,
    decode_stream: TokenizerDecodeStream<'a>,
    history: TokenHistory,
    stop_buffer: StopBuffer,
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
        let scheduler_config = SchedulerConfig {
            decode_capacity: 1,
            prefill_sequence_capacity: 1,
            prefill_token_capacity: max_tokens
                .min(SchedulerConfig::default().prefill_token_capacity),
            max_active_sequences: 1,
            max_context_tokens: max_tokens,
            speculative_drafts: 0,
        };
        let mut scheduler = Qwen36Scheduler::new_with_cache_config(
            model,
            scheduler_config,
            SequenceCacheConfig {
                max_retained_bytes: 0,
            },
        )?;
        let request_id = scheduler.add_request(
            prompt_tokens.to_vec(),
            RequestConfig {
                sampling: config.sampling,
                max_new_tokens: config.max_new_tokens,
                eos_token_ids: config.eos_token_ids.clone(),
            },
        )?;
        let finish_reason = (config.max_new_tokens == 0).then_some(GenerationFinishReason::Length);
        Ok(Self {
            scheduler,
            request_id,
            decode_stream: tokenizer.decode_stream(true),
            history: TokenHistory::from_tokens(prompt_tokens.iter().copied()),
            stop_buffer: StopBuffer::new(config.stop_sequences.clone()),
            generated_tokens: 0,
            finish_reason,
        })
    }

    /// Generates the next token, or returns `None` after the session finishes.
    pub fn next_token(&mut self) -> Result<Option<GeneratedToken>> {
        if self.finish_reason.is_some() {
            return Ok(None);
        }
        let sampled = loop {
            let tick = self.scheduler.tick()?;
            if let Some(token) = tick
                .generated
                .into_iter()
                .find(|token| token.request_id == self.request_id)
            {
                break token;
            }
        };
        self.generated_tokens += 1;
        self.history.push(sampled.id);

        let mut finish_reason = sampled.finish_reason.map(|reason| match reason {
            RequestFinishReason::Eos => GenerationFinishReason::Eos,
            RequestFinishReason::Length => GenerationFinishReason::Length,
            RequestFinishReason::ToolCalls => GenerationFinishReason::ToolCalls,
        });
        let mut text = String::new();
        if finish_reason == Some(GenerationFinishReason::Eos) {
            text.push_str(&self.stop_buffer.finish());
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

        if matches!(
            finish_reason,
            Some(GenerationFinishReason::Length | GenerationFinishReason::ToolCalls)
        ) {
            text.push_str(&self.stop_buffer.finish());
        }
        if let Some(reason) = &finish_reason {
            self.finish_reason = Some(reason.clone());
            if sampled.finish_reason.is_none() {
                self.scheduler.cancel_request(self.request_id);
            }
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
}

/// One Nemotron 3 generation request with isolated recurrent, KV, and sampling state.
pub struct Nemotron3GenerationSession<'a> {
    model: &'a Nemotron3Model,
    sequence: Nemotron3Sequence,
    sequence_cache: Nemotron3SequenceCache,
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

impl<'a> Nemotron3GenerationSession<'a> {
    /// Creates a request session for an already-tokenized non-empty prompt.
    pub fn new(
        model: &'a Nemotron3Model,
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
            .find(|&&token| token as usize >= model.manifest().vocab_size)
        {
            return Err(Error::Shape {
                label: "generation prompt token",
                expected: format!("token < {}", model.manifest().vocab_size),
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
        let mut sequence_cache = new_nemotron3_sequence_cache(model, 1, max_tokens)?;
        let sequence = Nemotron3Sequence::admit(model, &mut sequence_cache, max_tokens)?;
        let sampler = Sampler::new(config.sampling)?;
        let finish_reason = (config.max_new_tokens == 0).then_some(GenerationFinishReason::Length);
        Ok(Self {
            model,
            sequence,
            sequence_cache,
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

    fn next_input_token(&mut self) -> Result<u32> {
        if self.prefilled {
            return self.last_token.ok_or_else(|| Error::Format {
                label: "generation session",
                detail: "prefilled session has no generated token".to_string(),
            });
        }
        for index in 0..self.prompt_tokens.len() - 1 {
            self.model.forward_one(
                &mut self.sequence,
                &mut self.sequence_cache,
                self.prompt_tokens[index],
            )?;
        }
        self.prefilled = true;
        Ok(*self.prompt_tokens.last().expect("non-empty prompt"))
    }

    fn decode_and_select(&mut self, input: u32) -> Result<SampledToken> {
        self.model
            .forward_one(&mut self.sequence, &mut self.sequence_cache, input)?;
        if self.sampler.config().uses_fast_argmax() {
            let (id, logit) = self.model.argmax_with_logit(&mut self.sequence)?;
            return Ok(SampledToken {
                id,
                logit,
                adjusted_logit: logit,
            });
        }
        let logits = self.model.logits_to_host(&self.sequence)?;
        Ok(self.sampler.sample(&logits, &self.history)?)
    }
}

#[cfg(test)]
mod tests {
    use eider_runtime::chat::ChatReasoningEffort;
    use eider_runtime::generation::GenerationConfig;
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
    fn qwen35_dense_uses_checkpoint_thinking_default() {
        let directory = std::env::temp_dir().join(format!(
            "eider-qwen38-chat-default-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory).expect("create config directory");
        fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id":[248046,248044]}"#,
        )
        .expect("write generation config");
        fs::write(directory.join("config.json"), r#"{"model_type":"qwen3_5"}"#)
            .expect("write model config");

        let config = GenerationConfig::from_model_dir(&directory).expect("generation config");
        assert!(config.chat_template.enable_thinking);
        assert_eq!(
            config.chat_template.reasoning_effort,
            Some(ChatReasoningEffort::XHigh)
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
