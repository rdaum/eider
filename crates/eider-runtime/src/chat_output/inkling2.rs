//! Inkling 2 token-ID framing for ordinary text and reasoning.
//!
//! Control-marker strings are not recognized inside normal decoded text. Tool
//! invocation is rejected until its complete native grammar is implemented.
use super::ChatOutputEvent;
use eider_format::{Error, Result};
use tokenizers::Tokenizer;

pub(super) struct SpecialTokens {
    model: u32,
    thinking: u32,
    text: u32,
    invoke: u32,
    end_message: u32,
    eos: Option<u32>,
}

impl SpecialTokens {
    pub fn new(tokenizer: &Tokenizer) -> Option<Self> {
        Some(Self {
            model: tokenizer.token_to_id("<|message_model|>")?,
            thinking: tokenizer.token_to_id("<|content_thinking|>")?,
            text: tokenizer.token_to_id("<|content_text|>")?,
            invoke: tokenizer.token_to_id("<|content_invoke_tool_json|>")?,
            end_message: tokenizer.token_to_id("<|end_message|>")?,
            eos: tokenizer.token_to_id("<|content_model_end_sampling|>"),
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Boundary,
    Header,
    Thinking,
    Text,
}

pub(super) struct State {
    mode: Mode,
}

impl State {
    pub fn new() -> Self {
        Self {
            mode: Mode::Boundary,
        }
    }

    pub fn is_reasoning(&self) -> bool {
        self.mode == Mode::Thinking
    }

    /// Returns true only for a real Inkling control-token ID.
    pub fn control(&mut self, tokens: &SpecialTokens, id: u32) -> Result<bool> {
        if id == tokens.model {
            self.mode = Mode::Header;
        } else if id == tokens.thinking {
            self.mode = Mode::Thinking;
        } else if id == tokens.text {
            self.mode = Mode::Text;
        } else if id == tokens.end_message {
            self.mode = Mode::Boundary;
        } else if id == tokens.invoke {
            return Err(Error::Format {
                label: "Inkling 2 output",
                detail: "tool invocation is not supported by the text-only adapter".into(),
            });
        } else if tokens.eos != Some(id) {
            return Ok(false);
        }
        Ok(true)
    }

    pub fn push_text(&mut self, text: &str) -> Vec<ChatOutputEvent> {
        match self.mode {
            // Inkling samples its assistant header and optional routing name.
            Mode::Header => Vec::new(),
            Mode::Thinking if !text.is_empty() => {
                vec![ChatOutputEvent::Reasoning(text.to_string())]
            }
            Mode::Boundary | Mode::Text if !text.is_empty() => {
                if self.mode == Mode::Boundary {
                    self.mode = Mode::Text;
                }
                vec![ChatOutputEvent::Text(text.to_string())]
            }
            _ => Vec::new(),
        }
    }

    /// Normal completion closes an open content block. Abort drops the parser instead.
    pub fn finish(&mut self) -> Vec<ChatOutputEvent> {
        self.mode = Mode::Boundary;
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{AddedToken, models::bpe::BPE};

    fn tokens() -> SpecialTokens {
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(
            &[
                "<|message_model|>",
                "<|content_thinking|>",
                "<|content_text|>",
                "<|content_invoke_tool_json|>",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
            ]
            .map(|marker| AddedToken::from(marker, true)),
        );
        SpecialTokens::new(&tokenizer).unwrap()
    }

    #[test]
    fn header_thinking_and_text_control_real_token_ids() {
        let tokens = tokens();
        let mut state = State::new();
        assert_eq!(
            state.push_text("before"),
            vec![ChatOutputEvent::Text("before".into())]
        );
        state.control(&tokens, tokens.model).unwrap();
        assert!(state.push_text("recipient").is_empty());
        state.control(&tokens, tokens.thinking).unwrap();
        assert!(state.is_reasoning());
        assert_eq!(
            state.push_text("reason"),
            vec![ChatOutputEvent::Reasoning("reason".into())]
        );
        state.control(&tokens, tokens.end_message).unwrap();
        state.control(&tokens, tokens.model).unwrap();
        state.control(&tokens, tokens.text).unwrap();
        assert!(!state.is_reasoning());
        assert_eq!(
            state.push_text("answer"),
            vec![ChatOutputEvent::Text("answer".into())]
        );
        state.control(&tokens, tokens.end_message).unwrap();
        assert!(state.finish().is_empty());
    }

    #[test]
    fn ordinary_marker_text_is_not_protocol() {
        let mut state = State::new();
        assert_eq!(
            state.push_text("<|message_model|> is literal text"),
            vec![ChatOutputEvent::Text(
                "<|message_model|> is literal text".into()
            )]
        );
    }
}
