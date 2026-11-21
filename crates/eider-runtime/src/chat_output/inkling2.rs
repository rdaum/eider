//! Inkling 2 token-ID framing for text, reasoning, and JSON tool invocation.
//!
//! Every block opens with `<|message_model|>`, an optional author name, and a
//! content-kind token, and closes with `<|end_message|>` or the end-of-sampling
//! token. Blocks repeat in any order. Control-marker strings inside decoded
//! text are not protocol.
use super::{ChatOutputEvent, next_tool_call_id};
use crate::chat::{ChatFunctionCall, ChatToolCall};
use eider_format::{Error, Result};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use tokenizers::Tokenizer;

pub(super) struct SpecialTokens {
    model: u32,
    thinking: u32,
    text: u32,
    invoke: u32,
    /// Raw-text and error tool blocks render as visible text.
    visible_tool_blocks: Vec<u32>,
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
            visible_tool_blocks: ["<|content_invoke_tool_text|>", "<|content_tool_error|>"]
                .iter()
                .filter_map(|token| tokenizer.token_to_id(token))
                .collect(),
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
    Tool,
}

pub(super) struct State {
    mode: Mode,
    /// Author name sampled after `<|message_model|>`; a tool block's function.
    header: String,
    tool_call: String,
    tools: BTreeSet<String>,
}

impl State {
    pub fn new(tools: BTreeSet<String>) -> Self {
        Self {
            mode: Mode::Boundary,
            header: String::new(),
            tool_call: String::new(),
            tools,
        }
    }

    pub fn is_reasoning(&self) -> bool {
        self.mode == Mode::Thinking
    }

    /// Handles a real Inkling control-token ID; returns `None` for other IDs.
    pub fn control(
        &mut self,
        tokens: &SpecialTokens,
        id: u32,
    ) -> Result<Option<Vec<ChatOutputEvent>>> {
        let mut events = Vec::new();
        if id == tokens.model {
            events = self.close_block()?;
            self.mode = Mode::Header;
            self.header.clear();
        } else if id == tokens.thinking {
            self.mode = Mode::Thinking;
        } else if id == tokens.text || tokens.visible_tool_blocks.contains(&id) {
            self.mode = Mode::Text;
        } else if id == tokens.invoke {
            if self.tools.is_empty() {
                return Err(invalid("tool invocation without declared tools"));
            }
            self.mode = Mode::Tool;
            self.tool_call.clear();
        } else if id == tokens.end_message || tokens.eos == Some(id) {
            events = self.close_block()?;
            self.mode = Mode::Boundary;
        } else {
            return Ok(None);
        }
        Ok(Some(events))
    }

    pub fn push_text(&mut self, text: &str) -> Vec<ChatOutputEvent> {
        match self.mode {
            Mode::Header => {
                self.header.push_str(text);
                Vec::new()
            }
            Mode::Tool => {
                self.tool_call.push_str(text);
                Vec::new()
            }
            Mode::Thinking if !text.is_empty() => {
                vec![ChatOutputEvent::Reasoning(text.to_string())]
            }
            Mode::Boundary | Mode::Text if !text.is_empty() => {
                self.mode = Mode::Text;
                vec![ChatOutputEvent::Text(text.to_string())]
            }
            _ => Vec::new(),
        }
    }

    /// Normal completion closes an open block, including a tool call.
    pub fn finish(&mut self) -> Result<Vec<ChatOutputEvent>> {
        let events = self.close_block()?;
        self.mode = Mode::Boundary;
        Ok(events)
    }

    /// Length truncation discards a partial tool call.
    pub fn finish_truncated(&mut self) -> Vec<ChatOutputEvent> {
        self.mode = Mode::Boundary;
        self.tool_call.clear();
        Vec::new()
    }

    fn close_block(&mut self) -> Result<Vec<ChatOutputEvent>> {
        if self.mode != Mode::Tool {
            return Ok(Vec::new());
        }
        self.mode = Mode::Boundary;
        let function = parse_tool_call(&std::mem::take(&mut self.tool_call), &self.header)?;
        if !self.tools.contains(&function.name) {
            return Err(invalid(format!("unknown function {:?}", function.name)));
        }
        Ok(vec![ChatOutputEvent::ToolCall(ChatToolCall {
            id: next_tool_call_id()?,
            function,
        })])
    }
}

/// Parses `{"name": ..., "args": {...}}`; the block author names the function
/// when the payload omits it.
fn parse_tool_call(payload: &str, author: &str) -> Result<ChatFunctionCall> {
    let value: Value = serde_json::from_str(payload.trim())
        .map_err(|error| invalid(format!("tool call is not JSON: {error}")))?;
    let Value::Object(mut object) = value else {
        return Err(invalid("tool call is not a JSON object"));
    };
    let name = match object.remove("name") {
        Some(Value::String(name)) => name,
        Some(_) => return Err(invalid("tool call name is not a string")),
        None if !author.trim().is_empty() => author.trim().to_string(),
        None => return Err(invalid("tool call has no function name")),
    };
    let arguments = ["args", "arguments", "parameters"]
        .iter()
        .find_map(|key| object.remove(*key))
        .unwrap_or_else(|| Value::Object(Map::new()));
    let arguments = match arguments {
        Value::String(text) => serde_json::from_str(&text)
            .map_err(|error| invalid(format!("tool call arguments are not JSON: {error}")))?,
        value => value,
    };
    let Value::Object(arguments) = arguments else {
        return Err(invalid("tool call arguments are not an object"));
    };
    Ok(ChatFunctionCall {
        name,
        arguments: arguments.into_iter().collect(),
    })
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Inkling 2 output",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
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

    fn tools() -> BTreeSet<String> {
        ["get_weather".to_string(), "read".to_string()].into()
    }

    fn control(state: &mut State, tokens: &SpecialTokens, id: u32) -> Vec<ChatOutputEvent> {
        state.control(tokens, id).unwrap().expect("control token")
    }

    #[test]
    fn header_thinking_and_text_control_real_token_ids() {
        let tokens = tokens();
        let mut state = State::new(BTreeSet::new());
        assert_eq!(
            state.push_text("before"),
            vec![ChatOutputEvent::Text("before".into())]
        );
        control(&mut state, &tokens, tokens.model);
        assert!(state.push_text("recipient").is_empty());
        control(&mut state, &tokens, tokens.thinking);
        assert!(state.is_reasoning());
        assert_eq!(
            state.push_text("reason"),
            vec![ChatOutputEvent::Reasoning("reason".into())]
        );
        control(&mut state, &tokens, tokens.end_message);
        control(&mut state, &tokens, tokens.model);
        control(&mut state, &tokens, tokens.text);
        assert!(!state.is_reasoning());
        assert_eq!(
            state.push_text("answer"),
            vec![ChatOutputEvent::Text("answer".into())]
        );
        control(&mut state, &tokens, tokens.end_message);
        assert!(state.finish().unwrap().is_empty());
        assert!(state.control(&tokens, 12345).unwrap().is_none());
    }

    #[test]
    fn ordinary_marker_text_is_not_protocol() {
        let mut state = State::new(BTreeSet::new());
        assert_eq!(
            state.push_text("<|message_model|> is literal text"),
            vec![ChatOutputEvent::Text(
                "<|message_model|> is literal text".into()
            )]
        );
    }

    #[test]
    fn parallel_json_tool_blocks_and_trailing_text_are_parsed() {
        let tokens = tokens();
        let mut state = State::new(tools());
        let mut events = Vec::new();
        for (name, payload) in [
            (
                "get_weather",
                r#"{"name":"get_weather","args":{"city":"SF","days":2}}"#,
            ),
            ("read", r#"{"args":{"path":"a.txt"}}"#),
        ] {
            control(&mut state, &tokens, tokens.model);
            state.push_text(name);
            control(&mut state, &tokens, tokens.invoke);
            assert!(!state.is_reasoning());
            // Split the payload across decoded pieces.
            let (left, right) = payload.split_at(payload.len() / 2);
            assert!(state.push_text(left).is_empty());
            assert!(state.push_text(right).is_empty());
            events.extend(control(&mut state, &tokens, tokens.end_message));
        }
        control(&mut state, &tokens, tokens.model);
        control(&mut state, &tokens, tokens.text);
        events.extend(state.push_text("done"));
        let calls = events
            .iter()
            .filter_map(|event| match event {
                ChatOutputEvent::ToolCall(call) => Some(&call.function),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], json!("SF"));
        assert_eq!(calls[0].arguments["days"], json!(2));
        assert_eq!(calls[1].name, "read");
        assert_eq!(calls[1].arguments["path"], json!("a.txt"));
        assert_eq!(events.last(), Some(&ChatOutputEvent::Text("done".into())));
    }

    #[test]
    fn end_of_sampling_closes_a_tool_block_and_truncation_drops_it() {
        let tokens = tokens();
        let mut state = State::new(tools());
        control(&mut state, &tokens, tokens.model);
        control(&mut state, &tokens, tokens.invoke);
        state.push_text(r#"{"name":"read","args":{}}"#);
        let events = control(&mut state, &tokens, tokens.eos.unwrap());
        assert!(
            matches!(&events[..], [ChatOutputEvent::ToolCall(call)] if call.function.name == "read")
        );

        let mut truncated = State::new(tools());
        control(&mut truncated, &tokens, tokens.model);
        control(&mut truncated, &tokens, tokens.invoke);
        truncated.push_text(r#"{"name":"read","ar"#);
        assert!(truncated.finish_truncated().is_empty());
    }

    #[test]
    fn undeclared_or_unknown_tools_are_rejected() {
        let tokens = tokens();
        let mut state = State::new(BTreeSet::new());
        control(&mut state, &tokens, tokens.model);
        assert!(state.control(&tokens, tokens.invoke).is_err());

        let mut state = State::new(tools());
        control(&mut state, &tokens, tokens.model);
        control(&mut state, &tokens, tokens.invoke);
        state.push_text(r#"{"name":"delete","args":{}}"#);
        assert!(state.control(&tokens, tokens.end_message).is_err());
    }
}
