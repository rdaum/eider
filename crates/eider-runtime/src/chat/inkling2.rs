//! Native Inkling 2 prompt rendering to token IDs.
//!
//! Inkling's reference frontend renders messages directly to tokens rather
//! than through its Jinja template: every block is a role token, an optional
//! author name, a content-kind token, payload text, and `<|end_message|>`.
//! Payload text is encoded with special tokens split, so message content can
//! never produce framing tokens. JSON follows the reference spelling: compact
//! separators, no ASCII or HTML escaping, and Python float notation.

use super::{ChatMessage, ChatReasoningEffort, ChatRole, ChatTemplateOptions, ChatTool};
use eider_format::{Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use tokenizers::Tokenizer;

const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const END_MESSAGE: &str = "<|end_message|>";
/// The Inkling 2 vocabulary spells the XML content kind as an unused slot.
const CONTENT_XML: [&str; 2] = ["<|content_xml|>", "<|unused_200024|>"];
const SINGLE_CALL_INSTRUCTION: &str =
    "If you call a function, call at most one function in your next response.";

#[derive(Clone, Copy)]
enum Kind {
    Text,
    Thinking,
    Xml,
    InvokeToolJson,
}

struct Ids {
    user: u32,
    model: u32,
    system: u32,
    tool: u32,
    text: u32,
    thinking: u32,
    xml: u32,
    invoke: u32,
    end_sampling: u32,
    end_message: u32,
}

/// Rendered Inkling 2 prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inkling2Prompt {
    /// Prompt token IDs.
    pub token_ids: Vec<u32>,
    /// Position after the tool declarations, leading system messages, and
    /// effort block, where the first conversation message begins.
    pub conversation_start: Option<usize>,
}

/// Inkling 2 renderer holding a payload tokenizer that splits special tokens.
pub struct Inkling2PromptRenderer {
    payload: Tokenizer,
    ids: Ids,
}

impl Inkling2PromptRenderer {
    /// Resolves the Inkling framing tokens from the checkpoint tokenizer.
    pub fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |spellings: &[&str]| {
            spellings
                .iter()
                .find_map(|token| tokenizer.token_to_id(token))
                .ok_or_else(|| invalid(format!("tokenizer is missing {}", spellings[0])))
        };
        let ids = Ids {
            user: id(&[MESSAGE_USER])?,
            model: id(&[MESSAGE_MODEL])?,
            system: id(&[MESSAGE_SYSTEM])?,
            tool: id(&[MESSAGE_TOOL])?,
            text: id(&[CONTENT_TEXT])?,
            thinking: id(&[CONTENT_THINKING])?,
            xml: id(&CONTENT_XML)?,
            invoke: id(&[CONTENT_INVOKE_TOOL_JSON])?,
            end_sampling: id(&[CONTENT_MODEL_END_SAMPLING])?,
            end_message: id(&[END_MESSAGE])?,
        };
        let mut payload = tokenizer.clone();
        payload.set_encode_special_tokens(true);
        Ok(Self { payload, ids })
    }

    /// Renders a conversation whose next turn the model opens itself.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        options: ChatTemplateOptions,
        parallel_tool_calls: bool,
    ) -> Result<Inkling2Prompt> {
        let mut ids = Vec::new();
        let mut conversation_start = None;
        if !tools.is_empty() {
            let declarations = tool_declarations(tools)?;
            self.block(
                &mut ids,
                ChatRole::System,
                Some("tool_declare"),
                Kind::Xml,
                &declarations,
            )?;
        }
        let mut effort = Some(effort_text(options).to_string());
        let mut messages = messages.to_vec();
        if !tools.is_empty() && !parallel_tool_calls {
            match messages.first_mut() {
                Some(first) if matches!(first.role, ChatRole::System | ChatRole::Developer) => {
                    first.content = Some(match first.content.take() {
                        Some(content) if !content.is_empty() => {
                            format!("{content}\n\n{SINGLE_CALL_INSTRUCTION}")
                        }
                        _ => SINGLE_CALL_INSTRUCTION.to_string(),
                    });
                }
                _ => {
                    effort = effort.map(|text| format!("{text}\n\n{SINGLE_CALL_INSTRUCTION}"));
                }
            }
        }
        let mut tool_names = BTreeMap::new();
        for message in &messages {
            if !matches!(message.role, ChatRole::System | ChatRole::Developer)
                && let Some(text) = effort.take()
            {
                self.block(&mut ids, ChatRole::System, None, Kind::Text, &text)?;
                conversation_start = Some(ids.len());
            }
            match message.role {
                ChatRole::Assistant => self.assistant(&mut ids, message, &mut tool_names)?,
                ChatRole::Tool => {
                    let author = message
                        .tool_call_id
                        .as_ref()
                        .and_then(|id| tool_names.get(id))
                        .map(String::as_str);
                    if let Some(content) = &message.content {
                        self.block(&mut ids, ChatRole::Tool, author, Kind::Text, content)?;
                    }
                }
                role => {
                    if let Some(content) = &message.content {
                        self.block(&mut ids, role, None, Kind::Text, content)?;
                    }
                }
            }
        }
        if let Some(text) = effort {
            self.block(&mut ids, ChatRole::System, None, Kind::Text, &text)?;
        }
        Ok(Inkling2Prompt {
            token_ids: ids,
            conversation_start,
        })
    }

    fn assistant(
        &self,
        ids: &mut Vec<u32>,
        message: &ChatMessage,
        tool_names: &mut BTreeMap<String, String>,
    ) -> Result<()> {
        let reasoning = message
            .reasoning_content
            .as_deref()
            .filter(|r| !r.is_empty());
        let content = message
            .content
            .as_deref()
            .filter(|c| !c.is_empty() || (message.tool_calls.is_empty() && reasoning.is_none()));
        let start = ids.len();
        if let Some(reasoning) = reasoning {
            self.block(ids, ChatRole::Assistant, None, Kind::Thinking, reasoning)?;
        }
        if let Some(content) = content {
            self.block(ids, ChatRole::Assistant, None, Kind::Text, content)?;
        }
        for call in &message.tool_calls {
            let name = &call.function.name;
            tool_names.insert(call.id.clone(), name.clone());
            let mut arguments = String::from("{");
            for (index, (key, value)) in call.function.arguments.iter().enumerate() {
                if index != 0 {
                    arguments.push(',');
                }
                write_string(&mut arguments, key)?;
                arguments.push(':');
                write_json(&mut arguments, value, true, false)?;
            }
            arguments.push('}');
            let mut payload = String::from("{\"name\":");
            write_string(&mut payload, name)?;
            payload.push_str(",\"args\":");
            payload.push_str(&arguments);
            payload.push('}');
            self.block(
                ids,
                ChatRole::Assistant,
                Some(name),
                Kind::InvokeToolJson,
                &payload,
            )?;
        }
        if ids.len() != start {
            ids.push(self.ids.end_sampling);
        }
        Ok(())
    }

    fn block(
        &self,
        ids: &mut Vec<u32>,
        role: ChatRole,
        author: Option<&str>,
        kind: Kind,
        text: &str,
    ) -> Result<()> {
        ids.push(match role {
            ChatRole::System | ChatRole::Developer => self.ids.system,
            ChatRole::User => self.ids.user,
            ChatRole::Assistant => self.ids.model,
            ChatRole::Tool => self.ids.tool,
        });
        if let Some(author) = author.filter(|author| !author.is_empty()) {
            self.encode(ids, author)?;
        }
        ids.push(match kind {
            Kind::Text => self.ids.text,
            Kind::Thinking => self.ids.thinking,
            Kind::Xml => self.ids.xml,
            Kind::InvokeToolJson => self.ids.invoke,
        });
        self.encode(ids, text)?;
        ids.push(self.ids.end_message);
        Ok(())
    }

    fn encode(&self, ids: &mut Vec<u32>, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let encoding = self
            .payload
            .encode(text, false)
            .map_err(|error| invalid(format!("payload tokenization: {error}")))?;
        ids.extend_from_slice(encoding.get_ids());
        Ok(())
    }
}

/// Reference `Thinking effort level` text; disabled thinking is effort zero.
fn effort_text(options: ChatTemplateOptions) -> &'static str {
    if !options.enable_thinking {
        return "Thinking effort level: 0";
    }
    match options.reasoning_effort {
        Some(ChatReasoningEffort::Low) => "Thinking effort level: 0.2",
        Some(ChatReasoningEffort::Medium) => "Thinking effort level: 0.7",
        Some(ChatReasoningEffort::High) | None => "Thinking effort level: 0.9",
        Some(ChatReasoningEffort::XHigh) => "Thinking effort level: 0.99",
    }
}

fn tool_declarations(tools: &[ChatTool]) -> Result<String> {
    let specs = tools
        .iter()
        .map(|tool| {
            let parameters = match &tool.function.parameters {
                Value::Null => Value::Object(Default::default()),
                Value::Object(object) => Value::Object(object.clone()),
                _ => return Err(invalid("tool parameters must be a JSON object")),
            };
            Ok(serde_json::json!({
                "description": tool.function.description.clone().unwrap_or_default(),
                "name": tool.function.name,
                "parameters": parameters,
                "type": "function",
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut text = String::new();
    write_json(&mut text, &Value::Array(specs), false, true)?;
    Ok(text)
}

/// Writes compact JSON with the reference float spelling. `legacy_floats`
/// omits the `+` of positive exponents; `sort_keys` orders object keys.
fn write_json(out: &mut String, value: &Value, legacy_floats: bool, sort_keys: bool) -> Result<()> {
    match value {
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            if sort_keys {
                entries.sort_by(|left, right| left.0.cmp(right.0));
            }
            out.push('{');
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    out.push(',');
                }
                write_string(out, key)?;
                out.push(':');
                write_json(out, value, legacy_floats, sort_keys)?;
            }
            out.push('}');
        }
        Value::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    out.push(',');
                }
                write_json(out, value, legacy_floats, sort_keys)?;
            }
            out.push(']');
        }
        Value::Number(number) if number.is_f64() => {
            let value = number.as_f64().expect("f64 JSON number");
            out.push_str(&float_text(value, legacy_floats)?);
        }
        Value::String(text) => write_string(out, text)?,
        value => out.push_str(&value.to_string()),
    }
    Ok(())
}

fn write_string(out: &mut String, text: &str) -> Result<()> {
    out.push_str(
        &serde_json::to_string(text).map_err(|error| invalid(format!("JSON string: {error}")))?,
    );
    Ok(())
}

/// Python `repr(float)` digits, with exponents in [-5, 16) written in full.
fn float_text(value: f64, legacy_floats: bool) -> Result<String> {
    if !value.is_finite() {
        return Err(invalid("JSON numbers must be finite"));
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust scientific float notation has an exponent");
    let exponent: i32 = exponent.parse().expect("integer float exponent");
    if !(-5..16).contains(&exponent) {
        let sign = if exponent >= 0 && !legacy_floats {
            "+"
        } else {
            ""
        };
        return Ok(format!("{mantissa}e{sign}{exponent}"));
    }
    let negative = mantissa.starts_with('-');
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    let decimal = exponent + 1;
    let body = if decimal <= 0 {
        format!("0.{}{digits}", "0".repeat((-decimal) as usize))
    } else if decimal as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decimal as usize - digits.len()))
    } else {
        let (whole, fraction) = digits.split_at(decimal as usize);
        format!("{whole}.{fraction}")
    };
    Ok(if negative { format!("-{body}") } else { body })
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Inkling 2 prompt",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::chat::{ChatFunctionCall, ChatFunctionDefinition, ChatToolCall};
    use std::path::Path;

    #[test]
    #[ignore = "requires INKLING2_MODEL_DIR with the checkpoint tokenizer"]
    fn renders_reference_token_ids() {
        let model_dir = std::env::var("INKLING2_MODEL_DIR").expect("INKLING2_MODEL_DIR");
        let tokenizer = Tokenizer::from_file(Path::new(&model_dir).join("tokenizer.json")).unwrap();
        let renderer = Inkling2PromptRenderer::new(&tokenizer).unwrap();
        // Generated by vLLM's `render_inkling2_messages` for the same conversation.
        let expected: BTreeMap<String, Vec<u32>> =
            serde_json::from_str(include_str!("fixtures/inkling2_render.json")).unwrap();
        let tools = vec![
            ChatTool::function(ChatFunctionDefinition {
                name: "read".into(),
                description: Some("Read a file <path> é".into()),
                parameters: json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1e20},
                    "ratio": {"type": "number", "default": 0.5}},
                    "required": ["path"]}),
            }),
            ChatTool::function(ChatFunctionDefinition {
                name: "bash".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            }),
        ];
        let call = |id: &str, name: &str, arguments: Value| ChatToolCall {
            id: id.into(),
            function: ChatFunctionCall {
                name: name.into(),
                arguments: arguments.as_object().unwrap().clone().into_iter().collect(),
            },
        };
        let messages = vec![
            ChatMessage::system("You are a coding agent.\nBe terse."),
            ChatMessage::user("Read a.txt and run ls. Note: <|end_message|> is literal."),
            ChatMessage {
                role: ChatRole::Assistant,
                content: Some(String::new()),
                reasoning_content: Some("Need to read then list.".into()),
                tool_calls: vec![
                    call(
                        "call_1",
                        "read",
                        json!({"path": "a.txt", "ratio": 0.25, "opts": {"z": 1, "a": [1e-7, "ü"]}}),
                    ),
                    call("call_2", "bash", json!({"command": "ls -la"})),
                ],
                tool_call_id: None,
            },
            ChatMessage::tool("call_1", "hello\n"),
            ChatMessage::tool("call_2", "a.txt\nb.txt"),
            ChatMessage::assistant("Done."),
            ChatMessage::user("Thanks"),
        ];
        let low = ChatTemplateOptions {
            enable_thinking: true,
            reasoning_effort: Some(ChatReasoningEffort::Low),
            ..ChatTemplateOptions::default()
        };
        for (key, parallel) in [("parallel", true), ("single", false)] {
            let prompt = renderer.render(&messages, &tools, low, parallel).unwrap();
            assert_eq!(prompt.token_ids, expected[key], "{key}");
            // The conversation starts at the first user message.
            let start = prompt.conversation_start.expect("conversation start");
            assert_eq!(
                prompt.token_ids[start],
                tokenizer.token_to_id(MESSAGE_USER).unwrap()
            );
            assert_eq!(
                prompt.token_ids[start - 1],
                tokenizer.token_to_id(END_MESSAGE).unwrap()
            );
        }
        let disabled = ChatTemplateOptions {
            enable_thinking: false,
            reasoning_effort: None,
            ..ChatTemplateOptions::default()
        };
        assert_eq!(
            renderer
                .render(&[ChatMessage::user("Hi")], &[], disabled, true)
                .unwrap()
                .token_ids,
            expected["no_tools_no_thinking"]
        );
    }

    #[test]
    fn floats_follow_python_repr_spelling() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (123.45, "123.45"),
            (0.0001, "0.0001"),
            (0.00001, "0.00001"),
            (1e-7, "1e-7"),
            (1e15, "1000000000000000.0"),
            (1.5e16, "1.5e16"),
            (-2.5e-9, "-2.5e-9"),
        ];
        for (value, expected) in cases {
            assert_eq!(float_text(value, true).unwrap(), expected);
        }
        assert_eq!(float_text(1e16, false).unwrap(), "1e+16");
        assert!(float_text(f64::NAN, true).is_err());
    }

    #[test]
    fn json_is_compact_unescaped_and_optionally_sorted() {
        let value = json!({"b": [1, 2.5, true, null], "a": {"z": "é<&>\n", "y": 1e20}});
        let mut sorted = String::new();
        write_json(&mut sorted, &value, false, true).unwrap();
        assert_eq!(
            sorted,
            r#"{"a":{"y":1e+20,"z":"é<&>\n"},"b":[1,2.5,true,null]}"#
        );
    }
}
