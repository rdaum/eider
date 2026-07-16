//! Incremental Qwen chat output decoding.

use super::chat::{ChatFunctionCall, ChatTool, ChatToolCall};
use nvfp4::{Error, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{DecodeStream, Tokenizer};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const THINK_CLOSE: &str = "</think>";

static NEXT_TOOL_CALL_ID: AtomicU64 = AtomicU64::new(1);

type TokenizerDecodeStream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// One structured event recovered from the generated token stream.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatOutputEvent {
    /// Model reasoning emitted before the checkpoint's closing thinking tag.
    Reasoning(String),
    /// Ordinary assistant content safe to forward to a client.
    Text(String),
    /// One complete function call with a server-generated identity.
    ToolCall(ChatToolCall),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputMode {
    Reasoning,
    Text,
    ToolCall,
}

/// Request-scoped tokenizer and Qwen tool-protocol decoder.
pub struct ChatOutputCodec<'tokenizer> {
    decode_stream: TokenizerDecodeStream<'tokenizer>,
    parser: ChatOutputParser,
    finished: bool,
}

impl<'tokenizer> ChatOutputCodec<'tokenizer> {
    /// Creates an output codec for one scheduled request.
    ///
    /// `starts_in_reasoning` must match the generation prefix rendered by the
    /// active checkpoint template.
    pub fn new(
        tokenizer: &'tokenizer Tokenizer,
        tools: &[ChatTool],
        starts_in_reasoning: bool,
    ) -> Result<Self> {
        Ok(Self {
            decode_stream: tokenizer.decode_stream(true),
            parser: ChatOutputParser::new(tools, starts_in_reasoning)?,
            finished: false,
        })
    }

    /// Decodes one generated vocabulary ID into zero or more structured events.
    pub fn push_token(&mut self, token_id: u32) -> Result<Vec<ChatOutputEvent>> {
        if self.finished {
            return Err(Error::Format {
                label: "chat output stream",
                detail: "cannot push a token after finish".to_string(),
            });
        }
        let Some(text) = self
            .decode_stream
            .step(token_id)
            .map_err(|error| Error::Format {
                label: "tokenizer decode stream",
                detail: error.to_string(),
            })?
        else {
            return Ok(Vec::new());
        };
        self.parser.push_text(&text)
    }

    /// Finishes protocol parsing and flushes safe pending text.
    pub fn finish(&mut self) -> Result<Vec<ChatOutputEvent>> {
        self.finish_with_truncation(false)
    }

    /// Finishes a length-truncated stream, discarding any partial tool call.
    pub fn finish_truncated(&mut self) -> Result<Vec<ChatOutputEvent>> {
        self.finish_with_truncation(true)
    }

    fn finish_with_truncation(&mut self, truncated: bool) -> Result<Vec<ChatOutputEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if truncated {
            self.parser.finish_truncated()
        } else {
            self.parser.finish()
        }
    }
}

struct ChatOutputParser {
    mode: OutputMode,
    pending: String,
    tool_call: String,
    trim_after_thinking: bool,
    trim_after_tool_call: bool,
    string_arguments: BTreeMap<String, BTreeSet<String>>,
    finished: bool,
}

impl ChatOutputParser {
    fn new(tools: &[ChatTool], starts_in_reasoning: bool) -> Result<Self> {
        let mut string_arguments = BTreeMap::new();
        for tool in tools {
            let name = tool.function.name.clone();
            let string_parameters = tool.function.parameters["properties"]
                .as_object()
                .map(|properties| {
                    properties
                        .iter()
                        .filter(|(_, schema)| schema["type"].as_str() == Some("string"))
                        .map(|(name, _)| name.clone())
                        .collect()
                })
                .unwrap_or_default();
            if string_arguments
                .insert(name.clone(), string_parameters)
                .is_some()
            {
                return Err(Error::Format {
                    label: "chat output tools",
                    detail: format!("duplicate function definition {name:?}"),
                });
            }
        }
        Ok(Self {
            mode: if starts_in_reasoning {
                OutputMode::Reasoning
            } else {
                OutputMode::Text
            },
            pending: String::new(),
            tool_call: String::new(),
            trim_after_thinking: false,
            trim_after_tool_call: false,
            string_arguments,
            finished: false,
        })
    }

    fn push_text(&mut self, text: &str) -> Result<Vec<ChatOutputEvent>> {
        if self.finished {
            return Err(Error::Format {
                label: "chat output parser",
                detail: "cannot push text after finish".to_string(),
            });
        }
        self.pending.push_str(text);
        let mut events = Vec::new();
        loop {
            let progressed = match self.mode {
                OutputMode::Reasoning => self.parse_reasoning(&mut events),
                OutputMode::Text => self.parse_text(&mut events),
                OutputMode::ToolCall => self.parse_tool_call(&mut events)?,
            };
            if !progressed {
                break;
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<ChatOutputEvent>> {
        self.finish_with_truncation(false)
    }

    fn finish_truncated(&mut self) -> Result<Vec<ChatOutputEvent>> {
        self.finish_with_truncation(true)
    }

    fn finish_with_truncation(&mut self, truncated: bool) -> Result<Vec<ChatOutputEvent>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        match self.mode {
            OutputMode::Reasoning => Ok(take_event(&mut self.pending, ChatOutputEvent::Reasoning)),
            OutputMode::Text => Ok(take_event(&mut self.pending, ChatOutputEvent::Text)),
            OutputMode::ToolCall if truncated => Ok(Vec::new()),
            OutputMode::ToolCall => Err(Error::Format {
                label: "chat tool call",
                detail: "generation ended inside an unterminated <tool_call>".to_string(),
            }),
        }
    }

    fn parse_reasoning(&mut self, events: &mut Vec<ChatOutputEvent>) -> bool {
        if let Some(index) = self.pending.find(THINK_CLOSE) {
            push_nonempty(
                events,
                ChatOutputEvent::Reasoning(self.pending[..index].to_string()),
            );
            self.pending.drain(..index + THINK_CLOSE.len());
            self.mode = OutputMode::Text;
            self.trim_after_thinking = true;
            return true;
        }
        flush_safe_prefix(
            &mut self.pending,
            THINK_CLOSE,
            events,
            ChatOutputEvent::Reasoning,
        )
    }

    fn parse_text(&mut self, events: &mut Vec<ChatOutputEvent>) -> bool {
        if self.trim_after_thinking || self.trim_after_tool_call {
            let trimmed = self.pending.trim_start_matches(['\r', '\n']).len();
            if trimmed != self.pending.len() {
                let removed = self.pending.len() - trimmed;
                self.pending.drain(..removed);
            }
            if self.pending.is_empty() {
                return false;
            }
            self.trim_after_thinking = false;
            self.trim_after_tool_call = false;
        }
        if let Some(index) = self.pending.find(TOOL_CALL_OPEN) {
            push_nonempty(
                events,
                ChatOutputEvent::Text(self.pending[..index].to_string()),
            );
            self.pending.drain(..index + TOOL_CALL_OPEN.len());
            self.mode = OutputMode::ToolCall;
            return true;
        }
        flush_safe_prefix(
            &mut self.pending,
            TOOL_CALL_OPEN,
            events,
            ChatOutputEvent::Text,
        )
    }

    fn parse_tool_call(&mut self, events: &mut Vec<ChatOutputEvent>) -> Result<bool> {
        self.tool_call.push_str(&self.pending);
        self.pending.clear();
        let Some(index) = self.tool_call.find(TOOL_CALL_CLOSE) else {
            return Ok(false);
        };
        let body = self.tool_call[..index].to_string();
        let remainder = self.tool_call[index + TOOL_CALL_CLOSE.len()..].to_string();
        self.tool_call.clear();
        self.pending = remainder;
        let function = parse_function_call(&body, &self.string_arguments)?;
        let id = next_tool_call_id()?;
        events.push(ChatOutputEvent::ToolCall(ChatToolCall { id, function }));
        self.mode = OutputMode::Text;
        self.trim_after_tool_call = true;
        Ok(true)
    }
}

fn flush_safe_prefix(
    pending: &mut String,
    marker: &str,
    events: &mut Vec<ChatOutputEvent>,
    make_event: fn(String) -> ChatOutputEvent,
) -> bool {
    let held = longest_marker_prefix_suffix(pending, marker);
    let emit_bytes = pending.len() - held;
    if emit_bytes == 0 {
        return false;
    }
    let emitted = pending[..emit_bytes].to_string();
    pending.drain(..emit_bytes);
    push_nonempty(events, make_event(emitted));
    true
}

fn longest_marker_prefix_suffix(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&length| text.ends_with(&marker[..length]))
        .unwrap_or(0)
}

fn take_event(
    pending: &mut String,
    make_event: fn(String) -> ChatOutputEvent,
) -> Vec<ChatOutputEvent> {
    if pending.is_empty() {
        Vec::new()
    } else {
        vec![make_event(std::mem::take(pending))]
    }
}

fn push_nonempty(events: &mut Vec<ChatOutputEvent>, event: ChatOutputEvent) {
    let empty = match &event {
        ChatOutputEvent::Reasoning(text) | ChatOutputEvent::Text(text) => text.is_empty(),
        ChatOutputEvent::ToolCall(_) => false,
    };
    if !empty {
        events.push(event);
    }
}

fn parse_function_call(
    body: &str,
    string_arguments: &BTreeMap<String, BTreeSet<String>>,
) -> Result<ChatFunctionCall> {
    let body = body.trim_matches(['\r', '\n']);
    let function = body
        .strip_prefix("<function=")
        .ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: "expected <function=...> immediately inside <tool_call>".to_string(),
        })?;
    let open_end = function.find('>').ok_or_else(|| Error::Format {
        label: "chat tool call",
        detail: "unterminated <function=...> tag".to_string(),
    })?;
    let name = &function[..open_end];
    validate_protocol_name("function", name)?;
    let function_body = &function[open_end + 1..];
    let close_start = function_body
        .rfind("</function>")
        .ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("missing </function> for {name:?}"),
        })?;
    if !function_body[close_start + "</function>".len()..]
        .trim()
        .is_empty()
    {
        return Err(Error::Format {
            label: "chat tool call",
            detail: "unexpected content after </function>".to_string(),
        });
    }

    let mut arguments = BTreeMap::new();
    let mut remaining = &function_body[..close_start];
    loop {
        remaining = remaining.trim_start_matches(['\r', '\n']);
        if remaining.is_empty() {
            break;
        }
        let parameter = remaining
            .strip_prefix("<parameter=")
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("unexpected content in function {name:?}"),
            })?;
        let parameter_open_end = parameter.find('>').ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: "unterminated <parameter=...> tag".to_string(),
        })?;
        let parameter_name = &parameter[..parameter_open_end];
        validate_protocol_name("parameter", parameter_name)?;
        let parameter_body = &parameter[parameter_open_end + 1..];
        let parameter_close = parameter_body
            .find("</parameter>")
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing </parameter> for {parameter_name:?}"),
            })?;
        let raw_value = strip_protocol_newlines(&parameter_body[..parameter_close]);
        let value = decode_argument(name, parameter_name, raw_value, string_arguments);
        if arguments
            .insert(parameter_name.to_string(), value)
            .is_some()
        {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate parameter {parameter_name:?}"),
            });
        }
        remaining = &parameter_body[parameter_close + "</parameter>".len()..];
    }

    Ok(ChatFunctionCall {
        name: name.to_string(),
        arguments,
    })
}

fn strip_protocol_newlines(mut value: &str) -> &str {
    value = value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value);
    value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(value)
}

fn decode_argument(
    function: &str,
    parameter: &str,
    raw_value: &str,
    string_arguments: &BTreeMap<String, BTreeSet<String>>,
) -> Value {
    if string_arguments
        .get(function)
        .is_some_and(|parameters| parameters.contains(parameter))
    {
        return Value::String(raw_value.to_string());
    }
    serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_string()))
}

fn validate_protocol_name(kind: &str, name: &str) -> Result<()> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Ok(());
    }
    Err(Error::Format {
        label: "chat tool call",
        detail: format!("invalid {kind} name {name:?}"),
    })
}

fn next_tool_call_id() -> Result<String> {
    let value = NEXT_TOOL_CALL_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::Format {
            label: "chat tool call ID",
            detail: "tool call ID space exhausted".to_string(),
        })?;
    Ok(format!("call_{value:016x}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::{ChatFunctionDefinition, ChatTool};
    use serde_json::json;
    use std::path::Path;

    fn tools() -> Vec<ChatTool> {
        vec![ChatTool::function(ChatFunctionDefinition {
            name: "write_file".to_string(),
            description: Some("Write a file".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                    "executable": {"type": "boolean"}
                }
            }),
        })]
    }

    fn protocol_text() -> &'static str {
        concat!(
            "I will update it.",
            "<tool_call>\n",
            "<function=write_file>\n",
            "<parameter=path>\nsrc/main.rs\n</parameter>\n",
            "<parameter=contents>\nfn main() {\n    println!(\"hi\");\n}\n</parameter>\n",
            "<parameter=executable>\nfalse\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        )
    }

    fn normalized(events: Vec<ChatOutputEvent>) -> Vec<ChatOutputEvent> {
        let mut normalized = Vec::new();
        for event in events {
            let event = match event {
                ChatOutputEvent::ToolCall(mut call) => {
                    call.id = "call_ID".to_string();
                    ChatOutputEvent::ToolCall(call)
                }
                event => event,
            };
            match (normalized.last_mut(), event) {
                (Some(ChatOutputEvent::Reasoning(previous)), ChatOutputEvent::Reasoning(next))
                | (Some(ChatOutputEvent::Text(previous)), ChatOutputEvent::Text(next)) => {
                    previous.push_str(&next);
                }
                (_, event) => normalized.push(event),
            }
        }
        normalized
    }

    fn expected() -> Vec<ChatOutputEvent> {
        vec![
            ChatOutputEvent::Text("I will update it.".to_string()),
            ChatOutputEvent::ToolCall(ChatToolCall {
                id: "call_ID".to_string(),
                function: ChatFunctionCall {
                    name: "write_file".to_string(),
                    arguments: BTreeMap::from([
                        (
                            "contents".to_string(),
                            json!("fn main() {\n    println!(\"hi\");\n}"),
                        ),
                        ("executable".to_string(), json!(false)),
                        ("path".to_string(), json!("src/main.rs")),
                    ]),
                },
            }),
        ]
    }

    fn parse_chunks(chunks: &[&str]) -> Vec<ChatOutputEvent> {
        let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push_text(chunk).unwrap());
        }
        events.extend(parser.finish().unwrap());
        normalized(events)
    }

    #[test]
    fn tool_protocol_survives_every_two_chunk_boundary() {
        let text = protocol_text();
        for split in 0..=text.len() {
            assert_eq!(parse_chunks(&[&text[..split], &text[split..]]), expected());
        }
    }

    #[test]
    fn tool_protocol_survives_single_byte_chunks() {
        let text = protocol_text();
        let chunks: Vec<_> = (0..text.len())
            .map(|index| &text[index..index + 1])
            .collect();
        assert_eq!(parse_chunks(&chunks), expected());
    }

    #[test]
    fn reasoning_and_text_tags_survive_split_boundaries() {
        let text = "checking details</think>\n\nThe result is ready.";
        for split in 0..=text.len() {
            let mut parser = ChatOutputParser::new(&[], true).unwrap();
            let mut events = parser.push_text(&text[..split]).unwrap();
            events.extend(parser.push_text(&text[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(
                normalized(events),
                [
                    ChatOutputEvent::Reasoning("checking details".to_string()),
                    ChatOutputEvent::Text("The result is ready.".to_string())
                ],
                "split {split}"
            );
        }
    }

    #[test]
    fn unfinished_tool_markup_is_an_error_and_never_text() {
        let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
        assert!(parser.push_text("hello <tool_call><function=x>").unwrap().iter().all(
            |event| !matches!(event, ChatOutputEvent::Text(text) if text.contains("tool_call"))
        ));
        assert!(parser.finish().is_err());
    }

    #[test]
    fn length_truncation_discards_unfinished_tool_markup() {
        let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
        let events = parser
            .push_text("hello <tool_call><function=write_file>")
            .unwrap();
        assert_eq!(events, [ChatOutputEvent::Text("hello ".to_string())]);
        assert!(parser.finish_truncated().unwrap().is_empty());
    }

    #[test]
    fn consecutive_tool_calls_do_not_emit_protocol_whitespace() {
        let call = &protocol_text()["I will update it.".len()..];
        let input = format!("{call}\n{call}");
        let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
        let mut events = parser.push_text(&input).unwrap();
        events.extend(parser.finish().unwrap());
        assert_eq!(events.len(), 2);
        assert!(
            events
                .iter()
                .all(|event| matches!(event, ChatOutputEvent::ToolCall(_)))
        );
        let ids: BTreeSet<_> = events
            .iter()
            .filter_map(|event| match event {
                ChatOutputEvent::ToolCall(call) => Some(call.id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    #[ignore = "requires the local Qwen3.6 checkpoint"]
    fn local_tokenizer_stream_recovers_reasoning_text_and_tool_call() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/qwen3.6-35b-a3-nvfp4");
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let generated = format!("checked</think>\n\n{}", protocol_text());
        let encoding = tokenizer.encode(generated, false).unwrap();
        let mut codec = ChatOutputCodec::new(&tokenizer, &tools(), true).unwrap();
        let mut events = Vec::new();
        for &token in encoding.get_ids() {
            events.extend(codec.push_token(token).unwrap());
        }
        events.extend(codec.finish().unwrap());

        let events = normalized(events);
        assert_eq!(events[0], ChatOutputEvent::Reasoning("checked".to_string()));
        assert_eq!(&events[1..], expected());
    }
}
