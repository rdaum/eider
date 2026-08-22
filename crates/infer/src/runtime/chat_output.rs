//! Incremental structured chat output decoding.

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
const GEMMA_TOOL_CALL_OPEN: &str = "<|tool_call>";
const GEMMA_TOOL_CALL_CLOSE: &str = "<tool_call|>";
const DSML_TOOL_CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
const DSML_TOOL_CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
const DSML_INVOKE_OPEN: &str = "<｜DSML｜invoke";
const DSML_INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const DSML_PARAMETER_OPEN: &str = "<｜DSML｜parameter";
const DSML_PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";
const ATEM_TOOL_CALLS_OPEN: &str = "<atem:function_calls>";
const ATEM_TOOL_CALLS_CLOSE: &str = "</atem:function_calls>";
const ATEM_INVOKE_OPEN: &str = "<atem:invoke";
const ATEM_INVOKE_CLOSE: &str = "</atem:invoke>";
const ATEM_PARAMETER_OPEN: &str = "<atem:parameter";
const ATEM_PARAMETER_CLOSE: &str = "</atem:parameter>";
const GEMMA_THINK_OPEN: &str = "<|channel>thought\n";
const GEMMA_THINK_CLOSE: &str = "<channel|>";
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
    DirectToolCall,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolCallProtocol {
    Standard,
    Gemma,
    Dsml,
    Atem,
}

impl ToolCallProtocol {
    fn close(self) -> &'static str {
        match self {
            Self::Standard => TOOL_CALL_CLOSE,
            Self::Gemma => GEMMA_TOOL_CALL_CLOSE,
            Self::Dsml => DSML_TOOL_CALLS_CLOSE,
            Self::Atem => ATEM_TOOL_CALLS_CLOSE,
        }
    }
}

/// Request-scoped tokenizer and model tool-protocol decoder.
pub struct ChatOutputCodec<'tokenizer> {
    decode_stream: TokenizerDecodeStream<'tokenizer>,
    parser: ChatOutputParser,
    gemma_special_tokens: Option<GemmaSpecialTokens>,
    muse_special_tokens: Option<MuseSpecialTokens>,
    muse_header: Option<String>,
    finished: bool,
}

struct GemmaSpecialTokens {
    channel_open: u32,
    channel_close: u32,
    tool_call_open: u32,
    tool_call_close: u32,
}

struct MuseSpecialTokens {
    start: u32,
    message: u32,
    end_message: u32,
    end_turn: u32,
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
        let gemma_special_tokens = match (
            tokenizer.token_to_id("<|channel>"),
            tokenizer.token_to_id(GEMMA_THINK_CLOSE),
            tokenizer.token_to_id(GEMMA_TOOL_CALL_OPEN),
            tokenizer.token_to_id(GEMMA_TOOL_CALL_CLOSE),
        ) {
            (
                Some(channel_open),
                Some(channel_close),
                Some(tool_call_open),
                Some(tool_call_close),
            ) => Some(GemmaSpecialTokens {
                channel_open,
                channel_close,
                tool_call_open,
                tool_call_close,
            }),
            _ => None,
        };
        let muse_special_tokens = match (
            tokenizer.token_to_id("<|start|>"),
            tokenizer.token_to_id("<|message|>"),
            tokenizer.token_to_id("<|eom|>"),
            tokenizer.token_to_id("<|eot|>"),
        ) {
            (Some(start), Some(message), Some(end_message), Some(end_turn)) => {
                Some(MuseSpecialTokens {
                    start,
                    message,
                    end_message,
                    end_turn,
                })
            }
            _ => None,
        };
        let muse_header = muse_special_tokens
            .as_ref()
            .map(|_| "assistant".to_string());
        Ok(Self {
            decode_stream: tokenizer.decode_stream(true),
            parser: ChatOutputParser::new(tools, starts_in_reasoning)?,
            gemma_special_tokens,
            muse_special_tokens,
            muse_header,
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
        if let Some(tokens) = &self.muse_special_tokens {
            if token_id == tokens.start {
                self.muse_header = Some(String::new());
                return Ok(Vec::new());
            }
            if token_id == tokens.message {
                let header = self.muse_header.take().ok_or_else(|| Error::Format {
                    label: "Muse Glimmer output",
                    detail: "message marker appeared outside a recipient header".to_string(),
                })?;
                self.parser.begin_muse_segment(&header)?;
                return Ok(Vec::new());
            }
            if token_id == tokens.end_message || token_id == tokens.end_turn {
                let events = self.parser.end_muse_segment()?;
                self.muse_header = Some(String::new());
                return Ok(events);
            }
        }
        if let Some(tokens) = &self.gemma_special_tokens {
            let marker = if token_id == tokens.channel_open {
                Some("<|channel>")
            } else if token_id == tokens.channel_close {
                Some(GEMMA_THINK_CLOSE)
            } else if token_id == tokens.tool_call_open {
                Some(GEMMA_TOOL_CALL_OPEN)
            } else if token_id == tokens.tool_call_close {
                Some(GEMMA_TOOL_CALL_CLOSE)
            } else {
                None
            };
            if let Some(marker) = marker {
                return self.parser.push_text(marker);
            }
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
        if let Some(header) = &mut self.muse_header {
            header.push_str(&text);
            return Ok(Vec::new());
        }
        self.parser.push_text(&text)
    }

    /// Returns whether the next generated token belongs to the thinking section.
    pub fn is_reasoning(&self) -> bool {
        self.parser.mode == OutputMode::Reasoning
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
    tool_call_protocol: ToolCallProtocol,
    trim_after_thinking: bool,
    trim_after_tool_call: bool,
    string_arguments: BTreeMap<String, BTreeSet<String>>,
    tool_parameters: BTreeMap<String, ToolParameters>,
    direct_tools: Vec<DirectTool>,
    active_direct_tool: Option<usize>,
    finished: bool,
}

struct ToolParameters {
    names: BTreeSet<String>,
    required: BTreeSet<String>,
}

struct DirectTool {
    name: String,
    parameter: String,
    open: String,
    close: String,
}

impl ChatOutputParser {
    fn new(tools: &[ChatTool], starts_in_reasoning: bool) -> Result<Self> {
        let mut string_arguments = BTreeMap::new();
        let mut tool_parameters = BTreeMap::new();
        let mut direct_tools = Vec::new();
        for tool in tools {
            let name = tool.function.name.clone();
            let properties = tool.function.parameters["properties"].as_object();
            let string_parameters: BTreeSet<String> = properties
                .map(|properties| {
                    properties
                        .iter()
                        .filter(|(_, schema)| schema["type"].as_str() == Some("string"))
                        .map(|(name, _)| name.clone())
                        .collect()
                })
                .unwrap_or_default();
            if string_arguments.contains_key(&name) {
                return Err(Error::Format {
                    label: "chat output tools",
                    detail: format!("duplicate function definition {name:?}"),
                });
            }
            let names = properties
                .map(|properties| properties.keys().cloned().collect())
                .unwrap_or_default();
            let required: BTreeSet<String> = tool.function.parameters["required"]
                .as_array()
                .map(|required| {
                    required
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            if required.len() == 1 {
                let parameter = required.iter().next().expect("one required parameter");
                if string_parameters.contains(parameter) {
                    direct_tools.push(DirectTool {
                        open: format!("<{name}>"),
                        close: format!("</{name}>"),
                        name: name.clone(),
                        parameter: parameter.clone(),
                    });
                }
            }
            string_arguments.insert(name.clone(), string_parameters);
            tool_parameters.insert(name, ToolParameters { names, required });
        }
        Ok(Self {
            mode: if starts_in_reasoning {
                OutputMode::Reasoning
            } else {
                OutputMode::Text
            },
            pending: String::new(),
            tool_call: String::new(),
            tool_call_protocol: ToolCallProtocol::Standard,
            trim_after_thinking: false,
            trim_after_tool_call: false,
            string_arguments,
            tool_parameters,
            direct_tools,
            active_direct_tool: None,
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
                OutputMode::DirectToolCall => self.parse_direct_tool_call(&mut events)?,
            };
            if !progressed {
                break;
            }
        }
        Ok(events)
    }

    fn begin_muse_segment(&mut self, header: &str) -> Result<()> {
        let header = header.trim();
        let recipient = header
            .strip_prefix("assistant")
            .map(str::trim)
            .and_then(|tail| tail.strip_prefix("to="))
            .map(str::trim)
            .ok_or_else(|| Error::Format {
                label: "Muse Glimmer output",
                detail: format!("invalid recipient header {header:?}"),
            })?;
        if recipient.is_empty() {
            return Err(Error::Format {
                label: "Muse Glimmer output",
                detail: "recipient header has an empty recipient".to_string(),
            });
        }
        self.mode = if recipient == "self" {
            OutputMode::Reasoning
        } else {
            OutputMode::Text
        };
        self.trim_after_thinking = false;
        self.trim_after_tool_call = false;
        Ok(())
    }

    fn end_muse_segment(&mut self) -> Result<Vec<ChatOutputEvent>> {
        let events = match self.mode {
            OutputMode::Reasoning => take_event(&mut self.pending, ChatOutputEvent::Reasoning),
            OutputMode::Text => take_event(&mut self.pending, ChatOutputEvent::Text),
            OutputMode::ToolCall => {
                return Err(Error::Format {
                    label: "Muse Glimmer tool call",
                    detail: format!("message ended before {}", self.tool_call_protocol.close()),
                });
            }
            OutputMode::DirectToolCall => {
                return Err(Error::Format {
                    label: "Muse Glimmer tool call",
                    detail: "message ended inside a direct tool call".to_string(),
                });
            }
        };
        self.mode = OutputMode::Text;
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
                detail: format!(
                    "generation ended before {}",
                    self.tool_call_protocol.close()
                ),
            }),
            OutputMode::DirectToolCall if truncated => Ok(Vec::new()),
            OutputMode::DirectToolCall => Err(Error::Format {
                label: "chat tool call",
                detail: "generation ended inside an unterminated direct tool call".to_string(),
            }),
        }
    }

    fn parse_reasoning(&mut self, events: &mut Vec<ChatOutputEvent>) -> bool {
        if self.pending.starts_with(GEMMA_THINK_OPEN) {
            self.pending.drain(..GEMMA_THINK_OPEN.len());
            return true;
        }
        if GEMMA_THINK_OPEN.starts_with(&self.pending) {
            return false;
        }
        let close = [THINK_CLOSE, GEMMA_THINK_CLOSE]
            .into_iter()
            .filter_map(|marker| self.pending.find(marker).map(|index| (index, marker)))
            .min_by_key(|(index, _)| *index);
        if let Some((index, marker)) = close {
            push_nonempty(
                events,
                ChatOutputEvent::Reasoning(self.pending[..index].to_string()),
            );
            self.pending.drain(..index + marker.len());
            self.mode = OutputMode::Text;
            self.trim_after_thinking = true;
            return true;
        }
        flush_safe_prefix_with_markers(
            &mut self.pending,
            [THINK_CLOSE, GEMMA_THINK_CLOSE].into_iter(),
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
        let direct_open = self
            .direct_tools
            .iter()
            .enumerate()
            .filter_map(|(tool, definition)| {
                self.pending
                    .find(&definition.open)
                    .map(|index| (index, tool))
            })
            .min_by_key(|(index, _)| *index);
        let protocol_open = [
            (TOOL_CALL_OPEN, ToolCallProtocol::Standard),
            (GEMMA_TOOL_CALL_OPEN, ToolCallProtocol::Gemma),
            (DSML_TOOL_CALLS_OPEN, ToolCallProtocol::Dsml),
            (ATEM_TOOL_CALLS_OPEN, ToolCallProtocol::Atem),
        ]
        .into_iter()
        .filter_map(|(open, protocol)| self.pending.find(open).map(|index| (index, open, protocol)))
        .min_by_key(|(index, _, _)| *index);
        if let Some((index, open, protocol)) = protocol_open.filter(|(index, _, _)| {
            direct_open.is_none_or(|(direct_index, _)| *index <= direct_index)
        }) {
            push_nonempty(
                events,
                ChatOutputEvent::Text(self.pending[..index].to_string()),
            );
            self.pending.drain(..index + open.len());
            self.tool_call_protocol = protocol;
            self.mode = OutputMode::ToolCall;
            return true;
        }
        if let Some((index, tool)) = direct_open {
            push_nonempty(
                events,
                ChatOutputEvent::Text(self.pending[..index].to_string()),
            );
            let open_len = self.direct_tools[tool].open.len();
            self.pending.drain(..index + open_len);
            self.active_direct_tool = Some(tool);
            self.mode = OutputMode::DirectToolCall;
            return true;
        }
        flush_safe_prefix_with_markers(
            &mut self.pending,
            [
                TOOL_CALL_OPEN,
                GEMMA_TOOL_CALL_OPEN,
                DSML_TOOL_CALLS_OPEN,
                ATEM_TOOL_CALLS_OPEN,
            ]
            .into_iter()
            .chain(self.direct_tools.iter().map(|tool| tool.open.as_str())),
            events,
            ChatOutputEvent::Text,
        )
    }

    fn parse_tool_call(&mut self, events: &mut Vec<ChatOutputEvent>) -> Result<bool> {
        self.tool_call.push_str(&self.pending);
        self.pending.clear();
        let close = self.tool_call_protocol.close();
        let Some(index) = self.tool_call.find(close) else {
            return Ok(false);
        };
        let body = self.tool_call[..index].to_string();
        let remainder = self.tool_call[index + close.len()..].to_string();
        self.tool_call.clear();
        self.pending = remainder;
        let functions = match self.tool_call_protocol {
            ToolCallProtocol::Standard => vec![parse_function_call(
                &body,
                &self.string_arguments,
                &self.tool_parameters,
            )?],
            ToolCallProtocol::Gemma => vec![parse_gemma_function_call(
                &body,
                &self.string_arguments,
                &self.tool_parameters,
            )?],
            ToolCallProtocol::Dsml => parse_dsml_function_calls(&body, &self.tool_parameters)?,
            ToolCallProtocol::Atem => {
                parse_atem_function_calls(&body, &self.string_arguments, &self.tool_parameters)?
            }
        };
        for function in functions {
            let id = next_tool_call_id()?;
            events.push(ChatOutputEvent::ToolCall(ChatToolCall { id, function }));
        }
        self.mode = OutputMode::Text;
        self.trim_after_tool_call = true;
        Ok(true)
    }

    fn parse_direct_tool_call(&mut self, events: &mut Vec<ChatOutputEvent>) -> Result<bool> {
        self.tool_call.push_str(&self.pending);
        self.pending.clear();
        let tool = self
            .active_direct_tool
            .expect("direct tool call has an active tool");
        let definition = &self.direct_tools[tool];
        let Some(index) = self.tool_call.find(&definition.close) else {
            return Ok(false);
        };
        let argument = strip_protocol_newlines(&self.tool_call[..index]).to_string();
        let remainder = self.tool_call[index + definition.close.len()..].to_string();
        let function = ChatFunctionCall {
            name: definition.name.clone(),
            arguments: BTreeMap::from([(definition.parameter.clone(), Value::String(argument))]),
        };
        self.tool_call.clear();
        self.pending = remainder;
        self.active_direct_tool = None;
        let id = next_tool_call_id()?;
        events.push(ChatOutputEvent::ToolCall(ChatToolCall { id, function }));
        self.mode = OutputMode::Text;
        self.trim_after_tool_call = true;
        Ok(true)
    }
}

fn flush_safe_prefix_with_markers<'a>(
    pending: &mut String,
    markers: impl Iterator<Item = &'a str>,
    events: &mut Vec<ChatOutputEvent>,
    make_event: fn(String) -> ChatOutputEvent,
) -> bool {
    let held = markers
        .map(|marker| longest_marker_prefix_suffix(pending, marker))
        .max()
        .unwrap_or(0);
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
    marker
        .char_indices()
        .map(|(index, _)| index)
        .filter(|&length| length != 0)
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

fn parse_atem_function_calls(
    body: &str,
    string_arguments: &BTreeMap<String, BTreeSet<String>>,
    tool_parameters: &BTreeMap<String, ToolParameters>,
) -> Result<Vec<ChatFunctionCall>> {
    let mut remaining = body;
    let mut functions = Vec::new();
    loop {
        remaining = trim_protocol_whitespace_start(remaining);
        if remaining.is_empty() {
            break;
        }
        let (mut attributes, invoke_body) =
            parse_dsml_open_tag(remaining, ATEM_INVOKE_OPEN, "ATEM invoke")?;
        let name = take_dsml_attribute(&mut attributes, "name", "ATEM invoke")?;
        reject_dsml_attributes(&attributes, "ATEM invoke")?;
        validate_protocol_name("function", &name)?;
        if !tool_parameters.contains_key(&name) {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("unknown function {name:?} in ATEM invocation"),
            });
        }
        let close = invoke_body
            .find(ATEM_INVOKE_CLOSE)
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing {ATEM_INVOKE_CLOSE} for {name:?}"),
            })?;
        let arguments =
            parse_atem_parameters(&name, &invoke_body[..close], string_arguments.get(&name))?;
        functions.push(ChatFunctionCall { name, arguments });
        remaining = &invoke_body[close + ATEM_INVOKE_CLOSE.len()..];
    }
    if functions.is_empty() {
        return Err(Error::Format {
            label: "chat tool call",
            detail: "ATEM function_calls block does not contain an invocation".to_string(),
        });
    }
    Ok(functions)
}

fn parse_atem_parameters(
    function: &str,
    body: &str,
    string_arguments: Option<&BTreeSet<String>>,
) -> Result<BTreeMap<String, Value>> {
    let mut remaining = body;
    let mut arguments = BTreeMap::new();
    loop {
        remaining = trim_protocol_whitespace_start(remaining);
        if remaining.is_empty() {
            break;
        }
        let (mut attributes, parameter_body) =
            parse_dsml_open_tag(remaining, ATEM_PARAMETER_OPEN, "ATEM parameter")?;
        let name = take_dsml_attribute(&mut attributes, "name", "ATEM parameter")?;
        reject_dsml_attributes(&attributes, "ATEM parameter")?;
        validate_protocol_name("parameter", &name)?;
        let close = parameter_body
            .find(ATEM_PARAMETER_CLOSE)
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing {ATEM_PARAMETER_CLOSE} for {name:?}"),
            })?;
        let raw_value = &parameter_body[..close];
        let value = if string_arguments.is_some_and(|names| names.contains(&name)) {
            Value::String(raw_value.to_string())
        } else {
            serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_string()))
        };
        if arguments.insert(name.clone(), value).is_some() {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate parameter {name:?} in ATEM invocation {function:?}"),
            });
        }
        remaining = &parameter_body[close + ATEM_PARAMETER_CLOSE.len()..];
    }
    Ok(arguments)
}

fn parse_dsml_function_calls(
    body: &str,
    tool_parameters: &BTreeMap<String, ToolParameters>,
) -> Result<Vec<ChatFunctionCall>> {
    let mut remaining = body;
    let mut functions = Vec::new();
    loop {
        remaining = trim_protocol_whitespace_start(remaining);
        if remaining.is_empty() {
            break;
        }
        let (mut attributes, invoke_body) =
            parse_dsml_open_tag(remaining, DSML_INVOKE_OPEN, "invoke")?;
        let name = take_dsml_attribute(&mut attributes, "name", "invoke")?;
        reject_dsml_attributes(&attributes, "invoke")?;
        validate_protocol_name("function", &name)?;
        if !tool_parameters.contains_key(&name) {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("unknown function {name:?} in DSML invocation"),
            });
        }
        let close = invoke_body
            .find(DSML_INVOKE_CLOSE)
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing {DSML_INVOKE_CLOSE} for {name:?}"),
            })?;
        let arguments = parse_dsml_parameters(&name, &invoke_body[..close])?;
        functions.push(ChatFunctionCall { name, arguments });
        remaining = &invoke_body[close + DSML_INVOKE_CLOSE.len()..];
    }
    if functions.is_empty() {
        return Err(Error::Format {
            label: "chat tool call",
            detail: "DSML tool_calls block does not contain an invocation".to_string(),
        });
    }
    Ok(functions)
}

fn parse_dsml_parameters(function: &str, body: &str) -> Result<BTreeMap<String, Value>> {
    let mut remaining = body;
    let mut arguments = BTreeMap::new();
    loop {
        remaining = trim_protocol_whitespace_start(remaining);
        if remaining.is_empty() {
            break;
        }
        let (mut attributes, parameter_body) =
            parse_dsml_open_tag(remaining, DSML_PARAMETER_OPEN, "parameter")?;
        let name = take_dsml_attribute(&mut attributes, "name", "parameter")?;
        let string = take_dsml_attribute(&mut attributes, "string", "parameter")?;
        reject_dsml_attributes(&attributes, "parameter")?;
        validate_protocol_name("parameter", &name)?;
        let close = parameter_body
            .find(DSML_PARAMETER_CLOSE)
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing {DSML_PARAMETER_CLOSE} for {name:?}"),
            })?;
        let raw_value = &parameter_body[..close];
        let value = match string.as_str() {
            "true" => Value::String(raw_value.to_string()),
            "false" => serde_json::from_str(raw_value).map_err(|error| Error::Format {
                label: "chat tool call",
                detail: format!(
                    "invalid JSON value for DSML parameter {name:?} in {function:?}: {error}"
                ),
            })?,
            value => {
                return Err(Error::Format {
                    label: "chat tool call",
                    detail: format!(
                        "DSML parameter {name:?} has invalid string attribute {value:?}"
                    ),
                });
            }
        };
        if arguments.insert(name.clone(), value).is_some() {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate parameter {name:?} in DSML invocation {function:?}"),
            });
        }
        remaining = &parameter_body[close + DSML_PARAMETER_CLOSE.len()..];
    }
    Ok(arguments)
}

fn parse_dsml_open_tag<'a>(
    input: &'a str,
    marker: &str,
    kind: &str,
) -> Result<(BTreeMap<String, String>, &'a str)> {
    let tag = input.strip_prefix(marker).ok_or_else(|| Error::Format {
        label: "chat tool call",
        detail: format!("expected {marker} in DSML tool call"),
    })?;
    if !tag.starts_with(|character: char| character == '>' || character.is_ascii_whitespace()) {
        return Err(Error::Format {
            label: "chat tool call",
            detail: format!("invalid DSML {kind} opening tag"),
        });
    }
    let end = tag.find('>').ok_or_else(|| Error::Format {
        label: "chat tool call",
        detail: format!("unterminated DSML {kind} opening tag"),
    })?;
    let attributes = parse_dsml_attributes(&tag[..end], kind)?;
    Ok((attributes, &tag[end + 1..]))
}

fn parse_dsml_attributes(input: &str, kind: &str) -> Result<BTreeMap<String, String>> {
    let mut remaining = input;
    let mut attributes = BTreeMap::new();
    loop {
        remaining = remaining.trim_start_matches(char::is_whitespace);
        if remaining.is_empty() {
            return Ok(attributes);
        }
        let name_end = remaining
            .find(|character: char| character == '=' || character.is_whitespace())
            .unwrap_or(remaining.len());
        let name = &remaining[..name_end];
        validate_protocol_name("attribute", name)?;
        remaining = remaining[name_end..].trim_start_matches(char::is_whitespace);
        remaining = remaining.strip_prefix('=').ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("DSML {kind} attribute {name:?} is missing '='"),
        })?;
        remaining = remaining.trim_start_matches(char::is_whitespace);
        remaining = remaining.strip_prefix('"').ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("DSML {kind} attribute {name:?} must use double quotes"),
        })?;
        let value_end = remaining.find('"').ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("unterminated DSML {kind} attribute {name:?}"),
        })?;
        let value = remaining[..value_end].to_string();
        if attributes.insert(name.to_string(), value).is_some() {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate DSML {kind} attribute {name:?}"),
            });
        }
        remaining = &remaining[value_end + 1..];
    }
}

fn take_dsml_attribute(
    attributes: &mut BTreeMap<String, String>,
    name: &str,
    kind: &str,
) -> Result<String> {
    attributes.remove(name).ok_or_else(|| Error::Format {
        label: "chat tool call",
        detail: format!("DSML {kind} is missing its {name:?} attribute"),
    })
}

fn reject_dsml_attributes(attributes: &BTreeMap<String, String>, kind: &str) -> Result<()> {
    let Some(name) = attributes.keys().next() else {
        return Ok(());
    };
    Err(Error::Format {
        label: "chat tool call",
        detail: format!("unexpected DSML {kind} attribute {name:?}"),
    })
}

fn trim_protocol_whitespace_start(input: &str) -> &str {
    input.trim_start_matches([' ', '\t', '\r', '\n'])
}

fn parse_gemma_function_call(
    body: &str,
    string_arguments: &BTreeMap<String, BTreeSet<String>>,
    tool_parameters: &BTreeMap<String, ToolParameters>,
) -> Result<ChatFunctionCall> {
    let call = body
        .trim_matches(['\r', '\n'])
        .strip_prefix("call:")
        .ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: "expected call:name{...} inside <|tool_call>".to_string(),
        })?;
    let open = call.find('{').ok_or_else(|| Error::Format {
        label: "chat tool call",
        detail: "Gemma tool call is missing its argument object".to_string(),
    })?;
    let name = call[..open].trim();
    validate_protocol_name("function", name)?;
    if !tool_parameters.contains_key(name) {
        return Err(Error::Format {
            label: "chat tool call",
            detail: format!("unknown function {name:?}"),
        });
    }
    let object = call[open..].trim();
    let inner = object
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: "Gemma tool call has an unterminated argument object".to_string(),
        })?;
    let mut arguments = BTreeMap::new();
    for field in split_gemma_top_level(inner, ',')? {
        if field.trim().is_empty() {
            continue;
        }
        let (raw_name, raw_value) = split_gemma_field(field)?;
        let parameter = strip_gemma_quote(raw_name.trim()).to_string();
        validate_protocol_name("parameter", &parameter)?;
        let value = if string_arguments
            .get(name)
            .is_some_and(|parameters| parameters.contains(&parameter))
        {
            Value::String(strip_gemma_quote(raw_value.trim()).to_string())
        } else {
            parse_gemma_value(raw_value.trim())?
        };
        if arguments.insert(parameter.clone(), value).is_some() {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate parameter {parameter:?}"),
            });
        }
    }
    Ok(ChatFunctionCall {
        name: name.to_string(),
        arguments,
    })
}

fn parse_gemma_value(raw: &str) -> Result<Value> {
    let raw = raw.trim();
    if let Some(value) = raw
        .strip_prefix("<|\"|>")
        .and_then(|value| value.strip_suffix("<|\"|>"))
    {
        return Ok(Value::String(value.to_string()));
    }
    if let Some(inner) = raw
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
    {
        let mut object = serde_json::Map::new();
        for field in split_gemma_top_level(inner, ',')? {
            if field.trim().is_empty() {
                continue;
            }
            let (name, value) = split_gemma_field(field)?;
            object.insert(
                strip_gemma_quote(name.trim()).to_string(),
                parse_gemma_value(value.trim())?,
            );
        }
        return Ok(Value::Object(object));
    }
    if let Some(inner) = raw
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
    {
        return split_gemma_top_level(inner, ',')?
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .map(|value| parse_gemma_value(value.trim()))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array);
    }
    serde_json::from_str(raw).or_else(|_| Ok(Value::String(strip_gemma_quote(raw).to_string())))
}

fn split_gemma_field(field: &str) -> Result<(&str, &str)> {
    let mut parts = split_gemma_top_level(field, ':')?;
    if parts.len() != 2 {
        return Err(Error::Format {
            label: "chat tool call",
            detail: format!("expected parameter:value, got {field:?}"),
        });
    }
    let value = parts.pop().expect("two field parts");
    let name = parts.pop().expect("two field parts");
    Ok((name, value))
}

fn split_gemma_top_level(input: &str, delimiter: char) -> Result<Vec<&str>> {
    const QUOTE: &str = "<|\"|>";
    let mut parts = Vec::new();
    let mut start = 0;
    let mut index = 0;
    let mut depth = 0usize;
    let mut quoted = false;
    while index < input.len() {
        let tail = &input[index..];
        if tail.starts_with(QUOTE) {
            quoted = !quoted;
            index += QUOTE.len();
            continue;
        }
        let character = tail.chars().next().expect("index is within input");
        let width = character.len_utf8();
        if !quoted {
            match character {
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth = depth.checked_sub(1).ok_or_else(|| Error::Format {
                        label: "chat tool call",
                        detail: "unbalanced Gemma tool argument delimiters".to_string(),
                    })?;
                }
                _ if character == delimiter && depth == 0 => {
                    parts.push(&input[start..index]);
                    start = index + width;
                }
                _ => {}
            }
        }
        index += width;
    }
    if quoted || depth != 0 {
        return Err(Error::Format {
            label: "chat tool call",
            detail: "unterminated Gemma tool argument".to_string(),
        });
    }
    parts.push(&input[start..]);
    Ok(parts)
}

fn strip_gemma_quote(value: &str) -> &str {
    value
        .strip_prefix("<|\"|>")
        .and_then(|value| value.strip_suffix("<|\"|>"))
        .or_else(|| {
            value
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
        })
        .unwrap_or(value)
}

fn parse_function_call(
    body: &str,
    string_arguments: &BTreeMap<String, BTreeSet<String>>,
    tool_parameters: &BTreeMap<String, ToolParameters>,
) -> Result<ChatFunctionCall> {
    let body = body.trim_matches(['\r', '\n']);
    if body.starts_with('{') {
        return parse_json_function_call(body, tool_parameters);
    }
    if !body.starts_with("<function=") {
        return parse_poolside_function_call(body, string_arguments);
    }
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
    let emitted_name = &function[..open_end];
    validate_protocol_name("function", emitted_name)?;
    let parameters = tool_parameters
        .get(emitted_name)
        .ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("unknown function {emitted_name:?}"),
        })?;
    let name = emitted_name.to_string();
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
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        let parameter = remaining
            .strip_prefix("<parameter=")
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!(
                    "unexpected content {} in function {name:?}",
                    protocol_preview(remaining)
                ),
            })?;
        let parameter_open_end = parameter.find('>').ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: "unterminated <parameter=...> tag".to_string(),
        })?;
        let emitted_parameter_name = &parameter[..parameter_open_end];
        validate_protocol_name("parameter", emitted_parameter_name)?;
        if !parameters.names.contains(emitted_parameter_name) {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!(
                    "unknown parameter {emitted_parameter_name:?} for function {name:?}"
                ),
            });
        }
        let parameter_name = emitted_parameter_name.to_string();
        let parameter_body = &parameter[parameter_open_end + 1..];
        let parameter_close = parameter_body
            .find("</parameter>")
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing </parameter> for {emitted_parameter_name:?}"),
            })?;
        let raw_value = strip_protocol_newlines(&parameter_body[..parameter_close]);
        let value = decode_argument(&name, &parameter_name, raw_value, string_arguments);
        if arguments.insert(parameter_name.clone(), value).is_some() {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate parameter {parameter_name:?}"),
            });
        }
        remaining = &parameter_body[parameter_close + "</parameter>".len()..];
    }

    if let Some(missing) = parameters
        .required
        .iter()
        .find(|name| !arguments.contains_key(*name))
    {
        return Err(Error::Format {
            label: "chat tool call",
            detail: format!("missing required parameter {missing:?} for function {name:?}"),
        });
    }

    Ok(ChatFunctionCall { name, arguments })
}

fn protocol_preview(value: &str) -> String {
    const LIMIT: usize = 96;
    let mut characters = value.chars();
    let preview: String = characters.by_ref().take(LIMIT).collect();
    if characters.next().is_some() {
        format!("{preview:?}...")
    } else {
        format!("{preview:?}")
    }
}

fn parse_poolside_function_call(
    body: &str,
    string_arguments: &BTreeMap<String, BTreeSet<String>>,
) -> Result<ChatFunctionCall> {
    const KEY_OPEN: &str = "<arg_key>";
    const KEY_CLOSE: &str = "</arg_key>";
    const VALUE_OPEN: &str = "<arg_value>";
    const VALUE_CLOSE: &str = "</arg_value>";

    let body = body.trim_matches(['\r', '\n']);
    let first_argument = body.find(KEY_OPEN).unwrap_or(body.len());
    let name = body[..first_argument].trim();
    validate_protocol_name("function", name)?;
    let mut remaining = &body[first_argument..];
    let mut arguments = BTreeMap::new();
    while !remaining.is_empty() {
        let key = remaining
            .strip_prefix(KEY_OPEN)
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("unexpected content in Poolside function {name:?}"),
            })?;
        let key_end = key.find(KEY_CLOSE).ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("unterminated Poolside argument name in {name:?}"),
        })?;
        let parameter = &key[..key_end];
        validate_protocol_name("parameter", parameter)?;
        let value = key[key_end + KEY_CLOSE.len()..]
            .strip_prefix(VALUE_OPEN)
            .ok_or_else(|| Error::Format {
                label: "chat tool call",
                detail: format!("missing <arg_value> for {parameter:?}"),
            })?;
        let value_end = value.find(VALUE_CLOSE).ok_or_else(|| Error::Format {
            label: "chat tool call",
            detail: format!("unterminated Poolside value for {parameter:?}"),
        })?;
        let raw_value = strip_protocol_newlines(&value[..value_end]);
        if arguments
            .insert(
                parameter.to_string(),
                decode_argument(name, parameter, raw_value, string_arguments),
            )
            .is_some()
        {
            return Err(Error::Format {
                label: "chat tool call",
                detail: format!("duplicate parameter {parameter:?}"),
            });
        }
        remaining = &value[value_end + VALUE_CLOSE.len()..];
    }
    Ok(ChatFunctionCall {
        name: name.to_string(),
        arguments,
    })
}

fn parse_json_function_call(
    body: &str,
    tool_parameters: &BTreeMap<String, ToolParameters>,
) -> Result<ChatFunctionCall> {
    let call: Value = serde_json::from_str(body).map_err(|error| Error::Format {
        label: "chat tool call",
        detail: format!("invalid JSON tool call: {error}"),
    })?;
    let object = call.as_object().ok_or_else(|| Error::Format {
        label: "chat tool call",
        detail: "JSON tool call must be an object".to_string(),
    })?;
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let arguments = match function
        .get("arguments")
        .or_else(|| function.get("parameters"))
    {
        None if function.get("name").is_none() => function.clone().into_iter().collect(),
        None | Some(Value::Null) => BTreeMap::new(),
        Some(Value::Object(arguments)) => arguments.clone().into_iter().collect(),
        Some(Value::String(arguments)) => {
            let arguments: Value =
                serde_json::from_str(arguments).map_err(|error| Error::Format {
                    label: "chat tool call",
                    detail: format!("JSON tool call has invalid string arguments: {error}"),
                })?;
            arguments
                .as_object()
                .cloned()
                .ok_or_else(|| Error::Format {
                    label: "chat tool call",
                    detail: "JSON tool call arguments must be an object".to_string(),
                })?
                .into_iter()
                .collect()
        }
        Some(_) => {
            return Err(Error::Format {
                label: "chat tool call",
                detail: "JSON tool call arguments must be an object".to_string(),
            });
        }
    };
    let name = match function.get("name").and_then(Value::as_str) {
        Some(name) => name.to_string(),
        None => infer_json_function_name(&arguments, tool_parameters)?,
    };
    validate_protocol_name("function", &name)?;
    Ok(ChatFunctionCall { name, arguments })
}

fn infer_json_function_name(
    arguments: &BTreeMap<String, Value>,
    tool_parameters: &BTreeMap<String, ToolParameters>,
) -> Result<String> {
    let argument_names: BTreeSet<_> = arguments.keys().collect();
    let candidates: Vec<_> = tool_parameters
        .iter()
        .filter(|(_, parameters)| {
            !parameters.names.is_empty()
                && argument_names
                    .iter()
                    .all(|name| parameters.names.contains(*name))
                && parameters
                    .required
                    .iter()
                    .all(|name| argument_names.contains(name))
        })
        .map(|(name, _)| name)
        .collect();
    match candidates.as_slice() {
        [name] => Ok((*name).clone()),
        [] => Err(Error::Format {
            label: "chat tool call",
            detail: "JSON tool call is missing its function name and does not match a tool schema"
                .to_string(),
        }),
        _ => Err(Error::Format {
            label: "chat tool call",
            detail: "JSON tool call is missing its function name and matches multiple tool schemas"
                .to_string(),
        }),
    }
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
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
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

    fn dsml_tools() -> Vec<ChatTool> {
        vec![
            ChatTool::function(ChatFunctionDefinition {
                name: "write_file".to_string(),
                description: Some("Write a file".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "contents": {"type": "string"},
                        "executable": {"type": "boolean"}
                    },
                    "required": ["path", "contents"]
                }),
            }),
            ChatTool::function(ChatFunctionDefinition {
                name: "bash".to_string(),
                description: Some("Run a command".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "options": {"type": "object"}
                    },
                    "required": ["command"]
                }),
            }),
        ]
    }

    fn dsml_protocol_text() -> &'static str {
        concat!(
            "I will update it.",
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"write_file\">\n",
            "<｜DSML｜parameter name=\"path\" string=\"true\">src/main.rs",
            "</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"contents\" string=\"true\">",
            "fn main() {\n    println!(\"hi\");\n}",
            "</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"executable\" string=\"false\">false",
            "</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"bash\">\n",
            "<｜DSML｜parameter name=\"command\" string=\"true\">cargo test",
            "</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"options\" string=\"false\">",
            "{\"cwd\":\"/workspace\"}",
            "</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>"
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

    fn dsml_expected() -> Vec<ChatOutputEvent> {
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
            ChatOutputEvent::ToolCall(ChatToolCall {
                id: "call_ID".to_string(),
                function: ChatFunctionCall {
                    name: "bash".to_string(),
                    arguments: BTreeMap::from([
                        ("command".to_string(), json!("cargo test")),
                        ("options".to_string(), json!({"cwd": "/workspace"})),
                    ]),
                },
            }),
        ]
    }

    #[test]
    fn poolside_tool_call_decodes_tagged_arguments() {
        let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
        let events = parser
            .push_text(concat!(
                "<tool_call>write_file",
                "<arg_key>path</arg_key><arg_value>src/main.rs</arg_value>",
                "<arg_key>contents</arg_key><arg_value>fn main() {}</arg_value>",
                "<arg_key>executable</arg_key><arg_value>false</arg_value>",
                "</tool_call>"
            ))
            .unwrap();
        assert_eq!(
            normalized(events),
            vec![ChatOutputEvent::ToolCall(ChatToolCall {
                id: "call_ID".to_string(),
                function: ChatFunctionCall {
                    name: "write_file".to_string(),
                    arguments: BTreeMap::from([
                        ("contents".to_string(), json!("fn main() {}")),
                        ("executable".to_string(), json!(false)),
                        ("path".to_string(), json!("src/main.rs")),
                    ]),
                },
            })]
        );
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
    fn dsml_tool_protocol_survives_every_character_boundary() {
        let text = dsml_protocol_text();
        for split in text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()))
        {
            let mut parser = ChatOutputParser::new(&dsml_tools(), false).unwrap();
            let mut events = parser.push_text(&text[..split]).unwrap();
            events.extend(parser.push_text(&text[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(normalized(events), dsml_expected(), "split {split}");
        }
    }

    #[test]
    fn muse_recipient_segments_separate_reasoning_and_answer_text() {
        let mut parser = ChatOutputParser::new(&[], false).unwrap();
        parser.begin_muse_segment("assistant to=self").unwrap();
        assert_eq!(
            parser.push_text("check the result").unwrap(),
            [ChatOutputEvent::Reasoning("check the result".to_string())]
        );
        assert!(parser.end_muse_segment().unwrap().is_empty());
        parser.begin_muse_segment("assistant to=user").unwrap();
        assert_eq!(
            parser.push_text("The result is four.").unwrap(),
            [ChatOutputEvent::Text("The result is four.".to_string())]
        );
        assert!(parser.end_muse_segment().unwrap().is_empty());
    }

    #[test]
    fn atem_tool_protocol_survives_every_character_boundary() {
        let text = concat!(
            "<atem:function_calls>\n",
            "<atem:invoke name=\"write_file\">\n",
            "<atem:parameter name=\"path\">src/main.rs</atem:parameter>\n",
            "<atem:parameter name=\"contents\">fn main() {}</atem:parameter>\n",
            "<atem:parameter name=\"executable\">false</atem:parameter>\n",
            "</atem:invoke>\n",
            "</atem:function_calls>"
        );
        let expected = vec![ChatOutputEvent::ToolCall(ChatToolCall {
            id: "call_ID".to_string(),
            function: ChatFunctionCall {
                name: "write_file".to_string(),
                arguments: BTreeMap::from([
                    ("contents".to_string(), json!("fn main() {}")),
                    ("executable".to_string(), json!(false)),
                    ("path".to_string(), json!("src/main.rs")),
                ]),
            },
        })];
        for split in text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()))
        {
            let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
            let mut events = parser.push_text(&text[..split]).unwrap();
            events.extend(parser.push_text(&text[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(normalized(events), expected, "split {split}");
        }
    }

    #[test]
    fn dsml_tool_protocol_rejects_malformed_calls() {
        let cases = [
            concat!(
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"unknown\"></｜DSML｜invoke>",
                "</｜DSML｜tool_calls>"
            ),
            concat!(
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"bash\">",
                "<｜DSML｜parameter name=\"options\" string=\"false\">not-json",
                "</｜DSML｜parameter>",
                "</｜DSML｜invoke>",
                "</｜DSML｜tool_calls>"
            ),
            concat!(
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"bash\" extra=\"value\"></｜DSML｜invoke>",
                "</｜DSML｜tool_calls>"
            ),
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
        ];
        for input in cases {
            let mut parser = ChatOutputParser::new(&dsml_tools(), false).unwrap();
            assert!(parser.push_text(input).is_err(), "{input}");
        }
    }

    #[test]
    fn dsml_truncation_discards_an_incomplete_tool_block() {
        let mut parser = ChatOutputParser::new(&dsml_tools(), false).unwrap();
        let events = parser
            .push_text(concat!(
                "working",
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"bash\">",
                "<｜DSML｜parameter name=\"command\" string=\"true\">git status"
            ))
            .unwrap();
        assert_eq!(events, [ChatOutputEvent::Text("working".to_string())]);
        assert!(parser.finish_truncated().unwrap().is_empty());

        let mut parser = ChatOutputParser::new(&dsml_tools(), false).unwrap();
        parser
            .push_text("<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">")
            .unwrap();
        let error = parser.finish().unwrap_err();
        assert!(error.to_string().contains(DSML_TOOL_CALLS_CLOSE));
    }

    #[test]
    fn json_tool_protocol_survives_every_chunk_boundary() {
        let text = concat!(
            "I will update it.<tool_call>",
            r#"{"name":"write_file","arguments":{"path":"src/main.rs","contents":"fn main() {\n    println!(\"hi\");\n}","executable":false}}"#,
            "</tool_call>"
        );
        for split in 0..=text.len() {
            assert_eq!(
                parse_chunks(&[&text[..split], &text[split..]]),
                expected(),
                "split {split}"
            );
        }
    }

    #[test]
    fn nameless_json_tool_call_uses_a_unique_schema_match() {
        let tools = vec![
            ChatTool::function(ChatFunctionDefinition {
                name: "bash".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"}
                    },
                    "required": ["command"]
                }),
            }),
            ChatTool::function(ChatFunctionDefinition {
                name: "apty".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"]
                }),
            }),
        ];
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let events = parser
            .push_text(r#"<tool_call>{"command":"git status","timeout":5}</tool_call>"#)
            .unwrap();
        assert_eq!(
            normalized(events),
            [ChatOutputEvent::ToolCall(ChatToolCall {
                id: "call_ID".to_string(),
                function: ChatFunctionCall {
                    name: "bash".to_string(),
                    arguments: BTreeMap::from([
                        ("command".to_string(), json!("git status")),
                        ("timeout".to_string(), json!(5)),
                    ]),
                },
            })]
        );
    }

    #[test]
    fn direct_tool_wrapper_survives_every_chunk_boundary() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "bash".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                },
                "required": ["command"]
            }),
        })];
        let text = "<bash>\ngit diff crates/infer/src/runtime/chat_output.rs\n</bash>";
        for split in 0..=text.len() {
            let mut parser = ChatOutputParser::new(&tools, false).unwrap();
            let mut events = parser.push_text(&text[..split]).unwrap();
            events.extend(parser.push_text(&text[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(
                normalized(events),
                [ChatOutputEvent::ToolCall(ChatToolCall {
                    id: "call_ID".to_string(),
                    function: ChatFunctionCall {
                        name: "bash".to_string(),
                        arguments: BTreeMap::from([(
                            "command".to_string(),
                            json!("git diff crates/infer/src/runtime/chat_output.rs"),
                        )]),
                    },
                })],
                "split {split}"
            );
        }
    }

    #[test]
    fn nameless_json_tool_call_rejects_ambiguous_schemas() {
        let mut alternative = tools().pop().unwrap();
        alternative.function.name = "rewrite_file".to_string();
        let tools = vec![tools().pop().unwrap(), alternative];
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let error = parser
            .push_text(r#"<tool_call>{"path":"src/main.rs"}</tool_call>"#)
            .unwrap_err();
        assert!(error.to_string().contains("matches multiple tool schemas"));
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
    fn closing_think_tag_switches_output_mode_to_text() {
        let mut parser = ChatOutputParser::new(&[], true).unwrap();
        assert_eq!(parser.mode, OutputMode::Reasoning);
        parser.push_text("checking</think>\n\nanswer").unwrap();
        assert_eq!(parser.mode, OutputMode::Text);
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
    fn standard_tool_call_accepts_indented_parameter_tags() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "list_files".to_string(),
            description: Some("List files".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        })];
        let text = concat!(
            "<tool_call>\n",
            "<function=list_files>\n",
            "  <parameter=path>\n.\n</parameter>\n",
            " \t</function>\n",
            "</tool_call>"
        );
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let mut events = parser.push_text(text).unwrap();
        events.extend(parser.finish().unwrap());
        assert_eq!(events.len(), 1);
        let ChatOutputEvent::ToolCall(call) = &events[0] else {
            panic!("expected tool call");
        };
        assert_eq!(call.function.name, "list_files");
        assert_eq!(call.function.arguments.get("path"), Some(&json!(".")));
    }

    #[test]
    fn standard_tool_call_rejects_wrong_function_name_casing() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "bash".to_string(),
            description: Some("Run a shell command".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }),
        })];
        let text = concat!(
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\npwd\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let message = parser.push_text(text).unwrap_err().to_string();
        assert!(message.contains(r#"unknown function "Bash""#), "{message}");
    }

    #[test]
    fn standard_tool_call_rejects_unadvertised_function_alias() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "ls".to_string(),
            description: Some("List files".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        })];
        let text = concat!(
            "<tool_call>\n",
            "<function=list_files>\n",
            "<parameter=path>\n/home/ryan/src/spark-infer\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let message = parser.push_text(text).unwrap_err().to_string();
        assert!(
            message.contains(r#"unknown function "list_files""#),
            "{message}"
        );
    }

    #[test]
    fn standard_tool_call_rejects_unadvertised_parameter_alias() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "read".to_string(),
            description: Some("Read a file".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"}
                },
                "required": ["path"]
            }),
        })];
        let text = concat!(
            "<tool_call>\n",
            "<function=read>\n",
            "<parameter=file_path>\n/home/ryan/src/spark-infer/README.md\n</parameter>\n",
            "<parameter=offset>\n1\n</parameter>\n",
            "<parameter=limit>\n100\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let message = parser.push_text(text).unwrap_err().to_string();
        assert!(
            message.contains(r#"unknown parameter "file_path""#),
            "{message}"
        );
    }

    #[test]
    fn standard_tool_call_rejects_missing_required_parameter() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "read".to_string(),
            description: Some("Read a file".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "number"}
                },
                "required": ["path"]
            }),
        })];
        let text = concat!(
            "<tool_call>\n",
            "<function=read>\n",
            "<parameter=offset>\n1\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let mut parser = ChatOutputParser::new(&tools, false).unwrap();
        let message = parser.push_text(text).unwrap_err().to_string();
        assert!(
            message.contains(r#"missing required parameter "path""#),
            "{message}"
        );
    }

    #[test]
    fn standard_tool_call_reports_unexpected_function_body() {
        let text = concat!(
            "<tool_call>\n",
            "<function=write_file>\n",
            "{\"name\":\"bash\"}\n",
            "</function>\n",
            "</tool_call>"
        );
        let mut parser = ChatOutputParser::new(&tools(), false).unwrap();
        let error = parser.push_text(text).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("unexpected content"), "{message}");
        assert!(message.contains(r#"{\"name\":\"bash\"}"#), "{message}");
        assert!(message.contains(r#"function "write_file""#), "{message}");
    }

    #[test]
    fn gemma_reasoning_channel_survives_every_chunk_boundary() {
        let text = "<|channel>thought\nchecking details<channel|>\nThe result is ready.";
        for split in 0..=text.len() {
            let mut parser = ChatOutputParser::new(&[], true).unwrap();
            let mut events = parser.push_text(&text[..split]).unwrap();
            events.extend(parser.push_text(&text[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(
                normalized(events),
                [
                    ChatOutputEvent::Reasoning("checking details".to_string()),
                    ChatOutputEvent::Text("The result is ready.".to_string()),
                ],
                "split {split}"
            );
        }
    }

    #[test]
    fn gemma_tool_call_survives_every_chunk_boundary() {
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "bash".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                },
                "required": ["command"]
            }),
        })];
        let text = r#"<|tool_call>call:bash{command:<|"|>git diff<|"|>,timeout:10}<tool_call|>"#;
        for split in 0..=text.len() {
            let mut parser = ChatOutputParser::new(&tools, false).unwrap();
            let mut events = parser.push_text(&text[..split]).unwrap();
            events.extend(parser.push_text(&text[split..]).unwrap());
            events.extend(parser.finish().unwrap());
            assert_eq!(
                normalized(events),
                [ChatOutputEvent::ToolCall(ChatToolCall {
                    id: "call_ID".to_string(),
                    function: ChatFunctionCall {
                        name: "bash".to_string(),
                        arguments: BTreeMap::from([
                            ("command".to_string(), json!("git diff")),
                            ("timeout".to_string(), json!(10)),
                        ]),
                    },
                })],
                "split {split}"
            );
        }
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

    #[test]
    #[ignore = "requires a prepared local DeepSeek V4 thin checkpoint"]
    fn local_deepseek_tokenizer_stream_recovers_reasoning_and_dsml_tools() {
        let model_dir =
            std::env::var("DEEPSEEK4_THIN_DIR").expect("DEEPSEEK4_THIN_DIR must be set");
        let tokenizer = Tokenizer::from_file(Path::new(&model_dir).join("tokenizer.json")).unwrap();
        let generated = format!("checked</think>\n\n{}", dsml_protocol_text());
        let encoding = tokenizer.encode(generated, false).unwrap();
        let mut codec = ChatOutputCodec::new(&tokenizer, &dsml_tools(), true).unwrap();
        let mut events = Vec::new();
        for &token in encoding.get_ids() {
            events.extend(codec.push_token(token).unwrap());
        }
        events.extend(codec.finish().unwrap());

        let events = normalized(events);
        assert_eq!(events[0], ChatOutputEvent::Reasoning("checked".to_string()));
        assert_eq!(&events[1..], dsml_expected());
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_gemma_tokenizer_stream_recovers_reasoning_and_tool_call() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let tools = vec![ChatTool::function(ChatFunctionDefinition {
            name: "bash".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                },
                "required": ["command"]
            }),
        })];
        let generated = concat!(
            "<|channel>thought\nchecked<channel|>",
            "<|tool_call>call:bash{command:<|\"|>git diff<|\"|>,timeout:10}<tool_call|>"
        );
        let encoding = tokenizer.encode(generated, false).unwrap();
        let mut codec = ChatOutputCodec::new(&tokenizer, &tools, true).unwrap();
        let mut events = Vec::new();
        for &token in encoding.get_ids() {
            events.extend(codec.push_token(token).unwrap());
        }
        events.extend(codec.finish().unwrap());
        assert_eq!(
            normalized(events),
            [
                ChatOutputEvent::Reasoning("checked".to_string()),
                ChatOutputEvent::ToolCall(ChatToolCall {
                    id: "call_ID".to_string(),
                    function: ChatFunctionCall {
                        name: "bash".to_string(),
                        arguments: BTreeMap::from([
                            ("command".to_string(), json!("git diff")),
                            ("timeout".to_string(), json!(10)),
                        ]),
                    },
                }),
            ]
        );
    }
}
