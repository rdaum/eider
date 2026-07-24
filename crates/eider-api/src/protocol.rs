//! Responses API request translation and streaming event construction.

use infer::runtime::chat::{
    ChatFunctionCall, ChatFunctionDefinition, ChatMessage, ChatReasoningEffort, ChatRole,
    ChatTemplateOptions, ChatTool, ChatToolCall,
};
use infer::runtime::chat_output::ChatOutputEvent;
use infer::runtime::generation::GenerationConfig;
use infer::runtime::sampling::SamplingConfig;
use infer::runtime::scheduler::RequestConfig;
use infer::runtime::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;

static NEXT_RESPONSE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ITEM_ID: AtomicU64 = AtomicU64::new(1);

/// JSON body accepted by `POST /v1/responses`.
#[derive(Clone, Debug, Deserialize)]
pub struct ResponseRequest {
    pub model: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub reasoning: Option<ResponseReasoning>,
    #[serde(default)]
    pub stop: Option<OneOrMany<String>>,
}

/// A field represented as either one value or an array by compatible clients.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

/// Reasoning controls accepted from Responses-compatible clients.
#[derive(Clone, Debug, Deserialize)]
pub struct ResponseReasoning {
    pub effort: Option<String>,
}

impl<T> OneOrMany<T> {
    pub(crate) fn into_vec(self) -> Vec<T> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

/// Stable OpenAI-style error body.
#[derive(Clone, Debug, Serialize)]
pub struct ErrorEnvelope {
    pub error: ApiError,
}

/// One API request or inference failure.
#[derive(Clone, Debug, Serialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn invalid(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "invalid_request_error",
            param: Some(param.into()),
            code: None,
        }
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "server_error",
            param: None,
            code: None,
        }
    }

    pub fn envelope(self) -> ErrorEnvelope {
        ErrorEnvelope { error: self }
    }
}

impl ResponseRequest {
    /// Converts a Responses request into the runtime's complete chat contract.
    pub fn into_chat_request(self, defaults: &GenerationConfig) -> Result<ChatRequest, ApiError> {
        if self.parallel_tool_calls == Some(true) {
            return Err(ApiError::invalid(
                "parallel_tool_calls",
                "parallel tool calls are not supported",
            ));
        }
        let expose_tools = validate_tool_choice(self.tool_choice.as_ref())?;

        let mut messages = Vec::new();
        if let Some(instructions) = self.instructions {
            messages.push(ChatMessage::system(instructions));
        }
        messages.extend(parse_input(self.input)?);
        messages = coalesce_system_messages(messages);
        if messages.is_empty() {
            return Err(ApiError::invalid("input", "input must not be empty"));
        }
        let preserve_thinking = messages
            .iter()
            .any(|message| message.reasoning_content.is_some());

        let tools = if expose_tools {
            self.tools
                .into_iter()
                .filter_map(parse_tool)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let has_sampling_override = self.temperature.is_some()
            || self.top_k.is_some()
            || self.top_p.is_some()
            || self.seed.is_some()
            || self.presence_penalty.is_some()
            || self.frequency_penalty.is_some();
        let mut sampling = if defaults.sampling.uses_fast_argmax() && has_sampling_override {
            SamplingConfig::default()
        } else {
            defaults.sampling
        };
        if let Some(temperature) = self.temperature {
            sampling.temperature = temperature;
        }
        if let Some(top_k) = self.top_k {
            sampling.top_k = top_k;
        }
        if let Some(top_p) = self.top_p {
            sampling.top_p = top_p;
        }
        if let Some(seed) = self.seed {
            sampling.seed = Some(seed);
        }
        if let Some(penalty) = self.presence_penalty {
            sampling.presence_penalty = penalty;
        }
        if let Some(penalty) = self.frequency_penalty {
            sampling.frequency_penalty = penalty;
        }
        let generation = RequestConfig {
            sampling,
            max_new_tokens: self.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
            eos_token_ids: defaults.eos_token_ids.clone(),
        };
        generation
            .validate()
            .map_err(|error| ApiError::invalid("sampling", error.to_string()))?;

        let requested_reasoning_effort = self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref());
        let enable_thinking =
            self.reasoning.is_some() && requested_reasoning_effort != Some("none");
        let reasoning_effort = requested_reasoning_effort
            .filter(|effort| *effort != "none")
            .map(parse_reasoning_effort)
            .transpose()?;

        Ok(ChatRequest {
            messages,
            tools,
            template: ChatTemplateOptions {
                enable_thinking,
                preserve_thinking,
                reasoning_effort,
                ..ChatTemplateOptions::default()
            },
            generation,
            stop_sequences: self.stop.map(OneOrMany::into_vec).unwrap_or_default(),
        })
    }
}

fn parse_reasoning_effort(value: &str) -> Result<ChatReasoningEffort, ApiError> {
    match value {
        "low" => Ok(ChatReasoningEffort::Low),
        "medium" => Ok(ChatReasoningEffort::Medium),
        "high" => Ok(ChatReasoningEffort::High),
        _ => Err(ApiError::invalid(
            "reasoning.effort",
            "reasoning effort must be low, medium, or high",
        )),
    }
}

fn coalesce_system_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut system = Vec::new();
    let mut conversation = Vec::new();
    for message in messages {
        if message.role == ChatRole::System {
            if let Some(content) = message.content {
                system.push(content);
            }
        } else {
            conversation.push(message);
        }
    }
    if system.is_empty() {
        return conversation;
    }
    let mut messages = Vec::with_capacity(conversation.len() + 1);
    messages.push(ChatMessage::system(system.join("\n\n")));
    messages.extend(conversation);
    messages
}

fn validate_tool_choice(choice: Option<&Value>) -> Result<bool, ApiError> {
    match choice {
        None => Ok(true),
        Some(Value::String(value)) if value == "auto" => Ok(true),
        Some(Value::String(value)) if value == "none" => Ok(false),
        Some(_) => Err(ApiError::invalid(
            "tool_choice",
            "only auto and none tool choice are supported",
        )),
    }
}

fn parse_tool(value: Value) -> Option<Result<ChatTool, ApiError>> {
    let kind = value.get("type").and_then(Value::as_str)?;
    if kind != "function" {
        return None;
    }
    Some((|| {
        let name = required_string(&value, "name", "tools")?;
        let parameters = value
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        if !parameters.is_object() {
            return Err(ApiError::invalid(
                "tools",
                format!("function {name:?} parameters must be a JSON Schema object"),
            ));
        }
        Ok(ChatTool::function(ChatFunctionDefinition {
            name,
            description: value
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
            parameters,
        }))
    })())
}

fn parse_input(input: Value) -> Result<Vec<ChatMessage>, ApiError> {
    match input {
        Value::String(text) => Ok(vec![ChatMessage::user(text)]),
        Value::Array(items) => parse_input_items(items),
        Value::Null => Ok(Vec::new()),
        _ => Err(ApiError::invalid(
            "input",
            "input must be a string or an array of input items",
        )),
    }
}

fn parse_input_items(items: Vec<Value>) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::new();
    let mut assistant = None;
    for item in items {
        let kind = match item.get("type").and_then(Value::as_str) {
            Some(kind) => kind.to_string(),
            None if item.get("role").and_then(Value::as_str).is_some() => "message".to_string(),
            None => {
                return Err(ApiError::invalid(
                    "input",
                    "input item is missing a string type field",
                ));
            }
        };
        match kind.as_str() {
            "message" => {
                let message = parse_message(&item)?;
                if message.role == ChatRole::Assistant {
                    append_assistant_message(&mut assistant, message);
                } else {
                    flush_assistant_message(&mut messages, &mut assistant);
                    messages.push(message);
                }
            }
            "function_call" => {
                let call = parse_function_call(&item)?;
                pending_assistant(&mut assistant).tool_calls.push(call);
            }
            "function_call_output" => {
                flush_assistant_message(&mut messages, &mut assistant);
                messages.push(parse_function_output(&item)?);
            }
            "reasoning" => {
                if let Some(reasoning) = parse_reasoning(&item)? {
                    append_text(
                        &mut pending_assistant(&mut assistant).reasoning_content,
                        reasoning,
                    );
                }
            }
            other => {
                return Err(ApiError::invalid(
                    "input",
                    format!("unsupported input item type {other:?}"),
                ));
            }
        }
    }
    flush_assistant_message(&mut messages, &mut assistant);
    Ok(messages)
}

fn pending_assistant(assistant: &mut Option<ChatMessage>) -> &mut ChatMessage {
    assistant.get_or_insert_with(|| ChatMessage::assistant_tool_calls(None, None, Vec::new()))
}

fn append_assistant_message(assistant: &mut Option<ChatMessage>, mut message: ChatMessage) {
    let pending = pending_assistant(assistant);
    if let Some(content) = message.content.take() {
        append_text(&mut pending.content, content);
    }
    if let Some(reasoning) = message.reasoning_content.take() {
        append_text(&mut pending.reasoning_content, reasoning);
    }
    pending.tool_calls.append(&mut message.tool_calls);
}

fn append_text(target: &mut Option<String>, text: String) {
    match target {
        Some(target) => target.push_str(&text),
        None => *target = Some(text),
    }
}

fn flush_assistant_message(messages: &mut Vec<ChatMessage>, assistant: &mut Option<ChatMessage>) {
    if let Some(assistant) = assistant.take() {
        messages.push(assistant);
    }
}

fn parse_message(value: &Value) -> Result<ChatMessage, ApiError> {
    let role = match required_string(value, "role", "input")?.as_str() {
        "developer" | "system" => ChatRole::System,
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        other => {
            return Err(ApiError::invalid(
                "input",
                format!("unsupported message role {other:?}"),
            ));
        }
    };
    let content = parse_content(value.get("content"))?;
    if role == ChatRole::Tool {
        let call_id = required_string(value, "tool_call_id", "input")?;
        return Ok(ChatMessage::tool(call_id, content));
    }
    Ok(match role {
        ChatRole::System => ChatMessage::system(content),
        ChatRole::User => ChatMessage::user(content),
        ChatRole::Assistant => ChatMessage::assistant(content),
        ChatRole::Tool => unreachable!(),
    })
}

fn parse_content(value: Option<&Value>) -> Result<String, ApiError> {
    match value {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                let kind = required_string(part, "type", "input")?;
                match kind.as_str() {
                    "input_text" | "output_text" => {
                        text.push_str(&required_string(part, "text", "input")?);
                    }
                    other => {
                        return Err(ApiError::invalid(
                            "input",
                            format!("unsupported message content type {other:?}"),
                        ));
                    }
                }
            }
            Ok(text)
        }
        None | Some(Value::Null) => Ok(String::new()),
        Some(_) => Err(ApiError::invalid(
            "input",
            "message content must be text or an array of text parts",
        )),
    }
}

fn parse_reasoning(value: &Value) -> Result<Option<String>, ApiError> {
    let Some(summary) = value.get("summary") else {
        return Ok(None);
    };
    let Value::Array(parts) = summary else {
        return Err(ApiError::invalid(
            "input",
            "reasoning summary must be an array",
        ));
    };
    let mut reasoning = String::new();
    for part in parts {
        if part.get("type").and_then(Value::as_str) != Some("summary_text") {
            continue;
        }
        reasoning.push_str(&required_string(part, "text", "input")?);
    }
    Ok((!reasoning.is_empty()).then_some(reasoning))
}

fn parse_function_call(value: &Value) -> Result<ChatToolCall, ApiError> {
    let call_id = required_string(value, "call_id", "input")?;
    let name = required_string(value, "name", "input")?;
    let arguments = required_string(value, "arguments", "input")?;
    let arguments =
        serde_json::from_str::<BTreeMap<String, Value>>(&arguments).map_err(|error| {
            ApiError::invalid(
                "input",
                format!("function call {name:?} has invalid JSON arguments: {error}"),
            )
        })?;
    Ok(ChatToolCall {
        id: call_id,
        function: ChatFunctionCall { name, arguments },
    })
}

fn parse_function_output(value: &Value) -> Result<ChatMessage, ApiError> {
    let call_id = required_string(value, "call_id", "input")?;
    let output = match value.get("output") {
        Some(Value::String(output)) => output.clone(),
        Some(output) => serde_json::to_string(output).expect("JSON value serializes"),
        None => {
            return Err(ApiError::invalid(
                "input",
                "function output is missing output",
            ));
        }
    };
    Ok(ChatMessage::tool(call_id, output))
}

fn required_string(value: &Value, key: &str, param: &str) -> Result<String, ApiError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ApiError::invalid(param, format!("missing string field {key:?}")))
}

/// One inference event delivered from the actor to an API request.
#[derive(Clone, Debug)]
pub enum InferenceEvent {
    Output(ChatOutputEvent),
    Finished(InferenceFinished),
    Error(String),
}

/// Request terminal metadata after the actor releases runtime state.
#[derive(Clone, Debug)]
pub struct InferenceFinished {
    pub finish_reason: ChatFinishReason,
    pub usage: ChatUsage,
}

/// Per-response event builder and non-streaming accumulator.
pub struct ResponseStream {
    response_id: String,
    model: String,
    created_at: u64,
    output: Vec<Value>,
    reasoning: Option<ReasoningItem>,
    text: Option<TextItem>,
    completed: bool,
}

struct ReasoningItem {
    id: String,
    output_index: usize,
    text: String,
}

struct TextItem {
    id: String,
    output_index: usize,
    text: String,
}

impl ResponseStream {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            response_id: next_id("resp", &NEXT_RESPONSE_ID),
            model: model.into(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            output: Vec::new(),
            reasoning: None,
            text: None,
            completed: false,
        }
    }

    pub fn response_id(&self) -> &str {
        &self.response_id
    }

    pub fn created(&self) -> Value {
        event(
            "response.created",
            json!({"response": self.response("in_progress", None)}),
        )
    }

    pub fn push(&mut self, inference: InferenceEvent) -> Vec<Value> {
        match inference {
            InferenceEvent::Output(ChatOutputEvent::Reasoning(delta)) => self.push_reasoning(delta),
            InferenceEvent::Output(ChatOutputEvent::Text(delta)) => {
                let mut events = self.close_reasoning();
                events.extend(self.push_text(delta));
                events
            }
            InferenceEvent::Output(ChatOutputEvent::ToolCall(call)) => self.push_tool(call),
            InferenceEvent::Finished(finished) => self.finish(finished),
            InferenceEvent::Error(message) => {
                self.completed = true;
                vec![event(
                    "error",
                    json!({
                        "error": {
                            "type": "server_error",
                            "code": "inference_error",
                            "message": message,
                            "param": null
                        }
                    }),
                )]
            }
        }
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    fn push_reasoning(&mut self, delta: String) -> Vec<Value> {
        if delta.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        if self.reasoning.is_none() {
            let id = next_id("rs", &NEXT_ITEM_ID);
            let output_index = self.output.len();
            events.push(event(
                "response.output_item.added",
                json!({
                    "output_index": output_index,
                    "item": reasoning_item(&id, "in_progress", "")
                }),
            ));
            events.push(event(
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": reasoning_summary("")
                }),
            ));
            self.reasoning = Some(ReasoningItem {
                id,
                output_index,
                text: String::new(),
            });
        }
        let reasoning = self.reasoning.as_mut().expect("reasoning item exists");
        reasoning.text.push_str(&delta);
        events.push(event(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": reasoning.id,
                "output_index": reasoning.output_index,
                "summary_index": 0,
                "delta": delta
            }),
        ));
        events
    }

    fn close_reasoning(&mut self) -> Vec<Value> {
        let Some(reasoning) = self.reasoning.take() else {
            return Vec::new();
        };
        let item = reasoning_item(&reasoning.id, "completed", &reasoning.text);
        self.output.push(item.clone());
        vec![
            event(
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": reasoning.id,
                    "output_index": reasoning.output_index,
                    "summary_index": 0,
                    "text": reasoning.text
                }),
            ),
            event(
                "response.reasoning_summary_part.done",
                json!({
                    "item_id": reasoning.id,
                    "output_index": reasoning.output_index,
                    "summary_index": 0,
                    "part": reasoning_summary(&reasoning.text)
                }),
            ),
            event(
                "response.output_item.done",
                json!({"output_index": reasoning.output_index, "item": item}),
            ),
        ]
    }

    fn push_text(&mut self, delta: String) -> Vec<Value> {
        if delta.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        if self.text.is_none() {
            let id = next_id("msg", &NEXT_ITEM_ID);
            let output_index = self.output.len();
            events.push(event(
                "response.output_item.added",
                json!({
                    "output_index": output_index,
                    "item": message_item(&id, "in_progress", "")
                }),
            ));
            events.push(event(
                "response.content_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": output_text("")
                }),
            ));
            self.text = Some(TextItem {
                id,
                output_index,
                text: String::new(),
            });
        }
        let text = self.text.as_mut().expect("text item exists");
        text.text.push_str(&delta);
        events.push(event(
            "response.output_text.delta",
            json!({
                "item_id": text.id,
                "output_index": text.output_index,
                "content_index": 0,
                "delta": delta
            }),
        ));
        events
    }

    fn close_text(&mut self) -> Vec<Value> {
        let Some(text) = self.text.take() else {
            return Vec::new();
        };
        let item = message_item(&text.id, "completed", &text.text);
        self.output.push(item.clone());
        vec![
            event(
                "response.output_text.done",
                json!({
                    "item_id": text.id,
                    "output_index": text.output_index,
                    "content_index": 0,
                    "text": text.text
                }),
            ),
            event(
                "response.content_part.done",
                json!({
                    "item_id": text.id,
                    "output_index": text.output_index,
                    "content_index": 0,
                    "part": output_text(&text.text)
                }),
            ),
            event(
                "response.output_item.done",
                json!({"output_index": text.output_index, "item": item}),
            ),
        ]
    }

    fn push_tool(&mut self, call: ChatToolCall) -> Vec<Value> {
        let mut events = self.close_reasoning();
        events.extend(self.close_text());
        let id = next_id("fc", &NEXT_ITEM_ID);
        let output_index = self.output.len();
        let arguments = serde_json::to_string(&call.function.arguments)
            .expect("tool arguments are serializable");
        let in_progress = function_item(&id, &call.id, &call.function.name, "", "in_progress");
        let completed = function_item(&id, &call.id, &call.function.name, &arguments, "completed");
        events.extend([
            event(
                "response.output_item.added",
                json!({"output_index": output_index, "item": in_progress}),
            ),
            event(
                "response.function_call_arguments.delta",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "delta": arguments
                }),
            ),
            event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "name": call.function.name,
                    "arguments": arguments
                }),
            ),
            event(
                "response.output_item.done",
                json!({"output_index": output_index, "item": completed.clone()}),
            ),
        ]);
        self.output.push(completed);
        events
    }

    fn finish(&mut self, finished: InferenceFinished) -> Vec<Value> {
        let mut events = self.close_reasoning();
        events.extend(self.close_text());
        self.completed = true;
        let incomplete = matches!(finished.finish_reason, ChatFinishReason::Length)
            .then(|| json!({"reason": "max_output_tokens"}));
        let status = if incomplete.is_some() {
            "incomplete"
        } else {
            "completed"
        };
        let event_type = if incomplete.is_some() {
            "response.incomplete"
        } else {
            "response.completed"
        };
        events.push(event(
            event_type,
            json!({"response": self.response(status, Some((&finished.usage, incomplete)))}),
        ));
        events
    }

    fn response(&self, status: &str, terminal: Option<(&ChatUsage, Option<Value>)>) -> Value {
        let (usage, incomplete_details) =
            terminal.map_or((Value::Null, Value::Null), |(usage, incomplete)| {
                (
                    json!({
                        "input_tokens": usage.prompt_tokens,
                        "input_tokens_details": {"cached_tokens": usage.cached_prompt_tokens},
                        "output_tokens": usage.completion_tokens,
                        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
                        "total_tokens": usage.total_tokens()
                    }),
                    incomplete.unwrap_or(Value::Null),
                )
            });
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": null,
            "incomplete_details": incomplete_details,
            "instructions": null,
            "max_output_tokens": null,
            "model": self.model,
            "output": self.output,
            "parallel_tool_calls": false,
            "previous_response_id": null,
            "reasoning": null,
            "store": false,
            "temperature": null,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": null,
            "truncation": "disabled",
            "usage": usage,
            "user": null,
            "metadata": {}
        })
    }
}

fn event(kind: &str, fields: Value) -> Value {
    let mut object = match fields {
        Value::Object(object) => object,
        _ => Map::new(),
    };
    object.insert("type".to_string(), Value::String(kind.to_string()));
    Value::Object(object)
}

fn message_item(id: &str, status: &str, text: &str) -> Value {
    let content = if status == "in_progress" {
        Vec::new()
    } else {
        vec![output_text(text)]
    };
    json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content
    })
}

fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
}

fn reasoning_item(id: &str, status: &str, text: &str) -> Value {
    let summary = if status == "in_progress" {
        Vec::new()
    } else {
        vec![reasoning_summary(text)]
    };
    json!({
        "id": id,
        "type": "reasoning",
        "status": status,
        "summary": summary,
        "content": [],
        "encrypted_content": null
    })
}

fn reasoning_summary(text: &str) -> Value {
    json!({"type": "summary_text", "text": text})
}

fn function_item(id: &str, call_id: &str, name: &str, arguments: &str, status: &str) -> Value {
    json!({
        "id": id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments
    })
}

fn next_id(prefix: &str, counter: &AtomicU64) -> String {
    format!("{prefix}_{:016x}", counter.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 64,
            ..GenerationConfig::default()
        }
    }

    #[test]
    fn omitted_max_output_tokens_uses_api_default() {
        let request: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": "hello"
        }))
        .unwrap();
        let chat = request.into_chat_request(&defaults()).unwrap();
        assert_eq!(chat.generation.max_new_tokens, DEFAULT_MAX_OUTPUT_TOKENS);
        assert!(!chat.template.enable_thinking);
    }

    #[test]
    fn explicit_sampling_overrides_greedy_server_defaults() {
        let mut defaults = defaults();
        defaults.sampling.temperature = 0.0;
        let request: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": "hello",
            "top_p": 0.8
        }))
        .unwrap();
        let chat = request.into_chat_request(&defaults).unwrap();
        assert_eq!(chat.generation.sampling.temperature, 1.0);
        assert_eq!(chat.generation.sampling.top_p, 0.8);
    }

    #[test]
    fn omitted_sampling_preserves_greedy_server_defaults() {
        let mut defaults = defaults();
        defaults.sampling.temperature = 0.0;
        let request: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": "hello"
        }))
        .unwrap();
        let chat = request.into_chat_request(&defaults).unwrap();
        assert_eq!(chat.generation.sampling.temperature, 0.0);
    }

    #[test]
    fn sampled_server_defaults_are_preserved_and_individually_overridden() {
        let mut defaults = defaults();
        defaults.sampling.temperature = 0.7;
        defaults.sampling.top_k = 20;
        defaults.sampling.top_p = 0.95;
        let omitted: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": "hello"
        }))
        .unwrap();
        let omitted = omitted.into_chat_request(&defaults).unwrap();
        assert_eq!(omitted.generation.sampling.temperature, 0.7);
        assert_eq!(omitted.generation.sampling.top_k, 20);
        assert_eq!(omitted.generation.sampling.top_p, 0.95);

        let overridden: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": "hello",
            "top_p": 0.8
        }))
        .unwrap();
        let overridden = overridden.into_chat_request(&defaults).unwrap();
        assert_eq!(overridden.generation.sampling.temperature, 0.7);
        assert_eq!(overridden.generation.sampling.top_k, 20);
        assert_eq!(overridden.generation.sampling.top_p, 0.8);
    }

    #[test]
    fn codex_request_maps_messages_functions_and_ignores_builtin_tools() {
        let request: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "instructions": "be concise",
            "input": [
                {"role":"developer","content":"rules"},
                {"role":"user","content":[{"type":"input_text","text":"run it"}]},
                {"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
                {"type":"function_call_output","call_id":"call_1","output":"/tmp"}
            ],
            "tools": [
                {"type":"function","name":"exec_command","description":"run","parameters":{"type":"object"}},
                {"type":"namespace","name":"mcp"},
                {"type":"web_search","external_web_access":false}
            ],
            "parallel_tool_calls": false,
            "temperature": 0.7,
            "top_k": 12,
            "top_p": 0.8,
            "seed": 42,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "reasoning": {"effort":"low","summary":"auto"},
            "max_output_tokens": 12
        })).unwrap();
        let chat = request.into_chat_request(&defaults()).unwrap();
        assert_eq!(chat.messages.len(), 4);
        assert_eq!(chat.tools.len(), 1);
        assert_eq!(chat.generation.max_new_tokens, 12);
        assert_eq!(chat.generation.sampling.temperature, 0.7);
        assert_eq!(chat.generation.sampling.top_k, 12);
        assert_eq!(chat.generation.sampling.top_p, 0.8);
        assert_eq!(chat.generation.sampling.seed, Some(42));
        assert_eq!(chat.generation.sampling.presence_penalty, 0.2);
        assert_eq!(chat.generation.sampling.frequency_penalty, 0.1);
        assert_eq!(
            chat.template.reasoning_effort,
            Some(ChatReasoningEffort::Low)
        );
        assert!(chat.template.enable_thinking);
        assert!(
            chat.messages[0]
                .content
                .as_deref()
                .unwrap()
                .contains("be concise\n\nrules")
        );
        assert_eq!(chat.messages[2].tool_calls[0].id, "call_1");
        assert_eq!(chat.messages[3].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn pi_responses_history_preserves_one_assistant_turn() {
        let request: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": [
                {"role":"user","content":[{"type":"input_text","text":"inspect the repository"}]},
                {
                    "type":"reasoning",
                    "summary":[
                        {"type":"summary_text","text":"I should inspect the git history and status."}
                    ]
                },
                {
                    "type":"message",
                    "role":"assistant",
                    "content":[
                        {"type":"output_text","text":"I'll inspect the repository.","annotations":[]}
                    ]
                },
                {
                    "type":"function_call",
                    "call_id":"call_1",
                    "name":"bash",
                    "arguments":"{\"command\":\"git status --short\"}"
                },
                {
                    "type":"function_call_output",
                    "call_id":"call_1",
                    "output":" M README.md"
                },
                {
                    "type":"reasoning",
                    "summary":[
                        {"type":"summary_text","text":"Now I can report the result."}
                    ]
                },
                {
                    "type":"message",
                    "role":"assistant",
                    "content":[
                        {"type":"output_text","text":"README.md is modified.","annotations":[]}
                    ]
                }
            ]
        }))
        .unwrap();

        let chat = request.into_chat_request(&defaults()).unwrap();
        assert_eq!(chat.messages.len(), 4);
        assert_eq!(
            chat.messages[0],
            ChatMessage::user("inspect the repository")
        );
        assert_eq!(
            chat.messages[1],
            ChatMessage::assistant_tool_calls(
                Some("I'll inspect the repository.".into()),
                Some("I should inspect the git history and status.".into()),
                vec![ChatToolCall {
                    id: "call_1".into(),
                    function: ChatFunctionCall {
                        name: "bash".into(),
                        arguments: BTreeMap::from([(
                            "command".into(),
                            json!("git status --short")
                        )]),
                    },
                }],
            )
        );
        assert_eq!(
            chat.messages[2],
            ChatMessage::tool("call_1", " M README.md")
        );
        assert_eq!(
            chat.messages[3],
            ChatMessage::assistant_tool_calls(
                Some("README.md is modified.".into()),
                Some("Now I can report the result.".into()),
                Vec::new(),
            )
        );
        assert!(chat.template.preserve_thinking);
    }

    #[test]
    fn reasoning_effort_none_disables_checkpoint_thinking() {
        let request: ResponseRequest = serde_json::from_value(json!({
            "model": "eider",
            "input": "answer directly",
            "reasoning": {"effort": "none"}
        }))
        .unwrap();

        let chat = request.into_chat_request(&defaults()).unwrap();
        assert!(!chat.template.enable_thinking);
        assert_eq!(chat.template.reasoning_effort, None);
    }

    #[test]
    fn response_events_follow_text_and_tool_lifecycle() {
        let mut stream = ResponseStream::new("eider");
        assert_eq!(stream.created()["type"], "response.created");
        let text = stream.push(InferenceEvent::Output(ChatOutputEvent::Text("hi".into())));
        assert_eq!(text.len(), 3);
        assert_eq!(text[2]["type"], "response.output_text.delta");
        let tool = stream.push(InferenceEvent::Output(ChatOutputEvent::ToolCall(
            ChatToolCall {
                id: "call_7".into(),
                function: ChatFunctionCall {
                    name: "exec_command".into(),
                    arguments: BTreeMap::from([("cmd".into(), json!("pwd"))]),
                },
            },
        )));
        assert_eq!(tool[0]["type"], "response.output_text.done");
        assert_eq!(tool[3]["type"], "response.output_item.added");
        let finished = InferenceFinished {
            finish_reason: ChatFinishReason::ToolCalls,
            usage: ChatUsage {
                prompt_tokens: 8,
                cached_prompt_tokens: 4,
                completion_tokens: 4,
                reasoning_tokens: 0,
            },
        };
        let done = stream.push(InferenceEvent::Finished(finished));
        assert_eq!(done.last().unwrap()["type"], "response.completed");
        assert_eq!(
            done.last().unwrap()["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            4
        );
        assert_eq!(
            done.last().unwrap()["response"]["output"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn response_events_stream_reasoning_before_visible_text() {
        let mut stream = ResponseStream::new("eider");
        let started = stream.push(InferenceEvent::Output(ChatOutputEvent::Reasoning(
            "checking".into(),
        )));
        assert_eq!(started.len(), 3);
        assert_eq!(started[0]["type"], "response.output_item.added");
        assert_eq!(started[0]["item"]["type"], "reasoning");
        assert_eq!(started[1]["type"], "response.reasoning_summary_part.added");
        assert_eq!(started[2]["type"], "response.reasoning_summary_text.delta");

        let text = stream.push(InferenceEvent::Output(ChatOutputEvent::Text("done".into())));
        assert_eq!(text[0]["type"], "response.reasoning_summary_text.done");
        assert_eq!(text[1]["type"], "response.reasoning_summary_part.done");
        assert_eq!(text[2]["type"], "response.output_item.done");
        assert_eq!(text[3]["type"], "response.output_item.added");
        assert_eq!(text.last().unwrap()["type"], "response.output_text.delta");

        let done = stream.push(InferenceEvent::Finished(InferenceFinished {
            finish_reason: ChatFinishReason::Length,
            usage: ChatUsage {
                prompt_tokens: 2,
                cached_prompt_tokens: 0,
                completion_tokens: 5,
                reasoning_tokens: 3,
            },
        }));
        let response = &done.last().unwrap()["response"];
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert_eq!(response["output"][0]["summary"][0]["text"], "checking");
        assert_eq!(
            response["usage"]["output_tokens_details"]["reasoning_tokens"],
            3
        );
    }
}
