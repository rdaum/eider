//! Chat Completions request translation and response construction.

use crate::protocol::{ApiError, InferenceEvent, InferenceFinished, OneOrMany};
use infer::runtime::chat::{
    ChatFunctionCall, ChatFunctionDefinition, ChatMessage, ChatReasoningEffort, ChatRole,
    ChatTemplateOptions, ChatTool, ChatToolCall,
};
use infer::runtime::generation::GenerationConfig;
use infer::runtime::sampling::SamplingConfig;
use infer::runtime::scheduler::RequestConfig;
use infer::runtime::serving::{ChatFinishReason, ChatRequest, ChatUsage};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_COMPLETION_TOKENS: usize = 4096;

static NEXT_CHAT_COMPLETION_ID: AtomicU64 = AtomicU64::new(1);

/// JSON body accepted by `POST /v1/chat/completions`.
#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<ChatStreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
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
    pub stop: Option<OneOrMany<String>>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
pub struct ChatStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

impl ChatCompletionRequest {
    /// Converts a Chat Completions request into the shared runtime contract.
    pub fn into_chat_request(self, defaults: &GenerationConfig) -> Result<ChatRequest, ApiError> {
        if self.parallel_tool_calls == Some(true) {
            return Err(ApiError::invalid(
                "parallel_tool_calls",
                "parallel tool calls are not supported",
            ));
        }
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::invalid("n", "only n=1 is supported"));
        }
        if self.logprobs == Some(true) || self.top_logprobs.is_some() {
            return Err(ApiError::invalid(
                "logprobs",
                "token log probabilities are not supported",
            ));
        }
        let expose_tools = validate_tool_choice(self.tool_choice.as_ref())?;
        let mut messages = self
            .messages
            .iter()
            .map(parse_message)
            .collect::<Result<Vec<_>, _>>()?;
        messages = coalesce_system_messages(messages);
        if messages.is_empty() {
            return Err(ApiError::invalid("messages", "messages must not be empty"));
        }
        let tools = if expose_tools {
            self.tools
                .iter()
                .map(parse_tool)
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
            max_new_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS),
            eos_token_ids: defaults.eos_token_ids.clone(),
        };
        generation
            .validate()
            .map_err(|error| ApiError::invalid("sampling", error.to_string()))?;

        let enable_thinking =
            self.reasoning_effort.as_deref() != Some("none") && self.reasoning_effort.is_some();
        let reasoning_effort = self
            .reasoning_effort
            .as_deref()
            .filter(|effort| *effort != "none")
            .map(parse_reasoning_effort)
            .transpose()?;

        Ok(ChatRequest {
            messages,
            tools,
            template: ChatTemplateOptions {
                enable_thinking,
                reasoning_effort,
                ..ChatTemplateOptions::default()
            },
            generation,
            stop_sequences: self.stop.map(OneOrMany::into_vec).unwrap_or_default(),
        })
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .is_some_and(|options| options.include_usage)
    }
}

fn parse_message(value: &Value) -> Result<ChatMessage, ApiError> {
    let role = required_string(value, "role", "messages")?;
    let content = parse_content(value.get("content"))?;
    match role.as_str() {
        "developer" | "system" => Ok(ChatMessage::system(content.unwrap_or_default())),
        "user" => Ok(ChatMessage::user(content.unwrap_or_default())),
        "tool" => Ok(ChatMessage::tool(
            required_string(value, "tool_call_id", "messages")?,
            content.unwrap_or_default(),
        )),
        "assistant" => {
            let reasoning_content = value
                .get("reasoning_content")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let tool_calls = value
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(|calls| calls.iter().map(parse_tool_call).collect())
                .transpose()?
                .unwrap_or_default();
            Ok(ChatMessage::assistant_tool_calls(
                content,
                reasoning_content,
                tool_calls,
            ))
        }
        other => Err(ApiError::invalid(
            "messages",
            format!("unsupported message role {other:?}"),
        )),
    }
}

fn parse_content(value: Option<&Value>) -> Result<Option<String>, ApiError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                let kind = required_string(part, "type", "messages")?;
                match kind.as_str() {
                    "text" | "input_text" | "output_text" => {
                        text.push_str(&required_string(part, "text", "messages")?);
                    }
                    other => {
                        return Err(ApiError::invalid(
                            "messages",
                            format!("unsupported message content type {other:?}"),
                        ));
                    }
                }
            }
            Ok(Some(text))
        }
        Some(_) => Err(ApiError::invalid(
            "messages",
            "message content must be text or an array of text parts",
        )),
    }
}

fn parse_tool(value: &Value) -> Result<ChatTool, ApiError> {
    if required_string(value, "type", "tools")? != "function" {
        return Err(ApiError::invalid(
            "tools",
            "only function tools are supported",
        ));
    }
    let function = value
        .get("function")
        .ok_or_else(|| ApiError::invalid("tools", "function tool is missing function"))?;
    let name = required_string(function, "name", "tools")?;
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type":"object","properties":{}}));
    if !parameters.is_object() {
        return Err(ApiError::invalid(
            "tools",
            format!("function {name:?} parameters must be a JSON Schema object"),
        ));
    }
    Ok(ChatTool::function(ChatFunctionDefinition {
        name,
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        parameters,
    }))
}

fn parse_tool_call(value: &Value) -> Result<ChatToolCall, ApiError> {
    if required_string(value, "type", "messages")? != "function" {
        return Err(ApiError::invalid(
            "messages",
            "only function tool calls are supported",
        ));
    }
    let function = value
        .get("function")
        .ok_or_else(|| ApiError::invalid("messages", "tool call is missing function"))?;
    let name = required_string(function, "name", "messages")?;
    let arguments = required_string(function, "arguments", "messages")?;
    let arguments =
        serde_json::from_str::<BTreeMap<String, Value>>(&arguments).map_err(|error| {
            ApiError::invalid(
                "messages",
                format!("function call {name:?} has invalid JSON arguments: {error}"),
            )
        })?;
    Ok(ChatToolCall {
        id: required_string(value, "id", "messages")?,
        function: ChatFunctionCall { name, arguments },
    })
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

fn parse_reasoning_effort(value: &str) -> Result<ChatReasoningEffort, ApiError> {
    match value {
        "low" => Ok(ChatReasoningEffort::Low),
        "medium" => Ok(ChatReasoningEffort::Medium),
        "high" => Ok(ChatReasoningEffort::High),
        _ => Err(ApiError::invalid(
            "reasoning_effort",
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

fn required_string(value: &Value, key: &str, param: &str) -> Result<String, ApiError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ApiError::invalid(param, format!("missing string field {key:?}")))
}

/// Chat Completions streaming event builder and non-streaming accumulator.
pub struct ChatCompletionStream {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    content: String,
    reasoning_content: String,
    tool_calls: Vec<ChatToolCall>,
    finished: Option<InferenceFinished>,
    error: Option<String>,
}

impl ChatCompletionStream {
    pub fn new(model: impl Into<String>, include_usage: bool) -> Self {
        Self {
            id: format!(
                "chatcmpl-{:016x}",
                NEXT_CHAT_COMPLETION_ID.fetch_add(1, Ordering::Relaxed)
            ),
            model: model.into(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            include_usage,
            content: String::new(),
            reasoning_content: String::new(),
            tool_calls: Vec::new(),
            finished: None,
            error: None,
        }
    }

    pub fn initial_chunk(&self) -> Value {
        self.chunk(
            json!({"role":"assistant","content":""}),
            Value::Null,
            Value::Null,
        )
    }

    pub fn push(&mut self, inference: InferenceEvent) -> Vec<Value> {
        match inference {
            InferenceEvent::Output(infer::runtime::chat_output::ChatOutputEvent::Reasoning(
                delta,
            )) => {
                if delta.is_empty() {
                    Vec::new()
                } else {
                    self.reasoning_content.push_str(&delta);
                    vec![self.chunk(json!({"reasoning_content":delta}), Value::Null, Value::Null)]
                }
            }
            InferenceEvent::Output(infer::runtime::chat_output::ChatOutputEvent::Text(delta)) => {
                if delta.is_empty() {
                    Vec::new()
                } else {
                    self.content.push_str(&delta);
                    vec![self.chunk(json!({"content":delta}), Value::Null, Value::Null)]
                }
            }
            InferenceEvent::Output(infer::runtime::chat_output::ChatOutputEvent::ToolCall(
                call,
            )) => {
                let index = self.tool_calls.len();
                let tool_call = chat_tool_call(&call);
                self.tool_calls.push(call);
                vec![self.chunk(
                    json!({"tool_calls":[{
                        "index":index,
                        "id":tool_call["id"],
                        "type":"function",
                        "function":tool_call["function"]
                    }]}),
                    Value::Null,
                    Value::Null,
                )]
            }
            InferenceEvent::Finished(finished) => {
                let finish_reason = finish_reason(&finished.finish_reason);
                let usage = chat_usage(&finished.usage);
                self.finished = Some(finished);
                let mut chunks = vec![self.chunk(json!({}), json!(finish_reason), Value::Null)];
                if self.include_usage {
                    chunks.push(self.usage_chunk(usage));
                }
                chunks
            }
            InferenceEvent::Error(message) => {
                self.error = Some(message.clone());
                vec![json!({
                    "error": {
                        "message": message,
                        "type": "server_error",
                        "param": null,
                        "code": "inference_error"
                    }
                })]
            }
        }
    }

    pub fn is_completed(&self) -> bool {
        self.finished.is_some() || self.error.is_some()
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn response(&self) -> Option<Value> {
        let finished = self.finished.as_ref()?;
        let mut message = Map::from_iter([
            ("role".to_string(), json!("assistant")),
            (
                "content".to_string(),
                if self.content.is_empty() && !self.tool_calls.is_empty() {
                    Value::Null
                } else {
                    json!(self.content)
                },
            ),
            ("refusal".to_string(), Value::Null),
        ]);
        if !self.reasoning_content.is_empty() {
            message.insert(
                "reasoning_content".to_string(),
                json!(self.reasoning_content),
            );
        }
        if !self.tool_calls.is_empty() {
            message.insert(
                "tool_calls".to_string(),
                Value::Array(self.tool_calls.iter().map(chat_tool_call).collect()),
            );
        }
        Some(json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": null,
                "finish_reason": finish_reason(&finished.finish_reason)
            }],
            "usage": chat_usage(&finished.usage),
            "system_fingerprint": null
        }))
    }

    fn chunk(&self, delta: Value, finish_reason: Value, usage: Value) -> Value {
        let mut chunk = Map::from_iter([
            ("id".to_string(), json!(self.id)),
            ("object".to_string(), json!("chat.completion.chunk")),
            ("created".to_string(), json!(self.created)),
            ("model".to_string(), json!(self.model)),
            (
                "choices".to_string(),
                json!([{
                    "index": 0,
                    "delta": delta,
                    "logprobs": null,
                    "finish_reason": finish_reason
                }]),
            ),
            ("system_fingerprint".to_string(), Value::Null),
        ]);
        if self.include_usage {
            chunk.insert("usage".to_string(), usage);
        }
        Value::Object(chunk)
    }

    fn usage_chunk(&self, usage: Value) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage,
            "system_fingerprint": null
        })
    }
}

fn chat_tool_call(call: &ChatToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": serde_json::to_string(&call.function.arguments)
                .expect("tool arguments are serializable")
        }
    })
}

fn finish_reason(reason: &ChatFinishReason) -> &'static str {
    match reason {
        ChatFinishReason::Eos | ChatFinishReason::Stop(_) => "stop",
        ChatFinishReason::Length => "length",
        ChatFinishReason::ToolCalls => "tool_calls",
    }
}

fn chat_usage(usage: &ChatUsage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens(),
        "prompt_tokens_details": {"cached_tokens": usage.cached_prompt_tokens},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens}
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer::runtime::chat_output::ChatOutputEvent;

    fn defaults() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 64,
            ..GenerationConfig::default()
        }
    }

    #[test]
    fn request_maps_messages_tools_and_sampling() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[
                {"role":"system","content":"be concise"},
                {"role":"user","content":"run pwd"},
                {"role":"assistant","content":null,"reasoning_content":"checking","tool_calls":[{
                    "id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"pwd\"}"}
                }]},
                {"role":"tool","tool_call_id":"call_1","content":"/tmp"}
            ],
            "tools":[{"type":"function","function":{
                "name":"bash","description":"run a command","parameters":{"type":"object"}
            }}],
            "max_completion_tokens":12,
            "temperature":0.7,
            "top_k":16,
            "top_p":0.8,
            "seed":42,
            "reasoning_effort":"low"
        })).expect("request");
        let chat = request
            .into_chat_request(&defaults())
            .expect("chat request");
        assert_eq!(chat.messages.len(), 4);
        assert_eq!(chat.messages[2].tool_calls[0].id, "call_1");
        assert_eq!(chat.messages[3].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.tools.len(), 1);
        assert_eq!(chat.generation.max_new_tokens, 12);
        assert_eq!(chat.generation.sampling.temperature, 0.7);
        assert_eq!(chat.generation.sampling.top_k, 16);
        assert_eq!(chat.generation.sampling.top_p, 0.8);
        assert_eq!(chat.generation.sampling.seed, Some(42));
        assert_eq!(
            chat.template.reasoning_effort,
            Some(ChatReasoningEffort::Low)
        );
        assert!(chat.template.enable_thinking);
    }

    #[test]
    fn sampling_overrides_greedy_server_defaults_only_when_present() {
        let mut defaults = defaults();
        defaults.sampling.temperature = 0.0;
        let omitted: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[{"role":"user","content":"hello"}]
        }))
        .expect("request without sampling");
        let omitted = omitted
            .into_chat_request(&defaults)
            .expect("chat request without sampling");
        assert_eq!(omitted.generation.sampling.temperature, 0.0);

        let overridden: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[{"role":"user","content":"hello"}],
            "top_p":0.8
        }))
        .expect("request with sampling");
        let overridden = overridden
            .into_chat_request(&defaults)
            .expect("chat request with sampling");
        assert_eq!(overridden.generation.sampling.temperature, 1.0);
        assert_eq!(overridden.generation.sampling.top_p, 0.8);
    }

    #[test]
    fn sampled_server_defaults_are_preserved_and_individually_overridden() {
        let mut defaults = defaults();
        defaults.sampling.temperature = 0.7;
        defaults.sampling.top_k = 20;
        defaults.sampling.top_p = 0.95;
        let omitted: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[{"role":"user","content":"hello"}]
        }))
        .expect("request without sampling");
        let omitted = omitted
            .into_chat_request(&defaults)
            .expect("chat request without sampling");
        assert_eq!(omitted.generation.sampling.temperature, 0.7);
        assert_eq!(omitted.generation.sampling.top_k, 20);
        assert_eq!(omitted.generation.sampling.top_p, 0.95);

        let overridden: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[{"role":"user","content":"hello"}],
            "top_p":0.8
        }))
        .expect("request with sampling");
        let overridden = overridden
            .into_chat_request(&defaults)
            .expect("chat request with sampling");
        assert_eq!(overridden.generation.sampling.temperature, 0.7);
        assert_eq!(overridden.generation.sampling.top_k, 20);
        assert_eq!(overridden.generation.sampling.top_p, 0.8);
    }

    #[test]
    fn request_accepts_legacy_token_limit_and_none_tool_choice() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],
            "tools":[{"type":"function","function":{"name":"unused"}}],
            "tool_choice":"none",
            "max_tokens":17,
            "stop":["END","STOP"]
        }))
        .expect("request");
        let chat = request
            .into_chat_request(&defaults())
            .expect("chat request");
        assert_eq!(chat.messages[0].content.as_deref(), Some("hello"));
        assert!(chat.tools.is_empty());
        assert_eq!(chat.generation.max_new_tokens, 17);
        assert_eq!(chat.stop_sequences, ["END", "STOP"]);
        assert!(!chat.template.enable_thinking);
    }

    #[test]
    fn request_rejects_unsupported_parallel_choices() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model":"eider",
            "messages":[{"role":"user","content":"hello"}],
            "parallel_tool_calls":true
        }))
        .expect("request");
        let error = request
            .into_chat_request(&defaults())
            .expect_err("parallel calls must fail");
        assert_eq!(error.param.as_deref(), Some("parallel_tool_calls"));
    }

    #[test]
    fn non_streaming_response_preserves_reasoning_text_tools_and_usage() {
        let mut stream = ChatCompletionStream::new("eider", false);
        stream.push(InferenceEvent::Output(ChatOutputEvent::Reasoning(
            "checking".into(),
        )));
        stream.push(InferenceEvent::Output(ChatOutputEvent::Text("done".into())));
        stream.push(InferenceEvent::Output(ChatOutputEvent::ToolCall(
            ChatToolCall {
                id: "call_7".into(),
                function: ChatFunctionCall {
                    name: "bash".into(),
                    arguments: BTreeMap::from([("command".into(), json!("pwd"))]),
                },
            },
        )));
        stream.push(InferenceEvent::Finished(InferenceFinished {
            finish_reason: ChatFinishReason::ToolCalls,
            usage: ChatUsage {
                prompt_tokens: 8,
                cached_prompt_tokens: 4,
                completion_tokens: 5,
                reasoning_tokens: 2,
            },
        }));
        let response = stream.response().expect("response");
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["choices"][0]["message"]["content"], "done");
        assert_eq!(
            response["choices"][0]["message"]["reasoning_content"],
            "checking"
        );
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_7"
        );
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            response["usage"]["prompt_tokens_details"]["cached_tokens"],
            4
        );
        assert_eq!(
            response["usage"]["completion_tokens_details"]["reasoning_tokens"],
            2
        );
    }

    #[test]
    fn streaming_response_emits_role_deltas_finish_usage_and_done_state() {
        let mut stream = ChatCompletionStream::new("eider", true);
        let initial = stream.initial_chunk();
        assert_eq!(initial["choices"][0]["delta"]["role"], "assistant");
        assert!(initial["usage"].is_null());
        let text = stream.push(InferenceEvent::Output(ChatOutputEvent::Text("hi".into())));
        assert_eq!(text[0]["choices"][0]["delta"]["content"], "hi");
        let done = stream.push(InferenceEvent::Finished(InferenceFinished {
            finish_reason: ChatFinishReason::Length,
            usage: ChatUsage {
                prompt_tokens: 3,
                cached_prompt_tokens: 0,
                completion_tokens: 2,
                reasoning_tokens: 0,
            },
        }));
        assert_eq!(done.len(), 2);
        assert_eq!(done[0]["choices"][0]["finish_reason"], "length");
        assert_eq!(done[1]["choices"].as_array().unwrap().len(), 0);
        assert_eq!(done[1]["usage"]["completion_tokens"], 2);
        assert!(stream.is_completed());
    }

    #[test]
    fn streaming_response_omits_usage_when_not_requested() {
        let stream = ChatCompletionStream::new("eider", false);
        assert!(stream.initial_chunk().get("usage").is_none());
    }

    #[test]
    fn streaming_response_emits_chat_tool_call_delta() {
        let mut stream = ChatCompletionStream::new("eider", false);
        let chunks = stream.push(InferenceEvent::Output(ChatOutputEvent::ToolCall(
            ChatToolCall {
                id: "call_9".into(),
                function: ChatFunctionCall {
                    name: "bash".into(),
                    arguments: BTreeMap::from([("command".into(), json!("pwd"))]),
                },
            },
        )));
        let call = &chunks[0]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["id"], "call_9");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "bash");
        assert_eq!(call["function"]["arguments"], r#"{"command":"pwd"}"#);
    }
}
