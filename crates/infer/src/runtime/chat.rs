//! Checkpoint-driven chat prompt rendering and tokenization.

use minijinja::{Environment, Error as TemplateError, ErrorKind, UndefinedBehavior, context};
use nvfp4::{Error, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const TEMPLATE_NAME: &str = "chat_template";

/// Role attached to one structured chat message.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// Instructions that apply to the whole conversation.
    System,
    /// Input supplied by the user.
    User,
    /// Text or tool calls previously produced by the model.
    Assistant,
    /// Result returned by a tool implementation.
    Tool,
}

/// One function invocation retained in conversation history.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatToolCall {
    /// Frontend identity used to associate a later tool result.
    pub id: String,
    /// Function invocation rendered into the checkpoint prompt format.
    pub function: ChatFunctionCall,
}

/// Function name and structured arguments for a tool call.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatFunctionCall {
    /// Function name advertised to the model.
    pub name: String,
    /// Parsed argument values keyed by parameter name.
    pub arguments: BTreeMap<String, Value>,
}

/// One structured text-model conversation message.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatMessage {
    /// Message author.
    pub role: ChatRole,
    /// Visible message text. Tool-only assistant messages may omit it.
    pub content: Option<String>,
    /// Model reasoning retained separately from visible assistant text.
    pub reasoning_content: Option<String>,
    /// Function calls produced by an assistant message.
    pub tool_calls: Vec<ChatToolCall>,
    /// Tool-call identity associated with a tool response.
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(ChatRole::System, content)
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(ChatRole::User, content)
    }

    /// Creates an assistant text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(ChatRole::Assistant, content)
    }

    /// Creates a tool response associated with an earlier tool call.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Creates an assistant message containing structured function calls.
    pub fn assistant_tool_calls(
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ChatToolCall>,
    ) -> Self {
        Self {
            role: ChatRole::Assistant,
            content,
            reasoning_content,
            tool_calls,
            tool_call_id: None,
        }
    }

    fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// JSON-schema function definition exposed to the model.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatFunctionDefinition {
    /// Function name used in generated calls.
    pub name: String,
    /// Human-readable behaviour presented to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing accepted function arguments.
    pub parameters: Value,
}

/// Tool definition accepted by the text-model chat template.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatTool {
    /// Function metadata and argument schema.
    pub function: ChatFunctionDefinition,
    #[serde(rename = "type")]
    kind: ChatToolKind,
}

impl ChatTool {
    /// Creates a function tool definition.
    pub fn function(function: ChatFunctionDefinition) -> Self {
        Self {
            function,
            kind: ChatToolKind::Function,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ChatToolKind {
    Function,
}

/// Rendering controls exposed by the Qwen checkpoint template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    /// Append the assistant prefix used to begin generation.
    pub add_generation_prompt: bool,
    /// Allow the model to produce a thinking section.
    pub enable_thinking: bool,
    /// Preserve reasoning from assistant messages before the latest user turn.
    pub preserve_thinking: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_thinking: false,
        }
    }
}

/// Rendered checkpoint prompt and its exact token IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizedChatPrompt {
    /// Fully rendered checkpoint prompt before tokenization.
    pub text: String,
    /// Prompt token IDs produced without an additional tokenizer post-processor.
    pub token_ids: Vec<u32>,
}

/// Checkpoint-owned chat template paired with its tokenizer.
pub struct CheckpointChatTemplate {
    environment: Environment<'static>,
    tokenizer: Tokenizer,
    template_path: PathBuf,
}

impl CheckpointChatTemplate {
    /// Loads `chat_template.jinja` and `tokenizer.json` from a model directory.
    ///
    /// Older checkpoints that only embed `chat_template` in
    /// `tokenizer_config.json` are supported as a fallback.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let (source, template_path) = load_template_source(model_dir)?;
        let environment = build_environment(source)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| Error::Format {
            label: "tokenizer.json",
            detail: format!("{}: {error}", tokenizer_path.display()),
        })?;
        Ok(Self {
            environment,
            tokenizer,
            template_path,
        })
    }

    /// Returns the checkpoint file supplying the active template.
    pub fn template_path(&self) -> &Path {
        &self.template_path
    }

    /// Returns the tokenizer paired with the checkpoint template.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Renders structured messages and tool definitions with the checkpoint template.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        validate_chat(messages, tools)?;
        render_with_environment(&self.environment, messages, tools, options)
    }

    /// Renders and tokenizes a chat prompt without adding another special-token layer.
    pub fn render_and_tokenize(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatTool],
        options: ChatTemplateOptions,
    ) -> Result<TokenizedChatPrompt> {
        let text = self.render(messages, tools, options)?;
        let encoding = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|error| Error::Format {
                label: "chat prompt tokenization",
                detail: error.to_string(),
            })?;
        Ok(TokenizedChatPrompt {
            text,
            token_ids: encoding.get_ids().to_vec(),
        })
    }
}

fn validate_chat(messages: &[ChatMessage], tools: &[ChatTool]) -> Result<()> {
    if messages.is_empty() {
        return Err(Error::Format {
            label: "chat messages",
            detail: "at least one message is required".to_string(),
        });
    }
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System && index != 0 {
            return Err(Error::Format {
                label: "chat messages",
                detail: format!("system message at index {index}; it must be first"),
            });
        }
        if message.role == ChatRole::Tool && message.tool_call_id.is_none() {
            return Err(Error::Format {
                label: "chat tool response",
                detail: format!("tool message at index {index} has no tool_call_id"),
            });
        }
        if message.role != ChatRole::Assistant && !message.tool_calls.is_empty() {
            return Err(Error::Format {
                label: "chat tool calls",
                detail: format!("non-assistant message at index {index} contains tool calls"),
            });
        }
        for call in &message.tool_calls {
            validate_name("chat function call", &call.function.name)?;
            if call.id.is_empty() {
                return Err(Error::Format {
                    label: "chat function call",
                    detail: "tool call IDs must not be empty".to_string(),
                });
            }
        }
    }
    for tool in tools {
        validate_name("chat tool definition", &tool.function.name)?;
        if !tool.function.parameters.is_object() {
            return Err(Error::Format {
                label: "chat tool definition",
                detail: format!(
                    "parameters for {:?} must be a JSON object",
                    tool.function.name
                ),
            });
        }
    }
    Ok(())
}

fn validate_name(label: &'static str, name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(Error::Format {
            label,
            detail: format!("invalid function name {name:?}"),
        });
    }
    Ok(())
}

fn load_template_source(model_dir: &Path) -> Result<(String, PathBuf)> {
    let standalone = model_dir.join("chat_template.jinja");
    if standalone.is_file() {
        let source = std::fs::read_to_string(&standalone).map_err(|error| Error::Format {
            label: "chat_template.jinja",
            detail: format!("{}: {error}", standalone.display()),
        })?;
        return Ok((source, standalone));
    }

    let config_path = model_dir.join("tokenizer_config.json");
    let contents = std::fs::read_to_string(&config_path).map_err(|error| Error::Format {
        label: "tokenizer_config.json",
        detail: format!("{}: {error}", config_path.display()),
    })?;
    let config: Value = serde_json::from_str(&contents).map_err(|error| Error::Format {
        label: "tokenizer_config.json",
        detail: error.to_string(),
    })?;
    let source = config["chat_template"]
        .as_str()
        .ok_or_else(|| Error::Format {
            label: "tokenizer_config.json",
            detail: "missing string chat_template".to_string(),
        })?
        .to_string();
    Ok((source, config_path))
}

fn build_environment(source: String) -> Result<Environment<'static>> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, TemplateError> {
            Err(TemplateError::new(ErrorKind::InvalidOperation, message))
        },
    );
    environment
        .add_template_owned(TEMPLATE_NAME, source)
        .map_err(template_error)?;
    Ok(environment)
}

fn render_with_environment(
    environment: &Environment<'_>,
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: ChatTemplateOptions,
) -> Result<String> {
    environment
        .get_template(TEMPLATE_NAME)
        .map_err(template_error)?
        .render(context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => options.add_generation_prompt,
            enable_thinking => options.enable_thinking,
            preserve_thinking => options.preserve_thinking,
            add_vision_id => false,
        })
        .map_err(template_error)
}

fn template_error(error: TemplateError) -> Error {
    Error::Format {
        label: "chat template",
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_definition() -> ChatTool {
        ChatTool::function(ChatFunctionDefinition {
            name: "read_file".to_string(),
            description: Some("Read a local file".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        })
    }

    #[test]
    fn structured_values_reach_checkpoint_template_without_shape_adapters() {
        let source = concat!(
            "{{ messages[0].role }}:{{ messages[0].content }}|",
            "{{ tools[0] | tojson }}|",
            "{{ add_generation_prompt }}:{{ enable_thinking }}:{{ preserve_thinking }}"
        );
        let environment = build_environment(source.to_string()).unwrap();
        let rendered = render_with_environment(
            &environment,
            &[ChatMessage::user("hello")],
            &[tool_definition()],
            ChatTemplateOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                preserve_thinking: true,
            },
        )
        .unwrap();
        assert!(rendered.starts_with("user:hello|"));
        assert!(rendered.contains(r#""name":"read_file""#));
        assert!(rendered.ends_with("|true:false:true"));
    }

    #[test]
    fn validation_rejects_misplaced_system_and_invalid_tools() {
        let messages = [ChatMessage::user("hello"), ChatMessage::system("late")];
        assert!(validate_chat(&messages, &[]).is_err());

        let mut tool = tool_definition();
        tool.function.name = "bad name".to_string();
        assert!(validate_chat(&[ChatMessage::user("hello")], &[tool]).is_err());
    }

    #[test]
    #[ignore = "requires the local Qwen3.6 checkpoint"]
    fn local_qwen36_template_renders_tools_history_and_generation_prefix() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/qwen3.6-35b-a3-nvfp4");
        let template = CheckpointChatTemplate::from_model_dir(model_dir).unwrap();
        let call = ChatToolCall {
            id: "call_1".to_string(),
            function: ChatFunctionCall {
                name: "read_file".to_string(),
                arguments: BTreeMap::from([("path".to_string(), json!("src/main.rs"))]),
            },
        };
        let prompt = template
            .render_and_tokenize(
                &[
                    ChatMessage::system("Be precise."),
                    ChatMessage::user("Inspect the entry point."),
                    ChatMessage::assistant_tool_calls(None, None, vec![call]),
                    ChatMessage::tool("call_1", "fn main() {}"),
                ],
                &[tool_definition()],
                ChatTemplateOptions {
                    enable_thinking: false,
                    ..ChatTemplateOptions::default()
                },
            )
            .unwrap();

        assert!(prompt.text.contains("# Tools"));
        assert!(prompt.text.contains("<function=read_file>"));
        assert!(
            prompt
                .text
                .contains("<tool_response>\nfn main() {}\n</tool_response>")
        );
        assert!(
            prompt
                .text
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
        assert!(!prompt.token_ids.is_empty());
    }
}
