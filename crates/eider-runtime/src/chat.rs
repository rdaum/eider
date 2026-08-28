//! Checkpoint-driven chat prompt rendering and tokenization.

use eider_format::{Error, Result};
use minijinja::value::Kwargs;
use minijinja::{
    Environment, Error as TemplateError, ErrorKind, UndefinedBehavior, Value as TemplateValue,
    context,
};
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

/// Step reasoning budget rendered into checkpoint system prompts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatReasoningEffort {
    Low,
    Medium,
    High,
    XHigh,
}

impl ChatReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
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
    /// Optional checkpoint-native reasoning budget.
    pub reasoning_effort: Option<ChatReasoningEffort>,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_thinking: false,
            reasoning_effort: None,
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
    bos_token: String,
    eos_token: String,
}

impl CheckpointChatTemplate {
    /// Loads `chat_template.jinja` and `tokenizer.json` from a model directory.
    ///
    /// Older checkpoints that only embed `chat_template` in
    /// `tokenizer_config.json` are supported as a fallback.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let (source, template_path) = load_template_source(model_dir)?;
        Self::from_source_and_tokenizer_files(
            source,
            template_path,
            model_dir.join("tokenizer.json"),
            model_dir.join("tokenizer_config.json"),
        )
    }

    /// Builds a checkpoint template from an explicitly supplied source and tokenizer files.
    pub fn from_source_and_tokenizer_files(
        source: String,
        template_path: PathBuf,
        tokenizer_path: PathBuf,
        tokenizer_config_path: PathBuf,
    ) -> Result<Self> {
        let environment = build_environment(source)?;
        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|error| Error::Format {
                label: "tokenizer.json",
                detail: format!("{}: {error}", tokenizer_path.display()),
            })?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| Error::Format {
                label: "tokenizer.json truncation",
                detail: error.to_string(),
            })?;
        let tokenizer_config = std::fs::read_to_string(&tokenizer_config_path)
            .ok()
            .and_then(|contents| serde_json::from_str::<Value>(&contents).ok())
            .unwrap_or(Value::Null);
        Ok(Self {
            environment,
            tokenizer,
            template_path,
            bos_token: special_token_content(&tokenizer_config["bos_token"]),
            eos_token: special_token_content(&tokenizer_config["eos_token"]),
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
        render_with_environment(
            &self.environment,
            messages,
            tools,
            options,
            &self.bos_token,
            &self.eos_token,
        )
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
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
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
    let source = normalise_generation_blocks(source);
    let mut environment = Environment::new();
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    // Hugging Face checkpoint templates use Jinja's default falsey behaviour
    // for optional message and JSON-schema members.
    environment.set_undefined_behavior(UndefinedBehavior::Lenient);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_filter("tojson", tojson_compat);
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

fn normalise_generation_blocks(mut source: String) -> String {
    for (annotation, ordinary) in [
        ("{% generation %}", "{% if true %}"),
        ("{%- generation %}", "{%- if true %}"),
        ("{% generation -%}", "{% if true -%}"),
        ("{%- generation -%}", "{%- if true -%}"),
        ("{% endgeneration %}", "{% endif %}"),
        ("{%- endgeneration %}", "{%- endif %}"),
        ("{% endgeneration -%}", "{% endif -%}"),
        ("{%- endgeneration -%}", "{%- endif -%}"),
    ] {
        source = source.replace(annotation, ordinary);
    }
    // Minijinja requires a conditional expression used as a keyword argument
    // to be parenthesized; Hugging Face Jinja accepts the unparenthesized form.
    source = source.replace(
        "namespace(name=tcid if tcid else '')",
        "namespace(name=(tcid if tcid else ''))",
    );
    source
}

fn tojson_compat(
    value: &TemplateValue,
    indent: Option<TemplateValue>,
    kwargs: Kwargs,
) -> std::result::Result<TemplateValue, TemplateError> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let ensure_ascii = ensure_ascii.unwrap_or(true);
    let indent = match indent {
        Some(indent) => Some(indent),
        None => kwargs.get("indent")?,
    };
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(value) => match bool::try_from(value.clone()).ok() {
            Some(true) => Some(2),
            Some(false) => None,
            None => Some(usize::try_from(value)?),
        },
    };
    // Hugging Face renders checkpoint templates with Jinja's `tojson` filter.
    // Its default JSON policy sorts object keys and uses Python's `", "` and
    // `": "` separators. Preserve that exact prompt text: compact serde JSON
    // measurably changes tokenization for tool-heavy prompts.
    let mut value = serde_json::to_value(value).map_err(|error| {
        TemplateError::new(ErrorKind::InvalidOperation, "cannot serialize to JSON")
            .with_source(error)
    })?;
    sort_json_object_keys(&mut value);
    let mut serialized = if let Some(indent) = indent {
        let mut output = Vec::new();
        let whitespace = " ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(whitespace.as_bytes());
        let mut serializer = serde_json::Serializer::with_formatter(&mut output, formatter);
        serde::Serialize::serialize(&value, &mut serializer)
            .map(|()| String::from_utf8(output).expect("serde_json emitted valid UTF-8"))
    } else {
        let mut output = Vec::new();
        let mut serializer =
            serde_json::Serializer::with_formatter(&mut output, JinjaJsonFormatter);
        serde::Serialize::serialize(&value, &mut serializer)
            .map(|()| String::from_utf8(output).expect("serde_json emitted valid UTF-8"))
    }
    .map_err(|error| {
        TemplateError::new(ErrorKind::InvalidOperation, "cannot serialize to JSON")
            .with_source(error)
    })?;
    if ensure_ascii {
        serialized = escape_non_ascii(&serialized);
    }
    let mut safe = String::with_capacity(serialized.len());
    for character in serialized.chars() {
        match character {
            '<' => safe.push_str("\\u003c"),
            '>' => safe.push_str("\\u003e"),
            '&' => safe.push_str("\\u0026"),
            '\'' => safe.push_str("\\u0027"),
            _ => safe.push(character),
        }
    }
    Ok(TemplateValue::from_safe_string(safe))
}

fn sort_json_object_keys(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(sort_json_object_keys),
        Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            for (_, value) in &mut entries {
                sort_json_object_keys(value);
            }
            object.extend(entries);
        }
        _ => {}
    }
}

struct JinjaJsonFormatter;

impl serde_json::ser::Formatter for JinjaJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

fn escape_non_ascii(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii() {
            escaped.push(character);
            continue;
        }
        let scalar = character as u32;
        if scalar <= 0xffff {
            use std::fmt::Write;
            write!(escaped, "\\u{scalar:04x}").expect("writing to a String cannot fail");
        } else {
            let scalar = scalar - 0x1_0000;
            let high = 0xd800 + (scalar >> 10);
            let low = 0xdc00 + (scalar & 0x3ff);
            use std::fmt::Write;
            write!(escaped, "\\u{high:04x}\\u{low:04x}").expect("writing to a String cannot fail");
        }
    }
    escaped
}

fn render_with_environment(
    environment: &Environment<'_>,
    messages: &[ChatMessage],
    tools: &[ChatTool],
    options: ChatTemplateOptions,
    bos_token: &str,
    eos_token: &str,
) -> Result<String> {
    let reasoning_effort = options
        .reasoning_effort
        .map(ChatReasoningEffort::as_str)
        .map(TemplateValue::from)
        .unwrap_or(TemplateValue::UNDEFINED);
    environment
        .get_template(TEMPLATE_NAME)
        .map_err(template_error)?
        .render(context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => options.add_generation_prompt,
            enable_thinking => options.enable_thinking,
            preserve_thinking => options.preserve_thinking,
            thinking_mode => if options.enable_thinking { "thinking" } else { "chat" },
            drop_thinking => !options.preserve_thinking,
            reasoning_effort => reasoning_effort,
            reasoning_strength => reasoning_effort,
            add_vision_id => false,
            bos_token => bos_token,
            eos_token => eos_token,
        })
        .map_err(template_error)
}

fn special_token_content(value: &Value) -> String {
    value
        .as_str()
        .or_else(|| value["content"].as_str())
        .unwrap_or_default()
        .to_string()
}

fn template_error(error: TemplateError) -> Error {
    let range = error
        .range()
        .map(|range| format!(" at bytes {}..{}", range.start, range.end))
        .unwrap_or_default();
    Error::Format {
        label: "chat template",
        detail: format!("{error}{range}"),
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
                reasoning_effort: None,
            },
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert!(rendered.starts_with("user:hello|"));
        assert!(rendered.contains(r#""name": "read_file""#));
        assert!(rendered.ends_with("|true:false:true"));
    }

    #[test]
    fn deepseek_thinking_aliases_follow_common_template_options() {
        let environment =
            build_environment("{{ thinking_mode }}:{{ drop_thinking }}".to_string()).unwrap();
        let rendered = render_with_environment(
            &environment,
            &[ChatMessage::user("hello")],
            &[],
            ChatTemplateOptions {
                enable_thinking: false,
                preserve_thinking: true,
                ..ChatTemplateOptions::default()
            },
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert_eq!(rendered, "chat:false");
    }

    #[test]
    fn tojson_matches_jinja_key_order_spacing_and_ascii_defaults() {
        let environment = build_environment("{{ tools[0] | tojson }}".to_string()).unwrap();
        let mut tool = tool_definition();
        tool.function.description = Some("Read café paths".to_string());
        let rendered = render_with_environment(
            &environment,
            &[ChatMessage::user("hello")],
            &[tool],
            ChatTemplateOptions::default(),
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert_eq!(
            rendered,
            concat!(
                r#"{"function": {"description": "Read caf\u00e9 paths", "#,
                r#""name": "read_file", "parameters": {"properties": {"path": {"type": "#,
                r#""string"}}, "required": ["path"], "type": "object"}}, "type": "function"}"#
            )
        );
    }

    #[test]
    fn tojson_accepts_jinja_ensure_ascii_false() {
        let environment =
            build_environment("{{ tools[0] | tojson(ensure_ascii=False) }}".to_string()).unwrap();
        let mut tool = tool_definition();
        tool.function.description = Some("Read café paths".to_string());
        let rendered = render_with_environment(
            &environment,
            &[ChatMessage::user("hello")],
            &[tool],
            ChatTemplateOptions::default(),
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert!(rendered.contains("café"), "{rendered}");
    }

    #[test]
    fn generation_annotation_renders_as_an_ordinary_block() {
        let environment = build_environment(
            "before{%- generation -%}assistant{%- endgeneration -%}after".to_string(),
        )
        .unwrap();
        let rendered = render_with_environment(
            &environment,
            &[],
            &[],
            ChatTemplateOptions::default(),
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert_eq!(rendered, "beforeassistantafter");
    }

    #[test]
    fn missing_optional_schema_members_are_falsey() {
        let environment = build_environment(
            concat!(
                "{% if tools[0]['function']['parameters']['properties']",
                "['path']['nullable'] %}nullable{% else %}required{% endif %}"
            )
            .to_string(),
        )
        .unwrap();
        let rendered = render_with_environment(
            &environment,
            &[ChatMessage::user("hello")],
            &[tool_definition()],
            ChatTemplateOptions::default(),
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert_eq!(rendered, "required");
    }

    #[test]
    fn checkpoint_special_tokens_reach_the_template() {
        let environment = build_environment("{{ bos_token }}x{{ eos_token }}".to_string()).unwrap();
        let rendered = render_with_environment(
            &environment,
            &[ChatMessage::user("hello")],
            &[],
            ChatTemplateOptions::default(),
            "<bos>",
            "<eos>",
        )
        .unwrap();
        assert_eq!(rendered, "<bos>x<eos>");
    }

    #[test]
    #[ignore = "requires the local Step-3.7 checkpoint"]
    fn local_step37_template_renders_generation_prefix() {
        let model_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.7-flash-nvfp4");
        let template = CheckpointChatTemplate::from_model_dir(model_dir).unwrap();
        let rendered = template
            .render(
                &[ChatMessage::user("hello")],
                &[tool_definition()],
                ChatTemplateOptions {
                    reasoning_effort: Some(ChatReasoningEffort::Low),
                    ..ChatTemplateOptions::default()
                },
            )
            .unwrap();
        assert!(
            rendered.starts_with(
                "<｜begin▁of▁sentence｜><|im_start|>system\nReasoning: low\n\n# Tools"
            ),
            "{rendered:?}"
        );
        assert!(
            rendered.contains("<|im_start|>user\nhello<|im_end|>"),
            "{rendered:?}"
        );
        assert!(
            rendered.ends_with("<|im_start|>assistant\n<think>\n"),
            "{rendered:?}"
        );
        assert!(rendered.contains("<tools>"), "{rendered:?}");

        let long_prompt = template
            .render_and_tokenize(
                &[ChatMessage::user("word ".repeat(3_000))],
                &[tool_definition()],
                ChatTemplateOptions {
                    reasoning_effort: Some(ChatReasoningEffort::Low),
                    ..ChatTemplateOptions::default()
                },
            )
            .unwrap();
        assert!(long_prompt.token_ids.len() > 2_048);
        assert!(
            long_prompt
                .text
                .ends_with("<|im_start|>assistant\n<think>\n")
        );
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_gemma4_template_accepts_standard_tool_schema() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let template = CheckpointChatTemplate::from_model_dir(model_dir).unwrap();
        let prompt = template
            .render_and_tokenize(
                &[ChatMessage::user("hello")],
                &[tool_definition()],
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert!(prompt.text.contains("<|tool>"), "{:?}", prompt.text);
        assert!(
            prompt.text.contains("declaration:read_file"),
            "{:?}",
            prompt.text
        );
        assert!(!prompt.token_ids.is_empty());
    }

    #[test]
    #[ignore = "requires a prepared local DeepSeek V4 thin checkpoint"]
    fn local_deepseek4_template_renders_tool_generation_prefix() {
        let model_dir = PathBuf::from(
            std::env::var("DEEPSEEK4_THIN_DIR")
                .expect("set DEEPSEEK4_THIN_DIR to the prepared thin checkpoint"),
        );
        let template = CheckpointChatTemplate::from_model_dir(model_dir).unwrap();
        let prompt = template
            .render_and_tokenize(
                &[ChatMessage::user("inspect the repository")],
                &[tool_definition()],
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert!(
            prompt.text.starts_with("<｜begin▁of▁sentence｜>"),
            "{:?}",
            prompt.text
        );
        assert!(
            prompt.text.contains("<｜DSML｜tool_calls>"),
            "{:?}",
            prompt.text
        );
        assert!(
            prompt.text.ends_with("<｜Assistant｜><think>"),
            "{:?}",
            prompt.text
        );
        assert!(!prompt.token_ids.is_empty());

        let history = [
            ChatMessage::user("inspect README.md"),
            ChatMessage::assistant_tool_calls(
                None,
                Some("I should read the file.".to_string()),
                vec![ChatToolCall {
                    id: "call_read".to_string(),
                    function: ChatFunctionCall {
                        name: "read_file".to_string(),
                        arguments: BTreeMap::from([("path".to_string(), json!("README.md"))]),
                    },
                }],
            ),
            ChatMessage::tool("call_read", "Eider documentation"),
            ChatMessage::user("summarize it"),
        ];
        let history_prompt = template
            .render_and_tokenize(
                &history,
                &[tool_definition()],
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert!(
            history_prompt
                .text
                .contains("<｜DSML｜invoke name=\"read_file\">"),
            "{:?}",
            history_prompt.text
        );
        assert!(
            history_prompt
                .text
                .contains("<tool_result>Eider documentation</tool_result>"),
            "{:?}",
            history_prompt.text
        );
        assert!(
            history_prompt.text.ends_with("<｜Assistant｜><think>"),
            "{:?}",
            history_prompt.text
        );
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

    #[test]
    #[ignore = "requires the local Muse Glimmer checkpoint"]
    fn local_muse_template_renders_atem_history_and_recipient_prefix() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/muse-glimmer-30b-nvfp4");
        let template = CheckpointChatTemplate::from_model_dir(model_dir).unwrap();
        let call = ChatToolCall {
            id: "call_1".to_string(),
            function: ChatFunctionCall {
                name: "functions.read_file".to_string(),
                arguments: BTreeMap::from([("path".to_string(), json!("src/main.rs"))]),
            },
        };
        let mut tool = tool_definition();
        tool.function.name = "functions.read_file".to_string();
        let prompt = template
            .render_and_tokenize(
                &[
                    ChatMessage::system("Be precise."),
                    ChatMessage::user("Inspect the entry point."),
                    ChatMessage::assistant_tool_calls(
                        None,
                        Some("I should inspect the file.".to_string()),
                        vec![call],
                    ),
                    ChatMessage::tool("call_1", "fn main() {}"),
                ],
                &[tool],
                ChatTemplateOptions {
                    reasoning_effort: Some(ChatReasoningEffort::XHigh),
                    ..ChatTemplateOptions::default()
                },
            )
            .unwrap();

        assert!(prompt.text.contains("Reasoning strength: xhigh."));
        assert!(
            prompt
                .text
                .contains("<atem:invoke name=\"functions.read_file\">")
        );
        assert!(
            prompt
                .text
                .contains("<tool_output name=\"functions.read_file\">")
        );
        assert!(prompt.text.ends_with("<|start|>assistant"));
        assert!(!prompt.token_ids.is_empty());
    }
}
