//! Request-scoped constrained decoding for Qwen XML tool calls.

use super::chat::ChatTool;
use llguidance::{Constraint, ParserFactory, api::TopLevelGrammar};
use nvfp4::{Error, Result};
use serde_json::{Map, Value};
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::ByteTokenizer;

const TOOL_CALL_OPEN: &str = "<tool_call>\n";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const PARAMETER_CLOSE: &str = "\n</parameter>\n";
const MAX_PERMUTED_PARAMETERS: usize = 6;

/// Shared tokenizer-specific compiler for request-scoped Qwen tool grammars.
pub(crate) struct QwenXmlGrammarFactory {
    parser: ParserFactory,
    mask_words: usize,
    tool_call_open: Option<u32>,
    tool_call_close: Option<u32>,
}

impl QwenXmlGrammarFactory {
    pub(crate) fn new(tokenizer: &Tokenizer, vocab: usize) -> Result<Self> {
        let tokenizer_json = tokenizer.to_string(false).map_err(|error| Error::Format {
            label: "Qwen tool grammar tokenizer",
            detail: error.to_string(),
        })?;
        let byte_tokenizer = ByteTokenizer::from_json_bytes(tokenizer_json.as_bytes())
            .map_err(grammar_error("Qwen tool grammar tokenizer"))?;
        let token_env = byte_tokenizer
            .into_tok_env(Some(vocab))
            .map_err(grammar_error("Qwen tool grammar tokenizer"))?;
        let mut parser = ParserFactory::new_simple(&token_env)
            .map_err(grammar_error("Qwen tool grammar compiler"))?;
        parser.quiet();
        Ok(Self {
            parser,
            mask_words: vocab.div_ceil(32),
            tool_call_open: tokenizer.token_to_id("<tool_call>"),
            tool_call_close: tokenizer.token_to_id("</tool_call>"),
        })
    }

    pub(crate) fn build(&self, tools: &[ChatTool]) -> Result<Option<QwenXmlToolGrammar>> {
        if tools.is_empty() {
            return Ok(None);
        }
        let (source, triggers) =
            build_qwen_xml_grammar(tools, self.tool_call_open, self.tool_call_close)?;
        let parser = self
            .parser
            .create_parser(TopLevelGrammar::from_lark(source))
            .map_err(grammar_error("Qwen tool grammar"))?;
        Ok(Some(QwenXmlToolGrammar {
            constraint: Constraint::new(parser),
            mask_words: self.mask_words,
            triggers,
            trigger_buffer: Vec::new(),
            pending_tokens: Vec::new(),
            active: false,
            mask_pending: false,
        }))
    }
}

/// Lazy grammar state owned by one generated Qwen response.
pub(crate) struct QwenXmlToolGrammar {
    constraint: Constraint,
    mask_words: usize,
    triggers: Arc<[Vec<u8>]>,
    trigger_buffer: Vec<u8>,
    pending_tokens: Vec<(u32, Vec<u8>)>,
    active: bool,
    mask_pending: bool,
}

impl QwenXmlToolGrammar {
    pub(crate) fn deep_clone(&self) -> Self {
        Self {
            constraint: self.constraint.deep_clone(),
            mask_words: self.mask_words,
            triggers: Arc::clone(&self.triggers),
            trigger_buffer: self.trigger_buffer.clone(),
            pending_tokens: self.pending_tokens.clone(),
            active: self.active,
            mask_pending: self.mask_pending,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.active && self.constraint.has_pending_stop()
    }

    /// Computes the allowed-token bitset once the lazy trigger has fired.
    pub(crate) fn mask(&mut self) -> Result<Option<Vec<u32>>> {
        if !self.active || self.is_complete() {
            return Ok(None);
        }
        let result = self
            .constraint
            .compute_mask()
            .map_err(grammar_error("Qwen tool grammar mask"))?;
        let mask = result.sample_mask.as_ref().ok_or_else(|| Error::Format {
            label: "Qwen tool grammar mask",
            detail: "grammar stopped before the tool call was emitted".to_string(),
        })?;
        if mask.as_slice().len() < self.mask_words {
            return Err(Error::Shape {
                label: "Qwen tool grammar mask",
                expected: format!("at least {} words", self.mask_words),
                actual: format!("{} words", mask.as_slice().len()),
            });
        }
        self.mask_pending = true;
        Ok(Some(mask.as_slice()[..self.mask_words].to_vec()))
    }

    /// Advances the grammar with one target-committed output token.
    pub(crate) fn commit(&mut self, token: u32) -> Result<()> {
        if !self.active {
            let piece = self.constraint.tok_trie().decode(&[token]);
            self.pending_tokens.push((token, piece.clone()));
            self.trigger_buffer.extend_from_slice(&piece);
            let max_trigger = self.triggers.iter().map(Vec::len).max().unwrap_or(0);
            if let Some(start) = self
                .triggers
                .iter()
                .filter_map(|trigger| find_bytes(&self.trigger_buffer, trigger))
                .min()
            {
                let mut end = 0usize;
                let first = self
                    .pending_tokens
                    .iter()
                    .position(|(_, piece)| {
                        end += piece.len();
                        end > start
                    })
                    .unwrap_or(0);
                let trigger_tokens = self.pending_tokens[first..]
                    .iter()
                    .map(|(token, _)| *token)
                    .collect::<Vec<_>>();
                self.constraint.start_without_prompt();
                self.constraint
                    .force_tokens(&trigger_tokens)
                    .map_err(grammar_error("Qwen tool grammar trigger"))?;
                self.pending_tokens.clear();
                self.trigger_buffer.clear();
                self.active = true;
            } else {
                let keep = max_trigger.saturating_add(32);
                while self.trigger_buffer.len() > keep && self.pending_tokens.len() > 1 {
                    let (_, piece) = self.pending_tokens.remove(0);
                    self.trigger_buffer.drain(..piece.len());
                }
            }
            return Ok(());
        }
        if self.is_complete() {
            return Err(Error::Format {
                label: "Qwen tool grammar token",
                detail: format!("token {token} followed a complete tool call"),
            });
        }
        if !self.mask_pending {
            self.mask()?;
        }
        let allowed = self
            .constraint
            .step_result()
            .sample_mask
            .as_ref()
            .is_some_and(|mask| mask.is_allowed(token));
        if !allowed {
            return Err(Error::Format {
                label: "Qwen tool grammar token",
                detail: format!("target committed disallowed token {token}"),
            });
        }
        self.constraint
            .commit_token(Some(token))
            .map_err(grammar_error("Qwen tool grammar token"))?;
        self.mask_pending = false;
        Ok(())
    }

    pub(crate) fn token_allowed(mask: &[u32], token: u32) -> bool {
        let token = token as usize;
        mask.get(token / 32)
            .is_some_and(|word| word & (1 << (token % 32)) != 0)
    }
}

fn build_qwen_xml_grammar(
    tools: &[ChatTool],
    tool_call_open: Option<u32>,
    tool_call_close: Option<u32>,
) -> Result<(String, Arc<[Vec<u8>]>)> {
    let mut grammar = String::from("start: wrapped_call");
    for index in 0..tools.len() {
        grammar.push_str(&format!(" | direct_call_{index}"));
    }
    grammar.push('\n');
    grammar.push_str("wrapped_call: ");
    grammar.push_str(&tool_call_literal(TOOL_CALL_OPEN, tool_call_open));
    grammar.push_str(" function ");
    grammar.push_str(&tool_call_literal(TOOL_CALL_CLOSE, tool_call_close));
    grammar.push('\n');
    grammar.push_str("function:");
    for index in 0..tools.len() {
        grammar.push_str(if index == 0 { " " } else { " | " });
        grammar.push_str(&format!("function_{index}"));
    }
    grammar.push('\n');

    let mut triggers = vec![b"<tool_call>".to_vec()];
    for (tool_index, tool) in tools.iter().enumerate() {
        let name = &tool.function.name;
        validate_xml_name("function", name)?;
        let function_open = format!("<function={name}>\n");
        triggers.push(format!("<function={name}>").into_bytes());

        grammar.push_str(&format!(
            "direct_call_{tool_index}: direct_head_{tool_index} args_{tool_index} "
        ));
        grammar.push_str(&lark_literal("</function>\n"));
        grammar.push(' ');
        grammar.push_str(&tool_call_literal("</tool_call>", tool_call_close));
        grammar.push('\n');
        grammar.push_str(&format!("direct_head_{tool_index}[lazy]: TEXT "));
        grammar.push_str(&lark_literal(&function_open));
        grammar.push('\n');
        grammar.push_str(&format!("function_{tool_index}: "));
        grammar.push_str(&lark_literal(&function_open));
        grammar.push_str(&format!(" args_{tool_index} "));
        grammar.push_str(&lark_literal("</function>\n"));
        grammar.push('\n');

        build_arguments(&mut grammar, tool_index, &tool.function.parameters)?;
    }
    grammar.push_str("TEXT: /(\\n|.)*/\n");
    Ok((grammar, triggers.into()))
}

fn build_arguments(grammar: &mut String, tool: usize, schema: &Value) -> Result<()> {
    let empty = Map::new();
    let properties = schema["properties"].as_object().unwrap_or(&empty);
    let required = schema["required"]
        .as_array()
        .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    for name in &required {
        if !properties.contains_key(*name) {
            return Err(Error::Format {
                label: "Qwen tool grammar schema",
                detail: format!("required parameter {name:?} has no property schema"),
            });
        }
    }

    let required_parameters = properties
        .keys()
        .enumerate()
        .filter(|(_, name)| required.iter().any(|required| required == name))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let optional_parameters = properties
        .keys()
        .enumerate()
        .filter(|(_, name)| !required.iter().any(|required| required == name))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    grammar.push_str(&format!("args_{tool}: required_{tool}"));
    if !optional_parameters.is_empty() {
        grammar.push_str(&format!(" optional_{tool}"));
    }
    grammar.push('\n');
    build_required_states(grammar, tool, &required_parameters);
    if !optional_parameters.is_empty() {
        build_optional_states(grammar, tool, &optional_parameters);
    }

    for (parameter, (name, parameter_schema)) in properties.iter().enumerate() {
        validate_xml_name("parameter", name)?;
        grammar.push_str(&format!("parameter_{tool}_{parameter}: "));
        grammar.push_str(&lark_literal(&format!("<parameter={name}>\n")));
        if resolves_to_string(parameter_schema) {
            grammar.push_str(&format!(" string_value_{tool}_{parameter}\n"));
            grammar.push_str(&format!("string_value_{tool}_{parameter}[lazy]: TEXT "));
            grammar.push_str(&lark_literal(PARAMETER_CLOSE));
            grammar.push('\n');
        } else {
            grammar.push_str(" %json ");
            let mut parameter_schema = parameter_schema.clone();
            if let (Some(target), Some(definitions)) =
                (parameter_schema.as_object_mut(), schema.get("$defs"))
            {
                target.insert("$defs".to_string(), definitions.clone());
            }
            grammar.push_str(&parameter_schema.to_string());
            grammar.push(' ');
            grammar.push_str(&lark_literal(PARAMETER_CLOSE));
            grammar.push('\n');
        }
    }
    Ok(())
}

fn build_required_states(grammar: &mut String, tool: usize, required: &[usize]) {
    if required.len() > MAX_PERMUTED_PARAMETERS {
        grammar.push_str(&format!("required_{tool}:"));
        for parameter in required {
            grammar.push_str(&format!(" parameter_{tool}_{parameter}"));
        }
        grammar.push('\n');
        return;
    }
    let full = (1u64 << required.len()) - 1;
    for seen in 0..=full {
        grammar.push_str(&format!("required_{tool}_{seen}:"));
        if seen == full {
            grammar.push_str(" \"\"\n");
            continue;
        }
        let mut first = true;
        for (bit, parameter) in required.iter().enumerate() {
            if seen & (1 << bit) != 0 {
                continue;
            }
            grammar.push_str(if first { " " } else { " | " });
            first = false;
            let next = seen | (1 << bit);
            grammar.push_str(&format!(
                "parameter_{tool}_{parameter} required_{tool}_{next}"
            ));
        }
        grammar.push('\n');
    }
    grammar.push_str(&format!("required_{tool}: required_{tool}_0\n"));
}

fn build_optional_states(grammar: &mut String, tool: usize, optional: &[usize]) {
    if optional.len() > MAX_PERMUTED_PARAMETERS {
        for position in 0..=optional.len() {
            grammar.push_str(&format!("optional_{tool}_{position}:"));
            if let Some(parameter) = optional.get(position) {
                grammar.push_str(&format!(
                    " optional_{tool}_{} | parameter_{tool}_{parameter} optional_{tool}_{}\n",
                    position + 1,
                    position + 1
                ));
            } else {
                grammar.push_str(" \"\"\n");
            }
        }
        grammar.push_str(&format!("optional_{tool}: optional_{tool}_0\n"));
        return;
    }

    let full = (1u64 << optional.len()) - 1;
    for seen in 0..=full {
        grammar.push_str(&format!("optional_{tool}_{seen}: \"\""));
        for (bit, parameter) in optional.iter().enumerate() {
            if seen & (1 << bit) != 0 {
                continue;
            }
            let next = seen | (1 << bit);
            grammar.push_str(&format!(
                " | parameter_{tool}_{parameter} optional_{tool}_{next}"
            ));
        }
        grammar.push('\n');
    }
    grammar.push_str(&format!("optional_{tool}: optional_{tool}_0\n"));
}

fn resolves_to_string(schema: &Value) -> bool {
    match &schema["type"] {
        Value::String(kind) => kind == "string",
        Value::Array(kinds) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
        _ => schema["enum"]
            .as_array()
            .is_some_and(|values| !values.is_empty() && values.iter().all(Value::is_string)),
    }
}

fn validate_xml_name(kind: &'static str, name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(Error::Format {
            label: "Qwen tool grammar",
            detail: format!("invalid {kind} name {name:?}"),
        });
    }
    Ok(())
}

fn lark_literal(value: &str) -> String {
    serde_json::to_string(value).expect("strings serialize as JSON")
}

fn tool_call_literal(value: &str, token: Option<u32>) -> String {
    let (marker, suffix) = if let Some(suffix) = value.strip_prefix("<tool_call>") {
        ("<tool_call>", suffix)
    } else if let Some(suffix) = value.strip_prefix("</tool_call>") {
        ("</tool_call>", suffix)
    } else {
        return lark_literal(value);
    };
    match token {
        Some(token) if suffix.is_empty() => format!("<[{token}]>"),
        Some(token) => format!("<[{token}]> {}", lark_literal(suffix)),
        None => lark_literal(&format!("{marker}{suffix}")),
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    (!needle.is_empty())
        .then(|| {
            haystack
                .windows(needle.len())
                .position(|window| window == needle)
        })
        .flatten()
}

fn grammar_error(label: &'static str) -> impl FnOnce(anyhow::Error) -> Error {
    move |error| Error::Format {
        label,
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::ChatFunctionDefinition;
    use serde_json::json;
    use std::path::Path;

    fn tool() -> ChatTool {
        ChatTool::function(ChatFunctionDefinition {
            name: "read".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1}
                },
                "required": ["path"]
            }),
        })
    }

    #[test]
    fn qwen_xml_source_uses_exact_tools_and_schema_values() {
        let (source, triggers) = build_qwen_xml_grammar(&[tool()], None, None).unwrap();
        assert!(source.contains("<function=read>\\n"));
        assert!(source.contains("<parameter=path>\\n"));
        assert!(source.contains("%json "));
        assert!(source.contains("\"minimum\":1"));
        assert!(source.contains("\"type\":\"integer\""));
        assert!(!source.contains("read_file"));
        assert_eq!(triggers.len(), 2);
    }

    #[test]
    fn optional_parameters_use_uniqueness_tracking() {
        let tool = tool();
        let optional_index = tool.function.parameters["properties"]
            .as_object()
            .unwrap()
            .keys()
            .position(|name| name == "line")
            .unwrap();
        let (source, _) = build_qwen_xml_grammar(&[tool], None, None).unwrap();

        assert!(source.contains(&format!(
            "optional_0_0: \"\" | parameter_0_{optional_index} optional_0_1"
        )));
        assert!(source.contains("optional_0_1: \"\""));
        assert!(!source.contains("optional_0: ("));
    }

    #[test]
    fn qwen_xml_rejects_invalid_protocol_names() {
        let mut invalid = tool();
        invalid.function.name = "read file".to_string();
        assert!(build_qwen_xml_grammar(&[invalid], None, None).is_err());
    }

    #[test]
    fn many_required_parameters_use_checkpoint_order() {
        let mut properties = Map::new();
        let mut required = Vec::new();
        for index in 0..=MAX_PERMUTED_PARAMETERS {
            let name = format!("arg_{index}");
            properties.insert(name.clone(), json!({"type": "string"}));
            required.push(Value::String(name));
        }
        let tool = ChatTool::function(ChatFunctionDefinition {
            name: "many".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": properties,
                "required": required
            }),
        });
        let (source, _) = build_qwen_xml_grammar(&[tool], None, None).unwrap();
        assert!(source.contains(
            "required_0: parameter_0_0 parameter_0_1 parameter_0_2 parameter_0_3 parameter_0_4 parameter_0_5 parameter_0_6"
        ));
        assert!(!source.contains("required_0_0:"));
    }

    #[test]
    #[ignore = "requires EIDER_QWEN38_MODEL_DIR"]
    fn local_qwen_xml_grammar_activates_and_rejects_wrong_tool() {
        let model_dir = std::env::var("EIDER_QWEN38_MODEL_DIR").unwrap();
        let tokenizer = Tokenizer::from_file(Path::new(&model_dir).join("tokenizer.json")).unwrap();
        let factory =
            QwenXmlGrammarFactory::new(&tokenizer, tokenizer.get_vocab_size(true)).unwrap();
        let mut grammar = factory.build(&[tool()]).unwrap().unwrap();
        let valid = tokenizer
            .encode(
                "Some reasoning.\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>",
                false,
            )
            .unwrap();
        for &token in valid.get_ids() {
            if let Some(mask) = grammar.mask().unwrap() {
                assert_eq!(mask.len(), tokenizer.get_vocab_size(true).div_ceil(32));
                assert!(QwenXmlToolGrammar::token_allowed(&mask, token));
            }
            grammar.commit(token).unwrap();
        }
        assert!(grammar.is_complete());
        assert!(grammar.mask().unwrap().is_none());

        let mut invalid = factory.build(&[tool()]).unwrap().unwrap();
        let prefix = tokenizer.encode("<tool_call>\n", false).unwrap();
        for &token in prefix.get_ids() {
            invalid.commit(token).unwrap();
        }
        assert!(invalid.is_active());
        let wrong = tokenizer.encode("<function=read_file>\n", false).unwrap();
        let mut rejected = false;
        for &token in wrong.get_ids() {
            let mask = invalid.mask().unwrap().unwrap();
            if !QwenXmlToolGrammar::token_allowed(&mask, token) {
                rejected = true;
                break;
            }
            invalid.commit(token).unwrap();
        }
        assert!(rejected);

        let mut duplicate = factory.build(&[tool()]).unwrap().unwrap();
        let duplicate_call = tokenizer
            .encode(
                "<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=line>\n1\n</parameter>\n<parameter=line>\n2\n</parameter>\n</function>\n</tool_call>",
                false,
            )
            .unwrap();
        let mut rejected = false;
        for &token in duplicate_call.get_ids() {
            if let Some(mask) = duplicate.mask().unwrap()
                && !QwenXmlToolGrammar::token_allowed(&mask, token)
            {
                rejected = true;
                break;
            }
            duplicate.commit(token).unwrap();
        }
        assert!(rejected);
    }
}
