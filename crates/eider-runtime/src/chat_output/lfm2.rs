//! Literal-only parser for LFM's `[function(argument=value), ...]` protocol.

use crate::chat::ChatFunctionCall;
use eider_format::{Error, Result};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

// The marker can also occur inside a generated string argument.
pub(super) fn closing_offset(body: &str, marker: &str) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    for (index, c) in body.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if let Some(delimiter) = quote {
            if c == '\\' {
                escaped = true;
            } else if c == delimiter {
                quote = None;
            }
        } else if c == '\'' || c == '"' {
            quote = Some(c);
        } else if body[index..].starts_with(marker) {
            return Some(index);
        }
    }
    None
}

pub(super) fn parse(body: &str) -> Result<Vec<ChatFunctionCall>> {
    let mut p = Parser { rest: body };
    p.expect('[')?;
    let mut calls = Vec::new();
    if !p.take(']') {
        loop {
            let name = p.name()?.to_string();
            p.expect('(')?;
            let mut arguments = BTreeMap::new();
            if !p.take(')') {
                loop {
                    let key = p.name()?.to_string();
                    p.expect('=')?;
                    if arguments.insert(key, p.value(0)?).is_some() {
                        return Err(invalid("duplicate argument"));
                    }
                    if p.take(')') {
                        break;
                    }
                    p.expect(',')?;
                    if p.take(')') {
                        break;
                    }
                }
            }
            calls.push(ChatFunctionCall { name, arguments });
            if p.take(']') {
                break;
            }
            p.expect(',')?;
            if p.take(']') {
                break;
            }
        }
    }
    if !p.rest.trim().is_empty() || calls.is_empty() {
        return Err(invalid(
            "expected a nonempty tool-call list with no trailing text",
        ));
    }
    Ok(calls)
}

struct Parser<'a> {
    rest: &'a str,
}

impl<'a> Parser<'a> {
    fn take(&mut self, c: char) -> bool {
        self.rest = self.rest.trim_start();
        if let Some(rest) = self.rest.strip_prefix(c) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<()> {
        if self.take(c) {
            Ok(())
        } else {
            Err(invalid(format!("expected {c:?}")))
        }
    }

    fn name(&mut self) -> Result<&'a str> {
        self.rest = self.rest.trim_start();
        let end = self
            .rest
            .find(|c: char| !c.is_ascii_alphanumeric() && !matches!(c, '_' | '-' | '.'))
            .unwrap_or(self.rest.len());
        let name = &self.rest[..end];
        super::validate_protocol_name("function or parameter", name)?;
        self.rest = &self.rest[end..];
        Ok(name)
    }

    fn string(&mut self) -> Result<String> {
        self.rest = self.rest.trim_start();
        let quote = self
            .rest
            .chars()
            .next()
            .ok_or_else(|| invalid("missing string"))?;
        if quote != '\'' && quote != '"' {
            return Err(invalid("expected quoted string"));
        }
        let mut escaped = false;
        let mut end = None;
        for (i, c) in self.rest[1..].char_indices() {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == quote {
                end = Some(i + 1);
                break;
            }
        }
        let end = end.ok_or_else(|| invalid("unterminated string"))?;
        let raw = &self.rest[1..end];
        let mut json = String::from("\"");
        let mut chars = raw.chars();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    let next = chars.next().ok_or_else(|| invalid("unfinished escape"))?;
                    if next == '\'' {
                        json.push('\'');
                    } else {
                        json.push('\\');
                        json.push(next);
                    }
                }
                '"' => json.push_str("\\\""),
                _ => json.push(c),
            }
        }
        json.push('"');
        let value =
            serde_json::from_str(&json).map_err(|e| invalid(format!("invalid string: {e}")))?;
        self.rest = &self.rest[end + 1..];
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth >= 64 {
            return Err(invalid("argument nesting exceeds 64 levels"));
        }
        self.rest = self.rest.trim_start();
        if self.rest.starts_with(['\'', '"']) {
            return self.string().map(Value::String);
        }
        if self.take('[') {
            let mut values = Vec::new();
            if !self.take(']') {
                loop {
                    values.push(self.value(depth + 1)?);
                    if self.take(']') {
                        break;
                    }
                    self.expect(',')?;
                    if self.take(']') {
                        break;
                    }
                }
            }
            return Ok(Value::Array(values));
        }
        if self.take('{') {
            let mut values = Map::new();
            if !self.take('}') {
                loop {
                    let key = self.string()?;
                    self.expect(':')?;
                    if values.insert(key, self.value(depth + 1)?).is_some() {
                        return Err(invalid("duplicate object key"));
                    }
                    if self.take('}') {
                        break;
                    }
                    self.expect(',')?;
                    if self.take('}') {
                        break;
                    }
                }
            }
            return Ok(Value::Object(values));
        }
        let end = self
            .rest
            .find(|c: char| c.is_whitespace() || matches!(c, ',' | ')' | ']' | '}'))
            .unwrap_or(self.rest.len());
        let raw = &self.rest[..end];
        let value = match raw {
            "True" | "true" => Value::Bool(true),
            "False" | "false" => Value::Bool(false),
            "None" | "null" => Value::Null,
            _ => {
                let value: Value = serde_json::from_str(raw)
                    .map_err(|_| invalid("expected a literal argument value"))?;
                if !value.is_number() {
                    return Err(invalid("expected a numeric literal"));
                }
                value
            }
        };
        self.rest = &self.rest[end..];
        Ok(value)
    }
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "LFM2 tool call",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nested_literals_and_multiple_calls() {
        let calls = parse(r#"[write(path='a,b).rs', contents='it\'s a \\ path\n', options={"list": [True, null, -2.5e2]}), run(command="echo \"hi\"", dry=False)]"#).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["contents"], json!("it's a \\ path\n"));
        assert_eq!(
            calls[0].arguments["options"],
            json!({"list": [true, null, -250.0]})
        );
        assert_eq!(calls[1].arguments["command"], json!("echo \"hi\""));
    }

    #[test]
    fn rejects_expressions_and_malformed_calls() {
        for text in [
            "[run(x=eval('x'))]",
            "[run(x=a.b)]",
            "[run(x=1,x=2)]",
            "[run(1)]",
            "[run(x='unterminated)]",
            "[run()] trailing",
            "[]",
            "[run(x=NaN)]",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
        let text = format!("[run(x={}0{})]", "[".repeat(65), "]".repeat(65));
        assert!(parse(&text).is_err());
    }

    #[test]
    fn closing_marker_in_a_string_is_literal() {
        let body = "[run(text='<|tool_call_end|>', value='it\\'s')]<|tool_call_end|>tail";
        let end = closing_offset(body, "<|tool_call_end|>").unwrap();
        let calls = parse(&body[..end]).unwrap();
        assert_eq!(calls[0].arguments["text"], json!("<|tool_call_end|>"));
    }
}
