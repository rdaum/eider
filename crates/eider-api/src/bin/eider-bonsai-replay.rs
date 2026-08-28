//! Deterministically replays one Responses request through Bonsai.

use anyhow::{Result, anyhow};
use eider_api::protocol::ResponseRequest;
use eider_format::{GgufIndex, GgufValue};
use eider_inference::bonsai::{BonsaiModel, BonsaiPrefillMode};
use eider_inference::bonsai::{BonsaiSequence, new_bonsai_sequence_cache};
use eider_runtime::chat::{ChatMessage, CheckpointChatTemplate};
use eider_runtime::chat_output::{ChatOutputCodec, ChatOutputEvent};
use eider_runtime::generation::GenerationConfig;
use serde_json::{Value, json};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const GGUF_NAME: &str = "Ternary-Bonsai-8B-Q2_0_g64.gguf";
const DEFAULT_MAX_NEW_TOKENS: usize = 64;
const LOGIT_CANDIDATES: usize = 5;

fn main() -> Result<()> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "eider-bonsai-replay".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage(&program))?;
    let request_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage(&program))?;
    let mode = args
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(|| usage(&program))?;
    let mode = parse_mode(&mode).ok_or_else(|| usage(&program))?;
    let output_path = args.next().map(PathBuf::from);
    let max_new_tokens = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| failure("max-new-tokens", error))?
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let continuation_assistant = args
        .next()
        .map(|value| value.to_string_lossy().into_owned());
    let continuation_user = args
        .next()
        .map(|value| value.to_string_lossy().into_owned());
    if args.next().is_some()
        || max_new_tokens == 0
        || continuation_assistant.is_some() != continuation_user.is_some()
    {
        return Err(usage(&program));
    }

    let request = fs::read_to_string(&request_path).map_err(|error| {
        failure(
            "Responses replay request",
            format!("{}: {error}", request_path.display()),
        )
    })?;
    let request: ResponseRequest = serde_json::from_str(&request).map_err(|error| {
        failure(
            "Responses replay request",
            format!("{}: {error}", request_path.display()),
        )
    })?;
    let defaults = GenerationConfig::from_model_dir(&model_dir)?;
    let mut chat = request
        .into_chat_request(&defaults)
        .map_err(|error| failure("Responses replay request", error.message))?;
    if let (Some(assistant), Some(user)) = (continuation_assistant, continuation_user) {
        chat.messages.push(ChatMessage::assistant(assistant));
        chat.messages.push(ChatMessage::user(user));
    }
    let template = bonsai_chat_template(&model_dir)?;
    let prompt = template.render_and_tokenize(&chat.messages, &chat.tools, chat.template)?;
    let starts_in_reasoning = chat.template.add_generation_prompt && chat.template.enable_thinking;

    let loaded_at = Instant::now();
    let model = BonsaiModel::load_with_prefill_mode(&model_dir.join(GGUF_NAME), mode)?;
    let load_ms = loaded_at.elapsed().as_secs_f64() * 1000.0;
    let capacity = prompt.token_ids.len() + max_new_tokens;
    let mut cache = new_bonsai_sequence_cache(&model, 1, capacity)?;
    let mut sequence = BonsaiSequence::admit(&model, &mut cache, capacity)?;
    let mut prefill_workspace = model.new_prefill_workspace(prompt.token_ids.len(), capacity)?;
    let prefill_at = Instant::now();
    model.prefill(
        &mut prefill_workspace,
        &mut sequence,
        &prompt.token_ids,
        &mut cache,
    )?;
    let prefill_ms = prefill_at.elapsed().as_secs_f64() * 1000.0;

    let mut output = ChatOutputCodec::new(template.tokenizer(), &chat.tools, starts_in_reasoning)?;
    let mut token_ids = Vec::new();
    let mut events = Vec::new();
    let mut steps = Vec::new();
    let decode_at = Instant::now();
    let mut finish_reason = "length";
    for step in 0..max_new_tokens {
        if step != 0 {
            model.forward_one(&mut sequence, token_ids[step - 1], &mut cache)?;
        }
        let logits = model.logits_to_host(&mut sequence)?;
        let candidates = top_logits(&logits, LOGIT_CANDIDATES);
        let token_id = candidates[0].0;
        token_ids.push(token_id);
        steps.push(json!({
            "index": step,
            "selected_token": token_id,
            "selected_logit": candidates[0].1,
            "top_logits": candidates
                .into_iter()
                .map(|(id, logit)| json!({"token": id, "logit": logit}))
                .collect::<Vec<_>>(),
        }));
        events.extend(output.push_token(token_id)?);
        if chat.generation.eos_token_ids.contains(&token_id) {
            finish_reason = "eos";
            break;
        }
    }
    let decode_ms = decode_at.elapsed().as_secs_f64() * 1000.0;
    events.extend(if finish_reason == "length" {
        output.finish_truncated()?
    } else {
        output.finish()?
    });
    let raw_text = template
        .tokenizer()
        .decode(&token_ids, false)
        .map_err(|error| failure("Responses replay output", error))?;
    sequence.finish(&mut cache)?;
    let report = serde_json::to_string_pretty(&json!({
        "prefill_mode": mode_name(mode),
        "prompt_token_count": prompt.token_ids.len(),
        "max_new_tokens": max_new_tokens,
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": prompt.token_ids.len() as f64 * 1000.0 / prefill_ms,
        "decode_ms": decode_ms,
        "decode_tokens_per_second": token_ids.len() as f64 * 1000.0 / decode_ms,
        "finish_reason": finish_reason,
        "raw_token_ids": token_ids,
        "raw_text": raw_text,
        "parsed_events": events.into_iter().map(event_json).collect::<Vec<_>>(),
        "steps": steps,
    }))
    .expect("replay report serializes");
    if let Some(output_path) = output_path {
        fs::write(&output_path, report).map_err(|error| {
            failure(
                "Responses replay output",
                format!("{}: {error}", output_path.display()),
            )
        })?;
    } else {
        println!("{report}");
    }
    Ok(())
}

fn parse_mode(value: &str) -> Option<BonsaiPrefillMode> {
    match value {
        "bf16" => Some(BonsaiPrefillMode::Bf16),
        "nvfp4" => Some(BonsaiPrefillMode::Nvfp4),
        _ => None,
    }
}

fn bonsai_chat_template(model_dir: &std::path::Path) -> Result<CheckpointChatTemplate> {
    let gguf = model_dir.join(GGUF_NAME);
    let index = GgufIndex::open(&gguf).map_err(|error| failure("Bonsai GGUF import", error))?;
    let source = index
        .metadata()
        .get("tokenizer.chat_template")
        .and_then(GgufValue::as_str)
        .ok_or_else(|| {
            failure(
                "Bonsai chat template",
                format!("{} has no tokenizer.chat_template string", gguf.display()),
            )
        })?
        .to_string();
    Ok(CheckpointChatTemplate::from_source_and_tokenizer_files(
        source,
        gguf,
        model_dir.join("tokenizer.json"),
        model_dir.join("tokenizer_config.json"),
    )?)
}

fn mode_name(mode: BonsaiPrefillMode) -> &'static str {
    match mode {
        BonsaiPrefillMode::Bf16 => "bf16",
        BonsaiPrefillMode::Nvfp4 => "nvfp4",
    }
}

fn top_logits(logits: &[f32], limit: usize) -> Vec<(u32, f32)> {
    let mut values = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token, logit)| (token as u32, logit))
        .collect::<Vec<_>>();
    values.select_nth_unstable_by(limit - 1, |left, right| right.1.total_cmp(&left.1));
    values.truncate(limit);
    values.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    values
}

fn usage(program: &str) -> anyhow::Error {
    failure(
        "usage",
        format!(
            "{program} <model-dir> <responses-request.json> <bf16|nvfp4> [output.json] \
             [max-new-tokens] [assistant-continuation user-continuation]"
        ),
    )
}

fn failure(label: &str, detail: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{label}: {detail}")
}

fn event_json(event: ChatOutputEvent) -> Value {
    match event {
        ChatOutputEvent::Reasoning(text) => json!({"type": "reasoning", "text": text}),
        ChatOutputEvent::Text(text) => json!({"type": "text", "text": text}),
        ChatOutputEvent::ToolCall(call) => json!({
            "type": "tool_call",
            "id": call.id,
            "name": call.function.name,
            "arguments": call.function.arguments,
        }),
    }
}
