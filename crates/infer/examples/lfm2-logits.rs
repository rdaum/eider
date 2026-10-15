//! Compare full logits across revisions with the same prompt and continuation.
use eider_inference::lfm2::Lfm2Model;
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use serde::{Deserialize, Serialize};
use std::{env, fs, time::Instant};

#[derive(Deserialize)]
struct Input {
    text: String,
    #[serde(default)]
    continuation: Vec<u32>,
    #[serde(default)]
    timing_only: bool,
}
#[derive(Serialize)]
struct Output {
    prompt_tokens: usize,
    continuation: Vec<u32>,
    logits: Vec<Vec<f32>>,
    prefill_seconds: f64,
    decode_seconds: f64,
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "usage: lfm2-logits CHECKPOINT INPUT_JSON OUTPUT_JSON CHUNK"
    );
    let input: Input = serde_json::from_slice(&fs::read(&args[2])?)?;
    anyhow::ensure!(
        !input.timing_only || !input.continuation.is_empty(),
        "timing_only requires a fixed continuation"
    );
    let chunk: usize = args[4].parse()?;
    anyhow::ensure!(chunk > 0, "chunk must be positive");
    let template = CheckpointChatTemplate::from_model_dir(&args[1])?;
    let prompt = template.render_and_tokenize(
        &[ChatMessage::user(input.text)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    let model = Lfm2Model::load(&args[1])?;
    let count = if input.continuation.is_empty() {
        16
    } else {
        input.continuation.len()
    };
    let mut sequence = model.new_sequence(prompt.token_ids.len() + count)?;
    let mut workspace = model.new_prefill_workspace(chunk)?;
    let start = Instant::now();
    let (last, prefix) = prompt
        .token_ids
        .split_last()
        .ok_or_else(|| anyhow::anyhow!("empty prompt"))?;
    for tokens in prefix.chunks(chunk) {
        model.prefill(&mut workspace, &mut sequence, tokens)?;
    }
    model.prefill(&mut workspace, &mut sequence, &[*last])?;
    model.synchronize()?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let mut logits = Vec::new();
    if !input.timing_only {
        logits.push(model.logits_to_host(&sequence)?);
    }
    let mut decode_seconds = 0.0;
    let mut continuation = Vec::new();
    for step in 0..count {
        let next = match input.continuation.get(step) {
            Some(&token) => token,
            None => model.argmax_with_logit(&mut sequence)?.0,
        };
        continuation.push(next);
        let start = Instant::now();
        model.decode(&mut sequence, next)?;
        model.synchronize()?;
        decode_seconds += start.elapsed().as_secs_f64();
        if !input.timing_only {
            logits.push(model.logits_to_host(&sequence)?);
        }
    }
    fs::write(
        &args[3],
        serde_json::to_vec(&Output {
            prompt_tokens: prompt.token_ids.len(),
            continuation,
            logits,
            prefill_seconds,
            decode_seconds,
        })?,
    )?;
    println!(
        "prompt_tokens={} prefill_seconds={prefill_seconds:.3} decode_seconds={decode_seconds:.3}",
        prompt.token_ids.len()
    );
    Ok(())
}
