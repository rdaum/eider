//! Dory generation and checkpoint comparison without an HTTP server.
use eider_inference::dory::DoryModel;
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use std::{path::Path, time::Instant};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: dory_generate MODEL_DIR [LOOPS=2] [TOKENS=64] [CHUNK=2048] [PROMPT] [LOGITS_JSON]"
    );
    let path = Path::new(&args[1]);
    let loops = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(2);
    let count = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let chunk = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(2048);
    let prompt = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("What is 7 times 8? Answer briefly.");
    let template = CheckpointChatTemplate::from_model_dir(path)?;
    let rendered = template.render_and_tokenize(
        &[ChatMessage::user(prompt)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    let model = DoryModel::load(path, Some(loops))?;
    let mut sequence = model.new_sequence(rendered.token_ids.len() + count + 1)?;
    let mut workspace = model.new_prefill_workspace(chunk)?;
    let start = Instant::now();
    for ids in rendered.token_ids.chunks(chunk) {
        model.prefill(&mut workspace, &mut sequence, ids)?;
    }
    model.synchronize()?;
    eprintln!(
        "loops={loops} prompt={} prefill_ms={:.2} prefill_tok_s={:.1} state_mib={:.1}",
        rendered.token_ids.len(),
        start.elapsed().as_secs_f64() * 1000.,
        rendered.token_ids.len() as f64 / start.elapsed().as_secs_f64(),
        sequence.device_bytes() as f64 / 1048576.
    );
    if let Some(file) = args.get(6) {
        std::fs::write(
            file,
            serde_json::to_vec(
                &serde_json::json!({"tokens":rendered.token_ids,"logits":model.logits_to_host(&sequence)?}),
            )?,
        )?;
    }
    let start = Instant::now();
    let mut output = Vec::new();
    for i in 0..count {
        let token = model.argmax_with_logit(&mut sequence)?.0;
        output.push(token);
        if [2, 11].contains(&token) {
            break;
        }
        if i + 1 < count {
            model.decode(&mut sequence, token)?;
        }
    }
    model.synchronize()?;
    eprintln!(
        "output={} decode_ms={:.2} decode_tok_s={:.1}",
        output.len(),
        start.elapsed().as_secs_f64() * 1000.,
        output.len().saturating_sub(1) as f64 / start.elapsed().as_secs_f64()
    );
    println!(
        "{}",
        template
            .tokenizer()
            .decode(&output, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    );
    Ok(())
}
