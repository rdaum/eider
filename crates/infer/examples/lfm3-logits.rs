//! Raw-token LFM3 parity probe: CHECKPOINT OUTPUT_JSON TOKEN...
use eider_inference::lfm3::Lfm3Model;
use std::{path::PathBuf, time::Instant};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("CHECKPOINT"));
    let output = PathBuf::from(args.next().expect("OUTPUT_JSON"));
    let tokens = args
        .map(|s| s.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(!tokens.is_empty(), "provide token IDs");
    let model = Lfm3Model::load(root)?;
    let mut workspace = model.new_workspace(tokens.len().max(1))?;
    let mut state = model.new_sequence(tokens.len() + 32)?;
    let start = Instant::now();
    model.forward(&mut workspace, &mut state, &tokens)?;
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let logits = model.logits_to_host(&state)?;
    let mut generated = Vec::new();
    let start = Instant::now();
    for _ in 0..32 {
        let token = model.argmax_with_logit(&mut state)?.0;
        generated.push(token);
        model.forward(&mut workspace, &mut state, &[token])?;
    }
    let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(
        output,
        serde_json::to_vec(&serde_json::json!({
            "tokens": tokens, "logits": logits, "generated": generated,
            "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        }))?,
    )?;
    eprintln!(
        "prefill={prefill_ms:.1} ms decode={:.1} tok/s",
        32000.0 / decode_ms
    );
    Ok(())
}
