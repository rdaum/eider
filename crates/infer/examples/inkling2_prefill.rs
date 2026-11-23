//! Prompt-prefill throughput for a local Inkling 2 Mini checkpoint.
//!
//! Prefills a synthetic prompt of `TOKENS` tokens in chunks of `CHUNK` rows,
//! twice, and reports the warm run's throughput. A third prefill in 32-row
//! chunks, which use the row-grouped attention kernel, checks the final
//! logits against the timed path.
//!
//! Usage: `cargo run --release -p eider-inference --example inkling2_prefill -- MODEL_DIR [TOKENS] [CHUNK]`

use eider_cuda::Result;
use eider_inference::inkling2::Inkling2Model;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: inkling2_prefill MODEL_DIR [TOKENS] [CHUNK]");
        std::process::exit(2);
    }));
    let tokens: usize = args
        .next()
        .map_or(4096, |n| n.parse().expect("token count"));
    let chunk: usize = args.next().map_or(1024, |n| n.parse().expect("chunk rows"));
    let model = Inkling2Model::load(&root)?;
    // A deterministic spread of ordinary vocabulary ids routes like text.
    let prompt: Vec<u32> = (0..tokens as u32)
        .map(|i| 1000 + i.wrapping_mul(2_654_435_761) % 100_000)
        .collect();
    let mut workspace = model.new_workspace(chunk)?;
    let mut cache = model.new_sequence_cache(1, tokens + 1, 0)?;
    let mut prefill = |rows: usize| -> Result<(f64, Vec<f32>)> {
        let mut sequence = model
            .admit(&mut cache, tokens + 1, &prompt)?
            .expect("an empty cache admits one sequence");
        let start = Instant::now();
        for part in prompt.chunks(rows) {
            model.forward(&mut workspace, &mut sequence, &mut cache, part)?;
        }
        let seconds = start.elapsed().as_secs_f64();
        let logits = model.logits_to_host(&sequence)?;
        model.finish(&mut cache, sequence)?;
        Ok((seconds, logits))
    };
    let mut timed = Vec::new();
    for run in 0..2 {
        let (seconds, logits) = prefill(chunk)?;
        println!(
            "run {run}: {tokens} tokens in {chunk}-row chunks, {seconds:.3} s ({:.0} tok/s)",
            tokens as f64 / seconds
        );
        timed = logits;
    }
    let (_, reference) = prefill(32)?;
    let argmax = |logits: &[f32]| {
        (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .expect("vocabulary")
    };
    let difference = timed
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "32-row reference: argmax {} vs {}, max |logit difference| {difference:.4}",
        argmax(&timed),
        argmax(&reference)
    );
    Ok(())
}
