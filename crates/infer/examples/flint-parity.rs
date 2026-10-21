//! Compare causal chunk sizes using the downloaded Flint checkpoint.
use eider_inference::flint::FlintModel;
use std::path::PathBuf;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Reference {
    tokens: Vec<u32>,
    logits: Vec<f32>,
}

fn top(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: flint-parity MODEL.gguf [REFERENCE.json]"))?,
    );
    let external = std::env::args_os()
        .nth(2)
        .map(|path| -> anyhow::Result<Reference> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let model = FlintModel::load(&path)?;
    let tokens: Vec<u32> = external
        .as_ref()
        .map(|r| r.tokens.clone())
        .unwrap_or_else(|| {
            std::iter::once(150003)
                .chain((0..519).map(|i| 100 + (i * 31 % 1200) as u32))
                .chain(std::iter::once(150004))
                .collect()
        });
    let mut cache = model.new_sequence_cache(3, tokens.len() + 3, 0)?;
    let mut reference = model.admit(&mut cache, tokens.len() + 3, &[])?.unwrap();
    let mut work = model.new_workspace(64.min(tokens.len() + 3), tokens.len() + 3)?;
    let started = Instant::now();
    for &token in &tokens {
        model.forward(&mut work, &mut reference, &mut cache, &[token])?;
    }
    let expected = model.logits(&reference)?;
    if let Some(external) = external {
        anyhow::ensure!(
            external.logits.len() == expected.len(),
            "reference vocabulary mismatch"
        );
        let max_abs = expected
            .iter()
            .zip(&external.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms = (expected
            .iter()
            .zip(&external.logits)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        println!(
            "external reference: top={}, max_abs={max_abs:.6}, rms={rms:.6}",
            top(&external.logits)
        );
        anyhow::ensure!(
            top(&expected) == top(&external.logits),
            "external reference top token differs"
        );
    }
    println!(
        "single-row reference: {} tokens in {:.3}s, top={}",
        tokens.len(),
        started.elapsed().as_secs_f64(),
        top(&expected)
    );
    for token in [123, 234, 345] {
        model.forward(&mut work, &mut reference, &mut cache, &[token])?;
    }
    let continued = model.logits(&reference)?;
    for chunk in [7, 64] {
        let mut sequence = model.admit(&mut cache, tokens.len() + 3, &[])?.unwrap();
        let started = Instant::now();
        for tokens in tokens.chunks(chunk) {
            model.forward(&mut work, &mut sequence, &mut cache, tokens)?;
        }
        let actual = model.logits(&sequence)?;
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        anyhow::ensure!(actual.iter().all(|v| v.is_finite()), "non-finite logits");
        println!(
            "chunk={chunk}: {:.3}s, top={}, max_abs={max_abs:.6}, rms={rms:.6}",
            started.elapsed().as_secs_f64(),
            top(&actual)
        );
        anyhow::ensure!(actual == expected, "chunk parity failed");
        for token in [123, 234, 345] {
            model.forward(&mut work, &mut sequence, &mut cache, &[token])?;
        }
        anyhow::ensure!(
            model.logits(&sequence)? == continued,
            "decode after chunked prefill differs"
        );
        model.finish(&mut cache, sequence)?;
    }
    model.finish(&mut cache, reference)?;
    Ok(())
}
