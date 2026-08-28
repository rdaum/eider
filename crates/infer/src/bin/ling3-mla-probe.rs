//! Compare multi-token Ling 3 Tiny MLA attention with the CPU reference.

use eider_cuda::{CudaStream, DeviceBuffer, Error, Result};
use eider_format::{ModelOptCheckpoint, SafeTensorShard};
use eider_inference::ling3::{Ling3Manifest, Ling3MlaAttention};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).ok_or_else(usage)?;
    let reference_path = args.next().map(PathBuf::from).ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }
    let manifest = Ling3Manifest::load(&model_dir)?;
    let checkpoint = ModelOptCheckpoint::open(&model_dir)?;
    let reference = SafeTensorShard::open(reference_path)?;
    let input = reference.read_float_tensor_as_f32("input")?;
    if !input.len().is_multiple_of(manifest.hidden_size) {
        return Err(Error::Shape {
            label: "Ling MLA reference input",
            expected: format!("a multiple of {}", manifest.hidden_size),
            actual: input.len().to_string(),
        });
    }
    let tokens = input.len() / manifest.hidden_size;
    let layer = Ling3MlaAttention::load(&checkpoint, &manifest, 3)?;
    let mut state = layer.new_state(tokens)?;
    let mut workspace = layer.new_workspace()?;
    let stream = CudaStream::new_non_blocking()?;
    let mut actual = Vec::with_capacity(tokens * manifest.hidden_size);
    for row in input.chunks_exact(manifest.hidden_size) {
        let input = DeviceBuffer::from_host(row)?;
        layer.run_one_token(&input, &mut workspace, &mut state, &stream)?;
        actual.extend(
            layer
                .output(&workspace)
                .copy_to_host(&stream)?
                .iter()
                .copied(),
        );
    }
    let expected = reference.read_float_tensor_as_f32("output")?;
    compare(&actual, &expected, 0.9999, 0.02)?;
    println!(
        "Ling 3 Tiny MLA layer 3 parity passed: tokens={tokens} storage={} weights={:.3} MiB cache={:.3} MiB",
        if manifest.fp8.is_some() {
            "mixed FP8"
        } else {
            "BF16"
        },
        layer.device_bytes() as f64 / (1024.0 * 1024.0),
        state.device_bytes() as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

fn usage() -> Error {
    Error::Format {
        label: "ling3-mla-probe arguments",
        detail: "usage: ling3-mla-probe <model-dir> <reference.safetensors>".to_string(),
    }
}

fn compare(
    actual: &[f32],
    expected: &[f32],
    minimum_cosine: f64,
    maximum_nrmse: f64,
) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::Shape {
            label: "Ling MLA output",
            expected: format!("{} values", expected.len()),
            actual: format!("{} values", actual.len()),
        });
    }
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut worst = 0.0f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        dot += actual as f64 * expected as f64;
        actual_norm += (actual as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        squared_error += ((actual - expected) as f64).powi(2);
        worst = worst.max((actual - expected).abs());
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    println!("  output: cosine={cosine:.6} nrmse={nrmse:.6} worst_abs={worst:.6}");
    if cosine < minimum_cosine || nrmse > maximum_nrmse {
        return Err(Error::Format {
            label: "Ling MLA parity",
            detail: format!(
                "cosine={cosine:.6} required>={minimum_cosine:.6}, nrmse={nrmse:.6} required<={maximum_nrmse:.6}"
            ),
        });
    }
    Ok(())
}
