//! Compare Ling 3 Tiny mixed FP8/NVFP4 layer zero with the CPU artifact.

use eider_cuda::{CudaStream, DeviceBuffer, Error, Result};
use eider_format::{ModelOptCheckpoint, SafeTensorShard};
use eider_inference::ling3::{Ling3KdaDenseLayer, Ling3Manifest};
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
    let layer = Ling3KdaDenseLayer::load(&checkpoint, &manifest, 0)?;
    let mut workspace = layer.new_workspace()?;
    let mut state = layer.new_state()?;
    let input = reference_values(&reference, "input")?;
    let input = DeviceBuffer::from_host(&input)?;
    let stream = CudaStream::new_non_blocking()?;
    layer.run_one_token(&input, &mut workspace, &mut state, &stream)?;

    compare(
        &reference,
        "normed",
        workspace.normed(),
        0.99999,
        0.005,
        &stream,
    )?;
    compare(
        &reference,
        "query",
        workspace.query(),
        0.9999,
        0.02,
        &stream,
    )?;
    compare(&reference, "key", workspace.key(), 0.9999, 0.02, &stream)?;
    compare(&reference, "value", workspace.value(), 0.999, 0.03, &stream)?;
    compare(&reference, "gate", workspace.gate(), 0.9999, 0.01, &stream)?;
    compare(&reference, "beta", workspace.beta(), 0.9999, 0.01, &stream)?;
    compare(
        &reference,
        "recurrent_output",
        workspace.recurrent_output(),
        0.999,
        0.04,
        &stream,
    )?;
    compare(
        &reference,
        "gated_output",
        workspace.gated_output(),
        0.999,
        0.04,
        &stream,
    )?;
    compare(
        &reference,
        "attention_output",
        workspace.attention_output(),
        0.999,
        0.04,
        &stream,
    )?;
    compare(
        &reference,
        "post_attention",
        workspace.post_attention(),
        0.9999,
        0.02,
        &stream,
    )?;
    compare(
        &reference,
        "ffn_input",
        workspace.ffn_input(),
        0.9999,
        0.02,
        &stream,
    )?;
    compare(
        &reference,
        "mlp_output",
        workspace.mlp_output(),
        0.98,
        0.20,
        &stream,
    )?;
    compare(
        &reference,
        "output",
        workspace.output(),
        0.999,
        0.03,
        &stream,
    )?;
    println!(
        "Ling 3 Tiny layer 0 parity passed: storage={} weights={:.3} MiB state={:.3} MiB",
        if manifest.fp8.is_some() {
            "mixed"
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
        label: "ling3-layer-probe arguments",
        detail: "usage: ling3-layer-probe <model-dir> <reference.safetensors>".to_string(),
    }
}

fn reference_values(reference: &SafeTensorShard, name: &str) -> Result<Vec<f32>> {
    Ok(reference.read_float_tensor_as_f32(name)?)
}

fn compare(
    reference: &SafeTensorShard,
    name: &str,
    actual: &DeviceBuffer<f32>,
    minimum_cosine: f64,
    maximum_nrmse: f64,
    stream: &CudaStream,
) -> Result<()> {
    let actual = actual.copy_to_host(stream)?.into_vec();
    let expected = reference_values(reference, name)?;
    if actual.len() != expected.len() {
        return Err(Error::Shape {
            label: "Ling 3 layer reference",
            expected: format!("{} values for {name}", expected.len()),
            actual: format!("{} values", actual.len()),
        });
    }
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        dot += actual as f64 * expected as f64;
        actual_norm += (actual as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        squared_error += ((actual - expected) as f64).powi(2);
        let error = (actual - expected).abs();
        if error > worst.0 {
            worst = (error, index, actual, expected);
        }
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    println!(
        "  {name}: cosine={cosine:.6} nrmse={nrmse:.6} worst_abs={:.6}",
        worst.0,
    );
    if !cosine.is_finite() || !nrmse.is_finite() || cosine < minimum_cosine || nrmse > maximum_nrmse
    {
        return Err(Error::Format {
            label: "Ling 3 layer parity",
            detail: format!(
                "{name}: cosine={cosine:.6} required>={minimum_cosine:.6}, nrmse={nrmse:.6} required<={maximum_nrmse:.6}; worst index={} actual={} expected={} abs={}",
                worst.1, worst.2, worst.3, worst.0,
            ),
        });
    }
    Ok(())
}
