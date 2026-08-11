//! Compare Ling 3 Tiny layer-1 MoE with the independent CPU artifact.

use infer::ling3::{Ling3Manifest, Ling3Moe};
use infer::nvfp4::{CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, SafeTensorShard};
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
    let layer = Ling3Moe::load(&checkpoint, &manifest, 1)?;
    let mut workspace = layer.new_workspace()?;
    let stream = CudaStream::new_non_blocking()?;
    let input = DeviceBuffer::from_host(&reference.read_float_tensor_as_f32("input")?)?;
    layer.run_one_token(&input, &mut workspace, &stream)?;

    let indices = layer.indices(&workspace, &stream)?;
    let expected_indices = reference
        .read_float_tensor_as_f32("indices")?
        .into_iter()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    if indices != expected_indices {
        return Err(Error::Format {
            label: "Ling MoE route indices",
            detail: format!("actual={indices:?} expected={expected_indices:?}"),
        });
    }
    compare(
        "route weights",
        &layer.weights(&workspace, &stream)?,
        &reference.read_float_tensor_as_f32("weights")?,
        0.999999,
        1.0e-4,
    )?;
    compare(
        "output",
        &layer.output(&workspace).copy_to_host(&stream)?,
        &reference.read_float_tensor_as_f32("output")?,
        0.98,
        0.20,
    )?;
    println!(
        "Ling 3 Tiny MoE layer 1 parity passed: routes={indices:?} storage={} weights={:.3} MiB",
        if manifest.fp8.is_some() {
            "NVFP4"
        } else {
            "BF16"
        },
        layer.device_bytes() as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

fn usage() -> Error {
    Error::Format {
        label: "ling3-moe-probe arguments",
        detail: "usage: ling3-moe-probe <model-dir> <reference.safetensors>".to_string(),
    }
}

fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    minimum_cosine: f64,
    maximum_nrmse: f64,
) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::Shape {
            label: "Ling MoE reference",
            expected: format!("{} {label} values", expected.len()),
            actual: actual.len().to_string(),
        });
    }
    if !actual.iter().all(|value| value.is_finite()) {
        let index = actual
            .iter()
            .position(|value| !value.is_finite())
            .expect("non-finite value exists");
        return Err(Error::Format {
            label: "Ling MoE parity",
            detail: format!(
                "{label}: non-finite value {} at index {index}",
                actual[index]
            ),
        });
    }
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| actual as f64 * expected as f64)
        .sum::<f64>();
    let actual_norm = actual
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>();
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    println!("  {label}: cosine={cosine:.6} nrmse={nrmse:.6}");
    if cosine < minimum_cosine || nrmse > maximum_nrmse {
        return Err(Error::Format {
            label: "Ling MoE parity",
            detail: format!("{label}: cosine={cosine:.6} nrmse={nrmse:.6}"),
        });
    }
    Ok(())
}
