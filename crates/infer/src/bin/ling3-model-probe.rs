//! Compare the complete Ling 3 Tiny decoder with an independent CPU artifact.

use infer::ling3::Ling3Model;
use infer::nvfp4::{CudaStream, Error, Result, SafeTensorShard};
use std::path::PathBuf;

const MIN_NVFP4_COSINE: f64 = 0.94;
const MAX_NVFP4_NRMSE: f64 = 0.35;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).ok_or_else(usage)?;
    let reference_path = args.next().map(PathBuf::from).ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }

    let reference = SafeTensorShard::open(reference_path)?;
    let tokens = reference
        .read_float_tensor_as_f32("tokens")?
        .into_iter()
        .map(|token| token as u32)
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err(Error::Format {
            label: "Ling model reference",
            detail: "the reference token sequence is empty".to_string(),
        });
    }
    let expected = reference.read_float_tensor_as_f32("logits")?;

    println!("loading complete Ling 3 model...");
    let model = Ling3Model::load(&model_dir)?;
    let mut state = model.new_state(tokens.len())?;
    let mut workspace = model.new_workspace()?;
    let stream = CudaStream::new_non_blocking()?;
    println!(
        "loaded {:.3} GiB; decoding {} reference tokens",
        model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        tokens.len()
    );

    let vocab = expected.len() / tokens.len();
    if vocab * tokens.len() != expected.len() {
        return Err(Error::Shape {
            label: "Ling model reference logits",
            expected: "a multiple of the token count".to_string(),
            actual: expected.len().to_string(),
        });
    }
    for (position, &token) in tokens.iter().enumerate() {
        model.decode_token(token, &mut state, &mut workspace, &stream)?;
        let actual = model.logits(&workspace).copy_to_host(&stream)?;
        let expected = &expected[position * vocab..(position + 1) * vocab];
        compare(position, token, &actual, expected)?;
    }
    println!("Ling 3 complete prompt/decode parity passed");
    Ok(())
}

fn usage() -> Error {
    Error::Format {
        label: "ling3-model-probe arguments",
        detail: "usage: ling3-model-probe <model-dir> <reference.safetensors>".to_string(),
    }
}

fn compare(position: usize, token: u32, actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::Shape {
            label: "Ling model logits",
            expected: expected.len().to_string(),
            actual: actual.len().to_string(),
        });
    }
    if !actual.iter().all(|value| value.is_finite()) {
        return Err(Error::Format {
            label: "Ling model logits",
            detail: format!("non-finite value at token position {position}"),
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
    let actual_top = argmax(actual);
    let expected_top = argmax(expected);
    println!(
        "  position={position} token={token}: cosine={cosine:.6} nrmse={nrmse:.6} top={actual_top} reference_top={expected_top}"
    );
    if cosine < MIN_NVFP4_COSINE || nrmse > MAX_NVFP4_NRMSE || actual_top != expected_top {
        return Err(Error::Format {
            label: "Ling model parity",
            detail: format!(
                "position {position}: cosine={cosine:.6} nrmse={nrmse:.6} top={actual_top} reference_top={expected_top}"
            ),
        });
    }
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or_default()
}
