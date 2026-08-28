use eider_cuda::{CublasLt, CudaStream, DeviceBuffer, Result};
use infer::qwen3::qwen36::{Qwen36LayerBlock, Qwen36Model, Qwen36TextModel};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest();
    let block = Qwen36LayerBlock::load(&model, 0)?;
    let mut workspace = block.workspace(&model, 8)?;
    let mut state = block.sequence_state(&model, 8)?;
    let lt = CublasLt::new()?;
    let stream = CudaStream::new_non_blocking()?;

    let text = Qwen36TextModel::open(&model_dir)?;
    let mut hidden = DeviceBuffer::zeroed(manifest.hidden)?;
    let token_id_device = DeviceBuffer::from_host(&[0u32])?;
    text.gather_embedding(&token_id_device, hidden.output(), &stream)?;
    let h = hidden.copy_to_host(&stream)?;
    println!("embedding[0]: first={:.6} max|={:.6}", h[0], max_abs(&h));

    println!("running layer 0 block (kind={:?})", block.kind);
    let step = block.run_one_token(
        &lt,
        &mut workspace,
        &mut state,
        manifest,
        &hidden,
        0,
        &stream,
        None,
        None,
    )?;
    let out = step.output.copy_to_host(&stream)?;
    println!(
        "  block out: len={} first={:.6} max|={:.6}",
        out.len(),
        out[0],
        max_abs(&out)
    );
    Ok(())
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()))
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-moe-probe".to_string());
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(eider_cuda::Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir>"),
        }),
    }
}
