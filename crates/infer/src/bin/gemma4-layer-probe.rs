use infer::gemma4::{Gemma4Checkpoint, Gemma4DecoderLayer};
use infer::nvfp4::{CudaStream, DeviceBuffer, Error, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let checkpoint = Gemma4Checkpoint::open(&model_dir)?;
    let config = checkpoint.config();
    let stream = CudaStream::new_blocking()?;
    let layer = Gemma4DecoderLayer::load(&checkpoint, 0)?;
    let mut workspace = layer.new_workspace()?;
    let mut cache = layer.new_kv_cache(1)?;
    let mut compact_attention = layer.new_compact_attention_workspace(1)?;
    let input = DeviceBuffer::from_host(
        &(0..config.hidden_size)
            .map(|index| ((index % 97) as f32 - 48.0) / 48.0)
            .collect::<Vec<_>>(),
    )?;
    layer.run_decode_into(
        &input,
        &mut workspace,
        &mut cache,
        &mut compact_attention,
        0,
        &stream,
    )?;
    let output = layer.output(&workspace).copy_to_host(&stream)?;
    let finite = output.iter().all(|value| value.is_finite());
    let l2 = output.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !finite || l2 == 0.0 {
        return Err(Error::Format {
            label: "Gemma 4 layer probe",
            detail: format!("invalid layer output: finite={finite} l2={l2}"),
        });
    }
    println!("Gemma 4 layer 0 completed: output_l2={l2:.6}");
    Ok(())
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "gemma4-layer-probe".to_string());
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir>"),
        }),
    }
}
