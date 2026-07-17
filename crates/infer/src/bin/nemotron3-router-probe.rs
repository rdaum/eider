use infer::nemotron3::{Nemotron3Manifest, Nemotron3Router};
use infer::nvfp4::{CudaStream, DeviceBuffer, ModelOptCheckpoint};
use std::path::PathBuf;

fn main() -> infer::nvfp4::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| infer::nvfp4::Error::Format {
            label: "nemotron3-router-probe arguments",
            detail: "usage: nemotron3-router-probe <model-dir> [layer]".to_string(),
        })?;
    let layer = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| infer::nvfp4::Error::Format {
            label: "nemotron3-router-probe layer",
            detail: error.to_string(),
        })?
        .unwrap_or(1);
    let manifest = Nemotron3Manifest::from_model_dir(&model_dir)?;
    let checkpoint = ModelOptCheckpoint::open(&model_dir)?;
    let router = Nemotron3Router::load(&checkpoint, &manifest, layer)?;
    let mut workspace = router.workspace()?;
    let hidden = (0..manifest.hidden_size)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.001)
        .collect::<Vec<_>>();
    let hidden = DeviceBuffer::from_host(&hidden)?;
    let stream = CudaStream::new_non_blocking()?;
    router.run(&hidden, &mut workspace, &stream)?;
    let indices = workspace.indices().copy_to_host(&stream)?;
    let weights = workspace.weights().copy_to_host(&stream)?;
    println!("Nemotron 3 router layer {layer}: {indices:?}");
    println!("weights: {weights:?}");
    println!("weight sum: {:.8}", weights.iter().sum::<f32>());
    Ok(())
}
