use eider_cuda::{CudaStream, DeviceBuffer, ModelOptCheckpoint};
use eider_inference::nemotron3::{Nemotron3MambaLayer, Nemotron3Manifest};
use std::path::PathBuf;

fn main() -> eider_cuda::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eider_cuda::Error::Format {
            label: "nemotron3-layer-probe arguments",
            detail: "usage: nemotron3-layer-probe <model-dir> [layer]".to_string(),
        })?;
    let layer = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| eider_cuda::Error::Format {
            label: "nemotron3-layer-probe layer",
            detail: error.to_string(),
        })?
        .unwrap_or(0);
    let manifest = Nemotron3Manifest::from_model_dir(&model_dir)?;
    let checkpoint = ModelOptCheckpoint::open(&model_dir)?;
    let weights = Nemotron3MambaLayer::load(&checkpoint, &manifest, layer)?;
    let mut workspace = weights.workspace()?;
    let mut state = weights.sequence_state()?;
    let hidden = (0..manifest.hidden_size)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.001)
        .collect::<Vec<_>>();
    let hidden = DeviceBuffer::from_host(&hidden)?;
    let stream = CudaStream::new_non_blocking()?;
    weights.run_one_token(&hidden, &mut workspace, &mut state, &stream)?;
    let output = weights.output(&workspace).copy_to_host(&stream)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(eider_cuda::Error::Format {
            label: "Nemotron 3 Mamba layer probe",
            detail: "layer output contains a non-finite value".to_string(),
        });
    }
    let checksum = output.iter().sum::<f32>();
    println!(
        "Nemotron 3 Mamba layer {layer}: weights={:.3} GiB state={:.3} MiB checksum={checksum:.8}",
        weights.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        state.device_bytes() as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}
