use eider_cuda::{CudaStream, DeviceBuffer, ModelOptCheckpoint};
use infer::nemotron3::{Nemotron3AttentionLayer, Nemotron3Manifest};
use std::path::PathBuf;

fn main() -> eider_cuda::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eider_cuda::Error::Format {
            label: "nemotron3-attention-layer-probe arguments",
            detail: "usage: nemotron3-attention-layer-probe <model-dir> [layer]".to_string(),
        })?;
    let layer = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| eider_cuda::Error::Format {
            label: "nemotron3-attention-layer-probe layer",
            detail: error.to_string(),
        })?
        .unwrap_or(7);
    let manifest = Nemotron3Manifest::from_model_dir(&model_dir)?;
    let checkpoint = ModelOptCheckpoint::open(&model_dir)?;
    let weights = Nemotron3AttentionLayer::load(&checkpoint, &manifest, layer)?;
    let mut workspace = weights.workspace()?;
    let mut state = weights.sequence_state(3)?;
    let stream = CudaStream::new_non_blocking()?;
    let mut checksum = 0.0;
    for token in 0..3 {
        let hidden = (0..manifest.hidden_size)
            .map(|index| ((index % 31) as f32 - 15.0 + token as f32) * 0.001)
            .collect::<Vec<_>>();
        let hidden = DeviceBuffer::from_host(&hidden)?;
        weights.run_one_token(&hidden, &mut workspace, &mut state, None, &stream)?;
        let output = weights.output(&workspace).copy_to_host(&stream)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(eider_cuda::Error::Format {
                label: "Nemotron 3 attention layer probe",
                detail: "layer output contains a non-finite value".to_string(),
            });
        }
        checksum = output.iter().sum::<f32>();
    }
    println!(
        "Nemotron 3 attention layer {layer}: weights={:.3} GiB tokens={} checksum={checksum:.8}",
        weights.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        state.len(),
    );
    Ok(())
}
