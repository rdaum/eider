use infer::nemotron3::Nemotron3Model;
use std::path::PathBuf;

fn main() -> infer::nvfp4::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| infer::nvfp4::Error::Format {
            label: "nemotron3-model-probe arguments",
            detail: "usage: nemotron3-model-probe <model-dir> [token]".to_string(),
        })?;
    let token = std::env::args()
        .nth(2)
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|error| infer::nvfp4::Error::Format {
            label: "nemotron3-model-probe token",
            detail: error.to_string(),
        })?
        .unwrap_or(1);
    let model = Nemotron3Model::load(&model_dir)?;
    let mut state = model.sequence_state(4)?;
    model.forward_one(&mut state, token)?;
    let next = model.argmax(&mut state)?;
    println!(
        "Nemotron 3 model: weights={:.3} GiB input={token} next={next}",
        model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok(())
}
