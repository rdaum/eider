use eider_cuda::Result;
use infer::qwen3::infer::Qwen3Model;
use infer::qwen3::layer0::DEFAULT_MODEL_DIR;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let mut token_id = args
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|err| eider_cuda::Error::Format {
            label: "initial token id",
            detail: err.to_string(),
        })?
        .unwrap_or(0);
    let steps = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| eider_cuda::Error::Format {
            label: "decode steps",
            detail: err.to_string(),
        })?
        .unwrap_or(16);

    println!("Qwen3-8B iterative token-id decode");
    println!("  model dir: {}", model_dir.display());
    println!("  initial token: {token_id}");
    println!("  steps: {steps}");

    let model = Qwen3Model::load(&model_dir)?;
    let mut state = model.new_decode_state(steps)?;
    for step in 0..steps {
        let next = model.decode_one(&mut state, token_id)?;
        println!(
            "  step {step:02}: input={} next={} logit={:.6e}",
            next.input_token, next.token, next.logit
        );
        token_id = next.token;
    }

    Ok(())
}
