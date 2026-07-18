use infer::gemma4::Gemma4Model;
use infer::nvfp4::{CudaStream, Error, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let (model_dir, tokens) = parse_args()?;
    let model = Gemma4Model::load(&model_dir)?;
    let mut state = model.new_decode_state(tokens.len())?;
    let stream = CudaStream::new_blocking()?;
    for token in tokens {
        let next = model.decode_one(&mut state, token, &stream)?;
        println!(
            "Gemma 4 decode: input_token={} next_token={} logit={}",
            next.input_token, next.token, next.logit
        );
    }
    Ok(())
}

fn parse_args() -> Result<(PathBuf, Vec<u32>)> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "gemma4-generate".to_string());
    let Some(path) = args.next() else {
        return Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir> <token-id> [token-id ...]"),
        });
    };
    let tokens = args
        .map(|token| {
            token
                .into_string()
                .ok()
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| Error::Format {
                    label: "token id",
                    detail: "expected an unsigned integer".to_string(),
                })
        })
        .collect::<Result<Vec<u32>>>()?;
    if tokens.is_empty() {
        return Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir> <token-id> [token-id ...]"),
        });
    }
    Ok((PathBuf::from(path), tokens))
}
