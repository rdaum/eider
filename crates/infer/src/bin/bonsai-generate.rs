use infer::bonsai::BonsaiModel;
use infer::nvfp4::{Error, Result};
use infer::runtime::bonsai_sequence_cache::{BonsaiSequence, new_bonsai_sequence_cache};
use std::path::PathBuf;

fn main() -> Result<()> {
    let (gguf, tokens) = parse_args()?;
    let model = BonsaiModel::load(&gguf)?;
    let mut cache = new_bonsai_sequence_cache(&model, 1, tokens.len())?;
    let mut sequence = BonsaiSequence::admit(&model, &mut cache, tokens.len())?;
    for token in tokens {
        model.forward_one(&mut sequence, token, &mut cache)?;
        let (next, logit) = model.argmax_with_logit(&mut sequence)?;
        println!("Bonsai decode: input_token={token} next_token={next} logit={logit}");
    }
    sequence.finish(&mut cache)?;
    Ok(())
}

fn parse_args() -> Result<(PathBuf, Vec<u32>)> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "bonsai-generate".to_string());
    let Some(path) = args.next() else {
        return Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model.gguf> <token-id> [token-id ...]"),
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
            detail: format!("{program} <model.gguf> <token-id> [token-id ...]"),
        });
    }
    Ok((PathBuf::from(path), tokens))
}
