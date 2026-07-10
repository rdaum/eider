use infer::nvfp4::{Error, Result};
use infer::qwen3::qwen36::Qwen36TextModel;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let (model_dir, start_token, steps, max_tokens) = parse_args()?;
    let model = Qwen36TextModel::open(&model_dir)?;
    let manifest = model.manifest();
    println!(
        "Qwen3.6 decode probe: layers={} hidden={} vocab={} ffn={}",
        manifest.layers,
        manifest.hidden,
        manifest.vocab,
        ffn_label(manifest.ffn)
    );
    let mut state = model.new_decode_state(max_tokens)?;
    let mut token_id = start_token;
    for step in 0..steps {
        let next = model.decode_one_token(&mut state, token_id)?;
        println!(
            "  step {step}: in={token_id} out={} value={:.6}",
            next.id, next.value
        );
        token_id = next.id;
    }
    Ok(())
}

fn ffn_label(ffn: infer::qwen3::infer::QwenFfnConfig) -> String {
    use infer::qwen3::infer::QwenFfnConfig;
    match ffn {
        QwenFfnConfig::Dense => "dense".to_string(),
        QwenFfnConfig::Moe {
            experts,
            experts_per_token,
            expert_intermediate,
            norm_topk_prob,
        } => format!(
            "moe experts={experts} top_k={experts_per_token} intermediate={expert_intermediate} norm_topk_prob={norm_topk_prob}"
        ),
    }
}

fn parse_args() -> Result<(PathBuf, u32, usize, usize)> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-decode-probe".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir> [start-token] [steps] [max-tokens]"),
        })?;
    let start_token = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "start token",
            detail: err.to_string(),
        })?
        .unwrap_or(0);
    let steps = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "steps",
            detail: err.to_string(),
        })?
        .unwrap_or(8);
    let max_tokens = args
        .next()
        .and_then(|value| value.into_string().ok())
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|err| Error::Format {
            label: "max-tokens",
            detail: err.to_string(),
        })?
        .unwrap_or(steps.max(1));
    Ok((model_dir, start_token, steps, max_tokens))
}
