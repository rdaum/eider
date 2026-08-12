use infer::nvfp4::{CudaStream, Error, Result};
use infer::qwen3::qwen36::{Qwen36DecodeRow, Qwen36TextModel};
use infer::runtime::qwen36_sequence::{Qwen36Sequence, new_qwen36_sequence_cache};
use std::env;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let (model_dir, prompt) = parse_args()?;
    let tokenizer =
        Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|error| Error::Format {
            label: "tokenizer",
            detail: error.to_string(),
        })?;
    let rendered = render_prompt(&prompt);
    let encoding = tokenizer
        .encode(rendered, false)
        .map_err(|error| Error::Format {
            label: "tokenizer encode",
            detail: error.to_string(),
        })?;
    let prompt_ids = encoding.get_ids();
    if prompt_ids.is_empty() {
        return Err(Error::Format {
            label: "prompt",
            detail: "prompt tokenized to zero tokens".to_string(),
        });
    }

    let model = Qwen36TextModel::open(&model_dir)?;
    let max_tokens = prompt_ids.len() + 1;
    let stream = CudaStream::new_non_blocking()?;
    let mut cache = new_qwen36_sequence_cache(&model, 1, max_tokens)?;
    let mut sequence = Qwen36Sequence::admit(&model, &mut cache, max_tokens, &stream)?;
    let mut workspace = model.new_decode_batch_workspace(1, max_tokens)?;
    let mut logits = Vec::new();
    for &token_id in prompt_ids {
        let mut rows = [Qwen36DecodeRow {
            token_id,
            sequence: &mut sequence,
        }];
        logits = model
            .decode_batch(&mut workspace, &mut rows, &mut cache)?
            .copy_logits()?;
    }

    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.select_nth_unstable_by(20, |left, right| right.1.total_cmp(&left.1));
    ranked.truncate(20);
    ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));

    println!("prompt_tokens={prompt_ids:?}");
    for (rank, (token, logit)) in ranked.into_iter().enumerate() {
        let text = tokenizer
            .decode(&[token as u32], true)
            .unwrap_or_else(|_| "<decode error>".to_string());
        println!("{rank:2}: token={token:6} logit={logit:10.5} text={text:?}");
    }
    Ok(())
}

fn render_prompt(prompt: &str) -> String {
    if prompt.contains("<|im_start|>") {
        return prompt.to_string();
    }
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

fn parse_args() -> Result<(PathBuf, String)> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-logit-dump".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir> [prompt]"),
        })?;
    let prompt = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "What is the meaning of life".to_string());
    if args.next().is_some() {
        return Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir> [prompt]"),
        });
    }
    Ok((model_dir, prompt))
}
