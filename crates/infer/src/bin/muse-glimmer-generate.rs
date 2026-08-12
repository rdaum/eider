//! Minimal Muse Glimmer load and greedy-decode probe.

use infer::muse_glimmer::MuseGlimmerModel;
use infer::nvfp4::{Error, Result};
use infer::runtime::muse_glimmer_sequence_cache::{
    MuseGlimmerSequence, new_muse_glimmer_sequence_cache,
};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "muse-glimmer-generate".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!("{program} MODEL_DIR [initial-token] [tokens]"),
        })?;
    let mut token = args
        .next()
        .map(|value| parse(&program, value, "initial-token"))
        .transpose()?
        .unwrap_or(200_000);
    let tokens = args
        .next()
        .map(|value| parse(&program, value, "tokens"))
        .transpose()?
        .unwrap_or(1);
    if args.next().is_some() || tokens == 0 {
        return Err(Error::Format {
            label: "usage",
            detail: format!("{program} MODEL_DIR [initial-token] [tokens]"),
        });
    }

    let prompt = std::env::var("MUSE_GLIMMER_PROMPT").ok();
    let tokenizer = prompt
        .as_ref()
        .map(|_| {
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|error| Error::Format {
                label: "Muse Glimmer tokenizer",
                detail: error.to_string(),
            })
        })
        .transpose()?;
    let prompt_tokens = match (&prompt, &tokenizer) {
        (Some(prompt), Some(tokenizer)) => tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|error| Error::Format {
                label: "Muse Glimmer prompt",
                detail: error.to_string(),
            })?
            .get_ids()
            .to_vec(),
        _ => Vec::new(),
    };

    let dflash = std::env::var_os("MUSE_GLIMMER_DFLASH").map(PathBuf::from);
    let load_started = Instant::now();
    let model = match &dflash {
        Some(path) => MuseGlimmerModel::load_with_dflash(&model_dir, path)?,
        None => MuseGlimmerModel::load(&model_dir)?,
    };
    println!(
        "loaded Muse Glimmer in {:.3}s weights_gib={:.3}",
        load_started.elapsed().as_secs_f64(),
        model.device_bytes() as f64 / (1u64 << 30) as f64,
    );
    let capacity = prompt_tokens.len() + tokens as usize + usize::from(dflash.is_some()) * 16;
    let mut cache = new_muse_glimmer_sequence_cache(&model, 1, capacity.max(1))?;
    let mut sequence = MuseGlimmerSequence::admit(&model, &mut cache, capacity.max(1))?;
    if dflash.is_none()
        && let Some((&last, prefix)) = prompt_tokens.split_last()
    {
        for &prompt_token in prefix {
            model.consume_one(&mut sequence, prompt_token, &mut cache)?;
        }
        token = last;
        println!("prompt_tokens={} {prompt_tokens:?}", prompt_tokens.len());
    }

    let decode_started = Instant::now();
    if dflash.is_some() {
        let target_prompt = if prompt_tokens.is_empty() {
            std::slice::from_ref(&token)
        } else {
            prompt_tokens.as_slice()
        };
        for (index, chunk) in target_prompt.chunks(16).enumerate() {
            let output_logits = (index + 1) * 16 >= target_prompt.len();
            model.dflash_prefill_chunk(&mut sequence, chunk, output_logits, &mut cache)?;
        }
        println!("prompt_tokens={} {target_prompt:?}", target_prompt.len());
        let (mut anchor, _) = model.argmax_with_logit(&mut sequence)?;
        let mut generated: Vec<u32> = Vec::with_capacity(tokens as usize);
        while generated.len() < tokens as usize {
            let cycle_started = Instant::now();
            let cycle = model.dflash_cycle(&mut sequence, anchor, &mut cache)?;
            let remaining = tokens as usize - generated.len();
            generated.extend(cycle.tokens.iter().take(remaining).copied());
            println!(
                "dflash: accepted={}/{} emitted={} next={} ms={:.3}",
                cycle.accepted_drafts,
                cycle.drafted_tokens,
                cycle.tokens.len().min(remaining),
                cycle.next_token,
                cycle_started.elapsed().as_secs_f64() * 1_000.0,
            );
            anchor = cycle.next_token;
        }
        let elapsed = decode_started.elapsed();
        println!("tokens={generated:?}");
        println!(
            "tokens={} decode_s={:.3} decode_tok_s={:.3}",
            generated.len(),
            elapsed.as_secs_f64(),
            generated.len() as f64 / elapsed.as_secs_f64(),
        );
        sequence.finish(&model, &mut cache)?;
        return Ok(());
    }
    let mut generated = Vec::with_capacity(tokens as usize);
    for step in 0..tokens {
        let token_started = Instant::now();
        let next = model.decode_one(&mut sequence, token, &mut cache)?;
        println!(
            "decode {step:03}: {token} -> {} (logit {:.6}) ms={:.3}",
            next.token,
            next.logit,
            token_started.elapsed().as_secs_f64() * 1_000.0,
        );
        token = next.token;
        generated.push(token);
        if let Some(tokenizer) = &tokenizer {
            let text = tokenizer
                .decode(&generated, true)
                .map_err(|error| Error::Format {
                    label: "Muse Glimmer decode",
                    detail: error.to_string(),
                })?;
            println!("text={text:?}");
        }
    }
    let elapsed = decode_started.elapsed();
    println!(
        "tokens={tokens} decode_s={:.3} decode_tok_s={:.3}",
        elapsed.as_secs_f64(),
        tokens as f64 / elapsed.as_secs_f64(),
    );
    sequence.finish(&model, &mut cache)?;
    Ok(())
}

fn parse(program: &str, value: std::ffi::OsString, label: &'static str) -> Result<u32> {
    value
        .into_string()
        .map_err(|_| Error::Format {
            label: "usage",
            detail: format!("{program}: {label} must be UTF-8"),
        })?
        .parse()
        .map_err(|error| Error::Format {
            label: "usage",
            detail: format!("{program}: invalid {label}: {error}"),
        })
}
