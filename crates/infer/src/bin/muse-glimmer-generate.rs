//! Minimal Muse Glimmer load and greedy-decode probe.

use infer::muse_glimmer::MuseGlimmerModel;
use infer::nvfp4::{Error, Result};
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

    let load_started = Instant::now();
    let model = MuseGlimmerModel::load(&model_dir)?;
    println!(
        "loaded Muse Glimmer in {:.3}s weights_gib={:.3}",
        load_started.elapsed().as_secs_f64(),
        model.device_bytes() as f64 / (1u64 << 30) as f64,
    );
    let mut state = model.new_decode_state(prompt_tokens.len() + tokens as usize)?;
    if let Some((&last, prefix)) = prompt_tokens.split_last() {
        for &prompt_token in prefix {
            model.consume_one(&mut state, prompt_token)?;
        }
        token = last;
        println!("prompt_tokens={} {prompt_tokens:?}", prompt_tokens.len());
    }

    let decode_started = Instant::now();
    let mut generated = Vec::with_capacity(tokens as usize);
    for step in 0..tokens {
        let token_started = Instant::now();
        let next = model.decode_one(&mut state, token)?;
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
