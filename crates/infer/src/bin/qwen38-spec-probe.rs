//! Measures the Qwen3.8 speculative cycle end to end against the canonical
//! greedy stream.
//!
//! Each cycle verifies `[frontier, drafts..]` in one batched forward pass,
//! commits the longest accepted prefix, and advances the frontier to the
//! target's argmax. The emitted stream must be byte-identical to the canonical
//! greedy decode because every committed token is the target's own argmax at
//! that position. This probe reports decode throughput for both paths and the
//! first divergence, if any.

use infer::nvfp4::{CudaStream, DeviceBuffer, Error, Result};
use infer::qwen3::qwen36::{
    Qwen36Bf16Storage, Qwen36Bf16StorageConfig, Qwen36DecodeRow, Qwen36Fp8AttentionStorage,
    Qwen36PrefillRow, Qwen36SpeculativeFrontier, Qwen36TextModel,
};
use infer::runtime::qwen36_sequence::{Qwen36Sequence, new_qwen36_sequence_cache};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model_dir: PathBuf,
    prompt: String,
    prompt_file: Option<PathBuf>,
    tokens: usize,
    drafts: usize,
}

struct SpecReport {
    tokens: Vec<u32>,
    elapsed: f64,
    cycles: usize,
    replayed: usize,
    total_accepted: usize,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let prompt = match &args.prompt_file {
        Some(path) => std::fs::read_to_string(path).map_err(|err| Error::Format {
            label: "prompt file",
            detail: format!("{}: {err}", path.display()),
        })?,
        None => args.prompt.clone(),
    };
    infer::nvfp4::set_cuda_device(0)?;
    let model = Qwen36TextModel::open_with_storage(
        &args.model_dir,
        Qwen36Bf16StorageConfig::new(Qwen36Bf16Storage::Bf16, Qwen36Bf16Storage::Bf16),
        Qwen36Fp8AttentionStorage::Fp8,
    )?;
    let manifest = model.manifest();
    println!(
        "qwen38 spec probe: layers={} hidden={} vocab={} mtp={}",
        manifest.layers,
        manifest.hidden,
        manifest.vocab,
        model.mtp_weights().is_some()
    );
    if model.mtp_weights().is_none() {
        return Err(Error::Format {
            label: "qwen3.8 spec probe",
            detail: "checkpoint has no MTP weights".to_string(),
        });
    }
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let prompt_ids = encode_prompt(&tokenizer, &prompt)?;
    if prompt_ids.len() < 3 {
        return Err(Error::Format {
            label: "qwen3.8 spec probe",
            detail: "prompt must contain at least three tokens".to_string(),
        });
    }
    let max_tokens = (prompt_ids.len() + args.tokens + args.drafts + 16).div_ceil(128) * 128;

    let (reference, canonical_elapsed) =
        run_canonical(&model, &prompt_ids, args.tokens, max_tokens)?;
    let spec = run_speculative(&model, &prompt_ids, args.tokens, args.drafts, max_tokens)?;

    let compared = args.tokens.min(spec.tokens.len());
    let mut first_divergence = None;
    for (index, &canonical) in reference.iter().take(compared).enumerate() {
        if canonical != spec.tokens[index] {
            first_divergence = Some(index);
            break;
        }
    }
    println!(
        "canonical: {:.3} tok/s ({} tokens in {:.3}s)",
        args.tokens as f64 / canonical_elapsed.max(1e-6),
        args.tokens,
        canonical_elapsed
    );
    println!(
        "speculative (drafts={}): {:.3} tok/s ({} tokens in {:.3}s)",
        args.drafts,
        spec.tokens.len() as f64 / spec.elapsed.max(1e-6),
        spec.tokens.len(),
        spec.elapsed
    );
    println!(
        "  cycles={} replayed={} accepted_drafts={} (avg {:.3}/cycle)",
        spec.cycles,
        spec.replayed,
        spec.total_accepted,
        spec.total_accepted as f64 / spec.cycles.max(1) as f64
    );
    match first_divergence {
        None => println!("byte-identical: yes ({} tokens)", compared),
        Some(index) => {
            println!(
                "byte-identical: no (first divergence at token {index}: canonical={} speculative={})",
                reference[index], spec.tokens[index]
            );
        }
    }
    Ok(())
}

fn run_canonical(
    model: &Qwen36TextModel,
    prompt_ids: &[u32],
    tokens: usize,
    max_tokens: usize,
) -> Result<(Vec<u32>, f64)> {
    let stream = CudaStream::new_non_blocking()?;
    let mut cache = new_qwen36_sequence_cache(model, 1, max_tokens)?;
    let mut sequence = Qwen36Sequence::admit(model, &mut cache, max_tokens, &stream)?;
    let mut prefill =
        model.new_prefill_batch_workspace(1, prompt_ids.len().max(256), max_tokens)?;
    let mut decode = model.new_decode_batch_workspace(1, max_tokens)?;
    {
        let mut rows = [Qwen36PrefillRow {
            token_ids: &prompt_ids[..prompt_ids.len() - 1],
            sequence: &mut sequence,
        }];
        model.prefill_batch(&mut prefill, &mut rows, &mut cache)?;
    }
    let mut out = Vec::with_capacity(tokens);
    let mut next = *prompt_ids.last().expect("non-empty prompt");
    let start = Instant::now();
    for _ in 0..tokens {
        let mut rows = [Qwen36DecodeRow {
            token_id: next,
            sequence: &mut sequence,
        }];
        let mut decoded = model.decode_batch(&mut decode, &mut rows, &mut cache)?;
        let token = decoded.top1()?.into_iter().next().expect("one row").id;
        out.push(token);
        next = token;
    }
    Ok((out, start.elapsed().as_secs_f64()))
}

fn run_speculative(
    model: &Qwen36TextModel,
    prompt_ids: &[u32],
    tokens: usize,
    drafts: usize,
    max_tokens: usize,
) -> Result<SpecReport> {
    let stream = CudaStream::new_blocking()?;
    let hidden = model.manifest().hidden;
    let mut cache = new_qwen36_sequence_cache(model, 1, max_tokens)?;
    let mut sequence = Qwen36Sequence::admit(model, &mut cache, max_tokens, &stream)?;
    let mut prefill =
        model.new_prefill_batch_workspace(1, prompt_ids.len().max(256), max_tokens)?;
    let mut spec_workspace = model.new_speculative_cycle_workspace(drafts, max_tokens)?;
    let mut mtp_state = model.new_mtp_sequence_state(max_tokens)?;

    {
        let mut rows = [Qwen36PrefillRow {
            token_ids: &prompt_ids[..prompt_ids.len() - 1],
            sequence: &mut sequence,
        }];
        model.prefill_batch(&mut prefill, &mut rows, &mut cache)?;
    }
    {
        let mut mtp_workspace = model.new_mtp_draft_workspace(max_tokens)?;
        model.mtp_warmup_kv(
            &mut mtp_state,
            &mut mtp_workspace,
            &prompt_ids[..prompt_ids.len() - 1],
            prefill.prompt_hidden(),
            &stream,
        )?;
    }
    stream.synchronize()?;

    let mut frontier = Qwen36SpeculativeFrontier {
        token: *prompt_ids.last().expect("non-empty prompt"),
        prev_hidden: DeviceBuffer::zeroed(hidden)?,
    };
    frontier.prev_hidden.copy_range_from_device_on_stream(
        0,
        prefill.prompt_hidden(),
        (prompt_ids.len() - 2) * hidden,
        hidden,
        &stream,
    )?;
    stream.synchronize()?;

    let mut emitted = Vec::with_capacity(tokens);
    let mut cycles = 0usize;
    let mut replayed = 0usize;
    let mut total_accepted = 0usize;
    let mut first = true;
    let start = Instant::now();
    while emitted.len() < tokens {
        let outcome = model.speculative_cycle_argmax(
            &mut spec_workspace,
            &mut frontier,
            &mut sequence,
            &mut mtp_state,
            &mut cache,
        )?;
        let skip = if first { 1 } else { 0 };
        emitted.extend_from_slice(&outcome.committed[skip..]);
        cycles += 1;
        total_accepted += outcome.accepted_drafts;
        if outcome.replayed {
            replayed += 1;
        }
        first = false;
    }
    Ok(SpecReport {
        tokens: emitted,
        elapsed: start.elapsed().as_secs_f64(),
        cycles,
        replayed,
        total_accepted,
    })
}

fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|err| Error::Format {
        label: "tokenizer.json",
        detail: format!("{}: {err}", tokenizer_path.display()),
    })
}

fn encode_prompt(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(prompt, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|err| Error::Format {
            label: "tokenize prompt",
            detail: err.to_string(),
        })
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model_dir: PathBuf::new(),
        prompt: "Write a small rust program that parses shell-like quoted arguments, then explain the parsing rules.".to_string(),
        prompt_file: None,
        tokens: 192,
        drafts: 2,
    };
    let mut iter = env::args().skip(1);
    let mut model_dir = None;
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--prompt" => args.prompt = iter.next().ok_or_else(|| usage("--prompt value"))?,
            "--prompt-file" => {
                args.prompt_file = Some(PathBuf::from(
                    iter.next().ok_or_else(|| usage("--prompt-file path"))?,
                ));
            }
            "--tokens" => args.tokens = parse(&iter.next(), "--tokens")?,
            "--drafts" => args.drafts = parse(&iter.next(), "--drafts")?,
            _ if model_dir.is_none() && !arg.starts_with('-') => {
                model_dir = Some(PathBuf::from(&arg));
            }
            _ => return Err(usage(&arg)),
        }
    }
    args.model_dir = model_dir.ok_or_else(|| usage("<model-dir>"))?;
    if args.drafts == 0 {
        return Err(usage("--drafts must be positive"));
    }
    Ok(args)
}

fn parse<T>(value: &Option<String>, label: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .as_deref()
        .ok_or_else(|| usage(label))?
        .parse::<T>()
        .map_err(|err| Error::Format {
            label: "probe argument value",
            detail: format!("{label}: {err}"),
        })
}

fn usage(arg: &str) -> Error {
    Error::Format {
        label: "usage",
        detail: format!(
            "qwen38-spec-probe <model-dir> [--prompt text | --prompt-file path] [--tokens n] \
             [--drafts k]; unexpected {arg}"
        ),
    }
}
