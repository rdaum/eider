//! Compares native Flash Next MTP cycles with canonical target decode.

use eider_cuda::{Error, Result};
use eider_inference::qwen38_flash_next::{Qwen38FlashNextModel, probe_speculative_cycles};
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model_dir: PathBuf,
    artifact_dir: PathBuf,
    prompt: String,
    prompt_file: Option<PathBuf>,
    cycles: usize,
    drafts: Vec<usize>,
    prefill_tokens: usize,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let prompt = match &args.prompt_file {
        Some(path) => std::fs::read_to_string(path).map_err(|error| Error::Format {
            label: "Qwen3.8 Flash Next MTP probe prompt",
            detail: format!("{}: {error}", path.display()),
        })?,
        None => args.prompt,
    };
    let template = CheckpointChatTemplate::from_model_dir(&args.model_dir)?;
    let rendered = template.render_and_tokenize(
        &[ChatMessage::user(prompt)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    if rendered.token_ids.is_empty() {
        return Err(Error::Format {
            label: "Qwen3.8 Flash Next MTP probe prompt",
            detail: "chat template produced no tokens".to_string(),
        });
    }

    let load_started = Instant::now();
    let mut model = Qwen38FlashNextModel::open(&args.model_dir, args.artifact_dir)?;
    model.enable_mtp()?;
    eprintln!(
        "loaded Flash Next with MTP in {:.2}s; prompt_tokens={} cycles={} drafts={:?} prefill_tokens={}",
        load_started.elapsed().as_secs_f64(),
        rendered.token_ids.len(),
        args.cycles,
        args.drafts,
        args.prefill_tokens,
    );

    for drafts in args.drafts {
        let report = probe_speculative_cycles(
            &mut model,
            &rendered.token_ids,
            args.cycles,
            drafts,
            args.prefill_tokens,
        )?;
        println!("drafts: {}", report.drafts);
        println!(
            "committed: {} tokens in {} cycles",
            report.committed_tokens, report.cycles
        );
        println!(
            "accepted drafts: {} ({:.3}/cycle)",
            report.accepted_drafts,
            report.accepted_drafts_per_cycle()
        );
        println!(
            "canonical: {:.3} tokens/sec ({:.3}s)",
            report.canonical_tokens_per_second(),
            report.canonical_duration.as_secs_f64()
        );
        println!(
            "speculative: {:.3} tokens/sec ({:.3}s)",
            report.speculative_tokens_per_second(),
            report.speculative_duration.as_secs_f64()
        );
        match report.first_token_mismatch {
            Some(mismatch) => println!(
                "token mismatch: cycle={} output={} expected={} actual={}",
                mismatch.cycle, mismatch.output_index, mismatch.expected.id, mismatch.actual.id,
            ),
            None => println!("token mismatch: none"),
        }
        match report.first_state_mismatch {
            Some((cycle, component)) => {
                println!("target-state mismatch: cycle={cycle} component={component}")
            }
            None => println!("target-state mismatch: none"),
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut model_dir = None;
    let mut artifact_dir = None;
    let mut prompt = "Review a Rust inference runtime for speculative-decoding correctness. \
        Trace sequence state, cache transactions, target verification, recurrent attention, \
        mixture-of-experts execution, and vocabulary selection. Identify concrete invariants, \
        distinguish numerical drift from a state-management defect, and recommend the smallest \
        reliable discriminator before changing production code. "
        .repeat(4);
    let mut prompt_file = None;
    let mut cycles = 4usize;
    let mut drafts = vec![1usize, 2, 4];
    let mut prefill_tokens = 64usize;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--artifact-dir" => {
                artifact_dir = Some(PathBuf::from(next_arg(&mut iter, "--artifact-dir")?));
            }
            "--prompt" => prompt = next_arg(&mut iter, "--prompt")?,
            "--prompt-file" => {
                prompt_file = Some(PathBuf::from(next_arg(&mut iter, "--prompt-file")?));
            }
            "--cycles" => cycles = parse_value(&mut iter, "--cycles")?,
            "--drafts" => drafts = parse_list(&next_arg(&mut iter, "--drafts")?)?,
            "--prefill-tokens" => {
                prefill_tokens = parse_value(&mut iter, "--prefill-tokens")?;
            }
            _ if model_dir.is_none() && !arg.starts_with('-') => {
                model_dir = Some(PathBuf::from(arg));
            }
            _ => return Err(usage(&arg)),
        }
    }
    let model_dir = model_dir.ok_or_else(|| usage("<model-dir>"))?;
    let artifact_dir = artifact_dir.unwrap_or(default_artifact_dir()?);
    if cycles == 0 || drafts.is_empty() || drafts.contains(&0) || prefill_tokens == 0 {
        return Err(usage("cycle, draft, and prefill values must be positive"));
    }
    Ok(Args {
        model_dir,
        artifact_dir,
        prompt,
        prompt_file,
        cycles,
        drafts,
        prefill_tokens,
    })
}

fn parse_list(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(|item| {
            item.parse::<usize>().map_err(|error| Error::Format {
                label: "Qwen3.8 Flash Next MTP probe drafts",
                detail: format!("{item}: {error}"),
            })
        })
        .collect()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, label: &str) -> Result<String> {
    iter.next().ok_or_else(|| usage(label))
}

fn parse_value<T>(iter: &mut impl Iterator<Item = String>, label: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    next_arg(iter, label)?
        .parse::<T>()
        .map_err(|error| Error::Format {
            label: "Qwen3.8 Flash Next MTP probe argument",
            detail: format!("{label}: {error}"),
        })
}

fn default_artifact_dir() -> Result<PathBuf> {
    let root = if let Some(path) = env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(path)
    } else {
        let home = env::var_os("HOME").ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next MTP probe artifact directory",
            detail: "HOME and XDG_CACHE_HOME are unset".to_string(),
        })?;
        PathBuf::from(home).join(".cache")
    };
    Ok(root.join("eider/qwen38-flash-next-mtp-probe"))
}

fn usage(unexpected: &str) -> Error {
    Error::Format {
        label: "usage",
        detail: format!(
            "qwen38-flash-next-mtp-probe <model-dir> [--artifact-dir path] \
             [--prompt text | --prompt-file path] [--cycles n] [--drafts 1,2,4] \
             [--prefill-tokens n]; unexpected {unexpected}"
        ),
    }
}
