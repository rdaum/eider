//! Compares Flash Next serial decode with its two-row target verifier.

use eider_cuda::{Error, Result};
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use infer::qwen38_flash_next::{
    Qwen38FlashNextModel, Qwen38VectorVerifierProbeMode, probe_verification_paths,
};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model_dir: PathBuf,
    artifact_dir: PathBuf,
    prompt: String,
    prompt_file: Option<PathBuf>,
    cycles: usize,
    prefill_tokens: usize,
    modes: Vec<Qwen38VectorVerifierProbeMode>,
    trace_layers: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let prompt = match &args.prompt_file {
        Some(path) => std::fs::read_to_string(path).map_err(|error| Error::Format {
            label: "Qwen3.8 Flash Next probe prompt",
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
            label: "Qwen3.8 Flash Next probe prompt",
            detail: "chat template produced no tokens".to_string(),
        });
    }

    let load_started = Instant::now();
    let mut model = Qwen38FlashNextModel::open(&args.model_dir, args.artifact_dir)?;
    eprintln!(
        "loaded Flash Next in {:.2}s; prompt_tokens={} cycles={} prefill_tokens={}",
        load_started.elapsed().as_secs_f64(),
        rendered.token_ids.len(),
        args.cycles,
        args.prefill_tokens
    );
    for mode in args.modes {
        println!("mode: {}", mode_name(mode));
        let report = probe_verification_paths(
            &mut model,
            &rendered.token_ids,
            args.cycles,
            args.prefill_tokens,
            mode,
            args.trace_layers,
        )?;

        println!(
            "argmax agreement: {}/{} ({:.2}%)",
            report.matching_rows,
            report.compared_rows,
            100.0 * report.matching_rows as f64 / report.compared_rows as f64
        );
        println!(
            "serial decode: {:.3} tokens/sec ({:.3}s)",
            report.serial_tokens_per_second(),
            report.serial_duration.as_secs_f64()
        );
        println!(
            "two-row verifier: {:.3} tokens/sec ({:.3}s)",
            report.verification_tokens_per_second(),
            report.verification_duration.as_secs_f64()
        );
        println!(
            "worst residual difference: max_abs={:.6} cosine={:.9} relative_rmse={:.9}",
            report.worst_stream_difference.maximum_absolute_error,
            report.worst_stream_difference.cosine_similarity,
            report.worst_stream_difference.relative_rmse
        );
        match report.first_mismatch {
            Some(mismatch) => {
                let input = decode_token(template.tokenizer(), mismatch.input_token)?;
                let serial = decode_token(template.tokenizer(), mismatch.serial.id)?;
                let verification = decode_token(template.tokenizer(), mismatch.verification.id)?;
                println!(
                    "first divergence: cycle={} row={} output_index={} input={} {:?} serial={} {:?} verifier={} {:?}",
                    mismatch.cycle,
                    mismatch.row,
                    mismatch.output_index,
                    mismatch.input_token,
                    input,
                    mismatch.serial.id,
                    serial,
                    mismatch.verification.id,
                    verification
                );
            }
            None => println!(
                "first divergence: none across {} target rows",
                report.compared_rows
            ),
        }
        if let Some(divergence) = report.first_layer_divergence {
            println!(
                "first layer-state divergence: cycle={} row={} layer={} stage={} component={} mismatched_values={} max_abs={:.9} cosine={:.12} relative_rmse={:.12}",
                divergence.cycle,
                divergence.row,
                divergence.layer,
                divergence.stage,
                divergence.moe_component.unwrap_or("unclassified"),
                divergence.mismatched_values,
                divergence.difference.maximum_absolute_error,
                divergence.difference.cosine_similarity,
                divergence.difference.relative_rmse,
            );
        }
    }
    Ok(())
}

fn mode_name(mode: Qwen38VectorVerifierProbeMode) -> &'static str {
    match mode {
        Qwen38VectorVerifierProbeMode::Fast => "fast",
        Qwen38VectorVerifierProbeMode::SerialGdnProjections => "serial-gdn",
        Qwen38VectorVerifierProbeMode::CanonicalMoeLinears => "canonical-moe-linears",
        Qwen38VectorVerifierProbeMode::CanonicalMoeLinearsSerialGdn => {
            "canonical-moe-linears-serial-gdn"
        }
        Qwen38VectorVerifierProbeMode::ExactMoe => "exact-moe",
        Qwen38VectorVerifierProbeMode::Exact => "exact",
    }
}

fn decode_token(tokenizer: &tokenizers::Tokenizer, token: u32) -> Result<String> {
    tokenizer
        .decode(&[token], false)
        .map_err(|error| Error::Format {
            label: "Qwen3.8 Flash Next probe token decode",
            detail: error.to_string(),
        })
}

fn parse_args() -> Result<Args> {
    let mut model_dir = None;
    let mut artifact_dir = None;
    let mut prompt = "Review a Rust inference runtime for speculative-decoding correctness. \
        Trace sequence state, cache transactions, target verification, recurrent attention, \
        mixture-of-experts execution, and vocabulary selection. Identify concrete invariants, \
        distinguish numerical drift from a state-management defect, and recommend the smallest \
        reliable discriminator before changing production code. Explain how the test keeps both \
        target histories identical. Repeat the audit across several token classes and inspect the \
        first divergence instead of relying only on aggregate acceptance. "
        .repeat(4);
    let mut prompt_file = None;
    let mut cycles = 32usize;
    let mut prefill_tokens = 64usize;
    let mut modes = vec![Qwen38VectorVerifierProbeMode::Fast];
    let mut trace_layers = false;
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
            "--prefill-tokens" => {
                prefill_tokens = parse_value(&mut iter, "--prefill-tokens")?;
            }
            "--mode" => modes = parse_modes(&next_arg(&mut iter, "--mode")?)?,
            "--trace-layers" => trace_layers = true,
            _ if model_dir.is_none() && !arg.starts_with('-') => {
                model_dir = Some(PathBuf::from(arg));
            }
            _ => return Err(usage(&arg)),
        }
    }
    let model_dir = model_dir.ok_or_else(|| usage("<model-dir>"))?;
    let artifact_dir = artifact_dir.unwrap_or(default_artifact_dir()?);
    if cycles == 0 || prefill_tokens == 0 {
        return Err(usage("cycle and prefill values must be positive"));
    }
    Ok(Args {
        model_dir,
        artifact_dir,
        prompt,
        prompt_file,
        cycles,
        prefill_tokens,
        modes,
        trace_layers,
    })
}

fn parse_modes(value: &str) -> Result<Vec<Qwen38VectorVerifierProbeMode>> {
    let modes = match value {
        "fast" => vec![Qwen38VectorVerifierProbeMode::Fast],
        "serial-gdn" => vec![Qwen38VectorVerifierProbeMode::SerialGdnProjections],
        "canonical-moe-linears" => vec![Qwen38VectorVerifierProbeMode::CanonicalMoeLinears],
        "canonical-moe-linears-serial-gdn" => {
            vec![Qwen38VectorVerifierProbeMode::CanonicalMoeLinearsSerialGdn]
        }
        "exact-moe" => vec![Qwen38VectorVerifierProbeMode::ExactMoe],
        "exact" => vec![Qwen38VectorVerifierProbeMode::Exact],
        "all" => vec![
            Qwen38VectorVerifierProbeMode::Fast,
            Qwen38VectorVerifierProbeMode::SerialGdnProjections,
            Qwen38VectorVerifierProbeMode::CanonicalMoeLinears,
            Qwen38VectorVerifierProbeMode::CanonicalMoeLinearsSerialGdn,
            Qwen38VectorVerifierProbeMode::ExactMoe,
            Qwen38VectorVerifierProbeMode::Exact,
        ],
        _ => return Err(usage(value)),
    };
    Ok(modes)
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
            label: "Qwen3.8 Flash Next probe argument",
            detail: format!("{label}: {error}"),
        })
}

fn default_artifact_dir() -> Result<PathBuf> {
    let root = if let Some(path) = env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(path)
    } else {
        let home = env::var_os("HOME").ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next probe artifact directory",
            detail: "HOME and XDG_CACHE_HOME are unset".to_string(),
        })?;
        PathBuf::from(home).join(".cache")
    };
    Ok(root.join("eider/qwen38-flash-next-probe"))
}

fn usage(unexpected: &str) -> Error {
    Error::Format {
        label: "usage",
        detail: format!(
            "qwen38-flash-next-spec-probe <model-dir> [--artifact-dir path] \
             [--prompt text | --prompt-file path] [--cycles n] [--prefill-tokens n]; \
             [--mode fast|serial-gdn|canonical-moe-linears|\
             canonical-moe-linears-serial-gdn|exact-moe|exact|all]; \
             [--trace-layers]; \
             unexpected {unexpected}"
        ),
    }
}
