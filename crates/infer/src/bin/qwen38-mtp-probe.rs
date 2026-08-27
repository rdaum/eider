//! Measures Qwen3.8 MTP draft acceptance against the canonical target stream.
//!
//! Each cycle presents the drafter with one committed-but-unprocessed frontier
//! token, drafts `k` chained MTP tokens, then advances the target canonically
//! far enough to score the whole draft chain and re-anchor the next cycle.
//! Catch-up rewrites the drafter K/V slots covered by the new committed
//! tokens, so the drafter always drafts from a faithful frontier.
//!
//! Acceptance is the agreement between the drafter's argmax and the target's
//! own sampled/greedy choice at each position, which is exactly the prefix a
//! speculative verifier would commit. Both regimes are measured from the same
//! prompt in separate passes.

use infer::nvfp4::{CudaStream, DeviceBuffer, Error, Result};
use infer::qwen3::qwen36::{
    Qwen36Bf16Storage, Qwen36Bf16StorageConfig, Qwen36DecodeBatchWorkspace, Qwen36DecodeRow,
    Qwen36Fp8Storage, Qwen36PrefillRow, Qwen36TextModel,
};
use infer::qwen3::qwen36::{Qwen36Sequence, Qwen36SequenceCache, new_qwen36_sequence_cache};
use infer::runtime::sampling::{Sampler, SamplingConfig, TokenHistory};
use std::env;
use std::path::{Path, PathBuf};

struct Args {
    model_dir: PathBuf,
    prompt: String,
    prompt_file: Option<PathBuf>,
    tokens: usize,
    drafts: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    skip_sampled: bool,
}

struct PassReport {
    label: String,
    committed: usize,
    cycles: usize,
    accepted: Vec<usize>,
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
        Qwen36Fp8Storage::Fp8,
    )?;
    let manifest = model.manifest();
    println!(
        "qwen38 mtp probe: layers={} hidden={} vocab={} mtp={}",
        manifest.layers,
        manifest.hidden,
        manifest.vocab,
        model.mtp_weights().is_some()
    );
    if model.mtp_weights().is_none() {
        return Err(Error::Format {
            label: "qwen3.8 mtp probe",
            detail: "checkpoint has no MTP weights".to_string(),
        });
    }
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let prompt_ids = encode_prompt(&tokenizer, &prompt)?;
    if prompt_ids.len() < 3 {
        return Err(Error::Format {
            label: "qwen3.8 mtp probe",
            detail: "prompt must contain at least three tokens".to_string(),
        });
    }
    let max_tokens = (prompt_ids.len() + args.tokens + args.drafts + 16).div_ceil(128) * 128;

    let greedy = run_pass(&model, &prompt_ids, None, &args, max_tokens)?;
    print_report(&greedy);
    if !args.skip_sampled {
        let sampling = SamplingConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: Some(args.seed),
        };
        let sampled = run_pass(&model, &prompt_ids, Some(sampling), &args, max_tokens)?;
        print_report(&sampled);
    }
    Ok(())
}

fn run_pass(
    model: &Qwen36TextModel,
    prompt_ids: &[u32],
    sampling: Option<SamplingConfig>,
    args: &Args,
    max_tokens: usize,
) -> Result<PassReport> {
    let stream = CudaStream::new_non_blocking()?;
    let mut cache = new_qwen36_sequence_cache(model, 1, max_tokens)?;
    let mut sequence = Qwen36Sequence::admit(model, &mut cache, max_tokens, &stream)?;
    let hidden = model.manifest().hidden;
    let mut prefill =
        model.new_prefill_batch_workspace(1, prompt_ids.len().max(256), max_tokens)?;
    let mut decode = model.new_decode_batch_workspace(1, max_tokens)?;
    let mut mtp_state = model.new_mtp_sequence_state(max_tokens)?;
    let mut mtp_workspace = model.new_mtp_draft_workspace(max_tokens)?;
    let mut mtp_hidden_scratch = DeviceBuffer::zeroed(hidden)?;
    let mut history = TokenHistory::from_tokens(prompt_ids.iter().copied());
    let mut sampler = sampling
        .map(|config| Sampler::new(config).map(|sampler| (sampler, Vec::new())))
        .transpose()?;

    // Prefill every prompt token except the last. The last prompt token is the
    // first cycle's frontier: committed but not yet processed by the target.
    {
        let mut rows = [Qwen36PrefillRow {
            token_ids: &prompt_ids[..prompt_ids.len() - 1],
            sequence: &mut sequence,
        }];
        model.prefill_batch(&mut prefill, &mut rows, &mut cache)?;
    }

    // Warm the drafter K/V over prompt slots 0..P-2: slot j pairs prompt[j]
    // with the target hidden at position j - 1 (zeros at position 0), so the
    // warmup hiddens are the prefill hidden rows themselves.
    model.mtp_warmup_kv(
        &mut mtp_state,
        &mut mtp_workspace,
        &mut mtp_hidden_scratch,
        &prompt_ids[..prompt_ids.len() - 1],
        prefill.prompt_hidden(),
        0,
        &DeviceBuffer::zeroed(model.manifest().hidden)?,
        prefill.stream(),
    )?;
    prefill.stream().synchronize()?;

    let mut committed = 0usize;
    let mut cycles = 0usize;
    let mut accepted = vec![0usize; args.drafts];

    // Cycle entry state: `frontier` token at position P-1, with the hidden
    // produced while processing its predecessor (the last prefill row).
    let mut frontier_token = *prompt_ids.last().expect("non-empty prompt");
    let mut frontier_prev_hidden = DeviceBuffer::zeroed(hidden)?;
    frontier_prev_hidden.copy_range_from_device_on_stream(
        0,
        prefill.prompt_hidden(),
        (prompt_ids.len() - 2) * hidden,
        hidden,
        &stream,
    )?;

    while committed < args.tokens {
        cycles += 1;
        // A. Draft k tokens from the drafter frontier.
        let drafts = model.mtp_draft_chain_argmax(
            &mut mtp_state,
            &mut mtp_workspace,
            &mut mtp_hidden_scratch,
            frontier_token,
            &frontier_prev_hidden,
            args.drafts,
            &stream,
        )?;
        // B. Canonical target step for the frontier token, producing both its
        // hidden and the first sampled continuation.
        let (w, mut step_hidden) = decode_step(
            model,
            &mut decode,
            &mut sequence,
            &mut cache,
            frontier_token,
            sampler.as_mut(),
            &mut history,
            hidden,
            &stream,
        )?;
        committed += 1;
        // C. Lookahead: k+1 further canonical steps. `steps[j]` pairs sampled
        // token t_j with the hidden produced while processing t_{j-1}.
        let mut steps: Vec<(u32, DeviceBuffer<f32>)> = Vec::with_capacity(args.drafts + 1);
        let mut input = w;
        for _ in 0..=args.drafts {
            let (token, hidden_after) = decode_step(
                model,
                &mut decode,
                &mut sequence,
                &mut cache,
                input,
                sampler.as_mut(),
                &mut history,
                hidden,
                &stream,
            )?;
            input = token;
            committed += 1;
            steps.push((token, hidden_after));
        }
        // D. Score the draft chain against the committed stream w, t_0, t_1, ..
        let mut matches_prefix = true;
        for (position, accepted_count) in accepted.iter_mut().enumerate() {
            let actual = if position == 0 {
                w
            } else {
                steps[position - 1].0
            };
            matches_prefix = matches_prefix && drafts[position] == actual;
            if matches_prefix {
                *accepted_count += 1;
            }
        }
        // E. Catch up drafter K/V over the newly committed tokens. The draft
        // chain wrote k slots starting at the frontier slot; keeping the
        // frontier slot (written with the correct pair) requires retaining
        // len - drafts + 1 rows.
        mtp_state.truncate(mtp_state.len() - args.drafts + 1)?;
        model.mtp_append_kv(&mut mtp_state, &mut mtp_workspace, w, &step_hidden, &stream)?;
        for (token, hidden_after_previous) in steps.iter().take(args.drafts) {
            model.mtp_append_kv(
                &mut mtp_state,
                &mut mtp_workspace,
                *token,
                hidden_after_previous,
                &stream,
            )?;
        }
        // F. Next frontier: the last sampled token with its step's hidden.
        step_hidden = steps
            .last()
            .map(|(_, hidden)| hidden)
            .expect("lookahead steps")
            .clone_via(&stream)?;
        frontier_token = steps.last().expect("lookahead steps").0;
        frontier_prev_hidden = step_hidden;
    }
    Ok(PassReport {
        label: sampling_label(sampler.as_ref().map(|(sampler, _)| sampler.config())),
        committed,
        cycles,
        accepted,
    })
}

/// Runs one canonical target decode: processes `input_token`, samples the next
/// token (argmax when `sampler` is `None`), and returns the sampled token plus
/// the pre-final-norm hidden produced by this step.
#[allow(clippy::too_many_arguments)]
fn decode_step(
    model: &Qwen36TextModel,
    workspace: &mut Qwen36DecodeBatchWorkspace,
    sequence: &mut Qwen36Sequence,
    cache: &mut Qwen36SequenceCache,
    input_token: u32,
    sampler: Option<&mut (Sampler, Vec<f32>)>,
    history: &mut TokenHistory,
    hidden: usize,
    stream: &CudaStream,
) -> Result<(u32, DeviceBuffer<f32>)> {
    let mut rows = [Qwen36DecodeRow {
        token_id: input_token,
        sequence,
    }];
    let mut decoded = model.decode_batch(workspace, &mut rows, cache)?;
    let mut step_hidden = DeviceBuffer::zeroed(hidden)?;
    step_hidden.copy_prefix_from_device_on_stream(decoded.hidden(), hidden, stream)?;
    let token = match sampler {
        Some((sampler, logits)) => {
            logits.resize(decoded.vocab(), 0.0);
            logits.copy_from_slice(&decoded.copy_logits()?[..decoded.vocab()]);
            sampler.sample(logits, history)?.id
        }
        None => decoded.top1()?.into_iter().next().expect("one row").id,
    };
    history.push(token);
    Ok((token, step_hidden))
}

trait CloneVia {
    fn clone_via(&self, stream: &CudaStream) -> Result<DeviceBuffer<f32>>;
}

impl CloneVia for DeviceBuffer<f32> {
    fn clone_via(&self, stream: &CudaStream) -> Result<DeviceBuffer<f32>> {
        let mut cloned = DeviceBuffer::zeroed(self.len())?;
        cloned.copy_prefix_from_device_on_stream(self, self.len(), stream)?;
        Ok(cloned)
    }
}

fn sampling_label(config: Option<SamplingConfig>) -> String {
    match config {
        Some(config) => format!(
            "sampled temp={} top_k={} top_p={} seed={:?}",
            config.temperature, config.top_k, config.top_p, config.seed
        ),
        None => "greedy".to_string(),
    }
}

fn print_report(report: &PassReport) {
    println!("pass: {}", report.label);
    println!(
        "  cycles={} committed_tokens={}",
        report.cycles, report.committed
    );
    let mut tokens_per_cycle = 1.0f64;
    for (position, &accepted) in report.accepted.iter().enumerate() {
        let rate = accepted as f64 / report.cycles.max(1) as f64;
        tokens_per_cycle += rate;
        println!(
            "  draft position {}: prefix acceptance={}/{} ({:.3})",
            position, accepted, report.cycles, rate
        );
    }
    println!("  expected tokens per target pass: {tokens_per_cycle:.3}");
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
        drafts: 5,
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
        seed: 20260816,
        skip_sampled: false,
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
            "--temperature" => args.temperature = parse(&iter.next(), "--temperature")?,
            "--top-k" => args.top_k = parse(&iter.next(), "--top-k")?,
            "--top-p" => args.top_p = parse(&iter.next(), "--top-p")?,
            "--seed" => args.seed = parse(&iter.next(), "--seed")?,
            "--greedy-only" => args.skip_sampled = true,
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
            "qwen38-mtp-probe <model-dir> [--prompt text | --prompt-file path] [--tokens n] \
             [--drafts k] [--temperature t] [--top-k k] [--top-p p] [--seed s] [--greedy-only]; \
             unexpected {arg}"
        ),
    }
}
