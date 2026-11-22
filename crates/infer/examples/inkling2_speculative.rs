//! Greedy DSpark speculative decoding check for a local Inkling 2 Mini checkpoint.
//!
//! Generates the same greedy continuation twice: one token per forward, then
//! with draft blocks verified by the target. Reports where the two diverge
//! (batched verification is not bitwise identical to single-token decode),
//! the mean accepted draft length, and decode throughput.
//!
//! Usage: `cargo run --release -p eider-inference --example inkling2_speculative -- MODEL_DIR [PROMPT] [TOKENS] [DRAFTS]`

use eider_cuda::Result;
use eider_inference::inkling2::{Inkling2Model, Inkling2SequenceCache, Inkling2Workspace};
use eider_runtime::chat::{
    ChatMessage, ChatTemplateOptions, CheckpointChatTemplate, Inkling2PromptRenderer,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: inkling2_speculative MODEL_DIR [PROMPT] [TOKENS] [DRAFTS]");
        std::process::exit(2);
    }));
    let text = args.next().unwrap_or_else(|| {
        "Write a short Python function that returns the n-th Fibonacci number, then explain it."
            .to_string()
    });
    let tokens: usize = args.next().map_or(256, |n| n.parse().expect("token count"));
    let drafts: usize = args.next().map_or(4, |n| n.parse().expect("draft count"));
    let template = CheckpointChatTemplate::from_model_dir(&root)?;
    let prompt = Inkling2PromptRenderer::new(template.tokenizer())?
        .render(
            &[ChatMessage::user(text)],
            &[],
            ChatTemplateOptions::default(),
            true,
        )?
        .token_ids;
    let model = Inkling2Model::load_with_options(&root, true)?;
    let mut workspace = model.new_workspace(prompt.len())?;
    let mut cache = model.new_sequence_cache(1, prompt.len() + tokens + 8, 0)?;

    let start = Instant::now();
    let reference = plain(&model, &mut workspace, &mut cache, &prompt, tokens)?;
    let plain_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (speculative, blocks) =
        speculative(&model, &mut workspace, &mut cache, &prompt, tokens, drafts)?;
    let speculative_seconds = start.elapsed().as_secs_f64();

    let agree = reference
        .iter()
        .zip(&speculative)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "plain: {} tokens in {plain_seconds:.2} s ({:.1} tok/s)",
        reference.len(),
        reference.len() as f64 / plain_seconds
    );
    println!(
        "speculative: {} tokens in {} blocks, {speculative_seconds:.2} s ({:.1} tok/s), {:.2} tokens per block",
        speculative.len(),
        blocks,
        speculative.len() as f64 / speculative_seconds,
        speculative.len() as f64 / blocks as f64
    );
    println!(
        "identical for the first {agree} of {} tokens",
        reference.len()
    );
    println!(
        "{}",
        template
            .tokenizer()
            .decode(&speculative, false)
            .unwrap_or_default()
    );
    Ok(())
}

fn plain(
    model: &Inkling2Model,
    workspace: &mut Inkling2Workspace,
    cache: &mut Inkling2SequenceCache,
    prompt: &[u32],
    tokens: usize,
) -> Result<Vec<u32>> {
    let mut sequence = model
        .admit(cache, prompt.len() + tokens + 8, prompt)?
        .expect("an empty cache admits one sequence");
    model.forward(workspace, &mut sequence, cache, prompt)?;
    let mut generated = Vec::new();
    for _ in 0..tokens {
        let token = model.argmax_with_logit(&mut sequence)?.0;
        generated.push(token);
        if token == model.config().eos_token_id {
            break;
        }
        model.forward(workspace, &mut sequence, cache, &[token])?;
    }
    model.finish(cache, sequence)?;
    Ok(generated)
}

fn speculative(
    model: &Inkling2Model,
    workspace: &mut Inkling2Workspace,
    cache: &mut Inkling2SequenceCache,
    prompt: &[u32],
    tokens: usize,
    drafts: usize,
) -> Result<(Vec<u32>, usize)> {
    let vocab = model.config().text_config.vocab_size;
    let eos = model.config().eos_token_id;
    let mut sequence = model
        .admit(cache, prompt.len() + tokens + 8, prompt)?
        .expect("an empty cache admits one sequence");
    model.forward(workspace, &mut sequence, cache, prompt)?;
    let mut anchor = model.argmax_with_logit(&mut sequence)?.0;
    let mut generated = vec![anchor];
    let mut blocks = 0;
    let mut timing = [0f64; 3];
    while generated.len() < tokens && anchor != eos {
        let started = Instant::now();
        let steps = drafts.min(tokens - generated.len());
        let drafted = model.draft(
            workspace,
            &mut sequence,
            anchor,
            steps,
            &mut |ids, logits| {
                let best = (0..ids.len())
                    .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
                    .expect("candidates");
                Ok(ids[best])
            },
        )?;
        let mut block = vec![anchor];
        block.extend(&drafted);
        timing[0] += started.elapsed().as_secs_f64();
        let started = Instant::now();
        model.verify(workspace, &mut sequence, cache, &block)?;
        let logits = model.verified_logits_to_host(workspace, &sequence)?;
        timing[1] += started.elapsed().as_secs_f64();
        let argmax = |row: usize| {
            let row = &logits[row * vocab..(row + 1) * vocab];
            (0..vocab)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
                .expect("vocabulary") as u32
        };
        let mut kept = 1;
        let mut next = argmax(0);
        for &token in &drafted {
            if token != next || generated.len() >= tokens || next == eos {
                break;
            }
            generated.push(token);
            next = argmax(kept);
            kept += 1;
        }
        let started = Instant::now();
        model.accept(workspace, &mut sequence, cache, kept)?;
        timing[2] += started.elapsed().as_secs_f64();
        blocks += 1;
        if generated.len() < tokens {
            generated.push(next);
        }
        anchor = next;
    }
    let per_block = |seconds: f64| seconds * 1e3 / blocks.max(1) as f64;
    println!(
        "per block: draft {:.1} ms, verify {:.1} ms, accept {:.1} ms",
        per_block(timing[0]),
        per_block(timing[1]),
        per_block(timing[2])
    );
    model.finish(cache, sequence)?;
    generated.truncate(tokens);
    if let Some(end) = generated.iter().position(|&token| token == eos) {
        generated.truncate(end + 1);
    }
    Ok((generated, blocks.max(1)))
}
