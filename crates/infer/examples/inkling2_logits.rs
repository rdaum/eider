//! Greedy text-generation probe for a local Inkling 2 Mini checkpoint.
//!
//! The prompt runs twice through one sequence cache. The first run retains
//! its page-aligned prompt prefix; the second restores it and must generate
//! the same tokens. Prompts shorter than one 128-token page are not retained.
//!
//! Usage: `cargo run --release -p eider-inference --example inkling2_logits -- MODEL_DIR [PROMPT]`

use eider_cuda::Result;
use eider_inference::inkling2::{Inkling2Model, Inkling2SequenceCache, Inkling2Workspace};
use eider_runtime::chat::{
    ChatMessage, ChatTemplateOptions, CheckpointChatTemplate, Inkling2PromptRenderer,
};
use eider_runtime::chat_output::ChatOutputCodec;
use std::path::PathBuf;
use std::time::Instant;

const STEPS: usize = 32;
const RETAINED_BYTES: usize = 1 << 30;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: inkling2_logits MODEL_DIR [PROMPT]");
        std::process::exit(2);
    }));
    let text = args
        .next()
        .unwrap_or_else(|| "Explain looped inference in one sentence.".to_string());
    let template = CheckpointChatTemplate::from_model_dir(&root)?;
    let prompt = Inkling2PromptRenderer::new(template.tokenizer())?.render(
        &[ChatMessage::user(text)],
        &[],
        ChatTemplateOptions::default(),
        true,
    )?;
    println!("prompt token ids: {prompt:?}");
    let model = Inkling2Model::load(root)?;
    let mut workspace = model.new_workspace(prompt.len().max(1))?;
    let mut cache = model.new_sequence_cache(1, prompt.len() + STEPS, RETAINED_BYTES)?;
    let first = generate(&model, &mut workspace, &mut cache, &prompt)?;
    let second = generate(&model, &mut workspace, &mut cache, &prompt)?;
    println!("generated token ids: {first:?}");
    if first != second {
        println!("restored-prefix tokens differ: {second:?}");
    }
    let mut codec = ChatOutputCodec::new(template.tokenizer(), &[], false)?;
    let mut events = Vec::new();
    for &token in first.iter().filter(|&&t| t != model.config().eos_token_id) {
        events.extend(codec.push_token(token)?);
    }
    events.extend(codec.finish()?);
    println!("{events:?}");
    Ok(())
}

fn generate(
    model: &Inkling2Model,
    workspace: &mut Inkling2Workspace,
    cache: &mut Inkling2SequenceCache,
    prompt: &[u32],
) -> Result<Vec<u32>> {
    let start = Instant::now();
    let target = cache.cacheable_prefix_tokens(prompt.len());
    let mut sequence = model
        .admit(cache, prompt.len() + STEPS, prompt)?
        .expect("an empty cache admits one sequence");
    let cached = sequence.position();
    if cached < target {
        model.forward(workspace, &mut sequence, cache, &prompt[cached..target])?;
        model.retain_prefix(cache, &mut sequence, prompt)?;
    }
    let position = sequence.position();
    model.forward(workspace, &mut sequence, cache, &prompt[position..])?;
    println!(
        "cached {cached} of {} prompt tokens; first token after {:.1} ms",
        prompt.len(),
        start.elapsed().as_secs_f64() * 1e3
    );
    let mut generated = Vec::new();
    for _ in 0..STEPS {
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
