//! Greedy text-generation probe for a local Inkling 2 Mini checkpoint.
//!
//! Usage: `cargo run --release -p eider-inference --example inkling2_logits -- MODEL_DIR [PROMPT]`

use eider_cuda::Result;
use eider_inference::inkling2::Inkling2Model;
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use eider_runtime::chat_output::ChatOutputCodec;
use std::path::PathBuf;

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
    let prompt = template.render_and_tokenize(
        &[ChatMessage::user(text)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    println!("prompt token ids: {:?}", prompt.token_ids);
    let model = Inkling2Model::load(root)?;
    let mut workspace = model.new_workspace(prompt.token_ids.len().max(1))?;
    let steps = 32;
    let mut sequence = model.new_sequence(prompt.token_ids.len() + steps)?;
    model.forward(&mut workspace, &mut sequence, &prompt.token_ids)?;
    let mut codec = ChatOutputCodec::new(template.tokenizer(), &[], false)?;
    let mut events = Vec::new();
    let mut generated = Vec::new();
    for _ in 0..steps {
        let token = model.argmax_with_logit(&mut sequence)?.0;
        generated.push(token);
        if token == model.config().eos_token_id {
            break;
        }
        events.extend(codec.push_token(token)?);
        model.forward(&mut workspace, &mut sequence, &[token])?;
    }
    events.extend(codec.finish()?);
    println!("generated token ids: {generated:?}");
    println!("{events:?}");
    Ok(())
}
