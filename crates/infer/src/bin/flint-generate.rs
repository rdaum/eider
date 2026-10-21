//! Flint checkpoint inspection and single-sequence generation.
use eider_format::GgufIndex;
use eider_inference::flint::{FlintConfig, FlintModel};
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use eider_runtime::sampling::{Sampler, SamplingConfig, TokenHistory};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args().skip(1);
    let gguf = PathBuf::from(args.next().ok_or_else(|| {
        anyhow::anyhow!(
            "usage: flint-generate MODEL.gguf [--inspect | TOKENIZER_DIR [PROMPT [TOKENS]] ]"
        )
    })?);
    let Some(directory) = args.next().filter(|s| s != "--inspect") else {
        let index = GgufIndex::open(&gguf)?;
        println!("{:#?}", FlintConfig::from_index(&index)?);
        let weight = index.read_nvfp4_linear("blk.2.ffn_gate_exps.weight", Some(0))?;
        println!(
            "Imported expert: {}x{}, {} weight bytes, {} scale bytes, global scale {}",
            weight.out_features,
            weight.in_features,
            weight.packed_weight.len(),
            weight.weight_scale.len(),
            weight.weight_scale_2
        );
        return Ok(());
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is 2 + 2? Answer briefly.".into());
    let count: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(64);
    let template = CheckpointChatTemplate::from_model_dir(&directory)?;
    let rendered = template.render_and_tokenize(
        &[ChatMessage::user(prompt)],
        &[],
        ChatTemplateOptions::default(),
    )?;
    let start = Instant::now();
    let model = FlintModel::load(&gguf)?;
    eprintln!("loaded in {:.2}s", start.elapsed().as_secs_f64());
    let capacity = rendered.token_ids.len() + count.max(1);
    let mut cache = model.new_sequence_cache(1, capacity, 0)?;
    let mut state = model
        .admit(&mut cache, capacity, &[])?
        .expect("cache capacity");
    let mut work = model.new_workspace(64.min(capacity), capacity)?;
    let started = Instant::now();
    for chunk in rendered.token_ids.chunks(64) {
        model.forward(&mut work, &mut state, &mut cache, chunk)?;
    }
    eprintln!(
        "prefill: {} tokens in {:.3}s",
        rendered.token_ids.len(),
        started.elapsed().as_secs_f64()
    );
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    })?;
    let mut history = TokenHistory::from_tokens(rendered.token_ids.iter().copied());
    let mut output = Vec::new();
    let started = Instant::now();
    for i in 0..count {
        let logits = model.logits(&state)?;
        anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let next = sampler.sample(&logits, &history)?.id;
        if [150001, 150003, 150005, 150018].contains(&next) {
            break;
        }
        output.push(next);
        history.push(next);
        if i + 1 < count {
            model.forward(&mut work, &mut state, &mut cache, &[next])?;
        }
    }
    println!(
        "{}",
        template
            .tokenizer()
            .decode(&output, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    );
    eprintln!(
        "decode: {} tokens in {:.3}s",
        output.len(),
        started.elapsed().as_secs_f64()
    );
    model.finish(&mut cache, state)?;
    Ok(())
}
