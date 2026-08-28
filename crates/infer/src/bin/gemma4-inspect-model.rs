use eider_cuda::{Error, Result};
use infer::gemma4::Gemma4Checkpoint;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let checkpoint = Gemma4Checkpoint::open(&model_dir)?;
    let config = checkpoint.config();
    let first_global = (0..config.num_hidden_layers)
        .find(|&layer| config.is_full_attention_layer(layer).unwrap_or(false));

    println!("Gemma 4 checkpoint: {}", model_dir.display());
    println!(
        "  layers={} hidden={} context={} sliding_window={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.max_position_embeddings,
        config.sliding_window
    );
    println!(
        "  attention: {} query heads, {} local KV heads, {} global KV heads, local/global head dim {}/{}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_global_key_value_heads,
        config.head_dim,
        config.global_head_dim,
    );
    println!(
        "  MoE: {} experts, top {}, expert intermediate {}",
        config.num_experts, config.top_k_experts, config.moe_intermediate_size
    );
    println!("  first global-attention layer: {first_global:?}");
    Ok(())
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "gemma4-inspect-model".to_string());
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir>"),
        }),
    }
}
