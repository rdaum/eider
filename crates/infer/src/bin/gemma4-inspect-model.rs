use infer::gemma4::{Gemma4Attention, Gemma4Checkpoint};
use infer::nvfp4::{CudaStream, DeviceBuffer, Error, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let checkpoint = Gemma4Checkpoint::open(&model_dir)?;
    let config = checkpoint.config();
    let first_global = (0..config.num_hidden_layers)
        .find(|&layer| config.is_full_attention_layer(layer).unwrap_or(false));

    let local_attention = Gemma4Attention::load(&checkpoint, 0)?;
    let global_layer = first_global.expect("Gemma 4 configuration must contain a global layer");
    let global_attention = Gemma4Attention::load(&checkpoint, global_layer)?;
    let stream = CudaStream::new_blocking()?;
    let input = DeviceBuffer::from_host(
        &(0..config.hidden_size)
            .map(|index| {
                (index as f32 - config.hidden_size as f32 / 2.0) / config.hidden_size as f32
            })
            .collect::<Vec<_>>(),
    )?;
    run_attention_smoke(&local_attention, &input, &stream)?;
    run_attention_smoke(&global_attention, &input, &stream)?;

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
    println!("  validated local and global attention projection layouts");
    println!("  ran one-token local and global compact-KV attention smoke checks");
    Ok(())
}

fn run_attention_smoke(
    attention: &Gemma4Attention,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let mut workspace = attention.new_workspace()?;
    let mut cache = attention.new_kv_cache(1)?;
    let mut compact_attention = attention.new_compact_attention_workspace(1)?;
    attention.run_decode_into(
        input,
        &mut workspace,
        &mut cache,
        &mut compact_attention,
        0,
        stream,
    )
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
