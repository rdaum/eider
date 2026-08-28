use eider_cuda::ModelOptCheckpoint;
use infer::nemotron3::{Nemotron3LayerKind, Nemotron3Manifest};
use std::path::PathBuf;

fn main() -> eider_cuda::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eider_cuda::Error::Format {
            label: "nemotron3-inspect-model arguments",
            detail: "usage: nemotron3-inspect-model <model-dir>".to_string(),
        })?;
    let manifest = Nemotron3Manifest::from_model_dir(&model_dir)?;
    let checkpoint = ModelOptCheckpoint::open(&model_dir)?;
    manifest.validate_checkpoint_index(&checkpoint)?;

    let count = |kind| manifest.layers.iter().filter(|&&item| item == kind).count();
    println!("Nemotron 3 checkpoint: {}", model_dir.display());
    println!(
        "  layers: {} ({} Mamba, {} MoE, {} attention)",
        manifest.layers.len(),
        count(Nemotron3LayerKind::Mamba),
        count(Nemotron3LayerKind::Moe),
        count(Nemotron3LayerKind::Attention),
    );
    println!(
        "  hidden/vocab/context: {} / {} / {}",
        manifest.hidden_size, manifest.vocab_size, manifest.max_position_embeddings
    );
    println!(
        "  attention: {} query heads, {} KV heads, head dim {}",
        manifest.attention_heads, manifest.kv_heads, manifest.attention_head_dim
    );
    println!(
        "  Mamba: {} heads x {}, {} groups, state {}, conv {}, projection {}",
        manifest.mamba_heads,
        manifest.mamba_head_dim,
        manifest.mamba_groups,
        manifest.mamba_state_size,
        manifest.mamba_conv_kernel,
        manifest.mamba_projection_size(),
    );
    println!(
        "  MoE: {} routed experts, top {}, latent {:?}, intermediate {}",
        manifest.routed_experts,
        manifest.experts_per_token,
        manifest.moe_latent_size,
        manifest.moe_intermediate_size,
    );
    println!(
        "  per-sequence FP32 Mamba state: {:.3} GiB",
        manifest.mamba_state_bytes_fp32() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!("  checkpoint tensor topology: valid");
    Ok(())
}
