use eider_cuda::{Error, Result};
use eider_inference::ling3::{Ling3FfnKind, Ling3Manifest};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let inspection = Ling3Manifest::inspect(&model_dir)?;
    let manifest = &inspection.manifest;
    let dense_layers = (0..manifest.num_hidden_layers)
        .filter(|&layer| matches!(manifest.ffn_kind(layer), Ok(Ling3FfnKind::Dense)))
        .count();
    let moe_layers = manifest.num_hidden_layers - dense_layers;

    println!("Ling 3 checkpoint: {}", model_dir.display());
    println!(
        "  layers: {} ({} KDA, {} MLA; {} dense, {} MoE)",
        manifest.num_hidden_layers,
        manifest.kda_layers(),
        manifest.mla_layers(),
        dense_layers,
        moe_layers,
    );
    println!(
        "  hidden/vocab/context: {} / {} / {}",
        manifest.hidden_size, manifest.vocab_size, manifest.max_position_embeddings,
    );
    println!(
        "  KDA: {} heads x {}, conv {}, FP32 state {:.3} MiB/sequence",
        manifest.attention_heads,
        manifest.head_dim,
        manifest.conv_kernel_size,
        (manifest.kda_layers()
            * (manifest.recurrent_state_values_per_kda_layer()
                + manifest.conv_state_values_per_kda_layer())
            * size_of::<f32>()) as f64
            / (1024.0 * 1024.0),
    );
    println!(
        "  MLA: q_lora={:?} kv_lora={} qk={}+{} value={}",
        manifest.q_lora_rank,
        manifest.kv_lora_rank,
        manifest.qk_nope_head_dim,
        manifest.qk_rope_head_dim,
        manifest.v_head_dim,
    );
    println!(
        "  MoE: {} experts, top {}, groups {}/{}, intermediate {}, shared {}",
        manifest.routed_experts,
        manifest.experts_per_token,
        manifest.selected_expert_groups,
        manifest.expert_groups,
        manifest.expert_intermediate_size,
        manifest.shared_expert_intermediate_size,
    );
    match &manifest.fp8 {
        Some(fp8) => println!(
            "  storage: block FP8 {:?}, scale {:?}",
            fp8.weight_block_size, fp8.scale_format,
        ),
        None => println!("  storage: BF16"),
    }
    println!(
        "  next-token prediction layers: {}",
        manifest.nextn_predict_layers
    );
    println!("  representative tensors:");
    let mut invalid = 0usize;
    for tensor in &inspection.tensors {
        if tensor.shape_matches() {
            println!(
                "    ok      {:72} dtype={:8} shape={}",
                tensor.name,
                tensor.dtype.as_deref().unwrap_or("?"),
                shape_label(tensor.shape.as_deref().unwrap_or(&[])),
            );
        } else {
            invalid += 1;
            println!(
                "    invalid {:72} expected={} actual={}",
                tensor.name,
                shape_label(&tensor.expected_shape),
                tensor
                    .shape
                    .as_deref()
                    .map(shape_label)
                    .unwrap_or_else(|| "missing".to_string()),
            );
        }
    }
    if invalid != 0 {
        return Err(Error::Format {
            label: "Ling 3 checkpoint inspection",
            detail: format!("{invalid} representative tensors were missing or had the wrong shape"),
        });
    }
    Ok(())
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "ling3-inspect-model".to_string());
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir>"),
        }),
    }
}

fn shape_label(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
    )
}
