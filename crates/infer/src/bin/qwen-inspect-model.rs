use eider_cuda::{Error, Result};
use eider_inference::qwen3::infer::{
    QwenArchitecture, QwenFfnConfig, QwenLayerKind, QwenModelManifest,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let inspection = QwenModelManifest::inspect(&model_dir)?;
    let manifest = &inspection.manifest;
    let full_layers = manifest
        .layer_kinds
        .iter()
        .filter(|kind| **kind == QwenLayerKind::FullAttention)
        .count();
    let linear_layers = manifest
        .layer_kinds
        .iter()
        .filter(|kind| **kind == QwenLayerKind::LinearAttention)
        .count();

    println!("Qwen model inspection");
    println!("  model dir: {}", model_dir.display());
    println!("  architecture: {}", arch_label(manifest.architecture));
    println!("  tensor prefix: {}", manifest.tensor_prefix);
    println!(
        "  layers: {} full_attention={} linear_attention={}",
        manifest.layers, full_layers, linear_layers
    );
    println!(
        "  hidden={} vocab={} q_heads={} kv_heads={} head_dim={} rotary_dim={}",
        manifest.hidden,
        manifest.vocab,
        manifest.q_heads,
        manifest.kv_heads,
        manifest.head_dim,
        manifest.rotary_dim
    );
    println!("  ffn: {}", ffn_label(manifest.ffn));
    if let Some(linear) = manifest.linear_attention {
        println!(
            "  gated_delta_net: key_heads={} value_heads={} key_dim={} value_dim={} conv_kernel={}",
            linear.key_heads,
            linear.value_heads,
            linear.key_head_dim,
            linear.value_head_dim,
            linear.conv_kernel
        );
        if linear.key_heads != linear.value_heads {
            println!("  value-head layout: load-time grouped-to-tiled reorder required");
        }
    }
    if let Some(shared) = manifest.shared_expert_intermediate {
        println!("  shared expert intermediate: {shared}");
    }
    println!("  mtp layers: {}", manifest.mtp_layers);
    println!("  representative tensors:");
    for tensor in &inspection.tensors {
        if tensor.present {
            println!(
                "    ok      {:72} dtype={:8} shape={}",
                tensor.name,
                tensor.dtype.as_deref().unwrap_or("?"),
                shape_label(tensor.shape.as_deref().unwrap_or(&[]))
            );
        } else {
            println!("    missing {}", tensor.name);
        }
    }
    Ok(())
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen-inspect-model".to_string());
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir>"),
        }),
    }
}

fn arch_label(architecture: QwenArchitecture) -> &'static str {
    match architecture {
        QwenArchitecture::Qwen3 => "qwen3",
        QwenArchitecture::Qwen35Hybrid => "qwen3_5 hybrid",
        QwenArchitecture::Qwen38FlashNext => "qwen3_8_flash_next",
    }
}

fn ffn_label(ffn: QwenFfnConfig) -> String {
    match ffn {
        QwenFfnConfig::Dense => "dense".to_string(),
        QwenFfnConfig::Moe {
            experts,
            experts_per_token,
            expert_intermediate,
            norm_topk_prob,
        } => format!(
            "moe experts={experts} top_k={experts_per_token} expert_intermediate={expert_intermediate} norm_topk_prob={norm_topk_prob}"
        ),
    }
}

fn shape_label(shape: &[usize]) -> String {
    let mut out = String::from("[");
    for (idx, dim) in shape.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&dim.to_string());
    }
    out.push(']');
    out
}
