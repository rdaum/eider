//! Prepare and inspect DeepSeek V4 routed-expert and thin-checkpoint artifacts.

use eider_cuda::{Error, Result};
use eider_inference::deepseek4::{
    finalise_thin_checkpoint, inspect_expert_artifacts, inspect_hot_expert_cache,
    inspect_nvfp4_expert_layer, inspect_nvfp4_expert_store, inspect_nvfp4_mtp_layer,
    inspect_thin_checkpoint, inspect_thin_checkpoint_shard, preflight_expert_artifacts,
    prepare_all_experts, prepare_expert_layer, prepare_hot_expert_layer,
    prepare_nvfp4_expert_layer, prepare_nvfp4_mtp_layer, prepare_thin_checkpoint_shard,
};
use std::ffi::OsString;
use std::path::PathBuf;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "deepseek4-experts".to_string());
    let Some(command) = args.next().and_then(|value| value.into_string().ok()) else {
        return usage(&program);
    };
    let args = args.collect::<Vec<_>>();
    match command.as_str() {
        "preflight" if args.len() == 2 => {
            let required =
                preflight_expert_artifacts(PathBuf::from(&args[0]), PathBuf::from(&args[1]))?;
            tracing::info!(
                missing_gib = required as f64 / (1u64 << 30) as f64,
                "validated DeepSeek V4 expert artifact disk space"
            );
            Ok(())
        }
        "prepare" if args.len() == 2 => {
            let info = prepare_all_experts(PathBuf::from(&args[0]), PathBuf::from(&args[1]))?;
            tracing::info!(
                layers = info.manifest.layers,
                artifact_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "prepared DeepSeek V4 expert artifact"
            );
            Ok(())
        }
        "prepare-layer" if args.len() == 3 => prepare_expert_layer(
            PathBuf::from(&args[0]),
            PathBuf::from(&args[1]),
            parse_usize(&program, "layer", &args[2])?,
        ),
        "inspect" if args.len() == 2 => {
            let info = inspect_expert_artifacts(PathBuf::from(&args[1]))?;
            tracing::info!(
                layers = info.manifest.layers,
                experts = info.manifest.routed_experts,
                artifact_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "validated DeepSeek V4 expert artifact"
            );
            Ok(())
        }
        "prepare-hot-layer" if args.len() >= 4 => {
            let capacity = parse_usize(&program, "capacity", &args[2])?;
            let layer = parse_usize(&program, "layer", &args[3])?;
            let experts = args[4..]
                .iter()
                .map(|value| parse_usize(&program, "expert", value))
                .collect::<Result<Vec<_>>>()?;
            let info = prepare_hot_expert_layer(
                PathBuf::from(&args[0]),
                PathBuf::from(&args[1]),
                capacity,
                layer,
                &experts,
            )?;
            tracing::info!(
                layer,
                experts = info.experts,
                cache_mib = info.file_bytes as f64 / (1u64 << 20) as f64,
                "prepared DeepSeek V4 hot-expert cache layer"
            );
            Ok(())
        }
        "inspect-hot" if args.len() == 3 => {
            let capacity = parse_usize(&program, "capacity", &args[2])?;
            let info = inspect_hot_expert_cache(
                PathBuf::from(&args[0]),
                PathBuf::from(&args[1]),
                capacity,
            )?;
            tracing::info!(
                experts = info.experts,
                cache_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "validated DeepSeek V4 hot-expert cache"
            );
            Ok(())
        }
        "prepare-nvfp4-layer" if args.len() == 3 => {
            let layer = parse_usize(&program, "layer", &args[2])?;
            let info = prepare_nvfp4_expert_layer(
                PathBuf::from(&args[0]),
                PathBuf::from(&args[1]),
                layer,
            )?;
            tracing::info!(
                layer,
                experts = info.experts,
                layer_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "prepared exact DeepSeek V4 NVFP4 expert layer"
            );
            Ok(())
        }
        "inspect-nvfp4-layer" if args.len() == 3 => {
            let layer = parse_usize(&program, "layer", &args[2])?;
            let info = inspect_nvfp4_expert_layer(
                PathBuf::from(&args[0]),
                PathBuf::from(&args[1]),
                layer,
            )?;
            tracing::info!(
                layer,
                experts = info.experts,
                layer_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "validated exact DeepSeek V4 NVFP4 expert layer"
            );
            Ok(())
        }
        "inspect-nvfp4" if args.len() == 2 => {
            let info =
                inspect_nvfp4_expert_store(PathBuf::from(&args[0]), PathBuf::from(&args[1]))?;
            tracing::info!(
                experts = info.experts,
                store_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "validated exact DeepSeek V4 NVFP4 expert store"
            );
            Ok(())
        }
        "prepare-nvfp4-mtp" if args.len() == 2 => {
            let info = prepare_nvfp4_mtp_layer(PathBuf::from(&args[0]), PathBuf::from(&args[1]))?;
            tracing::info!(
                experts = info.experts,
                layer_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "prepared exact DeepSeek V4 MTP NVFP4 expert layer"
            );
            Ok(())
        }
        "inspect-nvfp4-mtp" if args.len() == 2 => {
            let info = inspect_nvfp4_mtp_layer(PathBuf::from(&args[0]), PathBuf::from(&args[1]))?;
            tracing::info!(
                experts = info.experts,
                layer_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "validated exact DeepSeek V4 MTP NVFP4 expert layer"
            );
            Ok(())
        }
        "prepare-thin-shard" if args.len() == 3 => {
            let shard = args[2].to_str().ok_or_else(|| Error::Format {
                label: "usage",
                detail: format!("{program}: shard filename must be UTF-8"),
            })?;
            let bytes = prepare_thin_checkpoint_shard(
                PathBuf::from(&args[0]),
                PathBuf::from(&args[1]),
                shard,
            )?;
            tracing::info!(
                shard,
                file_mib = bytes as f64 / (1u64 << 20) as f64,
                "prepared DeepSeek V4 thin checkpoint shard"
            );
            Ok(())
        }
        "inspect-thin-shard" if args.len() == 3 => {
            let shard = args[2].to_str().ok_or_else(|| Error::Format {
                label: "usage",
                detail: format!("{program}: shard filename must be UTF-8"),
            })?;
            let bytes = inspect_thin_checkpoint_shard(
                PathBuf::from(&args[0]),
                PathBuf::from(&args[1]),
                shard,
            )?;
            tracing::info!(
                shard,
                file_mib = bytes as f64 / (1u64 << 20) as f64,
                "validated DeepSeek V4 thin checkpoint shard"
            );
            Ok(())
        }
        "finalise-thin" if args.len() == 2 => {
            let info = finalise_thin_checkpoint(PathBuf::from(&args[0]), PathBuf::from(&args[1]))?;
            tracing::info!(
                tensors = info.tensors,
                checkpoint_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "published DeepSeek V4 thin checkpoint"
            );
            Ok(())
        }
        "inspect-thin" if args.len() == 1 => {
            let info = inspect_thin_checkpoint(PathBuf::from(&args[0]))?;
            tracing::info!(
                tensors = info.tensors,
                checkpoint_gib = info.file_bytes as f64 / (1u64 << 30) as f64,
                "validated DeepSeek V4 thin checkpoint"
            );
            Ok(())
        }
        _ => usage(&program),
    }
}

fn parse_usize(program: &str, label: &str, value: &OsString) -> Result<usize> {
    value
        .to_str()
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!("{program}: {label} must be UTF-8"),
        })?
        .parse::<usize>()
        .map_err(|error| Error::Format {
            label: "usage",
            detail: format!("{program}: invalid {label}: {error}"),
        })
}

fn usage(program: &str) -> Result<()> {
    Err(Error::Format {
        label: "usage",
        detail: format!(
            "{program} <preflight|prepare|prepare-layer|inspect> <model-dir> <artifact-dir> [layer]\n\
             {program} prepare-hot-layer <model-dir> <cache-dir> <capacity> <layer> <expert>...\n\
             {program} inspect-hot <model-dir> <cache-dir> <capacity>\n\
             {program} prepare-nvfp4-layer <model-dir> <store-dir> <layer>\n\
             {program} inspect-nvfp4-layer <model-dir> <store-dir> <layer>\n\
             {program} inspect-nvfp4 <model-dir> <store-dir>\n\
             {program} prepare-nvfp4-mtp <model-dir> <store-dir>\n\
             {program} inspect-nvfp4-mtp <model-dir> <store-dir>\n\
             {program} prepare-thin-shard <model-dir> <thin-dir> <shard>\n\
             {program} inspect-thin-shard <model-dir> <thin-dir> <shard>\n\
             {program} finalise-thin <model-dir> <thin-dir>\n\
             {program} inspect-thin <thin-dir>"
        ),
    })
}
