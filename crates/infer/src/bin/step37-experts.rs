//! Prepare and inspect the Step-3.7 routed-expert cache.

use infer::nvfp4::{Error, Result};
use infer::step35::{FIRST_MOE_LAYER, Step35ResidentExperts, prepare_all, prepare_one};
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
        .unwrap_or_else(|| "step37-experts".to_string());
    let command = args.next().and_then(|value| value.into_string().ok());
    let model_dir = args.next().map(PathBuf::from);
    let layer = args
        .next()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| Error::Format {
                    label: "usage",
                    detail: format!("{program}: layer must be UTF-8"),
                })?
                .parse::<usize>()
                .map_err(|error| Error::Format {
                    label: "usage",
                    detail: format!("{program}: invalid layer: {error}"),
                })
        })
        .transpose()?;
    if args.next().is_some() {
        return usage(&program);
    }
    let Some(command) = command else {
        return usage(&program);
    };
    let model_dir = model_dir.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.7-flash-nvfp4")
    });
    match command.as_str() {
        "prepare" if layer.is_none() => prepare_all(&model_dir),
        "prepare-layer" => prepare_one(&model_dir, layer.unwrap_or(FIRST_MOE_LAYER)),
        "residency" if layer.is_none() => {
            let resident = Step35ResidentExperts::load(&model_dir)?;
            println!(
                "full prepared residency succeeded: {:.3} GiB",
                resident.device_bytes() as f64 / (1u64 << 30) as f64
            );
            Ok(())
        }
        _ => usage(&program),
    }
}

fn usage(program: &str) -> Result<()> {
    Err(Error::Format {
        label: "usage",
        detail: format!("{program} <prepare|prepare-layer|residency> [model-dir] [layer]"),
    })
}
