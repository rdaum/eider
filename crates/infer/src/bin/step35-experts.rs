use infer::nvfp4::{Error, Result};
use infer::step35::{Step35ResidentExperts, prepare_all};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "step35-experts".to_string());
    let command = args.next().and_then(|value| value.into_string().ok());
    let model_dir = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return usage(&program);
    }
    let Some(command) = command else {
        return usage(&program);
    };
    let model_dir = model_dir.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.5-flash-nvfp4")
    });
    match command.as_str() {
        "prepare" => prepare_all(&model_dir),
        "residency" => {
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
        detail: format!("{program} <prepare|residency> [model-dir]"),
    })
}
