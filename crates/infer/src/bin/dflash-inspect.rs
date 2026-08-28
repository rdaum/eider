//! Inspect an official DFlash GGUF without loading tensor payloads.

use eider_cuda::{Error, Result};
use infer::muse_glimmer::DFlashConfig;
use std::path::PathBuf;

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: "dflash-inspect DFLASH_GGUF".to_string(),
        })?;
    println!("{:#?}", DFlashConfig::open(path)?);
    Ok(())
}
