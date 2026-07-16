//! Compare representative Step-3.7 layers with the Python reference.

use infer::nvfp4::Result;
use infer::step37_probe::validate_reference_layers;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.7-flash-nvfp4")
    });
    let reference = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join(".eider-cache/step37-layer-reference-v1.safetensors"));
    validate_reference_layers(model_dir, reference)
}
