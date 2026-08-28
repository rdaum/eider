//! Checkpoint deployment selection owned by inference.

use crate::{InferenceError, InferenceResult};
use serde::Deserialize;
use std::path::Path;

/// Supported checkpoint execution family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointArchitecture {
    /// BitNet b1.58.
    BitNet,
    /// Ling 3.
    Ling3,
    /// Muse Glimmer.
    MuseGlimmer,
    /// Ternary Bonsai.
    Bonsai,
    /// Qwen3.5/3.6 hybrid MoE.
    Qwen36,
    /// Qwen3.8 Flash Next.
    Qwen38FlashNext,
    /// Step-3.7.
    Step37,
    /// Nemotron 3 hybrid.
    Nemotron3,
    /// Gemma 4.
    Gemma4,
    /// Laguna-S-2.1.
    Laguna,
    /// DeepSeek V4.
    Deepseek4,
}

#[derive(Deserialize)]
struct CheckpointConfig {
    model_type: String,
}

/// Reads the checkpoint configuration and selects its execution family.
pub fn checkpoint_architecture(model_dir: &Path) -> InferenceResult<CheckpointArchitecture> {
    let path = model_dir.join("config.json");
    let contents = std::fs::read_to_string(&path).map_err(|error| {
        InferenceError::Deployment(format!("failed to read {}: {error}", path.display()))
    })?;
    let config: CheckpointConfig = serde_json::from_str(&contents).map_err(|error| {
        InferenceError::Deployment(format!("failed to parse {}: {error}", path.display()))
    })?;
    match config.model_type.as_str() {
        "bitnet" => Ok(CheckpointArchitecture::BitNet),
        "bailing_hybrid" => Ok(CheckpointArchitecture::Ling3),
        "muse_glimmer" => Ok(CheckpointArchitecture::MuseGlimmer),
        "bonsai" => Ok(CheckpointArchitecture::Bonsai),
        "qwen3_5" | "qwen3_5_moe" => Ok(CheckpointArchitecture::Qwen36),
        "qwen3_8_flash_next" => Ok(CheckpointArchitecture::Qwen38FlashNext),
        "step3p7" => Ok(CheckpointArchitecture::Step37),
        "nemotron_h" | "nemotron_h_puzzle" => Ok(CheckpointArchitecture::Nemotron3),
        "gemma4" => Ok(CheckpointArchitecture::Gemma4),
        "laguna" => Ok(CheckpointArchitecture::Laguna),
        "deepseek_v4" => Ok(CheckpointArchitecture::Deepseek4),
        other => Err(InferenceError::Deployment(format!(
            "unsupported model_type {other:?} in {}",
            path.display()
        ))),
    }
}
