//! Model loading and inference execution for DGX Spark.
//!
//! This crate is intentionally thin while the lower-level CUDA storage and
//! cuBLASLt path are still being validated. It will own model loading, layer
//! composition, KV-cache policy, and decode/prefill orchestration. Low-level
//! FP4 tensor storage and matmul execution live in `eider-cuda`.

#![deny(unsafe_code)]

// POSIX vectored expert-record reads require one reviewed libc call. All model
// composition and execution code remains under the crate-wide unsafe denial.
#[allow(unsafe_code)]
mod system_io;

mod error;
pub use error::{InferenceError, InferenceResult};

mod paged_prefill_attention;

/// Physical SM12x KV-page storage shared by inference model state.
pub(crate) mod sm12x_cache;

/// Model-owned execution support and device-resident KV cache storage.
pub mod execution;

/// BitNet b1.58 ternary text-model loading and inference.
pub mod bitnet;

/// Ternary Bonsai dense Qwen3 model loading and inference.
pub mod bonsai;

/// Qwen3 model loading and decode experiments.
pub mod qwen3;

/// Qwen3.8 Flash Next text-model loading and inference.
pub mod qwen38_flash_next;

/// Hashed n-gram embedding identifiers and transactional token-window state.
pub mod ngram;

/// Gemma 4 text-model loading and inference.
pub mod gemma4;

/// Muse Glimmer dense text-model loading and inference.
pub mod muse_glimmer;

/// Nemotron 3 hybrid Mamba/attention/MoE model support.
pub mod nemotron3;

/// DeepSeek V4 expert storage preparation and memory-bounded execution support.
pub mod deepseek4;

/// InclusionAI Ling 3 hybrid KDA/MLA sparse-MoE model support.
pub mod ling3;

/// Poolside Laguna sparse-MoE model support.
pub mod laguna;

/// Step-3.7 expert preparation and residency experiments.
pub mod step37;

/// Step-3.7 layer correctness probes.
pub mod step37_probe;

/// Runtime and cache metrics.
pub mod metrics;

/// Returns the currently linked NVFP4 backend label.
pub fn backend_name() -> &'static str {
    "eider-nvfp4"
}
