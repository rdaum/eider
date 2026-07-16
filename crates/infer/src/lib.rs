//! Inference-facing crate for DGX Spark experiments.
//!
//! This crate is intentionally thin while the lower-level NVFP4 storage and
//! cuBLASLt path are still being validated. It will own model loading, layer
//! composition, KV-cache policy, and decode/prefill orchestration. Low-level
//! FP4 tensor storage and matmul execution live in `nvfp4`.

pub use nvfp4;

/// Runtime state and device-resident KV cache storage.
pub mod runtime;
pub use runtime::kv_cache;

/// Qwen3 model loading and decode experiments.
pub mod qwen3;

/// Step-3.5 expert preparation and residency experiments.
pub mod step35;

/// Runtime and cache metrics.
pub mod metrics;

/// Returns the currently linked NVFP4 backend label.
pub fn backend_name() -> &'static str {
    "eider-nvfp4"
}
