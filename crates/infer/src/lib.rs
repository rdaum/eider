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

/// BitNet b1.58 ternary text-model loading and inference.
pub mod bitnet;

/// Ternary Bonsai dense Qwen3 model loading and inference.
pub mod bonsai;

/// Minimal GGUF v3 checkpoint indexing.
pub mod gguf;

/// CPU import support for GGML K-quantized tensors.
pub mod gguf_quant;

/// Qwen3 model loading and decode experiments.
pub mod qwen3;

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
