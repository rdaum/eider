//! CUDA-backed operation families used by the inference layers.

#[cfg(feature = "cuda-oxide")]
pub(crate) mod core_oxide;
pub(crate) mod deepseek4;
pub(crate) mod gemma4_attention;
pub(crate) mod non_gemm;
pub(crate) mod qwen36_gdn;
#[cfg(feature = "cuda-oxide")]
pub(crate) mod qwen36_gdn_oxide;
#[cfg(test)]
mod qwen36_gdn_reference;
pub(crate) mod qwen38;
pub(crate) mod sm121_w4a16;
#[cfg(feature = "cuda-oxide")]
pub(crate) mod sm121_w4a16_oxide;
pub(crate) mod sm12x_kv_cache;
#[cfg(feature = "cuda-oxide")]
pub(crate) mod sm12x_kv_cache_oxide;
pub(crate) mod sm12x_mma;
#[cfg(feature = "cuda-oxide")]
pub(crate) mod w4a16_matvec_oxide;
