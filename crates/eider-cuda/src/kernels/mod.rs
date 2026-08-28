//! CUDA-backed operation families used by the inference layers.

pub(crate) mod deepseek4;
pub(crate) mod gemma4_attention;
pub(crate) mod non_gemm;
pub(crate) mod qwen36_gdn;
#[cfg(test)]
mod qwen36_gdn_reference;
pub(crate) mod qwen38;
pub(crate) mod sm121_w4a16;
pub(crate) mod sm12x_kv_cache;
pub(crate) mod sm12x_mma;
