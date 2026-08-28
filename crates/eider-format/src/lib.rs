//! Host-side checkpoint and artifact formats for Eider.
//!
//! This crate indexes and decodes file representations only. It does not
//! allocate device memory, select CUDA devices, or prepare execution plans.

#![forbid(unsafe_code)]

mod checkpoint;
mod error;
mod gguf;
mod gguf_quant;
mod modelopt;
mod nvfp4_artifact;
mod safetensors;

pub use checkpoint::SafeTensorCheckpoint;
pub use error::{Error, Result};
pub use gguf::{GgufIndex, GgufTensor, GgufValue};
pub use gguf_quant::{GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, dequantize_to_bf16, quantized_byte_len};
pub use modelopt::{
    ModelOptBlockScaledFp8Linear, ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear,
    modelopt_scales_to_cublaslt,
};
pub use nvfp4_artifact::Nvfp4Artifact;
pub use safetensors::{SafeTensorInfo, SafeTensorShard};
