//! Host-side checkpoint and artifact formats for Eider.
//!
//! This crate indexes and decodes file representations only. It does not
//! allocate device memory, select CUDA devices, or prepare execution plans.

#![forbid(unsafe_code)]

mod error;
mod gguf;
mod gguf_quant;
mod safetensors;

pub use error::{Error, Result};
pub use gguf::{GgufIndex, GgufTensor, GgufValue};
pub use gguf_quant::{GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, dequantize_to_bf16, quantized_byte_len};
pub use safetensors::{SafeTensorInfo, SafeTensorShard};
