//! Request policy and text-output state shared by Eider inference engines.
//!
//! This crate does not select CUDA devices, allocate device memory, or depend
//! on a model implementation. Inference engines report their capabilities at
//! the execution boundary.

#![forbid(unsafe_code)]

pub mod cache;
pub mod chat;
pub mod chat_output;
pub mod generation;
pub mod request;
pub mod sampling;
pub mod scheduler;
pub mod stop;
pub mod tool_grammar;
