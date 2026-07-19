//! OpenAI-compatible Responses and Chat Completions serving for Eider.

pub mod actor;
pub mod chat_completions;
pub mod deployment;
pub mod metrics;
pub mod protocol;
pub mod server;

pub use actor::{InferenceActor, InferenceActorConfig};
pub use server::{ApiConfig, serve};
