//! OpenAI Responses-compatible serving for Eider.

pub mod actor;
pub mod protocol;
pub mod server;

pub use actor::{InferenceActor, InferenceActorConfig};
pub use server::{ApiConfig, serve};
