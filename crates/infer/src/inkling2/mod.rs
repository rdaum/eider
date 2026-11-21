//! Inkling 2 Mini fixed-pass text-model loading and inference.
//!
//! The first execution path is text-only. Vision, DSpark, variable loop counts,
//! native MXFP8 activations, and retained prefixes are intentionally excluded.

mod config;
mod model;
mod weights;

pub use config::Inkling2Config;
pub use model::{Inkling2Model, Inkling2Sequence, Inkling2Workspace};
