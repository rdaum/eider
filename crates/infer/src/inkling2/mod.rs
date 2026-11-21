//! Inkling 2 Mini fixed-pass text-model loading and inference.
//!
//! The execution path is text-only. Vision, DSpark, variable loop counts, and
//! native MXFP8 activations are intentionally excluded.

mod config;
mod model;
mod sequence;
mod weights;

pub use config::Inkling2Config;
pub use model::{Inkling2Model, Inkling2Sequence, Inkling2Workspace};
pub use sequence::{Inkling2SequenceCache, PAGE_TOKENS};
