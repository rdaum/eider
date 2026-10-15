//! LFM2 MoE inference with resident ModelOpt NVFP4 routed experts.

mod config;
mod model;
pub use config::Lfm2Config;
pub(crate) use model::Lfm2PrefixCache;
pub use model::{Lfm2Model, Lfm2PrefillWorkspace, Lfm2Sequence};
