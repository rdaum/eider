//! LFM2 MoE inference with resident ModelOpt NVFP4 routed experts.

mod config;
mod model;
pub use config::Lfm2Config;
pub(crate) use model::{
    Lfm2DecisionReadout, Lfm2DecisionSequence, Lfm2PrefixCache, Lfm2TreeWorkspace,
};
pub use model::{Lfm2Model, Lfm2PrefillWorkspace, Lfm2Sequence};
