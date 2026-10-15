//! LiLiCorr companion for Nemotron 3.5: checkpoint contract and draft execution.

mod config;
mod draft;
mod head;
mod linear;
mod proposal;

pub use config::{LiLiCorrConfig, inspect_lilicorr_config, validate_lilicorr_checkpoint};
pub use draft::{LiLiCorrDraft, LiLiCorrSequence, LiLiCorrSnapshot, LiLiCorrWorkspace};
pub use proposal::LiLiCorrProposal;
