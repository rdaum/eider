//! InclusionAI Ling 3 hybrid-model configuration and checkpoint topology.

mod config;
mod execution;
mod kda;
pub mod kda_reference;
mod layer;
mod mla;
mod model;
mod moe;
mod sequence;

pub use config::{
    Ling3AttentionKind, Ling3FfnKind, Ling3Fp8Config, Ling3Manifest, Ling3ModelInspection,
    Ling3TensorCheck,
};
pub(crate) use execution::{Ling3SequenceId, Ling3SequencePool};
pub use kda::{Ling3KdaAttention, Ling3KdaAttentionState, Ling3KdaAttentionWorkspace};
pub use layer::{Ling3KdaDenseLayer, Ling3KdaLayerState, Ling3KdaLayerWorkspace};
pub use mla::{Ling3MlaAttention, Ling3MlaState, Ling3MlaWorkspace};
pub use model::{Ling3Model, Ling3PrefillWorkspace};
pub(crate) use model::{Ling3ModelState, Ling3ModelWorkspace};
pub use moe::{Ling3Moe, Ling3MoeWorkspace};
pub(crate) use sequence::{Ling3CacheContext, Ling3MlaPagePool, Ling3Page, ling3_cache_error};
pub use sequence::{
    Ling3Sequence, Ling3SequenceCache, admit_ling3_sequence, new_ling3_sequence_cache,
};
