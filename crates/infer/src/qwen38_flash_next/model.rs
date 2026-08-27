use super::qsa::{Qwen38QsaPrefillWorkspace, Qwen38QsaWeights, Qwen38QsaWorkspace};
use super::{
    Qwen38ExactPleWorkspace, Qwen38FlashNextConfig, Qwen38HyperConnectionWeights,
    Qwen38HyperConnectionWorkspace, Qwen38PagedPle, Qwen38PleState, Qwen38PleTokenWindow,
    Qwen38PleWeights, Qwen38PleWorkspace,
};
use crate::nvfp4::{
    CublasLt, CudaStream, DeviceBuffer, Error, GpuSampledToken, GpuSamplingRow, GpuTokenSampler,
    ModelOptCheckpoint, Result, add_f32_into_on_stream, qwen38_repeat_streams_f32_into_on_stream,
    rms_norm_f32_into_on_stream,
};
use crate::qwen3::infer::{QwenLayerKind, QwenModelManifest};
use crate::qwen3::qwen36::{
    Bf16Linear, Qwen36BatchModelView, Qwen36Embedding, Qwen36ExactMoePairWorkspace,
    Qwen36HybridPrefillWorkspace, Qwen36LinearAttentionState, Qwen36LinearAttentionWeights,
    Qwen36LinearAttentionWorkspace, Qwen36LmHead, Qwen36LmHeadWorkspace, Qwen36MoeProbeSnapshot,
    Qwen36MoeWeights, Qwen36MoeWorkspace, load_hybrid_full_attention, load_hybrid_linear_attention,
    read_bf16_vector_delta_as_f32_device,
};
use crate::qwen38_flash_next::{
    Qwen38FlashNextMtpSequenceCache, Qwen38FlashNextSequence, Qwen38FlashNextSequenceCache,
    qwen38_flash_next_cache_error,
};
use crate::sm12x_cache::{Sm12xCacheContext, Sm12xPageTable};
use seqcache::{AdmissionOutcome, AdmissionRequest, AppendReservation, SequenceId};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_QWEN38_FLASH_NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

enum Qwen38AttentionWeights {
    Linear(Qwen36LinearAttentionWeights),
    Qsa(Qwen38QsaWeights),
}

enum Qwen38AttentionWorkspace {
    Linear(Qwen36LinearAttentionWorkspace),
    Qsa(Qwen38QsaWorkspace),
}

impl Qwen38AttentionWorkspace {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Linear(workspace) => workspace.device_bytes(),
            Self::Qsa(workspace) => workspace.device_bytes(),
        }
    }
}

enum Qwen38AttentionState {
    Linear(Qwen36LinearAttentionState),
    Qsa,
}

struct Qwen38Layer {
    attention_hyper: Qwen38HyperConnectionWeights,
    mlp_hyper: Qwen38HyperConnectionWeights,
    attention: Qwen38AttentionWeights,
    moe: Qwen36MoeWeights,
}

/// Fully resident neural body with a direct-paged BF16 PLE table.
pub struct Qwen38FlashNextModel {
    model_id: u64,
    config: Qwen38FlashNextConfig,
    manifest: QwenModelManifest,
    checkpoint: ModelOptCheckpoint,
    artifact_dir: PathBuf,
    lt: CublasLt,
    embedding: Qwen36Embedding,
    layers: Vec<Qwen38Layer>,
    ple_weights: Qwen38PleWeights,
    final_mixer: Qwen38HyperConnectionWeights,
    lm_head: Qwen36LmHead,
    mtp: Option<Box<Qwen38FlashNextMtpWeights>>,
}

struct Qwen38FlashNextMtpWeights {
    manifest: QwenModelManifest,
    pre_fc_norm_embedding: DeviceBuffer<f32>,
    pre_fc_norm_hidden: DeviceBuffer<f32>,
    fc_embedding: Bf16Linear,
    fc_hidden: Bf16Linear,
    attention_hyper: Qwen38HyperConnectionWeights,
    attention: Qwen38QsaWeights,
    mlp_hyper: Qwen38HyperConnectionWeights,
    moe: Qwen36MoeWeights,
    final_mixer: Qwen38HyperConnectionWeights,
}

/// Persistent private QSA position for one native MTP drafter.
pub(crate) struct Qwen38FlashNextMtpSequenceState {
    pub(crate) cache_id: SequenceId,
    pub(crate) page_table: Sm12xPageTable,
    position: usize,
    max_tokens: usize,
}

/// Reusable one-row storage for the released native MTP block.
pub(crate) struct Qwen38FlashNextMtpWorkspace {
    token: DeviceBuffer<u32>,
    embedded: DeviceBuffer<f32>,
    normed_embedding: DeviceBuffer<f32>,
    projected_embedding: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    projected_hidden: DeviceBuffer<f32>,
    repeated_embedding: DeviceBuffer<f32>,
    streams_a: DeviceBuffer<f32>,
    streams_b: DeviceBuffer<f32>,
    zero_hidden: DeviceBuffer<f32>,
    attention_hyper: Qwen38HyperConnectionWorkspace,
    attention: Qwen38QsaWorkspace,
    attention_output: DeviceBuffer<f32>,
    mlp_hyper: Qwen38HyperConnectionWorkspace,
    moe: Qwen36MoeWorkspace,
    final_hyper: Qwen38HyperConnectionWorkspace,
    final_hidden: DeviceBuffer<f32>,
    lm_head: Qwen36LmHeadWorkspace,
    prefill: Box<Qwen38FlashNextMtpPrefillWorkspace>,
}

/// Batched prompt-cache preparation for the released native MTP block.
struct Qwen38FlashNextMtpPrefillWorkspace {
    token_capacity: usize,
    tokens: DeviceBuffer<u32>,
    embedded: DeviceBuffer<f32>,
    normed_embedding: DeviceBuffer<f32>,
    projected_embedding: DeviceBuffer<f32>,
    previous_target_streams: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    projected_hidden: DeviceBuffer<f32>,
    repeated_embedding: DeviceBuffer<f32>,
    streams: DeviceBuffer<f32>,
    attention_hyper: Qwen38HyperConnectionWorkspace,
    attention: Qwen38QsaPrefillWorkspace,
}

/// Mutable one-sequence state and reusable single-token workspace.
pub struct Qwen38FlashNextDecodeState {
    model_id: u64,
    stream: CudaStream,
    token_id: DeviceBuffer<u32>,
    streams_a: DeviceBuffer<f32>,
    streams_b: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    zero_hidden: DeviceBuffer<f32>,
    attention_hyper: Qwen38HyperConnectionWorkspace,
    mlp_hyper: Qwen38HyperConnectionWorkspace,
    final_hyper: Qwen38HyperConnectionWorkspace,
    attention_workspaces: Vec<Qwen38AttentionWorkspace>,
    attention_states: Vec<Qwen38AttentionState>,
    rollback_linear_states: Vec<Option<Qwen36LinearAttentionState>>,
    moe: Qwen36MoeWorkspace,
    ple_pager: Qwen38PagedPle,
    ple_window: Qwen38PleTokenWindow,
    ple_state: Qwen38PleState,
    ple_workspace: Qwen38PleWorkspace,
    lm_head: Qwen36LmHeadWorkspace,
    position: usize,
    max_tokens: usize,
}

/// Shared vectorized workspace for one scheduler-selected prompt chunk.
pub(crate) struct Qwen38FlashNextPrefillWorkspace {
    token_capacity: usize,
    token_ids: DeviceBuffer<u32>,
    streams_a: DeviceBuffer<f32>,
    streams_b: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    qsa: Qwen38QsaPrefillWorkspace,
    qsa_serial_output: DeviceBuffer<f32>,
    qsa_serial_row_hidden: DeviceBuffer<f32>,
    attention_hyper: Qwen38HyperConnectionWorkspace,
    mlp_hyper: Qwen38HyperConnectionWorkspace,
    ple_pager: Qwen38PagedPle,
    ple: Qwen38PleWorkspace,
    exact_ple: Option<Qwen38ExactPleWorkspace>,
    hybrid: Qwen36HybridPrefillWorkspace,
    exact_gdn: Option<Qwen38ExactGdnPrefillWorkspace>,
    exact_moe: Option<Qwen38ExactMoePrefillWorkspace>,
    serial_gdn_projections: bool,
    serial_qsa_rows: bool,
    canonical_moe_linears: bool,
    exact_hyper_projections: bool,
    layer_trace: Option<Vec<Qwen38LayerProbeTrace>>,
    linear_layers: Vec<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen38LayerProbeStage {
    Ple,
    Attention,
    MlpMix,
    MlpFfn,
    Mlp,
}

pub(crate) struct Qwen38LayerProbeTrace {
    pub(crate) layer: usize,
    pub(crate) stage: Qwen38LayerProbeStage,
    pub(crate) rows: usize,
    pub(crate) streams: Vec<f32>,
    pub(crate) moe: Option<Vec<Qwen36MoeProbeSnapshot>>,
}

fn capture_layer_probe_trace(
    layer: usize,
    stage: Qwen38LayerProbeStage,
    rows: usize,
    hc_dim: usize,
    streams: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<Qwen38LayerProbeTrace> {
    Ok(Qwen38LayerProbeTrace {
        layer,
        stage,
        rows,
        streams: streams
            .copy_prefix_to_host(rows * hc_dim, stream)?
            .into_vec(),
        moe: None,
    })
}

struct Qwen38ExactGdnPrefillWorkspace {
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    gdn: Qwen36LinearAttentionWorkspace,
}

impl Qwen38ExactGdnPrefillWorkspace {
    fn new(
        manifest: &QwenModelManifest,
        weights: &Qwen36LinearAttentionWeights,
        token_capacity: usize,
    ) -> Result<Self> {
        let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
            label: "Qwen3.8 exact GDN verifier",
            detail: "manifest has no linear-attention configuration".to_string(),
        })?;
        Ok(Self {
            input: DeviceBuffer::zeroed(manifest.hidden)?,
            output: DeviceBuffer::zeroed(token_capacity * manifest.hidden)?,
            gdn: Qwen36LinearAttentionWorkspace::new(manifest, linear, weights)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'a>(
        &'a mut self,
        weights: &Qwen36LinearAttentionWeights,
        state: &mut Qwen36LinearAttentionState,
        hidden: &DeviceBuffer<f32>,
        tokens: usize,
        hidden_width: usize,
        rms_eps: f32,
        mut snapshot: Option<(&mut Qwen36HybridPrefillWorkspace, usize)>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        for row in 0..tokens {
            self.input.copy_range_from_device_on_stream(
                0,
                hidden,
                row * hidden_width,
                hidden_width,
                stream,
            )?;
            let step = weights.run_one_token_sigmoid_output_gate(
                &mut self.gdn,
                state,
                &self.input,
                rms_eps,
                stream,
            )?;
            if row == 0
                && let Some((hybrid, layer)) = snapshot.as_mut()
            {
                hybrid.capture_gdn_state_snapshot(*layer, 0, stream)?;
            }
            self.output.copy_range_from_device_on_stream(
                row * hidden_width,
                step.output,
                0,
                hidden_width,
                stream,
            )?;
        }
        Ok(&self.output)
    }
}

struct Qwen38ExactMoePrefillWorkspace {
    pair: Qwen36ExactMoePairWorkspace,
    oracle_inputs: Vec<DeviceBuffer<f32>>,
    oracle_rows: Vec<Qwen36MoeWorkspace>,
    zero_hidden: DeviceBuffer<f32>,
    oracle_snapshots: Option<Vec<Qwen36MoeProbeSnapshot>>,
}

impl Qwen38ExactMoePrefillWorkspace {
    fn new(manifest: &QwenModelManifest, token_capacity: usize) -> Result<Self> {
        if token_capacity != 2 {
            return Err(Error::Shape {
                label: "Qwen3.8 exact MoE verifier capacity",
                expected: "exactly two rows".to_string(),
                actual: token_capacity.to_string(),
            });
        }
        Ok(Self {
            pair: Qwen36ExactMoePairWorkspace::new(manifest)?,
            oracle_inputs: (0..2)
                .map(|_| DeviceBuffer::zeroed(manifest.hidden))
                .collect::<Result<Vec<_>>>()?,
            oracle_rows: (0..2)
                .map(|_| Qwen36MoeWorkspace::new(manifest))
                .collect::<Result<Vec<_>>>()?,
            zero_hidden: DeviceBuffer::zeroed(manifest.hidden)?,
            oracle_snapshots: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'a>(
        &'a mut self,
        _lt: &CublasLt,
        manifest: &QwenModelManifest,
        weights: &Qwen36MoeWeights,
        hidden: &DeviceBuffer<f32>,
        tokens: usize,
        capture_oracle: bool,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if tokens != 2 {
            return Err(Error::Shape {
                label: "Qwen3.8 exact MoE verifier rows",
                expected: "exactly two rows".to_string(),
                actual: tokens.to_string(),
            });
        }
        weights.run_exact_pair(&mut self.pair, manifest, hidden, stream)?;
        self.oracle_snapshots = None;
        if capture_oracle {
            let mut snapshots = Vec::with_capacity(2);
            for row in 0..2 {
                self.oracle_inputs[row].copy_range_from_device_on_stream(
                    0,
                    hidden,
                    row * manifest.hidden,
                    manifest.hidden,
                    stream,
                )?;
                weights.run_one_token(
                    _lt,
                    &mut self.oracle_rows[row],
                    manifest,
                    &self.oracle_inputs[row],
                    &self.zero_hidden,
                    stream,
                    None,
                    None,
                )?;
                snapshots.push(self.oracle_rows[row].probe_snapshot(stream)?);
            }
            self.oracle_snapshots = Some(snapshots);
        }
        Ok(self.pair.output())
    }

    fn probe_snapshots(
        &mut self,
        weights: &Qwen36MoeWeights,
        stream: &CudaStream,
    ) -> Result<Vec<Qwen36MoeProbeSnapshot>> {
        let mut snapshots = weights.probe_repeat_exact_pair_gate_up(&mut self.pair, stream)?;
        if let Some(oracle) = self.oracle_snapshots.as_ref() {
            for (snapshot, oracle) in snapshots.iter_mut().zip(oracle) {
                snapshot.oracle_routed_gate_up = Some(oracle.routed_gate_up.clone());
            }
        }
        Ok(snapshots)
    }
}

/// Immutable page-aligned snapshot of Flash Next's non-pageable sequence state.
///
/// QSA K/V and index-key pages remain owned by the shared sequence cache.
pub struct Qwen38FlashNextSequenceSnapshot {
    model_id: u64,
    position: usize,
    linear_states: Vec<Option<Qwen36LinearAttentionState>>,
    ple_window: Qwen38PleTokenWindow,
    ple_conv: DeviceBuffer<f32>,
    frontier_streams: DeviceBuffer<f32>,
    device_bytes: usize,
}

impl Qwen38FlashNextSequenceSnapshot {
    /// Returns the page-aligned position represented by this snapshot.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns exact device bytes retained outside shared QSA pages.
    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }
}

/// Greedy next-token result from the native decode path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38NextToken {
    /// Argmax token identifier.
    pub id: u32,
    /// Winning logit.
    pub value: f32,
}

/// Committed-but-unprocessed token and the target streams that produced it.
pub(crate) struct Qwen38FlashNextSpeculativeFrontier {
    pub(crate) token: u32,
    pub(crate) logit: f32,
    pub(crate) previous_streams: DeviceBuffer<f32>,
}

pub(crate) struct Qwen38FlashNextSpeculativeOutcome {
    pub(crate) committed: Vec<Qwen38NextToken>,
    pub(crate) accepted_drafts: usize,
}

pub(crate) struct Qwen38FlashNextSpeculativeWorkspace {
    verify: Qwen38FlashNextVectorVerifierProbeWorkspace,
    host_tokens: Vec<u32>,
}

pub(crate) struct Qwen38FlashNextVectorVerifierProbeWorkspace {
    verify: Qwen38FlashNextPrefillWorkspace,
    final_hyper: Qwen38HyperConnectionWorkspace,
    final_hidden: DeviceBuffer<f32>,
    exact_logits: DeviceBuffer<f32>,
    top1_scratch_values: DeviceBuffer<f32>,
    top1_scratch_indices: DeviceBuffer<u32>,
    argmax_indices: DeviceBuffer<u32>,
    argmax_values: DeviceBuffer<f32>,
    exact_hyper_projections: bool,
    exact_output_head: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38VectorVerifierProbeMode {
    Fast,
    SerialGdnProjections,
    CanonicalMoeLinears,
    CanonicalMoeLinearsSerialGdn,
    ExactMoe,
    Exact,
}

impl Qwen38VectorVerifierProbeMode {
    fn serial_gdn_projections(self) -> bool {
        matches!(
            self,
            Self::SerialGdnProjections | Self::CanonicalMoeLinearsSerialGdn | Self::Exact
        )
    }

    fn canonical_moe_linears(self) -> bool {
        matches!(
            self,
            Self::CanonicalMoeLinears | Self::CanonicalMoeLinearsSerialGdn
        )
    }

    fn exact_moe(self) -> bool {
        matches!(self, Self::ExactMoe | Self::Exact)
    }

    fn exact_hyper_projections(self) -> bool {
        matches!(self, Self::Exact)
    }

    fn exact_ple(self) -> bool {
        matches!(self, Self::Exact)
    }

    fn exact_output_head(self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Vocabulary-head work requested for one committed token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38LogitsMode {
    /// Skip the vocabulary projection for an intermediate prompt token.
    None,
    /// Compute logits and their device-side argmax.
    Top1,
    /// Compute full logits for top-k/top-p sampling.
    Full,
}

impl Qwen38FlashNextModel {
    /// Loads the released Inferact checkpoint without materializing the PLE table.
    pub fn open(model_dir: impl AsRef<Path>, artifact_dir: impl Into<PathBuf>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let manifest = config.qwen_manifest();
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let artifact_dir = artifact_dir.into();
        let lt = CublasLt::new()?;
        let embedding = Qwen36Embedding::load(
            &checkpoint,
            "model.language_model.embed_tokens",
            config.vocab,
            config.hidden,
        )?;
        let ple_weights = Qwen38PleWeights::load(&checkpoint, &config)?;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let prefix = format!("model.language_model.layers.{layer}");
            let attention_hyper = Qwen38HyperConnectionWeights::load(
                &checkpoint,
                &format!("{prefix}.attn_hyper_connection"),
                &config,
                true,
            )?;
            let mlp_hyper = Qwen38HyperConnectionWeights::load(
                &checkpoint,
                &format!("{prefix}.mlp_hyper_connection"),
                &config,
                true,
            )?;
            let attention = match manifest.layer_kinds[layer] {
                QwenLayerKind::LinearAttention => Qwen38AttentionWeights::Linear(
                    load_hybrid_linear_attention(&checkpoint, &manifest, &artifact_dir, layer)?,
                ),
                QwenLayerKind::FullAttention => {
                    let attention =
                        load_hybrid_full_attention(&checkpoint, &manifest, &artifact_dir, layer)?;
                    Qwen38AttentionWeights::Qsa(Qwen38QsaWeights::load(
                        &checkpoint,
                        &config,
                        layer,
                        attention,
                    )?)
                }
            };
            let moe = Qwen36MoeWeights::load_checkpoint_layout(
                &checkpoint,
                &manifest,
                &artifact_dir,
                layer,
            )?;
            layers.push(Qwen38Layer {
                attention_hyper,
                mlp_hyper,
                attention,
                moe,
            });
        }
        let final_mixer = Qwen38HyperConnectionWeights::load(
            &checkpoint,
            "model.language_model.hyper_connection_mixer",
            &config,
            false,
        )?;
        let lm_head = Qwen36LmHead::load_bf16(&checkpoint, config.vocab, config.hidden)?;
        if lm_head.shape() != (config.vocab, config.hidden) {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next lm_head",
                expected: format!("[{}, {}]", config.vocab, config.hidden),
                actual: format!("{:?}", lm_head.shape()),
            });
        }
        Ok(Self {
            model_id: NEXT_QWEN38_FLASH_NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            manifest,
            checkpoint,
            artifact_dir,
            lt,
            embedding,
            layers,
            ple_weights,
            final_mixer,
            lm_head,
            mtp: None,
        })
    }

    /// Loads the released one-layer QSA/MoE MTP drafter on demand.
    pub fn enable_mtp(&mut self) -> Result<()> {
        if self.mtp.is_some() {
            return Ok(());
        }
        if self.config.mtp_layers != 1
            || !self.checkpoint.contains_tensor("mtp.fc_embedding.weight")
        {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next MTP",
                detail: "checkpoint does not contain the released one-layer MTP block".to_string(),
            });
        }
        let hidden = self.config.hidden;
        let hc_dim = hidden * self.config.hc_count;
        let mut manifest = self.manifest.clone();
        manifest.tensor_prefix = "mtp".to_string();
        manifest.layers = 1;
        manifest.layer_kinds = vec![QwenLayerKind::FullAttention];
        manifest.linear_attention = None;
        manifest.mtp_layers = 0;
        let artifact_dir = self.artifact_dir.join("mtp");
        let attention = load_hybrid_full_attention(&self.checkpoint, &manifest, &artifact_dir, 0)?;
        let attention = Qwen38QsaWeights::load_at_prefix(
            &self.checkpoint,
            &self.config,
            "mtp.layers.0.self_attn.indexer",
            attention,
        )?;
        let mtp = Qwen38FlashNextMtpWeights {
            manifest: manifest.clone(),
            pre_fc_norm_embedding: read_bf16_vector_delta_as_f32_device(
                &self.checkpoint,
                "mtp.pre_fc_norm_embedding.weight",
                hidden,
            )?,
            pre_fc_norm_hidden: read_bf16_vector_delta_as_f32_device(
                &self.checkpoint,
                "mtp.pre_fc_norm_hidden.weight",
                hc_dim,
            )?,
            fc_embedding: Bf16Linear::load(
                &self.checkpoint,
                "mtp.fc_embedding.weight",
                hidden,
                hidden,
            )?,
            fc_hidden: Bf16Linear::load(&self.checkpoint, "mtp.fc_hidden.weight", hidden, hidden)?,
            attention_hyper: Qwen38HyperConnectionWeights::load(
                &self.checkpoint,
                "mtp.layers.0.attn_hyper_connection",
                &self.config,
                true,
            )?,
            attention,
            mlp_hyper: Qwen38HyperConnectionWeights::load(
                &self.checkpoint,
                "mtp.layers.0.mlp_hyper_connection",
                &self.config,
                true,
            )?,
            moe: Qwen36MoeWeights::load_checkpoint_layout(
                &self.checkpoint,
                &manifest,
                &artifact_dir,
                0,
            )?,
            final_mixer: Qwen38HyperConnectionWeights::load(
                &self.checkpoint,
                "mtp.hyper_connection_mixer",
                &self.config,
                false,
            )?,
        };
        self.mtp = Some(Box::new(mtp));
        Ok(())
    }

    pub(crate) fn mtp_enabled(&self) -> bool {
        self.mtp.is_some()
    }

    pub(crate) fn new_mtp_workspace(
        &self,
        max_tokens: usize,
        prefill_token_capacity: usize,
    ) -> Result<Qwen38FlashNextMtpWorkspace> {
        let mtp = self.mtp.as_deref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next MTP workspace",
            detail: "MTP weights are not enabled".to_string(),
        })?;
        let hidden = self.config.hidden;
        let hc_dim = hidden * self.config.hc_count;
        if prefill_token_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP prefill capacity",
                expected: "positive token capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let mtp_model = Qwen36BatchModelView::new(&self.lt, &mtp.manifest, &[false]);
        let prefill = Qwen38FlashNextMtpPrefillWorkspace {
            token_capacity: prefill_token_capacity,
            tokens: DeviceBuffer::zeroed(prefill_token_capacity)?,
            embedded: DeviceBuffer::zeroed(prefill_token_capacity * hidden)?,
            normed_embedding: DeviceBuffer::zeroed(prefill_token_capacity * hidden)?,
            projected_embedding: DeviceBuffer::zeroed(prefill_token_capacity * hidden)?,
            previous_target_streams: DeviceBuffer::zeroed(prefill_token_capacity * hc_dim)?,
            normed_hidden: DeviceBuffer::zeroed(prefill_token_capacity * hc_dim)?,
            projected_hidden: DeviceBuffer::zeroed(prefill_token_capacity * hc_dim)?,
            repeated_embedding: DeviceBuffer::zeroed(prefill_token_capacity * hc_dim)?,
            streams: DeviceBuffer::zeroed(prefill_token_capacity * hc_dim)?,
            attention_hyper: Qwen38HyperConnectionWorkspace::new_prefill(
                &self.config,
                prefill_token_capacity,
            )?,
            attention: mtp.attention.new_prefill_workspace(
                &mtp_model,
                &self.config,
                prefill_token_capacity,
                1,
            )?,
        };
        Ok(Qwen38FlashNextMtpWorkspace {
            token: DeviceBuffer::zeroed(1)?,
            embedded: DeviceBuffer::zeroed(hidden)?,
            normed_embedding: DeviceBuffer::zeroed(hidden)?,
            projected_embedding: DeviceBuffer::zeroed(hidden)?,
            normed_hidden: DeviceBuffer::zeroed(hc_dim)?,
            projected_hidden: DeviceBuffer::zeroed(hc_dim)?,
            repeated_embedding: DeviceBuffer::zeroed(hc_dim)?,
            streams_a: DeviceBuffer::zeroed(hc_dim)?,
            streams_b: DeviceBuffer::zeroed(hc_dim)?,
            zero_hidden: DeviceBuffer::zeroed(hidden)?,
            attention_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, 1)?,
            attention: Qwen38QsaWorkspace::new(
                &self.config,
                &mtp.manifest,
                &mtp.attention,
                max_tokens,
            )?,
            attention_output: DeviceBuffer::zeroed(hidden)?,
            mlp_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, 1)?,
            moe: Qwen36MoeWorkspace::new(&mtp.manifest)?,
            final_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, 1)?,
            final_hidden: DeviceBuffer::zeroed(hidden)?,
            lm_head: Qwen36LmHeadWorkspace::new(self.config.vocab, hidden)?,
            prefill: Box::new(prefill),
        })
    }

    pub(crate) fn new_speculative_workspace(
        &self,
        drafts: usize,
    ) -> Result<Qwen38FlashNextSpeculativeWorkspace> {
        if drafts != 1 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next speculative drafts",
                expected: "exactly one draft in the initial native MTP path".to_string(),
                actual: drafts.to_string(),
            });
        }
        Ok(Qwen38FlashNextSpeculativeWorkspace {
            verify: self
                .new_vector_verifier_probe_workspace(2, Qwen38VectorVerifierProbeMode::Exact)?,
            host_tokens: Vec::with_capacity(2),
        })
    }

    pub(crate) fn new_vector_verifier_probe_workspace(
        &self,
        rows: usize,
        mode: Qwen38VectorVerifierProbeMode,
    ) -> Result<Qwen38FlashNextVectorVerifierProbeWorkspace> {
        if rows < 2 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next verifier probe rows",
                expected: "at least two rows".to_string(),
                actual: rows.to_string(),
            });
        }
        let mut verify = self.new_prefill_workspace(rows)?;
        let batch_model =
            Qwen36BatchModelView::new(&self.lt, &self.manifest, &verify.linear_layers);
        verify
            .hybrid
            .enable_state_snapshots(&batch_model, rows - 1)?;
        verify.serial_gdn_projections = mode.serial_gdn_projections();
        verify.serial_qsa_rows = true;
        if mode.serial_gdn_projections() {
            let first_linear = self
                .layers
                .iter()
                .find_map(|layer| match &layer.attention {
                    Qwen38AttentionWeights::Linear(weights) => Some(weights),
                    Qwen38AttentionWeights::Qsa(_) => None,
                })
                .ok_or_else(|| Error::Format {
                    label: "Qwen3.8 exact GDN verifier",
                    detail: "model has no GDN layer".to_string(),
                })?;
            verify.exact_gdn = Some(Qwen38ExactGdnPrefillWorkspace::new(
                &self.manifest,
                first_linear,
                rows,
            )?);
        }
        verify.canonical_moe_linears = mode.canonical_moe_linears();
        verify.exact_hyper_projections = mode.exact_hyper_projections();
        if mode.exact_ple() {
            verify.exact_ple = Some(Qwen38ExactPleWorkspace::new(&self.config)?);
        }
        if mode.exact_moe() {
            verify.exact_moe = Some(Qwen38ExactMoePrefillWorkspace::new(&self.manifest, rows)?);
        }
        Ok(Qwen38FlashNextVectorVerifierProbeWorkspace {
            verify,
            final_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, rows)?,
            final_hidden: DeviceBuffer::zeroed(rows * self.config.hidden)?,
            exact_logits: DeviceBuffer::zeroed(rows * self.config.vocab)?,
            top1_scratch_values: DeviceBuffer::zeroed(rows * self.config.vocab.div_ceil(8))?,
            top1_scratch_indices: DeviceBuffer::zeroed(rows * self.config.vocab.div_ceil(8))?,
            argmax_indices: DeviceBuffer::zeroed(rows)?,
            argmax_values: DeviceBuffer::zeroed(rows)?,
            exact_hyper_projections: mode.exact_hyper_projections(),
            exact_output_head: mode.exact_output_head(),
        })
    }

    pub(crate) fn new_mtp_sequence_state(
        &self,
        cache: &mut Qwen38FlashNextMtpSequenceCache,
        max_tokens: usize,
        prompt_tokens: &[u32],
        stream: &CudaStream,
    ) -> Result<Qwen38FlashNextMtpSequenceState> {
        if !self.mtp_enabled() {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next MTP sequence",
                detail: "MTP weights are not enabled".to_string(),
            });
        }
        let mut page_table = Sm12xPageTable::new(max_tokens)?;
        let prefix = cache.lookup_prefix(prompt_tokens);
        let mut restored_position = 0usize;
        let outcome = cache
            .admit(
                prefix,
                AdmissionRequest {
                    max_position: max_tokens,
                    private_state_bytes: 0,
                    page_table_bytes: page_table.managed_bytes(),
                    allow_emergency: false,
                },
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut page_table,
                },
                |_snapshot, position| {
                    restored_position = position;
                    Ok(())
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        let AdmissionOutcome::Admitted(cache_id) = outcome else {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next MTP admission",
                detail: "configured drafter cache has insufficient capacity".to_string(),
            });
        };
        Ok(Qwen38FlashNextMtpSequenceState {
            cache_id,
            page_table,
            position: restored_position,
            max_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mtp_forward_token(
        &self,
        state: &mut Qwen38FlashNextMtpSequenceState,
        workspace: &mut Qwen38FlashNextMtpWorkspace,
        cache: &mut Qwen38FlashNextMtpSequenceCache,
        token: u32,
        previous_target_streams: &DeviceBuffer<f32>,
        logits: bool,
        stream: &CudaStream,
    ) -> Result<Option<Qwen38NextToken>> {
        let mtp = self.mtp.as_deref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next MTP forward",
            detail: "MTP weights are not enabled".to_string(),
        })?;
        if state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP position",
                expected: format!("position < {}", state.max_tokens),
                actual: state.position.to_string(),
            });
        }
        let hidden = self.config.hidden;
        let hc_dim = hidden * self.config.hc_count;
        if previous_target_streams.len() < hc_dim {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP target streams",
                expected: format!("at least {hc_dim} values"),
                actual: previous_target_streams.len().to_string(),
            });
        }
        workspace.token.copy_from_host(&[token])?;
        self.embedding.gather_prefix(
            self.config.vocab,
            hidden,
            &workspace.token,
            workspace.embedded.output(),
            1,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            hidden,
            &workspace.embedded,
            &mtp.pre_fc_norm_embedding,
            workspace.normed_embedding.output(),
            self.config.rms_eps(),
            stream,
        )?;
        mtp.fc_embedding.run_into(
            &workspace.normed_embedding,
            &mut workspace.projected_embedding,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            hc_dim,
            previous_target_streams,
            &mtp.pre_fc_norm_hidden,
            workspace.normed_hidden.output(),
            self.config.rms_eps(),
            stream,
        )?;
        mtp.fc_hidden.run_batch_into(
            &workspace.normed_hidden,
            &mut workspace.projected_hidden,
            self.config.hc_count,
            stream,
        )?;
        qwen38_repeat_streams_f32_into_on_stream(
            &workspace.projected_embedding,
            workspace.repeated_embedding.output(),
            1,
            hidden,
            self.config.hc_count,
            stream,
        )?;
        add_f32_into_on_stream(
            &workspace.projected_hidden,
            &workspace.repeated_embedding,
            workspace.streams_a.output(),
            stream,
        )?;
        mtp.attention_hyper.mix(
            &workspace.streams_a,
            &mut workspace.attention_hyper,
            1,
            stream,
        )?;

        let reservation = cache
            .reserve_append(
                state.cache_id,
                1,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut state.page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        let attention = cache
            .with_append_pages(&reservation, |backend, pages| {
                let page = pages.iter().next().ok_or_else(|| Error::Format {
                    label: "Qwen3.8 Flash Next MTP QSA append",
                    detail: "one-token reservation contains no physical page".to_string(),
                })?;
                let output = mtp.attention.run_one_token(
                    &mut workspace.attention,
                    backend,
                    state.page_table.device(),
                    page.page(),
                    page.segment().page_offset(),
                    &self.config,
                    &mtp.manifest,
                    workspace.attention_hyper.mixed(),
                    0,
                    state.position,
                    stream,
                )?;
                workspace
                    .attention_output
                    .copy_prefix_from_device_on_stream(output, hidden, stream)?;
                Ok(())
            })
            .map_err(qwen38_flash_next_cache_error);
        if let Err(error) = attention {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream,
                        page_table: &mut state.page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        let body = (|| -> Result<Option<Qwen38NextToken>> {
            mtp.attention_hyper.combine(
                &workspace.streams_a,
                &workspace.attention_output,
                &mut workspace.attention_hyper,
                &mut workspace.streams_b,
                1,
                stream,
            )?;
            std::mem::swap(&mut workspace.streams_a, &mut workspace.streams_b);
            mtp.mlp_hyper
                .mix(&workspace.streams_a, &mut workspace.mlp_hyper, 1, stream)?;
            let ffn = mtp.moe.run_one_token(
                &self.lt,
                &mut workspace.moe,
                &mtp.manifest,
                workspace.mlp_hyper.mixed(),
                &workspace.zero_hidden,
                stream,
                None,
                None,
            )?;
            mtp.mlp_hyper.combine(
                &workspace.streams_a,
                ffn.ffn_out,
                &mut workspace.mlp_hyper,
                &mut workspace.streams_b,
                1,
                stream,
            )?;
            std::mem::swap(&mut workspace.streams_a, &mut workspace.streams_b);
            if !logits {
                return Ok(None);
            }
            mtp.final_mixer
                .mix(&workspace.streams_a, &mut workspace.final_hyper, 1, stream)?;
            workspace.final_hidden.copy_prefix_from_device_on_stream(
                workspace.final_hyper.mixed(),
                hidden,
                stream,
            )?;
            self.lm_head.run_top1(
                &self.lt,
                &workspace.final_hidden,
                &mut workspace.lm_head,
                stream,
            )?;
            let (id, value) = workspace.lm_head.read_top1(stream)?;
            Ok(Some(Qwen38NextToken { id, value }))
        })();
        match body {
            Ok(next) => {
                cache
                    .commit_append(
                        reservation,
                        1,
                        &mut Sm12xCacheContext {
                            stream,
                            page_table: &mut state.page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                state.position += 1;
                Ok(next)
            }
            Err(error) => {
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream,
                            page_table: &mut state.page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                Err(error)
            }
        }
    }

    /// Advances the native MTP prompt cache without evaluating discarded
    /// attention and MoE outputs for every prompt row.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mtp_prefill_tokens(
        &self,
        state: &mut Qwen38FlashNextMtpSequenceState,
        workspace: &mut Qwen38FlashNextMtpWorkspace,
        cache: &mut Qwen38FlashNextMtpSequenceCache,
        tokens: &[u32],
        target_prefill: &Qwen38FlashNextPrefillWorkspace,
        initial_previous_target_streams: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let mtp = self.mtp.as_deref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next MTP prefill",
            detail: "MTP weights are not enabled".to_string(),
        })?;
        let rows = tokens.len();
        if rows == 0 || rows > workspace.prefill.token_capacity {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP prefill tokens",
                expected: format!("1..={} tokens", workspace.prefill.token_capacity),
                actual: rows.to_string(),
            });
        }
        let end = state
            .position
            .checked_add(rows)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 Flash Next MTP prefill position",
                expected: "position + tokens without overflow".to_string(),
                actual: format!("position={} tokens={rows}", state.position),
            })?;
        if end > state.max_tokens {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP prefill position",
                expected: format!("end <= {}", state.max_tokens),
                actual: end.to_string(),
            });
        }
        let hidden = self.config.hidden;
        let hc_dim = hidden * self.config.hc_count;
        if initial_previous_target_streams.len() < hc_dim {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP prefill target streams",
                expected: format!("at least {hc_dim} initial values"),
                actual: initial_previous_target_streams.len().to_string(),
            });
        }
        if target_prefill.streams_a.len() < rows * hc_dim {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next MTP prefill target rows",
                expected: format!("at least {} target values", rows * hc_dim),
                actual: target_prefill.streams_a.len().to_string(),
            });
        }

        let reservation = cache
            .reserve_append(
                state.cache_id,
                rows,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut state.page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        let body = (|| -> Result<()> {
            let prefill = workspace.prefill.as_mut();
            prefill.tokens.copy_prefix_from_host(tokens)?;
            self.embedding.gather_prefix(
                self.config.vocab,
                hidden,
                &prefill.tokens,
                prefill.embedded.output(),
                rows,
                stream,
            )?;
            rms_norm_f32_into_on_stream(
                rows,
                hidden,
                &prefill.embedded,
                &mtp.pre_fc_norm_embedding,
                prefill.normed_embedding.output(),
                self.config.rms_eps(),
                stream,
            )?;
            mtp.fc_embedding.run_batch_into(
                &prefill.normed_embedding,
                &mut prefill.projected_embedding,
                rows,
                stream,
            )?;
            prefill
                .previous_target_streams
                .copy_range_from_device_on_stream(
                    0,
                    initial_previous_target_streams,
                    0,
                    hc_dim,
                    stream,
                )?;
            if rows > 1 {
                prefill
                    .previous_target_streams
                    .copy_range_from_device_on_stream(
                        hc_dim,
                        &target_prefill.streams_a,
                        0,
                        (rows - 1) * hc_dim,
                        stream,
                    )?;
            }
            rms_norm_f32_into_on_stream(
                rows,
                hc_dim,
                &prefill.previous_target_streams,
                &mtp.pre_fc_norm_hidden,
                prefill.normed_hidden.output(),
                self.config.rms_eps(),
                stream,
            )?;
            mtp.fc_hidden.run_batch_into(
                &prefill.normed_hidden,
                &mut prefill.projected_hidden,
                rows * self.config.hc_count,
                stream,
            )?;
            qwen38_repeat_streams_f32_into_on_stream(
                &prefill.projected_embedding,
                prefill.repeated_embedding.output(),
                rows,
                hidden,
                self.config.hc_count,
                stream,
            )?;
            add_f32_into_on_stream(
                &prefill.projected_hidden,
                &prefill.repeated_embedding,
                prefill.streams.output(),
                stream,
            )?;
            mtp.attention_hyper.mix(
                &prefill.streams,
                &mut prefill.attention_hyper,
                rows,
                stream,
            )?;
            let mtp_model = Qwen36BatchModelView::new(&self.lt, &mtp.manifest, &[false]);
            mtp.attention.prepare_prefill(
                &mtp_model,
                &mut prefill.attention,
                &self.config,
                prefill.attention_hyper.mixed(),
                rows,
                state.position,
                stream,
            )?;
            cache
                .with_append_pages(&reservation, |backend, pages| {
                    for page in pages.iter() {
                        let segment = page.segment();
                        for offset in 0..segment.rows() {
                            let row = segment.input_offset() + offset;
                            mtp.attention.append_prepared_prefill_row(
                                &mtp_model,
                                &mut prefill.attention,
                                &mut workspace.attention,
                                backend,
                                page.page(),
                                segment.page_offset() + offset,
                                &self.config,
                                0,
                                row,
                                stream,
                            )?;
                        }
                    }
                    Ok(())
                })
                .map_err(qwen38_flash_next_cache_error)
        })();
        match body {
            Ok(()) => {
                if let Err(error) = cache.commit_append(
                    reservation.clone(),
                    rows,
                    &mut Sm12xCacheContext {
                        stream,
                        page_table: &mut state.page_table,
                    },
                ) {
                    cache
                        .abort_append(
                            reservation,
                            &mut Sm12xCacheContext {
                                stream,
                                page_table: &mut state.page_table,
                            },
                        )
                        .map_err(qwen38_flash_next_cache_error)?;
                    return Err(qwen38_flash_next_cache_error(error));
                }
                state.position = end;
                Ok(())
            }
            Err(error) => {
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream,
                            page_table: &mut state.page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn speculative_cycle_argmax(
        &mut self,
        workspace: &mut Qwen38FlashNextSpeculativeWorkspace,
        frontier: &mut Qwen38FlashNextSpeculativeFrontier,
        target_state: &mut Qwen38FlashNextDecodeState,
        target_cache: &mut Qwen38FlashNextSequenceCache,
        target_cache_id: SequenceId,
        target_page_table: &mut Sm12xPageTable,
        mtp_state: &mut Qwen38FlashNextMtpSequenceState,
        mtp_workspace: &mut Qwen38FlashNextMtpWorkspace,
        mtp_cache: &mut Qwen38FlashNextMtpSequenceCache,
    ) -> Result<Qwen38FlashNextSpeculativeOutcome> {
        let draft = self
            .mtp_forward_token(
                mtp_state,
                mtp_workspace,
                mtp_cache,
                frontier.token,
                &frontier.previous_streams,
                true,
                &target_state.stream,
            )?
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 Flash Next MTP draft",
                detail: "draft step produced no vocabulary result".to_string(),
            })?;
        let old_frontier = Qwen38NextToken {
            id: frontier.token,
            value: frontier.logit,
        };
        workspace.host_tokens.clear();
        workspace.host_tokens.push(frontier.token);
        workspace.host_tokens.push(draft.id);
        let rows = workspace.host_tokens.len();
        let reservation = target_cache
            .reserve_append(
                target_cache_id,
                rows,
                &mut Sm12xCacheContext {
                    stream: &target_state.stream,
                    page_table: target_page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        if let Err(error) = target_state.begin_append() {
            target_cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &target_state.stream,
                        page_table: target_page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        if let Err(error) = workspace
            .verify
            .verify
            .ple_pager
            .begin_read_tokens(&mut target_state.ple_window, &workspace.host_tokens)
        {
            target_state.abort_append()?;
            target_cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &target_state.stream,
                        page_table: target_page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        let verification = (|| -> Result<Vec<Qwen38NextToken>> {
            self.prefill_tokens_inner(
                target_state,
                &mut workspace.verify.verify,
                target_cache,
                &reservation,
                target_page_table,
                &workspace.host_tokens,
                Qwen38LogitsMode::None,
            )?;
            self.vector_verifier_top1(&mut workspace.verify, rows, &target_state.stream)
        })();
        let verification = match verification {
            Ok(verification) => verification,
            Err(error) => {
                target_state.abort_append()?;
                target_cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: &target_state.stream,
                            page_table: target_page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                return Err(error);
            }
        };
        let target_next = verification[0];
        let accepted = usize::from(target_next.id == draft.id);
        let committed_rows = accepted + 1;
        let hc_dim = self.config.hidden * self.config.hc_count;
        if accepted == 0 {
            let restore = (|| -> Result<()> {
                workspace
                    .verify
                    .verify
                    .hybrid
                    .restore_gdn_state_snapshot(0, &target_state.stream)?;
                workspace
                    .verify
                    .verify
                    .exact_ple
                    .as_ref()
                    .expect("exact verifier has a PLE workspace")
                    .restore_frontier_state(&mut target_state.ple_state, &target_state.stream)?;
                target_state.streams_a.copy_range_from_device_on_stream(
                    0,
                    &workspace.verify.verify.streams_a,
                    0,
                    hc_dim,
                    &target_state.stream,
                )
            })();
            if let Err(error) = restore {
                target_state.abort_append()?;
                target_cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: &target_state.stream,
                            page_table: target_page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                return Err(error);
            }
        }
        if let Err(error) = target_cache
            .commit_append(
                reservation,
                committed_rows,
                &mut Sm12xCacheContext {
                    stream: &target_state.stream,
                    page_table: target_page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)
        {
            target_state.abort_append()?;
            return Err(error);
        }
        if accepted == 0 {
            target_state
                .ple_window
                .commit_append_prefix(&workspace.host_tokens[..1])?;
            target_state.ple_state.commit_append()?;
            target_state.position += 1;
        } else {
            target_state.commit_append(rows)?;
        }

        let mut committed = vec![old_frontier];
        if accepted == 1 {
            frontier.previous_streams.copy_range_from_device_on_stream(
                0,
                &workspace.verify.verify.streams_a,
                0,
                hc_dim,
                &target_state.stream,
            )?;
            self.mtp_forward_token(
                mtp_state,
                mtp_workspace,
                mtp_cache,
                draft.id,
                &frontier.previous_streams,
                false,
                &target_state.stream,
            )?;
            committed.push(Qwen38NextToken {
                id: draft.id,
                value: target_next.value,
            });
        }
        frontier.token = verification[accepted].id;
        frontier.logit = verification[accepted].value;
        frontier.previous_streams.copy_range_from_device_on_stream(
            0,
            &workspace.verify.verify.streams_a,
            accepted * hc_dim,
            hc_dim,
            &target_state.stream,
        )?;
        Ok(Qwen38FlashNextSpeculativeOutcome {
            committed,
            accepted_drafts: accepted,
        })
    }

    fn vector_verifier_top1(
        &self,
        workspace: &mut Qwen38FlashNextVectorVerifierProbeWorkspace,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<Vec<Qwen38NextToken>> {
        if workspace.exact_hyper_projections {
            self.final_mixer.mix_exact_two_rows(
                &workspace.verify.streams_a,
                &mut workspace.final_hyper,
                stream,
            )?;
        } else {
            self.final_mixer.mix(
                &workspace.verify.streams_a,
                &mut workspace.final_hyper,
                rows,
                stream,
            )?;
        }
        workspace.final_hidden.copy_prefix_from_device_on_stream(
            workspace.final_hyper.mixed(),
            rows * self.config.hidden,
            stream,
        )?;
        if workspace.exact_output_head {
            if rows != 2 {
                return Err(Error::Shape {
                    label: "Qwen3.8 exact verifier output head",
                    expected: "exactly two rows".to_string(),
                    actual: rows.to_string(),
                });
            }
            self.lm_head.run_bf16_exact_two_rows_top1(
                &workspace.final_hidden,
                &mut workspace.exact_logits,
                &mut workspace.argmax_indices,
                &mut workspace.argmax_values,
                stream,
            )?;
        } else {
            self.lm_head.run_bf16_top1_batch(
                &workspace.final_hidden,
                &mut workspace.top1_scratch_values,
                &workspace.top1_scratch_indices,
                &mut workspace.argmax_indices,
                &mut workspace.argmax_values,
                rows,
                rows,
                stream,
            )?;
        }
        let indices = workspace.argmax_indices.copy_prefix_to_host(rows, stream)?;
        let values = workspace.argmax_values.copy_prefix_to_host(rows, stream)?;
        Ok(indices
            .iter()
            .copied()
            .zip(values.iter().copied())
            .map(|(id, value)| Qwen38NextToken { id, value })
            .collect())
    }

    /// Runs the exact two-row target-verification path for a diagnostic probe.
    ///
    /// The probe commits every supplied input token. Callers must supply target
    /// tokens so that the serial and verification sequences consume identical
    /// token streams.
    pub(crate) fn probe_verification_argmax(
        &mut self,
        workspace: &mut Qwen38FlashNextVectorVerifierProbeWorkspace,
        sequence: &mut Qwen38FlashNextSequence,
        cache: &mut Qwen38FlashNextSequenceCache,
        tokens: &[u32],
    ) -> Result<Vec<Qwen38NextToken>> {
        let rows = tokens.len();
        if rows == 0 || rows > workspace.verify.token_capacity {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next verification probe",
                expected: format!("1..={} input tokens", workspace.verify.token_capacity),
                actual: rows.to_string(),
            });
        }
        let state = &mut sequence.state;
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                rows,
                &mut Sm12xCacheContext {
                    stream: &state.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        if let Err(error) = state.begin_append() {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        if let Err(error) = workspace
            .verify
            .ple_pager
            .begin_read_tokens(&mut state.ple_window, tokens)
        {
            state.abort_append()?;
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        if let Err(error) = self.prefill_tokens_inner(
            state,
            &mut workspace.verify,
            cache,
            &reservation,
            &sequence.page_table,
            tokens,
            Qwen38LogitsMode::None,
        ) {
            state.abort_append()?;
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        let top1 = self.vector_verifier_top1(workspace, rows, &state.stream);
        let top1 = match top1 {
            Ok(top1) => top1,
            Err(error) => {
                state.abort_append()?;
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: &state.stream,
                            page_table: &mut sequence.page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                return Err(error);
            }
        };
        if let Err(error) = cache
            .commit_append(
                reservation,
                rows,
                &mut Sm12xCacheContext {
                    stream: &state.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)
        {
            state.abort_append()?;
            return Err(error);
        }
        state.commit_append(rows)?;
        Ok(top1)
    }

    pub(crate) fn probe_target_streams(
        &self,
        sequence: &Qwen38FlashNextSequence,
    ) -> Result<Vec<f32>> {
        sequence
            .state
            .streams_a
            .copy_to_host(&sequence.state.stream)
            .map(|streams| streams.into_vec())
    }

    pub(crate) fn probe_decode_token_trace(
        &mut self,
        sequence: &mut Qwen38FlashNextSequence,
        cache: &mut Qwen38FlashNextSequenceCache,
        token: u32,
    ) -> Result<(Qwen38NextToken, Vec<Qwen38LayerProbeTrace>)> {
        let mut trace = Vec::new();
        let next = self
            .forward_token_with_trace(
                &mut sequence.state,
                cache,
                sequence.cache_id,
                &mut sequence.page_table,
                token,
                Qwen38LogitsMode::Top1,
                Some(&mut trace),
            )?
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 serial layer probe",
                detail: "top-1 decode returned no token".to_string(),
            })?;
        Ok((next, trace))
    }

    pub(crate) fn probe_verification_argmax_trace(
        &mut self,
        workspace: &mut Qwen38FlashNextVectorVerifierProbeWorkspace,
        sequence: &mut Qwen38FlashNextSequence,
        cache: &mut Qwen38FlashNextSequenceCache,
        tokens: &[u32],
    ) -> Result<(Vec<Qwen38NextToken>, Vec<Qwen38LayerProbeTrace>)> {
        workspace.verify.layer_trace = Some(Vec::new());
        let result = self.probe_verification_argmax(workspace, sequence, cache, tokens);
        let trace = workspace.verify.layer_trace.take().unwrap_or_default();
        result.map(|next| (next, trace))
    }

    /// Allocates private recurrent state and one-token workspaces for a sequence.
    pub fn new_decode_state(&self, max_tokens: usize) -> Result<Qwen38FlashNextDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_position_embeddings {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next context capacity",
                expected: format!("1..={}", self.config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        let linear = self
            .manifest
            .linear_attention
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 Flash Next linear attention",
                detail: "manifest is missing GDN dimensions".to_string(),
            })?;
        let mut attention_workspaces = Vec::with_capacity(self.layers.len());
        let mut attention_states = Vec::with_capacity(self.layers.len());
        let mut rollback_linear_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match &layer.attention {
                Qwen38AttentionWeights::Linear(weights) => {
                    attention_workspaces.push(Qwen38AttentionWorkspace::Linear(
                        Qwen36LinearAttentionWorkspace::new(&self.manifest, linear, weights)?,
                    ));
                    attention_states.push(Qwen38AttentionState::Linear(
                        Qwen36LinearAttentionState::new(linear, weights)?,
                    ));
                    rollback_linear_states
                        .push(Some(Qwen36LinearAttentionState::new(linear, weights)?));
                }
                Qwen38AttentionWeights::Qsa(weights) => {
                    attention_workspaces.push(Qwen38AttentionWorkspace::Qsa(
                        Qwen38QsaWorkspace::new(&self.config, &self.manifest, weights, max_tokens)?,
                    ));
                    attention_states.push(Qwen38AttentionState::Qsa);
                    rollback_linear_states.push(None);
                }
            }
        }
        let hc_dim = self.config.hidden * self.config.hc_count;
        Ok(Qwen38FlashNextDecodeState {
            model_id: self.model_id,
            stream: CudaStream::new_non_blocking()?,
            token_id: DeviceBuffer::zeroed(1)?,
            streams_a: DeviceBuffer::zeroed(hc_dim)?,
            streams_b: DeviceBuffer::zeroed(hc_dim)?,
            hidden: DeviceBuffer::zeroed(self.config.hidden)?,
            zero_hidden: DeviceBuffer::zeroed(self.config.hidden)?,
            attention_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, 1)?,
            mlp_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, 1)?,
            final_hyper: Qwen38HyperConnectionWorkspace::new(&self.config, 1)?,
            attention_workspaces,
            attention_states,
            rollback_linear_states,
            moe: Qwen36MoeWorkspace::new(&self.manifest)?,
            ple_pager: Qwen38PagedPle::open(&self.checkpoint, &self.config, 1)?,
            ple_window: Qwen38PleTokenWindow::new(
                self.config.ngram_size,
                self.config.eos_token_id,
            )?,
            ple_state: Qwen38PleState::new(&self.config)?,
            ple_workspace: Qwen38PleWorkspace::new(&self.config, 1)?,
            lm_head: Qwen36LmHeadWorkspace::new(self.config.vocab, self.config.hidden)?,
            position: 0,
            max_tokens,
        })
    }

    /// Allocates one shared prompt workspace for vectorized scheduler chunks.
    pub(crate) fn new_prefill_workspace(
        &self,
        token_capacity: usize,
    ) -> Result<Qwen38FlashNextPrefillWorkspace> {
        if token_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next prefill capacity",
                expected: "positive token capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let linear_layers = self
            .layers
            .iter()
            .map(|layer| matches!(layer.attention, Qwen38AttentionWeights::Linear(_)))
            .collect::<Vec<_>>();
        let first_linear = self
            .layers
            .iter()
            .find_map(|layer| match &layer.attention {
                Qwen38AttentionWeights::Linear(weights) => Some(weights),
                Qwen38AttentionWeights::Qsa(_) => None,
            })
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 Flash Next prefill",
                detail: "model has no GDN layer".to_string(),
            })?;
        let first_moe = &self
            .layers
            .first()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 Flash Next prefill",
                detail: "model has no transformer layers".to_string(),
            })?
            .moe;
        let first_qsa = self
            .layers
            .iter()
            .find_map(|layer| match &layer.attention {
                Qwen38AttentionWeights::Linear(_) => None,
                Qwen38AttentionWeights::Qsa(weights) => Some(weights),
            })
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 Flash Next prefill",
                detail: "model has no QSA layer".to_string(),
            })?;
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &linear_layers);
        let hybrid =
            Qwen36HybridPrefillWorkspace::new(&model, first_linear, first_moe, token_capacity)?;
        let qsa = first_qsa.new_prefill_workspace(
            &model,
            &self.config,
            token_capacity,
            self.config.max_position_embeddings,
        )?;
        let hc_dim = self.config.hidden * self.config.hc_count;
        Ok(Qwen38FlashNextPrefillWorkspace {
            token_capacity,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            streams_a: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            streams_b: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            hidden: DeviceBuffer::zeroed(token_capacity * self.config.hidden)?,
            qsa,
            qsa_serial_output: DeviceBuffer::zeroed(token_capacity * self.config.hidden)?,
            qsa_serial_row_hidden: DeviceBuffer::zeroed(self.config.hidden)?,
            attention_hyper: Qwen38HyperConnectionWorkspace::new_prefill(
                &self.config,
                token_capacity,
            )?,
            mlp_hyper: Qwen38HyperConnectionWorkspace::new_prefill(&self.config, token_capacity)?,
            ple_pager: Qwen38PagedPle::open(&self.checkpoint, &self.config, token_capacity)?,
            ple: Qwen38PleWorkspace::new(&self.config, token_capacity)?,
            exact_ple: None,
            hybrid,
            exact_gdn: None,
            exact_moe: None,
            serial_gdn_projections: false,
            serial_qsa_rows: false,
            canonical_moe_linears: false,
            exact_hyper_projections: false,
            layer_trace: None,
            linear_layers,
        })
    }

    /// Copies recurrent GDN and PLE state for a retained prompt prefix.
    pub(crate) fn snapshot_sequence(
        &self,
        source: &Qwen38FlashNextDecodeState,
    ) -> Result<Qwen38FlashNextSequenceSnapshot> {
        if source.model_id != self.model_id
            || source.position == 0
            || !source
                .position
                .is_multiple_of(crate::nvfp4::SM12X_KV_PAGE_TOKENS)
        {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next sequence snapshot",
                expected: "matching model and nonzero page-aligned position".to_string(),
                actual: format!(
                    "model={} expected_model={} position={}",
                    source.model_id, self.model_id, source.position
                ),
            });
        }
        let ple_window = source.ple_window.snapshot()?;
        let ple_conv = source.ple_state.snapshot_on_stream(&source.stream)?;
        let mut frontier_streams = DeviceBuffer::zeroed(source.streams_a.len())?;
        frontier_streams.copy_prefix_from_device_on_stream(
            &source.streams_a,
            source.streams_a.len(),
            &source.stream,
        )?;
        let mut linear_states = Vec::with_capacity(source.attention_states.len());
        let mut device_bytes = ple_conv.device_bytes() + frontier_streams.device_bytes();
        for state in &source.attention_states {
            match state {
                Qwen38AttentionState::Linear(linear_source) => {
                    let mut destination = Qwen36LinearAttentionState {
                        conv_state: DeviceBuffer::zeroed(linear_source.conv_state.len())?,
                        recurrent_state: DeviceBuffer::zeroed(linear_source.recurrent_state.len())?,
                    };
                    destination.copy_from_on_stream(linear_source, &source.stream)?;
                    device_bytes = device_bytes
                        .checked_add(destination.device_bytes())
                        .ok_or_else(|| Error::Shape {
                            label: "Qwen3.8 Flash Next snapshot bytes",
                            expected: "byte total without overflow".to_string(),
                            actual: device_bytes.to_string(),
                        })?;
                    linear_states.push(Some(destination));
                }
                Qwen38AttentionState::Qsa => linear_states.push(None),
            }
        }
        source.stream.synchronize()?;
        Ok(Qwen38FlashNextSequenceSnapshot {
            model_id: self.model_id,
            position: source.position,
            linear_states,
            ple_window,
            ple_conv,
            frontier_streams,
            device_bytes,
        })
    }

    /// Restores a retained recurrent snapshot into an empty sequence state.
    pub(crate) fn restore_sequence_snapshot(
        &self,
        snapshot: &Qwen38FlashNextSequenceSnapshot,
        destination: &mut Qwen38FlashNextDecodeState,
    ) -> Result<()> {
        if snapshot.model_id != self.model_id
            || destination.model_id != self.model_id
            || destination.position != 0
            || snapshot.position > destination.max_tokens
            || snapshot.linear_states.len() != destination.attention_states.len()
        {
            return Err(Error::Format {
                label: "Qwen3.8 Flash Next sequence snapshot restore",
                detail: "snapshot and empty destination are incompatible".to_string(),
            });
        }
        for (layer_snapshot, layer_destination) in snapshot
            .linear_states
            .iter()
            .zip(&mut destination.attention_states)
        {
            match (layer_snapshot, layer_destination) {
                (Some(source), Qwen38AttentionState::Linear(linear_destination)) => {
                    linear_destination.copy_from_on_stream(source, &destination.stream)?;
                }
                (None, Qwen38AttentionState::Qsa) => {}
                _ => {
                    return Err(Error::Format {
                        label: "Qwen3.8 Flash Next sequence snapshot restore",
                        detail: "snapshot layer kinds differ from destination".to_string(),
                    });
                }
            }
        }
        destination
            .ple_state
            .restore_from_on_stream(&snapshot.ple_conv, &destination.stream)?;
        destination.streams_a.copy_prefix_from_device_on_stream(
            &snapshot.frontier_streams,
            snapshot.frontier_streams.len(),
            &destination.stream,
        )?;
        destination.ple_window.restore_from(&snapshot.ple_window)?;
        destination.stream.synchronize()?;
        destination.position = snapshot.position;
        Ok(())
    }

    /// Evaluates one token and commits all recurrent state only after success.
    pub fn decode_token(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        cache: &mut Qwen38FlashNextSequenceCache,
        cache_id: SequenceId,
        page_table: &mut Sm12xPageTable,
        token: u32,
    ) -> Result<Qwen38NextToken> {
        self.forward_token(
            state,
            cache,
            cache_id,
            page_table,
            token,
            Qwen38LogitsMode::Top1,
        )?
        .ok_or_else(|| Error::Format {
            label: "Qwen3.8 Flash Next decode",
            detail: "top-1 decode produced no token".to_string(),
        })
    }

    /// Evaluates and transactionally commits one contiguous prompt chunk.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tokens(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        workspace: &mut Qwen38FlashNextPrefillWorkspace,
        cache: &mut Qwen38FlashNextSequenceCache,
        cache_id: SequenceId,
        page_table: &mut Sm12xPageTable,
        tokens: &[u32],
        logits: Qwen38LogitsMode,
    ) -> Result<Option<Qwen38NextToken>> {
        if tokens.is_empty() || tokens.len() > workspace.token_capacity {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next prefill tokens",
                expected: format!("1..={} tokens", workspace.token_capacity),
                actual: tokens.len().to_string(),
            });
        }
        let end = state
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 Flash Next prefill position",
                expected: "position + tokens without overflow".to_string(),
                actual: format!("position={} tokens={}", state.position, tokens.len()),
            })?;
        if end > state.max_tokens {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next prefill position",
                expected: format!("end <= {}", state.max_tokens),
                actual: end.to_string(),
            });
        }
        if let Some(token) = tokens
            .iter()
            .find(|&&token| token as usize >= self.config.vocab)
        {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next prefill token",
                expected: format!("token < {}", self.config.vocab),
                actual: token.to_string(),
            });
        }
        let reservation = cache
            .reserve_append(
                cache_id,
                tokens.len(),
                &mut Sm12xCacheContext {
                    stream: &state.stream,
                    page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        if let Err(error) = state.begin_append() {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        if let Err(error) = workspace
            .ple_pager
            .begin_read_tokens(&mut state.ple_window, tokens)
        {
            state.abort_append()?;
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        let result = self.prefill_tokens_inner(
            state,
            workspace,
            cache,
            &reservation,
            page_table,
            tokens,
            logits,
        );
        match result {
            Ok(next) => {
                if let Err(error) = cache.commit_append(
                    reservation.clone(),
                    tokens.len(),
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table,
                    },
                ) {
                    let rollback = state.abort_append().err();
                    cache
                        .abort_append(
                            reservation,
                            &mut Sm12xCacheContext {
                                stream: &state.stream,
                                page_table,
                            },
                        )
                        .map_err(qwen38_flash_next_cache_error)?;
                    return Err(rollback.unwrap_or_else(|| qwen38_flash_next_cache_error(error)));
                }
                state.commit_append(tokens.len())?;
                Ok(next)
            }
            Err(error) => {
                state.abort_append()?;
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: &state.stream,
                            page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_tokens_inner(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        workspace: &mut Qwen38FlashNextPrefillWorkspace,
        cache: &mut Qwen38FlashNextSequenceCache,
        reservation: &AppendReservation,
        page_table: &Sm12xPageTable,
        tokens: &[u32],
        logits: Qwen38LogitsMode,
    ) -> Result<Option<Qwen38NextToken>> {
        let token_count = tokens.len();
        if let Some(trace) = workspace.layer_trace.as_mut() {
            trace.clear();
        }
        workspace.token_ids.copy_prefix_from_host(tokens)?;
        self.embedding.gather_prefix(
            self.config.vocab,
            self.config.hidden,
            &workspace.token_ids,
            workspace.hidden.output(),
            token_count,
            &state.stream,
        )?;
        qwen38_repeat_streams_f32_into_on_stream(
            &workspace.hidden,
            workspace.streams_a.output(),
            token_count,
            self.config.hidden,
            self.config.hc_count,
            &state.stream,
        )?;

        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &workspace.linear_layers);
        workspace.hybrid.begin_gdn_prefill(token_count)?;
        for (layer, attention_state) in state.attention_states.iter_mut().enumerate() {
            if let Qwen38AttentionState::Linear(linear_state) = attention_state {
                workspace.hybrid.bind_gdn_state(layer, linear_state)?;
            }
        }
        workspace.hybrid.finish_gdn_prefill()?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if layer_index == self.config.ple_layer {
                let ple = if let Some(exact_ple) = workspace.exact_ple.as_mut() {
                    self.ple_weights
                        .run_exact_two_rows(
                            &mut workspace.ple_pager,
                            &workspace.streams_a,
                            &mut state.ple_state,
                            exact_ple,
                            &state.stream,
                        )?
                        .0
                } else {
                    self.ple_weights
                        .run(
                            &mut workspace.ple_pager,
                            &workspace.streams_a,
                            &mut state.ple_state,
                            &mut workspace.ple,
                            token_count,
                            &state.stream,
                        )?
                        .0
                };
                add_f32_into_on_stream(
                    &workspace.streams_a,
                    ple,
                    workspace.streams_b.output(),
                    &state.stream,
                )?;
                std::mem::swap(&mut workspace.streams_a, &mut workspace.streams_b);
                if let Some(trace) = workspace.layer_trace.as_mut() {
                    trace.push(capture_layer_probe_trace(
                        layer_index,
                        Qwen38LayerProbeStage::Ple,
                        token_count,
                        self.config.hidden * self.config.hc_count,
                        &workspace.streams_a,
                        &state.stream,
                    )?);
                }
            }

            if workspace.exact_hyper_projections {
                layer.attention_hyper.mix_exact_two_rows(
                    &workspace.streams_a,
                    &mut workspace.attention_hyper,
                    &state.stream,
                )?;
            } else {
                layer.attention_hyper.mix(
                    &workspace.streams_a,
                    &mut workspace.attention_hyper,
                    token_count,
                    &state.stream,
                )?;
            }
            let attention_output = match (
                &layer.attention,
                &mut state.attention_workspaces[layer_index],
                &mut state.attention_states[layer_index],
            ) {
                (
                    Qwen38AttentionWeights::Linear(weights),
                    Qwen38AttentionWorkspace::Linear(_),
                    Qwen38AttentionState::Linear(linear_state),
                ) => {
                    if let Some(exact_gdn) = workspace.exact_gdn.as_mut() {
                        exact_gdn.run(
                            weights,
                            linear_state,
                            workspace.attention_hyper.mixed(),
                            token_count,
                            self.config.hidden,
                            self.config.rms_eps(),
                            Some((&mut workspace.hybrid, layer_index)),
                            &state.stream,
                        )?
                    } else {
                        workspace.hybrid.run_gdn(
                            &model,
                            weights,
                            workspace.attention_hyper.mixed(),
                            layer_index,
                            token_count,
                            workspace.serial_gdn_projections,
                            &state.stream,
                        )?
                    }
                }
                (
                    Qwen38AttentionWeights::Qsa(weights),
                    Qwen38AttentionWorkspace::Qsa(qsa_workspace),
                    Qwen38AttentionState::Qsa,
                ) => {
                    let mixed = workspace.attention_hyper.mixed();
                    if workspace.serial_qsa_rows {
                        cache
                            .with_append_pages(reservation, |backend, pages| {
                                for page in pages.iter() {
                                    let segment = page.segment();
                                    for offset in 0..segment.rows() {
                                        let row = segment.input_offset() + offset;
                                        weights.run_prefill_row(
                                            qsa_workspace,
                                            backend,
                                            page_table.device(),
                                            page.page(),
                                            segment.page_offset() + offset,
                                            &self.config,
                                            &self.manifest,
                                            mixed,
                                            &mut workspace.qsa_serial_row_hidden,
                                            &mut workspace.qsa_serial_output,
                                            row,
                                            layer_index,
                                            state.position + row,
                                            &state.stream,
                                        )?;
                                    }
                                }
                                Ok(())
                            })
                            .map_err(qwen38_flash_next_cache_error)?;
                        &workspace.qsa_serial_output
                    } else {
                        weights.prepare_prefill(
                            &model,
                            &mut workspace.qsa,
                            &self.config,
                            mixed,
                            token_count,
                            state.position,
                            &state.stream,
                        )?;
                        cache
                            .with_append_pages(reservation, |backend, pages| {
                                for page in pages.iter() {
                                    let segment = page.segment();
                                    for offset in 0..segment.rows() {
                                        let row = segment.input_offset() + offset;
                                        weights.run_prepared_prefill_row(
                                            &model,
                                            &mut workspace.qsa,
                                            qsa_workspace,
                                            backend,
                                            page_table.device(),
                                            page.page(),
                                            segment.page_offset() + offset,
                                            &self.config,
                                            row,
                                            layer_index,
                                            state.position + row,
                                            &state.stream,
                                        )?;
                                    }
                                }
                                Ok(())
                            })
                            .map_err(qwen38_flash_next_cache_error)?;
                        weights.finish_prefill(
                            &model,
                            &mut workspace.qsa,
                            token_count,
                            &state.stream,
                        )?
                    }
                }
                _ => {
                    return Err(Error::Format {
                        label: "Qwen3.8 Flash Next prefill attention",
                        detail: format!("layer {layer_index} state topology mismatch"),
                    });
                }
            };
            if workspace.exact_hyper_projections {
                layer.attention_hyper.combine_exact_two_rows(
                    &workspace.streams_a,
                    attention_output,
                    &mut workspace.attention_hyper,
                    &mut workspace.streams_b,
                    &state.stream,
                )?;
            } else {
                layer.attention_hyper.combine(
                    &workspace.streams_a,
                    attention_output,
                    &mut workspace.attention_hyper,
                    &mut workspace.streams_b,
                    token_count,
                    &state.stream,
                )?;
            }
            std::mem::swap(&mut workspace.streams_a, &mut workspace.streams_b);
            if let Some(trace) = workspace.layer_trace.as_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::Attention,
                    token_count,
                    self.config.hidden * self.config.hc_count,
                    &workspace.streams_a,
                    &state.stream,
                )?);
            }

            if workspace.exact_hyper_projections {
                layer.mlp_hyper.mix_exact_two_rows(
                    &workspace.streams_a,
                    &mut workspace.mlp_hyper,
                    &state.stream,
                )?;
            } else {
                layer.mlp_hyper.mix(
                    &workspace.streams_a,
                    &mut workspace.mlp_hyper,
                    token_count,
                    &state.stream,
                )?;
            }
            if let Some(trace) = workspace.layer_trace.as_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::MlpMix,
                    token_count,
                    self.config.hidden,
                    workspace.mlp_hyper.mixed(),
                    &state.stream,
                )?);
            }
            let ffn = if let Some(exact_moe) = workspace.exact_moe.as_mut() {
                let capture_oracle = workspace.layer_trace.is_some();
                exact_moe.run(
                    &self.lt,
                    &self.manifest,
                    &layer.moe,
                    workspace.mlp_hyper.mixed(),
                    token_count,
                    capture_oracle,
                    &state.stream,
                )?
            } else if workspace.canonical_moe_linears {
                workspace.hybrid.run_moe_with_canonical_linears(
                    &model,
                    &layer.moe,
                    workspace.mlp_hyper.mixed(),
                    token_count,
                    &state.stream,
                )?
            } else {
                workspace.hybrid.run_moe(
                    &model,
                    &layer.moe,
                    workspace.mlp_hyper.mixed(),
                    token_count,
                    &state.stream,
                )?
            };
            if let Some(trace) = workspace.layer_trace.as_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::MlpFfn,
                    token_count,
                    self.config.hidden,
                    ffn,
                    &state.stream,
                )?);
            }
            if workspace.exact_hyper_projections {
                layer.mlp_hyper.combine_exact_two_rows(
                    &workspace.streams_a,
                    ffn,
                    &mut workspace.mlp_hyper,
                    &mut workspace.streams_b,
                    &state.stream,
                )?;
            } else {
                layer.mlp_hyper.combine(
                    &workspace.streams_a,
                    ffn,
                    &mut workspace.mlp_hyper,
                    &mut workspace.streams_b,
                    token_count,
                    &state.stream,
                )?;
            }
            std::mem::swap(&mut workspace.streams_a, &mut workspace.streams_b);
            if let (Some(exact_moe), Some(trace)) =
                (workspace.exact_moe.as_mut(), workspace.layer_trace.as_mut())
            {
                let snapshots = exact_moe.probe_snapshots(&layer.moe, &state.stream)?;
                if let Some(ffn_trace) = trace.iter_mut().rev().find(|entry| {
                    entry.layer == layer_index && entry.stage == Qwen38LayerProbeStage::MlpFfn
                }) {
                    ffn_trace.moe = Some(snapshots);
                }
            }
            if let Some(trace) = workspace.layer_trace.as_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::Mlp,
                    token_count,
                    self.config.hidden * self.config.hc_count,
                    &workspace.streams_a,
                    &state.stream,
                )?);
            }
        }

        let hc_dim = self.config.hidden * self.config.hc_count;
        state.streams_a.copy_range_from_device_on_stream(
            0,
            &workspace.streams_a,
            (token_count - 1) * hc_dim,
            hc_dim,
            &state.stream,
        )?;
        if logits == Qwen38LogitsMode::None {
            state.stream.synchronize()?;
            return Ok(None);
        }
        self.final_mixer
            .mix(&state.streams_a, &mut state.final_hyper, 1, &state.stream)?;
        state.hidden.copy_prefix_from_device_on_stream(
            state.final_hyper.mixed(),
            self.config.hidden,
            &state.stream,
        )?;
        match logits {
            Qwen38LogitsMode::None => unreachable!("no-logits mode returned before final mix"),
            Qwen38LogitsMode::Top1 => {
                self.lm_head.run_top1(
                    &self.lt,
                    &state.hidden,
                    &mut state.lm_head,
                    &state.stream,
                )?;
                let (id, value) = state.lm_head.read_top1(&state.stream)?;
                Ok(Some(Qwen38NextToken { id, value }))
            }
            Qwen38LogitsMode::Full => {
                self.lm_head.run_logits(
                    &self.lt,
                    &state.hidden,
                    &mut state.lm_head,
                    &state.stream,
                )?;
                state.stream.synchronize()?;
                Ok(None)
            }
        }
    }

    /// Evaluates and commits one token with the requested vocabulary-head work.
    pub(crate) fn forward_token(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        cache: &mut Qwen38FlashNextSequenceCache,
        cache_id: SequenceId,
        page_table: &mut Sm12xPageTable,
        token: u32,
        logits: Qwen38LogitsMode,
    ) -> Result<Option<Qwen38NextToken>> {
        self.forward_token_with_trace(state, cache, cache_id, page_table, token, logits, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_token_with_trace(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        cache: &mut Qwen38FlashNextSequenceCache,
        cache_id: SequenceId,
        page_table: &mut Sm12xPageTable,
        token: u32,
        logits: Qwen38LogitsMode,
        trace: Option<&mut Vec<Qwen38LayerProbeTrace>>,
    ) -> Result<Option<Qwen38NextToken>> {
        if state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "Qwen3.8 Flash Next decode position",
                expected: format!("position < {}", state.max_tokens),
                actual: state.position.to_string(),
            });
        }
        let reservation = cache
            .reserve_append(
                cache_id,
                1,
                &mut Sm12xCacheContext {
                    stream: &state.stream,
                    page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)?;
        if let Err(error) = state.begin_append() {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        if let Err(error) = state
            .ple_pager
            .begin_read_tokens(&mut state.ple_window, &[token])
        {
            state.abort_append()?;
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table,
                    },
                )
                .map_err(qwen38_flash_next_cache_error)?;
            return Err(error);
        }
        let result =
            self.decode_token_inner(state, cache, &reservation, page_table, token, logits, trace);
        match result {
            Ok(next) => {
                if let Err(error) = cache.commit_append(
                    reservation.clone(),
                    1,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table,
                    },
                ) {
                    let rollback = state.abort_append().err();
                    cache
                        .abort_append(
                            reservation,
                            &mut Sm12xCacheContext {
                                stream: &state.stream,
                                page_table,
                            },
                        )
                        .map_err(qwen38_flash_next_cache_error)?;
                    return Err(rollback.unwrap_or_else(|| qwen38_flash_next_cache_error(error)));
                }
                state.commit_append(1)?;
                Ok(next)
            }
            Err(error) => {
                state.abort_append()?;
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: &state.stream,
                            page_table,
                        },
                    )
                    .map_err(qwen38_flash_next_cache_error)?;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_token_inner(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        cache: &mut Qwen38FlashNextSequenceCache,
        reservation: &AppendReservation,
        page_table: &Sm12xPageTable,
        token: u32,
        logits: Qwen38LogitsMode,
        mut trace: Option<&mut Vec<Qwen38LayerProbeTrace>>,
    ) -> Result<Option<Qwen38NextToken>> {
        state.token_id.copy_from_host(&[token])?;
        self.embedding.gather_prefix(
            self.config.vocab,
            self.config.hidden,
            &state.token_id,
            state.hidden.output(),
            1,
            &state.stream,
        )?;
        qwen38_repeat_streams_f32_into_on_stream(
            &state.hidden,
            state.streams_a.output(),
            1,
            self.config.hidden,
            self.config.hc_count,
            &state.stream,
        )?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            if layer_index == self.config.ple_layer {
                let (ple, _) = self.ple_weights.run(
                    &mut state.ple_pager,
                    &state.streams_a,
                    &mut state.ple_state,
                    &mut state.ple_workspace,
                    1,
                    &state.stream,
                )?;
                add_f32_into_on_stream(
                    &state.streams_a,
                    ple,
                    state.streams_b.output(),
                    &state.stream,
                )?;
                std::mem::swap(&mut state.streams_a, &mut state.streams_b);
                if let Some(trace) = trace.as_deref_mut() {
                    trace.push(capture_layer_probe_trace(
                        layer_index,
                        Qwen38LayerProbeStage::Ple,
                        1,
                        self.config.hidden * self.config.hc_count,
                        &state.streams_a,
                        &state.stream,
                    )?);
                }
            }

            layer.attention_hyper.mix(
                &state.streams_a,
                &mut state.attention_hyper,
                1,
                &state.stream,
            )?;
            let attention_output = match (
                &layer.attention,
                &mut state.attention_workspaces[layer_index],
                &mut state.attention_states[layer_index],
            ) {
                (
                    Qwen38AttentionWeights::Linear(weights),
                    Qwen38AttentionWorkspace::Linear(workspace),
                    Qwen38AttentionState::Linear(sequence),
                ) => {
                    weights
                        .run_one_token_sigmoid_output_gate(
                            workspace,
                            sequence,
                            state.attention_hyper.mixed(),
                            self.config.rms_eps(),
                            &state.stream,
                        )?
                        .output
                }
                (
                    Qwen38AttentionWeights::Qsa(weights),
                    Qwen38AttentionWorkspace::Qsa(workspace),
                    Qwen38AttentionState::Qsa,
                ) => cache
                    .with_append_pages(reservation, |backend, pages| {
                        let page = pages.iter().next().ok_or_else(|| Error::Format {
                            label: "Qwen3.8 QSA append",
                            detail: "one-token reservation contains no physical page".to_string(),
                        })?;
                        if pages.iter().count() != 1 || page.segment().rows() != 1 {
                            return Err(Error::Format {
                                label: "Qwen3.8 QSA append",
                                detail: "decode reservation does not cover exactly one row"
                                    .to_string(),
                            });
                        }
                        weights.run_one_token(
                            workspace,
                            backend,
                            page_table.device(),
                            page.page(),
                            page.segment().page_offset(),
                            &self.config,
                            &self.manifest,
                            state.attention_hyper.mixed(),
                            layer_index,
                            state.position,
                            &state.stream,
                        )
                    })
                    .map_err(qwen38_flash_next_cache_error)?,
                _ => {
                    return Err(Error::Format {
                        label: "Qwen3.8 Flash Next attention",
                        detail: format!("layer {layer_index} state topology mismatch"),
                    });
                }
            };
            layer.attention_hyper.combine(
                &state.streams_a,
                attention_output,
                &mut state.attention_hyper,
                &mut state.streams_b,
                1,
                &state.stream,
            )?;
            std::mem::swap(&mut state.streams_a, &mut state.streams_b);
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::Attention,
                    1,
                    self.config.hidden * self.config.hc_count,
                    &state.streams_a,
                    &state.stream,
                )?);
            }

            layer
                .mlp_hyper
                .mix(&state.streams_a, &mut state.mlp_hyper, 1, &state.stream)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::MlpMix,
                    1,
                    self.config.hidden,
                    state.mlp_hyper.mixed(),
                    &state.stream,
                )?);
            }
            let ffn = layer.moe.run_one_token(
                &self.lt,
                &mut state.moe,
                &self.manifest,
                state.mlp_hyper.mixed(),
                &state.zero_hidden,
                &state.stream,
                None,
                None,
            )?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::MlpFfn,
                    1,
                    self.config.hidden,
                    ffn.ffn_out,
                    &state.stream,
                )?);
            }
            layer.mlp_hyper.combine(
                &state.streams_a,
                ffn.ffn_out,
                &mut state.mlp_hyper,
                &mut state.streams_b,
                1,
                &state.stream,
            )?;
            std::mem::swap(&mut state.streams_a, &mut state.streams_b);
            if trace.is_some() {
                let snapshot = layer
                    .moe
                    .probe_repeat_workspace_gate_up(&mut state.moe, &state.stream)?;
                if let Some(ffn_trace) = trace.as_deref_mut().and_then(|trace| {
                    trace.iter_mut().rev().find(|entry| {
                        entry.layer == layer_index && entry.stage == Qwen38LayerProbeStage::MlpFfn
                    })
                }) {
                    ffn_trace.moe = Some(vec![snapshot]);
                }
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(capture_layer_probe_trace(
                    layer_index,
                    Qwen38LayerProbeStage::Mlp,
                    1,
                    self.config.hidden * self.config.hc_count,
                    &state.streams_a,
                    &state.stream,
                )?);
            }
        }

        if logits == Qwen38LogitsMode::None {
            state.stream.synchronize()?;
            return Ok(None);
        }
        self.final_mixer
            .mix(&state.streams_a, &mut state.final_hyper, 1, &state.stream)?;
        state.hidden.copy_prefix_from_device_on_stream(
            state.final_hyper.mixed(),
            self.config.hidden,
            &state.stream,
        )?;
        match logits {
            Qwen38LogitsMode::None => unreachable!("no-logits mode returned before final mix"),
            Qwen38LogitsMode::Top1 => {
                self.lm_head.run_top1(
                    &self.lt,
                    &state.hidden,
                    &mut state.lm_head,
                    &state.stream,
                )?;
                let (id, value) = state.lm_head.read_top1(&state.stream)?;
                Ok(Some(Qwen38NextToken { id, value }))
            }
            Qwen38LogitsMode::Full => {
                self.lm_head.run_logits(
                    &self.lt,
                    &state.hidden,
                    &mut state.lm_head,
                    &state.stream,
                )?;
                state.stream.synchronize()?;
                Ok(None)
            }
        }
    }

    pub(crate) fn read_top1(&self, state: &Qwen38FlashNextDecodeState) -> Result<Qwen38NextToken> {
        let (id, value) = state.lm_head.read_top1(&state.stream)?;
        Ok(Qwen38NextToken { id, value })
    }

    pub(crate) fn sample_logits_gpu(
        &self,
        state: &Qwen38FlashNextDecodeState,
        sampler: &mut GpuTokenSampler,
        row: &mut GpuSamplingRow<'_>,
    ) -> Result<GpuSampledToken> {
        sampler
            .sample(
                state.lm_head.logits(),
                std::slice::from_mut(row),
                self.config.vocab,
                &state.stream,
            )?
            .pop()
            .ok_or_else(|| Error::Format {
                label: "Qwen3.8 Flash Next GPU sampling",
                detail: "sampler returned no token".to_string(),
            })
    }

    pub(crate) fn logits_to_host(&self, state: &Qwen38FlashNextDecodeState) -> Result<Vec<f32>> {
        state.lm_head.read_logits(&state.stream)
    }

    /// Parsed released-model configuration.
    pub fn config(&self) -> &Qwen38FlashNextConfig {
        &self.config
    }

    pub(crate) fn manifest(&self) -> &QwenModelManifest {
        &self.manifest
    }

    pub(crate) fn copy_prefill_target_streams(
        &self,
        workspace: &Qwen38FlashNextPrefillWorkspace,
        row: usize,
        destination: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.config.hidden * self.config.hc_count;
        destination.copy_range_from_device_on_stream(
            0,
            &workspace.streams_a,
            row * width,
            width,
            stream,
        )
    }

    pub(crate) fn copy_decode_target_streams(
        &self,
        state: &Qwen38FlashNextDecodeState,
        destination: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        destination.copy_prefix_from_device_on_stream(
            &state.streams_a,
            state.streams_a.len(),
            &state.stream,
        )
    }

    /// Checkpoint retained by the loaded runtime.
    pub fn checkpoint(&self) -> &ModelOptCheckpoint {
        &self.checkpoint
    }

    /// Derived-artifact root used by converted expert weights.
    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }
}

impl Qwen38FlashNextDecodeState {
    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Returns the exact device bytes owned outside shared QSA pages.
    pub fn device_bytes(&self) -> usize {
        self.token_id.device_bytes()
            + self.streams_a.device_bytes()
            + self.streams_b.device_bytes()
            + self.hidden.device_bytes()
            + self.zero_hidden.device_bytes()
            + self.attention_hyper.device_bytes()
            + self.mlp_hyper.device_bytes()
            + self.final_hyper.device_bytes()
            + self
                .attention_workspaces
                .iter()
                .map(Qwen38AttentionWorkspace::device_bytes)
                .sum::<usize>()
            + self
                .attention_states
                .iter()
                .map(|state| match state {
                    Qwen38AttentionState::Linear(state) => state.device_bytes(),
                    Qwen38AttentionState::Qsa => 0,
                })
                .sum::<usize>()
            + self
                .rollback_linear_states
                .iter()
                .flatten()
                .map(Qwen36LinearAttentionState::device_bytes)
                .sum::<usize>()
            + self.moe.device_bytes()
            + self.ple_state.device_bytes()
            + self.ple_workspace.device_bytes()
            + self.lm_head.device_bytes()
    }

    fn begin_append(&mut self) -> Result<()> {
        self.ple_window.begin_append()?;
        if let Err(error) = self.ple_state.begin_append(&self.stream) {
            self.ple_window.abort_append()?;
            return Err(error);
        }
        for (active, rollback) in self
            .attention_states
            .iter()
            .zip(&mut self.rollback_linear_states)
        {
            if let (Qwen38AttentionState::Linear(active), Some(rollback)) = (active, rollback) {
                rollback.copy_from_on_stream(active, &self.stream)?;
            }
        }
        Ok(())
    }

    fn commit_append(&mut self, tokens: usize) -> Result<()> {
        self.ple_window.commit_append()?;
        self.ple_state.commit_append()?;
        self.position += tokens;
        Ok(())
    }

    fn abort_append(&mut self) -> Result<()> {
        for (active, rollback) in self
            .attention_states
            .iter_mut()
            .zip(&self.rollback_linear_states)
        {
            if let (Qwen38AttentionState::Linear(active), Some(rollback)) = (active, rollback) {
                active.copy_from_on_stream(rollback, &self.stream)?;
            }
        }
        self.ple_state.abort_append(&self.stream)?;
        self.ple_window.abort_append()
    }

    /// Number of committed tokens.
    pub fn position(&self) -> usize {
        self.position
    }
}

impl Qwen38FlashNextMtpSequenceState {
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.page_table.managed_bytes()
    }

    pub(crate) fn finish(
        self,
        cache: &mut Qwen38FlashNextMtpSequenceCache,
        stream: &CudaStream,
    ) -> Result<()> {
        let mut page_table = self.page_table;
        cache
            .finish(
                self.cache_id,
                &mut Sm12xCacheContext {
                    stream,
                    page_table: &mut page_table,
                },
            )
            .map_err(qwen38_flash_next_cache_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sync(stream: &CudaStream, stage: &str) -> Result<()> {
        stream.synchronize().map_err(|error| Error::Format {
            label: "Qwen3.8 released layer forward",
            detail: format!("{stage}: {error}"),
        })
    }

    fn run_released_layer(model_dir: &Path, layer_index: usize) -> Result<()> {
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let manifest = config.qwen_manifest();
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-layer-{}", std::process::id()));
        let prefix = format!("model.language_model.layers.{layer_index}");
        let attention_hyper = Qwen38HyperConnectionWeights::load(
            &checkpoint,
            &format!("{prefix}.attn_hyper_connection"),
            &config,
            true,
        )?;
        let mlp_hyper = Qwen38HyperConnectionWeights::load(
            &checkpoint,
            &format!("{prefix}.mlp_hyper_connection"),
            &config,
            true,
        )?;
        let attention = match manifest.layer_kinds[layer_index] {
            QwenLayerKind::LinearAttention => Qwen38AttentionWeights::Linear(
                load_hybrid_linear_attention(&checkpoint, &manifest, &artifact_dir, layer_index)?,
            ),
            QwenLayerKind::FullAttention => {
                let full =
                    load_hybrid_full_attention(&checkpoint, &manifest, &artifact_dir, layer_index)?;
                Qwen38AttentionWeights::Qsa(Qwen38QsaWeights::load(
                    &checkpoint,
                    &config,
                    layer_index,
                    full,
                )?)
            }
        };
        let moe = Qwen36MoeWeights::load_checkpoint_layout(
            &checkpoint,
            &manifest,
            &artifact_dir,
            layer_index,
        )?;
        let stream = CudaStream::new_non_blocking()?;
        let input = DeviceBuffer::from_host(
            &(0..config.hidden)
                .map(|index| (index as f32 % 31.0 - 15.0) / 32.0)
                .collect::<Vec<_>>(),
        )?;
        let hc_dim = config.hidden * config.hc_count;
        let mut streams_a = DeviceBuffer::zeroed(hc_dim)?;
        let mut streams_b = DeviceBuffer::zeroed(hc_dim)?;
        qwen38_repeat_streams_f32_into_on_stream(
            &input,
            streams_a.output(),
            1,
            config.hidden,
            config.hc_count,
            &stream,
        )?;
        sync(&stream, "repeat streams")?;

        if layer_index == config.ple_layer {
            let mut pager = Qwen38PagedPle::open(&checkpoint, &config, 1)?;
            let weights = Qwen38PleWeights::load(&checkpoint, &config)?;
            let mut state = Qwen38PleState::new(&config)?;
            let mut workspace = Qwen38PleWorkspace::new(&config, 1)?;
            let mut window = Qwen38PleTokenWindow::new(config.ngram_size, config.eos_token_id)?;
            window.begin_append()?;
            state.begin_append(&stream)?;
            pager.begin_read_tokens(&mut window, &[config.eos_token_id])?;
            let (ple, _) = weights.run(
                &mut pager,
                &streams_a,
                &mut state,
                &mut workspace,
                1,
                &stream,
            )?;
            add_f32_into_on_stream(&streams_a, ple, streams_b.output(), &stream)?;
            sync(&stream, "PLE")?;
            std::mem::swap(&mut streams_a, &mut streams_b);
        }

        let mut attention_hc_workspace = Qwen38HyperConnectionWorkspace::new(&config, 1)?;
        attention_hyper.mix(&streams_a, &mut attention_hc_workspace, 1, &stream)?;
        sync(&stream, "attention hyperconnection mix")?;
        let attention_output = match attention {
            Qwen38AttentionWeights::Linear(weights) => {
                let linear = manifest.linear_attention.expect("linear config");
                let mut workspace =
                    Qwen36LinearAttentionWorkspace::new(&manifest, linear, &weights)?;
                let mut state = Qwen36LinearAttentionState::new(linear, &weights)?;
                let output = weights
                    .run_one_token_sigmoid_output_gate(
                        &mut workspace,
                        &mut state,
                        attention_hc_workspace.mixed(),
                        config.rms_eps(),
                        &stream,
                    )?
                    .output;
                sync(&stream, "linear attention")?;
                output.copy_to_host(&stream)?.into_vec()
            }
            Qwen38AttentionWeights::Qsa(weights) => {
                let capacity = crate::nvfp4::SM12X_KV_PAGE_TOKENS;
                let mut workspace =
                    Qwen38QsaWorkspace::new(&config, &manifest, &weights, capacity)?;
                let mut backend = crate::qwen38_flash_next::Qwen38FlashNextPageBackend::new(
                    manifest
                        .layer_kinds
                        .iter()
                        .map(|kind| *kind == QwenLayerKind::FullAttention),
                    1,
                    manifest.kv_heads,
                    manifest.head_dim,
                    config.indexer_head_dim,
                )?;
                let page_table = DeviceBuffer::from_host(&[0u32])?;
                let page = crate::sm12x_cache::Sm12xPage::from_slot(0);
                let output = weights.run_one_token(
                    &mut workspace,
                    &mut backend,
                    &page_table,
                    &page,
                    0,
                    &config,
                    &manifest,
                    attention_hc_workspace.mixed(),
                    layer_index,
                    0,
                    &stream,
                )?;
                sync(&stream, "native QSA")?;
                output.copy_to_host(&stream)?.into_vec()
            }
        };
        let attention_output = DeviceBuffer::from_host(&attention_output)?;
        attention_hyper.combine(
            &streams_a,
            &attention_output,
            &mut attention_hc_workspace,
            &mut streams_b,
            1,
            &stream,
        )?;
        sync(&stream, "attention hyperconnection combine")?;
        std::mem::swap(&mut streams_a, &mut streams_b);

        let mut mlp_hc_workspace = Qwen38HyperConnectionWorkspace::new(&config, 1)?;
        mlp_hyper.mix(&streams_a, &mut mlp_hc_workspace, 1, &stream)?;
        sync(&stream, "MLP hyperconnection mix")?;
        let mut moe_workspace = Qwen36MoeWorkspace::new(&manifest)?;
        let zero = DeviceBuffer::zeroed(config.hidden)?;
        let lt = CublasLt::new()?;
        let ffn = moe.run_one_token(
            &lt,
            &mut moe_workspace,
            &manifest,
            mlp_hc_workspace.mixed(),
            &zero,
            &stream,
            None,
            None,
        )?;
        sync(&stream, "MoE")?;
        let ffn = ffn.ffn_out.copy_to_host(&stream)?.into_vec();
        let grouped_routed = moe_workspace.moe_out.copy_to_host(&stream)?.into_vec();
        let route_indices = moe_workspace
            .route
            .indices
            .copy_to_host(&stream)?
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        let route_weights = moe_workspace
            .route
            .weights
            .copy_to_host(&stream)?
            .into_vec();
        moe.run_w4a16_moe_slots_only(
            &mut moe_workspace,
            mlp_hc_workspace.mixed(),
            &route_indices,
            &route_weights,
            &stream,
        )?;
        sync(&stream, "MoE W4A16 oracle")?;
        let oracle_routed = moe_workspace.moe_out.copy_to_host(&stream)?;
        let max_routed_error = grouped_routed
            .iter()
            .zip(oracle_routed.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        if max_routed_error > 2.0 {
            return Err(Error::Format {
                label: "Qwen3.8 released layer forward",
                detail: format!(
                    "layer {layer_index} grouped routed MoE differs from W4A16 oracle by {max_routed_error}"
                ),
            });
        }
        let ffn = DeviceBuffer::from_host(&ffn)?;
        mlp_hyper.combine(
            &streams_a,
            &ffn,
            &mut mlp_hc_workspace,
            &mut streams_b,
            1,
            &stream,
        )?;
        sync(&stream, "MLP hyperconnection combine")?;
        let output = streams_b.copy_to_host(&stream)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::Format {
                label: "Qwen3.8 released layer forward",
                detail: format!("layer {layer_index} produced a non-finite value"),
            });
        }
        Ok(())
    }

    #[test]
    fn released_linear_and_full_layers_run() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        run_released_layer(Path::new(&model_dir), 0).expect("released linear layer");
        run_released_layer(Path::new(&model_dir), 3).expect("released full layer");
    }

    #[test]
    fn released_layer_batch_primitives_match_serial_tokens() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        let config = Qwen38FlashNextConfig::load(&model_dir).expect("config");
        let manifest = config.qwen_manifest();
        let checkpoint = ModelOptCheckpoint::open(&model_dir).expect("checkpoint");
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-batch-{}", std::process::id()));
        let linear = load_hybrid_linear_attention(&checkpoint, &manifest, &artifact_dir, 0)
            .expect("linear weights");
        let moe =
            Qwen36MoeWeights::load_checkpoint_layout(&checkpoint, &manifest, &artifact_dir, 0)
                .expect("MoE weights");
        let lt = CublasLt::new().expect("cuBLASLt");
        let stream = CudaStream::new_non_blocking().expect("stream");
        const TOKENS: usize = 4;
        let input_host = (0..TOKENS * config.hidden)
            .map(|index| (index as f32 % 31.0 - 15.0) / 32.0)
            .collect::<Vec<_>>();
        let batch_input = DeviceBuffer::from_host(&input_host).expect("batch input");
        let linear_config = manifest.linear_attention.expect("linear config");

        let mut serial_linear_workspace =
            Qwen36LinearAttentionWorkspace::new(&manifest, linear_config, &linear)
                .expect("serial linear workspace");
        let mut serial_linear_state =
            Qwen36LinearAttentionState::new(linear_config, &linear).expect("serial linear state");
        let mut serial_linear = Vec::with_capacity(TOKENS * config.hidden);
        for row in 0..TOKENS {
            let input = DeviceBuffer::from_host(
                &input_host[row * config.hidden..(row + 1) * config.hidden],
            )
            .expect("serial input");
            serial_linear.extend_from_slice(
                &linear
                    .run_one_token_sigmoid_output_gate(
                        &mut serial_linear_workspace,
                        &mut serial_linear_state,
                        &input,
                        config.rms_eps(),
                        &stream,
                    )
                    .expect("serial linear")
                    .output
                    .copy_to_host(&stream)
                    .expect("serial linear readback"),
            );
        }

        let linear_layers = manifest
            .layer_kinds
            .iter()
            .map(|kind| *kind == QwenLayerKind::LinearAttention)
            .collect::<Vec<_>>();
        let model = Qwen36BatchModelView::new(&lt, &manifest, &linear_layers);
        let mut batch = Qwen36HybridPrefillWorkspace::new(&model, &linear, &moe, TOKENS)
            .expect("batch workspace");
        let mut batch_linear_state =
            Qwen36LinearAttentionState::new(linear_config, &linear).expect("batch linear state");
        batch.begin_gdn_prefill(TOKENS).expect("begin GDN");
        batch
            .bind_gdn_state(0, &mut batch_linear_state)
            .expect("bind GDN");
        batch.finish_gdn_prefill().expect("finish GDN setup");
        let batch_linear = batch
            .run_gdn(&model, &linear, &batch_input, 0, TOKENS, false, &stream)
            .expect("batch linear")
            .copy_to_host(&stream)
            .expect("batch linear readback");
        let linear_error = serial_linear
            .iter()
            .zip(batch_linear.iter())
            .map(|(serial, batch)| (serial - batch).abs())
            .fold(0.0f32, f32::max);
        let (linear_dot, linear_serial_norm, linear_batch_norm, linear_squared_error) =
            serial_linear.iter().zip(batch_linear.iter()).fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |sum, (&serial, &batch)| {
                    let serial = serial as f64;
                    let batch = batch as f64;
                    (
                        sum.0 + serial * batch,
                        sum.1 + serial * serial,
                        sum.2 + batch * batch,
                        sum.3 + (serial - batch) * (serial - batch),
                    )
                },
            );
        let linear_cosine = linear_dot / (linear_serial_norm * linear_batch_norm).sqrt();
        let linear_relative_rmse = (linear_squared_error / linear_serial_norm).sqrt();
        assert!(
            linear_cosine >= 0.98 && linear_relative_rmse <= 0.2,
            "batch GDN maximum_error={linear_error} cosine={linear_cosine} relative_rmse={linear_relative_rmse}"
        );

        let zero = DeviceBuffer::zeroed(config.hidden).expect("zero residual");
        let mut serial_moe_workspace = Qwen36MoeWorkspace::new(&manifest).expect("serial MoE");
        let mut serial_moe = Vec::with_capacity(TOKENS * config.hidden);
        for row in 0..TOKENS {
            let input = DeviceBuffer::from_host(
                &input_host[row * config.hidden..(row + 1) * config.hidden],
            )
            .expect("serial input");
            serial_moe.extend_from_slice(
                &moe.run_one_token(
                    &lt,
                    &mut serial_moe_workspace,
                    &manifest,
                    &input,
                    &zero,
                    &stream,
                    None,
                    None,
                )
                .expect("serial MoE")
                .ffn_out
                .copy_to_host(&stream)
                .expect("serial MoE readback"),
            );
        }
        let mut exact_pair = Qwen36ExactMoePairWorkspace::new(&manifest).expect("exact MoE pair");
        let exact_pair = moe
            .run_exact_pair(&mut exact_pair, &manifest, &batch_input, &stream)
            .expect("exact pair MoE")
            .copy_to_host(&stream)
            .expect("exact pair MoE readback");
        assert_eq!(
            exact_pair.as_slice(),
            &serial_moe[..2 * config.hidden],
            "two-row MoE must match canonical rows bitwise",
        );
        let batch_moe = batch
            .run_moe(&model, &moe, &batch_input, TOKENS, &stream)
            .expect("batch MoE")
            .copy_to_host(&stream)
            .expect("batch MoE readback");
        let moe_error = serial_moe
            .iter()
            .zip(batch_moe.iter())
            .map(|(serial, batch)| (serial - batch).abs())
            .fold(0.0f32, f32::max);
        let (moe_dot, moe_serial_norm, moe_batch_norm, moe_squared_error) =
            serial_moe.iter().zip(batch_moe.iter()).fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |sum, (&serial, &batch)| {
                    let serial = serial as f64;
                    let batch = batch as f64;
                    (
                        sum.0 + serial * batch,
                        sum.1 + serial * serial,
                        sum.2 + batch * batch,
                        sum.3 + (serial - batch) * (serial - batch),
                    )
                },
            );
        let moe_cosine = moe_dot / (moe_serial_norm * moe_batch_norm).sqrt();
        let moe_relative_rmse = (moe_squared_error / moe_serial_norm).sqrt();
        assert!(
            moe_error <= 0.02 && moe_cosine >= 0.9 && moe_relative_rmse <= 0.4,
            "batch MoE maximum_error={moe_error} cosine={moe_cosine} relative_rmse={moe_relative_rmse}"
        );
    }

    #[test]
    fn released_exact_moe_pair_matches_serial_across_layers() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        let config = Qwen38FlashNextConfig::load(&model_dir).expect("config");
        let manifest = config.qwen_manifest();
        let checkpoint = ModelOptCheckpoint::open(&model_dir).expect("checkpoint");
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-exact-moe-{}", std::process::id()));
        let lt = CublasLt::new().expect("cuBLASLt");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let zero = DeviceBuffer::zeroed(config.hidden).expect("zero residual");
        let mut serial = Qwen36MoeWorkspace::new(&manifest).expect("serial MoE");
        let mut exact = Qwen36ExactMoePairWorkspace::new(&manifest).expect("exact MoE pair");

        for layer in 0..2 {
            let weights = Qwen36MoeWeights::load_checkpoint_layout(
                &checkpoint,
                &manifest,
                &artifact_dir,
                layer,
            )
            .expect("MoE weights");
            let input_host = (0..2 * config.hidden)
                .map(|index| {
                    let index = index + layer * 17;
                    (index as f32 % 47.0 - 23.0) / 32.0
                })
                .collect::<Vec<_>>();
            let batch_input = DeviceBuffer::from_host(&input_host).expect("batch input");
            let exact_output = weights
                .run_exact_pair(&mut exact, &manifest, &batch_input, &stream)
                .expect("exact pair")
                .copy_to_host(&stream)
                .expect("exact readback")
                .into_vec();
            let replay = weights
                .probe_repeat_exact_pair_gate_up(&mut exact, &stream)
                .expect("gate/up replay");
            for (row, snapshot) in replay.iter().enumerate() {
                assert_eq!(
                    snapshot.routed_gate_up,
                    *snapshot
                        .repeated_routed_gate_up
                        .as_ref()
                        .expect("repeated gate/up"),
                    "layer {layer} row {row} gate/up replay",
                );
            }
            for row in 0..2 {
                let input = DeviceBuffer::from_host(
                    &input_host[row * config.hidden..(row + 1) * config.hidden],
                )
                .expect("serial input");
                let serial_output = weights
                    .run_one_token(
                        &lt,
                        &mut serial,
                        &manifest,
                        &input,
                        &zero,
                        &stream,
                        None,
                        None,
                    )
                    .expect("serial MoE")
                    .ffn_out
                    .copy_to_host(&stream)
                    .expect("serial readback");
                assert_eq!(
                    &exact_output[row * config.hidden..(row + 1) * config.hidden],
                    serial_output.as_slice(),
                    "layer {layer} row {row}",
                );
            }

            if layer == 1 {
                let duplicate_row = (0..config.hidden)
                    .map(|index| ((index as f32 * 0.0137).sin() * 1.75).clamp(-2.0, 2.0))
                    .collect::<Vec<_>>();
                let duplicate_input = DeviceBuffer::from_host(
                    &duplicate_row
                        .iter()
                        .chain(&duplicate_row)
                        .copied()
                        .collect::<Vec<_>>(),
                )
                .expect("duplicate input");
                weights
                    .run_exact_pair(&mut exact, &manifest, &duplicate_input, &stream)
                    .expect("duplicate exact pair");
                let duplicate = exact
                    .probe_snapshots(
                        config.experts,
                        config.experts_per_token,
                        config.hidden,
                        &stream,
                    )
                    .expect("duplicate snapshots");
                assert_eq!(duplicate[0].route_indices, duplicate[1].route_indices);
                assert_eq!(
                    duplicate[0].gate_up_input_values,
                    duplicate[1].gate_up_input_values
                );
                assert_eq!(
                    duplicate[0].gate_up_input_scales,
                    duplicate[1].gate_up_input_scales
                );
                assert_eq!(
                    duplicate[0].routed_gate_up, duplicate[1].routed_gate_up,
                    "independent grouped workspaces must agree",
                );
            }
        }
    }

    #[test]
    fn released_qsa_prefill_rows_match_serial_tokens() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        const TOKENS: usize = 4;
        const LAYER: usize = 3;
        let config = Qwen38FlashNextConfig::load(&model_dir).expect("config");
        let manifest = config.qwen_manifest();
        let checkpoint = ModelOptCheckpoint::open(&model_dir).expect("checkpoint");
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-qsa-{}", std::process::id()));
        let attention = load_hybrid_full_attention(&checkpoint, &manifest, &artifact_dir, LAYER)
            .expect("attention weights");
        let weights =
            Qwen38QsaWeights::load(&checkpoint, &config, LAYER, attention).expect("QSA weights");
        let layer_mask = manifest
            .layer_kinds
            .iter()
            .map(|kind| *kind == QwenLayerKind::FullAttention)
            .collect::<Vec<_>>();
        let input_host = (0..TOKENS * config.hidden)
            .map(|index| (index as f32 % 37.0 - 18.0) / 32.0)
            .collect::<Vec<_>>();
        let batch_input = DeviceBuffer::from_host(&input_host).expect("batch input");
        let page_table = DeviceBuffer::from_host(&[0u32]).expect("page table");
        let page = crate::sm12x_cache::Sm12xPage::from_slot(0);
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut serial_backend = crate::qwen38_flash_next::Qwen38FlashNextPageBackend::new(
            layer_mask.clone(),
            1,
            manifest.kv_heads,
            manifest.head_dim,
            config.indexer_head_dim,
        )
        .expect("serial backend");
        let mut serial_workspace =
            Qwen38QsaWorkspace::new(&config, &manifest, &weights, 128).expect("serial workspace");
        let mut serial_output = Vec::with_capacity(TOKENS * config.hidden);
        for row in 0..TOKENS {
            let input = DeviceBuffer::from_host(
                &input_host[row * config.hidden..(row + 1) * config.hidden],
            )
            .expect("serial input");
            serial_output.extend_from_slice(
                &weights
                    .run_one_token(
                        &mut serial_workspace,
                        &mut serial_backend,
                        &page_table,
                        &page,
                        row,
                        &config,
                        &manifest,
                        &input,
                        LAYER,
                        row,
                        &stream,
                    )
                    .expect("serial QSA")
                    .copy_to_host(&stream)
                    .expect("serial QSA readback"),
            );
        }

        let mut batch_backend = crate::qwen38_flash_next::Qwen38FlashNextPageBackend::new(
            layer_mask.clone(),
            1,
            manifest.kv_heads,
            manifest.head_dim,
            config.indexer_head_dim,
        )
        .expect("batch backend");
        let mut batch_workspace =
            Qwen38QsaWorkspace::new(&config, &manifest, &weights, 128).expect("batch workspace");
        let lt = CublasLt::new().expect("cuBLASLt");
        let batch_model = Qwen36BatchModelView::new(&lt, &manifest, &layer_mask);
        let mut prefill_workspace = weights
            .new_prefill_workspace(&batch_model, &config, TOKENS, 128)
            .expect("QSA prefill workspace");
        weights
            .prepare_prefill(
                &batch_model,
                &mut prefill_workspace,
                &config,
                &batch_input,
                TOKENS,
                0,
                &stream,
            )
            .expect("prepare batch QSA");
        for row in 0..TOKENS {
            weights
                .run_prepared_prefill_row(
                    &batch_model,
                    &mut prefill_workspace,
                    &mut batch_workspace,
                    &mut batch_backend,
                    &page_table,
                    &page,
                    row,
                    &config,
                    row,
                    LAYER,
                    row,
                    &stream,
                )
                .expect("batch QSA");
        }
        let batch_output = weights
            .finish_prefill(&batch_model, &mut prefill_workspace, TOKENS, &stream)
            .expect("finish batch QSA")
            .copy_to_host(&stream)
            .expect("batch QSA readback");
        let max_error = serial_output
            .iter()
            .zip(batch_output.iter())
            .map(|(serial, batch)| (serial - batch).abs())
            .fold(0.0f32, f32::max);
        let dot = serial_output
            .iter()
            .zip(batch_output.iter())
            .map(|(serial, batch)| serial * batch)
            .sum::<f32>();
        let serial_norm = serial_output.iter().map(|value| value * value).sum::<f32>();
        let batch_norm = batch_output.iter().map(|value| value * value).sum::<f32>();
        let squared_error = serial_output
            .iter()
            .zip(batch_output.iter())
            .map(|(serial, batch)| {
                let error = serial - batch;
                error * error
            })
            .sum::<f32>();
        let cosine = dot / (serial_norm * batch_norm).sqrt();
        let relative_rmse = (squared_error / serial_norm).sqrt();
        eprintln!(
            "batch QSA maximum_error={max_error} cosine={cosine} relative_rmse={relative_rmse}"
        );
        assert!(
            max_error <= 0.005 && cosine >= 0.999 && relative_rmse <= 0.01,
            "batch QSA maximum_error={max_error} cosine={cosine} relative_rmse={relative_rmse}"
        );
    }
}
