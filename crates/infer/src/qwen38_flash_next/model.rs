use super::qsa::{Qwen38QsaWeights, Qwen38QsaWorkspace};
use super::{
    Qwen38FlashNextConfig, Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace,
    Qwen38PagedPle, Qwen38PleState, Qwen38PleTokenWindow, Qwen38PleWeights, Qwen38PleWorkspace,
};
use crate::nvfp4::{
    CublasLt, CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, add_f32_into_on_stream,
    qwen38_repeat_streams_f32_into_on_stream,
};
use crate::qwen3::infer::{QwenLayerKind, QwenModelManifest};
use crate::qwen3::qwen36::{
    Qwen36Embedding, Qwen36LinearAttentionState, Qwen36LinearAttentionWeights,
    Qwen36LinearAttentionWorkspace, Qwen36LmHead, Qwen36LmHeadWorkspace, Qwen36MoeWeights,
    Qwen36MoeWorkspace, load_hybrid_full_attention, load_hybrid_linear_attention,
};
use crate::runtime::qwen38_flash_next_sequence::{
    Qwen38FlashNextSequenceCache, qwen38_flash_next_cache_error,
};
use crate::runtime::sm12x_sequence_cache::{Sm12xCacheContext, Sm12xPageTable};
use seqcache::{AppendReservation, SequenceId};
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
    ple_pager: Qwen38PagedPle,
    ple_weights: Qwen38PleWeights,
    final_mixer: Qwen38HyperConnectionWeights,
    lm_head: Qwen36LmHead,
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
    ple_window: Qwen38PleTokenWindow,
    ple_state: Qwen38PleState,
    ple_workspace: Qwen38PleWorkspace,
    lm_head: Qwen36LmHeadWorkspace,
    position: usize,
    max_tokens: usize,
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
        let ple_pager = Qwen38PagedPle::open(&checkpoint, &config, 1)?;
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
            ple_pager,
            ple_weights,
            final_mixer,
            lm_head,
        })
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
        let mut linear_states = Vec::with_capacity(source.attention_states.len());
        let mut device_bytes = ple_conv.device_bytes();
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
        if let Err(error) = self
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
        let result = self.decode_token_inner(state, cache, &reservation, page_table, token);
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
                state.commit_append()?;
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

    fn decode_token_inner(
        &mut self,
        state: &mut Qwen38FlashNextDecodeState,
        cache: &mut Qwen38FlashNextSequenceCache,
        reservation: &AppendReservation,
        page_table: &Sm12xPageTable,
        token: u32,
    ) -> Result<Qwen38NextToken> {
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
            self.config.hidden,
            self.config.hc_count,
            &state.stream,
        )?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            if layer_index == self.config.ple_layer {
                let (ple, _) = self.ple_weights.run(
                    &mut self.ple_pager,
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

            layer
                .mlp_hyper
                .mix(&state.streams_a, &mut state.mlp_hyper, 1, &state.stream)?;
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
            layer.mlp_hyper.combine(
                &state.streams_a,
                ffn.ffn_out,
                &mut state.mlp_hyper,
                &mut state.streams_b,
                1,
                &state.stream,
            )?;
            std::mem::swap(&mut state.streams_a, &mut state.streams_b);
        }

        self.final_mixer
            .mix(&state.streams_a, &mut state.final_hyper, 1, &state.stream)?;
        state.hidden.copy_prefix_from_device_on_stream(
            state.final_hyper.mixed(),
            self.config.hidden,
            &state.stream,
        )?;
        self.lm_head
            .run_top1(&self.lt, &state.hidden, &mut state.lm_head, &state.stream)?;
        let (id, value) = state.lm_head.read_top1(&state.stream)?;
        Ok(Qwen38NextToken { id, value })
    }

    /// Parsed released-model configuration.
    pub fn config(&self) -> &Qwen38FlashNextConfig {
        &self.config
    }

    pub(crate) fn manifest(&self) -> &QwenModelManifest {
        &self.manifest
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

    fn commit_append(&mut self) -> Result<()> {
        self.ple_window.commit_append()?;
        self.ple_state.commit_append()?;
        self.position += 1;
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
                let mut backend =
                    crate::runtime::qwen38_flash_next_sequence::Qwen38FlashNextPageBackend::new(
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
                let page = crate::runtime::sm12x_sequence_cache::Sm12xPage::from_slot(0);
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
}
