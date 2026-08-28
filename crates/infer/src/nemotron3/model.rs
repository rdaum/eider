use super::linear::{Nemotron3Linear, load_bf16, load_bf16_as_f32};
use super::mtp::{Nemotron3Mtp, Nemotron3MtpState};
use super::{
    Nemotron3AttentionLayer, Nemotron3AttentionRowsWorkspace, Nemotron3AttentionWorkspace,
    Nemotron3CacheContext, Nemotron3KvCacheStorage, Nemotron3LayerKind, Nemotron3MambaLayer,
    Nemotron3MambaRowsWorkspace, Nemotron3MambaState, Nemotron3MambaWorkspace, Nemotron3Manifest,
    Nemotron3MoeLayer, Nemotron3MoeRowsWorkspace, Nemotron3MoeWorkspace, Nemotron3MtpWorkspace,
    Nemotron3Sequence, Nemotron3SequenceCache, Nemotron3StorageConfig, nemotron3_cache_error,
};
use crate::runtime::kv_cache::LayerKvCacheCheckpoint;
use eider_cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, Error, Result, Sm12xKvAttentionWorkspace,
    argmax_f32_into_on_stream, copy_bf16_row_to_f32_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, copy_row_f32_into_on_stream,
    gather_group_row_f32_into_on_stream, prepend_u32_rows_into_on_stream,
    rms_norm_f32_into_on_stream, select_bf16_state_snapshot_into_on_stream,
    speculative_accept_argmax_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::path::Path;
use tracing::info;

const NEMOTRON3_SPECULATIVE_DRAFTS: usize = 3;
const NEMOTRON3_SPECULATIVE_ROWS: usize = NEMOTRON3_SPECULATIVE_DRAFTS + 1;

/// Fully resident Nemotron 3 backbone and language-model head.
pub struct Nemotron3Model {
    manifest: Nemotron3Manifest,
    embedding: DeviceBuffer<u16>,
    layers: Vec<Nemotron3Layer>,
    final_norm: DeviceBuffer<f32>,
    lm_head: Nemotron3Linear,
    mtp: Option<Nemotron3Mtp>,
    compact_kv_cache: bool,
    stream: CudaStream,
}

impl Nemotron3Model {
    /// Loads a Nemotron 3 checkpoint into device memory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_storage(model_dir, Nemotron3StorageConfig::default())
    }

    /// Loads a Nemotron 3 checkpoint with an explicit dense-linear storage policy.
    pub fn load_with_storage(
        model_dir: impl AsRef<Path>,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let manifest = Nemotron3Manifest::from_model_dir(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        manifest.validate_checkpoint_index(&checkpoint)?;
        info!(?storage, "selected Nemotron 3 dense-linear storage");
        let embedding = load_bf16(
            &checkpoint,
            "backbone.embeddings.weight",
            &[manifest.vocab_size, manifest.hidden_size],
        )?;
        let mut layers = Vec::with_capacity(manifest.layers.len());
        let mut device_bytes = embedding.device_bytes();
        for (layer, kind) in manifest.layers.iter().copied().enumerate() {
            let loaded = match kind {
                Nemotron3LayerKind::Mamba => Nemotron3Layer::Mamba(Box::new(
                    Nemotron3MambaLayer::load_with_storage(&checkpoint, &manifest, layer, storage)?,
                )),
                Nemotron3LayerKind::Moe => Nemotron3Layer::Moe(Box::new(
                    Nemotron3MoeLayer::load_with_storage(&checkpoint, &manifest, layer, storage)?,
                )),
                Nemotron3LayerKind::Attention => {
                    Nemotron3Layer::Attention(Box::new(Nemotron3AttentionLayer::load_with_storage(
                        &checkpoint,
                        &manifest,
                        layer,
                        storage,
                    )?))
                }
            };
            device_bytes += loaded.device_bytes();
            info!(
                layer,
                kind = kind.as_str(),
                device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded Nemotron 3 layer"
            );
            layers.push(loaded);
        }
        let final_norm = load_bf16_as_f32(
            &checkpoint,
            "backbone.norm_f.weight",
            &[manifest.hidden_size],
        )?;
        let lm_head = Nemotron3Linear::load(
            &checkpoint,
            "lm_head",
            manifest.vocab_size,
            manifest.hidden_size,
            storage,
        )?;
        let mtp = Nemotron3Mtp::load(&checkpoint, &manifest)?;
        if let Some(mtp) = &mtp {
            device_bytes += final_norm.device_bytes() + lm_head.device_bytes() + mtp.device_bytes();
            info!(
                device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                mtp_weights_gib = mtp.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded Nemotron 3 MTP block"
            );
        }
        Ok(Self {
            manifest,
            embedding,
            layers,
            final_norm,
            lm_head,
            mtp,
            compact_kv_cache: storage.kv_cache == Nemotron3KvCacheStorage::Nvfp4,
            stream: CudaStream::new_non_blocking()?,
        })
    }

    /// Returns the validated model architecture.
    pub fn manifest(&self) -> &Nemotron3Manifest {
        &self.manifest
    }

    /// Returns whether this checkpoint includes a multi-token predictor.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn kv_cache_storage(&self) -> Nemotron3KvCacheStorage {
        if self.compact_kv_cache {
            Nemotron3KvCacheStorage::Nvfp4
        } else {
            Nemotron3KvCacheStorage::F32
        }
    }

    /// Allocates recurrent, KV-cache, and scratch state for one sequence.
    pub(crate) fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3DecodeState> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(layer.sequence_state()?);
        }
        let mut rollback_mamba = Vec::with_capacity(layers.len());
        for layer in &layers {
            rollback_mamba.push(match layer {
                Nemotron3LayerState::Mamba { state, .. } => {
                    Some(state.checkpoint_on_stream(&self.stream)?)
                }
                _ => None,
            });
        }
        Ok(Nemotron3DecodeState {
            hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            layers,
            rollback_mamba,
            rollback_tokens: 0,
            append_pending: false,
            final_hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(self.manifest.vocab_size)?,
            next_token: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            mtp: self
                .mtp
                .as_ref()
                .map(|mtp| mtp.sequence_state(max_tokens))
                .transpose()?,
            compact_attention: self
                .compact_kv_cache
                .then(|| {
                    Sm12xKvAttentionWorkspace::new_gqa(
                        max_tokens,
                        self.manifest.attention_heads,
                        self.manifest.kv_heads,
                        self.manifest.attention_head_dim,
                    )
                })
                .transpose()?,
            tokens: 0,
            max_tokens,
        })
    }

    /// Copies the non-paged state required to resume a processed prompt prefix.
    pub(crate) fn snapshot_sequence(
        &self,
        state: &Nemotron3DecodeState,
    ) -> Result<Nemotron3SequenceSnapshot> {
        let mut layers = Vec::with_capacity(state.layers.len());
        for layer in &state.layers {
            layers.push(match layer {
                Nemotron3LayerState::Mamba { state, .. } => {
                    Nemotron3SnapshotLayer::Mamba(state.checkpoint_on_stream(&self.stream)?)
                }
                Nemotron3LayerState::Moe(_) => Nemotron3SnapshotLayer::Moe,
                Nemotron3LayerState::Attention { .. } => Nemotron3SnapshotLayer::Attention,
            });
        }
        let mut final_hidden = DeviceBuffer::zeroed(state.final_hidden.len())?;
        let mut logits = DeviceBuffer::zeroed(state.logits.len())?;
        final_hidden.copy_prefix_from_device_on_stream(
            &state.final_hidden,
            state.final_hidden.len(),
            &self.stream,
        )?;
        logits.copy_prefix_from_device_on_stream(
            &state.logits,
            state.logits.len(),
            &self.stream,
        )?;
        let mtp = state
            .mtp
            .as_ref()
            .map(|mtp| mtp.cache.checkpoint_on_stream(&self.stream))
            .transpose()?;
        self.stream.synchronize()?;
        Ok(Nemotron3SequenceSnapshot {
            layers,
            final_hidden,
            logits,
            mtp,
            tokens: state.tokens,
        })
    }

    /// Restores retained non-paged state into a new sequence capacity.
    pub(crate) fn restore_sequence_snapshot(
        &self,
        snapshot: &Nemotron3SequenceSnapshot,
        state: &mut Nemotron3DecodeState,
    ) -> Result<()> {
        let max_tokens = state.max_tokens;
        if snapshot.tokens > max_tokens {
            return Err(Error::Shape {
                label: "Nemotron 3 retained prefix capacity",
                expected: format!("at least {} tokens", snapshot.tokens),
                actual: format!("{max_tokens} tokens"),
            });
        }
        if state.layers.len() != snapshot.layers.len() {
            return Err(Error::Shape {
                label: "Nemotron 3 retained prefix layers",
                expected: format!("{} layers", state.layers.len()),
                actual: format!("{} layers", snapshot.layers.len()),
            });
        }
        for (state_layer, snapshot_layer) in state.layers.iter_mut().zip(&snapshot.layers) {
            match (state_layer, snapshot_layer) {
                (
                    Nemotron3LayerState::Mamba { state, .. },
                    Nemotron3SnapshotLayer::Mamba(checkpoint),
                ) => state.restore_checkpoint_on_stream(checkpoint, &self.stream)?,
                (Nemotron3LayerState::Moe(_), Nemotron3SnapshotLayer::Moe) => {}
                (Nemotron3LayerState::Attention { .. }, Nemotron3SnapshotLayer::Attention) => {}
                _ => {
                    return Err(Error::Format {
                        label: "Nemotron 3 retained prefix",
                        detail: "snapshot layer topology does not match the loaded model"
                            .to_string(),
                    });
                }
            }
        }
        state.final_hidden.copy_prefix_from_device_on_stream(
            &snapshot.final_hidden,
            snapshot.final_hidden.len(),
            &self.stream,
        )?;
        state.logits.copy_prefix_from_device_on_stream(
            &snapshot.logits,
            snapshot.logits.len(),
            &self.stream,
        )?;
        match (&mut state.mtp, &snapshot.mtp) {
            (Some(state_mtp), Some(checkpoint_mtp)) => state_mtp
                .cache
                .restore_checkpoint_on_stream(checkpoint_mtp, &self.stream)?,
            (None, None) => {}
            _ => {
                return Err(Error::Format {
                    label: "Nemotron 3 retained prefix MTP",
                    detail: "snapshot MTP state does not match the loaded model".to_string(),
                });
            }
        }
        state.tokens = snapshot.tokens;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Allocates an exact-shape workspace for flattened MTP execution.
    pub fn mtp_workspace(
        &self,
        sequence_count: usize,
        rows: usize,
    ) -> Result<Nemotron3MtpWorkspace> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP workspace",
            detail: "the loaded checkpoint has no MTP block".to_string(),
        })?;
        mtp.workspace(sequence_count, rows)
    }

    /// Runs one ragged MTP block using target-model hidden states in matching
    /// flattened sequence order.
    pub fn forward_mtp_block(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        token_chunks: &[&[u32]],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP execution",
            detail: "the loaded checkpoint has no MTP block".to_string(),
        })?;
        let mut caches = Vec::with_capacity(sequences.len());
        for (sequence, state) in sequences
            .iter_mut()
            .map(|sequence| &mut sequence.state)
            .enumerate()
        {
            let mtp_state = state.mtp.as_mut().ok_or_else(|| Error::Format {
                label: "Nemotron 3 MTP state",
                detail: format!("sequence {sequence} has no MTP state"),
            })?;
            caches.push(&mut mtp_state.cache);
        }
        mtp.run_rows(
            &self.embedding,
            &self.lm_head,
            &mut caches,
            token_chunks,
            target_hidden,
            workspace,
            &self.stream,
        )
    }

    /// Adds one prompt token to the MTP cache using the preceding target hidden state.
    ///
    /// The first target token has no predecessor and must not be passed here.
    pub fn append_mtp_prompt_token(
        &self,
        sequence: &mut Nemotron3Sequence,
        token: u32,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        let state = &mut sequence.state;
        if state.tokens == 0 {
            return Ok(());
        }
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP prompt cache",
            detail: "the loaded checkpoint has no MTP block".to_string(),
        })?;
        let target_hidden = &state.final_hidden;
        let mtp_state = state.mtp.as_mut().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP prompt cache",
            detail: "sequence state has no MTP cache".to_string(),
        })?;
        mtp.append_cache_rows(
            &self.embedding,
            &mut [&mut mtp_state.cache],
            &[std::slice::from_ref(&token)],
            target_hidden,
            workspace,
            &self.stream,
        )
    }

    fn append_mtp_cache_block(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        token_chunks: &[&[u32]],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP cache append",
            detail: "the loaded checkpoint has no MTP block".to_string(),
        })?;
        let mut caches = Vec::with_capacity(states.len());
        for (sequence, state) in states.iter_mut().enumerate() {
            let mtp_state = state.mtp.as_mut().ok_or_else(|| Error::Format {
                label: "Nemotron 3 MTP state",
                detail: format!("sequence {sequence} has no MTP state"),
            })?;
            caches.push(&mut mtp_state.cache);
        }
        mtp.append_cache_rows(
            &self.embedding,
            &mut caches,
            token_chunks,
            target_hidden,
            workspace,
            &self.stream,
        )
    }

    /// Copies the current final hidden state of each sequence into rows of a
    /// caller-owned buffer before a target-model prompt block overwrites it.
    pub fn capture_final_hidden_rows(
        &self,
        sequences: &[&mut Nemotron3Sequence],
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let expected = sequences.len().saturating_mul(self.manifest.hidden_size);
        if output.len() != expected {
            return Err(Error::Shape {
                label: "Nemotron 3 prefill previous hidden states",
                expected: format!("{expected} values"),
                actual: format!("{} values", output.len()),
            });
        }
        for (sequence, state) in sequences.iter().map(|sequence| &sequence.state).enumerate() {
            output.copy_range_from_device_on_stream(
                sequence * self.manifest.hidden_size,
                &state.final_hidden,
                0,
                self.manifest.hidden_size,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// Appends the shifted MTP prompt state after a ragged target-model block.
    ///
    /// Each MTP row pairs a prompt token with the preceding target hidden
    /// state. A sequence beginning at position zero therefore omits its first
    /// token, while later sequences use their captured pre-block final state.
    #[allow(clippy::too_many_arguments)]
    pub fn append_mtp_prompt_block(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        token_chunks: &[&[u32]],
        start_positions: &[usize],
        row_offsets: &[u32],
        previous_hidden: &DeviceBuffer<f32>,
        block_final_hidden: &DeviceBuffer<f32>,
        mtp_hidden: &mut DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        if sequences.len() != token_chunks.len()
            || sequences.len() != start_positions.len()
            || sequences.len() != row_offsets.len()
            || previous_hidden.len() != sequences.len().saturating_mul(self.manifest.hidden_size)
        {
            return Err(Error::Shape {
                label: "Nemotron 3 MTP prompt block",
                expected: "matching sequence metadata and previous hidden rows".to_string(),
                actual: format!(
                    "states={} chunks={} starts={} offsets={} previous_hidden={}",
                    sequences.len(),
                    token_chunks.len(),
                    start_positions.len(),
                    row_offsets.len(),
                    previous_hidden.len()
                ),
            });
        }
        let expected_rows = token_chunks
            .iter()
            .zip(start_positions)
            .map(|(chunk, &start)| chunk.len().saturating_sub(usize::from(start == 0)))
            .sum::<usize>();
        if expected_rows == 0 {
            return Ok(());
        }
        let expected_hidden = expected_rows.saturating_mul(self.manifest.hidden_size);
        if mtp_hidden.len() != expected_hidden {
            return Err(Error::Shape {
                label: "Nemotron 3 MTP prompt hidden rows",
                expected: format!("{expected_hidden} values"),
                actual: format!("{} values", mtp_hidden.len()),
            });
        }

        let mut selected_states = Vec::new();
        let mut selected_chunks = Vec::new();
        let mut destination_row = 0;
        for (sequence, ((state, chunk), (&start, &row_offset))) in sequences
            .iter_mut()
            .map(|sequence| &mut sequence.state)
            .zip(token_chunks)
            .zip(start_positions.iter().zip(row_offsets))
            .enumerate()
        {
            let skip = usize::from(start == 0);
            if chunk.len() <= skip {
                continue;
            }
            if start != 0 {
                mtp_hidden.copy_range_from_device_on_stream(
                    destination_row * self.manifest.hidden_size,
                    previous_hidden,
                    sequence * self.manifest.hidden_size,
                    self.manifest.hidden_size,
                    &self.stream,
                )?;
                destination_row += 1;
            }
            if chunk.len() > 1 {
                let rows = chunk.len() - 1;
                mtp_hidden.copy_range_from_device_on_stream(
                    destination_row * self.manifest.hidden_size,
                    block_final_hidden,
                    row_offset as usize * self.manifest.hidden_size,
                    rows * self.manifest.hidden_size,
                    &self.stream,
                )?;
                destination_row += rows;
            }
            selected_states.push(state);
            selected_chunks.push(&chunk[skip..]);
        }
        debug_assert_eq!(destination_row, expected_rows);
        self.append_mtp_cache_block(
            &mut selected_states,
            &selected_chunks,
            mtp_hidden,
            workspace,
        )
    }

    /// Drafts three greedy tokens with the repeated MTP block. Draft tokens
    /// remain device resident in `workspace` in sequence-major order.
    pub fn draft_three_mtp_argmax(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        initial_tokens: &[u32],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP drafting",
            detail: "the loaded checkpoint has no MTP block".to_string(),
        })?;
        let mut caches = Vec::with_capacity(sequences.len());
        for (sequence, state) in sequences
            .iter_mut()
            .map(|sequence| &mut sequence.state)
            .enumerate()
        {
            let mtp_state = state.mtp.as_mut().ok_or_else(|| Error::Format {
                label: "Nemotron 3 MTP state",
                detail: format!("sequence {sequence} has no MTP state"),
            })?;
            caches.push(&mut mtp_state.cache);
        }
        mtp.draft_three_argmax(
            &self.embedding,
            &self.lm_head,
            &mut caches,
            initial_tokens,
            target_hidden,
            workspace,
            &self.stream,
        )
    }

    /// Allocates an exact-shape workspace for a flattened multi-sequence block.
    pub fn block_workspace(
        &self,
        sequence_count: usize,
        rows: usize,
    ) -> Result<Nemotron3BlockWorkspace> {
        Nemotron3BlockWorkspace::new(self, sequence_count, rows)
    }

    /// Allocates an exact-shape workspace for transactional speculative
    /// verification. Each active sequence contributes `draft_count` rows.
    pub fn speculative_workspace(
        &self,
        sequence_count: usize,
        draft_count: usize,
    ) -> Result<Nemotron3BlockWorkspace> {
        if !(1..=4).contains(&draft_count) {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative draft count",
                expected: "1..=4 drafts per sequence".to_string(),
                actual: draft_count.to_string(),
            });
        }
        Nemotron3BlockWorkspace::new_transactional(self, sequence_count, draft_count)
    }

    /// Allocates reusable storage for one batched three-draft MTP cycle.
    pub fn speculative_cycle_workspace(
        &self,
        sequence_count: usize,
    ) -> Result<Nemotron3SpeculativeCycleWorkspace> {
        Nemotron3SpeculativeCycleWorkspace::new(self, sequence_count)
    }

    /// Runs one ragged token block across multiple sequence states.
    ///
    /// Rows are flattened in sequence order. Every chunk must be non-empty and
    /// the workspace shape must exactly match the sequence and row counts.
    pub fn forward_block(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        token_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
        cache: &mut Nemotron3SequenceCache,
    ) -> Result<()> {
        if sequences.is_empty() || token_chunks.len() != sequences.len() {
            return Err(Error::Shape {
                label: "Nemotron 3 block sequences",
                expected: "matching non-empty sequence and token-chunk slices".to_string(),
                actual: format!(
                    "sequences={} token_chunks={}",
                    sequences.len(),
                    token_chunks.len()
                ),
            });
        }
        if workspace.draft_count.is_some() {
            return Err(Error::Format {
                label: "Nemotron 3 block execution",
                detail: "transactional workspaces must use speculative verification".to_string(),
            });
        }
        let rows = token_chunks.iter().map(|chunk| chunk.len()).sum();
        workspace.require_model(self, sequences.len(), rows)?;
        if token_chunks.iter().any(|chunk| chunk.is_empty()) {
            return Err(Error::Shape {
                label: "Nemotron 3 block tokens",
                expected: "one or more tokens per sequence".to_string(),
                actual: token_chunks
                    .iter()
                    .map(|chunk| chunk.len().to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            });
        }
        let mut reservations = Vec::with_capacity(sequences.len());
        for index in 0..sequences.len() {
            let sequence = &mut sequences[index];
            match cache.reserve_append(
                sequence.cache_id,
                token_chunks[index].len(),
                &mut Nemotron3CacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            ) {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for (sequence, reservation) in
                        sequences[..index].iter_mut().zip(reservations.drain(..))
                    {
                        cache
                            .abort_append(
                                reservation,
                                &mut Nemotron3CacheContext {
                                    stream: &self.stream,
                                    page_table: &mut sequence.page_table,
                                },
                            )
                            .map_err(nemotron3_cache_error)?;
                    }
                    return Err(nemotron3_cache_error(error));
                }
            }
        }
        for index in 0..sequences.len() {
            if let Err(error) = sequences[index].state.begin_append(&self.stream) {
                for sequence in &mut sequences[..index] {
                    let _ = sequence.state.abort_append(&self.stream);
                }
                for (sequence, reservation) in sequences.iter_mut().zip(reservations.drain(..)) {
                    cache
                        .abort_append(
                            reservation,
                            &mut Nemotron3CacheContext {
                                stream: &self.stream,
                                page_table: &mut sequence.page_table,
                            },
                        )
                        .map_err(nemotron3_cache_error)?;
                }
                return Err(error);
            }
        }
        let result = {
            let mut states = Vec::with_capacity(sequences.len());
            let mut page_tables = Vec::with_capacity(sequences.len());
            for sequence in sequences.iter_mut() {
                states.push(&mut sequence.state);
                page_tables.push(sequence.page_table.device());
            }
            self.forward_block_impl(
                &mut states,
                token_chunks,
                workspace,
                true,
                cache,
                &reservations,
                &page_tables,
            )
        };
        if let Err(error) = result {
            let mut rollback_error = None;
            for sequence in sequences.iter_mut() {
                if let Err(error) = sequence.state.abort_append(&self.stream) {
                    rollback_error.get_or_insert(error);
                }
            }
            for (sequence, reservation) in sequences.iter_mut().zip(reservations.drain(..)) {
                cache
                    .abort_append(
                        reservation,
                        &mut Nemotron3CacheContext {
                            stream: &self.stream,
                            page_table: &mut sequence.page_table,
                        },
                    )
                    .map_err(nemotron3_cache_error)?;
            }
            return Err(rollback_error.unwrap_or(error));
        }
        for index in 0..sequences.len() {
            let sequence = &mut sequences[index];
            let rows = token_chunks[index].len();
            if let Err(error) = cache.commit_append(
                reservations[index].clone(),
                rows,
                &mut Nemotron3CacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            ) {
                let mut rollback_error = sequence.state.abort_append(&self.stream).err();
                cache
                    .abort_append(
                        reservations[index].clone(),
                        &mut Nemotron3CacheContext {
                            stream: &self.stream,
                            page_table: &mut sequence.page_table,
                        },
                    )
                    .map_err(nemotron3_cache_error)?;
                for pending in index + 1..sequences.len() {
                    let pending_sequence = &mut sequences[pending];
                    if let Err(error) = pending_sequence.state.abort_append(&self.stream) {
                        rollback_error.get_or_insert(error);
                    }
                    cache
                        .abort_append(
                            reservations[pending].clone(),
                            &mut Nemotron3CacheContext {
                                stream: &self.stream,
                                page_table: &mut pending_sequence.page_table,
                            },
                        )
                        .map_err(nemotron3_cache_error)?;
                }
                return Err(rollback_error.unwrap_or_else(|| nemotron3_cache_error(error)));
            }
            sequence.state.commit_append(rows);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_block_impl(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        token_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
        commit_all: bool,
        cache: &mut Nemotron3SequenceCache,
        reservations: &[seqcache::AppendReservation],
        page_tables: &[&DeviceBuffer<u32>],
    ) -> Result<()> {
        let sequence_count = states.len();
        if sequence_count == 0
            || token_chunks.len() != sequence_count
            || reservations.len() != sequence_count
            || page_tables.len() != sequence_count
        {
            return Err(Error::Shape {
                label: "Nemotron 3 block sequences",
                expected: "matching non-empty state and token-chunk slices".to_string(),
                actual: format!(
                    "states={sequence_count} token_chunks={} reservations={} page_tables={}",
                    token_chunks.len(),
                    reservations.len(),
                    page_tables.len()
                ),
            });
        }
        let rows = token_chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
        workspace.require_model(self, sequence_count, rows)?;
        if let Some(draft_count) = workspace.draft_count
            && token_chunks.iter().any(|chunk| chunk.len() != draft_count)
        {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative chunks",
                expected: format!("{draft_count} rows per sequence"),
                actual: token_chunks
                    .iter()
                    .map(|chunk| chunk.len().to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            });
        }
        let mut tokens = Vec::with_capacity(rows);
        let mut offsets = Vec::with_capacity(sequence_count);
        let mut lengths = Vec::with_capacity(sequence_count);
        let mut starts = Vec::with_capacity(sequence_count);
        for (sequence, (state, chunk)) in states.iter().zip(token_chunks).enumerate() {
            if chunk.is_empty() {
                return Err(Error::Shape {
                    label: "Nemotron 3 block chunk",
                    expected: "at least one token per active sequence".to_string(),
                    actual: format!("sequence {sequence} has 0 tokens"),
                });
            }
            if chunk
                .iter()
                .any(|&token| token as usize >= self.manifest.vocab_size)
            {
                return Err(Error::Shape {
                    label: "Nemotron 3 block token",
                    expected: format!("every token < {}", self.manifest.vocab_size),
                    actual: format!("sequence {sequence} contains an out-of-range token"),
                });
            }
            if state.tokens.saturating_add(chunk.len()) > state.max_tokens()? {
                return Err(Error::Shape {
                    label: "Nemotron 3 block sequence capacity",
                    expected: format!("at most {} total tokens", state.max_tokens()?),
                    actual: format!("{} total tokens", state.tokens.saturating_add(chunk.len())),
                });
            }
            offsets.push(u32::try_from(tokens.len()).map_err(|_| Error::Shape {
                label: "Nemotron 3 block row offset",
                expected: "u32 row offset".to_string(),
                actual: tokens.len().to_string(),
            })?);
            lengths.push(u32::try_from(chunk.len()).map_err(|_| Error::Shape {
                label: "Nemotron 3 block chunk length",
                expected: "u32 chunk length".to_string(),
                actual: chunk.len().to_string(),
            })?);
            starts.push(u32::try_from(state.tokens).map_err(|_| Error::Shape {
                label: "Nemotron 3 block start position",
                expected: "u32 start position".to_string(),
                actual: state.tokens.to_string(),
            })?);
            tokens.extend_from_slice(chunk);
        }
        workspace.tokens.copy_from_host(&tokens)?;
        workspace.sequence_offsets.copy_from_host(&offsets)?;
        workspace.sequence_lengths.copy_from_host(&lengths)?;
        workspace.start_positions.copy_from_host(&starts)?;

        self.enqueue_prepared_block(
            states,
            &offsets,
            &lengths,
            workspace,
            commit_all,
            cache,
            reservations,
            page_tables,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prepared_block(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        offsets: &[u32],
        lengths: &[u32],
        workspace: &mut Nemotron3BlockWorkspace,
        commit_all: bool,
        cache: &mut Nemotron3SequenceCache,
        reservations: &[seqcache::AppendReservation],
        page_tables: &[&DeviceBuffer<u32>],
    ) -> Result<()> {
        let sequence_count = states.len();
        let rows = workspace.rows;
        let mut conv_ptrs = Vec::with_capacity(workspace.mamba_layers * sequence_count);
        let mut ssm_ptrs = Vec::with_capacity(workspace.mamba_layers * sequence_count);
        for layer in 0..self.layers.len() {
            match &self.layers[layer] {
                Nemotron3Layer::Mamba(_) => {
                    for state in states.iter_mut() {
                        let Nemotron3LayerState::Mamba { state, .. } = &mut state.layers[layer]
                        else {
                            return Err(Error::Format {
                                label: "Nemotron 3 block Mamba state",
                                detail: format!("state variant mismatch at layer {layer}"),
                            });
                        };
                        conv_ptrs.push(state.conv_address());
                        ssm_ptrs.push(state.ssm_address());
                    }
                }
                Nemotron3Layer::Attention(_) => {}
                Nemotron3Layer::Moe(_) => {}
            }
        }
        workspace.conv_state_table.copy_from_host(&conv_ptrs)?;
        workspace.ssm_state_table.copy_from_host(&ssm_ptrs)?;
        workspace.page_table_table.copy_from_host(
            &page_tables
                .iter()
                .map(|table| table.cuda_address())
                .collect::<Vec<_>>(),
        )?;

        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            &self.embedding,
            &workspace.tokens,
            workspace.hidden.output(),
            &self.stream,
        )?;
        self.enqueue_block_layers(
            states,
            offsets,
            lengths,
            workspace,
            sequence_count,
            rows,
            cache,
            reservations,
            page_tables,
            &self.stream,
        )?;
        let last = workspace
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 block model",
                detail: "model has no layers".to_string(),
            })?
            .output();
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            last,
            &self.final_norm,
            workspace.final_hidden.output(),
            self.manifest.norm_epsilon,
            &self.stream,
        )?;
        self.lm_head.run_rows(
            &workspace.final_hidden,
            &mut workspace.logits,
            rows,
            &self.stream,
        )?;
        if commit_all {
            for (sequence, state) in states.iter_mut().enumerate() {
                let last_row = offsets[sequence] as usize + lengths[sequence] as usize - 1;
                copy_row_f32_into_on_stream(
                    rows,
                    self.manifest.vocab_size,
                    last_row,
                    &workspace.logits,
                    state.logits.output(),
                    &self.stream,
                )?;
                copy_row_f32_into_on_stream(
                    rows,
                    self.manifest.hidden_size,
                    last_row,
                    &workspace.final_hidden,
                    state.final_hidden.output(),
                    &self.stream,
                )?;
            }
        }
        Ok(())
    }

    /// Verifies greedy speculative drafts and commits only the accepted prefix.
    /// Acceptance and Mamba state selection remain device resident; only the
    /// compact per-sequence result metadata is copied to the host.
    pub fn verify_speculative_argmax(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        draft_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
        cache: &mut Nemotron3SequenceCache,
    ) -> Result<Nemotron3SpeculativeResult> {
        let draft_count = workspace.draft_count.ok_or_else(|| Error::Format {
            label: "Nemotron 3 speculative verification",
            detail: "a transactional speculative workspace is required".to_string(),
        })?;
        if sequences.len() != workspace.sequence_count || draft_chunks.len() != sequences.len() {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative sequences",
                expected: format!("{} states and chunks", workspace.sequence_count),
                actual: format!("states={} chunks={}", sequences.len(), draft_chunks.len()),
            });
        }
        let mut reservations = Vec::with_capacity(sequences.len());
        for (index, sequence) in sequences.iter_mut().enumerate() {
            match cache.reserve_append(
                sequence.cache_id,
                draft_count,
                &mut Nemotron3CacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            ) {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for (sequence, reservation) in
                        sequences[..index].iter_mut().zip(reservations.drain(..))
                    {
                        cache
                            .abort_append(
                                reservation,
                                &mut Nemotron3CacheContext {
                                    stream: &self.stream,
                                    page_table: &mut sequence.page_table,
                                },
                            )
                            .map_err(nemotron3_cache_error)?;
                    }
                    return Err(nemotron3_cache_error(error));
                }
            }
        }
        for index in 0..sequences.len() {
            if let Err(error) = sequences[index].state.begin_append(&self.stream) {
                for sequence in &mut sequences[..index] {
                    let _ = sequence.state.abort_append(&self.stream);
                }
                for (sequence, reservation) in sequences.iter_mut().zip(reservations.drain(..)) {
                    cache
                        .abort_append(
                            reservation,
                            &mut Nemotron3CacheContext {
                                stream: &self.stream,
                                page_table: &mut sequence.page_table,
                            },
                        )
                        .map_err(nemotron3_cache_error)?;
                }
                return Err(error);
            }
        }
        let result = (|| -> Result<Nemotron3SpeculativeResult> {
            let previous_logits = sequences
                .iter()
                .map(|sequence| sequence.state.logits.cuda_address())
                .collect::<Vec<_>>();
            workspace
                .previous_logits_table
                .copy_from_host(&previous_logits)?;
            let mut states = Vec::with_capacity(sequences.len());
            let mut page_tables = Vec::with_capacity(sequences.len());
            for sequence in sequences.iter_mut() {
                states.push(&mut sequence.state);
                page_tables.push(sequence.page_table.device());
            }
            self.forward_block_impl(
                &mut states,
                draft_chunks,
                workspace,
                false,
                cache,
                &reservations,
                &page_tables,
            )?;
            self.accept_speculative_argmax(&mut states, workspace, draft_count)
        })();
        self.finish_speculative_reservations(sequences, reservations, cache, result)
    }

    /// Verifies sequence-major drafts already resident on the device.
    pub fn verify_speculative_device_argmax(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        drafted_tokens: &DeviceBuffer<u32>,
        workspace: &mut Nemotron3BlockWorkspace,
        cache: &mut Nemotron3SequenceCache,
    ) -> Result<Nemotron3SpeculativeResult> {
        let draft_count = workspace.draft_count.ok_or_else(|| Error::Format {
            label: "Nemotron 3 speculative verification",
            detail: "a transactional speculative workspace is required".to_string(),
        })?;
        let sequence_count = sequences.len();
        if sequence_count == 0
            || sequence_count != workspace.sequence_count
            || drafted_tokens.len() != workspace.rows
        {
            return Err(Error::Shape {
                label: "Nemotron 3 device speculative drafts",
                expected: format!(
                    "{} sequences and {} device tokens",
                    workspace.sequence_count, workspace.rows
                ),
                actual: format!(
                    "sequences={sequence_count} device_tokens={}",
                    drafted_tokens.len()
                ),
            });
        }
        let mut offsets = Vec::with_capacity(sequence_count);
        let mut lengths = Vec::with_capacity(sequence_count);
        let mut starts = Vec::with_capacity(sequence_count);
        for (sequence, state) in sequences.iter().map(|sequence| &sequence.state).enumerate() {
            if state.tokens.saturating_add(draft_count) > state.max_tokens()? {
                return Err(Error::Shape {
                    label: "Nemotron 3 speculative sequence capacity",
                    expected: format!("at most {} total tokens", state.max_tokens()?),
                    actual: format!("{} total tokens", state.tokens + draft_count),
                });
            }
            offsets.push(
                u32::try_from(sequence * draft_count).map_err(|_| Error::Shape {
                    label: "Nemotron 3 speculative row offset",
                    expected: "u32 row offset".to_string(),
                    actual: (sequence * draft_count).to_string(),
                })?,
            );
            lengths.push(draft_count as u32);
            starts.push(u32::try_from(state.tokens).map_err(|_| Error::Shape {
                label: "Nemotron 3 speculative start position",
                expected: "u32 start position".to_string(),
                actual: state.tokens.to_string(),
            })?);
        }
        let previous_logits = sequences
            .iter()
            .map(|sequence| sequence.state.logits.cuda_address())
            .collect::<Vec<_>>();
        workspace
            .previous_logits_table
            .copy_from_host(&previous_logits)?;
        workspace.tokens.copy_prefix_from_device_on_stream(
            drafted_tokens,
            workspace.rows,
            &self.stream,
        )?;
        workspace.sequence_offsets.copy_from_host(&offsets)?;
        workspace.sequence_lengths.copy_from_host(&lengths)?;
        workspace.start_positions.copy_from_host(&starts)?;
        let mut reservations = Vec::with_capacity(sequence_count);
        for (index, sequence) in sequences.iter_mut().enumerate() {
            match cache.reserve_append(
                sequence.cache_id,
                draft_count,
                &mut Nemotron3CacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            ) {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for (sequence, reservation) in
                        sequences[..index].iter_mut().zip(reservations.drain(..))
                    {
                        cache
                            .abort_append(
                                reservation,
                                &mut Nemotron3CacheContext {
                                    stream: &self.stream,
                                    page_table: &mut sequence.page_table,
                                },
                            )
                            .map_err(nemotron3_cache_error)?;
                    }
                    return Err(nemotron3_cache_error(error));
                }
            }
        }
        for index in 0..sequences.len() {
            if let Err(error) = sequences[index].state.begin_append(&self.stream) {
                let rollback_error = self.abort_speculative_reservations_from(
                    sequences,
                    &reservations,
                    cache,
                    0,
                    index,
                    reservations.len(),
                );
                return Err(rollback_error.unwrap_or(error));
            }
        }
        let result = (|| -> Result<Nemotron3SpeculativeResult> {
            let mut states = Vec::with_capacity(sequence_count);
            let mut page_tables = Vec::with_capacity(sequence_count);
            for sequence in sequences.iter_mut() {
                states.push(&mut sequence.state);
                page_tables.push(sequence.page_table.device());
            }
            self.enqueue_prepared_block(
                &mut states,
                &offsets,
                &lengths,
                workspace,
                false,
                cache,
                &reservations,
                &page_tables,
            )?;
            self.accept_speculative_argmax(&mut states, workspace, draft_count)
        })();
        self.finish_speculative_reservations(sequences, reservations, cache, result)
    }

    fn abort_speculative_reservations_from(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        reservations: &[seqcache::AppendReservation],
        cache: &mut Nemotron3SequenceCache,
        start: usize,
        state_end: usize,
        end: usize,
    ) -> Option<Error> {
        let mut first_error = None;
        for index in start..end {
            let sequence = &mut sequences[index];
            if index < state_end
                && sequence.state.append_pending
                && let Err(error) = sequence.state.abort_append(&self.stream)
            {
                first_error.get_or_insert(error);
            }
            if let Err(error) = cache.abort_append(
                reservations[index].clone(),
                &mut Nemotron3CacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            ) {
                first_error.get_or_insert_with(|| nemotron3_cache_error(error));
            }
        }
        first_error
    }

    fn finish_speculative_reservations(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        reservations: Vec<seqcache::AppendReservation>,
        cache: &mut Nemotron3SequenceCache,
        result: Result<Nemotron3SpeculativeResult>,
    ) -> Result<Nemotron3SpeculativeResult> {
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                let rollback_error = self.abort_speculative_reservations_from(
                    sequences,
                    &reservations,
                    cache,
                    0,
                    sequences.len(),
                    reservations.len(),
                );
                return Err(rollback_error.unwrap_or(error));
            }
        };
        for index in 0..sequences.len() {
            let accepted = result.accepted_counts()[index];
            if accepted == 0 {
                if let Some(error) = self.abort_speculative_reservations_from(
                    sequences,
                    &reservations,
                    cache,
                    index,
                    index + 1,
                    index + 1,
                ) {
                    let _ = self.abort_speculative_reservations_from(
                        sequences,
                        &reservations,
                        cache,
                        index + 1,
                        sequences.len(),
                        reservations.len(),
                    );
                    return Err(error);
                }
            } else {
                let sequence = &mut sequences[index];
                let reservation = &reservations[index];
                if let Err(error) = cache.commit_append(
                    reservation.clone(),
                    accepted as usize,
                    &mut Nemotron3CacheContext {
                        stream: &self.stream,
                        page_table: &mut sequence.page_table,
                    },
                ) {
                    let state_error = sequence.state.abort_append(&self.stream).err();
                    let rollback_error = self.abort_speculative_reservations_from(
                        sequences,
                        &reservations,
                        cache,
                        index,
                        sequences.len(),
                        reservations.len(),
                    );
                    return Err(state_error
                        .or(rollback_error)
                        .unwrap_or_else(|| nemotron3_cache_error(error)));
                }
                sequence.state.commit_append(accepted as usize);
            }
        }
        Ok(result)
    }

    /// Runs one complete greedy MTP draft-and-verify cycle for active sequences.
    ///
    /// Each input token is the target token sampled after the sequence's
    /// current final hidden state. The caller has already emitted that token;
    /// this cycle commits it, verifies three MTP drafts, and returns the newly
    /// accepted draft prefix followed by the next target token.
    pub fn speculative_cycle_argmax(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        input_tokens: &[u32],
        workspace: &mut Nemotron3SpeculativeCycleWorkspace,
        cache: &mut Nemotron3SequenceCache,
    ) -> Result<Nemotron3SpeculativeCycleResult> {
        let sequence_count = sequences.len();
        workspace.require(self, sequence_count)?;
        if input_tokens.len() != sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative cycle inputs",
                expected: format!("{sequence_count} input tokens"),
                actual: format!("{} input tokens", input_tokens.len()),
            });
        }
        let mut mtp_base_lengths = Vec::with_capacity(sequence_count);
        for (sequence, state) in sequences.iter().map(|sequence| &sequence.state).enumerate() {
            let mtp = state.mtp.as_ref().ok_or_else(|| Error::Format {
                label: "Nemotron 3 speculative cycle MTP state",
                detail: format!("sequence {sequence} has no MTP state"),
            })?;
            if mtp
                .cache
                .len()
                .saturating_add(NEMOTRON3_SPECULATIVE_DRAFTS + 1)
                > mtp.cache.max_tokens()
            {
                return Err(Error::Shape {
                    label: "Nemotron 3 speculative MTP capacity",
                    expected: format!("at most {} cached rows", mtp.cache.max_tokens()),
                    actual: format!(
                        "{} cached rows",
                        mtp.cache.len() + NEMOTRON3_SPECULATIVE_DRAFTS + 1
                    ),
                });
            }
            mtp_base_lengths.push(mtp.cache.len());
        }
        workspace.input_tokens.copy_from_host(input_tokens)?;
        for (sequence, state) in sequences.iter().map(|sequence| &sequence.state).enumerate() {
            workspace.target_hidden.copy_range_from_device_on_stream(
                sequence * self.manifest.hidden_size,
                &state.final_hidden,
                0,
                self.manifest.hidden_size,
                &self.stream,
            )?;
        }
        self.draft_three_mtp_argmax(
            sequences,
            input_tokens,
            &workspace.target_hidden,
            &mut workspace.mtp,
        )?;
        prepend_u32_rows_into_on_stream(
            &workspace.input_tokens,
            workspace.mtp.drafted_tokens(),
            workspace.verification_tokens.output(),
            sequence_count,
            NEMOTRON3_SPECULATIVE_DRAFTS,
            &self.stream,
        )?;
        let verification = self.verify_speculative_device_argmax(
            sequences,
            &workspace.verification_tokens,
            &mut workspace.verification,
            cache,
        )?;
        gather_group_row_f32_into_on_stream(
            workspace.verification.final_hidden(),
            workspace.catchup_hidden.output(),
            sequence_count,
            NEMOTRON3_SPECULATIVE_ROWS,
            NEMOTRON3_SPECULATIVE_ROWS - 2,
            self.manifest.hidden_size,
            &self.stream,
        )?;
        let drafted_tokens = workspace
            .mtp
            .drafted_tokens()
            .copy_to_host(&self.stream)?
            .into_vec();
        let catchup_tokens = drafted_tokens
            .chunks_exact(NEMOTRON3_SPECULATIVE_DRAFTS)
            .map(|tokens| std::slice::from_ref(&tokens[NEMOTRON3_SPECULATIVE_DRAFTS - 1]))
            .collect::<Vec<_>>();
        {
            let mut states = sequences
                .iter_mut()
                .map(|sequence| &mut sequence.state)
                .collect::<Vec<_>>();
            self.append_mtp_cache_block(
                &mut states,
                &catchup_tokens,
                &workspace.catchup_hidden,
                &mut workspace.catchup,
            )?;
        }
        let mut accepted_drafts = Vec::with_capacity(sequence_count);
        for (sequence, ((state, &base), &accepted)) in sequences
            .iter_mut()
            .map(|sequence| &mut sequence.state)
            .zip(&mtp_base_lengths)
            .zip(verification.accepted_counts())
            .enumerate()
        {
            if accepted == 0 {
                return Err(Error::Format {
                    label: "Nemotron 3 speculative cycle input",
                    detail: format!(
                        "sequence {sequence} input token was not the current target argmax"
                    ),
                });
            }
            let committed = base + accepted as usize;
            let mtp = state.mtp.as_mut().ok_or_else(|| Error::Format {
                label: "Nemotron 3 speculative cycle MTP state",
                detail: format!("sequence {sequence} lost its validated MTP state"),
            })?;
            mtp.cache.truncate_len(committed)?;
            accepted_drafts.push(accepted - 1);
        }
        Ok(Nemotron3SpeculativeCycleResult {
            accepted_drafts,
            next_tokens: verification.next_tokens,
            drafted_tokens,
        })
    }

    fn accept_speculative_argmax(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        workspace: &mut Nemotron3BlockWorkspace,
        draft_count: usize,
    ) -> Result<Nemotron3SpeculativeResult> {
        speculative_accept_argmax_f32_into_on_stream(
            &workspace.previous_logits_table,
            &workspace.logits,
            &workspace.tokens,
            workspace.accepted_counts.output(),
            workspace.next_tokens.output(),
            states.len(),
            draft_count,
            self.manifest.vocab_size,
            &self.stream,
        )?;
        let accepted = workspace
            .accepted_counts
            .copy_to_host(&self.stream)?
            .into_vec();
        let next_tokens = workspace.next_tokens.copy_to_host(&self.stream)?.into_vec();
        for (sequence, &accepted) in accepted.iter().enumerate() {
            if accepted as usize > draft_count {
                return Err(Error::Format {
                    label: "Nemotron 3 speculative acceptance",
                    detail: format!(
                        "sequence {sequence} accepted {accepted} of {draft_count} drafts"
                    ),
                });
            }
        }
        let snapshots = workspace
            .mamba_snapshots
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 speculative Mamba state",
                detail: "transaction snapshot storage is missing".to_string(),
            })?;
        let snapshot_slots = draft_count;
        if accepted
            .iter()
            .any(|&accepted| accepted as usize != snapshot_slots)
        {
            let conv_state_size = self
                .manifest
                .mamba_conv_channels()
                .saturating_mul(self.manifest.mamba_conv_kernel);
            let ssm_state_size = self
                .manifest
                .mamba_intermediate_size()
                .saturating_mul(self.manifest.mamba_state_size);
            for (mamba_layer, snapshot) in snapshots.iter().enumerate() {
                let state_table_offset = mamba_layer * states.len();
                select_bf16_state_snapshot_into_on_stream(
                    &workspace.conv_state_table,
                    state_table_offset,
                    &snapshot.conv,
                    &workspace.accepted_counts,
                    states.len(),
                    snapshot_slots,
                    conv_state_size,
                    &self.stream,
                )?;
                select_bf16_state_snapshot_into_on_stream(
                    &workspace.ssm_state_table,
                    state_table_offset,
                    &snapshot.ssm,
                    &workspace.accepted_counts,
                    states.len(),
                    snapshot_slots,
                    ssm_state_size,
                    &self.stream,
                )?;
            }
        }
        for (sequence, (state, &accepted)) in states.iter_mut().zip(&accepted).enumerate() {
            let accepted = accepted as usize;
            if accepted != 0 {
                let row = sequence * draft_count + accepted - 1;
                copy_row_f32_into_on_stream(
                    workspace.rows,
                    self.manifest.vocab_size,
                    row,
                    &workspace.logits,
                    state.logits.output(),
                    &self.stream,
                )?;
                copy_row_f32_into_on_stream(
                    workspace.rows,
                    self.manifest.hidden_size,
                    row,
                    &workspace.final_hidden,
                    state.final_hidden.output(),
                    &self.stream,
                )?;
            }
        }
        Ok(Nemotron3SpeculativeResult {
            accepted_counts: accepted,
            next_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_block_layers(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        offsets: &[u32],
        lengths: &[u32],
        workspace: &mut Nemotron3BlockWorkspace,
        sequence_count: usize,
        rows: usize,
        cache: &mut Nemotron3SequenceCache,
        reservations: &[seqcache::AppendReservation],
        page_tables: &[&DeviceBuffer<u32>],
        stream: &CudaStream,
    ) -> Result<()> {
        let mut mamba_layer = 0;
        let mut attention_layer = 0;
        for layer in 0..self.layers.len() {
            self.enqueue_block_layer(
                states,
                offsets,
                lengths,
                workspace,
                layer,
                sequence_count,
                rows,
                mamba_layer,
                attention_layer,
                cache,
                reservations,
                page_tables,
                stream,
            )?;
            match self.layers[layer] {
                Nemotron3Layer::Mamba(_) => mamba_layer += 1,
                Nemotron3Layer::Attention(_) => attention_layer += 1,
                Nemotron3Layer::Moe(_) => {}
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_block_layer(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        offsets: &[u32],
        lengths: &[u32],
        workspace: &mut Nemotron3BlockWorkspace,
        layer: usize,
        sequence_count: usize,
        rows: usize,
        mamba_layer: usize,
        _attention_layer: usize,
        cache: &mut Nemotron3SequenceCache,
        reservations: &[seqcache::AppendReservation],
        page_tables: &[&DeviceBuffer<u32>],
        stream: &CudaStream,
    ) -> Result<()> {
        let (previous, current) = workspace.layers.split_at_mut(layer);
        let input = if layer == 0 {
            &workspace.hidden
        } else {
            previous[layer - 1].output()
        };
        match (&self.layers[layer], &mut current[0]) {
            (Nemotron3Layer::Mamba(weights), Nemotron3LayerRowsWorkspace::Mamba(scratch)) => {
                if let Some(snapshots) = workspace.mamba_snapshots.as_mut() {
                    let snapshot = &mut snapshots[mamba_layer];
                    weights.run_rows_transactional(
                        input,
                        scratch,
                        &workspace.conv_state_table,
                        &workspace.ssm_state_table,
                        mamba_layer * sequence_count,
                        &workspace.sequence_offsets,
                        &workspace.sequence_lengths,
                        sequence_count,
                        rows,
                        &mut snapshot.conv,
                        &mut snapshot.ssm,
                        workspace.draft_count.expect("transaction draft count"),
                        stream,
                    )
                } else {
                    weights.run_rows(
                        input,
                        scratch,
                        &workspace.conv_state_table,
                        &workspace.ssm_state_table,
                        mamba_layer * sequence_count,
                        &workspace.sequence_offsets,
                        &workspace.sequence_lengths,
                        sequence_count,
                        rows,
                        stream,
                    )
                }
            }
            (Nemotron3Layer::Moe(weights), Nemotron3LayerRowsWorkspace::Moe(scratch)) => {
                weights.run_rows(input, scratch, rows, stream)
            }
            (
                Nemotron3Layer::Attention(weights),
                Nemotron3LayerRowsWorkspace::Attention(scratch),
            ) => {
                let starts = states
                    .iter()
                    .map(|state| state.tokens as u32)
                    .collect::<Vec<_>>();
                cache
                    .with_append_reservations(reservations, |backend, pages| {
                        weights.run_rows_paged(
                            input,
                            scratch,
                            backend,
                            pages,
                            &workspace.page_table_table,
                            page_tables,
                            &workspace.sequence_offsets,
                            &workspace.sequence_lengths,
                            &workspace.start_positions,
                            offsets,
                            lengths,
                            &starts,
                            sequence_count,
                            rows,
                            workspace.compact_attention.as_mut(),
                            stream,
                        )
                    })
                    .map_err(nemotron3_cache_error)
            }
            _ => Err(Error::Format {
                label: "Nemotron 3 block layer workspace",
                detail: format!("workspace variant mismatch at layer {layer}"),
            }),
        }
    }

    /// Runs one token through the complete backbone and language-model head.
    pub fn forward_one(
        &self,
        sequence: &mut Nemotron3Sequence,
        cache: &mut Nemotron3SequenceCache,
        token: u32,
    ) -> Result<()> {
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                1,
                &mut Nemotron3CacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(nemotron3_cache_error)?;
        if let Err(error) = sequence.state.begin_append(&self.stream) {
            cache
                .abort_append(
                    reservation,
                    &mut Nemotron3CacheContext {
                        stream: &self.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(nemotron3_cache_error)?;
            return Err(error);
        }
        let result = self.forward_one_reserved(
            &mut sequence.state,
            cache,
            &reservation,
            sequence.page_table.device(),
            token,
        );
        if let Err(error) = result {
            let state_error = sequence.state.abort_append(&self.stream).err();
            cache
                .abort_append(
                    reservation.clone(),
                    &mut Nemotron3CacheContext {
                        stream: &self.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(nemotron3_cache_error)?;
            return Err(state_error.unwrap_or(error));
        }
        if let Err(error) = cache.commit_append(
            reservation.clone(),
            1,
            &mut Nemotron3CacheContext {
                stream: &self.stream,
                page_table: &mut sequence.page_table,
            },
        ) {
            let state_error = sequence.state.abort_append(&self.stream).err();
            cache
                .abort_append(
                    reservation,
                    &mut Nemotron3CacheContext {
                        stream: &self.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(nemotron3_cache_error)?;
            return Err(state_error.unwrap_or_else(|| nemotron3_cache_error(error)));
        }
        sequence.state.commit_append(1);
        Ok(())
    }

    fn forward_one_reserved(
        &self,
        state: &mut Nemotron3DecodeState,
        cache: &mut Nemotron3SequenceCache,
        reservation: &seqcache::AppendReservation,
        page_table: &DeviceBuffer<u32>,
        token: u32,
    ) -> Result<()> {
        if token as usize >= self.manifest.vocab_size {
            return Err(Error::Shape {
                label: "Nemotron 3 token",
                expected: format!("token < {}", self.manifest.vocab_size),
                actual: token.to_string(),
            });
        }
        if state.tokens >= state.max_tokens()? {
            return Err(Error::Shape {
                label: "Nemotron 3 sequence capacity",
                expected: format!("fewer than {} tokens", state.max_tokens()?),
                actual: state.tokens.to_string(),
            });
        }
        copy_bf16_row_to_f32_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            token as usize,
            &self.embedding,
            state.hidden.output(),
            &self.stream,
        )?;
        for layer in 0..self.layers.len() {
            let (previous, current) = state.layers.split_at_mut(layer);
            let input = if layer == 0 {
                &state.hidden
            } else {
                previous[layer - 1].output()
            };
            match (&self.layers[layer], &mut current[0]) {
                (
                    Nemotron3Layer::Mamba(weights),
                    Nemotron3LayerState::Mamba { workspace, state },
                ) => {
                    weights.run_one_token(input, workspace, state, &self.stream)?;
                }
                (Nemotron3Layer::Moe(weights), Nemotron3LayerState::Moe(workspace)) => {
                    weights.run_one_token(input, workspace, &self.stream)?;
                }
                (
                    Nemotron3Layer::Attention(weights),
                    Nemotron3LayerState::Attention { workspace },
                ) => {
                    cache
                        .with_append_pages(reservation, |backend, pages| {
                            weights.run_one_token_paged(
                                input,
                                workspace,
                                backend,
                                pages,
                                page_table,
                                state.tokens,
                                state.compact_attention.as_mut(),
                                &self.stream,
                            )
                        })
                        .map_err(nemotron3_cache_error)?;
                }
                _ => {
                    return Err(Error::Format {
                        label: "Nemotron 3 layer state",
                        detail: format!("state variant mismatch at layer {layer}"),
                    });
                }
            }
        }
        let last = state
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 model",
                detail: "model has no layers".to_string(),
            })?
            .output();
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            last,
            &self.final_norm,
            state.final_hidden.output(),
            self.manifest.norm_epsilon,
            &self.stream,
        )?;
        self.lm_head
            .run(&state.final_hidden, &mut state.logits, &self.stream)?;
        Ok(())
    }

    /// Returns the maximum-logit token after [`Self::forward_one`].
    pub fn argmax(&self, sequence: &mut Nemotron3Sequence) -> Result<u32> {
        Ok(self.argmax_with_logit(sequence)?.0)
    }

    /// Returns the maximum-logit token and its unmodified logit.
    pub fn argmax_with_logit(&self, sequence: &mut Nemotron3Sequence) -> Result<(u32, f32)> {
        let state = &mut sequence.state;
        argmax_f32_into_on_stream(
            &state.logits,
            state.next_token.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        let token = state.next_token.copy_to_host(&self.stream)?[0];
        let value = state.next_value.copy_to_host(&self.stream)?[0];
        Ok((token, value))
    }

    /// Copies the current vocabulary logits to host memory for sampling.
    pub fn logits_to_host(&self, sequence: &Nemotron3Sequence) -> Result<Vec<f32>> {
        Ok(sequence.state.logits.copy_to_host(&self.stream)?.into_vec())
    }

    /// Copies flattened MTP logits to host memory.
    pub fn mtp_logits_to_host(&self, workspace: &Nemotron3MtpWorkspace) -> Result<Vec<f32>> {
        Ok(workspace.logits().copy_to_host(&self.stream)?.into_vec())
    }

    /// Copies three draft-token rows to host in sequence-major order.
    pub fn mtp_drafted_tokens_to_host(
        &self,
        workspace: &Nemotron3MtpWorkspace,
    ) -> Result<Vec<u32>> {
        Ok(workspace
            .drafted_tokens()
            .copy_to_host(&self.stream)?
            .into_vec())
    }

    /// Waits for all work enqueued by this model instance.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Returns bytes owned by device-resident model weights.
    pub fn device_bytes(&self) -> usize {
        self.embedding.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3Layer::device_bytes)
                .sum::<usize>()
            + self.final_norm.device_bytes()
            + self.lm_head.device_bytes()
            + self.mtp.as_ref().map_or(0, Nemotron3Mtp::device_bytes)
    }
}

enum Nemotron3LayerRowsWorkspace {
    Mamba(Box<Nemotron3MambaRowsWorkspace>),
    Moe(Box<Nemotron3MoeRowsWorkspace>),
    Attention(Nemotron3AttentionRowsWorkspace),
}

/// Compact host-visible outcome of one device-accepted speculative batch.
pub struct Nemotron3SpeculativeResult {
    accepted_counts: Vec<u32>,
    next_tokens: Vec<u32>,
}

impl Nemotron3SpeculativeResult {
    /// Returns the accepted draft-prefix length for each sequence.
    pub fn accepted_counts(&self) -> &[u32] {
        &self.accepted_counts
    }

    /// Returns the target fallback token, or bonus token after full acceptance.
    pub fn next_tokens(&self) -> &[u32] {
        &self.next_tokens
    }
}

/// Host-visible tokens and acceptance metadata from a complete MTP cycle.
pub struct Nemotron3SpeculativeCycleResult {
    accepted_drafts: Vec<u32>,
    next_tokens: Vec<u32>,
    drafted_tokens: Vec<u32>,
}

impl Nemotron3SpeculativeCycleResult {
    /// Returns the accepted draft-prefix length for each sequence.
    pub fn accepted_counts(&self) -> &[u32] {
        &self.accepted_drafts
    }

    /// Returns all three draft tokens in sequence-major order.
    pub fn drafted_tokens(&self) -> &[u32] {
        &self.drafted_tokens
    }

    /// Returns the target fallback or bonus token for each sequence.
    pub fn next_tokens(&self) -> &[u32] {
        &self.next_tokens
    }

    /// Returns the accepted draft prefix followed by the fallback or bonus
    /// token for one sequence.
    pub fn emitted_tokens(&self, sequence: usize) -> Result<Vec<u32>> {
        let Some(&accepted) = self.accepted_counts().get(sequence) else {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative result sequence",
                expected: format!("sequence < {}", self.accepted_counts().len()),
                actual: sequence.to_string(),
            });
        };
        let begin = sequence * NEMOTRON3_SPECULATIVE_DRAFTS;
        let mut emitted = self.drafted_tokens[begin..begin + accepted as usize].to_vec();
        emitted.push(self.next_tokens()[sequence]);
        Ok(emitted)
    }
}

/// Reusable target, MTP, verification, and catch-up storage for one batched
/// speculative cycle.
pub struct Nemotron3SpeculativeCycleWorkspace {
    input_tokens: DeviceBuffer<u32>,
    target_hidden: DeviceBuffer<f32>,
    mtp: Nemotron3MtpWorkspace,
    verification_tokens: DeviceBuffer<u32>,
    verification: Nemotron3BlockWorkspace,
    catchup: Nemotron3MtpWorkspace,
    catchup_hidden: DeviceBuffer<f32>,
    sequence_count: usize,
    hidden_size: usize,
}

impl Nemotron3SpeculativeCycleWorkspace {
    fn new(model: &Nemotron3Model, sequence_count: usize) -> Result<Self> {
        if sequence_count == 0 {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative cycle workspace",
                expected: "at least one sequence".to_string(),
                actual: "0 sequences".to_string(),
            });
        }
        Ok(Self {
            input_tokens: DeviceBuffer::zeroed(sequence_count)?,
            target_hidden: DeviceBuffer::zeroed(sequence_count * model.manifest.hidden_size)?,
            mtp: model.mtp_workspace(sequence_count, sequence_count)?,
            verification_tokens: DeviceBuffer::zeroed(sequence_count * NEMOTRON3_SPECULATIVE_ROWS)?,
            verification: model
                .speculative_workspace(sequence_count, NEMOTRON3_SPECULATIVE_ROWS)?,
            catchup: model.mtp_workspace(sequence_count, sequence_count)?,
            catchup_hidden: DeviceBuffer::zeroed(sequence_count * model.manifest.hidden_size)?,
            sequence_count,
            hidden_size: model.manifest.hidden_size,
        })
    }

    fn require(&self, model: &Nemotron3Model, sequence_count: usize) -> Result<()> {
        if self.sequence_count == sequence_count && self.hidden_size == model.manifest.hidden_size {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 speculative cycle workspace",
            expected: format!(
                "sequences={sequence_count} hidden={}",
                model.manifest.hidden_size
            ),
            actual: format!(
                "sequences={} hidden={}",
                self.sequence_count, self.hidden_size
            ),
        })
    }

    /// Returns bytes owned by cycle scratch, graphs, and transaction slots.
    pub fn device_bytes(&self) -> usize {
        self.input_tokens.device_bytes()
            + self.target_hidden.device_bytes()
            + self.mtp.device_bytes()
            + self.verification_tokens.device_bytes()
            + self.verification.device_bytes()
            + self.catchup.device_bytes()
            + self.catchup_hidden.device_bytes()
    }
}

struct Nemotron3MambaSnapshots {
    conv: DeviceBuffer<u16>,
    ssm: DeviceBuffer<u16>,
}

impl Nemotron3LayerRowsWorkspace {
    fn output(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Mamba(workspace) => workspace.output(),
            Self::Moe(workspace) => workspace.output(),
            Self::Attention(workspace) => workspace.output(),
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Self::Mamba(workspace) => workspace.device_bytes(),
            Self::Moe(workspace) => workspace.device_bytes(),
            Self::Attention(workspace) => workspace.device_bytes(),
        }
    }
}

/// Exact-shape scratch, metadata, and pointer tables for block execution.
pub struct Nemotron3BlockWorkspace {
    tokens: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layers: Vec<Nemotron3LayerRowsWorkspace>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    start_positions: DeviceBuffer<u32>,
    conv_state_table: DeviceBuffer<DeviceAddress<u16>>,
    ssm_state_table: DeviceBuffer<DeviceAddress<u16>>,
    page_table_table: DeviceBuffer<DeviceAddress<u32>>,
    compact_attention: Option<Sm12xKvAttentionWorkspace>,
    previous_logits_table: DeviceBuffer<DeviceAddress<f32>>,
    accepted_counts: DeviceBuffer<u32>,
    next_tokens: DeviceBuffer<u32>,
    mamba_snapshots: Option<Vec<Nemotron3MambaSnapshots>>,
    sequence_count: usize,
    rows: usize,
    hidden_size: usize,
    vocab_size: usize,
    mamba_layers: usize,
    draft_count: Option<usize>,
}

impl Nemotron3BlockWorkspace {
    fn new(model: &Nemotron3Model, sequence_count: usize, rows: usize) -> Result<Self> {
        Self::new_impl(model, sequence_count, rows, None)
    }

    fn new_transactional(
        model: &Nemotron3Model,
        sequence_count: usize,
        draft_count: usize,
    ) -> Result<Self> {
        Self::new_impl(
            model,
            sequence_count,
            sequence_count.saturating_mul(draft_count),
            Some(draft_count),
        )
    }

    fn new_impl(
        model: &Nemotron3Model,
        sequence_count: usize,
        rows: usize,
        draft_count: Option<usize>,
    ) -> Result<Self> {
        if sequence_count == 0 || rows < sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 block workspace",
                expected: "non-empty sequences with at least one row each".to_string(),
                actual: format!("sequences={sequence_count} rows={rows}"),
            });
        }
        let mamba_layers = model
            .layers
            .iter()
            .filter(|layer| matches!(layer, Nemotron3Layer::Mamba(_)))
            .count();
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(match layer {
                Nemotron3Layer::Mamba(layer) => {
                    Nemotron3LayerRowsWorkspace::Mamba(Box::new(layer.rows_workspace(rows)?))
                }
                Nemotron3Layer::Moe(layer) => {
                    Nemotron3LayerRowsWorkspace::Moe(Box::new(layer.rows_workspace(rows)?))
                }
                Nemotron3Layer::Attention(layer) => {
                    Nemotron3LayerRowsWorkspace::Attention(layer.rows_workspace(rows)?)
                }
            });
        }
        let mamba_snapshots = draft_count
            .map(|draft_count| {
                let snapshot_slots = draft_count;
                let conv_state_size = model
                    .manifest
                    .mamba_conv_channels()
                    .saturating_mul(model.manifest.mamba_conv_kernel);
                let ssm_state_size = model
                    .manifest
                    .mamba_intermediate_size()
                    .saturating_mul(model.manifest.mamba_state_size);
                (0..mamba_layers)
                    .map(|_| {
                        Ok(Nemotron3MambaSnapshots {
                            conv: DeviceBuffer::zeroed(
                                sequence_count * snapshot_slots * conv_state_size,
                            )?,
                            ssm: DeviceBuffer::zeroed(
                                sequence_count * snapshot_slots * ssm_state_size,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let workspace = Self {
            tokens: DeviceBuffer::zeroed(rows)?,
            hidden: DeviceBuffer::zeroed(rows * model.manifest.hidden_size)?,
            layers,
            final_hidden: DeviceBuffer::zeroed(rows * model.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(rows * model.manifest.vocab_size)?,
            sequence_offsets: DeviceBuffer::zeroed(sequence_count)?,
            sequence_lengths: DeviceBuffer::zeroed(sequence_count)?,
            start_positions: DeviceBuffer::zeroed(sequence_count)?,
            conv_state_table: DeviceBuffer::zeroed(mamba_layers * sequence_count)?,
            ssm_state_table: DeviceBuffer::zeroed(mamba_layers * sequence_count)?,
            page_table_table: DeviceBuffer::zeroed(sequence_count)?,
            compact_attention: model
                .compact_kv_cache
                .then(|| {
                    Sm12xKvAttentionWorkspace::new_gqa_batched(
                        model.manifest.max_position_embeddings,
                        model.manifest.attention_heads,
                        model.manifest.kv_heads,
                        model.manifest.attention_head_dim,
                        8,
                    )
                })
                .transpose()?,
            previous_logits_table: DeviceBuffer::zeroed(sequence_count)?,
            accepted_counts: DeviceBuffer::zeroed(sequence_count)?,
            next_tokens: DeviceBuffer::zeroed(sequence_count)?,
            mamba_snapshots,
            sequence_count,
            rows,
            hidden_size: model.manifest.hidden_size,
            vocab_size: model.manifest.vocab_size,
            mamba_layers,
            draft_count,
        };
        Ok(workspace)
    }

    fn require_model(
        &self,
        model: &Nemotron3Model,
        sequence_count: usize,
        rows: usize,
    ) -> Result<()> {
        if self.sequence_count == sequence_count
            && self.rows == rows
            && self.hidden_size == model.manifest.hidden_size
            && self.vocab_size == model.manifest.vocab_size
            && self.layers.len() == model.layers.len()
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 block workspace",
            expected: format!("sequences={sequence_count} rows={rows} matching loaded model"),
            actual: format!(
                "sequences={} rows={} hidden={} vocab={} layers={}",
                self.sequence_count,
                self.rows,
                self.hidden_size,
                self.vocab_size,
                self.layers.len()
            ),
        })
    }

    /// Returns all row logits in flattened sequence order.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// Returns normalized final hidden states in flattened sequence order.
    pub fn final_hidden(&self) -> &DeviceBuffer<f32> {
        &self.final_hidden
    }

    /// Returns bytes owned by block scratch and pointer tables.
    pub fn device_bytes(&self) -> usize {
        self.tokens.device_bytes()
            + self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3LayerRowsWorkspace::device_bytes)
                .sum::<usize>()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.sequence_offsets.device_bytes()
            + self.sequence_lengths.device_bytes()
            + self.start_positions.device_bytes()
            + self.conv_state_table.device_bytes()
            + self.ssm_state_table.device_bytes()
            + self.page_table_table.device_bytes()
            + self
                .compact_attention
                .as_ref()
                .map_or(0, Sm12xKvAttentionWorkspace::device_bytes)
            + self.previous_logits_table.device_bytes()
            + self.accepted_counts.device_bytes()
            + self.next_tokens.device_bytes()
            + self.mamba_snapshots.as_ref().map_or(0, |snapshots| {
                snapshots
                    .iter()
                    .map(|snapshot| snapshot.conv.device_bytes() + snapshot.ssm.device_bytes())
                    .sum()
            })
    }
}

enum Nemotron3Layer {
    Mamba(Box<Nemotron3MambaLayer>),
    Moe(Box<Nemotron3MoeLayer>),
    Attention(Box<Nemotron3AttentionLayer>),
}

impl Nemotron3Layer {
    fn sequence_state(&self) -> Result<Nemotron3LayerState> {
        match self {
            Self::Mamba(layer) => Ok(Nemotron3LayerState::Mamba {
                workspace: layer.workspace()?,
                state: layer.sequence_state()?,
            }),
            Self::Moe(layer) => Ok(Nemotron3LayerState::Moe(layer.workspace()?)),
            Self::Attention(layer) => Ok(Nemotron3LayerState::Attention {
                workspace: layer.workspace()?,
            }),
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Self::Mamba(layer) => layer.device_bytes(),
            Self::Moe(layer) => layer.device_bytes(),
            Self::Attention(layer) => layer.device_bytes(),
        }
    }
}

enum Nemotron3LayerState {
    Mamba {
        workspace: Nemotron3MambaWorkspace,
        state: Nemotron3MambaState,
    },
    Moe(Nemotron3MoeWorkspace),
    Attention {
        workspace: Nemotron3AttentionWorkspace,
    },
}

enum Nemotron3SnapshotLayer {
    Mamba(Nemotron3MambaState),
    Moe,
    Attention,
}

impl Nemotron3LayerState {
    fn output(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Mamba { workspace, .. } => &workspace.output,
            Self::Moe(workspace) => &workspace.output,
            Self::Attention { workspace, .. } => &workspace.output,
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Self::Mamba { workspace, state } => workspace.device_bytes() + state.device_bytes(),
            Self::Moe(workspace) => workspace.device_bytes(),
            Self::Attention { workspace } => workspace.device_bytes(),
        }
    }
}

/// Compact device-resident non-paged state for one reusable Nemotron prompt prefix.
pub struct Nemotron3SequenceSnapshot {
    layers: Vec<Nemotron3SnapshotLayer>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    mtp: Option<LayerKvCacheCheckpoint>,
    tokens: usize,
}

impl Nemotron3SequenceSnapshot {
    /// Returns bytes retained by the compact snapshot.
    pub fn device_bytes(&self) -> usize {
        self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self
                .layers
                .iter()
                .map(|layer| match layer {
                    Nemotron3SnapshotLayer::Mamba(state) => state.device_bytes(),
                    Nemotron3SnapshotLayer::Moe => 0,
                    Nemotron3SnapshotLayer::Attention => 0,
                })
                .sum::<usize>()
            + self
                .mtp
                .as_ref()
                .map_or(0, LayerKvCacheCheckpoint::device_bytes)
    }
}

impl seqcache::RetainedSnapshot for Nemotron3SequenceSnapshot {
    fn retained_bytes(&self) -> usize {
        self.device_bytes()
    }
}

/// Per-sequence state for complete-model decode.
pub(crate) struct Nemotron3DecodeState {
    hidden: DeviceBuffer<f32>,
    layers: Vec<Nemotron3LayerState>,
    rollback_mamba: Vec<Option<Nemotron3MambaState>>,
    rollback_tokens: usize,
    append_pending: bool,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_token: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    mtp: Option<Nemotron3MtpState>,
    compact_attention: Option<Sm12xKvAttentionWorkspace>,
    tokens: usize,
    pub(crate) max_tokens: usize,
}

impl Nemotron3DecodeState {
    /// Returns the number of tokens already processed by the backbone.
    pub fn len(&self) -> usize {
        self.tokens
    }

    /// Returns bytes owned by this sequence's device-resident state and scratch.
    pub fn device_bytes(&self) -> usize {
        self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3LayerState::device_bytes)
                .sum::<usize>()
            + self
                .rollback_mamba
                .iter()
                .flatten()
                .map(Nemotron3MambaState::device_bytes)
                .sum::<usize>()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_token.device_bytes()
            + self.next_value.device_bytes()
            + self.mtp.as_ref().map_or(0, Nemotron3MtpState::device_bytes)
            + self
                .compact_attention
                .as_ref()
                .map_or(0, Sm12xKvAttentionWorkspace::device_bytes)
    }

    fn max_tokens(&self) -> Result<usize> {
        Ok(self.max_tokens)
    }

    fn begin_append(&mut self, stream: &CudaStream) -> Result<()> {
        if self.append_pending {
            return Err(Error::Format {
                label: "Nemotron 3 recurrent transaction",
                detail: "an append transaction is already pending".to_string(),
            });
        }
        for (layer, rollback) in self.layers.iter().zip(&mut self.rollback_mamba) {
            match (layer, rollback) {
                (Nemotron3LayerState::Mamba { state, .. }, Some(rollback)) => {
                    rollback.restore_checkpoint_on_stream(state, stream)?;
                }
                (Nemotron3LayerState::Moe(_), None)
                | (Nemotron3LayerState::Attention { .. }, None) => {}
                _ => unreachable!("Nemotron recurrent rollback topology matches active state"),
            }
        }
        self.rollback_tokens = self.tokens;
        self.append_pending = true;
        Ok(())
    }

    fn commit_append(&mut self, rows: usize) {
        assert!(self.append_pending, "Nemotron recurrent append is pending");
        self.tokens = self.rollback_tokens + rows;
        self.append_pending = false;
    }

    fn abort_append(&mut self, stream: &CudaStream) -> Result<()> {
        if !self.append_pending {
            return Err(Error::Format {
                label: "Nemotron 3 recurrent transaction",
                detail: "no append transaction is pending".to_string(),
            });
        }
        for (layer, rollback) in self.layers.iter_mut().zip(&self.rollback_mamba) {
            match (layer, rollback) {
                (Nemotron3LayerState::Mamba { state, .. }, Some(rollback)) => {
                    state.restore_checkpoint_on_stream(rollback, stream)?;
                }
                (Nemotron3LayerState::Moe(_), None)
                | (Nemotron3LayerState::Attention { .. }, None) => {}
                _ => unreachable!("Nemotron recurrent rollback topology matches active state"),
            }
        }
        self.tokens = self.rollback_tokens;
        self.append_pending = false;
        Ok(())
    }
}
