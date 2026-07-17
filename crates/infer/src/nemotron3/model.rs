use super::linear::{Nemotron3Linear, load_bf16, load_bf16_as_f32};
use super::mtp::{Nemotron3Mtp, Nemotron3MtpState};
use super::{
    Nemotron3AttentionLayer, Nemotron3AttentionRowsWorkspace, Nemotron3AttentionWorkspace,
    Nemotron3LayerKind, Nemotron3MambaLayer, Nemotron3MambaRowsWorkspace, Nemotron3MambaState,
    Nemotron3MambaWorkspace, Nemotron3Manifest, Nemotron3MoeLayer, Nemotron3MoeRowsWorkspace,
    Nemotron3MoeWorkspace, Nemotron3MtpWorkspace, Nemotron3StorageConfig,
};
use crate::runtime::kv_cache::{LayerKvCache, LayerKvCacheCheckpoint};
use nvfp4::{
    CudaGraphExec, CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result,
    argmax_f32_into_on_stream, copy_bf16_row_to_f32_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, copy_row_f32_into_on_stream,
    gather_group_row_f32_into_on_stream, prepend_u32_rows_into_on_stream,
    rms_norm_f32_into_on_stream, select_bf16_state_snapshot_into_on_stream,
    speculative_accept_argmax_f32_into_on_stream,
};
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

    /// Allocates recurrent, KV-cache, and scratch state for one sequence.
    pub fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3DecodeState> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(layer.sequence_state(max_tokens)?);
        }
        Ok(Nemotron3DecodeState {
            hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            layers,
            final_hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(self.manifest.vocab_size)?,
            next_token: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            mtp: self
                .mtp
                .as_ref()
                .map(|mtp| mtp.sequence_state(max_tokens))
                .transpose()?,
            tokens: 0,
        })
    }

    /// Returns the compact device footprint of a reusable prefix checkpoint.
    pub fn checkpoint_sequence_device_bytes(&self, state: &Nemotron3DecodeState) -> usize {
        state.final_hidden.device_bytes()
            + state.logits.device_bytes()
            + state
                .layers
                .iter()
                .map(Nemotron3LayerState::checkpoint_device_bytes)
                .sum::<usize>()
            + state
                .mtp
                .as_ref()
                .map_or(0, |mtp| mtp.cache.checkpoint_device_bytes())
    }

    /// Copies the state required to resume a processed prompt prefix.
    pub fn checkpoint_sequence(
        &self,
        state: &Nemotron3DecodeState,
    ) -> Result<Nemotron3SequenceCheckpoint> {
        let mut layers = Vec::with_capacity(state.layers.len());
        for layer in &state.layers {
            layers.push(match layer {
                Nemotron3LayerState::Mamba { state, .. } => {
                    Nemotron3CheckpointLayer::Mamba(state.checkpoint_on_stream(&self.stream)?)
                }
                Nemotron3LayerState::Moe(_) => Nemotron3CheckpointLayer::Moe,
                Nemotron3LayerState::Attention { cache, .. } => {
                    Nemotron3CheckpointLayer::Attention(cache.checkpoint_on_stream(&self.stream)?)
                }
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
        Ok(Nemotron3SequenceCheckpoint {
            layers,
            final_hidden,
            logits,
            mtp,
            tokens: state.tokens,
        })
    }

    /// Restores a compact prefix checkpoint into a new sequence capacity.
    pub fn restore_sequence_checkpoint(
        &self,
        checkpoint: &Nemotron3SequenceCheckpoint,
        max_tokens: usize,
    ) -> Result<Nemotron3DecodeState> {
        if checkpoint.tokens > max_tokens {
            return Err(Error::Shape {
                label: "Nemotron 3 prefix checkpoint capacity",
                expected: format!("at least {} tokens", checkpoint.tokens),
                actual: format!("{max_tokens} tokens"),
            });
        }
        let mut state = self.sequence_state(max_tokens)?;
        if state.layers.len() != checkpoint.layers.len() {
            return Err(Error::Shape {
                label: "Nemotron 3 prefix checkpoint layers",
                expected: format!("{} layers", state.layers.len()),
                actual: format!("{} layers", checkpoint.layers.len()),
            });
        }
        for (state_layer, checkpoint_layer) in state.layers.iter_mut().zip(&checkpoint.layers) {
            match (state_layer, checkpoint_layer) {
                (
                    Nemotron3LayerState::Mamba { state, .. },
                    Nemotron3CheckpointLayer::Mamba(checkpoint),
                ) => state.restore_checkpoint_on_stream(checkpoint, &self.stream)?,
                (Nemotron3LayerState::Moe(_), Nemotron3CheckpointLayer::Moe) => {}
                (
                    Nemotron3LayerState::Attention { cache, .. },
                    Nemotron3CheckpointLayer::Attention(checkpoint),
                ) => cache.restore_checkpoint_on_stream(checkpoint, &self.stream)?,
                _ => {
                    return Err(Error::Format {
                        label: "Nemotron 3 prefix checkpoint",
                        detail: "checkpoint layer topology does not match the loaded model"
                            .to_string(),
                    });
                }
            }
        }
        state.final_hidden.copy_prefix_from_device_on_stream(
            &checkpoint.final_hidden,
            checkpoint.final_hidden.len(),
            &self.stream,
        )?;
        state.logits.copy_prefix_from_device_on_stream(
            &checkpoint.logits,
            checkpoint.logits.len(),
            &self.stream,
        )?;
        match (&mut state.mtp, &checkpoint.mtp) {
            (Some(state_mtp), Some(checkpoint_mtp)) => state_mtp
                .cache
                .restore_checkpoint_on_stream(checkpoint_mtp, &self.stream)?,
            (None, None) => {}
            _ => {
                return Err(Error::Format {
                    label: "Nemotron 3 prefix checkpoint MTP",
                    detail: "checkpoint MTP state does not match the loaded model".to_string(),
                });
            }
        }
        state.tokens = checkpoint.tokens;
        self.stream.synchronize()?;
        Ok(state)
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
        states: &mut [&mut Nemotron3DecodeState],
        token_chunks: &[&[u32]],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP execution",
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
        state: &mut Nemotron3DecodeState,
        token: u32,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
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

    /// Drafts three greedy tokens with the repeated MTP block. Draft tokens
    /// remain device resident in `workspace` in sequence-major order.
    pub fn draft_three_mtp_argmax(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        initial_tokens: &[u32],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Nemotron 3 MTP drafting",
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
        states: &mut [&mut Nemotron3DecodeState],
        token_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
    ) -> Result<()> {
        if workspace.draft_count.is_some() {
            return Err(Error::Format {
                label: "Nemotron 3 block execution",
                detail: "transactional workspaces must use speculative verification".to_string(),
            });
        }
        self.forward_block_impl(states, token_chunks, workspace, true)
    }

    fn forward_block_impl(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        token_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
        commit_all: bool,
    ) -> Result<()> {
        let sequence_count = states.len();
        if sequence_count == 0 || token_chunks.len() != sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 block sequences",
                expected: "matching non-empty state and token-chunk slices".to_string(),
                actual: format!(
                    "states={sequence_count} token_chunks={}",
                    token_chunks.len()
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

        self.enqueue_prepared_block(states, &offsets, &lengths, workspace, commit_all)
    }

    fn enqueue_prepared_block(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        offsets: &[u32],
        lengths: &[u32],
        workspace: &mut Nemotron3BlockWorkspace,
        commit_all: bool,
    ) -> Result<()> {
        let sequence_count = states.len();
        let rows = workspace.rows;
        let mut conv_ptrs = Vec::with_capacity(workspace.mamba_layers * sequence_count);
        let mut ssm_ptrs = Vec::with_capacity(workspace.mamba_layers * sequence_count);
        let mut key_ptrs = Vec::with_capacity(workspace.attention_layers * sequence_count);
        let mut value_ptrs = Vec::with_capacity(workspace.attention_layers * sequence_count);
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
                        conv_ptrs.push(state.conv_ptr());
                        ssm_ptrs.push(state.ssm_ptr());
                    }
                }
                Nemotron3Layer::Attention(_) => {
                    for state in states.iter_mut() {
                        let Nemotron3LayerState::Attention { cache, .. } = &mut state.layers[layer]
                        else {
                            return Err(Error::Format {
                                label: "Nemotron 3 block attention state",
                                detail: format!("state variant mismatch at layer {layer}"),
                            });
                        };
                        key_ptrs.push(cache.key_ptr());
                        value_ptrs.push(cache.value_ptr());
                    }
                }
                Nemotron3Layer::Moe(_) => {}
            }
        }
        workspace.conv_state_table.copy_from_host(&conv_ptrs)?;
        workspace.ssm_state_table.copy_from_host(&ssm_ptrs)?;
        workspace.key_cache_table.copy_from_host(&key_ptrs)?;
        workspace.value_cache_table.copy_from_host(&value_ptrs)?;

        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            &self.embedding,
            &workspace.tokens,
            workspace.hidden.output(),
            &self.stream,
        )?;
        if let Some(graphs) = &workspace.layer_graphs {
            for graph in graphs {
                graph.launch(&self.stream)?;
            }
        } else {
            self.enqueue_block_layers(workspace, sequence_count, rows, &self.stream)?;
        }
        let last = workspace
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 block model",
                detail: "model has no layers".to_string(),
            })?
            .output();
        if let Some(graph) = &workspace.tail_graph {
            graph.launch(&self.stream)?;
        } else {
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
        }
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
            for (sequence, state) in states.iter_mut().enumerate() {
                for layer in &mut state.layers {
                    if let Nemotron3LayerState::Attention { cache, .. } = layer {
                        cache.advance_len(lengths[sequence] as usize)?;
                    }
                }
                state.tokens += lengths[sequence] as usize;
            }
        }
        Ok(())
    }

    /// Verifies greedy speculative drafts and commits only the accepted prefix.
    /// Acceptance and Mamba state selection remain device resident; only the
    /// compact per-sequence result metadata is copied to the host.
    pub fn verify_speculative_argmax(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        draft_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
    ) -> Result<Nemotron3SpeculativeResult> {
        let draft_count = workspace.draft_count.ok_or_else(|| Error::Format {
            label: "Nemotron 3 speculative verification",
            detail: "a transactional speculative workspace is required".to_string(),
        })?;
        if states.len() != workspace.sequence_count || draft_chunks.len() != states.len() {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative sequences",
                expected: format!("{} states and chunks", workspace.sequence_count),
                actual: format!("states={} chunks={}", states.len(), draft_chunks.len()),
            });
        }
        let previous_logits = states
            .iter()
            .map(|state| state.logits.as_const_ptr().cast::<f32>())
            .collect::<Vec<_>>();
        workspace
            .previous_logits_table
            .copy_from_host(&previous_logits)?;
        self.forward_block_impl(states, draft_chunks, workspace, false)?;
        self.accept_speculative_argmax(states, workspace, draft_count)
    }

    /// Verifies sequence-major drafts already resident on the device.
    pub fn verify_speculative_device_argmax(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        drafted_tokens: &DeviceBuffer<u32>,
        workspace: &mut Nemotron3BlockWorkspace,
    ) -> Result<Nemotron3SpeculativeResult> {
        let draft_count = workspace.draft_count.ok_or_else(|| Error::Format {
            label: "Nemotron 3 speculative verification",
            detail: "a transactional speculative workspace is required".to_string(),
        })?;
        let sequence_count = states.len();
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
        for (sequence, state) in states.iter().enumerate() {
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
        let previous_logits = states
            .iter()
            .map(|state| state.logits.as_const_ptr().cast::<f32>())
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
        self.enqueue_prepared_block(states, &offsets, &lengths, workspace, false)?;
        self.accept_speculative_argmax(states, workspace, draft_count)
    }

    /// Runs one complete greedy MTP draft-and-verify cycle for active sequences.
    ///
    /// Each input token is the target token sampled after the sequence's
    /// current final hidden state. The caller has already emitted that token;
    /// this cycle commits it, verifies three MTP drafts, and returns the newly
    /// accepted draft prefix followed by the next target token.
    pub fn speculative_cycle_argmax(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        input_tokens: &[u32],
        workspace: &mut Nemotron3SpeculativeCycleWorkspace,
    ) -> Result<Nemotron3SpeculativeCycleResult> {
        let sequence_count = states.len();
        workspace.require(self, sequence_count)?;
        if input_tokens.len() != sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative cycle inputs",
                expected: format!("{sequence_count} input tokens"),
                actual: format!("{} input tokens", input_tokens.len()),
            });
        }
        let mut mtp_base_lengths = Vec::with_capacity(sequence_count);
        for (sequence, state) in states.iter().enumerate() {
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
        for (sequence, state) in states.iter().enumerate() {
            workspace.target_hidden.copy_range_from_device_on_stream(
                sequence * self.manifest.hidden_size,
                &state.final_hidden,
                0,
                self.manifest.hidden_size,
                &self.stream,
            )?;
        }
        self.draft_three_mtp_argmax(
            states,
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
            states,
            &workspace.verification_tokens,
            &mut workspace.verification,
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
        self.append_mtp_cache_block(
            states,
            &catchup_tokens,
            &workspace.catchup_hidden,
            &mut workspace.catchup,
        )?;
        let mut accepted_drafts = Vec::with_capacity(sequence_count);
        for (sequence, ((state, &base), &accepted)) in states
            .iter_mut()
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
            for layer in &mut state.layers {
                if let Nemotron3LayerState::Attention { cache, .. } = layer {
                    cache.advance_len(accepted)?;
                }
            }
            state.tokens += accepted;
        }
        Ok(Nemotron3SpeculativeResult {
            accepted_counts: accepted,
            next_tokens,
        })
    }

    fn enqueue_block_layers(
        &self,
        workspace: &mut Nemotron3BlockWorkspace,
        sequence_count: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let mut mamba_layer = 0;
        let mut attention_layer = 0;
        for layer in 0..self.layers.len() {
            self.enqueue_block_layer(
                workspace,
                layer,
                sequence_count,
                rows,
                mamba_layer,
                attention_layer,
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
        workspace: &mut Nemotron3BlockWorkspace,
        layer: usize,
        sequence_count: usize,
        rows: usize,
        mamba_layer: usize,
        attention_layer: usize,
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
            ) => weights.run_rows(
                input,
                scratch,
                &workspace.key_cache_table,
                &workspace.value_cache_table,
                attention_layer * sequence_count,
                &workspace.sequence_offsets,
                &workspace.sequence_lengths,
                &workspace.start_positions,
                sequence_count,
                rows,
                stream,
            ),
            _ => Err(Error::Format {
                label: "Nemotron 3 block layer workspace",
                detail: format!("workspace variant mismatch at layer {layer}"),
            }),
        }
    }

    fn capture_block_graphs(
        &self,
        workspace: &mut Nemotron3BlockWorkspace,
    ) -> Result<(Vec<CudaGraphExec>, CudaGraphExec)> {
        let mut graphs = Vec::with_capacity(self.layers.len());
        let mut mamba_layer = 0;
        let mut attention_layer = 0;
        for layer in 0..self.layers.len() {
            graphs.push(self.stream.capture(|stream| {
                self.enqueue_block_layer(
                    workspace,
                    layer,
                    workspace.sequence_count,
                    workspace.rows,
                    mamba_layer,
                    attention_layer,
                    stream,
                )
            })?);
            match self.layers[layer] {
                Nemotron3Layer::Mamba(_) => mamba_layer += 1,
                Nemotron3Layer::Attention(_) => attention_layer += 1,
                Nemotron3Layer::Moe(_) => {}
            }
        }
        let last = workspace
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 block graph",
                detail: "model has no layers".to_string(),
            })?
            .output();
        let tail = self.stream.capture(|stream| {
            rms_norm_f32_into_on_stream(
                workspace.rows,
                self.manifest.hidden_size,
                last,
                &self.final_norm,
                workspace.final_hidden.output(),
                self.manifest.norm_epsilon,
                stream,
            )?;
            self.lm_head.run_rows(
                &workspace.final_hidden,
                &mut workspace.logits,
                workspace.rows,
                stream,
            )
        })?;
        Ok((graphs, tail))
    }

    /// Runs one token through the complete backbone and language-model head.
    pub fn forward_one(&self, state: &mut Nemotron3DecodeState, token: u32) -> Result<()> {
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
            self.layers[layer].run_one(&mut current[0], input, &self.stream)?;
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
        state.tokens += 1;
        Ok(())
    }

    /// Returns the maximum-logit token after [`Self::forward_one`].
    pub fn argmax(&self, state: &mut Nemotron3DecodeState) -> Result<u32> {
        Ok(self.argmax_with_logit(state)?.0)
    }

    /// Returns the maximum-logit token and its unmodified logit.
    pub fn argmax_with_logit(&self, state: &mut Nemotron3DecodeState) -> Result<(u32, f32)> {
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
    pub fn logits_to_host(&self, state: &Nemotron3DecodeState) -> Result<Vec<f32>> {
        Ok(state.logits.copy_to_host(&self.stream)?.into_vec())
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
    conv_state_table: DeviceBuffer<*mut u16>,
    ssm_state_table: DeviceBuffer<*mut u16>,
    key_cache_table: DeviceBuffer<*mut f32>,
    value_cache_table: DeviceBuffer<*mut f32>,
    previous_logits_table: DeviceBuffer<*const f32>,
    accepted_counts: DeviceBuffer<u32>,
    next_tokens: DeviceBuffer<u32>,
    mamba_snapshots: Option<Vec<Nemotron3MambaSnapshots>>,
    layer_graphs: Option<Vec<CudaGraphExec>>,
    tail_graph: Option<CudaGraphExec>,
    sequence_count: usize,
    rows: usize,
    hidden_size: usize,
    vocab_size: usize,
    mamba_layers: usize,
    attention_layers: usize,
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
        let attention_layers = model
            .layers
            .iter()
            .filter(|layer| matches!(layer, Nemotron3Layer::Attention(_)))
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
        let mut workspace = Self {
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
            key_cache_table: DeviceBuffer::zeroed(attention_layers * sequence_count)?,
            value_cache_table: DeviceBuffer::zeroed(attention_layers * sequence_count)?,
            previous_logits_table: DeviceBuffer::zeroed(sequence_count)?,
            accepted_counts: DeviceBuffer::zeroed(sequence_count)?,
            next_tokens: DeviceBuffer::zeroed(sequence_count)?,
            mamba_snapshots,
            layer_graphs: None,
            tail_graph: None,
            sequence_count,
            rows,
            hidden_size: model.manifest.hidden_size,
            vocab_size: model.manifest.vocab_size,
            mamba_layers,
            attention_layers,
            draft_count,
        };
        let enable_graphs = !std::env::var("EIDER_DISABLE_DECODE_GRAPHS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
        if enable_graphs {
            let (layers, tail) = model.capture_block_graphs(&mut workspace)?;
            workspace.layer_graphs = Some(layers);
            workspace.tail_graph = Some(tail);
        }
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
            + self.key_cache_table.device_bytes()
            + self.value_cache_table.device_bytes()
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
    fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3LayerState> {
        match self {
            Self::Mamba(layer) => Ok(Nemotron3LayerState::Mamba {
                workspace: layer.workspace()?,
                state: layer.sequence_state()?,
            }),
            Self::Moe(layer) => Ok(Nemotron3LayerState::Moe(layer.workspace()?)),
            Self::Attention(layer) => Ok(Nemotron3LayerState::Attention {
                workspace: layer.workspace()?,
                cache: layer.sequence_state(max_tokens)?,
            }),
        }
    }

    fn run_one(
        &self,
        state: &mut Nemotron3LayerState,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        match (self, state) {
            (Self::Mamba(layer), Nemotron3LayerState::Mamba { workspace, state }) => {
                layer.run_one_token(input, workspace, state, stream)
            }
            (Self::Moe(layer), Nemotron3LayerState::Moe(workspace)) => {
                layer.run_one_token(input, workspace, stream)
            }
            (Self::Attention(layer), Nemotron3LayerState::Attention { workspace, cache }) => {
                layer.run_one_token(input, workspace, cache, stream)
            }
            _ => Err(Error::Format {
                label: "Nemotron 3 layer state",
                detail: "layer weights and sequence state variants do not match".to_string(),
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
        cache: LayerKvCache,
    },
}

enum Nemotron3CheckpointLayer {
    Mamba(Nemotron3MambaState),
    Moe,
    Attention(LayerKvCacheCheckpoint),
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
            Self::Attention { workspace, cache } => workspace.device_bytes() + cache.device_bytes(),
        }
    }

    fn checkpoint_device_bytes(&self) -> usize {
        match self {
            Self::Mamba { state, .. } => state.device_bytes(),
            Self::Moe(_) => 0,
            Self::Attention { cache, .. } => cache.checkpoint_device_bytes(),
        }
    }
}

/// Compact device-resident state for one reusable Nemotron prompt prefix.
pub struct Nemotron3SequenceCheckpoint {
    layers: Vec<Nemotron3CheckpointLayer>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    mtp: Option<LayerKvCacheCheckpoint>,
    tokens: usize,
}

impl Nemotron3SequenceCheckpoint {
    /// Returns the number of target tokens represented by this checkpoint.
    pub fn position(&self) -> usize {
        self.tokens
    }

    /// Returns bytes retained by the compact checkpoint.
    pub fn device_bytes(&self) -> usize {
        self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self
                .layers
                .iter()
                .map(|layer| match layer {
                    Nemotron3CheckpointLayer::Mamba(state) => state.device_bytes(),
                    Nemotron3CheckpointLayer::Moe => 0,
                    Nemotron3CheckpointLayer::Attention(cache) => cache.device_bytes(),
                })
                .sum::<usize>()
            + self
                .mtp
                .as_ref()
                .map_or(0, LayerKvCacheCheckpoint::device_bytes)
    }
}

/// Per-sequence state for complete-model decode.
pub struct Nemotron3DecodeState {
    hidden: DeviceBuffer<f32>,
    layers: Vec<Nemotron3LayerState>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_token: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    mtp: Option<Nemotron3MtpState>,
    tokens: usize,
}

impl Nemotron3DecodeState {
    /// Returns the number of tokens already processed by the backbone.
    pub fn len(&self) -> usize {
        self.tokens
    }

    /// Returns true before the first token is processed.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Returns the current language-model logits.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// Returns the normalized final hidden state for the most recent token.
    pub fn final_hidden(&self) -> &DeviceBuffer<f32> {
        &self.final_hidden
    }

    /// Returns bytes owned by this sequence's device-resident state and scratch.
    pub fn device_bytes(&self) -> usize {
        self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3LayerState::device_bytes)
                .sum::<usize>()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_token.device_bytes()
            + self.next_value.device_bytes()
            + self.mtp.as_ref().map_or(0, Nemotron3MtpState::device_bytes)
    }

    fn max_tokens(&self) -> Result<usize> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                Nemotron3LayerState::Attention { cache, .. } => Some(cache.max_tokens()),
                _ => None,
            })
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 sequence state",
                detail: "model has no attention KV cache".to_string(),
            })
    }
}
