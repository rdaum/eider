//! Nemotron 3 backbone loading and batched text-model execution.

use super::lilicorr::{
    LiLiCorrDraft, LiLiCorrProposal, LiLiCorrSequence, LiLiCorrSnapshot, LiLiCorrWorkspace,
};
use super::linear::{Nemotron3Linear, load_bf16, load_bf16_as_f32, norm};
use super::mtp::{Nemotron3Mtp, Nemotron3MtpState};
use super::{
    Nemotron3AttentionLayer, Nemotron3AttentionRowsWorkspace, Nemotron3AttentionWorkspace,
    Nemotron3CacheContext, Nemotron3KvCacheStorage, Nemotron3LayerKind, Nemotron3MambaLayer,
    Nemotron3MambaRowsWorkspace, Nemotron3MambaState, Nemotron3MambaWorkspace, Nemotron3Manifest,
    Nemotron3MoeLayer, Nemotron3MoeLayerConfig, Nemotron3MoeRowsWorkspace, Nemotron3MoeWorkspace,
    Nemotron3MtpWorkspace, Nemotron3Sequence, Nemotron3SequenceCache, Nemotron3StorageConfig,
    nemotron3_cache_error,
};
use crate::execution::kv_cache::LayerKvCacheCheckpoint;
use eider_cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceAddress, DeviceBuffer, Error, F32DenseGemm,
    PinnedHostBuffer, Result, Sm12xKvAttentionWorkspace, argmax_f32_batch_into_on_stream,
    argmax_f32_into_on_stream, copy_bf16_row_to_f32_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, copy_row_f32_into_on_stream,
    gather_group_row_f32_into_on_stream, prepend_u32_rows_into_on_stream,
    select_bf16_state_snapshot_into_on_stream, select_mamba_ssm_state_snapshot_into_on_stream,
    speculative_accept_argmax_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use eider_runtime::sampling::{Sampler, TokenHistory};
use std::path::Path;
use tracing::info;

mod decision;
pub(crate) use decision::Nemotron3DecisionReadout;

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
    lilicorr: Option<LiLiCorrDraft>,
    compact_kv_cache: bool,
    compute_precision: eider_cuda::GemmPrecision,
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
        let mut manifest = Nemotron3Manifest::from_model_dir(model_dir)?;
        if let Some(dtype) = storage.mamba_ssm_dtype {
            manifest.mamba_ssm_dtype = dtype;
        }
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        manifest.validate_checkpoint_index(&checkpoint)?;
        info!(?storage, "selected Nemotron 3 dense-linear storage");
        let embedding = load_bf16(
            &checkpoint,
            &manifest.tensor_name("backbone.embeddings.weight"),
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
            &manifest.tensor_name("backbone.norm_f.weight"),
            &[manifest.hidden_size],
        )?;
        let lm_head = Nemotron3Linear::load(
            &checkpoint,
            &manifest.tensor_name("lm_head"),
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
            lilicorr: None,
            compact_kv_cache: storage.kv_cache == Nemotron3KvCacheStorage::Nvfp4,
            compute_precision: storage.compute_precision,
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

    /// Enables the external LiLiCorr companion before creating any sequences.
    pub fn enable_lilicorr(&mut self, root: impl AsRef<Path>) -> Result<()> {
        let draft = LiLiCorrDraft::load(root, &self.manifest)?;
        if !draft.config().has_own_lm_head && self.lm_head.lilicorr_head().is_none() {
            return Err(Error::Format {
                label: "LiLiCorr shared LM head",
                detail: "Bolt requires its target BF16 or Q8_0 LM head; retain --nemotron-bf16-storage bf16".into(),
            });
        }
        self.lilicorr = Some(draft);
        Ok(())
    }

    /// Returns the loaded LiLiCorr companion, when configured.
    pub fn lilicorr(&self) -> Option<&LiLiCorrDraft> {
        self.lilicorr.as_ref()
    }

    /// Adds committed rows from a prefill or verification workspace to draft context.
    pub fn append_lilicorr_block_context(
        &self,
        sequence: &mut Nemotron3Sequence,
        block: &Nemotron3BlockWorkspace,
        row: usize,
        rows: usize,
        workspace: &mut LiLiCorrWorkspace,
    ) -> Result<()> {
        let draft = self.lilicorr.as_ref().ok_or_else(|| Error::Format {
            label: "LiLiCorr context",
            detail: "companion is not enabled".into(),
        })?;
        let context = sequence
            .state
            .lilicorr
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "LiLiCorr context",
                detail: "sequence has no draft context".into(),
            })?;
        if context.position().checked_add(rows) != Some(sequence.state.tokens) {
            return Err(Error::Format {
                label: "LiLiCorr context",
                detail: "committed target and draft positions differ".into(),
            });
        }
        let captured = block
            .captured_features
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "LiLiCorr context",
                detail: "block did not capture target features".into(),
            })?;
        draft.append_context(context, captured, row, rows, workspace, &self.stream)
    }

    /// Adds the last committed ordinary decode token to draft context.
    pub fn append_lilicorr_decode_context(
        &self,
        sequence: &mut Nemotron3Sequence,
        workspace: &mut LiLiCorrWorkspace,
    ) -> Result<()> {
        let draft = self.lilicorr.as_ref().ok_or_else(|| Error::Format {
            label: "LiLiCorr context",
            detail: "companion is not enabled".into(),
        })?;
        let context = sequence
            .state
            .lilicorr
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "LiLiCorr context",
                detail: "sequence has no draft context".into(),
            })?;
        if context.position().checked_add(1) != Some(sequence.state.tokens) {
            return Err(Error::Format {
                label: "LiLiCorr context",
                detail: "committed target and draft positions differ".into(),
            });
        }
        let captured = sequence
            .state
            .captured_features
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "LiLiCorr context",
                detail: "decode did not capture target features".into(),
            })?;
        draft.append_context(context, captured, 0, 1, workspace, &self.stream)
    }

    /// Proposes a suffix, verifies it with the target, and commits only its accepted prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn lilicorr_cycle(
        &self,
        sequence: &mut Nemotron3Sequence,
        anchor: u32,
        drafts: usize,
        draft_workspace: &mut LiLiCorrWorkspace,
        verification: &mut Nemotron3BlockWorkspace,
        sampler: &mut Sampler,
        history: &TokenHistory,
        cache: &mut Nemotron3SequenceCache,
    ) -> Result<Nemotron3SpeculativeCycleResult> {
        let draft = self.lilicorr.as_ref().ok_or_else(|| Error::Format {
            label: "LiLiCorr cycle",
            detail: "companion is not enabled".into(),
        })?;
        let context = sequence
            .state
            .lilicorr
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "LiLiCorr cycle",
                detail: "sequence has no draft context".into(),
            })?;
        if !(1..=7).contains(&drafts)
            || context.position() != sequence.state.tokens
            || verification.draft_count != Some(drafts + 1)
            || history.tokens().last() != Some(&anchor)
        {
            return Err(Error::Format {
                label: "LiLiCorr cycle",
                detail: "anchor, verification shape, or context position is inconsistent".into(),
            });
        }
        draft.propose(
            &self.embedding,
            self.lm_head.lilicorr_head(),
            context,
            anchor,
            drafts,
            draft_workspace,
            &self.stream,
        )?;
        let proposal =
            LiLiCorrProposal::select(draft_workspace, anchor, drafts, sampler, &self.stream)?;
        let result = self.verify_speculative_by(
            &mut [sequence],
            &[proposal.tokens()],
            verification,
            cache,
            |states, ws, count| {
                let (accepted, next) = if sampler.config().uses_fast_argmax() {
                    argmax_f32_batch_into_on_stream(
                        &ws.logits,
                        ws.row_argmax.output(),
                        ws.row_maximum.output(),
                        count,
                        self.manifest.vocab_size,
                        &self.stream,
                    )?;
                    let target = ws.row_argmax.copy_to_host(&self.stream)?;
                    let accepted = proposal.tokens()[1..]
                        .iter()
                        .zip(target.iter())
                        .take_while(|(draft, target)| draft == target)
                        .count();
                    (accepted, target[accepted])
                } else {
                    let logits = ws.logits.copy_to_host(&self.stream)?;
                    proposal.verify(&logits, self.manifest.vocab_size, sampler, history)?
                };
                // The anchor was already sampled and emitted. Its target row always commits.
                let committed = (accepted + 1) as u32;
                if ws.upload_pending {
                    ws.upload_done.synchronize()?;
                }
                ws.host_lengths.copy_from_slice(&[committed])?;
                ws.accepted_counts.copy_range_from_pinned_on_stream(
                    0,
                    &ws.host_lengths,
                    &self.stream,
                )?;
                ws.upload_done.record_on_stream(&self.stream)?;
                ws.upload_pending = true;
                self.select_speculative_prefix(states, ws, count, vec![committed], vec![next])
            },
        )?;
        self.append_lilicorr_block_context(
            sequence,
            verification,
            0,
            result.accepted_counts[0] as usize,
            draft_workspace,
        )?;
        Ok(Nemotron3SpeculativeCycleResult {
            draft_count: drafts,
            accepted_drafts: vec![result.accepted_counts[0] - 1],
            next_tokens: result.next_tokens,
            drafted_tokens: proposal.tokens()[1..].to_vec(),
        })
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
        self.sequence_state_impl(max_tokens, true)
    }

    pub(crate) fn decision_sequence_state(
        &self,
        max_tokens: usize,
    ) -> Result<Nemotron3DecodeState> {
        self.sequence_state_impl(max_tokens, false)
    }

    fn sequence_state_impl(
        &self,
        max_tokens: usize,
        generation: bool,
    ) -> Result<Nemotron3DecodeState> {
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
        let dense_gemm = if self.compute_precision == eider_cuda::GemmPrecision::Bf16 {
            let mut weights = Vec::new();
            for layer in &self.layers {
                match layer {
                    Nemotron3Layer::Mamba(l) => weights.extend(l.dense_linears()),
                    Nemotron3Layer::Moe(l) => weights.extend(l.dense_linears()),
                    Nemotron3Layer::Attention(l) => weights.extend(l.dense_linears()),
                }
            }
            if generation {
                weights.push(&self.lm_head);
            }
            Some(Nemotron3Linear::dense_workspace(
                1,
                self.compute_precision,
                weights.into_iter(),
            )?)
        } else {
            None
        };
        Ok(Nemotron3DecodeState {
            dense_gemm,
            decode_warm: false,
            decode_graphs: (0..=self.layers.len()).map(|_| None).collect(),
            hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            layers,
            rollback_mamba,
            rollback_tokens: 0,
            append_pending: false,
            final_hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(if generation {
                self.manifest.vocab_size
            } else {
                0
            })?,
            next_token: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            lilicorr: self
                .lilicorr
                .as_ref()
                .filter(|_| generation)
                .map(LiLiCorrDraft::new_sequence)
                .transpose()?,
            captured_features: self
                .lilicorr
                .as_ref()
                .filter(|_| generation)
                .map(|d| {
                    DeviceBuffer::zeroed(
                        self.manifest.hidden_size * d.config().target_layers().len(),
                    )
                })
                .transpose()?,
            mtp: self
                .mtp
                .as_ref()
                .filter(|_| generation)
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
        let lilicorr = match (&self.lilicorr, &state.lilicorr) {
            (Some(draft), Some(context)) if context.position() == state.tokens => {
                Some(draft.snapshot(context, &self.stream)?)
            }
            (None, None) => None,
            _ => {
                return Err(Error::Format {
                    label: "Nemotron LiLiCorr prefix",
                    detail: "draft context is not aligned with the target prefix".into(),
                });
            }
        };
        self.stream.synchronize()?;
        Ok(Nemotron3SequenceSnapshot {
            layers,
            final_hidden,
            logits,
            mtp,
            lilicorr,
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
        match (&self.lilicorr, &snapshot.lilicorr, &mut state.lilicorr) {
            (Some(draft), Some(saved), Some(context)) if saved.position() == snapshot.tokens => {
                draft.restore(saved, context, &self.stream)?
            }
            (None, None, None) => {}
            _ => {
                return Err(Error::Format {
                    label: "Nemotron LiLiCorr prefix",
                    detail: "snapshot draft state does not match the target".into(),
                });
            }
        }
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
        mtp.workspace(&self.lm_head, sequence_count, rows)
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

    /// Allocates prompt scratch without vocabulary logits for intermediate rows.
    /// Final hidden states remain available for MTP cache preparation.
    pub fn prefill_workspace(
        &self,
        sequence_count: usize,
        rows: usize,
    ) -> Result<Nemotron3BlockWorkspace> {
        Nemotron3BlockWorkspace::new_impl(self, sequence_count, rows, None, true)
    }

    pub(crate) fn decision_workspace(
        &self,
        sequences: usize,
        rows: usize,
    ) -> Result<Nemotron3BlockWorkspace> {
        let mut workspace = self.prefill_workspace(sequences, rows)?;
        workspace.captured_features = None;
        workspace.capture_graphs = true;
        workspace.segment_dense_rows = true;
        Ok(workspace)
    }

    /// Allocates an exact-shape workspace for transactional speculative
    /// verification. Each active sequence contributes `draft_count` rows.
    pub fn speculative_workspace(
        &self,
        sequence_count: usize,
        draft_count: usize,
    ) -> Result<Nemotron3BlockWorkspace> {
        if !(1..=8).contains(&draft_count) {
            return Err(Error::Shape {
                label: "Nemotron 3 speculative draft count",
                expected: "1..=8 verification rows per sequence".to_string(),
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
        if workspace.upload_pending {
            workspace.upload_done.synchronize()?;
        }
        workspace.host_tokens.copy_from_slice(&tokens)?;
        workspace.host_offsets.copy_from_slice(&offsets)?;
        workspace.host_lengths.copy_from_slice(&lengths)?;
        workspace.host_starts.copy_from_slice(&starts)?;
        workspace.tokens.copy_range_from_pinned_on_stream(
            0,
            &workspace.host_tokens,
            &self.stream,
        )?;
        workspace
            .sequence_offsets
            .copy_range_from_pinned_on_stream(0, &workspace.host_offsets, &self.stream)?;
        workspace
            .sequence_lengths
            .copy_range_from_pinned_on_stream(0, &workspace.host_lengths, &self.stream)?;
        workspace.start_positions.copy_range_from_pinned_on_stream(
            0,
            &workspace.host_starts,
            &self.stream,
        )?;

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
        if workspace.segment_dense_rows && workspace.dense_row_lengths != lengths {
            workspace.graphs.iter_mut().for_each(|graph| *graph = None);
            workspace.graph_warm = false;
            workspace.dense_gemm.set_row_segments(lengths)?;
            workspace.dense_row_lengths = lengths.to_vec();
        }
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
        if !conv_ptrs.is_empty() {
            workspace.host_conv.copy_from_slice(&conv_ptrs)?;
            workspace.host_ssm.copy_from_slice(&ssm_ptrs)?;
            workspace
                .conv_state_table
                .copy_range_from_pinned_on_stream(0, &workspace.host_conv, &self.stream)?;
            workspace.ssm_state_table.copy_range_from_pinned_on_stream(
                0,
                &workspace.host_ssm,
                &self.stream,
            )?;
        }
        workspace.host_pages.copy_from_slice(
            &page_tables
                .iter()
                .map(|table| table.cuda_address())
                .collect::<Vec<_>>(),
        )?;
        workspace
            .page_table_table
            .copy_range_from_pinned_on_stream(0, &workspace.host_pages, &self.stream)?;
        workspace.upload_done.record_on_stream(&self.stream)?;
        workspace.upload_pending = true;

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
        if let Some(graph) = &workspace.graphs[self.layers.len()] {
            graph.launch(&self.stream)?;
        } else if workspace.graph_warm {
            let graph = self
                .stream
                .capture(|stream| self.enqueue_block_head(workspace, stream))?;
            graph.launch(&self.stream)?;
            workspace.graphs[self.layers.len()] = Some(graph);
        } else {
            self.enqueue_block_head(workspace, &self.stream)?;
        }
        workspace.graph_warm = workspace.capture_graphs;
        if commit_all {
            for (sequence, state) in states.iter_mut().enumerate() {
                let last_row = offsets[sequence] as usize + lengths[sequence] as usize - 1;
                copy_row_f32_into_on_stream(
                    rows,
                    self.manifest.hidden_size,
                    last_row,
                    &workspace.final_hidden,
                    state.final_hidden.output(),
                    &self.stream,
                )?;
                if state.logits.is_empty() {
                    continue;
                }
                if workspace.last_logits_only {
                    self.lm_head.run(
                        &state.final_hidden,
                        &mut state.logits,
                        state.dense_gemm.as_mut(),
                        &self.stream,
                    )?;
                } else {
                    copy_row_f32_into_on_stream(
                        rows,
                        self.manifest.vocab_size,
                        last_row,
                        &workspace.logits,
                        state.logits.output(),
                        &self.stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn enqueue_block_head(
        &self,
        workspace: &mut Nemotron3BlockWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = workspace.rows;
        norm(
            self.compute_precision,
            rows,
            self.manifest.hidden_size,
            &workspace.hidden,
            &self.final_norm,
            workspace.final_hidden.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        if !workspace.last_logits_only {
            self.lm_head.run_rows(
                &workspace.final_hidden,
                &mut workspace.logits,
                rows,
                &mut workspace.dense_gemm,
                stream,
            )?;
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
        self.verify_speculative_by(
            sequences,
            draft_chunks,
            workspace,
            cache,
            |states, workspace, count| self.accept_speculative_argmax(states, workspace, count),
        )
    }

    fn verify_speculative_by<F>(
        &self,
        sequences: &mut [&mut Nemotron3Sequence],
        draft_chunks: &[&[u32]],
        workspace: &mut Nemotron3BlockWorkspace,
        cache: &mut Nemotron3SequenceCache,
        decide: F,
    ) -> Result<Nemotron3SpeculativeResult>
    where
        F: FnOnce(
            &mut [&mut Nemotron3DecodeState],
            &mut Nemotron3BlockWorkspace,
            usize,
        ) -> Result<Nemotron3SpeculativeResult>,
    {
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
            decide(&mut states, workspace, draft_count)
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
        if workspace.upload_pending {
            workspace.upload_done.synchronize()?;
        }
        workspace.host_offsets.copy_from_slice(&offsets)?;
        workspace.host_lengths.copy_from_slice(&lengths)?;
        workspace.host_starts.copy_from_slice(&starts)?;
        workspace
            .sequence_offsets
            .copy_range_from_pinned_on_stream(0, &workspace.host_offsets, &self.stream)?;
        workspace
            .sequence_lengths
            .copy_range_from_pinned_on_stream(0, &workspace.host_lengths, &self.stream)?;
        workspace.start_positions.copy_range_from_pinned_on_stream(
            0,
            &workspace.host_starts,
            &self.stream,
        )?;
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
            draft_count: NEMOTRON3_SPECULATIVE_DRAFTS,
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
        self.select_speculative_prefix(states, workspace, draft_count, accepted, next_tokens)
    }

    fn select_speculative_prefix(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        workspace: &mut Nemotron3BlockWorkspace,
        draft_count: usize,
        accepted: Vec<u32>,
        next_tokens: Vec<u32>,
    ) -> Result<Nemotron3SpeculativeResult> {
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
                select_mamba_ssm_state_snapshot_into_on_stream(
                    &workspace.ssm_state_table,
                    state_table_offset,
                    &snapshot.ssm,
                    &workspace.accepted_counts,
                    states.len(),
                    snapshot_slots,
                    ssm_state_size,
                    self.manifest.mamba_ssm_dtype,
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
        let mut layer = 0;
        while layer < self.layers.len() {
            if matches!(self.layers[layer], Nemotron3Layer::Attention(_)) {
                self.enqueue_block_layer(
                    states,
                    offsets,
                    lengths,
                    workspace,
                    layer,
                    sequence_count,
                    rows,
                    mamba_layer,
                    0,
                    cache,
                    reservations,
                    page_tables,
                    stream,
                )?;
                layer += 1;
                continue;
            }
            let end = (layer..self.layers.len())
                .find(|&index| matches!(self.layers[index], Nemotron3Layer::Attention(_)))
                .unwrap_or(self.layers.len());
            if let Some(graph) = &workspace.graphs[layer] {
                graph.launch(stream)?;
            } else if workspace.graph_warm {
                let graph = stream.capture(|stream| {
                    self.enqueue_block_segment(
                        states,
                        offsets,
                        lengths,
                        workspace,
                        layer,
                        end,
                        sequence_count,
                        rows,
                        mamba_layer,
                        cache,
                        reservations,
                        page_tables,
                        stream,
                    )
                })?;
                graph.launch(stream)?;
                workspace.graphs[layer] = Some(graph);
            } else {
                self.enqueue_block_segment(
                    states,
                    offsets,
                    lengths,
                    workspace,
                    layer,
                    end,
                    sequence_count,
                    rows,
                    mamba_layer,
                    cache,
                    reservations,
                    page_tables,
                    stream,
                )?;
            }
            mamba_layer += self.layers[layer..end]
                .iter()
                .filter(|layer| matches!(layer, Nemotron3Layer::Mamba(_)))
                .count();
            layer = end;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_block_segment(
        &self,
        states: &mut [&mut Nemotron3DecodeState],
        offsets: &[u32],
        lengths: &[u32],
        workspace: &mut Nemotron3BlockWorkspace,
        start: usize,
        end: usize,
        sequence_count: usize,
        rows: usize,
        mut mamba_layer: usize,
        cache: &mut Nemotron3SequenceCache,
        reservations: &[seqcache::AppendReservation],
        page_tables: &[&DeviceBuffer<u32>],
        stream: &CudaStream,
    ) -> Result<()> {
        for layer in start..end {
            self.enqueue_block_layer(
                states,
                offsets,
                lengths,
                workspace,
                layer,
                sequence_count,
                rows,
                mamba_layer,
                0,
                cache,
                reservations,
                page_tables,
                stream,
            )?;
            if matches!(self.layers[layer], Nemotron3Layer::Mamba(_)) {
                mamba_layer += 1;
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
        let slot = workspace.layer_slots[layer];
        let input = &workspace.hidden;
        match (&self.layers[layer], &mut workspace.layers[slot]) {
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
                        &mut workspace.dense_gemm,
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
                        &mut workspace.dense_gemm,
                        stream,
                    )
                }
            }
            (Nemotron3Layer::Moe(weights), Nemotron3LayerRowsWorkspace::Moe(scratch)) => {
                weights.run_rows(input, scratch, rows, &mut workspace.dense_gemm, stream)
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
                            &mut workspace.dense_gemm,
                            stream,
                        )
                    })
                    .map_err(nemotron3_cache_error)
            }
            _ => Err(Error::Format {
                label: "Nemotron 3 block layer workspace",
                detail: format!("workspace variant mismatch at layer {layer}"),
            }),
        }?;
        workspace.hidden.copy_range_from_device_on_stream(
            0,
            workspace.layers[slot].output(),
            0,
            rows * self.manifest.hidden_size,
            stream,
        )?;
        if let (Some(draft), Some(captured)) = (&self.lilicorr, &mut workspace.captured_features)
            && let Some(tap) = draft
                .config()
                .target_layers()
                .iter()
                .position(|&id| id == layer)
        {
            eider_cuda::lilicorr::capture(
                &workspace.hidden,
                captured,
                rows,
                self.manifest.hidden_size,
                draft.config().target_layers().len(),
                tap,
                stream,
            )?;
        }
        Ok(())
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
        if state.dense_gemm.is_some() && !state.decode_warm {
            self.enqueue_decode_layers(
                state,
                cache,
                reservation,
                page_table,
                0,
                self.layers.len(),
            )?;
            self.enqueue_decode_head(state)?;
            state.decode_warm = true;
            return Ok(());
        }
        let mut start = 0;
        while start < self.layers.len() {
            if matches!(self.layers[start], Nemotron3Layer::Attention(_)) {
                self.enqueue_decode_layers(
                    state,
                    cache,
                    reservation,
                    page_table,
                    start,
                    start + 1,
                )?;
                start += 1;
                continue;
            }
            let end = (start..self.layers.len())
                .find(|&layer| matches!(self.layers[layer], Nemotron3Layer::Attention(_)))
                .unwrap_or(self.layers.len());
            if let Some(graph) = &state.decode_graphs[start] {
                graph.launch(&self.stream)?;
            } else {
                let graph = self.stream.capture(|_| {
                    self.enqueue_decode_layers(state, cache, reservation, page_table, start, end)
                })?;
                graph.launch(&self.stream)?;
                state.decode_graphs[start] = Some(graph);
            }
            start = end;
        }
        let index = self.layers.len();
        if let Some(graph) = &state.decode_graphs[index] {
            return graph.launch(&self.stream);
        }
        let graph = self.stream.capture(|_| self.enqueue_decode_head(state))?;
        graph.launch(&self.stream)?;
        state.decode_graphs[index] = Some(graph);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_decode_layers(
        &self,
        state: &mut Nemotron3DecodeState,
        cache: &mut Nemotron3SequenceCache,
        reservation: &seqcache::AppendReservation,
        page_table: &DeviceBuffer<u32>,
        start: usize,
        end: usize,
    ) -> Result<()> {
        for layer in start..end {
            let (previous, current) = state.layers.split_at_mut(layer);
            let input = if layer == 0 {
                &state.hidden
            } else {
                previous[layer - 1].output()
            };
            match (&self.layers[layer], &mut current[0]) {
                (
                    Nemotron3Layer::Mamba(weights),
                    Nemotron3LayerState::Mamba {
                        workspace,
                        state: recurrent,
                    },
                ) => {
                    weights.run_one_token(
                        input,
                        workspace,
                        recurrent,
                        state.dense_gemm.as_mut(),
                        &self.stream,
                    )?;
                }
                (Nemotron3Layer::Moe(weights), Nemotron3LayerState::Moe(workspace)) => {
                    weights.run_one_token(
                        input,
                        workspace,
                        state.dense_gemm.as_mut(),
                        &self.stream,
                    )?;
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
                                state.dense_gemm.as_mut(),
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
            if let (Some(draft), Some(captured)) = (&self.lilicorr, &mut state.captured_features)
                && let Some(tap) = draft
                    .config()
                    .target_layers()
                    .iter()
                    .position(|&id| id == layer)
            {
                eider_cuda::lilicorr::capture(
                    current[0].output(),
                    captured,
                    1,
                    self.manifest.hidden_size,
                    draft.config().target_layers().len(),
                    tap,
                    &self.stream,
                )?;
            }
        }
        Ok(())
    }

    fn enqueue_decode_head(&self, state: &mut Nemotron3DecodeState) -> Result<()> {
        let last = state
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 model",
                detail: "model has no layers".to_string(),
            })?
            .output();
        norm(
            self.compute_precision,
            1,
            self.manifest.hidden_size,
            last,
            &self.final_norm,
            state.final_hidden.output(),
            self.manifest.norm_epsilon,
            &self.stream,
        )?;
        self.lm_head.run(
            &state.final_hidden,
            &mut state.logits,
            state.dense_gemm.as_mut(),
            &self.stream,
        )?;
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
                .lilicorr
                .as_ref()
                .map_or(0, LiLiCorrDraft::device_bytes)
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Nemotron3RowsShape {
    Mamba,
    Moe(Nemotron3MoeLayerConfig, bool, bool),
    Attention,
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
    draft_count: usize,
    accepted_drafts: Vec<u32>,
    next_tokens: Vec<u32>,
    drafted_tokens: Vec<u32>,
}

impl Nemotron3SpeculativeCycleResult {
    /// Returns the accepted draft-prefix length for each sequence.
    pub fn accepted_counts(&self) -> &[u32] {
        &self.accepted_drafts
    }

    /// Returns draft tokens in sequence-major order.
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
        let begin = sequence * self.draft_count;
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
    ssm: DeviceBuffer<u8>,
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
    graphs: Vec<Option<CudaGraphExec>>,
    graph_warm: bool,
    capture_graphs: bool,
    segment_dense_rows: bool,
    dense_row_lengths: Vec<u32>,
    dense_gemm: F32DenseGemm,
    host_tokens: PinnedHostBuffer<u32>,
    host_offsets: PinnedHostBuffer<u32>,
    host_lengths: PinnedHostBuffer<u32>,
    host_starts: PinnedHostBuffer<u32>,
    host_conv: PinnedHostBuffer<DeviceAddress<u16>>,
    host_ssm: PinnedHostBuffer<DeviceAddress<u8>>,
    host_pages: PinnedHostBuffer<DeviceAddress<u32>>,
    upload_done: CudaEvent,
    upload_pending: bool,
    tokens: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    captured_features: Option<DeviceBuffer<f32>>,
    layers: Vec<Nemotron3LayerRowsWorkspace>,
    layer_slots: Vec<usize>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    row_argmax: DeviceBuffer<u32>,
    row_maximum: DeviceBuffer<f32>,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    start_positions: DeviceBuffer<u32>,
    conv_state_table: DeviceBuffer<DeviceAddress<u16>>,
    ssm_state_table: DeviceBuffer<DeviceAddress<u8>>,
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
    last_logits_only: bool,
}

impl Drop for Nemotron3BlockWorkspace {
    fn drop(&mut self) {
        if self.upload_pending {
            let _ = self.upload_done.synchronize();
        }
    }
}

impl Nemotron3BlockWorkspace {
    fn new(model: &Nemotron3Model, sequence_count: usize, rows: usize) -> Result<Self> {
        Self::new_impl(model, sequence_count, rows, None, false)
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
            false,
        )
    }

    fn new_impl(
        model: &Nemotron3Model,
        sequence_count: usize,
        rows: usize,
        draft_count: Option<usize>,
        last_logits_only: bool,
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
        // Row temporaries are reused by shape. Only recurrent and KV state is
        // layer-specific; hidden carries each residual to the next layer.
        let precision = model.compute_precision;
        let mut layers = Vec::new();
        let mut shapes = Vec::new();
        let mut layer_slots = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let shape = match layer {
                Nemotron3Layer::Mamba(_) => Nemotron3RowsShape::Mamba,
                Nemotron3Layer::Moe(layer) => Nemotron3RowsShape::Moe(
                    layer.config(),
                    layer.has_calibrated_experts(),
                    layer.has_ggml_experts(),
                ),
                Nemotron3Layer::Attention(_) => Nemotron3RowsShape::Attention,
            };
            if let Some(slot) = shapes.iter().position(|existing| *existing == shape) {
                layer_slots.push(slot);
                continue;
            }
            layer_slots.push(layers.len());
            shapes.push(shape);
            layers.push(match layer {
                Nemotron3Layer::Mamba(layer) => {
                    Nemotron3LayerRowsWorkspace::Mamba(Box::new(layer.rows_workspace(rows)?))
                }
                Nemotron3Layer::Moe(layer) => Nemotron3LayerRowsWorkspace::Moe(Box::new(
                    layer.rows_workspace(rows, precision)?,
                )),
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
                                sequence_count
                                    * snapshot_slots
                                    * ssm_state_size
                                    * model.manifest.mamba_ssm_dtype.element_bytes(),
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let mut dense_weights = Vec::new();
        for layer in &model.layers {
            match layer {
                Nemotron3Layer::Mamba(layer) => dense_weights.extend(layer.dense_linears()),
                Nemotron3Layer::Moe(layer) => dense_weights.extend(layer.dense_linears()),
                Nemotron3Layer::Attention(layer) => dense_weights.extend(layer.dense_linears()),
            }
        }
        if !last_logits_only {
            dense_weights.push(&model.lm_head);
        }
        let mut dense_gemm =
            Nemotron3Linear::dense_workspace(rows, precision, dense_weights.into_iter())?;
        dense_gemm.set_precision(precision);
        let workspace = Self {
            graphs: (0..=model.layers.len()).map(|_| None).collect(),
            graph_warm: false,
            capture_graphs: draft_count.is_some(),
            segment_dense_rows: false,
            dense_row_lengths: Vec::new(),
            dense_gemm,
            host_tokens: PinnedHostBuffer::zeroed(rows)?,
            host_offsets: PinnedHostBuffer::zeroed(sequence_count)?,
            host_lengths: PinnedHostBuffer::zeroed(sequence_count)?,
            host_starts: PinnedHostBuffer::zeroed(sequence_count)?,
            host_conv: PinnedHostBuffer::zeroed((mamba_layers * sequence_count).max(1))?,
            host_ssm: PinnedHostBuffer::zeroed((mamba_layers * sequence_count).max(1))?,
            host_pages: PinnedHostBuffer::zeroed(sequence_count)?,
            upload_done: CudaEvent::new_sync()?,
            upload_pending: false,
            tokens: DeviceBuffer::zeroed(rows)?,
            hidden: DeviceBuffer::zeroed(rows * model.manifest.hidden_size)?,
            captured_features: model
                .lilicorr
                .as_ref()
                .map(|d| {
                    DeviceBuffer::zeroed(
                        rows * model.manifest.hidden_size * d.config().target_layers().len(),
                    )
                })
                .transpose()?,
            layers,
            layer_slots,
            final_hidden: DeviceBuffer::zeroed(rows * model.manifest.hidden_size)?,
            row_argmax: DeviceBuffer::zeroed(rows)?,
            row_maximum: DeviceBuffer::zeroed(rows)?,
            logits: DeviceBuffer::zeroed(if last_logits_only {
                0
            } else {
                rows * model.manifest.vocab_size
            })?,
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
            last_logits_only,
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
            && self.layer_slots.len() == model.layers.len()
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
                self.layer_slots.len()
            ),
        })
    }

    /// Returns all row logits in flattened sequence order, or an empty buffer
    /// for a prefill workspace. Prefill writes final logits to each sequence.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// Returns normalized final hidden states in flattened sequence order.
    pub fn final_hidden(&self) -> &DeviceBuffer<f32> {
        &self.final_hidden
    }

    /// Returns bytes owned by block scratch and pointer tables.
    pub fn device_bytes(&self) -> usize {
        self.dense_gemm.device_bytes()
            + self
                .captured_features
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes)
            + self.tokens.device_bytes()
            + self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3LayerRowsWorkspace::device_bytes)
                .sum::<usize>()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.row_argmax.device_bytes()
            + self.row_maximum.device_bytes()
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
    lilicorr: Option<LiLiCorrSnapshot>,
    tokens: usize,
}

impl Nemotron3SequenceSnapshot {
    /// Returns bytes retained by the compact snapshot.
    pub fn device_bytes(&self) -> usize {
        self.final_hidden.device_bytes()
            + self
                .lilicorr
                .as_ref()
                .map_or(0, LiLiCorrSnapshot::device_bytes)
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
    // Graphs contain addresses owned by this sequence and the resident model.
    decode_graphs: Vec<Option<CudaGraphExec>>,
    decode_warm: bool,
    dense_gemm: Option<F32DenseGemm>,
    hidden: DeviceBuffer<f32>,
    layers: Vec<Nemotron3LayerState>,
    rollback_mamba: Vec<Option<Nemotron3MambaState>>,
    rollback_tokens: usize,
    append_pending: bool,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_token: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    lilicorr: Option<LiLiCorrSequence>,
    captured_features: Option<DeviceBuffer<f32>>,
    mtp: Option<Nemotron3MtpState>,
    compact_attention: Option<Sm12xKvAttentionWorkspace>,
    tokens: usize,
    pub(crate) max_tokens: usize,
}

impl Nemotron3DecodeState {
    /// Maximum snapshot size, including full companion and MTP context storage.
    pub(crate) fn snapshot_capacity_bytes(&self) -> usize {
        self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self
                .layers
                .iter()
                .map(|layer| match layer {
                    Nemotron3LayerState::Mamba { state, .. } => state.device_bytes(),
                    _ => 0,
                })
                .sum::<usize>()
            + self
                .lilicorr
                .as_ref()
                .map_or(0, LiLiCorrSequence::device_bytes)
            + self.mtp.as_ref().map_or(0, Nemotron3MtpState::device_bytes)
    }

    /// Returns the number of tokens already processed by the backbone.
    pub fn len(&self) -> usize {
        self.tokens
    }

    /// Returns bytes owned by this sequence's device-resident state and scratch.
    pub fn device_bytes(&self) -> usize {
        self.hidden.device_bytes()
            + self
                .dense_gemm
                .as_ref()
                .map_or(0, F32DenseGemm::device_bytes)
            + self
                .lilicorr
                .as_ref()
                .map_or(0, LiLiCorrSequence::device_bytes)
            + self
                .captured_features
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes)
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

#[cfg(test)]
mod lilicorr_tests {
    use super::*;
    use crate::nemotron3::new_nemotron3_sequence_cache;
    use eider_runtime::sampling::SamplingConfig;

    #[test]
    #[ignore = "loads the full target and LiLiCorr; requires NEMOTRON_MODEL_DIR and LILICORR_DIR"]
    fn lilicorr_commits_match_serial_target() {
        check_lilicorr_commits(Nemotron3StorageConfig::default(), 1, false);
    }

    #[test]
    #[ignore = "loads the full target and LiLiCorr; requires NEMOTRON_MODEL_DIR and LILICORR_DIR"]
    fn lilicorr_bf16_commits_match_serial_target() {
        check_lilicorr_commits(
            Nemotron3StorageConfig {
                compute_precision: eider_cuda::GemmPrecision::Bf16,
                mamba_ssm_dtype: Some(eider_cuda::MambaSsmDtype::Bf16),
                ..Nemotron3StorageConfig::default()
            },
            1,
            false,
        );
    }

    #[test]
    #[ignore = "loads the full target and LiLiCorr; requires NEMOTRON_MODEL_DIR and LILICORR_DIR"]
    fn lilicorr_bf16_matrices_f32_state_commits_match_serial_target() {
        check_lilicorr_commits(
            Nemotron3StorageConfig {
                compute_precision: eider_cuda::GemmPrecision::Bf16,
                ..Nemotron3StorageConfig::default()
            },
            1,
            false,
        );
    }

    #[test]
    #[ignore = "loads the full target and LiLiCorr; requires NEMOTRON_MODEL_DIR and LILICORR_DIR"]
    fn lilicorr_bf16_ssd_prefill_commits_match_serial_target() {
        check_lilicorr_commits(
            Nemotron3StorageConfig {
                compute_precision: eider_cuda::GemmPrecision::Bf16,
                mamba_ssm_dtype: Some(eider_cuda::MambaSsmDtype::Bf16),
                ..Nemotron3StorageConfig::default()
            },
            3,
            false,
        );
    }

    #[test]
    #[ignore = "loads the full target and LiLiCorr; requires NEMOTRON_MODEL_DIR and LILICORR_DIR"]
    fn lilicorr_bf16_trailing_space_commits_match_serial_target() {
        check_lilicorr_commits(
            Nemotron3StorageConfig {
                compute_precision: eider_cuda::GemmPrecision::Bf16,
                mamba_ssm_dtype: Some(eider_cuda::MambaSsmDtype::Bf16),
                ..Nemotron3StorageConfig::default()
            },
            1,
            true,
        );
    }

    fn check_lilicorr_commits(
        storage: Nemotron3StorageConfig,
        prompt_repetitions: usize,
        trailing_space: bool,
    ) {
        let root = std::env::var("NEMOTRON_MODEL_DIR").unwrap();
        let mut model = Nemotron3Model::load_with_storage(&root, storage).unwrap();
        model
            .enable_lilicorr(std::env::var("LILICORR_DIR").unwrap())
            .unwrap();
        let tokenizer =
            tokenizers::Tokenizer::from_file(Path::new(&root).join("tokenizer.json")).unwrap();
        let prompt = tokenizer.encode("The capital of Canada is Ottawa. Explain briefly why Ottawa was selected as the capital, and name the province in which it is located.".repeat(prompt_repetitions) + if trailing_space { " " } else { "" },true).unwrap();
        let tokens = prompt.get_ids();
        let mut cache = new_nemotron3_sequence_cache(&model, 2, 1024).unwrap();
        let mut serial = Nemotron3Sequence::admit(&model, &mut cache, 1024).unwrap();
        let mut spec = Nemotron3Sequence::admit(&model, &mut cache, 1024).unwrap();
        let mut prefill = model.prefill_workspace(1, tokens.len()).unwrap();
        let mut draft = model.lilicorr().unwrap().new_workspace().unwrap();
        for sequence in [&mut serial, &mut spec] {
            model
                .forward_block(&mut [sequence], &[tokens], &mut prefill, &mut cache)
                .unwrap();
            model
                .append_lilicorr_block_context(sequence, &prefill, 0, tokens.len(), &mut draft)
                .unwrap();
        }
        let mut expected = vec![model.argmax(&mut serial).unwrap()];
        for _ in 1..64 {
            model
                .forward_one(&mut serial, &mut cache, *expected.last().unwrap())
                .unwrap();
            expected.push(model.argmax(&mut serial).unwrap());
        }
        let mut actual = vec![model.argmax(&mut spec).unwrap()];
        let mut history =
            TokenHistory::from_tokens(tokens.iter().copied().chain(actual.iter().copied()));
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.,
            ..SamplingConfig::default()
        })
        .unwrap();
        let mut workspaces = std::collections::BTreeMap::new();
        let mut accepted = 0;
        let mut cycles = 0;
        while actual.len() < expected.len() {
            let remaining = expected.len() - actual.len();
            if remaining == 1 {
                model
                    .forward_one(&mut spec, &mut cache, *actual.last().unwrap())
                    .unwrap();
                model
                    .append_lilicorr_decode_context(&mut spec, &mut draft)
                    .unwrap();
                actual.push(model.argmax(&mut spec).unwrap());
                break;
            }
            // Exercise shortened proposals as well as the released seven-token block.
            let drafts = if cycles % 3 == 2 { 3 } else { 7 }.min(remaining - 1);
            if let std::collections::btree_map::Entry::Vacant(entry) = workspaces.entry(drafts) {
                entry.insert(model.speculative_workspace(1, drafts + 1).unwrap());
            }
            let result = model
                .lilicorr_cycle(
                    &mut spec,
                    *actual.last().unwrap(),
                    drafts,
                    &mut draft,
                    workspaces.get_mut(&drafts).unwrap(),
                    &mut sampler,
                    &history,
                    &mut cache,
                )
                .unwrap();
            accepted += result.accepted_counts()[0];
            cycles += 1;
            for token in result.emitted_tokens(0).unwrap() {
                history.push(token);
                actual.push(token);
            }
            assert_eq!(spec.position(), tokens.len() + actual.len() - 1);
            assert_eq!(
                spec.state.lilicorr.as_ref().unwrap().position(),
                spec.position()
            );
            assert_eq!(
                actual,
                expected[..actual.len()],
                "diverged after cycle {cycles}, accepted {accepted}"
            );
        }
        assert_eq!(actual, expected);
        assert!(accepted > 0, "real checkpoint should accept some drafts");
        spec.finish(&model, &mut cache).unwrap();
        serial.finish(&model, &mut cache).unwrap();
    }

    #[test]
    #[ignore = "loads the full target and LiLiCorr; requires NEMOTRON_MODEL_DIR, LILICORR_DIR, and LILICORR_CONTEXT_OUT"]
    fn capture_lilicorr_reference_context() {
        let root = std::env::var("NEMOTRON_MODEL_DIR").unwrap();
        let mut model = Nemotron3Model::load(&root).unwrap();
        model
            .enable_lilicorr(std::env::var("LILICORR_DIR").unwrap())
            .unwrap();
        let tokenizer =
            tokenizers::Tokenizer::from_file(Path::new(&root).join("tokenizer.json")).unwrap();
        let prompt = tokenizer.encode("The capital of Canada is Ottawa. Explain briefly why Ottawa was selected as the capital, and name the province in which it is located.",true).unwrap();
        let tokens = prompt.get_ids();
        let mut cache = new_nemotron3_sequence_cache(&model, 1, 1024).unwrap();
        let mut sequence = Nemotron3Sequence::admit(&model, &mut cache, 1024).unwrap();
        let mut block = model.prefill_workspace(1, tokens.len()).unwrap();
        model
            .forward_block(&mut [&mut sequence], &[tokens], &mut block, &mut cache)
            .unwrap();
        let features = block
            .captured_features
            .as_ref()
            .unwrap()
            .copy_to_host(&model.stream)
            .unwrap()
            .into_vec();
        let anchor = model.argmax(&mut sequence).unwrap();
        let fixture =
            serde_json::json!({"rows": tokens.len(), "features": features, "anchor_token": anchor});
        std::fs::write(
            std::env::var("LILICORR_CONTEXT_OUT").unwrap(),
            serde_json::to_vec(&fixture).unwrap(),
        )
        .unwrap();
        sequence.finish(&model, &mut cache).unwrap();
    }
}
