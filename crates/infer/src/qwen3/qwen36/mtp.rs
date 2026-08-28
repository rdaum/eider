//! Qwen3.8 MTP (multi-token prediction) drafter weights and draft execution.
//!
//! The dense Qwen3.5-family checkpoint ships a BF16 MTP block beside the text
//! stack, kept outside the NVFP4 quantization groups: two pre-projection
//! RMSNorms, a fused `[embedding | hidden]` projection, one full-attention
//! decoder layer with a dense SwiGLU feed-forward, and a final norm. The
//! drafter shares the target's embedding table and LM head, and keeps a
//! private single-layer compact K/V cache per sequence.
//!
//! The block semantics mirror the Qwen3-Next MTP family:
//!
//! - the projection input is
//!   `concat([pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)])`,
//! - the layer is a standard gated full-attention decoder layer with residual
//!   adds (`h = y + attention(norm_in(y)); out = h + mlp(norm_post(h))`),
//! - the block output passes through `mtp.norm` before the shared LM head, and
//!   the same post-norm vector chains into the next draft step.
//!
//! Draft quality only affects speculative acceptance rates; the target model
//! decides which tokens commit. The MTP attention Q/K/V/O weights remain BF16,
//! while its private fusion and feed-forward projections use NVFP4 because
//! they propose tokens only. The drafter reuses the target LM head.

use super::{
    Nvfp4DeviceLinear, Qwen36Fp8Nvfp4Cache, Qwen36FullAttentionState, Qwen36FullAttentionWeights,
    Qwen36FullAttentionWorkspace, Qwen36LmHeadWorkspace, Qwen36TextModel, read_bf16_matrix_host,
    read_bf16_vector_delta_as_f32_device,
};
use crate::qwen3::infer::QwenModelManifest;
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, Result, add_f32_into_on_stream,
    concat_f32_rows_into_on_stream, rms_norm_f32_into_on_stream,
    round_f32_to_bf16_in_place_on_stream, silu_mul_halves_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;

/// Device-ready weights for the Qwen3.8 MTP drafter block.
pub struct Qwen36MtpWeights {
    pre_fc_norm_embedding: DeviceBuffer<f32>,
    pre_fc_norm_hidden: DeviceBuffer<f32>,
    fc: Nvfp4DeviceLinear,
    input_norm: DeviceBuffer<f32>,
    attention: Qwen36FullAttentionWeights,
    post_attn_norm: DeviceBuffer<f32>,
    gate_up: Nvfp4DeviceLinear,
    down: Nvfp4DeviceLinear,
    final_norm: DeviceBuffer<f32>,
}

/// Persistent MTP drafter state owned by one generated sequence.
pub struct Qwen36MtpSequenceState {
    attention: Qwen36FullAttentionState,
}

fn load_nvfp4_bf16_linear(
    checkpoint: &ModelOptCheckpoint,
    tensor: &str,
    prefix: &str,
    rows: usize,
    cols: usize,
) -> Result<Nvfp4DeviceLinear> {
    let host = read_bf16_matrix_host(checkpoint, tensor, rows, cols)?;
    Nvfp4DeviceLinear::from_bf16_host(prefix, &host, rows, cols)
}

impl Qwen36MtpSequenceState {
    /// Returns the number of K/V rows currently owned by the drafter cache.
    pub fn len(&self) -> usize {
        self.attention.compact_cache.len()
    }

    /// Returns whether the drafter cache holds no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops drafter K/V rows beyond `len` so rejected draft rows can be
    /// rewritten by the next catch-up or draft pass.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.attention.compact_cache.truncate(len)
    }
}

/// Reusable single-token scratch for the MTP drafter block.
pub struct Qwen36MtpDraftWorkspace {
    tokens: DeviceBuffer<u32>,
    draft_tokens: DeviceBuffer<u32>,
    embedded: DeviceBuffer<f32>,
    normed_embedding: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    fused: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    pre_attention: DeviceBuffer<f32>,
    attention: Qwen36FullAttentionWorkspace,
    attention_residual: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    block: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    lm_head: Qwen36LmHeadWorkspace,
}

impl Qwen36MtpDraftWorkspace {
    /// Returns the exact device bytes owned by the drafter workspace.
    pub fn device_bytes(&self) -> usize {
        self.tokens.device_bytes()
            + self.draft_tokens.device_bytes()
            + self.embedded.device_bytes()
            + self.normed_embedding.device_bytes()
            + self.normed_hidden.device_bytes()
            + self.fused.device_bytes()
            + self.projected.device_bytes()
            + self.pre_attention.device_bytes()
            + self.attention.device_bytes()
            + self.attention_residual.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.down.device_bytes()
            + self.block.device_bytes()
            + self.final_hidden.device_bytes()
            + self.lm_head.device_bytes()
    }
}

impl Qwen36MtpWeights {
    /// Loads the BF16 MTP block from the checkpoint's `mtp.*` tensors.
    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &QwenModelManifest,
        fp8_nvfp4_cache: &Qwen36Fp8Nvfp4Cache,
    ) -> Result<Self> {
        let hidden = manifest.hidden;
        let intermediate = manifest.intermediate;
        let layer = "mtp.layers.0";
        let prefix = format!("{layer}.self_attn");
        let attention = Qwen36FullAttentionWeights {
            q: super::Qwen36Linear::load(
                checkpoint,
                &format!("{prefix}.q_proj"),
                manifest.q_heads * manifest.head_dim * 2,
                hidden,
                super::Qwen36Bf16Storage::Bf16,
                super::Qwen36Fp8Storage::Fp8,
                fp8_nvfp4_cache,
            )?,
            k: super::Qwen36Linear::load(
                checkpoint,
                &format!("{prefix}.k_proj"),
                manifest.kv_heads * manifest.head_dim,
                hidden,
                super::Qwen36Bf16Storage::Bf16,
                super::Qwen36Fp8Storage::Fp8,
                fp8_nvfp4_cache,
            )?,
            v: super::Qwen36Linear::load(
                checkpoint,
                &format!("{prefix}.v_proj"),
                manifest.kv_heads * manifest.head_dim,
                hidden,
                super::Qwen36Bf16Storage::Bf16,
                super::Qwen36Fp8Storage::Fp8,
                fp8_nvfp4_cache,
            )?,
            o: super::Qwen36Linear::load(
                checkpoint,
                &format!("{prefix}.o_proj"),
                hidden,
                manifest.q_heads * manifest.head_dim,
                super::Qwen36Bf16Storage::Bf16,
                super::Qwen36Fp8Storage::Fp8,
                fp8_nvfp4_cache,
            )?,
            q_norm_weight: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{prefix}.q_norm.weight"),
                manifest.head_dim,
            )?,
            k_norm_weight: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{prefix}.k_norm.weight"),
                manifest.head_dim,
            )?,
        };
        // The dense SwiGLU halves concatenate so the fused silu-mul kernel sees
        // [gate | up], exactly like the NVFP4 dense feed-forward path.
        let gate_up = {
            let mut gate_up_rows = read_bf16_matrix_host(
                checkpoint,
                &format!("{layer}.mlp.gate_proj.weight"),
                intermediate,
                hidden,
            )?;
            let up_rows = read_bf16_matrix_host(
                checkpoint,
                &format!("{layer}.mlp.up_proj.weight"),
                intermediate,
                hidden,
            )?;
            gate_up_rows.extend_from_slice(&up_rows);
            Nvfp4DeviceLinear::from_bf16_host(
                &format!("{layer}.mlp.gate_up_proj"),
                &gate_up_rows,
                2 * intermediate,
                hidden,
            )?
        };
        let down = load_nvfp4_bf16_linear(
            checkpoint,
            &format!("{layer}.mlp.down_proj.weight"),
            &format!("{layer}.mlp.down_proj"),
            hidden,
            intermediate,
        )?;
        Ok(Self {
            pre_fc_norm_embedding: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                "mtp.pre_fc_norm_embedding.weight",
                hidden,
            )?,
            pre_fc_norm_hidden: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                "mtp.pre_fc_norm_hidden.weight",
                hidden,
            )?,
            fc: load_nvfp4_bf16_linear(checkpoint, "mtp.fc.weight", "mtp.fc", hidden, 2 * hidden)?,
            input_norm: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{layer}.input_layernorm.weight"),
                hidden,
            )?,
            attention,
            post_attn_norm: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{layer}.post_attention_layernorm.weight"),
                hidden,
            )?,
            gate_up,
            down,
            final_norm: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                "mtp.norm.weight",
                hidden,
            )?,
        })
    }

    /// Projects the `[embedding | hidden]` pair into the block input.
    ///
    /// The gathered embedding embedding row comes from `workspace.tokens`.
    fn prepare_input(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut Qwen36MtpDraftWorkspace,
        prev_hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let manifest = &model.manifest;
        let hidden = manifest.hidden;
        model.embedding.gather_prefix(
            manifest.vocab,
            hidden,
            &workspace.tokens,
            workspace.embedded.output(),
            1,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            hidden,
            &workspace.embedded,
            &self.pre_fc_norm_embedding,
            workspace.normed_embedding.output(),
            manifest.rms_eps,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            hidden,
            prev_hidden,
            &self.pre_fc_norm_hidden,
            workspace.normed_hidden.output(),
            manifest.rms_eps,
            stream,
        )?;
        concat_f32_rows_into_on_stream(
            1,
            hidden,
            &workspace.normed_embedding,
            &workspace.normed_hidden,
            workspace.fused.output(),
            stream,
        )?;
        self.fc
            .run_f32_into(&workspace.fused, &mut workspace.projected, stream)
    }

    /// Computes this token's attention K/V and appends them to the drafter
    /// cache at `position`, without running attention or the feed-forward.
    fn append_kv(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut Qwen36MtpDraftWorkspace,
        state: &mut Qwen36MtpSequenceState,
        prev_hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let position = state.attention.compact_cache.len();
        self.prepare_input(model, workspace, prev_hidden, stream)?;
        rms_norm_f32_into_on_stream(
            1,
            model.manifest.hidden,
            &workspace.projected,
            &self.input_norm,
            workspace.pre_attention.output(),
            model.manifest.rms_eps,
            stream,
        )?;
        self.attention.prepare_qkv_one_token(
            &mut workspace.attention,
            &state.attention,
            &model.manifest,
            &workspace.pre_attention,
            position,
            stream,
        )?;
        state.attention.compact_cache.append_at_on_stream(
            &workspace.attention.k_rope,
            &workspace.attention.v,
            position,
            stream,
        )
    }

    /// Runs the full drafter block for one token.
    ///
    /// Appends the token's K/V at the current cache position and leaves the
    /// post-`mtp.norm` hidden in `workspace.final_hidden`, which chains into
    /// the next draft step.
    fn draft_step(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut Qwen36MtpDraftWorkspace,
        state: &mut Qwen36MtpSequenceState,
        prev_hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let manifest = &model.manifest;
        let position = state.attention.compact_cache.len();
        self.prepare_input(model, workspace, prev_hidden, stream)?;
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            &workspace.projected,
            &self.input_norm,
            workspace.pre_attention.output(),
            manifest.rms_eps,
            stream,
        )?;
        {
            let step = self.attention.run_one_token(
                &mut workspace.attention,
                &mut state.attention,
                manifest,
                &workspace.pre_attention,
                position,
                stream,
            )?;
            add_f32_into_on_stream(
                &workspace.projected,
                step.output,
                workspace.attention_residual.output(),
                stream,
            )?;
        }
        round_f32_to_bf16_in_place_on_stream(workspace.attention_residual.inout(), stream)?;
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            &workspace.attention_residual,
            &self.post_attn_norm,
            workspace.ffn_norm.output(),
            manifest.rms_eps,
            stream,
        )?;
        self.gate_up
            .run_f32_into(&workspace.ffn_norm, &mut workspace.gate_up, stream)?;
        silu_mul_halves_f32_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            self.down.in_features,
            stream,
        )?;
        self.down
            .run_f32_into(&workspace.activated, &mut workspace.down, stream)?;
        add_f32_into_on_stream(
            &workspace.attention_residual,
            &workspace.down,
            workspace.block.output(),
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.block.inout(), stream)?;
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            &workspace.block,
            &self.final_norm,
            workspace.final_hidden.output(),
            manifest.rms_eps,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.final_hidden.inout(), stream)
    }
}

impl Qwen36TextModel {
    /// Returns the loaded MTP drafter weights, when the checkpoint ships them.
    pub fn mtp_weights(&self) -> Option<&Qwen36MtpWeights> {
        self.mtp.as_ref()
    }

    /// Allocates the per-sequence MTP drafter cache.
    pub fn new_mtp_sequence_state(&self, max_tokens: usize) -> Result<Qwen36MtpSequenceState> {
        if self.mtp.is_none() {
            return Err(Error::Format {
                label: "Qwen3.8 MTP sequence state",
                detail: "model has no MTP weights".to_string(),
            });
        }
        Ok(Qwen36MtpSequenceState {
            attention: Qwen36FullAttentionState::new(&self.manifest, max_tokens)?,
        })
    }

    /// Allocates the shared-scratch MTP drafter workspace.
    pub fn new_mtp_draft_workspace(&self, max_tokens: usize) -> Result<Qwen36MtpDraftWorkspace> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 MTP drafter workspace",
            detail: "model has no MTP weights".to_string(),
        })?;
        Ok(Qwen36MtpDraftWorkspace {
            tokens: DeviceBuffer::zeroed(1)?,
            draft_tokens: DeviceBuffer::zeroed(max_tokens)?,
            embedded: DeviceBuffer::zeroed(self.manifest.hidden)?,
            normed_embedding: DeviceBuffer::zeroed(self.manifest.hidden)?,
            normed_hidden: DeviceBuffer::zeroed(self.manifest.hidden)?,
            fused: DeviceBuffer::zeroed(2 * self.manifest.hidden)?,
            projected: DeviceBuffer::zeroed(self.manifest.hidden)?,
            pre_attention: DeviceBuffer::zeroed(self.manifest.hidden)?,
            attention: Qwen36FullAttentionWorkspace::new(
                &self.manifest,
                &mtp.attention,
                max_tokens,
            )?,
            attention_residual: DeviceBuffer::zeroed(self.manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(self.manifest.hidden)?,
            gate_up: DeviceBuffer::zeroed(2 * self.manifest.intermediate)?,
            activated: DeviceBuffer::zeroed(self.manifest.intermediate)?,
            down: DeviceBuffer::zeroed(self.manifest.hidden)?,
            block: DeviceBuffer::zeroed(self.manifest.hidden)?,
            final_hidden: DeviceBuffer::zeroed(self.manifest.hidden)?,
            lm_head: Qwen36LmHeadWorkspace {
                logits: DeviceBuffer::zeroed(self.manifest.vocab)?,
                dynamic_input: DeviceBuffer::zeroed(self.manifest.hidden)?,
                dynamic_input_scale: DeviceBuffer::zeroed(1)?,
                scratch_value: DeviceBuffer::zeroed(self.manifest.vocab.div_ceil(8))?,
                scratch_index: DeviceBuffer::zeroed(self.manifest.vocab.div_ceil(8))?,
                next_index: DeviceBuffer::zeroed(1)?,
                next_value: DeviceBuffer::zeroed(1)?,
            },
        })
    }

    /// Warms the drafter K/V cache over a prompt chunk.
    ///
    /// `tokens` are the sequence's token ids over rows `0..tokens.len()`, and
    /// `prev_hiddens` row `j` is the target's final hidden state for position
    /// `j` within the chunk, starting at element offset `hidden_offset`. Warmup
    /// pairs token j+1 with row j and token 0 with `initial_hidden`, so the
    /// buffer must hold at least `tokens.len() - 1` hidden rows after the
    /// offset. Rows append
    /// contiguously from the state's current cache length.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_warmup_kv(
        &self,
        state: &mut Qwen36MtpSequenceState,
        workspace: &mut Qwen36MtpDraftWorkspace,
        prev_row: &mut DeviceBuffer<f32>,
        tokens: &[u32],
        prev_hiddens: &DeviceBuffer<f32>,
        hidden_offset: usize,
        initial_hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 MTP warmup",
            detail: "model has no MTP weights".to_string(),
        })?;
        let hidden = self.manifest.hidden;
        let required = hidden_offset
            .checked_add(tokens.len().saturating_sub(1) * hidden)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 MTP warmup hiddens",
                expected: "offset + rows without overflow".to_string(),
                actual: format!("offset={hidden_offset} rows={}", tokens.len()),
            })?;
        if prev_hiddens.len() < required {
            return Err(Error::Shape {
                label: "Qwen3.8 MTP warmup hiddens",
                expected: format!(
                    "at least {} values after offset {hidden_offset}",
                    tokens.len().saturating_sub(1) * hidden
                ),
                actual: format!("{} values", prev_hiddens.len()),
            });
        }
        if prev_row.len() < hidden {
            return Err(Error::Shape {
                label: "Qwen3.8 MTP warmup scratch",
                expected: format!("at least {hidden} values"),
                actual: format!("{} values", prev_row.len()),
            });
        }
        for (row, &token) in tokens.iter().enumerate() {
            workspace
                .tokens
                .copy_from_host(std::slice::from_ref(&token))?;
            if row == 0 {
                prev_row.copy_prefix_from_device_on_stream(initial_hidden, hidden, stream)?;
            } else {
                prev_row.copy_range_from_device_on_stream(
                    0,
                    prev_hiddens,
                    hidden_offset + (row - 1) * hidden,
                    hidden,
                    stream,
                )?;
            }
            mtp.append_kv(self, workspace, state, prev_row, stream)?;
        }
        Ok(())
    }

    /// Appends one committed token's drafter K/V from its (token, previous
    /// target hidden) pair, filling the slot at the current cache length.
    ///
    /// The drafter K/V written by a draft chain is only valid while the
    /// drafted tokens remain unverified; catch-up must truncate rejected rows
    /// first and then append the committed tokens in order.
    pub fn mtp_append_kv(
        &self,
        state: &mut Qwen36MtpSequenceState,
        workspace: &mut Qwen36MtpDraftWorkspace,
        token: u32,
        prev_hidden: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 MTP catch-up",
            detail: "model has no MTP weights".to_string(),
        })?;
        workspace
            .tokens
            .copy_from_host(std::slice::from_ref(&token))?;
        mtp.append_kv(self, workspace, state, prev_hidden, stream)
    }

    /// Drafts `drafts` chained greedy tokens from the target frontier.
    ///
    /// `initial_token` is the token the target just sampled and
    /// `target_hidden` is the pre-final-norm hidden state that produced it.
    /// The chain advances the drafter K/V cache; rejected draft K/V rows are
    /// overwritten by the next catch-up pass, so no rollback is required
    /// between speculative cycles.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_draft_chain_argmax(
        &self,
        state: &mut Qwen36MtpSequenceState,
        workspace: &mut Qwen36MtpDraftWorkspace,
        chained: &mut DeviceBuffer<f32>,
        initial_token: u32,
        target_hidden: &DeviceBuffer<f32>,
        drafts: usize,
        stream: &CudaStream,
    ) -> Result<Vec<u32>> {
        let mtp = self.mtp.as_ref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 MTP draft",
            detail: "model has no MTP weights".to_string(),
        })?;
        if drafts == 0 {
            return Ok(Vec::new());
        }
        if chained.len() < self.manifest.hidden {
            return Err(Error::Shape {
                label: "Qwen3.8 MTP draft scratch",
                expected: format!("at least {} values", self.manifest.hidden),
                actual: format!("{} values", chained.len()),
            });
        }
        if workspace.draft_tokens.len() < drafts {
            return Err(Error::Shape {
                label: "Qwen3.8 MTP draft token scratch",
                expected: format!("at least {drafts} tokens"),
                actual: format!("{} tokens", workspace.draft_tokens.len()),
            });
        }
        workspace
            .tokens
            .copy_from_host(std::slice::from_ref(&initial_token))?;
        for step in 0..drafts {
            if step == 0 {
                mtp.draft_step(self, workspace, state, target_hidden, stream)?;
            } else {
                chained.copy_prefix_from_device_on_stream(
                    &workspace.final_hidden,
                    self.manifest.hidden,
                    stream,
                )?;
                workspace.tokens.copy_prefix_from_device_on_stream(
                    &workspace.lm_head.next_index,
                    1,
                    stream,
                )?;
                mtp.draft_step(self, workspace, state, chained, stream)?;
            }
            self.lm_head.run_top1(
                &self.lt,
                &workspace.final_hidden,
                &mut workspace.lm_head,
                stream,
            )?;
            workspace.draft_tokens.copy_range_from_device_on_stream(
                step,
                &workspace.lm_head.next_index,
                0,
                1,
                stream,
            )?;
        }
        Ok(workspace
            .draft_tokens
            .copy_prefix_to_host(drafts, stream)?
            .into_vec())
    }
}
