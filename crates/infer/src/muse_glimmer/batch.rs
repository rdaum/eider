use super::*;
use nvfp4::{
    Sm12xKvTailSnapshot, add_f32_prefix_into_on_stream, argmax_f32_batch_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, copy_f32_rows_into_columns_on_stream,
    rope_neox_sequence_f32_into_on_stream, round_f32_to_bf16_prefix_in_place_on_stream,
    sigmoid_mul_f32_prefix_into_on_stream, silu_mul_f32_prefix_into_on_stream,
};
use std::collections::HashMap;

pub(super) const DFLASH_BLOCK_SIZE: usize = 16;
const DFLASH_EXTRACT_COUNT: usize = 5;
const GEMM_WORKSPACE_LIMIT: u64 = 8 << 20;

pub(super) struct MuseBatchLinearWorkspace {
    capacity: usize,
    rows: usize,
    lt: CublasLt,
    activations: HashMap<usize, Nvfp4Matrix>,
    plans: HashMap<(usize, usize, usize), Fp4TnMatmulPlan>,
}

impl MuseBatchLinearWorkspace {
    pub(super) fn new(rows: usize) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Shape {
                label: "Muse batch rows",
                expected: "at least one row".to_string(),
                actual: "0".to_string(),
            });
        }
        Ok(Self {
            capacity: rows,
            rows,
            lt: CublasLt::new()?,
            activations: HashMap::new(),
            plans: HashMap::new(),
        })
    }

    pub(super) fn set_rows(&mut self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(Error::Shape {
                label: "Muse batch linear rows",
                expected: format!("1..={}", self.capacity),
                actual: rows.to_string(),
            });
        }
        self.rows = rows;
        for activation in self.activations.values_mut() {
            activation.cols = rows;
        }
        Ok(())
    }

    pub(super) fn run(
        &mut self,
        linear: &MuseNvfp4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let (out_features, in_features) = linear.shape();
        let input_len = self.rows * in_features;
        let output_len = self.rows * out_features;
        if input.len() < input_len || output.len() < output_len {
            return Err(Error::Shape {
                label: "Muse batch linear buffers",
                expected: format!("input >= {input_len} output >= {output_len}"),
                actual: format!("input={} output={}", input.len(), output.len()),
            });
        }
        if !self.activations.contains_key(&in_features) {
            let mut activation = Nvfp4Matrix::zeroed_col_major(in_features, self.capacity)?;
            activation.cols = self.rows;
            self.activations.insert(in_features, activation);
        }
        let key = (out_features, in_features, self.rows);
        if !self.plans.contains_key(&key) {
            let activation = self
                .activations
                .get(&in_features)
                .expect("batch activation exists");
            self.plans.insert(
                key,
                Fp4TnMatmulPlan::new_f32_output_for_shape(
                    &self.lt,
                    GemmShape::new(out_features, self.rows, in_features),
                    Nvfp4TnInputs::new(linear.weight.matrix(), activation),
                    GEMM_WORKSPACE_LIMIT,
                )?,
            );
        }
        let activation = self
            .activations
            .get_mut(&in_features)
            .expect("batch activation exists");
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            in_features,
            self.rows,
            input,
            activation,
            linear.weight.input_scale(),
            stream,
        )?;
        self.plans
            .get(&key)
            .expect("batch plan exists")
            .run_with_alpha_beta_f32_inout_buffer_on_stream(
                &self.lt,
                Nvfp4TnInputs::new(linear.weight.matrix(), activation),
                output.inout(),
                linear.weight.matmul_alpha(),
                0.0,
                stream,
            )
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.activations
            .values()
            .map(Nvfp4Matrix::device_bytes)
            .sum::<usize>()
            + self
                .plans
                .values()
                .map(Fp4TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
    }
}

pub(super) struct MuseTargetBatchWorkspace {
    tokens: DeviceBuffer<u32>,
    linear: MuseBatchLinearWorkspace,
    current: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_positioned: DeviceBuffer<f32>,
    k_positioned: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    feedforward_input: DeviceBuffer<f32>,
    mlp_gate: DeviceBuffer<f32>,
    mlp_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    mlp_output: DeviceBuffer<f32>,
    feedforward_output: DeviceBuffer<f32>,
    layer_output: DeviceBuffer<f32>,
    local_attention: Option<Sm12xKvAttentionWorkspace>,
    global_attention: Option<Sm12xKvAttentionWorkspace>,
    tail_snapshots: Vec<Sm12xKvTailSnapshot>,
    pub(super) features: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) argmax_indices: DeviceBuffer<u32>,
    pub(super) argmax_values: DeviceBuffer<f32>,
}

impl MuseTargetBatchWorkspace {
    pub(super) fn new(model: &MuseGlimmerModel, max_tokens: usize) -> Result<Self> {
        let rows = DFLASH_BLOCK_SIZE;
        let hidden = model.config.hidden_size;
        let q_width = model.config.num_attention_heads * model.config.head_dim;
        let kv_width = model.config.num_key_value_heads * model.config.head_dim;
        let intermediate = model.config.intermediate_size;
        let local = model
            .layers
            .iter()
            .find(|layer| layer.attention.window.is_some());
        let global = model
            .layers
            .iter()
            .find(|layer| layer.attention.window.is_none());
        let attention_capacity =
            max_tokens.div_ceil(nvfp4::SM12X_KV_PAGE_TOKENS) * nvfp4::SM12X_KV_PAGE_TOKENS;
        Ok(Self {
            tokens: DeviceBuffer::zeroed(rows)?,
            linear: MuseBatchLinearWorkspace::new(rows)?,
            current: DeviceBuffer::zeroed(rows * hidden)?,
            normalized: DeviceBuffer::zeroed(rows * hidden)?,
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            gate: DeviceBuffer::zeroed(rows * q_width)?,
            q_normed: DeviceBuffer::zeroed(rows * q_width)?,
            k_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            q_positioned: DeviceBuffer::zeroed(rows * q_width)?,
            k_positioned: DeviceBuffer::zeroed(rows * kv_width)?,
            attended: DeviceBuffer::zeroed(rows * q_width)?,
            gated: DeviceBuffer::zeroed(rows * q_width)?,
            attention_output: DeviceBuffer::zeroed(rows * hidden)?,
            residual: DeviceBuffer::zeroed(rows * hidden)?,
            feedforward_input: DeviceBuffer::zeroed(rows * hidden)?,
            mlp_gate: DeviceBuffer::zeroed(rows * intermediate)?,
            mlp_up: DeviceBuffer::zeroed(rows * intermediate)?,
            activated: DeviceBuffer::zeroed(rows * intermediate)?,
            mlp_output: DeviceBuffer::zeroed(rows * hidden)?,
            feedforward_output: DeviceBuffer::zeroed(rows * hidden)?,
            layer_output: DeviceBuffer::zeroed(rows * hidden)?,
            local_attention: local
                .map(|layer| {
                    Sm12xKvAttentionWorkspace::new_gqa_batched(
                        attention_capacity,
                        layer.attention.q_heads,
                        layer.attention.kv_heads,
                        layer.attention.head_dim,
                        rows,
                    )
                })
                .transpose()?,
            global_attention: global
                .map(|layer| {
                    Sm12xKvAttentionWorkspace::new_gqa_batched(
                        attention_capacity,
                        layer.attention.q_heads,
                        layer.attention.kv_heads,
                        layer.attention.head_dim,
                        rows,
                    )
                })
                .transpose()?,
            tail_snapshots: (0..model.config.num_hidden_layers)
                .map(|_| {
                    Sm12xKvTailSnapshot::new(
                        model.config.num_key_value_heads,
                        model.config.head_dim,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            features: DeviceBuffer::zeroed(rows * DFLASH_EXTRACT_COUNT * hidden)?,
            final_hidden: DeviceBuffer::zeroed(rows * hidden)?,
            logits: DeviceBuffer::zeroed(rows * model.config.vocab_size)?,
            argmax_indices: DeviceBuffer::zeroed(rows)?,
            argmax_values: DeviceBuffer::zeroed(rows)?,
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.tokens.device_bytes()
            + self.linear.device_bytes()
            + [
                &self.current,
                &self.normalized,
                &self.q,
                &self.k,
                &self.v,
                &self.gate,
                &self.q_normed,
                &self.k_normed,
                &self.q_positioned,
                &self.k_positioned,
                &self.attended,
                &self.gated,
                &self.attention_output,
                &self.residual,
                &self.feedforward_input,
                &self.mlp_gate,
                &self.mlp_up,
                &self.activated,
                &self.mlp_output,
                &self.feedforward_output,
                &self.layer_output,
                &self.features,
                &self.final_hidden,
                &self.logits,
                &self.argmax_values,
            ]
            .into_iter()
            .map(DeviceBuffer::device_bytes)
            .sum::<usize>()
            + self.argmax_indices.device_bytes()
            + self
                .local_attention
                .as_ref()
                .map_or(0, Sm12xKvAttentionWorkspace::device_bytes)
            + self
                .global_attention
                .as_ref()
                .map_or(0, Sm12xKvAttentionWorkspace::device_bytes)
            + self
                .tail_snapshots
                .iter()
                .map(Sm12xKvTailSnapshot::device_bytes)
                .sum::<usize>()
    }
}

impl MuseGlimmerModel {
    fn reserve_dflash_target_rows(
        &self,
        sequence: &mut MuseGlimmerSequence,
        tokens: &[u32],
        extract_layers: &[usize; DFLASH_EXTRACT_COUNT],
        output_logits: bool,
        snapshot_tails: bool,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<sequence_cache::AppendReservation> {
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                tokens.len(),
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(muse_glimmer_cache_error)?;
        if let Err(error) = self.run_dflash_target_rows(
            &mut sequence.state,
            tokens,
            extract_layers,
            output_logits,
            cache,
            MuseGlimmerAppend {
                reservation: &reservation,
                page_table: sequence.page_table.device(),
                snapshot_tails,
            },
        ) {
            let restore_result = if snapshot_tails {
                self.restore_verification_tail_prefix(
                    sequence,
                    &reservation,
                    reservation.start_position() % DFLASH_BLOCK_SIZE,
                    cache,
                )
            } else {
                Ok(())
            };
            let abort_result = cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &self.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(muse_glimmer_cache_error);
            restore_result?;
            abort_result?;
            return Err(error);
        }
        Ok(reservation)
    }

    fn run_dflash_target_rows(
        &self,
        state: &mut MuseGlimmerDecodeState,
        tokens: &[u32],
        extract_layers: &[usize; DFLASH_EXTRACT_COUNT],
        output_logits: bool,
        cache: &mut MuseGlimmerSequenceCache,
        append: MuseGlimmerAppend<'_>,
    ) -> Result<()> {
        let active_rows = tokens.len();
        let start_position = state.position;
        if active_rows == 0
            || active_rows > DFLASH_BLOCK_SIZE
            || start_position + active_rows > state.max_tokens
            || tokens
                .iter()
                .any(|&token| token as usize >= self.config.vocab_size)
        {
            return Err(Error::Shape {
                label: "Muse DFlash target rows",
                expected: format!(
                    "1..={DFLASH_BLOCK_SIZE} valid tokens through capacity {}",
                    state.max_tokens
                ),
                actual: format!("position={start_position} tokens={tokens:?}"),
            });
        }
        let workspace = state.verification.as_mut().ok_or_else(|| Error::Format {
            label: "Muse DFlash target rows",
            detail: "decode state has no DFlash verification workspace".to_string(),
        })?;

        // Keep every target projection at fixed N=16. NVFP4 accumulation can
        // change at a GEMM shape boundary, so prompt and verification rows must
        // use the same numerical regime for speculative decoding to be exact.
        workspace.linear.set_rows(DFLASH_BLOCK_SIZE)?;
        let mut padded_tokens = [0_u32; DFLASH_BLOCK_SIZE];
        padded_tokens[..active_rows].copy_from_slice(tokens);
        workspace.tokens.copy_from_host(&padded_tokens)?;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.config.vocab_size,
            self.config.hidden_size,
            &self.embedding,
            &workspace.tokens,
            workspace.current.output(),
            &self.stream,
        )?;
        self.embedding_norm.run_into(
            DFLASH_BLOCK_SIZE,
            self.config.hidden_size,
            &workspace.current,
            &mut workspace.normalized,
            &self.stream,
        )?;
        workspace.current.copy_prefix_from_device_on_stream(
            &workspace.normalized,
            workspace.current.len(),
            &self.stream,
        )?;

        for layer_index in 0..self.layers.len() {
            if let Some(extract_index) = extract_layers
                .iter()
                .position(|&extract| extract == layer_index)
            {
                copy_f32_rows_into_columns_on_stream(
                    active_rows,
                    self.config.hidden_size,
                    DFLASH_EXTRACT_COUNT * self.config.hidden_size,
                    extract_index * self.config.hidden_size,
                    &workspace.current,
                    workspace.features.output(),
                    &self.stream,
                )?;
            }
            let layer = &self.layers[layer_index];
            let attention = &layer.attention;
            layer.input_norm.run_into(
                active_rows,
                self.config.hidden_size,
                &workspace.current,
                &mut workspace.normalized,
                &self.stream,
            )?;
            workspace.linear.run(
                &attention.q,
                &workspace.normalized,
                &mut workspace.q,
                &self.stream,
            )?;
            workspace.linear.run(
                &attention.k,
                &workspace.normalized,
                &mut workspace.k,
                &self.stream,
            )?;
            workspace.linear.run(
                &attention.v,
                &workspace.normalized,
                &mut workspace.v,
                &self.stream,
            )?;
            workspace.linear.run(
                &attention.gate,
                &workspace.normalized,
                &mut workspace.gate,
                &self.stream,
            )?;
            let gate_len = active_rows * attention.q_heads * attention.head_dim;
            round_f32_to_bf16_prefix_in_place_on_stream(
                workspace.gate.inout(),
                gate_len,
                &self.stream,
            )?;
            attention.q_norm.run_into(
                active_rows * attention.q_heads,
                attention.head_dim,
                &workspace.q,
                &mut workspace.q_normed,
                &self.stream,
            )?;
            attention.k_norm.run_into(
                active_rows * attention.kv_heads,
                attention.head_dim,
                &workspace.k,
                &mut workspace.k_normed,
                &self.stream,
            )?;
            if let Some(theta) = attention.rope_theta {
                rope_neox_sequence_f32_into_on_stream(
                    active_rows,
                    attention.q_heads,
                    attention.head_dim,
                    &workspace.q_normed,
                    workspace.q_positioned.output(),
                    start_position,
                    theta,
                    &self.stream,
                )?;
                rope_neox_sequence_f32_into_on_stream(
                    active_rows,
                    attention.kv_heads,
                    attention.head_dim,
                    &workspace.k_normed,
                    workspace.k_positioned.output(),
                    start_position,
                    theta,
                    &self.stream,
                )?;
            } else {
                let q_len = active_rows * attention.q_heads * attention.head_dim;
                let k_len = active_rows * attention.kv_heads * attention.head_dim;
                workspace.q_positioned.copy_prefix_from_device_on_stream(
                    &workspace.q_normed,
                    q_len,
                    &self.stream,
                )?;
                workspace.k_positioned.copy_prefix_from_device_on_stream(
                    &workspace.k_normed,
                    k_len,
                    &self.stream,
                )?;
            }
            let local = attention.window.is_some();
            let MuseTargetBatchWorkspace {
                local_attention,
                global_attention,
                q_positioned,
                k_positioned,
                v,
                attended,
                tail_snapshots,
                ..
            } = workspace.as_mut();
            let attention_workspace = if local {
                local_attention.as_mut()
            } else {
                global_attention.as_mut()
            }
            .ok_or_else(|| Error::Format {
                label: "Muse verification attention workspace",
                detail: format!(
                    "missing {} attention workspace",
                    if local { "local" } else { "global" }
                ),
            })?;
            cache
                .with_append_pages(append.reservation, |backend, pages| {
                    let pool = backend.pool_mut(layer_index)?;
                    if append.snapshot_tails {
                        let page = pages.iter().next().expect("append has a first page");
                        pool.snapshot_tail_on_stream(
                            page.page().slot(),
                            &mut tail_snapshots[layer_index],
                            &self.stream,
                        )?;
                    }
                    for page in pages.iter() {
                        let segment = page.segment();
                        let mut processed = 0;
                        while processed < segment.rows() {
                            let input_row = segment.input_offset() + processed;
                            let position = start_position + input_row;
                            let rows = (segment.rows() - processed)
                                .min(DFLASH_BLOCK_SIZE - position % DFLASH_BLOCK_SIZE);
                            pool.append_rows_at_offset_on_stream(
                                page.page().slot(),
                                segment.page_offset() + processed,
                                k_positioned,
                                v,
                                input_row,
                                rows,
                                &self.stream,
                            )?;
                            attention_workspace
                                .attention_paged_causal_rows_at_offset_into_on_stream(
                                    pool,
                                    append.page_table,
                                    position,
                                    q_positioned,
                                    input_row,
                                    rows,
                                    attention.window,
                                    attended.output(),
                                    &self.stream,
                                )?;
                            processed += rows;
                        }
                    }
                    Ok(())
                })
                .map_err(muse_glimmer_cache_error)?;
            sigmoid_mul_f32_prefix_into_on_stream(
                &workspace.gate,
                &workspace.attended,
                workspace.gated.output(),
                active_rows * attention.q_heads * attention.head_dim,
                &self.stream,
            )?;
            workspace.linear.run(
                &attention.output,
                &workspace.gated,
                &mut workspace.attention_output,
                &self.stream,
            )?;
            layer.post_attention_norm.run_into(
                active_rows,
                self.config.hidden_size,
                &workspace.attention_output,
                &mut workspace.normalized,
                &self.stream,
            )?;
            let residual_len = active_rows * self.config.hidden_size;
            add_f32_prefix_into_on_stream(
                &workspace.current,
                &workspace.normalized,
                workspace.residual.output(),
                residual_len,
                &self.stream,
            )?;
            layer.pre_feedforward_norm.run_into(
                active_rows,
                self.config.hidden_size,
                &workspace.residual,
                &mut workspace.feedforward_input,
                &self.stream,
            )?;
            workspace.linear.run(
                &layer.mlp.gate,
                &workspace.feedforward_input,
                &mut workspace.mlp_gate,
                &self.stream,
            )?;
            workspace.linear.run(
                &layer.mlp.up,
                &workspace.feedforward_input,
                &mut workspace.mlp_up,
                &self.stream,
            )?;
            let activated_len = active_rows * self.config.intermediate_size;
            silu_mul_f32_prefix_into_on_stream(
                &workspace.mlp_gate,
                &workspace.mlp_up,
                workspace.activated.output(),
                activated_len,
                &self.stream,
            )?;
            workspace.linear.run(
                &layer.mlp.down,
                &workspace.activated,
                &mut workspace.mlp_output,
                &self.stream,
            )?;
            layer.post_feedforward_norm.run_into(
                active_rows,
                self.config.hidden_size,
                &workspace.mlp_output,
                &mut workspace.feedforward_output,
                &self.stream,
            )?;
            let layer_output_len = active_rows * self.config.hidden_size;
            add_f32_prefix_into_on_stream(
                &workspace.residual,
                &workspace.feedforward_output,
                workspace.layer_output.output(),
                layer_output_len,
                &self.stream,
            )?;
            workspace.current.copy_prefix_from_device_on_stream(
                &workspace.layer_output,
                layer_output_len,
                &self.stream,
            )?;
        }

        if output_logits {
            self.final_norm.run_into(
                active_rows,
                self.config.hidden_size,
                &workspace.current,
                &mut workspace.final_hidden,
                &self.stream,
            )?;
            workspace.linear.run(
                &self.lm_head,
                &workspace.final_hidden,
                &mut workspace.logits,
                &self.stream,
            )?;
            let logits_len = active_rows * self.config.vocab_size;
            round_f32_to_bf16_prefix_in_place_on_stream(
                workspace.logits.inout(),
                logits_len,
                &self.stream,
            )?;
            argmax_f32_batch_into_on_stream(
                &workspace.logits,
                workspace.argmax_indices.output(),
                workspace.argmax_values.output(),
                DFLASH_BLOCK_SIZE,
                self.config.vocab_size,
                &self.stream,
            )?;
        }

        if let Some(dflash) = &self.dflash {
            let MuseGlimmerDecodeState {
                verification,
                dflash_state,
                ..
            } = state;
            dflash.inject_features(
                &verification
                    .as_ref()
                    .expect("DFlash verification workspace")
                    .features,
                dflash_state.as_mut().expect("DFlash sequence state"),
                active_rows,
                start_position,
                &self.stream,
            )?;
        }
        Ok(())
    }

    pub(super) fn forward_verification_block(
        &self,
        sequence: &mut MuseGlimmerSequence,
        tokens: &[u32; DFLASH_BLOCK_SIZE],
        extract_layers: &[usize; DFLASH_EXTRACT_COUNT],
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<sequence_cache::AppendReservation> {
        sequence.state.batch_logits_row = None;
        self.reserve_dflash_target_rows(sequence, tokens, extract_layers, true, true, cache)
    }

    pub(super) fn restore_verification_tail_prefix(
        &self,
        sequence: &mut MuseGlimmerSequence,
        reservation: &sequence_cache::AppendReservation,
        rows: usize,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<()> {
        let snapshots = &sequence
            .state
            .verification
            .as_ref()
            .expect("DFlash verification workspace")
            .tail_snapshots;
        cache
            .with_append_pages(reservation, |backend, pages| {
                let page = pages.iter().next().expect("append has a first page");
                let slot = page.page().slot();
                for (layer_index, snapshot) in snapshots.iter().enumerate() {
                    backend
                        .pool_mut(layer_index)?
                        .restore_tail_prefix_on_stream(slot, snapshot, rows, &self.stream)?;
                }
                Ok(())
            })
            .map_err(muse_glimmer_cache_error)
    }

    /// Advances a greedy DFlash request through one prompt chunk using the
    /// target's fixed-N=16 projection regime.
    pub fn dflash_prefill_chunk(
        &self,
        sequence: &mut MuseGlimmerSequence,
        tokens: &[u32],
        output_logits: bool,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<()> {
        let target_layers = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "Muse Glimmer DFlash prefill",
                detail: "model was loaded without a DFlash companion".to_string(),
            })?
            .config
            .target_layers;
        let start_position = sequence.state.position;
        let reservation = self.reserve_dflash_target_rows(
            sequence,
            tokens,
            &target_layers,
            output_logits,
            false,
            cache,
        )?;
        cache
            .commit_append(
                reservation,
                tokens.len(),
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(muse_glimmer_cache_error)?;
        sequence.state.position = start_position + tokens.len();
        sequence.state.batch_logits_row = output_logits.then_some(tokens.len() - 1);
        Ok(())
    }
}
