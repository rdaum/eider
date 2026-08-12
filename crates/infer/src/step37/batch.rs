use super::*;
use nvfp4::{
    Bf16TnMatmulPlan, CublasLt, GemmShape, append_rows_f32_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, f32_to_bf16_into_on_stream,
    rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream,
    silu_mul_halves_clamped_f32_batch_into_on_stream, silu_mul_halves_f32_batch_into_on_stream,
    step37_sigmoid_top8_f32_batch_into_on_stream,
};
use std::collections::HashMap;

const CUBLAS_WORKSPACE_LIMIT: u64 = 32 << 20;
const MAX_Q_HEADS: usize = 96;

/// One ragged prompt chunk and its persistent Step sequence state.
pub struct Step37PrefillRow<'tokens, 'state> {
    pub token_ids: &'tokens [u32],
    pub sequence: &'state mut Step37Sequence,
}

struct Step37PrefillStateRow<'tokens, 'state> {
    token_ids: &'tokens [u32],
    pub state: &'state mut Step37DecodeState,
}

struct Step37BatchLinearWorkspace {
    lt: CublasLt,
    input_bf16: DeviceBuffer<u16>,
    plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
}

impl Step37BatchLinearWorkspace {
    fn new(token_capacity: usize, max_input_features: usize) -> Result<Self> {
        Ok(Self {
            lt: CublasLt::new()?,
            input_bf16: DeviceBuffer::zeroed(token_capacity * max_input_features)?,
            plans: HashMap::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_bf16(
        &mut self,
        weight: &DeviceBuffer<u16>,
        out_features: usize,
        in_features: usize,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let input_len = rows.checked_mul(in_features).ok_or_else(|| Error::Shape {
            label: "Step-3.7 prefill BF16 input",
            expected: "rows * input features without overflow".to_string(),
            actual: format!("rows={rows} input_features={in_features}"),
        })?;
        if input.len() != input_len || input_len > self.input_bf16.len() {
            return Err(Error::Shape {
                label: "Step-3.7 prefill BF16 input",
                expected: format!("input={input_len} scratch>={input_len}"),
                actual: format!("input={} scratch={}", input.len(), self.input_bf16.len()),
            });
        }
        f32_to_bf16_into_on_stream(input, self.input_bf16.output(), stream)?;
        let key = (out_features, rows, in_features);
        if !self.plans.contains_key(&key) {
            let plan = Bf16TnMatmulPlan::new(
                &self.lt,
                GemmShape::new(out_features, rows, in_features),
                CUBLAS_WORKSPACE_LIMIT,
            )?;
            self.plans.insert(key, plan);
        }
        self.plans[&key].run_on_stream(&self.lt, weight, &self.input_bf16, output.output(), stream)
    }

    fn run(
        &mut self,
        linear: &Step37Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match &linear.weight {
            Step37LinearWeight::Bf16(weight) => self.run_bf16(
                weight,
                linear.out_features,
                linear.in_features,
                input,
                output,
                rows,
                stream,
            ),
            Step37LinearWeight::Nvfp4 { .. } => linear.run_into(input, output, rows, stream),
        }
    }
}

struct Step37BatchAttentionWorkspace {
    q_heads: usize,
    compact: Sm12xKvAttentionWorkspace,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Step37BatchAttentionWorkspace {
    fn new(token_capacity: usize, max_context_tokens: usize, q_heads: usize) -> Result<Self> {
        let q_values = token_capacity * q_heads * HEAD_DIM;
        let kv_values = token_capacity * KV_HEADS * HEAD_DIM;
        Ok(Self {
            q_heads,
            compact: Sm12xKvAttentionWorkspace::new_gqa_batched(
                max_context_tokens,
                q_heads,
                KV_HEADS,
                HEAD_DIM,
                8,
            )?,
            q: DeviceBuffer::zeroed(q_values)?,
            k: DeviceBuffer::zeroed(kv_values)?,
            v: DeviceBuffer::zeroed(kv_values)?,
            q_normed: DeviceBuffer::zeroed(q_values)?,
            k_normed: DeviceBuffer::zeroed(kv_values)?,
            q_rope: DeviceBuffer::zeroed(q_values)?,
            k_rope: DeviceBuffer::zeroed(kv_values)?,
            attended: DeviceBuffer::zeroed(q_values)?,
            gate: DeviceBuffer::zeroed(token_capacity * q_heads)?,
            gated: DeviceBuffer::zeroed(q_values)?,
            output: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
        })
    }
}

struct Step37BatchMlpWorkspace {
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Step37BatchMlpWorkspace {
    fn new(token_capacity: usize, intermediate: usize) -> Result<Self> {
        Ok(Self {
            gate_up: DeviceBuffer::zeroed(token_capacity * intermediate * 2)?,
            activated: DeviceBuffer::zeroed(token_capacity * intermediate)?,
            output: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
        })
    }
}

/// Reusable layer-major Step prompt workspace shared across all active sessions.
pub struct Step37PrefillBatchWorkspace {
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
    token_ids: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layer_output: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    full_attention: Step37BatchAttentionWorkspace,
    sliding_attention: Step37BatchAttentionWorkspace,
    post_attention: DeviceBuffer<f32>,
    ffn_input: DeviceBuffer<f32>,
    mlp: HashMap<usize, Step37BatchMlpWorkspace>,
    router_logits: DeviceBuffer<f32>,
    router_indices: DeviceBuffer<u32>,
    router_weights: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    combined: DeviceBuffer<f32>,
    token_ffn_input: DeviceBuffer<f32>,
    token_route_weights: DeviceBuffer<f32>,
    paged: Step37PagedExpertWorkspace,
    linear: Step37BatchLinearWorkspace,
}

impl Step37TextModel {
    pub fn new_prefill_batch_workspace(
        &self,
        sequence_capacity: usize,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Step37PrefillBatchWorkspace> {
        if sequence_capacity == 0 || token_capacity == 0 || max_context_tokens == 0 {
            return Err(Error::Shape {
                label: "Step-3.7 prefill workspace",
                expected: "non-zero sequence, token, and context capacities".to_string(),
                actual: format!(
                    "sequences={sequence_capacity} tokens={token_capacity} context={max_context_tokens}"
                ),
            });
        }
        let mut intermediates = self
            .layers
            .iter()
            .map(|layer| match &layer.ffn {
                Step37LayerFfn::Dense(mlp) => mlp.intermediate,
                Step37LayerFfn::Moe { shared, .. } => shared.intermediate,
            })
            .collect::<Vec<_>>();
        intermediates.sort_unstable();
        intermediates.dedup();
        let max_input_features = intermediates
            .iter()
            .copied()
            .max()
            .unwrap_or(HIDDEN)
            .max(MAX_Q_HEADS * HEAD_DIM);
        let mlp = intermediates
            .into_iter()
            .map(|intermediate| {
                Step37BatchMlpWorkspace::new(token_capacity, intermediate)
                    .map(|workspace| (intermediate, workspace))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Step37PrefillBatchWorkspace {
            sequence_capacity,
            token_capacity,
            max_context_tokens,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            hidden: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            layer_output: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            normed: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            full_attention: Step37BatchAttentionWorkspace::new(
                token_capacity,
                max_context_tokens,
                64,
            )?,
            sliding_attention: Step37BatchAttentionWorkspace::new(
                token_capacity,
                max_context_tokens,
                96,
            )?,
            post_attention: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            ffn_input: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            mlp,
            router_logits: DeviceBuffer::zeroed(token_capacity * EXPERTS)?,
            router_indices: DeviceBuffer::zeroed(token_capacity * 8)?,
            router_weights: DeviceBuffer::zeroed(token_capacity * 8)?,
            routed: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            combined: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            token_ffn_input: DeviceBuffer::zeroed(HIDDEN)?,
            token_route_weights: DeviceBuffer::zeroed(8)?,
            paged: Step37PagedExpertWorkspace::new()?,
            linear: Step37BatchLinearWorkspace::new(token_capacity, max_input_features)?,
        })
    }

    /// Advances ragged prompt chunks with layer-major resident projection batching.
    pub fn prefill_batch(
        &mut self,
        workspace: &mut Step37PrefillBatchWorkspace,
        rows: &mut [Step37PrefillRow<'_, '_>],
        cache: &mut Step37SequenceCache,
    ) -> Result<()> {
        let mut reservations = Vec::with_capacity(rows.len());
        for index in 0..rows.len() {
            let reservation = {
                let row = &mut rows[index];
                cache.reserve_append(
                    row.sequence.cache_id,
                    row.token_ids.len(),
                    &mut Sm12xCacheContext {
                        stream: &self.stream,
                        page_table: &mut row.sequence.page_table,
                    },
                )
            };
            match reservation {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for (row, reservation) in rows[..index].iter_mut().zip(reservations.drain(..)) {
                        cache
                            .abort_append(
                                reservation,
                                &mut Sm12xCacheContext {
                                    stream: &self.stream,
                                    page_table: &mut row.sequence.page_table,
                                },
                            )
                            .map_err(step37_cache_error)?;
                    }
                    return Err(step37_cache_error(error));
                }
            }
        }
        let result = {
            let mut state_rows = Vec::with_capacity(rows.len());
            let mut appends = Vec::with_capacity(rows.len());
            for (row, reservation) in rows.iter_mut().zip(&reservations) {
                let sequence = &mut *row.sequence;
                state_rows.push(Step37PrefillStateRow {
                    token_ids: row.token_ids,
                    state: &mut sequence.state,
                });
                appends.push(Step37Append {
                    reservation,
                    page_table: sequence.page_table.device(),
                });
            }
            self.prefill_batch_impl(workspace, &mut state_rows, cache, &appends)
        };
        if let Err(error) = result {
            for (row, reservation) in rows.iter_mut().zip(reservations) {
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: &self.stream,
                            page_table: &mut row.sequence.page_table,
                        },
                    )
                    .map_err(step37_cache_error)?;
            }
            return Err(error);
        }
        for (row, reservation) in rows.iter_mut().zip(reservations) {
            let tokens = row.token_ids.len();
            cache
                .commit_append(
                    reservation,
                    tokens,
                    &mut Sm12xCacheContext {
                        stream: &self.stream,
                        page_table: &mut row.sequence.page_table,
                    },
                )
                .map_err(step37_cache_error)?;
            row.sequence.state.position += tokens;
        }
        Ok(())
    }

    fn prefill_batch_impl(
        &mut self,
        workspace: &mut Step37PrefillBatchWorkspace,
        rows: &mut [Step37PrefillStateRow<'_, '_>],
        cache: &mut Step37SequenceCache,
        appends: &[Step37Append<'_>],
    ) -> Result<()> {
        if rows.is_empty() || rows.len() > workspace.sequence_capacity {
            return Err(Error::Shape {
                label: "Step-3.7 prefill rows",
                expected: format!("1..={} sequences", workspace.sequence_capacity),
                actual: rows.len().to_string(),
            });
        }
        let total_tokens = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.token_ids.len())
                .ok_or_else(|| Error::Shape {
                    label: "Step-3.7 prefill token count",
                    expected: "total token count without overflow".to_string(),
                    actual: format!("total={total} row={}", row.token_ids.len()),
                })
        })?;
        if total_tokens == 0 || total_tokens > workspace.token_capacity {
            return Err(Error::Shape {
                label: "Step-3.7 prefill token count",
                expected: format!("1..={} tokens", workspace.token_capacity),
                actual: total_tokens.to_string(),
            });
        }
        for row in rows.iter() {
            if row.token_ids.is_empty() {
                return Err(Error::Format {
                    label: "Step-3.7 prefill row",
                    detail: "prompt chunks must not be empty".to_string(),
                });
            }
            if let Some(token) = row
                .token_ids
                .iter()
                .find(|&&token| token as usize >= self.vocab)
            {
                return Err(Error::Shape {
                    label: "Step-3.7 prefill token",
                    expected: format!("token < {}", self.vocab),
                    actual: token.to_string(),
                });
            }
            let position = row.state.position;
            let end = position
                .checked_add(row.token_ids.len())
                .ok_or_else(|| Error::Shape {
                    label: "Step-3.7 prefill context",
                    expected: "position + tokens without overflow".to_string(),
                    actual: format!("position={position} tokens={}", row.token_ids.len()),
                })?;
            if end > workspace.max_context_tokens {
                return Err(Error::Shape {
                    label: "Step-3.7 prefill context",
                    expected: format!("end <= {}", workspace.max_context_tokens),
                    actual: format!("end={end}"),
                });
            }
            if end > row.state.max_tokens {
                return Err(Error::Shape {
                    label: "Step-3.7 prefill state",
                    expected: format!("end <= {}", row.state.max_tokens),
                    actual: format!("end={end}"),
                });
            }
        }

        let mut tokens = rows
            .iter()
            .flat_map(|row| row.token_ids.iter().copied())
            .collect::<Vec<_>>();
        tokens.resize(workspace.token_capacity, 0);
        workspace.token_ids.copy_from_host(&tokens)?;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.vocab,
            HIDDEN,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            &self.stream,
        )?;

        for layer_index in 0..self.layers.len() {
            run_layer_prefill(
                &mut self.layers[layer_index],
                layer_index,
                workspace,
                rows,
                total_tokens,
                &self.stream,
                cache,
                appends,
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_layer_prefill(
    layer: &mut Step37Layer,
    layer_index: usize,
    workspace: &mut Step37PrefillBatchWorkspace,
    rows: &mut [Step37PrefillStateRow<'_, '_>],
    total_tokens: usize,
    stream: &CudaStream,
    cache: &mut Step37SequenceCache,
    appends: &[Step37Append<'_>],
) -> Result<()> {
    let capacity = workspace.token_capacity;
    layer.input_norm.run_into(
        &workspace.hidden,
        &mut workspace.normed,
        capacity,
        HIDDEN,
        stream,
    )?;
    let attention_workspace = if layer.attention.q_heads == 64 {
        &mut workspace.full_attention
    } else {
        &mut workspace.sliding_attention
    };
    run_attention_prefill(
        &layer.attention,
        attention_workspace,
        &mut workspace.linear,
        &workspace.normed,
        rows,
        layer_index,
        capacity,
        stream,
        cache,
        appends,
    )?;
    add_f32_into_on_stream(
        &workspace.hidden,
        &attention_workspace.output,
        workspace.post_attention.output(),
        stream,
    )?;
    layer.post_attention_norm.run_into(
        &workspace.post_attention,
        &mut workspace.ffn_input,
        capacity,
        HIDDEN,
        stream,
    )?;

    let ffn = match &mut layer.ffn {
        Step37LayerFfn::Dense(mlp) => run_mlp_prefill(
            mlp,
            &mut workspace.mlp,
            &mut workspace.linear,
            &workspace.ffn_input,
            capacity,
            stream,
        )?,
        Step37LayerFfn::Moe {
            shared,
            router,
            paged,
        } => {
            workspace.linear.run_bf16(
                &router.weight,
                EXPERTS,
                HIDDEN,
                &workspace.ffn_input,
                &mut workspace.router_logits,
                capacity,
                stream,
            )?;
            step37_sigmoid_top8_f32_batch_into_on_stream(
                &workspace.router_logits,
                &router.bias,
                workspace.router_indices.output(),
                workspace.router_weights.output(),
                capacity,
                stream,
            )?;
            let host_indices = workspace
                .router_indices
                .copy_prefix_to_host(total_tokens * 8, stream)?
                .into_vec();
            for token in 0..total_tokens {
                copy_row_f32_into_on_stream(
                    capacity,
                    HIDDEN,
                    token,
                    &workspace.ffn_input,
                    workspace.token_ffn_input.output(),
                    stream,
                )?;
                copy_row_f32_into_on_stream(
                    capacity,
                    8,
                    token,
                    &workspace.router_weights,
                    workspace.token_route_weights.output(),
                    stream,
                )?;
                let route = &host_indices[token * 8..(token + 1) * 8];
                paged.resolve_at_offset(route, &workspace.router_indices, token * 8, stream)?;
                let routed = paged.run_routed(
                    &mut workspace.paged,
                    &workspace.token_ffn_input,
                    &workspace.token_route_weights,
                    stream,
                )?;
                append_rows_f32_into_on_stream(
                    routed,
                    workspace.routed.output(),
                    token,
                    1,
                    HIDDEN,
                    stream,
                )?;
            }
            let shared_output = run_mlp_prefill(
                shared,
                &mut workspace.mlp,
                &mut workspace.linear,
                &workspace.ffn_input,
                capacity,
                stream,
            )?;
            add_f32_into_on_stream(
                &workspace.routed,
                shared_output,
                workspace.combined.output(),
                stream,
            )?;
            &workspace.combined
        }
    };
    add_f32_into_on_stream(
        &workspace.post_attention,
        ffn,
        workspace.layer_output.output(),
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attention_prefill(
    attention: &Step37Attention,
    workspace: &mut Step37BatchAttentionWorkspace,
    linear: &mut Step37BatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    rows: &mut [Step37PrefillStateRow<'_, '_>],
    layer_index: usize,
    capacity: usize,
    stream: &CudaStream,
    cache: &mut Step37SequenceCache,
    appends: &[Step37Append<'_>],
) -> Result<()> {
    debug_assert_eq!(workspace.q_heads, attention.q_heads);
    linear.run(&attention.q, input, &mut workspace.q, capacity, stream)?;
    linear.run(&attention.k, input, &mut workspace.k, capacity, stream)?;
    linear.run(&attention.v, input, &mut workspace.v, capacity, stream)?;
    attention.q_norm.run_into(
        &workspace.q,
        &mut workspace.q_normed,
        capacity * attention.q_heads,
        HEAD_DIM,
        stream,
    )?;
    attention.k_norm.run_into(
        &workspace.k,
        &mut workspace.k_normed,
        capacity * KV_HEADS,
        HEAD_DIM,
        stream,
    )?;
    let mut row_offset = 0;
    for (row, append) in rows.iter_mut().zip(appends) {
        let position = row.state.position;
        rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            attention.q_heads,
            HEAD_DIM,
            attention.rotary_dim,
            &workspace.q_normed,
            &attention.inv_freq,
            workspace.q_rope.output(),
            row_offset,
            position,
            stream,
        )?;
        rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            KV_HEADS,
            HEAD_DIM,
            attention.rotary_dim,
            &workspace.k_normed,
            &attention.inv_freq,
            workspace.k_rope.output(),
            row_offset,
            position,
            stream,
        )?;
        if append.reservation.start_position() != position
            || append.reservation.rows() != row.token_ids.len()
        {
            return Err(Error::Format {
                label: "Step-3.7 prefill append",
                detail: "reservation does not match the prompt chunk".to_string(),
            });
        }
        cache
            .with_append_pages(append.reservation, |backend, pages| {
                let pool = backend.pool_mut(layer_index)?;
                let q_width = attention.q_heads * HEAD_DIM;
                let kv_width = KV_HEADS * HEAD_DIM;
                for page in pages.iter() {
                    let segment = page.segment();
                    for local_row in 0..segment.rows() {
                        let token = segment.input_offset() + local_row;
                        pool.append_at_offsets_on_stream(
                            page.page().slot(),
                            segment.page_offset() + local_row,
                            &workspace.k_rope,
                            (row_offset + token) * kv_width,
                            &workspace.v,
                            (row_offset + token) * kv_width,
                            stream,
                        )?;
                        let cache_len = position + token + 1;
                        let window_start = attention
                            .window
                            .map_or(0, |window| cache_len.saturating_sub(window));
                        workspace
                            .compact
                            .attention_paged_window_offsets_into_on_stream(
                                pool,
                                append.page_table,
                                cache_len,
                                &workspace.q_rope,
                                (row_offset + token) * q_width,
                                workspace.attended.output(),
                                (row_offset + token) * q_width,
                                window_start,
                                stream,
                            )?;
                    }
                }
                Ok(())
            })
            .map_err(step37_cache_error)?;
        row_offset += row.token_ids.len();
    }
    linear.run(
        &attention.gate,
        input,
        &mut workspace.gate,
        capacity,
        stream,
    )?;
    sigmoid_scale_heads_f32_into_on_stream(
        &workspace.gate,
        &workspace.attended,
        workspace.gated.output(),
        HEAD_DIM,
        stream,
    )?;
    linear.run(
        &attention.output,
        &workspace.gated,
        &mut workspace.output,
        capacity,
        stream,
    )
}

fn run_mlp_prefill<'a>(
    mlp: &Step37Mlp,
    workspaces: &'a mut HashMap<usize, Step37BatchMlpWorkspace>,
    linear: &mut Step37BatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    capacity: usize,
    stream: &CudaStream,
) -> Result<&'a DeviceBuffer<f32>> {
    let workspace = workspaces
        .get_mut(&mlp.intermediate)
        .expect("prefill MLP workspace created for model shape");
    linear.run(
        &mlp.gate_up,
        input,
        &mut workspace.gate_up,
        capacity,
        stream,
    )?;
    if let Some(limit) = mlp.swiglu_limit {
        silu_mul_halves_clamped_f32_batch_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            capacity,
            mlp.intermediate,
            limit,
            stream,
        )?;
    } else {
        silu_mul_halves_f32_batch_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            capacity,
            mlp.intermediate,
            stream,
        )?;
    }
    linear.run(
        &mlp.down,
        &workspace.activated,
        &mut workspace.output,
        capacity,
        stream,
    )?;
    Ok(&workspace.output)
}
