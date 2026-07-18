use super::*;
use nvfp4::{
    copy_bf16_rows_to_f32_indexed_into_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_weighted_accumulate_slots_f32_batch_on_stream,
    rope_neox_proportional_sequence_f32_at_offset_into_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream,
};

/// One scheduler-selected Gemma prompt chunk and its persistent sequence state.
pub struct Gemma4PrefillRow<'tokens, 'state> {
    pub token_ids: &'tokens [u32],
    pub state: &'state mut Gemma4DecodeState,
}

struct Gemma4BatchLinearWorkspace {
    rows: usize,
}

impl Gemma4BatchLinearWorkspace {
    fn new(rows: usize) -> Self {
        Self { rows }
    }

    fn run(
        &mut self,
        linear: &Gemma4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        linear.run_rows_into(input, output, self.rows, stream)
    }

    fn device_bytes(&self) -> usize {
        0
    }
}

struct Gemma4BatchAttentionWorkspace {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    v_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Gemma4BatchAttentionWorkspace {
    fn new(attention: &Gemma4Attention, rows: usize) -> Result<Self> {
        let q_width = attention.q_heads * attention.head_dim;
        let kv_width = attention.kv_heads * attention.head_dim;
        Ok(Self {
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            q_normed: DeviceBuffer::zeroed(rows * q_width)?,
            k_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            v_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            q_rope: DeviceBuffer::zeroed(rows * q_width)?,
            k_rope: DeviceBuffer::zeroed(rows * kv_width)?,
            attended: DeviceBuffer::zeroed(rows * q_width)?,
            output: DeviceBuffer::zeroed(rows * attention.output.out_features)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_normed.device_bytes()
            + self.k_normed.device_bytes()
            + self.v_normed.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attended.device_bytes()
            + self.output.device_bytes()
    }
}

struct Gemma4BatchRouterWorkspace {
    normalized: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    normalized_weights: DeviceBuffer<f32>,
    route_weights: DeviceBuffer<f32>,
    input_norm_row_scale: DeviceBuffer<f32>,
}

impl Gemma4BatchRouterWorkspace {
    fn new(router: &Gemma4Router, rows: usize) -> Result<Self> {
        let (experts, hidden) = router.projection.shape();
        let routes = rows * router.top_k;
        Ok(Self {
            normalized: DeviceBuffer::zeroed(rows * hidden)?,
            logits: DeviceBuffer::zeroed(rows * experts)?,
            indices: DeviceBuffer::zeroed(routes)?,
            normalized_weights: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            input_norm_row_scale: DeviceBuffer::from_host(&vec![
                router.input_norm_scalar_value;
                rows
            ])?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.normalized.device_bytes()
            + self.logits.device_bytes()
            + self.indices.device_bytes()
            + self.normalized_weights.device_bytes()
            + self.route_weights.device_bytes()
            + self.input_norm_row_scale.device_bytes()
    }
}

struct Gemma4BatchMoeWorkspace {
    router: Gemma4BatchRouterWorkspace,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    expert_input_table: DeviceBuffer<*const f32>,
    gate_output_table: DeviceBuffer<*mut f32>,
    up_output_table: DeviceBuffer<*mut f32>,
    down_input_table: DeviceBuffer<*const f32>,
    down_output_table: DeviceBuffer<*mut f32>,
    down_result_table: DeviceBuffer<*const f32>,
    output: DeviceBuffer<f32>,
}

impl Gemma4BatchMoeWorkspace {
    fn new(moe: &Gemma4Moe, expert_input: &DeviceBuffer<f32>, rows: usize) -> Result<Self> {
        let routes_per_row = moe.router.top_k;
        let routes = rows * routes_per_row;
        let gate = DeviceBuffer::zeroed(routes * moe.intermediate_size)?;
        let up = DeviceBuffer::zeroed(routes * moe.intermediate_size)?;
        let activated = DeviceBuffer::zeroed(routes * moe.intermediate_size)?;
        let down = DeviceBuffer::zeroed(routes * moe.hidden_size)?;
        Ok(Self {
            router: Gemma4BatchRouterWorkspace::new(&moe.router, rows)?,
            expert_input_table: repeated_row_pointer_table(
                expert_input.as_const_ptr().cast::<f32>(),
                rows,
                routes_per_row,
                moe.hidden_size,
            )?,
            gate_output_table: mutable_row_pointer_table(&gate, routes, moe.intermediate_size)?,
            up_output_table: mutable_row_pointer_table(&up, routes, moe.intermediate_size)?,
            down_input_table: const_row_pointer_table(&activated, routes, moe.intermediate_size)?,
            down_output_table: mutable_row_pointer_table(&down, routes, moe.hidden_size)?,
            down_result_table: const_row_pointer_table(&down, routes, moe.hidden_size)?,
            gate,
            up,
            activated,
            down,
            output: DeviceBuffer::zeroed(rows * moe.hidden_size)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.gate.device_bytes()
            + self.up.device_bytes()
            + self.activated.device_bytes()
            + self.down.device_bytes()
            + self.expert_input_table.device_bytes()
            + self.gate_output_table.device_bytes()
            + self.up_output_table.device_bytes()
            + self.down_input_table.device_bytes()
            + self.down_output_table.device_bytes()
            + self.down_result_table.device_bytes()
            + self.output.device_bytes()
    }
}

/// Reusable layer-major storage for ragged Gemma prompt chunks.
pub struct Gemma4PrefillBatchWorkspace {
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
    token_ids: DeviceBuffer<u32>,
    host_token_ids: Vec<u32>,
    hidden: DeviceBuffer<f32>,
    layer_output: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    dense_input: DeviceBuffer<f32>,
    dense_output: DeviceBuffer<f32>,
    moe_input: DeviceBuffer<f32>,
    moe_output: DeviceBuffer<f32>,
    combined: DeviceBuffer<f32>,
    local_attention: Gemma4BatchAttentionWorkspace,
    global_attention: Gemma4BatchAttentionWorkspace,
    dense: Gemma4MlpWorkspace,
    moe: Gemma4BatchMoeWorkspace,
    embedding_row_scale: DeviceBuffer<f32>,
    layer_row_scales: Vec<DeviceBuffer<f32>>,
    linear: Gemma4BatchLinearWorkspace,
}

impl Gemma4PrefillBatchWorkspace {
    /// Returns the exact device bytes retained by this shared workspace.
    pub fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.hidden.device_bytes()
            + self.layer_output.device_bytes()
            + self.normalized.device_bytes()
            + self.residual.device_bytes()
            + self.dense_input.device_bytes()
            + self.dense_output.device_bytes()
            + self.moe_input.device_bytes()
            + self.moe_output.device_bytes()
            + self.combined.device_bytes()
            + self.local_attention.device_bytes()
            + self.global_attention.device_bytes()
            + self.dense.gate.device_bytes()
            + self.dense.up.device_bytes()
            + self.dense.activated.device_bytes()
            + self.dense.output.device_bytes()
            + self.moe.device_bytes()
            + self.embedding_row_scale.device_bytes()
            + self
                .layer_row_scales
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
            + self.linear.device_bytes()
    }
}

impl Gemma4Model {
    /// Allocates shared scratch for ragged prompt prefill.
    pub fn new_prefill_batch_workspace(
        &self,
        sequence_capacity: usize,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Gemma4PrefillBatchWorkspace> {
        if sequence_capacity == 0 || token_capacity == 0 || max_context_tokens == 0 {
            return Err(Error::Shape {
                label: "Gemma 4 prefill workspace",
                expected: "positive sequence, token, and context capacities".to_string(),
                actual: format!(
                    "sequences={sequence_capacity} tokens={token_capacity} context={max_context_tokens}"
                ),
            });
        }
        let local = self
            .layers
            .iter()
            .find(|layer| layer.attention.window.is_some())
            .expect("Gemma 4 has local-attention layers");
        let global = self
            .layers
            .iter()
            .find(|layer| layer.attention.window.is_none())
            .expect("Gemma 4 has global-attention layers");
        let linear = Gemma4BatchLinearWorkspace::new(token_capacity);
        let moe_input = DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?;
        let moe = Gemma4BatchMoeWorkspace::new(&local.moe, &moe_input, token_capacity)?;
        Ok(Gemma4PrefillBatchWorkspace {
            sequence_capacity,
            token_capacity,
            max_context_tokens,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            host_token_ids: vec![0; token_capacity],
            hidden: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            layer_output: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            normalized: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            residual: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            dense_input: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            dense_output: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            moe_input,
            moe_output: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            combined: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            local_attention: Gemma4BatchAttentionWorkspace::new(&local.attention, token_capacity)?,
            global_attention: Gemma4BatchAttentionWorkspace::new(
                &global.attention,
                token_capacity,
            )?,
            dense: local.dense.new_workspace(token_capacity)?,
            moe,
            embedding_row_scale: DeviceBuffer::from_host(&vec![
                self.embedding_scalar_value;
                token_capacity
            ])?,
            layer_row_scales: self
                .layers
                .iter()
                .map(|layer| {
                    DeviceBuffer::from_host(&vec![layer.layer_scalar_value; token_capacity])
                })
                .collect::<Result<Vec<_>>>()?,
            linear,
        })
    }

    /// Advances one or more persistent sequence states by flattened prompt chunks.
    pub fn prefill_batch(
        &self,
        workspace: &mut Gemma4PrefillBatchWorkspace,
        rows: &mut [Gemma4PrefillRow<'_, '_>],
        stream: &CudaStream,
    ) -> Result<()> {
        if rows.is_empty() || rows.len() > workspace.sequence_capacity {
            return Err(Error::Shape {
                label: "Gemma 4 prefill rows",
                expected: format!("1..={} sequences", workspace.sequence_capacity),
                actual: rows.len().to_string(),
            });
        }
        let total_tokens = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.token_ids.len())
                .ok_or_else(|| Error::Shape {
                    label: "Gemma 4 prefill token count",
                    expected: "total token count without overflow".to_string(),
                    actual: format!("total={total} row={}", row.token_ids.len()),
                })
        })?;
        if total_tokens == 0 || total_tokens > workspace.token_capacity {
            return Err(Error::Shape {
                label: "Gemma 4 prefill token count",
                expected: format!("1..={} tokens", workspace.token_capacity),
                actual: total_tokens.to_string(),
            });
        }
        for row in rows.iter() {
            if row.token_ids.is_empty() {
                return Err(Error::Format {
                    label: "Gemma 4 prefill row",
                    detail: "prompt chunks must not be empty".to_string(),
                });
            }
            if let Some(token) = row
                .token_ids
                .iter()
                .find(|&&token| token as usize >= self.config.vocab_size)
            {
                return Err(Error::Shape {
                    label: "Gemma 4 prefill token",
                    expected: format!("token < {}", self.config.vocab_size),
                    actual: token.to_string(),
                });
            }
            let end = row.state.position.saturating_add(row.token_ids.len());
            if end > row.state.max_tokens || row.state.max_tokens > workspace.max_context_tokens {
                return Err(Error::Shape {
                    label: "Gemma 4 prefill context",
                    expected: format!(
                        "end <= sequence max_tokens <= {}",
                        workspace.max_context_tokens
                    ),
                    actual: format!("end={end} max_tokens={}", row.state.max_tokens),
                });
            }
            if row
                .state
                .kv_caches
                .iter()
                .any(|cache| cache.len() != row.state.position)
            {
                return Err(Error::Format {
                    label: "Gemma 4 prefill state",
                    detail: "layer KV positions disagree".to_string(),
                });
            }
        }

        workspace.host_token_ids.fill(0);
        let mut offset = 0;
        for row in rows.iter() {
            let end = offset + row.token_ids.len();
            workspace.host_token_ids[offset..end].copy_from_slice(row.token_ids);
            offset = end;
        }
        workspace
            .token_ids
            .copy_from_host(&workspace.host_token_ids)?;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.config.vocab_size,
            self.config.hidden_size,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            stream,
        )?;
        scale_channel_f32_device_row_scalar_in_place_on_stream(
            workspace.hidden.inout(),
            &self.embedding_channel_scale,
            &workspace.embedding_row_scale,
            workspace.token_capacity,
            self.config.hidden_size,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.hidden.inout(), stream)?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            run_layer_prefill(layer, layer_index, workspace, rows, stream)?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
        }
        for row in rows {
            row.state.position += row.token_ids.len();
        }
        Ok(())
    }
}

fn run_layer_prefill(
    layer: &Gemma4DecoderLayer,
    layer_index: usize,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    rows: &mut [Gemma4PrefillRow<'_, '_>],
    stream: &CudaStream,
) -> Result<()> {
    let capacity = workspace.token_capacity;
    let hidden = layer.attention.q.in_features;
    layer.input_norm.run_into(
        capacity,
        hidden,
        &workspace.hidden,
        &mut workspace.normalized,
        stream,
    )?;
    let attention_workspace = if layer.attention.window.is_some() {
        &mut workspace.local_attention
    } else {
        &mut workspace.global_attention
    };
    run_attention_prefill(
        &layer.attention,
        attention_workspace,
        &mut workspace.linear,
        &workspace.normalized,
        rows,
        layer_index,
        capacity,
        stream,
    )?;
    layer.post_attention_norm.run_into(
        capacity,
        hidden,
        &attention_workspace.output,
        &mut workspace.normalized,
        stream,
    )?;
    add_f32_into_on_stream(
        &workspace.hidden,
        &workspace.normalized,
        workspace.residual.output(),
        stream,
    )?;

    layer.dense_input_norm.run_into(
        capacity,
        hidden,
        &workspace.residual,
        &mut workspace.dense_input,
        stream,
    )?;
    run_mlp_prefill(
        &layer.dense,
        &mut workspace.dense,
        &mut workspace.linear,
        &workspace.dense_input,
        stream,
    )?;
    layer.dense_post_norm.run_into(
        capacity,
        hidden,
        &workspace.dense.output,
        &mut workspace.dense_output,
        stream,
    )?;

    layer.moe_input_norm.run_into(
        capacity,
        hidden,
        &workspace.residual,
        &mut workspace.moe_input,
        stream,
    )?;
    run_moe_prefill(
        &layer.moe,
        &mut workspace.moe,
        &mut workspace.linear,
        &workspace.residual,
        capacity,
        stream,
    )?;
    layer.moe_post_norm.run_into(
        capacity,
        hidden,
        &workspace.moe.output,
        &mut workspace.moe_output,
        stream,
    )?;
    add_f32_into_on_stream(
        &workspace.dense_output,
        &workspace.moe_output,
        workspace.combined.output(),
        stream,
    )?;
    layer.post_feedforward_norm.run_into(
        capacity,
        hidden,
        &workspace.combined,
        &mut workspace.normalized,
        stream,
    )?;
    add_f32_into_on_stream(
        &workspace.residual,
        &workspace.normalized,
        workspace.layer_output.output(),
        stream,
    )?;
    scale_channel_f32_device_row_scalar_in_place_on_stream(
        workspace.layer_output.inout(),
        &layer.layer_scale_channels,
        &workspace.layer_row_scales[layer_index],
        capacity,
        hidden,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attention_prefill(
    attention: &Gemma4Attention,
    workspace: &mut Gemma4BatchAttentionWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    rows: &mut [Gemma4PrefillRow<'_, '_>],
    layer_index: usize,
    capacity: usize,
    stream: &CudaStream,
) -> Result<()> {
    linear.run(&attention.q, input, &mut workspace.q, stream)?;
    linear.run(&attention.k, input, &mut workspace.k, stream)?;
    if let Some(v) = &attention.v {
        linear.run(v, input, &mut workspace.v, stream)?;
    }
    attention.q_norm.run_into(
        capacity * attention.q_heads,
        attention.head_dim,
        &workspace.q,
        &mut workspace.q_normed,
        stream,
    )?;
    attention.k_norm.run_into(
        capacity * attention.kv_heads,
        attention.head_dim,
        &workspace.k,
        &mut workspace.k_normed,
        stream,
    )?;
    let value_input = attention.v.as_ref().map_or(&workspace.k, |_| &workspace.v);
    rms_norm_f32_into_on_stream(
        capacity * attention.kv_heads,
        attention.head_dim,
        value_input,
        &attention.value_norm_weight,
        workspace.v_normed.output(),
        attention.q_norm.eps,
        stream,
    )?;

    let mut offset = 0;
    for row in rows {
        let position = row.state.position;
        rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            attention.q_heads,
            attention.head_dim,
            attention.rotary_dim,
            &workspace.q_normed,
            workspace.q_rope.output(),
            offset,
            position,
            attention.rope_theta,
            stream,
        )?;
        rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            attention.kv_heads,
            attention.head_dim,
            attention.rotary_dim,
            &workspace.k_normed,
            workspace.k_rope.output(),
            offset,
            position,
            attention.rope_theta,
            stream,
        )?;
        row.state.compact_attention[layer_index].append_causal_rows_at_offset_into_on_stream(
            &mut row.state.kv_caches[layer_index],
            &workspace.q_rope,
            &workspace.k_rope,
            &workspace.v_normed,
            offset,
            row.token_ids.len(),
            attention.window,
            workspace.attended.output(),
            stream,
        )?;
        offset += row.token_ids.len();
    }
    linear.run(
        &attention.output,
        &workspace.attended,
        &mut workspace.output,
        stream,
    )
}

fn run_mlp_prefill(
    mlp: &Gemma4Mlp,
    workspace: &mut Gemma4MlpWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    linear.run(&mlp.gate, input, &mut workspace.gate, stream)?;
    linear.run(&mlp.up, input, &mut workspace.up, stream)?;
    gelu_tanh_mul_f32_into_on_stream(
        &workspace.gate,
        &workspace.up,
        workspace.activated.output(),
        stream,
    )?;
    linear.run(
        &mlp.down,
        &workspace.activated,
        &mut workspace.output,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_moe_prefill(
    moe: &Gemma4Moe,
    workspace: &mut Gemma4BatchMoeWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    router_input: &DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let experts = moe.router.projection.out_features;
    let routes_per_row = moe.router.top_k;
    rms_norm_f32_into_on_stream(
        rows,
        moe.hidden_size,
        router_input,
        &moe.router.input_norm_weight,
        workspace.router.normalized.output(),
        moe.router.rms_norm_eps,
        stream,
    )?;
    scale_channel_f32_device_row_scalar_in_place_on_stream(
        workspace.router.normalized.inout(),
        &moe.router.router_scale,
        &workspace.router.input_norm_row_scale,
        rows,
        moe.hidden_size,
        stream,
    )?;
    linear.run(
        &moe.router.projection,
        &workspace.router.normalized,
        &mut workspace.router.logits,
        stream,
    )?;
    moe_topk_f32_batch_into_on_stream(
        &workspace.router.logits,
        workspace.router.indices.output(),
        workspace.router.normalized_weights.output(),
        rows,
        experts,
        routes_per_row,
        true,
        stream,
    )?;
    gather_indexed_mul_f32_into_on_stream(
        &moe.router.per_expert_scale,
        &workspace.router.indices,
        &workspace.router.normalized_weights,
        workspace.router.route_weights.output(),
        stream,
    )?;
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
        &workspace.router.indices,
        &workspace.expert_input_table,
        &moe.gate_packed_table,
        &moe.gate_scale_table,
        &moe.gate_scale_2,
        &workspace.gate_output_table,
        moe.intermediate_size,
        moe.hidden_size,
        stream,
    )?;
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
        &workspace.router.indices,
        &workspace.expert_input_table,
        &moe.up_packed_table,
        &moe.up_scale_table,
        &moe.up_scale_2,
        &workspace.up_output_table,
        moe.intermediate_size,
        moe.hidden_size,
        stream,
    )?;
    gelu_tanh_mul_f32_into_on_stream(
        &workspace.gate,
        &workspace.up,
        workspace.activated.output(),
        stream,
    )?;
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
        &workspace.router.indices,
        &workspace.down_input_table,
        &moe.down_packed_table,
        &moe.down_scale_table,
        &moe.down_scale_2,
        &workspace.down_output_table,
        moe.hidden_size,
        moe.intermediate_size,
        stream,
    )?;
    moe_weighted_accumulate_slots_f32_batch_on_stream(
        &workspace.router.indices,
        &workspace.router.route_weights,
        &workspace.down_result_table,
        &moe.expert_alpha,
        workspace.output.inout(),
        rows,
        routes_per_row,
        stream,
    )
}

fn const_row_pointer_table(
    buffer: &DeviceBuffer<f32>,
    rows: usize,
    width: usize,
) -> Result<DeviceBuffer<*const f32>> {
    let base = buffer.as_const_ptr().cast::<f32>();
    DeviceBuffer::from_host(
        &(0..rows)
            .map(|row| unsafe { base.add(row * width) })
            .collect::<Vec<_>>(),
    )
}

fn mutable_row_pointer_table(
    buffer: &DeviceBuffer<f32>,
    rows: usize,
    width: usize,
) -> Result<DeviceBuffer<*mut f32>> {
    DeviceBuffer::from_host(
        &const_row_pointer_table_host(buffer, rows, width)
            .into_iter()
            .map(|pointer| pointer.cast_mut())
            .collect::<Vec<_>>(),
    )
}

fn const_row_pointer_table_host(
    buffer: &DeviceBuffer<f32>,
    rows: usize,
    width: usize,
) -> Vec<*const f32> {
    let base = buffer.as_const_ptr().cast::<f32>();
    (0..rows)
        .map(|row| unsafe { base.add(row * width) })
        .collect()
}

fn repeated_row_pointer_table(
    base: *const f32,
    rows: usize,
    repeats: usize,
    width: usize,
) -> Result<DeviceBuffer<*const f32>> {
    DeviceBuffer::from_host(
        &(0..rows)
            .flat_map(|row| {
                let pointer = unsafe { base.add(row * width) };
                std::iter::repeat_n(pointer, repeats)
            })
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_bf16_batched_projection_matches_reference() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let checkpoint = Gemma4Checkpoint::open(model_dir).expect("open Gemma checkpoint");
        let linear = Gemma4Linear::load(
            &checkpoint,
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        .expect("load q projection");
        let rows = 8;
        let input = (0..rows * linear.in_features)
            .map(|index| ((index % 101) as f32 - 50.0) / 50.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input).expect("upload input");
        let mut reference =
            DeviceBuffer::zeroed(rows * linear.out_features).expect("reference output");
        let mut actual = DeviceBuffer::zeroed(rows * linear.out_features).expect("batch output");
        let stream = CudaStream::new_blocking().expect("stream");
        linear
            .run_rows_into(&input, &mut reference, rows, &stream)
            .expect("reference projection");
        let mut workspace = Gemma4BatchLinearWorkspace::new(rows);
        workspace
            .run(&linear, &input, &mut actual, &stream)
            .expect("batched projection");
        let reference = reference.copy_to_host(&stream).expect("reference download");
        let actual = actual.copy_to_host(&stream).expect("actual download");
        let max_error = actual
            .iter()
            .zip(reference.iter())
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 0.25, "max projection error={max_error}");
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_batched_moe_matches_independent_tokens() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let checkpoint = Gemma4Checkpoint::open(model_dir).expect("open Gemma checkpoint");
        let moe = Gemma4Moe::load(&checkpoint, 0).expect("load layer-zero MoE");
        let rows = 3;
        let router_host = (0..rows * moe.hidden_size)
            .map(|index| ((index % 97) as f32 - 48.0) / 48.0)
            .collect::<Vec<_>>();
        let expert_host = (0..rows * moe.hidden_size)
            .map(|index| ((index % 89) as f32 - 44.0) / 44.0)
            .collect::<Vec<_>>();
        let router_input = DeviceBuffer::from_host(&router_host).expect("router input");
        let expert_input = DeviceBuffer::from_host(&expert_host).expect("expert input");
        let stream = CudaStream::new_blocking().expect("stream");
        let mut workspace =
            Gemma4BatchMoeWorkspace::new(&moe, &expert_input, rows).expect("batch MoE workspace");
        let mut linear = Gemma4BatchLinearWorkspace::new(rows);
        run_moe_prefill(
            &moe,
            &mut workspace,
            &mut linear,
            &router_input,
            rows,
            &stream,
        )
        .expect("batched MoE");
        let actual = workspace
            .output
            .copy_to_host(&stream)
            .expect("batch output");

        for row in 0..rows {
            let start = row * moe.hidden_size;
            let end = start + moe.hidden_size;
            let router_row = DeviceBuffer::from_host(&router_host[start..end]).expect("router row");
            let expert_row = DeviceBuffer::from_host(&expert_host[start..end]).expect("expert row");
            let mut reference_workspace = moe.new_workspace().expect("reference workspace");
            moe.run_into(&router_row, &expert_row, &mut reference_workspace, &stream)
                .expect("reference MoE");
            let reference = reference_workspace
                .output
                .copy_to_host(&stream)
                .expect("reference output");
            let max_error = actual[start..end]
                .iter()
                .zip(reference.iter())
                .map(|(actual, reference)| (actual - reference).abs())
                .fold(0.0f32, f32::max);
            assert!(max_error <= 0.5, "row={row} max MoE error={max_error}");
        }
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_batched_first_layer_matches_token_serial_execution() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let model = Gemma4Model::load(model_dir).expect("load Gemma 4");
        let rows = 4;
        let hidden = model.config.hidden_size;
        let input_host = (0..rows * hidden)
            .map(|index| ((index % 83) as f32 - 41.0) / 41.0)
            .collect::<Vec<_>>();
        let mut batch_state = model.new_decode_state(rows).expect("batch state");
        let mut serial_state = model.new_decode_state(rows).expect("serial state");
        let mut workspace = model
            .new_prefill_batch_workspace(1, rows, rows)
            .expect("batch workspace");
        workspace
            .hidden
            .copy_from_host(&input_host)
            .expect("upload layer input");
        let stream = CudaStream::new_blocking().expect("stream");
        let token_ids = vec![0; rows];
        run_layer_prefill(
            &model.layers[0],
            0,
            &mut workspace,
            &mut [Gemma4PrefillRow {
                token_ids: &token_ids,
                state: &mut batch_state,
            }],
            &stream,
        )
        .expect("batch layer");
        let actual = workspace
            .layer_output
            .copy_to_host(&stream)
            .expect("batch output");
        for row in 0..rows {
            let start = row * hidden;
            let end = start + hidden;
            let input = DeviceBuffer::from_host(&input_host[start..end]).expect("serial input");
            model.layers[0]
                .run_decode_into(
                    &input,
                    &mut serial_state.layers[0],
                    &mut serial_state.kv_caches[0],
                    &mut serial_state.compact_attention[0],
                    row,
                    &stream,
                )
                .expect("serial layer");
            let reference = model.layers[0]
                .output(&serial_state.layers[0])
                .copy_to_host(&stream)
                .expect("serial output");
            let max_error = actual[start..end]
                .iter()
                .zip(reference.iter())
                .map(|(actual, reference)| (actual - reference).abs())
                .fold(0.0f32, f32::max);
            assert!(max_error <= 1.0, "row={row} max layer error={max_error}");
        }
    }
}
