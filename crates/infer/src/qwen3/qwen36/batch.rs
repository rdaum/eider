use super::{
    Fp8Linear, Qwen36Attention, Qwen36AttentionState, Qwen36DownStorage,
    Qwen36FullAttentionWeights, Qwen36GateUpStorage, Qwen36LayerBlock, Qwen36LinearAttentionState,
    Qwen36LinearAttentionWeights, Qwen36LmHead, Qwen36MoeWeights, Qwen36NextToken,
    Qwen36ParallelMoe, Qwen36SequenceState, Qwen36SharedExpertStorage, Qwen36TextModel,
    Sm12xGateUpWorkspace, maybe_round_device_f32_to_bf16,
};
use crate::nvfp4::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, Fp8TnMatmulPlan, GemmShape,
    MarlinNvfp4GateUpBatchWorkspace, MropeSections, Result, Sm12xKvAttentionWorkspace,
    add_f32_into_on_stream, argmax_f32_batch_into_on_stream,
    bf16_linear_logits_f32_batch_into_on_stream, copy_bf16_rows_to_f32_indexed_into_on_stream,
    fill_f32_into_on_stream, fp8_linear_f32_batch_into_on_stream,
    gated_delta_net_128_f32_batch_into_on_stream, gated_delta_net_128_f32_chunks_into_on_stream,
    gated_rms_norm_f32_into_on_stream, indexed_grouped_gemv_on_stream,
    moe_silu_quantize_bf16_slots_on_stream, moe_topk_f32_batch_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream,
    qwen36_ffn_finalize_routed_batch_f32_into_on_stream,
    qwen36_full_attn_prep_f32_batch_into_on_stream, qwen36_gdn_gate_batch_into_on_stream,
    qwen36_gdn_prep_batch_into_on_stream, qwen36_gdn_prep_chunks_into_on_stream,
    rms_norm_f32_into_on_stream, rope_imrope_text_batch_f32_into_on_stream,
    round_f32_to_bf16_in_place_on_stream, scale_channel_f32_device_row_scalar_in_place_on_stream,
    sigmoid_mul_f32_into_on_stream, silu_mul_halves_f32_batch_into_on_stream,
};

/// One scheduler-selected prompt chunk for batched prefill.
pub struct Qwen36PrefillRow<'tokens, 'state> {
    /// Non-empty contiguous prompt tokens consumed by this operation.
    pub token_ids: &'tokens [u32],
    /// Persistent state advanced by every token in `token_ids`.
    pub state: &'state mut Qwen36SequenceState,
}

/// One scheduler-selected sequence row for a decode tick.
pub struct Qwen36DecodeRow<'a> {
    /// Token consumed by this decode step.
    pub token_id: u32,
    /// Persistent state advanced by this decode step.
    pub state: &'a mut Qwen36SequenceState,
}

/// Device-resident output of one decode batch.
pub struct Qwen36DecodedBatch<'a> {
    workspace: &'a mut Qwen36DecodeBatchWorkspace,
    rows: usize,
    vocab: usize,
}

impl Qwen36DecodedBatch<'_> {
    /// Returns the number of decoded sequence rows.
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Returns whether the decoded batch contains no rows.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Returns the row-major device logits. Only the first `len() * vocab()`
    /// values belong to this result; the remainder is workspace padding.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.workspace.logits
    }

    /// Returns the number of logits per decoded row.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Copies active row-major logits to the host.
    pub fn copy_logits(&self) -> Result<Vec<f32>> {
        let mut logits = self
            .workspace
            .logits
            .copy_to_host(&self.workspace.stream)?
            .into_vec();
        logits.truncate(self.rows * self.vocab);
        Ok(logits)
    }

    /// Reduces each active logit row and copies the winning tokens to the host.
    pub fn top1(&mut self) -> Result<Vec<Qwen36NextToken>> {
        argmax_f32_batch_into_on_stream(
            &self.workspace.logits,
            self.workspace.next_indices.output(),
            self.workspace.next_values.output(),
            self.workspace.capacity,
            self.vocab,
            &self.workspace.stream,
        )?;
        let indices = self
            .workspace
            .next_indices
            .copy_to_host(&self.workspace.stream)?;
        let values = self
            .workspace
            .next_values
            .copy_to_host(&self.workspace.stream)?;
        Ok(indices
            .iter()
            .copied()
            .zip(values.iter().copied())
            .take(self.rows)
            .map(|(id, value)| Qwen36NextToken { id, value })
            .collect())
    }
}

struct BatchLinearPlan {
    plan: Fp8TnMatmulPlan,
    scalar_channel_scale: DeviceBuffer<f32>,
}

impl BatchLinearPlan {
    fn new(model: &Qwen36TextModel, linear: &Fp8Linear, capacity: usize) -> Result<Self> {
        Ok(Self {
            plan: Fp8TnMatmulPlan::new(
                &model.lt,
                GemmShape::new(linear.rows, capacity, linear.cols),
                8 << 20,
            )?,
            scalar_channel_scale: DeviceBuffer::zeroed(linear.rows)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.plan.workspace_bytes() + self.scalar_channel_scale.device_bytes()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_fp8_batch(
    model: &Qwen36TextModel,
    linear: &Fp8Linear,
    plan: &mut BatchLinearPlan,
    raw_input: &DeviceBuffer<f32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    w8a16_threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    if linear.channel_weight_scale.is_none() && std::env::var_os("QWEN36_FP8_W8A8").is_none() {
        fp8_linear_f32_batch_into_on_stream(
            raw_input,
            &linear.weight,
            output.output(),
            rows,
            linear.rows,
            linear.cols,
            linear.weight_scale,
            w8a16_threads,
            stream,
        )?;
        return maybe_round_device_f32_to_bf16(output, stream);
    }
    plan.plan.run_with_alpha_on_stream(
        &model.lt,
        &linear.weight,
        input,
        output.output(),
        1.0,
        stream,
    )?;
    let channel_scale = if let Some(channel_scale) = &linear.channel_weight_scale {
        channel_scale
    } else {
        let channel_scale = &mut plan.scalar_channel_scale;
        fill_f32_into_on_stream(channel_scale.output(), linear.weight_scale, stream)?;
        &*channel_scale
    };
    scale_channel_f32_device_row_scalar_in_place_on_stream(
        output.inout(),
        channel_scale,
        input_scale,
        rows,
        linear.rows,
        stream,
    )?;
    maybe_round_device_f32_to_bf16(output, stream)
}

struct BatchLinearAttentionWorkspace {
    hidden_quantized: DeviceBuffer<u8>,
    hidden_scale: DeviceBuffer<f32>,
    value_quantized: DeviceBuffer<u8>,
    value_scale: DeviceBuffer<f32>,
    qkv_output: DeviceBuffer<f32>,
    z_output: DeviceBuffer<f32>,
    alpha: DeviceBuffer<f32>,
    beta_input: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    conv_state_table: DeviceBuffer<*mut f32>,
    recurrent_state_table: DeviceBuffer<*mut f32>,
    conv_state_ptrs: Vec<*mut f32>,
    recurrent_state_ptrs: Vec<*mut f32>,
    padding_states: Vec<Qwen36LinearAttentionState>,
    gdn_output: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    qkv_plan: BatchLinearPlan,
    z_plan: BatchLinearPlan,
    out_plan: BatchLinearPlan,
}

impl BatchLinearAttentionWorkspace {
    fn new(
        model: &Qwen36TextModel,
        weights: &Qwen36LinearAttentionWeights,
        row_capacity: usize,
        state_capacity: usize,
    ) -> Result<Self> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let state_table_len = model.layers.len() * state_capacity;
        let nulls = vec![std::ptr::null_mut(); state_table_len];
        let mut padding_states = Vec::with_capacity(state_capacity);
        for _ in 0..state_capacity {
            padding_states.push(Qwen36LinearAttentionState::new(linear, weights)?);
        }
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(row_capacity * model.manifest.hidden)?,
            hidden_scale: DeviceBuffer::zeroed(row_capacity)?,
            value_quantized: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            value_scale: DeviceBuffer::zeroed(row_capacity)?,
            qkv_output: DeviceBuffer::zeroed(row_capacity * weights.qkv.rows)?,
            z_output: DeviceBuffer::zeroed(row_capacity * weights.z.rows)?,
            alpha: DeviceBuffer::zeroed(row_capacity * linear.value_heads)?,
            beta_input: DeviceBuffer::zeroed(row_capacity * linear.value_heads)?,
            gate: DeviceBuffer::zeroed(row_capacity * linear.value_heads)?,
            beta: DeviceBuffer::zeroed(row_capacity * linear.value_heads)?,
            q: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            k: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            v: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            conv_state_table: DeviceBuffer::from_host(&nulls)?,
            recurrent_state_table: DeviceBuffer::from_host(&nulls)?,
            conv_state_ptrs: nulls.clone(),
            recurrent_state_ptrs: nulls,
            padding_states,
            gdn_output: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            normed: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            output: DeviceBuffer::zeroed(row_capacity * model.manifest.hidden)?,
            qkv_plan: BatchLinearPlan::new(model, &weights.qkv, row_capacity)?,
            z_plan: BatchLinearPlan::new(model, &weights.z, row_capacity)?,
            out_plan: BatchLinearPlan::new(model, &weights.out, row_capacity)?,
        })
    }

    fn update_state_tables(
        &mut self,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_count: usize,
        capacity: usize,
    ) -> Result<()> {
        for layer_idx in 0..layer_count {
            for row_idx in 0..capacity {
                let table_idx = layer_idx * capacity + row_idx;
                let state = if let Some(row) = rows.get_mut(row_idx) {
                    match &mut row.state.layer_states[layer_idx].attention {
                        Qwen36AttentionState::LinearAttention(state) => state,
                        Qwen36AttentionState::FullAttention(_) => {
                            self.conv_state_ptrs[table_idx] = std::ptr::null_mut();
                            self.recurrent_state_ptrs[table_idx] = std::ptr::null_mut();
                            continue;
                        }
                    }
                } else {
                    &mut self.padding_states[row_idx]
                };
                self.conv_state_ptrs[table_idx] =
                    state.conv_state.as_const_ptr().cast_mut().cast::<f32>();
                self.recurrent_state_ptrs[table_idx] = state
                    .recurrent_state
                    .as_const_ptr()
                    .cast_mut()
                    .cast::<f32>();
            }
        }
        self.conv_state_table
            .copy_from_host(&self.conv_state_ptrs)?;
        self.recurrent_state_table
            .copy_from_host(&self.recurrent_state_ptrs)
    }

    fn update_prefill_state_tables(
        &mut self,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        layer_count: usize,
        state_capacity: usize,
    ) -> Result<()> {
        for layer_idx in 0..layer_count {
            for row_idx in 0..state_capacity {
                let table_idx = layer_idx * state_capacity + row_idx;
                let state = if let Some(row) = rows.get_mut(row_idx) {
                    match &mut row.state.layer_states[layer_idx].attention {
                        Qwen36AttentionState::LinearAttention(state) => state,
                        Qwen36AttentionState::FullAttention(_) => {
                            self.conv_state_ptrs[table_idx] = std::ptr::null_mut();
                            self.recurrent_state_ptrs[table_idx] = std::ptr::null_mut();
                            continue;
                        }
                    }
                } else {
                    &mut self.padding_states[row_idx]
                };
                self.conv_state_ptrs[table_idx] =
                    state.conv_state.as_const_ptr().cast_mut().cast::<f32>();
                self.recurrent_state_ptrs[table_idx] = state
                    .recurrent_state
                    .as_const_ptr()
                    .cast_mut()
                    .cast::<f32>();
            }
        }
        self.conv_state_table
            .copy_from_host(&self.conv_state_ptrs)?;
        self.recurrent_state_table
            .copy_from_host(&self.recurrent_state_ptrs)
    }

    fn device_bytes(&self) -> usize {
        self.hidden_quantized.device_bytes()
            + self.hidden_scale.device_bytes()
            + self.value_quantized.device_bytes()
            + self.value_scale.device_bytes()
            + self.qkv_output.device_bytes()
            + self.z_output.device_bytes()
            + self.alpha.device_bytes()
            + self.beta_input.device_bytes()
            + self.gate.device_bytes()
            + self.beta.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.conv_state_table.device_bytes()
            + self.recurrent_state_table.device_bytes()
            + self
                .padding_states
                .iter()
                .map(Qwen36LinearAttentionState::device_bytes)
                .sum::<usize>()
            + self.gdn_output.device_bytes()
            + self.normed.device_bytes()
            + self.output.device_bytes()
            + self.qkv_plan.device_bytes()
            + self.z_plan.device_bytes()
            + self.out_plan.device_bytes()
    }
}

struct BatchFullAttentionWorkspace {
    hidden_quantized: DeviceBuffer<u8>,
    hidden_scale: DeviceBuffer<f32>,
    value_quantized: DeviceBuffer<u8>,
    value_scale: DeviceBuffer<f32>,
    q_proj: DeviceBuffer<f32>,
    k_raw: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    gated_attention: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    compact_attention: Sm12xKvAttentionWorkspace,
    q_plan: BatchLinearPlan,
    k_plan: BatchLinearPlan,
    v_plan: BatchLinearPlan,
    o_plan: BatchLinearPlan,
}

impl BatchFullAttentionWorkspace {
    fn new(
        model: &Qwen36TextModel,
        weights: &Qwen36FullAttentionWeights,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
        let compact_attention = Sm12xKvAttentionWorkspace::new(
            max_context_tokens,
            model.manifest.kv_heads,
            model.manifest.head_dim,
        )?;
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            hidden_scale: DeviceBuffer::zeroed(capacity)?,
            value_quantized: DeviceBuffer::zeroed(capacity * q_width)?,
            value_scale: DeviceBuffer::zeroed(capacity)?,
            q_proj: DeviceBuffer::zeroed(capacity * weights.q.rows)?,
            k_raw: DeviceBuffer::zeroed(capacity * kv_width)?,
            v: DeviceBuffer::zeroed(capacity * kv_width)?,
            q: DeviceBuffer::zeroed(capacity * q_width)?,
            gate: DeviceBuffer::zeroed(capacity * q_width)?,
            k: DeviceBuffer::zeroed(capacity * kv_width)?,
            q_rope: DeviceBuffer::zeroed(capacity * q_width)?,
            k_rope: DeviceBuffer::zeroed(capacity * kv_width)?,
            attention: DeviceBuffer::zeroed(capacity * q_width)?,
            gated_attention: DeviceBuffer::zeroed(capacity * q_width)?,
            output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            compact_attention,
            q_plan: BatchLinearPlan::new(model, &weights.q, capacity)?,
            k_plan: BatchLinearPlan::new(model, &weights.k, capacity)?,
            v_plan: BatchLinearPlan::new(model, &weights.v, capacity)?,
            o_plan: BatchLinearPlan::new(model, &weights.o, capacity)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.hidden_quantized.device_bytes()
            + self.hidden_scale.device_bytes()
            + self.value_quantized.device_bytes()
            + self.value_scale.device_bytes()
            + self.q_proj.device_bytes()
            + self.k_raw.device_bytes()
            + self.v.device_bytes()
            + self.q.device_bytes()
            + self.gate.device_bytes()
            + self.k.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attention.device_bytes()
            + self.gated_attention.device_bytes()
            + self.output.device_bytes()
            + self.compact_attention.device_bytes()
            + self.q_plan.device_bytes()
            + self.k_plan.device_bytes()
            + self.v_plan.device_bytes()
            + self.o_plan.device_bytes()
    }
}

struct BatchMoeWorkspace {
    router_logits: DeviceBuffer<f32>,
    route_indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    marlin: MarlinNvfp4GateUpBatchWorkspace,
    sm12x_down: Sm12xGateUpWorkspace,
    shared_gate_up: DeviceBuffer<f32>,
    shared_activated: DeviceBuffer<f32>,
    shared_output: DeviceBuffer<f32>,
    shared_gate: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

struct BatchMoeStreamSync {
    fork: CudaEvent,
    join: CudaEvent,
}

enum BatchLayerGraph {
    Linear(CudaGraphExec),
    Full {
        pre_attention: CudaGraphExec,
        post_attention: CudaGraphExec,
    },
}

/// Reusable execution storage for ragged Qwen3.6 prompt chunks.
pub struct Qwen36PrefillBatchWorkspace {
    model_id: u64,
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
    stream: CudaStream,
    shared_moe_stream: CudaStream,
    moe_stream_sync: Vec<BatchMoeStreamSync>,
    token_ids: DeviceBuffer<u32>,
    positions: DeviceBuffer<u32>,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    host_token_ids: Vec<u32>,
    host_positions: Vec<u32>,
    host_sequence_offsets: Vec<u32>,
    host_sequence_lengths: Vec<u32>,
    hidden: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    attn_residual: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    linear: BatchLinearAttentionWorkspace,
    full: BatchFullAttentionWorkspace,
    moe: BatchMoeWorkspace,
}

impl Qwen36PrefillBatchWorkspace {
    /// Returns the maximum number of independent prompt chunks per call.
    pub fn sequence_capacity(&self) -> usize {
        self.sequence_capacity
    }

    /// Returns the maximum total prompt tokens per call.
    pub fn token_capacity(&self) -> usize {
        self.token_capacity
    }

    /// Returns the largest sequence context accepted by this workspace.
    pub fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    /// Returns the exact device bytes owned by the prefill workspace.
    pub fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.positions.device_bytes()
            + self.sequence_offsets.device_bytes()
            + self.sequence_lengths.device_bytes()
            + self.hidden.device_bytes()
            + self.normed_hidden.device_bytes()
            + self.attn_residual.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.linear.device_bytes()
            + self.full.device_bytes()
            + self.moe.device_bytes()
    }
}

impl BatchMoeWorkspace {
    fn new(model: &Qwen36TextModel, weights: &Qwen36MoeWeights, capacity: usize) -> Result<Self> {
        let marlin = match &weights.gate_up_storage {
            Qwen36GateUpStorage::Marlin(marlin) => marlin.new_batch_workspace(capacity)?,
            Qwen36GateUpStorage::Grouped { .. } | Qwen36GateUpStorage::Fp8 => {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 batched routed gate/up",
                    detail: "the current model does not use the Marlin NVFP4 route".to_string(),
                });
            }
        };
        if weights.storage_plan.down != Qwen36DownStorage::Sm12x {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 batched routed down",
                detail: "the current model does not use the SM12x routed-down path".to_string(),
            });
        }
        let routes = capacity * weights.experts_per_token;
        let gate_up_width = weights.expert_intermediate * 2;
        Ok(Self {
            router_logits: DeviceBuffer::zeroed(capacity * weights.num_experts)?,
            route_indices: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            marlin,
            sm12x_down: Sm12xGateUpWorkspace::new(
                model.manifest.hidden,
                weights.expert_intermediate,
                routes,
                routes,
            )?,
            shared_gate_up: DeviceBuffer::zeroed(capacity * gate_up_width)?,
            shared_activated: DeviceBuffer::zeroed(capacity * weights.expert_intermediate)?,
            shared_output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            shared_gate: DeviceBuffer::zeroed(capacity)?,
            output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.router_logits.device_bytes()
            + self.route_indices.device_bytes()
            + self.route_weights.device_bytes()
            + self.marlin.device_bytes()
            + self.sm12x_down.device_bytes()
            + self.shared_gate_up.device_bytes()
            + self.shared_activated.device_bytes()
            + self.shared_output.device_bytes()
            + self.shared_gate.device_bytes()
            + self.output.device_bytes()
    }
}

/// Reusable execution storage for a changing set of decode sequences.
pub struct Qwen36DecodeBatchWorkspace {
    model_id: u64,
    capacity: usize,
    max_context_tokens: usize,
    layer_graphs: Option<Vec<BatchLayerGraph>>,
    stream: CudaStream,
    shared_moe_stream: CudaStream,
    moe_stream_sync: Vec<BatchMoeStreamSync>,
    token_ids: DeviceBuffer<u32>,
    positions: DeviceBuffer<u32>,
    host_token_ids: Vec<u32>,
    host_positions: Vec<u32>,
    hidden: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    attn_residual: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    linear: BatchLinearAttentionWorkspace,
    full: BatchFullAttentionWorkspace,
    moe: BatchMoeWorkspace,
    lm_head_plan: Option<BatchLinearPlan>,
    lm_head_quantized: DeviceBuffer<u8>,
    lm_head_scale: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_indices: DeviceBuffer<u32>,
    next_values: DeviceBuffer<f32>,
}

impl Qwen36DecodeBatchWorkspace {
    /// Returns the maximum number of sequence rows executable in one tick.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the largest sequence context accepted by this workspace.
    pub fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.positions.device_bytes()
            + self.hidden.device_bytes()
            + self.normed_hidden.device_bytes()
            + self.attn_residual.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.final_hidden.device_bytes()
            + self.linear.device_bytes()
            + self.full.device_bytes()
            + self.moe.device_bytes()
            + self
                .lm_head_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self.lm_head_quantized.device_bytes()
            + self.lm_head_scale.device_bytes()
            + self.logits.device_bytes()
            + self.next_indices.device_bytes()
            + self.next_values.device_bytes()
    }
}

impl Qwen36LayerBlock {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_tail(
        &self,
        model: &Qwen36TextModel,
        moe: &mut BatchMoeWorkspace,
        hidden: &DeviceBuffer<f32>,
        attention_output: &DeviceBuffer<f32>,
        attn_residual: &mut DeviceBuffer<f32>,
        ffn_norm: &mut DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
    ) -> Result<()> {
        add_f32_into_on_stream(hidden, attention_output, attn_residual.output(), stream)?;
        rms_norm_f32_into_on_stream(
            capacity,
            model.manifest.hidden,
            attn_residual,
            &self.post_attn_norm,
            ffn_norm.output(),
            model.manifest.rms_eps,
            stream,
        )?;
        self.moe.run_batch(
            model,
            moe,
            ffn_norm,
            attn_residual,
            capacity,
            stream,
            parallel_moe,
        )
    }
}

impl Qwen36TextModel {
    /// Allocates shared scratch and execution plans for ragged prompt prefill.
    pub fn new_prefill_batch_workspace(
        &self,
        sequence_capacity: usize,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36PrefillBatchWorkspace> {
        if sequence_capacity == 0 || token_capacity == 0 || max_context_tokens == 0 {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill batch workspace",
                expected: "positive sequence, token, and context capacities".to_string(),
                actual: format!(
                    "sequence_capacity={sequence_capacity} token_capacity={token_capacity} max_context_tokens={max_context_tokens}"
                ),
            });
        }
        let first_linear = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::LinearAttention(weights) => Some(weights),
                Qwen36Attention::FullAttention(_) => None,
            })
            .expect("Qwen3.6 has linear-attention layers");
        let first_full = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::FullAttention(weights) => Some(weights),
                Qwen36Attention::LinearAttention(_) => None,
            })
            .expect("Qwen3.6 has full-attention layers");
        let first_moe = &self.layers.first().expect("Qwen3.6 has layers").moe;
        let mut moe_stream_sync = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            moe_stream_sync.push(BatchMoeStreamSync {
                fork: CudaEvent::new_sync()?,
                join: CudaEvent::new_sync()?,
            });
        }
        Ok(Qwen36PrefillBatchWorkspace {
            model_id: self.model_id,
            sequence_capacity,
            token_capacity,
            max_context_tokens,
            stream: CudaStream::new_blocking()?,
            shared_moe_stream: CudaStream::new_non_blocking()?,
            moe_stream_sync,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            positions: DeviceBuffer::zeroed(token_capacity)?,
            sequence_offsets: DeviceBuffer::zeroed(sequence_capacity)?,
            sequence_lengths: DeviceBuffer::zeroed(sequence_capacity)?,
            host_token_ids: vec![0; token_capacity],
            host_positions: vec![0; token_capacity],
            host_sequence_offsets: vec![0; sequence_capacity],
            host_sequence_lengths: vec![0; sequence_capacity],
            hidden: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            normed_hidden: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            attn_residual: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            linear: BatchLinearAttentionWorkspace::new(
                self,
                first_linear,
                token_capacity,
                sequence_capacity,
            )?,
            full: BatchFullAttentionWorkspace::new(
                self,
                first_full,
                token_capacity,
                max_context_tokens,
            )?,
            moe: BatchMoeWorkspace::new(self, first_moe, token_capacity)?,
        })
    }

    /// Advances persistent sequence state by ragged prompt chunks.
    ///
    /// This operation intentionally does not run the final norm or language
    /// head. A scheduler should retain the final prompt token and pass it to
    /// [`Self::decode_batch`] to obtain the first completion logits.
    pub fn prefill_batch(
        &self,
        workspace: &mut Qwen36PrefillBatchWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
    ) -> Result<()> {
        if workspace.model_id != self.model_id {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 prefill batch workspace",
                detail: "workspace was created by a different model instance".to_string(),
            });
        }
        if rows.is_empty() || rows.len() > workspace.sequence_capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill batch rows",
                expected: format!("1..={}", workspace.sequence_capacity),
                actual: rows.len().to_string(),
            });
        }
        let total_tokens = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.token_ids.len())
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill token count",
                    expected: "total token count without overflow".to_string(),
                    actual: format!("total={total} row={}", row.token_ids.len()),
                })
        })?;
        if total_tokens == 0 || total_tokens > workspace.token_capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill token count",
                expected: format!("1..={}", workspace.token_capacity),
                actual: total_tokens.to_string(),
            });
        }
        for row in rows.iter() {
            if row.token_ids.is_empty() {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill row",
                    expected: "at least one token".to_string(),
                    actual: "0 tokens".to_string(),
                });
            }
            if row.state.model_id != self.model_id {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 prefill sequence state",
                    detail: "state was created by a different model instance".to_string(),
                });
            }
            if let Some(token) = row
                .token_ids
                .iter()
                .find(|&&token| token as usize >= self.manifest.vocab)
            {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill token id",
                    expected: format!("token < {}", self.manifest.vocab),
                    actual: token.to_string(),
                });
            }
            let end = row
                .state
                .position
                .checked_add(row.token_ids.len())
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill sequence capacity",
                    expected: "position + tokens without overflow".to_string(),
                    actual: format!(
                        "position={} tokens={}",
                        row.state.position,
                        row.token_ids.len()
                    ),
                })?;
            if end > row.state.max_tokens || row.state.max_tokens > workspace.max_context_tokens {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill sequence capacity",
                    expected: format!(
                        "end <= sequence max_tokens <= {}",
                        workspace.max_context_tokens
                    ),
                    actual: format!("end={end} max_tokens={}", row.state.max_tokens),
                });
            }
            for layer in &row.state.layer_states {
                if let Qwen36AttentionState::FullAttention(state) = &layer.attention
                    && (state.compact_cache.len() != row.state.position
                        || state.cache_capacity != row.state.max_tokens)
                {
                    return Err(crate::nvfp4::Error::Format {
                        label: "Qwen3.6 prefill sequence state",
                        detail: format!(
                            "full-attention cache length/capacity {}/{} does not match sequence {}/{}",
                            state.compact_cache.len(),
                            state.cache_capacity,
                            row.state.position,
                            row.state.max_tokens
                        ),
                    });
                }
            }
        }

        workspace.host_token_ids.fill(0);
        workspace.host_positions.fill(0);
        workspace.host_sequence_offsets.fill(0);
        workspace.host_sequence_lengths.fill(0);
        let mut offset = 0usize;
        for (sequence, row) in rows.iter().enumerate() {
            workspace.host_sequence_offsets[sequence] = offset as u32;
            workspace.host_sequence_lengths[sequence] = row.token_ids.len() as u32;
            for (token_offset, &token) in row.token_ids.iter().enumerate() {
                workspace.host_token_ids[offset + token_offset] = token;
                workspace.host_positions[offset + token_offset] =
                    (row.state.position + token_offset) as u32;
            }
            offset += row.token_ids.len();
        }
        workspace
            .token_ids
            .copy_from_host(&workspace.host_token_ids)?;
        workspace
            .positions
            .copy_from_host(&workspace.host_positions)?;
        workspace
            .sequence_offsets
            .copy_from_host(&workspace.host_sequence_offsets)?;
        workspace
            .sequence_lengths
            .copy_from_host(&workspace.host_sequence_lengths)?;
        workspace.linear.update_prefill_state_tables(
            rows,
            self.layers.len(),
            workspace.sequence_capacity,
        )?;

        let stream = &workspace.stream;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab,
            self.manifest.hidden,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            stream,
        )?;
        for (layer_idx, block) in self.layers.iter().enumerate() {
            rms_norm_f32_into_on_stream(
                workspace.token_capacity,
                self.manifest.hidden,
                &workspace.hidden,
                &block.input_norm,
                workspace.normed_hidden.output(),
                self.manifest.rms_eps,
                stream,
            )?;
            let attention_output = match &block.attention {
                Qwen36Attention::LinearAttention(weights) => {
                    weights.enqueue_prefill_chunks(
                        self,
                        &mut workspace.linear,
                        &workspace.normed_hidden,
                        &workspace.sequence_offsets,
                        &workspace.sequence_lengths,
                        layer_idx,
                        workspace.sequence_capacity,
                        rows.len(),
                        total_tokens,
                        workspace.token_capacity,
                        stream,
                    )?;
                    &workspace.linear.output
                }
                Qwen36Attention::FullAttention(weights) => {
                    weights.enqueue_batch_pre(
                        self,
                        &mut workspace.full,
                        &workspace.normed_hidden,
                        &workspace.positions,
                        workspace.token_capacity,
                        stream,
                    )?;
                    weights.enqueue_prefill_cache(
                        self,
                        &mut workspace.full,
                        rows,
                        &workspace.host_sequence_offsets,
                        layer_idx,
                        stream,
                    )?;
                    weights.enqueue_batch_post(
                        self,
                        &mut workspace.full,
                        workspace.token_capacity,
                        stream,
                    )?;
                    &workspace.full.output
                }
            };
            let sync = &workspace.moe_stream_sync[layer_idx];
            block.enqueue_batch_tail(
                self,
                &mut workspace.moe,
                &workspace.hidden,
                attention_output,
                &mut workspace.attn_residual,
                &mut workspace.ffn_norm,
                workspace.token_capacity,
                stream,
                Some(Qwen36ParallelMoe {
                    shared_stream: &workspace.shared_moe_stream,
                    fork: &sync.fork,
                    join: &sync.join,
                }),
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
        }
        stream.synchronize()?;
        for row in rows {
            row.state.position += row.token_ids.len();
        }
        Ok(())
    }

    /// Allocates shared scratch and execution plans for batched decode.
    pub fn new_decode_batch_workspace(
        &self,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36DecodeBatchWorkspace> {
        if capacity == 0 || max_context_tokens == 0 {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 decode batch workspace",
                expected: "capacity > 0 and max_context_tokens > 0".to_string(),
                actual: format!("capacity={capacity} max_context_tokens={max_context_tokens}"),
            });
        }
        let first_linear = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::LinearAttention(weights) => Some(weights),
                Qwen36Attention::FullAttention(_) => None,
            })
            .expect("Qwen3.6 has linear-attention layers");
        let first_full = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::FullAttention(weights) => Some(weights),
                Qwen36Attention::LinearAttention(_) => None,
            })
            .expect("Qwen3.6 has full-attention layers");
        let first_moe = &self.layers.first().expect("Qwen3.6 has layers").moe;
        let lm_head_plan = match &self.lm_head {
            Qwen36LmHead::Nvfp4(_) => None,
            Qwen36LmHead::Fp8 { linear, .. } => Some(BatchLinearPlan::new(self, linear, capacity)?),
        };
        let mut moe_stream_sync = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            moe_stream_sync.push(BatchMoeStreamSync {
                fork: CudaEvent::new_sync()?,
                join: CudaEvent::new_sync()?,
            });
        }
        let mut workspace = Qwen36DecodeBatchWorkspace {
            model_id: self.model_id,
            capacity,
            max_context_tokens,
            layer_graphs: None,
            stream: CudaStream::new_blocking()?,
            shared_moe_stream: CudaStream::new_non_blocking()?,
            moe_stream_sync,
            token_ids: DeviceBuffer::zeroed(capacity)?,
            positions: DeviceBuffer::zeroed(capacity)?,
            host_token_ids: vec![0; capacity],
            host_positions: vec![0; capacity],
            hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            normed_hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            attn_residual: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            final_hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            linear: BatchLinearAttentionWorkspace::new(self, first_linear, capacity, capacity)?,
            full: BatchFullAttentionWorkspace::new(self, first_full, capacity, max_context_tokens)?,
            moe: BatchMoeWorkspace::new(self, first_moe, capacity)?,
            lm_head_plan,
            lm_head_quantized: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            lm_head_scale: DeviceBuffer::zeroed(capacity)?,
            logits: DeviceBuffer::zeroed(capacity * self.manifest.vocab)?,
            next_indices: DeviceBuffer::zeroed(capacity)?,
            next_values: DeviceBuffer::zeroed(capacity)?,
        };
        let enable_segmented_graphs = !std::env::var("EIDER_DISABLE_DECODE_GRAPHS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
        if enable_segmented_graphs {
            workspace.layer_graphs = Some(self.capture_batch_layer_graphs(&mut workspace)?);
        }
        Ok(workspace)
    }

    fn capture_batch_layer_graphs(
        &self,
        workspace: &mut Qwen36DecodeBatchWorkspace,
    ) -> Result<Vec<BatchLayerGraph>> {
        let Qwen36DecodeBatchWorkspace {
            capacity,
            layer_graphs: _,
            stream,
            shared_moe_stream,
            moe_stream_sync,
            positions,
            normed_hidden,
            ffn_norm,
            attn_residual,
            hidden,
            linear,
            full,
            moe,
            ..
        } = workspace;
        let mut graphs = Vec::with_capacity(self.layers.len());
        for (layer_idx, (block, sync)) in self.layers.iter().zip(moe_stream_sync).enumerate() {
            let parallel_moe = || Qwen36ParallelMoe {
                shared_stream: shared_moe_stream,
                fork: &sync.fork,
                join: &sync.join,
            };
            let graph = match &block.attention {
                Qwen36Attention::LinearAttention(weights) => {
                    let graph = stream.capture(|stream| {
                        rms_norm_f32_into_on_stream(
                            *capacity,
                            self.manifest.hidden,
                            hidden,
                            &block.input_norm,
                            normed_hidden.output(),
                            self.manifest.rms_eps,
                            stream,
                        )?;
                        weights.enqueue_batch(
                            self,
                            linear,
                            normed_hidden,
                            layer_idx,
                            *capacity,
                            stream,
                        )?;
                        block.enqueue_batch_tail(
                            self,
                            moe,
                            hidden,
                            &linear.output,
                            attn_residual,
                            ffn_norm,
                            *capacity,
                            stream,
                            Some(parallel_moe()),
                        )
                    })?;
                    BatchLayerGraph::Linear(graph)
                }
                Qwen36Attention::FullAttention(weights) => {
                    let pre_attention = stream.capture(|stream| {
                        rms_norm_f32_into_on_stream(
                            *capacity,
                            self.manifest.hidden,
                            hidden,
                            &block.input_norm,
                            normed_hidden.output(),
                            self.manifest.rms_eps,
                            stream,
                        )?;
                        weights.enqueue_batch_pre(
                            self,
                            full,
                            normed_hidden,
                            positions,
                            *capacity,
                            stream,
                        )
                    })?;
                    let post_attention = stream.capture(|stream| {
                        weights.enqueue_batch_post(self, full, *capacity, stream)?;
                        block.enqueue_batch_tail(
                            self,
                            moe,
                            hidden,
                            &full.output,
                            attn_residual,
                            ffn_norm,
                            *capacity,
                            stream,
                            Some(parallel_moe()),
                        )
                    })?;
                    BatchLayerGraph::Full {
                        pre_attention,
                        post_attention,
                    }
                }
            };
            graphs.push(graph);
            std::mem::swap(hidden, &mut moe.output);
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(hidden, &mut moe.output);
        }
        Ok(graphs)
    }

    /// Decodes one scheduler tick for arbitrary persistent sequence rows.
    ///
    /// Rows may be reordered between calls and may have different positions and
    /// cache capacities. Results preserve input order. CUDA launch padding up to
    /// workspace capacity is private to the execution plan.
    pub fn decode_batch<'w>(
        &self,
        workspace: &'w mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
    ) -> Result<Qwen36DecodedBatch<'w>> {
        if workspace.model_id != self.model_id {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 decode batch workspace",
                detail: "workspace was created by a different model instance".to_string(),
            });
        }
        if rows.is_empty() || rows.len() > workspace.capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 decode batch rows",
                expected: format!("1..={}", workspace.capacity),
                actual: rows.len().to_string(),
            });
        }
        for row in rows.iter() {
            if row.state.model_id != self.model_id {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 sequence state",
                    detail: "state was created by a different model instance".to_string(),
                });
            }
            if row.token_id as usize >= self.manifest.vocab {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 batch token id",
                    expected: format!("token < {}", self.manifest.vocab),
                    actual: row.token_id.to_string(),
                });
            }
            if row.state.position >= row.state.max_tokens
                || row.state.max_tokens > workspace.max_context_tokens
            {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 batch sequence capacity",
                    expected: format!(
                        "position < sequence max_tokens <= {}",
                        workspace.max_context_tokens
                    ),
                    actual: format!(
                        "position={} max_tokens={}",
                        row.state.position, row.state.max_tokens
                    ),
                });
            }
            for layer in &row.state.layer_states {
                if let Qwen36AttentionState::FullAttention(state) = &layer.attention
                    && (state.compact_cache.len() != row.state.position
                        || state.cache_capacity != row.state.max_tokens)
                {
                    return Err(crate::nvfp4::Error::Format {
                        label: "Qwen3.6 sequence state",
                        detail: format!(
                            "full-attention cache length/capacity {}/{} does not match sequence {}/{}",
                            state.compact_cache.len(),
                            state.cache_capacity,
                            row.state.position,
                            row.state.max_tokens
                        ),
                    });
                }
            }
        }

        let active_rows = rows.len();
        workspace.host_token_ids.fill(0);
        workspace.host_positions.fill(0);
        for (slot, row) in rows.iter().enumerate() {
            workspace.host_token_ids[slot] = row.token_id;
            workspace.host_positions[slot] = row.state.position as u32;
        }
        workspace
            .token_ids
            .copy_from_host(&workspace.host_token_ids)?;
        workspace
            .positions
            .copy_from_host(&workspace.host_positions)?;
        workspace
            .linear
            .update_state_tables(rows, self.layers.len(), workspace.capacity)?;
        let stream = &workspace.stream;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab,
            self.manifest.hidden,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            stream,
        )?;

        for (layer_idx, block) in self.layers.iter().enumerate() {
            if let Some(graph) = workspace
                .layer_graphs
                .as_ref()
                .map(|graphs| &graphs[layer_idx])
            {
                match graph {
                    BatchLayerGraph::Linear(graph) => {
                        graph.launch(stream)?;
                    }
                    BatchLayerGraph::Full {
                        pre_attention,
                        post_attention,
                    } => {
                        let Qwen36Attention::FullAttention(weights) = &block.attention else {
                            unreachable!("full-attention graph matches its layer")
                        };
                        pre_attention.launch(stream)?;
                        weights.enqueue_batch_cache(
                            self,
                            &mut workspace.full,
                            rows,
                            layer_idx,
                            active_rows,
                            stream,
                        )?;
                        post_attention.launch(stream)?;
                    }
                }
                std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
                continue;
            }
            rms_norm_f32_into_on_stream(
                workspace.capacity,
                self.manifest.hidden,
                &workspace.hidden,
                &block.input_norm,
                workspace.normed_hidden.output(),
                self.manifest.rms_eps,
                stream,
            )?;

            let attention_output = match (&block.attention, block.kind) {
                (Qwen36Attention::LinearAttention(weights), _) => {
                    weights.enqueue_batch(
                        self,
                        &mut workspace.linear,
                        &workspace.normed_hidden,
                        layer_idx,
                        workspace.capacity,
                        stream,
                    )?;
                    &workspace.linear.output
                }
                (Qwen36Attention::FullAttention(weights), _) => {
                    weights.enqueue_batch_pre(
                        self,
                        &mut workspace.full,
                        &workspace.normed_hidden,
                        &workspace.positions,
                        workspace.capacity,
                        stream,
                    )?;
                    weights.enqueue_batch_cache(
                        self,
                        &mut workspace.full,
                        rows,
                        layer_idx,
                        active_rows,
                        stream,
                    )?;
                    weights.enqueue_batch_post(
                        self,
                        &mut workspace.full,
                        workspace.capacity,
                        stream,
                    )?;
                    &workspace.full.output
                }
            };
            let moe_sync = &workspace.moe_stream_sync[layer_idx];
            block.enqueue_batch_tail(
                self,
                &mut workspace.moe,
                &workspace.hidden,
                attention_output,
                &mut workspace.attn_residual,
                &mut workspace.ffn_norm,
                workspace.capacity,
                stream,
                Some(Qwen36ParallelMoe {
                    shared_stream: &workspace.shared_moe_stream,
                    fork: &moe_sync.fork,
                    join: &moe_sync.join,
                }),
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
        }

        rms_norm_f32_into_on_stream(
            workspace.capacity,
            self.manifest.hidden,
            &workspace.hidden,
            &self.final_norm,
            workspace.final_hidden.output(),
            self.manifest.rms_eps,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.final_hidden.inout(), stream)?;
        match &self.lm_head {
            Qwen36LmHead::Nvfp4(linear) => linear.run_f32_batch_into(
                &workspace.final_hidden,
                &mut workspace.logits,
                workspace.capacity,
                stream,
            )?,
            Qwen36LmHead::Fp8 { linear, .. } => {
                quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
                    &workspace.final_hidden,
                    &mut workspace.lm_head_quantized,
                    &mut workspace.lm_head_scale,
                    workspace.capacity,
                    self.manifest.hidden,
                    stream,
                )?;
                run_fp8_batch(
                    self,
                    linear,
                    workspace
                        .lm_head_plan
                        .as_mut()
                        .expect("FP8 lm head has a batch plan"),
                    &workspace.final_hidden,
                    &workspace.lm_head_quantized,
                    &workspace.lm_head_scale,
                    &mut workspace.logits,
                    workspace.capacity,
                    256,
                    stream,
                )?;
            }
        }
        for row in rows.iter_mut() {
            row.state.position += 1;
        }
        Ok(Qwen36DecodedBatch {
            workspace,
            rows: active_rows,
            vocab: self.manifest.vocab,
        })
    }
}

impl Qwen36LinearAttentionWeights {
    fn enqueue_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        layer_idx: usize,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            capacity,
            model.manifest.hidden,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.qkv,
            &mut workspace.qkv_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.qkv_output,
            capacity,
            128,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.z,
            &mut workspace.z_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.z_output,
            capacity,
            128,
            stream,
        )?;
        bf16_linear_logits_f32_batch_into_on_stream(
            hidden,
            &self.alpha.weight,
            workspace.alpha.output(),
            capacity,
            self.alpha.rows,
            self.alpha.cols,
            stream,
        )?;
        bf16_linear_logits_f32_batch_into_on_stream(
            hidden,
            &self.beta.weight,
            workspace.beta_input.output(),
            capacity,
            self.beta.rows,
            self.beta.cols,
            stream,
        )?;
        qwen36_gdn_prep_batch_into_on_stream(
            &workspace.qkv_output,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            &workspace.conv_state_table,
            layer_idx * capacity,
            capacity,
            linear.key_heads,
            linear.value_heads,
            linear.value_head_dim,
            stream,
        )?;
        qwen36_gdn_gate_batch_into_on_stream(
            &workspace.alpha,
            &workspace.beta_input,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            capacity,
            linear.value_heads,
            stream,
        )?;
        gated_delta_net_128_f32_batch_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            &workspace.recurrent_state_table,
            workspace.gdn_output.output(),
            layer_idx * capacity,
            capacity,
            linear.value_heads,
            stream,
        )?;
        gated_rms_norm_f32_into_on_stream(
            &workspace.gdn_output,
            &workspace.z_output,
            &self.norm_weight,
            workspace.normed.output(),
            capacity * linear.value_heads,
            linear.value_head_dim,
            model.manifest.rms_eps,
            stream,
        )?;
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &workspace.normed,
            &mut workspace.value_quantized,
            &mut workspace.value_scale,
            capacity,
            value_dim,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.out,
            &mut workspace.out_plan,
            &workspace.normed,
            &workspace.value_quantized,
            &workspace.value_scale,
            &mut workspace.output,
            capacity,
            256,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_chunks(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        layer_idx: usize,
        sequence_capacity: usize,
        sequence_count: usize,
        total_tokens: usize,
        row_capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            row_capacity,
            model.manifest.hidden,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.qkv,
            &mut workspace.qkv_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.qkv_output,
            row_capacity,
            128,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.z,
            &mut workspace.z_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.z_output,
            row_capacity,
            128,
            stream,
        )?;
        bf16_linear_logits_f32_batch_into_on_stream(
            hidden,
            &self.alpha.weight,
            workspace.alpha.output(),
            row_capacity,
            self.alpha.rows,
            self.alpha.cols,
            stream,
        )?;
        bf16_linear_logits_f32_batch_into_on_stream(
            hidden,
            &self.beta.weight,
            workspace.beta_input.output(),
            row_capacity,
            self.beta.rows,
            self.beta.cols,
            stream,
        )?;
        if total_tokens == sequence_count {
            qwen36_gdn_prep_batch_into_on_stream(
                &workspace.qkv_output,
                &self.conv_weight,
                workspace.q.output(),
                workspace.k.output(),
                workspace.v.output(),
                &workspace.conv_state_table,
                layer_idx * sequence_capacity,
                sequence_count,
                linear.key_heads,
                linear.value_heads,
                linear.value_head_dim,
                stream,
            )?;
        } else {
            qwen36_gdn_prep_chunks_into_on_stream(
                &workspace.qkv_output,
                &self.conv_weight,
                workspace.q.output(),
                workspace.k.output(),
                workspace.v.output(),
                &workspace.conv_state_table,
                layer_idx * sequence_capacity,
                sequence_offsets,
                sequence_lengths,
                sequence_count,
                total_tokens,
                linear.key_heads,
                linear.value_heads,
                linear.value_head_dim,
                stream,
            )?;
        }
        qwen36_gdn_gate_batch_into_on_stream(
            &workspace.alpha,
            &workspace.beta_input,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            row_capacity,
            linear.value_heads,
            stream,
        )?;
        if total_tokens == sequence_count {
            gated_delta_net_128_f32_batch_into_on_stream(
                &workspace.q,
                &workspace.k,
                &workspace.v,
                &workspace.gate,
                &workspace.beta,
                &workspace.recurrent_state_table,
                workspace.gdn_output.output(),
                layer_idx * sequence_capacity,
                sequence_count,
                linear.value_heads,
                stream,
            )?;
        } else {
            gated_delta_net_128_f32_chunks_into_on_stream(
                &workspace.q,
                &workspace.k,
                &workspace.v,
                &workspace.gate,
                &workspace.beta,
                &workspace.recurrent_state_table,
                layer_idx * sequence_capacity,
                sequence_offsets,
                sequence_lengths,
                workspace.gdn_output.output(),
                sequence_count,
                total_tokens,
                linear.value_heads,
                stream,
            )?;
        }
        gated_rms_norm_f32_into_on_stream(
            &workspace.gdn_output,
            &workspace.z_output,
            &self.norm_weight,
            workspace.normed.output(),
            row_capacity * linear.value_heads,
            linear.value_head_dim,
            model.manifest.rms_eps,
            stream,
        )?;
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &workspace.normed,
            &mut workspace.value_quantized,
            &mut workspace.value_scale,
            row_capacity,
            value_dim,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.out,
            &mut workspace.out_plan,
            &workspace.normed,
            &workspace.value_quantized,
            &workspace.value_scale,
            &mut workspace.output,
            row_capacity,
            256,
            stream,
        )
    }
}

impl Qwen36FullAttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_pre(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            capacity,
            model.manifest.hidden,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.q,
            &mut workspace.q_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.q_proj,
            capacity,
            128,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.k,
            &mut workspace.k_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.k_raw,
            capacity,
            128,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.v,
            &mut workspace.v_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            &mut workspace.v,
            capacity,
            128,
            stream,
        )?;
        qwen36_full_attn_prep_f32_batch_into_on_stream(
            &workspace.q_proj,
            &workspace.k_raw,
            &self.q_norm_weight,
            &self.k_norm_weight,
            workspace.q.output(),
            workspace.gate.output(),
            workspace.k.output(),
            capacity,
            model.manifest.q_heads,
            model.manifest.kv_heads,
            model.manifest.head_dim,
            model.manifest.rms_eps,
            stream,
        )?;
        let sections =
            model
                .manifest
                .mrope_sections
                .ok_or_else(|| crate::nvfp4::Error::Format {
                    label: "Qwen3.6 batched IMRoPE",
                    detail: "mrope_sections not set in manifest".to_string(),
                })?;
        let sections = MropeSections {
            v0: sections[0],
            v1: sections[1],
            v2: sections[2],
            v3: sections[3],
        };
        rope_imrope_text_batch_f32_into_on_stream(
            capacity,
            model.manifest.q_heads,
            model.manifest.head_dim,
            model.manifest.rotary_dim,
            sections,
            positions,
            &workspace.q,
            workspace.q_rope.output(),
            model.manifest.rope_theta,
            stream,
        )?;
        rope_imrope_text_batch_f32_into_on_stream(
            capacity,
            model.manifest.kv_heads,
            model.manifest.head_dim,
            model.manifest.rotary_dim,
            sections,
            positions,
            &workspace.k,
            workspace.k_rope.output(),
            model.manifest.rope_theta,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_cache(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_idx: usize,
        active_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
        for (row, decode_row) in rows.iter_mut().enumerate().take(active_rows) {
            let state = match &mut decode_row.state.layer_states[layer_idx].attention {
                Qwen36AttentionState::FullAttention(state) => state,
                Qwen36AttentionState::LinearAttention(_) => {
                    unreachable!("layer kind validated when sequence state was created")
                }
            };
            let position = state.compact_cache.len();
            state.compact_cache.append_at_offsets_on_stream(
                &workspace.k_rope,
                row * kv_width,
                &workspace.v,
                row * kv_width,
                position,
                stream,
            )?;
            workspace
                .compact_attention
                .attention_offsets_into_on_stream(
                    &state.compact_cache,
                    &workspace.q_rope,
                    row * q_width,
                    workspace.attention.output(),
                    row * q_width,
                    stream,
                )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_cache(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        row_offsets: &[u32],
        layer_idx: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        for (sequence, row) in rows.iter_mut().enumerate() {
            let state = match &mut row.state.layer_states[layer_idx].attention {
                Qwen36AttentionState::FullAttention(state) => state,
                Qwen36AttentionState::LinearAttention(_) => {
                    unreachable!("layer kind validated when sequence state was created")
                }
            };
            let row_offset = row_offsets[sequence] as usize;
            if row.token_ids.len() == 1 {
                let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
                let q_width = model.manifest.q_heads * model.manifest.head_dim;
                let position = state.compact_cache.len();
                state.compact_cache.append_at_offsets_on_stream(
                    &workspace.k_rope,
                    row_offset * kv_width,
                    &workspace.v,
                    row_offset * kv_width,
                    position,
                    stream,
                )?;
                workspace
                    .compact_attention
                    .attention_offsets_into_on_stream(
                        &state.compact_cache,
                        &workspace.q_rope,
                        row_offset * q_width,
                        workspace.attention.output(),
                        row_offset * q_width,
                        stream,
                    )?;
            } else {
                workspace
                    .compact_attention
                    .append_causal_rows_at_offset_into_on_stream(
                        &mut state.compact_cache,
                        &workspace.q_rope,
                        &workspace.k_rope,
                        &workspace.v,
                        row_offset,
                        row.token_ids.len(),
                        workspace.attention.output(),
                        stream,
                    )?;
            }
        }
        Ok(())
    }

    fn enqueue_batch_post(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        sigmoid_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.attention,
            workspace.gated_attention.output(),
            stream,
        )?;
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &workspace.gated_attention,
            &mut workspace.value_quantized,
            &mut workspace.value_scale,
            capacity,
            q_width,
            stream,
        )?;
        run_fp8_batch(
            model,
            &self.o,
            &mut workspace.o_plan,
            &workspace.gated_attention,
            &workspace.value_quantized,
            &workspace.value_scale,
            &mut workspace.output,
            capacity,
            256,
            stream,
        )
    }
}

impl Qwen36MoeWeights {
    fn enqueue_shared_batch(
        &self,
        workspace: &mut BatchMoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.shared {
            Qwen36SharedExpertStorage::Nvfp4(shared) => {
                shared.gate_up.run_f32_batch_into(
                    ffn_norm,
                    &mut workspace.shared_gate_up,
                    capacity,
                    stream,
                )?;
                silu_mul_halves_f32_batch_into_on_stream(
                    &workspace.shared_gate_up,
                    workspace.shared_activated.output(),
                    capacity,
                    self.expert_intermediate,
                    stream,
                )?;
                shared.down.run_f32_batch_into(
                    &workspace.shared_activated,
                    &mut workspace.shared_output,
                    capacity,
                    stream,
                )?;
            }
            Qwen36SharedExpertStorage::Fp8 { .. } => {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 batched shared expert",
                    detail: "the current model does not use NVFP4 shared experts".to_string(),
                });
            }
        }
        bf16_linear_logits_f32_batch_into_on_stream(
            ffn_norm,
            &self.shared_gate.weight,
            workspace.shared_gate.output(),
            capacity,
            self.shared_gate.rows,
            self.shared_gate.cols,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchMoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
    ) -> Result<()> {
        if let Some(parallel_moe) = parallel_moe {
            parallel_moe.fork.record_on_stream(stream)?;
            parallel_moe.shared_stream.wait_event(parallel_moe.fork)?;
            self.enqueue_shared_batch(workspace, ffn_norm, capacity, parallel_moe.shared_stream)?;
        }

        bf16_linear_logits_f32_batch_into_on_stream(
            ffn_norm,
            &self.router.weight,
            workspace.router_logits.output(),
            capacity,
            self.router.rows,
            self.router.cols,
            stream,
        )?;
        moe_topk_f32_batch_into_on_stream(
            &workspace.router_logits,
            workspace.route_indices.output(),
            workspace.route_weights.output(),
            capacity,
            self.num_experts,
            self.experts_per_token,
            self.norm_topk_prob,
            stream,
        )?;
        let Qwen36GateUpStorage::Marlin(marlin) = &self.gate_up_storage else {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 batched routed gate/up",
                detail: "workspace and layer storage disagree".to_string(),
            });
        };
        marlin.run_batch_bf16_on_stream(
            &workspace.marlin,
            &workspace.route_indices,
            ffn_norm,
            stream,
        )?;
        moe_silu_quantize_bf16_slots_on_stream(
            &workspace.route_indices,
            workspace.marlin.output_bf16(),
            &mut workspace.sm12x_down.b_tiles,
            &mut workspace.sm12x_down.b_scales,
            &self.expert_ptrs.down_input_scales,
            &self.gate_up_w4a16_unity_alphas,
            self.expert_intermediate,
            capacity * self.experts_per_token,
            stream,
        )?;
        indexed_grouped_gemv_on_stream(
            &workspace.route_indices,
            self.sm12x_down_tiles
                .as_ref()
                .expect("SM12x routed down tiles"),
            self.sm12x_down_scales
                .as_ref()
                .expect("SM12x routed down scales"),
            self.num_experts,
            &workspace.sm12x_down.b_tiles,
            &workspace.sm12x_down.b_scales,
            &workspace.sm12x_down.d,
            self.sm12x_down_m_tiles,
            self.sm12x_down_k_tiles,
            capacity * self.experts_per_token,
            stream,
        )?;
        if let Some(parallel_moe) = parallel_moe {
            parallel_moe
                .join
                .record_on_stream(parallel_moe.shared_stream)?;
            stream.wait_event(parallel_moe.join)?;
        } else {
            self.enqueue_shared_batch(workspace, ffn_norm, capacity, stream)?;
        }
        qwen36_ffn_finalize_routed_batch_f32_into_on_stream(
            &workspace.route_indices,
            &workspace.route_weights,
            &workspace.sm12x_down.c,
            &self.expert_ptrs.down_alphas,
            &workspace.shared_gate,
            &workspace.shared_output,
            residual,
            workspace.output.output(),
            capacity,
            model.manifest.hidden,
            self.experts_per_token,
            stream,
        )
    }
}
