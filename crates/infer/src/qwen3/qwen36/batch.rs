use super::{
    Fp8Linear, Qwen36Attention, Qwen36AttentionState, Qwen36DownStorage,
    Qwen36FullAttentionWeights, Qwen36GateUpStorage, Qwen36LinearAttentionState,
    Qwen36LinearAttentionWeights, Qwen36LmHead, Qwen36MoeWeights, Qwen36NextToken,
    Qwen36SequenceState, Qwen36SharedExpertStorage, Qwen36TextModel, Sm12xGateUpWorkspace,
    maybe_round_device_f32_to_bf16,
};
use crate::nvfp4::{
    CudaStream, DeviceBuffer, Fp8TnMatmulPlan, GemmShape, MarlinNvfp4GateUpBatchWorkspace,
    MropeSections, Result, Sm12xKvAttentionWorkspace, add_f32_into_on_stream,
    argmax_f32_batch_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, fill_f32_into_on_stream,
    fp8_linear_f32_batch_into_on_stream, gated_delta_net_128_f32_batch_into_on_stream,
    gated_rms_norm_f32_into_on_stream, indexed_grouped_gemv_on_stream,
    moe_silu_quantize_bf16_slots_on_stream, moe_topk_f32_batch_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream,
    qwen36_ffn_finalize_routed_batch_f32_into_on_stream,
    qwen36_full_attn_prep_f32_batch_into_on_stream, qwen36_gdn_gate_batch_into_on_stream,
    qwen36_gdn_prep_batch_into_on_stream, rms_norm_f32_into_on_stream,
    rope_imrope_text_batch_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream, sigmoid_mul_f32_into_on_stream,
    silu_mul_halves_f32_batch_into_on_stream,
};

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
        capacity: usize,
    ) -> Result<Self> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let nulls = vec![std::ptr::null_mut(); capacity];
        let mut padding_states = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            padding_states.push(Qwen36LinearAttentionState::new(linear, weights)?);
        }
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            hidden_scale: DeviceBuffer::zeroed(capacity)?,
            value_quantized: DeviceBuffer::zeroed(capacity * value_dim)?,
            value_scale: DeviceBuffer::zeroed(capacity)?,
            qkv_output: DeviceBuffer::zeroed(capacity * weights.qkv.rows)?,
            z_output: DeviceBuffer::zeroed(capacity * weights.z.rows)?,
            alpha: DeviceBuffer::zeroed(capacity * linear.value_heads)?,
            beta_input: DeviceBuffer::zeroed(capacity * linear.value_heads)?,
            gate: DeviceBuffer::zeroed(capacity * linear.value_heads)?,
            beta: DeviceBuffer::zeroed(capacity * linear.value_heads)?,
            q: DeviceBuffer::zeroed(capacity * value_dim)?,
            k: DeviceBuffer::zeroed(capacity * value_dim)?,
            v: DeviceBuffer::zeroed(capacity * value_dim)?,
            conv_state_table: DeviceBuffer::from_host(&nulls)?,
            recurrent_state_table: DeviceBuffer::from_host(&nulls)?,
            conv_state_ptrs: nulls.clone(),
            recurrent_state_ptrs: nulls,
            padding_states,
            gdn_output: DeviceBuffer::zeroed(capacity * value_dim)?,
            normed: DeviceBuffer::zeroed(capacity * value_dim)?,
            output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            qkv_plan: BatchLinearPlan::new(model, &weights.qkv, capacity)?,
            z_plan: BatchLinearPlan::new(model, &weights.z, capacity)?,
            out_plan: BatchLinearPlan::new(model, &weights.out, capacity)?,
        })
    }

    fn update_state_tables(
        &mut self,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_idx: usize,
        capacity: usize,
    ) -> Result<()> {
        for row in 0..capacity {
            let state = if let Some(row) = rows.get_mut(row) {
                match &mut row.state.layer_states[layer_idx].attention {
                    Qwen36AttentionState::LinearAttention(state) => state,
                    Qwen36AttentionState::FullAttention(_) => {
                        unreachable!("layer kind validated when sequence state was created")
                    }
                }
            } else {
                &mut self.padding_states[row]
            };
            self.conv_state_ptrs[row] = state.conv_state.as_const_ptr().cast_mut().cast::<f32>();
            self.recurrent_state_ptrs[row] = state
                .recurrent_state
                .as_const_ptr()
                .cast_mut()
                .cast::<f32>();
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
    stream: CudaStream,
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

impl Qwen36TextModel {
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
        Ok(Qwen36DecodeBatchWorkspace {
            model_id: self.model_id,
            capacity,
            max_context_tokens,
            stream: CudaStream::new_blocking()?,
            token_ids: DeviceBuffer::zeroed(capacity)?,
            positions: DeviceBuffer::zeroed(capacity)?,
            host_token_ids: vec![0; capacity],
            host_positions: vec![0; capacity],
            hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            normed_hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            attn_residual: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            final_hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            linear: BatchLinearAttentionWorkspace::new(self, first_linear, capacity)?,
            full: BatchFullAttentionWorkspace::new(self, first_full, capacity, max_context_tokens)?,
            moe: BatchMoeWorkspace::new(self, first_moe, capacity)?,
            lm_head_plan,
            lm_head_quantized: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            lm_head_scale: DeviceBuffer::zeroed(capacity)?,
            logits: DeviceBuffer::zeroed(capacity * self.manifest.vocab)?,
            next_indices: DeviceBuffer::zeroed(capacity)?,
            next_values: DeviceBuffer::zeroed(capacity)?,
        })
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
                    weights.run_batch(
                        self,
                        &mut workspace.linear,
                        rows,
                        layer_idx,
                        &workspace.normed_hidden,
                        workspace.capacity,
                        stream,
                    )?;
                    &workspace.linear.output
                }
                (Qwen36Attention::FullAttention(weights), _) => {
                    weights.run_batch(
                        self,
                        &mut workspace.full,
                        rows,
                        layer_idx,
                        &workspace.normed_hidden,
                        &workspace.positions,
                        active_rows,
                        workspace.capacity,
                        stream,
                    )?;
                    &workspace.full.output
                }
            };
            add_f32_into_on_stream(
                &workspace.hidden,
                attention_output,
                workspace.attn_residual.output(),
                stream,
            )?;
            rms_norm_f32_into_on_stream(
                workspace.capacity,
                self.manifest.hidden,
                &workspace.attn_residual,
                &block.post_attn_norm,
                workspace.ffn_norm.output(),
                self.manifest.rms_eps,
                stream,
            )?;
            block.moe.run_batch(
                self,
                &mut workspace.moe,
                &workspace.ffn_norm,
                &workspace.attn_residual,
                workspace.capacity,
                stream,
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
    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_idx: usize,
        hidden: &DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        workspace.update_state_tables(rows, layer_idx, capacity)?;
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
}

impl Qwen36FullAttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_idx: usize,
        hidden: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        active_rows: usize,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
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
        )?;
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
    fn run_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchMoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
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
        )?;
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
