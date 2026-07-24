use super::*;
use crate::metrics::metrics;
use nvfp4::{
    Bf16TnMatmulPlan, CublasLt, CutlassFp4GroupedGemmPlan, GemmShape, MoeSortedNvfp4Rows,
    MoeSortedRoutes, add_f32_prefix_into_on_stream, causal_window_softmax_f32_to_bf16_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, f32_to_bf16_prefix_into_on_stream,
    fill_f32_prefix_into_on_stream, moe_silu_quantize_bf16_expert_sorted_slots_on_stream,
    moe_weighted_accumulate_sorted_slots_f32_batch_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream,
    pack_token_heads_bf16_at_offset_into_on_stream,
    rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream, silu_mul_f32_prefix_into_on_stream,
    softplus_scale_heads_f32_prefix_into_on_stream, unpack_heads_f32_at_offset_into_on_stream,
};
use std::collections::HashMap;
use std::mem::size_of;

const CUBLAS_WORKSPACE_LIMIT: u64 = 32 << 20;
const PREFILL_ATTENTION_WORKSPACE_LIMIT: u64 = 4 << 20;
const ATTENTION_SCORE_BUDGET_BYTES: usize = 192 << 20;
const ATTENTION_QUERY_TILE_ROWS: usize = 256;
const TENSOR_CORE_ATTENTION_MIN_ROWS: usize = 32;
const MAX_Q_HEADS: usize = 72;

fn use_compact_prefill_attention(start_position: usize, query_rows: usize) -> bool {
    start_position != 0 || query_rows < TENSOR_CORE_ATTENTION_MIN_ROWS
}

/// One ragged Laguna prompt chunk and its persistent sequence state.
pub struct LagunaPrefillRow<'tokens, 'state> {
    /// Tokens to append to this sequence.
    pub token_ids: &'tokens [u32],
    /// Sequence state receiving the new K/V rows.
    pub state: &'state mut LagunaDecodeState,
}

struct LagunaBatchLinearWorkspace {
    lt: CublasLt,
    input_bf16: DeviceBuffer<u16>,
    plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
}

impl LagunaBatchLinearWorkspace {
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
        linear: &Bf16Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let input_len = rows.checked_mul(linear.cols).ok_or_else(|| Error::Shape {
            label: "Laguna prefill BF16 input",
            expected: "rows * input features without overflow".to_string(),
            actual: format!("rows={rows} input_features={}", linear.cols),
        })?;
        let output_len = rows.checked_mul(linear.rows).ok_or_else(|| Error::Shape {
            label: "Laguna prefill BF16 output",
            expected: "rows * output features without overflow".to_string(),
            actual: format!("rows={rows} output_features={}", linear.rows),
        })?;
        if input.len() < input_len || output.len() < output_len || input_len > self.input_bf16.len()
        {
            return Err(Error::Shape {
                label: "Laguna prefill BF16 buffers",
                expected: format!("input={input_len} output={output_len} scratch>={input_len}"),
                actual: format!(
                    "input={} output={} scratch={}",
                    input.len(),
                    output.len(),
                    self.input_bf16.len()
                ),
            });
        }
        f32_to_bf16_prefix_into_on_stream(input, self.input_bf16.output(), input_len, stream)?;
        let key = (linear.rows, rows, linear.cols);
        if !self.plans.contains_key(&key) {
            self.plans.insert(
                key,
                Bf16TnMatmulPlan::new(
                    &self.lt,
                    GemmShape::new(linear.rows, rows, linear.cols),
                    CUBLAS_WORKSPACE_LIMIT,
                )?,
            );
        }
        self.plans[&key]
            .run_on_stream(
                &self.lt,
                &linear.weight,
                &self.input_bf16,
                output.output(),
                stream,
            )
            .map_err(|error| Error::Format {
                label: "Laguna batched BF16 linear execution",
                detail: format!(
                    "{} [{}, {}] rows={rows}: {error}",
                    linear.name, linear.rows, linear.cols
                ),
            })?;
        round_f32_to_bf16_prefix_in_place_on_stream(output.inout(), output_len, stream)
    }
}

struct LagunaBatchAttentionWorkspace {
    q_heads: usize,
    compact: Sm12xKvAttentionWorkspace,
    tensor_core: LagunaTensorCoreAttentionWorkspace,
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

impl LagunaBatchAttentionWorkspace {
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
                16,
            )?,
            tensor_core: LagunaTensorCoreAttentionWorkspace::new(
                token_capacity,
                q_heads,
                KV_HEADS,
                HEAD_DIM,
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

struct LagunaTensorCoreAttentionWorkspace {
    lt: CublasLt,
    qk_plans: HashMap<(usize, usize), Bf16TnMatmulPlan>,
    pv_plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    packed_query: DeviceBuffer<u16>,
    packed_key: DeviceBuffer<u16>,
    packed_value: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    packed_probabilities: DeviceBuffer<u16>,
    packed_output: DeviceBuffer<f32>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl LagunaTensorCoreAttentionWorkspace {
    fn new(rows: usize, q_heads: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        let q_values = rows * q_heads * head_dim;
        let kv_values = rows * kv_heads * head_dim;
        Ok(Self {
            lt: CublasLt::new()?,
            qk_plans: HashMap::new(),
            pv_plans: HashMap::new(),
            packed_query: DeviceBuffer::zeroed(q_values)?,
            packed_key: DeviceBuffer::zeroed(kv_values)?,
            packed_value: DeviceBuffer::zeroed(kv_values)?,
            scores: DeviceBuffer::zeroed(rows * q_heads * rows)?,
            packed_probabilities: DeviceBuffer::zeroed(rows * q_heads * rows)?,
            packed_output: DeviceBuffer::zeroed(q_values)?,
            q_heads,
            kv_heads,
            head_dim,
        })
    }

    fn tile_rows(&self, requested: usize, key_tokens: usize) -> usize {
        let values_per_row = self.q_heads.saturating_mul(key_tokens).max(1);
        let budget_rows = (ATTENTION_SCORE_BUDGET_BYTES / size_of::<f32>())
            .checked_div(values_per_row)
            .unwrap_or(0)
            .max(1);
        let rows = requested.min(budget_rows).min(ATTENTION_QUERY_TILE_ROWS);
        if rows >= 16 { rows / 16 * 16 } else { rows }
    }

    fn ensure_capacity(
        &mut self,
        query_rows: usize,
        cache_tokens: usize,
        score_key_tokens: usize,
    ) -> Result<()> {
        grow_device_buffer(
            &mut self.packed_query,
            query_rows * self.q_heads * self.head_dim,
        )?;
        let kv_values = cache_tokens * self.kv_heads * self.head_dim;
        grow_device_buffer(&mut self.packed_key, kv_values)?;
        grow_device_buffer(&mut self.packed_value, kv_values)?;
        let score_values = query_rows * self.q_heads * score_key_tokens;
        grow_device_buffer(&mut self.scores, score_values)?;
        grow_device_buffer(&mut self.packed_probabilities, score_values)?;
        grow_device_buffer(
            &mut self.packed_output,
            query_rows * self.q_heads * self.head_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sequence(
        &mut self,
        cache: &mut Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        window_tokens: Option<usize>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let start_position = cache.len();
        let cache_tokens = start_position + rows;
        let cache_values = cache_tokens * self.kv_heads * self.head_dim;
        grow_device_buffer(&mut self.packed_key, cache_values)?;
        grow_device_buffer(&mut self.packed_value, cache_values)?;
        if start_position == 0 {
            cache.append_initial_rows_and_stage_bf16_on_stream(
                key,
                value,
                input_row_offset,
                rows,
                self.packed_key.output(),
                self.packed_value.output(),
                stream,
            )?;
        } else {
            cache.append_rows_at_offset_on_stream(key, value, input_row_offset, rows, stream)?;
            cache.unpack_bf16_on_stream(
                self.packed_key.output(),
                self.packed_value.output(),
                stream,
            )?;
        }

        let queries_per_kv = self.q_heads / self.kv_heads;
        let mut query_offset = 0;
        while query_offset < rows {
            let requested = rows - query_offset;
            let absolute_query_start = start_position + query_offset;
            let tentative_key_start = window_tokens
                .map(|window| (absolute_query_start + 1).saturating_sub(window))
                .unwrap_or(0);
            let tentative_key_tokens = absolute_query_start + requested - tentative_key_start;
            let query_rows = self.tile_rows(requested, tentative_key_tokens);
            let key_start = window_tokens
                .map(|window| (absolute_query_start + 1).saturating_sub(window))
                .unwrap_or(0);
            let key_end = absolute_query_start + query_rows;
            let key_tokens = key_end - key_start;
            self.ensure_capacity(query_rows, cache_tokens, key_tokens)?;
            pack_token_heads_bf16_at_offset_into_on_stream(
                query,
                self.packed_query.output(),
                query_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;

            let qk_key = (key_tokens, query_rows);
            if !self.qk_plans.contains_key(&qk_key) {
                self.qk_plans.insert(
                    qk_key,
                    Bf16TnMatmulPlan::new_strided_batch(
                        &self.lt,
                        GemmShape::new(key_tokens, query_rows * queries_per_kv, self.head_dim),
                        self.kv_heads,
                        cache_tokens * self.head_dim,
                        queries_per_kv * query_rows * self.head_dim,
                        queries_per_kv * query_rows * key_tokens,
                        PREFILL_ATTENTION_WORKSPACE_LIMIT,
                    )?,
                );
            }
            self.qk_plans[&qk_key].run_offsets_on_stream(
                &self.lt,
                &self.packed_key,
                key_start * self.head_dim,
                &self.packed_query,
                0,
                self.scores.output(),
                0,
                stream,
            )?;
            causal_window_softmax_f32_to_bf16_on_stream(
                &self.scores,
                self.packed_probabilities.output(),
                query_rows,
                key_tokens,
                absolute_query_start - key_start,
                self.q_heads,
                self.head_dim,
                window_tokens,
                stream,
            )?;

            let pv_key = (key_tokens, query_rows, cache_tokens);
            if !self.pv_plans.contains_key(&pv_key) {
                self.pv_plans.insert(
                    pv_key,
                    Bf16TnMatmulPlan::new_strided_batch_with_a_leading_dimension(
                        &self.lt,
                        GemmShape::new(self.head_dim, query_rows * queries_per_kv, key_tokens),
                        cache_tokens,
                        self.kv_heads,
                        self.head_dim * cache_tokens,
                        queries_per_kv * query_rows * key_tokens,
                        queries_per_kv * query_rows * self.head_dim,
                        PREFILL_ATTENTION_WORKSPACE_LIMIT,
                    )?,
                );
            }
            self.pv_plans[&pv_key].run_offsets_on_stream(
                &self.lt,
                &self.packed_value,
                key_start,
                &self.packed_probabilities,
                0,
                self.packed_output.output(),
                0,
                stream,
            )?;
            unpack_heads_f32_at_offset_into_on_stream(
                &self.packed_output,
                output.output(),
                query_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;
            query_offset += query_rows;
        }
        Ok(())
    }
}

fn grow_device_buffer<T: Copy>(buffer: &mut DeviceBuffer<T>, required: usize) -> Result<()> {
    if buffer.len() < required {
        *buffer = DeviceBuffer::zeroed(required)?;
    }
    Ok(())
}

struct LagunaBatchMlpWorkspace {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl LagunaBatchMlpWorkspace {
    fn new(token_capacity: usize, intermediate: usize) -> Result<Self> {
        Ok(Self {
            gate: DeviceBuffer::zeroed(token_capacity * intermediate)?,
            up: DeviceBuffer::zeroed(token_capacity * intermediate)?,
            activated: DeviceBuffer::zeroed(token_capacity * intermediate)?,
            output: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
        })
    }
}

struct LagunaBatchDownWorkspace {
    b_tiles: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u32>,
    _outputs: Vec<F32Matrix>,
    inputs: DeviceBuffer<*const f32>,
    outputs: DeviceBuffer<*mut f32>,
}

impl LagunaBatchDownWorkspace {
    fn new(routes: usize) -> Result<Self> {
        let mut matrices = Vec::with_capacity(routes);
        let mut inputs = Vec::with_capacity(routes);
        let mut outputs = Vec::with_capacity(routes);
        for _ in 0..routes {
            let mut output = F32Matrix::zeroed(HIDDEN, 1)?;
            inputs.push(output.data_ptr());
            outputs.push(output.data_mut_ptr());
            matrices.push(output);
        }
        Ok(Self {
            b_tiles: DeviceBuffer::zeroed(routes * (EXPERT_INTERMEDIATE / 64) * 512)?,
            b_scales: DeviceBuffer::zeroed(routes * (EXPERT_INTERMEDIATE / 64))?,
            _outputs: matrices,
            inputs: DeviceBuffer::from_host(&inputs)?,
            outputs: DeviceBuffer::from_host(&outputs)?,
        })
    }
}

struct LagunaBatchMoeWorkspace {
    router_logits: DeviceBuffer<f32>,
    route_indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    sorted_routes: MoeSortedRoutes,
    gate_up_input: MoeSortedNvfp4Rows,
    gate_up_plan: CutlassFp4GroupedGemmPlan,
    gate_up_output: DeviceBuffer<u16>,
    gate_up_output_table: DeviceBuffer<*mut u16>,
    down: LagunaBatchDownWorkspace,
    routed: DeviceBuffer<f32>,
    shared: LagunaBatchMlpWorkspace,
    output: DeviceBuffer<f32>,
}

impl LagunaBatchMoeWorkspace {
    fn new(token_capacity: usize) -> Result<Self> {
        let routes = token_capacity * TOP_K;
        Ok(Self {
            router_logits: DeviceBuffer::zeroed(token_capacity * EXPERTS)?,
            route_indices: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            sorted_routes: MoeSortedRoutes::new(routes, EXPERTS)?,
            gate_up_input: MoeSortedNvfp4Rows::new(token_capacity, TOP_K, EXPERTS, HIDDEN)?,
            gate_up_plan: CutlassFp4GroupedGemmPlan::new(
                EXPERT_INTERMEDIATE * 2,
                routes,
                HIDDEN,
                EXPERTS,
            )?,
            gate_up_output: DeviceBuffer::zeroed(routes * EXPERT_INTERMEDIATE * 2)?,
            gate_up_output_table: DeviceBuffer::zeroed(EXPERTS)?,
            down: LagunaBatchDownWorkspace::new(routes)?,
            routed: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            shared: LagunaBatchMlpWorkspace::new(token_capacity, SHARED_INTERMEDIATE)?,
            output: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
        })
    }
}

/// Reusable layer-major Laguna prompt workspace shared across active sessions.
pub struct LagunaPrefillBatchWorkspace {
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
    token_ids: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layer_output: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    full_attention: LagunaBatchAttentionWorkspace,
    sliding_attention: LagunaBatchAttentionWorkspace,
    post_attention: DeviceBuffer<f32>,
    ffn_input: DeviceBuffer<f32>,
    dense: LagunaBatchMlpWorkspace,
    moe: LagunaBatchMoeWorkspace,
    linear: LagunaBatchLinearWorkspace,
}

impl LagunaModel {
    /// Allocates reusable layer-major prompt execution storage.
    pub fn new_prefill_batch_workspace(
        &self,
        sequence_capacity: usize,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<LagunaPrefillBatchWorkspace> {
        if sequence_capacity == 0 || token_capacity == 0 || max_context_tokens == 0 {
            return Err(Error::Shape {
                label: "Laguna prefill workspace",
                expected: "non-zero sequence, token, and context capacities".to_string(),
                actual: format!(
                    "sequences={sequence_capacity} tokens={token_capacity} context={max_context_tokens}"
                ),
            });
        }
        Ok(LagunaPrefillBatchWorkspace {
            sequence_capacity,
            token_capacity,
            max_context_tokens,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            hidden: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            layer_output: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            normed: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            full_attention: LagunaBatchAttentionWorkspace::new(
                token_capacity,
                max_context_tokens,
                48,
            )?,
            sliding_attention: LagunaBatchAttentionWorkspace::new(
                token_capacity,
                max_context_tokens,
                72,
            )?,
            post_attention: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            ffn_input: DeviceBuffer::zeroed(token_capacity * HIDDEN)?,
            dense: LagunaBatchMlpWorkspace::new(token_capacity, DENSE_INTERMEDIATE)?,
            moe: LagunaBatchMoeWorkspace::new(token_capacity)?,
            linear: LagunaBatchLinearWorkspace::new(
                token_capacity,
                DENSE_INTERMEDIATE.max(MAX_Q_HEADS * HEAD_DIM),
            )?,
        })
    }

    /// Advances ragged prompt chunks with layer-major projection and MoE batching.
    pub fn prefill_batch(
        &self,
        workspace: &mut LagunaPrefillBatchWorkspace,
        rows: &mut [LagunaPrefillRow<'_, '_>],
    ) -> Result<()> {
        validate_rows(self, workspace, rows)?;
        let total_tokens = rows.iter().map(|row| row.token_ids.len()).sum::<usize>();
        let mut tokens = rows
            .iter()
            .flat_map(|row| row.token_ids.iter().copied())
            .collect::<Vec<_>>();
        tokens.resize(workspace.token_capacity, 0);
        workspace.token_ids.copy_from_host(&tokens)?;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            VOCAB,
            HIDDEN,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            total_tokens,
            &self.stream,
        )?;

        for layer_index in 0..LAYERS {
            run_layer_prefill(
                &self.layers[layer_index],
                layer_index,
                workspace,
                rows,
                total_tokens,
                &self.stream,
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
        }
        for row in rows {
            row.state.position += row.token_ids.len();
        }
        debug_assert!(total_tokens <= workspace.token_capacity);
        Ok(())
    }
}

fn validate_rows(
    model: &LagunaModel,
    workspace: &LagunaPrefillBatchWorkspace,
    rows: &[LagunaPrefillRow<'_, '_>],
) -> Result<()> {
    if rows.is_empty() || rows.len() > workspace.sequence_capacity {
        return Err(Error::Shape {
            label: "Laguna prefill rows",
            expected: format!("1..={} sequences", workspace.sequence_capacity),
            actual: rows.len().to_string(),
        });
    }
    let total_tokens = rows.iter().try_fold(0usize, |total, row| {
        total
            .checked_add(row.token_ids.len())
            .ok_or_else(|| Error::Shape {
                label: "Laguna prefill token count",
                expected: "total token count without overflow".to_string(),
                actual: format!("total={total} row={}", row.token_ids.len()),
            })
    })?;
    if total_tokens == 0 || total_tokens > workspace.token_capacity {
        return Err(Error::Shape {
            label: "Laguna prefill token count",
            expected: format!("1..={} tokens", workspace.token_capacity),
            actual: total_tokens.to_string(),
        });
    }
    for row in rows {
        if row.token_ids.is_empty() {
            return Err(Error::Format {
                label: "Laguna prefill row",
                detail: "prompt chunks must not be empty".to_string(),
            });
        }
        if row.state.model_id != model.model_id {
            return Err(Error::Format {
                label: "Laguna prefill state",
                detail: "state belongs to a different model instance".to_string(),
            });
        }
        if let Some(token) = row.token_ids.iter().find(|&&token| token as usize >= VOCAB) {
            return Err(Error::Shape {
                label: "Laguna prefill token",
                expected: format!("token < {VOCAB}"),
                actual: token.to_string(),
            });
        }
        let end = row
            .state
            .position
            .checked_add(row.token_ids.len())
            .ok_or_else(|| Error::Shape {
                label: "Laguna prefill context",
                expected: "position + tokens without overflow".to_string(),
                actual: format!(
                    "position={} tokens={}",
                    row.state.position,
                    row.token_ids.len()
                ),
            })?;
        if end > row.state.max_tokens || end > workspace.max_context_tokens {
            return Err(Error::Shape {
                label: "Laguna prefill context",
                expected: format!(
                    "end <= min({}, {})",
                    row.state.max_tokens, workspace.max_context_tokens
                ),
                actual: format!("end={end}"),
            });
        }
        if row
            .state
            .kv_cache
            .iter()
            .any(|cache| cache.len() != row.state.position)
        {
            return Err(Error::Format {
                label: "Laguna prefill state",
                detail: "layer K/V positions disagree".to_string(),
            });
        }
    }
    Ok(())
}

fn run_layer_prefill(
    layer: &LagunaLayer,
    layer_index: usize,
    workspace: &mut LagunaPrefillBatchWorkspace,
    rows: &mut [LagunaPrefillRow<'_, '_>],
    active_tokens: usize,
    stream: &CudaStream,
) -> Result<()> {
    layer.input_norm.run_into(
        &workspace.hidden,
        &mut workspace.normed,
        active_tokens,
        stream,
    )?;
    let attention_workspace = if layer.attention.q_heads == 48 {
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
        active_tokens,
        stream,
    )?;
    add_f32_prefix_into_on_stream(
        &workspace.hidden,
        &attention_workspace.output,
        workspace.post_attention.output(),
        active_tokens * HIDDEN,
        stream,
    )?;
    layer.post_attention_norm.run_into(
        &workspace.post_attention,
        &mut workspace.ffn_input,
        active_tokens,
        stream,
    )?;
    match &layer.ffn {
        LagunaFfn::Dense(mlp) => {
            run_mlp_prefill(
                mlp,
                &mut workspace.dense,
                &mut workspace.linear,
                &workspace.ffn_input,
                active_tokens,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &workspace.post_attention,
                &workspace.dense.output,
                workspace.layer_output.output(),
                active_tokens * HIDDEN,
                stream,
            )
        }
        LagunaFfn::Moe(moe) => {
            run_moe_prefill(
                moe,
                &mut workspace.moe,
                &mut workspace.linear,
                &workspace.ffn_input,
                active_tokens,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &workspace.post_attention,
                &workspace.moe.output,
                workspace.layer_output.output(),
                active_tokens * HIDDEN,
                stream,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_attention_prefill(
    attention: &LagunaAttention,
    workspace: &mut LagunaBatchAttentionWorkspace,
    linear: &mut LagunaBatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    rows: &mut [LagunaPrefillRow<'_, '_>],
    layer_index: usize,
    capacity: usize,
    stream: &CudaStream,
) -> Result<()> {
    debug_assert_eq!(workspace.q_heads, attention.q_heads);
    linear.run_bf16(&attention.q, input, &mut workspace.q, capacity, stream)?;
    linear.run_bf16(&attention.k, input, &mut workspace.k, capacity, stream)?;
    linear.run_bf16(&attention.v, input, &mut workspace.v, capacity, stream)?;
    attention.q_norm.run_into(
        &workspace.q,
        &mut workspace.q_normed,
        capacity * attention.q_heads,
        stream,
    )?;
    attention.k_norm.run_into(
        &workspace.k,
        &mut workspace.k_normed,
        capacity * KV_HEADS,
        stream,
    )?;
    let mut row_offset = 0;
    for row in rows.iter() {
        let position = row.state.position;
        rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            attention.q_heads,
            HEAD_DIM,
            attention.rotary_dim,
            &workspace.q_normed,
            &attention.inv_freq,
            workspace.q_rope.output(),
            row_offset,
            position,
            attention.rope_scale,
            stream,
        )?;
        rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            KV_HEADS,
            HEAD_DIM,
            attention.rotary_dim,
            &workspace.k_normed,
            &attention.inv_freq,
            workspace.k_rope.output(),
            row_offset,
            position,
            attention.rope_scale,
            stream,
        )?;
        row_offset += row.token_ids.len();
    }
    round_f32_to_bf16_prefix_in_place_on_stream(
        workspace.q_rope.inout(),
        capacity * attention.q_heads * HEAD_DIM,
        stream,
    )?;
    round_f32_to_bf16_prefix_in_place_on_stream(
        workspace.k_rope.inout(),
        capacity * KV_HEADS * HEAD_DIM,
        stream,
    )?;
    row_offset = 0;
    for row in rows {
        if use_compact_prefill_attention(row.state.position, row.token_ids.len()) {
            workspace
                .compact
                .append_causal_rows_at_offset_into_on_stream(
                    &mut row.state.kv_cache[layer_index],
                    &workspace.q_rope,
                    &workspace.k_rope,
                    &workspace.v,
                    row_offset,
                    row.token_ids.len(),
                    attention.window,
                    workspace.attended.output(),
                    stream,
                )?;
            metrics()
                .laguna_compact_prefill_attention_rows
                .add(row.token_ids.len().min(isize::MAX as usize) as isize);
        } else {
            workspace.tensor_core.run_sequence(
                &mut row.state.kv_cache[layer_index],
                &workspace.q_rope,
                &workspace.k_rope,
                &workspace.v,
                row_offset,
                row.token_ids.len(),
                attention.window,
                &mut workspace.attended,
                stream,
            )?;
            metrics()
                .laguna_tensor_core_prefill_attention_rows
                .add(row.token_ids.len().min(isize::MAX as usize) as isize);
        }
        row_offset += row.token_ids.len();
    }
    linear.run_bf16(
        &attention.gate,
        input,
        &mut workspace.gate,
        capacity,
        stream,
    )?;
    softplus_scale_heads_f32_prefix_into_on_stream(
        &workspace.gate,
        &workspace.attended,
        workspace.gated.output(),
        HEAD_DIM,
        capacity * attention.q_heads,
        stream,
    )?;
    round_f32_to_bf16_prefix_in_place_on_stream(
        workspace.gated.inout(),
        capacity * attention.q_heads * HEAD_DIM,
        stream,
    )?;
    linear.run_bf16(
        &attention.o,
        &workspace.gated,
        &mut workspace.output,
        capacity,
        stream,
    )
}

fn run_mlp_prefill(
    mlp: &LagunaMlp,
    workspace: &mut LagunaBatchMlpWorkspace,
    linear: &mut LagunaBatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    capacity: usize,
    stream: &CudaStream,
) -> Result<()> {
    linear.run_bf16(&mlp.gate, input, &mut workspace.gate, capacity, stream)?;
    linear.run_bf16(&mlp.up, input, &mut workspace.up, capacity, stream)?;
    silu_mul_f32_prefix_into_on_stream(
        &workspace.gate,
        &workspace.up,
        workspace.activated.output(),
        capacity * mlp.gate.rows,
        stream,
    )?;
    let active_values = capacity * mlp.gate.rows;
    round_f32_to_bf16_prefix_in_place_on_stream(
        workspace.activated.inout(),
        active_values,
        stream,
    )?;
    linear.run_bf16(
        &mlp.down,
        &workspace.activated,
        &mut workspace.output,
        capacity,
        stream,
    )
}

fn run_moe_prefill(
    moe: &LagunaMoe,
    workspace: &mut LagunaBatchMoeWorkspace,
    linear: &mut LagunaBatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    capacity: usize,
    stream: &CudaStream,
) -> Result<()> {
    linear.run_bf16(
        &moe.router,
        input,
        &mut workspace.router_logits,
        capacity,
        stream,
    )?;
    nemotron3_sigmoid_topk_f32_batch_into_on_stream(
        &workspace.router_logits,
        &moe.correction_bias,
        workspace.route_indices.output(),
        workspace.route_weights.output(),
        capacity,
        TOP_K,
        1,
        1,
        true,
        ROUTED_SCALE,
        stream,
    )?;
    workspace.sorted_routes.set_routes(capacity * TOP_K)?;
    workspace
        .sorted_routes
        .sort_on_stream(&workspace.route_indices, stream)?;
    workspace.gate_up_input.set_rows(capacity)?;
    workspace
        .gate_up_input
        .gather_quantize_on_stream(input, &workspace.sorted_routes, stream)?;
    workspace.gate_up_input.build_pointer_tables_on_stream(
        &workspace.sorted_routes,
        &mut workspace.gate_up_output,
        &mut workspace.gate_up_output_table,
        EXPERT_INTERMEDIATE * 2,
        stream,
    )?;
    workspace.gate_up_plan.run_on_stream(
        &moe.gate_up_values,
        &moe.gate_up_scales,
        workspace.gate_up_input.packed_table(),
        workspace.gate_up_input.scale_table(),
        &workspace.gate_up_output_table,
        &moe.gate_up_alpha_table,
        workspace.sorted_routes.expert_counts(),
        stream,
    )?;
    moe_silu_quantize_bf16_expert_sorted_slots_on_stream(
        workspace.sorted_routes.sorted_experts(),
        &workspace.gate_up_output,
        &mut workspace.down.b_tiles,
        &mut workspace.down.b_scales,
        &moe.down_input_scales,
        &moe.gate_up_unity_alphas,
        EXPERT_INTERMEDIATE,
        capacity * TOP_K,
        stream,
    )?;
    indexed_grouped_gemv_on_stream(
        workspace.sorted_routes.sorted_experts(),
        &moe.down_tiles,
        &moe.down_scales,
        EXPERTS,
        &workspace.down.b_tiles,
        &workspace.down.b_scales,
        &workspace.down.outputs,
        HIDDEN / 16,
        EXPERT_INTERMEDIATE / 64,
        capacity * TOP_K,
        stream,
    )?;
    fill_f32_prefix_into_on_stream(workspace.routed.output(), 0.0, capacity * HIDDEN, stream)?;
    moe_weighted_accumulate_sorted_slots_f32_batch_on_stream(
        &workspace.sorted_routes,
        &workspace.route_indices,
        &workspace.route_weights,
        &workspace.down.inputs,
        &moe.down_alphas,
        workspace.routed.inout(),
        capacity,
        TOP_K,
        HIDDEN,
        stream,
    )?;
    run_mlp_prefill(
        &moe.shared,
        &mut workspace.shared,
        linear,
        input,
        capacity,
        stream,
    )?;
    add_f32_prefix_into_on_stream(
        &workspace.routed,
        &workspace.shared.output,
        workspace.output.output(),
        capacity * HIDDEN,
        stream,
    )
}

#[cfg(test)]
pub(super) fn validate_initial_batch_layers(model: &LagunaModel, token: u32) {
    let mut serial = model.new_decode_state(32).expect("serial diagnostic state");
    serial
        .token
        .copy_from_host(&[token])
        .expect("diagnostic token");
    copy_bf16_row_to_f32_indexed_into_on_stream(
        VOCAB,
        HIDDEN,
        &model.embedding,
        &serial.token,
        serial.hidden.output(),
        &model.stream,
    )
    .expect("serial diagnostic embedding");
    let mut serial_q = DeviceBuffer::zeroed(48 * HEAD_DIM).expect("serial diagnostic q");
    model.layers[0]
        .attention
        .q
        .run_into(&serial.hidden, &mut serial_q, &model.stream)
        .expect("serial diagnostic q projection");

    let mut workspace = model
        .new_prefill_batch_workspace(1, 1, 32)
        .expect("batch diagnostic workspace");
    workspace
        .token_ids
        .copy_from_host(&[token])
        .expect("batch diagnostic token");
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
        VOCAB,
        HIDDEN,
        &model.embedding,
        &workspace.token_ids,
        workspace.hidden.output(),
        1,
        &model.stream,
    )
    .expect("batch diagnostic embedding");
    workspace
        .linear
        .run_bf16(
            &model.layers[0].attention.q,
            &workspace.hidden,
            &mut workspace.full_attention.q,
            1,
            &model.stream,
        )
        .expect("batch diagnostic q projection");
    let expected = serial_q
        .copy_to_host(&model.stream)
        .expect("serial diagnostic q host");
    let actual = workspace
        .full_attention
        .q
        .copy_to_host(&model.stream)
        .expect("batch diagnostic q host");
    let squared_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(nrmse <= 0.01, "batched q projection nrmse={nrmse:.6}");

    model.layers[0]
        .run_one(
            &mut serial.layers[0],
            &serial.hidden,
            &mut serial.kv_cache[0],
            serial.compact_attention.for_layer(0),
            0,
            &model.stream,
        )
        .expect("serial diagnostic layer");
    let mut batch_state = model.new_decode_state(32).expect("batch diagnostic state");
    run_layer_prefill(
        &model.layers[0],
        0,
        &mut workspace,
        &mut [LagunaPrefillRow {
            token_ids: &[token],
            state: &mut batch_state,
        }],
        1,
        &model.stream,
    )
    .expect("batch diagnostic layer");
    let expected = serial.layers[0]
        .output
        .copy_to_host(&model.stream)
        .expect("serial diagnostic layer host");
    let actual = workspace
        .layer_output
        .copy_to_host(&model.stream)
        .expect("batch diagnostic layer host");
    let squared_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(nrmse <= 0.05, "batched first layer nrmse={nrmse:.6}");

    let (serial_previous, serial_current) = serial.layers.split_at_mut(1);
    model.layers[1]
        .run_one(
            &mut serial_current[0],
            &serial_previous[0].output,
            &mut serial.kv_cache[1],
            serial.compact_attention.for_layer(1),
            0,
            &model.stream,
        )
        .expect("serial diagnostic second layer");
    std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
    run_layer_prefill(
        &model.layers[1],
        1,
        &mut workspace,
        &mut [LagunaPrefillRow {
            token_ids: &[token],
            state: &mut batch_state,
        }],
        1,
        &model.stream,
    )
    .expect("batch diagnostic second layer");
    let expected_attention = serial.layers[1]
        .attention_residual
        .copy_to_host(&model.stream)
        .expect("serial diagnostic second attention host");
    let actual_attention = workspace
        .post_attention
        .copy_to_host(&model.stream)
        .expect("batch diagnostic second attention host");
    let squared_error = actual_attention
        .iter()
        .zip(expected_attention.iter())
        .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected_attention
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let attention_nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        attention_nrmse <= 0.05,
        "batched second-layer attention nrmse={attention_nrmse:.6}"
    );
    let serial_route = match &serial.layers[1].ffn {
        LagunaFfnWorkspace::Moe(workspace) => workspace
            .route_indices
            .copy_to_host(&model.stream)
            .expect("serial diagnostic route host"),
        LagunaFfnWorkspace::Dense(_) => panic!("second Laguna layer should be sparse"),
    };
    let batch_route = workspace
        .moe
        .route_indices
        .copy_prefix_to_host(TOP_K, &model.stream)
        .expect("batch diagnostic route host");
    let mut serial_route_set = serial_route.to_vec();
    serial_route_set.sort_unstable();
    let mut batch_route_set = batch_route.to_vec();
    batch_route_set.sort_unstable();
    assert_eq!(
        batch_route_set, serial_route_set,
        "batched second-layer router selected different experts"
    );
    let expected = serial.layers[1]
        .output
        .copy_to_host(&model.stream)
        .expect("serial diagnostic second layer host");
    let actual = workspace
        .layer_output
        .copy_to_host(&model.stream)
        .expect("batch diagnostic second layer host");
    let squared_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(nrmse <= 0.05, "batched second layer nrmse={nrmse:.6}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_path_keeps_continuations_on_compact_storage() {
        assert!(!use_compact_prefill_attention(0, 4_096));
        assert!(!use_compact_prefill_attention(0, 32));
        assert!(use_compact_prefill_attention(0, 31));
        assert!(use_compact_prefill_attention(4_096, 1_024));
    }

    #[test]
    fn tensor_core_attention_is_stable_across_prompt_chunks() {
        const TOKENS: usize = 64;
        const CHUNK: usize = 32;
        const Q_HEADS: usize = 48;
        const MAX_TOKENS: usize = 96;

        let q_width = Q_HEADS * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        let values = |len: usize, factor: usize| {
            (0..len)
                .map(|index| ((index * factor % 251) as f32 - 125.0) / 256.0)
                .collect::<Vec<_>>()
        };
        let query = DeviceBuffer::from_host(&values(TOKENS * q_width, 17)).expect("query buffer");
        let key = DeviceBuffer::from_host(&values(TOKENS * kv_width, 29)).expect("key buffer");
        let value = DeviceBuffer::from_host(&values(TOKENS * kv_width, 43)).expect("value buffer");
        let stream = CudaStream::new_blocking().expect("stream");

        let mut whole_workspace =
            LagunaTensorCoreAttentionWorkspace::new(TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("whole workspace");
        let mut whole_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("whole cache");
        let mut whole_output = DeviceBuffer::zeroed(TOKENS * q_width).expect("whole output");
        whole_workspace
            .run_sequence(
                &mut whole_cache,
                &query,
                &key,
                &value,
                0,
                TOKENS,
                None,
                &mut whole_output,
                &stream,
            )
            .expect("whole attention");

        let mut split_workspace =
            LagunaTensorCoreAttentionWorkspace::new(TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("split workspace");
        let mut split_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("split cache");
        let mut split_output = DeviceBuffer::zeroed(TOKENS * q_width).expect("split output");
        for offset in [0, CHUNK] {
            split_workspace
                .run_sequence(
                    &mut split_cache,
                    &query,
                    &key,
                    &value,
                    offset,
                    CHUNK,
                    None,
                    &mut split_output,
                    &stream,
                )
                .expect("split attention");
        }

        let whole = whole_output
            .copy_to_host(&stream)
            .expect("whole output download");
        let split = split_output
            .copy_to_host(&stream)
            .expect("split output download");
        let squared_error = split
            .iter()
            .zip(whole.iter())
            .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
            .sum::<f64>();
        let expected_norm = whole
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
        let max_error = split
            .iter()
            .zip(whole.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            nrmse <= 0.01,
            "split tensor-core attention nrmse={nrmse:.6} max_error={max_error:.6}"
        );
    }
}
