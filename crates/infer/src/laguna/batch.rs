use super::*;
use crate::metrics::metrics;
use crate::paged_prefill_attention::PagedTensorCorePrefillAttention;
use crate::runtime::laguna_sequence_cache::{
    LagunaAppend, LagunaSequence, LagunaSequenceCache, laguna_cache_error,
};
use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;
use nvfp4::{
    Bf16TnMatmulPlan, CublasLt, CutlassFp4GroupedGemmPlan, GemmShape, MoeSortedNvfp4Rows,
    MoeSortedRoutes, add_f32_prefix_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, f32_to_bf16_prefix_into_on_stream,
    fill_f32_prefix_into_on_stream, moe_silu_quantize_bf16_expert_sorted_slots_on_stream,
    moe_weighted_accumulate_sorted_slots_f32_batch_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream,
    rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream, silu_mul_f32_prefix_into_on_stream,
    softplus_scale_heads_f32_prefix_into_on_stream,
};
use std::collections::HashMap;

const CUBLAS_WORKSPACE_LIMIT: u64 = 32 << 20;
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
    pub sequence: &'state mut LagunaSequence,
}

struct LagunaPrefillStateRow<'tokens, 'state> {
    token_ids: &'tokens [u32],
    state: &'state mut LagunaDecodeState,
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
    tensor_core: PagedTensorCorePrefillAttention,
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
            tensor_core: PagedTensorCorePrefillAttention::new(
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
        cache: &mut LagunaSequenceCache,
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
                            .map_err(laguna_cache_error)?;
                    }
                    return Err(laguna_cache_error(error));
                }
            }
        }
        let result = {
            let mut state_rows = Vec::with_capacity(rows.len());
            let mut appends = Vec::with_capacity(rows.len());
            for (row, reservation) in rows.iter_mut().zip(&reservations) {
                let sequence = &mut *row.sequence;
                state_rows.push(LagunaPrefillStateRow {
                    token_ids: row.token_ids,
                    state: &mut sequence.state,
                });
                appends.push(LagunaAppend {
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
                    .map_err(laguna_cache_error)?;
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
                .map_err(laguna_cache_error)?;
            row.sequence.state.position += tokens;
        }
        Ok(())
    }

    fn prefill_batch_impl(
        &self,
        workspace: &mut LagunaPrefillBatchWorkspace,
        rows: &mut [LagunaPrefillStateRow<'_, '_>],
        cache: &mut LagunaSequenceCache,
        appends: &[LagunaAppend<'_>],
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
                cache,
                appends,
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
        }
        debug_assert!(total_tokens <= workspace.token_capacity);
        Ok(())
    }
}

fn validate_rows(
    model: &LagunaModel,
    workspace: &LagunaPrefillBatchWorkspace,
    rows: &[LagunaPrefillStateRow<'_, '_>],
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
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_layer_prefill(
    layer: &LagunaLayer,
    layer_index: usize,
    workspace: &mut LagunaPrefillBatchWorkspace,
    rows: &mut [LagunaPrefillStateRow<'_, '_>],
    active_tokens: usize,
    stream: &CudaStream,
    cache: &mut LagunaSequenceCache,
    appends: &[LagunaAppend<'_>],
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
        cache,
        appends,
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
    rows: &mut [LagunaPrefillStateRow<'_, '_>],
    layer_index: usize,
    capacity: usize,
    stream: &CudaStream,
    cache: &mut LagunaSequenceCache,
    appends: &[LagunaAppend<'_>],
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
    for (row, append) in rows.iter_mut().zip(appends) {
        let position = row.state.position;
        if append.reservation.start_position() != position
            || append.reservation.rows() != row.token_ids.len()
        {
            return Err(Error::Format {
                label: "Laguna prefill append",
                detail: "reservation does not match the prompt chunk".to_string(),
            });
        }
        cache
            .with_append_pages(append.reservation, |backend, pages| {
                let pool = backend.pool_mut(layer_index)?;
                for page in pages.iter() {
                    let segment = page.segment();
                    let mut processed = 0;
                    while processed < segment.rows() {
                        let token = segment.input_offset() + processed;
                        let query_position = position + token;
                        let chunk_rows = (segment.rows() - processed).min(16 - query_position % 16);
                        pool.append_rows_at_offset_on_stream(
                            page.page().slot(),
                            segment.page_offset() + processed,
                            &workspace.k_rope,
                            &workspace.v,
                            row_offset + token,
                            chunk_rows,
                            stream,
                        )?;
                        processed += chunk_rows;
                    }
                }
                if use_compact_prefill_attention(position, row.token_ids.len()) {
                    for page in pages.iter() {
                        let segment = page.segment();
                        let mut processed = 0;
                        while processed < segment.rows() {
                            let token = segment.input_offset() + processed;
                            let query_position = position + token;
                            let chunk_rows =
                                (segment.rows() - processed).min(16 - query_position % 16);
                            workspace
                                .compact
                                .attention_paged_causal_rows_at_offset_into_on_stream(
                                    pool,
                                    append.page_table,
                                    query_position,
                                    &workspace.q_rope,
                                    row_offset + token,
                                    chunk_rows,
                                    attention.window,
                                    workspace.attended.output(),
                                    stream,
                                )?;
                            processed += chunk_rows;
                        }
                    }
                } else {
                    workspace.tensor_core.run(
                        pool,
                        append.page_table,
                        position,
                        &workspace.q_rope,
                        row_offset,
                        row.token_ids.len(),
                        attention.window,
                        &mut workspace.attended,
                        stream,
                    )?;
                }
                Ok(())
            })
            .map_err(laguna_cache_error)?;
        if use_compact_prefill_attention(position, row.token_ids.len()) {
            metrics()
                .laguna_compact_prefill_attention_rows
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
mod tests {
    use super::use_compact_prefill_attention;

    #[test]
    fn tensor_core_attention_is_reserved_for_large_initial_chunks() {
        assert!(!use_compact_prefill_attention(0, 32));
        assert!(!use_compact_prefill_attention(0, 2_048));
        assert!(use_compact_prefill_attention(0, 31));
        assert!(use_compact_prefill_attention(128, 2_048));
    }
}
