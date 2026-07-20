use super::*;
use crate::metrics::metrics;
use nvfp4::{
    Bf16TnMatmulPlan, CublasLt, CutlassFp4GroupedGemmPlan, Fp4TnMatmulPlan, GemmShape,
    Gemma4LocalPrefillAttention, MoeSortedNvfp4Rows, MoeSortedRoutes, Nvfp4Matrix, Nvfp4TnInputs,
    causal_window_softmax_f32_to_bf16_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, copy_row_f32_into_on_stream,
    dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_into_on_stream,
    dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream,
    gather_indexed_mul_f32_prefix_into_on_stream,
    gelu_tanh_mul_quantize_nvfp4_col_major_f32_into_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    pack_token_heads_bf16_at_offset_into_on_stream,
    rms_norm_add_then_rms_norm_quantize_nvfp4_f32_into_on_stream,
    rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    rms_norm_quantize_nvfp4_pair_col_major_f32_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream,
    unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream,
};
use std::collections::HashMap;
use std::mem::size_of;

#[cfg(test)]
use nvfp4::quantize_nvfp4_col_major_f32_device_into_on_stream;

const PREFILL_GEMM_WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;
const ATTENTION_SCORE_BUDGET_BYTES: usize = 192 * 1024 * 1024;
const ATTENTION_QUERY_TILE_ROWS: usize = 256;
// Direct compact-cache attention overtakes whole-cache BF16 staging at a 16:1
// cached-prefix/query ratio in the integrated local-attention micromeasure.
const COMPACT_LOCAL_ATTENTION_MIN_PREFIX_PER_QUERY: usize = 16;

fn use_compact_local_attention(start_position: usize, query_rows: usize) -> bool {
    start_position != 0
        && start_position >= query_rows.saturating_mul(COMPACT_LOCAL_ATTENTION_MIN_PREFIX_PER_QUERY)
}

/// One scheduler-selected Gemma prompt chunk and its persistent sequence state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4PrefillOutput {
    None,
    FullLogits,
    Top1,
}

pub struct Gemma4PrefillRow<'tokens, 'state> {
    pub token_ids: &'tokens [u32],
    pub state: &'state mut Gemma4DecodeState,
    pub output: Gemma4PrefillOutput,
}

struct Gemma4BatchLinearWorkspace {
    capacity: usize,
    rows: usize,
    lt: CublasLt,
    activations: HashMap<usize, Nvfp4Matrix>,
    plans: HashMap<(usize, usize, usize), Gemma4BatchLinearPlan>,
}

struct Gemma4BatchLinearPlan {
    plan: Fp4TnMatmulPlan,
}

impl Gemma4BatchLinearWorkspace {
    fn new(rows: usize) -> Result<Self> {
        Ok(Self {
            capacity: rows,
            rows,
            lt: CublasLt::new()?,
            activations: HashMap::new(),
            plans: HashMap::new(),
        })
    }

    fn set_rows(&mut self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(Error::Shape {
                label: "Gemma 4 batch linear rows",
                expected: format!("1..={}", self.capacity),
                actual: rows.to_string(),
            });
        }
        if rows != self.rows {
            self.plans.clear();
        }
        self.rows = rows;
        for activation in self.activations.values_mut() {
            activation.cols = rows;
        }
        Ok(())
    }

    fn ensure_plan(&mut self, linear: &Gemma4Linear) -> Result<()> {
        let (out_features, in_features) = linear.shape();
        if !self.activations.contains_key(&in_features) {
            self.activations.insert(
                in_features,
                Nvfp4Matrix::zeroed_col_major(in_features, self.capacity)?,
            );
            self.activations
                .get_mut(&in_features)
                .expect("batch activation exists")
                .cols = self.rows;
        }
        let key = (out_features, in_features, self.rows);
        if !self.plans.contains_key(&key) {
            let activation = self
                .activations
                .get(&in_features)
                .expect("batch activation exists");
            let plan = Fp4TnMatmulPlan::new_f32_output_for_shape(
                &self.lt,
                GemmShape::new(out_features, self.rows, in_features),
                Nvfp4TnInputs::new(linear.cublaslt_weight().matrix(), activation),
                PREFILL_GEMM_WORKSPACE_LIMIT,
            )?;
            self.plans.insert(key, Gemma4BatchLinearPlan { plan });
        }
        Ok(())
    }

    fn run_quantized(
        &self,
        linear: &Gemma4Linear,
        input_scale: f32,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let (_, in_features) = linear.shape();
        let activation = self
            .activations
            .get(&in_features)
            .expect("batch activation exists");
        self.run_quantized_activation(linear, activation, input_scale, output, 0.0, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_quantized_activation(
        &self,
        linear: &Gemma4Linear,
        activation: &Nvfp4Matrix,
        input_scale: f32,
        output: &mut DeviceBuffer<f32>,
        beta: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        let (out_features, in_features) = linear.shape();
        let plan = self
            .plans
            .get(&(out_features, in_features, self.rows))
            .expect("prefill GEMM plan exists");
        let weight = linear.cublaslt_weight();
        plan.plan.run_with_alpha_beta_f32_inout_buffer_on_stream(
            &self.lt,
            Nvfp4TnInputs::new(weight.matrix(), activation),
            output.inout(),
            weight.weight_scale_2() * input_scale,
            beta,
            stream,
        )
    }

    #[cfg(test)]
    fn run(
        &mut self,
        linear: &Gemma4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let (_, in_features) = linear.shape();
        self.ensure_plan(linear)?;
        let weight = linear.cublaslt_weight();
        let input_scale = weight.input_scale();
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            in_features,
            self.rows,
            input,
            self.activations
                .get_mut(&in_features)
                .expect("batch activation exists"),
            input_scale,
            stream,
        )?;
        self.run_quantized(linear, input_scale, output, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rms_norm_pair(
        &mut self,
        first: &Gemma4Linear,
        first_output: &mut DeviceBuffer<f32>,
        second: &Gemma4Linear,
        second_output: &mut DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        norm: &Gemma4RmsNorm,
        stream: &CudaStream,
    ) -> Result<f32> {
        let (_, in_features) = first.shape();
        let (_, second_in_features) = second.shape();
        if second_in_features != in_features || norm.weight.len() != in_features {
            return Err(Error::Shape {
                label: "RMS-normalized paired Gemma 4 linears",
                expected: format!("matching input and norm width {in_features}"),
                actual: format!(
                    "second_input={second_in_features} norm={}",
                    norm.weight.len()
                ),
            });
        }
        self.ensure_plan(first)?;
        self.ensure_plan(second)?;
        let input_scale = first.cublaslt_weight().input_scale();
        rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
            self.rows,
            in_features,
            input,
            &norm.weight,
            self.activations
                .get_mut(&in_features)
                .expect("batch activation exists"),
            norm.eps,
            input_scale,
            stream,
        )?;
        self.run_quantized(first, input_scale, first_output, stream)?;
        self.run_quantized(second, input_scale, second_output, stream)?;
        Ok(input_scale)
    }

    fn device_bytes(&self) -> usize {
        self.activations
            .values()
            .map(Nvfp4Matrix::device_bytes)
            .sum::<usize>()
            + self
                .plans
                .values()
                .map(|plan| plan.plan.workspace_bytes())
                .sum::<usize>()
    }
}

struct Gemma4BatchAttentionWorkspace {
    tensor_core: Gemma4TensorCoreAttentionWorkspace,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    v_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

struct Gemma4TensorCoreAttentionWorkspace {
    lt: CublasLt,
    local: Option<Gemma4LocalPrefillAttention>,
    qk_plans: HashMap<(usize, usize), Bf16TnMatmulPlan>,
    pv_plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    packed_query: DeviceBuffer<u16>,
    packed_key: DeviceBuffer<u16>,
    packed_value: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    packed_probabilities: DeviceBuffer<u16>,
    packed_output: DeviceBuffer<u16>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Gemma4TensorCoreAttentionWorkspace {
    fn new(attention: &Gemma4Attention, rows: usize) -> Result<Self> {
        let initial_context = rows;
        let q_values = rows * attention.q_heads * attention.head_dim;
        let kv_values = initial_context * attention.kv_heads * attention.head_dim;
        let score_values = attention.q_heads * rows * initial_context;
        Ok(Self {
            lt: CublasLt::new()?,
            local: (attention.window == Some(1024)
                && attention.q_heads == 16
                && attention.kv_heads == 8
                && attention.head_dim == 256)
                .then(Gemma4LocalPrefillAttention::new)
                .transpose()?,
            qk_plans: HashMap::new(),
            pv_plans: HashMap::new(),
            packed_query: DeviceBuffer::zeroed(q_values)?,
            packed_key: DeviceBuffer::zeroed(kv_values)?,
            packed_value: DeviceBuffer::zeroed(kv_values)?,
            scores: DeviceBuffer::zeroed(score_values)?,
            packed_probabilities: DeviceBuffer::zeroed(score_values)?,
            packed_output: DeviceBuffer::zeroed(q_values)?,
            q_heads: attention.q_heads,
            kv_heads: attention.kv_heads,
            head_dim: attention.head_dim,
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
        let query_values = query_rows * self.q_heads * self.head_dim;
        let key_values = cache_tokens * self.kv_heads * self.head_dim;
        let score_values = query_rows * self.q_heads * score_key_tokens;
        grow_device_buffer(&mut self.packed_query, query_values)?;
        grow_device_buffer(&mut self.packed_key, key_values)?;
        grow_device_buffer(&mut self.packed_value, key_values)?;
        grow_device_buffer(&mut self.scores, score_values)?;
        grow_device_buffer(&mut self.packed_probabilities, score_values)?;
        grow_device_buffer(&mut self.packed_output, query_values)
    }

    fn device_bytes(&self) -> usize {
        self.packed_query.device_bytes()
            + self.packed_key.device_bytes()
            + self.packed_value.device_bytes()
            + self.scores.device_bytes()
            + self.packed_probabilities.device_bytes()
            + self.packed_output.device_bytes()
            + self
                .qk_plans
                .values()
                .map(Bf16TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
            + self
                .pv_plans
                .values()
                .map(Bf16TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
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
        output: &mut Nvfp4Matrix,
        output_row_offset: usize,
        output_input_scale: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        let start_position = cache.len();
        let cache_tokens = start_position + rows;

        if let Some(local) = &self.local {
            let query_values = rows * self.q_heads * self.head_dim;
            grow_device_buffer(&mut self.packed_query, query_values)?;
            grow_device_buffer(&mut self.packed_output, query_values)?;
            pack_token_heads_bf16_at_offset_into_on_stream(
                query,
                self.packed_query.output(),
                rows,
                self.q_heads,
                self.head_dim,
                input_row_offset,
                stream,
            )?;

            if use_compact_local_attention(start_position, rows) {
                cache.append_rows_at_offset_on_stream(
                    key,
                    value,
                    input_row_offset,
                    rows,
                    stream,
                )?;
                local.run_compact_on_stream(
                    &self.packed_query,
                    cache,
                    self.packed_output.output(),
                    rows,
                    start_position,
                    stream,
                )?;
                unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
                    &self.packed_output,
                    output,
                    rows,
                    self.q_heads,
                    self.head_dim,
                    output_row_offset,
                    output_input_scale,
                    stream,
                )?;
                metrics()
                    .gemma4_compact_local_prefill_rows
                    .add(rows.min(isize::MAX as usize) as isize);
                return Ok(());
            }
        }

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

        if let Some(local) = &self.local {
            local.run_on_stream(
                &self.packed_query,
                &self.packed_key,
                &self.packed_value,
                self.packed_output.output(),
                rows,
                cache_tokens,
                start_position,
                stream,
            )?;
            unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
                &self.packed_output,
                output,
                rows,
                self.q_heads,
                self.head_dim,
                output_row_offset,
                output_input_scale,
                stream,
            )?;
            metrics()
                .gemma4_bf16_local_prefill_rows
                .add(rows.min(isize::MAX as usize) as isize);
            return Ok(());
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
                        PREFILL_GEMM_WORKSPACE_LIMIT,
                    )?,
                );
            }
            let qk = self.qk_plans.get(&qk_key).expect("QK plan exists");
            qk.run_offsets_on_stream(
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
                    Bf16TnMatmulPlan::new_strided_batch_with_a_leading_dimension_bf16_output(
                        &self.lt,
                        GemmShape::new(self.head_dim, query_rows * queries_per_kv, key_tokens),
                        cache_tokens,
                        self.kv_heads,
                        self.head_dim * cache_tokens,
                        queries_per_kv * query_rows * key_tokens,
                        queries_per_kv * query_rows * self.head_dim,
                        PREFILL_GEMM_WORKSPACE_LIMIT,
                    )?,
                );
            }
            let pv = self.pv_plans.get(&pv_key).expect("PV plan exists");
            pv.run_bf16_offsets_on_stream(
                &self.lt,
                &self.packed_value,
                key_start,
                &self.packed_probabilities,
                0,
                self.packed_output.output(),
                0,
                stream,
            )?;
            unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
                &self.packed_output,
                output,
                query_rows,
                self.q_heads,
                self.head_dim,
                output_row_offset + query_offset,
                output_input_scale,
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

impl Gemma4BatchAttentionWorkspace {
    fn new(attention: &Gemma4Attention, rows: usize, _max_context_tokens: usize) -> Result<Self> {
        let q_width = attention.q_heads * attention.head_dim;
        let kv_width = attention.kv_heads * attention.head_dim;
        Ok(Self {
            tensor_core: Gemma4TensorCoreAttentionWorkspace::new(attention, rows)?,
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            v_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            q_rope: DeviceBuffer::zeroed(rows * q_width)?,
            k_rope: DeviceBuffer::zeroed(rows * kv_width)?,
            output: DeviceBuffer::zeroed(rows * attention.output.out_features)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.tensor_core.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.v_normed.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.output.device_bytes()
    }
}

struct Gemma4BatchRouterWorkspace {
    activation: Nvfp4Matrix,
    residual_activation: Nvfp4Matrix,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    normalized_weights: DeviceBuffer<f32>,
    route_weights: DeviceBuffer<f32>,
}

impl Gemma4BatchRouterWorkspace {
    fn new(router: &Gemma4Router, rows: usize) -> Result<Self> {
        let (experts, hidden) = router.projection.shape();
        let routes = rows * router.top_k;
        Ok(Self {
            activation: Nvfp4Matrix::zeroed_col_major(hidden, rows)?,
            residual_activation: Nvfp4Matrix::zeroed_col_major(hidden, rows)?,
            logits: DeviceBuffer::zeroed(rows * experts)?,
            indices: DeviceBuffer::zeroed(routes)?,
            normalized_weights: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.activation.device_bytes()
            + self.residual_activation.device_bytes()
            + self.logits.device_bytes()
            + self.indices.device_bytes()
            + self.normalized_weights.device_bytes()
            + self.route_weights.device_bytes()
    }
}

struct Gemma4BatchMoeWorkspace {
    capacity_rows: usize,
    routes_per_row: usize,
    router: Gemma4BatchRouterWorkspace,
    sorted_routes: MoeSortedRoutes,
    gate_up_input: MoeSortedNvfp4Rows,
    down_input: MoeSortedNvfp4Rows,
    gate_up_plan: CutlassFp4GroupedGemmPlan,
    down_plan: CutlassFp4GroupedGemmPlan,
    gate: DeviceBuffer<u16>,
    up: DeviceBuffer<u16>,
    down: DeviceBuffer<u16>,
    gate_output_table: DeviceBuffer<*mut u16>,
    up_output_table: DeviceBuffer<*mut u16>,
    down_output_table: DeviceBuffer<*mut u16>,
    output: DeviceBuffer<f32>,
}

impl Gemma4BatchMoeWorkspace {
    fn new(moe: &Gemma4Moe, rows: usize) -> Result<Self> {
        let routes_per_row = moe.router.top_k;
        let routes = rows * routes_per_row;
        let experts = moe.gate_packed_table.len();
        let gate = DeviceBuffer::zeroed(routes * moe.intermediate_size)?;
        let up = DeviceBuffer::zeroed(routes * moe.intermediate_size)?;
        let down = DeviceBuffer::zeroed(routes * moe.hidden_size)?;
        Ok(Self {
            capacity_rows: rows,
            routes_per_row,
            router: Gemma4BatchRouterWorkspace::new(&moe.router, rows)?,
            sorted_routes: MoeSortedRoutes::new(routes, experts)?,
            gate_up_input: MoeSortedNvfp4Rows::new(rows, routes_per_row, experts, moe.hidden_size)?,
            down_input: MoeSortedNvfp4Rows::new(
                rows,
                routes_per_row,
                experts,
                moe.intermediate_size,
            )?,
            gate_up_plan: CutlassFp4GroupedGemmPlan::new(
                moe.intermediate_size,
                routes,
                moe.hidden_size,
                experts,
            )?,
            down_plan: CutlassFp4GroupedGemmPlan::new(
                moe.hidden_size,
                routes,
                moe.intermediate_size,
                experts,
            )?,
            gate_output_table: DeviceBuffer::zeroed(experts)?,
            up_output_table: DeviceBuffer::zeroed(experts)?,
            down_output_table: DeviceBuffer::zeroed(experts)?,
            gate,
            up,
            down,
            output: DeviceBuffer::zeroed(rows * moe.hidden_size)?,
        })
    }

    fn set_rows(&mut self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.capacity_rows {
            return Err(Error::Shape {
                label: "Gemma 4 batch MoE rows",
                expected: format!("1..={}", self.capacity_rows),
                actual: rows.to_string(),
            });
        }
        let routes = rows * self.routes_per_row;
        self.sorted_routes.set_routes(routes)?;
        self.gate_up_input.set_rows(rows)?;
        self.down_input.set_rows(rows)
    }

    fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.sorted_routes.device_bytes()
            + self.gate_up_input.device_bytes()
            + self.down_input.device_bytes()
            + self.gate.device_bytes()
            + self.up.device_bytes()
            + self.down.device_bytes()
            + self.gate_output_table.device_bytes()
            + self.up_output_table.device_bytes()
            + self.down_output_table.device_bytes()
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
    residual: DeviceBuffer<f32>,
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
            + self.residual.device_bytes()
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
        let linear = Gemma4BatchLinearWorkspace::new(token_capacity)?;
        let moe = Gemma4BatchMoeWorkspace::new(&local.moe, token_capacity)?;
        Ok(Gemma4PrefillBatchWorkspace {
            sequence_capacity,
            token_capacity,
            max_context_tokens,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            host_token_ids: vec![0; token_capacity],
            hidden: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            layer_output: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            residual: DeviceBuffer::zeroed(token_capacity * self.config.hidden_size)?,
            local_attention: Gemma4BatchAttentionWorkspace::new(
                &local.attention,
                token_capacity,
                max_context_tokens,
            )?,
            global_attention: Gemma4BatchAttentionWorkspace::new(
                &global.attention,
                token_capacity,
                max_context_tokens,
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

        workspace.linear.set_rows(total_tokens)?;
        workspace.moe.set_rows(total_tokens)?;

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
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            self.config.vocab_size,
            self.config.hidden_size,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            total_tokens,
            stream,
        )?;
        scale_channel_f32_device_row_scalar_in_place_on_stream(
            workspace.hidden.inout(),
            &self.embedding_channel_scale,
            &workspace.embedding_row_scale,
            total_tokens,
            self.config.hidden_size,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            workspace.hidden.inout(),
            total_tokens * self.config.hidden_size,
            stream,
        )?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            run_layer_prefill(layer, layer_index, workspace, rows, stream)?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
        }
        let mut row_offset = 0;
        for row in rows.iter_mut() {
            if row.output != Gemma4PrefillOutput::None {
                let final_row = row_offset + row.token_ids.len() - 1;
                copy_row_f32_into_on_stream(
                    workspace.token_capacity,
                    self.config.hidden_size,
                    final_row,
                    &workspace.hidden,
                    row.state.hidden.output(),
                    stream,
                )?;
                let normalized = &mut row
                    .state
                    .layers
                    .last_mut()
                    .expect("Gemma 4 state has every layer")
                    .output;
                self.final_norm.run_into(
                    1,
                    self.config.hidden_size,
                    &row.state.hidden,
                    normalized,
                    stream,
                )?;
                match row.output {
                    Gemma4PrefillOutput::None => unreachable!(),
                    Gemma4PrefillOutput::FullLogits => bf16_linear_argmax_f32_into_on_stream(
                        normalized,
                        &self.embedding,
                        row.state.lm_logits.output(),
                        row.state.lm_argmax.output(),
                        row.state.lm_argmax_value.output(),
                        self.config.vocab_size,
                        self.config.hidden_size,
                        stream,
                    )?,
                    Gemma4PrefillOutput::Top1 => lm_head_top1_f32_into_on_stream(
                        normalized,
                        &self.embedding,
                        &row.state.lm_logits,
                        &row.state.lm_top1_scratch_index,
                        &row.state.lm_argmax,
                        &row.state.lm_argmax_value,
                        self.config.vocab_size,
                        self.config.hidden_size,
                        stream,
                    )?,
                }
            }
            row_offset += row.token_ids.len();
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
    run_layer_pre_attention_prefill(layer, workspace, stream)?;
    run_layer_attention_prefill(layer, layer_index, workspace, rows, stream)?;
    run_layer_post_attention_prefill(layer, layer_index, workspace, stream)
}

fn run_layer_pre_attention_prefill(
    layer: &Gemma4DecoderLayer,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    stream: &CudaStream,
) -> Result<()> {
    let attention_workspace = if layer.attention.window.is_some() {
        &mut workspace.local_attention
    } else {
        &mut workspace.global_attention
    };
    run_attention_prefill_pre(
        &layer.attention,
        &layer.input_norm,
        attention_workspace,
        &mut workspace.linear,
        &workspace.hidden,
        stream,
    )
}

fn run_layer_attention_prefill(
    layer: &Gemma4DecoderLayer,
    layer_index: usize,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    rows: &mut [Gemma4PrefillRow<'_, '_>],
    stream: &CudaStream,
) -> Result<()> {
    let attention_workspace = if layer.attention.window.is_some() {
        &mut workspace.local_attention
    } else {
        &mut workspace.global_attention
    };
    run_attention_prefill_body(
        &layer.attention,
        attention_workspace,
        &mut workspace.linear,
        rows,
        layer_index,
        stream,
    )
}

fn run_layer_post_attention_prefill(
    layer: &Gemma4DecoderLayer,
    layer_index: usize,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    stream: &CudaStream,
) -> Result<()> {
    let active_rows = workspace.linear.rows;
    let hidden = layer.attention.q.in_features;
    let attention_workspace = if layer.attention.window.is_some() {
        &mut workspace.local_attention
    } else {
        &mut workspace.global_attention
    };
    run_attention_prefill_output(
        &layer.attention,
        attention_workspace,
        &mut workspace.linear,
        stream,
    )?;
    workspace.linear.ensure_plan(&layer.dense.gate)?;
    workspace.linear.ensure_plan(&layer.dense.up)?;
    let dense_input_scale = layer.dense.gate.cublaslt_weight().input_scale();
    rms_norm_add_then_rms_norm_quantize_nvfp4_f32_into_on_stream(
        active_rows,
        hidden,
        &attention_workspace.output,
        &layer.post_attention_norm.weight,
        &workspace.hidden,
        workspace.residual.output(),
        layer.post_attention_norm.eps,
        &layer.dense_input_norm.weight,
        workspace
            .linear
            .activations
            .get_mut(&hidden)
            .expect("dense input activation exists"),
        layer.dense_input_norm.eps,
        dense_input_scale,
        stream,
    )?;

    run_mlp_prefill_quantized(
        &layer.dense,
        &mut workspace.dense,
        &mut workspace.linear,
        dense_input_scale,
        stream,
    )?;
    run_moe_prefill(
        &layer.moe,
        &layer.moe_input_norm,
        &mut workspace.moe,
        &mut workspace.linear,
        &workspace.residual,
        active_rows,
        stream,
    )?;
    dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_into_on_stream(
        active_rows,
        hidden,
        &workspace.dense.output,
        &layer.dense_post_norm.weight,
        layer.dense_post_norm.eps,
        &workspace.moe.output,
        &layer.moe_post_norm.weight,
        layer.moe_post_norm.eps,
        &layer.post_feedforward_norm.weight,
        layer.post_feedforward_norm.eps,
        &workspace.residual,
        &layer.layer_scale_channels,
        &workspace.layer_row_scales[layer_index],
        workspace.layer_output.output(),
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attention_prefill_pre(
    attention: &Gemma4Attention,
    input_norm: &Gemma4RmsNorm,
    workspace: &mut Gemma4BatchAttentionWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    linear.run_rms_norm_pair(
        &attention.q,
        &mut workspace.q,
        &attention.k,
        &mut workspace.k,
        input,
        input_norm,
        stream,
    )?;
    if let Some(v) = &attention.v {
        linear.ensure_plan(v)?;
        let (_, in_features) = v.shape();
        let input_scale = v.cublaslt_weight().input_scale();
        rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
            linear.rows,
            in_features,
            input,
            &input_norm.weight,
            linear
                .activations
                .get_mut(&in_features)
                .expect("batch activation exists"),
            input_norm.eps,
            input_scale,
            stream,
        )?;
        linear.run_quantized(v, input_scale, &mut workspace.v, stream)?;
    }
    let value_input = attention.v.as_ref().map_or(&workspace.k, |_| &workspace.v);
    rms_norm_f32_into_on_stream(
        linear.rows * attention.kv_heads,
        attention.head_dim,
        value_input,
        &attention.value_norm_weight,
        workspace.v_normed.output(),
        attention.q_norm.eps,
        stream,
    )
}

fn run_attention_prefill_body(
    attention: &Gemma4Attention,
    workspace: &mut Gemma4BatchAttentionWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    rows: &mut [Gemma4PrefillRow<'_, '_>],
    layer_index: usize,
    stream: &CudaStream,
) -> Result<()> {
    let mut offset = 0;
    linear.ensure_plan(&attention.output)?;
    let attention_width = attention.output.in_features;
    let output_input_scale = attention.output.cublaslt_weight().input_scale();
    for row in rows {
        let position = row.state.position;
        dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            row.token_ids.len(),
            attention.q_heads,
            attention.kv_heads,
            attention.head_dim,
            attention.rotary_dim,
            &workspace.q,
            &attention.q_norm.weight,
            workspace.q_rope.output(),
            attention.q_norm.eps,
            &workspace.k,
            &attention.k_norm.weight,
            workspace.k_rope.output(),
            attention.k_norm.eps,
            offset,
            position,
            attention.rope_theta,
            stream,
        )?;
        workspace.tensor_core.run_sequence(
            &mut row.state.kv_caches[layer_index],
            &workspace.q_rope,
            &workspace.k_rope,
            &workspace.v_normed,
            offset,
            row.token_ids.len(),
            attention.window,
            linear
                .activations
                .get_mut(&attention_width)
                .expect("attention output activation exists"),
            offset,
            output_input_scale,
            stream,
        )?;
        offset += row.token_ids.len();
    }
    Ok(())
}

fn run_attention_prefill_output(
    attention: &Gemma4Attention,
    workspace: &mut Gemma4BatchAttentionWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    stream: &CudaStream,
) -> Result<()> {
    let output_input_scale = attention.output.cublaslt_weight().input_scale();
    linear.run_quantized(
        &attention.output,
        output_input_scale,
        &mut workspace.output,
        stream,
    )
}

fn run_mlp_prefill_quantized(
    mlp: &Gemma4Mlp,
    workspace: &mut Gemma4MlpWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    linear.run_quantized(&mlp.gate, input_scale, &mut workspace.gate, stream)?;
    linear.run_quantized(&mlp.up, input_scale, &mut workspace.up, stream)?;
    linear.ensure_plan(&mlp.down)?;
    let input_scale = mlp.down.cublaslt_weight().input_scale();
    gelu_tanh_mul_quantize_nvfp4_col_major_f32_into_on_stream(
        linear.rows,
        mlp.intermediate_size,
        &workspace.gate,
        &workspace.up,
        linear
            .activations
            .get_mut(&mlp.intermediate_size)
            .expect("batch activation exists"),
        input_scale,
        stream,
    )?;
    linear.run_quantized(&mlp.down, input_scale, &mut workspace.output, stream)
}

#[allow(clippy::too_many_arguments)]
fn run_moe_prefill(
    moe: &Gemma4Moe,
    expert_input_norm: &Gemma4RmsNorm,
    workspace: &mut Gemma4BatchMoeWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    router_input: &DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let experts = moe.router.projection.out_features;
    let routes_per_row = moe.router.top_k;
    linear.ensure_plan(&moe.router.projection)?;
    workspace.router.activation.cols = rows;
    workspace.router.residual_activation.cols = rows;
    let router_input_scale = moe.router.projection.cublaslt_weight().input_scale();
    let router_quant_scale = router_input_scale / moe.router.input_norm_scalar_value;
    rms_norm_quantize_nvfp4_pair_col_major_f32_into_on_stream(
        rows,
        moe.hidden_size,
        router_input,
        &moe.router.router_scale,
        &mut workspace.router.activation,
        &mut workspace.router.residual_activation,
        moe.router.rms_norm_eps,
        router_quant_scale,
        stream,
    )?;
    linear.run_quantized_activation(
        &moe.router.projection,
        &workspace.router.activation,
        router_input_scale,
        &mut workspace.router.logits,
        0.0,
        stream,
    )?;
    linear.run_quantized_activation(
        &moe.router.projection,
        &workspace.router.residual_activation,
        router_input_scale,
        &mut workspace.router.logits,
        1.0,
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
    gather_indexed_mul_f32_prefix_into_on_stream(
        &moe.router.per_expert_scale,
        &workspace.router.indices,
        &workspace.router.normalized_weights,
        workspace.router.route_weights.output(),
        rows * routes_per_row,
        stream,
    )?;
    workspace
        .sorted_routes
        .sort_on_stream(&workspace.router.indices, stream)?;
    workspace.gate_up_input.gather_rms_norm_quantize_on_stream(
        router_input,
        &expert_input_norm.weight,
        expert_input_norm.eps,
        &workspace.sorted_routes,
        stream,
    )?;
    workspace.gate_up_input.build_pointer_tables_on_stream(
        &workspace.sorted_routes,
        &mut workspace.gate,
        &mut workspace.gate_output_table,
        moe.intermediate_size,
        stream,
    )?;
    workspace.gate_up_plan.run_on_stream(
        &moe.gate_packed_table,
        &moe.gate_tiled_scale_table,
        workspace.gate_up_input.packed_table(),
        workspace.gate_up_input.scale_table(),
        &workspace.gate_output_table,
        &moe.gate_alpha_table,
        workspace.sorted_routes.expert_counts(),
        stream,
    )?;
    workspace.gate_up_input.build_pointer_tables_on_stream(
        &workspace.sorted_routes,
        &mut workspace.up,
        &mut workspace.up_output_table,
        moe.intermediate_size,
        stream,
    )?;
    workspace.gate_up_plan.run_on_stream(
        &moe.up_packed_table,
        &moe.up_tiled_scale_table,
        workspace.gate_up_input.packed_table(),
        workspace.gate_up_input.scale_table(),
        &workspace.up_output_table,
        &moe.up_alpha_table,
        workspace.sorted_routes.expert_counts(),
        stream,
    )?;
    workspace
        .down_input
        .gelu_tanh_mul_quantize_sorted_on_stream(
            &workspace.gate,
            &workspace.up,
            &workspace.sorted_routes,
            stream,
        )?;
    workspace.down_input.build_pointer_tables_on_stream(
        &workspace.sorted_routes,
        &mut workspace.down,
        &mut workspace.down_output_table,
        moe.hidden_size,
        stream,
    )?;
    workspace.down_plan.run_on_stream(
        &moe.down_packed_table,
        &moe.down_tiled_scale_table,
        workspace.down_input.packed_table(),
        workspace.down_input.scale_table(),
        &workspace.down_output_table,
        &moe.down_alpha_table,
        workspace.sorted_routes.expert_counts(),
        stream,
    )?;
    moe_weighted_accumulate_sorted_bf16_batch_on_stream(
        &workspace.sorted_routes,
        &workspace.router.route_weights,
        &workspace.down,
        workspace.output.output(),
        rows,
        routes_per_row,
        moe.hidden_size,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_local_attention_requires_a_long_prefix_relative_to_the_query() {
        assert!(!use_compact_local_attention(0, 1));
        assert!(!use_compact_local_attention(4_096, 512));
        assert!(use_compact_local_attention(8_192, 512));
        assert!(use_compact_local_attention(2_688, 128));
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_w4a4_projection_matches_w4a16_reference() {
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
        let mut workspace = Gemma4BatchLinearWorkspace::new(rows).expect("linear workspace");
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
        let error_rms = (actual
            .iter()
            .zip(reference.iter())
            .map(|(actual, reference)| (actual - reference).powi(2))
            .sum::<f32>()
            / actual.len() as f32)
            .sqrt();
        let reference_rms = (reference.iter().map(|value| value.powi(2)).sum::<f32>()
            / reference.len() as f32)
            .sqrt();
        assert!(
            error_rms <= reference_rms * 0.125,
            "max_error={max_error} error_rms={error_rms} reference_rms={reference_rms}"
        );
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_batched_moe_matches_independent_tokens() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let checkpoint = Gemma4Checkpoint::open(model_dir).expect("open Gemma checkpoint");
        let moe = Gemma4Moe::load(&checkpoint, 0).expect("load layer-zero MoE");
        let moe_input_norm = Gemma4RmsNorm::load(
            &checkpoint,
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
            moe.hidden_size,
        )
        .expect("load layer-zero MoE input norm");
        let rows = 3;
        let router_host = (0..rows * moe.hidden_size)
            .map(|index| ((index % 97) as f32 - 48.0) / 48.0)
            .collect::<Vec<_>>();
        let router_input = DeviceBuffer::from_host(&router_host).expect("router input");
        let stream = CudaStream::new_blocking().expect("stream");
        let mut workspace = Gemma4BatchMoeWorkspace::new(&moe, rows).expect("batch MoE workspace");
        let mut linear = Gemma4BatchLinearWorkspace::new(rows).expect("linear workspace");
        run_moe_prefill(
            &moe,
            &moe_input_norm,
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
            let mut expert_row = DeviceBuffer::zeroed(moe.hidden_size).expect("expert row");
            moe_input_norm
                .run_into(1, moe.hidden_size, &router_row, &mut expert_row, &stream)
                .expect("expert input norm");
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
                output: Gemma4PrefillOutput::None,
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
                    serial_state
                        .compact_attention
                        .for_layer_mut(true)
                        .expect("local compact attention workspace"),
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

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_active_rows_match_exact_workspace() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let model = Gemma4Model::load(model_dir).expect("load Gemma 4");
        let tokens = [2, 3, 2, 3];
        let mut exact_state = model.new_decode_state(tokens.len()).expect("exact state");
        let mut padded_state = model.new_decode_state(tokens.len()).expect("padded state");
        let mut exact = model
            .new_prefill_batch_workspace(1, tokens.len(), tokens.len())
            .expect("exact workspace");
        let mut padded = model
            .new_prefill_batch_workspace(1, 8, tokens.len())
            .expect("padded workspace");
        let stream = CudaStream::new_blocking().expect("stream");

        model
            .prefill_batch(
                &mut exact,
                &mut [Gemma4PrefillRow {
                    token_ids: &tokens,
                    state: &mut exact_state,
                    output: Gemma4PrefillOutput::None,
                }],
                &stream,
            )
            .expect("exact prefill");
        model
            .prefill_batch(
                &mut padded,
                &mut [Gemma4PrefillRow {
                    token_ids: &tokens,
                    state: &mut padded_state,
                    output: Gemma4PrefillOutput::None,
                }],
                &stream,
            )
            .expect("padded prefill");

        let expected = exact.hidden.copy_to_host(&stream).expect("exact output");
        let actual = padded.hidden.copy_to_host(&stream).expect("padded output");
        let active_values = tokens.len() * model.config.hidden_size;
        let max_error = actual[..active_values]
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 1.0e-6, "max active-row error={max_error}");
    }
}
