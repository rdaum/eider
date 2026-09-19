//! Batched Gemma 4 prefill execution and reusable device workspaces.

use super::*;
use crate::gemma4::{Gemma4Append, Gemma4Sequence, Gemma4SequenceCache, gemma4_cache_error};
use crate::paged_prefill_attention::PagedTensorCorePrefillAttention;
use crate::sm12x_cache::Sm12xCacheContext;
use eider_cuda::{
    CublasLt, CutlassFp4GroupedGemmPlan, DeviceAddress, DeviceBuffer, Fp4TnMatmulPlan, GemmShape,
    MoeSortedNvfp4Rows, MoeSortedRoutes, Nvfp4Matrix, Nvfp4TnInputs,
    bf16_linear_logits_f32_batch_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, copy_row_f32_into_on_stream,
    dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_into_on_stream,
    dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream,
    gather_indexed_mul_f32_prefix_into_on_stream,
    gelu_tanh_mul_quantize_nvfp4_col_major_f32_into_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    rms_norm_add_then_rms_norm_quantize_nvfp4_f32_into_on_stream,
    rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    rms_norm_quantize_nvfp4_pair_col_major_f32_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream,
};
use std::collections::HashMap;

const PREFILL_GEMM_WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;
const COMPACT_LOCAL_ATTENTION_MIN_PREFIX_PER_QUERY: usize = 16;

fn use_compact_prefill_attention(
    window: Option<usize>,
    start_position: usize,
    query_rows: usize,
) -> bool {
    window.is_some()
        && start_position != 0
        && start_position >= query_rows.saturating_mul(COMPACT_LOCAL_ATTENTION_MIN_PREFIX_PER_QUERY)
}
/// One scheduler-selected Gemma prompt chunk and its persistent sequence state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4PrefillOutput {
    None,
    FinalHidden,
    FullLogits,
    Top1,
}

pub struct Gemma4PrefillRow<'tokens, 'state> {
    pub token_ids: &'tokens [u32],
    pub sequence: &'state mut Gemma4Sequence,
    pub output: Gemma4PrefillOutput,
}

/// Fixed 64-row tied-embedding projection used by native decision requests.
pub struct Gemma4DecisionReadout {
    token_ids: Vec<u32>,
    weight: DeviceBuffer<u16>,
    hidden: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    capacity: usize,
    hidden_size: usize,
}

impl Gemma4DecisionReadout {
    /// Returns the vocabulary rows retained by this compact head.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Returns exact device bytes owned by the compact head and its workspace.
    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
            + self.hidden.device_bytes()
            + self.normalized.device_bytes()
            + self.logits.device_bytes()
    }

    /// Stages one final-normalized sequence row for a batched compact projection.
    pub(crate) fn stage_sequence(
        &mut self,
        row: usize,
        sequence: &Gemma4Sequence,
        stream: &CudaStream,
    ) -> Result<()> {
        if row >= self.capacity {
            return Err(Error::Shape {
                label: "Gemma 4 decision readout row",
                expected: format!("row < {}", self.capacity),
                actual: row.to_string(),
            });
        }
        self.hidden.copy_range_from_device_on_stream(
            row * self.hidden_size,
            sequence.state.final_hidden(),
            0,
            self.hidden_size,
            stream,
        )
    }

    /// Projects staged rows and returns soft-capped logits in row-major order.
    pub(crate) fn selected_logits(
        &mut self,
        model: &Gemma4Model,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<Vec<f32>> {
        if rows == 0 || rows > self.capacity {
            return Err(Error::Shape {
                label: "Gemma 4 decision readout rows",
                expected: format!("1..={}", self.capacity),
                actual: rows.to_string(),
            });
        }
        bf16_linear_logits_f32_batch_into_on_stream(
            &self.hidden,
            &self.weight,
            self.logits.output(),
            rows,
            self.token_ids.len(),
            self.hidden_size,
            stream,
        )?;
        let mut logits = self
            .logits
            .copy_prefix_to_host(rows * self.token_ids.len(), stream)?
            .into_vec();
        for logit in &mut logits {
            *logit = model.softcap_logit(*logit);
        }
        Ok(logits)
    }

    fn selected_tree_logits(
        &mut self,
        model: &Gemma4Model,
        source: &DeviceBuffer<f32>,
        source_rows: &[usize],
        stream: &CudaStream,
    ) -> Result<Vec<f32>> {
        if source_rows.is_empty() || source_rows.len() > self.capacity {
            return Err(Error::Shape {
                label: "Gemma 4 decision tree readout rows",
                expected: format!("1..={}", self.capacity),
                actual: source_rows.len().to_string(),
            });
        }
        for (destination, source_row) in source_rows.iter().copied().enumerate() {
            let source_offset =
                source_row
                    .checked_mul(self.hidden_size)
                    .ok_or_else(|| Error::Shape {
                        label: "Gemma 4 decision tree readout offset",
                        expected: "row * hidden size without overflow".to_string(),
                        actual: format!("row={source_row} hidden={}", self.hidden_size),
                    })?;
            self.hidden.copy_range_from_device_on_stream(
                destination * self.hidden_size,
                source,
                source_offset,
                self.hidden_size,
                stream,
            )?;
        }
        model.final_norm.run_into(
            source_rows.len(),
            self.hidden_size,
            &self.hidden,
            &mut self.normalized,
            stream,
        )?;
        bf16_linear_logits_f32_batch_into_on_stream(
            &self.normalized,
            &self.weight,
            self.logits.output(),
            source_rows.len(),
            self.token_ids.len(),
            self.hidden_size,
            stream,
        )?;
        let mut logits = self
            .logits
            .copy_prefix_to_host(source_rows.len() * self.token_ids.len(), stream)?
            .into_vec();
        for logit in &mut logits {
            *logit = model.softcap_logit(*logit);
        }
        Ok(logits)
    }
}

struct Gemma4PrefillStateRow<'tokens, 'state> {
    token_ids: &'tokens [u32],
    state: &'state mut Gemma4DecodeState,
    output: Gemma4PrefillOutput,
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
        if rows != self.rows {
            self.plans.clear();
        }
        self.set_rows_preserving_plans(rows)
    }

    fn set_rows_preserving_plans(&mut self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(Error::Shape {
                label: "Gemma 4 batch linear rows",
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

    fn retain_plans_for_rows(&mut self, rows: usize) {
        self.plans.retain(|&(_, _, plan_rows), _| plan_rows == rows);
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
    compact: Sm12xKvAttentionWorkspace,
    tensor_core: PagedTensorCorePrefillAttention,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    v_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Gemma4BatchAttentionWorkspace {
    fn new(attention: &Gemma4Attention, rows: usize, max_context_tokens: usize) -> Result<Self> {
        let q_width = attention.q_heads * attention.head_dim;
        let kv_width = attention.kv_heads * attention.head_dim;
        Ok(Self {
            compact: Sm12xKvAttentionWorkspace::new_gqa_batched(
                max_context_tokens,
                attention.q_heads,
                attention.kv_heads,
                attention.head_dim,
                16,
            )?,
            tensor_core: PagedTensorCorePrefillAttention::new(
                rows,
                attention.q_heads,
                attention.kv_heads,
                attention.head_dim,
            )?,
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            v_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            q_rope: DeviceBuffer::zeroed(rows * q_width)?,
            k_rope: DeviceBuffer::zeroed(rows * kv_width)?,
            attended: DeviceBuffer::zeroed(rows * q_width)?,
            output: DeviceBuffer::zeroed(rows * attention.output.out_features)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.compact.device_bytes()
            + self.tensor_core.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.v_normed.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attended.device_bytes()
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
    gate_output_table: DeviceBuffer<DeviceAddress<u16>>,
    up_output_table: DeviceBuffer<DeviceAddress<u16>>,
    down_output_table: DeviceBuffer<DeviceAddress<u16>>,
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
    /// Builds the fixed tied-embedding projection used by decision requests.
    pub fn new_decision_readout(
        &self,
        token_ids: &[u32],
        capacity: usize,
    ) -> Result<Gemma4DecisionReadout> {
        if token_ids.len() != 64
            || capacity == 0
            || token_ids
                .iter()
                .any(|token| *token as usize >= self.config.vocab_size)
            || token_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != token_ids.len()
        {
            return Err(Error::Shape {
                label: "Gemma 4 decision label head",
                expected: "64 distinct vocabulary token IDs and positive capacity".to_string(),
                actual: format!("{} token IDs, capacity {capacity}", token_ids.len()),
            });
        }
        let stream = CudaStream::new_blocking()?;
        let mut weight = DeviceBuffer::zeroed(token_ids.len() * self.config.hidden_size)?;
        for (row, token) in token_ids.iter().copied().enumerate() {
            weight.copy_range_from_device_on_stream(
                row * self.config.hidden_size,
                &self.embedding,
                token as usize * self.config.hidden_size,
                self.config.hidden_size,
                &stream,
            )?;
        }
        stream.synchronize()?;
        Ok(Gemma4DecisionReadout {
            token_ids: token_ids.to_vec(),
            weight,
            hidden: DeviceBuffer::zeroed(capacity * self.config.hidden_size)?,
            normalized: DeviceBuffer::zeroed(capacity * self.config.hidden_size)?,
            logits: DeviceBuffer::zeroed(capacity * token_ids.len())?,
            capacity,
            hidden_size: self.config.hidden_size,
        })
    }

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
        let attention_capacity = sequence::gemma4_state_capacity(max_context_tokens)?;
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
                attention_capacity,
            )?,
            global_attention: Gemma4BatchAttentionWorkspace::new(
                &global.attention,
                token_capacity,
                attention_capacity,
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

    pub(crate) fn decision_tree_fits(
        &self,
        workspace: &Gemma4PrefillBatchWorkspace,
        readout: &Gemma4DecisionReadout,
        prefix: &[u32],
        branches: &[&[u32]],
    ) -> bool {
        let Some(total_tokens) = branches.iter().try_fold(prefix.len(), |total, branch| {
            total.checked_add(branch.len())
        }) else {
            return false;
        };
        !prefix.is_empty()
            && !branches.is_empty()
            && branches.iter().all(|branch| !branch.is_empty())
            && total_tokens <= workspace.token_capacity
            && branches
                .iter()
                .all(|branch| prefix.len() + branch.len() <= workspace.max_context_tokens)
            && branches.len() <= readout.capacity
            && workspace
                .local_attention
                .tensor_core
                .tree_rows_fit(total_tokens)
            && workspace
                .global_attention
                .tensor_core
                .tree_rows_fit(total_tokens)
    }

    /// Runs one layer-major tree prefill for a shared prefix and isolated branches.
    pub(crate) fn decision_tree_selected_logits(
        &self,
        workspace: &mut Gemma4PrefillBatchWorkspace,
        readout: &mut Gemma4DecisionReadout,
        prefix: &[u32],
        branches: &[&[u32]],
        stream: &CudaStream,
    ) -> Result<Vec<f32>> {
        if prefix.is_empty()
            || branches.is_empty()
            || branches.iter().any(|branch| branch.is_empty())
        {
            return Err(Error::Shape {
                label: "Gemma 4 decision tree",
                expected: "a non-empty prefix and non-empty branches".to_string(),
                actual: format!("prefix={} branches={}", prefix.len(), branches.len()),
            });
        }
        let total_tokens = branches.iter().try_fold(prefix.len(), |total, branch| {
            total.checked_add(branch.len()).ok_or_else(|| Error::Shape {
                label: "Gemma 4 decision tree token count",
                expected: "prefix + branches without overflow".to_string(),
                actual: "overflow".to_string(),
            })
        })?;
        let longest_branch = branches
            .iter()
            .map(|branch| prefix.len() + branch.len())
            .max()
            .expect("branches are not empty");
        if total_tokens > workspace.token_capacity
            || longest_branch > workspace.max_context_tokens
            || branches.len() > readout.capacity
        {
            return Err(Error::Shape {
                label: "Gemma 4 decision tree capacity",
                expected: format!(
                    "tokens <= {}, logical branch <= {}, branches <= {}",
                    workspace.token_capacity, workspace.max_context_tokens, readout.capacity
                ),
                actual: format!(
                    "tokens={total_tokens} logical_branch={longest_branch} branches={}",
                    branches.len()
                ),
            });
        }
        if prefix
            .iter()
            .chain(branches.iter().flat_map(|branch| branch.iter()))
            .any(|token| *token as usize >= self.config.vocab_size)
        {
            return Err(Error::Shape {
                label: "Gemma 4 decision tree token",
                expected: format!("token < {}", self.config.vocab_size),
                actual: "out-of-range token".to_string(),
            });
        }

        workspace.linear.set_rows(total_tokens)?;
        workspace.linear.retain_plans_for_rows(total_tokens);
        workspace.moe.set_rows(total_tokens)?;
        workspace.host_token_ids.fill(0);
        workspace.host_token_ids[..prefix.len()].copy_from_slice(prefix);
        let mut lengths = Vec::with_capacity(branches.len() + 1);
        lengths.push(prefix.len());
        let mut tip_rows = Vec::with_capacity(branches.len());
        let mut offset = prefix.len();
        for tokens in branches {
            let end = offset + tokens.len();
            workspace.host_token_ids[offset..end].copy_from_slice(tokens);
            lengths.push(tokens.len());
            tip_rows.push(end - 1);
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
            run_layer_tree_prefill(layer, layer_index, workspace, &lengths, stream)?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.layer_output);
        }
        workspace.linear.retain_plans_for_rows(total_tokens);
        readout.selected_tree_logits(self, &workspace.hidden, &tip_rows, stream)
    }

    /// Advances one or more persistent sequence states by flattened prompt chunks.
    pub fn prefill_batch(
        &self,
        workspace: &mut Gemma4PrefillBatchWorkspace,
        rows: &mut [Gemma4PrefillRow<'_, '_>],
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
    ) -> Result<()> {
        let mut reservations = Vec::with_capacity(rows.len());
        for index in 0..rows.len() {
            let reservation = {
                let row = &mut rows[index];
                cache.reserve_append(
                    row.sequence.cache_id,
                    row.token_ids.len(),
                    &mut Sm12xCacheContext {
                        stream,
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
                                    stream,
                                    page_table: &mut row.sequence.page_table,
                                },
                            )
                            .map_err(gemma4_cache_error)?;
                    }
                    return Err(gemma4_cache_error(error));
                }
            }
        }
        let result = {
            let mut state_rows = Vec::with_capacity(rows.len());
            let mut appends = Vec::with_capacity(rows.len());
            for (row, reservation) in rows.iter_mut().zip(&reservations) {
                let sequence = &mut *row.sequence;
                state_rows.push(Gemma4PrefillStateRow {
                    token_ids: row.token_ids,
                    state: &mut sequence.state,
                    output: row.output,
                });
                appends.push(Gemma4Append {
                    reservation,
                    page_table: sequence.page_table.device(),
                });
            }
            self.prefill_batch_impl(workspace, &mut state_rows, stream, cache, &appends)
        };
        if let Err(error) = result {
            for (row, reservation) in rows.iter_mut().zip(reservations) {
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream,
                            page_table: &mut row.sequence.page_table,
                        },
                    )
                    .map_err(gemma4_cache_error)?;
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
                        stream,
                        page_table: &mut row.sequence.page_table,
                    },
                )
                .map_err(gemma4_cache_error)?;
            row.sequence.state.position += tokens;
        }
        Ok(())
    }

    fn prefill_batch_impl(
        &self,
        workspace: &mut Gemma4PrefillBatchWorkspace,
        rows: &mut [Gemma4PrefillStateRow<'_, '_>],
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
        appends: &[Gemma4Append<'_>],
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
            run_layer_prefill(layer, layer_index, workspace, rows, stream, cache, appends)?;
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
                let state = &mut *row.state;
                let normalized = &mut state
                    .layers
                    .last_mut()
                    .expect("Gemma 4 state has every layer")
                    .output;
                self.final_norm.run_into(
                    1,
                    self.config.hidden_size,
                    &state.hidden,
                    normalized,
                    stream,
                )?;
                state.hidden.copy_range_from_device_on_stream(
                    0,
                    normalized,
                    0,
                    self.config.hidden_size,
                    stream,
                )?;
                match row.output {
                    Gemma4PrefillOutput::None => unreachable!(),
                    Gemma4PrefillOutput::FinalHidden => {}
                    Gemma4PrefillOutput::FullLogits => {
                        let lm_head = state.lm_head.as_mut().ok_or_else(|| Error::Format {
                            label: "Gemma 4 LM head workspace",
                            detail: "sequence was allocated without generation scratch".to_string(),
                        })?;
                        bf16_linear_argmax_f32_into_on_stream(
                            &state.hidden,
                            &self.embedding,
                            lm_head.lm_logits.output(),
                            lm_head.lm_argmax.output(),
                            lm_head.lm_argmax_value.output(),
                            self.config.vocab_size,
                            self.config.hidden_size,
                            stream,
                        )?
                    }
                    Gemma4PrefillOutput::Top1 => {
                        let lm_head = state.lm_head.as_ref().ok_or_else(|| Error::Format {
                            label: "Gemma 4 LM head workspace",
                            detail: "sequence was allocated without generation scratch".to_string(),
                        })?;
                        lm_head_top1_f32_into_on_stream(
                            &state.hidden,
                            &self.embedding,
                            &lm_head.lm_logits,
                            &lm_head.lm_top1_scratch_index,
                            &lm_head.lm_argmax,
                            &lm_head.lm_argmax_value,
                            self.config.vocab_size,
                            self.config.hidden_size,
                            stream,
                        )?
                    }
                }
            }
            row_offset += row.token_ids.len();
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_layer_prefill(
    layer: &Gemma4DecoderLayer,
    layer_index: usize,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    rows: &mut [Gemma4PrefillStateRow<'_, '_>],
    stream: &CudaStream,
    cache: &mut Gemma4SequenceCache,
    appends: &[Gemma4Append<'_>],
) -> Result<()> {
    run_layer_pre_attention_prefill(layer, workspace, stream)?;
    run_layer_attention_prefill(layer, layer_index, workspace, rows, stream, cache, appends)?;
    run_layer_post_attention_prefill(layer, layer_index, workspace, stream)
}

fn run_layer_tree_prefill(
    layer: &Gemma4DecoderLayer,
    layer_index: usize,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    lengths: &[usize],
    stream: &CudaStream,
) -> Result<()> {
    run_layer_pre_attention_prefill(layer, workspace, stream)?;
    run_attention_tree_prefill_body(&layer.attention, workspace, lengths, stream)?;
    run_attention_tree_prefill_output(&layer.attention, workspace, lengths, stream)?;
    run_layer_post_attention_prefill_body(layer, layer_index, workspace, stream)
}

fn run_attention_tree_prefill_body(
    attention: &Gemma4Attention,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    lengths: &[usize],
    stream: &CudaStream,
) -> Result<()> {
    let attention_workspace = if attention.window.is_some() {
        &mut workspace.local_attention
    } else {
        &mut workspace.global_attention
    };
    let mut offset = 0;
    for (segment, &rows) in lengths.iter().enumerate() {
        let logical_start = if segment == 0 { 0 } else { lengths[0] };
        dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            rows,
            attention.q_heads,
            attention.kv_heads,
            attention.head_dim,
            attention.rotary_dim,
            &attention_workspace.q,
            &attention.q_norm.weight,
            attention_workspace.q_rope.output(),
            attention.q_norm.eps,
            &attention_workspace.k,
            &attention.k_norm.weight,
            attention_workspace.k_rope.output(),
            attention.k_norm.eps,
            offset,
            logical_start,
            attention.rope_theta,
            stream,
        )?;
        offset += rows;
    }
    attention_workspace.tensor_core.run_tree(
        &attention_workspace.q_rope,
        &attention_workspace.k_rope,
        &attention_workspace.v_normed,
        lengths,
        attention.window,
        &mut attention_workspace.attended,
        stream,
    )
}

fn run_attention_tree_prefill_output(
    attention: &Gemma4Attention,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    lengths: &[usize],
    stream: &CudaStream,
) -> Result<()> {
    // cuBLASLt can select row-count-dependent FP4 reductions. Project each logical
    // segment independently so a branch produces the same result whether or not
    // sibling branches share the tree request.
    let total_rows = lengths.iter().sum::<usize>();
    let attention_width = attention.output.in_features;
    let output_width = attention.output.out_features;
    let output_input_scale = attention.output.cublaslt_weight().input_scale();
    let attention_workspace = if attention.window.is_some() {
        &mut workspace.local_attention
    } else {
        &mut workspace.global_attention
    };
    let result = (|| {
        let mut offset = 0;
        for &rows in lengths {
            workspace.linear.set_rows_preserving_plans(rows)?;
            workspace.linear.ensure_plan(&attention.output)?;
            attention_workspace.q.copy_range_from_device_on_stream(
                0,
                &attention_workspace.attended,
                offset * attention_width,
                rows * attention_width,
                stream,
            )?;
            quantize_nvfp4_col_major_f32_device_into_on_stream(
                attention_width,
                rows,
                &attention_workspace.q,
                workspace
                    .linear
                    .activations
                    .get_mut(&attention_width)
                    .expect("attention output activation exists"),
                output_input_scale,
                stream,
            )?;
            workspace.linear.run_quantized(
                &attention.output,
                output_input_scale,
                &mut workspace.residual,
                stream,
            )?;
            attention_workspace
                .output
                .copy_range_from_device_on_stream(
                    offset * output_width,
                    &workspace.residual,
                    0,
                    rows * output_width,
                    stream,
                )?;
            offset += rows;
        }
        Ok(())
    })();
    workspace.linear.set_rows_preserving_plans(total_rows)?;
    result
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
    rows: &mut [Gemma4PrefillStateRow<'_, '_>],
    stream: &CudaStream,
    cache: &mut Gemma4SequenceCache,
    appends: &[Gemma4Append<'_>],
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
        cache,
        appends,
    )
}

fn run_layer_post_attention_prefill(
    layer: &Gemma4DecoderLayer,
    layer_index: usize,
    workspace: &mut Gemma4PrefillBatchWorkspace,
    stream: &CudaStream,
) -> Result<()> {
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
    run_layer_post_attention_prefill_body(layer, layer_index, workspace, stream)
}

fn run_layer_post_attention_prefill_body(
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

#[allow(clippy::too_many_arguments)]
fn run_attention_prefill_body(
    attention: &Gemma4Attention,
    workspace: &mut Gemma4BatchAttentionWorkspace,
    linear: &mut Gemma4BatchLinearWorkspace,
    rows: &mut [Gemma4PrefillStateRow<'_, '_>],
    layer_index: usize,
    stream: &CudaStream,
    cache: &mut Gemma4SequenceCache,
    appends: &[Gemma4Append<'_>],
) -> Result<()> {
    let mut offset = 0;
    linear.ensure_plan(&attention.output)?;
    let attention_width = attention.output.in_features;
    let output_input_scale = attention.output.cublaslt_weight().input_scale();
    for (row, append) in rows.iter_mut().zip(appends) {
        let position = row.state.position;
        if append.reservation.start_position() != position
            || append.reservation.rows() != row.token_ids.len()
        {
            return Err(Error::Format {
                label: "Gemma 4 prefill append",
                detail: "reservation does not match the prompt chunk".to_string(),
            });
        }
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
                            &workspace.v_normed,
                            offset + token,
                            chunk_rows,
                            stream,
                        )?;
                        processed += chunk_rows;
                    }
                }
                if use_compact_prefill_attention(attention.window, position, row.token_ids.len()) {
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
                                    offset + token,
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
                        offset,
                        row.token_ids.len(),
                        attention.window,
                        &mut workspace.attended,
                        stream,
                    )?;
                }
                Ok(())
            })
            .map_err(gemma4_cache_error)?;
        offset += row.token_ids.len();
    }
    quantize_nvfp4_col_major_f32_device_into_on_stream(
        attention_width,
        linear.rows,
        &workspace.attended,
        linear
            .activations
            .get_mut(&attention_width)
            .expect("attention output activation exists"),
        output_input_scale,
        stream,
    )?;
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
        &moe.gate_grouped_packed_table,
        &moe.gate_grouped_tiled_scale_table,
        workspace.gate_up_input.packed_table(),
        workspace.gate_up_input.scale_table(),
        &workspace.gate_output_table,
        &moe.gate_grouped_alpha_table,
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
        &moe.up_grouped_packed_table,
        &moe.up_grouped_tiled_scale_table,
        workspace.gate_up_input.packed_table(),
        workspace.gate_up_input.scale_table(),
        &workspace.up_output_table,
        &moe.up_grouped_alpha_table,
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
        &moe.down_grouped_packed_table,
        &moe.down_grouped_tiled_scale_table,
        workspace.down_input.packed_table(),
        workspace.down_input.scale_table(),
        &workspace.down_output_table,
        &moe.down_grouped_alpha_table,
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
    use eider_cuda::SM12X_KV_PAGE_TOKENS;
    use eider_runtime::chat::CheckpointChatTemplate;
    use eider_runtime::decision::{
        DecisionPromptCompiler, DecisionPromptQuestion, DecisionPromptRequest,
    };
    use serde_json::json;

    fn local_model_dir() -> std::path::PathBuf {
        std::env::var_os("EIDER_GEMMA4_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("models/gemma-4-26b-a4b-nvfp4")
            })
    }

    fn assert_decision_fork_matches_full_head(
        model: &Gemma4Model,
        prefix_len: usize,
        stream: &CudaStream,
        cache: &mut Gemma4SequenceCache,
        workspace: &mut Gemma4PrefillBatchWorkspace,
        readout: &mut Gemma4DecisionReadout,
    ) -> (Vec<f32>, Vec<f32>) {
        let prefix = vec![2; prefix_len];
        let suffix = [3, 4, 5];
        let max_tokens = prefix.len() + suffix.len();
        let mut parent =
            Gemma4Sequence::admit_decision(model, cache, prefix.len(), stream).expect("parent");
        model
            .prefill_batch(
                workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &prefix,
                    sequence: &mut parent,
                    output: Gemma4PrefillOutput::None,
                }],
                stream,
                cache,
            )
            .expect("parent prefill");
        stream.synchronize().expect("parent prefill completion");
        let mut branch = Gemma4Sequence::branch_decision(model, &parent, cache, max_tokens, stream)
            .expect("branch")
            .expect("branch capacity");
        stream.synchronize().expect("branch copy completion");
        model
            .prefill_batch(
                workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &suffix[..suffix.len() - 1],
                    sequence: &mut branch,
                    output: Gemma4PrefillOutput::None,
                }],
                stream,
                cache,
            )
            .expect("branch suffix prefix");
        stream.synchronize().expect("branch suffix completion");
        model
            .prefill_batch(
                workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &suffix[suffix.len() - 1..],
                    sequence: &mut branch,
                    output: Gemma4PrefillOutput::FinalHidden,
                }],
                stream,
                cache,
            )
            .expect("branch final token");
        readout
            .stage_sequence(0, &branch, stream)
            .expect("stage decision row");
        let compact = readout
            .selected_logits(model, 1, stream)
            .expect("compact logits");

        let mut reference =
            Gemma4Sequence::admit(model, cache, max_tokens, stream).expect("reference");
        model
            .prefill_batch(
                workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &prefix,
                    sequence: &mut reference,
                    output: Gemma4PrefillOutput::None,
                }],
                stream,
                cache,
            )
            .expect("reference prefix");
        stream.synchronize().expect("reference prefix completion");
        model
            .prefill_batch(
                workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &suffix[..suffix.len() - 1],
                    sequence: &mut reference,
                    output: Gemma4PrefillOutput::None,
                }],
                stream,
                cache,
            )
            .expect("reference suffix prefix");
        stream.synchronize().expect("reference suffix completion");
        model
            .prefill_batch(
                workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &suffix[suffix.len() - 1..],
                    sequence: &mut reference,
                    output: Gemma4PrefillOutput::FullLogits,
                }],
                stream,
                cache,
            )
            .expect("reference final token");
        let full = model
            .logits_to_host(&reference.state, stream)
            .expect("full logits");
        readout
            .stage_sequence(0, &reference, stream)
            .expect("stage reference row");
        let compact_reference = readout
            .selected_logits(model, 1, stream)
            .expect("compact reference logits");
        let head_error = readout
            .token_ids()
            .iter()
            .enumerate()
            .map(|(row, token)| (compact_reference[row] - full[*token as usize]).abs())
            .fold(0.0f32, f32::max);
        let fork_error = compact
            .iter()
            .zip(&compact_reference)
            .map(|(branch, reference)| (branch - reference).abs())
            .fold(0.0f32, f32::max);
        assert!(
            head_error <= 1.0e-4,
            "prefix={prefix_len} head_error={head_error}"
        );
        assert!(
            fork_error <= 1.0e-4,
            "prefix={prefix_len} fork_error={fork_error}"
        );

        parent.finish(cache, stream).expect("finish parent");
        branch.finish(cache, stream).expect("finish branch");
        reference.finish(cache, stream).expect("finish reference");
        (compact, compact_reference)
    }

    fn assert_logits_match(label: &str, actual: &[f32], expected: &[f32]) {
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 1.0e-4, "{label} max_error={max_error}");
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_decision_head_matches_full_vocab_after_aligned_and_unaligned_forks() {
        let model = Gemma4Model::load(local_model_dir()).expect("load Gemma 4");
        let stream = CudaStream::new_blocking().expect("stream");
        let max_tokens = SM12X_KV_PAGE_TOKENS + 4;
        let mut cache = crate::gemma4::new_gemma4_sequence_cache(&model, 3, max_tokens)
            .expect("sequence cache");
        let mut workspace = model
            .new_prefill_batch_workspace(1, max_tokens, max_tokens)
            .expect("prefill workspace");
        let label_ids = (0..64).collect::<Vec<_>>();
        let mut readout = model
            .new_decision_readout(&label_ids, 1)
            .expect("decision readout");
        let full_bytes = model
            .new_sequence_state(max_tokens)
            .expect("full state")
            .device_bytes();
        let decision_bytes = model
            .new_decision_sequence_state(max_tokens)
            .expect("decision state")
            .device_bytes();
        let vocabulary_scratch_bytes = model.vocab_size() * (size_of::<f32>() + size_of::<u32>())
            + size_of::<u32>()
            + size_of::<f32>();
        assert_eq!(full_bytes - decision_bytes, vocabulary_scratch_bytes);

        let (aligned_branch, aligned_reference) = assert_decision_fork_matches_full_head(
            &model,
            SM12X_KV_PAGE_TOKENS,
            &stream,
            &mut cache,
            &mut workspace,
            &mut readout,
        );
        let (aligned_branch_repeat, aligned_reference_repeat) =
            assert_decision_fork_matches_full_head(
                &model,
                SM12X_KV_PAGE_TOKENS,
                &stream,
                &mut cache,
                &mut workspace,
                &mut readout,
            );
        assert_logits_match(
            "repeated aligned branch",
            &aligned_branch_repeat,
            &aligned_branch,
        );
        assert_logits_match(
            "repeated aligned reference",
            &aligned_reference_repeat,
            &aligned_reference,
        );

        let (unaligned_branch, unaligned_reference) = assert_decision_fork_matches_full_head(
            &model,
            SM12X_KV_PAGE_TOKENS + 1,
            &stream,
            &mut cache,
            &mut workspace,
            &mut readout,
        );
        let (unaligned_branch_repeat, unaligned_reference_repeat) =
            assert_decision_fork_matches_full_head(
                &model,
                SM12X_KV_PAGE_TOKENS + 1,
                &stream,
                &mut cache,
                &mut workspace,
                &mut readout,
            );
        assert_logits_match(
            "repeated unaligned branch",
            &unaligned_branch_repeat,
            &unaligned_branch,
        );
        assert_logits_match(
            "repeated unaligned reference",
            &unaligned_reference_repeat,
            &unaligned_reference,
        );
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_decision_tree_matches_isolated_branches() {
        let model = Gemma4Model::load(local_model_dir()).expect("load Gemma 4");
        let stream = CudaStream::new_blocking().expect("stream");
        let prefix = vec![2, 17, 23, 31];
        let branch_tokens = [vec![3, 4, 5], vec![6, 7], vec![8, 9, 10, 11]];
        let branch_slices = branch_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let total_tokens = prefix.len() + branch_tokens.iter().map(Vec::len).sum::<usize>();
        let max_tokens = prefix.len() + branch_tokens.iter().map(Vec::len).max().unwrap();
        let mut cache =
            crate::gemma4::new_gemma4_sequence_cache(&model, branch_tokens.len() + 1, max_tokens)
                .expect("sequence cache");
        let mut workspace = model
            .new_prefill_batch_workspace(branch_tokens.len(), total_tokens, max_tokens)
            .expect("prefill workspace");
        let label_ids = (0..64).collect::<Vec<_>>();
        let mut readout = model
            .new_decision_readout(&label_ids, branch_tokens.len())
            .expect("decision readout");

        let mut parent = Gemma4Sequence::admit_decision(&model, &mut cache, prefix.len(), &stream)
            .expect("parent");
        model
            .prefill_batch(
                &mut workspace,
                &mut [Gemma4PrefillRow {
                    token_ids: &prefix,
                    sequence: &mut parent,
                    output: Gemma4PrefillOutput::None,
                }],
                &stream,
                &mut cache,
            )
            .expect("parent prefill");
        stream.synchronize().expect("parent prefill completion");
        let mut branches = branch_tokens
            .iter()
            .map(|tokens| {
                Gemma4Sequence::branch_decision(
                    &model,
                    &parent,
                    &mut cache,
                    prefix.len() + tokens.len(),
                    &stream,
                )
                .expect("branch")
                .expect("branch capacity")
            })
            .collect::<Vec<_>>();
        stream.synchronize().expect("branch completion");
        let mut rows = branch_tokens
            .iter()
            .zip(&mut branches)
            .map(|(tokens, sequence)| Gemma4PrefillRow {
                token_ids: tokens,
                sequence,
                output: Gemma4PrefillOutput::FinalHidden,
            })
            .collect::<Vec<_>>();
        model
            .prefill_batch(&mut workspace, &mut rows, &stream, &mut cache)
            .expect("branch prefill");
        drop(rows);
        for (row, branch) in branches.iter().enumerate() {
            readout
                .stage_sequence(row, branch, &stream)
                .expect("stage branch");
        }
        let isolated = readout
            .selected_logits(&model, branch_tokens.len(), &stream)
            .expect("isolated logits");
        let tree = model
            .decision_tree_selected_logits(
                &mut workspace,
                &mut readout,
                &prefix,
                &branch_slices,
                &stream,
            )
            .expect("tree logits");
        let mut individual_tree = Vec::with_capacity(tree.len());
        for branch in &branch_tokens {
            let logits = model
                .decision_tree_selected_logits(
                    &mut workspace,
                    &mut readout,
                    &prefix,
                    &[branch.as_slice()],
                    &stream,
                )
                .expect("individual tree logits");
            individual_tree.extend(logits);
        }

        let label_count = label_ids.len();
        for branch in 0..branch_tokens.len() {
            let isolated = &isolated[branch * label_count..(branch + 1) * label_count];
            let tree = &tree[branch * label_count..(branch + 1) * label_count];
            let individual = &individual_tree[branch * label_count..(branch + 1) * label_count];
            let isolated_top = isolated
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0;
            let tree_top = tree
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0;
            let individual_top = individual
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0;
            let cache_error = isolated
                .iter()
                .zip(tree)
                .map(|(isolated, tree)| (isolated - tree).abs())
                .fold(0.0f32, f32::max);
            let packing_error = individual
                .iter()
                .zip(tree)
                .map(|(individual, tree)| (individual - tree).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "branch={branch} cache_top={isolated_top} tree_top={tree_top} individual_top={individual_top} cache_error={cache_error} packing_error={packing_error}"
            );
            assert_eq!(
                tree_top, individual_top,
                "branch={branch} packing_error={packing_error}"
            );
        }

        parent.finish(&mut cache, &stream).expect("finish parent");
        for branch in branches {
            branch.finish(&mut cache, &stream).expect("finish branch");
        }
    }

    #[test]
    #[ignore = "requires the local Gemma 4 checkpoint"]
    fn local_decision_tree_real_prompt_matches_individual_branches() {
        let model_dir = local_model_dir();
        let compiler = DecisionPromptCompiler::new(
            CheckpointChatTemplate::from_model_dir(&model_dir).expect("load chat template"),
        )
        .expect("decision compiler");
        let long_review = [
            "The passes are expensive after parking. The waterpark and lazy river are pleasant on a hot day, but traffic is terrible.",
            "The family pool is crowded, children reserve unused chairs, and staff do not resolve conflicts.",
            "The drinks are expensive, weak, and too sugary. An adults-only area would make the visit much more relaxing.",
            "Weekends are too busy, weekdays are not much better, and I am looking for another pool this summer.",
        ]
        .repeat(12)
        .join(" ");
        let request = compiler
            .prepare(DecisionPromptRequest {
                state: json!({
                    "movie_review": "weighty and ponderous but every bit as filling as the treat of the title.",
                    "news_article": "California adopted rules intended to reduce dairy-farm air pollution.",
                    "business_review": long_review,
                }),
                questions: vec![
                    (
                        "positive_sentiment".to_string(),
                        DecisionPromptQuestion::Noul {
                            instructions: json!("Does movie_review express positive sentiment?"),
                            true_description: "The review is positive.".to_string(),
                            false_description: "The review is negative.".to_string(),
                        },
                    ),
                    (
                        "news_topic".to_string(),
                        DecisionPromptQuestion::Choice {
                            instructions: json!("Which topic best describes news_article?"),
                            options: vec![
                                (
                                    "world".to_string(),
                                    Some("World politics and international events.".to_string()),
                                ),
                                (
                                    "sports".to_string(),
                                    Some("Sports teams, athletes, and competitions.".to_string()),
                                ),
                                (
                                    "business".to_string(),
                                    Some(
                                        "Companies, markets, finance, and the economy.".to_string(),
                                    ),
                                ),
                                (
                                    "science_and_technology".to_string(),
                                    Some("Science, computing, and technology.".to_string()),
                                ),
                            ],
                        },
                    ),
                    (
                        "star_rating".to_string(),
                        DecisionPromptQuestion::Score {
                            instructions: json!(
                                "What star rating did the writer give business_review?"
                            ),
                            levels: [
                                "One star.",
                                "Two stars.",
                                "Three stars.",
                                "Four stars.",
                                "Five stars.",
                            ]
                            .into_iter()
                            .map(str::to_string)
                            .collect(),
                        },
                    ),
                ],
            })
            .expect("compile decision request");
        let model = Gemma4Model::load(&model_dir).expect("load Gemma 4");
        let stream = CudaStream::new_blocking().expect("stream");
        let branches = request
            .branches
            .iter()
            .map(|branch| branch.suffix_tokens.as_slice())
            .collect::<Vec<_>>();
        let total_tokens = request.prefix_tokens.len()
            + request
                .branches
                .iter()
                .map(|branch| branch.suffix_tokens.len())
                .sum::<usize>();
        let max_tokens = request.longest_branch_tokens();
        assert!(
            request.prefix_tokens.len() > 512,
            "regression prompt must exercise a long shared prefix"
        );
        let mut workspace = model
            .new_prefill_batch_workspace(branches.len(), total_tokens, max_tokens)
            .expect("prefill workspace");
        let mut readout = model
            .new_decision_readout(&request.label_token_ids, branches.len())
            .expect("decision readout");
        let packed_logits = model
            .decision_tree_selected_logits(
                &mut workspace,
                &mut readout,
                &request.prefix_tokens,
                &branches,
                &stream,
            )
            .expect("packed decision tree");
        let labels = request.label_token_ids.len();
        for (branch_index, branch) in branches.iter().enumerate() {
            let individual_logits = model
                .decision_tree_selected_logits(
                    &mut workspace,
                    &mut readout,
                    &request.prefix_tokens,
                    &[*branch],
                    &stream,
                )
                .expect("individual decision tree");
            let packed = &packed_logits[branch_index * labels..(branch_index + 1) * labels];
            let packed_top = packed
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0;
            let individual_top = individual_logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0;
            let logit_error = packed
                .iter()
                .zip(&individual_logits)
                .map(|(packed, individual)| (packed - individual).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                packed_top, individual_top,
                "branch={branch_index} logit_error={logit_error}"
            );
            assert!(
                logit_error <= 1.0e-4,
                "branch={branch_index} logit_error={logit_error}"
            );
        }
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
    fn local_active_rows_match_exact_workspace() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/gemma-4-26b-a4b-nvfp4");
        let model = Gemma4Model::load(model_dir).expect("load Gemma 4");
        let tokens = [2, 3, 2, 3];
        let mut cache = crate::gemma4::new_gemma4_sequence_cache(&model, 2, tokens.len())
            .expect("sequence cache");
        let mut exact = model
            .new_prefill_batch_workspace(1, tokens.len(), tokens.len())
            .expect("exact workspace");
        let mut padded = model
            .new_prefill_batch_workspace(1, 8, tokens.len())
            .expect("padded workspace");
        let stream = CudaStream::new_blocking().expect("stream");
        let mut exact_state = Gemma4Sequence::admit(&model, &mut cache, tokens.len(), &stream)
            .expect("exact sequence");
        let mut padded_state = Gemma4Sequence::admit(&model, &mut cache, tokens.len(), &stream)
            .expect("padded sequence");

        model
            .prefill_batch(
                &mut exact,
                &mut [Gemma4PrefillRow {
                    token_ids: &tokens,
                    sequence: &mut exact_state,
                    output: Gemma4PrefillOutput::None,
                }],
                &stream,
                &mut cache,
            )
            .expect("exact prefill");
        model
            .prefill_batch(
                &mut padded,
                &mut [Gemma4PrefillRow {
                    token_ids: &tokens,
                    sequence: &mut padded_state,
                    output: Gemma4PrefillOutput::None,
                }],
                &stream,
                &mut cache,
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
