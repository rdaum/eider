//! Shared tensor-core prompt attention over compact paged K/V storage.

use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaStream, DeviceBuffer, DeviceRepr, GemmShape, Result,
    Sm12xKvPagePool, causal_window_softmax_f32_to_bf16_on_stream,
    pack_token_heads_bf16_at_offset_into_on_stream, unpack_heads_f32_at_offset_into_on_stream,
};
use std::collections::HashMap;
use std::mem::size_of;

const CUBLAS_WORKSPACE_LIMIT: u64 = 4 << 20;
const SCORE_BUDGET_BYTES: usize = 192 << 20;
const QUERY_TILE_ROWS: usize = 256;

pub(crate) struct PagedTensorCorePrefillAttention {
    lt: CublasLt,
    qk_plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    pv_plans: HashMap<(usize, usize, usize, usize), Bf16TnMatmulPlan>,
    packed_query: DeviceBuffer<u16>,
    packed_key: DeviceBuffer<u16>,
    packed_value: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    probabilities: DeviceBuffer<u16>,
    packed_output: DeviceBuffer<f32>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl PagedTensorCorePrefillAttention {
    pub(crate) fn new(
        rows: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            lt: CublasLt::new()?,
            qk_plans: HashMap::new(),
            pv_plans: HashMap::new(),
            packed_query: DeviceBuffer::zeroed(rows * q_heads * head_dim)?,
            packed_key: DeviceBuffer::zeroed(rows * kv_heads * head_dim)?,
            packed_value: DeviceBuffer::zeroed(rows * kv_heads * head_dim)?,
            scores: DeviceBuffer::zeroed(rows.min(QUERY_TILE_ROWS) * q_heads * rows)?,
            probabilities: DeviceBuffer::zeroed(rows.min(QUERY_TILE_ROWS) * q_heads * rows)?,
            packed_output: DeviceBuffer::zeroed(rows * q_heads * head_dim)?,
            q_heads,
            kv_heads,
            head_dim,
        })
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.packed_query.device_bytes()
            + self.packed_key.device_bytes()
            + self.packed_value.device_bytes()
            + self.scores.device_bytes()
            + self.probabilities.device_bytes()
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

    fn grow<T: DeviceRepr>(buffer: &mut DeviceBuffer<T>, required: usize) -> Result<()> {
        if buffer.len() < required {
            *buffer = DeviceBuffer::zeroed(required)?;
        }
        Ok(())
    }

    fn tile_rows(&self, requested: usize, key_tokens: usize) -> usize {
        let values_per_row = self.q_heads.saturating_mul(key_tokens).max(1);
        let budget_rows = (SCORE_BUDGET_BYTES / size_of::<f32>())
            .checked_div(values_per_row)
            .unwrap_or(0)
            .max(1);
        let rows = requested.min(budget_rows).min(QUERY_TILE_ROWS);
        if rows >= 16 { rows / 16 * 16 } else { rows }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        pool: &Sm12xKvPagePool,
        page_table: &DeviceBuffer<u32>,
        start_position: usize,
        query: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        window_tokens: Option<usize>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let cache_tokens = start_position + rows;
        let kv_values = cache_tokens * self.kv_heads * self.head_dim;
        Self::grow(&mut self.packed_key, kv_values)?;
        Self::grow(&mut self.packed_value, kv_values)?;
        pool.unpack_paged_bf16_on_stream(
            page_table,
            cache_tokens,
            self.packed_key.output(),
            self.packed_value.output(),
            stream,
        )?;
        let queries_per_kv = self.q_heads / self.kv_heads;
        let mut query_offset = 0;
        while query_offset < rows {
            let absolute_query_start = start_position + query_offset;
            let requested = rows - query_offset;
            let tentative_key_start = window_tokens
                .map(|window| (absolute_query_start + 1).saturating_sub(window))
                .unwrap_or(0);
            let tentative_key_tokens = absolute_query_start + requested - tentative_key_start;
            let query_rows = self.tile_rows(requested, tentative_key_tokens);
            let key_start = window_tokens
                .map(|window| (absolute_query_start + 1).saturating_sub(window))
                .unwrap_or(0);
            let key_tokens = absolute_query_start + query_rows - key_start;
            Self::grow(
                &mut self.packed_query,
                query_rows * self.q_heads * self.head_dim,
            )?;
            let score_values = query_rows * self.q_heads * key_tokens;
            Self::grow(&mut self.scores, score_values)?;
            Self::grow(&mut self.probabilities, score_values)?;
            Self::grow(
                &mut self.packed_output,
                query_rows * self.q_heads * self.head_dim,
            )?;
            pack_token_heads_bf16_at_offset_into_on_stream(
                query,
                self.packed_query.output(),
                query_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;

            let qk_key = (key_tokens, query_rows, cache_tokens);
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
                        CUBLAS_WORKSPACE_LIMIT,
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
                self.probabilities.output(),
                query_rows,
                key_tokens,
                absolute_query_start - key_start,
                self.q_heads,
                self.head_dim,
                window_tokens,
                stream,
            )?;

            let pv_key = (key_tokens, query_rows, cache_tokens, key_start);
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
                        CUBLAS_WORKSPACE_LIMIT,
                    )?,
                );
            }
            self.pv_plans[&pv_key].run_offsets_on_stream(
                &self.lt,
                &self.packed_value,
                key_start,
                &self.probabilities,
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
