//! Shared tensor-core prompt attention over compact paged K/V storage.

use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaStream, DeviceBuffer, DeviceRepr, GemmShape, Result,
    Sm12xKvPagePool, causal_window_softmax_f32_to_bf16_on_stream,
    pack_token_heads_bf16_at_offset_into_on_stream, pack_tree_kv_bf16_into_on_stream,
    pack_value_heads_bf16_into_on_stream, unpack_heads_f32_at_offset_into_on_stream,
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

    pub(crate) fn tree_rows_fit(&self, rows: usize) -> bool {
        rows != 0
            && rows
                .checked_mul(self.q_heads)
                .and_then(|values| values.checked_mul(self.head_dim))
                .is_some()
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

    #[allow(clippy::too_many_arguments)]
    fn run_direct_segment(
        &mut self,
        query: &DeviceBuffer<f32>,
        input_row_offset: usize,
        query_rows: usize,
        logical_query_start: usize,
        logical_tokens: usize,
        window_tokens: Option<usize>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let queries_per_kv = self.q_heads / self.kv_heads;
        let mut query_offset = 0;
        while query_offset < query_rows {
            let absolute_query_start = logical_query_start + query_offset;
            let requested = query_rows - query_offset;
            let tentative_key_start = window_tokens
                .map(|window| (absolute_query_start + 1).saturating_sub(window))
                .unwrap_or(0);
            let tentative_key_tokens = absolute_query_start + requested - tentative_key_start;
            let tile_rows = self.tile_rows(requested, tentative_key_tokens);
            let key_start = window_tokens
                .map(|window| (absolute_query_start + 1).saturating_sub(window))
                .unwrap_or(0);
            let key_tokens = absolute_query_start + tile_rows - key_start;
            Self::grow(
                &mut self.packed_query,
                tile_rows * self.q_heads * self.head_dim,
            )?;
            let score_values = tile_rows * self.q_heads * key_tokens;
            Self::grow(&mut self.scores, score_values)?;
            Self::grow(&mut self.probabilities, score_values)?;
            Self::grow(
                &mut self.packed_output,
                tile_rows * self.q_heads * self.head_dim,
            )?;
            pack_token_heads_bf16_at_offset_into_on_stream(
                query,
                self.packed_query.output(),
                tile_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;

            let qk_key = (key_tokens, tile_rows, logical_tokens);
            if !self.qk_plans.contains_key(&qk_key) {
                self.qk_plans.insert(
                    qk_key,
                    Bf16TnMatmulPlan::new_strided_batch(
                        &self.lt,
                        GemmShape::new(key_tokens, tile_rows * queries_per_kv, self.head_dim),
                        self.kv_heads,
                        logical_tokens * self.head_dim,
                        queries_per_kv * tile_rows * self.head_dim,
                        queries_per_kv * tile_rows * key_tokens,
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
                tile_rows,
                key_tokens,
                absolute_query_start - key_start,
                self.q_heads,
                self.head_dim,
                window_tokens,
                stream,
            )?;

            let pv_key = (key_tokens, tile_rows, logical_tokens, key_start);
            if !self.pv_plans.contains_key(&pv_key) {
                self.pv_plans.insert(
                    pv_key,
                    Bf16TnMatmulPlan::new_strided_batch_with_a_leading_dimension(
                        &self.lt,
                        GemmShape::new(self.head_dim, tile_rows * queries_per_kv, key_tokens),
                        logical_tokens,
                        self.kv_heads,
                        self.head_dim * logical_tokens,
                        queries_per_kv * tile_rows * key_tokens,
                        queries_per_kv * tile_rows * self.head_dim,
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
                tile_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;
            query_offset += tile_rows;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_tree(
        &mut self,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        lengths: &[usize],
        window_tokens: Option<usize>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some((&prefix_tokens, branch_lengths)) = lengths.split_first() else {
            return Err(eider_cuda::Error::Shape {
                label: "tree prefill attention segments",
                expected: "one prefix and at least one branch".to_string(),
                actual: "no segments".to_string(),
            });
        };
        let rows = lengths.iter().try_fold(0usize, |total, length| {
            total
                .checked_add(*length)
                .ok_or_else(|| eider_cuda::Error::Shape {
                    label: "tree prefill attention rows",
                    expected: "segment total without overflow".to_string(),
                    actual: format!("lengths={lengths:?}"),
                })
        })?;
        let q_values = rows
            .checked_mul(self.q_heads)
            .and_then(|values| values.checked_mul(self.head_dim))
            .unwrap_or(usize::MAX);
        let kv_values = rows
            .checked_mul(self.kv_heads)
            .and_then(|values| values.checked_mul(self.head_dim))
            .unwrap_or(usize::MAX);
        if prefix_tokens == 0
            || branch_lengths.is_empty()
            || branch_lengths.contains(&0)
            || !self.tree_rows_fit(rows)
            || query.len() < q_values
            || key.len() < kv_values
            || value.len() < kv_values
            || output.len() < q_values
        {
            return Err(eider_cuda::Error::Shape {
                label: "tree prefill attention",
                expected: "non-empty prefix and branches with matching buffers".to_string(),
                actual: format!(
                    "lengths={lengths:?} query={} key={} value={} output={}",
                    query.len(),
                    key.len(),
                    value.len(),
                    output.len()
                ),
            });
        }

        let prefix_kv_values = prefix_tokens * self.kv_heads * self.head_dim;
        Self::grow(&mut self.packed_key, prefix_kv_values)?;
        Self::grow(&mut self.packed_value, prefix_kv_values)?;
        pack_token_heads_bf16_at_offset_into_on_stream(
            key,
            self.packed_key.output(),
            prefix_tokens,
            self.kv_heads,
            self.head_dim,
            0,
            stream,
        )?;
        pack_value_heads_bf16_into_on_stream(
            value,
            self.packed_value.output(),
            prefix_tokens,
            self.kv_heads,
            self.head_dim,
            stream,
        )?;
        self.run_direct_segment(
            query,
            0,
            prefix_tokens,
            0,
            prefix_tokens,
            window_tokens,
            output,
            stream,
        )?;

        let mut branch_offset = prefix_tokens;
        for &branch_tokens in branch_lengths {
            let logical_tokens = prefix_tokens + branch_tokens;
            let logical_kv_values = logical_tokens * self.kv_heads * self.head_dim;
            Self::grow(&mut self.packed_key, logical_kv_values)?;
            Self::grow(&mut self.packed_value, logical_kv_values)?;
            pack_tree_kv_bf16_into_on_stream(
                key,
                value,
                self.packed_key.output(),
                self.packed_value.output(),
                prefix_tokens,
                branch_tokens,
                branch_offset,
                self.kv_heads,
                self.head_dim,
                stream,
            )?;
            self.run_direct_segment(
                query,
                branch_offset,
                branch_tokens,
                prefix_tokens,
                logical_tokens,
                window_tokens,
                output,
                stream,
            )?;
            branch_offset += branch_tokens;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_attention_is_invariant_to_sibling_rows_at_real_prompt_shape() {
        let prefix = 603;
        let branches = [57, 104, 98];
        let rows = prefix + branches.iter().sum::<usize>();
        let q_heads = 16;
        let kv_heads = 8;
        let head_dim = 256;
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let query = (0..rows * q_width)
            .map(|index| ((index % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let key = (0..rows * kv_width)
            .map(|index| ((index * 3 % 263) as f32 - 131.0) / 131.0)
            .collect::<Vec<_>>();
        let value = (0..rows * kv_width)
            .map(|index| ((index * 5 % 269) as f32 - 134.0) / 134.0)
            .collect::<Vec<_>>();
        let query_device = DeviceBuffer::from_host(&query).expect("query");
        let key_device = DeviceBuffer::from_host(&key).expect("key");
        let value_device = DeviceBuffer::from_host(&value).expect("value");
        let mut packed_output = DeviceBuffer::zeroed(rows * q_width).expect("packed output");
        let stream = CudaStream::new_blocking().expect("stream");
        let mut attention = PagedTensorCorePrefillAttention::new(rows, q_heads, kv_heads, head_dim)
            .expect("attention");
        attention
            .run_tree(
                &query_device,
                &key_device,
                &value_device,
                &[prefix, branches[0], branches[1], branches[2]],
                None,
                &mut packed_output,
                &stream,
            )
            .expect("packed tree attention");
        let packed_output = packed_output
            .copy_to_host(&stream)
            .expect("download packed output");

        let mut branch_offset = prefix;
        for (branch_index, &branch_rows) in branches.iter().enumerate() {
            let logical_rows = prefix + branch_rows;
            let mut individual_query = Vec::with_capacity(logical_rows * q_width);
            individual_query.extend_from_slice(&query[..prefix * q_width]);
            individual_query.extend_from_slice(
                &query[branch_offset * q_width..(branch_offset + branch_rows) * q_width],
            );
            let mut individual_key = Vec::with_capacity(logical_rows * kv_width);
            individual_key.extend_from_slice(&key[..prefix * kv_width]);
            individual_key.extend_from_slice(
                &key[branch_offset * kv_width..(branch_offset + branch_rows) * kv_width],
            );
            let mut individual_value = Vec::with_capacity(logical_rows * kv_width);
            individual_value.extend_from_slice(&value[..prefix * kv_width]);
            individual_value.extend_from_slice(
                &value[branch_offset * kv_width..(branch_offset + branch_rows) * kv_width],
            );
            let mut individual_output =
                DeviceBuffer::zeroed(logical_rows * q_width).expect("individual output");
            attention
                .run_tree(
                    &DeviceBuffer::from_host(&individual_query).expect("individual query"),
                    &DeviceBuffer::from_host(&individual_key).expect("individual key"),
                    &DeviceBuffer::from_host(&individual_value).expect("individual value"),
                    &[prefix, branch_rows],
                    None,
                    &mut individual_output,
                    &stream,
                )
                .expect("individual tree attention");
            let individual_output = individual_output
                .copy_to_host(&stream)
                .expect("download individual output");
            let mut max_error = 0.0f32;
            let mut mismatches = 0usize;
            for logical_row in 0..logical_rows {
                let packed_row = if logical_row < prefix {
                    logical_row
                } else {
                    branch_offset + logical_row - prefix
                };
                let packed = &packed_output[packed_row * q_width..(packed_row + 1) * q_width];
                let individual =
                    &individual_output[logical_row * q_width..(logical_row + 1) * q_width];
                for (packed, individual) in packed.iter().zip(individual) {
                    max_error = max_error.max((packed - individual).abs());
                    mismatches += usize::from(packed.to_bits() != individual.to_bits());
                }
            }
            eprintln!(
                "branch={branch_index} attention max_error={max_error} mismatches={mismatches}"
            );
            assert_eq!(mismatches, 0, "branch={branch_index} max_error={max_error}");
            branch_offset += branch_rows;
        }
    }
}
