#![allow(clippy::too_many_arguments)]

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::kernels::non_gemm::MoeSortedRoutes;

const SCALE_BLOCK: usize = 128;
const HYPER_STREAMS: usize = 4;
const HYPER_MIX: usize = 24;

/// Pointer tables and lengths for one batched DeepSeek attention operation.
pub struct Deepseek4AttentionBatch<'a> {
    /// Per-sequence pointers to `[sliding_capacity, head_dim]` ring storage.
    pub sliding_tables: &'a DeviceBuffer<*const f32>,
    /// Number of valid chronological entries in each sliding ring.
    pub sliding_lengths: &'a DeviceBuffer<u32>,
    /// Oldest valid physical slot in each sliding ring.
    pub sliding_starts: &'a DeviceBuffer<u32>,
    /// Per-sequence pointers to `[compressed_length, head_dim]` storage.
    pub compressed_tables: &'a DeviceBuffer<*const f32>,
    /// Number of valid compressed entries for each sequence.
    pub compressed_lengths: &'a DeviceBuffer<u32>,
    /// CSA indices, flattened as `[batch, selected_count]`; absent for HCA.
    pub selected_indices: Option<&'a DeviceBuffer<i32>>,
}

/// Prior cache and current-chunk metadata for causal prefill or decode.
pub struct Deepseek4CausalAttentionBatch<'a> {
    /// Per-query pointers to `[sliding_capacity, head_dim]` prior ring storage.
    pub sliding_tables: &'a DeviceBuffer<*const f32>,
    /// Number of valid prior entries for each query.
    pub sliding_lengths: &'a DeviceBuffer<u32>,
    /// Oldest valid physical prior slot for each query.
    pub sliding_starts: &'a DeviceBuffer<u32>,
    /// Current chunks concatenated as `[current_rows, head_dim]`.
    pub current_kv: &'a DeviceBuffer<f32>,
    /// Row at which each query's current sequence chunk begins.
    pub current_sequence_starts: &'a DeviceBuffer<u32>,
    /// Query row relative to its current sequence chunk.
    pub query_offsets: &'a DeviceBuffer<u32>,
    /// Absolute token position for every query.
    pub positions: &'a DeviceBuffer<u32>,
    /// Per-query pointers to completed compressed entries.
    pub compressed_tables: &'a DeviceBuffer<*const f32>,
    /// Number of completed compressed entries for each query.
    pub compressed_lengths: &'a DeviceBuffer<u32>,
    /// CSA selections and their logical per-row width.
    pub selected_indices: Option<(&'a DeviceBuffer<i32>, usize)>,
}

/// Applies a row-major block-scaled E4M3 weight to F32 activation rows.
pub fn block_fp8_linear_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = batch_rows.saturating_mul(cols);
    let weight_len = rows.saturating_mul(cols);
    let scale_len = (rows / SCALE_BLOCK).saturating_mul(cols / SCALE_BLOCK);
    let output_len = batch_rows.saturating_mul(rows);
    if batch_rows == 0
        || rows == 0
        || cols == 0
        || !rows.is_multiple_of(SCALE_BLOCK)
        || !cols.is_multiple_of(SCALE_BLOCK)
        || [batch_rows, rows, cols]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || input.len() < input_len
        || weight.len() != weight_len
        || weight_scale.len() != scale_len
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 block-scaled FP8 linear",
            expected: format!(
                "batch>0 rows/cols multiple of {SCALE_BLOCK}, input>={input_len} weight={weight_len} scales={scale_len} output>={output_len}"
            ),
            actual: format!(
                "batch={batch_rows} rows={rows} cols={cols} input={} weight={} scales={} output={}",
                input.len(),
                weight.len(),
                weight_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_block_fp8_linear_f32_on_stream",
            ffi::infer_deepseek4_block_fp8_linear_f32_on_stream(
                input.as_const_ptr().cast(),
                weight.as_const_ptr().cast(),
                weight_scale.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                batch_rows as u32,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies independent block-scaled FP8 projections to contiguous input groups.
#[allow(clippy::too_many_arguments)]
pub fn block_fp8_grouped_linear_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    groups: usize,
    rows_per_group: usize,
    cols_per_group: usize,
    stream: &CudaStream,
) -> Result<()> {
    let total_rows = groups.saturating_mul(rows_per_group);
    let total_cols = groups.saturating_mul(cols_per_group);
    let input_len = batch_rows.saturating_mul(total_cols);
    let weight_len = total_rows.saturating_mul(cols_per_group);
    let scale_len = (total_rows / SCALE_BLOCK).saturating_mul(cols_per_group / SCALE_BLOCK);
    let output_len = batch_rows.saturating_mul(total_rows);
    if batch_rows == 0
        || groups == 0
        || rows_per_group == 0
        || cols_per_group == 0
        || !rows_per_group.is_multiple_of(SCALE_BLOCK)
        || !cols_per_group.is_multiple_of(SCALE_BLOCK)
        || [
            batch_rows,
            groups,
            rows_per_group,
            cols_per_group,
            total_rows,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
        || input.len() < input_len
        || weight.len() != weight_len
        || weight_scale.len() != scale_len
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 grouped block-scaled FP8 linear",
            expected: format!(
                "input>={input_len} weight={weight_len} scales={scale_len} output>={output_len}"
            ),
            actual: format!(
                "batch={batch_rows} groups={groups} rows/group={rows_per_group} cols/group={cols_per_group} input={} weight={} scales={} output={}",
                input.len(),
                weight.len(),
                weight_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_block_fp8_grouped_linear_f32_on_stream",
            ffi::infer_deepseek4_block_fp8_grouped_linear_f32_on_stream(
                input.as_const_ptr().cast(),
                weight.as_const_ptr().cast(),
                weight_scale.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                batch_rows as u32,
                groups as u32,
                rows_per_group as u32,
                cols_per_group as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies a row-major block-scaled E4M3 weight to one F32 activation row.
pub fn block_fp8_linear_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    block_fp8_linear_f32_batch_into_on_stream(
        input,
        weight,
        weight_scale,
        output,
        1,
        rows,
        cols,
        stream,
    )
}

/// Computes exact DeepSeek V4 mHC post/combination weights and collapsed input.
#[allow(clippy::too_many_arguments)]
pub fn hyper_prepare_f32_batch_into_on_stream(
    streams: &DeviceBuffer<f32>,
    function: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    mut post: DeviceOutput<'_, f32>,
    mut combination: DeviceOutput<'_, f32>,
    mut collapsed: DeviceOutput<'_, f32>,
    batch_rows: usize,
    hidden: usize,
    rms_eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
    stream: &CudaStream,
) -> Result<()> {
    let stream_len = batch_rows
        .checked_mul(HYPER_STREAMS)
        .and_then(|value| value.checked_mul(hidden))
        .unwrap_or(usize::MAX);
    let function_len = HYPER_MIX
        .checked_mul(HYPER_STREAMS)
        .and_then(|value| value.checked_mul(hidden))
        .unwrap_or(usize::MAX);
    let post_len = batch_rows.saturating_mul(HYPER_STREAMS);
    let combination_len = post_len.saturating_mul(HYPER_STREAMS);
    let collapsed_len = batch_rows.saturating_mul(hidden);
    if batch_rows == 0
        || hidden == 0
        || sinkhorn_iters == 0
        || [batch_rows, hidden, sinkhorn_iters]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || !rms_eps.is_finite()
        || rms_eps <= 0.0
        || !hc_eps.is_finite()
        || hc_eps <= 0.0
        || streams.len() < stream_len
        || function.len() != function_len
        || base.len() != HYPER_MIX
        || scale.len() != 3
        || post.len() < post_len
        || combination.len() < combination_len
        || collapsed.len() < collapsed_len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 hyper-connection prepare",
            expected: format!(
                "streams>={stream_len} function={function_len} base={HYPER_MIX} scale=3 post>={post_len} comb>={combination_len} collapsed>={collapsed_len}"
            ),
            actual: format!(
                "batch={batch_rows} hidden={hidden} streams={} function={} base={} scale={} post={} comb={} collapsed={}",
                streams.len(),
                function.len(),
                base.len(),
                scale.len(),
                post.len(),
                combination.len(),
                collapsed.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_hyper_prepare_f32_on_stream",
            ffi::infer_deepseek4_hyper_prepare_f32_on_stream(
                streams.as_const_ptr().cast(),
                function.as_const_ptr().cast(),
                base.as_const_ptr().cast(),
                scale.as_const_ptr().cast(),
                post.as_mut_ptr().cast(),
                combination.as_mut_ptr().cast(),
                collapsed.as_mut_ptr().cast(),
                batch_rows as u32,
                hidden as u32,
                rms_eps,
                hc_eps,
                sinkhorn_iters as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies `post * sublayer + combᵀ * streams` for DeepSeek V4 mHC.
#[allow(clippy::too_many_arguments)]
pub fn hyper_apply_f32_batch_into_on_stream(
    streams: &DeviceBuffer<f32>,
    sublayer: &DeviceBuffer<f32>,
    post: &DeviceBuffer<f32>,
    combination: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    hidden: usize,
    stream: &CudaStream,
) -> Result<()> {
    let stream_len = batch_rows
        .checked_mul(HYPER_STREAMS)
        .and_then(|value| value.checked_mul(hidden))
        .unwrap_or(usize::MAX);
    let sublayer_len = batch_rows.saturating_mul(hidden);
    let post_len = batch_rows.saturating_mul(HYPER_STREAMS);
    let combination_len = post_len.saturating_mul(HYPER_STREAMS);
    if batch_rows == 0
        || hidden == 0
        || [batch_rows, hidden]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || streams.len() < stream_len
        || sublayer.len() < sublayer_len
        || post.len() < post_len
        || combination.len() < combination_len
        || output.len() < stream_len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 hyper-connection apply",
            expected: format!(
                "streams/output>={stream_len} sublayer>={sublayer_len} post>={post_len} comb>={combination_len}"
            ),
            actual: format!(
                "batch={batch_rows} hidden={hidden} streams={} sublayer={} post={} comb={} output={}",
                streams.len(),
                sublayer.len(),
                post.len(),
                combination.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_hyper_apply_f32_on_stream",
            ffi::infer_deepseek4_hyper_apply_f32_on_stream(
                streams.as_const_ptr().cast(),
                sublayer.as_const_ptr().cast(),
                post.as_const_ptr().cast(),
                combination.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                batch_rows as u32,
                hidden as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Collapses the final four DeepSeek mHC streams into one hidden row.
pub fn hyper_head_f32_batch_into_on_stream(
    streams: &DeviceBuffer<f32>,
    function: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    hidden: usize,
    rms_eps: f32,
    hc_eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let stream_len = batch_rows
        .saturating_mul(HYPER_STREAMS)
        .saturating_mul(hidden);
    let function_len = HYPER_STREAMS
        .saturating_mul(HYPER_STREAMS)
        .saturating_mul(hidden);
    let output_len = batch_rows.saturating_mul(hidden);
    if batch_rows == 0
        || hidden == 0
        || [batch_rows, hidden]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || !rms_eps.is_finite()
        || rms_eps <= 0.0
        || !hc_eps.is_finite()
        || hc_eps <= 0.0
        || streams.len() < stream_len
        || function.len() != function_len
        || base.len() != HYPER_STREAMS
        || scale.len() != 1
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 hyper head",
            expected: format!(
                "streams>={stream_len} function={function_len} base={HYPER_STREAMS} scale=1 output>={output_len}"
            ),
            actual: format!(
                "batch={batch_rows} hidden={hidden} streams={} function={} base={} scale={} output={}",
                streams.len(),
                function.len(),
                base.len(),
                scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_hyper_head_f32_on_stream",
            ffi::infer_deepseek4_hyper_head_f32_on_stream(
                streams.as_const_ptr().cast(),
                function.as_const_ptr().cast(),
                base.as_const_ptr().cast(),
                scale.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                batch_rows as u32,
                hidden as u32,
                rms_eps,
                hc_eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies DeepSeek V4's interleaved RoPE to the trailing channels in place.
///
/// `direction` is `1.0` for query/key rotation and `-1.0` for the conjugate
/// rotation applied to attention output.
#[allow(clippy::too_many_arguments)]
pub fn rope_interleaved_trailing_f32_indexed_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    inv_freq: &DeviceBuffer<f32>,
    positions: &DeviceBuffer<u32>,
    batch_rows: usize,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    direction: f32,
    stream: &CudaStream,
) -> Result<()> {
    let values_len = batch_rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    if batch_rows == 0
        || heads == 0
        || head_dim == 0
        || rope_dim == 0
        || rope_dim > head_dim
        || !rope_dim.is_multiple_of(2)
        || (direction != 1.0 && direction != -1.0)
        || [batch_rows, heads, head_dim, rope_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || values.len() < values_len
        || inv_freq.len() != rope_dim / 2
        || positions.len() < batch_rows
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 trailing interleaved RoPE",
            expected: format!(
                "values>={values_len} inv_freq={} positions>={batch_rows} even rope_dim<=head_dim direction=+/-1",
                rope_dim / 2
            ),
            actual: format!(
                "batch={batch_rows} heads={heads} head_dim={head_dim} rope_dim={rope_dim} direction={direction} values={} inv_freq={} positions={}",
                values.len(),
                inv_freq.len(),
                positions.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_rope_interleaved_trailing_f32_on_stream",
            ffi::infer_deepseek4_rope_interleaved_trailing_f32_on_stream(
                values.as_mut_ptr().cast(),
                inv_freq.as_const_ptr().cast(),
                positions.as_const_ptr().cast(),
                batch_rows as u32,
                heads as u32,
                head_dim as u32,
                rope_dim as u32,
                direction,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes batched DeepSeek shared-KV attention over sliding and compressed state.
#[allow(clippy::too_many_arguments)]
pub fn attention_f32_batch_into_on_stream(
    query: &DeviceBuffer<f32>,
    state: Deepseek4AttentionBatch<'_>,
    sinks: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    heads: usize,
    head_dim: usize,
    sliding_capacity: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values_len = batch_rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    let selected_count = match state.selected_indices {
        Some(indices) if batch_rows != 0 && indices.len().is_multiple_of(batch_rows) => {
            indices.len() / batch_rows
        }
        Some(indices) => {
            return Err(Error::Shape {
                label: "DeepSeek V4 CSA indices",
                expected: format!("length divisible by batch {batch_rows}"),
                actual: indices.len().to_string(),
            });
        }
        None => 0,
    };
    if batch_rows == 0
        || heads == 0
        || head_dim == 0
        || sliding_capacity == 0
        || [
            batch_rows,
            heads,
            head_dim,
            sliding_capacity,
            selected_count,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
        || query.len() < values_len
        || output.len() < values_len
        || sinks.len() != heads
        || state.sliding_tables.len() < batch_rows
        || state.sliding_lengths.len() < batch_rows
        || state.sliding_starts.len() < batch_rows
        || state.compressed_tables.len() < batch_rows
        || state.compressed_lengths.len() < batch_rows
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 attention",
            expected: format!(
                "query/output>={values_len} sinks={heads} batch metadata>={batch_rows}"
            ),
            actual: format!(
                "batch={batch_rows} heads={heads} head_dim={head_dim} capacity={sliding_capacity} query={} output={} sinks={} sliding_ptrs={} sliding_lengths={} sliding_starts={} compressed_ptrs={} compressed_lengths={} selected={selected_count}",
                query.len(),
                output.len(),
                sinks.len(),
                state.sliding_tables.len(),
                state.sliding_lengths.len(),
                state.sliding_starts.len(),
                state.compressed_tables.len(),
                state.compressed_lengths.len(),
            ),
        });
    }
    let selected_ptr = state
        .selected_indices
        .map_or(std::ptr::null(), |indices| indices.as_const_ptr().cast());
    unsafe {
        check_cuda(
            "infer_deepseek4_attention_f32_on_stream",
            ffi::infer_deepseek4_attention_f32_on_stream(
                query.as_const_ptr().cast(),
                state.sliding_tables.as_const_ptr().cast(),
                state.sliding_lengths.as_const_ptr().cast(),
                state.sliding_starts.as_const_ptr().cast(),
                state.compressed_tables.as_const_ptr().cast(),
                state.compressed_lengths.as_const_ptr().cast(),
                selected_ptr,
                sinks.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                batch_rows as u32,
                heads as u32,
                head_dim as u32,
                sliding_capacity as u32,
                selected_count as u32,
                1.0 / (head_dim as f32).sqrt(),
                stream.as_raw(),
            ),
        )
    }
}

/// Computes causal DeepSeek attention over prior state and current chunks.
///
/// A ring contains at most `sliding_capacity` tokens preceding the current
/// chunk, so the effective sliding window is `sliding_capacity + 1` including
/// the current query token. Completed compressed entries are independently
/// bounded by each query's absolute causal threshold.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention_f32_batch_into_on_stream(
    query: &DeviceBuffer<f32>,
    state: Deepseek4CausalAttentionBatch<'_>,
    sinks: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    batch_rows: usize,
    heads: usize,
    head_dim: usize,
    sliding_capacity: usize,
    compression_ratio: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values_len = batch_rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    let current_rows = if head_dim == 0 || !state.current_kv.len().is_multiple_of(head_dim) {
        0
    } else {
        state.current_kv.len() / head_dim
    };
    let selected_count = match state.selected_indices {
        Some((indices, count))
            if count != 0 && indices.len() >= batch_rows.saturating_mul(count) =>
        {
            count
        }
        Some((indices, count)) => {
            return Err(Error::Shape {
                label: "DeepSeek V4 causal CSA indices",
                expected: format!("at least batch * count values for batch {batch_rows}"),
                actual: format!("indices={} count={count}", indices.len()),
            });
        }
        None => 0,
    };
    let metadata_lengths = [
        state.sliding_tables.len(),
        state.sliding_lengths.len(),
        state.sliding_starts.len(),
        state.current_sequence_starts.len(),
        state.query_offsets.len(),
        state.positions.len(),
        state.compressed_tables.len(),
        state.compressed_lengths.len(),
    ];
    if batch_rows == 0
        || current_rows == 0
        || heads == 0
        || head_dim == 0
        || head_dim > 512
        || sliding_capacity == 0
        || [
            batch_rows,
            current_rows,
            heads,
            head_dim,
            sliding_capacity,
            compression_ratio,
            selected_count,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
        || query.len() < values_len
        || output.len() < values_len
        || sinks.len() != heads
        || metadata_lengths.into_iter().any(|len| len < batch_rows)
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 causal attention",
            expected: format!(
                "query/output>={values_len} current rows>0 head_dim<=512 sinks={heads} metadata>={batch_rows}"
            ),
            actual: format!(
                "batch={batch_rows} current_rows={current_rows} heads={heads} head_dim={head_dim} sliding_capacity={sliding_capacity} compression_ratio={compression_ratio} query={} output={} sinks={} metadata={metadata_lengths:?} selected={selected_count}",
                query.len(),
                output.len(),
                sinks.len(),
            ),
        });
    }
    let selected_ptr = state
        .selected_indices
        .map_or(std::ptr::null(), |(indices, _)| {
            indices.as_const_ptr().cast()
        });
    unsafe {
        check_cuda(
            "infer_deepseek4_causal_attention_f32_on_stream",
            ffi::infer_deepseek4_causal_attention_f32_on_stream(
                query.as_const_ptr().cast(),
                state.sliding_tables.as_const_ptr().cast(),
                state.sliding_lengths.as_const_ptr().cast(),
                state.sliding_starts.as_const_ptr().cast(),
                state.current_kv.as_const_ptr().cast(),
                state.current_sequence_starts.as_const_ptr().cast(),
                state.query_offsets.as_const_ptr().cast(),
                state.positions.as_const_ptr().cast(),
                state.compressed_tables.as_const_ptr().cast(),
                state.compressed_lengths.as_const_ptr().cast(),
                selected_ptr,
                sinks.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                batch_rows as u32,
                current_rows as u32,
                heads as u32,
                head_dim as u32,
                sliding_capacity as u32,
                compression_ratio as u32,
                selected_count as u32,
                1.0 / (head_dim as f32).sqrt(),
                stream.as_raw(),
            ),
        )
    }
}

/// Selects causal compressed entries with DeepSeek's Lightning Indexer score.
#[allow(clippy::too_many_arguments)]
pub fn indexer_topk_f32_batch_into_on_stream(
    query: &DeviceBuffer<f32>,
    head_weights: &DeviceBuffer<f32>,
    compressed_tables: &DeviceBuffer<*const f32>,
    compressed_lengths: &DeviceBuffer<u32>,
    positions: &DeviceBuffer<u32>,
    mut selected_indices: DeviceOutput<'_, i32>,
    batch_rows: usize,
    heads: usize,
    head_dim: usize,
    compression_ratio: usize,
    top_k: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_values = batch_rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    let weight_values = batch_rows.saturating_mul(heads);
    let selected_values = batch_rows.saturating_mul(top_k);
    if batch_rows == 0
        || heads == 0
        || heads > 256
        || head_dim == 0
        || compression_ratio == 0
        || top_k == 0
        || top_k > 4096
        || [batch_rows, heads, head_dim, compression_ratio, top_k]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || query.len() < query_values
        || head_weights.len() < weight_values
        || compressed_tables.len() < batch_rows
        || compressed_lengths.len() < batch_rows
        || positions.len() < batch_rows
        || selected_indices.len() < selected_values
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 indexer top-k",
            expected: format!(
                "query>={query_values} weights>={weight_values} metadata>={batch_rows} selected>={selected_values}"
            ),
            actual: format!(
                "batch={batch_rows} heads={heads} dim={head_dim} ratio={compression_ratio} top_k={top_k} query={} weights={} tables={} lengths={} positions={} selected={}",
                query.len(),
                head_weights.len(),
                compressed_tables.len(),
                compressed_lengths.len(),
                positions.len(),
                selected_indices.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_indexer_topk_f32_on_stream",
            ffi::infer_deepseek4_indexer_topk_f32_on_stream(
                query.as_const_ptr().cast(),
                head_weights.as_const_ptr().cast(),
                compressed_tables.as_const_ptr().cast(),
                compressed_lengths.as_const_ptr().cast(),
                positions.as_const_ptr().cast(),
                selected_indices.as_mut_ptr().cast(),
                batch_rows as u32,
                heads as u32,
                head_dim as u32,
                compression_ratio as u32,
                top_k as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Selects and weights DeepSeek's learned `sqrtsoftplus` routes.
#[allow(clippy::too_many_arguments)]
pub fn router_topk_f32_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut indices: DeviceOutput<'_, u32>,
    mut weights: DeviceOutput<'_, f32>,
    batch_rows: usize,
    experts: usize,
    top_k: usize,
    routed_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_router_buffers(
        logits.len(),
        indices.len(),
        weights.len(),
        batch_rows,
        experts,
        top_k,
        routed_scale,
    )?;
    if bias.len() != experts {
        return Err(Error::Shape {
            label: "DeepSeek V4 router bias",
            expected: format!("{experts} values"),
            actual: format!("{} values", bias.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_router_topk_f32_on_stream",
            ffi::infer_deepseek4_router_topk_f32_on_stream(
                logits.as_const_ptr().cast(),
                bias.as_const_ptr().cast(),
                indices.as_mut_ptr().cast(),
                weights.as_mut_ptr().cast(),
                batch_rows as u32,
                experts as u32,
                top_k as u32,
                routed_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Weights DeepSeek's checkpoint-selected hash routes with `sqrtsoftplus`.
#[allow(clippy::too_many_arguments)]
pub fn router_hash_f32_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    token_to_expert: &DeviceBuffer<i64>,
    token_ids: &DeviceBuffer<u32>,
    mut indices: DeviceOutput<'_, u32>,
    mut weights: DeviceOutput<'_, f32>,
    batch_rows: usize,
    vocab: usize,
    experts: usize,
    top_k: usize,
    routed_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_router_buffers(
        logits.len(),
        indices.len(),
        weights.len(),
        batch_rows,
        experts,
        top_k,
        routed_scale,
    )?;
    let table_len = vocab.saturating_mul(top_k);
    if token_to_expert.len() != table_len || token_ids.len() < batch_rows || top_k > 32 {
        return Err(Error::Shape {
            label: "DeepSeek V4 hash router",
            expected: format!("token table={table_len} token_ids>={batch_rows} top_k<=32"),
            actual: format!(
                "table={} token_ids={} top_k={top_k}",
                token_to_expert.len(),
                token_ids.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_router_hash_f32_on_stream",
            ffi::infer_deepseek4_router_hash_f32_on_stream(
                logits.as_const_ptr().cast(),
                token_to_expert.as_const_ptr().cast(),
                token_ids.as_const_ptr().cast(),
                indices.as_mut_ptr().cast(),
                weights.as_mut_ptr().cast(),
                batch_rows as u32,
                vocab as u32,
                experts as u32,
                top_k as u32,
                routed_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Compresses complete DeepSeek HCA or overlapping CSA token windows.
#[allow(clippy::too_many_arguments)]
pub fn compress_windows_f32_into_on_stream(
    kv: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    position_bias: &DeviceBuffer<f32>,
    prior: Option<(&DeviceBuffer<f32>, &DeviceBuffer<f32>)>,
    mut output: DeviceOutput<'_, f32>,
    windows: usize,
    ratio: usize,
    compressed_width: usize,
    overlapping: bool,
    stream: &CudaStream,
) -> Result<()> {
    let projected_width = if overlapping {
        compressed_width.saturating_mul(2)
    } else {
        compressed_width
    };
    let input_len = windows
        .checked_mul(ratio)
        .and_then(|value| value.checked_mul(projected_width))
        .unwrap_or(usize::MAX);
    let bias_len = ratio.saturating_mul(projected_width);
    let output_len = windows.saturating_mul(compressed_width);
    let prior_len = ratio.saturating_mul(compressed_width);
    if windows == 0
        || ratio == 0
        || ratio > 256
        || compressed_width == 0
        || [windows, ratio, compressed_width]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || kv.len() < input_len
        || gate.len() < input_len
        || position_bias.len() != bias_len
        || output.len() < output_len
        || prior.is_some_and(|(prior_kv, prior_gate)| {
            prior_kv.len() < prior_len || prior_gate.len() < prior_len
        })
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 window compressor",
            expected: format!(
                "kv/gate>={input_len} bias={bias_len} output>={output_len} prior>={prior_len}"
            ),
            actual: format!(
                "windows={windows} ratio={ratio} width={compressed_width} overlap={overlapping} kv={} gate={} bias={} output={} prior={:?}",
                kv.len(),
                gate.len(),
                position_bias.len(),
                output.len(),
                prior.map(|(left, right)| (left.len(), right.len()))
            ),
        });
    }
    let (prior_kv, prior_gate, has_prior) = match prior {
        Some((prior_kv, prior_gate)) => (
            prior_kv.as_const_ptr().cast(),
            prior_gate.as_const_ptr().cast(),
            true,
        ),
        None => (std::ptr::null(), std::ptr::null(), false),
    };
    unsafe {
        check_cuda(
            "infer_deepseek4_compress_windows_f32_on_stream",
            ffi::infer_deepseek4_compress_windows_f32_on_stream(
                kv.as_const_ptr().cast(),
                gate.as_const_ptr().cast(),
                position_bias.as_const_ptr().cast(),
                prior_kv,
                prior_gate,
                output.as_mut_ptr().cast(),
                windows as u32,
                ratio as u32,
                compressed_width as u32,
                overlapping,
                has_prior,
                stream.as_raw(),
            ),
        )
    }
}

/// Retains the final CSA Ca half-window with its position-biased gate.
#[allow(clippy::too_many_arguments)]
pub fn store_compression_overlap_f32_into_on_stream(
    kv: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    position_bias: &DeviceBuffer<f32>,
    mut overlap_kv: DeviceOutput<'_, f32>,
    mut overlap_gate: DeviceOutput<'_, f32>,
    windows: usize,
    ratio: usize,
    compressed_width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let projected_width = compressed_width.saturating_mul(2);
    let input_values = windows
        .checked_mul(ratio)
        .and_then(|value| value.checked_mul(projected_width))
        .unwrap_or(usize::MAX);
    let overlap_values = ratio.saturating_mul(compressed_width);
    let bias_values = ratio.saturating_mul(projected_width);
    if windows == 0
        || ratio == 0
        || compressed_width == 0
        || [windows - 1, ratio, compressed_width]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || kv.len() < input_values
        || gate.len() < input_values
        || position_bias.len() != bias_values
        || overlap_kv.len() < overlap_values
        || overlap_gate.len() < overlap_values
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 compression overlap",
            expected: format!(
                "kv/gate>={input_values} bias={bias_values} overlap>={overlap_values}"
            ),
            actual: format!(
                "windows={windows} ratio={ratio} width={compressed_width} kv={} gate={} bias={} overlap_kv={} overlap_gate={}",
                kv.len(),
                gate.len(),
                position_bias.len(),
                overlap_kv.len(),
                overlap_gate.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_store_compression_overlap_f32_on_stream",
            ffi::infer_deepseek4_store_compression_overlap_f32_on_stream(
                kv.as_const_ptr().cast(),
                gate.as_const_ptr().cast(),
                position_bias.as_const_ptr().cast(),
                overlap_kv.as_mut_ptr().cast(),
                overlap_gate.as_mut_ptr().cast(),
                (windows - 1) as u32,
                ratio as u32,
                compressed_width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Fills arithmetic absolute positions without a host transfer.
pub fn arithmetic_positions_u32_into_on_stream(
    mut positions: DeviceOutput<'_, u32>,
    len: usize,
    start: usize,
    stride: usize,
    stream: &CudaStream,
) -> Result<()> {
    let last = start.saturating_add(len.saturating_sub(1).saturating_mul(stride));
    if len == 0
        || stride == 0
        || [len, start, stride, last]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || positions.len() < len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 arithmetic positions",
            expected: format!("output>={len} and u32 arithmetic without overflow"),
            actual: format!(
                "output={} len={len} start={start} stride={stride} last={last}",
                positions.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_arithmetic_positions_u32_on_stream",
            ffi::infer_deepseek4_arithmetic_positions_u32_on_stream(
                positions.as_mut_ptr().cast(),
                len as u32,
                start as u32,
                stride as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Duplicates each hidden row into DeepSeek's four initial mHC streams.
pub fn repeat_hyper_streams_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    hidden: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.saturating_mul(hidden);
    let output_len = input_len.saturating_mul(HYPER_STREAMS);
    if rows == 0
        || hidden == 0
        || rows > u32::MAX as usize
        || hidden > u32::MAX as usize
        || input.len() < input_len
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 initial hyper streams",
            expected: format!("input>={input_len} output>={output_len}"),
            actual: format!(
                "rows={rows} hidden={hidden} input={} output={}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_repeat_hyper_streams_f32_on_stream",
            ffi::infer_deepseek4_repeat_hyper_streams_f32_on_stream(
                input.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                hidden as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies an unclamped SwiGLU activation to separately projected gate and up tensors.
pub fn swiglu_pair_f32_batch_into_on_stream(
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = rows.saturating_mul(width);
    if rows == 0
        || width == 0
        || rows > u32::MAX as usize
        || width > u32::MAX as usize
        || gate.len() < values
        || up.len() < values
        || output.len() < values
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 SwiGLU",
            expected: format!("gate/up/output>={values}"),
            actual: format!(
                "rows={rows} width={width} gate={} up={} output={}",
                gate.len(),
                up.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_swiglu_pair_f32_on_stream",
            ffi::infer_deepseek4_swiglu_pair_f32_on_stream(
                gate.as_const_ptr().cast(),
                up.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies DeepSeek's separately projected, clamped SwiGLU activation.
pub fn swiglu_pair_clamped_f32_batch_into_on_stream(
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    width: usize,
    limit: f32,
    stream: &CudaStream,
) -> Result<()> {
    let values = rows.saturating_mul(width);
    if rows == 0
        || width == 0
        || rows > u32::MAX as usize
        || width > u32::MAX as usize
        || !limit.is_finite()
        || limit <= 0.0
        || gate.len() < values
        || up.len() < values
        || output.len() < values
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 clamped SwiGLU",
            expected: format!("gate/up/output>={values} finite positive limit"),
            actual: format!(
                "rows={rows} width={width} limit={limit} gate={} up={} output={}",
                gate.len(),
                up.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_swiglu_pair_clamped_f32_on_stream",
            ffi::infer_deepseek4_swiglu_pair_clamped_f32_on_stream(
                gate.as_const_ptr().cast(),
                up.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                width as u32,
                limit,
                stream.as_raw(),
            ),
        )
    }
}

/// Reduces contiguous route-major outputs with per-route router weights.
pub fn routed_accumulate_f32_batch_into_on_stream(
    route_output: &DeviceBuffer<f32>,
    route_weights: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    routes_per_row: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let routes = rows.saturating_mul(routes_per_row);
    let route_values = routes.saturating_mul(width);
    let output_values = rows.saturating_mul(width);
    if rows == 0
        || routes_per_row == 0
        || width == 0
        || [rows, routes_per_row, width]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || route_output.len() < route_values
        || route_weights.len() < routes
        || output.len() < output_values
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 routed accumulation",
            expected: format!(
                "route_output>={route_values} weights>={routes} output>={output_values}"
            ),
            actual: format!(
                "rows={rows} routes/row={routes_per_row} width={width} route_output={} weights={} output={}",
                route_output.len(),
                route_weights.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_routed_accumulate_f32_on_stream",
            ffi::infer_deepseek4_routed_accumulate_f32_on_stream(
                route_output.as_const_ptr().cast(),
                route_weights.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                routes_per_row as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Gathers token rows for one contiguous expert-major route segment.
pub fn gather_sorted_route_rows_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    routes: &MoeSortedRoutes,
    mut output: DeviceOutput<'_, f32>,
    route_offset: usize,
    route_count: usize,
    input_rows: usize,
    routes_per_row: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let route_end = route_offset.saturating_add(route_count);
    let input_values = input_rows.saturating_mul(width);
    let output_values = route_count.saturating_mul(width);
    if route_count == 0
        || input_rows == 0
        || routes_per_row == 0
        || width == 0
        || route_end > routes.active_routes()
        || input.len() < input_values
        || output.len() < output_values
        || [route_offset, route_count, input_rows, routes_per_row, width]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 sorted route gather",
            expected: format!(
                "non-empty route range ending at or before {}, input>={input_values}, output>={output_values}",
                routes.active_routes()
            ),
            actual: format!(
                "offset={route_offset} routes={route_count} input_rows={input_rows} routes/row={routes_per_row} width={width} input={} output={}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_gather_sorted_route_rows_f32_on_stream",
            ffi::infer_deepseek4_gather_sorted_route_rows_f32_on_stream(
                input.as_const_ptr().cast(),
                routes.sorted_routes().as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                route_offset as u32,
                route_count as u32,
                routes_per_row as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Reduces expert-major route outputs back into their original token rows.
pub fn routed_accumulate_sorted_f32_batch_into_on_stream(
    sorted_route_output: &DeviceBuffer<f32>,
    routes: &MoeSortedRoutes,
    route_weights: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    routes_per_row: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let route_count = rows.saturating_mul(routes_per_row);
    let route_values = route_count.saturating_mul(width);
    let output_values = rows.saturating_mul(width);
    if rows == 0
        || routes_per_row == 0
        || width == 0
        || route_count != routes.active_routes()
        || sorted_route_output.len() < route_values
        || route_weights.len() < route_count
        || output.len() < output_values
        || [rows, routes_per_row, width]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 sorted routed accumulation",
            expected: format!(
                "active routes={route_count}, route_output>={route_values}, weights>={route_count}, output>={output_values}"
            ),
            actual: format!(
                "active_routes={} rows={rows} routes/row={routes_per_row} width={width} route_output={} weights={} output={}",
                routes.active_routes(),
                sorted_route_output.len(),
                route_weights.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_deepseek4_routed_accumulate_sorted_f32_on_stream",
            ffi::infer_deepseek4_routed_accumulate_sorted_f32_on_stream(
                sorted_route_output.as_const_ptr().cast(),
                routes.route_to_sorted().as_const_ptr().cast(),
                route_weights.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                routes_per_row as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_router_buffers(
    logits_len: usize,
    indices_len: usize,
    weights_len: usize,
    batch_rows: usize,
    experts: usize,
    top_k: usize,
    routed_scale: f32,
) -> Result<()> {
    let logits_expected = batch_rows.saturating_mul(experts);
    let routes_expected = batch_rows.saturating_mul(top_k);
    if batch_rows == 0
        || experts == 0
        || top_k == 0
        || top_k > experts
        || [batch_rows, experts, top_k]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || !routed_scale.is_finite()
        || routed_scale <= 0.0
        || logits_len < logits_expected
        || indices_len < routes_expected
        || weights_len < routes_expected
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 router",
            expected: format!(
                "logits>={logits_expected} indices/weights>={routes_expected} 0<top_k<=experts"
            ),
            actual: format!(
                "batch={batch_rows} experts={experts} top_k={top_k} scale={routed_scale} logits={logits_len} indices={indices_len} weights={weights_len}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Deepseek4AttentionBatch, Deepseek4CausalAttentionBatch, HYPER_MIX, HYPER_STREAMS,
        SCALE_BLOCK, arithmetic_positions_u32_into_on_stream, attention_f32_batch_into_on_stream,
        block_fp8_grouped_linear_f32_batch_into_on_stream,
        block_fp8_linear_f32_batch_into_on_stream, causal_attention_f32_batch_into_on_stream,
        compress_windows_f32_into_on_stream, hyper_apply_f32_batch_into_on_stream,
        hyper_head_f32_batch_into_on_stream, hyper_prepare_f32_batch_into_on_stream,
        indexer_topk_f32_batch_into_on_stream, repeat_hyper_streams_f32_into_on_stream,
        rope_interleaved_trailing_f32_indexed_in_place_on_stream,
        routed_accumulate_f32_batch_into_on_stream,
        routed_accumulate_sorted_f32_batch_into_on_stream, router_hash_f32_batch_into_on_stream,
        router_topk_f32_batch_into_on_stream, store_compression_overlap_f32_into_on_stream,
        swiglu_pair_clamped_f32_batch_into_on_stream, swiglu_pair_f32_batch_into_on_stream,
    };
    use crate::{
        CudaStream, DeviceBuffer, MoeSortedRoutes, format,
        gather_sorted_route_rows_f32_into_on_stream,
    };

    #[test]
    fn block_scaled_fp8_linear_matches_cpu_reference() {
        const BATCH: usize = 2;
        const ROWS: usize = 256;
        const COLS: usize = 256;
        let input = (0..BATCH * COLS)
            .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let weight = (0..ROWS * COLS)
            .map(|index| format::cuda_e4m3_code(((index % 11) as f32 - 5.0) / 4.0))
            .collect::<Vec<_>>();
        let scales = [127u8, 126, 128, 125];
        let mut expected = Vec::with_capacity(BATCH * ROWS);
        for batch in 0..BATCH {
            let input_row = &input[batch * COLS..(batch + 1) * COLS];
            for row in 0..ROWS {
                expected.push(
                    (0..COLS)
                        .map(|col| {
                            let scale = 2.0f32.powi(
                                scales
                                    [(row / SCALE_BLOCK) * (COLS / SCALE_BLOCK) + col / SCALE_BLOCK]
                                    as i32
                                    - 127,
                            );
                            input_row[col] * format::e4m3_value(weight[row * COLS + col]) * scale
                        })
                        .sum::<f32>(),
                );
            }
        }
        let input = DeviceBuffer::from_host(&input).expect("input");
        let weight = DeviceBuffer::from_host(&weight).expect("weight");
        let scales = DeviceBuffer::from_host(&scales).expect("scales");
        let mut output = DeviceBuffer::zeroed(BATCH * ROWS).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        block_fp8_linear_f32_batch_into_on_stream(
            &input,
            &weight,
            &scales,
            output.output(),
            BATCH,
            ROWS,
            COLS,
            &stream,
        )
        .expect("linear");
        let actual = output.copy_to_host(&stream).expect("read output");
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let allowed = 2.0e-4 + 2.0e-4 * expected.abs();
            assert!(
                (actual - expected).abs() <= allowed,
                "mismatch at {index}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn grouped_block_scaled_fp8_linear_uses_matching_input_group() {
        const BATCH: usize = 2;
        const GROUPS: usize = 2;
        const ROWS: usize = 128;
        const COLS: usize = 128;
        let input = (0..BATCH * GROUPS * COLS)
            .map(|index| ((index / COLS) as f32 + 1.0) / 4.0)
            .collect::<Vec<_>>();
        let weight = vec![crate::format::cuda_e4m3_code(1.0); GROUPS * ROWS * COLS];
        let scales = vec![127u8; GROUPS];
        let mut output = DeviceBuffer::zeroed(BATCH * GROUPS * ROWS).expect("output");
        let input = DeviceBuffer::from_host(&input).expect("input");
        let weight = DeviceBuffer::from_host(&weight).expect("weight");
        let scales = DeviceBuffer::from_host(&scales).expect("scales");
        let stream = CudaStream::new_non_blocking().expect("stream");
        block_fp8_grouped_linear_f32_batch_into_on_stream(
            &input,
            &weight,
            &scales,
            output.output(),
            BATCH,
            GROUPS,
            ROWS,
            COLS,
            &stream,
        )
        .expect("grouped linear");
        let actual = output.copy_to_host(&stream).expect("read grouped");
        for batch in 0..BATCH {
            for group in 0..GROUPS {
                let expected = ((batch * GROUPS + group) as f32 + 1.0) / 4.0 * COLS as f32;
                for row in 0..ROWS {
                    let index = (batch * GROUPS + group) * ROWS + row;
                    assert!((actual[index] - expected).abs() < 1.0e-4);
                }
            }
        }
    }

    #[test]
    fn hyper_connection_matches_cpu_reference() {
        const BATCH: usize = 2;
        const HIDDEN: usize = 128;
        const RMS_EPS: f32 = 1.0e-6;
        const HC_EPS: f32 = 1.0e-6;
        const SINKHORN: usize = 20;
        let streams = (0..BATCH * HYPER_STREAMS * HIDDEN)
            .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>();
        let function = (0..HYPER_MIX * HYPER_STREAMS * HIDDEN)
            .map(|index| ((index % 13) as f32 - 6.0) / 512.0)
            .collect::<Vec<_>>();
        let base = (0..HYPER_MIX)
            .map(|index| ((index % 7) as f32 - 3.0) / 16.0)
            .collect::<Vec<_>>();
        let scale = [0.5f32, 0.75, 0.25];
        let sublayer = (0..BATCH * HIDDEN)
            .map(|index| ((index % 19) as f32 - 9.0) / 8.0)
            .collect::<Vec<_>>();
        let (expected_post, expected_comb, expected_collapsed) = cpu_hyper_prepare(
            &streams, &function, &base, &scale, BATCH, HIDDEN, RMS_EPS, HC_EPS, SINKHORN,
        );
        let expected_output = cpu_hyper_apply(
            &streams,
            &sublayer,
            &expected_post,
            &expected_comb,
            BATCH,
            HIDDEN,
        );

        let streams_device = DeviceBuffer::from_host(&streams).expect("streams");
        let function_device = DeviceBuffer::from_host(&function).expect("function");
        let base_device = DeviceBuffer::from_host(&base).expect("base");
        let scale_device = DeviceBuffer::from_host(&scale).expect("scale");
        let sublayer_device = DeviceBuffer::from_host(&sublayer).expect("sublayer");
        let mut post = DeviceBuffer::zeroed(BATCH * HYPER_STREAMS).expect("post");
        let mut combination =
            DeviceBuffer::zeroed(BATCH * HYPER_STREAMS * HYPER_STREAMS).expect("comb");
        let mut collapsed = DeviceBuffer::zeroed(BATCH * HIDDEN).expect("collapsed");
        let mut output = DeviceBuffer::zeroed(BATCH * HYPER_STREAMS * HIDDEN).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        hyper_prepare_f32_batch_into_on_stream(
            &streams_device,
            &function_device,
            &base_device,
            &scale_device,
            post.output(),
            combination.output(),
            collapsed.output(),
            BATCH,
            HIDDEN,
            RMS_EPS,
            HC_EPS,
            SINKHORN,
            &stream,
        )
        .expect("prepare");
        hyper_apply_f32_batch_into_on_stream(
            &streams_device,
            &sublayer_device,
            &post,
            &combination,
            output.output(),
            BATCH,
            HIDDEN,
            &stream,
        )
        .expect("apply");

        assert_close(
            &post.copy_to_host(&stream).expect("read post"),
            &expected_post,
            5.0e-5,
        );
        assert_close(
            &combination.copy_to_host(&stream).expect("read comb"),
            &expected_comb,
            5.0e-5,
        );
        assert_close(
            &collapsed.copy_to_host(&stream).expect("read collapsed"),
            &expected_collapsed,
            5.0e-5,
        );
        assert_close(
            &output.copy_to_host(&stream).expect("read output"),
            &expected_output,
            1.0e-4,
        );
    }

    #[test]
    fn hyper_head_matches_cpu_reference() {
        const BATCH: usize = 2;
        const HIDDEN: usize = 128;
        let streams = test_rows(BATCH * HYPER_STREAMS, HIDDEN, -0.2);
        let function = test_rows(HYPER_STREAMS, HYPER_STREAMS * HIDDEN, 0.01)
            .into_iter()
            .map(|value| value / 64.0)
            .collect::<Vec<_>>();
        let base = [0.1f32, -0.2, 0.3, -0.4];
        let scale = [0.75f32];
        let expected = cpu_hyper_head(
            &streams, &function, &base, scale[0], BATCH, HIDDEN, 1.0e-6, 1.0e-6,
        );
        let streams = DeviceBuffer::from_host(&streams).expect("streams");
        let function = DeviceBuffer::from_host(&function).expect("function");
        let base = DeviceBuffer::from_host(&base).expect("base");
        let scale = DeviceBuffer::from_host(&scale).expect("scale");
        let mut output = DeviceBuffer::zeroed(BATCH * HIDDEN).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        hyper_head_f32_batch_into_on_stream(
            &streams,
            &function,
            &base,
            &scale,
            output.output(),
            BATCH,
            HIDDEN,
            1.0e-6,
            1.0e-6,
            &stream,
        )
        .expect("hyper head");
        assert_close(
            &output.copy_to_host(&stream).expect("read hyper head"),
            &expected,
            4.0e-5,
        );
    }

    fn cpu_hyper_head(
        streams: &[f32],
        function: &[f32],
        base: &[f32],
        scale: f32,
        batch: usize,
        hidden: usize,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Vec<f32> {
        let flat = HYPER_STREAMS * hidden;
        let mut output = vec![0.0; batch * hidden];
        for batch_index in 0..batch {
            let row = &streams[batch_index * flat..(batch_index + 1) * flat];
            let inverse_rms = (row.iter().map(|value| value * value).sum::<f32>() / flat as f32
                + rms_eps)
                .sqrt()
                .recip();
            let mut weights = [0.0; HYPER_STREAMS];
            for stream in 0..HYPER_STREAMS {
                let mixed = row
                    .iter()
                    .zip(&function[stream * flat..(stream + 1) * flat])
                    .map(|(value, weight)| value * inverse_rms * weight)
                    .sum::<f32>();
                weights[stream] = 1.0 / (1.0 + (-(mixed * scale + base[stream])).exp()) + hc_eps;
            }
            for feature in 0..hidden {
                output[batch_index * hidden + feature] = (0..HYPER_STREAMS)
                    .map(|stream| weights[stream] * row[stream * hidden + feature])
                    .sum();
            }
        }
        output
    }

    #[test]
    fn trailing_interleaved_rope_matches_cpu_and_conjugates() {
        const BATCH: usize = 2;
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const ROPE_DIM: usize = 4;
        let original = (0..BATCH * HEADS * HEAD_DIM)
            .map(|index| index as f32 / 13.0 - 1.0)
            .collect::<Vec<_>>();
        let positions = [0u32, 3];
        let inv_freq = [1.0f32, 0.1];
        let mut expected = original.clone();
        cpu_rope(
            &mut expected,
            &inv_freq,
            &positions,
            BATCH,
            HEADS,
            HEAD_DIM,
            ROPE_DIM,
            1.0,
        );
        let mut values = DeviceBuffer::from_host(&original).expect("values");
        let positions = DeviceBuffer::from_host(&positions).expect("positions");
        let inv_freq = DeviceBuffer::from_host(&inv_freq).expect("inv freq");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            values.inout(),
            &inv_freq,
            &positions,
            BATCH,
            HEADS,
            HEAD_DIM,
            ROPE_DIM,
            1.0,
            &stream,
        )
        .expect("forward rope");
        assert_close(
            &values.copy_to_host(&stream).expect("read forward"),
            &expected,
            2.0e-6,
        );
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            values.inout(),
            &inv_freq,
            &positions,
            BATCH,
            HEADS,
            HEAD_DIM,
            ROPE_DIM,
            -1.0,
            &stream,
        )
        .expect("conjugate rope");
        assert_close(
            &values.copy_to_host(&stream).expect("read conjugate"),
            &original,
            2.0e-6,
        );
    }

    #[test]
    fn initial_hyper_streams_repeat_each_embedding_row() {
        const ROWS: usize = 3;
        const HIDDEN: usize = 5;
        let input = (0..ROWS * HIDDEN)
            .map(|index| index as f32 - 4.0)
            .collect::<Vec<_>>();
        let expected = input
            .chunks_exact(HIDDEN)
            .flat_map(|row| std::iter::repeat_n(row, HYPER_STREAMS).flatten().copied())
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input).expect("input");
        let mut output = DeviceBuffer::zeroed(ROWS * HYPER_STREAMS * HIDDEN).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        repeat_hyper_streams_f32_into_on_stream(&input, output.output(), ROWS, HIDDEN, &stream)
            .expect("repeat streams");
        assert_eq!(
            output
                .copy_to_host(&stream)
                .expect("read output")
                .as_slice(),
            expected
        );
    }

    #[test]
    fn attention_combines_ring_selected_compressed_and_sink() {
        const BATCH: usize = 2;
        const HEADS: usize = 2;
        const DIM: usize = 8;
        const CAPACITY: usize = 4;
        let query = (0..BATCH * HEADS * DIM)
            .map(|index| (index as f32 % 9.0 - 4.0) / 5.0)
            .collect::<Vec<_>>();
        let sliding_a = test_rows(CAPACITY, DIM, 0.1);
        let sliding_b = test_rows(CAPACITY, DIM, -0.2);
        let compressed_a = test_rows(3, DIM, 0.4);
        let compressed_b = test_rows(2, DIM, -0.6);
        let lengths = [3u32, 2];
        let starts = [2u32, 1];
        let compressed_lengths = [3u32, 2];
        let selected = [2i32, 0, 1, -1];
        let sinks = [0.25f32, -0.5];
        let expected = cpu_attention(
            &query,
            [&sliding_a, &sliding_b],
            &lengths,
            &starts,
            [&compressed_a, &compressed_b],
            &compressed_lengths,
            Some(&selected),
            &sinks,
            BATCH,
            HEADS,
            DIM,
            CAPACITY,
        );

        let query = DeviceBuffer::from_host(&query).expect("query");
        let sliding_a = DeviceBuffer::from_host(&sliding_a).expect("sliding a");
        let sliding_b = DeviceBuffer::from_host(&sliding_b).expect("sliding b");
        let compressed_a = DeviceBuffer::from_host(&compressed_a).expect("compressed a");
        let compressed_b = DeviceBuffer::from_host(&compressed_b).expect("compressed b");
        let sliding_ptrs = DeviceBuffer::from_host(&[
            sliding_a.input().as_const_ptr().cast::<f32>(),
            sliding_b.input().as_const_ptr().cast::<f32>(),
        ])
        .expect("sliding pointers");
        let compressed_ptrs = DeviceBuffer::from_host(&[
            compressed_a.input().as_const_ptr().cast::<f32>(),
            compressed_b.input().as_const_ptr().cast::<f32>(),
        ])
        .expect("compressed pointers");
        let lengths = DeviceBuffer::from_host(&lengths).expect("lengths");
        let starts = DeviceBuffer::from_host(&starts).expect("starts");
        let compressed_lengths =
            DeviceBuffer::from_host(&compressed_lengths).expect("compressed lengths");
        let selected = DeviceBuffer::from_host(&selected).expect("selected");
        let sinks = DeviceBuffer::from_host(&sinks).expect("sinks");
        let mut output = DeviceBuffer::zeroed(BATCH * HEADS * DIM).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        attention_f32_batch_into_on_stream(
            &query,
            Deepseek4AttentionBatch {
                sliding_tables: &sliding_ptrs,
                sliding_lengths: &lengths,
                sliding_starts: &starts,
                compressed_tables: &compressed_ptrs,
                compressed_lengths: &compressed_lengths,
                selected_indices: Some(&selected),
            },
            &sinks,
            output.output(),
            BATCH,
            HEADS,
            DIM,
            CAPACITY,
            &stream,
        )
        .expect("attention");
        assert_close(
            &output.copy_to_host(&stream).expect("read attention"),
            &expected,
            3.0e-5,
        );
    }

    #[test]
    fn causal_attention_masks_prior_current_and_compressed_entries() {
        const ROWS: usize = 3;
        const HEADS: usize = 1;
        const DIM: usize = 4;
        const PRIOR_CAPACITY: usize = 3;
        const RATIO: usize = 4;
        let query = test_rows(ROWS, DIM, -0.15);
        let prior_chronological = test_rows(PRIOR_CAPACITY, DIM, 0.1);
        let mut prior_physical = vec![0.0; PRIOR_CAPACITY * DIM];
        for logical in 0..PRIOR_CAPACITY {
            let slot = (1 + logical) % PRIOR_CAPACITY;
            prior_physical[slot * DIM..(slot + 1) * DIM]
                .copy_from_slice(&prior_chronological[logical * DIM..(logical + 1) * DIM]);
        }
        let current = test_rows(ROWS, DIM, 0.55);
        let compressed = test_rows(2, DIM, -0.7);
        let sinks = [0.2f32];
        let positions = [5u32, 6, 7];
        let mut expected = Vec::with_capacity(ROWS * DIM);
        for row in 0..ROWS {
            let mut sources = prior_chronological.clone();
            sources.extend_from_slice(&current[..(row + 1) * DIM]);
            let visible = sources.len() / DIM;
            let keep = (PRIOR_CAPACITY + 1).min(visible);
            let mut rows = sources[(visible - keep) * DIM..].to_vec();
            let compressed_visible = (((positions[row] as usize) + 1) / RATIO).min(2);
            rows.extend_from_slice(&compressed[..compressed_visible * DIM]);
            let q = &query[row * DIM..(row + 1) * DIM];
            let logits = rows
                .chunks_exact(DIM)
                .map(|kv| {
                    q.iter()
                        .zip(kv)
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                        / (DIM as f32).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = logits.iter().copied().fold(sinks[0], f32::max);
            let denominator = (sinks[0] - maximum).exp()
                + logits
                    .iter()
                    .map(|logit| (*logit - maximum).exp())
                    .sum::<f32>();
            for feature in 0..DIM {
                expected.push(
                    rows.chunks_exact(DIM)
                        .zip(&logits)
                        .map(|(kv, logit)| kv[feature] * (*logit - maximum).exp() / denominator)
                        .sum(),
                );
            }
        }

        let query = DeviceBuffer::from_host(&query).expect("query");
        let prior = DeviceBuffer::from_host(&prior_physical).expect("prior");
        let current = DeviceBuffer::from_host(&current).expect("current");
        let compressed = DeviceBuffer::from_host(&compressed).expect("compressed");
        let prior_pointer = prior.input().as_const_ptr().cast::<f32>();
        let compressed_pointer = compressed.input().as_const_ptr().cast::<f32>();
        let prior_tables = DeviceBuffer::from_host(&[prior_pointer; ROWS]).expect("prior pointers");
        let compressed_tables =
            DeviceBuffer::from_host(&[compressed_pointer; ROWS]).expect("compressed pointers");
        let prior_lengths =
            DeviceBuffer::from_host(&[PRIOR_CAPACITY as u32; ROWS]).expect("prior lengths");
        let prior_starts = DeviceBuffer::from_host(&[1u32; ROWS]).expect("prior starts");
        let current_starts = DeviceBuffer::from_host(&[0u32; ROWS]).expect("current starts");
        let query_offsets = DeviceBuffer::from_host(&[0u32, 1, 2]).expect("query offsets");
        let positions = DeviceBuffer::from_host(&positions).expect("positions");
        let compressed_lengths =
            DeviceBuffer::from_host(&[2u32; ROWS]).expect("compressed lengths");
        let sinks = DeviceBuffer::from_host(&sinks).expect("sinks");
        let mut output = DeviceBuffer::zeroed(ROWS * HEADS * DIM).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        causal_attention_f32_batch_into_on_stream(
            &query,
            Deepseek4CausalAttentionBatch {
                sliding_tables: &prior_tables,
                sliding_lengths: &prior_lengths,
                sliding_starts: &prior_starts,
                current_kv: &current,
                current_sequence_starts: &current_starts,
                query_offsets: &query_offsets,
                positions: &positions,
                compressed_tables: &compressed_tables,
                compressed_lengths: &compressed_lengths,
                selected_indices: None,
            },
            &sinks,
            output.output(),
            ROWS,
            HEADS,
            DIM,
            PRIOR_CAPACITY,
            RATIO,
            &stream,
        )
        .expect("causal attention");
        assert_close(
            &output.copy_to_host(&stream).expect("causal output"),
            &expected,
            3.0e-5,
        );
    }

    #[test]
    fn indexer_topk_matches_causal_cpu_scores() {
        const ROWS: usize = 2;
        const HEADS: usize = 3;
        const DIM: usize = 4;
        const RATIO: usize = 2;
        const TOP_K: usize = 3;
        let query = test_rows(ROWS * HEADS, DIM, -0.4);
        let head_weights = test_rows(ROWS, HEADS, 0.15);
        let compressed_a = test_rows(4, DIM, -0.25);
        let compressed_b = test_rows(3, DIM, 0.6);
        let compressed_lengths = [4u32, 3];
        let positions = [4u32, 2];
        let mut expected = Vec::with_capacity(ROWS * TOP_K);
        for row in 0..ROWS {
            let compressed = if row == 0 {
                &compressed_a
            } else {
                &compressed_b
            };
            let causal =
                (compressed_lengths[row] as usize).min((positions[row] as usize + 1) / RATIO);
            let mut scores = (0..causal)
                .map(|entry| {
                    let score = (0..HEADS)
                        .map(|head| {
                            let q =
                                &query[(row * HEADS + head) * DIM..(row * HEADS + head + 1) * DIM];
                            let key = &compressed[entry * DIM..(entry + 1) * DIM];
                            let dot = q
                                .iter()
                                .zip(key)
                                .map(|(left, right)| left * right)
                                .sum::<f32>()
                                / (DIM as f32).sqrt();
                            head_weights[row * HEADS + head] * dot.max(0.0) / (HEADS as f32).sqrt()
                        })
                        .sum::<f32>();
                    (score, entry as i32)
                })
                .collect::<Vec<_>>();
            scores.sort_by(|left, right| right.0.total_cmp(&left.0));
            expected.extend(scores.iter().take(TOP_K).map(|(_, index)| *index));
            expected.resize((row + 1) * TOP_K, -1);
        }

        let query = DeviceBuffer::from_host(&query).expect("query");
        let head_weights = DeviceBuffer::from_host(&head_weights).expect("head weights");
        let compressed_a = DeviceBuffer::from_host(&compressed_a).expect("compressed a");
        let compressed_b = DeviceBuffer::from_host(&compressed_b).expect("compressed b");
        let compressed_tables = DeviceBuffer::from_host(&[
            compressed_a.input().as_const_ptr().cast::<f32>(),
            compressed_b.input().as_const_ptr().cast::<f32>(),
        ])
        .expect("compressed tables");
        let compressed_lengths =
            DeviceBuffer::from_host(&compressed_lengths).expect("compressed lengths");
        let positions = DeviceBuffer::from_host(&positions).expect("positions");
        let mut selected = DeviceBuffer::zeroed(ROWS * TOP_K).expect("selected");
        let stream = CudaStream::new_non_blocking().expect("stream");
        indexer_topk_f32_batch_into_on_stream(
            &query,
            &head_weights,
            &compressed_tables,
            &compressed_lengths,
            &positions,
            selected.output(),
            ROWS,
            HEADS,
            DIM,
            RATIO,
            TOP_K,
            &stream,
        )
        .expect("indexer");
        assert_eq!(
            selected
                .copy_to_host(&stream)
                .expect("selected host")
                .as_slice(),
            expected
        );
    }

    #[test]
    fn learned_and_hash_routers_match_sqrtsoftplus_reference() {
        const BATCH: usize = 2;
        const EXPERTS: usize = 8;
        const TOP_K: usize = 3;
        const SCALE: f32 = 1.5;
        let logits = (0..BATCH * EXPERTS)
            .map(|index| index as f32 / 3.0 - 2.0)
            .collect::<Vec<_>>();
        let bias = [0.0f32, 0.3, -0.2, 0.1, 0.0, -0.4, 0.2, 0.0];
        let expected_topk = cpu_topk_router(&logits, &bias, BATCH, EXPERTS, TOP_K, SCALE);
        let logits_device = DeviceBuffer::from_host(&logits).expect("logits");
        let bias_device = DeviceBuffer::from_host(&bias).expect("bias");
        let mut indices = DeviceBuffer::zeroed(BATCH * TOP_K).expect("indices");
        let mut weights = DeviceBuffer::zeroed(BATCH * TOP_K).expect("weights");
        let stream = CudaStream::new_non_blocking().expect("stream");
        router_topk_f32_batch_into_on_stream(
            &logits_device,
            &bias_device,
            indices.output(),
            weights.output(),
            BATCH,
            EXPERTS,
            TOP_K,
            SCALE,
            &stream,
        )
        .expect("topk");
        assert_eq!(
            indices
                .copy_to_host(&stream)
                .expect("read indices")
                .as_ref(),
            expected_topk.0
        );
        assert_close(
            &weights.copy_to_host(&stream).expect("read weights"),
            &expected_topk.1,
            2.0e-6,
        );

        let token_ids = [1u32, 0];
        let token_table = [0i64, 2, 7, 6, 4, 1];
        let expected_hash =
            cpu_hash_router(&logits, &token_table, &token_ids, EXPERTS, TOP_K, SCALE);
        let token_ids = DeviceBuffer::from_host(&token_ids).expect("token ids");
        let token_table = DeviceBuffer::from_host(&token_table).expect("table");
        router_hash_f32_batch_into_on_stream(
            &logits_device,
            &token_table,
            &token_ids,
            indices.output(),
            weights.output(),
            BATCH,
            2,
            EXPERTS,
            TOP_K,
            SCALE,
            &stream,
        )
        .expect("hash");
        assert_eq!(
            indices
                .copy_to_host(&stream)
                .expect("read hash indices")
                .as_ref(),
            expected_hash.0
        );
        assert_close(
            &weights.copy_to_host(&stream).expect("read hash weights"),
            &expected_hash.1,
            2.0e-6,
        );
    }

    #[test]
    fn compressor_matches_hca_and_overlapping_csa_reference() {
        for overlapping in [false, true] {
            const WINDOWS: usize = 2;
            const RATIO: usize = 4;
            const WIDTH: usize = 8;
            let projected = if overlapping { 2 * WIDTH } else { WIDTH };
            let kv = test_rows(WINDOWS * RATIO, projected, -0.25);
            let gate = test_rows(WINDOWS * RATIO, projected, 0.1);
            let bias = test_rows(RATIO, projected, -0.05);
            let prior_kv = test_rows(RATIO, WIDTH, 0.7);
            let prior_gate = test_rows(RATIO, WIDTH, -0.3);
            let expected = cpu_compress(
                &kv,
                &gate,
                &bias,
                Some((&prior_kv, &prior_gate)),
                WINDOWS,
                RATIO,
                WIDTH,
                overlapping,
            );
            let kv = DeviceBuffer::from_host(&kv).expect("kv");
            let gate = DeviceBuffer::from_host(&gate).expect("gate");
            let bias = DeviceBuffer::from_host(&bias).expect("bias");
            let prior_kv = DeviceBuffer::from_host(&prior_kv).expect("prior kv");
            let prior_gate = DeviceBuffer::from_host(&prior_gate).expect("prior gate");
            let mut output = DeviceBuffer::zeroed(WINDOWS * WIDTH).expect("output");
            let stream = CudaStream::new_non_blocking().expect("stream");
            compress_windows_f32_into_on_stream(
                &kv,
                &gate,
                &bias,
                Some((&prior_kv, &prior_gate)),
                output.output(),
                WINDOWS,
                RATIO,
                WIDTH,
                overlapping,
                &stream,
            )
            .expect("compress");
            assert_close(
                &output.copy_to_host(&stream).expect("read compressed"),
                &expected,
                4.0e-5,
            );
            if overlapping {
                let mut overlap_kv = DeviceBuffer::zeroed(RATIO * WIDTH).expect("overlap kv");
                let mut overlap_gate = DeviceBuffer::zeroed(RATIO * WIDTH).expect("overlap gate");
                store_compression_overlap_f32_into_on_stream(
                    &kv,
                    &gate,
                    &bias,
                    overlap_kv.output(),
                    overlap_gate.output(),
                    WINDOWS,
                    RATIO,
                    WIDTH,
                    &stream,
                )
                .expect("store overlap");
                let kv = kv.copy_to_host(&stream).expect("kv host");
                let gate = gate.copy_to_host(&stream).expect("gate host");
                let bias = bias.copy_to_host(&stream).expect("bias host");
                let mut expected_kv = Vec::with_capacity(RATIO * WIDTH);
                let mut expected_gate = Vec::with_capacity(RATIO * WIDTH);
                for slot in 0..RATIO {
                    for feature in 0..WIDTH {
                        let source = ((WINDOWS - 1) * RATIO + slot) * projected + feature;
                        expected_kv.push(kv[source]);
                        expected_gate.push(gate[source] + bias[slot * projected + feature]);
                    }
                }
                assert_close(
                    &overlap_kv.copy_to_host(&stream).expect("overlap kv host"),
                    &expected_kv,
                    0.0,
                );
                assert_close(
                    &overlap_gate
                        .copy_to_host(&stream)
                        .expect("overlap gate host"),
                    &expected_gate,
                    0.0,
                );
            }
        }

        let mut positions = DeviceBuffer::zeroed(4).expect("positions");
        let stream = CudaStream::new_non_blocking().expect("stream");
        arithmetic_positions_u32_into_on_stream(positions.output(), 4, 12, 4, &stream)
            .expect("positions");
        assert_eq!(
            positions
                .copy_to_host(&stream)
                .expect("positions host")
                .as_slice(),
            &[12, 16, 20, 24]
        );
    }

    #[test]
    fn clamped_swiglu_and_routed_accumulation_match_cpu() {
        const ROWS: usize = 2;
        const ROUTES: usize = 3;
        const WIDTH: usize = 5;
        const LIMIT: f32 = 1.25;
        let gate = test_rows(ROWS * ROUTES, WIDTH, -0.8);
        let up = test_rows(ROWS * ROUTES, WIDTH, 0.35);
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| {
                let gate = gate.min(LIMIT);
                let up = up.clamp(-LIMIT, LIMIT);
                gate * (1.0 / (1.0 + (-gate).exp())) * up
            })
            .collect::<Vec<_>>();
        let weights = vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3];
        let mut expected = vec![0.0; ROWS * WIDTH];
        for row in 0..ROWS {
            for route in 0..ROUTES {
                for feature in 0..WIDTH {
                    expected[row * WIDTH + feature] += weights[row * ROUTES + route]
                        * activated[(row * ROUTES + route) * WIDTH + feature];
                }
            }
        }
        let gate = DeviceBuffer::from_host(&gate).expect("gate");
        let up = DeviceBuffer::from_host(&up).expect("up");
        let weights = DeviceBuffer::from_host(&weights).expect("weights");
        let mut activated_device = DeviceBuffer::zeroed(ROWS * ROUTES * WIDTH).expect("activated");
        let mut output = DeviceBuffer::zeroed(ROWS * WIDTH).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        swiglu_pair_clamped_f32_batch_into_on_stream(
            &gate,
            &up,
            activated_device.output(),
            ROWS * ROUTES,
            WIDTH,
            LIMIT,
            &stream,
        )
        .expect("SwiGLU");
        routed_accumulate_f32_batch_into_on_stream(
            &activated_device,
            &weights,
            output.output(),
            ROWS,
            ROUTES,
            WIDTH,
            &stream,
        )
        .expect("accumulate");
        assert_close(
            &output.copy_to_host(&stream).expect("read output"),
            &expected,
            2.0e-6,
        );
    }

    #[test]
    fn sorted_route_gather_and_accumulation_match_original_order() {
        const ROWS: usize = 3;
        const ROUTES_PER_ROW: usize = 2;
        const WIDTH: usize = 4;
        let route_indices = [2u32, 0, 1, 2, 0, 1];
        let route_weights = [0.25f32, 0.75, 0.4, 0.6, 0.2, 0.8];
        let input = (0..ROWS * WIDTH)
            .map(|index| index as f32 * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let route_output = (0..ROWS * ROUTES_PER_ROW * WIDTH)
            .map(|index| index as f32 * 0.0625 - 0.75)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let route_indices_device = DeviceBuffer::from_host(&route_indices).expect("route indices");
        let route_weights_device = DeviceBuffer::from_host(&route_weights).expect("route weights");
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let mut routes = MoeSortedRoutes::new(ROWS * ROUTES_PER_ROW, 3).expect("sorted routes");
        routes
            .sort_on_stream(&route_indices_device, &stream)
            .expect("sort routes");
        let offsets = routes
            .expert_offsets()
            .copy_to_host(&stream)
            .expect("expert offsets")
            .into_vec();
        let sorted_route_ids = routes
            .sorted_routes()
            .copy_to_host(&stream)
            .expect("sorted routes")
            .into_vec();
        let mut gathered = DeviceBuffer::zeroed(ROWS * WIDTH).expect("gathered rows");
        for expert in 0..3 {
            let start = offsets[expert] as usize;
            let end = offsets[expert + 1] as usize;
            let count = end - start;
            gather_sorted_route_rows_f32_into_on_stream(
                &input_device,
                &routes,
                gathered.output(),
                start,
                count,
                ROWS,
                ROUTES_PER_ROW,
                WIDTH,
                &stream,
            )
            .expect("gather expert rows");
            let actual = gathered
                .copy_to_host(&stream)
                .expect("gathered rows readback");
            for segment_row in 0..count {
                let source_row = sorted_route_ids[start + segment_row] as usize / ROUTES_PER_ROW;
                assert_eq!(
                    &actual[segment_row * WIDTH..(segment_row + 1) * WIDTH],
                    &input[source_row * WIDTH..(source_row + 1) * WIDTH]
                );
            }
        }

        let mut sorted_output = vec![0.0f32; route_output.len()];
        for (sorted, &original) in sorted_route_ids.iter().enumerate() {
            let source = original as usize * WIDTH;
            sorted_output[sorted * WIDTH..(sorted + 1) * WIDTH]
                .copy_from_slice(&route_output[source..source + WIDTH]);
        }
        let sorted_output = DeviceBuffer::from_host(&sorted_output).expect("sorted output");
        let mut output = DeviceBuffer::zeroed(ROWS * WIDTH).expect("output");
        routed_accumulate_sorted_f32_batch_into_on_stream(
            &sorted_output,
            &routes,
            &route_weights_device,
            output.output(),
            ROWS,
            ROUTES_PER_ROW,
            WIDTH,
            &stream,
        )
        .expect("sorted accumulation");
        let actual = output.copy_to_host(&stream).expect("output readback");
        for row in 0..ROWS {
            for feature in 0..WIDTH {
                let expected = (0..ROUTES_PER_ROW)
                    .map(|route| {
                        let original = row * ROUTES_PER_ROW + route;
                        route_weights[original] * route_output[original * WIDTH + feature]
                    })
                    .sum::<f32>();
                assert!((actual[row * WIDTH + feature] - expected).abs() <= 1.0e-6);
            }
        }
    }

    #[test]
    fn unclamped_swiglu_pair_matches_cpu_and_preserves_capacity_tail() {
        const ROWS: usize = 2;
        const WIDTH: usize = 3;
        let gate = vec![12.0f32, -12.0, 2.0, -3.0, 0.5, 9.0, 101.0, 102.0];
        let up = vec![4.0f32, -5.0, 6.0, 7.0, -8.0, 9.0, 103.0, 104.0];
        let expected = gate[..ROWS * WIDTH]
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| gate * (1.0 / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>();
        let gate = DeviceBuffer::from_host(&gate).expect("gate");
        let up = DeviceBuffer::from_host(&up).expect("up");
        let mut output = DeviceBuffer::from_host(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 105.0, 106.0])
            .expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        swiglu_pair_f32_batch_into_on_stream(&gate, &up, output.output(), ROWS, WIDTH, &stream)
            .expect("SwiGLU");
        let actual = output.copy_to_host(&stream).expect("read output");
        assert_close(&actual[..ROWS * WIDTH], &expected, 2.0e-6);
        assert_eq!(&actual[ROWS * WIDTH..], &[105.0, 106.0]);
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_compress(
        kv: &[f32],
        gate: &[f32],
        bias: &[f32],
        prior: Option<(&[f32], &[f32])>,
        windows: usize,
        ratio: usize,
        width: usize,
        overlapping: bool,
    ) -> Vec<f32> {
        let projected = if overlapping { 2 * width } else { width };
        let mut output = vec![0.0; windows * width];
        for window in 0..windows {
            for feature in 0..width {
                let mut entries = Vec::new();
                if overlapping {
                    for slot in 0..ratio {
                        if window > 0 {
                            let row = ((window - 1) * ratio + slot) * projected;
                            entries.push((
                                kv[row + feature],
                                gate[row + feature] + bias[slot * projected + feature],
                            ));
                        } else if let Some((prior_kv, prior_gate)) = prior {
                            entries.push((
                                prior_kv[slot * width + feature],
                                prior_gate[slot * width + feature],
                            ));
                        }
                    }
                }
                let component = if overlapping {
                    width + feature
                } else {
                    feature
                };
                for slot in 0..ratio {
                    let row = (window * ratio + slot) * projected;
                    entries.push((
                        kv[row + component],
                        gate[row + component] + bias[slot * projected + component],
                    ));
                }
                let maximum = entries
                    .iter()
                    .map(|entry| entry.1)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator = entries
                    .iter()
                    .map(|entry| (entry.1 - maximum).exp())
                    .sum::<f32>();
                output[window * width + feature] = entries
                    .iter()
                    .map(|entry| entry.0 * (entry.1 - maximum).exp() / denominator)
                    .sum();
            }
        }
        output
    }

    fn sqrt_softplus(value: f32) -> f32 {
        (1.0 + value.exp()).ln().sqrt()
    }

    fn cpu_topk_router(
        logits: &[f32],
        bias: &[f32],
        batch: usize,
        experts: usize,
        top_k: usize,
        scale: f32,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for row in logits.chunks_exact(experts).take(batch) {
            let scores = row
                .iter()
                .map(|&value| sqrt_softplus(value))
                .collect::<Vec<_>>();
            let mut order = (0..experts).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                (scores[right] + bias[right])
                    .total_cmp(&(scores[left] + bias[left]))
                    .then(left.cmp(&right))
            });
            let selected = &order[..top_k];
            let sum = selected.iter().map(|&index| scores[index]).sum::<f32>() + 1.0e-20;
            for &index in selected {
                indices.push(index as u32);
                weights.push(scores[index] * scale / sum);
            }
        }
        (indices, weights)
    }

    fn cpu_hash_router(
        logits: &[f32],
        table: &[i64],
        token_ids: &[u32],
        experts: usize,
        top_k: usize,
        scale: f32,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for (batch, &token) in token_ids.iter().enumerate() {
            let selected = &table[token as usize * top_k..(token as usize + 1) * top_k];
            let scores = selected
                .iter()
                .map(|&expert| sqrt_softplus(logits[batch * experts + expert as usize]))
                .collect::<Vec<_>>();
            let sum = scores.iter().sum::<f32>() + 1.0e-20;
            for (&expert, score) in selected.iter().zip(scores) {
                indices.push(expert as u32);
                weights.push(score * scale / sum);
            }
        }
        (indices, weights)
    }

    fn test_rows(rows: usize, width: usize, offset: f32) -> Vec<f32> {
        (0..rows * width)
            .map(|index| offset + (index as f32 % 11.0 - 5.0) / 7.0)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_attention(
        query: &[f32],
        sliding: [&[f32]; 2],
        sliding_lengths: &[u32],
        sliding_starts: &[u32],
        compressed: [&[f32]; 2],
        compressed_lengths: &[u32],
        selected: Option<&[i32]>,
        sinks: &[f32],
        batch_rows: usize,
        heads: usize,
        dim: usize,
        capacity: usize,
    ) -> Vec<f32> {
        let selected_count = selected.map_or(0, |values| values.len() / batch_rows);
        let mut output = vec![0.0; query.len()];
        for batch in 0..batch_rows {
            let mut entries = Vec::<&[f32]>::new();
            for logical in 0..sliding_lengths[batch] as usize {
                let slot = (sliding_starts[batch] as usize + logical) % capacity;
                entries.push(&sliding[batch][slot * dim..(slot + 1) * dim]);
            }
            if let Some(selected) = selected {
                for &index in &selected[batch * selected_count..(batch + 1) * selected_count] {
                    if index >= 0 && index < compressed_lengths[batch] as i32 {
                        let index = index as usize;
                        entries.push(&compressed[batch][index * dim..(index + 1) * dim]);
                    }
                }
            } else {
                for index in 0..compressed_lengths[batch] as usize {
                    entries.push(&compressed[batch][index * dim..(index + 1) * dim]);
                }
            }
            for head in 0..heads {
                let q = &query[(batch * heads + head) * dim..(batch * heads + head + 1) * dim];
                let logits = entries
                    .iter()
                    .map(|entry| {
                        q.iter()
                            .zip(*entry)
                            .map(|(left, right)| left * right)
                            .sum::<f32>()
                            / (dim as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let maximum = logits.iter().copied().fold(sinks[head], f32::max);
                let denominator = (sinks[head] - maximum).exp()
                    + logits
                        .iter()
                        .map(|logit| (logit - maximum).exp())
                        .sum::<f32>();
                let out =
                    &mut output[(batch * heads + head) * dim..(batch * heads + head + 1) * dim];
                for (logit, entry) in logits.iter().zip(entries.iter()) {
                    let weight = (logit - maximum).exp() / denominator;
                    for (target, &value) in out.iter_mut().zip(*entry) {
                        *target += weight * value;
                    }
                }
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_rope(
        values: &mut [f32],
        inv_freq: &[f32],
        positions: &[u32],
        batch_rows: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        direction: f32,
    ) {
        for batch in 0..batch_rows {
            for head in 0..heads {
                let base = (batch * heads + head) * head_dim + head_dim - rope_dim;
                for (pair, &frequency) in inv_freq.iter().enumerate() {
                    let angle = positions[batch] as f32 * frequency * direction;
                    let (sine, cosine) = angle.sin_cos();
                    let even = values[base + 2 * pair];
                    let odd = values[base + 2 * pair + 1];
                    values[base + 2 * pair] = even * cosine - odd * sine;
                    values[base + 2 * pair + 1] = odd * cosine + even * sine;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_hyper_prepare(
        streams: &[f32],
        function: &[f32],
        base: &[f32],
        scale: &[f32],
        batch_rows: usize,
        hidden: usize,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let flat = HYPER_STREAMS * hidden;
        let mut post = vec![0.0; batch_rows * HYPER_STREAMS];
        let mut combination = vec![0.0; batch_rows * HYPER_STREAMS * HYPER_STREAMS];
        let mut collapsed = vec![0.0; batch_rows * hidden];
        for batch in 0..batch_rows {
            let row = &streams[batch * flat..(batch + 1) * flat];
            let inverse_rms = (row.iter().map(|value| value * value).sum::<f32>() / flat as f32
                + rms_eps)
                .sqrt()
                .recip();
            let mixed = function
                .chunks_exact(flat)
                .map(|weights| {
                    row.iter()
                        .zip(weights)
                        .map(|(value, weight)| value * inverse_rms * weight)
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
            let pre = (0..HYPER_STREAMS)
                .map(|stream| sigmoid(mixed[stream] * scale[0] + base[stream]) + hc_eps)
                .collect::<Vec<_>>();
            for stream in 0..HYPER_STREAMS {
                post[batch * HYPER_STREAMS + stream] = 2.0
                    * sigmoid(
                        mixed[HYPER_STREAMS + stream] * scale[1] + base[HYPER_STREAMS + stream],
                    );
            }
            let comb = &mut combination[batch * HYPER_STREAMS * HYPER_STREAMS
                ..(batch + 1) * HYPER_STREAMS * HYPER_STREAMS];
            for source in 0..HYPER_STREAMS {
                let logits = (0..HYPER_STREAMS)
                    .map(|target| {
                        let index = source * HYPER_STREAMS + target;
                        mixed[2 * HYPER_STREAMS + index] * scale[2]
                            + base[2 * HYPER_STREAMS + index]
                    })
                    .collect::<Vec<_>>();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum = logits.iter().map(|value| (value - max).exp()).sum::<f32>();
                for target in 0..HYPER_STREAMS {
                    comb[source * HYPER_STREAMS + target] =
                        (logits[target] - max).exp() / sum + hc_eps;
                }
            }
            normalize_columns(comb, hc_eps);
            for _ in 1..sinkhorn_iters {
                normalize_rows(comb, hc_eps);
                normalize_columns(comb, hc_eps);
            }
            for feature in 0..hidden {
                collapsed[batch * hidden + feature] = (0..HYPER_STREAMS)
                    .map(|stream| pre[stream] * row[stream * hidden + feature])
                    .sum();
            }
        }
        (post, combination, collapsed)
    }

    fn cpu_hyper_apply(
        streams: &[f32],
        sublayer: &[f32],
        post: &[f32],
        comb: &[f32],
        batch_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_rows * HYPER_STREAMS * hidden];
        for batch in 0..batch_rows {
            for target in 0..HYPER_STREAMS {
                for feature in 0..hidden {
                    let mut value =
                        post[batch * HYPER_STREAMS + target] * sublayer[batch * hidden + feature];
                    for source in 0..HYPER_STREAMS {
                        value += comb[batch * HYPER_STREAMS * HYPER_STREAMS
                            + source * HYPER_STREAMS
                            + target]
                            * streams[batch * HYPER_STREAMS * hidden + source * hidden + feature];
                    }
                    output[batch * HYPER_STREAMS * hidden + target * hidden + feature] = value;
                }
            }
        }
        output
    }

    fn normalize_rows(values: &mut [f32], eps: f32) {
        for row in values.chunks_exact_mut(HYPER_STREAMS) {
            let sum = row.iter().sum::<f32>() + eps;
            row.iter_mut().for_each(|value| *value /= sum);
        }
    }

    fn normalize_columns(values: &mut [f32], eps: f32) {
        for column in 0..HYPER_STREAMS {
            let sum = (0..HYPER_STREAMS)
                .map(|row| values[row * HYPER_STREAMS + column])
                .sum::<f32>()
                + eps;
            for row in 0..HYPER_STREAMS {
                values[row * HYPER_STREAMS + column] /= sum;
            }
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let allowed = tolerance + tolerance * expected.abs();
            assert!(
                (actual - expected).abs() <= allowed,
                "mismatch at {index}: actual={actual} expected={expected} allowed={allowed}"
            );
        }
    }
}
