//! Qwen3.8 Flash Next hyperconnection elementwise kernels.

use crate::SM12X_KV_PAGE_TOKENS;
use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use std::mem::size_of;

/// Stable-slot BF16 storage for the raw QSA index key emitted for each token.
pub struct Qwen38QsaIndexPool {
    values: DeviceBuffer<u16>,
    page_slots: usize,
    head_dim: usize,
}

/// Reusable GPU workspace for QSA scoring and exact micro-block selection.
pub struct Qwen38QsaSelectionWorkspace {
    query: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    selected_blocks: DeviceBuffer<u8>,
    selected_tiles: DeviceBuffer<u8>,
    max_tokens: usize,
    heads: usize,
    head_dim: usize,
    compress_ratio: usize,
    budget: usize,
}

/// Sparse masks and effective token count produced for one QSA query.
pub struct Qwen38QsaSelection<'a> {
    /// One byte per four-token QSA micro-block.
    pub selected_blocks: &'a DeviceBuffer<u8>,
    /// One byte per 64-token compact-attention tile.
    pub selected_tiles: &'a DeviceBuffer<u8>,
    /// Number of visible tokens selected by QSA, including the incomplete tail.
    pub selected_tokens: usize,
}

impl Qwen38QsaIndexPool {
    /// Preallocates raw index-key pages using the compact KV pool's slot geometry.
    pub fn new(page_slots: usize, head_dim: usize) -> Result<Self> {
        if page_slots == 0 || head_dim == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 QSA index-key pool",
                expected: "positive page slots and head dimension".to_string(),
                actual: format!("page_slots={page_slots} head_dim={head_dim}"),
            });
        }
        let values = page_slots
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 QSA index-key pool",
                expected: "pool value count without overflow".to_string(),
                actual: format!("page_slots={page_slots} head_dim={head_dim}"),
            })?;
        Ok(Self {
            values: DeviceBuffer::uninitialized(values)?,
            page_slots,
            head_dim,
        })
    }

    /// Returns the bytes occupied by one physical page slot.
    pub fn page_bytes(&self) -> usize {
        SM12X_KV_PAGE_TOKENS * self.head_dim * size_of::<u16>()
    }

    /// Returns total bytes in the preallocated pool.
    pub fn device_bytes(&self) -> usize {
        self.values.device_bytes()
    }

    /// Copies one physical page slot on the explicit CUDA stream.
    pub fn copy_page_on_stream(
        &mut self,
        source_slot: usize,
        destination_slot: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if source_slot >= self.page_slots || destination_slot >= self.page_slots {
            return Err(Error::Shape {
                label: "Qwen3.8 QSA index-key page copy",
                expected: format!("slots below {}", self.page_slots),
                actual: format!("source={source_slot} destination={destination_slot}"),
            });
        }
        let bytes = self.page_bytes();
        let source_offset = source_slot * bytes;
        let destination_offset = destination_slot * bytes;
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D QSA index-key page)",
                ffi::cudaMemcpyAsync(
                    self.values.ptr.cast::<u8>().add(destination_offset).cast(),
                    self.values.ptr.cast::<u8>().add(source_offset).cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Appends one raw index key without scoring or selecting historical rows.
    pub fn append_key_on_stream(
        &mut self,
        projection: &DeviceBuffer<f32>,
        slot: usize,
        page_offset: usize,
        heads: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let projection_values =
            (heads + 1)
                .checked_mul(self.head_dim)
                .ok_or_else(|| Error::Shape {
                    label: "Qwen3.8 QSA index-key append",
                    expected: "projection size without overflow".to_string(),
                    actual: format!("heads={heads} head_dim={}", self.head_dim),
                })?;
        if projection.len() != projection_values
            || slot >= self.page_slots
            || page_offset >= SM12X_KV_PAGE_TOKENS
            || heads == 0
            || [slot, page_offset, heads, self.head_dim]
                .into_iter()
                .any(|value| value > u32::MAX as usize)
        {
            return Err(Error::Shape {
                label: "Qwen3.8 QSA index-key append",
                expected: format!(
                    "projection={projection_values}, valid slot/page offset, and positive heads"
                ),
                actual: format!(
                    "projection={} slot={slot}/{} page_offset={page_offset} heads={heads} head_dim={}",
                    projection.len(),
                    self.page_slots,
                    self.head_dim
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_qwen38_qsa_append_key_on_stream",
                ffi::infer_qwen38_qsa_append_key_on_stream(
                    projection.ptr,
                    self.values.ptr,
                    slot as u32,
                    page_offset as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    self.page_slots as u32,
                    heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}

impl Qwen38QsaSelectionWorkspace {
    /// Allocates selection scratch for the released QSA geometry.
    pub fn new(
        max_tokens: usize,
        heads: usize,
        head_dim: usize,
        compress_ratio: usize,
        budget: usize,
    ) -> Result<Self> {
        if max_tokens == 0
            || heads == 0
            || head_dim == 0
            || compress_ratio != 4
            || budget == 0
            || !budget.is_multiple_of(compress_ratio)
            || [max_tokens, heads, head_dim, budget]
                .into_iter()
                .any(|value| value > u32::MAX as usize)
        {
            return Err(Error::Shape {
                label: "Qwen3.8 QSA selection workspace",
                expected: "positive u32 dimensions, four-token compression, and divisible budget"
                    .to_string(),
                actual: format!(
                    "max_tokens={max_tokens} heads={heads} head_dim={head_dim} compress={compress_ratio} budget={budget}"
                ),
            });
        }
        let blocks = max_tokens.div_ceil(compress_ratio);
        let tiles = max_tokens.div_ceil(64);
        Ok(Self {
            query: DeviceBuffer::zeroed(heads * head_dim)?,
            scores: DeviceBuffer::zeroed(blocks)?,
            selected_blocks: DeviceBuffer::zeroed(blocks)?,
            selected_tiles: DeviceBuffer::zeroed(tiles)?,
            max_tokens,
            heads,
            head_dim,
            compress_ratio,
            budget,
        })
    }

    /// Appends the raw index key and selects the visible QSA micro-blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_and_select_on_stream<'a>(
        &'a mut self,
        projection: &DeviceBuffer<f32>,
        q_norm: &DeviceBuffer<f32>,
        k_norm: &DeviceBuffer<f32>,
        pool: &mut Qwen38QsaIndexPool,
        page_table: &DeviceBuffer<u32>,
        slot: usize,
        page_offset: usize,
        cache_len: usize,
        rotary_dim: usize,
        eps: f32,
        theta: f32,
        stream: &CudaStream,
    ) -> Result<Qwen38QsaSelection<'a>> {
        let projection_values = (self.heads + 1) * self.head_dim;
        let logical_pages = cache_len.div_ceil(SM12X_KV_PAGE_TOKENS);
        if projection.len() != projection_values
            || q_norm.len() != self.head_dim
            || k_norm.len() != self.head_dim
            || pool.head_dim != self.head_dim
            || slot >= pool.page_slots
            || page_offset >= SM12X_KV_PAGE_TOKENS
            || cache_len == 0
            || cache_len > self.max_tokens
            || page_table.len() < logical_pages
            || rotary_dim == 0
            || rotary_dim > self.head_dim
            || !rotary_dim.is_multiple_of(2)
            || eps <= 0.0
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(Error::Shape {
                label: "Qwen3.8 QSA selection",
                expected: format!(
                    "projection={projection_values}, norms={}, valid pool/page/cache/RoPE geometry",
                    self.head_dim
                ),
                actual: format!(
                    "projection={} q_norm={} k_norm={} slot={slot}/{} page_offset={page_offset} cache_len={cache_len}/{} pages={} needed={logical_pages} rotary_dim={rotary_dim} eps={eps} theta={theta}",
                    projection.len(),
                    q_norm.len(),
                    k_norm.len(),
                    pool.page_slots,
                    self.max_tokens,
                    page_table.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_qwen38_qsa_prepare_and_select_on_stream",
                ffi::infer_qwen38_qsa_prepare_and_select_on_stream(
                    projection.ptr,
                    q_norm.ptr,
                    k_norm.ptr,
                    pool.values.ptr,
                    page_table.as_const_ptr().cast(),
                    self.query.ptr,
                    self.scores.ptr,
                    self.selected_blocks.ptr,
                    self.selected_tiles.ptr,
                    slot as u32,
                    page_offset as u32,
                    cache_len as u32,
                    self.max_tokens as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    pool.page_slots as u32,
                    self.heads as u32,
                    self.head_dim as u32,
                    rotary_dim as u32,
                    self.compress_ratio as u32,
                    self.budget as u32,
                    eps,
                    theta,
                    stream.as_raw(),
                ),
            )?;
        }
        let complete_blocks = cache_len / self.compress_ratio;
        let tail = cache_len % self.compress_ratio;
        let selected_tokens =
            complete_blocks.min(self.budget / self.compress_ratio) * self.compress_ratio + tail;
        Ok(Qwen38QsaSelection {
            selected_blocks: &self.selected_blocks,
            selected_tiles: &self.selected_tiles,
            selected_tokens,
        })
    }

    /// Returns the normalized, RoPE'd index query for focused validation.
    pub fn query(&self) -> &DeviceBuffer<f32> {
        &self.query
    }

    /// Returns block scores for focused validation.
    pub fn scores(&self) -> &DeviceBuffer<f32> {
        &self.scores
    }

    /// Returns the exact bytes owned by selection scratch.
    pub fn device_bytes(&self) -> usize {
        self.query.device_bytes()
            + self.scores.device_bytes()
            + self.selected_blocks.device_bytes()
            + self.selected_tiles.device_bytes()
    }
}

/// Applies per-branch Gemma-style RMSNorm to Qwen hyperconnection streams.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_hc_norm_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    delta_weight: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 hyperconnection width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let values = checked_values(tokens, hc_dim, "Qwen3.8 hyperconnection norm")?;
    validate_dims(tokens, hidden, hc_count)?;
    if input.len() < values || output.len() < values || delta_weight.len() != hc_dim || eps <= 0.0 {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection norm",
            expected: format!(
                "input/output >= {values}, delta weight = {hc_dim}, positive epsilon"
            ),
            actual: format!(
                "input={} output={} delta_weight={} eps={eps}",
                input.len(),
                output.len(),
                delta_weight.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_norm_f32_on_stream",
            ffi::infer_qwen38_hc_norm_f32_on_stream(
                input.ptr,
                delta_weight.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies `SiLU(x / hc_count)` in place to the low-rank mix projection.
pub fn qwen38_hc_silu_scale_f32_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    count: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    if count == 0 || count > values.len() || hc_count == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection low-rank activation",
            expected: "non-empty in-place prefix and positive hc_count".to_string(),
            actual: format!("count={count} buffer={} hc_count={hc_count}", values.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_silu_scale_f32_on_stream",
            ffi::infer_qwen38_hc_silu_scale_f32_on_stream(
                values.as_mut_ptr().cast(),
                count,
                1.0 / hc_count as f32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies sigmoid mix gates and averages the normalized residual streams.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_hc_collapse_f32_into_on_stream(
    normed: &DeviceBuffer<f32>,
    gate_logits: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 hyperconnection width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let stream_values = checked_values(tokens, hc_dim, "Qwen3.8 hyperconnection collapse")?;
    let output_values = checked_values(tokens, hidden, "Qwen3.8 hyperconnection collapse")?;
    validate_dims(tokens, hidden, hc_count)?;
    if normed.len() < stream_values
        || gate_logits.len() < stream_values
        || output.len() < output_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection collapse",
            expected: format!("normed/gates >= {stream_values}, output >= {output_values}"),
            actual: format!(
                "normed={} gates={} output={}",
                normed.len(),
                gate_logits.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_collapse_f32_on_stream",
            ffi::infer_qwen38_hc_collapse_f32_on_stream(
                normed.ptr,
                gate_logits.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Injects one block output into each residual stream with learned sigmoid gates.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_hc_combine_f32_into_on_stream(
    residual: &DeviceBuffer<f32>,
    block_output: &DeviceBuffer<f32>,
    inject_logits: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 hyperconnection width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let residual_values = checked_values(tokens, hc_dim, "Qwen3.8 hyperconnection combine")?;
    let block_values = checked_values(tokens, hidden, "Qwen3.8 hyperconnection combine")?;
    let inject_values = checked_values(tokens, hc_count, "Qwen3.8 hyperconnection combine")?;
    validate_dims(tokens, hidden, hc_count)?;
    if residual.len() < residual_values
        || block_output.len() < block_values
        || inject_logits.len() < inject_values
        || output.len() < residual_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection combine",
            expected: format!(
                "residual/output >= {residual_values}, block >= {block_values}, inject >= {inject_values}"
            ),
            actual: format!(
                "residual={} block={} inject={} output={}",
                residual.len(),
                block_output.len(),
                inject_logits.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_combine_f32_on_stream",
            ffi::infer_qwen38_hc_combine_f32_on_stream(
                residual.ptr,
                block_output.ptr,
                inject_logits.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Repeats each hidden row into its initial hyperconnection streams.
pub fn qwen38_repeat_streams_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_dims(tokens, hidden, hc_count)?;
    let input_count = checked_values(tokens, hidden, "Qwen3.8 initial hidden")?;
    let count = checked_values(input_count, hc_count, "Qwen3.8 initial streams")?;
    if input.len() < input_count || output.len() < count {
        return Err(Error::Shape {
            label: "Qwen3.8 initial streams",
            expected: format!("input>={input_count}, output>={count}"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_repeat_streams_f32_on_stream",
            ffi::infer_qwen38_repeat_streams_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes the signed-square-root PLE gate and broadcasts its value projection.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_ple_gate_value_f32_into_on_stream(
    key_normed: &DeviceBuffer<f32>,
    query_normed: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    mut gated: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 PLE gate width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let stream_values = checked_values(tokens, hc_dim, "Qwen3.8 PLE gate")?;
    let value_values = checked_values(tokens, hidden, "Qwen3.8 PLE gate")?;
    validate_dims(tokens, hidden, hc_count)?;
    if key_normed.len() < stream_values
        || query_normed.len() < stream_values
        || value.len() < value_values
        || gated.len() < stream_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 PLE gate",
            expected: format!("key/query/gated >= {stream_values}, value >= {value_values}"),
            actual: format!(
                "key={} query={} value={} gated={}",
                key_normed.len(),
                query_normed.len(),
                value.len(),
                gated.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_ple_gate_value_f32_on_stream",
            ffi::infer_qwen38_ple_gate_value_f32_on_stream(
                key_normed.ptr,
                query_normed.ptr,
                value.ptr,
                gated.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies the causal dilated PLE depthwise convolution and updates its state.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_ple_conv_update_f32_into_on_stream(
    normalized: &DeviceBuffer<f32>,
    gated: &DeviceBuffer<f32>,
    weight_bf16: &DeviceBuffer<u16>,
    state: &mut DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    channels: usize,
    kernel: usize,
    dilation: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = checked_values(tokens, channels, "Qwen3.8 PLE convolution")?;
    let weight_values = checked_values(channels, kernel, "Qwen3.8 PLE convolution")?;
    let history = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 PLE convolution history",
            expected: "(kernel - 1) * dilation without overflow".to_string(),
            actual: format!("kernel={kernel} dilation={dilation}"),
        })?;
    let state_values = checked_values(channels, history, "Qwen3.8 PLE convolution")?;
    if tokens == 0
        || channels == 0
        || kernel < 2
        || dilation == 0
        || tokens > u32::MAX as usize
        || channels > u32::MAX as usize
        || kernel > u32::MAX as usize
        || dilation > u32::MAX as usize
        || normalized.len() < values
        || gated.len() < values
        || output.len() < values
        || weight_bf16.len() != weight_values
        || state.len() != state_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 PLE convolution",
            expected: format!(
                "normalized/gated/output >= {values}, weights={weight_values}, state={state_values}, valid u32 dimensions"
            ),
            actual: format!(
                "normalized={} gated={} output={} weights={} state={} tokens={tokens} channels={channels} kernel={kernel} dilation={dilation}",
                normalized.len(),
                gated.len(),
                output.len(),
                weight_bf16.len(),
                state.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_ple_conv_update_f32_on_stream",
            ffi::infer_qwen38_ple_conv_update_f32_on_stream(
                normalized.ptr,
                gated.ptr,
                weight_bf16.ptr,
                state.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                channels as u32,
                kernel as u32,
                dilation as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn checked_values(tokens: usize, width: usize, label: &'static str) -> Result<usize> {
    tokens.checked_mul(width).ok_or_else(|| Error::Shape {
        label,
        expected: "tokens * width without overflow".to_string(),
        actual: format!("tokens={tokens} width={width}"),
    })
}

fn validate_dims(tokens: usize, hidden: usize, hc_count: usize) -> Result<()> {
    if tokens == 0
        || hidden == 0
        || hc_count == 0
        || tokens > u32::MAX as usize
        || hidden > u32::MAX as usize
        || hc_count > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection dimensions",
            expected: "positive u32-sized tokens, hidden, and hc_count".to_string(),
            actual: format!("tokens={tokens} hidden={hidden} hc_count={hc_count}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen38QsaIndexPool, Qwen38QsaSelectionWorkspace, qwen38_hc_collapse_f32_into_on_stream,
        qwen38_hc_combine_f32_into_on_stream, qwen38_hc_norm_f32_into_on_stream,
        qwen38_hc_silu_scale_f32_in_place_on_stream, qwen38_ple_conv_update_f32_into_on_stream,
        qwen38_ple_gate_value_f32_into_on_stream, qwen38_repeat_streams_f32_into_on_stream,
    };
    use crate::format::{bf16_to_f32, f32_to_bf16};
    use crate::{CudaStream, DeviceBuffer};

    #[test]
    fn qsa_paged_selector_matches_released_micro_block_formula() {
        const HEADS: usize = 4;
        const HEAD_DIM: usize = 128;
        const ROTARY_DIM: usize = 64;
        const COMPRESS: usize = 4;
        const BUDGET: usize = 2_048;
        const CACHE_LEN: usize = 2_059;
        const MAX_TOKENS: usize = 2_176;
        const EPS: f32 = 1e-6;
        const THETA: f32 = 10_000_000.0;

        let logical_pages = MAX_TOKENS.div_ceil(crate::SM12X_KV_PAGE_TOKENS);
        let slots = (0..logical_pages)
            .map(|page| ((page * 7) % logical_pages) as u32)
            .collect::<Vec<_>>();
        let page_table = DeviceBuffer::from_host(&slots).expect("page table");
        let mut pool = Qwen38QsaIndexPool::new(logical_pages, HEAD_DIM).expect("index pool");
        let mut physical = vec![0u16; logical_pages * crate::SM12X_KV_PAGE_TOKENS * HEAD_DIM];
        let mut logical = vec![0u16; CACHE_LEN * HEAD_DIM];
        for token in 0..CACHE_LEN {
            for dim in 0..HEAD_DIM {
                let value = (((token * 17 + dim * 13) % 257) as f32 - 128.0) / 96.0;
                let encoded = f32_to_bf16(value);
                logical[token * HEAD_DIM + dim] = encoded;
                let slot = slots[token / crate::SM12X_KV_PAGE_TOKENS] as usize;
                let page_offset = token % crate::SM12X_KV_PAGE_TOKENS;
                physical[(slot * crate::SM12X_KV_PAGE_TOKENS + page_offset) * HEAD_DIM + dim] =
                    encoded;
            }
        }
        pool.values
            .copy_from_host(&physical)
            .expect("populate index pool");

        let projection_host = (0..(HEADS + 1) * HEAD_DIM)
            .map(|index| ((index * 29 % 251) as f32 - 125.0) / 80.0)
            .collect::<Vec<_>>();
        for dim in 0..HEAD_DIM {
            logical[(CACHE_LEN - 1) * HEAD_DIM + dim] =
                f32_to_bf16(projection_host[HEADS * HEAD_DIM + dim]);
        }
        let projection = DeviceBuffer::from_host(&projection_host).expect("projection");
        let q_norm_host = (0..HEAD_DIM)
            .map(|dim| 0.75 + dim as f32 / 512.0)
            .collect::<Vec<_>>();
        let k_norm_host = (0..HEAD_DIM)
            .map(|dim| 0.85 + dim as f32 / 640.0)
            .collect::<Vec<_>>();
        let q_norm = DeviceBuffer::from_host(&q_norm_host).expect("q norm");
        let k_norm = DeviceBuffer::from_host(&k_norm_host).expect("k norm");
        let mut workspace =
            Qwen38QsaSelectionWorkspace::new(MAX_TOKENS, HEADS, HEAD_DIM, COMPRESS, BUDGET)
                .expect("selector");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let last_page = CACHE_LEN - 1;
        let last_slot = slots[last_page / crate::SM12X_KV_PAGE_TOKENS] as usize;
        let selected_tokens = workspace
            .prepare_and_select_on_stream(
                &projection,
                &q_norm,
                &k_norm,
                &mut pool,
                &page_table,
                last_slot,
                last_page % crate::SM12X_KV_PAGE_TOKENS,
                CACHE_LEN,
                ROTARY_DIM,
                EPS,
                THETA,
                &stream,
            )
            .expect("select")
            .selected_tokens;
        assert_eq!(selected_tokens, BUDGET + CACHE_LEN % COMPRESS);

        let query_actual = workspace
            .query()
            .copy_to_host(&stream)
            .expect("query readback");
        let scores_actual = workspace
            .scores()
            .copy_to_host(&stream)
            .expect("score readback");
        let blocks_actual = workspace
            .selected_blocks
            .copy_to_host(&stream)
            .expect("block mask readback");
        let tiles_actual = workspace
            .selected_tiles
            .copy_to_host(&stream)
            .expect("tile mask readback");

        let mut query_expected = vec![0.0; HEADS * HEAD_DIM];
        for head in 0..HEADS {
            let source = &projection_host[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            let normalized = rms_norm(source, &q_norm_host, EPS);
            query_expected[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&rope(
                &normalized,
                ROTARY_DIM,
                CACHE_LEN - 1,
                THETA,
            ));
        }
        assert_close(&query_actual, &query_expected, 5e-4);

        let complete_blocks = CACHE_LEN / COMPRESS;
        let mut scores_expected = vec![0.0f32; complete_blocks];
        for block in 0..complete_blocks {
            let mut pooled = vec![0.0; HEAD_DIM];
            for dim in 0..HEAD_DIM {
                let mean = (0..COMPRESS)
                    .map(|row| bf16_to_f32(logical[(block * COMPRESS + row) * HEAD_DIM + dim]))
                    .sum::<f32>()
                    / COMPRESS as f32;
                pooled[dim] = bf16_to_f32(f32_to_bf16(mean));
            }
            let key = rope(
                &rms_norm(&pooled, &k_norm_host, EPS),
                ROTARY_DIM,
                block * COMPRESS,
                THETA,
            );
            scores_expected[block] = (0..HEADS)
                .map(|head| {
                    query_expected[head * HEAD_DIM..(head + 1) * HEAD_DIM]
                        .iter()
                        .zip(&key)
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        .max(0.0)
                })
                .sum::<f32>()
                / (HEAD_DIM as f32).sqrt();
        }
        assert_close(&scores_actual[..complete_blocks], &scores_expected, 2e-3);

        let mut ranking = (0..complete_blocks).collect::<Vec<_>>();
        ranking.sort_by(|&left, &right| {
            scores_expected[right]
                .total_cmp(&scores_expected[left])
                .then_with(|| left.cmp(&right))
        });
        let mut blocks_expected = vec![0u8; MAX_TOKENS.div_ceil(COMPRESS)];
        for &block in &ranking[..BUDGET / COMPRESS] {
            blocks_expected[block] = 1;
        }
        if !CACHE_LEN.is_multiple_of(COMPRESS) {
            blocks_expected[complete_blocks] = 1;
        }
        assert_eq!(&*blocks_actual, &blocks_expected);
        let mut tiles_expected = vec![0u8; MAX_TOKENS.div_ceil(64)];
        for (block, &selected) in blocks_expected.iter().enumerate() {
            if selected != 0 {
                tiles_expected[block / 16] = 1;
            }
        }
        assert_eq!(&*tiles_actual, &tiles_expected);
    }

    #[test]
    fn qsa_append_only_key_matches_selector_append() {
        const HEADS: usize = 4;
        const HEAD_DIM: usize = 128;
        let projection_host = (0..(HEADS + 1) * HEAD_DIM)
            .map(|index| (index as f32 - 177.0) / 91.0)
            .collect::<Vec<_>>();
        let projection = DeviceBuffer::from_host(&projection_host).expect("projection");
        let norm = DeviceBuffer::from_host(&vec![1.0f32; HEAD_DIM]).expect("norm");
        let page_table = DeviceBuffer::from_host(&[0u32]).expect("page table");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut selected_pool = Qwen38QsaIndexPool::new(1, HEAD_DIM).expect("selected pool");
        let mut append_pool = Qwen38QsaIndexPool::new(1, HEAD_DIM).expect("append pool");
        let empty_page = vec![0u16; crate::SM12X_KV_PAGE_TOKENS * HEAD_DIM];
        selected_pool
            .values
            .copy_from_host(&empty_page)
            .expect("clear selected pool");
        append_pool
            .values
            .copy_from_host(&empty_page)
            .expect("clear append pool");
        let mut selector =
            Qwen38QsaSelectionWorkspace::new(128, HEADS, HEAD_DIM, 4, 128).expect("selector");
        selector
            .prepare_and_select_on_stream(
                &projection,
                &norm,
                &norm,
                &mut selected_pool,
                &page_table,
                0,
                0,
                1,
                64,
                1e-6,
                10_000_000.0,
                &stream,
            )
            .expect("selector append");
        append_pool
            .append_key_on_stream(&projection, 0, 0, HEADS, &stream)
            .expect("append only");
        assert_eq!(
            selected_pool
                .values
                .copy_to_host(&stream)
                .expect("selected pool readback"),
            append_pool
                .values
                .copy_to_host(&stream)
                .expect("append pool readback")
        );
    }

    fn rms_norm(values: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let inverse = (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32
            + eps)
            .sqrt()
            .recip();
        values
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inverse * weight)
            .collect()
    }

    fn rope(values: &[f32], rotary_dim: usize, position: usize, theta: f32) -> Vec<f32> {
        let mut output = values.to_vec();
        let half = rotary_dim / 2;
        for pair in 0..half {
            let angle = position as f32 * theta.powf(-2.0 * pair as f32 / rotary_dim as f32);
            let (sine, cosine) = angle.sin_cos();
            let first = values[pair];
            let second = values[pair + half];
            output[pair] = first * cosine - second * sine;
            output[pair + half] = second * cosine + first * sine;
        }
        output
    }

    #[test]
    fn hyperconnection_elementwise_kernels_match_cpu_formula() {
        const TOKENS: usize = 2;
        const HIDDEN: usize = 4;
        const HC: usize = 2;
        const EPS: f32 = 1e-6;
        let hidden_host = (0..TOKENS * HIDDEN)
            .map(|index| (index as f32 - 3.0) / 5.0)
            .collect::<Vec<_>>();
        let hidden = DeviceBuffer::from_host(&hidden_host).expect("hidden");
        let mut repeated = DeviceBuffer::zeroed(TOKENS * HIDDEN * HC).expect("repeated");
        let stream = CudaStream::new_non_blocking().expect("stream");
        qwen38_repeat_streams_f32_into_on_stream(
            &hidden,
            repeated.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("repeat streams");
        let repeated_host = repeated.copy_to_host(&stream).expect("repeat readback");
        let repeated_expected = hidden_host
            .chunks_exact(HIDDEN)
            .flat_map(|row| std::iter::repeat_n(row, HC).flatten().copied())
            .collect::<Vec<_>>();
        assert_eq!(repeated_host, repeated_expected);

        let input_host = (0..TOKENS * HIDDEN * HC)
            .map(|index| (index as f32 - 5.0) / 4.0)
            .collect::<Vec<_>>();
        let delta_host = (0..HIDDEN * HC)
            .map(|index| (index as f32 - 3.0) / 32.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let delta = DeviceBuffer::from_host(&delta_host).expect("delta");
        let mut normed = DeviceBuffer::zeroed(input_host.len()).expect("normed");
        qwen38_hc_norm_f32_into_on_stream(
            &input,
            &delta,
            normed.output(),
            TOKENS,
            HIDDEN,
            HC,
            EPS,
            &stream,
        )
        .expect("norm");
        let normed_host = normed.copy_to_host(&stream).expect("norm readback");
        let mut norm_expected = vec![0.0f32; input_host.len()];
        for token in 0..TOKENS {
            for branch in 0..HC {
                let offset = (token * HC + branch) * HIDDEN;
                let square_mean = input_host[offset..offset + HIDDEN]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / HIDDEN as f32;
                let inverse_rms = 1.0 / (square_mean + EPS).sqrt();
                for col in 0..HIDDEN {
                    norm_expected[offset + col] = input_host[offset + col]
                        * inverse_rms
                        * (1.0 + delta_host[branch * HIDDEN + col]);
                }
            }
        }
        assert_close(&normed_host, &norm_expected, 2e-5);

        let gate_host = (0..input_host.len())
            .map(|index| (index as f32 - 7.0) / 3.0)
            .collect::<Vec<_>>();
        let gates = DeviceBuffer::from_host(&gate_host).expect("gates");
        let mut mixed = DeviceBuffer::zeroed(TOKENS * HIDDEN).expect("mixed");
        qwen38_hc_collapse_f32_into_on_stream(
            &normed,
            &gates,
            mixed.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("collapse");
        let mixed_host = mixed.copy_to_host(&stream).expect("mixed readback");
        let mut mixed_expected = vec![0.0f32; TOKENS * HIDDEN];
        for token in 0..TOKENS {
            for col in 0..HIDDEN {
                for branch in 0..HC {
                    let offset = (token * HC + branch) * HIDDEN + col;
                    mixed_expected[token * HIDDEN + col] +=
                        sigmoid(gate_host[offset]) * norm_expected[offset] / HC as f32;
                }
            }
        }
        assert_close(&mixed_host, &mixed_expected, 2e-5);

        let inject_host = vec![-1.0, 0.5, 2.0, -0.25];
        let inject = DeviceBuffer::from_host(&inject_host).expect("inject");
        let mut combined = DeviceBuffer::zeroed(input_host.len()).expect("combined");
        qwen38_hc_combine_f32_into_on_stream(
            &input,
            &mixed,
            &inject,
            combined.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("combine");
        let combined_host = combined.copy_to_host(&stream).expect("combine readback");
        let combined_expected = input_host
            .iter()
            .enumerate()
            .map(|(index, residual)| {
                let token = index / (HC * HIDDEN);
                let within = index % (HC * HIDDEN);
                let branch = within / HIDDEN;
                let col = within % HIDDEN;
                residual
                    + 2.0
                        * sigmoid(inject_host[token * HC + branch] / HC as f32)
                        * mixed_expected[token * HIDDEN + col]
            })
            .collect::<Vec<_>>();
        assert_close(&combined_host, &combined_expected, 2e-5);

        let activation_host = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
        let mut activation = DeviceBuffer::from_host(&activation_host).expect("activation");
        qwen38_hc_silu_scale_f32_in_place_on_stream(
            activation.inout(),
            activation_host.len(),
            HC,
            &stream,
        )
        .expect("activation");
        let activation_actual = activation
            .copy_to_host(&stream)
            .expect("activation readback");
        let activation_expected = activation_host
            .into_iter()
            .map(|value| {
                let scaled = value / HC as f32;
                scaled * sigmoid(scaled)
            })
            .collect::<Vec<_>>();
        assert_close(&activation_actual, &activation_expected, 2e-5);
    }

    #[test]
    fn ple_gate_and_dilated_convolution_match_cpu_formula() {
        const TOKENS: usize = 3;
        const HIDDEN: usize = 4;
        const HC: usize = 2;
        let key_host = (0..TOKENS * HIDDEN * HC)
            .map(|index| (index as f32 - 9.0) / 7.0)
            .collect::<Vec<_>>();
        let query_host = (0..key_host.len())
            .map(|index| (5.0 - index as f32) / 9.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * HIDDEN)
            .map(|index| (index as f32 + 1.0) / 8.0)
            .collect::<Vec<_>>();
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let value = DeviceBuffer::from_host(&value_host).expect("value");
        let mut gated = DeviceBuffer::zeroed(key_host.len()).expect("gated");
        let stream = CudaStream::new_non_blocking().expect("stream");
        qwen38_ple_gate_value_f32_into_on_stream(
            &key,
            &query,
            &value,
            gated.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("gate value");
        let gated_host = gated.copy_to_host(&stream).expect("gated readback");
        let mut gated_expected = vec![0.0f32; key_host.len()];
        for token in 0..TOKENS {
            for branch in 0..HC {
                let offset = (token * HC + branch) * HIDDEN;
                let scaled_dot = key_host[offset..offset + HIDDEN]
                    .iter()
                    .zip(&query_host[offset..offset + HIDDEN])
                    .map(|(key, query)| key * query)
                    .sum::<f32>()
                    / (HIDDEN as f32).sqrt();
                let signed_root = scaled_dot.signum() * scaled_dot.abs().max(1e-6).sqrt();
                let gate = sigmoid(signed_root);
                for col in 0..HIDDEN {
                    gated_expected[offset + col] = gate * value_host[token * HIDDEN + col];
                }
            }
        }
        assert_close(&gated_host, &gated_expected, 2e-5);

        const CHANNELS: usize = HIDDEN * HC;
        const KERNEL: usize = 3;
        const DILATION: usize = 2;
        const HISTORY: usize = (KERNEL - 1) * DILATION;
        let normalized_host = (0..TOKENS * CHANNELS)
            .map(|index| (index as f32 - 4.0) / 11.0)
            .collect::<Vec<_>>();
        let state_host = (0..CHANNELS * HISTORY)
            .map(|index| (index as f32 - 8.0) / 13.0)
            .collect::<Vec<_>>();
        let weight_bf16 = (0..CHANNELS * KERNEL)
            .map(|index| f32_to_bf16((index as f32 - 6.0) / 17.0))
            .collect::<Vec<_>>();
        let normalized = DeviceBuffer::from_host(&normalized_host).expect("normalized");
        let weights = DeviceBuffer::from_host(&weight_bf16).expect("weights");
        let mut state = DeviceBuffer::from_host(&state_host).expect("state");
        let mut output = DeviceBuffer::zeroed(TOKENS * CHANNELS).expect("output");
        qwen38_ple_conv_update_f32_into_on_stream(
            &normalized,
            &gated,
            &weights,
            &mut state,
            output.output(),
            TOKENS,
            CHANNELS,
            KERNEL,
            DILATION,
            &stream,
        )
        .expect("convolution");
        let output_actual = output.copy_to_host(&stream).expect("output readback");
        let state_actual = state.copy_to_host(&stream).expect("state readback");
        let mut output_expected = vec![0.0f32; TOKENS * CHANNELS];
        let mut state_expected = vec![0.0f32; CHANNELS * HISTORY];
        for channel in 0..CHANNELS {
            let extended = state_host[channel * HISTORY..(channel + 1) * HISTORY]
                .iter()
                .copied()
                .chain((0..TOKENS).map(|token| normalized_host[token * CHANNELS + channel]))
                .collect::<Vec<_>>();
            for token in 0..TOKENS {
                let conv = (0..KERNEL)
                    .map(|tap| {
                        let source = HISTORY + token - (KERNEL - 1 - tap) * DILATION;
                        extended[source] * bf16_to_f32(weight_bf16[channel * KERNEL + tap])
                    })
                    .sum::<f32>();
                output_expected[token * CHANNELS + channel] =
                    gated_expected[token * CHANNELS + channel] + conv * sigmoid(conv);
            }
            state_expected[channel * HISTORY..(channel + 1) * HISTORY]
                .copy_from_slice(&extended[TOKENS..TOKENS + HISTORY]);
        }
        assert_close(&output_actual, &output_expected, 3e-5);
        assert_close(&state_actual, &state_expected, 1e-7);
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual} expected={expected}"
            );
        }
    }
}
