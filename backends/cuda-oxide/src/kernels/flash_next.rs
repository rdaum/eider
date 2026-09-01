//! Qwen3.8 Flash Next hyperconnection, PLE, and QSA kernels.

use crate::common::*;
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{
    SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

/// CUDA entry points specific to Qwen3.8 Flash Next.
#[cuda_module]
mod device {
    use super::*;

    #[inline(always)]
    unsafe fn qwen38_rope_value(
        values: *const f32,
        dim: u32,
        rotary_dim: u32,
        position: u32,
        theta: f32,
    ) -> f32 {
        if dim >= rotary_dim {
            return unsafe { *values.add(dim as usize) };
        }
        let half = rotary_dim / 2;
        let pair = dim % half;
        let exponent = -2.0 * pair as f32 / rotary_dim as f32;
        let angle = position as f32 * theta.powf(exponent);
        let first = unsafe { *values.add(pair as usize) };
        let second = unsafe { *values.add((pair + half) as usize) };
        if dim < half {
            first * angle.cos() - second * angle.sin()
        } else {
            second * angle.cos() + first * angle.sin()
        }
    }

    /// Applies per-stream RMS normalization for Qwen3.8 Flash Next hyperconnections.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_hc_norm_f32(
        input: *const f32,
        delta_weight: *const f32,
        output: *mut f32,
        hidden: u32,
        hc_count: u32,
        eps: f32,
    ) {
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let group = thread::blockIdx_x();
        let branch = group % hc_count;
        let lane = thread::threadIdx_x();
        let offset = group as usize * hidden as usize;
        let mut square_sum = 0.0f32;
        let mut col = lane;
        while col < hidden {
            let value = unsafe { *input.add(offset + col as usize) };
            square_sum = value.mul_add(value, square_sum);
            col += thread::blockDim_x();
        }
        unsafe { reduction.add(lane as usize).write(square_sum) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe {
                    *reduction.add(lane as usize) += *reduction.add((lane + stride) as usize)
                };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_rms = 1.0 / (unsafe { *reduction } / hidden as f32 + eps).sqrt();
        let weight_offset = branch as usize * hidden as usize;
        col = lane;
        while col < hidden {
            let index = offset + col as usize;
            unsafe {
                output.add(index).write(
                    *input.add(index)
                        * inverse_rms
                        * (1.0 + *delta_weight.add(weight_offset + col as usize)),
                )
            };
            col += thread::blockDim_x();
        }
    }

    /// Applies scaled SiLU for a Qwen3.8 Flash Next hyperconnection.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen38_hc_silu_scale_f32(values: *mut f32, count: u64, scale: f32) {
        let index = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
            + thread::threadIdx_x() as u64;
        if index < count {
            let value = unsafe { *values.add(index as usize) } * scale;
            unsafe { values.add(index as usize).write(value * sigmoid(value)) };
        }
    }

    /// Collapses Qwen3.8 Flash Next streams with learned sigmoid gates.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_hc_collapse_f32(
        normed: *const f32,
        gate_logits: *const f32,
        output: *mut f32,
        tokens: u32,
        hidden: u32,
        hc_count: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let count = tokens * hidden;
        if index >= count {
            return;
        }
        let token = index / hidden;
        let col = index - token * hidden;
        let token_offset = token as usize * hc_count as usize * hidden as usize;
        let mut sum = 0.0f32;
        let mut branch = 0;
        while branch < hc_count {
            let offset = token_offset + branch as usize * hidden as usize + col as usize;
            sum += sigmoid(unsafe { *gate_logits.add(offset) }) * unsafe { *normed.add(offset) };
            branch += 1;
        }
        unsafe { output.add(index as usize).write(sum / hc_count as f32) };
    }

    /// Injects one Qwen3.8 Flash Next block output into all streams.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_hc_combine_f32(
        residual: *const f32,
        block_output: *const f32,
        inject_logits: *const f32,
        output: *mut f32,
        tokens: u32,
        hidden: u32,
        hc_count: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stream_width = hc_count * hidden;
        let count = tokens * stream_width;
        if index >= count {
            return;
        }
        let token = index / stream_width;
        let within_token = index - token * stream_width;
        let branch = within_token / hidden;
        let col = within_token - branch * hidden;
        let logit =
            unsafe { *inject_logits.add((token * hc_count + branch) as usize) } / hc_count as f32;
        let injection = 2.0 * sigmoid(logit);
        unsafe {
            output.add(index as usize).write(
                *residual.add(index as usize)
                    + injection * *block_output.add((token * hidden + col) as usize),
            )
        };
    }

    /// Repeats each Qwen3.8 Flash Next hidden row across its streams.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_repeat_streams_f32(
        input: *const f32,
        output: *mut f32,
        tokens: u32,
        hidden: u32,
        hc_count: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stream_width = hc_count * hidden;
        let count = tokens * stream_width;
        if index < count {
            let token = index / stream_width;
            let col = index % hidden;
            unsafe {
                output
                    .add(index as usize)
                    .write(*input.add((token * hidden + col) as usize))
            };
        }
    }

    /// Computes the signed-square-root gate for Qwen3.8 Flash Next PLE values.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_ple_gate_value_f32(
        key: *const f32,
        query: *const f32,
        value: *const f32,
        gated: *mut f32,
        hidden: u32,
        hc_count: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let group = thread::blockIdx_x();
        let token = group / hc_count;
        let lane = thread::threadIdx_x();
        let stream_offset = group as usize * hidden as usize;
        let mut dot = 0.0f32;
        let mut col = lane;
        while col < hidden {
            let index = stream_offset + col as usize;
            dot = unsafe { (*key.add(index)).mul_add(*query.add(index), dot) };
            col += thread::blockDim_x();
        }
        unsafe { reduction.add(lane as usize).write(dot) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe {
                    *reduction.add(lane as usize) += *reduction.add((lane + stride) as usize)
                };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let scaled = unsafe { *reduction } / (hidden as f32).sqrt();
        let signed_root = if scaled > 0.0 {
            scaled.max(1.0e-6).sqrt()
        } else if scaled < 0.0 {
            -(-scaled).max(1.0e-6).sqrt()
        } else {
            0.0
        };
        let gate = sigmoid(signed_root);
        let value_offset = token as usize * hidden as usize;
        col = lane;
        while col < hidden {
            unsafe {
                gated
                    .add(stream_offset + col as usize)
                    .write(gate * *value.add(value_offset + col as usize))
            };
            col += thread::blockDim_x();
        }
    }

    /// Applies the Qwen3.8 Flash Next PLE convolution and updates its state.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_ple_conv_update_f32(
        normalized: *const f32,
        gated: *const f32,
        weight_bf16: *const u16,
        state: *mut f32,
        output: *mut f32,
        tokens: u32,
        channels: u32,
        kernel: u32,
        dilation: u32,
        history: u32,
    ) {
        let channel = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if channel >= channels {
            return;
        }
        let state_offset = channel as usize * history as usize;
        let weight_offset = channel as usize * kernel as usize;
        let mut token = 0;
        while token < tokens {
            let mut conv = 0.0f32;
            let mut tap = 0;
            while tap < kernel {
                let lag = (kernel - 1 - tap) * dilation;
                let centre = history + token;
                let source = centre - lag;
                let x = if source < history {
                    unsafe { *state.add(state_offset + source as usize) }
                } else {
                    unsafe { *normalized.add(((source - history) * channels + channel) as usize) }
                };
                conv = x.mul_add(
                    bf16_to_f32(unsafe { *weight_bf16.add(weight_offset + tap as usize) }),
                    conv,
                );
                tap += 1;
            }
            let index = (token * channels + channel) as usize;
            unsafe {
                output
                    .add(index)
                    .write(*gated.add(index) + conv * sigmoid(conv))
            };
            token += 1;
        }
        let mut position = 0;
        while position < history {
            let source = tokens + position;
            let next = if source < history {
                unsafe { *state.add(state_offset + source as usize) }
            } else {
                unsafe { *normalized.add(((source - history) * channels + channel) as usize) }
            };
            unsafe { state.add(state_offset + position as usize).write(next) };
            position += 1;
        }
    }

    /// Clears Qwen3.8 Flash Next QSA micro-block and tile masks.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_qsa_clear_masks(
        selected_blocks: *mut u8,
        selected_tiles: *mut u8,
        max_blocks: u32,
        max_tiles: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < max_blocks {
            unsafe { selected_blocks.add(index as usize).write(0) };
        }
        if index < max_tiles {
            unsafe { selected_tiles.add(index as usize).write(0) };
        }
    }

    /// Prepares normalized, rotary Qwen3.8 Flash Next QSA queries.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn qwen38_qsa_prepare_query_f32(
        projection: *const f32,
        q_norm: *const f32,
        query: *mut f32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        eps: f32,
        theta: f32,
    ) {
        static mut VALUES: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let head = thread::blockIdx_x();
        let dim = thread::threadIdx_x();
        let index = (head * head_dim + dim) as usize;
        let value = unsafe { *projection.add(index) };
        unsafe {
            values.add(dim as usize).write(value);
            reduction.add(dim as usize).write(value * value);
        }
        thread::sync_threads();
        let mut stride = head_dim / 2;
        while stride != 0 {
            if dim < stride {
                unsafe { *reduction.add(dim as usize) += *reduction.add((dim + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_rms = 1.0 / (unsafe { *reduction } / head_dim as f32 + eps).sqrt();
        unsafe { *values.add(dim as usize) = value * inverse_rms * *q_norm.add(dim as usize) };
        thread::sync_threads();
        unsafe {
            query
                .add(index)
                .write(qwen38_rope_value(values, dim, rotary_dim, position, theta))
        };
    }

    /// Appends the raw BF16 QSA index key for one Qwen3.8 Flash Next token.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn qwen38_qsa_append_key_f32(
        projection: *const f32,
        key_pool_bf16: *mut u16,
        slot: u32,
        page_offset: u32,
        page_tokens: u32,
        heads: u32,
        head_dim: u32,
    ) {
        let dim = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if dim < head_dim {
            let source = (heads * head_dim + dim) as usize;
            let destination = ((slot * page_tokens + page_offset) * head_dim + dim) as usize;
            let encoded = convert::cvt_bf16x2_f32(unsafe { *projection.add(source) }, 0.0) as u16;
            unsafe { key_pool_bf16.add(destination).write(encoded) };
        }
    }

    /// Scores complete four-token QSA micro-blocks.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn qwen38_qsa_score_blocks_f32(
        query: *const f32,
        key_pool_bf16: *const u16,
        page_table: *const u32,
        k_norm: *const f32,
        scores: *mut f32,
        page_tokens: u32,
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        eps: f32,
        theta: f32,
    ) {
        static mut VALUES: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let block = thread::blockIdx_x();
        let dim = thread::threadIdx_x();
        let token = block * 4;
        let page_slot = unsafe { *page_table.add((token / page_tokens) as usize) };
        let page_offset = token % page_tokens;
        let page_base = page_slot as usize * page_tokens as usize * head_dim as usize;
        let mut pooled = 0.0f32;
        let mut row = 0;
        while row < 4 {
            pooled += bf16_to_f32(unsafe {
                *key_pool_bf16.add(
                    page_base + (page_offset + row) as usize * head_dim as usize + dim as usize,
                )
            });
            row += 1;
        }
        pooled = round_to_bf16(pooled * 0.25);
        unsafe {
            values.add(dim as usize).write(pooled);
            reduction.add(dim as usize).write(pooled * pooled);
        }
        thread::sync_threads();
        let mut stride = head_dim / 2;
        while stride != 0 {
            if dim < stride {
                unsafe { *reduction.add(dim as usize) += *reduction.add((dim + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_rms = 1.0 / (unsafe { *reduction } / head_dim as f32 + eps).sqrt();
        unsafe { *values.add(dim as usize) = pooled * inverse_rms * *k_norm.add(dim as usize) };
        thread::sync_threads();
        let key = unsafe { qwen38_rope_value(values, dim, rotary_dim, token, theta) };
        let mut score = 0.0f32;
        let mut head = 0;
        while head < heads {
            let dot = unsafe { *query.add((head * head_dim + dim) as usize) } * key;
            unsafe { reduction.add(dim as usize).write(dot) };
            thread::sync_threads();
            stride = head_dim / 2;
            while stride != 0 {
                if dim < stride {
                    unsafe {
                        *reduction.add(dim as usize) += *reduction.add((dim + stride) as usize)
                    };
                }
                thread::sync_threads();
                stride /= 2;
            }
            if dim == 0 {
                score += unsafe { *reduction }.max(0.0);
            }
            thread::sync_threads();
            head += 1;
        }
        if dim == 0 {
            unsafe {
                scores
                    .add(block as usize)
                    .write(score / (head_dim as f32).sqrt())
            };
        }
    }

    /// Selects the highest-scoring QSA micro-blocks with stable index ties.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_qsa_select_blocks_f32(
        scores: *const f32,
        selected_blocks: *mut u8,
        complete_blocks: u32,
        selected_complete_blocks: u32,
        tail_tokens: u32,
    ) {
        static mut HISTOGRAM: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut PREFIX: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut RANK: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut COUNT: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut TIE_CUTOFF: SharedArray<u32, 1> = SharedArray::UNINIT;
        let histogram = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HISTOGRAM) };
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let prefix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PREFIX) };
        let rank = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RANK) };
        let count = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COUNT) };
        let tie_cutoff = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TIE_CUTOFF) };
        let lane = thread::threadIdx_x();
        if complete_blocks <= selected_complete_blocks {
            let mut block = lane;
            while block < complete_blocks {
                unsafe { selected_blocks.add(block as usize).write(1) };
                block += thread::blockDim_x();
            }
            if tail_tokens != 0 && lane == 0 {
                unsafe { selected_blocks.add(complete_blocks as usize).write(1) };
            }
            return;
        }
        if lane == 0 {
            unsafe {
                prefix.write(0);
                rank.write(selected_complete_blocks);
            }
        }
        thread::sync_threads();
        let mut shift = 24u32;
        loop {
            unsafe { histogram.add(lane as usize).write(0) };
            thread::sync_threads();
            let higher_mask = if shift == 24 {
                0
            } else {
                !((1u32 << (shift + 8)) - 1)
            };
            let current_prefix = unsafe { *prefix };
            let mut block = lane;
            while block < complete_blocks {
                let score = unsafe { *scores.add(block as usize) };
                let bits = if score.is_finite() {
                    score.max(0.0)
                } else {
                    0.0
                }
                .to_bits();
                if bits & higher_mask == current_prefix {
                    let bucket = ((bits >> shift) & 0xff) as usize;
                    unsafe {
                        BlockAtomicU32::from_ptr(histogram.add(bucket))
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    }
                }
                block += thread::blockDim_x();
            }
            thread::sync_threads();
            if lane == 0 {
                let mut higher = 0u32;
                let mut byte = 255u32;
                loop {
                    let next = higher + unsafe { *histogram.add(byte as usize) };
                    if unsafe { *rank } <= next {
                        unsafe {
                            *prefix |= byte << shift;
                            *rank -= higher;
                        }
                        break;
                    }
                    higher = next;
                    if byte == 0 {
                        break;
                    }
                    byte -= 1;
                }
            }
            thread::sync_threads();
            if shift == 0 {
                break;
            }
            shift -= 8;
        }
        let selected_bits = unsafe { *prefix };
        let mut local_greater = 0u32;
        let mut block = lane;
        while block < complete_blocks {
            let score = unsafe { *scores.add(block as usize) };
            let bits = if score.is_finite() {
                score.max(0.0)
            } else {
                0.0
            }
            .to_bits();
            local_greater += u32::from(bits > selected_bits);
            block += thread::blockDim_x();
        }
        unsafe { reduction.add(lane as usize).write(local_greater) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe {
                    *reduction.add(lane as usize) += *reduction.add((lane + stride) as usize)
                };
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            unsafe { count.write(selected_complete_blocks - *reduction) };
            let mut seen = 0u32;
            let mut candidate = 0u32;
            while candidate < complete_blocks {
                let score = unsafe { *scores.add(candidate as usize) };
                let bits = if score.is_finite() {
                    score.max(0.0)
                } else {
                    0.0
                }
                .to_bits();
                if bits == selected_bits {
                    seen += 1;
                    if seen == unsafe { *count } {
                        unsafe { tie_cutoff.write(candidate) };
                        break;
                    }
                }
                candidate += 1;
            }
        }
        thread::sync_threads();
        block = lane;
        while block < complete_blocks {
            let score = unsafe { *scores.add(block as usize) };
            let bits = if score.is_finite() {
                score.max(0.0)
            } else {
                0.0
            }
            .to_bits();
            let selected =
                bits > selected_bits || (bits == selected_bits && block <= unsafe { *tie_cutoff });
            unsafe {
                selected_blocks
                    .add(block as usize)
                    .write(u8::from(selected))
            };
            block += thread::blockDim_x();
        }
        if tail_tokens != 0 && lane == 0 {
            unsafe { selected_blocks.add(complete_blocks as usize).write(1) };
        }
    }

    /// Builds the selected compact-attention tile mask from QSA micro-blocks.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen38_qsa_build_tile_mask(
        selected_blocks: *const u8,
        selected_tiles: *mut u8,
        visible_blocks: u32,
    ) {
        let tile = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let block_start = tile * 16;
        if block_start >= visible_blocks {
            return;
        }
        let block_end = (block_start + 16).min(visible_blocks);
        let mut selected = false;
        let mut block = block_start;
        while block < block_end {
            selected |= unsafe { *selected_blocks.add(block as usize) } != 0;
            block += 1;
        }
        unsafe { selected_tiles.add(tile as usize).write(u8::from(selected)) };
    }
}
