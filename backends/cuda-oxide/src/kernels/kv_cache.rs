//! Compact SM121 KV cache and attention kernels.

use crate::common::*;
use cuda_device::{
    SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

/// CUDA entry points for compact KV storage and attention.
#[cuda_module]
mod device {
    use super::*;

    /// Copies one or more dense K/V rows into the compact cache tail.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn sm12x_kv_copy_tail_f32(
        key: *const f32,
        value: *const f32,
        key_tail: *mut f32,
        value_tail: *mut f32,
        position: u32,
        width: u32,
    ) {
        let row = thread::blockIdx_y();
        let column = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if column >= width {
            return;
        }
        let destination = ((position + row) & 15) as usize * width as usize + column as usize;
        let source = row as usize * width as usize + column as usize;
        unsafe {
            key_tail.add(destination).write(*key.add(source));
            value_tail.add(destination).write(*value.add(source));
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn finalize_compact_key(
        key_tail: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        head: u32,
        k_block: u32,
    ) {
        let width = kv_heads * head_dim;
        let tail_start = (position & 15) & !7;
        let token_tiles = max_tokens.div_ceil(8);
        let k_tiles = head_dim / 64;
        let token_tile = position / 8;
        let k_tile = k_block / 4;
        let scale_block = k_block & 3;
        let tile = (head * token_tiles + token_tile) * k_tiles + k_tile;
        let packed = unsafe { key_values.add(tile as usize * 256) };
        let mut token = 0;
        while token < 8 {
            let mut maximum = 0.0f32;
            let mut offset = 0;
            while offset < 16 {
                let index = (tail_start + token) as usize * width as usize
                    + head as usize * head_dim as usize
                    + k_block as usize * 16
                    + offset as usize;
                let value = unsafe { *key_tail.add(index) };
                if value.is_finite() {
                    maximum = maximum.max(value.abs());
                }
                offset += 1;
            }
            let scale_code = if maximum == 0.0 {
                0
            } else {
                ue4m3_code(maximum / 6.0)
            };
            let scale = e4m3_value(scale_code);
            offset = 0;
            while offset < 16 {
                let source = (tail_start + token) as usize * width as usize
                    + head as usize * head_dim as usize
                    + k_block as usize * 16
                    + offset as usize;
                let value = unsafe { *key_tail.add(source) };
                let code = e2m1_code(if scale == 0.0 { 0.0 } else { value / scale });
                let nibble = token * 64 + scale_block * 16 + offset;
                let byte = unsafe { packed.add((nibble / 2) as usize) };
                let previous = unsafe { *byte };
                unsafe {
                    byte.write(if nibble & 1 == 0 {
                        (previous & 0xf0) | code
                    } else {
                        (previous & 0x0f) | (code << 4)
                    })
                };
                offset += 1;
            }
            unsafe {
                key_scales
                    .add(((tile * 8 + token) * 4 + scale_block) as usize)
                    .write(scale_code)
            };
            token += 1;
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn finalize_compact_value(
        value_tail: *const f32,
        value_values: *mut u8,
        value_scales: *mut u8,
        position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        head: u32,
        dim_tile: u32,
    ) {
        let width = kv_heads * head_dim;
        let token_tiles = max_tokens.div_ceil(64);
        let token_tile = position / 64;
        let scale_block = (position & 63) / 16;
        let tile = (head * (head_dim / 8) + dim_tile) * token_tiles + token_tile;
        let packed = unsafe { value_values.add(tile as usize * 256) };
        let mut dim = 0;
        while dim < 8 {
            let mut maximum = 0.0f32;
            let mut token = 0;
            while token < 16 {
                let index = token as usize * width as usize
                    + head as usize * head_dim as usize
                    + dim_tile as usize * 8
                    + dim as usize;
                let value = unsafe { *value_tail.add(index) };
                if value.is_finite() {
                    maximum = maximum.max(value.abs());
                }
                token += 1;
            }
            let scale_code = if maximum == 0.0 {
                0
            } else {
                ue4m3_code(maximum / 6.0)
            };
            let scale = e4m3_value(scale_code);
            token = 0;
            while token < 16 {
                let source = token as usize * width as usize
                    + head as usize * head_dim as usize
                    + dim_tile as usize * 8
                    + dim as usize;
                let value = unsafe { *value_tail.add(source) };
                let code = e2m1_code(if scale == 0.0 { 0.0 } else { value / scale });
                let nibble = dim * 64 + scale_block * 16 + token;
                let byte = unsafe { packed.add((nibble / 2) as usize) };
                let previous = unsafe { *byte };
                unsafe {
                    byte.write(if nibble & 1 == 0 {
                        (previous & 0xf0) | code
                    } else {
                        (previous & 0x0f) | (code << 4)
                    })
                };
                token += 1;
            }
            unsafe {
                value_scales
                    .add(((tile * 8 + dim) * 4 + scale_block) as usize)
                    .write(scale_code)
            };
            dim += 1;
        }
    }

    /// Finalizes complete token-major FP4 key tiles from the cache tail.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 3, coordinates = u32, block = (1, 1, 1))]
    pub unsafe fn sm12x_kv_finalize_key_f32(
        key_tail: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        let position = position + thread::blockIdx_z();
        if position & 7 != 7 {
            return;
        }
        let head = thread::blockIdx_x();
        let k_block = thread::blockIdx_y();
        unsafe {
            finalize_compact_key(
                key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim, head,
                k_block,
            )
        };
    }

    /// Finalizes complete transposed FP4 value tiles from the cache tail.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 3, coordinates = u32, block = (1, 1, 1))]
    pub unsafe fn sm12x_kv_finalize_value_f32(
        value_tail: *const f32,
        value_values: *mut u8,
        value_scales: *mut u8,
        position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        let position = position + thread::blockIdx_z();
        if position & 15 != 15 {
            return;
        }
        let head = thread::blockIdx_x();
        let dim_tile = thread::blockIdx_y();
        unsafe {
            finalize_compact_value(
                value_tail,
                value_values,
                value_scales,
                position,
                max_tokens,
                kv_heads,
                head_dim,
                head,
                dim_tile,
            )
        };
    }

    /// Packs complete eight-token key groups directly from dense prompt rows.
    #[kernel]
    #[launch_bounds(16)]
    #[launch_contract(domain = 3, coordinates = u32, block = (16, 1, 1))]
    pub unsafe fn sm12x_kv_finalize_key_rows_f32(
        key: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        key_output: *mut u16,
        output_tokens: u32,
        input_row_offset: u32,
        start_position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        static mut VALUES: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let scale = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE) };
        let head = thread::blockIdx_x();
        let k_block = thread::blockIdx_y();
        let token_group = thread::blockIdx_z();
        let lane = thread::threadIdx_x();
        let width = kv_heads * head_dim;
        let token_tiles = max_tokens.div_ceil(8);
        let k_tiles = head_dim / 64;
        let token_tile = start_position / 8 + token_group;
        let k_tile = k_block / 4;
        let scale_block = k_block & 3;
        let tile = (head * token_tiles + token_tile) * k_tiles + k_tile;
        let packed = unsafe { key_values.add(tile as usize * 256) };
        let mut token = 0;
        while token < 8 {
            let source = (input_row_offset + token_group * 8 + token) as usize * width as usize
                + head as usize * head_dim as usize
                + k_block as usize * 16
                + lane as usize;
            let value = unsafe { *key.add(source) };
            unsafe {
                values
                    .add(lane as usize)
                    .write(if value.is_finite() { value } else { 0.0 })
            };
            thread::sync_threads();
            if lane == 0 {
                let mut maximum = 0.0f32;
                let mut index = 0;
                while index < 16 {
                    maximum = maximum.max(unsafe { *values.add(index) }.abs());
                    index += 1;
                }
                let scale_code = if maximum == 0.0 {
                    0
                } else {
                    ue4m3_code(maximum / 6.0)
                };
                unsafe {
                    key_scales
                        .add(((tile * 8 + token) * 4 + scale_block) as usize)
                        .write(scale_code);
                    scale.write(e4m3_value(scale_code));
                }
            }
            thread::sync_threads();
            let scale_value = unsafe { *scale };
            if lane < 8 {
                let first = unsafe { *values.add((lane * 2) as usize) };
                let second = unsafe { *values.add((lane * 2 + 1) as usize) };
                let low = e2m1_code(if scale_value == 0.0 {
                    0.0
                } else {
                    first / scale_value
                });
                let high = e2m1_code(if scale_value == 0.0 {
                    0.0
                } else {
                    second / scale_value
                });
                let nibble = token * 64 + scale_block * 16 + lane * 2;
                unsafe { packed.add((nibble / 2) as usize).write(low | (high << 4)) };
            }
            if !key_output.is_null() {
                let code = e2m1_code(if scale_value == 0.0 {
                    0.0
                } else {
                    (unsafe { *values.add(lane as usize) }) / scale_value
                });
                let bf16 = convert::cvt_bf16x2_f32(e2m1_value(code) * scale_value, 0.0) as u16;
                let output_token = start_position + token_group * 8 + token;
                let output_dim = k_block * 16 + lane;
                let output_index = (head * output_tokens + output_token) as usize
                    * head_dim as usize
                    + output_dim as usize;
                unsafe { key_output.add(output_index).write(bf16) };
            }
            thread::sync_threads();
            token += 1;
        }
    }

    /// Packs complete 16-token value groups directly from dense prompt rows.
    #[kernel]
    #[launch_bounds(16)]
    #[launch_contract(domain = 3, coordinates = u32, block = (16, 1, 1))]
    pub unsafe fn sm12x_kv_finalize_value_rows_f32(
        value: *const f32,
        value_values: *mut u8,
        value_scales: *mut u8,
        value_output: *mut u16,
        output_tokens: u32,
        input_row_offset: u32,
        start_position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        static mut VALUES: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let scale = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE) };
        let head = thread::blockIdx_x();
        let dim_tile = thread::blockIdx_y();
        let token_group = thread::blockIdx_z();
        let lane = thread::threadIdx_x();
        let width = kv_heads * head_dim;
        let position = start_position + token_group * 16;
        let context_tiles = max_tokens.div_ceil(64);
        let token_tile = position / 64;
        let scale_block = (position & 63) / 16;
        let tile = (head * (head_dim / 8) + dim_tile) * context_tiles + token_tile;
        let packed = unsafe { value_values.add(tile as usize * 256) };
        let mut dim = 0;
        while dim < 8 {
            let source = (input_row_offset + token_group * 16 + lane) as usize * width as usize
                + head as usize * head_dim as usize
                + dim_tile as usize * 8
                + dim as usize;
            let value = unsafe { *value.add(source) };
            unsafe {
                values
                    .add(lane as usize)
                    .write(if value.is_finite() { value } else { 0.0 })
            };
            thread::sync_threads();
            if lane == 0 {
                let mut maximum = 0.0f32;
                let mut index = 0;
                while index < 16 {
                    maximum = maximum.max(unsafe { *values.add(index) }.abs());
                    index += 1;
                }
                let scale_code = if maximum == 0.0 {
                    0
                } else {
                    ue4m3_code(maximum / 6.0)
                };
                unsafe {
                    value_scales
                        .add(((tile * 8 + dim) * 4 + scale_block) as usize)
                        .write(scale_code);
                    scale.write(e4m3_value(scale_code));
                }
            }
            thread::sync_threads();
            let scale_value = unsafe { *scale };
            if lane < 8 {
                let first = unsafe { *values.add((lane * 2) as usize) };
                let second = unsafe { *values.add((lane * 2 + 1) as usize) };
                let low = e2m1_code(if scale_value == 0.0 {
                    0.0
                } else {
                    first / scale_value
                });
                let high = e2m1_code(if scale_value == 0.0 {
                    0.0
                } else {
                    second / scale_value
                });
                let nibble = dim * 64 + scale_block * 16 + lane * 2;
                unsafe { packed.add((nibble / 2) as usize).write(low | (high << 4)) };
            }
            if !value_output.is_null() {
                let code = e2m1_code(if scale_value == 0.0 {
                    0.0
                } else {
                    (unsafe { *values.add(lane as usize) }) / scale_value
                });
                let bf16 = convert::cvt_bf16x2_f32(e2m1_value(code) * scale_value, 0.0) as u16;
                let output_token = position + lane;
                let output_dim = dim_tile * 8 + dim;
                let output_index = (head * head_dim + output_dim) as usize * output_tokens as usize
                    + output_token as usize;
                unsafe { value_output.add(output_index).write(bf16) };
            }
            thread::sync_threads();
            dim += 1;
        }
    }

    /// Stages unquantized prompt-tail rows in the BF16 attention layouts.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn sm12x_kv_stage_tail_bf16(
        key: *const f32,
        value: *const f32,
        key_output: *mut u16,
        value_output: *mut u16,
        input_row_offset: u32,
        output_row_offset: u32,
        rows: u32,
        output_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let width = kv_heads * head_dim;
        if index >= rows * width {
            return;
        }
        let dim = index % head_dim;
        let head = (index / head_dim) % kv_heads;
        let row = index / width;
        let input_index = (input_row_offset + row) as usize * width as usize
            + head as usize * head_dim as usize
            + dim as usize;
        let output_token = output_row_offset + row;
        let key_bf16 = convert::cvt_bf16x2_f32(unsafe { *key.add(input_index) }, 0.0) as u16;
        let value_bf16 = convert::cvt_bf16x2_f32(unsafe { *value.add(input_index) }, 0.0) as u16;
        unsafe {
            key_output
                .add(((head * output_tokens + output_token) * head_dim + dim) as usize)
                .write(key_bf16);
            value_output
                .add(((head * head_dim + dim) * output_tokens + output_token) as usize)
                .write(value_bf16);
        }
    }

    /// Quantizes grouped attention queries into native SM121 FP4 tiles.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 3, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn sm12x_kv_quantize_query_f32(
        query: *const f32,
        query_tiles: *mut u8,
        query_scales: *mut u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        input_row_offset: u32,
    ) {
        static mut SCALE_CODES: SharedArray<u8, 32> = SharedArray::UNINIT;
        let scale_codes = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE_CODES) };
        let batch_row = thread::blockIdx_z();
        let group = thread::blockIdx_x();
        let k_tile = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        let queries_per_kv = q_heads / kv_heads;
        let query_tiles_per_kv = queries_per_kv.div_ceil(8);
        let query_groups = kv_heads * query_tiles_per_kv;
        let head_k_tiles = head_dim / 64;
        let query =
            unsafe { query.add(((input_row_offset + batch_row) * q_heads * head_dim) as usize) };
        let query_tiles =
            unsafe { query_tiles.add((batch_row * query_groups * head_k_tiles * 512) as usize) };
        let query_scales =
            unsafe { query_scales.add((batch_row * query_groups * head_k_tiles * 8) as usize) };
        let kv_head = group / query_tiles_per_kv;
        let query_base = kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
        let tile = unsafe { query_tiles.add(((group * head_k_tiles + k_tile) * 512) as usize) };
        if lane < 32 {
            let row = lane / 4;
            let k_block = lane & 3;
            let q_head = query_base + row;
            let mut maximum = 0.0f32;
            let mut offset = 0;
            while offset < 16 {
                let value = if q_head < (kv_head + 1) * queries_per_kv {
                    unsafe {
                        *query
                            .add((q_head * head_dim + k_tile * 64 + k_block * 16 + offset) as usize)
                    }
                } else {
                    0.0
                };
                if value.is_finite() {
                    maximum = maximum.max(value.abs());
                }
                offset += 1;
            }
            unsafe {
                scale_codes.add(lane as usize).write(if maximum == 0.0 {
                    0
                } else {
                    ue4m3_code(maximum / 6.0)
                })
            };
        }
        thread::sync_threads();
        let mut byte = lane;
        while byte < 512 {
            let mut packed = 0;
            let mut nibble = 0;
            while nibble < 2 {
                let index = byte * 2 + nibble;
                let fragment_lane = index / 32;
                let value_index = index & 31;
                let t0 = fragment_lane & 3;
                let t1 = fragment_lane >> 2;
                let v0 = value_index & 7;
                let v1 = (value_index >> 3) & 1;
                let v2 = (value_index >> 4) & 1;
                let row = t1 + 8 * v1;
                let col = t0 * 8 + v0 + 32 * v2;
                let q_head = query_base + row;
                let value = if row < 8 && q_head < (kv_head + 1) * queries_per_kv {
                    unsafe { *query.add((q_head * head_dim + k_tile * 64 + col) as usize) }
                } else {
                    0.0
                };
                let scale = if row < 8 {
                    e4m3_value(unsafe { *scale_codes.add((row * 4 + col / 16) as usize) })
                } else {
                    0.0
                };
                packed |= e2m1_code(if scale == 0.0 { 0.0 } else { value / scale }) << (nibble * 4);
                nibble += 1;
            }
            unsafe { tile.add(byte as usize).write(packed) };
            byte += thread::blockDim_x();
        }
        if lane < 8 {
            let tile_index = group * head_k_tiles + k_tile;
            unsafe {
                query_scales
                    .add((tile_index * 8 + lane) as usize)
                    .write(scale_word(scale_codes.add((lane * 4) as usize)))
            };
        }
    }

    /// Computes FP4 query-key scores against a contiguous compact cache.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 1, 1))]
    pub unsafe fn sm12x_kv_qk_f32(
        query_tiles: *const u8,
        query_scales: *const u32,
        key_values: *const u8,
        key_scales: *const u8,
        key_tail: *const f32,
        scores: *mut f32,
        cache_len: u32,
        cache_len_device: *const u32,
        window_start: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page_table: *const u32,
        page_tokens: u32,
        page_stride_bytes: u32,
        selected_blocks: *const u8,
        causal_start_position: u32,
        window_tokens: u32,
    ) {
        static mut KEY_TILE: SharedArray<u8, 512> = SharedArray::UNINIT;
        let key_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEY_TILE) };
        let batch_row = thread::blockIdx_z();
        let mut cache_len = if cache_len_device.is_null() {
            cache_len
        } else {
            unsafe { *cache_len_device }
        };
        let mut window_start = window_start;
        if causal_start_position != u32::MAX {
            cache_len = causal_start_position + batch_row + 1;
            window_start = if window_tokens == 0 || cache_len <= window_tokens {
                0
            } else {
                cache_len - window_tokens
            };
        }
        let group = thread::blockIdx_x();
        let token_tile = thread::blockIdx_y();
        if cache_len == 0 || cache_len > max_tokens || token_tile >= cache_len.div_ceil(8) {
            return;
        }
        if token_tile * 8 + 7 < window_start {
            return;
        }
        if !selected_blocks.is_null() {
            let first_block = token_tile * 2;
            let first_selected = unsafe { *selected_blocks.add(first_block as usize) } != 0;
            let second_selected = token_tile * 8 + 4 < cache_len
                && unsafe { *selected_blocks.add((first_block + 1) as usize) } != 0;
            if !first_selected && !second_selected {
                return;
            }
        }
        let lane = thread::threadIdx_x();
        let queries_per_kv = q_heads / kv_heads;
        let query_tiles_per_kv = queries_per_kv.div_ceil(8);
        let query_groups = kv_heads * query_tiles_per_kv;
        let kv_head = group / query_tiles_per_kv;
        let query_base = kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
        let complete_tiles = cache_len / 8;
        let compact = token_tile < complete_tiles;
        let tail_len = cache_len & 7;
        let head_k_tiles = head_dim / 64;
        let query_tiles =
            unsafe { query_tiles.add((batch_row * query_groups * head_k_tiles * 512) as usize) };
        let query_scales =
            unsafe { query_scales.add((batch_row * query_groups * head_k_tiles * 8) as usize) };
        let scores = unsafe { scores.add((batch_row * q_heads * max_tokens) as usize) };
        let max_token_tiles = max_tokens.div_ceil(8);
        let logical_token = token_tile * 8;
        let page_slot = if page_table.is_null() {
            0
        } else {
            unsafe { *page_table.add((logical_token / page_tokens) as usize) }
        };
        let storage_token_tile = if page_table.is_null() {
            token_tile
        } else {
            (logical_token % page_tokens) / 8
        };
        let storage_token_tiles = if page_table.is_null() {
            max_token_tiles
        } else {
            page_tokens / 8
        };
        let page_key_values =
            unsafe { key_values.add(page_slot as usize * page_stride_bytes as usize) };
        let page_key_scales =
            unsafe { key_scales.add(page_slot as usize * page_stride_bytes as usize) };
        let tail_page_slot = if page_table.is_null() || cache_len == 0 {
            0
        } else {
            unsafe { *page_table.add(((cache_len - 1) / page_tokens) as usize) }
        };
        let page_key_tail = unsafe {
            key_tail
                .cast::<u8>()
                .add(tail_page_slot as usize * page_stride_bytes as usize)
                .cast::<f32>()
        };
        let width = kv_heads * head_dim;
        let tail_start = (complete_tiles * 8) & 15;
        let mut accumulators = [0.0f32; 4];
        let row = lane >> 2;
        let t0 = lane & 3;
        let mut k_tile = 0;
        while k_tile < head_k_tiles {
            let query_tile =
                unsafe { query_tiles.add(((group * head_k_tiles + k_tile) * 512) as usize) };
            let mut index = lane;
            while index < 512 {
                unsafe { key_tile.add(index as usize).write(0) };
                index += 32;
            }
            let compact_tile_index =
                (kv_head * storage_token_tiles + storage_token_tile) * head_k_tiles + k_tile;
            let compact_tile = unsafe { page_key_values.add(compact_tile_index as usize * 256) };
            thread::sync_threads();

            let mut tail_scale_codes = [0u8; 4];
            let mut tail_scales = [0.0f32; 4];
            if !compact && row < tail_len {
                let mut k_block = 0;
                while k_block < 4 {
                    let mut maximum = 0.0f32;
                    let mut offset = 0;
                    while offset < 16 {
                        let source = (tail_start + row) as usize * width as usize
                            + kv_head as usize * head_dim as usize
                            + k_tile as usize * 64
                            + k_block as usize * 16
                            + offset as usize;
                        let value = unsafe { *page_key_tail.add(source) };
                        if value.is_finite() {
                            maximum = maximum.max(value.abs());
                        }
                        offset += 1;
                    }
                    tail_scale_codes[k_block as usize] = if maximum == 0.0 {
                        0
                    } else {
                        ue4m3_code(maximum / 6.0)
                    };
                    tail_scales[k_block as usize] = e4m3_value(tail_scale_codes[k_block as usize]);
                    k_block += 1;
                }
            }
            let mut pair = 0;
            while pair < 8 {
                let mut packed = 0;
                let mut half = 0;
                while half < 2 {
                    let v = pair * 2 + half;
                    let col = t0 * 8 + (v & 7) + 32 * ((v >> 3) & 1);
                    let code = if compact {
                        let nibble = row * 64 + col;
                        let byte = unsafe { *compact_tile.add((nibble / 2) as usize) };
                        if nibble & 1 == 0 {
                            byte & 0x0f
                        } else {
                            byte >> 4
                        }
                    } else if row < tail_len {
                        let source = (tail_start + row) as usize * width as usize
                            + kv_head as usize * head_dim as usize
                            + k_tile as usize * 64
                            + col as usize;
                        let value = unsafe { *page_key_tail.add(source) };
                        let scale = tail_scales[(col / 16) as usize];
                        e2m1_code(if scale == 0.0 { 0.0 } else { value / scale })
                    } else {
                        0
                    };
                    packed |= code << (half * 4);
                    half += 1;
                }
                unsafe { key_tile.add((lane * 16 + pair) as usize).write(packed) };
                pair += 1;
            }
            thread::sync_threads();
            let query_lane = unsafe { query_tile.add(lane as usize * 16) };
            let key_lane = unsafe { key_tile.add(lane as usize * 16) };
            let a = unsafe {
                [
                    load_u32(query_lane, 0),
                    load_u32(query_lane, 1),
                    load_u32(query_lane, 2),
                    load_u32(query_lane, 3),
                ]
            };
            let b = unsafe { [load_u32(key_lane, 0), load_u32(key_lane, 1)] };
            let scale_a = if lane & 3 == 0 {
                unsafe { *query_scales.add(((group * head_k_tiles + k_tile) * 8 + row) as usize) }
            } else {
                0
            };
            let scale_b = if compact {
                unsafe {
                    scale_word(page_key_scales.add(((compact_tile_index * 8 + row) * 4) as usize))
                }
            } else {
                u32::from(tail_scale_codes[0])
                    | (u32::from(tail_scale_codes[1]) << 8)
                    | (u32::from(tail_scale_codes[2]) << 16)
                    | (u32::from(tail_scale_codes[3]) << 24)
            };
            accumulators = unsafe { mma_m16n8k64_nvfp4(a, b, scale_a, scale_b, accumulators) };
            thread::sync_threads();
            k_tile += 1;
        }
        let output_row = lane >> 2;
        let output_col = (lane & 3) * 2;
        let q_head = query_base + output_row;
        let token = token_tile * 8 + output_col;
        let scale = cuda_device::float::rsqrt_approx_f32(head_dim as f32);
        if q_head < (kv_head + 1) * queries_per_kv {
            if token >= window_start
                && token < cache_len
                && (selected_blocks.is_null()
                    || unsafe { *selected_blocks.add((token / 4) as usize) } != 0)
            {
                unsafe {
                    scores
                        .add((q_head * max_tokens + token) as usize)
                        .write(accumulators[0] * scale)
                };
            }
            if token + 1 >= window_start
                && token + 1 < cache_len
                && (selected_blocks.is_null()
                    || unsafe { *selected_blocks.add(((token + 1) / 4) as usize) } != 0)
            {
                unsafe {
                    scores
                        .add((q_head * max_tokens + token + 1) as usize)
                        .write(accumulators[1] * scale)
                };
            }
        }
    }

    /// Applies stable row-wise softmax to compact-attention scores.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn sm12x_kv_softmax_f32(
        scores: *mut f32,
        cache_len: u32,
        cache_len_device: *const u32,
        window_start: u32,
        max_tokens: u32,
        q_heads: u32,
        selected_blocks: *const u8,
        causal_start_position: u32,
        window_tokens: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let batch_row = thread::blockIdx_y();
        let mut cache_len = if cache_len_device.is_null() {
            cache_len
        } else {
            unsafe { *cache_len_device }
        };
        let mut window_start = window_start;
        if causal_start_position != u32::MAX {
            cache_len = causal_start_position + batch_row + 1;
            window_start = if window_tokens == 0 || cache_len <= window_tokens {
                0
            } else {
                cache_len - window_tokens
            };
        }
        if cache_len == 0 || cache_len > max_tokens {
            return;
        }
        let head = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        let row = unsafe { scores.add(((batch_row * q_heads + head) * max_tokens) as usize) };
        let mut maximum = f32::NEG_INFINITY;
        let mut token = lane;
        token += window_start;
        while token < cache_len {
            if selected_blocks.is_null()
                || unsafe { *selected_blocks.add((token / 4) as usize) } != 0
            {
                maximum = maximum.max(unsafe { *row.add(token as usize) });
            }
            token += thread::blockDim_x();
        }
        unsafe { partial.add(lane as usize).write(maximum) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe {
                    partial.add(lane as usize).write(
                        (*partial.add(lane as usize)).max(*partial.add((lane + stride) as usize)),
                    )
                };
            }
            thread::sync_threads();
            stride /= 2;
        }
        maximum = unsafe { *partial };
        let mut sum = 0.0f32;
        token = window_start + lane;
        while token < cache_len {
            if selected_blocks.is_null()
                || unsafe { *selected_blocks.add((token / 4) as usize) } != 0
            {
                sum += (unsafe { *row.add(token as usize) } - maximum).exp();
            }
            token += thread::blockDim_x();
        }
        unsafe { partial.add(lane as usize).write(sum) };
        thread::sync_threads();
        stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_sum = 1.0 / unsafe { *partial };
        token = window_start + lane;
        while token < cache_len {
            unsafe {
                let value = row.add(token as usize);
                value.write(
                    if selected_blocks.is_null() || *selected_blocks.add((token / 4) as usize) != 0
                    {
                        (*value - maximum).exp() * inverse_sum
                    } else {
                        0.0
                    },
                );
            }
            token += thread::blockDim_x();
        }
        let _ = q_heads;
    }

    /// Quantizes softmax probabilities into native SM121 FP4 tiles.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn sm12x_kv_quantize_probability_f32(
        scores: *const f32,
        probability_tiles: *mut u8,
        probability_scales: *mut u32,
        cache_len: u32,
        cache_len_device: *const u32,
        window_start: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        selected_blocks: *const u8,
        selected_tokens: u32,
        causal_start_position: u32,
        window_tokens: u32,
    ) {
        static mut SCALE_CODES: SharedArray<u8, 32> = SharedArray::UNINIT;
        let scale_codes = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE_CODES) };
        let batch_row = thread::blockIdx_z();
        let mut cache_len = if cache_len_device.is_null() {
            cache_len
        } else {
            unsafe { *cache_len_device }
        };
        let mut window_start = window_start;
        if causal_start_position != u32::MAX {
            cache_len = causal_start_position + batch_row + 1;
            window_start = if window_tokens == 0 || cache_len <= window_tokens {
                0
            } else {
                cache_len - window_tokens
            };
        }
        let group = thread::blockIdx_x();
        let context_tile = thread::blockIdx_y();
        if cache_len == 0 || cache_len > max_tokens || context_tile >= cache_len.div_ceil(64) {
            return;
        }
        if context_tile * 64 + 63 < window_start {
            return;
        }
        let lane = thread::threadIdx_x();
        let queries_per_kv = q_heads / kv_heads;
        let query_tiles_per_kv = queries_per_kv.div_ceil(8);
        let query_groups = kv_heads * query_tiles_per_kv;
        let kv_head = group / query_tiles_per_kv;
        let query_base = kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
        let max_context_tiles = max_tokens.div_ceil(64);
        let scores = unsafe { scores.add((batch_row * q_heads * max_tokens) as usize) };
        let probability_tiles = unsafe {
            probability_tiles.add((batch_row * query_groups * max_context_tiles * 512) as usize)
        };
        let probability_scales = unsafe {
            probability_scales.add((batch_row * query_groups * max_context_tiles * 8) as usize)
        };
        let amplification = probability_amplification(if selected_blocks.is_null() {
            cache_len - window_start
        } else {
            selected_tokens
        });
        let tile = unsafe {
            probability_tiles.add(((group * max_context_tiles + context_tile) * 512) as usize)
        };
        if lane < 32 {
            let row = lane / 4;
            let k_block = lane & 3;
            let q_head = query_base + row;
            let mut maximum = 0.0f32;
            let mut offset = 0;
            while offset < 16 {
                let token = context_tile * 64 + k_block * 16 + offset;
                if token >= window_start
                    && token < cache_len
                    && q_head < (kv_head + 1) * queries_per_kv
                    && (selected_blocks.is_null()
                        || unsafe { *selected_blocks.add((token / 4) as usize) } != 0)
                {
                    maximum =
                        maximum.max(unsafe { *scores.add((q_head * max_tokens + token) as usize) });
                }
                offset += 1;
            }
            unsafe {
                scale_codes.add(lane as usize).write(if maximum == 0.0 {
                    0
                } else {
                    ue4m3_code(maximum * amplification / 6.0)
                })
            };
        }
        thread::sync_threads();
        let mut byte = lane;
        while byte < 512 {
            let mut packed = 0;
            let mut nibble = 0;
            while nibble < 2 {
                let index = byte * 2 + nibble;
                let fragment_lane = index / 32;
                let value_index = index & 31;
                let t0 = fragment_lane & 3;
                let t1 = fragment_lane >> 2;
                let v0 = value_index & 7;
                let v1 = (value_index >> 3) & 1;
                let v2 = (value_index >> 4) & 1;
                let row = t1 + 8 * v1;
                let col = t0 * 8 + v0 + 32 * v2;
                let token = context_tile * 64 + col;
                let q_head = query_base + row;
                let value = if row < 8
                    && q_head < (kv_head + 1) * queries_per_kv
                    && token >= window_start
                    && token < cache_len
                    && (selected_blocks.is_null()
                        || unsafe { *selected_blocks.add((token / 4) as usize) } != 0)
                {
                    unsafe { *scores.add((q_head * max_tokens + token) as usize) }
                } else {
                    0.0
                };
                let scale = if row < 8 {
                    e4m3_value(unsafe { *scale_codes.add((row * 4 + col / 16) as usize) })
                } else {
                    0.0
                };
                packed |= e2m1_code(if scale == 0.0 {
                    0.0
                } else {
                    value * amplification / scale
                }) << (nibble * 4);
                nibble += 1;
            }
            unsafe { tile.add(byte as usize).write(packed) };
            byte += thread::blockDim_x();
        }
        if lane < 8 {
            unsafe {
                probability_scales
                    .add(((group * max_context_tiles + context_tile) * 8 + lane) as usize)
                    .write(scale_word(scale_codes.add((lane * 4) as usize)))
            };
        }
    }

    /// Computes FP4 probability-value products from a compact cache.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 3, coordinates = u32, block = (32, 1, 1))]
    pub unsafe fn sm12x_kv_pv_f32(
        probability_tiles: *const u8,
        probability_scales: *const u32,
        value_values: *const u8,
        value_scales: *const u8,
        value_tail: *const f32,
        output: *mut f32,
        partial_output: *mut f32,
        cache_len: u32,
        cache_len_device: *const u32,
        window_start: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        pv_splits: u32,
        page_table: *const u32,
        page_tokens: u32,
        page_stride_bytes: u32,
        selected_tiles: *const u8,
        selected_tokens: u32,
        causal_start_position: u32,
        window_tokens: u32,
        output_row_offset: u32,
    ) {
        static mut VALUE_TILE: SharedArray<u8, 512> = SharedArray::UNINIT;
        let value_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUE_TILE) };
        let batch_row = thread::blockIdx_z() / pv_splits;
        let split = thread::blockIdx_z() % pv_splits;
        let mut cache_len = if cache_len_device.is_null() {
            cache_len
        } else {
            unsafe { *cache_len_device }
        };
        let mut window_start = window_start;
        if causal_start_position != u32::MAX {
            cache_len = causal_start_position + batch_row + 1;
            window_start = if window_tokens == 0 || cache_len <= window_tokens {
                0
            } else {
                cache_len - window_tokens
            };
        }
        if cache_len == 0 || cache_len > max_tokens {
            return;
        }
        let group = thread::blockIdx_x();
        let dim_tile = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        let queries_per_kv = q_heads / kv_heads;
        let query_tiles_per_kv = queries_per_kv.div_ceil(8);
        let query_groups = kv_heads * query_tiles_per_kv;
        let kv_head = group / query_tiles_per_kv;
        let query_base = kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
        let context_tiles = cache_len.div_ceil(64);
        let max_context_tiles = max_tokens.div_ceil(64);
        let probability_tiles = unsafe {
            probability_tiles.add((batch_row * query_groups * max_context_tiles * 512) as usize)
        };
        let probability_scales = unsafe {
            probability_scales.add((batch_row * query_groups * max_context_tiles * 8) as usize)
        };
        let destination = if pv_splits == 1 {
            unsafe { output.add(((output_row_offset + batch_row) * q_heads * head_dim) as usize) }
        } else {
            unsafe {
                partial_output.add(((batch_row * pv_splits + split) * q_heads * head_dim) as usize)
            }
        };
        let full_tokens = cache_len / 16 * 16;
        let tail_len = cache_len & 15;
        let width = kv_heads * head_dim;
        let correction = probability_amplification(if selected_tiles.is_null() {
            cache_len - window_start
        } else {
            selected_tokens
        });
        let first_context_tile = window_start / 64;
        let active_context_tiles = context_tiles - first_context_tile;
        let context_begin = first_context_tile + active_context_tiles * split / pv_splits;
        let context_end = first_context_tile + active_context_tiles * (split + 1) / pv_splits;
        let dim = lane >> 2;
        let t0 = lane & 3;
        let mut accumulators = [0.0f32; 4];
        let mut context_tile = context_begin;
        while context_tile < context_end {
            if !selected_tiles.is_null()
                && unsafe { *selected_tiles.add(context_tile as usize) } == 0
            {
                context_tile += 1;
                continue;
            }
            let probability_tile = unsafe {
                probability_tiles.add(((group * max_context_tiles + context_tile) * 512) as usize)
            };
            let logical_token = context_tile * 64;
            let page_slot = if page_table.is_null() {
                0
            } else {
                unsafe { *page_table.add((logical_token / page_tokens) as usize) }
            };
            let storage_context_tile = if page_table.is_null() {
                context_tile
            } else {
                (logical_token % page_tokens) / 64
            };
            let storage_context_tiles = if page_table.is_null() {
                max_context_tiles
            } else {
                page_tokens / 64
            };
            let value_tile_index = (kv_head * (head_dim / 8) + dim_tile) * storage_context_tiles
                + storage_context_tile;
            let page_value_values =
                unsafe { value_values.add(page_slot as usize * page_stride_bytes as usize) };
            let page_value_scales =
                unsafe { value_scales.add(page_slot as usize * page_stride_bytes as usize) };
            let tail_page_slot = if page_table.is_null() || cache_len == 0 {
                0
            } else {
                unsafe { *page_table.add(((cache_len - 1) / page_tokens) as usize) }
            };
            let page_value_tail = unsafe {
                value_tail
                    .cast::<u8>()
                    .add(tail_page_slot as usize * page_stride_bytes as usize)
                    .cast::<f32>()
            };
            let compact_tile = unsafe { page_value_values.add(value_tile_index as usize * 256) };
            let mut index = lane;
            while index < 512 {
                unsafe { value_tile.add(index as usize).write(0) };
                index += 32;
            }
            thread::sync_threads();
            let mut scale_codes = [0u8; 4];
            let mut tail_scales = [0.0f32; 4];
            let mut k_block = 0;
            while k_block < 4 {
                let block_start = context_tile * 64 + k_block * 16;
                if block_start + 16 <= full_tokens {
                    scale_codes[k_block as usize] = unsafe {
                        *page_value_scales
                            .add(((value_tile_index * 8 + dim) * 4 + k_block) as usize)
                    };
                } else if block_start == full_tokens && tail_len != 0 {
                    let mut maximum = 0.0f32;
                    let mut token = 0;
                    while token < tail_len {
                        let source = token as usize * width as usize
                            + kv_head as usize * head_dim as usize
                            + dim_tile as usize * 8
                            + dim as usize;
                        let value = unsafe { *page_value_tail.add(source) };
                        if value.is_finite() {
                            maximum = maximum.max(value.abs());
                        }
                        token += 1;
                    }
                    scale_codes[k_block as usize] = if maximum == 0.0 {
                        0
                    } else {
                        ue4m3_code(maximum / 6.0)
                    };
                    tail_scales[k_block as usize] = e4m3_value(scale_codes[k_block as usize]);
                }
                k_block += 1;
            }
            let mut pair = 0;
            while pair < 8 {
                let mut packed = 0;
                let mut half = 0;
                while half < 2 {
                    let v = pair * 2 + half;
                    let col = t0 * 8 + (v & 7) + 32 * ((v >> 3) & 1);
                    let token = context_tile * 64 + col;
                    let code = if token < full_tokens {
                        let nibble = dim * 64 + col;
                        let byte = unsafe { *compact_tile.add((nibble / 2) as usize) };
                        if nibble & 1 == 0 {
                            byte & 0x0f
                        } else {
                            byte >> 4
                        }
                    } else if token < cache_len {
                        let source = (token - full_tokens) as usize * width as usize
                            + kv_head as usize * head_dim as usize
                            + dim_tile as usize * 8
                            + dim as usize;
                        let value = unsafe { *page_value_tail.add(source) };
                        let scale = tail_scales[(col / 16) as usize];
                        e2m1_code(if scale == 0.0 { 0.0 } else { value / scale })
                    } else {
                        0
                    };
                    packed |= code << (half * 4);
                    half += 1;
                }
                unsafe { value_tile.add((lane * 16 + pair) as usize).write(packed) };
                pair += 1;
            }
            thread::sync_threads();
            let probability_lane = unsafe { probability_tile.add(lane as usize * 16) };
            let value_lane = unsafe { value_tile.add(lane as usize * 16) };
            let a = unsafe {
                [
                    load_u32(probability_lane, 0),
                    load_u32(probability_lane, 1),
                    load_u32(probability_lane, 2),
                    load_u32(probability_lane, 3),
                ]
            };
            let b = unsafe { [load_u32(value_lane, 0), load_u32(value_lane, 1)] };
            let scale_a = if lane & 3 == 0 {
                unsafe {
                    *probability_scales
                        .add(((group * max_context_tiles + context_tile) * 8 + dim) as usize)
                }
            } else {
                0
            };
            let scale_b = u32::from(scale_codes[0])
                | (u32::from(scale_codes[1]) << 8)
                | (u32::from(scale_codes[2]) << 16)
                | (u32::from(scale_codes[3]) << 24);
            accumulators = unsafe { mma_m16n8k64_nvfp4(a, b, scale_a, scale_b, accumulators) };
            thread::sync_threads();
            context_tile += 1;
        }
        let output_row = lane >> 2;
        let output_col = (lane & 3) * 2;
        let q_head = query_base + output_row;
        if q_head < (kv_head + 1) * queries_per_kv {
            unsafe {
                destination
                    .add((q_head * head_dim + dim_tile * 8 + output_col) as usize)
                    .write(accumulators[0] / correction);
                destination
                    .add((q_head * head_dim + dim_tile * 8 + output_col + 1) as usize)
                    .write(accumulators[1] / correction);
            }
        }
    }

    /// Reduces split compact-attention PV outputs.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn sm12x_kv_pv_reduce_f32(
        partial_output: *const f32,
        output: *mut f32,
        pv_splits: u32,
        width: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= width {
            return;
        }
        let mut sum = 0.0f32;
        let mut split = 0;
        while split < pv_splits {
            sum += unsafe { *partial_output.add((split * width + index) as usize) };
            split += 1;
        }
        unsafe { output.add(index as usize).write(sum) };
    }

    /// Copies one dense K/V row using a device-resident cache position.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn sm12x_kv_copy_tail_indexed_f32(
        key: *const f32,
        value: *const f32,
        key_tail: *mut f32,
        value_tail: *mut f32,
        position: *const u32,
        max_tokens: u32,
        width: u32,
    ) {
        let position = unsafe { *position };
        let column = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if position >= max_tokens || column >= width {
            return;
        }
        let destination = (position & 15) as usize * width as usize + column as usize;
        unsafe {
            key_tail.add(destination).write(*key.add(column as usize));
            value_tail
                .add(destination)
                .write(*value.add(column as usize));
        }
    }

    /// Finalizes a key tile using a device-resident cache position.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 2, coordinates = u32, block = (1, 1, 1))]
    pub unsafe fn sm12x_kv_finalize_key_indexed_f32(
        key_tail: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        position: *const u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        let position = unsafe { *position };
        if position >= max_tokens || position & 7 != 7 {
            return;
        }
        unsafe {
            finalize_compact_key(
                key_tail,
                key_values,
                key_scales,
                position,
                max_tokens,
                kv_heads,
                head_dim,
                thread::blockIdx_x(),
                thread::blockIdx_y(),
            )
        }
    }

    /// Finalizes a value tile using a device-resident cache position.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 2, coordinates = u32, block = (1, 1, 1))]
    pub unsafe fn sm12x_kv_finalize_value_indexed_f32(
        value_tail: *const f32,
        value_values: *mut u8,
        value_scales: *mut u8,
        position: *const u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
    ) {
        let position = unsafe { *position };
        if position >= max_tokens || position & 15 != 15 {
            return;
        }
        unsafe {
            finalize_compact_value(
                value_tail,
                value_values,
                value_scales,
                position,
                max_tokens,
                kv_heads,
                head_dim,
                thread::blockIdx_x(),
                thread::blockIdx_y(),
            )
        }
    }
}
