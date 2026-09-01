//! Rotary position, activation quantization, and logit kernels.

use cuda_device::{
    SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

/// CUDA entry points for position and logit preparation.
#[cuda_module]
mod device {
    use super::*;

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn rope_neox_partial<const MODE: u32>(
        input: *const f32,
        output: *mut f32,
        len: u32,
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        start_position: u32,
        indexed_position: *const u32,
        theta: f32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= len {
            return;
        }
        let dim = index % head_dim;
        if dim >= rotary_dim {
            unsafe { output.add(index as usize).write(*input.add(index as usize)) };
            return;
        }
        let half = rotary_dim / 2;
        if dim >= half {
            return;
        }
        let row = index / head_dim;
        let row_start = row * head_dim;
        let position = if MODE == 0 {
            start_position
        } else if MODE == 1 {
            unsafe { *indexed_position }
        } else {
            start_position + row / heads
        };
        let exponent = -2.0 * dim as f32 / rotary_dim as f32;
        let angle = position as f32 * theta.powf(exponent);
        let sine = cuda_device::float::sin_approx_f32(angle);
        let cosine = cuda_device::float::cos_approx_f32(angle);
        let first = unsafe { *input.add((row_start + dim) as usize) };
        let second = unsafe { *input.add((row_start + dim + half) as usize) };
        unsafe {
            output
                .add((row_start + dim) as usize)
                .write(first * cosine - second * sine);
            output
                .add((row_start + dim + half) as usize)
                .write(first * sine + second * cosine);
        }
    }

    /// Applies partial NeoX RoPE at one host-supplied position.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn rope_neox_partial_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        theta: f32,
    ) {
        unsafe {
            rope_neox_partial::<0>(
                input,
                output,
                rows * head_dim,
                1,
                head_dim,
                rotary_dim,
                position,
                core::ptr::null(),
                theta,
            )
        }
    }

    /// Applies partial NeoX RoPE at one device-supplied position.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn rope_neox_partial_indexed_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: *const u32,
        theta: f32,
    ) {
        unsafe {
            rope_neox_partial::<1>(
                input,
                output,
                rows * head_dim,
                1,
                head_dim,
                rotary_dim,
                0,
                position,
                theta,
            )
        }
    }

    /// Applies partial NeoX RoPE across a contiguous token sequence.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn rope_neox_partial_sequence_f32(
        input: *const f32,
        output: *mut f32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        start_position: u32,
        theta: f32,
    ) {
        unsafe {
            rope_neox_partial::<2>(
                input,
                output,
                tokens * heads * head_dim,
                heads,
                head_dim,
                rotary_dim,
                start_position,
                core::ptr::null(),
                theta,
            )
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn rope_imrope<const MODE: u32>(
        input: *const f32,
        output: *mut f32,
        positions: *const u32,
        position_count: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
        pos_t: u32,
        pos_h: u32,
        pos_w: u32,
        pos_extra: u32,
        theta: f32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = rows * heads * head_dim;
        if index >= len {
            return;
        }
        let dim = index % head_dim;
        if dim >= rotary_dim {
            unsafe { output.add(index as usize).write(*input.add(index as usize)) };
            return;
        }
        let half = rotary_dim / 2;
        if dim >= half {
            return;
        }

        let batch = index / (heads * head_dim);
        let (position_t, position_h, position_w, position_extra) = if MODE == 0 {
            (pos_t, pos_h, pos_w, pos_extra)
        } else if MODE == 1 {
            let position_t = unsafe { *positions };
            if position_count == 1 {
                (position_t, position_t, position_t, 0)
            } else {
                (
                    position_t,
                    unsafe { *positions.add(1) },
                    unsafe { *positions.add(2) },
                    unsafe { *positions.add(3) },
                )
            }
        } else {
            let position = unsafe { *positions.add(batch as usize) };
            (position, position, position, 0)
        };
        let section_dims = v0 + v1 + v2 + v3;
        let sector = dim % section_dims;
        let position = if sector % 3 == 1 && sector < 3 * v1 {
            position_h
        } else if sector % 3 == 2 && sector < 3 * v2 {
            position_w
        } else if sector % 3 == 0 && sector < 3 * v0 {
            position_t
        } else {
            position_extra
        };
        let exponent = -2.0 * dim as f32 / rotary_dim as f32;
        let angle = position as f32 * theta.powf(exponent);
        let sine = cuda_device::float::sin_approx_f32(angle);
        let cosine = cuda_device::float::cos_approx_f32(angle);
        let row_start = (index / head_dim) * head_dim;
        let first = unsafe { *input.add((row_start + dim) as usize) };
        let second = unsafe { *input.add((row_start + dim + half) as usize) };
        unsafe {
            output
                .add((row_start + dim) as usize)
                .write(first * cosine - second * sine);
            output
                .add((row_start + dim + half) as usize)
                .write(first * sine + second * cosine);
        }
    }

    /// Applies interleaved MRoPE at host-supplied positions.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_imrope_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
        pos_t: u32,
        pos_h: u32,
        pos_w: u32,
        pos_extra: u32,
        theta: f32,
    ) {
        unsafe {
            rope_imrope::<0>(
                input,
                output,
                core::ptr::null(),
                0,
                rows,
                1,
                head_dim,
                rotary_dim,
                v0,
                v1,
                v2,
                v3,
                pos_t,
                pos_h,
                pos_w,
                pos_extra,
                theta,
            )
        }
    }

    /// Applies interleaved MRoPE at device-supplied positions.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_imrope_indexed_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
        positions: *const u32,
        position_count: u32,
        theta: f32,
    ) {
        unsafe {
            rope_imrope::<1>(
                input,
                output,
                positions,
                position_count,
                rows,
                1,
                head_dim,
                rotary_dim,
                v0,
                v1,
                v2,
                v3,
                0,
                0,
                0,
                0,
                theta,
            )
        }
    }

    /// Applies text interleaved MRoPE to a batch of head rows.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_imrope_text_batch_f32(
        input: *const f32,
        output: *mut f32,
        positions: *const u32,
        batch_size: u32,
        heads_per_row: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
        theta: f32,
    ) {
        unsafe {
            rope_imrope::<2>(
                input,
                output,
                positions,
                batch_size,
                batch_size,
                heads_per_row,
                head_dim,
                rotary_dim,
                v0,
                v1,
                v2,
                v3,
                0,
                0,
                0,
                0,
                theta,
            )
        }
    }

    /// Dynamically quantizes independent f32 rows to E4M3.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn quantize_fp8_e4m3_dynamic_f32(
        input: *const f32,
        quantized: *mut u8,
        input_scale: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let row = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if row >= rows {
            return;
        }
        let row_offset = row as usize * cols as usize;
        let mut maximum = 0.0f32;
        let mut col = lane;
        while col < cols {
            let value = unsafe { *input.add(row_offset + col as usize) };
            if value.is_finite() {
                maximum = maximum.max(value.abs());
            }
            col += thread::blockDim_x();
        }
        unsafe { reduction.add(lane as usize).write(maximum) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe {
                    *reduction.add(lane as usize) = (*reduction.add(lane as usize))
                        .max(*reduction.add((lane + stride) as usize))
                };
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            let maximum = unsafe { *reduction };
            unsafe {
                input_scale.add(row as usize).write(if maximum == 0.0 {
                    1.0
                } else {
                    maximum / 448.0
                })
            };
        }
        thread::sync_threads();
        let scale = unsafe { *input_scale.add(row as usize) };
        let pairs = cols.div_ceil(2);
        let mut pair = lane;
        while pair < pairs {
            let first_col = pair * 2;
            let first = (unsafe { *input.add(row_offset + first_col as usize) }) / scale;
            let second = if first_col + 1 < cols {
                (unsafe { *input.add(row_offset + first_col as usize + 1) }) / scale
            } else {
                0.0
            };
            let packed = convert::cvt_rn_satfinite_e4m3x2_f32(first, second);
            let output_offset = row_offset + first_col as usize;
            if first_col + 1 < cols {
                unsafe { quantized.add(output_offset).cast::<u16>().write(packed) };
            } else {
                unsafe { quantized.add(output_offset).write(packed as u8) };
            }
            pair += thread::blockDim_x();
        }
    }

    /// Applies channel and scalar output scales in place.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn scale_channel_f32_device_scalar(
        values: *mut f32,
        channel_scale: *const f32,
        scalar: *const f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                let value = values.add(index as usize);
                *value *= *channel_scale.add(index as usize) * *scalar;
            }
        }
    }

    /// Applies channel and per-row output scales in place.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn scale_channel_f32_device_row_scalar(
        values: *mut f32,
        channel_scale: *const f32,
        row_scale: *const f32,
        channels: u32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            let row = index / channels;
            let channel = index - row * channels;
            unsafe {
                let value = values.add(index as usize);
                *value *= *channel_scale.add(channel as usize) * *row_scale.add(row as usize);
            }
        }
    }

    /// Selects the maximum value and lowest matching index from each row.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn argmax_f32(
        values: *const f32,
        out_index: *mut u32,
        out_value: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        static mut MAX_VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAX_INDICES: SharedArray<u32, 256> = SharedArray::UNINIT;
        let max_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAX_VALUES) };
        let max_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAX_INDICES) };
        let row = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if row >= rows {
            return;
        }
        let row_offset = row as usize * cols as usize;
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = 0;
        let mut col = lane;
        while col < cols {
            let value = unsafe { *values.add(row_offset + col as usize) };
            if value > best_value || (value == best_value && col < best_index) {
                best_value = value;
                best_index = col;
            }
            col += thread::blockDim_x();
        }
        unsafe {
            max_values.add(lane as usize).write(best_value);
            max_indices.add(lane as usize).write(best_index);
        }
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                let other_value = unsafe { *max_values.add((lane + stride) as usize) };
                let other_index = unsafe { *max_indices.add((lane + stride) as usize) };
                let current_value = unsafe { *max_values.add(lane as usize) };
                let current_index = unsafe { *max_indices.add(lane as usize) };
                if other_value > current_value
                    || (other_value == current_value && other_index < current_index)
                {
                    unsafe {
                        max_values.add(lane as usize).write(other_value);
                        max_indices.add(lane as usize).write(other_index);
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            unsafe {
                out_index.add(row as usize).write(*max_indices);
                out_value.add(row as usize).write(*max_values);
            }
        }
    }

    /// Masks disallowed row-major logits with negative infinity.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn mask_logits_f32_batch(
        logits: *mut f32,
        allowed: *const u32,
        rows: u32,
        cols: u32,
        mask_words: u32,
    ) {
        let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let row = thread::blockIdx_y();
        if row >= rows || col >= cols {
            return;
        }
        let word = unsafe { *allowed.add((row * mask_words + col / 32) as usize) };
        if word & (1 << (col % 32)) == 0 {
            unsafe {
                logits
                    .add((row as usize * cols as usize) + col as usize)
                    .write(f32::NEG_INFINITY)
            };
        }
    }
}
