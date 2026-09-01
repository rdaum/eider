//! Linear projection, quantization, and normalization kernels.

use crate::common::*;
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread};

/// CUDA entry points for linear algebra and normalization.
#[cuda_module]
mod device {
    use super::*;

    /// Projects a small f32 batch through row-major BF16 weights.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn bf16_linear_logits_f32_batch(
        input: *const f32,
        weight: *const u16,
        logits: *mut f32,
        batch_size: u32,
        rows: u32,
        cols: u32,
    ) {
        let warp_index = thread::threadIdx_x() >> 5;
        let lane = thread::threadIdx_x() & 31;
        let row = thread::blockIdx_x() * 8 + warp_index;
        if row >= rows {
            return;
        }
        let batch_base = thread::blockIdx_y() * 8;
        let active = 8u32.min(batch_size - batch_base);
        let row_weight = unsafe { weight.add(row as usize * cols as usize) };
        let mut accumulators = [0.0f32; 8];
        let mut col = lane * 4;
        while col < cols {
            let weight0 = bf16_to_f32(unsafe { *row_weight.add(col as usize) });
            let weight1 = bf16_to_f32(unsafe { *row_weight.add(col as usize + 1) });
            let weight2 = bf16_to_f32(unsafe { *row_weight.add(col as usize + 2) });
            let weight3 = bf16_to_f32(unsafe { *row_weight.add(col as usize + 3) });
            let mut batch = 0;
            while batch < active {
                let input_row = unsafe { input.add((batch_base + batch) as usize * cols as usize) };
                accumulators[batch as usize] = unsafe {
                    weight0.mul_add(*input_row.add(col as usize), accumulators[batch as usize])
                };
                accumulators[batch as usize] = unsafe {
                    weight1.mul_add(
                        *input_row.add(col as usize + 1),
                        accumulators[batch as usize],
                    )
                };
                accumulators[batch as usize] = unsafe {
                    weight2.mul_add(
                        *input_row.add(col as usize + 2),
                        accumulators[batch as usize],
                    )
                };
                accumulators[batch as usize] = unsafe {
                    weight3.mul_add(
                        *input_row.add(col as usize + 3),
                        accumulators[batch as usize],
                    )
                };
                batch += 1;
            }
            col += 128;
        }
        let mut batch = 0;
        while batch < active {
            let value = warp_sum(accumulators[batch as usize]);
            if lane == 0 {
                unsafe {
                    logits
                        .add((batch_base + batch) as usize * rows as usize + row as usize)
                        .write(value)
                };
            }
            batch += 1;
        }
    }

    /// Projects a small f32 batch through BF16 weights for arbitrary widths.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn bf16_linear_logits_f32_batch_scalar(
        input: *const f32,
        weight: *const u16,
        logits: *mut f32,
        batch_size: u32,
        rows: u32,
        cols: u32,
    ) {
        let warp_index = thread::threadIdx_x() >> 5;
        let lane = thread::threadIdx_x() & 31;
        let row = thread::blockIdx_x() * 8 + warp_index;
        if row >= rows {
            return;
        }
        let batch_base = thread::blockIdx_y() * 8;
        let active = 8u32.min(batch_size - batch_base);
        let row_weight = unsafe { weight.add(row as usize * cols as usize) };
        let mut accumulators = [0.0f32; 8];
        let mut col = lane;
        while col < cols {
            let weight_value = bf16_to_f32(unsafe { *row_weight.add(col as usize) });
            let mut batch = 0;
            while batch < active {
                let input_value = unsafe {
                    *input.add((batch_base + batch) as usize * cols as usize + col as usize)
                };
                accumulators[batch as usize] =
                    weight_value.mul_add(input_value, accumulators[batch as usize]);
                batch += 1;
            }
            col += 32;
        }
        let mut batch = 0;
        while batch < active {
            let value = warp_sum(accumulators[batch as usize]);
            if lane == 0 {
                unsafe {
                    logits
                        .add((batch_base + batch) as usize * rows as usize + row as usize)
                        .write(value)
                };
            }
            batch += 1;
        }
    }

    /// Quantizes a column-major f32 matrix to cuBLASLt NVFP4 layout.
    #[kernel]
    #[launch_bounds(32)]
    pub unsafe fn quantize_nvfp4_col_major_f32(
        input: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        cols: u32,
        input_scale: f32,
    ) {
        let group = thread::blockIdx_x();
        let row_blocks = rows.div_ceil(16);
        let col = group / row_blocks;
        let row_block = group - col * row_blocks;
        if col >= cols {
            return;
        }
        let row_start = row_block * 16;
        let row_end = (row_start + 16).min(rows);
        let lane = thread::threadIdx_x();
        let mut maximum = 0.0;
        if row_start + lane < row_end {
            let value =
                unsafe { *input.add((row_start + lane + col * rows) as usize) } / input_scale;
            maximum = if value.is_finite() { value.abs() } else { 0.0 };
        }
        maximum = warp_max(maximum);
        let scale_code = if maximum == 0.0 {
            0
        } else {
            ue4m3_code(maximum / 6.0)
        };
        if lane == 0 {
            unsafe {
                scales
                    .add(ue4m3_tiled_scale_offset(col, row_block, rows))
                    .write(scale_code)
            };
        }
        let scale = e4m3_value(scale_code);
        if lane < 8 && row_start + lane * 2 < row_end {
            let row = row_start + lane * 2;
            let low_value = if scale == 0.0 {
                0.0
            } else {
                (unsafe { *input.add((row + col * rows) as usize) }) / input_scale / scale
            };
            let low = e2m1_code(low_value);
            let high = if row + 1 < row_end {
                let high_value = if scale == 0.0 {
                    0.0
                } else {
                    (unsafe { *input.add((row + 1 + col * rows) as usize) }) / input_scale / scale
                };
                e2m1_code(high_value)
            } else {
                0
            };
            unsafe {
                packed
                    .add(((row + col * rows) / 2) as usize)
                    .write(low | high << 4)
            };
        }
    }

    /// Applies row-wise RMS normalization.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn rms_norm_f32(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let row = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if row >= rows {
            return;
        }
        let offset = row as usize * cols as usize;
        let mut square_sum = 0.0;
        let mut col = lane;
        while col < cols {
            let value = unsafe { *input.add(offset + col as usize) };
            square_sum += value * value;
            col += thread::blockDim_x();
        }
        unsafe { partial.add(lane as usize).write(square_sum) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_rms =
            cuda_device::float::rsqrt_approx_f32(unsafe { *partial } / cols as f32 + eps);
        let mut col = lane;
        while col < cols {
            unsafe {
                output.add(offset + col as usize).write(
                    *input.add(offset + col as usize) * inverse_rms * *weight.add(col as usize),
                )
            };
            col += thread::blockDim_x();
        }
    }

    /// Applies row-wise RMS normalization and a SiLU gate.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn gated_rms_norm_f32(
        input: *const f32,
        gate: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let row = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if row >= rows {
            return;
        }
        let offset = row as usize * cols as usize;
        let mut square_sum = 0.0;
        let mut col = lane;
        while col < cols {
            let value = unsafe { *input.add(offset + col as usize) };
            square_sum += value * value;
            col += thread::blockDim_x();
        }
        unsafe { partial.add(lane as usize).write(square_sum) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_rms =
            cuda_device::float::rsqrt_approx_f32(unsafe { *partial } / cols as f32 + eps);
        let mut col = lane;
        while col < cols {
            unsafe {
                let index = offset + col as usize;
                let gate = *gate.add(index);
                output.add(index).write(
                    *input.add(index)
                        * inverse_rms
                        * *weight.add(col as usize)
                        * gate
                        * sigmoid(gate),
                )
            };
            col += thread::blockDim_x();
        }
    }

    /// Applies per-head gated RMSNorm and writes column-major NVFP4 storage.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn gated_rms_norm_quantize_nvfp4_f32(
        input: *const f32,
        gate: *const f32,
        weight: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        heads: u32,
        eps: f32,
        input_scale: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut SCALE_CODES: SharedArray<u8, 8> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let scale_codes = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE_CODES) };
        let block = thread::blockIdx_x();
        let row = block / heads;
        let head = block - row * heads;
        let lane = thread::threadIdx_x();
        let cols = heads * 128;
        let head_offset = row as usize * cols as usize + head as usize * 128;
        let index = head_offset + lane as usize;
        let input_value = unsafe { *input.add(index) };
        unsafe { partial.add(lane as usize).write(input_value * input_value) };
        thread::sync_threads();
        let mut stride = 64;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inverse_rms = cuda_device::float::rsqrt_approx_f32(unsafe { *partial } / 128.0 + eps);
        let gate_value = unsafe { *gate.add(index) };
        let value = input_value
            * inverse_rms
            * unsafe { *weight.add(lane as usize) }
            * gate_value
            * sigmoid(gate_value)
            / input_scale;
        unsafe { values.add(lane as usize).write(value) };
        thread::sync_threads();

        if lane < 8 {
            let group_start = lane * 16;
            let mut maximum = 0.0f32;
            let mut item = 0;
            while item < 16 {
                maximum =
                    maximum.max(unsafe { (*values.add((group_start + item) as usize)).abs() });
                item += 1;
            }
            let scale_code = if maximum == 0.0 {
                0
            } else {
                ue4m3_code(maximum / 6.0)
            };
            unsafe {
                scale_codes.add(lane as usize).write(scale_code);
                scales
                    .add(ue4m3_tiled_scale_offset(row, head * 8 + lane, cols))
                    .write(scale_code);
            }
        }
        thread::sync_threads();

        if lane < 64 {
            let first = lane * 2;
            let scale = e4m3_value(unsafe { *scale_codes.add((first / 16) as usize) });
            let low = if scale == 0.0 {
                0
            } else {
                e2m1_code(unsafe { *values.add(first as usize) } / scale)
            };
            let high = if scale == 0.0 {
                0
            } else {
                e2m1_code(unsafe { *values.add(first as usize + 1) } / scale)
            };
            unsafe {
                packed
                    .add(head_offset / 2 + lane as usize)
                    .write(low | high << 4)
            };
        }
    }
}
