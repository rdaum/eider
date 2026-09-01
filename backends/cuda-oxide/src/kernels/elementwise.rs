//! Elementwise, conversion, embedding, and small linear kernels.

use crate::common::*;
use cuda_device::{
    SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};

/// CUDA entry points for small independent operations.
#[cuda_module]
mod device {
    use super::*;

    /// Fills an active f32 buffer.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn fill_f32(output: *mut f32, value: f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe { output.add(index as usize).write(value) };
        }
    }

    /// Adds two active f32 buffers.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn add_f32(left: *const f32, right: *const f32, output: *mut f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                output
                    .add(index as usize)
                    .write(*left.add(index as usize) + *right.add(index as usize))
            };
        }
    }

    /// Accumulates a scaled f32 input into an output buffer.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn scaled_add_f32(input: *const f32, output: *mut f32, scale: f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                let output = output.add(index as usize);
                *output += *input.add(index as usize) * scale;
            }
        }
    }

    /// Applies `SiLU(gate) * up` to one row of concatenated halves.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn silu_mul_halves_f32(gate_up: *const f32, output: *mut f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                let gate = *gate_up.add(index as usize);
                let up = *gate_up.add((len + index) as usize);
                output.add(index as usize).write(gate * sigmoid(gate) * up);
            }
        }
    }

    /// Applies `SiLU(gate) * up` to row-major concatenated halves.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn silu_mul_halves_f32_batch(
        gate_up: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = rows * cols;
        if index < len {
            let row = index / cols;
            let col = index - row * cols;
            let base = row * cols * 2;
            unsafe {
                let gate = *gate_up.add((base + col) as usize);
                let up = *gate_up.add((base + cols + col) as usize);
                output.add(index as usize).write(gate * sigmoid(gate) * up);
            }
        }
    }

    /// Multiplies an input by an elementwise sigmoid gate.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn sigmoid_mul_f32(gate: *const f32, input: *const f32, output: *mut f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                output
                    .add(index as usize)
                    .write(*input.add(index as usize) * sigmoid(*gate.add(index as usize)));
            }
        }
    }

    /// Rounds f32 values to BF16 precision while retaining f32 storage.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn round_f32_to_bf16(input: *const f32, output: *mut f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                output
                    .add(index as usize)
                    .write(round_to_bf16(*input.add(index as usize)));
            }
        }
    }

    /// Converts f32 values to BF16 storage.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn f32_to_bf16(input: *const f32, output: *mut u16, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            let encoded =
                convert::cvt_bf16x2_f32(unsafe { *input.add(index as usize) }, 0.0) as u16;
            unsafe { output.add(index as usize).write(encoded) };
        }
    }

    /// Converts BF16 storage to f32 values.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn convert_bf16_to_f32(input: *const u16, output: *mut f32, len: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < len {
            unsafe {
                output
                    .add(index as usize)
                    .write(bf16_to_f32(*input.add(index as usize)))
            };
        }
    }

    /// Gathers mapped-host BF16 rows from byte offsets into contiguous f32 rows.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn paged_bf16_rows_to_f32(
        pages: *const u8,
        row_offsets: *const u32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * cols {
            return;
        }
        let row = index / cols;
        let col = index - row * cols;
        let offset = unsafe { *row_offsets.add(row as usize) } as usize;
        let values = unsafe { pages.add(offset).cast::<u16>() };
        unsafe {
            output
                .add(index as usize)
                .write(bf16_to_f32(*values.add(col as usize)))
        };
    }

    /// Selects and normalizes the largest MoE router logits for each row.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn moe_topk_normalized_f32(
        logits: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        experts: u32,
        top_k: u32,
    ) {
        static mut REDUCTION_VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut REDUCTION_INDICES: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut SELECTED: SharedArray<u8, 1024> = SharedArray::UNINIT;
        static mut TOP_VALUES: SharedArray<f32, 32> = SharedArray::UNINIT;
        let reduction_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION_VALUES) };
        let reduction_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION_INDICES) };
        let selected = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SELECTED) };
        let top_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOP_VALUES) };
        let row = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        let logits = unsafe { logits.add((row * experts) as usize) };
        let out_indices = unsafe { out_indices.add((row * top_k) as usize) };
        let out_weights = unsafe { out_weights.add((row * top_k) as usize) };

        let mut expert = lane;
        while expert < experts {
            unsafe { selected.add(expert as usize).write(0) };
            expert += thread::blockDim_x();
        }
        thread::sync_threads();

        let mut slot = 0;
        while slot < top_k {
            let mut best_value = f32::NEG_INFINITY;
            let mut best_index = u32::MAX;
            expert = lane;
            while expert < experts {
                if unsafe { *selected.add(expert as usize) } == 0 {
                    let mut value = unsafe { *logits.add(expert as usize) };
                    if value.is_nan() {
                        value = f32::NEG_INFINITY;
                    } else if value == f32::INFINITY {
                        value = f32::MAX;
                    } else if value == 0.0 {
                        value = 0.0;
                    }
                    if value > best_value || (value == best_value && expert < best_index) {
                        best_value = value;
                        best_index = expert;
                    }
                }
                expert += thread::blockDim_x();
            }
            unsafe {
                reduction_values.add(lane as usize).write(best_value);
                reduction_indices.add(lane as usize).write(best_index);
            }
            thread::sync_threads();
            let mut stride = thread::blockDim_x() / 2;
            while stride != 0 {
                if lane < stride {
                    let other_value = unsafe { *reduction_values.add((lane + stride) as usize) };
                    let other_index = unsafe { *reduction_indices.add((lane + stride) as usize) };
                    let value = unsafe { *reduction_values.add(lane as usize) };
                    let index = unsafe { *reduction_indices.add(lane as usize) };
                    if other_value > value || (other_value == value && other_index < index) {
                        unsafe {
                            reduction_values.add(lane as usize).write(other_value);
                            reduction_indices.add(lane as usize).write(other_index);
                        }
                    }
                }
                thread::sync_threads();
                stride /= 2;
            }
            if lane == 0 {
                let index = unsafe { *reduction_indices };
                let value = unsafe { *reduction_values };
                unsafe {
                    out_indices.add(slot as usize).write(index);
                    top_values.add(slot as usize).write(value);
                    if index < experts {
                        selected.add(index as usize).write(1);
                    }
                }
            }
            thread::sync_threads();
            slot += 1;
        }

        if lane == 0 {
            let maximum = unsafe { *top_values };
            if !maximum.is_finite() {
                slot = 0;
                while slot < top_k {
                    unsafe {
                        out_indices.add(slot as usize).write(slot);
                        out_weights
                            .add(slot as usize)
                            .write(if slot == 0 { 1.0 } else { 0.0 });
                    }
                    slot += 1;
                }
                return;
            }
            let mut sum = 0.0f32;
            slot = 0;
            while slot < top_k {
                let probability = (unsafe { *top_values.add(slot as usize) } - maximum).exp();
                unsafe { top_values.add(slot as usize).write(probability) };
                sum += probability;
                slot += 1;
            }
            slot = 0;
            while slot < top_k {
                unsafe {
                    out_weights
                        .add(slot as usize)
                        .write(*top_values.add(slot as usize) / sum)
                };
                slot += 1;
            }
        }
    }

    /// Combines contiguous route-major expert rows with router weights.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn moe_weighted_accumulate_contiguous_f32(
        route_weights: *const f32,
        routed: *const f32,
        output: *mut f32,
        rows: u32,
        routes_per_row: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * cols {
            return;
        }
        let row = index / cols;
        let col = index - row * cols;
        let route_base = row * routes_per_row;
        let mut sum = 0.0f32;
        let mut slot = 0;
        while slot < routes_per_row {
            let route = route_base + slot;
            sum += unsafe {
                *routed.add((route * cols + col) as usize) * *route_weights.add(route as usize)
            };
            slot += 1;
        }
        unsafe { output.add(index as usize).write(sum) };
    }

    /// Finalizes routed and shared Qwen MoE rows with BF16 residual rounding.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_ffn_finalize_batch_f32(
        routed: *const f32,
        shared_gate: *const f32,
        shared: *const f32,
        residual: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * cols {
            return;
        }
        let row = index / cols;
        let shared_scale = sigmoid(unsafe { *shared_gate.add(row as usize) });
        let value = unsafe {
            *residual.add(index as usize)
                + *routed.add(index as usize)
                + *shared.add(index as usize) * shared_scale
        };
        unsafe { output.add(index as usize).write(round_to_bf16(value)) };
    }

    /// Gathers BF16 embedding rows into row-major f32 output.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn copy_bf16_rows_to_f32_indexed(
        input: *const u16,
        rows: *const u32,
        output: *mut f32,
        row_count: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = row_count * cols;
        if index < len {
            let output_row = index / cols;
            let col = index - output_row * cols;
            let input_row = unsafe { *rows.add(output_row as usize) };
            let value = unsafe { *input.add((input_row * cols + col) as usize) };
            unsafe { output.add(index as usize).write(bf16_to_f32(value)) };
        }
    }

    /// Gathers scaled E4M3 embedding rows into row-major f32 output.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn copy_fp8_rows_to_f32_indexed(
        input: *const u8,
        row_scales: *const f32,
        rows: *const u32,
        output: *mut f32,
        row_count: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = row_count * cols;
        if index < len {
            let output_row = index / cols;
            let col = index - output_row * cols;
            let input_row = unsafe { *rows.add(output_row as usize) };
            let value = unsafe { *input.add((input_row * cols + col) as usize) };
            let scale = unsafe { *row_scales.add(input_row as usize) };
            unsafe { output.add(index as usize).write(e4m3_value(value) * scale) };
        }
    }

    /// Concatenates two row-major f32 matrices along their column dimension.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn concat_f32_rows(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let len = rows * cols * 2;
        if index < len {
            let output_cols = cols * 2;
            let row = index / output_cols;
            let col = index - row * output_cols;
            let value = if col < cols {
                unsafe { *left.add((row * cols + col) as usize) }
            } else {
                unsafe { *right.add((row * cols + col - cols) as usize) }
            };
            unsafe { output.add(index as usize).write(value) };
        }
    }

    /// Projects one dynamically quantized E4M3 row through channel-scaled E4M3 weights.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn fp8_linear_quantized_channel_scaled_f32(
        input: *const u8,
        weight: *const u8,
        channel_scale: *const f32,
        input_scale: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 16> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let lane = thread::threadIdx_x() & 31;
        let warp_index = thread::threadIdx_x() >> 5;
        let mut sum = 0.0f32;
        let mut col = thread::threadIdx_x();
        while col < cols {
            unsafe {
                sum += e4m3_value(*input.add(col as usize))
                    * e4m3_value(*weight.add((row * cols + col) as usize));
            }
            col += thread::blockDim_x();
        }
        sum = warp_sum(sum);
        if lane == 0 {
            unsafe { partial.add(warp_index as usize).write(sum) };
        }
        thread::sync_threads();
        if thread::threadIdx_x() == 0 {
            let warps = thread::blockDim_x() >> 5;
            let mut total = 0.0f32;
            let mut warp = 0;
            while warp < warps {
                total += unsafe { *partial.add(warp as usize) };
                warp += 1;
            }
            unsafe {
                output
                    .add(row as usize)
                    .write(total * *input_scale * *channel_scale.add(row as usize))
            };
        }
    }

    /// Projects one f32 row through channel-scaled E4M3 weights.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn fp8_linear_channel_scaled_f32(
        input: *const f32,
        weight: *const u8,
        channel_scale: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 16> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let lane = thread::threadIdx_x() & 31;
        let warp_index = thread::threadIdx_x() >> 5;
        let mut sum = 0.0f32;
        let mut col = thread::threadIdx_x();
        while col < cols {
            unsafe {
                sum +=
                    *input.add(col as usize) * e4m3_value(*weight.add((row * cols + col) as usize));
            }
            col += thread::blockDim_x();
        }
        sum = warp_sum(sum);
        if lane == 0 {
            unsafe { partial.add(warp_index as usize).write(sum) };
        }
        thread::sync_threads();
        if thread::threadIdx_x() == 0 {
            let warps = thread::blockDim_x() >> 5;
            let mut total = 0.0f32;
            let mut warp = 0;
            while warp < warps {
                total += unsafe { *partial.add(warp as usize) };
                warp += 1;
            }
            unsafe {
                output
                    .add(row as usize)
                    .write(total * *channel_scale.add(row as usize))
            };
        }
    }
}
