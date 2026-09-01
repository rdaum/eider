//! Qwen Gated DeltaNet preparation and recurrent kernels.

use crate::common::*;
use cuda_device::{
    SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

/// CUDA entry points for recurrent Gated DeltaNet execution.
#[cuda_module]
mod device {
    use super::*;

    /// Splits dense Qwen query/gate heads and RMS-normalizes query/key heads.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn qwen36_full_attn_prep_f32(
        q_full: *const f32,
        k_raw: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        q: *mut f32,
        gate: *mut f32,
        k: *mut f32,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let heads_per_row = q_heads + kv_heads;
        let block = thread::blockIdx_x();
        let batch = block / heads_per_row;
        let head = block - batch * heads_per_row;
        let lane = thread::threadIdx_x();
        if batch >= rows || lane >= head_dim {
            return;
        }

        let (value, norm, output, output_index) = if head < q_heads {
            let q_width = q_heads as usize * head_dim as usize;
            let input_base = batch as usize * q_width * 2 + head as usize * head_dim as usize * 2;
            let output_index =
                batch as usize * q_width + head as usize * head_dim as usize + lane as usize;
            unsafe {
                gate.add(output_index)
                    .write(*q_full.add(input_base + head_dim as usize + lane as usize));
            }
            (
                unsafe { *q_full.add(input_base + lane as usize) },
                q_norm,
                q,
                output_index,
            )
        } else {
            let k_head = head - q_heads;
            let kv_width = kv_heads as usize * head_dim as usize;
            let output_index =
                batch as usize * kv_width + k_head as usize * head_dim as usize + lane as usize;
            (unsafe { *k_raw.add(output_index) }, k_norm, k, output_index)
        };
        unsafe { partial.add(lane as usize).write(value * value) };
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
            cuda_device::float::rsqrt_approx_f32(unsafe { *partial } / head_dim as f32 + eps);
        unsafe {
            output
                .add(output_index)
                .write(value * inverse_rms * *norm.add(lane as usize));
        }
    }

    /// Applies the single-token Qwen GDN causal depthwise convolution.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_prep_f32(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state: *mut f32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        if index >= conv_dim {
            return;
        }
        let index = index as usize;
        let weight = unsafe { conv_weight_bf16.add(index * 4) };
        let state = unsafe { conv_state.add(index * 3) };
        let input = unsafe { *qkv.add(index) };
        let mut mixed = input * bf16_to_f32(unsafe { *weight.add(3) });
        mixed += unsafe { *state } * bf16_to_f32(unsafe { *weight });
        mixed += unsafe { *state.add(1) } * bf16_to_f32(unsafe { *weight.add(1) });
        mixed += unsafe { *state.add(2) } * bf16_to_f32(unsafe { *weight.add(2) });
        let activated = mixed * sigmoid(mixed);
        unsafe {
            *state = *state.add(1);
            *state.add(1) = *state.add(2);
            *state.add(2) = input;
        }
        if index < key_dim as usize {
            let mut repeat = 0;
            while repeat < value_heads / key_heads {
                unsafe {
                    q.add(repeat as usize * key_dim as usize + index)
                        .write(activated)
                };
                repeat += 1;
            }
        } else if index < (key_dim * 2) as usize {
            let key_index = index - key_dim as usize;
            let mut repeat = 0;
            while repeat < value_heads / key_heads {
                unsafe {
                    k.add(repeat as usize * key_dim as usize + key_index)
                        .write(activated)
                };
                repeat += 1;
            }
        } else {
            let value_index = index - (key_dim * 2) as usize;
            let value_head = value_index / head_dim as usize;
            let sub = value_index % head_dim as usize;
            let values_per_key = (value_heads / key_heads) as usize;
            let key_head = value_head / values_per_key;
            let value_subhead = value_head % values_per_key;
            let tiled = value_subhead * key_dim as usize + key_head * head_dim as usize + sub;
            unsafe { v.add(tiled).write(activated) };
        }
    }

    /// Applies one-token Qwen GDN convolution to a device pointer-table batch.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_prep_batch_f32(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state_table: *const *mut f32,
        batch_size: u32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
    ) {
        let linear = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let total = batch_size * conv_dim;
        if linear >= total {
            return;
        }
        let batch = linear / conv_dim;
        let index = linear - batch * conv_dim;
        let input_base = batch as usize * conv_dim as usize;
        let output_base = batch as usize * value_dim as usize;
        let state = unsafe { (*conv_state_table.add(batch as usize)).add(index as usize * 3) };
        let weight = unsafe { conv_weight_bf16.add(index as usize * 4) };
        let input = unsafe { *qkv.add(input_base + index as usize) };
        let mut mixed = input * bf16_to_f32(unsafe { *weight.add(3) });
        mixed += unsafe { *state } * bf16_to_f32(unsafe { *weight });
        mixed += unsafe { *state.add(1) } * bf16_to_f32(unsafe { *weight.add(1) });
        mixed += unsafe { *state.add(2) } * bf16_to_f32(unsafe { *weight.add(2) });
        let activated = mixed * sigmoid(mixed);
        unsafe {
            *state = *state.add(1);
            *state.add(1) = *state.add(2);
            *state.add(2) = input;
        }
        if index < key_dim {
            let mut repeat = 0;
            while repeat < value_heads / key_heads {
                unsafe {
                    q.add(output_base + (repeat * key_dim + index) as usize)
                        .write(activated)
                };
                repeat += 1;
            }
        } else if index < key_dim * 2 {
            let key_index = index - key_dim;
            let mut repeat = 0;
            while repeat < value_heads / key_heads {
                unsafe {
                    k.add(output_base + (repeat * key_dim + key_index) as usize)
                        .write(activated)
                };
                repeat += 1;
            }
        } else {
            let value_index = index - key_dim * 2;
            let value_head = value_index / head_dim;
            let sub = value_index - value_head * head_dim;
            let values_per_key = value_heads / key_heads;
            let key_head = value_head / values_per_key;
            let value_subhead = value_head - key_head * values_per_key;
            let tiled = value_subhead * key_dim + key_head * head_dim + sub;
            unsafe { v.add(output_base + tiled as usize).write(activated) };
        }
    }

    /// Applies Qwen GDN convolution to ragged prompt chunks in token order.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen36_gdn_prep_chunks_f32(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
    ) {
        let sequence = thread::blockIdx_y();
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        if index >= conv_dim {
            return;
        }
        let offset = unsafe { *sequence_offsets.add(sequence as usize) };
        let length = unsafe { *sequence_lengths.add(sequence as usize) };
        let state = unsafe { (*conv_state_table.add(sequence as usize)).add(index as usize * 3) };
        let mut state0 = unsafe { *state };
        let mut state1 = unsafe { *state.add(1) };
        let mut state2 = unsafe { *state.add(2) };
        let weight = unsafe { conv_weight_bf16.add(index as usize * 4) };
        let weight0 = bf16_to_f32(unsafe { *weight });
        let weight1 = bf16_to_f32(unsafe { *weight.add(1) });
        let weight2 = bf16_to_f32(unsafe { *weight.add(2) });
        let weight3 = bf16_to_f32(unsafe { *weight.add(3) });
        let mut token = 0;
        while token < length {
            let row = offset + token;
            let input_index = row as usize * conv_dim as usize + index as usize;
            let input = unsafe { *qkv.add(input_index) };
            let mut mixed = input * weight3;
            mixed = state0.mul_add(weight0, mixed);
            mixed = state1.mul_add(weight1, mixed);
            mixed = state2.mul_add(weight2, mixed);
            let activated = mixed * sigmoid(mixed);
            state0 = state1;
            state1 = state2;
            state2 = input;
            let output_base = row as usize * value_dim as usize;
            if index < key_dim {
                let mut repeat = 0;
                while repeat < value_heads / key_heads {
                    unsafe {
                        q.add(output_base + (repeat * key_dim + index) as usize)
                            .write(activated)
                    };
                    repeat += 1;
                }
            } else if index < key_dim * 2 {
                let key_index = index - key_dim;
                let mut repeat = 0;
                while repeat < value_heads / key_heads {
                    unsafe {
                        k.add(output_base + (repeat * key_dim + key_index) as usize)
                            .write(activated)
                    };
                    repeat += 1;
                }
            } else {
                let value_index = index - key_dim * 2;
                let value_head = value_index / head_dim;
                let sub = value_index - value_head * head_dim;
                let values_per_key = value_heads / key_heads;
                let key_head = value_head / values_per_key;
                let value_subhead = value_head - key_head * values_per_key;
                let tiled = value_subhead * key_dim + key_head * head_dim + sub;
                unsafe { v.add(output_base + tiled as usize).write(activated) };
            }
            token += 1;
        }
        unsafe {
            state.write(state0);
            state.add(1).write(state1);
            state.add(2).write(state2);
        }
    }

    /// Computes token-parallel BF16 Qwen GDN prompt inputs.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_prep_chunks_bf16(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        conv_state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        sequence_count: u32,
        total_tokens: u32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
    ) {
        let linear = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        if linear >= total_tokens * conv_dim {
            return;
        }
        let row = linear / conv_dim;
        let index = linear - row * conv_dim;
        let mut sequence = 0;
        while sequence + 1 < sequence_count {
            let end = unsafe {
                *sequence_offsets.add(sequence as usize) + *sequence_lengths.add(sequence as usize)
            };
            if row < end {
                break;
            }
            sequence += 1;
        }
        let offset = unsafe { *sequence_offsets.add(sequence as usize) };
        let token = row - offset;
        let state = unsafe { (*conv_state_table.add(sequence as usize)).add(index as usize * 3) };
        let weight = unsafe { conv_weight_bf16.add(index as usize * 4) };
        let input = unsafe { *qkv.add(row as usize * conv_dim as usize + index as usize) };
        let mut mixed = input * bf16_to_f32(unsafe { *weight.add(3) });
        let mut lag = 1;
        while lag <= 3 {
            let history = if token >= lag {
                unsafe { *qkv.add((row - lag) as usize * conv_dim as usize + index as usize) }
            } else {
                unsafe { *state.add((3 + token - lag) as usize) }
            };
            mixed = history.mul_add(
                bf16_to_f32(unsafe { *weight.add((3 - lag) as usize) }),
                mixed,
            );
            lag += 1;
        }
        let encoded = convert::cvt_bf16x2_f32(mixed * sigmoid(mixed), 0.0) as u16;
        let output_base = row as usize * value_dim as usize;
        if index < key_dim {
            let mut repeat = 0;
            while repeat < value_heads / key_heads {
                unsafe {
                    q.add(output_base + (repeat * key_dim + index) as usize)
                        .write(encoded)
                };
                repeat += 1;
            }
        } else if index < key_dim * 2 {
            let key_index = index - key_dim;
            let mut repeat = 0;
            while repeat < value_heads / key_heads {
                unsafe {
                    k.add(output_base + (repeat * key_dim + key_index) as usize)
                        .write(encoded)
                };
                repeat += 1;
            }
        } else {
            let value_index = index - key_dim * 2;
            let value_head = value_index / head_dim;
            let sub = value_index - value_head * head_dim;
            let values_per_key = value_heads / key_heads;
            let key_head = value_head / values_per_key;
            let value_subhead = value_head - key_head * values_per_key;
            let tiled = value_subhead * key_dim + key_head * head_dim + sub;
            unsafe { v.add(output_base + tiled as usize).write(encoded) };
        }
    }

    /// Advances Qwen GDN convolution state after token-parallel preparation.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen36_gdn_update_conv_state(
        qkv: *const f32,
        conv_state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        conv_dim: u32,
    ) {
        let sequence = thread::blockIdx_y();
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= conv_dim {
            return;
        }
        let offset = unsafe { *sequence_offsets.add(sequence as usize) };
        let length = unsafe { *sequence_lengths.add(sequence as usize) };
        let state = unsafe { (*conv_state_table.add(sequence as usize)).add(index as usize * 3) };
        let old = unsafe { [*state, *state.add(1), *state.add(2)] };
        let mut item = 0;
        while item < 3 {
            let timeline = length + item;
            let value = if timeline < 3 {
                old[timeline as usize]
            } else {
                unsafe {
                    *qkv.add((offset + timeline - 3) as usize * conv_dim as usize + index as usize)
                }
            };
            unsafe { state.add(item as usize).write(value) };
            item += 1;
        }
    }

    /// L2-normalizes contiguous 128-wide heads in place.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn l2_norm_heads_128_f32(values: *mut f32, heads: u32) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let head = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if head >= heads || lane >= 128 {
            return;
        }
        let index = head as usize * 128 + lane as usize;
        let value = unsafe { *values.add(index) };
        unsafe { partial.add(lane as usize).write(value * value) };
        thread::sync_threads();
        let mut stride = 64;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let norm = unsafe { *partial }.sqrt().max(1.0e-6);
        unsafe { values.add(index).write(value / norm) };
    }

    /// L2-normalizes contiguous BF16 128-wide heads in place.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn l2_norm_heads_128_bf16(values: *mut u16, heads: u32) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let head = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if head >= heads || lane >= 128 {
            return;
        }
        let index = head as usize * 128 + lane as usize;
        let value = bf16_to_f32(unsafe { *values.add(index) });
        unsafe { partial.add(lane as usize).write(value * value) };
        thread::sync_threads();
        let mut stride = 64;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        let norm = unsafe { *partial }.sqrt().max(1.0e-6);
        let encoded = convert::cvt_bf16x2_f32(value / norm, 0.0) as u16;
        unsafe { values.add(index).write(encoded) };
    }

    /// Computes Qwen GDN log-decay and beta gates.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_gate_f32(
        alpha: *const f32,
        beta_input: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut f32,
        beta: *mut f32,
        heads: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= heads {
            return;
        }
        let index = index as usize;
        let a_log = bf16_to_f32(unsafe { *a_log_bf16.add(index) });
        let dt_bias = bf16_to_f32(unsafe { *dt_bias_bf16.add(index) });
        let dt = unsafe { *alpha.add(index) } + dt_bias;
        let softplus = (-dt.abs()).exp().ln_1p() + dt.max(0.0);
        unsafe {
            gate.add(index).write(-a_log.exp() * softplus);
            beta.add(index).write(sigmoid(*beta_input.add(index)));
        }
    }

    /// Computes Qwen GDN gates for separate batched alpha and beta inputs.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_gate_batch_f32(
        alpha: *const f32,
        beta_input: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut f32,
        beta: *mut f32,
        rows: u32,
        heads: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * heads {
            return;
        }
        let head = index % heads;
        let a_log = bf16_to_f32(unsafe { *a_log_bf16.add(head as usize) });
        let dt_bias = bf16_to_f32(unsafe { *dt_bias_bf16.add(head as usize) });
        let dt = unsafe { *alpha.add(index as usize) } + dt_bias;
        let softplus = (-dt.abs()).exp().ln_1p() + dt.max(0.0);
        unsafe {
            gate.add(index as usize).write(-a_log.exp() * softplus);
            beta.add(index as usize)
                .write(sigmoid(*beta_input.add(index as usize)));
        }
    }

    /// Computes Qwen GDN gates into BF16 for separate batched inputs.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_gate_batch_bf16(
        alpha: *const f32,
        beta_input: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut u16,
        beta: *mut u16,
        rows: u32,
        heads: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * heads {
            return;
        }
        let head = index % heads;
        let a_log = bf16_to_f32(unsafe { *a_log_bf16.add(head as usize) });
        let dt_bias = bf16_to_f32(unsafe { *dt_bias_bf16.add(head as usize) });
        let dt = unsafe { *alpha.add(index as usize) } + dt_bias;
        let softplus = (-dt.abs()).exp().ln_1p() + dt.max(0.0);
        let gate_value = convert::cvt_bf16x2_f32(-a_log.exp() * softplus, 0.0) as u16;
        let beta_value =
            convert::cvt_bf16x2_f32(sigmoid(unsafe { *beta_input.add(index as usize) }), 0.0)
                as u16;
        unsafe {
            gate.add(index as usize).write(gate_value);
            beta.add(index as usize).write(beta_value);
        }
    }

    /// Computes paired-projection Qwen GDN gates for an f32 row batch.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_gate_paired_batch_f32(
        alpha_beta: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut f32,
        beta: *mut f32,
        rows: u32,
        heads: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * heads {
            return;
        }
        let row = index / heads;
        let head = index - row * heads;
        let pair_offset = row as usize * heads as usize * 2;
        let a_log = bf16_to_f32(unsafe { *a_log_bf16.add(head as usize) });
        let dt_bias = bf16_to_f32(unsafe { *dt_bias_bf16.add(head as usize) });
        let dt = unsafe { *alpha_beta.add(pair_offset + head as usize) } + dt_bias;
        let softplus = (-dt.abs()).exp().ln_1p() + dt.max(0.0);
        unsafe {
            gate.add(index as usize).write(-a_log.exp() * softplus);
            beta.add(index as usize).write(sigmoid(
                *alpha_beta.add(pair_offset + heads as usize + head as usize),
            ));
        }
    }

    /// Computes paired-projection Qwen GDN gates for a BF16 row batch.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn qwen36_gdn_gate_paired_batch_bf16(
        alpha_beta: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut u16,
        beta: *mut u16,
        rows: u32,
        heads: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * heads {
            return;
        }
        let row = index / heads;
        let head = index - row * heads;
        let pair_offset = row as usize * heads as usize * 2;
        let a_log = bf16_to_f32(unsafe { *a_log_bf16.add(head as usize) });
        let dt_bias = bf16_to_f32(unsafe { *dt_bias_bf16.add(head as usize) });
        let dt = unsafe { *alpha_beta.add(pair_offset + head as usize) } + dt_bias;
        let softplus = (-dt.abs()).exp().ln_1p() + dt.max(0.0);
        let gate_value = convert::cvt_bf16x2_f32(-a_log.exp() * softplus, 0.0) as u16;
        let beta_value = convert::cvt_bf16x2_f32(
            sigmoid(unsafe { *alpha_beta.add(pair_offset + heads as usize + head as usize) }),
            0.0,
        ) as u16;
        unsafe {
            gate.add(index as usize).write(gate_value);
            beta.add(index as usize).write(beta_value);
        }
    }

    /// Updates one Qwen 128-wide gated-delta recurrent state.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn gated_delta_net_128_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        heads: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, 5> = SharedArray::UNINIT;
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let head = thread::blockIdx_x();
        let col = thread::blockIdx_y();
        let row = thread::threadIdx_x();
        if head >= heads || col >= 128 || row >= 128 {
            return;
        }
        let head_base = head as usize * 128;
        let state_index = head as usize * 128 * 128 + col as usize * 128 + row as usize;
        let q_value = unsafe { *q.add(head_base + row as usize) };
        let k_value = unsafe { *k.add(head_base + row as usize) };
        let old_state = unsafe { *state.add(state_index) };
        let lane = row & 31;
        let warp_index = row >> 5;

        let mut state_dot_k = warp_sum(old_state * k_value);
        if lane == 0 {
            unsafe { reduction.add(warp_index as usize).write(state_dot_k) };
        }
        thread::sync_threads();
        if warp_index == 0 {
            state_dot_k = warp_sum(if lane < 4 {
                unsafe { *reduction.add(lane as usize) }
            } else {
                0.0
            });
            if lane == 0 {
                unsafe { reduction.add(4).write(state_dot_k) };
            }
        }
        thread::sync_threads();

        let decay = unsafe { *gate.add(head as usize) }.exp();
        let delta = (unsafe { *v.add(head_base + col as usize) }
            - decay * unsafe { *reduction.add(4) })
            * unsafe { *beta.add(head as usize) };
        let new_state = decay * old_state + k_value * delta;
        unsafe { state.add(state_index).write(new_state) };

        let mut output_value = warp_sum(new_state * q_value);
        if lane == 0 {
            unsafe { reduction.add(warp_index as usize).write(output_value) };
        }
        thread::sync_threads();
        if warp_index == 0 {
            output_value = warp_sum(if lane < 4 {
                unsafe { *reduction.add(lane as usize) }
            } else {
                0.0
            });
            if lane == 0 {
                unsafe {
                    output
                        .add(head_base + col as usize)
                        .write(output_value * 0.088_388_35)
                };
            }
        }
    }

    /// Updates one Gated Delta Net token for every pointer-table batch row.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn gated_delta_net_128_f32_batch(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state_table: *const *mut f32,
        output: *mut f32,
        heads: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, 5> = SharedArray::UNINIT;
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let batch_head = thread::blockIdx_x();
        let batch = batch_head / heads;
        let head = batch_head - batch * heads;
        let col = thread::blockIdx_y();
        let row = thread::threadIdx_x();
        if col >= 128 || row >= 128 {
            return;
        }
        let vector_base = batch_head as usize * 128;
        let state_index = head as usize * 128 * 128 + col as usize * 128 + row as usize;
        let state = unsafe { *state_table.add(batch as usize) };
        let q_value = unsafe { *q.add(vector_base + row as usize) };
        let k_value = unsafe { *k.add(vector_base + row as usize) };
        let old_state = unsafe { *state.add(state_index) };
        let lane = row & 31;
        let warp_index = row >> 5;
        let mut state_dot_k = warp_sum(old_state * k_value);
        if lane == 0 {
            unsafe { reduction.add(warp_index as usize).write(state_dot_k) };
        }
        thread::sync_threads();
        if warp_index == 0 {
            state_dot_k = warp_sum(if lane < 4 {
                unsafe { *reduction.add(lane as usize) }
            } else {
                0.0
            });
            if lane == 0 {
                unsafe { reduction.add(4).write(state_dot_k) };
            }
        }
        thread::sync_threads();
        let decay = unsafe { *gate.add(batch_head as usize) }.exp();
        let delta = (unsafe { *v.add(vector_base + col as usize) }
            - decay * unsafe { *reduction.add(4) })
            * unsafe { *beta.add(batch_head as usize) };
        let new_state = decay * old_state + k_value * delta;
        unsafe { state.add(state_index).write(new_state) };
        let mut output_value = warp_sum(new_state * q_value);
        if lane == 0 {
            unsafe { reduction.add(warp_index as usize).write(output_value) };
        }
        thread::sync_threads();
        if warp_index == 0 {
            output_value = warp_sum(if lane < 4 {
                unsafe { *reduction.add(lane as usize) }
            } else {
                0.0
            });
            if lane == 0 {
                unsafe {
                    output
                        .add(vector_base + col as usize)
                        .write(output_value * 0.088_388_35)
                };
            }
        }
    }

    /// Updates ragged Gated Delta Net prompt chunks in token order.
    #[kernel]
    #[launch_bounds(128)]
    pub unsafe fn gated_delta_net_128_f32_chunks(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        output: *mut f32,
        heads: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, 5> = SharedArray::UNINIT;
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let sequence_head = thread::blockIdx_x();
        let sequence = sequence_head / heads;
        let head = sequence_head - sequence * heads;
        let col = thread::blockIdx_y();
        let row = thread::threadIdx_x();
        if col >= 128 || row >= 128 {
            return;
        }
        let offset = unsafe { *sequence_offsets.add(sequence as usize) };
        let length = unsafe { *sequence_lengths.add(sequence as usize) };
        let state_index = head as usize * 128 * 128 + col as usize * 128 + row as usize;
        let state = unsafe { *state_table.add(sequence as usize) };
        let lane = row & 31;
        let warp_index = row >> 5;
        let mut state_value = unsafe { *state.add(state_index) };
        let mut token = 0;
        while token < length {
            let token_head = (offset + token) * heads + head;
            let vector_base = token_head as usize * 128;
            let q_value = unsafe { *q.add(vector_base + row as usize) };
            let k_value = unsafe { *k.add(vector_base + row as usize) };
            let mut state_dot_k = warp_sum(state_value * k_value);
            if lane == 0 {
                unsafe { reduction.add(warp_index as usize).write(state_dot_k) };
            }
            thread::sync_threads();
            if warp_index == 0 {
                state_dot_k = warp_sum(if lane < 4 {
                    unsafe { *reduction.add(lane as usize) }
                } else {
                    0.0
                });
                if lane == 0 {
                    unsafe { reduction.add(4).write(state_dot_k) };
                }
            }
            thread::sync_threads();
            let decay = unsafe { *gate.add(token_head as usize) }.exp();
            let delta = (unsafe { *v.add(vector_base + col as usize) }
                - decay * unsafe { *reduction.add(4) })
                * unsafe { *beta.add(token_head as usize) };
            state_value = decay * state_value + k_value * delta;
            let mut output_value = warp_sum(state_value * q_value);
            if lane == 0 {
                unsafe { reduction.add(warp_index as usize).write(output_value) };
            }
            thread::sync_threads();
            if warp_index == 0 {
                output_value = warp_sum(if lane < 4 {
                    unsafe { *reduction.add(lane as usize) }
                } else {
                    0.0
                });
                if lane == 0 {
                    unsafe {
                        output
                            .add(vector_base + col as usize)
                            .write(output_value * 0.088_388_35)
                    };
                }
            }
            thread::sync_threads();
            token += 1;
        }
        unsafe { state.add(state_index).write(state_value) };
    }

    /// Updates long ragged Gated Delta Net chunks with eight columns per block.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn gated_delta_net_128_f32_chunks_multiwarp(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        output: *mut f32,
        heads: u32,
    ) {
        static mut QUERY: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut KEY: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut DECAY: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut BETA: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut DELTA: SharedArray<f32, 8> = SharedArray::UNINIT;
        let query_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let key_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEY) };
        let decay_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DECAY) };
        let beta_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BETA) };
        let delta_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DELTA) };
        let sequence_head = thread::blockIdx_x();
        let sequence = sequence_head / heads;
        let head = sequence_head - sequence * heads;
        let warp_index = thread::threadIdx_x() >> 5;
        let lane = thread::threadIdx_x() & 31;
        let col = thread::blockIdx_y() * 8 + warp_index;
        let offset = unsafe { *sequence_offsets.add(sequence as usize) };
        let length = unsafe { *sequence_lengths.add(sequence as usize) };
        let state_base = head as usize * 128 * 128 + col as usize * 128;
        let state = unsafe { *state_table.add(sequence as usize) };
        let mut state_values = [0.0f32; 4];
        let mut item = 0;
        while item < 4 {
            state_values[item] = unsafe { *state.add(state_base + lane as usize + item * 32) };
            item += 1;
        }
        let mut token = 0;
        while token < length {
            let token_head = (offset + token) * heads + head;
            let vector_base = token_head as usize * 128;
            if thread::threadIdx_x() < 128 {
                unsafe {
                    query_shared
                        .add(thread::threadIdx_x() as usize)
                        .write(*q.add(vector_base + thread::threadIdx_x() as usize));
                    key_shared
                        .add(thread::threadIdx_x() as usize)
                        .write(*k.add(vector_base + thread::threadIdx_x() as usize));
                }
            }
            if thread::threadIdx_x() == 0 {
                unsafe {
                    decay_shared.write((*gate.add(token_head as usize)).exp());
                    beta_shared.write(*beta.add(token_head as usize));
                }
            }
            thread::sync_threads();
            let mut state_dot_key = 0.0;
            let mut item = 0;
            while item < 4 {
                let row = lane as usize + item * 32;
                state_dot_key =
                    unsafe { state_values[item].mul_add(*key_shared.add(row), state_dot_key) };
                item += 1;
            }
            state_dot_key = warp_sum(state_dot_key);
            if lane == 0 {
                unsafe {
                    delta_shared.add(warp_index as usize).write(
                        (*v.add(vector_base + col as usize) - *decay_shared * state_dot_key)
                            * *beta_shared,
                    )
                };
            }
            thread::sync_threads();
            let delta = unsafe { *delta_shared.add(warp_index as usize) };
            let mut output_value = 0.0;
            let mut item = 0;
            while item < 4 {
                let row = lane as usize + item * 32;
                state_values[item] = unsafe {
                    (*key_shared.add(row)).mul_add(delta, *decay_shared * state_values[item])
                };
                output_value =
                    unsafe { state_values[item].mul_add(*query_shared.add(row), output_value) };
                item += 1;
            }
            output_value = warp_sum(output_value);
            if lane == 0 {
                unsafe {
                    output
                        .add(vector_base + col as usize)
                        .write(output_value * 0.088_388_35)
                };
            }
            thread::sync_threads();
            token += 1;
        }
        let mut item = 0;
        while item < 4 {
            unsafe {
                state
                    .add(state_base + lane as usize + item * 32)
                    .write(state_values[item])
            };
            item += 1;
        }
    }
}
