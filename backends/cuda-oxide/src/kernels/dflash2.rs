//! DFlash2 capture, projection, attention, and selection kernels.

use crate::common::*;
use cuda_device::{
    DynamicSharedArray, SharedArray, cuda_module, kernel, launch_bounds, launch_contract,
};
use cuda_device::thread;

/// CUDA entry points for DFlash2 execution.
#[cuda_module]
mod device {
    use super::*;

    #[inline(always)]
    fn sampling_key_id(key: u64) -> u32 {
        u32::MAX - key as u32
    }

    #[inline(always)]
    fn sampling_key(value: f32, id: u32) -> u64 {
        let bits = value.to_bits();
        let ordered = if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        };
        (u64::from(ordered) << 32) | u64::from(u32::MAX - id)
    }

    #[inline(always)]
    fn sampling_key_value(key: u64) -> f32 {
        let ordered = (key >> 32) as u32;
        let bits = if ordered & 0x8000_0000 != 0 {
            ordered ^ 0x8000_0000
        } else {
            !ordered
        };
        f32::from_bits(bits)
    }

    #[inline(always)]
    fn total_order_key(value: f32) -> i32 {
        let mut bits = value.to_bits() as i32;
        bits ^= (bits >> 31) & i32::MAX;
        bits
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SamplingParams {
        temperature: f32,
        top_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        draw: f32,
        top_k: u32,
        token_counts: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SamplingResult {
        id: u32,
        logit: f32,
        adjusted_logit: f32,
        status: u32,
    }

    /// Gathers equal-sized f32 rows through a device pointer table.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn gather_f32_pointer_rows(
        input_table: *const *mut f32,
        output: *mut f32,
        row_values: u32,
    ) {
        let row = thread::blockIdx_y();
        let value = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if value < row_values {
            let input = unsafe { *input_table.add(row as usize) };
            unsafe {
                output
                    .add(row as usize * row_values as usize + value as usize)
                    .write(*input.add(value as usize))
            };
        }
    }

    /// Scatters equal-sized f32 rows through a device pointer table.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn scatter_f32_pointer_rows(
        input: *const f32,
        output_table: *const *mut f32,
        row_values: u32,
    ) {
        let row = thread::blockIdx_y();
        let value = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if value < row_values {
            let output = unsafe { *output_table.add(row as usize) };
            unsafe {
                output
                    .add(value as usize)
                    .write(*input.add(row as usize * row_values as usize + value as usize))
            };
        }
    }

    /// Captures one target residual tap into `[row, tap, hidden]` storage.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn dflash2_capture_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        hidden: u32,
        taps: u32,
        tap: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * hidden {
            return;
        }
        let row = index / hidden;
        let col = index - row * hidden;
        let output_index =
            (row as usize * taps as usize + tap as usize) * hidden as usize + col as usize;
        unsafe { output.add(output_index).write(*input.add(index as usize)) };
    }

    /// Applies one side of DFlash2 dynamic grouped convolution.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn dflash2_grouped_conv_f32(
        input: *const f32,
        coefficients: *const f32,
        base: *const f32,
        output: *mut f32,
        rows: u32,
        hidden: u32,
        groups: u32,
        taps: u32,
        block_size: u32,
        side: u32,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index >= rows * hidden {
            return;
        }
        let row = index / hidden;
        let channel = index - row * hidden;
        let group_size = hidden / groups;
        let group = channel / group_size;
        let available = taps.min(row % block_size + 1);
        let mut value = 0.0;
        let mut tap = 0;
        while tap < available {
            let base_index =
                (side as usize * taps as usize + tap as usize) * hidden as usize + channel as usize;
            let coefficient_index = row as usize * 2 * taps as usize * groups as usize
                + (side as usize * taps as usize + tap as usize) * groups as usize
                + group as usize;
            value = unsafe {
                (*base.add(base_index) + *coefficients.add(coefficient_index)).mul_add(
                    *input.add((row - tap) as usize * hidden as usize + channel as usize),
                    value,
                )
            };
            tap += 1;
        }
        unsafe { output.add(index as usize).write(value) };
    }

    /// Evaluates DFlash2 non-causal proposal attention over its ring cache.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn dflash2_noncausal_attention_f32(
        query: *const f32,
        context_key: *const f32,
        context_value: *const f32,
        block_key: *const f32,
        block_value: *const f32,
        output: *mut f32,
        context_end: u32,
        context_len: u32,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window: u32,
    ) {
        let shared = DynamicSharedArray::<f32>::get();
        let query_shared = shared;
        let scores = unsafe { shared.add(head_dim as usize) };
        let reduction = unsafe { scores.add(thread::blockDim_x() as usize) };
        let query_row = thread::blockIdx_y();
        let query_head = thread::blockIdx_x();
        if query_row >= rows || query_head >= q_heads {
            return;
        }
        let groups_per_kv = q_heads / kv_heads;
        let kv_head = query_head / groups_per_kv;
        let scale = cuda_device::float::rsqrt_approx_f32(head_dim as f32);
        let query_base =
            unsafe { query.add(((query_row * q_heads + query_head) * head_dim) as usize) };
        let lane = thread::threadIdx_x();
        let mut dim = lane;
        while dim < head_dim {
            unsafe {
                query_shared
                    .add(dim as usize)
                    .write(*query_base.add(dim as usize))
            };
            dim += thread::blockDim_x();
        }
        thread::sync_threads();

        let sequence_end = context_end + rows;
        let first_key = sequence_end.saturating_sub(window);
        let retained_start = context_end - context_len;
        let context_start = retained_start.max(first_key);
        let context_rows = context_end - context_start;
        let key_count = context_rows + rows;
        let mut running_max = f32::NEG_INFINITY;
        let mut running_total = 0.0;
        let mut accumulator = 0.0;
        let mut tile_start = 0;
        while tile_start < key_count {
            let key_index = tile_start + lane;
            let mut score = f32::NEG_INFINITY;
            if key_index < key_count {
                let from_context = key_index < context_rows;
                let logical_position = context_start + key_index;
                let row = if from_context {
                    logical_position % window
                } else {
                    key_index - context_rows
                };
                let key_base = if from_context { context_key } else { block_key };
                let key = unsafe { key_base.add(((row * kv_heads + kv_head) * head_dim) as usize) };
                let mut dot = 0.0;
                let mut component = 0;
                while component < head_dim {
                    dot = unsafe {
                        (*query_shared.add(component as usize))
                            .mul_add(*key.add(component as usize), dot)
                    };
                    component += 1;
                }
                score = dot * scale;
            }
            unsafe {
                scores.add(lane as usize).write(score);
                reduction.add(lane as usize).write(score);
            }
            thread::sync_threads();
            let mut stride = thread::blockDim_x() / 2;
            while stride != 0 {
                if lane < stride {
                    unsafe {
                        reduction.add(lane as usize).write(
                            (*reduction.add(lane as usize))
                                .max(*reduction.add((lane + stride) as usize)),
                        )
                    };
                }
                thread::sync_threads();
                stride /= 2;
            }
            let tile_max = unsafe { *reduction };
            let weight = if key_index < key_count {
                (score - tile_max).exp()
            } else {
                0.0
            };
            unsafe {
                scores.add(lane as usize).write(weight);
                reduction.add(lane as usize).write(weight);
            }
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
                let merged_max = running_max.max(tile_max);
                unsafe {
                    reduction.add(1).write(if running_max.is_finite() {
                        (running_max - merged_max).exp()
                    } else {
                        0.0
                    });
                    reduction.add(2).write((tile_max - merged_max).exp());
                    reduction
                        .add(3)
                        .write(running_total * *reduction.add(1) + *reduction * *reduction.add(2));
                    reduction.add(4).write(merged_max);
                }
            }
            thread::sync_threads();
            if lane < head_dim {
                let mut tile_accumulator = 0.0;
                let tile_rows = thread::blockDim_x().min(key_count - tile_start);
                let mut item = 0;
                while item < tile_rows {
                    let absolute_index = tile_start + item;
                    let from_context = absolute_index < context_rows;
                    let logical_position = context_start + absolute_index;
                    let row = if from_context {
                        logical_position % window
                    } else {
                        absolute_index - context_rows
                    };
                    let value_base = if from_context {
                        context_value
                    } else {
                        block_value
                    };
                    let value =
                        unsafe { value_base.add(((row * kv_heads + kv_head) * head_dim) as usize) };
                    tile_accumulator = unsafe {
                        (*scores.add(item as usize))
                            .mul_add(*value.add(lane as usize), tile_accumulator)
                    };
                    item += 1;
                }
                accumulator = unsafe {
                    accumulator * *reduction.add(1) + tile_accumulator * *reduction.add(2)
                };
            }
            if lane == 0 {
                running_total = unsafe { *reduction.add(3) };
                running_max = unsafe { *reduction.add(4) };
            }
            thread::sync_threads();
            tile_start += thread::blockDim_x();
        }
        if lane == 0 {
            unsafe { reduction.write(running_total) };
        }
        thread::sync_threads();
        if lane < head_dim {
            let output_index = ((query_row * q_heads + query_head) * head_dim + lane) as usize;
            unsafe { output.add(output_index).write(accumulator / *reduction) };
        }
    }

    /// Projects DFlash2 hidden rows through its row-major BF16 matrix.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn dflash2_hidden_projection_f32(
        hidden: *const f32,
        weight_bf16: *const u16,
        projected: *mut f32,
        hidden_size: u32,
        rank: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let output_index = thread::blockIdx_x();
        let row = output_index / rank;
        let component = output_index - row * rank;
        let input = unsafe { hidden.add(row as usize * hidden_size as usize) };
        let weight = unsafe { weight_bf16.add(component as usize * hidden_size as usize) };
        let lane = thread::threadIdx_x();
        let mut sum = 0.0;
        let mut feature = lane;
        while feature < hidden_size {
            sum = unsafe {
                (*input.add(feature as usize))
                    .mul_add(bf16_to_f32(*weight.add(feature as usize)), sum)
            };
            feature += thread::blockDim_x();
        }
        unsafe { partial.add(lane as usize).write(sum) };
        thread::sync_threads();
        let mut stride = thread::blockDim_x() / 2;
        while stride != 0 {
            if lane < stride {
                unsafe { *partial.add(lane as usize) += *partial.add((lane + stride) as usize) };
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            unsafe { projected.add(output_index as usize).write(*partial) };
        }
    }

    /// Reduces 1,024 logits to 32 ordered sampling keys per block.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn sampling_logits_topk_f32(
        logits: *const f32,
        params: *const SamplingParams,
        output_keys: *mut u64,
        vocab: u32,
        chunks_per_row: u32,
    ) {
        static mut KEYS: SharedArray<u64, 1024> = SharedArray::UNINIT;
        let keys = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEYS) };
        let block = thread::blockIdx_x();
        let row = block / chunks_per_row;
        let chunk = block - row * chunks_per_row;
        let lane = thread::threadIdx_x();
        let chunk_start = chunk * 1024;
        let row_logits = unsafe { logits.add(row as usize * vocab as usize) };
        let config = if params.is_null() {
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                draw: 0.0,
                top_k: 32,
                token_counts: 0,
            }
        } else {
            unsafe { *params.add(row as usize) }
        };
        let counts = config.token_counts as *const u32;
        let mut item = 0;
        while item < 4 {
            let local = lane + item * 256;
            let token = chunk_start + local;
            let key = if token < vocab {
                let value = unsafe { *row_logits.add(token as usize) };
                if value.is_finite() {
                    let count = if counts.is_null() {
                        0
                    } else {
                        unsafe { *counts.add(token as usize) }
                    };
                    let adjusted = value
                        - if count == 0 {
                            0.0
                        } else {
                            config.presence_penalty
                        }
                        - config.frequency_penalty * count as f32;
                    sampling_key(adjusted, token)
                } else {
                    0
                }
            } else {
                0
            };
            unsafe { keys.add(local as usize).write(key) };
            item += 1;
        }
        thread::sync_threads();

        let mut width = 2;
        while width <= 1024 {
            let mut distance = width / 2;
            while distance != 0 {
                let mut item = 0;
                while item < 4 {
                    let index = lane + item * 256;
                    let partner = index ^ distance;
                    if partner > index {
                        let left = unsafe { *keys.add(index as usize) };
                        let right = unsafe { *keys.add(partner as usize) };
                        let descending = index & width == 0;
                        if (descending && left < right) || (!descending && left > right) {
                            unsafe {
                                keys.add(index as usize).write(right);
                                keys.add(partner as usize).write(left);
                            }
                        }
                    }
                    item += 1;
                }
                thread::sync_threads();
                distance /= 2;
            }
            width *= 2;
        }

        if lane < 32 {
            let output = (row * chunks_per_row + chunk) * 32 + lane;
            unsafe {
                output_keys
                    .add(output as usize)
                    .write(*keys.add(lane as usize))
            };
        }
    }

    /// Selects one token per row from ordered sampling keys.
    #[kernel]
    #[launch_bounds(32)]
    pub unsafe fn sampling_finalize_f32(
        logits: *const f32,
        params: *const SamplingParams,
        top_keys: *const u64,
        results: *mut SamplingResult,
        vocab: u32,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let row = thread::blockIdx_x();
        let config = unsafe { *params.add(row as usize) };
        let keys = unsafe { top_keys.add(row as usize * 32) };
        let row_logits = unsafe { logits.add(row as usize * vocab as usize) };
        let first = unsafe { *keys };
        if first == 0 {
            unsafe {
                results.add(row as usize).write(SamplingResult {
                    id: u32::MAX,
                    logit: 0.0,
                    adjusted_logit: 0.0,
                    status: 1,
                })
            };
            return;
        }

        let k = if config.temperature == 0.0 {
            1
        } else {
            config.top_k
        };
        let mut selected_slot = 0;
        if config.temperature != 0.0 {
            let mut weights = [0.0f32; 32];
            let best_value = sampling_key_value(first);
            let mut total = 0.0;
            let mut slot = 0;
            while slot < k {
                let key = unsafe { *keys.add(slot as usize) };
                if key == 0 {
                    break;
                }
                let weight = ((sampling_key_value(key) - best_value) / config.temperature).exp();
                weights[slot as usize] = weight;
                total += weight;
                slot += 1;
            }
            if !total.is_finite() || total <= 0.0 {
                unsafe {
                    results.add(row as usize).write(SamplingResult {
                        id: u32::MAX,
                        logit: 0.0,
                        adjusted_logit: 0.0,
                        status: 2,
                    })
                };
                return;
            }
            let mut cumulative = 0.0;
            let mut retained = 0;
            while retained < k {
                let key = unsafe { *keys.add(retained as usize) };
                if key == 0 {
                    break;
                }
                cumulative += weights[retained as usize] / total;
                retained += 1;
                if cumulative >= config.top_p {
                    break;
                }
            }
            let mut retained_weight = 0.0;
            let mut slot = 0;
            while slot < retained {
                retained_weight += weights[slot as usize];
                slot += 1;
            }
            let bounded_draw = config.draw.max(0.0).min(f32::from_bits(0x3f7f_ffff));
            let mut draw = bounded_draw * retained_weight;
            selected_slot = retained - 1;
            let mut slot = 0;
            while slot < retained {
                if draw < weights[slot as usize] {
                    selected_slot = slot;
                    break;
                }
                draw -= weights[slot as usize];
                slot += 1;
            }
        }

        let selected_key = unsafe { *keys.add(selected_slot as usize) };
        let id = sampling_key_id(selected_key);
        unsafe {
            results.add(row as usize).write(SamplingResult {
                id,
                logit: *row_logits.add(id as usize),
                adjusted_logit: sampling_key_value(selected_key),
                status: 0,
            })
        };
        let counts = config.token_counts as *mut u32;
        if !counts.is_null() {
            unsafe { *counts.add(id as usize) += 1 };
        }
    }

    /// Reduces 1,024 existing sampling keys to 32 ordered keys per block.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn sampling_keys_topk(
        input_keys: *const u64,
        output_keys: *mut u64,
        input_count_per_row: u32,
        output_chunks_per_row: u32,
    ) {
        static mut KEYS: SharedArray<u64, 1024> = SharedArray::UNINIT;
        let keys = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEYS) };
        let block = thread::blockIdx_x();
        let row = block / output_chunks_per_row;
        let chunk = block - row * output_chunks_per_row;
        let lane = thread::threadIdx_x();
        let chunk_start = chunk * 1024;
        let row_keys = unsafe { input_keys.add(row as usize * input_count_per_row as usize) };
        let mut item = 0;
        while item < 4 {
            let local = lane + item * 256;
            let index = chunk_start + local;
            let key = if index < input_count_per_row {
                unsafe { *row_keys.add(index as usize) }
            } else {
                0
            };
            unsafe { keys.add(local as usize).write(key) };
            item += 1;
        }
        thread::sync_threads();

        let mut width = 2;
        while width <= 1024 {
            let mut distance = width / 2;
            while distance != 0 {
                let mut item = 0;
                while item < 4 {
                    let index = lane + item * 256;
                    let partner = index ^ distance;
                    if partner > index {
                        let left = unsafe { *keys.add(index as usize) };
                        let right = unsafe { *keys.add(partner as usize) };
                        let descending = index & width == 0;
                        if (descending && left < right) || (!descending && left > right) {
                            unsafe {
                                keys.add(index as usize).write(right);
                                keys.add(partner as usize).write(left);
                            }
                        }
                    }
                    item += 1;
                }
                thread::sync_threads();
                distance /= 2;
            }
            width *= 2;
        }

        if lane < 32 {
            let output = (row * output_chunks_per_row + chunk) * 32 + lane;
            unsafe {
                output_keys
                    .add(output as usize)
                    .write(*keys.add(lane as usize))
            };
        }
    }

    /// Selects a coherent DFlash2 draft path from per-row top-k candidates.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn dflash2_select_path_f32(
        projected: *const f32,
        top_keys: *const u64,
        predecessor_codebook_bf16: *const u16,
        successor_codebook_bf16: *const u16,
        output_tokens: *mut u32,
        anchor_token: u32,
        drafts: u32,
        rank: u32,
        top_k: u32,
        key_stride: u32,
    ) {
        static mut TERMS: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut CANDIDATE_SCORES: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut CANDIDATE_TOKENS: SharedArray<u32, 32> = SharedArray::UNINIT;
        static mut PREDECESSOR: SharedArray<u32, 1> = SharedArray::UNINIT;
        let terms = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TERMS) };
        let candidate_scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATE_SCORES) };
        let candidate_tokens = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATE_TOKENS) };
        let predecessor = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PREDECESSOR) };
        let lane = thread::threadIdx_x();
        if lane == 0 {
            unsafe {
                predecessor.write(anchor_token);
                output_tokens.write(anchor_token);
            }
        }
        thread::sync_threads();

        let lanes_per_candidate = thread::blockDim_x() / top_k;
        let candidate_slot = lane / lanes_per_candidate;
        let candidate_lane = lane - candidate_slot * lanes_per_candidate;
        let mut step = 0;
        while step < drafts {
            let hidden = unsafe { projected.add(step as usize * rank as usize) };
            let predecessor_code =
                unsafe { predecessor_codebook_bf16.add(*predecessor as usize * rank as usize) };
            let candidates = unsafe { top_keys.add(step as usize * key_stride as usize) };
            if candidate_slot < top_k {
                let candidate_key = unsafe { *candidates.add(candidate_slot as usize) };
                if candidate_key != 0 {
                    let candidate = sampling_key_id(candidate_key);
                    let successor_code =
                        unsafe { successor_codebook_bf16.add(candidate as usize * rank as usize) };
                    let mut component = candidate_lane;
                    while component < rank {
                        let predecessor_value =
                            bf16_to_f32(unsafe { *predecessor_code.add(component as usize) });
                        let successor_value =
                            bf16_to_f32(unsafe { *successor_code.add(component as usize) });
                        unsafe {
                            terms
                                .add(candidate_slot as usize * 256 + component as usize)
                                .write(
                                    predecessor_value
                                        * *hidden.add(component as usize)
                                        * successor_value,
                                )
                        };
                        component += lanes_per_candidate;
                    }
                }
            }
            thread::sync_threads();

            if lane < top_k {
                let candidate_key = unsafe { *candidates.add(lane as usize) };
                if candidate_key != 0 {
                    let mut transition = 0.0;
                    let mut component = 0;
                    while component < rank {
                        transition +=
                            unsafe { *terms.add(lane as usize * 256 + component as usize) };
                        component += 1;
                    }
                    unsafe {
                        candidate_scores
                            .add(lane as usize)
                            .write(sampling_key_value(candidate_key) + transition);
                        candidate_tokens
                            .add(lane as usize)
                            .write(sampling_key_id(candidate_key));
                    }
                } else {
                    unsafe {
                        candidate_scores.add(lane as usize).write(f32::NEG_INFINITY);
                        candidate_tokens.add(lane as usize).write(0);
                    }
                }
            }
            thread::sync_threads();

            if lane == 0 {
                let mut found = false;
                let mut best_token = 0;
                let mut best_score = f32::NEG_INFINITY;
                let mut best_key = total_order_key(best_score);
                let mut slot = 0;
                while slot < top_k {
                    let candidate_key = unsafe { *candidates.add(slot as usize) };
                    if candidate_key != 0 {
                        let score = unsafe { *candidate_scores.add(slot as usize) };
                        let candidate = unsafe { *candidate_tokens.add(slot as usize) };
                        let score_key = total_order_key(score);
                        if !found
                            || score_key > best_key
                            || (score.to_bits() == best_score.to_bits() && candidate < best_token)
                        {
                            found = true;
                            best_token = candidate;
                            best_score = score;
                            best_key = score_key;
                        }
                    }
                    slot += 1;
                }
                let selected = if found { best_token } else { 0 };
                unsafe {
                    output_tokens.add(step as usize + 1).write(selected);
                    predecessor.write(selected);
                }
            }
            thread::sync_threads();
            step += 1;
        }
    }
}
