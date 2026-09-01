//! SM121 NVFP4 matrix operations.

use crate::common::*;
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{
    DynamicSharedArray, SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract,
};
use cuda_device::{ptx_asm, thread, warp};

/// CUDA entry points for SM121 NVFP4 matrix operations.
#[cuda_module]
mod device {
    use super::*;

    #[inline(always)]
    unsafe fn nvfp4_row_dot_warp(
        packed_row: *const u8,
        row_scale: *const u8,
        input: *const f32,
        cols: usize,
    ) -> f32 {
        let lane = warp::lane_id() as usize;
        let mut acc = 0.0f32;
        let mut col = lane * 4;
        while col < cols {
            let (b0, b1, scale, x0, x1, x2, x3) = unsafe {
                (
                    *packed_row.add(col / 2),
                    *packed_row.add(col / 2 + 1),
                    e4m3_value(*row_scale.add(col / 16)),
                    *input.add(col),
                    *input.add(col + 1),
                    *input.add(col + 2),
                    *input.add(col + 3),
                )
            };
            acc = x0.mul_add(e2m1_value(b0 & 0x0f) * scale, acc);
            acc = x1.mul_add(e2m1_value(b0 >> 4) * scale, acc);
            acc = x2.mul_add(e2m1_value(b1 & 0x0f) * scale, acc);
            acc = x3.mul_add(e2m1_value(b1 >> 4) * scale, acc);
            col += LANES * 4;
        }
        acc += warp::shuffle_xor_f32(acc, 16);
        acc += warp::shuffle_xor_f32(acc, 8);
        acc += warp::shuffle_xor_f32(acc, 4);
        acc += warp::shuffle_xor_f32(acc, 2);
        acc + warp::shuffle_xor_f32(acc, 1)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn routed_w4a16<const WARPS: usize, const FIXED_TOP_K: usize, const WRITE_F32: bool>(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
        partial: *mut f32,
    ) {
        let lane = warp::lane_id() as usize;
        let warp_index = thread::threadIdx_x() as usize / LANES;
        let route = thread::blockIdx_y() as usize;
        let out_tile = thread::blockIdx_x() as usize;
        let routes = if FIXED_TOP_K == 0 {
            batch_size as usize * top_k as usize
        } else {
            FIXED_TOP_K
        };
        if route >= routes || out_tile * TILE_M >= out_features as usize {
            return;
        }

        let expert = unsafe { *indices.add(route) } as usize;
        let input_row = if FIXED_TOP_K == 0 {
            route / top_k as usize
        } else {
            0
        };
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        let k_tiles = in_features / TILE_K;
        let expert_weight_stride = out_features * in_features / 2;
        let expert_scale_stride = out_features * in_features / TILE_K;
        let expert_weight = unsafe { tiled_weight.add(expert * expert_weight_stride) };
        let expert_scales = unsafe { tiled_scales.add(expert * expert_scale_stride) };
        let global_scale = unsafe { *global_scales.add(expert) };

        let row0 = lane >> 2;
        let row1 = row0 + 8;
        let pair0 = (lane & 3) * 2;
        let pair1 = pair0 + 8;
        let mut d = [0.0f32; 4];

        let mut k_tile = warp_index;
        while k_tile < k_tiles {
            let tile = out_tile * k_tiles + k_tile;
            let tile_weight = unsafe { expert_weight.add(tile * PACKED_TILE_BYTES) };
            let tile_scales = unsafe { expert_scales.add(tile * SCALE_TILE_BYTES) };

            let mut scale0 = 0.0f32;
            let mut scale1 = 0.0f32;
            if lane & 3 == 0 {
                unsafe {
                    scale0 = e4m3_value(*tile_scales.add(row0)) * global_scale;
                    scale1 = e4m3_value(*tile_scales.add(row1)) * global_scale;
                }
            }
            scale0 = warp::shuffle_f32(scale0, (row0 * 4) as u32);
            scale1 = warp::shuffle_f32(scale1, (row0 * 4) as u32);

            let weight_row0 = unsafe { tile_weight.add(row0 * (TILE_K / 2)) };
            let weight_row1 = unsafe { tile_weight.add(row1 * (TILE_K / 2)) };
            let (a0, a1, a2, a3) = unsafe {
                (
                    dequant_bf16_pair(*weight_row0.add(pair0 / 2), scale0),
                    dequant_bf16_pair(*weight_row1.add(pair0 / 2), scale1),
                    dequant_bf16_pair(*weight_row0.add(pair1 / 2), scale0),
                    dequant_bf16_pair(*weight_row1.add(pair1 / 2), scale1),
                )
            };

            let mut b0 = 0u32;
            let mut b1 = 0u32;
            if lane < 4 {
                let input_tile = unsafe { input.add(input_row * in_features + k_tile * TILE_K) };
                unsafe {
                    b0 =
                        convert::cvt_bf16x2_f32(*input_tile.add(pair0), *input_tile.add(pair0 + 1));
                    b1 =
                        convert::cvt_bf16x2_f32(*input_tile.add(pair1), *input_tile.add(pair1 + 1));
                }
            }
            b0 = warp::shuffle(b0, (lane & 3) as u32);
            b1 = warp::shuffle(b1, (lane & 3) as u32);

            let n0: f32;
            let n1: f32;
            let n2: f32;
            let n3: f32;
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
                     {%0, %1, %2, %3}, \
                     {%4, %5, %6, %7}, \
                     {%8, %9}, \
                     {%10, %11, %12, %13};",
                    out("=f") n0,
                    out("=f") n1,
                    out("=f") n2,
                    out("=f") n3,
                    in("r") a0,
                    in("r") a1,
                    in("r") a2,
                    in("r") a3,
                    in("r") b0,
                    in("r") b1,
                    in("f") d[0],
                    in("f") d[1],
                    in("f") d[2],
                    in("f") d[3],
                    options(register_only),
                );
            }
            d = [n0, n1, n2, n3];
            k_tile += WARPS;
        }

        if lane & 3 == 0 {
            unsafe {
                partial.add(warp_index * TILE_M + row0).write(d[0]);
                partial.add(warp_index * TILE_M + row1).write(d[2]);
            }
        }
        thread::sync_threads();

        let thread_index = thread::threadIdx_x() as usize;
        if thread_index < TILE_M {
            let mut value = 0.0f32;
            let mut partial_index = 0;
            while partial_index < WARPS {
                value += unsafe { *partial.add(partial_index * TILE_M + thread_index) };
                partial_index += 1;
            }
            let packed = convert::cvt_bf16x2_f32(value, 0.0);
            let output_value = packed as u16;
            let rounded = convert::cvt_f32_bf16x2_lo(packed);
            let output_index = route * out_features + out_tile * TILE_M + thread_index;
            unsafe {
                *output_bf16.add(output_index) = output_value;
                if WRITE_F32 {
                    *output_f32.add(output_index) = rounded;
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn routed_single<const FIXED_TOP_K: usize, const WRITE_F32: bool>(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        unsafe {
            routed_w4a16::<SINGLE_WARPS, FIXED_TOP_K, WRITE_F32>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
                partial,
            );
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn routed_batch<const WRITE_F32: bool>(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        static mut PARTIAL: SharedArray<f32, 128> = SharedArray::UNINIT;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        unsafe {
            routed_w4a16::<BATCH_WARPS, 0, WRITE_F32>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
                partial,
            );
        }
    }

    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain = 2, coordinates = u32, block = (512, 1, 1))]
    pub unsafe fn w4a16_single_bf16(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_single::<0, false>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain = 2, coordinates = u32, block = (512, 1, 1))]
    pub unsafe fn w4a16_single_f32(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_single::<0, true>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain = 2, coordinates = u32, block = (512, 1, 1))]
    pub unsafe fn w4a16_top8_bf16(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_single::<8, false>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain = 2, coordinates = u32, block = (512, 1, 1))]
    pub unsafe fn w4a16_top8_f32(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_single::<8, true>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain = 2, coordinates = u32, block = (512, 1, 1))]
    pub unsafe fn w4a16_top10_bf16(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_single::<10, false>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain = 2, coordinates = u32, block = (512, 1, 1))]
    pub unsafe fn w4a16_top10_f32(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_single::<10, true>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn w4a16_batch_bf16(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_batch::<false>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn w4a16_batch_f32(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
        out_features: u32,
        in_features: u32,
    ) {
        unsafe {
            routed_batch::<true>(
                indices,
                input,
                tiled_weight,
                tiled_scales,
                global_scales,
                output_bf16,
                output_f32,
                batch_size,
                top_k,
                out_features,
                in_features,
            );
        }
    }

    /// Sorts route IDs by expert and builds compact groups of at most 16 rows.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub unsafe fn w4a4_build_route_groups(
        indices: *const u32,
        sorted_routes: *mut u32,
        group_experts: *mut u32,
        group_starts: *mut u32,
        group_lengths: *mut u32,
        group_count: *mut u32,
        routes: u32,
        experts: u32,
    ) {
        static mut COUNTS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut CURSORS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        let counts = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COUNTS) };
        let cursors = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CURSORS) };
        let thread_index = thread::threadIdx_x();
        let mut expert = thread_index;
        while expert < experts {
            unsafe { counts.add(expert as usize).write(0) };
            expert += thread::blockDim_x();
        }
        thread::sync_threads();

        let mut route = thread_index;
        while route < routes {
            let expert = unsafe { *indices.add(route as usize) };
            if expert < experts {
                unsafe {
                    BlockAtomicU32::from_ptr(counts.add(expert as usize))
                        .fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
            route += thread::blockDim_x();
        }
        thread::sync_threads();

        if thread_index == 0 {
            let mut offset = 0u32;
            expert = 0;
            while expert < experts {
                unsafe { cursors.add(expert as usize).write(offset) };
                offset += unsafe { *counts.add(expert as usize) };
                expert += 1;
            }
            route = 0;
            while route < routes {
                let expert = unsafe { *indices.add(route as usize) };
                if expert < experts {
                    let cursor = unsafe { *cursors.add(expert as usize) };
                    unsafe {
                        sorted_routes.add(cursor as usize).write(route);
                        cursors.add(expert as usize).write(cursor + 1);
                    }
                }
                route += 1;
            }
            let mut groups = 0u32;
            let mut start = 0u32;
            expert = 0;
            while expert < experts {
                let count = unsafe { *counts.add(expert as usize) };
                let mut consumed = 0u32;
                while consumed < count {
                    let length = (count - consumed).min(16);
                    unsafe {
                        group_experts.add(groups as usize).write(expert);
                        group_starts.add(groups as usize).write(start + consumed);
                        group_lengths.add(groups as usize).write(length);
                    }
                    groups += 1;
                    consumed += length;
                }
                start += count;
                expert += 1;
            }
            unsafe { group_count.write(groups) };
        }
    }

    /// Quantizes sorted route groups into native SM121 FP4 A fragments.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn w4a4_quantize_route_groups_f32(
        input: *const f32,
        sorted_routes: *const u32,
        group_starts: *const u32,
        group_lengths: *const u32,
        group_count: *const u32,
        tiles: *mut u8,
        scales: *mut u32,
        top_k: u32,
        in_features: u32,
        workers: u32,
    ) {
        static mut SCALE_CODES: SharedArray<u8, 64> = SharedArray::UNINIT;
        let scale_codes = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE_CODES) };
        let k_tile = thread::blockIdx_x();
        let worker = thread::blockIdx_y();
        let thread_index = thread::threadIdx_x();
        let k_tiles = in_features / 64;
        let groups = unsafe { *group_count };
        let mut group = worker;
        while group < groups {
            let start = unsafe { *group_starts.add(group as usize) };
            let length = unsafe { *group_lengths.add(group as usize) };
            if thread_index < 64 {
                let row = thread_index / 4;
                let k_block = thread_index & 3;
                let mut maximum = 0.0f32;
                if row < length {
                    let route = unsafe { *sorted_routes.add((start + row) as usize) };
                    let input_row = route / top_k;
                    let mut offset = 0u32;
                    while offset < 16 {
                        let value = unsafe {
                            *input.add(
                                (input_row * in_features + k_tile * 64 + k_block * 16 + offset)
                                    as usize,
                            )
                        };
                        if value.is_finite() {
                            maximum = maximum.max(value.abs());
                        }
                        offset += 1;
                    }
                }
                unsafe {
                    scale_codes
                        .add(thread_index as usize)
                        .write(if maximum == 0.0 {
                            0
                        } else {
                            ue4m3_code(maximum / 6.0)
                        });
                }
            }
            thread::sync_threads();
            let tile = unsafe { tiles.add(((group * k_tiles + k_tile) * 512) as usize) };
            let mut byte = thread_index;
            while byte < 512 {
                let mut packed = 0u8;
                let mut nibble = 0u32;
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
                    let value = if row < length {
                        let route = unsafe { *sorted_routes.add((start + row) as usize) };
                        let input_row = route / top_k;
                        unsafe {
                            *input.add((input_row * in_features + k_tile * 64 + col) as usize)
                        }
                    } else {
                        0.0
                    };
                    let scale =
                        e4m3_value(unsafe { *scale_codes.add((row * 4 + col / 16) as usize) });
                    packed |=
                        e2m1_code(if scale == 0.0 { 0.0 } else { value / scale }) << (nibble * 4);
                    nibble += 1;
                }
                unsafe { tile.add(byte as usize).write(packed) };
                byte += thread::blockDim_x();
            }
            if thread_index < 16 {
                unsafe {
                    scales
                        .add(((group * k_tiles + k_tile) * 16 + thread_index) as usize)
                        .write(scale_word(scale_codes.add((thread_index * 4) as usize)));
                }
            }
            thread::sync_threads();
            group += workers;
        }
    }

    /// Runs grouped SM121 W4A4 GEMM and restores route-major output order.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 1, 1))]
    pub unsafe fn w4a4_route_groups_f32(
        sorted_routes: *const u32,
        group_experts: *const u32,
        group_starts: *const u32,
        group_lengths: *const u32,
        group_count: *const u32,
        input_tiles: *const u8,
        input_scales: *const u32,
        tiled_weight: *const u8,
        tiled_scales: *const u8,
        global_scales: *const f32,
        output: *mut f32,
        out_features: u32,
        in_features: u32,
        workers: u32,
    ) {
        let out_tile8 = thread::blockIdx_x();
        let worker = thread::blockIdx_y();
        let lane = warp::lane_id();
        let k_tiles16 = in_features / 16;
        let k_tiles64 = in_features / 64;
        let out_tiles16 = out_features / 16;
        let expert_weight_stride = out_features * in_features / 2;
        let expert_scale_stride = out_features * in_features / 16;
        let groups = unsafe { *group_count };
        let mut group = worker;
        while group < groups {
            let expert = unsafe { *group_experts.add(group as usize) };
            let start = unsafe { *group_starts.add(group as usize) };
            let length = unsafe { *group_lengths.add(group as usize) };
            let expert_weight =
                unsafe { tiled_weight.add((expert * expert_weight_stride) as usize) };
            let expert_scales =
                unsafe { tiled_scales.add((expert * expert_scale_stride) as usize) };
            let mut accumulators = [0.0f32; 4];
            let output_col = out_tile8 * 8 + lane / 4;
            let out_tile16 = output_col / 16;
            let output_row = output_col & 15;
            let t0 = lane & 3;
            let mut k_tile64 = 0u32;
            while k_tile64 < k_tiles64 {
                let input_tile =
                    unsafe { input_tiles.add(((group * k_tiles64 + k_tile64) * 512) as usize) };
                let input_lane = unsafe { input_tile.add((lane * 16) as usize) };
                let a = unsafe {
                    [
                        load_u32(input_lane, 0),
                        load_u32(input_lane, 1),
                        load_u32(input_lane, 2),
                        load_u32(input_lane, 3),
                    ]
                };
                let first_k16 = k_tile64 * 4 + t0 / 2;
                let second_k16 = first_k16 + 2;
                let half = (t0 & 1) * 4;
                let first_tile = unsafe {
                    expert_weight.add(
                        ((out_tile16 * k_tiles16 + first_k16) * PACKED_TILE_BYTES as u32
                            + output_row * 8
                            + half) as usize,
                    )
                };
                let second_tile = unsafe {
                    expert_weight.add(
                        ((out_tile16 * k_tiles16 + second_k16) * PACKED_TILE_BYTES as u32
                            + output_row * 8
                            + half) as usize,
                    )
                };
                let b = unsafe { [load_u32(first_tile, 0), load_u32(second_tile, 0)] };
                let scale_a = if t0 < 2 {
                    let scale_row = lane / 4 + 8 * t0;
                    unsafe {
                        *input_scales
                            .add(((group * k_tiles64 + k_tile64) * 16 + scale_row) as usize)
                    }
                } else {
                    0
                };
                let scale_b = if t0 == 0 {
                    let mut word = 0u32;
                    let mut block = 0u32;
                    while block < 4 {
                        let scale = unsafe {
                            *expert_scales.add(
                                ((out_tile16 * k_tiles16 + k_tile64 * 4 + block)
                                    * SCALE_TILE_BYTES as u32
                                    + output_row) as usize,
                            )
                        };
                        word |= u32::from(scale) << (block * 8);
                        block += 1;
                    }
                    word
                } else {
                    0
                };
                accumulators = unsafe { mma_m16n8k64_nvfp4(a, b, scale_a, scale_b, accumulators) };
                k_tile64 += 1;
            }
            let global_scale = unsafe { *global_scales.add(expert as usize) };
            let column = (lane & 3) * 2;
            let row0 = lane / 4;
            let row1 = row0 + 8;
            if row0 < length {
                let route = unsafe { *sorted_routes.add((start + row0) as usize) };
                let index = route * out_features + out_tile8 * 8 + column;
                let packed = convert::cvt_bf16x2_f32(
                    accumulators[0] * global_scale,
                    accumulators[1] * global_scale,
                );
                unsafe {
                    output
                        .add(index as usize)
                        .write(convert::cvt_f32_bf16x2_lo(packed));
                    output
                        .add((index + 1) as usize)
                        .write(convert::cvt_f32_bf16x2_hi(packed));
                }
            }
            if row1 < length {
                let route = unsafe { *sorted_routes.add((start + row1) as usize) };
                let index = route * out_features + out_tile8 * 8 + column;
                let packed = convert::cvt_bf16x2_f32(
                    accumulators[2] * global_scale,
                    accumulators[3] * global_scale,
                );
                unsafe {
                    output
                        .add(index as usize)
                        .write(convert::cvt_f32_bf16x2_lo(packed));
                    output
                        .add((index + 1) as usize)
                        .write(convert::cvt_f32_bf16x2_hi(packed));
                }
            }
            group += workers;
        }
        let _ = out_tiles16;
    }

    /// Computes one row-major ModelOpt W4A16 matrix-vector product.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn nvfp4_w4a16_matvec_f32_warp_rows(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        output: *mut f32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
    ) {
        let input_shared = DynamicSharedArray::<f32>::get();
        let thread_index = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let in_features = in_features as usize;
        let mut col = thread_index;
        while col < in_features {
            unsafe { input_shared.add(col).write(*input.add(col)) };
            col += threads;
        }
        thread::sync_threads();

        let warps_per_block = threads / LANES;
        let warp_index = thread_index / LANES;
        let row = thread::blockIdx_x() as usize * warps_per_block + warp_index;
        if row >= out_features as usize {
            return;
        }
        let packed_row = unsafe { packed_weight.add(row * (in_features / 2)) };
        let row_scale = unsafe { weight_scale.add(row * (in_features / TILE_K)) };
        let value = unsafe { nvfp4_row_dot_warp(packed_row, row_scale, input_shared, in_features) }
            * weight_scale_2;
        if warp::lane_id() == 0 {
            unsafe { output.add(row).write(value) };
        }
    }

    /// Computes independent row-major W4A16 products for a batch.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn nvfp4_w4a16_matvec_f32_warp_rows_batch(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        output: *mut f32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
    ) {
        let input_shared = DynamicSharedArray::<f32>::get();
        let batch = thread::blockIdx_y() as usize;
        let out_features = out_features as usize;
        let in_features = in_features as usize;
        let input = unsafe { input.add(batch * in_features) };
        let output = unsafe { output.add(batch * out_features) };
        let thread_index = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let mut col = thread_index;
        while col < in_features {
            unsafe { input_shared.add(col).write(*input.add(col)) };
            col += threads;
        }
        thread::sync_threads();

        let warps_per_block = threads / LANES;
        let warp_index = thread_index / LANES;
        let row = thread::blockIdx_x() as usize * warps_per_block + warp_index;
        if row >= out_features {
            return;
        }
        let packed_row = unsafe { packed_weight.add(row * (in_features / 2)) };
        let row_scale = unsafe { weight_scale.add(row * (in_features / TILE_K)) };
        let value = unsafe { nvfp4_row_dot_warp(packed_row, row_scale, input_shared, in_features) }
            * weight_scale_2;
        if warp::lane_id() == 0 {
            unsafe { output.add(row).write(value) };
        }
    }

    /// Computes block-local top-1 candidates for a row-major W4A16 matvec.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn nvfp4_w4a16_top1_pass1_f32(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        scratch_value: *mut f32,
        scratch_index: *mut u32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
    ) {
        let input_shared = DynamicSharedArray::<f32>::get();
        let thread_index = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let in_features = in_features as usize;
        let mut col = thread_index;
        while col < in_features {
            unsafe { input_shared.add(col).write(*input.add(col)) };
            col += threads;
        }
        let warps_per_block = threads / LANES;
        let warp_values = unsafe { input_shared.add(in_features) };
        let warp_indices = unsafe { warp_values.add(warps_per_block).cast::<u32>() };
        thread::sync_threads();

        let warp_index = thread_index / LANES;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * warps_per_block + warp_index;
        let mut value = f32::NEG_INFINITY;
        let mut index = 0;
        if row < out_features as usize {
            let packed_row = unsafe { packed_weight.add(row * (in_features / 2)) };
            let row_scale = unsafe { weight_scale.add(row * (in_features / TILE_K)) };
            value = unsafe { nvfp4_row_dot_warp(packed_row, row_scale, input_shared, in_features) }
                * weight_scale_2;
            index = row as u32;
        }
        if lane == 0 {
            unsafe {
                warp_values.add(warp_index).write(value);
                warp_indices.add(warp_index).write(index);
            }
        }
        thread::sync_threads();

        let mut stride = warps_per_block / 2;
        while stride != 0 {
            if thread_index < stride {
                let other_value = unsafe { *warp_values.add(thread_index + stride) };
                let other_index = unsafe { *warp_indices.add(thread_index + stride) };
                let current_value = unsafe { *warp_values.add(thread_index) };
                let current_index = unsafe { *warp_indices.add(thread_index) };
                if other_value > current_value
                    || (other_value == current_value && other_index < current_index)
                {
                    unsafe {
                        warp_values.add(thread_index).write(other_value);
                        warp_indices.add(thread_index).write(other_index);
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if thread_index == 0 {
            unsafe {
                scratch_value
                    .add(thread::blockIdx_x() as usize)
                    .write(*warp_values);
                scratch_index
                    .add(thread::blockIdx_x() as usize)
                    .write(*warp_indices);
            }
        }
    }

    /// Reduces W4A16 block candidates to one top-1 result.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub unsafe fn nvfp4_w4a16_top1_final_f32(
        scratch_value: *const f32,
        scratch_index: *const u32,
        out_index: *mut u32,
        out_value: *mut f32,
        len: u32,
    ) {
        static mut MAX_VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut MAX_INDICES: SharedArray<u32, 128> = SharedArray::UNINIT;
        let max_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAX_VALUES) };
        let max_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAX_INDICES) };
        let lane = thread::threadIdx_x();
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = 0;
        let mut candidate = lane;
        while candidate < len {
            let value = unsafe { *scratch_value.add(candidate as usize) };
            let index = unsafe { *scratch_index.add(candidate as usize) };
            if value > best_value || (value == best_value && index < best_index) {
                best_value = value;
                best_index = index;
            }
            candidate += thread::blockDim_x();
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
                out_index.write(*max_indices);
                out_value.write(*max_values);
            }
        }
    }

    /// Reuses one W4A16 weight row across at most four activation rows.
    #[kernel]
    #[launch_bounds(1024)]
    pub unsafe fn nvfp4_w4a16_matvec_f32_reuse_weights_batch(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        output: *mut f32,
        batch_size: u32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
    ) {
        let thread_index = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let warps_per_block = threads / LANES;
        let warp_index = thread_index / LANES;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * warps_per_block + warp_index;
        let out_features = out_features as usize;
        let in_features = in_features as usize;
        if row >= out_features {
            return;
        }

        let packed_row = unsafe { packed_weight.add(row * (in_features / 2)) };
        let row_scale = unsafe { weight_scale.add(row * (in_features / TILE_K)) };
        let mut acc = [0.0f32; 4];
        let mut col = lane * 4;
        while col < in_features {
            let (b0, b1, scale) = unsafe {
                (
                    *packed_row.add(col / 2),
                    *packed_row.add(col / 2 + 1),
                    e4m3_value(*row_scale.add(col / TILE_K)),
                )
            };
            let weights = [
                e2m1_value(b0 & 0x0f) * scale,
                e2m1_value(b0 >> 4) * scale,
                e2m1_value(b1 & 0x0f) * scale,
                e2m1_value(b1 >> 4) * scale,
            ];
            let mut batch = 0;
            while batch < batch_size as usize {
                let input_row = unsafe { input.add(batch * in_features + col) };
                acc[batch] = unsafe { (*input_row).mul_add(weights[0], acc[batch]) };
                acc[batch] = unsafe { (*input_row.add(1)).mul_add(weights[1], acc[batch]) };
                acc[batch] = unsafe { (*input_row.add(2)).mul_add(weights[2], acc[batch]) };
                acc[batch] = unsafe { (*input_row.add(3)).mul_add(weights[3], acc[batch]) };
                batch += 1;
            }
            col += LANES * 4;
        }

        let mut batch = 0;
        while batch < batch_size as usize {
            acc[batch] += warp::shuffle_xor_f32(acc[batch], 16);
            acc[batch] += warp::shuffle_xor_f32(acc[batch], 8);
            acc[batch] += warp::shuffle_xor_f32(acc[batch], 4);
            acc[batch] += warp::shuffle_xor_f32(acc[batch], 2);
            acc[batch] += warp::shuffle_xor_f32(acc[batch], 1);
            if lane == 0 {
                unsafe {
                    output
                        .add(batch * out_features + row)
                        .write(acc[batch] * weight_scale_2)
                };
            }
            batch += 1;
        }
    }
}
