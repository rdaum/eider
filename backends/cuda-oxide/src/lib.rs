//! cuda-oxide implementation of Eider's SM121 W4A16 device kernels.
//!
//! The crate produces PTX only. The `eider-cuda` build script assembles the PTX
//! into a CUBIN and embeds it in the stable Rust host crate.

#![deny(unsafe_op_in_unsafe_fn)]

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{
    DynamicSharedArray, SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract,
};
use cuda_device::{ptx_asm, thread, warp};

const LANES: usize = 32;
const TILE_M: usize = 16;
const TILE_K: usize = 16;
const PACKED_TILE_BYTES: usize = TILE_M * TILE_K / 2;
const SCALE_TILE_BYTES: usize = TILE_M;
const SINGLE_WARPS: usize = 16;
const BATCH_WARPS: usize = 8;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn e2m1_value(code: u8) -> f32 {
        let magnitude = u32::from(code & 0x7);
        let exponent = magnitude >> 1;
        let mantissa = magnitude & 1;
        let magnitude_bits = if exponent == 0 {
            mantissa * 0x3f00_0000
        } else {
            ((exponent + 126) << 23) | (mantissa << 22)
        };
        let sign = u32::from(code & 0x8) << 28;
        f32::from_bits(sign | magnitude_bits)
    }

    #[inline(always)]
    fn e2m1_code(value: f32) -> u8 {
        let negative = value.to_bits() >> 31 != 0;
        let magnitude = value.abs();
        let code = if magnitude.is_nan() || magnitude <= 0.25 {
            0
        } else if magnitude < 0.75 {
            1
        } else if magnitude <= 1.25 {
            2
        } else if magnitude < 1.75 {
            3
        } else if magnitude <= 2.5 {
            4
        } else if magnitude < 3.5 {
            5
        } else if magnitude <= 5.0 {
            6
        } else {
            7
        };
        code | if negative { 0x8 } else { 0 }
    }

    #[inline(always)]
    fn ue4m3_code(value: f32) -> u8 {
        convert::cvt_rn_satfinite_e4m3x2_f32(value, value) as u8
    }

    #[inline(always)]
    unsafe fn load_u32(pointer: *const u8, word: usize) -> u32 {
        unsafe { *pointer.cast::<u32>().add(word) }
    }

    #[inline(always)]
    unsafe fn mma_m16n8k64_nvfp4(
        a: [u32; 4],
        b: [u32; 2],
        scale_a: u32,
        scale_b: u32,
        accumulators: [f32; 4],
    ) -> [f32; 4] {
        let d0: f32;
        let d1: f32;
        let d2: f32;
        let d3: f32;
        let selector = 0u16;
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.\
                 m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
                 {%0, %1, %2, %3}, \
                 {%4, %5, %6, %7}, \
                 {%8, %9}, \
                 {%10, %11, %12, %13}, \
                 {%14}, {%15, %16}, {%17}, {%18, %19};",
                out("=f") d0,
                out("=f") d1,
                out("=f") d2,
                out("=f") d3,
                in("r") a[0],
                in("r") a[1],
                in("r") a[2],
                in("r") a[3],
                in("r") b[0],
                in("r") b[1],
                in("f") accumulators[0],
                in("f") accumulators[1],
                in("f") accumulators[2],
                in("f") accumulators[3],
                in("r") scale_a,
                in("h") selector,
                in("h") selector,
                in("r") scale_b,
                in("h") selector,
                in("h") selector,
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    #[inline(always)]
    unsafe fn scale_word(scales: *const u8) -> u32 {
        unsafe {
            u32::from(*scales)
                | (u32::from(*scales.add(1)) << 8)
                | (u32::from(*scales.add(2)) << 16)
                | (u32::from(*scales.add(3)) << 24)
        }
    }

    #[inline(always)]
    fn probability_amplification(tokens: u32) -> f32 {
        let minimum = (3 * tokens + 255) / 256;
        let mut amplification = 1;
        while amplification < minimum {
            amplification <<= 1;
        }
        amplification as f32
    }

    #[inline(always)]
    fn e4m3_value(code: u8) -> f32 {
        let sign = u32::from(code & 0x80) << 24;
        let exponent = u32::from((code >> 3) & 0x0f);
        let mantissa = u32::from(code & 0x07);
        if exponent == 0 {
            let value = mantissa as f32 * 0.001_953_125;
            return if sign == 0 { value } else { -value };
        }
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::from_bits(sign | 0x7fff_ffff);
        }
        f32::from_bits(sign | ((exponent + 120) << 23) | (mantissa << 20))
    }

    #[inline(always)]
    fn dequant_bf16_pair(packed: u8, scale: f32) -> u32 {
        convert::cvt_bf16x2_f32(
            e2m1_value(packed & 0x0f) * scale,
            e2m1_value(packed >> 4) * scale,
        )
    }

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
                            *input.add(
                                (input_row * in_features + k_tile * 64 + col) as usize,
                            )
                        }
                    } else {
                        0.0
                    };
                    let scale = e4m3_value(unsafe {
                        *scale_codes.add((row * 4 + col / 16) as usize)
                    });
                    packed |= e2m1_code(if scale == 0.0 { 0.0 } else { value / scale })
                        << (nibble * 4);
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
            let expert_weight = unsafe {
                tiled_weight.add((expert * expert_weight_stride) as usize)
            };
            let expert_scales = unsafe {
                tiled_scales.add((expert * expert_scale_stride) as usize)
            };
            let mut accumulators = [0.0f32; 4];
            let output_col = out_tile8 * 8 + lane / 4;
            let out_tile16 = output_col / 16;
            let output_row = output_col & 15;
            let t0 = lane & 3;
            let mut k_tile64 = 0u32;
            while k_tile64 < k_tiles64 {
                let input_tile = unsafe {
                    input_tiles.add(((group * k_tiles64 + k_tile64) * 512) as usize)
                };
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
                        *input_scales.add(
                            ((group * k_tiles64 + k_tile64) * 16 + scale_row) as usize,
                        )
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
                accumulators = unsafe {
                    mma_m16n8k64_nvfp4(a, b, scale_a, scale_b, accumulators)
                };
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
            value = unsafe {
                nvfp4_row_dot_warp(packed_row, row_scale, input_shared, in_features)
            } * weight_scale_2;
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

    #[inline(always)]
    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    #[inline(always)]
    fn round_to_bf16(value: f32) -> f32 {
        convert::cvt_f32_bf16x2_lo(convert::cvt_bf16x2_f32(value, 0.0))
    }

    #[inline(always)]
    fn bf16_to_f32(value: u16) -> f32 {
        convert::cvt_f32_bf16x2_lo(u32::from(value))
    }

    #[inline(always)]
    fn pack_bf16(first: u16, second: u16) -> u32 {
        u32::from(first) | (u32::from(second) << 16)
    }

    #[inline(always)]
    fn f32_pair_to_bf16(first: f32, second: f32) -> u32 {
        convert::cvt_bf16x2_f32(first, second)
    }

    #[inline(always)]
    unsafe fn store_bf16_pair(output: *mut u16, index: usize, first: f32, second: f32) {
        let packed = f32_pair_to_bf16(first, second);
        unsafe {
            output.add(index).write(packed as u16);
            output.add(index + 1).write((packed >> 16) as u16);
        }
    }

    #[inline(always)]
    unsafe fn mma_bf16_m16n8k16(accumulators: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
        let d0: f32;
        let d1: f32;
        let d2: f32;
        let d3: f32;
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
                 {%0, %1, %2, %3}, \
                 {%4, %5, %6, %7}, \
                 {%8, %9}, \
                 {%10, %11, %12, %13};",
                out("=f") d0,
                out("=f") d1,
                out("=f") d2,
                out("=f") d3,
                in("r") a[0],
                in("r") a[1],
                in("r") a[2],
                in("r") a[3],
                in("r") b[0],
                in("r") b[1],
                in("f") accumulators[0],
                in("f") accumulators[1],
                in("f") accumulators[2],
                in("f") accumulators[3],
                options(register_only),
            );
        }
        [d0, d1, d2, d3]
    }

    const GDN_HEADS: usize = 32;
    const GDN_DIM: usize = 128;
    const GDN_CHUNK: usize = 64;

    #[inline(always)]
    fn gdn_vector_index(token: usize, head: usize, feature: usize) -> usize {
        (token * GDN_HEADS + head) * GDN_DIM + feature
    }

    #[inline(always)]
    fn gdn_scalar_index(token: usize, head: usize) -> usize {
        token * GDN_HEADS + head
    }

    #[inline(always)]
    fn gdn_triangle_index(token: usize, head: usize, col: usize) -> usize {
        (token * GDN_HEADS + head) * GDN_CHUNK + col
    }

    #[inline(always)]
    unsafe fn store_gdn_kkt_element(
        beta: *const u16,
        gate_cumsum: *const f32,
        a: *mut f32,
        start: usize,
        head: usize,
        length: usize,
        row: usize,
        col: usize,
        dot: f32,
    ) {
        if row >= length {
            return;
        }
        let value = if row == col {
            bf16_to_f32(unsafe { *beta.add(gdn_scalar_index(start + row, head)) })
        } else if col < row && col < length {
            let row_beta = bf16_to_f32(unsafe { *beta.add(gdn_scalar_index(start + row, head)) });
            let decay = (unsafe {
                *gate_cumsum.add(gdn_scalar_index(start + row, head))
                    - *gate_cumsum.add(gdn_scalar_index(start + col, head))
            })
            .exp();
            row_beta * decay * dot
        } else {
            0.0
        };
        unsafe {
            a.add(gdn_triangle_index(start + row, head, col))
                .write(value)
        };
    }

    #[inline(always)]
    unsafe fn gdn_chunk_bounds(
        chunk: usize,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: usize,
    ) -> (usize, usize, usize) {
        let sequence = unsafe { *chunk_indices.add(chunk * 2) } as usize;
        let local_chunk = unsafe { *chunk_indices.add(chunk * 2 + 1) } as usize;
        let sequence_start = unsafe { *cu_seqlens.add(sequence) } as usize;
        let sequence_end = (unsafe { *cu_seqlens.add(sequence + 1) } as usize).min(total_tokens);
        let start = sequence_start + local_chunk * GDN_CHUNK;
        let length = sequence_end.saturating_sub(start).min(GDN_CHUNK);
        (sequence, start, length)
    }

    #[inline(always)]
    fn warp_sum(mut value: f32) -> f32 {
        value += warp::shuffle_xor_f32(value, 16);
        value += warp::shuffle_xor_f32(value, 8);
        value += warp::shuffle_xor_f32(value, 4);
        value += warp::shuffle_xor_f32(value, 2);
        value + warp::shuffle_xor_f32(value, 1)
    }

    #[inline(always)]
    fn warp_max(mut value: f32) -> f32 {
        value = value.max(warp::shuffle_xor_f32(value, 16));
        value = value.max(warp::shuffle_xor_f32(value, 8));
        value = value.max(warp::shuffle_xor_f32(value, 4));
        value = value.max(warp::shuffle_xor_f32(value, 2));
        value.max(warp::shuffle_xor_f32(value, 1))
    }

    #[inline(always)]
    fn ue4m3_tiled_scale_offset(outer: u32, inner_block: u32, inner_dim: u32) -> usize {
        let inner_scale_blocks = inner_dim.div_ceil(16);
        let scale_inner = inner_scale_blocks.div_ceil(4) * 4;
        let tile_outer = outer / 128;
        let outer_in_tile = outer % 128;
        let tile_inner = inner_block / 4;
        let inner_in_tile = inner_block % 4;
        let tile_base = (tile_inner * 4 + tile_outer * scale_inner) * 128;
        (tile_base + (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile) as usize
    }

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
        let reduction_values =
            unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION_VALUES) };
        let reduction_indices =
            unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION_INDICES) };
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
                *routed.add((route * cols + col) as usize)
                    * *route_weights.add(route as usize)
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

    /// Computes one inclusive GDN gate prefix sum per 64-token chunk and head.
    #[kernel]
    #[launch_bounds(64)]
    pub unsafe fn qwen36_gdn_chunk_cumsum(
        gate: *const u16,
        gate_cumsum: *mut f32,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut WARP_TOTALS: SharedArray<f32, 2> = SharedArray::UNINIT;
        let warp_totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_TOTALS) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let token = thread::threadIdx_x() as usize;
        let lane = token & 31;
        let warp_index = token >> 5;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let mut sum = if token < length {
            bf16_to_f32(unsafe { *gate.add(gdn_scalar_index(start + token, head)) })
        } else {
            0.0
        };
        let mut offset = 1;
        while offset < 32 {
            let previous = warp::shuffle_up_f32(sum, offset as u32);
            if lane >= offset {
                sum += previous;
            }
            offset *= 2;
        }
        if lane == 31 {
            unsafe { warp_totals.add(warp_index).write(sum) };
        }
        thread::sync_threads();
        if warp_index == 1 {
            sum += unsafe { *warp_totals };
        }
        if token < length {
            unsafe {
                gate_cumsum
                    .add(gdn_scalar_index(start + token, head))
                    .write(sum)
            };
        }
    }

    /// Forms the lower-triangular GDN key Gram transform with BF16 MMA.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_kkt(
        key: *const u16,
        beta: *const u16,
        gate_cumsum: *const f32,
        a: *mut f32,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut KEY: SharedArray<u16, 8192> = SharedArray::UNINIT;
        let shared_key = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEY) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let mut index = thread::threadIdx_x() as usize;
        while index < GDN_CHUNK * GDN_DIM {
            let token = index / GDN_DIM;
            let value = if token < length {
                unsafe { *key.add(gdn_vector_index(start + token, head, index % GDN_DIM)) }
            } else {
                0
            };
            unsafe { shared_key.add(index).write(value) };
            index += thread::blockDim_x() as usize;
        }
        thread::sync_threads();
        let tile_row = (warp_index / 4) * 16;
        let tile_col_16 = (warp_index & 3) * 16;
        let group = lane >> 2;
        let lane_in_group = lane & 3;
        let row0 = tile_row + group;
        let row1 = row0 + 8;
        let col_in_tile = lane_in_group * 2;

        let mut half = 0;
        while half < 2 {
            let tile_col = tile_col_16 + half * 8;
            let mut d = [0.0f32; 4];
            let mut feature = 0;
            while feature < GDN_DIM {
                let feature0 = feature + lane_in_group * 2;
                let feature1 = feature0 + 8;
                let load_key_pair = |row: usize, col: usize| -> u32 {
                    if row >= length {
                        return 0;
                    }
                    unsafe {
                        pack_bf16(
                            *shared_key.add(row * GDN_DIM + col),
                            *shared_key.add(row * GDN_DIM + col + 1),
                        )
                    }
                };
                let a_fragment = [
                    load_key_pair(row0, feature0),
                    load_key_pair(row1, feature0),
                    load_key_pair(row0, feature1),
                    load_key_pair(row1, feature1),
                ];
                let output_col = tile_col + group;
                let load_transposed_pair = |first_feature: usize| -> u32 {
                    if output_col >= length {
                        return 0;
                    }
                    unsafe {
                        pack_bf16(
                            *shared_key.add(output_col * GDN_DIM + first_feature),
                            *shared_key.add(output_col * GDN_DIM + first_feature + 1),
                        )
                    }
                };
                let b_fragment = [
                    load_transposed_pair(feature0),
                    load_transposed_pair(feature1),
                ];
                d = unsafe { mma_bf16_m16n8k16(d, a_fragment, b_fragment) };
                feature += 16;
            }

            let col = tile_col + col_in_tile;
            unsafe {
                store_gdn_kkt_element(beta, gate_cumsum, a, start, head, length, row0, col, d[0]);
                store_gdn_kkt_element(
                    beta,
                    gate_cumsum,
                    a,
                    start,
                    head,
                    length,
                    row0,
                    col + 1,
                    d[1],
                );
                store_gdn_kkt_element(beta, gate_cumsum, a, start, head, length, row1, col, d[2]);
                store_gdn_kkt_element(
                    beta,
                    gate_cumsum,
                    a,
                    start,
                    head,
                    length,
                    row1,
                    col + 1,
                    d[3],
                );
            }
            half += 1;
        }
    }

    /// Solves the per-chunk lower-triangular GDN transform.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen36_gdn_chunk_solve(
        a: *mut f32,
        a_inverse: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut TRANSFORM: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut LOWER_ROW: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut DIAGONAL: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut PRODUCT: SharedArray<f32, 256> = SharedArray::UNINIT;
        let transform = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TRANSFORM) };
        let lower_row = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOWER_ROW) };
        let diagonal = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DIAGONAL) };
        let product = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PRODUCT) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let col = thread::threadIdx_x() as usize;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };

        if length < GDN_CHUNK {
            let mut row = 0;
            while row < length {
                if col < GDN_CHUNK {
                    unsafe {
                        lower_row
                            .add(col)
                            .write(*a.add(gdn_triangle_index(start + row, head, col)))
                    };
                }
                thread::sync_threads();
                if col < GDN_CHUNK {
                    let value = if col <= row {
                        let mut value = if col == row {
                            unsafe { *lower_row.add(row) }
                        } else {
                            0.0
                        };
                        let mut inner = col;
                        while inner < row {
                            value -= unsafe {
                                *lower_row.add(inner) * *transform.add(inner * GDN_CHUNK + col)
                            };
                            inner += 1;
                        }
                        value
                    } else {
                        0.0
                    };
                    unsafe { transform.add(row * GDN_CHUNK + col).write(value) };
                }
                thread::sync_threads();
                row += 1;
            }
            if col < GDN_CHUNK {
                row = 0;
                while row < length {
                    let value = unsafe { *transform.add(row * GDN_CHUNK + col) };
                    unsafe {
                        a_inverse
                            .add(gdn_triangle_index(start + row, head, col))
                            .write(f32_pair_to_bf16(value, 0.0) as u16)
                    };
                    row += 1;
                }
            }
            return;
        }

        if col < GDN_CHUNK {
            unsafe {
                diagonal
                    .add(col)
                    .write(*a.add(gdn_triangle_index(start + col, head, col)))
            };
        }
        thread::sync_threads();

        let warp_index = col / 32;
        let lane = col & 31;
        if warp_index < 4 && lane < 16 {
            let block_start = warp_index * 16;
            let mut local_row = 0;
            while local_row < 16 {
                let row = block_start + local_row;
                let matrix_col = block_start + lane;
                let mut inverse = if lane == local_row { 1.0 } else { 0.0 };
                if lane < local_row {
                    let mut inner = lane;
                    while inner < local_row {
                        inverse -= unsafe {
                            *a.add(gdn_triangle_index(start + row, head, block_start + inner))
                                * *a.add(gdn_triangle_index(
                                    start + block_start + inner,
                                    head,
                                    matrix_col,
                                ))
                        };
                        inner += 1;
                    }
                }
                warp::sync_mask(0x0000_ffff);
                unsafe {
                    a.add(gdn_triangle_index(start + row, head, matrix_col))
                        .write(if lane <= local_row { inverse } else { 0.0 })
                };
                warp::sync_mask(0x0000_ffff);
                local_row += 1;
            }
        }
        thread::sync_threads();

        let mut index = col;
        while index < GDN_CHUNK * GDN_CHUNK {
            let row = index / GDN_CHUNK;
            let matrix_col = index % GDN_CHUNK;
            let value = if row / 16 == matrix_col / 16 {
                unsafe {
                    *a.add(gdn_triangle_index(start + row, head, matrix_col))
                        * *diagonal.add(matrix_col)
                }
            } else {
                0.0
            };
            unsafe { transform.add(index).write(value) };
            index += thread::blockDim_x() as usize;
        }
        thread::sync_threads();

        let local_row = col / 16;
        let local_col = col % 16;
        let mut block_row = 1;
        while block_row < 4 {
            let mut block_col = 0;
            while block_col < block_row {
                let row = block_row * 16 + local_row;
                let matrix_col = block_col * 16 + local_col;
                let mut sum = 0.0;
                let mut middle_block = block_col;
                while middle_block < block_row {
                    let mut inner = 0;
                    while inner < 16 {
                        sum += unsafe {
                            *a.add(gdn_triangle_index(
                                start + row,
                                head,
                                middle_block * 16 + inner,
                            )) * *transform
                                .add((middle_block * 16 + inner) * GDN_CHUNK + matrix_col)
                        };
                        inner += 1;
                    }
                    middle_block += 1;
                }
                unsafe { product.add(col).write(sum) };
                thread::sync_threads();
                let mut solved = 0.0;
                let mut inner = 0;
                while inner < 16 {
                    solved -= unsafe {
                        *a.add(gdn_triangle_index(
                            start + block_row * 16 + local_row,
                            head,
                            block_row * 16 + inner,
                        )) * *product.add(inner * 16 + local_col)
                    };
                    inner += 1;
                }
                unsafe { transform.add(row * GDN_CHUNK + matrix_col).write(solved) };
                thread::sync_threads();
                block_col += 1;
            }
            block_row += 1;
        }

        index = col;
        while index < GDN_CHUNK * GDN_CHUNK {
            let row = index / GDN_CHUNK;
            let matrix_col = index % GDN_CHUNK;
            let value = unsafe { *transform.add(index) };
            unsafe {
                a_inverse
                    .add(gdn_triangle_index(start + row, head, matrix_col))
                    .write(f32_pair_to_bf16(value, 0.0) as u16)
            };
            index += thread::blockDim_x() as usize;
        }
    }

    /// Applies the solved chunk transform to keys and values with BF16 MMA.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_wu(
        key: *const u16,
        value: *const u16,
        a_inverse: *const u16,
        gate_cumsum: *const f32,
        w: *mut u16,
        u: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut SCALED_KEY: SharedArray<u16, 8192> = SharedArray::UNINIT;
        let scaled_key = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALED_KEY) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let mut index = thread::threadIdx_x() as usize;
        while index < GDN_CHUNK * GDN_DIM {
            let token = index / GDN_DIM;
            let packed = if token < length {
                let decay =
                    unsafe { *gate_cumsum.add(gdn_scalar_index(start + token, head)) }.exp();
                let value = bf16_to_f32(unsafe {
                    *key.add(gdn_vector_index(start + token, head, index % GDN_DIM))
                });
                f32_pair_to_bf16(decay * value, 0.0) as u16
            } else {
                0
            };
            unsafe { scaled_key.add(index).write(packed) };
            index += thread::blockDim_x() as usize;
        }
        thread::sync_threads();
        let group = lane >> 2;
        let lane_in_group = lane & 3;
        let mut tile = warp_index;
        while tile < 64 {
            let tile_row = (tile / 16) * 16;
            let tile_col = (tile & 15) * 8;
            let row0 = tile_row + group;
            let row1 = row0 + 8;
            let output_col = tile_col + group;
            let mut w_acc = [0.0f32; 4];
            let mut u_acc = [0.0f32; 4];
            let mut source = 0;
            while source < GDN_CHUNK {
                let source0 = source + lane_in_group * 2;
                let source1 = source0 + 8;
                let load_transform = |row: usize, col: usize| -> u16 {
                    if row < length && col < length {
                        unsafe { *a_inverse.add(gdn_triangle_index(start + row, head, col)) }
                    } else {
                        0
                    }
                };
                let a_fragment = [
                    pack_bf16(
                        load_transform(row0, source0),
                        load_transform(row0, source0 + 1),
                    ),
                    pack_bf16(
                        load_transform(row1, source0),
                        load_transform(row1, source0 + 1),
                    ),
                    pack_bf16(
                        load_transform(row0, source1),
                        load_transform(row0, source1 + 1),
                    ),
                    pack_bf16(
                        load_transform(row1, source1),
                        load_transform(row1, source1 + 1),
                    ),
                ];
                let load_scaled_key = |token: usize| -> u16 {
                    if token >= length {
                        return 0;
                    }
                    unsafe { *scaled_key.add(token * GDN_DIM + output_col) }
                };
                let load_value = |token: usize| -> u16 {
                    if token < length {
                        unsafe { *value.add(gdn_vector_index(start + token, head, output_col)) }
                    } else {
                        0
                    }
                };
                let w_fragment = [
                    pack_bf16(load_scaled_key(source0), load_scaled_key(source0 + 1)),
                    pack_bf16(load_scaled_key(source1), load_scaled_key(source1 + 1)),
                ];
                let u_fragment = [
                    pack_bf16(load_value(source0), load_value(source0 + 1)),
                    pack_bf16(load_value(source1), load_value(source1 + 1)),
                ];
                w_acc = unsafe { mma_bf16_m16n8k16(w_acc, a_fragment, w_fragment) };
                u_acc = unsafe { mma_bf16_m16n8k16(u_acc, a_fragment, u_fragment) };
                source += 16;
            }
            let col = tile_col + lane_in_group * 2;
            if row0 < length {
                unsafe {
                    store_bf16_pair(
                        w,
                        gdn_vector_index(start + row0, head, col),
                        w_acc[0],
                        w_acc[1],
                    );
                    store_bf16_pair(
                        u,
                        gdn_vector_index(start + row0, head, col),
                        u_acc[0],
                        u_acc[1],
                    );
                }
            }
            if row1 < length {
                unsafe {
                    store_bf16_pair(
                        w,
                        gdn_vector_index(start + row1, head, col),
                        w_acc[2],
                        w_acc[3],
                    );
                    store_bf16_pair(
                        u,
                        gdn_vector_index(start + row1, head, col),
                        u_acc[2],
                        u_acc[3],
                    );
                }
            }
            tile += 16;
        }
    }

    /// Propagates each chunk through recurrent state and snapshots its input state.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_h(
        key: *const u16,
        u: *const u16,
        w: *const u16,
        value_new: *mut u16,
        gate_cumsum: *const f32,
        h: *mut u16,
        state: *mut f32,
        cu_seqlens: *const i32,
        chunk_offsets: *const i64,
        total_tokens: u32,
    ) {
        static mut MATRIX: SharedArray<u16, 8192> = SharedArray::UNINIT;
        static mut STATE: SharedArray<u16, 4096> = SharedArray::UNINIT;
        static mut VALUE: SharedArray<u16, 2048> = SharedArray::UNINIT;
        static mut DECAY: SharedArray<f32, 64> = SharedArray::UNINIT;
        let shared_matrix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MATRIX) };
        let shared_state = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STATE) };
        let shared_value = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUE) };
        let shared_decay = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DECAY) };
        let value_partition = thread::blockIdx_x() as usize;
        let sequence = thread::blockIdx_y() as usize;
        let head = thread::blockIdx_z() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let first_value = value_partition * 32;
        let sequence_start = unsafe { *cu_seqlens.add(sequence) } as usize;
        let sequence_end =
            (unsafe { *cu_seqlens.add(sequence + 1) } as usize).min(total_tokens as usize);
        let first_chunk = unsafe { *chunk_offsets.add(sequence) } as usize;
        let end_chunk = unsafe { *chunk_offsets.add(sequence + 1) } as usize;
        let head_state = unsafe { state.add((sequence * GDN_HEADS + head) * GDN_DIM * GDN_DIM) };

        let group = lane >> 2;
        let lane_in_group = lane & 3;
        let mut chunk = first_chunk;
        while chunk < end_chunk {
            let start = sequence_start + (chunk - first_chunk) * GDN_CHUNK;
            let length = sequence_end.saturating_sub(start).min(GDN_CHUNK);
            let chunk_h = unsafe { h.add((chunk * GDN_HEADS + head) * GDN_DIM * GDN_DIM) };

            let mut index = thread::threadIdx_x() as usize;
            while index < 32 * GDN_DIM {
                let local_value = index / GDN_DIM;
                let key_feature = index % GDN_DIM;
                let state_index = (first_value + local_value) * GDN_DIM + key_feature;
                let packed = f32_pair_to_bf16(unsafe { *head_state.add(state_index) }, 0.0) as u16;
                unsafe {
                    shared_state.add(index).write(packed);
                    chunk_h.add(state_index).write(packed);
                }
                index += thread::blockDim_x() as usize;
            }
            index = thread::threadIdx_x() as usize;
            while index < GDN_CHUNK * GDN_DIM {
                let token = index / GDN_DIM;
                let packed = if token < length {
                    unsafe { *w.add(gdn_vector_index(start + token, head, index % GDN_DIM)) }
                } else {
                    0
                };
                unsafe { shared_matrix.add(index).write(packed) };
                index += thread::blockDim_x() as usize;
            }
            index = thread::threadIdx_x() as usize;
            while index < GDN_CHUNK * 32 {
                let token = index / 32;
                let local_value = index % 32;
                let packed = if token < length {
                    unsafe {
                        *u.add(gdn_vector_index(
                            start + token,
                            head,
                            first_value + local_value,
                        ))
                    }
                } else {
                    0
                };
                unsafe { shared_value.add(index).write(packed) };
                index += thread::blockDim_x() as usize;
            }
            thread::sync_threads();

            let mut correction_tile = warp_index;
            while correction_tile < 16 {
                let token_tile = (correction_tile / 4) * 16;
                let value_tile = (correction_tile & 3) * 8;
                let token0 = token_tile + group;
                let token1 = token0 + 8;
                let local_value = value_tile + group;
                let mut correction = [0.0f32; 4];
                let mut key_feature = 0;
                while key_feature < GDN_DIM {
                    let key0 = key_feature + lane_in_group * 2;
                    let key1 = key0 + 8;
                    let load_w_pair = |token: usize, feature: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_matrix.add(token * GDN_DIM + feature),
                                *shared_matrix.add(token * GDN_DIM + feature + 1),
                            )
                        }
                    };
                    let a_fragment = [
                        load_w_pair(token0, key0),
                        load_w_pair(token1, key0),
                        load_w_pair(token0, key1),
                        load_w_pair(token1, key1),
                    ];
                    let load_state_pair = |first_key: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_state.add(local_value * GDN_DIM + first_key),
                                *shared_state.add(local_value * GDN_DIM + first_key + 1),
                            )
                        }
                    };
                    let b_fragment = [load_state_pair(key0), load_state_pair(key1)];
                    correction = unsafe { mma_bf16_m16n8k16(correction, a_fragment, b_fragment) };
                    key_feature += 16;
                }
                let output_value = value_tile + lane_in_group * 2;
                if token0 < length {
                    let shared_base = token0 * 32 + output_value;
                    let output_base =
                        gdn_vector_index(start + token0, head, first_value + output_value);
                    let first =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base) }) - correction[0];
                    let second =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base + 1) }) - correction[1];
                    unsafe {
                        store_bf16_pair(shared_value, shared_base, first, second);
                        store_bf16_pair(value_new, output_base, first, second);
                    }
                }
                if token1 < length {
                    let shared_base = token1 * 32 + output_value;
                    let output_base =
                        gdn_vector_index(start + token1, head, first_value + output_value);
                    let first =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base) }) - correction[2];
                    let second =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base + 1) }) - correction[3];
                    unsafe {
                        store_bf16_pair(shared_value, shared_base, first, second);
                        store_bf16_pair(value_new, output_base, first, second);
                    }
                }
                correction_tile += 16;
            }
            thread::sync_threads();

            let chunk_gate =
                unsafe { *gate_cumsum.add(gdn_scalar_index(start + length - 1, head)) };
            let chunk_decay = chunk_gate.exp();
            if thread::threadIdx_x() < GDN_CHUNK as u32 {
                let token = thread::threadIdx_x() as usize;
                let decay = if token < length {
                    (chunk_gate
                        - unsafe { *gate_cumsum.add(gdn_scalar_index(start + token, head)) })
                    .exp()
                } else {
                    0.0
                };
                unsafe { shared_decay.add(token).write(decay) };
            }
            thread::sync_threads();
            index = thread::threadIdx_x() as usize;
            while index < GDN_CHUNK * GDN_DIM {
                let token = index / GDN_DIM;
                let value = if token < length {
                    bf16_to_f32(unsafe {
                        *key.add(gdn_vector_index(start + token, head, index % GDN_DIM))
                    }) * unsafe { *shared_decay.add(token) }
                } else {
                    0.0
                };
                unsafe {
                    shared_matrix
                        .add(index)
                        .write(f32_pair_to_bf16(value, 0.0) as u16)
                };
                index += thread::blockDim_x() as usize;
            }
            thread::sync_threads();

            let mut state_tile = warp_index;
            while state_tile < 32 {
                let value_row_tile = (state_tile / 16) * 16;
                let key_col_tile = (state_tile & 15) * 8;
                let local_value0 = value_row_tile + group;
                let local_value1 = local_value0 + 8;
                let key_col = key_col_tile + lane_in_group * 2;
                let state_row0 = first_value + local_value0;
                let state_row1 = first_value + local_value1;
                let mut state_acc = unsafe {
                    [
                        *head_state.add(state_row0 * GDN_DIM + key_col) * chunk_decay,
                        *head_state.add(state_row0 * GDN_DIM + key_col + 1) * chunk_decay,
                        *head_state.add(state_row1 * GDN_DIM + key_col) * chunk_decay,
                        *head_state.add(state_row1 * GDN_DIM + key_col + 1) * chunk_decay,
                    ]
                };
                let mut token = 0;
                while token < GDN_CHUNK {
                    let token0 = token + lane_in_group * 2;
                    let token1 = token0 + 8;
                    let load_value_pair = |value_feature: usize, first_token: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_value.add(first_token * 32 + value_feature),
                                *shared_value.add((first_token + 1) * 32 + value_feature),
                            )
                        }
                    };
                    let a_fragment = [
                        load_value_pair(local_value0, token0),
                        load_value_pair(local_value1, token0),
                        load_value_pair(local_value0, token1),
                        load_value_pair(local_value1, token1),
                    ];
                    let output_key = key_col_tile + group;
                    let load_scaled_key_pair = |first_token: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_matrix.add(first_token * GDN_DIM + output_key),
                                *shared_matrix.add((first_token + 1) * GDN_DIM + output_key),
                            )
                        }
                    };
                    let b_fragment = [load_scaled_key_pair(token0), load_scaled_key_pair(token1)];
                    state_acc = unsafe { mma_bf16_m16n8k16(state_acc, a_fragment, b_fragment) };
                    token += 16;
                }
                unsafe {
                    head_state
                        .add(state_row0 * GDN_DIM + key_col)
                        .write(state_acc[0]);
                    head_state
                        .add(state_row0 * GDN_DIM + key_col + 1)
                        .write(state_acc[1]);
                    head_state
                        .add(state_row1 * GDN_DIM + key_col)
                        .write(state_acc[2]);
                    head_state
                        .add(state_row1 * GDN_DIM + key_col + 1)
                        .write(state_acc[3]);
                }
                state_tile += 16;
            }
            thread::sync_threads();
            chunk += 1;
        }
    }

    /// Computes chunk outputs from the chunk-input state and causal delta attention.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_output(
        query: *const u16,
        key: *const u16,
        value_new: *const u16,
        h: *const u16,
        gate_cumsum: *const f32,
        output: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        scale: f32,
    ) {
        static mut ATTENTION: SharedArray<u16, 2048> = SharedArray::UNINIT;
        let attention = unsafe { SharedArray::as_raw_mut_ptr(&raw mut ATTENTION) };
        let token_partition = thread::blockIdx_x() as usize;
        let chunk = thread::blockIdx_y() as usize;
        let head = thread::blockIdx_z() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let first_token = token_partition * 32;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let chunk_h = unsafe { h.add((chunk * GDN_HEADS + head) * GDN_DIM * GDN_DIM) };
        let group = lane >> 2;
        let lane_in_group = lane & 3;

        let attention_row_tile = (warp_index / 8) * 16;
        let attention_col = (warp_index & 7) * 8;
        let local_row0 = attention_row_tile + group;
        let local_row1 = local_row0 + 8;
        let key_token = attention_col + group;
        let mut scores = [0.0f32; 4];
        let mut feature = 0;
        while feature < GDN_DIM {
            let feature0 = feature + lane_in_group * 2;
            let feature1 = feature0 + 8;
            let load_query_pair = |local_row: usize, first_feature: usize| -> u32 {
                let token = first_token + local_row;
                if token >= length {
                    return 0;
                }
                unsafe {
                    pack_bf16(
                        *query.add(gdn_vector_index(start + token, head, first_feature)),
                        *query.add(gdn_vector_index(start + token, head, first_feature + 1)),
                    )
                }
            };
            let a_fragment = [
                load_query_pair(local_row0, feature0),
                load_query_pair(local_row1, feature0),
                load_query_pair(local_row0, feature1),
                load_query_pair(local_row1, feature1),
            ];
            let load_key_pair = |first_feature: usize| -> u32 {
                if key_token >= length {
                    return 0;
                }
                unsafe {
                    pack_bf16(
                        *key.add(gdn_vector_index(start + key_token, head, first_feature)),
                        *key.add(gdn_vector_index(start + key_token, head, first_feature + 1)),
                    )
                }
            };
            let b_fragment = [load_key_pair(feature0), load_key_pair(feature1)];
            scores = unsafe { mma_bf16_m16n8k16(scores, a_fragment, b_fragment) };
            feature += 16;
        }
        let score_value = |row: usize, col: usize, value: f32| -> f32 {
            if row < length && col <= row && col < length {
                value
                    * (unsafe {
                        *gate_cumsum.add(gdn_scalar_index(start + row, head))
                            - *gate_cumsum.add(gdn_scalar_index(start + col, head))
                    })
                    .exp()
            } else {
                0.0
            }
        };
        let score_col = attention_col + lane_in_group * 2;
        let global_row0 = first_token + local_row0;
        let global_row1 = first_token + local_row1;
        unsafe {
            store_bf16_pair(
                attention,
                local_row0 * GDN_CHUNK + score_col,
                score_value(global_row0, score_col, scores[0]),
                score_value(global_row0, score_col + 1, scores[1]),
            );
            store_bf16_pair(
                attention,
                local_row1 * GDN_CHUNK + score_col,
                score_value(global_row1, score_col, scores[2]),
                score_value(global_row1, score_col + 1, scores[3]),
            );
        }
        thread::sync_threads();

        let mut output_tile = warp_index;
        while output_tile < 32 {
            let output_row_tile = (output_tile / 16) * 16;
            let output_col_tile = (output_tile & 15) * 8;
            let output_row0 = output_row_tile + group;
            let output_row1 = output_row0 + 8;
            let output_feature = output_col_tile + group;
            let mut values = [0.0f32; 4];
            let mut key_feature = 0;
            while key_feature < GDN_DIM {
                let key0 = key_feature + lane_in_group * 2;
                let key1 = key0 + 8;
                let load_scaled_query_pair = |local_row: usize, first_key: usize| -> u32 {
                    let token = first_token + local_row;
                    if token >= length {
                        return 0;
                    }
                    let decay =
                        unsafe { *gate_cumsum.add(gdn_scalar_index(start + token, head)) }.exp();
                    let first = bf16_to_f32(unsafe {
                        *query.add(gdn_vector_index(start + token, head, first_key))
                    });
                    let second = bf16_to_f32(unsafe {
                        *query.add(gdn_vector_index(start + token, head, first_key + 1))
                    });
                    f32_pair_to_bf16(decay * first, decay * second)
                };
                let a_fragment = [
                    load_scaled_query_pair(output_row0, key0),
                    load_scaled_query_pair(output_row1, key0),
                    load_scaled_query_pair(output_row0, key1),
                    load_scaled_query_pair(output_row1, key1),
                ];
                let load_state_pair = |first_key: usize| -> u32 {
                    unsafe {
                        pack_bf16(
                            *chunk_h.add(output_feature * GDN_DIM + first_key),
                            *chunk_h.add(output_feature * GDN_DIM + first_key + 1),
                        )
                    }
                };
                let b_fragment = [load_state_pair(key0), load_state_pair(key1)];
                values = unsafe { mma_bf16_m16n8k16(values, a_fragment, b_fragment) };
                key_feature += 16;
            }
            let mut source = 0;
            while source < GDN_CHUNK {
                let source0 = source + lane_in_group * 2;
                let source1 = source0 + 8;
                let load_attention_pair = |local_row: usize, first_source: usize| -> u32 {
                    unsafe {
                        pack_bf16(
                            *attention.add(local_row * GDN_CHUNK + first_source),
                            *attention.add(local_row * GDN_CHUNK + first_source + 1),
                        )
                    }
                };
                let a_fragment = [
                    load_attention_pair(output_row0, source0),
                    load_attention_pair(output_row1, source0),
                    load_attention_pair(output_row0, source1),
                    load_attention_pair(output_row1, source1),
                ];
                let load_value_pair = |first_source: usize| -> u32 {
                    let first = if first_source < length {
                        unsafe {
                            *value_new.add(gdn_vector_index(
                                start + first_source,
                                head,
                                output_feature,
                            ))
                        }
                    } else {
                        0
                    };
                    let second = if first_source + 1 < length {
                        unsafe {
                            *value_new.add(gdn_vector_index(
                                start + first_source + 1,
                                head,
                                output_feature,
                            ))
                        }
                    } else {
                        0
                    };
                    pack_bf16(first, second)
                };
                let b_fragment = [load_value_pair(source0), load_value_pair(source1)];
                values = unsafe { mma_bf16_m16n8k16(values, a_fragment, b_fragment) };
                source += 16;
            }
            let output_col = output_col_tile + lane_in_group * 2;
            let global_output_row0 = first_token + output_row0;
            let global_output_row1 = first_token + output_row1;
            if global_output_row0 < length {
                unsafe {
                    store_bf16_pair(
                        output,
                        gdn_vector_index(start + global_output_row0, head, output_col),
                        scale * values[0],
                        scale * values[1],
                    )
                };
            }
            if global_output_row1 < length {
                unsafe {
                    store_bf16_pair(
                        output,
                        gdn_vector_index(start + global_output_row1, head, output_col),
                        scale * values[2],
                        scale * values[3],
                    )
                };
            }
            output_tile += 16;
        }
    }

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
