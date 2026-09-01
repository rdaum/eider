//! Shared numeric formats and warp-level device helpers.

use cuda_device::{convert, ptx_asm, warp};

pub(crate) const LANES: usize = 32;
pub(crate) const TILE_M: usize = 16;
pub(crate) const TILE_K: usize = 16;
pub(crate) const PACKED_TILE_BYTES: usize = TILE_M * TILE_K / 2;
pub(crate) const SCALE_TILE_BYTES: usize = TILE_M;
pub(crate) const SINGLE_WARPS: usize = 16;
pub(crate) const BATCH_WARPS: usize = 8;
#[inline(always)]
pub(crate) fn e2m1_value(code: u8) -> f32 {
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
pub(crate) fn e2m1_code(value: f32) -> u8 {
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
pub(crate) fn ue4m3_code(value: f32) -> u8 {
    convert::cvt_rn_satfinite_e4m3x2_f32(value, value) as u8
}

#[inline(always)]
pub(crate) unsafe fn load_u32(pointer: *const u8, word: usize) -> u32 {
    unsafe { *pointer.cast::<u32>().add(word) }
}

#[inline(always)]
pub(crate) unsafe fn mma_m16n8k64_nvfp4(
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
pub(crate) unsafe fn scale_word(scales: *const u8) -> u32 {
    unsafe {
        u32::from(*scales)
            | (u32::from(*scales.add(1)) << 8)
            | (u32::from(*scales.add(2)) << 16)
            | (u32::from(*scales.add(3)) << 24)
    }
}

#[inline(always)]
pub(crate) fn probability_amplification(tokens: u32) -> f32 {
    let minimum = (3 * tokens + 255) / 256;
    let mut amplification = 1;
    while amplification < minimum {
        amplification <<= 1;
    }
    amplification as f32
}

#[inline(always)]
pub(crate) fn e4m3_value(code: u8) -> f32 {
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
pub(crate) fn dequant_bf16_pair(packed: u8, scale: f32) -> u32 {
    convert::cvt_bf16x2_f32(
        e2m1_value(packed & 0x0f) * scale,
        e2m1_value(packed >> 4) * scale,
    )
}

#[inline(always)]
pub(crate) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[inline(always)]
pub(crate) fn round_to_bf16(value: f32) -> f32 {
    convert::cvt_f32_bf16x2_lo(convert::cvt_bf16x2_f32(value, 0.0))
}

#[inline(always)]
pub(crate) fn bf16_to_f32(value: u16) -> f32 {
    convert::cvt_f32_bf16x2_lo(u32::from(value))
}

#[inline(always)]
pub(crate) fn pack_bf16(first: u16, second: u16) -> u32 {
    u32::from(first) | (u32::from(second) << 16)
}

#[inline(always)]
pub(crate) fn f32_pair_to_bf16(first: f32, second: f32) -> u32 {
    convert::cvt_bf16x2_f32(first, second)
}

#[inline(always)]
pub(crate) unsafe fn store_bf16_pair(output: *mut u16, index: usize, first: f32, second: f32) {
    let packed = f32_pair_to_bf16(first, second);
    unsafe {
        output.add(index).write(packed as u16);
        output.add(index + 1).write((packed >> 16) as u16);
    }
}

#[inline(always)]
pub(crate) unsafe fn mma_bf16_m16n8k16(
    accumulators: [f32; 4],
    a: [u32; 4],
    b: [u32; 2],
) -> [f32; 4] {
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

#[inline(always)]
pub(crate) fn warp_sum(mut value: f32) -> f32 {
    value += warp::shuffle_xor_f32(value, 16);
    value += warp::shuffle_xor_f32(value, 8);
    value += warp::shuffle_xor_f32(value, 4);
    value += warp::shuffle_xor_f32(value, 2);
    value + warp::shuffle_xor_f32(value, 1)
}

#[inline(always)]
pub(crate) fn warp_max(mut value: f32) -> f32 {
    value = value.max(warp::shuffle_xor_f32(value, 16));
    value = value.max(warp::shuffle_xor_f32(value, 8));
    value = value.max(warp::shuffle_xor_f32(value, 4));
    value = value.max(warp::shuffle_xor_f32(value, 2));
    value.max(warp::shuffle_xor_f32(value, 1))
}

#[inline(always)]
pub(crate) fn ue4m3_tiled_scale_offset(outer: u32, inner_block: u32, inner_dim: u32) -> usize {
    let inner_scale_blocks = inner_dim.div_ceil(16);
    let scale_inner = inner_scale_blocks.div_ceil(4) * 4;
    let tile_outer = outer / 128;
    let outer_in_tile = outer % 128;
    let tile_inner = inner_block / 4;
    let inner_in_tile = inner_block % 4;
    let tile_base = (tile_inner * 4 + tile_outer * scale_inner) * 128;
    (tile_base + (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile) as usize
}
