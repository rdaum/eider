//! Host-side data-format helpers for the current NVFP4 path.
//!
//! These helpers are intentionally explicit about cuBLASLt's packed layouts.
//! The quantizer here is still intentionally simple, but it produces the same
//! kind of packed E2M1 values and tiled UE4M3 scale layout consumed by
//! cuBLASLt's block-scaled FP4 matmul.

/// Host-side result of quantizing a matrix to cuBLASLt's NVFP4 layout.
pub struct QuantizedNvfp4 {
    /// Packed E2M1 values, two matrix elements per byte.
    pub packed_values: Vec<u8>,
    /// UE4M3 scales in cuBLASLt's tiled `VEC16` layout.
    pub scales: Vec<u8>,
    /// Dequantized column-major values represented by `packed_values` and
    /// `scales`.
    pub dequantized_values: Vec<f32>,
}

/// Host-side result of 4-row-grouped NVFP4 quantization for SM12x `mma.sync`.
///
/// The SM12x `mma.sync` SFA applies one scale per **4-row group** per K-sf
/// block, not per individual row. This quantizer groups 4 consecutive M rows
/// and computes a single UE4M3 scale per `(4-row group, 16-K block)` pair.
/// Scales are stored as a simple `[M/4, K/16]` row-major byte array.
pub struct QuantizedNvfp4Grouped {
    /// Packed E2M1 values, two matrix elements per byte, column-major [K, M].
    pub packed_values: Vec<u8>,
    /// UE4M3 scales in simple `[M/4, K/16]` row-major layout.
    pub group_scales: Vec<u8>,
    /// Dequantized column-major values represented by `packed_values` and
    /// `group_scales`.
    pub dequantized_values: Vec<f32>,
}

/// Rounds `value` up to a multiple of `multiple`.
pub fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

/// Returns the 4-bit E2M1 code for `value` using the finite round-to-nearest
/// behavior matched against CUDA's `cuda_fp4.h` helper in the smoke tests.
///
/// NaN maps to the positive saturated code used by CUDA for this focused check.
pub fn e2m1_code(value: f32) -> u8 {
    const POSITIVE_VALUES: [(u8, f32); 8] = [
        (0x0, 0.0),
        (0x1, 0.5),
        (0x2, 1.0),
        (0x3, 1.5),
        (0x4, 2.0),
        (0x5, 3.0),
        (0x6, 4.0),
        (0x7, 6.0),
    ];

    if value.is_nan() {
        return 0x7;
    }

    let sign = if value.is_sign_negative() { 0x8 } else { 0x0 };
    let abs = value.abs();
    if abs > 6.0 {
        return sign | 0x7;
    }

    let mut best_code = 0x0;
    let mut best_error = f32::INFINITY;
    for (code, candidate) in POSITIVE_VALUES {
        let error = (abs - candidate).abs();
        if error < best_error || (error == best_error && (code & 1) == 0 && (best_code & 1) == 1) {
            best_code = code;
            best_error = error;
        }
    }
    sign | best_code
}

/// Decodes a 4-bit E2M1 code to f32.
pub fn e2m1_value(code: u8) -> f32 {
    let magnitude = match code & 0x7 {
        0x0 => 0.0,
        0x1 => 0.5,
        0x2 => 1.0,
        0x3 => 1.5,
        0x4 => 2.0,
        0x5 => 3.0,
        0x6 => 4.0,
        _ => 6.0,
    };
    if code & 0x8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Packs f32 values into E2M1 nibbles.
///
/// The first element in each pair is stored in the low nibble and the second in
/// the high nibble, matching the cuBLASLt input path used by this crate.
pub fn pack_e2m1(values: &[f32]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for pair in values.chunks(2) {
        let lo = e2m1_code(pair[0]);
        let hi = pair.get(1).map(|v| e2m1_code(*v)).unwrap_or(0);
        packed.push(lo | (hi << 4));
    }
    packed
}

/// Returns CUDA's E2M1 conversion result for focused host-side cross-checks.
pub fn cuda_e2m1_code(value: f32) -> u8 {
    unsafe { crate::ffi::infer_cuda_e2m1_rn(value) & 0x0f }
}

/// Decodes an NVIDIA E4M3 byte to f32.
///
/// This is used for the unsigned-positive UE4M3 scale path. Negative and NaN
/// encodings are decoded for completeness, but quantization only emits positive
/// finite codes.
pub fn e4m3_value(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (code >> 3) & 0x0f;
    let mant = code & 0x07;

    if exp == 0 {
        sign * (mant as f32) * 2.0f32.powi(-9)
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + (mant as f32) / 8.0) * 2.0f32.powi((exp as i32) - 7)
    }
}

/// Encodes a positive finite scale as E4M3, using nearest representable value.
///
/// Values outside E4M3's finite range saturate. Non-positive or non-finite
/// values encode as zero. Ties choose the even mantissa code.
pub fn ue4m3_code(value: f32) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }

    let mut best_code = 0u8;
    let mut best_error = value.abs();
    for code in 0x01..=0x7e {
        let candidate = e4m3_value(code);
        let error = (value - candidate).abs();
        if error < best_error || (error == best_error && (code & 1) == 0 && (best_code & 1) == 1) {
            best_code = code;
            best_error = error;
        }
    }
    best_code
}

/// Returns CUDA's E4M3 saturating conversion result for host-side cross-checks.
pub fn cuda_e4m3_code(value: f32) -> u8 {
    unsafe { crate::ffi::infer_cuda_e4m3_satfinite(value) }
}

/// Returns the number of bytes needed for cuBLASLt's tiled UE4M3 scale layout.
///
/// `outer_dim` is the matrix dimension indexed by scale-vector owner, and
/// `inner_dim` is the dimension split into 16-element scale blocks.
pub fn ue4m3_scale_layout_len(outer_dim: usize, inner_dim: usize) -> usize {
    let inner_scale_blocks = inner_dim.div_ceil(16);
    let sf_inner_dim = round_up(inner_scale_blocks, 4);
    round_up(outer_dim, 128) * sf_inner_dim
}

/// `const` version of [`ue4m3_scale_layout_len`] for benchmark byte formulas.
pub const fn ue4m3_scale_layout_len_const(outer_dim: usize, inner_dim: usize) -> usize {
    let inner_scale_blocks = inner_dim.div_ceil(16);
    let sf_inner_dim = inner_scale_blocks.div_ceil(4) * 4;
    outer_dim.div_ceil(128) * 128 * sf_inner_dim
}

/// Computes a byte offset in cuBLASLt's 128x4 tiled UE4M3 scale layout.
pub fn ue4m3_tiled_scale_offset(outer: usize, inner_block: usize, inner_dim: usize) -> usize {
    let inner_scale_blocks = inner_dim.div_ceil(16);
    let sf_inner_dim = round_up(inner_scale_blocks, 4);
    let tile_outer = outer / 128;
    let outer_in_tile = outer % 128;
    let tile_inner = inner_block / 4;
    let inner_in_tile = inner_block % 4;
    let tile_base = (tile_inner * 4 + tile_outer * sf_inner_dim) * 128;
    tile_base + (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile
}

/// Creates a cuBLASLt UE4M3 scale layout filled with scale `1.0`.
///
/// UE4M3 `1.0` is encoded as `0x38`. This is only the current smoke-test path,
/// not a real quantizer.
pub fn ue4m3_ones_scale_layout(outer_dim: usize, inner_dim: usize) -> Vec<u8> {
    let mut layout = vec![0; ue4m3_scale_layout_len(outer_dim, inner_dim)];
    for outer in 0..outer_dim {
        for inner_block in 0..inner_dim.div_ceil(16) {
            let offset = ue4m3_tiled_scale_offset(outer, inner_block, inner_dim);
            layout[offset] = 0x38;
        }
    }
    layout
}

/// Quantizes a column-major matrix to packed E2M1 values plus UE4M3 block
/// scales.
///
/// A scale is chosen for each 16-value block along the row dimension of each
/// column. The scale is `max_abs / 6`, rounded to the nearest positive finite
/// E4M3 value, because E2M1's largest finite magnitude is 6. The returned
/// dequantized values use the rounded scale and packed E2M1 codes, which makes
/// them the right CPU reference for cuBLASLt correctness tests.
pub fn quantize_nvfp4_col_major(rows: usize, cols: usize, values: &[f32]) -> QuantizedNvfp4 {
    assert_eq!(values.len(), rows * cols);

    let mut scaled_values = vec![0.0f32; values.len()];
    let mut dequantized_values = vec![0.0f32; values.len()];
    let mut scales = vec![0; ue4m3_scale_layout_len(cols, rows)];

    for col in 0..cols {
        for row_block in 0..rows.div_ceil(16) {
            let row_start = row_block * 16;
            let row_end = (row_start + 16).min(rows);
            let mut max_abs = 0.0f32;
            for row in row_start..row_end {
                let value = values[row + col * rows];
                if value.is_finite() {
                    max_abs = max_abs.max(value.abs());
                }
            }

            let scale_code = if max_abs == 0.0 {
                0
            } else {
                ue4m3_code(max_abs / 6.0)
            };
            let scale = e4m3_value(scale_code);
            let scale_offset = ue4m3_tiled_scale_offset(col, row_block, rows);
            scales[scale_offset] = scale_code;

            for row in row_start..row_end {
                let idx = row + col * rows;
                let scaled = if scale == 0.0 {
                    0.0
                } else {
                    values[idx] / scale
                };
                let code = e2m1_code(scaled);
                scaled_values[idx] = e2m1_value(code);
                dequantized_values[idx] = scaled_values[idx] * scale;
            }
        }
    }

    QuantizedNvfp4 {
        packed_values: pack_e2m1(&scaled_values),
        scales,
        dequantized_values,
    }
}

/// Quantizes a column-major matrix with 4-row-grouped scales for SM12x `mma.sync`.
///
/// `rows` = K, `cols` = M. Values are column-major [rows, cols]. The scale
/// for a `(4-row M group, 16-element K block)` pair is computed as
/// `max_abs / 6` over all 4×16 = 64 elements in the group, rounded to UE4M3.
/// Scales are stored as `[cols/4, rows/16]` row-major bytes. The packed E2M1
/// values use the group scale (not the per-row scale) and are stored in the
/// same column-major [rows, cols] order as [`quantize_nvfp4_col_major`].
pub fn quantize_nvfp4_4row_groups(
    rows: usize,
    cols: usize,
    values: &[f32],
) -> QuantizedNvfp4Grouped {
    assert_eq!(values.len(), rows * cols);
    assert!(
        cols.is_multiple_of(4),
        "4-row-grouped quantization requires cols (M) divisible by 4"
    );
    assert!(
        rows.is_multiple_of(16),
        "4-row-grouped quantization requires rows (K) divisible by 16"
    );

    let n_m_groups = cols / 4;
    let n_k_blocks = rows / 16;
    let mut scaled_values = vec![0.0f32; values.len()];
    let mut dequantized_values = vec![0.0f32; values.len()];
    let mut group_scales = vec![0u8; n_m_groups * n_k_blocks];

    for m_group in 0..n_m_groups {
        for k_block in 0..n_k_blocks {
            let m_start = m_group * 4;
            let k_start = k_block * 16;
            let mut max_abs = 0.0f32;
            for m in m_start..m_start + 4 {
                for k in k_start..k_start + 16 {
                    let value = values[k + m * rows];
                    if value.is_finite() {
                        max_abs = max_abs.max(value.abs());
                    }
                }
            }

            let scale_code = if max_abs == 0.0 {
                0
            } else {
                ue4m3_code(max_abs / 6.0)
            };
            let scale = e4m3_value(scale_code);
            group_scales[m_group * n_k_blocks + k_block] = scale_code;

            for m in m_start..m_start + 4 {
                for k in k_start..k_start + 16 {
                    let idx = k + m * rows;
                    let scaled = if scale == 0.0 {
                        0.0
                    } else {
                        values[idx] / scale
                    };
                    let code = e2m1_code(scaled);
                    scaled_values[idx] = e2m1_value(code);
                    dequantized_values[idx] = scaled_values[idx] * scale;
                }
            }
        }
    }

    QuantizedNvfp4Grouped {
        packed_values: pack_e2m1(&scaled_values),
        group_scales,
        dequantized_values,
    }
}

/// Converts a BF16 bit pattern to f32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts f32 to a BF16 bit pattern using round-to-nearest-even.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

/// CPU reference GEMM for column-major `A[M,K] * B[K,N]`.
pub fn cpu_matmul_col_major(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0;
            for inner in 0..k {
                acc += a[row + inner * m] * b[inner + col * k];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

/// Focused edge cases used to compare the Rust E2M1 packer with CUDA's header
/// conversion helper.
pub fn e2m1_oracle_values() -> Vec<f32> {
    let mut values = vec![
        f32::NEG_INFINITY,
        -99.0,
        -6.1,
        -6.0,
        -5.0,
        -3.5,
        -2.5,
        -1.75,
        -1.25,
        -0.75,
        -0.26,
        -0.25,
        -0.24,
        -0.0,
        0.0,
        0.24,
        0.25,
        0.26,
        0.75,
        1.25,
        1.26,
        1.75,
        2.5,
        3.5,
        5.0,
        6.0,
        6.1,
        99.0,
        f32::INFINITY,
        f32::NAN,
    ];
    for bits in [
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3e7f_ffff,
        0x3e80_0000,
        0x3eff_ffff,
        0x3f00_0000,
        0x3f7f_ffff,
        0x3f80_0000,
        0x40bf_ffff,
        0x40c0_0000,
    ] {
        values.push(f32::from_bits(bits));
        values.push(-f32::from_bits(bits));
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_e2m1_low_nibble_first() {
        assert_eq!(pack_e2m1(&[1.0, -1.0]), vec![0xa2]);
        assert_eq!(pack_e2m1(&[1.0, 0.0, -1.0]), vec![0x02, 0x0a]);
    }

    #[test]
    fn ue4m3_encoder_matches_cuda_for_positive_scales() {
        for value in [
            0.0, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 6.0, 16.0, 448.0, 1000.0,
        ] {
            assert_eq!(
                ue4m3_code(value),
                cuda_e4m3_code(value),
                "UE4M3 mismatch for value={value}"
            );
        }
    }

    #[test]
    fn quantizes_per_sixteen_value_block() {
        let values = (0..20).map(|i| (i as f32) - 10.0).collect::<Vec<_>>();
        let quantized = quantize_nvfp4_col_major(20, 1, &values);
        let first_scale = quantized.scales[ue4m3_tiled_scale_offset(0, 0, 20)];
        let second_scale = quantized.scales[ue4m3_tiled_scale_offset(0, 1, 20)];
        assert_ne!(first_scale, 0);
        assert_ne!(second_scale, 0);
        assert_eq!(quantized.packed_values.len(), 10);
        assert_eq!(quantized.dequantized_values.len(), 20);
    }

    #[test]
    fn e2m1_matches_cuda_header_conversion() {
        for value in e2m1_oracle_values() {
            let rust = e2m1_code(value);
            let cuda = cuda_e2m1_code(value);
            assert_eq!(
                rust, cuda,
                "E2M1 mismatch for value={value:?}: rust=0x{rust:x}, cuda=0x{cuda:x}"
            );
        }
    }

    #[test]
    fn ue4m3_scale_layout_uses_cublaslt_tile_order() {
        assert_eq!(ue4m3_scale_layout_len(128, 128), 1024);
        assert_eq!(ue4m3_tiled_scale_offset(0, 0, 128), 0);
        assert_eq!(ue4m3_tiled_scale_offset(1, 0, 128), 16);
        assert_eq!(ue4m3_tiled_scale_offset(32, 0, 128), 4);
        assert_eq!(ue4m3_tiled_scale_offset(0, 1, 128), 1);
        assert_eq!(ue4m3_tiled_scale_offset(0, 4, 128), 512);
    }
}
