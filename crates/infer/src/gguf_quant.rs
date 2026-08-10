//! Import-time decoding for the GGML K-quantized tensor formats used by
//! official companion checkpoints.

use nvfp4::{Error, Result};

/// GGML tensor type identifier for Q4_K.
pub const GGML_TYPE_Q4_K: u32 = 12;
/// GGML tensor type identifier for Q6_K.
pub const GGML_TYPE_Q6_K: u32 = 14;

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;

/// Returns the packed byte length for a supported K-quantized tensor.
pub fn quantized_byte_len(kind: u32, elements: usize) -> Result<usize> {
    if !elements.is_multiple_of(QK_K) {
        return Err(Error::Shape {
            label: "GGML K-quant tensor",
            expected: format!("element count divisible by {QK_K}"),
            actual: format!("{elements} elements"),
        });
    }
    let block_bytes = match kind {
        GGML_TYPE_Q4_K => Q4_K_BLOCK_BYTES,
        GGML_TYPE_Q6_K => Q6_K_BLOCK_BYTES,
        _ => {
            return Err(Error::Format {
                label: "GGML K-quant tensor",
                detail: format!("unsupported tensor type {kind}"),
            });
        }
    };
    (elements / QK_K)
        .checked_mul(block_bytes)
        .ok_or_else(|| Error::Shape {
            label: "GGML K-quant tensor",
            expected: "packed byte length without overflow".to_string(),
            actual: format!("{elements} elements of type {kind}"),
        })
}

/// Decodes a supported GGML K-quantized tensor directly to row-major BF16.
pub fn dequantize_to_bf16(kind: u32, bytes: &[u8], elements: usize) -> Result<Vec<u16>> {
    let expected = quantized_byte_len(kind, elements)?;
    if bytes.len() != expected {
        return Err(Error::Shape {
            label: "GGML K-quant tensor bytes",
            expected: format!("{expected} bytes"),
            actual: format!("{} bytes", bytes.len()),
        });
    }

    let mut values = Vec::with_capacity(elements);
    match kind {
        GGML_TYPE_Q4_K => {
            for block in bytes.chunks_exact(Q4_K_BLOCK_BYTES) {
                dequantize_q4_k_block(block, |value| {
                    values.push(nvfp4::format::f32_to_bf16(value));
                });
            }
        }
        GGML_TYPE_Q6_K => {
            values.resize(elements, 0);
            for (block_index, block) in bytes.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
                let start = block_index * QK_K;
                dequantize_q6_k_block(block, &mut values[start..start + QK_K]);
            }
        }
        _ => unreachable!("validated by quantized_byte_len"),
    }
    Ok(values)
}

fn dequantize_q4_k_block(block: &[u8], mut emit: impl FnMut(f32)) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let quants = &block[16..];

    for group in 0..4 {
        let (scale_low, min_low) = q4_k_scale_min(group * 2, scales);
        let (scale_high, min_high) = q4_k_scale_min(group * 2 + 1, scales);
        let packed = &quants[group * 32..group * 32 + 32];
        let delta_low = d * f32::from(scale_low);
        let offset_low = dmin * f32::from(min_low);
        let delta_high = d * f32::from(scale_high);
        let offset_high = dmin * f32::from(min_high);
        for &quant in packed {
            emit(delta_low * f32::from(quant & 0x0f) - offset_low);
        }
        for &quant in packed {
            emit(delta_high * f32::from(quant >> 4) - offset_high);
        }
    }
}

fn q4_k_scale_min(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

fn dequantize_q6_k_block(block: &[u8], output: &mut [u16]) {
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql = &ql[half * 64..];
        let qh = &qh[half * 32..];
        let scales = &scales[half * 8..];
        let output = &mut output[half * 128..half * 128 + 128];
        for lane in 0..32 {
            let scale_index = lane / 16;
            let q1 = i32::from((ql[lane] & 0x0f) | ((qh[lane] & 3) << 4)) - 32;
            let q2 = i32::from((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            output[lane] = encode_scaled_q6(d, scales[scale_index], q1);
            output[lane + 32] = encode_scaled_q6(d, scales[scale_index + 2], q2);
            output[lane + 64] = encode_scaled_q6(d, scales[scale_index + 4], q3);
            output[lane + 96] = encode_scaled_q6(d, scales[scale_index + 6], q4);
        }
    }
}

fn encode_scaled_q6(delta: f32, scale: u8, quant: i32) -> u16 {
    let scale = i8::from_ne_bytes([scale]);
    nvfp4::format::f32_to_bf16(delta * f32::from(scale) * quant as f32)
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = u32::from(value & 0x03ff);
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = fraction.leading_zeros() - 22;
            let normalized = (fraction << (leading + 1)) & 0x03ff;
            let exponent = 112u32 - leading;
            sign | (exponent << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_bf16(values: &[u16]) -> Vec<f32> {
        values
            .iter()
            .map(|&value| nvfp4::format::bf16_to_f32(value))
            .collect()
    }

    #[test]
    fn decodes_q4_k_scale_and_nibble_layout() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[4..8].copy_from_slice(&[1, 2, 3, 4]);
        block[12..16].copy_from_slice(&[5, 6, 7, 8]);
        block[16..].fill(0x32);

        let values =
            decode_bf16(&dequantize_to_bf16(GGML_TYPE_Q4_K, &block, QK_K).expect("decode Q4_K"));
        for group in 0..8 {
            let expected_quant = if group % 2 == 0 { 2.0 } else { 3.0 };
            let expected_scale = (group + 1) as f32;
            assert_eq!(
                &values[group * 32..group * 32 + 32],
                vec![expected_quant * expected_scale; 32]
            );
        }
    }

    #[test]
    fn decodes_q6_k_signed_scale_and_segment_layout() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        for (index, scale) in (1i8..=16).enumerate() {
            block[192 + index] = scale.to_ne_bytes()[0];
        }
        block[208..210].copy_from_slice(&0x3800u16.to_le_bytes());

        let values =
            decode_bf16(&dequantize_to_bf16(GGML_TYPE_Q6_K, &block, QK_K).expect("decode Q6_K"));
        let expected_scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (segment, scale) in expected_scales.into_iter().enumerate() {
            assert_eq!(
                &values[segment * 16..segment * 16 + 16],
                vec![-16.0 * scale; 16]
            );
        }
        for (segment, scale) in (9..=16).map(|value| value as f32).enumerate() {
            let start = 128 + segment * 16;
            assert_eq!(&values[start..start + 16], vec![-16.0 * scale; 16]);
        }
    }

    #[test]
    fn converts_half_subnormals_and_special_values() {
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
