//! GGML block records stored directly in ModelOpt safetensors.
//!
//! IQ2's grid is from ggml-org/llama.cpp `ggml/src/ggml-common.h`.
//! Its MIT licence is retained in `data/ggml-LICENSE`.

use crate::{Error, Result};

/// Packed formats in NVIDIA's Bolt 24GB checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GgmlBlockFormat {
    /// A FP16 scale and 32 signed bytes: GGML Q8_0.
    Q8 = 0,
    /// A FP16 scale and an indexed 256-value grid: GGML IQ2_XXS.
    Iq2Xxs = 1,
}

impl GgmlBlockFormat {
    /// Number of logical input channels per block.
    pub const fn block_elements(self) -> usize {
        match self {
            Self::Q8 => 32,
            Self::Iq2Xxs => 256,
        }
    }

    /// Bytes in one stored block, including its scale.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q8 => 34,
            Self::Iq2Xxs => 66,
        }
    }

    /// Identifies a packed `[output, blocks, bytes]` tensor.
    pub fn from_shape(shape: &[usize]) -> Option<Self> {
        match shape {
            [_, _, 34] => Some(Self::Q8),
            [_, _, 66] => Some(Self::Iq2Xxs),
            _ => None,
        }
    }

    /// Checks a complete packed matrix without changing its representation.
    pub fn validate(self, bytes: &[u8], rows: usize, cols: usize) -> Result<()> {
        let expected = rows
            .checked_mul(cols / self.block_elements())
            .and_then(|n| n.checked_mul(self.block_bytes()));
        if rows == 0
            || cols == 0
            || !cols.is_multiple_of(self.block_elements())
            || expected != Some(bytes.len())
        {
            return Err(Error::Format {
                label: "GGML packed matrix",
                detail: format!(
                    "invalid {self:?} matrix: {rows}x{cols}, {} bytes",
                    bytes.len()
                ),
            });
        }
        for block in bytes.chunks_exact(self.block_bytes()) {
            let scale = u16::from_le_bytes([block[0], block[1]]);
            if scale & 0x7c00 == 0x7c00 {
                return Err(Error::Format {
                    label: "GGML block scale",
                    detail: "non-finite FP16 scale".into(),
                });
            }
        }
        Ok(())
    }

    /// Independent host dequantization for import checks and CUDA references.
    pub fn dequantize(self, bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
        self.validate(bytes, rows, cols)?;
        let mut result = Vec::with_capacity(rows * cols);
        for block in bytes.chunks_exact(self.block_bytes()) {
            let d = super::gguf_quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            match self {
                Self::Q8 => result.extend(block[2..].iter().map(|&q| d * f32::from(q as i8))),
                Self::Iq2Xxs => {
                    for group in block[2..].as_chunks::<8>().0 {
                        let signs_scale = u32::from_le_bytes(group[4..8].try_into().unwrap());
                        let scale = d * (0.5 + (signs_scale >> 28) as f32) * 0.25;
                        for (part, &index) in group[..4].iter().enumerate() {
                            let signs = (signs_scale >> (7 * part)) & 127;
                            let signs = signs | ((signs.count_ones() & 1) << 7);
                            for (lane, &q) in IQ2_XXS_GRID[usize::from(index) * 8..][..8]
                                .iter()
                                .enumerate()
                            {
                                result.push(
                                    scale
                                        * f32::from(q)
                                        * if signs & (1 << lane) == 0 { 1. } else { -1. },
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

/// Little-endian, eight-byte grid entries shared by the host and CUDA decoders.
pub const IQ2_XXS_GRID: &[u8; 2048] = include_bytes!("data/iq2xxs_grid.bin");

/// One row-major matrix in a released GGML block format.
#[derive(Clone, Debug)]
pub struct ModelOptGgmlLinear {
    pub format: GgmlBlockFormat,
    pub out_features: usize,
    pub in_features: usize,
    pub packed: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_signed_values_and_malformed_blocks() {
        let mut b = vec![0x00, 0x38]; // 0.5 in FP16.
        b.extend((0..32).map(|i| (i as i8 - 16) as u8));
        let v = GgmlBlockFormat::Q8.dequantize(&b, 1, 32).unwrap();
        assert_eq!(
            v,
            (0..32).map(|i| (i - 16) as f32 * 0.5).collect::<Vec<_>>()
        );
        assert!(GgmlBlockFormat::Q8.validate(&b, 1, 31).is_err());
        b[1] = 0x7c;
        assert!(GgmlBlockFormat::Q8.validate(&b, 1, 32).is_err());
    }

    #[test]
    fn iq2_scale_and_sign_parity() {
        let mut b = vec![0u8; 66];
        b[1] = 0x3c; // Unit FP16 block scale.
        let word = (3u32 << 28) | 1 | (3 << 7) | (127 << 14);
        b[6..10].copy_from_slice(&word.to_le_bytes());
        let v = GgmlBlockFormat::Iq2Xxs.dequantize(&b, 1, 256).unwrap();
        // Grid zero contains eight 8s; subgroup scale is 3.5/4.
        assert_eq!(&v[..8], &[-7., 7., 7., 7., 7., 7., 7., -7.]);
        assert_eq!(&v[8..16], &[-7., -7., 7., 7., 7., 7., 7., 7.]);
        assert_eq!(&v[16..24], &[-7.; 8]);
        assert_eq!(&v[32..], &[1.; 224]);
    }
}
