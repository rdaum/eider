//! Lossless import of GGML NVFP4 blocks into row-major ModelOpt storage.

use crate::{Error, GgufIndex, ModelOptNvfp4Linear, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// GGML NVFP4: four K16 scales followed by 32 nibble-packed bytes.
pub const GGML_TYPE_NVFP4: u32 = 40;

/// Converts GGML's split-half nibbles to adjacent ModelOpt nibbles.
/// No floating-point decoding or requantization occurs.
pub fn unpack_gguf_nvfp4(
    bytes: &[u8],
    out_features: usize,
    in_features: usize,
    global_scale: f32,
) -> Result<ModelOptNvfp4Linear> {
    let elements = out_features
        .checked_mul(in_features)
        .ok_or_else(|| invalid("shape overflow"))?;
    if out_features == 0
        || in_features == 0
        || !in_features.is_multiple_of(64)
        || bytes.len() != elements / 64 * 36
        || !global_scale.is_finite()
        || global_scale <= 0.0
    {
        return Err(invalid(format!(
            "invalid {out_features}x{in_features} NVFP4 record: {} bytes, scale {global_scale}",
            bytes.len()
        )));
    }
    let mut packed_weight = vec![0; elements / 2];
    let mut weight_scale = Vec::with_capacity(elements / 16);
    for (block_index, block) in bytes.as_chunks::<36>().0.iter().enumerate() {
        // UE4M3 has no sign. 0x7f is not a finite scale.
        if block[..4].iter().any(|&scale| scale >= 0x7f) {
            return Err(invalid(format!(
                "invalid UE4M3 scale in block {block_index}"
            )));
        }
        weight_scale.extend_from_slice(&block[..4]);
        for group in 0..4 {
            let source = &block[4 + group * 8..4 + (group + 1) * 8];
            let target = &mut packed_weight[block_index * 32 + group * 8..][..8];
            for j in 0..4 {
                target[j] = (source[2 * j] & 15) | ((source[2 * j + 1] & 15) << 4);
                target[j + 4] = (source[2 * j] >> 4) | (source[2 * j + 1] & 0xf0);
            }
        }
    }
    Ok(ModelOptNvfp4Linear {
        prefix: String::new(),
        out_features,
        in_features,
        packed_weight,
        weight_scale,
        weight_scale_2: global_scale,
        input_scale: 1.0,
    })
}

impl GgufIndex {
    /// Reads one matrix or one expert from a rank-three NVFP4 tensor.
    /// Global scales are scalar for dense tensors and per-expert otherwise.
    pub fn read_nvfp4_linear(
        &self,
        name: &str,
        expert: Option<usize>,
    ) -> Result<ModelOptNvfp4Linear> {
        let tensor = self.tensor(name)?;
        let (cols, rows, count) = match *tensor.dimensions.as_slice() {
            [cols, rows] if expert.is_none() => (cols, rows, 1),
            [cols, rows, count] if expert.is_some() => (cols, rows, count),
            _ => {
                return Err(invalid(format!(
                    "{name}: expected matrix or expert table, got {:?}",
                    tensor.dimensions
                )));
            }
        };
        let expert = expert.unwrap_or(0);
        if tensor.kind != GGML_TYPE_NVFP4
            || expert as u64 >= count
            || cols == 0
            || rows == 0
            || !cols.is_multiple_of(64)
        {
            return Err(invalid(format!(
                "{name}: invalid NVFP4 type, shape or expert index"
            )));
        }
        let byte_len = rows
            .checked_mul(cols / 64)
            .and_then(|v| v.checked_mul(36))
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("matrix byte count overflow"))?;
        let prefix = name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid("weight name must end in .weight"))?;
        let scale = self.tensor(&format!("{prefix}.scale"))?;
        if scale.kind != 0 || scale.dimensions != [count] {
            return Err(invalid(format!("{prefix}.scale: expected F32 [{count}]")));
        }
        let mut file = File::open(self.path()).map_err(|e| invalid(e.to_string()))?;
        let scale_offset = (expert as u64)
            .checked_mul(4)
            .and_then(|v| scale.offset.checked_add(v))
            .ok_or_else(|| invalid("scale offset overflow"))?;
        file.seek(SeekFrom::Start(scale_offset))
            .map_err(|e| invalid(e.to_string()))?;
        let mut global = [0; 4];
        file.read_exact(&mut global)
            .map_err(|e| invalid(e.to_string()))?;
        let offset = (expert as u64)
            .checked_mul(byte_len as u64)
            .and_then(|v| tensor.offset.checked_add(v))
            .ok_or_else(|| invalid("weight offset overflow"))?;
        let file_len = file.metadata().map_err(|e| invalid(e.to_string()))?.len();
        if offset
            .checked_add(byte_len as u64)
            .is_none_or(|end| end > file_len)
        {
            return Err(invalid(format!("{name}: weight exceeds file length")));
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| invalid(e.to_string()))?;
        let mut bytes = vec![0; byte_len];
        file.read_exact(&mut bytes)
            .map_err(|e| invalid(e.to_string()))?;
        let mut linear = unpack_gguf_nvfp4(
            &bytes,
            rows as usize,
            cols as usize,
            f32::from_le_bytes(global),
        )?;
        linear.prefix = format!("{prefix}[{expert}]");
        Ok(linear)
    }
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "GGUF NVFP4",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_expert_ranges_scales_and_precision_names() {
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        string(&mut bytes, "general.tensor_extra.name");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        string(&mut bytes, "experts.weight");
        string(&mut bytes, "general.tensor_extra.prec_a4");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(0);
        for (name, dims, kind, offset) in [
            ("experts.weight", vec![64u64, 1, 2], 40u32, 0u64),
            ("experts.scale", vec![2], 0, 96),
        ] {
            string(&mut bytes, name);
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        let data = bytes.len().div_ceil(32) * 32;
        bytes.resize(data, 0);
        for packed in [0x21, 0x43] {
            bytes.extend_from_slice(&[0x38; 4]);
            bytes.extend_from_slice(&[packed; 32]);
        }
        bytes.resize(data + 96, 0);
        bytes.extend_from_slice(&2f32.to_le_bytes());
        bytes.extend_from_slice(&3f32.to_le_bytes());
        let path =
            std::env::temp_dir().join(format!("eider-nvfp4-expert-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let index = GgufIndex::open(&path).unwrap();
        assert_eq!(
            index.metadata()["general.tensor_extra.name"],
            crate::GgufValue::StringArray(vec!["experts.weight".into()])
        );
        let a = index.read_nvfp4_linear("experts.weight", Some(0)).unwrap();
        let b = index.read_nvfp4_linear("experts.weight", Some(1)).unwrap();
        assert_eq!(a.weight_scale_2, 2.0);
        assert_eq!(b.weight_scale_2, 3.0);
        assert_eq!(
            &a.packed_weight[..8],
            &[0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22]
        );
        assert_eq!(
            &b.packed_weight[..8],
            &[0x33, 0x33, 0x33, 0x33, 0x44, 0x44, 0x44, 0x44]
        );
        assert!(index.read_nvfp4_linear("experts.weight", None).is_err());
        assert!(index.read_nvfp4_linear("experts.weight", Some(2)).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn preserves_every_nibble_and_k16_scale_across_rows() {
        let mut blocks = Vec::new();
        for block in 0..4u8 {
            blocks.extend_from_slice(&[0, 1 + block, 0x38, 0x7e]);
            for group in 0..4u8 {
                for j in 0..8u8 {
                    blocks.push(((j + group + block) & 15) | (((j + 8 + group + block) & 15) << 4));
                }
            }
        }
        let out = unpack_gguf_nvfp4(&blocks, 2, 128, 0.125).unwrap();
        for i in 0..256 {
            let block = i / 64;
            let group = (i % 64) / 16;
            let lane = i % 16;
            let nibble = (out.packed_weight[i / 2] >> ((i % 2) * 4)) & 15;
            assert_eq!(nibble as usize, (lane + group + block) & 15);
            assert_eq!(out.weight_scale[i / 16], blocks[block * 36 + group]);
        }
        assert_eq!(out.weight_scale_2, 0.125);
        assert_eq!(out.input_scale, 1.0);
    }

    #[test]
    fn rejects_bad_lengths_tails_and_scales() {
        for (rows, cols, scale) in [(1, 16, 1.0), (0, 64, 1.0), (1, 64, 0.0), (1, 64, f32::NAN)] {
            assert!(unpack_gguf_nvfp4(&[0; 36], rows, cols, scale).is_err());
        }
        assert!(unpack_gguf_nvfp4(&[0; 35], 1, 64, 1.0).is_err());
        let mut block = [0; 36];
        block[0] = 0x7f;
        assert!(unpack_gguf_nvfp4(&block, 1, 64, 1.0).is_err());
    }
}
