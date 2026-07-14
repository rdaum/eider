//! Host preprocessing for the focused NVFP4 Marlin MoE path.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::format::e4m3_value;
use crate::modelopt::ModelOptNvfp4Linear;

/// Marlin-repacked NVFP4 weight and scale data for one expert linear.
pub struct MarlinNvfp4HostWeight {
    /// Repacked E2M1 values in Marlin 16x64 tensor-core tile order.
    pub packed_weight: Vec<u32>,
    /// Processed positive E4M3 scales in Marlin order.
    pub weight_scale: Vec<u8>,
    /// Global scale adjusted for Marlin's fast E2M1/E4M3 dequantization.
    pub global_scale: f32,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
}

/// Persistent Qwen3.6 top-8 NVFP4 Marlin gate/up plan.
pub struct MarlinNvfp4GateUp {
    experts: usize,
    repacked_weight: DeviceBuffer<u32>,
    weight_scale: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    output_bf16: DeviceBuffer<u16>,
    reduce_tmp: DeviceBuffer<f32>,
    locks: DeviceBuffer<i32>,
    sorted_token_ids: DeviceBuffer<i32>,
    expert_ids: DeviceBuffer<i32>,
    num_tokens_past_padded: DeviceBuffer<i32>,
}

/// Persistent Marlin plan for one Qwen3.6 shared-expert projection.
pub struct MarlinNvfp4Linear {
    repacked_weight: DeviceBuffer<u32>,
    weight_scale: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    output_bf16: DeviceBuffer<u16>,
    reduce_tmp: DeviceBuffer<f32>,
    locks: DeviceBuffer<i32>,
    sorted_token_ids: DeviceBuffer<i32>,
    expert_ids: DeviceBuffer<i32>,
    num_tokens_past_padded: DeviceBuffer<i32>,
    out_features: usize,
    in_features: usize,
}

impl MarlinNvfp4GateUp {
    /// Creates a plan from raw ModelOpt gate/up weights in expert-table order.
    pub fn new(weights: &[ModelOptNvfp4Linear]) -> Result<Self> {
        const HIDDEN: usize = 2048;
        const GATE_UP: usize = 1024;
        const TOP_K: usize = 8;
        const MOE_BLOCK: usize = 8;
        if weights.is_empty() {
            return Err(Error::Shape {
                label: "Marlin gate/up experts",
                expected: "at least one expert".to_string(),
                actual: "0 experts".to_string(),
            });
        }
        if unsafe { ffi::infer_marlin_nvfp4_gate_up_supported() } == 0 {
            return Err(Error::Format {
                label: "Marlin NVFP4 gate/up device support",
                detail: "requires a device accepted by the compiled Marlin kernel".to_string(),
            });
        }

        let mut repacked_weight = Vec::with_capacity(weights.len() * GATE_UP * HIDDEN / 8);
        let mut weight_scale = Vec::with_capacity(weights.len() * GATE_UP * HIDDEN / 16);
        let mut global_scale = Vec::with_capacity(weights.len());
        for weight in weights {
            if weight.out_features != GATE_UP || weight.in_features != HIDDEN {
                return Err(Error::Shape {
                    label: "Marlin Qwen3.6 gate/up expert",
                    expected: format!("out={GATE_UP} in={HIDDEN}"),
                    actual: format!(
                        "{} out={} in={}",
                        weight.prefix, weight.out_features, weight.in_features
                    ),
                });
            }
            let weight = MarlinNvfp4HostWeight::from_modelopt(weight)?;
            repacked_weight.extend_from_slice(&weight.packed_weight);
            weight_scale.extend_from_slice(&weight.weight_scale);
            global_scale.push(weight.global_scale);
        }

        Ok(Self {
            experts: weights.len(),
            repacked_weight: DeviceBuffer::from_host(&repacked_weight)?,
            weight_scale: DeviceBuffer::from_host(&weight_scale)?,
            global_scale: DeviceBuffer::from_host(&global_scale)?,
            input_bf16: DeviceBuffer::zeroed(HIDDEN)?,
            output_bf16: DeviceBuffer::zeroed(TOP_K * GATE_UP)?,
            reduce_tmp: DeviceBuffer::zeroed(GATE_UP * TOP_K * MOE_BLOCK)?,
            locks: DeviceBuffer::zeroed(128)?,
            sorted_token_ids: DeviceBuffer::zeroed(TOP_K * MOE_BLOCK)?,
            expert_ids: DeviceBuffer::zeroed(TOP_K)?,
            num_tokens_past_padded: DeviceBuffer::zeroed(1)?,
        })
    }

    /// Returns the number of experts stored by the plan.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Runs routed gate/up for one token and eight device-resident expert indices.
    pub fn run_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        const HIDDEN: usize = 2048;
        const GATE_UP: usize = 1024;
        const TOP_K: usize = 8;
        if indices.len() != TOP_K || input.len() != HIDDEN || output.len() != TOP_K * GATE_UP {
            return Err(Error::Shape {
                label: "Marlin Qwen3.6 gate/up buffers",
                expected: format!("indices={TOP_K} input={HIDDEN} output={}", TOP_K * GATE_UP),
                actual: format!(
                    "indices={} input={} output={}",
                    indices.len(),
                    input.len(),
                    output.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_gate_up_on_stream",
                ffi::infer_marlin_nvfp4_gate_up_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    output.buffer_mut().ptr,
                    self.input_bf16.ptr,
                    self.output_bf16.ptr,
                    self.reduce_tmp.ptr,
                    self.locks.ptr,
                    self.sorted_token_ids.ptr,
                    self.expert_ids.ptr,
                    self.num_tokens_past_padded.ptr,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Runs routed gate/up while retaining Marlin's native BF16 output for a
    /// following activation kernel.
    pub fn run_bf16_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&DeviceBuffer<u16>> {
        const HIDDEN: usize = 2048;
        const TOP_K: usize = 8;
        if indices.len() != TOP_K || input.len() != HIDDEN {
            return Err(Error::Shape {
                label: "Marlin Qwen3.6 BF16 gate/up buffers",
                expected: format!("indices={TOP_K} input={HIDDEN}"),
                actual: format!("indices={} input={}", indices.len(), input.len()),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_gate_up_on_stream",
                ffi::infer_marlin_nvfp4_gate_up_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    std::ptr::null_mut(),
                    self.input_bf16.ptr,
                    self.output_bf16.ptr,
                    self.reduce_tmp.ptr,
                    self.locks.ptr,
                    self.sorted_token_ids.ptr,
                    self.expert_ids.ptr,
                    self.num_tokens_past_padded.ptr,
                    stream.as_raw(),
                ),
            )?;
        }
        Ok(&self.output_bf16)
    }

    /// Returns Marlin's persistent BF16 routed gate/up output.
    pub fn output_bf16(&self) -> &DeviceBuffer<u16> {
        &self.output_bf16
    }
}

impl MarlinNvfp4Linear {
    /// Repackages and uploads one supported shared-expert projection.
    pub fn new(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        let (out_features, in_features) = (weight.out_features, weight.in_features);
        if !matches!((out_features, in_features), (1024, 2048) | (2048, 512)) {
            return Err(Error::Shape {
                label: "Marlin Qwen3.6 shared linear",
                expected: "out/in 1024/2048 or 2048/512".to_string(),
                actual: format!("out={out_features} in={in_features}"),
            });
        }
        if unsafe { ffi::infer_marlin_nvfp4_gate_up_supported() } == 0 {
            return Err(Error::Format {
                label: "Marlin NVFP4 shared linear device support",
                detail: "requires a device accepted by the compiled Marlin kernel".to_string(),
            });
        }
        let weight = MarlinNvfp4HostWeight::from_modelopt(weight)?;
        Ok(Self {
            repacked_weight: DeviceBuffer::from_host(&weight.packed_weight)?,
            weight_scale: DeviceBuffer::from_host(&weight.weight_scale)?,
            global_scale: DeviceBuffer::from_host(&[weight.global_scale])?,
            input_bf16: DeviceBuffer::zeroed(in_features)?,
            output_bf16: DeviceBuffer::zeroed(out_features)?,
            reduce_tmp: DeviceBuffer::zeroed(out_features * 8)?,
            locks: DeviceBuffer::zeroed(128)?,
            sorted_token_ids: DeviceBuffer::zeroed(8)?,
            expert_ids: DeviceBuffer::zeroed(1)?,
            num_tokens_past_padded: DeviceBuffer::zeroed(1)?,
            out_features,
            in_features,
        })
    }

    /// Runs this projection on `stream`.
    pub fn run_on_stream(
        &self,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err(Error::Shape {
                label: "Marlin Qwen3.6 shared linear buffers",
                expected: format!("input={} output={}", self.in_features, self.out_features),
                actual: format!("input={} output={}", input.len(), output.len()),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_linear_on_stream",
                ffi::infer_marlin_nvfp4_linear_on_stream(
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    output.buffer_mut().ptr,
                    self.input_bf16.ptr,
                    self.output_bf16.ptr,
                    self.reduce_tmp.ptr,
                    self.locks.ptr,
                    self.sorted_token_ids.ptr,
                    self.expert_ids.ptr,
                    self.num_tokens_past_padded.ptr,
                    self.out_features as u32,
                    self.in_features as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns `(out_features, in_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }
}

impl MarlinNvfp4HostWeight {
    /// Converts a raw ModelOpt W4A16 linear into the fixed Marlin layout.
    pub fn from_modelopt(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        let n = weight.out_features;
        let k = weight.in_features;
        if n == 0 || k == 0 || !n.is_multiple_of(64) || !k.is_multiple_of(16) {
            return Err(Error::Shape {
                label: "Marlin NVFP4 weight shape",
                expected: "non-zero N divisible by 64 and K divisible by 16".to_string(),
                actual: format!("N={n} K={k}"),
            });
        }
        let expected_weight = n.checked_mul(k / 2).ok_or_else(|| Error::Shape {
            label: "Marlin NVFP4 weight bytes",
            expected: "N * K / 2 without overflow".to_string(),
            actual: format!("N={n} K={k}"),
        })?;
        let expected_scales = n.checked_mul(k / 16).ok_or_else(|| Error::Shape {
            label: "Marlin NVFP4 scale bytes",
            expected: "N * K / 16 without overflow".to_string(),
            actual: format!("N={n} K={k}"),
        })?;
        if weight.packed_weight.len() != expected_weight
            || weight.weight_scale.len() != expected_scales
        {
            return Err(Error::Shape {
                label: "Marlin NVFP4 source buffers",
                expected: format!("weight={expected_weight} scales={expected_scales}"),
                actual: format!(
                    "weight={} scales={}",
                    weight.packed_weight.len(),
                    weight.weight_scale.len()
                ),
            });
        }

        let packed_weight = repack_weight(&weight.packed_weight, n, k);
        let (weight_scale, scale_factor) = repack_scales(&weight.weight_scale, n, k);
        let global_scale = weight.weight_scale_2 * 2.0f32.powi(119) / scale_factor;
        if !global_scale.is_finite() {
            return Err(Error::Format {
                label: "Marlin NVFP4 global scale",
                detail: format!(
                    "weight_scale_2={} scale_factor={scale_factor} produced {global_scale}",
                    weight.weight_scale_2
                ),
            });
        }

        Ok(Self {
            packed_weight,
            weight_scale,
            global_scale,
            out_features: n,
            in_features: k,
        })
    }
}

fn repack_weight(source: &[u8], n: usize, k: usize) -> Vec<u32> {
    const TILE_K: usize = 16;
    const TILE_N: usize = 64;
    const TILE_WORDS: usize = TILE_K * TILE_N / 8;
    const TC_OFFSETS: [usize; 4] = [0, 1, 8, 9];
    const PACK_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    let k_tiles = k / TILE_K;
    let n_tiles = n / TILE_N;
    let mut output = vec![0u32; source.len() / 4];
    for k_tile in 0..k_tiles {
        let first_k = k_tile * TILE_K;
        for n_tile in 0..n_tiles {
            let first_n = n_tile * TILE_N;
            let out_base = (k_tile * n_tiles + n_tile) * TILE_WORDS;
            for warp in 0..4 {
                for lane in 0..32 {
                    let tc_col = lane / 4;
                    let tc_row = (lane % 4) * 2;
                    let cur_n = first_n + warp * 16 + tc_col;
                    let mut values = [0u32; 8];
                    for i in 0..4 {
                        let k_offset = tc_row + TC_OFFSETS[i];
                        values[i] = packed_nibble(source, cur_n, first_k + k_offset, k);
                        values[4 + i] = packed_nibble(source, cur_n + 8, first_k + k_offset, k);
                    }
                    let mut packed = 0u32;
                    for (dst, src) in PACK_ORDER.into_iter().enumerate() {
                        packed |= values[src] << (dst * 4);
                    }
                    output[out_base + lane * 4 + warp] = packed;
                }
            }
        }
    }
    output
}

fn packed_nibble(source: &[u8], row: usize, col: usize, k: usize) -> u32 {
    let byte = source[row * (k / 2) + col / 2];
    u32::from(if col.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    })
}

fn repack_scales(source: &[u8], n: usize, k: usize) -> (Vec<u8>, f32) {
    const SCALE_PERM: [usize; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53,
        61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ];
    const FP4_SCALE_PERM: [usize; 4] = [0, 2, 1, 3];

    let groups = k / 16;
    let mut transposed = vec![0u8; source.len()];
    for group in 0..groups {
        for row in 0..n {
            transposed[group * n + row] = source[row * groups + group];
        }
    }

    let mut permuted = vec![0u8; source.len()];
    for (input, output) in transposed
        .chunks_exact(64)
        .zip(permuted.chunks_exact_mut(64))
    {
        for (dst, src) in SCALE_PERM.into_iter().enumerate() {
            output[dst] = input[src];
        }
    }
    for chunk in permuted.chunks_exact_mut(4) {
        let input = *<&[u8; 4]>::try_from(&*chunk).expect("four scale values");
        for (dst, src) in FP4_SCALE_PERM.into_iter().enumerate() {
            chunk[dst] = input[src];
        }
    }

    let max_scaled = permuted
        .iter()
        .map(|&code| e4m3_value(code) * 128.0)
        .fold(0.0f32, f32::max);
    let scale_factor = if max_scaled > 0.0 && max_scaled < 448.0 * 128.0 {
        2.0f32.powf((448.0 * 128.0 / max_scaled).log2().floor())
    } else {
        1.0
    };

    for code in &mut permuted {
        let value = e4m3_value(*code) * scale_factor * 128.0;
        *code = if value < 2.0 {
            0
        } else {
            (positive_f32_to_f16_bits(value) >> 7) as u8
        };
    }
    (permuted, scale_factor)
}

fn positive_f32_to_f16_bits(value: f32) -> u16 {
    if value <= 0.0 || !value.is_finite() {
        return 0;
    }
    if value >= 65_504.0 {
        return 0x7bff;
    }
    let bits = value.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return 0;
    }
    let mantissa = bits & 0x7f_ffff;
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    let half_mantissa = if rounded & 0x80_0000 != 0 {
        exponent += 1;
        0
    } else {
        (rounded >> 13) as u16
    };
    if exponent >= 31 {
        0x7bff
    } else {
        ((exponent as u16) << 10) | half_mantissa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marlin_repack_preserves_buffer_sizes() {
        let n = 64;
        let k = 128;
        let weight = ModelOptNvfp4Linear {
            prefix: "test".to_string(),
            out_features: n,
            in_features: k,
            packed_weight: (0..n * k / 2).map(|idx| idx as u8).collect(),
            weight_scale: vec![0x38; n * k / 16],
            weight_scale_2: 0.25,
            input_scale: 0.5,
        };
        let repacked = MarlinNvfp4HostWeight::from_modelopt(&weight).expect("repack");
        assert_eq!(repacked.packed_weight.len() * 4, weight.packed_weight.len());
        assert_eq!(repacked.weight_scale.len(), weight.weight_scale.len());
        assert!(repacked.global_scale.is_finite());
    }

    #[test]
    fn marlin_weight_repack_matches_vllm_cuda_operator() {
        const N: usize = 64;
        const K: usize = 16;
        let mut source = vec![0u8; N * K / 2];
        for row in 0..N {
            for k_pack in 0..K / 8 {
                let word = (k_pack * N + row) as u32;
                let offset = row * (K / 2) + k_pack * 4;
                source[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let actual = repack_weight(&source, N, K);
        let expected_prefix = [
            1_077_970_944,
            1_364_297_728,
            1_650_624_512,
            1_936_951_296,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1_077_975_313,
            1_364_302_097,
            1_650_628_881,
            1_936_955_665,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(&actual[..expected_prefix.len()], &expected_prefix);
    }

    #[test]
    fn marlin_scale_repack_matches_vllm_preprocessing() {
        const N: usize = 64;
        const K: usize = 128;
        let groups = K / 16;
        let codes = [0x38, 0x30, 0x40, 0x28];
        let mut source = vec![0u8; N * groups];
        for row in 0..N {
            for group in 0..groups {
                source[row * groups + group] = codes[row % codes.len()];
            }
        }
        let (actual, factor) = repack_scales(&source, N, K);
        let expected_prefix = [
            232, 232, 232, 232, 232, 232, 232, 232, 224, 224, 224, 224, 224, 224, 224, 224, 240,
            240, 240, 240, 240, 240, 240, 240, 216, 216, 216, 216, 216, 216, 216, 216, 232, 232,
            232, 232, 232, 232, 232, 232, 224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240,
            240, 240, 240, 240, 240, 216, 216, 216, 216, 216, 216, 216, 216,
        ];
        assert_eq!(factor, 128.0);
        assert_eq!(&actual[..expected_prefix.len()], &expected_prefix);
    }
}
