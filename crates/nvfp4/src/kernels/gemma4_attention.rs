use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::ptr::null_mut;

const CUBIN: &[u8] = include_bytes!("../../native/gemma4_local_attention_sm121.cubin");
const KERNEL_NAME: &[u8] = b"gemma4_local_attention\0";
const HEAD_DIM: usize = 256;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 8;
const BLOCK_M: usize = 64;
const SHARED_MEMORY_BYTES: u32 = 73_728;

/// Loaded SM121 Triton kernel for Gemma 4's 256-wide local prefill attention.
pub struct Gemma4LocalPrefillAttention {
    module: ffi::CUmodule,
    function: ffi::CUfunction,
}

impl Gemma4LocalPrefillAttention {
    /// Loads the checked-in cubin into the current CUDA context.
    pub fn new() -> Result<Self> {
        let mut module = null_mut();
        let mut function = null_mut();
        unsafe {
            check_driver(
                "cuModuleLoadData(Gemma 4 local attention)",
                ffi::cuModuleLoadData(&mut module, CUBIN.as_ptr().cast()),
            )?;
            if let Err(error) = check_driver(
                "cuModuleGetFunction(Gemma 4 local attention)",
                ffi::cuModuleGetFunction(&mut function, module, KERNEL_NAME.as_ptr().cast()),
            ) {
                let _ = ffi::cuModuleUnload(module);
                return Err(error);
            }
            if let Err(error) = check_driver(
                "cuFuncSetAttribute(Gemma 4 local attention)",
                ffi::cuFuncSetAttribute(
                    function,
                    ffi::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    SHARED_MEMORY_BYTES as i32,
                ),
            ) {
                let _ = ffi::cuModuleUnload(module);
                return Err(error);
            }
        }
        Ok(Self { module, function })
    }

    /// Runs causal window attention over head-major BF16 Q, K, V, and output.
    #[allow(clippy::too_many_arguments)]
    pub fn run_on_stream(
        &self,
        query: &DeviceBuffer<u16>,
        key: &DeviceBuffer<u16>,
        value: &DeviceBuffer<u16>,
        mut output: DeviceOutput<'_, u16>,
        query_tokens: usize,
        key_tokens: usize,
        start_position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let query_values = QUERY_HEADS
            .checked_mul(query_tokens)
            .and_then(|values| values.checked_mul(HEAD_DIM))
            .unwrap_or(usize::MAX);
        let key_values = KV_HEADS
            .checked_mul(key_tokens)
            .and_then(|values| values.checked_mul(HEAD_DIM))
            .unwrap_or(usize::MAX);
        if query_tokens == 0
            || key_tokens == 0
            || start_position
                .checked_add(query_tokens)
                .is_none_or(|end| end > key_tokens)
            || [
                query_tokens,
                key_tokens,
                start_position,
                query_values,
                key_values,
            ]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
            || query.len() < query_values
            || key.len() < key_values
            || value.len() < key_values
            || output.len() < query_values
        {
            return Err(Error::Shape {
                label: "Gemma 4 local prefill attention",
                expected: format!(
                    "Q/O >= {query_values}, K/V >= {key_values}, start + query <= key"
                ),
                actual: format!(
                    "Q={} K={} V={} O={} query={query_tokens} key={key_tokens} start={start_position}",
                    query.len(),
                    key.len(),
                    value.len(),
                    output.len(),
                ),
            });
        }

        let mut query_ptr = query.as_const_ptr();
        let mut key_ptr = key.as_const_ptr();
        let mut value_ptr = value.as_const_ptr();
        let mut output_ptr = output.as_mut_ptr();
        let mut softmax_scale = (HEAD_DIM as f32).sqrt().recip() * std::f32::consts::LOG2_E;
        let mut query_tokens = query_tokens as u32;
        let mut key_tokens = key_tokens as u32;
        let mut start_position = start_position as u32;
        let mut stride_qt = HEAD_DIM as u32;
        let mut stride_qh = query_tokens * HEAD_DIM as u32;
        let mut stride_kt = HEAD_DIM as u32;
        let mut stride_kh = key_tokens * HEAD_DIM as u32;
        let mut stride_vt = 1u32;
        let mut stride_vh = key_tokens * HEAD_DIM as u32;
        let mut stride_vd = key_tokens;
        let mut stride_ot = HEAD_DIM as u32;
        let mut stride_oh = query_tokens * HEAD_DIM as u32;
        let mut global_scratch: *mut c_void = null_mut();
        let mut profile_scratch: *mut c_void = null_mut();
        let mut params = [
            parameter(&mut query_ptr),
            parameter(&mut key_ptr),
            parameter(&mut value_ptr),
            parameter(&mut output_ptr),
            parameter(&mut softmax_scale),
            parameter(&mut query_tokens),
            parameter(&mut key_tokens),
            parameter(&mut start_position),
            parameter(&mut stride_qt),
            parameter(&mut stride_qh),
            parameter(&mut stride_kt),
            parameter(&mut stride_kh),
            parameter(&mut stride_vt),
            parameter(&mut stride_vh),
            parameter(&mut stride_vd),
            parameter(&mut stride_ot),
            parameter(&mut stride_oh),
            parameter(&mut global_scratch),
            parameter(&mut profile_scratch),
        ];
        unsafe {
            check_driver(
                "cuLaunchKernel(Gemma 4 local attention)",
                ffi::cuLaunchKernel(
                    self.function,
                    1,
                    QUERY_HEADS as u32,
                    query_tokens.div_ceil(BLOCK_M as u32),
                    256,
                    1,
                    1,
                    SHARED_MEMORY_BYTES,
                    stream.as_raw(),
                    params.as_mut_ptr(),
                    null_mut(),
                ),
            )
        }
    }
}

impl Drop for Gemma4LocalPrefillAttention {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cuModuleUnload(self.module);
        }
    }
}

fn parameter<T>(value: &mut T) -> *mut c_void {
    std::ptr::from_mut(value).cast()
}

fn check_driver(call: &'static str, status: ffi::CUresult) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cuda(call, status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{bf16_to_f32, f32_to_bf16};

    #[test]
    fn local_attention_respects_causal_sliding_window() {
        let query_tokens = 4;
        let start_position = 1_100;
        let key_tokens = start_position + query_tokens;
        let query = DeviceBuffer::zeroed(QUERY_HEADS * query_tokens * HEAD_DIM).expect("query");
        let key = DeviceBuffer::zeroed(KV_HEADS * key_tokens * HEAD_DIM).expect("key");
        let mut value = vec![0u16; KV_HEADS * key_tokens * HEAD_DIM];
        for head in 0..KV_HEADS {
            for token in 0..key_tokens {
                let encoded = f32_to_bf16(token as f32 / 1_024.0);
                for dimension in 0..HEAD_DIM {
                    value[(head * HEAD_DIM + dimension) * key_tokens + token] = encoded;
                }
            }
        }
        let value = DeviceBuffer::from_host(&value).expect("value");
        let mut output =
            DeviceBuffer::zeroed(QUERY_HEADS * query_tokens * HEAD_DIM).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        Gemma4LocalPrefillAttention::new()
            .expect("kernel")
            .run_on_stream(
                &query,
                &key,
                &value,
                output.output(),
                query_tokens,
                key_tokens,
                start_position,
                &stream,
            )
            .expect("attention");
        let output = output.copy_to_host(&stream).expect("readback");
        for head in 0..QUERY_HEADS {
            for query_row in 0..query_tokens {
                let absolute_query = start_position + query_row;
                let first_key = absolute_query + 1 - 1_024;
                let expected = (first_key + absolute_query) as f32 / (2.0 * 1_024.0);
                let actual = bf16_to_f32(output[(head * query_tokens + query_row) * HEAD_DIM]);
                assert!(
                    (actual - expected).abs() <= 0.02,
                    "head={head} query={query_row} expected={expected} actual={actual}"
                );
            }
        }
    }

    #[test]
    fn local_attention_matches_cpu_reference() {
        let query_tokens = 5;
        let start_position = 3;
        let key_tokens = start_position + query_tokens;
        let query_host = (0..QUERY_HEADS * query_tokens * HEAD_DIM)
            .map(|index| f32_to_bf16(((index * 17 % 101) as f32 - 50.0) / 25.0))
            .collect::<Vec<_>>();
        let key_host = (0..KV_HEADS * key_tokens * HEAD_DIM)
            .map(|index| f32_to_bf16(((index * 13 % 97) as f32 - 48.0) / 24.0))
            .collect::<Vec<_>>();
        let value_token_major = (0..KV_HEADS * key_tokens * HEAD_DIM)
            .map(|index| f32_to_bf16(((index * 7 % 89) as f32 - 44.0) / 22.0))
            .collect::<Vec<_>>();
        let mut value_transposed = vec![0u16; value_token_major.len()];
        for head in 0..KV_HEADS {
            for token in 0..key_tokens {
                for dimension in 0..HEAD_DIM {
                    value_transposed[(head * HEAD_DIM + dimension) * key_tokens + token] =
                        value_token_major[(head * key_tokens + token) * HEAD_DIM + dimension];
                }
            }
        }
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let value = DeviceBuffer::from_host(&value_transposed).expect("value");
        let mut output =
            DeviceBuffer::zeroed(QUERY_HEADS * query_tokens * HEAD_DIM).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        Gemma4LocalPrefillAttention::new()
            .expect("kernel")
            .run_on_stream(
                &query,
                &key,
                &value,
                output.output(),
                query_tokens,
                key_tokens,
                start_position,
                &stream,
            )
            .expect("attention");
        let output = output.copy_to_host(&stream).expect("readback");
        let scale = (HEAD_DIM as f32).sqrt().recip();
        let mut max_error = 0.0f32;
        for head in 0..QUERY_HEADS {
            let key_head = head / (QUERY_HEADS / KV_HEADS);
            for query_row in 0..query_tokens {
                let key_end = start_position + query_row + 1;
                let mut scores = Vec::with_capacity(key_end);
                for token in 0..key_end {
                    let dot = (0..HEAD_DIM)
                        .map(|dimension| {
                            bf16_to_f32(
                                query_host
                                    [(head * query_tokens + query_row) * HEAD_DIM + dimension],
                            ) * bf16_to_f32(
                                key_host[(key_head * key_tokens + token) * HEAD_DIM + dimension],
                            )
                        })
                        .sum::<f32>()
                        * scale;
                    scores.push(dot);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let sum = weights.iter().sum::<f32>();
                for dimension in 0..HEAD_DIM {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| {
                            weight
                                * bf16_to_f32(
                                    value_token_major
                                        [(key_head * key_tokens + token) * HEAD_DIM + dimension],
                                )
                        })
                        .sum::<f32>()
                        / sum;
                    let actual = bf16_to_f32(
                        output[(head * query_tokens + query_row) * HEAD_DIM + dimension],
                    );
                    max_error = max_error.max((actual - expected).abs());
                }
            }
        }
        assert!(max_error <= 0.04, "max attention error={max_error}");
    }
}
