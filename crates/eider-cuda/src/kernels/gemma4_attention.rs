use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::kernels::sm12x_kv_cache::Sm12xKvCache;

const HEAD_DIM: usize = 256;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 8;

/// Native CUDA kernel for Gemma 4's 256-wide local prefill attention.
pub struct Gemma4LocalPrefillAttention;

impl Gemma4LocalPrefillAttention {
    /// Creates the stateless native CUDA launcher.
    pub fn new() -> Result<Self> {
        Ok(Self)
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

        unsafe {
            check_cuda(
                "infer_gemma4_local_attention_bf16_on_stream",
                ffi::infer_gemma4_local_attention_bf16_on_stream(
                    query.as_const_ptr().cast(),
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    query_tokens as u32,
                    key_tokens as u32,
                    start_position as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Runs causal window attention directly from the compact FP4 K/V cache.
    pub fn run_compact_on_stream(
        &self,
        query: &DeviceBuffer<u16>,
        cache: &Sm12xKvCache,
        mut output: DeviceOutput<'_, u16>,
        query_tokens: usize,
        start_position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let query_values = QUERY_HEADS
            .checked_mul(query_tokens)
            .and_then(|values| values.checked_mul(HEAD_DIM))
            .unwrap_or(usize::MAX);
        if query_tokens == 0
            || cache.kv_heads() != KV_HEADS
            || cache.head_dim() != HEAD_DIM
            || start_position
                .checked_add(query_tokens)
                .is_none_or(|end| end > cache.len())
            || [
                query_tokens,
                cache.len(),
                cache.max_tokens(),
                start_position,
                query_values,
            ]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
            || query.len() < query_values
            || output.len() < query_values
        {
            return Err(Error::Shape {
                label: "Gemma 4 compact local prefill attention",
                expected: format!(
                    "Q/O >= {query_values}, 8 KV heads, head dim 256, start + query <= cache"
                ),
                actual: format!(
                    "Q={} O={} query={query_tokens} cache={} capacity={} heads={} dim={} start={start_position}",
                    query.len(),
                    output.len(),
                    cache.len(),
                    cache.max_tokens(),
                    cache.kv_heads(),
                    cache.head_dim(),
                ),
            });
        }

        let parts = cache.compact_parts()?;
        unsafe {
            check_cuda(
                "infer_gemma4_local_attention_compact_on_stream",
                ffi::infer_gemma4_local_attention_compact_on_stream(
                    query.as_const_ptr().cast(),
                    parts.key_values.as_const_ptr(),
                    parts.key_scales.as_const_ptr(),
                    parts.value_values.as_const_ptr(),
                    parts.value_scales.as_const_ptr(),
                    parts.key_tail.as_const_ptr(),
                    parts.value_tail.as_const_ptr(),
                    output.as_mut_ptr().cast(),
                    query_tokens as u32,
                    cache.len() as u32,
                    cache.max_tokens() as u32,
                    start_position as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{bf16_to_f32, f32_to_bf16};

    #[test]
    fn local_attention_respects_causal_sliding_window() {
        let query_tokens = 65;
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

    #[test]
    fn compact_attention_matches_unpacked_cache() {
        let query_tokens = 65;
        let start_position = 1_100;
        let cache_tokens = start_position + query_tokens;
        let query_host = (0..QUERY_HEADS * query_tokens * HEAD_DIM)
            .map(|index| f32_to_bf16(((index * 17 % 101) as f32 - 50.0) / 50.0))
            .collect::<Vec<_>>();
        let key_host = (0..KV_HEADS * cache_tokens * HEAD_DIM)
            .map(|index| ((index * 13 % 97) as f32 - 48.0) / 48.0)
            .collect::<Vec<_>>();
        let value_host = (0..KV_HEADS * cache_tokens * HEAD_DIM)
            .map(|index| ((index * 7 % 89) as f32 - 44.0) / 44.0)
            .collect::<Vec<_>>();
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let value = DeviceBuffer::from_host(&value_host).expect("value");
        let mut cache = Sm12xKvCache::new(cache_tokens, KV_HEADS, HEAD_DIM).expect("cache");
        let stream = CudaStream::new_non_blocking().expect("stream");
        cache
            .append_rows_at_offset_on_stream(&key, &value, 0, cache_tokens, &stream)
            .expect("append cache");
        let mut unpacked_key =
            DeviceBuffer::zeroed(KV_HEADS * cache_tokens * HEAD_DIM).expect("unpacked key");
        let mut unpacked_value =
            DeviceBuffer::zeroed(KV_HEADS * cache_tokens * HEAD_DIM).expect("unpacked value");
        cache
            .unpack_bf16_on_stream(unpacked_key.output(), unpacked_value.output(), &stream)
            .expect("unpack cache");
        let mut reference =
            DeviceBuffer::zeroed(QUERY_HEADS * query_tokens * HEAD_DIM).expect("reference");
        let mut compact =
            DeviceBuffer::zeroed(QUERY_HEADS * query_tokens * HEAD_DIM).expect("compact");
        let attention = Gemma4LocalPrefillAttention::new().expect("attention");
        attention
            .run_on_stream(
                &query,
                &unpacked_key,
                &unpacked_value,
                reference.output(),
                query_tokens,
                cache_tokens,
                start_position,
                &stream,
            )
            .expect("unpacked attention");
        attention
            .run_compact_on_stream(
                &query,
                &cache,
                compact.output(),
                query_tokens,
                start_position,
                &stream,
            )
            .expect("compact attention");
        let reference = reference.copy_to_host(&stream).expect("reference readback");
        let compact = compact.copy_to_host(&stream).expect("compact readback");
        let max_error = reference
            .iter()
            .zip(compact)
            .map(|(reference, compact)| (bf16_to_f32(*reference) - bf16_to_f32(compact)).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 0.01, "compact attention max error={max_error}");
    }
}
