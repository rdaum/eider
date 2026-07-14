//! Compact SM12x FP4 cache storage and append-time quantization.

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceBuffer, DeviceOutput, Error, Result};

const K_TOKEN_TILE: usize = 8;
const V_TOKEN_BLOCK: usize = 16;
const MMA_K: usize = 64;
const MMA_N: usize = 8;
const COMPACT_TILE_BYTES: usize = MMA_N * MMA_K / 2;
const SCALE_BYTES_PER_TILE: usize = MMA_K / 16;

/// Persistent compact FP4 K/V storage for one attention layer.
///
/// Completed K groups are stored token-major in `[8 tokens, 64 dimensions]`
/// tiles with independent K16 scales for each token. Completed V blocks are
/// stored transposed in `[8 dimensions, 64 tokens]` tiles with independent
/// K16 scales for each dimension. The V layout needs a bounded f32 tail so a
/// token-axis scale block can be finalized without rewriting earlier entries.
pub struct Sm12xKvCache {
    key_values: DeviceBuffer<u8>,
    key_scales: DeviceBuffer<u8>,
    value_values: DeviceBuffer<u8>,
    value_scales: DeviceBuffer<u8>,
    key_tail: DeviceBuffer<f32>,
    value_tail: DeviceBuffer<f32>,
    max_tokens: usize,
    len: usize,
    kv_heads: usize,
    head_dim: usize,
}

/// Reusable device workspace for compact-cache FP4 attention.
pub struct Sm12xKvAttentionWorkspace {
    query_tiles: DeviceBuffer<u8>,
    query_scales: DeviceBuffer<u32>,
    scores: DeviceBuffer<f32>,
    probability_tiles: DeviceBuffer<u8>,
    probability_scales: DeviceBuffer<u32>,
    max_tokens: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Sm12xKvCache {
    /// Allocates an empty compact cache.
    pub fn new(max_tokens: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        if max_tokens == 0 || kv_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(MMA_K) {
            return Err(Error::Shape {
                label: "SM12x KV cache",
                expected: "non-zero sizes and head_dim multiple of 64".to_string(),
                actual: format!("max_tokens={max_tokens} kv_heads={kv_heads} head_dim={head_dim}"),
            });
        }
        u32::try_from(max_tokens).map_err(|_| Error::Shape {
            label: "SM12x KV cache max tokens",
            expected: "value fitting u32".to_string(),
            actual: max_tokens.to_string(),
        })?;
        u32::try_from(kv_heads).map_err(|_| Error::Shape {
            label: "SM12x KV cache heads",
            expected: "value fitting u32".to_string(),
            actual: kv_heads.to_string(),
        })?;
        u32::try_from(head_dim).map_err(|_| Error::Shape {
            label: "SM12x KV cache head dimension",
            expected: "value fitting u32".to_string(),
            actual: head_dim.to_string(),
        })?;

        let key_tiles = checked_product(
            "SM12x K tile count",
            &[
                kv_heads,
                max_tokens.div_ceil(K_TOKEN_TILE),
                head_dim / MMA_K,
            ],
        )?;
        let value_tiles = checked_product(
            "SM12x V tile count",
            &[kv_heads, head_dim / MMA_N, max_tokens.div_ceil(MMA_K)],
        )?;
        let tail_values = checked_product("SM12x KV tail", &[V_TOKEN_BLOCK, kv_heads, head_dim])?;

        Ok(Self {
            key_values: DeviceBuffer::zeroed(checked_product(
                "SM12x K values",
                &[key_tiles, COMPACT_TILE_BYTES],
            )?)?,
            key_scales: DeviceBuffer::zeroed(checked_product(
                "SM12x K scales",
                &[key_tiles, K_TOKEN_TILE, SCALE_BYTES_PER_TILE],
            )?)?,
            value_values: DeviceBuffer::zeroed(checked_product(
                "SM12x V values",
                &[value_tiles, COMPACT_TILE_BYTES],
            )?)?,
            value_scales: DeviceBuffer::zeroed(checked_product(
                "SM12x V scales",
                &[value_tiles, MMA_N, SCALE_BYTES_PER_TILE],
            )?)?,
            key_tail: DeviceBuffer::zeroed(tail_values)?,
            value_tail: DeviceBuffer::zeroed(tail_values)?,
            max_tokens,
            len: 0,
            kv_heads,
            head_dim,
        })
    }

    /// Returns the number of appended tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether no tokens have been appended.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the allocated token capacity.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns the number of K tokens finalized into compact 8-token tiles.
    pub fn compact_key_tokens(&self) -> usize {
        self.len / K_TOKEN_TILE * K_TOKEN_TILE
    }

    /// Returns the number of V tokens finalized into compact 16-token blocks.
    pub fn compact_value_tokens(&self) -> usize {
        self.len / V_TOKEN_BLOCK * V_TOKEN_BLOCK
    }

    /// Returns the number of incomplete f32 K rows.
    pub fn key_tail_len(&self) -> usize {
        self.len % K_TOKEN_TILE
    }

    /// Returns the number of incomplete f32 V rows.
    pub fn value_tail_len(&self) -> usize {
        self.len % V_TOKEN_BLOCK
    }

    /// Returns compact token-major K codes.
    pub fn key_values(&self) -> &DeviceBuffer<u8> {
        &self.key_values
    }

    /// Returns compact K scale bytes, four per token in each 8-by-64 tile.
    pub fn key_scales(&self) -> &DeviceBuffer<u8> {
        &self.key_scales
    }

    /// Returns compact transposed-V codes.
    pub fn value_values(&self) -> &DeviceBuffer<u8> {
        &self.value_values
    }

    /// Returns compact V scale bytes, four per dimension in each 8-by-64 tile.
    pub fn value_scales(&self) -> &DeviceBuffer<u8> {
        &self.value_scales
    }

    /// Returns the 16-row circular f32 K staging allocation.
    pub fn key_tail(&self) -> &DeviceBuffer<f32> {
        &self.key_tail
    }

    /// Returns the 16-row circular f32 V staging allocation.
    pub fn value_tail(&self) -> &DeviceBuffer<f32> {
        &self.value_tail
    }

    /// Appends one device-resident K/V row and synchronizes before returning.
    pub fn append(&mut self, key: &DeviceBuffer<f32>, value: &DeviceBuffer<f32>) -> Result<()> {
        let stream = CudaStream::new_blocking()?;
        self.append_on_stream(key, value, &stream)?;
        stream.synchronize()
    }

    /// Enqueues one K/V append and advances the logical cache length.
    pub fn append_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_heads * self.head_dim;
        if key.len() != width || value.len() != width {
            return Err(Error::Shape {
                label: "SM12x KV append",
                expected: format!("one K/V row of {width} values"),
                actual: format!("key={} value={}", key.len(), value.len()),
            });
        }
        if self.len >= self.max_tokens {
            return Err(Error::Shape {
                label: "SM12x KV append capacity",
                expected: format!("fewer than {} rows", self.max_tokens),
                actual: format!("{} rows", self.len + 1),
            });
        }

        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    self.key_values.as_mut_ptr().cast(),
                    self.key_scales.as_mut_ptr().cast(),
                    self.value_values.as_mut_ptr().cast(),
                    self.value_scales.as_mut_ptr().cast(),
                    self.key_tail.as_mut_ptr().cast(),
                    self.value_tail.as_mut_ptr().cast(),
                    self.len as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        self.len += 1;
        Ok(())
    }

    /// Shortens the logical cache without moving device memory.
    ///
    /// Subsequent appends overwrite any compact tiles or staging rows that
    /// become reachable again. This is suitable for speculative decode
    /// rollback and for reusing a measured decode position.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(Error::Shape {
                label: "SM12x KV cache truncate",
                expected: format!("at most {} rows", self.len),
                actual: format!("{len} rows"),
            });
        }
        self.len = len;
        Ok(())
    }
}

impl Sm12xKvAttentionWorkspace {
    /// Allocates scratch for a cache with the given shape.
    pub fn new(max_tokens: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        if max_tokens == 0 || kv_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(MMA_K) {
            return Err(Error::Shape {
                label: "SM12x KV attention workspace",
                expected: "non-zero sizes and head_dim multiple of 64".to_string(),
                actual: format!("max_tokens={max_tokens} kv_heads={kv_heads} head_dim={head_dim}"),
            });
        }
        let head_k_tiles = head_dim / MMA_K;
        let context_tiles = max_tokens.div_ceil(MMA_K);
        let query_tile_count = checked_product("SM12x KV query tiles", &[kv_heads, head_k_tiles])?;
        let probability_tile_count =
            checked_product("SM12x KV probability tiles", &[kv_heads, context_tiles])?;
        Ok(Self {
            query_tiles: DeviceBuffer::zeroed(checked_product(
                "SM12x KV query tile bytes",
                &[query_tile_count, 512],
            )?)?,
            query_scales: DeviceBuffer::zeroed(query_tile_count)?,
            scores: DeviceBuffer::zeroed(checked_product(
                "SM12x KV scores",
                &[kv_heads, MMA_N, max_tokens],
            )?)?,
            probability_tiles: DeviceBuffer::zeroed(checked_product(
                "SM12x KV probability tile bytes",
                &[probability_tile_count, 512],
            )?)?,
            probability_scales: DeviceBuffer::zeroed(probability_tile_count)?,
            max_tokens,
            kv_heads,
            head_dim,
        })
    }

    /// Enqueues Q-to-FP4, QK, f32 online softmax, P-to-FP4, and PV.
    pub fn attention_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len == 0 {
            return Err(Error::Shape {
                label: "SM12x KV attention cache length",
                expected: "at least one token".to_string(),
                actual: "0".to_string(),
            });
        }
        if cache.max_tokens != self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x KV attention cache",
                expected: format!(
                    "max_tokens={} kv_heads={} head_dim={}",
                    self.max_tokens, self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "max_tokens={} kv_heads={} head_dim={}",
                    cache.max_tokens, cache.kv_heads, cache.head_dim
                ),
            });
        }
        let output_values = self.kv_heads * MMA_N * self.head_dim;
        if query.len() != output_values || output.len() != output_values {
            return Err(Error::Shape {
                label: "SM12x KV attention query/output",
                expected: format!("{output_values} values"),
                actual: format!("query={} output={}", query.len(), output.len()),
            });
        }

        unsafe {
            check_cuda(
                "infer_sm12x_kv_attention_on_stream",
                crate::ffi::infer_sm12x_kv_attention_on_stream(
                    query.as_const_ptr().cast(),
                    cache.key_values.as_const_ptr().cast(),
                    cache.key_scales.as_const_ptr().cast(),
                    cache.key_tail.as_const_ptr().cast(),
                    cache.value_values.as_const_ptr().cast(),
                    cache.value_scales.as_const_ptr().cast(),
                    cache.value_tail.as_const_ptr().cast(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cache.len as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns the f32 score/probability scratch for validation.
    pub fn scores(&self) -> &DeviceBuffer<f32> {
        &self.scores
    }
}

fn checked_product(label: &'static str, values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product.checked_mul(*value).ok_or_else(|| Error::Shape {
            label,
            expected: "size without overflow".to_string(),
            actual: format!("factors={values:?}"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_nibble(packed: &mut [u8], index: usize, value: u8) {
        let byte = &mut packed[index / 2];
        if index & 1 == 0 {
            *byte = (*byte & 0xf0) | (value & 0x0f);
        } else {
            *byte = (*byte & 0x0f) | ((value & 0x0f) << 4);
        }
    }

    fn scale_for(values: impl Iterator<Item = f32>) -> (u8, f32) {
        let max_abs = values
            .filter(|value| value.is_finite())
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        let code = if max_abs == 0.0 {
            0
        } else {
            crate::format::ue4m3_code(max_abs / 6.0)
        };
        (code, crate::format::e4m3_value(code))
    }

    #[test]
    fn compact_append_matches_k_and_transposed_v_reference_with_tails() {
        const MAX_TOKENS: usize = 65;
        const TOKENS: usize = 17;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        let width = KV_HEADS * HEAD_DIM;
        let key_rows = (0..TOKENS)
            .map(|token| {
                (0..width)
                    .map(|column| ((token * 37 + column * 13) % 257) as f32 / 64.0 - 2.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let value_rows = (0..TOKENS)
            .map(|token| {
                (0..width)
                    .map(|column| ((token * 29 + column * 17) % 251) as f32 / 80.0 - 1.5)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        for token in 0..TOKENS {
            let key = DeviceBuffer::from_host(&key_rows[token]).expect("key row");
            let value = DeviceBuffer::from_host(&value_rows[token]).expect("value row");
            cache.append(&key, &value).expect("append");
        }

        assert_eq!(cache.len(), TOKENS);
        assert_eq!(cache.compact_key_tokens(), 16);
        assert_eq!(cache.compact_value_tokens(), 16);
        assert_eq!(cache.key_tail_len(), 1);
        assert_eq!(cache.value_tail_len(), 1);

        let stream = CudaStream::new_blocking().expect("stream");
        let key_values = cache.key_values.copy_to_host(&stream).expect("K values");
        let key_scales = cache.key_scales.copy_to_host(&stream).expect("K scales");
        let value_values = cache.value_values.copy_to_host(&stream).expect("V values");
        let value_scales = cache.value_scales.copy_to_host(&stream).expect("V scales");

        let key_token_tiles = MAX_TOKENS.div_ceil(K_TOKEN_TILE);
        let mut expected_key_values = vec![0u8; key_values.len()];
        let mut expected_key_scales = vec![0u8; key_scales.len()];
        for head in 0..KV_HEADS {
            for token_tile in 0..2 {
                for scale_block in 0..4 {
                    for token in 0..8 {
                        let block_values = (0..16).map(|offset| {
                            key_rows[token_tile * 8 + token]
                                [head * HEAD_DIM + scale_block * 16 + offset]
                        });
                        let (scale_code, scale) = scale_for(block_values);
                        let tile = head * key_token_tiles + token_tile;
                        expected_key_scales[(tile * K_TOKEN_TILE + token) * 4 + scale_block] =
                            scale_code;
                        for offset in 0..16 {
                            let value = key_rows[token_tile * 8 + token]
                                [head * HEAD_DIM + scale_block * 16 + offset];
                            let code = crate::format::e2m1_code(if scale == 0.0 {
                                0.0
                            } else {
                                value / scale
                            });
                            set_nibble(
                                &mut expected_key_values
                                    [tile * COMPACT_TILE_BYTES..(tile + 1) * COMPACT_TILE_BYTES],
                                token * 64 + scale_block * 16 + offset,
                                code,
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(&*key_values, expected_key_values);
        assert_eq!(&*key_scales, expected_key_scales);

        let value_token_tiles = MAX_TOKENS.div_ceil(MMA_K);
        let value_dim_tiles = HEAD_DIM / MMA_N;
        let mut expected_value_values = vec![0u8; value_values.len()];
        let mut expected_value_scales = vec![0u8; value_scales.len()];
        for head in 0..KV_HEADS {
            for dim_tile in 0..value_dim_tiles {
                for dim in 0..8 {
                    let (scale_code, scale) = scale_for(
                        value_rows
                            .iter()
                            .take(16)
                            .map(|row| row[head * HEAD_DIM + dim_tile * 8 + dim]),
                    );
                    let tile = (head * value_dim_tiles + dim_tile) * value_token_tiles;
                    expected_value_scales[(tile * MMA_N + dim) * 4] = scale_code;
                    for token in 0..16 {
                        let value = value_rows[token][head * HEAD_DIM + dim_tile * 8 + dim];
                        let code = crate::format::e2m1_code(if scale == 0.0 {
                            0.0
                        } else {
                            value / scale
                        });
                        set_nibble(
                            &mut expected_value_values
                                [tile * COMPACT_TILE_BYTES..(tile + 1) * COMPACT_TILE_BYTES],
                            dim * 64 + token,
                            code,
                        );
                    }
                }
            }
        }
        assert_eq!(&*value_values, expected_value_values);
        assert_eq!(&*value_scales, expected_value_scales);

        let key_tail = cache.key_tail.copy_to_host(&stream).expect("K tail");
        let value_tail = cache.value_tail.copy_to_host(&stream).expect("V tail");
        for slot in 0..V_TOKEN_BLOCK {
            let source_token = if slot == 0 { 16 } else { slot };
            assert_eq!(
                &key_tail[slot * width..(slot + 1) * width],
                &key_rows[source_token]
            );
            assert_eq!(
                &value_tail[slot * width..(slot + 1) * width],
                &value_rows[source_token]
            );
        }
    }

    #[test]
    fn compact_mma_attention_tracks_f32_gqa_through_incomplete_tails() {
        const MAX_TOKENS: usize = 64;
        const TOKENS: usize = 17;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = KV_HEADS * MMA_N;
        let width = KV_HEADS * HEAD_DIM;
        let key_host = (0..TOKENS * width)
            .map(|index| ((index * 31 % 251) as f32 - 125.0) / 512.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * width)
            .map(|index| ((index * 47 % 257) as f32 - 128.0) / 384.0)
            .collect::<Vec<_>>();
        let query_host = (0..Q_HEADS * HEAD_DIM)
            .map(|index| ((index * 19 % 239) as f32 - 119.0) / 448.0)
            .collect::<Vec<_>>();

        let mut cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        for token in 0..TOKENS {
            let key = DeviceBuffer::from_host(&key_host[token * width..(token + 1) * width])
                .expect("key row");
            let value = DeviceBuffer::from_host(&value_host[token * width..(token + 1) * width])
                .expect("value row");
            cache.append(&key, &value).expect("append");
        }

        let stream = CudaStream::new_blocking().expect("stream");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let key_f32 = DeviceBuffer::from_host(&key_host).expect("K cache");
        let value_f32 = DeviceBuffer::from_host(&value_host).expect("V cache");
        let mut expected = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("f32 output");
        crate::cached_gqa_attention_f32_into_on_stream(
            &query,
            &key_f32,
            &value_f32,
            expected.output(),
            TOKENS,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("f32 attention");
        let mut actual = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("FP4 output");
        let mut workspace =
            Sm12xKvAttentionWorkspace::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("workspace");
        workspace
            .attention_into_on_stream(&cache, &query, actual.output(), &stream)
            .expect("compact attention");
        let expected = expected.copy_to_host(&stream).expect("f32 copy");
        let actual = actual.copy_to_host(&stream).expect("FP4 copy");
        let max_abs = expected
            .iter()
            .zip(actual.iter())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.25,
            "compact FP4 attention error too large: max_abs={max_abs}"
        );
    }

    #[test]
    fn compact_mma_attention_tracks_f32_gqa_at_qwen_shape() {
        const MAX_TOKENS: usize = 128;
        const TOKENS: usize = 65;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const Q_HEADS: usize = KV_HEADS * MMA_N;
        let width = KV_HEADS * HEAD_DIM;
        let key_host = (0..TOKENS * width)
            .map(|index| ((index * 31 % 509) as f32 - 254.0) / 768.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * width)
            .map(|index| ((index * 43 % 503) as f32 - 251.0) / 640.0)
            .collect::<Vec<_>>();
        let query_host = (0..Q_HEADS * HEAD_DIM)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) / 576.0)
            .collect::<Vec<_>>();

        let mut cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        for token in 0..TOKENS {
            let key = DeviceBuffer::from_host(&key_host[token * width..(token + 1) * width])
                .expect("key row");
            let value = DeviceBuffer::from_host(&value_host[token * width..(token + 1) * width])
                .expect("value row");
            cache.append(&key, &value).expect("append");
        }

        let stream = CudaStream::new_blocking().expect("stream");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let key_f32 = DeviceBuffer::from_host(&key_host).expect("K cache");
        let value_f32 = DeviceBuffer::from_host(&value_host).expect("V cache");
        let mut expected = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("f32 output");
        crate::cached_gqa_attention_f32_into_on_stream(
            &query,
            &key_f32,
            &value_f32,
            expected.output(),
            TOKENS,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("f32 attention");
        let mut actual = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("FP4 output");
        let mut workspace =
            Sm12xKvAttentionWorkspace::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("workspace");
        workspace
            .attention_into_on_stream(&cache, &query, actual.output(), &stream)
            .expect("compact attention");
        let expected = expected.copy_to_host(&stream).expect("f32 copy");
        let actual = actual.copy_to_host(&stream).expect("FP4 copy");
        let max_abs = expected
            .iter()
            .zip(actual.iter())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.25,
            "Qwen-shape compact FP4 attention error too large: max_abs={max_abs}"
        );
    }
}
