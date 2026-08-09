//! Compact SM12x FP4 cache storage and append-time quantization.

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceBuffer, DeviceOutput, Error, Result};
use std::mem::{align_of, size_of};

const K_TOKEN_TILE: usize = 8;
const V_TOKEN_BLOCK: usize = 16;
const MMA_K: usize = 64;
const MMA_N: usize = 8;
const COMPACT_TILE_BYTES: usize = MMA_N * MMA_K / 2;
const SCALE_BYTES_PER_TILE: usize = MMA_K / 16;
const PV_SPLIT_CAPACITY: usize = 32;
const PV_SPLIT_MIN_TOKENS: usize = 1_024;

fn pv_split_count(tokens: usize) -> usize {
    if tokens < PV_SPLIT_MIN_TOKENS {
        1
    } else {
        PV_SPLIT_CAPACITY
    }
}

/// Persistent compact FP4 K/V storage for one attention layer.
///
/// Completed K groups are stored token-major in `[8 tokens, 64 dimensions]`
/// tiles with independent K16 scales for each token. Completed V blocks are
/// stored transposed in `[8 dimensions, 64 tokens]` tiles with independent
/// K16 scales for each dimension. The V layout needs a bounded f32 tail so a
/// token-axis scale block can be finalized without rewriting earlier entries.
pub struct Sm12xKvCache {
    storage: DeviceBuffer<u8>,
    layout: Sm12xKvCacheLayout,
    max_tokens: usize,
    len: usize,
    kv_heads: usize,
    head_dim: usize,
}

struct Sm12xKvCacheLayout {
    key_values: usize,
    key_scales: usize,
    value_values: usize,
    value_scales: usize,
    key_tail: usize,
    value_tail: usize,
    #[cfg(test)]
    key_values_bytes: usize,
    #[cfg(test)]
    key_scales_bytes: usize,
    #[cfg(test)]
    value_values_bytes: usize,
    #[cfg(test)]
    value_scales_bytes: usize,
    #[cfg(test)]
    tail_values: usize,
    total_bytes: usize,
}

pub(crate) struct Sm12xKvCacheParts {
    pub key_values: *const u8,
    pub key_scales: *const u8,
    pub value_values: *const u8,
    pub value_scales: *const u8,
    pub key_tail: *const f32,
    pub value_tail: *const f32,
}

/// Reusable device workspace for compact-cache FP4 attention.
pub struct Sm12xKvAttentionWorkspace {
    query_tiles: DeviceBuffer<u8>,
    query_scales: DeviceBuffer<u32>,
    scores: DeviceBuffer<f32>,
    probability_tiles: DeviceBuffer<u8>,
    probability_scales: DeviceBuffer<u32>,
    pv_partials: DeviceBuffer<f32>,
    max_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal_row_capacity: usize,
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

        let layout = Self::layout(max_tokens, kv_heads, head_dim)?;

        Ok(Self {
            // Cache kernels only read positions below `len`. Append writes a
            // tail row before publishing the new length, and aligned restore
            // writes every compact tile covered by the restored length.
            storage: DeviceBuffer::uninitialized(layout.total_bytes)?,
            layout,
            max_tokens,
            len: 0,
            kv_heads,
            head_dim,
        })
    }

    fn layout(max_tokens: usize, kv_heads: usize, head_dim: usize) -> Result<Sm12xKvCacheLayout> {
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
        let key_values_bytes = checked_product("SM12x K values", &[key_tiles, COMPACT_TILE_BYTES])?;
        let key_scales_bytes = checked_product(
            "SM12x K scales",
            &[key_tiles, K_TOKEN_TILE, SCALE_BYTES_PER_TILE],
        )?;
        let value_values_bytes =
            checked_product("SM12x V values", &[value_tiles, COMPACT_TILE_BYTES])?;
        let value_scales_bytes = checked_product(
            "SM12x V scales",
            &[value_tiles, MMA_N, SCALE_BYTES_PER_TILE],
        )?;
        let tail_bytes = checked_product("SM12x KV tail bytes", &[tail_values, size_of::<f32>()])?;
        let mut offset = 0usize;
        let key_values = offset;
        offset = checked_sum("SM12x K values end", offset, key_values_bytes)?;
        let key_scales = offset;
        offset = checked_sum("SM12x K scales end", offset, key_scales_bytes)?;
        let value_values = offset;
        offset = checked_sum("SM12x V values end", offset, value_values_bytes)?;
        let value_scales = offset;
        offset = checked_sum("SM12x V scales end", offset, value_scales_bytes)?;
        let key_tail = align_up(offset, align_of::<f32>())?;
        offset = checked_sum("SM12x K tail end", key_tail, tail_bytes)?;
        let value_tail = offset;
        let total_bytes = checked_sum("SM12x V tail end", value_tail, tail_bytes)?;
        Ok(Sm12xKvCacheLayout {
            key_values,
            key_scales,
            value_values,
            value_scales,
            key_tail,
            value_tail,
            #[cfg(test)]
            key_values_bytes,
            #[cfg(test)]
            key_scales_bytes,
            #[cfg(test)]
            value_values_bytes,
            #[cfg(test)]
            value_scales_bytes,
            #[cfg(test)]
            tail_values,
            total_bytes,
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

    pub(crate) fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns the number of device bytes owned by this cache.
    pub fn device_bytes(&self) -> usize {
        self.storage.device_bytes()
    }

    pub(crate) fn compact_parts(&self) -> Sm12xKvCacheParts {
        Sm12xKvCacheParts {
            key_values: self.key_values_ptr(),
            key_scales: self.key_scales_ptr(),
            value_values: self.value_values_ptr(),
            value_scales: self.value_scales_ptr(),
            key_tail: self.key_tail_ptr(),
            value_tail: self.value_tail_ptr(),
        }
    }

    /// Returns the device bytes this cache shape would allocate at another capacity.
    pub fn device_bytes_for_capacity(&self, max_tokens: usize) -> Result<usize> {
        if max_tokens == 0 || max_tokens > u32::MAX as usize {
            return Err(Error::Shape {
                label: "SM12x KV cache byte estimate",
                expected: "token capacity in 1..=u32::MAX".to_string(),
                actual: max_tokens.to_string(),
            });
        }
        Ok(Self::layout(max_tokens, self.kv_heads, self.head_dim)?.total_bytes)
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

    fn byte_ptr(&self, offset: usize) -> *const u8 {
        unsafe { self.storage.as_const_ptr().cast::<u8>().add(offset) }
    }

    fn byte_mut_ptr(&mut self, offset: usize) -> *mut u8 {
        unsafe { self.storage.as_mut_ptr().cast::<u8>().add(offset) }
    }

    fn key_values_ptr(&self) -> *const u8 {
        self.byte_ptr(self.layout.key_values)
    }

    fn key_scales_ptr(&self) -> *const u8 {
        self.byte_ptr(self.layout.key_scales)
    }

    fn value_values_ptr(&self) -> *const u8 {
        self.byte_ptr(self.layout.value_values)
    }

    fn value_scales_ptr(&self) -> *const u8 {
        self.byte_ptr(self.layout.value_scales)
    }

    fn key_tail_ptr(&self) -> *const f32 {
        self.byte_ptr(self.layout.key_tail).cast()
    }

    fn value_tail_ptr(&self) -> *const f32 {
        self.byte_ptr(self.layout.value_tail).cast()
    }

    fn key_values_mut_ptr(&mut self) -> *mut u8 {
        let offset = self.layout.key_values;
        self.byte_mut_ptr(offset)
    }

    fn key_scales_mut_ptr(&mut self) -> *mut u8 {
        let offset = self.layout.key_scales;
        self.byte_mut_ptr(offset)
    }

    fn value_values_mut_ptr(&mut self) -> *mut u8 {
        let offset = self.layout.value_values;
        self.byte_mut_ptr(offset)
    }

    fn value_scales_mut_ptr(&mut self) -> *mut u8 {
        let offset = self.layout.value_scales;
        self.byte_mut_ptr(offset)
    }

    fn key_tail_mut_ptr(&mut self) -> *mut f32 {
        let offset = self.layout.key_tail;
        self.byte_mut_ptr(offset).cast()
    }

    fn value_tail_mut_ptr(&mut self) -> *mut f32 {
        let offset = self.layout.value_tail;
        self.byte_mut_ptr(offset).cast()
    }

    #[cfg(test)]
    fn copy_region_to_host<T: Copy + Default>(
        &self,
        pointer: *const T,
        len: usize,
        stream: &CudaStream,
    ) -> Result<Vec<T>> {
        stream.synchronize()?;
        let mut values = vec![T::default(); len];
        unsafe {
            check_cuda(
                "cudaMemcpy(D2H cache region)",
                crate::ffi::cudaMemcpy(
                    values.as_mut_ptr().cast(),
                    pointer.cast(),
                    len * size_of::<T>(),
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
            )?;
        }
        Ok(values)
    }

    #[cfg(test)]
    fn key_values_to_host(&self, stream: &CudaStream) -> Result<Vec<u8>> {
        self.copy_region_to_host(self.key_values_ptr(), self.layout.key_values_bytes, stream)
    }

    #[cfg(test)]
    fn key_scales_to_host(&self, stream: &CudaStream) -> Result<Vec<u8>> {
        self.copy_region_to_host(self.key_scales_ptr(), self.layout.key_scales_bytes, stream)
    }

    #[cfg(test)]
    fn value_values_to_host(&self, stream: &CudaStream) -> Result<Vec<u8>> {
        self.copy_region_to_host(
            self.value_values_ptr(),
            self.layout.value_values_bytes,
            stream,
        )
    }

    #[cfg(test)]
    fn value_scales_to_host(&self, stream: &CudaStream) -> Result<Vec<u8>> {
        self.copy_region_to_host(
            self.value_scales_ptr(),
            self.layout.value_scales_bytes,
            stream,
        )
    }

    #[cfg(test)]
    fn key_tail_to_host(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        self.copy_region_to_host(self.key_tail_ptr(), self.layout.tail_values, stream)
    }

    #[cfg(test)]
    fn value_tail_to_host(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        self.copy_region_to_host(self.value_tail_ptr(), self.layout.tail_values, stream)
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
        self.append_at_on_stream(key, value, self.len, stream)
    }

    /// Enqueues one K/V append at an explicit host-visible cache position.
    pub fn append_at_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        position: usize,
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
        if position >= self.max_tokens {
            return Err(Error::Shape {
                label: "SM12x KV append capacity",
                expected: format!("fewer than {} rows", self.max_tokens),
                actual: format!("{} rows", position + 1),
            });
        }

        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    self.key_values_mut_ptr(),
                    self.key_scales_mut_ptr(),
                    self.value_values_mut_ptr(),
                    self.value_scales_mut_ptr(),
                    self.key_tail_mut_ptr(),
                    self.value_tail_mut_ptr(),
                    position as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        self.len = position + 1;
        Ok(())
    }

    /// Enqueues one K/V append from row offsets in larger dense buffers.
    pub fn append_at_offsets_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        key_offset: usize,
        value: &DeviceBuffer<f32>,
        value_offset: usize,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_heads * self.head_dim;
        if key_offset
            .checked_add(width)
            .is_none_or(|end| end > key.len())
            || value_offset
                .checked_add(width)
                .is_none_or(|end| end > value.len())
        {
            return Err(Error::Shape {
                label: "SM12x KV append offsets",
                expected: format!("{width} readable values at each row offset"),
                actual: format!(
                    "key_len={} key_offset={key_offset} value_len={} value_offset={value_offset}",
                    key.len(),
                    value.len()
                ),
            });
        }
        if position >= self.max_tokens {
            return Err(Error::Shape {
                label: "SM12x KV append capacity",
                expected: format!("fewer than {} rows", self.max_tokens),
                actual: format!("{} rows", position + 1),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_on_stream(
                    key.as_const_ptr().cast::<f32>().add(key_offset),
                    value.as_const_ptr().cast::<f32>().add(value_offset),
                    self.key_values_mut_ptr(),
                    self.key_scales_mut_ptr(),
                    self.value_values_mut_ptr(),
                    self.value_scales_mut_ptr(),
                    self.key_tail_mut_ptr(),
                    self.value_tail_mut_ptr(),
                    position as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        self.len = position + 1;
        Ok(())
    }

    /// Enqueues a contiguous set of device-resident K/V rows.
    ///
    /// Input rows are read from larger row-major buffers beginning at
    /// `input_row_offset`. The cache is extended from its current logical
    /// length and compact tiles are finalized as their boundaries are crossed.
    pub fn append_rows_at_offset_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_heads * self.head_dim;
        let input_end = input_row_offset
            .checked_add(rows)
            .and_then(|end| end.checked_mul(width))
            .ok_or_else(|| Error::Shape {
                label: "SM12x KV row append",
                expected: "input row range without overflow".to_string(),
                actual: format!("input_row_offset={input_row_offset} rows={rows} width={width}"),
            })?;
        let cache_end = self.len.checked_add(rows).ok_or_else(|| Error::Shape {
            label: "SM12x KV row append",
            expected: "cache row range without overflow".to_string(),
            actual: format!("len={} rows={rows}", self.len),
        })?;
        if rows == 0
            || rows > u32::MAX as usize
            || input_row_offset > u32::MAX as usize
            || input_end > key.len()
            || input_end > value.len()
            || cache_end > self.max_tokens
        {
            return Err(Error::Shape {
                label: "SM12x KV row append",
                expected: format!(
                    "non-empty rows through cache capacity {} and input buffers >= {input_end}",
                    self.max_tokens
                ),
                actual: format!(
                    "rows={rows} cache_end={cache_end} key={} value={}",
                    key.len(),
                    value.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_rows_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_rows_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    self.key_values_mut_ptr(),
                    self.key_scales_mut_ptr(),
                    self.value_values_mut_ptr(),
                    self.value_scales_mut_ptr(),
                    self.key_tail_mut_ptr(),
                    self.value_tail_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    input_row_offset as u32,
                    self.len as u32,
                    rows as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        self.len = cache_end;
        Ok(())
    }

    /// Appends the first prompt rows while staging the exact BF16 cache values for attention.
    #[allow(clippy::too_many_arguments)]
    pub fn append_initial_rows_and_stage_bf16_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        mut key_output: DeviceOutput<'_, u16>,
        mut value_output: DeviceOutput<'_, u16>,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_heads * self.head_dim;
        let input_end = input_row_offset
            .checked_add(rows)
            .and_then(|end| end.checked_mul(width))
            .ok_or_else(|| Error::Shape {
                label: "SM12x initial KV row staging",
                expected: "input row range without overflow".to_string(),
                actual: format!("input_row_offset={input_row_offset} rows={rows} width={width}"),
            })?;
        let output_values = rows.checked_mul(width).ok_or_else(|| Error::Shape {
            label: "SM12x initial KV row staging",
            expected: "rows * width without overflow".to_string(),
            actual: format!("rows={rows} width={width}"),
        })?;
        if self.len != 0
            || rows == 0
            || rows > self.max_tokens
            || rows > u32::MAX as usize
            || input_row_offset > u32::MAX as usize
            || input_end > key.len()
            || input_end > value.len()
            || key_output.len() < output_values
            || value_output.len() < output_values
        {
            return Err(Error::Shape {
                label: "SM12x initial KV row staging",
                expected: format!(
                    "empty cache, non-empty rows through capacity {}, inputs >= {input_end}, outputs >= {output_values}",
                    self.max_tokens
                ),
                actual: format!(
                    "cache_len={} rows={rows} key={} value={} key_output={} value_output={}",
                    self.len,
                    key.len(),
                    value.len(),
                    key_output.len(),
                    value_output.len(),
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_rows_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_rows_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    self.key_values_mut_ptr(),
                    self.key_scales_mut_ptr(),
                    self.value_values_mut_ptr(),
                    self.value_scales_mut_ptr(),
                    self.key_tail_mut_ptr(),
                    self.value_tail_mut_ptr(),
                    key_output.as_mut_ptr().cast(),
                    value_output.as_mut_ptr().cast(),
                    rows as u32,
                    input_row_offset as u32,
                    0,
                    rows as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        self.len = rows;
        Ok(())
    }

    /// Dequantizes the logical cache into BF16 matrices for prompt attention.
    ///
    /// Keys are written as `[kv_heads, tokens, head_dim]`; values are written
    /// as `[kv_heads, head_dim, tokens]` for direct tensor-core QK and PV use.
    pub fn unpack_bf16_on_stream(
        &self,
        mut key_output: DeviceOutput<'_, u16>,
        mut value_output: DeviceOutput<'_, u16>,
        stream: &CudaStream,
    ) -> Result<()> {
        let values = checked_product(
            "SM12x KV BF16 unpack",
            &[self.len, self.kv_heads, self.head_dim],
        )?;
        if self.len == 0 || key_output.len() < values || value_output.len() < values {
            return Err(Error::Shape {
                label: "SM12x KV BF16 unpack",
                expected: format!("non-empty cache and K/V output >= {values} values"),
                actual: format!(
                    "cache_len={} key={} value={}",
                    self.len,
                    key_output.len(),
                    value_output.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_unpack_bf16_on_stream",
                crate::ffi::infer_sm12x_kv_cache_unpack_bf16_on_stream(
                    self.key_values_ptr(),
                    self.key_scales_ptr(),
                    self.value_values_ptr(),
                    self.value_scales_ptr(),
                    self.key_tail_ptr(),
                    self.value_tail_ptr(),
                    key_output.as_mut_ptr().cast(),
                    value_output.as_mut_ptr().cast(),
                    self.len as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues one K/V append using a device-resident position for CUDA graphs.
    pub fn append_indexed_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        position: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_heads * self.head_dim;
        if key.len() != width || value.len() != width || position.len() != 1 {
            return Err(Error::Shape {
                label: "SM12x indexed KV append",
                expected: format!("K/V rows of {width} values and one position"),
                actual: format!(
                    "key={} value={} position={}",
                    key.len(),
                    value.len(),
                    position.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_indexed_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_indexed_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    self.key_values_mut_ptr(),
                    self.key_scales_mut_ptr(),
                    self.value_values_mut_ptr(),
                    self.value_scales_mut_ptr(),
                    self.key_tail_mut_ptr(),
                    self.value_tail_mut_ptr(),
                    position.as_const_ptr().cast(),
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Copies a complete 128-token-aligned prefix from another compact cache.
    ///
    /// The source and destination may have different token capacities. Only
    /// finalized compact tiles are copied; a 128-token boundary has no live K
    /// or V tail rows.
    pub fn copy_aligned_prefix_from_on_stream(
        &mut self,
        source: &Self,
        prefix_tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.len != 0
            || source.kv_heads != self.kv_heads
            || source.head_dim != self.head_dim
            || prefix_tokens == 0
            || !prefix_tokens.is_multiple_of(128)
            || prefix_tokens > source.len
            || prefix_tokens > self.max_tokens
        {
            return Err(Error::Shape {
                label: "SM12x KV aligned prefix copy",
                expected: format!(
                    "empty destination, matching {}/{} shape, and a nonzero 128-token prefix <= source len {} and destination capacity {}",
                    self.kv_heads, self.head_dim, source.len, self.max_tokens
                ),
                actual: format!(
                    "destination_len={} source_shape={}/{} prefix_tokens={prefix_tokens}",
                    self.len, source.kv_heads, source.head_dim
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_copy_aligned_prefix_on_stream",
                crate::ffi::infer_sm12x_kv_cache_copy_aligned_prefix_on_stream(
                    source.key_values_ptr(),
                    source.key_scales_ptr(),
                    source.value_values_ptr(),
                    source.value_scales_ptr(),
                    self.key_values_mut_ptr(),
                    self.key_scales_mut_ptr(),
                    self.value_values_mut_ptr(),
                    self.value_scales_mut_ptr(),
                    prefix_tokens as u32,
                    source.max_tokens as u32,
                    self.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        self.len = prefix_tokens;
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
        Self::new_gqa(max_tokens, kv_heads * MMA_N, kv_heads, head_dim)
    }

    /// Allocates scratch for an explicit grouped-query attention shape.
    pub fn new_gqa(
        max_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_gqa_batched(max_tokens, q_heads, kv_heads, head_dim, 1)
    }

    /// Allocates scratch for causal prompt attention over several rows per launch.
    pub fn new_gqa_batched(
        max_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        causal_row_capacity: usize,
    ) -> Result<Self> {
        if max_tokens == 0
            || q_heads == 0
            || kv_heads == 0
            || !q_heads.is_multiple_of(kv_heads)
            || head_dim == 0
            || !head_dim.is_multiple_of(MMA_K)
            || causal_row_capacity == 0
            || causal_row_capacity > V_TOKEN_BLOCK
        {
            return Err(Error::Shape {
                label: "SM12x KV attention workspace",
                expected: "non-zero sizes and head_dim multiple of 64".to_string(),
                actual: format!(
                    "max_tokens={max_tokens} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim} causal_rows={causal_row_capacity}"
                ),
            });
        }
        let head_k_tiles = head_dim / MMA_K;
        let context_tiles = max_tokens.div_ceil(MMA_K);
        let query_groups = kv_heads * (q_heads / kv_heads).div_ceil(MMA_N);
        let query_tile_count = checked_product(
            "SM12x KV query tiles",
            &[causal_row_capacity, query_groups, head_k_tiles],
        )?;
        let probability_tile_count = checked_product(
            "SM12x KV probability tiles",
            &[causal_row_capacity, query_groups, context_tiles],
        )?;
        Ok(Self {
            query_tiles: DeviceBuffer::zeroed(checked_product(
                "SM12x KV query tile bytes",
                &[query_tile_count, 512],
            )?)?,
            query_scales: DeviceBuffer::zeroed(checked_product(
                "SM12x KV query scale words",
                &[query_tile_count, MMA_N],
            )?)?,
            scores: DeviceBuffer::zeroed(checked_product(
                "SM12x KV scores",
                &[causal_row_capacity, q_heads, max_tokens],
            )?)?,
            probability_tiles: DeviceBuffer::zeroed(checked_product(
                "SM12x KV probability tile bytes",
                &[probability_tile_count, 512],
            )?)?,
            probability_scales: DeviceBuffer::zeroed(checked_product(
                "SM12x KV probability scale words",
                &[probability_tile_count, MMA_N],
            )?)?,
            pv_partials: DeviceBuffer::zeroed(checked_product(
                "SM12x KV PV partials",
                &[PV_SPLIT_CAPACITY, q_heads, head_dim],
            )?)?,
            max_tokens,
            q_heads,
            kv_heads,
            head_dim,
            causal_row_capacity,
        })
    }

    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.query_tiles.device_bytes()
            + self.query_scales.device_bytes()
            + self.scores.device_bytes()
            + self.probability_tiles.device_bytes()
            + self.probability_scales.device_bytes()
            + self.pv_partials.device_bytes()
    }

    /// Enqueues Q-to-FP4 and compact-cache QK into the internal score workspace.
    pub fn qk_scores_into_workspace_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len == 0
            || cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x KV QK cache",
                expected: format!(
                    "non-empty cache with max_tokens={} kv_heads={} head_dim={}",
                    self.max_tokens, self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "len={} max_tokens={} kv_heads={} head_dim={}",
                    cache.len, cache.max_tokens, cache.kv_heads, cache.head_dim
                ),
            });
        }
        let query_values = self.q_heads * self.head_dim;
        if query.len() != query_values {
            return Err(Error::Shape {
                label: "SM12x KV QK query",
                expected: format!("{query_values} values"),
                actual: format!("{} values", query.len()),
            });
        }

        unsafe {
            check_cuda(
                "infer_sm12x_kv_qk_on_stream",
                crate::ffi::infer_sm12x_kv_qk_on_stream(
                    query.as_const_ptr().cast(),
                    cache.key_values_ptr(),
                    cache.key_scales_ptr(),
                    cache.key_tail_ptr(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    cache.len as u32,
                    cache.max_tokens as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
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
        if cache.max_tokens > self.max_tokens
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
        let output_values = self.q_heads * self.head_dim;
        if query.len() != output_values || output.len() != output_values {
            return Err(Error::Shape {
                label: "SM12x KV attention query/output",
                expected: format!("{output_values} values"),
                actual: format!("query={} output={}", query.len(), output.len()),
            });
        }

        let pv_splits = pv_split_count(cache.len);
        unsafe {
            check_cuda(
                "infer_sm12x_kv_attention_on_stream",
                crate::ffi::infer_sm12x_kv_attention_on_stream(
                    query.as_const_ptr().cast(),
                    cache.key_values_ptr(),
                    cache.key_scales_ptr(),
                    cache.key_tail_ptr(),
                    cache.value_values_ptr(),
                    cache.value_scales_ptr(),
                    cache.value_tail_ptr(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cache.len as u32,
                    cache.max_tokens as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    pv_splits as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues compact attention over `window_start..cache.len()`.
    pub fn attention_window_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        window_start: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len == 0 || window_start >= cache.len {
            return Err(Error::Shape {
                label: "SM12x KV attention window",
                expected: format!("window_start < nonzero cache length {}", cache.len),
                actual: window_start.to_string(),
            });
        }
        if cache.max_tokens > self.max_tokens
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
        let output_values = self.q_heads * self.head_dim;
        if query.len() != output_values || output.len() != output_values {
            return Err(Error::Shape {
                label: "SM12x KV attention query/output",
                expected: format!("{output_values} values"),
                actual: format!("query={} output={}", query.len(), output.len()),
            });
        }

        let pv_splits = pv_split_count(cache.len - window_start);
        unsafe {
            check_cuda(
                "infer_sm12x_kv_attention_window_on_stream",
                crate::ffi::infer_sm12x_kv_attention_window_on_stream(
                    query.as_const_ptr().cast(),
                    cache.key_values_ptr(),
                    cache.key_scales_ptr(),
                    cache.key_tail_ptr(),
                    cache.value_values_ptr(),
                    cache.value_scales_ptr(),
                    cache.value_tail_ptr(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cache.len as u32,
                    window_start as u32,
                    cache.max_tokens as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    pv_splits as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Appends a prompt chunk and computes causal compact-cache attention in
    /// bounded row batches that do not cross the V-tail wrap boundary.
    ///
    /// Query, key, value, and output buffers are row-major. `input_row_offset`
    /// selects a contiguous chunk in larger flattened prompt buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn append_causal_rows_at_offset_into_on_stream(
        &mut self,
        cache: &mut Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        window_tokens: Option<usize>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x causal row attention cache",
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
        let q_width = self.q_heads * self.head_dim;
        let kv_width = self.kv_heads * self.head_dim;
        let row_end = input_row_offset
            .checked_add(rows)
            .ok_or_else(|| Error::Shape {
                label: "SM12x causal row attention",
                expected: "input row range without overflow".to_string(),
                actual: format!("input_row_offset={input_row_offset} rows={rows}"),
            })?;
        let q_end = row_end.checked_mul(q_width).ok_or_else(|| Error::Shape {
            label: "SM12x causal row attention",
            expected: "query row range without overflow".to_string(),
            actual: format!("row_end={row_end} q_width={q_width}"),
        })?;
        let kv_end = row_end.checked_mul(kv_width).ok_or_else(|| Error::Shape {
            label: "SM12x causal row attention",
            expected: "KV row range without overflow".to_string(),
            actual: format!("row_end={row_end} kv_width={kv_width}"),
        })?;
        let cache_end = cache.len.checked_add(rows).ok_or_else(|| Error::Shape {
            label: "SM12x causal row attention",
            expected: "cache row range without overflow".to_string(),
            actual: format!("len={} rows={rows}", cache.len),
        })?;
        if rows == 0
            || rows > u32::MAX as usize
            || input_row_offset > u32::MAX as usize
            || q_end > query.len()
            || q_end > output.len()
            || kv_end > key.len()
            || kv_end > value.len()
            || cache_end > cache.max_tokens
            || window_tokens.is_some_and(|window| window == 0 || window > u32::MAX as usize)
        {
            return Err(Error::Shape {
                label: "SM12x causal row attention buffers",
                expected: format!(
                    "rows through cache capacity {}, q/output >= {q_end}, k/v >= {kv_end}",
                    cache.max_tokens
                ),
                actual: format!(
                    "rows={rows} cache_end={cache_end} query={} key={} value={} output={}",
                    query.len(),
                    key.len(),
                    value.len(),
                    output.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_append_causal_attention_rows_on_stream",
                crate::ffi::infer_sm12x_kv_append_causal_attention_rows_on_stream(
                    query.as_const_ptr().cast(),
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    cache.key_values_mut_ptr(),
                    cache.key_scales_mut_ptr(),
                    cache.value_values_mut_ptr(),
                    cache.value_scales_mut_ptr(),
                    cache.key_tail_mut_ptr(),
                    cache.value_tail_mut_ptr(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    input_row_offset as u32,
                    cache.len as u32,
                    rows as u32,
                    cache.max_tokens as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    window_tokens.unwrap_or(0) as u32,
                    self.causal_row_capacity as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        cache.len = cache_end;
        Ok(())
    }

    /// Enqueues attention from and into row offsets in larger dense buffers.
    pub fn attention_offsets_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        query_offset: usize,
        mut output: DeviceOutput<'_, f32>,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len == 0
            || cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x offset KV attention cache",
                expected: format!(
                    "non-empty max_tokens={} kv_heads={} head_dim={}",
                    self.max_tokens, self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "len={} max_tokens={} kv_heads={} head_dim={}",
                    cache.len, cache.max_tokens, cache.kv_heads, cache.head_dim
                ),
            });
        }
        let width = self.q_heads * self.head_dim;
        if query_offset
            .checked_add(width)
            .is_none_or(|end| end > query.len())
            || output_offset
                .checked_add(width)
                .is_none_or(|end| end > output.len())
        {
            return Err(Error::Shape {
                label: "SM12x KV attention offsets",
                expected: format!("{width} readable/writable values at each row offset"),
                actual: format!(
                    "query_len={} query_offset={query_offset} output_len={} output_offset={output_offset}",
                    query.len(),
                    output.len()
                ),
            });
        }
        let pv_splits = pv_split_count(cache.len);
        unsafe {
            check_cuda(
                "infer_sm12x_kv_attention_on_stream",
                crate::ffi::infer_sm12x_kv_attention_on_stream(
                    query.as_const_ptr().cast::<f32>().add(query_offset),
                    cache.key_values_ptr(),
                    cache.key_scales_ptr(),
                    cache.key_tail_ptr(),
                    cache.value_values_ptr(),
                    cache.value_scales_ptr(),
                    cache.value_tail_ptr(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast::<f32>().add(output_offset),
                    cache.len as u32,
                    cache.max_tokens as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    pv_splits as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues compact attention using a device-resident cache length for CUDA graphs.
    pub fn attention_indexed_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        cache_len: &DeviceBuffer<u32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x indexed KV attention cache",
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
        if query.len() != output_values || output.len() != output_values || cache_len.len() != 1 {
            return Err(Error::Shape {
                label: "SM12x indexed KV attention",
                expected: format!("query/output of {output_values} values and one cache length"),
                actual: format!(
                    "query={} output={} cache_len={}",
                    query.len(),
                    output.len(),
                    cache_len.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_sm12x_kv_attention_indexed_on_stream",
                crate::ffi::infer_sm12x_kv_attention_indexed_on_stream(
                    query.as_const_ptr().cast(),
                    cache.key_values_ptr(),
                    cache.key_scales_ptr(),
                    cache.key_tail_ptr(),
                    cache.value_values_ptr(),
                    cache.value_scales_ptr(),
                    cache.value_tail_ptr(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cache_len.as_const_ptr().cast(),
                    cache.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    PV_SPLIT_CAPACITY as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues P-to-FP4 and PV from caller-provided f32 probabilities.
    pub fn pv_from_probabilities_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        probabilities: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len == 0
            || cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x KV PV cache",
                expected: format!(
                    "non-empty cache with max_tokens={} kv_heads={} head_dim={}",
                    self.max_tokens, self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "len={} max_tokens={} kv_heads={} head_dim={}",
                    cache.len, cache.max_tokens, cache.kv_heads, cache.head_dim
                ),
            });
        }
        let probability_values = self.kv_heads * MMA_N * self.max_tokens;
        let output_values = self.kv_heads * MMA_N * self.head_dim;
        if probabilities.len() != probability_values || output.len() != output_values {
            return Err(Error::Shape {
                label: "SM12x KV PV probabilities/output",
                expected: format!("{probability_values} probabilities and {output_values} outputs"),
                actual: format!(
                    "probabilities={} output={}",
                    probabilities.len(),
                    output.len()
                ),
            });
        }

        unsafe {
            check_cuda(
                "infer_sm12x_kv_pv_from_probabilities_on_stream",
                crate::ffi::infer_sm12x_kv_pv_from_probabilities_on_stream(
                    probabilities.as_const_ptr().cast(),
                    cache.value_values_ptr(),
                    cache.value_scales_ptr(),
                    cache.value_tail_ptr(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cache.len as u32,
                    cache.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues P-to-FP4 and context-split PV from caller-provided probabilities.
    pub fn pv_from_probabilities_split_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        probabilities: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        pv_splits: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if cache.len == 0
            || cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x split KV PV cache",
                expected: format!(
                    "non-empty cache with max_tokens={} kv_heads={} head_dim={}",
                    self.max_tokens, self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "len={} max_tokens={} kv_heads={} head_dim={}",
                    cache.len, cache.max_tokens, cache.kv_heads, cache.head_dim
                ),
            });
        }
        let probability_values = self.kv_heads * MMA_N * self.max_tokens;
        let output_values = self.kv_heads * MMA_N * self.head_dim;
        if probabilities.len() != probability_values
            || output.len() != output_values
            || !(2..=PV_SPLIT_CAPACITY).contains(&pv_splits)
        {
            return Err(Error::Shape {
                label: "SM12x split KV PV probabilities/output",
                expected: format!(
                    "{probability_values} probabilities, {output_values} outputs, and 2..={PV_SPLIT_CAPACITY} splits"
                ),
                actual: format!(
                    "probabilities={} output={} pv_splits={pv_splits}",
                    probabilities.len(),
                    output.len()
                ),
            });
        }

        unsafe {
            check_cuda(
                "infer_sm12x_kv_pv_from_probabilities_split_on_stream",
                crate::ffi::infer_sm12x_kv_pv_from_probabilities_split_on_stream(
                    probabilities.as_const_ptr().cast(),
                    cache.value_values_ptr(),
                    cache.value_scales_ptr(),
                    cache.value_tail_ptr(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cache.len as u32,
                    cache.max_tokens as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    pv_splits as u32,
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

fn checked_sum(label: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| Error::Shape {
        label,
        expected: "offset without overflow".to_string(),
        actual: format!("left={left} right={right}"),
    })
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let mask = alignment - 1;
    checked_sum("SM12x KV cache alignment", value, mask).map(|value| value & !mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::bf16_to_f32;

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
    fn pv_split_policy_uses_measured_crossover() {
        assert_eq!(pv_split_count(1), 1);
        assert_eq!(pv_split_count(64), 1);
        assert_eq!(pv_split_count(1_023), 1);
        assert_eq!(pv_split_count(1_024), PV_SPLIT_CAPACITY);
        assert_eq!(pv_split_count(4_096), PV_SPLIT_CAPACITY);
        assert_eq!(pv_split_count(32_768), 32);
        assert_eq!(pv_split_count(131_072), PV_SPLIT_CAPACITY);
    }

    #[test]
    fn aligned_prefix_copy_preserves_attention_across_capacities() {
        const TOKENS: usize = 128;
        const SOURCE_CAPACITY: usize = 192;
        const DESTINATION_CAPACITY: usize = 320;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = 8;
        let width = KV_HEADS * HEAD_DIM;
        let key = (0..TOKENS * width)
            .map(|index| ((index * 17 + 11) % 251) as f32 / 64.0 - 1.5)
            .collect::<Vec<_>>();
        let value = (0..TOKENS * width)
            .map(|index| ((index * 29 + 7) % 257) as f32 / 80.0 - 1.25)
            .collect::<Vec<_>>();
        let query = (0..Q_HEADS * HEAD_DIM)
            .map(|index| ((index * 13 + 5) % 127) as f32 / 48.0 - 1.0)
            .collect::<Vec<_>>();
        let key = DeviceBuffer::from_host(&key).expect("key");
        let value = DeviceBuffer::from_host(&value).expect("value");
        let query = DeviceBuffer::from_host(&query).expect("query");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut source = Sm12xKvCache::new(SOURCE_CAPACITY, KV_HEADS, HEAD_DIM).expect("source");
        source
            .append_rows_at_offset_on_stream(&key, &value, 0, TOKENS, &stream)
            .expect("append source");
        let mut destination =
            Sm12xKvCache::new(DESTINATION_CAPACITY, KV_HEADS, HEAD_DIM).expect("destination");
        assert_eq!(
            source
                .device_bytes_for_capacity(DESTINATION_CAPACITY)
                .expect("destination byte estimate"),
            destination.device_bytes()
        );
        destination
            .copy_aligned_prefix_from_on_stream(&source, TOKENS, &stream)
            .expect("copy prefix");

        let mut workspace =
            Sm12xKvAttentionWorkspace::new_gqa(DESTINATION_CAPACITY, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("attention workspace");
        let mut source_output = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("source output");
        workspace
            .attention_into_on_stream(&source, &query, source_output.output(), &stream)
            .expect("source attention");
        let mut destination_output =
            DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("destination output");
        workspace
            .attention_into_on_stream(&destination, &query, destination_output.output(), &stream)
            .expect("destination attention");

        assert_eq!(destination.len(), TOKENS);
        assert_eq!(
            source_output
                .copy_to_host(&stream)
                .expect("source output read"),
            destination_output
                .copy_to_host(&stream)
                .expect("destination output read")
        );
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

        let key_dense =
            DeviceBuffer::from_host(&key_rows.iter().flatten().copied().collect::<Vec<_>>())
                .expect("dense keys");
        let value_dense =
            DeviceBuffer::from_host(&value_rows.iter().flatten().copied().collect::<Vec<_>>())
                .expect("dense values");
        let mut rows_cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("row cache");
        let rows_stream = CudaStream::new_non_blocking().expect("row append stream");
        rows_cache
            .append_rows_at_offset_on_stream(&key_dense, &value_dense, 0, TOKENS, &rows_stream)
            .expect("row append");
        rows_stream.synchronize().expect("row append sync");

        assert_eq!(cache.len(), TOKENS);
        assert_eq!(cache.compact_key_tokens(), 16);
        assert_eq!(cache.compact_value_tokens(), 16);
        assert_eq!(cache.key_tail_len(), 1);
        assert_eq!(cache.value_tail_len(), 1);

        let stream = CudaStream::new_blocking().expect("stream");
        let key_values = cache.key_values_to_host(&stream).expect("K values");
        let key_scales = cache.key_scales_to_host(&stream).expect("K scales");
        let value_values = cache.value_values_to_host(&stream).expect("V values");
        let value_scales = cache.value_scales_to_host(&stream).expect("V scales");
        assert_eq!(rows_cache.len(), cache.len());
        assert_eq!(
            rows_cache
                .key_values_to_host(&stream)
                .expect("row K values"),
            key_values
        );
        assert_eq!(
            rows_cache
                .key_scales_to_host(&stream)
                .expect("row K scales"),
            key_scales
        );
        assert_eq!(
            rows_cache
                .value_values_to_host(&stream)
                .expect("row V values"),
            value_values
        );
        assert_eq!(
            rows_cache
                .value_scales_to_host(&stream)
                .expect("row V scales"),
            value_scales
        );

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
        assert_eq!(key_values, expected_key_values);
        assert_eq!(key_scales, expected_key_scales);

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
                    for (token, row) in value_rows.iter().enumerate().take(16) {
                        let value = row[head * HEAD_DIM + dim_tile * 8 + dim];
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
        assert_eq!(value_values, expected_value_values);
        assert_eq!(value_scales, expected_value_scales);

        let key_tail = cache.key_tail_to_host(&stream).expect("K tail");
        let value_tail = cache.value_tail_to_host(&stream).expect("V tail");
        assert_eq!(
            rows_cache.key_tail_to_host(&stream).expect("row K tail"),
            key_tail
        );
        assert_eq!(
            rows_cache.value_tail_to_host(&stream).expect("row V tail"),
            value_tail
        );
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
    fn compact_causal_rows_match_repeated_append_and_attention() {
        const MAX_TOKENS: usize = 32;
        const TOKENS: usize = 23;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = KV_HEADS * MMA_N;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let key = DeviceBuffer::from_host(
            &(0..TOKENS * kv_width)
                .map(|idx| ((idx * 17 % 251) as f32 - 125.0) / 384.0)
                .collect::<Vec<_>>(),
        )
        .expect("key");
        let value = DeviceBuffer::from_host(
            &(0..TOKENS * kv_width)
                .map(|idx| ((idx * 29 % 257) as f32 - 128.0) / 448.0)
                .collect::<Vec<_>>(),
        )
        .expect("value");
        let query = DeviceBuffer::from_host(
            &(0..TOKENS * q_width)
                .map(|idx| ((idx * 31 % 263) as f32 - 131.0) / 512.0)
                .collect::<Vec<_>>(),
        )
        .expect("query");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut chunk_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("chunk cache");
        let mut chunk_workspace =
            Sm12xKvAttentionWorkspace::new_gqa_batched(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, 8)
                .expect("chunk workspace");
        let mut chunk_output = DeviceBuffer::<f32>::zeroed(TOKENS * q_width).expect("chunk output");
        chunk_workspace
            .append_causal_rows_at_offset_into_on_stream(
                &mut chunk_cache,
                &query,
                &key,
                &value,
                0,
                TOKENS,
                None,
                chunk_output.output(),
                &stream,
            )
            .expect("chunk attention");

        let mut repeated_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("repeated cache");
        let mut repeated_workspace = Sm12xKvAttentionWorkspace::new(MAX_TOKENS, KV_HEADS, HEAD_DIM)
            .expect("repeated workspace");
        let mut repeated_output =
            DeviceBuffer::<f32>::zeroed(TOKENS * q_width).expect("repeated output");
        for token in 0..TOKENS {
            repeated_cache
                .append_at_offsets_on_stream(
                    &key,
                    token * kv_width,
                    &value,
                    token * kv_width,
                    token,
                    &stream,
                )
                .expect("repeated append");
            repeated_workspace
                .attention_offsets_into_on_stream(
                    &repeated_cache,
                    &query,
                    token * q_width,
                    repeated_output.output(),
                    token * q_width,
                    &stream,
                )
                .expect("repeated attention");
        }

        let chunk = chunk_output.copy_to_host(&stream).expect("chunk download");
        let repeated = repeated_output
            .copy_to_host(&stream)
            .expect("repeated download");
        let max_abs = chunk
            .iter()
            .zip(repeated.iter())
            .map(|(chunk, repeated)| (chunk - repeated).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0e-6, "causal row max_abs={max_abs}");
        assert_eq!(chunk_cache.len(), repeated_cache.len());
        assert_eq!(
            chunk_cache.key_tail_to_host(&stream).expect("chunk K tail"),
            repeated_cache
                .key_tail_to_host(&stream)
                .expect("repeated K tail")
        );
        assert_eq!(
            chunk_cache
                .value_tail_to_host(&stream)
                .expect("chunk V tail"),
            repeated_cache
                .value_tail_to_host(&stream)
                .expect("repeated V tail")
        );
    }

    #[test]
    fn compact_causal_rows_support_step_sliding_gqa() {
        const MAX_TOKENS: usize = 32;
        const TOKENS: usize = 7;
        const WINDOW: usize = 4;
        const KV_HEADS: usize = 8;
        const Q_HEADS: usize = 96;
        const HEAD_DIM: usize = 128;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let key_host = (0..TOKENS * kv_width)
            .map(|idx| ((idx * 17 % 251) as f32 - 125.0) / 384.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * kv_width)
            .map(|idx| ((idx * 29 % 257) as f32 - 128.0) / 448.0)
            .collect::<Vec<_>>();
        let query_host = (0..TOKENS * q_width)
            .map(|idx| ((idx * 31 % 263) as f32 - 131.0) / 512.0)
            .collect::<Vec<_>>();
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let value = DeviceBuffer::from_host(&value_host).expect("value");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut chunk_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("chunk cache");
        let mut chunk_workspace =
            Sm12xKvAttentionWorkspace::new_gqa_batched(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, 8)
                .expect("chunk workspace");
        let mut chunk_output = DeviceBuffer::<f32>::zeroed(TOKENS * q_width).expect("chunk output");
        chunk_workspace
            .append_causal_rows_at_offset_into_on_stream(
                &mut chunk_cache,
                &query,
                &key,
                &value,
                0,
                TOKENS,
                Some(WINDOW),
                chunk_output.output(),
                &stream,
            )
            .expect("chunk attention");

        let mut repeated_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("repeated cache");
        let mut repeated_workspace =
            Sm12xKvAttentionWorkspace::new_gqa(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("repeated workspace");
        let mut repeated = Vec::with_capacity(TOKENS * q_width);
        for token in 0..TOKENS {
            repeated_cache
                .append_at_offsets_on_stream(
                    &key,
                    token * kv_width,
                    &value,
                    token * kv_width,
                    token,
                    &stream,
                )
                .expect("repeated append");
            let query_row =
                DeviceBuffer::from_host(&query_host[token * q_width..(token + 1) * q_width])
                    .expect("query row");
            let mut output = DeviceBuffer::zeroed(q_width).expect("output row");
            let window_start = repeated_cache.len().saturating_sub(WINDOW);
            repeated_workspace
                .attention_window_into_on_stream(
                    &repeated_cache,
                    &query_row,
                    output.output(),
                    window_start,
                    &stream,
                )
                .expect("repeated attention");
            repeated.extend_from_slice(&output.copy_to_host(&stream).expect("output download"));
        }

        let chunk = chunk_output.copy_to_host(&stream).expect("chunk download");
        let max_abs = chunk
            .iter()
            .zip(&repeated)
            .map(|(chunk, repeated)| (chunk - repeated).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0e-6, "sliding causal row max_abs={max_abs}");
        assert_eq!(chunk_cache.len(), repeated_cache.len());
    }

    #[test]
    fn compact_mma_attention_tracks_f32_gqa_through_incomplete_tails() {
        const MAX_TOKENS: usize = 64;
        const TOKENS: usize = 17;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = KV_HEADS * 12;
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
            Sm12xKvAttentionWorkspace::new_gqa(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("workspace");
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
    fn compact_mma_attention_window_ignores_older_tokens() {
        const MAX_TOKENS: usize = 64;
        const TOKENS: usize = 17;
        const WINDOW_START: usize = 5;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = KV_HEADS * 12;
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
        let key_f32 = DeviceBuffer::from_host(&key_host[WINDOW_START * width..]).expect("K window");
        let value_f32 =
            DeviceBuffer::from_host(&value_host[WINDOW_START * width..]).expect("V window");
        let mut expected = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("f32 output");
        crate::cached_gqa_attention_f32_into_on_stream(
            &query,
            &key_f32,
            &value_f32,
            expected.output(),
            TOKENS - WINDOW_START,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("f32 window attention");
        let mut actual = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("FP4 output");
        let mut workspace =
            Sm12xKvAttentionWorkspace::new_gqa(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("workspace");
        workspace
            .attention_window_into_on_stream(&cache, &query, actual.output(), WINDOW_START, &stream)
            .expect("compact window attention");
        let expected = expected.copy_to_host(&stream).expect("f32 copy");
        let actual = actual.copy_to_host(&stream).expect("FP4 copy");
        let max_abs = expected
            .iter()
            .zip(actual.iter())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.25,
            "compact FP4 window attention error too large: max_abs={max_abs}"
        );
    }

    #[test]
    fn compact_mma_attention_tracks_f32_gqa_at_nemotron_shape() {
        const MAX_TOKENS: usize = 128;
        const TOKENS: usize = 65;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 128;
        const Q_HEADS: usize = 32;
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
            Sm12xKvAttentionWorkspace::new_gqa(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("workspace");
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
            "Nemotron-shape compact FP4 attention error too large: max_abs={max_abs}"
        );
    }

    #[test]
    fn compact_mma_attention_tracks_f32_gqa_at_bitnet_shape_and_context() {
        const TOKENS: usize = 4_016;
        const KV_HEADS: usize = 5;
        const HEAD_DIM: usize = 128;
        const Q_HEADS: usize = 20;
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

        let stream = CudaStream::new_non_blocking().expect("stream");
        let key = DeviceBuffer::from_host(&key_host).expect("K cache");
        let value = DeviceBuffer::from_host(&value_host).expect("V cache");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let mut cache =
            Sm12xKvCache::new(TOKENS, KV_HEADS, HEAD_DIM).expect("BitNet compact cache");
        cache
            .append_rows_at_offset_on_stream(&key, &value, 0, TOKENS, &stream)
            .expect("append BitNet cache");

        let mut expected = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("f32 output");
        crate::cached_gqa_attention_f32_into_on_stream(
            &query,
            &key,
            &value,
            expected.output(),
            TOKENS,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("f32 attention");
        let mut actual = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("FP4 output");
        let mut workspace = Sm12xKvAttentionWorkspace::new_gqa(TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
            .expect("workspace");
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
            "BitNet-shape compact FP4 attention error too large: max_abs={max_abs}"
        );
    }

    #[test]
    fn dense_row_offsets_match_independent_cache_and_attention() {
        const MAX_TOKENS: usize = 4;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = KV_HEADS * MMA_N;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let keys = (0..2 * kv_width)
            .map(|index| ((index * 29 % 257) as f32 - 128.0) / 96.0)
            .collect::<Vec<_>>();
        let values = (0..2 * kv_width)
            .map(|index| ((index * 43 % 251) as f32 - 125.0) / 80.0)
            .collect::<Vec<_>>();
        let queries = (0..2 * q_width)
            .map(|index| ((index * 17 % 239) as f32 - 119.0) / 112.0)
            .collect::<Vec<_>>();
        let key_rows = DeviceBuffer::from_host(&keys).expect("dense keys");
        let value_rows = DeviceBuffer::from_host(&values).expect("dense values");
        let query_rows = DeviceBuffer::from_host(&queries).expect("dense queries");
        let key = DeviceBuffer::from_host(&keys[kv_width..]).expect("key row");
        let value = DeviceBuffer::from_host(&values[kv_width..]).expect("value row");
        let query = DeviceBuffer::from_host(&queries[q_width..]).expect("query row");
        let stream = CudaStream::new_blocking().expect("stream");

        let mut direct_cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        direct_cache
            .append_on_stream(&key, &value, &stream)
            .expect("direct append");
        let mut offset_cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        offset_cache
            .append_at_offsets_on_stream(&key_rows, kv_width, &value_rows, kv_width, 0, &stream)
            .expect("offset append");

        assert_eq!(
            direct_cache
                .key_tail_to_host(&stream)
                .expect("direct key tail"),
            offset_cache
                .key_tail_to_host(&stream)
                .expect("offset key tail")
        );
        assert_eq!(
            direct_cache
                .value_tail_to_host(&stream)
                .expect("direct value tail"),
            offset_cache
                .value_tail_to_host(&stream)
                .expect("offset value tail")
        );

        let mut direct_workspace =
            Sm12xKvAttentionWorkspace::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("workspace");
        let mut direct_output = DeviceBuffer::zeroed(q_width).expect("direct output");
        direct_workspace
            .attention_into_on_stream(&direct_cache, &query, direct_output.output(), &stream)
            .expect("direct attention");
        let mut offset_workspace =
            Sm12xKvAttentionWorkspace::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("workspace");
        let mut offset_output = DeviceBuffer::zeroed(2 * q_width).expect("offset output");
        offset_workspace
            .attention_offsets_into_on_stream(
                &offset_cache,
                &query_rows,
                q_width,
                offset_output.output(),
                q_width,
                &stream,
            )
            .expect("offset attention");
        let direct_output = direct_output
            .copy_to_host(&stream)
            .expect("direct output copy");
        let offset_output = offset_output
            .copy_to_host(&stream)
            .expect("offset output copy");
        assert_eq!(&*direct_output, &offset_output[q_width..]);
    }

    #[test]
    fn compact_cache_unpacks_to_bf16_tensor_core_layouts() {
        const TOKENS: usize = 19;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        let width = KV_HEADS * HEAD_DIM;
        let keys = (0..TOKENS * width)
            .map(|index| ((index * 29 % 127) as f32 - 63.0) / 64.0)
            .collect::<Vec<_>>();
        let values = (0..TOKENS * width)
            .map(|index| ((index * 43 % 113) as f32 - 56.0) / 64.0)
            .collect::<Vec<_>>();
        let key_rows = DeviceBuffer::from_host(&keys).expect("keys");
        let value_rows = DeviceBuffer::from_host(&values).expect("values");
        let stream = CudaStream::new_blocking().expect("stream");
        let mut cache = Sm12xKvCache::new(32, KV_HEADS, HEAD_DIM).expect("cache");
        cache
            .append_rows_at_offset_on_stream(&key_rows, &value_rows, 0, TOKENS, &stream)
            .expect("append rows");
        let mut key_output = DeviceBuffer::zeroed(TOKENS * width).expect("unpacked keys");
        let mut value_output = DeviceBuffer::zeroed(TOKENS * width).expect("unpacked values");
        cache
            .unpack_bf16_on_stream(key_output.output(), value_output.output(), &stream)
            .expect("unpack cache");
        let actual_keys = key_output.copy_to_host(&stream).expect("read keys");
        let actual_values = value_output.copy_to_host(&stream).expect("read values");

        let mut staged_cache = Sm12xKvCache::new(32, KV_HEADS, HEAD_DIM).expect("staged cache");
        let mut staged_keys = DeviceBuffer::zeroed(TOKENS * width).expect("staged keys");
        let mut staged_values = DeviceBuffer::zeroed(TOKENS * width).expect("staged values");
        staged_cache
            .append_initial_rows_and_stage_bf16_on_stream(
                &key_rows,
                &value_rows,
                0,
                TOKENS,
                staged_keys.output(),
                staged_values.output(),
                &stream,
            )
            .expect("append and stage rows");
        assert_eq!(
            &*staged_keys.copy_to_host(&stream).expect("read staged keys"),
            &*actual_keys,
        );
        assert_eq!(
            &*staged_values
                .copy_to_host(&stream)
                .expect("read staged values"),
            &*actual_values,
        );

        let mut max_key_error = 0.0f32;
        let mut max_value_error = 0.0f32;
        for token in 0..TOKENS {
            for head in 0..KV_HEADS {
                for dim in 0..HEAD_DIM {
                    let dense = token * width + head * HEAD_DIM + dim;
                    let packed_key = (head * TOKENS + token) * HEAD_DIM + dim;
                    let packed_value = (head * HEAD_DIM + dim) * TOKENS + token;
                    max_key_error = max_key_error
                        .max((bf16_to_f32(actual_keys[packed_key]) - keys[dense]).abs());
                    max_value_error = max_value_error
                        .max((bf16_to_f32(actual_values[packed_value]) - values[dense]).abs());
                }
            }
        }
        assert!(max_key_error < 0.20, "key max error {max_key_error}");
        assert!(max_value_error < 0.20, "value max error {max_value_error}");
    }
}
