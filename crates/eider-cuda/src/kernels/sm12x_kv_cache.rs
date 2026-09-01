//! Compact SM12x FP4 cache storage and append-time quantization.

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceAddress, DeviceBuffer, DeviceOutput, Error, Result};
use std::mem::{align_of, size_of};

const K_TOKEN_TILE: usize = 8;
const V_TOKEN_BLOCK: usize = 16;
const MMA_K: usize = 64;
const MMA_N: usize = 8;
const COMPACT_TILE_BYTES: usize = MMA_N * MMA_K / 2;
const SCALE_BYTES_PER_TILE: usize = MMA_K / 16;
const PV_SPLIT_CAPACITY: usize = 32;
const PV_SPLIT_MIN_TOKENS: usize = 1_024;
/// Initial physical page size used by Eider's compact SM12x cache pool.
pub const SM12X_KV_PAGE_TOKENS: usize = 128;

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

/// Stable-slot compact FP4 K/V pages for one attention layer.
pub struct Sm12xKvPagePool {
    storage: DeviceBuffer<u8>,
    layout: Sm12xKvCacheLayout,
    page_slots: usize,
    kv_heads: usize,
    head_dim: usize,
}

/// Device-resident copy of the f32 staging rows needed to roll back a
/// speculative append that wraps the cache's 16-row tail.
pub struct Sm12xKvTailSnapshot {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    kv_heads: usize,
    head_dim: usize,
}

impl Sm12xKvTailSnapshot {
    /// Allocates one snapshot for a compact cache shape.
    pub fn new(kv_heads: usize, head_dim: usize) -> Result<Self> {
        if kv_heads == 0 || head_dim == 0 {
            return Err(Error::Shape {
                label: "SM12x KV tail snapshot",
                expected: "non-zero head dimensions".to_string(),
                actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
            });
        }
        let values = V_TOKEN_BLOCK
            .checked_mul(kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::Shape {
                label: "SM12x KV tail snapshot",
                expected: "shape without overflow".to_string(),
                actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
            })?;
        Ok(Self {
            key: DeviceBuffer::zeroed(values)?,
            value: DeviceBuffer::zeroed(values)?,
            kv_heads,
            head_dim,
        })
    }

    /// Returns the resident bytes used by this snapshot.
    pub fn device_bytes(&self) -> usize {
        self.key.device_bytes() + self.value.device_bytes()
    }
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
    pub key_values: DeviceAddress<u8>,
    pub key_scales: DeviceAddress<u8>,
    pub value_values: DeviceAddress<u8>,
    pub value_scales: DeviceAddress<u8>,
    pub key_tail: DeviceAddress<f32>,
    pub value_tail: DeviceAddress<f32>,
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

    pub(crate) fn compact_parts(&self) -> Result<Sm12xKvCacheParts> {
        let base = self.storage.cuda_address();
        Ok(Sm12xKvCacheParts {
            key_values: base.offset(self.layout.key_values)?,
            key_scales: base.offset(self.layout.key_scales)?,
            value_values: base.offset(self.layout.value_values)?,
            value_scales: base.offset(self.layout.value_scales)?,
            key_tail: base.offset(self.layout.key_tail)?.cast(),
            value_tail: base.offset(self.layout.value_tail)?.cast(),
        })
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

        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append(
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
            )?;
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append(
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
            )?;
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append_rows(
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
            )?;
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append_rows(
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
            )?;
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append_indexed(
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
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
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

    /// Captures the complete circular f32 tail before a speculative append.
    pub fn snapshot_tail_on_stream(
        &self,
        snapshot: &mut Sm12xKvTailSnapshot,
        stream: &CudaStream,
    ) -> Result<()> {
        self.validate_tail_snapshot(snapshot)?;
        let bytes = V_TOKEN_BLOCK * self.kv_heads * self.head_dim * size_of::<f32>();
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D KV key tail snapshot)",
                crate::ffi::cudaMemcpyAsync(
                    snapshot.key.ptr.cast(),
                    self.key_tail_ptr().cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )?;
            check_cuda(
                "cudaMemcpyAsync(D2D KV value tail snapshot)",
                crate::ffi::cudaMemcpyAsync(
                    snapshot.value.ptr.cast(),
                    self.value_tail_ptr().cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Restores old rows at the start of the circular tail after speculative
    /// rows were discarded within the same 16-token value block.
    pub fn restore_tail_prefix_on_stream(
        &mut self,
        snapshot: &Sm12xKvTailSnapshot,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.validate_tail_snapshot(snapshot)?;
        if rows > V_TOKEN_BLOCK {
            return Err(Error::Shape {
                label: "SM12x KV tail restore",
                expected: format!("at most {V_TOKEN_BLOCK} rows"),
                actual: format!("{rows} rows"),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let bytes = rows * self.kv_heads * self.head_dim * size_of::<f32>();
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D KV key tail restore)",
                crate::ffi::cudaMemcpyAsync(
                    self.key_tail_mut_ptr().cast(),
                    snapshot.key.ptr.cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )?;
            check_cuda(
                "cudaMemcpyAsync(D2D KV value tail restore)",
                crate::ffi::cudaMemcpyAsync(
                    self.value_tail_mut_ptr().cast(),
                    snapshot.value.ptr.cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    fn validate_tail_snapshot(&self, snapshot: &Sm12xKvTailSnapshot) -> Result<()> {
        if snapshot.kv_heads != self.kv_heads || snapshot.head_dim != self.head_dim {
            return Err(Error::Shape {
                label: "SM12x KV tail snapshot shape",
                expected: format!("kv_heads={} head_dim={}", self.kv_heads, self.head_dim),
                actual: format!(
                    "kv_heads={} head_dim={}",
                    snapshot.kv_heads, snapshot.head_dim
                ),
            });
        }
        Ok(())
    }
}

impl Sm12xKvPagePool {
    /// Preallocates stable physical slots for one full-attention layer.
    pub fn new(page_slots: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        if page_slots == 0 {
            return Err(Error::Shape {
                label: "SM12x KV page pool",
                expected: "at least one physical page slot".to_string(),
                actual: "0".to_string(),
            });
        }
        let layout = Sm12xKvCache::layout(SM12X_KV_PAGE_TOKENS, kv_heads, head_dim)?;
        u32::try_from(layout.total_bytes).map_err(|_| Error::Shape {
            label: "SM12x KV page stride",
            expected: "page bytes fitting u32".to_string(),
            actual: layout.total_bytes.to_string(),
        })?;
        let total_bytes =
            layout
                .total_bytes
                .checked_mul(page_slots)
                .ok_or_else(|| Error::Shape {
                    label: "SM12x KV page pool",
                    expected: "pool byte count without overflow".to_string(),
                    actual: format!("page_slots={page_slots} page_bytes={}", layout.total_bytes),
                })?;
        Ok(Self {
            storage: DeviceBuffer::uninitialized(total_bytes)?,
            layout,
            page_slots,
            kv_heads,
            head_dim,
        })
    }

    /// Returns the number of stable physical slots.
    pub fn page_slots(&self) -> usize {
        self.page_slots
    }

    /// Returns the exact bytes occupied by one slot.
    pub fn page_bytes(&self) -> usize {
        self.layout.total_bytes
    }

    /// Returns total device bytes preallocated by the pool.
    pub fn device_bytes(&self) -> usize {
        self.storage.device_bytes()
    }

    /// Allocates rollback storage matching this pool's circular f32 tail.
    pub fn tail_snapshot(&self) -> Result<Sm12xKvTailSnapshot> {
        Sm12xKvTailSnapshot::new(self.kv_heads, self.head_dim)
    }

    fn check_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.page_slots {
            return Err(Error::Shape {
                label: "SM12x KV page slot",
                expected: format!("slot < {}", self.page_slots),
                actual: slot.to_string(),
            });
        }
        Ok(())
    }

    fn component_ptr(&self, offset: usize) -> *const u8 {
        unsafe { self.storage.as_const_ptr().cast::<u8>().add(offset) }
    }

    fn component_mut_ptr(&mut self, slot: usize, offset: usize) -> *mut u8 {
        let byte = slot * self.layout.total_bytes + offset;
        unsafe { self.storage.as_mut_ptr().cast::<u8>().add(byte) }
    }

    /// Captures one physical page's circular f32 tail before a speculative append.
    pub fn snapshot_tail_on_stream(
        &self,
        slot: usize,
        snapshot: &mut Sm12xKvTailSnapshot,
        stream: &CudaStream,
    ) -> Result<()> {
        self.check_slot(slot)?;
        self.validate_tail_snapshot(snapshot)?;
        let bytes = V_TOKEN_BLOCK * self.kv_heads * self.head_dim * size_of::<f32>();
        let page_offset = slot * self.layout.total_bytes;
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D paged KV key tail snapshot)",
                crate::ffi::cudaMemcpyAsync(
                    snapshot.key.ptr.cast(),
                    self.component_ptr(page_offset + self.layout.key_tail)
                        .cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )?;
            check_cuda(
                "cudaMemcpyAsync(D2D paged KV value tail snapshot)",
                crate::ffi::cudaMemcpyAsync(
                    snapshot.value.ptr.cast(),
                    self.component_ptr(page_offset + self.layout.value_tail)
                        .cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Restores valid rows overwritten when a speculative append wrapped a page tail.
    pub fn restore_tail_prefix_on_stream(
        &mut self,
        slot: usize,
        snapshot: &Sm12xKvTailSnapshot,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.check_slot(slot)?;
        self.validate_tail_snapshot(snapshot)?;
        if rows > V_TOKEN_BLOCK {
            return Err(Error::Shape {
                label: "SM12x paged KV tail restore",
                expected: format!("at most {V_TOKEN_BLOCK} rows"),
                actual: format!("{rows} rows"),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let bytes = rows * self.kv_heads * self.head_dim * size_of::<f32>();
        let key_tail_offset = self.layout.key_tail;
        let value_tail_offset = self.layout.value_tail;
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D paged KV key tail restore)",
                crate::ffi::cudaMemcpyAsync(
                    self.component_mut_ptr(slot, key_tail_offset).cast(),
                    snapshot.key.ptr.cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )?;
            check_cuda(
                "cudaMemcpyAsync(D2D paged KV value tail restore)",
                crate::ffi::cudaMemcpyAsync(
                    self.component_mut_ptr(slot, value_tail_offset).cast(),
                    snapshot.value.ptr.cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    fn validate_tail_snapshot(&self, snapshot: &Sm12xKvTailSnapshot) -> Result<()> {
        if snapshot.kv_heads != self.kv_heads || snapshot.head_dim != self.head_dim {
            return Err(Error::Shape {
                label: "SM12x paged KV tail snapshot shape",
                expected: format!("kv_heads={} head_dim={}", self.kv_heads, self.head_dim),
                actual: format!(
                    "kv_heads={} head_dim={}",
                    snapshot.kv_heads, snapshot.head_dim
                ),
            });
        }
        Ok(())
    }

    /// Copies one physical page slot on the explicit stream.
    pub fn copy_page_on_stream(
        &mut self,
        source_slot: usize,
        destination_slot: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.check_slot(source_slot)?;
        self.check_slot(destination_slot)?;
        let bytes = self.layout.total_bytes;
        let source_offset = source_slot * bytes;
        let destination_offset = destination_slot * bytes;
        unsafe {
            let source = self.storage.as_const_ptr().cast::<u8>().add(source_offset);
            let destination = self
                .storage
                .as_mut_ptr()
                .cast::<u8>()
                .add(destination_offset);
            check_cuda(
                "cudaMemcpyAsync(D2D SM12x KV page)",
                crate::ffi::cudaMemcpyAsync(
                    destination.cast(),
                    source.cast(),
                    bytes,
                    crate::ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Appends one projected K/V row to a physical page slot.
    #[allow(clippy::too_many_arguments)]
    pub fn append_at_offsets_on_stream(
        &mut self,
        slot: usize,
        position: usize,
        key: &DeviceBuffer<f32>,
        key_offset: usize,
        value: &DeviceBuffer<f32>,
        value_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.check_slot(slot)?;
        let width = self.kv_heads * self.head_dim;
        if position >= SM12X_KV_PAGE_TOKENS
            || key_offset
                .checked_add(width)
                .is_none_or(|end| end > key.len())
            || value_offset
                .checked_add(width)
                .is_none_or(|end| end > value.len())
        {
            return Err(Error::Shape {
                label: "SM12x paged KV append",
                expected: format!(
                    "position < {SM12X_KV_PAGE_TOKENS} and {width} readable K/V values"
                ),
                actual: format!(
                    "position={position} key_len={} key_offset={key_offset} value_len={} value_offset={value_offset}",
                    key.len(),
                    value.len()
                ),
            });
        }
        let key_values_offset = self.layout.key_values;
        let key_scales_offset = self.layout.key_scales;
        let value_values_offset = self.layout.value_values;
        let value_scales_offset = self.layout.value_scales;
        let key_tail_offset = self.layout.key_tail;
        let value_tail_offset = self.layout.value_tail;
        let page_base = self.component_mut_ptr(slot, 0);
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append(
                key.as_const_ptr().cast::<f32>().add(key_offset),
                value.as_const_ptr().cast::<f32>().add(value_offset),
                page_base.add(key_values_offset),
                page_base.add(key_scales_offset),
                page_base.add(value_values_offset),
                page_base.add(value_scales_offset),
                page_base.add(key_tail_offset).cast(),
                page_base.add(value_tail_offset).cast(),
                position as u32,
                SM12X_KV_PAGE_TOKENS as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_on_stream(
                    key.as_const_ptr().cast::<f32>().add(key_offset),
                    value.as_const_ptr().cast::<f32>().add(value_offset),
                    page_base.add(key_values_offset),
                    page_base.add(key_scales_offset),
                    page_base.add(value_values_offset),
                    page_base.add(value_scales_offset),
                    page_base.add(key_tail_offset).cast(),
                    page_base.add(value_tail_offset).cast(),
                    position as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Appends projected rows which fit wholly within one physical page.
    #[allow(clippy::too_many_arguments)]
    pub fn append_rows_at_offset_on_stream(
        &mut self,
        slot: usize,
        start_position: usize,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.check_slot(slot)?;
        let width = self.kv_heads * self.head_dim;
        let input_end = input_row_offset
            .checked_add(rows)
            .and_then(|end| end.checked_mul(width));
        if rows == 0
            || start_position
                .checked_add(rows)
                .is_none_or(|end| end > SM12X_KV_PAGE_TOKENS)
            || input_end.is_none_or(|end| end > key.len() || end > value.len())
        {
            return Err(Error::Shape {
                label: "SM12x paged KV row append",
                expected: format!(
                    "non-empty rows within a {SM12X_KV_PAGE_TOKENS}-token page and input buffers"
                ),
                actual: format!(
                    "start={start_position} rows={rows} input_row_offset={input_row_offset} key={} value={}",
                    key.len(),
                    value.len()
                ),
            });
        }
        let key_values_offset = self.layout.key_values;
        let key_scales_offset = self.layout.key_scales;
        let value_values_offset = self.layout.value_values;
        let value_scales_offset = self.layout.value_scales;
        let key_tail_offset = self.layout.key_tail;
        let value_tail_offset = self.layout.value_tail;
        let page_base = self.component_mut_ptr(slot, 0);
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append_rows(
                key.as_const_ptr().cast(),
                value.as_const_ptr().cast(),
                page_base.add(key_values_offset),
                page_base.add(key_scales_offset),
                page_base.add(value_values_offset),
                page_base.add(value_scales_offset),
                page_base.add(key_tail_offset).cast(),
                page_base.add(value_tail_offset).cast(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                input_row_offset as u32,
                start_position as u32,
                rows as u32,
                SM12X_KV_PAGE_TOKENS as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_append_rows_on_stream",
                crate::ffi::infer_sm12x_kv_cache_append_rows_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    page_base.add(key_values_offset),
                    page_base.add(key_scales_offset),
                    page_base.add(value_values_offset),
                    page_base.add(value_scales_offset),
                    page_base.add(key_tail_offset).cast(),
                    page_base.add(value_tail_offset).cast(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    input_row_offset as u32,
                    start_position as u32,
                    rows as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Dequantizes a logical paged cache into the BF16 tensor-core layouts.
    ///
    /// Keys are written as `[kv_heads, tokens, head_dim]`; values are written
    /// as `[kv_heads, head_dim, tokens]`. The page table must contain every
    /// logical page intersecting `cache_len` in logical order.
    pub fn unpack_paged_bf16_on_stream(
        &self,
        page_table: &DeviceBuffer<u32>,
        cache_len: usize,
        mut key_output: DeviceOutput<'_, u16>,
        mut value_output: DeviceOutput<'_, u16>,
        stream: &CudaStream,
    ) -> Result<()> {
        let logical_pages = cache_len.div_ceil(SM12X_KV_PAGE_TOKENS);
        let values = checked_product(
            "SM12x paged KV BF16 unpack",
            &[cache_len, self.kv_heads, self.head_dim],
        )?;
        if cache_len == 0
            || cache_len > u32::MAX as usize
            || page_table.len() < logical_pages
            || key_output.len() < values
            || value_output.len() < values
        {
            return Err(Error::Shape {
                label: "SM12x paged KV BF16 unpack",
                expected: format!(
                    "non-empty cache, page table >= {logical_pages}, and K/V outputs >= {values} values"
                ),
                actual: format!(
                    "cache_len={cache_len} page_table={} key={} value={}",
                    page_table.len(),
                    key_output.len(),
                    value_output.len()
                ),
            });
        }
        let layout = &self.layout;
        unsafe {
            check_cuda(
                "infer_sm12x_kv_cache_unpack_paged_bf16_on_stream",
                crate::ffi::infer_sm12x_kv_cache_unpack_paged_bf16_on_stream(
                    self.component_ptr(layout.key_values),
                    self.component_ptr(layout.key_scales),
                    self.component_ptr(layout.value_values),
                    self.component_ptr(layout.value_scales),
                    self.component_ptr(layout.key_tail).cast(),
                    self.component_ptr(layout.value_tail).cast(),
                    page_table.as_const_ptr().cast(),
                    key_output.as_mut_ptr().cast(),
                    value_output.as_mut_ptr().cast(),
                    cache_len as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    layout.total_bytes as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    stream.as_raw(),
                ),
            )
        }
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
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
                std::ptr::null(),
                cache.max_tokens as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                pv_splits as u32,
                0,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                u32::MAX,
                0,
                0,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
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
                std::ptr::null(),
                cache.max_tokens as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                pv_splits as u32,
                window_start as u32,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                u32::MAX,
                0,
                0,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::append_causal_attention_rows(
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
            )?;
        }
        #[cfg(not(feature = "cuda-oxide"))]
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

    /// Computes causal attention for a row chunk already appended to paged storage.
    ///
    /// The rows must not cross the compact cache's 16-token tail boundary. Query
    /// and output buffers are row-major and use the same row offset.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_paged_causal_rows_at_offset_into_on_stream(
        &mut self,
        pool: &Sm12xKvPagePool,
        page_table: &DeviceBuffer<u32>,
        start_position: usize,
        query: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        window_tokens: Option<usize>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let logical_capacity = page_table
            .len()
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .ok_or_else(|| Error::Shape {
                label: "SM12x paged causal row capacity",
                expected: "page-table capacity without overflow".to_string(),
                actual: page_table.len().to_string(),
            })?;
        if logical_capacity > self.max_tokens
            || pool.kv_heads != self.kv_heads
            || pool.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x paged causal row attention cache",
                expected: format!(
                    "workspace max >= {logical_capacity}, kv_heads={}, head_dim={}",
                    self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "workspace_max={} pool_shape={}/{}",
                    self.max_tokens, pool.kv_heads, pool.head_dim
                ),
            });
        }
        let q_width = self.q_heads * self.head_dim;
        let row_end = input_row_offset
            .checked_add(rows)
            .ok_or_else(|| Error::Shape {
                label: "SM12x paged causal row attention",
                expected: "input row range without overflow".to_string(),
                actual: format!("input_row_offset={input_row_offset} rows={rows}"),
            })?;
        let q_end = row_end.checked_mul(q_width).ok_or_else(|| Error::Shape {
            label: "SM12x paged causal row attention",
            expected: "query row range without overflow".to_string(),
            actual: format!("row_end={row_end} q_width={q_width}"),
        })?;
        let cache_end = start_position
            .checked_add(rows)
            .ok_or_else(|| Error::Shape {
                label: "SM12x paged causal row attention",
                expected: "cache row range without overflow".to_string(),
                actual: format!("start={start_position} rows={rows}"),
            })?;
        if rows == 0
            || rows > u32::MAX as usize
            || input_row_offset > u32::MAX as usize
            || start_position > u32::MAX as usize
            || rows > self.causal_row_capacity
            || rows > V_TOKEN_BLOCK - start_position % V_TOKEN_BLOCK
            || q_end > query.len()
            || q_end > output.len()
            || cache_end > logical_capacity
            || window_tokens.is_some_and(|window| window == 0 || window > u32::MAX as usize)
        {
            return Err(Error::Shape {
                label: "SM12x paged causal row attention buffers",
                expected: format!(
                    "rows within one 16-token tail through logical capacity {logical_capacity}, q/output >= {q_end}"
                ),
                actual: format!(
                    "start={start_position} rows={rows} query={} output={}",
                    query.len(),
                    output.len()
                ),
            });
        }
        let layout = &pool.layout;
        let key_values = layout.key_values;
        let key_scales = layout.key_scales;
        let value_values = layout.value_values;
        let value_scales = layout.value_scales;
        let key_tail = layout.key_tail;
        let value_tail = layout.value_tail;
        let page_stride = layout.total_bytes;
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
                query.as_const_ptr().cast(),
                pool.component_ptr(key_values),
                pool.component_ptr(key_scales),
                pool.component_ptr(key_tail).cast(),
                pool.component_ptr(value_values),
                pool.component_ptr(value_scales),
                pool.component_ptr(value_tail).cast(),
                self.query_tiles.as_mut_ptr().cast(),
                self.query_scales.as_mut_ptr().cast(),
                self.scores.as_mut_ptr().cast(),
                self.probability_tiles.as_mut_ptr().cast(),
                self.probability_scales.as_mut_ptr().cast(),
                self.pv_partials.as_mut_ptr().cast(),
                output.as_mut_ptr().cast(),
                0,
                std::ptr::null(),
                logical_capacity as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                1,
                0,
                page_table.as_const_ptr().cast(),
                SM12X_KV_PAGE_TOKENS as u32,
                page_stride as u32,
                std::ptr::null(),
                std::ptr::null(),
                0,
                input_row_offset as u32,
                rows as u32,
                start_position as u32,
                window_tokens.unwrap_or(0) as u32,
                input_row_offset as u32,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm12x_kv_paged_causal_attention_rows_on_stream",
                crate::ffi::infer_sm12x_kv_paged_causal_attention_rows_on_stream(
                    query.as_const_ptr().cast(),
                    pool.component_ptr(key_values),
                    pool.component_ptr(key_scales),
                    pool.component_ptr(value_values),
                    pool.component_ptr(value_scales),
                    pool.component_ptr(key_tail).cast(),
                    pool.component_ptr(value_tail).cast(),
                    page_table.as_const_ptr().cast(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    input_row_offset as u32,
                    start_position as u32,
                    rows as u32,
                    logical_capacity as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    page_stride as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    window_tokens.unwrap_or(0) as u32,
                    self.causal_row_capacity as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Computes non-causal attention for several query rows over one fixed
    /// compact-cache window. The caller must append the block's K/V rows first
    /// so every query can attend the complete block.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_rows_window_at_offset_into_on_stream(
        &mut self,
        cache: &Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        window_start: usize,
        mut output: DeviceOutput<'_, f32>,
        output_row_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = self.q_heads * self.head_dim;
        let input_end = input_row_offset
            .checked_add(rows)
            .and_then(|end| end.checked_mul(q_width))
            .unwrap_or(usize::MAX);
        let output_end = output_row_offset
            .checked_add(rows)
            .and_then(|end| end.checked_mul(q_width))
            .unwrap_or(usize::MAX);
        if cache.len == 0
            || cache.max_tokens > self.max_tokens
            || cache.kv_heads != self.kv_heads
            || cache.head_dim != self.head_dim
            || rows == 0
            || rows > self.causal_row_capacity
            || window_start >= cache.len
            || input_end > query.len()
            || output_end > output.len()
            || input_row_offset > u32::MAX as usize
            || output_row_offset > u32::MAX as usize
        {
            return Err(Error::Shape {
                label: "SM12x non-causal row attention",
                expected: format!(
                    "1..={} rows over cache len {} with matching q/output buffers",
                    self.causal_row_capacity, cache.len
                ),
                actual: format!(
                    "rows={rows} window_start={window_start} query={} output={} cache_len={} cache_max={} workspace_max={}",
                    query.len(),
                    output.len(),
                    cache.len,
                    cache.max_tokens,
                    self.max_tokens
                ),
            });
        }
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
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
                std::ptr::null(),
                cache.max_tokens as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                1,
                window_start as u32,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                input_row_offset as u32,
                rows as u32,
                u32::MAX,
                0,
                output_row_offset as u32,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm12x_kv_attention_rows_window_on_stream",
                crate::ffi::infer_sm12x_kv_attention_rows_window_on_stream(
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
                    output.as_mut_ptr().cast(),
                    input_row_offset as u32,
                    output_row_offset as u32,
                    rows as u32,
                    cache.len as u32,
                    window_start as u32,
                    cache.max_tokens as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    self.causal_row_capacity as u32,
                    stream.as_raw(),
                ),
            )
        }
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
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
                std::ptr::null(),
                cache.max_tokens as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                pv_splits as u32,
                0,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                u32::MAX,
                0,
                0,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
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

    /// Enqueues compact attention through a stable device page table.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_paged_offsets_into_on_stream(
        &mut self,
        pool: &Sm12xKvPagePool,
        page_table: &DeviceBuffer<u32>,
        cache_len: usize,
        query: &DeviceBuffer<f32>,
        query_offset: usize,
        output: DeviceOutput<'_, f32>,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.attention_paged_window_offsets_into_on_stream(
            pool,
            page_table,
            cache_len,
            query,
            query_offset,
            output,
            output_offset,
            0,
            stream,
        )
    }

    /// Enqueues QSA-selected compact attention through a stable device page table.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_paged_sparse_offsets_into_on_stream(
        &mut self,
        pool: &Sm12xKvPagePool,
        page_table: &DeviceBuffer<u32>,
        cache_len: usize,
        selected_blocks: &DeviceBuffer<u8>,
        selected_tiles: &DeviceBuffer<u8>,
        selected_tokens: usize,
        query: &DeviceBuffer<f32>,
        query_offset: usize,
        mut output: DeviceOutput<'_, f32>,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let logical_capacity = page_table
            .len()
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .ok_or_else(|| Error::Shape {
                label: "SM12x paged sparse KV capacity",
                expected: "page-table capacity without overflow".to_string(),
                actual: page_table.len().to_string(),
            })?;
        let required_blocks = logical_capacity.div_ceil(4);
        let required_tiles = logical_capacity.div_ceil(64);
        if cache_len == 0
            || cache_len > logical_capacity
            || logical_capacity > self.max_tokens
            || selected_tokens == 0
            || selected_tokens > cache_len
            || selected_blocks.len() < required_blocks
            || selected_tiles.len() < required_tiles
            || pool.kv_heads != self.kv_heads
            || pool.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x paged sparse KV attention cache",
                expected: format!(
                    "cache within {logical_capacity}, selected masks >= {required_blocks}/{required_tiles}, workspace max >= capacity, kv_heads={}, head_dim={}",
                    self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "cache_len={cache_len} selected_tokens={selected_tokens} masks={}/{} workspace_max={} pool_shape={}/{}",
                    selected_blocks.len(),
                    selected_tiles.len(),
                    self.max_tokens,
                    pool.kv_heads,
                    pool.head_dim
                ),
            });
        }
        let query_width = self.q_heads * self.head_dim;
        if query_offset
            .checked_add(query_width)
            .is_none_or(|end| end > query.len())
            || output_offset
                .checked_add(query_width)
                .is_none_or(|end| end > output.len())
        {
            return Err(Error::Shape {
                label: "SM12x paged sparse KV attention offsets",
                expected: format!("{query_width} readable/writable values at row offsets"),
                actual: format!(
                    "query_len={} query_offset={query_offset} output_len={} output_offset={output_offset}",
                    query.len(),
                    output.len()
                ),
            });
        }
        let layout = &pool.layout;
        let pv_splits = pv_split_count(cache_len);
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
                query.as_const_ptr().cast::<f32>().add(query_offset),
                pool.component_ptr(layout.key_values),
                pool.component_ptr(layout.key_scales),
                pool.component_ptr(layout.key_tail).cast(),
                pool.component_ptr(layout.value_values),
                pool.component_ptr(layout.value_scales),
                pool.component_ptr(layout.value_tail).cast(),
                self.query_tiles.as_mut_ptr().cast(),
                self.query_scales.as_mut_ptr().cast(),
                self.scores.as_mut_ptr().cast(),
                self.probability_tiles.as_mut_ptr().cast(),
                self.probability_scales.as_mut_ptr().cast(),
                self.pv_partials.as_mut_ptr().cast(),
                output.as_mut_ptr().cast::<f32>().add(output_offset),
                cache_len as u32,
                std::ptr::null(),
                logical_capacity as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                pv_splits as u32,
                0,
                page_table.as_const_ptr().cast(),
                SM12X_KV_PAGE_TOKENS as u32,
                layout.total_bytes as u32,
                selected_blocks.as_const_ptr().cast(),
                selected_tiles.as_const_ptr().cast(),
                selected_tokens as u32,
                0,
                1,
                u32::MAX,
                0,
                0,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm12x_kv_paged_sparse_attention_on_stream",
                crate::ffi::infer_sm12x_kv_paged_sparse_attention_on_stream(
                    query.as_const_ptr().cast::<f32>().add(query_offset),
                    pool.component_ptr(layout.key_values),
                    pool.component_ptr(layout.key_scales),
                    pool.component_ptr(layout.key_tail).cast(),
                    pool.component_ptr(layout.value_values),
                    pool.component_ptr(layout.value_scales),
                    pool.component_ptr(layout.value_tail).cast(),
                    page_table.as_const_ptr().cast(),
                    selected_blocks.as_const_ptr().cast(),
                    selected_tiles.as_const_ptr().cast(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast::<f32>().add(output_offset),
                    cache_len as u32,
                    selected_tokens as u32,
                    logical_capacity as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    layout.total_bytes as u32,
                    self.q_heads as u32,
                    self.kv_heads as u32,
                    self.head_dim as u32,
                    pv_splits as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues compact windowed attention through a stable device page table.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_paged_window_offsets_into_on_stream(
        &mut self,
        pool: &Sm12xKvPagePool,
        page_table: &DeviceBuffer<u32>,
        cache_len: usize,
        query: &DeviceBuffer<f32>,
        query_offset: usize,
        mut output: DeviceOutput<'_, f32>,
        output_offset: usize,
        window_start: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let logical_capacity = page_table
            .len()
            .checked_mul(SM12X_KV_PAGE_TOKENS)
            .ok_or_else(|| Error::Shape {
                label: "SM12x paged KV capacity",
                expected: "page-table capacity without overflow".to_string(),
                actual: page_table.len().to_string(),
            })?;
        if cache_len == 0
            || cache_len > logical_capacity
            || window_start >= cache_len
            || logical_capacity > self.max_tokens
            || pool.kv_heads != self.kv_heads
            || pool.head_dim != self.head_dim
        {
            return Err(Error::Shape {
                label: "SM12x paged KV attention cache",
                expected: format!(
                    "cache_len in 1..={logical_capacity}, workspace max >= capacity, kv_heads={}, head_dim={}",
                    self.kv_heads, self.head_dim
                ),
                actual: format!(
                    "cache_len={cache_len} workspace_max={} pool_shape={}/{}",
                    self.max_tokens, pool.kv_heads, pool.head_dim
                ),
            });
        }
        let query_width = self.q_heads * self.head_dim;
        if query_offset
            .checked_add(query_width)
            .is_none_or(|end| end > query.len())
            || output_offset
                .checked_add(query_width)
                .is_none_or(|end| end > output.len())
        {
            return Err(Error::Shape {
                label: "SM12x paged KV attention offsets",
                expected: format!("{query_width} readable/writable values at row offsets"),
                actual: format!(
                    "query_len={} query_offset={query_offset} output_len={} output_offset={output_offset}",
                    query.len(),
                    output.len()
                ),
            });
        }
        let layout = &pool.layout;
        let pv_splits = pv_split_count(cache_len);
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
                query.as_const_ptr().cast::<f32>().add(query_offset),
                pool.component_ptr(layout.key_values),
                pool.component_ptr(layout.key_scales),
                pool.component_ptr(layout.key_tail).cast(),
                pool.component_ptr(layout.value_values),
                pool.component_ptr(layout.value_scales),
                pool.component_ptr(layout.value_tail).cast(),
                self.query_tiles.as_mut_ptr().cast(),
                self.query_scales.as_mut_ptr().cast(),
                self.scores.as_mut_ptr().cast(),
                self.probability_tiles.as_mut_ptr().cast(),
                self.probability_scales.as_mut_ptr().cast(),
                self.pv_partials.as_mut_ptr().cast(),
                output.as_mut_ptr().cast::<f32>().add(output_offset),
                cache_len as u32,
                std::ptr::null(),
                logical_capacity as u32,
                self.q_heads as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                pv_splits as u32,
                window_start as u32,
                page_table.as_const_ptr().cast(),
                SM12X_KV_PAGE_TOKENS as u32,
                layout.total_bytes as u32,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                u32::MAX,
                0,
                0,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm12x_kv_paged_attention_on_stream",
                crate::ffi::infer_sm12x_kv_paged_attention_on_stream(
                    query.as_const_ptr().cast::<f32>().add(query_offset),
                    pool.component_ptr(layout.key_values),
                    pool.component_ptr(layout.key_scales),
                    pool.component_ptr(layout.key_tail).cast(),
                    pool.component_ptr(layout.value_values),
                    pool.component_ptr(layout.value_scales),
                    pool.component_ptr(layout.value_tail).cast(),
                    page_table.as_const_ptr().cast(),
                    self.query_tiles.as_mut_ptr().cast(),
                    self.query_scales.as_mut_ptr().cast(),
                    self.scores.as_mut_ptr().cast(),
                    self.probability_tiles.as_mut_ptr().cast(),
                    self.probability_scales.as_mut_ptr().cast(),
                    self.pv_partials.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast::<f32>().add(output_offset),
                    cache_len as u32,
                    window_start as u32,
                    logical_capacity as u32,
                    SM12X_KV_PAGE_TOKENS as u32,
                    layout.total_bytes as u32,
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
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            super::sm12x_kv_cache_oxide::attention(
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
                0,
                cache_len.as_const_ptr().cast(),
                cache.max_tokens as u32,
                self.kv_heads as u32 * MMA_N as u32,
                self.kv_heads as u32,
                self.head_dim as u32,
                PV_SPLIT_CAPACITY as u32,
                0,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                1,
                u32::MAX,
                0,
                0,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
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
    fn paged_attention_matches_contiguous_cache_across_page_boundaries() {
        const TOKENS: usize = 257;
        const CAPACITY: usize = 3 * SM12X_KV_PAGE_TOKENS;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = 8;
        let kv_width = KV_HEADS * HEAD_DIM;
        let key = (0..TOKENS * kv_width)
            .map(|index| ((index * 17 + 11) % 251) as f32 / 64.0 - 1.5)
            .collect::<Vec<_>>();
        let value = (0..TOKENS * kv_width)
            .map(|index| ((index * 29 + 7) % 257) as f32 / 80.0 - 1.25)
            .collect::<Vec<_>>();
        let query = (0..Q_HEADS * HEAD_DIM)
            .map(|index| ((index * 13 + 5) % 127) as f32 / 48.0 - 1.0)
            .collect::<Vec<_>>();
        let key = DeviceBuffer::from_host(&key).expect("key");
        let value = DeviceBuffer::from_host(&value).expect("value");
        let query = DeviceBuffer::from_host(&query).expect("query");
        let page_table = DeviceBuffer::from_host(&[2_u32, 0, 3]).expect("page table");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut contiguous =
            Sm12xKvCache::new(CAPACITY, KV_HEADS, HEAD_DIM).expect("contiguous cache");
        let mut pool = Sm12xKvPagePool::new(4, KV_HEADS, HEAD_DIM).expect("page pool");
        let mut workspace =
            Sm12xKvAttentionWorkspace::new_gqa(CAPACITY, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("attention workspace");
        let checkpoints = [1, 7, 8, 15, 16, 127, 128, 129, 255, 256, 257];

        for token in 0..TOKENS {
            contiguous
                .append_at_offsets_on_stream(
                    &key,
                    token * kv_width,
                    &value,
                    token * kv_width,
                    token,
                    &stream,
                )
                .expect("contiguous append");
            let logical_page = token / SM12X_KV_PAGE_TOKENS;
            let physical_slot = [2, 0, 3][logical_page];
            pool.append_at_offsets_on_stream(
                physical_slot,
                token % SM12X_KV_PAGE_TOKENS,
                &key,
                token * kv_width,
                &value,
                token * kv_width,
                &stream,
            )
            .expect("paged append");
            let cache_len = token + 1;
            if !checkpoints.contains(&cache_len) {
                continue;
            }
            let mut contiguous_output =
                DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("contiguous output");
            workspace
                .attention_into_on_stream(&contiguous, &query, contiguous_output.output(), &stream)
                .expect("contiguous attention");
            let mut paged_output = DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("paged output");
            workspace
                .attention_paged_offsets_into_on_stream(
                    &pool,
                    &page_table,
                    cache_len,
                    &query,
                    0,
                    paged_output.output(),
                    0,
                    &stream,
                )
                .expect("paged attention");
            let selected_blocks =
                DeviceBuffer::from_host(&vec![1u8; CAPACITY.div_ceil(4)]).expect("selected blocks");
            let selected_tiles =
                DeviceBuffer::from_host(&vec![1u8; CAPACITY.div_ceil(64)]).expect("selected tiles");
            let mut sparse_output =
                DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("sparse output");
            workspace
                .attention_paged_sparse_offsets_into_on_stream(
                    &pool,
                    &page_table,
                    cache_len,
                    &selected_blocks,
                    &selected_tiles,
                    cache_len,
                    &query,
                    0,
                    sparse_output.output(),
                    0,
                    &stream,
                )
                .expect("sparse attention");
            assert_eq!(
                paged_output
                    .copy_to_host(&stream)
                    .expect("paged output read"),
                contiguous_output
                    .copy_to_host(&stream)
                    .expect("contiguous output read"),
                "cache_len={cache_len}"
            );
            assert_eq!(
                sparse_output
                    .copy_to_host(&stream)
                    .expect("sparse output read"),
                paged_output
                    .copy_to_host(&stream)
                    .expect("paged output read"),
                "sparse cache_len={cache_len}"
            );
            let window_start = cache_len.saturating_sub(17);
            let mut contiguous_window =
                DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("contiguous window output");
            workspace
                .attention_window_into_on_stream(
                    &contiguous,
                    &query,
                    contiguous_window.output(),
                    window_start,
                    &stream,
                )
                .expect("contiguous window attention");
            let mut paged_window =
                DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("paged window output");
            workspace
                .attention_paged_window_offsets_into_on_stream(
                    &pool,
                    &page_table,
                    cache_len,
                    &query,
                    0,
                    paged_window.output(),
                    0,
                    window_start,
                    &stream,
                )
                .expect("paged window attention");
            assert_eq!(
                paged_window
                    .copy_to_host(&stream)
                    .expect("paged window output read"),
                contiguous_window
                    .copy_to_host(&stream)
                    .expect("contiguous window output read"),
                "windowed cache_len={cache_len} window_start={window_start}"
            );

            let sparse_window_start = cache_len.saturating_sub(65) / 4 * 4;
            let mut sparse_blocks = vec![0u8; CAPACITY.div_ceil(4)];
            let mut sparse_tiles = vec![0u8; CAPACITY.div_ceil(64)];
            for block in sparse_window_start / 4..cache_len.div_ceil(4) {
                sparse_blocks[block] = 1;
                sparse_tiles[(block * 4) / 64] = 1;
            }
            let sparse_blocks =
                DeviceBuffer::from_host(&sparse_blocks).expect("sparse window blocks");
            let sparse_tiles = DeviceBuffer::from_host(&sparse_tiles).expect("sparse window tiles");
            let mut paged_sparse_window =
                DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("paged sparse window output");
            workspace
                .attention_paged_sparse_offsets_into_on_stream(
                    &pool,
                    &page_table,
                    cache_len,
                    &sparse_blocks,
                    &sparse_tiles,
                    cache_len - sparse_window_start,
                    &query,
                    0,
                    paged_sparse_window.output(),
                    0,
                    &stream,
                )
                .expect("paged sparse window attention");
            let mut paged_aligned_window =
                DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("paged aligned window output");
            workspace
                .attention_paged_window_offsets_into_on_stream(
                    &pool,
                    &page_table,
                    cache_len,
                    &query,
                    0,
                    paged_aligned_window.output(),
                    0,
                    sparse_window_start,
                    &stream,
                )
                .expect("paged aligned window attention");
            assert_eq!(
                paged_sparse_window
                    .copy_to_host(&stream)
                    .expect("paged sparse window output read"),
                paged_aligned_window
                    .copy_to_host(&stream)
                    .expect("paged aligned window output read"),
                "sparse window cache_len={cache_len} window_start={sparse_window_start}"
            );
        }

        pool.copy_page_on_stream(2, 1, &stream)
            .expect("copy physical page");
        let source_table = DeviceBuffer::from_host(&[2_u32]).expect("source table");
        let copied_table = DeviceBuffer::from_host(&[1_u32]).expect("copied table");
        let mut source_output =
            DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("source page output");
        workspace
            .attention_paged_offsets_into_on_stream(
                &pool,
                &source_table,
                SM12X_KV_PAGE_TOKENS,
                &query,
                0,
                source_output.output(),
                0,
                &stream,
            )
            .expect("source page attention");
        let mut copied_output =
            DeviceBuffer::zeroed(Q_HEADS * HEAD_DIM).expect("copied page output");
        workspace
            .attention_paged_offsets_into_on_stream(
                &pool,
                &copied_table,
                SM12X_KV_PAGE_TOKENS,
                &query,
                0,
                copied_output.output(),
                0,
                &stream,
            )
            .expect("copied page attention");
        assert_eq!(
            copied_output
                .copy_to_host(&stream)
                .expect("copied output read"),
            source_output
                .copy_to_host(&stream)
                .expect("source output read")
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
    fn speculative_tail_snapshot_restores_rows_overwritten_by_wraparound() {
        const MAX_TOKENS: usize = 32;
        const PREFIX: usize = 3;
        const SPECULATIVE: usize = 16;
        const REPLACEMENT: usize = 13;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 64;
        let width = KV_HEADS * HEAD_DIM;
        let rows = |count: usize, seed: usize| {
            DeviceBuffer::from_host(
                &(0..count * width)
                    .map(|index| ((index * 19 + seed) % 251) as f32 / 96.0 - 1.25)
                    .collect::<Vec<_>>(),
            )
            .expect("rows")
        };
        let prefix_key = rows(PREFIX, 3);
        let prefix_value = rows(PREFIX, 7);
        let speculative_key = rows(SPECULATIVE, 11);
        let speculative_value = rows(SPECULATIVE, 13);
        let replacement_key = rows(REPLACEMENT, 17);
        let replacement_value = rows(REPLACEMENT, 23);
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut rolled = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("rolled cache");
        rolled
            .append_rows_at_offset_on_stream(&prefix_key, &prefix_value, 0, PREFIX, &stream)
            .expect("prefix append");
        let mut snapshot = Sm12xKvTailSnapshot::new(KV_HEADS, HEAD_DIM).expect("snapshot");
        rolled
            .snapshot_tail_on_stream(&mut snapshot, &stream)
            .expect("snapshot tail");
        rolled
            .append_rows_at_offset_on_stream(
                &speculative_key,
                &speculative_value,
                0,
                SPECULATIVE,
                &stream,
            )
            .expect("speculative append");
        rolled
            .restore_tail_prefix_on_stream(&snapshot, PREFIX, &stream)
            .expect("restore prefix");
        rolled.truncate(PREFIX).expect("truncate");
        rolled
            .append_rows_at_offset_on_stream(
                &replacement_key,
                &replacement_value,
                0,
                REPLACEMENT,
                &stream,
            )
            .expect("replacement append");

        let mut reference =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("reference cache");
        reference
            .append_rows_at_offset_on_stream(&prefix_key, &prefix_value, 0, PREFIX, &stream)
            .expect("reference prefix");
        reference
            .append_rows_at_offset_on_stream(
                &replacement_key,
                &replacement_value,
                0,
                REPLACEMENT,
                &stream,
            )
            .expect("reference replacement");
        stream.synchronize().expect("sync");

        assert_eq!(rolled.len(), reference.len());
        assert_eq!(
            rolled.key_values_to_host(&stream).expect("rolled K"),
            reference.key_values_to_host(&stream).expect("reference K")
        );
        assert_eq!(
            rolled.key_scales_to_host(&stream).expect("rolled K scales"),
            reference
                .key_scales_to_host(&stream)
                .expect("reference K scales")
        );
        assert_eq!(
            rolled.value_values_to_host(&stream).expect("rolled V"),
            reference
                .value_values_to_host(&stream)
                .expect("reference V")
        );
        assert_eq!(
            rolled
                .value_scales_to_host(&stream)
                .expect("rolled V scales"),
            reference
                .value_scales_to_host(&stream)
                .expect("reference V scales")
        );
    }

    #[test]
    fn paged_speculative_tail_snapshot_restores_muse_shaped_attention() {
        const PREFIX: usize = 3;
        const SPECULATIVE: usize = 16;
        const REPLACEMENT: usize = 13;
        const KV_HEADS: usize = 2;
        const Q_HEADS: usize = 32;
        const HEAD_DIM: usize = 128;
        const SLOT: usize = 2;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let host_rows = |count: usize, seed: usize| {
            (0..count * kv_width)
                .map(|index| ((index * 19 + seed) % 251) as f32 / 96.0 - 1.25)
                .collect::<Vec<_>>()
        };
        let prefix_key = host_rows(PREFIX, 3);
        let prefix_value = host_rows(PREFIX, 7);
        let speculative_key = host_rows(SPECULATIVE, 11);
        let speculative_value = host_rows(SPECULATIVE, 13);
        let replacement_key = host_rows(REPLACEMENT, 17);
        let replacement_value = host_rows(REPLACEMENT, 23);
        let prefix_key_device = DeviceBuffer::from_host(&prefix_key).expect("prefix K");
        let prefix_value_device = DeviceBuffer::from_host(&prefix_value).expect("prefix V");
        let speculative_key_device =
            DeviceBuffer::from_host(&speculative_key).expect("speculative K");
        let speculative_value_device =
            DeviceBuffer::from_host(&speculative_value).expect("speculative V");
        let replacement_key_device =
            DeviceBuffer::from_host(&replacement_key).expect("replacement K");
        let replacement_value_device =
            DeviceBuffer::from_host(&replacement_value).expect("replacement V");
        let query = DeviceBuffer::from_host(
            &(0..q_width)
                .map(|index| ((index * 31 + 5) % 263) as f32 / 128.0 - 1.0)
                .collect::<Vec<_>>(),
        )
        .expect("query");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut pool = Sm12xKvPagePool::new(3, KV_HEADS, HEAD_DIM).expect("page pool");
        pool.append_rows_at_offset_on_stream(
            SLOT,
            0,
            &prefix_key_device,
            &prefix_value_device,
            0,
            PREFIX,
            &stream,
        )
        .expect("prefix append");
        let mut snapshot = Sm12xKvTailSnapshot::new(KV_HEADS, HEAD_DIM).expect("snapshot");
        pool.snapshot_tail_on_stream(SLOT, &mut snapshot, &stream)
            .expect("snapshot tail");
        pool.append_rows_at_offset_on_stream(
            SLOT,
            PREFIX,
            &speculative_key_device,
            &speculative_value_device,
            0,
            SPECULATIVE,
            &stream,
        )
        .expect("speculative append");
        pool.restore_tail_prefix_on_stream(SLOT, &snapshot, PREFIX, &stream)
            .expect("restore prefix");
        pool.append_rows_at_offset_on_stream(
            SLOT,
            PREFIX,
            &replacement_key_device,
            &replacement_value_device,
            0,
            REPLACEMENT,
            &stream,
        )
        .expect("replacement append");

        let mut reference_key = prefix_key;
        reference_key.extend_from_slice(&replacement_key);
        let mut reference_value = prefix_value;
        reference_value.extend_from_slice(&replacement_value);
        let reference_key = DeviceBuffer::from_host(&reference_key).expect("reference K");
        let reference_value = DeviceBuffer::from_host(&reference_value).expect("reference V");
        let mut reference =
            Sm12xKvCache::new(SM12X_KV_PAGE_TOKENS, KV_HEADS, HEAD_DIM).expect("reference cache");
        reference
            .append_rows_at_offset_on_stream(
                &reference_key,
                &reference_value,
                0,
                PREFIX + REPLACEMENT,
                &stream,
            )
            .expect("reference append");

        let page_table = DeviceBuffer::from_host(&[SLOT as u32]).expect("page table");
        let mut workspace =
            Sm12xKvAttentionWorkspace::new_gqa(SM12X_KV_PAGE_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("workspace");
        let mut paged_output = DeviceBuffer::zeroed(q_width).expect("paged output");
        workspace
            .attention_paged_offsets_into_on_stream(
                &pool,
                &page_table,
                PREFIX + REPLACEMENT,
                &query,
                0,
                paged_output.output(),
                0,
                &stream,
            )
            .expect("paged attention");
        let mut reference_output = DeviceBuffer::zeroed(q_width).expect("reference output");
        workspace
            .attention_into_on_stream(&reference, &query, reference_output.output(), &stream)
            .expect("reference attention");
        assert_eq!(
            paged_output.copy_to_host(&stream).expect("paged read"),
            reference_output
                .copy_to_host(&stream)
                .expect("reference read")
        );
    }

    #[test]
    fn paged_causal_rows_match_repeated_muse_shaped_attention_across_pages() {
        const PREFIX: usize = 123;
        const ROWS: usize = 16;
        const CAPACITY: usize = 2 * SM12X_KV_PAGE_TOKENS;
        const WINDOW: usize = 64;
        const KV_HEADS: usize = 2;
        const Q_HEADS: usize = 32;
        const HEAD_DIM: usize = 128;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let prefix_key = DeviceBuffer::from_host(
            &(0..PREFIX * kv_width)
                .map(|index| ((index * 17 + 3) % 251) as f32 / 96.0 - 1.25)
                .collect::<Vec<_>>(),
        )
        .expect("prefix K");
        let prefix_value = DeviceBuffer::from_host(
            &(0..PREFIX * kv_width)
                .map(|index| ((index * 29 + 7) % 257) as f32 / 112.0 - 1.0)
                .collect::<Vec<_>>(),
        )
        .expect("prefix V");
        let key = DeviceBuffer::from_host(
            &(0..ROWS * kv_width)
                .map(|index| ((index * 19 + 11) % 263) as f32 / 128.0 - 0.75)
                .collect::<Vec<_>>(),
        )
        .expect("K rows");
        let value = DeviceBuffer::from_host(
            &(0..ROWS * kv_width)
                .map(|index| ((index * 31 + 13) % 269) as f32 / 144.0 - 0.625)
                .collect::<Vec<_>>(),
        )
        .expect("V rows");
        let query = DeviceBuffer::from_host(
            &(0..ROWS * q_width)
                .map(|index| ((index * 37 + 5) % 271) as f32 / 160.0 - 0.75)
                .collect::<Vec<_>>(),
        )
        .expect("query rows");
        let page_table = DeviceBuffer::from_host(&[2_u32, 0]).expect("page table");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut batched_pool = Sm12xKvPagePool::new(3, KV_HEADS, HEAD_DIM).expect("batched pool");
        let mut repeated_pool = Sm12xKvPagePool::new(3, KV_HEADS, HEAD_DIM).expect("repeated pool");
        batched_pool
            .append_rows_at_offset_on_stream(2, 0, &prefix_key, &prefix_value, 0, PREFIX, &stream)
            .expect("batched prefix");
        repeated_pool
            .append_rows_at_offset_on_stream(2, 0, &prefix_key, &prefix_value, 0, PREFIX, &stream)
            .expect("repeated prefix");

        let mut batched_workspace =
            Sm12xKvAttentionWorkspace::new_gqa_batched(CAPACITY, Q_HEADS, KV_HEADS, HEAD_DIM, ROWS)
                .expect("batched workspace");
        let mut batched_output = DeviceBuffer::zeroed(ROWS * q_width).expect("batched output");
        let mut processed = 0;
        while processed < ROWS {
            let position = PREFIX + processed;
            let local_position = position % SM12X_KV_PAGE_TOKENS;
            let rows = (ROWS - processed)
                .min(SM12X_KV_PAGE_TOKENS - local_position)
                .min(V_TOKEN_BLOCK - position % V_TOKEN_BLOCK);
            let slot = [2, 0][position / SM12X_KV_PAGE_TOKENS];
            batched_pool
                .append_rows_at_offset_on_stream(
                    slot,
                    local_position,
                    &key,
                    &value,
                    processed,
                    rows,
                    &stream,
                )
                .expect("batched append");
            batched_workspace
                .attention_paged_causal_rows_at_offset_into_on_stream(
                    &batched_pool,
                    &page_table,
                    position,
                    &query,
                    processed,
                    rows,
                    Some(WINDOW),
                    batched_output.output(),
                    &stream,
                )
                .expect("batched causal attention");
            processed += rows;
        }

        let mut repeated_workspace =
            Sm12xKvAttentionWorkspace::new_gqa(CAPACITY, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("repeated workspace");
        let mut repeated_output = DeviceBuffer::zeroed(ROWS * q_width).expect("repeated output");
        for row in 0..ROWS {
            let position = PREFIX + row;
            let slot = [2, 0][position / SM12X_KV_PAGE_TOKENS];
            repeated_pool
                .append_at_offsets_on_stream(
                    slot,
                    position % SM12X_KV_PAGE_TOKENS,
                    &key,
                    row * kv_width,
                    &value,
                    row * kv_width,
                    &stream,
                )
                .expect("repeated append");
            let cache_len = position + 1;
            repeated_workspace
                .attention_paged_window_offsets_into_on_stream(
                    &repeated_pool,
                    &page_table,
                    cache_len,
                    &query,
                    row * q_width,
                    repeated_output.output(),
                    row * q_width,
                    cache_len.saturating_sub(WINDOW),
                    &stream,
                )
                .expect("repeated attention");
        }
        assert_eq!(
            batched_output.copy_to_host(&stream).expect("batched read"),
            repeated_output
                .copy_to_host(&stream)
                .expect("repeated read")
        );
    }

    #[test]
    fn fixed_window_row_attention_tracks_independent_full_cache_attention() {
        const TOKENS: usize = 6;
        const ROWS: usize = 4;
        const KV_HEADS: usize = 1;
        const Q_HEADS: usize = 8;
        const HEAD_DIM: usize = 64;
        let kv_width = KV_HEADS * HEAD_DIM;
        let q_width = Q_HEADS * HEAD_DIM;
        let key_host = (0..TOKENS * kv_width)
            .map(|index| ((index * 19 % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * kv_width)
            .map(|index| ((index * 31 % 257) as f32 - 128.0) / 320.0)
            .collect::<Vec<_>>();
        let query_host = (0..ROWS * q_width)
            .map(|index| ((index * 17 % 263) as f32 - 131.0) / 288.0)
            .collect::<Vec<_>>();
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let value = DeviceBuffer::from_host(&value_host).expect("value");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut cache = Sm12xKvCache::new(TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        cache
            .append_rows_at_offset_on_stream(&key, &value, 0, TOKENS, &stream)
            .expect("cache append");

        let mut actual = DeviceBuffer::zeroed(ROWS * q_width).expect("actual");
        let mut workspace =
            Sm12xKvAttentionWorkspace::new_gqa_batched(TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, ROWS)
                .expect("workspace");
        workspace
            .attention_rows_window_at_offset_into_on_stream(
                &cache,
                &query,
                0,
                ROWS,
                0,
                actual.output(),
                0,
                &stream,
            )
            .expect("fixed-window attention");

        let key = DeviceBuffer::from_host(&key_host).expect("reference key");
        let value = DeviceBuffer::from_host(&value_host).expect("reference value");
        let mut expected_host = Vec::with_capacity(ROWS * q_width);
        for row in 0..ROWS {
            let row_query =
                DeviceBuffer::from_host(&query_host[row * q_width..(row + 1) * q_width])
                    .expect("row query");
            let mut row_output = DeviceBuffer::zeroed(q_width).expect("row output");
            crate::cached_gqa_attention_f32_into_on_stream(
                &row_query,
                &key,
                &value,
                row_output.output(),
                TOKENS,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                &stream,
            )
            .expect("reference attention");
            expected_host.extend(
                row_output
                    .copy_to_host(&stream)
                    .expect("reference read")
                    .iter()
                    .copied(),
            );
        }
        let actual = actual.copy_to_host(&stream).expect("actual read");
        let max_abs = actual
            .iter()
            .zip(&expected_host)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 0.25, "fixed-window max_abs={max_abs}");
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
    fn indexed_compact_append_and_attention_match_host_positions() {
        const MAX_TOKENS: usize = 64;
        const TOKENS: usize = 17;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const Q_HEADS: usize = KV_HEADS * MMA_N;
        let kv_width = KV_HEADS * HEAD_DIM;
        let query_width = Q_HEADS * HEAD_DIM;
        let key_host = (0..TOKENS * kv_width)
            .map(|index| ((index * 31 % 251) as f32 - 125.0) / 512.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * kv_width)
            .map(|index| ((index * 47 % 257) as f32 - 128.0) / 384.0)
            .collect::<Vec<_>>();
        let query_host = (0..query_width)
            .map(|index| ((index * 19 % 239) as f32 - 119.0) / 448.0)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut host_cache = Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("host cache");
        let mut indexed_cache =
            Sm12xKvCache::new(MAX_TOKENS, KV_HEADS, HEAD_DIM).expect("indexed cache");
        for token in 0..TOKENS {
            let key = DeviceBuffer::from_host(&key_host[token * kv_width..(token + 1) * kv_width])
                .expect("key row");
            let value =
                DeviceBuffer::from_host(&value_host[token * kv_width..(token + 1) * kv_width])
                    .expect("value row");
            host_cache
                .append_at_on_stream(&key, &value, token, &stream)
                .expect("host-position append");
            let position = DeviceBuffer::from_host(&[token as u32]).expect("position");
            indexed_cache
                .append_indexed_on_stream(&key, &value, &position, &stream)
                .expect("indexed append");
        }

        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let cache_len = DeviceBuffer::from_host(&[TOKENS as u32]).expect("cache length");
        let mut expected = DeviceBuffer::zeroed(query_width).expect("expected");
        let mut actual = DeviceBuffer::zeroed(query_width).expect("actual");
        let mut host_workspace =
            Sm12xKvAttentionWorkspace::new_gqa(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("host workspace");
        let mut indexed_workspace =
            Sm12xKvAttentionWorkspace::new_gqa(MAX_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM)
                .expect("indexed workspace");
        host_workspace
            .attention_into_on_stream(&host_cache, &query, expected.output(), &stream)
            .expect("host-position attention");
        indexed_workspace
            .attention_indexed_into_on_stream(
                &indexed_cache,
                &query,
                &cache_len,
                actual.output(),
                &stream,
            )
            .expect("indexed attention");

        let expected = expected.copy_to_host(&stream).expect("expected copy");
        let actual = actual.copy_to_host(&stream).expect("actual copy");
        let max_abs = expected
            .iter()
            .zip(actual.iter())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0e-6, "indexed attention max_abs={max_abs}");
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

    #[test]
    fn paged_compact_cache_unpack_matches_contiguous_across_page_boundary() {
        const TOKENS: usize = SM12X_KV_PAGE_TOKENS + 13;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 128;
        let width = KV_HEADS * HEAD_DIM;
        let key = DeviceBuffer::from_host(
            &(0..TOKENS * width)
                .map(|index| ((index * 19 + 3) % 251) as f32 / 96.0 - 1.25)
                .collect::<Vec<_>>(),
        )
        .expect("keys");
        let value = DeviceBuffer::from_host(
            &(0..TOKENS * width)
                .map(|index| ((index * 29 + 7) % 257) as f32 / 112.0 - 1.0)
                .collect::<Vec<_>>(),
        )
        .expect("values");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut contiguous = Sm12xKvCache::new(TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        contiguous
            .append_rows_at_offset_on_stream(&key, &value, 0, TOKENS, &stream)
            .expect("contiguous append");
        let mut pool = Sm12xKvPagePool::new(4, KV_HEADS, HEAD_DIM).expect("pool");
        pool.append_rows_at_offset_on_stream(3, 0, &key, &value, 0, SM12X_KV_PAGE_TOKENS, &stream)
            .expect("first paged append");
        pool.append_rows_at_offset_on_stream(
            1,
            0,
            &key,
            &value,
            SM12X_KV_PAGE_TOKENS,
            TOKENS - SM12X_KV_PAGE_TOKENS,
            &stream,
        )
        .expect("second paged append");
        let page_table = DeviceBuffer::from_host(&[3_u32, 1]).expect("page table");
        let values = TOKENS * width;
        let mut contiguous_key = DeviceBuffer::zeroed(values).expect("contiguous K output");
        let mut contiguous_value = DeviceBuffer::zeroed(values).expect("contiguous V output");
        contiguous
            .unpack_bf16_on_stream(contiguous_key.output(), contiguous_value.output(), &stream)
            .expect("contiguous unpack");
        let mut paged_key = DeviceBuffer::zeroed(values).expect("paged K output");
        let mut paged_value = DeviceBuffer::zeroed(values).expect("paged V output");
        pool.unpack_paged_bf16_on_stream(
            &page_table,
            TOKENS,
            paged_key.output(),
            paged_value.output(),
            &stream,
        )
        .expect("paged unpack");

        assert_eq!(
            paged_key.copy_to_host(&stream).expect("paged K read"),
            contiguous_key
                .copy_to_host(&stream)
                .expect("contiguous K read")
        );
        assert_eq!(
            paged_value.copy_to_host(&stream).expect("paged V read"),
            contiguous_value
                .copy_to_host(&stream)
                .expect("contiguous V read")
        );
    }
}
