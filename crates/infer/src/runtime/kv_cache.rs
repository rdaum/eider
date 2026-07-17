//! Device-resident KV cache storage for decode.

use nvfp4::{
    CudaStream, DeviceBuffer, DeviceOutput, Error, Result, append_rows_f32_indexed_into_on_stream,
    append_rows_f32_into_on_stream, cached_gqa_attention_f32_indexed_into_on_stream,
    cached_gqa_attention_f32_into_on_stream, prefill_gqa_attention_f32_into,
};

/// Device-resident K/V cache for one sequence across all transformer layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    /// Allocates identical per-layer K/V caches for one sequence.
    pub fn new(
        n_layers: usize,
        max_tokens: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if n_layers == 0 {
            return Err(Error::Shape {
                label: "KV cache",
                expected: "at least one layer".to_string(),
                actual: "0 layers".to_string(),
            });
        }
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKvCache::new(max_tokens, kv_heads, head_dim)?);
        }
        Ok(Self { layers })
    }

    /// Returns the number of layer caches.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the current valid token count for this sequence.
    pub fn len(&self) -> usize {
        self.layers[0].len()
    }

    /// Returns true when the sequence cache has no valid token rows.
    pub fn is_empty(&self) -> bool {
        self.layers[0].is_empty()
    }

    /// Returns an immutable layer cache by index.
    pub fn layer(&self, index: usize) -> Result<&LayerKvCache> {
        self.layers.get(index).ok_or_else(|| Error::Shape {
            label: "KV cache layer",
            expected: format!("index < {}", self.layers.len()),
            actual: format!("index={index}"),
        })
    }

    /// Returns a mutable layer cache by index.
    pub fn layer_mut(&mut self, index: usize) -> Result<&mut LayerKvCache> {
        let n_layers = self.layers.len();
        self.layers.get_mut(index).ok_or_else(|| Error::Shape {
            label: "KV cache layer",
            expected: format!("index < {n_layers}"),
            actual: format!("index={index}"),
        })
    }

    /// Advances all layer cache lengths by `rows` after externally enqueued
    /// appends have completed.
    pub fn advance_all(&mut self, rows: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.advance_len(rows)?;
        }
        Ok(())
    }
}

/// Device-resident K/V cache for one transformer layer.
///
/// Rows are token positions and columns are `kv_heads * head_dim`, matching the
/// post-QK-norm, post-RoPE K layout and the V layout consumed by attention.
pub struct LayerKvCache {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    max_tokens: usize,
    len: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl LayerKvCache {
    /// Allocates a zero-initialized cache for one layer.
    pub fn new(max_tokens: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        if max_tokens == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(Error::Shape {
                label: "layer KV cache",
                expected: "non-zero max_tokens, kv_heads, and head_dim".to_string(),
                actual: format!("max_tokens={max_tokens} kv_heads={kv_heads} head_dim={head_dim}"),
            });
        }
        let width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
            label: "layer KV cache width",
            expected: "kv_heads * head_dim without overflow".to_string(),
            actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
        })?;
        let len = max_tokens.checked_mul(width).ok_or_else(|| Error::Shape {
            label: "layer KV cache allocation",
            expected: "max_tokens * kv_width without overflow".to_string(),
            actual: format!("max_tokens={max_tokens} kv_width={width}"),
        })?;
        Ok(Self {
            key: DeviceBuffer::zeroed(len)?,
            value: DeviceBuffer::zeroed(len)?,
            max_tokens,
            len: 0,
            kv_heads,
            head_dim,
        })
    }

    /// Returns the number of valid token rows currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when the cache has no valid token rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the maximum number of token rows this cache can hold.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns `kv_heads * head_dim`.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Returns bytes owned by this layer's device-resident key/value storage.
    pub fn device_bytes(&self) -> usize {
        self.key.device_bytes() + self.value.device_bytes()
    }

    /// Appends one or more contiguous K/V rows and advances the valid length.
    pub fn append(&mut self, key: &DeviceBuffer<f32>, value: &DeviceBuffer<f32>) -> Result<()> {
        let width = self.kv_width();
        if key.len() != value.len() || !key.len().is_multiple_of(width) {
            return Err(Error::Shape {
                label: "layer KV append",
                expected: format!("matching multiples of {width} values"),
                actual: format!("key={} value={}", key.len(), value.len()),
            });
        }
        let rows = key.len() / width;
        if self.len + rows > self.max_tokens {
            return Err(Error::Shape {
                label: "layer KV append capacity",
                expected: format!("at most {} rows", self.max_tokens - self.len),
                actual: format!("{rows} rows"),
            });
        }

        let stream = CudaStream::new_blocking()?;
        append_rows_f32_into_on_stream(key, self.key.output(), self.len, rows, width, &stream)?;
        append_rows_f32_into_on_stream(value, self.value.output(), self.len, rows, width, &stream)?;
        stream.synchronize()?;
        self.len += rows;
        Ok(())
    }

    /// Appends K/V rows on `stream` and advances the valid length.
    pub fn append_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_width();
        if key.len() != value.len() || !key.len().is_multiple_of(width) {
            return Err(Error::Shape {
                label: "layer KV append",
                expected: format!("matching multiples of {width} values"),
                actual: format!("key={} value={}", key.len(), value.len()),
            });
        }
        let rows = key.len() / width;
        if self.len + rows > self.max_tokens {
            return Err(Error::Shape {
                label: "layer KV append capacity",
                expected: format!("at most {} rows", self.max_tokens - self.len),
                actual: format!("{rows} rows"),
            });
        }

        append_rows_f32_into_on_stream(key, self.key.output(), self.len, rows, width, stream)?;
        append_rows_f32_into_on_stream(value, self.value.output(), self.len, rows, width, stream)?;
        self.len += rows;
        Ok(())
    }

    /// Enqueues K/V append using a device-resident destination row.
    ///
    /// This does not advance `len`; call [`LayerKvCache::advance_len`] after
    /// the queued work has completed, or [`KvCache::advance_all`] when all
    /// layers append one row through a captured graph.
    pub fn append_indexed_on_stream(
        &mut self,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        dst_start_row: &DeviceBuffer<u32>,
        max_start_row: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let width = self.kv_width();
        if key.len() != value.len() || !key.len().is_multiple_of(width) {
            return Err(Error::Shape {
                label: "layer KV indexed append",
                expected: format!("matching multiples of {width} values"),
                actual: format!("key={} value={}", key.len(), value.len()),
            });
        }
        let rows = key.len() / width;
        if max_start_row + rows > self.max_tokens {
            return Err(Error::Shape {
                label: "layer KV indexed append capacity",
                expected: format!("start + rows <= {}", self.max_tokens),
                actual: format!("start={max_start_row} rows={rows}"),
            });
        }

        append_rows_f32_indexed_into_on_stream(
            key,
            self.key.output(),
            dst_start_row,
            max_start_row,
            rows,
            width,
            stream,
        )?;
        append_rows_f32_indexed_into_on_stream(
            value,
            self.value.output(),
            dst_start_row,
            max_start_row,
            rows,
            width,
            stream,
        )
    }

    /// Advances the valid length after externally enqueued appends complete.
    pub fn advance_len(&mut self, rows: usize) -> Result<()> {
        if self.len + rows > self.max_tokens {
            return Err(Error::Shape {
                label: "layer KV advance",
                expected: format!("at most {} rows", self.max_tokens - self.len),
                actual: format!("{rows} rows"),
            });
        }
        self.len += rows;
        Ok(())
    }

    /// Enqueues one-token grouped-query attention over all valid cached rows
    /// into `output` on `stream`.
    pub fn decode_attention_into_on_stream(
        &self,
        query: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        q_heads: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.len == 0 {
            return Err(Error::Shape {
                label: "layer KV cached attention",
                expected: "at least one cached row".to_string(),
                actual: "0 cached rows".to_string(),
            });
        }
        cached_gqa_attention_f32_into_on_stream(
            query,
            &self.key,
            &self.value,
            output,
            self.len,
            q_heads,
            self.kv_heads,
            self.head_dim,
            stream,
        )
    }

    /// Enqueues one-token grouped-query attention using device-resident cache
    /// length.
    pub fn decode_attention_indexed_into_on_stream(
        &self,
        query: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        cache_len: &DeviceBuffer<u32>,
        max_cache_len: usize,
        q_heads: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if max_cache_len == 0 || max_cache_len > self.max_tokens {
            return Err(Error::Shape {
                label: "layer KV indexed attention length",
                expected: format!("1..={}", self.max_tokens),
                actual: max_cache_len.to_string(),
            });
        }
        cached_gqa_attention_f32_indexed_into_on_stream(
            query,
            &self.key,
            &self.value,
            output,
            cache_len,
            max_cache_len,
            q_heads,
            self.kv_heads,
            self.head_dim,
            stream,
        )
    }

    /// Runs causal grouped-query attention for a contiguous prefill chunk.
    pub fn prefill_attention_into(
        &self,
        query: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        tokens: usize,
        start_position: usize,
        q_heads: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Err(Error::Shape {
                label: "layer KV prefill attention",
                expected: "at least one query token".to_string(),
                actual: "0 query tokens".to_string(),
            });
        }
        if self.len < start_position + tokens {
            return Err(Error::Shape {
                label: "layer KV prefill attention cache",
                expected: format!("at least {} cached rows", start_position + tokens),
                actual: format!("{} cached rows", self.len),
            });
        }
        prefill_gqa_attention_f32_into(
            query,
            &self.key,
            &self.value,
            output.output(),
            tokens,
            start_position,
            q_heads,
            self.kv_heads,
            self.head_dim,
        )
    }
}
