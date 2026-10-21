//! Causal attention with learned per-head sinks and bounded BF16 KV storage.

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceAddress, DeviceBuffer, Error, Result};
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_paged_sink_decode_batch(
        q: *const f32,
        keys: *const DeviceAddress<u16>,
        values: *const DeviceAddress<u16>,
        pages: *const DeviceAddress<DeviceAddress<u16>>,
        positions: *const u32,
        k: *const f32,
        v: *const f32,
        sinks: *const f32,
        partial: *mut f32,
        out: *mut f32,
        rows: u32,
        qheads: u32,
        kvheads: u32,
        capacity: u32,
        splits: u32,
        layer_offset: u32,
        layers: u32,
        layer: u32,
        local: bool,
        stream: *mut c_void,
    ) -> i32;
    fn infer_paged_sink_attention(
        q: *const f32,
        ck: *mut u16,
        cv: *mut u16,
        pages: *const DeviceAddress<u16>,
        k: *const f32,
        v: *const f32,
        sinks: *const f32,
        partial: *mut f32,
        out: *mut f32,
        end: u32,
        position: *const u32,
        rows: u32,
        qheads: u32,
        kvheads: u32,
        capacity: u32,
        splits: u32,
        layer_offset: u32,
        page_tokens: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_causal_sink_attention(
        q: *const f32,
        ck: *mut u16,
        cv: *mut u16,
        k: *const f32,
        v: *const f32,
        sinks: *const f32,
        partial: *mut f32,
        out: *mut f32,
        end: u32,
        rows: u32,
        qheads: u32,
        kvheads: u32,
        capacity: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// BF16 KV ring. Capacity is the local window or the global context limit.
pub struct SinkAttentionCache {
    key: DeviceBuffer<u16>,
    value: DeviceBuffer<u16>,
    capacity: usize,
    kv_heads: usize,
    position: usize,
}

impl SinkAttentionCache {
    /// Allocates storage for 128-dimensional heads.
    pub fn new(capacity: usize, kv_heads: usize) -> Result<Self> {
        let len = capacity
            .checked_mul(kv_heads)
            .and_then(|v| v.checked_mul(128))
            .filter(|_| {
                capacity > 0
                    && capacity <= u32::MAX as usize
                    && kv_heads > 0
                    && kv_heads <= u32::MAX as usize
            })
            .ok_or_else(|| invalid("invalid KV capacity"))?;
        Ok(Self {
            key: DeviceBuffer::zeroed(len)?,
            value: DeviceBuffer::zeroed(len)?,
            capacity,
            kv_heads,
            position: 0,
        })
    }
    /// Number of tokens appended, including evicted local tokens.
    pub fn position(&self) -> usize {
        self.position
    }
    /// Records the committed position after a captured append is replayed.
    pub fn record_replay_position(&mut self, position: usize) -> Result<()> {
        if position > u32::MAX as usize {
            return Err(invalid("position overflow"));
        }
        self.position = position;
        Ok(())
    }
    /// Storage owned by the KV ring.
    pub fn device_bytes(&self) -> usize {
        self.key.device_bytes() + self.value.device_bytes()
    }
    /// Copies bounded local history on the caller's stream.
    pub fn snapshot(&self, stream: &CudaStream) -> Result<Self> {
        let mut snapshot = Self::new(self.capacity, self.kv_heads)?;
        snapshot.restore(self, stream)?;
        Ok(snapshot)
    }
    /// Restores a local ring with identical geometry.
    pub fn restore(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        if self.capacity != source.capacity || self.kv_heads != source.kv_heads {
            return Err(invalid("local snapshot geometry mismatch"));
        }
        self.key
            .copy_prefix_from_device_on_stream(&source.key, self.key.len(), stream)?;
        self.value
            .copy_prefix_from_device_on_stream(&source.value, self.value.len(), stream)?;
        self.position = source.position;
        Ok(())
    }
}

/// Stable device tables for one decode token per independent sequence.
/// The caller keeps each referenced KV allocation alive until execution completes.
pub struct SinkAttentionDecodeBatch {
    positions: DeviceBuffer<u32>,
    keys: DeviceBuffer<DeviceAddress<u16>>,
    values: DeviceBuffer<DeviceAddress<u16>>,
    pages: DeviceBuffer<DeviceAddress<DeviceAddress<u16>>>,
    host_positions: Vec<u32>,
    host_keys: Vec<DeviceAddress<u16>>,
    host_values: Vec<DeviceAddress<u16>>,
    host_pages: Vec<DeviceAddress<DeviceAddress<u16>>>,
    host_geometry: Vec<(usize, usize)>,
    initialized: Vec<bool>,
    layers: usize,
    rows: usize,
    end_bound: usize,
}
impl SinkAttentionDecodeBatch {
    /// Allocates reusable pointer tables for the model layer count.
    pub fn new(capacity: usize, layers: usize) -> Result<Self> {
        let count = capacity
            .checked_mul(layers)
            .filter(|_| {
                capacity > 0 && layers > 0 && capacity <= 65535 && layers <= u32::MAX as usize
            })
            .ok_or_else(|| invalid("invalid decode table capacity"))?;
        Ok(Self {
            positions: DeviceBuffer::zeroed(capacity)?,
            keys: DeviceBuffer::zeroed(count)?,
            values: DeviceBuffer::zeroed(count)?,
            pages: DeviceBuffer::zeroed(capacity)?,
            host_positions: vec![0; capacity],
            host_keys: vec![DeviceAddress::null(); count],
            host_values: vec![DeviceAddress::null(); count],
            host_pages: vec![DeviceAddress::null(); capacity],
            host_geometry: vec![(0, 0); count],
            initialized: vec![false; capacity],
            layers,
            rows: 0,
            end_bound: 0,
        })
    }
    /// Sets one row after the sequence cache has reserved its writable tail.
    pub fn set_row(
        &mut self,
        row: usize,
        position: usize,
        pages: &DeviceBuffer<DeviceAddress<u16>>,
        local: &[Option<SinkAttentionCache>],
    ) -> Result<()> {
        if row >= self.host_positions.len()
            || position >= u32::MAX as usize
            || pages.len() < (position + 1).div_ceil(128)
            || local.len() != self.layers
        {
            return Err(invalid("decode row geometry mismatch"));
        }
        self.initialized[row] = true;
        self.host_positions[row] = position as u32;
        self.host_pages[row] = pages.cuda_address();
        for (layer, cache) in local.iter().enumerate() {
            let slot = row * self.layers + layer;
            self.host_geometry[slot] = cache.as_ref().map_or((0, 0), |c| (c.capacity, c.kv_heads));
            self.host_keys[slot] = cache
                .as_ref()
                .map_or(DeviceAddress::null(), |c| c.key.cuda_address());
            self.host_values[slot] = cache
                .as_ref()
                .map_or(DeviceAddress::null(), |c| c.value.cuda_address());
        }
        Ok(())
    }
    /// Uploads the selected rows. The caller synchronizes previous execution first.
    pub fn upload(&mut self, rows: usize) -> Result<()> {
        if rows == 0
            || rows > self.host_positions.len()
            || self.initialized[..rows].contains(&false)
        {
            return Err(invalid("invalid decode row count"));
        }
        self.positions
            .copy_range_from_host(0, &self.host_positions[..rows])?;
        self.keys
            .copy_range_from_host(0, &self.host_keys[..rows * self.layers])?;
        self.values
            .copy_range_from_host(0, &self.host_values[..rows * self.layers])?;
        self.pages
            .copy_range_from_host(0, &self.host_pages[..rows])?;
        self.rows = rows;
        self.end_bound = self.host_positions[..rows].iter().copied().max().unwrap() as usize + 1;
        Ok(())
    }
    /// Device positions used by the batched RoPE operation.
    pub fn positions(&self) -> &DeviceBuffer<u32> {
        &self.positions
    }
}

/// Split-attention scratch shared across model layers.
pub struct SinkAttentionWorkspace {
    partial: DeviceBuffer<f32>,
    rows: usize,
    heads: usize,
    max_context: usize,
}

impl SinkAttentionWorkspace {
    /// Evaluates a local ring or shared global pages. `position` is a stable
    /// device scalar; `end_bound` bounds the launch while graphs replay.
    /// Global page bundles use 128 tokens per page.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paged(
        &mut self,
        local: Option<&mut SinkAttentionCache>,
        pages: &DeviceBuffer<DeviceAddress<u16>>,
        layer_offset: usize,
        page_tokens: usize,
        position: &DeviceBuffer<u32>,
        end: usize,
        end_bound: usize,
        kv_heads: usize,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let next = end
            .checked_add(rows)
            .ok_or_else(|| invalid("position overflow"))?;
        let capacity = local
            .as_ref()
            .map_or(self.max_context, |cache| cache.capacity);
        if rows == 0
            || rows > self.rows
            || kv_heads == 0
            || !self.heads.is_multiple_of(kv_heads)
            || local
                .as_ref()
                .is_some_and(|cache| cache.kv_heads != kv_heads)
            || position.len() != 1
            || next > end_bound
            || end_bound > self.max_context
            || page_tokens != 128
            || pages.len() < next.div_ceil(page_tokens)
            || q.len() < rows * self.heads * 128
            || k.len() < rows * kv_heads * 128
            || v.len() < rows * kv_heads * 128
            || out.len() < rows * self.heads * 128
            || sinks.len() != self.heads
            || layer_offset > u32::MAX as usize
        {
            return Err(invalid("paged attention geometry mismatch"));
        }
        let splits = end_bound.min(capacity).div_ceil(256);
        let (ck, cv, table) = match local {
            Some(cache) => {
                cache.position = next;
                (cache.key.ptr, cache.value.ptr, std::ptr::null())
            }
            None => (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                pages.ptr.cast_const(),
            ),
        };
        unsafe {
            check_cuda(
                "infer_paged_sink_attention",
                infer_paged_sink_attention(
                    q.ptr,
                    ck,
                    cv,
                    table,
                    k.ptr,
                    v.ptr,
                    sinks.ptr,
                    self.partial.ptr,
                    out.ptr,
                    end as u32,
                    position.ptr,
                    rows as u32,
                    self.heads as u32,
                    kv_heads as u32,
                    capacity as u32,
                    splits as u32,
                    layer_offset as u32,
                    page_tokens as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
    /// Evaluates one decode token for each selected sequence using stable tables.
    #[allow(clippy::too_many_arguments)]
    pub fn run_decode_batch(
        &mut self,
        batch: &SinkAttentionDecodeBatch,
        layer: usize,
        layer_offset: usize,
        local_window: Option<usize>,
        end_bound: usize,
        kv_heads: usize,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let capacity = local_window.unwrap_or(self.max_context);
        if rows == 0
            || rows != batch.rows
            || rows > self.rows
            || layer >= batch.layers
            || capacity == 0
            || capacity > u32::MAX as usize
            || kv_heads == 0
            || !self.heads.is_multiple_of(kv_heads)
            || batch.end_bound > end_bound
            || end_bound > self.max_context
            || q.len() < rows * self.heads * 128
            || k.len() < rows * kv_heads * 128
            || v.len() < rows * kv_heads * 128
            || out.len() < rows * self.heads * 128
            || sinks.len() != self.heads
            || layer_offset > u32::MAX as usize
        {
            return Err(invalid("batched decode attention geometry mismatch"));
        }
        let geometry = local_window.map_or((0, 0), |window| (window, kv_heads));
        if (0..rows).any(|row| batch.host_geometry[row * batch.layers + layer] != geometry) {
            return Err(invalid("decode KV geometry does not match the layer"));
        }
        let splits = end_bound.min(capacity).div_ceil(256);
        unsafe {
            check_cuda(
                "infer_paged_sink_decode_batch",
                infer_paged_sink_decode_batch(
                    q.ptr,
                    batch.keys.ptr,
                    batch.values.ptr,
                    batch.pages.ptr,
                    batch.positions.ptr,
                    k.ptr,
                    v.ptr,
                    sinks.ptr,
                    self.partial.ptr,
                    out.ptr,
                    rows as u32,
                    self.heads as u32,
                    kv_heads as u32,
                    capacity as u32,
                    splits as u32,
                    layer_offset as u32,
                    batch.layers as u32,
                    layer as u32,
                    local_window.is_some(),
                    stream.as_raw(),
                ),
            )
        }
    }
    /// Allocates scratch for an explicit row and context limit.
    pub fn new(rows: usize, heads: usize, max_context: usize) -> Result<Self> {
        let len = rows
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(max_context.div_ceil(256)))
            .and_then(|v| v.checked_mul(130))
            .filter(|_| {
                rows > 0
                    && rows <= u32::MAX as usize
                    && heads > 0
                    && heads <= u32::MAX as usize
                    && max_context > 0
                    && max_context <= u32::MAX as usize
            })
            .ok_or_else(|| invalid("invalid attention workspace dimensions"))?;
        Ok(Self {
            partial: DeviceBuffer::zeroed(len)?,
            rows,
            heads,
            max_context,
        })
    }
    /// Returns scratch storage bytes.
    pub fn device_bytes(&self) -> usize {
        self.partial.device_bytes()
    }
    /// Appends a causal chunk after evaluating it against the previous KV ring.
    /// Inputs must contain BF16-valued Q/K/V in F32 storage.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        cache: &mut SinkAttentionCache,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let end = cache
            .position
            .checked_add(rows)
            .filter(|&v| v <= u32::MAX as usize)
            .ok_or_else(|| invalid("position overflow"))?;
        if rows == 0
            || rows > self.rows
            || cache.capacity > self.max_context
            || !self.heads.is_multiple_of(cache.kv_heads)
            || sinks.len() != self.heads
            || q.len() < rows * self.heads * 128
            || out.len() < rows * self.heads * 128
            || k.len() < rows * cache.kv_heads * 128
            || v.len() < rows * cache.kv_heads * 128
        {
            return Err(invalid("attention buffer or capacity mismatch"));
        }
        unsafe {
            check_cuda(
                "infer_causal_sink_attention",
                infer_causal_sink_attention(
                    q.ptr,
                    cache.key.ptr,
                    cache.value.ptr,
                    k.ptr,
                    v.ptr,
                    sinks.ptr,
                    self.partial.ptr,
                    out.ptr,
                    cache.position as u32,
                    rows as u32,
                    self.heads as u32,
                    cache.kv_heads as u32,
                    cache.capacity as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        cache.position = end;
        Ok(())
    }
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "causal sink attention",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(x: f32) -> f32 {
        let b = x.to_bits();
        f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1))) & 0xffff0000)
    }

    #[test]
    fn packed_sink_decode_preserves_independent_histories_and_graph_replay() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let (heads, kvheads, width, offset) = (4, 2, 256, 256);
        let positions = [0usize, 127, 512, 1023];
        let sinks = [-2.0f32, 0.0, 2.0, 8.0];
        let ds = DeviceBuffer::from_host(&sinks)?;
        let value = |seq: usize, pos: usize, col: usize, salt: usize| {
            bf(((seq * 31 + pos * 17 + col * salt) % 101) as f32 / 64.0 - 0.75)
        };
        for window in [Some(65), Some(513), None] {
            let mut pages = Vec::new();
            let mut tables = Vec::new();
            let mut locals = Vec::new();
            for (seq, &pos) in positions.iter().enumerate() {
                let mut row_pages = Vec::new();
                for page in 0..(pos + 2).div_ceil(128) {
                    let mut data = vec![0u16; offset + 2 * 128 * width];
                    for token in page * 128..((page + 1) * 128).min(pos) {
                        for col in 0..width {
                            data[offset + (token % 128) * width + col] =
                                (value(seq, token, col, 13).to_bits() >> 16) as u16;
                            data[offset + 128 * width + (token % 128) * width + col] =
                                (value(seq, token, col, 7).to_bits() >> 16) as u16;
                        }
                    }
                    row_pages.push(DeviceBuffer::from_host(&data)?);
                }
                tables.push(DeviceBuffer::from_host(
                    &row_pages
                        .iter()
                        .map(DeviceBuffer::cuda_address)
                        .collect::<Vec<_>>(),
                )?);
                pages.push(row_pages);
                let local = if let Some(capacity) = window {
                    let mut cache = SinkAttentionCache::new(capacity, kvheads)?;
                    let mut keys = vec![0u16; capacity * width];
                    let mut values = keys.clone();
                    for token in pos.saturating_sub(capacity)..pos {
                        for col in 0..width {
                            keys[token % capacity * width + col] =
                                (value(seq, token, col, 13).to_bits() >> 16) as u16;
                            values[token % capacity * width + col] =
                                (value(seq, token, col, 7).to_bits() >> 16) as u16;
                        }
                    }
                    cache.key.copy_range_from_host(0, &keys)?;
                    cache.value.copy_range_from_host(0, &values)?;
                    Some(cache)
                } else {
                    None
                };
                locals.push(vec![None, local]);
            }
            let mut batch = SinkAttentionDecodeBatch::new(4, 2)?;
            let mut work = SinkAttentionWorkspace::new(4, heads, 1152)?;
            let mut dq = DeviceBuffer::zeroed(4 * heads * 128)?;
            let mut dk = DeviceBuffer::zeroed(4 * width)?;
            let mut dv = DeviceBuffer::zeroed(4 * width)?;
            let mut out = DeviceBuffer::zeroed(4 * heads * 128)?;
            let mut graph = None;
            for step in 0..2 {
                let order = if step == 0 {
                    [0, 1, 2, 3]
                } else {
                    [3, 0, 2, 1]
                };
                let mut q = Vec::new();
                let mut k = Vec::new();
                let mut v = Vec::new();
                for (row, &seq) in order.iter().enumerate() {
                    let pos = positions[seq] + step;
                    batch.set_row(row, pos, &tables[seq], &locals[seq])?;
                    q.extend((0..heads * 128).map(|col| value(seq, pos, col, 11)));
                    k.extend((0..width).map(|col| value(seq, pos, col, 13)));
                    v.extend((0..width).map(|col| value(seq, pos, col, 7)));
                }
                dq.copy_range_from_host(0, &q)?;
                dk.copy_range_from_host(0, &k)?;
                dv.copy_range_from_host(0, &v)?;
                batch.upload(4)?;
                if graph.is_none() {
                    graph = Some(stream.capture(|_| {
                        work.run_decode_batch(
                            &batch, 1, offset, window, 1152, kvheads, &dq, &dk, &dv, &ds, &mut out,
                            4, &stream,
                        )
                    })?);
                }
                graph.as_ref().unwrap().launch(&stream)?;
                let actual = out.copy_to_host(&stream)?;
                for (row, &seq) in order.iter().enumerate() {
                    let pos = positions[seq] + step;
                    let first = (pos + 1).saturating_sub(window.unwrap_or(1152));
                    for head in 0..heads {
                        let scores = (first..=pos)
                            .map(|token| {
                                (0..128)
                                    .map(|col| {
                                        f64::from(q[(row * heads + head) * 128 + col])
                                            * f64::from(value(seq, token, head / 2 * 128 + col, 13))
                                    })
                                    .sum::<f64>()
                                    / 128f64.sqrt()
                            })
                            .collect::<Vec<_>>();
                        let maximum = scores
                            .iter()
                            .copied()
                            .fold(f64::from(sinks[head]), f64::max);
                        let denominator = (f64::from(sinks[head]) - maximum).exp()
                            + scores.iter().map(|&x| (x - maximum).exp()).sum::<f64>();
                        for col in 0..128 {
                            let expected = scores
                                .iter()
                                .enumerate()
                                .map(|(i, &x)| {
                                    (x - maximum).exp()
                                        * f64::from(value(seq, first + i, head / 2 * 128 + col, 7))
                                })
                                .sum::<f64>()
                                / denominator;
                            assert!(
                                (f64::from(actual[(row * heads + head) * 128 + col]) - expected)
                                    .abs()
                                    < 2e-5,
                                "window={window:?} step={step} row={row} head={head} col={col}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn causal_sink_attention_matches_reference_across_chunks_and_ring_wraps() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let heads = 4;
        let kvheads = 2;
        let sinks = [-2.0, 0.0, 2.0, 8.0];
        let ds = DeviceBuffer::from_host(&sinks)?;
        let total = 1057;
        let q: Vec<_> = (0..total * heads * 128)
            .map(|i| bf(((i * 17 % 113) as f32 - 56.0) / 80.0))
            .collect();
        let k: Vec<_> = (0..total * kvheads * 128)
            .map(|i| bf(((i * 13 % 97) as f32 - 48.0) / 70.0))
            .collect();
        let v: Vec<_> = (0..total * kvheads * 128)
            .map(|i| bf(((i * 7 % 83) as f32 - 41.0) / 50.0))
            .collect();
        for capacity in [65, 513, 1057] {
            let mut cache = SinkAttentionCache::new(capacity, kvheads)?;
            let mut w = SinkAttentionWorkspace::new(129, heads, capacity)?;
            let pages = (0..total.div_ceil(128))
                .map(|_| DeviceBuffer::<u16>::zeroed(2 * 128 * kvheads * 128 + 256))
                .collect::<Result<Vec<_>>>()?;
            let table = DeviceBuffer::from_host(
                &pages
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )?;
            let mut local = (capacity < total)
                .then(|| SinkAttentionCache::new(capacity, kvheads))
                .transpose()?;
            let mut paged = SinkAttentionWorkspace::new(129, heads, total)?;
            let mut position = DeviceBuffer::zeroed(1)?;
            let mut start = 0;
            while start < total {
                let rows = (total - start).min(if start == 0 { 1 } else { 129 });
                let dq =
                    DeviceBuffer::from_host(&q[start * heads * 128..(start + rows) * heads * 128])?;
                let dk = DeviceBuffer::from_host(
                    &k[start * kvheads * 128..(start + rows) * kvheads * 128],
                )?;
                let dv = DeviceBuffer::from_host(
                    &v[start * kvheads * 128..(start + rows) * kvheads * 128],
                )?;
                let mut out = DeviceBuffer::zeroed(rows * heads * 128)?;
                let mut paged_out = DeviceBuffer::zeroed(rows * heads * 128)?;
                w.run(&mut cache, &dq, &dk, &dv, &ds, &mut out, rows, &stream)?;
                stream.synchronize()?;
                position.copy_range_from_host(0, &[start as u32])?;
                paged.run_paged(
                    local.as_mut(),
                    &table,
                    256,
                    128,
                    &position,
                    start,
                    total,
                    kvheads,
                    &dq,
                    &dk,
                    &dv,
                    &ds,
                    &mut paged_out,
                    rows,
                    &stream,
                )?;
                stream.synchronize()?;
                let actual = out.copy_to_host(&stream)?;
                let actual = actual.as_slice();
                assert_eq!(
                    actual,
                    paged_out.copy_to_host(&stream)?.as_slice(),
                    "paged/ring attention differs at {start}, window {capacity}"
                );
                for row in [0, rows - 1] {
                    let pos = start + row;
                    for head in 0..heads {
                        let first = (pos + 1).saturating_sub(capacity);
                        let logits: Vec<f64> = (first..=pos)
                            .map(|p| {
                                (0..128)
                                    .map(|c| {
                                        q[(pos * heads + head) * 128 + c] as f64
                                            * k[(p * kvheads + head / 2) * 128 + c] as f64
                                    })
                                    .sum::<f64>()
                                    / (128f64).sqrt()
                            })
                            .collect();
                        let max = logits.iter().copied().fold(sinks[head] as f64, f64::max);
                        let den = (sinks[head] as f64 - max).exp()
                            + logits.iter().map(|x| (x - max).exp()).sum::<f64>();
                        for c in 0..128 {
                            let expected = logits
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    (x - max).exp()
                                        * v[((first + i) * kvheads + head / 2) * 128 + c] as f64
                                })
                                .sum::<f64>()
                                / den;
                            assert!(
                                (actual[(row * heads + head) * 128 + c] as f64 - expected).abs()
                                    < 2e-5,
                                "capacity={capacity} position={pos} head={head} col={c}"
                            );
                        }
                    }
                }
                start += rows;
            }
            assert_eq!(cache.position(), total);
        }
        Ok(())
    }
}
