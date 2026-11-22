//! Inkling 2 DSpark draft operators.
//!
//! Draft activations are BF16 values in F32 storage. Draft K/V reuse the
//! target's BF16 layouts: local layers keep a [`SinkAttentionCache`] ring and
//! the global layer occupies one slot of the sequence's page bundles.

use super::invalid;
use crate::Result;
use crate::cuda::{CudaStream, DeviceAddress, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::kernels::sink_attention::SinkAttentionCache;
use std::ffi::c_void;

/// Largest candidate count kept per draft row.
pub const MAX_CANDIDATES: usize = 32;
/// Largest draft block: one anchor and seven masks.
pub const MAX_BLOCK_ROWS: usize = 8;
/// Key splits of one draft attention launch.
pub const ATTENTION_SPLITS: usize = 64;
const TOPK_CHUNK: usize = 2048;

unsafe extern "C" {
    fn infer_inkling2_finish_logit_rows(
        logits: *mut f32,
        rows: u32,
        vocab: u32,
        real_vocab: u32,
        softcap: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_mix(
        features: *const f32,
        alpha: *const f32,
        weight: *const f32,
        out: *mut f32,
        first: u32,
        rows: u32,
        tap_rows: u32,
        taps: u32,
        width: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_norm_rope(
        x: *mut f32,
        weight: *const f32,
        rows: u32,
        heads: u32,
        row_stride: u32,
        first: u32,
        theta: f32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_attention(
        q: *const f32,
        bk: *const f32,
        bv: *const f32,
        ck: *const u16,
        cv: *const u16,
        pages: *const DeviceAddress<u16>,
        layer_offset: u32,
        first: u32,
        end: u32,
        capacity: u32,
        rows: u32,
        qheads: u32,
        kvheads: u32,
        splits: u32,
        partial: *mut f32,
        out: *mut f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_store(
        k: *const f32,
        v: *const f32,
        ck: *mut u16,
        cv: *mut u16,
        pages: *const DeviceAddress<u16>,
        layer_offset: u32,
        first: u32,
        rows: u32,
        width: u32,
        capacity: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_fill(
        mask: *const f32,
        out: *mut f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_add_broadcast(
        x: *mut f32,
        a: *const f32,
        b: *const f32,
        rows: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_markov_input(
        state: *const f32,
        embedding: *const u16,
        token: *const u32,
        out: *mut f32,
        rank: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_markov_gate(
        raw: *const f32,
        state: *mut f32,
        latent: *mut f32,
        rank: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_markov_logits(
        base: *const f32,
        bias: *const f32,
        fold_bias: *const f32,
        out: *mut f32,
        vocab: u32,
        real_vocab: u32,
        softcap: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_draft_topk(
        logits: *const f32,
        rows: u32,
        stride: u32,
        real_vocab: u32,
        k: u32,
        chunk_values: *mut f32,
        chunk_ids: *mut u32,
        values: *mut f32,
        ids: *mut u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Draft K/V history read by block attention.
pub enum DraftKv<'a> {
    /// A local layer's window ring.
    Ring(&'a SinkAttentionCache),
    /// The global layer's slot in the sequence's page bundles.
    Pages {
        /// Device addresses of the sequence's page bundles.
        table: &'a DeviceBuffer<DeviceAddress<u16>>,
        /// Element offset of the draft slot within each bundle.
        layer_offset: usize,
    },
}

/// Draft K/V history written by context stores.
pub enum DraftKvMut<'a> {
    /// A local layer's window ring.
    Ring(&'a mut SinkAttentionCache),
    /// The global layer's slot in the sequence's page bundles.
    Pages {
        /// Device addresses of the sequence's page bundles.
        table: &'a DeviceBuffer<DeviceAddress<u16>>,
        /// Element offset of the draft slot within each bundle.
        layer_offset: usize,
    },
}

fn u32_of(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))
}

/// Applies the final softcap to `rows` logit rows and masks padded entries.
pub fn finish_logit_rows(
    logits: &mut DeviceBuffer<f32>,
    rows: usize,
    vocab: usize,
    real_vocab: usize,
    softcap: f32,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0 || logits.len() < rows * vocab || real_vocab > vocab || softcap <= 0.0 {
        return Err(invalid("logit-row geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_finish_logit_rows",
            infer_inkling2_finish_logit_rows(
                logits.ptr,
                u32_of(rows, "rows")?,
                u32_of(vocab, "vocabulary")?,
                real_vocab as u32,
                softcap,
                stream.as_raw(),
            ),
        )
    }
}

/// Normalizes the `alpha`-weighted sum of feature taps `[tap][tap_rows][width]`
/// for rows `first..first + rows`.
#[allow(clippy::too_many_arguments)]
pub fn mix(
    features: &DeviceBuffer<f32>,
    alpha: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    first: usize,
    rows: usize,
    tap_rows: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let taps = alpha.len();
    let width = weight.len();
    if rows == 0
        || first + rows > tap_rows
        || features.len() < taps * tap_rows * width
        || out.len() < rows * width
    {
        return Err(invalid("draft feature mix geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_mix",
            infer_inkling2_draft_mix(
                features.ptr,
                alpha.ptr,
                weight.ptr,
                out.buffer_mut().ptr,
                u32_of(first, "first row")?,
                u32_of(rows, "rows")?,
                u32_of(tap_rows, "tap rows")?,
                u32_of(taps, "taps")?,
                u32_of(width, "width")?,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Optionally RMS-normalizes each 128-wide head, then applies NeoX RoPE at
/// position `first + row`.
#[allow(clippy::too_many_arguments)]
pub fn norm_rope(
    mut x: DeviceInOut<'_, f32>,
    weight: Option<&DeviceBuffer<f32>>,
    rows: usize,
    heads: usize,
    row_stride: usize,
    first: usize,
    theta: f32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || heads * 128 > row_stride
        || x.len() < rows * row_stride
        || weight.is_some_and(|w| w.len() != 128)
    {
        return Err(invalid("draft RoPE geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_norm_rope",
            infer_inkling2_draft_norm_rope(
                x.buffer_mut().ptr,
                weight.map_or(std::ptr::null(), |w| w.ptr.cast_const()),
                u32_of(rows, "rows")?,
                u32_of(heads, "heads")?,
                u32_of(row_stride, "row stride")?,
                u32_of(first, "position")?,
                theta,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Bidirectional attention of `rows` block queries over context positions
/// `[first, end)` and the block's own keys. `partial` holds
/// [`ATTENTION_SPLITS`] softmax partials per query.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &DeviceBuffer<f32>,
    block_k: &DeviceBuffer<f32>,
    block_v: &DeviceBuffer<f32>,
    kv: DraftKv<'_>,
    first: usize,
    end: usize,
    rows: usize,
    qheads: usize,
    kvheads: usize,
    partial: &mut DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let (ck, cv, pages, layer_offset, capacity) = match kv {
        DraftKv::Ring(cache) => {
            if cache.kv_heads != kvheads || end - first > cache.capacity {
                return Err(invalid("draft ring does not hold the context window"));
            }
            (
                cache.key.ptr.cast_const(),
                cache.value.ptr.cast_const(),
                std::ptr::null(),
                0,
                cache.capacity,
            )
        }
        DraftKv::Pages {
            table,
            layer_offset,
        } => {
            if table.len() < end.div_ceil(128) {
                return Err(invalid("draft page table does not cover the context"));
            }
            (
                std::ptr::null(),
                std::ptr::null(),
                table.ptr.cast_const(),
                layer_offset,
                0,
            )
        }
    };
    if rows == 0
        || rows > MAX_BLOCK_ROWS
        || first > end
        || kvheads == 0
        || !qheads.is_multiple_of(kvheads)
        || qheads / kvheads > 3
        || q.len() < rows * qheads * 128
        || block_k.len() < rows * kvheads * 128
        || block_v.len() < rows * kvheads * 128
        || partial.len() < ATTENTION_SPLITS * rows * qheads * 130
        || out.len() < rows * qheads * 128
    {
        return Err(invalid("draft attention geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_attention",
            infer_inkling2_draft_attention(
                q.ptr,
                block_k.ptr,
                block_v.ptr,
                ck,
                cv,
                pages,
                u32_of(layer_offset, "layer offset")?,
                u32_of(first, "first position")?,
                u32_of(end, "end position")?,
                u32_of(capacity, "capacity")?,
                rows as u32,
                qheads as u32,
                kvheads as u32,
                ATTENTION_SPLITS as u32,
                partial.ptr,
                out.buffer_mut().ptr,
                stream.as_raw(),
            ),
        )
    }
}

/// Stores `rows` context K/V rows at positions `first..first + rows`.
pub fn store(
    k: &DeviceBuffer<f32>,
    v: &DeviceBuffer<f32>,
    kv: DraftKvMut<'_>,
    first: usize,
    rows: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0 || k.len() < rows * width || v.len() < rows * width {
        return Err(invalid("draft store geometry mismatch"));
    }
    let (ck, cv, pages, layer_offset, capacity) = match kv {
        DraftKvMut::Ring(cache) => {
            if cache.kv_heads * 128 != width {
                return Err(invalid("draft ring width mismatch"));
            }
            (
                cache.key.ptr,
                cache.value.ptr,
                std::ptr::null(),
                0,
                cache.capacity,
            )
        }
        DraftKvMut::Pages {
            table,
            layer_offset,
        } => {
            if table.len() < (first + rows).div_ceil(128) {
                return Err(invalid("draft page table does not cover the store"));
            }
            (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                table.ptr.cast_const(),
                layer_offset,
                0,
            )
        }
    };
    unsafe {
        check_cuda(
            "infer_inkling2_draft_store",
            infer_inkling2_draft_store(
                k.ptr,
                v.ptr,
                ck,
                cv,
                pages,
                u32_of(layer_offset, "layer offset")?,
                u32_of(first, "first position")?,
                u32_of(rows, "rows")?,
                u32_of(width, "width")?,
                u32_of(capacity, "capacity")?,
                stream.as_raw(),
            ),
        )
    }
}

/// Writes `mask` into `rows` consecutive rows.
pub fn fill(
    mask: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    offset: usize,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let width = mask.len();
    if rows == 0 || out.len() < offset + rows * width {
        return Err(invalid("draft mask fill geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_fill",
            infer_inkling2_draft_fill(
                mask.ptr,
                out.ptr.add(offset),
                u32_of(rows, "rows")?,
                u32_of(width, "width")?,
                stream.as_raw(),
            ),
        )
    }
}

/// Adds broadcast rows `a` and optional `b` to each of `rows` rows of `x`.
pub fn add_broadcast(
    mut x: DeviceInOut<'_, f32>,
    a: &DeviceBuffer<f32>,
    b: Option<&DeviceBuffer<f32>>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let width = a.len();
    if rows == 0 || x.len() < rows * width || b.is_some_and(|b| b.len() != width) {
        return Err(invalid("draft broadcast geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_add_broadcast",
            infer_inkling2_draft_add_broadcast(
                x.buffer_mut().ptr,
                a.ptr,
                b.map_or(std::ptr::null(), |b| b.ptr.cast_const()),
                u32_of(rows, "rows")?,
                u32_of(width, "width")?,
                stream.as_raw(),
            ),
        )
    }
}

/// Concatenates the recurrent state with the embedding of the previous token.
pub fn markov_input(
    state: &DeviceBuffer<f32>,
    embedding: &DeviceBuffer<u16>,
    token: &DeviceBuffer<u32>,
    mut out: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let rank = state.len();
    if token.is_empty() || out.len() < 2 * rank || !embedding.len().is_multiple_of(rank) {
        return Err(invalid("draft Markov input geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_markov_input",
            infer_inkling2_draft_markov_input(
                state.ptr,
                embedding.ptr,
                token.ptr,
                out.buffer_mut().ptr,
                u32_of(rank, "rank")?,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances the recurrent state and writes the step's latent.
pub fn markov_gate(
    raw: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut latent: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let rank = state.len();
    if raw.len() < 3 * rank || latent.len() < rank {
        return Err(invalid("draft Markov gate geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_markov_gate",
            infer_inkling2_draft_markov_gate(
                raw.ptr,
                state.buffer_mut().ptr,
                latent.buffer_mut().ptr,
                u32_of(rank, "rank")?,
                stream.as_raw(),
            ),
        )
    }
}

/// Softcapped draft logits of one row: `base` (at `base_offset`) plus the
/// folded Markov bias `bias + fold_bias`.
#[allow(clippy::too_many_arguments)]
pub fn markov_logits(
    base: &DeviceBuffer<f32>,
    base_offset: usize,
    bias: &DeviceBuffer<f32>,
    fold_bias: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    real_vocab: usize,
    softcap: f32,
    stream: &CudaStream,
) -> Result<()> {
    let vocab = fold_bias.len();
    if base.len() < base_offset + vocab
        || bias.len() < vocab
        || out.len() < vocab
        || real_vocab > vocab
        || softcap <= 0.0
    {
        return Err(invalid("draft Markov logit geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_draft_markov_logits",
            infer_inkling2_draft_markov_logits(
                base.ptr.add(base_offset),
                bias.ptr,
                fold_bias.ptr,
                out.buffer_mut().ptr,
                u32_of(vocab, "vocabulary")?,
                real_vocab as u32,
                softcap,
                stream.as_raw(),
            ),
        )
    }
}

/// Scratch for per-row top-k selection over a vocabulary.
pub struct TopK {
    chunk_values: DeviceBuffer<f32>,
    chunk_ids: DeviceBuffer<u32>,
    rows: usize,
    k: usize,
    real_vocab: usize,
}

impl TopK {
    /// Allocates selection scratch for `rows` rows of `k` candidates.
    pub fn new(rows: usize, k: usize, real_vocab: usize) -> Result<Self> {
        let chunks = real_vocab.div_ceil(TOPK_CHUNK);
        if rows == 0 || k == 0 || k > MAX_CANDIDATES || chunks * k > 4096 {
            return Err(invalid("draft top-k geometry mismatch"));
        }
        Ok(Self {
            chunk_values: DeviceBuffer::zeroed(rows * chunks * k)?,
            chunk_ids: DeviceBuffer::zeroed(rows * chunks * k)?,
            rows,
            k,
            real_vocab,
        })
    }

    /// Selects the `k` largest of the first `real_vocab` entries of each row.
    pub fn run(
        &mut self,
        logits: &DeviceBuffer<f32>,
        rows: usize,
        stride: usize,
        values: &mut DeviceBuffer<f32>,
        ids: &mut DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0
            || rows > self.rows
            || stride < self.real_vocab
            || logits.len() < rows * stride
            || values.len() < rows * self.k
            || ids.len() < rows * self.k
        {
            return Err(invalid("draft top-k buffer mismatch"));
        }
        unsafe {
            check_cuda(
                "infer_inkling2_draft_topk",
                infer_inkling2_draft_topk(
                    logits.ptr,
                    rows as u32,
                    u32_of(stride, "stride")?,
                    self.real_vocab as u32,
                    self.k as u32,
                    self.chunk_values.ptr,
                    self.chunk_ids.ptr,
                    values.ptr,
                    ids.ptr,
                    stream.as_raw(),
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn value(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn values(count: usize, seed: &mut u64) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                value(bf16(
                    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0,
                ))
            })
            .collect()
    }

    /// Bidirectional block attention over context positions `first..end`
    /// followed by the block keys.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q: &[f32],
        block_k: &[f32],
        block_v: &[f32],
        context: impl Fn(usize, usize) -> (f32, f32),
        first: usize,
        end: usize,
        rows: usize,
        qheads: usize,
        kvheads: usize,
    ) -> Vec<f32> {
        let width = kvheads * 128;
        let mut out = vec![0.0; rows * qheads * 128];
        for row in 0..rows {
            for head in 0..qheads {
                let kvhead = head / (qheads / kvheads);
                let query = &q[(row * qheads + head) * 128..][..128];
                let keys = (first..end)
                    .map(|position| {
                        (0..128)
                            .map(|d| context(position, kvhead * 128 + d))
                            .collect::<Vec<_>>()
                    })
                    .chain((0..rows).map(|key| {
                        (0..128)
                            .map(|d| {
                                let offset = key * width + kvhead * 128 + d;
                                (block_k[offset], block_v[offset])
                            })
                            .collect()
                    }))
                    .collect::<Vec<_>>();
                let scores = keys
                    .iter()
                    .map(|key| {
                        key.iter().zip(query).map(|((k, _), q)| k * q).sum::<f32>() / 128f32.sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let total = weights.iter().sum::<f32>();
                for d in 0..128 {
                    let sum = keys
                        .iter()
                        .zip(&weights)
                        .map(|(key, weight)| key[d].1 * weight)
                        .sum::<f32>();
                    out[(row * qheads + head) * 128 + d] = value(bf16(sum / total));
                }
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 1e-2, "index {index}: {a} vs {e}");
        }
    }

    #[test]
    fn draft_attention_matches_reference_over_rings_and_pages() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let (rows, qheads, kvheads) = (5, 6, 2);
        let width = kvheads * 128;
        let mut seed = 7;
        let q = values(rows * qheads * 128, &mut seed);
        let block_k = values(rows * width, &mut seed);
        let block_v = values(rows * width, &mut seed);
        let q_device = DeviceBuffer::from_host(&q)?;
        let k_device = DeviceBuffer::from_host(&block_k)?;
        let v_device = DeviceBuffer::from_host(&block_v)?;
        let mut partial = DeviceBuffer::zeroed(ATTENTION_SPLITS * rows * qheads * 130)?;
        let mut out = DeviceBuffer::zeroed(rows * qheads * 128)?;

        // A wrapped local window: positions 36..100 in a 64-slot ring.
        let capacity = 64;
        let ring_keys = values(capacity * width, &mut seed);
        let ring_values = values(capacity * width, &mut seed);
        let mut ring = SinkAttentionCache::new(capacity, kvheads)?;
        ring.key =
            DeviceBuffer::from_host(&ring_keys.iter().map(|&x| bf16(x)).collect::<Vec<_>>())?;
        ring.value =
            DeviceBuffer::from_host(&ring_values.iter().map(|&x| bf16(x)).collect::<Vec<_>>())?;
        let (first, end) = (36, 100);
        attention(
            &q_device,
            &k_device,
            &v_device,
            DraftKv::Ring(&ring),
            first,
            end,
            rows,
            qheads,
            kvheads,
            &mut partial,
            out.output(),
            &stream,
        )?;
        let expected = reference(
            &q,
            &block_k,
            &block_v,
            |position, column| {
                let offset = position % capacity * width + column;
                (ring_keys[offset], ring_values[offset])
            },
            first,
            end,
            rows,
            qheads,
            kvheads,
        );
        assert_close(&out.copy_to_host(&stream)?, &expected);

        // The second slot of two-slot page bundles, positions 0..200.
        let slot = 2 * 128 * width;
        let pages = (0..2)
            .map(|_| values(2 * slot, &mut seed))
            .collect::<Vec<_>>();
        let page_buffers = pages
            .iter()
            .map(|page| DeviceBuffer::from_host(&page.iter().map(|&x| bf16(x)).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>>>()?;
        let table = DeviceBuffer::from_host(
            &page_buffers
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )?;
        let end = 200;
        attention(
            &q_device,
            &k_device,
            &v_device,
            DraftKv::Pages {
                table: &table,
                layer_offset: slot,
            },
            0,
            end,
            rows,
            qheads,
            kvheads,
            &mut partial,
            out.output(),
            &stream,
        )?;
        let expected = reference(
            &q,
            &block_k,
            &block_v,
            |position, column| {
                let page = &pages[position / 128];
                let offset = slot + position % 128 * width + column;
                (page[offset], page[offset + 128 * width])
            },
            0,
            end,
            rows,
            qheads,
            kvheads,
        );
        assert_close(&out.copy_to_host(&stream)?, &expected);
        Ok(())
    }

    #[test]
    fn top_k_selects_the_largest_real_vocabulary_entries() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let (rows, stride, real, k) = (2, 5000, 4990, 8);
        let mut seed = 11;
        let mut logits = values(rows * stride, &mut seed);
        // A padded entry beats every real logit but must be ignored.
        logits[real + 3] = 10.0;
        let device = DeviceBuffer::from_host(&logits)?;
        let mut topk = TopK::new(rows, k, real)?;
        let mut top_values = DeviceBuffer::zeroed(rows * k)?;
        let mut top_ids = DeviceBuffer::zeroed(rows * k)?;
        topk.run(
            &device,
            rows,
            stride,
            &mut top_values,
            &mut top_ids,
            &stream,
        )?;
        let ids = top_ids.copy_to_host(&stream)?.into_vec();
        for row in 0..rows {
            let mut expected = (0..real as u32).collect::<Vec<_>>();
            let entries = &logits[row * stride..];
            expected.sort_by(|&a, &b| {
                entries[b as usize]
                    .total_cmp(&entries[a as usize])
                    .then(a.cmp(&b))
            });
            assert_eq!(&ids[row * k..(row + 1) * k], &expected[..k]);
        }
        Ok(())
    }
}
