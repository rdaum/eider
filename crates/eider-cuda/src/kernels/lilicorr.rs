//! BF16-valued LiLiCorr lattice operations on an explicit CUDA stream.

#![allow(clippy::too_many_arguments)]

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceBuffer, Error, Result};
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_lilicorr_context_attention(
        q: *const f32,
        ck: *const f32,
        cv: *const f32,
        bk: *const f32,
        bv: *const f32,
        sinks: *const f32,
        partial: *mut f32,
        out: *mut f32,
        end: u32,
        retained: u32,
        rows: u32,
        qheads: u32,
        kvheads: u32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_candidates(
        x: *const f32,
        ids: *mut u32,
        logp: *mut f32,
        rows: u32,
        vocab: u32,
        k: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_bias_activation(
        x: *mut f32,
        bias: *const f32,
        n: u32,
        w: u32,
        silu: bool,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_features(
        p: *const f32,
        g: *const f32,
        b: *const f32,
        out: *mut f32,
        n: u32,
        k: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_hidden(
        t: *const f32,
        p: *const f32,
        f: *const f32,
        slot: *const f32,
        rank: *const f32,
        out: *mut f32,
        n: u32,
        w: u32,
        k: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_attention(
        qkv: *const f32,
        rel: *const f32,
        same: *const f32,
        out: *mut f32,
        n: u32,
        w: u32,
        heads: u32,
        k: u32,
        slots: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_cross(
        x: *const f32,
        a: *const f32,
        out: *mut f32,
        n: u32,
        w: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_factors(
        x: *const f32,
        a: *const f32,
        cross: *const f32,
        out: *mut f32,
        n: u32,
        w: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_normalize(x: *mut f32, rows: u32, w: u32, eps: f32, s: *mut c_void) -> i32;
    fn infer_lilicorr_edges(
        out: *const f32,
        incoming: *const f32,
        a: *const f32,
        scores: *mut f32,
        slots: u32,
        w: u32,
        k: u32,
        scale: f32,
        s: *mut c_void,
    ) -> i32;
    fn infer_lilicorr_capture(
        x: *const f32,
        out: *mut f32,
        n: u32,
        w: u32,
        taps: u32,
        tap: u32,
        s: *mut c_void,
    ) -> i32;
}

/// Reusable split-context storage for 128-dimensional noncausal attention.
pub struct ContextAttentionWorkspace {
    partial: DeviceBuffer<f32>,
    rows: usize,
    heads: usize,
    window: usize,
}

impl ContextAttentionWorkspace {
    /// Allocates partial outputs for the sliding context and proposal rows.
    pub fn new(rows: usize, heads: usize, window: usize) -> Result<Self> {
        require(
            rows > 0
                && heads > 0
                && window > 0
                && rows <= window
                && rows <= u32::MAX as usize
                && heads <= u32::MAX as usize
                && window <= u32::MAX as usize,
            "attention dimensions",
        )?;
        let len = rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(window.checked_add(rows)?.div_ceil(256)))
            .and_then(|n| n.checked_mul(130))
            .ok_or_else(|| Error::Format {
                label: "LiLiCorr attention",
                detail: "workspace size overflow".into(),
            })?;
        Ok(Self {
            partial: DeviceBuffer::zeroed(len)?,
            rows,
            heads,
            window,
        })
    }

    /// Runs sliding noncausal attention with a ring-buffer context and a dense proposal.
    pub fn run(
        &mut self,
        q: &DeviceBuffer<f32>,
        ck: &DeviceBuffer<f32>,
        cv: &DeviceBuffer<f32>,
        bk: &DeviceBuffer<f32>,
        bv: &DeviceBuffer<f32>,
        sinks: Option<&DeviceBuffer<f32>>,
        out: &mut DeviceBuffer<f32>,
        end: usize,
        retained: usize,
        rows: usize,
        kvheads: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        require(
            rows > 0
                && rows <= self.rows
                && sinks.is_none_or(|s| s.len() == self.heads)
                && kvheads > 0
                && self.heads.is_multiple_of(kvheads)
                && retained <= self.window
                && retained <= end
                && end
                    .checked_add(rows)
                    .is_some_and(|n| n <= u32::MAX as usize)
                && q.len() >= rows * self.heads * 128
                && out.len() >= rows * self.heads * 128
                && ck.len() >= self.window * kvheads * 128
                && cv.len() >= self.window * kvheads * 128
                && bk.len() >= rows * kvheads * 128
                && bv.len() >= rows * kvheads * 128,
            "attention buffers or positions",
        )?;
        unsafe {
            check_cuda(
                "infer_lilicorr_context_attention",
                infer_lilicorr_context_attention(
                    q.ptr,
                    ck.ptr,
                    cv.ptr,
                    bk.ptr,
                    bv.ptr,
                    sinks.map_or(std::ptr::null(), |s| s.ptr),
                    self.partial.ptr,
                    out.ptr,
                    end as u32,
                    retained as u32,
                    rows as u32,
                    self.heads as u32,
                    kvheads as u32,
                    self.window as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}

fn require(ok: bool, detail: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Format {
            label: "LiLiCorr CUDA",
            detail: detail.into(),
        })
    }
}

fn count(rows: usize, width: usize) -> Result<usize> {
    let n = rows.checked_mul(width);
    require(
        rows > 0 && width > 0 && n.is_some_and(|n| n <= u32::MAX as usize),
        "invalid dimensions",
    )?;
    Ok(n.unwrap())
}

/// Full-vocabulary log-softmax followed by top-k, ordered by score then token ID.
pub fn candidates(
    x: &DeviceBuffer<f32>,
    ids: &mut DeviceBuffer<u32>,
    logp: &mut DeviceBuffer<f32>,
    rows: usize,
    vocab: usize,
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(rows, vocab)?;
    require(
        k > 0
            && k <= 16
            && k <= vocab
            && x.len() >= n
            && ids.len() >= rows * k
            && logp.len() >= rows * k,
        "invalid candidate buffers or top-k",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_candidates",
            infer_lilicorr_candidates(
                x.ptr,
                ids.ptr,
                logp.ptr,
                rows as u32,
                vocab as u32,
                k as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Adds an optional row bias, rounds to BF16, and optionally applies BF16 SiLU.
pub fn bias_activation(
    x: &mut DeviceBuffer<f32>,
    bias: Option<&DeviceBuffer<f32>>,
    rows: usize,
    width: usize,
    silu: bool,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(rows, width)?;
    require(
        x.len() >= n && bias.is_none_or(|b| b.len() == width),
        "invalid bias buffers",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_bias_activation",
            infer_lilicorr_bias_activation(
                x.ptr,
                bias.map_or(std::ptr::null(), |b| b.ptr),
                n as u32,
                width as u32,
                silu,
                stream.as_raw(),
            ),
        )
    }
}

/// Builds the five candidate features and applies the trained BF16 LayerNorm.
pub fn features(
    logp: &DeviceBuffer<f32>,
    gamma: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    slots: usize,
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(slots, k)?;
    require(
        k <= 16
            && n <= u32::MAX as usize / 5
            && logp.len() >= n
            && output.len() >= n * 5
            && gamma.len() == 5
            && beta.len() == 5,
        "invalid feature buffers",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_features",
            infer_lilicorr_features(
                logp.ptr,
                gamma.ptr,
                beta.ptr,
                output.ptr,
                n as u32,
                k as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Adds projected token, pass, feature, slot, and rank states with BF16 rounding.
pub fn hidden(
    token: &DeviceBuffer<f32>,
    pass: &DeviceBuffer<f32>,
    feature: &DeviceBuffer<f32>,
    slot: &DeviceBuffer<f32>,
    rank: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    slots: usize,
    k: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let nodes = count(slots, k)?;
    let n = count(nodes, width)?;
    require(
        token.len() >= n
            && pass.len() >= slots * width
            && feature.len() >= n
            && slot.len() >= slots * width
            && rank.len() >= k * width
            && output.len() >= n,
        "invalid hidden-state buffers",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_hidden",
            infer_lilicorr_hidden(
                token.ptr,
                pass.ptr,
                feature.ptr,
                slot.ptr,
                rank.ptr,
                output.ptr,
                n as u32,
                width as u32,
                k as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Non-causal head-128 lattice attention with the trained relative-slot offset.
pub fn attention(
    qkv: &DeviceBuffer<f32>,
    relative: &DeviceBuffer<f32>,
    same: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    slots: usize,
    trained_slots: usize,
    k: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let nodes = count(slots, k)?;
    let width = count(heads, 128)?;
    let n = count(nodes, width)?;
    require(
        slots <= trained_slots
            && trained_slots <= 7
            && k <= 16
            && n <= u32::MAX as usize / 3
            && qkv.len() >= n * 3
            && relative.len() == heads * (trained_slots * 2 + 1)
            && same.len() == heads
            && output.len() >= n,
        "invalid attention buffers or slot count",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_attention",
            infer_lilicorr_attention(
                qkv.ptr,
                relative.ptr,
                same.ptr,
                output.ptr,
                nodes as u32,
                width as u32,
                heads as u32,
                k as u32,
                trained_slots as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Forms the BF16 elementwise product of each candidate with the context anchor.
pub fn cross(
    input: &DeviceBuffer<f32>,
    anchor: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(rows, width)?;
    require(
        input.len() >= n && anchor.len() >= width && output.len() >= n,
        "invalid anchor product buffers",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_cross",
            infer_lilicorr_cross(
                input.ptr,
                anchor.ptr,
                output.ptr,
                n as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Combines the three trained factor projections before BF16 SiLU.
pub fn factors(
    input: &DeviceBuffer<f32>,
    anchor: &DeviceBuffer<f32>,
    cross: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(rows, width)?;
    require(
        input.len() >= n && anchor.len() >= width && cross.len() >= n && output.len() >= n,
        "invalid factor buffers",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_factors",
            infer_lilicorr_factors(
                input.ptr,
                anchor.ptr,
                cross.ptr,
                output.ptr,
                n as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Normalizes each factor vector with the reference's BF16 norm and epsilon.
pub fn normalize(
    input: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(rows, width)?;
    require(
        input.len() >= n && eps.is_finite() && eps > 0.,
        "invalid normalization buffers or epsilon",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_normalize",
            infer_lilicorr_normalize(input.ptr, rows as u32, width as u32, eps, stream.as_raw()),
        )
    }
}

/// Scores anchor and adjacent-slot transitions, yielding `[slot, predecessor, next]`.
pub fn edges(
    outgoing: &DeviceBuffer<f32>,
    incoming: &DeviceBuffer<f32>,
    anchor: &DeviceBuffer<f32>,
    scores: &mut DeviceBuffer<f32>,
    slots: usize,
    k: usize,
    width: usize,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let nodes = count(slots, k)?;
    let n = count(nodes, width)?;
    let edge_count = count(nodes, k)?;
    require(
        outgoing.len() >= n
            && incoming.len() >= n
            && anchor.len() >= width
            && scores.len() >= edge_count
            && scale.is_finite()
            && scale > 0.,
        "invalid edge buffers or scale",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_edges",
            infer_lilicorr_edges(
                outgoing.ptr,
                incoming.ptr,
                anchor.ptr,
                scores.ptr,
                slots as u32,
                width as u32,
                k as u32,
                scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Copies one target residual stream into interleaved BF16-valued feature rows.
pub fn capture(
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
    taps: usize,
    tap: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = count(rows, width)?;
    let total = count(n, taps)?;
    require(
        tap < taps && input.len() >= n && output.len() >= total,
        "invalid capture buffers or tap",
    )?;
    unsafe {
        check_cuda(
            "infer_lilicorr_capture",
            infer_lilicorr_capture(
                input.ptr,
                output.ptr,
                n as u32,
                width as u32,
                taps as u32,
                tap as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn bf(x: f32) -> f32 {
        let b = x.to_bits();
        f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
    }

    #[test]
    #[serial]
    fn candidates_use_full_vocabulary_log_probs_and_stable_ties() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let x = [
            0., 4., -2., 4., 1., 3., 2., -1., 1., 0., 0., 1., 3., 3., 2., 4., 3., -4.,
        ];
        let input = DeviceBuffer::from_host(&x).unwrap();
        let mut ids = DeviceBuffer::zeroed(8).unwrap();
        let mut logp = DeviceBuffer::zeroed(8).unwrap();
        candidates(&input, &mut ids, &mut logp, 2, 9, 4, &stream).unwrap();
        let ids = ids.copy_to_host(&stream).unwrap();
        let logp = logp.copy_to_host(&stream).unwrap();
        assert_eq!(ids, [1, 3, 5, 6, 6, 3, 4, 7]);
        for row in 0..2 {
            let values = &x[row * 9..(row + 1) * 9];
            let normalizer = values.iter().map(|&v| f64::from(v).exp()).sum::<f64>().ln();
            for rank in 0..4 {
                let i = row * 4 + rank;
                let expected = f64::from(values[ids[i] as usize]) - normalizer;
                assert!((f64::from(logp[i]) - expected).abs() < 2.0e-6);
            }
        }
    }

    #[test]
    #[serial]
    fn shortened_lattice_keeps_trained_relative_bias_offset() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let (slots, trained, k, heads, width) = (3, 7, 2, 2, 256);
        let nodes = slots * k;
        let qkv = (0..nodes * 3 * width)
            .map(|i| bf(((i * 13 % 41) as f32 - 20.) / 32.))
            .collect::<Vec<_>>();
        let relative = (0..heads * (2 * trained + 1))
            .map(|i| bf((i % 15) as f32 / 8. - 0.5))
            .collect::<Vec<_>>();
        let same = [0.25, -0.5];
        let mut expected = vec![0.; nodes * width];
        for q in 0..nodes {
            for head in 0..heads {
                let mut scores = Vec::new();
                for key in 0..nodes {
                    let dot = (0..128)
                        .map(|j| {
                            f64::from(qkv[q * 3 * width + head * 128 + j])
                                * f64::from(qkv[key * 3 * width + width + head * 128 + j])
                        })
                        .sum::<f64>();
                    let distance = (q / k) as isize - (key / k) as isize;
                    let bias = bf(relative[head * 15 + (distance + trained as isize) as usize]
                        + if distance == 0 { same[head] } else { 0. });
                    scores.push(dot / 128f64.sqrt() + f64::from(bias));
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
                for j in 0..128 {
                    let value = (0..nodes)
                        .map(|key| {
                            (scores[key] - max).exp() / z
                                * f64::from(qkv[key * 3 * width + 2 * width + head * 128 + j])
                        })
                        .sum::<f64>();
                    expected[q * width + head * 128 + j] = bf(value as f32);
                }
            }
        }
        let qkv = DeviceBuffer::from_host(&qkv).unwrap();
        let relative = DeviceBuffer::from_host(&relative).unwrap();
        let same = DeviceBuffer::from_host(&same).unwrap();
        let mut output = DeviceBuffer::zeroed(nodes * width).unwrap();
        attention(
            &qkv,
            &relative,
            &same,
            &mut output,
            slots,
            trained,
            k,
            heads,
            &stream,
        )
        .unwrap();
        for (i, (a, e)) in output
            .copy_to_host(&stream)
            .unwrap()
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert!((a - e).abs() <= 0.001, "index {i}: {a} != {e}");
        }
    }

    #[test]
    #[serial]
    fn factors_normalize_zero_vectors_and_score_adjacent_slots() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let mut out = DeviceBuffer::from_host(&[
            3., 4., 0., 0., 0., 5., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
        ])
        .unwrap();
        let mut incoming = DeviceBuffer::from_host(&[
            1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
        ])
        .unwrap();
        let mut anchor = DeviceBuffer::from_host(&[0., 0., 0., 0.]).unwrap();
        normalize(&mut out, 4, 4, 0.0001, &stream).unwrap();
        normalize(&mut incoming, 4, 4, 0.0001, &stream).unwrap();
        normalize(&mut anchor, 1, 4, 0.0001, &stream).unwrap();
        let mut scores = DeviceBuffer::zeroed(8).unwrap();
        edges(&out, &incoming, &anchor, &mut scores, 2, 2, 4, 8., &stream).unwrap();
        assert_eq!(
            scores.copy_to_host(&stream).unwrap(),
            [0., 0., 0., 0., 8. * bf(0.6), 8. * bf(0.8), 0., 8.]
        );
    }

    #[test]
    #[serial]
    fn capture_interleaves_target_taps_without_overwriting_other_taps() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let input = DeviceBuffer::from_host(&[1.01, 2.02, 3.03, 4.04]).unwrap();
        let mut output = DeviceBuffer::from_host(&[-9.; 12]).unwrap();
        capture(&input, &mut output, 2, 2, 3, 1, &stream).unwrap();
        assert_eq!(
            output.copy_to_host(&stream).unwrap(),
            [
                -9.,
                -9.,
                bf(1.01),
                bf(2.02),
                -9.,
                -9.,
                -9.,
                -9.,
                bf(3.03),
                bf(4.04),
                -9.,
                -9.
            ]
        );
    }
}
