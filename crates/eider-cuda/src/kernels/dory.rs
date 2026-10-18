//! Dory BF16-valued nGPT operations on an explicit stream.
#![allow(clippy::too_many_arguments)]
use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceBuffer, DeviceOutput, Error, Result};
use std::ffi::c_void;
unsafe extern "C" {
    fn infer_dory_norm(
        a: *const f32,
        b: *const f32,
        p: *const f32,
        q: *const f32,
        o: *mut f32,
        rows: u32,
        w: u32,
        mode: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_dory_rope_norm(
        x: *const f32,
        scale: *const f32,
        o: *mut f32,
        pos: *const u32,
        rows: u32,
        heads: u32,
        theta: f32,
        s: *mut c_void,
    ) -> i32;
    fn infer_dory_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        ck: *mut u16,
        cv: *mut u16,
        p: *mut f32,
        o: *mut f32,
        pos: *const u32,
        rows: u32,
        qh: u32,
        kh: u32,
        window: u32,
        splits: u32,
        capacity: u32,
        s: *mut c_void,
    ) -> i32;
    fn infer_dory_logits(x: *mut f32, scales: *const f32, n: u32, s: *mut c_void) -> i32;
}
/// Normalization inputs, with scale vectors prepared at model load time.
pub enum DoryNorm<'a> {
    /// Unit L2 normalization with epsilon 1e-6.
    L2,
    /// Round a residual sum to BF16 before normalization.
    Add(&'a DeviceBuffer<f32>),
    /// Interpolate independently normalized residual and branch.
    Residual {
        /// Sublayer projection.
        branch: &'a DeviceBuffer<f32>,
        /// Absolute checkpoint alpha times its initialization factor.
        alpha: &'a DeviceBuffer<f32>,
    },
    /// Sigmoid attention gate followed by L2 normalization.
    AttentionGate {
        /// Gate projection.
        gate: &'a DeviceBuffer<f32>,
        /// Squared checkpoint gate scale including initialization factor.
        scale: &'a DeviceBuffer<f32>,
    },
    /// Scaled BF16 SwiGLU followed by L2 normalization.
    SwiGlu {
        /// Up projection.
        up: &'a DeviceBuffer<f32>,
        /// Gate scale times sqrt(hidden size).
        gate_scale: &'a DeviceBuffer<f32>,
        /// Up scale times sqrt(hidden size).
        up_scale: &'a DeviceBuffer<f32>,
    },
}
/// Applies L2 normalization and the specified BF16 arithmetic boundaries.
pub fn dory_norm(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    width: usize,
    mode: DoryNorm<'_>,
    stream: &CudaStream,
) -> Result<()> {
    let (kind, b, p, q) = match mode {
        DoryNorm::L2 => (0, None, None, None),
        DoryNorm::Add(b) => (1, Some(b), None, None),
        DoryNorm::Residual { branch, alpha } => (2, Some(branch), Some(alpha), None),
        DoryNorm::AttentionGate { gate, scale } => (3, Some(gate), Some(scale), None),
        DoryNorm::SwiGlu {
            up,
            gate_scale,
            up_scale,
        } => (4, Some(up), Some(gate_scale), Some(up_scale)),
    };
    let n = rows.saturating_mul(width);
    if rows == 0
        || width == 0
        || width > 8192
        || n > u32::MAX as usize
        || input.len() < n
        || output.len() < n
        || b.is_some_and(|b| b.len() < n)
        || p.is_some_and(|p| p.len() != width)
        || q.is_some_and(|q| q.len() != width)
    {
        return Err(invalid("normalization dimensions"));
    }
    unsafe {
        check_cuda(
            "infer_dory_norm",
            infer_dory_norm(
                input.as_const_ptr().cast(),
                b.map_or(std::ptr::null(), |x| x.as_const_ptr().cast()),
                p.map_or(std::ptr::null(), |x| x.as_const_ptr().cast()),
                q.map_or(std::ptr::null(), |x| x.as_const_ptr().cast()),
                output.as_mut_ptr().cast(),
                rows as u32,
                width as u32,
                kind,
                stream.as_raw(),
            ),
        )
    }
}
/// Applies NeoX RoPE, BF16 rounding, L2 head normalization, and optional Q scale.
pub fn dory_rope_norm(
    input: &DeviceBuffer<f32>,
    scale: Option<&DeviceBuffer<f32>>,
    mut output: DeviceOutput<'_, f32>,
    position: &DeviceBuffer<u32>,
    rows: usize,
    heads: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let n = rows.saturating_mul(heads).saturating_mul(256);
    if n == 0
        || n > u32::MAX as usize
        || input.len() < n
        || output.len() < n
        || position.len() != 1
        || !theta.is_finite()
        || theta <= 0.0
        || scale.is_some_and(|s| s.len() != heads * 256)
    {
        return Err(invalid("RoPE dimensions or theta"));
    }
    unsafe {
        check_cuda(
            "infer_dory_rope_norm",
            infer_dory_rope_norm(
                input.as_const_ptr().cast(),
                scale.map_or(std::ptr::null(), |x| x.as_const_ptr().cast()),
                output.as_mut_ptr().cast(),
                position.as_const_ptr().cast(),
                rows as u32,
                heads as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}
/// Appends K/V and computes causal attention. `position + rows` must fit `capacity`.
/// Position is a device scalar so this operation can be captured for decode.
pub fn dory_attention(
    query: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    mut cache_key: DeviceOutput<'_, u16>,
    mut cache_value: DeviceOutput<'_, u16>,
    mut partial: DeviceOutput<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    position: &DeviceBuffer<u32>,
    rows: usize,
    qheads: usize,
    kvheads: usize,
    capacity: usize,
    window: usize,
    stream: &CudaStream,
) -> Result<()> {
    let splits = capacity
        .min(if window == 0 { capacity } else { window })
        .div_ceil(256);
    let qn = rows.saturating_mul(qheads).saturating_mul(256);
    let kn = rows.saturating_mul(kvheads).saturating_mul(256);
    let cn = capacity.saturating_mul(kvheads).saturating_mul(256);
    let pn = rows
        .saturating_mul(qheads)
        .saturating_mul(splits)
        .saturating_mul(258);
    if rows == 0
        || rows > capacity
        || rows > 65535
        || splits > 65535
        || window > u32::MAX as usize
        || capacity > u32::MAX as usize
        || kvheads == 0
        || qheads == 0
        || !qheads.is_multiple_of(kvheads)
        || qn > u32::MAX as usize
        || cn > u32::MAX as usize
        || query.len() < qn
        || key.len() < kn
        || value.len() < kn
        || output.len() < qn
        || cache_key.len() < cn
        || cache_value.len() < cn
        || partial.len() < pn
        || position.len() != 1
    {
        return Err(invalid("attention dimensions"));
    }
    unsafe {
        check_cuda(
            "infer_dory_attention",
            infer_dory_attention(
                query.as_const_ptr().cast(),
                key.as_const_ptr().cast(),
                value.as_const_ptr().cast(),
                cache_key.as_mut_ptr().cast(),
                cache_value.as_mut_ptr().cast(),
                partial.as_mut_ptr().cast(),
                output.as_mut_ptr().cast(),
                position.as_const_ptr().cast(),
                rows as u32,
                qheads as u32,
                kvheads as u32,
                window as u32,
                splits as u32,
                capacity as u32,
                stream.as_raw(),
            ),
        )
    }
}
/// Rounds head outputs to BF16 and applies per-vocabulary checkpoint scales.
pub fn dory_scale_logits(
    mut logits: DeviceOutput<'_, f32>,
    scales: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if scales.is_empty() || scales.len() > u32::MAX as usize || logits.len() != scales.len() {
        return Err(invalid("logit scale dimensions"));
    }
    unsafe {
        check_cuda(
            "infer_dory_logits",
            infer_dory_logits(
                logits.as_mut_ptr().cast(),
                scales.as_const_ptr().cast(),
                scales.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}
fn invalid(detail: &str) -> Error {
    Error::Format {
        label: "Dory kernel",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{bf16_to_f32, f32_to_bf16};
    use serial_test::serial;
    fn bf(x: f32) -> f32 {
        bf16_to_f32(f32_to_bf16(x))
    }
    fn norm(x: &[f32]) -> Vec<f32> {
        let inv = (x.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
        x.iter().map(|x| bf(x * inv)).collect()
    }
    #[test]
    #[serial]
    fn dory_residual_preserves_bf16_normalization_boundaries() -> Result<()> {
        let a = (0..2560)
            .map(|i| bf(((i * 17 % 101) as f32 - 50.) / 100.))
            .collect::<Vec<_>>();
        let b = (0..2560)
            .map(|i| bf(((i * 13 % 97) as f32 - 48.) / 80.))
            .collect::<Vec<_>>();
        let alpha = (0..2560)
            .map(|i| 0.01 + (i % 9) as f32 * 0.01)
            .collect::<Vec<_>>();
        let na = norm(&a);
        let nb = norm(&b);
        let expected = norm(
            &na.iter()
                .zip(nb)
                .zip(&alpha)
                .map(|((&a, b), &rate)| a + rate * (b - a))
                .collect::<Vec<_>>(),
        );
        let a = DeviceBuffer::from_host(&a)?;
        let b = DeviceBuffer::from_host(&b)?;
        let alpha = DeviceBuffer::from_host(&alpha)?;
        let mut out = DeviceBuffer::zeroed(2560)?;
        let stream = CudaStream::new_non_blocking()?;
        dory_norm(
            &a,
            out.output(),
            1,
            2560,
            DoryNorm::Residual {
                branch: &b,
                alpha: &alpha,
            },
            &stream,
        )?;
        for (a, b) in out.copy_to_host(&stream)?.iter().zip(expected) {
            assert!((a - b).abs() <= 0.00025, "{a} != {b}");
        }
        Ok(())
    }
    #[test]
    #[serial]
    fn dory_attention_is_causal_and_excludes_expired_window() -> Result<()> {
        let start = 1030;
        let rows = 3;
        let capacity = start + rows;
        let qh = 4;
        let kh = 1;
        let width = 256;
        let window = 1024;
        let q = DeviceBuffer::zeroed(rows * qh * width)?;
        let k = DeviceBuffer::zeroed(rows * kh * width)?;
        let values = (0..capacity)
            .flat_map(|t| vec![bf(t as f32 / 1024.); width])
            .collect::<Vec<_>>();
        let v = DeviceBuffer::from_host(&values[start * width..])?;
        let mut ck = DeviceBuffer::zeroed(capacity * width)?;
        let mut cv =
            DeviceBuffer::from_host(&values.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>())?;
        let pos = DeviceBuffer::from_host(&[start as u32])?;
        let mut partial = DeviceBuffer::zeroed(rows * qh * 4 * 258)?;
        let mut out = DeviceBuffer::zeroed(rows * qh * width)?;
        let stream = CudaStream::new_non_blocking()?;
        dory_attention(
            &q,
            &k,
            &v,
            ck.output(),
            cv.output(),
            partial.output(),
            out.output(),
            &pos,
            rows,
            qh,
            kh,
            capacity,
            window,
            &stream,
        )?;
        let actual = out.copy_to_host(&stream)?;
        for row in 0..rows {
            let end = start + row + 1;
            let expected =
                bf((end - window..end).map(|t| values[t * width]).sum::<f32>() / window as f32);
            for &a in &actual[row * qh * width..(row + 1) * qh * width] {
                assert!((a - expected).abs() < 0.004, "row={row} {a} != {expected}");
            }
        }
        Ok(())
    }
}
