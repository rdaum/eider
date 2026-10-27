//! LFM3 BF16 operator boundaries with F32 accumulation and storage.
use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, Error, Result};
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_lfm3_norm(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        rows: u32,
        width: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm3_conv(
        x: *const f32,
        weight: *const f32,
        state: *mut f32,
        out: *mut f32,
        rows: u32,
        hidden: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm3_rope(
        x: *const f32,
        freq: *const f32,
        out: *mut f32,
        rows: u32,
        heads: u32,
        dim: u32,
        start: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm3_silu(
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm3_head_gate(
        gate: *const f32,
        x: *const f32,
        out: *mut f32,
        heads: u32,
        dim: u32,
        stream: *mut c_void,
    ) -> i32;
}
fn invalid() -> Error {
    Error::Format {
        label: "LFM3 operator",
        detail: "invalid dimensions or buffer lengths".into(),
    }
}
fn size(rows: usize, width: usize) -> Result<usize> {
    rows.checked_mul(width)
        .filter(|&n| rows > 0 && width > 0 && n <= u32::MAX as usize)
        .ok_or_else(invalid)
}
/// RMSNorm rounds the normalized value before multiplying the BF16 weight.
#[allow(clippy::too_many_arguments)]
pub fn norm(
    rows: usize,
    width: usize,
    x: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let n = size(rows, width)?;
    if n > x.len() || n > out.len() || weight.len() != width || !eps.is_finite() || eps <= 0. {
        return Err(invalid());
    }
    unsafe {
        check_cuda(
            "infer_lfm3_norm",
            infer_lfm3_norm(
                x.ptr,
                weight.ptr,
                out.buffer_mut().ptr,
                rows as u32,
                width as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}
/// Gated causal convolution with BF16 products, output and retained history.
#[allow(clippy::too_many_arguments)]
pub fn short_conv(
    x: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut out: DeviceOutput<'_, f32>,
    hidden: usize,
    width: usize,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = size(rows, hidden)?;
    if !(2..=32).contains(&width)
        || rows > 65535
        || n.checked_mul(3).is_none_or(|n| n > x.len())
        || n > out.len()
        || hidden.checked_mul(width) != Some(weight.len())
        || hidden.checked_mul(width - 1) != Some(state.len())
    {
        return Err(invalid());
    }
    unsafe {
        check_cuda(
            "infer_lfm3_conv",
            infer_lfm3_conv(
                x.ptr,
                weight.ptr,
                state.buffer_mut().ptr,
                out.buffer_mut().ptr,
                rows as u32,
                hidden as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}
/// Full-head NeoX pairing with BF16 trigonometric values and zero-tail frequencies.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    rows: usize,
    heads: usize,
    dim: usize,
    x: &DeviceBuffer<f32>,
    freq: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    start: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = size(size(rows, heads)?, dim)?;
    if !dim.is_multiple_of(2)
        || freq.len() != dim / 2
        || n > x.len()
        || n > out.len()
        || start
            .checked_add(rows)
            .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(invalid());
    }
    unsafe {
        check_cuda(
            "infer_lfm3_rope",
            infer_lfm3_rope(
                x.ptr,
                freq.ptr,
                out.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                dim as u32,
                start as u32,
                stream.as_raw(),
            ),
        )
    }
}
/// BF16 SiLU followed by BF16 multiplication.
pub fn silu(
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    count: usize,
    stream: &CudaStream,
) -> Result<()> {
    size(count, 1)?;
    if count > gate.len() || count > up.len() || count > out.len() {
        return Err(invalid());
    }
    unsafe {
        check_cuda(
            "infer_lfm3_silu",
            infer_lfm3_silu(
                gate.ptr,
                up.ptr,
                out.buffer_mut().ptr,
                count as u32,
                stream.as_raw(),
            ),
        )
    }
}
/// BF16 sigmoid gate broadcast across each BF16 attention head.
pub fn head_gate(
    gate: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    heads: usize,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = size(heads, dim)?;
    if heads > gate.len() || n > x.len() || n > out.len() {
        return Err(invalid());
    }
    unsafe {
        check_cuda(
            "infer_lfm3_head_gate",
            infer_lfm3_head_gate(
                gate.ptr,
                x.ptr,
                out.buffer_mut().ptr,
                heads as u32,
                dim as u32,
                stream.as_raw(),
            ),
        )
    }
}
