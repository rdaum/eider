#![allow(clippy::too_many_arguments)]

//! CUDA kernels for non-GEMM decode operations.

use crate::cuda::{
    CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda, max_shared_memory_per_block,
};
use crate::error::{Error, Result};
use crate::ffi;
use crate::format;
use crate::matrix::{Bf16Matrix, Nvfp4Matrix};
use std::mem::size_of;

/// Builds a pointer table in stream order, repeating each input row `repeats` times.
pub fn repeat_row_pointer_table_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut table: DeviceOutput<'_, *const f32>,
    routes: usize,
    repeats: usize,
    row_stride: usize,
    stream: &CudaStream,
) -> Result<()> {
    let rows = routes
        .checked_div(repeats)
        .filter(|_| routes.checked_rem(repeats) == Some(0));
    if routes == 0
        || repeats == 0
        || row_stride == 0
        || routes > u32::MAX as usize
        || repeats > u32::MAX as usize
        || row_stride > u32::MAX as usize
        || rows.is_none_or(|rows| input.len() < rows.saturating_mul(row_stride))
        || table.len() < routes
    {
        return Err(Error::Shape {
            label: "repeated row pointer table",
            expected: format!(
                "routes divisible by repeats, input>={} and table>={routes}",
                routes
                    .checked_div(repeats)
                    .unwrap_or_default()
                    .saturating_mul(row_stride)
            ),
            actual: format!(
                "routes={routes} repeats={repeats} row_stride={row_stride} input={} table={}",
                input.len(),
                table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_repeat_row_pointer_table_f32_on_stream",
            ffi::infer_repeat_row_pointer_table_f32_on_stream(
                input.ptr,
                table.buffer_mut().ptr,
                routes as u32,
                repeats as u32,
                row_stride as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues row-wise RMSNorm into an existing output buffer on `stream`.
pub fn rms_norm_f32_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "RMSNorm input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() < input_len || output.len() < input_len {
        return Err(Error::Shape {
            label: "RMSNorm buffers",
            expected: format!("{input_len} values"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    if weight.len() != cols {
        return Err(Error::Shape {
            label: "RMSNorm weight",
            expected: format!("{cols} values"),
            actual: format!("{} values", weight.len()),
        });
    }
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "RMSNorm dimensions",
            expected: "u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }

    unsafe {
        check_cuda(
            "infer_rms_norm_f32_on_stream",
            ffi::infer_rms_norm_f32_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies row-wise RMSNorm and adds a residual without materializing the normalized input.
pub fn rms_norm_add_f32_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    if len == 0
        || input.len() < len
        || weight.len() != cols
        || residual.len() < len
        || output.len() < len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(Error::Shape {
            label: "RMSNorm residual-add buffers",
            expected: format!("input/residual/output={len} weight={cols} with valid dimensions"),
            actual: format!(
                "input={} weight={} residual={} output={} eps={eps}",
                input.len(),
                weight.len(),
                residual.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_rms_norm_add_f32_on_stream",
            ffi::infer_rms_norm_add_f32_on_stream(
                input.ptr,
                weight.ptr,
                residual.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Adds two independently RMS-normalized row sets in one pass.
#[allow(clippy::too_many_arguments)]
pub fn dual_rms_norm_add_f32_into_on_stream(
    rows: usize,
    cols: usize,
    left: &DeviceBuffer<f32>,
    left_weight: &DeviceBuffer<f32>,
    left_eps: f32,
    right: &DeviceBuffer<f32>,
    right_weight: &DeviceBuffer<f32>,
    right_eps: f32,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    if len == 0
        || left.len() < len
        || left_weight.len() != cols
        || right.len() < len
        || right_weight.len() != cols
        || output.len() < len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !left_eps.is_finite()
        || left_eps < 0.0
        || !right_eps.is_finite()
        || right_eps < 0.0
    {
        return Err(Error::Shape {
            label: "dual RMSNorm add buffers",
            expected: format!("left/right/output={len} weights={cols} with valid dimensions"),
            actual: format!(
                "left={} left_weight={} right={} right_weight={} output={} eps={left_eps}/{right_eps}",
                left.len(),
                left_weight.len(),
                right.len(),
                right_weight.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_dual_rms_norm_add_f32_on_stream",
            ffi::infer_dual_rms_norm_add_f32_on_stream(
                left.ptr,
                left_weight.ptr,
                right.ptr,
                right_weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                left_eps,
                right_eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies RMSNorm, adds a residual, then applies channel and row scales.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_add_channel_row_scale_f32_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    channel_scale: &DeviceBuffer<f32>,
    row_scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    if len == 0
        || input.len() < len
        || weight.len() != cols
        || residual.len() < len
        || channel_scale.len() != cols
        || row_scale.len() < rows
        || output.len() < len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(Error::Shape {
            label: "scaled RMSNorm residual-add buffers",
            expected: format!(
                "input/residual/output={len}, weight/channel={cols}, row_scale={rows}"
            ),
            actual: format!(
                "input={} weight={} residual={} channel={} row_scale={} output={} eps={eps}",
                input.len(),
                weight.len(),
                residual.len(),
                channel_scale.len(),
                row_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_rms_norm_add_channel_row_scale_f32_on_stream",
            ffi::infer_rms_norm_add_channel_row_scale_f32_on_stream(
                input.ptr,
                weight.ptr,
                residual.ptr,
                channel_scale.ptr,
                row_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies RMSNorm and a residual add, then RMS-normalizes and quantizes the result.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_add_then_rms_norm_quantize_nvfp4_f32_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    input_weight: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    input_eps: f32,
    quant_weight: &DeviceBuffer<f32>,
    quant_output: &mut Nvfp4Matrix,
    quant_eps: f32,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "fused RMSNorm residual-add quantization input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let shared_bytes = cols
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::Shape {
            label: "fused RMSNorm residual-add quantization shared memory",
            expected: "cols * sizeof(f32) without overflow".to_string(),
            actual: format!("cols={cols}"),
        })?;
    if rows == 0
        || cols == 0
        || input.len() < len
        || input_weight.len() != cols
        || residual.len() < len
        || output.len() < len
        || quant_weight.len() != cols
        || quant_output.rows != cols
        || quant_output.cols < rows
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || shared_bytes > max_shared_memory_per_block()?
        || !input_eps.is_finite()
        || input_eps < 0.0
        || !quant_eps.is_finite()
        || quant_eps < 0.0
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "fused RMSNorm residual-add quantization buffers",
            expected: format!(
                "input/residual/output={len}, weights={cols}, quant_output={cols}x{rows} with valid dimensions and scales"
            ),
            actual: format!(
                "input={} input_weight={} residual={} output={} quant_weight={} quant_output={}x{} shared_bytes={shared_bytes} input_eps={input_eps} quant_eps={quant_eps} input_scale={input_scale}",
                input.len(),
                input_weight.len(),
                residual.len(),
                output.len(),
                quant_weight.len(),
                quant_output.rows,
                quant_output.cols,
            ),
        });
    }
    let mut quant_output = quant_output.output();
    unsafe {
        check_cuda(
            "infer_rms_norm_add_then_rms_norm_quantize_nvfp4_f32_on_stream",
            ffi::infer_rms_norm_add_then_rms_norm_quantize_nvfp4_f32_on_stream(
                input.ptr,
                input_weight.ptr,
                residual.ptr,
                output.buffer_mut().ptr,
                quant_weight.ptr,
                quant_output.values_mut_ptr().cast(),
                quant_output.scales_mut_ptr().cast(),
                rows as u32,
                cols as u32,
                input_eps,
                quant_eps,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies the two Gemma feed-forward RMS paths and final scaled residual in one pass.
#[allow(clippy::too_many_arguments)]
pub fn dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_into_on_stream(
    rows: usize,
    cols: usize,
    left: &DeviceBuffer<f32>,
    left_weight: &DeviceBuffer<f32>,
    left_eps: f32,
    right: &DeviceBuffer<f32>,
    right_weight: &DeviceBuffer<f32>,
    right_eps: f32,
    final_weight: &DeviceBuffer<f32>,
    final_eps: f32,
    residual: &DeviceBuffer<f32>,
    channel_scale: &DeviceBuffer<f32>,
    row_scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "fused dual RMSNorm final residual input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let shared_bytes = cols
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::Shape {
            label: "fused dual RMSNorm final residual shared memory",
            expected: "cols * sizeof(f32) without overflow".to_string(),
            actual: format!("cols={cols}"),
        })?;
    if rows == 0
        || cols == 0
        || left.len() < len
        || left_weight.len() != cols
        || right.len() < len
        || right_weight.len() != cols
        || final_weight.len() != cols
        || residual.len() < len
        || channel_scale.len() != cols
        || row_scale.len() < rows
        || output.len() < len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || shared_bytes > max_shared_memory_per_block()?
        || !left_eps.is_finite()
        || left_eps < 0.0
        || !right_eps.is_finite()
        || right_eps < 0.0
        || !final_eps.is_finite()
        || final_eps < 0.0
    {
        return Err(Error::Shape {
            label: "fused dual RMSNorm final residual buffers",
            expected: format!(
                "left/right/residual/output={len}, weights/channel={cols}, row_scale={rows} with valid dimensions"
            ),
            actual: format!(
                "left={} left_weight={} right={} right_weight={} final_weight={} residual={} channel={} row_scale={} output={} shared_bytes={shared_bytes} eps={left_eps}/{right_eps}/{final_eps}",
                left.len(),
                left_weight.len(),
                right.len(),
                right_weight.len(),
                final_weight.len(),
                residual.len(),
                channel_scale.len(),
                row_scale.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_on_stream",
            ffi::infer_dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_on_stream(
                left.ptr,
                left_weight.ptr,
                right.ptr,
                right_weight.ptr,
                final_weight.ptr,
                residual.ptr,
                channel_scale.ptr,
                row_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                left_eps,
                right_eps,
                final_eps,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn rms_norm_rope_neox_f32_indexed_into_on_stream(
    rows: usize,
    head_dim: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    position: &DeviceBuffer<u32>,
    theta: f32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "RMSNorm RoPE input",
        expected: "rows * head_dim without overflow".to_string(),
        actual: format!("rows={rows} head_dim={head_dim}"),
    })?;
    if input.len() != input_len
        || output.len() != input_len
        || weight.len() != head_dim
        || position.len() != 1
    {
        return Err(Error::Shape {
            label: "RMSNorm RoPE buffers",
            expected: format!("input/output={input_len} weight={head_dim} position=1"),
            actual: format!(
                "input={} output={} weight={} position={}",
                input.len(),
                output.len(),
                weight.len(),
                position.len()
            ),
        });
    }
    if rows > u32::MAX as usize || head_dim > u32::MAX as usize || !head_dim.is_multiple_of(2) {
        return Err(Error::Shape {
            label: "RMSNorm RoPE dimensions",
            expected: "u32-sized rows and even head_dim".to_string(),
            actual: format!("rows={rows} head_dim={head_dim}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_rms_norm_rope_neox_f32_indexed_on_stream",
            ffi::infer_rms_norm_rope_neox_f32_indexed_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                position.ptr,
                theta,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues `silu(gate) * up` into an existing output buffer on `stream`.
pub fn silu_mul_f32_into_on_stream(
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if gate.len() != up.len() || output.len() != gate.len() {
        return Err(Error::Shape {
            label: "SiLU multiply buffers",
            expected: format!("{} values", gate.len()),
            actual: format!(
                "gate={} up={} output={}",
                gate.len(),
                up.len(),
                output.len()
            ),
        });
    }
    silu_mul_f32_prefix_into_on_stream(gate, up, output, gate.len(), stream)
}

/// Enqueues `silu(gate) * up` for an active prefix on `stream`.
pub fn silu_mul_f32_prefix_into_on_stream(
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0
        || len > u32::MAX as usize
        || gate.len() < len
        || up.len() < len
        || output.len() < len
    {
        return Err(Error::Shape {
            label: "SiLU multiply prefix",
            expected: format!("gate/up/output at least {len} values"),
            actual: format!(
                "gate={} up={} output={} active={len}",
                gate.len(),
                up.len(),
                output.len()
            ),
        });
    }

    unsafe {
        check_cuda(
            "infer_silu_mul_f32_on_stream",
            ffi::infer_silu_mul_f32_on_stream(
                gate.ptr,
                up.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies Gemma's tanh-approximated GELU activation on `stream`.
pub fn gelu_tanh_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty() || input.len() != output.len() || input.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "GELU-tanh buffers",
            expected: "equal non-empty u32-sized input and output".to_string(),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_gelu_tanh_f32_on_stream",
            ffi::infer_gelu_tanh_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies `GELU-tanh(gate) * up` on `stream`.
pub fn gelu_tanh_mul_f32_into_on_stream(
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if gate.is_empty()
        || gate.len() > u32::MAX as usize
        || gate.len() != up.len()
        || output.len() != gate.len()
    {
        return Err(Error::Shape {
            label: "GELU-tanh multiply buffers",
            expected: "equal non-empty u32-sized gate, up, and output".to_string(),
            actual: format!(
                "gate={} up={} output={}",
                gate.len(),
                up.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gelu_tanh_mul_f32_on_stream",
            ffi::infer_gelu_tanh_mul_f32_on_stream(
                gate.ptr,
                up.ptr,
                output.buffer_mut().ptr,
                gate.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies `GELU-tanh(gate) * up` to a concatenated `[gate, up]` vector.
pub fn gelu_tanh_mul_halves_f32_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0 || len > u32::MAX as usize || gate_up.len() != len * 2 || output.len() != len {
        return Err(Error::Shape {
            label: "GELU-tanh multiply halves buffers",
            expected: format!("gate_up={} output={len}", len * 2),
            actual: format!("gate_up={} output={}", gate_up.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_gelu_tanh_mul_halves_f32_on_stream",
            ffi::infer_gelu_tanh_mul_halves_f32_on_stream(
                gate_up.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn silu_mul_halves_f32_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if gate_up.len() != len * 2 || output.len() != len {
        return Err(Error::Shape {
            label: "SiLU multiply halves buffers",
            expected: format!("gate_up={} output={len}", len * 2),
            actual: format!("gate_up={} output={}", gate_up.len(), output.len()),
        });
    }
    if len == 0 || len > u32::MAX as usize {
        return Err(Error::Shape {
            label: "SiLU multiply halves",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{len} values"),
        });
    }

    unsafe {
        check_cuda(
            "infer_silu_mul_halves_f32_on_stream",
            ffi::infer_silu_mul_halves_f32_on_stream(
                gate_up.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies the Step SwiGLU clamp followed by `SiLU(gate) * up`.
pub fn silu_mul_halves_clamped_f32_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    limit: f32,
    stream: &CudaStream,
) -> Result<()> {
    if gate_up.len() != len * 2 || output.len() != len || len == 0 || len > u32::MAX as usize {
        return Err(Error::Shape {
            label: "clamped SiLU multiply halves",
            expected: format!("gate_up={} output={len}", len * 2),
            actual: format!("gate_up={} output={}", gate_up.len(), output.len()),
        });
    }
    if !limit.is_finite() || limit <= 0.0 {
        return Err(Error::Format {
            label: "clamped SiLU multiply halves limit",
            detail: format!("expected a positive finite limit, got {limit}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_silu_mul_halves_clamped_f32_on_stream",
            ffi::infer_silu_mul_halves_clamped_f32_on_stream(
                gate_up.ptr,
                output.buffer_mut().ptr,
                len as u32,
                limit,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies `SiLU(gate) * up` independently to row-major concatenated rows.
pub fn silu_mul_halves_f32_batch_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(2))
        .unwrap_or(usize::MAX);
    let output_len = rows.saturating_mul(cols);
    if rows == 0
        || cols == 0
        || gate_up.len() < input_len
        || output.len() < output_len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched SiLU halves buffers",
            expected: format!("gate_up={input_len} output={output_len}"),
            actual: format!(
                "gate_up={} output={} rows={rows} cols={cols}",
                gate_up.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_silu_mul_halves_f32_batch_on_stream",
            ffi::infer_silu_mul_halves_f32_batch_on_stream(
                gate_up.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies the Step SwiGLU clamp to row-major concatenated gate/up rows.
pub fn silu_mul_halves_clamped_f32_batch_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    limit: f32,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(2))
        .unwrap_or(usize::MAX);
    let output_len = rows.saturating_mul(cols);
    if rows == 0
        || cols == 0
        || gate_up.len() < input_len
        || output.len() < output_len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched clamped SiLU halves buffers",
            expected: format!("gate_up>={input_len} output>={output_len}"),
            actual: format!(
                "gate_up={} output={} rows={rows} cols={cols}",
                gate_up.len(),
                output.len()
            ),
        });
    }
    if !limit.is_finite() || limit <= 0.0 {
        return Err(Error::Format {
            label: "batched clamped SiLU halves limit",
            detail: format!("expected a positive finite limit, got {limit}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_silu_mul_halves_clamped_f32_batch_on_stream",
            ffi::infer_silu_mul_halves_clamped_f32_batch_on_stream(
                gate_up.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                limit,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn fill_f32_into_on_stream(
    output: DeviceOutput<'_, f32>,
    value: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = output.len();
    fill_f32_prefix_into_on_stream(output, value, len, stream)
}

/// Fills an active prefix of an F32 buffer on `stream`.
pub fn fill_f32_prefix_into_on_stream(
    mut output: DeviceOutput<'_, f32>,
    value: f32,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0 || len > u32::MAX as usize || output.len() < len || !value.is_finite() {
        return Err(Error::Shape {
            label: "fill f32 prefix",
            expected: format!("output at least {len} values and finite value"),
            actual: format!("output={} active={len} value={value}", output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_fill_f32_on_stream",
            ffi::infer_fill_f32_on_stream(
                output.buffer_mut().ptr,
                value,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn scaled_add_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceInOut<'_, f32>,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    if input.len() != output.len()
        || input.is_empty()
        || input.len() > u32::MAX as usize
        || !scale.is_finite()
    {
        return Err(Error::Shape {
            label: "scaled add f32",
            expected: "matching non-empty u32-sized buffers and finite scale".to_string(),
            actual: format!(
                "input={} output={} scale={scale}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_scaled_add_f32_on_stream",
            ffi::infer_scaled_add_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                scale,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Splits a Qwen3.6 full-attention Q projection into `[query, gate]` halves.
pub fn split_q_gate_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut q: DeviceOutput<'_, f32>,
    mut gate: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if q.len() != gate.len() || input.len() != q.len() * 2 {
        return Err(Error::Shape {
            label: "split q/gate buffers",
            expected: format!("input={} q={} gate={}", q.len() * 2, q.len(), gate.len()),
            actual: format!("input={} q={} gate={}", input.len(), q.len(), gate.len()),
        });
    }
    if q.is_empty() || q.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "split q/gate dimensions",
            expected: "non-empty u32-sized q/gate".to_string(),
            actual: format!("q={} gate={}", q.len(), gate.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_split_q_gate_f32_on_stream",
            ffi::infer_split_q_gate_f32_on_stream(
                input.ptr,
                q.buffer_mut().ptr,
                gate.buffer_mut().ptr,
                q.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes `output = input * sigmoid(gate)` elementwise.
pub fn sigmoid_mul_f32_into_on_stream(
    gate: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    sigmoid_mul_f32_prefix_into_on_stream(gate, input, output, input.len(), stream)
}

/// Computes `output = input * sigmoid(gate)` for an active prefix.
pub fn sigmoid_mul_f32_prefix_into_on_stream(
    gate: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if gate.len() < len || input.len() < len || output.len() < len {
        return Err(Error::Shape {
            label: "sigmoid multiply buffers",
            expected: format!("gate/input/output at least {len} values"),
            actual: format!(
                "gate={} input={} output={}",
                gate.len(),
                input.len(),
                output.len()
            ),
        });
    }
    if len == 0 || len > u32::MAX as usize {
        return Err(Error::Shape {
            label: "sigmoid multiply dimensions",
            expected: "non-empty u32-sized length".to_string(),
            actual: format!("len={len}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_sigmoid_mul_f32_on_stream",
            ffi::infer_sigmoid_mul_f32_on_stream(
                gate.ptr,
                input.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Broadcasts one sigmoid gate per attention head across its head dimension.
pub fn sigmoid_scale_heads_f32_into_on_stream(
    gate: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    sigmoid_scale_heads_f32_prefix_into_on_stream(gate, input, output, gate.len(), head_dim, stream)
}

/// Broadcasts an active prefix of sigmoid head gates across their head dimension.
pub fn sigmoid_scale_heads_f32_prefix_into_on_stream(
    gate: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = heads.saturating_mul(head_dim);
    if gate.is_empty()
        || heads == 0
        || head_dim == 0
        || gate.len() < heads
        || input.len() < len
        || output.len() < len
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "sigmoid head-gate buffers",
            expected: format!("gate>={heads} input/output>={len}"),
            actual: format!(
                "gate={} input={} output={} heads={heads} head_dim={head_dim}",
                gate.len(),
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sigmoid_scale_heads_f32_on_stream",
            ffi::infer_sigmoid_scale_heads_f32_on_stream(
                gate.ptr,
                input.ptr,
                output.buffer_mut().ptr,
                heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Broadcasts one softplus gate per attention head across its head dimension.
pub fn softplus_scale_heads_f32_into_on_stream(
    gate: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    softplus_scale_heads_f32_prefix_into_on_stream(
        gate,
        input,
        output,
        head_dim,
        gate.len(),
        stream,
    )
}

/// Broadcasts an active prefix of softplus head gates across head dimensions.
pub fn softplus_scale_heads_f32_prefix_into_on_stream(
    gate: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    head_dim: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = heads.saturating_mul(head_dim);
    if heads == 0
        || head_dim == 0
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || gate.len() < heads
        || input.len() < values
        || output.len() < values
    {
        return Err(Error::Shape {
            label: "softplus head-gate prefix buffers",
            expected: format!("gate>={heads} input/output>={values}"),
            actual: format!(
                "gate={} input={} output={} heads={heads} head_dim={head_dim}",
                gate.len(),
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_softplus_scale_heads_f32_on_stream",
            ffi::infer_softplus_scale_heads_f32_on_stream(
                gate.ptr,
                input.ptr,
                output.buffer_mut().ptr,
                heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes `output = input * sigmoid(gate_logit[0])` elementwise, reading a
/// single scalar gate and broadcasting it. Used for the Qwen3.6 shared-expert
/// gate, replacing a host readback + broadcast + sigmoid_mul sequence.
pub fn sigmoid_scale_scalar_f32_into_on_stream(
    gate_logit: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if gate_logit.len() != 1 {
        return Err(Error::Shape {
            label: "sigmoid scale scalar gate",
            expected: "1 value".to_string(),
            actual: format!("{} values", gate_logit.len()),
        });
    }
    if output.len() != input.len() {
        return Err(Error::Shape {
            label: "sigmoid scale scalar buffers",
            expected: format!("input=output={} values", input.len()),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    if input.is_empty() || input.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "sigmoid scale scalar dimensions",
            expected: "non-empty u32-sized length".to_string(),
            actual: format!("len={}", input.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_sigmoid_scale_scalar_f32_on_stream",
            ffi::infer_sigmoid_scale_scalar_f32_on_stream(
                gate_logit.ptr,
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Splits Qwen3.6 full-attention Q/G projection and RMS-normalizes Q/K heads.
///
/// Qwen3Next lays out q_proj output per query head as `[query_head, gate_head]`.
/// This helper preserves that interleaving while producing contiguous query and
/// gate buffers. RoPE/MRoPE is intentionally not applied here.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_full_attn_prep_f32_into_on_stream(
    q_full: &DeviceBuffer<f32>,
    k_raw: &DeviceBuffer<f32>,
    q_norm: &DeviceBuffer<f32>,
    k_norm: &DeviceBuffer<f32>,
    mut q: DeviceOutput<'_, f32>,
    mut gate: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let q_width = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "Qwen3.6 full-attn prep q",
        expected: "q_heads * head_dim without overflow".to_string(),
        actual: format!("q_heads={q_heads} head_dim={head_dim}"),
    })?;
    let q_full_len = q_width.checked_mul(2).ok_or_else(|| Error::Shape {
        label: "Qwen3.6 full-attn prep q_full",
        expected: "q_width * 2 without overflow".to_string(),
        actual: format!("q_width={q_width}"),
    })?;
    let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "Qwen3.6 full-attn prep k",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    if q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || !head_dim.is_power_of_two()
        || head_dim > 1024
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Qwen3.6 full-attn prep dimensions",
            expected: "non-zero u32-sized heads and power-of-two head_dim <= 1024".to_string(),
            actual: format!("q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"),
        });
    }
    if q_full.len() != q_full_len
        || k_raw.len() != kv_width
        || q_norm.len() != head_dim
        || k_norm.len() != head_dim
        || q.len() != q_width
        || gate.len() != q_width
        || k.len() != kv_width
    {
        return Err(Error::Shape {
            label: "Qwen3.6 full-attn prep buffers",
            expected: format!(
                "q_full={q_full_len} k_raw/k={kv_width} q/gate={q_width} norms={head_dim}"
            ),
            actual: format!(
                "q_full={} k_raw={} q_norm={} k_norm={} q={} gate={} k={}",
                q_full.len(),
                k_raw.len(),
                q_norm.len(),
                k_norm.len(),
                q.len(),
                gate.len(),
                k.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_full_attn_prep_f32_on_stream",
            ffi::infer_qwen36_full_attn_prep_f32_on_stream(
                q_full.ptr,
                k_raw.ptr,
                q_norm.ptr,
                k_norm.ptr,
                q.buffer_mut().ptr,
                gate.buffer_mut().ptr,
                k.buffer_mut().ptr,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Splits and QK-normalizes full-attention projections for a dense batch.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_full_attn_prep_f32_batch_into_on_stream(
    q_full: &DeviceBuffer<f32>,
    k_raw: &DeviceBuffer<f32>,
    q_norm: &DeviceBuffer<f32>,
    k_norm: &DeviceBuffer<f32>,
    mut q: DeviceOutput<'_, f32>,
    mut gate: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let q_width = q_heads.saturating_mul(head_dim);
    let kv_width = kv_heads.saturating_mul(head_dim);
    let q_len = rows.saturating_mul(q_width);
    let q_full_len = q_len.saturating_mul(2);
    let kv_len = rows.saturating_mul(kv_width);
    if rows == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || !head_dim.is_power_of_two()
        || head_dim > 1024
        || q_full.len() < q_full_len
        || k_raw.len() < kv_len
        || q_norm.len() != head_dim
        || k_norm.len() != head_dim
        || q.len() < q_len
        || gate.len() < q_len
        || k.len() < kv_len
        || rows > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Qwen3.6 batched full-attn prep buffers",
            expected: format!("q_full={q_full_len} k={kv_len} q/gate={q_len} norms={head_dim}"),
            actual: format!(
                "q_full={} k_raw={} q_norm={} k_norm={} q={} gate={} k={} rows={rows}",
                q_full.len(),
                k_raw.len(),
                q_norm.len(),
                k_norm.len(),
                q.len(),
                gate.len(),
                k.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_full_attn_prep_f32_batch_on_stream",
            ffi::infer_qwen36_full_attn_prep_f32_batch_on_stream(
                q_full.ptr,
                k_raw.ptr,
                q_norm.ptr,
                k_norm.ptr,
                q.buffer_mut().ptr,
                gate.buffer_mut().ptr,
                k.buffer_mut().ptr,
                rows as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn split_qkv_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if k.len() != v.len() || input.len() != q.len() + k.len() + v.len() {
        return Err(Error::Shape {
            label: "split qkv buffers",
            expected: format!(
                "input={} q={} k={} v={}",
                q.len() + k.len() + v.len(),
                q.len(),
                k.len(),
                v.len()
            ),
            actual: format!(
                "input={} q={} k={} v={}",
                input.len(),
                q.len(),
                k.len(),
                v.len()
            ),
        });
    }
    if q.is_empty() || k.is_empty() || q.len() > u32::MAX as usize || k.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "split qkv dimensions",
            expected: "non-empty u32-sized q/k/v".to_string(),
            actual: format!("q={} k={} v={}", q.len(), k.len(), v.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_split_qkv_f32_on_stream",
            ffi::infer_split_qkv_f32_on_stream(
                input.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                q.len() as u32,
                k.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Splits row-major fused Q/K/V rows into three row-major batches on `stream`.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    batch_rows: usize,
    q_width: usize,
    kv_width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = batch_rows
        .checked_mul(q_width + 2 * kv_width)
        .ok_or_else(|| Error::Shape {
            label: "split QKV batch",
            expected: "batch_rows * fused_width without overflow".to_string(),
            actual: format!("batch_rows={batch_rows} q_width={q_width} kv_width={kv_width}"),
        })?;
    if batch_rows == 0
        || q_width == 0
        || kv_width == 0
        || input.len() != input_len
        || q.len() != batch_rows * q_width
        || k.len() != batch_rows * kv_width
        || v.len() != batch_rows * kv_width
        || batch_rows > u32::MAX as usize
        || q_width > u32::MAX as usize
        || kv_width > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "split QKV batch",
            expected: format!(
                "input={input_len} q={} k={} v={}",
                batch_rows * q_width,
                batch_rows * kv_width,
                batch_rows * kv_width,
            ),
            actual: format!(
                "input={} q={} k={} v={} batch_rows={batch_rows}",
                input.len(),
                q.len(),
                k.len(),
                v.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_split_qkv_f32_batch_on_stream",
            ffi::infer_split_qkv_f32_batch_on_stream(
                input.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                batch_rows as u32,
                q_width as u32,
                kv_width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues MoE softmax top-k routing into fixed device output buffers.
///
/// `out_indices` and `out_weights` must both have length `k`. Weights match
/// the host reference path: softmax probabilities over all experts, optionally
/// renormalized over the selected top-k experts.
pub fn moe_topk_f32_into_on_stream(
    logits: &DeviceBuffer<f32>,
    mut out_indices: DeviceOutput<'_, u32>,
    mut out_weights: DeviceOutput<'_, f32>,
    k: usize,
    norm_topk_prob: bool,
    stream: &CudaStream,
) -> Result<()> {
    if logits.is_empty()
        || k == 0
        || k > logits.len()
        || out_indices.len() != k
        || out_weights.len() != k
    {
        return Err(Error::Shape {
            label: "MoE top-k buffers",
            expected: "0 < k <= experts and k-sized outputs".to_string(),
            actual: format!(
                "experts={} k={} indices={} weights={}",
                logits.len(),
                k,
                out_indices.len(),
                out_weights.len()
            ),
        });
    }
    if logits.len() > u32::MAX as usize || k > u32::MAX as usize {
        return Err(Error::Shape {
            label: "MoE top-k dimensions",
            expected: "u32-sized experts and k".to_string(),
            actual: format!("experts={} k={k}", logits.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_moe_topk_f32_on_stream",
            ffi::infer_moe_topk_f32_on_stream(
                logits.ptr,
                out_indices.buffer_mut().ptr,
                out_weights.buffer_mut().ptr,
                logits.len() as u32,
                k as u32,
                i32::from(norm_topk_prob),
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues Step-3.5 sigmoid routing with biased top-8 selection.
///
/// Selection ranks `sigmoid(logit) + bias`; output weights use the original
/// selected sigmoid probabilities, normalized to sum to 3.
pub fn step37_sigmoid_top8_f32_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut out_indices: DeviceOutput<'_, u32>,
    mut out_weights: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if logits.len() < 8
        || logits.len() != bias.len()
        || out_indices.len() != 8
        || out_weights.len() != 8
        || logits.len() > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Step-3.5 sigmoid top-8 buffers",
            expected: "matching logits/bias with at least 8 experts and top-8 outputs".to_string(),
            actual: format!(
                "logits={} bias={} indices={} weights={}",
                logits.len(),
                bias.len(),
                out_indices.len(),
                out_weights.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_step37_sigmoid_top8_f32_on_stream",
            ffi::infer_step37_sigmoid_top8_f32_on_stream(
                logits.ptr,
                bias.ptr,
                out_indices.buffer_mut().ptr,
                out_weights.buffer_mut().ptr,
                logits.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues Nemotron 3 grouped sigmoid routing with correction bias.
///
/// Selection uses `sigmoid(logit) + bias`; returned weights use the original
/// sigmoid probabilities, optionally normalized over the selected experts,
/// and multiplied by `scaling_factor`.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_sigmoid_topk_f32_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut out_indices: DeviceOutput<'_, u32>,
    mut out_weights: DeviceOutput<'_, f32>,
    k: usize,
    groups: usize,
    topk_groups: usize,
    normalize: bool,
    scaling_factor: f32,
    stream: &CudaStream,
) -> Result<()> {
    if logits.is_empty()
        || logits.len() > 512
        || logits.len() != bias.len()
        || k == 0
        || k > logits.len()
        || groups == 0
        || groups > 64
        || !logits.len().is_multiple_of(groups)
        || topk_groups == 0
        || topk_groups > groups
        || out_indices.len() != k
        || out_weights.len() != k
        || !scaling_factor.is_finite()
    {
        return Err(Error::Shape {
            label: "Nemotron 3 sigmoid top-k buffers",
            expected: "matching <=512 logits/bias; valid grouped top-k; k-sized outputs"
                .to_string(),
            actual: format!(
                "logits={} bias={} k={k} groups={groups} topk_groups={topk_groups} indices={} weights={} scale={scaling_factor}",
                logits.len(),
                bias.len(),
                out_indices.len(),
                out_weights.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_sigmoid_topk_f32_on_stream",
            ffi::infer_nemotron3_sigmoid_topk_f32_on_stream(
                logits.ptr,
                bias.ptr,
                out_indices.buffer_mut().ptr,
                out_weights.buffer_mut().ptr,
                logits.len() as u32,
                k as u32,
                groups as u32,
                topk_groups as u32,
                i32::from(normalize),
                scaling_factor,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues independent Nemotron 3 grouped sigmoid routing for dense rows.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_sigmoid_topk_f32_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut out_indices: DeviceOutput<'_, u32>,
    mut out_weights: DeviceOutput<'_, f32>,
    rows: usize,
    k: usize,
    groups: usize,
    topk_groups: usize,
    normalize: bool,
    scaling_factor: f32,
    stream: &CudaStream,
) -> Result<()> {
    let experts = bias.len();
    let logits_len = rows.saturating_mul(experts);
    let routes = rows.saturating_mul(k);
    if rows == 0
        || experts == 0
        || experts > 512
        || logits.len() < logits_len
        || k == 0
        || k > experts
        || groups == 0
        || groups > 64
        || !experts.is_multiple_of(groups)
        || topk_groups == 0
        || topk_groups > groups
        || out_indices.len() < routes
        || out_weights.len() < routes
        || rows > u32::MAX as usize
        || !scaling_factor.is_finite()
    {
        return Err(Error::Shape {
            label: "batched Nemotron 3 sigmoid top-k buffers",
            expected: format!(
                "logits={logits_len} bias={experts}; valid grouped top-k; outputs={routes}"
            ),
            actual: format!(
                "rows={rows} logits={} bias={} k={k} groups={groups} topk_groups={topk_groups} indices={} weights={} scale={scaling_factor}",
                logits.len(),
                bias.len(),
                out_indices.len(),
                out_weights.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_sigmoid_topk_f32_batch_on_stream",
            ffi::infer_nemotron3_sigmoid_topk_f32_batch_on_stream(
                logits.ptr,
                bias.ptr,
                out_indices.buffer_mut().ptr,
                out_weights.buffer_mut().ptr,
                rows as u32,
                experts as u32,
                k as u32,
                groups as u32,
                topk_groups as u32,
                i32::from(normalize),
                scaling_factor,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues independent Step sigmoid routing with biased top-8 selection.
pub fn step37_sigmoid_top8_f32_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut out_indices: DeviceOutput<'_, u32>,
    mut out_weights: DeviceOutput<'_, f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let logits_len = rows.saturating_mul(bias.len());
    let routes = rows.saturating_mul(8);
    if rows == 0
        || bias.len() < 8
        || logits.len() != logits_len
        || out_indices.len() != routes
        || out_weights.len() != routes
        || rows > u32::MAX as usize
        || bias.len() > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched Step sigmoid top-8 buffers",
            expected: format!("logits={logits_len} bias>=8 indices={routes} weights={routes}"),
            actual: format!(
                "logits={} bias={} indices={} weights={} rows={rows}",
                logits.len(),
                bias.len(),
                out_indices.len(),
                out_weights.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_step37_sigmoid_top8_f32_batch_on_stream",
            ffi::infer_step37_sigmoid_top8_f32_batch_on_stream(
                logits.ptr,
                bias.ptr,
                out_indices.buffer_mut().ptr,
                out_weights.buffer_mut().ptr,
                rows as u32,
                bias.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Remaps logical expert indices through a device-resident slot table.
///
/// Missing or out-of-range experts produce `u32::MAX` in the corresponding
/// output position.
pub fn remap_expert_indices_into_on_stream(
    expert_indices: &DeviceBuffer<u32>,
    expert_to_slot: &DeviceBuffer<u32>,
    slot_indices: DeviceOutput<'_, u32>,
    stream: &CudaStream,
) -> Result<()> {
    remap_expert_indices_at_offset_into_on_stream(
        expert_indices,
        0,
        expert_to_slot,
        slot_indices,
        stream,
    )
}

/// Remaps a contiguous logical-expert range through a device-resident slot table.
pub fn remap_expert_indices_at_offset_into_on_stream(
    expert_indices: &DeviceBuffer<u32>,
    expert_offset: usize,
    expert_to_slot: &DeviceBuffer<u32>,
    slot_indices: DeviceOutput<'_, u32>,
    stream: &CudaStream,
) -> Result<()> {
    let count = slot_indices.len();
    remap_expert_indices_range_into_on_stream(
        expert_indices,
        expert_offset,
        expert_to_slot,
        slot_indices,
        count,
        stream,
    )
}

/// Remaps a logical-expert range into an active prefix of a larger output.
pub fn remap_expert_indices_range_into_on_stream(
    expert_indices: &DeviceBuffer<u32>,
    expert_offset: usize,
    expert_to_slot: &DeviceBuffer<u32>,
    mut slot_indices: DeviceOutput<'_, u32>,
    count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let expert_end = expert_offset.saturating_add(count);
    if slot_indices.is_empty()
        || count == 0
        || count > slot_indices.len()
        || expert_to_slot.is_empty()
        || expert_end > expert_indices.len()
        || expert_offset > u32::MAX as usize
        || count > u32::MAX as usize
        || expert_to_slot.len() > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "expert slot remap",
            expected: "non-empty in-range source/output and non-empty expert table".to_string(),
            actual: format!(
                "indices={} offset={expert_offset} slots={} active={count} table={}",
                expert_indices.len(),
                slot_indices.len(),
                expert_to_slot.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_remap_expert_indices_on_stream",
            ffi::infer_remap_expert_indices_on_stream(
                expert_indices.ptr,
                expert_to_slot.ptr,
                slot_indices.buffer_mut().ptr,
                expert_offset as u32,
                count as u32,
                expert_to_slot.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Accumulates routed expert usage without copying route IDs to the host.
pub fn record_expert_indices_u64_on_stream(
    expert_indices: &DeviceBuffer<u32>,
    counts: DeviceInOut<'_, u64>,
    stream: &CudaStream,
) -> Result<()> {
    record_expert_indices_prefix_u64_on_stream(expert_indices, expert_indices.len(), counts, stream)
}

/// Accumulates a prefix of routed expert IDs without copying them to the host.
pub fn record_expert_indices_prefix_u64_on_stream(
    expert_indices: &DeviceBuffer<u32>,
    len: usize,
    mut counts: DeviceInOut<'_, u64>,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0
        || len > expert_indices.len()
        || counts.is_empty()
        || len > u32::MAX as usize
        || counts.len() > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "expert usage histogram",
            expected: "non-empty route-ID prefix and expert counts fitting u32".to_string(),
            actual: format!(
                "indices={} prefix={len} experts={}",
                expert_indices.len(),
                counts.len()
            ),
        });
    }
    let experts = counts.len();
    unsafe {
        check_cuda(
            "infer_record_expert_indices_u64_on_stream",
            ffi::infer_record_expert_indices_u64_on_stream(
                expert_indices.ptr,
                counts.buffer_mut().ptr,
                len as u32,
                experts as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Clears a device-resident expert usage histogram.
pub fn clear_expert_counts_u64_on_stream(
    mut counts: DeviceOutput<'_, u64>,
    stream: &CudaStream,
) -> Result<()> {
    if counts.is_empty() || counts.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "expert usage histogram",
            expected: "non-empty expert counts fitting u32".to_string(),
            actual: format!("experts={}", counts.len()),
        });
    }
    let experts = counts.len();
    unsafe {
        check_cuda(
            "infer_clear_expert_counts_u64_on_stream",
            ffi::infer_clear_expert_counts_u64_on_stream(
                counts.buffer_mut().ptr,
                experts as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues independent softmax top-k routing for a row-major batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_f32_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    mut out_indices: DeviceOutput<'_, u32>,
    mut out_weights: DeviceOutput<'_, f32>,
    rows: usize,
    experts: usize,
    k: usize,
    norm_topk_prob: bool,
    stream: &CudaStream,
) -> Result<()> {
    let routes = rows.saturating_mul(k);
    let logits_len = rows.saturating_mul(experts);
    if rows == 0
        || experts == 0
        || k == 0
        || k > experts
        || logits.len() < logits_len
        || out_indices.len() < routes
        || out_weights.len() < routes
        || rows > u32::MAX as usize
        || experts > u32::MAX as usize
        || k > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched MoE top-k buffers",
            expected: format!(
                "logits={logits_len} indices={routes} weights={routes} with 0 < k <= experts"
            ),
            actual: format!(
                "logits={} indices={} weights={} rows={rows} experts={experts} k={k}",
                logits.len(),
                out_indices.len(),
                out_weights.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_moe_topk_f32_batch_on_stream",
            ffi::infer_moe_topk_f32_batch_on_stream(
                logits.ptr,
                out_indices.buffer_mut().ptr,
                out_weights.buffer_mut().ptr,
                rows as u32,
                experts as u32,
                k as u32,
                i32::from(norm_topk_prob),
                stream.as_raw(),
            ),
        )
    }
}

/// Reusable device storage for expert-major MoE route ordering.
pub struct MoeSortedRoutes {
    capacity_routes: usize,
    routes: usize,
    experts: usize,
    expert_counts: DeviceBuffer<u32>,
    expert_offsets: DeviceBuffer<u32>,
    expert_cursors: DeviceBuffer<u32>,
    sorted_routes: DeviceBuffer<u32>,
    sorted_experts: DeviceBuffer<u32>,
    route_to_sorted: DeviceBuffer<u32>,
}

impl MoeSortedRoutes {
    /// Allocates route sorting storage for one fixed-capacity prompt batch.
    pub fn new(routes: usize, experts: usize) -> Result<Self> {
        if routes == 0
            || experts == 0
            || experts > 1024
            || routes > u32::MAX as usize
            || experts > u32::MAX as usize
        {
            return Err(Error::Shape {
                label: "MoE sorted routes",
                expected: "nonzero u32-sized routes and 1..=1024 experts".to_string(),
                actual: format!("routes={routes} experts={experts}"),
            });
        }
        Ok(Self {
            capacity_routes: routes,
            routes,
            experts,
            expert_counts: DeviceBuffer::zeroed(experts)?,
            expert_offsets: DeviceBuffer::zeroed(experts + 1)?,
            expert_cursors: DeviceBuffer::zeroed(experts)?,
            sorted_routes: DeviceBuffer::zeroed(routes)?,
            sorted_experts: DeviceBuffer::zeroed(routes)?,
            route_to_sorted: DeviceBuffer::zeroed(routes)?,
        })
    }

    /// Selects an active route prefix while retaining the allocated capacity.
    pub fn set_routes(&mut self, routes: usize) -> Result<()> {
        if routes == 0 || routes > self.capacity_routes {
            return Err(Error::Shape {
                label: "active MoE sorted routes",
                expected: format!("1..={}", self.capacity_routes),
                actual: routes.to_string(),
            });
        }
        self.routes = routes;
        Ok(())
    }

    /// Sorts `indices` into expert-major route order without host readback.
    pub fn sort_on_stream(
        &mut self,
        indices: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if indices.len() < self.routes {
            return Err(Error::Shape {
                label: "MoE route indices",
                expected: format!("{} indices", self.routes),
                actual: format!("{} indices", indices.len()),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_sort_routes_on_stream",
                ffi::infer_moe_sort_routes_on_stream(
                    indices.ptr,
                    self.expert_counts.ptr,
                    self.expert_offsets.ptr,
                    self.expert_cursors.ptr,
                    self.sorted_routes.ptr,
                    self.sorted_experts.ptr,
                    self.route_to_sorted.ptr,
                    self.routes as u32,
                    self.experts as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns one route count per expert.
    pub fn expert_counts(&self) -> &DeviceBuffer<u32> {
        &self.expert_counts
    }

    /// Returns the `experts + 1` exclusive expert offsets.
    pub fn expert_offsets(&self) -> &DeviceBuffer<u32> {
        &self.expert_offsets
    }

    /// Returns original route identifiers in expert-major order.
    pub fn sorted_routes(&self) -> &DeviceBuffer<u32> {
        &self.sorted_routes
    }

    /// Returns the expert identifier for every sorted route.
    pub fn sorted_experts(&self) -> &DeviceBuffer<u32> {
        &self.sorted_experts
    }

    /// Returns the inverse mapping from original route to sorted position.
    pub fn route_to_sorted(&self) -> &DeviceBuffer<u32> {
        &self.route_to_sorted
    }

    /// Number of routes active in the current ordering.
    pub fn active_routes(&self) -> usize {
        self.routes
    }

    /// Returns device bytes retained by the route ordering workspace.
    pub fn device_bytes(&self) -> usize {
        self.expert_counts.device_bytes()
            + self.expert_offsets.device_bytes()
            + self.expert_cursors.device_bytes()
            + self.sorted_routes.device_bytes()
            + self.sorted_experts.device_bytes()
            + self.route_to_sorted.device_bytes()
    }
}

/// Expert-major NVFP4 activation storage for grouped MoE GEMMs.
pub struct MoeSortedNvfp4Rows {
    capacity_routes: usize,
    routes: usize,
    experts: usize,
    routes_per_row: usize,
    in_features: usize,
    scale_stride: usize,
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    source_packed: DeviceBuffer<u8>,
    source_scales: DeviceBuffer<u8>,
    packed_table: DeviceBuffer<*const u8>,
    scale_table: DeviceBuffer<*const u8>,
}

impl MoeSortedNvfp4Rows {
    /// Allocates expert-major activation storage for a fixed prompt capacity.
    pub fn new(
        rows: usize,
        routes_per_row: usize,
        experts: usize,
        in_features: usize,
    ) -> Result<Self> {
        let routes = rows.saturating_mul(routes_per_row);
        if rows == 0
            || routes_per_row == 0
            || experts == 0
            || routes == 0
            || in_features == 0
            || !in_features.is_multiple_of(16)
            || routes > u32::MAX as usize
            || routes_per_row > u32::MAX as usize
            || experts > u32::MAX as usize
            || in_features > u32::MAX as usize
        {
            return Err(Error::Shape {
                label: "sorted MoE NVFP4 rows",
                expected: "positive u32-sized dimensions and K divisible by 16".to_string(),
                actual: format!(
                    "rows={rows} routes_per_row={routes_per_row} experts={experts} in={in_features}"
                ),
            });
        }
        let scale_stride = format::ue4m3_scale_layout_len(routes, in_features);
        let rows = routes / routes_per_row;
        Ok(Self {
            capacity_routes: routes,
            routes,
            experts,
            routes_per_row,
            in_features,
            scale_stride,
            packed: DeviceBuffer::zeroed(routes * in_features / 2)?,
            scales: DeviceBuffer::zeroed(experts * scale_stride)?,
            source_packed: DeviceBuffer::zeroed(rows * in_features / 2)?,
            source_scales: DeviceBuffer::zeroed(rows * in_features / 16)?,
            packed_table: DeviceBuffer::zeroed(experts)?,
            scale_table: DeviceBuffer::zeroed(experts)?,
        })
    }

    /// Selects an active token-row prefix while retaining the allocated storage.
    pub fn set_rows(&mut self, rows: usize) -> Result<()> {
        let routes = rows.saturating_mul(self.routes_per_row);
        if rows == 0 || routes > self.capacity_routes {
            return Err(Error::Shape {
                label: "active sorted MoE NVFP4 rows",
                expected: format!("1..={} routes", self.capacity_routes),
                actual: format!("rows={rows} routes={routes}"),
            });
        }
        self.routes = routes;
        Ok(())
    }

    /// RMS-normalizes token rows, then gathers and quantizes them in sorted route order.
    pub fn gather_rms_norm_quantize_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        routes: &MoeSortedRoutes,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = self.routes / self.routes_per_row;
        if routes.routes != self.routes
            || routes.experts != self.experts
            || input.len() < rows * self.in_features
            || weight.len() != self.in_features
        {
            return Err(Error::Shape {
                label: "sorted MoE NVFP4 gather quantization",
                expected: format!(
                    "routes={} experts={} input={} weight={}",
                    self.routes,
                    self.experts,
                    rows * self.in_features,
                    self.in_features,
                ),
                actual: format!(
                    "routes={} experts={} input={} weight={}",
                    routes.routes,
                    routes.experts,
                    input.len(),
                    weight.len(),
                ),
            });
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(Error::Format {
                label: "sorted MoE RMSNorm quantization epsilon",
                detail: format!("expected non-negative finite epsilon, got {eps}"),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_gather_rms_norm_quantize_sorted_routes_nvfp4_on_stream",
                ffi::infer_moe_gather_rms_norm_quantize_sorted_routes_nvfp4_on_stream(
                    input.ptr,
                    weight.ptr,
                    routes.sorted_routes.ptr,
                    routes.sorted_experts.ptr,
                    routes.expert_offsets.ptr,
                    self.source_packed.ptr,
                    self.source_scales.ptr,
                    self.packed.ptr,
                    self.scales.ptr,
                    rows as u32,
                    self.routes as u32,
                    self.routes_per_row as u32,
                    self.in_features as u32,
                    self.scale_stride as u32,
                    eps,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Quantizes already-sorted route rows to NVFP4 without gathering.
    pub fn quantize_sorted_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        routes: &MoeSortedRoutes,
        stream: &CudaStream,
    ) -> Result<()> {
        self.quantize_on_stream(input, routes, self.routes, false, stream)
    }

    /// Fuses sorted-route gated GELU activation with NVFP4 quantization.
    pub fn gelu_tanh_mul_quantize_sorted_on_stream(
        &mut self,
        gate: &DeviceBuffer<u16>,
        up: &DeviceBuffer<u16>,
        routes: &MoeSortedRoutes,
        stream: &CudaStream,
    ) -> Result<()> {
        let len = self.routes * self.in_features;
        if routes.routes != self.routes
            || routes.experts != self.experts
            || gate.len() < len
            || up.len() < len
        {
            return Err(Error::Shape {
                label: "sorted MoE GELU NVFP4 quantization",
                expected: format!(
                    "routes={} experts={} gate/up={len}",
                    self.routes, self.experts
                ),
                actual: format!(
                    "routes={} experts={} gate={} up={}",
                    routes.routes,
                    routes.experts,
                    gate.len(),
                    up.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_gelu_tanh_mul_quantize_sorted_routes_nvfp4_on_stream",
                ffi::infer_moe_gelu_tanh_mul_quantize_sorted_routes_nvfp4_on_stream(
                    gate.ptr,
                    up.ptr,
                    routes.sorted_experts.ptr,
                    routes.expert_offsets.ptr,
                    self.packed.ptr,
                    self.scales.ptr,
                    self.routes as u32,
                    self.in_features as u32,
                    self.scale_stride as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Fuses sorted-route gated SiLU from a concatenated gate/up tensor with
    /// NVFP4 quantization.
    pub fn silu_mul_halves_quantize_sorted_on_stream(
        &mut self,
        gate_up: &DeviceBuffer<u16>,
        routes: &MoeSortedRoutes,
        stream: &CudaStream,
    ) -> Result<()> {
        let len = self.routes * self.in_features * 2;
        if routes.routes != self.routes || routes.experts != self.experts || gate_up.len() < len {
            return Err(Error::Shape {
                label: "sorted MoE SiLU NVFP4 quantization",
                expected: format!(
                    "routes={} experts={} gate_up={len}",
                    self.routes, self.experts
                ),
                actual: format!(
                    "routes={} experts={} gate_up={}",
                    routes.routes,
                    routes.experts,
                    gate_up.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_silu_mul_halves_quantize_sorted_routes_nvfp4_on_stream",
                ffi::infer_moe_silu_mul_halves_quantize_sorted_routes_nvfp4_on_stream(
                    gate_up.ptr,
                    routes.sorted_experts.ptr,
                    routes.expert_offsets.ptr,
                    self.packed.ptr,
                    self.scales.ptr,
                    self.routes as u32,
                    self.in_features as u32,
                    self.scale_stride as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Gathers token rows into sorted expert order and quantizes them to NVFP4.
    pub fn gather_quantize_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        routes: &MoeSortedRoutes,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = self.routes / self.routes_per_row;
        if routes.routes != self.routes
            || routes.experts != self.experts
            || input.len() < rows * self.in_features
        {
            return Err(Error::Shape {
                label: "sorted MoE NVFP4 gather quantization",
                expected: format!(
                    "routes={} experts={} input={}",
                    self.routes,
                    self.experts,
                    rows * self.in_features
                ),
                actual: format!(
                    "routes={} experts={} input={}",
                    routes.routes,
                    routes.experts,
                    input.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_gather_quantize_sorted_routes_nvfp4_on_stream",
                ffi::infer_moe_gather_quantize_sorted_routes_nvfp4_on_stream(
                    input.ptr,
                    routes.sorted_routes.ptr,
                    routes.sorted_experts.ptr,
                    routes.expert_offsets.ptr,
                    self.source_packed.ptr,
                    self.source_scales.ptr,
                    self.packed.ptr,
                    self.scales.ptr,
                    rows as u32,
                    self.routes as u32,
                    self.routes_per_row as u32,
                    self.in_features as u32,
                    self.scale_stride as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    fn quantize_on_stream(
        &mut self,
        input: &DeviceBuffer<f32>,
        routes: &MoeSortedRoutes,
        source_rows: usize,
        gather_rows: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        if routes.routes != self.routes
            || routes.experts != self.experts
            || input.len() < source_rows * self.in_features
        {
            return Err(Error::Shape {
                label: "sorted MoE NVFP4 quantization",
                expected: format!(
                    "routes={} experts={} input={}",
                    self.routes,
                    self.experts,
                    source_rows * self.in_features
                ),
                actual: format!(
                    "routes={} experts={} input={}",
                    routes.routes,
                    routes.experts,
                    input.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_quantize_sorted_routes_nvfp4_on_stream",
                ffi::infer_moe_quantize_sorted_routes_nvfp4_on_stream(
                    input.ptr,
                    routes.sorted_routes.ptr,
                    routes.sorted_experts.ptr,
                    routes.expert_offsets.ptr,
                    self.packed.ptr,
                    self.scales.ptr,
                    self.routes as u32,
                    self.routes_per_row as u32,
                    self.in_features as u32,
                    self.scale_stride as u32,
                    i32::from(gather_rows),
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Builds device pointer tables for grouped GEMM input and output operands.
    pub fn build_pointer_tables_on_stream(
        &mut self,
        routes: &MoeSortedRoutes,
        output: &mut DeviceBuffer<u16>,
        output_table: &mut DeviceBuffer<*mut u16>,
        out_features: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if routes.routes != self.routes
            || routes.experts != self.experts
            || output.len() < self.routes * out_features
            || output_table.len() != self.experts
            || out_features == 0
            || out_features > u32::MAX as usize
        {
            return Err(Error::Shape {
                label: "sorted MoE grouped pointer tables",
                expected: format!(
                    "routes={} experts={} output={}",
                    self.routes,
                    self.experts,
                    self.routes * out_features
                ),
                actual: format!(
                    "routes={} experts={} output={} output_table={} out={out_features}",
                    routes.routes,
                    routes.experts,
                    output.len(),
                    output_table.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_moe_grouped_pointer_tables_on_stream",
                ffi::infer_moe_grouped_pointer_tables_on_stream(
                    routes.expert_offsets.ptr,
                    self.packed.ptr,
                    self.scales.ptr,
                    output.ptr,
                    self.packed_table.ptr,
                    self.scale_table.ptr,
                    output_table.ptr,
                    self.experts as u32,
                    self.in_features as u32,
                    out_features as u32,
                    self.scale_stride as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns the expert-indexed packed activation pointer table.
    pub fn packed_table(&self) -> &DeviceBuffer<*const u8> {
        &self.packed_table
    }

    /// Returns the expert-indexed tiled scale pointer table.
    pub fn scale_table(&self) -> &DeviceBuffer<*const u8> {
        &self.scale_table
    }

    /// Returns exact device bytes retained by the grouped activation storage.
    pub fn device_bytes(&self) -> usize {
        self.packed.device_bytes()
            + self.scales.device_bytes()
            + self.source_packed.device_bytes()
            + self.source_scales.device_bytes()
            + self.packed_table.device_bytes()
            + self.scale_table.device_bytes()
    }
}

#[allow(missing_docs)]
pub struct GroupedGemvPointerBuffers<'a> {
    pub indices: &'a DeviceBuffer<u32>,
    pub a_values_table: &'a DeviceBuffer<*const u8>,
    pub a_scales_table: &'a DeviceBuffer<*const u8>,
    pub b_values: *const u8,
    pub b_scales: *const u8,
    pub c_table: DeviceInOut<'a, *const f32>,
    pub d_table: DeviceInOut<'a, *mut f32>,
    pub out_a_values: DeviceOutput<'a, *const u8>,
    pub out_a_scales: DeviceOutput<'a, *const u8>,
    pub out_b_values: DeviceOutput<'a, *const u8>,
    pub out_b_scales: DeviceOutput<'a, *const u8>,
}

#[allow(missing_docs)]
pub struct GroupedGemvPointerTableBuffers<'a> {
    pub indices: &'a DeviceBuffer<u32>,
    pub a_values_table: &'a DeviceBuffer<*const u8>,
    pub a_scales_table: &'a DeviceBuffer<*const u8>,
    pub b_values_table: &'a DeviceBuffer<*const u8>,
    pub b_scales_table: &'a DeviceBuffer<*const u8>,
    pub c_table: DeviceInOut<'a, *const f32>,
    pub d_table: DeviceInOut<'a, *mut f32>,
    pub out_a_values: DeviceOutput<'a, *const u8>,
    pub out_a_scales: DeviceOutput<'a, *const u8>,
    pub out_b_values: DeviceOutput<'a, *const u8>,
    pub out_b_scales: DeviceOutput<'a, *const u8>,
}

/// Gathers selected expert FP4 matrix pointers into grouped GEMV operand arrays.
pub fn gather_nvfp4_grouped_gemv_ptrs_on_stream(
    mut buffers: GroupedGemvPointerBuffers<'_>,
    stream: &CudaStream,
) -> Result<()> {
    let groups = buffers.indices.len();
    let table_len = buffers.a_values_table.len();
    if groups == 0
        || table_len == 0
        || buffers.a_scales_table.len() != table_len
        || buffers.c_table.len() != groups
        || buffers.d_table.len() != groups
        || buffers.out_a_values.len() != groups
        || buffers.out_a_scales.len() != groups
        || buffers.out_b_values.len() != groups
        || buffers.out_b_scales.len() != groups
        || groups > u32::MAX as usize
        || table_len > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "grouped GEMV pointer gather",
            expected: "matching non-empty pointer tables and group outputs".to_string(),
            actual: format!(
                "groups={groups} table={} a_scales={} c={} d={} out_a={} out_b={}",
                table_len,
                buffers.a_scales_table.len(),
                buffers.c_table.len(),
                buffers.d_table.len(),
                buffers.out_a_values.len(),
                buffers.out_b_values.len()
            ),
        });
    }

    unsafe {
        let c_table = buffers.c_table.buffer().ptr as *const *const f32;
        let d_table = buffers.d_table.buffer().ptr as *const *mut f32;
        let out_a_values = buffers.out_a_values.buffer_mut().ptr;
        let out_a_scales = buffers.out_a_scales.buffer_mut().ptr;
        let out_b_values = buffers.out_b_values.buffer_mut().ptr;
        let out_b_scales = buffers.out_b_scales.buffer_mut().ptr;
        let out_c = buffers.c_table.buffer_mut().ptr;
        let out_d = buffers.d_table.buffer_mut().ptr;
        check_cuda(
            "infer_gather_nvfp4_grouped_gemv_ptrs_on_stream",
            ffi::infer_gather_nvfp4_grouped_gemv_ptrs_on_stream(
                buffers.indices.ptr,
                buffers.a_values_table.ptr,
                buffers.a_scales_table.ptr,
                buffers.b_values,
                buffers.b_scales,
                c_table,
                d_table,
                groups as u32,
                table_len as u32,
                out_a_values,
                out_a_scales,
                out_b_values,
                out_b_scales,
                out_c,
                out_d,
                stream.as_raw(),
            ),
        )
    }
}

/// Gathers selected expert and per-slot FP4 matrix pointers into grouped GEMV operands.
pub fn gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
    mut buffers: GroupedGemvPointerTableBuffers<'_>,
    stream: &CudaStream,
) -> Result<()> {
    let groups = buffers.indices.len();
    let table_len = buffers.a_values_table.len();
    if groups == 0
        || table_len == 0
        || buffers.a_scales_table.len() != table_len
        || buffers.b_values_table.len() != groups
        || buffers.b_scales_table.len() != groups
        || buffers.c_table.len() != groups
        || buffers.d_table.len() != groups
        || buffers.out_a_values.len() != groups
        || buffers.out_a_scales.len() != groups
        || buffers.out_b_values.len() != groups
        || buffers.out_b_scales.len() != groups
        || groups > u32::MAX as usize
        || table_len > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "grouped GEMV pointer table gather",
            expected: "matching non-empty selected A table, slot B table, and group outputs"
                .to_string(),
            actual: format!(
                "groups={groups} table={} b_values={} b_scales={} c={} d={}",
                table_len,
                buffers.b_values_table.len(),
                buffers.b_scales_table.len(),
                buffers.c_table.len(),
                buffers.d_table.len()
            ),
        });
    }

    unsafe {
        let c_table = buffers.c_table.buffer().ptr as *const *const f32;
        let d_table = buffers.d_table.buffer().ptr as *const *mut f32;
        let out_a_values = buffers.out_a_values.buffer_mut().ptr;
        let out_a_scales = buffers.out_a_scales.buffer_mut().ptr;
        let out_b_values = buffers.out_b_values.buffer_mut().ptr;
        let out_b_scales = buffers.out_b_scales.buffer_mut().ptr;
        let out_c = buffers.c_table.buffer_mut().ptr;
        let out_d = buffers.d_table.buffer_mut().ptr;
        check_cuda(
            "infer_gather_nvfp4_grouped_gemv_ptr_tables_on_stream",
            ffi::infer_gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
                buffers.indices.ptr,
                buffers.a_values_table.ptr,
                buffers.a_scales_table.ptr,
                buffers.b_values_table.ptr,
                buffers.b_scales_table.ptr,
                c_table,
                d_table,
                groups as u32,
                table_len as u32,
                out_a_values,
                out_a_scales,
                out_b_values,
                out_b_scales,
                out_c,
                out_d,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub struct MoeSiluQuantizeSlotBuffers<'a> {
    pub indices: &'a DeviceBuffer<u32>,
    pub gate_up_table: &'a DeviceBuffer<*const f32>,
    pub packed_table: DeviceOutput<'a, *mut u8>,
    pub scales_table: DeviceOutput<'a, *mut u8>,
    pub input_scale_table: &'a DeviceBuffer<f32>,
    pub gate_up_alpha_table: &'a DeviceBuffer<f32>,
}

/// Enqueues per-slot `silu(gate) * up` plus NVFP4 quantization using selected expert scales.
pub fn moe_silu_quantize_slots_nvfp4_on_stream(
    mut buffers: MoeSiluQuantizeSlotBuffers<'_>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = buffers.indices.len();
    if rows == 0
        || groups == 0
        || buffers.gate_up_table.len() != groups
        || buffers.packed_table.len() != groups
        || buffers.scales_table.len() != groups
        || buffers.input_scale_table.is_empty()
        || buffers.gate_up_alpha_table.is_empty()
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "MoE slot SiLU quantize",
            expected: "non-empty rows, matching slot tables, and expert scale tables".to_string(),
            actual: format!(
                "rows={rows} groups={groups} gate_up={} packed={} scales={} input_scales={} gate_up_alphas={}",
                buffers.gate_up_table.len(),
                buffers.packed_table.len(),
                buffers.scales_table.len(),
                buffers.input_scale_table.len(),
                buffers.gate_up_alpha_table.len()
            ),
        });
    }

    unsafe {
        let packed_table = buffers.packed_table.buffer_mut().ptr;
        let scales_table = buffers.scales_table.buffer_mut().ptr;
        check_cuda(
            "infer_moe_silu_quantize_slots_nvfp4_on_stream",
            ffi::infer_moe_silu_quantize_slots_nvfp4_on_stream(
                buffers.indices.ptr,
                buffers.gate_up_table.ptr,
                packed_table,
                scales_table,
                buffers.input_scale_table.ptr,
                buffers.gate_up_alpha_table.ptr,
                rows as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
    mut buffers: MoeSiluQuantizeSlotBuffers<'_>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = buffers.indices.len();
    if rows == 0
        || groups == 0
        || buffers.gate_up_table.len() != groups
        || buffers.packed_table.len() != groups
        || buffers.scales_table.len() != groups
        || buffers.input_scale_table.is_empty()
        || buffers.gate_up_alpha_table.is_empty()
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "MoE slot SiLU simple-scale quantize",
            expected: "non-empty rows, matching slot tables, and expert scale tables".to_string(),
            actual: format!(
                "rows={rows} groups={groups} gate_up={} packed={} scales={} input_scales={} gate_up_alphas={}",
                buffers.gate_up_table.len(),
                buffers.packed_table.len(),
                buffers.scales_table.len(),
                buffers.input_scale_table.len(),
                buffers.gate_up_alpha_table.len()
            ),
        });
    }

    unsafe {
        let packed_table = buffers.packed_table.buffer_mut().ptr;
        let scales_table = buffers.scales_table.buffer_mut().ptr;
        check_cuda(
            "infer_moe_silu_quantize_slots_nvfp4_simple_scales_on_stream",
            ffi::infer_moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
                buffers.indices.ptr,
                buffers.gate_up_table.ptr,
                packed_table,
                scales_table,
                buffers.input_scale_table.ptr,
                buffers.gate_up_alpha_table.ptr,
                rows as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues per-slot `silu(gate) * up` into f32 output vectors.
pub fn moe_silu_slots_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    gate_up_table: &DeviceBuffer<*const f32>,
    output_table: &DeviceBuffer<*mut f32>,
    gate_up_alpha_table: &DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    if rows == 0
        || groups == 0
        || gate_up_table.len() != groups
        || output_table.len() != groups
        || gate_up_alpha_table.is_empty()
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "MoE slot f32 SiLU buffers",
            expected: "non-empty rows, matching slot tables, and expert alpha table".to_string(),
            actual: format!(
                "rows={rows} groups={groups} gate_up={} output={} alphas={}",
                gate_up_table.len(),
                output_table.len(),
                gate_up_alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_moe_silu_slots_f32_on_stream",
            ffi::infer_moe_silu_slots_f32_on_stream(
                indices.ptr,
                gate_up_table.ptr,
                output_table.ptr,
                gate_up_alpha_table.ptr,
                rows as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Writes the weighted sum of per-slot f32 expert outputs into `output`.
pub fn moe_weighted_accumulate_slots_f32_on_stream(
    indices: &DeviceBuffer<u32>,
    route_weights: &DeviceBuffer<f32>,
    inputs: &DeviceBuffer<*const f32>,
    alpha_table: &DeviceBuffer<f32>,
    mut output: DeviceInOut<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    if output.is_empty()
        || groups == 0
        || route_weights.len() != groups
        || inputs.len() != groups
        || alpha_table.is_empty()
        || output.len() > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "MoE weighted slot accumulate",
            expected: "non-empty output and matching route/input groups".to_string(),
            actual: format!(
                "output={} groups={groups} weights={} inputs={} alphas={}",
                output.len(),
                route_weights.len(),
                inputs.len(),
                alpha_table.len()
            ),
        });
    }

    unsafe {
        check_cuda(
            "infer_moe_weighted_accumulate_slots_f32_on_stream",
            ffi::infer_moe_weighted_accumulate_slots_f32_on_stream(
                indices.ptr,
                route_weights.ptr,
                inputs.ptr,
                alpha_table.ptr,
                output.buffer_mut().ptr,
                output.len() as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Writes one weighted sum of per-slot expert outputs for every dense row.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_accumulate_slots_f32_batch_on_stream(
    indices: &DeviceBuffer<u32>,
    route_weights: &DeviceBuffer<f32>,
    inputs: &DeviceBuffer<*const f32>,
    alpha_table: &DeviceBuffer<f32>,
    output: DeviceInOut<'_, f32>,
    rows: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = output.len().checked_div(rows).unwrap_or(0);
    moe_weighted_accumulate_slots_f32_batch_prefix_on_stream(
        indices,
        route_weights,
        inputs,
        alpha_table,
        output,
        rows,
        groups,
        len,
        stream,
    )
}

/// Writes a weighted sum for an active prefix of dense rows.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_accumulate_slots_f32_batch_prefix_on_stream(
    indices: &DeviceBuffer<u32>,
    route_weights: &DeviceBuffer<f32>,
    inputs: &DeviceBuffer<*const f32>,
    alpha_table: &DeviceBuffer<f32>,
    mut output: DeviceInOut<'_, f32>,
    rows: usize,
    groups: usize,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    let routes = rows.saturating_mul(groups);
    let output_len = rows.saturating_mul(len);
    if rows == 0
        || groups == 0
        || len == 0
        || output.len() < output_len
        || indices.len() < routes
        || route_weights.len() < routes
        || inputs.len() < routes
        || alpha_table.is_empty()
        || rows > u32::MAX as usize
        || len > u32::MAX as usize
        || groups > u32::MAX as usize
        || output_len > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched MoE weighted slot accumulate",
            expected: format!(
                "output={rows}xnonzero routes={routes} matching indices/weights/inputs"
            ),
            actual: format!(
                "output={} rows={rows} groups={groups} indices={} weights={} inputs={} alphas={}",
                output.len(),
                indices.len(),
                route_weights.len(),
                inputs.len(),
                alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_moe_weighted_accumulate_slots_f32_batch_on_stream",
            ffi::infer_moe_weighted_accumulate_slots_f32_batch_on_stream(
                indices.ptr,
                route_weights.ptr,
                inputs.ptr,
                alpha_table.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                len as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Writes weighted row sums from expert-major F32 route outputs.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_accumulate_sorted_slots_f32_batch_on_stream(
    routes: &MoeSortedRoutes,
    indices: &DeviceBuffer<u32>,
    route_weights: &DeviceBuffer<f32>,
    sorted_inputs: &DeviceBuffer<*const f32>,
    alpha_table: &DeviceBuffer<f32>,
    mut output: DeviceInOut<'_, f32>,
    rows: usize,
    groups: usize,
    features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let route_count = rows.saturating_mul(groups);
    if rows == 0
        || groups == 0
        || features == 0
        || output.len() < rows.saturating_mul(features)
        || routes.route_to_sorted().len() < route_count
        || indices.len() < route_count
        || route_weights.len() < route_count
        || sorted_inputs.len() < route_count
        || alpha_table.is_empty()
        || rows > u32::MAX as usize
        || features > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "sorted batched MoE weighted slot accumulate",
            expected: format!(
                "output={rows}xnonzero routes={route_count} matching ordering/indices/weights/inputs"
            ),
            actual: format!(
                "output={} ordering={} indices={} weights={} inputs={} alphas={}",
                output.len(),
                routes.route_to_sorted().len(),
                indices.len(),
                route_weights.len(),
                sorted_inputs.len(),
                alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_moe_weighted_accumulate_sorted_slots_f32_batch_on_stream",
            ffi::infer_moe_weighted_accumulate_sorted_slots_f32_batch_on_stream(
                routes.route_to_sorted.ptr,
                indices.ptr,
                route_weights.ptr,
                sorted_inputs.ptr,
                alpha_table.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                features as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Writes weighted per-row sums from expert-major BF16 route output.
pub fn moe_weighted_accumulate_sorted_bf16_batch_on_stream(
    routes: &MoeSortedRoutes,
    route_weights: &DeviceBuffer<f32>,
    sorted_inputs: &DeviceBuffer<u16>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    routes_per_row: usize,
    features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let route_count = rows.saturating_mul(routes_per_row);
    if rows == 0
        || routes_per_row == 0
        || features == 0
        || route_count != routes.routes
        || route_weights.len() < route_count
        || sorted_inputs.len() < route_count.saturating_mul(features)
        || output.len() < rows.saturating_mul(features)
        || rows > u32::MAX as usize
        || features > u32::MAX as usize
        || routes_per_row > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "sorted MoE weighted accumulate",
            expected: format!(
                "routes={} weights={} sorted={} output={}x{}",
                route_count,
                route_count,
                route_count.saturating_mul(features),
                rows,
                features
            ),
            actual: format!(
                "routes={} weights={} sorted={} output={} rows={rows} routes_per_row={routes_per_row}",
                routes.routes,
                route_weights.len(),
                sorted_inputs.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_moe_weighted_accumulate_sorted_bf16_batch_on_stream",
            ffi::infer_moe_weighted_accumulate_sorted_bf16_batch_on_stream(
                routes.route_to_sorted.ptr,
                route_weights.ptr,
                sorted_inputs.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                features as u32,
                routes_per_row as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies ReLU squared elementwise on `stream`.
pub fn relu_squared_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty() || output.len() != input.len() || input.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "ReLU squared f32 buffers",
            expected: "matching non-empty input and output".to_string(),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_relu_squared_f32_on_stream",
            ffi::infer_relu_squared_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Combines routed and gated shared FFN outputs with the residual, then writes
/// the result rounded to BF16 precision in F32 storage.
pub fn qwen36_ffn_finalize_f32_into_on_stream(
    moe_output: &DeviceBuffer<f32>,
    shared_gate_logit: &DeviceBuffer<f32>,
    shared_output: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let len = residual.len();
    if len == 0
        || len > u32::MAX as usize
        || moe_output.len() != len
        || shared_gate_logit.len() != 1
        || shared_output.len() != len
        || output.len() != len
    {
        return Err(Error::Shape {
            label: "Qwen3.6 FFN finalize",
            expected: "matching non-empty FFN/residual buffers and one shared gate logit"
                .to_string(),
            actual: format!(
                "moe={} gate={} shared={} residual={} output={}",
                moe_output.len(),
                shared_gate_logit.len(),
                shared_output.len(),
                residual.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_ffn_finalize_f32_on_stream",
            ffi::infer_qwen36_ffn_finalize_f32_on_stream(
                moe_output.ptr,
                shared_gate_logit.ptr,
                shared_output.ptr,
                residual.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Combines one routed result per batch row with the gated shared FFN and
/// residual, then writes BF16-rounded F32 output.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_ffn_finalize_batch_f32_into_on_stream(
    routed_output: &DeviceBuffer<f32>,
    shared_gate_logit: &DeviceBuffer<f32>,
    shared_output: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || routed_output.len() < len
        || shared_gate_logit.len() < rows
        || shared_output.len() < len
        || residual.len() < len
        || output.len() < len
    {
        return Err(Error::Shape {
            label: "Qwen3.6 batch FFN finalize",
            expected: format!("rows={rows} cols={cols} values={len}"),
            actual: format!(
                "routed={} gate={} shared={} residual={} output={}",
                routed_output.len(),
                shared_gate_logit.len(),
                shared_output.len(),
                residual.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_ffn_finalize_batch_f32_on_stream",
            ffi::infer_qwen36_ffn_finalize_batch_f32_on_stream(
                routed_output.ptr,
                shared_gate_logit.ptr,
                shared_output.ptr,
                residual.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Accumulates routed slot outputs, applies the shared-expert gate, adds the
/// residual, and writes BF16-rounded F32 output in one kernel.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_ffn_finalize_routed_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    route_weights: &DeviceBuffer<f32>,
    routed_outputs: &DeviceBuffer<*const f32>,
    alpha_table: &DeviceBuffer<f32>,
    shared_gate_logit: &DeviceBuffer<f32>,
    shared_output: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    let len = residual.len();
    if len == 0
        || len > u32::MAX as usize
        || groups == 0
        || groups > u32::MAX as usize
        || route_weights.len() != groups
        || routed_outputs.len() != groups
        || alpha_table.is_empty()
        || shared_gate_logit.len() != 1
        || shared_output.len() != len
        || output.len() != len
    {
        return Err(Error::Shape {
            label: "Qwen3.6 routed FFN finalize",
            expected:
                "matching routed groups, non-empty FFN/residual buffers, and one shared gate logit"
                    .to_string(),
            actual: format!(
                "indices={} weights={} routed={} alphas={} gate={} shared={} residual={} output={}",
                indices.len(),
                route_weights.len(),
                routed_outputs.len(),
                alpha_table.len(),
                shared_gate_logit.len(),
                shared_output.len(),
                residual.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_ffn_finalize_routed_f32_on_stream",
            ffi::infer_qwen36_ffn_finalize_routed_f32_on_stream(
                indices.ptr,
                route_weights.ptr,
                routed_outputs.ptr,
                alpha_table.ptr,
                shared_gate_logit.ptr,
                shared_output.ptr,
                residual.ptr,
                output.buffer_mut().ptr,
                len as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Finalizes routed and shared FFNs for independent batch rows.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_ffn_finalize_routed_batch_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    route_weights: &DeviceBuffer<f32>,
    routed_outputs: &DeviceBuffer<*const f32>,
    alpha_table: &DeviceBuffer<f32>,
    shared_gate_logit: &DeviceBuffer<f32>,
    shared_output: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    groups_per_row: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    let groups = rows.saturating_mul(groups_per_row);
    if rows == 0
        || cols == 0
        || groups_per_row == 0
        || indices.len() < groups
        || route_weights.len() < groups
        || routed_outputs.len() < groups
        || alpha_table.is_empty()
        || shared_gate_logit.len() < rows
        || shared_output.len() < len
        || residual.len() < len
        || output.len() < len
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || groups_per_row > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Qwen3.6 batched routed FFN finalize",
            expected: format!("routes={groups} gate={rows} shared/residual/output={len}"),
            actual: format!(
                "indices={} weights={} routed={} gate={} shared={} residual={} output={} rows={rows} cols={cols} groups_per_row={groups_per_row}",
                indices.len(),
                route_weights.len(),
                routed_outputs.len(),
                shared_gate_logit.len(),
                shared_output.len(),
                residual.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_ffn_finalize_routed_batch_f32_on_stream",
            ffi::infer_qwen36_ffn_finalize_routed_batch_f32_on_stream(
                indices.ptr,
                route_weights.ptr,
                routed_outputs.ptr,
                alpha_table.ptr,
                shared_gate_logit.ptr,
                shared_output.ptr,
                residual.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                groups_per_row as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one-position Neox RoPE into an existing output buffer on `stream`.
pub fn rope_neox_f32_into_on_stream(
    rows: usize,
    head_dim: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    position: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_rope_neox_f32(rows, head_dim, input, &output, Some(position), theta)?;
    unsafe {
        check_cuda(
            "infer_rope_neox_f32_on_stream",
            ffi::infer_rope_neox_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                position as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one-position NeoX RoPE over only the first `rotary_dim` channels.
///
/// Dimensions `rotary_dim..head_dim` are copied through unchanged. This matches
/// Qwen3.6 text full-attention, where `partial_rotary_factor=0.25` gives
/// `rotary_dim=64` for `head_dim=256`.
pub fn rope_neox_partial_f32_into_on_stream(
    rows: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    position: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_rope_neox_f32(rows, head_dim, input, &output, Some(position), theta)?;
    if rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "partial RoPE dimensions",
            expected: "non-zero even rotary_dim <= head_dim".to_string(),
            actual: format!("rotary_dim={rotary_dim} head_dim={head_dim}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_rope_neox_partial_f32_on_stream",
            ffi::infer_rope_neox_partial_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                rotary_dim as u32,
                position as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues proportional partial NeoX RoPE for one position.
///
/// `rotary_dim / 2` leading frequency pairs are rotated using ordinary
/// full-head NeoX pairing and frequencies. Remaining pairs pass through.
pub fn rope_neox_proportional_f32_into_on_stream(
    rows: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    position: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_rope_neox_f32(rows, head_dim, input, &output, Some(position), theta)?;
    if rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim / 2 > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "proportional partial RoPE dimensions",
            expected: "non-zero even rotary_dim <= head_dim".to_string(),
            actual: format!("rotary_dim={rotary_dim} head_dim={head_dim}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_rope_neox_proportional_f32_on_stream",
            ffi::infer_rope_neox_proportional_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                (rotary_dim / 2) as u32,
                position as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues Gemma-style proportional partial NeoX RoPE for a dense sequence span.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    input_token_offset: usize,
    start_position: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let end_tokens = input_token_offset.saturating_add(tokens);
    let required = end_tokens.saturating_mul(heads).saturating_mul(head_dim);
    if tokens == 0
        || heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || input.len() < required
        || output.len() < required
        || tokens > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || rotary_dim > u32::MAX as usize
        || input_token_offset > u32::MAX as usize
        || start_position > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "proportional sequence RoPE",
            expected: "matching non-empty sequence, head, rotary, and offset dimensions"
                .to_string(),
            actual: format!(
                "tokens={tokens} heads={heads} head_dim={head_dim} rotary_dim={rotary_dim} input={} output={} input_token_offset={input_token_offset} start_position={start_position}",
                input.len(),
                output.len()
            ),
        });
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::Format {
            label: "proportional sequence RoPE theta",
            detail: format!("expected positive finite theta, got {theta}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_rope_neox_proportional_sequence_f32_on_stream",
            ffi::infer_rope_neox_proportional_sequence_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                heads as u32,
                head_dim as u32,
                (rotary_dim / 2) as u32,
                input_token_offset as u32,
                start_position as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// RMS-normalizes and applies Gemma proportional RoPE to Q and K sequence spans.
#[allow(clippy::too_many_arguments)]
pub fn dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
    tokens: usize,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    q_input: &DeviceBuffer<f32>,
    q_weight: &DeviceBuffer<f32>,
    mut q_output: DeviceOutput<'_, f32>,
    q_eps: f32,
    k_input: &DeviceBuffer<f32>,
    k_weight: &DeviceBuffer<f32>,
    mut k_output: DeviceOutput<'_, f32>,
    k_eps: f32,
    input_token_offset: usize,
    start_position: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let end_tokens = input_token_offset
        .checked_add(tokens)
        .ok_or_else(|| Error::Shape {
            label: "dual RMSNorm proportional sequence RoPE",
            expected: "token offset plus span without overflow".to_string(),
            actual: format!("offset={input_token_offset} tokens={tokens}"),
        })?;
    let q_required = end_tokens
        .checked_mul(q_heads)
        .and_then(|rows| rows.checked_mul(head_dim));
    let k_required = end_tokens
        .checked_mul(k_heads)
        .and_then(|rows| rows.checked_mul(head_dim));
    let total_rows = tokens.saturating_mul(q_heads.saturating_add(k_heads));
    let dimensions_valid = tokens > 0
        && q_heads > 0
        && k_heads > 0
        && head_dim > 0
        && head_dim.is_multiple_of(2)
        && rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim.is_multiple_of(2)
        && tokens <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && k_heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && rotary_dim <= u32::MAX as usize
        && input_token_offset <= u32::MAX as usize
        && start_position <= u32::MAX as usize
        && total_rows <= u32::MAX as usize;
    let buffers_valid = q_required
        .is_some_and(|required| q_input.len() >= required && q_output.len() >= required)
        && k_required
            .is_some_and(|required| k_input.len() >= required && k_output.len() >= required)
        && q_weight.len() == head_dim
        && k_weight.len() == head_dim;
    if !dimensions_valid || !buffers_valid {
        return Err(Error::Shape {
            label: "dual RMSNorm proportional sequence RoPE",
            expected: "matching non-empty Q/K sequence, head, rotary, and offset dimensions"
                .to_string(),
            actual: format!(
                "tokens={tokens} q_heads={q_heads} k_heads={k_heads} head_dim={head_dim} rotary_dim={rotary_dim} q_input={} q_weight={} q_output={} k_input={} k_weight={} k_output={} input_token_offset={input_token_offset} start_position={start_position}",
                q_input.len(),
                q_weight.len(),
                q_output.len(),
                k_input.len(),
                k_weight.len(),
                k_output.len(),
            ),
        });
    }
    if !theta.is_finite()
        || theta <= 0.0
        || !q_eps.is_finite()
        || q_eps < 0.0
        || !k_eps.is_finite()
        || k_eps < 0.0
    {
        return Err(Error::Format {
            label: "dual RMSNorm proportional sequence RoPE parameters",
            detail: format!("theta={theta} q_eps={q_eps} k_eps={k_eps}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_dual_rms_norm_rope_neox_proportional_sequence_f32_on_stream",
            ffi::infer_dual_rms_norm_rope_neox_proportional_sequence_f32_on_stream(
                q_input.ptr,
                q_weight.ptr,
                q_output.buffer_mut().ptr,
                k_input.ptr,
                k_weight.ptr,
                k_output.buffer_mut().ptr,
                tokens as u32,
                q_heads as u32,
                k_heads as u32,
                head_dim as u32,
                (rotary_dim / 2) as u32,
                input_token_offset as u32,
                start_position as u32,
                theta,
                q_eps,
                k_eps,
                stream.as_raw(),
            ),
        )
    }
}

/// MRoPE/IMRoPE sections `[v0,v1,v2,v3]` (t,h,w,extra), summing to
/// `rotary_dim / 2`. For text-only Qwen3.5/3.6, v3 is 0 and the four positions
/// are `[position, position, position, 0]`.
#[derive(Clone, Copy, Debug)]
pub struct MropeSections {
    /// Time/temporal section size in pairs.
    pub v0: usize,
    /// Height section size in pairs.
    pub v1: usize,
    /// Width section size in pairs.
    pub v2: usize,
    /// Extra (vision) section size in pairs; 0 for text-only.
    pub v3: usize,
}

impl MropeSections {
    /// Validates that the sections sum to `rotary_dim / 2`.
    pub fn validate(&self, rotary_dim: usize) -> Result<()> {
        let sum = self.v0 + self.v1 + self.v2 + self.v3;
        if sum != rotary_dim / 2 {
            return Err(Error::Shape {
                label: "MRoPE sections",
                expected: format!("sections sum to rotary_dim/2={}", rotary_dim / 2),
                actual: format!("v0+v1+v2+v3={sum}"),
            });
        }
        Ok(())
    }
}

/// Applies IMRoPE/MRoPE to contiguous f32 rows of length `head_dim`.
///
/// Each row is one attention head. The first `rotary_dim` channels are paired
/// as `(x[i], x[i + rotary_dim/2])` for `i in [0, rotary_dim/2)`. Each pair `i`
/// is assigned to a section via `i % sect_dims` (matching llama.cpp's IMRoPE
/// sector mapping) and rotated by the corresponding position from
/// `[pos_t, pos_h, pos_w, pos_extra]` times `theta^(-2*i/rotary_dim)`.
/// Channels in `[rotary_dim, head_dim)` are copied unchanged.
pub fn rope_imrope_f32_into_on_stream(
    rows: usize,
    head_dim: usize,
    rotary_dim: usize,
    sections: MropeSections,
    positions: [u32; 4],
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_rope_neox_f32(rows, head_dim, input, &output, None, theta)?;
    if rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "IMRoPE dimensions",
            expected: "non-zero even rotary_dim <= head_dim".to_string(),
            actual: format!("rotary_dim={rotary_dim} head_dim={head_dim}"),
        });
    }
    sections.validate(rotary_dim)?;
    unsafe {
        check_cuda(
            "infer_rope_imrope_f32_on_stream",
            ffi::infer_rope_imrope_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                rotary_dim as u32,
                sections.v0 as u32,
                sections.v1 as u32,
                sections.v2 as u32,
                sections.v3 as u32,
                positions[0],
                positions[1],
                positions[2],
                positions[3],
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies IMRoPE/MRoPE using four device-resident positions.
#[allow(clippy::too_many_arguments)]
pub fn rope_imrope_f32_indexed_into_on_stream(
    rows: usize,
    head_dim: usize,
    rotary_dim: usize,
    sections: MropeSections,
    positions: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_rope_neox_f32(rows, head_dim, input, &output, None, theta)?;
    if rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim > u32::MAX as usize
        || !matches!(positions.len(), 1 | 4)
    {
        return Err(Error::Shape {
            label: "indexed IMRoPE dimensions",
            expected: "non-zero even rotary_dim <= head_dim and one or four positions".to_string(),
            actual: format!(
                "rotary_dim={rotary_dim} head_dim={head_dim} positions={}",
                positions.len()
            ),
        });
    }
    sections.validate(rotary_dim)?;
    unsafe {
        check_cuda(
            "infer_rope_imrope_f32_indexed_on_stream",
            ffi::infer_rope_imrope_f32_indexed_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                rotary_dim as u32,
                sections.v0 as u32,
                sections.v1 as u32,
                sections.v2 as u32,
                sections.v3 as u32,
                positions.ptr,
                positions.len() as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies text IMRoPE using one device-resident position per batch row.
#[allow(clippy::too_many_arguments)]
pub fn rope_imrope_text_batch_f32_into_on_stream(
    rows: usize,
    heads_per_row: usize,
    head_dim: usize,
    rotary_dim: usize,
    sections: MropeSections,
    positions: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows
        .checked_mul(heads_per_row)
        .and_then(|value| value.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    if rows == 0
        || heads_per_row == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || positions.len() < rows
        || input.len() < len
        || output.len() < len
        || rows > u32::MAX as usize
        || heads_per_row > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || rotary_dim > u32::MAX as usize
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(Error::Shape {
            label: "batched text IMRoPE buffers",
            expected: format!("positions={rows} input/output={len}"),
            actual: format!(
                "positions={} input={} output={} rows={rows} heads={heads_per_row} head_dim={head_dim} rotary_dim={rotary_dim}",
                positions.len(),
                input.len(),
                output.len()
            ),
        });
    }
    sections.validate(rotary_dim)?;
    unsafe {
        check_cuda(
            "infer_rope_imrope_text_batch_f32_on_stream",
            ffi::infer_rope_imrope_text_batch_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                positions.ptr,
                rows as u32,
                heads_per_row as u32,
                head_dim as u32,
                rotary_dim as u32,
                sections.v0 as u32,
                sections.v1 as u32,
                sections.v2 as u32,
                sections.v3 as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_rope_neox_f32(
    rows: usize,
    head_dim: usize,
    input: &DeviceBuffer<f32>,
    output: &DeviceOutput<'_, f32>,
    position: Option<usize>,
    theta: f32,
) -> Result<()> {
    let len = rows.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "RoPE input",
        expected: "rows * head_dim without overflow".to_string(),
        actual: format!("rows={rows} head_dim={head_dim}"),
    })?;
    if input.len() != len || output.len() != len {
        return Err(Error::Shape {
            label: "RoPE buffers",
            expected: format!("{len} values"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    if head_dim == 0 || !head_dim.is_multiple_of(2) || rows == 0 {
        return Err(Error::Shape {
            label: "RoPE dimensions",
            expected: "non-zero rows and even non-zero head_dim".to_string(),
            actual: format!("rows={rows} head_dim={head_dim}"),
        });
    }
    if rows > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || position.is_some_and(|position| position > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "RoPE dimensions",
            expected: "u32-sized rows, head_dim, and position".to_string(),
            actual: format!("rows={rows} head_dim={head_dim} position={position:?}"),
        });
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::Format {
            label: "RoPE theta",
            detail: format!("expected positive finite theta, got {theta}"),
        });
    }
    Ok(())
}

/// Enqueues one-position Neox RoPE using a device-resident position scalar.
pub fn rope_neox_f32_indexed_into_on_stream(
    rows: usize,
    head_dim: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    position: &DeviceBuffer<u32>,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_rope_neox_f32(rows, head_dim, input, &output, None, theta)?;
    if position.len() != 1 {
        return Err(Error::Shape {
            label: "RoPE indexed position",
            expected: "1 value".to_string(),
            actual: format!("{} values", position.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_rope_neox_f32_indexed_on_stream",
            ffi::infer_rope_neox_f32_indexed_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                head_dim as u32,
                position.ptr,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues sequence Neox RoPE into an existing output buffer on `stream`.
pub fn rope_neox_sequence_f32_into_on_stream(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    start_position: usize,
    theta: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = tokens
        .checked_mul(heads)
        .and_then(|rows| rows.checked_mul(head_dim))
        .ok_or_else(|| Error::Shape {
            label: "sequence RoPE input",
            expected: "tokens * heads * head_dim without overflow".to_string(),
            actual: format!("tokens={tokens} heads={heads} head_dim={head_dim}"),
        })?;
    if input.len() < len || output.len() < len {
        return Err(Error::Shape {
            label: "sequence RoPE buffers",
            expected: format!("at least {len} values"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    if tokens == 0
        || heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || tokens > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || start_position > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "sequence RoPE dimensions",
            expected: "non-zero u32-sized dims and even head_dim".to_string(),
            actual: format!(
                "tokens={tokens} heads={heads} head_dim={head_dim} start_position={start_position}"
            ),
        });
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::Format {
            label: "sequence RoPE theta",
            detail: format!("expected positive finite theta, got {theta}"),
        });
    }

    unsafe {
        check_cuda(
            "infer_rope_neox_sequence_f32_on_stream",
            ffi::infer_rope_neox_sequence_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                heads as u32,
                head_dim as u32,
                start_position as u32,
                theta,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues sequence NeoX RoPE using a device-resident inverse-frequency table.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_inv_freq_sequence_f32_into_on_stream(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    inv_freq: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    start_position: usize,
    stream: &CudaStream,
) -> Result<()> {
    rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream(
        tokens,
        heads,
        head_dim,
        rotary_dim,
        input,
        inv_freq,
        output,
        0,
        start_position,
        stream,
    )
}

/// Enqueues inverse-frequency sequence NeoX RoPE at a token offset in dense buffers.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    inv_freq: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    input_token_offset: usize,
    start_position: usize,
    stream: &CudaStream,
) -> Result<()> {
    rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream(
        tokens,
        heads,
        head_dim,
        rotary_dim,
        input,
        inv_freq,
        output,
        input_token_offset,
        start_position,
        1.0,
        stream,
    )
}

/// Enqueues inverse-frequency sequence NeoX RoPE with scaled rotary channels.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    inv_freq: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    start_position: usize,
    attention_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream(
        tokens,
        heads,
        head_dim,
        rotary_dim,
        input,
        inv_freq,
        output,
        0,
        start_position,
        attention_scale,
        stream,
    )
}

/// Enqueues scaled inverse-frequency RoPE at a token offset in dense buffers.
#[allow(clippy::too_many_arguments)]
pub fn rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    input: &DeviceBuffer<f32>,
    inv_freq: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    input_token_offset: usize,
    start_position: usize,
    attention_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let end_tokens = input_token_offset.saturating_add(tokens);
    let required = end_tokens.saturating_mul(heads).saturating_mul(head_dim);
    if tokens == 0
        || heads == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || input.len() < required
        || output.len() < required
        || inv_freq.len() != rotary_dim / 2
        || tokens > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || rotary_dim > u32::MAX as usize
        || input_token_offset > u32::MAX as usize
        || start_position > u32::MAX as usize
        || !attention_scale.is_finite()
        || attention_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "inverse-frequency sequence RoPE",
            expected: "matching non-empty sequence, head, rotary, and frequency dimensions"
                .to_string(),
            actual: format!(
                "tokens={tokens} heads={heads} head_dim={head_dim} rotary_dim={rotary_dim} input={} output={} inv_freq={} input_token_offset={input_token_offset} start_position={start_position} attention_scale={attention_scale}",
                input.len(),
                output.len(),
                inv_freq.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_rope_neox_inv_freq_sequence_f32_on_stream",
            ffi::infer_rope_neox_inv_freq_sequence_f32_on_stream(
                input.ptr,
                inv_freq.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                heads as u32,
                head_dim as u32,
                rotary_dim as u32,
                input_token_offset as u32,
                start_position as u32,
                attention_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues elementwise f32 addition into an existing output buffer on
/// `stream`.
pub fn add_f32_into_on_stream(
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    validate_add_f32(left, right)?;
    if output.len() != left.len() {
        return Err(Error::Shape {
            label: "f32 add output",
            expected: format!("{} values", left.len()),
            actual: format!("{} values", output.len()),
        });
    }

    add_f32_prefix_into_on_stream(left, right, output, left.len(), stream)
}

/// Enqueues elementwise f32 addition for an active prefix on `stream`.
pub fn add_f32_prefix_into_on_stream(
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0
        || len > u32::MAX as usize
        || left.len() < len
        || right.len() < len
        || output.len() < len
    {
        return Err(Error::Shape {
            label: "f32 prefix add",
            expected: format!("left/right/output at least {len} values"),
            actual: format!(
                "left={} right={} output={} active={len}",
                left.len(),
                right.len(),
                output.len()
            ),
        });
    }

    unsafe {
        check_cuda(
            "infer_add_f32_on_stream",
            ffi::infer_add_f32_on_stream(
                left.ptr,
                right.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_add_f32(left: &DeviceBuffer<f32>, right: &DeviceBuffer<f32>) -> Result<()> {
    if left.len() != right.len() {
        return Err(Error::Shape {
            label: "f32 add",
            expected: format!("{} values", left.len()),
            actual: format!("{} values", right.len()),
        });
    }
    if left.is_empty() || left.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "f32 add",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{} values", left.len()),
        });
    }
    Ok(())
}

/// Concatenates row-major `[rows, cols]` f32 inputs into row-major
/// `[rows, 2 * cols]` output on `stream`.
pub fn concat_f32_rows_into_on_stream(
    rows: usize,
    cols: usize,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "concatenate f32 rows",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let output_len = input_len.checked_mul(2).ok_or_else(|| Error::Shape {
        label: "concatenate f32 row output",
        expected: "2 * rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize / 2 {
        return Err(Error::Shape {
            label: "concatenate f32 row dimensions",
            expected: "non-zero u32-sized rows and doubled columns".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }
    for (label, actual) in [
        ("concatenate f32 left input", left.len()),
        ("concatenate f32 right input", right.len()),
    ] {
        if actual != input_len {
            return Err(Error::Shape {
                label,
                expected: format!("{input_len} values"),
                actual: format!("{actual} values"),
            });
        }
    }
    if output.len() != output_len {
        return Err(Error::Shape {
            label: "concatenate f32 row output",
            expected: format!("{output_len} values"),
            actual: format!("{} values", output.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_concat_f32_rows_on_stream",
            ffi::infer_concat_f32_rows_on_stream(
                left.ptr,
                right.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Copies a row-major matrix into a contiguous column range of a wider
/// row-major matrix on `stream`.
#[allow(clippy::too_many_arguments)]
pub fn copy_f32_rows_into_columns_on_stream(
    rows: usize,
    input_cols: usize,
    output_cols: usize,
    output_col_offset: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.saturating_mul(input_cols);
    let output_len = rows.saturating_mul(output_cols);
    if rows == 0
        || input_cols == 0
        || output_cols == 0
        || rows > u32::MAX as usize
        || input_cols > u32::MAX as usize
        || output_cols > u32::MAX as usize
        || output_col_offset > output_cols
        || input_cols > output_cols - output_col_offset
        || input.len() < input_len
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "copy f32 rows into columns",
            expected: format!(
                "input at least {input_len}, output at least {output_len}, and columns within output"
            ),
            actual: format!(
                "input={} output={} rows={rows} input_cols={input_cols} output_cols={output_cols} offset={output_col_offset}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_copy_f32_rows_into_columns_on_stream",
            ffi::infer_copy_f32_rows_into_columns_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                input_cols as u32,
                output_cols as u32,
                output_col_offset as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Adds `increment` to every u32 value on `stream`.
pub fn increment_u32_in_place_on_stream(
    mut values: DeviceInOut<'_, u32>,
    increment: u32,
    stream: &CudaStream,
) -> Result<()> {
    if values.is_empty() || values.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "increment u32",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{} values", values.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_increment_u32_on_stream",
            ffi::infer_increment_u32_on_stream(
                values.as_mut_ptr().cast(),
                values.len() as u32,
                increment,
                stream.as_raw(),
            ),
        )
    }
}

/// Stores one dense u32 input vector into a column of a row-major matrix.
pub fn store_u32_column_into_on_stream(
    input: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, u32>,
    rows: usize,
    columns: usize,
    column: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || columns == 0
        || column >= columns
        || rows > u32::MAX as usize
        || columns > u32::MAX as usize
        || input.len() != rows
        || output.len() != rows.saturating_mul(columns)
    {
        return Err(Error::Shape {
            label: "u32 matrix-column store",
            expected: format!(
                "input={rows} output={} column < {columns}",
                rows.saturating_mul(columns)
            ),
            actual: format!(
                "input={} output={} rows={rows} columns={columns} column={column}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_store_u32_column_on_stream",
            ffi::infer_store_u32_column_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                columns as u32,
                column as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Prepends one dense u32 value to each row of a row-major matrix.
pub fn prepend_u32_rows_into_on_stream(
    first: &DeviceBuffer<u32>,
    remaining: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, u32>,
    rows: usize,
    remaining_columns: usize,
    stream: &CudaStream,
) -> Result<()> {
    let remaining_len = rows.saturating_mul(remaining_columns);
    let output_len = rows.saturating_mul(remaining_columns.saturating_add(1));
    if rows == 0
        || remaining_columns == 0
        || rows > u32::MAX as usize
        || remaining_columns >= u32::MAX as usize
        || first.len() != rows
        || remaining.len() != remaining_len
        || output.len() != output_len
    {
        return Err(Error::Shape {
            label: "prepend u32 rows",
            expected: format!("first={rows} remaining={remaining_len} output={output_len}"),
            actual: format!(
                "first={} remaining={} output={} rows={rows} remaining_columns={remaining_columns}",
                first.len(),
                remaining.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_prepend_u32_rows_on_stream",
            ffi::infer_prepend_u32_rows_on_stream(
                first.ptr,
                remaining.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                remaining_columns as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Transposes row-major `[rows, cols]` f32 into column-major `[rows, cols]`.
#[cfg(test)]
pub fn row_major_to_col_major_f32(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
) -> Result<DeviceBuffer<f32>> {
    transpose_layout_f32(
        "row-major to column-major",
        rows,
        cols,
        input,
        ffi::infer_row_major_to_col_major_f32,
    )
}

/// Transposes column-major `[rows, cols]` f32 into row-major `[rows, cols]`.
#[cfg(test)]
pub fn col_major_to_row_major_f32(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
) -> Result<DeviceBuffer<f32>> {
    transpose_layout_f32(
        "column-major to row-major",
        rows,
        cols,
        input,
        ffi::infer_col_major_to_row_major_f32,
    )
}

#[cfg(test)]
fn transpose_layout_f32(
    label: &'static str,
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    kernel: unsafe extern "C" fn(*const f32, *mut f32, u32, u32) -> ffi::cudaError_t,
) -> Result<DeviceBuffer<f32>> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label,
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != len {
        return Err(Error::Shape {
            label,
            expected: format!("{len} values"),
            actual: format!("{} values", input.len()),
        });
    }
    if len == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label,
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }

    let output = DeviceBuffer::zeroed(len)?;
    unsafe {
        check_cuda(
            label,
            kernel(input.ptr, output.ptr, rows as u32, cols as u32),
        )?;
    }
    Ok(output)
}

/// Copies one row from row-major `[rows, cols]` f32 input into a new vector.
#[cfg(test)]
pub fn copy_row_f32(
    rows: usize,
    cols: usize,
    row: usize,
    input: &DeviceBuffer<f32>,
) -> Result<DeviceBuffer<f32>> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy row f32",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != len {
        return Err(Error::Shape {
            label: "copy row f32 input",
            expected: format!("{len} values"),
            actual: format!("{} values", input.len()),
        });
    }
    if row >= rows || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "copy row f32 dimensions",
            expected: "row < rows and u32-sized non-zero cols".to_string(),
            actual: format!("rows={rows} cols={cols} row={row}"),
        });
    }

    let output = DeviceBuffer::zeroed(cols)?;
    unsafe {
        check_cuda(
            "infer_copy_row_f32",
            ffi::infer_copy_row_f32(input.ptr, output.ptr, row as u32, cols as u32),
        )?;
    }
    Ok(output)
}

/// Enqueues a row copy from row-major `[rows, cols]` f32 input into `output`.
pub fn copy_row_f32_into_on_stream(
    rows: usize,
    cols: usize,
    row: usize,
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    validate_copy_row("copy row f32", rows, cols, row, input.len(), output.len())?;
    unsafe {
        check_cuda(
            "infer_copy_row_f32_on_stream",
            ffi::infer_copy_row_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                row as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Gathers `values[indices[i]] * multipliers[i]` into `output` on `stream`.
///
/// Out-of-range source indices produce zero, matching the missing-expert
/// behaviour of the routed-MoE helpers.
pub fn gather_indexed_mul_f32_into_on_stream(
    values: &DeviceBuffer<f32>,
    indices: &DeviceBuffer<u32>,
    multipliers: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    gather_indexed_mul_f32_prefix_into_on_stream(
        values,
        indices,
        multipliers,
        output,
        indices.len(),
        stream,
    )
}

/// Gathers an active prefix of `values[indices[i]] * multipliers[i]`.
pub fn gather_indexed_mul_f32_prefix_into_on_stream(
    values: &DeviceBuffer<f32>,
    indices: &DeviceBuffer<u32>,
    multipliers: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    count: usize,
    stream: &CudaStream,
) -> Result<()> {
    if count == 0
        || count > u32::MAX as usize
        || values.is_empty()
        || values.len() > u32::MAX as usize
        || indices.len() < count
        || multipliers.len() < count
        || output.len() < count
    {
        return Err(Error::Shape {
            label: "indexed f32 gather multiply",
            expected: format!(
                "non-empty values, indices/multipliers/output={count}, u32-sized dimensions"
            ),
            actual: format!(
                "values={} indices={} multipliers={} output={}",
                values.len(),
                indices.len(),
                multipliers.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gather_indexed_mul_f32_on_stream",
            ffi::infer_gather_indexed_mul_f32_on_stream(
                values.ptr,
                indices.ptr,
                multipliers.ptr,
                output.buffer_mut().ptr,
                count as u32,
                values.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Gathers the same row from each group of a row-major f32 tensor.
#[allow(clippy::too_many_arguments)]
pub fn gather_group_row_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    groups: usize,
    rows_per_group: usize,
    row: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = groups.saturating_mul(rows_per_group).saturating_mul(cols);
    let output_len = groups.saturating_mul(cols);
    if groups == 0
        || rows_per_group == 0
        || row >= rows_per_group
        || cols == 0
        || groups > u32::MAX as usize
        || rows_per_group > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() != input_len
        || output.len() != output_len
    {
        return Err(Error::Shape {
            label: "gather grouped f32 row",
            expected: format!("input={input_len} output={output_len} row < {rows_per_group}"),
            actual: format!(
                "input={} output={} groups={groups} rows_per_group={rows_per_group} row={row} cols={cols}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gather_group_row_f32_on_stream",
            ffi::infer_gather_group_row_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                groups as u32,
                rows_per_group as u32,
                row as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues BF16 row-to-f32 copy selected by a device-resident row index.
pub fn copy_bf16_row_to_f32_indexed_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<u16>,
    row: &DeviceBuffer<u32>,
    output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let mut output = output;
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy BF16 row to f32 input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != len || row.len() != 1 || output.len() != cols {
        return Err(Error::Shape {
            label: "copy BF16 row to f32 buffers",
            expected: format!("input={len} row=1 output={cols}"),
            actual: format!(
                "input={} row={} output={}",
                input.len(),
                row.len(),
                output.len()
            ),
        });
    }
    if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "copy BF16 row to f32 dimensions",
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }

    unsafe {
        check_cuda(
            "infer_copy_bf16_row_to_f32_indexed_on_stream",
            ffi::infer_copy_bf16_row_to_f32_indexed_on_stream(
                input.ptr,
                row.ptr,
                output.buffer_mut().ptr,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Copies one host-selected BF16 row to f32 on `stream`.
pub fn copy_bf16_row_to_f32_into_on_stream(
    rows: usize,
    cols: usize,
    row: usize,
    input: &DeviceBuffer<u16>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy BF16 row to f32 input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || row >= rows
        || row > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() != input_len
        || output.len() != cols
    {
        return Err(Error::Shape {
            label: "copy BF16 row to f32 buffers",
            expected: format!("input={input_len} row < {rows} output={cols}"),
            actual: format!(
                "input={} row={row} output={} rows={rows} cols={cols}",
                input.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_copy_bf16_row_to_f32_on_stream",
            ffi::infer_copy_bf16_row_to_f32_on_stream(
                input.ptr,
                row as u32,
                output.buffer_mut().ptr,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Copies one BF16 embedding row per device-resident index into a dense batch.
pub fn copy_bf16_rows_to_f32_indexed_into_on_stream(
    vocab_rows: usize,
    cols: usize,
    input: &DeviceBuffer<u16>,
    rows: &DeviceBuffer<u32>,
    output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
        vocab_rows,
        cols,
        input,
        rows,
        output,
        rows.len(),
        stream,
    )
}

/// Copies an active prefix of device-resident embedding indices into a dense batch.
pub fn copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
    vocab_rows: usize,
    cols: usize,
    input: &DeviceBuffer<u16>,
    rows: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    row_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = vocab_rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy BF16 rows to f32 input",
        expected: "vocab_rows * cols without overflow".to_string(),
        actual: format!("vocab_rows={vocab_rows} cols={cols}"),
    })?;
    let output_len = row_count.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy BF16 rows to f32 output",
        expected: "batch_size * cols without overflow".to_string(),
        actual: format!("batch_size={row_count} cols={cols}"),
    })?;
    if vocab_rows == 0
        || cols == 0
        || row_count == 0
        || vocab_rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || row_count > u32::MAX as usize
        || input.len() != input_len
        || rows.len() < row_count
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "copy BF16 rows to f32 buffers",
            expected: format!("input={input_len} rows>0 output={output_len}"),
            actual: format!(
                "input={} rows={} output={}",
                input.len(),
                row_count,
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_copy_bf16_rows_to_f32_indexed_on_stream",
            ffi::infer_copy_bf16_rows_to_f32_indexed_on_stream(
                input.ptr,
                rows.ptr,
                output.buffer_mut().ptr,
                row_count as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Copies FP8 embedding rows to f32 and applies one scale per vocabulary row.
pub fn copy_fp8_rows_to_f32_indexed_prefix_into_on_stream(
    vocab_rows: usize,
    cols: usize,
    input: &DeviceBuffer<u8>,
    row_scales: &DeviceBuffer<f32>,
    rows: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    row_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = vocab_rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy FP8 rows to f32 input",
        expected: "vocab_rows * cols without overflow".to_string(),
        actual: format!("vocab_rows={vocab_rows} cols={cols}"),
    })?;
    let output_len = row_count.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "copy FP8 rows to f32 output",
        expected: "row_count * cols without overflow".to_string(),
        actual: format!("row_count={row_count} cols={cols}"),
    })?;
    if vocab_rows == 0
        || cols == 0
        || row_count == 0
        || vocab_rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || row_count > u32::MAX as usize
        || input.len() != input_len
        || row_scales.len() != vocab_rows
        || rows.len() < row_count
        || output.len() < output_len
    {
        return Err(Error::Shape {
            label: "copy FP8 rows to f32 buffers",
            expected: format!(
                "input={input_len} scales={vocab_rows} rows>={row_count} output>={output_len}"
            ),
            actual: format!(
                "input={} scales={} rows={} output={}",
                input.len(),
                row_scales.len(),
                rows.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_copy_fp8_rows_to_f32_indexed_on_stream",
            ffi::infer_copy_fp8_rows_to_f32_indexed_on_stream(
                input.ptr,
                row_scales.ptr,
                rows.ptr,
                output.buffer_mut().ptr,
                row_count as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_copy_row(
    label: &'static str,
    rows: usize,
    cols: usize,
    row: usize,
    input_len: usize,
    output_len: usize,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label,
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input_len != len || output_len != cols {
        return Err(Error::Shape {
            label,
            expected: format!("input={len} output={cols}"),
            actual: format!("input={input_len} output={output_len}"),
        });
    }
    if row >= rows || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label,
            expected: "row < rows and u32-sized non-zero cols".to_string(),
            actual: format!("rows={rows} cols={cols} row={row}"),
        });
    }
    Ok(())
}

/// Quantizes a device-resident column-major f32 matrix to NVFP4.
///
/// `input_scale` follows the ModelOpt W4A4 activation convention: input values
/// are divided by this tensor-wide scale before per-16-value block
/// quantization. The returned matrix contains packed E2M1 values and cuBLASLt
/// tiled UE4M3 scales.
pub fn quantize_nvfp4_col_major_f32_device(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    input_scale: f32,
) -> Result<Nvfp4Matrix> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "NVFP4 device quantization input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != len {
        return Err(Error::Shape {
            label: "NVFP4 device quantization input",
            expected: format!("{len} values"),
            actual: format!("{} values", input.len()),
        });
    }
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "NVFP4 device quantization dimensions",
            expected: "u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }
    if !input_scale.is_finite() || input_scale <= 0.0 {
        return Err(Error::Format {
            label: "NVFP4 device quantization input_scale",
            detail: format!("expected positive finite scale, got {input_scale}"),
        });
    }

    let packed = DeviceBuffer::zeroed((rows * cols).div_ceil(2))?;
    let scales = DeviceBuffer::zeroed(format::ue4m3_scale_layout_len(cols, rows))?;
    unsafe {
        check_cuda(
            "infer_quantize_nvfp4_col_major_f32",
            ffi::infer_quantize_nvfp4_col_major_f32(
                input.ptr,
                packed.ptr,
                scales.ptr,
                rows as u32,
                cols as u32,
                input_scale,
            ),
        )?;
    }
    Nvfp4Matrix::from_device_col_major_parts(rows, cols, packed, scales)
}

/// Enqueues NVFP4 quantization into an existing matrix on `stream`.
pub fn quantize_nvfp4_col_major_f32_device_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "NVFP4 device quantization input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() < len {
        return Err(Error::Shape {
            label: "NVFP4 device quantization input",
            expected: format!("at least {len} values"),
            actual: format!("{} values", input.len()),
        });
    }
    if (output.rows, output.cols) != (rows, cols) {
        return Err(Error::Shape {
            label: "NVFP4 device quantization output",
            expected: format!("{rows}x{cols}"),
            actual: format!("{}x{}", output.rows, output.cols),
        });
    }
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "NVFP4 device quantization dimensions",
            expected: "u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }
    if !input_scale.is_finite() || input_scale <= 0.0 {
        return Err(Error::Format {
            label: "NVFP4 device quantization input_scale",
            detail: format!("expected positive finite scale, got {input_scale}"),
        });
    }

    let mut output = output.output();
    let values = output.values_mut_ptr().cast();
    let scales = output.scales_mut_ptr().cast();
    unsafe {
        check_cuda(
            "infer_quantize_nvfp4_col_major_f32_on_stream",
            ffi::infer_quantize_nvfp4_col_major_f32_on_stream(
                input.ptr,
                values,
                scales,
                rows as u32,
                cols as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Fuses row-wise RMSNorm with column-major NVFP4 activation quantization.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    eps: f32,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "RMSNorm NVFP4 quantization input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || input.len() < len
        || weight.len() != cols
        || output.rows != cols
        || output.cols < rows
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !eps.is_finite()
        || eps < 0.0
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "RMSNorm NVFP4 quantization buffers",
            expected: format!(
                "input={len} weight={cols} output={cols}x{rows} with valid dimensions and scales"
            ),
            actual: format!(
                "input={} weight={} output={}x{} eps={eps} input_scale={input_scale}",
                input.len(),
                weight.len(),
                output.rows,
                output.cols
            ),
        });
    }
    let mut output = output.output();
    unsafe {
        check_cuda(
            "infer_rms_norm_quantize_nvfp4_col_major_f32_on_stream",
            ffi::infer_rms_norm_quantize_nvfp4_col_major_f32_on_stream(
                input.ptr,
                weight.ptr,
                output.values_mut_ptr().cast(),
                output.scales_mut_ptr().cast(),
                rows as u32,
                cols as u32,
                eps,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Quantizes an RMS-normalized matrix as a primary FP4 term and FP4 residual.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_quantize_nvfp4_pair_col_major_f32_into_on_stream(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    residual_output: &mut Nvfp4Matrix,
    eps: f32,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "paired RMSNorm NVFP4 quantization input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || input.len() < len
        || weight.len() != cols
        || output.rows != cols
        || output.cols < rows
        || residual_output.rows != cols
        || residual_output.cols < rows
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !eps.is_finite()
        || eps < 0.0
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "paired RMSNorm NVFP4 quantization buffers",
            expected: format!(
                "input={len} weight={cols} outputs={cols}x{rows} with valid dimensions and scales"
            ),
            actual: format!(
                "input={} weight={} output={}x{} residual={}x{} eps={eps} input_scale={input_scale}",
                input.len(),
                weight.len(),
                output.rows,
                output.cols,
                residual_output.rows,
                residual_output.cols
            ),
        });
    }
    let mut output = output.output();
    let mut residual_output = residual_output.output();
    unsafe {
        check_cuda(
            "infer_rms_norm_quantize_nvfp4_pair_col_major_f32_on_stream",
            ffi::infer_rms_norm_quantize_nvfp4_pair_col_major_f32_on_stream(
                input.ptr,
                weight.ptr,
                output.values_mut_ptr().cast(),
                output.scales_mut_ptr().cast(),
                residual_output.values_mut_ptr().cast(),
                residual_output.scales_mut_ptr().cast(),
                rows as u32,
                cols as u32,
                eps,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Fuses gated GELU-tanh activation with column-major NVFP4 quantization.
#[allow(clippy::too_many_arguments)]
pub fn gelu_tanh_mul_quantize_nvfp4_col_major_f32_into_on_stream(
    rows: usize,
    cols: usize,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "GELU NVFP4 quantization input",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || gate.len() < len
        || up.len() < len
        || output.rows != cols
        || output.cols < rows
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "GELU NVFP4 quantization buffers",
            expected: format!("gate/up={len} output={cols}x{rows} with valid scale"),
            actual: format!(
                "gate={} up={} output={}x{} input_scale={input_scale}",
                gate.len(),
                up.len(),
                output.rows,
                output.cols
            ),
        });
    }
    let mut output = output.output();
    unsafe {
        check_cuda(
            "infer_gelu_tanh_mul_quantize_nvfp4_col_major_f32_on_stream",
            ffi::infer_gelu_tanh_mul_quantize_nvfp4_col_major_f32_on_stream(
                gate.ptr,
                up.ptr,
                output.values_mut_ptr().cast(),
                output.scales_mut_ptr().cast(),
                rows as u32,
                cols as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn quantize_nvfp4_vector_simple_scales_f32_into_on_stream(
    rows: usize,
    input: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    simple_scales: &mut DeviceBuffer<u8>,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || input.len() != rows
        || output.rows != rows
        || output.cols != 1
        || simple_scales.len() != rows.div_ceil(16)
        || rows > u32::MAX as usize
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "NVFP4 vector simple-scale quantization",
            expected: "matching one-column output and simple scales".to_string(),
            actual: format!(
                "rows={rows} input={} output={}x{} scales={} input_scale={input_scale}",
                input.len(),
                output.rows,
                output.cols,
                simple_scales.len()
            ),
        });
    }
    let mut output = output.output();
    unsafe {
        check_cuda(
            "infer_quantize_nvfp4_vector_simple_scales_f32_on_stream",
            ffi::infer_quantize_nvfp4_vector_simple_scales_f32_on_stream(
                input.ptr,
                output.values_mut_ptr().cast(),
                simple_scales.ptr,
                rows as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Quantizes a flat f32 device buffer to packed NVFP4 E2M1 with one UE4M3
/// scale per 16 consecutive values.
///
/// This is a compact streaming layout for cache experiments, not cuBLASLt's
/// tiled matrix-scale layout.
pub fn quantize_nvfp4_simple_scales_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty()
        || packed.len() != input.len().div_ceil(2)
        || scales.len() != input.len().div_ceil(16)
        || input.len() > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "NVFP4 simple-scale quantization",
            expected: "non-empty input with packed=len/2 and scales=len/16".to_string(),
            actual: format!(
                "input={} packed={} scales={}",
                input.len(),
                packed.len(),
                scales.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_quantize_nvfp4_vector_simple_scales_f32_on_stream",
            ffi::infer_quantize_nvfp4_vector_simple_scales_f32_on_stream(
                input.ptr,
                packed.ptr,
                scales.ptr,
                input.len() as u32,
                1.0,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one-token GQA over an NVFP4 K/V cache.
///
/// K/V use packed E2M1 values with one UE4M3 scale per 16 consecutive cache
/// elements. Q and online-softmax accumulation stay f32. This is a focused
/// cache-format probe and does not yet issue FP4 MMA instructions.
pub fn cached_gqa_attention_nvfp4_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<u8>,
    key_scales: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    value_scales: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_len = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "NVFP4 cached GQA query",
        expected: "q_heads * head_dim without overflow".to_string(),
        actual: format!("q_heads={q_heads} head_dim={head_dim}"),
    })?;
    let cache_values = cache_len
        .checked_mul(kv_heads)
        .and_then(|len| len.checked_mul(head_dim))
        .ok_or_else(|| Error::Shape {
            label: "NVFP4 cached GQA cache",
            expected: "cache_len * kv_heads * head_dim without overflow".to_string(),
            actual: format!("cache_len={cache_len} kv_heads={kv_heads} head_dim={head_dim}"),
        })?;
    if query.len() != query_len
        || output.len() != query_len
        || key_cache.len() != cache_values.div_ceil(2)
        || value_cache.len() != cache_values.div_ceil(2)
        || key_scales.len() != cache_values.div_ceil(16)
        || value_scales.len() != cache_values.div_ceil(16)
        || cache_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || head_dim > 256
        || !q_heads.is_multiple_of(kv_heads)
        || cache_len > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "NVFP4 cached GQA dimensions",
            expected: "matching packed K/V, per-16 scales, and valid GQA dimensions".to_string(),
            actual: format!(
                "query={} output={} cache_values={cache_values} key={} key_scales={} value={} value_scales={} cache_len={cache_len} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}",
                query.len(),
                output.len(),
                key_cache.len(),
                key_scales.len(),
                value_cache.len(),
                value_scales.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_cached_gqa_attention_nvfp4_on_stream",
            ffi::infer_cached_gqa_attention_nvfp4_on_stream(
                query.ptr,
                key_cache.ptr,
                key_scales.ptr,
                value_cache.ptr,
                value_scales.ptr,
                output.buffer_mut().ptr,
                cache_len as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies a numerically stable f32 softmax to one device-resident vector.
pub fn softmax_f32_in_place_on_stream(
    values: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if values.is_empty() || values.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "f32 softmax",
            expected: "non-empty u32-sized vector".to_string(),
            actual: values.len().to_string(),
        });
    }
    unsafe {
        check_cuda(
            "infer_softmax_f32_in_place_on_stream",
            ffi::infer_softmax_f32_in_place_on_stream(
                values.ptr,
                values.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn silu_mul_halves_quantize_nvfp4_col_major_f32_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    if output.cols != 1 {
        return Err(Error::Shape {
            label: "SiLU halves NVFP4 quantization output",
            expected: "one output column".to_string(),
            actual: format!("{} columns", output.cols),
        });
    }
    let rows = output.rows;
    if gate_up.len() != rows * 2 {
        return Err(Error::Shape {
            label: "SiLU halves NVFP4 quantization input",
            expected: format!("{} values", rows * 2),
            actual: format!("{} values", gate_up.len()),
        });
    }
    if rows > u32::MAX as usize {
        return Err(Error::Shape {
            label: "SiLU halves NVFP4 quantization rows",
            expected: "u32-sized rows".to_string(),
            actual: rows.to_string(),
        });
    }
    if !input_scale.is_finite() || input_scale <= 0.0 {
        return Err(Error::Format {
            label: "SiLU halves NVFP4 quantization input_scale",
            detail: format!("expected positive finite scale, got {input_scale}"),
        });
    }

    let mut output = output.output();
    let values = output.values_mut_ptr().cast();
    let scales = output.scales_mut_ptr().cast();
    unsafe {
        check_cuda(
            "infer_silu_mul_halves_quantize_nvfp4_col_major_f32_on_stream",
            ffi::infer_silu_mul_halves_quantize_nvfp4_col_major_f32_on_stream(
                gate_up.ptr,
                values,
                scales,
                rows as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs the degenerate sequence-length-1 grouped-query attention value path.
///
/// With one cached key/value token, causal softmax assigns probability 1 to
/// that token. The output is therefore the value head copied to each query head
/// in its GQA group.
#[cfg(test)]
pub fn single_token_gqa_attention_f32(
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<DeviceBuffer<f32>> {
    let value_len = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "single-token GQA value",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    if value.len() != value_len {
        return Err(Error::Shape {
            label: "single-token GQA value",
            expected: format!("{value_len} values"),
            actual: format!("{} values", value.len()),
        });
    }
    if key.len() != value_len {
        return Err(Error::Shape {
            label: "single-token GQA key",
            expected: format!("{value_len} values"),
            actual: format!("{} values", key.len()),
        });
    }
    if q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "single-token GQA dimensions",
            expected: "non-zero u32-sized dims with q_heads divisible by kv_heads".to_string(),
            actual: format!("q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"),
        });
    }

    let output = DeviceBuffer::zeroed(q_heads * head_dim)?;
    unsafe {
        check_cuda(
            "infer_single_token_gqa_f32",
            ffi::infer_single_token_gqa_f32(
                key.ptr,
                value.ptr,
                output.ptr,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
            ),
        )?;
    }
    Ok(output)
}

/// Enqueues row append into a preallocated row-major destination on `stream`.
pub fn append_rows_f32_into_on_stream(
    src: &DeviceBuffer<f32>,
    mut dst: DeviceOutput<'_, f32>,
    dst_start_row: usize,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_append_rows_f32(src.len(), dst.len(), dst_start_row, rows, cols)?;
    unsafe {
        check_cuda(
            "infer_append_rows_f32_on_stream",
            ffi::infer_append_rows_f32_on_stream(
                src.ptr,
                dst.buffer_mut().ptr,
                dst_start_row as u32,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues a row append using a device-resident destination start row.
pub fn append_rows_f32_indexed_into_on_stream(
    src: &DeviceBuffer<f32>,
    mut dst: DeviceOutput<'_, f32>,
    dst_start_row: &DeviceBuffer<u32>,
    max_start_row: usize,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_append_rows_f32(src.len(), dst.len(), max_start_row, rows, cols)?;
    if dst_start_row.len() != 1 {
        return Err(Error::Shape {
            label: "f32 indexed row append start",
            expected: "1 value".to_string(),
            actual: format!("{} values", dst_start_row.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_append_rows_f32_indexed_on_stream",
            ffi::infer_append_rows_f32_indexed_on_stream(
                src.ptr,
                dst.buffer_mut().ptr,
                dst_start_row.ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_append_rows_f32(
    src_len_actual: usize,
    dst_len_actual: usize,
    dst_start_row: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let src_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "f32 row append source",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let dst_required = dst_start_row
        .checked_add(rows)
        .and_then(|row| row.checked_mul(cols))
        .ok_or_else(|| Error::Shape {
            label: "f32 row append destination",
            expected: "(dst_start_row + rows) * cols without overflow".to_string(),
            actual: format!("dst_start_row={dst_start_row} rows={rows} cols={cols}"),
        })?;
    if src_len_actual != src_len {
        return Err(Error::Shape {
            label: "f32 row append source",
            expected: format!("{src_len} values"),
            actual: format!("{} values", src_len_actual),
        });
    }
    if dst_len_actual < dst_required {
        return Err(Error::Shape {
            label: "f32 row append destination",
            expected: format!("at least {dst_required} values"),
            actual: format!("{} values", dst_len_actual),
        });
    }
    if dst_start_row > u32::MAX as usize
        || rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "f32 row append dimensions",
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("dst_start_row={dst_start_row} rows={rows} cols={cols}"),
        });
    }
    Ok(())
}

/// Runs the degenerate one-token GQA value path from cached K/V rows.
///
/// This is the cache-backed form of [`single_token_gqa_attention_f32`]. It
/// reads the K/V row at `position` from row-major cache buffers of width
/// `kv_heads * head_dim`.
#[cfg(test)]
pub fn single_token_gqa_attention_f32_from_cache(
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    position: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<DeviceBuffer<f32>> {
    let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cached GQA width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    let required = position
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(kv_width))
        .ok_or_else(|| Error::Shape {
            label: "cached GQA buffer",
            expected: "(position + 1) * kv_width without overflow".to_string(),
            actual: format!("position={position} kv_width={kv_width}"),
        })?;
    if key_cache.len() < required {
        return Err(Error::Shape {
            label: "cached GQA key",
            expected: format!("at least {required} values"),
            actual: format!("{} values", key_cache.len()),
        });
    }
    if value_cache.len() < required {
        return Err(Error::Shape {
            label: "cached GQA value",
            expected: format!("at least {required} values"),
            actual: format!("{} values", value_cache.len()),
        });
    }
    if q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || position > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "cached GQA dimensions",
            expected: "non-zero u32-sized dims with q_heads divisible by kv_heads".to_string(),
            actual: format!(
                "position={position} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"
            ),
        });
    }

    let output = DeviceBuffer::zeroed(q_heads * head_dim)?;
    unsafe {
        check_cuda(
            "infer_single_token_gqa_f32_from_cache",
            ffi::infer_single_token_gqa_f32_from_cache(
                key_cache.ptr,
                value_cache.ptr,
                output.ptr,
                position as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
            ),
        )?;
    }
    Ok(output)
}

/// Runs one-token grouped-query attention over cached K/V prefix rows.
///
/// `query` is one row laid out as `[q_heads, head_dim]`. K/V caches are
/// row-major `[max_tokens, kv_heads * head_dim]`; only the first `cache_len`
/// rows participate. The output is `[q_heads, head_dim]`. The current kernel
/// supports `head_dim <= 256`.
#[cfg(test)]
pub fn cached_gqa_attention_f32(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<DeviceBuffer<f32>> {
    let query_len = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cached GQA query",
        expected: "q_heads * head_dim without overflow".to_string(),
        actual: format!("q_heads={q_heads} head_dim={head_dim}"),
    })?;
    let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cached GQA width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    let required_cache = cache_len
        .checked_mul(kv_width)
        .ok_or_else(|| Error::Shape {
            label: "cached GQA buffer",
            expected: "cache_len * kv_width without overflow".to_string(),
            actual: format!("cache_len={cache_len} kv_width={kv_width}"),
        })?;
    if query.len() != query_len {
        return Err(Error::Shape {
            label: "cached GQA query",
            expected: format!("{query_len} values"),
            actual: format!("{} values", query.len()),
        });
    }
    if key_cache.len() < required_cache {
        return Err(Error::Shape {
            label: "cached GQA key",
            expected: format!("at least {required_cache} values"),
            actual: format!("{} values", key_cache.len()),
        });
    }
    if value_cache.len() < required_cache {
        return Err(Error::Shape {
            label: "cached GQA value",
            expected: format!("at least {required_cache} values"),
            actual: format!("{} values", value_cache.len()),
        });
    }
    if cache_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || head_dim > 256
        || !q_heads.is_multiple_of(kv_heads)
        || cache_len > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "cached GQA dimensions",
            expected: "non-zero u32-sized dims, head_dim <= 256, and q_heads divisible by kv_heads"
                .to_string(),
            actual: format!(
                "cache_len={cache_len} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"
            ),
        });
    }

    let output = DeviceBuffer::zeroed(query_len)?;
    unsafe {
        check_cuda(
            "infer_cached_gqa_attention_f32",
            ffi::infer_cached_gqa_attention_f32(
                query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                output.ptr,
                cache_len as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
            ),
        )?;
    }
    Ok(output)
}

/// Enqueues one-token grouped-query attention into an existing output buffer on
/// `stream`.
pub fn cached_gqa_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_len = validate_cached_gqa_attention_f32(
        query,
        key_cache,
        value_cache,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
    )?;
    if output.len() != query_len {
        return Err(Error::Shape {
            label: "cached GQA output",
            expected: format!("{query_len} values"),
            actual: format!("{} values", output.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_cached_gqa_attention_f32_on_stream",
            ffi::infer_cached_gqa_attention_f32_on_stream(
                query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                output.buffer_mut().ptr,
                cache_len as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Captures one target residual tap into row-major `[row, tap, hidden]` storage.
pub fn dflash2_capture_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    hidden: usize,
    taps: usize,
    tap: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_values = rows.checked_mul(hidden).ok_or_else(|| Error::Shape {
        label: "DFlash2 target capture",
        expected: "rows * hidden without overflow".to_string(),
        actual: format!("rows={rows} hidden={hidden}"),
    })?;
    let output_values = input_values.checked_mul(taps).ok_or_else(|| Error::Shape {
        label: "DFlash2 target capture",
        expected: "rows * hidden * taps without overflow".to_string(),
        actual: format!("rows={rows} hidden={hidden} taps={taps}"),
    })?;
    if rows == 0
        || hidden == 0
        || taps == 0
        || tap >= taps
        || input.len() < input_values
        || output.len() < output_values
        || [rows, hidden, taps, tap]
            .iter()
            .any(|&value| value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "DFlash2 target capture",
            expected: format!("input>={input_values}, output>={output_values}, and tap < {taps}"),
            actual: format!(
                "input={} output={} rows={rows} hidden={hidden} taps={taps} tap={tap}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_dflash2_capture_f32_on_stream",
            ffi::infer_dflash2_capture_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                hidden as u32,
                taps as u32,
                tap as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies one side of DFlash2's dynamic grouped convolution.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_grouped_conv_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    coefficients: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    hidden: usize,
    groups: usize,
    taps: usize,
    block_size: usize,
    side: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = rows.checked_mul(hidden).ok_or_else(|| Error::Shape {
        label: "DFlash2 grouped convolution",
        expected: "rows * hidden without overflow".to_string(),
        actual: format!("rows={rows} hidden={hidden}"),
    })?;
    let coefficient_values = rows
        .checked_mul(2)
        .and_then(|value| value.checked_mul(taps))
        .and_then(|value| value.checked_mul(groups))
        .ok_or_else(|| Error::Shape {
            label: "DFlash2 grouped convolution coefficients",
            expected: "rows * 2 * taps * groups without overflow".to_string(),
            actual: format!("rows={rows} taps={taps} groups={groups}"),
        })?;
    let base_values = 2usize
        .checked_mul(taps)
        .and_then(|value| value.checked_mul(hidden))
        .ok_or_else(|| Error::Shape {
            label: "DFlash2 grouped convolution base",
            expected: "2 * taps * hidden without overflow".to_string(),
            actual: format!("taps={taps} hidden={hidden}"),
        })?;
    if rows == 0
        || hidden == 0
        || groups == 0
        || !hidden.is_multiple_of(groups)
        || taps == 0
        || block_size == 0
        || side >= 2
        || input.len() < values
        || output.len() < values
        || coefficients.len() < coefficient_values
        || base.len() < base_values
        || [rows, hidden, groups, taps, block_size, side]
            .iter()
            .any(|&value| value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "DFlash2 grouped convolution",
            expected: format!(
                "input/output>={values}, coefficients>={coefficient_values}, base>={base_values}, side < 2"
            ),
            actual: format!(
                "input={} output={} coefficients={} base={} rows={rows} hidden={hidden} groups={groups} taps={taps} block={block_size} side={side}",
                input.len(),
                output.len(),
                coefficients.len(),
                base.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_dflash2_grouped_conv_f32_on_stream",
            ffi::infer_dflash2_grouped_conv_f32_on_stream(
                input.ptr,
                coefficients.ptr,
                base.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                hidden as u32,
                groups as u32,
                taps as u32,
                block_size as u32,
                side as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Evaluates DFlash2's non-causal proposal-block attention over a ring cache.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_noncausal_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    context_key: &DeviceBuffer<f32>,
    context_value: &DeviceBuffer<f32>,
    block_key: &DeviceBuffer<f32>,
    block_value: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    context_end: usize,
    context_len: usize,
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
    stream: &CudaStream,
) -> Result<()> {
    let q_width = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "DFlash2 attention query width",
        expected: "q_heads * head_dim without overflow".to_string(),
        actual: format!("q_heads={q_heads} head_dim={head_dim}"),
    })?;
    let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "DFlash2 attention KV width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    let query_values = rows * q_width;
    let block_values = rows * kv_width;
    let context_values = window * kv_width;
    if rows == 0
        || rows > window
        || q_heads == 0
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || head_dim == 0
        || head_dim > 256
        || window == 0
        || context_len > window
        || context_len > context_end
        || query.len() < query_values
        || output.len() < query_values
        || block_key.len() < block_values
        || block_value.len() < block_values
        || context_key.len() < context_values
        || context_value.len() < context_values
        || [
            context_end,
            context_len,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            window,
        ]
        .iter()
        .any(|&value| value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "DFlash2 non-causal attention",
            expected: format!(
                "query/output>={query_values}, block KV>={block_values}, context KV>={context_values}"
            ),
            actual: format!(
                "query={} output={} block_key={} block_value={} context_key={} context_value={} context_end={context_end} context_len={context_len} rows={rows}",
                query.len(),
                output.len(),
                block_key.len(),
                block_value.len(),
                context_key.len(),
                context_value.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_dflash2_noncausal_attention_f32_on_stream",
            ffi::infer_dflash2_noncausal_attention_f32_on_stream(
                query.ptr,
                context_key.ptr,
                context_value.ptr,
                block_key.ptr,
                block_value.ptr,
                output.buffer_mut().ptr,
                context_end as u32,
                context_len as u32,
                rows as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                window as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one-token grouped-query attention with device-resident cache length.
pub fn cached_gqa_attention_f32_indexed_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    cache_len: &DeviceBuffer<u32>,
    max_cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_len = validate_cached_gqa_attention_f32(
        query,
        key_cache,
        value_cache,
        max_cache_len,
        q_heads,
        kv_heads,
        head_dim,
    )?;
    if output.len() != query_len {
        return Err(Error::Shape {
            label: "indexed cached GQA output",
            expected: format!("{query_len} values"),
            actual: format!("{} values", output.len()),
        });
    }
    if cache_len.len() != 1 {
        return Err(Error::Shape {
            label: "indexed cached GQA cache_len",
            expected: "1 value".to_string(),
            actual: format!("{} values", cache_len.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_cached_gqa_attention_f32_indexed_on_stream",
            ffi::infer_cached_gqa_attention_f32_indexed_on_stream(
                query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                output.buffer_mut().ptr,
                cache_len.ptr,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_cached_gqa_attention_f32(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<usize> {
    let query_len = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cached GQA query",
        expected: "q_heads * head_dim without overflow".to_string(),
        actual: format!("q_heads={q_heads} head_dim={head_dim}"),
    })?;
    let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cached GQA width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    let required_cache = cache_len
        .checked_mul(kv_width)
        .ok_or_else(|| Error::Shape {
            label: "cached GQA buffer",
            expected: "cache_len * kv_width without overflow".to_string(),
            actual: format!("cache_len={cache_len} kv_width={kv_width}"),
        })?;
    if query.len() != query_len {
        return Err(Error::Shape {
            label: "cached GQA query",
            expected: format!("{query_len} values"),
            actual: format!("{} values", query.len()),
        });
    }
    if key_cache.len() < required_cache {
        return Err(Error::Shape {
            label: "cached GQA key",
            expected: format!("at least {required_cache} values"),
            actual: format!("{} values", key_cache.len()),
        });
    }
    if value_cache.len() < required_cache {
        return Err(Error::Shape {
            label: "cached GQA value",
            expected: format!("at least {required_cache} values"),
            actual: format!("{} values", value_cache.len()),
        });
    }
    if cache_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || head_dim > 256
        || !q_heads.is_multiple_of(kv_heads)
        || cache_len > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "cached GQA dimensions",
            expected: "non-zero u32-sized dims, head_dim <= 256, and q_heads divisible by kv_heads"
                .to_string(),
            actual: format!(
                "cache_len={cache_len} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"
            ),
        });
    }
    Ok(query_len)
}

/// Runs causal grouped-query attention for row-major prefill queries.
///
/// `query` is `[tokens, q_heads, head_dim]`. K/V caches are row-major
/// `[max_tokens, kv_heads * head_dim]` and must already contain the rows
/// through `start_position + tokens`. Token `t` attends rows
/// `0..=start_position + t`. The output is row-major
/// `[tokens, q_heads * head_dim]`.
#[cfg(test)]
pub fn prefill_gqa_attention_f32(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    tokens: usize,
    start_position: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<DeviceBuffer<f32>> {
    let query_len = prefill_gqa_attention_len(
        query,
        key_cache,
        value_cache,
        tokens,
        start_position,
        q_heads,
        kv_heads,
        head_dim,
    )?;
    let mut output = DeviceBuffer::zeroed(query_len)?;
    prefill_gqa_attention_f32_into(
        query,
        key_cache,
        value_cache,
        output.output(),
        tokens,
        start_position,
        q_heads,
        kv_heads,
        head_dim,
    )?;
    Ok(output)
}

#[allow(missing_docs)]
pub fn prefill_gqa_attention_f32_into(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    start_position: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let query_len = prefill_gqa_attention_len(
        query,
        key_cache,
        value_cache,
        tokens,
        start_position,
        q_heads,
        kv_heads,
        head_dim,
    )?;
    if output.len() != query_len {
        return Err(Error::Shape {
            label: "prefill GQA output",
            expected: format!("{query_len} values"),
            actual: format!("{} values", output.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_prefill_gqa_attention_f32",
            ffi::infer_prefill_gqa_attention_f32(
                query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                start_position as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
            ),
        )
    }
}

/// Enqueues causal grouped-query prefill attention on `stream`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_gqa_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    start_position: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_len = prefill_gqa_attention_len(
        query,
        key_cache,
        value_cache,
        tokens,
        start_position,
        q_heads,
        kv_heads,
        head_dim,
    )?;
    if output.len() != query_len {
        return Err(Error::Shape {
            label: "prefill GQA output",
            expected: format!("{query_len} values"),
            actual: format!("{} values", output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_prefill_gqa_attention_f32_on_stream",
            ffi::infer_prefill_gqa_attention_f32_on_stream(
                query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                start_position as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Appends flattened ragged K/V rows into per-sequence cache pointer tables.
#[allow(clippy::too_many_arguments)]
pub fn append_ragged_kv_f32_into_on_stream(
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_cache_table: &DeviceBuffer<*mut f32>,
    value_cache_table: &DeviceBuffer<*mut f32>,
    cache_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    start_positions: &DeviceBuffer<u32>,
    sequence_count: usize,
    total_tokens: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = total_tokens.saturating_mul(width);
    if sequence_count == 0
        || total_tokens == 0
        || width == 0
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || width > u32::MAX as usize
        || key.len() != values
        || value.len() != values
        || cache_table_offset.saturating_add(sequence_count) > key_cache_table.len()
        || cache_table_offset.saturating_add(sequence_count) > value_cache_table.len()
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
        || start_positions.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "ragged KV append buffers",
            expected: format!("key/value={values}; cache and metadata tables >= {sequence_count}"),
            actual: format!(
                "key={} value={} key_cache={} value_cache={} offsets={} lengths={} starts={} sequences={sequence_count} tokens={total_tokens} width={width}",
                key.len(),
                value.len(),
                key_cache_table.len().saturating_sub(cache_table_offset),
                value_cache_table.len().saturating_sub(cache_table_offset),
                sequence_offsets.len(),
                sequence_lengths.len(),
                start_positions.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_append_ragged_kv_f32_on_stream",
            ffi::infer_append_ragged_kv_f32_on_stream(
                key.ptr,
                value.ptr,
                key_cache_table.ptr.add(cache_table_offset),
                value_cache_table.ptr.add(cache_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                start_positions.ptr,
                sequence_count as u32,
                total_tokens as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs causal GQA for flattened ragged rows over per-sequence cache tables.
#[allow(clippy::too_many_arguments)]
pub fn ragged_gqa_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_cache_table: &DeviceBuffer<*mut f32>,
    value_cache_table: &DeviceBuffer<*mut f32>,
    cache_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    start_positions: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    sequence_count: usize,
    total_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_width = q_heads.saturating_mul(head_dim);
    let values = total_tokens.saturating_mul(query_width);
    if sequence_count == 0
        || total_tokens == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > 256
        || query.len() != values
        || output.len() != values
        || cache_table_offset.saturating_add(sequence_count) > key_cache_table.len()
        || cache_table_offset.saturating_add(sequence_count) > value_cache_table.len()
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
        || start_positions.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "ragged GQA buffers",
            expected: format!(
                "query/output={values}; cache and metadata tables >= {sequence_count}; valid GQA heads"
            ),
            actual: format!(
                "query={} output={} key_cache={} value_cache={} offsets={} lengths={} starts={} sequences={sequence_count} tokens={total_tokens} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}",
                query.len(),
                output.len(),
                key_cache_table.len().saturating_sub(cache_table_offset),
                value_cache_table.len().saturating_sub(cache_table_offset),
                sequence_offsets.len(),
                sequence_lengths.len(),
                start_positions.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ragged_gqa_attention_f32_on_stream",
            ffi::infer_ragged_gqa_attention_f32_on_stream(
                query.ptr,
                key_cache_table.ptr.add(cache_table_offset),
                value_cache_table.ptr.add(cache_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                start_positions.ptr,
                output.buffer_mut().ptr,
                sequence_count as u32,
                total_tokens as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Appends flattened ragged K/V rows into physical F32 page pools.
#[allow(clippy::too_many_arguments)]
pub fn append_ragged_paged_kv_f32_into_on_stream(
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    key_pool: &mut DeviceBuffer<f32>,
    value_pool: &mut DeviceBuffer<f32>,
    page_tables: &DeviceBuffer<*const u32>,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    start_positions: &DeviceBuffer<u32>,
    sequence_count: usize,
    total_tokens: usize,
    page_tokens: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = total_tokens.saturating_mul(width);
    if sequence_count == 0
        || total_tokens == 0
        || page_tokens == 0
        || width == 0
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || page_tokens > u32::MAX as usize
        || width > u32::MAX as usize
        || key.len() != values
        || value.len() != values
        || key_pool.len() != value_pool.len()
        || !key_pool
            .len()
            .is_multiple_of(page_tokens.saturating_mul(width))
        || page_tables.len() < sequence_count
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
        || start_positions.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "ragged paged KV append buffers",
            expected: format!(
                "key/value={values}; aligned page pools and metadata >= {sequence_count}"
            ),
            actual: format!(
                "key={} value={} key_pool={} value_pool={} tables={} offsets={} lengths={} starts={} sequences={sequence_count} tokens={total_tokens} page_tokens={page_tokens} width={width}",
                key.len(),
                value.len(),
                key_pool.len(),
                value_pool.len(),
                page_tables.len(),
                sequence_offsets.len(),
                sequence_lengths.len(),
                start_positions.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_append_ragged_paged_kv_f32_on_stream",
            ffi::infer_append_ragged_paged_kv_f32_on_stream(
                key.ptr,
                value.ptr,
                key_pool.ptr,
                value_pool.ptr,
                page_tables.ptr,
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                start_positions.ptr,
                sequence_count as u32,
                total_tokens as u32,
                page_tokens as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs causal GQA for flattened ragged rows over physical F32 page pools.
#[allow(clippy::too_many_arguments)]
pub fn ragged_paged_gqa_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_pool: &DeviceBuffer<f32>,
    value_pool: &DeviceBuffer<f32>,
    page_tables: &DeviceBuffer<*const u32>,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    start_positions: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    sequence_count: usize,
    total_tokens: usize,
    page_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let query_width = q_heads.saturating_mul(head_dim);
    let values = total_tokens.saturating_mul(query_width);
    let kv_width = kv_heads.saturating_mul(head_dim);
    if sequence_count == 0
        || total_tokens == 0
        || page_tokens == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || page_tokens > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > 256
        || query.len() != values
        || output.len() != values
        || key_pool.len() != value_pool.len()
        || !key_pool
            .len()
            .is_multiple_of(page_tokens.saturating_mul(kv_width))
        || page_tables.len() < sequence_count
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
        || start_positions.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "ragged paged GQA buffers",
            expected: format!(
                "query/output={values}; aligned page pools and metadata >= {sequence_count}"
            ),
            actual: format!(
                "query={} output={} key_pool={} value_pool={} tables={} offsets={} lengths={} starts={} sequences={sequence_count} tokens={total_tokens} page_tokens={page_tokens} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}",
                query.len(),
                output.len(),
                key_pool.len(),
                value_pool.len(),
                page_tables.len(),
                sequence_offsets.len(),
                sequence_lengths.len(),
                start_positions.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ragged_paged_gqa_attention_f32_on_stream",
            ffi::infer_ragged_paged_gqa_attention_f32_on_stream(
                query.ptr,
                key_pool.ptr,
                value_pool.ptr,
                page_tables.ptr,
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                start_positions.ptr,
                output.buffer_mut().ptr,
                sequence_count as u32,
                total_tokens as u32,
                page_tokens as u32,
                q_heads as u32,
                kv_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn prefill_gqa_attention_len(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    tokens: usize,
    start_position: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<usize> {
    let query_len = tokens
        .checked_mul(q_heads)
        .and_then(|rows| rows.checked_mul(head_dim))
        .ok_or_else(|| Error::Shape {
            label: "prefill GQA query",
            expected: "tokens * q_heads * head_dim without overflow".to_string(),
            actual: format!("tokens={tokens} q_heads={q_heads} head_dim={head_dim}"),
        })?;
    let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "prefill GQA width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    let required_cache = start_position
        .checked_add(tokens)
        .and_then(|rows| rows.checked_mul(kv_width))
        .ok_or_else(|| Error::Shape {
            label: "prefill GQA cache",
            expected: "(start_position + tokens) * kv_width without overflow".to_string(),
            actual: format!("start_position={start_position} tokens={tokens} kv_width={kv_width}"),
        })?;
    if query.len() != query_len {
        return Err(Error::Shape {
            label: "prefill GQA query",
            expected: format!("{query_len} values"),
            actual: format!("{} values", query.len()),
        });
    }
    if key_cache.len() < required_cache || value_cache.len() < required_cache {
        return Err(Error::Shape {
            label: "prefill GQA cache",
            expected: format!("at least {required_cache} values"),
            actual: format!("key={} value={}", key_cache.len(), value_cache.len()),
        });
    }
    if tokens == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || head_dim > 256
        || !q_heads.is_multiple_of(kv_heads)
        || tokens > u32::MAX as usize
        || start_position > u32::MAX as usize
        || q_heads > u32::MAX as usize
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "prefill GQA dimensions",
            expected: "non-zero u32-sized dims, head_dim <= 256, and q_heads divisible by kv_heads"
                .to_string(),
            actual: format!(
                "tokens={tokens} start_position={start_position} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"
            ),
        });
    }

    Ok(query_len)
}

/// Result of a device-side argmax over BF16 linear logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArgmaxResult {
    /// Index of the largest logit.
    pub index: u32,
    /// Largest logit value.
    pub value: f32,
}

/// Computes `weight * input` for a BF16 row-major matrix and returns argmax.
///
/// `input` has `cols` f32 values. `weight` is row-major BF16 with shape
/// `[rows, cols]`, represented as raw BF16 `u16` values. Logits and the argmax
/// reduction stay on device; only the winning index and logit are copied back.
pub fn bf16_linear_argmax_f32(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
) -> Result<ArgmaxResult> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "BF16 linear argmax weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != cols {
        return Err(Error::Shape {
            label: "BF16 linear argmax input",
            expected: format!("{cols} values"),
            actual: format!("{} values", input.len()),
        });
    }
    if weight.len() != weight_len {
        return Err(Error::Shape {
            label: "BF16 linear argmax weight",
            expected: format!("{weight_len} values"),
            actual: format!("{} values", weight.len()),
        });
    }
    if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "BF16 linear argmax dimensions",
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }

    let stream = CudaStream::new_non_blocking()?;
    let mut logits = DeviceBuffer::<f32>::zeroed(rows)?;
    let mut out_index = DeviceBuffer::<u32>::zeroed(1)?;
    let mut out_value = DeviceBuffer::<f32>::zeroed(1)?;
    bf16_linear_argmax_f32_into_on_stream(
        input,
        weight,
        logits.output(),
        out_index.output(),
        out_value.output(),
        rows,
        cols,
        &stream,
    )?;
    Ok(ArgmaxResult {
        index: out_index.copy_to_host(&stream)?[0],
        value: out_value.copy_to_host(&stream)?[0],
    })
}

/// Enqueues BF16 matvec logits plus device argmax into caller-owned buffers on
/// `stream`.
pub fn bf16_linear_argmax_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    mut logits: DeviceOutput<'_, f32>,
    mut out_index: DeviceOutput<'_, u32>,
    mut out_value: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_bf16_linear(input, weight, rows, cols, "BF16 linear argmax")?;
    if logits.len() != rows || out_index.len() != 1 || out_value.len() != 1 {
        return Err(Error::Shape {
            label: "BF16 linear argmax outputs",
            expected: format!("logits={rows} out_index=1 out_value=1"),
            actual: format!(
                "logits={} out_index={} out_value={}",
                logits.len(),
                out_index.len(),
                out_value.len()
            ),
        });
    }

    unsafe {
        check_cuda(
            "infer_bf16_linear_argmax_f32_on_stream",
            ffi::infer_bf16_linear_argmax_f32_on_stream(
                input.ptr,
                weight.ptr,
                logits.buffer_mut().ptr,
                out_index.buffer_mut().ptr,
                out_value.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(missing_docs)]
pub fn argmax_f32_into_on_stream(
    values: &DeviceBuffer<f32>,
    mut out_index: DeviceOutput<'_, u32>,
    mut out_value: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if values.is_empty() || values.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "argmax f32 values",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{} values", values.len()),
        });
    }
    if out_index.len() != 1 || out_value.len() != 1 {
        return Err(Error::Shape {
            label: "argmax f32 outputs",
            expected: "out_index=1 out_value=1".to_string(),
            actual: format!(
                "out_index={} out_value={}",
                out_index.len(),
                out_value.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_argmax_f32_on_stream",
            ffi::infer_argmax_f32_on_stream(
                values.ptr,
                out_index.buffer_mut().ptr,
                out_value.buffer_mut().ptr,
                values.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one argmax reduction per row of a dense f32 matrix.
pub fn argmax_f32_batch_into_on_stream(
    values: &DeviceBuffer<f32>,
    mut out_index: DeviceOutput<'_, u32>,
    mut out_value: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    if rows == 0
        || cols == 0
        || values.len() != len
        || out_index.len() != rows
        || out_value.len() != rows
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched argmax f32 buffers",
            expected: format!("values={len} index/value={rows}"),
            actual: format!(
                "values={} index={} value={} rows={rows} cols={cols}",
                values.len(),
                out_index.len(),
                out_value.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_argmax_f32_batch_on_stream",
            ffi::infer_argmax_f32_batch_on_stream(
                values.ptr,
                out_index.buffer_mut().ptr,
                out_value.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Masks disallowed vocabulary logits in place with negative infinity.
pub fn mask_logits_f32_batch_in_place_on_stream(
    mut logits: DeviceInOut<'_, f32>,
    allowed: &DeviceBuffer<u32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "logit grammar mask",
        expected: "rows * vocabulary without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let mask_words = cols.div_ceil(32);
    let mask_values = rows.checked_mul(mask_words).ok_or_else(|| Error::Shape {
        label: "logit grammar mask",
        expected: "rows * mask words without overflow".to_string(),
        actual: format!("rows={rows} words={mask_words}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || mask_words > u32::MAX as usize
        || logits.len() < values
        || allowed.len() < mask_values
    {
        return Err(Error::Shape {
            label: "logit grammar mask",
            expected: format!("logits>={values} mask>={mask_values}"),
            actual: format!("logits={} mask={}", logits.len(), allowed.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_mask_logits_f32_batch_on_stream",
            ffi::infer_mask_logits_f32_batch_on_stream(
                logits.buffer_mut().ptr,
                allowed.ptr,
                rows as u32,
                cols as u32,
                mask_words as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Greedily accepts a contiguous speculative prefix for each sequence.
///
/// Draft tokens and verification-logit rows are sequence-major. The first
/// draft is checked against each sequence's previous logits; subsequent
/// drafts are checked against the preceding verification row. `next_tokens`
/// receives the target token at the first rejection, or the bonus token when
/// every draft is accepted.
#[allow(clippy::too_many_arguments)]
pub fn speculative_accept_argmax_f32_into_on_stream(
    previous_logits: &DeviceBuffer<*const f32>,
    verification_logits: &DeviceBuffer<f32>,
    drafted_tokens: &DeviceBuffer<u32>,
    mut accepted_counts: DeviceOutput<'_, u32>,
    mut next_tokens: DeviceOutput<'_, u32>,
    sequence_count: usize,
    draft_count: usize,
    vocab_size: usize,
    stream: &CudaStream,
) -> Result<()> {
    let rows = sequence_count.saturating_mul(draft_count);
    let logits_len = rows.saturating_mul(vocab_size);
    if sequence_count == 0
        || draft_count == 0
        || draft_count > 4
        || vocab_size == 0
        || sequence_count > u32::MAX as usize
        || draft_count > u32::MAX as usize
        || vocab_size > u32::MAX as usize
        || previous_logits.len() != sequence_count
        || verification_logits.len() != logits_len
        || drafted_tokens.len() != rows
        || accepted_counts.len() != sequence_count
        || next_tokens.len() != sequence_count
    {
        return Err(Error::Shape {
            label: "speculative argmax acceptance buffers",
            expected: format!(
                "previous/accepted/next={sequence_count} drafts={rows} verification_logits={logits_len} with 1..=4 drafts"
            ),
            actual: format!(
                "previous={} drafts={} verification_logits={} accepted={} next={} sequences={sequence_count} draft_count={draft_count} vocab={vocab_size}",
                previous_logits.len(),
                drafted_tokens.len(),
                verification_logits.len(),
                accepted_counts.len(),
                next_tokens.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_speculative_accept_argmax_f32_on_stream",
            ffi::infer_speculative_accept_argmax_f32_on_stream(
                previous_logits.ptr,
                verification_logits.ptr,
                drafted_tokens.ptr,
                accepted_counts.buffer_mut().ptr,
                next_tokens.buffer_mut().ptr,
                sequence_count as u32,
                draft_count as u32,
                vocab_size as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Largest top-k candidate set supported by the low-latency GPU sampler.
pub const GPU_SAMPLING_MAX_TOP_K: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceSamplingParams {
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    draw: f32,
    top_k: u32,
    token_counts: u64,
}

/// One compact token result produced by device-resident sampling.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSampledToken {
    /// Selected vocabulary ID.
    pub id: u32,
    /// Original model logit before penalties and temperature.
    pub logit: f32,
    /// Logit after presence and frequency penalties.
    pub adjusted_logit: f32,
    status: u32,
}

/// Per-row sampling inputs consumed by [`GpuTokenSampler`].
pub struct GpuSamplingRow<'a> {
    /// Softmax temperature. Zero selects the adjusted argmax.
    pub temperature: f32,
    /// Maximum candidates retained before nucleus sampling.
    pub top_k: usize,
    /// Cumulative probability retained by nucleus sampling.
    pub top_p: f32,
    /// One-time penalty for tokens already present.
    pub presence_penalty: f32,
    /// Per-occurrence token penalty.
    pub frequency_penalty: f32,
    /// Uniform random draw in `[0, 1)` for this row.
    pub draw: f32,
    /// Optional vocabulary-sized occurrence counts updated after sampling.
    pub token_counts: Option<&'a mut DeviceBuffer<u32>>,
}

/// Reusable device and host storage for batched top-k/top-p token sampling.
pub struct GpuTokenSampler {
    capacity: usize,
    vocab: usize,
    host_params: Vec<DeviceSamplingParams>,
    params: DeviceBuffer<DeviceSamplingParams>,
    stage_one_keys: DeviceBuffer<u64>,
    stage_two_keys: DeviceBuffer<u64>,
    top_keys: DeviceBuffer<u64>,
    results: DeviceBuffer<GpuSampledToken>,
}

impl GpuTokenSampler {
    /// Allocates sampling metadata and hierarchical top-k storage.
    pub fn new(capacity: usize, vocab: usize) -> Result<Self> {
        const ITEMS_PER_BLOCK: usize = 1024;
        const MAX_VOCAB: usize = ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;
        if capacity == 0 || vocab == 0 || vocab > MAX_VOCAB {
            return Err(Error::Shape {
                label: "GPU token sampler shape",
                expected: format!("capacity > 0 and vocab=1..={MAX_VOCAB}"),
                actual: format!("capacity={capacity} vocab={vocab}"),
            });
        }
        let stage_one_chunks = vocab.div_ceil(ITEMS_PER_BLOCK);
        let stage_one_count = stage_one_chunks * GPU_SAMPLING_MAX_TOP_K;
        let stage_two_chunks = stage_one_count.div_ceil(ITEMS_PER_BLOCK);
        let stage_two_count = stage_two_chunks * GPU_SAMPLING_MAX_TOP_K;
        let empty = DeviceSamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            draw: 0.0,
            top_k: 1,
            token_counts: 0,
        };
        Ok(Self {
            capacity,
            vocab,
            host_params: vec![empty; capacity],
            params: DeviceBuffer::zeroed(capacity)?,
            stage_one_keys: DeviceBuffer::zeroed(capacity * stage_one_count)?,
            stage_two_keys: DeviceBuffer::zeroed(capacity * stage_two_count)?,
            top_keys: DeviceBuffer::zeroed(capacity * GPU_SAMPLING_MAX_TOP_K)?,
            results: DeviceBuffer::zeroed(capacity)?,
        })
    }

    /// Returns the device bytes owned by reusable sampling storage.
    pub fn device_bytes(&self) -> usize {
        self.params.device_bytes()
            + self.stage_one_keys.device_bytes()
            + self.stage_two_keys.device_bytes()
            + self.top_keys.device_bytes()
            + self.results.device_bytes()
    }

    /// Samples one token per active logit row and copies only compact results.
    pub fn sample(
        &mut self,
        logits: &DeviceBuffer<f32>,
        rows: &mut [GpuSamplingRow<'_>],
        vocab: usize,
        stream: &CudaStream,
    ) -> Result<Vec<GpuSampledToken>> {
        if rows.is_empty()
            || rows.len() > self.capacity
            || self.capacity > u32::MAX as usize
            || vocab != self.vocab
            || logits.len() != self.capacity.saturating_mul(vocab)
            || vocab > u32::MAX as usize
        {
            return Err(Error::Shape {
                label: "GPU token sampling buffers",
                expected: format!(
                    "rows=1..={} logits={} vocab>0",
                    self.capacity,
                    self.capacity.saturating_mul(vocab)
                ),
                actual: format!("rows={} logits={} vocab={vocab}", rows.len(), logits.len()),
            });
        }
        self.host_params.fill(DeviceSamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            draw: 0.0,
            top_k: 1,
            token_counts: 0,
        });
        for (slot, row) in rows.iter_mut().enumerate() {
            let effective_top_k = if row.temperature == 0.0 { 1 } else { row.top_k };
            if !row.temperature.is_finite()
                || row.temperature < 0.0
                || !row.top_p.is_finite()
                || row.top_p <= 0.0
                || row.top_p > 1.0
                || !row.presence_penalty.is_finite()
                || !row.frequency_penalty.is_finite()
                || !row.draw.is_finite()
                || !(0.0..1.0).contains(&row.draw)
            {
                return Err(Error::Format {
                    label: "GPU token sampling parameters",
                    detail: format!("invalid parameters for row {slot}"),
                });
            }
            if effective_top_k == 0 || effective_top_k > GPU_SAMPLING_MAX_TOP_K {
                return Err(Error::Shape {
                    label: "GPU token sampling top-k",
                    expected: format!("1..={GPU_SAMPLING_MAX_TOP_K}"),
                    actual: effective_top_k.to_string(),
                });
            }
            let token_counts = match row.token_counts.as_deref_mut() {
                Some(counts) if counts.len() == vocab => counts.ptr as usize as u64,
                Some(counts) => {
                    return Err(Error::Shape {
                        label: "GPU sampling token counts",
                        expected: format!("{vocab} values"),
                        actual: format!("{} values", counts.len()),
                    });
                }
                None => 0,
            };
            self.host_params[slot] = DeviceSamplingParams {
                temperature: row.temperature,
                top_p: row.top_p,
                presence_penalty: row.presence_penalty,
                frequency_penalty: row.frequency_penalty,
                draw: row.draw,
                top_k: effective_top_k as u32,
                token_counts,
            };
        }
        self.params.copy_from_host(&self.host_params)?;
        unsafe {
            check_cuda(
                "infer_sample_topk_topp_f32_batch_on_stream",
                ffi::infer_sample_topk_topp_f32_batch_on_stream(
                    logits.ptr,
                    self.params.ptr.cast(),
                    self.stage_one_keys.ptr,
                    self.stage_two_keys.ptr,
                    self.top_keys.ptr,
                    self.results.ptr.cast(),
                    rows.len() as u32,
                    vocab as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        let results = self
            .results
            .copy_prefix_to_host(rows.len(), stream)?
            .into_vec();
        if let Some((row, result)) = results
            .iter()
            .enumerate()
            .find(|(_, result)| result.status != 0)
        {
            let detail = match result.status {
                1 => "no finite logits",
                2 => "invalid probability mass",
                _ => "unknown device sampling failure",
            };
            return Err(Error::Format {
                label: "GPU token sampling",
                detail: format!("row {row}: {detail}"),
            });
        }
        Ok(results)
    }
}

/// Enqueues the fused direct top-1 lm-head kernel on `stream`.
///
/// Computes argmax(`weight * input`) directly without materializing a full
/// `rows`-length logits vector to global memory. The caller supplies scratch
/// buffers sized to at least `rows` pairs (rounded up to a multiple of 8) plus
/// the final `(out_index, out_value)` slots.
///
/// - `input`: hidden vector, `cols` f32 values.
/// - `weight`: row-major BF16 `[rows, cols]`.
/// - `scratch_value` / `scratch_index`: caller-owned device buffers of length
///   `rows` rounded up to a multiple of 8.
/// - `out_index` / `out_value`: 1-element output device buffers.
/// - `rows = VOCAB`, `cols = HIDDEN`.
pub fn lm_head_top1_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    scratch_value: &DeviceBuffer<f32>,
    scratch_index: &DeviceBuffer<u32>,
    out_index: &DeviceBuffer<u32>,
    out_value: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_bf16_linear(input, weight, rows, cols, "lm-head top1")?;
    const WARPS_PER_BLOCK: usize = 8;
    let scratch_len = rows.div_ceil(WARPS_PER_BLOCK) * WARPS_PER_BLOCK;
    if scratch_value.len() < scratch_len || scratch_index.len() < scratch_len {
        return Err(Error::Shape {
            label: "lm-head top1 scratch",
            expected: format!("scratch_value/scratch_index length >= {scratch_len}"),
            actual: format!(
                "scratch_value={} scratch_index={}",
                scratch_value.len(),
                scratch_index.len()
            ),
        });
    }
    if out_index.len() != 1 || out_value.len() != 1 {
        return Err(Error::Shape {
            label: "lm-head top1 outputs",
            expected: "out_index=1 out_value=1".to_string(),
            actual: format!(
                "out_index={} out_value={}",
                out_index.len(),
                out_value.len()
            ),
        });
    }
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "lm-head top1 dimensions",
            expected: "rows and cols fit in u32".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }
    if !cols.is_multiple_of(4) {
        return Err(Error::Shape {
            label: "lm-head top1 cols alignment",
            expected: "cols divisible by 4 (vectorized bf16x2 loads)".to_string(),
            actual: format!("cols={cols}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_lm_head_top1_f32_on_stream",
            ffi::infer_lm_head_top1_f32_on_stream(
                input.ptr,
                weight.ptr,
                scratch_value.ptr,
                scratch_index.ptr,
                scratch_len as u32,
                out_index.ptr,
                out_value.ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues an exact batched BF16 lm-head projection and direct top-1
/// reduction on `stream`.
///
/// The projection reuses each weight row across up to four input rows and
/// reduces every eight vocabulary rows to one scratch candidate. This avoids
/// materializing `[batch_size, rows]` logits while preserving the accumulation
/// order of [`bf16_linear_logits_f32_batch_into_on_stream`].
#[allow(clippy::too_many_arguments)]
pub fn lm_head_top1_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    scratch_value: &DeviceBuffer<f32>,
    scratch_index: &DeviceBuffer<u32>,
    out_index: &DeviceBuffer<u32>,
    out_value: &DeviceBuffer<f32>,
    batch_size: usize,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    const WARPS_PER_BLOCK: usize = 8;
    let input_len = batch_size.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "batched lm-head top1 input",
        expected: "batch_size * cols without overflow".to_string(),
        actual: format!("batch_size={batch_size} cols={cols}"),
    })?;
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "batched lm-head top1 weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let scratch_len = batch_size
        .checked_mul(rows.div_ceil(WARPS_PER_BLOCK))
        .ok_or_else(|| Error::Shape {
            label: "batched lm-head top1 scratch",
            expected: "batch_size * ceil(rows / 8) without overflow".to_string(),
            actual: format!("batch_size={batch_size} rows={rows}"),
        })?;
    if batch_size == 0
        || rows == 0
        || cols == 0
        || batch_size > u32::MAX as usize
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || scratch_len > u32::MAX as usize
        || input.len() < input_len
        || weight.len() != weight_len
        || scratch_value.len() < scratch_len
        || scratch_index.len() < scratch_len
        || out_index.len() < batch_size
        || out_value.len() < batch_size
    {
        return Err(Error::Shape {
            label: "batched lm-head top1 buffers",
            expected: format!(
                "input>={input_len} weight={weight_len} scratch>={scratch_len} output>={batch_size}"
            ),
            actual: format!(
                "input={} weight={} scratch_value={} scratch_index={} out_index={} out_value={}",
                input.len(),
                weight.len(),
                scratch_value.len(),
                scratch_index.len(),
                out_index.len(),
                out_value.len()
            ),
        });
    }
    if !cols.is_multiple_of(4) {
        return Err(Error::Shape {
            label: "batched lm-head top1 cols alignment",
            expected: "cols divisible by 4 (vectorized bf16x2 loads)".to_string(),
            actual: format!("cols={cols}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_lm_head_top1_f32_batch_on_stream",
            ffi::infer_lm_head_top1_f32_batch_on_stream(
                input.ptr,
                weight.ptr,
                scratch_value.ptr,
                scratch_index.ptr,
                scratch_len as u32,
                out_index.ptr,
                out_value.ptr,
                batch_size as u32,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

///
/// `input` has `cols` f32 values. `weight` is row-major BF16 with shape
/// `[rows, cols]`. The returned device buffer contains `rows` f32 logits.
#[cfg(test)]
pub fn bf16_linear_logits_f32(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer<f32>> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "BF16 linear logits weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != cols {
        return Err(Error::Shape {
            label: "BF16 linear logits input",
            expected: format!("{cols} values"),
            actual: format!("{} values", input.len()),
        });
    }
    if weight.len() != weight_len {
        return Err(Error::Shape {
            label: "BF16 linear logits weight",
            expected: format!("{weight_len} values"),
            actual: format!("{} values", weight.len()),
        });
    }
    if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "BF16 linear logits dimensions",
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }

    let logits = DeviceBuffer::<f32>::zeroed(rows)?;
    unsafe {
        check_cuda(
            "infer_bf16_linear_logits_f32",
            ffi::infer_bf16_linear_logits_f32(
                input.ptr,
                weight.ptr,
                logits.ptr,
                rows as u32,
                cols as u32,
            ),
        )?;
    }
    Ok(logits)
}

#[allow(missing_docs)]
pub fn bf16_linear_logits_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    mut logits: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_bf16_linear(input, weight, rows, cols, "BF16 linear logits")?;
    if logits.len() != rows {
        return Err(Error::Shape {
            label: "BF16 linear logits output",
            expected: format!("{rows} values"),
            actual: format!("{} values", logits.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_bf16_linear_logits_f32_on_stream",
            ffi::infer_bf16_linear_logits_f32_on_stream(
                input.ptr,
                weight.ptr,
                logits.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues a BF16-weight projection for every row in a decode batch.
pub fn bf16_linear_logits_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    mut logits: DeviceOutput<'_, f32>,
    batch_size: usize,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = batch_size.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "batched BF16 linear input",
        expected: "batch_size * cols without overflow".to_string(),
        actual: format!("batch_size={batch_size} cols={cols}"),
    })?;
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "batched BF16 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let output_len = batch_size.checked_mul(rows).ok_or_else(|| Error::Shape {
        label: "batched BF16 linear output",
        expected: "batch_size * rows without overflow".to_string(),
        actual: format!("batch_size={batch_size} rows={rows}"),
    })?;
    if batch_size == 0
        || rows == 0
        || cols == 0
        || batch_size > u32::MAX as usize
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() < input_len
        || weight.len() != weight_len
        || logits.len() < output_len
    {
        return Err(Error::Shape {
            label: "batched BF16 linear buffers",
            expected: format!("input={input_len} weight={weight_len} output={output_len}"),
            actual: format!(
                "input={} weight={} output={}",
                input.len(),
                weight.len(),
                logits.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_bf16_linear_logits_f32_batch_on_stream",
            ffi::infer_bf16_linear_logits_f32_batch_on_stream(
                input.ptr,
                weight.ptr,
                logits.buffer_mut().ptr,
                batch_size as u32,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues two BF16-weight projections over the same f32 input as one CUDA grid.
#[allow(clippy::too_many_arguments)]
pub fn bf16_linear_pair_logits_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    first_weight: &DeviceBuffer<u16>,
    second_weight: &DeviceBuffer<u16>,
    mut first_logits: DeviceOutput<'_, f32>,
    mut second_logits: DeviceOutput<'_, f32>,
    first_rows: usize,
    second_rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_bf16_linear(
        input,
        first_weight,
        first_rows,
        cols,
        "first BF16 linear pair",
    )?;
    validate_bf16_linear(
        input,
        second_weight,
        second_rows,
        cols,
        "second BF16 linear pair",
    )?;
    if first_logits.len() != first_rows || second_logits.len() != second_rows {
        return Err(Error::Shape {
            label: "BF16 linear pair outputs",
            expected: format!("first={first_rows} second={second_rows}"),
            actual: format!(
                "first={} second={}",
                first_logits.len(),
                second_logits.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_bf16_linear_pair_logits_f32_on_stream",
            ffi::infer_bf16_linear_pair_logits_f32_on_stream(
                input.ptr,
                first_weight.ptr,
                second_weight.ptr,
                first_logits.buffer_mut().ptr,
                second_logits.buffer_mut().ptr,
                first_rows as u32,
                second_rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_bf16_linear(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
    label: &'static str,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label,
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if input.len() != cols {
        return Err(Error::Shape {
            label,
            expected: format!("input={cols} values"),
            actual: format!("input={} values", input.len()),
        });
    }
    if weight.len() != weight_len {
        return Err(Error::Shape {
            label,
            expected: format!("weight={weight_len} values"),
            actual: format!("weight={} values", weight.len()),
        });
    }
    if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label,
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }
    Ok(())
}

/// Converts a device-resident BF16 matrix to a device-resident f32 buffer.
#[cfg(test)]
pub fn bf16_matrix_to_f32(matrix: &Bf16Matrix) -> Result<DeviceBuffer<f32>> {
    let len = matrix
        .rows
        .checked_mul(matrix.cols)
        .ok_or_else(|| Error::Shape {
            label: "BF16 to f32 matrix",
            expected: "rows * cols without overflow".to_string(),
            actual: format!("rows={} cols={}", matrix.rows, matrix.cols),
        })?;
    if len == 0 || len > u32::MAX as usize {
        return Err(Error::Shape {
            label: "BF16 to f32 dimensions",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{len} values"),
        });
    }

    let mut output = DeviceBuffer::<f32>::zeroed(len)?;
    let stream = CudaStream::new_non_blocking()?;
    bf16_matrix_to_f32_into_on_stream(matrix, output.output(), &stream)?;
    stream.synchronize()?;
    Ok(output)
}

/// Enqueues BF16-to-F32 conversion into an existing output buffer on `stream`.
pub fn bf16_matrix_to_f32_into_on_stream(
    matrix: &Bf16Matrix,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let len = matrix
        .rows
        .checked_mul(matrix.cols)
        .ok_or_else(|| Error::Shape {
            label: "BF16 to f32 matrix",
            expected: "rows * cols without overflow".to_string(),
            actual: format!("rows={} cols={}", matrix.rows, matrix.cols),
        })?;
    if len == 0 || len > u32::MAX as usize {
        return Err(Error::Shape {
            label: "BF16 to f32 dimensions",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{len} values"),
        });
    }
    if output.len() != len {
        return Err(Error::Shape {
            label: "BF16 to f32 output",
            expected: format!("{len} values"),
            actual: format!("{} values", output.len()),
        });
    }

    let matrix = matrix.input();
    unsafe {
        check_cuda(
            "infer_bf16_to_f32_on_stream",
            ffi::infer_bf16_to_f32_on_stream(
                matrix.data_ptr().cast_mut(),
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Converts a prefix of device-resident BF16 values to f32 on `stream`.
pub fn bf16_to_f32_prefix_into_on_stream(
    input: &DeviceBuffer<u16>,
    mut output: DeviceOutput<'_, f32>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0 || len > u32::MAX as usize || input.len() < len || output.len() < len {
        return Err(Error::Shape {
            label: "BF16 to f32 prefix buffers",
            expected: format!("input and output >= {len} values for non-zero u32-sized len"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_bf16_to_f32_on_stream",
            ffi::infer_bf16_to_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Converts device-resident f32 values to BF16 storage on `stream`.
pub fn f32_to_bf16_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, u16>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty() || input.len() > u32::MAX as usize || output.len() < input.len() {
        return Err(Error::Shape {
            label: "F32 to BF16 buffers",
            expected: format!(
                "input in 1..=u32::MAX values and output >= input ({})",
                input.len()
            ),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_f32_to_bf16_on_stream",
            ffi::infer_f32_to_bf16_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Converts a prefix of device-resident f32 values to BF16 storage on `stream`.
pub fn f32_to_bf16_prefix_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, u16>,
    len: usize,
    stream: &CudaStream,
) -> Result<()> {
    if len == 0 || len > u32::MAX as usize || input.len() < len || output.len() < len {
        return Err(Error::Shape {
            label: "F32 to BF16 prefix buffers",
            expected: format!("input and output >= {len} values for non-zero u32-sized len"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_f32_to_bf16_on_stream",
            ffi::infer_f32_to_bf16_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn checked_attention_product(label: &'static str, factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |total, factor| {
        total.checked_mul(*factor).ok_or_else(|| Error::Shape {
            label,
            expected: "dimension product without overflow".to_string(),
            actual: format!("factors={factors:?}"),
        })
    })
}

/// Packs `[tokens, heads, head_dim]` f32 rows as BF16 `[heads, tokens, head_dim]`.
pub fn pack_token_heads_bf16_into_on_stream(
    input: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, u16>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    pack_token_heads_bf16_at_offset_into_on_stream(
        input, output, tokens, heads, head_dim, 0, stream,
    )
}

/// Packs f32 token/head rows beginning at `input_row_offset` into head-major BF16.
#[allow(clippy::too_many_arguments)]
pub fn pack_token_heads_bf16_at_offset_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, u16>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    input_row_offset: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = checked_attention_product("packed token heads", &[tokens, heads, head_dim])?;
    let input_end = input_row_offset
        .checked_add(tokens)
        .and_then(|rows| rows.checked_mul(heads))
        .and_then(|values| values.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    if len == 0
        || len > u32::MAX as usize
        || input_row_offset > u32::MAX as usize
        || input.len() < input_end
        || output.len() < len
    {
        return Err(Error::Shape {
            label: "packed token heads",
            expected: format!("input and output >= {len} values"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_pack_token_heads_bf16_on_stream",
            ffi::infer_pack_token_heads_bf16_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                heads as u32,
                head_dim as u32,
                input_row_offset as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Packs `[tokens, heads, head_dim]` f32 values as BF16 `[heads, head_dim, tokens]`.
pub fn pack_value_heads_bf16_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, u16>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = checked_attention_product("packed value heads", &[tokens, heads, head_dim])?;
    if len == 0
        || len > u32::MAX as usize
        || input.len() < len
        || output.len() < len
        || tokens > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "packed value heads",
            expected: format!("input and output >= {len} values"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_pack_value_heads_bf16_on_stream",
            ffi::infer_pack_value_heads_bf16_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies causal/windowed softmax to `[heads, queries, keys]` f32 score rows.
#[allow(clippy::too_many_arguments)]
pub fn causal_window_softmax_f32_in_place_on_stream(
    mut scores: DeviceInOut<'_, f32>,
    query_tokens: usize,
    key_tokens: usize,
    start_position: usize,
    heads: usize,
    head_dim: usize,
    window_tokens: Option<usize>,
    stream: &CudaStream,
) -> Result<()> {
    let len = checked_attention_product("causal score rows", &[heads, query_tokens, key_tokens])?;
    if len == 0
        || len > scores.len()
        || query_tokens > u32::MAX as usize
        || key_tokens > u32::MAX as usize
        || start_position > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || window_tokens.is_some_and(|window| window == 0 || window > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "causal score rows",
            expected: format!("{len} writable values and u32-sized non-zero dimensions"),
            actual: format!(
                "scores={} queries={query_tokens} keys={key_tokens}",
                scores.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_causal_window_softmax_f32_on_stream",
            ffi::infer_causal_window_softmax_f32_on_stream(
                scores.buffer_mut().ptr,
                query_tokens as u32,
                key_tokens as u32,
                start_position as u32,
                heads as u32,
                head_dim as u32,
                window_tokens.unwrap_or(0) as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies causal/windowed softmax and writes BF16 `[heads, queries, keys]` rows.
#[allow(clippy::too_many_arguments)]
pub fn causal_window_softmax_f32_to_bf16_on_stream(
    scores: &DeviceBuffer<f32>,
    mut probabilities: DeviceOutput<'_, u16>,
    query_tokens: usize,
    key_tokens: usize,
    start_position: usize,
    heads: usize,
    head_dim: usize,
    window_tokens: Option<usize>,
    stream: &CudaStream,
) -> Result<()> {
    let len = checked_attention_product("causal score rows", &[heads, query_tokens, key_tokens])?;
    if len == 0
        || len > scores.len()
        || len > probabilities.len()
        || query_tokens > u32::MAX as usize
        || key_tokens > u32::MAX as usize
        || start_position > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || window_tokens.is_some_and(|window| window == 0 || window > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "causal BF16 probability rows",
            expected: format!("{len} input and output values with u32-sized dimensions"),
            actual: format!(
                "scores={} probabilities={} queries={query_tokens} keys={key_tokens}",
                scores.len(),
                probabilities.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_causal_window_softmax_f32_to_bf16_on_stream",
            ffi::infer_causal_window_softmax_f32_to_bf16_on_stream(
                scores.ptr,
                probabilities.buffer_mut().ptr,
                query_tokens as u32,
                key_tokens as u32,
                start_position as u32,
                heads as u32,
                head_dim as u32,
                window_tokens.unwrap_or(0) as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Unpacks `[heads, tokens, head_dim]` f32 values to `[tokens, heads, head_dim]`.
pub fn unpack_heads_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    unpack_heads_f32_at_offset_into_on_stream(input, output, tokens, heads, head_dim, 0, stream)
}

/// Unpacks head-major f32 values into token rows beginning at `output_row_offset`.
#[allow(clippy::too_many_arguments)]
pub fn unpack_heads_f32_at_offset_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    output_row_offset: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = checked_attention_product("unpacked token heads", &[tokens, heads, head_dim])?;
    let output_end = output_row_offset
        .checked_add(tokens)
        .and_then(|rows| rows.checked_mul(heads))
        .and_then(|values| values.checked_mul(head_dim))
        .unwrap_or(usize::MAX);
    if len == 0
        || len > u32::MAX as usize
        || output_row_offset > u32::MAX as usize
        || input.len() < len
        || output.len() < output_end
    {
        return Err(Error::Shape {
            label: "unpacked token heads",
            expected: format!("input and output >= {len} values"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_unpack_heads_f32_on_stream",
            ffi::infer_unpack_heads_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                heads as u32,
                head_dim as u32,
                output_row_offset as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Transposes head-major attention output directly into a column-major NVFP4 activation.
#[allow(clippy::too_many_arguments)]
pub fn unpack_heads_quantize_nvfp4_col_major_f32_at_offset_into_on_stream(
    input: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    output_row_offset: usize,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_unpack_heads_quantize_nvfp4(
        input.len(),
        output,
        tokens,
        heads,
        head_dim,
        output_row_offset,
        input_scale,
    )?;
    let mut output = output.output();
    unsafe {
        check_cuda(
            "infer_unpack_heads_quantize_nvfp4_col_major_f32_on_stream",
            ffi::infer_unpack_heads_quantize_nvfp4_col_major_f32_on_stream(
                input.ptr,
                output.values_mut_ptr().cast(),
                output.scales_mut_ptr().cast(),
                tokens as u32,
                heads as u32,
                head_dim as u32,
                output_row_offset as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Transposes BF16 head-major attention output directly into a column-major NVFP4 activation.
#[allow(clippy::too_many_arguments)]
pub fn unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
    input: &DeviceBuffer<u16>,
    output: &mut Nvfp4Matrix,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    output_row_offset: usize,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_unpack_heads_quantize_nvfp4(
        input.len(),
        output,
        tokens,
        heads,
        head_dim,
        output_row_offset,
        input_scale,
    )?;
    let mut output = output.output();
    unsafe {
        check_cuda(
            "infer_unpack_heads_quantize_nvfp4_col_major_bf16_on_stream",
            ffi::infer_unpack_heads_quantize_nvfp4_col_major_bf16_on_stream(
                input.ptr,
                output.values_mut_ptr().cast(),
                output.scales_mut_ptr().cast(),
                tokens as u32,
                heads as u32,
                head_dim as u32,
                output_row_offset as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_unpack_heads_quantize_nvfp4(
    input_len: usize,
    output: &Nvfp4Matrix,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    output_row_offset: usize,
    input_scale: f32,
) -> Result<()> {
    let features = heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "head-major NVFP4 output width",
        expected: "heads * head_dim without overflow".to_string(),
        actual: format!("heads={heads} head_dim={head_dim}"),
    })?;
    let required_input_len = tokens.saturating_mul(features);
    let output_rows = output_row_offset.saturating_add(tokens);
    if tokens == 0
        || features == 0
        || input_len < required_input_len
        || output.rows != features
        || output.cols < output_rows
        || tokens > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || output_row_offset > u32::MAX as usize
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "head-major NVFP4 output",
            expected: format!(
                "input >= {required_input_len}, output={features}x>={output_rows}, and valid dimensions"
            ),
            actual: format!(
                "input={} output={}x{} tokens={tokens} heads={heads} head_dim={head_dim} input_scale={input_scale}",
                input_len, output.rows, output.cols
            ),
        });
    }
    Ok(())
}

/// Rounds a device-resident f32 buffer in place to BF16 precision, stored as f32.
pub fn round_f32_to_bf16_in_place_on_stream(
    values: DeviceInOut<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let count = values.len();
    round_f32_to_bf16_prefix_in_place_on_stream(values, count, stream)
}

/// Rounds an active prefix of a device-resident f32 buffer to BF16 precision.
pub fn round_f32_to_bf16_prefix_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    count: usize,
    stream: &CudaStream,
) -> Result<()> {
    if count == 0 || count > u32::MAX as usize || values.len() < count {
        return Err(Error::Shape {
            label: "F32 to BF16 round length",
            expected: format!("at least {count} values with 1..=u32::MAX active"),
            actual: format!("{} values with {count} active", values.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_round_f32_to_bf16_in_place_on_stream",
            ffi::infer_round_f32_to_bf16_in_place_on_stream(
                values.buffer_mut().ptr,
                count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Rounds a device-resident f32 buffer to BF16 precision and writes f32 output.
pub fn round_f32_to_bf16_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty() || input.len() > u32::MAX as usize || output.len() != input.len() {
        return Err(Error::Shape {
            label: "F32 to BF16 round buffers",
            expected: format!("input=output in 1..=u32::MAX values ({})", input.len()),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_round_f32_to_bf16_on_stream",
            ffi::infer_round_f32_to_bf16_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one-token Gated Delta Net decode for `S_v=128`.
///
/// `q`, `k`, `v`, and `output` are laid out as `heads` contiguous rows of 128
/// f32 values. `gate` and `beta` contain one value per head, where `gate` is
/// already in log-decay form and the kernel applies `exp(gate)`. `state` is
/// updated in place and uses transposed per-head storage:
/// `state[head][col][row]`, with `128 * 128` values per head.
pub fn gated_delta_net_128_f32_into_on_stream(
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<f32>,
    v: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let vector_len = heads.checked_mul(128).ok_or_else(|| Error::Shape {
        label: "Gated Delta Net vectors",
        expected: "heads * 128 without overflow".to_string(),
        actual: format!("heads={heads}"),
    })?;
    let state_len = heads.checked_mul(128 * 128).ok_or_else(|| Error::Shape {
        label: "Gated Delta Net state",
        expected: "heads * 128 * 128 without overflow".to_string(),
        actual: format!("heads={heads}"),
    })?;
    if heads == 0 || heads > u32::MAX as usize {
        return Err(Error::Shape {
            label: "Gated Delta Net heads",
            expected: "1..=u32::MAX heads".to_string(),
            actual: heads.to_string(),
        });
    }
    if q.len() != vector_len
        || k.len() != vector_len
        || v.len() != vector_len
        || output.len() != vector_len
        || gate.len() != heads
        || beta.len() != heads
        || state.len() != state_len
    {
        return Err(Error::Shape {
            label: "Gated Delta Net buffers",
            expected: format!("q/k/v/output={vector_len} gate/beta={heads} state={state_len}"),
            actual: format!(
                "q={} k={} v={} output={} gate={} beta={} state={}",
                q.len(),
                k.len(),
                v.len(),
                output.len(),
                gate.len(),
                beta.len(),
                state.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gated_delta_net_128_f32_on_stream",
            ffi::infer_gated_delta_net_128_f32_on_stream(
                q.ptr,
                k.ptr,
                v.ptr,
                gate.ptr,
                beta.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues a ragged-sequence batch of one-token Gated Delta Net updates.
///
/// Dense inputs and outputs are row-major by sequence. `state_table` contains
/// one persistent recurrent-state pointer per sequence, allowing batch
/// membership to change without moving the state itself. `state_table_offset`
/// selects the first row from a larger table.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_net_128_f32_batch_into_on_stream(
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<f32>,
    v: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    state_table: &DeviceBuffer<*mut f32>,
    mut output: DeviceOutput<'_, f32>,
    state_table_offset: usize,
    batch_size: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let vectors = batch_size
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(128))
        .ok_or_else(|| Error::Shape {
            label: "batched Gated Delta Net vectors",
            expected: "batch_size * heads * 128 without overflow".to_string(),
            actual: format!("batch_size={batch_size} heads={heads}"),
        })?;
    let scalars = batch_size.checked_mul(heads).ok_or_else(|| Error::Shape {
        label: "batched Gated Delta Net scalars",
        expected: "batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    let state_table_end =
        state_table_offset
            .checked_add(batch_size)
            .ok_or_else(|| Error::Shape {
                label: "batched Gated Delta Net state table",
                expected: "state_table_offset + batch_size without overflow".to_string(),
                actual: format!("state_table_offset={state_table_offset} batch_size={batch_size}"),
            })?;
    if batch_size == 0
        || heads == 0
        || batch_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || q.len() < vectors
        || k.len() < vectors
        || v.len() < vectors
        || output.len() < vectors
        || gate.len() < scalars
        || beta.len() < scalars
        || state_table_end > state_table.len()
    {
        return Err(Error::Shape {
            label: "batched Gated Delta Net buffers",
            expected: format!(
                "q/k/v/output={vectors} gate/beta={scalars} state_table>={state_table_end}"
            ),
            actual: format!(
                "q={} k={} v={} output={} gate={} beta={} state_table={}",
                q.len(),
                k.len(),
                v.len(),
                output.len(),
                gate.len(),
                beta.len(),
                state_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gated_delta_net_128_f32_batch_on_stream",
            ffi::infer_gated_delta_net_128_f32_batch_on_stream(
                q.ptr,
                k.ptr,
                v.ptr,
                gate.ptr,
                beta.ptr,
                state_table.ptr.add(state_table_offset),
                output.buffer_mut().ptr,
                batch_size as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues token-ordered Gated Delta Net updates for ragged prompt chunks.
///
/// Dense rows are flattened by sequence. `sequence_offsets` and
/// `sequence_lengths` describe each contiguous span, while `state_table`
/// contains one recurrent-state pointer per sequence.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_net_128_f32_chunks_into_on_stream(
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<f32>,
    v: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    state_table: &DeviceBuffer<*mut f32>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    sequence_count: usize,
    total_tokens: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let vectors = total_tokens
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(128))
        .ok_or_else(|| Error::Shape {
            label: "chunked Gated Delta Net vectors",
            expected: "total_tokens * heads * 128 without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} heads={heads}"),
        })?;
    let scalars = total_tokens
        .checked_mul(heads)
        .ok_or_else(|| Error::Shape {
            label: "chunked Gated Delta Net scalars",
            expected: "total_tokens * heads without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} heads={heads}"),
        })?;
    if sequence_count == 0
        || total_tokens == 0
        || heads == 0
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || heads > u32::MAX as usize
        || q.len() < vectors
        || k.len() < vectors
        || v.len() < vectors
        || output.len() < vectors
        || gate.len() < scalars
        || beta.len() < scalars
        || state_table_offset
            .checked_add(sequence_count)
            .is_none_or(|end| end > state_table.len())
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "chunked Gated Delta Net buffers",
            expected: format!(
                "q/k/v/output>={vectors} gate/beta>={scalars} metadata/state>={sequence_count}"
            ),
            actual: format!(
                "q={} k={} v={} output={} gate={} beta={} state={} offsets={} lengths={}",
                q.len(),
                k.len(),
                v.len(),
                output.len(),
                gate.len(),
                beta.len(),
                state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gated_delta_net_128_f32_chunks_on_stream",
            ffi::infer_gated_delta_net_128_f32_chunks_on_stream(
                q.ptr,
                k.ptr,
                v.ptr,
                gate.ptr,
                beta.ptr,
                state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                output.buffer_mut().ptr,
                sequence_count as u32,
                total_tokens as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Gathers equally sized f32 rows from a device pointer table.
pub fn gather_f32_pointer_rows_into_on_stream(
    input_table: &DeviceBuffer<*mut f32>,
    table_offset: usize,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    row_values: usize,
    stream: &CudaStream,
) -> Result<()> {
    gather_f32_pointer_rows_range_into_on_stream(
        input_table,
        table_offset,
        output.buffer_mut(),
        0,
        rows,
        row_values,
        stream,
    )
}

/// Gathers equally sized f32 rows into a contiguous output range.
#[allow(clippy::too_many_arguments)]
pub fn gather_f32_pointer_rows_range_into_on_stream(
    input_table: &DeviceBuffer<*mut f32>,
    table_offset: usize,
    output: &mut DeviceBuffer<f32>,
    output_offset: usize,
    rows: usize,
    row_values: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = rows.checked_mul(row_values).ok_or_else(|| Error::Shape {
        label: "gathered pointer rows",
        expected: "rows * row_values without overflow".to_string(),
        actual: format!("rows={rows} row_values={row_values}"),
    })?;
    if rows == 0
        || rows > u32::MAX as usize
        || row_values == 0
        || row_values > u32::MAX as usize
        || table_offset
            .checked_add(rows)
            .is_none_or(|end| end > input_table.len())
        || output_offset
            .checked_add(values)
            .is_none_or(|end| end > output.len())
    {
        return Err(Error::Shape {
            label: "gathered pointer row buffers",
            expected: format!(
                "table>={} output range {}..{}",
                table_offset.saturating_add(rows),
                output_offset,
                output_offset.saturating_add(values)
            ),
            actual: format!("table={} output={}", input_table.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_gather_f32_pointer_rows_on_stream",
            ffi::infer_gather_f32_pointer_rows_on_stream(
                input_table.ptr.add(table_offset),
                output.ptr.add(output_offset),
                rows as u32,
                row_values as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Scatters equally sized f32 rows through a device pointer table.
pub fn scatter_f32_pointer_rows_on_stream(
    input: &DeviceBuffer<f32>,
    output_table: &DeviceBuffer<*mut f32>,
    table_offset: usize,
    rows: usize,
    row_values: usize,
    stream: &CudaStream,
) -> Result<()> {
    scatter_f32_pointer_rows_range_on_stream(
        input,
        0,
        output_table,
        table_offset,
        rows,
        row_values,
        stream,
    )
}

/// Scatters equally sized f32 rows from a contiguous input range.
#[allow(clippy::too_many_arguments)]
pub fn scatter_f32_pointer_rows_range_on_stream(
    input: &DeviceBuffer<f32>,
    input_offset: usize,
    output_table: &DeviceBuffer<*mut f32>,
    table_offset: usize,
    rows: usize,
    row_values: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = rows.checked_mul(row_values).ok_or_else(|| Error::Shape {
        label: "scattered pointer rows",
        expected: "rows * row_values without overflow".to_string(),
        actual: format!("rows={rows} row_values={row_values}"),
    })?;
    if rows == 0
        || rows > u32::MAX as usize
        || row_values == 0
        || row_values > u32::MAX as usize
        || input_offset
            .checked_add(values)
            .is_none_or(|end| end > input.len())
        || table_offset
            .checked_add(rows)
            .is_none_or(|end| end > output_table.len())
    {
        return Err(Error::Shape {
            label: "scattered pointer row buffers",
            expected: format!(
                "input range {}..{} table>={}",
                input_offset,
                input_offset.saturating_add(values),
                table_offset.saturating_add(rows)
            ),
            actual: format!("input={} table={}", input.len(), output_table.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_scatter_f32_pointer_rows_on_stream",
            ffi::infer_scatter_f32_pointer_rows_on_stream(
                input.ptr.add(input_offset),
                output_table.ptr.add(table_offset),
                rows as u32,
                row_values as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues f32-input, FP8 E4M3-weight linear projection.
///
/// `weight` is row-major `[rows, cols]` E4M3 bytes. The kernel dequantizes
/// weights to f32, accumulates in f32, and applies `weight_scale` to the final
/// row sum. This is the first Qwen3.6 FP8 projection path; it deliberately
/// avoids runtime activation FP8 quantization until the surrounding layer path
/// is correct.
pub fn fp8_linear_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    fp8_linear_configured_f32_into_on_stream(
        input,
        weight,
        output,
        rows,
        cols,
        weight_scale,
        256,
        stream,
    )
}

/// Enqueues scalar-scaled W8A16 FP8 projections for a dense f32 batch.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    batch_size: usize,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = batch_size.saturating_mul(cols);
    let weight_len = rows.saturating_mul(cols);
    let output_len = batch_size.saturating_mul(rows);
    if batch_size == 0
        || rows == 0
        || cols == 0
        || input.len() < input_len
        || weight.len() != weight_len
        || output.len() < output_len
        || batch_size > u32::MAX as usize
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !(64..=512).contains(&threads)
        || !threads.is_multiple_of(32)
    {
        return Err(Error::Shape {
            label: "batched FP8 W8A16 linear buffers",
            expected: format!("input={input_len} weight={weight_len} output={output_len}"),
            actual: format!(
                "input={} weight={} output={} batch={batch_size} rows={rows} cols={cols} threads={threads}",
                input.len(),
                weight.len(),
                output.len()
            ),
        });
    }
    if !weight_scale.is_finite() {
        return Err(Error::Format {
            label: "batched FP8 W8A16 weight scale",
            detail: format!("expected finite scale, got {weight_scale}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_f32_batch_on_stream",
            ffi::infer_fp8_linear_f32_batch_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                batch_size as u32,
                rows as u32,
                cols as u32,
                weight_scale,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues the FP8 projection with a selected CUDA block size.
///
/// This configured entry point exists for shape-exact schedule measurements.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_configured_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "FP8 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !(64..=512).contains(&threads)
        || !threads.is_multiple_of(32)
    {
        return Err(Error::Shape {
            label: "FP8 linear dimensions",
            expected: "non-zero u32-sized rows and cols; threads a multiple of 32 in 64..=512"
                .to_string(),
            actual: format!("rows={rows} cols={cols} threads={threads}"),
        });
    }
    if !weight_scale.is_finite() {
        return Err(Error::Format {
            label: "FP8 linear weight_scale",
            detail: format!("expected finite scale, got {weight_scale}"),
        });
    }
    if input.len() != cols || weight.len() != weight_len || output.len() != rows {
        return Err(Error::Shape {
            label: "FP8 linear buffers",
            expected: format!("input={cols} weight={weight_len} output={rows}"),
            actual: format!(
                "input={} weight={} output={}",
                input.len(),
                weight.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_f32_configured_on_stream",
            ffi::infer_fp8_linear_f32_configured_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                weight_scale,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues two f32-input, FP8-weight projections as one segmented CUDA grid.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_pair_configured_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    first_weight: &DeviceBuffer<u8>,
    second_weight: &DeviceBuffer<u8>,
    mut first_output: DeviceOutput<'_, f32>,
    mut second_output: DeviceOutput<'_, f32>,
    first_rows: usize,
    second_rows: usize,
    cols: usize,
    first_scale: f32,
    second_scale: f32,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_segmented_fp8_linear(
        input,
        &[
            (first_weight, first_output.len(), first_rows, first_scale),
            (
                second_weight,
                second_output.len(),
                second_rows,
                second_scale,
            ),
        ],
        cols,
        threads,
    )?;
    unsafe {
        check_cuda(
            "infer_fp8_linear_pair_f32_configured_on_stream",
            ffi::infer_fp8_linear_pair_f32_configured_on_stream(
                input.ptr,
                first_weight.ptr,
                second_weight.ptr,
                first_output.buffer_mut().ptr,
                second_output.buffer_mut().ptr,
                first_rows as u32,
                second_rows as u32,
                cols as u32,
                first_scale,
                second_scale,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues three f32-input, FP8-weight projections as one segmented CUDA grid.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_triple_configured_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    first_weight: &DeviceBuffer<u8>,
    second_weight: &DeviceBuffer<u8>,
    third_weight: &DeviceBuffer<u8>,
    mut first_output: DeviceOutput<'_, f32>,
    mut second_output: DeviceOutput<'_, f32>,
    mut third_output: DeviceOutput<'_, f32>,
    first_rows: usize,
    second_rows: usize,
    third_rows: usize,
    cols: usize,
    first_scale: f32,
    second_scale: f32,
    third_scale: f32,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_segmented_fp8_linear(
        input,
        &[
            (first_weight, first_output.len(), first_rows, first_scale),
            (
                second_weight,
                second_output.len(),
                second_rows,
                second_scale,
            ),
            (third_weight, third_output.len(), third_rows, third_scale),
        ],
        cols,
        threads,
    )?;
    unsafe {
        check_cuda(
            "infer_fp8_linear_triple_f32_configured_on_stream",
            ffi::infer_fp8_linear_triple_f32_configured_on_stream(
                input.ptr,
                first_weight.ptr,
                second_weight.ptr,
                third_weight.ptr,
                first_output.buffer_mut().ptr,
                second_output.buffer_mut().ptr,
                third_output.buffer_mut().ptr,
                first_rows as u32,
                second_rows as u32,
                third_rows as u32,
                cols as u32,
                first_scale,
                second_scale,
                third_scale,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_segmented_fp8_linear(
    input: &DeviceBuffer<f32>,
    segments: &[(&DeviceBuffer<u8>, usize, usize, f32)],
    cols: usize,
    threads: usize,
) -> Result<()> {
    if cols == 0
        || cols > u32::MAX as usize
        || !(64..=512).contains(&threads)
        || !threads.is_multiple_of(32)
        || input.len() != cols
    {
        return Err(Error::Shape {
            label: "segmented FP8 linear dimensions",
            expected: "input=cols; non-zero u32 cols; threads a multiple of 32 in 64..=512"
                .to_string(),
            actual: format!("input={} cols={cols} threads={threads}", input.len()),
        });
    }
    for (index, (weight, output_len, rows, scale)) in segments.iter().enumerate() {
        let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
            label: "segmented FP8 linear weight",
            expected: "rows * cols without overflow".to_string(),
            actual: format!("segment={index} rows={rows} cols={cols}"),
        })?;
        if *rows == 0
            || *rows > u32::MAX as usize
            || weight.len() != weight_len
            || *output_len != *rows
        {
            return Err(Error::Shape {
                label: "segmented FP8 linear buffers",
                expected: format!("segment={index} weight={weight_len} output={rows}"),
                actual: format!(
                    "segment={index} weight={} output={output_len} rows={rows}",
                    weight.len()
                ),
            });
        }
        if !scale.is_finite() {
            return Err(Error::Format {
                label: "segmented FP8 linear weight scale",
                detail: format!("segment={index} expected finite scale, got {scale}"),
            });
        }
    }
    Ok(())
}

/// Enqueues an f32-input, FP8-weight linear projection with one dequantization
/// scale per output channel.
pub fn fp8_linear_channel_scaled_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    channel_weight_scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "channel-scaled FP8 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !(64..=512).contains(&threads)
        || !threads.is_multiple_of(32)
    {
        return Err(Error::Shape {
            label: "channel-scaled FP8 linear dimensions",
            expected: "non-zero u32-sized rows and cols; threads a multiple of 32 in 64..=512"
                .to_string(),
            actual: format!("rows={rows} cols={cols} threads={threads}"),
        });
    }
    if input.len() != cols
        || weight.len() != weight_len
        || channel_weight_scale.len() != rows
        || output.len() != rows
    {
        return Err(Error::Shape {
            label: "channel-scaled FP8 linear buffers",
            expected: format!("input={cols} weight={weight_len} scales={rows} output={rows}"),
            actual: format!(
                "input={} weight={} scales={} output={}",
                input.len(),
                weight.len(),
                channel_weight_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_channel_scaled_f32_configured_on_stream",
            ffi::infer_fp8_linear_channel_scaled_f32_configured_on_stream(
                input.ptr,
                weight.ptr,
                channel_weight_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues channel-scaled W8A16 FP8 projections for a dense f32 batch.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_channel_scaled_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    channel_weight_scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    batch_size: usize,
    rows: usize,
    cols: usize,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = batch_size.saturating_mul(cols);
    let weight_len = rows.saturating_mul(cols);
    let output_len = batch_size.saturating_mul(rows);
    if batch_size == 0
        || rows == 0
        || cols == 0
        || input.len() < input_len
        || weight.len() != weight_len
        || channel_weight_scale.len() != rows
        || output.len() < output_len
        || batch_size > u32::MAX as usize
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !(64..=512).contains(&threads)
        || !threads.is_multiple_of(32)
    {
        return Err(Error::Shape {
            label: "batched channel-scaled FP8 W8A16 linear buffers",
            expected: format!(
                "input={input_len} weight={weight_len} scales={rows} output={output_len}"
            ),
            actual: format!(
                "input={} weight={} scales={} output={} batch={batch_size} rows={rows} cols={cols} threads={threads}",
                input.len(),
                weight.len(),
                channel_weight_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_channel_scaled_f32_batch_configured_on_stream",
            ffi::infer_fp8_linear_channel_scaled_f32_batch_configured_on_stream(
                input.ptr,
                weight.ptr,
                channel_weight_scale.ptr,
                output.buffer_mut().ptr,
                batch_size as u32,
                rows as u32,
                cols as u32,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues a channel-scaled FP8 projection with dynamic per-token E4M3
/// activation quantization.
pub fn fp8_linear_channel_scaled_dynamic_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    channel_weight_scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "dynamic channel-scaled FP8 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() != cols
        || weight.len() != weight_len
        || channel_weight_scale.len() != rows
        || output.len() != rows
    {
        return Err(Error::Shape {
            label: "dynamic channel-scaled FP8 linear",
            expected: format!("input={cols} weight={weight_len} scales={rows} output={rows}"),
            actual: format!(
                "rows={rows} cols={cols} input={} weight={} scales={} output={}",
                input.len(),
                weight.len(),
                channel_weight_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_channel_scaled_dynamic_f32_on_stream",
            ffi::infer_fp8_linear_channel_scaled_dynamic_f32_on_stream(
                input.ptr,
                weight.ptr,
                channel_weight_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues channel-scaled FP8 projection with dynamic per-token E4M3
/// activation quantization, using persistent storage for the reduced scale.
pub fn fp8_linear_channel_scaled_precomputed_dynamic_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    channel_weight_scale: &DeviceBuffer<f32>,
    input_scale: &mut DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "precomputed dynamic channel-scaled FP8 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() != cols
        || weight.len() != weight_len
        || channel_weight_scale.len() != rows
        || input_scale.len() != 1
        || output.len() != rows
    {
        return Err(Error::Shape {
            label: "precomputed dynamic channel-scaled FP8 linear",
            expected: format!(
                "input={cols} weight={weight_len} scales={rows} input_scale=1 output={rows}"
            ),
            actual: format!(
                "rows={rows} cols={cols} input={} weight={} scales={} input_scale={} output={}",
                input.len(),
                weight.len(),
                channel_weight_scale.len(),
                input_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_channel_scaled_precomputed_dynamic_f32_on_stream",
            ffi::infer_fp8_linear_channel_scaled_precomputed_dynamic_f32_on_stream(
                input.ptr,
                weight.ptr,
                channel_weight_scale.ptr,
                input_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues channel-scaled FP8 projection after dynamically quantizing the
/// activation vector once into persistent storage.
pub fn fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    quantized_input: &mut DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    channel_weight_scale: &DeviceBuffer<f32>,
    input_scale: &mut DeviceBuffer<f32>,
    output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    fp8_linear_channel_scaled_dynamic_quantized_f32_configured_into_on_stream(
        input,
        quantized_input,
        weight,
        channel_weight_scale,
        input_scale,
        output,
        rows,
        cols,
        256,
        stream,
    )
}

/// Enqueues the dynamically quantized channel-scaled FP8 projection with an
/// explicit row-reduction block size for shape-specific scheduling.
#[allow(clippy::too_many_arguments)]
pub fn fp8_linear_channel_scaled_dynamic_quantized_f32_configured_into_on_stream(
    input: &DeviceBuffer<f32>,
    quantized_input: &mut DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    channel_weight_scale: &DeviceBuffer<f32>,
    input_scale: &mut DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "quantized dynamic channel-scaled FP8 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || !(64..=512).contains(&threads)
        || !threads.is_multiple_of(32)
        || input.len() != cols
        || quantized_input.len() < cols
        || weight.len() != weight_len
        || channel_weight_scale.len() != rows
        || input_scale.len() != 1
        || output.len() != rows
    {
        return Err(Error::Shape {
            label: "quantized dynamic channel-scaled FP8 linear",
            expected: format!(
                "input={cols} quantized_input>={cols} weight={weight_len} scales={rows} input_scale=1 output={rows} threads=64..512/multiple-of-32"
            ),
            actual: format!(
                "rows={rows} cols={cols} input={} quantized_input={} weight={} scales={} input_scale={} output={} threads={threads}",
                input.len(),
                quantized_input.len(),
                weight.len(),
                channel_weight_scale.len(),
                input_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_channel_scaled_dynamic_quantized_f32_configured_on_stream",
            ffi::infer_fp8_linear_channel_scaled_dynamic_quantized_f32_configured_on_stream(
                input.ptr,
                quantized_input.ptr,
                weight.ptr,
                channel_weight_scale.ptr,
                input_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                threads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues device-routed channel-scaled FP8 gate and up projections.
#[allow(clippy::too_many_arguments)]
pub fn fp8_moe_grouped_gate_up_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    gate_weights: &DeviceBuffer<*const u8>,
    gate_scales: &DeviceBuffer<*const f32>,
    up_weights: &DeviceBuffer<*const u8>,
    up_scales: &DeviceBuffer<*const f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    slots: usize,
    stream: &CudaStream,
) -> Result<()> {
    let output_len = rows.saturating_mul(2).saturating_mul(slots);
    if rows == 0
        || cols == 0
        || slots == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || slots > u32::MAX as usize
        || rows.saturating_mul(slots) > u32::MAX as usize
        || indices.len() < slots
        || input.len() != cols
        || input_scale.len() != 1
        || gate_weights.len() != gate_scales.len()
        || gate_weights.len() != up_weights.len()
        || gate_weights.len() != up_scales.len()
        || output.len() != output_len
    {
        return Err(Error::Shape {
            label: "grouped FP8 gate/up",
            expected: format!(
                "indices>={slots} input={cols} input_scale=1 matching expert tables output={output_len}"
            ),
            actual: format!(
                "indices={} input={} input_scale={} gate_weights={} gate_scales={} up_weights={} up_scales={} output={}",
                indices.len(),
                input.len(),
                input_scale.len(),
                gate_weights.len(),
                gate_scales.len(),
                up_weights.len(),
                up_scales.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_moe_grouped_gate_up_f32_on_stream",
            ffi::infer_fp8_moe_grouped_gate_up_f32_on_stream(
                indices.ptr,
                input.ptr,
                input_scale.ptr,
                gate_weights.ptr,
                gate_scales.ptr,
                up_weights.ptr,
                up_scales.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                slots as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies SiLU to grouped gate/up slots and dynamically quantizes each slot to FP8.
pub fn moe_silu_quantize_fp8_slots_f32_into_on_stream(
    gate_up: &DeviceBuffer<f32>,
    quantized: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<f32>,
    rows: usize,
    slots: usize,
    stream: &CudaStream,
) -> Result<()> {
    let gate_up_len = rows.saturating_mul(2).saturating_mul(slots);
    let quantized_len = rows.saturating_mul(slots);
    if rows == 0
        || slots == 0
        || rows > u32::MAX as usize
        || slots > u32::MAX as usize
        || gate_up.len() != gate_up_len
        || quantized.len() != quantized_len
        || scales.len() != slots
    {
        return Err(Error::Shape {
            label: "MoE SiLU FP8 slot quantization",
            expected: format!("gate_up={gate_up_len} quantized={quantized_len} scales={slots}"),
            actual: format!(
                "gate_up={} quantized={} scales={}",
                gate_up.len(),
                quantized.len(),
                scales.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_moe_silu_quantize_fp8_slots_f32_on_stream",
            ffi::infer_moe_silu_quantize_fp8_slots_f32_on_stream(
                gate_up.ptr,
                quantized.ptr,
                scales.ptr,
                rows as u32,
                slots as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues device-routed channel-scaled FP8 down projections for quantized slots.
#[allow(clippy::too_many_arguments)]
pub fn fp8_moe_grouped_down_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    inputs: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<*const u8>,
    weight_scales: &DeviceBuffer<*const f32>,
    outputs: &DeviceBuffer<*mut f32>,
    rows: usize,
    cols: usize,
    slots: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = cols.saturating_mul(slots);
    if rows == 0
        || cols == 0
        || slots == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || slots > u32::MAX as usize
        || rows.saturating_mul(slots) > u32::MAX as usize
        || indices.len() < slots
        || inputs.len() != input_len
        || input_scales.len() != slots
        || weights.len() != weight_scales.len()
        || outputs.len() != slots
    {
        return Err(Error::Shape {
            label: "grouped FP8 down",
            expected: format!(
                "indices>={slots} inputs={input_len} input_scales={slots} matching expert tables outputs={slots}"
            ),
            actual: format!(
                "indices={} inputs={} input_scales={} weights={} weight_scales={} outputs={}",
                indices.len(),
                inputs.len(),
                input_scales.len(),
                weights.len(),
                weight_scales.len(),
                outputs.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_moe_grouped_down_f32_on_stream",
            ffi::infer_fp8_moe_grouped_down_f32_on_stream(
                indices.ptr,
                inputs.ptr,
                input_scales.ptr,
                weights.ptr,
                weight_scales.ptr,
                outputs.ptr,
                rows as u32,
                cols as u32,
                slots as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Dynamically quantizes one f32 activation vector to E4M3 on `stream`.
///
/// The device-resident `input_scale` receives `max(abs(input)) / 448` and can
/// be consumed by subsequent kernels without synchronizing it to the host.
pub fn quantize_fp8_e4m3_dynamic_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    quantized_input: &mut DeviceBuffer<u8>,
    input_scale: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty()
        || input.len() > u32::MAX as usize
        || quantized_input.len() < input.len()
        || input_scale.len() != 1
    {
        return Err(Error::Shape {
            label: "dynamic FP8 quantization",
            expected: format!(
                "non-empty input<=u32::MAX quantized_input>={} input_scale=1",
                input.len()
            ),
            actual: format!(
                "input={} quantized_input={} input_scale={}",
                input.len(),
                quantized_input.len(),
                input_scale.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_quantize_fp8_e4m3_dynamic_f32_on_stream",
            ffi::infer_quantize_fp8_e4m3_dynamic_f32_on_stream(
                input.ptr,
                quantized_input.ptr,
                input_scale.ptr,
                input.len() as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Dynamically quantizes each row of an f32 matrix to E4M3 independently.
pub fn quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    quantized_input: &mut DeviceBuffer<u8>,
    input_scale: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "batched dynamic FP8 quantization",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() < len
        || quantized_input.len() < len
        || input_scale.len() < rows
    {
        return Err(Error::Shape {
            label: "batched dynamic FP8 quantization buffers",
            expected: format!("input={len} quantized_input>={len} input_scale={rows}"),
            actual: format!(
                "input={} quantized_input={} input_scale={}",
                input.len(),
                quantized_input.len(),
                input_scale.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_quantize_fp8_e4m3_dynamic_f32_batch_on_stream",
            ffi::infer_quantize_fp8_e4m3_dynamic_f32_batch_on_stream(
                input.ptr,
                quantized_input.ptr,
                input_scale.ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Multiplies each f32 value by its channel scale and one device scalar.
pub fn scale_channel_f32_device_scalar_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    channel_scale: &DeviceBuffer<f32>,
    scalar: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if values.is_empty()
        || values.len() > u32::MAX as usize
        || channel_scale.len() != values.len()
        || scalar.len() != 1
    {
        return Err(Error::Shape {
            label: "channel-scaled device-scalar f32",
            expected: format!("values=scales={} scalar=1", values.len()),
            actual: format!(
                "values={} scales={} scalar={}",
                values.len(),
                channel_scale.len(),
                scalar.len()
            ),
        });
    }
    let len = values.len();
    unsafe {
        check_cuda(
            "infer_scale_channel_f32_device_scalar_on_stream",
            ffi::infer_scale_channel_f32_device_scalar_on_stream(
                values.buffer_mut().ptr,
                channel_scale.ptr,
                scalar.ptr,
                len as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies one channel scale and one per-row device scale to a row-major matrix.
pub fn scale_channel_f32_device_row_scalar_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    channel_scale: &DeviceBuffer<f32>,
    row_scale: &DeviceBuffer<f32>,
    rows: usize,
    channels: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(channels).ok_or_else(|| Error::Shape {
        label: "channel-scaled device-row-scalar f32",
        expected: "rows * channels without overflow".to_string(),
        actual: format!("rows={rows} channels={channels}"),
    })?;
    if rows == 0
        || channels == 0
        || rows > u32::MAX as usize
        || channels > u32::MAX as usize
        || values.len() < len
        || channel_scale.len() != channels
        || row_scale.len() < rows
    {
        return Err(Error::Shape {
            label: "channel-scaled device-row-scalar f32 buffers",
            expected: format!("values={len} channel_scale={channels} row_scale={rows}"),
            actual: format!(
                "values={} channel_scale={} row_scale={}",
                values.len(),
                channel_scale.len(),
                row_scale.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_scale_channel_f32_device_row_scalar_on_stream",
            ffi::infer_scale_channel_f32_device_row_scalar_on_stream(
                values.buffer_mut().ptr,
                channel_scale.ptr,
                row_scale.ptr,
                rows as u32,
                channels as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues f32-input, FP8 E4M3-weight linear projection with static FP8
/// activation quantization using the checkpoint input scale.
pub fn fp8_linear_w8a8_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let weight_len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "FP8 W8A8 linear weight",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0 || cols == 0 || rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err(Error::Shape {
            label: "FP8 W8A8 linear dimensions",
            expected: "non-zero u32-sized rows and cols".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        });
    }
    if !weight_scale.is_finite() || !input_scale.is_finite() || input_scale <= 0.0 {
        return Err(Error::Format {
            label: "FP8 W8A8 linear scales",
            detail: format!("weight_scale={weight_scale} input_scale={input_scale}"),
        });
    }
    if input.len() != cols || weight.len() != weight_len || output.len() != rows {
        return Err(Error::Shape {
            label: "FP8 W8A8 linear buffers",
            expected: format!("input={cols} weight={weight_len} output={rows}"),
            actual: format!(
                "input={} weight={} output={}",
                input.len(),
                weight.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_fp8_linear_w8a8_f32_on_stream",
            ffi::infer_fp8_linear_w8a8_f32_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                weight_scale,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Quantizes an f32 activation vector to E4M3 with a static calibrated scale.
pub fn quantize_fp8_e4m3_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, u8>,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty() || input.len() > u32::MAX as usize || output.len() != input.len() {
        return Err(Error::Shape {
            label: "FP8 activation quantization buffers",
            expected: format!("non-empty equal lengths <= {}", u32::MAX),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    if !input_scale.is_finite() || input_scale <= 0.0 {
        return Err(Error::Format {
            label: "FP8 activation input_scale",
            detail: format!("expected positive finite scale, got {input_scale}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_quantize_fp8_e4m3_f32_on_stream",
            ffi::infer_quantize_fp8_e4m3_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Quantizes a row-major BF16 matrix to E4M3 with one scale per row.
pub fn quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream(
    input: &DeviceBuffer<u16>,
    channel_scale: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, u8>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "channel-scaled BF16-to-FP8 quantization",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || len > u32::MAX as usize
        || input.len() != len
        || output.len() != len
        || channel_scale.len() != rows
    {
        return Err(Error::Shape {
            label: "channel-scaled BF16-to-FP8 quantization buffers",
            expected: format!("input={len} scales={rows} output={len}; u32-sized dimensions"),
            actual: format!(
                "input={} scales={} output={} rows={rows} cols={cols}",
                input.len(),
                channel_scale.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_quantize_fp8_e4m3_bf16_channel_scaled_on_stream",
            ffi::infer_quantize_fp8_e4m3_bf16_channel_scaled_on_stream(
                input.ptr,
                channel_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn validate_nvfp4_w4a16_matvec(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    output_len: usize,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
) -> Result<()> {
    if in_features == 0 || !in_features.is_multiple_of(16) || out_features == 0 {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 matvec dimensions",
            expected: "non-zero out_features, in_features divisible by 16".to_string(),
            actual: format!("out={out_features} in={in_features}"),
        });
    }
    if !weight_scale_2.is_finite() {
        return Err(Error::Format {
            label: "NVFP4 W4A16 weight_scale_2",
            detail: format!("expected finite scale, got {weight_scale_2}"),
        });
    }
    let weight_bytes = out_features * in_features / 2;
    let scale_bytes = out_features * (in_features / 16);
    if input.len() != in_features
        || packed_weight.len() != weight_bytes
        || weight_scale.len() != scale_bytes
        || output_len != out_features
    {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 matvec buffers",
            expected: format!(
                "input={in_features} weight={weight_bytes} scale={scale_bytes} output={out_features}"
            ),
            actual: format!(
                "input={} weight={} scale={} output={output_len}",
                input.len(),
                packed_weight.len(),
                weight_scale.len(),
            ),
        });
    }
    if out_features > u32::MAX as usize || in_features > u32::MAX as usize {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 matvec dimensions",
            expected: "u32-sized dimensions".to_string(),
            actual: format!("out={out_features} in={in_features}"),
        });
    }
    Ok(())
}

/// Enqueues the original block-per-row W4A16 matvec schedule.
///
/// This entry point is retained for schedule micromeasurements. Production
/// callers should use [`nvfp4_w4a16_matvec_f32_into_on_stream`].
pub fn nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    stream: &CudaStream,
) -> Result<()> {
    validate_nvfp4_w4a16_matvec(
        input,
        packed_weight,
        weight_scale,
        output.len(),
        out_features,
        in_features,
        weight_scale_2,
    )?;
    unsafe {
        check_cuda(
            "infer_nvfp4_w4a16_matvec_f32_on_stream",
            ffi::infer_nvfp4_w4a16_matvec_f32_on_stream(
                input.ptr,
                packed_weight.ptr,
                weight_scale.ptr,
                output.buffer_mut().ptr,
                out_features as u32,
                in_features as u32,
                weight_scale_2,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues a W4A16 NVFP4 matvec: `output = weight * input * weight_scale_2`.
///
/// `input` is `in_features` f32 values. `packed_weight` is row-major
/// `[out_features, in_features]` packed E2M1 (2 values per byte, low nibble
/// first). `weight_scale` is `[out_features, in_features / 16]` UE4M3
/// per-block scales. `weight_scale_2` is the scalar tensor-wide weight scale
/// from ModelOpt.
pub fn nvfp4_w4a16_matvec_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    output: DeviceOutput<'_, f32>,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    stream: &CudaStream,
) -> Result<()> {
    nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
        input,
        packed_weight,
        weight_scale,
        output,
        out_features,
        in_features,
        weight_scale_2,
        8,
        stream,
    )
}

/// Enqueues one W4A16 matvec per row of a dense activation batch.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_w4a16_matvec_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    stream: &CudaStream,
) -> Result<()> {
    let expected_input = rows.saturating_mul(in_features);
    let expected_output = rows.saturating_mul(out_features);
    let weight_bytes = out_features.saturating_mul(in_features / 2);
    let scale_bytes = out_features.saturating_mul(in_features / 16);
    if rows == 0
        || in_features == 0
        || !in_features.is_multiple_of(16)
        || out_features == 0
        || input.len() < expected_input
        || output.len() < expected_output
        || packed_weight.len() != weight_bytes
        || weight_scale.len() != scale_bytes
        || rows > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "batched NVFP4 W4A16 matvec buffers",
            expected: format!(
                "input={expected_input} weight={weight_bytes} scale={scale_bytes} output={expected_output}"
            ),
            actual: format!(
                "input={} weight={} scale={} output={} rows={rows} out={out_features} in={in_features}",
                input.len(),
                packed_weight.len(),
                weight_scale.len(),
                output.len()
            ),
        });
    }
    if !weight_scale_2.is_finite() {
        return Err(Error::Format {
            label: "batched NVFP4 W4A16 weight_scale_2",
            detail: format!("expected finite scale, got {weight_scale_2}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_on_stream",
            ffi::infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_on_stream(
                input.ptr,
                packed_weight.ptr,
                weight_scale.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                out_features as u32,
                in_features as u32,
                weight_scale_2,
                8,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues one W4A16 matvec per activation row using the packed payload of a
/// cuBLASLt-layout NVFP4 weight matrix.
///
/// The matrix must be the `K x M` representation of a row-major `[M, K]`
/// ModelOpt weight. Its tiled scale metadata is ignored here; `weight_scale`
/// supplies the simple row-major scales required by the decode kernel.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_w4a16_matrix_matvec_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &Nvfp4Matrix,
    weight_scale: &DeviceBuffer<u8>,
    output: DeviceOutput<'_, f32>,
    rows: usize,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    stream: &CudaStream,
) -> Result<()> {
    if (packed_weight.rows, packed_weight.cols) != (in_features, out_features) {
        return Err(Error::Shape {
            label: "batched NVFP4 W4A16 matrix weight",
            expected: format!("{}x{} KxM matrix", in_features, out_features),
            actual: format!("{}x{}", packed_weight.rows, packed_weight.cols),
        });
    }
    nvfp4_w4a16_matvec_f32_batch_into_on_stream(
        input,
        &packed_weight.values,
        weight_scale,
        output,
        rows,
        out_features,
        in_features,
        weight_scale_2,
        stream,
    )
}

/// Enqueues the W4A16 matvec using one warp per output row.
///
/// This configured entry point exists to measure launch schedules on real model
/// shapes. Production callers should use [`nvfp4_w4a16_matvec_f32_into_on_stream`].
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    mut output: DeviceOutput<'_, f32>,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    warps_per_block: usize,
    stream: &CudaStream,
) -> Result<()> {
    if !matches!(warps_per_block, 4 | 8 | 16 | 32) {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 warp-row schedule",
            expected: "4, 8, 16, or 32 warps per block".to_string(),
            actual: warps_per_block.to_string(),
        });
    }
    validate_nvfp4_w4a16_matvec(
        input,
        packed_weight,
        weight_scale,
        output.len(),
        out_features,
        in_features,
        weight_scale_2,
    )?;
    unsafe {
        check_cuda(
            "infer_nvfp4_w4a16_matvec_f32_warp_rows_on_stream",
            ffi::infer_nvfp4_w4a16_matvec_f32_warp_rows_on_stream(
                input.ptr,
                packed_weight.ptr,
                weight_scale.ptr,
                output.buffer_mut().ptr,
                out_features as u32,
                in_features as u32,
                weight_scale_2,
                warps_per_block as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues device-routed grouped W4A16 NVFP4 matvecs.
///
/// Every route selects one raw ModelOpt weight and scale pair. The shared f32
/// activation remains unquantized, and each selected expert's
/// `weight_scale_2` is applied exactly once to its f32 output.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    packed_weight_table: &DeviceBuffer<*const u8>,
    weight_scale_table: &DeviceBuffer<*const u8>,
    weight_scale_2_table: &DeviceBuffer<f32>,
    output_table: &DeviceBuffer<*mut f32>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    let table_len = packed_weight_table.len();
    let shared_memory_bytes = in_features
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Shape {
            label: "NVFP4 grouped W4A16 shared memory",
            expected: "in_features * sizeof(f32) without overflow".to_string(),
            actual: format!("in_features={in_features}"),
        })?;
    if groups == 0
        || table_len == 0
        || weight_scale_table.len() != table_len
        || weight_scale_2_table.len() != table_len
        || output_table.len() != groups
        || input.len() != in_features
        || out_features == 0
        || in_features == 0
        || !in_features.is_multiple_of(16)
        || groups > u32::MAX as usize
        || table_len > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "NVFP4 grouped W4A16 matvec buffers",
            expected: "matching non-empty expert tables, route/output tables, and W4A16 dimensions"
                .to_string(),
            actual: format!(
                "indices={} input={} weights={} scales={} weight_scale_2={} outputs={} out={out_features} in={in_features} shared_memory_bytes={shared_memory_bytes}",
                groups,
                input.len(),
                table_len,
                weight_scale_table.len(),
                weight_scale_2_table.len(),
                output_table.len(),
            ),
        });
    }
    let max_shared_memory_bytes = max_shared_memory_per_block()?;
    if shared_memory_bytes > max_shared_memory_bytes {
        return Err(Error::Shape {
            label: "NVFP4 grouped W4A16 shared memory",
            expected: format!("at most {max_shared_memory_bytes} bytes per block"),
            actual: format!("{shared_memory_bytes} bytes for in_features={in_features}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_nvfp4_w4a16_grouped_matvec_f32_on_stream",
            ffi::infer_nvfp4_w4a16_grouped_matvec_f32_on_stream(
                indices.ptr,
                input.ptr,
                packed_weight_table.ptr,
                weight_scale_table.ptr,
                weight_scale_2_table.ptr,
                output_table.ptr,
                table_len as u32,
                groups as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues device-routed grouped W4A16 matvecs with one f32 input per route.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input_table: &DeviceBuffer<*const f32>,
    packed_weight_table: &DeviceBuffer<*const u8>,
    weight_scale_table: &DeviceBuffer<*const u8>,
    weight_scale_2_table: &DeviceBuffer<f32>,
    output_table: &DeviceBuffer<*mut f32>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    nvfp4_w4a16_grouped_inputs_matvec_f32_prefix_into_on_stream(
        indices,
        input_table,
        packed_weight_table,
        weight_scale_table,
        weight_scale_2_table,
        output_table,
        indices.len(),
        out_features,
        in_features,
        stream,
    )
}

/// Enqueues an active prefix of device-routed grouped W4A16 matvecs.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_w4a16_grouped_inputs_matvec_f32_prefix_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input_table: &DeviceBuffer<*const f32>,
    packed_weight_table: &DeviceBuffer<*const u8>,
    weight_scale_table: &DeviceBuffer<*const u8>,
    weight_scale_2_table: &DeviceBuffer<f32>,
    output_table: &DeviceBuffer<*mut f32>,
    groups: usize,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let table_len = packed_weight_table.len();
    let shared_memory_bytes = in_features
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Shape {
            label: "NVFP4 grouped-input W4A16 shared memory",
            expected: "in_features * sizeof(f32) without overflow".to_string(),
            actual: format!("in_features={in_features}"),
        })?;
    if groups == 0
        || table_len == 0
        || indices.len() < groups
        || input_table.len() < groups
        || weight_scale_table.len() != table_len
        || weight_scale_2_table.len() != table_len
        || output_table.len() < groups
        || out_features == 0
        || in_features == 0
        || !in_features.is_multiple_of(16)
        || groups > u32::MAX as usize
        || table_len > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "NVFP4 grouped-input W4A16 matvec buffers",
            expected: "matching non-empty input, expert, route, and output tables".to_string(),
            actual: format!(
                "indices={} inputs={} weights={} scales={} weight_scale_2={} outputs={} out={out_features} in={in_features}",
                groups,
                input_table.len(),
                table_len,
                weight_scale_table.len(),
                weight_scale_2_table.len(),
                output_table.len(),
            ),
        });
    }
    let max_shared_memory_bytes = max_shared_memory_per_block()?;
    if shared_memory_bytes > max_shared_memory_bytes {
        return Err(Error::Shape {
            label: "NVFP4 grouped-input W4A16 shared memory",
            expected: format!("at most {max_shared_memory_bytes} bytes per block"),
            actual: format!("{shared_memory_bytes} bytes for in_features={in_features}"),
        });
    }
    unsafe {
        check_cuda(
            "infer_nvfp4_w4a16_grouped_inputs_matvec_f32_on_stream",
            ffi::infer_nvfp4_w4a16_grouped_inputs_matvec_f32_on_stream(
                indices.ptr,
                input_table.ptr,
                packed_weight_table.ptr,
                weight_scale_table.ptr,
                weight_scale_2_table.ptr,
                output_table.ptr,
                table_len as u32,
                groups as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(clippy::too_many_arguments)]
/// Enqueues fused NVFP4 W4A16 matvec plus top-1 selection without writing logits.
pub fn nvfp4_w4a16_top1_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    scratch_value: &DeviceBuffer<f32>,
    scratch_index: &DeviceBuffer<u32>,
    out_index: &DeviceBuffer<u32>,
    out_value: &DeviceBuffer<f32>,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    stream: &CudaStream,
) -> Result<()> {
    nvfp4_w4a16_top1_configured_f32_into_on_stream(
        input,
        packed_weight,
        weight_scale,
        scratch_value,
        scratch_index,
        out_index,
        out_value,
        out_features,
        in_features,
        weight_scale_2,
        16,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
/// Enqueues fused NVFP4 W4A16 top-1 with a selected number of row warps per block.
pub fn nvfp4_w4a16_top1_configured_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    packed_weight: &DeviceBuffer<u8>,
    weight_scale: &DeviceBuffer<u8>,
    scratch_value: &DeviceBuffer<f32>,
    scratch_index: &DeviceBuffer<u32>,
    out_index: &DeviceBuffer<u32>,
    out_value: &DeviceBuffer<f32>,
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
    warps_per_block: usize,
    stream: &CudaStream,
) -> Result<()> {
    if in_features == 0 || !in_features.is_multiple_of(16) || out_features == 0 {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 top1 dimensions",
            expected: "non-zero out_features, in_features divisible by 16".to_string(),
            actual: format!("out={out_features} in={in_features}"),
        });
    }
    let weight_bytes = out_features * in_features / 2;
    let scale_bytes = out_features * (in_features / 16);
    if !matches!(warps_per_block, 4 | 8 | 16 | 32) {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 top1 warps per block",
            expected: "4, 8, 16, or 32".to_string(),
            actual: warps_per_block.to_string(),
        });
    }
    let scratch_len = out_features.div_ceil(warps_per_block);
    if input.len() != in_features
        || packed_weight.len() != weight_bytes
        || weight_scale.len() != scale_bytes
        || scratch_value.len() < scratch_len
        || scratch_index.len() < scratch_len
        || out_index.len() != 1
        || out_value.len() != 1
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
        || !weight_scale_2.is_finite()
    {
        return Err(Error::Shape {
            label: "NVFP4 W4A16 top1 buffers",
            expected: "matching input, weight, scratch, and output buffers".to_string(),
            actual: format!(
                "input={} weight={} scale={} scratch_value={} scratch_index={} out_index={} out_value={} out={out_features} in={in_features} weight_scale_2={weight_scale_2}",
                input.len(),
                packed_weight.len(),
                weight_scale.len(),
                scratch_value.len(),
                scratch_index.len(),
                out_index.len(),
                out_value.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nvfp4_w4a16_top1_f32_on_stream",
            ffi::infer_nvfp4_w4a16_top1_f32_on_stream(
                input.ptr,
                packed_weight.ptr,
                weight_scale.ptr,
                scratch_value.ptr,
                scratch_index.ptr,
                scratch_len as u32,
                out_index.ptr,
                out_value.ptr,
                out_features as u32,
                in_features as u32,
                weight_scale_2,
                warps_per_block as u32,
                stream.as_raw(),
            ),
        )
    }
}

///
/// This applies the rolling causal depthwise conv over the last three cached
/// pre-conv QKV values plus the current QKV projection, updates the recurrent
/// conv state, splits Q/K/V, repeats Q/K from key heads to value heads, and
/// L2-normalizes Q/K per 128-wide value head.
pub fn qwen36_gdn_prep_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    mut conv_state: DeviceInOut<'_, f32>,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let key_dim = key_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.6 GDN prep",
            expected: "key_heads * head_dim without overflow".to_string(),
            actual: format!("key_heads={key_heads} head_dim={head_dim}"),
        })?;
    let value_dim = value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.6 GDN prep",
            expected: "value_heads * head_dim without overflow".to_string(),
            actual: format!("value_heads={value_heads} head_dim={head_dim}"),
        })?;
    let conv_dim = key_dim * 2 + value_dim;
    if key_heads == 0
        || value_heads == 0
        || head_dim != 128
        || !value_heads.is_multiple_of(key_heads)
        || key_heads > u32::MAX as usize
        || value_heads > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Qwen3.6 GDN prep dimensions",
            expected: "head_dim=128 and value_heads divisible by key_heads".to_string(),
            actual: format!("key_heads={key_heads} value_heads={value_heads} head_dim={head_dim}"),
        });
    }
    if qkv.len() != conv_dim
        || conv_weight_bf16.len() != conv_dim * 4
        || q.len() != value_dim
        || k.len() != value_dim
        || v.len() != value_dim
        || conv_state.len() != conv_dim * 3
    {
        return Err(Error::Shape {
            label: "Qwen3.6 GDN prep buffers",
            expected: format!(
                "qkv={conv_dim} conv_weight={} q/k/v={value_dim} conv_state={}",
                conv_dim * 4,
                conv_dim * 3
            ),
            actual: format!(
                "qkv={} conv_weight={} q={} k={} v={} conv_state={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_prep_on_stream",
            ffi::infer_qwen36_gdn_prep_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state.buffer_mut().ptr,
                key_heads as u32,
                value_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies Ling 3's causal depthwise convolution and exact Q/K L2 normalization.
///
/// Q, K, and V projections and convolution weights are concatenated in that
/// order. The persistent convolution state stores the previous three raw
/// projection values for every channel.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_prep_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    mut conv_state: DeviceInOut<'_, f32>,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let projection = heads.checked_mul(128).ok_or_else(|| Error::Shape {
        label: "Ling 3 KDA preparation",
        expected: "heads * 128 without overflow".to_string(),
        actual: format!("heads={heads}"),
    })?;
    let conv_dim = projection.saturating_mul(3);
    if heads == 0
        || heads > u32::MAX as usize
        || qkv.len() != conv_dim
        || conv_weight_bf16.len() != conv_dim * 4
        || q.len() != projection
        || k.len() != projection
        || v.len() != projection
        || conv_state.len() != conv_dim * 3
    {
        return Err(Error::Shape {
            label: "Ling 3 KDA preparation buffers",
            expected: format!(
                "qkv={conv_dim} conv_weight={} q/k/v={projection} conv_state={}",
                conv_dim * 4,
                conv_dim * 3,
            ),
            actual: format!(
                "qkv={} conv_weight={} q={} k={} v={} conv_state={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_prep_on_stream",
            ffi::infer_ling3_kda_prep_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state.buffer_mut().ptr,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies Qwen3.6 convolution/GDN preparation to a changing sequence batch.
/// `state_table_offset` selects the first row from a larger state-pointer table.
/// Applies Ling 3 convolution and Q/K normalization to ragged prompt chunks.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_prep_chunks_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    conv_state_table: &DeviceBuffer<*mut f32>,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    sequence_count: usize,
    total_tokens: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let width = heads.saturating_mul(128);
    let conv_width = width.saturating_mul(3);
    let values = total_tokens.saturating_mul(width);
    if sequence_count == 0
        || total_tokens == 0
        || heads == 0
        || [sequence_count, total_tokens, heads]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || qkv.len() != total_tokens.saturating_mul(conv_width)
        || conv_weight_bf16.len() != conv_width.saturating_mul(4)
        || q.len() != values
        || k.len() != values
        || v.len() != values
        || conv_state_table.len() < sequence_count
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "Ling 3 chunked KDA preparation buffers",
            expected: format!(
                "qkv={} weight={} q/k/v={values} state/metadata>={sequence_count}",
                total_tokens.saturating_mul(conv_width),
                conv_width.saturating_mul(4)
            ),
            actual: format!(
                "qkv={} weight={} q={} k={} v={} states={} offsets={} lengths={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_prep_chunks_on_stream",
            ffi::infer_ling3_kda_prep_chunks_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state_table.ptr,
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                sequence_count as u32,
                total_tokens as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies Ling 3 convolution and Q/K normalisation to one contiguous prompt.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_prep_rows_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    mut conv_state: DeviceInOut<'_, f32>,
    rows: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let projection = heads.saturating_mul(128);
    let conv_width = projection.saturating_mul(3);
    let values = rows.saturating_mul(projection);
    if rows == 0
        || heads == 0
        || rows > u32::MAX as usize
        || heads > u32::MAX as usize
        || qkv.len() != rows.saturating_mul(conv_width)
        || conv_weight_bf16.len() != conv_width.saturating_mul(4)
        || q.len() != values
        || k.len() != values
        || v.len() != values
        || conv_state.len() != conv_width.saturating_mul(3)
    {
        return Err(Error::Shape {
            label: "Ling 3 contiguous KDA preparation buffers",
            expected: format!(
                "qkv={} weight={} q/k/v={values} state={}",
                rows.saturating_mul(conv_width),
                conv_width.saturating_mul(4),
                conv_width.saturating_mul(3)
            ),
            actual: format!(
                "qkv={} weight={} q={} k={} v={} state={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_prep_rows_on_stream",
            ffi::infer_ling3_kda_prep_rows_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies one-token Qwen3.6 GDN preparation to a changing decode batch.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_prep_batch_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    conv_state_table: &DeviceBuffer<*mut f32>,
    state_table_offset: usize,
    batch_size: usize,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let key_dim = key_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "batched Qwen3.6 GDN prep",
            expected: "key_heads * head_dim without overflow".to_string(),
            actual: format!("key_heads={key_heads} head_dim={head_dim}"),
        })?;
    let value_dim = value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "batched Qwen3.6 GDN prep",
            expected: "value_heads * head_dim without overflow".to_string(),
            actual: format!("value_heads={value_heads} head_dim={head_dim}"),
        })?;
    let conv_dim = key_dim * 2 + value_dim;
    let qkv_len = batch_size
        .checked_mul(conv_dim)
        .ok_or_else(|| Error::Shape {
            label: "batched Qwen3.6 GDN prep",
            expected: "batch_size * conv_dim without overflow".to_string(),
            actual: format!("batch_size={batch_size} conv_dim={conv_dim}"),
        })?;
    let value_len = batch_size
        .checked_mul(value_dim)
        .ok_or_else(|| Error::Shape {
            label: "batched Qwen3.6 GDN prep",
            expected: "batch_size * value_dim without overflow".to_string(),
            actual: format!("batch_size={batch_size} value_dim={value_dim}"),
        })?;
    let state_table_end =
        state_table_offset
            .checked_add(batch_size)
            .ok_or_else(|| Error::Shape {
                label: "batched Qwen3.6 GDN prep state table",
                expected: "state_table_offset + batch_size without overflow".to_string(),
                actual: format!("state_table_offset={state_table_offset} batch_size={batch_size}"),
            })?;
    if batch_size == 0
        || key_heads == 0
        || value_heads == 0
        || head_dim != 128
        || !value_heads.is_multiple_of(key_heads)
        || batch_size > u32::MAX as usize
        || key_heads > u32::MAX as usize
        || value_heads > u32::MAX as usize
        || qkv.len() < qkv_len
        || conv_weight_bf16.len() != conv_dim * 4
        || q.len() < value_len
        || k.len() < value_len
        || v.len() < value_len
        || state_table_end > conv_state_table.len()
    {
        return Err(Error::Shape {
            label: "batched Qwen3.6 GDN prep buffers",
            expected: format!(
                "qkv={qkv_len} conv_weight={} q/k/v={value_len} state_table>={state_table_end}",
                conv_dim * 4
            ),
            actual: format!(
                "qkv={} conv_weight={} q={} k={} v={} state_table={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_prep_batch_on_stream",
            ffi::infer_qwen36_gdn_prep_batch_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state_table.ptr.add(state_table_offset),
                batch_size as u32,
                key_heads as u32,
                value_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies Qwen3.6 convolution/GDN preparation to ragged prompt chunks.
///
/// Each sequence's convolution state is advanced in token order. Dense rows
/// are flattened by sequence and described by device-resident offsets and
/// lengths.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_prep_chunks_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, f32>,
    mut k: DeviceOutput<'_, f32>,
    mut v: DeviceOutput<'_, f32>,
    conv_state_table: &DeviceBuffer<*mut f32>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    sequence_count: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let key_dim = key_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "chunked Qwen3.6 GDN prep",
            expected: "key_heads * head_dim without overflow".to_string(),
            actual: format!("key_heads={key_heads} head_dim={head_dim}"),
        })?;
    let value_dim = value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "chunked Qwen3.6 GDN prep",
            expected: "value_heads * head_dim without overflow".to_string(),
            actual: format!("value_heads={value_heads} head_dim={head_dim}"),
        })?;
    let conv_dim = key_dim * 2 + value_dim;
    let qkv_len = total_tokens
        .checked_mul(conv_dim)
        .ok_or_else(|| Error::Shape {
            label: "chunked Qwen3.6 GDN prep",
            expected: "total_tokens * conv_dim without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} conv_dim={conv_dim}"),
        })?;
    let value_len = total_tokens
        .checked_mul(value_dim)
        .ok_or_else(|| Error::Shape {
            label: "chunked Qwen3.6 GDN prep",
            expected: "total_tokens * value_dim without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} value_dim={value_dim}"),
        })?;
    if sequence_count == 0
        || total_tokens == 0
        || key_heads == 0
        || value_heads == 0
        || head_dim != 128
        || !value_heads.is_multiple_of(key_heads)
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || qkv.len() < qkv_len
        || conv_weight_bf16.len() != conv_dim * 4
        || q.len() < value_len
        || k.len() < value_len
        || v.len() < value_len
        || state_table_offset
            .checked_add(sequence_count)
            .is_none_or(|end| end > conv_state_table.len())
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "chunked Qwen3.6 GDN prep buffers",
            expected: format!(
                "qkv>={qkv_len} conv_weight={} q/k/v>={value_len} metadata/state>={sequence_count}",
                conv_dim * 4
            ),
            actual: format!(
                "qkv={} conv_weight={} q={} k={} v={} state={} offsets={} lengths={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_prep_chunks_on_stream",
            ffi::infer_qwen36_gdn_prep_chunks_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                sequence_count as u32,
                total_tokens as u32,
                key_heads as u32,
                value_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies token-parallel Qwen3.6 convolution preparation directly to BF16 GDN inputs.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_prep_chunks_bf16_into_on_stream(
    qkv: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    mut q: DeviceOutput<'_, u16>,
    mut k: DeviceOutput<'_, u16>,
    mut v: DeviceOutput<'_, u16>,
    conv_state_table: &DeviceBuffer<*mut f32>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    sequence_count: usize,
    total_tokens: usize,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let key_dim = key_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "BF16 chunked Qwen3.6 GDN prep",
            expected: "key_heads * head_dim without overflow".to_string(),
            actual: format!("key_heads={key_heads} head_dim={head_dim}"),
        })?;
    let value_dim = value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Shape {
            label: "BF16 chunked Qwen3.6 GDN prep",
            expected: "value_heads * head_dim without overflow".to_string(),
            actual: format!("value_heads={value_heads} head_dim={head_dim}"),
        })?;
    let conv_dim = key_dim * 2 + value_dim;
    let qkv_len = total_tokens
        .checked_mul(conv_dim)
        .ok_or_else(|| Error::Shape {
            label: "BF16 chunked Qwen3.6 GDN prep",
            expected: "total_tokens * conv_dim without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} conv_dim={conv_dim}"),
        })?;
    let value_len = total_tokens
        .checked_mul(value_dim)
        .ok_or_else(|| Error::Shape {
            label: "BF16 chunked Qwen3.6 GDN prep",
            expected: "total_tokens * value_dim without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} value_dim={value_dim}"),
        })?;
    if sequence_count == 0
        || total_tokens == 0
        || key_heads == 0
        || value_heads == 0
        || head_dim != 128
        || !value_heads.is_multiple_of(key_heads)
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || qkv.len() < qkv_len
        || conv_weight_bf16.len() != conv_dim * 4
        || q.len() < value_len
        || k.len() < value_len
        || v.len() < value_len
        || state_table_offset
            .checked_add(sequence_count)
            .is_none_or(|end| end > conv_state_table.len())
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "BF16 chunked Qwen3.6 GDN prep buffers",
            expected: format!(
                "qkv>={qkv_len} conv_weight={} q/k/v>={value_len} metadata/state>={sequence_count}",
                conv_dim * 4
            ),
            actual: format!(
                "qkv={} conv_weight={} q={} k={} v={} state={} offsets={} lengths={}",
                qkv.len(),
                conv_weight_bf16.len(),
                q.len(),
                k.len(),
                v.len(),
                conv_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_prep_chunks_bf16_on_stream",
            ffi::infer_qwen36_gdn_prep_chunks_bf16_on_stream(
                qkv.ptr,
                conv_weight_bf16.ptr,
                q.buffer_mut().ptr,
                k.buffer_mut().ptr,
                v.buffer_mut().ptr,
                conv_state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                sequence_count as u32,
                total_tokens as u32,
                key_heads as u32,
                value_heads as u32,
                head_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes Qwen3.6 GDN log-decay gate and beta from alpha/beta projections.
pub fn qwen36_gdn_gate_into_on_stream(
    alpha: &DeviceBuffer<f32>,
    beta_input: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    mut gate: DeviceOutput<'_, f32>,
    mut beta: DeviceOutput<'_, f32>,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    if heads == 0
        || heads > u32::MAX as usize
        || alpha.len() != heads
        || beta_input.len() != heads
        || a_log_bf16.len() != heads
        || dt_bias_bf16.len() != heads
        || gate.len() != heads
        || beta.len() != heads
    {
        return Err(Error::Shape {
            label: "Qwen3.6 GDN gate buffers",
            expected: format!("all buffers={heads}"),
            actual: format!(
                "alpha={} beta_input={} a_log={} dt_bias={} gate={} beta={}",
                alpha.len(),
                beta_input.len(),
                a_log_bf16.len(),
                dt_bias_bf16.len(),
                gate.len(),
                beta.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_gate_on_stream",
            ffi::infer_qwen36_gdn_gate_on_stream(
                alpha.ptr,
                beta_input.ptr,
                a_log_bf16.ptr,
                dt_bias_bf16.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes Ling 3's bounded diagonal KDA gate and sigmoid beta.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_gate_f32_into_on_stream(
    raw_gate: &DeviceBuffer<f32>,
    beta_input: &DeviceBuffer<f32>,
    a_log: &DeviceBuffer<f32>,
    dt_bias: &DeviceBuffer<f32>,
    mut gate: DeviceOutput<'_, f32>,
    mut beta: DeviceOutput<'_, f32>,
    heads: usize,
    lower_bound: f32,
    stream: &CudaStream,
) -> Result<()> {
    let vector_len = heads.saturating_mul(128);
    if heads == 0
        || heads > u32::MAX as usize
        || raw_gate.len() != vector_len
        || beta_input.len() != heads
        || a_log.len() != heads
        || dt_bias.len() != vector_len
        || gate.len() != vector_len
        || beta.len() != heads
        || !lower_bound.is_finite()
        || lower_bound >= 0.0
    {
        return Err(Error::Shape {
            label: "Ling 3 KDA gate buffers",
            expected: format!("gate/dt={vector_len} beta/A={heads}, lower_bound<0"),
            actual: format!(
                "raw_gate={} beta_input={} A={} dt={} gate={} beta={} lower_bound={lower_bound}",
                raw_gate.len(),
                beta_input.len(),
                a_log.len(),
                dt_bias.len(),
                gate.len(),
                beta.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_gate_f32_on_stream",
            ffi::infer_ling3_kda_gate_f32_on_stream(
                raw_gate.ptr,
                beta_input.ptr,
                a_log.ptr,
                dt_bias.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                heads as u32,
                lower_bound,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes Ling 3 KDA gates for a dense prompt block.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_gate_f32_batch_into_on_stream(
    raw_gate: &DeviceBuffer<f32>,
    beta_input: &DeviceBuffer<f32>,
    a_log: &DeviceBuffer<f32>,
    dt_bias: &DeviceBuffer<f32>,
    mut gate: DeviceOutput<'_, f32>,
    mut beta: DeviceOutput<'_, f32>,
    rows: usize,
    heads: usize,
    lower_bound: f32,
    stream: &CudaStream,
) -> Result<()> {
    let width = heads.saturating_mul(128);
    let vectors = rows.saturating_mul(width);
    let scalars = rows.saturating_mul(heads);
    if rows == 0
        || heads == 0
        || rows > u32::MAX as usize
        || heads > u32::MAX as usize
        || raw_gate.len() != vectors
        || beta_input.len() != scalars
        || a_log.len() != heads
        || dt_bias.len() != width
        || gate.len() != vectors
        || beta.len() != scalars
        || !lower_bound.is_finite()
        || lower_bound >= 0.0
    {
        return Err(Error::Shape {
            label: "batched Ling 3 KDA gate buffers",
            expected: format!("vectors={vectors} scalars={scalars} heads={heads}"),
            actual: format!(
                "raw_gate={} beta_input={} a_log={} dt_bias={} gate={} beta={} rows={rows} heads={heads}",
                raw_gate.len(),
                beta_input.len(),
                a_log.len(),
                dt_bias.len(),
                gate.len(),
                beta.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_gate_f32_batch_on_stream",
            ffi::infer_ling3_kda_gate_f32_batch_on_stream(
                raw_gate.ptr,
                beta_input.ptr,
                a_log.ptr,
                dt_bias.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                lower_bound,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances one Ling 3 KDA token with `[head,key,value]` FP32 state.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_128_f32_into_on_stream(
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<f32>,
    v: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let vector_len = heads.saturating_mul(128);
    let state_len = vector_len.saturating_mul(128);
    if heads == 0
        || heads > u32::MAX as usize
        || q.len() != vector_len
        || k.len() != vector_len
        || v.len() != vector_len
        || gate.len() != vector_len
        || beta.len() != heads
        || state.len() != state_len
        || output.len() != vector_len
    {
        return Err(Error::Shape {
            label: "Ling 3 KDA buffers",
            expected: format!("q/k/v/g/output={vector_len} beta={heads} state={state_len}"),
            actual: format!(
                "q={} k={} v={} gate={} beta={} state={} output={}",
                q.len(),
                k.len(),
                v.len(),
                gate.len(),
                beta.len(),
                state.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_128_f32_on_stream",
            ffi::infer_ling3_kda_128_f32_on_stream(
                q.ptr,
                k.ptr,
                v.ptr,
                gate.ptr,
                beta.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances one Ling 3 KDA state through a dense prompt block in token order.
#[allow(clippy::too_many_arguments)]
pub fn ling3_kda_128_f32_chunks_into_on_stream(
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<f32>,
    v: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let width = heads.saturating_mul(128);
    let vectors = rows.saturating_mul(width);
    let state_len = width.saturating_mul(128);
    if rows == 0
        || heads == 0
        || rows > u32::MAX as usize
        || heads > u32::MAX as usize
        || q.len() != vectors
        || k.len() != vectors
        || v.len() != vectors
        || gate.len() != vectors
        || beta.len() != rows.saturating_mul(heads)
        || state.len() != state_len
        || output.len() != vectors
    {
        return Err(Error::Shape {
            label: "chunked Ling 3 KDA buffers",
            expected: format!("q/k/v/gate/output={vectors} state={state_len}"),
            actual: format!(
                "q={} k={} v={} gate={} beta={} state={} output={} rows={rows} heads={heads}",
                q.len(),
                k.len(),
                v.len(),
                gate.len(),
                beta.len(),
                state.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_kda_128_f32_chunks_on_stream",
            ffi::infer_ling3_kda_128_f32_chunks_on_stream(
                q.ptr,
                k.ptr,
                v.ptr,
                gate.ptr,
                beta.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies per-row RMSNorm and Ling's sigmoid output gate.
#[allow(clippy::too_many_arguments)]
pub fn ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.saturating_mul(cols);
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() != len
        || gate.len() != len
        || weight.len() != cols
        || output.len() != len
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(Error::Shape {
            label: "Ling 3 sigmoid-gated RMSNorm buffers",
            expected: format!("input/gate/output={len} weight={cols}"),
            actual: format!(
                "input={} gate={} weight={} output={} rows={rows} cols={cols} eps={eps}",
                input.len(),
                gate.len(),
                weight.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_sigmoid_gated_rms_norm_f32_on_stream",
            ffi::infer_ling3_sigmoid_gated_rms_norm_f32_on_stream(
                input.ptr,
                gate.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Packs Ling MLA projections into per-head query, key, and value rows.
#[allow(clippy::too_many_arguments)]
pub fn ling3_mla_pack_f32_into_on_stream(
    query_projection: &DeviceBuffer<f32>,
    kv_projection: &DeviceBuffer<f32>,
    shared_rope_key: &DeviceBuffer<f32>,
    mut query: DeviceOutput<'_, f32>,
    mut key: DeviceOutput<'_, f32>,
    mut value: DeviceOutput<'_, f32>,
    heads: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let qk_dim = qk_nope_dim.saturating_add(rope_dim);
    let qk_len = heads.saturating_mul(qk_dim);
    let value_len = heads.saturating_mul(value_dim);
    let kv_len = heads.saturating_mul(qk_nope_dim.saturating_add(value_dim));
    if heads == 0
        || qk_nope_dim == 0
        || rope_dim == 0
        || value_dim == 0
        || [heads, qk_nope_dim, rope_dim, value_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || query_projection.len() != qk_len
        || kv_projection.len() != kv_len
        || shared_rope_key.len() != rope_dim
        || query.len() != qk_len
        || key.len() != qk_len
        || value.len() != value_len
    {
        return Err(Error::Shape {
            label: "Ling 3 MLA projection packing",
            expected: format!(
                "query={qk_len} kv={kv_len} rope={rope_dim} key={qk_len} value={value_len}"
            ),
            actual: format!(
                "query_projection={} kv_projection={} rope={} query={} key={} value={} heads={heads}",
                query_projection.len(),
                kv_projection.len(),
                shared_rope_key.len(),
                query.len(),
                key.len(),
                value.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_mla_pack_f32_on_stream",
            ffi::infer_ling3_mla_pack_f32_on_stream(
                query_projection.ptr,
                kv_projection.ptr,
                shared_rope_key.ptr,
                query.buffer_mut().ptr,
                key.buffer_mut().ptr,
                value.buffer_mut().ptr,
                heads as u32,
                qk_nope_dim as u32,
                rope_dim as u32,
                value_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Splits batched Ling MLA compressed-KV and shared-RoPE projections.
pub fn ling3_mla_split_kv_a_f32_batch_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut compressed: DeviceOutput<'_, f32>,
    mut rope: DeviceOutput<'_, f32>,
    rows: usize,
    compressed_dim: usize,
    rope_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let input_len = rows.saturating_mul(compressed_dim.saturating_add(rope_dim));
    let compressed_len = rows.saturating_mul(compressed_dim);
    let rope_len = rows.saturating_mul(rope_dim);
    if rows == 0
        || compressed_dim == 0
        || rope_dim == 0
        || [rows, compressed_dim, rope_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || input.len() < input_len
        || compressed.len() < compressed_len
        || rope.len() < rope_len
    {
        return Err(Error::Shape {
            label: "batched Ling 3 MLA KV-A split",
            expected: format!("input>={input_len} compressed>={compressed_len} rope>={rope_len}"),
            actual: format!(
                "input={} compressed={} rope={} rows={rows}",
                input.len(),
                compressed.len(),
                rope.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_mla_split_kv_a_f32_on_stream",
            ffi::infer_ling3_mla_split_kv_a_f32_on_stream(
                input.ptr,
                compressed.buffer_mut().ptr,
                rope.buffer_mut().ptr,
                rows as u32,
                compressed_dim as u32,
                rope_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Packs batched Ling MLA projections into per-head query, key, and value rows.
#[allow(clippy::too_many_arguments)]
pub fn ling3_mla_pack_f32_batch_into_on_stream(
    query_projection: &DeviceBuffer<f32>,
    kv_projection: &DeviceBuffer<f32>,
    shared_rope_key: &DeviceBuffer<f32>,
    mut query: DeviceOutput<'_, f32>,
    mut key: DeviceOutput<'_, f32>,
    mut value: DeviceOutput<'_, f32>,
    rows: usize,
    heads: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    let qk_dim = qk_nope_dim.saturating_add(rope_dim);
    let qk_len = rows.saturating_mul(heads).saturating_mul(qk_dim);
    let value_len = rows.saturating_mul(heads).saturating_mul(value_dim);
    let kv_len = rows
        .saturating_mul(heads)
        .saturating_mul(qk_nope_dim.saturating_add(value_dim));
    let rope_len = rows.saturating_mul(rope_dim);
    if rows == 0
        || heads == 0
        || qk_nope_dim == 0
        || rope_dim == 0
        || value_dim == 0
        || [rows, heads, qk_nope_dim, rope_dim, value_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || query_projection.len() < qk_len
        || kv_projection.len() < kv_len
        || shared_rope_key.len() < rope_len
        || query.len() < qk_len
        || key.len() < qk_len
        || value.len() < value_len
    {
        return Err(Error::Shape {
            label: "batched Ling 3 MLA projection packing",
            expected: format!(
                "query>={qk_len} kv>={kv_len} rope>={rope_len} key>={qk_len} value>={value_len}"
            ),
            actual: format!(
                "query_projection={} kv_projection={} rope={} query={} key={} value={} rows={rows} heads={heads}",
                query_projection.len(),
                kv_projection.len(),
                shared_rope_key.len(),
                query.len(),
                key.len(),
                value.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_mla_pack_f32_batch_on_stream",
            ffi::infer_ling3_mla_pack_f32_batch_on_stream(
                query_projection.ptr,
                kv_projection.ptr,
                shared_rope_key.ptr,
                query.buffer_mut().ptr,
                key.buffer_mut().ptr,
                value.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                qk_nope_dim as u32,
                rope_dim as u32,
                value_dim as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes one-token Ling MLA attention over a contiguous causal KV cache.
#[allow(clippy::too_many_arguments)]
pub fn ling3_mla_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_cache: &DeviceBuffer<f32>,
    value_cache: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    cache_len: usize,
    heads: usize,
    qk_dim: usize,
    value_dim: usize,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let query_len = heads.saturating_mul(qk_dim);
    let output_len = heads.saturating_mul(value_dim);
    let required_keys = cache_len.saturating_mul(query_len);
    let required_values = cache_len.saturating_mul(output_len);
    if cache_len == 0
        || heads == 0
        || qk_dim == 0
        || qk_dim > 512
        || value_dim == 0
        || value_dim > 256
        || [cache_len, heads, qk_dim, value_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || query.len() != query_len
        || key_cache.len() < required_keys
        || value_cache.len() < required_values
        || output.len() != output_len
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Shape {
            label: "Ling 3 MLA attention",
            expected: format!(
                "query={query_len} keys>={required_keys} values>={required_values} output={output_len}"
            ),
            actual: format!(
                "query={} keys={} values={} output={} cache={cache_len} heads={heads} qk={qk_dim} value={value_dim} scale={scale}",
                query.len(),
                key_cache.len(),
                value_cache.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_mla_attention_f32_on_stream",
            ffi::infer_ling3_mla_attention_f32_on_stream(
                query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                output.buffer_mut().ptr,
                cache_len as u32,
                heads as u32,
                qk_dim as u32,
                value_dim as u32,
                scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes one-token Ling MLA attention over physical F32 KV pages.
#[allow(clippy::too_many_arguments)]
pub fn ling3_mla_paged_attention_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_pool: &DeviceBuffer<f32>,
    value_pool: &DeviceBuffer<f32>,
    page_table: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    cache_len: usize,
    page_tokens: usize,
    heads: usize,
    qk_dim: usize,
    value_dim: usize,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let query_len = heads.saturating_mul(qk_dim);
    let output_len = heads.saturating_mul(value_dim);
    let pages = cache_len.div_ceil(page_tokens);
    if cache_len == 0
        || page_tokens == 0
        || heads == 0
        || qk_dim == 0
        || qk_dim > 512
        || value_dim == 0
        || value_dim > 256
        || [cache_len, page_tokens, heads, qk_dim, value_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || query.len() != query_len
        || !key_pool.len().is_multiple_of(page_tokens * query_len)
        || !value_pool.len().is_multiple_of(page_tokens * output_len)
        || page_table.len() < pages
        || output.len() != output_len
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Shape {
            label: "Ling 3 paged MLA attention",
            expected: format!("query={query_len} aligned pools table>={pages} output={output_len}"),
            actual: format!(
                "query={} keys={} values={} table={} output={} cache={cache_len} page={page_tokens}",
                query.len(),
                key_pool.len(),
                value_pool.len(),
                page_table.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_mla_paged_attention_f32_on_stream",
            ffi::infer_ling3_mla_paged_attention_f32_on_stream(
                query.ptr,
                key_pool.ptr,
                value_pool.ptr,
                page_table.ptr,
                output.buffer_mut().ptr,
                cache_len as u32,
                page_tokens as u32,
                heads as u32,
                qk_dim as u32,
                value_dim as u32,
                scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes causal Ling MLA attention for a row chunk over physical F32 KV pages.
#[allow(clippy::too_many_arguments)]
pub fn ling3_mla_paged_causal_rows_f32_into_on_stream(
    query: &DeviceBuffer<f32>,
    key_pool: &DeviceBuffer<f32>,
    value_pool: &DeviceBuffer<f32>,
    page_table: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    start_position: usize,
    rows: usize,
    page_tokens: usize,
    heads: usize,
    qk_dim: usize,
    value_dim: usize,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let qk_width = heads.saturating_mul(qk_dim);
    let value_width = heads.saturating_mul(value_dim);
    let query_len = rows.saturating_mul(qk_width);
    let output_len = rows.saturating_mul(value_width);
    let cache_len = start_position.saturating_add(rows);
    let pages = cache_len.div_ceil(page_tokens.max(1));
    if rows == 0
        || page_tokens == 0
        || heads == 0
        || qk_dim == 0
        || qk_dim > 512
        || value_dim == 0
        || value_dim > 256
        || [start_position, rows, page_tokens, heads, qk_dim, value_dim]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        || query.len() < query_len
        || !key_pool.len().is_multiple_of(page_tokens * qk_width)
        || !value_pool.len().is_multiple_of(page_tokens * value_width)
        || page_table.len() < pages
        || output.len() < output_len
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Shape {
            label: "batched causal Ling 3 paged MLA attention",
            expected: format!(
                "query>={query_len} aligned pools table>={pages} output>={output_len}"
            ),
            actual: format!(
                "query={} keys={} values={} table={} output={} start={start_position} rows={rows} page={page_tokens}",
                query.len(),
                key_pool.len(),
                value_pool.len(),
                page_table.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_ling3_mla_paged_causal_rows_f32_on_stream",
            ffi::infer_ling3_mla_paged_causal_rows_f32_on_stream(
                query.ptr,
                key_pool.ptr,
                value_pool.ptr,
                page_table.ptr,
                output.buffer_mut().ptr,
                start_position as u32,
                rows as u32,
                page_tokens as u32,
                heads as u32,
                qk_dim as u32,
                value_dim as u32,
                scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes Qwen3.6 GDN gates for every row in a decode batch.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_gate_batch_into_on_stream(
    alpha: &DeviceBuffer<f32>,
    beta_input: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    mut gate: DeviceOutput<'_, f32>,
    mut beta: DeviceOutput<'_, f32>,
    batch_size: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = batch_size.checked_mul(heads).ok_or_else(|| Error::Shape {
        label: "batched Qwen3.6 GDN gate",
        expected: "batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    if batch_size == 0
        || heads == 0
        || batch_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || alpha.len() < len
        || beta_input.len() < len
        || gate.len() < len
        || beta.len() < len
        || a_log_bf16.len() != heads
        || dt_bias_bf16.len() != heads
    {
        return Err(Error::Shape {
            label: "batched Qwen3.6 GDN gate buffers",
            expected: format!("alpha/beta_input/gate/beta={len} weights={heads}"),
            actual: format!(
                "alpha={} beta_input={} gate={} beta={} a_log={} dt_bias={}",
                alpha.len(),
                beta_input.len(),
                gate.len(),
                beta.len(),
                a_log_bf16.len(),
                dt_bias_bf16.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_gate_batch_on_stream",
            ffi::infer_qwen36_gdn_gate_batch_on_stream(
                alpha.ptr,
                beta_input.ptr,
                a_log_bf16.ptr,
                dt_bias_bf16.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                batch_size as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes Qwen3.6 GDN gates directly into BF16 chunk-kernel inputs.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_gate_batch_bf16_into_on_stream(
    alpha: &DeviceBuffer<f32>,
    beta_input: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    mut gate: DeviceOutput<'_, u16>,
    mut beta: DeviceOutput<'_, u16>,
    batch_size: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = batch_size.checked_mul(heads).ok_or_else(|| Error::Shape {
        label: "batched BF16 Qwen3.6 GDN gate",
        expected: "batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    if batch_size == 0
        || heads == 0
        || batch_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || alpha.len() < len
        || beta_input.len() < len
        || gate.len() < len
        || beta.len() < len
        || a_log_bf16.len() != heads
        || dt_bias_bf16.len() != heads
    {
        return Err(Error::Shape {
            label: "batched BF16 Qwen3.6 GDN gate buffers",
            expected: format!("alpha/beta_input/gate/beta={len} weights={heads}"),
            actual: format!(
                "alpha={} beta_input={} gate={} beta={} a_log={} dt_bias={}",
                alpha.len(),
                beta_input.len(),
                gate.len(),
                beta.len(),
                a_log_bf16.len(),
                dt_bias_bf16.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_gate_batch_bf16_on_stream",
            ffi::infer_qwen36_gdn_gate_batch_bf16_on_stream(
                alpha.ptr,
                beta_input.ptr,
                a_log_bf16.ptr,
                dt_bias_bf16.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                batch_size as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes batched Qwen3.6 GDN gates from a single `[row, alpha | beta]`
/// projection output.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_gate_paired_batch_into_on_stream(
    alpha_beta: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    mut gate: DeviceOutput<'_, f32>,
    mut beta: DeviceOutput<'_, f32>,
    batch_size: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = batch_size.checked_mul(heads).ok_or_else(|| Error::Shape {
        label: "paired batched Qwen3.6 GDN gate",
        expected: "batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    let paired_len = len.checked_mul(2).ok_or_else(|| Error::Shape {
        label: "paired batched Qwen3.6 GDN gate",
        expected: "2 * batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    if batch_size == 0
        || heads == 0
        || batch_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || alpha_beta.len() < paired_len
        || gate.len() < len
        || beta.len() < len
        || a_log_bf16.len() != heads
        || dt_bias_bf16.len() != heads
    {
        return Err(Error::Shape {
            label: "paired batched Qwen3.6 GDN gate buffers",
            expected: format!("alpha_beta={paired_len} gate/beta={len} weights={heads}"),
            actual: format!(
                "alpha_beta={} gate={} beta={} a_log={} dt_bias={}",
                alpha_beta.len(),
                gate.len(),
                beta.len(),
                a_log_bf16.len(),
                dt_bias_bf16.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_gate_paired_batch_on_stream",
            ffi::infer_qwen36_gdn_gate_paired_batch_on_stream(
                alpha_beta.ptr,
                a_log_bf16.ptr,
                dt_bias_bf16.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                batch_size as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes paired-projection Qwen3.6 GDN gates directly into BF16 chunk inputs.
#[allow(clippy::too_many_arguments)]
pub fn qwen36_gdn_gate_paired_batch_bf16_into_on_stream(
    alpha_beta: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    mut gate: DeviceOutput<'_, u16>,
    mut beta: DeviceOutput<'_, u16>,
    batch_size: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let len = batch_size.checked_mul(heads).ok_or_else(|| Error::Shape {
        label: "paired batched BF16 Qwen3.6 GDN gate",
        expected: "batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    let paired_len = len.checked_mul(2).ok_or_else(|| Error::Shape {
        label: "paired batched BF16 Qwen3.6 GDN gate",
        expected: "2 * batch_size * heads without overflow".to_string(),
        actual: format!("batch_size={batch_size} heads={heads}"),
    })?;
    if batch_size == 0
        || heads == 0
        || batch_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || alpha_beta.len() < paired_len
        || gate.len() < len
        || beta.len() < len
        || a_log_bf16.len() != heads
        || dt_bias_bf16.len() != heads
    {
        return Err(Error::Shape {
            label: "paired batched BF16 Qwen3.6 GDN gate buffers",
            expected: format!("alpha_beta={paired_len} gate/beta={len} weights={heads}"),
            actual: format!(
                "alpha_beta={} gate={} beta={} a_log={} dt_bias={}",
                alpha_beta.len(),
                gate.len(),
                beta.len(),
                a_log_bf16.len(),
                dt_bias_bf16.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen36_gdn_gate_paired_batch_bf16_on_stream",
            ffi::infer_qwen36_gdn_gate_paired_batch_bf16_on_stream(
                alpha_beta.ptr,
                a_log_bf16.ptr,
                dt_bias_bf16.ptr,
                gate.buffer_mut().ptr,
                beta.buffer_mut().ptr,
                batch_size as u32,
                heads as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Enqueues RMSNorm(input) * silu(gate) into `output`.
pub fn gated_rms_norm_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    cols: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "gated RMSNorm",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || rows > u32::MAX as usize
        || cols > u32::MAX as usize
        || input.len() < len
        || gate.len() < len
        || output.len() < len
        || weight.len() != cols
    {
        return Err(Error::Shape {
            label: "gated RMSNorm buffers",
            expected: format!("input/gate/output={len} weight={cols}"),
            actual: format!(
                "input={} gate={} output={} weight={}",
                input.len(),
                gate.len(),
                output.len(),
                weight.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_gated_rms_norm_f32_on_stream",
            ffi::infer_gated_rms_norm_f32_on_stream(
                input.ptr,
                gate.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                cols as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Fuses per-head gated RMSNorm with column-major NVFP4 activation quantization.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
    rows: usize,
    heads: usize,
    head_dim: usize,
    input: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut Nvfp4Matrix,
    eps: f32,
    input_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let cols = heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "gated RMSNorm NVFP4 quantization",
        expected: "heads * head_dim without overflow".to_string(),
        actual: format!("heads={heads} head_dim={head_dim}"),
    })?;
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "gated RMSNorm NVFP4 quantization",
        expected: "rows * heads * head_dim without overflow".to_string(),
        actual: format!("rows={rows} heads={heads} head_dim={head_dim}"),
    })?;
    if rows == 0
        || heads == 0
        || head_dim != 128
        || rows > u32::MAX as usize
        || heads > u32::MAX as usize
        || input.len() < len
        || gate.len() < len
        || weight.len() != head_dim
        || output.rows != cols
        || output.cols < rows
        || !eps.is_finite()
        || eps < 0.0
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return Err(Error::Shape {
            label: "gated RMSNorm NVFP4 quantization buffers",
            expected: format!(
                "input/gate={len} weight={head_dim} output={cols}x{rows} with valid dimensions"
            ),
            actual: format!(
                "input={} gate={} weight={} output={}x{} eps={eps} input_scale={input_scale}",
                input.len(),
                gate.len(),
                weight.len(),
                output.rows,
                output.cols
            ),
        });
    }
    let mut output = output.output();
    unsafe {
        check_cuda(
            "infer_gated_rms_norm_quantize_nvfp4_col_major_f32_on_stream",
            ffi::infer_gated_rms_norm_quantize_nvfp4_col_major_f32_on_stream(
                input.ptr,
                gate.ptr,
                weight.ptr,
                output.values_mut_ptr().cast(),
                output.scales_mut_ptr().cast(),
                rows as u32,
                heads as u32,
                head_dim as u32,
                eps,
                input_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances the one-token causal depthwise convolution in a Nemotron 3 Mamba layer.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_mamba_conv_update_f32_into_on_stream(
    projected: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    conv_bias_bf16: &DeviceBuffer<u16>,
    mut conv_state: DeviceInOut<'_, u16>,
    mut conv_output: DeviceOutput<'_, f32>,
    intermediate_size: usize,
    conv_channels: usize,
    conv_kernel: usize,
    stream: &CudaStream,
) -> Result<()> {
    let projection_size = intermediate_size
        .checked_add(conv_channels)
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 Mamba convolution",
            expected: "intermediate_size + conv_channels without overflow".to_string(),
            actual: format!("intermediate_size={intermediate_size} conv_channels={conv_channels}"),
        })?;
    let state_len = conv_channels
        .checked_mul(conv_kernel)
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 Mamba convolution",
            expected: "conv_channels * conv_kernel without overflow".to_string(),
            actual: format!("conv_channels={conv_channels} conv_kernel={conv_kernel}"),
        })?;
    if intermediate_size == 0
        || conv_channels == 0
        || conv_kernel == 0
        || intermediate_size > u32::MAX as usize
        || conv_channels > u32::MAX as usize
        || conv_kernel > u32::MAX as usize
        || projected.len() < projection_size
        || conv_weight_bf16.len() != state_len
        || conv_bias_bf16.len() != conv_channels
        || conv_state.len() != state_len
        || conv_output.len() != conv_channels
    {
        return Err(Error::Shape {
            label: "Nemotron 3 Mamba convolution buffers",
            expected: format!(
                "projected>={projection_size} weight/state={state_len} bias/output={conv_channels}"
            ),
            actual: format!(
                "projected={} weight={} bias={} state={} output={}",
                projected.len(),
                conv_weight_bf16.len(),
                conv_bias_bf16.len(),
                conv_state.len(),
                conv_output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_mamba_conv_update_f32_on_stream",
            ffi::infer_nemotron3_mamba_conv_update_f32_on_stream(
                projected.ptr,
                conv_weight_bf16.ptr,
                conv_bias_bf16.ptr,
                conv_state.buffer_mut().ptr,
                conv_output.buffer_mut().ptr,
                intermediate_size as u32,
                conv_channels as u32,
                conv_kernel as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances ragged, token-ordered convolution chunks for multiple Nemotron 3
/// sequences. Dense rows are flattened by sequence, and each state-table entry
/// identifies the persistent convolution state for one sequence.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_mamba_conv_update_f32_chunks_into_on_stream(
    projected: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    conv_bias_bf16: &DeviceBuffer<u16>,
    conv_state_table: &DeviceBuffer<*mut u16>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    mut conv_output: DeviceOutput<'_, f32>,
    sequence_count: usize,
    total_tokens: usize,
    intermediate_size: usize,
    conv_channels: usize,
    conv_kernel: usize,
    stream: &CudaStream,
) -> Result<()> {
    let minimum_projection_size =
        intermediate_size
            .checked_add(conv_channels)
            .ok_or_else(|| Error::Shape {
                label: "chunked Nemotron 3 Mamba convolution",
                expected: "intermediate_size + conv_channels without overflow".to_string(),
                actual: format!(
                    "intermediate_size={intermediate_size} conv_channels={conv_channels}"
                ),
            })?;
    let projection_size = projected.len().checked_div(total_tokens).unwrap_or(0);
    let state_len = conv_channels
        .checked_mul(conv_kernel)
        .ok_or_else(|| Error::Shape {
            label: "chunked Nemotron 3 Mamba convolution",
            expected: "conv_channels * conv_kernel without overflow".to_string(),
            actual: format!("conv_channels={conv_channels} conv_kernel={conv_kernel}"),
        })?;
    let output_len = total_tokens
        .checked_mul(conv_channels)
        .ok_or_else(|| Error::Shape {
            label: "chunked Nemotron 3 Mamba convolution",
            expected: "total_tokens * conv_channels without overflow".to_string(),
            actual: format!("total_tokens={total_tokens} conv_channels={conv_channels}"),
        })?;
    let state_table_end = state_table_offset
        .checked_add(sequence_count)
        .ok_or_else(|| Error::Shape {
            label: "chunked Nemotron 3 Mamba convolution state table",
            expected: "state_table_offset + sequence_count without overflow".to_string(),
            actual: format!(
                "state_table_offset={state_table_offset} sequence_count={sequence_count}"
            ),
        })?;
    if sequence_count == 0
        || total_tokens == 0
        || intermediate_size == 0
        || conv_channels == 0
        || conv_kernel == 0
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || projection_size > u32::MAX as usize
        || intermediate_size > u32::MAX as usize
        || conv_channels > u32::MAX as usize
        || conv_kernel > u32::MAX as usize
        || projected.len() != total_tokens.saturating_mul(projection_size)
        || projection_size < minimum_projection_size
        || conv_weight_bf16.len() != state_len
        || conv_bias_bf16.len() != conv_channels
        || conv_output.len() != output_len
        || state_table_end > conv_state_table.len()
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "chunked Nemotron 3 Mamba convolution buffers",
            expected: format!(
                "projected={total_tokens}x>={minimum_projection_size} weight={state_len} bias={conv_channels} output={output_len} metadata/state>={sequence_count}"
            ),
            actual: format!(
                "projected={} projection_size={projection_size} weight={} bias={} output={} state={} offsets={} lengths={}",
                projected.len(),
                conv_weight_bf16.len(),
                conv_bias_bf16.len(),
                conv_output.len(),
                conv_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_mamba_conv_update_f32_chunks_on_stream",
            ffi::infer_nemotron3_mamba_conv_update_f32_chunks_on_stream(
                projected.ptr,
                conv_weight_bf16.ptr,
                conv_bias_bf16.ptr,
                conv_state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                conv_output.buffer_mut().ptr,
                sequence_count as u32,
                projection_size as u32,
                intermediate_size as u32,
                conv_channels as u32,
                conv_kernel as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances ragged Nemotron 3 convolution chunks while recording the initial
/// and per-row recurrent states as BF16 transaction snapshots.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_mamba_conv_update_f32_chunks_snapshot_into_on_stream(
    projected: &DeviceBuffer<f32>,
    conv_weight_bf16: &DeviceBuffer<u16>,
    conv_bias_bf16: &DeviceBuffer<u16>,
    conv_state_table: &DeviceBuffer<*mut u16>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    mut conv_output: DeviceOutput<'_, f32>,
    mut state_snapshots_bf16: DeviceOutput<'_, u16>,
    sequence_count: usize,
    total_tokens: usize,
    snapshot_slots: usize,
    intermediate_size: usize,
    conv_channels: usize,
    conv_kernel: usize,
    stream: &CudaStream,
) -> Result<()> {
    let projection_size = projected.len().checked_div(total_tokens).unwrap_or(0);
    let state_len = conv_channels.saturating_mul(conv_kernel);
    let output_len = total_tokens.saturating_mul(conv_channels);
    let snapshot_len = sequence_count
        .saturating_mul(snapshot_slots)
        .saturating_mul(state_len);
    let state_table_end = state_table_offset.saturating_add(sequence_count);
    if sequence_count == 0
        || total_tokens == 0
        || snapshot_slots == 0
        || intermediate_size == 0
        || conv_channels == 0
        || conv_kernel == 0
        || sequence_count > u32::MAX as usize
        || snapshot_slots > u32::MAX as usize
        || projection_size > u32::MAX as usize
        || intermediate_size > u32::MAX as usize
        || conv_channels > u32::MAX as usize
        || conv_kernel > u32::MAX as usize
        || projected.len() != total_tokens.saturating_mul(projection_size)
        || projection_size < intermediate_size.saturating_add(conv_channels)
        || conv_weight_bf16.len() != state_len
        || conv_bias_bf16.len() != conv_channels
        || conv_output.len() != output_len
        || state_snapshots_bf16.len() != snapshot_len
        || state_table_end > conv_state_table.len()
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "transactional Nemotron 3 Mamba convolution buffers",
            expected: format!(
                "output={output_len} snapshots={snapshot_len} states/metadata>={sequence_count} slots={snapshot_slots}"
            ),
            actual: format!(
                "projected={} weight={} bias={} output={} snapshots={} states={} offsets={} lengths={}",
                projected.len(),
                conv_weight_bf16.len(),
                conv_bias_bf16.len(),
                conv_output.len(),
                state_snapshots_bf16.len(),
                conv_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_mamba_conv_update_f32_chunks_snapshot_on_stream",
            ffi::infer_nemotron3_mamba_conv_update_f32_chunks_snapshot_on_stream(
                projected.ptr,
                conv_weight_bf16.ptr,
                conv_bias_bf16.ptr,
                conv_state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                conv_output.buffer_mut().ptr,
                state_snapshots_bf16.buffer_mut().ptr,
                sequence_count as u32,
                snapshot_slots as u32,
                projection_size as u32,
                intermediate_size as u32,
                conv_channels as u32,
                conv_kernel as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances one token of Nemotron 3 Mamba selective state and applies its
/// gate-before-group-RMSNorm operation.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_mamba_state_update_f32_into_on_stream(
    projected: &DeviceBuffer<f32>,
    conv_output: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    d_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    norm_weight_bf16: &DeviceBuffer<u16>,
    mut ssm_state: DeviceInOut<'_, u16>,
    mut output: DeviceOutput<'_, f32>,
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_size: usize,
    dt_floor: f32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let intermediate_size = heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "Nemotron 3 Mamba state",
        expected: "heads * head_dim without overflow".to_string(),
        actual: format!("heads={heads} head_dim={head_dim}"),
    })?;
    let bc_width = groups.checked_mul(state_size).ok_or_else(|| Error::Shape {
        label: "Nemotron 3 Mamba state",
        expected: "groups * state_size without overflow".to_string(),
        actual: format!("groups={groups} state_size={state_size}"),
    })?;
    let conv_channels = intermediate_size + 2 * bc_width;
    let projection_size = intermediate_size + conv_channels + heads;
    let state_len = intermediate_size
        .checked_mul(state_size)
        .ok_or_else(|| Error::Shape {
            label: "Nemotron 3 Mamba state",
            expected: "intermediate_size * state_size without overflow".to_string(),
            actual: format!("intermediate_size={intermediate_size} state_size={state_size}"),
        })?;
    if heads == 0
        || head_dim == 0
        || groups == 0
        || state_size == 0
        || !heads.is_multiple_of(groups)
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || groups > u32::MAX as usize
        || state_size > u32::MAX as usize
        || !dt_floor.is_finite()
        || dt_floor <= 0.0
        || !eps.is_finite()
        || eps <= 0.0
        || projected.len() != projection_size
        || conv_output.len() != conv_channels
        || a_log_bf16.len() != heads
        || d_bf16.len() != heads
        || dt_bias_bf16.len() != heads
        || norm_weight_bf16.len() != intermediate_size
        || ssm_state.len() != state_len
        || output.len() != intermediate_size
    {
        return Err(Error::Shape {
            label: "Nemotron 3 Mamba state buffers",
            expected: format!(
                "projected={projection_size} conv={conv_channels} head params={heads} norm/output={intermediate_size} state={state_len}"
            ),
            actual: format!(
                "projected={} conv={} a_log={} D={} dt_bias={} norm={} state={} output={}",
                projected.len(),
                conv_output.len(),
                a_log_bf16.len(),
                d_bf16.len(),
                dt_bias_bf16.len(),
                norm_weight_bf16.len(),
                ssm_state.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_mamba_state_update_f32_on_stream",
            ffi::infer_nemotron3_mamba_state_update_f32_on_stream(
                projected.ptr,
                conv_output.ptr,
                a_log_bf16.ptr,
                d_bf16.ptr,
                dt_bias_bf16.ptr,
                norm_weight_bf16.ptr,
                ssm_state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                heads as u32,
                head_dim as u32,
                groups as u32,
                state_size as u32,
                dt_floor,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances ragged, token-ordered selective-state chunks for multiple
/// Nemotron 3 sequences and applies group RMS normalization to every row.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_mamba_state_update_f32_chunks_into_on_stream(
    projected: &DeviceBuffer<f32>,
    conv_output: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    d_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    norm_weight_bf16: &DeviceBuffer<u16>,
    ssm_state_table: &DeviceBuffer<*mut u16>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    sequence_count: usize,
    total_tokens: usize,
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_size: usize,
    dt_floor: f32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let intermediate_size = heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "chunked Nemotron 3 Mamba state",
        expected: "heads * head_dim without overflow".to_string(),
        actual: format!("heads={heads} head_dim={head_dim}"),
    })?;
    let bc_width = groups.checked_mul(state_size).ok_or_else(|| Error::Shape {
        label: "chunked Nemotron 3 Mamba state",
        expected: "groups * state_size without overflow".to_string(),
        actual: format!("groups={groups} state_size={state_size}"),
    })?;
    let conv_channels = intermediate_size + 2 * bc_width;
    let projection_size = intermediate_size + conv_channels + heads;
    let projected_len = total_tokens.saturating_mul(projection_size);
    let conv_len = total_tokens.saturating_mul(conv_channels);
    let output_len = total_tokens.saturating_mul(intermediate_size);
    let state_table_end = state_table_offset.saturating_add(sequence_count);
    if sequence_count == 0
        || total_tokens == 0
        || heads == 0
        || head_dim == 0
        || groups == 0
        || state_size == 0
        || !heads.is_multiple_of(groups)
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || projection_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || groups > u32::MAX as usize
        || state_size > u32::MAX as usize
        || !dt_floor.is_finite()
        || dt_floor <= 0.0
        || !eps.is_finite()
        || eps <= 0.0
        || projected.len() != projected_len
        || conv_output.len() != conv_len
        || a_log_bf16.len() != heads
        || d_bf16.len() != heads
        || dt_bias_bf16.len() != heads
        || norm_weight_bf16.len() != intermediate_size
        || output.len() != output_len
        || state_table_end > ssm_state_table.len()
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "chunked Nemotron 3 Mamba state buffers",
            expected: format!(
                "projected={projected_len} conv={conv_len} head params={heads} norm={intermediate_size} output={output_len} metadata/state>={sequence_count}"
            ),
            actual: format!(
                "projected={} conv={} a_log={} D={} dt_bias={} norm={} output={} state={} offsets={} lengths={}",
                projected.len(),
                conv_output.len(),
                a_log_bf16.len(),
                d_bf16.len(),
                dt_bias_bf16.len(),
                norm_weight_bf16.len(),
                output.len(),
                ssm_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_mamba_state_update_f32_chunks_on_stream",
            ffi::infer_nemotron3_mamba_state_update_f32_chunks_on_stream(
                projected.ptr,
                conv_output.ptr,
                a_log_bf16.ptr,
                d_bf16.ptr,
                dt_bias_bf16.ptr,
                norm_weight_bf16.ptr,
                ssm_state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                output.buffer_mut().ptr,
                sequence_count as u32,
                total_tokens as u32,
                projection_size as u32,
                heads as u32,
                head_dim as u32,
                groups as u32,
                state_size as u32,
                dt_floor,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Advances ragged Nemotron 3 selective-state chunks while recording the
/// initial and per-row recurrent states as BF16 transaction snapshots.
#[allow(clippy::too_many_arguments)]
pub fn nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream(
    projected: &DeviceBuffer<f32>,
    conv_output: &DeviceBuffer<f32>,
    a_log_bf16: &DeviceBuffer<u16>,
    d_bf16: &DeviceBuffer<u16>,
    dt_bias_bf16: &DeviceBuffer<u16>,
    norm_weight_bf16: &DeviceBuffer<u16>,
    ssm_state_table: &DeviceBuffer<*mut u16>,
    state_table_offset: usize,
    sequence_offsets: &DeviceBuffer<u32>,
    sequence_lengths: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    mut state_snapshots_bf16: DeviceOutput<'_, u16>,
    sequence_count: usize,
    total_tokens: usize,
    snapshot_slots: usize,
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_size: usize,
    dt_floor: f32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let intermediate_size = heads.saturating_mul(head_dim);
    let bc_width = groups.saturating_mul(state_size);
    let conv_channels = intermediate_size.saturating_add(2 * bc_width);
    let projection_size = intermediate_size
        .saturating_add(conv_channels)
        .saturating_add(heads);
    let state_len = intermediate_size.saturating_mul(state_size);
    let snapshot_len = sequence_count
        .saturating_mul(snapshot_slots)
        .saturating_mul(state_len);
    let state_table_end = state_table_offset.saturating_add(sequence_count);
    if sequence_count == 0
        || total_tokens == 0
        || snapshot_slots == 0
        || heads == 0
        || head_dim == 0
        || groups == 0
        || state_size == 0
        || !heads.is_multiple_of(groups)
        || sequence_count > u32::MAX as usize
        || total_tokens > u32::MAX as usize
        || snapshot_slots > u32::MAX as usize
        || projection_size > u32::MAX as usize
        || heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || groups > u32::MAX as usize
        || state_size > u32::MAX as usize
        || !dt_floor.is_finite()
        || dt_floor <= 0.0
        || !eps.is_finite()
        || eps <= 0.0
        || projected.len() != total_tokens.saturating_mul(projection_size)
        || conv_output.len() != total_tokens.saturating_mul(conv_channels)
        || a_log_bf16.len() != heads
        || d_bf16.len() != heads
        || dt_bias_bf16.len() != heads
        || norm_weight_bf16.len() != intermediate_size
        || output.len() != total_tokens.saturating_mul(intermediate_size)
        || state_snapshots_bf16.len() != snapshot_len
        || state_table_end > ssm_state_table.len()
        || sequence_offsets.len() < sequence_count
        || sequence_lengths.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "transactional Nemotron 3 Mamba state buffers",
            expected: format!(
                "projected={} conv={} output={} snapshots={snapshot_len} states/metadata>={sequence_count}",
                total_tokens.saturating_mul(projection_size),
                total_tokens.saturating_mul(conv_channels),
                total_tokens.saturating_mul(intermediate_size),
            ),
            actual: format!(
                "projected={} conv={} output={} snapshots={} states={} offsets={} lengths={}",
                projected.len(),
                conv_output.len(),
                output.len(),
                state_snapshots_bf16.len(),
                ssm_state_table.len(),
                sequence_offsets.len(),
                sequence_lengths.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_nemotron3_mamba_state_update_f32_chunks_snapshot_on_stream",
            ffi::infer_nemotron3_mamba_state_update_f32_chunks_snapshot_on_stream(
                projected.ptr,
                conv_output.ptr,
                a_log_bf16.ptr,
                d_bf16.ptr,
                dt_bias_bf16.ptr,
                norm_weight_bf16.ptr,
                ssm_state_table.ptr.add(state_table_offset),
                sequence_offsets.ptr,
                sequence_lengths.ptr,
                output.buffer_mut().ptr,
                state_snapshots_bf16.buffer_mut().ptr,
                sequence_count as u32,
                total_tokens as u32,
                snapshot_slots as u32,
                projection_size as u32,
                heads as u32,
                head_dim as u32,
                groups as u32,
                state_size as u32,
                dt_floor,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Selects one BF16 transaction snapshot per sequence into persistent BF16
/// recurrent-state buffers without staging the states through host memory.
pub fn select_bf16_state_snapshot_into_on_stream(
    state_table: &DeviceBuffer<*mut u16>,
    state_table_offset: usize,
    snapshots_bf16: &DeviceBuffer<u16>,
    selected_slots: &DeviceBuffer<u32>,
    sequence_count: usize,
    snapshot_slots: usize,
    state_size: usize,
    stream: &CudaStream,
) -> Result<()> {
    let state_table_end = state_table_offset.saturating_add(sequence_count);
    let snapshot_len = sequence_count
        .saturating_mul(snapshot_slots)
        .saturating_mul(state_size);
    if sequence_count == 0
        || snapshot_slots == 0
        || state_size == 0
        || sequence_count > u32::MAX as usize
        || snapshot_slots > u32::MAX as usize
        || state_size > u32::MAX as usize
        || state_table_end > state_table.len()
        || snapshots_bf16.len() != snapshot_len
        || selected_slots.len() < sequence_count
    {
        return Err(Error::Shape {
            label: "BF16 recurrent-state snapshot selection buffers",
            expected: format!(
                "states/slots>={sequence_count} snapshots={snapshot_len} slots={snapshot_slots} state_size={state_size}"
            ),
            actual: format!(
                "states={} selected={} snapshots={}",
                state_table.len(),
                selected_slots.len(),
                snapshots_bf16.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_select_bf16_state_snapshot_on_stream",
            ffi::infer_select_bf16_state_snapshot_on_stream(
                state_table.ptr.add(state_table_offset),
                snapshots_bf16.ptr,
                selected_slots.ptr,
                sequence_count as u32,
                snapshot_slots as u32,
                state_size as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{bf16_to_f32, f32_to_bf16};
    use crate::{F32Matrix, synchronize_device};

    #[test]
    fn grammar_mask_applies_independent_rows_and_partial_words() {
        let cols = 35usize;
        let mut host_logits = (0..2 * cols).map(|value| value as f32).collect::<Vec<_>>();
        host_logits[3] = 1000.0;
        host_logits[cols + 34] = 2000.0;
        let mut logits = DeviceBuffer::from_host(&host_logits).expect("logits");
        let allowed = DeviceBuffer::from_host(&[1u32 << 3, 0, 1u32 << 1, 1u32 << (34 - 32)])
            .expect("allowed mask");
        let mut indices = DeviceBuffer::<u32>::zeroed(2).expect("indices");
        let mut values = DeviceBuffer::<f32>::zeroed(2).expect("values");
        let stream = CudaStream::new_non_blocking().expect("stream");

        mask_logits_f32_batch_in_place_on_stream(logits.inout(), &allowed, 2, cols, &stream)
            .expect("mask logits");
        argmax_f32_batch_into_on_stream(
            &logits,
            indices.output(),
            values.output(),
            2,
            cols,
            &stream,
        )
        .expect("masked argmax");

        assert_eq!(
            indices.copy_to_host(&stream).expect("indices download"),
            [3, 34]
        );
        let masked = logits.copy_to_host(&stream).expect("logits download");
        assert!(masked[2].is_infinite() && masked[2].is_sign_negative());
        assert!(masked[cols + 33].is_infinite() && masked[cols + 33].is_sign_negative());
    }

    #[test]
    fn dflash2_capture_interleaves_target_taps_by_row() {
        let input = DeviceBuffer::from_host(&[1.0f32, 2.0, 3.0, 4.0]).expect("input");
        let mut output = DeviceBuffer::from_host(&[-1.0f32; 12]).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        dflash2_capture_f32_into_on_stream(&input, output.output(), 2, 2, 3, 1, &stream)
            .expect("capture tap");
        assert_eq!(
            output.copy_to_host(&stream).expect("captured").as_slice(),
            [
                -1.0, -1.0, 1.0, 2.0, -1.0, -1.0, -1.0, -1.0, 3.0, 4.0, -1.0, -1.0
            ]
        );
    }

    #[test]
    fn dflash2_grouped_convolution_resets_at_each_block() {
        let input =
            DeviceBuffer::from_host(&(1..=24).map(|value| value as f32).collect::<Vec<_>>())
                .expect("input");
        let coefficients = DeviceBuffer::zeroed(6 * 8).expect("coefficients");
        let base = DeviceBuffer::from_host(&[
            1.0f32, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 2.0, 20.0, 20.0, 20.0,
            20.0,
        ])
        .expect("base");
        let mut output = DeviceBuffer::zeroed(24).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        dflash2_grouped_conv_f32_into_on_stream(
            &input,
            &coefficients,
            &base,
            output.output(),
            6,
            4,
            2,
            2,
            3,
            0,
            &stream,
        )
        .expect("grouped convolution");
        assert_eq!(
            output.copy_to_host(&stream).expect("output").as_slice(),
            [
                1.0, 2.0, 3.0, 4.0, 15.0, 26.0, 37.0, 48.0, 59.0, 70.0, 81.0, 92.0, 13.0, 14.0,
                15.0, 16.0, 147.0, 158.0, 169.0, 180.0, 191.0, 202.0, 213.0, 224.0,
            ]
        );
    }

    #[test]
    fn dflash2_attention_reads_future_proposal_rows() {
        let query = DeviceBuffer::from_host(&[1.0f32, 1.0]).expect("query");
        let context_key = DeviceBuffer::zeroed(2).expect("context key");
        let context_value = DeviceBuffer::zeroed(2).expect("context value");
        let block_key = DeviceBuffer::from_host(&[0.0f32, 2.0]).expect("block key");
        let block_value = DeviceBuffer::from_host(&[10.0f32, 20.0]).expect("block value");
        let mut output = DeviceBuffer::zeroed(2).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        dflash2_noncausal_attention_f32_into_on_stream(
            &query,
            &context_key,
            &context_value,
            &block_key,
            &block_value,
            output.output(),
            0,
            0,
            2,
            1,
            1,
            1,
            2,
            &stream,
        )
        .expect("non-causal attention");
        let output = output.copy_to_host(&stream).expect("output");
        assert!(output[0] > 18.0, "future proposal row must be visible");
        assert!((output[0] - output[1]).abs() < 1.0e-6);
    }

    #[test]
    fn dflash2_attention_matches_wrapped_multi_tile_reference() {
        let rows = 3;
        let q_heads = 2;
        let kv_heads = 1;
        let head_dim = 4;
        let window = 320;
        let context_end = 400;
        let context_len = 320;
        let query = (0..rows * q_heads * head_dim)
            .map(|index| ((index * 13 % 29) as f32 - 14.0) / 17.0)
            .collect::<Vec<_>>();
        let mut context_key = vec![0.0f32; window * kv_heads * head_dim];
        let mut context_value = vec![0.0f32; window * kv_heads * head_dim];
        for position in context_end - context_len..context_end {
            for dim in 0..head_dim {
                let index = (position % window) * head_dim + dim;
                context_key[index] = ((position * 7 + dim * 3) % 31) as f32 / 19.0 - 0.7;
                context_value[index] = ((position * 5 + dim * 11) % 37) as f32 / 23.0 - 0.8;
            }
        }
        let block_key = (0..rows * head_dim)
            .map(|index| ((index * 17 % 23) as f32 - 11.0) / 13.0)
            .collect::<Vec<_>>();
        let block_value = (0..rows * head_dim)
            .map(|index| ((index * 19 % 41) as f32 - 20.0) / 21.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; query.len()];
        let context_start = (context_end - context_len).max(context_end + rows - window);
        let scale = 1.0 / (head_dim as f32).sqrt();
        for row in 0..rows {
            for head in 0..q_heads {
                let q = &query
                    [(row * q_heads + head) * head_dim..(row * q_heads + head + 1) * head_dim];
                let mut scores = Vec::with_capacity(window);
                let mut values = Vec::with_capacity(window);
                for position in context_start..context_end {
                    let slot = position % window;
                    let key = &context_key[slot * head_dim..(slot + 1) * head_dim];
                    scores.push(q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale);
                    values.push(&context_value[slot * head_dim..(slot + 1) * head_dim]);
                }
                for block_row in 0..rows {
                    let key = &block_key[block_row * head_dim..(block_row + 1) * head_dim];
                    scores.push(q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale);
                    values.push(&block_value[block_row * head_dim..(block_row + 1) * head_dim]);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let total = weights.iter().sum::<f32>();
                for dim in 0..head_dim {
                    expected[(row * q_heads + head) * head_dim + dim] = weights
                        .iter()
                        .zip(&values)
                        .map(|(weight, value)| weight * value[dim])
                        .sum::<f32>()
                        / total;
                }
            }
        }

        let query_device = DeviceBuffer::from_host(&query).expect("query");
        let context_key_device = DeviceBuffer::from_host(&context_key).expect("context key");
        let context_value_device = DeviceBuffer::from_host(&context_value).expect("context value");
        let block_key_device = DeviceBuffer::from_host(&block_key).expect("block key");
        let block_value_device = DeviceBuffer::from_host(&block_value).expect("block value");
        let mut output = DeviceBuffer::zeroed(query.len()).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        dflash2_noncausal_attention_f32_into_on_stream(
            &query_device,
            &context_key_device,
            &context_value_device,
            &block_key_device,
            &block_value_device,
            output.output(),
            context_end,
            context_len,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            window,
            &stream,
        )
        .expect("multi-tile attention");
        let actual = output.copy_to_host(&stream).expect("attention output");
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2.0e-5,
                "value {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn copies_active_rows_into_interleaved_feature_columns() {
        let input = DeviceBuffer::from_host(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("input");
        let mut output = DeviceBuffer::from_host(&[-1.0f32; 18]).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        copy_f32_rows_into_columns_on_stream(2, 3, 9, 3, &input, output.output(), &stream)
            .expect("column copy");
        assert_eq!(
            output.copy_to_host(&stream).expect("output").as_slice(),
            [
                -1.0, -1.0, -1.0, 1.0, 2.0, 3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.0, 5.0, 6.0,
                -1.0, -1.0, -1.0,
            ]
        );
    }

    #[test]
    fn active_prefix_elementwise_ops_preserve_padding() {
        let left = DeviceBuffer::from_host(&[1.0f32, 2.0, 3.0, 4.0, 5.0]).expect("left");
        let right = DeviceBuffer::from_host(&[0.5f32, 1.0, 1.5, 2.0, 2.5]).expect("right");
        let mut output = DeviceBuffer::from_host(&[99.0f32; 5]).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");

        add_f32_prefix_into_on_stream(&left, &right, output.output(), 3, &stream)
            .expect("prefix add");
        assert_eq!(
            output.copy_to_host(&stream).expect("add output").as_slice(),
            [1.5, 3.0, 4.5, 99.0, 99.0]
        );

        sigmoid_mul_f32_prefix_into_on_stream(&left, &right, output.output(), 3, &stream)
            .expect("prefix sigmoid multiply");
        let actual = output.copy_to_host(&stream).expect("sigmoid output");
        for (actual, (gate, input)) in actual[..3].iter().zip(
            left.copy_to_host(&stream).expect("left copy")[..3]
                .iter()
                .zip(right.copy_to_host(&stream).expect("right copy")[..3].iter()),
        ) {
            let expected = input / (1.0 + (-gate).exp());
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(&actual[3..], [99.0, 99.0]);
        let untouched_active_value = actual[2];
        drop(actual);

        fill_f32_prefix_into_on_stream(output.output(), -7.0, 2, &stream).expect("prefix fill");
        assert_eq!(
            output
                .copy_to_host(&stream)
                .expect("fill output")
                .as_slice(),
            [-7.0, -7.0, untouched_active_value, 99.0, 99.0]
        );

        silu_mul_f32_prefix_into_on_stream(&left, &right, output.output(), 3, &stream)
            .expect("prefix SiLU multiply");
        let actual = output.copy_to_host(&stream).expect("SiLU output");
        for (actual, (gate, input)) in actual[..3].iter().zip(
            left.copy_to_host(&stream).expect("left copy")[..3]
                .iter()
                .zip(right.copy_to_host(&stream).expect("right copy")[..3].iter()),
        ) {
            let expected = gate / (1.0 + (-gate).exp()) * input;
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(&actual[3..], [99.0, 99.0]);

        let gates = DeviceBuffer::from_host(&[-1.0f32, 2.0, 8.0]).expect("head gates");
        let values =
            DeviceBuffer::from_host(&[0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5]).expect("head values");
        let mut scaled = DeviceBuffer::from_host(&[99.0f32; 6]).expect("scaled output");
        softplus_scale_heads_f32_prefix_into_on_stream(
            &gates,
            &values,
            scaled.output(),
            2,
            2,
            &stream,
        )
        .expect("prefix softplus head scale");
        let actual = scaled.copy_to_host(&stream).expect("softplus output");
        let values = values.copy_to_host(&stream).expect("values copy");
        for index in 0..4 {
            let gate = [-1.0f32, 2.0][index / 2];
            let softplus = (1.0 + (-gate.abs()).exp()).ln() + gate.max(0.0);
            assert!((actual[index] - values[index] * softplus).abs() < 1.0e-6);
        }
        assert_eq!(&actual[4..], [99.0, 99.0]);
    }

    #[test]
    fn moe_routes_are_grouped_by_expert_on_device() {
        let indices = DeviceBuffer::from_host(&[3u32, 1, 3, 0, 1, 2, 3, 2]).expect("route indices");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut sorted = MoeSortedRoutes::new(8, 4).expect("sorted routes");
        sorted
            .sort_on_stream(&indices, &stream)
            .expect("sort routes");

        let counts = sorted
            .expert_counts()
            .copy_to_host(&stream)
            .expect("expert counts");
        let offsets = sorted
            .expert_offsets()
            .copy_to_host(&stream)
            .expect("expert offsets");
        let routes = sorted
            .sorted_routes()
            .copy_to_host(&stream)
            .expect("sorted routes");
        let experts = sorted
            .sorted_experts()
            .copy_to_host(&stream)
            .expect("sorted experts");
        let inverse = sorted
            .route_to_sorted()
            .copy_to_host(&stream)
            .expect("inverse routes");

        assert_eq!(counts, [1, 2, 2, 3]);
        assert_eq!(offsets, [0, 1, 3, 5, 8]);
        assert_eq!(experts, [0, 1, 1, 2, 2, 3, 3, 3]);
        let indices = indices.copy_to_host(&stream).expect("indices");
        for (sorted_index, &route) in routes.iter().enumerate() {
            assert_eq!(inverse[route as usize], sorted_index as u32);
            assert_eq!(indices[route as usize], experts[sorted_index]);
        }
    }

    #[test]
    fn sorted_bf16_moe_accumulation_matches_route_order() {
        let rows = 2;
        let routes_per_row = 2;
        let indices = DeviceBuffer::from_host(&[1u32, 0, 1, 0]).expect("route indices");
        let route_weights = [0.25f32, 0.75, 0.4, 0.6];
        let route_weights_device = DeviceBuffer::from_host(&route_weights).expect("route weights");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut routes = MoeSortedRoutes::new(rows * routes_per_row, 2).expect("sorted routes");
        routes
            .sort_on_stream(&indices, &stream)
            .expect("sort routes");
        let sorted_routes = routes
            .sorted_routes()
            .copy_to_host(&stream)
            .expect("sorted route order");

        for features in [5usize, 6] {
            let source = (0..rows * routes_per_row * features)
                .map(|index| (index as f32 - 7.0) * 0.125)
                .collect::<Vec<_>>();
            let mut sorted = Vec::with_capacity(source.len());
            for &route in sorted_routes.iter() {
                let start = route as usize * features;
                sorted.extend(
                    source[start..start + features]
                        .iter()
                        .copied()
                        .map(f32_to_bf16),
                );
            }
            let sorted = DeviceBuffer::from_host(&sorted).expect("sorted inputs");
            let mut output = DeviceBuffer::zeroed(rows * features + 7).expect("output");
            moe_weighted_accumulate_sorted_bf16_batch_on_stream(
                &routes,
                &route_weights_device,
                &sorted,
                output.output(),
                rows,
                routes_per_row,
                features,
                &stream,
            )
            .expect("weighted accumulation");
            let actual = output.copy_to_host(&stream).expect("output download");
            let expected = (0..rows)
                .flat_map(|row| {
                    (0..features).map({
                        let source = &source;
                        move |col| {
                            (0..routes_per_row)
                                .map(|slot| {
                                    let route = row * routes_per_row + slot;
                                    bf16_to_f32(f32_to_bf16(source[route * features + col]))
                                        * route_weights[route]
                                })
                                .sum::<f32>()
                        }
                    })
                })
                .collect::<Vec<_>>();
            assert_close(
                &actual[..rows * features],
                &expected,
                1.0e-6,
                "sorted BF16 weighted accumulation",
            );
        }
    }

    #[test]
    fn speculative_acceptance_stops_at_first_mismatch_and_returns_bonus() {
        const SEQUENCES: usize = 3;
        const DRAFTS: usize = 3;
        const VOCAB: usize = 12;
        let logits = |token: usize| {
            let mut row = vec![-10.0f32; VOCAB];
            row[token] = 10.0;
            row
        };
        let mut previous = Vec::new();
        let mut previous_ptrs = Vec::new();
        for token in [1, 8, 7] {
            let mut row = DeviceBuffer::from_host(&logits(token)).expect("previous logits");
            previous_ptrs.push(row.as_mut_ptr().cast::<f32>().cast_const());
            previous.push(row);
        }
        let mut verification = Vec::new();
        for token in [2, 9, 0, 5, 6, 7, 8, 9, 10] {
            verification.extend(logits(token));
        }
        let drafts = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];
        let previous_ptrs = DeviceBuffer::from_host(&previous_ptrs).expect("logit pointers");
        let verification = DeviceBuffer::from_host(&verification).expect("verification logits");
        let drafts = DeviceBuffer::from_host(&drafts).expect("drafts");
        let mut accepted = DeviceBuffer::zeroed(SEQUENCES).expect("accepted counts");
        let mut next = DeviceBuffer::zeroed(SEQUENCES).expect("next tokens");
        let stream = CudaStream::new_non_blocking().expect("stream");
        speculative_accept_argmax_f32_into_on_stream(
            &previous_ptrs,
            &verification,
            &drafts,
            accepted.output(),
            next.output(),
            SEQUENCES,
            DRAFTS,
            VOCAB,
            &stream,
        )
        .expect("speculative acceptance");
        assert_eq!(
            accepted.copy_to_host(&stream).expect("accepted download"),
            [2, 0, 3]
        );
        assert_eq!(
            next.copy_to_host(&stream).expect("next download"),
            [9, 8, 10]
        );
        drop(previous);
    }

    #[test]
    fn bf16_state_snapshot_selection_stays_on_device() {
        const SEQUENCES: usize = 3;
        const SLOTS: usize = 4;
        const STATE: usize = 5;
        let mut states = (0..SEQUENCES)
            .map(|_| DeviceBuffer::from_host(&[f32_to_bf16(-1.0); STATE]).expect("state"))
            .collect::<Vec<_>>();
        let pointers = states
            .iter_mut()
            .map(|state| state.as_mut_ptr().cast::<u16>())
            .collect::<Vec<_>>();
        let pointers = DeviceBuffer::from_host(&pointers).expect("state pointers");
        let snapshots = (0..SEQUENCES * SLOTS * STATE)
            .map(|index| f32_to_bf16(index as f32 * 0.25))
            .collect::<Vec<_>>();
        let snapshots_device = DeviceBuffer::from_host(&snapshots).expect("snapshots");
        let selected = DeviceBuffer::from_host(&[0u32, 3, 4]).expect("selected slots");
        let stream = CudaStream::new_non_blocking().expect("stream");
        select_bf16_state_snapshot_into_on_stream(
            &pointers,
            0,
            &snapshots_device,
            &selected,
            SEQUENCES,
            SLOTS,
            STATE,
            &stream,
        )
        .expect("state selection");
        for (sequence, state) in states.iter().enumerate() {
            if sequence == 2 {
                assert_eq!(
                    state.copy_to_host(&stream).expect("state download"),
                    [f32_to_bf16(-1.0); STATE]
                );
                continue;
            }
            let slot = if sequence == 0 { 0 } else { 3 };
            let begin = (sequence * SLOTS + slot) * STATE;
            let expected = snapshots[begin..begin + STATE].to_vec();
            assert_eq!(
                state.copy_to_host(&stream).expect("state download"),
                expected
            );
        }
    }

    #[test]
    fn nemotron3_mamba_decode_matches_cpu_reference() {
        const HEADS: usize = 4;
        const HEAD_DIM: usize = 4;
        const GROUPS: usize = 2;
        const STATE_SIZE: usize = 3;
        const CONV_KERNEL: usize = 4;
        const DT_FLOOR: f32 = 1.0e-4;
        const EPS: f32 = 1.0e-5;
        const INTERMEDIATE: usize = HEADS * HEAD_DIM;
        const BC_WIDTH: usize = GROUPS * STATE_SIZE;
        const CONV_CHANNELS: usize = INTERMEDIATE + 2 * BC_WIDTH;
        const PROJECTION: usize = INTERMEDIATE + CONV_CHANNELS + HEADS;

        let mut projected = (0..PROJECTION)
            .map(|index| (index as f32 - 17.0) * 0.017)
            .collect::<Vec<_>>();
        for head in 0..HEADS {
            projected[INTERMEDIATE + CONV_CHANNELS + head] = -0.3 + head as f32 * 0.08;
        }
        let conv_weight = (0..CONV_CHANNELS * CONV_KERNEL)
            .map(|index| f32_to_bf16(((index % 9) as f32 - 4.0) * 0.035))
            .collect::<Vec<_>>();
        let conv_bias = (0..CONV_CHANNELS)
            .map(|index| f32_to_bf16((index as f32 - 8.0) * 0.003))
            .collect::<Vec<_>>();
        let initial_conv_state = (0..CONV_CHANNELS * CONV_KERNEL)
            .map(|index| f32_to_bf16(((index % 13) as f32 - 6.0) * 0.011))
            .collect::<Vec<_>>();

        let mut expected_conv_state = initial_conv_state
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let mut expected_conv = vec![0.0; CONV_CHANNELS];
        for channel in 0..CONV_CHANNELS {
            let state =
                &mut expected_conv_state[channel * CONV_KERNEL..(channel + 1) * CONV_KERNEL];
            state.rotate_left(1);
            state[CONV_KERNEL - 1] = bf16_to_f32(f32_to_bf16(projected[INTERMEDIATE + channel]));
            let mut value = bf16_to_f32(conv_bias[channel]);
            for index in 0..CONV_KERNEL {
                value += state[index] * bf16_to_f32(conv_weight[channel * CONV_KERNEL + index]);
            }
            expected_conv[channel] = value / (1.0 + (-value).exp());
        }

        let stream = CudaStream::new_non_blocking().expect("stream");
        let projected_device = DeviceBuffer::from_host(&projected).expect("projected");
        let conv_weight_device = DeviceBuffer::from_host(&conv_weight).expect("conv weight");
        let conv_bias_device = DeviceBuffer::from_host(&conv_bias).expect("conv bias");
        let mut conv_state_device =
            DeviceBuffer::from_host(&initial_conv_state).expect("conv state");
        let mut conv_output_device = DeviceBuffer::zeroed(CONV_CHANNELS).expect("conv output");
        nemotron3_mamba_conv_update_f32_into_on_stream(
            &projected_device,
            &conv_weight_device,
            &conv_bias_device,
            conv_state_device.inout(),
            conv_output_device.output(),
            INTERMEDIATE,
            CONV_CHANNELS,
            CONV_KERNEL,
            &stream,
        )
        .expect("Mamba convolution");
        let actual_conv = conv_output_device
            .copy_to_host(&stream)
            .expect("conv output download");
        assert_close(&actual_conv, &expected_conv, 2.0e-6, "Mamba convolution");
        assert_eq!(
            conv_state_device
                .copy_to_host(&stream)
                .expect("conv state download"),
            expected_conv_state
                .iter()
                .copied()
                .map(f32_to_bf16)
                .collect::<Vec<_>>()
        );

        let a_log = (0..HEADS)
            .map(|head| f32_to_bf16(-0.2 + head as f32 * 0.07))
            .collect::<Vec<_>>();
        let d = (0..HEADS)
            .map(|head| f32_to_bf16(0.8 + head as f32 * 0.05))
            .collect::<Vec<_>>();
        let dt_bias = (0..HEADS)
            .map(|head| f32_to_bf16(-0.1 + head as f32 * 0.03))
            .collect::<Vec<_>>();
        let norm_weight = (0..INTERMEDIATE)
            .map(|index| f32_to_bf16(0.9 + index as f32 * 0.01))
            .collect::<Vec<_>>();
        let initial_ssm = (0..INTERMEDIATE * STATE_SIZE)
            .map(|index| f32_to_bf16(((index % 11) as f32 - 5.0) * 0.009))
            .collect::<Vec<_>>();
        let mut expected_ssm = initial_ssm
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let mut expected_output = vec![0.0; INTERMEDIATE];
        let heads_per_group = HEADS / GROUPS;
        let group_width = heads_per_group * HEAD_DIM;
        for group in 0..GROUPS {
            for group_index in 0..group_width {
                let flat = group * group_width + group_index;
                let head = flat / HEAD_DIM;
                let raw_dt =
                    projected[INTERMEDIATE + CONV_CHANNELS + head] + bf16_to_f32(dt_bias[head]);
                let dt = (1.0 + raw_dt.exp()).ln().max(DT_FLOOR);
                let decay = (-dt * bf16_to_f32(a_log[head]).exp()).exp();
                let x = actual_conv[flat];
                let b = &actual_conv
                    [INTERMEDIATE + group * STATE_SIZE..INTERMEDIATE + (group + 1) * STATE_SIZE];
                let c_offset = INTERMEDIATE + BC_WIDTH + group * STATE_SIZE;
                let c = &actual_conv[c_offset..c_offset + STATE_SIZE];
                let state = &mut expected_ssm[flat * STATE_SIZE..(flat + 1) * STATE_SIZE];
                let mut y = bf16_to_f32(d[head]) * x;
                for state_index in 0..STATE_SIZE {
                    let updated = state[state_index] * decay + dt * b[state_index] * x;
                    state[state_index] = bf16_to_f32(f32_to_bf16(updated));
                    y += updated * c[state_index];
                }
                let gate = projected[flat];
                expected_output[flat] = y * gate / (1.0 + (-gate).exp());
            }
            let values = &expected_output[group * group_width..(group + 1) * group_width];
            let mean_square =
                values.iter().map(|value| value * value).sum::<f32>() / group_width as f32;
            let inv_rms = (mean_square + EPS).sqrt().recip();
            for group_index in 0..group_width {
                let flat = group * group_width + group_index;
                expected_output[flat] *= inv_rms * bf16_to_f32(norm_weight[flat]);
            }
        }

        let a_log_device = DeviceBuffer::from_host(&a_log).expect("A log");
        let d_device = DeviceBuffer::from_host(&d).expect("D");
        let dt_bias_device = DeviceBuffer::from_host(&dt_bias).expect("dt bias");
        let norm_weight_device = DeviceBuffer::from_host(&norm_weight).expect("norm weight");
        let mut ssm_state_device = DeviceBuffer::from_host(&initial_ssm).expect("SSM state");
        let mut output_device = DeviceBuffer::zeroed(INTERMEDIATE).expect("Mamba output");
        nemotron3_mamba_state_update_f32_into_on_stream(
            &projected_device,
            &conv_output_device,
            &a_log_device,
            &d_device,
            &dt_bias_device,
            &norm_weight_device,
            ssm_state_device.inout(),
            output_device.output(),
            HEADS,
            HEAD_DIM,
            GROUPS,
            STATE_SIZE,
            DT_FLOOR,
            EPS,
            &stream,
        )
        .expect("Mamba state update");
        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("Mamba output download"),
            &expected_output,
            3.0e-5,
            "Mamba output",
        );
        assert_eq!(
            ssm_state_device
                .copy_to_host(&stream)
                .expect("SSM state download"),
            expected_ssm
                .iter()
                .copied()
                .map(f32_to_bf16)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn nemotron3_mamba_chunks_match_repeated_one_token_updates() {
        const HEADS: usize = 4;
        const HEAD_DIM: usize = 16;
        const GROUPS: usize = 2;
        const STATE_SIZE: usize = 40;
        const CONV_KERNEL: usize = 4;
        const INTERMEDIATE: usize = HEADS * HEAD_DIM;
        const BC_WIDTH: usize = GROUPS * STATE_SIZE;
        const CONV_CHANNELS: usize = INTERMEDIATE + 2 * BC_WIDTH;
        const PROJECTION: usize = INTERMEDIATE + CONV_CHANNELS + HEADS;
        const TOTAL_TOKENS: usize = 5;

        let projected = (0..TOTAL_TOKENS * PROJECTION)
            .map(|index| ((index * 17 % 113) as f32 - 56.0) * 0.007)
            .collect::<Vec<_>>();
        let conv_weight = (0..CONV_CHANNELS * CONV_KERNEL)
            .map(|index| f32_to_bf16(((index * 7 % 29) as f32 - 14.0) * 0.019))
            .collect::<Vec<_>>();
        let conv_bias = (0..CONV_CHANNELS)
            .map(|index| f32_to_bf16((index as f32 - 11.0) * 0.002))
            .collect::<Vec<_>>();
        let initial_conv_states = (0..2)
            .map(|sequence| {
                (0..CONV_CHANNELS * CONV_KERNEL)
                    .map(|index| {
                        f32_to_bf16(
                            ((index * 5 % 31) as f32 - 15.0) * 0.004 + sequence as f32 * 0.013,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let offsets = [0u32, 2];
        let lengths = [2u32, 3];
        let stream = CudaStream::new_non_blocking().expect("stream");
        let conv_weight_device = DeviceBuffer::from_host(&conv_weight).expect("conv weight");
        let conv_bias_device = DeviceBuffer::from_host(&conv_bias).expect("conv bias");

        let mut expected_conv_states = initial_conv_states
            .iter()
            .map(|state| DeviceBuffer::from_host(state).expect("expected conv state"))
            .collect::<Vec<_>>();
        let mut expected_conv = vec![0.0f32; TOTAL_TOKENS * CONV_CHANNELS];
        for sequence in 0..2 {
            let begin = offsets[sequence] as usize;
            let end = begin + lengths[sequence] as usize;
            for row in begin..end {
                let row_projected =
                    DeviceBuffer::from_host(&projected[row * PROJECTION..(row + 1) * PROJECTION])
                        .expect("row projection");
                let mut row_output = DeviceBuffer::zeroed(CONV_CHANNELS).expect("row conv output");
                nemotron3_mamba_conv_update_f32_into_on_stream(
                    &row_projected,
                    &conv_weight_device,
                    &conv_bias_device,
                    expected_conv_states[sequence].inout(),
                    row_output.output(),
                    INTERMEDIATE,
                    CONV_CHANNELS,
                    CONV_KERNEL,
                    &stream,
                )
                .expect("sequential convolution");
                expected_conv[row * CONV_CHANNELS..(row + 1) * CONV_CHANNELS].copy_from_slice(
                    &row_output
                        .copy_to_host(&stream)
                        .expect("row conv output copy"),
                );
            }
        }

        let projected_device = DeviceBuffer::from_host(&projected).expect("projected");
        let offsets_device = DeviceBuffer::from_host(&offsets).expect("offsets");
        let lengths_device = DeviceBuffer::from_host(&lengths).expect("lengths");
        let mut actual_conv_states = initial_conv_states
            .iter()
            .map(|state| DeviceBuffer::from_host(state).expect("actual conv state"))
            .collect::<Vec<_>>();
        let conv_state_ptrs = actual_conv_states
            .iter_mut()
            .map(|state| state.as_mut_ptr().cast::<u16>())
            .collect::<Vec<_>>();
        let conv_state_table = DeviceBuffer::from_host(&conv_state_ptrs).expect("conv state table");
        let mut actual_conv =
            DeviceBuffer::zeroed(TOTAL_TOKENS * CONV_CHANNELS).expect("chunk conv output");
        nemotron3_mamba_conv_update_f32_chunks_into_on_stream(
            &projected_device,
            &conv_weight_device,
            &conv_bias_device,
            &conv_state_table,
            0,
            &offsets_device,
            &lengths_device,
            actual_conv.output(),
            2,
            TOTAL_TOKENS,
            INTERMEDIATE,
            CONV_CHANNELS,
            CONV_KERNEL,
            &stream,
        )
        .expect("chunked convolution");
        assert_close(
            &actual_conv
                .copy_to_host(&stream)
                .expect("chunk conv output copy"),
            &expected_conv,
            0.0,
            "chunked convolution output",
        );
        for sequence in 0..2 {
            assert_eq!(
                actual_conv_states[sequence]
                    .copy_to_host(&stream)
                    .expect("actual conv state copy"),
                expected_conv_states[sequence]
                    .copy_to_host(&stream)
                    .expect("expected conv state copy"),
                "chunked convolution state {sequence}",
            );
        }

        let a_log = (0..HEADS)
            .map(|head| f32_to_bf16(-0.3 + head as f32 * 0.04))
            .collect::<Vec<_>>();
        let d = (0..HEADS)
            .map(|head| f32_to_bf16(0.7 + head as f32 * 0.03))
            .collect::<Vec<_>>();
        let dt_bias = (0..HEADS)
            .map(|head| f32_to_bf16(-0.2 + head as f32 * 0.02))
            .collect::<Vec<_>>();
        let norm_weight = (0..INTERMEDIATE)
            .map(|index| f32_to_bf16(0.8 + index as f32 * 0.008))
            .collect::<Vec<_>>();
        let initial_ssm_states = (0..2)
            .map(|sequence| {
                (0..INTERMEDIATE * STATE_SIZE)
                    .map(|index| {
                        f32_to_bf16(
                            ((index * 11 % 37) as f32 - 18.0) * 0.003 + sequence as f32 * 0.009,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let a_log_device = DeviceBuffer::from_host(&a_log).expect("A log");
        let d_device = DeviceBuffer::from_host(&d).expect("D");
        let dt_bias_device = DeviceBuffer::from_host(&dt_bias).expect("dt bias");
        let norm_weight_device = DeviceBuffer::from_host(&norm_weight).expect("norm weight");
        let mut expected_ssm_states = initial_ssm_states
            .iter()
            .map(|state| DeviceBuffer::from_host(state).expect("expected SSM state"))
            .collect::<Vec<_>>();
        let mut expected_output = vec![0.0f32; TOTAL_TOKENS * INTERMEDIATE];
        for sequence in 0..2 {
            let begin = offsets[sequence] as usize;
            let end = begin + lengths[sequence] as usize;
            for row in begin..end {
                let row_projected =
                    DeviceBuffer::from_host(&projected[row * PROJECTION..(row + 1) * PROJECTION])
                        .expect("row projection");
                let row_conv = DeviceBuffer::from_host(
                    &expected_conv[row * CONV_CHANNELS..(row + 1) * CONV_CHANNELS],
                )
                .expect("row convolution");
                let mut row_output = DeviceBuffer::zeroed(INTERMEDIATE).expect("row Mamba output");
                nemotron3_mamba_state_update_f32_into_on_stream(
                    &row_projected,
                    &row_conv,
                    &a_log_device,
                    &d_device,
                    &dt_bias_device,
                    &norm_weight_device,
                    expected_ssm_states[sequence].inout(),
                    row_output.output(),
                    HEADS,
                    HEAD_DIM,
                    GROUPS,
                    STATE_SIZE,
                    1.0e-4,
                    1.0e-5,
                    &stream,
                )
                .expect("sequential state update");
                expected_output[row * INTERMEDIATE..(row + 1) * INTERMEDIATE].copy_from_slice(
                    &row_output
                        .copy_to_host(&stream)
                        .expect("row Mamba output copy"),
                );
            }
        }

        let mut actual_ssm_states = initial_ssm_states
            .iter()
            .map(|state| DeviceBuffer::from_host(state).expect("actual SSM state"))
            .collect::<Vec<_>>();
        let ssm_state_ptrs = actual_ssm_states
            .iter_mut()
            .map(|state| state.as_mut_ptr().cast::<u16>())
            .collect::<Vec<_>>();
        let ssm_state_table = DeviceBuffer::from_host(&ssm_state_ptrs).expect("SSM state table");
        let mut actual_output =
            DeviceBuffer::zeroed(TOTAL_TOKENS * INTERMEDIATE).expect("chunk Mamba output");
        nemotron3_mamba_state_update_f32_chunks_into_on_stream(
            &projected_device,
            &actual_conv,
            &a_log_device,
            &d_device,
            &dt_bias_device,
            &norm_weight_device,
            &ssm_state_table,
            0,
            &offsets_device,
            &lengths_device,
            actual_output.output(),
            2,
            TOTAL_TOKENS,
            HEADS,
            HEAD_DIM,
            GROUPS,
            STATE_SIZE,
            1.0e-4,
            1.0e-5,
            &stream,
        )
        .expect("chunked state update");
        assert_close(
            &actual_output
                .copy_to_host(&stream)
                .expect("chunk Mamba output copy"),
            &expected_output,
            0.0,
            "chunked Mamba output",
        );
        for sequence in 0..2 {
            assert_eq!(
                actual_ssm_states[sequence]
                    .copy_to_host(&stream)
                    .expect("actual SSM state copy"),
                expected_ssm_states[sequence]
                    .copy_to_host(&stream)
                    .expect("expected SSM state copy"),
                "chunked SSM state {sequence}",
            );
        }
    }

    #[test]
    fn nemotron3_sigmoid_topk_matches_grouped_cpu_reference() {
        const EXPERTS: usize = 512;
        const K: usize = 22;
        const GROUPS: usize = 8;
        const TOPK_GROUPS: usize = 3;
        const SCALE: f32 = 5.0;
        let logits = (0..EXPERTS)
            .map(|expert| ((expert * 37 % 101) as f32 - 50.0) * 0.031)
            .collect::<Vec<_>>();
        let bias = (0..EXPERTS)
            .map(|expert| ((expert * 17 % 29) as f32 - 14.0) * 0.004)
            .collect::<Vec<_>>();
        let probabilities = logits
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let scores = probabilities
            .iter()
            .zip(&bias)
            .map(|(probability, bias)| probability + bias)
            .collect::<Vec<_>>();
        let experts_per_group = EXPERTS / GROUPS;
        let mut group_scores = (0..GROUPS)
            .map(|group| {
                let mut values =
                    scores[group * experts_per_group..(group + 1) * experts_per_group].to_vec();
                values.sort_by(|left, right| right.total_cmp(left));
                (group, values[0] + values[1])
            })
            .collect::<Vec<_>>();
        group_scores.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut selected_groups = [false; GROUPS];
        for &(group, _) in &group_scores[..TOPK_GROUPS] {
            selected_groups[group] = true;
        }
        let mut candidates = (0..EXPERTS)
            .filter(|expert| selected_groups[expert / experts_per_group])
            .collect::<Vec<_>>();
        candidates.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        let expected_indices = candidates[..K]
            .iter()
            .map(|&expert| expert as u32)
            .collect::<Vec<_>>();
        let denominator = expected_indices
            .iter()
            .map(|&expert| probabilities[expert as usize])
            .sum::<f32>()
            + 1.0e-20;
        let expected_weights = expected_indices
            .iter()
            .map(|&expert| probabilities[expert as usize] / denominator * SCALE)
            .collect::<Vec<_>>();

        let stream = CudaStream::new_non_blocking().expect("stream");
        let logits = DeviceBuffer::from_host(&logits).expect("logits");
        let bias = DeviceBuffer::from_host(&bias).expect("bias");
        let mut indices = DeviceBuffer::zeroed(K).expect("indices");
        let mut weights = DeviceBuffer::zeroed(K).expect("weights");
        nemotron3_sigmoid_topk_f32_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            K,
            GROUPS,
            TOPK_GROUPS,
            true,
            SCALE,
            &stream,
        )
        .expect("Nemotron router");
        assert_eq!(
            indices.copy_to_host(&stream).expect("indices download"),
            expected_indices
        );
        assert_close(
            &weights.copy_to_host(&stream).expect("weights download"),
            &expected_weights,
            2.0e-6,
            "Nemotron router weights",
        );
    }

    #[test]
    fn nemotron3_sigmoid_topk_batch_matches_independent_rows() {
        const ROWS: usize = 3;
        const EXPERTS: usize = 512;
        const K: usize = 22;
        const GROUPS: usize = 8;
        const TOPK_GROUPS: usize = 3;
        const SCALE: f32 = 5.0;
        let logits = (0..ROWS * EXPERTS)
            .map(|index| {
                let row = index / EXPERTS;
                let expert = index % EXPERTS;
                (((expert * 37 + row * 19) % 101) as f32 - 50.0) * 0.031
            })
            .collect::<Vec<_>>();
        let bias = (0..EXPERTS)
            .map(|expert| ((expert * 17 % 29) as f32 - 14.0) * 0.004)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let logits_device = DeviceBuffer::from_host(&logits).expect("logits");
        let bias_device = DeviceBuffer::from_host(&bias).expect("bias");
        let mut actual_indices = DeviceBuffer::zeroed(ROWS * K).expect("batch indices");
        let mut actual_weights = DeviceBuffer::zeroed(ROWS * K).expect("batch weights");
        nemotron3_sigmoid_topk_f32_batch_into_on_stream(
            &logits_device,
            &bias_device,
            actual_indices.output(),
            actual_weights.output(),
            ROWS,
            K,
            GROUPS,
            TOPK_GROUPS,
            true,
            SCALE,
            &stream,
        )
        .expect("batched Nemotron router");
        let actual_indices = actual_indices
            .copy_to_host(&stream)
            .expect("batch indices download");
        let actual_weights = actual_weights
            .copy_to_host(&stream)
            .expect("batch weights download");

        for row in 0..ROWS {
            let row_logits = DeviceBuffer::from_host(&logits[row * EXPERTS..(row + 1) * EXPERTS])
                .expect("row logits");
            let mut expected_indices = DeviceBuffer::zeroed(K).expect("row indices");
            let mut expected_weights = DeviceBuffer::zeroed(K).expect("row weights");
            nemotron3_sigmoid_topk_f32_into_on_stream(
                &row_logits,
                &bias_device,
                expected_indices.output(),
                expected_weights.output(),
                K,
                GROUPS,
                TOPK_GROUPS,
                true,
                SCALE,
                &stream,
            )
            .expect("independent Nemotron router");
            assert_eq!(
                &actual_indices[row * K..(row + 1) * K],
                &*expected_indices
                    .copy_to_host(&stream)
                    .expect("row indices download")
            );
            assert_close(
                &actual_weights[row * K..(row + 1) * K],
                &expected_weights
                    .copy_to_host(&stream)
                    .expect("row weights download"),
                0.0,
                &format!("batched Nemotron router row {row}"),
            );
        }
    }

    #[test]
    fn relu_squared_matches_cpu_reference() {
        let values = [-3.0, -0.0, 0.25, 2.0, 7.5];
        let expected = [0.0, 0.0, 0.0625, 4.0, 56.25];
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&values).expect("input");
        let mut output = DeviceBuffer::zeroed(values.len()).expect("output");
        relu_squared_f32_into_on_stream(&input, output.output(), &stream).expect("ReLU squared");
        assert_eq!(
            output.copy_to_host(&stream).expect("output download"),
            expected
        );
    }

    #[test]
    fn clamped_silu_halves_matches_step_reference_for_single_and_batch_rows() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let limit = 7.0f32;
        let rows = [
            10.0f32, -10.0, 20.0, -20.0, 2.0, 8.0, -3.0, 9.0, 91.0, 92.0, 93.0, 94.0,
        ];
        let input = DeviceBuffer::from_host(&rows).expect("gate/up");
        let mut batch =
            DeviceBuffer::from_host(&[0.0, 0.0, 0.0, 0.0, 101.0, 102.0]).expect("batch output");
        silu_mul_halves_clamped_f32_batch_into_on_stream(
            &input,
            batch.output(),
            2,
            2,
            limit,
            &stream,
        )
        .expect("batched clamped SwiGLU");

        let first = DeviceBuffer::from_host(&rows[..4]).expect("first row");
        let mut single = DeviceBuffer::zeroed(2).expect("single output");
        silu_mul_halves_clamped_f32_into_on_stream(&first, single.output(), 2, limit, &stream)
            .expect("single clamped SwiGLU");

        let expected = [[10.0f32, -10.0, 20.0, -20.0], [2.0, 8.0, -3.0, 9.0]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let expected = expected
            .chunks_exact(4)
            .flat_map(|row| {
                (0..2).map(|column| {
                    let gate = row[column].min(limit);
                    let up = row[2 + column].clamp(-limit, limit);
                    (gate / (1.0 + (-gate).exp())) * up
                })
            })
            .collect::<Vec<_>>();
        let actual = batch.copy_to_host(&stream).expect("batch readback");
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
        assert_eq!(&actual[4..], [101.0, 102.0]);
        assert_eq!(
            single.copy_to_host(&stream).expect("single readback"),
            actual[..2]
        );
    }

    #[test]
    fn expert_indices_remap_through_device_slot_table() {
        let indices = DeviceBuffer::from_host(&[3u32, 0, 4, 9]).expect("indices");
        let map = DeviceBuffer::from_host(&[7u32, u32::MAX, 2, 5, 1]).expect("map");
        let mut slots = DeviceBuffer::zeroed(4).expect("slots");
        let stream = CudaStream::new_blocking().expect("stream");
        remap_expert_indices_into_on_stream(&indices, &map, slots.output(), &stream)
            .expect("remap");
        assert_eq!(
            slots.copy_to_host(&stream).expect("readback"),
            [5, 7, 1, u32::MAX]
        );
    }

    #[test]
    fn expert_indices_remap_accepts_source_offset() {
        let indices = DeviceBuffer::from_host(&[99u32, 3, 0, 4, 9, 99]).expect("indices");
        let map = DeviceBuffer::from_host(&[7u32, u32::MAX, 2, 5, 1]).expect("map");
        let mut slots = DeviceBuffer::zeroed(4).expect("slots");
        let stream = CudaStream::new_blocking().expect("stream");
        remap_expert_indices_at_offset_into_on_stream(&indices, 1, &map, slots.output(), &stream)
            .expect("offset remap");
        assert_eq!(
            slots.copy_to_host(&stream).expect("readback"),
            [5, 7, 1, u32::MAX]
        );
    }

    #[test]
    fn expert_usage_histogram_accumulates_and_clears_on_device() {
        let first = DeviceBuffer::from_host(&[3u32, 0, 3, 7, 1]).expect("first indices");
        let second = DeviceBuffer::from_host(&[1u32, 3, 2]).expect("second indices");
        let mut counts = DeviceBuffer::zeroed(4).expect("counts");
        let stream = CudaStream::new_blocking().expect("stream");
        record_expert_indices_u64_on_stream(&first, counts.inout(), &stream).expect("record first");
        record_expert_indices_u64_on_stream(&second, counts.inout(), &stream)
            .expect("record second");
        assert_eq!(
            counts.copy_to_host(&stream).expect("readback"),
            [1, 2, 1, 3]
        );
        clear_expert_counts_u64_on_stream(counts.output(), &stream).expect("clear counts");
        assert_eq!(
            counts.copy_to_host(&stream).expect("cleared readback"),
            [0, 0, 0, 0]
        );
    }

    #[test]
    fn indexed_f32_gather_multiply_stays_on_device() {
        let values = DeviceBuffer::from_host(&[0.25f32, -2.0, 4.0, 8.0]).expect("values");
        let indices = DeviceBuffer::from_host(&[2u32, 1, 8, 0]).expect("indices");
        let multipliers = DeviceBuffer::from_host(&[0.5f32, -1.0, 3.0, 4.0]).expect("multipliers");
        let mut output = DeviceBuffer::zeroed(4).expect("output");
        let stream = CudaStream::new_blocking().expect("stream");
        gather_indexed_mul_f32_into_on_stream(
            &values,
            &indices,
            &multipliers,
            output.output(),
            &stream,
        )
        .expect("indexed gather multiply");
        assert_eq!(
            output.copy_to_host(&stream).expect("readback"),
            [2.0, 2.0, 0.0, 1.0]
        );
    }

    #[test]
    fn gpu_token_sampler_keeps_logits_on_device_and_applies_penalties() {
        let vocab = 64usize;
        let mut logits = vec![-20.0f32; 2 * vocab];
        logits[1] = 4.0;
        logits[2] = 5.0;
        logits[vocab + 7] = 3.0;
        logits[vocab + 9] = 2.0;
        logits[vocab + 11] = 1.0;
        let logits = DeviceBuffer::from_host(&logits).expect("logits");
        let mut counts = vec![0u32; vocab];
        counts[2] = 2;
        let mut counts = DeviceBuffer::from_host(&counts).expect("counts");
        let mut sampler = GpuTokenSampler::new(2, vocab).expect("sampler");
        let stream = CudaStream::new_blocking().expect("stream");
        let sampled = sampler
            .sample(
                &logits,
                &mut [
                    GpuSamplingRow {
                        temperature: 0.0,
                        top_k: 20,
                        top_p: 0.95,
                        presence_penalty: 1.0,
                        frequency_penalty: 1.0,
                        draw: 0.25,
                        token_counts: Some(&mut counts),
                    },
                    GpuSamplingRow {
                        temperature: 1.0,
                        top_k: 3,
                        top_p: 1.0,
                        presence_penalty: 0.0,
                        frequency_penalty: 0.0,
                        draw: 0.999,
                        token_counts: None,
                    },
                ],
                vocab,
                &stream,
            )
            .expect("sample");
        assert_eq!(sampled[0].id, 1);
        assert_eq!(sampled[0].logit, 4.0);
        assert_eq!(sampled[0].adjusted_logit, 4.0);
        assert_eq!(sampled[1].id, 11);
        assert_eq!(sampled[1].logit, 1.0);
        let counts = counts.copy_to_host(&stream).expect("counts readback");
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 2);
    }

    #[test]
    fn gpu_token_sampler_reduces_candidates_across_multiple_stages() {
        let vocab = 35_000usize;
        let mut logits = vec![-100.0f32; 2 * vocab];
        let candidate_ids = (0..32).map(|slot| 17 + slot * 1_051).collect::<Vec<_>>();
        for (slot, &id) in candidate_ids.iter().enumerate() {
            logits[id] = 10.0 - slot as f32 * 0.25;
        }
        logits[vocab + 9_001] = 7.0;
        logits[vocab + 2_001] = 7.0;
        logits[vocab + 34_001] = 6.0;

        let logits = DeviceBuffer::from_host(&logits).expect("logits");
        let mut sampler = GpuTokenSampler::new(2, vocab).expect("sampler");
        let stream = CudaStream::new_blocking().expect("stream");
        let draw = 0.73f32;
        let sampled = sampler
            .sample(
                &logits,
                &mut [
                    GpuSamplingRow {
                        temperature: 0.8,
                        top_k: 32,
                        top_p: 0.83,
                        presence_penalty: 0.0,
                        frequency_penalty: 0.0,
                        draw,
                        token_counts: None,
                    },
                    GpuSamplingRow {
                        temperature: 0.0,
                        top_k: 0,
                        top_p: 1.0,
                        presence_penalty: 0.0,
                        frequency_penalty: 0.0,
                        draw: 0.0,
                        token_counts: None,
                    },
                ],
                vocab,
                &stream,
            )
            .expect("sample");

        let values = candidate_ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id as u32, 10.0 - slot as f32 * 0.25))
            .collect::<Vec<_>>();
        let weights = values
            .iter()
            .map(|(_, value)| ((*value - values[0].1) / 0.8).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        let retained = weights
            .iter()
            .scan(0.0, |sum, weight| {
                *sum += weight / total;
                Some(*sum)
            })
            .position(|sum| sum >= 0.83)
            .map_or(weights.len(), |slot| slot + 1);
        let retained_weight = weights[..retained].iter().sum::<f32>();
        let mut target = draw * retained_weight;
        let expected = weights[..retained]
            .iter()
            .position(|weight| {
                if target < *weight {
                    true
                } else {
                    target -= *weight;
                    false
                }
            })
            .unwrap_or(retained - 1);

        assert_eq!(sampled[0].id, values[expected].0);
        assert_eq!(sampled[0].logit, values[expected].1);
        assert_eq!(sampled[1].id, 2_001, "equal logits prefer the lower ID");
    }

    #[test]
    fn rms_norm_f32_matches_cpu_reference() {
        let rows = 3;
        let eps = 1.0e-6;
        for cols in [128, 1536] {
            let input = (0..rows * cols)
                .map(|idx| {
                    if cols == 1536 {
                        ((idx % 97) as f32 - 48.0) / 96.0
                    } else {
                        ((idx % 19) as f32 - 9.0) * 0.125
                    }
                })
                .collect::<Vec<_>>();
            let weight = (0..cols)
                .map(|idx| 0.5 + (idx % 7) as f32 * 0.03125)
                .collect::<Vec<_>>();

            let input_device = DeviceBuffer::from_host(&input).expect("input upload");
            let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
            let mut output_device =
                DeviceBuffer::zeroed(rows * cols).expect("RMSNorm output alloc");
            let stream = CudaStream::new_non_blocking().expect("stream");
            rms_norm_f32_into_on_stream(
                rows,
                cols,
                &input_device,
                &weight_device,
                output_device.output(),
                eps,
                &stream,
            )
            .expect("RMSNorm launch");
            let output = output_device
                .copy_to_host(&stream)
                .expect("RMSNorm download");

            let expected = cpu_rms_norm(rows, cols, &input, &weight, eps);
            for (idx, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
                let error = (actual - expected).abs();
                assert!(
                    error <= 1.0e-5,
                    "RMSNorm {cols} mismatch at {idx}: actual={actual} expected={expected} error={error}"
                );
            }
        }
    }

    #[test]
    fn silu_mul_f32_matches_cpu_reference() {
        let gate = (0..257)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.25)
            .collect::<Vec<_>>();
        let up = (0..257)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();

        let gate_device = DeviceBuffer::from_host(&gate).expect("gate upload");
        let up_device = DeviceBuffer::from_host(&up).expect("up upload");
        let mut output_device = DeviceBuffer::zeroed(gate.len()).expect("SiLU output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        silu_mul_f32_into_on_stream(&gate_device, &up_device, output_device.output(), &stream)
            .expect("SiLU multiply launch");
        let output = output_device
            .copy_to_host(&stream)
            .expect("SiLU multiply download");

        for (idx, ((actual, gate), up)) in output.iter().zip(gate.iter()).zip(up.iter()).enumerate()
        {
            let expected = gate * (1.0 / (1.0 + (-gate).exp())) * up;
            let error = (actual - expected).abs();
            assert!(
                error <= 1.0e-6,
                "SiLU multiply mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
    }

    #[test]
    fn gelu_tanh_f32_matches_cpu_reference() {
        let input = (0..513)
            .map(|idx| ((idx % 47) as f32 - 23.0) * 0.125)
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let mut output_device = DeviceBuffer::zeroed(input.len()).expect("GELU output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        gelu_tanh_f32_into_on_stream(&input_device, output_device.output(), &stream)
            .expect("GELU launch");
        let output = output_device.copy_to_host(&stream).expect("GELU download");

        for (idx, (actual, input)) in output.iter().zip(input.iter()).enumerate() {
            let expected = 0.5
                * input
                * (1.0 + (0.797_884_6 * (input + 0.044715 * input * input * input)).tanh());
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "GELU mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn gelu_tanh_mul_f32_matches_cpu_reference() {
        let gate = (0..513)
            .map(|idx| ((idx % 47) as f32 - 23.0) * 0.125)
            .collect::<Vec<_>>();
        let up = (0..513)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.0625)
            .collect::<Vec<_>>();
        let gate_device = DeviceBuffer::from_host(&gate).expect("gate upload");
        let up_device = DeviceBuffer::from_host(&up).expect("up upload");
        let mut output_device = DeviceBuffer::zeroed(gate.len()).expect("GELU output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        gelu_tanh_mul_f32_into_on_stream(&gate_device, &up_device, output_device.output(), &stream)
            .expect("GELU multiply launch");
        let output = output_device
            .copy_to_host(&stream)
            .expect("GELU multiply download");

        for idx in 0..gate.len() {
            let value = gate[idx];
            let expected = 0.5
                * value
                * (1.0 + (0.797_884_6 * (value + 0.044715 * value * value * value)).tanh())
                * up[idx];
            assert!(
                (output[idx] - expected).abs() <= 2.0e-5,
                "GELU multiply mismatch at {idx}: actual={} expected={expected}",
                output[idx]
            );
        }
    }

    #[test]
    fn gelu_tanh_mul_halves_matches_cpu_reference() {
        let len = 513;
        let gate_up = (0..len * 2)
            .map(|idx| ((idx % 53) as f32 - 26.0) * 0.125)
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&gate_up).expect("input upload");
        let mut output_device = DeviceBuffer::zeroed(len).expect("GELU output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        gelu_tanh_mul_halves_f32_into_on_stream(
            &input_device,
            output_device.output(),
            len,
            &stream,
        )
        .expect("GELU multiply launch");
        let output = output_device.copy_to_host(&stream).expect("GELU download");

        for idx in 0..len {
            let gate = gate_up[idx];
            let expected = 0.5
                * gate
                * (1.0 + (0.797_884_6 * (gate + 0.044715 * gate * gate * gate)).tanh())
                * gate_up[len + idx];
            assert!(
                (output[idx] - expected).abs() <= 2.0e-5,
                "GELU multiply mismatch at {idx}: actual={} expected={expected}",
                output[idx]
            );
        }
    }

    #[test]
    fn split_q_gate_f32_matches_cpu_reference() {
        let len = 257usize;
        let input = (0..len * 2)
            .map(|idx| ((idx % 43) as f32 - 21.0) * 0.03125)
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let mut q_device = DeviceBuffer::<f32>::zeroed(len).expect("q alloc");
        let mut gate_device = DeviceBuffer::<f32>::zeroed(len).expect("gate alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        split_q_gate_f32_into_on_stream(
            &input_device,
            q_device.output(),
            gate_device.output(),
            &stream,
        )
        .expect("split q/gate enqueue");

        assert_close(
            &q_device.copy_to_host(&stream).expect("q download"),
            &input[..len],
            0.0,
            "split q",
        );
        assert_close(
            &gate_device.copy_to_host(&stream).expect("gate download"),
            &input[len..],
            0.0,
            "split gate",
        );
    }

    #[test]
    fn sigmoid_mul_f32_matches_cpu_reference() {
        let len = 257usize;
        let gate = (0..len)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.25)
            .collect::<Vec<_>>();
        let input = (0..len)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let expected = gate
            .iter()
            .zip(input.iter())
            .map(|(gate, input)| input * (1.0 / (1.0 + (-gate).exp())))
            .collect::<Vec<_>>();

        let gate_device = DeviceBuffer::from_host(&gate).expect("gate upload");
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(len).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        sigmoid_mul_f32_into_on_stream(
            &gate_device,
            &input_device,
            output_device.output(),
            &stream,
        )
        .expect("sigmoid multiply enqueue");

        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("output download"),
            &expected,
            1.0e-6,
            "sigmoid multiply",
        );
    }

    #[test]
    fn sigmoid_scale_scalar_f32_matches_cpu_reference() {
        let len = 257usize;
        let gate_scalar = -2.675996f32;
        let input = (0..len)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let sigmoid = 1.0 / (1.0 + (-gate_scalar).exp());
        let expected: Vec<f32> = input.iter().map(|x| x * sigmoid).collect();

        let gate_device = DeviceBuffer::from_host(&[gate_scalar]).expect("gate upload");
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(len).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        sigmoid_scale_scalar_f32_into_on_stream(
            &gate_device,
            &input_device,
            output_device.output(),
            &stream,
        )
        .expect("sigmoid scale enqueue");

        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("output download"),
            &expected,
            1.0e-6,
            "sigmoid scale scalar",
        );
    }

    #[test]
    fn softplus_scale_heads_f32_matches_cpu_reference() {
        let heads = 3usize;
        let head_dim = 5usize;
        let gate = [-4.0f32, 0.0, 3.0];
        let input = (0..heads * head_dim)
            .map(|idx| idx as f32 * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let expected = input
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let value = gate[idx / head_dim];
                let softplus = (1.0 + (-value.abs()).exp()).ln() + value.max(0.0);
                input * softplus
            })
            .collect::<Vec<_>>();
        let gate_device = DeviceBuffer::from_host(&gate).expect("gate upload");
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let mut output_device = DeviceBuffer::zeroed(input.len()).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        softplus_scale_heads_f32_into_on_stream(
            &gate_device,
            &input_device,
            output_device.output(),
            head_dim,
            &stream,
        )
        .expect("softplus scale enqueue");

        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("output download"),
            &expected,
            1.0e-6,
            "softplus scale heads",
        );
    }

    #[test]
    fn qwen36_full_attn_prep_f32_matches_cpu_reference() {
        let q_heads = 3usize;
        let kv_heads = 2usize;
        let head_dim = 8usize;
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let q_full = (0..q_width * 2)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.03125)
            .collect::<Vec<_>>();
        let k_raw = (0..kv_width)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.025)
            .collect::<Vec<_>>();
        let q_norm = (0..head_dim)
            .map(|idx| 0.75 + idx as f32 * 0.03125)
            .collect::<Vec<_>>();
        let k_norm = (0..head_dim)
            .map(|idx| 1.25 - idx as f32 * 0.025)
            .collect::<Vec<_>>();
        let (expected_q, expected_gate, expected_k) = cpu_qwen36_full_attn_prep(
            &q_full, &k_raw, &q_norm, &k_norm, q_heads, kv_heads, head_dim, 1.0e-6,
        );

        let q_full_device = DeviceBuffer::from_host(&q_full).expect("q full upload");
        let k_raw_device = DeviceBuffer::from_host(&k_raw).expect("k upload");
        let q_norm_device = DeviceBuffer::from_host(&q_norm).expect("q norm upload");
        let k_norm_device = DeviceBuffer::from_host(&k_norm).expect("k norm upload");
        let mut q_device = DeviceBuffer::<f32>::zeroed(q_width).expect("q alloc");
        let mut gate_device = DeviceBuffer::<f32>::zeroed(q_width).expect("gate alloc");
        let mut k_device = DeviceBuffer::<f32>::zeroed(kv_width).expect("k alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        qwen36_full_attn_prep_f32_into_on_stream(
            &q_full_device,
            &k_raw_device,
            &q_norm_device,
            &k_norm_device,
            q_device.output(),
            gate_device.output(),
            k_device.output(),
            q_heads,
            kv_heads,
            head_dim,
            1.0e-6,
            &stream,
        )
        .expect("full-attn prep enqueue");

        assert_close(
            &q_device.copy_to_host(&stream).expect("q download"),
            &expected_q,
            1.0e-6,
            "full-attn q",
        );
        assert_close(
            &gate_device.copy_to_host(&stream).expect("gate download"),
            &expected_gate,
            0.0,
            "full-attn gate",
        );
        assert_close(
            &k_device.copy_to_host(&stream).expect("k download"),
            &expected_k,
            1.0e-6,
            "full-attn k",
        );
    }

    #[test]
    fn qwen36_full_attn_prep_batch_matches_independent_rows() {
        let batch = 2usize;
        let q_heads = 3usize;
        let kv_heads = 2usize;
        let head_dim = 8usize;
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let q_full = (0..batch * q_width * 2)
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.03125)
            .collect::<Vec<_>>();
        let k_raw = (0..batch * kv_width)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.025)
            .collect::<Vec<_>>();
        let q_norm = (0..head_dim)
            .map(|idx| 0.75 + idx as f32 * 0.03125)
            .collect::<Vec<_>>();
        let k_norm = (0..head_dim)
            .map(|idx| 1.25 - idx as f32 * 0.025)
            .collect::<Vec<_>>();
        let q_full_device = DeviceBuffer::from_host(&q_full).expect("q full upload");
        let k_raw_device = DeviceBuffer::from_host(&k_raw).expect("k upload");
        let q_norm_device = DeviceBuffer::from_host(&q_norm).expect("q norm upload");
        let k_norm_device = DeviceBuffer::from_host(&k_norm).expect("k norm upload");
        let mut q_device = DeviceBuffer::<f32>::zeroed(batch * q_width).expect("q alloc");
        let mut gate_device = DeviceBuffer::<f32>::zeroed(batch * q_width).expect("gate alloc");
        let mut k_device = DeviceBuffer::<f32>::zeroed(batch * kv_width).expect("k alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        qwen36_full_attn_prep_f32_batch_into_on_stream(
            &q_full_device,
            &k_raw_device,
            &q_norm_device,
            &k_norm_device,
            q_device.output(),
            gate_device.output(),
            k_device.output(),
            batch,
            q_heads,
            kv_heads,
            head_dim,
            1.0e-6,
            &stream,
        )
        .expect("batch full-attn prep");
        let actual_q = q_device.copy_to_host(&stream).expect("q download");
        let actual_gate = gate_device.copy_to_host(&stream).expect("gate download");
        let actual_k = k_device.copy_to_host(&stream).expect("k download");
        for row in 0..batch {
            let q_range = row * q_width..(row + 1) * q_width;
            let q_full_range = row * q_width * 2..(row + 1) * q_width * 2;
            let k_range = row * kv_width..(row + 1) * kv_width;
            let (expected_q, expected_gate, expected_k) = cpu_qwen36_full_attn_prep(
                &q_full[q_full_range],
                &k_raw[k_range.clone()],
                &q_norm,
                &k_norm,
                q_heads,
                kv_heads,
                head_dim,
                1.0e-6,
            );
            assert_close(&actual_q[q_range.clone()], &expected_q, 1.0e-6, "batch q");
            assert_close(&actual_gate[q_range], &expected_gate, 0.0, "batch gate");
            assert_close(&actual_k[k_range], &expected_k, 1.0e-6, "batch k");
        }
    }

    #[test]
    fn moe_topk_f32_matches_cpu_reference() {
        let logits = [0.0f32, 3.0, 1.0, 2.0];
        let logits_device = DeviceBuffer::from_host(&logits).expect("logits upload");
        let mut indices_device = DeviceBuffer::<u32>::zeroed(2).expect("indices alloc");
        let mut weights_device = DeviceBuffer::<f32>::zeroed(2).expect("weights alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        moe_topk_f32_into_on_stream(
            &logits_device,
            indices_device.output(),
            weights_device.output(),
            2,
            true,
            &stream,
        )
        .expect("top-k launch");

        let indices = indices_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("indices download");
        let weights = weights_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("weights download");
        assert_eq!(indices, [1, 3]);
        let sum = weights.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1.0e-6, "selected sum={sum}");
        assert!(weights[0] > weights[1]);

        moe_topk_f32_into_on_stream(
            &logits_device,
            indices_device.output(),
            weights_device.output(),
            2,
            false,
            &stream,
        )
        .expect("top-k launch");

        let indices = indices_device
            .copy_to_host(&stream)
            .expect("indices download");
        let weights = weights_device
            .copy_to_host(&stream)
            .expect("weights download");
        assert_eq!(indices, [1, 3]);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let softmax_sum = logits.iter().map(|value| (*value - max).exp()).sum::<f32>();
        let expected = [
            (logits[1] - max).exp() / softmax_sum,
            (logits[3] - max).exp() / softmax_sum,
        ];
        for (actual, expected) in weights.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn step37_sigmoid_top8_matches_cpu_reference() {
        let logits = (0..288)
            .map(|expert| ((expert * 37 % 101) as f32 - 50.0) * 0.03125)
            .collect::<Vec<_>>();
        let bias = (0..288)
            .map(|expert| ((expert * 19 % 47) as f32 - 23.0) * 0.002)
            .collect::<Vec<_>>();
        let probabilities = logits
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let mut expected_indices = (0..288).collect::<Vec<_>>();
        expected_indices.sort_unstable_by(|&left, &right| {
            (probabilities[right] + bias[right])
                .partial_cmp(&(probabilities[left] + bias[left]))
                .expect("finite score")
                .then_with(|| left.cmp(&right))
        });
        let expected_indices = expected_indices[..8].to_vec();
        let selected_sum = expected_indices
            .iter()
            .map(|&expert| probabilities[expert])
            .sum::<f32>();
        let expected_weights = expected_indices
            .iter()
            .map(|&expert| probabilities[expert] / selected_sum * 3.0)
            .collect::<Vec<_>>();

        let logits = DeviceBuffer::from_host(&logits).expect("logits upload");
        let bias = DeviceBuffer::from_host(&bias).expect("bias upload");
        let mut indices = DeviceBuffer::zeroed(8).expect("indices alloc");
        let mut weights = DeviceBuffer::zeroed(8).expect("weights alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        step37_sigmoid_top8_f32_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            &stream,
        )
        .expect("Step router launch");
        let actual_indices = indices.copy_to_host(&stream).expect("indices download");
        let actual_weights = weights.copy_to_host(&stream).expect("weights download");
        assert_eq!(
            actual_indices.as_ref(),
            expected_indices
                .iter()
                .map(|&expert| expert as u32)
                .collect::<Vec<_>>()
        );
        for (&actual, &expected) in actual_weights.iter().zip(&expected_weights) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn step37_sigmoid_top8_batch_matches_independent_rows() {
        const EXPERTS: usize = 288;
        const ROWS: usize = 2;
        let first = (0..EXPERTS)
            .map(|expert| ((expert * 37 % 101) as f32 - 50.0) * 0.03125)
            .collect::<Vec<_>>();
        let second = (0..EXPERTS)
            .map(|expert| ((expert * 53 % 127) as f32 - 63.0) * 0.025)
            .collect::<Vec<_>>();
        let bias = (0..EXPERTS)
            .map(|expert| ((expert * 19 % 47) as f32 - 23.0) * 0.002)
            .collect::<Vec<_>>();
        let mut logits = first.clone();
        logits.extend_from_slice(&second);
        let logits = DeviceBuffer::from_host(&logits).expect("logits upload");
        let bias = DeviceBuffer::from_host(&bias).expect("bias upload");
        let mut indices = DeviceBuffer::zeroed(ROWS * 8).expect("indices");
        let mut weights = DeviceBuffer::zeroed(ROWS * 8).expect("weights");
        let stream = CudaStream::new_non_blocking().expect("stream");
        step37_sigmoid_top8_f32_batch_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            ROWS,
            &stream,
        )
        .expect("batch router");
        let actual_indices = indices.copy_to_host(&stream).expect("indices download");
        let actual_weights = weights.copy_to_host(&stream).expect("weights download");
        for (row, row_logits) in [first, second].iter().enumerate() {
            let row_logits = DeviceBuffer::from_host(row_logits).expect("row logits");
            let mut row_indices = DeviceBuffer::zeroed(8).expect("row indices");
            let mut row_weights = DeviceBuffer::zeroed(8).expect("row weights");
            step37_sigmoid_top8_f32_into_on_stream(
                &row_logits,
                &bias,
                row_indices.output(),
                row_weights.output(),
                &stream,
            )
            .expect("independent router");
            assert_eq!(
                &actual_indices[row * 8..(row + 1) * 8],
                row_indices
                    .copy_to_host(&stream)
                    .expect("row indices download")
                    .as_slice()
            );
            assert_eq!(
                &actual_weights[row * 8..(row + 1) * 8],
                row_weights
                    .copy_to_host(&stream)
                    .expect("row weights download")
                    .as_slice()
            );
        }
    }

    #[test]
    fn moe_top8_norm256_orders_ties_by_expert_index() {
        let mut logits = vec![-10.0f32; 256];
        for &expert in &[19usize, 3, 241, 7, 8, 9, 10, 11, 12] {
            logits[expert] = 4.0;
        }
        let logits_device = DeviceBuffer::from_host(&logits).expect("logits upload");
        let mut indices_device = DeviceBuffer::<u32>::zeroed(8).expect("indices alloc");
        let mut weights_device = DeviceBuffer::<f32>::zeroed(8).expect("weights alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        moe_topk_f32_into_on_stream(
            &logits_device,
            indices_device.output(),
            weights_device.output(),
            8,
            true,
            &stream,
        )
        .expect("top-k launch");
        let indices = indices_device.copy_to_host(&stream).expect("indices");
        let weights = weights_device.copy_to_host(&stream).expect("weights");
        assert_eq!(indices, [3, 7, 8, 9, 10, 11, 12, 19]);
        for weight in weights.iter() {
            assert!((*weight - 0.125).abs() < 1.0e-6, "weight={weight}");
        }
    }

    #[test]
    fn moe_topk_batch_matches_independent_rows() {
        let rows = 3usize;
        let experts = 256usize;
        let k = 8usize;
        let logits = (0..rows * experts)
            .map(|idx| ((idx * 37 % 509) as f32 - 254.0) / 32.0)
            .collect::<Vec<_>>();
        let logits_device = DeviceBuffer::from_host(&logits).expect("logits upload");
        let mut indices = DeviceBuffer::<u32>::zeroed(rows * k).expect("indices");
        let mut weights = DeviceBuffer::<f32>::zeroed(rows * k).expect("weights");
        let stream = CudaStream::new_non_blocking().expect("stream");
        moe_topk_f32_batch_into_on_stream(
            &logits_device,
            indices.output(),
            weights.output(),
            rows,
            experts,
            k,
            true,
            &stream,
        )
        .expect("batch top-k");
        let indices = indices.copy_to_host(&stream).expect("indices download");
        let weights = weights.copy_to_host(&stream).expect("weights download");
        for row in 0..rows {
            let row_logits = DeviceBuffer::from_host(&logits[row * experts..(row + 1) * experts])
                .expect("row logits");
            let mut row_indices = DeviceBuffer::<u32>::zeroed(k).expect("row indices");
            let mut row_weights = DeviceBuffer::<f32>::zeroed(k).expect("row weights");
            moe_topk_f32_into_on_stream(
                &row_logits,
                row_indices.output(),
                row_weights.output(),
                k,
                true,
                &stream,
            )
            .expect("row top-k");
            assert_eq!(
                &indices[row * k..(row + 1) * k],
                &*row_indices.copy_to_host(&stream).expect("row indices copy")
            );
            assert_eq!(
                &weights[row * k..(row + 1) * k],
                &*row_weights.copy_to_host(&stream).expect("row weights copy")
            );
        }
    }

    #[test]
    fn rope_neox_f32_matches_cpu_reference() {
        let rows = 5;
        let head_dim = 128;
        let position = 17;
        let theta = 1_000_000.0;
        let input = (0..rows * head_dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.0625)
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("RoPE input upload");
        let mut output_device = DeviceBuffer::zeroed(input.len()).expect("RoPE output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_neox_f32_into_on_stream(
            rows,
            head_dim,
            &input_device,
            output_device.output(),
            position,
            theta,
            &stream,
        )
        .expect("RoPE launch");
        let output = output_device.copy_to_host(&stream).expect("RoPE download");

        let expected = cpu_rope_neox(rows, head_dim, &input, position, theta);
        for (idx, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-5,
                "RoPE mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
    }

    #[test]
    fn rope_neox_sequence_f32_matches_cpu_reference() {
        let tokens = 3;
        let heads = 2;
        let head_dim = 16;
        let start_position = 11;
        let theta = 1_000_000.0;
        let input = (0..tokens * heads * head_dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.0625)
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("sequence RoPE input upload");
        let mut output_device =
            DeviceBuffer::zeroed(input.len()).expect("sequence RoPE output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_neox_sequence_f32_into_on_stream(
            tokens,
            heads,
            head_dim,
            &input_device,
            output_device.output(),
            start_position,
            theta,
            &stream,
        )
        .expect("sequence RoPE launch");
        let output = output_device
            .copy_to_host(&stream)
            .expect("sequence RoPE download");

        let mut expected = Vec::with_capacity(input.len());
        for token in 0..tokens {
            let start = token * heads * head_dim;
            let end = start + heads * head_dim;
            expected.extend(cpu_rope_neox(
                heads,
                head_dim,
                &input[start..end],
                start_position + token,
                theta,
            ));
        }
        for (idx, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-5,
                "sequence RoPE mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
    }

    #[test]
    fn rope_neox_proportional_sequence_at_offset_matches_cpu_reference() {
        let capacity = 5;
        let offset = 1;
        let tokens = 3;
        let heads = 2;
        let head_dim = 16;
        let rotary_dim = 8;
        let start_position = 29;
        let theta = 1_000_000.0;
        let input = (0..capacity * heads * head_dim)
            .map(|idx| ((idx % 43) as f32 - 21.0) * 0.03125)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; input.len()];
        for token in 0..tokens {
            let row = offset + token;
            let start = row * heads * head_dim;
            let end = start + heads * head_dim;
            expected[start..end].copy_from_slice(&cpu_rope_neox_proportional(
                heads,
                head_dim,
                rotary_dim,
                &input[start..end],
                start_position + token,
                theta,
            ));
        }

        let input_device = DeviceBuffer::from_host(&input).expect("sequence RoPE input upload");
        let mut output_device = DeviceBuffer::zeroed(input.len()).expect("sequence RoPE alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            tokens,
            heads,
            head_dim,
            rotary_dim,
            &input_device,
            output_device.output(),
            offset,
            start_position,
            theta,
            &stream,
        )
        .expect("proportional sequence RoPE launch");
        let output = output_device
            .copy_to_host(&stream)
            .expect("proportional sequence RoPE download");
        let start = offset * heads * head_dim;
        let end = (offset + tokens) * heads * head_dim;
        assert_close(
            &output[start..end],
            &expected[start..end],
            2.0e-5,
            "sequence RoPE",
        );
    }

    #[test]
    fn dual_rms_norm_rope_sequence_matches_staged_operations() {
        let capacity = 5;
        let offset = 1;
        let tokens = 3;
        let q_heads = 3;
        let k_heads = 2;
        let head_dim = 16;
        let rotary_dim = 8;
        let start_position = 29;
        let theta = 1_000_000.0;
        let q_eps = 1.0e-6;
        let k_eps = 2.0e-6;
        let q = (0..capacity * q_heads * head_dim)
            .map(|idx| ((idx % 43) as f32 - 21.0) * 0.03125)
            .collect::<Vec<_>>();
        let k = (0..capacity * k_heads * head_dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.046875)
            .collect::<Vec<_>>();
        let q_weight = (0..head_dim)
            .map(|idx| 0.75 + idx as f32 * 0.025)
            .collect::<Vec<_>>();
        let k_weight = (0..head_dim)
            .map(|idx| 1.25 - idx as f32 * 0.02)
            .collect::<Vec<_>>();
        let q_device = DeviceBuffer::from_host(&q).expect("Q upload");
        let k_device = DeviceBuffer::from_host(&k).expect("K upload");
        let q_weight_device = DeviceBuffer::from_host(&q_weight).expect("Q weight upload");
        let k_weight_device = DeviceBuffer::from_host(&k_weight).expect("K weight upload");
        let mut q_normalized = DeviceBuffer::zeroed(q.len()).expect("normalized Q");
        let mut k_normalized = DeviceBuffer::zeroed(k.len()).expect("normalized K");
        let mut expected_q = DeviceBuffer::zeroed(q.len()).expect("expected Q");
        let mut expected_k = DeviceBuffer::zeroed(k.len()).expect("expected K");
        let mut actual_q = DeviceBuffer::zeroed(q.len()).expect("actual Q");
        let mut actual_k = DeviceBuffer::zeroed(k.len()).expect("actual K");
        let stream = CudaStream::new_non_blocking().expect("stream");

        rms_norm_f32_into_on_stream(
            capacity * q_heads,
            head_dim,
            &q_device,
            &q_weight_device,
            q_normalized.output(),
            q_eps,
            &stream,
        )
        .expect("Q RMSNorm");
        rms_norm_f32_into_on_stream(
            capacity * k_heads,
            head_dim,
            &k_device,
            &k_weight_device,
            k_normalized.output(),
            k_eps,
            &stream,
        )
        .expect("K RMSNorm");
        rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            tokens,
            q_heads,
            head_dim,
            rotary_dim,
            &q_normalized,
            expected_q.output(),
            offset,
            start_position,
            theta,
            &stream,
        )
        .expect("Q RoPE");
        rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            tokens,
            k_heads,
            head_dim,
            rotary_dim,
            &k_normalized,
            expected_k.output(),
            offset,
            start_position,
            theta,
            &stream,
        )
        .expect("K RoPE");
        dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream(
            tokens,
            q_heads,
            k_heads,
            head_dim,
            rotary_dim,
            &q_device,
            &q_weight_device,
            actual_q.output(),
            q_eps,
            &k_device,
            &k_weight_device,
            actual_k.output(),
            k_eps,
            offset,
            start_position,
            theta,
            &stream,
        )
        .expect("fused Q/K RMSNorm RoPE");

        let q_start = offset * q_heads * head_dim;
        let q_end = (offset + tokens) * q_heads * head_dim;
        let k_start = offset * k_heads * head_dim;
        let k_end = (offset + tokens) * k_heads * head_dim;
        let expected_q = expected_q
            .copy_to_host(&stream)
            .expect("expected Q download");
        let expected_k = expected_k
            .copy_to_host(&stream)
            .expect("expected K download");
        let actual_q = actual_q.copy_to_host(&stream).expect("actual Q download");
        let actual_k = actual_k.copy_to_host(&stream).expect("actual K download");
        assert_close(
            &actual_q[q_start..q_end],
            &expected_q[q_start..q_end],
            3.0e-5,
            "fused Q RMSNorm RoPE",
        );
        assert_close(
            &actual_k[k_start..k_end],
            &expected_k[k_start..k_end],
            3.0e-5,
            "fused K RMSNorm RoPE",
        );
    }

    #[test]
    fn moe_gather_rms_norm_quantization_matches_staged_rows() {
        let rows = 3;
        let routes_per_row = 2;
        let experts = 4;
        let in_features = 64;
        let input = (0..rows * in_features)
            .map(|idx| ((idx % 47) as f32 - 23.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight = (0..in_features)
            .map(|idx| 0.75 + idx as f32 * 0.0078125)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input).expect("input upload");
        let weight = DeviceBuffer::from_host(&weight).expect("weight upload");
        let indices = DeviceBuffer::from_host(&[0, 1, 2, 3, 1, 0]).expect("indices upload");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut routes =
            MoeSortedRoutes::new(rows * routes_per_row, experts).expect("route workspace");
        routes
            .sort_on_stream(&indices, &stream)
            .expect("route sort");
        let mut fused = MoeSortedNvfp4Rows::new(rows, routes_per_row, experts, in_features)
            .expect("fused workspace");
        fused
            .gather_rms_norm_quantize_on_stream(&input, &weight, 1.0e-6, &routes, &stream)
            .expect("fused quantization");

        let mut normalized = DeviceBuffer::zeroed(rows * in_features).expect("normalized rows");
        rms_norm_f32_into_on_stream(
            rows,
            in_features,
            &input,
            &weight,
            normalized.output(),
            1.0e-6,
            &stream,
        )
        .expect("staged RMSNorm");
        let mut expected_packed =
            DeviceBuffer::zeroed(rows * in_features / 2).expect("expected packed");
        let mut expected_scales =
            DeviceBuffer::zeroed(rows * in_features / 16).expect("expected scales");
        quantize_nvfp4_simple_scales_f32_into_on_stream(
            &normalized,
            &mut expected_packed,
            &mut expected_scales,
            &stream,
        )
        .expect("staged quantization");

        assert_eq!(
            fused
                .source_scales
                .copy_to_host(&stream)
                .expect("fused scales"),
            expected_scales
                .copy_to_host(&stream)
                .expect("expected scales")
        );
        assert_eq!(
            fused
                .source_packed
                .copy_to_host(&stream)
                .expect("fused packed"),
            expected_packed
                .copy_to_host(&stream)
                .expect("expected packed")
        );
    }

    #[test]
    fn rope_neox_partial_f32_matches_cpu_reference() {
        let rows = 3usize;
        let head_dim = 12usize;
        let rotary_dim = 4usize;
        let position = 17usize;
        let theta = 1_000_000.0f32;
        let input = (0..rows * head_dim)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.05)
            .collect::<Vec<_>>();
        let expected = cpu_rope_neox_partial(rows, head_dim, rotary_dim, &input, position, theta);

        let input_device = DeviceBuffer::from_host(&input).expect("partial RoPE input upload");
        let mut output_device =
            DeviceBuffer::<f32>::zeroed(input.len()).expect("partial RoPE alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_neox_partial_f32_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            &input_device,
            output_device.output(),
            position,
            theta,
            &stream,
        )
        .expect("partial RoPE enqueue");

        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("partial RoPE download"),
            &expected,
            2.0e-6,
            "partial RoPE",
        );
    }

    #[test]
    fn rope_neox_proportional_f32_matches_cpu_reference() {
        let rows = 3usize;
        let head_dim = 12usize;
        let rotary_dim = 4usize;
        let position = 17usize;
        let theta = 1_000_000.0f32;
        let input = (0..rows * head_dim)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.05)
            .collect::<Vec<_>>();
        let expected =
            cpu_rope_neox_proportional(rows, head_dim, rotary_dim, &input, position, theta);

        let input_device = DeviceBuffer::from_host(&input).expect("proportional RoPE input upload");
        let mut output_device =
            DeviceBuffer::<f32>::zeroed(input.len()).expect("proportional RoPE alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_neox_proportional_f32_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            &input_device,
            output_device.output(),
            position,
            theta,
            &stream,
        )
        .expect("proportional RoPE enqueue");

        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("proportional RoPE download"),
            &expected,
            2.0e-6,
            "proportional RoPE",
        );
    }

    #[test]
    fn rope_imrope_f32_matches_cpu_reference() {
        // Qwen3.6 full-attention: head_dim=256, partial_rotary=0.25 -> rotary_dim=64,
        // mrope_section=[11,11,10,0]. Text positions: t=h=w=token pos, extra=0.
        let rows = 16usize;
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let sections = MropeSections {
            v0: 11,
            v1: 11,
            v2: 10,
            v3: 0,
        };
        let position = 42u32;
        let positions = [position, position, position, 0];
        let theta = 10_000_000.0f32;
        let input = (0..rows * head_dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01)
            .collect::<Vec<_>>();
        let expected = cpu_rope_imrope(
            rows, head_dim, rotary_dim, sections, positions, &input, theta,
        );

        let input_device = DeviceBuffer::from_host(&input).expect("IMRoPE input upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(input.len()).expect("IMRoPE alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_imrope_f32_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            sections,
            positions,
            &input_device,
            output_device.output(),
            theta,
            &stream,
        )
        .expect("IMRoPE enqueue");

        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("IMRoPE download"),
            &expected,
            2.0e-6,
            "IMRoPE",
        );
    }

    #[test]
    fn rope_imrope_text_positions_match_standard_partial_rope() {
        let rows = 16usize;
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let sections = MropeSections {
            v0: 11,
            v1: 11,
            v2: 10,
            v3: 0,
        };
        let position = 47usize;
        let theta = 10_000_000.0f32;
        let input = (0..rows * head_dim)
            .map(|idx| ((idx * 13 % 97) as f32 - 48.0) * 0.01)
            .collect::<Vec<_>>();
        let expected = cpu_rope_neox_partial(rows, head_dim, rotary_dim, &input, position, theta);
        let input_device = DeviceBuffer::from_host(&input).expect("IMRoPE input upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(input.len()).expect("IMRoPE alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_imrope_f32_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            sections,
            [position as u32, position as u32, position as u32, 0],
            &input_device,
            output_device.output(),
            theta,
            &stream,
        )
        .expect("IMRoPE enqueue");
        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("IMRoPE download"),
            &expected,
            2.0e-6,
            "text IMRoPE equals standard partial RoPE",
        );
    }

    #[test]
    fn rope_imrope_extra_section_is_identity() {
        // When extra section has nonzero size and pos_extra=0, pairs in the
        // extra sector get theta_base=0 -> no rotation. With v3=2 and
        // sections summing to rotary_dim/2, the last 2 pairs must be identity.
        let rows = 2usize;
        let head_dim = 16usize;
        let rotary_dim = 8usize;
        let sections = MropeSections {
            v0: 1,
            v1: 1,
            v2: 0,
            v3: 2,
        };
        let positions = [5u32, 5, 5, 0];
        let theta = 10_000.0f32;
        let input = (0..rows * head_dim)
            .map(|idx| idx as f32 * 0.1)
            .collect::<Vec<_>>();
        let expected = cpu_rope_imrope(
            rows, head_dim, rotary_dim, sections, positions, &input, theta,
        );

        let input_device = DeviceBuffer::from_host(&input).expect("IMRoPE identity input upload");
        let mut output_device =
            DeviceBuffer::<f32>::zeroed(input.len()).expect("IMRoPE identity alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        rope_imrope_f32_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            sections,
            positions,
            &input_device,
            output_device.output(),
            theta,
            &stream,
        )
        .expect("IMRoPE identity enqueue");
        let actual = output_device
            .copy_to_host(&stream)
            .expect("IMRoPE identity download");
        assert_close(&actual, &expected, 2.0e-6, "IMRoPE identity");
        // Pairs 2,3 (the v3=2 extra section with pos=0) must equal the input.
        for row in 0..rows {
            let rs = row * head_dim;
            assert_eq!(
                actual[rs + 2],
                input[rs + 2],
                "extra pair 2 row {row} not identity"
            );
            assert_eq!(
                actual[rs + 3],
                input[rs + 3],
                "extra pair 3 row {row} not identity"
            );
            assert_eq!(
                actual[rs + 2 + 4],
                input[rs + 2 + 4],
                "extra pair 2 high row {row} not identity"
            );
            assert_eq!(
                actual[rs + 3 + 4],
                input[rs + 3 + 4],
                "extra pair 3 high row {row} not identity"
            );
        }
    }

    #[test]
    fn add_f32_matches_cpu_reference() {
        let left = (0..513)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.125)
            .collect::<Vec<_>>();
        let right = (0..513)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>();

        let left_device = DeviceBuffer::from_host(&left).expect("left upload");
        let right_device = DeviceBuffer::from_host(&right).expect("right upload");
        let mut output_device = DeviceBuffer::zeroed(left.len()).expect("add output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        add_f32_into_on_stream(&left_device, &right_device, output_device.output(), &stream)
            .expect("add launch");
        let output = output_device.copy_to_host(&stream).expect("add download");

        for (idx, ((actual, left), right)) in
            output.iter().zip(left.iter()).zip(right.iter()).enumerate()
        {
            let expected = left + right;
            assert_eq!(
                *actual, expected,
                "add mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn concat_f32_rows_matches_cpu_reference() {
        const ROWS: usize = 3;
        const COLS: usize = 5;
        let left = (0..ROWS * COLS)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let right = (0..ROWS * COLS)
            .map(|value| 100.0 + value as f32)
            .collect::<Vec<_>>();
        let left_device = DeviceBuffer::from_host(&left).expect("left upload");
        let right_device = DeviceBuffer::from_host(&right).expect("right upload");
        let mut output = DeviceBuffer::zeroed(ROWS * COLS * 2).expect("output allocation");
        let stream = CudaStream::new_non_blocking().expect("stream");

        concat_f32_rows_into_on_stream(
            ROWS,
            COLS,
            &left_device,
            &right_device,
            output.output(),
            &stream,
        )
        .expect("concatenate rows");
        let actual = output.copy_to_host(&stream).expect("output download");

        let expected = (0..ROWS)
            .flat_map(|row| {
                left[row * COLS..(row + 1) * COLS]
                    .iter()
                    .chain(&right[row * COLS..(row + 1) * COLS])
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.as_ref(), expected.as_slice());
    }

    #[test]
    fn increment_u32_matches_cpu_reference() {
        let mut values = DeviceBuffer::from_host(&[0u32, 7, u32::MAX - 1]).expect("upload");
        let stream = CudaStream::new_non_blocking().expect("stream");
        increment_u32_in_place_on_stream(values.inout(), 2, &stream).expect("increment");
        let actual = values.copy_to_host(&stream).expect("download");
        assert_eq!(actual.as_ref(), &[2, 9, 0]);
    }

    #[test]
    fn store_u32_column_writes_sequence_major_drafts() {
        let input = DeviceBuffer::from_host(&[11u32, 21, 31]).expect("input upload");
        let mut output =
            DeviceBuffer::from_host(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9]).expect("output upload");
        let stream = CudaStream::new_non_blocking().expect("stream");
        store_u32_column_into_on_stream(&input, output.output(), 3, 3, 1, &stream)
            .expect("column store");
        assert_eq!(
            output.copy_to_host(&stream).expect("output download"),
            [1, 11, 3, 4, 21, 6, 7, 31, 9]
        );
    }

    #[test]
    fn prepend_u32_rows_interleaves_sequence_inputs() {
        let first = DeviceBuffer::from_host(&[10u32, 20, 30]).expect("first upload");
        let remaining = DeviceBuffer::from_host(&[11u32, 12, 13, 21, 22, 23, 31, 32, 33])
            .expect("remaining upload");
        let mut output = DeviceBuffer::zeroed(12).expect("output allocation");
        let stream = CudaStream::new_non_blocking().expect("stream");
        prepend_u32_rows_into_on_stream(&first, &remaining, output.output(), 3, 3, &stream)
            .expect("prepend rows");
        assert_eq!(
            output.copy_to_host(&stream).expect("output download"),
            [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
        );
    }

    #[test]
    fn moe_weighted_accumulate_batch_matches_independent_rows() {
        const ROWS: usize = 3;
        const GROUPS: usize = 4;
        const LEN: usize = 19;
        let indices = [2u32, 0, 3, 1, 1, 3, 0, 2, 3, 2, 1, 0];
        let weights = [
            0.4f32, 0.3, 0.2, 0.1, 0.35, 0.3, 0.2, 0.15, 0.5, 0.25, 0.15, 0.1,
        ];
        let alphas = [0.75f32, 1.0, 1.25, 0.875];
        let routed = (0..ROWS * GROUPS)
            .map(|route| {
                DeviceBuffer::from_host(
                    &(0..LEN)
                        .map(|column| {
                            (((column * 7 + route * 11) % 101) as f32 - 50.0) * 0.00390625
                        })
                        .collect::<Vec<_>>(),
                )
                .expect("routed output")
            })
            .collect::<Vec<_>>();
        let routed_ptrs = DeviceBuffer::from_host(
            &routed
                .iter()
                .map(|values| values.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("routed pointers");
        let indices_device = DeviceBuffer::from_host(&indices).expect("indices");
        let weights_device = DeviceBuffer::from_host(&weights).expect("weights");
        let alphas_device = DeviceBuffer::from_host(&alphas).expect("alphas");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut actual = DeviceBuffer::zeroed(ROWS * LEN).expect("batch output");
        moe_weighted_accumulate_slots_f32_batch_on_stream(
            &indices_device,
            &weights_device,
            &routed_ptrs,
            &alphas_device,
            actual.inout(),
            ROWS,
            GROUPS,
            &stream,
        )
        .expect("batched route accumulation");
        let actual = actual.copy_to_host(&stream).expect("batch output download");

        for row in 0..ROWS {
            let begin = row * GROUPS;
            let end = begin + GROUPS;
            let row_indices = DeviceBuffer::from_host(&indices[begin..end]).expect("row indices");
            let row_weights = DeviceBuffer::from_host(&weights[begin..end]).expect("row weights");
            let row_ptrs = DeviceBuffer::from_host(
                &routed[begin..end]
                    .iter()
                    .map(|values| values.as_const_ptr().cast::<f32>())
                    .collect::<Vec<_>>(),
            )
            .expect("row pointers");
            let mut expected = DeviceBuffer::zeroed(LEN).expect("row output");
            moe_weighted_accumulate_slots_f32_on_stream(
                &row_indices,
                &row_weights,
                &row_ptrs,
                &alphas_device,
                expected.inout(),
                &stream,
            )
            .expect("independent route accumulation");
            assert_close(
                &actual[row * LEN..(row + 1) * LEN],
                &expected.copy_to_host(&stream).expect("row output download"),
                0.0,
                &format!("batched route accumulation row {row}"),
            );
        }
    }

    #[test]
    fn sorted_f32_moe_accumulation_matches_route_order() {
        const ROWS: usize = 3;
        const GROUPS: usize = 4;
        const LEN: usize = 19;
        let indices = [2u32, 0, 3, 1, 1, 3, 0, 2, 3, 2, 1, 0];
        let weights = [
            0.4f32, 0.3, 0.2, 0.1, 0.35, 0.3, 0.2, 0.15, 0.5, 0.25, 0.15, 0.1,
        ];
        let alphas = [0.75f32, 1.0, 1.25, 0.875];
        let routed = (0..ROWS * GROUPS)
            .map(|route| {
                DeviceBuffer::from_host(
                    &(0..LEN)
                        .map(|column| {
                            (((column * 7 + route * 11) % 101) as f32 - 50.0) * 0.00390625
                        })
                        .collect::<Vec<_>>(),
                )
                .expect("routed output")
            })
            .collect::<Vec<_>>();
        let indices_device = DeviceBuffer::from_host(&indices).expect("indices");
        let weights_device = DeviceBuffer::from_host(&weights).expect("weights");
        let alphas_device = DeviceBuffer::from_host(&alphas).expect("alphas");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut routes = MoeSortedRoutes::new(ROWS * GROUPS, alphas.len()).expect("route ordering");
        routes
            .sort_on_stream(&indices_device, &stream)
            .expect("sort routes");
        let sorted_routes = routes
            .sorted_routes()
            .copy_to_host(&stream)
            .expect("sorted routes");
        let sorted_ptrs = DeviceBuffer::from_host(
            &sorted_routes
                .iter()
                .map(|&route| routed[route as usize].as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("sorted routed pointers");
        let unsorted_ptrs = DeviceBuffer::from_host(
            &routed
                .iter()
                .map(|values| values.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("unsorted routed pointers");
        let mut expected = DeviceBuffer::zeroed(ROWS * LEN).expect("reference output");
        moe_weighted_accumulate_slots_f32_batch_on_stream(
            &indices_device,
            &weights_device,
            &unsorted_ptrs,
            &alphas_device,
            expected.inout(),
            ROWS,
            GROUPS,
            &stream,
        )
        .expect("route-order accumulation");
        let mut actual = DeviceBuffer::zeroed(ROWS * LEN).expect("sorted output");
        moe_weighted_accumulate_sorted_slots_f32_batch_on_stream(
            &routes,
            &indices_device,
            &weights_device,
            &sorted_ptrs,
            &alphas_device,
            actual.inout(),
            ROWS,
            GROUPS,
            LEN,
            &stream,
        )
        .expect("sorted accumulation");

        assert_eq!(
            actual
                .copy_to_host(&stream)
                .expect("sorted output download")
                .as_ref(),
            expected
                .copy_to_host(&stream)
                .expect("reference output download")
                .as_ref()
        );
    }

    #[test]
    fn qwen36_routed_ffn_finalize_matches_unfused_sequence() {
        let len = 2048;
        let groups = 8;
        let indices =
            DeviceBuffer::from_host(&[2u32, 0, 3, 1, 2, 3, 0, 1]).expect("route indices upload");
        let route_weights =
            DeviceBuffer::from_host(&[0.19f32, 0.17, 0.15, 0.14, 0.12, 0.1, 0.08, 0.05])
                .expect("route weights upload");
        let alpha_table =
            DeviceBuffer::from_host(&[0.75f32, 1.0, 1.25, 0.875]).expect("alpha table upload");
        let routed = (0..groups)
            .map(|slot| {
                DeviceBuffer::from_host(
                    &(0..len)
                        .map(|idx| (((idx * 7 + slot * 11) % 101) as f32 - 50.0) * 0.00390625)
                        .collect::<Vec<_>>(),
                )
                .expect("routed output upload")
            })
            .collect::<Vec<_>>();
        let routed_ptrs = DeviceBuffer::from_host(
            &routed
                .iter()
                .map(|values| values.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("routed pointer table upload");
        let shared_gate_logit = DeviceBuffer::from_host(&[0.375f32]).expect("gate upload");
        let shared_output = DeviceBuffer::from_host(
            &(0..len)
                .map(|idx| ((idx % 79) as f32 - 39.0) * 0.0078125)
                .collect::<Vec<_>>(),
        )
        .expect("shared output upload");
        let residual = DeviceBuffer::from_host(
            &(0..len)
                .map(|idx| ((idx % 67) as f32 - 33.0) * 0.015625)
                .collect::<Vec<_>>(),
        )
        .expect("residual upload");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut moe_output = DeviceBuffer::zeroed(len).expect("MoE output alloc");
        moe_weighted_accumulate_slots_f32_on_stream(
            &indices,
            &route_weights,
            &routed_ptrs,
            &alpha_table,
            moe_output.inout(),
            &stream,
        )
        .expect("routed accumulation");
        let mut shared_gated = DeviceBuffer::zeroed(len).expect("shared gated alloc");
        sigmoid_scale_scalar_f32_into_on_stream(
            &shared_gate_logit,
            &shared_output,
            shared_gated.output(),
            &stream,
        )
        .expect("shared gate");
        let mut ffn_output = DeviceBuffer::zeroed(len).expect("FFN output alloc");
        add_f32_into_on_stream(&moe_output, &shared_gated, ffn_output.output(), &stream)
            .expect("FFN add");
        let mut residual_output = DeviceBuffer::zeroed(len).expect("residual output alloc");
        add_f32_into_on_stream(&residual, &ffn_output, residual_output.output(), &stream)
            .expect("residual add");
        let mut reference = DeviceBuffer::zeroed(len).expect("reference alloc");
        round_f32_to_bf16_into_on_stream(&residual_output, reference.output(), &stream)
            .expect("reference BF16 round");

        let mut candidate = DeviceBuffer::zeroed(len).expect("candidate alloc");
        qwen36_ffn_finalize_routed_f32_into_on_stream(
            &indices,
            &route_weights,
            &routed_ptrs,
            &alpha_table,
            &shared_gate_logit,
            &shared_output,
            &residual,
            candidate.output(),
            &stream,
        )
        .expect("fused FFN finalize");

        assert_eq!(
            candidate
                .copy_to_host(&stream)
                .expect("candidate download")
                .into_vec(),
            reference
                .copy_to_host(&stream)
                .expect("reference download")
                .into_vec()
        );
    }

    #[test]
    #[ignore = "CUDA graph capture must not run alongside parallel default-stream CUDA tests"]
    fn add_f32_replays_from_cuda_graph() {
        let left = (0..513)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.125)
            .collect::<Vec<_>>();
        let right = (0..513)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>();

        let left_device = DeviceBuffer::from_host(&left).expect("left upload");
        let right_device = DeviceBuffer::from_host(&right).expect("right upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(left.len()).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");
        let graph = stream
            .capture(|stream| {
                add_f32_into_on_stream(&left_device, &right_device, output_device.output(), stream)
            })
            .expect("graph capture");

        graph.launch(&stream).expect("graph launch 0");
        graph.launch(&stream).expect("graph launch 1");
        let output = output_device.copy_to_host(&stream).expect("add download");

        for (idx, ((actual, left), right)) in
            output.iter().zip(left.iter()).zip(right.iter()).enumerate()
        {
            let expected = left + right;
            assert_eq!(
                *actual, expected,
                "graph add mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    #[ignore = "CUDA graph capture must not run alongside parallel default-stream CUDA tests"]
    fn decode_primitives_replay_from_cuda_graph() {
        let len = 4096;
        let left = (0..len)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.125)
            .collect::<Vec<_>>();
        let right = (0..len)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>();
        let weight = vec![1.0f32; len];

        let left_device = DeviceBuffer::from_host(&left).expect("left upload");
        let right_device = DeviceBuffer::from_host(&right).expect("right upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut norm_device = DeviceBuffer::<f32>::zeroed(len).expect("norm alloc");
        let mut rope_device = DeviceBuffer::<f32>::zeroed(len).expect("rope alloc");
        let mut add_device = DeviceBuffer::<f32>::zeroed(len).expect("add alloc");
        let mut silu_device = DeviceBuffer::<f32>::zeroed(len).expect("silu alloc");
        let mut cache_device = DeviceBuffer::<f32>::zeroed(len * 2).expect("cache alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");
        let graph = stream
            .capture(|stream| {
                rms_norm_f32_into_on_stream(
                    1,
                    len,
                    &left_device,
                    &weight_device,
                    norm_device.output(),
                    1.0e-6,
                    stream,
                )?;
                rope_neox_f32_into_on_stream(
                    32,
                    128,
                    &norm_device,
                    rope_device.output(),
                    3,
                    1.0e6,
                    stream,
                )?;
                add_f32_into_on_stream(&rope_device, &right_device, add_device.output(), stream)?;
                silu_mul_f32_into_on_stream(
                    &add_device,
                    &right_device,
                    silu_device.output(),
                    stream,
                )?;
                append_rows_f32_into_on_stream(
                    &silu_device,
                    cache_device.output(),
                    1,
                    1,
                    len,
                    stream,
                )
            })
            .expect("graph capture");

        graph.launch(&stream).expect("graph launch");
        let actual = silu_device.copy_to_host(&stream).expect("silu download");

        assert!(
            actual.iter().all(|value| value.is_finite()),
            "graph primitive chain produced non-finite values"
        );
        assert_ne!(actual, vec![0.0; len]);
        let cache = cache_device.copy_to_host(&stream).expect("cache download");
        assert_eq!(&cache[len..], actual.as_slice());
    }

    #[test]
    fn layout_transpose_and_copy_row_match_cpu_reference() {
        let rows = 3;
        let cols = 4;
        let input = (0..rows * cols)
            .map(|idx| idx as f32 + 0.25)
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("layout input upload");

        let col_major_device =
            row_major_to_col_major_f32(rows, cols, &input_device).expect("row to col");
        let roundtrip_device =
            col_major_to_row_major_f32(rows, cols, &col_major_device).expect("col to row");
        let row_device = copy_row_f32(rows, cols, 1, &input_device).expect("copy row");
        let mut row_stream_device = DeviceBuffer::<f32>::zeroed(cols).expect("stream row alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        copy_row_f32_into_on_stream(
            rows,
            cols,
            2,
            &input_device,
            row_stream_device.output(),
            &stream,
        )
        .expect("copy row on stream");
        synchronize_device().expect("layout sync");

        let col_major = col_major_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("col download");
        let roundtrip = roundtrip_device
            .copy_to_host(&stream)
            .expect("roundtrip download");
        let row = row_device.copy_to_host(&stream).expect("row download");
        let row_stream = row_stream_device
            .copy_to_host(&stream)
            .expect("stream row download");

        let mut expected_col_major = vec![0.0; input.len()];
        for row in 0..rows {
            for col in 0..cols {
                expected_col_major[row + col * rows] = input[row * cols + col];
            }
        }
        assert_eq!(col_major, expected_col_major);
        assert_eq!(roundtrip, input);
        assert_eq!(row, input[cols..2 * cols]);
        assert_eq!(row_stream, input[2 * cols..3 * cols]);
    }

    #[test]
    fn gather_group_row_matches_cpu_reference() {
        const GROUPS: usize = 3;
        const ROWS: usize = 4;
        const COLS: usize = 5;
        let input = (0..GROUPS * ROWS * COLS)
            .map(|index| index as f32 * 0.25)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input).expect("input upload");
        let mut output = DeviceBuffer::zeroed(GROUPS * COLS).expect("output allocation");
        let stream = CudaStream::new_non_blocking().expect("stream");
        gather_group_row_f32_into_on_stream(
            &input,
            output.output(),
            GROUPS,
            ROWS,
            2,
            COLS,
            &stream,
        )
        .expect("grouped row gather");
        let expected = (0..GROUPS)
            .flat_map(|group| {
                let begin = (group * ROWS + 2) * COLS;
                (begin..begin + COLS).map(|index| index as f32 * 0.25)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            output.copy_to_host(&stream).expect("output download"),
            expected
        );
    }

    #[test]
    fn copy_bf16_row_to_f32_indexed_matches_cpu_reference() {
        let rows = 4;
        let cols = 5;
        let values = (0..rows * cols)
            .map(|idx| format::f32_to_bf16(idx as f32 * 0.5 - 3.0))
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&values).expect("BF16 input upload");
        let mut row_device = DeviceBuffer::from_host(&[2u32]).expect("row upload");

        let mut output_device = DeviceBuffer::zeroed(cols).expect("BF16 row output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");
        copy_bf16_row_to_f32_indexed_into_on_stream(
            rows,
            cols,
            &input_device,
            &row_device,
            output_device.output(),
            &stream,
        )
        .expect("BF16 row copy");
        let output = output_device.copy_to_host(&stream).expect("row download");

        let expected = values[2 * cols..3 * cols]
            .iter()
            .map(|value| format::bf16_to_f32(*value))
            .collect::<Vec<_>>();
        assert_eq!(output, expected);

        row_device.copy_from_host(&[1u32]).expect("row update");
        let mut stream_output = DeviceBuffer::<f32>::zeroed(cols).expect("stream output alloc");
        copy_bf16_row_to_f32_indexed_into_on_stream(
            rows,
            cols,
            &input_device,
            &row_device,
            stream_output.output(),
            &stream,
        )
        .expect("BF16 row copy on stream");
        let stream_values = stream_output
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("stream row download");
        let expected_stream = values[cols..2 * cols]
            .iter()
            .map(|value| format::bf16_to_f32(*value))
            .collect::<Vec<_>>();
        assert_eq!(stream_values, expected_stream);

        let mut host_index_output = DeviceBuffer::<f32>::zeroed(cols).expect("host index output");
        copy_bf16_row_to_f32_into_on_stream(
            rows,
            cols,
            3,
            &input_device,
            host_index_output.output(),
            &stream,
        )
        .expect("host-indexed BF16 row copy");
        let expected_host_index = values[3 * cols..4 * cols]
            .iter()
            .map(|value| format::bf16_to_f32(*value))
            .collect::<Vec<_>>();
        assert_eq!(
            host_index_output
                .copy_to_host(&stream)
                .expect("host-indexed row download"),
            expected_host_index,
        );
    }

    #[test]
    fn copy_fp8_rows_to_f32_indexed_applies_selected_row_scales() {
        let rows = 3;
        let cols = 4;
        let values = vec![
            0x38, 0x30, 0xb8, 0x00, 0x38, 0x40, 0x44, 0xb0, 0x30, 0x38, 0x40, 0x44,
        ];
        let scales = vec![0.5, 2.0, 4.0];
        let indices = vec![2u32, 0];
        let input = DeviceBuffer::from_host(&values).expect("FP8 embedding upload");
        let scales_device = DeviceBuffer::from_host(&scales).expect("FP8 scale upload");
        let indices_device = DeviceBuffer::from_host(&indices).expect("row index upload");
        let mut output = DeviceBuffer::zeroed(indices.len() * cols).expect("FP8 row output");
        let stream = CudaStream::new_non_blocking().expect("stream");

        copy_fp8_rows_to_f32_indexed_prefix_into_on_stream(
            rows,
            cols,
            &input,
            &scales_device,
            &indices_device,
            output.output(),
            indices.len(),
            &stream,
        )
        .expect("FP8 embedding gather");

        let expected = indices
            .iter()
            .flat_map(|&row| {
                let scale = scales[row as usize];
                values[row as usize * cols..(row as usize + 1) * cols]
                    .iter()
                    .map(move |&value| format::e4m3_value(value) * scale)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            output.copy_to_host(&stream).expect("FP8 row download"),
            expected
        );
    }

    #[test]
    fn quantize_nvfp4_col_major_f32_device_matches_host_quantizer() {
        let rows = 96;
        let cols = 3;
        let input_scale = 0.25;
        let input = (0..rows * cols)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let scaled = input
            .iter()
            .map(|value| value / input_scale)
            .collect::<Vec<_>>();
        let expected = format::quantize_nvfp4_col_major(rows, cols, &scaled);

        let input_device = DeviceBuffer::from_host(&input).expect("quant input upload");
        let matrix = quantize_nvfp4_col_major_f32_device(rows, cols, &input_device, input_scale)
            .expect("device quantization");
        synchronize_device().expect("device quantization sync");

        let packed = matrix
            .values
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("packed download");
        let scales = matrix
            .scales
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("scales download");
        assert_eq!(packed, expected.packed_values);
        assert_eq!(scales, expected.scales);
    }

    #[test]
    fn fused_rms_norm_nvfp4_quantization_matches_staged_path() {
        let rows = 3;
        // More than 16 32-feature pairs catches launch/loop warp-count
        // mismatches that leave alternating feature bands unwritten.
        let cols = 544;
        let eps = 1.0e-6;
        let input_scale = 0.375;
        let input = (0..rows * cols)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.015625)
            .collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|idx| 0.75 + (idx % 13) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input).expect("input upload");
        let weight = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut normalized = DeviceBuffer::zeroed(rows * cols).expect("normalized allocation");
        let mut expected = Nvfp4Matrix::zeroed_col_major(cols, rows).expect("expected matrix");
        let mut actual = Nvfp4Matrix::zeroed_col_major(cols, rows).expect("actual matrix");
        let stream = CudaStream::new_non_blocking().expect("stream");

        rms_norm_f32_into_on_stream(
            rows,
            cols,
            &input,
            &weight,
            normalized.output(),
            eps,
            &stream,
        )
        .expect("staged RMSNorm");
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            cols,
            rows,
            &normalized,
            &mut expected,
            input_scale,
            &stream,
        )
        .expect("staged quantization");
        rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
            rows,
            cols,
            &input,
            &weight,
            &mut actual,
            eps,
            input_scale,
            &stream,
        )
        .expect("fused quantization");

        assert_eq!(
            actual.values.copy_to_host(&stream).expect("actual values"),
            expected
                .values
                .copy_to_host(&stream)
                .expect("expected values")
        );
        assert_eq!(
            actual.scales.copy_to_host(&stream).expect("actual scales"),
            expected
                .scales
                .copy_to_host(&stream)
                .expect("expected scales")
        );
    }

    #[test]
    fn fused_rms_norm_residual_paths_match_staged_operations() {
        let rows = 3;
        let cols = 96;
        let eps = 1.0e-6;
        let input = (0..rows * cols)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.015625)
            .collect::<Vec<_>>();
        let right = (0..rows * cols)
            .map(|idx| ((idx * 29 % 107) as f32 - 53.0) * 0.01171875)
            .collect::<Vec<_>>();
        let residual = (0..rows * cols)
            .map(|idx| ((idx * 7 % 61) as f32 - 30.0) * 0.0078125)
            .collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|idx| 0.75 + (idx % 13) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let right_weight = (0..cols)
            .map(|idx| 0.625 + (idx % 11) as f32 * 0.0234375)
            .collect::<Vec<_>>();
        let channel_scale = (0..cols)
            .map(|idx| 0.875 + (idx % 7) as f32 * 0.015625)
            .collect::<Vec<_>>();
        let row_scale = vec![0.75, 1.0, 1.25];
        let input = DeviceBuffer::from_host(&input).expect("input upload");
        let right = DeviceBuffer::from_host(&right).expect("right upload");
        let residual = DeviceBuffer::from_host(&residual).expect("residual upload");
        let weight = DeviceBuffer::from_host(&weight).expect("weight upload");
        let right_weight = DeviceBuffer::from_host(&right_weight).expect("right weight upload");
        let channel_scale = DeviceBuffer::from_host(&channel_scale).expect("channel scale upload");
        let row_scale = DeviceBuffer::from_host(&row_scale).expect("row scale upload");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut left_norm = DeviceBuffer::zeroed(rows * cols).expect("left norm");
        let mut right_norm = DeviceBuffer::zeroed(rows * cols).expect("right norm");
        let mut expected = DeviceBuffer::zeroed(rows * cols).expect("expected");
        let mut actual = DeviceBuffer::zeroed(rows * cols).expect("actual");
        rms_norm_f32_into_on_stream(
            rows,
            cols,
            &input,
            &weight,
            left_norm.output(),
            eps,
            &stream,
        )
        .expect("left norm");
        add_f32_into_on_stream(&left_norm, &residual, expected.output(), &stream)
            .expect("staged residual add");
        rms_norm_add_f32_into_on_stream(
            rows,
            cols,
            &input,
            &weight,
            &residual,
            actual.output(),
            eps,
            &stream,
        )
        .expect("fused residual add");
        assert_f32_buffers_close(&actual, &expected, &stream, 2.0e-6);

        rms_norm_f32_into_on_stream(
            rows,
            cols,
            &right,
            &right_weight,
            right_norm.output(),
            eps,
            &stream,
        )
        .expect("right norm");
        add_f32_into_on_stream(&left_norm, &right_norm, expected.output(), &stream)
            .expect("staged dual add");
        dual_rms_norm_add_f32_into_on_stream(
            rows,
            cols,
            &input,
            &weight,
            eps,
            &right,
            &right_weight,
            eps,
            actual.output(),
            &stream,
        )
        .expect("fused dual add");
        assert_f32_buffers_close(&actual, &expected, &stream, 2.0e-6);

        add_f32_into_on_stream(&left_norm, &residual, expected.output(), &stream)
            .expect("staged scaled add");
        scale_channel_f32_device_row_scalar_in_place_on_stream(
            expected.inout(),
            &channel_scale,
            &row_scale,
            rows,
            cols,
            &stream,
        )
        .expect("staged scale");
        rms_norm_add_channel_row_scale_f32_into_on_stream(
            rows,
            cols,
            &input,
            &weight,
            &residual,
            &channel_scale,
            &row_scale,
            actual.output(),
            eps,
            &stream,
        )
        .expect("fused scaled add");
        assert_f32_buffers_close(&actual, &expected, &stream, 2.0e-6);
    }

    #[test]
    fn fused_gemma_rms_sequences_match_staged_operations() {
        let rows = 3;
        let cols = 96;
        let eps = 1.0e-6;
        let input_scale = 0.375;
        let values = |multiplier: usize, modulus: usize, scale: f32| {
            (0..rows * cols)
                .map(|idx| ((idx * multiplier % modulus) as f32 - modulus as f32 * 0.5) * scale)
                .collect::<Vec<_>>()
        };
        let weights = |offset: f32, modulus: usize, scale: f32| {
            (0..cols)
                .map(|idx| offset + (idx % modulus) as f32 * scale)
                .collect::<Vec<_>>()
        };
        let input = DeviceBuffer::from_host(&values(17, 101, 0.015625)).expect("input upload");
        let right = DeviceBuffer::from_host(&values(29, 107, 0.01171875)).expect("right upload");
        let residual = DeviceBuffer::from_host(&values(7, 61, 0.0078125)).expect("residual upload");
        let input_weight =
            DeviceBuffer::from_host(&weights(0.75, 13, 0.03125)).expect("input weight upload");
        let right_weight =
            DeviceBuffer::from_host(&weights(0.625, 11, 0.0234375)).expect("right weight upload");
        let quant_weight =
            DeviceBuffer::from_host(&weights(0.6875, 17, 0.01953125)).expect("quant weight upload");
        let final_weight =
            DeviceBuffer::from_host(&weights(0.8125, 19, 0.01171875)).expect("final weight upload");
        let channel_scale =
            DeviceBuffer::from_host(&weights(0.875, 7, 0.015625)).expect("channel scale upload");
        let row_scale = DeviceBuffer::from_host(&[0.75, 1.0, 1.25]).expect("row scale upload");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut staged_residual = DeviceBuffer::zeroed(rows * cols).expect("staged residual");
        let mut fused_residual = DeviceBuffer::zeroed(rows * cols).expect("fused residual");
        let mut staged_quant =
            Nvfp4Matrix::zeroed_col_major(cols, rows).expect("staged quantization");
        let mut fused_quant =
            Nvfp4Matrix::zeroed_col_major(cols, rows).expect("fused quantization");
        rms_norm_add_f32_into_on_stream(
            rows,
            cols,
            &input,
            &input_weight,
            &residual,
            staged_residual.output(),
            eps,
            &stream,
        )
        .expect("staged residual add");
        rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
            rows,
            cols,
            &staged_residual,
            &quant_weight,
            &mut staged_quant,
            eps,
            input_scale,
            &stream,
        )
        .expect("staged residual quantization");
        rms_norm_add_then_rms_norm_quantize_nvfp4_f32_into_on_stream(
            rows,
            cols,
            &input,
            &input_weight,
            &residual,
            fused_residual.output(),
            eps,
            &quant_weight,
            &mut fused_quant,
            eps,
            input_scale,
            &stream,
        )
        .expect("fused residual quantization");
        assert_f32_buffers_close(&fused_residual, &staged_residual, &stream, 2.0e-6);
        assert_eq!(
            fused_quant
                .values
                .copy_to_host(&stream)
                .expect("fused quantized values"),
            staged_quant
                .values
                .copy_to_host(&stream)
                .expect("staged quantized values")
        );
        assert_eq!(
            fused_quant
                .scales
                .copy_to_host(&stream)
                .expect("fused quantized scales"),
            staged_quant
                .scales
                .copy_to_host(&stream)
                .expect("staged quantized scales")
        );

        let mut staged_combined = DeviceBuffer::zeroed(rows * cols).expect("staged combined");
        let mut staged_output = DeviceBuffer::zeroed(rows * cols).expect("staged output");
        let mut fused_output = DeviceBuffer::zeroed(rows * cols).expect("fused output");
        dual_rms_norm_add_f32_into_on_stream(
            rows,
            cols,
            &input,
            &input_weight,
            eps,
            &right,
            &right_weight,
            eps,
            staged_combined.output(),
            &stream,
        )
        .expect("staged dual RMSNorm add");
        rms_norm_add_channel_row_scale_f32_into_on_stream(
            rows,
            cols,
            &staged_combined,
            &final_weight,
            &residual,
            &channel_scale,
            &row_scale,
            staged_output.output(),
            eps,
            &stream,
        )
        .expect("staged final residual");
        dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_into_on_stream(
            rows,
            cols,
            &input,
            &input_weight,
            eps,
            &right,
            &right_weight,
            eps,
            &final_weight,
            eps,
            &residual,
            &channel_scale,
            &row_scale,
            fused_output.output(),
            &stream,
        )
        .expect("fused final residual");
        assert_f32_buffers_close(&fused_output, &staged_output, &stream, 2.0e-6);
    }

    fn assert_f32_buffers_close(
        actual: &DeviceBuffer<f32>,
        expected: &DeviceBuffer<f32>,
        stream: &CudaStream,
        tolerance: f32,
    ) {
        let actual = actual.copy_to_host(stream).expect("actual download");
        let expected = expected.copy_to_host(stream).expect("expected download");
        let max_error = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= tolerance, "max_error={max_error}");
    }

    #[test]
    fn fused_gelu_nvfp4_quantization_matches_staged_path() {
        let rows = 3;
        let cols = 96;
        let input_scale = 0.375;
        let gate = (0..rows * cols)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.015625)
            .collect::<Vec<_>>();
        let up = (0..rows * cols)
            .map(|idx| ((idx * 29 % 107) as f32 - 53.0) * 0.01171875)
            .collect::<Vec<_>>();
        let gate = DeviceBuffer::from_host(&gate).expect("gate upload");
        let up = DeviceBuffer::from_host(&up).expect("up upload");
        let mut activated = DeviceBuffer::zeroed(rows * cols).expect("activation allocation");
        let mut expected = Nvfp4Matrix::zeroed_col_major(cols, rows).expect("expected matrix");
        let mut actual = Nvfp4Matrix::zeroed_col_major(cols, rows).expect("actual matrix");
        let stream = CudaStream::new_non_blocking().expect("stream");

        gelu_tanh_mul_f32_into_on_stream(&gate, &up, activated.output(), &stream)
            .expect("staged GELU");
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            cols,
            rows,
            &activated,
            &mut expected,
            input_scale,
            &stream,
        )
        .expect("staged quantization");
        gelu_tanh_mul_quantize_nvfp4_col_major_f32_into_on_stream(
            rows,
            cols,
            &gate,
            &up,
            &mut actual,
            input_scale,
            &stream,
        )
        .expect("fused quantization");

        assert_eq!(
            actual.values.copy_to_host(&stream).expect("actual values"),
            expected
                .values
                .copy_to_host(&stream)
                .expect("expected values")
        );
        assert_eq!(
            actual.scales.copy_to_host(&stream).expect("actual scales"),
            expected
                .scales
                .copy_to_host(&stream)
                .expect("expected scales")
        );
    }

    #[test]
    fn fused_head_unpack_nvfp4_quantization_matches_staged_path() {
        let tokens = 5;
        let heads = 3;
        let head_dim = 32;
        let features = heads * head_dim;
        let output_row_offset = 2;
        let output_rows = output_row_offset + tokens;
        let input_scale = 0.375;
        let token_major = (0..tokens * features)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.015625)
            .collect::<Vec<_>>();
        let mut head_major = vec![0.0; token_major.len()];
        for token in 0..tokens {
            for head in 0..heads {
                for dim in 0..head_dim {
                    head_major[(head * tokens + token) * head_dim + dim] =
                        token_major[(token * heads + head) * head_dim + dim];
                }
            }
        }
        let input = DeviceBuffer::from_host(&head_major).expect("input upload");
        let mut unpacked = DeviceBuffer::zeroed(output_rows * features).expect("unpacked output");
        let mut expected =
            Nvfp4Matrix::zeroed_col_major(features, output_rows).expect("expected matrix");
        let mut actual =
            Nvfp4Matrix::zeroed_col_major(features, output_rows).expect("actual matrix");
        let stream = CudaStream::new_non_blocking().expect("stream");

        unpack_heads_f32_at_offset_into_on_stream(
            &input,
            unpacked.output(),
            tokens,
            heads,
            head_dim,
            output_row_offset,
            &stream,
        )
        .expect("staged unpack");
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            features,
            output_rows,
            &unpacked,
            &mut expected,
            input_scale,
            &stream,
        )
        .expect("staged quantization");
        unpack_heads_quantize_nvfp4_col_major_f32_at_offset_into_on_stream(
            &input,
            &mut actual,
            tokens,
            heads,
            head_dim,
            output_row_offset,
            input_scale,
            &stream,
        )
        .expect("fused unpack quantization");

        assert_eq!(
            actual.values.copy_to_host(&stream).expect("actual values"),
            expected
                .values
                .copy_to_host(&stream)
                .expect("expected values")
        );
        assert_eq!(
            actual.scales.copy_to_host(&stream).expect("actual scales"),
            expected
                .scales
                .copy_to_host(&stream)
                .expect("expected scales")
        );
    }

    #[test]
    fn fused_bf16_head_unpack_nvfp4_quantization_matches_staged_path() {
        let tokens = 3;
        let heads = 2;
        let head_dim = 32;
        let features = heads * head_dim;
        let output_row_offset = 1;
        let output_rows = output_row_offset + tokens;
        let input_scale = 0.375;
        let input = (0..tokens * features)
            .map(|idx| f32_to_bf16(((idx * 17 % 101) as f32 - 50.0) * 0.0137))
            .collect::<Vec<_>>();
        let staged_input = input.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input).expect("BF16 input upload");
        let staged_input = DeviceBuffer::from_host(&staged_input).expect("f32 input upload");
        let mut unpacked = DeviceBuffer::zeroed(output_rows * features).expect("unpacked output");
        let mut expected =
            Nvfp4Matrix::zeroed_col_major(features, output_rows).expect("expected matrix");
        let mut actual =
            Nvfp4Matrix::zeroed_col_major(features, output_rows).expect("actual matrix");
        let stream = CudaStream::new_non_blocking().expect("stream");

        unpack_heads_f32_at_offset_into_on_stream(
            &staged_input,
            unpacked.output(),
            tokens,
            heads,
            head_dim,
            output_row_offset,
            &stream,
        )
        .expect("staged unpack");
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            features,
            output_rows,
            &unpacked,
            &mut expected,
            input_scale,
            &stream,
        )
        .expect("staged quantization");
        unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream(
            &input,
            &mut actual,
            tokens,
            heads,
            head_dim,
            output_row_offset,
            input_scale,
            &stream,
        )
        .expect("fused BF16 unpack quantization");

        assert_eq!(
            actual.values.copy_to_host(&stream).expect("actual values"),
            expected
                .values
                .copy_to_host(&stream)
                .expect("expected values")
        );
        assert_eq!(
            actual.scales.copy_to_host(&stream).expect("actual scales"),
            expected
                .scales
                .copy_to_host(&stream)
                .expect("expected scales")
        );
    }

    #[test]
    fn single_token_gqa_attention_f32_matches_cpu_reference() {
        let q_heads = 8;
        let kv_heads = 2;
        let head_dim = 16;
        let value = (0..kv_heads * head_dim)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.125)
            .collect::<Vec<_>>();

        let value_device = DeviceBuffer::from_host(&value).expect("value upload");
        let key_device = DeviceBuffer::from_host(&value).expect("key upload");
        let output_device =
            single_token_gqa_attention_f32(&key_device, &value_device, q_heads, kv_heads, head_dim)
                .expect("single-token GQA");
        synchronize_device().expect("single-token GQA sync");
        let output = output_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("GQA download");
        let expected = cpu_single_token_gqa(&value, q_heads, kv_heads, head_dim);
        assert_eq!(output, expected);
    }

    #[test]
    fn append_rows_f32_writes_into_destination_offset() {
        let rows = 2;
        let cols = 5;
        let dst_rows = 4;
        let src = (0..rows * cols)
            .map(|idx| 10.0 + idx as f32)
            .collect::<Vec<_>>();
        let src_device = DeviceBuffer::from_host(&src).expect("source upload");
        let mut dst_device =
            DeviceBuffer::<f32>::zeroed(dst_rows * cols).expect("destination alloc");
        let stream = CudaStream::new_non_blocking().expect("stream");

        append_rows_f32_into_on_stream(&src_device, dst_device.output(), 1, rows, cols, &stream)
            .expect("append rows");

        let dst = dst_device
            .copy_to_host(&stream)
            .expect("destination download");
        let mut expected = vec![0.0; dst_rows * cols];
        expected[cols..cols + src.len()].copy_from_slice(&src);
        assert_eq!(dst, expected);
    }

    #[test]
    fn single_token_gqa_attention_f32_from_cache_reads_position() {
        let q_heads = 8;
        let kv_heads = 2;
        let head_dim = 16;
        let kv_width = kv_heads * head_dim;
        let position = 2;
        let mut key_cache = vec![0.0; 4 * kv_width];
        let mut value_cache = vec![0.0; 4 * kv_width];
        for row in 0..4 {
            for col in 0..kv_width {
                key_cache[row * kv_width + col] = 1000.0 + (row * kv_width + col) as f32;
                value_cache[row * kv_width + col] = (row * kv_width + col) as f32 * 0.125;
            }
        }

        let key_device = DeviceBuffer::from_host(&key_cache).expect("key cache upload");
        let value_device = DeviceBuffer::from_host(&value_cache).expect("value cache upload");
        let output_device = single_token_gqa_attention_f32_from_cache(
            &key_device,
            &value_device,
            position,
            q_heads,
            kv_heads,
            head_dim,
        )
        .expect("cached single-token GQA");
        synchronize_device().expect("cached GQA sync");

        let output = output_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("cached GQA download");
        let expected = cpu_single_token_gqa(
            &value_cache[position * kv_width..(position + 1) * kv_width],
            q_heads,
            kv_heads,
            head_dim,
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn cached_gqa_attention_f32_matches_cpu_reference() {
        let q_heads = 8;
        let kv_heads = 2;
        let head_dim = 16;
        let cache_len = 5;
        let query = (0..q_heads * head_dim)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let key_cache = (0..cache_len * kv_heads * head_dim)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.025)
            .collect::<Vec<_>>();
        let value_cache = (0..cache_len * kv_heads * head_dim)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.02)
            .collect::<Vec<_>>();

        let query_device = DeviceBuffer::from_host(&query).expect("query upload");
        let key_device = DeviceBuffer::from_host(&key_cache).expect("key upload");
        let value_device = DeviceBuffer::from_host(&value_cache).expect("value upload");
        let output_device = cached_gqa_attention_f32(
            &query_device,
            &key_device,
            &value_device,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
        )
        .expect("cached GQA attention");
        synchronize_device().expect("cached GQA sync");

        let output = output_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("cached GQA download");
        let expected = cpu_cached_gqa_attention(
            &query,
            &key_cache,
            &value_cache,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
        );
        for (idx, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-6,
                "cached GQA mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
    }

    #[test]
    fn indexed_cached_gqa_attention_matches_qwen36_long_cache() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let q_heads = 16;
        let kv_heads = 2;
        let head_dim = 256;
        let cache_len = 512usize;
        let query = (0..q_heads * head_dim)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.015625)
            .collect::<Vec<_>>();
        let key_cache = (0..cache_len * kv_heads * head_dim)
            .map(|idx| ((idx * 13 % 113) as f32 - 56.0) * 0.0125)
            .collect::<Vec<_>>();
        let value_cache = (0..cache_len * kv_heads * head_dim)
            .map(|idx| ((idx * 19 % 127) as f32 - 63.0) * 0.01)
            .collect::<Vec<_>>();
        let query_device = DeviceBuffer::from_host(&query).expect("query upload");
        let key_device = DeviceBuffer::from_host(&key_cache).expect("key upload");
        let value_device = DeviceBuffer::from_host(&value_cache).expect("value upload");
        let mut output_device =
            DeviceBuffer::<f32>::zeroed(q_heads * head_dim).expect("attention output");
        let cache_len_device =
            DeviceBuffer::from_host(&[cache_len as u32]).expect("cache length upload");

        cached_gqa_attention_f32_indexed_into_on_stream(
            &query_device,
            &key_device,
            &value_device,
            output_device.output(),
            &cache_len_device,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            &stream,
        )
        .expect("indexed cached GQA attention");
        let output = output_device
            .copy_to_host(&stream)
            .expect("attention download");
        let expected = cpu_cached_gqa_attention(
            &query,
            &key_cache,
            &value_cache,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
        );
        let max_error = output
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 2.0e-5, "max error {max_error}");
    }

    #[test]
    fn indexed_decode_primitives_match_host_parameter_variants() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let rows = 5;
        let head_dim = 128;
        let position = 17u32;
        let theta = 1_000_000.0;
        let rope_input = (0..rows * head_dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.0625)
            .collect::<Vec<_>>();
        let rope_input_device = DeviceBuffer::from_host(&rope_input).expect("RoPE input upload");
        let mut indexed_rope = DeviceBuffer::<f32>::zeroed(rope_input.len()).expect("RoPE output");
        let position_device = DeviceBuffer::from_host(&[position]).expect("position upload");
        rope_neox_f32_indexed_into_on_stream(
            rows,
            head_dim,
            &rope_input_device,
            indexed_rope.output(),
            &position_device,
            theta,
            &stream,
        )
        .expect("indexed RoPE");
        let actual_rope = indexed_rope
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("indexed RoPE download");
        let expected_rope = cpu_rope_neox(rows, head_dim, &rope_input, position as usize, theta);
        for (idx, (actual, expected)) in actual_rope.iter().zip(expected_rope.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-5,
                "indexed RoPE mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }

        let rotary_dim = 64;
        let sections = MropeSections {
            v0: 11,
            v1: 11,
            v2: 10,
            v3: 0,
        };
        let positions = [position, position, position, 0];
        let positions_device = DeviceBuffer::from_host(&[position]).expect("position upload");
        let mut host_imrope = DeviceBuffer::<f32>::zeroed(rope_input.len()).expect("host IMRoPE");
        let mut indexed_imrope =
            DeviceBuffer::<f32>::zeroed(rope_input.len()).expect("indexed IMRoPE");
        rope_imrope_f32_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            sections,
            positions,
            &rope_input_device,
            host_imrope.output(),
            theta,
            &stream,
        )
        .expect("host-parameter IMRoPE");
        rope_imrope_f32_indexed_into_on_stream(
            rows,
            head_dim,
            rotary_dim,
            sections,
            &positions_device,
            &rope_input_device,
            indexed_imrope.output(),
            theta,
            &stream,
        )
        .expect("indexed IMRoPE");
        let host_imrope = host_imrope
            .copy_to_host(&stream)
            .expect("host-parameter IMRoPE download");
        let indexed_imrope = indexed_imrope
            .copy_to_host(&stream)
            .expect("indexed IMRoPE download");
        assert_eq!(indexed_imrope, host_imrope);

        let append_rows = 1;
        let append_cols = 8;
        let dst_rows = 4;
        let append_src = (0..append_rows * append_cols)
            .map(|idx| 20.0 + idx as f32)
            .collect::<Vec<_>>();
        let append_src_device = DeviceBuffer::from_host(&append_src).expect("append src upload");
        let mut append_dst_device =
            DeviceBuffer::<f32>::zeroed(dst_rows * append_cols).expect("append dst");
        let dst_start_device = DeviceBuffer::from_host(&[2u32]).expect("append start upload");
        append_rows_f32_indexed_into_on_stream(
            &append_src_device,
            append_dst_device.output(),
            &dst_start_device,
            2,
            append_rows,
            append_cols,
            &stream,
        )
        .expect("indexed append");
        let append_dst = append_dst_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("append dst download");
        let mut expected_append = vec![0.0; dst_rows * append_cols];
        expected_append[2 * append_cols..3 * append_cols].copy_from_slice(&append_src);
        assert_eq!(append_dst, expected_append);

        let q_heads = 8;
        let kv_heads = 2;
        let head_dim = 16;
        let cache_len = 5usize;
        let query = (0..q_heads * head_dim)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let key_cache = (0..cache_len * kv_heads * head_dim)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.025)
            .collect::<Vec<_>>();
        let value_cache = (0..cache_len * kv_heads * head_dim)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.02)
            .collect::<Vec<_>>();
        let query_device = DeviceBuffer::from_host(&query).expect("query upload");
        let key_device = DeviceBuffer::from_host(&key_cache).expect("key upload");
        let value_device = DeviceBuffer::from_host(&value_cache).expect("value upload");
        let mut output_device =
            DeviceBuffer::<f32>::zeroed(q_heads * head_dim).expect("indexed attention output");
        let cache_len_device =
            DeviceBuffer::from_host(&[cache_len as u32]).expect("cache length upload");
        cached_gqa_attention_f32_indexed_into_on_stream(
            &query_device,
            &key_device,
            &value_device,
            output_device.output(),
            &cache_len_device,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
            &stream,
        )
        .expect("indexed cached GQA attention");
        let output = output_device
            .copy_to_host(&stream)
            .expect("indexed cached GQA download");
        let expected = cpu_cached_gqa_attention(
            &query,
            &key_cache,
            &value_cache,
            cache_len,
            q_heads,
            kv_heads,
            head_dim,
        );
        for (idx, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-6,
                "indexed cached GQA mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
    }

    #[test]
    fn prefill_gqa_attention_f32_matches_cpu_reference() {
        let tokens = 3;
        let start_position = 2;
        let q_heads = 8;
        let kv_heads = 2;
        let head_dim = 16;
        let kv_width = kv_heads * head_dim;
        let query = (0..tokens * q_heads * head_dim)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let cache_len = start_position + tokens;
        let key_cache = (0..cache_len * kv_width)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.025)
            .collect::<Vec<_>>();
        let value_cache = (0..cache_len * kv_width)
            .map(|idx| ((idx % 31) as f32 - 15.0) * 0.02)
            .collect::<Vec<_>>();

        let query_device = DeviceBuffer::from_host(&query).expect("prefill query upload");
        let key_device = DeviceBuffer::from_host(&key_cache).expect("prefill key upload");
        let value_device = DeviceBuffer::from_host(&value_cache).expect("prefill value upload");
        let output_device = prefill_gqa_attention_f32(
            &query_device,
            &key_device,
            &value_device,
            tokens,
            start_position,
            q_heads,
            kv_heads,
            head_dim,
        )
        .expect("prefill GQA attention");
        synchronize_device().expect("prefill GQA sync");

        let output = output_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("prefill GQA download");
        let mut expected = Vec::with_capacity(output.len());
        for token in 0..tokens {
            let q_start = token * q_heads * head_dim;
            let q_end = q_start + q_heads * head_dim;
            expected.extend(cpu_cached_gqa_attention(
                &query[q_start..q_end],
                &key_cache,
                &value_cache,
                start_position + token + 1,
                q_heads,
                kv_heads,
                head_dim,
            ));
        }
        for (idx, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-6,
                "prefill GQA mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
    }

    #[test]
    fn ragged_gqa_attention_matches_independent_sequence_reference() {
        const SEQUENCES: usize = 2;
        const TOKENS: usize = 5;
        const Q_HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 8;
        const MAX_TOKENS: usize = 6;
        const PAGE_TOKENS: usize = 4;
        const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
        const QUERY_WIDTH: usize = Q_HEADS * HEAD_DIM;
        let offsets = [0u32, 2];
        let lengths = [2u32, 3];
        let starts = [3u32, 3];
        let query = (0..TOKENS * QUERY_WIDTH)
            .map(|index| ((index * 13 % 47) as f32 - 23.0) * 0.0234375)
            .collect::<Vec<_>>();
        let key = (0..TOKENS * KV_WIDTH)
            .map(|index| ((index * 17 % 53) as f32 - 26.0) * 0.01953125)
            .collect::<Vec<_>>();
        let value = (0..TOKENS * KV_WIDTH)
            .map(|index| ((index * 19 % 59) as f32 - 29.0) * 0.015625)
            .collect::<Vec<_>>();
        let mut expected_keys = (0..SEQUENCES)
            .map(|sequence| {
                (0..MAX_TOKENS * KV_WIDTH)
                    .map(|index| {
                        if index < starts[sequence] as usize * KV_WIDTH {
                            (((index + sequence * 7) * 11 % 43) as f32 - 21.0) * 0.02734375
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut expected_values = (0..SEQUENCES)
            .map(|sequence| {
                (0..MAX_TOKENS * KV_WIDTH)
                    .map(|index| {
                        if index < starts[sequence] as usize * KV_WIDTH {
                            (((index + sequence * 5) * 7 % 41) as f32 - 20.0) * 0.03125
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let initial_keys = expected_keys.clone();
        let initial_values = expected_values.clone();
        for sequence in 0..SEQUENCES {
            let begin = offsets[sequence] as usize;
            for local in 0..lengths[sequence] as usize {
                let source = (begin + local) * KV_WIDTH;
                let destination = (starts[sequence] as usize + local) * KV_WIDTH;
                expected_keys[sequence][destination..destination + KV_WIDTH]
                    .copy_from_slice(&key[source..source + KV_WIDTH]);
                expected_values[sequence][destination..destination + KV_WIDTH]
                    .copy_from_slice(&value[source..source + KV_WIDTH]);
            }
        }

        let stream = CudaStream::new_non_blocking().expect("stream");
        let query_device = DeviceBuffer::from_host(&query).expect("query");
        let key_device = DeviceBuffer::from_host(&key).expect("key");
        let value_device = DeviceBuffer::from_host(&value).expect("value");
        let offsets_device = DeviceBuffer::from_host(&offsets).expect("offsets");
        let lengths_device = DeviceBuffer::from_host(&lengths).expect("lengths");
        let starts_device = DeviceBuffer::from_host(&starts).expect("starts");
        let mut key_caches = initial_keys
            .iter()
            .map(|cache| DeviceBuffer::from_host(cache).expect("key cache"))
            .collect::<Vec<_>>();
        let mut value_caches = initial_values
            .iter()
            .map(|cache| DeviceBuffer::from_host(cache).expect("value cache"))
            .collect::<Vec<_>>();
        let key_table = DeviceBuffer::from_host(
            &key_caches
                .iter_mut()
                .map(|cache| cache.as_mut_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("key table");
        let value_table = DeviceBuffer::from_host(
            &value_caches
                .iter_mut()
                .map(|cache| cache.as_mut_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("value table");
        append_ragged_kv_f32_into_on_stream(
            &key_device,
            &value_device,
            &key_table,
            &value_table,
            0,
            &offsets_device,
            &lengths_device,
            &starts_device,
            SEQUENCES,
            TOKENS,
            KV_WIDTH,
            &stream,
        )
        .expect("ragged KV append");
        let mut actual = DeviceBuffer::zeroed(TOKENS * QUERY_WIDTH).expect("output");
        ragged_gqa_attention_f32_into_on_stream(
            &query_device,
            &key_table,
            &value_table,
            0,
            &offsets_device,
            &lengths_device,
            &starts_device,
            actual.output(),
            SEQUENCES,
            TOKENS,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("ragged GQA");

        for sequence in 0..SEQUENCES {
            assert_eq!(
                key_caches[sequence]
                    .copy_to_host(&stream)
                    .expect("key cache download"),
                expected_keys[sequence],
            );
            assert_eq!(
                value_caches[sequence]
                    .copy_to_host(&stream)
                    .expect("value cache download"),
                expected_values[sequence],
            );
        }
        let actual = actual.copy_to_host(&stream).expect("output download");
        let mut expected = Vec::with_capacity(actual.len());
        for sequence in 0..SEQUENCES {
            let begin = offsets[sequence] as usize;
            for local in 0..lengths[sequence] as usize {
                let row = begin + local;
                expected.extend(cpu_cached_gqa_attention(
                    &query[row * QUERY_WIDTH..(row + 1) * QUERY_WIDTH],
                    &expected_keys[sequence],
                    &expected_values[sequence],
                    starts[sequence] as usize + local + 1,
                    Q_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                ));
            }
        }
        assert_close(&actual, &expected, 2.0e-6, "ragged GQA");

        let page_slots = 4;
        let page_table_rows = [[2u32, 0], [3u32, 1]];
        let page_tables = page_table_rows
            .iter()
            .map(|table| DeviceBuffer::from_host(table).expect("page table"))
            .collect::<Vec<_>>();
        let page_table_ptrs = DeviceBuffer::from_host(
            &page_tables
                .iter()
                .map(|table| table.as_const_ptr().cast::<u32>())
                .collect::<Vec<_>>(),
        )
        .expect("page table pointers");
        let mut paged_keys = vec![0.0; page_slots * PAGE_TOKENS * KV_WIDTH];
        let mut paged_values = vec![0.0; page_slots * PAGE_TOKENS * KV_WIDTH];
        for sequence in 0..SEQUENCES {
            for logical_row in 0..MAX_TOKENS {
                let slot = page_table_rows[sequence][logical_row / PAGE_TOKENS] as usize;
                let physical_row = slot * PAGE_TOKENS + logical_row % PAGE_TOKENS;
                let source = logical_row * KV_WIDTH;
                let destination = physical_row * KV_WIDTH;
                paged_keys[destination..destination + KV_WIDTH]
                    .copy_from_slice(&initial_keys[sequence][source..source + KV_WIDTH]);
                paged_values[destination..destination + KV_WIDTH]
                    .copy_from_slice(&initial_values[sequence][source..source + KV_WIDTH]);
            }
        }
        let mut paged_keys = DeviceBuffer::from_host(&paged_keys).expect("paged keys");
        let mut paged_values = DeviceBuffer::from_host(&paged_values).expect("paged values");
        append_ragged_paged_kv_f32_into_on_stream(
            &key_device,
            &value_device,
            &mut paged_keys,
            &mut paged_values,
            &page_table_ptrs,
            &offsets_device,
            &lengths_device,
            &starts_device,
            SEQUENCES,
            TOKENS,
            PAGE_TOKENS,
            KV_WIDTH,
            &stream,
        )
        .expect("ragged paged KV append");
        let mut paged_output = DeviceBuffer::zeroed(TOKENS * QUERY_WIDTH).expect("paged output");
        ragged_paged_gqa_attention_f32_into_on_stream(
            &query_device,
            &paged_keys,
            &paged_values,
            &page_table_ptrs,
            &offsets_device,
            &lengths_device,
            &starts_device,
            paged_output.output(),
            SEQUENCES,
            TOKENS,
            PAGE_TOKENS,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("ragged paged GQA");
        let paged_output = paged_output
            .copy_to_host(&stream)
            .expect("paged output download");
        assert_close(&paged_output, &expected, 2.0e-6, "ragged paged GQA");
    }

    #[test]
    fn bf16_linear_argmax_f32_matches_cpu_reference() {
        let rows = 17;
        let cols = 19;
        let input = (0..cols)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let weight_f32 = (0..rows * cols)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight_bf16 = weight_f32
            .iter()
            .map(|value| format::f32_to_bf16(*value))
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight_bf16).expect("weight upload");
        let actual =
            bf16_linear_argmax_f32(&input_device, &weight_device, rows, cols).expect("BF16 argmax");

        let mut expected_index = 0;
        let mut expected_value = f32::NEG_INFINITY;
        for row in 0..rows {
            let mut logit = 0.0;
            for col in 0..cols {
                logit += input[col] * format::bf16_to_f32(weight_bf16[row * cols + col]);
            }
            if logit > expected_value {
                expected_value = logit;
                expected_index = row as u32;
            }
        }

        assert_eq!(actual.index, expected_index);
        assert!(
            (actual.value - expected_value).abs() <= 1.0e-6,
            "logit mismatch: actual={} expected={}",
            actual.value,
            expected_value
        );
    }

    #[test]
    fn bf16_linear_logits_f32_matches_cpu_reference() {
        let rows = 17;
        let cols = 19;
        let input = (0..cols)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let weight_f32 = (0..rows * cols)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight_bf16 = weight_f32
            .iter()
            .map(|value| format::f32_to_bf16(*value))
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight_bf16).expect("weight upload");
        let logits_device =
            bf16_linear_logits_f32(&input_device, &weight_device, rows, cols).expect("BF16 logits");
        synchronize_device().expect("BF16 logits sync");
        let logits = logits_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("logits download");
        let argmax =
            bf16_linear_argmax_f32(&input_device, &weight_device, rows, cols).expect("BF16 argmax");

        let mut expected = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut logit = 0.0;
            for col in 0..cols {
                logit += input[col] * format::bf16_to_f32(weight_bf16[row * cols + col]);
            }
            expected.push(logit);
        }

        for (idx, (actual, expected)) in logits.iter().zip(expected.iter()).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 1.0e-6,
                "logit mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
        }
        let expected_argmax = expected
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(idx, value)| (idx as u32, *value))
            .unwrap();
        assert_eq!(argmax.index, expected_argmax.0);
        assert!((argmax.value - expected_argmax.1).abs() <= 1.0e-6);
    }

    #[test]
    fn bf16_linear_supports_laguna_dense_down_width() {
        let rows = 2;
        let cols = 12_288;
        let input = DeviceBuffer::from_host(&vec![1.0f32; cols]).expect("input upload");
        let weight = DeviceBuffer::from_host(&vec![format::f32_to_bf16(0.0); rows * cols])
            .expect("weight upload");
        let logits =
            bf16_linear_logits_f32(&input, &weight, rows, cols).expect("Laguna-width BF16 logits");
        let stream = CudaStream::new_blocking().expect("copy stream");
        assert_eq!(
            logits
                .copy_to_host(&stream)
                .expect("logits download")
                .as_ref(),
            &[0.0, 0.0],
        );
    }

    #[test]
    fn bf16_linear_pair_matches_separate_projections() {
        let cols = 19usize;
        let rows = [5usize, 3];
        let input = (0..cols)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let weights = rows
            .iter()
            .enumerate()
            .map(|(segment, rows)| {
                (0..rows * cols)
                    .map(|idx| {
                        format::f32_to_bf16((((idx + segment * 7) % 23) as f32 - 11.0) * 0.03125)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_devices = weights
            .iter()
            .map(|weight| DeviceBuffer::from_host(weight).expect("weight upload"))
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream create");
        let mut expected = rows
            .iter()
            .map(|rows| DeviceBuffer::<f32>::zeroed(*rows).expect("expected alloc"))
            .collect::<Vec<_>>();
        for segment in 0..2 {
            bf16_linear_logits_f32_into_on_stream(
                &input_device,
                &weight_devices[segment],
                expected[segment].output(),
                rows[segment],
                cols,
                &stream,
            )
            .expect("separate BF16 projection");
        }
        let expected = expected
            .iter()
            .map(|output| {
                output
                    .copy_to_host(&stream)
                    .expect("expected download")
                    .as_slice()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        let mut actual = rows
            .iter()
            .map(|rows| DeviceBuffer::<f32>::zeroed(*rows).expect("actual alloc"))
            .collect::<Vec<_>>();
        let (first, second) = actual.split_at_mut(1);
        bf16_linear_pair_logits_f32_into_on_stream(
            &input_device,
            &weight_devices[0],
            &weight_devices[1],
            first[0].output(),
            second[0].output(),
            rows[0],
            rows[1],
            cols,
            &stream,
        )
        .expect("paired BF16 projections");
        for segment in 0..2 {
            let actual = actual[segment]
                .copy_to_host(&stream)
                .expect("actual download");
            assert_close(
                &actual,
                &expected[segment],
                1.0e-6,
                &format!("BF16 linear pair segment {segment}"),
            );
        }
    }

    #[test]
    fn bf16_linear_batch_matches_independent_rows() {
        let batch_size = 3usize;
        let rows = 5usize;
        let cols = 19usize;
        let input = (0..batch_size * cols)
            .map(|idx| ((idx * 7 % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>();
        let weight = (0..rows * cols)
            .map(|idx| format::f32_to_bf16(((idx * 11 % 37) as f32 - 18.0) * 0.03125))
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight");
        let mut actual = DeviceBuffer::zeroed(batch_size * rows).expect("actual");
        let stream = CudaStream::new_non_blocking().expect("stream");
        bf16_linear_logits_f32_batch_into_on_stream(
            &input_device,
            &weight_device,
            actual.output(),
            batch_size,
            rows,
            cols,
            &stream,
        )
        .expect("batched projection");
        let actual = actual.copy_to_host(&stream).expect("actual download");
        for batch in 0..batch_size {
            let row_input = DeviceBuffer::from_host(&input[batch * cols..(batch + 1) * cols])
                .expect("row input");
            let mut expected = DeviceBuffer::zeroed(rows).expect("expected");
            bf16_linear_logits_f32_into_on_stream(
                &row_input,
                &weight_device,
                expected.output(),
                rows,
                cols,
                &stream,
            )
            .expect("row projection");
            assert_close(
                &actual[batch * rows..(batch + 1) * rows],
                &expected.copy_to_host(&stream).expect("expected download"),
                1.0e-6,
                "batched BF16 linear",
            );
        }
    }

    #[test]
    fn lm_head_top1_f32_batch_matches_materialized_logits_exactly() {
        let batch_size = 3usize;
        let rows = 19usize;
        let cols = 20usize;
        let input = (0..batch_size * cols)
            .map(|idx| 1.0 + (idx * 7 % 31) as f32 * 0.015625)
            .collect::<Vec<_>>();
        let mut weight = (0..rows * cols)
            .map(|idx| format::f32_to_bf16(((idx * 11 % 41) as f32 - 20.0) * 0.03125))
            .collect::<Vec<_>>();
        for col in 0..cols {
            let tied = format::f32_to_bf16(16.0 + col as f32 * 0.125);
            weight[2 * cols + col] = tied;
            weight[7 * cols + col] = tied;
        }

        let input = DeviceBuffer::from_host(&input).expect("input");
        let weight = DeviceBuffer::from_host(&weight).expect("weight");
        let mut logits = DeviceBuffer::<f32>::zeroed(batch_size * rows).expect("logits");
        let mut expected_index = DeviceBuffer::<u32>::zeroed(batch_size).expect("expected index");
        let mut expected_value = DeviceBuffer::<f32>::zeroed(batch_size).expect("expected value");
        let scratch_len = batch_size * rows.div_ceil(8);
        let scratch_value = DeviceBuffer::<f32>::zeroed(scratch_len).expect("scratch value");
        let scratch_index = DeviceBuffer::<u32>::zeroed(scratch_len).expect("scratch index");
        let actual_index = DeviceBuffer::<u32>::zeroed(batch_size).expect("actual index");
        let actual_value = DeviceBuffer::<f32>::zeroed(batch_size).expect("actual value");
        let stream = CudaStream::new_non_blocking().expect("stream");

        bf16_linear_logits_f32_batch_into_on_stream(
            &input,
            &weight,
            logits.output(),
            batch_size,
            rows,
            cols,
            &stream,
        )
        .expect("materialized logits");
        argmax_f32_batch_into_on_stream(
            &logits,
            expected_index.output(),
            expected_value.output(),
            batch_size,
            rows,
            &stream,
        )
        .expect("materialized argmax");
        lm_head_top1_f32_batch_into_on_stream(
            &input,
            &weight,
            &scratch_value,
            &scratch_index,
            &actual_index,
            &actual_value,
            batch_size,
            rows,
            cols,
            &stream,
        )
        .expect("direct top1");

        let expected_index = expected_index
            .copy_to_host(&stream)
            .expect("expected indices");
        assert_eq!(expected_index, vec![2; batch_size]);
        assert_eq!(
            actual_index.copy_to_host(&stream).expect("actual indices"),
            expected_index
        );
        assert_eq!(
            actual_value.copy_to_host(&stream).expect("actual values"),
            expected_value
                .copy_to_host(&stream)
                .expect("expected values")
        );
    }

    #[test]
    fn lm_head_top1_f32_matches_cpu_reference_small() {
        // Small case: rows not a multiple of 8 to exercise the padding path.
        let rows = 17;
        let cols = 20;
        let input = vec![1.0f32; cols];
        let weight_f32 = (0..rows * cols)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        // Inject a controlled winner at row 5.
        let winner_row = 5usize;
        let mut weight_f32 = weight_f32;
        for col in 0..cols {
            weight_f32[winner_row * cols + col] = 1000.0;
        }
        let weight_bf16 = weight_f32
            .iter()
            .map(|value| format::f32_to_bf16(*value))
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight_bf16).expect("weight upload");
        let scratch_len = rows.div_ceil(8) * 8;
        let scratch_value = DeviceBuffer::<f32>::zeroed(scratch_len).expect("scratch1 alloc");
        let scratch_index = DeviceBuffer::<u32>::zeroed(scratch_len).expect("scratch2 alloc");
        let out_index = DeviceBuffer::<u32>::zeroed(1).expect("out_index alloc");
        let out_value = DeviceBuffer::<f32>::zeroed(1).expect("out_value alloc");

        let stream = CudaStream::new_non_blocking().expect("stream create");
        lm_head_top1_f32_into_on_stream(
            &input_device,
            &weight_device,
            &scratch_value,
            &scratch_index,
            &out_index,
            &out_value,
            rows,
            cols,
            &stream,
        )
        .expect("lm-head top1 enqueued");

        let index = out_index.copy_to_host(&stream).expect("download index")[0];
        let value = out_value.copy_to_host(&stream).expect("download value")[0];
        assert_eq!(index, winner_row as u32);

        let mut expected = 0.0f32;
        for col in 0..cols {
            expected += input[col] * format::bf16_to_f32(weight_bf16[winner_row * cols + col]);
        }
        assert!(
            (value - expected).abs() <= 1.0,
            "logit mismatch: actual={} expected={}",
            value,
            expected
        );
    }

    #[test]
    fn lm_head_top1_f32_matches_cpu_reference_full_vocab() {
        // Full Qwen3 vocab/hidden shape, controlled max at row 12345.
        let rows = 151_936;
        let cols = 4_096;
        // Use identity input: input[col] = 1 for all col. That makes logit = sum
        // of weight row. Then we boost row 12345 by adding a large positive
        // constant to every col.
        let input = vec![1.0f32; cols];
        let winner_row = 12_345usize;
        let weight_f32 = (0..rows * cols)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.03125)
            .collect::<Vec<_>>();
        let mut weight_f32 = weight_f32;
        // Big positive constant on every col of winner row.
        for col in 0..cols {
            weight_f32[winner_row * cols + col] = 1000.0;
        }
        let weight_bf16 = weight_f32
            .iter()
            .map(|value| format::f32_to_bf16(*value))
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight_bf16).expect("weight upload");
        let scratch_len = rows.div_ceil(8) * 8;
        let scratch_value = DeviceBuffer::<f32>::zeroed(scratch_len).expect("scratch1 alloc");
        let scratch_index = DeviceBuffer::<u32>::zeroed(scratch_len).expect("scratch2 alloc");
        let out_index = DeviceBuffer::<u32>::zeroed(1).expect("out_index alloc");
        let out_value = DeviceBuffer::<f32>::zeroed(1).expect("out_value alloc");

        let stream = CudaStream::new_non_blocking().expect("stream create");
        lm_head_top1_f32_into_on_stream(
            &input_device,
            &weight_device,
            &scratch_value,
            &scratch_index,
            &out_index,
            &out_value,
            rows,
            cols,
            &stream,
        )
        .expect("lm-head top1 enqueued");

        let index = out_index.copy_to_host(&stream).expect("download index")[0];
        let value = out_value.copy_to_host(&stream).expect("download value")[0];
        assert_eq!(index, winner_row as u32);
        // Reference logit for the winner row.
        let mut expected = 0.0f32;
        for col in 0..cols {
            expected += input[col] * format::bf16_to_f32(weight_bf16[winner_row * cols + col]);
        }
        assert!(
            (value - expected).abs() <= 1.0,
            "winner logit mismatch: actual={} expected={}",
            value,
            expected
        );
    }

    #[test]
    fn bf16_matrix_to_f32_matches_cpu_reference() {
        let rows = 7;
        let cols = 3;
        let values = (0..rows * cols)
            .map(|idx| format::f32_to_bf16(((idx % 13) as f32 - 6.0) * 0.125))
            .collect::<Vec<_>>();
        let matrix = Bf16Matrix::from_bf16_host(rows, cols, &values).expect("matrix upload");
        let output_device = bf16_matrix_to_f32(&matrix).expect("BF16 to f32");
        synchronize_device().expect("BF16 to f32 sync");
        let output = output_device
            .copy_to_host(&CudaStream::new_blocking().expect("copy stream"))
            .expect("download");
        let expected = values
            .iter()
            .map(|value| format::bf16_to_f32(*value))
            .collect::<Vec<_>>();
        assert_eq!(output, expected);
    }

    #[test]
    fn gated_delta_net_128_matches_cpu_reference() {
        let heads = 2usize;
        let len = heads * 128;
        let state_len = heads * 128 * 128;
        let q = (0..len)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let k = (0..len)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.025)
            .collect::<Vec<_>>();
        let v = (0..len)
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.02)
            .collect::<Vec<_>>();
        let gate = vec![-0.25f32, -0.5];
        let beta = vec![0.2f32, 0.75];
        let state = (0..state_len)
            .map(|idx| ((idx % 29) as f32 - 14.0) * 0.003)
            .collect::<Vec<_>>();

        let mut expected_state = state.clone();
        let expected_output =
            cpu_gated_delta_net_128(&q, &k, &v, &gate, &beta, &mut expected_state, heads);

        let q_device = DeviceBuffer::from_host(&q).expect("q upload");
        let k_device = DeviceBuffer::from_host(&k).expect("k upload");
        let v_device = DeviceBuffer::from_host(&v).expect("v upload");
        let gate_device = DeviceBuffer::from_host(&gate).expect("gate upload");
        let beta_device = DeviceBuffer::from_host(&beta).expect("beta upload");
        let mut state_device = DeviceBuffer::from_host(&state).expect("state upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(len).expect("output alloc");
        let stream = CudaStream::new_blocking().expect("stream create");

        gated_delta_net_128_f32_into_on_stream(
            &q_device,
            &k_device,
            &v_device,
            &gate_device,
            &beta_device,
            state_device.inout(),
            output_device.output(),
            heads,
            &stream,
        )
        .expect("gated delta net enqueue");

        let output = output_device
            .copy_to_host(&stream)
            .expect("output download");
        let actual_state = state_device.copy_to_host(&stream).expect("state download");
        assert_close(&output, &expected_output, 2.0e-6, "gdn output");
        assert_close(&actual_state, &expected_state, 2.0e-6, "gdn state");
    }

    #[test]
    fn ling3_kda_primitives_match_cpu_reference() {
        let heads = 2usize;
        let len = heads * 128;
        let state_len = heads * 128 * 128;
        let mut q = (0..len)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.0075)
            .collect::<Vec<_>>();
        let mut k = (0..len)
            .map(|idx| ((idx * 13 % 103) as f32 - 51.0) * 0.00625)
            .collect::<Vec<_>>();
        normalize_ling3_heads_128(&mut q, heads);
        normalize_ling3_heads_128(&mut k, heads);
        let v = (0..len)
            .map(|idx| ((idx * 19 % 107) as f32 - 53.0) * 0.01)
            .collect::<Vec<_>>();
        let raw_gate = (0..len)
            .map(|idx| ((idx * 23 % 109) as f32 - 54.0) * 0.0125)
            .collect::<Vec<_>>();
        let beta_input = vec![-0.75f32, 1.25];
        let a_log = vec![-0.5f32, 0.35];
        let dt_bias = (0..len)
            .map(|idx| ((idx * 29 % 113) as f32 - 56.0) * 0.005)
            .collect::<Vec<_>>();
        let state = (0..state_len)
            .map(|idx| ((idx * 31 % 127) as f32 - 63.0) * 0.0005)
            .collect::<Vec<_>>();
        let lower_bound = -5.0f32;
        let (expected_gate, expected_beta) =
            cpu_ling3_kda_gate(&raw_gate, &beta_input, &a_log, &dt_bias, heads, lower_bound);
        let mut expected_state = state.clone();
        let expected_output = cpu_ling3_kda_128(
            &q,
            &k,
            &v,
            &expected_gate,
            &expected_beta,
            &mut expected_state,
            heads,
        );

        let q = DeviceBuffer::from_host(&q).expect("q upload");
        let k = DeviceBuffer::from_host(&k).expect("k upload");
        let v = DeviceBuffer::from_host(&v).expect("v upload");
        let raw_gate = DeviceBuffer::from_host(&raw_gate).expect("raw gate upload");
        let beta_input = DeviceBuffer::from_host(&beta_input).expect("beta input upload");
        let a_log = DeviceBuffer::from_host(&a_log).expect("A log upload");
        let dt_bias = DeviceBuffer::from_host(&dt_bias).expect("dt bias upload");
        let mut gate = DeviceBuffer::zeroed(len).expect("gate allocation");
        let mut beta = DeviceBuffer::zeroed(heads).expect("beta allocation");
        let mut state = DeviceBuffer::from_host(&state).expect("state upload");
        let mut output = DeviceBuffer::zeroed(len).expect("output allocation");
        let stream = CudaStream::new_non_blocking().expect("stream");

        ling3_kda_gate_f32_into_on_stream(
            &raw_gate,
            &beta_input,
            &a_log,
            &dt_bias,
            gate.output(),
            beta.output(),
            heads,
            lower_bound,
            &stream,
        )
        .expect("Ling KDA gate");
        ling3_kda_128_f32_into_on_stream(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            state.inout(),
            output.output(),
            heads,
            &stream,
        )
        .expect("Ling KDA recurrence");

        assert_close(
            &gate.copy_to_host(&stream).expect("gate download"),
            &expected_gate,
            2.0e-6,
            "Ling KDA gate",
        );
        assert_close(
            &beta.copy_to_host(&stream).expect("beta download"),
            &expected_beta,
            2.0e-6,
            "Ling KDA beta",
        );
        assert_close(
            &output.copy_to_host(&stream).expect("output download"),
            &expected_output,
            4.0e-6,
            "Ling KDA output",
        );
        assert_close(
            &state.copy_to_host(&stream).expect("state download"),
            &expected_state,
            4.0e-6,
            "Ling KDA state",
        );
    }

    #[test]
    fn ling3_chunked_gate_and_recurrence_match_repeated_tokens() {
        const ROWS: usize = 3;
        const HEADS: usize = 2;
        let width = HEADS * 128;
        let state_len = width * 128;
        let q = (0..ROWS * width)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.003)
            .collect::<Vec<_>>();
        let k = (0..ROWS * width)
            .map(|index| ((index * 13 % 103) as f32 - 51.0) * 0.0025)
            .collect::<Vec<_>>();
        let v = (0..ROWS * width)
            .map(|index| ((index * 19 % 107) as f32 - 53.0) * 0.004)
            .collect::<Vec<_>>();
        let raw_gate = (0..ROWS * width)
            .map(|index| ((index * 23 % 109) as f32 - 54.0) * 0.006)
            .collect::<Vec<_>>();
        let beta_input = (0..ROWS * HEADS)
            .map(|index| index as f32 * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let a_log = vec![-0.5f32, 0.35];
        let dt_bias = (0..width)
            .map(|index| ((index * 29 % 113) as f32 - 56.0) * 0.005)
            .collect::<Vec<_>>();
        let initial_state = (0..state_len)
            .map(|index| ((index * 31 % 127) as f32 - 63.0) * 0.0005)
            .collect::<Vec<_>>();
        let lower_bound = -5.0f32;
        let stream = CudaStream::new_non_blocking().expect("stream");
        let q = DeviceBuffer::from_host(&q).expect("Q");
        let k = DeviceBuffer::from_host(&k).expect("K");
        let v = DeviceBuffer::from_host(&v).expect("V");
        let raw_gate = DeviceBuffer::from_host(&raw_gate).expect("raw gate");
        let beta_input = DeviceBuffer::from_host(&beta_input).expect("beta input");
        let a_log = DeviceBuffer::from_host(&a_log).expect("A log");
        let dt_bias = DeviceBuffer::from_host(&dt_bias).expect("dt bias");

        let mut chunk_gate = DeviceBuffer::zeroed(ROWS * width).expect("chunk gate");
        let mut chunk_beta = DeviceBuffer::zeroed(ROWS * HEADS).expect("chunk beta");
        ling3_kda_gate_f32_batch_into_on_stream(
            &raw_gate,
            &beta_input,
            &a_log,
            &dt_bias,
            chunk_gate.output(),
            chunk_beta.output(),
            ROWS,
            HEADS,
            lower_bound,
            &stream,
        )
        .expect("chunk gate");
        let mut chunk_state = DeviceBuffer::from_host(&initial_state).expect("chunk state");
        let mut chunk_output = DeviceBuffer::zeroed(ROWS * width).expect("chunk output");
        ling3_kda_128_f32_chunks_into_on_stream(
            &q,
            &k,
            &v,
            &chunk_gate,
            &chunk_beta,
            chunk_state.inout(),
            chunk_output.output(),
            ROWS,
            HEADS,
            &stream,
        )
        .expect("chunk recurrence");

        let gate_host = chunk_gate.copy_to_host(&stream).expect("gate read");
        let beta_host = chunk_beta.copy_to_host(&stream).expect("beta read");
        let q_host = q.copy_to_host(&stream).expect("Q read");
        let k_host = k.copy_to_host(&stream).expect("K read");
        let v_host = v.copy_to_host(&stream).expect("V read");
        let mut expected_state = initial_state;
        let mut expected_output = Vec::with_capacity(ROWS * width);
        let mut repeated_state = DeviceBuffer::from_host(&expected_state).expect("repeated state");
        let mut repeated_output = Vec::with_capacity(ROWS * width);
        for row in 0..ROWS {
            let raw_gate_row = DeviceBuffer::from_host(
                &raw_gate.copy_to_host(&stream).unwrap()[row * width..(row + 1) * width],
            )
            .expect("raw gate row");
            let beta_input_row = DeviceBuffer::from_host(
                &beta_input.copy_to_host(&stream).unwrap()[row * HEADS..(row + 1) * HEADS],
            )
            .expect("beta input row");
            let mut scalar_gate = DeviceBuffer::zeroed(width).expect("scalar gate");
            let mut scalar_beta = DeviceBuffer::zeroed(HEADS).expect("scalar beta");
            ling3_kda_gate_f32_into_on_stream(
                &raw_gate_row,
                &beta_input_row,
                &a_log,
                &dt_bias,
                scalar_gate.output(),
                scalar_beta.output(),
                HEADS,
                lower_bound,
                &stream,
            )
            .expect("scalar gate");
            assert_close(
                &gate_host[row * width..(row + 1) * width],
                &scalar_gate.copy_to_host(&stream).unwrap(),
                0.0,
                "batched Ling gate",
            );
            assert_close(
                &beta_host[row * HEADS..(row + 1) * HEADS],
                &scalar_beta.copy_to_host(&stream).unwrap(),
                0.0,
                "batched Ling beta",
            );
            let q_row =
                DeviceBuffer::from_host(&q_host[row * width..(row + 1) * width]).expect("Q row");
            let k_row =
                DeviceBuffer::from_host(&k_host[row * width..(row + 1) * width]).expect("K row");
            let v_row =
                DeviceBuffer::from_host(&v_host[row * width..(row + 1) * width]).expect("V row");
            let mut output_row = DeviceBuffer::zeroed(width).expect("output row");
            ling3_kda_128_f32_into_on_stream(
                &q_row,
                &k_row,
                &v_row,
                &scalar_gate,
                &scalar_beta,
                repeated_state.inout(),
                output_row.output(),
                HEADS,
                &stream,
            )
            .expect("scalar recurrence");
            repeated_output.extend(output_row.copy_to_host(&stream).unwrap().iter().copied());
            expected_output.extend(cpu_ling3_kda_128(
                &q_host[row * width..(row + 1) * width],
                &k_host[row * width..(row + 1) * width],
                &v_host[row * width..(row + 1) * width],
                &gate_host[row * width..(row + 1) * width],
                &beta_host[row * HEADS..(row + 1) * HEADS],
                &mut expected_state,
                HEADS,
            ));
        }
        assert_close(
            &chunk_output.copy_to_host(&stream).expect("output read"),
            &expected_output,
            5.0e-6,
            "chunked Ling KDA output",
        );
        assert_close(
            &chunk_output.copy_to_host(&stream).expect("output read"),
            &repeated_output,
            0.0,
            "chunked versus scalar Ling KDA output",
        );
        assert_close(
            &chunk_state.copy_to_host(&stream).expect("state read"),
            &repeated_state
                .copy_to_host(&stream)
                .expect("repeated state read"),
            0.0,
            "chunked versus scalar Ling KDA state",
        );
        assert_close(
            &chunk_state.copy_to_host(&stream).expect("state read"),
            &expected_state,
            5.0e-6,
            "chunked Ling KDA state",
        );
    }

    #[test]
    fn ling3_kda_prep_matches_causal_convolution_reference() {
        let heads = 2usize;
        let projection = heads * 128;
        let conv_dim = projection * 3;
        let qkv = (0..conv_dim)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.00625)
            .collect::<Vec<_>>();
        let conv_weight = (0..conv_dim * 4)
            .map(|idx| f32_to_bf16(((idx * 13 % 97) as f32 - 48.0) * 0.01))
            .collect::<Vec<_>>();
        let conv_state = (0..conv_dim * 3)
            .map(|idx| ((idx * 19 % 103) as f32 - 51.0) * 0.0025)
            .collect::<Vec<_>>();
        let (expected_q, expected_k, expected_v, expected_state) =
            cpu_ling3_kda_prep(&qkv, &conv_weight, &conv_state, heads);

        let qkv = DeviceBuffer::from_host(&qkv).expect("QKV upload");
        let conv_weight = DeviceBuffer::from_host(&conv_weight).expect("conv weight upload");
        let mut conv_state = DeviceBuffer::from_host(&conv_state).expect("conv state upload");
        let mut q = DeviceBuffer::zeroed(projection).expect("Q allocation");
        let mut k = DeviceBuffer::zeroed(projection).expect("K allocation");
        let mut v = DeviceBuffer::zeroed(projection).expect("V allocation");
        let stream = CudaStream::new_non_blocking().expect("stream");
        ling3_kda_prep_into_on_stream(
            &qkv,
            &conv_weight,
            q.output(),
            k.output(),
            v.output(),
            conv_state.inout(),
            heads,
            &stream,
        )
        .expect("Ling KDA preparation");

        assert_close(
            &q.copy_to_host(&stream).expect("Q download"),
            &expected_q,
            2.0e-6,
            "Ling normalized Q",
        );
        assert_close(
            &k.copy_to_host(&stream).expect("K download"),
            &expected_k,
            2.0e-6,
            "Ling normalized K",
        );
        assert_close(
            &v.copy_to_host(&stream).expect("V download"),
            &expected_v,
            2.0e-6,
            "Ling convolved V",
        );
        assert_close(
            &conv_state
                .copy_to_host(&stream)
                .expect("conv state download"),
            &expected_state,
            0.0,
            "Ling convolution state",
        );
    }

    #[test]
    fn ling3_contiguous_prep_matches_repeated_causal_convolution() {
        const ROWS: usize = 5;
        const HEADS: usize = 2;
        let projection = HEADS * 128;
        let conv_width = projection * 3;
        let qkv = (0..ROWS * conv_width)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.00625)
            .collect::<Vec<_>>();
        let conv_weight = (0..conv_width * 4)
            .map(|index| f32_to_bf16(((index * 13 % 97) as f32 - 48.0) * 0.01))
            .collect::<Vec<_>>();
        let initial_state = (0..conv_width * 3)
            .map(|index| ((index * 19 % 103) as f32 - 51.0) * 0.0025)
            .collect::<Vec<_>>();
        let mut expected_state = initial_state.clone();
        let mut expected_q = Vec::with_capacity(ROWS * projection);
        let mut expected_k = Vec::with_capacity(ROWS * projection);
        let mut expected_v = Vec::with_capacity(ROWS * projection);
        for row in 0..ROWS {
            let (q, k, v, state) = cpu_ling3_kda_prep(
                &qkv[row * conv_width..(row + 1) * conv_width],
                &conv_weight,
                &expected_state,
                HEADS,
            );
            expected_q.extend(q);
            expected_k.extend(k);
            expected_v.extend(v);
            expected_state = state;
        }

        let qkv = DeviceBuffer::from_host(&qkv).expect("QKV");
        let conv_weight = DeviceBuffer::from_host(&conv_weight).expect("weights");
        let mut state = DeviceBuffer::from_host(&initial_state).expect("state");
        let mut q = DeviceBuffer::zeroed(ROWS * projection).expect("Q");
        let mut k = DeviceBuffer::zeroed(ROWS * projection).expect("K");
        let mut v = DeviceBuffer::zeroed(ROWS * projection).expect("V");
        let stream = CudaStream::new_non_blocking().expect("stream");
        ling3_kda_prep_rows_into_on_stream(
            &qkv,
            &conv_weight,
            q.output(),
            k.output(),
            v.output(),
            state.inout(),
            ROWS,
            HEADS,
            &stream,
        )
        .expect("contiguous prep");
        assert_close(
            &q.copy_to_host(&stream).expect("Q read"),
            &expected_q,
            2.0e-6,
            "contiguous Ling normalized Q",
        );
        assert_close(
            &k.copy_to_host(&stream).expect("K read"),
            &expected_k,
            2.0e-6,
            "contiguous Ling normalized K",
        );
        assert_close(
            &v.copy_to_host(&stream).expect("V read"),
            &expected_v,
            2.0e-6,
            "contiguous Ling convolved V",
        );
        assert_close(
            &state.copy_to_host(&stream).expect("state read"),
            &expected_state,
            0.0,
            "contiguous Ling convolution state",
        );
    }

    #[test]
    fn ling3_mla_pack_and_attention_match_cpu_reference() {
        let heads = 2usize;
        let qk_nope = 3usize;
        let rope = 2usize;
        let qk_dim = qk_nope + rope;
        let value_dim = 4usize;
        let query_projection = (0..heads * qk_dim)
            .map(|index| index as f32 * 0.1 - 0.4)
            .collect::<Vec<_>>();
        let kv_projection = (0..heads * (qk_nope + value_dim))
            .map(|index| index as f32 * 0.05 - 0.25)
            .collect::<Vec<_>>();
        let shared_rope = vec![0.75f32, -0.5];
        let expected_query = query_projection.clone();
        let mut expected_key = vec![0.0f32; heads * qk_dim];
        let mut expected_value = vec![0.0f32; heads * value_dim];
        for head in 0..heads {
            expected_key[head * qk_dim..head * qk_dim + qk_nope].copy_from_slice(
                &kv_projection
                    [head * (qk_nope + value_dim)..head * (qk_nope + value_dim) + qk_nope],
            );
            expected_key[head * qk_dim + qk_nope..(head + 1) * qk_dim]
                .copy_from_slice(&shared_rope);
            expected_value[head * value_dim..(head + 1) * value_dim].copy_from_slice(
                &kv_projection
                    [head * (qk_nope + value_dim) + qk_nope..(head + 1) * (qk_nope + value_dim)],
            );
        }

        let query_projection =
            DeviceBuffer::from_host(&query_projection).expect("query projection");
        let kv_projection = DeviceBuffer::from_host(&kv_projection).expect("KV projection");
        let shared_rope = DeviceBuffer::from_host(&shared_rope).expect("shared rope");
        let mut query = DeviceBuffer::zeroed(heads * qk_dim).expect("query");
        let mut key = DeviceBuffer::zeroed(heads * qk_dim).expect("key");
        let mut value = DeviceBuffer::zeroed(heads * value_dim).expect("value");
        let stream = CudaStream::new_non_blocking().expect("stream");
        ling3_mla_pack_f32_into_on_stream(
            &query_projection,
            &kv_projection,
            &shared_rope,
            query.output(),
            key.output(),
            value.output(),
            heads,
            qk_nope,
            rope,
            value_dim,
            &stream,
        )
        .expect("MLA pack");
        assert_eq!(
            query.copy_to_host(&stream).expect("query download"),
            expected_query
        );
        assert_eq!(
            key.copy_to_host(&stream).expect("key download"),
            expected_key
        );
        assert_eq!(
            value.copy_to_host(&stream).expect("value download"),
            expected_value
        );

        let rows = 2usize;
        let query_projection_rows = [expected_query.clone(), expected_query.clone()].concat();
        let kv_projection_rows = [
            kv_projection.copy_to_host(&stream).unwrap().into_vec(),
            kv_projection.copy_to_host(&stream).unwrap().into_vec(),
        ]
        .concat();
        let shared_rope_rows = [
            shared_rope.copy_to_host(&stream).unwrap().into_vec(),
            vec![0.25, 0.5],
        ]
        .concat();
        let mut expected_key_row_1 = expected_key.clone();
        for head in 0..heads {
            expected_key_row_1[head * qk_dim + qk_nope..(head + 1) * qk_dim]
                .copy_from_slice(&[0.25, 0.5]);
        }
        let expected_query_rows = query_projection_rows.clone();
        let expected_key_rows = [expected_key.clone(), expected_key_row_1].concat();
        let expected_value_rows = [expected_value.clone(), expected_value.clone()].concat();
        let query_projection_rows =
            DeviceBuffer::from_host(&query_projection_rows).expect("query projection rows");
        let kv_projection_rows =
            DeviceBuffer::from_host(&kv_projection_rows).expect("KV projection rows");
        let shared_rope_rows = DeviceBuffer::from_host(&shared_rope_rows).expect("rope rows");
        let mut query_rows = DeviceBuffer::zeroed(rows * heads * qk_dim).expect("query rows");
        let mut key_rows = DeviceBuffer::zeroed(rows * heads * qk_dim).expect("key rows");
        let mut value_rows = DeviceBuffer::zeroed(rows * heads * value_dim).expect("value rows");
        ling3_mla_pack_f32_batch_into_on_stream(
            &query_projection_rows,
            &kv_projection_rows,
            &shared_rope_rows,
            query_rows.output(),
            key_rows.output(),
            value_rows.output(),
            rows,
            heads,
            qk_nope,
            rope,
            value_dim,
            &stream,
        )
        .expect("batched MLA pack");
        assert_eq!(
            query_rows.copy_to_host(&stream).unwrap(),
            expected_query_rows
        );
        assert_eq!(key_rows.copy_to_host(&stream).unwrap(), expected_key_rows);
        assert_eq!(
            value_rows.copy_to_host(&stream).unwrap(),
            expected_value_rows
        );

        let cache_len = 3usize;
        let key_cache = (0..cache_len * heads * qk_dim)
            .map(|index| ((index * 7 % 31) as f32 - 15.0) * 0.04)
            .collect::<Vec<_>>();
        let value_cache = (0..cache_len * heads * value_dim)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) * 0.03)
            .collect::<Vec<_>>();
        let scale = (qk_dim as f32).sqrt().recip();
        let expected = cpu_ling3_mla_attention(
            &expected_query,
            &key_cache,
            &value_cache,
            cache_len,
            heads,
            qk_dim,
            value_dim,
            scale,
        );
        let key_cache = DeviceBuffer::from_host(&key_cache).expect("key cache");
        let value_cache = DeviceBuffer::from_host(&value_cache).expect("value cache");
        let mut output = DeviceBuffer::zeroed(heads * value_dim).expect("attention output");
        ling3_mla_attention_f32_into_on_stream(
            &query,
            &key_cache,
            &value_cache,
            output.output(),
            cache_len,
            heads,
            qk_dim,
            value_dim,
            scale,
            &stream,
        )
        .expect("MLA attention");
        assert_close(
            &output.copy_to_host(&stream).expect("attention download"),
            &expected,
            2.0e-6,
            "Ling MLA attention",
        );

        let page_tokens = 2usize;
        let page_table = [2u32, 0];
        let page_slots = 3usize;
        let mut paged_keys = vec![0.0; page_slots * page_tokens * heads * qk_dim];
        let mut paged_values = vec![0.0; page_slots * page_tokens * heads * value_dim];
        for token in 0..cache_len {
            let slot = page_table[token / page_tokens] as usize;
            let row = slot * page_tokens + token % page_tokens;
            let key_width = heads * qk_dim;
            let value_width = heads * value_dim;
            paged_keys[row * key_width..(row + 1) * key_width].copy_from_slice(
                &key_cache.copy_to_host(&stream).unwrap()
                    [token * key_width..(token + 1) * key_width],
            );
            paged_values[row * value_width..(row + 1) * value_width].copy_from_slice(
                &value_cache.copy_to_host(&stream).unwrap()
                    [token * value_width..(token + 1) * value_width],
            );
        }
        let paged_keys = DeviceBuffer::from_host(&paged_keys).expect("paged keys");
        let paged_values = DeviceBuffer::from_host(&paged_values).expect("paged values");
        let page_table = DeviceBuffer::from_host(&page_table).expect("page table");
        let mut paged_output = DeviceBuffer::zeroed(heads * value_dim).expect("paged output");
        ling3_mla_paged_attention_f32_into_on_stream(
            &query,
            &paged_keys,
            &paged_values,
            &page_table,
            paged_output.output(),
            cache_len,
            page_tokens,
            heads,
            qk_dim,
            value_dim,
            scale,
            &stream,
        )
        .expect("paged MLA attention");
        assert_close(
            &paged_output
                .copy_to_host(&stream)
                .expect("paged output download"),
            &expected,
            2.0e-6,
            "Ling paged MLA attention",
        );

        let query_rows = [expected_query.clone(), expected_query.clone()].concat();
        let expected_rows = [
            cpu_ling3_mla_attention(
                &expected_query,
                &key_cache.copy_to_host(&stream).unwrap(),
                &value_cache.copy_to_host(&stream).unwrap(),
                2,
                heads,
                qk_dim,
                value_dim,
                scale,
            ),
            expected.clone(),
        ]
        .concat();
        let query_rows = DeviceBuffer::from_host(&query_rows).expect("query rows");
        let mut causal_rows = DeviceBuffer::zeroed(2 * heads * value_dim).expect("causal rows");
        ling3_mla_paged_causal_rows_f32_into_on_stream(
            &query_rows,
            &paged_keys,
            &paged_values,
            &page_table,
            causal_rows.output(),
            1,
            2,
            page_tokens,
            heads,
            qk_dim,
            value_dim,
            scale,
            &stream,
        )
        .expect("causal paged MLA rows");
        assert_close(
            &causal_rows
                .copy_to_host(&stream)
                .expect("causal rows download"),
            &expected_rows,
            2.0e-6,
            "Ling causal paged MLA rows",
        );
    }

    #[test]
    fn qwen36_paired_batch_gates_match_separate_inputs() {
        let rows = 3usize;
        let heads = 4usize;
        let alpha = (0..rows * heads)
            .map(|idx| (idx as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let beta_input = (0..rows * heads)
            .map(|idx| (7.0 - idx as f32) * 0.1)
            .collect::<Vec<_>>();
        let mut alpha_beta = Vec::with_capacity(rows * heads * 2);
        for row in 0..rows {
            alpha_beta.extend_from_slice(&alpha[row * heads..(row + 1) * heads]);
            alpha_beta.extend_from_slice(&beta_input[row * heads..(row + 1) * heads]);
        }
        let a_log = (0..heads)
            .map(|idx| format::f32_to_bf16(-2.0 - idx as f32 * 0.25))
            .collect::<Vec<_>>();
        let dt_bias = (0..heads)
            .map(|idx| format::f32_to_bf16(-0.5 + idx as f32 * 0.125))
            .collect::<Vec<_>>();
        let alpha = DeviceBuffer::from_host(&alpha).expect("alpha");
        let beta_input = DeviceBuffer::from_host(&beta_input).expect("beta input");
        let alpha_beta = DeviceBuffer::from_host(&alpha_beta).expect("alpha beta");
        let a_log = DeviceBuffer::from_host(&a_log).expect("a log");
        let dt_bias = DeviceBuffer::from_host(&dt_bias).expect("dt bias");
        let mut expected_gate = DeviceBuffer::zeroed(rows * heads).expect("expected gate");
        let mut expected_beta = DeviceBuffer::zeroed(rows * heads).expect("expected beta");
        let mut actual_gate = DeviceBuffer::zeroed(rows * heads).expect("actual gate");
        let mut actual_beta = DeviceBuffer::zeroed(rows * heads).expect("actual beta");
        let mut expected_gate_bf16 =
            DeviceBuffer::zeroed(rows * heads).expect("expected BF16 gate");
        let mut expected_beta_bf16 =
            DeviceBuffer::zeroed(rows * heads).expect("expected BF16 beta");
        let mut actual_gate_bf16 = DeviceBuffer::zeroed(rows * heads).expect("actual BF16 gate");
        let mut actual_beta_bf16 = DeviceBuffer::zeroed(rows * heads).expect("actual BF16 beta");
        let stream = CudaStream::new_blocking().expect("stream");

        qwen36_gdn_gate_batch_into_on_stream(
            &alpha,
            &beta_input,
            &a_log,
            &dt_bias,
            expected_gate.output(),
            expected_beta.output(),
            rows,
            heads,
            &stream,
        )
        .expect("separate gate");
        qwen36_gdn_gate_paired_batch_into_on_stream(
            &alpha_beta,
            &a_log,
            &dt_bias,
            actual_gate.output(),
            actual_beta.output(),
            rows,
            heads,
            &stream,
        )
        .expect("paired gate");
        qwen36_gdn_gate_batch_bf16_into_on_stream(
            &alpha,
            &beta_input,
            &a_log,
            &dt_bias,
            expected_gate_bf16.output(),
            expected_beta_bf16.output(),
            rows,
            heads,
            &stream,
        )
        .expect("separate BF16 gate");
        qwen36_gdn_gate_paired_batch_bf16_into_on_stream(
            &alpha_beta,
            &a_log,
            &dt_bias,
            actual_gate_bf16.output(),
            actual_beta_bf16.output(),
            rows,
            heads,
            &stream,
        )
        .expect("paired BF16 gate");

        assert_eq!(
            actual_gate
                .copy_to_host(&stream)
                .expect("paired gate download"),
            expected_gate
                .copy_to_host(&stream)
                .expect("separate gate download")
        );
        assert_eq!(
            actual_beta
                .copy_to_host(&stream)
                .expect("paired beta download"),
            expected_beta
                .copy_to_host(&stream)
                .expect("separate beta download")
        );
        assert_eq!(
            actual_gate_bf16
                .copy_to_host(&stream)
                .expect("paired BF16 gate download"),
            expected_gate_bf16
                .copy_to_host(&stream)
                .expect("separate BF16 gate download")
        );
        assert_eq!(
            actual_beta_bf16
                .copy_to_host(&stream)
                .expect("paired BF16 beta download"),
            expected_beta_bf16
                .copy_to_host(&stream)
                .expect("separate BF16 beta download")
        );
    }

    #[test]
    fn gated_delta_net_128_matches_cpu_after_long_recurrence() {
        let heads = 2usize;
        let len = heads * 128;
        let mut expected_state = vec![0.0f32; heads * 128 * 128];
        let mut state_device =
            DeviceBuffer::<f32>::zeroed(expected_state.len()).expect("state alloc");
        let mut q_device = DeviceBuffer::<f32>::zeroed(len).expect("q alloc");
        let mut k_device = DeviceBuffer::<f32>::zeroed(len).expect("k alloc");
        let mut v_device = DeviceBuffer::<f32>::zeroed(len).expect("v alloc");
        let mut gate_device = DeviceBuffer::<f32>::zeroed(heads).expect("gate alloc");
        let mut beta_device = DeviceBuffer::<f32>::zeroed(heads).expect("beta alloc");
        let mut output_device = DeviceBuffer::<f32>::zeroed(len).expect("output alloc");
        let stream = CudaStream::new_blocking().expect("stream create");
        let mut expected_output = Vec::new();

        for step in 0..512usize {
            let mut q = (0..len)
                .map(|idx| (((idx * 17 + step * 7) % 101) as f32 - 50.0) * 0.015625)
                .collect::<Vec<_>>();
            let mut k = (0..len)
                .map(|idx| (((idx * 13 + step * 11) % 103) as f32 - 51.0) * 0.015625)
                .collect::<Vec<_>>();
            normalize_heads_128(&mut q, heads);
            normalize_heads_128(&mut k, heads);
            let v = (0..len)
                .map(|idx| (((idx * 19 + step * 5) % 107) as f32 - 53.0) * 0.01)
                .collect::<Vec<_>>();
            let gate = vec![-0.015 - step as f32 * 1.0e-6, -0.04];
            let beta = vec![0.35, 0.7];
            expected_output =
                cpu_gated_delta_net_128(&q, &k, &v, &gate, &beta, &mut expected_state, heads);

            q_device.copy_from_host(&q).expect("q upload");
            k_device.copy_from_host(&k).expect("k upload");
            v_device.copy_from_host(&v).expect("v upload");
            gate_device.copy_from_host(&gate).expect("gate upload");
            beta_device.copy_from_host(&beta).expect("beta upload");
            gated_delta_net_128_f32_into_on_stream(
                &q_device,
                &k_device,
                &v_device,
                &gate_device,
                &beta_device,
                state_device.inout(),
                output_device.output(),
                heads,
                &stream,
            )
            .expect("gated delta net enqueue");
        }

        let output = output_device
            .copy_to_host(&stream)
            .expect("output download");
        let actual_state = state_device.copy_to_host(&stream).expect("state download");
        assert_close(&output, &expected_output, 3.0e-5, "long GDN output");
        assert_close(&actual_state, &expected_state, 3.0e-5, "long GDN state");
    }

    #[test]
    fn qwen36_batched_gdn_matches_independent_sequence_updates() {
        let batch_size = 2usize;
        let key_heads = 1usize;
        let value_heads = 2usize;
        let head_dim = 128usize;
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let state_len = value_heads * head_dim * head_dim;
        let qkv = (0..batch_size * conv_dim)
            .map(|idx| ((idx * 17 % 97) as f32 - 48.0) * 0.0025)
            .collect::<Vec<_>>();
        let conv_weight = (0..conv_dim * 4)
            .map(|idx| format::f32_to_bf16(((idx * 7 % 31) as f32 - 15.0) * 0.01))
            .collect::<Vec<_>>();
        let alpha = (0..batch_size * value_heads)
            .map(|idx| (idx as f32 - 1.5) * 0.125)
            .collect::<Vec<_>>();
        let beta_input = (0..batch_size * value_heads)
            .map(|idx| (1.5 - idx as f32) * 0.25)
            .collect::<Vec<_>>();
        let a_log = (0..value_heads)
            .map(|idx| format::f32_to_bf16(-2.0 - idx as f32 * 0.25))
            .collect::<Vec<_>>();
        let dt_bias = (0..value_heads)
            .map(|idx| format::f32_to_bf16(-0.5 + idx as f32 * 0.125))
            .collect::<Vec<_>>();
        let conv_initial = (0..batch_size)
            .map(|batch| {
                (0..conv_dim * 3)
                    .map(|idx| ((idx * 11 + batch * 13) % 89) as f32 * 0.0005)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let recurrent_initial = (0..batch_size)
            .map(|batch| {
                (0..state_len)
                    .map(|idx| ((idx * 5 + batch * 19) % 101) as f32 * 0.00001)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");

        let qkv_device = DeviceBuffer::from_host(&qkv).expect("qkv");
        let conv_weight_device = DeviceBuffer::from_host(&conv_weight).expect("conv weight");
        let alpha_device = DeviceBuffer::from_host(&alpha).expect("alpha");
        let beta_input_device = DeviceBuffer::from_host(&beta_input).expect("beta input");
        let a_log_device = DeviceBuffer::from_host(&a_log).expect("a log");
        let dt_bias_device = DeviceBuffer::from_host(&dt_bias).expect("dt bias");
        let mut batch_conv_states = conv_initial
            .iter()
            .map(|state| DeviceBuffer::from_host(state).expect("batch conv state"))
            .collect::<Vec<_>>();
        let mut batch_recurrent_states = recurrent_initial
            .iter()
            .map(|state| DeviceBuffer::from_host(state).expect("batch recurrent state"))
            .collect::<Vec<_>>();
        let state_table_offset = 3;
        let mut conv_ptrs = vec![std::ptr::null_mut(); state_table_offset];
        conv_ptrs.extend(
            batch_conv_states
                .iter_mut()
                .map(|state| state.as_mut_ptr().cast::<f32>()),
        );
        let mut recurrent_ptrs = vec![std::ptr::null_mut(); state_table_offset];
        recurrent_ptrs.extend(
            batch_recurrent_states
                .iter_mut()
                .map(|state| state.as_mut_ptr().cast::<f32>()),
        );
        let conv_table = DeviceBuffer::from_host(&conv_ptrs).expect("conv table");
        let recurrent_table = DeviceBuffer::from_host(&recurrent_ptrs).expect("recurrent table");
        let mut batch_q = DeviceBuffer::zeroed(batch_size * value_dim).expect("batch q");
        let mut batch_k = DeviceBuffer::zeroed(batch_size * value_dim).expect("batch k");
        let mut batch_v = DeviceBuffer::zeroed(batch_size * value_dim).expect("batch v");
        let mut batch_gate = DeviceBuffer::zeroed(batch_size * value_heads).expect("batch gate");
        let mut batch_beta = DeviceBuffer::zeroed(batch_size * value_heads).expect("batch beta");
        let mut batch_output = DeviceBuffer::zeroed(batch_size * value_dim).expect("batch output");

        qwen36_gdn_prep_batch_into_on_stream(
            &qkv_device,
            &conv_weight_device,
            batch_q.output(),
            batch_k.output(),
            batch_v.output(),
            &conv_table,
            state_table_offset,
            batch_size,
            key_heads,
            value_heads,
            head_dim,
            &stream,
        )
        .expect("batch prep");
        qwen36_gdn_gate_batch_into_on_stream(
            &alpha_device,
            &beta_input_device,
            &a_log_device,
            &dt_bias_device,
            batch_gate.output(),
            batch_beta.output(),
            batch_size,
            value_heads,
            &stream,
        )
        .expect("batch gate");
        gated_delta_net_128_f32_batch_into_on_stream(
            &batch_q,
            &batch_k,
            &batch_v,
            &batch_gate,
            &batch_beta,
            &recurrent_table,
            batch_output.output(),
            state_table_offset,
            batch_size,
            value_heads,
            &stream,
        )
        .expect("batch recurrent update");

        let batch_q_host = batch_q.copy_to_host(&stream).expect("batch q download");
        let batch_k_host = batch_k.copy_to_host(&stream).expect("batch k download");
        let batch_v_host = batch_v.copy_to_host(&stream).expect("batch v download");
        let batch_gate_host = batch_gate
            .copy_to_host(&stream)
            .expect("batch gate download");
        let batch_beta_host = batch_beta
            .copy_to_host(&stream)
            .expect("batch beta download");
        let batch_output_host = batch_output
            .copy_to_host(&stream)
            .expect("batch output download");

        for batch in 0..batch_size {
            let qkv_row = DeviceBuffer::from_host(&qkv[batch * conv_dim..(batch + 1) * conv_dim])
                .expect("row qkv");
            let mut conv_state =
                DeviceBuffer::from_host(&conv_initial[batch]).expect("row conv state");
            let mut recurrent_state =
                DeviceBuffer::from_host(&recurrent_initial[batch]).expect("row recurrent state");
            let mut q = DeviceBuffer::zeroed(value_dim).expect("row q");
            let mut k = DeviceBuffer::zeroed(value_dim).expect("row k");
            let mut v = DeviceBuffer::zeroed(value_dim).expect("row v");
            qwen36_gdn_prep_into_on_stream(
                &qkv_row,
                &conv_weight_device,
                q.output(),
                k.output(),
                v.output(),
                conv_state.inout(),
                key_heads,
                value_heads,
                head_dim,
                &stream,
            )
            .expect("row prep");
            let alpha_row =
                DeviceBuffer::from_host(&alpha[batch * value_heads..(batch + 1) * value_heads])
                    .expect("row alpha");
            let beta_input_row = DeviceBuffer::from_host(
                &beta_input[batch * value_heads..(batch + 1) * value_heads],
            )
            .expect("row beta input");
            let mut gate = DeviceBuffer::zeroed(value_heads).expect("row gate");
            let mut beta = DeviceBuffer::zeroed(value_heads).expect("row beta");
            qwen36_gdn_gate_into_on_stream(
                &alpha_row,
                &beta_input_row,
                &a_log_device,
                &dt_bias_device,
                gate.output(),
                beta.output(),
                value_heads,
                &stream,
            )
            .expect("row gate");
            let mut output = DeviceBuffer::zeroed(value_dim).expect("row output");
            gated_delta_net_128_f32_into_on_stream(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                recurrent_state.inout(),
                output.output(),
                value_heads,
                &stream,
            )
            .expect("row recurrent update");

            let range = batch * value_dim..(batch + 1) * value_dim;
            assert_close(
                &batch_q_host[range.clone()],
                &q.copy_to_host(&stream).expect("row q download"),
                1.0e-6,
                "batched GDN q",
            );
            assert_close(
                &batch_k_host[range.clone()],
                &k.copy_to_host(&stream).expect("row k download"),
                1.0e-6,
                "batched GDN k",
            );
            assert_close(
                &batch_v_host[range.clone()],
                &v.copy_to_host(&stream).expect("row v download"),
                1.0e-6,
                "batched GDN v",
            );
            let scalar_range = batch * value_heads..(batch + 1) * value_heads;
            assert_close(
                &batch_gate_host[scalar_range.clone()],
                &gate.copy_to_host(&stream).expect("row gate download"),
                1.0e-6,
                "batched GDN gate",
            );
            assert_close(
                &batch_beta_host[scalar_range],
                &beta.copy_to_host(&stream).expect("row beta download"),
                1.0e-6,
                "batched GDN beta",
            );
            assert_close(
                &batch_output_host[range],
                &output.copy_to_host(&stream).expect("row output download"),
                1.0e-5,
                "batched GDN output",
            );
            assert_close(
                &batch_conv_states[batch]
                    .copy_to_host(&stream)
                    .expect("batch conv state download"),
                &conv_state
                    .copy_to_host(&stream)
                    .expect("row conv state download"),
                1.0e-6,
                "batched GDN conv state",
            );
            assert_close(
                &batch_recurrent_states[batch]
                    .copy_to_host(&stream)
                    .expect("batch recurrent state download"),
                &recurrent_state
                    .copy_to_host(&stream)
                    .expect("row recurrent state download"),
                1.0e-5,
                "batched GDN recurrent state",
            );
        }
    }

    #[test]
    fn qwen36_chunked_gdn_matches_repeated_sequence_updates() {
        let tokens = 6usize;
        let key_heads = 1usize;
        let value_heads = 2usize;
        let head_dim = 128usize;
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let state_len = value_heads * head_dim * head_dim;
        let qkv = (0..tokens * conv_dim)
            .map(|idx| ((idx * 17 % 101) as f32 - 50.0) * 0.0078125)
            .collect::<Vec<_>>();
        let conv_weight = (0..conv_dim * 4)
            .map(|idx| format::f32_to_bf16(((idx % 13) as f32 - 6.0) * 0.03125))
            .collect::<Vec<_>>();
        let gate = (0..tokens * value_heads)
            .map(|idx| -0.02 - (idx % value_heads) as f32 * 0.01)
            .collect::<Vec<_>>();
        let beta = (0..tokens * value_heads)
            .map(|idx| 0.2 + (idx % value_heads) as f32 * 0.1)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let qkv_device = DeviceBuffer::from_host(&qkv).expect("qkv");
        let weight_device = DeviceBuffer::from_host(&conv_weight).expect("weight");
        let gate_device = DeviceBuffer::from_host(&gate).expect("gate");
        let beta_device = DeviceBuffer::from_host(&beta).expect("beta");

        let mut chunk_conv = DeviceBuffer::<f32>::zeroed(conv_dim * 3).expect("chunk conv");
        let mut chunk_recurrent = DeviceBuffer::<f32>::zeroed(state_len).expect("chunk recurrent");
        let chunk_conv_table = DeviceBuffer::from_host(&[chunk_conv.as_mut_ptr().cast::<f32>()])
            .expect("chunk conv table");
        let chunk_recurrent_table =
            DeviceBuffer::from_host(&[chunk_recurrent.as_mut_ptr().cast::<f32>()])
                .expect("chunk recurrent table");
        let offsets = DeviceBuffer::from_host(&[0u32]).expect("offsets");
        let lengths = DeviceBuffer::from_host(&[tokens as u32]).expect("lengths");
        let mut chunk_q = DeviceBuffer::<f32>::zeroed(tokens * value_dim).expect("chunk q");
        let mut chunk_k = DeviceBuffer::<f32>::zeroed(tokens * value_dim).expect("chunk k");
        let mut chunk_v = DeviceBuffer::<f32>::zeroed(tokens * value_dim).expect("chunk v");
        let mut chunk_output =
            DeviceBuffer::<f32>::zeroed(tokens * value_dim).expect("chunk output");
        qwen36_gdn_prep_chunks_into_on_stream(
            &qkv_device,
            &weight_device,
            chunk_q.output(),
            chunk_k.output(),
            chunk_v.output(),
            &chunk_conv_table,
            0,
            &offsets,
            &lengths,
            1,
            tokens,
            key_heads,
            value_heads,
            head_dim,
            &stream,
        )
        .expect("chunk prep");
        gated_delta_net_128_f32_chunks_into_on_stream(
            &chunk_q,
            &chunk_k,
            &chunk_v,
            &gate_device,
            &beta_device,
            &chunk_recurrent_table,
            0,
            &offsets,
            &lengths,
            chunk_output.output(),
            1,
            tokens,
            value_heads,
            &stream,
        )
        .expect("chunk recurrent");

        let mut repeated_conv = DeviceBuffer::<f32>::zeroed(conv_dim * 3).expect("repeated conv");
        let mut repeated_recurrent =
            DeviceBuffer::<f32>::zeroed(state_len).expect("repeated recurrent");
        let mut repeated_q = Vec::with_capacity(tokens * value_dim);
        let mut repeated_k = Vec::with_capacity(tokens * value_dim);
        let mut repeated_v = Vec::with_capacity(tokens * value_dim);
        let mut repeated_output = Vec::with_capacity(tokens * value_dim);
        for token in 0..tokens {
            let qkv_row = DeviceBuffer::from_host(&qkv[token * conv_dim..(token + 1) * conv_dim])
                .expect("qkv row");
            let mut q = DeviceBuffer::<f32>::zeroed(value_dim).expect("q row");
            let mut k = DeviceBuffer::<f32>::zeroed(value_dim).expect("k row");
            let mut v = DeviceBuffer::<f32>::zeroed(value_dim).expect("v row");
            qwen36_gdn_prep_into_on_stream(
                &qkv_row,
                &weight_device,
                q.output(),
                k.output(),
                v.output(),
                repeated_conv.inout(),
                key_heads,
                value_heads,
                head_dim,
                &stream,
            )
            .expect("repeated prep");
            let gate_row =
                DeviceBuffer::from_host(&gate[token * value_heads..(token + 1) * value_heads])
                    .expect("gate row");
            let beta_row =
                DeviceBuffer::from_host(&beta[token * value_heads..(token + 1) * value_heads])
                    .expect("beta row");
            let mut output = DeviceBuffer::<f32>::zeroed(value_dim).expect("output row");
            gated_delta_net_128_f32_into_on_stream(
                &q,
                &k,
                &v,
                &gate_row,
                &beta_row,
                repeated_recurrent.inout(),
                output.output(),
                value_heads,
                &stream,
            )
            .expect("repeated recurrent");
            repeated_q.extend(q.copy_to_host(&stream).expect("q download"));
            repeated_k.extend(k.copy_to_host(&stream).expect("k download"));
            repeated_v.extend(v.copy_to_host(&stream).expect("v download"));
            repeated_output.extend(output.copy_to_host(&stream).expect("output download"));
        }

        assert_close(
            &chunk_q.copy_to_host(&stream).expect("chunk q download"),
            &repeated_q,
            1.0e-6,
            "chunked prep q",
        );
        assert_close(
            &chunk_k.copy_to_host(&stream).expect("chunk k download"),
            &repeated_k,
            1.0e-6,
            "chunked prep k",
        );
        assert_close(
            &chunk_v.copy_to_host(&stream).expect("chunk v download"),
            &repeated_v,
            1.0e-6,
            "chunked prep v",
        );
        assert_close(
            &chunk_output
                .copy_to_host(&stream)
                .expect("chunk output download"),
            &repeated_output,
            1.0e-5,
            "chunked GDN output",
        );
        assert_close(
            &chunk_conv
                .copy_to_host(&stream)
                .expect("chunk conv download"),
            &repeated_conv
                .copy_to_host(&stream)
                .expect("repeated conv download"),
            1.0e-6,
            "chunked conv state",
        );
        assert_close(
            &chunk_recurrent
                .copy_to_host(&stream)
                .expect("chunk recurrent download"),
            &repeated_recurrent
                .copy_to_host(&stream)
                .expect("repeated recurrent download"),
            1.0e-5,
            "chunked recurrent state",
        );
    }

    #[test]
    fn qwen36_short_chunk_recurrence_is_bit_exact_to_single_token_batch() {
        let tokens = 3usize;
        let heads = 2usize;
        let vector_dim = heads * 128;
        let state_len = heads * 128 * 128;
        let vectors = tokens * vector_dim;
        let scalars = tokens * heads;
        let values = |multiplier: usize, modulus: usize, offset: f32, scale: f32| {
            (0..vectors)
                .map(|index| ((index * multiplier % modulus) as f32 - offset) * scale)
                .collect::<Vec<_>>()
        };
        let q_host = values(17, 103, 49.3, 0.0073);
        let k_host = values(29, 107, 51.7, 0.0061);
        let v_host = values(43, 109, 53.2, 0.0059);
        let gate_host = (0..scalars)
            .map(|index| -0.0137 - index as f32 * 0.0043)
            .collect::<Vec<_>>();
        let beta_host = (0..scalars)
            .map(|index| 0.173 + index as f32 * 0.031)
            .collect::<Vec<_>>();
        let state_host = (0..state_len)
            .map(|index| ((index * 31 % 113) as f32 - 55.4) * 0.00017)
            .collect::<Vec<_>>();
        let q = DeviceBuffer::from_host(&q_host).expect("q upload");
        let k = DeviceBuffer::from_host(&k_host).expect("k upload");
        let v = DeviceBuffer::from_host(&v_host).expect("v upload");
        let gate = DeviceBuffer::from_host(&gate_host).expect("gate upload");
        let beta = DeviceBuffer::from_host(&beta_host).expect("beta upload");
        let offsets = DeviceBuffer::from_host(&[0u32]).expect("offset upload");
        let lengths = DeviceBuffer::from_host(&[tokens as u32]).expect("length upload");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut chunk_state = DeviceBuffer::from_host(&state_host).expect("chunk state upload");
        let chunk_table = DeviceBuffer::from_host(&[chunk_state.as_mut_ptr().cast::<f32>()])
            .expect("chunk table");
        let mut chunk_output = DeviceBuffer::zeroed(vectors).expect("chunk output");
        gated_delta_net_128_f32_chunks_into_on_stream(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &chunk_table,
            0,
            &offsets,
            &lengths,
            chunk_output.output(),
            1,
            tokens,
            heads,
            &stream,
        )
        .expect("chunk recurrence");

        let mut repeated_state =
            DeviceBuffer::from_host(&state_host).expect("repeated state upload");
        let repeated_table = DeviceBuffer::from_host(&[repeated_state.as_mut_ptr().cast::<f32>()])
            .expect("repeated table");
        let mut repeated_output = Vec::with_capacity(vectors);
        for token in 0..tokens {
            let range = token * vector_dim..(token + 1) * vector_dim;
            let scalar_range = token * heads..(token + 1) * heads;
            let q_row = DeviceBuffer::from_host(&q_host[range.clone()]).expect("q row upload");
            let k_row = DeviceBuffer::from_host(&k_host[range.clone()]).expect("k row upload");
            let v_row = DeviceBuffer::from_host(&v_host[range]).expect("v row upload");
            let gate_row =
                DeviceBuffer::from_host(&gate_host[scalar_range.clone()]).expect("gate row upload");
            let beta_row =
                DeviceBuffer::from_host(&beta_host[scalar_range]).expect("beta row upload");
            let mut output = DeviceBuffer::zeroed(vector_dim).expect("row output");
            gated_delta_net_128_f32_batch_into_on_stream(
                &q_row,
                &k_row,
                &v_row,
                &gate_row,
                &beta_row,
                &repeated_table,
                output.output(),
                0,
                1,
                heads,
                &stream,
            )
            .expect("single-token recurrence");
            repeated_output.extend(output.copy_to_host(&stream).expect("row output download"));
        }

        assert_eq!(
            chunk_output
                .copy_to_host(&stream)
                .expect("chunk output download")
                .as_slice(),
            repeated_output,
        );
        assert_eq!(
            chunk_state
                .copy_to_host(&stream)
                .expect("chunk state download")
                .as_slice(),
            repeated_state
                .copy_to_host(&stream)
                .expect("repeated state download")
                .as_slice(),
        );
    }

    #[test]
    fn channel_scaled_bf16_to_fp8_quantization_uses_row_scales() {
        let rows = 2usize;
        let cols = 4usize;
        let values = [-4.0f32, -2.0, 0.0, 4.0, -32.0, -8.0, 8.0, 32.0];
        let scales = [0.25f32, 2.0];
        let bf16 = values
            .iter()
            .map(|&value| crate::format::f32_to_bf16(value))
            .collect::<Vec<_>>();
        let expected = bf16
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                crate::format::cuda_e4m3_code(
                    crate::format::bf16_to_f32(value) / scales[index / cols],
                )
            })
            .collect::<Vec<_>>();
        let bf16 = DeviceBuffer::from_host(&bf16).expect("BF16 upload");
        let scales = DeviceBuffer::from_host(&scales).expect("scale upload");
        let mut quantized = DeviceBuffer::zeroed(values.len()).expect("FP8 output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream(
            &bf16,
            &scales,
            quantized.output(),
            rows,
            cols,
            &stream,
        )
        .expect("channel-scaled quantization");
        assert_eq!(
            quantized.copy_to_host(&stream).expect("FP8 readback"),
            expected
        );
    }

    #[test]
    fn fp8_linear_schedules_match_cpu_reference() {
        let rows = 5usize;
        let cols = 7usize;
        let input = (0..cols)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.25)
            .collect::<Vec<_>>();
        let weight_f32 = (0..rows * cols)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.125)
            .collect::<Vec<_>>();
        let weight = weight_f32
            .iter()
            .map(|value| format::cuda_e4m3_code(*value))
            .collect::<Vec<_>>();
        let weight_scale = 0.75f32;
        let expected = cpu_fp8_linear_f32(&input, &weight, rows, cols, weight_scale);

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(rows).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");
        for threads in [64, 96, 128, 160, 192, 256, 512] {
            fp8_linear_configured_f32_into_on_stream(
                &input_device,
                &weight_device,
                output_device.output(),
                rows,
                cols,
                weight_scale,
                threads,
                &stream,
            )
            .expect("fp8 linear enqueue");

            let output = output_device
                .copy_to_host(&stream)
                .expect("output download");
            assert_close(
                &output,
                &expected,
                2.0e-6,
                &format!("fp8 linear threads={threads}"),
            );
        }
    }

    #[test]
    fn fp8_linear_batch_matches_independent_rows() {
        let batch = 3usize;
        let rows = 5usize;
        let cols = 7usize;
        let input = (0..batch * cols)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let weight = (0..rows * cols)
            .map(|idx| format::cuda_e4m3_code(((idx % 13) as f32 - 6.0) * 0.125))
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut actual = DeviceBuffer::<f32>::zeroed(batch * rows).expect("batch output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        fp8_linear_f32_batch_into_on_stream(
            &input_device,
            &weight_device,
            actual.output(),
            batch,
            rows,
            cols,
            0.75,
            128,
            &stream,
        )
        .expect("batch FP8 linear");
        let actual = actual.copy_to_host(&stream).expect("batch output copy");
        for row in 0..batch {
            let row_input =
                DeviceBuffer::from_host(&input[row * cols..(row + 1) * cols]).expect("row input");
            let mut expected = DeviceBuffer::<f32>::zeroed(rows).expect("row output");
            fp8_linear_configured_f32_into_on_stream(
                &row_input,
                &weight_device,
                expected.output(),
                rows,
                cols,
                0.75,
                128,
                &stream,
            )
            .expect("row FP8 linear");
            assert_eq!(
                &actual[row * rows..(row + 1) * rows],
                &*expected.copy_to_host(&stream).expect("row output copy")
            );
        }
    }

    #[test]
    fn fp8_channel_scaled_linear_batch_matches_independent_rows() {
        let batch = 3usize;
        let rows = 5usize;
        let cols = 7usize;
        let input = (0..batch * cols)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let weight = (0..rows * cols)
            .map(|idx| format::cuda_e4m3_code(((idx % 13) as f32 - 6.0) * 0.125))
            .collect::<Vec<_>>();
        let scales = (0..rows)
            .map(|row| 0.5 + row as f32 * 0.125)
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let scales_device = DeviceBuffer::from_host(&scales).expect("scale upload");
        let mut actual = DeviceBuffer::<f32>::zeroed(batch * rows).expect("batch output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        fp8_linear_channel_scaled_f32_batch_into_on_stream(
            &input_device,
            &weight_device,
            &scales_device,
            actual.output(),
            batch,
            rows,
            cols,
            128,
            &stream,
        )
        .expect("batch channel-scaled FP8 linear");
        let actual = actual.copy_to_host(&stream).expect("batch output copy");
        for row in 0..batch {
            let row_input =
                DeviceBuffer::from_host(&input[row * cols..(row + 1) * cols]).expect("row input");
            let mut expected = DeviceBuffer::<f32>::zeroed(rows).expect("row output");
            fp8_linear_channel_scaled_f32_into_on_stream(
                &row_input,
                &weight_device,
                &scales_device,
                expected.output(),
                rows,
                cols,
                128,
                &stream,
            )
            .expect("row channel-scaled FP8 linear");
            assert_eq!(
                &actual[row * rows..(row + 1) * rows],
                &*expected.copy_to_host(&stream).expect("row output copy")
            );
        }
    }

    #[test]
    fn segmented_fp8_linears_match_cpu_reference() {
        let cols = 7usize;
        let rows = [5usize, 3, 2];
        let scales = [0.75f32, 0.5, 1.25];
        let input = (0..cols)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.25)
            .collect::<Vec<_>>();
        let weights = rows
            .iter()
            .enumerate()
            .map(|(segment, rows)| {
                (0..rows * cols)
                    .map(|idx| {
                        format::cuda_e4m3_code((((idx + segment * 3) % 13) as f32 - 6.0) * 0.125)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = weights
            .iter()
            .enumerate()
            .map(|(segment, weight)| {
                cpu_fp8_linear_f32(&input, weight, rows[segment], cols, scales[segment])
            })
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_devices = weights
            .iter()
            .map(|weight| DeviceBuffer::from_host(weight).expect("weight upload"))
            .collect::<Vec<_>>();
        let mut outputs = rows
            .iter()
            .map(|rows| DeviceBuffer::<f32>::zeroed(*rows).expect("output alloc"))
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream create");

        let [first, second, third] = outputs.as_mut_slice() else {
            unreachable!()
        };
        fp8_linear_triple_configured_f32_into_on_stream(
            &input_device,
            &weight_devices[0],
            &weight_devices[1],
            &weight_devices[2],
            first.output(),
            second.output(),
            third.output(),
            rows[0],
            rows[1],
            rows[2],
            cols,
            scales[0],
            scales[1],
            scales[2],
            128,
            &stream,
        )
        .expect("segmented triple enqueue");
        for (segment, output) in outputs.iter().enumerate() {
            let actual = output.copy_to_host(&stream).expect("output download");
            assert_close(
                &actual,
                &expected[segment],
                2.0e-6,
                &format!("segmented FP8 linear segment {segment}"),
            );
        }

        let (first, rest) = outputs.split_at_mut(1);
        fp8_linear_pair_configured_f32_into_on_stream(
            &input_device,
            &weight_devices[0],
            &weight_devices[1],
            first[0].output(),
            rest[0].output(),
            rows[0],
            rows[1],
            cols,
            scales[0],
            scales[1],
            128,
            &stream,
        )
        .expect("segmented pair enqueue");
        for segment in 0..2 {
            let actual = outputs[segment]
                .copy_to_host(&stream)
                .expect("output download");
            assert_close(
                &actual,
                &expected[segment],
                2.0e-6,
                &format!("segmented FP8 pair segment {segment}"),
            );
        }
    }

    #[test]
    fn fp8_linear_channel_scales_match_cpu_reference() {
        let rows = 5usize;
        let cols = 7usize;
        let input = (0..cols)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.25)
            .collect::<Vec<_>>();
        let weight = (0..rows * cols)
            .map(|idx| format::cuda_e4m3_code(((idx % 13) as f32 - 6.0) * 0.125))
            .collect::<Vec<_>>();
        let scales = vec![0.25f32, 0.5, 0.75, 1.0, 1.25];
        let mut expected = cpu_fp8_linear_f32(&input, &weight, rows, cols, 1.0);
        for (row, value) in expected.iter_mut().enumerate() {
            *value *= scales[row];
        }

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let scale_device = DeviceBuffer::from_host(&scales).expect("scale upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(rows).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");
        fp8_linear_channel_scaled_f32_into_on_stream(
            &input_device,
            &weight_device,
            &scale_device,
            output_device.output(),
            rows,
            cols,
            128,
            &stream,
        )
        .expect("channel-scaled FP8 linear enqueue");

        let output = output_device
            .copy_to_host(&stream)
            .expect("output download");
        assert_close(&output, &expected, 2.0e-6, "channel-scaled FP8 linear");
    }

    #[test]
    fn fp8_linear_dynamic_channel_scales_match_cpu_reference() {
        let rows = 5usize;
        let cols = 32usize;
        let input = (0..cols)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let weight = (0..rows * cols)
            .map(|idx| format::cuda_e4m3_code(((idx % 13) as f32 - 6.0) * 0.125))
            .collect::<Vec<_>>();
        let scales = vec![0.25f32, 0.5, 0.75, 1.0, 1.25];
        let input_scale = input.iter().fold(0.0f32, |max, value| max.max(value.abs())) / 448.0;
        let quantized_input = input
            .iter()
            .map(|value| {
                format::e4m3_value(format::cuda_e4m3_code(*value / input_scale)) * input_scale
            })
            .collect::<Vec<_>>();
        let mut expected = cpu_fp8_linear_f32(&quantized_input, &weight, rows, cols, 1.0);
        for (row, value) in expected.iter_mut().enumerate() {
            *value *= scales[row];
        }

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let scale_device = DeviceBuffer::from_host(&scales).expect("scale upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(rows).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");
        fp8_linear_channel_scaled_dynamic_f32_into_on_stream(
            &input_device,
            &weight_device,
            &scale_device,
            output_device.output(),
            rows,
            cols,
            &stream,
        )
        .expect("dynamic channel-scaled FP8 linear enqueue");

        let output = output_device
            .copy_to_host(&stream)
            .expect("output download");
        assert_close(
            &output,
            &expected,
            2.0e-6,
            "dynamic channel-scaled FP8 linear",
        );

        let mut input_scale_device = DeviceBuffer::<f32>::zeroed(1).expect("scale alloc");
        let mut precomputed_output = DeviceBuffer::<f32>::zeroed(rows).expect("output alloc");
        fp8_linear_channel_scaled_precomputed_dynamic_f32_into_on_stream(
            &input_device,
            &weight_device,
            &scale_device,
            &mut input_scale_device,
            precomputed_output.output(),
            rows,
            cols,
            &stream,
        )
        .expect("precomputed dynamic channel-scaled FP8 linear enqueue");
        assert_close(
            &precomputed_output
                .copy_to_host(&stream)
                .expect("precomputed output download"),
            &expected,
            2.0e-6,
            "precomputed dynamic channel-scaled FP8 linear",
        );

        let mut quantized_input = DeviceBuffer::<u8>::zeroed(cols).expect("quantized input alloc");
        let mut quantized_output = DeviceBuffer::<f32>::zeroed(rows).expect("output alloc");
        fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream(
            &input_device,
            &mut quantized_input,
            &weight_device,
            &scale_device,
            &mut input_scale_device,
            quantized_output.output(),
            rows,
            cols,
            &stream,
        )
        .expect("quantized dynamic channel-scaled FP8 linear enqueue");
        assert_close(
            &quantized_output
                .copy_to_host(&stream)
                .expect("quantized output download"),
            &expected,
            2.0e-6,
            "quantized dynamic channel-scaled FP8 linear",
        );
    }

    #[test]
    fn fp8_grouped_moe_matches_quantized_cpu_reference() {
        let experts = 3usize;
        let slots = 2usize;
        let hidden = 32usize;
        let intermediate = 5usize;
        let input = (0..hidden)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let indices = vec![2u32, 0];
        let make_weight = |salt: usize, rows: usize, cols: usize| {
            (0..rows * cols)
                .map(|idx| format::cuda_e4m3_code(((idx + salt) % 13) as f32 * 0.125 - 0.75))
                .collect::<Vec<_>>()
        };
        let make_scales = |salt: usize, rows: usize| {
            (0..rows)
                .map(|row| 0.25 + ((row + salt) % 5) as f32 * 0.125)
                .collect::<Vec<_>>()
        };
        let gate_host = (0..experts)
            .map(|expert| make_weight(expert, intermediate, hidden))
            .collect::<Vec<_>>();
        let up_host = (0..experts)
            .map(|expert| make_weight(expert + 3, intermediate, hidden))
            .collect::<Vec<_>>();
        let down_host = (0..experts)
            .map(|expert| make_weight(expert + 6, hidden, intermediate))
            .collect::<Vec<_>>();
        let gate_scale_host = (0..experts)
            .map(|expert| make_scales(expert, intermediate))
            .collect::<Vec<_>>();
        let up_scale_host = (0..experts)
            .map(|expert| make_scales(expert + 2, intermediate))
            .collect::<Vec<_>>();
        let down_scale_host = (0..experts)
            .map(|expert| make_scales(expert + 4, hidden))
            .collect::<Vec<_>>();

        let gate = gate_host
            .iter()
            .map(|weight| DeviceBuffer::from_host(weight).expect("gate upload"))
            .collect::<Vec<_>>();
        let up = up_host
            .iter()
            .map(|weight| DeviceBuffer::from_host(weight).expect("up upload"))
            .collect::<Vec<_>>();
        let down = down_host
            .iter()
            .map(|weight| DeviceBuffer::from_host(weight).expect("down upload"))
            .collect::<Vec<_>>();
        let gate_scales = gate_scale_host
            .iter()
            .map(|scale| DeviceBuffer::from_host(scale).expect("gate scale upload"))
            .collect::<Vec<_>>();
        let up_scales = up_scale_host
            .iter()
            .map(|scale| DeviceBuffer::from_host(scale).expect("up scale upload"))
            .collect::<Vec<_>>();
        let down_scales = down_scale_host
            .iter()
            .map(|scale| DeviceBuffer::from_host(scale).expect("down scale upload"))
            .collect::<Vec<_>>();
        let gate_table = DeviceBuffer::from_host(
            &gate
                .iter()
                .map(|buffer| buffer.as_const_ptr().cast::<u8>())
                .collect::<Vec<_>>(),
        )
        .expect("gate table upload");
        let up_table = DeviceBuffer::from_host(
            &up.iter()
                .map(|buffer| buffer.as_const_ptr().cast::<u8>())
                .collect::<Vec<_>>(),
        )
        .expect("up table upload");
        let down_table = DeviceBuffer::from_host(
            &down
                .iter()
                .map(|buffer| buffer.as_const_ptr().cast::<u8>())
                .collect::<Vec<_>>(),
        )
        .expect("down table upload");
        let gate_scale_table = DeviceBuffer::from_host(
            &gate_scales
                .iter()
                .map(|buffer| buffer.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("gate scale table upload");
        let up_scale_table = DeviceBuffer::from_host(
            &up_scales
                .iter()
                .map(|buffer| buffer.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("up scale table upload");
        let down_scale_table = DeviceBuffer::from_host(
            &down_scales
                .iter()
                .map(|buffer| buffer.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("down scale table upload");

        let stream = CudaStream::new_non_blocking().expect("stream create");
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let indices_device = DeviceBuffer::from_host(&indices).expect("indices upload");
        let mut input_fp8 = DeviceBuffer::<u8>::zeroed(hidden).expect("input FP8 alloc");
        let mut input_scale = DeviceBuffer::<f32>::zeroed(1).expect("input scale alloc");
        let mut gate_up_output =
            DeviceBuffer::<f32>::zeroed(slots * intermediate * 2).expect("gate/up alloc");
        let mut down_input =
            DeviceBuffer::<u8>::zeroed(slots * intermediate).expect("down input alloc");
        let mut down_input_scales =
            DeviceBuffer::<f32>::zeroed(slots).expect("down input scales alloc");
        let mut down_outputs = (0..slots)
            .map(|_| DeviceBuffer::<f32>::zeroed(hidden).expect("down output alloc"))
            .collect::<Vec<_>>();
        let down_output_table = DeviceBuffer::from_host(
            &down_outputs
                .iter_mut()
                .map(|buffer| buffer.as_mut_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("down output table upload");

        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            &input_device,
            &mut input_fp8,
            &mut input_scale,
            &stream,
        )
        .expect("input quantization");
        fp8_moe_grouped_gate_up_f32_into_on_stream(
            &indices_device,
            &input_fp8,
            &input_scale,
            &gate_table,
            &gate_scale_table,
            &up_table,
            &up_scale_table,
            gate_up_output.output(),
            intermediate,
            hidden,
            slots,
            &stream,
        )
        .expect("grouped gate/up");
        moe_silu_quantize_fp8_slots_f32_into_on_stream(
            &gate_up_output,
            &mut down_input,
            &mut down_input_scales,
            intermediate,
            slots,
            &stream,
        )
        .expect("SiLU quantization");
        fp8_moe_grouped_down_f32_into_on_stream(
            &indices_device,
            &down_input,
            &down_input_scales,
            &down_table,
            &down_scale_table,
            &down_output_table,
            hidden,
            intermediate,
            slots,
            &stream,
        )
        .expect("grouped down");

        let input_scale_value =
            input.iter().fold(0.0f32, |max, value| max.max(value.abs())) / 448.0;
        let quantized_input = input
            .iter()
            .map(|value| {
                format::e4m3_value(format::cuda_e4m3_code(*value / input_scale_value))
                    * input_scale_value
            })
            .collect::<Vec<_>>();
        for (slot, &expert) in indices.iter().enumerate() {
            let expert = expert as usize;
            let mut gate_output = cpu_fp8_linear_f32(
                &quantized_input,
                &gate_host[expert],
                intermediate,
                hidden,
                1.0,
            );
            let mut up_output = cpu_fp8_linear_f32(
                &quantized_input,
                &up_host[expert],
                intermediate,
                hidden,
                1.0,
            );
            for row in 0..intermediate {
                gate_output[row] *= gate_scale_host[expert][row];
                up_output[row] *= up_scale_host[expert][row];
            }
            let activated = gate_output
                .iter()
                .zip(&up_output)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect::<Vec<_>>();
            let scale = activated
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()))
                / 448.0;
            let quantized_activated = activated
                .iter()
                .map(|value| format::e4m3_value(format::cuda_e4m3_code(*value / scale)) * scale)
                .collect::<Vec<_>>();
            let mut expected = cpu_fp8_linear_f32(
                &quantized_activated,
                &down_host[expert],
                hidden,
                intermediate,
                1.0,
            );
            for row in 0..hidden {
                expected[row] *= down_scale_host[expert][row];
            }
            let actual = down_outputs[slot]
                .copy_to_host(&stream)
                .expect("down output download");
            assert_close(&actual, &expected, 2.0e-5, "grouped FP8 MoE down");
        }
    }

    #[test]
    fn nvfp4_w4a16_warp_row_schedules_match_block_per_row() {
        let rows = 37usize;
        let cols = 128usize;
        let input = (0..cols)
            .map(|idx| (((idx * 11) % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let mut packed = vec![0u8; rows * cols / 2];
        for idx in 0..rows * cols {
            let code = ((idx * 7 + 5) % 16) as u8;
            if idx & 1 == 0 {
                packed[idx / 2] = code;
            } else {
                packed[idx / 2] |= code << 4;
            }
        }
        let scales = (0..rows * (cols / 16))
            .map(|idx| [0x28u8, 0x30, 0x38, 0x40][idx % 4])
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let packed_device = DeviceBuffer::from_host(&packed).expect("packed weight");
        let scale_device = DeviceBuffer::from_host(&scales).expect("weight scale");
        let mut reference_device = DeviceBuffer::<f32>::zeroed(rows).expect("reference");
        let mut actual_device = DeviceBuffer::<f32>::zeroed(rows).expect("actual");
        let stream = CudaStream::new_non_blocking().expect("stream");
        nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream(
            &input_device,
            &packed_device,
            &scale_device,
            reference_device.output(),
            rows,
            cols,
            0.75,
            &stream,
        )
        .expect("block-per-row matvec");
        let reference = reference_device
            .copy_to_host(&stream)
            .expect("reference download");

        for warps in [4, 8, 16, 32] {
            nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
                &input_device,
                &packed_device,
                &scale_device,
                actual_device.output(),
                rows,
                cols,
                0.75,
                warps,
                &stream,
            )
            .expect("warp-row matvec");
            let actual = actual_device
                .copy_to_host(&stream)
                .expect("actual download");
            assert_close(&actual, &reference, 2.0e-5, "W4A16 warp-row schedule");
        }
    }

    #[test]
    fn nvfp4_w4a16_batch_matches_independent_rows() {
        let batch = 3usize;
        let out_features = 37usize;
        let in_features = 128usize;
        let input = (0..batch * in_features)
            .map(|idx| (((idx * 11) % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let mut packed = vec![0u8; out_features * in_features / 2];
        for idx in 0..out_features * in_features {
            let code = ((idx * 7 + 5) % 16) as u8;
            if idx & 1 == 0 {
                packed[idx / 2] = code;
            } else {
                packed[idx / 2] |= code << 4;
            }
        }
        let scales = (0..out_features * (in_features / 16))
            .map(|idx| [0x28u8, 0x30, 0x38, 0x40][idx % 4])
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let packed_device = DeviceBuffer::from_host(&packed).expect("packed weight");
        let scale_device = DeviceBuffer::from_host(&scales).expect("weight scale");
        let mut actual = DeviceBuffer::<f32>::zeroed(batch * out_features).expect("batch output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        nvfp4_w4a16_matvec_f32_batch_into_on_stream(
            &input_device,
            &packed_device,
            &scale_device,
            actual.output(),
            batch,
            out_features,
            in_features,
            0.75,
            &stream,
        )
        .expect("batch W4A16");
        let actual = actual.copy_to_host(&stream).expect("batch output copy");
        for row in 0..batch {
            let row_input =
                DeviceBuffer::from_host(&input[row * in_features..(row + 1) * in_features])
                    .expect("row input");
            let mut expected = DeviceBuffer::<f32>::zeroed(out_features).expect("row output");
            nvfp4_w4a16_matvec_f32_into_on_stream(
                &row_input,
                &packed_device,
                &scale_device,
                expected.output(),
                out_features,
                in_features,
                0.75,
                &stream,
            )
            .expect("row W4A16");
            assert_eq!(
                &actual[row * out_features..(row + 1) * out_features],
                &*expected.copy_to_host(&stream).expect("row output copy")
            );
        }
    }

    #[test]
    fn nvfp4_grouped_inputs_match_independent_shared_inputs() {
        const BATCH: usize = 3;
        const ROUTES: usize = 2;
        const EXPERTS: usize = 4;
        const OUT: usize = 37;
        const INPUT: usize = 128;
        let inputs = (0..BATCH * INPUT)
            .map(|index| (((index * 11) % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let indices = [0u32, 3, 2, 1, 3, 0];
        let packed = (0..EXPERTS)
            .map(|expert| {
                let mut values = vec![0u8; OUT * INPUT / 2];
                for index in 0..OUT * INPUT {
                    let code = ((index * 7 + expert * 3 + 5) % 16) as u8;
                    if index & 1 == 0 {
                        values[index / 2] = code;
                    } else {
                        values[index / 2] |= code << 4;
                    }
                }
                DeviceBuffer::from_host(&values).expect("packed expert")
            })
            .collect::<Vec<_>>();
        let scales = (0..EXPERTS)
            .map(|expert| {
                DeviceBuffer::from_host(
                    &(0..OUT * (INPUT / 16))
                        .map(|index| [0x28u8, 0x30, 0x38, 0x40][(index + expert) % 4])
                        .collect::<Vec<_>>(),
                )
                .expect("expert scales")
            })
            .collect::<Vec<_>>();
        let packed_table = DeviceBuffer::from_host(
            &packed
                .iter()
                .map(|weight| weight.as_const_ptr().cast::<u8>())
                .collect::<Vec<_>>(),
        )
        .expect("packed table");
        let scale_table = DeviceBuffer::from_host(
            &scales
                .iter()
                .map(|scale| scale.as_const_ptr().cast::<u8>())
                .collect::<Vec<_>>(),
        )
        .expect("scale table");
        let scale_2 = DeviceBuffer::from_host(&[0.75f32, 1.0, 1.25, 0.875]).expect("scale 2 table");
        let input_rows = (0..BATCH)
            .map(|row| {
                DeviceBuffer::from_host(&inputs[row * INPUT..(row + 1) * INPUT]).expect("input row")
            })
            .collect::<Vec<_>>();
        let input_table = DeviceBuffer::from_host(
            &(0..BATCH)
                .flat_map(|row| {
                    std::iter::repeat_n(input_rows[row].as_const_ptr().cast::<f32>(), ROUTES)
                })
                .collect::<Vec<_>>(),
        )
        .expect("input table");
        let mut actual_outputs = (0..BATCH * ROUTES)
            .map(|_| DeviceBuffer::<f32>::zeroed(OUT).expect("actual output"))
            .collect::<Vec<_>>();
        let actual_table = DeviceBuffer::from_host(
            &actual_outputs
                .iter_mut()
                .map(|output| output.as_mut_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("actual table");
        let indices_device = DeviceBuffer::from_host(&indices).expect("indices");
        let stream = CudaStream::new_non_blocking().expect("stream");
        nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
            &indices_device,
            &input_table,
            &packed_table,
            &scale_table,
            &scale_2,
            &actual_table,
            OUT,
            INPUT,
            &stream,
        )
        .expect("grouped inputs");

        for (row, input_row) in input_rows.iter().enumerate() {
            let begin = row * ROUTES;
            let row_indices =
                DeviceBuffer::from_host(&indices[begin..begin + ROUTES]).expect("row indices");
            let mut expected_outputs = (0..ROUTES)
                .map(|_| DeviceBuffer::<f32>::zeroed(OUT).expect("expected output"))
                .collect::<Vec<_>>();
            let expected_table = DeviceBuffer::from_host(
                &expected_outputs
                    .iter_mut()
                    .map(|output| output.as_mut_ptr().cast::<f32>())
                    .collect::<Vec<_>>(),
            )
            .expect("expected table");
            nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
                &row_indices,
                input_row,
                &packed_table,
                &scale_table,
                &scale_2,
                &expected_table,
                OUT,
                INPUT,
                &stream,
            )
            .expect("shared input routes");
            for route in 0..ROUTES {
                assert_eq!(
                    actual_outputs[begin + route]
                        .copy_to_host(&stream)
                        .expect("actual download"),
                    expected_outputs[route]
                        .copy_to_host(&stream)
                        .expect("expected download"),
                    "row {row} route {route}",
                );
            }
        }
    }

    #[test]
    fn nvfp4_w4a16_top1_matches_matvec_argmax() {
        let rows = 17usize;
        let cols = 64usize;
        let input = (0..cols)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let mut packed = vec![0u8; rows * cols / 2];
        for idx in 0..rows * cols {
            let code = ((idx * 5 + 3) % 16) as u8;
            let byte = &mut packed[idx / 2];
            if idx & 1 == 0 {
                *byte = (*byte & 0xf0) | code;
            } else {
                *byte = (*byte & 0x0f) | (code << 4);
            }
        }
        let scales = (0..rows * (cols / 16))
            .map(|idx| [0x38u8, 0x39, 0x3a, 0x3b][idx % 4])
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let packed_device = DeviceBuffer::from_host(&packed).expect("packed");
        let scale_device = DeviceBuffer::from_host(&scales).expect("scales");
        let mut logits_device = DeviceBuffer::<f32>::zeroed(rows).expect("logits");
        let scratch_len = rows.div_ceil(8) * 8;
        let scratch_value = DeviceBuffer::<f32>::zeroed(scratch_len).expect("scratch value");
        let scratch_index = DeviceBuffer::<u32>::zeroed(scratch_len).expect("scratch index");
        let out_index = DeviceBuffer::<u32>::zeroed(1).expect("out index");
        let out_value = DeviceBuffer::<f32>::zeroed(1).expect("out value");
        let stream = CudaStream::new_non_blocking().expect("stream");
        nvfp4_w4a16_matvec_f32_into_on_stream(
            &input_device,
            &packed_device,
            &scale_device,
            logits_device.output(),
            rows,
            cols,
            0.75,
            &stream,
        )
        .expect("matvec");
        nvfp4_w4a16_top1_f32_into_on_stream(
            &input_device,
            &packed_device,
            &scale_device,
            &scratch_value,
            &scratch_index,
            &out_index,
            &out_value,
            rows,
            cols,
            0.75,
            &stream,
        )
        .expect("top1");
        let logits = logits_device.copy_to_host(&stream).expect("logits");
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(ai, av), (bi, bv)| av.total_cmp(bv).then_with(|| bi.cmp(ai)))
            .map(|(idx, value)| (idx as u32, *value))
            .expect("expected");
        let actual_index = out_index.copy_to_host(&stream).expect("index")[0];
        let actual_value = out_value.copy_to_host(&stream).expect("value")[0];
        assert_eq!(actual_index, expected.0);
        assert!((actual_value - expected.1).abs() <= 1.0e-5);
    }

    #[test]
    fn nvfp4_w4a16_grouped_matvec_matches_cpu_reference() {
        let experts = 3usize;
        let groups = 4usize;
        let rows = 17usize;
        let cols = 64usize;
        let routes = vec![2u32, 0, 1, 2];
        let weight_scale_2 = vec![0.75f32, 1.25, 0.5];
        let input = (0..cols)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.125)
            .collect::<Vec<_>>();

        let mut packed_host = Vec::with_capacity(experts);
        let mut scales_host = Vec::with_capacity(experts);
        for expert in 0..experts {
            let mut packed = vec![0u8; rows * cols / 2];
            for idx in 0..rows * cols {
                let code = ((idx * 5 + expert * 3 + 1) % 16) as u8;
                if idx & 1 == 0 {
                    packed[idx / 2] = code;
                } else {
                    packed[idx / 2] |= code << 4;
                }
            }
            packed_host.push(packed);
            scales_host.push(
                (0..rows * (cols / 16))
                    .map(|idx| [0x30u8, 0x34, 0x38, 0x3c][(idx + expert) % 4])
                    .collect::<Vec<_>>(),
            );
        }

        let packed_device = packed_host
            .iter()
            .map(|values| DeviceBuffer::from_host(values).expect("packed upload"))
            .collect::<Vec<_>>();
        let scales_device = scales_host
            .iter()
            .map(|values| DeviceBuffer::from_host(values).expect("scale upload"))
            .collect::<Vec<_>>();
        let packed_ptrs = packed_device
            .iter()
            .map(|values| values.as_const_ptr().cast::<u8>())
            .collect::<Vec<_>>();
        let scale_ptrs = scales_device
            .iter()
            .map(|values| values.as_const_ptr().cast::<u8>())
            .collect::<Vec<_>>();
        let packed_table = DeviceBuffer::from_host(&packed_ptrs).expect("packed table");
        let scale_table = DeviceBuffer::from_host(&scale_ptrs).expect("scale table");
        let weight_scale_2_table =
            DeviceBuffer::from_host(&weight_scale_2).expect("weight scale 2 table");
        let routes_device = DeviceBuffer::from_host(&routes).expect("routes upload");
        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let mut outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(rows, 1).expect("output alloc"))
            .collect::<Vec<_>>();
        let output_ptrs = outputs
            .iter_mut()
            .map(F32Matrix::data_mut_ptr)
            .collect::<Vec<_>>();
        let output_table = DeviceBuffer::from_host(&output_ptrs).expect("output table");
        let stream = CudaStream::new_non_blocking().expect("stream");

        nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
            &routes_device,
            &input_device,
            &packed_table,
            &scale_table,
            &weight_scale_2_table,
            &output_table,
            rows,
            cols,
            &stream,
        )
        .expect("grouped W4A16 matvec");

        for (group, &expert) in routes.iter().enumerate() {
            let expert = expert as usize;
            let actual = outputs[group]
                .data()
                .copy_to_host(&stream)
                .expect("output download");
            let mut expected = vec![0.0f32; rows];
            for row in 0..rows {
                let mut sum = 0.0f32;
                for col in 0..cols {
                    let byte = packed_host[expert][row * (cols / 2) + col / 2];
                    let code = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                    let scale = scales_host[expert][row * (cols / 16) + col / 16];
                    sum += input[col] * format::e2m1_value(code) * format::e4m3_value(scale);
                }
                expected[row] = sum * weight_scale_2[expert];
            }
            assert_close(&actual, &expected, 2.0e-5, "grouped W4A16");
        }
    }

    #[test]
    fn nvfp4_w4a16_grouped_matvec_rejects_excess_shared_memory() {
        let max_shared_memory_bytes =
            crate::cuda::max_shared_memory_per_block().expect("shared memory limit");
        let cols = (max_shared_memory_bytes / std::mem::size_of::<f32>() + 1).div_ceil(16) * 16;
        let indices = DeviceBuffer::from_host(&[0u32]).expect("indices");
        let input = DeviceBuffer::<f32>::zeroed(cols).expect("input");
        let packed_weight_table =
            DeviceBuffer::from_host(&[std::ptr::null::<u8>()]).expect("weight table");
        let weight_scale_table =
            DeviceBuffer::from_host(&[std::ptr::null::<u8>()]).expect("scale table");
        let weight_scale_2_table = DeviceBuffer::from_host(&[1.0f32]).expect("scale 2 table");
        let output_table =
            DeviceBuffer::from_host(&[std::ptr::null_mut::<f32>()]).expect("output table");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let error = nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
            &indices,
            &input,
            &packed_weight_table,
            &weight_scale_table,
            &weight_scale_2_table,
            &output_table,
            16,
            cols,
            &stream,
        )
        .expect_err("oversized dynamic shared memory must be rejected");
        match error {
            Error::Shape {
                label,
                expected,
                actual,
            } => {
                assert_eq!(label, "NVFP4 grouped W4A16 shared memory");
                assert!(expected.contains(&max_shared_memory_bytes.to_string()));
                assert!(actual.contains(&(cols * std::mem::size_of::<f32>()).to_string()));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn qwen36_gdn_prep_matches_cpu_reference() {
        let key_heads = 1usize;
        let value_heads = 2usize;
        let head_dim = 128usize;
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv = (0..conv_dim)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01)
            .collect::<Vec<_>>();
        let conv_weight_f32 = (0..conv_dim * 4)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.0625)
            .collect::<Vec<_>>();
        let conv_weight = conv_weight_f32
            .iter()
            .map(|value| format::f32_to_bf16(*value))
            .collect::<Vec<_>>();
        let initial_state = vec![0.0f32; conv_dim * 3];
        let (expected_q, expected_k, expected_v, expected_state) = cpu_qwen36_gdn_prep(
            &qkv,
            &conv_weight,
            &initial_state,
            key_heads,
            value_heads,
            head_dim,
        );

        let qkv_device = DeviceBuffer::from_host(&qkv).expect("qkv upload");
        let conv_device = DeviceBuffer::from_host(&conv_weight).expect("conv upload");
        let mut q_device = DeviceBuffer::<f32>::zeroed(value_dim).expect("q alloc");
        let mut k_device = DeviceBuffer::<f32>::zeroed(value_dim).expect("k alloc");
        let mut v_device = DeviceBuffer::<f32>::zeroed(value_dim).expect("v alloc");
        let mut state_device = DeviceBuffer::<f32>::zeroed(conv_dim * 3).expect("state alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");

        qwen36_gdn_prep_into_on_stream(
            &qkv_device,
            &conv_device,
            q_device.output(),
            k_device.output(),
            v_device.output(),
            state_device.inout(),
            key_heads,
            value_heads,
            head_dim,
            &stream,
        )
        .expect("prep enqueue");

        assert_close(
            &q_device.copy_to_host(&stream).expect("q download"),
            &expected_q,
            2.0e-6,
            "prep q",
        );
        assert_close(
            &k_device.copy_to_host(&stream).expect("k download"),
            &expected_k,
            2.0e-6,
            "prep k",
        );
        assert_close(
            &v_device.copy_to_host(&stream).expect("v download"),
            &expected_v,
            2.0e-6,
            "prep v",
        );
        assert_close(
            &state_device.copy_to_host(&stream).expect("state download"),
            &expected_state,
            2.0e-6,
            "prep state",
        );

        let next_qkv = (0..conv_dim)
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.0075)
            .collect::<Vec<_>>();
        let (expected_next_q, expected_next_k, expected_next_v, expected_next_state) =
            cpu_qwen36_gdn_prep(
                &next_qkv,
                &conv_weight,
                &expected_state,
                key_heads,
                value_heads,
                head_dim,
            );
        let next_qkv_device = DeviceBuffer::from_host(&next_qkv).expect("next qkv upload");
        qwen36_gdn_prep_into_on_stream(
            &next_qkv_device,
            &conv_device,
            q_device.output(),
            k_device.output(),
            v_device.output(),
            state_device.inout(),
            key_heads,
            value_heads,
            head_dim,
            &stream,
        )
        .expect("next prep enqueue");
        assert_close(
            &q_device.copy_to_host(&stream).expect("next q download"),
            &expected_next_q,
            2.0e-6,
            "next prep q",
        );
        assert_close(
            &k_device.copy_to_host(&stream).expect("next k download"),
            &expected_next_k,
            2.0e-6,
            "next prep k",
        );
        assert_close(
            &v_device.copy_to_host(&stream).expect("next v download"),
            &expected_next_v,
            2.0e-6,
            "next prep v",
        );
        assert_close(
            &state_device
                .copy_to_host(&stream)
                .expect("next state download"),
            &expected_next_state,
            2.0e-6,
            "next prep state",
        );
    }

    #[test]
    fn qwen36_gdn_gate_matches_cpu_reference() {
        let heads = 4usize;
        let alpha = vec![-2.0, -0.25, 0.5, 3.0];
        let beta_input = vec![-1.0, 0.0, 1.0, 2.0];
        let a_log = vec![0.0, -0.5, 0.25, 1.0]
            .into_iter()
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let dt_bias = vec![0.125, -0.25, 0.5, -1.0]
            .into_iter()
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let (expected_gate, expected_beta) =
            cpu_qwen36_gdn_gate(&alpha, &beta_input, &a_log, &dt_bias);

        let alpha_device = DeviceBuffer::from_host(&alpha).expect("alpha upload");
        let beta_input_device = DeviceBuffer::from_host(&beta_input).expect("beta upload");
        let a_log_device = DeviceBuffer::from_host(&a_log).expect("a_log upload");
        let dt_bias_device = DeviceBuffer::from_host(&dt_bias).expect("dt_bias upload");
        let mut gate_device = DeviceBuffer::<f32>::zeroed(heads).expect("gate alloc");
        let mut beta_device = DeviceBuffer::<f32>::zeroed(heads).expect("beta alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");

        qwen36_gdn_gate_into_on_stream(
            &alpha_device,
            &beta_input_device,
            &a_log_device,
            &dt_bias_device,
            gate_device.output(),
            beta_device.output(),
            heads,
            &stream,
        )
        .expect("gate enqueue");

        assert_close(
            &gate_device.copy_to_host(&stream).expect("gate download"),
            &expected_gate,
            2.0e-6,
            "gate",
        );
        assert_close(
            &beta_device.copy_to_host(&stream).expect("beta download"),
            &expected_beta,
            2.0e-6,
            "beta",
        );
    }

    #[test]
    fn gated_rms_norm_f32_matches_cpu_reference() {
        let rows = 3usize;
        let cols = 5usize;
        let input = (0..rows * cols)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.125)
            .collect::<Vec<_>>();
        let gate = (0..rows * cols)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.25)
            .collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|idx| 1.0 + idx as f32 * 0.125)
            .collect::<Vec<_>>();
        let expected = cpu_gated_rms_norm(&input, &gate, &weight, rows, cols, 1.0e-6);

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let gate_device = DeviceBuffer::from_host(&gate).expect("gate upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut output_device = DeviceBuffer::<f32>::zeroed(rows * cols).expect("output alloc");
        let stream = CudaStream::new_non_blocking().expect("stream create");
        gated_rms_norm_f32_into_on_stream(
            &input_device,
            &gate_device,
            &weight_device,
            output_device.output(),
            rows,
            cols,
            1.0e-6,
            &stream,
        )
        .expect("gated rms enqueue");
        assert_close(
            &output_device
                .copy_to_host(&stream)
                .expect("output download"),
            &expected,
            2.0e-6,
            "gated rms",
        );
    }

    #[test]
    fn ling3_sigmoid_gated_rms_norm_matches_cpu_reference() {
        let rows = 3usize;
        let cols = 128usize;
        let input = (0..rows * cols)
            .map(|idx| ((idx * 17 % 97) as f32 - 48.0) * 0.0125)
            .collect::<Vec<_>>();
        let gate = (0..rows * cols)
            .map(|idx| ((idx * 19 % 101) as f32 - 50.0) * 0.025)
            .collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|idx| 0.75 + (idx % 11) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let eps = 1.0e-6;
        let expected = cpu_ling3_sigmoid_gated_rms_norm(&input, &gate, &weight, rows, cols, eps);
        let input = DeviceBuffer::from_host(&input).expect("input upload");
        let gate = DeviceBuffer::from_host(&gate).expect("gate upload");
        let weight = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut output = DeviceBuffer::zeroed(rows * cols).expect("output allocation");
        let stream = CudaStream::new_non_blocking().expect("stream");

        ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
            &input,
            &gate,
            &weight,
            output.output(),
            rows,
            cols,
            eps,
            &stream,
        )
        .expect("Ling sigmoid-gated RMSNorm");
        assert_close(
            &output.copy_to_host(&stream).expect("output download"),
            &expected,
            2.0e-6,
            "Ling sigmoid-gated RMSNorm",
        );
    }

    fn cpu_rms_norm(rows: usize, cols: usize, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let mean_square = input[start..end]
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                / cols as f64;
            let inv = ((mean_square as f32) + eps).sqrt().recip();
            for col in 0..cols {
                output[start + col] = input[start + col] * inv * weight[col];
            }
        }
        output
    }

    fn cpu_single_token_gqa(
        value: &[f32],
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let groups_per_kv = q_heads / kv_heads;
        let mut output = vec![0.0; q_heads * head_dim];
        for q_head in 0..q_heads {
            let kv_head = q_head / groups_per_kv;
            for dim in 0..head_dim {
                output[q_head * head_dim + dim] = value[kv_head * head_dim + dim];
            }
        }
        output
    }

    fn cpu_cached_gqa_attention(
        query: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        cache_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let groups_per_kv = q_heads / kv_heads;
        let kv_width = kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; q_heads * head_dim];
        for q_head in 0..q_heads {
            let kv_head = q_head / groups_per_kv;
            let q = &query[q_head * head_dim..(q_head + 1) * head_dim];
            let mut scores = Vec::with_capacity(cache_len);
            for row in 0..cache_len {
                let k_offset = row * kv_width + kv_head * head_dim;
                let k = &key_cache[k_offset..k_offset + head_dim];
                let score = q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() * scale;
                scores.push(score);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - max_score).exp())
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f32>();
            for dim in 0..head_dim {
                let mut accum = 0.0;
                for (row, weight) in weights.iter().enumerate() {
                    let v_offset = row * kv_width + kv_head * head_dim;
                    accum += weight * value_cache[v_offset + dim];
                }
                output[q_head * head_dim + dim] = accum / sum;
            }
        }
        output
    }

    fn cpu_rope_neox(
        rows: usize,
        head_dim: usize,
        input: &[f32],
        position: usize,
        theta: f32,
    ) -> Vec<f32> {
        let half = head_dim / 2;
        let mut output = input.to_vec();
        for row in 0..rows {
            let row_start = row * head_dim;
            for i in 0..half {
                let inv_freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let a = input[row_start + i];
                let b = input[row_start + i + half];
                output[row_start + i] = a * cos - b * sin;
                output[row_start + i + half] = a * sin + b * cos;
            }
        }
        output
    }

    fn cpu_rope_neox_partial(
        rows: usize,
        head_dim: usize,
        rotary_dim: usize,
        input: &[f32],
        position: usize,
        theta: f32,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut output = input.to_vec();
        for row in 0..rows {
            let row_start = row * head_dim;
            for i in 0..half {
                let inv_freq = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                let angle = position as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let a = input[row_start + i];
                let b = input[row_start + i + half];
                output[row_start + i] = a * cos - b * sin;
                output[row_start + i + half] = a * sin + b * cos;
            }
        }
        output
    }

    fn cpu_rope_neox_proportional(
        rows: usize,
        head_dim: usize,
        rotary_dim: usize,
        input: &[f32],
        position: usize,
        theta: f32,
    ) -> Vec<f32> {
        let half = head_dim / 2;
        let rotary_pairs = rotary_dim / 2;
        let mut output = input.to_vec();
        for row in 0..rows {
            let row_start = row * head_dim;
            for i in 0..rotary_pairs {
                let inv_freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let a = input[row_start + i];
                let b = input[row_start + i + half];
                output[row_start + i] = a * cos - b * sin;
                output[row_start + i + half] = a * sin + b * cos;
            }
        }
        output
    }

    /// CPU reference for IMRoPE/MRoPE matching llama.cpp's rope_multi IMRoPE
    /// branch with text positions `[pos_t, pos_h, pos_w, pos_extra]`.
    fn cpu_rope_imrope(
        rows: usize,
        head_dim: usize,
        rotary_dim: usize,
        sections: MropeSections,
        positions: [u32; 4],
        input: &[f32],
        theta: f32,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let sect_dims = sections.v0 + sections.v1 + sections.v2 + sections.v3;
        let pos = [
            positions[0] as f32,
            positions[1] as f32,
            positions[2] as f32,
            positions[3] as f32,
        ];
        let mut output = input.to_vec();
        for row in 0..rows {
            let row_start = row * head_dim;
            for i in 0..half {
                let sector = i % sect_dims;
                let section_base = if sector % 3 == 1 && sector < 3 * sections.v1 {
                    pos[1]
                } else if sector % 3 == 2 && sector < 3 * sections.v2 {
                    pos[2]
                } else if sector.is_multiple_of(3) && sector < 3 * sections.v0 {
                    pos[0]
                } else {
                    pos[3]
                };
                let inv_freq = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                let section_theta = section_base * inv_freq;
                let (sin, cos) = section_theta.sin_cos();
                let a = input[row_start + i];
                let b = input[row_start + i + half];
                output[row_start + i] = a * cos - b * sin;
                output[row_start + i + half] = a * sin + b * cos;
            }
        }
        output
    }

    fn cpu_gated_delta_net_128(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &mut [f32],
        heads: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; heads * 128];
        let scale = 1.0 / (128.0f32).sqrt();
        for head in 0..heads {
            let head_base = head * 128;
            let state_head_base = head * 128 * 128;
            let decay = gate[head].exp();
            for col in 0..128 {
                let state_col_base = state_head_base + col * 128;
                let mut kv = 0.0f32;
                for row in 0..128 {
                    kv += state[state_col_base + row] * k[head_base + row];
                }
                let delta = (v[head_base + col] - decay * kv) * beta[head];
                let mut acc = 0.0f32;
                for row in 0..128 {
                    let next = decay * state[state_col_base + row] + k[head_base + row] * delta;
                    state[state_col_base + row] = next;
                    acc += next * q[head_base + row];
                }
                output[head_base + col] = acc * scale;
            }
        }
        output
    }

    fn normalize_ling3_heads_128(values: &mut [f32], heads: usize) {
        for head in 0..heads {
            let row = &mut values[head * 128..(head + 1) * 128];
            let norm = (row.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
            for value in row {
                *value /= norm;
            }
        }
    }

    fn cpu_ling3_kda_prep(
        qkv: &[f32],
        conv_weight: &[u16],
        conv_state: &[f32],
        heads: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let projection = heads * 128;
        let conv_dim = projection * 3;
        let mut mixed = vec![0.0f32; conv_dim];
        let mut next_state = conv_state.to_vec();
        for index in 0..conv_dim {
            let mut value = qkv[index] * bf16_to_f32(conv_weight[index * 4 + 3]);
            for offset in 0..3 {
                value +=
                    conv_state[index * 3 + offset] * bf16_to_f32(conv_weight[index * 4 + offset]);
            }
            mixed[index] = value / (1.0 + (-value).exp());
            next_state[index * 3] = conv_state[index * 3 + 1];
            next_state[index * 3 + 1] = conv_state[index * 3 + 2];
            next_state[index * 3 + 2] = qkv[index];
        }
        let mut q = mixed[..projection].to_vec();
        let mut k = mixed[projection..projection * 2].to_vec();
        let v = mixed[projection * 2..].to_vec();
        normalize_ling3_heads_128(&mut q, heads);
        normalize_ling3_heads_128(&mut k, heads);
        (q, k, v, next_state)
    }

    fn cpu_ling3_kda_gate(
        raw_gate: &[f32],
        beta_input: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        heads: usize,
        lower_bound: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut gate = vec![0.0f32; heads * 128];
        let mut beta = vec![0.0f32; heads];
        for (head, beta) in beta.iter_mut().enumerate() {
            let a = a_log[head].exp();
            *beta = 1.0 / (1.0 + (-beta_input[head]).exp());
            for key in 0..128 {
                let index = head * 128 + key;
                let activated = a * (raw_gate[index] + dt_bias[index]);
                gate[index] = lower_bound / (1.0 + (-activated).exp());
            }
        }
        (gate, beta)
    }

    fn cpu_ling3_kda_128(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &mut [f32],
        heads: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; heads * 128];
        let scale = 128.0f32.sqrt().recip();
        for (head, &beta) in beta.iter().take(heads).enumerate() {
            let vector_offset = head * 128;
            let state_offset = head * 128 * 128;
            for key in 0..128 {
                let decay = gate[vector_offset + key].exp();
                for value in 0..128 {
                    state[state_offset + key * 128 + value] *= decay;
                }
            }
            for value in 0..128 {
                let prediction = (0..128)
                    .map(|key| state[state_offset + key * 128 + value] * k[vector_offset + key])
                    .sum::<f32>();
                let delta = (v[vector_offset + value] - prediction) * beta;
                for key in 0..128 {
                    state[state_offset + key * 128 + value] += k[vector_offset + key] * delta;
                }
                output[vector_offset + value] = (0..128)
                    .map(|key| state[state_offset + key * 128 + value] * q[vector_offset + key])
                    .sum::<f32>()
                    * scale;
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_ling3_mla_attention(
        query: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        cache_len: usize,
        heads: usize,
        qk_dim: usize,
        value_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; heads * value_dim];
        for head in 0..heads {
            let q = &query[head * qk_dim..(head + 1) * qk_dim];
            let scores = (0..cache_len)
                .map(|token| {
                    let offset = (token * heads + head) * qk_dim;
                    q.iter()
                        .zip(&key_cache[offset..offset + qk_dim])
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .sum::<f32>();
            for value_feature in 0..value_dim {
                output[head * value_dim + value_feature] = scores
                    .iter()
                    .enumerate()
                    .map(|(token, score)| {
                        let offset = (token * heads + head) * value_dim;
                        (score - maximum).exp() * value_cache[offset + value_feature]
                    })
                    .sum::<f32>()
                    / denominator;
            }
        }
        output
    }

    fn normalize_heads_128(values: &mut [f32], heads: usize) {
        for head in 0..heads {
            let row = &mut values[head * 128..(head + 1) * 128];
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            for value in row {
                *value /= norm;
            }
        }
    }

    fn cpu_fp8_linear_f32(
        input: &[f32],
        weight: &[u8],
        rows: usize,
        cols: usize,
        weight_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for col in 0..cols {
                sum += input[col] * format::e4m3_value(weight[row * cols + col]);
            }
            output[row] = sum * weight_scale;
        }
        output
    }

    fn cpu_qwen36_gdn_prep(
        qkv: &[f32],
        conv_weight: &[u16],
        state: &[f32],
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let key_dim = key_heads * head_dim;
        let value_dim = value_heads * head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let mut mixed = vec![0.0f32; conv_dim];
        let mut next_state = state.to_vec();
        for idx in 0..conv_dim {
            let value = state[idx * 3] * format::bf16_to_f32(conv_weight[idx * 4])
                + state[idx * 3 + 1] * format::bf16_to_f32(conv_weight[idx * 4 + 1])
                + state[idx * 3 + 2] * format::bf16_to_f32(conv_weight[idx * 4 + 2])
                + qkv[idx] * format::bf16_to_f32(conv_weight[idx * 4 + 3]);
            mixed[idx] = value / (1.0 + (-value).exp());
            next_state[idx * 3] = state[idx * 3 + 1];
            next_state[idx * 3 + 1] = state[idx * 3 + 2];
            next_state[idx * 3 + 2] = qkv[idx];
        }
        let mut q = vec![0.0f32; value_dim];
        let mut k = vec![0.0f32; value_dim];
        let mut v = vec![0.0f32; value_dim];
        for repeat in 0..(value_heads / key_heads) {
            q[repeat * key_dim..(repeat + 1) * key_dim].copy_from_slice(&mixed[..key_dim]);
            k[repeat * key_dim..(repeat + 1) * key_dim]
                .copy_from_slice(&mixed[key_dim..key_dim * 2]);
        }
        // Reorder V from grouped-by-K to tiled order to match Q/K repeat layout.
        // Grouped: [K0_V0, K0_V1, ..., K1_V0, K1_V1, ...]
        // Tiled:   [K0_V0, K1_V0, ..., K0_V1, K1_V1, ...]
        let v_per_k = value_heads / key_heads;
        for v_k_head in 0..value_heads {
            let k_head = v_k_head / v_per_k;
            let v_sub = v_k_head % v_per_k;
            for dim in 0..head_dim {
                v[v_sub * key_heads * head_dim + k_head * head_dim + dim] =
                    mixed[key_dim * 2 + v_k_head * head_dim + dim];
            }
        }
        l2_norm_heads(&mut q, value_heads, head_dim);
        l2_norm_heads(&mut k, value_heads, head_dim);
        (q, k, v, next_state)
    }

    fn cpu_qwen36_gdn_gate(
        alpha: &[f32],
        beta_input: &[f32],
        a_log: &[u16],
        dt_bias: &[u16],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut gate = vec![0.0f32; alpha.len()];
        let mut beta = vec![0.0f32; alpha.len()];
        for idx in 0..alpha.len() {
            let dt = alpha[idx] + format::bf16_to_f32(dt_bias[idx]);
            let softplus = (1.0 + (-dt.abs()).exp()).ln() + dt.max(0.0);
            gate[idx] = -format::bf16_to_f32(a_log[idx]).exp() * softplus;
            beta[idx] = 1.0 / (1.0 + (-beta_input[idx]).exp());
        }
        (gate, beta)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_qwen36_full_attn_prep(
        q_full: &[f32],
        k_raw: &[f32],
        q_norm: &[f32],
        k_norm: &[f32],
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0f32; q_heads * head_dim];
        let mut gate = vec![0.0f32; q_heads * head_dim];
        let mut k = vec![0.0f32; kv_heads * head_dim];

        for head in 0..q_heads {
            let q_full_base = head * head_dim * 2;
            let out_base = head * head_dim;
            let mean_square = q_full[q_full_base..q_full_base + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / head_dim as f32;
            let inv_rms = (mean_square + eps).sqrt().recip();
            for dim in 0..head_dim {
                q[out_base + dim] = q_full[q_full_base + dim] * inv_rms * q_norm[dim];
                gate[out_base + dim] = q_full[q_full_base + head_dim + dim];
            }
        }

        for head in 0..kv_heads {
            let base = head * head_dim;
            let mean_square = k_raw[base..base + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / head_dim as f32;
            let inv_rms = (mean_square + eps).sqrt().recip();
            for dim in 0..head_dim {
                k[base + dim] = k_raw[base + dim] * inv_rms * k_norm[dim];
            }
        }

        (q, gate, k)
    }

    fn cpu_gated_rms_norm(
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        for row in 0..rows {
            let start = row * cols;
            let mean_square = input[start..start + cols]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / cols as f32;
            let inv_rms = (mean_square + eps).sqrt().recip();
            for col in 0..cols {
                let gate_value = gate[start + col];
                let silu_gate = gate_value / (1.0 + (-gate_value).exp());
                output[start + col] = input[start + col] * inv_rms * weight[col] * silu_gate;
            }
        }
        output
    }

    fn cpu_ling3_sigmoid_gated_rms_norm(
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        for row in 0..rows {
            let start = row * cols;
            let mean_square = input[start..start + cols]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / cols as f32;
            let inv_rms = (mean_square + eps).sqrt().recip();
            for (col, &weight) in weight.iter().take(cols).enumerate() {
                let index = start + col;
                let sigmoid_gate = 1.0 / (1.0 + (-gate[index]).exp());
                output[index] = input[index] * inv_rms * weight * sigmoid_gate;
            }
        }
        output
    }

    fn l2_norm_heads(values: &mut [f32], heads: usize, head_dim: usize) {
        for head in 0..heads {
            let start = head * head_dim;
            let sum = values[start..start + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let inv = 1.0 / sum.sqrt().max(1.0e-6);
            for value in &mut values[start..start + head_dim] {
                *value *= inv;
            }
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "{label}[{idx}] mismatch: actual={} expected={} tolerance={}",
                actual,
                expected,
                tolerance
            );
        }
    }
}
