#![allow(clippy::too_many_arguments)]

//! CUDA kernels for non-GEMM decode operations.

use crate::cuda::{
    CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda, max_shared_memory_per_block,
};
use crate::error::{Error, Result};
use crate::ffi;
use crate::format;
use crate::matrix::{Bf16Matrix, Nvfp4Matrix};

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
    if input.len() != input_len || output.len() != input_len {
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
    mut output: DeviceOutput<'_, f32>,
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
    if gate.is_empty() || gate.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "SiLU multiply",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{} values", gate.len()),
        });
    }

    unsafe {
        check_cuda(
            "infer_silu_mul_f32_on_stream",
            ffi::infer_silu_mul_f32_on_stream(
                gate.ptr,
                up.ptr,
                output.buffer_mut().ptr,
                gate.len() as u32,
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

#[allow(missing_docs)]
pub fn fill_f32_into_on_stream(
    mut output: DeviceOutput<'_, f32>,
    value: f32,
    stream: &CudaStream,
) -> Result<()> {
    if output.is_empty() || output.len() > u32::MAX as usize || !value.is_finite() {
        return Err(Error::Shape {
            label: "fill f32",
            expected: "non-empty u32-sized output and finite value".to_string(),
            actual: format!("len={} value={value}", output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_fill_f32_on_stream",
            ffi::infer_fill_f32_on_stream(
                output.buffer_mut().ptr,
                value,
                output.len() as u32,
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
    mut output: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if gate.len() != input.len() || output.len() != input.len() {
        return Err(Error::Shape {
            label: "sigmoid multiply buffers",
            expected: format!("gate/input/output={} values", input.len()),
            actual: format!(
                "gate={} input={} output={}",
                gate.len(),
                input.len(),
                output.len()
            ),
        });
    }
    if input.is_empty() || input.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "sigmoid multiply dimensions",
            expected: "non-empty u32-sized length".to_string(),
            actual: format!("len={}", input.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_sigmoid_mul_f32_on_stream",
            ffi::infer_sigmoid_mul_f32_on_stream(
                gate.ptr,
                input.ptr,
                output.buffer_mut().ptr,
                input.len() as u32,
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
    if input.len() != len || output.len() != len {
        return Err(Error::Shape {
            label: "sequence RoPE buffers",
            expected: format!("{len} values"),
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

/// Enqueues elementwise f32 addition into an existing output buffer on
/// `stream`.
pub fn add_f32_into_on_stream(
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
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

    unsafe {
        check_cuda(
            "infer_add_f32_on_stream",
            ffi::infer_add_f32_on_stream(
                left.ptr,
                right.ptr,
                output.buffer_mut().ptr,
                left.len() as u32,
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
    if input.len() != len {
        return Err(Error::Shape {
            label: "NVFP4 device quantization input",
            expected: format!("{len} values"),
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

/// Rounds a device-resident f32 buffer in place to BF16 precision, stored as f32.
pub fn round_f32_to_bf16_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if values.is_empty() || values.len() > u32::MAX as usize {
        return Err(Error::Shape {
            label: "F32 to BF16 round length",
            expected: "1..=u32::MAX values".to_string(),
            actual: format!("{} values", values.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_round_f32_to_bf16_in_place_on_stream",
            ffi::infer_round_f32_to_bf16_in_place_on_stream(
                values.buffer_mut().ptr,
                values.len() as u32,
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
        || input.len() != len
        || gate.len() != len
        || output.len() != len
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{F32Matrix, synchronize_device};

    #[test]
    fn rms_norm_f32_matches_cpu_reference() {
        let rows = 3;
        let cols = 128;
        let eps = 1.0e-6;
        let input = (0..rows * cols)
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.125)
            .collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|idx| 0.5 + (idx % 7) as f32 * 0.03125)
            .collect::<Vec<_>>();

        let input_device = DeviceBuffer::from_host(&input).expect("input upload");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight upload");
        let mut output_device = DeviceBuffer::zeroed(rows * cols).expect("RMSNorm output alloc");
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
                "RMSNorm mismatch at {idx}: actual={actual} expected={expected} error={error}"
            );
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
