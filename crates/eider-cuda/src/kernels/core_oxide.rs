//! cuda-oxide launches for shared dense-model CUDA operations.

use crate::cuda_oxide::{Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::OnceLock;

const THREADS: u32 = 256;

struct Functions {
    fill: Kernel,
    add: Kernel,
    scaled_add: Kernel,
    silu_mul_halves: Kernel,
    silu_mul_halves_batch: Kernel,
    sigmoid_mul: Kernel,
    round_to_bf16: Kernel,
    f32_to_bf16: Kernel,
    bf16_to_f32: Kernel,
    copy_bf16_rows_to_f32_indexed: Kernel,
    copy_fp8_rows_to_f32_indexed: Kernel,
    concat_f32_rows: Kernel,
    fp8_linear_channel_scaled: Kernel,
    fp8_linear_quantized_channel_scaled: Kernel,
    bf16_linear_logits_batch: Kernel,
    bf16_linear_logits_batch_scalar: Kernel,
    quantize_nvfp4_col_major: Kernel,
    rms_norm: Kernel,
    gated_rms_norm: Kernel,
    gated_rms_norm_quantize_nvfp4: Kernel,
    qwen_full_attn_prep: Kernel,
    qwen_gdn_prep: Kernel,
    qwen_gdn_prep_batch: Kernel,
    qwen_gdn_prep_chunks: Kernel,
    qwen_gdn_prep_chunks_bf16: Kernel,
    qwen_gdn_update_conv_state: Kernel,
    l2_norm_heads_128: Kernel,
    l2_norm_heads_128_bf16: Kernel,
    qwen_gdn_gate: Kernel,
    qwen_gdn_gate_batch: Kernel,
    qwen_gdn_gate_batch_bf16: Kernel,
    qwen_gdn_gate_paired_batch: Kernel,
    qwen_gdn_gate_paired_batch_bf16: Kernel,
    gated_delta_net_128: Kernel,
    gated_delta_net_128_batch: Kernel,
    gated_delta_net_128_chunks: Kernel,
    gated_delta_net_128_chunks_multiwarp: Kernel,
    gather_f32_pointer_rows: Kernel,
    scatter_f32_pointer_rows: Kernel,
    dflash2_capture: Kernel,
    dflash2_grouped_conv: Kernel,
    dflash2_noncausal_attention: Kernel,
    dflash2_hidden_projection: Kernel,
    sampling_logits_topk: Kernel,
    sampling_keys_topk: Kernel,
    sampling_finalize: Kernel,
    dflash2_select_path: Kernel,
    rope_partial: Kernel,
    rope_partial_indexed: Kernel,
    rope_partial_sequence: Kernel,
    rope_imrope: Kernel,
    rope_imrope_indexed: Kernel,
    rope_imrope_text_batch: Kernel,
    quantize_fp8_dynamic: Kernel,
    scale_channel_scalar: Kernel,
    scale_channel_row_scalar: Kernel,
    argmax: Kernel,
    mask_logits_batch: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            fill: Kernel::load(c"fill_f32")?,
            add: Kernel::load(c"add_f32")?,
            scaled_add: Kernel::load(c"scaled_add_f32")?,
            silu_mul_halves: Kernel::load(c"silu_mul_halves_f32")?,
            silu_mul_halves_batch: Kernel::load(c"silu_mul_halves_f32_batch")?,
            sigmoid_mul: Kernel::load(c"sigmoid_mul_f32")?,
            round_to_bf16: Kernel::load(c"round_f32_to_bf16")?,
            f32_to_bf16: Kernel::load(c"f32_to_bf16")?,
            bf16_to_f32: Kernel::load(c"convert_bf16_to_f32")?,
            copy_bf16_rows_to_f32_indexed: Kernel::load(c"copy_bf16_rows_to_f32_indexed")?,
            copy_fp8_rows_to_f32_indexed: Kernel::load(c"copy_fp8_rows_to_f32_indexed")?,
            concat_f32_rows: Kernel::load(c"concat_f32_rows")?,
            fp8_linear_channel_scaled: Kernel::load(c"fp8_linear_channel_scaled_f32")?,
            fp8_linear_quantized_channel_scaled: Kernel::load(
                c"fp8_linear_quantized_channel_scaled_f32",
            )?,
            bf16_linear_logits_batch: Kernel::load(c"bf16_linear_logits_f32_batch")?,
            bf16_linear_logits_batch_scalar: Kernel::load(c"bf16_linear_logits_f32_batch_scalar")?,
            quantize_nvfp4_col_major: Kernel::load(c"quantize_nvfp4_col_major_f32")?,
            rms_norm: Kernel::load(c"rms_norm_f32")?,
            gated_rms_norm: Kernel::load(c"gated_rms_norm_f32")?,
            gated_rms_norm_quantize_nvfp4: Kernel::load(c"gated_rms_norm_quantize_nvfp4_f32")?,
            qwen_full_attn_prep: Kernel::load(c"qwen36_full_attn_prep_f32")?,
            qwen_gdn_prep: Kernel::load(c"qwen36_gdn_prep_f32")?,
            qwen_gdn_prep_batch: Kernel::load(c"qwen36_gdn_prep_batch_f32")?,
            qwen_gdn_prep_chunks: Kernel::load(c"qwen36_gdn_prep_chunks_f32")?,
            qwen_gdn_prep_chunks_bf16: Kernel::load(c"qwen36_gdn_prep_chunks_bf16")?,
            qwen_gdn_update_conv_state: Kernel::load(c"qwen36_gdn_update_conv_state")?,
            l2_norm_heads_128: Kernel::load(c"l2_norm_heads_128_f32")?,
            l2_norm_heads_128_bf16: Kernel::load(c"l2_norm_heads_128_bf16")?,
            qwen_gdn_gate: Kernel::load(c"qwen36_gdn_gate_f32")?,
            qwen_gdn_gate_batch: Kernel::load(c"qwen36_gdn_gate_batch_f32")?,
            qwen_gdn_gate_batch_bf16: Kernel::load(c"qwen36_gdn_gate_batch_bf16")?,
            qwen_gdn_gate_paired_batch: Kernel::load(c"qwen36_gdn_gate_paired_batch_f32")?,
            qwen_gdn_gate_paired_batch_bf16: Kernel::load(c"qwen36_gdn_gate_paired_batch_bf16")?,
            gated_delta_net_128: Kernel::load(c"gated_delta_net_128_f32")?,
            gated_delta_net_128_batch: Kernel::load(c"gated_delta_net_128_f32_batch")?,
            gated_delta_net_128_chunks: Kernel::load(c"gated_delta_net_128_f32_chunks")?,
            gated_delta_net_128_chunks_multiwarp: Kernel::load(
                c"gated_delta_net_128_f32_chunks_multiwarp",
            )?,
            gather_f32_pointer_rows: Kernel::load(c"gather_f32_pointer_rows")?,
            scatter_f32_pointer_rows: Kernel::load(c"scatter_f32_pointer_rows")?,
            dflash2_capture: Kernel::load(c"dflash2_capture_f32")?,
            dflash2_grouped_conv: Kernel::load(c"dflash2_grouped_conv_f32")?,
            dflash2_noncausal_attention: Kernel::load(c"dflash2_noncausal_attention_f32")?,
            dflash2_hidden_projection: Kernel::load(c"dflash2_hidden_projection_f32")?,
            sampling_logits_topk: Kernel::load(c"sampling_logits_topk_f32")?,
            sampling_keys_topk: Kernel::load(c"sampling_keys_topk")?,
            sampling_finalize: Kernel::load(c"sampling_finalize_f32")?,
            dflash2_select_path: Kernel::load(c"dflash2_select_path_f32")?,
            rope_partial: Kernel::load(c"rope_neox_partial_f32")?,
            rope_partial_indexed: Kernel::load(c"rope_neox_partial_indexed_f32")?,
            rope_partial_sequence: Kernel::load(c"rope_neox_partial_sequence_f32")?,
            rope_imrope: Kernel::load(c"rope_imrope_f32")?,
            rope_imrope_indexed: Kernel::load(c"rope_imrope_indexed_f32")?,
            rope_imrope_text_batch: Kernel::load(c"rope_imrope_text_batch_f32")?,
            quantize_fp8_dynamic: Kernel::load(c"quantize_fp8_e4m3_dynamic_f32")?,
            scale_channel_scalar: Kernel::load(c"scale_channel_f32_device_scalar")?,
            scale_channel_row_scalar: Kernel::load(c"scale_channel_f32_device_row_scalar")?,
            argmax: Kernel::load(c"argmax_f32")?,
            mask_logits_batch: Kernel::load(c"mask_logits_f32_batch")?,
        })
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide core module",
            detail: error.to_string(),
        }),
    }
}

fn grid(len: u32) -> [u32; 3] {
    [len.div_ceil(THREADS), 1, 1]
}

fn block() -> [u32; 3] {
    [THREADS, 1, 1]
}

/// Launches an f32 fill.
///
/// # Safety
///
/// `output` must contain at least `len` values and remain valid until `stream`
/// completes.
pub(crate) unsafe fn fill(
    output: *mut f32,
    value: f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut output_arg = output;
    let mut value_arg = value;
    let mut len_arg = len;
    let mut parameters = [
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut value_arg as *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.fill.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches elementwise f32 addition.
///
/// # Safety
///
/// All buffers must contain at least `len` values and remain valid until
/// `stream` completes.
pub(crate) unsafe fn add(
    left: *const f32,
    right: *const f32,
    output: *mut f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut left_arg = left;
    let mut right_arg = right;
    let mut output_arg = output;
    let mut len_arg = len;
    let mut parameters = [
        (&mut left_arg as *mut *const f32).cast::<c_void>(),
        (&mut right_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.add.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches an in-place scaled add.
///
/// # Safety
///
/// Both buffers must contain at least `len` values and remain valid until
/// `stream` completes.
pub(crate) unsafe fn scaled_add(
    input: *const f32,
    output: *mut f32,
    scale: f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut scale_arg = scale;
    let mut len_arg = len;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut scale_arg as *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.scaled_add.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one-row SwiGLU activation.
///
/// # Safety
///
/// `gate_up` must contain `2 * len` values. `output` must contain `len` values.
pub(crate) unsafe fn silu_mul_halves(
    gate_up: *const f32,
    output: *mut f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut gate_up_arg = gate_up;
    let mut output_arg = output;
    let mut len_arg = len;
    let mut parameters = [
        (&mut gate_up_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.silu_mul_halves.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches row-major batched SwiGLU activation.
///
/// # Safety
///
/// The buffers must satisfy the row-major `[gate, up]` dimensions.
pub(crate) unsafe fn silu_mul_halves_batch(
    gate_up: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut gate_up_arg = gate_up;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "cuda-oxide batched SwiGLU",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let mut parameters = [
        (&mut gate_up_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.silu_mul_halves_batch.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches elementwise sigmoid gating.
///
/// # Safety
///
/// All buffers must contain at least `len` values and remain valid until
/// `stream` completes.
pub(crate) unsafe fn sigmoid_mul(
    gate: *const f32,
    input: *const f32,
    output: *mut f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut gate_arg = gate;
    let mut input_arg = input;
    let mut output_arg = output;
    let mut len_arg = len;
    let mut parameters = [
        (&mut gate_arg as *mut *const f32).cast::<c_void>(),
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.sigmoid_mul.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches f32-to-BF16 rounding with f32 output storage.
///
/// # Safety
///
/// Both buffers must contain at least `len` values and may alias exactly.
pub(crate) unsafe fn round_to_bf16(
    input: *const f32,
    output: *mut f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut len_arg = len;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.round_to_bf16.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches f32-to-BF16 storage conversion.
///
/// # Safety
///
/// Both buffers must contain at least `len` values and remain valid until
/// `stream` completes.
pub(crate) unsafe fn f32_to_bf16(
    input: *const f32,
    output: *mut u16,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut len_arg = len;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut u16).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.f32_to_bf16.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches BF16-storage to f32 conversion.
///
/// # Safety
///
/// Both buffers must contain at least `len` values and remain valid until
/// `stream` completes.
pub(crate) unsafe fn bf16_to_f32(
    input: *const u16,
    output: *mut f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut len_arg = len;
    let mut parameters = [
        (&mut input_arg as *mut *const u16).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.bf16_to_f32.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches indexed BF16 embedding gather into f32 output.
///
/// # Safety
///
/// `input`, `rows`, and `output` must satisfy the supplied matrix dimensions
/// and remain valid until `stream` completes. Every row index must be valid.
pub(crate) unsafe fn copy_bf16_rows_to_f32_indexed(
    input: *const u16,
    rows: *const u32,
    output: *mut f32,
    row_count: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut rows_arg = rows;
    let mut output_arg = output;
    let mut row_count_arg = row_count;
    let mut cols_arg = cols;
    let len = row_count.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "cuda-oxide BF16 embedding gather",
        expected: "row_count * cols without overflow".to_string(),
        actual: format!("row_count={row_count} cols={cols}"),
    })?;
    let mut parameters = [
        (&mut input_arg as *mut *const u16).cast::<c_void>(),
        (&mut rows_arg as *mut *const u32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.copy_bf16_rows_to_f32_indexed.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches indexed scaled-E4M3 embedding gather into f32 output.
///
/// # Safety
///
/// `input`, `row_scales`, `rows`, and `output` must satisfy the supplied matrix
/// dimensions and remain valid until `stream` completes. Every row index must
/// be valid.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn copy_fp8_rows_to_f32_indexed(
    input: *const u8,
    row_scales: *const f32,
    rows: *const u32,
    output: *mut f32,
    row_count: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut row_scales_arg = row_scales;
    let mut rows_arg = rows;
    let mut output_arg = output;
    let mut row_count_arg = row_count;
    let mut cols_arg = cols;
    let len = row_count.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "cuda-oxide FP8 embedding gather",
        expected: "row_count * cols without overflow".to_string(),
        actual: format!("row_count={row_count} cols={cols}"),
    })?;
    let mut parameters = [
        (&mut input_arg as *mut *const u8).cast::<c_void>(),
        (&mut row_scales_arg as *mut *const f32).cast::<c_void>(),
        (&mut rows_arg as *mut *const u32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.copy_fp8_rows_to_f32_indexed.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches row-major f32 concatenation.
///
/// # Safety
///
/// Both input buffers must contain `rows * cols` values. The output must
/// contain twice that count. All buffers must remain valid until completion.
pub(crate) unsafe fn concat_f32_rows(
    left: *const f32,
    right: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut left_arg = left;
    let mut right_arg = right;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| Error::Shape {
            label: "cuda-oxide row concatenation",
            expected: "2 * rows * cols without overflow".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
    let mut parameters = [
        (&mut left_arg as *mut *const f32).cast::<c_void>(),
        (&mut right_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.concat_f32_rows.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one-row channel-scaled E4M3 W8A16 projection.
///
/// # Safety
///
/// The buffers must satisfy the supplied matrix dimensions and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fp8_linear_channel_scaled(
    input: *const f32,
    weight: *const u8,
    channel_scale: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    threads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut weight_arg = weight;
    let mut channel_scale_arg = channel_scale;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut channel_scale_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.fp8_linear_channel_scaled.launch(
            LaunchConfig::new([rows, 1, 1], [threads, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one-row channel-scaled E4M3 tensor projection.
///
/// # Safety
///
/// The buffers must satisfy the supplied matrix dimensions and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fp8_linear_quantized_channel_scaled(
    input: *const u8,
    weight: *const u8,
    channel_scale: *const f32,
    input_scale: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    threads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut weight_arg = weight;
    let mut channel_scale_arg = channel_scale;
    let mut input_scale_arg = input_scale;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut parameters = [
        (&mut input_arg as *mut *const u8).cast::<c_void>(),
        (&mut weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut channel_scale_arg as *mut *const f32).cast::<c_void>(),
        (&mut input_scale_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.fp8_linear_quantized_channel_scaled.launch(
            LaunchConfig::new([rows, 1, 1], [threads, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches a batched f32 by BF16 row-major projection.
///
/// # Safety
///
/// The buffers must satisfy the supplied matrix dimensions and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn bf16_linear_logits_batch(
    input: *const f32,
    weight: *const u16,
    logits: *mut f32,
    batch_size: u32,
    rows: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut weight_arg = weight;
    let mut logits_arg = logits;
    let mut batch_size_arg = batch_size;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut logits_arg as *mut *mut f32).cast::<c_void>(),
        (&mut batch_size_arg as *mut u32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    let kernel = if cols.is_multiple_of(4) {
        &functions()?.bf16_linear_logits_batch
    } else {
        &functions()?.bf16_linear_logits_batch_scalar
    };
    unsafe {
        kernel.launch(
            LaunchConfig::new([rows.div_ceil(8), batch_size.div_ceil(8), 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches column-major f32 activation quantization to cuBLASLt NVFP4.
///
/// # Safety
///
/// The buffers must satisfy the supplied column-major dimensions and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn quantize_nvfp4_col_major(
    input: *const f32,
    packed: *mut u8,
    scales: *mut u8,
    rows: u32,
    cols: u32,
    input_scale: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut packed_arg = packed;
    let mut scales_arg = scales;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut input_scale_arg = input_scale;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut packed_arg as *mut *mut u8).cast::<c_void>(),
        (&mut scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
        (&mut input_scale_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.quantize_nvfp4_col_major.launch(
            LaunchConfig::new([cols * rows.div_ceil(16), 1, 1], [32, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches row-wise RMSNorm.
///
/// # Safety
///
/// The buffers must satisfy the supplied row and column dimensions.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rms_norm(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    eps: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut weight_arg = weight;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut eps_arg = eps;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
        (&mut eps_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.rms_norm.launch(
            LaunchConfig::new([rows, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches row-wise RMSNorm with a SiLU gate.
///
/// # Safety
///
/// The buffers must satisfy the supplied row and column dimensions.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gated_rms_norm(
    input: *const f32,
    gate: *const f32,
    weight: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
    eps: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut gate_arg = gate;
    let mut weight_arg = weight;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut eps_arg = eps;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut gate_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
        (&mut eps_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.gated_rms_norm.launch(
            LaunchConfig::new([rows, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches per-head gated RMSNorm with direct NVFP4 quantization.
///
/// # Safety
///
/// The input and gate contain `rows * heads * 128` values, the weight contains
/// 128 values, and the output buffers use Eider's column-major NVFP4 layout.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gated_rms_norm_quantize_nvfp4(
    input: *const f32,
    gate: *const f32,
    weight: *const f32,
    packed: *mut u8,
    scales: *mut u8,
    rows: u32,
    heads: u32,
    eps: f32,
    input_scale: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut gate_arg = gate;
    let mut weight_arg = weight;
    let mut packed_arg = packed;
    let mut scales_arg = scales;
    let mut heads_arg = heads;
    let mut eps_arg = eps;
    let mut input_scale_arg = input_scale;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut gate_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const f32).cast::<c_void>(),
        (&mut packed_arg as *mut *mut u8).cast::<c_void>(),
        (&mut scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
        (&mut eps_arg as *mut f32).cast::<c_void>(),
        (&mut input_scale_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.gated_rms_norm_quantize_nvfp4.launch(
            LaunchConfig::new([rows * heads, 1, 1], [128, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches dense Qwen query/gate splitting and Q/K RMSNorm.
///
/// # Safety
///
/// The buffers must satisfy the supplied row, head, and dimension values.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_full_attn_prep(
    q_full: *const f32,
    k_raw: *const f32,
    q_norm: *const f32,
    k_norm: *const f32,
    q: *mut f32,
    gate: *mut f32,
    k: *mut f32,
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    eps: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut q_full_arg = q_full;
    let mut k_raw_arg = k_raw;
    let mut q_norm_arg = q_norm;
    let mut k_norm_arg = k_norm;
    let mut q_arg = q;
    let mut gate_arg = gate;
    let mut k_arg = k;
    let mut rows_arg = rows;
    let mut q_heads_arg = q_heads;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut eps_arg = eps;
    let mut parameters = [
        (&mut q_full_arg as *mut *const f32).cast::<c_void>(),
        (&mut k_raw_arg as *mut *const f32).cast::<c_void>(),
        (&mut q_norm_arg as *mut *const f32).cast::<c_void>(),
        (&mut k_norm_arg as *mut *const f32).cast::<c_void>(),
        (&mut q_arg as *mut *mut f32).cast::<c_void>(),
        (&mut gate_arg as *mut *mut f32).cast::<c_void>(),
        (&mut k_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut eps_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_full_attn_prep.launch(
            LaunchConfig::new([rows * (q_heads + kv_heads), 1, 1], [head_dim, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches single-token Qwen GDN convolution, splitting, and normalization.
///
/// # Safety
///
/// The buffers must satisfy the supplied head dimensions and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_prep(
    qkv: *const f32,
    conv_weight_bf16: *const u16,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    conv_state: *mut f32,
    key_heads: u32,
    value_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut qkv_arg = qkv;
    let mut conv_weight_arg = conv_weight_bf16;
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut conv_state_arg = conv_state;
    let mut key_heads_arg = key_heads;
    let mut value_heads_arg = value_heads;
    let mut head_dim_arg = head_dim;
    let conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    let mut parameters = [
        (&mut qkv_arg as *mut *const f32).cast::<c_void>(),
        (&mut conv_weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut q_arg as *mut *mut f32).cast::<c_void>(),
        (&mut k_arg as *mut *mut f32).cast::<c_void>(),
        (&mut v_arg as *mut *mut f32).cast::<c_void>(),
        (&mut conv_state_arg as *mut *mut f32).cast::<c_void>(),
        (&mut key_heads_arg as *mut u32).cast::<c_void>(),
        (&mut value_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_prep.launch(
            LaunchConfig::new(grid(conv_dim), block(), 0),
            stream,
            &mut parameters,
        )?;
    }

    for values in [q, k] {
        let mut values_arg = values;
        let mut heads_arg = value_heads;
        let mut parameters = [
            (&mut values_arg as *mut *mut f32).cast::<c_void>(),
            (&mut heads_arg as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            functions()?.l2_norm_heads_128.launch(
                LaunchConfig::new([value_heads, 1, 1], [128, 1, 1], 0),
                stream,
                &mut parameters,
            )?;
        }
    }
    Ok(())
}

unsafe fn l2_norm_heads_128(values: *mut f32, heads: u32, stream: ffi::cudaStream_t) -> Result<()> {
    let mut values_arg = values;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut values_arg as *mut *mut f32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.l2_norm_heads_128.launch(
            LaunchConfig::new([heads, 1, 1], [128, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

unsafe fn l2_norm_heads_128_bf16(
    values: *mut u16,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut values_arg = values;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut values_arg as *mut *mut u16).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.l2_norm_heads_128_bf16.launch(
            LaunchConfig::new([heads, 1, 1], [128, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one-token Qwen GDN preparation for a pointer-table batch.
///
/// # Safety
///
/// All buffers and pointer-table entries must satisfy the supplied dimensions
/// and remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_prep_batch(
    qkv: *const f32,
    conv_weight_bf16: *const u16,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    conv_state_table: *const *mut f32,
    batch_size: u32,
    key_heads: u32,
    value_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    let mut qkv_arg = qkv;
    let mut weight_arg = conv_weight_bf16;
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut state_arg = conv_state_table;
    let mut batch_arg = batch_size;
    let mut key_heads_arg = key_heads;
    let mut value_heads_arg = value_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut qkv_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut q_arg as *mut *mut f32).cast::<c_void>(),
        (&mut k_arg as *mut *mut f32).cast::<c_void>(),
        (&mut v_arg as *mut *mut f32).cast::<c_void>(),
        (&mut state_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut batch_arg as *mut u32).cast::<c_void>(),
        (&mut key_heads_arg as *mut u32).cast::<c_void>(),
        (&mut value_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_prep_batch.launch(
            LaunchConfig::new(grid(batch_size * conv_dim), block(), 0),
            stream,
            &mut parameters,
        )?;
        l2_norm_heads_128(q, batch_size * value_heads, stream)?;
        l2_norm_heads_128(k, batch_size * value_heads, stream)
    }
}

/// Launches token-ordered Qwen GDN preparation for ragged prompt chunks.
///
/// # Safety
///
/// All buffers and pointer-table entries must satisfy the supplied dimensions
/// and remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_prep_chunks(
    qkv: *const f32,
    conv_weight_bf16: *const u16,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    conv_state_table: *const *mut f32,
    sequence_offsets: *const u32,
    sequence_lengths: *const u32,
    sequence_count: u32,
    total_tokens: u32,
    key_heads: u32,
    value_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    let mut qkv_arg = qkv;
    let mut weight_arg = conv_weight_bf16;
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut state_arg = conv_state_table;
    let mut offsets_arg = sequence_offsets;
    let mut lengths_arg = sequence_lengths;
    let mut key_heads_arg = key_heads;
    let mut value_heads_arg = value_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut qkv_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut q_arg as *mut *mut f32).cast::<c_void>(),
        (&mut k_arg as *mut *mut f32).cast::<c_void>(),
        (&mut v_arg as *mut *mut f32).cast::<c_void>(),
        (&mut state_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut offsets_arg as *mut *const u32).cast::<c_void>(),
        (&mut lengths_arg as *mut *const u32).cast::<c_void>(),
        (&mut key_heads_arg as *mut u32).cast::<c_void>(),
        (&mut value_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_prep_chunks.launch(
            LaunchConfig::new([conv_dim.div_ceil(THREADS), sequence_count, 1], block(), 0),
            stream,
            &mut parameters,
        )?;
        l2_norm_heads_128(q, total_tokens * value_heads, stream)?;
        l2_norm_heads_128(k, total_tokens * value_heads, stream)
    }
}

/// Launches token-parallel BF16 Qwen GDN preparation for ragged chunks.
///
/// # Safety
///
/// All buffers and pointer-table entries must satisfy the supplied dimensions
/// and remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_prep_chunks_bf16(
    qkv: *const f32,
    conv_weight_bf16: *const u16,
    q: *mut u16,
    k: *mut u16,
    v: *mut u16,
    conv_state_table: *const *mut f32,
    sequence_offsets: *const u32,
    sequence_lengths: *const u32,
    sequence_count: u32,
    total_tokens: u32,
    key_heads: u32,
    value_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    let mut qkv_arg = qkv;
    let mut weight_arg = conv_weight_bf16;
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut state_arg = conv_state_table;
    let mut offsets_arg = sequence_offsets;
    let mut lengths_arg = sequence_lengths;
    let mut sequence_count_arg = sequence_count;
    let mut total_tokens_arg = total_tokens;
    let mut key_heads_arg = key_heads;
    let mut value_heads_arg = value_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut qkv_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut q_arg as *mut *mut u16).cast::<c_void>(),
        (&mut k_arg as *mut *mut u16).cast::<c_void>(),
        (&mut v_arg as *mut *mut u16).cast::<c_void>(),
        (&mut state_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut offsets_arg as *mut *const u32).cast::<c_void>(),
        (&mut lengths_arg as *mut *const u32).cast::<c_void>(),
        (&mut sequence_count_arg as *mut u32).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut key_heads_arg as *mut u32).cast::<c_void>(),
        (&mut value_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_prep_chunks_bf16.launch(
            LaunchConfig::new(grid(total_tokens * conv_dim), block(), 0),
            stream,
            &mut parameters,
        )?;
    }

    let mut qkv_arg = qkv;
    let mut state_arg = conv_state_table;
    let mut offsets_arg = sequence_offsets;
    let mut lengths_arg = sequence_lengths;
    let mut conv_dim_arg = conv_dim;
    let mut parameters = [
        (&mut qkv_arg as *mut *const f32).cast::<c_void>(),
        (&mut state_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut offsets_arg as *mut *const u32).cast::<c_void>(),
        (&mut lengths_arg as *mut *const u32).cast::<c_void>(),
        (&mut conv_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_update_conv_state.launch(
            LaunchConfig::new([conv_dim.div_ceil(THREADS), sequence_count, 1], block(), 0),
            stream,
            &mut parameters,
        )?;
        l2_norm_heads_128_bf16(q, total_tokens * value_heads, stream)?;
        l2_norm_heads_128_bf16(k, total_tokens * value_heads, stream)
    }
}

/// Launches Qwen GDN log-decay and beta preparation.
///
/// # Safety
///
/// All buffers must contain `heads` values and remain valid until `stream`
/// completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_gate(
    alpha: *const f32,
    beta_input: *const f32,
    a_log_bf16: *const u16,
    dt_bias_bf16: *const u16,
    gate: *mut f32,
    beta: *mut f32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut alpha_arg = alpha;
    let mut beta_input_arg = beta_input;
    let mut a_log_arg = a_log_bf16;
    let mut dt_bias_arg = dt_bias_bf16;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut alpha_arg as *mut *const f32).cast::<c_void>(),
        (&mut beta_input_arg as *mut *const f32).cast::<c_void>(),
        (&mut a_log_arg as *mut *const u16).cast::<c_void>(),
        (&mut dt_bias_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_arg as *mut *mut f32).cast::<c_void>(),
        (&mut beta_arg as *mut *mut f32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_gate.launch(
            LaunchConfig::new(grid(heads), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches batched Qwen GDN gate preparation for f32 outputs.
///
/// # Safety
///
/// The buffers must satisfy the row-major dimensions and remain valid until
/// `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_gate_batch(
    alpha: *const f32,
    beta_input: *const f32,
    a_log_bf16: *const u16,
    dt_bias_bf16: *const u16,
    gate: *mut f32,
    beta: *mut f32,
    rows: u32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut alpha_arg = alpha;
    let mut beta_input_arg = beta_input;
    let mut a_log_arg = a_log_bf16;
    let mut dt_bias_arg = dt_bias_bf16;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut rows_arg = rows;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut alpha_arg as *mut *const f32).cast::<c_void>(),
        (&mut beta_input_arg as *mut *const f32).cast::<c_void>(),
        (&mut a_log_arg as *mut *const u16).cast::<c_void>(),
        (&mut dt_bias_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_arg as *mut *mut f32).cast::<c_void>(),
        (&mut beta_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_gate_batch.launch(
            LaunchConfig::new(grid(rows * heads), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches batched Qwen GDN gate preparation for BF16 outputs.
///
/// # Safety
///
/// The buffers must satisfy the row-major dimensions and remain valid until
/// `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_gate_batch_bf16(
    alpha: *const f32,
    beta_input: *const f32,
    a_log_bf16: *const u16,
    dt_bias_bf16: *const u16,
    gate: *mut u16,
    beta: *mut u16,
    rows: u32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut alpha_arg = alpha;
    let mut beta_input_arg = beta_input;
    let mut a_log_arg = a_log_bf16;
    let mut dt_bias_arg = dt_bias_bf16;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut rows_arg = rows;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut alpha_arg as *mut *const f32).cast::<c_void>(),
        (&mut beta_input_arg as *mut *const f32).cast::<c_void>(),
        (&mut a_log_arg as *mut *const u16).cast::<c_void>(),
        (&mut dt_bias_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_arg as *mut *mut u16).cast::<c_void>(),
        (&mut beta_arg as *mut *mut u16).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_gate_batch_bf16.launch(
            LaunchConfig::new(grid(rows * heads), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches paired-projection Qwen GDN gate preparation for f32 outputs.
///
/// # Safety
///
/// The buffers must satisfy the row-major paired layout and remain valid until
/// `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_gate_paired_batch(
    alpha_beta: *const f32,
    a_log_bf16: *const u16,
    dt_bias_bf16: *const u16,
    gate: *mut f32,
    beta: *mut f32,
    rows: u32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut alpha_beta_arg = alpha_beta;
    let mut a_log_arg = a_log_bf16;
    let mut dt_bias_arg = dt_bias_bf16;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut rows_arg = rows;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut alpha_beta_arg as *mut *const f32).cast::<c_void>(),
        (&mut a_log_arg as *mut *const u16).cast::<c_void>(),
        (&mut dt_bias_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_arg as *mut *mut f32).cast::<c_void>(),
        (&mut beta_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_gate_paired_batch.launch(
            LaunchConfig::new(grid(rows * heads), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches paired-projection Qwen GDN gate preparation for BF16 outputs.
///
/// # Safety
///
/// The buffers must satisfy the row-major paired layout and remain valid until
/// `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qwen_gdn_gate_paired_batch_bf16(
    alpha_beta: *const f32,
    a_log_bf16: *const u16,
    dt_bias_bf16: *const u16,
    gate: *mut u16,
    beta: *mut u16,
    rows: u32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut alpha_beta_arg = alpha_beta;
    let mut a_log_arg = a_log_bf16;
    let mut dt_bias_arg = dt_bias_bf16;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut rows_arg = rows;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut alpha_beta_arg as *mut *const f32).cast::<c_void>(),
        (&mut a_log_arg as *mut *const u16).cast::<c_void>(),
        (&mut dt_bias_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_arg as *mut *mut u16).cast::<c_void>(),
        (&mut beta_arg as *mut *mut u16).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qwen_gdn_gate_paired_batch_bf16.launch(
            LaunchConfig::new(grid(rows * heads), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one-token Qwen gated-delta recurrence.
///
/// # Safety
///
/// The buffers must satisfy the fixed 128-wide layout for `heads` and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gated_delta_net_128(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state: *mut f32,
    output: *mut f32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut state_arg = state;
    let mut output_arg = output;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut q_arg as *mut *const f32).cast::<c_void>(),
        (&mut k_arg as *mut *const f32).cast::<c_void>(),
        (&mut v_arg as *mut *const f32).cast::<c_void>(),
        (&mut gate_arg as *mut *const f32).cast::<c_void>(),
        (&mut beta_arg as *mut *const f32).cast::<c_void>(),
        (&mut state_arg as *mut *mut f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.gated_delta_net_128.launch(
            LaunchConfig::new([heads, 128, 1], [128, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one Gated Delta Net update for every pointer-table batch row.
///
/// # Safety
///
/// The buffers and state pointers must satisfy the fixed 128-wide layout and
/// remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gated_delta_net_128_batch(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_table: *const *mut f32,
    output: *mut f32,
    batch_size: u32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut state_arg = state_table;
    let mut output_arg = output;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut q_arg as *mut *const f32).cast::<c_void>(),
        (&mut k_arg as *mut *const f32).cast::<c_void>(),
        (&mut v_arg as *mut *const f32).cast::<c_void>(),
        (&mut gate_arg as *mut *const f32).cast::<c_void>(),
        (&mut beta_arg as *mut *const f32).cast::<c_void>(),
        (&mut state_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.gated_delta_net_128_batch.launch(
            LaunchConfig::new([batch_size * heads, 128, 1], [128, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches token-ordered Gated Delta Net updates for ragged prompt chunks.
///
/// # Safety
///
/// The buffers, metadata, and state pointers must satisfy the fixed 128-wide
/// layout and remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gated_delta_net_128_chunks(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_table: *const *mut f32,
    sequence_offsets: *const u32,
    sequence_lengths: *const u32,
    output: *mut f32,
    sequence_count: u32,
    total_tokens: u32,
    heads: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut q_arg = q;
    let mut k_arg = k;
    let mut v_arg = v;
    let mut gate_arg = gate;
    let mut beta_arg = beta;
    let mut state_arg = state_table;
    let mut offsets_arg = sequence_offsets;
    let mut lengths_arg = sequence_lengths;
    let mut output_arg = output;
    let mut heads_arg = heads;
    let mut parameters = [
        (&mut q_arg as *mut *const f32).cast::<c_void>(),
        (&mut k_arg as *mut *const f32).cast::<c_void>(),
        (&mut v_arg as *mut *const f32).cast::<c_void>(),
        (&mut gate_arg as *mut *const f32).cast::<c_void>(),
        (&mut beta_arg as *mut *const f32).cast::<c_void>(),
        (&mut state_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut offsets_arg as *mut *const u32).cast::<c_void>(),
        (&mut lengths_arg as *mut *const u32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
    ];
    let (kernel, grid, threads) = if total_tokens / sequence_count >= 1024 {
        (
            &functions()?.gated_delta_net_128_chunks_multiwarp,
            [sequence_count * heads, 16, 1],
            [256, 1, 1],
        )
    } else {
        (
            &functions()?.gated_delta_net_128_chunks,
            [sequence_count * heads, 128, 1],
            [128, 1, 1],
        )
    };
    unsafe { kernel.launch(LaunchConfig::new(grid, threads, 0), stream, &mut parameters) }
}

/// Launches f32 pointer-table row gathering.
///
/// # Safety
///
/// Every table entry must address `row_values` readable values. All buffers
/// must remain valid until `stream` completes.
pub(crate) unsafe fn gather_f32_pointer_rows(
    input_table: *const *mut f32,
    output: *mut f32,
    rows: u32,
    row_values: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_table_arg = input_table;
    let mut output_arg = output;
    let mut row_values_arg = row_values;
    let mut parameters = [
        (&mut input_table_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut row_values_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.gather_f32_pointer_rows.launch(
            LaunchConfig::new([row_values.div_ceil(THREADS), rows, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches f32 pointer-table row scattering.
///
/// # Safety
///
/// Every table entry must address `row_values` writable values. All buffers
/// must remain valid until `stream` completes.
pub(crate) unsafe fn scatter_f32_pointer_rows(
    input: *const f32,
    output_table: *const *mut f32,
    rows: u32,
    row_values: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_table_arg = output_table;
    let mut row_values_arg = row_values;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_table_arg as *mut *const *mut f32).cast::<c_void>(),
        (&mut row_values_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.scatter_f32_pointer_rows.launch(
            LaunchConfig::new([row_values.div_ceil(THREADS), rows, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one DFlash2 target-residual capture.
///
/// # Safety
///
/// The buffers must satisfy the `[row, tap, hidden]` layout and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dflash2_capture(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    hidden: u32,
    taps: u32,
    tap: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut hidden_arg = hidden;
    let mut taps_arg = taps;
    let mut tap_arg = tap;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut taps_arg as *mut u32).cast::<c_void>(),
        (&mut tap_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.dflash2_capture.launch(
            LaunchConfig::new(grid(rows * hidden), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one side of DFlash2 dynamic grouped convolution.
///
/// # Safety
///
/// The buffers must satisfy the supplied grouped-convolution dimensions and
/// remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dflash2_grouped_conv(
    input: *const f32,
    coefficients: *const f32,
    base: *const f32,
    output: *mut f32,
    rows: u32,
    hidden: u32,
    groups: u32,
    taps: u32,
    block_size: u32,
    side: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut coefficients_arg = coefficients;
    let mut base_arg = base;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut hidden_arg = hidden;
    let mut groups_arg = groups;
    let mut taps_arg = taps;
    let mut block_size_arg = block_size;
    let mut side_arg = side;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut coefficients_arg as *mut *const f32).cast::<c_void>(),
        (&mut base_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut groups_arg as *mut u32).cast::<c_void>(),
        (&mut taps_arg as *mut u32).cast::<c_void>(),
        (&mut block_size_arg as *mut u32).cast::<c_void>(),
        (&mut side_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.dflash2_grouped_conv.launch(
            LaunchConfig::new(grid(rows * hidden), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches DFlash2 non-causal proposal attention over a ring cache.
///
/// # Safety
///
/// The buffers must satisfy the supplied attention dimensions and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dflash2_noncausal_attention(
    query: *const f32,
    context_key: *const f32,
    context_value: *const f32,
    block_key: *const f32,
    block_value: *const f32,
    output: *mut f32,
    context_end: u32,
    context_len: u32,
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    window: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut query_arg = query;
    let mut context_key_arg = context_key;
    let mut context_value_arg = context_value;
    let mut block_key_arg = block_key;
    let mut block_value_arg = block_value;
    let mut output_arg = output;
    let mut context_end_arg = context_end;
    let mut context_len_arg = context_len;
    let mut rows_arg = rows;
    let mut q_heads_arg = q_heads;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut window_arg = window;
    let mut parameters = [
        (&mut query_arg as *mut *const f32).cast::<c_void>(),
        (&mut context_key_arg as *mut *const f32).cast::<c_void>(),
        (&mut context_value_arg as *mut *const f32).cast::<c_void>(),
        (&mut block_key_arg as *mut *const f32).cast::<c_void>(),
        (&mut block_value_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut context_end_arg as *mut u32).cast::<c_void>(),
        (&mut context_len_arg as *mut u32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut window_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.dflash2_noncausal_attention.launch(
            LaunchConfig::new(
                [q_heads, rows, 1],
                block(),
                (head_dim + THREADS * 2) * size_of::<f32>() as u32,
            ),
            stream,
            &mut parameters,
        )
    }
}

/// Launches DFlash2 hidden projection through a row-major BF16 matrix.
///
/// # Safety
///
/// The buffers must satisfy the supplied matrix dimensions and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dflash2_hidden_projection(
    hidden: *const f32,
    weight_bf16: *const u16,
    projected: *mut f32,
    rows: u32,
    hidden_size: u32,
    rank: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut hidden_arg = hidden;
    let mut weight_arg = weight_bf16;
    let mut projected_arg = projected;
    let mut hidden_size_arg = hidden_size;
    let mut rank_arg = rank;
    let mut parameters = [
        (&mut hidden_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut projected_arg as *mut *mut f32).cast::<c_void>(),
        (&mut hidden_size_arg as *mut u32).cast::<c_void>(),
        (&mut rank_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.dflash2_hidden_projection.launch(
            LaunchConfig::new([rows * rank, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Reduces vocabulary logits to 32 ordered keys for each active row.
///
/// # Safety
///
/// The input and hierarchical workspaces must satisfy the dimensions derived
/// from `rows` and `vocab`. All storage must remain valid until `stream`
/// completes.
pub(crate) unsafe fn sampling_topk(
    logits: *const f32,
    stage_one_keys: *mut u64,
    stage_two_keys: *mut u64,
    top_keys: *mut u64,
    rows: u32,
    vocab: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    unsafe {
        sampling_hierarchical(
            logits,
            std::ptr::null(),
            stage_one_keys,
            stage_two_keys,
            top_keys,
            rows,
            vocab,
            stream,
        )
    }
}

/// Samples one token from each active vocabulary row.
///
/// # Safety
///
/// Parameters, results, logits, and hierarchical workspaces must satisfy the
/// supplied dimensions. All storage must remain valid until `stream`
/// completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sampling_sample(
    logits: *const f32,
    params: *const c_void,
    stage_one_keys: *mut u64,
    stage_two_keys: *mut u64,
    top_keys: *mut u64,
    results: *mut c_void,
    rows: u32,
    vocab: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    unsafe {
        sampling_hierarchical(
            logits,
            params,
            stage_one_keys,
            stage_two_keys,
            top_keys,
            rows,
            vocab,
            stream,
        )?;
    }
    let mut logits_arg = logits;
    let mut params_arg = params;
    let mut top_keys_arg = top_keys;
    let mut results_arg = results;
    let mut vocab_arg = vocab;
    let mut parameters = [
        (&mut logits_arg as *mut *const f32).cast::<c_void>(),
        (&mut params_arg as *mut *const c_void).cast::<c_void>(),
        (&mut top_keys_arg as *mut *mut u64).cast::<c_void>(),
        (&mut results_arg as *mut *mut c_void).cast::<c_void>(),
        (&mut vocab_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.sampling_finalize.launch(
            LaunchConfig::new([rows, 1, 1], [32, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn sampling_hierarchical(
    logits: *const f32,
    params: *const c_void,
    stage_one_keys: *mut u64,
    stage_two_keys: *mut u64,
    top_keys: *mut u64,
    rows: u32,
    vocab: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    const ITEMS_PER_BLOCK: u32 = 1024;
    const TOP_KEYS: u32 = 32;
    let stage_one_chunks = vocab.div_ceil(ITEMS_PER_BLOCK);
    let stage_one_count = stage_one_chunks * TOP_KEYS;
    let stage_two_chunks = stage_one_count.div_ceil(ITEMS_PER_BLOCK);

    let mut logits_arg = logits;
    let mut params_arg = params;
    let mut stage_one_arg = stage_one_keys;
    let mut vocab_arg = vocab;
    let mut stage_one_chunks_arg = stage_one_chunks;
    let mut logits_parameters = [
        (&mut logits_arg as *mut *const f32).cast::<c_void>(),
        (&mut params_arg as *mut *const c_void).cast::<c_void>(),
        (&mut stage_one_arg as *mut *mut u64).cast::<c_void>(),
        (&mut vocab_arg as *mut u32).cast::<c_void>(),
        (&mut stage_one_chunks_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.sampling_logits_topk.launch(
            LaunchConfig::new([rows * stage_one_chunks, 1, 1], block(), 0),
            stream,
            &mut logits_parameters,
        )?;
    }

    if stage_two_chunks == 1 {
        unsafe { sampling_keys_topk(stage_one_keys, top_keys, rows, stage_one_count, 1, stream) }
    } else {
        unsafe {
            sampling_keys_topk(
                stage_one_keys,
                stage_two_keys,
                rows,
                stage_one_count,
                stage_two_chunks,
                stream,
            )?;
            sampling_keys_topk(
                stage_two_keys,
                top_keys,
                rows,
                stage_two_chunks * TOP_KEYS,
                1,
                stream,
            )
        }
    }
}

/// Reduces existing key rows to 32 ordered keys per output chunk.
///
/// # Safety
///
/// Each input row must contain `input_count_per_row` keys. The output must
/// contain `rows * output_chunks_per_row * 32` keys. Storage must remain valid
/// until `stream` completes.
unsafe fn sampling_keys_topk(
    input_keys: *const u64,
    output_keys: *mut u64,
    rows: u32,
    input_count_per_row: u32,
    output_chunks_per_row: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input_keys;
    let mut output_arg = output_keys;
    let mut input_count_arg = input_count_per_row;
    let mut output_chunks_arg = output_chunks_per_row;
    let mut parameters = [
        (&mut input_arg as *mut *const u64).cast::<c_void>(),
        (&mut output_arg as *mut *mut u64).cast::<c_void>(),
        (&mut input_count_arg as *mut u32).cast::<c_void>(),
        (&mut output_chunks_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.sampling_keys_topk.launch(
            LaunchConfig::new([rows * output_chunks_per_row, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches coherent DFlash2 path selection from device-resident candidates.
///
/// # Safety
///
/// The buffers must satisfy the supplied selector dimensions and remain valid
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dflash2_select_path(
    projected: *const f32,
    top_keys: *const u64,
    predecessor_codebook_bf16: *const u16,
    successor_codebook_bf16: *const u16,
    output_tokens: *mut u32,
    anchor_token: u32,
    drafts: u32,
    rank: u32,
    top_k: u32,
    key_stride: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut projected_arg = projected;
    let mut top_keys_arg = top_keys;
    let mut predecessor_arg = predecessor_codebook_bf16;
    let mut successor_arg = successor_codebook_bf16;
    let mut output_arg = output_tokens;
    let mut anchor_arg = anchor_token;
    let mut drafts_arg = drafts;
    let mut rank_arg = rank;
    let mut top_k_arg = top_k;
    let mut key_stride_arg = key_stride;
    let mut parameters = [
        (&mut projected_arg as *mut *const f32).cast::<c_void>(),
        (&mut top_keys_arg as *mut *const u64).cast::<c_void>(),
        (&mut predecessor_arg as *mut *const u16).cast::<c_void>(),
        (&mut successor_arg as *mut *const u16).cast::<c_void>(),
        (&mut output_arg as *mut *mut u32).cast::<c_void>(),
        (&mut anchor_arg as *mut u32).cast::<c_void>(),
        (&mut drafts_arg as *mut u32).cast::<c_void>(),
        (&mut rank_arg as *mut u32).cast::<c_void>(),
        (&mut top_k_arg as *mut u32).cast::<c_void>(),
        (&mut key_stride_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.dflash2_select_path.launch(
            LaunchConfig::new([1, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches partial NeoX RoPE at one host-supplied position.
///
/// # Safety
///
/// The buffers must satisfy `rows * head_dim` and remain valid until `stream`
/// completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rope_partial(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    head_dim: u32,
    rotary_dim: u32,
    position: u32,
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut position_arg = position;
    let mut theta_arg = theta;
    let len = rows * head_dim;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut position_arg as *mut u32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.rope_partial.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches partial NeoX RoPE at one device-supplied position.
///
/// # Safety
///
/// The buffers must satisfy `rows * head_dim`; `position` must contain one
/// value. All storage must remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rope_partial_indexed(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    head_dim: u32,
    rotary_dim: u32,
    position: *const u32,
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut position_arg = position;
    let mut theta_arg = theta;
    let len = rows * head_dim;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut position_arg as *mut *const u32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.rope_partial_indexed.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches partial NeoX RoPE across a contiguous sequence.
///
/// # Safety
///
/// The buffers must satisfy `tokens * heads * head_dim` and remain valid until
/// `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rope_partial_sequence(
    input: *const f32,
    output: *mut f32,
    tokens: u32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    start_position: u32,
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut tokens_arg = tokens;
    let mut heads_arg = heads;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut start_position_arg = start_position;
    let mut theta_arg = theta;
    let len = tokens * heads * head_dim;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut tokens_arg as *mut u32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut start_position_arg as *mut u32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.rope_partial_sequence.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches interleaved MRoPE at host-supplied positions.
///
/// # Safety
///
/// The buffers must satisfy `rows * head_dim` and remain valid until `stream`
/// completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rope_imrope(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    head_dim: u32,
    rotary_dim: u32,
    sections: [u32; 4],
    positions: [u32; 4],
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut sections_arg = sections;
    let mut positions_arg = positions;
    let mut theta_arg = theta;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut sections_arg[0] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[1] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[2] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[3] as *mut u32).cast::<c_void>(),
        (&mut positions_arg[0] as *mut u32).cast::<c_void>(),
        (&mut positions_arg[1] as *mut u32).cast::<c_void>(),
        (&mut positions_arg[2] as *mut u32).cast::<c_void>(),
        (&mut positions_arg[3] as *mut u32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.rope_imrope.launch(
            LaunchConfig::new(grid(rows * head_dim), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches interleaved MRoPE at device-supplied positions.
///
/// # Safety
///
/// The buffers and position vector must remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rope_imrope_indexed(
    input: *const f32,
    output: *mut f32,
    rows: u32,
    head_dim: u32,
    rotary_dim: u32,
    sections: [u32; 4],
    positions: *const u32,
    position_count: u32,
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut rows_arg = rows;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut sections_arg = sections;
    let mut positions_arg = positions;
    let mut position_count_arg = position_count;
    let mut theta_arg = theta;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut sections_arg[0] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[1] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[2] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[3] as *mut u32).cast::<c_void>(),
        (&mut positions_arg as *mut *const u32).cast::<c_void>(),
        (&mut position_count_arg as *mut u32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.rope_imrope_indexed.launch(
            LaunchConfig::new(grid(rows * head_dim), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches text interleaved MRoPE for a batch of head rows.
///
/// # Safety
///
/// The buffers and positions must remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn rope_imrope_text_batch(
    input: *const f32,
    output: *mut f32,
    positions: *const u32,
    batch_size: u32,
    heads_per_row: u32,
    head_dim: u32,
    rotary_dim: u32,
    sections: [u32; 4],
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut positions_arg = positions;
    let mut batch_size_arg = batch_size;
    let mut heads_per_row_arg = heads_per_row;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut sections_arg = sections;
    let mut theta_arg = theta;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut positions_arg as *mut *const u32).cast::<c_void>(),
        (&mut batch_size_arg as *mut u32).cast::<c_void>(),
        (&mut heads_per_row_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut sections_arg[0] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[1] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[2] as *mut u32).cast::<c_void>(),
        (&mut sections_arg[3] as *mut u32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    let len = batch_size * heads_per_row * head_dim;
    unsafe {
        functions()?.rope_imrope_text_batch.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches independent per-row dynamic E4M3 quantization.
///
/// # Safety
///
/// The buffers must satisfy `rows * cols`; `input_scale` must contain `rows`
/// values. All storage must remain valid until `stream` completes.
pub(crate) unsafe fn quantize_fp8_dynamic(
    input: *const f32,
    quantized: *mut u8,
    input_scale: *mut f32,
    rows: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut quantized_arg = quantized;
    let mut input_scale_arg = input_scale;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut quantized_arg as *mut *mut u8).cast::<c_void>(),
        (&mut input_scale_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.quantize_fp8_dynamic.launch(
            LaunchConfig::new([rows, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches in-place channel and scalar scaling.
///
/// # Safety
///
/// `values` and `channel_scale` must contain `len` values; `scalar` must
/// contain one value.
pub(crate) unsafe fn scale_channel_scalar(
    values: *mut f32,
    channel_scale: *const f32,
    scalar: *const f32,
    len: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut values_arg = values;
    let mut channel_scale_arg = channel_scale;
    let mut scalar_arg = scalar;
    let mut len_arg = len;
    let mut parameters = [
        (&mut values_arg as *mut *mut f32).cast::<c_void>(),
        (&mut channel_scale_arg as *mut *const f32).cast::<c_void>(),
        (&mut scalar_arg as *mut *const f32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.scale_channel_scalar.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches in-place channel and per-row scaling.
///
/// # Safety
///
/// The buffers must satisfy `rows * channels` and remain valid until `stream`
/// completes.
pub(crate) unsafe fn scale_channel_row_scalar(
    values: *mut f32,
    channel_scale: *const f32,
    row_scale: *const f32,
    rows: u32,
    channels: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let len = rows * channels;
    let mut values_arg = values;
    let mut channel_scale_arg = channel_scale;
    let mut row_scale_arg = row_scale;
    let mut channels_arg = channels;
    let mut len_arg = len;
    let mut parameters = [
        (&mut values_arg as *mut *mut f32).cast::<c_void>(),
        (&mut channel_scale_arg as *mut *const f32).cast::<c_void>(),
        (&mut row_scale_arg as *mut *const f32).cast::<c_void>(),
        (&mut channels_arg as *mut u32).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.scale_channel_row_scalar.launch(
            LaunchConfig::new(grid(len), block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one argmax reduction per row.
///
/// # Safety
///
/// The input must contain `rows * cols` values. Both outputs must contain
/// `rows` values and remain valid until `stream` completes.
pub(crate) unsafe fn argmax(
    values: *const f32,
    out_index: *mut u32,
    out_value: *mut f32,
    rows: u32,
    cols: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut values_arg = values;
    let mut out_index_arg = out_index;
    let mut out_value_arg = out_value;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut parameters = [
        (&mut values_arg as *mut *const f32).cast::<c_void>(),
        (&mut out_index_arg as *mut *mut u32).cast::<c_void>(),
        (&mut out_value_arg as *mut *mut f32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.argmax.launch(
            LaunchConfig::new([rows, 1, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches row-major grammar masking.
///
/// # Safety
///
/// The buffers must satisfy `rows`, `cols`, and `mask_words` and remain valid
/// until `stream` completes.
pub(crate) unsafe fn mask_logits_batch(
    logits: *mut f32,
    allowed: *const u32,
    rows: u32,
    cols: u32,
    mask_words: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut logits_arg = logits;
    let mut allowed_arg = allowed;
    let mut rows_arg = rows;
    let mut cols_arg = cols;
    let mut mask_words_arg = mask_words;
    let mut parameters = [
        (&mut logits_arg as *mut *mut f32).cast::<c_void>(),
        (&mut allowed_arg as *mut *const u32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut cols_arg as *mut u32).cast::<c_void>(),
        (&mut mask_words_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.mask_logits_batch.launch(
            LaunchConfig::new([cols.div_ceil(THREADS), rows, 1], block(), 0),
            stream,
            &mut parameters,
        )
    }
}
