//! cuda-oxide launches for compact SM12x FP4 cache updates.

use crate::cuda_oxide::{Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::sync::OnceLock;

const THREADS: u32 = 256;

struct Functions {
    copy_tail: Kernel,
    finalize_key: Kernel,
    finalize_value: Kernel,
    finalize_key_rows: Kernel,
    finalize_value_rows: Kernel,
    stage_tail_bf16: Kernel,
    quantize_query: Kernel,
    qk: Kernel,
    softmax: Kernel,
    quantize_probability: Kernel,
    pv: Kernel,
    pv_reduce: Kernel,
    copy_tail_indexed: Kernel,
    finalize_key_indexed: Kernel,
    finalize_value_indexed: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            copy_tail: Kernel::load(c"sm12x_kv_copy_tail_f32")?,
            finalize_key: Kernel::load(c"sm12x_kv_finalize_key_f32")?,
            finalize_value: Kernel::load(c"sm12x_kv_finalize_value_f32")?,
            finalize_key_rows: Kernel::load(c"sm12x_kv_finalize_key_rows_f32")?,
            finalize_value_rows: Kernel::load(c"sm12x_kv_finalize_value_rows_f32")?,
            stage_tail_bf16: Kernel::load(c"sm12x_kv_stage_tail_bf16")?,
            quantize_query: Kernel::load(c"sm12x_kv_quantize_query_f32")?,
            qk: Kernel::load(c"sm12x_kv_qk_f32")?,
            softmax: Kernel::load(c"sm12x_kv_softmax_f32")?,
            quantize_probability: Kernel::load(c"sm12x_kv_quantize_probability_f32")?,
            pv: Kernel::load(c"sm12x_kv_pv_f32")?,
            pv_reduce: Kernel::load(c"sm12x_kv_pv_reduce_f32")?,
            copy_tail_indexed: Kernel::load(c"sm12x_kv_copy_tail_indexed_f32")?,
            finalize_key_indexed: Kernel::load(c"sm12x_kv_finalize_key_indexed_f32")?,
            finalize_value_indexed: Kernel::load(c"sm12x_kv_finalize_value_indexed_f32")?,
        })
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide compact KV module",
            detail: error.to_string(),
        }),
    }
}

/// Appends one dense K/V row to the compact cache.
///
/// # Safety
///
/// All pointers must refer to the validated cache layout and remain valid until
/// `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn append(
    key: *const f32,
    value: *const f32,
    key_values: *mut u8,
    key_scales: *mut u8,
    value_values: *mut u8,
    value_scales: *mut u8,
    key_tail: *mut f32,
    value_tail: *mut f32,
    position: u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cuda-oxide compact KV width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    unsafe { copy_tail(key, value, key_tail, value_tail, position, width, 1, stream)? };
    if position & 7 == 7 {
        unsafe {
            finalize_key(
                key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim, 1,
                stream,
            )?
        };
    }
    if position & 15 == 15 {
        unsafe {
            finalize_value(
                value_tail,
                value_values,
                value_scales,
                position,
                max_tokens,
                kv_heads,
                head_dim,
                1,
                stream,
            )?
        };
    }
    Ok(())
}

/// Appends one dense K/V row using a device-resident position.
///
/// # Safety
///
/// All pointers must refer to the validated cache layout and remain valid until
/// `stream` completes. `position` must point to one device-resident `u32`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn append_indexed(
    key: *const f32,
    value: *const f32,
    key_values: *mut u8,
    key_scales: *mut u8,
    value_values: *mut u8,
    value_scales: *mut u8,
    key_tail: *mut f32,
    value_tail: *mut f32,
    position: *const u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let functions = functions()?;
    let width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cuda-oxide indexed compact KV width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;

    let mut key_arg = key;
    let mut value_arg = value;
    let mut key_tail_arg = key_tail;
    let mut value_tail_arg = value_tail;
    let mut position_arg = position;
    let mut max_tokens_arg = max_tokens;
    let mut width_arg = width;
    let mut copy_parameters = [
        (&mut key_arg as *mut *const f32).cast::<c_void>(),
        (&mut value_arg as *mut *const f32).cast::<c_void>(),
        (&mut key_tail_arg as *mut *mut f32).cast::<c_void>(),
        (&mut value_tail_arg as *mut *mut f32).cast::<c_void>(),
        (&mut position_arg as *mut *const u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut width_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.copy_tail_indexed.launch(
            LaunchConfig::new([width.div_ceil(THREADS), 1, 1], [THREADS, 1, 1], 0),
            stream,
            &mut copy_parameters,
        )?
    };

    let mut key_values_arg = key_values;
    let mut key_scales_arg = key_scales;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut key_parameters = [
        (&mut key_tail_arg as *mut *mut f32).cast::<c_void>(),
        (&mut key_values_arg as *mut *mut u8).cast::<c_void>(),
        (&mut key_scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut position_arg as *mut *const u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.finalize_key_indexed.launch(
            LaunchConfig::new([kv_heads, head_dim / 16, 1], [1, 1, 1], 0),
            stream,
            &mut key_parameters,
        )?
    };

    let mut value_values_arg = value_values;
    let mut value_scales_arg = value_scales;
    let mut value_parameters = [
        (&mut value_tail_arg as *mut *mut f32).cast::<c_void>(),
        (&mut value_values_arg as *mut *mut u8).cast::<c_void>(),
        (&mut value_scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut position_arg as *mut *const u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.finalize_value_indexed.launch(
            LaunchConfig::new([kv_heads, head_dim / 8, 1], [1, 1, 1], 0),
            stream,
            &mut value_parameters,
        )
    }
}

/// Appends dense prompt rows and optionally stages their BF16 attention views.
///
/// # Safety
///
/// All pointers must satisfy the validated row and compact-cache dimensions and
/// remain valid until `stream` completes. The output pointers must both be null
/// or both refer to their complete BF16 matrices.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn append_rows(
    key: *const f32,
    value: *const f32,
    key_values: *mut u8,
    key_scales: *mut u8,
    value_values: *mut u8,
    value_scales: *mut u8,
    key_tail: *mut f32,
    value_tail: *mut f32,
    key_output: *mut u16,
    value_output: *mut u16,
    output_tokens: u32,
    input_row_offset: u32,
    start_position: u32,
    rows: u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let width = kv_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
        label: "cuda-oxide compact KV row width",
        expected: "kv_heads * head_dim without overflow".to_string(),
        actual: format!("kv_heads={kv_heads} head_dim={head_dim}"),
    })?;
    let mut processed = 0;
    if start_position & 15 != 0 {
        let position = start_position;
        let batch_rows = rows.min(16 - (position & 15));
        let input_offset = input_row_offset
            .checked_mul(width)
            .ok_or_else(|| Error::Shape {
                label: "cuda-oxide compact KV input offset",
                expected: "input row offset without overflow".to_string(),
                actual: format!("input_row_offset={input_row_offset} width={width}"),
            })? as usize;
        unsafe {
            copy_tail(
                key.add(input_offset),
                value.add(input_offset),
                key_tail,
                value_tail,
                position,
                width,
                batch_rows,
                stream,
            )?;
            finalize_key(
                key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim,
                batch_rows, stream,
            )?;
            finalize_value(
                value_tail,
                value_values,
                value_scales,
                position,
                max_tokens,
                kv_heads,
                head_dim,
                batch_rows,
                stream,
            )?;
        }
        processed += batch_rows;
    }

    let bulk_rows = (rows - processed) / 16 * 16;
    if bulk_rows != 0 {
        let bulk_position = start_position + processed;
        let bulk_input_row = input_row_offset + processed;
        unsafe {
            finalize_key_rows(
                key,
                key_values,
                key_scales,
                key_output,
                output_tokens,
                bulk_input_row,
                bulk_position,
                bulk_rows,
                max_tokens,
                kv_heads,
                head_dim,
                stream,
            )?;
            finalize_value_rows(
                value,
                value_values,
                value_scales,
                value_output,
                output_tokens,
                bulk_input_row,
                bulk_position,
                bulk_rows,
                max_tokens,
                kv_heads,
                head_dim,
                stream,
            )?;
        }
        let tail_input_row = bulk_input_row + bulk_rows - 16;
        let tail_position = bulk_position + bulk_rows - 16;
        let tail_input_offset = tail_input_row
            .checked_mul(width)
            .ok_or_else(|| Error::Shape {
                label: "cuda-oxide compact KV tail offset",
                expected: "tail row offset without overflow".to_string(),
                actual: format!("tail_input_row={tail_input_row} width={width}"),
            })? as usize;
        unsafe {
            copy_tail(
                key.add(tail_input_offset),
                value.add(tail_input_offset),
                key_tail,
                value_tail,
                tail_position,
                width,
                16,
                stream,
            )?
        };
        processed += bulk_rows;
    }

    if processed < rows {
        let position = start_position + processed;
        let batch_rows = rows - processed;
        let input_row = input_row_offset + processed;
        let input_offset = input_row.checked_mul(width).ok_or_else(|| Error::Shape {
            label: "cuda-oxide compact KV remainder offset",
            expected: "remainder row offset without overflow".to_string(),
            actual: format!("input_row={input_row} width={width}"),
        })? as usize;
        unsafe {
            copy_tail(
                key.add(input_offset),
                value.add(input_offset),
                key_tail,
                value_tail,
                position,
                width,
                batch_rows,
                stream,
            )?;
            finalize_key(
                key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim,
                batch_rows, stream,
            )?;
            finalize_value(
                value_tail,
                value_values,
                value_scales,
                position,
                max_tokens,
                kv_heads,
                head_dim,
                batch_rows,
                stream,
            )?;
        }
    }

    if !key_output.is_null() && rows & 15 != 0 {
        let tail_start = rows / 16 * 16;
        unsafe {
            stage_tail_bf16(
                key,
                value,
                key_output,
                value_output,
                input_row_offset + tail_start,
                tail_start,
                rows - tail_start,
                output_tokens,
                kv_heads,
                head_dim,
                stream,
            )?
        };
    }
    Ok(())
}

/// Appends prompt rows and computes causal compact attention in bounded batches.
///
/// # Safety
///
/// All pointers must satisfy the validated cache, workspace, and row dimensions
/// and remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn append_causal_attention_rows(
    query: *const f32,
    key: *const f32,
    value: *const f32,
    key_values: *mut u8,
    key_scales: *mut u8,
    value_values: *mut u8,
    value_scales: *mut u8,
    key_tail: *mut f32,
    value_tail: *mut f32,
    query_tiles: *mut u8,
    query_scales: *mut u32,
    scores: *mut f32,
    probability_tiles: *mut u8,
    probability_scales: *mut u32,
    output: *mut f32,
    input_row_offset: u32,
    start_position: u32,
    rows: u32,
    max_tokens: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    window_tokens: u32,
    workspace_rows: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut processed = 0;
    while processed < rows {
        let position = start_position + processed;
        let batch_rows = (rows - processed)
            .min(workspace_rows)
            .min(16 - (position & 15));
        let input_row = input_row_offset + processed;
        unsafe {
            append_rows(
                key,
                value,
                key_values,
                key_scales,
                value_values,
                value_scales,
                key_tail,
                value_tail,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                input_row,
                position,
                batch_rows,
                max_tokens,
                kv_heads,
                head_dim,
                stream,
            )?;
            attention(
                query,
                key_values,
                key_scales,
                key_tail,
                value_values,
                value_scales,
                value_tail,
                query_tiles,
                query_scales,
                scores,
                probability_tiles,
                probability_scales,
                core::ptr::null_mut(),
                output,
                0,
                core::ptr::null(),
                max_tokens,
                q_heads,
                kv_heads,
                head_dim,
                1,
                0,
                core::ptr::null(),
                0,
                0,
                core::ptr::null(),
                core::ptr::null(),
                0,
                input_row,
                batch_rows,
                position,
                window_tokens,
                input_row,
                stream,
            )?;
        }
        processed += batch_rows;
    }
    Ok(())
}

/// Runs contiguous compact-cache FP4 attention.
///
/// A non-null `cache_len_device` selects graph-stable launch bounds and reads
/// the active cache length on the device.
///
/// # Safety
///
/// All pointers must satisfy the validated workspace and cache dimensions and
/// remain valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attention(
    query: *const f32,
    key_values: *const u8,
    key_scales: *const u8,
    key_tail: *const f32,
    value_values: *const u8,
    value_scales: *const u8,
    value_tail: *const f32,
    query_tiles: *mut u8,
    query_scales: *mut u32,
    scores: *mut f32,
    probability_tiles: *mut u8,
    probability_scales: *mut u32,
    partial_output: *mut f32,
    output: *mut f32,
    cache_len: u32,
    cache_len_device: *const u32,
    max_tokens: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    pv_splits: u32,
    window_start: u32,
    page_table: *const u32,
    page_tokens: u32,
    page_stride_bytes: u32,
    selected_blocks: *const u8,
    selected_tiles: *const u8,
    selected_tokens: u32,
    input_row_offset: u32,
    rows: u32,
    causal_start_position: u32,
    window_tokens: u32,
    output_row_offset: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    if rows == 0 || (rows > 1 && pv_splits > 1) {
        return Err(Error::Shape {
            label: "cuda-oxide compact attention rows",
            expected: "non-zero rows and one PV split for batched rows".to_string(),
            actual: format!("rows={rows} pv_splits={pv_splits}"),
        });
    }
    let functions = functions()?;
    let query_tiles_per_kv = (q_heads / kv_heads).div_ceil(8);
    let query_groups = kv_heads
        .checked_mul(query_tiles_per_kv)
        .ok_or_else(|| Error::Shape {
            label: "cuda-oxide compact attention query groups",
            expected: "query group count without overflow".to_string(),
            actual: format!("q_heads={q_heads} kv_heads={kv_heads}"),
        })?;
    let head_k_tiles = head_dim / 64;
    let indexed = !cache_len_device.is_null();
    let launch_cache_len = if causal_start_position == u32::MAX {
        cache_len
    } else {
        causal_start_position
            .checked_add(rows)
            .ok_or_else(|| Error::Shape {
                label: "cuda-oxide causal attention length",
                expected: "causal start + rows without overflow".to_string(),
                actual: format!("start={causal_start_position} rows={rows}"),
            })?
    };
    let token_tiles = if indexed {
        max_tokens.div_ceil(8)
    } else {
        launch_cache_len.div_ceil(8)
    };
    let context_tiles = if indexed {
        max_tokens.div_ceil(64)
    } else {
        launch_cache_len.div_ceil(64)
    };

    let mut query_arg = query;
    let mut query_tiles_arg = query_tiles;
    let mut query_scales_arg = query_scales;
    let mut q_heads_arg = q_heads;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut input_row_offset_arg = input_row_offset;
    let mut query_parameters = [
        (&mut query_arg as *mut *const f32).cast::<c_void>(),
        (&mut query_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut query_scales_arg as *mut *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut input_row_offset_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.quantize_query.launch(
            LaunchConfig::new([query_groups, head_k_tiles, rows], [128, 1, 1], 0),
            stream,
            &mut query_parameters,
        )?
    };

    let mut key_values_arg = key_values;
    let mut key_scales_arg = key_scales;
    let mut key_tail_arg = key_tail;
    let mut scores_arg = scores;
    let mut cache_len_arg = cache_len;
    let mut cache_len_device_arg = cache_len_device;
    let mut window_start_arg = window_start;
    let mut max_tokens_arg = max_tokens;
    let mut page_table_arg = page_table;
    let mut page_tokens_arg = page_tokens;
    let mut page_stride_bytes_arg = page_stride_bytes;
    let mut selected_blocks_arg = selected_blocks;
    let mut causal_start_position_arg = causal_start_position;
    let mut window_tokens_arg = window_tokens;
    let mut qk_parameters = [
        (&mut query_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut query_scales_arg as *mut *mut u32).cast::<c_void>(),
        (&mut key_values_arg as *mut *const u8).cast::<c_void>(),
        (&mut key_scales_arg as *mut *const u8).cast::<c_void>(),
        (&mut key_tail_arg as *mut *const f32).cast::<c_void>(),
        (&mut scores_arg as *mut *mut f32).cast::<c_void>(),
        (&mut cache_len_arg as *mut u32).cast::<c_void>(),
        (&mut cache_len_device_arg as *mut *const u32).cast::<c_void>(),
        (&mut window_start_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut page_table_arg as *mut *const u32).cast::<c_void>(),
        (&mut page_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut page_stride_bytes_arg as *mut u32).cast::<c_void>(),
        (&mut selected_blocks_arg as *mut *const u8).cast::<c_void>(),
        (&mut causal_start_position_arg as *mut u32).cast::<c_void>(),
        (&mut window_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.qk.launch(
            LaunchConfig::new([query_groups, token_tiles, rows], [32, 1, 1], 0),
            stream,
            &mut qk_parameters,
        )?
    };

    let mut softmax_parameters = [
        (&mut scores_arg as *mut *mut f32).cast::<c_void>(),
        (&mut cache_len_arg as *mut u32).cast::<c_void>(),
        (&mut cache_len_device_arg as *mut *const u32).cast::<c_void>(),
        (&mut window_start_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut selected_blocks_arg as *mut *const u8).cast::<c_void>(),
        (&mut causal_start_position_arg as *mut u32).cast::<c_void>(),
        (&mut window_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.softmax.launch(
            LaunchConfig::new([q_heads, rows, 1], [256, 1, 1], 0),
            stream,
            &mut softmax_parameters,
        )?
    };

    let mut probability_tiles_arg = probability_tiles;
    let mut probability_scales_arg = probability_scales;
    let mut selected_tokens_arg = selected_tokens;
    let mut probability_parameters = [
        (&mut scores_arg as *mut *mut f32).cast::<c_void>(),
        (&mut probability_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut probability_scales_arg as *mut *mut u32).cast::<c_void>(),
        (&mut cache_len_arg as *mut u32).cast::<c_void>(),
        (&mut cache_len_device_arg as *mut *const u32).cast::<c_void>(),
        (&mut window_start_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut selected_blocks_arg as *mut *const u8).cast::<c_void>(),
        (&mut selected_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut causal_start_position_arg as *mut u32).cast::<c_void>(),
        (&mut window_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.quantize_probability.launch(
            LaunchConfig::new([query_groups, context_tiles, rows], [128, 1, 1], 0),
            stream,
            &mut probability_parameters,
        )?
    };

    let mut value_values_arg = value_values;
    let mut value_scales_arg = value_scales;
    let mut value_tail_arg = value_tail;
    let mut output_arg = output;
    let mut partial_output_arg = partial_output;
    let mut pv_splits_arg = pv_splits;
    let mut selected_tiles_arg = selected_tiles;
    let mut output_row_offset_arg = output_row_offset;
    let mut pv_parameters = [
        (&mut probability_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut probability_scales_arg as *mut *mut u32).cast::<c_void>(),
        (&mut value_values_arg as *mut *const u8).cast::<c_void>(),
        (&mut value_scales_arg as *mut *const u8).cast::<c_void>(),
        (&mut value_tail_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut partial_output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut cache_len_arg as *mut u32).cast::<c_void>(),
        (&mut cache_len_device_arg as *mut *const u32).cast::<c_void>(),
        (&mut window_start_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut q_heads_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut pv_splits_arg as *mut u32).cast::<c_void>(),
        (&mut page_table_arg as *mut *const u32).cast::<c_void>(),
        (&mut page_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut page_stride_bytes_arg as *mut u32).cast::<c_void>(),
        (&mut selected_tiles_arg as *mut *const u8).cast::<c_void>(),
        (&mut selected_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut causal_start_position_arg as *mut u32).cast::<c_void>(),
        (&mut window_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut output_row_offset_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.pv.launch(
            LaunchConfig::new(
                [query_groups, head_dim / 8, rows * pv_splits],
                [32, 1, 1],
                0,
            ),
            stream,
            &mut pv_parameters,
        )?
    };

    if pv_splits > 1 {
        let width = q_heads.checked_mul(head_dim).ok_or_else(|| Error::Shape {
            label: "cuda-oxide compact attention output width",
            expected: "q_heads * head_dim without overflow".to_string(),
            actual: format!("q_heads={q_heads} head_dim={head_dim}"),
        })?;
        let mut width_arg = width;
        let mut reduce_parameters = [
            (&mut partial_output_arg as *mut *mut f32).cast::<c_void>(),
            (&mut output_arg as *mut *mut f32).cast::<c_void>(),
            (&mut pv_splits_arg as *mut u32).cast::<c_void>(),
            (&mut width_arg as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            functions.pv_reduce.launch(
                LaunchConfig::new([width.div_ceil(THREADS), 1, 1], [THREADS, 1, 1], 0),
                stream,
                &mut reduce_parameters,
            )?
        };
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn copy_tail(
    key: *const f32,
    value: *const f32,
    key_tail: *mut f32,
    value_tail: *mut f32,
    position: u32,
    width: u32,
    rows: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_arg = key;
    let mut value_arg = value;
    let mut key_tail_arg = key_tail;
    let mut value_tail_arg = value_tail;
    let mut position_arg = position;
    let mut width_arg = width;
    let mut parameters = [
        (&mut key_arg as *mut *const f32).cast::<c_void>(),
        (&mut value_arg as *mut *const f32).cast::<c_void>(),
        (&mut key_tail_arg as *mut *mut f32).cast::<c_void>(),
        (&mut value_tail_arg as *mut *mut f32).cast::<c_void>(),
        (&mut position_arg as *mut u32).cast::<c_void>(),
        (&mut width_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.copy_tail.launch(
            LaunchConfig::new([width.div_ceil(THREADS), rows, 1], [THREADS, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn finalize_key(
    key_tail: *const f32,
    key_values: *mut u8,
    key_scales: *mut u8,
    position: u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    rows: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_tail_arg = key_tail;
    let mut key_values_arg = key_values;
    let mut key_scales_arg = key_scales;
    let mut position_arg = position;
    let mut max_tokens_arg = max_tokens;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut key_tail_arg as *mut *const f32).cast::<c_void>(),
        (&mut key_values_arg as *mut *mut u8).cast::<c_void>(),
        (&mut key_scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut position_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.finalize_key.launch(
            LaunchConfig::new([kv_heads, head_dim / 16, rows], [1, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn finalize_value(
    value_tail: *const f32,
    value_values: *mut u8,
    value_scales: *mut u8,
    position: u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    rows: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut value_tail_arg = value_tail;
    let mut value_values_arg = value_values;
    let mut value_scales_arg = value_scales;
    let mut position_arg = position;
    let mut max_tokens_arg = max_tokens;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut value_tail_arg as *mut *const f32).cast::<c_void>(),
        (&mut value_values_arg as *mut *mut u8).cast::<c_void>(),
        (&mut value_scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut position_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.finalize_value.launch(
            LaunchConfig::new([kv_heads, head_dim / 8, rows], [1, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn finalize_key_rows(
    key: *const f32,
    key_values: *mut u8,
    key_scales: *mut u8,
    key_output: *mut u16,
    output_tokens: u32,
    input_row_offset: u32,
    start_position: u32,
    rows: u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_arg = key;
    let mut key_values_arg = key_values;
    let mut key_scales_arg = key_scales;
    let mut key_output_arg = key_output;
    let mut output_tokens_arg = output_tokens;
    let mut input_row_offset_arg = input_row_offset;
    let mut start_position_arg = start_position;
    let mut max_tokens_arg = max_tokens;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut key_arg as *mut *const f32).cast::<c_void>(),
        (&mut key_values_arg as *mut *mut u8).cast::<c_void>(),
        (&mut key_scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut key_output_arg as *mut *mut u16).cast::<c_void>(),
        (&mut output_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut input_row_offset_arg as *mut u32).cast::<c_void>(),
        (&mut start_position_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.finalize_key_rows.launch(
            LaunchConfig::new([kv_heads, head_dim / 16, rows / 8], [16, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn finalize_value_rows(
    value: *const f32,
    value_values: *mut u8,
    value_scales: *mut u8,
    value_output: *mut u16,
    output_tokens: u32,
    input_row_offset: u32,
    start_position: u32,
    rows: u32,
    max_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut value_arg = value;
    let mut value_values_arg = value_values;
    let mut value_scales_arg = value_scales;
    let mut value_output_arg = value_output;
    let mut output_tokens_arg = output_tokens;
    let mut input_row_offset_arg = input_row_offset;
    let mut start_position_arg = start_position;
    let mut max_tokens_arg = max_tokens;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut value_arg as *mut *const f32).cast::<c_void>(),
        (&mut value_values_arg as *mut *mut u8).cast::<c_void>(),
        (&mut value_scales_arg as *mut *mut u8).cast::<c_void>(),
        (&mut value_output_arg as *mut *mut u16).cast::<c_void>(),
        (&mut output_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut input_row_offset_arg as *mut u32).cast::<c_void>(),
        (&mut start_position_arg as *mut u32).cast::<c_void>(),
        (&mut max_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.finalize_value_rows.launch(
            LaunchConfig::new([kv_heads, head_dim / 8, rows / 16], [16, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn stage_tail_bf16(
    key: *const f32,
    value: *const f32,
    key_output: *mut u16,
    value_output: *mut u16,
    input_row_offset: u32,
    output_row_offset: u32,
    rows: u32,
    output_tokens: u32,
    kv_heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let values = rows
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::Shape {
            label: "cuda-oxide compact KV BF16 tail",
            expected: "tail element count without overflow".to_string(),
            actual: format!("rows={rows} kv_heads={kv_heads} head_dim={head_dim}"),
        })?;
    let mut key_arg = key;
    let mut value_arg = value;
    let mut key_output_arg = key_output;
    let mut value_output_arg = value_output;
    let mut input_row_offset_arg = input_row_offset;
    let mut output_row_offset_arg = output_row_offset;
    let mut rows_arg = rows;
    let mut output_tokens_arg = output_tokens;
    let mut kv_heads_arg = kv_heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut key_arg as *mut *const f32).cast::<c_void>(),
        (&mut value_arg as *mut *const f32).cast::<c_void>(),
        (&mut key_output_arg as *mut *mut u16).cast::<c_void>(),
        (&mut value_output_arg as *mut *mut u16).cast::<c_void>(),
        (&mut input_row_offset_arg as *mut u32).cast::<c_void>(),
        (&mut output_row_offset_arg as *mut u32).cast::<c_void>(),
        (&mut rows_arg as *mut u32).cast::<c_void>(),
        (&mut output_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut kv_heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.stage_tail_bf16.launch(
            LaunchConfig::new([values.div_ceil(THREADS), 1, 1], [THREADS, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}
