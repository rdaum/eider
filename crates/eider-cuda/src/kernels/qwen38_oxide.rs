//! cuda-oxide launches for Qwen3.8 Flash Next hyperconnection, PLE, and QSA.

use crate::cuda_oxide::{Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::sync::OnceLock;

const THREADS: u32 = 256;

struct Functions {
    hc_norm: Kernel,
    hc_silu_scale: Kernel,
    hc_collapse: Kernel,
    hc_combine: Kernel,
    repeat_streams: Kernel,
    ple_gate_value: Kernel,
    ple_conv_update: Kernel,
    qsa_clear_masks: Kernel,
    qsa_prepare_query: Kernel,
    qsa_append_key: Kernel,
    qsa_score_blocks: Kernel,
    qsa_select_blocks: Kernel,
    qsa_build_tile_mask: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            hc_norm: Kernel::load(c"qwen38_hc_norm_f32")?,
            hc_silu_scale: Kernel::load(c"qwen38_hc_silu_scale_f32")?,
            hc_collapse: Kernel::load(c"qwen38_hc_collapse_f32")?,
            hc_combine: Kernel::load(c"qwen38_hc_combine_f32")?,
            repeat_streams: Kernel::load(c"qwen38_repeat_streams_f32")?,
            ple_gate_value: Kernel::load(c"qwen38_ple_gate_value_f32")?,
            ple_conv_update: Kernel::load(c"qwen38_ple_conv_update_f32")?,
            qsa_clear_masks: Kernel::load(c"qwen38_qsa_clear_masks")?,
            qsa_prepare_query: Kernel::load(c"qwen38_qsa_prepare_query_f32")?,
            qsa_append_key: Kernel::load(c"qwen38_qsa_append_key_f32")?,
            qsa_score_blocks: Kernel::load(c"qwen38_qsa_score_blocks_f32")?,
            qsa_select_blocks: Kernel::load(c"qwen38_qsa_select_blocks_f32")?,
            qsa_build_tile_mask: Kernel::load(c"qwen38_qsa_build_tile_mask")?,
        })
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide Qwen3.8 Flash Next module",
            detail: error.to_string(),
        }),
    }
}

fn grid(count: u64, threads: u32) -> [u32; 3] {
    [count.div_ceil(u64::from(threads)) as u32, 1, 1]
}

fn block(threads: u32) -> [u32; 3] {
    [threads, 1, 1]
}

/// Launches Qwen3.8 Flash Next per-stream RMS normalization.
///
/// # Safety
///
/// The caller must provide buffers for `tokens * hidden * hc_count` values.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn hc_norm(
    input: *const f32,
    delta_weight: *const f32,
    output: *mut f32,
    tokens: u32,
    hidden: u32,
    hc_count: u32,
    eps: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut delta_weight_arg = delta_weight;
    let mut output_arg = output;
    let mut hidden_arg = hidden;
    let mut hc_count_arg = hc_count;
    let mut eps_arg = eps;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut delta_weight_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut hc_count_arg as *mut u32).cast::<c_void>(),
        (&mut eps_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.hc_norm.launch(
            LaunchConfig::new([tokens * hc_count, 1, 1], block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches scaled SiLU for a Qwen3.8 Flash Next low-rank projection.
///
/// # Safety
///
/// `values` must contain `count` writable values.
pub(crate) unsafe fn hc_silu_scale(
    values: *mut f32,
    count: usize,
    scale: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut values_arg = values;
    let mut count_arg = count as u64;
    let mut scale_arg = scale;
    let mut parameters = [
        (&mut values_arg as *mut *mut f32).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
        (&mut scale_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.hc_silu_scale.launch(
            LaunchConfig::new(grid(count_arg, THREADS), block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches Qwen3.8 Flash Next hyperconnection stream collapse.
///
/// # Safety
///
/// All buffers must match the validated Qwen3.8 Flash Next stream geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn hc_collapse(
    normed: *const f32,
    gate_logits: *const f32,
    output: *mut f32,
    tokens: u32,
    hidden: u32,
    hc_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut normed_arg = normed;
    let mut gate_logits_arg = gate_logits;
    let mut output_arg = output;
    let mut tokens_arg = tokens;
    let mut hidden_arg = hidden;
    let mut hc_count_arg = hc_count;
    let mut parameters = [
        (&mut normed_arg as *mut *const f32).cast::<c_void>(),
        (&mut gate_logits_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut tokens_arg as *mut u32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut hc_count_arg as *mut u32).cast::<c_void>(),
    ];
    let count = u64::from(tokens) * u64::from(hidden);
    unsafe {
        functions()?.hc_collapse.launch(
            LaunchConfig::new(grid(count, THREADS), block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches Qwen3.8 Flash Next hyperconnection block injection.
///
/// # Safety
///
/// All buffers must match the validated Qwen3.8 Flash Next stream geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn hc_combine(
    residual: *const f32,
    block_output: *const f32,
    inject_logits: *const f32,
    output: *mut f32,
    tokens: u32,
    hidden: u32,
    hc_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut residual_arg = residual;
    let mut block_output_arg = block_output;
    let mut inject_logits_arg = inject_logits;
    let mut output_arg = output;
    let mut tokens_arg = tokens;
    let mut hidden_arg = hidden;
    let mut hc_count_arg = hc_count;
    let mut parameters = [
        (&mut residual_arg as *mut *const f32).cast::<c_void>(),
        (&mut block_output_arg as *mut *const f32).cast::<c_void>(),
        (&mut inject_logits_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut tokens_arg as *mut u32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut hc_count_arg as *mut u32).cast::<c_void>(),
    ];
    let count = u64::from(tokens) * u64::from(hidden) * u64::from(hc_count);
    unsafe {
        functions()?.hc_combine.launch(
            LaunchConfig::new(grid(count, THREADS), block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches Qwen3.8 Flash Next initial stream replication.
///
/// # Safety
///
/// The buffers must match the validated Qwen3.8 Flash Next stream geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn repeat_streams(
    input: *const f32,
    output: *mut f32,
    tokens: u32,
    hidden: u32,
    hc_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut output_arg = output;
    let mut tokens_arg = tokens;
    let mut hidden_arg = hidden;
    let mut hc_count_arg = hc_count;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut tokens_arg as *mut u32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut hc_count_arg as *mut u32).cast::<c_void>(),
    ];
    let count = u64::from(tokens) * u64::from(hidden) * u64::from(hc_count);
    unsafe {
        functions()?.repeat_streams.launch(
            LaunchConfig::new(grid(count, THREADS), block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches Qwen3.8 Flash Next PLE gate and value fusion.
///
/// # Safety
///
/// The buffers must match the validated Qwen3.8 Flash Next PLE geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn ple_gate_value(
    key: *const f32,
    query: *const f32,
    value: *const f32,
    gated: *mut f32,
    tokens: u32,
    hidden: u32,
    hc_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_arg = key;
    let mut query_arg = query;
    let mut value_arg = value;
    let mut gated_arg = gated;
    let mut hidden_arg = hidden;
    let mut hc_count_arg = hc_count;
    let mut parameters = [
        (&mut key_arg as *mut *const f32).cast::<c_void>(),
        (&mut query_arg as *mut *const f32).cast::<c_void>(),
        (&mut value_arg as *mut *const f32).cast::<c_void>(),
        (&mut gated_arg as *mut *mut f32).cast::<c_void>(),
        (&mut hidden_arg as *mut u32).cast::<c_void>(),
        (&mut hc_count_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.ple_gate_value.launch(
            LaunchConfig::new([tokens * hc_count, 1, 1], block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches the Qwen3.8 Flash Next PLE convolution and state update.
///
/// # Safety
///
/// The buffers must match the validated convolution geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn ple_conv_update(
    normalized: *const f32,
    gated: *const f32,
    weight_bf16: *const u16,
    state: *mut f32,
    output: *mut f32,
    tokens: u32,
    channels: u32,
    kernel: u32,
    dilation: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut normalized_arg = normalized;
    let mut gated_arg = gated;
    let mut weight_arg = weight_bf16;
    let mut state_arg = state;
    let mut output_arg = output;
    let mut tokens_arg = tokens;
    let mut channels_arg = channels;
    let mut kernel_arg = kernel;
    let mut dilation_arg = dilation;
    let mut history_arg = (kernel - 1) * dilation;
    let mut parameters = [
        (&mut normalized_arg as *mut *const f32).cast::<c_void>(),
        (&mut gated_arg as *mut *const f32).cast::<c_void>(),
        (&mut weight_arg as *mut *const u16).cast::<c_void>(),
        (&mut state_arg as *mut *mut f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut tokens_arg as *mut u32).cast::<c_void>(),
        (&mut channels_arg as *mut u32).cast::<c_void>(),
        (&mut kernel_arg as *mut u32).cast::<c_void>(),
        (&mut dilation_arg as *mut u32).cast::<c_void>(),
        (&mut history_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.ple_conv_update.launch(
            LaunchConfig::new(grid(u64::from(channels), THREADS), block(THREADS), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Appends one raw QSA key.
///
/// # Safety
///
/// The buffers must match the validated QSA page geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qsa_append_key(
    projection: *const f32,
    key_pool_bf16: *mut u16,
    slot: u32,
    page_offset: u32,
    page_tokens: u32,
    heads: u32,
    head_dim: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut projection_arg = projection;
    let mut key_pool_arg = key_pool_bf16;
    let mut slot_arg = slot;
    let mut page_offset_arg = page_offset;
    let mut page_tokens_arg = page_tokens;
    let mut heads_arg = heads;
    let mut head_dim_arg = head_dim;
    let mut parameters = [
        (&mut projection_arg as *mut *const f32).cast::<c_void>(),
        (&mut key_pool_arg as *mut *mut u16).cast::<c_void>(),
        (&mut slot_arg as *mut u32).cast::<c_void>(),
        (&mut page_offset_arg as *mut u32).cast::<c_void>(),
        (&mut page_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut heads_arg as *mut u32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qsa_append_key.launch(
            LaunchConfig::new(grid(u64::from(head_dim), 128), block(128), 0),
            stream,
            &mut parameters,
        )
    }
}

/// Prepares, scores, and selects Qwen3.8 Flash Next QSA blocks.
///
/// # Safety
///
/// The buffers must match the validated QSA workspace and page geometry.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn qsa_prepare_and_select(
    projection: *const f32,
    q_norm: *const f32,
    k_norm: *const f32,
    key_pool_bf16: *mut u16,
    page_table: *const u32,
    query: *mut f32,
    scores: *mut f32,
    selected_blocks: *mut u8,
    selected_tiles: *mut u8,
    slot: u32,
    page_offset: u32,
    cache_len: u32,
    max_tokens: u32,
    page_tokens: u32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    compress_ratio: u32,
    budget: u32,
    eps: f32,
    theta: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let complete_blocks = cache_len / compress_ratio;
    let tail_tokens = cache_len % compress_ratio;
    let visible_blocks = complete_blocks + u32::from(tail_tokens != 0);
    let max_blocks = max_tokens.div_ceil(compress_ratio);
    let max_tiles = max_tokens.div_ceil(64);
    let visible_tiles = cache_len.div_ceil(64);

    let mut selected_blocks_arg = selected_blocks;
    let mut selected_tiles_arg = selected_tiles;
    let mut max_blocks_arg = max_blocks;
    let mut max_tiles_arg = max_tiles;
    let mut clear_parameters = [
        (&mut selected_blocks_arg as *mut *mut u8).cast::<c_void>(),
        (&mut selected_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut max_blocks_arg as *mut u32).cast::<c_void>(),
        (&mut max_tiles_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qsa_clear_masks.launch(
            LaunchConfig::new(
                grid(u64::from(max_blocks.max(max_tiles)), THREADS),
                block(THREADS),
                0,
            ),
            stream,
            &mut clear_parameters,
        )?;
    }

    let mut projection_arg = projection;
    let mut q_norm_arg = q_norm;
    let mut query_arg = query;
    let mut head_dim_arg = head_dim;
    let mut rotary_dim_arg = rotary_dim;
    let mut position_arg = cache_len - 1;
    let mut eps_arg = eps;
    let mut theta_arg = theta;
    let mut query_parameters = [
        (&mut projection_arg as *mut *const f32).cast::<c_void>(),
        (&mut q_norm_arg as *mut *const f32).cast::<c_void>(),
        (&mut query_arg as *mut *mut f32).cast::<c_void>(),
        (&mut head_dim_arg as *mut u32).cast::<c_void>(),
        (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
        (&mut position_arg as *mut u32).cast::<c_void>(),
        (&mut eps_arg as *mut f32).cast::<c_void>(),
        (&mut theta_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qsa_prepare_query.launch(
            LaunchConfig::new([heads, 1, 1], block(head_dim), 0),
            stream,
            &mut query_parameters,
        )?;
        qsa_append_key(
            projection,
            key_pool_bf16,
            slot,
            page_offset,
            page_tokens,
            heads,
            head_dim,
            stream,
        )?;
    }

    if complete_blocks != 0 {
        let mut query_arg = query;
        let mut key_pool_arg = key_pool_bf16.cast_const();
        let mut page_table_arg = page_table;
        let mut k_norm_arg = k_norm;
        let mut scores_arg = scores;
        let mut page_tokens_arg = page_tokens;
        let mut heads_arg = heads;
        let mut head_dim_arg = head_dim;
        let mut rotary_dim_arg = rotary_dim;
        let mut eps_arg = eps;
        let mut theta_arg = theta;
        let mut score_parameters = [
            (&mut query_arg as *mut *mut f32).cast::<c_void>(),
            (&mut key_pool_arg as *mut *const u16).cast::<c_void>(),
            (&mut page_table_arg as *mut *const u32).cast::<c_void>(),
            (&mut k_norm_arg as *mut *const f32).cast::<c_void>(),
            (&mut scores_arg as *mut *mut f32).cast::<c_void>(),
            (&mut page_tokens_arg as *mut u32).cast::<c_void>(),
            (&mut heads_arg as *mut u32).cast::<c_void>(),
            (&mut head_dim_arg as *mut u32).cast::<c_void>(),
            (&mut rotary_dim_arg as *mut u32).cast::<c_void>(),
            (&mut eps_arg as *mut f32).cast::<c_void>(),
            (&mut theta_arg as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            functions()?.qsa_score_blocks.launch(
                LaunchConfig::new([complete_blocks, 1, 1], block(head_dim), 0),
                stream,
                &mut score_parameters,
            )?;
        }
    }

    let mut scores_arg = scores.cast_const();
    let mut selected_blocks_arg = selected_blocks;
    let mut complete_blocks_arg = complete_blocks;
    let mut selected_complete_blocks_arg = complete_blocks.min(budget / compress_ratio);
    let mut tail_tokens_arg = tail_tokens;
    let mut select_parameters = [
        (&mut scores_arg as *mut *const f32).cast::<c_void>(),
        (&mut selected_blocks_arg as *mut *mut u8).cast::<c_void>(),
        (&mut complete_blocks_arg as *mut u32).cast::<c_void>(),
        (&mut selected_complete_blocks_arg as *mut u32).cast::<c_void>(),
        (&mut tail_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qsa_select_blocks.launch(
            LaunchConfig::new([1, 1, 1], block(THREADS), 0),
            stream,
            &mut select_parameters,
        )?;
    }

    let mut selected_blocks_arg = selected_blocks.cast_const();
    let mut selected_tiles_arg = selected_tiles;
    let mut visible_blocks_arg = visible_blocks;
    let mut tile_parameters = [
        (&mut selected_blocks_arg as *mut *const u8).cast::<c_void>(),
        (&mut selected_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut visible_blocks_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.qsa_build_tile_mask.launch(
            LaunchConfig::new(grid(u64::from(visible_tiles), THREADS), block(THREADS), 0),
            stream,
            &mut tile_parameters,
        )
    }
}
