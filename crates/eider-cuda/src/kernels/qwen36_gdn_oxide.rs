//! cuda-oxide launches for Qwen3.8 chunked Gated DeltaNet prefill.

use crate::cuda_oxide::{Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::sync::OnceLock;

struct Functions {
    cumsum: Kernel,
    kkt: Kernel,
    solve: Kernel,
    wu: Kernel,
    h: Kernel,
    output: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            cumsum: Kernel::load(c"qwen36_gdn_chunk_cumsum")?,
            kkt: Kernel::load(c"qwen36_gdn_chunk_kkt")?,
            solve: Kernel::load(c"qwen36_gdn_chunk_solve")?,
            wu: Kernel::load(c"qwen36_gdn_chunk_wu")?,
            h: Kernel::load(c"qwen36_gdn_chunk_h")?,
            output: Kernel::load(c"qwen36_gdn_chunk_output")?,
        })
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide Qwen chunked GDN module",
            detail: error.to_string(),
        }),
    }
}

/// Launches the chunk-local gate prefix sum.
///
/// # Safety
///
/// All pointers must address the validated chunked-GDN buffers and remain
/// valid until `stream` completes.
pub(crate) unsafe fn cumsum(
    gate: *const u16,
    gate_cumsum: *mut f32,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    total_tokens: u32,
    chunk_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut gate_arg = gate;
    let mut gate_cumsum_arg = gate_cumsum;
    let mut cu_seqlens_arg = cu_seqlens;
    let mut chunk_indices_arg = chunk_indices;
    let mut total_tokens_arg = total_tokens;
    let mut parameters = [
        (&mut gate_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_cumsum_arg as *mut *mut f32).cast::<c_void>(),
        (&mut cu_seqlens_arg as *mut *const i32).cast::<c_void>(),
        (&mut chunk_indices_arg as *mut *const i32).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.cumsum.launch(
            LaunchConfig::new([chunk_count, 32, 1], [64, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches the chunk-local lower-triangular key transform.
///
/// # Safety
///
/// All pointers must address the validated chunked-GDN buffers and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn kkt(
    key: *const u16,
    beta: *const u16,
    gate_cumsum: *const f32,
    a: *mut f32,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    total_tokens: u32,
    chunk_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_arg = key;
    let mut beta_arg = beta;
    let mut gate_cumsum_arg = gate_cumsum;
    let mut a_arg = a;
    let mut cu_seqlens_arg = cu_seqlens;
    let mut chunk_indices_arg = chunk_indices;
    let mut total_tokens_arg = total_tokens;
    let mut parameters = [
        (&mut key_arg as *mut *const u16).cast::<c_void>(),
        (&mut beta_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_cumsum_arg as *mut *const f32).cast::<c_void>(),
        (&mut a_arg as *mut *mut f32).cast::<c_void>(),
        (&mut cu_seqlens_arg as *mut *const i32).cast::<c_void>(),
        (&mut chunk_indices_arg as *mut *const i32).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.kkt.launch(
            LaunchConfig::new([chunk_count, 32, 1], [512, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches the chunk-local triangular solve.
///
/// # Safety
///
/// All pointers must address the validated chunked-GDN buffers and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn solve(
    a: *mut f32,
    a_inverse: *mut u16,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    total_tokens: u32,
    chunk_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut a_arg = a;
    let mut a_inverse_arg = a_inverse;
    let mut cu_seqlens_arg = cu_seqlens;
    let mut chunk_indices_arg = chunk_indices;
    let mut total_tokens_arg = total_tokens;
    let mut parameters = [
        (&mut a_arg as *mut *mut f32).cast::<c_void>(),
        (&mut a_inverse_arg as *mut *mut u16).cast::<c_void>(),
        (&mut cu_seqlens_arg as *mut *const i32).cast::<c_void>(),
        (&mut chunk_indices_arg as *mut *const i32).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.solve.launch(
            LaunchConfig::new([chunk_count, 32, 1], [256, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches the transformed key and value projections.
///
/// # Safety
///
/// All pointers must address the validated chunked-GDN buffers and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn wu(
    key: *const u16,
    value: *const u16,
    a_inverse: *const u16,
    gate_cumsum: *const f32,
    w: *mut u16,
    u: *mut u16,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    total_tokens: u32,
    chunk_count: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_arg = key;
    let mut value_arg = value;
    let mut a_inverse_arg = a_inverse;
    let mut gate_cumsum_arg = gate_cumsum;
    let mut w_arg = w;
    let mut u_arg = u;
    let mut cu_seqlens_arg = cu_seqlens;
    let mut chunk_indices_arg = chunk_indices;
    let mut total_tokens_arg = total_tokens;
    let mut parameters = [
        (&mut key_arg as *mut *const u16).cast::<c_void>(),
        (&mut value_arg as *mut *const u16).cast::<c_void>(),
        (&mut a_inverse_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_cumsum_arg as *mut *const f32).cast::<c_void>(),
        (&mut w_arg as *mut *mut u16).cast::<c_void>(),
        (&mut u_arg as *mut *mut u16).cast::<c_void>(),
        (&mut cu_seqlens_arg as *mut *const i32).cast::<c_void>(),
        (&mut chunk_indices_arg as *mut *const i32).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.wu.launch(
            LaunchConfig::new([chunk_count, 32, 1], [512, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches sequential chunk-state propagation for every sequence and head.
///
/// # Safety
///
/// All pointers must address the validated chunked-GDN buffers and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn h(
    key: *const u16,
    u: *const u16,
    w: *const u16,
    value_new: *mut u16,
    gate_cumsum: *const f32,
    h: *mut u16,
    state: *mut f32,
    cu_seqlens: *const i32,
    chunk_offsets: *const i64,
    sequence_count: u32,
    total_tokens: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut key_arg = key;
    let mut u_arg = u;
    let mut w_arg = w;
    let mut value_new_arg = value_new;
    let mut gate_cumsum_arg = gate_cumsum;
    let mut h_arg = h;
    let mut state_arg = state;
    let mut cu_seqlens_arg = cu_seqlens;
    let mut chunk_offsets_arg = chunk_offsets;
    let mut total_tokens_arg = total_tokens;
    let mut parameters = [
        (&mut key_arg as *mut *const u16).cast::<c_void>(),
        (&mut u_arg as *mut *const u16).cast::<c_void>(),
        (&mut w_arg as *mut *const u16).cast::<c_void>(),
        (&mut value_new_arg as *mut *mut u16).cast::<c_void>(),
        (&mut gate_cumsum_arg as *mut *const f32).cast::<c_void>(),
        (&mut h_arg as *mut *mut u16).cast::<c_void>(),
        (&mut state_arg as *mut *mut f32).cast::<c_void>(),
        (&mut cu_seqlens_arg as *mut *const i32).cast::<c_void>(),
        (&mut chunk_offsets_arg as *mut *const i64).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.h.launch(
            LaunchConfig::new([4, sequence_count, 32], [512, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}

/// Launches causal chunk output projection.
///
/// # Safety
///
/// All pointers must address the validated chunked-GDN buffers and remain
/// valid until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn output(
    query: *const u16,
    key: *const u16,
    value_new: *const u16,
    h: *const u16,
    gate_cumsum: *const f32,
    output: *mut u16,
    cu_seqlens: *const i32,
    chunk_indices: *const i32,
    total_tokens: u32,
    chunk_count: u32,
    scale: f32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut query_arg = query;
    let mut key_arg = key;
    let mut value_new_arg = value_new;
    let mut h_arg = h;
    let mut gate_cumsum_arg = gate_cumsum;
    let mut output_arg = output;
    let mut cu_seqlens_arg = cu_seqlens;
    let mut chunk_indices_arg = chunk_indices;
    let mut total_tokens_arg = total_tokens;
    let mut scale_arg = scale;
    let mut parameters = [
        (&mut query_arg as *mut *const u16).cast::<c_void>(),
        (&mut key_arg as *mut *const u16).cast::<c_void>(),
        (&mut value_new_arg as *mut *const u16).cast::<c_void>(),
        (&mut h_arg as *mut *const u16).cast::<c_void>(),
        (&mut gate_cumsum_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut u16).cast::<c_void>(),
        (&mut cu_seqlens_arg as *mut *const i32).cast::<c_void>(),
        (&mut chunk_indices_arg as *mut *const i32).cast::<c_void>(),
        (&mut total_tokens_arg as *mut u32).cast::<c_void>(),
        (&mut scale_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.output.launch(
            LaunchConfig::new([2, chunk_count, 32], [512, 1, 1], 0),
            stream,
            &mut parameters,
        )
    }
}
