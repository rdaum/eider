//! cuda-oxide launch support for row-major ModelOpt W4A16 matvecs.

use crate::cuda_oxide::{Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::sync::OnceLock;

struct Functions {
    single: Kernel,
    batch: Kernel,
    reuse_weights_batch: Kernel,
    top1_pass1: Kernel,
    top1_final: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            single: Kernel::load(c"nvfp4_w4a16_matvec_f32_warp_rows")?
                .allow_max_dynamic_shared_memory()?,
            batch: Kernel::load(c"nvfp4_w4a16_matvec_f32_warp_rows_batch")?
                .allow_max_dynamic_shared_memory()?,
            reuse_weights_batch: Kernel::load(c"nvfp4_w4a16_matvec_f32_reuse_weights_batch")?,
            top1_pass1: Kernel::load(c"nvfp4_w4a16_top1_pass1_f32")?
                .allow_max_dynamic_shared_memory()?,
            top1_final: Kernel::load(c"nvfp4_w4a16_top1_final_f32")?,
        })
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide W4A16 matvec module",
            detail: error.to_string(),
        }),
    }
}

/// Launches a row-major ModelOpt W4A16 matrix-vector product.
///
/// # Safety
///
/// Each pointer must satisfy the dimensions and remain valid until `stream`
/// completes. `warps_per_block` must be 4, 8, 16, or 32.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch(
    input: *const f32,
    packed_weight: *const u8,
    weight_scale: *const u8,
    output: *mut f32,
    out_features: u32,
    in_features: u32,
    weight_scale_2: f32,
    warps_per_block: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut packed_weight_arg = packed_weight;
    let mut weight_scale_arg = weight_scale;
    let mut output_arg = output;
    let mut out_features_arg = out_features;
    let mut in_features_arg = in_features;
    let mut weight_scale_2_arg = weight_scale_2;
    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut packed_weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut weight_scale_arg as *mut *const u8).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut out_features_arg as *mut u32).cast::<c_void>(),
        (&mut in_features_arg as *mut u32).cast::<c_void>(),
        (&mut weight_scale_2_arg as *mut f32).cast::<c_void>(),
    ];
    let threads = warps_per_block * 32;
    let grid = out_features.div_ceil(warps_per_block);
    let shared_memory_bytes = in_features.checked_mul(4).ok_or_else(|| Error::Shape {
        label: "cuda-oxide W4A16 matvec shared memory",
        expected: "input bytes that fit in u32".to_string(),
        actual: in_features.to_string(),
    })?;
    unsafe {
        functions()?.single.launch(
            LaunchConfig::new([grid, 1, 1], [threads, 1, 1], shared_memory_bytes),
            stream,
            &mut parameters,
        )
    }
}

/// Launches one row-major W4A16 matvec per activation row.
///
/// # Safety
///
/// Each pointer must satisfy the dimensions and remain valid until `stream`
/// completes. `batch_size` must be non-zero and `warps_per_block` must be valid.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_batch(
    input: *const f32,
    packed_weight: *const u8,
    weight_scale: *const u8,
    output: *mut f32,
    batch_size: u32,
    out_features: u32,
    in_features: u32,
    weight_scale_2: f32,
    warps_per_block: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut packed_weight_arg = packed_weight;
    let mut weight_scale_arg = weight_scale;
    let mut output_arg = output;
    let mut batch_size_arg = batch_size;
    let mut out_features_arg = out_features;
    let mut in_features_arg = in_features;
    let mut weight_scale_2_arg = weight_scale_2;
    let threads = warps_per_block * 32;
    let grid = out_features.div_ceil(warps_per_block);
    if batch_size <= 4 {
        let mut parameters = [
            (&mut input_arg as *mut *const f32).cast::<c_void>(),
            (&mut packed_weight_arg as *mut *const u8).cast::<c_void>(),
            (&mut weight_scale_arg as *mut *const u8).cast::<c_void>(),
            (&mut output_arg as *mut *mut f32).cast::<c_void>(),
            (&mut batch_size_arg as *mut u32).cast::<c_void>(),
            (&mut out_features_arg as *mut u32).cast::<c_void>(),
            (&mut in_features_arg as *mut u32).cast::<c_void>(),
            (&mut weight_scale_2_arg as *mut f32).cast::<c_void>(),
        ];
        return unsafe {
            functions()?.reuse_weights_batch.launch(
                LaunchConfig::new([grid, 1, 1], [threads, 1, 1], 0),
                stream,
                &mut parameters,
            )
        };
    }

    let mut parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut packed_weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut weight_scale_arg as *mut *const u8).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut out_features_arg as *mut u32).cast::<c_void>(),
        (&mut in_features_arg as *mut u32).cast::<c_void>(),
        (&mut weight_scale_2_arg as *mut f32).cast::<c_void>(),
    ];
    let shared_memory_bytes = in_features.checked_mul(4).ok_or_else(|| Error::Shape {
        label: "cuda-oxide batched W4A16 matvec shared memory",
        expected: "input bytes that fit in u32".to_string(),
        actual: in_features.to_string(),
    })?;
    unsafe {
        functions()?.batch.launch(
            LaunchConfig::new([grid, batch_size, 1], [threads, 1, 1], shared_memory_bytes),
            stream,
            &mut parameters,
        )
    }
}

/// Launches a fused row-major W4A16 matvec and top-1 selection.
///
/// # Safety
///
/// Each pointer must satisfy the dimensions and remain valid until `stream`
/// completes. `warps_per_block` must be 4, 8, 16, or 32.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_top1(
    input: *const f32,
    packed_weight: *const u8,
    weight_scale: *const u8,
    scratch_value: *mut f32,
    scratch_index: *mut u32,
    out_index: *mut u32,
    out_value: *mut f32,
    out_features: u32,
    in_features: u32,
    weight_scale_2: f32,
    warps_per_block: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let mut input_arg = input;
    let mut packed_weight_arg = packed_weight;
    let mut weight_scale_arg = weight_scale;
    let mut scratch_value_arg = scratch_value;
    let mut scratch_index_arg = scratch_index;
    let mut out_features_arg = out_features;
    let mut in_features_arg = in_features;
    let mut weight_scale_2_arg = weight_scale_2;
    let threads = warps_per_block * 32;
    let grid = out_features.div_ceil(warps_per_block);
    let shared_memory_bytes = in_features
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(warps_per_block * 8))
        .ok_or_else(|| Error::Shape {
            label: "cuda-oxide W4A16 top-1 shared memory",
            expected: "input and candidate bytes that fit in u32".to_string(),
            actual: format!("in_features={in_features} warps={warps_per_block}"),
        })?;
    let mut pass1_parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut packed_weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut weight_scale_arg as *mut *const u8).cast::<c_void>(),
        (&mut scratch_value_arg as *mut *mut f32).cast::<c_void>(),
        (&mut scratch_index_arg as *mut *mut u32).cast::<c_void>(),
        (&mut out_features_arg as *mut u32).cast::<c_void>(),
        (&mut in_features_arg as *mut u32).cast::<c_void>(),
        (&mut weight_scale_2_arg as *mut f32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.top1_pass1.launch(
            LaunchConfig::new([grid, 1, 1], [threads, 1, 1], shared_memory_bytes),
            stream,
            &mut pass1_parameters,
        )?;
    }

    let mut out_index_arg = out_index;
    let mut out_value_arg = out_value;
    let mut grid_arg = grid;
    let mut final_parameters = [
        (&mut scratch_value_arg as *mut *mut f32).cast::<c_void>(),
        (&mut scratch_index_arg as *mut *mut u32).cast::<c_void>(),
        (&mut out_index_arg as *mut *mut u32).cast::<c_void>(),
        (&mut out_value_arg as *mut *mut f32).cast::<c_void>(),
        (&mut grid_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions()?.top1_final.launch(
            LaunchConfig::new([1, 1, 1], [128, 1, 1], 0),
            stream,
            &mut final_parameters,
        )
    }
}
