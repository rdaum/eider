//! cuda-oxide launch support for grouped SM121 W4A4 operations.

use crate::cuda_oxide::{self, Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::sync::OnceLock;

const WORKERS: u32 = 4;

struct Functions {
    build_routes: Kernel,
    quantize: Kernel,
    gemm: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            build_routes: Kernel::load(c"w4a4_build_route_groups")?,
            quantize: Kernel::load(c"w4a4_quantize_route_groups_f32")?,
            gemm: Kernel::load(c"w4a4_route_groups_f32")?,
        })
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide grouped W4A4 module",
            detail: error.to_string(),
        }),
    }
}

pub(crate) fn ensure_supported() -> Result<()> {
    cuda_oxide::ensure_supported()?;
    functions().map(|_| ())
}

/// Sorts routes, quantizes grouped inputs, and runs grouped W4A4 GEMM.
///
/// # Safety
///
/// Each pointer must refer to a device buffer that satisfies the dimensions.
/// The buffers must remain valid until the supplied stream completes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch(
    indices: *const u32,
    input: *const f32,
    tiled_weight: *const u8,
    tiled_scales: *const u8,
    global_scales: *const f32,
    sorted_routes: *mut u32,
    group_experts: *mut u32,
    group_starts: *mut u32,
    group_lengths: *mut u32,
    group_count: *mut u32,
    input_tiles: *mut u8,
    input_scales: *mut u32,
    output: *mut f32,
    rows: u32,
    experts: u32,
    top_k: u32,
    out_features: u32,
    in_features: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let routes = rows.checked_mul(top_k).ok_or_else(|| Error::Shape {
        label: "cuda-oxide grouped W4A4 routes",
        expected: "rows * top-k without overflow".to_string(),
        actual: format!("rows={rows} top_k={top_k}"),
    })?;
    let functions = functions()?;

    let mut indices_arg = indices;
    let mut sorted_routes_arg = sorted_routes;
    let mut group_experts_arg = group_experts;
    let mut group_starts_arg = group_starts;
    let mut group_lengths_arg = group_lengths;
    let mut group_count_arg = group_count;
    let mut routes_arg = routes;
    let mut experts_arg = experts;
    let mut route_parameters = [
        (&mut indices_arg as *mut *const u32).cast::<c_void>(),
        (&mut sorted_routes_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_experts_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_starts_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_lengths_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_count_arg as *mut *mut u32).cast::<c_void>(),
        (&mut routes_arg as *mut u32).cast::<c_void>(),
        (&mut experts_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.build_routes.launch(
            LaunchConfig::new([1, 1, 1], [256, 1, 1], 0),
            stream,
            &mut route_parameters,
        )?;
    }

    let mut input_arg = input;
    let mut input_tiles_arg = input_tiles;
    let mut input_scales_arg = input_scales;
    let mut top_k_arg = top_k;
    let mut in_features_arg = in_features;
    let mut workers_arg = WORKERS;
    let mut quantize_parameters = [
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut sorted_routes_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_starts_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_lengths_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_count_arg as *mut *mut u32).cast::<c_void>(),
        (&mut input_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut input_scales_arg as *mut *mut u32).cast::<c_void>(),
        (&mut top_k_arg as *mut u32).cast::<c_void>(),
        (&mut in_features_arg as *mut u32).cast::<c_void>(),
        (&mut workers_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.quantize.launch(
            LaunchConfig::new([in_features / 64, WORKERS, 1], [128, 1, 1], 0),
            stream,
            &mut quantize_parameters,
        )?;
    }

    let mut tiled_weight_arg = tiled_weight;
    let mut tiled_scales_arg = tiled_scales;
    let mut global_scales_arg = global_scales;
    let mut output_arg = output;
    let mut out_features_arg = out_features;
    let mut gemm_parameters = [
        (&mut sorted_routes_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_experts_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_starts_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_lengths_arg as *mut *mut u32).cast::<c_void>(),
        (&mut group_count_arg as *mut *mut u32).cast::<c_void>(),
        (&mut input_tiles_arg as *mut *mut u8).cast::<c_void>(),
        (&mut input_scales_arg as *mut *mut u32).cast::<c_void>(),
        (&mut tiled_weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut tiled_scales_arg as *mut *const u8).cast::<c_void>(),
        (&mut global_scales_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_arg as *mut *mut f32).cast::<c_void>(),
        (&mut out_features_arg as *mut u32).cast::<c_void>(),
        (&mut in_features_arg as *mut u32).cast::<c_void>(),
        (&mut workers_arg as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        functions.gemm.launch(
            LaunchConfig::new([out_features / 8, WORKERS, 1], [32, 1, 1], 0),
            stream,
            &mut gemm_parameters,
        )
    }
}
