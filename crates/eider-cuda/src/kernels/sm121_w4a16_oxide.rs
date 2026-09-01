//! cuda-oxide launch support for routed SM121 W4A16.

use crate::cuda_oxide::{self, Kernel, LaunchConfig};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::sync::OnceLock;

const SINGLE_THREADS: u32 = 512;
const BATCH_THREADS: u32 = 256;
const TILE_M: u32 = 16;

struct Functions {
    single_bf16: Kernel,
    single_f32: Kernel,
    top8_bf16: Kernel,
    top8_f32: Kernel,
    top10_bf16: Kernel,
    top10_f32: Kernel,
    batch_bf16: Kernel,
    batch_f32: Kernel,
}

impl Functions {
    fn load() -> Result<Self> {
        Ok(Self {
            single_bf16: Kernel::load(c"w4a16_single_bf16")?,
            single_f32: Kernel::load(c"w4a16_single_f32")?,
            top8_bf16: Kernel::load(c"w4a16_top8_bf16")?,
            top8_f32: Kernel::load(c"w4a16_top8_f32")?,
            top10_bf16: Kernel::load(c"w4a16_top10_bf16")?,
            top10_f32: Kernel::load(c"w4a16_top10_f32")?,
            batch_bf16: Kernel::load(c"w4a16_batch_bf16")?,
            batch_f32: Kernel::load(c"w4a16_batch_f32")?,
        })
    }

    fn select(&self, batch_size: u32, top_k: u32, write_f32: bool) -> &Kernel {
        if batch_size != 1 {
            return if write_f32 {
                &self.batch_f32
            } else {
                &self.batch_bf16
            };
        }
        match (top_k, write_f32) {
            (8, false) => &self.top8_bf16,
            (8, true) => &self.top8_f32,
            (10, false) => &self.top10_bf16,
            (10, true) => &self.top10_f32,
            (_, false) => &self.single_bf16,
            (_, true) => &self.single_f32,
        }
    }
}

static FUNCTIONS: OnceLock<Result<Functions>> = OnceLock::new();

fn functions() -> Result<&'static Functions> {
    match FUNCTIONS.get_or_init(Functions::load) {
        Ok(functions) => Ok(functions),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide W4A16 module",
            detail: error.to_string(),
        }),
    }
}

pub(crate) fn ensure_supported() -> Result<()> {
    cuda_oxide::ensure_supported()?;
    functions().map(|_| ())
}

/// Launches one routed W4A16 operation on an Eider-owned CUDA stream.
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
    output_bf16: *mut u16,
    output_f32: *mut f32,
    batch_size: u32,
    top_k: u32,
    out_features: u32,
    in_features: u32,
    stream: ffi::cudaStream_t,
) -> Result<()> {
    let functions = functions()?;
    let function = functions.select(batch_size, top_k, !output_f32.is_null());
    let mut indices_arg = indices;
    let mut input_arg = input;
    let mut tiled_weight_arg = tiled_weight;
    let mut tiled_scales_arg = tiled_scales;
    let mut global_scales_arg = global_scales;
    let mut output_bf16_arg = output_bf16;
    let mut output_f32_arg = output_f32;
    let mut batch_size_arg = batch_size;
    let mut top_k_arg = top_k;
    let mut out_features_arg = out_features;
    let mut in_features_arg = in_features;
    let mut parameters = [
        (&mut indices_arg as *mut *const u32).cast::<c_void>(),
        (&mut input_arg as *mut *const f32).cast::<c_void>(),
        (&mut tiled_weight_arg as *mut *const u8).cast::<c_void>(),
        (&mut tiled_scales_arg as *mut *const u8).cast::<c_void>(),
        (&mut global_scales_arg as *mut *const f32).cast::<c_void>(),
        (&mut output_bf16_arg as *mut *mut u16).cast::<c_void>(),
        (&mut output_f32_arg as *mut *mut f32).cast::<c_void>(),
        (&mut batch_size_arg as *mut u32).cast::<c_void>(),
        (&mut top_k_arg as *mut u32).cast::<c_void>(),
        (&mut out_features_arg as *mut u32).cast::<c_void>(),
        (&mut in_features_arg as *mut u32).cast::<c_void>(),
    ];
    let threads = if batch_size == 1 {
        SINGLE_THREADS
    } else {
        BATCH_THREADS
    };
    unsafe {
        function.launch(
            LaunchConfig::new(
                [out_features / TILE_M, batch_size * top_k, 1],
                [threads, 1, 1],
                0,
            ),
            stream,
            &mut parameters,
        )
    }
}
