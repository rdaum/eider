//! Loading and launching the embedded cuda-oxide device module.

use crate::cuda::check_cuda;
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::{CStr, c_void};
use std::ptr::null_mut;
use std::sync::OnceLock;

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eider_cuda_oxide.cubin"));

struct Module {
    context: ffi::CUcontext,
    _module: ffi::CUmodule,
    max_dynamic_shared_memory: u32,
}

// SAFETY: The process-wide module owns these handles for the life of the
// process. A launch makes the stored primary context current before use.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    fn load() -> Result<Self> {
        let mut device = 0;
        let mut major = 0;
        let mut minor = 0;
        let mut max_dynamic_shared_memory = 0;
        unsafe {
            check_cuda("cudaGetDevice", ffi::cudaGetDevice(&mut device))?;
            check_cuda(
                "cudaFree(cuda-oxide context init)",
                ffi::cudaFree(null_mut()),
            )?;
            check_cuda(
                "cudaDeviceGetAttribute(compute capability major)",
                ffi::cudaDeviceGetAttribute(
                    &mut major,
                    ffi::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
            )?;
            check_cuda(
                "cudaDeviceGetAttribute(compute capability minor)",
                ffi::cudaDeviceGetAttribute(
                    &mut minor,
                    ffi::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
            )?;
            check_cuda(
                "cudaDeviceGetAttribute(maximum opt-in shared memory)",
                ffi::cudaDeviceGetAttribute(
                    &mut max_dynamic_shared_memory,
                    ffi::CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    device,
                ),
            )?;
        }
        if (major, minor) != (12, 1) {
            return Err(Error::Format {
                label: "cuda-oxide device support",
                detail: format!("requires sm_121, found sm_{major}{minor}"),
            });
        }
        let max_dynamic_shared_memory =
            u32::try_from(max_dynamic_shared_memory).map_err(|_| Error::Format {
                label: "cuda-oxide dynamic shared memory",
                detail: format!("device reported invalid opt-in limit {max_dynamic_shared_memory}"),
            })?;

        let mut context = null_mut();
        check_driver("cuCtxGetCurrent(cuda-oxide)", unsafe {
            ffi::cuCtxGetCurrent(&mut context)
        })?;
        if context.is_null() {
            return Err(Error::Format {
                label: "cuda-oxide context",
                detail: "CUDA primary context is not current".to_string(),
            });
        }

        let mut module = null_mut();
        check_driver("cuModuleLoadData(cuda-oxide)", unsafe {
            ffi::cuModuleLoadData(&mut module, CUBIN.as_ptr().cast())
        })?;
        Ok(Self {
            context,
            _module: module,
            max_dynamic_shared_memory,
        })
    }

    fn bind_context(&self) -> Result<()> {
        check_driver("cuCtxSetCurrent(cuda-oxide)", unsafe {
            ffi::cuCtxSetCurrent(self.context)
        })
    }
}

static MODULE: OnceLock<Result<Module>> = OnceLock::new();

fn module() -> Result<&'static Module> {
    match MODULE.get_or_init(Module::load) {
        Ok(module) => Ok(module),
        Err(error) => Err(Error::Format {
            label: "cuda-oxide module",
            detail: error.to_string(),
        }),
    }
}

/// One function in the process-wide cuda-oxide module.
#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    function: ffi::CUfunction,
}

// SAFETY: The module remains loaded for the process lifetime. `launch` makes
// its primary context current on the calling thread before using this handle.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Resolves one kernel by its exported PTX name.
    pub(crate) fn load(name: &CStr) -> Result<Self> {
        let module = module()?;
        let mut function = null_mut();
        check_driver("cuModuleGetFunction(cuda-oxide)", unsafe {
            ffi::cuModuleGetFunction(&mut function, module._module, name.as_ptr())
        })?;
        Ok(Self { function })
    }

    /// Permits a dynamic shared-memory request for this kernel.
    pub(crate) fn allow_dynamic_shared_memory(self, bytes: u32) -> Result<Self> {
        let module = module()?;
        if bytes > module.max_dynamic_shared_memory {
            return Err(Error::Shape {
                label: "cuda-oxide dynamic shared memory",
                expected: format!("at most {} bytes", module.max_dynamic_shared_memory),
                actual: format!("{bytes} bytes"),
            });
        }
        module.bind_context()?;
        check_driver("cuFuncSetAttribute(cuda-oxide)", unsafe {
            ffi::cuFuncSetAttribute(
                self.function,
                ffi::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes as i32,
            )
        })?;
        Ok(self)
    }

    /// Permits this kernel to use the device's full dynamic shared memory.
    pub(crate) fn allow_max_dynamic_shared_memory(self) -> Result<Self> {
        let bytes = module()?.max_dynamic_shared_memory;
        self.allow_dynamic_shared_memory(bytes)
    }

    /// Launches this kernel on an Eider-owned CUDA stream.
    ///
    /// # Safety
    ///
    /// The parameter pointers must match the kernel signature. All referenced
    /// device storage must remain valid until the supplied stream completes.
    pub(crate) unsafe fn launch(
        &self,
        config: LaunchConfig,
        stream: ffi::cudaStream_t,
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        let module = module()?;
        if config.shared_memory_bytes > module.max_dynamic_shared_memory {
            return Err(Error::Shape {
                label: "cuda-oxide dynamic shared memory",
                expected: format!("at most {} bytes", module.max_dynamic_shared_memory),
                actual: format!("{} bytes", config.shared_memory_bytes),
            });
        }
        module.bind_context()?;
        check_driver("cuLaunchKernel(cuda-oxide)", unsafe {
            ffi::cuLaunchKernel(
                self.function,
                config.grid[0],
                config.grid[1],
                config.grid[2],
                config.block[0],
                config.block[1],
                config.block[2],
                config.shared_memory_bytes,
                stream,
                parameters.as_mut_ptr(),
                null_mut(),
            )
        })
    }
}

/// Raw launch dimensions for one embedded cuda-oxide kernel.
#[derive(Clone, Copy)]
pub(crate) struct LaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared_memory_bytes: u32,
}

impl LaunchConfig {
    /// Creates a one- or two-dimensional launch.
    pub(crate) const fn new(grid: [u32; 3], block: [u32; 3], shared_memory_bytes: u32) -> Self {
        Self {
            grid,
            block,
            shared_memory_bytes,
        }
    }
}

/// Loads the embedded module and checks its device requirement.
pub(crate) fn ensure_supported() -> Result<()> {
    module().map(|_| ())
}

fn check_driver(call: &'static str, status: ffi::CUresult) -> Result<()> {
    if status == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda(call, status))
    }
}
