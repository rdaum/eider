//! Ownership of cuBLASLt library handles.

use crate::cuda::check_cublas;
use crate::error::Result;
use crate::ffi;
use std::ptr::null_mut;

/// Owns a cuBLASLt handle.
///
/// Create one handle per host execution context and pass it to plans and smoke
/// checks. The handle is destroyed with `cublasLtDestroy` on drop.
pub struct CublasLt {
    pub(crate) handle: ffi::cublasLtHandle_t,
}

impl CublasLt {
    /// Creates a new cuBLASLt handle.
    pub fn new() -> Result<Self> {
        let mut handle = null_mut();
        unsafe {
            check_cublas("cublasLtCreate", ffi::cublasLtCreate(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Returns the cuBLASLt runtime version.
    pub fn version() -> usize {
        unsafe { ffi::cublasLtGetVersion() }
    }
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cublasLtDestroy(self.handle);
        }
    }
}
