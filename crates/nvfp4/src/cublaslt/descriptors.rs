use crate::cuda::check_cublas;
use crate::error::Result;
use crate::ffi;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;

pub(crate) struct MatmulDesc(pub(crate) ffi::cublasLtMatmulDesc_t);

impl MatmulDesc {
    pub(crate) fn create(
        compute_type: ffi::cublasComputeType_t,
        scale_type: ffi::cudaDataType_t,
    ) -> Result<Self> {
        let mut desc = null_mut();
        unsafe {
            check_cublas(
                "cublasLtMatmulDescCreate",
                ffi::cublasLtMatmulDescCreate(&mut desc, compute_type, scale_type),
            )?;
        }
        Ok(Self(desc))
    }

    pub(crate) fn set_i32(&self, attr: i32, value: i32, name: &'static str) -> Result<()> {
        unsafe {
            check_cublas(
                name,
                ffi::cublasLtMatmulDescSetAttribute(
                    self.0,
                    attr,
                    (&value as *const i32).cast(),
                    size_of::<i32>(),
                ),
            )
        }
    }

    pub(crate) fn set_ptr<T>(&self, attr: i32, ptr: *const T, name: &'static str) -> Result<()> {
        let ptr_value = ptr.cast::<c_void>();
        unsafe {
            check_cublas(
                name,
                ffi::cublasLtMatmulDescSetAttribute(
                    self.0,
                    attr,
                    (&ptr_value as *const *const c_void).cast(),
                    size_of::<*const c_void>(),
                ),
            )
        }
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cublasLtMatmulDescDestroy(self.0);
        }
    }
}

pub(crate) struct MatrixLayout(pub(crate) ffi::cublasLtMatrixLayout_t);

impl MatrixLayout {
    pub(crate) fn create(
        ty: ffi::cudaDataType_t,
        rows: usize,
        cols: usize,
        ld: usize,
    ) -> Result<Self> {
        let mut layout = null_mut();
        unsafe {
            check_cublas(
                "cublasLtMatrixLayoutCreate",
                ffi::cublasLtMatrixLayoutCreate(
                    &mut layout,
                    ty,
                    rows as u64,
                    cols as u64,
                    ld as i64,
                ),
            )?;
        }
        Ok(Self(layout))
    }

    pub(crate) fn set_i32(&self, attr: i32, value: i32, name: &'static str) -> Result<()> {
        unsafe {
            check_cublas(
                name,
                ffi::cublasLtMatrixLayoutSetAttribute(
                    self.0,
                    attr,
                    (&value as *const i32).cast(),
                    size_of::<i32>(),
                ),
            )
        }
    }

    pub(crate) fn set_i64(&self, attr: i32, value: i64, name: &'static str) -> Result<()> {
        unsafe {
            check_cublas(
                name,
                ffi::cublasLtMatrixLayoutSetAttribute(
                    self.0,
                    attr,
                    (&value as *const i64).cast(),
                    size_of::<i64>(),
                ),
            )
        }
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cublasLtMatrixLayoutDestroy(self.0);
        }
    }
}

pub(crate) struct MatmulPreference(pub(crate) ffi::cublasLtMatmulPreference_t);

impl MatmulPreference {
    pub(crate) fn create(workspace_limit: u64) -> Result<Self> {
        let mut pref = null_mut();
        unsafe {
            check_cublas(
                "cublasLtMatmulPreferenceCreate",
                ffi::cublasLtMatmulPreferenceCreate(&mut pref),
            )?;
            check_cublas(
                "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)",
                ffi::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    ffi::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&workspace_limit as *const u64).cast(),
                    size_of::<u64>(),
                ),
            )?;
        }
        Ok(Self(pref))
    }
}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cublasLtMatmulPreferenceDestroy(self.0);
        }
    }
}
