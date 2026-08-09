use super::GemmShape;
use super::descriptors::{MatmulDesc, MatmulPreference, MatrixLayout};
use super::handle::CublasLt;
use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, check_cublas};
use crate::error::{Error, Result};
use crate::ffi;
use std::mem::MaybeUninit;
use std::ptr::null_mut;

/// Persistent cuBLASLt plan for `D[M,N] = A[K,M]^T * B[K,N]` with INT8 inputs.
pub struct Int8TnMatmulPlan {
    shape: GemmShape,
    desc: MatmulDesc,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    d_layout: MatrixLayout,
    _pref: MatmulPreference,
    algo: ffi::cublasLtMatmulAlgo_t,
    workspace: Option<DeviceBuffer<u8>>,
    workspace_size: usize,
}

impl Int8TnMatmulPlan {
    /// Creates an INT8-to-INT32 tensor-core plan for a fixed shape.
    pub fn new(lt: &CublasLt, shape: GemmShape, workspace_limit: u64) -> Result<Self> {
        if shape.m == 0 || shape.n == 0 || shape.k == 0 {
            return Err(Error::Shape {
                label: "INT8 TN shape",
                expected: "non-zero M, N, and K".to_string(),
                actual: format!("M={} N={} K={}", shape.m, shape.n, shape.k),
            });
        }
        let desc = MatmulDesc::create(ffi::CUBLAS_COMPUTE_32I, ffi::CUDA_R_32I)?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_TRANSA,
            ffi::CUBLAS_OP_T,
            "cublasLtMatmulDescSetAttribute(INT8 TRANSA)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_TRANSB,
            ffi::CUBLAS_OP_N,
            "cublasLtMatmulDescSetAttribute(INT8 TRANSB)",
        )?;
        let a_layout = MatrixLayout::create(ffi::CUDA_R_8I, shape.k, shape.m, shape.k)?;
        let b_layout = MatrixLayout::create(ffi::CUDA_R_8I, shape.k, shape.n, shape.k)?;
        let d_layout = MatrixLayout::create(ffi::CUDA_R_32I, shape.m, shape.n, shape.m)?;
        let pref = MatmulPreference::create(workspace_limit)?;
        let mut heuristic = MaybeUninit::<ffi::cublasLtMatmulHeuristicResult_t>::zeroed();
        let mut returned = 0i32;
        unsafe {
            check_cublas(
                "cublasLtMatmulAlgoGetHeuristic(INT8 TN)",
                ffi::cublasLtMatmulAlgoGetHeuristic(
                    lt.handle,
                    desc.0,
                    a_layout.0,
                    b_layout.0,
                    d_layout.0,
                    d_layout.0,
                    pref.0,
                    1,
                    heuristic.as_mut_ptr(),
                    &mut returned,
                ),
            )?;
        }
        if returned == 0 {
            return Err(Error::EmptyHeuristic("INT8 TN heuristic"));
        }
        let heuristic = unsafe { heuristic.assume_init() };
        check_cublas("INT8 TN heuristic state", heuristic.state)?;
        let workspace = if heuristic.workspace_size == 0 {
            None
        } else {
            Some(DeviceBuffer::zeroed(heuristic.workspace_size)?)
        };
        Ok(Self {
            shape,
            desc,
            a_layout,
            b_layout,
            d_layout,
            _pref: pref,
            algo: heuristic.algo,
            workspace,
            workspace_size: heuristic.workspace_size,
        })
    }

    /// Device workspace bytes retained by this plan.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_size
    }

    /// Enqueues the planned integer GEMM on `stream`.
    pub fn run_on_stream(
        &self,
        lt: &CublasLt,
        a_kxm: &DeviceBuffer<i8>,
        b_kxn: &DeviceBuffer<i8>,
        mut output: DeviceOutput<'_, i32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let a_len = self.shape.k * self.shape.m;
        let b_len = self.shape.k * self.shape.n;
        let d_len = self.shape.m * self.shape.n;
        if a_kxm.len() != a_len || b_kxn.len() != b_len || output.len() != d_len {
            return Err(Error::Shape {
                label: "INT8 TN buffers",
                expected: format!("A={a_len} B={b_len} output={d_len}"),
                actual: format!(
                    "A={} B={} output={}",
                    a_kxm.len(),
                    b_kxn.len(),
                    output.len()
                ),
            });
        }
        let alpha = 1i32;
        let beta = 0i32;
        let workspace_ptr = self
            .workspace
            .as_ref()
            .map(|buffer| buffer.as_const_ptr().cast_mut())
            .unwrap_or(null_mut());
        unsafe {
            check_cublas(
                "cublasLtMatmul(INT8 -> INT32)",
                ffi::cublasLtMatmul(
                    lt.handle,
                    self.desc.0,
                    (&alpha as *const i32).cast(),
                    a_kxm.as_const_ptr(),
                    self.a_layout.0,
                    b_kxn.as_const_ptr(),
                    self.b_layout.0,
                    (&beta as *const i32).cast(),
                    output.as_mut_ptr(),
                    self.d_layout.0,
                    output.as_mut_ptr(),
                    self.d_layout.0,
                    &self.algo,
                    workspace_ptr,
                    self.workspace_size,
                    stream.as_raw(),
                ),
            )
        }
    }
}
