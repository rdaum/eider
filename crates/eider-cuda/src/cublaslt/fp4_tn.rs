use super::descriptors::{MatmulDesc, MatmulPreference, MatrixLayout};
use super::handle::CublasLt;
use crate::cuda::{CudaStream, DeviceAddress, DeviceBuffer, DeviceInOut, check_cublas, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::matrix::{Bf16Matrix, F32Matrix, Nvfp4Matrix};
use crate::tensor::Nvfp4Tensor2d;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::{null, null_mut};

/// Logical shape for `D[M,N] = A[K,M]^T * B[K,N]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmShape {
    /// Number of rows in output D.
    pub m: usize,
    /// Number of columns in output D.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
}

impl GemmShape {
    /// Creates a shape for `D[M,N] = A[K,M]^T * B[K,N]`.
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }

    /// Returns the batch or sequence dimension for inference-shaped GEMMs.
    ///
    /// The current tensor convention treats activations as `K x N`, so `N` is
    /// the token/sequence count for projection, FFN, and unembed operations.
    pub const fn token_columns(&self) -> usize {
        self.n
    }
}

/// Named dense-layer GEMM shapes used by Qwen-style inference.
///
/// These helpers do not add a new execution path. They name the model
/// operations that should map onto the current FP4 TN convention:
/// `D[M,N] = weight[K,M]^T * activation[K,N]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceGemm {
    /// Attention Q projection, `hidden -> n_heads * head_dim`.
    QProjection {
        /// Number of activation columns, usually prompt length or decode batch.
        tokens: usize,
        /// Hidden size.
        hidden: usize,
        /// Q projection output width.
        q_width: usize,
    },
    /// Attention K or V projection, `hidden -> n_kv_heads * head_dim`.
    KvProjection {
        /// Number of activation columns, usually prompt length or decode batch.
        tokens: usize,
        /// Hidden size.
        hidden: usize,
        /// K/V projection output width.
        kv_width: usize,
    },
    /// Attention output projection, `n_heads * head_dim -> hidden`.
    OProjection {
        /// Number of activation columns, usually prompt length or decode batch.
        tokens: usize,
        /// Hidden size.
        hidden: usize,
        /// Attention output width.
        attn_width: usize,
    },
    /// Fused FFN gate/up projection, `hidden -> 2 * intermediate`.
    FfnGateUp {
        /// Number of activation columns, usually prompt length or decode batch.
        tokens: usize,
        /// Hidden size.
        hidden: usize,
        /// FFN intermediate width.
        intermediate: usize,
    },
    /// FFN down projection, `intermediate -> hidden`.
    FfnDown {
        /// Number of activation columns, usually prompt length or decode batch.
        tokens: usize,
        /// Hidden size.
        hidden: usize,
        /// FFN intermediate width.
        intermediate: usize,
    },
    /// Tied or explicit unembed projection, `hidden -> vocab`.
    Unembed {
        /// Number of activation columns, usually prompt length or decode batch.
        tokens: usize,
        /// Hidden size.
        hidden: usize,
        /// Vocabulary size.
        vocab: usize,
    },
}

impl InferenceGemm {
    /// Qwen3-4B Q projection shape.
    pub const fn qwen3_4b_q_projection(tokens: usize) -> Self {
        Self::QProjection {
            tokens,
            hidden: 2560,
            q_width: 4096,
        }
    }

    /// Qwen3-4B K or V projection shape.
    pub const fn qwen3_4b_kv_projection(tokens: usize) -> Self {
        Self::KvProjection {
            tokens,
            hidden: 2560,
            kv_width: 1024,
        }
    }

    /// Qwen3-4B attention output projection shape.
    pub const fn qwen3_4b_o_projection(tokens: usize) -> Self {
        Self::OProjection {
            tokens,
            hidden: 2560,
            attn_width: 4096,
        }
    }

    /// Qwen3-4B fused FFN gate/up projection shape.
    pub const fn qwen3_4b_ffn_gate_up(tokens: usize) -> Self {
        Self::FfnGateUp {
            tokens,
            hidden: 2560,
            intermediate: 9728,
        }
    }

    /// Qwen3-4B FFN down projection shape.
    pub const fn qwen3_4b_ffn_down(tokens: usize) -> Self {
        Self::FfnDown {
            tokens,
            hidden: 2560,
            intermediate: 9728,
        }
    }

    /// Qwen3-4B tied unembed projection shape.
    pub const fn qwen3_4b_unembed(tokens: usize) -> Self {
        Self::Unembed {
            tokens,
            hidden: 2560,
            vocab: 151_936,
        }
    }

    /// Returns the FP4 TN GEMM shape for this inference operation.
    pub const fn gemm_shape(&self) -> GemmShape {
        match *self {
            Self::QProjection {
                tokens,
                hidden,
                q_width,
            } => GemmShape::new(q_width, tokens, hidden),
            Self::KvProjection {
                tokens,
                hidden,
                kv_width,
            } => GemmShape::new(kv_width, tokens, hidden),
            Self::OProjection {
                tokens,
                hidden,
                attn_width,
            } => GemmShape::new(hidden, tokens, attn_width),
            Self::FfnGateUp {
                tokens,
                hidden,
                intermediate,
            } => GemmShape::new(2 * intermediate, tokens, hidden),
            Self::FfnDown {
                tokens,
                hidden,
                intermediate,
            } => GemmShape::new(hidden, tokens, intermediate),
            Self::Unembed {
                tokens,
                hidden,
                vocab,
            } => GemmShape::new(vocab, tokens, hidden),
        }
    }

    /// Returns a short stable label for reports and benchmark names.
    pub const fn label(&self) -> &'static str {
        match *self {
            Self::QProjection { .. } => "q_proj",
            Self::KvProjection { .. } => "kv_proj",
            Self::OProjection { .. } => "o_proj",
            Self::FfnGateUp { .. } => "ffn_gate_up",
            Self::FfnDown { .. } => "ffn_down",
            Self::Unembed { .. } => "unembed",
        }
    }
}

/// Borrowed NVFP4 inputs for the TN matmul convention.
///
/// A is stored as `K x M`; B is stored as `K x N`.
#[derive(Clone, Copy)]
pub struct Nvfp4TnInputs<'a> {
    /// Left input, stored column-major as `K x M`.
    pub a_kxm: &'a Nvfp4Matrix,
    /// Right input, stored column-major as `K x N`.
    pub b_kxn: &'a Nvfp4Matrix,
}

impl<'a> Nvfp4TnInputs<'a> {
    /// Creates borrowed TN inputs.
    pub const fn new(a_kxm: &'a Nvfp4Matrix, b_kxn: &'a Nvfp4Matrix) -> Self {
        Self { a_kxm, b_kxn }
    }

    fn validate(&self, shape: GemmShape) -> Result<()> {
        let a = self.a_kxm.input();
        let b = self.b_kxn.input();
        if (a.rows, a.cols) != (shape.k, shape.m) {
            return Err(Error::Shape {
                label: "FP4 TN A",
                expected: format!("A is KxM = {}x{}", shape.k, shape.m),
                actual: format!("{}x{}", a.rows, a.cols),
            });
        }
        if (b.rows, b.cols) != (shape.k, shape.n) {
            return Err(Error::Shape {
                label: "FP4 TN B",
                expected: format!("B is KxN = {}x{}", shape.k, shape.n),
                actual: format!("{}x{}", b.rows, b.cols),
            });
        }
        Ok(())
    }
}

/// Metadata for a selected cuBLASLt FP4 TN matmul plan.
#[derive(Clone, Copy, Debug)]
pub struct Fp4TnPlanMetadata {
    /// Logical GEMM shape.
    pub shape: GemmShape,
    /// Workspace bytes required by the selected algorithm.
    pub workspace_bytes: usize,
    /// Raw cuBLASLt algorithm storage.
    pub algorithm_data: [u64; 8],
}

/// Plan for one cuBLASLt FP4 `D = A^T * B` matmul shape.
///
/// The plan owns the cuBLASLt matmul descriptor, matrix layouts, chosen
/// heuristic algorithm, preference object, and any workspace required by the
/// selected algorithm. It does not own A/B/C/D matrices.
///
/// The A and B scale-buffer pointers are rebound before every launch, so one
/// plan can serve different matrices with the same shape.
pub struct Fp4TnMatmulPlan {
    shape: GemmShape,
    output: Fp4TnOutput,
    desc: MatmulDesc,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,
    _pref: MatmulPreference,
    algo: ffi::cublasLtMatmulAlgo_t,
    workspace: Option<DeviceBuffer<u8>>,
    workspace_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp4TnOutput {
    Bf16,
    F32,
}

impl Fp4TnMatmulPlan {
    /// Creates a plan for `D = A^T * B`.
    ///
    /// `inputs.a_kxm` must be `K x M`; `inputs.b_kxn` must be `K x N`; `c`
    /// supplies the BF16 `M x N` C/D layout. `workspace_limit` is passed to
    /// cuBLASLt's heuristic preference.
    pub fn new(
        lt: &CublasLt,
        shape: GemmShape,
        inputs: Nvfp4TnInputs<'_>,
        c: &Bf16Matrix,
        workspace_limit: u64,
    ) -> Result<Self> {
        inputs.validate(shape)?;
        validate_bf16_layout("FP4 TN C", shape, c)?;
        Self::new_with_output_type(
            lt,
            shape,
            inputs,
            c.rows,
            c.cols,
            c.ld,
            ffi::CUDA_R_16BF,
            Fp4TnOutput::Bf16,
            workspace_limit,
        )
    }

    /// Creates a plan for `D = A^T * B` with F32 C/D storage.
    pub fn new_f32_output(
        lt: &CublasLt,
        shape: GemmShape,
        inputs: Nvfp4TnInputs<'_>,
        c: &F32Matrix,
        workspace_limit: u64,
    ) -> Result<Self> {
        inputs.validate(shape)?;
        validate_f32_layout("FP4 TN F32 C", shape, c)?;
        Self::new_f32_output_for_shape(lt, shape, inputs, workspace_limit)
    }

    /// Creates an F32-output plan from its logical shape without allocating a
    /// placeholder output matrix.
    pub fn new_f32_output_for_shape(
        lt: &CublasLt,
        shape: GemmShape,
        inputs: Nvfp4TnInputs<'_>,
        workspace_limit: u64,
    ) -> Result<Self> {
        inputs.validate(shape)?;
        Self::new_with_output_type(
            lt,
            shape,
            inputs,
            shape.m,
            shape.n,
            shape.m,
            ffi::CUDA_R_32F,
            Fp4TnOutput::F32,
            workspace_limit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_output_type(
        lt: &CublasLt,
        shape: GemmShape,
        inputs: Nvfp4TnInputs<'_>,
        output_rows: usize,
        output_cols: usize,
        output_ld: usize,
        output_type: ffi::cudaDataType_t,
        output: Fp4TnOutput,
        workspace_limit: u64,
    ) -> Result<Self> {
        let desc = MatmulDesc::create(ffi::CUBLAS_COMPUTE_32F, ffi::CUDA_R_32F)?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_TRANSA,
            ffi::CUBLAS_OP_T,
            "cublasLtMatmulDescSetAttribute(TRANSA)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_TRANSB,
            ffi::CUBLAS_OP_N,
            "cublasLtMatmulDescSetAttribute(TRANSB)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            ffi::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
            "cublasLtMatmulDescSetAttribute(A_SCALE_MODE)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            ffi::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
            "cublasLtMatmulDescSetAttribute(B_SCALE_MODE)",
        )?;
        let a = inputs.a_kxm.input();
        let b = inputs.b_kxn.input();
        desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            a.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)",
        )?;
        desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            b.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)",
        )?;

        let a_layout = MatrixLayout::create(ffi::CUDA_R_4F_E2M1, a.rows, a.cols, a.ld)?;
        let b_layout = MatrixLayout::create(ffi::CUDA_R_4F_E2M1, b.rows, b.cols, b.ld)?;
        let c_layout = MatrixLayout::create(output_type, output_rows, output_cols, output_ld)?;
        let pref = MatmulPreference::create(workspace_limit)?;

        let mut heuristic = MaybeUninit::<ffi::cublasLtMatmulHeuristicResult_t>::zeroed();
        let mut returned = 0i32;
        unsafe {
            check_cublas(
                "cublasLtMatmulAlgoGetHeuristic(FP4 TN)",
                ffi::cublasLtMatmulAlgoGetHeuristic(
                    lt.handle,
                    desc.0,
                    a_layout.0,
                    b_layout.0,
                    c_layout.0,
                    c_layout.0,
                    pref.0,
                    1,
                    heuristic.as_mut_ptr(),
                    &mut returned,
                ),
            )?;
        }
        if returned == 0 {
            return Err(Error::EmptyHeuristic("FP4 TN heuristic"));
        }
        let heuristic = unsafe { heuristic.assume_init() };
        check_cublas("FP4 TN heuristic state", heuristic.state)?;
        let workspace = if heuristic.workspace_size == 0 {
            None
        } else {
            Some(DeviceBuffer::zeroed(heuristic.workspace_size)?)
        };

        Ok(Self {
            shape,
            output,
            desc,
            a_layout,
            b_layout,
            c_layout,
            _pref: pref,
            algo: heuristic.algo,
            workspace,
            workspace_size: heuristic.workspace_size,
        })
    }

    /// Returns the logical GEMM shape.
    pub fn shape(&self) -> GemmShape {
        self.shape
    }

    /// Returns workspace bytes required by the selected algorithm.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_size
    }

    /// Returns the raw cuBLASLt algorithm storage for diagnostics.
    pub fn algorithm_data(&self) -> [u64; 8] {
        self.algo.data
    }

    /// Returns shape, workspace, and selected algorithm metadata.
    pub fn metadata(&self) -> Fp4TnPlanMetadata {
        Fp4TnPlanMetadata {
            shape: self.shape,
            workspace_bytes: self.workspace_size,
            algorithm_data: self.algo.data,
        }
    }

    /// Launches the planned matmul on the default stream.
    ///
    /// This function enqueues work and returns after the cuBLASLt call returns;
    /// use [`crate::synchronize_device`] or CUDA event timing when a completion
    /// boundary is required.
    pub fn run_on_default_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &Bf16Matrix,
        d: &mut Bf16Matrix,
    ) -> Result<()> {
        self.run_with_alpha_on_default_stream(lt, inputs, c, d, 1.0)
    }

    /// Launches the planned matmul with an explicit host-side alpha scale.
    ///
    /// The operation is `D = alpha * A^T * B`; beta remains zero for the
    /// current inference path. This is useful for imported checkpoint formats
    /// that carry tensor-wide scalar scales in addition to cuBLASLt's FP4
    /// block-scale tensors.
    pub fn run_with_alpha_on_default_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &Bf16Matrix,
        d: &mut Bf16Matrix,
        alpha: f32,
    ) -> Result<()> {
        self.run_with_alpha_on_raw_stream(lt, inputs, c, d, alpha, null_mut())
    }

    /// Launches the planned matmul with an explicit host-side alpha scale on
    /// `stream`.
    pub fn run_with_alpha_on_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &Bf16Matrix,
        d: &mut Bf16Matrix,
        alpha: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_with_alpha_on_raw_stream(lt, inputs, c, d, alpha, stream.as_raw())
    }

    fn run_with_alpha_on_raw_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &Bf16Matrix,
        d: &mut Bf16Matrix,
        alpha: f32,
        stream: ffi::cudaStream_t,
    ) -> Result<()> {
        inputs.validate(self.shape)?;
        validate_bf16_layout("FP4 TN C", self.shape, c)?;
        validate_bf16_layout("FP4 TN D", self.shape, d)?;
        if self.output != Fp4TnOutput::Bf16 {
            return Err(Error::Shape {
                label: "FP4 TN output type",
                expected: "BF16 plan".to_string(),
                actual: "non-BF16 plan".to_string(),
            });
        }

        let beta = 0.0f32;
        let a = inputs.a_kxm.input();
        let b = inputs.b_kxn.input();
        let c = c.input();
        let mut d = d.output();
        self.desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            a.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)",
        )?;
        self.desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            b.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)",
        )?;
        let workspace_ptr = self
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr.cast())
            .unwrap_or(null_mut());
        unsafe {
            check_cublas(
                "cublasLtMatmul(FP4 E2M1 -> BF16)",
                ffi::cublasLtMatmul(
                    lt.handle,
                    self.desc.0,
                    (&alpha as *const f32).cast(),
                    a.values_ptr().cast(),
                    self.a_layout.0,
                    b.values_ptr().cast(),
                    self.b_layout.0,
                    (&beta as *const f32).cast(),
                    c.data_ptr().cast(),
                    self.c_layout.0,
                    d.data_mut_ptr().cast(),
                    self.c_layout.0,
                    &self.algo,
                    workspace_ptr,
                    self.workspace_size,
                    stream,
                ),
            )
        }
    }

    /// Launches the planned matmul with F32 C/D storage on the default stream.
    pub fn run_with_alpha_f32_output_on_default_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &F32Matrix,
        d: &mut F32Matrix,
        alpha: f32,
    ) -> Result<()> {
        self.run_with_alpha_f32_output_on_raw_stream(lt, inputs, c, d, alpha, null_mut())
    }

    /// Launches the planned matmul with F32 C/D storage on `stream`.
    pub fn run_with_alpha_f32_output_on_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &F32Matrix,
        d: &mut F32Matrix,
        alpha: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_with_alpha_f32_output_on_raw_stream(lt, inputs, c, d, alpha, stream.as_raw())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(missing_docs)]
    pub fn run_with_alpha_beta_f32_inout_buffer_on_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        mut output: DeviceInOut<'_, f32>,
        alpha: f32,
        beta: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        inputs.validate(self.shape)?;
        let expected_len = self
            .shape
            .m
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "FP4 TN F32 output buffers",
                expected: "M * N without overflow".to_string(),
                actual: format!("M={} N={}", self.shape.m, self.shape.n),
            })?;
        if output.len() < expected_len {
            return Err(Error::Shape {
                label: "FP4 TN F32 output buffers",
                expected: format!("at least {expected_len} values"),
                actual: format!("{} values", output.len()),
            });
        }
        if self.output != Fp4TnOutput::F32 {
            return Err(Error::Shape {
                label: "FP4 TN output type",
                expected: "F32 plan".to_string(),
                actual: "non-F32 plan".to_string(),
            });
        }

        let a = inputs.a_kxm.input();
        let b = inputs.b_kxn.input();
        self.desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            a.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)",
        )?;
        self.desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            b.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)",
        )?;

        let workspace_ptr = self
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr.cast())
            .unwrap_or(null_mut());
        unsafe {
            check_cublas(
                "cublasLtMatmul(FP4 E2M1 -> F32 alpha beta buffers)",
                ffi::cublasLtMatmul(
                    lt.handle,
                    self.desc.0,
                    (&alpha as *const f32).cast(),
                    a.values_ptr().cast(),
                    self.a_layout.0,
                    b.values_ptr().cast(),
                    self.b_layout.0,
                    (&beta as *const f32).cast(),
                    output.as_const_ptr(),
                    self.c_layout.0,
                    output.as_mut_ptr(),
                    self.c_layout.0,
                    &self.algo,
                    workspace_ptr,
                    self.workspace_size,
                    stream.as_raw(),
                ),
            )
        }
    }

    #[doc(hidden)]
    pub fn cutlass_fp4_gemv_f32_supported(&self) -> bool {
        if self.output != Fp4TnOutput::F32 || self.shape.n != 1 {
            return false;
        }
        unsafe {
            ffi::infer_cutlass_fp4_gemv_f32_supported(self.shape.m as u32, self.shape.k as u32) != 0
        }
    }

    #[doc(hidden)]
    pub fn run_cutlass_fp4_gemv_f32_on_stream(
        &self,
        inputs: Nvfp4TnInputs<'_>,
        c: &F32Matrix,
        d: &mut F32Matrix,
        alpha: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        inputs.validate(self.shape)?;
        validate_f32_layout("CUTLASS FP4 GEMV F32 C", self.shape, c)?;
        validate_f32_layout("CUTLASS FP4 GEMV F32 D", self.shape, d)?;
        if !self.cutlass_fp4_gemv_f32_supported() {
            return Err(Error::Shape {
                label: "CUTLASS FP4 GEMV shape",
                expected: "F32 N=1 plan with supported M,K".to_string(),
                actual: format!("M={}, N={}, K={}", self.shape.m, self.shape.n, self.shape.k),
            });
        }
        let a = inputs.a_kxm.input();
        let b = inputs.b_kxn.input();
        let c = c.input();
        let mut d = d.output();
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_gemv_f32_on_stream",
                ffi::infer_cutlass_fp4_gemv_f32_on_stream(
                    a.values_ptr(),
                    a.scales_ptr(),
                    b.values_ptr(),
                    b.scales_ptr(),
                    c.data_ptr(),
                    d.data_mut_ptr(),
                    self.shape.m as u32,
                    self.shape.k as u32,
                    alpha,
                    stream.as_raw(),
                ),
            )
        }
    }

    fn run_with_alpha_f32_output_on_raw_stream(
        &self,
        lt: &CublasLt,
        inputs: Nvfp4TnInputs<'_>,
        c: &F32Matrix,
        d: &mut F32Matrix,
        alpha: f32,
        stream: ffi::cudaStream_t,
    ) -> Result<()> {
        inputs.validate(self.shape)?;
        validate_f32_layout("FP4 TN F32 C", self.shape, c)?;
        validate_f32_layout("FP4 TN F32 D", self.shape, d)?;
        if self.output != Fp4TnOutput::F32 {
            return Err(Error::Shape {
                label: "FP4 TN output type",
                expected: "F32 plan".to_string(),
                actual: "non-F32 plan".to_string(),
            });
        }

        let beta = 0.0f32;
        let a = inputs.a_kxm.input();
        let b = inputs.b_kxn.input();
        let c = c.input();
        let mut d = d.output();
        self.desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            a.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)",
        )?;
        self.desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            b.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)",
        )?;
        let workspace_ptr = self
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr.cast())
            .unwrap_or(null_mut());
        unsafe {
            check_cublas(
                "cublasLtMatmul(FP4 E2M1 -> F32)",
                ffi::cublasLtMatmul(
                    lt.handle,
                    self.desc.0,
                    (&alpha as *const f32).cast(),
                    a.values_ptr().cast(),
                    self.a_layout.0,
                    b.values_ptr().cast(),
                    self.b_layout.0,
                    (&beta as *const f32).cast(),
                    c.data_ptr().cast(),
                    self.c_layout.0,
                    d.data_mut_ptr().cast(),
                    self.c_layout.0,
                    &self.algo,
                    workspace_ptr,
                    self.workspace_size,
                    stream,
                ),
            )
        }
    }
}

/// Persistent CUTLASS grouped FP4 GEMV plan with F32 output.
pub struct CutlassFp4GroupedGemvF32Plan {
    raw: *mut c_void,
    m: usize,
    k: usize,
    groups: usize,
}

impl CutlassFp4GroupedGemvF32Plan {
    /// Returns whether the native grouped GEMV wrapper supports this shape.
    pub fn supported(m: usize, k: usize, groups: usize) -> bool {
        if m > u32::MAX as usize || k > u32::MAX as usize || groups > u32::MAX as usize {
            return false;
        }
        unsafe {
            ffi::infer_cutlass_fp4_grouped_gemv_f32_supported(m as u32, k as u32, groups as u32)
                != 0
        }
    }

    /// Creates a persistent native grouped GEMV plan.
    pub fn new(m: usize, k: usize, groups: usize) -> Result<Self> {
        if !Self::supported(m, k, groups) {
            return Err(Error::Shape {
                label: "CUTLASS grouped FP4 GEMV shape",
                expected: "supported M,K,groups".to_string(),
                actual: format!("M={m}, K={k}, groups={groups}"),
            });
        }
        let raw = unsafe {
            ffi::infer_cutlass_fp4_grouped_gemv_f32_create(m as u32, k as u32, groups as u32)
        };
        if raw.is_null() {
            return Err(Error::Cuda("infer_cutlass_fp4_grouped_gemv_f32_create", -1));
        }
        Ok(Self { raw, m, k, groups })
    }

    /// Launches the grouped GEMV on `stream` using typed device-address tables.
    #[allow(clippy::too_many_arguments)]
    pub fn run_addresses_on_stream(
        &self,
        a_values: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales: &DeviceBuffer<DeviceAddress<u8>>,
        b_values: &DeviceBuffer<DeviceAddress<u8>>,
        b_scales: &DeviceBuffer<DeviceAddress<u8>>,
        c: &DeviceBuffer<DeviceAddress<f32>>,
        d: &DeviceBuffer<DeviceAddress<f32>>,
        alpha: f32,
        beta: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        for (label, len) in [
            ("A values", a_values.len()),
            ("A scales", a_scales.len()),
            ("B values", b_values.len()),
            ("B scales", b_scales.len()),
            ("C", c.len()),
            ("D", d.len()),
        ] {
            if len != self.groups {
                return Err(Error::Shape {
                    label: "CUTLASS grouped FP4 GEMV address table",
                    expected: format!("{} entries", self.groups),
                    actual: format!("{label} has {len}"),
                });
            }
        }
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_grouped_gemv_f32_on_stream",
                ffi::infer_cutlass_fp4_grouped_gemv_f32_on_stream(
                    self.raw,
                    a_values.as_const_ptr().cast(),
                    a_scales.as_const_ptr().cast(),
                    b_values.as_const_ptr().cast(),
                    b_scales.as_const_ptr().cast(),
                    c.as_const_ptr().cast(),
                    d.as_const_ptr().cast(),
                    alpha,
                    beta,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Launches the grouped GEMV with one typed output-address table used for
    /// both the native C and D operands.
    #[allow(clippy::too_many_arguments)]
    pub fn run_output_addresses_on_stream(
        &self,
        a_values: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales: &DeviceBuffer<DeviceAddress<u8>>,
        b_values: &DeviceBuffer<DeviceAddress<u8>>,
        b_scales: &DeviceBuffer<DeviceAddress<u8>>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        alpha: f32,
        beta: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        for (label, len) in [
            ("A values", a_values.len()),
            ("A scales", a_scales.len()),
            ("B values", b_values.len()),
            ("B scales", b_scales.len()),
            ("outputs", outputs.len()),
        ] {
            if len != self.groups {
                return Err(Error::Shape {
                    label: "CUTLASS grouped FP4 GEMV address table",
                    expected: format!("{} entries", self.groups),
                    actual: format!("{label} has {len}"),
                });
            }
        }
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_grouped_gemv_f32_on_stream",
                ffi::infer_cutlass_fp4_grouped_gemv_f32_on_stream(
                    self.raw,
                    a_values.as_const_ptr().cast(),
                    a_scales.as_const_ptr().cast(),
                    b_values.as_const_ptr().cast(),
                    b_scales.as_const_ptr().cast(),
                    outputs.as_const_ptr().cast(),
                    outputs.as_const_ptr().cast(),
                    alpha,
                    beta,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Launches grouped GEMV with typed selected A, shared B, and output
    /// address tables.
    #[allow(clippy::too_many_arguments)]
    pub fn run_indexed_a_addresses_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        a_values_table: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales_table: &DeviceBuffer<DeviceAddress<u8>>,
        table_len: usize,
        b_values: DeviceAddress<u8>,
        b_scales: DeviceAddress<u8>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        alpha: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        if indices.len() != self.groups || outputs.len() != self.groups {
            return Err(Error::Shape {
                label: "CUTLASS grouped FP4 GEMV indexed arrays",
                expected: format!("{} entries", self.groups),
                actual: format!("indices={} outputs={}", indices.len(), outputs.len()),
            });
        }
        if a_values_table.len() != table_len || a_scales_table.len() != table_len {
            return Err(Error::Shape {
                label: "CUTLASS grouped FP4 GEMV expert table",
                expected: format!("{table_len} entries"),
                actual: format!(
                    "A values={} A scales={}",
                    a_values_table.len(),
                    a_scales_table.len()
                ),
            });
        }
        if table_len > u32::MAX as usize {
            return Err(Error::Shape {
                label: "CUTLASS grouped FP4 GEMV expert table",
                expected: "table_len <= u32::MAX".to_string(),
                actual: table_len.to_string(),
            });
        }
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_grouped_gemv_f32_indexed_a_on_stream",
                ffi::infer_cutlass_fp4_grouped_gemv_f32_indexed_a_on_stream(
                    self.raw,
                    indices.as_const_ptr().cast(),
                    a_values_table.as_const_ptr().cast(),
                    a_scales_table.as_const_ptr().cast(),
                    table_len as u32,
                    b_values.as_const_ptr(),
                    b_scales.as_const_ptr(),
                    outputs.as_const_ptr().cast(),
                    alpha,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Launches hardware block-scaled grouped GEMV with typed device address
    /// tables for selected expert weights and output rows.
    #[allow(clippy::too_many_arguments)]
    pub fn run_indexed_a_tiled_scale_addresses_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        a_values_table: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales_table: &DeviceBuffer<DeviceAddress<u8>>,
        alpha_table: &DeviceBuffer<f32>,
        b: &Nvfp4Matrix,
        c: &F32Matrix,
        d: &DeviceBuffer<DeviceAddress<f32>>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_indexed_a_tiled_scales_impl(
            indices,
            a_values_table,
            a_scales_table,
            alpha_table,
            b,
            c,
            d,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_indexed_a_tiled_scales_impl(
        &self,
        indices: &DeviceBuffer<u32>,
        a_values_table: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales_table: &DeviceBuffer<DeviceAddress<u8>>,
        alpha_table: &DeviceBuffer<f32>,
        b: &Nvfp4Matrix,
        c: &F32Matrix,
        d: &DeviceBuffer<DeviceAddress<f32>>,
        stream: &CudaStream,
    ) -> Result<()> {
        if indices.len() != self.groups || d.len() != self.groups {
            return Err(Error::Shape {
                label: "CUTLASS indexed block-scaled FP4 GEMV route arrays",
                expected: format!("{} entries", self.groups),
                actual: format!("indices={} D={}", indices.len(), d.len()),
            });
        }
        let table_len = a_values_table.len();
        if table_len == 0 || a_scales_table.len() != table_len || alpha_table.len() != table_len {
            return Err(Error::Shape {
                label: "CUTLASS indexed block-scaled FP4 GEMV expert tables",
                expected: "matching non-empty value, scale, and alpha tables".to_string(),
                actual: format!(
                    "values={} scales={} alphas={}",
                    table_len,
                    a_scales_table.len(),
                    alpha_table.len()
                ),
            });
        }
        if table_len > u32::MAX as usize {
            return Err(Error::Shape {
                label: "CUTLASS indexed block-scaled FP4 GEMV expert table",
                expected: "at most u32::MAX entries".to_string(),
                actual: table_len.to_string(),
            });
        }
        if (b.rows, b.cols) != (self.k, 1) || (c.rows, c.cols) != (self.m, 1) {
            return Err(Error::Shape {
                label: "CUTLASS indexed block-scaled FP4 GEMV operands",
                expected: format!("B={}x1 C={}x1", self.k, self.m),
                actual: format!("B={}x{} C={}x{}", b.rows, b.cols, c.rows, c.cols),
            });
        }
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_grouped_gemv_f32_indexed_a_tiled_scales_on_stream",
                ffi::infer_cutlass_fp4_grouped_gemv_f32_indexed_a_tiled_scales_on_stream(
                    self.raw,
                    indices.as_const_ptr().cast(),
                    a_values_table.as_const_ptr().cast(),
                    a_scales_table.as_const_ptr().cast(),
                    alpha_table.as_const_ptr().cast(),
                    table_len as u32,
                    b.values_ptr(),
                    b.scales_ptr(),
                    c.data_ptr(),
                    d.as_const_ptr().cast(),
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Launches grouped GEMV with typed selected A tables, contiguous per-slot
    /// B operands, and contiguous F32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn run_contiguous_b_addresses_on_stream(
        &self,
        a_values_table: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales_table: &DeviceBuffer<DeviceAddress<u8>>,
        b_values: &DeviceBuffer<u8>,
        b_scales: &DeviceBuffer<u8>,
        d: &mut DeviceBuffer<f32>,
        alpha: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        let expected_b_values = self.groups * self.k / 2;
        let expected_b_scales = self.groups * (self.k / 16);
        let expected_d = self.groups * self.m;
        if a_values_table.len() != self.groups
            || a_scales_table.len() != self.groups
            || b_values.len() != expected_b_values
            || b_scales.len() != expected_b_scales
            || d.len() != expected_d
        {
            return Err(Error::Shape {
                label: "CUTLASS grouped FP4 contiguous-B GEMV buffers",
                expected: format!(
                    "A tables={} B values={} B scales={} D={}",
                    self.groups, expected_b_values, expected_b_scales, expected_d
                ),
                actual: format!(
                    "A values={} A scales={} B values={} B scales={} D={}",
                    a_values_table.len(),
                    a_scales_table.len(),
                    b_values.len(),
                    b_scales.len(),
                    d.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_grouped_gemv_f32_contiguous_b_on_stream",
                ffi::infer_cutlass_fp4_grouped_gemv_f32_contiguous_b_on_stream(
                    self.raw,
                    a_values_table.as_const_ptr().cast(),
                    a_scales_table.as_const_ptr().cast(),
                    b_values.as_const_ptr().cast(),
                    b_scales.as_const_ptr().cast(),
                    d.as_const_ptr().cast_mut().cast(),
                    alpha,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns `(m, k, groups)` for this plan.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.m, self.k, self.groups)
    }
}

impl Drop for CutlassFp4GroupedGemvF32Plan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::infer_cutlass_fp4_grouped_gemv_f32_destroy(self.raw);
            }
        }
    }
}

/// Persistent CUTLASS grouped NVFP4 GEMM plan with BF16 expert outputs.
///
/// Each expert has its own weight, activation, scale, output, and scalar pointer.
/// The token count for every expert remains device-resident, so running the plan
/// does not require a route-count readback.
pub struct CutlassFp4GroupedGemmPlan {
    raw: *mut c_void,
    m: usize,
    max_n: usize,
    k: usize,
    groups: usize,
}

impl CutlassFp4GroupedGemmPlan {
    /// Returns whether the native grouped GEMM wrapper supports this shape.
    pub fn supported(m: usize, max_n: usize, k: usize, groups: usize) -> bool {
        if [m, max_n, k, groups]
            .into_iter()
            .any(|dimension| dimension > u32::MAX as usize)
        {
            return false;
        }
        unsafe {
            ffi::infer_cutlass_fp4_grouped_gemm_supported(
                m as u32,
                max_n as u32,
                k as u32,
                groups as u32,
            ) != 0
        }
    }

    /// Creates a persistent grouped GEMM plan.
    pub fn new(m: usize, max_n: usize, k: usize, groups: usize) -> Result<Self> {
        if !Self::supported(m, max_n, k, groups) {
            return Err(Error::Shape {
                label: "CUTLASS grouped FP4 GEMM shape",
                expected: "supported M,max-N,K,groups".to_string(),
                actual: format!("M={m}, max-N={max_n}, K={k}, groups={groups}"),
            });
        }
        let raw = unsafe {
            ffi::infer_cutlass_fp4_grouped_gemm_create(
                m as u32,
                max_n as u32,
                k as u32,
                groups as u32,
            )
        };
        if raw.is_null() {
            return Err(Error::Cuda("infer_cutlass_fp4_grouped_gemm_create", -1));
        }
        Ok(Self {
            raw,
            m,
            max_n,
            k,
            groups,
        })
    }

    /// Launches grouped GEMM using device-resident pointer tables and route counts.
    #[allow(clippy::too_many_arguments)]
    pub fn run_on_stream(
        &self,
        a_values: &DeviceBuffer<DeviceAddress<u8>>,
        a_scales: &DeviceBuffer<DeviceAddress<u8>>,
        b_values: &DeviceBuffer<DeviceAddress<u8>>,
        b_scales: &DeviceBuffer<DeviceAddress<u8>>,
        output: &DeviceBuffer<DeviceAddress<u16>>,
        alpha: &DeviceBuffer<DeviceAddress<f32>>,
        tokens_per_expert: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        for (label, len) in [
            ("A values", a_values.len()),
            ("A scales", a_scales.len()),
            ("B values", b_values.len()),
            ("B scales", b_scales.len()),
            ("output", output.len()),
            ("alpha", alpha.len()),
            ("tokens per expert", tokens_per_expert.len()),
        ] {
            if len != self.groups {
                return Err(Error::Shape {
                    label: "CUTLASS grouped FP4 GEMM arrays",
                    expected: format!("{} entries", self.groups),
                    actual: format!("{label} has {len}"),
                });
            }
        }
        unsafe {
            check_cuda(
                "infer_cutlass_fp4_grouped_gemm_on_stream",
                ffi::infer_cutlass_fp4_grouped_gemm_on_stream(
                    self.raw,
                    a_values.as_const_ptr().cast(),
                    a_scales.as_const_ptr().cast(),
                    b_values.as_const_ptr().cast(),
                    b_scales.as_const_ptr().cast(),
                    output.as_const_ptr().cast(),
                    alpha.as_const_ptr().cast(),
                    tokens_per_expert.as_const_ptr().cast(),
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns `(M, max-N, K, groups)` for this plan.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.m, self.max_n, self.k, self.groups)
    }
}

impl Drop for CutlassFp4GroupedGemmPlan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::infer_cutlass_fp4_grouped_gemm_destroy(self.raw);
            }
        }
    }
}

/// Owned FP4 TN matmul operation.
///
/// This owns the cuBLASLt handle, NVFP4 inputs, BF16 C/D buffers, and plan
/// together. It is the preferred API when the operation can own its matrices,
/// because the plan's raw scale pointers cannot outlive the input buffers.
pub struct Fp4TnMatmul {
    plan: Fp4TnMatmulPlan,
    lt: CublasLt,
    a_kxm: Nvfp4Matrix,
    b_kxn: Nvfp4Matrix,
    c_mxn: Bf16Matrix,
    d_mxn: Bf16Matrix,
}

impl Fp4TnMatmul {
    /// Builds an owned operation from already-created TN inputs.
    pub fn from_parts(
        shape: GemmShape,
        a_kxm: Nvfp4Matrix,
        b_kxn: Nvfp4Matrix,
        workspace_limit: u64,
    ) -> Result<Self> {
        let lt = CublasLt::new()?;
        let c_mxn = Bf16Matrix::zeroed(shape.m, shape.n)?;
        let d_mxn = Bf16Matrix::zeroed(shape.m, shape.n)?;
        let plan = Fp4TnMatmulPlan::new(
            &lt,
            shape,
            Nvfp4TnInputs::new(&a_kxm, &b_kxn),
            &c_mxn,
            workspace_limit,
        )?;

        Ok(Self {
            plan,
            lt,
            a_kxm,
            b_kxn,
            c_mxn,
            d_mxn,
        })
    }

    /// Builds an owned operation from device-owned 2D NVFP4 tensors.
    ///
    /// The tensor dimensions must match the TN convention: A is `K x M`, B is
    /// `K x N`, and the output is `M x N`.
    pub fn from_tensors(
        shape: GemmShape,
        a_kxm: Nvfp4Tensor2d,
        b_kxn: Nvfp4Tensor2d,
        workspace_limit: u64,
    ) -> Result<Self> {
        Self::from_parts(
            shape,
            a_kxm.into_matrix(),
            b_kxn.into_matrix(),
            workspace_limit,
        )
    }

    /// Quantizes host column-major A and B values and builds an owned operation.
    ///
    /// `a_kxm_values` must have `K*M` values. `b_kxn_values` must have `K*N`
    /// values.
    pub fn quantized_col_major_f32(
        shape: GemmShape,
        a_kxm_values: &[f32],
        b_kxn_values: &[f32],
        workspace_limit: u64,
    ) -> Result<Self> {
        let a_kxm = Nvfp4Matrix::quantize_col_major_f32(shape.k, shape.m, a_kxm_values)?;
        let b_kxn = Nvfp4Matrix::quantize_col_major_f32(shape.k, shape.n, b_kxn_values)?;
        Self::from_parts(shape, a_kxm, b_kxn, workspace_limit)
    }

    /// Builds an all-ones owned operation for smoke tests and benchmarks.
    pub fn ones(shape: GemmShape, workspace_limit: u64) -> Result<Self> {
        let a_kxm = Nvfp4Matrix::ones_col_major(shape.k, shape.m)?;
        let b_kxn = Nvfp4Matrix::ones_col_major(shape.k, shape.n)?;
        Self::from_parts(shape, a_kxm, b_kxn, workspace_limit)
    }

    /// Launches the operation on the default stream.
    pub fn run_on_default_stream(&mut self) -> Result<()> {
        self.plan.run_on_default_stream(
            &self.lt,
            Nvfp4TnInputs::new(&self.a_kxm, &self.b_kxn),
            &self.c_mxn,
            &mut self.d_mxn,
        )
    }

    /// Returns the BF16 output matrix.
    pub fn output(&self) -> &Bf16Matrix {
        &self.d_mxn
    }

    /// Returns the logical GEMM shape.
    pub fn shape(&self) -> GemmShape {
        self.plan.shape()
    }

    /// Returns plan metadata for diagnostics and benchmark reporting.
    pub fn metadata(&self) -> Fp4TnPlanMetadata {
        self.plan.metadata()
    }
}

fn validate_bf16_layout(label: &'static str, shape: GemmShape, matrix: &Bf16Matrix) -> Result<()> {
    if (matrix.rows, matrix.cols) != (shape.m, shape.n) {
        return Err(Error::Shape {
            label,
            expected: format!("MxN = {}x{}", shape.m, shape.n),
            actual: format!("{}x{}", matrix.rows, matrix.cols),
        });
    }
    Ok(())
}

fn validate_f32_layout(label: &'static str, shape: GemmShape, matrix: &F32Matrix) -> Result<()> {
    if (matrix.rows, matrix.cols) != (shape.m, shape.n) {
        return Err(Error::Shape {
            label,
            expected: format!("MxN = {}x{}", shape.m, shape.n),
            actual: format!("{}x{}", matrix.rows, matrix.cols),
        });
    }
    Ok(())
}

pub(crate) fn fp32_matmul_smoke(lt: &CublasLt) -> Result<Vec<f32>> {
    let a = DeviceBuffer::from_host(&[1.0f32, 3.0, 2.0, 4.0])?;
    let b = DeviceBuffer::from_host(&[5.0f32, 7.0, 6.0, 8.0])?;
    let mut c = DeviceBuffer::<f32>::zeroed(4)?;
    let stream = crate::CudaStream::new_non_blocking()?;
    let desc = MatmulDesc::create(ffi::CUBLAS_COMPUTE_32F, ffi::CUDA_R_32F)?;
    let layout = MatrixLayout::create(ffi::CUDA_R_32F, 2, 2, 2)?;
    let alpha = 1.0f32;
    let beta = 0.0f32;

    unsafe {
        check_cublas(
            "cublasLtMatmul(FP32 2x2)",
            ffi::cublasLtMatmul(
                lt.handle,
                desc.0,
                (&alpha as *const f32).cast(),
                a.as_const_ptr(),
                layout.0,
                b.as_const_ptr(),
                layout.0,
                (&beta as *const f32).cast(),
                c.as_const_ptr(),
                layout.0,
                c.as_mut_ptr(),
                layout.0,
                null(),
                null_mut(),
                0,
                null_mut(),
            ),
        )?;
    }
    crate::synchronize_device()?;
    let actual = c.copy_to_host(&stream)?.into_vec();
    let expected = vec![19.0f32, 43.0, 22.0, 50.0];
    if actual
        .iter()
        .zip(&expected)
        .all(|(a, e)| (*a - *e).abs() < 1e-4)
    {
        Ok(actual)
    } else {
        Err(Error::Mismatch { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format;
    use crate::synchronize_device;

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (self.0 >> 40) as u32;
            (bits as f32) / ((1u32 << 24) as f32)
        }

        fn signed_value(&mut self) -> f32 {
            let uniform = self.next_f32() * 2.0 - 1.0;
            let taper = 0.25 + self.next_f32() * 2.75;
            uniform * taper
        }
    }

    fn random_col_major(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..rows * cols).map(|_| rng.signed_value()).collect()
    }

    fn transpose_col_major(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut transposed = vec![0.0; rows * cols];
        for col in 0..cols {
            for row in 0..rows {
                transposed[col + row * cols] = values[row + col * rows];
            }
        }
        transposed
    }

    fn assert_fp4_tn_matches_quantized_reference(m: usize, n: usize, k: usize) {
        let shape = GemmShape::new(m, n, k);
        let a_host = random_col_major(k, m, 0x1234_5678_9abc_def0 ^ m as u64);
        let b_host = random_col_major(k, n, 0xfedc_ba98_7654_3210 ^ n as u64);
        let a_quantized = format::quantize_nvfp4_col_major(k, m, &a_host);
        let b_quantized = format::quantize_nvfp4_col_major(k, n, &b_host);
        let mut matmul =
            Fp4TnMatmul::quantized_col_major_f32(shape, &a_host, &b_host, 4 * 1024 * 1024)
                .expect("create FP4 TN matmul");
        assert_eq!(matmul.shape(), shape);
        assert_eq!(matmul.metadata().shape, shape);
        matmul.run_on_default_stream().expect("run FP4 TN");
        synchronize_device().expect("synchronize FP4 TN");
        let stream = crate::CudaStream::new_non_blocking().expect("stream create");

        let actual = matmul
            .output()
            .data()
            .copy_to_host(&stream)
            .expect("copy output")
            .iter()
            .copied()
            .map(format::bf16_to_f32)
            .collect::<Vec<_>>();
        let a_t = transpose_col_major(&a_quantized.dequantized_values, k, m);
        let expected = format::cpu_matmul_col_major(&a_t, &b_quantized.dequantized_values, m, n, k);

        let mut max_abs_error = 0.0f32;
        let mut max_allowed = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let error = (actual - expected).abs();
            let allowed = 0.125f32.max(expected.abs() * 0.01);
            max_abs_error = max_abs_error.max(error);
            max_allowed = max_allowed.max(allowed);
            assert!(
                error <= allowed,
                "FP4 TN mismatch for shape {m}x{n}x{k}: actual={actual}, expected={expected}, error={error}, allowed={allowed}, max_abs_error={max_abs_error}, max_allowed={max_allowed}"
            );
        }
    }

    #[test]
    fn qwen3_4b_inference_shapes_match_dense_layer_dimensions() {
        assert_eq!(
            InferenceGemm::qwen3_4b_q_projection(1).gemm_shape(),
            GemmShape::new(4096, 1, 2560)
        );
        assert_eq!(
            InferenceGemm::qwen3_4b_kv_projection(8).gemm_shape(),
            GemmShape::new(1024, 8, 2560)
        );
        assert_eq!(
            InferenceGemm::qwen3_4b_o_projection(128).gemm_shape(),
            GemmShape::new(2560, 128, 4096)
        );
        assert_eq!(
            InferenceGemm::qwen3_4b_ffn_gate_up(1).gemm_shape(),
            GemmShape::new(19456, 1, 2560)
        );
        assert_eq!(
            InferenceGemm::qwen3_4b_ffn_down(8).gemm_shape(),
            GemmShape::new(2560, 8, 9728)
        );
        assert_eq!(
            InferenceGemm::qwen3_4b_unembed(1).gemm_shape(),
            GemmShape::new(151_936, 1, 2560)
        );
    }

    #[test]
    fn randomized_quantized_fp4_tn_64_square() {
        assert_fp4_tn_matches_quantized_reference(64, 64, 64);
    }

    #[test]
    fn randomized_quantized_fp4_tn_128_square() {
        assert_fp4_tn_matches_quantized_reference(128, 128, 128);
    }

    #[test]
    fn randomized_quantized_fp4_tn_partial_scale_tiles() {
        assert_fp4_tn_matches_quantized_reference(96, 80, 96);
    }

    #[test]
    fn randomized_quantized_fp4_tn_f32_output() {
        let m = 64;
        let n = 8;
        let k = 64;
        let shape = GemmShape::new(m, n, k);
        let a_host = random_col_major(k, m, 0x1357_9bdf_2468_ace0);
        let b_host = random_col_major(k, n, 0x2468_ace0_1357_9bdf);
        let a_quantized = format::quantize_nvfp4_col_major(k, m, &a_host);
        let b_quantized = format::quantize_nvfp4_col_major(k, n, &b_host);
        let a = Nvfp4Matrix::from_packed_col_major_parts(
            k,
            m,
            &a_quantized.packed_values,
            &a_quantized.scales,
        )
        .expect("A upload");
        let b = Nvfp4Matrix::from_packed_col_major_parts(
            k,
            n,
            &b_quantized.packed_values,
            &b_quantized.scales,
        )
        .expect("B upload");
        let c = F32Matrix::zeroed(m, n).expect("C alloc");
        let mut d = F32Matrix::zeroed(m, n).expect("D alloc");
        let lt = CublasLt::new().expect("cuBLASLt create");
        let plan = Fp4TnMatmulPlan::new_f32_output(
            &lt,
            shape,
            Nvfp4TnInputs::new(&a, &b),
            &c,
            4 * 1024 * 1024,
        )
        .expect("F32 output plan");
        plan.run_with_alpha_f32_output_on_default_stream(
            &lt,
            Nvfp4TnInputs::new(&a, &b),
            &c,
            &mut d,
            1.0,
        )
        .expect("F32 output matmul");
        synchronize_device().expect("F32 output sync");

        let stream = crate::CudaStream::new_non_blocking().expect("stream create");
        let actual = d.data().copy_to_host(&stream).expect("copy F32 output");
        let a_t = transpose_col_major(&a_quantized.dequantized_values, k, m);
        let expected = format::cpu_matmul_col_major(&a_t, &b_quantized.dequantized_values, m, n, k);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let error = (actual - expected).abs();
            let allowed = 0.125f32.max(expected.abs() * 0.01);
            assert!(
                error <= allowed,
                "FP4 TN F32 mismatch: actual={actual}, expected={expected}, error={error}, allowed={allowed}"
            );
        }
    }

    #[test]
    fn randomized_cutlass_fp4_gemv_f32_output() {
        let m = 64;
        let n = 1;
        let k = 64;
        let shape = GemmShape::new(m, n, k);
        let a_host = random_col_major(k, m, 0xaaaa_1357_9bdf_2468);
        let b_host = random_col_major(k, n, 0xbbbb_2468_ace0_1357);
        let a_quantized = format::quantize_nvfp4_col_major(k, m, &a_host);
        let b_quantized = format::quantize_nvfp4_col_major(k, n, &b_host);
        let a = Nvfp4Matrix::from_packed_col_major_parts(
            k,
            m,
            &a_quantized.packed_values,
            &a_quantized.scales,
        )
        .expect("A upload");
        let b = Nvfp4Matrix::from_packed_col_major_parts(
            k,
            n,
            &b_quantized.packed_values,
            &b_quantized.scales,
        )
        .expect("B upload");
        let c = F32Matrix::zeroed(m, n).expect("C alloc");
        let mut d = F32Matrix::zeroed(m, n).expect("D alloc");
        let lt = CublasLt::new().expect("cuBLASLt create");
        let plan = Fp4TnMatmulPlan::new_f32_output(
            &lt,
            shape,
            Nvfp4TnInputs::new(&a, &b),
            &c,
            4 * 1024 * 1024,
        )
        .expect("F32 output plan");
        if !plan.cutlass_fp4_gemv_f32_supported() {
            return;
        }
        let stream = crate::CudaStream::new_non_blocking().expect("stream create");
        plan.run_cutlass_fp4_gemv_f32_on_stream(
            Nvfp4TnInputs::new(&a, &b),
            &c,
            &mut d,
            1.0,
            &stream,
        )
        .expect("CUTLASS F32 output gemv");
        let actual = d.data().copy_to_host(&stream).expect("copy F32 output");
        let a_t = transpose_col_major(&a_quantized.dequantized_values, k, m);
        let expected = format::cpu_matmul_col_major(&a_t, &b_quantized.dequantized_values, m, n, k);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let error = (actual - expected).abs();
            let allowed = 0.125f32.max(expected.abs() * 0.01);
            assert!(
                error <= allowed,
                "CUTLASS FP4 GEMV mismatch: actual={actual}, expected={expected}, error={error}, allowed={allowed}"
            );
        }
    }

    #[test]
    fn randomized_cutlass_fp4_grouped_gemv_f32_output() {
        let m = 64;
        let k = 64;
        let groups = 2;
        if !CutlassFp4GroupedGemvF32Plan::supported(m, k, groups) {
            return;
        }

        let mut a_value_ptrs = Vec::new();
        let mut a_scale_ptrs = Vec::new();
        let mut a_quantized_grouped = Vec::new();
        let mut b_matrices = Vec::new();
        let mut b_quantized = Vec::new();
        for group in 0..groups {
            let a_host = random_col_major(k, m, 0x1357_2468_aaaa_bbbb ^ group as u64);
            let b_host = random_col_major(k, 1, 0x2468_1357_cccc_dddd ^ group as u64);

            let aq = format::quantize_nvfp4_col_major(k, m, &a_host);
            let a_values = DeviceBuffer::from_host(&aq.packed_values).expect("A values upload");
            let mut raw_scales = vec![0u8; m * (k / 16)];
            for col in 0..m {
                for row_block in 0..k / 16 {
                    raw_scales[col * (k / 16) + row_block] =
                        aq.scales[format::ue4m3_tiled_scale_offset(col, row_block, k)];
                }
            }
            let a_scales = DeviceBuffer::from_host(&raw_scales).expect("A scales upload");
            a_value_ptrs.push(a_values);
            a_scale_ptrs.push(a_scales);
            a_quantized_grouped.push(aq);

            // B: standard per-row quantization (SFB is per N column, no grouping needed)
            let bq = format::quantize_nvfp4_col_major(k, 1, &b_host);
            b_matrices.push(
                Nvfp4Matrix::from_packed_col_major_parts(k, 1, &bq.packed_values, &bq.scales)
                    .expect("B upload"),
            );
            b_quantized.push(bq);
        }

        let a_values_table = DeviceBuffer::from_host(
            &a_value_ptrs
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("A values ptr table");
        let a_scales_table = DeviceBuffer::from_host(
            &a_scale_ptrs
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("A scales ptr table");
        let mut b_value_ptrs = Vec::with_capacity(groups);
        let mut b_scale_ptrs = Vec::with_capacity(groups);
        for matrix in &b_matrices {
            b_value_ptrs.push(matrix.values_address());
            b_scale_ptrs.push(matrix.scales_address());
        }
        let b_values = DeviceBuffer::from_host(&b_value_ptrs).expect("B values ptrs");
        let b_scales = DeviceBuffer::from_host(&b_scale_ptrs).expect("B scales ptrs");

        let outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1).expect("D alloc"))
            .collect::<Vec<_>>();
        let output_addresses = DeviceBuffer::from_host(
            &outputs
                .iter()
                .map(F32Matrix::data_address)
                .collect::<Vec<_>>(),
        )
        .expect("output addresses");

        let plan = CutlassFp4GroupedGemvF32Plan::new(m, k, groups).expect("grouped plan");
        let stream = crate::CudaStream::new_non_blocking().expect("stream create");
        plan.run_output_addresses_on_stream(
            &a_values_table,
            &a_scales_table,
            &b_values,
            &b_scales,
            &output_addresses,
            1.0,
            0.0,
            &stream,
        )
        .expect("grouped GEMV launch");
        for group in 0..groups {
            let actual = outputs[group]
                .data()
                .copy_to_host(&stream)
                .expect("copy output");
            let a_t = transpose_col_major(&a_quantized_grouped[group].dequantized_values, k, m);
            let expected =
                format::cpu_matmul_col_major(&a_t, &b_quantized[group].dequantized_values, m, 1, k);
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                let error = (actual - expected).abs();
                let allowed = 0.125f32.max(expected.abs() * 0.01);
                assert!(
                    error <= allowed,
                    "group {group} grouped GEMV mismatch: actual={actual}, expected={expected}, error={error}, allowed={allowed}"
                );
            }
        }
    }

    #[test]
    fn randomized_cutlass_fp4_grouped_gemm_bf16_output() {
        let m = 128;
        let k = 256;
        let columns = [3usize, 5usize];
        let groups = columns.len();
        let max_n = *columns.iter().max().expect("columns");
        if !CutlassFp4GroupedGemmPlan::supported(m, max_n, k, groups) {
            return;
        }

        let mut a_matrices = Vec::new();
        let mut b_matrices = Vec::new();
        let mut a_quantized = Vec::new();
        let mut b_quantized = Vec::new();
        for (group, &n) in columns.iter().enumerate() {
            let a_host = random_col_major(k, m, 0x3141_5926_5358_9793 ^ group as u64);
            let b_host = random_col_major(k, n, 0x2718_2818_2845_9045 ^ group as u64);
            let aq = format::quantize_nvfp4_col_major(k, m, &a_host);
            let bq = format::quantize_nvfp4_col_major(k, n, &b_host);
            a_matrices.push(
                Nvfp4Matrix::from_packed_col_major_parts(k, m, &aq.packed_values, &aq.scales)
                    .expect("A upload"),
            );
            b_matrices.push(
                Nvfp4Matrix::from_packed_col_major_parts(k, n, &bq.packed_values, &bq.scales)
                    .expect("B upload"),
            );
            a_quantized.push(aq);
            b_quantized.push(bq);
        }

        let a_values = DeviceBuffer::from_host(
            &a_matrices
                .iter()
                .map(Nvfp4Matrix::values_address)
                .collect::<Vec<_>>(),
        )
        .expect("A values table");
        let a_scales = DeviceBuffer::from_host(
            &a_matrices
                .iter()
                .map(Nvfp4Matrix::scales_address)
                .collect::<Vec<_>>(),
        )
        .expect("A scales table");
        let b_values = DeviceBuffer::from_host(
            &b_matrices
                .iter()
                .map(Nvfp4Matrix::values_address)
                .collect::<Vec<_>>(),
        )
        .expect("B values table");
        let b_scales = DeviceBuffer::from_host(
            &b_matrices
                .iter()
                .map(Nvfp4Matrix::scales_address)
                .collect::<Vec<_>>(),
        )
        .expect("B scales table");

        let mut outputs = columns
            .iter()
            .map(|&n| Bf16Matrix::zeroed(m, n).expect("D alloc"))
            .collect::<Vec<_>>();
        let output_ptrs = outputs
            .iter_mut()
            .map(|output| output.data_address())
            .collect::<Vec<_>>();
        let output = DeviceBuffer::from_host(&output_ptrs).expect("D table");
        let mut alpha_storage = (0..groups)
            .map(|_| DeviceBuffer::from_host(&[1.0f32]).expect("alpha"))
            .collect::<Vec<_>>();
        let alpha = DeviceBuffer::from_host(
            &alpha_storage
                .iter_mut()
                .map(|scalar| scalar.cuda_address())
                .collect::<Vec<_>>(),
        )
        .expect("alpha table");
        let tokens_per_expert =
            DeviceBuffer::from_host(&columns.iter().map(|&n| n as u32).collect::<Vec<_>>())
                .expect("token counts");

        let plan = CutlassFp4GroupedGemmPlan::new(m, max_n, k, groups).expect("grouped GEMM plan");
        let stream = crate::CudaStream::new_non_blocking().expect("stream");
        plan.run_on_stream(
            &a_values,
            &a_scales,
            &b_values,
            &b_scales,
            &output,
            &alpha,
            &tokens_per_expert,
            &stream,
        )
        .expect("grouped GEMM launch");

        for group in 0..groups {
            let actual = outputs[group]
                .data
                .copy_to_host(&stream)
                .expect("copy output");
            let a_t = transpose_col_major(&a_quantized[group].dequantized_values, k, m);
            let expected = format::cpu_matmul_col_major(
                &a_t,
                &b_quantized[group].dequantized_values,
                m,
                columns[group],
                k,
            );
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let actual = format::bf16_to_f32(*actual);
                let error = (actual - expected).abs();
                let allowed = 0.25f32.max(expected.abs() * 0.02);
                assert!(
                    error <= allowed,
                    "group {group} value {index}: actual={actual}, expected={expected}, error={error}, allowed={allowed}"
                );
            }
        }
    }

    #[test]
    fn randomized_cutlass_fp4_grouped_gemv_large_k_f32_output() {
        let m = 1024;
        let k = 2048;
        let groups = 2;
        if !CutlassFp4GroupedGemvF32Plan::supported(m, k, groups) {
            return;
        }

        let mut a_value_ptrs = Vec::new();
        let mut a_scale_ptrs = Vec::new();
        let mut a_quantized_grouped = Vec::new();
        let mut b_value_ptrs_host = Vec::new();
        let mut b_scale_ptrs_host = Vec::new();
        let mut b_quantized = Vec::new();
        let mut owned_a_values = Vec::new();
        let mut owned_a_scales = Vec::new();
        let mut owned_b_values = Vec::new();
        let mut owned_b_scales = Vec::new();
        for group in 0..groups {
            let a_host = random_col_major(k, m, 0x1357_2468_aaaa_bbbb ^ group as u64);
            let b_host = random_col_major(k, 1, 0x2468_1357_cccc_dddd ^ group as u64);
            let aq = format::quantize_nvfp4_col_major(k, m, &a_host);
            let mut raw_a_scales = vec![0u8; m * (k / 16)];
            for col in 0..m {
                for row_block in 0..k / 16 {
                    raw_a_scales[col * (k / 16) + row_block] =
                        aq.scales[format::ue4m3_tiled_scale_offset(col, row_block, k)];
                }
            }
            let bq = format::quantize_nvfp4_col_major(k, 1, &b_host);
            let mut raw_b_scales = vec![0u8; k / 16];
            for (row_block, scale) in raw_b_scales.iter_mut().enumerate() {
                *scale = bq.scales[format::ue4m3_tiled_scale_offset(0, row_block, k)];
            }
            owned_a_values.push(DeviceBuffer::from_host(&aq.packed_values).expect("A values"));
            owned_a_scales.push(DeviceBuffer::from_host(&raw_a_scales).expect("A scales"));
            owned_b_values.push(DeviceBuffer::from_host(&bq.packed_values).expect("B values"));
            owned_b_scales.push(DeviceBuffer::from_host(&raw_b_scales).expect("B scales"));
            a_quantized_grouped.push(aq);
            b_quantized.push(bq);
        }
        for group in 0..groups {
            a_value_ptrs.push(owned_a_values[group].cuda_address());
            a_scale_ptrs.push(owned_a_scales[group].cuda_address());
            b_value_ptrs_host.push(owned_b_values[group].cuda_address());
            b_scale_ptrs_host.push(owned_b_scales[group].cuda_address());
        }
        let a_values_table = DeviceBuffer::from_host(&a_value_ptrs).expect("A values ptrs");
        let a_scales_table = DeviceBuffer::from_host(&a_scale_ptrs).expect("A scales ptrs");
        let b_values = DeviceBuffer::from_host(&b_value_ptrs_host).expect("B values ptrs");
        let b_scales = DeviceBuffer::from_host(&b_scale_ptrs_host).expect("B scales ptrs");
        let outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1).expect("D alloc"))
            .collect::<Vec<_>>();
        let output_addresses = DeviceBuffer::from_host(
            &outputs
                .iter()
                .map(F32Matrix::data_address)
                .collect::<Vec<_>>(),
        )
        .expect("output addresses");
        let plan = CutlassFp4GroupedGemvF32Plan::new(m, k, groups).expect("grouped plan");
        let stream = crate::CudaStream::new_non_blocking().expect("stream");
        plan.run_output_addresses_on_stream(
            &a_values_table,
            &a_scales_table,
            &b_values,
            &b_scales,
            &output_addresses,
            1.0,
            0.0,
            &stream,
        )
        .expect("grouped GEMV launch");
        for group in 0..groups {
            let actual = outputs[group].data().copy_to_host(&stream).expect("copy");
            let a_t = transpose_col_major(&a_quantized_grouped[group].dequantized_values, k, m);
            let expected =
                format::cpu_matmul_col_major(&a_t, &b_quantized[group].dequantized_values, m, 1, k);
            for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let error = (actual - expected).abs();
                let allowed = 0.125f32.max(expected.abs() * 0.01);
                assert!(
                    error <= allowed,
                    "group {group} row {idx} grouped large-k GEMV mismatch: actual={actual}, expected={expected}, error={error}, allowed={allowed}"
                );
            }
        }
    }

    #[test]
    #[ignore = "CUDA graph capture must not run alongside parallel default-stream CUDA tests"]
    fn fp4_tn_matmul_replays_from_cuda_graph() {
        let m = 64;
        let n = 64;
        let k = 64;
        let shape = GemmShape::new(m, n, k);
        let a_host = random_col_major(k, m, 0x1111_2222_3333_4444);
        let b_host = random_col_major(k, n, 0x5555_6666_7777_8888);
        let a_quantized = format::quantize_nvfp4_col_major(k, m, &a_host);
        let b_quantized = format::quantize_nvfp4_col_major(k, n, &b_host);
        let mut matmul =
            Fp4TnMatmul::quantized_col_major_f32(shape, &a_host, &b_host, 4 * 1024 * 1024)
                .expect("create FP4 TN matmul");

        matmul.run_on_default_stream().expect("warm FP4 TN");
        synchronize_device().expect("warm FP4 TN sync");

        let stream = crate::CudaStream::new_non_blocking().expect("stream create");
        let graph = stream
            .capture(|stream| {
                matmul.plan.run_with_alpha_on_stream(
                    &matmul.lt,
                    Nvfp4TnInputs::new(&matmul.a_kxm, &matmul.b_kxn),
                    &matmul.c_mxn,
                    &mut matmul.d_mxn,
                    1.0,
                    stream,
                )
            })
            .expect("graph capture");
        graph.launch(&stream).expect("graph launch");
        let actual = matmul
            .output()
            .data()
            .copy_to_host(&stream)
            .expect("copy output")
            .iter()
            .copied()
            .map(format::bf16_to_f32)
            .collect::<Vec<_>>();
        let a_t = transpose_col_major(&a_quantized.dequantized_values, k, m);
        let expected = format::cpu_matmul_col_major(&a_t, &b_quantized.dequantized_values, m, n, k);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            let error = (actual - expected).abs();
            let allowed = 0.125f32.max(expected.abs() * 0.01);
            assert!(
                error <= allowed,
                "graph FP4 TN mismatch: actual={actual}, expected={expected}, error={error}, allowed={allowed}"
            );
        }
    }

    #[test]
    #[ignore = "diagnostic probe for cuBLASLt FP4 output type support"]
    fn probe_fp4_tn_output_type_support() {
        let lt = CublasLt::new().expect("cuBLASLt create");
        let shape = GemmShape::new(64, 1, 64);
        let a_host = random_col_major(shape.k, shape.m, 0xabab_abab_abab_abab);
        let b_host = random_col_major(shape.k, shape.n, 0xcdcd_cdcd_cdcd_cdcd);
        let a = Nvfp4Matrix::quantize_col_major_f32(shape.k, shape.m, &a_host).expect("A");
        let b = Nvfp4Matrix::quantize_col_major_f32(shape.k, shape.n, &b_host).expect("B");
        let inputs = Nvfp4TnInputs::new(&a, &b);

        let candidates = [
            ("f32", ffi::CUDA_R_32F),
            ("f16", ffi::CUDA_R_16F),
            ("bf16", ffi::CUDA_R_16BF),
            ("fp8_e4m3", ffi::CUDA_R_8F_E4M3),
            ("fp4_e2m1", ffi::CUDA_R_4F_E2M1),
        ];
        let mut supported = Vec::new();
        for (label, dtype) in candidates {
            match fp4_tn_heuristic_for_output_type(&lt, shape, inputs, dtype) {
                Ok(workspace_size) => {
                    println!("fp4_tn_output_type {label} supported workspace={workspace_size}");
                    supported.push(label);
                }
                Err(err) => {
                    println!("fp4_tn_output_type {label} unsupported {err}");
                }
            }
        }

        assert!(
            supported.contains(&"bf16"),
            "current BF16 path must remain supported"
        );
    }

    fn fp4_tn_heuristic_for_output_type(
        lt: &CublasLt,
        shape: GemmShape,
        inputs: Nvfp4TnInputs<'_>,
        output_type: ffi::cudaDataType_t,
    ) -> Result<usize> {
        inputs.validate(shape)?;
        let desc = MatmulDesc::create(ffi::CUBLAS_COMPUTE_32F, ffi::CUDA_R_32F)?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_TRANSA,
            ffi::CUBLAS_OP_T,
            "cublasLtMatmulDescSetAttribute(TRANSA)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_TRANSB,
            ffi::CUBLAS_OP_N,
            "cublasLtMatmulDescSetAttribute(TRANSB)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            ffi::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
            "cublasLtMatmulDescSetAttribute(A_SCALE_MODE)",
        )?;
        desc.set_i32(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            ffi::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
            "cublasLtMatmulDescSetAttribute(B_SCALE_MODE)",
        )?;
        let a = inputs.a_kxm.input();
        let b = inputs.b_kxn.input();
        desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            a.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)",
        )?;
        desc.set_ptr(
            ffi::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            b.scales_ptr().cast_mut(),
            "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)",
        )?;

        let a_layout = MatrixLayout::create(ffi::CUDA_R_4F_E2M1, a.rows, a.cols, a.ld)?;
        let b_layout = MatrixLayout::create(ffi::CUDA_R_4F_E2M1, b.rows, b.cols, b.ld)?;
        let c_layout = MatrixLayout::create(output_type, shape.m, shape.n, shape.m)?;
        let d_layout = MatrixLayout::create(output_type, shape.m, shape.n, shape.m)?;
        let pref = MatmulPreference::create(4 * 1024 * 1024)?;
        let mut heuristic = MaybeUninit::<ffi::cublasLtMatmulHeuristicResult_t>::zeroed();
        let mut returned = 0i32;
        unsafe {
            check_cublas(
                "cublasLtMatmulAlgoGetHeuristic(FP4 TN output probe)",
                ffi::cublasLtMatmulAlgoGetHeuristic(
                    lt.handle,
                    desc.0,
                    a_layout.0,
                    b_layout.0,
                    c_layout.0,
                    d_layout.0,
                    pref.0,
                    1,
                    heuristic.as_mut_ptr(),
                    &mut returned,
                ),
            )?;
        }
        if returned == 0 {
            return Err(Error::EmptyHeuristic("FP4 TN output type probe"));
        }
        let heuristic = unsafe { heuristic.assume_init() };
        check_cublas("FP4 TN output probe heuristic state", heuristic.state)?;
        Ok(heuristic.workspace_size)
    }
}
