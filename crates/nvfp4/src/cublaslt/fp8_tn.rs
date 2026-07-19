use super::GemmShape;
use super::descriptors::{MatmulDesc, MatmulPreference, MatrixLayout};
use super::handle::CublasLt;
use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, check_cublas};
use crate::error::{Error, Result};
use crate::ffi;
use std::mem::MaybeUninit;
use std::ptr::null_mut;

/// Persistent cuBLASLt plan for `D[M,N] = A[K,M]^T * B[K,N]` with E4M3 inputs.
pub struct Fp8TnMatmulPlan {
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

impl Fp8TnMatmulPlan {
    /// Creates a plan with E4M3 A/B inputs and f32 output.
    pub fn new(lt: &CublasLt, shape: GemmShape, workspace_limit: u64) -> Result<Self> {
        if shape.m == 0 || shape.n == 0 || shape.k == 0 {
            return Err(Error::Shape {
                label: "FP8 TN shape",
                expected: "non-zero M, N, and K".to_string(),
                actual: format!("M={} N={} K={}", shape.m, shape.n, shape.k),
            });
        }

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

        let a_layout = MatrixLayout::create(ffi::CUDA_R_8F_E4M3, shape.k, shape.m, shape.k)?;
        let b_layout = MatrixLayout::create(ffi::CUDA_R_8F_E4M3, shape.k, shape.n, shape.k)?;
        let d_layout = MatrixLayout::create(ffi::CUDA_R_32F, shape.m, shape.n, shape.m)?;
        let pref = MatmulPreference::create(workspace_limit)?;

        let mut heuristic = MaybeUninit::<ffi::cublasLtMatmulHeuristicResult_t>::zeroed();
        let mut returned = 0i32;
        unsafe {
            check_cublas(
                "cublasLtMatmulAlgoGetHeuristic(FP8 TN)",
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
            return Err(Error::EmptyHeuristic("FP8 TN heuristic"));
        }
        let heuristic = unsafe { heuristic.assume_init() };
        check_cublas("FP8 TN heuristic state", heuristic.state)?;
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

    /// Returns workspace bytes required by the selected algorithm.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_size
    }

    /// Enqueues the planned matmul with scalar alpha and beta zero.
    pub fn run_with_alpha_on_stream(
        &self,
        lt: &CublasLt,
        a_kxm: &DeviceBuffer<u8>,
        b_kxn: &DeviceBuffer<u8>,
        mut output: DeviceOutput<'_, f32>,
        alpha: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        let a_len = self
            .shape
            .k
            .checked_mul(self.shape.m)
            .ok_or_else(|| Error::Shape {
                label: "FP8 TN A length",
                expected: "K * M without overflow".to_string(),
                actual: format!("K={} M={}", self.shape.k, self.shape.m),
            })?;
        let b_len = self
            .shape
            .k
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "FP8 TN B length",
                expected: "K * N without overflow".to_string(),
                actual: format!("K={} N={}", self.shape.k, self.shape.n),
            })?;
        let d_len = self
            .shape
            .m
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "FP8 TN output length",
                expected: "M * N without overflow".to_string(),
                actual: format!("M={} N={}", self.shape.m, self.shape.n),
            })?;
        if a_kxm.len() != a_len || b_kxn.len() < b_len || output.len() < d_len {
            return Err(Error::Shape {
                label: "FP8 TN buffers",
                expected: format!("A={a_len} B>={b_len} output>={d_len}"),
                actual: format!(
                    "A={} B={} output={}",
                    a_kxm.len(),
                    b_kxn.len(),
                    output.len()
                ),
            });
        }
        if !alpha.is_finite() {
            return Err(Error::Format {
                label: "FP8 TN alpha",
                detail: format!("expected finite alpha, got {alpha}"),
            });
        }

        let beta = 0.0f32;
        let workspace_ptr = self
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr.cast())
            .unwrap_or(null_mut());
        let output_ptr = output.buffer_mut().ptr;
        unsafe {
            check_cublas(
                "cublasLtMatmul(FP8 E4M3 -> F32)",
                ffi::cublasLtMatmul(
                    lt.handle,
                    self.desc.0,
                    (&alpha as *const f32).cast(),
                    a_kxm.ptr.cast(),
                    self.a_layout.0,
                    b_kxn.ptr.cast(),
                    self.b_layout.0,
                    (&beta as *const f32).cast(),
                    output_ptr.cast(),
                    self.d_layout.0,
                    output_ptr.cast(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{cuda_e4m3_code, e4m3_value};
    use crate::kernels::non_gemm::quantize_fp8_e4m3_f32_into_on_stream;

    #[test]
    fn fp8_tn_matches_quantized_cpu_reference() {
        const M: usize = 256;
        const K: usize = 256;
        const INPUT_SCALE: f32 = 0.125;
        const WEIGHT_SCALE: f32 = 0.03125;

        let input: Vec<f32> = (0..K)
            .map(|idx| ((idx % 15) as f32 - 7.0) * INPUT_SCALE)
            .collect();
        let weight: Vec<u8> = (0..M * K)
            .map(|idx| [0x00, 0x30, 0x38, 0xb0, 0xb8][idx % 5])
            .collect();
        let expected: Vec<f32> = (0..M)
            .map(|row| {
                let sum = (0..K)
                    .map(|col| {
                        let input_code = cuda_e4m3_code(input[col] / INPUT_SCALE);
                        e4m3_value(weight[row * K + col]) * e4m3_value(input_code)
                    })
                    .sum::<f32>();
                sum * INPUT_SCALE * WEIGHT_SCALE
            })
            .collect();

        let lt = CublasLt::new().expect("cuBLASLt");
        let plan =
            Fp8TnMatmulPlan::new(&lt, GemmShape::new(M, 1, K), 8 << 20).expect("FP8 TN plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let weight_device = DeviceBuffer::from_host(&weight).expect("weight");
        let mut input_fp8 = DeviceBuffer::zeroed(K).expect("input FP8");
        let mut output = DeviceBuffer::zeroed(M).expect("output");

        quantize_fp8_e4m3_f32_into_on_stream(
            &input_device,
            input_fp8.output(),
            INPUT_SCALE,
            &stream,
        )
        .expect("quantize input");
        plan.run_with_alpha_on_stream(
            &lt,
            &weight_device,
            &input_fp8,
            output.output(),
            INPUT_SCALE * WEIGHT_SCALE,
            &stream,
        )
        .expect("FP8 TN matmul");
        let actual = output.copy_to_host(&stream).expect("read output");

        for (idx, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let error = (actual - expected).abs();
            let allowed = 1e-4 + 1e-4 * expected.abs();
            assert!(
                error <= allowed,
                "FP8 TN mismatch at {idx}: actual={actual} expected={expected} error={error} allowed={allowed}"
            );
        }
    }
}
