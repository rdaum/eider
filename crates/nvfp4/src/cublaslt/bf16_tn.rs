use super::GemmShape;
use super::descriptors::{MatmulDesc, MatmulPreference, MatrixLayout};
use super::handle::CublasLt;
use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, check_cublas};
use crate::error::{Error, Result};
use crate::ffi;
use std::mem::MaybeUninit;
use std::ptr::null_mut;

/// Persistent cuBLASLt plan for `D[M,N] = A[K,M]^T * B[K,N]` with BF16 inputs.
pub struct Bf16TnMatmulPlan {
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

impl Bf16TnMatmulPlan {
    /// Creates a plan with BF16 A/B inputs and f32 output.
    pub fn new(lt: &CublasLt, shape: GemmShape, workspace_limit: u64) -> Result<Self> {
        if shape.m == 0 || shape.n == 0 || shape.k == 0 {
            return Err(Error::Shape {
                label: "BF16 TN shape",
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

        let a_layout = MatrixLayout::create(ffi::CUDA_R_16BF, shape.k, shape.m, shape.k)?;
        let b_layout = MatrixLayout::create(ffi::CUDA_R_16BF, shape.k, shape.n, shape.k)?;
        let d_layout = MatrixLayout::create(ffi::CUDA_R_32F, shape.m, shape.n, shape.m)?;
        let pref = MatmulPreference::create(workspace_limit)?;

        let mut heuristic = MaybeUninit::<ffi::cublasLtMatmulHeuristicResult_t>::zeroed();
        let mut returned = 0i32;
        unsafe {
            check_cublas(
                "cublasLtMatmulAlgoGetHeuristic(BF16 TN)",
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
            return Err(Error::EmptyHeuristic("BF16 TN heuristic"));
        }
        let heuristic = unsafe { heuristic.assume_init() };
        check_cublas("BF16 TN heuristic state", heuristic.state)?;
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

    /// Enqueues the planned multiplication with scalar alpha and beta zero.
    pub fn run_on_stream(
        &self,
        lt: &CublasLt,
        a_kxm: &DeviceBuffer<u16>,
        b_kxn: &DeviceBuffer<u16>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let a_len = self
            .shape
            .k
            .checked_mul(self.shape.m)
            .ok_or_else(|| Error::Shape {
                label: "BF16 TN A length",
                expected: "K * M without overflow".to_string(),
                actual: format!("K={} M={}", self.shape.k, self.shape.m),
            })?;
        let b_len = self
            .shape
            .k
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "BF16 TN B length",
                expected: "K * N without overflow".to_string(),
                actual: format!("K={} N={}", self.shape.k, self.shape.n),
            })?;
        let d_len = self
            .shape
            .m
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "BF16 TN output length",
                expected: "M * N without overflow".to_string(),
                actual: format!("M={} N={}", self.shape.m, self.shape.n),
            })?;
        if a_kxm.len() != a_len || b_kxn.len() < b_len || output.len() != d_len {
            return Err(Error::Shape {
                label: "BF16 TN buffers",
                expected: format!("A={a_len} B>={b_len} output={d_len}"),
                actual: format!(
                    "A={} B={} output={}",
                    a_kxm.len(),
                    b_kxn.len(),
                    output.len()
                ),
            });
        }

        let alpha = 1.0f32;
        let beta = 0.0f32;
        let workspace_ptr = self
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr.cast())
            .unwrap_or(null_mut());
        let output_ptr = output.buffer_mut().ptr;
        unsafe {
            check_cublas(
                "cublasLtMatmul(BF16 -> F32)",
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
    use crate::format::{bf16_to_f32, f32_to_bf16};
    use crate::kernels::non_gemm::f32_to_bf16_into_on_stream;

    #[test]
    fn bf16_tn_matches_cpu_reference() {
        const M: usize = 96;
        const N: usize = 7;
        const K: usize = 128;
        let input = (0..N * K)
            .map(|idx| ((idx * 7 % 31) as f32 - 15.0) * 0.015625)
            .collect::<Vec<_>>();
        let weight = (0..M * K)
            .map(|idx| f32_to_bf16(((idx * 11 % 37) as f32 - 18.0) * 0.0078125))
            .collect::<Vec<_>>();
        let expected = (0..N)
            .flat_map(|row| {
                let input = &input;
                let weight = &weight;
                (0..M).map(move |out| {
                    (0..K)
                        .map(|col| {
                            bf16_to_f32(f32_to_bf16(input[row * K + col]))
                                * bf16_to_f32(weight[out * K + col])
                        })
                        .sum::<f32>()
                })
            })
            .collect::<Vec<_>>();

        let lt = CublasLt::new().expect("cuBLASLt");
        let plan =
            Bf16TnMatmulPlan::new(&lt, GemmShape { m: M, n: N, k: K }, 8 << 20).expect("BF16 plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input_f32 = DeviceBuffer::from_host(&input).expect("input f32");
        let mut input_bf16 = DeviceBuffer::zeroed(N * K).expect("input BF16");
        let weight = DeviceBuffer::from_host(&weight).expect("weight");
        let mut output = DeviceBuffer::zeroed(N * M).expect("output");
        f32_to_bf16_into_on_stream(&input_f32, input_bf16.output(), &stream)
            .expect("convert input");
        plan.run_on_stream(&lt, &weight, &input_bf16, output.output(), &stream)
            .expect("BF16 matmul");
        let actual = output.copy_to_host(&stream).expect("copy output");
        for (idx, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 0.002 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {idx}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
