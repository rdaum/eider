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
    a_leading_dimension: usize,
    batch_count: usize,
    a_batch_stride: usize,
    b_batch_stride: usize,
    d_batch_stride: usize,
    desc: MatmulDesc,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    d_layout: MatrixLayout,
    _pref: MatmulPreference,
    algo: ffi::cublasLtMatmulAlgo_t,
    workspace: Option<DeviceBuffer<u8>>,
    workspace_size: usize,
    output_type: Bf16TnOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bf16TnOutput {
    F32,
    Bf16,
}

impl Bf16TnMatmulPlan {
    /// Creates a plan with BF16 A/B inputs and f32 output.
    pub fn new(lt: &CublasLt, shape: GemmShape, workspace_limit: u64) -> Result<Self> {
        Self::new_with_a_leading_dimension(lt, shape, shape.k, workspace_limit)
    }

    /// Creates a plan whose A columns use an explicit leading dimension.
    pub fn new_with_a_leading_dimension(
        lt: &CublasLt,
        shape: GemmShape,
        a_leading_dimension: usize,
        workspace_limit: u64,
    ) -> Result<Self> {
        Self::new_strided_batch_with_a_leading_dimension(
            lt,
            shape,
            a_leading_dimension,
            1,
            0,
            0,
            0,
            workspace_limit,
        )
    }

    /// Creates a strided-batched plan with contiguous matrices in each batch.
    #[allow(clippy::too_many_arguments)]
    pub fn new_strided_batch(
        lt: &CublasLt,
        shape: GemmShape,
        batch_count: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
        d_batch_stride: usize,
        workspace_limit: u64,
    ) -> Result<Self> {
        Self::new_strided_batch_with_a_leading_dimension(
            lt,
            shape,
            shape.k,
            batch_count,
            a_batch_stride,
            b_batch_stride,
            d_batch_stride,
            workspace_limit,
        )
    }

    /// Creates a strided-batched plan with an explicit A leading dimension.
    #[allow(clippy::too_many_arguments)]
    pub fn new_strided_batch_with_a_leading_dimension(
        lt: &CublasLt,
        shape: GemmShape,
        a_leading_dimension: usize,
        batch_count: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
        d_batch_stride: usize,
        workspace_limit: u64,
    ) -> Result<Self> {
        Self::new_strided_batch_with_a_leading_dimension_and_output(
            lt,
            shape,
            a_leading_dimension,
            batch_count,
            a_batch_stride,
            b_batch_stride,
            d_batch_stride,
            workspace_limit,
            Bf16TnOutput::F32,
        )
    }

    /// Creates a strided-batched plan with an explicit A leading dimension and BF16 output.
    #[allow(clippy::too_many_arguments)]
    pub fn new_strided_batch_with_a_leading_dimension_bf16_output(
        lt: &CublasLt,
        shape: GemmShape,
        a_leading_dimension: usize,
        batch_count: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
        d_batch_stride: usize,
        workspace_limit: u64,
    ) -> Result<Self> {
        Self::new_strided_batch_with_a_leading_dimension_and_output(
            lt,
            shape,
            a_leading_dimension,
            batch_count,
            a_batch_stride,
            b_batch_stride,
            d_batch_stride,
            workspace_limit,
            Bf16TnOutput::Bf16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_strided_batch_with_a_leading_dimension_and_output(
        lt: &CublasLt,
        shape: GemmShape,
        a_leading_dimension: usize,
        batch_count: usize,
        a_batch_stride: usize,
        b_batch_stride: usize,
        d_batch_stride: usize,
        workspace_limit: u64,
        output_type: Bf16TnOutput,
    ) -> Result<Self> {
        if shape.m == 0 || shape.n == 0 || shape.k == 0 {
            return Err(Error::Shape {
                label: "BF16 TN shape",
                expected: "non-zero M, N, and K".to_string(),
                actual: format!("M={} N={} K={}", shape.m, shape.n, shape.k),
            });
        }
        if a_leading_dimension < shape.k {
            return Err(Error::Shape {
                label: "BF16 TN A leading dimension",
                expected: format!("at least K={}", shape.k),
                actual: a_leading_dimension.to_string(),
            });
        }
        if batch_count == 0 || batch_count > i32::MAX as usize {
            return Err(Error::Shape {
                label: "BF16 TN batch count",
                expected: format!("1..={}", i32::MAX),
                actual: batch_count.to_string(),
            });
        }
        if batch_count > 1
            && [a_batch_stride, b_batch_stride, d_batch_stride]
                .into_iter()
                .any(|stride| stride == 0 || stride > i64::MAX as usize)
        {
            return Err(Error::Shape {
                label: "BF16 TN batch strides",
                expected: "non-zero strides representable as i64".to_string(),
                actual: format!("A={a_batch_stride} B={b_batch_stride} D={d_batch_stride}"),
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

        let a_layout =
            MatrixLayout::create(ffi::CUDA_R_16BF, shape.k, shape.m, a_leading_dimension)?;
        let b_layout = MatrixLayout::create(ffi::CUDA_R_16BF, shape.k, shape.n, shape.k)?;
        let d_type = match output_type {
            Bf16TnOutput::F32 => ffi::CUDA_R_32F,
            Bf16TnOutput::Bf16 => ffi::CUDA_R_16BF,
        };
        let d_layout = MatrixLayout::create(d_type, shape.m, shape.n, shape.m)?;
        for (layout, stride) in [
            (&a_layout, a_batch_stride),
            (&b_layout, b_batch_stride),
            (&d_layout, d_batch_stride),
        ] {
            layout.set_i32(
                ffi::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                batch_count as i32,
                "cublasLtMatrixLayoutSetAttribute(BATCH_COUNT)",
            )?;
            if batch_count > 1 {
                layout.set_i64(
                    ffi::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    stride as i64,
                    "cublasLtMatrixLayoutSetAttribute(STRIDED_BATCH_OFFSET)",
                )?;
            }
        }
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
            a_leading_dimension,
            batch_count,
            a_batch_stride,
            b_batch_stride,
            d_batch_stride,
            desc,
            a_layout,
            b_layout,
            d_layout,
            _pref: pref,
            algo: heuristic.algo,
            workspace,
            workspace_size: heuristic.workspace_size,
            output_type,
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
        output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_offsets_on_stream(lt, a_kxm, 0, b_kxn, 0, output, 0, stream)
    }

    /// Enqueues the planned multiplication over contiguous submatrices.
    #[allow(clippy::too_many_arguments)]
    pub fn run_offsets_on_stream(
        &self,
        lt: &CublasLt,
        a_kxm: &DeviceBuffer<u16>,
        a_offset: usize,
        b_kxn: &DeviceBuffer<u16>,
        b_offset: usize,
        output: DeviceOutput<'_, f32>,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_offsets_impl(
            lt,
            a_kxm,
            a_offset,
            b_kxn,
            b_offset,
            output,
            output_offset,
            stream,
            Bf16TnOutput::F32,
        )
    }

    /// Enqueues the planned multiplication into a BF16 output buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn run_bf16_offsets_on_stream(
        &self,
        lt: &CublasLt,
        a_kxm: &DeviceBuffer<u16>,
        a_offset: usize,
        b_kxn: &DeviceBuffer<u16>,
        b_offset: usize,
        output: DeviceOutput<'_, u16>,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_offsets_impl(
            lt,
            a_kxm,
            a_offset,
            b_kxn,
            b_offset,
            output,
            output_offset,
            stream,
            Bf16TnOutput::Bf16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_offsets_impl<T: Copy>(
        &self,
        lt: &CublasLt,
        a_kxm: &DeviceBuffer<u16>,
        a_offset: usize,
        b_kxn: &DeviceBuffer<u16>,
        b_offset: usize,
        mut output: DeviceOutput<'_, T>,
        output_offset: usize,
        stream: &CudaStream,
        expected_output_type: Bf16TnOutput,
    ) -> Result<()> {
        if self.output_type != expected_output_type {
            return Err(Error::Shape {
                label: "BF16 TN output type",
                expected: format!("{expected_output_type:?}"),
                actual: format!("{:?}", self.output_type),
            });
        }
        let a_matrix_len = self
            .shape
            .m
            .checked_sub(1)
            .and_then(|columns| columns.checked_mul(self.a_leading_dimension))
            .and_then(|prefix| prefix.checked_add(self.shape.k))
            .ok_or_else(|| Error::Shape {
                label: "BF16 TN A length",
                expected: "(M - 1) * lda + K without overflow".to_string(),
                actual: format!(
                    "K={} M={} lda={}",
                    self.shape.k, self.shape.m, self.a_leading_dimension
                ),
            })?;
        let b_matrix_len = self
            .shape
            .k
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "BF16 TN B length",
                expected: "K * N without overflow".to_string(),
                actual: format!("K={} N={}", self.shape.k, self.shape.n),
            })?;
        let d_matrix_len = self
            .shape
            .m
            .checked_mul(self.shape.n)
            .ok_or_else(|| Error::Shape {
                label: "BF16 TN output length",
                expected: "M * N without overflow".to_string(),
                actual: format!("M={} N={}", self.shape.m, self.shape.n),
            })?;
        let batched_len = |matrix_len: usize, stride: usize| {
            (self.batch_count - 1)
                .checked_mul(stride)
                .and_then(|prefix| prefix.checked_add(matrix_len))
        };
        let a_len = batched_len(a_matrix_len, self.a_batch_stride).ok_or_else(|| Error::Shape {
            label: "BF16 TN batched A length",
            expected: "batch span without overflow".to_string(),
            actual: format!("count={} stride={}", self.batch_count, self.a_batch_stride),
        })?;
        let b_len = batched_len(b_matrix_len, self.b_batch_stride).ok_or_else(|| Error::Shape {
            label: "BF16 TN batched B length",
            expected: "batch span without overflow".to_string(),
            actual: format!("count={} stride={}", self.batch_count, self.b_batch_stride),
        })?;
        let d_len = batched_len(d_matrix_len, self.d_batch_stride).ok_or_else(|| Error::Shape {
            label: "BF16 TN batched output length",
            expected: "batch span without overflow".to_string(),
            actual: format!("count={} stride={}", self.batch_count, self.d_batch_stride),
        })?;
        if a_offset
            .checked_add(a_len)
            .is_none_or(|end| end > a_kxm.len())
            || b_offset
                .checked_add(b_len)
                .is_none_or(|end| end > b_kxn.len())
            || output_offset
                .checked_add(d_len)
                .is_none_or(|end| end > output.len())
        {
            return Err(Error::Shape {
                label: "BF16 TN buffers",
                expected: format!(
                    "A range {a_offset}..{}, B range {b_offset}..{}, output range {output_offset}..{}",
                    a_offset.saturating_add(a_len),
                    b_offset.saturating_add(b_len),
                    output_offset.saturating_add(d_len)
                ),
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
        let a_ptr = unsafe { a_kxm.ptr.add(a_offset) };
        let b_ptr = unsafe { b_kxn.ptr.add(b_offset) };
        let output_ptr = unsafe { output.buffer_mut().ptr.add(output_offset) };
        unsafe {
            check_cublas(
                "cublasLtMatmul(BF16 TN)",
                ffi::cublasLtMatmul(
                    lt.handle,
                    self.desc.0,
                    (&alpha as *const f32).cast(),
                    a_ptr.cast(),
                    self.a_layout.0,
                    b_ptr.cast(),
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
    use crate::kernels::non_gemm::{
        causal_window_softmax_f32_to_bf16_on_stream, f32_to_bf16_into_on_stream,
        prefill_gqa_attention_f32_into, unpack_heads_f32_into_on_stream,
    };
    use crate::{Sm12xKvCache, pack_token_heads_bf16_into_on_stream};

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

    #[test]
    fn bf16_tn_bf16_output_matches_cpu_reference() {
        const M: usize = 64;
        const N: usize = 9;
        const K: usize = 96;
        let a = (0..M * K)
            .map(|idx| f32_to_bf16(((idx * 11 % 37) as f32 - 18.0) * 0.0078125))
            .collect::<Vec<_>>();
        let b = (0..N * K)
            .map(|idx| f32_to_bf16(((idx * 7 % 31) as f32 - 15.0) * 0.015625))
            .collect::<Vec<_>>();
        let expected = (0..N)
            .flat_map(|row| {
                let a = &a;
                let b = &b;
                (0..M).map(move |out| {
                    f32_to_bf16(
                        (0..K)
                            .map(|col| {
                                bf16_to_f32(a[out * K + col]) * bf16_to_f32(b[row * K + col])
                            })
                            .sum::<f32>(),
                    )
                })
            })
            .collect::<Vec<_>>();

        let lt = CublasLt::new().expect("cuBLASLt");
        let plan = Bf16TnMatmulPlan::new_strided_batch_with_a_leading_dimension_bf16_output(
            &lt,
            GemmShape { m: M, n: N, k: K },
            K,
            1,
            0,
            0,
            0,
            8 << 20,
        )
        .expect("BF16-output plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a = DeviceBuffer::from_host(&a).expect("A");
        let b = DeviceBuffer::from_host(&b).expect("B");
        let mut output = DeviceBuffer::zeroed(N * M).expect("output");
        plan.run_bf16_offsets_on_stream(&lt, &a, 0, &b, 0, output.output(), 0, &stream)
            .expect("BF16-output matmul");
        assert_eq!(output.copy_to_host(&stream).expect("copy output"), expected);
    }

    #[test]
    fn tensor_core_gqa_matches_dense_prefill_attention() {
        const TOKENS: usize = 8;
        const Q_HEADS: usize = 8;
        const KV_HEADS: usize = 4;
        const HEAD_DIM: usize = 64;
        let q_len = TOKENS * Q_HEADS * HEAD_DIM;
        let kv_len = TOKENS * KV_HEADS * HEAD_DIM;
        let query_host = (0..q_len)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>();
        let key_host = (0..kv_len)
            .map(|index| ((index * 29 % 241) as f32 - 120.0) / 256.0)
            .collect::<Vec<_>>();
        let value_host = (0..kv_len)
            .map(|index| ((index * 43 % 239) as f32 - 119.0) / 256.0)
            .collect::<Vec<_>>();
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let value = DeviceBuffer::from_host(&value_host).expect("value");
        let stream = CudaStream::new_blocking().expect("stream");
        let mut reference = DeviceBuffer::zeroed(q_len).expect("reference");
        prefill_gqa_attention_f32_into(
            &query,
            &key,
            &value,
            reference.output(),
            TOKENS,
            0,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )
        .expect("dense attention");

        let mut cache = Sm12xKvCache::new(TOKENS, KV_HEADS, HEAD_DIM).expect("cache");
        cache
            .append_rows_at_offset_on_stream(&key, &value, 0, TOKENS, &stream)
            .expect("append cache");
        let mut packed_query = DeviceBuffer::zeroed(q_len).expect("packed query");
        let mut packed_key = DeviceBuffer::zeroed(kv_len).expect("packed key");
        let mut packed_value = DeviceBuffer::zeroed(kv_len).expect("packed value");
        let mut scores = DeviceBuffer::zeroed(Q_HEADS * TOKENS * TOKENS).expect("scores");
        let mut probabilities =
            DeviceBuffer::zeroed(Q_HEADS * TOKENS * TOKENS).expect("probabilities");
        let mut packed_output = DeviceBuffer::zeroed(q_len).expect("packed output");
        let mut output = DeviceBuffer::zeroed(q_len).expect("output");
        pack_token_heads_bf16_into_on_stream(
            &query,
            packed_query.output(),
            TOKENS,
            Q_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("pack query");
        cache
            .unpack_bf16_on_stream(packed_key.output(), packed_value.output(), &stream)
            .expect("unpack cache");
        let lt = CublasLt::new().expect("cuBLASLt");
        let queries_per_kv = Q_HEADS / KV_HEADS;
        let qk = Bf16TnMatmulPlan::new_strided_batch(
            &lt,
            GemmShape::new(TOKENS, TOKENS * queries_per_kv, HEAD_DIM),
            KV_HEADS,
            TOKENS * HEAD_DIM,
            queries_per_kv * TOKENS * HEAD_DIM,
            queries_per_kv * TOKENS * TOKENS,
            4 << 20,
        )
        .expect("QK plan");
        qk.run_on_stream(&lt, &packed_key, &packed_query, scores.output(), &stream)
            .expect("QK");
        causal_window_softmax_f32_to_bf16_on_stream(
            &scores,
            probabilities.output(),
            TOKENS,
            TOKENS,
            0,
            Q_HEADS,
            HEAD_DIM,
            None,
            &stream,
        )
        .expect("softmax");
        let pv = Bf16TnMatmulPlan::new_strided_batch(
            &lt,
            GemmShape::new(HEAD_DIM, TOKENS * queries_per_kv, TOKENS),
            KV_HEADS,
            HEAD_DIM * TOKENS,
            queries_per_kv * TOKENS * TOKENS,
            queries_per_kv * TOKENS * HEAD_DIM,
            4 << 20,
        )
        .expect("PV plan");
        pv.run_on_stream(
            &lt,
            &packed_value,
            &probabilities,
            packed_output.output(),
            &stream,
        )
        .expect("PV");
        unpack_heads_f32_into_on_stream(
            &packed_output,
            output.output(),
            TOKENS,
            Q_HEADS,
            HEAD_DIM,
            &stream,
        )
        .expect("unpack output");
        let reference = reference.copy_to_host(&stream).expect("reference read");
        let output = output.copy_to_host(&stream).expect("output read");
        let max_error = reference
            .iter()
            .zip(output.iter())
            .map(|(reference, output)| (reference - output).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.20, "attention max error {max_error}");
    }
}
