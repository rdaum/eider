use crate::cuda::{CudaStream, DeviceBuffer, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;

const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const CHUNK_TOKENS: usize = 64;

/// Native CUDA implementation of 64-token Qwen3.6 chunked Gated DeltaNet.
pub struct Qwen36ChunkedGdn;

impl Qwen36ChunkedGdn {
    /// Creates the stateless native CUDA launcher.
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        query: &DeviceBuffer<u16>,
        key: &DeviceBuffer<u16>,
        value: &DeviceBuffer<u16>,
        gate: &DeviceBuffer<u16>,
        beta: &DeviceBuffer<u16>,
        state: &DeviceBuffer<f32>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        chunk_offsets: &DeviceBuffer<i64>,
        gate_cumsum: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        a_inverse: &DeviceBuffer<u16>,
        w: &DeviceBuffer<u16>,
        u: &DeviceBuffer<u16>,
        h: &DeviceBuffer<u16>,
        value_new: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        sequence_count: usize,
        total_tokens: usize,
        chunk_count: usize,
    ) -> Result<()> {
        let vectors = total_tokens
            .checked_mul(HEADS)
            .and_then(|values| values.checked_mul(HEAD_DIM))
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 chunked GDN",
                expected: "token vector size without overflow".to_string(),
                actual: total_tokens.to_string(),
            })?;
        let token_heads = total_tokens
            .checked_mul(HEADS)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 chunked GDN",
                expected: "token-head size without overflow".to_string(),
                actual: total_tokens.to_string(),
            })?;
        let a_values = token_heads
            .checked_mul(CHUNK_TOKENS)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 chunked GDN",
                expected: "attention workspace size without overflow".to_string(),
                actual: token_heads.to_string(),
            })?;
        let recurrent_values = HEADS * HEAD_DIM * HEAD_DIM;
        let state_values = sequence_count
            .checked_mul(recurrent_values)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 chunked GDN",
                expected: "state workspace size without overflow".to_string(),
                actual: sequence_count.to_string(),
            })?;
        let h_values = chunk_count
            .checked_mul(recurrent_values)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.6 chunked GDN",
                expected: "chunk-state workspace size without overflow".to_string(),
                actual: chunk_count.to_string(),
            })?;
        if sequence_count == 0
            || total_tokens == 0
            || chunk_count == 0
            || total_tokens > u32::MAX as usize
            || chunk_count > u32::MAX as usize
            || sequence_count > u32::MAX as usize
            || [
                query.len(),
                key.len(),
                value.len(),
                w.len(),
                u.len(),
                value_new.len(),
                output.len(),
            ]
            .into_iter()
            .any(|len| len < vectors)
            || gate.len() < token_heads
            || beta.len() < token_heads
            || gate_cumsum.len() < token_heads
            || a.len() < a_values
            || a_inverse.len() < a_values
            || state.len() < state_values
            || h.len() < h_values
            || cu_seqlens.len() < sequence_count + 1
            || chunk_offsets.len() < sequence_count + 1
            || chunk_indices.len() < chunk_count * 2
        {
            return Err(Error::Shape {
                label: "Qwen3.6 chunked GDN",
                expected: format!(
                    "sequences={sequence_count} tokens={total_tokens} chunks={chunk_count} with complete workspaces"
                ),
                actual: format!(
                    "q={} k={} v={} gate={} beta={} state={} chunks={}",
                    query.len(),
                    key.len(),
                    value.len(),
                    gate.len(),
                    beta.len(),
                    state.len(),
                    chunk_indices.len() / 2,
                ),
            });
        }
        Ok(())
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn cumsum_on_stream(
        &self,
        gate: &DeviceBuffer<u16>,
        gate_cumsum: &mut DeviceBuffer<f32>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        total_tokens: usize,
        chunk_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_cuda(
                "infer_qwen36_gdn_chunk_cumsum_on_stream",
                ffi::infer_qwen36_gdn_chunk_cumsum_on_stream(
                    gate.as_const_ptr().cast(),
                    gate_cumsum.as_mut_ptr().cast(),
                    cu_seqlens.as_const_ptr().cast(),
                    chunk_indices.as_const_ptr().cast(),
                    total_tokens as u32,
                    chunk_count as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn kkt_on_stream(
        &self,
        key: &DeviceBuffer<u16>,
        beta: &DeviceBuffer<u16>,
        gate_cumsum: &DeviceBuffer<f32>,
        a: &mut DeviceBuffer<f32>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        total_tokens: usize,
        chunk_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_cuda(
                "infer_qwen36_gdn_chunk_kkt_on_stream",
                ffi::infer_qwen36_gdn_chunk_kkt_on_stream(
                    key.as_const_ptr().cast(),
                    beta.as_const_ptr().cast(),
                    gate_cumsum.as_const_ptr().cast(),
                    a.as_mut_ptr().cast(),
                    cu_seqlens.as_const_ptr().cast(),
                    chunk_indices.as_const_ptr().cast(),
                    total_tokens as u32,
                    chunk_count as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn solve_on_stream(
        &self,
        a: &mut DeviceBuffer<f32>,
        a_inverse: &mut DeviceBuffer<u16>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        total_tokens: usize,
        chunk_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_cuda(
                "infer_qwen36_gdn_chunk_solve_on_stream",
                ffi::infer_qwen36_gdn_chunk_solve_on_stream(
                    a.as_mut_ptr().cast(),
                    a_inverse.as_mut_ptr().cast(),
                    cu_seqlens.as_const_ptr().cast(),
                    chunk_indices.as_const_ptr().cast(),
                    total_tokens as u32,
                    chunk_count as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn wu_on_stream(
        &self,
        key: &DeviceBuffer<u16>,
        value: &DeviceBuffer<u16>,
        a_inverse: &DeviceBuffer<u16>,
        gate_cumsum: &DeviceBuffer<f32>,
        w: &mut DeviceBuffer<u16>,
        u: &mut DeviceBuffer<u16>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        total_tokens: usize,
        chunk_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_cuda(
                "infer_qwen36_gdn_chunk_wu_on_stream",
                ffi::infer_qwen36_gdn_chunk_wu_on_stream(
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    a_inverse.as_const_ptr().cast(),
                    gate_cumsum.as_const_ptr().cast(),
                    w.as_mut_ptr().cast(),
                    u.as_mut_ptr().cast(),
                    cu_seqlens.as_const_ptr().cast(),
                    chunk_indices.as_const_ptr().cast(),
                    total_tokens as u32,
                    chunk_count as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn h_on_stream(
        &self,
        key: &DeviceBuffer<u16>,
        u: &DeviceBuffer<u16>,
        w: &DeviceBuffer<u16>,
        value_new: &mut DeviceBuffer<u16>,
        gate_cumsum: &DeviceBuffer<f32>,
        h: &mut DeviceBuffer<u16>,
        state: &mut DeviceBuffer<f32>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_offsets: &DeviceBuffer<i64>,
        sequence_count: usize,
        total_tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_cuda(
                "infer_qwen36_gdn_chunk_h_on_stream",
                ffi::infer_qwen36_gdn_chunk_h_on_stream(
                    key.as_const_ptr().cast(),
                    u.as_const_ptr().cast(),
                    w.as_const_ptr().cast(),
                    value_new.as_mut_ptr().cast(),
                    gate_cumsum.as_const_ptr().cast(),
                    h.as_mut_ptr().cast(),
                    state.as_mut_ptr().cast(),
                    cu_seqlens.as_const_ptr().cast(),
                    chunk_offsets.as_const_ptr().cast(),
                    sequence_count as u32,
                    total_tokens as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn output_on_stream(
        &self,
        query: &DeviceBuffer<u16>,
        key: &DeviceBuffer<u16>,
        value_new: &DeviceBuffer<u16>,
        h: &DeviceBuffer<u16>,
        gate_cumsum: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<u16>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        total_tokens: usize,
        chunk_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_cuda(
                "infer_qwen36_gdn_chunk_output_on_stream",
                ffi::infer_qwen36_gdn_chunk_output_on_stream(
                    query.as_const_ptr().cast(),
                    key.as_const_ptr().cast(),
                    value_new.as_const_ptr().cast(),
                    h.as_const_ptr().cast(),
                    gate_cumsum.as_const_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    cu_seqlens.as_const_ptr().cast(),
                    chunk_indices.as_const_ptr().cast(),
                    total_tokens as u32,
                    chunk_count as u32,
                    (HEAD_DIM as f32).sqrt().recip(),
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Runs chunked GDN over packed ragged sequences and updates `state` in place.
    #[allow(clippy::too_many_arguments)]
    pub fn run_on_stream(
        &self,
        query: &DeviceBuffer<u16>,
        key: &DeviceBuffer<u16>,
        value: &DeviceBuffer<u16>,
        gate: &DeviceBuffer<u16>,
        beta: &DeviceBuffer<u16>,
        state: &mut DeviceBuffer<f32>,
        cu_seqlens: &DeviceBuffer<i32>,
        chunk_indices: &DeviceBuffer<i32>,
        chunk_offsets: &DeviceBuffer<i64>,
        gate_cumsum: &mut DeviceBuffer<f32>,
        a: &mut DeviceBuffer<f32>,
        a_inverse: &mut DeviceBuffer<u16>,
        w: &mut DeviceBuffer<u16>,
        u: &mut DeviceBuffer<u16>,
        h: &mut DeviceBuffer<u16>,
        value_new: &mut DeviceBuffer<u16>,
        output: &mut DeviceBuffer<u16>,
        sequence_count: usize,
        total_tokens: usize,
        chunk_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        Self::validate(
            query,
            key,
            value,
            gate,
            beta,
            state,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            gate_cumsum,
            a,
            a_inverse,
            w,
            u,
            h,
            value_new,
            output,
            sequence_count,
            total_tokens,
            chunk_count,
        )?;
        self.cumsum_on_stream(
            gate,
            gate_cumsum,
            cu_seqlens,
            chunk_indices,
            total_tokens,
            chunk_count,
            stream,
        )?;
        self.kkt_on_stream(
            key,
            beta,
            gate_cumsum,
            a,
            cu_seqlens,
            chunk_indices,
            total_tokens,
            chunk_count,
            stream,
        )?;
        self.solve_on_stream(
            a,
            a_inverse,
            cu_seqlens,
            chunk_indices,
            total_tokens,
            chunk_count,
            stream,
        )?;
        self.wu_on_stream(
            key,
            value,
            a_inverse,
            gate_cumsum,
            w,
            u,
            cu_seqlens,
            chunk_indices,
            total_tokens,
            chunk_count,
            stream,
        )?;
        self.h_on_stream(
            key,
            u,
            w,
            value_new,
            gate_cumsum,
            h,
            state,
            cu_seqlens,
            chunk_offsets,
            sequence_count,
            total_tokens,
            stream,
        )?;
        self.output_on_stream(
            query,
            key,
            value_new,
            h,
            gate_cumsum,
            output,
            cu_seqlens,
            chunk_indices,
            total_tokens,
            chunk_count,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{bf16_to_f32, f32_to_bf16};
    use crate::kernels::qwen36_gdn_reference::{
        chunk_output, gate_prefix_sum, propagate_chunk_state, recurrent_reference,
        solve_wy_transform, strict_lower_key_gram, transformed_w_u,
    };

    const TOKENS: usize = CHUNK_TOKENS;
    const VECTORS: usize = TOKENS * HEADS * HEAD_DIM;
    const SCALARS: usize = TOKENS * HEADS;
    const TRIANGLE: usize = SCALARS * CHUNK_TOKENS;
    const STATE: usize = HEADS * HEAD_DIM * HEAD_DIM;

    struct Fixture {
        kernels: Qwen36ChunkedGdn,
        query_host: Vec<f32>,
        key_host: Vec<f32>,
        value_host: Vec<f32>,
        gate_host: Vec<f32>,
        beta_host: Vec<f32>,
        state_host: Vec<f32>,
        query: DeviceBuffer<u16>,
        key: DeviceBuffer<u16>,
        value: DeviceBuffer<u16>,
        gate: DeviceBuffer<u16>,
        beta: DeviceBuffer<u16>,
        state: DeviceBuffer<f32>,
        cu_seqlens: DeviceBuffer<i32>,
        chunk_indices: DeviceBuffer<i32>,
        chunk_offsets: DeviceBuffer<i64>,
        gate_cumsum: DeviceBuffer<f32>,
        a: DeviceBuffer<f32>,
        a_inverse: DeviceBuffer<u16>,
        w: DeviceBuffer<u16>,
        u: DeviceBuffer<u16>,
        h: DeviceBuffer<u16>,
        value_new: DeviceBuffer<u16>,
        output: DeviceBuffer<u16>,
        stream: CudaStream,
    }

    impl Fixture {
        fn new() -> Self {
            let make_vectors = |period: usize, centre: f32, scale: f32| {
                (0..VECTORS)
                    .map(|index| {
                        let feature = index % HEAD_DIM;
                        let token = index / (HEADS * HEAD_DIM);
                        let value = ((feature * 7 + token * 11) % period) as f32 - centre;
                        f32_to_bf16(value * scale)
                    })
                    .collect::<Vec<_>>()
            };
            let query_bf16 = make_vectors(29, 14.0, 1.0 / 128.0);
            let key_bf16 = make_vectors(31, 15.0, 1.0 / 256.0);
            let value_bf16 = make_vectors(37, 18.0, 1.0 / 64.0);
            let gate_bf16 = (0..SCALARS)
                .map(|index| f32_to_bf16(-(((index / HEADS) % 4 + 1) as f32) / 128.0))
                .collect::<Vec<_>>();
            let beta_bf16 = (0..SCALARS)
                .map(|index| f32_to_bf16(0.25 + ((index / HEADS) % 5) as f32 / 64.0))
                .collect::<Vec<_>>();
            let state_host = (0..STATE)
                .map(|index| bf16_to_f32(f32_to_bf16(((index % 23) as f32 - 11.0) / 1024.0)))
                .collect::<Vec<_>>();
            Self {
                kernels: Qwen36ChunkedGdn::new().expect("native GDN launcher"),
                query_host: query_bf16.iter().copied().map(bf16_to_f32).collect(),
                key_host: key_bf16.iter().copied().map(bf16_to_f32).collect(),
                value_host: value_bf16.iter().copied().map(bf16_to_f32).collect(),
                gate_host: gate_bf16.iter().copied().map(bf16_to_f32).collect(),
                beta_host: beta_bf16.iter().copied().map(bf16_to_f32).collect(),
                state_host: state_host.clone(),
                query: DeviceBuffer::from_host(&query_bf16).expect("query upload"),
                key: DeviceBuffer::from_host(&key_bf16).expect("key upload"),
                value: DeviceBuffer::from_host(&value_bf16).expect("value upload"),
                gate: DeviceBuffer::from_host(&gate_bf16).expect("gate upload"),
                beta: DeviceBuffer::from_host(&beta_bf16).expect("beta upload"),
                state: DeviceBuffer::from_host(&state_host).expect("state upload"),
                cu_seqlens: DeviceBuffer::from_host(&[0, TOKENS as i32])
                    .expect("sequence metadata upload"),
                chunk_indices: DeviceBuffer::from_host(&[0, 0]).expect("chunk metadata upload"),
                chunk_offsets: DeviceBuffer::from_host(&[0, 1]).expect("chunk offset upload"),
                gate_cumsum: DeviceBuffer::zeroed(SCALARS).expect("cumsum allocation"),
                a: DeviceBuffer::zeroed(TRIANGLE).expect("A allocation"),
                a_inverse: DeviceBuffer::zeroed(TRIANGLE).expect("inverse allocation"),
                w: DeviceBuffer::zeroed(VECTORS).expect("W allocation"),
                u: DeviceBuffer::zeroed(VECTORS).expect("U allocation"),
                h: DeviceBuffer::zeroed(STATE).expect("H allocation"),
                value_new: DeviceBuffer::zeroed(VECTORS).expect("value-new allocation"),
                output: DeviceBuffer::zeroed(VECTORS).expect("output allocation"),
                stream: CudaStream::new_blocking().expect("test stream"),
            }
        }

        fn run_cumsum(&mut self) {
            self.kernels
                .cumsum_on_stream(
                    &self.gate,
                    &mut self.gate_cumsum,
                    &self.cu_seqlens,
                    &self.chunk_indices,
                    TOKENS,
                    1,
                    &self.stream,
                )
                .expect("cumsum enqueue");
        }

        fn run_kkt(&mut self) {
            self.run_cumsum();
            self.kernels
                .kkt_on_stream(
                    &self.key,
                    &self.beta,
                    &self.gate_cumsum,
                    &mut self.a,
                    &self.cu_seqlens,
                    &self.chunk_indices,
                    TOKENS,
                    1,
                    &self.stream,
                )
                .expect("KKT enqueue");
        }

        fn run_solve(&mut self) {
            self.run_kkt();
            self.kernels
                .solve_on_stream(
                    &mut self.a,
                    &mut self.a_inverse,
                    &self.cu_seqlens,
                    &self.chunk_indices,
                    TOKENS,
                    1,
                    &self.stream,
                )
                .expect("solve enqueue");
        }

        fn run_wu(&mut self) {
            self.run_solve();
            self.kernels
                .wu_on_stream(
                    &self.key,
                    &self.value,
                    &self.a_inverse,
                    &self.gate_cumsum,
                    &mut self.w,
                    &mut self.u,
                    &self.cu_seqlens,
                    &self.chunk_indices,
                    TOKENS,
                    1,
                    &self.stream,
                )
                .expect("W/U enqueue");
        }

        fn run_h(&mut self) {
            self.run_wu();
            self.kernels
                .h_on_stream(
                    &self.key,
                    &self.u,
                    &self.w,
                    &mut self.value_new,
                    &self.gate_cumsum,
                    &mut self.h,
                    &mut self.state,
                    &self.cu_seqlens,
                    &self.chunk_offsets,
                    1,
                    TOKENS,
                    &self.stream,
                )
                .expect("H enqueue");
        }

        fn head_vectors(values: &[f32]) -> Vec<f32> {
            (0..TOKENS)
                .flat_map(|token| {
                    (0..HEAD_DIM).map(move |feature| values[vector_index_host(token, 0, feature)])
                })
                .collect()
        }

        fn head_scalars(values: &[f32]) -> Vec<f32> {
            (0..TOKENS).map(|token| values[token * HEADS]).collect()
        }

        fn head_triangle_f32(values: &[f32]) -> Vec<f32> {
            (0..TOKENS)
                .flat_map(|token| {
                    (0..CHUNK_TOKENS).map(move |col| values[(token * HEADS) * CHUNK_TOKENS + col])
                })
                .collect()
        }

        fn head_triangle_bf16(values: &[u16]) -> Vec<f32> {
            (0..TOKENS)
                .flat_map(|token| {
                    (0..CHUNK_TOKENS)
                        .map(move |col| bf16_to_f32(values[(token * HEADS) * CHUNK_TOKENS + col]))
                })
                .collect()
        }

        fn head_vectors_bf16(values: &[u16]) -> Vec<f32> {
            (0..TOKENS)
                .flat_map(|token| {
                    (0..HEAD_DIM).map(move |feature| {
                        bf16_to_f32(values[vector_index_host(token, 0, feature)])
                    })
                })
                .collect()
        }
    }

    fn vector_index_host(token: usize, head: usize, feature: usize) -> usize {
        (token * HEADS + head) * HEAD_DIM + feature
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let (index, error) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("non-empty comparison");
        assert!(
            error <= tolerance,
            "{label} mismatch at {index}: actual={} expected={} error={error} tolerance={tolerance}",
            actual[index],
            expected[index]
        );
    }

    #[allow(clippy::type_complexity)]
    fn references(fixture: &Fixture) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let key = Fixture::head_vectors(&fixture.key_host);
        let value = Fixture::head_vectors(&fixture.value_host);
        let gate = Fixture::head_scalars(&fixture.gate_host);
        let beta = Fixture::head_scalars(&fixture.beta_host);
        let prefix = gate_prefix_sum(&gate);
        let lower = strict_lower_key_gram(&key, &beta, &prefix, HEAD_DIM);
        let transform = solve_wy_transform(&lower, &beta);
        let (w, u) = transformed_w_u(&transform, &key, &value, &prefix, HEAD_DIM, HEAD_DIM);
        (prefix, lower, transform, w, u)
    }

    #[test]
    fn native_cumsum_matches_cpu_stage() {
        let mut fixture = Fixture::new();
        fixture.run_cumsum();
        let actual = fixture
            .gate_cumsum
            .copy_to_host(&fixture.stream)
            .expect("cumsum download");
        let actual = Fixture::head_scalars(&actual);
        let expected = gate_prefix_sum(&Fixture::head_scalars(&fixture.gate_host));
        assert_close("cumsum", &actual, &expected, 1.0e-6);
    }

    #[test]
    fn native_kkt_matches_cpu_stage() {
        let mut fixture = Fixture::new();
        fixture.run_kkt();
        let actual = fixture.a.copy_to_host(&fixture.stream).expect("A download");
        let actual = Fixture::head_triangle_f32(&actual);
        let (_, mut expected, _, _, _) = references(&fixture);
        let beta = Fixture::head_scalars(&fixture.beta_host);
        for row in 0..TOKENS {
            expected[row * TOKENS + row] = beta[row];
        }
        assert_close("KKT", &actual, &expected, 2.0e-4);
    }

    #[test]
    fn native_solve_matches_cpu_stage() {
        let mut fixture = Fixture::new();
        fixture.run_solve();
        let actual = fixture
            .a_inverse
            .copy_to_host(&fixture.stream)
            .expect("inverse download");
        let actual = Fixture::head_triangle_bf16(&actual);
        let (_, _, expected, _, _) = references(&fixture);
        assert_close("WY solve", &actual, &expected, 2.0e-3);
    }

    #[test]
    fn native_wu_matches_cpu_stage() {
        let mut fixture = Fixture::new();
        fixture.run_wu();
        let actual_w = fixture.w.copy_to_host(&fixture.stream).expect("W download");
        let actual_u = fixture.u.copy_to_host(&fixture.stream).expect("U download");
        let actual_w = Fixture::head_vectors_bf16(&actual_w);
        let actual_u = Fixture::head_vectors_bf16(&actual_u);
        let (_, _, _, expected_w, expected_u) = references(&fixture);
        assert_close("W", &actual_w, &expected_w, 4.0e-3);
        assert_close("U", &actual_u, &expected_u, 4.0e-3);
    }

    #[test]
    fn native_h_matches_cpu_stage() {
        let mut fixture = Fixture::new();
        fixture.run_h();
        let actual_u = fixture.u.copy_to_host(&fixture.stream).expect("U download");
        let actual_value_new = fixture
            .value_new
            .copy_to_host(&fixture.stream)
            .expect("value-new download");
        let actual_state = fixture
            .state
            .copy_to_host(&fixture.stream)
            .expect("state download");
        let actual_u = Fixture::head_vectors_bf16(&actual_u);
        let actual_value_new = Fixture::head_vectors_bf16(&actual_value_new);
        let (_, _, _, expected_w, _) = references(&fixture);
        let key = Fixture::head_vectors(&fixture.key_host);
        let prefix = gate_prefix_sum(&Fixture::head_scalars(&fixture.gate_host));
        let (expected_value_new, expected_state) = propagate_chunk_state(
            &key,
            &expected_w,
            &actual_u,
            &prefix,
            &fixture.state_host[..HEAD_DIM * HEAD_DIM],
            HEAD_DIM,
            HEAD_DIM,
        );
        assert_close("value new", &actual_value_new, &expected_value_new, 6.0e-3);
        assert_close(
            "state",
            &actual_state[..HEAD_DIM * HEAD_DIM],
            &expected_state,
            8.0e-3,
        );
    }

    #[test]
    fn native_output_matches_cpu_stage() {
        let mut fixture = Fixture::new();
        fixture.run_h();
        let actual_value_new = fixture
            .value_new
            .copy_to_host(&fixture.stream)
            .expect("value-new download");
        let actual_value_new = Fixture::head_vectors_bf16(&actual_value_new);
        fixture
            .kernels
            .output_on_stream(
                &fixture.query,
                &fixture.key,
                &fixture.value_new,
                &fixture.h,
                &fixture.gate_cumsum,
                &mut fixture.output,
                &fixture.cu_seqlens,
                &fixture.chunk_indices,
                TOKENS,
                1,
                &fixture.stream,
            )
            .expect("output enqueue");
        let actual = fixture
            .output
            .copy_to_host(&fixture.stream)
            .expect("output download");
        let actual = Fixture::head_vectors_bf16(&actual);
        let query = Fixture::head_vectors(&fixture.query_host);
        let key = Fixture::head_vectors(&fixture.key_host);
        let prefix = gate_prefix_sum(&Fixture::head_scalars(&fixture.gate_host));
        let expected = chunk_output(
            &query,
            &key,
            &actual_value_new,
            &prefix,
            &fixture.state_host[..HEAD_DIM * HEAD_DIM],
            HEAD_DIM,
            HEAD_DIM,
        );
        assert_close("output", &actual, &expected, 8.0e-3);
    }

    #[test]
    fn native_ragged_chunks_match_recurrent_reference() {
        let lengths = [65usize, 17];
        let total_tokens = lengths.iter().sum::<usize>();
        let vectors = total_tokens * HEADS * HEAD_DIM;
        let scalars = total_tokens * HEADS;
        let triangle = scalars * CHUNK_TOKENS;
        let chunks = 3usize;
        let make_vectors = |period: usize, centre: f32, scale: f32| {
            (0..vectors)
                .map(|index| {
                    let feature = index % HEAD_DIM;
                    let token = index / (HEADS * HEAD_DIM);
                    f32_to_bf16((((feature * 5 + token * 13) % period) as f32 - centre) * scale)
                })
                .collect::<Vec<_>>()
        };
        let query_bf16 = make_vectors(29, 14.0, 1.0 / 128.0);
        let key_bf16 = make_vectors(31, 15.0, 1.0 / 256.0);
        let value_bf16 = make_vectors(37, 18.0, 1.0 / 64.0);
        let gate_bf16 = (0..scalars)
            .map(|index| f32_to_bf16(-(((index / HEADS) % 4 + 1) as f32) / 128.0))
            .collect::<Vec<_>>();
        let beta_bf16 = (0..scalars)
            .map(|index| f32_to_bf16(0.25 + ((index / HEADS) % 5) as f32 / 64.0))
            .collect::<Vec<_>>();
        let state_host = (0..lengths.len() * STATE)
            .map(|index| bf16_to_f32(f32_to_bf16(((index % 23) as f32 - 11.0) / 1024.0)))
            .collect::<Vec<_>>();
        let query_host = query_bf16
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let key_host = key_bf16
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let value_host = value_bf16
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let gate_host = gate_bf16
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let beta_host = beta_bf16
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();

        let query = DeviceBuffer::from_host(&query_bf16).expect("query upload");
        let key = DeviceBuffer::from_host(&key_bf16).expect("key upload");
        let value = DeviceBuffer::from_host(&value_bf16).expect("value upload");
        let gate = DeviceBuffer::from_host(&gate_bf16).expect("gate upload");
        let beta = DeviceBuffer::from_host(&beta_bf16).expect("beta upload");
        let mut state = DeviceBuffer::from_host(&state_host).expect("state upload");
        let cu_seqlens = DeviceBuffer::from_host(&[0, 65, 82]).expect("sequence metadata upload");
        let chunk_indices =
            DeviceBuffer::from_host(&[0, 0, 0, 1, 1, 0]).expect("chunk metadata upload");
        let chunk_offsets = DeviceBuffer::from_host(&[0i64, 2, 3]).expect("chunk offset upload");
        let mut gate_cumsum = DeviceBuffer::zeroed(scalars).expect("cumsum allocation");
        let mut a = DeviceBuffer::zeroed(triangle).expect("A allocation");
        let mut a_inverse = DeviceBuffer::zeroed(triangle).expect("inverse allocation");
        let mut w = DeviceBuffer::zeroed(vectors).expect("W allocation");
        let mut u = DeviceBuffer::zeroed(vectors).expect("U allocation");
        let mut h = DeviceBuffer::zeroed(chunks * STATE).expect("H allocation");
        let mut value_new = DeviceBuffer::zeroed(vectors).expect("value-new allocation");
        let mut output = DeviceBuffer::zeroed(vectors).expect("output allocation");
        let stream = CudaStream::new_blocking().expect("test stream");
        Qwen36ChunkedGdn::new()
            .expect("native GDN launcher")
            .run_on_stream(
                &query,
                &key,
                &value,
                &gate,
                &beta,
                &mut state,
                &cu_seqlens,
                &chunk_indices,
                &chunk_offsets,
                &mut gate_cumsum,
                &mut a,
                &mut a_inverse,
                &mut w,
                &mut u,
                &mut h,
                &mut value_new,
                &mut output,
                lengths.len(),
                total_tokens,
                chunks,
                &stream,
            )
            .expect("ragged native GDN enqueue");
        let actual_output = output
            .copy_to_host(&stream)
            .expect("output download")
            .into_vec();
        let actual_state = state
            .copy_to_host(&stream)
            .expect("state download")
            .into_vec();

        let mut token_offset = 0usize;
        for (sequence, &length) in lengths.iter().enumerate() {
            let extract_vectors = |values: &[f32]| {
                (0..length)
                    .flat_map(|token| {
                        (0..HEAD_DIM).map(move |feature| {
                            values[vector_index_host(token_offset + token, 0, feature)]
                        })
                    })
                    .collect::<Vec<_>>()
            };
            let q = extract_vectors(&query_host);
            let k = extract_vectors(&key_host);
            let v = extract_vectors(&value_host);
            let gate = (0..length)
                .map(|token| gate_host[(token_offset + token) * HEADS])
                .collect::<Vec<_>>();
            let beta = (0..length)
                .map(|token| beta_host[(token_offset + token) * HEADS])
                .collect::<Vec<_>>();
            let state_start = sequence * STATE;
            let (expected_output, expected_state) = recurrent_reference(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                &state_host[state_start..state_start + HEAD_DIM * HEAD_DIM],
                HEAD_DIM,
                HEAD_DIM,
            );
            let output_values = &actual_output;
            let output_token_offset = token_offset;
            let output_for_head = (0..length)
                .flat_map(|token| {
                    (0..HEAD_DIM).map(move |feature| {
                        bf16_to_f32(
                            output_values
                                [vector_index_host(output_token_offset + token, 0, feature)],
                        )
                    })
                })
                .collect::<Vec<_>>();
            assert_close(
                &format!("ragged output sequence {sequence}"),
                &output_for_head,
                &expected_output,
                2.0e-2,
            );
            assert_close(
                &format!("ragged state sequence {sequence}"),
                &actual_state[state_start..state_start + HEAD_DIM * HEAD_DIM],
                &expected_state,
                2.0e-2,
            );
            token_offset += length;
        }
    }
}
