use crate::cuda::{CudaStream, DeviceBuffer};
use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::ptr::null_mut;

const CUMSUM_CUBIN: &[u8] = include_bytes!("../../native/qwen36_gdn_cumsum_sm121.cubin");
const KKT_CUBIN: &[u8] = include_bytes!("../../native/qwen36_gdn_kkt_sm121.cubin");
const SOLVE_CUBIN: &[u8] = include_bytes!("../../native/qwen36_gdn_solve_sm121.cubin");
const WU_CUBIN: &[u8] = include_bytes!("../../native/qwen36_gdn_wu_sm121.cubin");
const H_CUBIN: &[u8] = include_bytes!("../../native/qwen36_gdn_h_sm121.cubin");
const OUTPUT_CUBIN: &[u8] = include_bytes!("../../native/qwen36_gdn_output_sm121.cubin");

const CUMSUM_NAME: &[u8] = b"chunk_local_cumsum_scalar_kernel\0";
const KKT_NAME: &[u8] = b"chunk_scaled_dot_kkt_fwd_kernel\0";
const SOLVE_NAME: &[u8] = b"merge_16x16_to_64x64_inverse_kernel\0";
const WU_NAME: &[u8] = b"recompute_w_u_fwd_kernel\0";
const H_NAME: &[u8] = b"chunk_gated_delta_rule_fwd_kernel_h_blockdim64\0";
const OUTPUT_NAME: &[u8] = b"chunk_fwd_kernel_o\0";

const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const CHUNK_TOKENS: usize = 64;

struct LoadedKernel {
    module: ffi::CUmodule,
    function: ffi::CUfunction,
    threads: u32,
    shared: u32,
}

impl LoadedKernel {
    fn new(cubin: &[u8], name: &[u8], threads: u32, shared: u32) -> Result<Self> {
        let mut module = null_mut();
        let mut function = null_mut();
        unsafe {
            check_driver(
                "cuModuleLoadData(Qwen3.6 chunked GDN)",
                ffi::cuModuleLoadData(&mut module, cubin.as_ptr().cast()),
            )?;
            if let Err(error) = check_driver(
                "cuModuleGetFunction(Qwen3.6 chunked GDN)",
                ffi::cuModuleGetFunction(&mut function, module, name.as_ptr().cast()),
            ) {
                let _ = ffi::cuModuleUnload(module);
                return Err(error);
            }
            if shared > 48 * 1024
                && let Err(error) = check_driver(
                    "cuFuncSetAttribute(Qwen3.6 chunked GDN)",
                    ffi::cuFuncSetAttribute(
                        function,
                        ffi::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared as i32,
                    ),
                )
            {
                let _ = ffi::cuModuleUnload(module);
                return Err(error);
            }
        }
        Ok(Self {
            module,
            function,
            threads,
            shared,
        })
    }

    fn launch(
        &self,
        grid: (u32, u32, u32),
        params: &mut [*mut c_void],
        stream: &CudaStream,
    ) -> Result<()> {
        unsafe {
            check_driver(
                "cuLaunchKernel(Qwen3.6 chunked GDN)",
                ffi::cuLaunchKernel(
                    self.function,
                    grid.0,
                    grid.1,
                    grid.2,
                    self.threads,
                    1,
                    1,
                    self.shared,
                    stream.as_raw(),
                    params.as_mut_ptr(),
                    null_mut(),
                ),
            )
        }
    }
}

impl Drop for LoadedKernel {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cuModuleUnload(self.module);
        }
    }
}

/// Checked-in SM121 Triton kernels for 64-token Qwen3.6 GDN prefill chunks.
pub struct Qwen36ChunkedGdn {
    cumsum: LoadedKernel,
    kkt: LoadedKernel,
    solve: LoadedKernel,
    wu: LoadedKernel,
    h: LoadedKernel,
    output: LoadedKernel,
}

impl Qwen36ChunkedGdn {
    /// Loads the checked-in cubins into the current CUDA context.
    pub fn new() -> Result<Self> {
        Ok(Self {
            cumsum: LoadedKernel::new(CUMSUM_CUBIN, CUMSUM_NAME, 64, 8)?,
            kkt: LoadedKernel::new(KKT_CUBIN, KKT_NAME, 128, 20_480)?,
            solve: LoadedKernel::new(SOLVE_CUBIN, SOLVE_NAME, 64, 10_240)?,
            wu: LoadedKernel::new(WU_CUBIN, WU_NAME, 64, 33_792)?,
            h: LoadedKernel::new(H_CUBIN, H_NAME, 128, 41_220)?,
            output: LoadedKernel::new(OUTPUT_CUBIN, OUTPUT_NAME, 128, 73_728)?,
        })
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
            || total_tokens > i32::MAX as usize
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

        let mut total_tokens_i32 = total_tokens as i32;
        let mut null_global: *mut c_void = null_mut();
        let mut null_profile: *mut c_void = null_mut();

        let mut gate_ptr = gate.as_const_ptr();
        let mut gate_cumsum_ptr = gate_cumsum.as_mut_ptr();
        let mut cu_seqlens_ptr = cu_seqlens.as_const_ptr();
        let mut chunk_indices_ptr = chunk_indices.as_const_ptr();
        self.cumsum.launch(
            (chunk_count as u32, HEADS as u32, 1),
            &mut [
                parameter(&mut gate_ptr),
                parameter(&mut gate_cumsum_ptr),
                parameter(&mut cu_seqlens_ptr),
                parameter(&mut chunk_indices_ptr),
                parameter(&mut total_tokens_i32),
                parameter(&mut null_global),
                parameter(&mut null_profile),
            ],
            stream,
        )?;

        let mut key_ptr = key.as_const_ptr();
        let mut beta_ptr = beta.as_const_ptr();
        let mut a_ptr = a.as_mut_ptr();
        self.kkt.launch(
            (chunk_count as u32, HEADS as u32, 1),
            &mut [
                parameter(&mut key_ptr),
                parameter(&mut beta_ptr),
                parameter(&mut gate_cumsum_ptr),
                parameter(&mut a_ptr),
                parameter(&mut cu_seqlens_ptr),
                parameter(&mut chunk_indices_ptr),
                parameter(&mut total_tokens_i32),
                parameter(&mut null_global),
                parameter(&mut null_profile),
            ],
            stream,
        )?;

        let mut a_inverse_ptr = a_inverse.as_mut_ptr();
        self.solve.launch(
            (chunk_count as u32, HEADS as u32, 1),
            &mut [
                parameter(&mut a_ptr),
                parameter(&mut a_inverse_ptr),
                parameter(&mut cu_seqlens_ptr),
                parameter(&mut chunk_indices_ptr),
                parameter(&mut total_tokens_i32),
                parameter(&mut null_global),
                parameter(&mut null_profile),
            ],
            stream,
        )?;

        let mut value_ptr = value.as_const_ptr();
        let mut w_ptr = w.as_mut_ptr();
        let mut u_ptr = u.as_mut_ptr();
        self.wu.launch(
            (chunk_count as u32, HEADS as u32, 1),
            &mut [
                parameter(&mut key_ptr),
                parameter(&mut value_ptr),
                parameter(&mut beta_ptr),
                parameter(&mut w_ptr),
                parameter(&mut u_ptr),
                parameter(&mut a_inverse_ptr),
                parameter(&mut gate_cumsum_ptr),
                parameter(&mut cu_seqlens_ptr),
                parameter(&mut chunk_indices_ptr),
                parameter(&mut total_tokens_i32),
                parameter(&mut null_global),
                parameter(&mut null_profile),
            ],
            stream,
        )?;

        let mut value_new_ptr = value_new.as_mut_ptr();
        let mut h_ptr = h.as_mut_ptr();
        let mut state_ptr = state.as_mut_ptr();
        let mut chunk_offsets_ptr = chunk_offsets.as_const_ptr();
        self.h.launch(
            (4, (sequence_count * HEADS) as u32, 1),
            &mut [
                parameter(&mut key_ptr),
                parameter(&mut u_ptr),
                parameter(&mut w_ptr),
                parameter(&mut value_new_ptr),
                parameter(&mut gate_cumsum_ptr),
                parameter(&mut h_ptr),
                parameter(&mut state_ptr),
                parameter(&mut state_ptr),
                parameter(&mut cu_seqlens_ptr),
                parameter(&mut chunk_offsets_ptr),
                parameter(&mut total_tokens_i32),
                parameter(&mut null_global),
                parameter(&mut null_profile),
            ],
            stream,
        )?;

        let mut query_ptr = query.as_const_ptr();
        let mut output_ptr = output.as_mut_ptr();
        let mut scale = (HEAD_DIM as f32).sqrt().recip();
        self.output.launch(
            (2, chunk_count as u32, HEADS as u32),
            &mut [
                parameter(&mut query_ptr),
                parameter(&mut key_ptr),
                parameter(&mut value_new_ptr),
                parameter(&mut h_ptr),
                parameter(&mut gate_cumsum_ptr),
                parameter(&mut output_ptr),
                parameter(&mut cu_seqlens_ptr),
                parameter(&mut chunk_indices_ptr),
                parameter(&mut scale),
                parameter(&mut total_tokens_i32),
                parameter(&mut null_global),
                parameter(&mut null_profile),
            ],
            stream,
        )
    }
}

fn parameter<T>(value: &mut T) -> *mut c_void {
    std::ptr::from_mut(value).cast()
}

fn check_driver(call: &'static str, status: ffi::CUresult) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cuda(call, status))
    }
}
