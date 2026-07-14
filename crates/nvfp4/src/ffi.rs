//! Minimal CUDA/cuBLASLt FFI bindings used by the crate.
//!
//! These bindings are hand-written and private to the crate while the API is still being shaped.
//! Public users should go through the safe-ish owner types exported from `lib.rs`.

use std::ffi::c_void;
use std::os::raw::{c_char, c_double};

#[allow(non_camel_case_types)]
pub(crate) type cudaError_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type cudaStream_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cudaEvent_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cudaGraph_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cudaGraphExec_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cudaMemcpyKind = i32;
#[allow(non_camel_case_types)]
pub(crate) type cudaStreamCaptureMode = i32;

pub(crate) const CUDA_SUCCESS: cudaError_t = 0;
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: cudaMemcpyKind = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: cudaMemcpyKind = 2;
pub(crate) const CUDA_STREAM_NON_BLOCKING: u32 = 1;
pub(crate) const CUDA_STREAM_CAPTURE_MODE_RELAXED: cudaStreamCaptureMode = 2;
pub(crate) const CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;

#[allow(non_camel_case_types)]
pub(crate) type cublasStatus_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cublasLtMatmulPreference_t = *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type cudaDataType_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type cublasComputeType_t = i32;

pub(crate) const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub(crate) const CUBLAS_OP_N: i32 = 0;
pub(crate) const CUBLAS_OP_T: i32 = 1;

#[allow(dead_code)]
pub(crate) const CUDA_R_16F: cudaDataType_t = 2;
pub(crate) const CUDA_R_16BF: cudaDataType_t = 14;
pub(crate) const CUDA_R_32F: cudaDataType_t = 0;
#[allow(dead_code)]
pub(crate) const CUDA_R_8F_E4M3: cudaDataType_t = 28;
pub(crate) const CUDA_R_4F_E2M1: cudaDataType_t = 33;
pub(crate) const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;

pub(crate) const CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3: i32 = 1;

pub(crate) const CUBLASLT_MATMUL_DESC_TRANSA: i32 = 3;
pub(crate) const CUBLASLT_MATMUL_DESC_TRANSB: i32 = 4;
pub(crate) const CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: i32 = 17;
pub(crate) const CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: i32 = 18;
pub(crate) const CUBLASLT_MATMUL_DESC_A_SCALE_MODE: i32 = 31;
pub(crate) const CUBLASLT_MATMUL_DESC_B_SCALE_MODE: i32 = 32;

pub(crate) const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct cublasLtMatmulAlgo_t {
    pub data: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct cublasLtMatmulHeuristicResult_t {
    pub algo: cublasLtMatmulAlgo_t,
    pub workspace_size: usize,
    pub state: cublasStatus_t,
    pub waves_count: f32,
    pub reserved: [i32; 4],
}

unsafe extern "C" {
    pub(crate) fn cudaGetDevice(device: *mut i32) -> cudaError_t;
    pub(crate) fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> cudaError_t;
    pub(crate) fn infer_cuda_e2m1_rn(value: f32) -> u8;
    pub(crate) fn infer_cuda_e4m3_satfinite(value: f32) -> u8;
    pub(crate) fn infer_gpu_counter_create(
        metric_names: *const *const c_char,
        metric_count: usize,
        out_handle: *mut *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub(crate) fn infer_gpu_counter_begin(
        handle: *mut c_void,
        range_name: *const c_char,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub(crate) fn infer_gpu_counter_end(
        handle: *mut c_void,
        all_passes_submitted: *mut i32,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub(crate) fn infer_gpu_counter_decode(
        handle: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> i32;
    pub(crate) fn infer_gpu_counter_value_count(handle: *mut c_void) -> usize;
    pub(crate) fn infer_gpu_counter_value(
        handle: *mut c_void,
        index: usize,
        name: *mut *const c_char,
        value: *mut c_double,
    ) -> i32;
    pub(crate) fn infer_gpu_counter_destroy(handle: *mut c_void);
    pub(crate) fn infer_cutlass_fp4_gemv_f32_supported(m: u32, k: u32) -> i32;
    pub(crate) fn infer_cutlass_fp4_gemv_f32_on_stream(
        a_values: *const u8,
        a_scales: *const u8,
        b_values: *const u8,
        b_scales: *const u8,
        c: *const f32,
        d: *mut f32,
        m: u32,
        k: u32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_supported(m: u32, k: u32, groups: u32) -> i32;
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_create(
        m: u32,
        k: u32,
        groups: u32,
    ) -> *mut c_void;
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_destroy(plan: *mut c_void);
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_on_stream(
        plan: *mut c_void,
        a_values: *const *const u8,
        a_scales: *const *const u8,
        b_values: *const *const u8,
        b_scales: *const *const u8,
        c: *const *const f32,
        d: *const *mut f32,
        alpha: f32,
        beta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_indexed_a_on_stream(
        plan: *mut c_void,
        indices: *const u32,
        a_values_table: *const *const u8,
        a_scales_table: *const *const u8,
        table_len: u32,
        b_values: *const u8,
        b_scales: *const u8,
        d: *const *mut f32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_contiguous_b_on_stream(
        plan: *mut c_void,
        a_values_table: *const *const u8,
        a_scales_table: *const *const u8,
        b_values: *const u8,
        b_scales: *const u8,
        d: *mut f32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_mma_zero_probe_on_stream(
        out: *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_mma_one_probe_on_stream(
        out: *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_ldmatrix_probe_on_stream(
        out: *mut u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_mma_tile_frag_on_stream(
        a_native_tile: *const u8,
        b_native_tile: *const u8,
        sfa: u32,
        sfb: u32,
        out: *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_mma_tile_frag_kloop_on_stream(
        a_native_tiles: *const u8,
        b_native_tiles: *const u8,
        sfa: *const u32,
        sfb: *const u32,
        k_tiles: u32,
        out: *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_mma_tile_kloop_on_stream(
        a_native_tiles: *const u8,
        b_native_tiles: *const u8,
        sfa: *const u32,
        sfb: *const u32,
        k_tiles: u32,
        out: *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(dead_code)]
    pub(crate) fn infer_sm12x_native_gemv_on_stream(
        a_native_tiles: *const u8,
        b_native_tiles: *const u8,
        sfa: *const u32,
        sfb: *const u32,
        m_tiles: u32,
        k_tiles: u32,
        out: *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_quantize_fixed_scale_vector_on_stream(
        input: *const f32,
        input_scale: f32,
        k: u32,
        b_native_tiles: *mut u8,
        sfb: *mut u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_quantize_dynamic_vector_on_stream(
        input: *const f32,
        k: u32,
        b_native_tiles: *mut u8,
        sfb: *mut u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_cache_append_on_stream(
        key: *const f32,
        value: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        value_values: *mut u8,
        value_scales: *mut u8,
        key_tail: *mut f32,
        value_tail: *mut f32,
        position: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_attention_on_stream(
        query: *const f32,
        key_values: *const u8,
        key_scales: *const u8,
        key_tail: *const f32,
        value_values: *const u8,
        value_scales: *const u8,
        value_tail: *const f32,
        query_tiles: *mut u8,
        query_scales: *mut u32,
        scores: *mut f32,
        probability_tiles: *mut u8,
        probability_scales: *mut u32,
        output: *mut f32,
        cache_len: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_moe_silu_quantize_slots_on_stream(
        indices: *const u32,
        gate_up_table: *const *const f32,
        b_native_tiles: *mut u8,
        sfb: *mut u32,
        input_scale_table: *const f32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_moe_silu_quantize_slots_reference_on_stream(
        indices: *const u32,
        gate_up_table: *const *const f32,
        b_native_tiles: *mut u8,
        sfb: *mut u32,
        input_scale_table: *const f32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_moe_silu_quantize_bf16_slots_on_stream(
        indices: *const u32,
        gate_up_bf16: *const u16,
        b_native_tiles: *mut u8,
        sfb: *mut u32,
        input_scale_table: *const f32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_indexed_gemv_on_stream(
        indices: *const u32,
        a_native_tiles_table: *const *const u8,
        a_scales_table: *const *const u32,
        table_len: u32,
        b_native_tiles: *const u8,
        sfb: *const u32,
        d: *const *mut f32,
        m_tiles: u32,
        k_tiles: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_indexed_grouped_gemv_on_stream(
        indices: *const u32,
        a_native_tiles_table: *const *const u8,
        a_scales_table: *const *const u32,
        table_len: u32,
        b_native_tiles: *const u8,
        sfb: *const u32,
        d: *const *mut f32,
        m_tiles: u32,
        k_tiles: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rms_norm_f32_on_stream(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rms_norm_rope_neox_f32_indexed_on_stream(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        position: *const u32,
        theta: f32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_silu_mul_f32_on_stream(
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_silu_mul_halves_f32_on_stream(
        gate_up: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fill_f32_on_stream(
        output: *mut f32,
        value: f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_scaled_add_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        scale: f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_split_q_gate_f32_on_stream(
        input: *const f32,
        q: *mut f32,
        gate: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sigmoid_mul_f32_on_stream(
        gate: *const f32,
        input: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sigmoid_scale_scalar_f32_on_stream(
        gate_logit: *const f32,
        input: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_full_attn_prep_f32_on_stream(
        q_full: *const f32,
        k_raw: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        q: *mut f32,
        gate: *mut f32,
        k: *mut f32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_split_qkv_f32_on_stream(
        input: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        q_len: u32,
        kv_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_topk_f32_on_stream(
        logits: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        experts: u32,
        k: u32,
        norm_topk_prob: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gather_nvfp4_grouped_gemv_ptrs_on_stream(
        indices: *const u32,
        a_values_table: *const *const u8,
        a_scales_table: *const *const u8,
        b_values: *const u8,
        b_scales: *const u8,
        c_table: *const *const f32,
        d_table: *const *mut f32,
        groups: u32,
        table_len: u32,
        out_a_values: *mut *const u8,
        out_a_scales: *mut *const u8,
        out_b_values: *mut *const u8,
        out_b_scales: *mut *const u8,
        out_c: *mut *const f32,
        out_d: *mut *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
        indices: *const u32,
        a_values_table: *const *const u8,
        a_scales_table: *const *const u8,
        b_values_table: *const *const u8,
        b_scales_table: *const *const u8,
        c_table: *const *const f32,
        d_table: *const *mut f32,
        groups: u32,
        table_len: u32,
        out_a_values: *mut *const u8,
        out_a_scales: *mut *const u8,
        out_b_values: *mut *const u8,
        out_b_scales: *mut *const u8,
        out_c: *mut *const f32,
        out_d: *mut *mut f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_silu_quantize_slots_nvfp4_on_stream(
        indices: *const u32,
        gate_up_table: *const *const f32,
        packed_table: *const *mut u8,
        scales_table: *const *mut u8,
        input_scale_table: *const f32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
        indices: *const u32,
        gate_up_table: *const *const f32,
        packed_table: *const *mut u8,
        scales_table: *const *mut u8,
        input_scale_table: *const f32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_weighted_accumulate_slots_f32_on_stream(
        indices: *const u32,
        route_weights: *const f32,
        inputs: *const *const f32,
        alpha_table: *const f32,
        output: *mut f32,
        len: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_ffn_finalize_f32_on_stream(
        moe_output: *const f32,
        shared_gate_logit: *const f32,
        shared_output: *const f32,
        residual: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_ffn_finalize_routed_f32_on_stream(
        indices: *const u32,
        route_weights: *const f32,
        routed_outputs: *const *const f32,
        alpha_table: *const f32,
        shared_gate_logit: *const f32,
        shared_output: *const f32,
        residual: *const f32,
        output: *mut f32,
        len: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_neox_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        position: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_neox_f32_indexed_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        position: *const u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_neox_partial_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_imrope_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
        pos_t: u32,
        pos_h: u32,
        pos_w: u32,
        pos_extra: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_imrope_f32_indexed_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
        positions: *const u32,
        position_count: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_neox_sequence_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        start_position: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_add_f32_on_stream(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_row_major_to_col_major_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_col_major_to_row_major_f32(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_copy_row_f32(
        input: *const f32,
        output: *mut f32,
        row: u32,
        cols: u32,
    ) -> cudaError_t;
    pub(crate) fn infer_copy_row_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        row: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_copy_bf16_row_to_f32_indexed_on_stream(
        input: *const u16,
        row: *const u32,
        output: *mut f32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_quantize_nvfp4_col_major_f32(
        input: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        cols: u32,
        input_scale: f32,
    ) -> cudaError_t;
    pub(crate) fn infer_quantize_nvfp4_col_major_f32_on_stream(
        input: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        cols: u32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_quantize_nvfp4_vector_simple_scales_f32_on_stream(
        input: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_cached_gqa_attention_nvfp4_on_stream(
        query: *const f32,
        key_cache: *const u8,
        key_scales: *const u8,
        value_cache: *const u8,
        value_scales: *const u8,
        output: *mut f32,
        cache_len: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_softmax_f32_in_place_on_stream(
        values: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_silu_mul_halves_quantize_nvfp4_col_major_f32_on_stream(
        gate_up: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_single_token_gqa_f32(
        key: *const f32,
        value: *const f32,
        output: *mut f32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> cudaError_t;
    pub(crate) fn infer_append_rows_f32_on_stream(
        src: *const f32,
        dst: *mut f32,
        dst_start_row: u32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_append_rows_f32_indexed_on_stream(
        src: *const f32,
        dst: *mut f32,
        dst_start_row: *const u32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_single_token_gqa_f32_from_cache(
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        position: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_cached_gqa_attention_f32(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        cache_len: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> cudaError_t;
    pub(crate) fn infer_cached_gqa_attention_f32_on_stream(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        cache_len: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_cached_gqa_attention_f32_indexed_on_stream(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        cache_len: *const u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_prefill_gqa_attention_f32(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        tokens: u32,
        start_position: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> cudaError_t;
    pub(crate) fn infer_bf16_linear_argmax_f32_on_stream(
        input: *const f32,
        weight: *const u16,
        logits: *mut f32,
        out_index: *mut u32,
        out_value: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_argmax_f32_on_stream(
        values: *const f32,
        out_index: *mut u32,
        out_value: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(test)]
    pub(crate) fn infer_bf16_linear_logits_f32(
        input: *const f32,
        weight: *const u16,
        logits: *mut f32,
        rows: u32,
        cols: u32,
    ) -> cudaError_t;
    pub(crate) fn infer_bf16_linear_logits_f32_on_stream(
        input: *const f32,
        weight: *const u16,
        logits: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_bf16_linear_pair_logits_f32_on_stream(
        input: *const f32,
        first_weight: *const u16,
        second_weight: *const u16,
        first_logits: *mut f32,
        second_logits: *mut f32,
        first_rows: u32,
        second_rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_lm_head_top1_f32_on_stream(
        input: *const f32,
        weight: *const u16,
        scratch_value: *mut f32,
        scratch_index: *mut u32,
        scratch_len: u32,
        out_index: *mut u32,
        out_value: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_bf16_to_f32_on_stream(
        input: *const u16,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_round_f32_to_bf16_in_place_on_stream(
        values: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_round_f32_to_bf16_on_stream(
        input: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gated_delta_net_128_f32_on_stream(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_f32_configured_on_stream(
        input: *const f32,
        weight: *const u8,
        output: *mut f32,
        rows: u32,
        cols: u32,
        weight_scale: f32,
        threads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_pair_f32_configured_on_stream(
        input: *const f32,
        first_weight: *const u8,
        second_weight: *const u8,
        first_output: *mut f32,
        second_output: *mut f32,
        first_rows: u32,
        second_rows: u32,
        cols: u32,
        first_scale: f32,
        second_scale: f32,
        threads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_triple_f32_configured_on_stream(
        input: *const f32,
        first_weight: *const u8,
        second_weight: *const u8,
        third_weight: *const u8,
        first_output: *mut f32,
        second_output: *mut f32,
        third_output: *mut f32,
        first_rows: u32,
        second_rows: u32,
        third_rows: u32,
        cols: u32,
        first_scale: f32,
        second_scale: f32,
        third_scale: f32,
        threads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_channel_scaled_f32_configured_on_stream(
        input: *const f32,
        weight: *const u8,
        channel_weight_scale: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        threads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_channel_scaled_dynamic_f32_on_stream(
        input: *const f32,
        weight: *const u8,
        channel_weight_scale: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_channel_scaled_precomputed_dynamic_f32_on_stream(
        input: *const f32,
        weight: *const u8,
        channel_weight_scale: *const f32,
        input_scale: *mut f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_channel_scaled_dynamic_quantized_f32_configured_on_stream(
        input: *const f32,
        quantized_input: *mut u8,
        weight: *const u8,
        channel_weight_scale: *const f32,
        input_scale: *mut f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        threads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_moe_grouped_gate_up_f32_on_stream(
        indices: *const u32,
        input: *const u8,
        input_scale: *const f32,
        gate_weights: *const *const u8,
        gate_scales: *const *const f32,
        up_weights: *const *const u8,
        up_scales: *const *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        slots: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_silu_quantize_fp8_slots_f32_on_stream(
        gate_up: *const f32,
        quantized: *mut u8,
        scales: *mut f32,
        rows: u32,
        slots: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_moe_grouped_down_f32_on_stream(
        indices: *const u32,
        inputs: *const u8,
        input_scales: *const f32,
        weights: *const *const u8,
        weight_scales: *const *const f32,
        outputs: *const *mut f32,
        rows: u32,
        cols: u32,
        slots: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_quantize_fp8_e4m3_dynamic_f32_on_stream(
        input: *const f32,
        quantized_input: *mut u8,
        input_scale: *mut f32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_scale_channel_f32_device_scalar_on_stream(
        values: *mut f32,
        channel_scale: *const f32,
        scalar: *const f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_fp8_linear_w8a8_f32_on_stream(
        input: *const f32,
        weight: *const u8,
        output: *mut f32,
        rows: u32,
        cols: u32,
        weight_scale: f32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_marlin_nvfp4_gate_up_supported() -> i32;
    pub(crate) fn infer_marlin_nvfp4_gate_up_on_stream(
        indices: *const u32,
        input: *const f32,
        repacked_weight: *const u32,
        weight_scale: *const u8,
        global_scale: *const f32,
        output: *mut f32,
        input_bf16: *mut u16,
        output_bf16: *mut u16,
        reduce_tmp: *mut f32,
        locks: *mut i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_past_padded: *mut i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_marlin_nvfp4_linear_on_stream(
        input: *const f32,
        repacked_weight: *const u32,
        weight_scale: *const u8,
        global_scale: *const f32,
        output: *mut f32,
        input_bf16: *mut u16,
        output_bf16: *mut u16,
        reduce_tmp: *mut f32,
        locks: *mut i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_past_padded: *mut i32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_quantize_fp8_e4m3_f32_on_stream(
        input: *const f32,
        output: *mut u8,
        len: u32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nvfp4_w4a16_matvec_f32_on_stream(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        output: *mut f32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nvfp4_w4a16_matvec_f32_warp_rows_on_stream(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        output: *mut f32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
        warps_per_block: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nvfp4_w4a16_grouped_matvec_f32_on_stream(
        indices: *const u32,
        input: *const f32,
        packed_weight_table: *const *const u8,
        weight_scale_table: *const *const u8,
        weight_scale_2_table: *const f32,
        output_table: *const *mut f32,
        table_len: u32,
        groups: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nvfp4_w4a16_top1_f32_on_stream(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        scratch_value: *mut f32,
        scratch_index: *mut u32,
        scratch_len: u32,
        out_index: *mut u32,
        out_value: *mut f32,
        out_features: u32,
        in_features: u32,
        weight_scale_2: f32,
        warps_per_block: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_prep_on_stream(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state: *mut f32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_gate_on_stream(
        alpha: *const f32,
        beta_input: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut f32,
        beta: *mut f32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gated_rms_norm_f32_on_stream(
        input: *const f32,
        gate: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub(crate) fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub(crate) fn cudaFree(dev_ptr: *mut c_void) -> cudaError_t;
    pub(crate) fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub(crate) fn cudaDeviceSynchronize() -> cudaError_t;
    pub(crate) fn cudaStreamCreateWithFlags(stream: *mut cudaStream_t, flags: u32) -> cudaError_t;
    pub(crate) fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub(crate) fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub(crate) fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub(crate) fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: u32,
    ) -> cudaError_t;
    pub(crate) fn cudaStreamBeginCapture(
        stream: cudaStream_t,
        mode: cudaStreamCaptureMode,
    ) -> cudaError_t;
    pub(crate) fn cudaStreamEndCapture(
        stream: cudaStream_t,
        graph: *mut cudaGraph_t,
    ) -> cudaError_t;
    pub(crate) fn cudaGraphInstantiate(
        graph_exec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: u64,
    ) -> cudaError_t;
    pub(crate) fn cudaGraphLaunch(graph_exec: cudaGraphExec_t, stream: cudaStream_t)
    -> cudaError_t;
    pub(crate) fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    pub(crate) fn cudaGraphExecDestroy(graph_exec: cudaGraphExec_t) -> cudaError_t;
    pub(crate) fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub(crate) fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub(crate) fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub(crate) fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub(crate) fn cudaEventElapsedTime(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;

    pub(crate) fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> cublasStatus_t;
    pub(crate) fn cublasLtDestroy(handle: cublasLtHandle_t) -> cublasStatus_t;
    pub(crate) fn cublasLtGetVersion() -> usize;

    pub(crate) fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: cublasComputeType_t,
        scale_type: cudaDataType_t,
    ) -> cublasStatus_t;
    pub(crate) fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> cublasStatus_t;
    pub(crate) fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: i32,
        buf: *const c_void,
        size_in_bytes: usize,
    ) -> cublasStatus_t;

    pub(crate) fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        ty: cudaDataType_t,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> cublasStatus_t;
    pub(crate) fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> cublasStatus_t;

    pub(crate) fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        compute_desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        a_desc: cublasLtMatrixLayout_t,
        b: *const c_void,
        b_desc: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        c_desc: cublasLtMatrixLayout_t,
        d: *mut c_void,
        d_desc: cublasLtMatrixLayout_t,
        algo: *const cublasLtMatmulAlgo_t,
        workspace: *mut c_void,
        workspace_size_in_bytes: usize,
        stream: cudaStream_t,
    ) -> cublasStatus_t;

    pub(crate) fn cublasLtMatmulPreferenceCreate(
        pref: *mut cublasLtMatmulPreference_t,
    ) -> cublasStatus_t;
    pub(crate) fn cublasLtMatmulPreferenceDestroy(
        pref: cublasLtMatmulPreference_t,
    ) -> cublasStatus_t;
    pub(crate) fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: i32,
        buf: *const c_void,
        size_in_bytes: usize,
    ) -> cublasStatus_t;
    pub(crate) fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        operation_desc: cublasLtMatmulDesc_t,
        a_desc: cublasLtMatrixLayout_t,
        b_desc: cublasLtMatrixLayout_t,
        c_desc: cublasLtMatrixLayout_t,
        d_desc: cublasLtMatrixLayout_t,
        preference: cublasLtMatmulPreference_t,
        requested_algo_count: i32,
        heuristic_results_array: *mut cublasLtMatmulHeuristicResult_t,
        return_algo_count: *mut i32,
    ) -> cublasStatus_t;
}
