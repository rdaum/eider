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
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: cudaMemcpyKind = 3;
pub(crate) const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;
pub(crate) const CUDA_STREAM_NON_BLOCKING: u32 = 1;
pub(crate) const CUDA_EVENT_DISABLE_TIMING: u32 = 2;
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
pub(crate) const CUDA_R_8I: cudaDataType_t = 3;
pub(crate) const CUDA_R_32I: cudaDataType_t = 10;
#[allow(dead_code)]
pub(crate) const CUDA_R_8F_E4M3: cudaDataType_t = 28;
pub(crate) const CUDA_R_4F_E2M1: cudaDataType_t = 33;
pub(crate) const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;
pub(crate) const CUBLAS_COMPUTE_32I: cublasComputeType_t = 72;

pub(crate) const CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3: i32 = 1;

pub(crate) const CUBLASLT_MATMUL_DESC_TRANSA: i32 = 3;
pub(crate) const CUBLASLT_MATMUL_DESC_TRANSB: i32 = 4;
pub(crate) const CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: i32 = 17;
pub(crate) const CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: i32 = 18;
pub(crate) const CUBLASLT_MATMUL_DESC_A_SCALE_MODE: i32 = 31;
pub(crate) const CUBLASLT_MATMUL_DESC_B_SCALE_MODE: i32 = 32;

pub(crate) const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: i32 = 1;
pub(crate) const CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: i32 = 5;
pub(crate) const CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: i32 = 6;

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
    pub(crate) fn infer_bitnet_quantize_i8_f32_on_stream(
        input: *const f32,
        output: *mut i8,
        dequant_scales: *mut f32,
        batch_rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn cudaSetDevice(device: i32) -> cudaError_t;
    pub(crate) fn infer_bitnet_w2a8_linear_f32_on_stream(
        input: *const i8,
        input_scales: *const f32,
        weight: *const u8,
        weight_scales: *const f32,
        output: *mut f32,
        batch_rows: u32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_bitnet_relu_squared_mul_halves_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        batch_rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_bitnet_scale_i32_f32_on_stream(
        input: *const i32,
        input_scales: *const f32,
        weight_scales: *const f32,
        output: *mut f32,
        batch_rows: u32,
        rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ternary_g64_quantize_i8_f32_on_stream(
        input: *const f32,
        output: *mut i8,
        dequant_scales: *mut f32,
        batch_rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ternary_g64_w2a8_linear_f32_on_stream(
        input: *const i8,
        input_scales: *const f32,
        weight: *const u8,
        weight_scales: *const f32,
        output: *mut f32,
        batch_rows: u32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ternary_g64_lookup_rows_f32_on_stream(
        weight: *const u8,
        weight_scales: *const f32,
        row_indices: *const u32,
        output: *mut f32,
        batch_rows: u32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ternary_g64_expand_bf16_on_stream(
        weight: *const u8,
        weight_scales: *const f32,
        output: *mut u16,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_block_fp8_linear_f32_on_stream(
        input: *const f32,
        weight: *const u8,
        scales: *const u8,
        output: *mut f32,
        batch_rows: u32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_block_fp8_grouped_linear_f32_on_stream(
        input: *const f32,
        weight: *const u8,
        scales: *const u8,
        output: *mut f32,
        batch_rows: u32,
        groups: u32,
        rows_per_group: u32,
        cols_per_group: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_hyper_prepare_f32_on_stream(
        streams: *const f32,
        function: *const f32,
        base: *const f32,
        scale: *const f32,
        post: *mut f32,
        combination: *mut f32,
        collapsed: *mut f32,
        batch_rows: u32,
        hidden: u32,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_hyper_apply_f32_on_stream(
        streams: *const f32,
        sublayer: *const f32,
        post: *const f32,
        combination: *const f32,
        output: *mut f32,
        batch_rows: u32,
        hidden: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_hyper_head_f32_on_stream(
        streams: *const f32,
        function: *const f32,
        base: *const f32,
        scale: *const f32,
        output: *mut f32,
        batch_rows: u32,
        hidden: u32,
        rms_eps: f32,
        hc_eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_rope_interleaved_trailing_f32_on_stream(
        values: *mut f32,
        inv_freq: *const f32,
        positions: *const u32,
        batch_rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        direction: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_attention_f32_on_stream(
        query: *const f32,
        sliding_tables: *const *const f32,
        sliding_lengths: *const u32,
        sliding_starts: *const u32,
        compressed_tables: *const *const f32,
        compressed_lengths: *const u32,
        selected_indices: *const i32,
        sinks: *const f32,
        output: *mut f32,
        batch_rows: u32,
        heads: u32,
        head_dim: u32,
        sliding_capacity: u32,
        selected_count: u32,
        scaling: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_causal_attention_f32_on_stream(
        query: *const f32,
        sliding_tables: *const *const f32,
        sliding_lengths: *const u32,
        sliding_starts: *const u32,
        current_kv: *const f32,
        current_sequence_starts: *const u32,
        query_offsets: *const u32,
        positions: *const u32,
        compressed_tables: *const *const f32,
        compressed_lengths: *const u32,
        selected_indices: *const i32,
        sinks: *const f32,
        output: *mut f32,
        batch_rows: u32,
        current_rows: u32,
        heads: u32,
        head_dim: u32,
        sliding_capacity: u32,
        compression_ratio: u32,
        selected_count: u32,
        scaling: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_indexer_topk_f32_on_stream(
        query: *const f32,
        head_weights: *const f32,
        compressed_tables: *const *const f32,
        compressed_lengths: *const u32,
        positions: *const u32,
        selected_indices: *mut i32,
        batch_rows: u32,
        heads: u32,
        head_dim: u32,
        compression_ratio: u32,
        top_k: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_router_topk_f32_on_stream(
        logits: *const f32,
        bias: *const f32,
        indices: *mut u32,
        weights: *mut f32,
        batch_rows: u32,
        experts: u32,
        top_k: u32,
        routed_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_router_hash_f32_on_stream(
        logits: *const f32,
        token_to_expert: *const i64,
        token_ids: *const u32,
        indices: *mut u32,
        weights: *mut f32,
        batch_rows: u32,
        vocab: u32,
        experts: u32,
        top_k: u32,
        routed_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_compress_windows_f32_on_stream(
        kv: *const f32,
        gate: *const f32,
        position_bias: *const f32,
        prior_kv: *const f32,
        prior_gate: *const f32,
        output: *mut f32,
        windows: u32,
        ratio: u32,
        compressed_width: u32,
        overlapping: bool,
        has_prior: bool,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_store_compression_overlap_f32_on_stream(
        kv: *const f32,
        gate: *const f32,
        position_bias: *const f32,
        overlap_kv: *mut f32,
        overlap_gate: *mut f32,
        window: u32,
        ratio: u32,
        compressed_width: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_arithmetic_positions_u32_on_stream(
        positions: *mut u32,
        len: u32,
        start: u32,
        stride: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_repeat_hyper_streams_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        hidden: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_swiglu_pair_f32_on_stream(
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        rows: u32,
        width: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_swiglu_pair_clamped_f32_on_stream(
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        rows: u32,
        width: u32,
        limit: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_routed_accumulate_f32_on_stream(
        route_output: *const f32,
        route_weights: *const f32,
        output: *mut f32,
        rows: u32,
        routes_per_row: u32,
        width: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_gather_sorted_route_rows_f32_on_stream(
        input: *const f32,
        sorted_routes: *const u32,
        output: *mut f32,
        route_offset: u32,
        routes: u32,
        routes_per_row: u32,
        width: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_deepseek4_routed_accumulate_sorted_f32_on_stream(
        sorted_route_output: *const f32,
        route_to_sorted: *const u32,
        route_weights: *const f32,
        output: *mut f32,
        rows: u32,
        routes_per_row: u32,
        width: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gemma4_local_attention_bf16_on_stream(
        query: *const u16,
        key: *const u16,
        value: *const u16,
        output: *mut u16,
        query_tokens: u32,
        key_tokens: u32,
        start_position: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gemma4_local_attention_compact_on_stream(
        query: *const u16,
        key_values: *const u8,
        key_scales: *const u8,
        value_values: *const u8,
        value_scales: *const u8,
        key_tail: *const f32,
        value_tail: *const f32,
        output: *mut u16,
        query_tokens: u32,
        cache_tokens: u32,
        cache_capacity: u32,
        start_position: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn cudaGetDevice(device: *mut i32) -> cudaError_t;
    pub(crate) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
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
    pub(crate) fn infer_cutlass_fp4_grouped_gemv_f32_indexed_a_tiled_scales_on_stream(
        plan: *mut c_void,
        indices: *const u32,
        a_values_table: *const *const u8,
        a_scales_table: *const *const u8,
        alpha_table: *const f32,
        table_len: u32,
        b_values: *const u8,
        b_scales: *const u8,
        c: *const f32,
        d: *const *mut f32,
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
    pub(crate) fn infer_cutlass_fp4_grouped_gemm_supported(
        m: u32,
        max_n: u32,
        k: u32,
        groups: u32,
    ) -> i32;
    pub(crate) fn infer_cutlass_fp4_grouped_gemm_create(
        m: u32,
        max_n: u32,
        k: u32,
        groups: u32,
    ) -> *mut c_void;
    pub(crate) fn infer_cutlass_fp4_grouped_gemm_destroy(plan: *mut c_void);
    pub(crate) fn infer_cutlass_fp4_grouped_gemm_on_stream(
        plan: *mut c_void,
        a_values: *const *const u8,
        a_scales: *const *const u8,
        b_values: *const *const u8,
        b_scales: *const *const u8,
        output: *const *mut u16,
        alpha: *const *mut f32,
        tokens_per_expert: *const u32,
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
    pub(crate) fn infer_sm12x_mma_sfa_lane_probe_on_stream(
        a_native_tile: *const u8,
        b_native_tile: *const u8,
        sfa_lanes: *const u32,
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
    pub(crate) fn infer_sm12x_quantize_dynamic_vectors_residual2_on_stream(
        input: *const f32,
        rows: u32,
        k: u32,
        primary_tiles: *mut u8,
        primary_scales: *mut u32,
        residual_tiles: *mut u8,
        residual_scales: *mut u32,
        residual2_tiles: *mut u8,
        residual2_scales: *mut u32,
        input_multiplier: f32,
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
    pub(crate) fn infer_sm12x_kv_cache_append_rows_on_stream(
        key: *const f32,
        value: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        value_values: *mut u8,
        value_scales: *mut u8,
        key_tail: *mut f32,
        value_tail: *mut f32,
        key_output: *mut u16,
        value_output: *mut u16,
        output_tokens: u32,
        input_row_offset: u32,
        start_position: u32,
        rows: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_cache_append_indexed_on_stream(
        key: *const f32,
        value: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        value_values: *mut u8,
        value_scales: *mut u8,
        key_tail: *mut f32,
        value_tail: *mut f32,
        position: *const u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_cache_unpack_bf16_on_stream(
        key_values: *const u8,
        key_scales: *const u8,
        value_values: *const u8,
        value_scales: *const u8,
        key_tail: *const f32,
        value_tail: *const f32,
        key_output: *mut u16,
        value_output: *mut u16,
        cache_len: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_cache_copy_aligned_prefix_on_stream(
        source_key_values: *const u8,
        source_key_scales: *const u8,
        source_value_values: *const u8,
        source_value_scales: *const u8,
        destination_key_values: *mut u8,
        destination_key_scales: *mut u8,
        destination_value_values: *mut u8,
        destination_value_scales: *mut u8,
        prefix_tokens: u32,
        source_max_tokens: u32,
        destination_max_tokens: u32,
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
        partial_output: *mut f32,
        output: *mut f32,
        cache_len: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        pv_splits: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_qk_on_stream(
        query: *const f32,
        key_values: *const u8,
        key_scales: *const u8,
        key_tail: *const f32,
        query_tiles: *mut u8,
        query_scales: *mut u32,
        scores: *mut f32,
        cache_len: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_attention_window_on_stream(
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
        partial_output: *mut f32,
        output: *mut f32,
        cache_len: u32,
        window_start: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        pv_splits: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_append_causal_attention_rows_on_stream(
        query: *const f32,
        key: *const f32,
        value: *const f32,
        key_values: *mut u8,
        key_scales: *mut u8,
        value_values: *mut u8,
        value_scales: *mut u8,
        key_tail: *mut f32,
        value_tail: *mut f32,
        query_tiles: *mut u8,
        query_scales: *mut u32,
        scores: *mut f32,
        probability_tiles: *mut u8,
        probability_scales: *mut u32,
        output: *mut f32,
        input_row_offset: u32,
        start_position: u32,
        rows: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window_tokens: u32,
        workspace_rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_attention_rows_window_on_stream(
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
        input_row_offset: u32,
        output_row_offset: u32,
        rows: u32,
        cache_len: u32,
        window_start: u32,
        max_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        workspace_rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_attention_indexed_on_stream(
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
        partial_output: *mut f32,
        output: *mut f32,
        cache_len: *const u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        pv_splits: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_pv_from_probabilities_on_stream(
        probabilities: *const f32,
        value_values: *const u8,
        value_scales: *const u8,
        value_tail: *const f32,
        probability_tiles: *mut u8,
        probability_scales: *mut u32,
        output: *mut f32,
        cache_len: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_kv_pv_from_probabilities_split_on_stream(
        probabilities: *const f32,
        value_values: *const u8,
        value_scales: *const u8,
        value_tail: *const f32,
        probability_tiles: *mut u8,
        probability_scales: *mut u32,
        partial_output: *mut f32,
        output: *mut f32,
        cache_len: u32,
        max_tokens: u32,
        kv_heads: u32,
        head_dim: u32,
        pv_splits: u32,
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
    pub(crate) fn infer_sm12x_moe_silu_quantize_slots_residual_on_stream(
        indices: *const u32,
        gate_up_table: *const *const f32,
        primary_tiles: *mut u8,
        primary_scales: *mut u32,
        residual_tiles: *mut u8,
        residual_scales: *mut u32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        swiglu_limit: f32,
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
    pub(crate) fn infer_sm12x_moe_silu_quantize_bf16_sorted_slots_on_stream(
        indices: *const u32,
        sorted_routes: *const u32,
        sorted_experts: *const u32,
        gate_up_bf16: *const u16,
        b_native_tiles: *mut u8,
        sfb: *mut u32,
        input_scale_table: *const f32,
        gate_up_alpha_table: *const f32,
        rows: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_moe_silu_quantize_bf16_expert_sorted_slots_on_stream(
        sorted_experts: *const u32,
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
    pub(crate) fn infer_sm12x_indexed_grouped_gemv_row_scales_on_stream(
        indices: *const u32,
        a_native_tiles_table: *const *const u8,
        a_row_scales_table: *const *const u32,
        table_len: u32,
        b_native_tiles: *const u8,
        sfb: *const u32,
        d: *const *mut f32,
        m_tiles: u32,
        k_tiles: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_indexed_grouped_gemv_row_scales_residual_on_stream(
        indices: *const u32,
        a_native_tiles_table: *const *const u8,
        a_row_scales_table: *const *const u32,
        table_len: u32,
        b_native_tiles: *const u8,
        sfb: *const u32,
        residual_native_tiles: *const u8,
        residual_sfb: *const u32,
        d: *const *mut f32,
        m_tiles: u32,
        k_tiles: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub(crate) fn infer_sm12x_gemv_row_scales_residual2_batch_on_stream(
        a_native_tiles: *const u8,
        a_row_scales: *const u32,
        b_native_tiles: *const u8,
        sfb: *const u32,
        residual_native_tiles: *const u8,
        residual_sfb: *const u32,
        residual2_native_tiles: *const u8,
        residual2_sfb: *const u32,
        output: *mut f32,
        rows: u32,
        m_tiles: u32,
        k_tiles: u32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sm12x_gemv_row_scales_residual2_splitk_batch_on_stream(
        a_native_tiles: *const u8,
        a_row_scales: *const u32,
        b_native_tiles: *const u8,
        sfb: *const u32,
        residual_native_tiles: *const u8,
        residual_sfb: *const u32,
        residual2_native_tiles: *const u8,
        residual2_sfb: *const u32,
        partials: *mut f32,
        output: *mut f32,
        rows: u32,
        m_tiles: u32,
        k_tiles: u32,
        k_splits: u32,
        alpha: f32,
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
    pub(crate) fn infer_rms_norm_add_f32_on_stream(
        input: *const f32,
        weight: *const f32,
        residual: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rms_norm_add_then_rms_norm_quantize_nvfp4_f32_on_stream(
        input: *const f32,
        input_weight: *const f32,
        residual: *const f32,
        output: *mut f32,
        quant_weight: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        cols: u32,
        input_eps: f32,
        quant_eps: f32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_dual_rms_norm_add_f32_on_stream(
        left: *const f32,
        left_weight: *const f32,
        right: *const f32,
        right_weight: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        left_eps: f32,
        right_eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rms_norm_add_channel_row_scale_f32_on_stream(
        input: *const f32,
        weight: *const f32,
        residual: *const f32,
        channel_scale: *const f32,
        row_scale: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_on_stream(
        left: *const f32,
        left_weight: *const f32,
        right: *const f32,
        right_weight: *const f32,
        final_weight: *const f32,
        residual: *const f32,
        channel_scale: *const f32,
        row_scale: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        left_eps: f32,
        right_eps: f32,
        final_eps: f32,
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
    pub(crate) fn infer_gelu_tanh_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gelu_tanh_mul_f32_on_stream(
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gelu_tanh_mul_halves_f32_on_stream(
        gate_up: *const f32,
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
    pub(crate) fn infer_silu_mul_halves_clamped_f32_on_stream(
        gate_up: *const f32,
        output: *mut f32,
        len: u32,
        limit: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_silu_mul_halves_f32_batch_on_stream(
        gate_up: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_silu_mul_halves_clamped_f32_batch_on_stream(
        gate_up: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        limit: f32,
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
    pub(crate) fn infer_sigmoid_scale_heads_f32_on_stream(
        gate: *const f32,
        input: *const f32,
        output: *mut f32,
        heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_softplus_scale_heads_f32_on_stream(
        gate: *const f32,
        input: *const f32,
        output: *mut f32,
        heads: u32,
        head_dim: u32,
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
    pub(crate) fn infer_qwen36_full_attn_prep_f32_batch_on_stream(
        q_full: *const f32,
        k_raw: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        q: *mut f32,
        gate: *mut f32,
        k: *mut f32,
        batch_size: u32,
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
    pub(crate) fn infer_split_qkv_f32_batch_on_stream(
        input: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        batch_rows: u32,
        q_width: u32,
        kv_width: u32,
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
    pub(crate) fn infer_step37_sigmoid_top8_f32_on_stream(
        logits: *const f32,
        bias: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        experts: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_sigmoid_topk_f32_on_stream(
        logits: *const f32,
        bias: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        experts: u32,
        k: u32,
        groups: u32,
        topk_groups: u32,
        normalize: i32,
        scaling_factor: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_sigmoid_topk_f32_batch_on_stream(
        logits: *const f32,
        bias: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        batch_size: u32,
        experts: u32,
        k: u32,
        groups: u32,
        topk_groups: u32,
        normalize: i32,
        scaling_factor: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_step37_sigmoid_top8_f32_batch_on_stream(
        logits: *const f32,
        bias: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        batch_size: u32,
        experts: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_topk_f32_batch_on_stream(
        logits: *const f32,
        out_indices: *mut u32,
        out_weights: *mut f32,
        batch_size: u32,
        experts: u32,
        k: u32,
        norm_topk_prob: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_sort_routes_on_stream(
        indices: *const u32,
        expert_counts: *mut u32,
        expert_offsets: *mut u32,
        expert_cursors: *mut u32,
        sorted_routes: *mut u32,
        sorted_experts: *mut u32,
        route_to_sorted: *mut u32,
        routes: u32,
        experts: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_quantize_sorted_routes_nvfp4_on_stream(
        input: *const f32,
        sorted_routes: *const u32,
        sorted_experts: *const u32,
        expert_offsets: *const u32,
        packed: *mut u8,
        scales: *mut u8,
        routes: u32,
        routes_per_row: u32,
        in_features: u32,
        scale_stride: u32,
        gather_rows: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_gelu_tanh_mul_quantize_sorted_routes_nvfp4_on_stream(
        gate: *const u16,
        up: *const u16,
        sorted_experts: *const u32,
        expert_offsets: *const u32,
        packed: *mut u8,
        scales: *mut u8,
        routes: u32,
        in_features: u32,
        scale_stride: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_silu_mul_halves_quantize_sorted_routes_nvfp4_on_stream(
        gate_up: *const u16,
        sorted_experts: *const u32,
        expert_offsets: *const u32,
        packed: *mut u8,
        scales: *mut u8,
        routes: u32,
        in_features: u32,
        scale_stride: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_gather_rms_norm_quantize_sorted_routes_nvfp4_on_stream(
        input: *const f32,
        weight: *const f32,
        sorted_routes: *const u32,
        sorted_experts: *const u32,
        expert_offsets: *const u32,
        source_packed: *mut u8,
        source_scales: *mut u8,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        routes: u32,
        routes_per_row: u32,
        in_features: u32,
        scale_stride: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_gather_quantize_sorted_routes_nvfp4_on_stream(
        input: *const f32,
        sorted_routes: *const u32,
        sorted_experts: *const u32,
        expert_offsets: *const u32,
        source_packed: *mut u8,
        source_scales: *mut u8,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        routes: u32,
        routes_per_row: u32,
        in_features: u32,
        scale_stride: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_grouped_pointer_tables_on_stream(
        expert_offsets: *const u32,
        packed: *const u8,
        scales: *const u8,
        output: *mut u16,
        packed_table: *mut *const u8,
        scale_table: *mut *const u8,
        output_table: *mut *mut u16,
        experts: u32,
        in_features: u32,
        out_features: u32,
        scale_stride: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_remap_expert_indices_on_stream(
        expert_indices: *const u32,
        expert_to_slot: *const u32,
        slot_indices: *mut u32,
        expert_offset: u32,
        count: u32,
        experts: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_record_expert_indices_u64_on_stream(
        expert_indices: *const u32,
        counts: *mut u64,
        count: u32,
        experts: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_clear_expert_counts_u64_on_stream(
        counts: *mut u64,
        experts: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gather_indexed_mul_f32_on_stream(
        values: *const f32,
        indices: *const u32,
        multipliers: *const f32,
        output: *mut f32,
        count: u32,
        values_len: u32,
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
    pub(crate) fn infer_moe_silu_slots_f32_on_stream(
        indices: *const u32,
        gate_up_table: *const *const f32,
        output_table: *const *mut f32,
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
    pub(crate) fn infer_moe_weighted_accumulate_slots_f32_batch_on_stream(
        indices: *const u32,
        route_weights: *const f32,
        inputs: *const *const f32,
        alpha_table: *const f32,
        output: *mut f32,
        rows: u32,
        len: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_weighted_accumulate_sorted_slots_f32_batch_on_stream(
        route_to_sorted: *const u32,
        indices: *const u32,
        route_weights: *const f32,
        sorted_inputs: *const *const f32,
        alpha_table: *const f32,
        output: *mut f32,
        rows: u32,
        len: u32,
        groups: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_moe_weighted_accumulate_sorted_bf16_batch_on_stream(
        route_to_sorted: *const u32,
        route_weights: *const f32,
        sorted_inputs: *const u16,
        output: *mut f32,
        rows: u32,
        len: u32,
        routes_per_row: u32,
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
    pub(crate) fn infer_qwen36_ffn_finalize_batch_f32_on_stream(
        routed_output: *const f32,
        shared_gate_logit: *const f32,
        shared_output: *const f32,
        residual: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
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
    pub(crate) fn infer_qwen36_ffn_finalize_routed_batch_f32_on_stream(
        indices: *const u32,
        route_weights: *const f32,
        routed_outputs: *const *const f32,
        alpha_table: *const f32,
        shared_gate_logit: *const f32,
        shared_output: *const f32,
        residual: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        groups_per_row: u32,
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
    pub(crate) fn infer_rope_neox_proportional_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        head_dim: u32,
        rotary_pairs: u32,
        position: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rope_neox_proportional_sequence_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        rotary_pairs: u32,
        input_token_offset: u32,
        start_position: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_dual_rms_norm_rope_neox_proportional_sequence_f32_on_stream(
        q_input: *const f32,
        q_weight: *const f32,
        q_output: *mut f32,
        k_input: *const f32,
        k_weight: *const f32,
        k_output: *mut f32,
        tokens: u32,
        q_heads: u32,
        k_heads: u32,
        head_dim: u32,
        rotary_pairs: u32,
        input_token_offset: u32,
        start_position: u32,
        theta: f32,
        q_eps: f32,
        k_eps: f32,
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
    pub(crate) fn infer_rope_imrope_text_batch_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        positions: *const u32,
        batch_size: u32,
        heads_per_row: u32,
        head_dim: u32,
        rotary_dim: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
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
    pub(crate) fn infer_rope_neox_inv_freq_sequence_f32_on_stream(
        input: *const f32,
        inv_freq: *const f32,
        output: *mut f32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        input_token_offset: u32,
        start_position: u32,
        attention_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_add_f32_on_stream(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_concat_f32_rows_on_stream(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_copy_f32_rows_into_columns_on_stream(
        input: *const f32,
        output: *mut f32,
        rows: u32,
        input_cols: u32,
        output_cols: u32,
        output_col_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_increment_u32_on_stream(
        values: *mut u32,
        len: u32,
        increment: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_store_u32_column_on_stream(
        input: *const u32,
        output: *mut u32,
        rows: u32,
        columns: u32,
        column: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_prepend_u32_rows_on_stream(
        first: *const u32,
        remaining: *const u32,
        output: *mut u32,
        rows: u32,
        remaining_columns: u32,
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
    pub(crate) fn infer_gather_group_row_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        groups: u32,
        rows_per_group: u32,
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
    pub(crate) fn infer_copy_bf16_row_to_f32_on_stream(
        input: *const u16,
        row: u32,
        output: *mut f32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_copy_bf16_rows_to_f32_indexed_on_stream(
        input: *const u16,
        rows: *const u32,
        output: *mut f32,
        batch_size: u32,
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
    pub(crate) fn infer_rms_norm_quantize_nvfp4_col_major_f32_on_stream(
        input: *const f32,
        weight: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        cols: u32,
        eps: f32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_rms_norm_quantize_nvfp4_pair_col_major_f32_on_stream(
        input: *const f32,
        weight: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        residual_packed: *mut u8,
        residual_scales: *mut u8,
        rows: u32,
        cols: u32,
        eps: f32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gelu_tanh_mul_quantize_nvfp4_col_major_f32_on_stream(
        gate: *const f32,
        up: *const f32,
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
    pub(crate) fn infer_prefill_gqa_attention_f32_on_stream(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        tokens: u32,
        start_position: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_append_ragged_kv_f32_on_stream(
        key: *const f32,
        value: *const f32,
        key_cache_table: *const *mut f32,
        value_cache_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        start_positions: *const u32,
        sequence_count: u32,
        total_tokens: u32,
        width: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ragged_gqa_attention_f32_on_stream(
        query: *const f32,
        key_cache_table: *const *mut f32,
        value_cache_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        start_positions: *const u32,
        output: *mut f32,
        sequence_count: u32,
        total_tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
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
    pub(crate) fn infer_argmax_f32_batch_on_stream(
        values: *const f32,
        out_index: *mut u32,
        out_value: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_speculative_accept_argmax_f32_on_stream(
        previous_logits: *const *const f32,
        verification_logits: *const f32,
        drafted_tokens: *const u32,
        accepted_counts: *mut u32,
        next_tokens: *mut u32,
        sequence_count: u32,
        draft_count: u32,
        vocab_size: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_sample_topk_topp_f32_batch_on_stream(
        logits: *const f32,
        params: *const c_void,
        stage_one_keys: *mut u64,
        stage_two_keys: *mut u64,
        top_keys: *mut u64,
        results: *mut c_void,
        rows: u32,
        vocab: u32,
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
    pub(crate) fn infer_bf16_linear_logits_f32_batch_on_stream(
        input: *const f32,
        weight: *const u16,
        logits: *mut f32,
        batch_size: u32,
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
    pub(crate) fn infer_f32_to_bf16_on_stream(
        input: *const f32,
        output: *mut u16,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_pack_token_heads_bf16_on_stream(
        input: *const f32,
        output: *mut u16,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        input_row_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_pack_value_heads_bf16_on_stream(
        input: *const f32,
        output: *mut u16,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_causal_window_softmax_f32_on_stream(
        scores: *mut f32,
        query_tokens: u32,
        key_tokens: u32,
        start_position: u32,
        heads: u32,
        head_dim: u32,
        window_tokens: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_causal_window_softmax_f32_to_bf16_on_stream(
        scores: *const f32,
        probabilities: *mut u16,
        query_tokens: u32,
        key_tokens: u32,
        start_position: u32,
        heads: u32,
        head_dim: u32,
        window_tokens: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_unpack_heads_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        output_row_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_unpack_heads_quantize_nvfp4_col_major_f32_on_stream(
        input: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        output_row_offset: u32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_unpack_heads_quantize_nvfp4_col_major_bf16_on_stream(
        input: *const u16,
        packed: *mut u8,
        scales: *mut u8,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        output_row_offset: u32,
        input_scale: f32,
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
    pub(crate) fn infer_gated_delta_net_128_f32_batch_on_stream(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state_table: *const *mut f32,
        output: *mut f32,
        batch_size: u32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gated_delta_net_128_f32_chunks_on_stream(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        output: *mut f32,
        sequence_count: u32,
        total_tokens: u32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_chunk_cumsum_on_stream(
        gate: *const u16,
        gate_cumsum: *mut f32,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        chunk_count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_chunk_kkt_on_stream(
        key: *const u16,
        beta: *const u16,
        gate_cumsum: *const f32,
        a: *mut f32,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        chunk_count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_chunk_solve_on_stream(
        a: *mut f32,
        a_inverse: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        chunk_count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_chunk_wu_on_stream(
        key: *const u16,
        value: *const u16,
        a_inverse: *const u16,
        gate_cumsum: *const f32,
        w: *mut u16,
        u: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        chunk_count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_chunk_h_on_stream(
        key: *const u16,
        u: *const u16,
        w: *const u16,
        value_new: *mut u16,
        gate_cumsum: *const f32,
        h: *mut u16,
        state: *mut f32,
        cu_seqlens: *const i32,
        chunk_offsets: *const i64,
        sequence_count: u32,
        total_tokens: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_chunk_output_on_stream(
        query: *const u16,
        key: *const u16,
        value_new: *const u16,
        h: *const u16,
        gate_cumsum: *const f32,
        output: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        chunk_count: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_gather_f32_pointer_rows_on_stream(
        input_table: *const *mut f32,
        output: *mut f32,
        rows: u32,
        row_values: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_scatter_f32_pointer_rows_on_stream(
        input: *const f32,
        output_table: *const *mut f32,
        rows: u32,
        row_values: u32,
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
    pub(crate) fn infer_fp8_linear_f32_batch_on_stream(
        input: *const f32,
        weight: *const u8,
        output: *mut f32,
        batch_size: u32,
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
    pub(crate) fn infer_fp8_linear_channel_scaled_f32_batch_configured_on_stream(
        input: *const f32,
        weight: *const u8,
        channel_weight_scale: *const f32,
        output: *mut f32,
        batch_size: u32,
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
    pub(crate) fn infer_quantize_fp8_e4m3_dynamic_f32_batch_on_stream(
        input: *const f32,
        quantized_input: *mut u8,
        input_scale: *mut f32,
        rows: u32,
        cols: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_quantize_fp8_e4m3_bf16_channel_scaled_on_stream(
        input: *const u16,
        channel_scale: *const f32,
        output: *mut u8,
        rows: u32,
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
    pub(crate) fn infer_scale_channel_f32_device_row_scalar_on_stream(
        values: *mut f32,
        channel_scale: *const f32,
        row_scale: *const f32,
        rows: u32,
        channels: u32,
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
    pub(crate) fn infer_sm121_w4a16_supported() -> i32;
    pub(crate) fn infer_sm121_w4a16_gate_up_on_stream(
        indices: *const u32,
        input: *const f32,
        tiled_weight: *const u8,
        weight_scale: *const u8,
        global_scale: *const f32,
        output_bf16: *mut u16,
        output_f32: *mut f32,
        batch_size: u32,
        top_k: u32,
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
    pub(crate) fn infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_on_stream(
        input: *const f32,
        packed_weight: *const u8,
        weight_scale: *const u8,
        output: *mut f32,
        batch_size: u32,
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
    pub(crate) fn infer_nvfp4_w4a16_grouped_inputs_matvec_f32_on_stream(
        indices: *const u32,
        input_table: *const *const f32,
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
    pub(crate) fn infer_q2_w2a16_grouped_matvec_f32_on_stream(
        indices: *const u32,
        input: *const f32,
        packed_weight_table: *const *const u8,
        weight_scale_table: *const *const u16,
        output_table: *const *mut f32,
        table_len: u32,
        groups: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_q2_w2a16_grouped_inputs_matvec_f32_on_stream(
        indices: *const u32,
        input_table: *const *const f32,
        packed_weight_table: *const *const u8,
        weight_scale_table: *const *const u16,
        output_table: *const *mut f32,
        table_len: u32,
        groups: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_q2_nvfp4_mixed_grouped_matvec_f32_on_stream(
        indices: *const u32,
        input: *const f32,
        q2_packed_weight_table: *const *const u8,
        q2_weight_scale_table: *const *const u16,
        expert_to_hot: *const u32,
        hot_packed_weight_table: *const *const u8,
        hot_weight_scale_table: *const *const u8,
        hot_weight_scale_2_table: *const *const f32,
        output_table: *const *mut f32,
        experts: u32,
        hot_capacity: u32,
        groups: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_q2_nvfp4_mixed_grouped_inputs_matvec_f32_on_stream(
        indices: *const u32,
        input_table: *const *const f32,
        q2_packed_weight_table: *const *const u8,
        q2_weight_scale_table: *const *const u16,
        expert_to_hot: *const u32,
        hot_packed_weight_table: *const *const u8,
        hot_weight_scale_table: *const *const u8,
        hot_weight_scale_2_table: *const *const f32,
        output_table: *const *mut f32,
        experts: u32,
        hot_capacity: u32,
        groups: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_q2_nvfp4_mixed_routed_matvec_f32_on_stream(
        indices: *const u32,
        input: *const f32,
        q2_packed_weight_table: *const *const u8,
        q2_weight_scale_table: *const *const u16,
        expert_to_hot: *const u32,
        hot_packed_weight_table: *const *const u8,
        hot_weight_scale_table: *const *const u8,
        hot_weight_scale_2_table: *const *const f32,
        output: *mut f32,
        experts: u32,
        hot_capacity: u32,
        routes: u32,
        routes_per_input: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_q3_nvfp4_mixed_routed_matvec_f32_on_stream(
        indices: *const u32,
        input: *const f32,
        q3_packed_weight_table: *const *const u8,
        q3_weight_scale_table: *const *const u16,
        expert_to_hot: *const u32,
        hot_packed_weight_table: *const *const u8,
        hot_weight_scale_table: *const *const u8,
        hot_weight_scale_2_table: *const *const f32,
        output: *mut f32,
        experts: u32,
        hot_capacity: u32,
        routes: u32,
        routes_per_input: u32,
        out_features: u32,
        in_features: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nvfp4_slot_routed_matvec_f32_on_stream(
        slots: *const u32,
        input: *const f32,
        packed_weight_table: *const *const u8,
        weight_scale_table: *const *const u8,
        weight_scale_2_table: *const f32,
        output: *mut f32,
        capacity: u32,
        routes: u32,
        routes_per_input: u32,
        out_features: u32,
        in_features: u32,
        output_route_offset: u32,
        output_stride: u32,
        output_offset: u32,
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
    pub(crate) fn infer_ling3_kda_gate_f32_on_stream(
        raw_gate: *const f32,
        beta_input: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        gate: *mut f32,
        beta: *mut f32,
        heads: u32,
        lower_bound: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ling3_kda_prep_on_stream(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state: *mut f32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ling3_kda_128_f32_on_stream(
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
    pub(crate) fn infer_ling3_sigmoid_gated_rms_norm_f32_on_stream(
        input: *const f32,
        gate: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ling3_mla_pack_f32_on_stream(
        query_projection: *const f32,
        kv_projection: *const f32,
        shared_rope_key: *const f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        heads: u32,
        qk_nope_dim: u32,
        rope_dim: u32,
        value_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_ling3_mla_attention_f32_on_stream(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        cache_len: u32,
        heads: u32,
        qk_dim: u32,
        value_dim: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_prep_batch_on_stream(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state_table: *const *mut f32,
        batch_size: u32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_prep_chunks_on_stream(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        conv_state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        sequence_count: u32,
        total_tokens: u32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_prep_chunks_bf16_on_stream(
        qkv: *const f32,
        conv_weight_bf16: *const u16,
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        conv_state_table: *const *mut f32,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        sequence_count: u32,
        total_tokens: u32,
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
    pub(crate) fn infer_qwen36_gdn_gate_batch_on_stream(
        alpha: *const f32,
        beta_input: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut f32,
        beta: *mut f32,
        batch_size: u32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_gate_batch_bf16_on_stream(
        alpha: *const f32,
        beta_input: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut u16,
        beta: *mut u16,
        rows: u32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_gate_paired_batch_on_stream(
        alpha_beta: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut f32,
        beta: *mut f32,
        rows: u32,
        heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_qwen36_gdn_gate_paired_batch_bf16_on_stream(
        alpha_beta: *const f32,
        a_log_bf16: *const u16,
        dt_bias_bf16: *const u16,
        gate: *mut u16,
        beta: *mut u16,
        rows: u32,
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
    pub(crate) fn infer_gated_rms_norm_quantize_nvfp4_col_major_f32_on_stream(
        input: *const f32,
        gate: *const f32,
        weight: *const f32,
        packed: *mut u8,
        scales: *mut u8,
        rows: u32,
        heads: u32,
        head_dim: u32,
        eps: f32,
        input_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_relu_squared_f32_on_stream(
        input: *const f32,
        output: *mut f32,
        len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_mamba_conv_update_f32_on_stream(
        projected: *const f32,
        conv_weight_bf16: *const u16,
        conv_bias_bf16: *const u16,
        conv_state: *mut u16,
        conv_output: *mut f32,
        intermediate_size: u32,
        conv_channels: u32,
        conv_kernel: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_mamba_conv_update_f32_chunks_on_stream(
        projected: *const f32,
        conv_weight_bf16: *const u16,
        conv_bias_bf16: *const u16,
        conv_state_table: *const *mut u16,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        conv_output: *mut f32,
        sequence_count: u32,
        projection_size: u32,
        intermediate_size: u32,
        conv_channels: u32,
        conv_kernel: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_mamba_conv_update_f32_chunks_snapshot_on_stream(
        projected: *const f32,
        conv_weight_bf16: *const u16,
        conv_bias_bf16: *const u16,
        conv_state_table: *const *mut u16,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        conv_output: *mut f32,
        state_snapshots_bf16: *mut u16,
        sequence_count: u32,
        snapshot_slots: u32,
        projection_size: u32,
        intermediate_size: u32,
        conv_channels: u32,
        conv_kernel: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_mamba_state_update_f32_on_stream(
        projected: *const f32,
        conv_output: *const f32,
        a_log_bf16: *const u16,
        d_bf16: *const u16,
        dt_bias_bf16: *const u16,
        norm_weight_bf16: *const u16,
        ssm_state: *mut u16,
        output: *mut f32,
        heads: u32,
        head_dim: u32,
        groups: u32,
        state_size: u32,
        dt_floor: f32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_mamba_state_update_f32_chunks_on_stream(
        projected: *const f32,
        conv_output: *const f32,
        a_log_bf16: *const u16,
        d_bf16: *const u16,
        dt_bias_bf16: *const u16,
        norm_weight_bf16: *const u16,
        ssm_state_table: *const *mut u16,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        output: *mut f32,
        sequence_count: u32,
        total_tokens: u32,
        projection_size: u32,
        heads: u32,
        head_dim: u32,
        groups: u32,
        state_size: u32,
        dt_floor: f32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_nemotron3_mamba_state_update_f32_chunks_snapshot_on_stream(
        projected: *const f32,
        conv_output: *const f32,
        a_log_bf16: *const u16,
        d_bf16: *const u16,
        dt_bias_bf16: *const u16,
        norm_weight_bf16: *const u16,
        ssm_state_table: *const *mut u16,
        sequence_offsets: *const u32,
        sequence_lengths: *const u32,
        output: *mut f32,
        state_snapshots_bf16: *mut u16,
        sequence_count: u32,
        total_tokens: u32,
        snapshot_slots: u32,
        projection_size: u32,
        heads: u32,
        head_dim: u32,
        groups: u32,
        state_size: u32,
        dt_floor: f32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn infer_select_bf16_state_snapshot_on_stream(
        state_table: *const *mut u16,
        snapshots_bf16: *const u16,
        selected_slots: *const u32,
        sequence_count: u32,
        snapshot_slots: u32,
        state_size: u32,
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
    pub(crate) fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn cudaHostAlloc(host_ptr: *mut *mut c_void, size: usize, flags: u32)
    -> cudaError_t;
    pub(crate) fn cudaFreeHost(host_ptr: *mut c_void) -> cudaError_t;
    pub(crate) fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> cudaError_t;
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
    pub(crate) fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: u32) -> cudaError_t;
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
    pub(crate) fn cublasLtMatrixLayoutSetAttribute(
        layout: cublasLtMatrixLayout_t,
        attr: i32,
        buf: *const c_void,
        size_in_bytes: usize,
    ) -> cublasStatus_t;

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
