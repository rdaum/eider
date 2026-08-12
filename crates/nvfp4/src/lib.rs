//! Rust host-side support for NVIDIA NVFP4 tensors on DGX Spark.
//!
//! This crate is deliberately narrow today: it owns CUDA/cuBLASLt resources for
//! a proven NVFP4 GEMM path, rather than trying to be a general tensor library.
//! The public API is shaped around cuBLASLt-compatible matrix layouts, packed
//! E2M1 values, UE4M3 block-scale storage, and a TN matmul plan.
//!
//! The current NVFP4 operation is `D = A^T * B`, where A and B are stored as
//! packed `CUDA_R_4F_E2M1` values with cuBLASLt `VEC16_UE4M3` scales, and D is
//! BF16. Prefer [`Fp4TnMatmul`] when the operation can own its matrices.

#![warn(missing_docs)]

pub mod format;

mod bitnet;
mod cublaslt;
mod cuda;
mod diagnostics;
mod error;
mod expert_slots;
mod ffi;
mod kernels;
mod matrix;
mod modelopt;
mod q2;
mod q3;
mod safetensors;
mod tensor;
mod ternary_g64;

pub use bitnet::{
    BitNetActivationWorkspace, BitNetMatrix, BitNetPackedLinear,
    relu_squared_mul_halves_f32_batch_into_on_stream,
};
pub use cublaslt::{
    Bf16TnMatmulPlan, CublasLt, CutlassFp4GroupedGemmPlan, CutlassFp4GroupedGemvF32Plan,
    Fp4TnMatmul, Fp4TnMatmulPlan, Fp4TnPlanMetadata, Fp8TnMatmulPlan, GemmShape, InferenceGemm,
    Int8TnMatmulPlan, Nvfp4TnInputs,
};
pub use cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, DeviceInOut, DeviceInput, DeviceOutput,
    HostRead, PinnedHostBuffer, device_memory_info, set_cuda_device, synchronize_device,
};
pub use diagnostics::gpu_counters::{GpuCounterCollector, GpuCounterMetric};
pub use diagnostics::smoke::{run_e2m1_oracle_check, run_fp4_ones_smoke, run_fp32_smoke};
pub use error::{Error, Result};
pub use expert_slots::Nvfp4LinearSlots;
pub use kernels::deepseek4::{
    Deepseek4AttentionBatch, Deepseek4CausalAttentionBatch,
    arithmetic_positions_u32_into_on_stream, attention_f32_batch_into_on_stream,
    block_fp8_grouped_linear_f32_batch_into_on_stream, block_fp8_linear_f32_batch_into_on_stream,
    block_fp8_linear_f32_into_on_stream, causal_attention_f32_batch_into_on_stream,
    compress_windows_f32_into_on_stream, gather_sorted_route_rows_f32_into_on_stream,
    hyper_apply_f32_batch_into_on_stream, hyper_head_f32_batch_into_on_stream,
    hyper_prepare_f32_batch_into_on_stream, indexer_topk_f32_batch_into_on_stream,
    repeat_hyper_streams_f32_into_on_stream,
    rope_interleaved_trailing_f32_indexed_in_place_on_stream,
    routed_accumulate_f32_batch_into_on_stream, routed_accumulate_sorted_f32_batch_into_on_stream,
    router_hash_f32_batch_into_on_stream, router_topk_f32_batch_into_on_stream,
    store_compression_overlap_f32_into_on_stream, swiglu_pair_clamped_f32_batch_into_on_stream,
    swiglu_pair_f32_batch_into_on_stream,
};
pub use kernels::gemma4_attention::Gemma4LocalPrefillAttention;
pub use kernels::non_gemm::{
    ArgmaxResult, GPU_SAMPLING_MAX_TOP_K, GpuSampledToken, GpuSamplingRow, GpuTokenSampler,
    GroupedGemvPointerBuffers, GroupedGemvPointerTableBuffers, MoeSiluQuantizeSlotBuffers,
    MoeSortedNvfp4Rows, MoeSortedRoutes, MropeSections, add_f32_into_on_stream,
    add_f32_prefix_into_on_stream, append_ragged_kv_f32_into_on_stream,
    append_rows_f32_indexed_into_on_stream, append_rows_f32_into_on_stream,
    argmax_f32_batch_into_on_stream, argmax_f32_into_on_stream, bf16_linear_argmax_f32,
    bf16_linear_argmax_f32_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, bf16_linear_pair_logits_f32_into_on_stream,
    bf16_matrix_to_f32_into_on_stream, bf16_to_f32_prefix_into_on_stream,
    cached_gqa_attention_f32_indexed_into_on_stream, cached_gqa_attention_f32_into_on_stream,
    cached_gqa_attention_nvfp4_into_on_stream, causal_window_softmax_f32_in_place_on_stream,
    causal_window_softmax_f32_to_bf16_on_stream, clear_expert_counts_u64_on_stream,
    concat_f32_rows_into_on_stream, copy_bf16_row_to_f32_indexed_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, copy_bf16_rows_to_f32_indexed_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, copy_f32_rows_into_columns_on_stream,
    copy_row_f32_into_on_stream, dual_rms_norm_add_f32_into_on_stream,
    dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_into_on_stream,
    dual_rms_norm_rope_neox_proportional_sequence_f32_at_offset_into_on_stream,
    f32_to_bf16_into_on_stream, f32_to_bf16_prefix_into_on_stream, fill_f32_into_on_stream,
    fill_f32_prefix_into_on_stream, fp8_linear_channel_scaled_dynamic_f32_into_on_stream,
    fp8_linear_channel_scaled_dynamic_quantized_f32_configured_into_on_stream,
    fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream,
    fp8_linear_channel_scaled_f32_batch_into_on_stream,
    fp8_linear_channel_scaled_f32_into_on_stream,
    fp8_linear_channel_scaled_precomputed_dynamic_f32_into_on_stream,
    fp8_linear_configured_f32_into_on_stream, fp8_linear_f32_batch_into_on_stream,
    fp8_linear_f32_into_on_stream, fp8_linear_pair_configured_f32_into_on_stream,
    fp8_linear_triple_configured_f32_into_on_stream, fp8_linear_w8a8_f32_into_on_stream,
    fp8_moe_grouped_down_f32_into_on_stream, fp8_moe_grouped_gate_up_f32_into_on_stream,
    gated_delta_net_128_f32_batch_into_on_stream, gated_delta_net_128_f32_chunks_into_on_stream,
    gated_delta_net_128_f32_into_on_stream, gated_rms_norm_f32_into_on_stream,
    gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    gather_f32_pointer_rows_into_on_stream, gather_group_row_f32_into_on_stream,
    gather_indexed_mul_f32_into_on_stream, gather_indexed_mul_f32_prefix_into_on_stream,
    gather_nvfp4_grouped_gemv_ptr_tables_on_stream, gather_nvfp4_grouped_gemv_ptrs_on_stream,
    gelu_tanh_f32_into_on_stream, gelu_tanh_mul_f32_into_on_stream,
    gelu_tanh_mul_halves_f32_into_on_stream,
    gelu_tanh_mul_quantize_nvfp4_col_major_f32_into_on_stream, increment_u32_in_place_on_stream,
    ling3_kda_128_f32_into_on_stream, ling3_kda_gate_f32_into_on_stream,
    ling3_kda_prep_into_on_stream, ling3_mla_attention_f32_into_on_stream,
    ling3_mla_pack_f32_into_on_stream, ling3_sigmoid_gated_rms_norm_f32_into_on_stream,
    lm_head_top1_f32_into_on_stream, moe_silu_quantize_fp8_slots_f32_into_on_stream,
    moe_silu_quantize_slots_nvfp4_on_stream, moe_silu_quantize_slots_nvfp4_simple_scales_on_stream,
    moe_silu_slots_f32_into_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_topk_f32_into_on_stream, moe_weighted_accumulate_slots_f32_batch_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    moe_weighted_accumulate_sorted_slots_f32_batch_on_stream,
    nemotron3_mamba_conv_update_f32_chunks_into_on_stream,
    nemotron3_mamba_conv_update_f32_chunks_snapshot_into_on_stream,
    nemotron3_mamba_conv_update_f32_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream,
    nemotron3_mamba_state_update_f32_into_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream, nemotron3_sigmoid_topk_f32_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_f32_into_on_stream,
    nvfp4_w4a16_matrix_matvec_f32_batch_into_on_stream,
    nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, nvfp4_w4a16_matvec_f32_into_on_stream,
    nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream,
    nvfp4_w4a16_top1_configured_f32_into_on_stream, nvfp4_w4a16_top1_f32_into_on_stream,
    pack_token_heads_bf16_at_offset_into_on_stream, pack_token_heads_bf16_into_on_stream,
    pack_value_heads_bf16_into_on_stream, prefill_gqa_attention_f32_into,
    prefill_gqa_attention_f32_into_on_stream, prepend_u32_rows_into_on_stream,
    quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_into_on_stream, quantize_fp8_e4m3_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    quantize_nvfp4_simple_scales_f32_into_on_stream,
    quantize_nvfp4_vector_simple_scales_f32_into_on_stream,
    qwen36_ffn_finalize_batch_f32_into_on_stream, qwen36_ffn_finalize_f32_into_on_stream,
    qwen36_ffn_finalize_routed_batch_f32_into_on_stream,
    qwen36_ffn_finalize_routed_f32_into_on_stream, qwen36_full_attn_prep_f32_batch_into_on_stream,
    qwen36_full_attn_prep_f32_into_on_stream, qwen36_gdn_gate_batch_bf16_into_on_stream,
    qwen36_gdn_gate_batch_into_on_stream, qwen36_gdn_gate_into_on_stream,
    qwen36_gdn_gate_paired_batch_bf16_into_on_stream, qwen36_gdn_gate_paired_batch_into_on_stream,
    qwen36_gdn_prep_batch_into_on_stream, qwen36_gdn_prep_chunks_bf16_into_on_stream,
    qwen36_gdn_prep_chunks_into_on_stream, qwen36_gdn_prep_into_on_stream,
    ragged_gqa_attention_f32_into_on_stream, record_expert_indices_prefix_u64_on_stream,
    record_expert_indices_u64_on_stream, relu_squared_f32_into_on_stream,
    remap_expert_indices_at_offset_into_on_stream, remap_expert_indices_into_on_stream,
    remap_expert_indices_range_into_on_stream, rms_norm_add_channel_row_scale_f32_into_on_stream,
    rms_norm_add_f32_into_on_stream, rms_norm_add_then_rms_norm_quantize_nvfp4_f32_into_on_stream,
    rms_norm_f32_into_on_stream, rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    rms_norm_quantize_nvfp4_pair_col_major_f32_into_on_stream,
    rms_norm_rope_neox_f32_indexed_into_on_stream, rope_imrope_f32_indexed_into_on_stream,
    rope_imrope_f32_into_on_stream, rope_imrope_text_batch_f32_into_on_stream,
    rope_neox_f32_indexed_into_on_stream, rope_neox_f32_into_on_stream,
    rope_neox_inv_freq_scaled_sequence_f32_at_offset_into_on_stream,
    rope_neox_inv_freq_scaled_sequence_f32_into_on_stream,
    rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream,
    rope_neox_inv_freq_sequence_f32_into_on_stream, rope_neox_partial_f32_into_on_stream,
    rope_neox_proportional_f32_into_on_stream,
    rope_neox_proportional_sequence_f32_at_offset_into_on_stream,
    rope_neox_sequence_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    round_f32_to_bf16_into_on_stream, round_f32_to_bf16_prefix_in_place_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream,
    scale_channel_f32_device_scalar_in_place_on_stream, scaled_add_f32_into_on_stream,
    scatter_f32_pointer_rows_on_stream, select_bf16_state_snapshot_into_on_stream,
    sigmoid_mul_f32_into_on_stream, sigmoid_mul_f32_prefix_into_on_stream,
    sigmoid_scale_heads_f32_into_on_stream, sigmoid_scale_scalar_f32_into_on_stream,
    silu_mul_f32_into_on_stream, silu_mul_f32_prefix_into_on_stream,
    silu_mul_halves_clamped_f32_batch_into_on_stream, silu_mul_halves_clamped_f32_into_on_stream,
    silu_mul_halves_f32_batch_into_on_stream, silu_mul_halves_f32_into_on_stream,
    silu_mul_halves_quantize_nvfp4_col_major_f32_into_on_stream, softmax_f32_in_place_on_stream,
    softplus_scale_heads_f32_into_on_stream, softplus_scale_heads_f32_prefix_into_on_stream,
    speculative_accept_argmax_f32_into_on_stream, split_q_gate_f32_into_on_stream,
    split_qkv_f32_batch_into_on_stream, split_qkv_f32_into_on_stream,
    step37_sigmoid_top8_f32_batch_into_on_stream, step37_sigmoid_top8_f32_into_on_stream,
    store_u32_column_into_on_stream, unpack_heads_f32_at_offset_into_on_stream,
    unpack_heads_f32_into_on_stream,
    unpack_heads_quantize_nvfp4_col_major_bf16_at_offset_into_on_stream,
    unpack_heads_quantize_nvfp4_col_major_f32_at_offset_into_on_stream,
};
pub use kernels::qwen36_gdn::Qwen36ChunkedGdn;
pub use kernels::sm12x_kv_cache::{
    SM12X_KV_PAGE_TOKENS, Sm12xKvAttentionWorkspace, Sm12xKvCache, Sm12xKvPagePool,
    Sm12xKvTailSnapshot,
};
pub use kernels::sm12x_mma::{
    Sm12xFp4DeviceGemmVector, Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmVector, Sm12xFp4GemmWeight,
    Sm12xFp4Tile, Sm12xFp4TileSet, Sm12xRequantizedVector, Sm12xRequantizedWeight,
    device_vector_from_native_parts, device_weight_gemv_native_vector_on_stream,
    device_weight_gemv_on_stream, gemv_row_scales_residual2_batch_on_stream,
    gemv_row_scales_residual2_splitk_batch_on_stream, indexed_gemv_on_stream,
    indexed_grouped_gemv_on_stream, indexed_grouped_gemv_row_scales_on_stream,
    indexed_grouped_gemv_row_scales_residual_on_stream, modelopt_m16_k64_row_scale_words,
    moe_silu_quantize_bf16_expert_sorted_slots_on_stream, moe_silu_quantize_bf16_slots_on_stream,
    moe_silu_quantize_bf16_sorted_slots_on_stream, moe_silu_quantize_slots_on_stream,
    moe_silu_quantize_slots_reference_on_stream, moe_silu_quantize_slots_residual_on_stream,
    quantize_dynamic_vector_on_stream, quantize_dynamic_vectors_residual2_on_stream,
    quantize_fixed_scale_vector_on_stream,
};
pub use kernels::sm121_w4a16::{
    Sm121W4A16GateUp, Sm121W4A16GateUpBatchWorkspace, Sm121W4A16HostWeight, Sm121W4A16Linear,
    Sm121W4A16LinearBatchWorkspace,
};
pub use matrix::{Bf16Matrix, F32Matrix, MatrixShape, Nvfp4Matrix};
pub use modelopt::{
    ModelOptBlockScaledFp8Linear, ModelOptCheckpoint, ModelOptCublasLtWeight, ModelOptFp8Linear,
    ModelOptNvfp4Activation, ModelOptNvfp4Linear, modelopt_scales_to_cublaslt,
};
pub use q2::{
    Q2_BLOCK_SIZE, Q2ExpertTable, Q2ExpertTableCacheInfo, Q2ExpertTableCacheWriter, Q2Matrix,
    Q2Nvfp4ExpertOverlay, QuantizedQ2, dequantize_q2_row_major,
    q2_nvfp4_mixed_grouped_inputs_matvec_f32_into_on_stream,
    q2_nvfp4_mixed_grouped_matvec_f32_into_on_stream,
    q2_w2a16_grouped_inputs_matvec_f32_into_on_stream, q2_w2a16_grouped_matvec_f32_into_on_stream,
    quantize_q2_row_major,
};
pub use q3::{
    Q3_BLOCK_SIZE, Q3ExpertTable, Q3ExpertTableCacheInfo, Q3ExpertTableCacheWriter,
    Q3Nvfp4ExpertOverlay, QuantizedQ3, dequantize_q3_row_major,
    q3_nvfp4_mixed_routed_matvec_f32_into_on_stream, quantize_q3_row_major,
};
pub use safetensors::{SafeTensorInfo, SafeTensorShard};
pub use tensor::{Bf16Tensor2d, Nvfp4Tensor2d, Tensor2dLayout, Tensor2dView};
pub use ternary_g64::{
    TERNARY_G64_GROUP_SIZE, TernaryG64ActivationWorkspace, TernaryG64Matrix, TernaryG64PackedLinear,
};
