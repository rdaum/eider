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

mod cublaslt;
mod cuda;
mod diagnostics;
mod error;
mod ffi;
mod kernels;
mod matrix;
mod modelopt;
mod safetensors;
mod tensor;

pub use cublaslt::{
    Bf16TnMatmulPlan, CublasLt, CutlassFp4GroupedGemvF32Plan, Fp4TnMatmul, Fp4TnMatmulPlan,
    Fp4TnPlanMetadata, Fp8TnMatmulPlan, GemmShape, InferenceGemm, Nvfp4TnInputs,
};
pub use cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, DeviceInOut, DeviceInput, DeviceOutput,
    HostRead, PinnedHostBuffer, device_memory_info, synchronize_device,
};
pub use diagnostics::gpu_counters::{GpuCounterCollector, GpuCounterMetric};
pub use diagnostics::smoke::{run_e2m1_oracle_check, run_fp4_ones_smoke, run_fp32_smoke};
pub use error::{Error, Result};
pub use kernels::marlin::{
    MarlinNvfp4GateUp, MarlinNvfp4GateUpBatchWorkspace, MarlinNvfp4HostWeight, MarlinNvfp4Linear,
};
pub use kernels::non_gemm::{
    ArgmaxResult, GPU_SAMPLING_MAX_TOP_K, GpuSampledToken, GpuSamplingRow, GpuTokenSampler,
    GroupedGemvPointerBuffers, GroupedGemvPointerTableBuffers, MoeSiluQuantizeSlotBuffers,
    MropeSections, add_f32_into_on_stream, append_ragged_kv_f32_into_on_stream,
    append_rows_f32_indexed_into_on_stream, append_rows_f32_into_on_stream,
    argmax_f32_batch_into_on_stream, argmax_f32_into_on_stream, bf16_linear_argmax_f32,
    bf16_linear_argmax_f32_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, bf16_linear_pair_logits_f32_into_on_stream,
    bf16_matrix_to_f32_into_on_stream, cached_gqa_attention_f32_indexed_into_on_stream,
    cached_gqa_attention_f32_into_on_stream, cached_gqa_attention_nvfp4_into_on_stream,
    concat_f32_rows_into_on_stream, copy_bf16_row_to_f32_indexed_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, copy_bf16_rows_to_f32_indexed_into_on_stream,
    copy_row_f32_into_on_stream, f32_to_bf16_into_on_stream, fill_f32_into_on_stream,
    fp8_linear_channel_scaled_dynamic_f32_into_on_stream,
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
    gather_group_row_f32_into_on_stream, gather_indexed_mul_f32_into_on_stream,
    gather_nvfp4_grouped_gemv_ptr_tables_on_stream, gather_nvfp4_grouped_gemv_ptrs_on_stream,
    gelu_tanh_f32_into_on_stream, gelu_tanh_mul_f32_into_on_stream,
    gelu_tanh_mul_halves_f32_into_on_stream, increment_u32_in_place_on_stream,
    lm_head_top1_f32_into_on_stream, moe_silu_quantize_fp8_slots_f32_into_on_stream,
    moe_silu_quantize_slots_nvfp4_on_stream, moe_silu_quantize_slots_nvfp4_simple_scales_on_stream,
    moe_silu_slots_f32_into_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_topk_f32_into_on_stream, moe_weighted_accumulate_slots_f32_batch_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream,
    nemotron3_mamba_conv_update_f32_chunks_into_on_stream,
    nemotron3_mamba_conv_update_f32_chunks_snapshot_into_on_stream,
    nemotron3_mamba_conv_update_f32_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream,
    nemotron3_mamba_state_update_f32_into_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream, nemotron3_sigmoid_topk_f32_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_f32_into_on_stream,
    nvfp4_w4a16_matvec_block_per_row_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, nvfp4_w4a16_matvec_f32_into_on_stream,
    nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream,
    nvfp4_w4a16_top1_configured_f32_into_on_stream, nvfp4_w4a16_top1_f32_into_on_stream,
    prefill_gqa_attention_f32_into, prepend_u32_rows_into_on_stream,
    quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_into_on_stream, quantize_fp8_e4m3_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    quantize_nvfp4_simple_scales_f32_into_on_stream,
    quantize_nvfp4_vector_simple_scales_f32_into_on_stream, qwen36_ffn_finalize_f32_into_on_stream,
    qwen36_ffn_finalize_routed_batch_f32_into_on_stream,
    qwen36_ffn_finalize_routed_f32_into_on_stream, qwen36_full_attn_prep_f32_batch_into_on_stream,
    qwen36_full_attn_prep_f32_into_on_stream, qwen36_gdn_gate_batch_into_on_stream,
    qwen36_gdn_gate_into_on_stream, qwen36_gdn_prep_batch_into_on_stream,
    qwen36_gdn_prep_chunks_into_on_stream, qwen36_gdn_prep_into_on_stream,
    ragged_gqa_attention_f32_into_on_stream, relu_squared_f32_into_on_stream,
    remap_expert_indices_at_offset_into_on_stream, remap_expert_indices_into_on_stream,
    rms_norm_f32_into_on_stream, rms_norm_rope_neox_f32_indexed_into_on_stream,
    rope_imrope_f32_indexed_into_on_stream, rope_imrope_f32_into_on_stream,
    rope_imrope_text_batch_f32_into_on_stream, rope_neox_f32_indexed_into_on_stream,
    rope_neox_f32_into_on_stream, rope_neox_inv_freq_sequence_f32_at_offset_into_on_stream,
    rope_neox_inv_freq_sequence_f32_into_on_stream, rope_neox_partial_f32_into_on_stream,
    rope_neox_proportional_f32_into_on_stream,
    rope_neox_proportional_sequence_f32_at_offset_into_on_stream,
    rope_neox_sequence_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    round_f32_to_bf16_into_on_stream, scale_channel_f32_device_row_scalar_in_place_on_stream,
    scale_channel_f32_device_scalar_in_place_on_stream, scaled_add_f32_into_on_stream,
    select_bf16_state_snapshot_into_on_stream, sigmoid_mul_f32_into_on_stream,
    sigmoid_scale_heads_f32_into_on_stream, sigmoid_scale_scalar_f32_into_on_stream,
    silu_mul_f32_into_on_stream, silu_mul_halves_clamped_f32_batch_into_on_stream,
    silu_mul_halves_clamped_f32_into_on_stream, silu_mul_halves_f32_batch_into_on_stream,
    silu_mul_halves_f32_into_on_stream,
    silu_mul_halves_quantize_nvfp4_col_major_f32_into_on_stream, softmax_f32_in_place_on_stream,
    speculative_accept_argmax_f32_into_on_stream, split_q_gate_f32_into_on_stream,
    split_qkv_f32_into_on_stream, step37_sigmoid_top8_f32_batch_into_on_stream,
    step37_sigmoid_top8_f32_into_on_stream, store_u32_column_into_on_stream,
};
pub use kernels::sm12x_kv_cache::{Sm12xKvAttentionWorkspace, Sm12xKvCache};
pub use kernels::sm12x_mma::{
    Sm12xFp4DeviceGemmVector, Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmVector, Sm12xFp4GemmWeight,
    Sm12xFp4Tile, Sm12xFp4TileSet, Sm12xRequantizedVector, Sm12xRequantizedWeight,
    device_vector_from_native_parts, device_weight_gemv_native_vector_on_stream,
    device_weight_gemv_on_stream, gemv_row_scales_residual2_batch_on_stream,
    gemv_row_scales_residual2_splitk_batch_on_stream, indexed_gemv_on_stream,
    indexed_grouped_gemv_on_stream, indexed_grouped_gemv_row_scales_on_stream,
    indexed_grouped_gemv_row_scales_residual_on_stream, modelopt_m16_k64_row_scale_words,
    moe_silu_quantize_bf16_slots_on_stream, moe_silu_quantize_slots_on_stream,
    moe_silu_quantize_slots_reference_on_stream, moe_silu_quantize_slots_residual_on_stream,
    quantize_dynamic_vector_on_stream, quantize_dynamic_vectors_residual2_on_stream,
    quantize_fixed_scale_vector_on_stream,
};
pub use matrix::{Bf16Matrix, F32Matrix, MatrixShape, Nvfp4Matrix};
pub use modelopt::{
    ModelOptCheckpoint, ModelOptCublasLtWeight, ModelOptFp8Linear, ModelOptNvfp4Activation,
    ModelOptNvfp4Linear, modelopt_scales_to_cublaslt,
};
pub use safetensors::{SafeTensorInfo, SafeTensorShard};
pub use tensor::{Bf16Tensor2d, Nvfp4Tensor2d, Tensor2dLayout, Tensor2dView};
