/*
 * Focused nvfp4 wrapper around the Apache-2.0 vLLM/Marlin MoE kernel.
 * See native/marlin/UPSTREAM.md for source attribution.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define MARLIN_NAMESPACE_NAME infer_marlin_moe
#include "moe/marlin_moe_wna16/marlin_template.h"

namespace {

constexpr int kHidden = 2048;
constexpr int kGateUp = 1024;
constexpr int kTopK = 8;
constexpr int kMoeBlockSize = 8;
constexpr int kThreads = 256;
// Exact storage used by the four-stage 1x8x8 W4A16 tile: route metadata,
// the B/reduction union, per-group scales, and the staged A tiles. Keeping the
// launch below half of an SM's 100 KiB budget permits two resident blocks.
constexpr int kDynamicSharedBytes = 45184;

using MarlinKernel = decltype(&infer_marlin_moe::Marlin<
    vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
    vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>);

MarlinKernel marlin_kernel() {
    return infer_marlin_moe::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>;
}

__global__ void prepare_routes_and_input_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    std::uint16_t* __restrict__ input_bf16,
    std::int32_t* __restrict__ sorted_token_ids,
    std::int32_t* __restrict__ expert_ids,
    std::int32_t* __restrict__ num_tokens_past_padded) {
    for (int col = threadIdx.x; col < kHidden; col += blockDim.x) {
        input_bf16[col] = __bfloat16_as_ushort(__float2bfloat16_rn(input[col]));
    }
    if (threadIdx.x < kTopK) {
        const int slot = threadIdx.x;
        const std::uint32_t expert = indices[slot];
        expert_ids[slot] = static_cast<std::int32_t>(expert);
        for (int row = 0; row < kMoeBlockSize; ++row) {
            sorted_token_ids[slot * kMoeBlockSize + row] = row == 0 ? slot : kTopK;
        }
    }
    if (threadIdx.x == 0) {
        num_tokens_past_padded[0] = kTopK * kMoeBlockSize;
    }
}

__global__ void prepare_routes_and_input_batch_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    std::uint16_t* __restrict__ input_bf16,
    std::int32_t* __restrict__ sorted_token_ids,
    std::int32_t* __restrict__ expert_ids,
    std::int32_t* __restrict__ num_tokens_past_padded,
    std::uint32_t batch_size) {
    const std::uint32_t batch = blockIdx.x;
    for (int col = threadIdx.x; col < kHidden; col += blockDim.x) {
        input_bf16[batch * kHidden + col] = __bfloat16_as_ushort(
            __float2bfloat16_rn(input[batch * kHidden + col]));
    }
    if (threadIdx.x < kTopK) {
        const std::uint32_t slot = threadIdx.x;
        const std::uint32_t group = batch * kTopK + slot;
        expert_ids[group] = static_cast<std::int32_t>(indices[group]);
        for (int row = 0; row < kMoeBlockSize; ++row) {
            sorted_token_ids[group * kMoeBlockSize + row] =
                row == 0 ? static_cast<std::int32_t>(group)
                         : static_cast<std::int32_t>(batch_size * kTopK);
        }
    }
    if (batch == 0 && threadIdx.x == 0) {
        num_tokens_past_padded[0] = batch_size * kTopK * kMoeBlockSize;
    }
}

__global__ void bf16_to_f32_kernel(
    const std::uint16_t* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = __bfloat162float(__ushort_as_bfloat16(input[idx]));
    }
}

__global__ void prepare_single_input_kernel(
    const float* __restrict__ input,
    std::uint16_t* __restrict__ input_bf16,
    std::int32_t* __restrict__ sorted_token_ids,
    std::int32_t* __restrict__ expert_ids,
    std::int32_t* __restrict__ num_tokens_past_padded,
    std::uint32_t input_len) {
    for (std::uint32_t col = threadIdx.x; col < input_len; col += blockDim.x) {
        input_bf16[col] = __bfloat16_as_ushort(__float2bfloat16_rn(input[col]));
    }
    if (threadIdx.x < kMoeBlockSize) {
        sorted_token_ids[threadIdx.x] = threadIdx.x == 0 ? 0 : 1;
    }
    if (threadIdx.x == 0) {
        expert_ids[0] = 0;
        num_tokens_past_padded[0] = kMoeBlockSize;
    }
}

}  // namespace

extern "C" int infer_marlin_nvfp4_gate_up_supported() {
    int device = 0;
    int major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        major < 8) {
        return 0;
    }
    return cudaFuncSetAttribute(
               marlin_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess
               ? 1
               : 0;
}

extern "C" cudaError_t infer_marlin_nvfp4_gate_up_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint32_t* repacked_weight,
    const std::uint8_t* weight_scale,
    const float* global_scale,
    float* output,
    std::uint16_t* input_bf16,
    std::uint16_t* output_bf16,
    float* reduce_tmp,
    std::int32_t* locks,
    std::int32_t* sorted_token_ids,
    std::int32_t* expert_ids,
    std::int32_t* num_tokens_past_padded,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr || repacked_weight == nullptr ||
        weight_scale == nullptr || global_scale == nullptr ||
        input_bf16 == nullptr || output_bf16 == nullptr || reduce_tmp == nullptr ||
        locks == nullptr || sorted_token_ids == nullptr || expert_ids == nullptr ||
        num_tokens_past_padded == nullptr) {
        return cudaErrorInvalidValue;
    }

    prepare_routes_and_input_kernel<<<1, kThreads, 0, stream>>>(
        indices, input, input_bf16, sorted_token_ids, expert_ids,
        num_tokens_past_padded);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    auto kernel = marlin_kernel();
    // Eight expert blocks times eight N tiles. One CUDA block per complete
    // MN tile avoids split-K reduction for the batch-one top-8 shape.
    constexpr int grid_blocks = kTopK * (kGateUp / 128);
    kernel<<<grid_blocks, kThreads, kDynamicSharedBytes, stream>>>(
        reinterpret_cast<const int4*>(input_bf16),
        reinterpret_cast<const int4*>(repacked_weight),
        reinterpret_cast<int4*>(output_bf16),
        reinterpret_cast<int4*>(reduce_tmp),
        nullptr,
        nullptr,
        reinterpret_cast<const int4*>(weight_scale),
        global_scale,
        nullptr,
        nullptr,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        nullptr,
        kTopK,
        false,
        kHidden / 16,
        1,
        kGateUp,
        kHidden,
        locks,
        false,
        false,
        true);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    constexpr std::uint32_t output_len = kTopK * kGateUp;
    constexpr int convert_threads = 256;
    constexpr int convert_blocks = (output_len + convert_threads - 1) / convert_threads;
    if (output != nullptr) {
        bf16_to_f32_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
            output_bf16, output, output_len);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t infer_marlin_nvfp4_gate_up_batch_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint32_t* repacked_weight,
    const std::uint8_t* weight_scale,
    const float* global_scale,
    float* output,
    std::uint16_t* input_bf16,
    std::uint16_t* output_bf16,
    float* reduce_tmp,
    std::int32_t* locks,
    std::int32_t* sorted_token_ids,
    std::int32_t* expert_ids,
    std::int32_t* num_tokens_past_padded,
    std::uint32_t batch_size,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr || repacked_weight == nullptr ||
        weight_scale == nullptr || global_scale == nullptr ||
        input_bf16 == nullptr || output_bf16 == nullptr || reduce_tmp == nullptr ||
        locks == nullptr || sorted_token_ids == nullptr || expert_ids == nullptr ||
        num_tokens_past_padded == nullptr || batch_size == 0) {
        return cudaErrorInvalidValue;
    }
    prepare_routes_and_input_batch_kernel<<<batch_size, kThreads, 0, stream>>>(
        indices, input, input_bf16, sorted_token_ids, expert_ids,
        num_tokens_past_padded, batch_size);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    auto kernel = marlin_kernel();
    const int routed_rows = batch_size * kTopK;
    const int grid_blocks = routed_rows * (kGateUp / 128);
    kernel<<<grid_blocks, kThreads, kDynamicSharedBytes, stream>>>(
        reinterpret_cast<const int4*>(input_bf16),
        reinterpret_cast<const int4*>(repacked_weight),
        reinterpret_cast<int4*>(output_bf16),
        reinterpret_cast<int4*>(reduce_tmp), nullptr, nullptr,
        reinterpret_cast<const int4*>(weight_scale), global_scale, nullptr, nullptr,
        sorted_token_ids, expert_ids, num_tokens_past_padded, nullptr, kTopK, false,
        kHidden / 16, batch_size, kGateUp, kHidden, locks, false, false, true);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const std::uint32_t output_len = routed_rows * kGateUp;
    constexpr int convert_threads = 256;
    const int convert_blocks = (output_len + convert_threads - 1) / convert_threads;
    if (output != nullptr) {
        bf16_to_f32_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
            output_bf16, output, output_len);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t infer_marlin_nvfp4_linear_on_stream(
    const float* input,
    const std::uint32_t* repacked_weight,
    const std::uint8_t* weight_scale,
    const float* global_scale,
    float* output,
    std::uint16_t* input_bf16,
    std::uint16_t* output_bf16,
    float* reduce_tmp,
    std::int32_t* locks,
    std::int32_t* sorted_token_ids,
    std::int32_t* expert_ids,
    std::int32_t* num_tokens_past_padded,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (input == nullptr || repacked_weight == nullptr || weight_scale == nullptr ||
        global_scale == nullptr || output == nullptr || input_bf16 == nullptr ||
        output_bf16 == nullptr || reduce_tmp == nullptr || locks == nullptr ||
        sorted_token_ids == nullptr || expert_ids == nullptr ||
        num_tokens_past_padded == nullptr ||
        !((out_features == 1024 && in_features == 2048) ||
          (out_features == 2048 && in_features == 512))) {
        return cudaErrorInvalidValue;
    }

    prepare_single_input_kernel<<<1, kThreads, 0, stream>>>(
        input, input_bf16, sorted_token_ids, expert_ids,
        num_tokens_past_padded, in_features);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    auto kernel = marlin_kernel();
    const int grid_blocks = out_features / 128;
    kernel<<<grid_blocks, kThreads, kDynamicSharedBytes, stream>>>(
        reinterpret_cast<const int4*>(input_bf16),
        reinterpret_cast<const int4*>(repacked_weight),
        reinterpret_cast<int4*>(output_bf16),
        reinterpret_cast<int4*>(reduce_tmp),
        nullptr,
        nullptr,
        reinterpret_cast<const int4*>(weight_scale),
        global_scale,
        nullptr,
        nullptr,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        nullptr,
        1,
        false,
        in_features / 16,
        1,
        out_features,
        in_features,
        locks,
        false,
        false,
        true);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    constexpr int convert_threads = 256;
    const int convert_blocks = (out_features + convert_threads - 1) / convert_threads;
    bf16_to_f32_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        output_bf16, output, out_features);
    return cudaGetLastError();
}
