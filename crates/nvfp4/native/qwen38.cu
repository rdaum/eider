#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    return value;
}

__global__ void qwen38_hc_norm_kernel(const float* input,
                                      const float* delta_weight,
                                      float* output,
                                      std::uint32_t hidden,
                                      std::uint32_t hc_count,
                                      float eps) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t branch = group % hc_count;
    const std::size_t offset = static_cast<std::size_t>(group) * hidden;
    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = input[offset + col];
        square_sum += value * value;
    }
    square_sum = block_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const std::size_t weight_offset = static_cast<std::size_t>(branch) * hidden;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        output[offset + col] = input[offset + col] * inverse_rms
            * (1.0f + delta_weight[weight_offset + col]);
    }
}

__global__ void qwen38_hc_silu_scale_kernel(float* values,
                                            std::size_t count,
                                            float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < count) {
        const float value = values[index] * scale;
        values[index] = value / (1.0f + expf(-value));
    }
}

__global__ void qwen38_hc_collapse_kernel(const float* normed,
                                          const float* gate_logits,
                                          float* output,
                                          std::uint32_t hidden,
                                          std::uint32_t hc_count,
                                          std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::size_t token = index / hidden;
    const std::size_t col = index % hidden;
    const std::size_t token_offset = token * hc_count * hidden;
    float sum = 0.0f;
    for (std::uint32_t branch = 0; branch < hc_count; ++branch) {
        const std::size_t offset = token_offset
            + static_cast<std::size_t>(branch) * hidden + col;
        const float gate = 1.0f / (1.0f + expf(-gate_logits[offset]));
        sum += gate * normed[offset];
    }
    output[index] = sum / static_cast<float>(hc_count);
}

__global__ void qwen38_hc_combine_kernel(const float* residual,
                                         const float* block_output,
                                         const float* inject_logits,
                                         float* output,
                                         std::uint32_t hidden,
                                         std::uint32_t hc_count,
                                         std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::size_t stream_width = static_cast<std::size_t>(hc_count) * hidden;
    const std::size_t token = index / stream_width;
    const std::size_t within_token = index % stream_width;
    const std::size_t branch = within_token / hidden;
    const std::size_t col = within_token % hidden;
    const float logit = inject_logits[token * hc_count + branch]
        / static_cast<float>(hc_count);
    const float injection = 2.0f / (1.0f + expf(-logit));
    output[index] = residual[index]
        + injection * block_output[token * hidden + col];
}

__global__ void qwen38_repeat_streams_kernel(const float* input,
                                             float* output,
                                             std::uint32_t hidden,
                                             std::uint32_t hc_count,
                                             std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < count) {
        output[index] = input[index % hidden];
    }
}

__global__ void qwen38_ple_gate_value_kernel(const float* key,
                                             const float* query,
                                             const float* value,
                                             float* gated,
                                             std::uint32_t hidden,
                                             std::uint32_t hc_count) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = group / hc_count;
    const std::uint32_t branch = group % hc_count;
    const std::size_t stream_offset = static_cast<std::size_t>(group) * hidden;
    float dot = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        dot += key[stream_offset + col] * query[stream_offset + col];
    }
    dot = block_sum(dot);
    __shared__ float gate;
    if (threadIdx.x == 0) {
        const float scaled = dot * rsqrtf(static_cast<float>(hidden));
        const float signed_root = scaled > 0.0f
            ? sqrtf(fmaxf(scaled, 1e-6f))
            : (scaled < 0.0f ? -sqrtf(fmaxf(-scaled, 1e-6f)) : 0.0f);
        gate = 1.0f / (1.0f + expf(-signed_root));
    }
    __syncthreads();
    const std::size_t value_offset = static_cast<std::size_t>(token) * hidden;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        gated[stream_offset + col] = gate * value[value_offset + col];
    }
}

__global__ void qwen38_ple_conv_update_kernel(const float* normalized,
                                              const float* gated,
                                              const __nv_bfloat16* weight,
                                              float* state,
                                              float* output,
                                              std::uint32_t tokens,
                                              std::uint32_t channels,
                                              std::uint32_t kernel,
                                              std::uint32_t dilation,
                                              std::uint32_t history) {
    const std::uint32_t channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    const std::size_t state_offset = static_cast<std::size_t>(channel) * history;
    const std::size_t weight_offset = static_cast<std::size_t>(channel) * kernel;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        float conv = 0.0f;
        for (std::uint32_t tap = 0; tap < kernel; ++tap) {
            const std::uint32_t lag = (kernel - 1 - tap) * dilation;
            const std::uint32_t centre = history + token;
            const std::uint32_t source = centre - lag;
            const float x = source < history
                ? state[state_offset + source]
                : normalized[static_cast<std::size_t>(source - history) * channels + channel];
            conv += x * __bfloat162float(weight[weight_offset + tap]);
        }
        const float activated = conv / (1.0f + expf(-conv));
        const std::size_t output_offset = static_cast<std::size_t>(token) * channels + channel;
        output[output_offset] = gated[output_offset] + activated;
    }
    for (std::uint32_t position = 0; position < history; ++position) {
        const std::uint32_t source = tokens + position;
        state[state_offset + position] = source < history
            ? state[state_offset + source]
            : normalized[static_cast<std::size_t>(source - history) * channels + channel];
    }
}

}  // namespace

extern "C" cudaError_t infer_qwen38_hc_norm_f32_on_stream(
        const float* input,
        const float* delta_weight,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        float eps,
        cudaStream_t stream) {
    if (input == nullptr || delta_weight == nullptr || output == nullptr
        || tokens == 0 || hidden == 0 || hc_count == 0 || eps <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    qwen38_hc_norm_kernel<<<tokens * hc_count, threads, 0, stream>>>(
        input, delta_weight, output, hidden, hc_count, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_hc_silu_scale_f32_on_stream(
        float* values,
        std::size_t count,
        float scale,
        cudaStream_t stream) {
    if (values == nullptr || count == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_hc_silu_scale_kernel<<<blocks, threads, 0, stream>>>(values, count, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_hc_collapse_f32_on_stream(
        const float* normed,
        const float* gate_logits,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (normed == nullptr || gate_logits == nullptr || output == nullptr
        || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = static_cast<std::size_t>(tokens) * hidden;
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_hc_collapse_kernel<<<blocks, threads, 0, stream>>>(
        normed, gate_logits, output, hidden, hc_count, count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_hc_combine_f32_on_stream(
        const float* residual,
        const float* block_output,
        const float* inject_logits,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (residual == nullptr || block_output == nullptr || inject_logits == nullptr
        || output == nullptr || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = static_cast<std::size_t>(tokens) * hc_count * hidden;
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_hc_combine_kernel<<<blocks, threads, 0, stream>>>(
        residual, block_output, inject_logits, output, hidden, hc_count, count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_repeat_streams_f32_on_stream(
        const float* input,
        float* output,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (input == nullptr || output == nullptr || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = static_cast<std::size_t>(hidden) * hc_count;
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_repeat_streams_kernel<<<blocks, threads, 0, stream>>>(
        input, output, hidden, hc_count, count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_ple_gate_value_f32_on_stream(
        const float* key,
        const float* query,
        const float* value,
        float* gated,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (key == nullptr || query == nullptr || value == nullptr || gated == nullptr
        || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    qwen38_ple_gate_value_kernel<<<tokens * hc_count, threads, 0, stream>>>(
        key, query, value, gated, hidden, hc_count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_ple_conv_update_f32_on_stream(
        const float* normalized,
        const float* gated,
        const std::uint16_t* weight_bf16,
        float* state,
        float* output,
        std::uint32_t tokens,
        std::uint32_t channels,
        std::uint32_t kernel,
        std::uint32_t dilation,
        cudaStream_t stream) {
    if (normalized == nullptr || gated == nullptr || weight_bf16 == nullptr
        || state == nullptr || output == nullptr || tokens == 0 || channels == 0
        || kernel < 2 || dilation == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t history = (kernel - 1) * dilation;
    constexpr int threads = 256;
    const unsigned int blocks = (channels + threads - 1) / threads;
    qwen38_ple_conv_update_kernel<<<blocks, threads, 0, stream>>>(
        normalized,
        gated,
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
        state,
        output,
        tokens,
        channels,
        kernel,
        dilation,
        history);
    return cudaGetLastError();
}
