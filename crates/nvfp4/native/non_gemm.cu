#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cub/block/block_radix_sort.cuh>
#include <cfloat>
#include <cstdint>

// This translation unit is grouped by operation family; see native/README.md.
// Keep the Rust-facing extern "C" wrappers close to the kernels they launch.

__device__ std::uint32_t infer_round_up_u32(std::uint32_t value,
                                                  std::uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

__device__ std::uint32_t infer_ue4m3_tiled_scale_offset(std::uint32_t outer,
                                                              std::uint32_t inner_block,
                                                              std::uint32_t inner_dim) {
    const std::uint32_t inner_scale_blocks = (inner_dim + 15) / 16;
    const std::uint32_t sf_inner_dim = infer_round_up_u32(inner_scale_blocks, 4);
    const std::uint32_t tile_outer = outer / 128;
    const std::uint32_t outer_in_tile = outer % 128;
    const std::uint32_t tile_inner = inner_block / 4;
    const std::uint32_t inner_in_tile = inner_block % 4;
    const std::uint32_t tile_base = (tile_inner * 4 + tile_outer * sf_inner_dim) * 128;
    return tile_base + (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile;
}

__device__ __forceinline__ float infer_e4m3_value(std::uint8_t code) {
    const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80) << 24;
    const std::uint32_t exp = (code >> 3) & 0x0f;
    const std::uint32_t mant = code & 0x07;
    if (exp == 0) {
        const float value = static_cast<float>(mant) * 0x1p-9f;
        return sign == 0 ? value : -value;
    }
    if (exp == 0x0f && mant == 0x07) {
        return __uint_as_float(sign | 0x7fffffffU);
    }
    return __uint_as_float(sign | ((exp + 120U) << 23) | (mant << 20));
}

__device__ __forceinline__ float infer_warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}

__device__ __forceinline__ float infer_block_reduce_sum(float value) {
    __shared__ float warp_sums[32];
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5;
    value = infer_warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31U) >> 5) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = infer_warp_reduce_sum(value);
    }
    return value;
}

__device__ __forceinline__ float infer_warp_reduce_max(float value) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 1));
    return value;
}

__device__ __forceinline__ float infer_block_reduce_max(float value) {
    __shared__ float warp_maxes[32];
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5;
    value = infer_warp_reduce_max(value);
    if (lane == 0) {
        warp_maxes[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31U) >> 5) ? warp_maxes[lane] : 0.0f;
    if (warp == 0) {
        value = infer_warp_reduce_max(value);
    }
    return value;
}

// Quantization and elementwise activation kernels.
__global__ void infer_quantize_nvfp4_col_major_f32_kernel(const float* input,
                                                               std::uint8_t* packed,
                                                               std::uint8_t* scales,
                                                               std::uint32_t rows,
                                                               std::uint32_t cols,
                                                               float input_scale) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t row_blocks = (rows + 15) / 16;
    const std::uint32_t col = group / row_blocks;
    const std::uint32_t row_block = group % row_blocks;
    if (col >= cols || threadIdx.x != 0) {
        return;
    }

    const std::uint32_t row_start = row_block * 16;
    const std::uint32_t row_end = min(row_start + 16, rows);
    float max_abs = 0.0f;
    for (std::uint32_t row = row_start; row < row_end; ++row) {
        const float value = input[row + col * rows] / input_scale;
        if (isfinite(value)) {
            max_abs = fmaxf(max_abs, fabsf(value));
        }
    }

    const std::uint8_t scale_code =
        max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                                  __nv_cvt_float_to_fp8(max_abs / 6.0f,
                                                        __NV_SATFINITE,
                                                        __NV_E4M3));
    const float scale = infer_e4m3_value(scale_code);
    scales[infer_ue4m3_tiled_scale_offset(col, row_block, rows)] = scale_code;

    for (std::uint32_t row = row_start; row < row_end; row += 2) {
        const float lo_value = scale == 0.0f ? 0.0f : (input[row + col * rows] / input_scale) / scale;
        const std::uint8_t lo =
            static_cast<std::uint8_t>(__nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        std::uint8_t hi = 0;
        if (row + 1 < row_end) {
            const float hi_value =
                scale == 0.0f ? 0.0f : (input[row + 1 + col * rows] / input_scale) / scale;
            hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        }
        packed[(row + col * rows) / 2] = lo | (hi << 4);
    }
}

extern "C" cudaError_t infer_quantize_nvfp4_col_major_f32(const float* input,
                                                                std::uint8_t* packed,
                                                                std::uint8_t* scales,
                                                                std::uint32_t rows,
                                                                std::uint32_t cols,
                                                                float input_scale) {
    if (input == nullptr || packed == nullptr || scales == nullptr || rows == 0 || cols == 0 ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }

    const std::uint32_t row_blocks = (rows + 15) / 16;
    infer_quantize_nvfp4_col_major_f32_kernel<<<cols * row_blocks, 1>>>(
        input, packed, scales, rows, cols, input_scale);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_quantize_nvfp4_col_major_f32_on_stream(
    const float* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || packed == nullptr || scales == nullptr || rows == 0 || cols == 0 ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }

    const std::uint32_t row_blocks = (rows + 15) / 16;
    infer_quantize_nvfp4_col_major_f32_kernel<<<cols * row_blocks, 1, 0, stream>>>(
        input, packed, scales, rows, cols, input_scale);
    return cudaGetLastError();
}

__global__ void infer_quantize_nvfp4_vector_simple_scales_f32_kernel(
    const float* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    float input_scale) {
    const std::uint32_t row_block = blockIdx.x;
    if (threadIdx.x != 0) return;
    const std::uint32_t row_start = row_block * 16;
    const std::uint32_t row_end = min(row_start + 16, rows);
    float max_abs = 0.0f;
    for (std::uint32_t row = row_start; row < row_end; ++row) {
        const float value = input[row] / input_scale;
        if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
    }
    const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float scale = infer_e4m3_value(scale_code);
    scales[row_block] = scale_code;
    for (std::uint32_t row = row_start; row < row_end; row += 2) {
        const float lo_value = scale == 0.0f ? 0.0f : (input[row] / input_scale) / scale;
        const std::uint8_t lo = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        std::uint8_t hi = 0;
        if (row + 1 < row_end) {
            const float hi_value = scale == 0.0f ? 0.0f : (input[row + 1] / input_scale) / scale;
            hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        }
        packed[row / 2] = lo | (hi << 4);
    }
}

extern "C" cudaError_t infer_quantize_nvfp4_vector_simple_scales_f32_on_stream(
    const float* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || packed == nullptr || scales == nullptr || rows == 0 ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    infer_quantize_nvfp4_vector_simple_scales_f32_kernel<<<(rows + 15) / 16, 1, 0, stream>>>(
        input, packed, scales, rows, input_scale);
    return cudaGetLastError();
}

__global__ void infer_silu_mul_halves_quantize_nvfp4_col_major_f32_kernel(
    const float* gate_up,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    float input_scale) {
    const std::uint32_t row_block = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const std::uint32_t row_start = row_block * 16;
    const std::uint32_t row_end = min(row_start + 16, rows);
    float max_abs = 0.0f;
    for (std::uint32_t row = row_start; row < row_end; ++row) {
        const float gate_value = gate_up[row];
        const float up_value = gate_up[rows + row];
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float value = (gate_value * sigmoid * up_value) / input_scale;
        if (isfinite(value)) {
            max_abs = fmaxf(max_abs, fabsf(value));
        }
    }

    const std::uint8_t scale_code =
        max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                                  __nv_cvt_float_to_fp8(max_abs / 6.0f,
                                                        __NV_SATFINITE,
                                                        __NV_E4M3));
    const float scale = infer_e4m3_value(scale_code);
    scales[infer_ue4m3_tiled_scale_offset(0, row_block, rows)] = scale_code;

    for (std::uint32_t row = row_start; row < row_end; row += 2) {
        const float lo_gate = gate_up[row];
        const float lo_up = gate_up[rows + row];
        const float lo_sigmoid = 1.0f / (1.0f + expf(-lo_gate));
        const float lo_activated = lo_gate * lo_sigmoid * lo_up;
        const float lo_value = scale == 0.0f ? 0.0f : (lo_activated / input_scale) / scale;
        const std::uint8_t lo =
            static_cast<std::uint8_t>(__nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        std::uint8_t hi = 0;
        if (row + 1 < row_end) {
            const float hi_gate = gate_up[row + 1];
            const float hi_up = gate_up[rows + row + 1];
            const float hi_sigmoid = 1.0f / (1.0f + expf(-hi_gate));
            const float hi_activated = hi_gate * hi_sigmoid * hi_up;
            const float hi_value = scale == 0.0f ? 0.0f : (hi_activated / input_scale) / scale;
            hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        }
        packed[row / 2] = lo | (hi << 4);
    }
}

extern "C" cudaError_t infer_silu_mul_halves_quantize_nvfp4_col_major_f32_on_stream(
    const float* gate_up,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    float input_scale,
    cudaStream_t stream) {
    if (gate_up == nullptr || packed == nullptr || scales == nullptr || rows == 0 ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }

    const std::uint32_t row_blocks = (rows + 15) / 16;
    infer_silu_mul_halves_quantize_nvfp4_col_major_f32_kernel<<<row_blocks, 1, 0, stream>>>(
        gate_up, packed, scales, rows, input_scale);
    return cudaGetLastError();
}

__global__ void infer_rms_norm_f32_kernel(const float* input,
                                                const float* weight,
                                                float* output,
                                                std::uint32_t rows,
                                                std::uint32_t cols,
                                                float eps) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = row_input[col];
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(cols) + eps);
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        row_output[col] = row_input[col] * inv_rms * weight[col];
    }
}

extern "C" cudaError_t infer_rms_norm_f32(const float* input,
                                                const float* weight,
                                                float* output,
                                                std::uint32_t rows,
                                                std::uint32_t cols,
                                                float eps) {
    if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    infer_rms_norm_f32_kernel<<<rows, kThreads, kThreads * sizeof(float)>>>(
        input, weight, output, rows, cols, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_rms_norm_f32_on_stream(const float* input,
                                                          const float* weight,
                                                          float* output,
                                                          std::uint32_t rows,
                                                          std::uint32_t cols,
                                                          float eps,
                                                          cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    infer_rms_norm_f32_kernel<<<rows, kThreads, kThreads * sizeof(float), stream>>>(
        input, weight, output, rows, cols, eps);
    return cudaGetLastError();
}

__global__ void infer_rms_norm_rope_neox_f32_indexed_kernel(
    const float* input,
    const float* weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    const std::uint32_t* position,
    float theta,
    float eps) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * head_dim;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
        const float value = row_input[col];
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    const std::uint32_t half = head_dim / 2;
    for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        const float inv_freq = powf(theta, -2.0f * static_cast<float>(i) /
                                              static_cast<float>(head_dim));
        float sin_value;
        float cos_value;
        sincosf(static_cast<float>(*position) * inv_freq, &sin_value, &cos_value);
        const std::uint32_t row_start = row * head_dim;
        const float a = row_input[i] * inv_rms * weight[i];
        const float b = row_input[i + half] * inv_rms * weight[i + half];
        output[row_start + i] = a * cos_value - b * sin_value;
        output[row_start + i + half] = a * sin_value + b * cos_value;
    }
}

extern "C" cudaError_t infer_rms_norm_rope_neox_f32_indexed_on_stream(
    const float* input,
    const float* weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    const std::uint32_t* position,
    float theta,
    float eps,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || position == nullptr ||
        rows == 0 || head_dim == 0 || (head_dim % 2) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_rms_norm_rope_neox_f32_indexed_kernel<<<
        rows, kThreads, kThreads * sizeof(float), stream>>>(
        input, weight, output, rows, head_dim, position, theta, eps);
    return cudaGetLastError();
}

__global__ void infer_silu_mul_f32_kernel(const float* gate,
                                                const float* up,
                                                float* output,
                                                std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    const float gate_value = gate[idx];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[idx] = gate_value * sigmoid * up[idx];
}

extern "C" cudaError_t infer_silu_mul_f32(const float* gate,
                                                const float* up,
                                                float* output,
                                                std::uint32_t len) {
    if (gate == nullptr || up == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_silu_mul_f32_kernel<<<blocks, kThreads>>>(gate, up, output, len);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_silu_mul_f32_on_stream(const float* gate,
                                                          const float* up,
                                                          float* output,
                                                          std::uint32_t len,
                                                          cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_silu_mul_f32_kernel<<<blocks, kThreads, 0, stream>>>(gate, up, output, len);
    return cudaGetLastError();
}

__global__ void infer_silu_mul_halves_f32_kernel(const float* gate_up,
                                                       float* output,
                                                       std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    const float gate_value = gate_up[idx];
    const float up_value = gate_up[len + idx];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[idx] = gate_value * sigmoid * up_value;
}

extern "C" cudaError_t infer_silu_mul_halves_f32_on_stream(const float* gate_up,
                                                                 float* output,
                                                                 std::uint32_t len,
                                                                 cudaStream_t stream) {
    if (gate_up == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_silu_mul_halves_f32_kernel<<<blocks, kThreads, 0, stream>>>(gate_up, output, len);
    return cudaGetLastError();
}

__global__ void infer_silu_mul_halves_f32_batch_kernel(
    const float* gate_up,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) return;
    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx - row * cols;
    const std::uint32_t base = row * cols * 2;
    const float gate = gate_up[base + col];
    output[idx] = (gate / (1.0f + expf(-gate))) * gate_up[base + cols + col];
}

extern "C" cudaError_t infer_silu_mul_halves_f32_batch_on_stream(
    const float* gate_up,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (gate_up == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t len = static_cast<std::uint64_t>(rows) * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_silu_mul_halves_f32_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        gate_up, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_fill_f32_kernel(float* output, float value, std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = value;
    }
}

extern "C" cudaError_t infer_fill_f32_on_stream(float* output,
                                                       float value,
                                                       std::uint32_t len,
                                                       cudaStream_t stream) {
    if (output == nullptr || len == 0 || !isfinite(value)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_fill_f32_kernel<<<blocks, kThreads, 0, stream>>>(output, value, len);
    return cudaGetLastError();
}

__global__ void infer_scaled_add_f32_kernel(const float* input,
                                                  float* output,
                                                  float scale,
                                                  std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] += input[idx] * scale;
    }
}

extern "C" cudaError_t infer_scaled_add_f32_on_stream(const float* input,
                                                            float* output,
                                                            float scale,
                                                            std::uint32_t len,
                                                            cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0 || !isfinite(scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_scaled_add_f32_kernel<<<blocks, kThreads, 0, stream>>>(input, output, scale, len);
    return cudaGetLastError();
}

__global__ void infer_split_q_gate_f32_kernel(const float* input,
                                                    float* q,
                                                    float* gate,
                                                    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    q[idx] = input[idx];
    gate[idx] = input[len + idx];
}

extern "C" cudaError_t infer_split_q_gate_f32_on_stream(const float* input,
                                                              float* q,
                                                              float* gate,
                                                              std::uint32_t len,
                                                              cudaStream_t stream) {
    if (input == nullptr || q == nullptr || gate == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_split_q_gate_f32_kernel<<<blocks, kThreads, 0, stream>>>(input, q, gate, len);
    return cudaGetLastError();
}

__global__ void infer_sigmoid_mul_f32_kernel(const float* gate,
                                                   const float* input,
                                                   float* output,
                                                   std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const float gate_value = gate[idx];
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    output[idx] = input[idx] * sigmoid;
}

extern "C" cudaError_t infer_sigmoid_mul_f32_on_stream(const float* gate,
                                                             const float* input,
                                                             float* output,
                                                             std::uint32_t len,
                                                             cudaStream_t stream) {
    if (gate == nullptr || input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_sigmoid_mul_f32_kernel<<<blocks, kThreads, 0, stream>>>(gate, input, output, len);
    return cudaGetLastError();
}

// Fused shared-expert gate: reads a single scalar logit from `gate_logit[0]`,
// computes sigmoid(scalar), and multiplies it element-wise into `input`,
// writing `shared_gated`. Replaces a host readback + broadcast + sigmoid_mul
// sequence that forced a stream sync on every token.
__global__ void infer_sigmoid_scale_scalar_f32_kernel(const float* gate_logit,
                                                            const float* input,
                                                            float* output,
                                                            std::uint32_t len) {
    const float scalar = 1.0f / (1.0f + expf(-gate_logit[0]));
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = input[idx] * scalar;
    }
}

extern "C" cudaError_t infer_sigmoid_scale_scalar_f32_on_stream(
    const float* gate_logit,
    const float* input,
    float* output,
    std::uint32_t len,
    cudaStream_t stream) {
    if (gate_logit == nullptr || input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_sigmoid_scale_scalar_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        gate_logit, input, output, len);
    return cudaGetLastError();
}

__global__ void infer_split_qkv_f32_kernel(const float* input,
                                                 float* q,
                                                 float* k,
                                                 float* v,
                                                 std::uint32_t q_len,
                                                 std::uint32_t kv_len) {
    const std::uint32_t total = q_len + kv_len + kv_len;
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    if (idx < q_len) {
        q[idx] = input[idx];
    } else if (idx < q_len + kv_len) {
        k[idx - q_len] = input[idx];
    } else {
        v[idx - q_len - kv_len] = input[idx];
    }
}

extern "C" cudaError_t infer_split_qkv_f32_on_stream(const float* input,
                                                           float* q,
                                                           float* k,
                                                           float* v,
                                                           std::uint32_t q_len,
                                                           std::uint32_t kv_len,
                                                           cudaStream_t stream) {
    if (input == nullptr || q == nullptr || k == nullptr || v == nullptr || q_len == 0 ||
        kv_len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t total = q_len + kv_len + kv_len;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_split_qkv_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, q, k, v, q_len, kv_len);
    return cudaGetLastError();
}

__global__ void infer_qwen36_full_attn_prep_f32_kernel(
    const float* q_full,
    const float* k_raw,
    const float* q_norm,
    const float* k_norm,
    float* q,
    float* gate,
    float* k,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float eps) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (lane >= head_dim) {
        return;
    }

    if (row < q_heads) {
        const std::uint32_t q_full_base = row * head_dim * 2;
        const std::uint32_t out_base = row * head_dim;
        const float value = q_full[q_full_base + lane];
        gate[out_base + lane] = q_full[q_full_base + head_dim + lane];
        partial[lane] = value * value;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
        q[out_base + lane] = value * inv_rms * q_norm[lane];
        return;
    }

    const std::uint32_t kv_row = row - q_heads;
    if (kv_row >= kv_heads) {
        return;
    }
    const std::uint32_t base = kv_row * head_dim;
    const float value = k_raw[base + lane];
    partial[lane] = value * value;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    k[base + lane] = value * inv_rms * k_norm[lane];
}

extern "C" cudaError_t infer_qwen36_full_attn_prep_f32_on_stream(
    const float* q_full,
    const float* k_raw,
    const float* q_norm,
    const float* k_norm,
    float* q,
    float* gate,
    float* k,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float eps,
    cudaStream_t stream) {
    if (q_full == nullptr || k_raw == nullptr || q_norm == nullptr || k_norm == nullptr ||
        q == nullptr || gate == nullptr || k == nullptr || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_qwen36_full_attn_prep_f32_kernel<<<
        q_heads + kv_heads,
        head_dim,
        head_dim * sizeof(float),
        stream>>>(q_full, k_raw, q_norm, k_norm, q, gate, k, q_heads, kv_heads, head_dim, eps);
    return cudaGetLastError();
}

__global__ void infer_qwen36_full_attn_prep_f32_batch_kernel(
    const float* q_full,
    const float* k_raw,
    const float* q_norm,
    const float* k_norm,
    float* q,
    float* gate,
    float* k,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float eps) {
    extern __shared__ float partial[];
    const std::uint32_t heads_per_batch = q_heads + kv_heads;
    const std::uint32_t batch = blockIdx.x / heads_per_batch;
    const std::uint32_t row = blockIdx.x - batch * heads_per_batch;
    const std::uint32_t lane = threadIdx.x;
    if (lane >= head_dim) return;

    if (row < q_heads) {
        const std::uint32_t q_width = q_heads * head_dim;
        const std::uint32_t q_full_base = batch * q_width * 2 + row * head_dim * 2;
        const std::uint32_t out_base = batch * q_width + row * head_dim;
        const float value = q_full[q_full_base + lane];
        gate[out_base + lane] = q_full[q_full_base + head_dim + lane];
        partial[lane] = value * value;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
        q[out_base + lane] = value * inv_rms * q_norm[lane];
        return;
    }
    const std::uint32_t kv_row = row - q_heads;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const std::uint32_t base = batch * kv_width + kv_row * head_dim;
    const float value = k_raw[base + lane];
    partial[lane] = value * value;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    k[base + lane] = value * inv_rms * k_norm[lane];
}

extern "C" cudaError_t infer_qwen36_full_attn_prep_f32_batch_on_stream(
    const float* q_full,
    const float* k_raw,
    const float* q_norm,
    const float* k_norm,
    float* q,
    float* gate,
    float* k,
    std::uint32_t batch_size,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float eps,
    cudaStream_t stream) {
    if (q_full == nullptr || k_raw == nullptr || q_norm == nullptr || k_norm == nullptr ||
        q == nullptr || gate == nullptr || k == nullptr || batch_size == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_qwen36_full_attn_prep_f32_batch_kernel<<<
        batch_size * (q_heads + kv_heads), head_dim, head_dim * sizeof(float), stream>>>(
        q_full, k_raw, q_norm, k_norm, q, gate, k,
        q_heads, kv_heads, head_dim, eps);
    return cudaGetLastError();
}

// MoE routing, grouped pointer gathering, and routed activation kernels.
__global__ void infer_moe_topk_f32_kernel(const float* logits,
                                                std::uint32_t* out_indices,
                                                float* out_weights,
                                                std::uint32_t experts,
                                                std::uint32_t k,
                                                bool norm_topk_prob) {
    const std::uint32_t batch = blockIdx.x;
    logits += batch * experts;
    out_indices += batch * k;
    out_weights += batch * k;

    if (norm_topk_prob) {
        if (threadIdx.x == 0) {
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                out_indices[slot] = UINT32_MAX;
                out_weights[slot] = -INFINITY;
            }
            for (std::uint32_t expert = 0; expert < experts; ++expert) {
                float value = logits[expert];
                if (isnan(value)) {
                    value = -INFINITY;
                } else if (value == INFINITY) {
                    value = FLT_MAX;
                }
                for (std::uint32_t slot = 0; slot < k; ++slot) {
                    if (value > out_weights[slot]) {
                        for (std::uint32_t move = k - 1; move > slot; --move) {
                            out_indices[move] = out_indices[move - 1];
                            out_weights[move] = out_weights[move - 1];
                        }
                        out_indices[slot] = expert;
                        out_weights[slot] = value;
                        break;
                    }
                }
            }
            if (!isfinite(out_weights[0])) {
                for (std::uint32_t slot = 0; slot < k; ++slot) {
                    out_indices[slot] = slot;
                    out_weights[slot] = slot == 0 ? 1.0f : 0.0f;
                }
                return;
            }
            const float selected_max = out_weights[0];
            float selected_sum = 0.0f;
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                const float prob = expf(out_weights[slot] - selected_max);
                out_weights[slot] = prob;
                selected_sum += prob;
            }
            if (selected_sum > 0.0f && isfinite(selected_sum)) {
                for (std::uint32_t slot = 0; slot < k; ++slot) {
                    out_weights[slot] /= selected_sum;
                }
            }
        }
        return;
    }

    __shared__ float partial[256];
    __shared__ float probs[1024];
    __shared__ float max_logit_shared;
    __shared__ float prob_sum_shared;

    float local_max = -INFINITY;
    for (std::uint32_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        float value = logits[expert];
        if (isnan(value)) {
            value = -INFINITY;
        } else if (value == INFINITY) {
            value = FLT_MAX;
        }
        local_max = fmaxf(local_max, value);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_logit_shared = partial[0];
    }
    __syncthreads();

    const float max_logit = max_logit_shared;
    if (!isfinite(max_logit)) {
        if (threadIdx.x == 0) {
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                out_indices[slot] = slot;
                out_weights[slot] = slot == 0 ? 1.0f : 0.0f;
            }
        }
        return;
    }

    float local_sum = 0.0f;
    for (std::uint32_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        float value = logits[expert];
        if (isnan(value)) {
            value = -INFINITY;
        } else if (value == INFINITY) {
            value = FLT_MAX;
        }
        const float prob = expf(value - max_logit);
        probs[expert] = prob;
        local_sum += prob;
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        prob_sum_shared = partial[0];
    }
    __syncthreads();

    const float prob_sum = prob_sum_shared;
    for (std::uint32_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        probs[expert] = probs[expert] / prob_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (std::uint32_t slot = 0; slot < k; ++slot) {
            out_indices[slot] = UINT32_MAX;
            out_weights[slot] = -INFINITY;
        }

        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            const float prob = probs[expert];
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                if (prob > out_weights[slot]) {
                    for (std::uint32_t move = k - 1; move > slot; --move) {
                        out_indices[move] = out_indices[move - 1];
                        out_weights[move] = out_weights[move - 1];
                    }
                    out_indices[slot] = expert;
                    out_weights[slot] = prob;
                    break;
                }
            }
        }

        if (norm_topk_prob) {
            float selected_sum = 0.0f;
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                selected_sum += out_weights[slot];
            }
            if (selected_sum > 0.0f && isfinite(selected_sum)) {
                for (std::uint32_t slot = 0; slot < k; ++slot) {
                    out_weights[slot] /= selected_sum;
                }
            }
        }
    }
}

__global__ void infer_moe_top8_norm256_f32_kernel(const float* logits,
                                                        std::uint32_t* out_indices,
                                                        float* out_weights) {
    const std::uint32_t batch = blockIdx.x;
    logits += batch * 256;
    out_indices += batch * 8;
    out_weights += batch * 8;
    const std::uint32_t expert = threadIdx.x;
    float value = logits[expert];
    if (isnan(value)) {
        value = -INFINITY;
    } else if (value == INFINITY) {
        value = FLT_MAX;
    }
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t ordered = (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
    std::uint64_t keys[1] = {
        (static_cast<std::uint64_t>(ordered) << 32) |
        static_cast<std::uint64_t>(UINT32_MAX - expert),
    };
    float sorted_values[1] = {value};

    using BlockSort = cub::BlockRadixSort<std::uint64_t, 256, 1, float>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ float top_values[8];
    __shared__ std::uint32_t top_indices[8];
    BlockSort(sort_storage).SortDescending(keys, sorted_values);
    __syncthreads();

    if (expert < 8) {
        top_values[expert] = sorted_values[0];
        top_indices[expert] = UINT32_MAX - static_cast<std::uint32_t>(keys[0]);
    }
    __syncthreads();

    if (expert == 0) {
        if (!isfinite(top_values[0])) {
            #pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                out_indices[slot] = slot;
                out_weights[slot] = slot == 0 ? 1.0f : 0.0f;
            }
            return;
        }

        const float selected_max = top_values[0];
        float selected_sum = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const float prob = expf(top_values[slot] - selected_max);
            top_values[slot] = prob;
            selected_sum += prob;
        }
        if (selected_sum > 0.0f && isfinite(selected_sum)) {
            #pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                out_indices[slot] = top_indices[slot];
                out_weights[slot] = top_values[slot] / selected_sum;
            }
        }
    }
}

extern "C" cudaError_t infer_moe_topk_f32_on_stream(const float* logits,
                                                          std::uint32_t* out_indices,
                                                          float* out_weights,
                                                          std::uint32_t experts,
                                                          std::uint32_t k,
                                                          int norm_topk_prob,
                                                          cudaStream_t stream) {
    if (logits == nullptr || out_indices == nullptr || out_weights == nullptr || experts == 0 ||
        experts > 1024 || k == 0 || k > experts) {
        return cudaErrorInvalidValue;
    }
    if (experts == 256 && k == 8 && norm_topk_prob != 0) {
        infer_moe_top8_norm256_f32_kernel<<<1, 256, 0, stream>>>(
            logits, out_indices, out_weights);
        return cudaGetLastError();
    }
    const int threads = norm_topk_prob != 0 ? 1 : 256;
    infer_moe_topk_f32_kernel<<<1, threads, 0, stream>>>(
        logits, out_indices, out_weights, experts, k, norm_topk_prob != 0);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_moe_topk_f32_batch_on_stream(
    const float* logits,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t batch_size,
    std::uint32_t experts,
    std::uint32_t k,
    int norm_topk_prob,
    cudaStream_t stream) {
    if (logits == nullptr || out_indices == nullptr || out_weights == nullptr ||
        batch_size == 0 || experts == 0 || experts > 1024 || k == 0 || k > experts) {
        return cudaErrorInvalidValue;
    }
    if (experts == 256 && k == 8 && norm_topk_prob != 0) {
        infer_moe_top8_norm256_f32_kernel<<<batch_size, 256, 0, stream>>>(
            logits, out_indices, out_weights);
        return cudaGetLastError();
    }
    const int threads = norm_topk_prob != 0 ? 1 : 256;
    infer_moe_topk_f32_kernel<<<batch_size, threads, 0, stream>>>(
        logits, out_indices, out_weights, experts, k, norm_topk_prob != 0);
    return cudaGetLastError();
}

__global__ void infer_gather_nvfp4_grouped_gemv_ptrs_kernel(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_values_table,
    const std::uint8_t* const* a_scales_table,
    const std::uint8_t* b_values,
    const std::uint8_t* b_scales,
    const float* const* c_table,
    float* const* d_table,
    const std::uint32_t groups,
    const std::uint32_t table_len,
    const std::uint8_t** out_a_values,
    const std::uint8_t** out_a_scales,
    const std::uint8_t** out_b_values,
    const std::uint8_t** out_b_scales,
    const float** out_c,
    float** out_d) {
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= groups) {
        return;
    }
    const std::uint32_t expert = indices[slot];
    if (expert >= table_len) {
        return;
    }
    out_a_values[slot] = a_values_table[expert];
    out_a_scales[slot] = a_scales_table[expert];
    out_b_values[slot] = b_values;
    out_b_scales[slot] = b_scales;
    out_c[slot] = c_table[slot];
    out_d[slot] = d_table[slot];
}

extern "C" cudaError_t infer_gather_nvfp4_grouped_gemv_ptrs_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_values_table,
    const std::uint8_t* const* a_scales_table,
    const std::uint8_t* b_values,
    const std::uint8_t* b_scales,
    const float* const* c_table,
    float* const* d_table,
    std::uint32_t groups,
    std::uint32_t table_len,
    const std::uint8_t** out_a_values,
    const std::uint8_t** out_a_scales,
    const std::uint8_t** out_b_values,
    const std::uint8_t** out_b_scales,
    const float** out_c,
    float** out_d,
    cudaStream_t stream) {
    if (indices == nullptr || a_values_table == nullptr || a_scales_table == nullptr ||
        b_values == nullptr || b_scales == nullptr || c_table == nullptr || d_table == nullptr ||
        out_a_values == nullptr || out_a_scales == nullptr || out_b_values == nullptr ||
        out_b_scales == nullptr || out_c == nullptr || out_d == nullptr || groups == 0 ||
        table_len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 32;
    const int blocks = static_cast<int>((groups + kThreads - 1) / kThreads);
    infer_gather_nvfp4_grouped_gemv_ptrs_kernel<<<blocks, kThreads, 0, stream>>>(
        indices,
        a_values_table,
        a_scales_table,
        b_values,
        b_scales,
        c_table,
        d_table,
        groups,
        table_len,
        out_a_values,
        out_a_scales,
        out_b_values,
        out_b_scales,
        out_c,
        out_d);
    return cudaGetLastError();
}

__global__ void infer_gather_nvfp4_grouped_gemv_ptr_tables_kernel(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_values_table,
    const std::uint8_t* const* a_scales_table,
    const std::uint8_t* const* b_values_table,
    const std::uint8_t* const* b_scales_table,
    const float* const* c_table,
    float* const* d_table,
    const std::uint32_t groups,
    const std::uint32_t table_len,
    const std::uint8_t** out_a_values,
    const std::uint8_t** out_a_scales,
    const std::uint8_t** out_b_values,
    const std::uint8_t** out_b_scales,
    const float** out_c,
    float** out_d) {
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= groups) {
        return;
    }
    const std::uint32_t expert = indices[slot];
    if (expert >= table_len) {
        return;
    }
    out_a_values[slot] = a_values_table[expert];
    out_a_scales[slot] = a_scales_table[expert];
    out_b_values[slot] = b_values_table[slot];
    out_b_scales[slot] = b_scales_table[slot];
    out_c[slot] = c_table[slot];
    out_d[slot] = d_table[slot];
}

extern "C" cudaError_t infer_gather_nvfp4_grouped_gemv_ptr_tables_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_values_table,
    const std::uint8_t* const* a_scales_table,
    const std::uint8_t* const* b_values_table,
    const std::uint8_t* const* b_scales_table,
    const float* const* c_table,
    float* const* d_table,
    std::uint32_t groups,
    std::uint32_t table_len,
    const std::uint8_t** out_a_values,
    const std::uint8_t** out_a_scales,
    const std::uint8_t** out_b_values,
    const std::uint8_t** out_b_scales,
    const float** out_c,
    float** out_d,
    cudaStream_t stream) {
    if (indices == nullptr || a_values_table == nullptr || a_scales_table == nullptr ||
        b_values_table == nullptr || b_scales_table == nullptr || c_table == nullptr ||
        d_table == nullptr || out_a_values == nullptr || out_a_scales == nullptr ||
        out_b_values == nullptr || out_b_scales == nullptr || out_c == nullptr ||
        out_d == nullptr || groups == 0 || table_len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 32;
    const int blocks = static_cast<int>((groups + kThreads - 1) / kThreads);
    infer_gather_nvfp4_grouped_gemv_ptr_tables_kernel<<<blocks, kThreads, 0, stream>>>(
        indices,
        a_values_table,
        a_scales_table,
        b_values_table,
        b_scales_table,
        c_table,
        d_table,
        groups,
        table_len,
        out_a_values,
        out_a_scales,
        out_b_values,
        out_b_scales,
        out_c,
        out_d);
    return cudaGetLastError();
}

__global__ void infer_moe_silu_quantize_slots_nvfp4_kernel(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* const* packed_table,
    std::uint8_t* const* scales_table,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups) {
    const std::uint32_t slot = blockIdx.x;
    const std::uint32_t row_block = blockIdx.y;
    if (slot >= groups || threadIdx.x != 0) {
        return;
    }
    const std::uint32_t expert = indices[slot];
    const float input_scale = input_scale_table[expert];
    if (input_scale <= 0.0f || !isfinite(input_scale)) {
        return;
    }
    const float gate_up_alpha = gate_up_alpha_table[expert];

    const float* gate_up = gate_up_table[slot];
    std::uint8_t* packed = packed_table[slot];
    std::uint8_t* scales = scales_table[slot];
    const std::uint32_t row_start = row_block * 16;
    const std::uint32_t row_end = min(row_start + 16, rows);
    float max_abs = 0.0f;
    for (std::uint32_t row = row_start; row < row_end; ++row) {
        const float gate_value = gate_up[row] * gate_up_alpha;
        const float up_value = gate_up[rows + row] * gate_up_alpha;
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float value = (gate_value * sigmoid * up_value) / input_scale;
        if (isfinite(value)) {
            max_abs = fmaxf(max_abs, fabsf(value));
        }
    }

    const std::uint8_t scale_code =
        max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                                  __nv_cvt_float_to_fp8(max_abs / 6.0f,
                                                        __NV_SATFINITE,
                                                        __NV_E4M3));
    const float scale = infer_e4m3_value(scale_code);
    scales[infer_ue4m3_tiled_scale_offset(0, row_block, rows)] = scale_code;

    for (std::uint32_t row = row_start; row < row_end; row += 2) {
        const float lo_gate = gate_up[row] * gate_up_alpha;
        const float lo_up = gate_up[rows + row] * gate_up_alpha;
        const float lo_sigmoid = 1.0f / (1.0f + expf(-lo_gate));
        const float lo_activated = lo_gate * lo_sigmoid * lo_up;
        const float lo_value = scale == 0.0f ? 0.0f : (lo_activated / input_scale) / scale;
        const std::uint8_t lo =
            static_cast<std::uint8_t>(__nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        std::uint8_t hi = 0;
        if (row + 1 < row_end) {
            const float hi_gate = gate_up[row + 1] * gate_up_alpha;
            const float hi_up = gate_up[rows + row + 1] * gate_up_alpha;
            const float hi_sigmoid = 1.0f / (1.0f + expf(-hi_gate));
            const float hi_activated = hi_gate * hi_sigmoid * hi_up;
            const float hi_value = scale == 0.0f ? 0.0f : (hi_activated / input_scale) / scale;
            hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        }
        packed[row / 2] = lo | (hi << 4);
    }
}

extern "C" cudaError_t infer_moe_silu_quantize_slots_nvfp4_on_stream(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* const* packed_table,
    std::uint8_t* const* scales_table,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (indices == nullptr || gate_up_table == nullptr || packed_table == nullptr ||
        scales_table == nullptr || input_scale_table == nullptr ||
        gate_up_alpha_table == nullptr || rows == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t row_blocks = (rows + 15) / 16;
    infer_moe_silu_quantize_slots_nvfp4_kernel<<<dim3(groups, row_blocks), 1, 0, stream>>>(
        indices, gate_up_table, packed_table, scales_table, input_scale_table,
        gate_up_alpha_table, rows, groups);
    return cudaGetLastError();
}

__global__ void infer_moe_silu_quantize_slots_nvfp4_simple_scales_kernel(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* const* packed_table,
    std::uint8_t* const* scales_table,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups) {
    const std::uint32_t slot = blockIdx.x;
    const std::uint32_t row_block = blockIdx.y;
    if (slot >= groups || threadIdx.x != 0) return;
    const std::uint32_t expert = indices[slot];
    const float input_scale = input_scale_table[expert];
    if (input_scale <= 0.0f || !isfinite(input_scale)) return;
    const float gate_up_alpha = gate_up_alpha_table[expert];
    const float* gate_up = gate_up_table[slot];
    std::uint8_t* packed = packed_table[slot];
    std::uint8_t* scales = scales_table[slot];
    const std::uint32_t row_start = row_block * 16;
    const std::uint32_t row_end = min(row_start + 16, rows);
    float max_abs = 0.0f;
    for (std::uint32_t row = row_start; row < row_end; ++row) {
        const float gate_value = gate_up[row] * gate_up_alpha;
        const float up_value = gate_up[rows + row] * gate_up_alpha;
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float value = (gate_value * sigmoid * up_value) / input_scale;
        if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
    }
    const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float scale = infer_e4m3_value(scale_code);
    scales[row_block] = scale_code;
    for (std::uint32_t row = row_start; row < row_end; row += 2) {
        const float lo_gate = gate_up[row] * gate_up_alpha;
        const float lo_up = gate_up[rows + row] * gate_up_alpha;
        const float lo_sigmoid = 1.0f / (1.0f + expf(-lo_gate));
        const float lo_value = scale == 0.0f ? 0.0f : ((lo_gate * lo_sigmoid * lo_up) / input_scale) / scale;
        const std::uint8_t lo = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        std::uint8_t hi = 0;
        if (row + 1 < row_end) {
            const float hi_gate = gate_up[row + 1] * gate_up_alpha;
            const float hi_up = gate_up[rows + row + 1] * gate_up_alpha;
            const float hi_sigmoid = 1.0f / (1.0f + expf(-hi_gate));
            const float hi_value = scale == 0.0f ? 0.0f : ((hi_gate * hi_sigmoid * hi_up) / input_scale) / scale;
            hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
        }
        packed[row / 2] = lo | (hi << 4);
    }
}

extern "C" cudaError_t infer_moe_silu_quantize_slots_nvfp4_simple_scales_on_stream(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* const* packed_table,
    std::uint8_t* const* scales_table,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (indices == nullptr || gate_up_table == nullptr || packed_table == nullptr ||
        scales_table == nullptr || input_scale_table == nullptr || gate_up_alpha_table == nullptr ||
        rows == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    infer_moe_silu_quantize_slots_nvfp4_simple_scales_kernel<<<dim3(groups, (rows + 15) / 16), 1, 0, stream>>>(
        indices, gate_up_table, packed_table, scales_table, input_scale_table, gate_up_alpha_table, rows, groups);
    return cudaGetLastError();
}

__global__ void infer_moe_weighted_accumulate_slots_f32_kernel(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* inputs,
    const float* alpha_table,
    float* output,
    std::uint32_t len,
    std::uint32_t groups) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < groups; ++slot) {
        const std::uint32_t expert = indices[slot];
        sum += inputs[slot][idx] * route_weights[slot] * alpha_table[expert];
    }
    output[idx] = sum;
}

extern "C" cudaError_t infer_moe_weighted_accumulate_slots_f32_on_stream(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* inputs,
    const float* alpha_table,
    float* output,
    std::uint32_t len,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (indices == nullptr || route_weights == nullptr || inputs == nullptr ||
        alpha_table == nullptr || output == nullptr || len == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_moe_weighted_accumulate_slots_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        indices, route_weights, inputs, alpha_table, output, len, groups);
    return cudaGetLastError();
}

__device__ __forceinline__ float infer_round_f32_to_bf16_value(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    const std::uint32_t rounded = bits + 0x7fffu + lsb;
    return __uint_as_float(rounded & 0xffff0000u);
}

__global__ void infer_qwen36_ffn_finalize_f32_kernel(
    const float* moe_output,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const float shared_scale = 1.0f / (1.0f + expf(-shared_gate_logit[0]));
    const float shared_gated = shared_output[idx] * shared_scale;
    const float ffn_output = moe_output[idx] + shared_gated;
    output[idx] = infer_round_f32_to_bf16_value(residual[idx] + ffn_output);
}

extern "C" cudaError_t infer_qwen36_ffn_finalize_f32_on_stream(
    const float* moe_output,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t len,
    cudaStream_t stream) {
    if (moe_output == nullptr || shared_gate_logit == nullptr || shared_output == nullptr ||
        residual == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_qwen36_ffn_finalize_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        moe_output, shared_gate_logit, shared_output, residual, output, len);
    return cudaGetLastError();
}

__global__ void infer_qwen36_ffn_finalize_routed_f32_kernel(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* routed_outputs,
    const float* alpha_table,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t len,
    std::uint32_t groups) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float routed_sum = 0.0f;
    for (std::uint32_t slot = 0; slot < groups; ++slot) {
        const std::uint32_t expert = indices[slot];
        routed_sum += routed_outputs[slot][idx] * route_weights[slot] * alpha_table[expert];
    }
    const float shared_scale = 1.0f / (1.0f + expf(-shared_gate_logit[0]));
    const float shared_gated = shared_output[idx] * shared_scale;
    const float ffn_output = routed_sum + shared_gated;
    output[idx] = infer_round_f32_to_bf16_value(residual[idx] + ffn_output);
}

__global__ void infer_qwen36_ffn_finalize_routed_batch_f32_kernel(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* routed_outputs,
    const float* alpha_table,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t groups_per_row) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) return;
    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx - row * cols;
    const std::uint32_t route_base = row * groups_per_row;
    float routed_sum = 0.0f;
    for (std::uint32_t slot = 0; slot < groups_per_row; ++slot) {
        const std::uint32_t route = route_base + slot;
        const std::uint32_t expert = indices[route];
        routed_sum += routed_outputs[route][col] * route_weights[route] * alpha_table[expert];
    }
    const float shared_scale = 1.0f / (1.0f + expf(-shared_gate_logit[row]));
    const float shared_gated = shared_output[idx] * shared_scale;
    output[idx] = infer_round_f32_to_bf16_value(residual[idx] + routed_sum + shared_gated);
}

extern "C" cudaError_t infer_qwen36_ffn_finalize_routed_f32_on_stream(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* routed_outputs,
    const float* alpha_table,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t len,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (indices == nullptr || route_weights == nullptr || routed_outputs == nullptr ||
        alpha_table == nullptr || shared_gate_logit == nullptr || shared_output == nullptr ||
        residual == nullptr || output == nullptr || len == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_qwen36_ffn_finalize_routed_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        indices, route_weights, routed_outputs, alpha_table, shared_gate_logit,
        shared_output, residual, output, len, groups);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen36_ffn_finalize_routed_batch_f32_on_stream(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* routed_outputs,
    const float* alpha_table,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t groups_per_row,
    cudaStream_t stream) {
    if (indices == nullptr || route_weights == nullptr || routed_outputs == nullptr ||
        alpha_table == nullptr || shared_gate_logit == nullptr || shared_output == nullptr ||
        residual == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        groups_per_row == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t len = static_cast<std::uint64_t>(rows) * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_qwen36_ffn_finalize_routed_batch_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        indices, route_weights, routed_outputs, alpha_table, shared_gate_logit,
        shared_output, residual, output, rows, cols, groups_per_row);
    return cudaGetLastError();
}

// RoPE, layout transforms, and attention/KV-cache kernels.
__global__ void infer_rope_neox_f32_kernel(const float* input,
                                                 float* output,
                                                 std::uint32_t rows,
                                                 std::uint32_t head_dim,
                                                 std::uint32_t position,
                                                 float theta) {
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total_pairs = rows * half;
    if (pair_idx >= total_pairs) {
        return;
    }

    const std::uint32_t row = pair_idx / half;
    const std::uint32_t i = pair_idx % half;
    const std::uint32_t row_start = row * head_dim;
    const float inv_freq = powf(theta, -2.0f * static_cast<float>(i) /
                                           static_cast<float>(head_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(position) * inv_freq, &sin_value, &cos_value);

    const float a = input[row_start + i];
    const float b = input[row_start + i + half];
    output[row_start + i] = a * cos_value - b * sin_value;
    output[row_start + i + half] = a * sin_value + b * cos_value;
}

__global__ void infer_rope_neox_f32_indexed_kernel(const float* input,
                                                         float* output,
                                                         std::uint32_t rows,
                                                         std::uint32_t head_dim,
                                                         const std::uint32_t* position,
                                                         float theta) {
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total_pairs = rows * half;
    if (pair_idx >= total_pairs) {
        return;
    }

    const std::uint32_t row = pair_idx / half;
    const std::uint32_t i = pair_idx % half;
    const std::uint32_t row_start = row * head_dim;
    const float inv_freq = powf(theta, -2.0f * static_cast<float>(i) /
                                           static_cast<float>(head_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(*position) * inv_freq, &sin_value, &cos_value);

    const float a = input[row_start + i];
    const float b = input[row_start + i + half];
    output[row_start + i] = a * cos_value - b * sin_value;
    output[row_start + i + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_neox_f32(const float* input,
                                                 float* output,
                                                 std::uint32_t rows,
                                                 std::uint32_t head_dim,
                                                 std::uint32_t position,
                                                 float theta) {
    if (input == nullptr || output == nullptr || rows == 0 || head_dim == 0 ||
        (head_dim % 2) != 0 || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t total_pairs = rows * half;
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_f32_kernel<<<blocks, kThreads>>>(
        input, output, rows, head_dim, position, theta);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_rope_neox_f32_on_stream(const float* input,
                                                           float* output,
                                                           std::uint32_t rows,
                                                           std::uint32_t head_dim,
                                                           std::uint32_t position,
                                                           float theta,
                                                           cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || head_dim == 0 ||
        (head_dim % 2) != 0 || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t total_pairs = rows * half;
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, position, theta);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_rope_neox_f32_indexed_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    const std::uint32_t* position,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || position == nullptr || rows == 0 ||
        head_dim == 0 || (head_dim % 2) != 0 || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t total_pairs = rows * half;
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_f32_indexed_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, position, theta);
    return cudaGetLastError();
}

__global__ void infer_rope_neox_partial_f32_kernel(const float* input,
                                                         float* output,
                                                         std::uint32_t rows,
                                                         std::uint32_t head_dim,
                                                         std::uint32_t rotary_dim,
                                                         std::uint32_t position,
                                                         float theta) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * head_dim;
    if (idx >= len) {
        return;
    }
    const std::uint32_t dim = idx % head_dim;
    if (dim >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }
    const std::uint32_t half = rotary_dim / 2;
    if (dim >= half) {
        return;
    }

    const std::uint32_t row_start = (idx / head_dim) * head_dim;
    const float inv_freq =
        powf(theta, -2.0f * static_cast<float>(dim) / static_cast<float>(rotary_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(position) * inv_freq, &sin_value, &cos_value);

    const float a = input[row_start + dim];
    const float b = input[row_start + dim + half];
    output[row_start + dim] = a * cos_value - b * sin_value;
    output[row_start + dim + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_neox_partial_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t position,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || head_dim == 0 ||
        rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 ||
        !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = rows * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_rope_neox_partial_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, rotary_dim, position, theta);
    return cudaGetLastError();
}

// IMRoPE / MRoPE kernel for Qwen3.5/3.6 full-attention text decode.
//
// rotary_dim is the number of channels per head that receive rotation
// (rotary_dim = head_dim * partial_rotary_factor). rotary_dim/2 pairs are
// partitioned into 4 sections [v0,v1,v2,v3] (t,h,w,extra) summing to
// rotary_dim/2. For each pair index i in [0, rotary_dim/2):
//   sector = i % sect_dims
//   position is chosen from [pos_t, pos_h, pos_w, pos_extra] per the IMRoPE
//   sector-to-dimension mapping, then the pair (x[i], x[i + rotary_dim/2])
//   is rotated by position * theta^(-2*i/rotary_dim).
// Channels in [rotary_dim, head_dim) are copied unchanged.
__global__ void infer_rope_imrope_f32_kernel(const float* input,
                                                    float* output,
                                                    std::uint32_t rows,
                                                    std::uint32_t head_dim,
                                                    std::uint32_t rotary_dim,
                                                    std::uint32_t v0,
                                                    std::uint32_t v1,
                                                    std::uint32_t v2,
                                                    std::uint32_t v3,
                                                    std::uint32_t pos_t,
                                                    std::uint32_t pos_h,
                                                    std::uint32_t pos_w,
                                                    std::uint32_t pos_extra,
                                                    const std::uint32_t* positions,
                                                    std::uint32_t position_count,
                                                    float theta) {
    if (positions != nullptr) {
        pos_t = positions[0];
        pos_h = position_count == 1 ? pos_t : positions[1];
        pos_w = position_count == 1 ? pos_t : positions[2];
        pos_extra = position_count == 1 ? 0 : positions[3];
    }
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * head_dim;
    if (idx >= len) {
        return;
    }
    const std::uint32_t dim = idx % head_dim;
    if (dim >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }
    const std::uint32_t half = rotary_dim / 2;
    if (dim >= half) {
        return;
    }

    const std::uint32_t sect_dims = v0 + v1 + v2 + v3;
    const std::uint32_t sector = dim % sect_dims;

    std::uint32_t position;
    if (sector % 3 == 1 && sector < 3 * v1) {
        position = pos_h;
    } else if (sector % 3 == 2 && sector < 3 * v2) {
        position = pos_w;
    } else if (sector % 3 == 0 && sector < 3 * v0) {
        position = pos_t;
    } else {
        position = pos_extra;
    }

    // Interleaving selects the position axis, not the frequency. Frequency i
    // remains theta^(-2*i/rotary_dim), so equal T/H/W text positions reduce
    // exactly to ordinary partial Neox RoPE.
    const float inv_freq = powf(
        theta,
        -2.0f * static_cast<float>(dim) / static_cast<float>(rotary_dim));
    const float section_theta = static_cast<float>(position) * inv_freq;

    float sin_value;
    float cos_value;
    sincosf(section_theta, &sin_value, &cos_value);

    const std::uint32_t row_start = (idx / head_dim) * head_dim;
    const float a = input[row_start + dim];
    const float b = input[row_start + dim + half];
    output[row_start + dim] = a * cos_value - b * sin_value;
    output[row_start + dim + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_imrope_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t v0,
    std::uint32_t v1,
    std::uint32_t v2,
    std::uint32_t v3,
    std::uint32_t pos_t,
    std::uint32_t pos_h,
    std::uint32_t pos_w,
    std::uint32_t pos_extra,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || head_dim == 0 ||
        rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 ||
        v0 + v1 + v2 + v3 != rotary_dim / 2 ||
        !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = rows * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_rope_imrope_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, rotary_dim, v0, v1, v2, v3,
        pos_t, pos_h, pos_w, pos_extra, nullptr, 0, theta);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_rope_imrope_f32_indexed_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t v0,
    std::uint32_t v1,
    std::uint32_t v2,
    std::uint32_t v3,
    const std::uint32_t* positions,
    std::uint32_t position_count,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || positions == nullptr || rows == 0 ||
        head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim % 2) != 0 || v0 + v1 + v2 + v3 != rotary_dim / 2 ||
        (position_count != 1 && position_count != 4) ||
        !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = rows * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_rope_imrope_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, rotary_dim, v0, v1, v2, v3,
        0, 0, 0, 0, positions, position_count, theta);
    return cudaGetLastError();
}

__global__ void infer_rope_imrope_text_batch_f32_kernel(
    const float* input,
    float* output,
    const std::uint32_t* positions,
    std::uint32_t batch_size,
    std::uint32_t heads_per_row,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    float theta) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t values_per_batch = heads_per_row * head_dim;
    const std::uint32_t len = batch_size * values_per_batch;
    if (idx >= len) return;
    const std::uint32_t dim = idx % head_dim;
    if (dim >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }
    const std::uint32_t half = rotary_dim / 2;
    if (dim >= half) return;
    const std::uint32_t batch = idx / values_per_batch;
    const float inv_freq = powf(
        theta, -2.0f * static_cast<float>(dim) / static_cast<float>(rotary_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(positions[batch]) * inv_freq, &sin_value, &cos_value);
    const std::uint32_t row_start = (idx / head_dim) * head_dim;
    const float a = input[row_start + dim];
    const float b = input[row_start + dim + half];
    output[row_start + dim] = a * cos_value - b * sin_value;
    output[row_start + dim + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_imrope_text_batch_f32_on_stream(
    const float* input,
    float* output,
    const std::uint32_t* positions,
    std::uint32_t batch_size,
    std::uint32_t heads_per_row,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t v0,
    std::uint32_t v1,
    std::uint32_t v2,
    std::uint32_t v3,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || positions == nullptr || batch_size == 0 ||
        heads_per_row == 0 || head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim % 2) != 0 || v0 + v1 + v2 + v3 != rotary_dim / 2 ||
        !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t len = static_cast<std::uint64_t>(batch_size) * heads_per_row * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_rope_imrope_text_batch_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, positions, batch_size, heads_per_row, head_dim, rotary_dim, theta);
    return cudaGetLastError();
}

__global__ void infer_rope_neox_sequence_f32_kernel(const float* input,
                                                          float* output,
                                                          std::uint32_t tokens,
                                                          std::uint32_t heads,
                                                          std::uint32_t head_dim,
                                                          std::uint32_t start_position,
                                                          float theta) {
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total_pairs = tokens * heads * half;
    if (pair_idx >= total_pairs) {
        return;
    }

    const std::uint32_t i = pair_idx % half;
    const std::uint32_t row = pair_idx / half;
    const std::uint32_t token = row / heads;
    const std::uint32_t row_start = row * head_dim;
    const std::uint32_t position = start_position + token;
    const float inv_freq = powf(theta, -2.0f * static_cast<float>(i) /
                                           static_cast<float>(head_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(position) * inv_freq, &sin_value, &cos_value);

    const float a = input[row_start + i];
    const float b = input[row_start + i + half];
    output[row_start + i] = a * cos_value - b * sin_value;
    output[row_start + i + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_neox_sequence_f32(const float* input,
                                                          float* output,
                                                          std::uint32_t tokens,
                                                          std::uint32_t heads,
                                                          std::uint32_t head_dim,
                                                          std::uint32_t start_position,
                                                          float theta) {
    if (input == nullptr || output == nullptr || tokens == 0 || heads == 0 || head_dim == 0 ||
        (head_dim % 2) != 0 || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t total_pairs = tokens * heads * (head_dim / 2);
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_sequence_f32_kernel<<<blocks, kThreads>>>(
        input, output, tokens, heads, head_dim, start_position, theta);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_rope_neox_sequence_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t start_position,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || tokens == 0 || heads == 0 || head_dim == 0 ||
        (head_dim % 2) != 0 || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t total_pairs = tokens * heads * (head_dim / 2);
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_sequence_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, tokens, heads, head_dim, start_position, theta);
    return cudaGetLastError();
}

__global__ void infer_add_f32_kernel(const float* left,
                                           const float* right,
                                           float* output,
                                           std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    output[idx] = left[idx] + right[idx];
}

extern "C" cudaError_t infer_add_f32(const float* left,
                                           const float* right,
                                           float* output,
                                           std::uint32_t len) {
    if (left == nullptr || right == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_add_f32_kernel<<<blocks, kThreads>>>(left, right, output, len);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_add_f32_on_stream(const float* left,
                                                     const float* right,
                                                     float* output,
                                                     std::uint32_t len,
                                                     cudaStream_t stream) {
    if (left == nullptr || right == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_add_f32_kernel<<<blocks, kThreads, 0, stream>>>(left, right, output, len);
    return cudaGetLastError();
}

__global__ void infer_row_major_to_col_major_f32_kernel(const float* input,
                                                             float* output,
                                                             std::uint32_t rows,
                                                             std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) {
        return;
    }
    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx % cols;
    output[row + col * rows] = input[row * cols + col];
}

extern "C" cudaError_t infer_row_major_to_col_major_f32(const float* input,
                                                              float* output,
                                                              std::uint32_t rows,
                                                              std::uint32_t cols) {
    if (input == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = rows * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_row_major_to_col_major_f32_kernel<<<blocks, kThreads>>>(
        input, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_col_major_to_row_major_f32_kernel(const float* input,
                                                             float* output,
                                                             std::uint32_t rows,
                                                             std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) {
        return;
    }
    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx % cols;
    output[row * cols + col] = input[row + col * rows];
}

extern "C" cudaError_t infer_col_major_to_row_major_f32(const float* input,
                                                              float* output,
                                                              std::uint32_t rows,
                                                              std::uint32_t cols) {
    if (input == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = rows * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_col_major_to_row_major_f32_kernel<<<blocks, kThreads>>>(
        input, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_copy_row_f32_kernel(const float* input,
                                                float* output,
                                                std::uint32_t row,
                                                std::uint32_t cols) {
    const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    output[col] = input[row * cols + col];
}

extern "C" cudaError_t infer_copy_row_f32(const float* input,
                                                float* output,
                                                std::uint32_t row,
                                                std::uint32_t cols) {
    if (input == nullptr || output == nullptr || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((cols + kThreads - 1) / kThreads);
    infer_copy_row_f32_kernel<<<blocks, kThreads>>>(input, output, row, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_copy_row_f32_on_stream(const float* input,
                                                          float* output,
                                                          std::uint32_t row,
                                                          std::uint32_t cols,
                                                          cudaStream_t stream) {
    if (input == nullptr || output == nullptr || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((cols + kThreads - 1) / kThreads);
    infer_copy_row_f32_kernel<<<blocks, kThreads, 0, stream>>>(input, output, row, cols);
    return cudaGetLastError();
}

__global__ void infer_copy_bf16_row_to_f32_indexed_kernel(const std::uint16_t* input,
                                                               const std::uint32_t* row,
                                                               float* output,
                                                               std::uint32_t cols) {
    const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    const std::uint32_t row_idx = *row;
    const std::uint16_t raw = input[row_idx * cols + col];
    const __nv_bfloat16 value = *reinterpret_cast<const __nv_bfloat16*>(&raw);
    output[col] = __bfloat162float(value);
}

extern "C" cudaError_t infer_copy_bf16_row_to_f32_indexed(const std::uint16_t* input,
                                                                const std::uint32_t* row,
                                                                float* output,
                                                                std::uint32_t cols) {
    if (input == nullptr || row == nullptr || output == nullptr || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((cols + kThreads - 1) / kThreads);
    infer_copy_bf16_row_to_f32_indexed_kernel<<<blocks, kThreads>>>(
        input, row, output, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_copy_bf16_row_to_f32_indexed_on_stream(
    const std::uint16_t* input,
    const std::uint32_t* row,
    float* output,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || row == nullptr || output == nullptr || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((cols + kThreads - 1) / kThreads);
    infer_copy_bf16_row_to_f32_indexed_kernel<<<blocks, kThreads, 0, stream>>>(
        input, row, output, cols);
    return cudaGetLastError();
}

__global__ void infer_copy_bf16_rows_to_f32_indexed_kernel(
    const std::uint16_t* input,
    const std::uint32_t* rows,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = batch_size * cols;
    if (idx >= len) return;
    const std::uint32_t batch = idx / cols;
    const std::uint32_t col = idx % cols;
    const std::uint16_t raw = input[rows[batch] * cols + col];
    output[idx] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&raw));
}

extern "C" cudaError_t infer_copy_bf16_rows_to_f32_indexed_on_stream(
    const std::uint16_t* input,
    const std::uint32_t* rows,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || rows == nullptr || output == nullptr || batch_size == 0 ||
        cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = batch_size * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_copy_bf16_rows_to_f32_indexed_kernel<<<blocks, kThreads, 0, stream>>>(
        input, rows, output, batch_size, cols);
    return cudaGetLastError();
}

__global__ void infer_single_token_gqa_f32_kernel(const float* value,
                                                        float* output,
                                                        std::uint32_t q_heads,
                                                        std::uint32_t kv_heads,
                                                        std::uint32_t head_dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = q_heads * head_dim;
    if (idx >= len) {
        return;
    }

    const std::uint32_t q_head = idx / head_dim;
    const std::uint32_t dim = idx % head_dim;
    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    output[idx] = value[kv_head * head_dim + dim];
}

extern "C" cudaError_t infer_single_token_gqa_f32(const float* key,
                                                        const float* value,
                                                        float* output,
                                                        std::uint32_t q_heads,
                                                        std::uint32_t kv_heads,
                                                        std::uint32_t head_dim) {
    if (key == nullptr || value == nullptr || output == nullptr || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = q_heads * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_single_token_gqa_f32_kernel<<<blocks, kThreads>>>(
        value, output, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

__global__ void infer_append_rows_f32_kernel(const float* src,
                                                   float* dst,
                                                   std::uint32_t dst_start_row,
                                                   std::uint32_t rows,
                                                   std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) {
        return;
    }

    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx % cols;
    dst[(dst_start_row + row) * cols + col] = src[row * cols + col];
}

__global__ void infer_append_rows_f32_indexed_kernel(const float* src,
                                                           float* dst,
                                                           const std::uint32_t* dst_start_row,
                                                           std::uint32_t rows,
                                                           std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) {
        return;
    }

    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx % cols;
    dst[(*dst_start_row + row) * cols + col] = src[row * cols + col];
}

extern "C" cudaError_t infer_append_rows_f32(const float* src,
                                                   float* dst,
                                                   std::uint32_t dst_start_row,
                                                   std::uint32_t rows,
                                                   std::uint32_t cols) {
    if (src == nullptr || dst == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = rows * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_append_rows_f32_kernel<<<blocks, kThreads>>>(
        src, dst, dst_start_row, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_append_rows_f32_on_stream(const float* src,
                                                             float* dst,
                                                             std::uint32_t dst_start_row,
                                                             std::uint32_t rows,
                                                             std::uint32_t cols,
                                                             cudaStream_t stream) {
    if (src == nullptr || dst == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = rows * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_append_rows_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        src, dst, dst_start_row, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_append_rows_f32_indexed_on_stream(
    const float* src,
    float* dst,
    const std::uint32_t* dst_start_row,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (src == nullptr || dst == nullptr || dst_start_row == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = rows * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_append_rows_f32_indexed_kernel<<<blocks, kThreads, 0, stream>>>(
        src, dst, dst_start_row, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_single_token_gqa_f32_from_cache(const float* key_cache,
                                                                   const float* value_cache,
                                                                   float* output,
                                                                   std::uint32_t position,
                                                                   std::uint32_t q_heads,
                                                                   std::uint32_t kv_heads,
                                                                   std::uint32_t head_dim) {
    if (key_cache == nullptr || value_cache == nullptr || output == nullptr || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    const std::uint32_t kv_width = kv_heads * head_dim;
    const float* value = value_cache + position * kv_width;
    constexpr int kThreads = 256;
    const std::uint32_t len = q_heads * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_single_token_gqa_f32_kernel<<<blocks, kThreads>>>(
        value, output, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

__global__ void infer_cached_gqa_attention_f32_kernel(const float* query,
                                                            const float* key_cache,
                                                            const float* value_cache,
                                                            float* output,
                                                            std::uint32_t cache_len,
                                                            std::uint32_t q_heads,
                                                            std::uint32_t kv_heads,
                                                            std::uint32_t head_dim) {
    extern __shared__ float partial[];
    const std::uint32_t q_head = blockIdx.x;
    if (q_head >= q_heads) {
        return;
    }

    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q = query + q_head * head_dim;

    float max_score = -INFINITY;
    for (std::uint32_t row = 0; row < cache_len; ++row) {
        const float* k = key_cache + row * kv_width + kv_head * head_dim;
        float dot = 0.0f;
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * k[dim];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            max_score = fmaxf(max_score, partial[0] * scale);
        }
        __syncthreads();
    }

    float accum = 0.0f;
    float total_weight = 0.0f;
    for (std::uint32_t row = 0; row < cache_len; ++row) {
        const float* k = key_cache + row * kv_width + kv_head * head_dim;
        float dot = 0.0f;
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * k[dim];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float weight = expf(partial[0] * scale - max_score);
            partial[0] = weight;
            total_weight += weight;
        }
        __syncthreads();
        const float weight = partial[0];
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            const float* v = value_cache + row * kv_width + kv_head * head_dim;
            accum += weight * v[dim];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[0] = total_weight;
    }
    __syncthreads();
    const float inv_total = 1.0f / partial[0];
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        output[q_head * head_dim + dim] = accum * inv_total;
    }
}

// Single-pass FlashAttention-style decode for the indexed-attention path.
//
// Reads K once and V once per row (instead of two K passes for max + softmax).
// Online softmax: track running (max, sum, weighted_acc); rescale on each new
// max. Caches Q in shared memory and accumulates along the head dimension
// using a warp reduction. Coalesces V reads by row and broadcasts the weight.
//
// Layout: head_dim <= blockDim.x (Qwen3 uses head_dim=128, blockDim=256).
// Threads 0..head_dim-1 each own exactly one element of Q/K/V; the remaining
// threads are idle for the dot product.
__global__ void infer_flash_decode_attention_f32_indexed_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    const std::uint32_t* cache_len,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    extern __shared__ float shmem[];
    float* q_sh = shmem;                                 // head_dim floats
    float* partial = shmem + head_dim;                    // blockDim.x floats
    const std::uint32_t q_head = blockIdx.x;
    if (q_head >= q_heads) {
        return;
    }

    const std::uint32_t actual_cache_len = *cache_len;
    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q_in = query + q_head * head_dim;

    // Cache Q for this head into shared memory.
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_sh[i] = q_in[i];
    }
    __syncthreads();

    if (head_dim > blockDim.x) {
        // Defensive guard: this kernel assumes head_dim <= blockDim.x.
        return;
    }

    // Each thread with tid < head_dim owns one accumulation slot for the
    // corresponding head dimension; idle threads do nothing useful for the
    // value gather but still participate in the reduction barrier.
    float m = -INFINITY;
    float s = 0.0f;
    float acc = 0.0f;  // Only meaningful when threadIdx.x < head_dim.
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t tid_in_warp = tid & 31u;
    const std::uint32_t warp_id = tid >> 5u;
    const std::uint32_t num_warps = blockDim.x >> 5u;

    for (std::uint32_t row = 0; row < actual_cache_len; ++row) {
        const float* k = key_cache + static_cast<std::size_t>(row) * kv_width +
                         kv_head * head_dim;
        const float* v = value_cache + static_cast<std::size_t>(row) * kv_width +
                         kv_head * head_dim;

        // Each thread loads its slice of K (one element when head_dim <= blockDim).
        float thread_dot = 0.0f;
        if (tid < head_dim) {
            thread_dot = q_sh[tid] * k[tid];
        }

        // Warp-level reduction: each warp produces its own partial dot.
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 16);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 8);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 4);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 2);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 1);
        if (tid_in_warp == 0) {
            partial[warp_id] = thread_dot;
        }
        __syncthreads();

        // Warp 0 reduces across warps.
        float dot = 0.0f;
        if (warp_id == 0 && tid_in_warp == 0) {
            for (std::uint32_t w = 0; w < num_warps; ++w) {
                dot += partial[w];
            }
            const float sc = dot * scale;
            float scale_factor;
            float weight;
            if (sc > m) {
                scale_factor = expf(m - sc);
                weight = 1.0f;
                m = sc;
            } else {
                scale_factor = 1.0f;
                weight = expf(sc - m);
            }
            s = s * scale_factor + weight;
            partial[0] = scale_factor;
            partial[1] = weight;
        }
        __syncthreads();
        const float scale_factor = partial[0];
        const float weight = partial[1];
        if (tid < head_dim) {
            acc = acc * scale_factor;
            acc = __fmaf_rn(weight, v[tid], acc);
        }
        __syncthreads();
    }

    // Write the normalized output. Thread 0 owns the true running sum; publish
    // it for all threads via shared memory.
    if (threadIdx.x == 0) {
        partial[0] = s;
    }
    __syncthreads();
    const float inv_s = 1.0f / partial[0];
    if (tid < head_dim) {
        output[q_head * head_dim + tid] = acc * inv_s;
    }
}

__global__ void infer_flash_decode_attention_f32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    std::uint32_t actual_cache_len,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    extern __shared__ float shmem[];
    float* q_sh = shmem;
    float* partial = shmem + head_dim;
    const std::uint32_t q_head = blockIdx.x;
    if (q_head >= q_heads) return;

    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q_in = query + q_head * head_dim;
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_sh[i] = q_in[i];
    }
    __syncthreads();
    if (head_dim > blockDim.x) return;

    float m = -INFINITY;
    float s = 0.0f;
    float acc = 0.0f;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t tid_in_warp = tid & 31u;
    const std::uint32_t warp_id = tid >> 5u;
    const std::uint32_t num_warps = blockDim.x >> 5u;
    for (std::uint32_t row = 0; row < actual_cache_len; ++row) {
        const float* k = key_cache + static_cast<std::size_t>(row) * kv_width + kv_head * head_dim;
        const float* v = value_cache + static_cast<std::size_t>(row) * kv_width + kv_head * head_dim;
        float thread_dot = 0.0f;
        if (tid < head_dim) {
            thread_dot = q_sh[tid] * k[tid];
        }
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 16);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 8);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 4);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 2);
        thread_dot += __shfl_xor_sync(0xffffffffu, thread_dot, 1);
        if (tid_in_warp == 0) {
            partial[warp_id] = thread_dot;
        }
        __syncthreads();
        float dot = 0.0f;
        if (warp_id == 0 && tid_in_warp == 0) {
            for (std::uint32_t w = 0; w < num_warps; ++w) {
                dot += partial[w];
            }
            const float sc = dot * scale;
            float scale_factor;
            float weight;
            if (sc > m) {
                scale_factor = expf(m - sc);
                weight = 1.0f;
                m = sc;
            } else {
                scale_factor = 1.0f;
                weight = expf(sc - m);
            }
            s = s * scale_factor + weight;
            partial[0] = scale_factor;
            partial[1] = weight;
        }
        __syncthreads();
        const float scale_factor = partial[0];
        const float weight = partial[1];
        if (tid < head_dim) {
            acc = acc * scale_factor;
            acc = __fmaf_rn(weight, v[tid], acc);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0] = s;
    }
    __syncthreads();
    const float inv_s = 1.0f / partial[0];
    if (tid < head_dim) {
        output[q_head * head_dim + tid] = acc * inv_s;
    }
}



extern "C" cudaError_t infer_cached_gqa_attention_f32(const float* query,
                                                            const float* key_cache,
                                                            const float* value_cache,
                                                            float* output,
                                                            std::uint32_t cache_len,
                                                            std::uint32_t q_heads,
                                                            std::uint32_t kv_heads,
                                                            std::uint32_t head_dim) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
        cache_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || head_dim > kThreads ||
        (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    infer_cached_gqa_attention_f32_kernel<<<q_heads, kThreads, kThreads * sizeof(float)>>>(
        query, key_cache, value_cache, output, cache_len, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_cached_gqa_attention_f32_on_stream(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
        cache_len == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || head_dim > kThreads ||
        (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    if (head_dim <= kThreads) {
        const std::size_t shmem = (head_dim + kThreads) * sizeof(float);
        infer_flash_decode_attention_f32_kernel<<<q_heads, kThreads, shmem, stream>>>(
            query, key_cache, value_cache, output, cache_len, q_heads, kv_heads, head_dim);
    } else {
        infer_cached_gqa_attention_f32_kernel<<<q_heads, kThreads, kThreads * sizeof(float), stream>>>(
            query, key_cache, value_cache, output, cache_len, q_heads, kv_heads, head_dim);
    }
    return cudaGetLastError();
}

// NVFP4 KV-cache viability probe. K/V values are packed E2M1 (two values per
// byte) with one UE4M3 scale per contiguous 16-value block. Q and the
// online-softmax accumulator remain f32. This measures whether cache traffic
// and numerical error justify a later SM12x MMA layout; it is not that layout.
__device__ __forceinline__ float infer_e2m1_cache_value(std::uint8_t nibble) {
    const float magnitude = (nibble & 0x7) == 0x0 ? 0.0f
        : (nibble & 0x7) == 0x1 ? 0.5f : (nibble & 0x7) == 0x2 ? 1.0f
        : (nibble & 0x7) == 0x3 ? 1.5f : (nibble & 0x7) == 0x4 ? 2.0f
        : (nibble & 0x7) == 0x5 ? 3.0f : (nibble & 0x7) == 0x6 ? 4.0f : 6.0f;
    return (nibble & 0x8) == 0 ? magnitude : -magnitude;
}

__device__ __forceinline__ float infer_nvfp4_cache_value(
    const std::uint8_t* packed, const std::uint8_t* scales, std::size_t index) {
    const std::uint8_t byte = packed[index >> 1];
    const std::uint8_t nibble = (index & 1) == 0 ? byte & 0x0f : byte >> 4;
    return infer_e2m1_cache_value(nibble) * infer_e4m3_value(scales[index >> 4]);
}

__global__ void infer_flash_decode_attention_nvfp4_kernel(
    const float* query, const std::uint8_t* key_cache, const std::uint8_t* key_scales,
    const std::uint8_t* value_cache, const std::uint8_t* value_scales, float* output,
    std::uint32_t cache_len, std::uint32_t q_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    extern __shared__ float shmem[];
    float* q_sh = shmem;
    float* partial = shmem + head_dim;
    const std::uint32_t q_head = blockIdx.x;
    if (q_head >= q_heads || head_dim > blockDim.x) return;
    const std::uint32_t kv_head = q_head / (q_heads / kv_heads);
    const std::uint32_t kv_width = kv_heads * head_dim;
    const float* q_in = query + q_head * head_dim;
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) q_sh[index] = q_in[index];
    __syncthreads();
    float maximum = -INFINITY;
    float sum = 0.0f;
    float accum = 0.0f;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t warp = tid >> 5u;
    for (std::uint32_t row = 0; row < cache_len; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kv_width + kv_head * head_dim;
        float dot = tid < head_dim ? q_sh[tid] * infer_nvfp4_cache_value(key_cache, key_scales, base + tid) : 0.0f;
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        if (lane == 0) partial[warp] = dot;
        __syncthreads();
        if (warp == 0 && lane == 0) {
            dot = 0.0f;
            for (std::uint32_t index = 0; index < blockDim.x / 32; ++index) dot += partial[index];
            const float score = dot * rsqrtf(static_cast<float>(head_dim));
            const float rescale = score > maximum ? expf(maximum - score) : 1.0f;
            const float weight = score > maximum ? 1.0f : expf(score - maximum);
            maximum = fmaxf(maximum, score);
            sum = sum * rescale + weight;
            partial[0] = rescale;
            partial[1] = weight;
        }
        __syncthreads();
        if (tid < head_dim) accum = __fmaf_rn(partial[1], infer_nvfp4_cache_value(value_cache, value_scales, base + tid), accum * partial[0]);
        __syncthreads();
    }
    if (tid == 0) partial[0] = sum;
    __syncthreads();
    if (tid < head_dim) output[q_head * head_dim + tid] = accum / partial[0];
}

extern "C" cudaError_t infer_cached_gqa_attention_nvfp4_on_stream(
    const float* query, const std::uint8_t* key_cache, const std::uint8_t* key_scales,
    const std::uint8_t* value_cache, const std::uint8_t* value_scales, float* output,
    std::uint32_t cache_len, std::uint32_t q_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache == nullptr || key_scales == nullptr || value_cache == nullptr ||
        value_scales == nullptr || output == nullptr || cache_len == 0 || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || head_dim > kThreads || (q_heads % kv_heads) != 0) return cudaErrorInvalidValue;
    infer_flash_decode_attention_nvfp4_kernel<<<q_heads, kThreads, (head_dim + kThreads) * sizeof(float), stream>>>(
        query, key_cache, key_scales, value_cache, value_scales, output, cache_len, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

__global__ void infer_softmax_f32_in_place_kernel(float* values, std::uint32_t len) {
    __shared__ float partial[256];
    const std::uint32_t tid = threadIdx.x;
    float maximum = -INFINITY;
    for (std::uint32_t index = tid; index < len; index += blockDim.x) {
        maximum = fmaxf(maximum, values[index]);
    }
    partial[tid] = maximum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    maximum = partial[0];
    float sum = 0.0f;
    for (std::uint32_t index = tid; index < len; index += blockDim.x) {
        const float weight = expf(values[index] - maximum);
        values[index] = weight;
        sum += weight;
    }
    partial[tid] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inverse_sum = 1.0f / partial[0];
    for (std::uint32_t index = tid; index < len; index += blockDim.x) values[index] *= inverse_sum;
}

extern "C" cudaError_t infer_softmax_f32_in_place_on_stream(
    float* values, std::uint32_t len, cudaStream_t stream) {
    if (values == nullptr || len == 0) return cudaErrorInvalidValue;
    infer_softmax_f32_in_place_kernel<<<1, 256, 0, stream>>>(values, len);
    return cudaGetLastError();
}

// Legacy two-pass indexed attention kernel (kept as a fallback for head_dim
// shapes that don't match the FlashAttention decode fast path).
__global__ void infer_cached_gqa_attention_f32_indexed_kernel(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    const std::uint32_t* cache_len,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    extern __shared__ float partial[];
    const std::uint32_t q_head = blockIdx.x;
    if (q_head >= q_heads) {
        return;
    }

    const std::uint32_t actual_cache_len = *cache_len;
    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q = query + q_head * head_dim;

    float max_score = -INFINITY;
    for (std::uint32_t row = 0; row < actual_cache_len; ++row) {
        const float* k = key_cache + row * kv_width + kv_head * head_dim;
        float dot = 0.0f;
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * k[dim];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            max_score = fmaxf(max_score, partial[0] * scale);
        }
        __syncthreads();
    }

    float accum = 0.0f;
    float total_weight = 0.0f;
    for (std::uint32_t row = 0; row < actual_cache_len; ++row) {
        const float* k = key_cache + row * kv_width + kv_head * head_dim;
        float dot = 0.0f;
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * k[dim];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float weight = expf(partial[0] * scale - max_score);
            partial[0] = weight;
            total_weight += weight;
        }
        __syncthreads();
        const float weight = partial[0];
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            const float* v = value_cache + row * kv_width + kv_head * head_dim;
            accum += weight * v[dim];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[0] = total_weight;
    }
    __syncthreads();
    const float inv_total = 1.0f / partial[0];
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        output[q_head * head_dim + dim] = accum * inv_total;
    }
}

extern "C" cudaError_t infer_cached_gqa_attention_f32_indexed_on_stream(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    const std::uint32_t* cache_len,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        output == nullptr || cache_len == nullptr || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || head_dim > kThreads || (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    // Use the single-pass FlashAttention-style decode path when head_dim
    // equals 128 (Qwen3). For other shapes fall back to the legacy
    // two-pass kernel to preserve correctness on unusual head dims.
    if (head_dim <= kThreads) {
        const std::size_t shmem = (head_dim + kThreads) * sizeof(float);
        infer_flash_decode_attention_f32_indexed_kernel<<<
            q_heads, kThreads, shmem, stream>>>(
            query, key_cache, value_cache, output, cache_len, q_heads,
            kv_heads, head_dim);
    } else {
        infer_cached_gqa_attention_f32_indexed_kernel<<<
            q_heads, kThreads, kThreads * sizeof(float), stream>>>(
            query, key_cache, value_cache, output, cache_len, q_heads, kv_heads, head_dim);
    }
    return cudaGetLastError();
}

__global__ void infer_prefill_gqa_attention_f32_kernel(const float* query,
                                                             const float* key_cache,
                                                             const float* value_cache,
                                                             float* output,
                                                             std::uint32_t tokens,
                                                             std::uint32_t start_position,
                                                             std::uint32_t q_heads,
                                                             std::uint32_t kv_heads,
                                                             std::uint32_t head_dim) {
    extern __shared__ float partial[];
    const std::uint32_t token = blockIdx.x;
    const std::uint32_t q_head = blockIdx.y;
    if (token >= tokens || q_head >= q_heads) {
        return;
    }

    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const std::uint32_t hidden = q_heads * head_dim;
    const std::uint32_t cache_len = start_position + token + 1;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q = query + token * hidden + q_head * head_dim;

    float max_score = -INFINITY;
    for (std::uint32_t row = 0; row < cache_len; ++row) {
        const float* k = key_cache + row * kv_width + kv_head * head_dim;
        float dot = 0.0f;
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * k[dim];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            max_score = fmaxf(max_score, partial[0] * scale);
        }
        __syncthreads();
    }

    float accum = 0.0f;
    float total_weight = 0.0f;
    for (std::uint32_t row = 0; row < cache_len; ++row) {
        const float* k = key_cache + row * kv_width + kv_head * head_dim;
        float dot = 0.0f;
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * k[dim];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float weight = expf(partial[0] * scale - max_score);
            partial[0] = weight;
            total_weight += weight;
        }
        __syncthreads();
        const float weight = partial[0];
        for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            const float* v = value_cache + row * kv_width + kv_head * head_dim;
            accum += weight * v[dim];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[0] = total_weight;
    }
    __syncthreads();
    const float inv_total = 1.0f / partial[0];
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        output[token * hidden + q_head * head_dim + dim] = accum * inv_total;
    }
}

extern "C" cudaError_t infer_prefill_gqa_attention_f32(const float* query,
                                                             const float* key_cache,
                                                             const float* value_cache,
                                                             float* output,
                                                             std::uint32_t tokens,
                                                             std::uint32_t start_position,
                                                             std::uint32_t q_heads,
                                                             std::uint32_t kv_heads,
                                                             std::uint32_t head_dim) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
        tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || head_dim > kThreads ||
        (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 blocks(tokens, q_heads);
    infer_prefill_gqa_attention_f32_kernel<<<blocks, kThreads, kThreads * sizeof(float)>>>(
        query, key_cache, value_cache, output, tokens, start_position, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

// lm-head reductions, BF16 conversion, and output preparation.
__global__ void infer_bf16_matvec_logits_kernel(const float* input,
                                                      const std::uint16_t* weight,
                                                      float* logits,
                                                      std::uint32_t rows,
                                                      std::uint32_t cols) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    // Cache input into shared memory to avoid redundant global reads. The
    // input vector (cols * 4 bytes) is read by every block and would otherwise
    // be re-fetched 256 times per row.
    float* input_sh = partial + blockDim.x;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    float acc = 0.0f;
    const std::uint16_t* row_weight = weight + static_cast<std::size_t>(row) * cols;
    // Vectorized bf16x2 loads: 4 elements per thread per iter, fully coalesced
    // 128-byte cache-line access per warp. Requires the row pointer to be
    // 4-byte aligned (true when cols is even). For odd cols the row stride is
    // 2-byte, so vec loads would be misaligned -- fall back to scalar.
    const bool aligned = (cols & 1u) == 0u;
    if (aligned) {
        const std::uint32_t cols_aligned = cols & ~std::uint32_t(3u);
        for (std::uint32_t col = threadIdx.x * 4; col < cols_aligned; col += blockDim.x * 4) {
            const __nv_bfloat162 w0 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col);
            const __nv_bfloat162 w1 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col + 2);
            acc = __fmaf_rn(__bfloat162float(__low2bfloat16(w0)), input_sh[col], acc);
            acc = __fmaf_rn(__bfloat162float(__high2bfloat16(w0)), input_sh[col + 1], acc);
            acc = __fmaf_rn(__bfloat162float(__low2bfloat16(w1)), input_sh[col + 2], acc);
            acc = __fmaf_rn(__bfloat162float(__high2bfloat16(w1)), input_sh[col + 3], acc);
        }
        for (std::uint32_t col = cols_aligned + threadIdx.x; col < cols; col += blockDim.x) {
            const __nv_bfloat16 w = *reinterpret_cast<const __nv_bfloat16*>(row_weight + col);
            acc = __fmaf_rn(__bfloat162float(w), input_sh[col], acc);
        }
    } else {
        // Scalar fallback for odd cols (rare; never hit by Qwen3 production shapes).
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            const __nv_bfloat16 w = *reinterpret_cast<const __nv_bfloat16*>(row_weight + col);
            acc = __fmaf_rn(__bfloat162float(w), input_sh[col], acc);
        }
    }
    partial[threadIdx.x] = acc;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        logits[row] = partial[0];
    }
}

__global__ void infer_bf16_matvec_pair_logits_kernel(
    const float* input,
    const std::uint16_t* first_weight,
    const std::uint16_t* second_weight,
    float* first_logits,
    float* second_logits,
    std::uint32_t first_rows,
    std::uint32_t second_rows,
    std::uint32_t cols) {
    extern __shared__ float partial[];
    const std::uint32_t global_row = blockIdx.x;
    const bool first = global_row < first_rows;
    const std::uint32_t row = first ? global_row : global_row - first_rows;
    if ((!first && row >= second_rows) || (first && row >= first_rows)) {
        return;
    }
    const std::uint16_t* weight = first ? first_weight : second_weight;
    float* logits = first ? first_logits : second_logits;

    float* input_sh = partial + blockDim.x;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    float acc = 0.0f;
    const std::uint16_t* row_weight = weight + static_cast<std::size_t>(row) * cols;
    if ((cols & 1u) == 0u) {
        const std::uint32_t cols_aligned = cols & ~std::uint32_t(3u);
        for (std::uint32_t col = threadIdx.x * 4; col < cols_aligned; col += blockDim.x * 4) {
            const __nv_bfloat162 w0 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col);
            const __nv_bfloat162 w1 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col + 2);
            acc = __fmaf_rn(__bfloat162float(__low2bfloat16(w0)), input_sh[col], acc);
            acc = __fmaf_rn(__bfloat162float(__high2bfloat16(w0)), input_sh[col + 1], acc);
            acc = __fmaf_rn(__bfloat162float(__low2bfloat16(w1)), input_sh[col + 2], acc);
            acc = __fmaf_rn(__bfloat162float(__high2bfloat16(w1)), input_sh[col + 3], acc);
        }
        for (std::uint32_t col = cols_aligned + threadIdx.x; col < cols; col += blockDim.x) {
            const __nv_bfloat16 w = *reinterpret_cast<const __nv_bfloat16*>(row_weight + col);
            acc = __fmaf_rn(__bfloat162float(w), input_sh[col], acc);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            const __nv_bfloat16 w = *reinterpret_cast<const __nv_bfloat16*>(row_weight + col);
            acc = __fmaf_rn(__bfloat162float(w), input_sh[col], acc);
        }
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        logits[row] = partial[0];
    }
}

__global__ void infer_bf16_matvec_logits_batch_kernel(
    const float* input,
    const std::uint16_t* weight,
    float* logits,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols) {
    extern __shared__ float partial[];
    const std::uint32_t batch = blockIdx.x / rows;
    const std::uint32_t row = blockIdx.x % rows;
    if (batch >= batch_size) return;
    const float* row_input = input + static_cast<std::size_t>(batch) * cols;
    float* input_sh = partial + blockDim.x;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        input_sh[col] = row_input[col];
    }
    __syncthreads();
    float acc = 0.0f;
    const std::uint16_t* row_weight = weight + static_cast<std::size_t>(row) * cols;
    if ((cols & 1u) == 0u) {
        const std::uint32_t cols_aligned = cols & ~std::uint32_t(3u);
        for (std::uint32_t col = threadIdx.x * 4; col < cols_aligned; col += blockDim.x * 4) {
            const __nv_bfloat162 w0 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col);
            const __nv_bfloat162 w1 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col + 2);
            acc = __fmaf_rn(__bfloat162float(__low2bfloat16(w0)), input_sh[col], acc);
            acc = __fmaf_rn(__bfloat162float(__high2bfloat16(w0)), input_sh[col + 1], acc);
            acc = __fmaf_rn(__bfloat162float(__low2bfloat16(w1)), input_sh[col + 2], acc);
            acc = __fmaf_rn(__bfloat162float(__high2bfloat16(w1)), input_sh[col + 3], acc);
        }
        for (std::uint32_t col = cols_aligned + threadIdx.x; col < cols;
             col += blockDim.x) {
            const __nv_bfloat16 w =
                *reinterpret_cast<const __nv_bfloat16*>(row_weight + col);
            acc = __fmaf_rn(__bfloat162float(w), input_sh[col], acc);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            const __nv_bfloat16 w =
                *reinterpret_cast<const __nv_bfloat16*>(row_weight + col);
            acc = __fmaf_rn(__bfloat162float(w), input_sh[col], acc);
        }
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) logits[batch * rows + row] = partial[0];
}

__global__ void infer_argmax_f32_kernel(const float* values,
                                              std::uint32_t* out_index,
                                              float* out_value,
                                              std::uint32_t len) {
    const std::uint32_t row = blockIdx.x;
    values += row * len;
    extern __shared__ unsigned char shared_raw[];
    float* max_values = reinterpret_cast<float*>(shared_raw);
    std::uint32_t* max_indices =
        reinterpret_cast<std::uint32_t*>(max_values + blockDim.x);

    float best_value = -INFINITY;
    std::uint32_t best_index = 0;
    for (std::uint32_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
        const float value = values[idx];
        if (value > best_value || (value == best_value && idx < best_index)) {
            best_value = value;
            best_index = idx;
        }
    }
    max_values[threadIdx.x] = best_value;
    max_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = max_values[threadIdx.x + stride];
            const std::uint32_t other_index = max_indices[threadIdx.x + stride];
            if (other_value > max_values[threadIdx.x] ||
                (other_value == max_values[threadIdx.x] && other_index < max_indices[threadIdx.x])) {
                max_values[threadIdx.x] = other_value;
                max_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_index[row] = max_indices[0];
        out_value[row] = max_values[0];
    }
}

extern "C" cudaError_t infer_argmax_f32_batch_on_stream(
    const float* values,
    std::uint32_t* out_index,
    float* out_value,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (values == nullptr || out_index == nullptr || out_value == nullptr ||
        rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::size_t shared_bytes = kThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_argmax_f32_kernel<<<rows, kThreads, shared_bytes, stream>>>(
        values, out_index, out_value, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_argmax_f32_on_stream(const float* values,
                                                        std::uint32_t* out_index,
                                                        float* out_value,
                                                        std::uint32_t len,
                                                        cudaStream_t stream) {
    if (values == nullptr || out_index == nullptr || out_value == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::size_t shared_bytes = kThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_argmax_f32_kernel<<<1, kThreads, shared_bytes, stream>>>(
        values, out_index, out_value, len);
    return cudaGetLastError();
}

// =============================================================================
// Direct top-1 lm-head path.
//
// Shape: weight is [VOCAB, HIDDEN] row-major BF16, input is [HIDDEN] f32.
//
// Goal: compute argmax(weight * input) without materializing VOCAB f32 logits to
// global memory. The bottleneck is reading the weight bytes (~1.18 GB for
// Qwen3 8B with VOCAB=151936, HIDDEN=4096). At ~273 GB/s peak on GB10 that is a
// theoretical ~4.3 ms floor; the legacy path was ~5.48 ms because it also wrote
// the full logits vector back to global memory and ran a separate single-block
// reduction over it.
//
// Strategy:
//   1) Cache the input vector in shared memory so each warp reads it from L1
//      instead of re-issuing 151936 global fetches per element.
//   2) Use vectorized __nv_bfloat162 loads to halve load instructions and get
//      wider coalesced reads from the weight buffer.
//   3) Each block handles a contiguous chunk of `kRowsPerBlock` rows. All
//      warps in the block cooperate on every row: compute partial dot, reduce
//      in-shmem, track the local chunk's argmax, write ONE (index, value)
//      pair to a small scratch buffer in global memory.
//   4) A final tiny block reduces the few scratch pairs into out_index/out_value.
//
// This drops the logits writeback to global memory entirely and turns the
// final reduction cost into O(gridDim), which is much smaller than VOCAB.
// =============================================================================

// One warp (32 threads) collaborates on a single row. cols must be a multiple
// of 64 (true for HIDDEN=4096). Two bf16 values per load => 32-element stride.
__device__ inline float infer_bf16_row_dot_warp(const std::uint16_t* row_weight,
                                                      const float* input_sh,
                                                      std::uint32_t cols) {
    float acc = 0.0f;
    const std::uint32_t tid = threadIdx.x & 31u;
    // Each thread reads 2 bf16x2 (4 elements) per step, 32 threads => 128 cols/step
    for (std::uint32_t col = tid * 4; col < cols; col += 32 * 4) {
        const __nv_bfloat162 w0 =
            *reinterpret_cast<const __nv_bfloat162*>(row_weight + col);
        const __nv_bfloat162 w1 =
            *reinterpret_cast<const __nv_bfloat162*>(row_weight + col + 2);
        const float w0x = __bfloat162float(__low2bfloat16(w0));
        const float w0y = __bfloat162float(__high2bfloat16(w0));
        const float w1x = __bfloat162float(__low2bfloat16(w1));
        const float w1y = __bfloat162float(__high2bfloat16(w1));
        acc = __fmaf_rn(w0x, input_sh[col], acc);
        acc = __fmaf_rn(w0y, input_sh[col + 1], acc);
        acc = __fmaf_rn(w1x, input_sh[col + 2], acc);
        acc = __fmaf_rn(w1y, input_sh[col + 3], acc);
    }
    // Warp reduction.
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    return acc;
}

__global__ void infer_lm_head_top1_pass1_kernel(
    const float* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    float* __restrict__ scratch_value,
    std::uint32_t* __restrict__ scratch_index,
    std::uint32_t rows,
    std::uint32_t cols) {
    extern __shared__ float input_sh[];
    // Cache the input vector (cols f32 values) into shared memory.
    for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        input_sh[i] = input[i];
    }
    // Layout of static shared-memory scratch that follows the input cache:
    //   warp_values[kWarpsPerBlock] (float)
    //   warp_indices[kWarpsPerBlock] (uint32)
    // kWarpsPerBlock is implicit in blockDim.x >> 5.
    const std::uint32_t warps_in_block = blockDim.x >> 5u;
    float* warp_values = input_sh + cols;
    std::uint32_t* warp_indices =
        reinterpret_cast<std::uint32_t*>(warp_values + warps_in_block);
    __syncthreads();

    const std::uint32_t warp_id = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    // Each warp handles one row. blockIdx.x maps to the row offset owned by
    // this block: row_base = blockIdx.x * warps_in_block. (Caller passes
    // kRowsPerBlock == warps_in_block in launch configuration.)
    const std::uint32_t row = blockIdx.x * warps_in_block + warp_id;

    if (row < rows) {
        const float logit =
            infer_bf16_row_dot_warp(weight + static_cast<std::size_t>(row) * cols,
                                           input_sh, cols);
        if (lane == 0) {
            scratch_value[blockIdx.x * warps_in_block + warp_id] = logit;
            scratch_index[blockIdx.x * warps_in_block + warp_id] = row;
        }
    } else if (lane == 0) {
        // Pad unused warp slots with -INF so the final reduction is correct.
        scratch_value[blockIdx.x * warps_in_block + warp_id] = -INFINITY;
        scratch_index[blockIdx.x * warps_in_block + warp_id] = 0;
    }
}

__global__ void infer_lm_head_top1_final_kernel(
    const float* __restrict__ scratch_value,
    const std::uint32_t* __restrict__ scratch_index,
    std::uint32_t* __restrict__ out_index,
    float* __restrict__ out_value,
    std::uint32_t len) {
    extern __shared__ unsigned char sh_raw[];
    float* sv = reinterpret_cast<float*>(sh_raw);
    std::uint32_t* si = reinterpret_cast<std::uint32_t*>(sv + blockDim.x);

    float best_value = -INFINITY;
    std::uint32_t best_index = 0;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        const float v = scratch_value[i];
        const std::uint32_t idx = scratch_index[i];
        if (v > best_value || (v == best_value && idx < best_index)) {
            best_value = v;
            best_index = idx;
        }
    }
    sv[threadIdx.x] = best_value;
    si[threadIdx.x] = best_index;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = sv[threadIdx.x + stride];
            const std::uint32_t oi = si[threadIdx.x + stride];
            if (ov > sv[threadIdx.x] ||
                (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_index = si[0];
        *out_value = sv[0];
    }
}

extern "C" cudaError_t infer_bf16_linear_argmax_f32(const float* input,
                                                           const std::uint16_t* weight,
                                                          float* logits,
                                                          std::uint32_t* out_index,
                                                          float* out_value,
                                                          std::uint32_t rows,
                                                          std::uint32_t cols) {
    if (input == nullptr || weight == nullptr || logits == nullptr || out_index == nullptr ||
        out_value == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::size_t matvec_shmem =
        kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
    infer_bf16_matvec_logits_kernel<<<rows, kThreads, matvec_shmem>>>(
        input, weight, logits, rows, cols);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    const std::size_t shared_bytes = kThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_argmax_f32_kernel<<<1, kThreads, shared_bytes>>>(
        logits, out_index, out_value, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bf16_linear_argmax_f32_on_stream(
    const float* input,
    const std::uint16_t* weight,
    float* logits,
    std::uint32_t* out_index,
    float* out_value,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || logits == nullptr || out_index == nullptr ||
        out_value == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::size_t matvec_shmem =
        kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
    infer_bf16_matvec_logits_kernel<<<rows, kThreads, matvec_shmem, stream>>>(
        input, weight, logits, rows, cols);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    const std::size_t shared_bytes = kThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_argmax_f32_kernel<<<1, kThreads, shared_bytes, stream>>>(
        logits, out_index, out_value, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bf16_linear_logits_f32(const float* input,
                                                          const std::uint16_t* weight,
                                                          float* logits,
                                                          std::uint32_t rows,
                                                          std::uint32_t cols) {
    if (input == nullptr || weight == nullptr || logits == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::size_t matvec_shmem =
        kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
    infer_bf16_matvec_logits_kernel<<<rows, kThreads, matvec_shmem>>>(
        input, weight, logits, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bf16_linear_logits_f32_on_stream(
    const float* input,
    const std::uint16_t* weight,
    float* logits,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || logits == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::size_t matvec_shmem =
        kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
    infer_bf16_matvec_logits_kernel<<<rows, kThreads, matvec_shmem, stream>>>(
        input, weight, logits, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bf16_linear_logits_f32_batch_on_stream(
    const float* input,
    const std::uint16_t* weight,
    float* logits,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || logits == nullptr || batch_size == 0 ||
        rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::size_t shmem =
        kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
    infer_bf16_matvec_logits_batch_kernel<<<batch_size * rows, kThreads, shmem, stream>>>(
        input, weight, logits, batch_size, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bf16_linear_pair_logits_f32_on_stream(
    const float* input,
    const std::uint16_t* first_weight,
    const std::uint16_t* second_weight,
    float* first_logits,
    float* second_logits,
    std::uint32_t first_rows,
    std::uint32_t second_rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || first_weight == nullptr || second_weight == nullptr ||
        first_logits == nullptr || second_logits == nullptr || first_rows == 0 ||
        second_rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::size_t matvec_shmem =
        kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
    infer_bf16_matvec_pair_logits_kernel<<<first_rows + second_rows, kThreads, matvec_shmem, stream>>>(
        input, first_weight, second_weight, first_logits, second_logits,
        first_rows, second_rows, cols);
    return cudaGetLastError();
}

__global__ void infer_bf16_to_f32_kernel(const std::uint16_t* input,
                                               float* output,
                                               std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const __nv_bfloat16 value = *reinterpret_cast<const __nv_bfloat16*>(input + idx);
    output[idx] = __bfloat162float(value);
}

extern "C" cudaError_t infer_bf16_to_f32(const std::uint16_t* input,
                                               float* output,
                                               std::uint32_t len) {
    if (input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_bf16_to_f32_kernel<<<blocks, kThreads>>>(input, output, len);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bf16_to_f32_on_stream(const std::uint16_t* input,
                                                         float* output,
                                                         std::uint32_t len,
                                                         cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_bf16_to_f32_kernel<<<blocks, kThreads, 0, stream>>>(input, output, len);
    return cudaGetLastError();
}

__global__ void infer_round_f32_to_bf16_kernel(const float* input,
                                                     float* output,
                                                     std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const std::uint32_t bits = __float_as_uint(input[idx]);
        const std::uint32_t lsb = (bits >> 16) & 1u;
        const std::uint32_t rounded = bits + 0x7fffu + lsb;
        output[idx] = __uint_as_float(rounded & 0xffff0000u);
    }
}

extern "C" cudaError_t infer_round_f32_to_bf16_on_stream(const float* input,
                                                               float* output,
                                                               std::uint32_t len,
                                                               cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_round_f32_to_bf16_kernel<<<blocks, kThreads, 0, stream>>>(input, output, len);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_round_f32_to_bf16_in_place_on_stream(float* values,
                                                                        std::uint32_t len,
                                                                        cudaStream_t stream) {
    if (values == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_round_f32_to_bf16_kernel<<<blocks, kThreads, 0, stream>>>(values, values, len);
    return cudaGetLastError();
}

// FP8 linear and Qwen3.6 Gated Delta Net kernels.
__global__ void infer_gated_delta_net_128_f32_kernel(const float* q,
                                                           const float* k,
                                                           const float* v,
                                                           const float* gate,
                                                           const float* beta,
                                                           float* state,
                                                           float* output,
                                                           std::uint32_t heads) {
    constexpr std::uint32_t kState = 128;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t col = blockIdx.y;
    const std::uint32_t row = threadIdx.x;
    if (head >= heads || col >= kState || row >= kState) {
        return;
    }

    const std::uint32_t head_base = head * kState;
    const std::uint32_t state_base = head * kState * kState + col * kState;
    const float q_value = q[head_base + row];
    const float k_value = k[head_base + row];
    const float old_state = state[state_base + row];

    __shared__ float partial[128];
    partial[row] = old_state * k_value;
    __syncthreads();

    for (std::uint32_t stride = kState / 2; stride > 0; stride >>= 1) {
        if (row < stride) {
            partial[row] += partial[row + stride];
        }
        __syncthreads();
    }

    const float decay = expf(gate[head]);
    const float state_dot_k = partial[0];
    // All lanes must consume the first reduction before lane 0 reuses partial[0].
    __syncthreads();
    const float delta = (v[head_base + col] - decay * state_dot_k) * beta[head];
    const float new_state = decay * old_state + k_value * delta;
    state[state_base + row] = new_state;

    partial[row] = new_state * q_value;
    __syncthreads();

    for (std::uint32_t stride = kState / 2; stride > 0; stride >>= 1) {
        if (row < stride) {
            partial[row] += partial[row + stride];
        }
        __syncthreads();
    }

    if (row == 0) {
        output[head_base + col] = partial[0] * 0.08838834764831845f; // 1 / sqrt(128)
    }
}

extern "C" cudaError_t infer_gated_delta_net_128_f32_on_stream(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr ||
        state == nullptr || output == nullptr || heads == 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(heads, 128, 1);
    infer_gated_delta_net_128_f32_kernel<<<grid, 128, 0, stream>>>(
        q, k, v, gate, beta, state, output, heads);
    return cudaGetLastError();
}

__global__ void infer_gated_delta_net_128_f32_batch_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* const* state_table,
    float* output,
    std::uint32_t heads) {
    constexpr std::uint32_t kState = 128;
    const std::uint32_t batch = blockIdx.x / heads;
    const std::uint32_t head = blockIdx.x % heads;
    const std::uint32_t col = blockIdx.y;
    const std::uint32_t row = threadIdx.x;
    if (col >= kState || row >= kState) {
        return;
    }

    const std::uint32_t vector_base = (batch * heads + head) * kState;
    const std::uint32_t state_base = head * kState * kState + col * kState;
    float* state = state_table[batch];
    const float q_value = q[vector_base + row];
    const float k_value = k[vector_base + row];
    const float old_state = state[state_base + row];

    __shared__ float partial[128];
    partial[row] = old_state * k_value;
    __syncthreads();
    for (std::uint32_t stride = kState / 2; stride > 0; stride >>= 1) {
        if (row < stride) partial[row] += partial[row + stride];
        __syncthreads();
    }

    const float decay = expf(gate[batch * heads + head]);
    const float state_dot_k = partial[0];
    __syncthreads();
    const float delta =
        (v[vector_base + col] - decay * state_dot_k) * beta[batch * heads + head];
    const float new_state = decay * old_state + k_value * delta;
    state[state_base + row] = new_state;

    partial[row] = new_state * q_value;
    __syncthreads();
    for (std::uint32_t stride = kState / 2; stride > 0; stride >>= 1) {
        if (row < stride) partial[row] += partial[row + stride];
        __syncthreads();
    }
    if (row == 0) {
        output[vector_base + col] = partial[0] * 0.08838834764831845f;
    }
}

extern "C" cudaError_t infer_gated_delta_net_128_f32_batch_on_stream(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* const* state_table,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr ||
        state_table == nullptr || output == nullptr || batch_size == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(batch_size * heads, 128, 1);
    infer_gated_delta_net_128_f32_batch_kernel<<<grid, 128, 0, stream>>>(
        q, k, v, gate, beta, state_table, output, heads);
    return cudaGetLastError();
}

__global__ void infer_fp8_linear_f32_kernel(const float* input,
                                                  const std::uint8_t* weight,
                                                  float* output,
                                                  std::uint32_t rows,
                                                  std::uint32_t cols,
                                                  float weight_scale,
                                                  const float* channel_weight_scale) {
    const std::uint32_t batch = blockIdx.y;
    input += batch * cols;
    output += batch * rows;
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    const std::uint32_t row_base = row * cols;
    if ((cols & 3U) == 0) {
        const auto* input4 = reinterpret_cast<const float4*>(input);
        const auto* weight4 = reinterpret_cast<const uchar4*>(weight + row_base);
        const std::uint32_t cols4 = cols >> 2;
        for (std::uint32_t col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
            const float4 in = input4[col4];
            const uchar4 w = weight4[col4];
            sum += in.x * infer_e4m3_value(w.x);
            sum += in.y * infer_e4m3_value(w.y);
            sum += in.z * infer_e4m3_value(w.z);
            sum += in.w * infer_e4m3_value(w.w);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            sum += input[col] * infer_e4m3_value(weight[row_base + col]);
        }
    }
    sum = infer_block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        const float scale = channel_weight_scale == nullptr
            ? weight_scale
            : channel_weight_scale[row];
        output[row] = sum * scale;
    }
}

__global__ void infer_fp8_linear_segmented_f32_kernel(
    const float* input,
    const std::uint8_t* first_weight,
    const std::uint8_t* second_weight,
    const std::uint8_t* third_weight,
    float* first_output,
    float* second_output,
    float* third_output,
    std::uint32_t first_rows,
    std::uint32_t second_rows,
    std::uint32_t third_rows,
    std::uint32_t cols,
    float first_scale,
    float second_scale,
    float third_scale) {
    const std::uint32_t global_row = blockIdx.x;
    const std::uint8_t* weight;
    float* output;
    std::uint32_t row;
    float scale;
    if (global_row < first_rows) {
        weight = first_weight;
        output = first_output;
        row = global_row;
        scale = first_scale;
    } else if (global_row < first_rows + second_rows) {
        weight = second_weight;
        output = second_output;
        row = global_row - first_rows;
        scale = second_scale;
    } else {
        weight = third_weight;
        output = third_output;
        row = global_row - first_rows - second_rows;
        scale = third_scale;
    }

    float sum = 0.0f;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;
    if ((cols & 3U) == 0) {
        const auto* input4 = reinterpret_cast<const float4*>(input);
        const auto* weight4 = reinterpret_cast<const uchar4*>(weight + row_base);
        const std::uint32_t cols4 = cols >> 2;
        for (std::uint32_t col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
            const float4 in = input4[col4];
            const uchar4 w = weight4[col4];
            sum += in.x * infer_e4m3_value(w.x);
            sum += in.y * infer_e4m3_value(w.y);
            sum += in.z * infer_e4m3_value(w.z);
            sum += in.w * infer_e4m3_value(w.w);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            sum += input[col] * infer_e4m3_value(weight[row_base + col]);
        }
    }
    sum = infer_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum * scale;
    }
}

static cudaError_t infer_launch_fp8_linear_f32(
    const float* input,
    const std::uint8_t* weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float weight_scale,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        !isfinite(weight_scale) || threads < 64 || threads > 512 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }

    infer_fp8_linear_f32_kernel<<<rows, threads, 0, stream>>>(
        input, weight, output, rows, cols, weight_scale, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_fp8_linear_f32_batch_on_stream(
    const float* input,
    const std::uint8_t* weight,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols,
    float weight_scale,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || batch_size == 0 ||
        rows == 0 || cols == 0 || !isfinite(weight_scale) || threads < 64 ||
        threads > 512 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_fp8_linear_f32_kernel<<<dim3(rows, batch_size), threads, 0, stream>>>(
        input, weight, output, rows, cols, weight_scale, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_fp8_linear_channel_scaled_f32_configured_on_stream(
    const float* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || channel_weight_scale == nullptr ||
        output == nullptr || rows == 0 || cols == 0 || threads < 64 || threads > 512 ||
        (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_fp8_linear_f32_kernel<<<rows, threads, 0, stream>>>(
        input, weight, output, rows, cols, 1.0f, channel_weight_scale);
    return cudaGetLastError();
}

__global__ void infer_fp8_linear_channel_scaled_dynamic_f32_kernel(
    const float* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float local_max = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[col];
        if (isfinite(value)) {
            local_max = fmaxf(local_max, fabsf(value));
        }
    }
    const float max_abs = infer_block_reduce_max(local_max);
    __shared__ float input_scale;
    if (threadIdx.x == 0) {
        input_scale = max_abs == 0.0f ? 1.0f : max_abs / 448.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    const std::uint32_t row_base = row * cols;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::uint8_t input_code = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(input[col] / input_scale, __NV_SATFINITE, __NV_E4M3));
        sum += infer_e4m3_value(input_code) * infer_e4m3_value(weight[row_base + col]);
    }
    sum = infer_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum * input_scale * channel_weight_scale[row];
    }
}

extern "C" cudaError_t infer_fp8_linear_channel_scaled_dynamic_f32_on_stream(
    const float* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || channel_weight_scale == nullptr ||
        output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_fp8_linear_channel_scaled_dynamic_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, channel_weight_scale, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_fp8_dynamic_input_scale_f32_kernel(
    const float* input,
    float* input_scale,
    std::uint32_t cols) {
    float local_max = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[col];
        if (isfinite(value)) {
            local_max = fmaxf(local_max, fabsf(value));
        }
    }
    const float max_abs = infer_block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        input_scale[0] = max_abs == 0.0f ? 1.0f : max_abs / 448.0f;
    }
}

__global__ void infer_fp8_linear_channel_scaled_precomputed_dynamic_f32_kernel(
    const float* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    const float* input_scale_ptr,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float input_scale = input_scale_ptr[0];
    float sum = 0.0f;
    const std::uint32_t row_base = row * cols;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::uint8_t input_code = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(input[col] / input_scale, __NV_SATFINITE, __NV_E4M3));
        sum += infer_e4m3_value(input_code) * infer_e4m3_value(weight[row_base + col]);
    }
    sum = infer_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum * input_scale * channel_weight_scale[row];
    }
}

extern "C" cudaError_t infer_fp8_linear_channel_scaled_precomputed_dynamic_f32_on_stream(
    const float* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    float* input_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || channel_weight_scale == nullptr ||
        input_scale == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_fp8_dynamic_input_scale_f32_kernel<<<1, kThreads, 0, stream>>>(
        input, input_scale, cols);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    infer_fp8_linear_channel_scaled_precomputed_dynamic_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, channel_weight_scale, input_scale, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_dynamic_quantize_fp8_e4m3_f32_kernel(
    const float* input,
    std::uint8_t* quantized_input,
    float* input_scale,
    std::uint32_t cols) {
    float local_max = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[col];
        if (isfinite(value)) {
            local_max = fmaxf(local_max, fabsf(value));
        }
    }
    const float max_abs = infer_block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        input_scale[0] = max_abs == 0.0f ? 1.0f : max_abs / 448.0f;
    }
    __syncthreads();
    const float scale = input_scale[0];
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        quantized_input[col] = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(input[col] / scale, __NV_SATFINITE, __NV_E4M3));
    }
}

__global__ void infer_fp8_linear_quantized_channel_scaled_f32_kernel(
    const std::uint8_t* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    const float* input_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    const std::uint32_t row_base = row * cols;
    if ((cols & 3U) == 0) {
        const auto* input4 = reinterpret_cast<const uchar4*>(input);
        const auto* weight4 = reinterpret_cast<const uchar4*>(weight + row_base);
        const std::uint32_t cols4 = cols >> 2;
        for (std::uint32_t col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
            const uchar4 in = input4[col4];
            const uchar4 w = weight4[col4];
            sum += infer_e4m3_value(in.x) * infer_e4m3_value(w.x);
            sum += infer_e4m3_value(in.y) * infer_e4m3_value(w.y);
            sum += infer_e4m3_value(in.z) * infer_e4m3_value(w.z);
            sum += infer_e4m3_value(in.w) * infer_e4m3_value(w.w);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            sum += infer_e4m3_value(input[col]) * infer_e4m3_value(weight[row_base + col]);
        }
    }
    sum = infer_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum * input_scale[0] * channel_weight_scale[row];
    }
}

extern "C" cudaError_t infer_quantize_fp8_e4m3_dynamic_f32_on_stream(
    const float* input,
    std::uint8_t* quantized_input,
    float* input_scale,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || quantized_input == nullptr || input_scale == nullptr || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_dynamic_quantize_fp8_e4m3_f32_kernel<<<1, kThreads, 0, stream>>>(
        input, quantized_input, input_scale, cols);
    return cudaGetLastError();
}

__global__ void infer_dynamic_quantize_fp8_e4m3_f32_batch_kernel(
    const float* input,
    std::uint8_t* quantized_input,
    float* input_scale,
    std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    const float* row_input = input + row * cols;
    std::uint8_t* row_output = quantized_input + row * cols;
    float local_max = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = row_input[col];
        if (isfinite(value)) local_max = fmaxf(local_max, fabsf(value));
    }
    const float max_abs = infer_block_reduce_max(local_max);
    if (threadIdx.x == 0) input_scale[row] = max_abs == 0.0f ? 1.0f : max_abs / 448.0f;
    __syncthreads();
    const float scale = input_scale[row];
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        row_output[col] = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(row_input[col] / scale, __NV_SATFINITE, __NV_E4M3));
    }
}

extern "C" cudaError_t infer_quantize_fp8_e4m3_dynamic_f32_batch_on_stream(
    const float* input,
    std::uint8_t* quantized_input,
    float* input_scale,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || quantized_input == nullptr || input_scale == nullptr ||
        rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_dynamic_quantize_fp8_e4m3_f32_batch_kernel<<<rows, kThreads, 0, stream>>>(
        input, quantized_input, input_scale, cols);
    return cudaGetLastError();
}

__global__ void infer_scale_channel_f32_device_scalar_kernel(
    float* values,
    const float* channel_scale,
    const float* scalar,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        values[idx] *= channel_scale[idx] * scalar[0];
    }
}

extern "C" cudaError_t infer_scale_channel_f32_device_scalar_on_stream(
    float* values,
    const float* channel_scale,
    const float* scalar,
    std::uint32_t len,
    cudaStream_t stream) {
    if (values == nullptr || channel_scale == nullptr || scalar == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_scale_channel_f32_device_scalar_kernel<<<blocks, kThreads, 0, stream>>>(
        values, channel_scale, scalar, len);
    return cudaGetLastError();
}

__global__ void infer_scale_channel_f32_device_row_scalar_kernel(
    float* values,
    const float* channel_scale,
    const float* row_scale,
    std::uint32_t channels,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const std::uint32_t row = idx / channels;
        const std::uint32_t channel = idx % channels;
        values[idx] *= channel_scale[channel] * row_scale[row];
    }
}

extern "C" cudaError_t infer_scale_channel_f32_device_row_scalar_on_stream(
    float* values,
    const float* channel_scale,
    const float* row_scale,
    std::uint32_t rows,
    std::uint32_t channels,
    cudaStream_t stream) {
    if (values == nullptr || channel_scale == nullptr || row_scale == nullptr ||
        rows == 0 || channels == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = rows * channels;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_scale_channel_f32_device_row_scalar_kernel<<<blocks, kThreads, 0, stream>>>(
        values, channel_scale, row_scale, channels, len);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_fp8_linear_channel_scaled_dynamic_quantized_f32_configured_on_stream(
    const float* input,
    std::uint8_t* quantized_input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    float* input_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || quantized_input == nullptr || weight == nullptr ||
        channel_weight_scale == nullptr || input_scale == nullptr || output == nullptr ||
        rows == 0 || cols == 0 || threads < 64 || threads > 512 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kQuantizeThreads = 256;
    infer_dynamic_quantize_fp8_e4m3_f32_kernel<<<1, kQuantizeThreads, 0, stream>>>(
        input, quantized_input, input_scale, cols);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    infer_fp8_linear_quantized_channel_scaled_f32_kernel<<<rows, threads, 0, stream>>>(
        quantized_input, weight, channel_weight_scale, input_scale, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_fp8_moe_grouped_gate_up_f32_kernel(
    const std::uint32_t* indices,
    const std::uint8_t* input,
    const float* input_scale,
    const std::uint8_t* const* gate_weights,
    const float* const* gate_scales,
    const std::uint8_t* const* up_weights,
    const float* const* up_scales,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t slots) {
    const std::uint32_t slot = blockIdx.x / rows;
    const std::uint32_t row = blockIdx.x % rows;
    if (slot >= slots) {
        return;
    }
    const std::uint32_t expert = indices[slot];
    const std::uint8_t* gate_weight = gate_weights[expert] + row * cols;
    const std::uint8_t* up_weight = up_weights[expert] + row * cols;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if ((cols & 3U) == 0) {
        const auto* input4 = reinterpret_cast<const uchar4*>(input);
        const auto* gate4 = reinterpret_cast<const uchar4*>(gate_weight);
        const auto* up4 = reinterpret_cast<const uchar4*>(up_weight);
        const std::uint32_t cols4 = cols >> 2;
        for (std::uint32_t col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
            const uchar4 in = input4[col4];
            const uchar4 gate = gate4[col4];
            const uchar4 up = up4[col4];
            const float ix = infer_e4m3_value(in.x);
            const float iy = infer_e4m3_value(in.y);
            const float iz = infer_e4m3_value(in.z);
            const float iw = infer_e4m3_value(in.w);
            gate_sum += ix * infer_e4m3_value(gate.x);
            gate_sum += iy * infer_e4m3_value(gate.y);
            gate_sum += iz * infer_e4m3_value(gate.z);
            gate_sum += iw * infer_e4m3_value(gate.w);
            up_sum += ix * infer_e4m3_value(up.x);
            up_sum += iy * infer_e4m3_value(up.y);
            up_sum += iz * infer_e4m3_value(up.z);
            up_sum += iw * infer_e4m3_value(up.w);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            const float value = infer_e4m3_value(input[col]);
            gate_sum += value * infer_e4m3_value(gate_weight[col]);
            up_sum += value * infer_e4m3_value(up_weight[col]);
        }
    }
    gate_sum = infer_block_reduce_sum(gate_sum);
    up_sum = infer_block_reduce_sum(up_sum);
    if (threadIdx.x == 0) {
        const std::uint32_t base = slot * rows * 2;
        output[base + row] = gate_sum * input_scale[0] * gate_scales[expert][row];
        output[base + rows + row] = up_sum * input_scale[0] * up_scales[expert][row];
    }
}

extern "C" cudaError_t infer_fp8_moe_grouped_gate_up_f32_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* input,
    const float* input_scale,
    const std::uint8_t* const* gate_weights,
    const float* const* gate_scales,
    const std::uint8_t* const* up_weights,
    const float* const* up_scales,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t slots,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr || input_scale == nullptr ||
        gate_weights == nullptr || gate_scales == nullptr || up_weights == nullptr ||
        up_scales == nullptr || output == nullptr || rows == 0 || cols == 0 || slots == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_fp8_moe_grouped_gate_up_f32_kernel<<<rows * slots, kThreads, 0, stream>>>(
        indices, input, input_scale, gate_weights, gate_scales, up_weights, up_scales,
        output, rows, cols, slots);
    return cudaGetLastError();
}

__global__ void infer_moe_silu_quantize_fp8_slots_f32_kernel(
    const float* gate_up,
    std::uint8_t* quantized,
    float* scales,
    std::uint32_t rows) {
    const std::uint32_t slot = blockIdx.x;
    const float* slot_gate = gate_up + slot * rows * 2;
    const float* slot_up = slot_gate + rows;
    std::uint8_t* slot_output = quantized + slot * rows;
    float local_max = 0.0f;
    for (std::uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const float gate = slot_gate[row];
        const float value = (gate / (1.0f + expf(-gate))) * slot_up[row];
        if (isfinite(value)) {
            local_max = fmaxf(local_max, fabsf(value));
        }
    }
    const float max_abs = infer_block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        scales[slot] = max_abs == 0.0f ? 1.0f : max_abs / 448.0f;
    }
    __syncthreads();
    const float scale = scales[slot];
    for (std::uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const float gate = slot_gate[row];
        const float value = (gate / (1.0f + expf(-gate))) * slot_up[row];
        slot_output[row] = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(value / scale, __NV_SATFINITE, __NV_E4M3));
    }
}

extern "C" cudaError_t infer_moe_silu_quantize_fp8_slots_f32_on_stream(
    const float* gate_up,
    std::uint8_t* quantized,
    float* scales,
    std::uint32_t rows,
    std::uint32_t slots,
    cudaStream_t stream) {
    if (gate_up == nullptr || quantized == nullptr || scales == nullptr || rows == 0 || slots == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_moe_silu_quantize_fp8_slots_f32_kernel<<<slots, kThreads, 0, stream>>>(
        gate_up, quantized, scales, rows);
    return cudaGetLastError();
}

__global__ void infer_fp8_moe_grouped_down_f32_kernel(
    const std::uint32_t* indices,
    const std::uint8_t* inputs,
    const float* input_scales,
    const std::uint8_t* const* weights,
    const float* const* weight_scales,
    float* const* outputs,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t slots) {
    const std::uint32_t slot = blockIdx.x / rows;
    const std::uint32_t row = blockIdx.x % rows;
    if (slot >= slots) {
        return;
    }
    const std::uint32_t expert = indices[slot];
    const std::uint8_t* input = inputs + slot * cols;
    const std::uint8_t* weight = weights[expert] + row * cols;
    float sum = 0.0f;
    if ((cols & 3U) == 0) {
        const auto* input4 = reinterpret_cast<const uchar4*>(input);
        const auto* weight4 = reinterpret_cast<const uchar4*>(weight);
        const std::uint32_t cols4 = cols >> 2;
        for (std::uint32_t col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
            const uchar4 in = input4[col4];
            const uchar4 w = weight4[col4];
            sum += infer_e4m3_value(in.x) * infer_e4m3_value(w.x);
            sum += infer_e4m3_value(in.y) * infer_e4m3_value(w.y);
            sum += infer_e4m3_value(in.z) * infer_e4m3_value(w.z);
            sum += infer_e4m3_value(in.w) * infer_e4m3_value(w.w);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            sum += infer_e4m3_value(input[col]) * infer_e4m3_value(weight[col]);
        }
    }
    sum = infer_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        outputs[slot][row] = sum * input_scales[slot] * weight_scales[expert][row];
    }
}

extern "C" cudaError_t infer_fp8_moe_grouped_down_f32_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* inputs,
    const float* input_scales,
    const std::uint8_t* const* weights,
    const float* const* weight_scales,
    float* const* outputs,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t slots,
    cudaStream_t stream) {
    if (indices == nullptr || inputs == nullptr || input_scales == nullptr || weights == nullptr ||
        weight_scales == nullptr || outputs == nullptr || rows == 0 || cols == 0 || slots == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_fp8_moe_grouped_down_f32_kernel<<<rows * slots, kThreads, 0, stream>>>(
        indices, inputs, input_scales, weights, weight_scales, outputs, rows, cols, slots);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_fp8_linear_f32_configured_on_stream(
    const float* input,
    const std::uint8_t* weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float weight_scale,
    std::uint32_t threads,
    cudaStream_t stream) {
    return infer_launch_fp8_linear_f32(
        input, weight, output, rows, cols, weight_scale, threads, stream);
}

extern "C" cudaError_t infer_fp8_linear_pair_f32_configured_on_stream(
    const float* input,
    const std::uint8_t* first_weight,
    const std::uint8_t* second_weight,
    float* first_output,
    float* second_output,
    std::uint32_t first_rows,
    std::uint32_t second_rows,
    std::uint32_t cols,
    float first_scale,
    float second_scale,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || first_weight == nullptr || second_weight == nullptr ||
        first_output == nullptr || second_output == nullptr || first_rows == 0 ||
        second_rows == 0 || cols == 0 || !isfinite(first_scale) || !isfinite(second_scale) ||
        threads < 64 || threads > 512 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_fp8_linear_segmented_f32_kernel<<<first_rows + second_rows, threads, 0, stream>>>(
        input, first_weight, second_weight, nullptr, first_output, second_output, nullptr,
        first_rows, second_rows, 0, cols, first_scale, second_scale, 0.0f);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_fp8_linear_triple_f32_configured_on_stream(
    const float* input,
    const std::uint8_t* first_weight,
    const std::uint8_t* second_weight,
    const std::uint8_t* third_weight,
    float* first_output,
    float* second_output,
    float* third_output,
    std::uint32_t first_rows,
    std::uint32_t second_rows,
    std::uint32_t third_rows,
    std::uint32_t cols,
    float first_scale,
    float second_scale,
    float third_scale,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || first_weight == nullptr || second_weight == nullptr ||
        third_weight == nullptr || first_output == nullptr || second_output == nullptr ||
        third_output == nullptr || first_rows == 0 || second_rows == 0 || third_rows == 0 ||
        cols == 0 || !isfinite(first_scale) || !isfinite(second_scale) ||
        !isfinite(third_scale) || threads < 64 || threads > 512 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_fp8_linear_segmented_f32_kernel<<<first_rows + second_rows + third_rows, threads, 0, stream>>>(
        input, first_weight, second_weight, third_weight, first_output, second_output, third_output,
        first_rows, second_rows, third_rows, cols, first_scale, second_scale, third_scale);
    return cudaGetLastError();
}

__global__ void infer_fp8_linear_w8a8_f32_kernel(const float* input,
                                                       const std::uint8_t* weight,
                                                       float* output,
                                                       std::uint32_t rows,
                                                       std::uint32_t cols,
                                                       float weight_scale,
                                                       float input_scale) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    const std::uint32_t row_base = row * cols;
    if ((cols & 3U) == 0) {
        const auto* input4 = reinterpret_cast<const float4*>(input);
        const auto* weight4 = reinterpret_cast<const uchar4*>(weight + row_base);
        const std::uint32_t cols4 = cols >> 2;
        for (std::uint32_t col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x) {
            const float4 in = input4[col4];
            const uchar4 w = weight4[col4];
            const std::uint8_t ix = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(in.x / input_scale, __NV_SATFINITE, __NV_E4M3));
            const std::uint8_t iy = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(in.y / input_scale, __NV_SATFINITE, __NV_E4M3));
            const std::uint8_t iz = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(in.z / input_scale, __NV_SATFINITE, __NV_E4M3));
            const std::uint8_t iw = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(in.w / input_scale, __NV_SATFINITE, __NV_E4M3));
            sum += infer_e4m3_value(ix) * input_scale * infer_e4m3_value(w.x);
            sum += infer_e4m3_value(iy) * input_scale * infer_e4m3_value(w.y);
            sum += infer_e4m3_value(iz) * input_scale * infer_e4m3_value(w.z);
            sum += infer_e4m3_value(iw) * input_scale * infer_e4m3_value(w.w);
        }
    } else {
        for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            const std::uint8_t input_code = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(input[col] / input_scale, __NV_SATFINITE, __NV_E4M3));
            const float input_value = infer_e4m3_value(input_code) * input_scale;
            sum += input_value * infer_e4m3_value(weight[row_base + col]);
        }
    }
    sum = infer_block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = sum * weight_scale;
    }
}

extern "C" cudaError_t infer_fp8_linear_w8a8_f32_on_stream(const float* input,
                                                                  const std::uint8_t* weight,
                                                                  float* output,
                                                                  std::uint32_t rows,
                                                                  std::uint32_t cols,
                                                                  float weight_scale,
                                                                  float input_scale,
                                                                  cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        !isfinite(weight_scale) || input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    infer_fp8_linear_w8a8_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, output, rows, cols, weight_scale, input_scale);
    return cudaGetLastError();
}

__global__ void infer_quantize_fp8_e4m3_f32_kernel(const float* input,
                                                          std::uint8_t* output,
                                                          std::uint32_t len,
                                                          float inverse_scale) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(input[idx] * inverse_scale, __NV_SATFINITE, __NV_E4M3));
    }
}

extern "C" cudaError_t infer_quantize_fp8_e4m3_f32_on_stream(
    const float* input,
    std::uint8_t* output,
    std::uint32_t len,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0 || input_scale <= 0.0f ||
        !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t blocks = (len + kThreads - 1) / kThreads;
    infer_quantize_fp8_e4m3_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, len, 1.0f / input_scale);
    return cudaGetLastError();
}

// Qwen3.6-specific preparation and final W4A16 helpers.
__global__ void infer_qwen36_gdn_prep_kernel(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* conv_state,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t key_dim = key_heads * head_dim;
    const std::uint32_t value_dim = value_heads * head_dim;
    const std::uint32_t conv_dim = key_dim * 2 + value_dim;
    if (idx >= conv_dim) {
        return;
    }

    float mixed = qkv[idx] *
                  __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                      conv_weight_bf16 + idx * 4 + 3));
    mixed += conv_state[idx * 3 + 0] *
             __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                 conv_weight_bf16 + idx * 4 + 0));
    mixed += conv_state[idx * 3 + 1] *
             __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                 conv_weight_bf16 + idx * 4 + 1));
    mixed += conv_state[idx * 3 + 2] *
             __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                 conv_weight_bf16 + idx * 4 + 2));
    const float activated = mixed / (1.0f + expf(-mixed));

    // Recurrent conv cache stores the last 3 pre-conv projected values per channel.
    conv_state[idx * 3 + 0] = conv_state[idx * 3 + 1];
    conv_state[idx * 3 + 1] = conv_state[idx * 3 + 2];
    conv_state[idx * 3 + 2] = qkv[idx];

    if (idx < key_dim) {
        for (std::uint32_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
            q[(repeat * key_heads * head_dim) + idx] = activated;
        }
    } else if (idx < key_dim * 2) {
        const std::uint32_t k_idx = idx - key_dim;
        for (std::uint32_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
            k[(repeat * key_heads * head_dim) + k_idx] = activated;
        }
    } else {
        // V is stored grouped by K head in the checkpoint: [K0_V0, K0_V1, ..., K1_V0, K1_V1, ...]
        // Q/K are repeated in tiled order: [R0_K0, R0_K1, ..., R1_K0, R1_K1, ...]
        // Reorder V to tiled to match Q/K: [K0_V0, K1_V0, ..., K0_V1, K1_V1, ...]
        const std::uint32_t v_idx = idx - key_dim * 2;
        const std::uint32_t v_k_head = v_idx / head_dim;
        const std::uint32_t v_sub    = v_idx % head_dim;
        const std::uint32_t v_per_k  = value_heads / key_heads;
        const std::uint32_t k_head    = v_k_head / v_per_k;
        const std::uint32_t v_sub_idx = v_k_head % v_per_k;
        const std::uint32_t tiled_v_idx = v_sub_idx * key_heads * head_dim + k_head * head_dim + v_sub;
        v[tiled_v_idx] = activated;
    }
}

__global__ void infer_l2_norm_heads_128_kernel(float* values, std::uint32_t heads) {
    constexpr std::uint32_t kDim = 128;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (head >= heads || lane >= kDim) {
        return;
    }
    float* row = values + head * kDim;
    __shared__ float partial[128];
    const float value = row[lane];
    partial[lane] = value * value;
    __syncthreads();
    for (std::uint32_t stride = kDim / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    row[lane] = value / fmaxf(sqrtf(partial[0]), 1.0e-6f);
}

extern "C" cudaError_t infer_qwen36_gdn_prep_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* conv_state,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || conv_state == nullptr || key_heads == 0 || value_heads == 0 ||
        head_dim != 128 || value_heads % key_heads != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    const int blocks = static_cast<int>((conv_dim + kThreads - 1) / kThreads);
    infer_qwen36_gdn_prep_kernel<<<blocks, kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state, key_heads, value_heads, head_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    infer_l2_norm_heads_128_kernel<<<value_heads, 128, 0, stream>>>(q, value_heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    infer_l2_norm_heads_128_kernel<<<value_heads, 128, 0, stream>>>(k, value_heads);
    return cudaGetLastError();
}

__global__ void infer_qwen36_gdn_prep_batch_kernel(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* const* conv_state_table,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    std::uint32_t conv_dim) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t batch = linear / conv_dim;
    const std::uint32_t idx = linear % conv_dim;
    const std::uint32_t key_dim = key_heads * head_dim;
    const std::uint32_t value_dim = value_heads * head_dim;
    float* conv_state = conv_state_table[batch];
    const std::uint32_t input_base = batch * conv_dim;
    const std::uint32_t output_base = batch * value_dim;

    float mixed = qkv[input_base + idx] *
                  __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                      conv_weight_bf16 + idx * 4 + 3));
    mixed += conv_state[idx * 3 + 0] *
             __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                 conv_weight_bf16 + idx * 4 + 0));
    mixed += conv_state[idx * 3 + 1] *
             __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                 conv_weight_bf16 + idx * 4 + 1));
    mixed += conv_state[idx * 3 + 2] *
             __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                 conv_weight_bf16 + idx * 4 + 2));
    const float activated = mixed / (1.0f + expf(-mixed));
    conv_state[idx * 3 + 0] = conv_state[idx * 3 + 1];
    conv_state[idx * 3 + 1] = conv_state[idx * 3 + 2];
    conv_state[idx * 3 + 2] = qkv[input_base + idx];

    if (idx < key_dim) {
        for (std::uint32_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
            q[output_base + repeat * key_dim + idx] = activated;
        }
    } else if (idx < key_dim * 2) {
        const std::uint32_t k_idx = idx - key_dim;
        for (std::uint32_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
            k[output_base + repeat * key_dim + k_idx] = activated;
        }
    } else {
        const std::uint32_t v_idx = idx - key_dim * 2;
        const std::uint32_t v_k_head = v_idx / head_dim;
        const std::uint32_t v_sub = v_idx % head_dim;
        const std::uint32_t v_per_k = value_heads / key_heads;
        const std::uint32_t k_head = v_k_head / v_per_k;
        const std::uint32_t v_sub_idx = v_k_head % v_per_k;
        const std::uint32_t tiled_v_idx =
            v_sub_idx * key_heads * head_dim + k_head * head_dim + v_sub;
        v[output_base + tiled_v_idx] = activated;
    }
}

extern "C" cudaError_t infer_qwen36_gdn_prep_batch_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* const* conv_state_table,
    std::uint32_t batch_size,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || conv_state_table == nullptr || batch_size == 0 || key_heads == 0 ||
        value_heads == 0 || head_dim != 128 || value_heads % key_heads != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    const std::uint32_t total = batch_size * conv_dim;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_qwen36_gdn_prep_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state_table, key_heads, value_heads,
        head_dim, conv_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_l2_norm_heads_128_kernel<<<batch_size * value_heads, 128, 0, stream>>>(
        q, batch_size * value_heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_l2_norm_heads_128_kernel<<<batch_size * value_heads, 128, 0, stream>>>(
        k, batch_size * value_heads);
    return cudaGetLastError();
}

__global__ void infer_qwen36_gdn_gate_kernel(const float* alpha,
                                                   const float* beta_input,
                                                   const std::uint16_t* a_log_bf16,
                                                   const std::uint16_t* dt_bias_bf16,
                                                   float* gate,
                                                   float* beta,
                                                   std::uint32_t heads) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads) {
        return;
    }
    const __nv_bfloat16 a_log_raw =
        *reinterpret_cast<const __nv_bfloat16*>(a_log_bf16 + idx);
    const __nv_bfloat16 dt_bias_raw =
        *reinterpret_cast<const __nv_bfloat16*>(dt_bias_bf16 + idx);
    const float a_log = __bfloat162float(a_log_raw);
    const float dt_bias = __bfloat162float(dt_bias_raw);
    const float dt = alpha[idx] + dt_bias;
    const float softplus = log1pf(expf(-fabsf(dt))) + fmaxf(dt, 0.0f);
    gate[idx] = -expf(a_log) * softplus;
    beta[idx] = 1.0f / (1.0f + expf(-beta_input[idx]));
}

extern "C" cudaError_t infer_qwen36_gdn_gate_on_stream(
    const float* alpha,
    const float* beta_input,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    float* gate,
    float* beta,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (alpha == nullptr || beta_input == nullptr || a_log_bf16 == nullptr ||
        dt_bias_bf16 == nullptr || gate == nullptr || beta == nullptr || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((heads + kThreads - 1) / kThreads);
    infer_qwen36_gdn_gate_kernel<<<blocks, kThreads, 0, stream>>>(
        alpha, beta_input, a_log_bf16, dt_bias_bf16, gate, beta, heads);
    return cudaGetLastError();
}

__global__ void infer_qwen36_gdn_gate_batch_kernel(
    const float* alpha,
    const float* beta_input,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    float* gate,
    float* beta,
    std::uint32_t heads,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const std::uint32_t head = idx % heads;
    const float a_log = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(a_log_bf16 + head));
    const float dt_bias = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(dt_bias_bf16 + head));
    const float dt = alpha[idx] + dt_bias;
    const float softplus = log1pf(expf(-fabsf(dt))) + fmaxf(dt, 0.0f);
    gate[idx] = -expf(a_log) * softplus;
    beta[idx] = 1.0f / (1.0f + expf(-beta_input[idx]));
}

extern "C" cudaError_t infer_qwen36_gdn_gate_batch_on_stream(
    const float* alpha,
    const float* beta_input,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    float* gate,
    float* beta,
    std::uint32_t batch_size,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (alpha == nullptr || beta_input == nullptr || a_log_bf16 == nullptr ||
        dt_bias_bf16 == nullptr || gate == nullptr || beta == nullptr ||
        batch_size == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = batch_size * heads;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_qwen36_gdn_gate_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        alpha, beta_input, a_log_bf16, dt_bias_bf16, gate, beta, heads, len);
    return cudaGetLastError();
}

__global__ void infer_gated_rms_norm_f32_kernel(const float* input,
                                                      const float* gate,
                                                      const float* weight,
                                                      float* output,
                                                      std::uint32_t rows,
                                                      std::uint32_t cols,
                                                      float eps) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_input = input + row * cols;
    const float* row_gate = gate + row * cols;
    float* row_output = output + row * cols;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = row_input[col];
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(cols) + eps);
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float gate_value = row_gate[col];
        const float silu_gate = gate_value / (1.0f + expf(-gate_value));
        row_output[col] = row_input[col] * inv_rms * weight[col] * silu_gate;
    }
}

extern "C" cudaError_t infer_gated_rms_norm_f32_on_stream(const float* input,
                                                                const float* gate,
                                                                const float* weight,
                                                                float* output,
                                                                std::uint32_t rows,
                                                                std::uint32_t cols,
                                                                float eps,
                                                                cudaStream_t stream) {
    if (input == nullptr || gate == nullptr || weight == nullptr || output == nullptr ||
        rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_gated_rms_norm_f32_kernel<<<rows, kThreads, kThreads * sizeof(float), stream>>>(
        input, gate, weight, output, rows, cols, eps);
    return cudaGetLastError();
}

// Direct top-1 lm-head path. Does NOT materialize a full logits vector to
// global memory; instead the pass-1 kernel writes one (value, index) pair per
// warp to scratch_*, and pass-2 reduces that scratch into the caller's
// out_index/out_value. logits buffer may be nullptr and is not written.
extern "C" cudaError_t infer_lm_head_top1_f32_on_stream(
    const float* input,
    const std::uint16_t* weight,
    float* scratch_value,
    std::uint32_t* scratch_index,
    std::uint32_t scratch_len,
    std::uint32_t* out_index,
    float* out_value,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || scratch_value == nullptr ||
        scratch_index == nullptr || out_index == nullptr || out_value == nullptr ||
        rows == 0 || cols == 0 || scratch_len == 0) {
        return cudaErrorInvalidValue;
    }

    // 8 warps (256 threads) per block, each warp handles one row.
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpsPerBlock * 32;
    const std::uint32_t warps_per_block = kWarpsPerBlock;
    const std::uint32_t grid = (rows + warps_per_block - 1) / warps_per_block;
    if (grid * warps_per_block > scratch_len) {
        return cudaErrorInvalidValue;
    }

    const std::size_t shmem_bytes =
        static_cast<std::size_t>(cols) * sizeof(float) +            // input cache
        kWarpsPerBlock * (sizeof(float) + sizeof(std::uint32_t));  // warp scratch

    infer_lm_head_top1_pass1_kernel<<<grid, kThreads, shmem_bytes, stream>>>(
        input, weight, scratch_value, scratch_index, rows, cols);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    constexpr int kFinalThreads = 128;
    const std::size_t final_shmem =
        kFinalThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_lm_head_top1_final_kernel<<<1, kFinalThreads, final_shmem, stream>>>(
        scratch_value, scratch_index, out_index, out_value,
        grid * warps_per_block);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// W4A16 NVFP4 matvec: f32 input × (E2M1 weight × UE4M3 per-block scale) → f32
//
// ModelOpt stores the weight as row-major [out, in] packed E2M1 (2 per byte,
// low nibble first), with per-16-element UE4M3 block scales in [out, in/16]
// row-major, and a scalar weight_scale_2. For W4A16 the activation stays f32;
// only weight_scale_2 is applied as the output scalar.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float infer_e2m1_value(std::uint8_t nibble) {
    const std::uint32_t magnitude = nibble & 0x7u;
    const std::uint32_t exponent = magnitude >> 1u;
    const std::uint32_t mantissa = magnitude & 1u;
    const std::uint32_t magnitude_bits = exponent == 0
        ? mantissa * 0x3f000000u
        : ((exponent + 126u) << 23u) | (mantissa << 22u);
    const std::uint32_t sign_bit = static_cast<std::uint32_t>(nibble & 0x8u) << 28u;
    return __uint_as_float(sign_bit | magnitude_bits);
}

__device__ float infer_ue4m3_value(std::uint8_t code) {
    if ((code & 0x80) == 0) {
        const std::uint8_t exp = (code >> 3) & 0x0f;
        const std::uint8_t mant = code & 0x07;
        if (exp == 0) {
            return static_cast<float>(mant) * 0.001953125f;
        }
        if (exp == 0x0f && mant == 0x07) {
            return NAN;
        }
        return __uint_as_float((static_cast<std::uint32_t>(exp) + 120u) << 23 |
                               static_cast<std::uint32_t>(mant) << 20);
    }
    const float sign = (code & 0x80) ? -1.0f : 1.0f;
    const std::uint8_t exp = (code >> 3) & 0x0f;
    const std::uint8_t mant = code & 0x07;
    if (exp == 0) {
        return sign * static_cast<float>(mant) * exp2f(-9.0f);
    }
    if (exp == 0x0f && mant == 0x07) {
        return NAN;
    }
    return sign * (1.0f + static_cast<float>(mant) / 8.0f) *
           exp2f(static_cast<float>(exp) - 7.0f);
}

__global__ void infer_nvfp4_w4a16_matvec_f32_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ packed_weight,  // [out, in] packed E2M1
    const std::uint8_t* __restrict__ weight_scale,   // [out, in/16] UE4M3
    float* output,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t in_blocks = in_features / 16;
    const std::uint32_t row_byte_base = row * (in_features / 2);
    const std::uint32_t row_scale_base = row * in_blocks;

    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        const std::uint8_t byte = packed_weight[row_byte_base + col / 2];
        const std::uint8_t nibble = (col & 1) ? (byte >> 4) : (byte & 0x0f);
        const float w_val = infer_e2m1_value(nibble) *
                            infer_ue4m3_value(weight_scale[row_scale_base + col / 16]);
        sum += input[col] * w_val;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = partial[0] * weight_scale_2;
    }
}

__device__ inline float infer_nvfp4_row_dot_warp(
    const std::uint8_t* packed_row,
    const std::uint8_t* row_scale,
    const float* input_sh,
    std::uint32_t cols);

__global__ void infer_nvfp4_w4a16_matvec_f32_warp_rows_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ packed_weight,
    const std::uint8_t* __restrict__ weight_scale,
    float* __restrict__ output,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2) {
    extern __shared__ float input_sh[];
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t warps_per_block = blockDim.x >> 5u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * warps_per_block + warp;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t row_byte_base = row * (in_features / 2);
    const std::uint32_t row_scale_base = row * (in_features / 16);
    const float value = infer_nvfp4_row_dot_warp(
        packed_weight + row_byte_base,
        weight_scale + row_scale_base,
        input_sh,
        in_features) * weight_scale_2;
    if (lane == 0) {
        output[row] = value;
    }
}

__global__ void infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ packed_weight,
    const std::uint8_t* __restrict__ weight_scale,
    float* __restrict__ output,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2) {
    extern __shared__ float input_sh[];
    const std::uint32_t batch = blockIdx.y;
    input += batch * in_features;
    output += batch * out_features;
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t warps_per_block = blockDim.x >> 5u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * warps_per_block + warp;
    if (row >= out_features) return;
    const std::uint32_t row_byte_base = row * (in_features / 2);
    const std::uint32_t row_scale_base = row * (in_features / 16);
    const float value = infer_nvfp4_row_dot_warp(
        packed_weight + row_byte_base,
        weight_scale + row_scale_base,
        input_sh,
        in_features) * weight_scale_2;
    if (lane == 0) output[row] = value;
}

__global__ void infer_nvfp4_w4a16_grouped_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* const* __restrict__ packed_weight_table,
    const std::uint8_t* const* __restrict__ weight_scale_table,
    const float* __restrict__ weight_scale_2_table,
    float* const* __restrict__ output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    extern __shared__ float input_sh[];
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t group = blockIdx.y;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (group >= groups || row >= out_features) {
        return;
    }
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) {
        return;
    }

    const std::uint8_t* packed_weight = packed_weight_table[expert];
    const std::uint8_t* weight_scale = weight_scale_table[expert];
    const std::uint32_t row_byte_base = row * (in_features / 2);
    const std::uint32_t row_scale_base = row * (in_features / 16);
    const float value = infer_nvfp4_row_dot_warp(
        packed_weight + row_byte_base,
        weight_scale + row_scale_base,
        input_sh,
        in_features) * weight_scale_2_table[expert];
    if (lane == 0) {
        output_table[group][row] = value;
    }
}

__device__ inline float infer_nvfp4_row_dot_warp(
    const std::uint8_t* packed_row,
    const std::uint8_t* row_scale,
    const float* input_sh,
    std::uint32_t cols) {
    float acc = 0.0f;
    const std::uint32_t lane = threadIdx.x & 31u;
    for (std::uint32_t col = lane * 4; col < cols; col += 32 * 4) {
        const std::uint8_t b0 = packed_row[col / 2];
        const std::uint8_t b1 = packed_row[col / 2 + 1];
        const float scale = infer_ue4m3_value(row_scale[col / 16]);
        acc = __fmaf_rn(input_sh[col], infer_e2m1_value(b0 & 0x0f) * scale, acc);
        acc = __fmaf_rn(input_sh[col + 1], infer_e2m1_value(b0 >> 4) * scale, acc);
        acc = __fmaf_rn(input_sh[col + 2], infer_e2m1_value(b1 & 0x0f) * scale, acc);
        acc = __fmaf_rn(input_sh[col + 3], infer_e2m1_value(b1 >> 4) * scale, acc);
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    return acc;
}

__global__ void infer_nvfp4_w4a16_top1_pass1_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ packed_weight,
    const std::uint8_t* __restrict__ weight_scale,
    float* __restrict__ scratch_value,
    std::uint32_t* __restrict__ scratch_index,
    std::uint32_t rows,
    std::uint32_t cols,
    float weight_scale_2) {
    extern __shared__ float input_sh[];
    for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        input_sh[i] = input[i];
    }
    const std::uint32_t warps_in_block = blockDim.x >> 5u;
    float* warp_values = input_sh + cols;
    std::uint32_t* warp_indices =
        reinterpret_cast<std::uint32_t*>(warp_values + warps_in_block);
    __syncthreads();

    const std::uint32_t warp_id = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * warps_in_block + warp_id;
    float logit = -INFINITY;
    std::uint32_t row_index = 0;
    if (row < rows) {
        const std::uint32_t row_byte_base = row * (cols / 2);
        const std::uint32_t row_scale_base = row * (cols / 16);
        logit = infer_nvfp4_row_dot_warp(
            packed_weight + row_byte_base,
            weight_scale + row_scale_base,
            input_sh,
            cols) * weight_scale_2;
        row_index = row;
    }
    if (lane == 0) {
        warp_values[warp_id] = logit;
        warp_indices[warp_id] = row_index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float best_value = lane < warps_in_block ? warp_values[lane] : -INFINITY;
        std::uint32_t best_index = lane < warps_in_block ? warp_indices[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
            const std::uint32_t other_index =
                __shfl_down_sync(0xffffffffu, best_index, offset);
            if (lane + offset < warps_in_block &&
                (other_value > best_value ||
                 (other_value == best_value && other_index < best_index))) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            scratch_value[blockIdx.x] = best_value;
            scratch_index[blockIdx.x] = best_index;
        }
    }
}

extern "C" cudaError_t infer_nvfp4_w4a16_matvec_f32_on_stream(
    const float* input,
    const std::uint8_t* packed_weight,
    const std::uint8_t* weight_scale,
    float* output,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2,
    cudaStream_t stream) {
    if (input == nullptr || packed_weight == nullptr || weight_scale == nullptr ||
        output == nullptr || out_features == 0 || in_features == 0 ||
        (in_features % 16) != 0 || !isfinite(weight_scale_2)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_nvfp4_w4a16_matvec_f32_kernel<<<
        out_features, kThreads, kThreads * sizeof(float), stream>>>(
        input, packed_weight, weight_scale, output,
        out_features, in_features, weight_scale_2);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nvfp4_w4a16_matvec_f32_warp_rows_on_stream(
    const float* input,
    const std::uint8_t* packed_weight,
    const std::uint8_t* weight_scale,
    float* output,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2,
    std::uint32_t warps_per_block,
    cudaStream_t stream) {
    if (input == nullptr || packed_weight == nullptr || weight_scale == nullptr ||
        output == nullptr || out_features == 0 || in_features == 0 ||
        (in_features % 16) != 0 || !isfinite(weight_scale_2) ||
        (warps_per_block != 4 && warps_per_block != 8 &&
         warps_per_block != 16 && warps_per_block != 32)) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t threads = warps_per_block * 32;
    const std::uint32_t grid = (out_features + warps_per_block - 1) / warps_per_block;
    const std::size_t shmem_bytes = static_cast<std::size_t>(in_features) * sizeof(float);
    infer_nvfp4_w4a16_matvec_f32_warp_rows_kernel<<<
        grid, threads, shmem_bytes, stream>>>(
        input, packed_weight, weight_scale, output,
        out_features, in_features, weight_scale_2);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_on_stream(
    const float* input,
    const std::uint8_t* packed_weight,
    const std::uint8_t* weight_scale,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2,
    std::uint32_t warps_per_block,
    cudaStream_t stream) {
    if (input == nullptr || packed_weight == nullptr || weight_scale == nullptr ||
        output == nullptr || batch_size == 0 || out_features == 0 || in_features == 0 ||
        (in_features % 16) != 0 || !isfinite(weight_scale_2) ||
        (warps_per_block != 4 && warps_per_block != 8 &&
         warps_per_block != 16 && warps_per_block != 32)) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t threads = warps_per_block * 32;
    const std::uint32_t grid_x = (out_features + warps_per_block - 1) / warps_per_block;
    const std::size_t shmem_bytes = static_cast<std::size_t>(in_features) * sizeof(float);
    infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_kernel<<<
        dim3(grid_x, batch_size), threads, shmem_bytes, stream>>>(
        input, packed_weight, weight_scale, output,
        out_features, in_features, weight_scale_2);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nvfp4_w4a16_grouped_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* const* packed_weight_table,
    const std::uint8_t* const* weight_scale_table,
    const float* weight_scale_2_table,
    float* const* output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr || packed_weight_table == nullptr ||
        weight_scale_table == nullptr || weight_scale_2_table == nullptr ||
        output_table == nullptr || table_len == 0 || groups == 0 ||
        out_features == 0 || in_features == 0 || (in_features % 16) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid((out_features + kWarpsPerBlock - 1) / kWarpsPerBlock, groups);
    const std::size_t shmem_bytes = static_cast<std::size_t>(in_features) * sizeof(float);
    infer_nvfp4_w4a16_grouped_matvec_f32_kernel<<<
        grid, kThreads, shmem_bytes, stream>>>(
        indices, input, packed_weight_table, weight_scale_table, weight_scale_2_table,
        output_table, table_len, groups, out_features, in_features);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nvfp4_w4a16_top1_f32_on_stream(
    const float* input,
    const std::uint8_t* packed_weight,
    const std::uint8_t* weight_scale,
    float* scratch_value,
    std::uint32_t* scratch_index,
    std::uint32_t scratch_len,
    std::uint32_t* out_index,
    float* out_value,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2,
    std::uint32_t warps_per_block,
    cudaStream_t stream) {
    if (input == nullptr || packed_weight == nullptr || weight_scale == nullptr ||
        scratch_value == nullptr || scratch_index == nullptr || out_index == nullptr ||
        out_value == nullptr || out_features == 0 || in_features == 0 ||
        (in_features % 16) != 0 || !isfinite(weight_scale_2) ||
        (warps_per_block != 4 && warps_per_block != 8 &&
         warps_per_block != 16 && warps_per_block != 32)) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t threads = warps_per_block * 32;
    const std::uint32_t grid = (out_features + warps_per_block - 1) / warps_per_block;
    if (grid > scratch_len) {
        return cudaErrorInvalidValue;
    }
    const std::size_t shmem_bytes = static_cast<std::size_t>(in_features) * sizeof(float) +
        warps_per_block * (sizeof(float) + sizeof(std::uint32_t));
    infer_nvfp4_w4a16_top1_pass1_kernel<<<grid, threads, shmem_bytes, stream>>>(
        input, packed_weight, weight_scale, scratch_value, scratch_index,
        out_features, in_features, weight_scale_2);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    constexpr int kFinalThreads = 128;
    const std::size_t final_shmem = kFinalThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_lm_head_top1_final_kernel<<<1, kFinalThreads, final_shmem, stream>>>(
        scratch_value, scratch_index, out_index, out_value, grid);
    return cudaGetLastError();
}
