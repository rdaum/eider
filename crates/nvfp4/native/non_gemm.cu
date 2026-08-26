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
    if (col >= cols) {
        return;
    }

    const std::uint32_t row_start = row_block * 16;
    const std::uint32_t row_end = min(row_start + 16, rows);
    const std::uint32_t lane = threadIdx.x;
    float max_abs = 0.0f;
    if (row_start + lane < row_end) {
        const float value = input[row_start + lane + col * rows] / input_scale;
        max_abs = isfinite(value) ? fabsf(value) : 0.0f;
    }
    max_abs = infer_warp_reduce_max(max_abs);

    std::uint32_t scale_word = 0;
    if (lane == 0) {
        scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        scales[infer_ue4m3_tiled_scale_offset(col, row_block, rows)] =
            static_cast<std::uint8_t>(scale_word);
    }
    scale_word = __shfl_sync(0xffffffffu, scale_word, 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(scale_word);
    const float scale = infer_e4m3_value(scale_code);

    if (lane < 8 && row_start + lane * 2 < row_end) {
        const std::uint32_t row = row_start + lane * 2;
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
    infer_quantize_nvfp4_col_major_f32_kernel<<<cols * row_blocks, 32>>>(
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
    infer_quantize_nvfp4_col_major_f32_kernel<<<cols * row_blocks, 32, 0, stream>>>(
        input, packed, scales, rows, cols, input_scale);
    return cudaGetLastError();
}

__global__ void infer_rms_norm_quantize_nvfp4_col_major_f32_kernel(
    const float* input,
    const float* weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    float input_scale) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t warps = blockDim.x / 32;
    const float* row_input = input + row * cols;

    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = row_input[col];
        square_sum += value * value;
    }
    square_sum = infer_block_reduce_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const std::uint32_t feature_blocks = (cols + 15) / 16;
    const std::uint32_t feature_pairs = (feature_blocks + 1) / 2;
    for (std::uint32_t feature_pair = warp; feature_pair < feature_pairs;
         feature_pair += warps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t feature_block = feature_pair * 2 + half;
        const std::uint32_t feature = feature_pair * 32 + lane;
        float value = 0.0f;
        if (feature < cols) {
            value = row_input[feature] * inverse_rms * weight[feature] / input_scale;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && feature_block < feature_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[infer_ue4m3_tiled_scale_offset(row, feature_block, cols)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
        const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
        if (half_lane < 8 && feature_block < feature_blocks) {
            const std::uint32_t lo_feature = feature_block * 16 + half_lane * 2;
            if (lo_feature < cols) {
                const std::uint8_t lo = static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp4(
                        scale == 0.0f ? 0.0f : lo_value / scale,
                        __NV_E2M1, cudaRoundNearest) & 0x0f);
                std::uint8_t hi = 0;
                if (lo_feature + 1 < cols) {
                    hi = static_cast<std::uint8_t>(
                        __nv_cvt_float_to_fp4(
                            scale == 0.0f ? 0.0f : hi_value / scale,
                            __NV_E2M1, cudaRoundNearest) & 0x0f);
                }
                packed[(row * cols + lo_feature) / 2] = lo | (hi << 4);
            }
        }
    }
}

extern "C" cudaError_t infer_rms_norm_quantize_nvfp4_col_major_f32_on_stream(
    const float* input,
    const float* weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || packed == nullptr || scales == nullptr ||
        rows == 0 || cols == 0 || input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_rms_norm_quantize_nvfp4_col_major_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, packed, scales, rows, cols, eps, input_scale);
    return cudaGetLastError();
}

__global__ void infer_rms_norm_quantize_nvfp4_pair_col_major_f32_kernel(
    const float* input,
    const float* weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint8_t* residual_packed,
    std::uint8_t* residual_scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    float input_scale) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5;
    constexpr std::uint32_t kWarps = 8;
    const float* row_input = input + row * cols;

    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = row_input[col];
        square_sum += value * value;
    }
    square_sum = infer_block_reduce_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const std::uint32_t feature_blocks = (cols + 15) / 16;
    const std::uint32_t feature_pairs = (feature_blocks + 1) / 2;
    for (std::uint32_t feature_pair = warp; feature_pair < feature_pairs;
         feature_pair += kWarps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t feature_block = feature_pair * 2 + half;
        const std::uint32_t feature = feature_pair * 32 + lane;
        float value = 0.0f;
        if (feature < cols) {
            value = row_input[feature] * inverse_rms * weight[feature] / input_scale;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && feature_block < feature_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[infer_ue4m3_tiled_scale_offset(row, feature_block, cols)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint8_t code = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp4(
                scale == 0.0f ? 0.0f : value / scale,
                __NV_E2M1, cudaRoundNearest) & 0x0f);
        const float residual = value - infer_e2m1_value(code) * scale;

        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const std::uint32_t lo_code = __shfl_sync(mask, static_cast<std::uint32_t>(code), pair_lane, 16);
        const std::uint32_t hi_code = __shfl_sync(mask, static_cast<std::uint32_t>(code), pair_lane + 1, 16);
        if (half_lane < 8 && feature_block < feature_blocks) {
            const std::uint32_t lo_feature = feature_block * 16 + half_lane * 2;
            if (lo_feature < cols) {
                packed[(row * cols + lo_feature) / 2] =
                    static_cast<std::uint8_t>(lo_code | (hi_code << 4));
            }
        }

        float residual_max_abs = fabsf(residual);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            residual_max_abs =
                fmaxf(residual_max_abs, __shfl_down_sync(mask, residual_max_abs, offset, 16));
        }
        std::uint32_t residual_scale_word = 0;
        if (half_lane == 0 && feature_block < feature_blocks) {
            residual_scale_word = residual_max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(residual_max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            residual_scales[infer_ue4m3_tiled_scale_offset(row, feature_block, cols)] =
                static_cast<std::uint8_t>(residual_scale_word);
        }
        residual_scale_word = __shfl_sync(mask, residual_scale_word, 0, 16);
        const float residual_scale =
            infer_e4m3_value(static_cast<std::uint8_t>(residual_scale_word));
        const std::uint8_t residual_code = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp4(
                residual_scale == 0.0f ? 0.0f : residual / residual_scale,
                __NV_E2M1, cudaRoundNearest) & 0x0f);
        const std::uint32_t lo_residual_code = __shfl_sync(
            mask, static_cast<std::uint32_t>(residual_code), pair_lane, 16);
        const std::uint32_t hi_residual_code = __shfl_sync(
            mask, static_cast<std::uint32_t>(residual_code), pair_lane + 1, 16);
        if (half_lane < 8 && feature_block < feature_blocks) {
            const std::uint32_t lo_feature = feature_block * 16 + half_lane * 2;
            if (lo_feature < cols) {
                residual_packed[(row * cols + lo_feature) / 2] =
                    static_cast<std::uint8_t>(lo_residual_code | (hi_residual_code << 4));
            }
        }
    }
}

extern "C" cudaError_t infer_rms_norm_quantize_nvfp4_pair_col_major_f32_on_stream(
    const float* input,
    const float* weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint8_t* residual_packed,
    std::uint8_t* residual_scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || packed == nullptr || scales == nullptr ||
        residual_packed == nullptr || residual_scales == nullptr || rows == 0 || cols == 0 ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_rms_norm_quantize_nvfp4_pair_col_major_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, packed, scales, residual_packed, residual_scales,
        rows, cols, eps, input_scale);
    return cudaGetLastError();
}

__global__ void infer_gelu_tanh_mul_quantize_nvfp4_col_major_f32_kernel(
    const float* gate,
    const float* up,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float input_scale) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5;
    constexpr std::uint32_t kWarps = 8;
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const std::uint32_t feature_blocks = (cols + 15) / 16;
    const std::uint32_t feature_pairs = (feature_blocks + 1) / 2;
    for (std::uint32_t feature_pair = warp; feature_pair < feature_pairs;
         feature_pair += kWarps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t feature_block = feature_pair * 2 + half;
        const std::uint32_t feature = feature_pair * 32 + lane;
        float value = 0.0f;
        if (feature < cols) {
            const std::uint32_t index = row * cols + feature;
            const float gate_value = gate[index];
            const float cubic = gate_value * gate_value * gate_value;
            const float gelu = 0.5f * gate_value *
                (1.0f + tanhf(kSqrtTwoOverPi *
                               (gate_value + 0.044715f * cubic)));
            value = gelu * up[index] / input_scale;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && feature_block < feature_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[infer_ue4m3_tiled_scale_offset(row, feature_block, cols)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
        const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
        if (half_lane < 8 && feature_block < feature_blocks) {
            const std::uint32_t lo_feature = feature_block * 16 + half_lane * 2;
            if (lo_feature < cols) {
                const std::uint8_t lo = static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp4(
                        scale == 0.0f ? 0.0f : lo_value / scale,
                        __NV_E2M1, cudaRoundNearest) & 0x0f);
                std::uint8_t hi = 0;
                if (lo_feature + 1 < cols) {
                    hi = static_cast<std::uint8_t>(
                        __nv_cvt_float_to_fp4(
                            scale == 0.0f ? 0.0f : hi_value / scale,
                            __NV_E2M1, cudaRoundNearest) & 0x0f);
                }
                packed[(row * cols + lo_feature) / 2] = lo | (hi << 4);
            }
        }
    }
}

extern "C" cudaError_t infer_gelu_tanh_mul_quantize_nvfp4_col_major_f32_on_stream(
    const float* gate,
    const float* up,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float input_scale,
    cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || packed == nullptr || scales == nullptr ||
        rows == 0 || cols == 0 || input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_gelu_tanh_mul_quantize_nvfp4_col_major_f32_kernel<<<
        rows, kThreads, 0, stream>>>(
        gate, up, packed, scales, rows, cols, input_scale);
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

__global__ void infer_rms_norm_add_f32_kernel(
    const float* input,
    const float* weight,
    const float* residual,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[row_offset + col];
        square_sum += value * value;
    }
    square_sum = infer_block_reduce_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row_offset + col] =
            input[row_offset + col] * inverse_rms * weight[col] + residual[row_offset + col];
    }
}

extern "C" cudaError_t infer_rms_norm_add_f32_on_stream(
    const float* input,
    const float* weight,
    const float* residual,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || residual == nullptr || output == nullptr ||
        rows == 0 || cols == 0 || eps < 0.0f || !isfinite(eps)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_rms_norm_add_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, residual, output, rows, cols, eps);
    return cudaGetLastError();
}

__global__ void infer_rms_norm_add_then_rms_norm_quantize_nvfp4_f32_kernel(
    const float* input,
    const float* input_weight,
    const float* residual,
    float* output,
    const float* quant_weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float input_eps,
    float quant_eps,
    float input_scale) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5;
    constexpr std::uint32_t kWarps = 8;
    const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
    extern __shared__ float staged[];

    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[row_offset + col];
        square_sum += value * value;
    }
    square_sum = infer_block_reduce_sum(square_sum);
    __shared__ float inverse_rms[2];
    if (threadIdx.x == 0) {
        inverse_rms[0] = rsqrtf(square_sum / static_cast<float>(cols) + input_eps);
    }
    __syncthreads();

    float output_square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value =
            input[row_offset + col] * inverse_rms[0] * input_weight[col] +
            residual[row_offset + col];
        output[row_offset + col] = value;
        staged[col] = value;
        output_square_sum += value * value;
    }
    output_square_sum = infer_block_reduce_sum(output_square_sum);
    if (threadIdx.x == 0) {
        inverse_rms[1] = rsqrtf(output_square_sum / static_cast<float>(cols) + quant_eps);
    }
    __syncthreads();

    const std::uint32_t feature_blocks = (cols + 15) / 16;
    const std::uint32_t feature_pairs = (feature_blocks + 1) / 2;
    for (std::uint32_t feature_pair = warp; feature_pair < feature_pairs;
         feature_pair += kWarps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t feature_block = feature_pair * 2 + half;
        const std::uint32_t feature = feature_pair * 32 + lane;
        float value = 0.0f;
        if (feature < cols) {
            value = staged[feature] * inverse_rms[1] * quant_weight[feature] / input_scale;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && feature_block < feature_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[infer_ue4m3_tiled_scale_offset(row, feature_block, cols)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
        const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
        if (half_lane < 8 && feature_block < feature_blocks) {
            const std::uint32_t lo_feature = feature_block * 16 + half_lane * 2;
            if (lo_feature < cols) {
                const std::uint8_t lo = static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp4(
                        scale == 0.0f ? 0.0f : lo_value / scale,
                        __NV_E2M1, cudaRoundNearest) & 0x0f);
                std::uint8_t hi = 0;
                if (lo_feature + 1 < cols) {
                    hi = static_cast<std::uint8_t>(
                        __nv_cvt_float_to_fp4(
                            scale == 0.0f ? 0.0f : hi_value / scale,
                            __NV_E2M1, cudaRoundNearest) & 0x0f);
                }
                packed[(row * cols + lo_feature) / 2] = lo | (hi << 4);
            }
        }
    }
}

extern "C" cudaError_t infer_rms_norm_add_then_rms_norm_quantize_nvfp4_f32_on_stream(
    const float* input,
    const float* input_weight,
    const float* residual,
    float* output,
    const float* quant_weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t cols,
    float input_eps,
    float quant_eps,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || input_weight == nullptr || residual == nullptr ||
        output == nullptr || quant_weight == nullptr || packed == nullptr || scales == nullptr ||
        rows == 0 || cols == 0 || input_eps < 0.0f || !isfinite(input_eps) ||
        quant_eps < 0.0f || !isfinite(quant_eps) ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_rms_norm_add_then_rms_norm_quantize_nvfp4_f32_kernel<<<
        rows, kThreads, static_cast<std::size_t>(cols) * sizeof(float), stream>>>(
        input, input_weight, residual, output, quant_weight, packed, scales,
        rows, cols, input_eps, quant_eps, input_scale);
    return cudaGetLastError();
}

__global__ void infer_dual_rms_norm_add_f32_kernel(
    const float* left,
    const float* left_weight,
    const float* right,
    const float* right_weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float left_eps,
    float right_eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
    float left_square_sum = 0.0f;
    float right_square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float left_value = left[row_offset + col];
        const float right_value = right[row_offset + col];
        left_square_sum += left_value * left_value;
        right_square_sum += right_value * right_value;
    }
    left_square_sum = infer_block_reduce_sum(left_square_sum);
    __shared__ float inverse_rms[2];
    if (threadIdx.x == 0) {
        inverse_rms[0] = rsqrtf(left_square_sum / static_cast<float>(cols) + left_eps);
    }
    __syncthreads();
    right_square_sum = infer_block_reduce_sum(right_square_sum);
    if (threadIdx.x == 0) {
        inverse_rms[1] = rsqrtf(right_square_sum / static_cast<float>(cols) + right_eps);
    }
    __syncthreads();
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row_offset + col] =
            left[row_offset + col] * inverse_rms[0] * left_weight[col] +
            right[row_offset + col] * inverse_rms[1] * right_weight[col];
    }
}

extern "C" cudaError_t infer_dual_rms_norm_add_f32_on_stream(
    const float* left,
    const float* left_weight,
    const float* right,
    const float* right_weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float left_eps,
    float right_eps,
    cudaStream_t stream) {
    if (left == nullptr || left_weight == nullptr || right == nullptr ||
        right_weight == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        left_eps < 0.0f || !isfinite(left_eps) || right_eps < 0.0f || !isfinite(right_eps)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_dual_rms_norm_add_f32_kernel<<<rows, kThreads, 0, stream>>>(
        left, left_weight, right, right_weight, output, rows, cols, left_eps, right_eps);
    return cudaGetLastError();
}

__global__ void infer_rms_norm_add_channel_row_scale_f32_kernel(
    const float* input,
    const float* weight,
    const float* residual,
    const float* channel_scale,
    const float* row_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = input[row_offset + col];
        square_sum += value * value;
    }
    square_sum = infer_block_reduce_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    const float row_multiplier = row_scale[row];
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float combined =
            input[row_offset + col] * inverse_rms * weight[col] + residual[row_offset + col];
        output[row_offset + col] = combined * channel_scale[col] * row_multiplier;
    }
}

extern "C" cudaError_t infer_rms_norm_add_channel_row_scale_f32_on_stream(
    const float* input,
    const float* weight,
    const float* residual,
    const float* channel_scale,
    const float* row_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || residual == nullptr ||
        channel_scale == nullptr || row_scale == nullptr || output == nullptr ||
        rows == 0 || cols == 0 || eps < 0.0f || !isfinite(eps)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_rms_norm_add_channel_row_scale_f32_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, residual, channel_scale, row_scale, output, rows, cols, eps);
    return cudaGetLastError();
}

__global__ void infer_dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_kernel(
    const float* left,
    const float* left_weight,
    const float* right,
    const float* right_weight,
    const float* final_weight,
    const float* residual,
    const float* channel_scale,
    const float* row_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float left_eps,
    float right_eps,
    float final_eps) {
    const std::uint32_t row = blockIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
    extern __shared__ float combined[];
    float left_square_sum = 0.0f;
    float right_square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float left_value = left[row_offset + col];
        const float right_value = right[row_offset + col];
        left_square_sum += left_value * left_value;
        right_square_sum += right_value * right_value;
    }
    left_square_sum = infer_block_reduce_sum(left_square_sum);
    __shared__ float inverse_rms[3];
    if (threadIdx.x == 0) {
        inverse_rms[0] = rsqrtf(left_square_sum / static_cast<float>(cols) + left_eps);
    }
    __syncthreads();
    right_square_sum = infer_block_reduce_sum(right_square_sum);
    if (threadIdx.x == 0) {
        inverse_rms[1] = rsqrtf(right_square_sum / static_cast<float>(cols) + right_eps);
    }
    __syncthreads();

    float combined_square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value =
            left[row_offset + col] * inverse_rms[0] * left_weight[col] +
            right[row_offset + col] * inverse_rms[1] * right_weight[col];
        combined[col] = value;
        combined_square_sum += value * value;
    }
    combined_square_sum = infer_block_reduce_sum(combined_square_sum);
    if (threadIdx.x == 0) {
        inverse_rms[2] = rsqrtf(combined_square_sum / static_cast<float>(cols) + final_eps);
    }
    __syncthreads();

    const float row_multiplier = row_scale[row];
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value =
            combined[col] * inverse_rms[2] * final_weight[col] + residual[row_offset + col];
        output[row_offset + col] = value * channel_scale[col] * row_multiplier;
    }
}

extern "C" cudaError_t
infer_dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_on_stream(
    const float* left,
    const float* left_weight,
    const float* right,
    const float* right_weight,
    const float* final_weight,
    const float* residual,
    const float* channel_scale,
    const float* row_scale,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float left_eps,
    float right_eps,
    float final_eps,
    cudaStream_t stream) {
    if (left == nullptr || left_weight == nullptr || right == nullptr || right_weight == nullptr ||
        final_weight == nullptr || residual == nullptr || channel_scale == nullptr ||
        row_scale == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        left_eps < 0.0f || !isfinite(left_eps) || right_eps < 0.0f ||
        !isfinite(right_eps) || final_eps < 0.0f || !isfinite(final_eps)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_dual_rms_norm_add_then_rms_norm_add_channel_row_scale_f32_kernel<<<
        rows, kThreads, static_cast<std::size_t>(cols) * sizeof(float), stream>>>(
        left, left_weight, right, right_weight, final_weight, residual,
        channel_scale, row_scale, output, rows, cols, left_eps, right_eps, final_eps);
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

__global__ void infer_gelu_tanh_f32_kernel(const float* input,
                                           float* output,
                                           std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const float value = input[idx];
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const float cubic = value * value * value;
    output[idx] = 0.5f * value * (1.0f + tanhf(kSqrtTwoOverPi * (value + 0.044715f * cubic)));
}

extern "C" cudaError_t infer_gelu_tanh_f32_on_stream(const float* input,
                                                       float* output,
                                                       std::uint32_t len,
                                                       cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_gelu_tanh_f32_kernel<<<blocks, kThreads, 0, stream>>>(input, output, len);
    return cudaGetLastError();
}

__global__ void infer_gelu_tanh_mul_f32_kernel(const float* gate,
                                                const float* up,
                                                float* output,
                                                std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const float value = gate[idx];
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const float cubic = value * value * value;
    const float gelu = 0.5f * value *
        (1.0f + tanhf(kSqrtTwoOverPi * (value + 0.044715f * cubic)));
    output[idx] = gelu * up[idx];
}

extern "C" cudaError_t infer_gelu_tanh_mul_f32_on_stream(
    const float* gate,
    const float* up,
    float* output,
    std::uint32_t len,
    cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_gelu_tanh_mul_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        gate, up, output, len);
    return cudaGetLastError();
}

__global__ void infer_gelu_tanh_mul_halves_f32_kernel(const float* gate_up,
                                                       float* output,
                                                       std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const float gate = gate_up[idx];
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const float cubic = gate * gate * gate;
    const float gelu = 0.5f * gate *
        (1.0f + tanhf(kSqrtTwoOverPi * (gate + 0.044715f * cubic)));
    output[idx] = gelu * gate_up[len + idx];
}

extern "C" cudaError_t infer_gelu_tanh_mul_halves_f32_on_stream(
    const float* gate_up,
    float* output,
    std::uint32_t len,
    cudaStream_t stream) {
    if (gate_up == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_gelu_tanh_mul_halves_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        gate_up, output, len);
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

__global__ void infer_relu_squared_f32_kernel(const float* input,
                                               float* output,
                                               std::uint32_t len) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        const float value = fmaxf(input[index], 0.0f);
        output[index] = value * value;
    }
}

extern "C" cudaError_t infer_relu_squared_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t len,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t blocks = (len + kThreads - 1) / kThreads;
    infer_relu_squared_f32_kernel<<<blocks, kThreads, 0, stream>>>(input, output, len);
    return cudaGetLastError();
}

__global__ void infer_silu_mul_halves_clamped_f32_kernel(
    const float* gate_up,
    float* output,
    std::uint32_t len,
    float limit) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const float gate = fminf(gate_up[idx], limit);
    const float up = fminf(fmaxf(gate_up[len + idx], -limit), limit);
    output[idx] = (gate / (1.0f + expf(-gate))) * up;
}

extern "C" cudaError_t infer_silu_mul_halves_clamped_f32_on_stream(
    const float* gate_up,
    float* output,
    std::uint32_t len,
    float limit,
    cudaStream_t stream) {
    if (gate_up == nullptr || output == nullptr || len == 0 ||
        !isfinite(limit) || limit <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_silu_mul_halves_clamped_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        gate_up, output, len, limit);
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

__global__ void infer_silu_mul_halves_clamped_f32_batch_kernel(
    const float* gate_up,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float limit) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) return;
    const std::uint32_t row = idx / cols;
    const std::uint32_t col = idx - row * cols;
    const std::uint32_t base = row * cols * 2;
    const float gate = fminf(gate_up[base + col], limit);
    const float up = fminf(fmaxf(gate_up[base + cols + col], -limit), limit);
    output[idx] = (gate / (1.0f + expf(-gate))) * up;
}

extern "C" cudaError_t infer_silu_mul_halves_clamped_f32_batch_on_stream(
    const float* gate_up,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float limit,
    cudaStream_t stream) {
    if (gate_up == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        !isfinite(limit) || limit <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t len = static_cast<std::uint64_t>(rows) * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_silu_mul_halves_clamped_f32_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        gate_up, output, rows, cols, limit);
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

__global__ void infer_sigmoid_scale_heads_f32_kernel(
    const float* gate,
    const float* input,
    float* output,
    std::uint32_t head_dim,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    output[idx] = input[idx] * (1.0f / (1.0f + expf(-gate[idx / head_dim])));
}

extern "C" cudaError_t infer_sigmoid_scale_heads_f32_on_stream(
    const float* gate,
    const float* input,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (gate == nullptr || input == nullptr || output == nullptr || heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t len = heads * head_dim;
    constexpr int kThreads = 256;
    infer_sigmoid_scale_heads_f32_kernel<<<(len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        gate, input, output, head_dim, len);
    return cudaGetLastError();
}

__global__ void infer_softplus_scale_heads_f32_kernel(
    const float* gate,
    const float* input,
    float* output,
    std::uint32_t head_dim,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const float value = gate[idx / head_dim];
    const float softplus = log1pf(expf(-fabsf(value))) + fmaxf(value, 0.0f);
    output[idx] = input[idx] * softplus;
}

extern "C" cudaError_t infer_softplus_scale_heads_f32_on_stream(
    const float* gate,
    const float* input,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (gate == nullptr || input == nullptr || output == nullptr || heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t len = heads * head_dim;
    constexpr int kThreads = 256;
    infer_softplus_scale_heads_f32_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        gate, input, output, head_dim, len);
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

__global__ void infer_split_qkv_f32_batch_kernel(const float* input,
                                                  float* q,
                                                  float* k,
                                                  float* v,
                                                  std::uint32_t q_width,
                                                  std::uint32_t kv_width) {
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t width = q_width + 2 * kv_width;
    const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    const float value = input[static_cast<std::size_t>(row) * width + col];
    if (col < q_width) {
        q[static_cast<std::size_t>(row) * q_width + col] = value;
    } else if (col < q_width + kv_width) {
        k[static_cast<std::size_t>(row) * kv_width + col - q_width] = value;
    } else {
        v[static_cast<std::size_t>(row) * kv_width + col - q_width - kv_width] = value;
    }
}

extern "C" cudaError_t infer_split_qkv_f32_batch_on_stream(
    const float* input,
    float* q,
    float* k,
    float* v,
    std::uint32_t batch_rows,
    std::uint32_t q_width,
    std::uint32_t kv_width,
    cudaStream_t stream) {
    if (input == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        batch_rows == 0 || q_width == 0 || kv_width == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t width = q_width + 2 * kv_width;
    const dim3 blocks((width + kThreads - 1) / kThreads, batch_rows);
    infer_split_qkv_f32_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        input, q, k, v, q_width, kv_width);
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

__global__ void infer_step37_sigmoid_top8_f32_kernel(
    const float* logits,
    const float* bias,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t experts) {
    const std::uint32_t batch = blockIdx.x;
    logits += static_cast<std::size_t>(batch) * experts;
    out_indices += static_cast<std::size_t>(batch) * 8;
    out_weights += static_cast<std::size_t>(batch) * 8;
    constexpr int kThreads = 256;
    constexpr int kItems = 2;
    std::uint64_t keys[kItems];
    float probs[kItems];
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const std::uint32_t expert = threadIdx.x + item * kThreads;
        if (expert < experts) {
            const float prob = 1.0f / (1.0f + expf(-logits[expert]));
            float score = prob + bias[expert];
            if (isnan(score)) score = -INFINITY;
            const std::uint32_t bits = __float_as_uint(score);
            const std::uint32_t ordered =
                (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
            keys[item] = (static_cast<std::uint64_t>(ordered) << 32) |
                static_cast<std::uint64_t>(UINT32_MAX - expert);
            probs[item] = prob;
        } else {
            keys[item] = 0;
            probs[item] = 0.0f;
        }
    }

    using BlockSort = cub::BlockRadixSort<std::uint64_t, kThreads, kItems, float>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ float top_probs[8];
    __shared__ std::uint32_t top_indices[8];
    BlockSort(sort_storage).SortDescending(keys, probs);
    __syncthreads();

    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = threadIdx.x * kItems + item;
        if (rank < 8) {
            top_probs[rank] = probs[item];
            top_indices[rank] = UINT32_MAX - static_cast<std::uint32_t>(keys[item]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float selected_sum = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) selected_sum += top_probs[slot];
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            out_indices[slot] = top_indices[slot];
            out_weights[slot] = top_probs[slot] / selected_sum * 3.0f;
        }
    }
}

__global__ void infer_nemotron3_sigmoid_topk_f32_kernel(
    const float* logits,
    const float* bias,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t experts,
    std::uint32_t k,
    std::uint32_t groups,
    std::uint32_t topk_groups,
    bool normalize,
    float scaling_factor) {
    constexpr int kThreads = 256;
    constexpr int kItems = 2;
    const std::uint32_t batch = blockIdx.x;
    logits += static_cast<std::size_t>(batch) * experts;
    out_indices += static_cast<std::size_t>(batch) * k;
    out_weights += static_cast<std::size_t>(batch) * k;
    __shared__ float scores[512];
    __shared__ float probabilities[512];
    __shared__ float group_scores[64];
    __shared__ bool selected_groups[64];
    const std::uint32_t experts_per_group = experts / groups;

    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const std::uint32_t expert = threadIdx.x + item * kThreads;
        if (expert < experts) {
            const float probability = 1.0f / (1.0f + expf(-logits[expert]));
            float score = probability + bias[expert];
            if (isnan(score)) score = -INFINITY;
            scores[expert] = score;
            probabilities[expert] = probability;
        }
    }
    __syncthreads();

    if (threadIdx.x < groups) {
        const std::uint32_t begin = threadIdx.x * experts_per_group;
        float first = -INFINITY;
        float second = -INFINITY;
        for (std::uint32_t expert = begin; expert < begin + experts_per_group; ++expert) {
            const float score = scores[expert];
            if (score > first) {
                second = first;
                first = score;
            } else if (score > second) {
                second = score;
            }
        }
        group_scores[threadIdx.x] = first + second;
        selected_groups[threadIdx.x] = false;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (std::uint32_t slot = 0; slot < topk_groups; ++slot) {
            std::uint32_t selected = UINT32_MAX;
            float best = -INFINITY;
            for (std::uint32_t group = 0; group < groups; ++group) {
                if (!selected_groups[group] && group_scores[group] > best) {
                    best = group_scores[group];
                    selected = group;
                }
            }
            if (selected != UINT32_MAX) selected_groups[selected] = true;
        }
    }
    __syncthreads();

    std::uint64_t keys[kItems];
    float values[kItems];
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const std::uint32_t expert = threadIdx.x + item * kThreads;
        if (expert < experts && selected_groups[expert / experts_per_group]) {
            const std::uint32_t bits = __float_as_uint(scores[expert]);
            const std::uint32_t ordered =
                (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
            keys[item] = (static_cast<std::uint64_t>(ordered) << 32) |
                static_cast<std::uint64_t>(UINT32_MAX - expert);
            values[item] = probabilities[expert];
        } else {
            keys[item] = 0;
            values[item] = 0.0f;
        }
    }
    using BlockSort = cub::BlockRadixSort<std::uint64_t, kThreads, kItems, float>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    BlockSort(sort_storage).SortDescending(keys, values);
    __syncthreads();

    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const std::uint32_t rank = threadIdx.x * kItems + item;
        if (rank < k) {
            out_indices[rank] = UINT32_MAX - static_cast<std::uint32_t>(keys[item]);
            out_weights[rank] = values[item];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float denominator = 1.0f;
        if (normalize) {
            denominator = 1.0e-20f;
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                denominator += out_weights[slot];
            }
        }
        for (std::uint32_t slot = 0; slot < k; ++slot) {
            out_weights[slot] = out_weights[slot] / denominator * scaling_factor;
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

extern "C" cudaError_t infer_step37_sigmoid_top8_f32_on_stream(
    const float* logits,
    const float* bias,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t experts,
    cudaStream_t stream) {
    if (logits == nullptr || bias == nullptr || out_indices == nullptr ||
        out_weights == nullptr || experts < 8) {
        return cudaErrorInvalidValue;
    }
    infer_step37_sigmoid_top8_f32_kernel<<<1, 256, 0, stream>>>(
        logits, bias, out_indices, out_weights, experts);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nemotron3_sigmoid_topk_f32_on_stream(
    const float* logits,
    const float* bias,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t experts,
    std::uint32_t k,
    std::uint32_t groups,
    std::uint32_t topk_groups,
    int normalize,
    float scaling_factor,
    cudaStream_t stream) {
    if (logits == nullptr || bias == nullptr || out_indices == nullptr ||
        out_weights == nullptr || experts == 0 || experts > 512 || k == 0 ||
        k > experts || groups == 0 || groups > 64 || experts % groups != 0 ||
        topk_groups == 0 || topk_groups > groups || !isfinite(scaling_factor)) {
        return cudaErrorInvalidValue;
    }
    infer_nemotron3_sigmoid_topk_f32_kernel<<<1, 256, 0, stream>>>(
        logits, bias, out_indices, out_weights, experts, k, groups, topk_groups,
        normalize != 0, scaling_factor);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nemotron3_sigmoid_topk_f32_batch_on_stream(
    const float* logits,
    const float* bias,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t batch_size,
    std::uint32_t experts,
    std::uint32_t k,
    std::uint32_t groups,
    std::uint32_t topk_groups,
    int normalize,
    float scaling_factor,
    cudaStream_t stream) {
    if (logits == nullptr || bias == nullptr || out_indices == nullptr ||
        out_weights == nullptr || batch_size == 0 || experts == 0 || experts > 512 ||
        k == 0 || k > experts || groups == 0 || groups > 64 || experts % groups != 0 ||
        topk_groups == 0 || topk_groups > groups || !isfinite(scaling_factor)) {
        return cudaErrorInvalidValue;
    }
    infer_nemotron3_sigmoid_topk_f32_kernel<<<batch_size, 256, 0, stream>>>(
        logits, bias, out_indices, out_weights, experts, k, groups, topk_groups,
        normalize != 0, scaling_factor);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_step37_sigmoid_top8_f32_batch_on_stream(
    const float* logits,
    const float* bias,
    std::uint32_t* out_indices,
    float* out_weights,
    std::uint32_t batch_size,
    std::uint32_t experts,
    cudaStream_t stream) {
    if (logits == nullptr || bias == nullptr || out_indices == nullptr ||
        out_weights == nullptr || batch_size == 0 || experts < 8) {
        return cudaErrorInvalidValue;
    }
    infer_step37_sigmoid_top8_f32_kernel<<<batch_size, 256, 0, stream>>>(
        logits, bias, out_indices, out_weights, experts);
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

__global__ void infer_moe_count_routes_kernel(
    const std::uint32_t* __restrict__ indices,
    std::uint32_t* __restrict__ expert_counts,
    std::uint32_t routes,
    std::uint32_t experts) {
    const std::uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) return;
    const std::uint32_t expert = indices[route];
    if (expert < experts) atomicAdd(expert_counts + expert, 1u);
}

__global__ void infer_moe_prefix_route_counts_kernel(
    const std::uint32_t* __restrict__ expert_counts,
    std::uint32_t* __restrict__ expert_offsets,
    std::uint32_t* __restrict__ expert_cursors,
    std::uint32_t experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint32_t offset = 0;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        expert_offsets[expert] = offset;
        expert_cursors[expert] = offset;
        offset += expert_counts[expert];
    }
    expert_offsets[experts] = offset;
}

__global__ void infer_moe_scatter_sorted_routes_kernel(
    const std::uint32_t* __restrict__ indices,
    std::uint32_t* __restrict__ expert_cursors,
    std::uint32_t* __restrict__ sorted_routes,
    std::uint32_t* __restrict__ sorted_experts,
    std::uint32_t* __restrict__ route_to_sorted,
    std::uint32_t routes,
    std::uint32_t experts) {
    const std::uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) return;
    const std::uint32_t expert = indices[route];
    if (expert >= experts) return;
    const std::uint32_t sorted = atomicAdd(expert_cursors + expert, 1u);
    sorted_routes[sorted] = route;
    sorted_experts[sorted] = expert;
    route_to_sorted[route] = sorted;
}

extern "C" cudaError_t infer_moe_sort_routes_on_stream(
    const std::uint32_t* indices,
    std::uint32_t* expert_counts,
    std::uint32_t* expert_offsets,
    std::uint32_t* expert_cursors,
    std::uint32_t* sorted_routes,
    std::uint32_t* sorted_experts,
    std::uint32_t* route_to_sorted,
    std::uint32_t routes,
    std::uint32_t experts,
    cudaStream_t stream) {
    if (indices == nullptr || expert_counts == nullptr || expert_offsets == nullptr ||
        expert_cursors == nullptr || sorted_routes == nullptr || sorted_experts == nullptr ||
        route_to_sorted == nullptr || routes == 0 || experts == 0 || experts > 1024) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemsetAsync(
        expert_counts, 0, static_cast<std::size_t>(experts) * sizeof(std::uint32_t), stream);
    if (status != cudaSuccess) return status;
    constexpr std::uint32_t kThreads = 256;
    infer_moe_count_routes_kernel<<<(routes + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        indices, expert_counts, routes, experts);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_moe_prefix_route_counts_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, expert_cursors, experts);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_moe_scatter_sorted_routes_kernel<<<
        (routes + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        indices, expert_cursors, sorted_routes, sorted_experts, route_to_sorted,
        routes, experts);
    return cudaGetLastError();
}

__global__ void infer_moe_quantize_sorted_routes_nvfp4_kernel(
    const float* __restrict__ input,
    const std::uint32_t* __restrict__ sorted_routes,
    const std::uint32_t* __restrict__ sorted_experts,
    const std::uint32_t* __restrict__ expert_offsets,
    std::uint8_t* __restrict__ packed,
    std::uint8_t* __restrict__ scales,
    std::uint32_t routes,
    std::uint32_t routes_per_row,
    std::uint32_t in_features,
    std::uint32_t scale_stride,
    bool gather_rows) {
    const std::uint32_t k_blocks = (in_features + 15) / 16;
    const std::uint32_t sorted = blockIdx.x;
    if (sorted >= routes) return;
    const std::uint32_t route = sorted_routes[sorted];
    const std::uint32_t source_row = gather_rows ? route / routes_per_row : sorted;
    const std::uint32_t expert = sorted_experts[sorted];
    const std::uint32_t expert_col = sorted - expert_offsets[expert];
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t warps = blockDim.x / 32;
    const float* source = input + static_cast<std::size_t>(source_row) * in_features;

    for (std::uint32_t k_block = warp; k_block < k_blocks; k_block += warps) {
        const std::uint32_t row_start = k_block * 16;
        float max_abs = 0.0f;
        if (row_start + lane < in_features) {
            const float value = source[row_start + lane];
            max_abs = isfinite(value) ? fabsf(value) : 0.0f;
        }
        max_abs = infer_warp_reduce_max(max_abs);
        std::uint32_t scale_word = 0;
        if (lane == 0) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[static_cast<std::size_t>(expert) * scale_stride +
                   infer_ue4m3_tiled_scale_offset(expert_col, k_block, in_features)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(0xffffffffu, scale_word, 0);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        if (lane < 8 && row_start + lane * 2 < in_features) {
            const std::uint32_t row = row_start + lane * 2;
            const float lo_value = scale == 0.0f ? 0.0f : source[row] / scale;
            const std::uint8_t lo = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
            std::uint8_t hi = 0;
            if (row + 1 < in_features) {
                const float hi_value = scale == 0.0f ? 0.0f : source[row + 1] / scale;
                hi = static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
            }
            packed[(static_cast<std::size_t>(sorted) * in_features + row) / 2] =
                lo | (hi << 4);
        }
    }
}

extern "C" cudaError_t infer_moe_quantize_sorted_routes_nvfp4_on_stream(
    const float* input,
    const std::uint32_t* sorted_routes,
    const std::uint32_t* sorted_experts,
    const std::uint32_t* expert_offsets,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t routes,
    std::uint32_t routes_per_row,
    std::uint32_t in_features,
    std::uint32_t scale_stride,
    int gather_rows,
    cudaStream_t stream) {
    if (input == nullptr || sorted_routes == nullptr || sorted_experts == nullptr ||
        expert_offsets == nullptr || packed == nullptr || scales == nullptr || routes == 0 ||
        routes_per_row == 0 || in_features == 0 || (in_features % 16) != 0 ||
        scale_stride == 0) {
        return cudaErrorInvalidValue;
    }
    infer_moe_quantize_sorted_routes_nvfp4_kernel<<<routes, 256, 0, stream>>>(
        input, sorted_routes, sorted_experts, expert_offsets, packed, scales,
        routes, routes_per_row, in_features, scale_stride, gather_rows != 0);
    return cudaGetLastError();
}

__global__ void infer_moe_gelu_tanh_mul_quantize_sorted_routes_nvfp4_kernel(
    const std::uint16_t* __restrict__ gate,
    const std::uint16_t* __restrict__ up,
    const std::uint32_t* __restrict__ sorted_experts,
    const std::uint32_t* __restrict__ expert_offsets,
    std::uint8_t* __restrict__ packed,
    std::uint8_t* __restrict__ scales,
    std::uint32_t routes,
    std::uint32_t in_features,
    std::uint32_t scale_stride) {
    const std::uint32_t sorted = blockIdx.x;
    if (sorted >= routes) return;
    const std::uint32_t expert = sorted_experts[sorted];
    const std::uint32_t expert_col = sorted - expert_offsets[expert];
    const std::uint32_t k_blocks = in_features / 16;
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t warps = blockDim.x / 32;
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const std::uint16_t* gate_row = gate + static_cast<std::size_t>(sorted) * in_features;
    const std::uint16_t* up_row = up + static_cast<std::size_t>(sorted) * in_features;

    const std::uint32_t k_block_pairs = (k_blocks + 1) / 2;
    for (std::uint32_t k_block_pair = warp; k_block_pair < k_block_pairs;
         k_block_pair += warps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t k_block = k_block_pair * 2 + half;
        const std::uint32_t row = k_block_pair * 32 + lane;
        float value = 0.0f;
        if (row < in_features) {
            const float gate_value = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(gate_row + row));
            const float cubic = gate_value * gate_value * gate_value;
            const float gelu = 0.5f * gate_value *
                (1.0f + tanhf(kSqrtTwoOverPi *
                               (gate_value + 0.044715f * cubic)));
            const float up_value = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(up_row + row));
            value = gelu * up_value;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && k_block < k_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[static_cast<std::size_t>(expert) * scale_stride +
                   infer_ue4m3_tiled_scale_offset(expert_col, k_block, in_features)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
        const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
        if (half_lane < 8 && k_block < k_blocks) {
            const std::uint32_t packed_row = k_block * 16 + half_lane * 2;
            const std::uint8_t lo = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(
                    scale == 0.0f ? 0.0f : lo_value / scale,
                    __NV_E2M1, cudaRoundNearest) & 0x0f);
            const std::uint8_t hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(
                    scale == 0.0f ? 0.0f : hi_value / scale,
                    __NV_E2M1, cudaRoundNearest) & 0x0f);
            packed[(static_cast<std::size_t>(sorted) * in_features + packed_row) / 2] =
                lo | (hi << 4);
        }
    }
}

extern "C" cudaError_t infer_moe_gelu_tanh_mul_quantize_sorted_routes_nvfp4_on_stream(
    const std::uint16_t* gate,
    const std::uint16_t* up,
    const std::uint32_t* sorted_experts,
    const std::uint32_t* expert_offsets,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t routes,
    std::uint32_t in_features,
    std::uint32_t scale_stride,
    cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || sorted_experts == nullptr ||
        expert_offsets == nullptr || packed == nullptr || scales == nullptr ||
        routes == 0 || in_features == 0 || (in_features % 16) != 0 ||
        scale_stride == 0) {
        return cudaErrorInvalidValue;
    }
    infer_moe_gelu_tanh_mul_quantize_sorted_routes_nvfp4_kernel<<<
        routes, 256, 0, stream>>>(
        gate, up, sorted_experts, expert_offsets, packed, scales, routes,
        in_features, scale_stride);
    return cudaGetLastError();
}

__global__ void infer_moe_silu_mul_halves_quantize_sorted_routes_nvfp4_kernel(
    const std::uint16_t* __restrict__ gate_up,
    const std::uint32_t* __restrict__ sorted_experts,
    const std::uint32_t* __restrict__ expert_offsets,
    std::uint8_t* __restrict__ packed,
    std::uint8_t* __restrict__ scales,
    std::uint32_t routes,
    std::uint32_t in_features,
    std::uint32_t scale_stride) {
    const std::uint32_t sorted = blockIdx.x;
    if (sorted >= routes) return;
    const std::uint32_t expert = sorted_experts[sorted];
    const std::uint32_t expert_col = sorted - expert_offsets[expert];
    const std::uint32_t k_blocks = in_features / 16;
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t warps = blockDim.x / 32;
    const std::uint16_t* gate_row =
        gate_up + static_cast<std::size_t>(sorted) * in_features * 2;
    const std::uint16_t* up_row = gate_row + in_features;

    const std::uint32_t k_block_pairs = (k_blocks + 1) / 2;
    for (std::uint32_t k_block_pair = warp; k_block_pair < k_block_pairs;
         k_block_pair += warps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t k_block = k_block_pair * 2 + half;
        const std::uint32_t row = k_block_pair * 32 + lane;
        float value = 0.0f;
        if (row < in_features) {
            const float gate_value = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(gate_row + row));
            const float silu = gate_value / (1.0f + expf(-gate_value));
            const float up_value = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(up_row + row));
            value = silu * up_value;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && k_block < k_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[static_cast<std::size_t>(expert) * scale_stride +
                   infer_ue4m3_tiled_scale_offset(expert_col, k_block, in_features)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
        const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
        if (half_lane < 8 && k_block < k_blocks) {
            const std::uint32_t packed_row = k_block * 16 + half_lane * 2;
            const std::uint8_t lo = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(
                    scale == 0.0f ? 0.0f : lo_value / scale,
                    __NV_E2M1, cudaRoundNearest) & 0x0f);
            const std::uint8_t hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(
                    scale == 0.0f ? 0.0f : hi_value / scale,
                    __NV_E2M1, cudaRoundNearest) & 0x0f);
            packed[(static_cast<std::size_t>(sorted) * in_features + packed_row) / 2] =
                lo | (hi << 4);
        }
    }
}

extern "C" cudaError_t infer_moe_silu_mul_halves_quantize_sorted_routes_nvfp4_on_stream(
    const std::uint16_t* gate_up,
    const std::uint32_t* sorted_experts,
    const std::uint32_t* expert_offsets,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t routes,
    std::uint32_t in_features,
    std::uint32_t scale_stride,
    cudaStream_t stream) {
    if (gate_up == nullptr || sorted_experts == nullptr || expert_offsets == nullptr ||
        packed == nullptr || scales == nullptr || routes == 0 || in_features == 0 ||
        (in_features % 16) != 0 || scale_stride == 0) {
        return cudaErrorInvalidValue;
    }
    infer_moe_silu_mul_halves_quantize_sorted_routes_nvfp4_kernel<<<
        routes, 256, 0, stream>>>(
        gate_up, sorted_experts, expert_offsets, packed, scales, routes,
        in_features, scale_stride);
    return cudaGetLastError();
}

__global__ void infer_moe_quantize_rows_nvfp4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    std::uint8_t* __restrict__ packed,
    std::uint8_t* __restrict__ scales,
    std::uint32_t rows,
    std::uint32_t in_features,
    float eps) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const std::uint32_t k_blocks = in_features / 16;
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t warps = blockDim.x / 32;
    const float* source = input + static_cast<std::size_t>(row) * in_features;
    float square_sum = 0.0f;
    if (weight != nullptr) {
        for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
            const float value = source[col];
            square_sum += value * value;
        }
        square_sum = infer_block_reduce_sum(square_sum);
    }
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = weight == nullptr
            ? 1.0f
            : rsqrtf(square_sum / static_cast<float>(in_features) + eps);
    }
    __syncthreads();

    for (std::uint32_t k_block = warp; k_block < k_blocks; k_block += warps) {
        const std::uint32_t row_start = k_block * 16;
        float max_abs = 0.0f;
        if (lane < 16) {
            const float value = weight == nullptr
                ? source[row_start + lane]
                : source[row_start + lane] * inverse_rms * weight[row_start + lane];
            max_abs = isfinite(value) ? fabsf(value) : 0.0f;
        }
        max_abs = infer_warp_reduce_max(max_abs);
        std::uint32_t scale_word = 0;
        if (lane == 0) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[static_cast<std::size_t>(row) * k_blocks + k_block] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(0xffffffffu, scale_word, 0);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        if (lane < 8) {
            const std::uint32_t col = row_start + lane * 2;
            const float lo_source = weight == nullptr
                ? source[col]
                : source[col] * inverse_rms * weight[col];
            const float hi_source = weight == nullptr
                ? source[col + 1]
                : source[col + 1] * inverse_rms * weight[col + 1];
            const float lo_value = scale == 0.0f ? 0.0f : lo_source / scale;
            const float hi_value = scale == 0.0f ? 0.0f : hi_source / scale;
            const std::uint8_t lo = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(lo_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
            const std::uint8_t hi = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp4(hi_value, __NV_E2M1, cudaRoundNearest) & 0x0f);
            packed[(static_cast<std::size_t>(row) * in_features + col) / 2] =
                lo | (hi << 4);
        }
    }
}

__global__ void infer_moe_scatter_quantized_rows_nvfp4_kernel(
    const std::uint8_t* __restrict__ source_packed,
    const std::uint8_t* __restrict__ source_scales,
    const std::uint32_t* __restrict__ sorted_routes,
    const std::uint32_t* __restrict__ sorted_experts,
    const std::uint32_t* __restrict__ expert_offsets,
    std::uint8_t* __restrict__ packed,
    std::uint8_t* __restrict__ scales,
    std::uint32_t routes,
    std::uint32_t routes_per_row,
    std::uint32_t in_features,
    std::uint32_t scale_stride) {
    const std::uint32_t sorted = blockIdx.x;
    if (sorted >= routes) return;
    const std::uint32_t source_row = sorted_routes[sorted] / routes_per_row;
    const std::uint32_t expert = sorted_experts[sorted];
    const std::uint32_t expert_col = sorted - expert_offsets[expert];
    const std::uint32_t packed_bytes = in_features / 2;
    const std::uint32_t k_blocks = in_features / 16;

    for (std::uint32_t col = threadIdx.x; col < packed_bytes; col += blockDim.x) {
        packed[static_cast<std::size_t>(sorted) * packed_bytes + col] =
            source_packed[static_cast<std::size_t>(source_row) * packed_bytes + col];
    }
    for (std::uint32_t k_block = threadIdx.x; k_block < k_blocks; k_block += blockDim.x) {
        scales[static_cast<std::size_t>(expert) * scale_stride +
               infer_ue4m3_tiled_scale_offset(expert_col, k_block, in_features)] =
            source_scales[static_cast<std::size_t>(source_row) * k_blocks + k_block];
    }
}

extern "C" cudaError_t infer_moe_gather_rms_norm_quantize_sorted_routes_nvfp4_on_stream(
    const float* input,
    const float* weight,
    const std::uint32_t* sorted_routes,
    const std::uint32_t* sorted_experts,
    const std::uint32_t* expert_offsets,
    std::uint8_t* source_packed,
    std::uint8_t* source_scales,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t routes,
    std::uint32_t routes_per_row,
    std::uint32_t in_features,
    std::uint32_t scale_stride,
    float eps,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || sorted_routes == nullptr || sorted_experts == nullptr ||
        expert_offsets == nullptr || source_packed == nullptr || source_scales == nullptr ||
        packed == nullptr || scales == nullptr || rows == 0 || routes == 0 ||
        routes_per_row == 0 || routes != rows * routes_per_row || in_features == 0 ||
        (in_features % 16) != 0 || scale_stride == 0 || !isfinite(eps) || eps < 0.0f) {
        return cudaErrorInvalidValue;
    }
    infer_moe_quantize_rows_nvfp4_kernel<<<rows, 256, 0, stream>>>(
        input, weight, source_packed, source_scales, rows, in_features, eps);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_moe_scatter_quantized_rows_nvfp4_kernel<<<routes, 256, 0, stream>>>(
        source_packed, source_scales, sorted_routes, sorted_experts, expert_offsets,
        packed, scales, routes, routes_per_row, in_features, scale_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_moe_gather_quantize_sorted_routes_nvfp4_on_stream(
    const float* input,
    const std::uint32_t* sorted_routes,
    const std::uint32_t* sorted_experts,
    const std::uint32_t* expert_offsets,
    std::uint8_t* source_packed,
    std::uint8_t* source_scales,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t routes,
    std::uint32_t routes_per_row,
    std::uint32_t in_features,
    std::uint32_t scale_stride,
    cudaStream_t stream) {
    if (input == nullptr || sorted_routes == nullptr || sorted_experts == nullptr ||
        expert_offsets == nullptr || source_packed == nullptr || source_scales == nullptr ||
        packed == nullptr || scales == nullptr || rows == 0 || routes == 0 ||
        routes_per_row == 0 || routes != rows * routes_per_row || in_features == 0 ||
        (in_features % 16) != 0 || scale_stride == 0) {
        return cudaErrorInvalidValue;
    }
    infer_moe_quantize_rows_nvfp4_kernel<<<rows, 256, 0, stream>>>(
        input, nullptr, source_packed, source_scales, rows, in_features, 0.0f);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_moe_scatter_quantized_rows_nvfp4_kernel<<<routes, 256, 0, stream>>>(
        source_packed, source_scales, sorted_routes, sorted_experts, expert_offsets,
        packed, scales, routes, routes_per_row, in_features, scale_stride);
    return cudaGetLastError();
}

__global__ void infer_moe_grouped_pointer_tables_kernel(
    const std::uint32_t* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ packed,
    const std::uint8_t* __restrict__ scales,
    std::uint16_t* __restrict__ output,
    const std::uint8_t** __restrict__ packed_table,
    const std::uint8_t** __restrict__ scale_table,
    std::uint16_t** __restrict__ output_table,
    std::uint32_t experts,
    std::uint32_t in_features,
    std::uint32_t out_features,
    std::uint32_t scale_stride) {
    const std::uint32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= experts) return;
    const std::size_t route_offset = expert_offsets[expert];
    packed_table[expert] = packed + route_offset * (in_features / 2);
    scale_table[expert] = scales + static_cast<std::size_t>(expert) * scale_stride;
    output_table[expert] = output + route_offset * out_features;
}

extern "C" cudaError_t infer_moe_grouped_pointer_tables_on_stream(
    const std::uint32_t* expert_offsets,
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    std::uint16_t* output,
    const std::uint8_t** packed_table,
    const std::uint8_t** scale_table,
    std::uint16_t** output_table,
    std::uint32_t experts,
    std::uint32_t in_features,
    std::uint32_t out_features,
    std::uint32_t scale_stride,
    cudaStream_t stream) {
    if (expert_offsets == nullptr || packed == nullptr || scales == nullptr || output == nullptr ||
        packed_table == nullptr || scale_table == nullptr || output_table == nullptr ||
        experts == 0 || in_features == 0 || out_features == 0 || scale_stride == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 128;
    infer_moe_grouped_pointer_tables_kernel<<<
        (experts + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        expert_offsets, packed, scales, output, packed_table, scale_table,
        output_table, experts, in_features, out_features, scale_stride);
    return cudaGetLastError();
}

__global__ void infer_repeat_row_pointer_table_f32_kernel(
    const float* input,
    const float** table,
    std::uint32_t routes,
    std::uint32_t repeats,
    std::uint32_t row_stride) {
    const std::uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) return;
    table[route] = input + static_cast<std::size_t>(route / repeats) * row_stride;
}

extern "C" cudaError_t infer_repeat_row_pointer_table_f32_on_stream(
    const float* input,
    const float** table,
    std::uint32_t routes,
    std::uint32_t repeats,
    std::uint32_t row_stride,
    cudaStream_t stream) {
    if (input == nullptr || table == nullptr || routes == 0 || repeats == 0 ||
        row_stride == 0 || routes % repeats != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 128;
    infer_repeat_row_pointer_table_f32_kernel<<<
        (routes + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        input, table, routes, repeats, row_stride);
    return cudaGetLastError();
}

__global__ void infer_remap_expert_indices_kernel(
    const std::uint32_t* expert_indices,
    const std::uint32_t* expert_to_slot,
    std::uint32_t* slot_indices,
    std::uint32_t count,
    std::uint32_t experts) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t expert = expert_indices[index];
    slot_indices[index] = expert < experts ? expert_to_slot[expert] : UINT32_MAX;
}

extern "C" cudaError_t infer_remap_expert_indices_on_stream(
    const std::uint32_t* expert_indices,
    const std::uint32_t* expert_to_slot,
    std::uint32_t* slot_indices,
    std::uint32_t expert_offset,
    std::uint32_t count,
    std::uint32_t experts,
    cudaStream_t stream) {
    if (expert_indices == nullptr || expert_to_slot == nullptr || slot_indices == nullptr ||
        count == 0 || experts == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 128;
    infer_remap_expert_indices_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        expert_indices + expert_offset, expert_to_slot, slot_indices, count, experts);
    return cudaGetLastError();
}

__global__ void infer_record_expert_indices_u64_kernel(
    const std::uint32_t* expert_indices,
    unsigned long long* counts,
    std::uint32_t count,
    std::uint32_t experts) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t expert = expert_indices[index];
    if (expert < experts) {
        atomicAdd(counts + expert, 1ull);
    }
}

extern "C" cudaError_t infer_record_expert_indices_u64_on_stream(
    const std::uint32_t* expert_indices,
    unsigned long long* counts,
    std::uint32_t count,
    std::uint32_t experts,
    cudaStream_t stream) {
    if (expert_indices == nullptr || counts == nullptr || count == 0 || experts == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 128;
    infer_record_expert_indices_u64_kernel<<<
        (count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        expert_indices, counts, count, experts);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_clear_expert_counts_u64_on_stream(
    unsigned long long* counts,
    std::uint32_t experts,
    cudaStream_t stream) {
    if (counts == nullptr || experts == 0) {
        return cudaErrorInvalidValue;
    }
    return cudaMemsetAsync(
        counts, 0, static_cast<std::size_t>(experts) * sizeof(unsigned long long), stream);
}

__global__ void infer_gather_indexed_mul_f32_kernel(
    const float* values,
    const std::uint32_t* indices,
    const float* multipliers,
    float* output,
    std::uint32_t count,
    std::uint32_t values_len) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t source = indices[index];
    output[index] = source < values_len ? values[source] * multipliers[index] : 0.0f;
}

extern "C" cudaError_t infer_gather_indexed_mul_f32_on_stream(
    const float* values,
    const std::uint32_t* indices,
    const float* multipliers,
    float* output,
    std::uint32_t count,
    std::uint32_t values_len,
    cudaStream_t stream) {
    if (values == nullptr || indices == nullptr || multipliers == nullptr || output == nullptr ||
        count == 0 || values_len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 128;
    infer_gather_indexed_mul_f32_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        values, indices, multipliers, output, count, values_len);
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

__global__ void infer_moe_silu_slots_f32_kernel(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    float* const* output_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups) {
    const std::uint32_t slot = blockIdx.x;
    if (slot >= groups) return;
    const std::uint32_t expert = indices[slot];
    const float alpha = gate_up_alpha_table[expert];
    const float* gate_up = gate_up_table[slot];
    float* output = output_table[slot];
    for (std::uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const float gate = gate_up[row] * alpha;
        const float up = gate_up[rows + row] * alpha;
        output[row] = gate * (1.0f / (1.0f + expf(-gate))) * up;
    }
}

extern "C" cudaError_t infer_moe_silu_slots_f32_on_stream(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    float* const* output_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (indices == nullptr || gate_up_table == nullptr || output_table == nullptr ||
        gate_up_alpha_table == nullptr || rows == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    infer_moe_silu_slots_f32_kernel<<<groups, 256, 0, stream>>>(
        indices, gate_up_table, output_table, gate_up_alpha_table, rows, groups);
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

__global__ void infer_moe_weighted_accumulate_slots_f32_batch_kernel(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* inputs,
    const float* alpha_table,
    float* output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t groups) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * len;
    if (idx >= total) {
        return;
    }
    const std::uint32_t row = idx / len;
    const std::uint32_t col = idx % len;
    const std::uint32_t route_begin = row * groups;
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < groups; ++slot) {
        const std::uint32_t route = route_begin + slot;
        const std::uint32_t expert = indices[route];
        sum += inputs[route][col] * route_weights[route] * alpha_table[expert];
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

extern "C" cudaError_t infer_moe_weighted_accumulate_slots_f32_batch_on_stream(
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* inputs,
    const float* alpha_table,
    float* output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (indices == nullptr || route_weights == nullptr || inputs == nullptr ||
        alpha_table == nullptr || output == nullptr || rows == 0 || len == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t total = rows * len;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_moe_weighted_accumulate_slots_f32_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        indices, route_weights, inputs, alpha_table, output, rows, len, groups);
    return cudaGetLastError();
}

__global__ void infer_moe_weighted_accumulate_sorted_slots_f32_batch_kernel(
    const std::uint32_t* route_to_sorted,
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* sorted_inputs,
    const float* alpha_table,
    float* output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t groups) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * len;
    if (idx >= total) return;
    const std::uint32_t row = idx / len;
    const std::uint32_t col = idx % len;
    const std::uint32_t route_begin = row * groups;
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < groups; ++slot) {
        const std::uint32_t route = route_begin + slot;
        const std::uint32_t sorted = route_to_sorted[route];
        const std::uint32_t expert = indices[route];
        sum += sorted_inputs[sorted][col] * route_weights[route] * alpha_table[expert];
    }
    output[idx] = sum;
}

extern "C" cudaError_t infer_moe_weighted_accumulate_sorted_slots_f32_batch_on_stream(
    const std::uint32_t* route_to_sorted,
    const std::uint32_t* indices,
    const float* route_weights,
    const float* const* sorted_inputs,
    const float* alpha_table,
    float* output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t groups,
    cudaStream_t stream) {
    if (route_to_sorted == nullptr || indices == nullptr || route_weights == nullptr ||
        sorted_inputs == nullptr || alpha_table == nullptr || output == nullptr ||
        rows == 0 || len == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t total = rows * len;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_moe_weighted_accumulate_sorted_slots_f32_batch_kernel<<<
        blocks, kThreads, 0, stream>>>(
        route_to_sorted, indices, route_weights, sorted_inputs, alpha_table,
        output, rows, len, groups);
    return cudaGetLastError();
}

__global__ void infer_moe_weighted_accumulate_sorted_bf16_batch_kernel(
    const std::uint32_t* route_to_sorted,
    const float* route_weights,
    const std::uint16_t* sorted_inputs,
    float* output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t routes_per_row) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * len;
    if (idx >= total) return;
    const std::uint32_t row = idx / len;
    const std::uint32_t col = idx % len;
    const std::uint32_t route_begin = row * routes_per_row;
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < routes_per_row; ++slot) {
        const std::uint32_t route = route_begin + slot;
        const std::uint32_t sorted = route_to_sorted[route];
        const auto value = *reinterpret_cast<const __nv_bfloat16*>(
            sorted_inputs + static_cast<std::size_t>(sorted) * len + col);
        sum += __bfloat162float(value) * route_weights[route];
    }
    output[idx] = sum;
}

extern "C" cudaError_t infer_moe_weighted_accumulate_sorted_bf16_batch_on_stream(
    const std::uint32_t* route_to_sorted,
    const float* route_weights,
    const std::uint16_t* sorted_inputs,
    float* output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t routes_per_row,
    cudaStream_t stream) {
    if (route_to_sorted == nullptr || route_weights == nullptr || sorted_inputs == nullptr ||
        output == nullptr || rows == 0 || len == 0 || routes_per_row == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t total = rows * len;
    infer_moe_weighted_accumulate_sorted_bf16_batch_kernel<<<
        (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        route_to_sorted, route_weights, sorted_inputs, output, rows, len,
        routes_per_row);
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

__global__ void infer_qwen36_ffn_finalize_batch_f32_kernel(
    const float* routed_output,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * cols;
    if (idx >= len) return;
    const std::uint32_t row = idx / cols;
    const float shared_scale = 1.0f / (1.0f + expf(-shared_gate_logit[row]));
    output[idx] = infer_round_f32_to_bf16_value(
        residual[idx] + routed_output[idx] + shared_output[idx] * shared_scale);
}

extern "C" cudaError_t infer_qwen36_ffn_finalize_batch_f32_on_stream(
    const float* routed_output,
    const float* shared_gate_logit,
    const float* shared_output,
    const float* residual,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (routed_output == nullptr || shared_gate_logit == nullptr || shared_output == nullptr ||
        residual == nullptr || output == nullptr || rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t len = static_cast<std::uint64_t>(rows) * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_qwen36_ffn_finalize_batch_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        routed_output, shared_gate_logit, shared_output, residual, output, rows, cols);
    return cudaGetLastError();
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

__global__ void infer_rope_neox_partial_f32_indexed_kernel(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    const std::uint32_t* position,
    float theta) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = rows * head_dim;
    if (idx >= len) return;
    const std::uint32_t dim = idx % head_dim;
    if (dim >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }
    const std::uint32_t half = rotary_dim / 2;
    if (dim >= half) return;
    const std::uint32_t row_start = (idx / head_dim) * head_dim;
    const float inv_freq =
        powf(theta, -2.0f * static_cast<float>(dim) / static_cast<float>(rotary_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(position[0]) * inv_freq, &sin_value, &cos_value);
    const float a = input[row_start + dim];
    const float b = input[row_start + dim + half];
    output[row_start + dim] = a * cos_value - b * sin_value;
    output[row_start + dim + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_neox_partial_f32_indexed_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    const std::uint32_t* position,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || position == nullptr || rows == 0 ||
        head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim % 2) != 0 || !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = rows * head_dim;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_rope_neox_partial_f32_indexed_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, rotary_dim, position, theta);
    return cudaGetLastError();
}

// Proportional partial RoPE keeps the ordinary NeoX pair stride and frequency
// denominator, but rotates only the leading fraction of frequency pairs. This
// is the layout used by Gemma 4 full attention: pair i is (i, i+head_dim/2),
// and pairs after rotary_pairs pass through unchanged.
__global__ void infer_rope_neox_proportional_f32_kernel(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_pairs,
    std::uint32_t position,
    float theta) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t total_pairs = rows * half;
    if (idx >= total_pairs) {
        return;
    }

    const std::uint32_t row = idx / half;
    const std::uint32_t pair = idx % half;
    const std::uint32_t row_start = row * head_dim;
    const float a = input[row_start + pair];
    const float b = input[row_start + pair + half];
    if (pair >= rotary_pairs) {
        output[row_start + pair] = a;
        output[row_start + pair + half] = b;
        return;
    }

    const float inv_freq =
        powf(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(position) * inv_freq, &sin_value, &cos_value);
    output[row_start + pair] = a * cos_value - b * sin_value;
    output[row_start + pair + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_neox_proportional_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t head_dim,
    std::uint32_t rotary_pairs,
    std::uint32_t position,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || head_dim == 0 ||
        (head_dim % 2) != 0 || rotary_pairs == 0 || rotary_pairs > head_dim / 2 ||
        !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t total_pairs = rows * (head_dim / 2);
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_proportional_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, head_dim, rotary_pairs, position, theta);
    return cudaGetLastError();
}

__global__ void infer_rope_neox_proportional_sequence_f32_kernel(
    const float* input,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_pairs,
    std::uint32_t input_token_offset,
    std::uint32_t start_position,
    float theta) {
    const std::uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t total_pairs = tokens * heads * half;
    if (pair_idx >= total_pairs) return;

    const std::uint32_t pair = pair_idx % half;
    const std::uint32_t row = pair_idx / half;
    const std::uint32_t token = row / heads;
    const std::uint32_t dense_row = input_token_offset * heads + row;
    const std::uint32_t row_start = dense_row * head_dim;
    const float a = input[row_start + pair];
    const float b = input[row_start + pair + half];
    if (pair >= rotary_pairs) {
        output[row_start + pair] = a;
        output[row_start + pair + half] = b;
        return;
    }

    const float inv_freq =
        powf(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(start_position + token) * inv_freq, &sin_value, &cos_value);
    output[row_start + pair] = a * cos_value - b * sin_value;
    output[row_start + pair + half] = a * sin_value + b * cos_value;
}

extern "C" cudaError_t infer_rope_neox_proportional_sequence_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_pairs,
    std::uint32_t input_token_offset,
    std::uint32_t start_position,
    float theta,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || tokens == 0 || heads == 0 ||
        head_dim == 0 || (head_dim % 2) != 0 || rotary_pairs == 0 ||
        rotary_pairs > head_dim / 2 || !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t total_pairs = tokens * heads * (head_dim / 2);
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_proportional_sequence_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, tokens, heads, head_dim, rotary_pairs,
        input_token_offset, start_position, theta);
    return cudaGetLastError();
}

__global__ void infer_dual_rms_norm_rope_neox_proportional_sequence_f32_kernel(
    const float* q_input,
    const float* q_weight,
    float* q_output,
    const float* k_input,
    const float* k_weight,
    float* k_output,
    std::uint32_t tokens,
    std::uint32_t q_heads,
    std::uint32_t k_heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_pairs,
    std::uint32_t input_token_offset,
    std::uint32_t start_position,
    float theta,
    float q_eps,
    float k_eps) {
    extern __shared__ float partial[];
    const std::uint32_t q_rows = tokens * q_heads;
    const bool is_q = blockIdx.x < q_rows;
    const std::uint32_t local_row = is_q ? blockIdx.x : blockIdx.x - q_rows;
    const std::uint32_t heads = is_q ? q_heads : k_heads;
    const std::uint32_t token = local_row / heads;
    const std::uint32_t head = local_row % heads;
    const std::uint32_t dense_row = (input_token_offset + token) * heads + head;
    const std::size_t row_start = static_cast<std::size_t>(dense_row) * head_dim;
    const float* input = is_q ? q_input : k_input;
    const float* weight = is_q ? q_weight : k_weight;
    float* output = is_q ? q_output : k_output;

    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
        const float value = input[row_start + col];
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
    const float eps = is_q ? q_eps : k_eps;
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    const std::uint32_t half = head_dim / 2;
    for (std::uint32_t pair = threadIdx.x; pair < half; pair += blockDim.x) {
        const float a = input[row_start + pair] * inv_rms * weight[pair];
        const float b = input[row_start + pair + half] * inv_rms * weight[pair + half];
        if (pair >= rotary_pairs) {
            output[row_start + pair] = a;
            output[row_start + pair + half] = b;
            continue;
        }
        const float inv_freq =
            powf(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
        float sin_value;
        float cos_value;
        sincosf(static_cast<float>(start_position + token) * inv_freq, &sin_value, &cos_value);
        output[row_start + pair] = a * cos_value - b * sin_value;
        output[row_start + pair + half] = a * sin_value + b * cos_value;
    }
}

extern "C" cudaError_t infer_dual_rms_norm_rope_neox_proportional_sequence_f32_on_stream(
    const float* q_input,
    const float* q_weight,
    float* q_output,
    const float* k_input,
    const float* k_weight,
    float* k_output,
    std::uint32_t tokens,
    std::uint32_t q_heads,
    std::uint32_t k_heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_pairs,
    std::uint32_t input_token_offset,
    std::uint32_t start_position,
    float theta,
    float q_eps,
    float k_eps,
    cudaStream_t stream) {
    if (q_input == nullptr || q_weight == nullptr || q_output == nullptr ||
        k_input == nullptr || k_weight == nullptr || k_output == nullptr ||
        tokens == 0 || q_heads == 0 || k_heads == 0 || head_dim == 0 ||
        (head_dim % 2) != 0 || rotary_pairs == 0 || rotary_pairs > head_dim / 2 ||
        !isfinite(theta) || theta <= 0.0f || !isfinite(q_eps) || q_eps < 0.0f ||
        !isfinite(k_eps) || k_eps < 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t blocks = tokens * (q_heads + k_heads);
    infer_dual_rms_norm_rope_neox_proportional_sequence_f32_kernel<<<
        blocks, kThreads, kThreads * sizeof(float), stream>>>(
        q_input, q_weight, q_output, k_input, k_weight, k_output,
        tokens, q_heads, k_heads, head_dim, rotary_pairs,
        input_token_offset, start_position, theta, q_eps, k_eps);
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

__global__ void infer_rope_neox_inv_freq_sequence_f32_kernel(
    const float* input,
    const float* inv_freq,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t input_token_offset,
    std::uint32_t start_position,
    float attention_scale) {
    const std::uint32_t half = rotary_dim / 2;
    const std::uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total_pairs = tokens * heads * half;
    if (pair_idx >= total_pairs) return;
    const std::uint32_t i = pair_idx % half;
    const std::uint32_t row = pair_idx / half;
    const std::uint32_t token = row / heads;
    const std::uint32_t row_start = (input_token_offset * heads + row) * head_dim;
    float sin_value;
    float cos_value;
    sincosf(static_cast<float>(start_position + token) * inv_freq[i], &sin_value, &cos_value);
    const float a = input[row_start + i];
    const float b = input[row_start + i + half];
    output[row_start + i] = (a * cos_value - b * sin_value) * attention_scale;
    output[row_start + i + half] = (a * sin_value + b * cos_value) * attention_scale;
}

extern "C" cudaError_t infer_rope_neox_inv_freq_sequence_f32_on_stream(
    const float* input,
    const float* inv_freq,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t input_token_offset,
    std::uint32_t start_position,
    float attention_scale,
    cudaStream_t stream) {
    if (input == nullptr || inv_freq == nullptr || output == nullptr || tokens == 0 ||
        heads == 0 || head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim % 2) != 0 || !isfinite(attention_scale) || attention_scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const std::size_t value_offset =
        static_cast<std::size_t>(input_token_offset) * heads * head_dim;
    const std::size_t bytes = static_cast<std::size_t>(tokens) * heads * head_dim * sizeof(float);
    cudaError_t status = cudaMemcpyAsync(
        output + value_offset, input + value_offset, bytes, cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) return status;
    constexpr int kThreads = 256;
    const std::uint32_t total_pairs = tokens * heads * (rotary_dim / 2);
    const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
    infer_rope_neox_inv_freq_sequence_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, inv_freq, output, tokens, heads, head_dim, rotary_dim,
        input_token_offset, start_position, attention_scale);
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

__global__ void infer_concat_f32_rows_kernel(const float* left,
                                             const float* right,
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
    const std::uint32_t output_offset = row * (2 * cols) + col;
    output[output_offset] = left[idx];
    output[output_offset + cols] = right[idx];
}

extern "C" cudaError_t infer_concat_f32_rows_on_stream(const float* left,
                                                        const float* right,
                                                        float* output,
                                                        std::uint32_t rows,
                                                        std::uint32_t cols,
                                                        cudaStream_t stream) {
    if (left == nullptr || right == nullptr || output == nullptr || rows == 0 || cols == 0 ||
        cols > UINT32_MAX / 2 || rows > UINT32_MAX / cols) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t len = rows * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_concat_f32_rows_kernel<<<blocks, kThreads, 0, stream>>>(
        left, right, output, rows, cols);
    return cudaGetLastError();
}

__global__ void infer_copy_f32_rows_into_columns_kernel(const float* input,
                                                         float* output,
                                                         std::uint32_t rows,
                                                         std::uint32_t input_cols,
                                                         std::uint32_t output_cols,
                                                         std::uint32_t output_col_offset) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = rows * input_cols;
    if (index >= count) return;
    const std::uint32_t row = index / input_cols;
    const std::uint32_t col = index % input_cols;
    output[row * output_cols + output_col_offset + col] = input[index];
}

extern "C" cudaError_t infer_copy_f32_rows_into_columns_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    std::uint32_t output_col_offset,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || input_cols == 0 ||
        output_cols == 0 || output_col_offset > output_cols ||
        input_cols > output_cols - output_col_offset) {
        return cudaErrorInvalidValue;
    }
    const std::uint64_t count = static_cast<std::uint64_t>(rows) * input_cols;
    if (count > UINT32_MAX) return cudaErrorInvalidValue;
    constexpr std::uint32_t threads = 256;
    infer_copy_f32_rows_into_columns_kernel<<<
        (static_cast<std::uint32_t>(count) + threads - 1) / threads, threads, 0, stream>>>(
        input, output, rows, input_cols, output_cols, output_col_offset);
    return cudaGetLastError();
}

__global__ void infer_increment_u32_kernel(std::uint32_t* values,
                                           std::uint32_t len,
                                           std::uint32_t increment) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        values[idx] += increment;
    }
}

extern "C" cudaError_t infer_increment_u32_on_stream(std::uint32_t* values,
                                                      std::uint32_t len,
                                                      std::uint32_t increment,
                                                      cudaStream_t stream) {
    if (values == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_increment_u32_kernel<<<blocks, kThreads, 0, stream>>>(values, len, increment);
    return cudaGetLastError();
}

__global__ void infer_store_u32_column_kernel(
    const std::uint32_t* input,
    std::uint32_t* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t column) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        output[static_cast<std::size_t>(row) * columns + column] = input[row];
    }
}

extern "C" cudaError_t infer_store_u32_column_on_stream(
    const std::uint32_t* input,
    std::uint32_t* output,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t column,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || columns == 0 ||
        column >= columns) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((rows + kThreads - 1) / kThreads);
    infer_store_u32_column_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, columns, column);
    return cudaGetLastError();
}

__global__ void infer_prepend_u32_rows_kernel(
    const std::uint32_t* first,
    const std::uint32_t* remaining,
    std::uint32_t* output,
    std::uint32_t rows,
    std::uint32_t remaining_columns) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t output_columns = remaining_columns + 1;
    const std::uint32_t len = rows * output_columns;
    if (linear >= len) {
        return;
    }
    const std::uint32_t row = linear / output_columns;
    const std::uint32_t column = linear % output_columns;
    output[linear] = column == 0
        ? first[row]
        : remaining[static_cast<std::size_t>(row) * remaining_columns + column - 1];
}

extern "C" cudaError_t infer_prepend_u32_rows_on_stream(
    const std::uint32_t* first,
    const std::uint32_t* remaining,
    std::uint32_t* output,
    std::uint32_t rows,
    std::uint32_t remaining_columns,
    cudaStream_t stream) {
    if (first == nullptr || remaining == nullptr || output == nullptr ||
        rows == 0 || remaining_columns == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = rows * (remaining_columns + 1);
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_prepend_u32_rows_kernel<<<blocks, kThreads, 0, stream>>>(
        first, remaining, output, rows, remaining_columns);
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

__global__ void infer_gather_group_row_f32_kernel(
    const float* input,
    float* output,
    std::uint32_t groups,
    std::uint32_t rows_per_group,
    std::uint32_t row,
    std::uint32_t cols) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = groups * cols;
    if (linear >= total) {
        return;
    }
    const std::uint32_t group = linear / cols;
    const std::uint32_t col = linear % cols;
    output[linear] = input[
        (static_cast<std::size_t>(group) * rows_per_group + row) * cols + col];
}

extern "C" cudaError_t infer_gather_group_row_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t groups,
    std::uint32_t rows_per_group,
    std::uint32_t row,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || groups == 0 ||
        rows_per_group == 0 || row >= rows_per_group || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t total = groups * cols;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_gather_group_row_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, groups, rows_per_group, row, cols);
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

__global__ void infer_copy_bf16_row_to_f32_kernel(
    const std::uint16_t* input,
    std::uint32_t row,
    float* output,
    std::uint32_t cols) {
    const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const std::uint16_t raw = input[static_cast<std::size_t>(row) * cols + col];
    output[col] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&raw));
}

extern "C" cudaError_t infer_copy_bf16_row_to_f32_on_stream(
    const std::uint16_t* input,
    std::uint32_t row,
    float* output,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || cols == 0) return cudaErrorInvalidValue;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((cols + kThreads - 1) / kThreads);
    infer_copy_bf16_row_to_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, row, output, cols);
    return cudaGetLastError();
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

__global__ void infer_copy_fp8_rows_to_f32_indexed_kernel(
    const std::uint8_t* input,
    const float* row_scales,
    const std::uint32_t* rows,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = batch_size * cols;
    if (idx >= len) return;
    const std::uint32_t batch = idx / cols;
    const std::uint32_t col = idx % cols;
    const std::uint32_t row = rows[batch];
    output[idx] = infer_e4m3_value(input[static_cast<std::size_t>(row) * cols + col]) *
                  row_scales[row];
}

extern "C" cudaError_t infer_copy_fp8_rows_to_f32_indexed_on_stream(
    const std::uint8_t* input,
    const float* row_scales,
    const std::uint32_t* rows,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || row_scales == nullptr || rows == nullptr || output == nullptr ||
        batch_size == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t len = batch_size * cols;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_copy_fp8_rows_to_f32_indexed_kernel<<<blocks, kThreads, 0, stream>>>(
        input, row_scales, rows, output, batch_size, cols);
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

__global__ void infer_dflash2_capture_f32_kernel(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t hidden,
    std::uint32_t taps,
    std::uint32_t tap) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t values = rows * hidden;
    if (index >= values) return;
    const std::uint32_t row = index / hidden;
    const std::uint32_t col = index - row * hidden;
    output[(static_cast<std::size_t>(row) * taps + tap) * hidden + col] = input[index];
}

extern "C" cudaError_t infer_dflash2_capture_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t hidden,
    std::uint32_t taps,
    std::uint32_t tap,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows == 0 || hidden == 0 ||
        taps == 0 || tap >= taps) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t values = rows * hidden;
    const std::uint32_t blocks = (values + kThreads - 1) / kThreads;
    infer_dflash2_capture_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, rows, hidden, taps, tap);
    return cudaGetLastError();
}

__global__ void infer_dflash2_grouped_conv_f32_kernel(
    const float* input,
    const float* coefficients,
    const float* base,
    float* output,
    std::uint32_t rows,
    std::uint32_t hidden,
    std::uint32_t groups,
    std::uint32_t taps,
    std::uint32_t block_size,
    std::uint32_t side) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t values = rows * hidden;
    if (index >= values) return;
    const std::uint32_t row = index / hidden;
    const std::uint32_t channel = index - row * hidden;
    const std::uint32_t group_size = hidden / groups;
    const std::uint32_t group = channel / group_size;
    const std::uint32_t available = min(taps, row % block_size + 1);
    float value = 0.0f;
    for (std::uint32_t tap = 0; tap < available; ++tap) {
        const std::size_t base_index =
            (static_cast<std::size_t>(side) * taps + tap) * hidden + channel;
        const std::size_t coefficient_index =
            static_cast<std::size_t>(row) * 2 * taps * groups +
            (side * taps + tap) * groups + group;
        value = __fmaf_rn(
            base[base_index] + coefficients[coefficient_index],
            input[static_cast<std::size_t>(row - tap) * hidden + channel],
            value);
    }
    output[index] = value;
}

extern "C" cudaError_t infer_dflash2_grouped_conv_f32_on_stream(
    const float* input,
    const float* coefficients,
    const float* base,
    float* output,
    std::uint32_t rows,
    std::uint32_t hidden,
    std::uint32_t groups,
    std::uint32_t taps,
    std::uint32_t block_size,
    std::uint32_t side,
    cudaStream_t stream) {
    if (input == nullptr || coefficients == nullptr || base == nullptr || output == nullptr ||
        rows == 0 || hidden == 0 || groups == 0 || (hidden % groups) != 0 || taps == 0 ||
        block_size == 0 || side >= 2) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t values = rows * hidden;
    const std::uint32_t blocks = (values + kThreads - 1) / kThreads;
    infer_dflash2_grouped_conv_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, coefficients, base, output, rows, hidden, groups, taps, block_size, side);
    return cudaGetLastError();
}

__global__ void infer_dflash2_noncausal_attention_f32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ context_key,
    const float* __restrict__ context_value,
    const float* __restrict__ block_key,
    const float* __restrict__ block_value,
    float* __restrict__ output,
    std::uint32_t context_end,
    std::uint32_t context_len,
    std::uint32_t rows,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t window) {
    extern __shared__ float shmem[];
    float* q_sh = shmem;
    float* scores = shmem + head_dim;
    float* reduction = scores + blockDim.x;
    const std::uint32_t query_row = blockIdx.y;
    const std::uint32_t q_head = blockIdx.x;
    if (query_row >= rows || q_head >= q_heads) return;

    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q = query +
        (static_cast<std::size_t>(query_row) * q_heads + q_head) * head_dim;
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        q_sh[dim] = q[dim];
    }
    __syncthreads();

    const std::uint32_t sequence_end = context_end + rows;
    const std::uint32_t first_key = sequence_end > window ? sequence_end - window : 0;
    const std::uint32_t retained_start = context_end - context_len;
    const std::uint32_t context_start = max(retained_start, first_key);
    const std::uint32_t context_rows = context_end - context_start;
    const std::uint32_t key_count = context_rows + rows;
    const std::uint32_t tid = threadIdx.x;
    float running_max = -INFINITY;
    float running_total = 0.0f;
    float accumulator = 0.0f;

    for (std::uint32_t tile_start = 0; tile_start < key_count; tile_start += blockDim.x) {
        const std::uint32_t key_index = tile_start + tid;
        float score = -INFINITY;
        if (key_index < key_count) {
            const bool from_context = key_index < context_rows;
            const std::uint32_t logical_position = context_start + key_index;
            const std::uint32_t row = from_context
                ? logical_position % window
                : key_index - context_rows;
            const float* key_base = from_context ? context_key : block_key;
            const float* key = key_base +
                (static_cast<std::size_t>(row) * kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                dot = __fmaf_rn(q_sh[dim], key[dim], dot);
            }
            score = dot * scale;
        }
        scores[tid] = score;
        reduction[tid] = score;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            __syncthreads();
        }
        const float tile_max = reduction[0];
        const float weight = key_index < key_count ? expf(score - tile_max) : 0.0f;
        scores[tid] = weight;
        reduction[tid] = weight;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float merged_max = fmaxf(running_max, tile_max);
            reduction[1] = isfinite(running_max) ? expf(running_max - merged_max) : 0.0f;
            reduction[2] = expf(tile_max - merged_max);
            reduction[3] = running_total * reduction[1] + reduction[0] * reduction[2];
            reduction[4] = merged_max;
        }
        __syncthreads();
        if (tid < head_dim) {
            float tile_accumulator = 0.0f;
            const std::uint32_t tile_rows = min(blockDim.x, key_count - tile_start);
            for (std::uint32_t index = 0; index < tile_rows; ++index) {
                const std::uint32_t absolute_index = tile_start + index;
                const bool from_context = absolute_index < context_rows;
                const std::uint32_t logical_position = context_start + absolute_index;
                const std::uint32_t row = from_context
                    ? logical_position % window
                    : absolute_index - context_rows;
                const float* value_base = from_context ? context_value : block_value;
                const float* value = value_base +
                    (static_cast<std::size_t>(row) * kv_heads + kv_head) * head_dim;
                tile_accumulator = __fmaf_rn(scores[index], value[tid], tile_accumulator);
            }
            accumulator = accumulator * reduction[1] + tile_accumulator * reduction[2];
        }
        if (tid == 0) {
            running_total = reduction[3];
            running_max = reduction[4];
        }
        __syncthreads();
    }
    if (tid == 0) reduction[0] = running_total;
    __syncthreads();
    if (tid < head_dim) {
        output[(static_cast<std::size_t>(query_row) * q_heads + q_head) * head_dim + tid] =
            accumulator / reduction[0];
    }
}

extern "C" cudaError_t infer_dflash2_noncausal_attention_f32_on_stream(
    const float* query,
    const float* context_key,
    const float* context_value,
    const float* block_key,
    const float* block_value,
    float* output,
    std::uint32_t context_end,
    std::uint32_t context_len,
    std::uint32_t rows,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t window,
    cudaStream_t stream) {
    constexpr std::uint32_t kThreads = 256;
    if (query == nullptr || context_key == nullptr || context_value == nullptr ||
        block_key == nullptr || block_value == nullptr || output == nullptr || rows == 0 ||
        q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0 || head_dim == 0 ||
        head_dim > kThreads || window == 0 || rows > window || context_len > window ||
        context_len > context_end) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(q_heads, rows);
    const std::size_t shared = (head_dim + 2 * kThreads) * sizeof(float);
    infer_dflash2_noncausal_attention_f32_kernel<<<grid, kThreads, shared, stream>>>(
        query, context_key, context_value, block_key, block_value, output,
        context_end, context_len, rows, q_heads, kv_heads, head_dim, window);
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
    extern __shared__ float shmem[];
    float* q_sh = shmem;
    float* partial = shmem + head_dim;
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
    const float* q = query + static_cast<std::size_t>(token) * hidden + q_head * head_dim;
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        q_sh[dim] = q[dim];
    }
    __syncthreads();

    float maximum = -INFINITY;
    float sum = 0.0f;
    float accum = 0.0f;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t warp = tid >> 5u;
    const std::uint32_t warps = blockDim.x >> 5u;
    for (std::uint32_t row = 0; row < cache_len; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kv_width + kv_head * head_dim;
        const float* k = key_cache + base;
        const float* v = value_cache + base;
        float dot = tid < head_dim ? q_sh[tid] * k[tid] : 0.0f;
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        if (lane == 0) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (warp == 0 && lane == 0) {
            dot = 0.0f;
            for (std::uint32_t index = 0; index < warps; ++index) {
                dot += partial[index];
            }
            const float score = dot * scale;
            const float rescale = score > maximum ? expf(maximum - score) : 1.0f;
            const float weight = score > maximum ? 1.0f : expf(score - maximum);
            maximum = fmaxf(maximum, score);
            sum = sum * rescale + weight;
            partial[0] = rescale;
            partial[1] = weight;
        }
        __syncthreads();
        if (tid < head_dim) {
            accum = __fmaf_rn(partial[1], v[tid], accum * partial[0]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[0] = sum;
    }
    __syncthreads();
    const float inv_total = 1.0f / partial[0];
    if (tid < head_dim) {
        output[static_cast<std::size_t>(token) * hidden + q_head * head_dim + tid] =
            accum * inv_total;
    }
}

__device__ __forceinline__ bool infer_ragged_row_location(
    std::uint32_t row,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t sequence_count,
    std::uint32_t* sequence,
    std::uint32_t* local_row) {
    for (std::uint32_t candidate = 0; candidate < sequence_count; ++candidate) {
        const std::uint32_t begin = sequence_offsets[candidate];
        const std::uint32_t length = sequence_lengths[candidate];
        if (row >= begin && row - begin < length) {
            *sequence = candidate;
            *local_row = row - begin;
            return true;
        }
    }
    return false;
}

__global__ void infer_append_ragged_kv_f32_kernel(
    const float* key,
    const float* value,
    float* const* key_cache_table,
    float* const* value_cache_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t width) {
    const std::uint32_t row = blockIdx.x;
    if (row >= total_tokens) return;
    __shared__ std::uint32_t sequence;
    __shared__ std::uint32_t local_row;
    __shared__ bool valid;
    if (threadIdx.x == 0) {
        valid = infer_ragged_row_location(
            row, sequence_offsets, sequence_lengths, sequence_count, &sequence, &local_row);
    }
    __syncthreads();
    if (!valid) return;
    const std::size_t source = static_cast<std::size_t>(row) * width;
    const std::size_t destination =
        static_cast<std::size_t>(start_positions[sequence] + local_row) * width;
    for (std::uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        key_cache_table[sequence][destination + column] = key[source + column];
        value_cache_table[sequence][destination + column] = value[source + column];
    }
}

extern "C" cudaError_t infer_append_ragged_kv_f32_on_stream(
    const float* key,
    const float* value,
    float* const* key_cache_table,
    float* const* value_cache_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t width,
    cudaStream_t stream) {
    if (key == nullptr || value == nullptr || key_cache_table == nullptr ||
        value_cache_table == nullptr || sequence_offsets == nullptr ||
        sequence_lengths == nullptr || start_positions == nullptr ||
        sequence_count == 0 || total_tokens == 0 || width == 0) {
        return cudaErrorInvalidValue;
    }
    infer_append_ragged_kv_f32_kernel<<<total_tokens, 256, 0, stream>>>(
        key, value, key_cache_table, value_cache_table, sequence_offsets,
        sequence_lengths, start_positions, sequence_count, total_tokens, width);
    return cudaGetLastError();
}

__global__ void infer_ragged_gqa_attention_f32_kernel(
    const float* query,
    float* const* key_cache_table,
    float* const* value_cache_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    float* output,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    extern __shared__ float shmem[];
    float* q_sh = shmem;
    float* partial = shmem + head_dim;
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t q_head = blockIdx.y;
    if (row >= total_tokens || q_head >= q_heads) return;
    __shared__ std::uint32_t sequence;
    __shared__ std::uint32_t local_row;
    __shared__ bool valid;
    if (threadIdx.x == 0) {
        valid = infer_ragged_row_location(
            row, sequence_offsets, sequence_lengths, sequence_count, &sequence, &local_row);
    }
    __syncthreads();
    if (!valid) return;

    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const std::uint32_t hidden = q_heads * head_dim;
    const std::uint32_t cache_len = start_positions[sequence] + local_row + 1;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q = query + static_cast<std::size_t>(row) * hidden + q_head * head_dim;
    const float* key_cache = key_cache_table[sequence];
    const float* value_cache = value_cache_table[sequence];
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        q_sh[dim] = q[dim];
    }
    __syncthreads();

    float maximum = -INFINITY;
    float total_weight = 0.0f;
    float accum = 0.0f;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t warps = blockDim.x >> 5u;
    for (std::uint32_t cache_row = 0; cache_row < cache_len; ++cache_row) {
        const float* k = key_cache +
            static_cast<std::size_t>(cache_row) * kv_width + kv_head * head_dim;
        const float* v = value_cache +
            static_cast<std::size_t>(cache_row) * kv_width + kv_head * head_dim;
        float dot = threadIdx.x < head_dim ? q_sh[threadIdx.x] * k[threadIdx.x] : 0.0f;
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        if (lane == 0) {
            partial[warp] = dot;
        }
        __syncthreads();
        if (warp == 0 && lane == 0) {
            dot = 0.0f;
            for (std::uint32_t index = 0; index < warps; ++index) {
                dot += partial[index];
            }
            const float score = dot * scale;
            const float rescale = score > maximum ? expf(maximum - score) : 1.0f;
            const float weight = score > maximum ? 1.0f : expf(score - maximum);
            maximum = fmaxf(maximum, score);
            total_weight = total_weight * rescale + weight;
            partial[0] = rescale;
            partial[1] = weight;
        }
        __syncthreads();
        if (threadIdx.x < head_dim) {
            accum *= partial[0];
            accum = __fmaf_rn(partial[1], v[threadIdx.x], accum);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0] = total_weight;
    }
    __syncthreads();
    if (threadIdx.x < head_dim) {
        output[static_cast<std::size_t>(row) * hidden + q_head * head_dim + threadIdx.x] =
            accum / partial[0];
    }
}

extern "C" cudaError_t infer_ragged_gqa_attention_f32_on_stream(
    const float* query,
    float* const* key_cache_table,
    float* const* value_cache_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    float* output,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache_table == nullptr || value_cache_table == nullptr ||
        sequence_offsets == nullptr || sequence_lengths == nullptr ||
        start_positions == nullptr || output == nullptr || sequence_count == 0 ||
        total_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        head_dim > kThreads || q_heads % kv_heads != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 blocks(total_tokens, q_heads);
    infer_ragged_gqa_attention_f32_kernel<<<
        blocks, kThreads, (head_dim + kThreads) * sizeof(float), stream>>>(
        query, key_cache_table, value_cache_table, sequence_offsets, sequence_lengths,
        start_positions, output, sequence_count, total_tokens, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

__device__ __forceinline__ std::uint32_t infer_paged_f32_slot(
    const std::uint32_t* page_table,
    std::uint32_t logical_row,
    std::uint32_t page_tokens) {
    return page_table[logical_row / page_tokens];
}

__global__ void infer_append_ragged_paged_kv_f32_kernel(
    const float* key,
    const float* value,
    float* key_pool,
    float* value_pool,
    const std::uint32_t* const* page_tables,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t page_tokens,
    std::uint32_t width) {
    const std::uint32_t row = blockIdx.x;
    if (row >= total_tokens) return;
    __shared__ std::uint32_t sequence;
    __shared__ std::uint32_t local_row;
    __shared__ bool valid;
    if (threadIdx.x == 0) {
        valid = infer_ragged_row_location(
            row, sequence_offsets, sequence_lengths, sequence_count, &sequence, &local_row);
    }
    __syncthreads();
    if (!valid) return;
    const std::uint32_t logical_row = start_positions[sequence] + local_row;
    const std::uint32_t slot =
        infer_paged_f32_slot(page_tables[sequence], logical_row, page_tokens);
    const std::size_t source = static_cast<std::size_t>(row) * width;
    const std::size_t destination =
        (static_cast<std::size_t>(slot) * page_tokens + logical_row % page_tokens) * width;
    for (std::uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        key_pool[destination + column] = key[source + column];
        value_pool[destination + column] = value[source + column];
    }
}

extern "C" cudaError_t infer_append_ragged_paged_kv_f32_on_stream(
    const float* key,
    const float* value,
    float* key_pool,
    float* value_pool,
    const std::uint32_t* const* page_tables,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t page_tokens,
    std::uint32_t width,
    cudaStream_t stream) {
    if (key == nullptr || value == nullptr || key_pool == nullptr || value_pool == nullptr ||
        page_tables == nullptr || sequence_offsets == nullptr || sequence_lengths == nullptr ||
        start_positions == nullptr || sequence_count == 0 || total_tokens == 0 ||
        page_tokens == 0 || width == 0) {
        return cudaErrorInvalidValue;
    }
    infer_append_ragged_paged_kv_f32_kernel<<<total_tokens, 256, 0, stream>>>(
        key, value, key_pool, value_pool, page_tables, sequence_offsets,
        sequence_lengths, start_positions, sequence_count, total_tokens, page_tokens, width);
    return cudaGetLastError();
}

__global__ void infer_ragged_paged_gqa_attention_f32_kernel(
    const float* query,
    const float* key_pool,
    const float* value_pool,
    const std::uint32_t* const* page_tables,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    float* output,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t page_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim) {
    extern __shared__ float shmem[];
    float* q_sh = shmem;
    float* partial = shmem + head_dim;
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t q_head = blockIdx.y;
    if (row >= total_tokens || q_head >= q_heads) return;
    __shared__ std::uint32_t sequence;
    __shared__ std::uint32_t local_row;
    __shared__ bool valid;
    if (threadIdx.x == 0) {
        valid = infer_ragged_row_location(
            row, sequence_offsets, sequence_lengths, sequence_count, &sequence, &local_row);
    }
    __syncthreads();
    if (!valid) return;

    const std::uint32_t groups_per_kv = q_heads / kv_heads;
    const std::uint32_t kv_head = q_head / groups_per_kv;
    const std::uint32_t kv_width = kv_heads * head_dim;
    const std::uint32_t hidden = q_heads * head_dim;
    const std::uint32_t cache_len = start_positions[sequence] + local_row + 1;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* q = query + static_cast<std::size_t>(row) * hidden + q_head * head_dim;
    const std::uint32_t* page_table = page_tables[sequence];
    for (std::uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) q_sh[dim] = q[dim];
    __syncthreads();

    float maximum = -INFINITY;
    float total_weight = 0.0f;
    float accum = 0.0f;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t warps = blockDim.x >> 5u;
    for (std::uint32_t cache_row = 0; cache_row < cache_len; ++cache_row) {
        const std::uint32_t slot = infer_paged_f32_slot(page_table, cache_row, page_tokens);
        const std::size_t storage_row =
            static_cast<std::size_t>(slot) * page_tokens + cache_row % page_tokens;
        const float* k = key_pool + storage_row * kv_width + kv_head * head_dim;
        const float* v = value_pool + storage_row * kv_width + kv_head * head_dim;
        float dot = threadIdx.x < head_dim ? q_sh[threadIdx.x] * k[threadIdx.x] : 0.0f;
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);
        if (lane == 0) partial[warp] = dot;
        __syncthreads();
        if (warp == 0 && lane == 0) {
            dot = 0.0f;
            for (std::uint32_t index = 0; index < warps; ++index) dot += partial[index];
            const float score = dot * scale;
            const float rescale = score > maximum ? expf(maximum - score) : 1.0f;
            const float weight = score > maximum ? 1.0f : expf(score - maximum);
            maximum = fmaxf(maximum, score);
            total_weight = total_weight * rescale + weight;
            partial[0] = rescale;
            partial[1] = weight;
        }
        __syncthreads();
        if (threadIdx.x < head_dim) {
            accum *= partial[0];
            accum = __fmaf_rn(partial[1], v[threadIdx.x], accum);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[0] = total_weight;
    __syncthreads();
    if (threadIdx.x < head_dim) {
        output[static_cast<std::size_t>(row) * hidden + q_head * head_dim + threadIdx.x] =
            accum / partial[0];
    }
}

extern "C" cudaError_t infer_ragged_paged_gqa_attention_f32_on_stream(
    const float* query,
    const float* key_pool,
    const float* value_pool,
    const std::uint32_t* const* page_tables,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    const std::uint32_t* start_positions,
    float* output,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t page_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_pool == nullptr || value_pool == nullptr ||
        page_tables == nullptr || sequence_offsets == nullptr || sequence_lengths == nullptr ||
        start_positions == nullptr || output == nullptr || sequence_count == 0 ||
        total_tokens == 0 || page_tokens == 0 || q_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || head_dim > kThreads || q_heads % kv_heads != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 blocks(total_tokens, q_heads);
    infer_ragged_paged_gqa_attention_f32_kernel<<<
        blocks, kThreads, (head_dim + kThreads) * sizeof(float), stream>>>(
        query, key_pool, value_pool, page_tables, sequence_offsets, sequence_lengths,
        start_positions, output, sequence_count, total_tokens, page_tokens,
        q_heads, kv_heads, head_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_prefill_gqa_attention_f32_on_stream(const float* query,
                                                             const float* key_cache,
                                                             const float* value_cache,
                                                             float* output,
                                                             std::uint32_t tokens,
                                                             std::uint32_t start_position,
                                                             std::uint32_t q_heads,
                                                             std::uint32_t kv_heads,
                                                             std::uint32_t head_dim,
                                                             cudaStream_t stream) {
    constexpr int kThreads = 256;
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
        tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || head_dim > kThreads ||
        (q_heads % kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 blocks(tokens, q_heads);
    infer_prefill_gqa_attention_f32_kernel<<<
        blocks, kThreads, (head_dim + kThreads) * sizeof(float), stream>>>(
        query, key_cache, value_cache, output, tokens, start_position, q_heads, kv_heads, head_dim);
    return cudaGetLastError();
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
    return infer_prefill_gqa_attention_f32_on_stream(
        query, key_cache, value_cache, output, tokens, start_position,
        q_heads, kv_heads, head_dim, nullptr);
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
    const std::uint32_t batch = blockIdx.y;
    const std::uint32_t row = blockIdx.x;
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

__global__ void infer_mask_logits_f32_batch_kernel(
    float* logits,
    const std::uint32_t* allowed,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t mask_words) {
    const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = blockIdx.y;
    if (row >= rows || col >= cols) return;
    const std::uint32_t word = allowed[row * mask_words + col / 32];
    if ((word & (1U << (col % 32))) == 0) {
        logits[static_cast<std::size_t>(row) * cols + col] = -INFINITY;
    }
}

extern "C" cudaError_t infer_mask_logits_f32_batch_on_stream(
    float* logits,
    const std::uint32_t* allowed,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t mask_words,
    cudaStream_t stream) {
    if (logits == nullptr || allowed == nullptr || rows == 0 || cols == 0 ||
        mask_words < (cols + 31) / 32) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const dim3 blocks((cols + kThreads - 1) / kThreads, rows);
    infer_mask_logits_f32_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        logits, allowed, rows, cols, mask_words);
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

__global__ void infer_speculative_accept_argmax_f32_kernel(
    const float* const* previous_logits,
    const float* verification_logits,
    const std::uint32_t* drafted_tokens,
    std::uint32_t* accepted_counts,
    std::uint32_t* next_tokens,
    std::uint32_t sequence_count,
    std::uint32_t draft_count,
    std::uint32_t vocab_size) {
    const std::uint32_t sequence = blockIdx.x;
    if (sequence >= sequence_count) {
        return;
    }
    extern __shared__ unsigned char shared_raw[];
    float* max_values = reinterpret_cast<float*>(shared_raw);
    std::uint32_t* max_indices =
        reinterpret_cast<std::uint32_t*>(max_values + blockDim.x);
    __shared__ std::uint32_t accepted;
    __shared__ std::uint32_t selected;
    if (threadIdx.x == 0) {
        accepted = 0;
        selected = 0;
    }
    __syncthreads();

    for (std::uint32_t step = 0; step <= draft_count; ++step) {
        const float* logits = step == 0
            ? previous_logits[sequence]
            : verification_logits +
                  (static_cast<std::size_t>(sequence) * draft_count + step - 1) *
                      vocab_size;
        float best_value = -INFINITY;
        std::uint32_t best_index = 0;
        for (std::uint32_t token = threadIdx.x; token < vocab_size;
             token += blockDim.x) {
            const float value = logits[token];
            if (value > best_value ||
                (value == best_value && token < best_index)) {
                best_value = value;
                best_index = token;
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
                    (other_value == max_values[threadIdx.x] &&
                     other_index < max_indices[threadIdx.x])) {
                    max_values[threadIdx.x] = other_value;
                    max_indices[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected = max_indices[0];
            if (step < draft_count &&
                selected == drafted_tokens[sequence * draft_count + step]) {
                ++accepted;
            } else {
                accepted_counts[sequence] = accepted;
                next_tokens[sequence] = selected;
            }
        }
        __syncthreads();
        if (accepted != step + 1) {
            return;
        }
    }
}

extern "C" cudaError_t infer_speculative_accept_argmax_f32_on_stream(
    const float* const* previous_logits,
    const float* verification_logits,
    const std::uint32_t* drafted_tokens,
    std::uint32_t* accepted_counts,
    std::uint32_t* next_tokens,
    std::uint32_t sequence_count,
    std::uint32_t draft_count,
    std::uint32_t vocab_size,
    cudaStream_t stream) {
    if (previous_logits == nullptr || verification_logits == nullptr ||
        drafted_tokens == nullptr || accepted_counts == nullptr ||
        next_tokens == nullptr || sequence_count == 0 || draft_count == 0 ||
        draft_count > 4 || vocab_size == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::size_t shared_bytes =
        kThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_speculative_accept_argmax_f32_kernel<<<
        sequence_count, kThreads, shared_bytes, stream>>>(
        previous_logits,
        verification_logits,
        drafted_tokens,
        accepted_counts,
        next_tokens,
        sequence_count,
        draft_count,
        vocab_size);
    return cudaGetLastError();
}

// =============================================================================
// Device-resident token sampling.

constexpr std::uint32_t kInferSamplingThreads = 256;
constexpr std::uint32_t kInferSamplingItemsPerThread = 4;
constexpr std::uint32_t kInferSamplingItemsPerBlock =
    kInferSamplingThreads * kInferSamplingItemsPerThread;
constexpr std::uint32_t kInferSamplingMaxTopK = 32;

struct InferSamplingParams {
    float temperature;
    float top_p;
    float presence_penalty;
    float frequency_penalty;
    float draw;
    std::uint32_t top_k;
    std::uint64_t token_counts;
};

struct InferSamplingResult {
    std::uint32_t id;
    float logit;
    float adjusted_logit;
    std::uint32_t status;
};

__device__ __forceinline__ std::uint64_t infer_sampling_key(
    float value,
    std::uint32_t id) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t ordered =
        (bits & 0x80000000U) != 0 ? ~bits : bits ^ 0x80000000U;
    return (static_cast<std::uint64_t>(ordered) << 32) |
           static_cast<std::uint64_t>(UINT32_MAX - id);
}

__device__ __forceinline__ std::uint32_t infer_sampling_key_id(
    std::uint64_t key) {
    return UINT32_MAX - static_cast<std::uint32_t>(key);
}

__device__ __forceinline__ float infer_sampling_key_value(std::uint64_t key) {
    const std::uint32_t ordered = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t bits =
        (ordered & 0x80000000U) != 0 ? ordered ^ 0x80000000U : ~ordered;
    return __uint_as_float(bits);
}

using InferSamplingBlockSort = cub::BlockRadixSort<
    std::uint64_t,
    kInferSamplingThreads,
    kInferSamplingItemsPerThread>;

// Each block sorts 1,024 vocabulary entries and emits its best 32. Later
// stages repeat the same reduction over those compact candidate lists.
__global__ void infer_sampling_logits_topk_kernel(
    const float* logits,
    const InferSamplingParams* params,
    std::uint64_t* output_keys,
    std::uint32_t vocab,
    std::uint32_t chunks_per_row) {
    const std::uint32_t row = blockIdx.x / chunks_per_row;
    const std::uint32_t chunk = blockIdx.x % chunks_per_row;
    const InferSamplingParams config = params[row];
    const auto* counts = reinterpret_cast<const std::uint32_t*>(config.token_counts);
    const float* row_logits = logits + static_cast<std::size_t>(row) * vocab;
    const std::uint32_t chunk_start = chunk * kInferSamplingItemsPerBlock;

    std::uint64_t keys[kInferSamplingItemsPerThread];
    #pragma unroll
    for (std::uint32_t item = 0; item < kInferSamplingItemsPerThread; ++item) {
        const std::uint32_t id = chunk_start + threadIdx.x + item * blockDim.x;
        std::uint64_t key = 0;
        if (id < vocab) {
            const float logit = row_logits[id];
            if (isfinite(logit)) {
                const std::uint32_t count = counts == nullptr ? 0U : counts[id];
                const float adjusted = logit -
                    (count == 0U ? 0.0f : config.presence_penalty) -
                    config.frequency_penalty * static_cast<float>(count);
                key = infer_sampling_key(adjusted, id);
            }
        }
        keys[item] = key;
    }

    __shared__ typename InferSamplingBlockSort::TempStorage sort_storage;
    InferSamplingBlockSort(sort_storage).SortDescending(keys);
    const std::uint32_t output_base =
        (row * chunks_per_row + chunk) * kInferSamplingMaxTopK;
    #pragma unroll
    for (std::uint32_t item = 0; item < kInferSamplingItemsPerThread; ++item) {
        const std::uint32_t rank = threadIdx.x * kInferSamplingItemsPerThread + item;
        if (rank < kInferSamplingMaxTopK) {
            output_keys[output_base + rank] = keys[item];
        }
    }
}

__global__ void infer_sampling_keys_topk_kernel(
    const std::uint64_t* input_keys,
    std::uint64_t* output_keys,
    std::uint32_t input_count_per_row,
    std::uint32_t output_chunks_per_row) {
    const std::uint32_t row = blockIdx.x / output_chunks_per_row;
    const std::uint32_t chunk = blockIdx.x % output_chunks_per_row;
    const std::uint32_t chunk_start = chunk * kInferSamplingItemsPerBlock;
    const std::uint64_t* row_input =
        input_keys + static_cast<std::size_t>(row) * input_count_per_row;

    std::uint64_t keys[kInferSamplingItemsPerThread];
    #pragma unroll
    for (std::uint32_t item = 0; item < kInferSamplingItemsPerThread; ++item) {
        const std::uint32_t index = chunk_start + threadIdx.x + item * blockDim.x;
        keys[item] = index < input_count_per_row ? row_input[index] : 0;
    }

    __shared__ typename InferSamplingBlockSort::TempStorage sort_storage;
    InferSamplingBlockSort(sort_storage).SortDescending(keys);
    const std::uint32_t output_base =
        (row * output_chunks_per_row + chunk) * kInferSamplingMaxTopK;
    #pragma unroll
    for (std::uint32_t item = 0; item < kInferSamplingItemsPerThread; ++item) {
        const std::uint32_t rank = threadIdx.x * kInferSamplingItemsPerThread + item;
        if (rank < kInferSamplingMaxTopK) {
            output_keys[output_base + rank] = keys[item];
        }
    }
}

__global__ void infer_sampling_finalize_kernel(
    const float* logits,
    const InferSamplingParams* params,
    const std::uint64_t* top_keys,
    InferSamplingResult* results,
    std::uint32_t vocab) {
    const std::uint32_t row = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }
    const InferSamplingParams config = params[row];
    const std::uint32_t k = config.temperature == 0.0f ? 1U : config.top_k;
    const std::uint64_t* row_keys = top_keys + row * kInferSamplingMaxTopK;
    const float* row_logits = logits + static_cast<std::size_t>(row) * vocab;

    InferSamplingResult result{};
    if (row_keys[0] == 0) {
        result.id = UINT32_MAX;
        result.status = 1;
        results[row] = result;
        return;
    }

    std::uint32_t selected_slot = 0;
    if (config.temperature != 0.0f) {
        float weights[kInferSamplingMaxTopK];
        float total = 0.0f;
        const float best_value = infer_sampling_key_value(row_keys[0]);
        for (std::uint32_t slot = 0; slot < k && row_keys[slot] != 0; ++slot) {
            const float weight = expf(
                (infer_sampling_key_value(row_keys[slot]) - best_value) /
                config.temperature);
            weights[slot] = weight;
            total += weight;
        }
        if (!isfinite(total) || total <= 0.0f) {
            result.id = UINT32_MAX;
            result.status = 2;
            results[row] = result;
            return;
        }

        float cumulative = 0.0f;
        std::uint32_t retained = 0;
        while (retained < k && row_keys[retained] != 0) {
            cumulative += weights[retained] / total;
            ++retained;
            if (cumulative >= config.top_p) {
                break;
            }
        }
        float retained_weight = 0.0f;
        for (std::uint32_t slot = 0; slot < retained; ++slot) {
            retained_weight += weights[slot];
        }
        float draw = fminf(fmaxf(config.draw, 0.0f), 0x1.fffffep-1f) * retained_weight;
        selected_slot = retained - 1;
        for (std::uint32_t slot = 0; slot < retained; ++slot) {
            if (draw < weights[slot]) {
                selected_slot = slot;
                break;
            }
            draw -= weights[slot];
        }
    }

    result.id = infer_sampling_key_id(row_keys[selected_slot]);
    result.logit = row_logits[result.id];
    result.adjusted_logit = infer_sampling_key_value(row_keys[selected_slot]);
    result.status = 0;
    results[row] = result;
    const auto* counts = reinterpret_cast<const std::uint32_t*>(config.token_counts);
    if (counts != nullptr) {
        auto* mutable_counts = const_cast<std::uint32_t*>(counts);
        mutable_counts[result.id] += 1U;
    }
}

extern "C" cudaError_t infer_sample_topk_topp_f32_batch_on_stream(
    const float* logits,
    const InferSamplingParams* params,
    std::uint64_t* stage_one_keys,
    std::uint64_t* stage_two_keys,
    std::uint64_t* top_keys,
    InferSamplingResult* results,
    std::uint32_t rows,
    std::uint32_t vocab,
    cudaStream_t stream) {
    if (logits == nullptr || params == nullptr || stage_one_keys == nullptr ||
        stage_two_keys == nullptr || top_keys == nullptr || results == nullptr ||
        rows == 0 || vocab == 0 || vocab > 1024U * 1024U) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t stage_one_chunks =
        (vocab + kInferSamplingItemsPerBlock - 1) / kInferSamplingItemsPerBlock;
    const std::uint32_t stage_one_count = stage_one_chunks * kInferSamplingMaxTopK;
    const std::uint32_t stage_two_chunks =
        (stage_one_count + kInferSamplingItemsPerBlock - 1) /
        kInferSamplingItemsPerBlock;

    infer_sampling_logits_topk_kernel<<<
        rows * stage_one_chunks, kInferSamplingThreads, 0, stream>>>(
        logits, params, stage_one_keys, vocab, stage_one_chunks);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    if (stage_two_chunks == 1) {
        infer_sampling_keys_topk_kernel<<<rows, kInferSamplingThreads, 0, stream>>>(
            stage_one_keys, top_keys, stage_one_count, 1);
    } else {
        infer_sampling_keys_topk_kernel<<<
            rows * stage_two_chunks, kInferSamplingThreads, 0, stream>>>(
            stage_one_keys, stage_two_keys, stage_one_count, stage_two_chunks);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
        const std::uint32_t stage_two_count =
            stage_two_chunks * kInferSamplingMaxTopK;
        infer_sampling_keys_topk_kernel<<<rows, kInferSamplingThreads, 0, stream>>>(
            stage_two_keys, top_keys, stage_two_count, 1);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    infer_sampling_finalize_kernel<<<rows, 32, 0, stream>>>(
        logits, params, top_keys, results, vocab);
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

__global__ void infer_bf16_matvec_logits_warp_rows_kernel(
    const float* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    float* __restrict__ logits,
    std::uint32_t rows,
    std::uint32_t cols) {
    extern __shared__ float input_sh[];
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t warps = blockDim.x >> 5u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * warps + warp;
    if (row >= rows) return;
    const float value = infer_bf16_row_dot_warp(
        weight + static_cast<std::size_t>(row) * cols, input_sh, cols);
    if (lane == 0) logits[row] = value;
}

__global__ void infer_bf16_matvec_logits_reuse_weights_batch_kernel(
    const float* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    float* __restrict__ logits,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols) {
    constexpr std::uint32_t kBatchTile = 8;
    const std::uint32_t warps = blockDim.x >> 5u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * warps + warp;
    if (row >= rows) return;
    const std::uint32_t batch_base = blockIdx.y * kBatchTile;
    const std::uint32_t active = min(kBatchTile, batch_size - batch_base);
    float acc[kBatchTile] = {};
    const std::uint16_t* row_weight = weight + static_cast<std::size_t>(row) * cols;
    for (std::uint32_t col = lane * 4; col < cols; col += 32 * 4) {
        const __nv_bfloat162 w0 =
            *reinterpret_cast<const __nv_bfloat162*>(row_weight + col);
        const __nv_bfloat162 w1 =
            *reinterpret_cast<const __nv_bfloat162*>(row_weight + col + 2);
        const float w0x = __bfloat162float(__low2bfloat16(w0));
        const float w0y = __bfloat162float(__high2bfloat16(w0));
        const float w1x = __bfloat162float(__low2bfloat16(w1));
        const float w1y = __bfloat162float(__high2bfloat16(w1));
#pragma unroll
        for (std::uint32_t batch = 0; batch < kBatchTile; ++batch) {
            if (batch >= active) continue;
            const float* input_row = input + static_cast<std::size_t>(batch_base + batch) * cols;
            acc[batch] = __fmaf_rn(w0x, input_row[col], acc[batch]);
            acc[batch] = __fmaf_rn(w0y, input_row[col + 1], acc[batch]);
            acc[batch] = __fmaf_rn(w1x, input_row[col + 2], acc[batch]);
            acc[batch] = __fmaf_rn(w1y, input_row[col + 3], acc[batch]);
        }
    }
#pragma unroll
    for (std::uint32_t batch = 0; batch < kBatchTile; ++batch) {
        if (batch >= active) continue;
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 16);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 8);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 4);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 2);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 1);
        if (lane == 0) {
            logits[static_cast<std::size_t>(batch_base + batch) * rows + row] = acc[batch];
        }
    }
}

// Reuses each BF16 weight row across a small input batch, then reduces the
// eight vocabulary rows owned by a block before writing one candidate per
// batch row to global memory. The dot-product loop intentionally matches
// infer_bf16_matvec_logits_reuse_weights_batch_kernel so target decisions stay
// bit-for-bit identical to materializing logits and reducing them afterwards.
__global__ void infer_bf16_lm_head_top1_batch_pass1_kernel(
    const float* __restrict__ input,
    const std::uint16_t* __restrict__ weight,
    float* __restrict__ scratch_value,
    std::uint32_t* __restrict__ scratch_index,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols) {
    constexpr std::uint32_t kBatchTile = 4;
    constexpr std::uint32_t kWarpsPerBlock = 8;
    __shared__ float block_values[kBatchTile * kWarpsPerBlock];
    __shared__ std::uint32_t block_indices[kBatchTile * kWarpsPerBlock];

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    const std::uint32_t batch_base = blockIdx.y * kBatchTile;
    const std::uint32_t active = min(kBatchTile, batch_size - batch_base);
    float acc[kBatchTile] = {};

    if (row < rows) {
        const std::uint16_t* row_weight =
            weight + static_cast<std::size_t>(row) * cols;
        for (std::uint32_t col = lane * 4; col < cols; col += 32 * 4) {
            const __nv_bfloat162 w0 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col);
            const __nv_bfloat162 w1 =
                *reinterpret_cast<const __nv_bfloat162*>(row_weight + col + 2);
            const float w0x = __bfloat162float(__low2bfloat16(w0));
            const float w0y = __bfloat162float(__high2bfloat16(w0));
            const float w1x = __bfloat162float(__low2bfloat16(w1));
            const float w1y = __bfloat162float(__high2bfloat16(w1));
#pragma unroll
            for (std::uint32_t batch = 0; batch < kBatchTile; ++batch) {
                if (batch >= active) continue;
                const float* input_row =
                    input + static_cast<std::size_t>(batch_base + batch) * cols;
                acc[batch] = __fmaf_rn(w0x, input_row[col], acc[batch]);
                acc[batch] = __fmaf_rn(w0y, input_row[col + 1], acc[batch]);
                acc[batch] = __fmaf_rn(w1x, input_row[col + 2], acc[batch]);
                acc[batch] = __fmaf_rn(w1y, input_row[col + 3], acc[batch]);
            }
        }
    }

#pragma unroll
    for (std::uint32_t batch = 0; batch < kBatchTile; ++batch) {
        if (batch >= active) continue;
        if (row < rows) {
            acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 16);
            acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 8);
            acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 4);
            acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 2);
            acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 1);
        }
        if (lane == 0) {
            const std::uint32_t offset = batch * kWarpsPerBlock + warp;
            block_values[offset] = row < rows ? acc[batch] : -INFINITY;
            block_indices[offset] = row < rows ? row : 0;
        }
    }
    __syncthreads();

    if (threadIdx.x < active) {
        const std::uint32_t batch = threadIdx.x;
        float best_value = -INFINITY;
        std::uint32_t best_index = 0;
#pragma unroll
        for (std::uint32_t candidate = 0; candidate < kWarpsPerBlock; ++candidate) {
            const std::uint32_t offset = batch * kWarpsPerBlock + candidate;
            const float value = block_values[offset];
            const std::uint32_t index = block_indices[offset];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        const std::uint32_t scratch_stride = gridDim.x;
        const std::size_t scratch_offset =
            static_cast<std::size_t>(batch_base + batch) * scratch_stride + blockIdx.x;
        scratch_value[scratch_offset] = best_value;
        scratch_index[scratch_offset] = best_index;
    }
}

__global__ void infer_lm_head_top1_batch_final_kernel(
    const float* __restrict__ scratch_value,
    const std::uint32_t* __restrict__ scratch_index,
    std::uint32_t* __restrict__ out_index,
    float* __restrict__ out_value,
    std::uint32_t scratch_stride) {
    extern __shared__ unsigned char sh_raw[];
    float* values = reinterpret_cast<float*>(sh_raw);
    std::uint32_t* indices =
        reinterpret_cast<std::uint32_t*>(values + blockDim.x);
    const std::size_t batch_offset =
        static_cast<std::size_t>(blockIdx.x) * scratch_stride;

    float best_value = -INFINITY;
    std::uint32_t best_index = 0;
    for (std::uint32_t candidate = threadIdx.x; candidate < scratch_stride;
         candidate += blockDim.x) {
        const float value = scratch_value[batch_offset + candidate];
        const std::uint32_t index = scratch_index[batch_offset + candidate];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const std::uint32_t other_index = indices[threadIdx.x + stride];
            if (other_value > values[threadIdx.x] ||
                (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_index[blockIdx.x] = indices[0];
        out_value[blockIdx.x] = values[0];
    }
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
    const cudaError_t shared_memory_status = cudaFuncSetAttribute(
        infer_bf16_matvec_logits_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(matvec_shmem));
    if (shared_memory_status != cudaSuccess) {
        return shared_memory_status;
    }
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
    const cudaError_t shared_memory_status = cudaFuncSetAttribute(
        infer_bf16_matvec_logits_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(matvec_shmem));
    if (shared_memory_status != cudaSuccess) {
        return shared_memory_status;
    }
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
    if (batch_size == 1 || (cols & 3u) != 0u) {
        const std::size_t shmem =
            kThreads * sizeof(float) + static_cast<std::size_t>(cols) * sizeof(float);
        const cudaError_t shared_memory_status = cudaFuncSetAttribute(
            infer_bf16_matvec_logits_batch_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem));
        if (shared_memory_status != cudaSuccess) {
            return shared_memory_status;
        }
        infer_bf16_matvec_logits_batch_kernel<<<dim3(rows, batch_size, 1), kThreads, shmem, stream>>>(
            input, weight, logits, batch_size, rows, cols);
        return cudaGetLastError();
    }
    constexpr std::uint32_t kBatchTile = 8;
    const std::uint32_t warps = kThreads / 32;
    infer_bf16_matvec_logits_reuse_weights_batch_kernel<<<
        dim3((rows + warps - 1) / warps, (batch_size + kBatchTile - 1) / kBatchTile),
        kThreads, 0, stream>>>(input, weight, logits, batch_size, rows, cols);
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

__global__ void infer_f32_to_bf16_kernel(const float* input,
                                         std::uint16_t* output,
                                         std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const __nv_bfloat16 value = __float2bfloat16_rn(input[idx]);
    output[idx] = *reinterpret_cast<const std::uint16_t*>(&value);
}

extern "C" cudaError_t infer_f32_to_bf16_on_stream(const float* input,
                                                     std::uint16_t* output,
                                                     std::uint32_t len,
                                                     cudaStream_t stream) {
    if (input == nullptr || output == nullptr || len == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((len + kThreads - 1) / kThreads);
    infer_f32_to_bf16_kernel<<<blocks, kThreads, 0, stream>>>(input, output, len);
    return cudaGetLastError();
}

__global__ void infer_pack_token_heads_bf16_kernel(
    const float* input,
    std::uint16_t* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t input_row_offset) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = tokens * heads * head_dim;
    if (index >= total) return;
    const std::uint32_t dim = index % head_dim;
    const std::uint32_t head = (index / head_dim) % heads;
    const std::uint32_t token = index / (heads * head_dim);
    const std::uint32_t destination = (head * tokens + token) * head_dim + dim;
    const std::uint32_t source =
        ((input_row_offset + token) * heads + head) * head_dim + dim;
    const __nv_bfloat16 value = __float2bfloat16_rn(input[source]);
    output[destination] = *reinterpret_cast<const std::uint16_t*>(&value);
}

extern "C" cudaError_t infer_pack_token_heads_bf16_on_stream(
    const float* input,
    std::uint16_t* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t input_row_offset,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || tokens == 0 || heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t total =
        static_cast<std::uint64_t>(tokens) * heads * head_dim;
    if (total > 0xffffffffu) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_pack_token_heads_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, tokens, heads, head_dim, input_row_offset);
    return cudaGetLastError();
}

__global__ void infer_pack_value_heads_bf16_kernel(
    const float* input,
    std::uint16_t* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = tokens * heads * head_dim;
    if (index >= total) return;
    const std::uint32_t dim = index % head_dim;
    const std::uint32_t head = (index / head_dim) % heads;
    const std::uint32_t token = index / (heads * head_dim);
    const std::uint32_t destination = (head * head_dim + dim) * tokens + token;
    const __nv_bfloat16 value = __float2bfloat16_rn(input[index]);
    output[destination] = *reinterpret_cast<const std::uint16_t*>(&value);
}

extern "C" cudaError_t infer_pack_value_heads_bf16_on_stream(
    const float* input,
    std::uint16_t* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || tokens == 0 || heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t total = static_cast<std::uint64_t>(tokens) * heads * head_dim;
    if (total > 0xffffffffu) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_pack_value_heads_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, tokens, heads, head_dim);
    return cudaGetLastError();
}

__global__ void infer_causal_window_softmax_f32_kernel(
    float* scores,
    std::uint32_t query_tokens,
    std::uint32_t key_tokens,
    std::uint32_t start_position,
    std::uint32_t window_tokens,
    float scale) {
    extern __shared__ float partial[];
    const std::uint32_t query = blockIdx.x;
    const std::uint32_t head = blockIdx.y;
    const std::uint32_t key_end = min(key_tokens, start_position + query + 1);
    const std::uint32_t key_start =
        window_tokens == 0 || key_end <= window_tokens ? 0 : key_end - window_tokens;
    float* row = scores + (head * query_tokens + query) * key_tokens;

    float local_max = -INFINITY;
    for (std::uint32_t key = key_start + threadIdx.x; key < key_end; key += blockDim.x) {
        local_max = fmaxf(local_max, row[key] * scale);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float row_max = partial[0];
    float local_sum = 0.0f;
    for (std::uint32_t key = threadIdx.x; key < key_tokens; key += blockDim.x) {
        float probability = 0.0f;
        if (key >= key_start && key < key_end) {
            probability = expf(row[key] * scale - row_max);
            local_sum += probability;
        }
        row[key] = probability;
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_sum = 1.0f / partial[0];
    for (std::uint32_t key = key_start + threadIdx.x; key < key_end; key += blockDim.x) {
        row[key] *= inverse_sum;
    }
}

extern "C" cudaError_t infer_causal_window_softmax_f32_on_stream(
    float* scores,
    std::uint32_t query_tokens,
    std::uint32_t key_tokens,
    std::uint32_t start_position,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t window_tokens,
    cudaStream_t stream) {
    if (scores == nullptr || query_tokens == 0 || key_tokens == 0 || heads == 0 ||
        head_dim == 0 || start_position > key_tokens || query_tokens > key_tokens - start_position) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    infer_causal_window_softmax_f32_kernel<<<
        dim3(query_tokens, heads, 1), kThreads, kThreads * sizeof(float), stream>>>(
        scores, query_tokens, key_tokens, start_position, window_tokens, scale);
    return cudaGetLastError();
}

__global__ void infer_causal_window_softmax_f32_to_bf16_kernel(
    const float* scores,
    std::uint16_t* probabilities,
    std::uint32_t query_tokens,
    std::uint32_t key_tokens,
    std::uint32_t start_position,
    std::uint32_t window_tokens,
    float scale) {
    extern __shared__ float partial[];
    const std::uint32_t query = blockIdx.x;
    const std::uint32_t head = blockIdx.y;
    const std::uint32_t key_end = min(key_tokens, start_position + query + 1);
    const std::uint32_t key_start =
        window_tokens == 0 || key_end <= window_tokens ? 0 : key_end - window_tokens;
    const float* row = scores + (head * query_tokens + query) * key_tokens;
    std::uint16_t* output =
        probabilities + (head * query_tokens + query) * key_tokens;

    float local_max = -INFINITY;
    for (std::uint32_t key = key_start + threadIdx.x; key < key_end; key += blockDim.x) {
        local_max = fmaxf(local_max, row[key] * scale);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = partial[0];

    float local_sum = 0.0f;
    for (std::uint32_t key = key_start + threadIdx.x; key < key_end; key += blockDim.x) {
        local_sum += expf(row[key] * scale - row_max);
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_sum = 1.0f / partial[0];
    for (std::uint32_t key = threadIdx.x; key < key_tokens; key += blockDim.x) {
        float probability = 0.0f;
        if (key >= key_start && key < key_end) {
            probability = expf(row[key] * scale - row_max) * inverse_sum;
        }
        const __nv_bfloat16 value = __float2bfloat16_rn(probability);
        output[key] = *reinterpret_cast<const std::uint16_t*>(&value);
    }
}

extern "C" cudaError_t infer_causal_window_softmax_f32_to_bf16_on_stream(
    const float* scores,
    std::uint16_t* probabilities,
    std::uint32_t query_tokens,
    std::uint32_t key_tokens,
    std::uint32_t start_position,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t window_tokens,
    cudaStream_t stream) {
    if (scores == nullptr || probabilities == nullptr || query_tokens == 0 ||
        key_tokens == 0 || heads == 0 || head_dim == 0 ||
        start_position > key_tokens || query_tokens > key_tokens - start_position) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    infer_causal_window_softmax_f32_to_bf16_kernel<<<
        dim3(query_tokens, heads, 1), kThreads, kThreads * sizeof(float), stream>>>(
        scores, probabilities, query_tokens, key_tokens, start_position,
        window_tokens, scale);
    return cudaGetLastError();
}

__global__ void infer_unpack_heads_f32_kernel(
    const float* input,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t output_row_offset) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = tokens * heads * head_dim;
    if (index >= total) return;
    const std::uint32_t dim = index % head_dim;
    const std::uint32_t head = (index / head_dim) % heads;
    const std::uint32_t token = index / (heads * head_dim);
    const std::uint32_t source = (head * tokens + token) * head_dim + dim;
    const std::uint32_t destination =
        ((output_row_offset + token) * heads + head) * head_dim + dim;
    output[destination] = input[source];
}

extern "C" cudaError_t infer_unpack_heads_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t output_row_offset,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || tokens == 0 || heads == 0 || head_dim == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t total =
        static_cast<std::uint64_t>(tokens) * heads * head_dim;
    if (total > 0xffffffffu) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_unpack_heads_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, tokens, heads, head_dim, output_row_offset);
    return cudaGetLastError();
}

__device__ __forceinline__ float infer_load_attention_output(const float* input,
                                                              std::size_t index) {
    return input[index];
}

__device__ __forceinline__ float infer_load_attention_output(
    const std::uint16_t* input,
    std::size_t index) {
    return __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(input + index));
}

template <typename Input>
__global__ void infer_unpack_heads_quantize_nvfp4_col_major_kernel(
    const Input* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t output_row_offset,
    float input_scale) {
    const std::uint32_t token = blockIdx.x;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5;
    constexpr std::uint32_t kWarps = 8;
    const std::uint32_t features = heads * head_dim;
    const std::uint32_t feature_blocks = (features + 15) / 16;
    const std::uint32_t feature_pairs = (feature_blocks + 1) / 2;
    const std::uint32_t output_row = output_row_offset + token;

    for (std::uint32_t feature_pair = warp; feature_pair < feature_pairs;
         feature_pair += kWarps) {
        const std::uint32_t half = lane >> 4;
        const std::uint32_t half_lane = lane & 15u;
        const std::uint32_t feature_block = feature_pair * 2 + half;
        const std::uint32_t feature = feature_pair * 32 + lane;
        float value = 0.0f;
        if (feature < features) {
            const std::uint32_t head = feature / head_dim;
            const std::uint32_t dim = feature % head_dim;
            const std::size_t source =
                (static_cast<std::size_t>(head) * tokens + token) * head_dim + dim;
            value = infer_load_attention_output(input, source) / input_scale;
        }
        const std::uint32_t mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
        float max_abs = fabsf(value);
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
        }
        std::uint32_t scale_word = 0;
        if (half_lane == 0 && feature_block < feature_blocks) {
            scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            scales[infer_ue4m3_tiled_scale_offset(output_row, feature_block, features)] =
                static_cast<std::uint8_t>(scale_word);
        }
        scale_word = __shfl_sync(mask, scale_word, 0, 16);
        const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
        const std::uint32_t pair_lane = (half_lane & 7u) * 2;
        const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
        const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
        if (half_lane < 8 && feature_block < feature_blocks) {
            const std::uint32_t lo_feature = feature_block * 16 + half_lane * 2;
            if (lo_feature < features) {
                const std::uint8_t lo = static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp4(
                        scale == 0.0f ? 0.0f : lo_value / scale,
                        __NV_E2M1, cudaRoundNearest) & 0x0f);
                std::uint8_t hi = 0;
                if (lo_feature + 1 < features) {
                    hi = static_cast<std::uint8_t>(
                        __nv_cvt_float_to_fp4(
                            scale == 0.0f ? 0.0f : hi_value / scale,
                            __NV_E2M1, cudaRoundNearest) & 0x0f);
                }
                packed[(static_cast<std::size_t>(output_row) * features + lo_feature) / 2] =
                    lo | (hi << 4);
            }
        }
    }
}

extern "C" cudaError_t infer_unpack_heads_quantize_nvfp4_col_major_f32_on_stream(
    const float* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t output_row_offset,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || packed == nullptr || scales == nullptr || tokens == 0 ||
        heads == 0 || head_dim == 0 || input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_unpack_heads_quantize_nvfp4_col_major_kernel<<<tokens, kThreads, 0, stream>>>(
        input, packed, scales, tokens, heads, head_dim, output_row_offset, input_scale);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_unpack_heads_quantize_nvfp4_col_major_bf16_on_stream(
    const std::uint16_t* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t output_row_offset,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || packed == nullptr || scales == nullptr || tokens == 0 ||
        heads == 0 || head_dim == 0 || input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_unpack_heads_quantize_nvfp4_col_major_kernel<<<tokens, kThreads, 0, stream>>>(
        input, packed, scales, tokens, heads, head_dim, output_row_offset, input_scale);
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

    const std::uint32_t lane = row & 31U;
    const std::uint32_t warp = row >> 5;
    __shared__ float warp_sums[4];
    __shared__ float reduced;
    float state_dot_k = infer_warp_reduce_sum(old_state * k_value);
    if (lane == 0) {
        warp_sums[warp] = state_dot_k;
    }
    __syncthreads();
    if (warp == 0) {
        state_dot_k = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
        if (lane == 0) {
            reduced = state_dot_k;
        }
    }
    __syncthreads();

    const float decay = expf(gate[head]);
    const float delta = (v[head_base + col] - decay * reduced) * beta[head];
    const float new_state = decay * old_state + k_value * delta;
    state[state_base + row] = new_state;

    float output_value = infer_warp_reduce_sum(new_state * q_value);
    if (lane == 0) {
        warp_sums[warp] = output_value;
    }
    __syncthreads();
    if (warp == 0) {
        output_value = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
        if (lane == 0) {
            output[head_base + col] =
                output_value * 0.08838834764831845f; // 1 / sqrt(128)
        }
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

// Ling 3 KDA keeps a distinct log-decay for every key coordinate. State is
// stored in the reference layout [head, key, value].
__global__ void infer_ling3_kda_128_f32_kernel(const float* q,
                                                const float* k,
                                                const float* v,
                                                const float* gate,
                                                const float* beta,
                                                float* state,
                                                float* output,
                                                std::uint32_t heads) {
    constexpr std::uint32_t kState = 128;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t value = blockIdx.y;
    const std::uint32_t key = threadIdx.x;
    if (head >= heads || value >= kState || key >= kState) return;

    const std::uint32_t vector_base = head * kState;
    const std::uint32_t state_index =
        head * kState * kState + key * kState + value;
    const float decayed = expf(gate[vector_base + key]) * state[state_index];
    const float k_value = k[vector_base + key];

    const std::uint32_t lane = key & 31U;
    const std::uint32_t warp = key >> 5;
    __shared__ float warp_sums[4];
    __shared__ float prediction;
    float sum = infer_warp_reduce_sum(decayed * k_value);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
        if (lane == 0) prediction = sum;
    }
    __syncthreads();

    const float delta =
        (v[vector_base + value] - prediction) * beta[head];
    const float updated = decayed + k_value * delta;
    state[state_index] = updated;

    sum = infer_warp_reduce_sum(updated * q[vector_base + key]);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
        if (lane == 0) {
            output[vector_base + value] =
                sum * 0.08838834764831845f; // 1 / sqrt(128)
        }
    }
}

extern "C" cudaError_t infer_ling3_kda_128_f32_on_stream(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr ||
        beta == nullptr || state == nullptr || output == nullptr || heads == 0) {
        return cudaErrorInvalidValue;
    }
    infer_ling3_kda_128_f32_kernel<<<dim3(heads, 128), 128, 0, stream>>>(
        q, k, v, gate, beta, state, output, heads);
    return cudaGetLastError();
}

__global__ void infer_ling3_kda_128_f32_chunks_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    std::uint32_t rows,
    std::uint32_t heads) {
    constexpr std::uint32_t kState = 128;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t value = blockIdx.y;
    const std::uint32_t key = threadIdx.x;
    if (head >= heads || value >= kState || key >= kState) return;
    const std::uint32_t lane = key & 31U;
    const std::uint32_t warp = key >> 5;
    __shared__ float warp_sums[4];
    __shared__ float prediction;
    const std::uint32_t state_index =
        head * kState * kState + key * kState + value;
    float state_value = state[state_index];
    for (std::uint32_t token = 0; token < rows; ++token) {
        const std::uint32_t vector_base = (token * heads + head) * kState;
        const float decayed = expf(gate[vector_base + key]) * state_value;
        const float k_value = k[vector_base + key];
        float sum = infer_warp_reduce_sum(decayed * k_value);
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
            if (lane == 0) prediction = sum;
        }
        __syncthreads();
        const float delta =
            (v[vector_base + value] - prediction) * beta[token * heads + head];
        state_value = decayed + k_value * delta;
        sum = infer_warp_reduce_sum(state_value * q[vector_base + key]);
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
            if (lane == 0) {
                output[vector_base + value] = sum * 0.08838834764831845f;
            }
        }
        __syncthreads();
    }
    state[state_index] = state_value;
}

extern "C" cudaError_t infer_ling3_kda_128_f32_chunks_on_stream(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    std::uint32_t rows,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr ||
        beta == nullptr || state == nullptr || output == nullptr || rows == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    infer_ling3_kda_128_f32_chunks_kernel<<<dim3(heads, 128), 128, 0, stream>>>(
        q, k, v, gate, beta, state, output, rows, heads);
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

    const std::uint32_t lane = row & 31U;
    const std::uint32_t warp = row >> 5;
    __shared__ float warp_sums[4];
    __shared__ float reduced;
    float state_dot_k = infer_warp_reduce_sum(old_state * k_value);
    if (lane == 0) {
        warp_sums[warp] = state_dot_k;
    }
    __syncthreads();
    if (warp == 0) {
        state_dot_k = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
        if (lane == 0) {
            reduced = state_dot_k;
        }
    }
    __syncthreads();

    const float decay = expf(gate[batch * heads + head]);
    const float delta =
        (v[vector_base + col] - decay * reduced) * beta[batch * heads + head];
    const float new_state = decay * old_state + k_value * delta;
    state[state_base + row] = new_state;

    float output_value = infer_warp_reduce_sum(new_state * q_value);
    if (lane == 0) {
        warp_sums[warp] = output_value;
    }
    __syncthreads();
    if (warp == 0) {
        output_value = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
        if (lane == 0) {
            output[vector_base + col] = output_value * 0.08838834764831845f;
        }
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

__global__ void infer_gated_delta_net_128_f32_chunks_warp_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* const* state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* output,
    std::uint32_t heads) {
    constexpr std::uint32_t kState = 128;
    const std::uint32_t sequence = blockIdx.x / heads;
    const std::uint32_t head = blockIdx.x % heads;
    const std::uint32_t col = blockIdx.y;
    const std::uint32_t row = threadIdx.x;
    if (col >= kState || row >= kState) return;

    const std::uint32_t offset = sequence_offsets[sequence];
    const std::uint32_t length = sequence_lengths[sequence];
    const std::uint32_t state_base = head * kState * kState + col * kState;
    float* state = state_table[sequence];

    const std::uint32_t lane = row & 31U;
    const std::uint32_t warp = row >> 5;
    __shared__ float warp_sums[4];
    __shared__ float reduced;

    float state_value = state[state_base + row];

    for (std::uint32_t token = 0; token < length; ++token) {
        const std::uint32_t vector_base = ((offset + token) * heads + head) * kState;
        const float q_value = q[vector_base + row];
        const float k_value = k[vector_base + row];

        float state_dot_k = infer_warp_reduce_sum(state_value * k_value);
        if (lane == 0) {
            warp_sums[warp] = state_dot_k;
        }
        __syncthreads();
        if (warp == 0) {
            state_dot_k = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
            if (lane == 0) {
                reduced = state_dot_k;
            }
        }
        __syncthreads();

        const float decay = expf(gate[(offset + token) * heads + head]);
        const float delta = (v[vector_base + col] - decay * reduced) *
            beta[(offset + token) * heads + head];
        state_value = decay * state_value + k_value * delta;

        float output_value = infer_warp_reduce_sum(state_value * q_value);
        if (lane == 0) {
            warp_sums[warp] = output_value;
        }
        __syncthreads();
        if (warp == 0) {
            output_value = infer_warp_reduce_sum(lane < 4 ? warp_sums[lane] : 0.0f);
            if (lane == 0) {
                output[vector_base + col] = output_value * 0.08838834764831845f;
            }
        }
        __syncthreads();
    }

    state[state_base + row] = state_value;
}

__global__ void infer_gated_delta_net_128_f32_chunks_multiwarp_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* const* state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* output,
    std::uint32_t heads) {
    constexpr std::uint32_t kState = 128;
    constexpr std::uint32_t kWarps = 8;
    const std::uint32_t sequence = blockIdx.x / heads;
    const std::uint32_t head = blockIdx.x % heads;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t col = blockIdx.y * kWarps + warp;
    if (col >= kState) return;

    const std::uint32_t offset = sequence_offsets[sequence];
    const std::uint32_t length = sequence_lengths[sequence];
    const std::uint32_t state_base = head * kState * kState + col * kState;
    float* state = state_table[sequence];
    float state_value[4];
    __shared__ float shared_q[128];
    __shared__ float shared_k[128];
    __shared__ float shared_decay;
    __shared__ float shared_beta;
#pragma unroll
    for (std::uint32_t item = 0; item < 4; ++item) {
        state_value[item] = state[state_base + lane + item * 32];
    }

    for (std::uint32_t token = 0; token < length; ++token) {
        const std::uint32_t vector_base = ((offset + token) * heads + head) * kState;
        if (threadIdx.x < kState) {
            shared_q[threadIdx.x] = q[vector_base + threadIdx.x];
            shared_k[threadIdx.x] = k[vector_base + threadIdx.x];
        }
        if (threadIdx.x == 0) {
            shared_decay = expf(gate[(offset + token) * heads + head]);
            shared_beta = beta[(offset + token) * heads + head];
        }
        __syncthreads();

        float state_dot_k = 0.0f;
#pragma unroll
        for (std::uint32_t item = 0; item < 4; ++item) {
            const std::uint32_t row = lane + item * 32;
            state_dot_k = fmaf(state_value[item], shared_k[row], state_dot_k);
        }
        state_dot_k = infer_warp_reduce_sum(state_dot_k);
        state_dot_k = __shfl_sync(0xffffffffu, state_dot_k, 0);

        float delta = 0.0f;
        if (lane == 0) {
            delta = (v[vector_base + col] - shared_decay * state_dot_k) * shared_beta;
        }
        delta = __shfl_sync(0xffffffffu, delta, 0);

        float output_value = 0.0f;
#pragma unroll
        for (std::uint32_t item = 0; item < 4; ++item) {
            const std::uint32_t row = lane + item * 32;
            state_value[item] =
                fmaf(shared_k[row], delta, shared_decay * state_value[item]);
            output_value = fmaf(state_value[item], shared_q[row], output_value);
        }
        output_value = infer_warp_reduce_sum(output_value);
        if (lane == 0) {
            output[vector_base + col] = output_value * 0.08838834764831845f;
        }
        __syncthreads();
    }
#pragma unroll
    for (std::uint32_t item = 0; item < 4; ++item) {
        state[state_base + lane + item * 32] = state_value[item];
    }
}

extern "C" cudaError_t infer_gated_delta_net_128_f32_chunks_on_stream(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* const* state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* output,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr ||
        state_table == nullptr || sequence_offsets == nullptr || sequence_lengths == nullptr ||
        output == nullptr || sequence_count == 0 || total_tokens == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    if (total_tokens / sequence_count >= 1024) {
        dim3 grid(sequence_count * heads, 8, 1);
        infer_gated_delta_net_128_f32_chunks_multiwarp_kernel<<<grid, 512, 0, stream>>>(
            q, k, v, gate, beta, state_table, sequence_offsets, sequence_lengths, output, heads);
    } else {
        dim3 grid(sequence_count * heads, 128, 1);
        infer_gated_delta_net_128_f32_chunks_warp_kernel<<<grid, 128, 0, stream>>>(
            q, k, v, gate, beta, state_table, sequence_offsets, sequence_lengths, output, heads);
    }
    return cudaGetLastError();
}

__global__ void infer_gather_f32_pointer_rows_kernel(
    float* const* input_table,
    float* output,
    std::uint32_t row_values) {
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t value = blockIdx.x * blockDim.x + threadIdx.x;
    if (value < row_values) {
        output[static_cast<std::size_t>(row) * row_values + value] =
            input_table[row][value];
    }
}

extern "C" cudaError_t infer_gather_f32_pointer_rows_on_stream(
    float* const* input_table,
    float* output,
    std::uint32_t rows,
    std::uint32_t row_values,
    cudaStream_t stream) {
    if (input_table == nullptr || output == nullptr || rows == 0 ||
        row_values == 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t blocks = (row_values + kThreads - 1) / kThreads;
    infer_gather_f32_pointer_rows_kernel<<<dim3(blocks, rows), kThreads, 0, stream>>>(
        input_table, output, row_values);
    return cudaGetLastError();
}

__global__ void infer_scatter_f32_pointer_rows_kernel(
    const float* input,
    float* const* output_table,
    std::uint32_t row_values) {
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t value = blockIdx.x * blockDim.x + threadIdx.x;
    if (value < row_values) {
        output_table[row][value] = input[static_cast<std::size_t>(row) * row_values + value];
    }
}

extern "C" cudaError_t infer_scatter_f32_pointer_rows_on_stream(
    const float* input,
    float* const* output_table,
    std::uint32_t rows,
    std::uint32_t row_values,
    cudaStream_t stream) {
    if (input == nullptr || output_table == nullptr || rows == 0 ||
        row_values == 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t blocks = (row_values + kThreads - 1) / kThreads;
    infer_scatter_f32_pointer_rows_kernel<<<dim3(blocks, rows), kThreads, 0, stream>>>(
        input, output_table, row_values);
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

extern "C" cudaError_t infer_fp8_linear_channel_scaled_f32_batch_configured_on_stream(
    const float* input,
    const std::uint8_t* weight,
    const float* channel_weight_scale,
    float* output,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t threads,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || channel_weight_scale == nullptr ||
        output == nullptr || batch_size == 0 || rows == 0 || cols == 0 ||
        threads < 64 || threads > 512 || (threads % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_fp8_linear_f32_kernel<<<dim3(rows, batch_size), threads, 0, stream>>>(
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

__global__ void infer_quantize_fp8_e4m3_bf16_channel_scaled_kernel(
    const std::uint16_t* input,
    const float* channel_scale,
    std::uint8_t* output,
    std::uint32_t len,
    std::uint32_t cols) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const auto value = *reinterpret_cast<const __nv_bfloat16*>(input + idx);
        const float scale = channel_scale[idx / cols];
        output[idx] = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(__bfloat162float(value) / scale,
                                  __NV_SATFINITE,
                                  __NV_E4M3));
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

extern "C" cudaError_t infer_quantize_fp8_e4m3_bf16_channel_scaled_on_stream(
    const std::uint16_t* input,
    const float* channel_scale,
    std::uint8_t* output,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || channel_scale == nullptr || output == nullptr || rows == 0 ||
        cols == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint64_t len = static_cast<std::uint64_t>(rows) * cols;
    if (len > UINT32_MAX) {
        return cudaErrorInvalidValue;
    }

    constexpr int kThreads = 256;
    const std::uint32_t blocks = (static_cast<std::uint32_t>(len) + kThreads - 1) / kThreads;
    infer_quantize_fp8_e4m3_bf16_channel_scaled_kernel<<<blocks, kThreads, 0, stream>>>(
        input, channel_scale, output, static_cast<std::uint32_t>(len), cols);
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

__global__ void infer_ling3_l2_norm_heads_128_kernel(float* values,
                                                      std::uint32_t heads) {
    constexpr std::uint32_t kDim = 128;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (head >= heads || lane >= kDim) return;
    float* row = values + head * kDim;
    __shared__ float partial[kDim];
    const float value = row[lane];
    partial[lane] = value * value;
    __syncthreads();
    for (std::uint32_t stride = kDim / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    row[lane] = value * rsqrtf(partial[0] + 1.0e-6f);
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

extern "C" cudaError_t infer_ling3_kda_prep_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* conv_state,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr ||
        k == nullptr || v == nullptr || conv_state == nullptr || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kHeadDim = 128;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t projection = heads * kHeadDim;
    const std::uint32_t conv_dim = projection * 3;
    infer_qwen36_gdn_prep_kernel<<<
        (conv_dim + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state, heads, heads, kHeadDim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_ling3_l2_norm_heads_128_kernel<<<heads, 128, 0, stream>>>(q, heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_ling3_l2_norm_heads_128_kernel<<<heads, 128, 0, stream>>>(k, heads);
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

__global__ void infer_qwen36_gdn_prep_chunks_kernel(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    std::uint32_t conv_dim) {
    const std::uint32_t sequence = blockIdx.y;
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_dim) return;
    const std::uint32_t key_dim = key_heads * head_dim;
    const std::uint32_t value_dim = value_heads * head_dim;
    const std::uint32_t offset = sequence_offsets[sequence];
    const std::uint32_t length = sequence_lengths[sequence];
    float* conv_state = conv_state_table[sequence];
    float s0 = conv_state[idx * 3 + 0];
    float s1 = conv_state[idx * 3 + 1];
    float s2 = conv_state[idx * 3 + 2];

    const float w0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 0));
    const float w1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 1));
    const float w2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 2));
    const float w3 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 3));

    for (std::uint32_t token = 0; token < length; ++token) {
        const std::uint32_t row = offset + token;
        const float input = qkv[row * conv_dim + idx];
        float mixed = input * w3;
        mixed += s0 * w0;
        mixed += s1 * w1;
        mixed += s2 * w2;
        const float activated = mixed / (1.0f + expf(-mixed));
        s0 = s1;
        s1 = s2;
        s2 = input;
        const std::uint32_t output_base = row * value_dim;
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
    conv_state[idx * 3 + 0] = s0;
    conv_state[idx * 3 + 1] = s1;
    conv_state[idx * 3 + 2] = s2;
}

extern "C" cudaError_t infer_qwen36_gdn_prep_chunks_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || conv_state_table == nullptr || sequence_offsets == nullptr ||
        sequence_lengths == nullptr || sequence_count == 0 || total_tokens == 0 ||
        key_heads == 0 || value_heads == 0 || head_dim != 128 ||
        value_heads % key_heads != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    infer_qwen36_gdn_prep_chunks_kernel<<<
        dim3((conv_dim + kThreads - 1) / kThreads, sequence_count, 1), kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state_table, sequence_offsets,
        sequence_lengths, key_heads, value_heads, head_dim, conv_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_l2_norm_heads_128_kernel<<<total_tokens * value_heads, 128, 0, stream>>>(
        q, total_tokens * value_heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_l2_norm_heads_128_kernel<<<total_tokens * value_heads, 128, 0, stream>>>(
        k, total_tokens * value_heads);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_ling3_kda_prep_chunks_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || conv_state_table == nullptr || sequence_offsets == nullptr ||
        sequence_lengths == nullptr || sequence_count == 0 || total_tokens == 0 ||
        heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kHeadDim = 128;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t conv_dim = heads * kHeadDim * 3;
    infer_qwen36_gdn_prep_chunks_kernel<<<
        dim3((conv_dim + kThreads - 1) / kThreads, sequence_count), kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state_table, sequence_offsets,
        sequence_lengths, heads, heads, kHeadDim, conv_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_ling3_l2_norm_heads_128_kernel<<<total_tokens * heads, 128, 0, stream>>>(
        q, total_tokens * heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_ling3_l2_norm_heads_128_kernel<<<total_tokens * heads, 128, 0, stream>>>(
        k, total_tokens * heads);
    return cudaGetLastError();
}

__global__ void infer_ling3_kda_prep_rows_kernel(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* conv_state,
    std::uint32_t rows,
    std::uint32_t projection) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t conv_dim = projection * 3;
    if (idx >= conv_dim) return;
    float s0 = conv_state[idx * 3 + 0];
    float s1 = conv_state[idx * 3 + 1];
    float s2 = conv_state[idx * 3 + 2];
    const float w0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 0));
    const float w1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 1));
    const float w2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 2));
    const float w3 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        conv_weight_bf16 + idx * 4 + 3));
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float input = qkv[static_cast<std::size_t>(row) * conv_dim + idx];
        float mixed = input * w3;
        mixed += s0 * w0;
        mixed += s1 * w1;
        mixed += s2 * w2;
        const float activated = mixed / (1.0f + expf(-mixed));
        s0 = s1;
        s1 = s2;
        s2 = input;
        float* destination = idx < projection
            ? q
            : (idx < projection * 2 ? k : v);
        const std::uint32_t feature = idx % projection;
        destination[static_cast<std::size_t>(row) * projection + feature] = activated;
    }
    conv_state[idx * 3 + 0] = s0;
    conv_state[idx * 3 + 1] = s1;
    conv_state[idx * 3 + 2] = s2;
}

extern "C" cudaError_t infer_ling3_kda_prep_rows_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    float* q,
    float* k,
    float* v,
    float* conv_state,
    std::uint32_t rows,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || conv_state == nullptr || rows == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kHeadDim = 128;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t projection = heads * kHeadDim;
    const std::uint32_t conv_dim = projection * 3;
    infer_ling3_kda_prep_rows_kernel<<<
        (conv_dim + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state, rows, projection);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_ling3_l2_norm_heads_128_kernel<<<rows * heads, 128, 0, stream>>>(q, rows * heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_ling3_l2_norm_heads_128_kernel<<<rows * heads, 128, 0, stream>>>(k, rows * heads);
    return cudaGetLastError();
}

__global__ void infer_qwen36_gdn_prep_chunks_bf16_kernel(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    std::uint16_t* q,
    std::uint16_t* k,
    std::uint16_t* v,
    float* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    std::uint32_t conv_dim) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = total_tokens * conv_dim;
    if (linear >= total) return;
    const std::uint32_t row = linear / conv_dim;
    const std::uint32_t idx = linear % conv_dim;
    std::uint32_t sequence = 0;
    while (sequence + 1 < sequence_count &&
           row >= sequence_offsets[sequence] + sequence_lengths[sequence]) {
        ++sequence;
    }
    const std::uint32_t offset = sequence_offsets[sequence];
    const std::uint32_t token = row - offset;
    const float* conv_state = conv_state_table[sequence] + idx * 3;
    float mixed = qkv[static_cast<std::size_t>(row) * conv_dim + idx] *
        __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
            conv_weight_bf16 + idx * 4 + 3));
#pragma unroll
    for (std::uint32_t lag = 1; lag <= 3; ++lag) {
        const float input = token >= lag
            ? qkv[(static_cast<std::size_t>(row - lag) * conv_dim) + idx]
            : conv_state[3 + token - lag];
        mixed = fmaf(
            input,
            __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                conv_weight_bf16 + idx * 4 + (3 - lag))),
            mixed);
    }
    const __nv_bfloat16 activated = __float2bfloat16(mixed / (1.0f + expf(-mixed)));
    const std::uint16_t encoded = *reinterpret_cast<const std::uint16_t*>(&activated);
    const std::uint32_t key_dim = key_heads * head_dim;
    const std::uint32_t value_dim = value_heads * head_dim;
    const std::uint32_t output_base = row * value_dim;
    if (idx < key_dim) {
        for (std::uint32_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
            q[output_base + repeat * key_dim + idx] = encoded;
        }
    } else if (idx < key_dim * 2) {
        const std::uint32_t k_idx = idx - key_dim;
        for (std::uint32_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
            k[output_base + repeat * key_dim + k_idx] = encoded;
        }
    } else {
        const std::uint32_t v_idx = idx - key_dim * 2;
        const std::uint32_t v_k_head = v_idx / head_dim;
        const std::uint32_t v_sub = v_idx % head_dim;
        const std::uint32_t v_per_k = value_heads / key_heads;
        const std::uint32_t k_head = v_k_head / v_per_k;
        const std::uint32_t v_sub_idx = v_k_head % v_per_k;
        const std::uint32_t tiled_v_idx =
            v_sub_idx * key_dim + k_head * head_dim + v_sub;
        v[output_base + tiled_v_idx] = encoded;
    }
}

__global__ void infer_qwen36_gdn_update_conv_state_kernel(
    const float* qkv,
    float* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t conv_dim) {
    const std::uint32_t sequence = blockIdx.y;
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_dim) return;
    const std::uint32_t offset = sequence_offsets[sequence];
    const std::uint32_t length = sequence_lengths[sequence];
    float* state = conv_state_table[sequence] + idx * 3;
    const float old_state[3] = {state[0], state[1], state[2]};
#pragma unroll
    for (std::uint32_t item = 0; item < 3; ++item) {
        const std::uint32_t timeline = length + item;
        state[item] = timeline < 3
            ? old_state[timeline]
            : qkv[(static_cast<std::size_t>(offset + timeline - 3) * conv_dim) + idx];
    }
}

__global__ void infer_l2_norm_heads_128_bf16_kernel(
    std::uint16_t* values,
    std::uint32_t heads) {
    constexpr std::uint32_t kDim = 128;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (head >= heads || lane >= kDim) return;
    std::uint16_t* row = values + static_cast<std::size_t>(head) * kDim;
    const float value = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(row + lane));
    __shared__ float partial[kDim];
    partial[lane] = value * value;
    __syncthreads();
    for (std::uint32_t stride = kDim / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const __nv_bfloat16 normalized =
        __float2bfloat16(value / fmaxf(sqrtf(partial[0]), 1.0e-6f));
    row[lane] = *reinterpret_cast<const std::uint16_t*>(&normalized);
}

extern "C" cudaError_t infer_qwen36_gdn_prep_chunks_bf16_on_stream(
    const float* qkv,
    const std::uint16_t* conv_weight_bf16,
    std::uint16_t* q,
    std::uint16_t* k,
    std::uint16_t* v,
    float* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_dim,
    cudaStream_t stream) {
    if (qkv == nullptr || conv_weight_bf16 == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || conv_state_table == nullptr || sequence_offsets == nullptr ||
        sequence_lengths == nullptr || sequence_count == 0 || total_tokens == 0 ||
        key_heads == 0 || value_heads == 0 || head_dim != 128 ||
        value_heads % key_heads != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t conv_dim = key_heads * head_dim * 2 + value_heads * head_dim;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t total = total_tokens * conv_dim;
    infer_qwen36_gdn_prep_chunks_bf16_kernel<<<
        (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        qkv, conv_weight_bf16, q, k, v, conv_state_table, sequence_offsets,
        sequence_lengths, sequence_count, total_tokens, key_heads, value_heads,
        head_dim, conv_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_qwen36_gdn_update_conv_state_kernel<<<
        dim3((conv_dim + kThreads - 1) / kThreads, sequence_count), kThreads, 0, stream>>>(
        qkv, conv_state_table, sequence_offsets, sequence_lengths, conv_dim);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_l2_norm_heads_128_bf16_kernel<<<total_tokens * value_heads, 128, 0, stream>>>(
        q, total_tokens * value_heads);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_l2_norm_heads_128_bf16_kernel<<<total_tokens * value_heads, 128, 0, stream>>>(
        k, total_tokens * value_heads);
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

__global__ void infer_ling3_kda_gate_f32_kernel(const float* raw_gate,
                                                 const float* beta_input,
                                                 const float* a_log,
                                                 const float* dt_bias,
                                                 float* gate,
                                                 float* beta,
                                                 std::uint32_t heads,
                                                 float lower_bound) {
    constexpr std::uint32_t kDim = 128;
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t len = heads * kDim;
    if (idx >= len) return;
    const std::uint32_t head = idx / kDim;
    const float activated = expf(a_log[head]) * (raw_gate[idx] + dt_bias[idx]);
    gate[idx] = lower_bound / (1.0f + expf(-activated));
    if ((idx % kDim) == 0) {
        beta[head] = 1.0f / (1.0f + expf(-beta_input[head]));
    }
}

extern "C" cudaError_t infer_ling3_kda_gate_f32_on_stream(
    const float* raw_gate,
    const float* beta_input,
    const float* a_log,
    const float* dt_bias,
    float* gate,
    float* beta,
    std::uint32_t heads,
    float lower_bound,
    cudaStream_t stream) {
    if (raw_gate == nullptr || beta_input == nullptr || a_log == nullptr ||
        dt_bias == nullptr || gate == nullptr || beta == nullptr || heads == 0 ||
        !isfinite(lower_bound) || lower_bound >= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t len = heads * 128;
    infer_ling3_kda_gate_f32_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        raw_gate, beta_input, a_log, dt_bias, gate, beta, heads, lower_bound);
    return cudaGetLastError();
}

__global__ void infer_ling3_kda_gate_f32_batch_kernel(
    const float* raw_gate,
    const float* beta_input,
    const float* a_log,
    const float* dt_bias,
    float* gate,
    float* beta,
    std::uint32_t rows,
    std::uint32_t heads,
    float lower_bound) {
    constexpr std::uint32_t kDim = 128;
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t width = heads * kDim;
    if (idx >= rows * width) return;
    const std::uint32_t feature = idx % width;
    const std::uint32_t head = feature / kDim;
    const float activated = expf(a_log[head]) * (raw_gate[idx] + dt_bias[feature]);
    gate[idx] = lower_bound / (1.0f + expf(-activated));
    if ((feature % kDim) == 0) {
        const std::uint32_t row = idx / width;
        beta[row * heads + head] =
            1.0f / (1.0f + expf(-beta_input[row * heads + head]));
    }
}

extern "C" cudaError_t infer_ling3_kda_gate_f32_batch_on_stream(
    const float* raw_gate,
    const float* beta_input,
    const float* a_log,
    const float* dt_bias,
    float* gate,
    float* beta,
    std::uint32_t rows,
    std::uint32_t heads,
    float lower_bound,
    cudaStream_t stream) {
    if (raw_gate == nullptr || beta_input == nullptr || a_log == nullptr ||
        dt_bias == nullptr || gate == nullptr || beta == nullptr || rows == 0 || heads == 0 ||
        !isfinite(lower_bound) || lower_bound >= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t len = rows * heads * 128;
    infer_ling3_kda_gate_f32_batch_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        raw_gate, beta_input, a_log, dt_bias, gate, beta, rows, heads, lower_bound);
    return cudaGetLastError();
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

__global__ void infer_qwen36_gdn_gate_batch_bf16_kernel(
    const float* alpha,
    const float* beta_input,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    std::uint16_t* gate,
    std::uint16_t* beta,
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
    const __nv_bfloat16 gate_value = __float2bfloat16(-expf(a_log) * softplus);
    const __nv_bfloat16 beta_value = __float2bfloat16(1.0f / (1.0f + expf(-beta_input[idx])));
    gate[idx] = *reinterpret_cast<const std::uint16_t*>(&gate_value);
    beta[idx] = *reinterpret_cast<const std::uint16_t*>(&beta_value);
}

extern "C" cudaError_t infer_qwen36_gdn_gate_batch_bf16_on_stream(
    const float* alpha,
    const float* beta_input,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    std::uint16_t* gate,
    std::uint16_t* beta,
    std::uint32_t rows,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (alpha == nullptr || beta_input == nullptr || a_log_bf16 == nullptr ||
        dt_bias_bf16 == nullptr || gate == nullptr || beta == nullptr ||
        rows == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t len = rows * heads;
    infer_qwen36_gdn_gate_batch_bf16_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        alpha, beta_input, a_log_bf16, dt_bias_bf16, gate, beta, heads, len);
    return cudaGetLastError();
}

__global__ void infer_qwen36_gdn_gate_paired_batch_bf16_kernel(
    const float* alpha_beta,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    std::uint16_t* gate,
    std::uint16_t* beta,
    std::uint32_t heads,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const std::uint32_t row = idx / heads;
    const std::uint32_t head = idx - row * heads;
    const std::uint32_t pair_offset = row * heads * 2;
    const float a_log = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(a_log_bf16 + head));
    const float dt_bias = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(dt_bias_bf16 + head));
    const float dt = alpha_beta[pair_offset + head] + dt_bias;
    const float softplus = log1pf(expf(-fabsf(dt))) + fmaxf(dt, 0.0f);
    const __nv_bfloat16 gate_value = __float2bfloat16(-expf(a_log) * softplus);
    const __nv_bfloat16 beta_value = __float2bfloat16(
        1.0f / (1.0f + expf(-alpha_beta[pair_offset + heads + head])));
    gate[idx] = *reinterpret_cast<const std::uint16_t*>(&gate_value);
    beta[idx] = *reinterpret_cast<const std::uint16_t*>(&beta_value);
}

extern "C" cudaError_t infer_qwen36_gdn_gate_paired_batch_bf16_on_stream(
    const float* alpha_beta,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    std::uint16_t* gate,
    std::uint16_t* beta,
    std::uint32_t rows,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (alpha_beta == nullptr || a_log_bf16 == nullptr || dt_bias_bf16 == nullptr ||
        gate == nullptr || beta == nullptr || rows == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t len = rows * heads;
    infer_qwen36_gdn_gate_paired_batch_bf16_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        alpha_beta, a_log_bf16, dt_bias_bf16, gate, beta, heads, len);
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

__global__ void infer_qwen36_gdn_gate_paired_batch_kernel(
    const float* alpha_beta,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    float* gate,
    float* beta,
    std::uint32_t heads,
    std::uint32_t len) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    const std::uint32_t row = idx / heads;
    const std::uint32_t head = idx - row * heads;
    const std::uint32_t pair_offset = row * heads * 2;
    const float a_log = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(a_log_bf16 + head));
    const float dt_bias = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(dt_bias_bf16 + head));
    const float dt = alpha_beta[pair_offset + head] + dt_bias;
    const float softplus = log1pf(expf(-fabsf(dt))) + fmaxf(dt, 0.0f);
    gate[idx] = -expf(a_log) * softplus;
    beta[idx] = 1.0f / (1.0f + expf(-alpha_beta[pair_offset + heads + head]));
}

extern "C" cudaError_t infer_qwen36_gdn_gate_paired_batch_on_stream(
    const float* alpha_beta,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* dt_bias_bf16,
    float* gate,
    float* beta,
    std::uint32_t rows,
    std::uint32_t heads,
    cudaStream_t stream) {
    if (alpha_beta == nullptr || a_log_bf16 == nullptr || dt_bias_bf16 == nullptr ||
        gate == nullptr || beta == nullptr || rows == 0 || heads == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t len = rows * heads;
    infer_qwen36_gdn_gate_paired_batch_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        alpha_beta, a_log_bf16, dt_bias_bf16, gate, beta, heads, len);
    return cudaGetLastError();
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

__global__ void infer_ling3_sigmoid_gated_rms_norm_f32_kernel(
    const float* input,
    const float* gate,
    const float* weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps) {
    extern __shared__ float partial[];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
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
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(cols) + eps);
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float sigmoid_gate = 1.0f / (1.0f + expf(-row_gate[col]));
        row_output[col] = row_input[col] * inv_rms * weight[col] * sigmoid_gate;
    }
}

extern "C" cudaError_t infer_ling3_sigmoid_gated_rms_norm_f32_on_stream(
    const float* input,
    const float* gate,
    const float* weight,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    float eps,
    cudaStream_t stream) {
    if (input == nullptr || gate == nullptr || weight == nullptr || output == nullptr ||
        rows == 0 || cols == 0 || !isfinite(eps) || eps < 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    infer_ling3_sigmoid_gated_rms_norm_f32_kernel<<<
        rows, kThreads, kThreads * sizeof(float), stream>>>(
        input, gate, weight, output, rows, cols, eps);
    return cudaGetLastError();
}

__global__ void infer_ling3_mla_pack_f32_kernel(
    const float* query_projection,
    const float* kv_projection,
    const float* shared_rope_key,
    float* query,
    float* key,
    float* value,
    std::uint32_t heads,
    std::uint32_t qk_nope_dim,
    std::uint32_t rope_dim,
    std::uint32_t value_dim,
    std::uint32_t rows) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t qk_dim = qk_nope_dim + rope_dim;
    const std::uint32_t qk_len = heads * qk_dim;
    const std::uint32_t value_len = heads * value_dim;
    const std::uint32_t row_len = qk_len > value_len ? qk_len : value_len;
    const std::uint32_t row = index / row_len;
    const std::uint32_t row_index = index % row_len;
    if (row >= rows) return;
    if (row_index < qk_len) {
        const std::uint32_t head = row_index / qk_dim;
        const std::uint32_t feature = row_index % qk_dim;
        const std::size_t qk_offset = static_cast<std::size_t>(row) * qk_len;
        const std::size_t kv_offset = static_cast<std::size_t>(row) * heads *
            (qk_nope_dim + value_dim);
        query[qk_offset + row_index] = query_projection[qk_offset + row_index];
        key[qk_offset + row_index] = feature < qk_nope_dim
            ? kv_projection[kv_offset + head * (qk_nope_dim + value_dim) + feature]
            : shared_rope_key[static_cast<std::size_t>(row) * rope_dim + feature - qk_nope_dim];
    }
    if (row_index < value_len) {
        const std::uint32_t head = row_index / value_dim;
        const std::uint32_t feature = row_index % value_dim;
        const std::size_t kv_offset = static_cast<std::size_t>(row) * heads *
            (qk_nope_dim + value_dim);
        value[static_cast<std::size_t>(row) * value_len + row_index] = kv_projection[
            kv_offset + head * (qk_nope_dim + value_dim) + qk_nope_dim + feature];
    }
}

__global__ void infer_ling3_mla_split_kv_a_f32_kernel(
    const float* input,
    float* compressed,
    float* rope,
    std::uint32_t rows,
    std::uint32_t compressed_dim,
    std::uint32_t rope_dim) {
    const std::uint32_t width = compressed_dim + rope_dim;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = index / width;
    const std::uint32_t feature = index % width;
    if (row >= rows) return;
    if (feature < compressed_dim) {
        compressed[static_cast<std::size_t>(row) * compressed_dim + feature] = input[index];
    } else {
        rope[static_cast<std::size_t>(row) * rope_dim + feature - compressed_dim] = input[index];
    }
}

extern "C" cudaError_t infer_ling3_mla_split_kv_a_f32_on_stream(
    const float* input,
    float* compressed,
    float* rope,
    std::uint32_t rows,
    std::uint32_t compressed_dim,
    std::uint32_t rope_dim,
    cudaStream_t stream) {
    if (input == nullptr || compressed == nullptr || rope == nullptr || rows == 0 ||
        compressed_dim == 0 || rope_dim == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint64_t len = static_cast<std::uint64_t>(rows) *
        (compressed_dim + rope_dim);
    if (len > UINT32_MAX) return cudaErrorInvalidValue;
    infer_ling3_mla_split_kv_a_f32_kernel<<<
        (static_cast<std::uint32_t>(len) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        input, compressed, rope, rows, compressed_dim, rope_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_ling3_mla_pack_f32_on_stream(
    const float* query_projection,
    const float* kv_projection,
    const float* shared_rope_key,
    float* query,
    float* key,
    float* value,
    std::uint32_t heads,
    std::uint32_t qk_nope_dim,
    std::uint32_t rope_dim,
    std::uint32_t value_dim,
    cudaStream_t stream) {
    if (query_projection == nullptr || kv_projection == nullptr ||
        shared_rope_key == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || heads == 0 || qk_nope_dim == 0 || rope_dim == 0 ||
        value_dim == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t qk_len = heads * (qk_nope_dim + rope_dim);
    const std::uint32_t value_len = heads * value_dim;
    const std::uint32_t len = qk_len > value_len ? qk_len : value_len;
    infer_ling3_mla_pack_f32_kernel<<<
        (len + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        query_projection, kv_projection, shared_rope_key, query, key, value,
        heads, qk_nope_dim, rope_dim, value_dim, 1);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_ling3_mla_pack_f32_batch_on_stream(
    const float* query_projection,
    const float* kv_projection,
    const float* shared_rope_key,
    float* query,
    float* key,
    float* value,
    std::uint32_t rows,
    std::uint32_t heads,
    std::uint32_t qk_nope_dim,
    std::uint32_t rope_dim,
    std::uint32_t value_dim,
    cudaStream_t stream) {
    if (query_projection == nullptr || kv_projection == nullptr ||
        shared_rope_key == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || rows == 0 || heads == 0 || qk_nope_dim == 0 ||
        rope_dim == 0 || value_dim == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t qk_len = heads * (qk_nope_dim + rope_dim);
    const std::uint32_t value_len = heads * value_dim;
    const std::uint32_t row_len = qk_len > value_len ? qk_len : value_len;
    const std::uint64_t len = static_cast<std::uint64_t>(rows) * row_len;
    if (len > UINT32_MAX) return cudaErrorInvalidValue;
    infer_ling3_mla_pack_f32_kernel<<<
        (static_cast<std::uint32_t>(len) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        query_projection, kv_projection, shared_rope_key, query, key, value,
        heads, qk_nope_dim, rope_dim, value_dim, rows);
    return cudaGetLastError();
}

__global__ void infer_ling3_mla_attention_f32_kernel(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t heads,
    std::uint32_t qk_dim,
    std::uint32_t value_dim,
    float scale) {
    const std::uint32_t head = blockIdx.x;
    if (head >= heads) return;
    const float* q = query + head * qk_dim;
    __shared__ float score;
    __shared__ float maximum;
    __shared__ float denominator;
    if (threadIdx.x == 0) maximum = -INFINITY;
    __syncthreads();
    for (std::uint32_t token = 0; token < cache_len; ++token) {
        const float* k = key_cache
            + (static_cast<std::size_t>(token) * heads + head) * qk_dim;
        float dot = 0.0f;
        for (std::uint32_t feature = threadIdx.x; feature < qk_dim;
             feature += blockDim.x) {
            dot = fmaf(q[feature], k[feature], dot);
        }
        dot = infer_block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            score = dot * scale;
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) denominator = 0.0f;
    float accumulator = 0.0f;
    __syncthreads();
    for (std::uint32_t token = 0; token < cache_len; ++token) {
        const float* k = key_cache
            + (static_cast<std::size_t>(token) * heads + head) * qk_dim;
        float dot = 0.0f;
        for (std::uint32_t feature = threadIdx.x; feature < qk_dim;
             feature += blockDim.x) {
            dot = fmaf(q[feature], k[feature], dot);
        }
        dot = infer_block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            score = expf(dot * scale - maximum);
            denominator += score;
        }
        __syncthreads();
        if (threadIdx.x < value_dim) {
            const float* v = value_cache
                + (static_cast<std::size_t>(token) * heads + head) * value_dim;
            accumulator = fmaf(score, v[threadIdx.x], accumulator);
        }
        __syncthreads();
    }
    if (threadIdx.x < value_dim) {
        output[head * value_dim + threadIdx.x] = accumulator / denominator;
    }
}

extern "C" cudaError_t infer_ling3_mla_attention_f32_on_stream(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t heads,
    std::uint32_t qk_dim,
    std::uint32_t value_dim,
    float scale,
    cudaStream_t stream) {
    if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        output == nullptr || cache_len == 0 || heads == 0 || qk_dim == 0 ||
        qk_dim > 512 || value_dim == 0 || value_dim > 256 ||
        !isfinite(scale) || scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    infer_ling3_mla_attention_f32_kernel<<<heads, 256, 0, stream>>>(
        query, key_cache, value_cache, output, cache_len, heads, qk_dim,
        value_dim, scale);
    return cudaGetLastError();
}

__global__ void infer_ling3_mla_paged_attention_f32_kernel(
    const float* query,
    const float* key_pool,
    const float* value_pool,
    const std::uint32_t* page_table,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t page_tokens,
    std::uint32_t heads,
    std::uint32_t qk_dim,
    std::uint32_t value_dim,
    float scale) {
    const std::uint32_t head = blockIdx.x;
    if (head >= heads) return;
    const float* q = query + head * qk_dim;
    __shared__ float score;
    __shared__ float maximum;
    __shared__ float denominator;
    if (threadIdx.x == 0) maximum = -INFINITY;
    __syncthreads();
    for (std::uint32_t token = 0; token < cache_len; ++token) {
        const std::uint32_t slot = page_table[token / page_tokens];
        const std::size_t row = static_cast<std::size_t>(slot) * page_tokens + token % page_tokens;
        const float* k = key_pool + (row * heads + head) * qk_dim;
        float dot = 0.0f;
        for (std::uint32_t feature = threadIdx.x; feature < qk_dim; feature += blockDim.x) {
            dot = fmaf(q[feature], k[feature], dot);
        }
        dot = infer_block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            score = dot * scale;
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) denominator = 0.0f;
    float accumulator = 0.0f;
    __syncthreads();
    for (std::uint32_t token = 0; token < cache_len; ++token) {
        const std::uint32_t slot = page_table[token / page_tokens];
        const std::size_t row = static_cast<std::size_t>(slot) * page_tokens + token % page_tokens;
        const float* k = key_pool + (row * heads + head) * qk_dim;
        float dot = 0.0f;
        for (std::uint32_t feature = threadIdx.x; feature < qk_dim; feature += blockDim.x) {
            dot = fmaf(q[feature], k[feature], dot);
        }
        dot = infer_block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            score = expf(dot * scale - maximum);
            denominator += score;
        }
        __syncthreads();
        if (threadIdx.x < value_dim) {
            const float* v = value_pool + (row * heads + head) * value_dim;
            accumulator = fmaf(score, v[threadIdx.x], accumulator);
        }
        __syncthreads();
    }
    if (threadIdx.x < value_dim) {
        output[head * value_dim + threadIdx.x] = accumulator / denominator;
    }
}

extern "C" cudaError_t infer_ling3_mla_paged_attention_f32_on_stream(
    const float* query,
    const float* key_pool,
    const float* value_pool,
    const std::uint32_t* page_table,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t page_tokens,
    std::uint32_t heads,
    std::uint32_t qk_dim,
    std::uint32_t value_dim,
    float scale,
    cudaStream_t stream) {
    if (query == nullptr || key_pool == nullptr || value_pool == nullptr ||
        page_table == nullptr || output == nullptr || cache_len == 0 || page_tokens == 0 ||
        heads == 0 || qk_dim == 0 || qk_dim > 512 || value_dim == 0 || value_dim > 256 ||
        !isfinite(scale) || scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    infer_ling3_mla_paged_attention_f32_kernel<<<heads, 256, 0, stream>>>(
        query, key_pool, value_pool, page_table, output, cache_len, page_tokens,
        heads, qk_dim, value_dim, scale);
    return cudaGetLastError();
}

__global__ void infer_ling3_mla_paged_causal_rows_f32_kernel(
    const float* query,
    const float* key_pool,
    const float* value_pool,
    const std::uint32_t* page_table,
    float* output,
    std::uint32_t start_position,
    std::uint32_t rows,
    std::uint32_t page_tokens,
    std::uint32_t heads,
    std::uint32_t qk_dim,
    std::uint32_t value_dim,
    float scale) {
    const std::uint32_t query_row = blockIdx.x / heads;
    const std::uint32_t head = blockIdx.x % heads;
    if (query_row >= rows) return;
    const std::uint32_t cache_len = start_position + query_row + 1;
    const float* q = query +
        (static_cast<std::size_t>(query_row) * heads + head) * qk_dim;
    __shared__ float score;
    __shared__ float maximum;
    __shared__ float denominator;
    if (threadIdx.x == 0) maximum = -INFINITY;
    __syncthreads();
    for (std::uint32_t token = 0; token < cache_len; ++token) {
        const std::uint32_t slot = page_table[token / page_tokens];
        const std::size_t row = static_cast<std::size_t>(slot) * page_tokens + token % page_tokens;
        const float* k = key_pool + (row * heads + head) * qk_dim;
        float dot = 0.0f;
        for (std::uint32_t feature = threadIdx.x; feature < qk_dim; feature += blockDim.x) {
            dot = fmaf(q[feature], k[feature], dot);
        }
        dot = infer_block_reduce_sum(dot);
        if (threadIdx.x == 0) maximum = fmaxf(maximum, dot * scale);
        __syncthreads();
    }
    if (threadIdx.x == 0) denominator = 0.0f;
    float accumulator = 0.0f;
    __syncthreads();
    for (std::uint32_t token = 0; token < cache_len; ++token) {
        const std::uint32_t slot = page_table[token / page_tokens];
        const std::size_t row = static_cast<std::size_t>(slot) * page_tokens + token % page_tokens;
        const float* k = key_pool + (row * heads + head) * qk_dim;
        float dot = 0.0f;
        for (std::uint32_t feature = threadIdx.x; feature < qk_dim; feature += blockDim.x) {
            dot = fmaf(q[feature], k[feature], dot);
        }
        dot = infer_block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            score = expf(dot * scale - maximum);
            denominator += score;
        }
        __syncthreads();
        if (threadIdx.x < value_dim) {
            const float* v = value_pool + (row * heads + head) * value_dim;
            accumulator = fmaf(score, v[threadIdx.x], accumulator);
        }
        __syncthreads();
    }
    if (threadIdx.x < value_dim) {
        output[(static_cast<std::size_t>(query_row) * heads + head) * value_dim + threadIdx.x] =
            accumulator / denominator;
    }
}

extern "C" cudaError_t infer_ling3_mla_paged_causal_rows_f32_on_stream(
    const float* query,
    const float* key_pool,
    const float* value_pool,
    const std::uint32_t* page_table,
    float* output,
    std::uint32_t start_position,
    std::uint32_t rows,
    std::uint32_t page_tokens,
    std::uint32_t heads,
    std::uint32_t qk_dim,
    std::uint32_t value_dim,
    float scale,
    cudaStream_t stream) {
    if (query == nullptr || key_pool == nullptr || value_pool == nullptr ||
        page_table == nullptr || output == nullptr || rows == 0 || page_tokens == 0 ||
        heads == 0 || qk_dim == 0 || qk_dim > 512 || value_dim == 0 || value_dim > 256 ||
        !isfinite(scale) || scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    infer_ling3_mla_paged_causal_rows_f32_kernel<<<rows * heads, 256, 0, stream>>>(
        query, key_pool, value_pool, page_table, output, start_position, rows,
        page_tokens, heads, qk_dim, value_dim, scale);
    return cudaGetLastError();
}

__global__ void infer_gated_rms_norm_quantize_nvfp4_col_major_f32_kernel(
    const float* input,
    const float* gate,
    const float* weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t heads,
    float eps,
    float input_scale) {
    constexpr std::uint32_t kHeadDim = 128;
    const std::uint32_t row = blockIdx.x / heads;
    const std::uint32_t head = blockIdx.x % heads;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t cols = heads * kHeadDim;
    const std::uint32_t head_offset = row * cols + head * kHeadDim;
    const float* head_input = input + head_offset;
    const float* head_gate = gate + head_offset;

    const float input_value = head_input[threadIdx.x];
    const float square_sum = infer_block_reduce_sum(input_value * input_value);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(kHeadDim) + eps);
    }
    __syncthreads();

    const std::uint32_t feature_pair = warp;
    const std::uint32_t half = lane >> 4;
    const std::uint32_t half_lane = lane & 15U;
    const std::uint32_t head_feature = feature_pair * 32 + lane;
    const std::uint32_t feature = head * kHeadDim + head_feature;
    const std::uint32_t feature_block = feature_pair * 2 + half;
    const float gate_value = head_gate[head_feature];
    const float silu_gate = gate_value / (1.0f + expf(-gate_value));
    const float value =
        input_value * inverse_rms * weight[head_feature] * silu_gate / input_scale;
    const std::uint32_t mask = half == 0 ? 0x0000ffffU : 0xffff0000U;
    float max_abs = fabsf(value);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(mask, max_abs, offset, 16));
    }
    std::uint32_t scale_word = 0;
    if (half_lane == 0) {
        scale_word = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const std::uint32_t global_block = head * 8 + feature_block;
        scales[infer_ue4m3_tiled_scale_offset(row, global_block, cols)] =
            static_cast<std::uint8_t>(scale_word);
    }
    scale_word = __shfl_sync(mask, scale_word, 0, 16);
    const float scale = infer_e4m3_value(static_cast<std::uint8_t>(scale_word));
    const std::uint32_t pair_lane = (half_lane & 7U) * 2;
    const float lo_value = __shfl_sync(mask, value, pair_lane, 16);
    const float hi_value = __shfl_sync(mask, value, pair_lane + 1, 16);
    if (half_lane < 8) {
        const std::uint32_t lo_feature =
            head * kHeadDim + feature_block * 16 + half_lane * 2;
        const std::uint8_t lo = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp4(
                scale == 0.0f ? 0.0f : lo_value / scale,
                __NV_E2M1, cudaRoundNearest) & 0x0f);
        const std::uint8_t hi = static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp4(
                scale == 0.0f ? 0.0f : hi_value / scale,
                __NV_E2M1, cudaRoundNearest) & 0x0f);
        packed[(row * cols + lo_feature) / 2] = lo | (hi << 4);
    }
}

extern "C" cudaError_t
infer_gated_rms_norm_quantize_nvfp4_col_major_f32_on_stream(
    const float* input,
    const float* gate,
    const float* weight,
    std::uint8_t* packed,
    std::uint8_t* scales,
    std::uint32_t rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    float eps,
    float input_scale,
    cudaStream_t stream) {
    if (input == nullptr || gate == nullptr || weight == nullptr || packed == nullptr ||
        scales == nullptr || rows == 0 || heads == 0 || head_dim != 128 ||
        input_scale <= 0.0f || !isfinite(input_scale)) {
        return cudaErrorInvalidValue;
    }
    infer_gated_rms_norm_quantize_nvfp4_col_major_f32_kernel<<<
        rows * heads, 128, 0, stream>>>(
        input, gate, weight, packed, scales, heads, eps, input_scale);
    return cudaGetLastError();
}

__global__ void infer_nemotron3_mamba_conv_update_f32_kernel(
    const float* projected,
    const std::uint16_t* conv_weight_bf16,
    const std::uint16_t* conv_bias_bf16,
    std::uint16_t* conv_state,
    float* conv_output,
    std::uint32_t intermediate_size,
    std::uint32_t conv_channels,
    std::uint32_t conv_kernel) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= conv_channels) {
        return;
    }
    std::uint16_t* state = conv_state + channel * conv_kernel;
    for (std::uint32_t index = 1; index < conv_kernel; ++index) {
        state[index - 1] = state[index];
    }
    *reinterpret_cast<__nv_bfloat16*>(state + conv_kernel - 1) =
        __float2bfloat16_rn(projected[intermediate_size + channel]);

    float value = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(conv_bias_bf16 + channel));
    const std::uint16_t* weight = conv_weight_bf16 + channel * conv_kernel;
    for (std::uint32_t index = 0; index < conv_kernel; ++index) {
        value = __fmaf_rn(
            __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(state + index)),
            __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(weight + index)),
            value);
    }
    conv_output[channel] = value / (1.0f + expf(-value));
}

extern "C" cudaError_t infer_nemotron3_mamba_conv_update_f32_on_stream(
    const float* projected,
    const std::uint16_t* conv_weight_bf16,
    const std::uint16_t* conv_bias_bf16,
    std::uint16_t* conv_state,
    float* conv_output,
    std::uint32_t intermediate_size,
    std::uint32_t conv_channels,
    std::uint32_t conv_kernel,
    cudaStream_t stream) {
    if (projected == nullptr || conv_weight_bf16 == nullptr ||
        conv_bias_bf16 == nullptr || conv_state == nullptr ||
        conv_output == nullptr || intermediate_size == 0 ||
        conv_channels == 0 || conv_kernel == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((conv_channels + kThreads - 1) / kThreads);
    infer_nemotron3_mamba_conv_update_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        projected,
        conv_weight_bf16,
        conv_bias_bf16,
        conv_state,
        conv_output,
        intermediate_size,
        conv_channels,
        conv_kernel);
    return cudaGetLastError();
}

__global__ void infer_nemotron3_mamba_conv_update_f32_chunks_kernel(
    const float* projected,
    const std::uint16_t* conv_weight_bf16,
    const std::uint16_t* conv_bias_bf16,
    std::uint16_t* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* conv_output,
    std::uint16_t* state_snapshots_bf16,
    std::uint32_t snapshot_slots,
    std::uint32_t projection_size,
    std::uint32_t intermediate_size,
    std::uint32_t conv_channels,
    std::uint32_t conv_kernel) {
    const std::uint32_t sequence = blockIdx.x;
    const std::uint32_t channel = blockIdx.y * blockDim.x + threadIdx.x;
    if (channel >= conv_channels) {
        return;
    }

    std::uint16_t* state = conv_state_table[sequence] + channel * conv_kernel;
    const std::uint16_t* weight = conv_weight_bf16 + channel * conv_kernel;
    const float bias = __bfloat162float(
        *reinterpret_cast<const __nv_bfloat16*>(conv_bias_bf16 + channel));
    const std::uint32_t begin = sequence_offsets[sequence];
    const std::uint32_t end = begin + sequence_lengths[sequence];
    const std::size_t state_size =
        static_cast<std::size_t>(conv_channels) * conv_kernel;
    if (state_snapshots_bf16 != nullptr) {
        std::uint16_t* initial = state_snapshots_bf16 +
            static_cast<std::size_t>(sequence) * snapshot_slots * state_size +
            static_cast<std::size_t>(channel) * conv_kernel;
        for (std::uint32_t index = 0; index < conv_kernel; ++index) {
            *reinterpret_cast<__nv_bfloat16*>(initial + index) =
                *reinterpret_cast<const __nv_bfloat16*>(state + index);
        }
    }
    for (std::uint32_t row = begin; row < end; ++row) {
        for (std::uint32_t index = 1; index < conv_kernel; ++index) {
            state[index - 1] = state[index];
        }
        *reinterpret_cast<__nv_bfloat16*>(state + conv_kernel - 1) =
            __float2bfloat16_rn(
                projected[static_cast<std::size_t>(row) * projection_size +
                          intermediate_size + channel]);

        float value = bias;
        for (std::uint32_t index = 0; index < conv_kernel; ++index) {
            value = __fmaf_rn(
                __bfloat162float(
                    *reinterpret_cast<const __nv_bfloat16*>(state + index)),
                __bfloat162float(
                    *reinterpret_cast<const __nv_bfloat16*>(weight + index)),
                value);
        }
        conv_output[static_cast<std::size_t>(row) * conv_channels + channel] =
            value / (1.0f + expf(-value));
        if (state_snapshots_bf16 != nullptr) {
            const std::uint32_t slot = row - begin + 1;
            if (slot < snapshot_slots) {
                std::uint16_t* snapshot = state_snapshots_bf16 +
                    (static_cast<std::size_t>(sequence) * snapshot_slots + slot) *
                        state_size +
                    static_cast<std::size_t>(channel) * conv_kernel;
                for (std::uint32_t index = 0; index < conv_kernel; ++index) {
                    *reinterpret_cast<__nv_bfloat16*>(snapshot + index) =
                        *reinterpret_cast<const __nv_bfloat16*>(state + index);
                }
            }
        }
    }
}

extern "C" cudaError_t infer_nemotron3_mamba_conv_update_f32_chunks_on_stream(
    const float* projected,
    const std::uint16_t* conv_weight_bf16,
    const std::uint16_t* conv_bias_bf16,
    std::uint16_t* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* conv_output,
    std::uint32_t sequence_count,
    std::uint32_t projection_size,
    std::uint32_t intermediate_size,
    std::uint32_t conv_channels,
    std::uint32_t conv_kernel,
    cudaStream_t stream) {
    if (projected == nullptr || conv_weight_bf16 == nullptr ||
        conv_bias_bf16 == nullptr || conv_state_table == nullptr ||
        sequence_offsets == nullptr || sequence_lengths == nullptr ||
        conv_output == nullptr || sequence_count == 0 || projection_size == 0 ||
        intermediate_size == 0 || conv_channels == 0 || conv_kernel == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t channel_blocks =
        (conv_channels + kThreads - 1) / kThreads;
    const dim3 blocks(sequence_count, channel_blocks, 1);
    infer_nemotron3_mamba_conv_update_f32_chunks_kernel<<<
        blocks, kThreads, 0, stream>>>(
        projected,
        conv_weight_bf16,
        conv_bias_bf16,
        conv_state_table,
        sequence_offsets,
        sequence_lengths,
        conv_output,
        nullptr,
        0,
        projection_size,
        intermediate_size,
        conv_channels,
        conv_kernel);
    return cudaGetLastError();
}

extern "C" cudaError_t
infer_nemotron3_mamba_conv_update_f32_chunks_snapshot_on_stream(
    const float* projected,
    const std::uint16_t* conv_weight_bf16,
    const std::uint16_t* conv_bias_bf16,
    std::uint16_t* const* conv_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* conv_output,
    std::uint16_t* state_snapshots_bf16,
    std::uint32_t sequence_count,
    std::uint32_t snapshot_slots,
    std::uint32_t projection_size,
    std::uint32_t intermediate_size,
    std::uint32_t conv_channels,
    std::uint32_t conv_kernel,
    cudaStream_t stream) {
    if (projected == nullptr || conv_weight_bf16 == nullptr ||
        conv_bias_bf16 == nullptr || conv_state_table == nullptr ||
        sequence_offsets == nullptr || sequence_lengths == nullptr ||
        conv_output == nullptr || state_snapshots_bf16 == nullptr ||
        sequence_count == 0 || snapshot_slots == 0 || projection_size == 0 ||
        intermediate_size == 0 || conv_channels == 0 || conv_kernel == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t channel_blocks =
        (conv_channels + kThreads - 1) / kThreads;
    const dim3 blocks(sequence_count, channel_blocks, 1);
    infer_nemotron3_mamba_conv_update_f32_chunks_kernel<<<
        blocks, kThreads, 0, stream>>>(
        projected,
        conv_weight_bf16,
        conv_bias_bf16,
        conv_state_table,
        sequence_offsets,
        sequence_lengths,
        conv_output,
        state_snapshots_bf16,
        snapshot_slots,
        projection_size,
        intermediate_size,
        conv_channels,
        conv_kernel);
    return cudaGetLastError();
}

__global__ void infer_nemotron3_mamba_state_update_f32_kernel(
    const float* projected,
    const float* conv_output,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* d_bf16,
    const std::uint16_t* dt_bias_bf16,
    const std::uint16_t* norm_weight_bf16,
    std::uint16_t* ssm_state,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t groups,
    std::uint32_t state_size,
    float dt_floor) {
    constexpr std::uint32_t kWarpsPerBlock = 8;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t flat = blockIdx.x * kWarpsPerBlock + warp;
    const std::uint32_t intermediate_size = heads * head_dim;
    if (flat >= intermediate_size) {
        return;
    }

    const std::uint32_t heads_per_group = heads / groups;
    const std::uint32_t group_width = heads_per_group * head_dim;
    const std::uint32_t group = flat / group_width;
    const std::uint32_t bc_width = groups * state_size;
    const std::uint32_t conv_channels = intermediate_size + 2 * bc_width;
    float x = 0.0f;
    float gate = 0.0f;
    float dt = 0.0f;
    float decay = 0.0f;
    float d = 0.0f;
    if (lane == 0) {
        const std::uint32_t head = flat / head_dim;
        x = conv_output[flat];
        gate = projected[flat];
        const float raw_dt = projected[intermediate_size + conv_channels + head];
        const float dt_bias = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(dt_bias_bf16 + head));
        dt = fmaxf(log1pf(expf(-fabsf(raw_dt + dt_bias))) +
                       fmaxf(raw_dt + dt_bias, 0.0f),
                   dt_floor);
        const float a_log = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(a_log_bf16 + head));
        decay = expf(-dt * expf(a_log));
        d = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(d_bf16 + head));
    }
    x = __shfl_sync(0xffffffff, x, 0);
    gate = __shfl_sync(0xffffffff, gate, 0);
    dt = __shfl_sync(0xffffffff, dt, 0);
    decay = __shfl_sync(0xffffffff, decay, 0);
    d = __shfl_sync(0xffffffff, d, 0);

    std::uint16_t* state = ssm_state + static_cast<std::size_t>(flat) * state_size;
    const float* b = conv_output + intermediate_size + group * state_size;
    const float* c = conv_output + intermediate_size + bc_width + group * state_size;
    float y = 0.0f;
    for (std::uint32_t state_index = lane; state_index < state_size; state_index += 32) {
        const float updated =
            __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                state + state_index)) * decay +
            dt * b[state_index] * x;
        *reinterpret_cast<__nv_bfloat16*>(state + state_index) =
            __float2bfloat16_rn(updated);
        y = __fmaf_rn(updated, c[state_index], y);
    }
    y = infer_warp_reduce_sum(y);
    if (lane == 0) {
        y += d * x;
        const float silu_gate = gate / (1.0f + expf(-gate));
        output[flat] = y * silu_gate;
    }
}

__global__ void infer_nemotron3_group_rms_norm_f32_kernel(
    float* output,
    const std::uint16_t* norm_weight_bf16,
    std::uint32_t group_width,
    float eps) {
    const std::uint32_t group_begin = blockIdx.x * group_width;
    float sum_squares = 0.0f;
    for (std::uint32_t group_index = threadIdx.x; group_index < group_width;
         group_index += blockDim.x) {
        const float value = output[group_begin + group_index];
        sum_squares = __fmaf_rn(value, value, sum_squares);
    }
    const float group_sum = infer_block_reduce_sum(sum_squares);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(group_sum / static_cast<float>(group_width) + eps);
    }
    __syncthreads();
    for (std::uint32_t group_index = threadIdx.x; group_index < group_width;
         group_index += blockDim.x) {
        const std::uint32_t flat = group_begin + group_index;
        const float weight = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(norm_weight_bf16 + flat));
        output[flat] *= inv_rms * weight;
    }
}

__global__ void infer_nemotron3_group_rms_norm_f32_chunks_kernel(
    float* output,
    const std::uint16_t* norm_weight_bf16,
    std::uint32_t groups,
    std::uint32_t group_width,
    float eps) {
    const std::uint32_t row = blockIdx.x / groups;
    const std::uint32_t group = blockIdx.x % groups;
    const std::uint32_t intermediate_size = groups * group_width;
    const std::uint32_t group_begin =
        row * intermediate_size + group * group_width;
    float sum_squares = 0.0f;
    for (std::uint32_t group_index = threadIdx.x; group_index < group_width;
         group_index += blockDim.x) {
        const float value = output[group_begin + group_index];
        sum_squares = __fmaf_rn(value, value, sum_squares);
    }
    const float group_sum = infer_block_reduce_sum(sum_squares);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(group_sum / static_cast<float>(group_width) + eps);
    }
    __syncthreads();
    for (std::uint32_t group_index = threadIdx.x; group_index < group_width;
         group_index += blockDim.x) {
        const std::uint32_t flat = group * group_width + group_index;
        const float weight = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(norm_weight_bf16 + flat));
        output[group_begin + group_index] *= inv_rms * weight;
    }
}

__global__ void infer_nemotron3_mamba_state_update_f32_chunks_kernel(
    const float* projected,
    const float* conv_output,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* d_bf16,
    const std::uint16_t* dt_bias_bf16,
    std::uint16_t* const* ssm_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* output,
    std::uint16_t* state_snapshots_bf16,
    std::uint32_t snapshot_slots,
    std::uint32_t projection_size,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t groups,
    std::uint32_t state_size,
    float dt_floor) {
    constexpr std::uint32_t kWarpsPerBlock = 8;
    const std::uint32_t sequence = blockIdx.x;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t flat = blockIdx.y * kWarpsPerBlock + warp;
    const std::uint32_t intermediate_size = heads * head_dim;
    if (flat >= intermediate_size) {
        return;
    }

    const std::uint32_t heads_per_group = heads / groups;
    const std::uint32_t group_width = heads_per_group * head_dim;
    const std::uint32_t group = flat / group_width;
    const std::uint32_t bc_width = groups * state_size;
    const std::uint32_t conv_channels = intermediate_size + 2 * bc_width;
    std::uint16_t* state = ssm_state_table[sequence] +
                   static_cast<std::size_t>(flat) * state_size;
    const std::uint32_t begin = sequence_offsets[sequence];
    const std::uint32_t end = begin + sequence_lengths[sequence];
    const std::size_t complete_state_size =
        static_cast<std::size_t>(intermediate_size) * state_size;
    if (state_snapshots_bf16 != nullptr) {
        std::uint16_t* initial = state_snapshots_bf16 +
            static_cast<std::size_t>(sequence) * snapshot_slots *
                complete_state_size +
            static_cast<std::size_t>(flat) * state_size;
        for (std::uint32_t state_index = lane; state_index < state_size;
             state_index += 32) {
            *reinterpret_cast<__nv_bfloat16*>(initial + state_index) =
                *reinterpret_cast<const __nv_bfloat16*>(state + state_index);
        }
    }
    for (std::uint32_t row = begin; row < end; ++row) {
        const float* row_projected =
            projected + static_cast<std::size_t>(row) * projection_size;
        const float* row_conv =
            conv_output + static_cast<std::size_t>(row) * conv_channels;
        float x = 0.0f;
        float gate = 0.0f;
        float dt = 0.0f;
        float decay = 0.0f;
        float d = 0.0f;
        if (lane == 0) {
            const std::uint32_t head = flat / head_dim;
            x = row_conv[flat];
            gate = row_projected[flat];
            const float raw_dt = row_projected[intermediate_size + conv_channels + head];
            const float dt_bias = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(dt_bias_bf16 + head));
            dt = fmaxf(log1pf(expf(-fabsf(raw_dt + dt_bias))) +
                           fmaxf(raw_dt + dt_bias, 0.0f),
                       dt_floor);
            const float a_log = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(a_log_bf16 + head));
            decay = expf(-dt * expf(a_log));
            d = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(d_bf16 + head));
        }
        x = __shfl_sync(0xffffffff, x, 0);
        gate = __shfl_sync(0xffffffff, gate, 0);
        dt = __shfl_sync(0xffffffff, dt, 0);
        decay = __shfl_sync(0xffffffff, decay, 0);
        d = __shfl_sync(0xffffffff, d, 0);

        const float* b = row_conv + intermediate_size + group * state_size;
        const float* c =
            row_conv + intermediate_size + bc_width + group * state_size;
        float y = 0.0f;
        for (std::uint32_t state_index = lane; state_index < state_size;
             state_index += 32) {
            const float updated =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                    state + state_index)) * decay +
                dt * b[state_index] * x;
            *reinterpret_cast<__nv_bfloat16*>(state + state_index) =
                __float2bfloat16_rn(updated);
            if (state_snapshots_bf16 != nullptr) {
                const std::uint32_t slot = row - begin + 1;
                if (slot < snapshot_slots) {
                    std::uint16_t* snapshot = state_snapshots_bf16 +
                        (static_cast<std::size_t>(sequence) * snapshot_slots + slot) *
                            complete_state_size +
                        static_cast<std::size_t>(flat) * state_size;
                    *reinterpret_cast<__nv_bfloat16*>(snapshot + state_index) =
                        __float2bfloat16_rn(updated);
                }
            }
            y = __fmaf_rn(updated, c[state_index], y);
        }
        y = infer_warp_reduce_sum(y);
        if (lane == 0) {
            y += d * x;
            const float silu_gate = gate / (1.0f + expf(-gate));
            output[static_cast<std::size_t>(row) * intermediate_size + flat] =
                y * silu_gate;
        }
    }
}

extern "C" cudaError_t infer_nemotron3_mamba_state_update_f32_on_stream(
    const float* projected,
    const float* conv_output,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* d_bf16,
    const std::uint16_t* dt_bias_bf16,
    const std::uint16_t* norm_weight_bf16,
    std::uint16_t* ssm_state,
    float* output,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t groups,
    std::uint32_t state_size,
    float dt_floor,
    float eps,
    cudaStream_t stream) {
    if (projected == nullptr || conv_output == nullptr || a_log_bf16 == nullptr ||
        d_bf16 == nullptr || dt_bias_bf16 == nullptr || norm_weight_bf16 == nullptr ||
        ssm_state == nullptr || output == nullptr || heads == 0 || head_dim == 0 ||
        groups == 0 || state_size == 0 || heads % groups != 0 ||
        !(dt_floor > 0.0f) || !(eps > 0.0f)) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    constexpr std::uint32_t kWarpsPerBlock = kThreads / 32;
    const std::uint32_t intermediate_size = heads * head_dim;
    const std::uint32_t state_blocks =
        (intermediate_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
    infer_nemotron3_mamba_state_update_f32_kernel<<<state_blocks, kThreads, 0, stream>>>(
        projected,
        conv_output,
        a_log_bf16,
        d_bf16,
        dt_bias_bf16,
        norm_weight_bf16,
        ssm_state,
        output,
        heads,
        head_dim,
        groups,
        state_size,
        dt_floor);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    const std::uint32_t group_width = intermediate_size / groups;
    infer_nemotron3_group_rms_norm_f32_kernel<<<groups, kThreads, 0, stream>>>(
        output, norm_weight_bf16, group_width, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_nemotron3_mamba_state_update_f32_chunks_on_stream(
    const float* projected,
    const float* conv_output,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* d_bf16,
    const std::uint16_t* dt_bias_bf16,
    const std::uint16_t* norm_weight_bf16,
    std::uint16_t* const* ssm_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* output,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t projection_size,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t groups,
    std::uint32_t state_size,
    float dt_floor,
    float eps,
    cudaStream_t stream) {
    if (projected == nullptr || conv_output == nullptr || a_log_bf16 == nullptr ||
        d_bf16 == nullptr || dt_bias_bf16 == nullptr || norm_weight_bf16 == nullptr ||
        ssm_state_table == nullptr || sequence_offsets == nullptr ||
        sequence_lengths == nullptr || output == nullptr || sequence_count == 0 ||
        total_tokens == 0 || projection_size == 0 || heads == 0 || head_dim == 0 ||
        groups == 0 || state_size == 0 || heads % groups != 0 ||
        !(dt_floor > 0.0f) || !(eps > 0.0f)) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    constexpr std::uint32_t kWarpsPerBlock = kThreads / 32;
    const std::uint32_t intermediate_size = heads * head_dim;
    const std::uint32_t state_blocks =
        (intermediate_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 blocks(sequence_count, state_blocks, 1);
    infer_nemotron3_mamba_state_update_f32_chunks_kernel<<<
        blocks, kThreads, 0, stream>>>(
        projected,
        conv_output,
        a_log_bf16,
        d_bf16,
        dt_bias_bf16,
        ssm_state_table,
        sequence_offsets,
        sequence_lengths,
        output,
        nullptr,
        0,
        projection_size,
        heads,
        head_dim,
        groups,
        state_size,
        dt_floor);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    const std::uint32_t group_width = intermediate_size / groups;
    const std::uint32_t norm_blocks = total_tokens * groups;
    infer_nemotron3_group_rms_norm_f32_chunks_kernel<<<
        norm_blocks, kThreads, 0, stream>>>(
        output, norm_weight_bf16, groups, group_width, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t
infer_nemotron3_mamba_state_update_f32_chunks_snapshot_on_stream(
    const float* projected,
    const float* conv_output,
    const std::uint16_t* a_log_bf16,
    const std::uint16_t* d_bf16,
    const std::uint16_t* dt_bias_bf16,
    const std::uint16_t* norm_weight_bf16,
    std::uint16_t* const* ssm_state_table,
    const std::uint32_t* sequence_offsets,
    const std::uint32_t* sequence_lengths,
    float* output,
    std::uint16_t* state_snapshots_bf16,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    std::uint32_t snapshot_slots,
    std::uint32_t projection_size,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t groups,
    std::uint32_t state_size,
    float dt_floor,
    float eps,
    cudaStream_t stream) {
    if (projected == nullptr || conv_output == nullptr || a_log_bf16 == nullptr ||
        d_bf16 == nullptr || dt_bias_bf16 == nullptr || norm_weight_bf16 == nullptr ||
        ssm_state_table == nullptr || sequence_offsets == nullptr ||
        sequence_lengths == nullptr || output == nullptr ||
        state_snapshots_bf16 == nullptr || sequence_count == 0 ||
        total_tokens == 0 || snapshot_slots == 0 || projection_size == 0 ||
        heads == 0 || head_dim == 0 || groups == 0 || state_size == 0 ||
        heads % groups != 0 || !(dt_floor > 0.0f) || !(eps > 0.0f)) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    constexpr std::uint32_t kWarpsPerBlock = kThreads / 32;
    const std::uint32_t intermediate_size = heads * head_dim;
    const std::uint32_t state_blocks =
        (intermediate_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 blocks(sequence_count, state_blocks, 1);
    infer_nemotron3_mamba_state_update_f32_chunks_kernel<<<
        blocks, kThreads, 0, stream>>>(
        projected,
        conv_output,
        a_log_bf16,
        d_bf16,
        dt_bias_bf16,
        ssm_state_table,
        sequence_offsets,
        sequence_lengths,
        output,
        state_snapshots_bf16,
        snapshot_slots,
        projection_size,
        heads,
        head_dim,
        groups,
        state_size,
        dt_floor);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    const std::uint32_t group_width = intermediate_size / groups;
    const std::uint32_t norm_blocks = total_tokens * groups;
    infer_nemotron3_group_rms_norm_f32_chunks_kernel<<<
        norm_blocks, kThreads, 0, stream>>>(
        output, norm_weight_bf16, groups, group_width, eps);
    return cudaGetLastError();
}

__global__ void infer_select_bf16_state_snapshot_kernel(
    std::uint16_t* const* state_table,
    const std::uint16_t* snapshots_bf16,
    const std::uint32_t* selected_slots,
    std::uint32_t snapshot_slots,
    std::uint32_t state_size) {
    const std::uint32_t sequence = blockIdx.x;
    const std::uint32_t slot = selected_slots[sequence];
    if (slot == snapshot_slots) {
        return;
    }
    if (slot > snapshot_slots) {
        return;
    }
    std::uint16_t* state = state_table[sequence];
    const std::uint16_t* snapshot = snapshots_bf16 +
        (static_cast<std::size_t>(sequence) * snapshot_slots + slot) * state_size;
    const std::uint32_t index = blockIdx.y * blockDim.x + threadIdx.x;
    if (index < state_size) {
        state[index] = snapshot[index];
    }
}

extern "C" cudaError_t infer_select_bf16_state_snapshot_on_stream(
    std::uint16_t* const* state_table,
    const std::uint16_t* snapshots_bf16,
    const std::uint32_t* selected_slots,
    std::uint32_t sequence_count,
    std::uint32_t snapshot_slots,
    std::uint32_t state_size,
    cudaStream_t stream) {
    if (state_table == nullptr || snapshots_bf16 == nullptr ||
        selected_slots == nullptr || sequence_count == 0 || snapshot_slots == 0 ||
        state_size == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint32_t state_blocks =
        (state_size + kThreads - 1) / kThreads;
    const dim3 blocks(sequence_count, state_blocks, 1);
    infer_select_bf16_state_snapshot_kernel<<<
        blocks, kThreads, 0, stream>>>(
        state_table,
        snapshots_bf16,
        selected_slots,
        snapshot_slots,
        state_size);
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

extern "C" cudaError_t infer_lm_head_top1_f32_batch_on_stream(
    const float* input,
    const std::uint16_t* weight,
    float* scratch_value,
    std::uint32_t* scratch_index,
    std::uint32_t scratch_len,
    std::uint32_t* out_index,
    float* out_value,
    std::uint32_t batch_size,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || scratch_value == nullptr ||
        scratch_index == nullptr || out_index == nullptr || out_value == nullptr ||
        batch_size == 0 || rows == 0 || cols == 0 || (cols & 3u) != 0u) {
        return cudaErrorInvalidValue;
    }

    constexpr std::uint32_t kWarpsPerBlock = 8;
    constexpr std::uint32_t kBatchTile = 4;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const std::uint32_t scratch_stride =
        (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const std::uint64_t required_scratch =
        static_cast<std::uint64_t>(batch_size) * scratch_stride;
    if (required_scratch > scratch_len) {
        return cudaErrorInvalidValue;
    }

    infer_bf16_lm_head_top1_batch_pass1_kernel<<<
        dim3(scratch_stride, (batch_size + kBatchTile - 1) / kBatchTile),
        kThreads, 0, stream>>>(
            input, weight, scratch_value, scratch_index, batch_size, rows, cols);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    constexpr std::uint32_t kFinalThreads = 256;
    const std::size_t final_shmem =
        kFinalThreads * (sizeof(float) + sizeof(std::uint32_t));
    infer_lm_head_top1_batch_final_kernel<<<
        batch_size, kFinalThreads, final_shmem, stream>>>(
            scratch_value, scratch_index, out_index, out_value, scratch_stride);
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

__global__ void infer_nvfp4_w4a16_matvec_f32_reuse_weights_batch_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ packed_weight,
    const std::uint8_t* __restrict__ weight_scale,
    float* __restrict__ output,
    std::uint32_t batch_size,
    std::uint32_t out_features,
    std::uint32_t in_features,
    float weight_scale_2) {
    const std::uint32_t warps_per_block = blockDim.x >> 5u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * warps_per_block + warp;
    if (row >= out_features) {
        return;
    }

    const std::uint8_t* packed_row = packed_weight + row * (in_features / 2);
    const std::uint8_t* row_scale = weight_scale + row * (in_features / 16);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (std::uint32_t col = lane * 4; col < in_features; col += 32 * 4) {
        const std::uint8_t b0 = packed_row[col / 2];
        const std::uint8_t b1 = packed_row[col / 2 + 1];
        const float scale = infer_ue4m3_value(row_scale[col / 16]);
        const float w0 = infer_e2m1_value(b0 & 0x0f) * scale;
        const float w1 = infer_e2m1_value(b0 >> 4) * scale;
        const float w2 = infer_e2m1_value(b1 & 0x0f) * scale;
        const float w3 = infer_e2m1_value(b1 >> 4) * scale;
#pragma unroll
        for (std::uint32_t batch = 0; batch < 4; ++batch) {
            if (batch >= batch_size) {
                continue;
            }
            const float* input_row = input + batch * in_features;
            acc[batch] = __fmaf_rn(input_row[col], w0, acc[batch]);
            acc[batch] = __fmaf_rn(input_row[col + 1], w1, acc[batch]);
            acc[batch] = __fmaf_rn(input_row[col + 2], w2, acc[batch]);
            acc[batch] = __fmaf_rn(input_row[col + 3], w3, acc[batch]);
        }
    }
#pragma unroll
    for (std::uint32_t batch = 0; batch < 4; ++batch) {
        if (batch >= batch_size) {
            continue;
        }
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 16);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 8);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 4);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 2);
        acc[batch] += __shfl_xor_sync(0xffffffffu, acc[batch], 1);
        if (lane == 0) {
            output[static_cast<std::size_t>(batch) * out_features + row] =
                acc[batch] * weight_scale_2;
        }
    }
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

__global__ void infer_nvfp4_w4a16_grouped_inputs_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* const* __restrict__ input_table,
    const std::uint8_t* const* __restrict__ packed_weight_table,
    const std::uint8_t* const* __restrict__ weight_scale_table,
    const float* __restrict__ weight_scale_2_table,
    float* const* __restrict__ output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    const std::uint32_t group = blockIdx.y;
    if (group >= groups) return;
    const float* input = input_table[group];
    extern __shared__ float input_sh[];
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= out_features) return;
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) return;
    const std::uint8_t* packed_weight = packed_weight_table[expert];
    const std::uint8_t* weight_scale = weight_scale_table[expert];
    const std::uint32_t row_byte_base = row * (in_features / 2);
    const std::uint32_t row_scale_base = row * (in_features / 16);
    const float value = infer_nvfp4_row_dot_warp(
        packed_weight + row_byte_base,
        weight_scale + row_scale_base,
        input_sh,
        in_features) * weight_scale_2_table[expert];
    if (lane == 0) output_table[group][row] = value;
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
    const cudaError_t shared_memory_status = cudaFuncSetAttribute(
        infer_nvfp4_w4a16_matvec_f32_warp_rows_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes));
    if (shared_memory_status != cudaSuccess) {
        return shared_memory_status;
    }
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
    if (batch_size > 1 && batch_size <= 4) {
        infer_nvfp4_w4a16_matvec_f32_reuse_weights_batch_kernel<<<
            grid_x, threads, 0, stream>>>(
            input, packed_weight, weight_scale, output, batch_size,
            out_features, in_features, weight_scale_2);
        return cudaGetLastError();
    }
    const std::size_t shmem_bytes = static_cast<std::size_t>(in_features) * sizeof(float);
    const cudaError_t shared_memory_status = cudaFuncSetAttribute(
        infer_nvfp4_w4a16_matvec_f32_warp_rows_batch_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes));
    if (shared_memory_status != cudaSuccess) {
        return shared_memory_status;
    }
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

extern "C" cudaError_t infer_nvfp4_w4a16_grouped_inputs_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* const* input_table,
    const std::uint8_t* const* packed_weight_table,
    const std::uint8_t* const* weight_scale_table,
    const float* weight_scale_2_table,
    float* const* output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input_table == nullptr || packed_weight_table == nullptr ||
        weight_scale_table == nullptr || weight_scale_2_table == nullptr ||
        output_table == nullptr || table_len == 0 || groups == 0 ||
        out_features == 0 || in_features == 0 || (in_features % 16) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const std::uint32_t grid_x = (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const std::size_t shared_bytes = static_cast<std::size_t>(in_features) * sizeof(float);
    infer_nvfp4_w4a16_grouped_inputs_matvec_f32_kernel<<<
        dim3(grid_x, groups), kThreads, shared_bytes, stream>>>(
        indices, input_table, packed_weight_table, weight_scale_table,
        weight_scale_2_table, output_table, table_len, groups, out_features, in_features);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Experimental blockwise-Q2 matvec.
//
// Four signed levels {-3, -1, 1, 3} share one BF16 scale per 64 consecutive
// input channels. The format is intentionally independent of external Q2
// checkpoint conventions while the resident-expert design is evaluated.
// ---------------------------------------------------------------------------

__device__ inline float infer_q2_row_dot_warp(
    const std::uint8_t* packed_row,
    const std::uint16_t* row_scales,
    const float* input_sh,
    std::uint32_t cols) {
    float acc = 0.0f;
    const std::uint32_t lane = threadIdx.x & 31u;
    for (std::uint32_t col = lane * 4; col < cols; col += 32 * 4) {
        const std::uint8_t packed = packed_row[col / 4];
        const float scale = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(
                row_scales + col / 64));
#pragma unroll
        for (std::uint32_t offset = 0; offset < 4; ++offset) {
            const std::uint32_t code = (packed >> (offset * 2)) & 0x03u;
            const float weight = static_cast<float>(
                static_cast<std::int32_t>(code) * 2 - 3) * scale;
            acc = __fmaf_rn(input_sh[col + offset], weight, acc);
        }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    return acc;
}

__global__ void infer_q2_w2a16_grouped_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* const* __restrict__ packed_weight_table,
    const std::uint16_t* const* __restrict__ weight_scale_table,
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
    const std::uint32_t packed_row_bytes = in_features / 4;
    const std::uint32_t scales_per_row = in_features / 64;
    const float value = infer_q2_row_dot_warp(
        packed_weight_table[expert] +
            static_cast<std::size_t>(row) * packed_row_bytes,
        weight_scale_table[expert] +
            static_cast<std::size_t>(row) * scales_per_row,
        input_sh,
        in_features);
    if (lane == 0) {
        output_table[group][row] = value;
    }
}

extern "C" cudaError_t infer_q2_w2a16_grouped_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* const* packed_weight_table,
    const std::uint16_t* const* weight_scale_table,
    float* const* output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr || packed_weight_table == nullptr ||
        weight_scale_table == nullptr || output_table == nullptr ||
        table_len == 0 || groups == 0 || out_features == 0 ||
        in_features == 0 || (in_features % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        groups);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_q2_w2a16_grouped_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        indices, input, packed_weight_table, weight_scale_table, output_table,
        table_len, groups, out_features, in_features);
    return cudaGetLastError();
}

__global__ void infer_q2_w2a16_grouped_inputs_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* const* __restrict__ input_table,
    const std::uint8_t* const* __restrict__ packed_weight_table,
    const std::uint16_t* const* __restrict__ weight_scale_table,
    float* const* __restrict__ output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    extern __shared__ float input_sh[];
    const std::uint32_t group = blockIdx.y;
    if (group >= groups) {
        return;
    }
    const float* input = input_table[group];
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) {
        return;
    }
    const std::uint32_t packed_row_bytes = in_features / 4;
    const std::uint32_t scales_per_row = in_features / 64;
    const float value = infer_q2_row_dot_warp(
        packed_weight_table[expert] +
            static_cast<std::size_t>(row) * packed_row_bytes,
        weight_scale_table[expert] +
            static_cast<std::size_t>(row) * scales_per_row,
        input_sh,
        in_features);
    if (lane == 0) {
        output_table[group][row] = value;
    }
}

extern "C" cudaError_t infer_q2_w2a16_grouped_inputs_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* const* input_table,
    const std::uint8_t* const* packed_weight_table,
    const std::uint16_t* const* weight_scale_table,
    float* const* output_table,
    std::uint32_t table_len,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input_table == nullptr ||
        packed_weight_table == nullptr || weight_scale_table == nullptr ||
        output_table == nullptr || table_len == 0 || groups == 0 ||
        out_features == 0 || in_features == 0 || (in_features % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        groups);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_q2_w2a16_grouped_inputs_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        indices, input_table, packed_weight_table, weight_scale_table,
        output_table, table_len, groups, out_features, in_features);
    return cudaGetLastError();
}

__global__ void infer_q2_nvfp4_mixed_grouped_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* const* __restrict__ q2_packed_weight_table,
    const std::uint16_t* const* __restrict__ q2_weight_scale_table,
    const std::uint32_t* __restrict__ expert_to_hot,
    const std::uint8_t* const* __restrict__ hot_packed_weight_table,
    const std::uint8_t* const* __restrict__ hot_weight_scale_table,
    const float* const* __restrict__ hot_weight_scale_2_table,
    float* const* __restrict__ output_table,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
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
    if (expert >= experts) {
        return;
    }

    float value;
    const std::uint32_t hot_slot = expert_to_hot[expert];
    if (hot_slot < hot_capacity) {
        const std::uint32_t row_byte_base = row * (in_features / 2);
        const std::uint32_t row_scale_base = row * (in_features / 16);
        value = infer_nvfp4_row_dot_warp(
            hot_packed_weight_table[hot_slot] + row_byte_base,
            hot_weight_scale_table[hot_slot] + row_scale_base,
            input_sh,
            in_features) * hot_weight_scale_2_table[hot_slot][row];
    } else {
        const std::uint32_t packed_row_bytes = in_features / 4;
        const std::uint32_t scales_per_row = in_features / 64;
        value = infer_q2_row_dot_warp(
            q2_packed_weight_table[expert] +
                static_cast<std::size_t>(row) * packed_row_bytes,
            q2_weight_scale_table[expert] +
                static_cast<std::size_t>(row) * scales_per_row,
            input_sh,
            in_features);
    }
    if (lane == 0) {
        output_table[group][row] = value;
    }
}

extern "C" cudaError_t infer_q2_nvfp4_mixed_grouped_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* const* q2_packed_weight_table,
    const std::uint16_t* const* q2_weight_scale_table,
    const std::uint32_t* expert_to_hot,
    const std::uint8_t* const* hot_packed_weight_table,
    const std::uint8_t* const* hot_weight_scale_table,
    const float* const* hot_weight_scale_2_table,
    float* const* output_table,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr ||
        q2_packed_weight_table == nullptr || q2_weight_scale_table == nullptr ||
        expert_to_hot == nullptr || hot_packed_weight_table == nullptr ||
        hot_weight_scale_table == nullptr || hot_weight_scale_2_table == nullptr ||
        output_table == nullptr || experts == 0 || hot_capacity == 0 ||
        groups == 0 || out_features == 0 || in_features == 0 ||
        (in_features % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        groups);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_q2_nvfp4_mixed_grouped_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        indices, input, q2_packed_weight_table, q2_weight_scale_table,
        expert_to_hot, hot_packed_weight_table, hot_weight_scale_table,
        hot_weight_scale_2_table, output_table, experts, hot_capacity, groups,
        out_features, in_features);
    return cudaGetLastError();
}

__global__ void infer_q2_nvfp4_mixed_grouped_inputs_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* const* __restrict__ input_table,
    const std::uint8_t* const* __restrict__ q2_packed_weight_table,
    const std::uint16_t* const* __restrict__ q2_weight_scale_table,
    const std::uint32_t* __restrict__ expert_to_hot,
    const std::uint8_t* const* __restrict__ hot_packed_weight_table,
    const std::uint8_t* const* __restrict__ hot_weight_scale_table,
    const float* const* __restrict__ hot_weight_scale_2_table,
    float* const* __restrict__ output_table,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    extern __shared__ float input_sh[];
    const std::uint32_t group = blockIdx.y;
    if (group >= groups) {
        return;
    }
    const float* input = input_table[group];
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input[col];
    }
    __syncthreads();

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t expert = indices[group];
    if (expert >= experts) {
        return;
    }

    float value;
    const std::uint32_t hot_slot = expert_to_hot[expert];
    if (hot_slot < hot_capacity) {
        const std::uint32_t row_byte_base = row * (in_features / 2);
        const std::uint32_t row_scale_base = row * (in_features / 16);
        value = infer_nvfp4_row_dot_warp(
            hot_packed_weight_table[hot_slot] + row_byte_base,
            hot_weight_scale_table[hot_slot] + row_scale_base,
            input_sh,
            in_features) * hot_weight_scale_2_table[hot_slot][row];
    } else {
        const std::uint32_t packed_row_bytes = in_features / 4;
        const std::uint32_t scales_per_row = in_features / 64;
        value = infer_q2_row_dot_warp(
            q2_packed_weight_table[expert] +
                static_cast<std::size_t>(row) * packed_row_bytes,
            q2_weight_scale_table[expert] +
                static_cast<std::size_t>(row) * scales_per_row,
            input_sh,
            in_features);
    }
    if (lane == 0) {
        output_table[group][row] = value;
    }
}

extern "C" cudaError_t infer_q2_nvfp4_mixed_grouped_inputs_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* const* input_table,
    const std::uint8_t* const* q2_packed_weight_table,
    const std::uint16_t* const* q2_weight_scale_table,
    const std::uint32_t* expert_to_hot,
    const std::uint8_t* const* hot_packed_weight_table,
    const std::uint8_t* const* hot_weight_scale_table,
    const float* const* hot_weight_scale_2_table,
    float* const* output_table,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t groups,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input_table == nullptr ||
        q2_packed_weight_table == nullptr || q2_weight_scale_table == nullptr ||
        expert_to_hot == nullptr || hot_packed_weight_table == nullptr ||
        hot_weight_scale_table == nullptr || hot_weight_scale_2_table == nullptr ||
        output_table == nullptr || experts == 0 || hot_capacity == 0 ||
        groups == 0 || out_features == 0 || in_features == 0 ||
        (in_features % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        groups);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_q2_nvfp4_mixed_grouped_inputs_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        indices, input_table, q2_packed_weight_table, q2_weight_scale_table,
        expert_to_hot, hot_packed_weight_table, hot_weight_scale_table,
        hot_weight_scale_2_table, output_table, experts, hot_capacity, groups,
        out_features, in_features);
    return cudaGetLastError();
}

__global__ void infer_q2_nvfp4_mixed_routed_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* const* __restrict__ q2_packed_weight_table,
    const std::uint16_t* const* __restrict__ q2_weight_scale_table,
    const std::uint32_t* __restrict__ expert_to_hot,
    const std::uint8_t* const* __restrict__ hot_packed_weight_table,
    const std::uint8_t* const* __restrict__ hot_weight_scale_table,
    const float* const* __restrict__ hot_weight_scale_2_table,
    float* __restrict__ output,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t routes,
    std::uint32_t routes_per_input,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    extern __shared__ float input_sh[];
    const std::uint32_t route = blockIdx.y;
    if (route >= routes) {
        return;
    }
    const float* input_row =
        input + static_cast<std::size_t>(route / routes_per_input) * in_features;
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input_row[col];
    }
    __syncthreads();

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t expert = indices[route];
    if (expert >= experts) {
        return;
    }

    float value;
    const std::uint32_t hot_slot = expert_to_hot[expert];
    if (hot_slot < hot_capacity) {
        const std::uint32_t row_byte_base = row * (in_features / 2);
        const std::uint32_t row_scale_base = row * (in_features / 16);
        value = infer_nvfp4_row_dot_warp(
            hot_packed_weight_table[hot_slot] + row_byte_base,
            hot_weight_scale_table[hot_slot] + row_scale_base,
            input_sh,
            in_features) * hot_weight_scale_2_table[hot_slot][row];
    } else {
        const std::uint32_t packed_row_bytes = in_features / 4;
        const std::uint32_t scales_per_row = in_features / 64;
        value = infer_q2_row_dot_warp(
            q2_packed_weight_table[expert] +
                static_cast<std::size_t>(row) * packed_row_bytes,
            q2_weight_scale_table[expert] +
                static_cast<std::size_t>(row) * scales_per_row,
            input_sh,
            in_features);
    }
    if (lane == 0) {
        output[static_cast<std::size_t>(route) * out_features + row] = value;
    }
}

extern "C" cudaError_t infer_q2_nvfp4_mixed_routed_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* const* q2_packed_weight_table,
    const std::uint16_t* const* q2_weight_scale_table,
    const std::uint32_t* expert_to_hot,
    const std::uint8_t* const* hot_packed_weight_table,
    const std::uint8_t* const* hot_weight_scale_table,
    const float* const* hot_weight_scale_2_table,
    float* output,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t routes,
    std::uint32_t routes_per_input,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr ||
        q2_packed_weight_table == nullptr || q2_weight_scale_table == nullptr ||
        expert_to_hot == nullptr || hot_packed_weight_table == nullptr ||
        hot_weight_scale_table == nullptr || hot_weight_scale_2_table == nullptr ||
        output == nullptr || experts == 0 || hot_capacity == 0 || routes == 0 ||
        routes_per_input == 0 || (routes % routes_per_input) != 0 ||
        out_features == 0 || in_features == 0 || (in_features % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        routes);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_q2_nvfp4_mixed_routed_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        indices, input, q2_packed_weight_table, q2_weight_scale_table,
        expert_to_hot, hot_packed_weight_table, hot_weight_scale_table,
        hot_weight_scale_2_table, output, experts, hot_capacity, routes,
        routes_per_input, out_features, in_features);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Blockwise-Q3 routed matvec with original-NVFP4 hot experts.
//
// Eight signed levels {-7, -5, -3, -1, 1, 3, 5, 7} share one BF16 scale per
// 128 consecutive input channels. Eight weights occupy three bytes.
// ---------------------------------------------------------------------------

__device__ inline float infer_q3_row_dot_warp(
    const std::uint8_t* packed_row,
    const std::uint16_t* row_scales,
    const float* input_sh,
    std::uint32_t cols) {
    float acc = 0.0f;
    const std::uint32_t lane = threadIdx.x & 31u;
    for (std::uint32_t col = lane * 8; col < cols; col += 32 * 8) {
        const std::uint32_t byte = (col * 3) / 8;
        const std::uint32_t packed =
            static_cast<std::uint32_t>(packed_row[byte]) |
            (static_cast<std::uint32_t>(packed_row[byte + 1]) << 8u) |
            (static_cast<std::uint32_t>(packed_row[byte + 2]) << 16u);
        const float scale = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(
                row_scales + col / 128));
#pragma unroll
        for (std::uint32_t offset = 0; offset < 8; ++offset) {
            const std::uint32_t code = (packed >> (offset * 3)) & 0x07u;
            const float weight = static_cast<float>(
                static_cast<std::int32_t>(code) * 2 - 7) * scale;
            acc = __fmaf_rn(input_sh[col + offset], weight, acc);
        }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    return acc;
}

__global__ void infer_q3_nvfp4_mixed_routed_matvec_f32_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* const* __restrict__ q3_packed_weight_table,
    const std::uint16_t* const* __restrict__ q3_weight_scale_table,
    const std::uint32_t* __restrict__ expert_to_hot,
    const std::uint8_t* const* __restrict__ hot_packed_weight_table,
    const std::uint8_t* const* __restrict__ hot_weight_scale_table,
    const float* const* __restrict__ hot_weight_scale_2_table,
    float* __restrict__ output,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t routes,
    std::uint32_t routes_per_input,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    extern __shared__ float input_sh[];
    const std::uint32_t route = blockIdx.y;
    if (route >= routes) {
        return;
    }
    const float* input_row =
        input + static_cast<std::size_t>(route / routes_per_input) * in_features;
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input_row[col];
    }
    __syncthreads();

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t expert = indices[route];
    if (expert >= experts) {
        return;
    }

    float value;
    const std::uint32_t hot_slot = expert_to_hot[expert];
    if (hot_slot < hot_capacity) {
        const std::uint32_t row_byte_base = row * (in_features / 2);
        const std::uint32_t row_scale_base = row * (in_features / 16);
        value = infer_nvfp4_row_dot_warp(
            hot_packed_weight_table[hot_slot] + row_byte_base,
            hot_weight_scale_table[hot_slot] + row_scale_base,
            input_sh,
            in_features) * hot_weight_scale_2_table[hot_slot][row];
    } else {
        const std::uint32_t packed_row_bytes = (in_features * 3) / 8;
        const std::uint32_t scales_per_row = in_features / 128;
        value = infer_q3_row_dot_warp(
            q3_packed_weight_table[expert] +
                static_cast<std::size_t>(row) * packed_row_bytes,
            q3_weight_scale_table[expert] +
                static_cast<std::size_t>(row) * scales_per_row,
            input_sh,
            in_features);
    }
    if (lane == 0) {
        output[static_cast<std::size_t>(route) * out_features + row] = value;
    }
}

extern "C" cudaError_t infer_q3_nvfp4_mixed_routed_matvec_f32_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* const* q3_packed_weight_table,
    const std::uint16_t* const* q3_weight_scale_table,
    const std::uint32_t* expert_to_hot,
    const std::uint8_t* const* hot_packed_weight_table,
    const std::uint8_t* const* hot_weight_scale_table,
    const float* const* hot_weight_scale_2_table,
    float* output,
    std::uint32_t experts,
    std::uint32_t hot_capacity,
    std::uint32_t routes,
    std::uint32_t routes_per_input,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr ||
        q3_packed_weight_table == nullptr || q3_weight_scale_table == nullptr ||
        expert_to_hot == nullptr || hot_packed_weight_table == nullptr ||
        hot_weight_scale_table == nullptr || hot_weight_scale_2_table == nullptr ||
        output == nullptr || experts == 0 || hot_capacity == 0 || routes == 0 ||
        routes_per_input == 0 || (routes % routes_per_input) != 0 ||
        out_features == 0 || in_features == 0 || (in_features % 128) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        routes);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_q3_nvfp4_mixed_routed_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        indices, input, q3_packed_weight_table, q3_weight_scale_table,
        expert_to_hot, hot_packed_weight_table, hot_weight_scale_table,
        hot_weight_scale_2_table, output, experts, hot_capacity, routes,
        routes_per_input, out_features, in_features);
    return cudaGetLastError();
}

__global__ void infer_nvfp4_slot_routed_matvec_f32_kernel(
    const std::uint32_t* __restrict__ slots,
    const float* __restrict__ input,
    const std::uint8_t* const* __restrict__ packed_weight_table,
    const std::uint8_t* const* __restrict__ weight_scale_table,
    const float* __restrict__ weight_scale_2_table,
    float* __restrict__ output,
    std::uint32_t capacity,
    std::uint32_t routes,
    std::uint32_t routes_per_input,
    std::uint32_t out_features,
    std::uint32_t in_features,
    std::uint32_t output_route_offset,
    std::uint32_t output_stride,
    std::uint32_t output_offset) {
    constexpr std::uint32_t kWarpsPerBlock = 16;
    extern __shared__ float input_sh[];
    const std::uint32_t route = blockIdx.y;
    if (route >= routes) {
        return;
    }
    const float* input_row =
        input + static_cast<std::size_t>(route / routes_per_input) * in_features;
    for (std::uint32_t col = threadIdx.x; col < in_features; col += blockDim.x) {
        input_sh[col] = input_row[col];
    }
    __syncthreads();

    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= out_features) {
        return;
    }
    const std::uint32_t slot = slots[route];
    if (slot >= capacity) {
        return;
    }
    const std::uint32_t row_byte_base = row * (in_features / 2);
    const std::uint32_t row_scale_base = row * (in_features / 16);
    const float value = infer_nvfp4_row_dot_warp(
        packed_weight_table[slot] + row_byte_base,
        weight_scale_table[slot] + row_scale_base,
        input_sh,
        in_features) * weight_scale_2_table[slot];
    if (lane == 0) {
        output[
            static_cast<std::size_t>(output_route_offset + route) * output_stride +
            output_offset + row] =
            value;
    }
}

extern "C" cudaError_t infer_nvfp4_slot_routed_matvec_f32_on_stream(
    const std::uint32_t* slots,
    const float* input,
    const std::uint8_t* const* packed_weight_table,
    const std::uint8_t* const* weight_scale_table,
    const float* weight_scale_2_table,
    float* output,
    std::uint32_t capacity,
    std::uint32_t routes,
    std::uint32_t routes_per_input,
    std::uint32_t out_features,
    std::uint32_t in_features,
    std::uint32_t output_route_offset,
    std::uint32_t output_stride,
    std::uint32_t output_offset,
    cudaStream_t stream) {
    if (slots == nullptr || input == nullptr || packed_weight_table == nullptr ||
        weight_scale_table == nullptr || weight_scale_2_table == nullptr ||
        output == nullptr || capacity == 0 || routes == 0 ||
        routes_per_input == 0 || (routes % routes_per_input) != 0 ||
        out_features == 0 || in_features == 0 || (in_features % 16) != 0 ||
        output_stride < out_features ||
        output_offset > output_stride - out_features) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 16;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid(
        (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock,
        routes);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(in_features) * sizeof(float);
    infer_nvfp4_slot_routed_matvec_f32_kernel<<<
        grid, kThreads, shared_bytes, stream>>>(
        slots, input, packed_weight_table, weight_scale_table,
        weight_scale_2_table, output, capacity, routes, routes_per_input,
        out_features, in_features, output_route_offset, output_stride,
        output_offset);
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
