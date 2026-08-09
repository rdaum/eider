#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr std::uint32_t kGroupSize = 64;

__device__ __forceinline__ std::uint32_t unpack_ternary(std::uint8_t packed) {
    std::uint32_t values = 0;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const std::uint32_t code = (packed >> (index * 2)) & 0x03u;
        const std::uint32_t value = code == 0 ? 0xffu : code - 1u;
        values |= value << (index * 8);
    }
    return values;
}

__global__ void ternary_g64_quantize_i8_f32_kernel(const float* input,
                                                     std::int8_t* output,
                                                     float* dequant_scales,
                                                     std::uint32_t cols) {
    const std::uint32_t batch = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    const std::uint32_t col = group * kGroupSize + threadIdx.x;
    const std::size_t input_index = static_cast<std::size_t>(batch) * cols + col;

    float maximum = fabsf(input[input_index]);
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    __shared__ float warp_maximum[2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_maximum[warp] = maximum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        warp_maximum[0] = fmaxf(warp_maximum[0], warp_maximum[1]);
        dequant_scales[static_cast<std::size_t>(batch) * (cols / kGroupSize) + group] =
            warp_maximum[0] / 127.0f;
    }
    __syncthreads();

    const float scale = warp_maximum[0] == 0.0f ? 0.0f : 127.0f / warp_maximum[0];
    const int quantized = __float2int_rn(input[input_index] * scale);
    output[input_index] = static_cast<std::int8_t>(max(-127, min(127, quantized)));
}

__global__ void ternary_g64_w2a8_linear_f32_kernel(
    const std::int8_t* input,
    const float* input_scales,
    const std::uint8_t* weight,
    const float* weight_scales,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    constexpr std::uint32_t kWarpsPerBlock = 8;
    constexpr std::uint32_t kPackedValuesPerGroup = kGroupSize / 4;
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) {
        return;
    }
    const std::uint32_t batch = blockIdx.y;
    const std::uint32_t groups = cols / kGroupSize;
    const auto* input4 = reinterpret_cast<const int*>(
        input + static_cast<std::size_t>(batch) * cols);
    const auto* row_weight =
        weight + static_cast<std::size_t>(row) * (cols / 4);
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row) * groups;
    const auto* batch_scales =
        input_scales + static_cast<std::size_t>(batch) * groups;

    float partial = 0.0f;
    for (std::uint32_t group = lane; group < groups; group += 32) {
        int sum = 0;
#pragma unroll
        for (std::uint32_t packed = 0; packed < kPackedValuesPerGroup; ++packed) {
            const std::uint32_t packed_col = group * kPackedValuesPerGroup + packed;
            const int packed_weight =
                static_cast<int>(unpack_ternary(row_weight[packed_col]));
            sum = __dp4a(packed_weight, input4[packed_col], sum);
        }
        partial += static_cast<float>(sum) * batch_scales[group] * row_scales[group];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) {
        output[static_cast<std::size_t>(batch) * rows + row] = partial;
    }
}

__global__ void ternary_g64_w2a8_linear_tiled_f32_kernel(
    const std::int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const std::uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t rows,
    std::uint32_t cols) {
    constexpr std::uint32_t kBatchTile = 8;
    constexpr std::uint32_t kRowTile = 32;
    constexpr std::uint32_t kPackedValuesPerGroup = kGroupSize / 4;
    __align__(16) __shared__ int input_tile[kBatchTile][kPackedValuesPerGroup];
    __shared__ int weight_tile[kPackedValuesPerGroup][kRowTile];
    __shared__ float input_scale_tile[kBatchTile];
    __shared__ float weight_scale_tile[kRowTile];

    const std::uint32_t local_row = threadIdx.x;
    const std::uint32_t local_batch = threadIdx.y;
    const std::uint32_t row = blockIdx.x * kRowTile + local_row;
    const std::uint32_t batch = blockIdx.y * kBatchTile + local_batch;
    const bool valid_row = row < rows;
    const bool valid_batch = batch < batch_rows;
    const std::uint32_t groups = cols / kGroupSize;
    const std::uint32_t packed_cols = cols / 4;
    float result = 0.0f;

    for (std::uint32_t group = 0; group < groups; ++group) {
        if (local_batch == 0) {
            if (valid_row) {
                const auto* row_weight =
                    weight + static_cast<std::size_t>(row) * packed_cols +
                    group * kPackedValuesPerGroup;
#pragma unroll
                for (std::uint32_t packed = 0; packed < kPackedValuesPerGroup; ++packed) {
                    weight_tile[packed][local_row] =
                        static_cast<int>(unpack_ternary(row_weight[packed]));
                }
                weight_scale_tile[local_row] =
                    weight_scales[static_cast<std::size_t>(row) * groups + group];
            } else {
#pragma unroll
                for (std::uint32_t packed = 0; packed < kPackedValuesPerGroup; ++packed) {
                    weight_tile[packed][local_row] = 0;
                }
                weight_scale_tile[local_row] = 0.0f;
            }
        }
        if (local_row < 4) {
            if (valid_batch) {
                const auto* source = reinterpret_cast<const int4*>(
                    input + static_cast<std::size_t>(batch) * cols + group * kGroupSize);
                auto* destination = reinterpret_cast<int4*>(
                    &input_tile[local_batch][local_row * 4]);
                *destination = source[local_row];
            } else {
                auto* destination = reinterpret_cast<int4*>(
                    &input_tile[local_batch][local_row * 4]);
                *destination = make_int4(0, 0, 0, 0);
            }
        }
        if (local_row == 0) {
            input_scale_tile[local_batch] = valid_batch
                ? input_scales[static_cast<std::size_t>(batch) * groups + group]
                : 0.0f;
        }
        __syncthreads();

        if (valid_row && valid_batch) {
            int sum = 0;
#pragma unroll
            for (std::uint32_t packed = 0; packed < kPackedValuesPerGroup; ++packed) {
                sum = __dp4a(weight_tile[packed][local_row],
                             input_tile[local_batch][packed], sum);
            }
            result += static_cast<float>(sum) * input_scale_tile[local_batch] *
                      weight_scale_tile[local_row];
        }
        __syncthreads();
    }
    if (valid_row && valid_batch) {
        output[static_cast<std::size_t>(batch) * rows + row] = result;
    }
}

__global__ void ternary_g64_lookup_rows_f32_kernel(
    const std::uint8_t* weight,
    const float* weight_scales,
    const std::uint32_t* row_indices,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t batch = blockIdx.x;
    const std::uint32_t row = row_indices[batch];
    if (row >= rows) {
        return;
    }
    const std::uint32_t groups = cols / kGroupSize;
    const auto* row_weight = weight + static_cast<std::size_t>(row) * (cols / 4);
    const auto* row_scales = weight_scales + static_cast<std::size_t>(row) * groups;
    auto* row_output = output + static_cast<std::size_t>(batch) * cols;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::uint8_t packed = row_weight[col / 4];
        const std::uint32_t code = (packed >> ((col % 4) * 2)) & 0x03u;
        row_output[col] = static_cast<float>(static_cast<int>(code) - 1) *
                          row_scales[col / kGroupSize];
    }
}

__global__ void ternary_g64_expand_bf16_kernel(
    const std::uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    std::uint16_t* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const std::uint32_t groups = cols / kGroupSize;
    const auto* row_weight = weight + static_cast<std::size_t>(row) * (cols / 4);
    const auto* row_scales = weight_scales + static_cast<std::size_t>(row) * groups;
    auto* row_output = output + static_cast<std::size_t>(row) * cols;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::uint8_t packed = row_weight[col / 4];
        const std::uint32_t code = (packed >> ((col % 4) * 2)) & 0x03u;
        const float value = static_cast<float>(static_cast<int>(code) - 1) *
                            row_scales[col / kGroupSize];
        row_output[col] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}

}  // namespace

extern "C" cudaError_t infer_ternary_g64_quantize_i8_f32_on_stream(
    const float* input,
    std::int8_t* output,
    float* dequant_scales,
    std::uint32_t batch_rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || dequant_scales == nullptr ||
        batch_rows == 0 || cols == 0 || (cols % kGroupSize) != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(batch_rows, cols / kGroupSize);
    ternary_g64_quantize_i8_f32_kernel<<<grid, kGroupSize, 0, stream>>>(
        input, output, dequant_scales, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_ternary_g64_w2a8_linear_f32_on_stream(
    const std::int8_t* input,
    const float* input_scales,
    const std::uint8_t* weight,
    const float* weight_scales,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || input_scales == nullptr || weight == nullptr ||
        weight_scales == nullptr || output == nullptr || batch_rows == 0 ||
        rows == 0 || cols == 0 || (cols % kGroupSize) != 0) {
        return cudaErrorInvalidValue;
    }
    if (batch_rows >= 4) {
        constexpr std::uint32_t kBatchTile = 8;
        constexpr std::uint32_t kRowTile = 32;
        const dim3 block(kRowTile, kBatchTile);
        const dim3 grid((rows + kRowTile - 1) / kRowTile,
                        (batch_rows + kBatchTile - 1) / kBatchTile);
        ternary_g64_w2a8_linear_tiled_f32_kernel<<<grid, block, 0, stream>>>(
            input, input_scales, weight, weight_scales, output, batch_rows, rows, cols);
    } else {
        constexpr std::uint32_t kWarpsPerBlock = 8;
        constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
        const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock, batch_rows);
        ternary_g64_w2a8_linear_f32_kernel<<<grid, kThreads, 0, stream>>>(
            input, input_scales, weight, weight_scales, output, rows, cols);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t infer_ternary_g64_lookup_rows_f32_on_stream(
    const std::uint8_t* weight,
    const float* weight_scales,
    const std::uint32_t* row_indices,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (weight == nullptr || weight_scales == nullptr || row_indices == nullptr ||
        output == nullptr || batch_rows == 0 || rows == 0 || cols == 0 ||
        (cols % kGroupSize) != 0) {
        return cudaErrorInvalidValue;
    }
    ternary_g64_lookup_rows_f32_kernel<<<batch_rows, 256, 0, stream>>>(
        weight, weight_scales, row_indices, output, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_ternary_g64_expand_bf16_on_stream(
    const std::uint8_t* weight,
    const float* weight_scales,
    std::uint16_t* output,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (weight == nullptr || weight_scales == nullptr || output == nullptr ||
        rows == 0 || cols == 0 || (cols % kGroupSize) != 0) {
        return cudaErrorInvalidValue;
    }
    ternary_g64_expand_bf16_kernel<<<rows, 256, 0, stream>>>(
        weight, weight_scales, output, rows, cols);
    return cudaGetLastError();
}
