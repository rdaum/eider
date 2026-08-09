#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void bitnet_quantize_i8_f32_kernel(const float* input,
                                               std::int8_t* output,
                                               float* dequant_scales,
                                               std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    const float* row_input = input + static_cast<std::size_t>(row) * cols;
    std::int8_t* row_output = output + static_cast<std::size_t>(row) * cols;

    float maximum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        maximum = fmaxf(maximum, fabsf(row_input[col]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    __shared__ float warp_maximum[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_maximum[warp] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 8 ? warp_maximum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
        }
        if (lane == 0) {
            warp_maximum[0] = maximum;
            dequant_scales[row] = maximum / 127.0f;
        }
    }
    __syncthreads();

    const float scale = warp_maximum[0] == 0.0f ? 0.0f : 127.0f / warp_maximum[0];
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const int quantized = __float2int_rn(row_input[col] * scale);
        row_output[col] = static_cast<std::int8_t>(max(-127, min(127, quantized)));
    }
}

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

__global__ void bitnet_w2a8_linear_f32_kernel(const std::int8_t* input,
                                               const float* input_scales,
                                               const std::uint8_t* weight,
                                               const float* weight_scales,
                                               float* output,
                                               std::uint32_t rows,
                                               std::uint32_t cols) {
    constexpr std::uint32_t kWarpsPerBlock = 8;
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) {
        return;
    }
    const std::uint32_t batch = blockIdx.y;
    const auto* input4 = reinterpret_cast<const int*>(
        input + static_cast<std::size_t>(batch) * cols);
    const auto* row_weight = weight + static_cast<std::size_t>(row) * (cols / 4);

    int sum = 0;
    for (std::uint32_t packed_col = lane; packed_col < cols / 4; packed_col += 32) {
        const int packed_activation = input4[packed_col];
        const int packed_weight = static_cast<int>(unpack_ternary(row_weight[packed_col]));
        sum = __dp4a(packed_weight, packed_activation, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        output[static_cast<std::size_t>(batch) * rows + row] =
            static_cast<float>(sum) * input_scales[batch] * weight_scales[row];
    }
}

__global__ void bitnet_relu_squared_mul_halves_f32_kernel(
    const float* input,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t cols) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = batch_rows * cols;
    if (index >= total) {
        return;
    }
    const std::uint32_t row = index / cols;
    const std::uint32_t col = index - row * cols;
    const std::size_t base = static_cast<std::size_t>(row) * cols * 2;
    const float gate = fmaxf(input[base + col], 0.0f);
    output[index] = gate * gate * input[base + cols + col];
}

}  // namespace

extern "C" cudaError_t infer_bitnet_quantize_i8_f32_on_stream(
    const float* input,
    std::int8_t* output,
    float* dequant_scales,
    std::uint32_t batch_rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || dequant_scales == nullptr ||
        batch_rows == 0 || cols == 0 || (cols % 4) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    bitnet_quantize_i8_f32_kernel<<<batch_rows, kThreads, 0, stream>>>(
        input, output, dequant_scales, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bitnet_w2a8_linear_f32_on_stream(
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
        rows == 0 || cols == 0 || (cols % 4) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kWarpsPerBlock = 8;
    constexpr std::uint32_t kThreads = kWarpsPerBlock * 32;
    const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock, batch_rows);
    bitnet_w2a8_linear_f32_kernel<<<grid, kThreads, 0, stream>>>(
        input, input_scales, weight, weight_scales, output, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_bitnet_relu_squared_mul_halves_f32_on_stream(
    const float* input,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || batch_rows == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t total = batch_rows * cols;
    const std::uint32_t blocks = (total + kThreads - 1) / kThreads;
    bitnet_relu_squared_mul_halves_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        input, output, batch_rows, cols);
    return cudaGetLastError();
}
