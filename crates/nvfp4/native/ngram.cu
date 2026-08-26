#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::size_t kMaxSharedEmbeddingBytes = 48 * 1024;

__device__ __forceinline__ float e4m3_value(std::uint8_t code) {
    const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80) << 24;
    const std::uint32_t exp = (code >> 3) & 0x0f;
    const std::uint32_t mant = code & 0x07;
    if (exp == 0) {
        const float value = static_cast<float>(mant) * 0x1p-9f;
        return sign == 0 ? value : -value;
    }
    if (exp == 0x0f && mant == 0x07) {
        return 0.0f;
    }
    return __uint_as_float(sign | ((exp + 120U) << 23) | (mant << 20));
}

__device__ __forceinline__ float e2m1_value(std::uint8_t nibble) {
    const std::uint32_t magnitude = nibble & 0x7u;
    const std::uint32_t exponent = magnitude >> 1u;
    const std::uint32_t mantissa = magnitude & 1u;
    const std::uint32_t magnitude_bits = exponent == 0
        ? mantissa * 0x3f000000u
        : ((exponent + 126u) << 23u) | (mantissa << 22u);
    const std::uint32_t sign_bit = static_cast<std::uint32_t>(nibble & 0x8u) << 28u;
    return __uint_as_float(sign_bit | magnitude_bits);
}

struct Bf16Rows {
    const std::uint16_t* values;
    std::uint32_t rows;

    __device__ __forceinline__ float load(std::uint32_t row,
                                          std::uint32_t col,
                                          std::uint32_t cols) const {
        if (row >= rows) return 0.0f;
        const std::uint16_t raw = values[static_cast<std::size_t>(row) * cols + col];
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&raw));
    }
};

struct Fp8Rows {
    const std::uint8_t* values;
    const float* row_scales;
    std::uint32_t rows;

    __device__ __forceinline__ float load(std::uint32_t row,
                                          std::uint32_t col,
                                          std::uint32_t cols) const {
        if (row >= rows) return 0.0f;
        return e4m3_value(values[static_cast<std::size_t>(row) * cols + col])
            * row_scales[row];
    }
};

struct Nvfp4Rows {
    const std::uint8_t* packed_values;
    const std::uint8_t* scales;
    std::uint32_t rows;

    __device__ __forceinline__ float load(std::uint32_t row,
                                          std::uint32_t col,
                                          std::uint32_t cols) const {
        if (row >= rows) return 0.0f;
        const std::size_t packed_row = static_cast<std::size_t>(row) * (cols / 2);
        const std::uint8_t pair = packed_values[packed_row + col / 2];
        const std::uint8_t code = (col & 1U) == 0 ? pair & 0x0f : pair >> 4;
        const std::size_t scale_row = static_cast<std::size_t>(row) * (cols / 16);
        return e2m1_value(code) * e4m3_value(scales[scale_row + col / 16]);
    }
};

template <typename Rows>
__global__ void gather_rows_kernel(Rows bank,
                                   const std::uint32_t* __restrict__ row_ids,
                                   float* __restrict__ output,
                                   std::uint32_t row_count,
                                   std::uint32_t cols) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t values = row_count * cols;
    if (index >= values) {
        return;
    }
    const std::uint32_t output_row = index / cols;
    const std::uint32_t col = index % cols;
    output[index] = bank.load(row_ids[output_row], col, cols);
}

template <typename Rows>
__global__ void fused_embedding_kernel(
    Rows bank,
    const float* __restrict__ word_embeddings,
    const std::uint32_t* __restrict__ row_ids,
    const std::uint16_t* __restrict__ projections,
    float* __restrict__ output,
    std::uint32_t table_count,
    std::uint32_t embedding_dim,
    std::uint32_t hidden_dim) {
    const std::uint32_t token_row = blockIdx.x;
    const std::uint32_t embedding_values = table_count * embedding_dim;
    extern __shared__ float selected_embeddings[];

    for (std::uint32_t flat = threadIdx.x; flat < embedding_values;
         flat += blockDim.x) {
        const std::uint32_t table = flat / embedding_dim;
        const std::uint32_t col = flat % embedding_dim;
        const std::uint32_t row = row_ids[
            static_cast<std::size_t>(token_row) * table_count + table];
        selected_embeddings[flat] = bank.load(row, col, embedding_dim);
    }
    __syncthreads();

    const float inverse_sources = 1.0f / static_cast<float>(table_count + 1);
    for (std::uint32_t hidden = threadIdx.x; hidden < hidden_dim;
         hidden += blockDim.x) {
        float value = word_embeddings[
            static_cast<std::size_t>(token_row) * hidden_dim + hidden];
        for (std::uint32_t flat = 0; flat < embedding_values; ++flat) {
            const std::uint16_t raw = projections[
                static_cast<std::size_t>(flat) * hidden_dim + hidden];
            const float projection =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&raw));
            value = fmaf(selected_embeddings[flat], projection, value);
        }
        output[static_cast<std::size_t>(token_row) * hidden_dim + hidden] =
            value * inverse_sources;
    }
}

template <typename Rows>
cudaError_t launch_gather(Rows bank,
                          const std::uint32_t* row_ids,
                          float* output,
                          std::uint32_t row_count,
                          std::uint32_t cols,
                          cudaStream_t stream) {
    if (row_ids == nullptr || output == nullptr || row_count == 0 || cols == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint64_t values =
        static_cast<std::uint64_t>(row_count) * static_cast<std::uint64_t>(cols);
    if (values > static_cast<std::uint64_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t blocks =
        (static_cast<std::uint32_t>(values) + kThreads - 1) / kThreads;
    gather_rows_kernel<<<blocks, kThreads, 0, stream>>>(
        bank, row_ids, output, row_count, cols);
    return cudaGetLastError();
}

template <typename Rows>
cudaError_t launch_fused(Rows bank,
                         const float* word_embeddings,
                         const std::uint32_t* row_ids,
                         const std::uint16_t* projections,
                         float* output,
                         std::uint32_t token_rows,
                         std::uint32_t table_count,
                         std::uint32_t embedding_dim,
                         std::uint32_t hidden_dim,
                         cudaStream_t stream) {
    if (word_embeddings == nullptr || row_ids == nullptr || projections == nullptr
        || output == nullptr || token_rows == 0 || table_count == 0
        || embedding_dim == 0 || hidden_dim == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t shared_bytes = static_cast<std::size_t>(table_count)
        * embedding_dim * sizeof(float);
    if (shared_bytes > kMaxSharedEmbeddingBytes) {
        return cudaErrorInvalidValue;
    }
    fused_embedding_kernel<<<token_rows, kThreads, shared_bytes, stream>>>(
        bank, word_embeddings, row_ids, projections, output, table_count,
        embedding_dim, hidden_dim);
    return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t infer_ngram_gather_bf16_on_stream(
    const std::uint16_t* values,
    std::uint32_t bank_rows,
    const std::uint32_t* row_ids,
    float* output,
    std::uint32_t row_count,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (values == nullptr || bank_rows == 0) {
        return cudaErrorInvalidValue;
    }
    return launch_gather(Bf16Rows{values, bank_rows}, row_ids, output, row_count, cols, stream);
}

extern "C" cudaError_t infer_ngram_gather_fp8_on_stream(
    const std::uint8_t* values,
    const float* row_scales,
    std::uint32_t bank_rows,
    const std::uint32_t* row_ids,
    float* output,
    std::uint32_t row_count,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (values == nullptr || row_scales == nullptr || bank_rows == 0) {
        return cudaErrorInvalidValue;
    }
    return launch_gather(
        Fp8Rows{values, row_scales, bank_rows}, row_ids, output, row_count, cols, stream);
}

extern "C" cudaError_t infer_ngram_gather_nvfp4_on_stream(
    const std::uint8_t* packed_values,
    const std::uint8_t* scales,
    std::uint32_t bank_rows,
    const std::uint32_t* row_ids,
    float* output,
    std::uint32_t row_count,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (packed_values == nullptr || scales == nullptr || bank_rows == 0
        || !cols || cols % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    return launch_gather(
        Nvfp4Rows{packed_values, scales, bank_rows}, row_ids, output, row_count, cols, stream);
}

extern "C" cudaError_t infer_ngram_fused_bf16_on_stream(
    const std::uint16_t* values,
    std::uint32_t bank_rows,
    const float* word_embeddings,
    const std::uint32_t* row_ids,
    const std::uint16_t* projections,
    float* output,
    std::uint32_t token_rows,
    std::uint32_t table_count,
    std::uint32_t embedding_dim,
    std::uint32_t hidden_dim,
    cudaStream_t stream) {
    if (values == nullptr || bank_rows == 0) {
        return cudaErrorInvalidValue;
    }
    return launch_fused(
        Bf16Rows{values, bank_rows}, word_embeddings, row_ids, projections, output,
        token_rows, table_count, embedding_dim, hidden_dim, stream);
}

extern "C" cudaError_t infer_ngram_fused_fp8_on_stream(
    const std::uint8_t* values,
    const float* row_scales,
    std::uint32_t bank_rows,
    const float* word_embeddings,
    const std::uint32_t* row_ids,
    const std::uint16_t* projections,
    float* output,
    std::uint32_t token_rows,
    std::uint32_t table_count,
    std::uint32_t embedding_dim,
    std::uint32_t hidden_dim,
    cudaStream_t stream) {
    if (values == nullptr || row_scales == nullptr || bank_rows == 0) {
        return cudaErrorInvalidValue;
    }
    return launch_fused(
        Fp8Rows{values, row_scales, bank_rows}, word_embeddings, row_ids, projections,
        output, token_rows, table_count, embedding_dim, hidden_dim, stream);
}

extern "C" cudaError_t infer_ngram_fused_nvfp4_on_stream(
    const std::uint8_t* packed_values,
    const std::uint8_t* scales,
    std::uint32_t bank_rows,
    const float* word_embeddings,
    const std::uint32_t* row_ids,
    const std::uint16_t* projections,
    float* output,
    std::uint32_t token_rows,
    std::uint32_t table_count,
    std::uint32_t embedding_dim,
    std::uint32_t hidden_dim,
    cudaStream_t stream) {
    if (packed_values == nullptr || scales == nullptr || bank_rows == 0 || !embedding_dim
        || embedding_dim % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    return launch_fused(
        Nvfp4Rows{packed_values, scales, bank_rows}, word_embeddings, row_ids, projections,
        output, token_rows, table_count, embedding_dim, hidden_dim, stream);
}
