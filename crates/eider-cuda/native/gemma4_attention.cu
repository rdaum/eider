#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

using Bf16 = __nv_bfloat16;

constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 256;
constexpr int kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr int kWindowTokens = 1024;
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kTile = 16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr float kSoftmaxScaleLog2 = 0.09016844005556021f;

constexpr int kQueryElements = kBlockM * kHeadDim;
constexpr int kKeyValueElements = kBlockN * kHeadDim;
constexpr int kBf16PerVector = sizeof(uint4) / sizeof(Bf16);
constexpr int kProbabilityElements = kBlockM * kBlockN;
constexpr int kStatisticElements = kWarps * kBlockM;
constexpr int kSharedBytes =
    (kQueryElements + kKeyValueElements + kProbabilityElements) * sizeof(Bf16) +
    kStatisticElements * sizeof(float);

__device__ __forceinline__ std::uint8_t packed_nibble(
    const std::uint8_t* packed,
    int index) {
    const std::uint8_t byte = packed[index >> 1];
    return static_cast<std::uint8_t>((index & 1) == 0 ? byte & 0x0f : byte >> 4);
}

__device__ __forceinline__ float e2m1_value(std::uint8_t code) {
    const std::uint32_t magnitude = code & 0x07U;
    const std::uint32_t exponent = magnitude >> 1U;
    const std::uint32_t mantissa = magnitude & 1U;
    const std::uint32_t magnitude_bits = exponent == 0
        ? mantissa * 0x3f000000U
        : ((exponent + 126U) << 23U) | (mantissa << 22U);
    const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x08U) << 28U;
    return __uint_as_float(sign | magnitude_bits);
}

__device__ __forceinline__ float e4m3_value(std::uint8_t code) {
    const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80) << 24;
    const std::uint32_t exponent = (code >> 3) & 0x0f;
    const std::uint32_t mantissa = code & 0x07;
    if (exponent == 0) {
        const float value = static_cast<float>(mantissa) * 0x1p-9f;
        return sign == 0 ? value : -value;
    }
    if (exponent == 0x0f && mantissa == 0x07) {
        return __uint_as_float(sign | 0x7fffffffU);
    }
    return __uint_as_float(sign | ((exponent + 120U) << 23) | (mantissa << 20));
}

__device__ __forceinline__ std::uint32_t pack_bf16_pair(float low, float high) {
    return static_cast<std::uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(low))) |
           (static_cast<std::uint32_t>(
                __bfloat16_as_ushort(__float2bfloat16_rn(high)))
            << 16);
}

struct Bf16Cache {
    const std::uint16_t* key;
    const std::uint16_t* value;
    int tokens;

    __device__ __forceinline__ uint4 load_key_vector(
        int head,
        int token,
        int dimension) const {
        const std::size_t offset =
            (static_cast<std::size_t>(head) * tokens + token) * kHeadDim + dimension;
        return *reinterpret_cast<const uint4*>(key + offset);
    }

    __device__ __forceinline__ Bf16 load_value(
        int head,
        int token,
        int dimension) const {
        const std::size_t offset =
            (static_cast<std::size_t>(head) * kHeadDim + dimension) * tokens + token;
        return __ushort_as_bfloat16(value[offset]);
    }

};

struct CompactCache {
    const std::uint8_t* key_values;
    const std::uint8_t* key_scales;
    const std::uint8_t* value_values;
    const std::uint8_t* value_scales;
    const float* key_tail;
    const float* value_tail;
    int tokens;
    int max_tokens;

    __device__ __forceinline__ float load_key(
        int head,
        int token,
        int dimension) const {
        const int compact_tokens = tokens / 8 * 8;
        if (token >= compact_tokens) {
            const int width = kKvHeads * kHeadDim;
            return key_tail[(token & 15) * width + head * kHeadDim + dimension];
        }
        const int token_tiles = (max_tokens + 7) / 8;
        const int dimension_tiles = kHeadDim / 64;
        const int token_tile = token / 8;
        const int token_in_tile = token & 7;
        const int dimension_tile = dimension / 64;
        const int dimension_in_tile = dimension & 63;
        const int tile =
            (head * token_tiles + token_tile) * dimension_tiles + dimension_tile;
        const std::uint8_t code = packed_nibble(
            key_values + tile * 256,
            token_in_tile * 64 + dimension_in_tile);
        const std::uint8_t scale =
            key_scales[(tile * 8 + token_in_tile) * 4 + dimension_in_tile / 16];
        return e2m1_value(code) * e4m3_value(scale);
    }

    __device__ __forceinline__ uint4 load_key_vector(
        int head,
        int token,
        int dimension) const {
        const int compact_tokens = tokens / 8 * 8;
        if (token < compact_tokens) {
            const int token_tiles = (max_tokens + 7) / 8;
            const int dimension_tiles = kHeadDim / 64;
            const int token_tile = token / 8;
            const int token_in_tile = token & 7;
            const int dimension_tile = dimension / 64;
            const int dimension_in_tile = dimension & 63;
            const int tile =
                (head * token_tiles + token_tile) * dimension_tiles + dimension_tile;
            const int element = token_in_tile * 64 + dimension_in_tile;
            const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
                key_values + tile * 256 + element / 2);
            const float scale = e4m3_value(
                key_scales[(tile * 8 + token_in_tile) * 4 + dimension_in_tile / 16]);
            const auto pair = [&](int shift) {
                return pack_bf16_pair(
                    e2m1_value((packed >> shift) & 0x0f) * scale,
                    e2m1_value((packed >> (shift + 4)) & 0x0f) * scale);
            };
            return make_uint4(pair(0), pair(8), pair(16), pair(24));
        }
        uint4 result;
        result.x = pack_bf16_pair(
            load_key(head, token, dimension),
            load_key(head, token, dimension + 1));
        result.y = pack_bf16_pair(
            load_key(head, token, dimension + 2),
            load_key(head, token, dimension + 3));
        result.z = pack_bf16_pair(
            load_key(head, token, dimension + 4),
            load_key(head, token, dimension + 5));
        result.w = pack_bf16_pair(
            load_key(head, token, dimension + 6),
            load_key(head, token, dimension + 7));
        return result;
    }

    __device__ __forceinline__ Bf16 load_value(
        int head,
        int token,
        int dimension) const {
        const int compact_tokens = tokens / 16 * 16;
        float value;
        if (token >= compact_tokens) {
            const int width = kKvHeads * kHeadDim;
            value = value_tail[(token & 15) * width + head * kHeadDim + dimension];
        } else {
            const int context_tiles = (max_tokens + 63) / 64;
            const int dimension_tile = dimension / 8;
            const int dimension_in_tile = dimension & 7;
            const int token_tile = token / 64;
            const int token_in_tile = token & 63;
            const int tile =
                (head * (kHeadDim / 8) + dimension_tile) * context_tiles + token_tile;
            const std::uint8_t code = packed_nibble(
                value_values + tile * 256,
                dimension_in_tile * 64 + token_in_tile);
            const std::uint8_t scale =
                value_scales[(tile * 8 + dimension_in_tile) * 4 + token_in_tile / 16];
            value = e2m1_value(code) * e4m3_value(scale);
        }
        return __float2bfloat16_rn(value);
    }

};

__device__ __forceinline__ float subgroup_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
    return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
}

__device__ __forceinline__ float subgroup_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value + __shfl_xor_sync(0xffffffff, value, 2);
}

__device__ __forceinline__ void accumulator_coordinate(int item, int& row, int& col) {
    const int lane = threadIdx.x % 32;
    row = lane / 4 + ((item & 2) != 0 ? 8 : 0);
    col = (lane % 4) * 2 + (item & 1);
}

struct MatrixA {
    std::uint32_t x[4];
};

struct MatrixBPair {
    std::uint32_t x[4];
};

struct Accumulator {
    float x[4];
};

__device__ __forceinline__ std::uint32_t shared_address(const void* pointer) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ int swizzled_column(int row, int column) {
    return column ^ ((row & 7) * 8);
}

__device__ __forceinline__ MatrixA load_matrix_a(const Bf16* base,
                                                 int row,
                                                 int column,
                                                 int stride) {
    const int lane = threadIdx.x % 32;
    const int lane_row = row + lane % 16;
    const int lane_column = column + (lane / 16) * 8;
    const Bf16* lane_base =
        base + lane_row * stride + swizzled_column(lane_row, lane_column);
    MatrixA fragment;
    const std::uint32_t address = shared_address(lane_base);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment.x[0]), "=r"(fragment.x[1]), "=r"(fragment.x[2]),
          "=r"(fragment.x[3])
        : "r"(address));
    return fragment;
}

__device__ __forceinline__ MatrixBPair load_matrix_b_32x8(const Bf16* base,
                                                           int row,
                                                           int column,
                                                           int stride) {
    const int lane = threadIdx.x % 32;
    const int lane_row = column + lane % 8;
    const int lane_column = row + (lane / 8) * 8;
    const Bf16* lane_base =
        base + lane_row * stride + swizzled_column(lane_row, lane_column);
    MatrixBPair fragment;
    const std::uint32_t address = shared_address(lane_base);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment.x[0]), "=r"(fragment.x[1]), "=r"(fragment.x[2]),
          "=r"(fragment.x[3])
        : "r"(address));
    return fragment;
}

__device__ __forceinline__ void clear(Accumulator& accumulator) {
    accumulator.x[0] = 0.0f;
    accumulator.x[1] = 0.0f;
    accumulator.x[2] = 0.0f;
    accumulator.x[3] = 0.0f;
}

__device__ __forceinline__ void mma(Accumulator& accumulator,
                                    const MatrixA& a,
                                    const MatrixBPair& b,
                                    int half) {
    const int offset = half * 2;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(accumulator.x[0]), "+f"(accumulator.x[1]),
          "+f"(accumulator.x[2]), "+f"(accumulator.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(b.x[offset]), "r"(b.x[offset + 1]));
}

template <typename Cache>
__global__ void gemma4_local_attention_kernel(
    const std::uint16_t* __restrict__ query,
    Cache cache,
    std::uint16_t* __restrict__ output,
    int query_tokens,
    int start_position) {
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* shared_query = reinterpret_cast<Bf16*>(shared_storage);
    auto* shared_key_value = shared_query + kQueryElements;
    auto* shared_probabilities = shared_key_value + kKeyValueElements;
    auto* shared_statistics = reinterpret_cast<float*>(
        shared_probabilities + kProbabilityElements);

    const int query_block = blockIdx.z;
    const int query_head = blockIdx.y;
    const int key_head = query_head / kQueriesPerKv;
    const int query_start = query_block * kBlockM;
    const int block_query_start = start_position + query_start;
    const int query_rows = min(kBlockM, query_tokens - query_start);
    const int key_start = max(0, block_query_start + 1 - kWindowTokens);
    const int key_end = min(cache.tokens, block_query_start + query_rows);

    const std::size_t query_head_offset =
        static_cast<std::size_t>(query_head) * query_tokens * kHeadDim;
#pragma unroll
    for (int chunk = threadIdx.x;
         chunk < kQueryElements / kBf16PerVector;
         chunk += kThreads) {
        const int row = chunk / (kHeadDim / kBf16PerVector);
        const int dimension = (chunk % (kHeadDim / kBf16PerVector)) * kBf16PerVector;
        const uint4 values = row < query_rows
            ? *reinterpret_cast<const uint4*>(
                  reinterpret_cast<const Bf16*>(query) + query_head_offset +
                  static_cast<std::size_t>(query_start + row) * kHeadDim + dimension)
            : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(
            shared_query + row * kHeadDim + swizzled_column(row, dimension)) = values;
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int lane_row = lane / 4;
    constexpr int kRowsPerLane = kBlockM / 8;
    float row_max[kRowsPerLane];
    float row_sum[kRowsPerLane];
#pragma unroll
    for (int slot = 0; slot < kRowsPerLane; ++slot) {
        row_max[slot] = -INFINITY;
        row_sum[slot] = 0.0f;
    }

    constexpr int kOutputFragmentsPerRowTile = 4;
    constexpr int kRowTiles = kBlockM / kTile;
    constexpr int kOutputFragmentsPerWarp =
        kRowTiles * kOutputFragmentsPerRowTile;
    Accumulator output_fragments[kOutputFragmentsPerWarp];
#pragma unroll
    for (int fragment = 0; fragment < kOutputFragmentsPerWarp; ++fragment) {
        clear(output_fragments[fragment]);
    }

    for (int key_tile_start = key_start; key_tile_start < key_end;
         key_tile_start += kBlockN) {
        const int key_rows = min(kBlockN, key_end - key_tile_start);
#pragma unroll
        for (int chunk = threadIdx.x;
             chunk < kKeyValueElements / kBf16PerVector;
             chunk += kThreads) {
            const int key_row = chunk / (kHeadDim / kBf16PerVector);
            const int dimension =
                (chunk % (kHeadDim / kBf16PerVector)) * kBf16PerVector;
            const uint4 values = key_row < key_rows
                ? cache.load_key_vector(key_head, key_tile_start + key_row, dimension)
                : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(
                shared_key_value + key_row * kHeadDim +
                swizzled_column(key_row, dimension)) = values;
        }
        __syncthreads();

        Accumulator score_fragments[kRowTiles];
#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
            clear(score_fragments[row_tile]);
        }
#pragma unroll
        for (int dimension = 0; dimension < kHeadDim; dimension += 32) {
            const MatrixBPair key_fragment = load_matrix_b_32x8(
                shared_key_value, dimension, warp * 8, kHeadDim);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
                    const MatrixA query_fragment = load_matrix_a(
                        shared_query,
                        row_tile * kTile,
                        dimension + half * kTile,
                        kHeadDim);
                    mma(score_fragments[row_tile], query_fragment, key_fragment, half);
                }
            }
        }

        float local_max[kRowsPerLane];
#pragma unroll
        for (int slot = 0; slot < kRowsPerLane; ++slot) {
            local_max[slot] = -INFINITY;
        }
#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                int local_row;
                int local_col;
                accumulator_coordinate(item, local_row, local_col);
                const int query_row = row_tile * kTile + local_row;
                const int key_column = warp * 8 + local_col;
                const int absolute_query = block_query_start + query_row;
                const int absolute_key = key_tile_start + key_column;
                const bool valid = query_row < query_rows && key_column < key_rows &&
                                   absolute_key <= absolute_query &&
                                   absolute_query - absolute_key < kWindowTokens;
                const float score = valid
                    ? score_fragments[row_tile].x[item] * kSoftmaxScaleLog2
                    : -1.0e8f;
                score_fragments[row_tile].x[item] = score;
                const int slot = row_tile * 2 + ((item & 2) != 0);
                local_max[slot] = fmaxf(local_max[slot], score);
            }
        }
#pragma unroll
        for (int slot = 0; slot < kRowsPerLane; ++slot) {
            local_max[slot] = subgroup_max(local_max[slot]);
            if ((lane & 3) == 0) {
                const int row_tile = slot / 2;
                const int row = row_tile * kTile + lane_row + (slot & 1) * 8;
                shared_statistics[warp * kBlockM + row] = local_max[slot];
            }
        }
        __syncthreads();

        const int reduction_row = threadIdx.x / 4;
        const int reduction_lane = threadIdx.x & 3;
        float reduced_max = fmaxf(
            shared_statistics[(reduction_lane * 2) * kBlockM + reduction_row],
            shared_statistics[(reduction_lane * 2 + 1) * kBlockM + reduction_row]);
        reduced_max = subgroup_max(reduced_max);
        if (reduction_lane == 0) {
            shared_statistics[reduction_row] = reduced_max;
        }
        __syncthreads();

        float next_max[kRowsPerLane];
        float correction[kRowsPerLane];
#pragma unroll
        for (int slot = 0; slot < kRowsPerLane; ++slot) {
            const int row_tile = slot / 2;
            const int row = row_tile * kTile + lane_row + (slot & 1) * 8;
            const float block_max = shared_statistics[row];
            next_max[slot] = fmaxf(row_max[slot], block_max);
            correction[slot] = exp2f(row_max[slot] - next_max[slot]);
        }

        float local_sum[kRowsPerLane] = {};
#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                int local_row;
                int local_col;
                accumulator_coordinate(item, local_row, local_col);
                const int query_row = row_tile * kTile + local_row;
                const int key_column = warp * 8 + local_col;
                const int slot = row_tile * 2 + ((item & 2) != 0);
                const float score = score_fragments[row_tile].x[item];
                const float probability = exp2f(score - next_max[slot]);
                shared_probabilities[
                    query_row * kBlockN + swizzled_column(query_row, key_column)] =
                        __float2bfloat16_rn(probability);
                local_sum[slot] += probability;
            }
        }
#pragma unroll
        for (int slot = 0; slot < kRowsPerLane; ++slot) {
            local_sum[slot] = subgroup_sum(local_sum[slot]);
            if ((lane & 3) == 0) {
                const int row_tile = slot / 2;
                const int row = row_tile * kTile + lane_row + (slot & 1) * 8;
                shared_statistics[warp * kBlockM + row] = local_sum[slot];
            }
        }
        __syncthreads();

        float reduced_sum =
            shared_statistics[(reduction_lane * 2) * kBlockM + reduction_row] +
            shared_statistics[(reduction_lane * 2 + 1) * kBlockM + reduction_row];
        reduced_sum = subgroup_sum(reduced_sum);
        if (reduction_lane == 0) {
            shared_statistics[reduction_row] = reduced_sum;
        }
        __syncthreads();

#pragma unroll
        for (int slot = 0; slot < kRowsPerLane; ++slot) {
            const int row_tile = slot / 2;
            const int row = row_tile * kTile + lane_row + (slot & 1) * 8;
            const float block_sum = shared_statistics[row];
            row_sum[slot] = row_sum[slot] * correction[slot] + block_sum;
            row_max[slot] = next_max[slot];
        }

#pragma unroll
        for (int fragment = 0; fragment < kOutputFragmentsPerWarp; ++fragment) {
            const int row_tile = fragment / kOutputFragmentsPerRowTile;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int slot = row_tile * 2 + ((item & 2) != 0);
                output_fragments[fragment].x[item] *= correction[slot];
            }
        }

#pragma unroll
        for (int index = threadIdx.x; index < kKeyValueElements; index += kThreads) {
            const int dimension = index / kBlockN;
            const int key_row = index % kBlockN;
            shared_key_value[
                dimension * kBlockN + swizzled_column(dimension, key_row)] =
                key_row < key_rows
                ? cache.load_value(key_head, key_tile_start + key_row, dimension)
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (int key_offset = 0; key_offset < kBlockN; key_offset += 32) {
            MatrixA probability_fragments[kRowTiles][2];
#pragma unroll
            for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    probability_fragments[row_tile][half] = load_matrix_a(
                        shared_probabilities,
                        row_tile * kTile,
                        key_offset + half * kTile,
                        kBlockN);
                }
            }
            MatrixBPair value_fragments[kOutputFragmentsPerRowTile];
#pragma unroll
            for (int dimension_group = 0;
                 dimension_group < kOutputFragmentsPerRowTile;
                 ++dimension_group) {
                value_fragments[dimension_group] = load_matrix_b_32x8(
                    shared_key_value,
                    key_offset,
                    warp * 8 + dimension_group * 64,
                    kBlockN);
            }
#pragma unroll
            for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
                for (int dimension_group = 0;
                     dimension_group < kOutputFragmentsPerRowTile;
                     ++dimension_group) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int fragment =
                            row_tile * kOutputFragmentsPerRowTile + dimension_group;
                        mma(
                            output_fragments[fragment],
                            probability_fragments[row_tile][half],
                            value_fragments[dimension_group],
                            half);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int fragment = 0; fragment < kOutputFragmentsPerWarp; ++fragment) {
        const int row_tile = fragment / kOutputFragmentsPerRowTile;
        const int dimension_tile = fragment % kOutputFragmentsPerRowTile;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            int local_row;
            int local_col;
            accumulator_coordinate(item, local_row, local_col);
            const int query_row = row_tile * kTile + local_row;
            if (query_row < query_rows) {
                const int slot = row_tile * 2 + ((item & 2) != 0);
                reinterpret_cast<Bf16*>(output)[
                    query_head_offset +
                    static_cast<std::size_t>(query_start + query_row) * kHeadDim +
                    warp * 8 + dimension_tile * 64 + local_col] =
                    __float2bfloat16_rn(output_fragments[fragment].x[item] / row_sum[slot]);
            }
        }
    }
}

}  // namespace

extern "C" cudaError_t infer_gemma4_local_attention_bf16_on_stream(
    const std::uint16_t* query,
    const std::uint16_t* key,
    const std::uint16_t* value,
    std::uint16_t* output,
    std::uint32_t query_tokens,
    std::uint32_t key_tokens,
    std::uint32_t start_position,
    cudaStream_t stream) {
    static const cudaError_t shared_memory_status = cudaFuncSetAttribute(
        gemma4_local_attention_kernel<Bf16Cache>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes);
    if (shared_memory_status != cudaSuccess) {
        return shared_memory_status;
    }
    const dim3 grid(1, kQueryHeads, (query_tokens + kBlockM - 1) / kBlockM);
    const Bf16Cache cache = {key, value, static_cast<int>(key_tokens)};
    gemma4_local_attention_kernel<Bf16Cache><<<grid, kThreads, kSharedBytes, stream>>>(
        query,
        cache,
        output,
        static_cast<int>(query_tokens),
        static_cast<int>(start_position));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_gemma4_local_attention_compact_on_stream(
    const std::uint16_t* query,
    const std::uint8_t* key_values,
    const std::uint8_t* key_scales,
    const std::uint8_t* value_values,
    const std::uint8_t* value_scales,
    const float* key_tail,
    const float* value_tail,
    std::uint16_t* output,
    std::uint32_t query_tokens,
    std::uint32_t cache_tokens,
    std::uint32_t cache_capacity,
    std::uint32_t start_position,
    cudaStream_t stream) {
    static const cudaError_t shared_memory_status = cudaFuncSetAttribute(
        gemma4_local_attention_kernel<CompactCache>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes);
    if (shared_memory_status != cudaSuccess) {
        return shared_memory_status;
    }
    const dim3 grid(1, kQueryHeads, (query_tokens + kBlockM - 1) / kBlockM);
    const CompactCache cache = {
        key_values,
        key_scales,
        value_values,
        value_scales,
        key_tail,
        value_tail,
        static_cast<int>(cache_tokens),
        static_cast<int>(cache_capacity),
    };
    gemma4_local_attention_kernel<CompactCache><<<grid, kThreads, kSharedBytes, stream>>>(
        query,
        cache,
        output,
        static_cast<int>(query_tokens),
        static_cast<int>(start_position));
    return cudaGetLastError();
}
