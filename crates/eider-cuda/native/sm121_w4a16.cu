#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr std::uint32_t kTileM = 16;
constexpr std::uint32_t kTileK = 16;
constexpr std::uint32_t kPackedTileBytes = kTileM * kTileK / 2;
constexpr std::uint32_t kScaleTileBytes = kTileM;

__device__ __forceinline__ std::uint32_t pack_bf16_pair(float low, float high) {
    return static_cast<std::uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(low)))
        | (static_cast<std::uint32_t>(
               __bfloat16_as_ushort(__float2bfloat16_rn(high)))
           << 16u);
}

__device__ __forceinline__ float e2m1_value(std::uint8_t code) {
    const std::uint32_t magnitude = code & 0x7u;
    const std::uint32_t exponent = magnitude >> 1u;
    const std::uint32_t mantissa = magnitude & 1u;
    const std::uint32_t magnitude_bits = exponent == 0
        ? mantissa * 0x3f000000u
        : ((exponent + 126u) << 23u) | (mantissa << 22u);
    const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x8u) << 28u;
    return __uint_as_float(sign | magnitude_bits);
}

__device__ __forceinline__ float e4m3_value(std::uint8_t code) {
    const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80u) << 24u;
    const std::uint32_t exponent = (code >> 3u) & 0x0fu;
    const std::uint32_t mantissa = code & 0x07u;
    if (exponent == 0) {
        const float value = static_cast<float>(mantissa) * 0x1p-9f;
        return sign == 0 ? value : -value;
    }
    if (exponent == 0x0f && mantissa == 0x07) {
        return __uint_as_float(sign | 0x7fffffffU);
    }
    return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
}

__device__ __forceinline__ std::uint32_t dequant_pair(
    std::uint8_t packed,
    float scale) {
    return pack_bf16_pair(
        e2m1_value(packed & 0x0fu) * scale,
        e2m1_value(packed >> 4u) * scale);
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    std::uint32_t a0,
    std::uint32_t a1,
    std::uint32_t a2,
    std::uint32_t a3,
    std::uint32_t b0,
    std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool WriteF32, std::uint32_t WarpsPerBlock, std::uint32_t FixedTopK>
__global__ __launch_bounds__(WarpsPerBlock * 32) void routed_w4a16_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ tiled_weight,
    const std::uint8_t* __restrict__ tiled_scales,
    const float* __restrict__ global_scales,
    __nv_bfloat16* __restrict__ output_bf16,
    float* __restrict__ output_f32,
    std::uint32_t batch_size,
    std::uint32_t top_k,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t route = blockIdx.y;
    const std::uint32_t out_tile = blockIdx.x;
    const std::uint32_t routes = FixedTopK == 0 ? batch_size * top_k : FixedTopK;
    if (route >= routes || out_tile * kTileM >= out_features) {
        return;
    }

    const std::uint32_t expert = indices[route];
    const std::uint32_t input_row = FixedTopK == 0 ? route / top_k : 0;
    const std::uint32_t k_tiles = in_features / kTileK;
    const std::size_t expert_weight_stride =
        static_cast<std::size_t>(out_features) * in_features / 2;
    const std::size_t expert_scale_stride =
        static_cast<std::size_t>(out_features) * in_features / kTileK;
    const std::uint8_t* expert_weight =
        tiled_weight + static_cast<std::size_t>(expert) * expert_weight_stride;
    const std::uint8_t* expert_scales =
        tiled_scales + static_cast<std::size_t>(expert) * expert_scale_stride;
    const float global_scale = global_scales[expert];

    const std::uint32_t row0 = lane >> 2u;
    const std::uint32_t row1 = row0 + 8u;
    const std::uint32_t pair0 = (lane & 3u) * 2u;
    const std::uint32_t pair1 = pair0 + 8u;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (std::uint32_t k_tile = warp; k_tile < k_tiles; k_tile += WarpsPerBlock) {
        const std::size_t tile =
            static_cast<std::size_t>(out_tile) * k_tiles + k_tile;
        const std::uint8_t* tile_weight =
            expert_weight + tile * kPackedTileBytes;
        const std::uint8_t* tile_scales =
            expert_scales + tile * kScaleTileBytes;

        float scale0 = 0.0f;
        float scale1 = 0.0f;
        if ((lane & 3u) == 0) {
            scale0 = e4m3_value(tile_scales[row0]) * global_scale;
            scale1 = e4m3_value(tile_scales[row1]) * global_scale;
        }
        scale0 = __shfl_sync(0xffffffffu, scale0, row0 * 4u);
        scale1 = __shfl_sync(0xffffffffu, scale1, row0 * 4u);

        const std::uint8_t* weight_row0 = tile_weight + row0 * (kTileK / 2);
        const std::uint8_t* weight_row1 = tile_weight + row1 * (kTileK / 2);
        const std::uint32_t a0 = dequant_pair(weight_row0[pair0 / 2], scale0);
        const std::uint32_t a1 = dequant_pair(weight_row1[pair0 / 2], scale1);
        const std::uint32_t a2 = dequant_pair(weight_row0[pair1 / 2], scale0);
        const std::uint32_t a3 = dequant_pair(weight_row1[pair1 / 2], scale1);

        std::uint32_t b0 = 0;
        std::uint32_t b1 = 0;
        if (lane < 4) {
            const float* input_tile =
                input + static_cast<std::size_t>(input_row) * in_features
                + k_tile * kTileK;
            b0 = pack_bf16_pair(input_tile[pair0], input_tile[pair0 + 1]);
            b1 = pack_bf16_pair(input_tile[pair1], input_tile[pair1 + 1]);
        }
        b0 = __shfl_sync(0xffffffffu, b0, lane & 3u);
        b1 = __shfl_sync(0xffffffffu, b1, lane & 3u);
        mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }

    __shared__ float partial[WarpsPerBlock][kTileM];
    if ((lane & 3u) == 0) {
        partial[warp][row0] = d0;
        partial[warp][row1] = d2;
    }
    __syncthreads();
    if (threadIdx.x < kTileM) {
        float value = 0.0f;
        #pragma unroll
        for (std::uint32_t partial_index = 0; partial_index < WarpsPerBlock;
             ++partial_index) {
            value += partial[partial_index][threadIdx.x];
        }
        const std::size_t output_base =
            static_cast<std::size_t>(route) * out_features + out_tile * kTileM;
        const __nv_bfloat16 output_value = __float2bfloat16_rn(value);
        output_bf16[output_base + threadIdx.x] = output_value;
        if constexpr (WriteF32) {
            output_f32[output_base + threadIdx.x] = __bfloat162float(output_value);
        }
    }
}

template <bool WriteF32, std::uint32_t WarpsPerBlock, std::uint32_t FixedTopK>
void launch_shape(
    dim3 grid,
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* tiled_weight,
    const std::uint8_t* tiled_scales,
    const float* global_scales,
    std::uint16_t* output_bf16,
    float* output_f32,
    std::uint32_t batch_size,
    std::uint32_t top_k,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    routed_w4a16_kernel<WriteF32, WarpsPerBlock, FixedTopK>
        <<<grid, WarpsPerBlock * 32, 0, stream>>>(
            indices,
            input,
            tiled_weight,
            tiled_scales,
            global_scales,
            reinterpret_cast<__nv_bfloat16*>(output_bf16),
            output_f32,
            batch_size,
            top_k,
            out_features,
            in_features);
}

template <bool WriteF32>
void launch_routed_shape(
    dim3 grid,
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* tiled_weight,
    const std::uint8_t* tiled_scales,
    const float* global_scales,
    std::uint16_t* output_bf16,
    float* output_f32,
    std::uint32_t batch_size,
    std::uint32_t top_k,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (batch_size != 1) {
        launch_shape<WriteF32, 8, 0>(
            grid, indices, input, tiled_weight, tiled_scales, global_scales,
            output_bf16, output_f32, batch_size, top_k, out_features,
            in_features, stream);
        return;
    }
    switch (top_k) {
        case 8:
            launch_shape<WriteF32, 16, 8>(
                grid, indices, input, tiled_weight, tiled_scales, global_scales,
                output_bf16, output_f32, batch_size, top_k, out_features,
                in_features, stream);
            break;
        case 10:
            launch_shape<WriteF32, 16, 10>(
                grid, indices, input, tiled_weight, tiled_scales, global_scales,
                output_bf16, output_f32, batch_size, top_k, out_features,
                in_features, stream);
            break;
        default:
            launch_shape<WriteF32, 16, 0>(
                grid, indices, input, tiled_weight, tiled_scales, global_scales,
                output_bf16, output_f32, batch_size, top_k, out_features,
                in_features, stream);
            break;
    }
}

cudaError_t launch_routed(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* tiled_weight,
    const std::uint8_t* tiled_scales,
    const float* global_scales,
    std::uint16_t* output_bf16,
    float* output_f32,
    std::uint32_t batch_size,
    std::uint32_t top_k,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    if (indices == nullptr || input == nullptr || tiled_weight == nullptr
        || tiled_scales == nullptr || global_scales == nullptr || output_bf16 == nullptr
        || batch_size == 0 || top_k == 0 || out_features == 0 || in_features == 0
        || out_features % kTileM != 0 || in_features % kTileK != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(out_features / kTileM, batch_size * top_k);
    if (output_f32 == nullptr) {
        launch_routed_shape<false>(
            grid, indices, input, tiled_weight, tiled_scales, global_scales,
            output_bf16, nullptr, batch_size, top_k, out_features, in_features,
            stream);
    } else {
        launch_routed_shape<true>(
            grid, indices, input, tiled_weight, tiled_scales, global_scales,
            output_bf16, output_f32, batch_size, top_k, out_features,
            in_features, stream);
    }
    return cudaGetLastError();
}

}  // namespace

namespace {
// Each MMA column consumes a different activation row. Reuse the decoded
// weight fragment across eight rows, retaining the routed kernel's K reduction.
template <unsigned Warps>
__global__ __launch_bounds__(Warps * 32) void dense_w4a16_batch_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ weight,
    const std::uint8_t* __restrict__ scales,
    const float* __restrict__ global,
    float* __restrict__ output,
    unsigned rows, unsigned out, unsigned in) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned row0 = lane >> 2;
    const unsigned row1 = row0 + 8;
    const unsigned pair = (lane & 3) * 2;
    const unsigned first = blockIdx.y * 8;
    const unsigned input_row = first + row0;
    const unsigned tiles = in / 16;
    const float scale = global[0];
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    for (unsigned k = warp; k < tiles; k += Warps) {
        const size_t tile = size_t(blockIdx.x) * tiles + k;
        const auto* w = weight + tile * kPackedTileBytes;
        const auto* s = scales + tile * kScaleTileBytes;
        float s0 = 0.f, s1 = 0.f;
        if ((lane & 3) == 0) {
            s0 = e4m3_value(s[row0]) * scale;
            s1 = e4m3_value(s[row1]) * scale;
        }
        s0 = __shfl_sync(0xffffffff, s0, row0 * 4);
        s1 = __shfl_sync(0xffffffff, s1, row0 * 4);
        const auto a0 = dequant_pair(w[row0 * 8 + pair / 2], s0);
        const auto a1 = dequant_pair(w[row1 * 8 + pair / 2], s1);
        const auto a2 = dequant_pair(w[row0 * 8 + pair / 2 + 4], s0);
        const auto a3 = dequant_pair(w[row1 * 8 + pair / 2 + 4], s1);
        unsigned b0 = 0, b1 = 0;
        if (input_row < rows) {
            const float* x = input + size_t(input_row) * in + k * 16 + pair;
            b0 = pack_bf16_pair(x[0], x[1]);
            b1 = pack_bf16_pair(x[8], x[9]);
        }
        mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }
    __shared__ float partial[Warps][16 * 8];
    partial[warp][row0 * 8 + pair] = d0;
    partial[warp][row0 * 8 + pair + 1] = d1;
    partial[warp][row1 * 8 + pair] = d2;
    partial[warp][row1 * 8 + pair + 1] = d3;
    __syncthreads();
    if (threadIdx.x < 128) {
        const unsigned index = threadIdx.x;
        const unsigned row = first + index % 8;
        if (row < rows) {
            float sum = 0.f;
            #pragma unroll
            for (unsigned w = 0; w < Warps; ++w) sum += partial[w][index];
            output[size_t(row) * out + blockIdx.x * 16 + index / 8] =
                __bfloat162float(__float2bfloat16_rn(sum));
        }
    }
}
}  // namespace

extern "C" cudaError_t infer_sm121_w4a16_dense_batch_on_stream(
    const float* input, const std::uint8_t* weight, const std::uint8_t* scales,
    const float* global, float* output, unsigned rows, unsigned out, unsigned in,
    cudaStream_t stream) {
    if (!input || !weight || !scales || !global || !output || !rows || !out || !in
        || out % 16 || in % 16) return cudaErrorInvalidValue;
    dense_w4a16_batch_kernel<8><<<dim3(out / 16, (rows + 7) / 8), 256, 0, stream>>>(
        input, weight, scales, global, output, rows, out, in);
    return cudaGetLastError();
}

namespace {
constexpr unsigned kGroupColumns = 64;

// Sorts routes by expert and splits each expert's routes into work items of
// at most kGroupColumns. Route order within an expert does not affect values:
// every route is a separate MMA column with the same K reduction.
__global__ void group_routes_kernel(
    const std::uint32_t* __restrict__ ids, unsigned routes, unsigned experts,
    std::uint32_t* __restrict__ sorted, std::uint32_t* __restrict__ item_expert,
    std::uint32_t* __restrict__ item_start, std::uint32_t* __restrict__ item_count,
    std::uint32_t* __restrict__ items) {
    extern __shared__ std::uint32_t shared[];
    std::uint32_t* counts = shared;
    std::uint32_t* cursor = shared + experts;
    for (unsigned e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    for (unsigned r = threadIdx.x; r < routes; r += blockDim.x) atomicAdd(&counts[ids[r]], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned offset = 0, item = 0;
        for (unsigned e = 0; e < experts; ++e) {
            cursor[e] = offset;
            for (unsigned first = 0; first < counts[e]; first += kGroupColumns) {
                item_expert[item] = e;
                item_start[item] = offset + first;
                item_count[item] = min(kGroupColumns, counts[e] - first);
                ++item;
            }
            offset += counts[e];
        }
        items[0] = item;
    }
    __syncthreads();
    for (unsigned r = threadIdx.x; r < routes; r += blockDim.x)
        sorted[atomicAdd(&cursor[ids[r]], 1u)] = r;
}

// Decode-sized batches: one CTA computes a 16-row output tile for one work
// item, splitting K across warps. Each decoded weight fragment feeds one MMA
// per eight routes, whose N columns are the routes. Items with more than
// `Groups * 8` routes repeat the K loop for the remaining columns.
template <unsigned Warps, unsigned Groups>
__global__ __launch_bounds__(Warps * 32) void grouped_w4a16_kernel(
    const std::uint32_t* __restrict__ sorted, const std::uint32_t* __restrict__ item_expert,
    const std::uint32_t* __restrict__ item_start, const std::uint32_t* __restrict__ item_count,
    const std::uint32_t* __restrict__ items, const float* __restrict__ input,
    const std::uint8_t* __restrict__ tiled_weight, const std::uint8_t* __restrict__ tiled_scales,
    const float* __restrict__ global_scales, float* __restrict__ output, unsigned top_k,
    bool per_route_input, unsigned out_features, unsigned in_features) {
    const unsigned item = blockIdx.y;
    if (item >= items[0]) return;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned row0 = lane >> 2, row1 = row0 + 8, pair = (lane & 3) * 2;
    const unsigned expert = item_expert[item], start = item_start[item], count = item_count[item];
    const unsigned tiles = in_features / kTileK;
    const std::uint8_t* weight =
        tiled_weight + size_t(expert) * out_features * in_features / 2;
    const std::uint8_t* scales = tiled_scales + size_t(expert) * out_features * in_features / kTileK;
    const float global = global_scales[expert];
    __shared__ float partial[Warps][Groups][16 * 8];
    for (unsigned first = 0; first < count; first += Groups * 8) {
        const float* rows[Groups];
        #pragma unroll
        for (unsigned g = 0; g < Groups; ++g) {
            const unsigned column = first + g * 8 + row0;
            rows[g] = nullptr;
            if (column < count) {
                const unsigned route = sorted[start + column];
                rows[g] = input + size_t(per_route_input ? route : route / top_k) * in_features;
            }
        }
        float d[Groups][4] = {};
        for (unsigned k = warp; k < tiles; k += Warps) {
            const size_t tile = size_t(blockIdx.x) * tiles + k;
            const auto* w = weight + tile * kPackedTileBytes;
            const auto* s = scales + tile * kScaleTileBytes;
            float s0 = 0.f, s1 = 0.f;
            if ((lane & 3) == 0) {
                s0 = e4m3_value(s[row0]) * global;
                s1 = e4m3_value(s[row1]) * global;
            }
            s0 = __shfl_sync(0xffffffffu, s0, row0 * 4);
            s1 = __shfl_sync(0xffffffffu, s1, row0 * 4);
            const auto a0 = dequant_pair(w[row0 * 8 + pair / 2], s0);
            const auto a1 = dequant_pair(w[row1 * 8 + pair / 2], s1);
            const auto a2 = dequant_pair(w[row0 * 8 + pair / 2 + 4], s0);
            const auto a3 = dequant_pair(w[row1 * 8 + pair / 2 + 4], s1);
            #pragma unroll
            for (unsigned g = 0; g < Groups; ++g) {
                if (first + g * 8 >= count) break;
                unsigned b0 = 0, b1 = 0;
                if (rows[g]) {
                    const float* x = rows[g] + k * kTileK + pair;
                    b0 = pack_bf16_pair(x[0], x[1]);
                    b1 = pack_bf16_pair(x[8], x[9]);
                }
                mma_m16n8k16(d[g][0], d[g][1], d[g][2], d[g][3], a0, a1, a2, a3, b0, b1);
            }
        }
        #pragma unroll
        for (unsigned g = 0; g < Groups; ++g) {
            partial[warp][g][row0 * 8 + pair] = d[g][0];
            partial[warp][g][row0 * 8 + pair + 1] = d[g][1];
            partial[warp][g][row1 * 8 + pair] = d[g][2];
            partial[warp][g][row1 * 8 + pair + 1] = d[g][3];
        }
        __syncthreads();
        for (unsigned index = threadIdx.x; index < Groups * 128; index += blockDim.x) {
            const unsigned g = index / 128, cell = index % 128;
            const unsigned column = first + g * 8 + cell % 8;
            if (column >= count) continue;
            float sum = 0.f;
            #pragma unroll
            for (unsigned w = 0; w < Warps; ++w) sum += partial[w][g][cell];
            const unsigned route = sorted[start + column];
            output[size_t(route) * out_features + blockIdx.x * kTileM + cell / 8] =
                __bfloat162float(__float2bfloat16_rn(sum));
        }
        __syncthreads();
    }
}

// Prefill-sized batches: one CTA computes 16 * Warps output rows for one work
// item, each warp owning a 16-row tile over the full K. Route activations are
// staged once per 64-wide K chunk as BF16 pairs, so every weight fragment is
// decoded once per item and every activation is read once per CTA.
template <unsigned Warps>
__global__ __launch_bounds__(Warps * 32) void grouped_w4a16_tiled_kernel(
    const std::uint32_t* __restrict__ sorted, const std::uint32_t* __restrict__ item_expert,
    const std::uint32_t* __restrict__ item_start, const std::uint32_t* __restrict__ item_count,
    const std::uint32_t* __restrict__ items, const float* __restrict__ input,
    const std::uint8_t* __restrict__ tiled_weight, const std::uint8_t* __restrict__ tiled_scales,
    const float* __restrict__ global_scales, float* __restrict__ output, unsigned top_k,
    bool per_route_input, unsigned out_features, unsigned in_features) {
    constexpr unsigned Groups = kGroupColumns / 8, Chunk = 64, Words = Chunk / 2, Stride = Words + 4;
    __shared__ std::uint32_t staged[kGroupColumns][Stride];
    __shared__ const float* rows[kGroupColumns];
    const unsigned item = blockIdx.y;
    if (item >= items[0]) return;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned row0 = lane >> 2, row1 = row0 + 8, quad = lane & 3;
    const unsigned expert = item_expert[item], start = item_start[item], count = item_count[item];
    for (unsigned column = threadIdx.x; column < kGroupColumns; column += blockDim.x) {
        rows[column] = nullptr;
        if (column < count) {
            const unsigned route = sorted[start + column];
            rows[column] = input + size_t(per_route_input ? route : route / top_k) * in_features;
        }
    }
    const unsigned out_tile = blockIdx.x * Warps + warp;
    const bool active = out_tile * kTileM < out_features;
    const unsigned tiles = in_features / kTileK;
    const std::uint8_t* weight =
        tiled_weight + size_t(expert) * out_features * in_features / 2;
    const std::uint8_t* scales = tiled_scales + size_t(expert) * out_features * in_features / kTileK;
    const float global = global_scales[expert];
    float d[Groups][4] = {};
    __syncthreads();
    for (unsigned k0 = 0; k0 < in_features; k0 += Chunk) {
        // Each thread converts four F32 pairs of one route row.
        for (unsigned index = threadIdx.x; index < kGroupColumns * (Words / 4); index += blockDim.x) {
            const unsigned column = index / (Words / 4), word = index % (Words / 4) * 4;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (rows[column] && k0 + word * 2 < in_features) {
                const float4 lo = *reinterpret_cast<const float4*>(rows[column] + k0 + word * 2);
                const float4 hi = *reinterpret_cast<const float4*>(rows[column] + k0 + word * 2 + 4);
                packed = make_uint4(pack_bf16_pair(lo.x, lo.y), pack_bf16_pair(lo.z, lo.w),
                                    pack_bf16_pair(hi.x, hi.y), pack_bf16_pair(hi.z, hi.w));
            }
            *reinterpret_cast<uint4*>(&staged[column][word]) = packed;
        }
        __syncthreads();
        if (active) {
            #pragma unroll
            for (unsigned kt = 0; kt < Chunk / kTileK; ++kt) {
                const unsigned k = k0 / kTileK + kt;
                if (k >= tiles) break;
                const size_t tile = size_t(out_tile) * tiles + k;
                const auto* w = weight + tile * kPackedTileBytes;
                const auto* sc = scales + tile * kScaleTileBytes;
                float s0 = 0.f, s1 = 0.f;
                if (quad == 0) {
                    s0 = e4m3_value(sc[row0]) * global;
                    s1 = e4m3_value(sc[row1]) * global;
                }
                s0 = __shfl_sync(0xffffffffu, s0, row0 * 4);
                s1 = __shfl_sync(0xffffffffu, s1, row0 * 4);
                const auto a0 = dequant_pair(w[row0 * 8 + quad], s0);
                const auto a1 = dequant_pair(w[row1 * 8 + quad], s1);
                const auto a2 = dequant_pair(w[row0 * 8 + quad + 4], s0);
                const auto a3 = dequant_pair(w[row1 * 8 + quad + 4], s1);
                // One ldmatrix.x4 loads the K16 B fragments of two groups:
                // matrices are (group, k 0-7), (group, k 8-15), then group + 1.
                // Columns past `count` were staged as zeros.
                const unsigned matrix = lane >> 3;
                const std::uint32_t* b_row =
                    staged[(matrix >> 1) * 8 + (lane & 7)] + kt * 8 + (matrix & 1) * 4;
                #pragma unroll
                for (unsigned g = 0; g < Groups; g += 2) {
                    if (g * 8 >= count) break;
                    std::uint32_t b[4];
                    const auto address = static_cast<std::uint32_t>(
                        __cvta_generic_to_shared(b_row + g * 8 * Stride));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(address));
                    mma_m16n8k16(d[g][0], d[g][1], d[g][2], d[g][3], a0, a1, a2, a3, b[0], b[1]);
                    mma_m16n8k16(d[g + 1][0], d[g + 1][1], d[g + 1][2], d[g + 1][3],
                                 a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;
    #pragma unroll
    for (unsigned g = 0; g < Groups; ++g) {
        #pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            const unsigned column = g * 8 + quad * 2 + half;
            if (column >= count) continue;
            float* out = output + size_t(sorted[start + column]) * out_features + out_tile * kTileM;
            out[row0] = __bfloat162float(__float2bfloat16_rn(d[g][half]));
            out[row1] = __bfloat162float(__float2bfloat16_rn(d[g][2 + half]));
        }
    }
}
}  // namespace

extern "C" cudaError_t infer_moe_group_routes_on_stream(
    const std::uint32_t* ids, unsigned routes, unsigned experts, std::uint32_t* sorted,
    std::uint32_t* item_expert, std::uint32_t* item_start, std::uint32_t* item_count,
    std::uint32_t* items, cudaStream_t stream) {
    if (!ids || !sorted || !item_expert || !item_start || !item_count || !items || !routes
        || !experts || experts > 4096) return cudaErrorInvalidValue;
    group_routes_kernel<<<1, 1024, 2 * experts * sizeof(std::uint32_t), stream>>>(
        ids, routes, experts, sorted, item_expert, item_start, item_count, items);
    return cudaGetLastError();
}

extern "C" unsigned infer_moe_group_columns() { return kGroupColumns; }

extern "C" cudaError_t infer_sm121_w4a16_grouped_routes_on_stream(
    const std::uint32_t* sorted, const std::uint32_t* item_expert,
    const std::uint32_t* item_start, const std::uint32_t* item_count,
    const std::uint32_t* items, unsigned max_items, const float* input,
    const std::uint8_t* tiled_weight, const std::uint8_t* tiled_scales,
    const float* global_scales, float* output, unsigned top_k, bool per_route_input,
    unsigned out_features, unsigned in_features, bool tiled, unsigned max_columns,
    cudaStream_t stream) {
    if (!sorted || !item_expert || !item_start || !item_count || !items || !max_items || !input
        || !tiled_weight || !tiled_scales || !global_scales || !output || !top_k
        || out_features % kTileM || in_features % kTileK) return cudaErrorInvalidValue;
    const dim3 grid(out_features / kTileM, max_items);
    if (tiled) {
        constexpr unsigned Warps = 8;
        if (in_features % 8) return cudaErrorInvalidValue;
        grouped_w4a16_tiled_kernel<Warps>
            <<<dim3((out_features / kTileM + Warps - 1) / Warps, max_items), Warps * 32, 0, stream>>>(
                sorted, item_expert, item_start, item_count, items, input, tiled_weight,
                tiled_scales, global_scales, output, top_k, per_route_input, out_features,
                in_features);
    } else if (max_columns <= 8) {
        // A token never selects an expert twice, so decode items hold at most
        // one route per row.
        grouped_w4a16_kernel<16, 1><<<grid, 16 * 32, 0, stream>>>(
            sorted, item_expert, item_start, item_count, items, input, tiled_weight, tiled_scales,
            global_scales, output, top_k, per_route_input, out_features, in_features);
    } else {
        grouped_w4a16_kernel<8, kGroupColumns / 8><<<grid, 8 * 32, 0, stream>>>(
            sorted, item_expert, item_start, item_count, items, input, tiled_weight, tiled_scales,
            global_scales, output, top_k, per_route_input, out_features, in_features);
    }
    return cudaGetLastError();
}

extern "C" int infer_sm121_w4a16_supported() {
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return 0;
    }
    return properties.major == 12 && properties.minor == 1 ? 1 : 0;
}

extern "C" cudaError_t infer_sm121_w4a16_gate_up_on_stream(
    const std::uint32_t* indices,
    const float* input,
    const std::uint8_t* tiled_weight,
    const std::uint8_t* tiled_scales,
    const float* global_scales,
    std::uint16_t* output_bf16,
    float* output_f32,
    std::uint32_t batch_size,
    std::uint32_t top_k,
    std::uint32_t out_features,
    std::uint32_t in_features,
    cudaStream_t stream) {
    return launch_routed(
        indices,
        input,
        tiled_weight,
        tiled_scales,
        global_scales,
        output_bf16,
        output_f32,
        batch_size,
        top_k,
        out_features,
        in_features,
        stream);
}
