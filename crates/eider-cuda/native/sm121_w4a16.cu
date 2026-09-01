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
