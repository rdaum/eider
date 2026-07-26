#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cmath>

__device__ __forceinline__ std::uint32_t infer_smem_u32(const void* ptr) {
    std::uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem; cvta.to.shared.u64 smem, %1; cvt.u32.u64 %0, smem; }"
        : "=r"(addr)
        : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void infer_store_m16n8_fragment(float* out, float d0, float d1, float d2, float d3) {
    const int lane = threadIdx.x;
    const int row_base = lane >> 2;
    const int col_base = (lane & 3) << 1;
    out[row_base + col_base * 16] = d0;
    out[row_base + (col_base + 1) * 16] = d1;
    out[row_base + 8 + col_base * 16] = d2;
    out[row_base + 8 + (col_base + 1) * 16] = d3;
}

__device__ __forceinline__ std::uint8_t infer_e2m1_code(float value) {
    const float sign = signbit(value) ? -1.0f : 1.0f;
    const float abs_value = fabsf(value);
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0;
    float best_diff = fabsf(abs_value - levels[0]);
    for (int idx = 1; idx < 8; ++idx) {
        const float diff = fabsf(abs_value - levels[idx]);
        if (diff < best_diff || (diff == best_diff && ((idx & 1) == 0) && ((best & 1) == 1))) {
            best = idx;
            best_diff = diff;
        }
    }
    return static_cast<std::uint8_t>(best | (sign < 0.0f ? 0x8 : 0x0));
}

__device__ __forceinline__ float infer_e2m1_value(std::uint8_t code) {
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = levels[code & 0x7];
    return (code & 0x8) == 0 ? value : -value;
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

__device__ __forceinline__ void infer_set_packed_nibble(std::uint8_t* packed, int index, std::uint8_t value) {
    std::uint8_t& byte = packed[index >> 1];
    if ((index & 1) == 0) {
        byte = static_cast<std::uint8_t>((byte & 0xf0u) | (value & 0x0fu));
    } else {
        byte = static_cast<std::uint8_t>((byte & 0x0fu) | ((value & 0x0fu) << 4));
    }
}

__device__ __forceinline__ std::uint8_t infer_get_packed_nibble(
    const std::uint8_t* packed, int index) {
    const std::uint8_t byte = packed[index >> 1];
    return static_cast<std::uint8_t>((index & 1) == 0 ? byte & 0x0fu : byte >> 4);
}

__device__ __forceinline__ std::uint32_t infer_scale_word(const std::uint8_t* scales) {
    return static_cast<std::uint32_t>(scales[0])
        | (static_cast<std::uint32_t>(scales[1]) << 8)
        | (static_cast<std::uint32_t>(scales[2]) << 16)
        | (static_cast<std::uint32_t>(scales[3]) << 24);
}

__device__ __forceinline__ float infer_probability_amplification(std::uint32_t cache_len) {
    const std::uint32_t minimum = (3u * cache_len + 255u) / 256u;
    std::uint32_t amplification = 1;
    while (amplification < minimum) amplification <<= 1;
    return static_cast<float>(amplification);
}

__device__ __forceinline__ void infer_mma_m16n8k64(
    std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
    std::uint32_t b0, std::uint32_t b1,
    std::uint32_t sfa, std::uint32_t sfb,
    float& d0, float& d1, float& d2, float& d3)
{
    float n0;
    float n1;
    float n2;
    float n3;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa), "h"(bid), "h"(tid),
          "r"(sfb), "h"(bid), "h"(tid));
    d0 = n0;
    d1 = n1;
    d2 = n2;
    d3 = n3;
}

__device__ __forceinline__ void infer_load_native_m16n8k64(
    const std::uint8_t* a_tile,
    const std::uint8_t* b_tile,
    std::uint32_t& a0,
    std::uint32_t& a1,
    std::uint32_t& a2,
    std::uint32_t& a3,
    std::uint32_t& b0,
    std::uint32_t& b1)
{
    const auto* a = reinterpret_cast<const std::uint32_t*>(a_tile + threadIdx.x * 16);
    const auto* b = reinterpret_cast<const std::uint32_t*>(b_tile + threadIdx.x * 16);
    a0 = a[0];
    a1 = a[1];
    a2 = a[2];
    a3 = a[3];
    b0 = b[0];
    b1 = b[1];
}

template <std::uint8_t Fill>
__global__ void infer_sm12x_mma_probe_kernel(float* out) {
    __shared__ __align__(16) std::uint8_t smem[4096];

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        smem[i] = Fill;
    }
    __syncthreads();

    std::uint32_t a0 = 0;
    std::uint32_t a1 = 0;
    std::uint32_t a2 = 0;
    std::uint32_t a3 = 0;
    std::uint32_t b0 = 0;
    std::uint32_t b1 = 0;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const float c0 = 0.0f;
    const float c1 = 0.0f;
    const float c2 = 0.0f;
    const float c3 = 0.0f;
    const std::uint32_t sfa = 0x38383838u;
    const std::uint32_t sfb = 0x38383838u;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    std::uint32_t a_addr = infer_smem_u32(smem + threadIdx.x * 16);
    std::uint32_t b_addr = infer_smem_u32(smem + 2048 + threadIdx.x * 16);

    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(a_addr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1)
        : "r"(b_addr));

    if constexpr (Fill == 0) {
        a0 <<= 2;
        a1 <<= 2;
        a2 <<= 2;
        a3 <<= 2;
        b0 <<= 2;
        b1 <<= 2;
    }

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bid), "h"(tid),
          "r"(sfb), "h"(bid), "h"(tid));

    if (threadIdx.x == 0) {
        out[0] = d0;
        out[1] = d1;
        out[2] = d2;
        out[3] = d3;
    }
}

extern "C" cudaError_t infer_sm12x_mma_zero_probe_on_stream(float* out, cudaStream_t stream) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_mma_probe_kernel<0><<<1, 32, 0, stream>>>(out);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_mma_one_probe_on_stream(float* out, cudaStream_t stream) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_mma_probe_kernel<0x04><<<1, 32, 0, stream>>>(out);
    return cudaGetLastError();
}

__global__ void infer_sm12x_ldmatrix_probe_kernel(std::uint32_t* out) {
    __shared__ __align__(16) std::uint8_t smem[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        smem[i] = 0x02;
    }
    __syncthreads();

    std::uint32_t r0 = 0;
    std::uint32_t r1 = 0;
    std::uint32_t r2 = 0;
    std::uint32_t r3 = 0;
    std::uint32_t addr = infer_smem_u32(smem + threadIdx.x * 16);
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
    if (threadIdx.x == 0) {
        out[0] = r0;
        out[1] = r1;
        out[2] = r2;
        out[3] = r3;
    }
}

extern "C" cudaError_t infer_sm12x_ldmatrix_probe_on_stream(std::uint32_t* out, cudaStream_t stream) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_ldmatrix_probe_kernel<<<1, 32, 0, stream>>>(out);
    return cudaGetLastError();
}

__global__ void infer_sm12x_mma_tile_frag_kernel(
    const std::uint8_t* a_native_tile,
    const std::uint8_t* b_native_tile,
    std::uint32_t sfa,
    std::uint32_t sfb,
    float* out)
{
    std::uint32_t a0 = 0;
    std::uint32_t a1 = 0;
    std::uint32_t a2 = 0;
    std::uint32_t a3 = 0;
    std::uint32_t b0 = 0;
    std::uint32_t b1 = 0;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const float c0 = 0.0f;
    const float c1 = 0.0f;
    const float c2 = 0.0f;
    const float c3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    infer_load_native_m16n8k64(a_native_tile, b_native_tile, a0, a1, a2, a3, b0, b1);

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bid), "h"(tid),
          "r"(sfb), "h"(bid), "h"(tid));

    const int base = threadIdx.x * 4;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

extern "C" cudaError_t infer_sm12x_mma_tile_frag_on_stream(
    const std::uint8_t* a_native_tile,
    const std::uint8_t* b_native_tile,
    std::uint32_t sfa,
    std::uint32_t sfb,
    float* out,
    cudaStream_t stream)
{
    if (a_native_tile == nullptr || b_native_tile == nullptr || out == nullptr) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_mma_tile_frag_kernel<<<1, 32, 0, stream>>>(a_native_tile, b_native_tile, sfa, sfb, out);
    return cudaGetLastError();
}

__global__ void infer_sm12x_mma_sfa_lane_probe_kernel(
    const std::uint8_t* a_native_tile,
    const std::uint8_t* b_native_tile,
    const std::uint32_t* sfa_lanes,
    std::uint32_t sfb,
    float* out)
{
    std::uint32_t a0;
    std::uint32_t a1;
    std::uint32_t a2;
    std::uint32_t a3;
    std::uint32_t b0;
    std::uint32_t b1;
    infer_load_native_m16n8k64(
        a_native_tile, b_native_tile, a0, a1, a2, a3, b0, b1);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    infer_mma_m16n8k64(
        a0, a1, a2, a3, b0, b1,
        sfa_lanes[threadIdx.x], sfb, d0, d1, d2, d3);
    infer_store_m16n8_fragment(out, d0, d1, d2, d3);
}

extern "C" cudaError_t infer_sm12x_mma_sfa_lane_probe_on_stream(
    const std::uint8_t* a_native_tile,
    const std::uint8_t* b_native_tile,
    const std::uint32_t* sfa_lanes,
    std::uint32_t sfb,
    float* out,
    cudaStream_t stream)
{
    if (a_native_tile == nullptr || b_native_tile == nullptr ||
        sfa_lanes == nullptr || out == nullptr) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_mma_sfa_lane_probe_kernel<<<1, 32, 0, stream>>>(
        a_native_tile, b_native_tile, sfa_lanes, sfb, out);
    return cudaGetLastError();
}

__global__ void infer_sm12x_mma_tile_frag_kloop_kernel(
    const std::uint8_t* a_native_tiles,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t k_tiles,
    float* out)
{
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    for (std::uint32_t tile = 0; tile < k_tiles; ++tile) {
        const std::uint8_t* a_tile = a_native_tiles + tile * 512;
        const std::uint8_t* b_tile = b_native_tiles + tile * 512;
        std::uint32_t a0 = 0;
        std::uint32_t a1 = 0;
        std::uint32_t a2 = 0;
        std::uint32_t a3 = 0;
        std::uint32_t b0 = 0;
        std::uint32_t b1 = 0;
        infer_load_native_m16n8k64(a_tile, b_tile, a0, a1, a2, a3, b0, b1);

        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[tile]), "h"(bid), "h"(tid),
              "r"(sfb[tile]), "h"(bid), "h"(tid));
        d0 = nd0;
        d1 = nd1;
        d2 = nd2;
        d3 = nd3;
    }

    const int base = threadIdx.x * 4;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

__global__ void infer_sm12x_mma_tile_kloop_kernel(
    const std::uint8_t* a_native_tiles,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t k_tiles,
    float* out)
{
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    for (std::uint32_t tile = 0; tile < k_tiles; ++tile) {
        const std::uint8_t* a_tile = a_native_tiles + tile * 512;
        const std::uint8_t* b_tile = b_native_tiles + tile * 512;
        std::uint32_t a0 = 0;
        std::uint32_t a1 = 0;
        std::uint32_t a2 = 0;
        std::uint32_t a3 = 0;
        std::uint32_t b0 = 0;
        std::uint32_t b1 = 0;
        infer_load_native_m16n8k64(a_tile, b_tile, a0, a1, a2, a3, b0, b1);

        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[tile]), "h"(bid), "h"(tid),
              "r"(sfb[tile]), "h"(bid), "h"(tid));
        d0 = nd0;
        d1 = nd1;
        d2 = nd2;
        d3 = nd3;
    }

    infer_store_m16n8_fragment(out, d0, d1, d2, d3);
}

extern "C" cudaError_t infer_sm12x_mma_tile_frag_kloop_on_stream(
    const std::uint8_t* a_native_tiles,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t k_tiles,
    float* out,
    cudaStream_t stream)
{
    if (a_native_tiles == nullptr || b_native_tiles == nullptr || sfa == nullptr || sfb == nullptr || out == nullptr || k_tiles == 0) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_mma_tile_frag_kloop_kernel<<<1, 32, 0, stream>>>(a_native_tiles, b_native_tiles, sfa, sfb, k_tiles, out);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_mma_tile_kloop_on_stream(
    const std::uint8_t* a_native_tiles,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t k_tiles,
    float* out,
    cudaStream_t stream)
{
    if (a_native_tiles == nullptr || b_native_tiles == nullptr || sfa == nullptr || sfb == nullptr || out == nullptr || k_tiles == 0) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_mma_tile_kloop_kernel<<<1, 32, 0, stream>>>(a_native_tiles, b_native_tiles, sfa, sfb, k_tiles, out);
    return cudaGetLastError();
}

__global__ void infer_sm12x_native_gemv_kernel(
    const std::uint8_t* a_native_tiles,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    float* out)
{
    const std::uint32_t m_tile = blockIdx.x;
    if (m_tile >= m_tiles) return;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::uint8_t* a_tile = a_native_tiles + (m_tile * k_tiles + k_tile) * 512;
        const std::uint8_t* b_tile = b_native_tiles + k_tile * 512;
        const std::uint32_t* a_regs = reinterpret_cast<const std::uint32_t*>(a_tile + threadIdx.x * 16);
        const std::uint32_t* b_regs = reinterpret_cast<const std::uint32_t*>(b_tile + threadIdx.x * 16);
        std::uint32_t a0 = a_regs[0];
        std::uint32_t a1 = a_regs[1];
        std::uint32_t a2 = a_regs[2];
        std::uint32_t a3 = a_regs[3];
        std::uint32_t b0 = b_regs[0];
        std::uint32_t b1 = b_regs[1];

        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[m_tile * k_tiles + k_tile]), "h"(bid), "h"(tid),
              "r"(sfb[k_tile]), "h"(bid), "h"(tid));
        d0 = nd0;
        d1 = nd1;
        d2 = nd2;
        d3 = nd3;
    }

    if ((threadIdx.x & 3) == 0) {
        const int row_base = threadIdx.x >> 2;
        const int out_base = static_cast<int>(m_tile) * 16;
        out[out_base + row_base] = d0;
        out[out_base + row_base + 8] = d2;
    }
}

extern "C" cudaError_t infer_sm12x_native_gemv_on_stream(
    const std::uint8_t* a_native_tiles,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    float* out,
    cudaStream_t stream)
{
    if (a_native_tiles == nullptr || b_native_tiles == nullptr || sfa == nullptr || sfb == nullptr || out == nullptr || m_tiles == 0 || k_tiles == 0) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_native_gemv_kernel<<<m_tiles, 32, 0, stream>>>(a_native_tiles, b_native_tiles, sfa, sfb, m_tiles, k_tiles, out);
    return cudaGetLastError();
}

__global__ void infer_sm12x_quantize_fixed_scale_vector_kernel(
    const float* __restrict__ input,
    float input_scale,
    std::uint32_t k_tiles,
    std::uint8_t* __restrict__ b_native_tiles,
    std::uint32_t* __restrict__ sfb)
{
    const std::uint32_t kt = blockIdx.x;
    if (kt >= k_tiles) return;
    std::uint8_t* tile = b_native_tiles + kt * 512;
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        tile[idx] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        std::uint8_t codes[64];
        for (int col = 0; col < 64; ++col) {
            codes[col] = infer_e2m1_code(input[kt * 64 + col] / input_scale);
        }
        for (int lane = 0; lane < 32; ++lane) {
            const int t0 = lane & 3;
            const int t1 = lane >> 2;
            for (int v = 0; v < 16; ++v) {
                const int v0 = v & 7;
                const int v1 = (v >> 3) & 1;
                const int col = t0 * 8 + v0 + 32 * v1;
                infer_set_packed_nibble(tile, lane * 32 + v, codes[col]);
            }
        }
        sfb[kt] = 0x38383838u;
    }
}

extern "C" cudaError_t infer_sm12x_quantize_fixed_scale_vector_on_stream(
    const float* input,
    float input_scale,
    std::uint32_t k,
    std::uint8_t* b_native_tiles,
    std::uint32_t* sfb,
    cudaStream_t stream)
{
    if (input == nullptr || b_native_tiles == nullptr || sfb == nullptr || input_scale <= 0.0f || !isfinite(input_scale) || k == 0 || (k % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_quantize_fixed_scale_vector_kernel<<<k / 64, 128, 0, stream>>>(input, input_scale, k / 64, b_native_tiles, sfb);
    return cudaGetLastError();
}

__global__ void infer_sm12x_quantize_dynamic_vector_kernel(
    const float* input, std::uint32_t k_tiles, std::uint8_t* b_native_tiles,
    std::uint32_t* sfb) {
    const std::uint32_t kt = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    if (kt >= k_tiles) return;
    input += row * k_tiles * 64;
    b_native_tiles += row * k_tiles * 512;
    sfb += row * k_tiles;
    __shared__ std::uint8_t codes[64];
    __shared__ std::uint8_t scale_codes[4];
    __shared__ float scales[4];
    std::uint8_t* tile = b_native_tiles + kt * 512;
    for (int index = threadIdx.x; index < 512; index += blockDim.x) tile[index] = 0;
    const int scale_group = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    float value = 0.0f;
    if (lane < 16) {
        value = input[kt * 64 + scale_group * 16 + lane];
    }
    float max_abs = lane < 16 && isfinite(value) ? fabsf(value) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        scale_codes[scale_group] = scale_code;
        scales[scale_group] = infer_e4m3_value(scale_code);
    }
    __syncthreads();
    if (lane < 16) {
        const float scale = scales[scale_group];
        codes[scale_group * 16 + lane] =
            infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
    }
    __syncthreads();
    for (int packed_idx = threadIdx.x; packed_idx < 256; packed_idx += blockDim.x) {
        const int output_lane = packed_idx >> 3;
        const int pair = packed_idx & 7;
        const int v = pair << 1;
        const int t0 = output_lane & 3;
        const int col0 = t0 * 8 + (v & 7) + 32 * ((v >> 3) & 1);
        const int next_v = v + 1;
        const int col1 = t0 * 8 + (next_v & 7) + 32 * ((next_v >> 3) & 1);
        tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            codes[col0] | (codes[col1] << 4));
    }
    if (threadIdx.x == 0) {
        sfb[kt] = static_cast<std::uint32_t>(scale_codes[0])
            | (static_cast<std::uint32_t>(scale_codes[1]) << 8)
            | (static_cast<std::uint32_t>(scale_codes[2]) << 16)
            | (static_cast<std::uint32_t>(scale_codes[3]) << 24);
    }
}

extern "C" cudaError_t infer_sm12x_quantize_dynamic_vector_on_stream(
    const float* input, std::uint32_t k, std::uint8_t* b_native_tiles,
    std::uint32_t* sfb, cudaStream_t stream) {
    if (input == nullptr || b_native_tiles == nullptr || sfb == nullptr || k == 0 || (k % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_quantize_dynamic_vector_kernel<<<k / 64, 128, 0, stream>>>(input, k / 64, b_native_tiles, sfb);
    return cudaGetLastError();
}

__global__ void infer_sm12x_quantize_dynamic_vectors_residual2_kernel(
    const float* __restrict__ input,
    std::uint32_t k_tiles,
    std::uint8_t* __restrict__ primary_tiles,
    std::uint32_t* __restrict__ primary_scales,
    std::uint8_t* __restrict__ residual_tiles,
    std::uint32_t* __restrict__ residual_scales,
    std::uint8_t* __restrict__ residual2_tiles,
    std::uint32_t* __restrict__ residual2_scales,
    float input_multiplier)
{
    const std::uint32_t kt = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    if (kt >= k_tiles) return;
    input += row * k_tiles * 64;
    primary_tiles += row * k_tiles * 512;
    primary_scales += row * k_tiles;
    residual_tiles += row * k_tiles * 512;
    residual_scales += row * k_tiles;
    residual2_tiles += row * k_tiles * 512;
    residual2_scales += row * k_tiles;

    __shared__ float values[64];
    __shared__ float residuals[64];
    __shared__ std::uint8_t primary_codes[64];
    __shared__ std::uint8_t residual_codes[64];
    __shared__ std::uint8_t residual2_codes[64];
    __shared__ std::uint8_t primary_scale_codes[4];
    __shared__ std::uint8_t residual_scale_codes[4];
    __shared__ std::uint8_t residual2_scale_codes[4];
    __shared__ float primary_scale_values[4];
    __shared__ float residual_scale_values[4];
    __shared__ float residual2_scale_values[4];
    std::uint8_t* primary_tile = primary_tiles + kt * 512;
    std::uint8_t* residual_tile = residual_tiles + kt * 512;
    std::uint8_t* residual2_tile = residual2_tiles + kt * 512;
    for (int index = threadIdx.x; index < 512; index += blockDim.x) {
        primary_tile[index] = 0;
        residual_tile[index] = 0;
        residual2_tile[index] = 0;
    }

    const int scale_group = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    float value = 0.0f;
    if (lane < 16) {
        value = input[kt * 64 + scale_group * 16 + lane] * input_multiplier;
        values[scale_group * 16 + lane] = value;
    }
    float max_abs = lane < 16 && isfinite(value) ? fabsf(value) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
        const std::uint8_t code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        primary_scale_codes[scale_group] = code;
        primary_scale_values[scale_group] = infer_e4m3_value(code);
    }
    __syncthreads();

    float residual = 0.0f;
    if (lane < 16) {
        const float scale = primary_scale_values[scale_group];
        const std::uint8_t code = infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
        primary_codes[scale_group * 16 + lane] = code;
        residual = value - infer_e2m1_value(code) * scale;
        residuals[scale_group * 16 + lane] = residual;
    }
    float residual_max = lane < 16 && isfinite(residual) ? fabsf(residual) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        residual_max = fmaxf(
            residual_max,
            __shfl_down_sync(0xffffffffu, residual_max, delta));
    }
    if (lane == 0) {
        const std::uint8_t code = residual_max == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(residual_max / 6.0f, __NV_SATFINITE, __NV_E4M3));
        residual_scale_codes[scale_group] = code;
        residual_scale_values[scale_group] = infer_e4m3_value(code);
    }
    __syncthreads();
    float residual2 = 0.0f;
    if (lane < 16) {
        const float scale = residual_scale_values[scale_group];
        const std::uint8_t code =
            infer_e2m1_code(scale == 0.0f ? 0.0f : residual / scale);
        residual_codes[scale_group * 16 + lane] = code;
        residual2 = residual - infer_e2m1_value(code) * scale;
    }
    float residual2_max = lane < 16 && isfinite(residual2) ? fabsf(residual2) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        residual2_max = fmaxf(
            residual2_max,
            __shfl_down_sync(0xffffffffu, residual2_max, delta));
    }
    if (lane == 0) {
        const std::uint8_t code = residual2_max == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(residual2_max / 6.0f, __NV_SATFINITE, __NV_E4M3));
        residual2_scale_codes[scale_group] = code;
        residual2_scale_values[scale_group] = infer_e4m3_value(code);
    }
    __syncthreads();
    if (lane < 16) {
        const float scale = residual2_scale_values[scale_group];
        residual2_codes[scale_group * 16 + lane] =
            infer_e2m1_code(scale == 0.0f ? 0.0f : residual2 / scale);
    }
    __syncthreads();

    for (int packed_idx = threadIdx.x; packed_idx < 256; packed_idx += blockDim.x) {
        const int output_lane = packed_idx >> 3;
        const int pair = packed_idx & 7;
        const int v = pair << 1;
        const int t0 = output_lane & 3;
        const int col0 = t0 * 8 + (v & 7) + 32 * ((v >> 3) & 1);
        const int next_v = v + 1;
        const int col1 = t0 * 8 + (next_v & 7) + 32 * ((next_v >> 3) & 1);
        primary_tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            primary_codes[col0] | (primary_codes[col1] << 4));
        residual_tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            residual_codes[col0] | (residual_codes[col1] << 4));
        residual2_tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            residual2_codes[col0] | (residual2_codes[col1] << 4));
    }
    if (threadIdx.x == 0) {
        primary_scales[kt] = static_cast<std::uint32_t>(primary_scale_codes[0])
            | (static_cast<std::uint32_t>(primary_scale_codes[1]) << 8)
            | (static_cast<std::uint32_t>(primary_scale_codes[2]) << 16)
            | (static_cast<std::uint32_t>(primary_scale_codes[3]) << 24);
        residual_scales[kt] = static_cast<std::uint32_t>(residual_scale_codes[0])
            | (static_cast<std::uint32_t>(residual_scale_codes[1]) << 8)
            | (static_cast<std::uint32_t>(residual_scale_codes[2]) << 16)
            | (static_cast<std::uint32_t>(residual_scale_codes[3]) << 24);
        residual2_scales[kt] = static_cast<std::uint32_t>(residual2_scale_codes[0])
            | (static_cast<std::uint32_t>(residual2_scale_codes[1]) << 8)
            | (static_cast<std::uint32_t>(residual2_scale_codes[2]) << 16)
            | (static_cast<std::uint32_t>(residual2_scale_codes[3]) << 24);
    }
}

extern "C" cudaError_t infer_sm12x_quantize_dynamic_vectors_residual2_on_stream(
    const float* input,
    std::uint32_t rows,
    std::uint32_t k,
    std::uint8_t* primary_tiles,
    std::uint32_t* primary_scales,
    std::uint8_t* residual_tiles,
    std::uint32_t* residual_scales,
    std::uint8_t* residual2_tiles,
    std::uint32_t* residual2_scales,
    float input_multiplier,
    cudaStream_t stream)
{
    if (input == nullptr || primary_tiles == nullptr || primary_scales == nullptr ||
        residual_tiles == nullptr || residual_scales == nullptr ||
        residual2_tiles == nullptr || residual2_scales == nullptr ||
        rows == 0 || k == 0 || (k % 64) != 0 ||
        input_multiplier <= 0.0f || !isfinite(input_multiplier)) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_quantize_dynamic_vectors_residual2_kernel<<<
        dim3(k / 64, rows), 128, 0, stream>>>(
        input, k / 64, primary_tiles, primary_scales,
        residual_tiles, residual_scales, residual2_tiles, residual2_scales,
        input_multiplier);
    return cudaGetLastError();
}

__global__ void infer_sm12x_kv_copy_tail_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ key_tail,
    float* __restrict__ value_tail,
    std::uint32_t position,
    std::uint32_t width)
{
    const std::uint32_t row = blockIdx.y;
    position += row;
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    const std::uint32_t destination = (position & 15u) * width + column;
    const std::uint32_t source = row * width + column;
    key_tail[destination] = key[source];
    value_tail[destination] = value[source];
}

__global__ void infer_sm12x_kv_finalize_key_kernel(
    const float* __restrict__ key_tail,
    std::uint8_t* __restrict__ key_values,
    std::uint8_t* __restrict__ key_scales,
    std::uint32_t position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    position += blockIdx.z;
    if ((position & 7u) != 7u) return;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t k_block = blockIdx.y;
    if (head >= kv_heads || k_block >= head_dim / 16 || threadIdx.x != 0) return;

    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t tail_start = (position & 15u) & ~7u;
    const std::uint32_t token_tiles = (max_tokens + 7) / 8;
    const std::uint32_t k_tiles = head_dim / 64;
    const std::uint32_t token_tile = position / 8;
    const std::uint32_t k_tile = k_block / 4;
    const std::uint32_t scale_block = k_block & 3u;
    const std::uint32_t tile = (head * token_tiles + token_tile) * k_tiles + k_tile;
    std::uint8_t* packed = key_values + tile * 256;
    for (std::uint32_t token = 0; token < 8; ++token) {
        float max_abs = 0.0f;
        for (std::uint32_t offset = 0; offset < 16; ++offset) {
            const float value = key_tail[(tail_start + token) * width + head * head_dim + k_block * 16 + offset];
            if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
        }
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const float scale = infer_e4m3_value(scale_code);
        for (std::uint32_t offset = 0; offset < 16; ++offset) {
            const float value = key_tail[(tail_start + token) * width + head * head_dim + k_block * 16 + offset];
            const std::uint8_t code = infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
            infer_set_packed_nibble(packed, token * 64 + scale_block * 16 + offset, code);
        }
        key_scales[(tile * 8 + token) * 4 + scale_block] = scale_code;
    }
}

__global__ void infer_sm12x_kv_finalize_value_kernel(
    const float* __restrict__ value_tail,
    std::uint8_t* __restrict__ value_values,
    std::uint8_t* __restrict__ value_scales,
    std::uint32_t position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    position += blockIdx.z;
    if ((position & 15u) != 15u) return;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t dim_tile = blockIdx.y;
    if (head >= kv_heads || dim_tile >= head_dim / 8 || threadIdx.x != 0) return;

    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t token_tiles = (max_tokens + 63) / 64;
    const std::uint32_t token_tile = position / 64;
    const std::uint32_t scale_block = (position & 63u) / 16;
    const std::uint32_t tile = (head * (head_dim / 8) + dim_tile) * token_tiles + token_tile;
    std::uint8_t* packed = value_values + tile * 256;
    for (std::uint32_t dim = 0; dim < 8; ++dim) {
        float max_abs = 0.0f;
        for (std::uint32_t token = 0; token < 16; ++token) {
            const float value = value_tail[token * width + head * head_dim + dim_tile * 8 + dim];
            if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
        }
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const float scale = infer_e4m3_value(scale_code);
        for (std::uint32_t token = 0; token < 16; ++token) {
            const float value = value_tail[token * width + head * head_dim + dim_tile * 8 + dim];
            const std::uint8_t code = infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
            infer_set_packed_nibble(packed, dim * 64 + scale_block * 16 + token, code);
        }
        value_scales[(tile * 8 + dim) * 4 + scale_block] = scale_code;
    }
}

__global__ void infer_sm12x_kv_copy_key_tiles_kernel(
    const std::uint8_t* __restrict__ source_values,
    const std::uint8_t* __restrict__ source_scales,
    std::uint8_t* __restrict__ destination_values,
    std::uint8_t* __restrict__ destination_scales,
    std::uint32_t source_token_tiles,
    std::uint32_t destination_token_tiles,
    std::uint32_t copied_token_tiles,
    std::uint32_t k_tiles)
{
    const std::uint32_t logical_tile = blockIdx.x;
    const std::uint32_t k_tile = logical_tile % k_tiles;
    const std::uint32_t token_head = logical_tile / k_tiles;
    const std::uint32_t token_tile = token_head % copied_token_tiles;
    const std::uint32_t head = token_head / copied_token_tiles;
    const std::uint32_t source_tile =
        (head * source_token_tiles + token_tile) * k_tiles + k_tile;
    const std::uint32_t destination_tile =
        (head * destination_token_tiles + token_tile) * k_tiles + k_tile;
    if (threadIdx.x < 256) {
        destination_values[destination_tile * 256 + threadIdx.x] =
            source_values[source_tile * 256 + threadIdx.x];
    }
    if (threadIdx.x < 32) {
        destination_scales[destination_tile * 32 + threadIdx.x] =
            source_scales[source_tile * 32 + threadIdx.x];
    }
}

__global__ void infer_sm12x_kv_copy_value_tiles_kernel(
    const std::uint8_t* __restrict__ source_values,
    const std::uint8_t* __restrict__ source_scales,
    std::uint8_t* __restrict__ destination_values,
    std::uint8_t* __restrict__ destination_scales,
    std::uint32_t source_context_tiles,
    std::uint32_t destination_context_tiles,
    std::uint32_t copied_context_tiles)
{
    const std::uint32_t logical_tile = blockIdx.x;
    const std::uint32_t context_tile = logical_tile % copied_context_tiles;
    const std::uint32_t head_dimension_tile = logical_tile / copied_context_tiles;
    const std::uint32_t source_tile =
        head_dimension_tile * source_context_tiles + context_tile;
    const std::uint32_t destination_tile =
        head_dimension_tile * destination_context_tiles + context_tile;
    if (threadIdx.x < 256) {
        destination_values[destination_tile * 256 + threadIdx.x] =
            source_values[source_tile * 256 + threadIdx.x];
    }
    if (threadIdx.x < 32) {
        destination_scales[destination_tile * 32 + threadIdx.x] =
            source_scales[source_tile * 32 + threadIdx.x];
    }
}

extern "C" cudaError_t infer_sm12x_kv_cache_copy_aligned_prefix_on_stream(
    const std::uint8_t* source_key_values,
    const std::uint8_t* source_key_scales,
    const std::uint8_t* source_value_values,
    const std::uint8_t* source_value_scales,
    std::uint8_t* destination_key_values,
    std::uint8_t* destination_key_scales,
    std::uint8_t* destination_value_values,
    std::uint8_t* destination_value_scales,
    std::uint32_t prefix_tokens,
    std::uint32_t source_max_tokens,
    std::uint32_t destination_max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (source_key_values == nullptr || source_key_scales == nullptr ||
        source_value_values == nullptr || source_value_scales == nullptr ||
        destination_key_values == nullptr || destination_key_scales == nullptr ||
        destination_value_values == nullptr || destination_value_scales == nullptr ||
        prefix_tokens == 0 || (prefix_tokens % 128) != 0 ||
        prefix_tokens > source_max_tokens || prefix_tokens > destination_max_tokens ||
        kv_heads == 0 || head_dim == 0 || (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }

    const std::uint32_t source_token_tiles = (source_max_tokens + 7) / 8;
    const std::uint32_t destination_token_tiles = (destination_max_tokens + 7) / 8;
    const std::uint32_t copied_token_tiles = prefix_tokens / 8;
    const std::uint32_t k_tiles = head_dim / 64;
    const std::uint32_t key_tiles = kv_heads * copied_token_tiles * k_tiles;
    infer_sm12x_kv_copy_key_tiles_kernel<<<key_tiles, 256, 0, stream>>>(
        source_key_values, source_key_scales, destination_key_values,
        destination_key_scales, source_token_tiles, destination_token_tiles,
        copied_token_tiles, k_tiles);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    const std::uint32_t source_context_tiles = (source_max_tokens + 63) / 64;
    const std::uint32_t destination_context_tiles = (destination_max_tokens + 63) / 64;
    const std::uint32_t copied_context_tiles = prefix_tokens / 64;
    const std::uint32_t value_tiles =
        kv_heads * (head_dim / 8) * copied_context_tiles;
    infer_sm12x_kv_copy_value_tiles_kernel<<<value_tiles, 256, 0, stream>>>(
        source_value_values, source_value_scales, destination_value_values,
        destination_value_scales, source_context_tiles, destination_context_tiles,
        copied_context_tiles);
    return cudaGetLastError();
}

__global__ void infer_sm12x_kv_copy_tail_indexed_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ key_tail,
    float* __restrict__ value_tail,
    const std::uint32_t* __restrict__ position,
    std::uint32_t max_tokens,
    std::uint32_t width)
{
    const std::uint32_t pos = *position;
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= max_tokens || column >= width) return;
    const std::uint32_t destination = (pos & 15u) * width + column;
    key_tail[destination] = key[column];
    value_tail[destination] = value[column];
}

__global__ void infer_sm12x_kv_finalize_key_indexed_kernel(
    const float* __restrict__ key_tail,
    std::uint8_t* __restrict__ key_values,
    std::uint8_t* __restrict__ key_scales,
    const std::uint32_t* __restrict__ position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    const std::uint32_t pos = *position;
    if (pos >= max_tokens || (pos & 7u) != 7u) return;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t k_block = blockIdx.y;
    if (head >= kv_heads || k_block >= head_dim / 16 || threadIdx.x != 0) return;
    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t tail_start = (pos & 15u) & ~7u;
    const std::uint32_t token_tiles = (max_tokens + 7) / 8;
    const std::uint32_t k_tiles = head_dim / 64;
    const std::uint32_t token_tile = pos / 8;
    const std::uint32_t k_tile = k_block / 4;
    const std::uint32_t scale_block = k_block & 3u;
    const std::uint32_t tile = (head * token_tiles + token_tile) * k_tiles + k_tile;
    std::uint8_t* packed = key_values + tile * 256;
    for (std::uint32_t token = 0; token < 8; ++token) {
        float max_abs = 0.0f;
        for (std::uint32_t offset = 0; offset < 16; ++offset) {
            const float value = key_tail[(tail_start + token) * width + head * head_dim + k_block * 16 + offset];
            if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
        }
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const float scale = infer_e4m3_value(scale_code);
        for (std::uint32_t offset = 0; offset < 16; ++offset) {
            const float value = key_tail[(tail_start + token) * width + head * head_dim + k_block * 16 + offset];
            infer_set_packed_nibble(packed, token * 64 + scale_block * 16 + offset,
                infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale));
        }
        key_scales[(tile * 8 + token) * 4 + scale_block] = scale_code;
    }
}

__global__ void infer_sm12x_kv_finalize_value_indexed_kernel(
    const float* __restrict__ value_tail,
    std::uint8_t* __restrict__ value_values,
    std::uint8_t* __restrict__ value_scales,
    const std::uint32_t* __restrict__ position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    const std::uint32_t pos = *position;
    if (pos >= max_tokens || (pos & 15u) != 15u) return;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t dim_tile = blockIdx.y;
    if (head >= kv_heads || dim_tile >= head_dim / 8 || threadIdx.x != 0) return;
    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t token_tiles = (max_tokens + 63) / 64;
    const std::uint32_t token_tile = pos / 64;
    const std::uint32_t scale_block = (pos & 63u) / 16;
    const std::uint32_t tile = (head * (head_dim / 8) + dim_tile) * token_tiles + token_tile;
    std::uint8_t* packed = value_values + tile * 256;
    for (std::uint32_t dim = 0; dim < 8; ++dim) {
        float max_abs = 0.0f;
        for (std::uint32_t token = 0; token < 16; ++token) {
            const float value = value_tail[token * width + head * head_dim + dim_tile * 8 + dim];
            if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
        }
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const float scale = infer_e4m3_value(scale_code);
        for (std::uint32_t token = 0; token < 16; ++token) {
            const float value = value_tail[token * width + head * head_dim + dim_tile * 8 + dim];
            infer_set_packed_nibble(packed, dim * 64 + scale_block * 16 + token,
                infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale));
        }
        value_scales[(tile * 8 + dim) * 4 + scale_block] = scale_code;
    }
}

extern "C" cudaError_t infer_sm12x_kv_cache_append_on_stream(
    const float* key,
    const float* value,
    std::uint8_t* key_values,
    std::uint8_t* key_scales,
    std::uint8_t* value_values,
    std::uint8_t* value_scales,
    float* key_tail,
    float* value_tail,
    std::uint32_t position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (key == nullptr || value == nullptr || key_values == nullptr || key_scales == nullptr ||
        value_values == nullptr || value_scales == nullptr || key_tail == nullptr ||
        value_tail == nullptr || position >= max_tokens || kv_heads == 0 || head_dim == 0 ||
        (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t width = kv_heads * head_dim;
    infer_sm12x_kv_copy_tail_kernel<<<(width + 255) / 256, 256, 0, stream>>>(
        key, value, key_tail, value_tail, position, width);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    if ((position & 7u) == 7u) {
        infer_sm12x_kv_finalize_key_kernel<<<dim3(kv_heads, head_dim / 16, 1), 1, 0, stream>>>(
            key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    if ((position & 15u) == 15u) {
        infer_sm12x_kv_finalize_value_kernel<<<dim3(kv_heads, head_dim / 8, 1), 1, 0, stream>>>(
            value_tail, value_values, value_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
    }
    return status;
}

__global__ void infer_sm12x_kv_finalize_key_rows_kernel(
    const float* __restrict__ key,
    std::uint8_t* __restrict__ key_values,
    std::uint8_t* __restrict__ key_scales,
    std::uint16_t* __restrict__ key_output,
    std::uint32_t output_tokens,
    std::uint32_t input_row_offset,
    std::uint32_t start_position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    __shared__ float values[16];
    __shared__ float scale;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t k_block = blockIdx.y;
    const std::uint32_t token_group = blockIdx.z;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t token_tiles = (max_tokens + 7) / 8;
    const std::uint32_t k_tiles = head_dim / 64;
    const std::uint32_t token_tile = start_position / 8 + token_group;
    const std::uint32_t k_tile = k_block / 4;
    const std::uint32_t scale_block = k_block & 3u;
    const std::uint32_t tile = (head * token_tiles + token_tile) * k_tiles + k_tile;
    std::uint8_t* packed = key_values + tile * 256;
    for (std::uint32_t token = 0; token < 8; ++token) {
        const float value = key[
            (input_row_offset + token_group * 8 + token) * width
            + head * head_dim + k_block * 16 + lane];
        values[lane] = isfinite(value) ? value : 0.0f;
        __syncthreads();
        if (lane == 0) {
            float max_abs = 0.0f;
            for (int index = 0; index < 16; ++index) {
                max_abs = fmaxf(max_abs, fabsf(values[index]));
            }
            const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            key_scales[(tile * 8 + token) * 4 + scale_block] = scale_code;
            scale = infer_e4m3_value(scale_code);
        }
        __syncthreads();
        if (lane < 8) {
            const std::uint8_t low = infer_e2m1_code(
                scale == 0.0f ? 0.0f : values[lane * 2] / scale);
            const std::uint8_t high = infer_e2m1_code(
                scale == 0.0f ? 0.0f : values[lane * 2 + 1] / scale);
            const std::uint32_t nibble = token * 64 + scale_block * 16 + lane * 2;
            packed[nibble / 2] = static_cast<std::uint8_t>(low | (high << 4));
        }
        if (key_output != nullptr) {
            const std::uint8_t code = infer_e2m1_code(
                scale == 0.0f ? 0.0f : values[lane] / scale);
            const float quantized = infer_e2m1_value(code) * scale;
            const __nv_bfloat16 bf16 = __float2bfloat16_rn(quantized);
            const std::uint32_t output_token = start_position + token_group * 8 + token;
            const std::uint32_t output_dim = k_block * 16 + lane;
            key_output[(head * output_tokens + output_token) * head_dim + output_dim] =
                *reinterpret_cast<const std::uint16_t*>(&bf16);
        }
        __syncthreads();
    }
}

__global__ void infer_sm12x_kv_finalize_value_rows_kernel(
    const float* __restrict__ value,
    std::uint8_t* __restrict__ value_values,
    std::uint8_t* __restrict__ value_scales,
    std::uint16_t* __restrict__ value_output,
    std::uint32_t output_tokens,
    std::uint32_t input_row_offset,
    std::uint32_t start_position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    __shared__ float values[16];
    __shared__ float scale;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t dim_tile = blockIdx.y;
    const std::uint32_t token_group = blockIdx.z;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t position = start_position + token_group * 16;
    const std::uint32_t context_tiles = (max_tokens + 63) / 64;
    const std::uint32_t token_tile = position / 64;
    const std::uint32_t scale_block = (position & 63u) / 16;
    const std::uint32_t tile =
        (head * (head_dim / 8) + dim_tile) * context_tiles + token_tile;
    std::uint8_t* packed = value_values + tile * 256;
    for (std::uint32_t dim = 0; dim < 8; ++dim) {
        const float element = value[
            (input_row_offset + token_group * 16 + lane) * width
            + head * head_dim + dim_tile * 8 + dim];
        values[lane] = isfinite(element) ? element : 0.0f;
        __syncthreads();
        if (lane == 0) {
            float max_abs = 0.0f;
            for (int index = 0; index < 16; ++index) {
                max_abs = fmaxf(max_abs, fabsf(values[index]));
            }
            const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
            value_scales[(tile * 8 + dim) * 4 + scale_block] = scale_code;
            scale = infer_e4m3_value(scale_code);
        }
        __syncthreads();
        if (lane < 8) {
            const std::uint8_t low = infer_e2m1_code(
                scale == 0.0f ? 0.0f : values[lane * 2] / scale);
            const std::uint8_t high = infer_e2m1_code(
                scale == 0.0f ? 0.0f : values[lane * 2 + 1] / scale);
            const std::uint32_t nibble = dim * 64 + scale_block * 16 + lane * 2;
            packed[nibble / 2] = static_cast<std::uint8_t>(low | (high << 4));
        }
        if (value_output != nullptr) {
            const std::uint8_t code = infer_e2m1_code(
                scale == 0.0f ? 0.0f : values[lane] / scale);
            const float quantized = infer_e2m1_value(code) * scale;
            const __nv_bfloat16 bf16 = __float2bfloat16_rn(quantized);
            const std::uint32_t output_token = position + lane;
            const std::uint32_t output_dim = dim_tile * 8 + dim;
            value_output[(head * head_dim + output_dim) * output_tokens + output_token] =
                *reinterpret_cast<const std::uint16_t*>(&bf16);
        }
        __syncthreads();
    }
}

__global__ void infer_sm12x_kv_stage_tail_bf16_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    std::uint16_t* __restrict__ key_output,
    std::uint16_t* __restrict__ value_output,
    std::uint32_t input_row_offset,
    std::uint32_t output_row_offset,
    std::uint32_t rows,
    std::uint32_t output_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t width = kv_heads * head_dim;
    if (index >= rows * width) return;
    const std::uint32_t dim = index % head_dim;
    const std::uint32_t head = (index / head_dim) % kv_heads;
    const std::uint32_t row = index / width;
    const std::uint32_t input_index =
        (input_row_offset + row) * width + head * head_dim + dim;
    const std::uint32_t output_token = output_row_offset + row;
    const __nv_bfloat16 key_bf16 = __float2bfloat16_rn(key[input_index]);
    const __nv_bfloat16 value_bf16 = __float2bfloat16_rn(value[input_index]);
    key_output[(head * output_tokens + output_token) * head_dim + dim] =
        *reinterpret_cast<const std::uint16_t*>(&key_bf16);
    value_output[(head * head_dim + dim) * output_tokens + output_token] =
        *reinterpret_cast<const std::uint16_t*>(&value_bf16);
}

extern "C" cudaError_t infer_sm12x_kv_cache_append_rows_on_stream(
    const float* key,
    const float* value,
    std::uint8_t* key_values,
    std::uint8_t* key_scales,
    std::uint8_t* value_values,
    std::uint8_t* value_scales,
    float* key_tail,
    float* value_tail,
    std::uint16_t* key_output,
    std::uint16_t* value_output,
    std::uint32_t output_tokens,
    std::uint32_t input_row_offset,
    std::uint32_t start_position,
    std::uint32_t rows,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (key == nullptr || value == nullptr || key_values == nullptr || key_scales == nullptr ||
        value_values == nullptr || value_scales == nullptr || key_tail == nullptr ||
        value_tail == nullptr || rows == 0 || start_position >= max_tokens ||
        rows > max_tokens - start_position || kv_heads == 0 || head_dim == 0 ||
        (head_dim % 64) != 0 ||
        ((key_output == nullptr) != (value_output == nullptr)) ||
        (key_output != nullptr && (start_position != 0 || output_tokens < rows))) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t width = kv_heads * head_dim;
    std::uint32_t processed = 0;
    if ((start_position & 15u) != 0) {
        const std::uint32_t position = start_position + processed;
        const std::uint32_t batch_rows = min(rows - processed, 16 - (position & 15u));
        const std::uint32_t input_offset = (input_row_offset + processed) * width;
        infer_sm12x_kv_copy_tail_kernel<<<
            dim3((width + 255) / 256, batch_rows, 1), 256, 0, stream>>>(
            key + input_offset, value + input_offset, key_tail, value_tail, position, width);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_key_kernel<<<
            dim3(kv_heads, head_dim / 16, batch_rows), 1, 0, stream>>>(
            key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_value_kernel<<<
            dim3(kv_heads, head_dim / 8, batch_rows), 1, 0, stream>>>(
            value_tail, value_values, value_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        processed += batch_rows;
    }
    const std::uint32_t bulk_rows = (rows - processed) / 16 * 16;
    if (bulk_rows != 0) {
        const std::uint32_t bulk_position = start_position + processed;
        const std::uint32_t bulk_input_row = input_row_offset + processed;
        infer_sm12x_kv_finalize_key_rows_kernel<<<
            dim3(kv_heads, head_dim / 16, bulk_rows / 8), 16, 0, stream>>>(
            key, key_values, key_scales, key_output, output_tokens,
            bulk_input_row, bulk_position,
            max_tokens, kv_heads, head_dim);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_value_rows_kernel<<<
            dim3(kv_heads, head_dim / 8, bulk_rows / 16), 16, 0, stream>>>(
            value, value_values, value_scales, value_output, output_tokens,
            bulk_input_row, bulk_position,
            max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        const std::uint32_t tail_input_row = bulk_input_row + bulk_rows - 16;
        const std::uint32_t tail_position = bulk_position + bulk_rows - 16;
        const std::uint32_t tail_input_offset = tail_input_row * width;
        infer_sm12x_kv_copy_tail_kernel<<<
            dim3((width + 255) / 256, 16, 1), 256, 0, stream>>>(
            key + tail_input_offset, value + tail_input_offset, key_tail,
            value_tail, tail_position, width);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        processed += bulk_rows;
    }
    if (processed < rows) {
        const std::uint32_t position = start_position + processed;
        const std::uint32_t batch_rows = rows - processed;
        const std::uint32_t input_offset = (input_row_offset + processed) * width;
        infer_sm12x_kv_copy_tail_kernel<<<
            dim3((width + 255) / 256, batch_rows, 1), 256, 0, stream>>>(
            key + input_offset, value + input_offset, key_tail, value_tail, position, width);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_key_kernel<<<
            dim3(kv_heads, head_dim / 16, batch_rows), 1, 0, stream>>>(
            key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_value_kernel<<<
            dim3(kv_heads, head_dim / 8, batch_rows), 1, 0, stream>>>(
            value_tail, value_values, value_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    if (key_output != nullptr && (rows & 15u) != 0) {
        const std::uint32_t tail_start = rows / 16 * 16;
        const std::uint32_t tail_rows = rows - tail_start;
        const std::uint32_t tail_values = tail_rows * width;
        infer_sm12x_kv_stage_tail_bf16_kernel<<<
            (tail_values + 255) / 256, 256, 0, stream>>>(
            key, value, key_output, value_output,
            input_row_offset + tail_start, tail_start, tail_rows,
            output_tokens, kv_heads, head_dim);
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

__global__ void infer_sm12x_kv_cache_unpack_bf16_kernel(
    const std::uint8_t* __restrict__ key_values,
    const std::uint8_t* __restrict__ key_scales,
    const std::uint8_t* __restrict__ value_values,
    const std::uint8_t* __restrict__ value_scales,
    const float* __restrict__ key_tail,
    const float* __restrict__ value_tail,
    std::uint16_t* __restrict__ key_output,
    std::uint16_t* __restrict__ value_output,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t total = cache_len * width;
    if (index >= total) return;
    const std::uint32_t dim = index % head_dim;
    const std::uint32_t head = (index / head_dim) % kv_heads;
    const std::uint32_t token = index / width;

    float key;
    const std::uint32_t compact_key_tokens = cache_len / 8 * 8;
    if (token < compact_key_tokens) {
        const std::uint32_t token_tiles = (max_tokens + 7) / 8;
        const std::uint32_t k_tiles = head_dim / 64;
        const std::uint32_t token_tile = token / 8;
        const std::uint32_t token_in_tile = token & 7u;
        const std::uint32_t k_tile = dim / 64;
        const std::uint32_t dim_in_tile = dim & 63u;
        const std::uint32_t tile =
            (head * token_tiles + token_tile) * k_tiles + k_tile;
        const std::uint8_t code = infer_get_packed_nibble(
            key_values + tile * 256, token_in_tile * 64 + dim_in_tile);
        const std::uint8_t scale_code =
            key_scales[(tile * 8 + token_in_tile) * 4 + dim_in_tile / 16];
        key = infer_e2m1_value(code) * infer_e4m3_value(scale_code);
    } else {
        key = key_tail[(token & 15u) * width + head * head_dim + dim];
    }

    float value;
    const std::uint32_t compact_value_tokens = cache_len / 16 * 16;
    if (token < compact_value_tokens) {
        const std::uint32_t context_tiles = (max_tokens + 63) / 64;
        const std::uint32_t dim_tile = dim / 8;
        const std::uint32_t dim_in_tile = dim & 7u;
        const std::uint32_t token_tile = token / 64;
        const std::uint32_t token_in_tile = token & 63u;
        const std::uint32_t tile =
            (head * (head_dim / 8) + dim_tile) * context_tiles + token_tile;
        const std::uint8_t code = infer_get_packed_nibble(
            value_values + tile * 256, dim_in_tile * 64 + token_in_tile);
        const std::uint8_t scale_code =
            value_scales[(tile * 8 + dim_in_tile) * 4 + token_in_tile / 16];
        value = infer_e2m1_value(code) * infer_e4m3_value(scale_code);
    } else {
        value = value_tail[(token & 15u) * width + head * head_dim + dim];
    }

    const __nv_bfloat16 key_bf16 = __float2bfloat16_rn(key);
    const __nv_bfloat16 value_bf16 = __float2bfloat16_rn(value);
    key_output[(head * cache_len + token) * head_dim + dim] =
        *reinterpret_cast<const std::uint16_t*>(&key_bf16);
    value_output[(head * head_dim + dim) * cache_len + token] =
        *reinterpret_cast<const std::uint16_t*>(&value_bf16);
}

extern "C" cudaError_t infer_sm12x_kv_cache_unpack_bf16_on_stream(
    const std::uint8_t* key_values,
    const std::uint8_t* key_scales,
    const std::uint8_t* value_values,
    const std::uint8_t* value_scales,
    const float* key_tail,
    const float* value_tail,
    std::uint16_t* key_output,
    std::uint16_t* value_output,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (key_values == nullptr || key_scales == nullptr || value_values == nullptr ||
        value_scales == nullptr || key_tail == nullptr || value_tail == nullptr ||
        key_output == nullptr || value_output == nullptr || cache_len == 0 ||
        cache_len > max_tokens || kv_heads == 0 || head_dim == 0 || (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const std::uint64_t total =
        static_cast<std::uint64_t>(cache_len) * kv_heads * head_dim;
    if (total > 0xffffffffu) return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    infer_sm12x_kv_cache_unpack_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
        key_values, key_scales, value_values, value_scales, key_tail, value_tail,
        key_output, value_output, cache_len, max_tokens, kv_heads, head_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_kv_cache_append_indexed_on_stream(
    const float* key,
    const float* value,
    std::uint8_t* key_values,
    std::uint8_t* key_scales,
    std::uint8_t* value_values,
    std::uint8_t* value_scales,
    float* key_tail,
    float* value_tail,
    const std::uint32_t* position,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (key == nullptr || value == nullptr || key_values == nullptr || key_scales == nullptr ||
        value_values == nullptr || value_scales == nullptr || key_tail == nullptr ||
        value_tail == nullptr || position == nullptr || max_tokens == 0 || kv_heads == 0 ||
        head_dim == 0 || (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t width = kv_heads * head_dim;
    infer_sm12x_kv_copy_tail_indexed_kernel<<<(width + 255) / 256, 256, 0, stream>>>(
        key, value, key_tail, value_tail, position, max_tokens, width);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_finalize_key_indexed_kernel<<<dim3(kv_heads, head_dim / 16, 1), 1, 0, stream>>>(
        key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_finalize_value_indexed_kernel<<<dim3(kv_heads, head_dim / 8, 1), 1, 0, stream>>>(
        value_tail, value_values, value_scales, position, max_tokens, kv_heads, head_dim);
    return cudaGetLastError();
}

__global__ void infer_sm12x_kv_quantize_query_kernel(
    const float* __restrict__ query,
    std::uint8_t* __restrict__ query_tiles,
    std::uint32_t* __restrict__ query_scales,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t input_row_offset)
{
    const std::uint32_t batch_row = blockIdx.z;
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t k_tile = blockIdx.y;
    const std::uint32_t queries_per_kv = q_heads / kv_heads;
    const std::uint32_t query_tiles_per_kv = (queries_per_kv + 7) / 8;
    const std::uint32_t query_groups = kv_heads * query_tiles_per_kv;
    const std::uint32_t head_k_tiles = head_dim / 64;
    query += (input_row_offset + batch_row) * q_heads * head_dim;
    query_tiles += batch_row * query_groups * head_k_tiles * 512;
    query_scales += batch_row * query_groups * head_k_tiles * 8;
    const std::uint32_t kv_head = group / query_tiles_per_kv;
    const std::uint32_t query_base =
        kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
    std::uint8_t* tile = query_tiles + (group * (head_dim / 64) + k_tile) * 512;
    __shared__ std::uint8_t scale_codes[8][4];
    if (threadIdx.x < 32) {
        const int row = threadIdx.x / 4;
        const int kb = threadIdx.x & 3;
            float max_abs = 0.0f;
            for (int offset = 0; offset < 16; ++offset) {
                const std::uint32_t q_head = query_base + row;
                const float value = q_head < (kv_head + 1) * queries_per_kv
                    ? query[q_head * head_dim + k_tile * 64 + kb * 16 + offset]
                    : 0.0f;
                if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
            }
        scale_codes[row][kb] = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
    }
    __syncthreads();
    for (int byte = threadIdx.x; byte < 512; byte += blockDim.x) {
        std::uint8_t packed = 0;
        for (int nibble = 0; nibble < 2; ++nibble) {
            const int index = byte * 2 + nibble;
            const int lane = index / 32;
            const int v = index & 31;
            const int t0 = lane & 3;
            const int t1 = lane >> 2;
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int v2 = (v >> 4) & 1;
            const int row = t1 + 8 * v1;
            const int col = t0 * 8 + v0 + 32 * v2;
            const int kb = col / 16;
            float value = 0.0f;
            const std::uint32_t q_head = query_base + row;
            if (row < 8 && q_head < (kv_head + 1) * queries_per_kv) {
                value = query[q_head * head_dim + k_tile * 64 + col];
            }
            const float scale = row < 8 ? infer_e4m3_value(scale_codes[row][kb]) : 0.0f;
            packed |= infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale) << (nibble * 4);
        }
        tile[byte] = packed;
    }
    const std::uint32_t tile_index = group * (head_dim / 64) + k_tile;
    if (threadIdx.x < 8) {
        const int row = threadIdx.x;
        query_scales[tile_index * 8 + row] = infer_scale_word(scale_codes[row]);
    }
}

__global__ void infer_sm12x_kv_qk_kernel(
    const std::uint8_t* __restrict__ query_tiles,
    const std::uint32_t* __restrict__ query_scales,
    const std::uint8_t* __restrict__ key_values,
    const std::uint8_t* __restrict__ key_scales,
    const float* __restrict__ key_tail,
    float* __restrict__ scores,
    std::uint32_t cache_len,
    const std::uint32_t* cache_len_device,
    std::uint32_t window_start,
    std::uint32_t max_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t causal_start_position,
    std::uint32_t window_tokens)
{
    const std::uint32_t batch_row = blockIdx.z;
    if (cache_len_device != nullptr) cache_len = *cache_len_device;
    if (causal_start_position != 0xffffffffu) {
        cache_len = causal_start_position + batch_row + 1;
        window_start = window_tokens == 0 || cache_len <= window_tokens
            ? 0
            : cache_len - window_tokens;
    }
    __shared__ __align__(16) std::uint8_t b_smem[512];
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t queries_per_kv = q_heads / kv_heads;
    const std::uint32_t query_tiles_per_kv = (queries_per_kv + 7) / 8;
    const std::uint32_t query_groups = kv_heads * query_tiles_per_kv;
    const std::uint32_t kv_head = group / query_tiles_per_kv;
    const std::uint32_t query_base =
        kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
    const std::uint32_t token_tile = blockIdx.y;
    if (token_tile * 8 + 7 < window_start) return;
    const std::uint32_t complete_tiles = cache_len / 8;
    const bool compact = token_tile < complete_tiles;
    const std::uint32_t tail_len = cache_len & 7u;
    const std::uint32_t head_k_tiles = head_dim / 64;
    query_tiles += batch_row * query_groups * head_k_tiles * 512;
    query_scales += batch_row * query_groups * head_k_tiles * 8;
    scores += batch_row * q_heads * max_tokens;
    const std::uint32_t max_token_tiles = (max_tokens + 7) / 8;
    if (token_tile >= (cache_len + 7) / 8) return;
    const std::uint32_t width = kv_heads * head_dim;
    const std::uint32_t tail_start = (complete_tiles * 8) & 15u;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (std::uint32_t kt = 0; kt < head_k_tiles; ++kt) {
        const std::uint8_t* a_tile = query_tiles + (group * head_k_tiles + kt) * 512;
        for (int index = threadIdx.x; index < 512; index += blockDim.x) {
            b_smem[index] = 0;
        }
        const std::uint8_t* compact_tile = nullptr;
        std::uint32_t tile = 0;
        if (compact) {
            tile = (kv_head * max_token_tiles + token_tile) * head_k_tiles + kt;
            compact_tile = key_values + tile * 256;
        }
        __syncthreads();

        const int lane = threadIdx.x;
        const int t0 = lane & 3;
        const int row = lane >> 2;
        std::uint8_t tail_scale_codes[4] = {};
        float tail_scales[4] = {};
        if (!compact && static_cast<std::uint32_t>(row) < tail_len) {
            for (int kb = 0; kb < 4; ++kb) {
                float max_abs = 0.0f;
                for (int offset = 0; offset < 16; ++offset) {
                    const float value = key_tail[(tail_start + row) * width + kv_head * head_dim + kt * 64 + kb * 16 + offset];
                    if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
                }
                tail_scale_codes[kb] = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
                tail_scales[kb] = infer_e4m3_value(tail_scale_codes[kb]);
            }
        }
        for (int v = 0; v < 16; ++v) {
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int col = t0 * 8 + v0 + 32 * v1;
            std::uint8_t code = 0;
            if (compact) {
                code = infer_get_packed_nibble(compact_tile, row * 64 + col);
            } else if (static_cast<std::uint32_t>(row) < tail_len) {
                const float value = key_tail[(tail_start + row) * width + kv_head * head_dim + kt * 64 + col];
                const float scale = tail_scales[col / 16];
                code = infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
            }
            infer_set_packed_nibble(b_smem + lane * 16, v, code);
        }
        __syncthreads();

        std::uint32_t a0;
        std::uint32_t a1;
        std::uint32_t a2;
        std::uint32_t a3;
        std::uint32_t b0;
        std::uint32_t b1;
        infer_load_native_m16n8k64(a_tile, b_smem, a0, a1, a2, a3, b0, b1);
        const std::uint32_t sfa = (lane & 3) == 0
            ? query_scales[((group * head_k_tiles + kt) * 8) + row]
            : 0;
        const std::uint32_t b_scale_word = compact
            ? infer_scale_word(key_scales + (tile * 8 + row) * 4)
            : infer_scale_word(tail_scale_codes);
        infer_mma_m16n8k64(
            a0, a1, a2, a3, b0, b1,
            sfa, b_scale_word,
            d0, d1, d2, d3);
        __syncthreads();
    }

    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 2;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const std::uint32_t q_head = query_base + row;
    const std::uint32_t token0 = token_tile * 8 + col;
    if (q_head < (kv_head + 1) * queries_per_kv) {
        if (token0 >= window_start && token0 < cache_len) {
            scores[q_head * max_tokens + token0] = d0 * scale;
        }
        if (token0 + 1 >= window_start && token0 + 1 < cache_len) {
            scores[q_head * max_tokens + token0 + 1] = d1 * scale;
        }
    }
}

struct InferOnlineSoftmaxState {
    float maximum;
    float sum;
};

__device__ __forceinline__ InferOnlineSoftmaxState infer_softmax_combine(
    InferOnlineSoftmaxState left, InferOnlineSoftmaxState right) {
    if (left.sum == 0.0f) return right;
    if (right.sum == 0.0f) return left;
    const float maximum = fmaxf(left.maximum, right.maximum);
    return {maximum,
        left.sum * expf(left.maximum - maximum) + right.sum * expf(right.maximum - maximum)};
}

__global__ void infer_sm12x_kv_softmax_kernel(
    float* scores, std::uint32_t cache_len, const std::uint32_t* cache_len_device,
    std::uint32_t window_start, std::uint32_t max_tokens, std::uint32_t q_heads,
    std::uint32_t causal_start_position, std::uint32_t window_tokens) {
    const std::uint32_t batch_row = blockIdx.y;
    if (cache_len_device != nullptr) cache_len = *cache_len_device;
    if (causal_start_position != 0xffffffffu) {
        cache_len = causal_start_position + batch_row + 1;
        window_start = window_tokens == 0 || cache_len <= window_tokens
            ? 0
            : cache_len - window_tokens;
    }
    __shared__ float maxima[256];
    __shared__ float sums[256];
    InferOnlineSoftmaxState state = {-INFINITY, 0.0f};
    float* row = scores + (batch_row * q_heads + blockIdx.x) * max_tokens;
    for (std::uint32_t token = window_start + threadIdx.x; token < cache_len;
         token += blockDim.x) {
        const float value = row[token];
        state = infer_softmax_combine(state, {value, 1.0f});
    }
    maxima[threadIdx.x] = state.maximum;
    sums[threadIdx.x] = state.sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const InferOnlineSoftmaxState combined = infer_softmax_combine(
                {maxima[threadIdx.x], sums[threadIdx.x]},
                {maxima[threadIdx.x + stride], sums[threadIdx.x + stride]});
            maxima[threadIdx.x] = combined.maximum;
            sums[threadIdx.x] = combined.sum;
        }
        __syncthreads();
    }
    const float maximum = maxima[0];
    const float inverse_sum = 1.0f / sums[0];
    for (std::uint32_t token = window_start + threadIdx.x; token < cache_len;
         token += blockDim.x) {
        row[token] = expf(row[token] - maximum) * inverse_sum;
    }
}

__global__ void infer_sm12x_kv_quantize_probability_kernel(
    const float* __restrict__ scores,
    std::uint8_t* __restrict__ probability_tiles,
    std::uint32_t* __restrict__ probability_scales,
    std::uint32_t cache_len,
    const std::uint32_t* cache_len_device,
    std::uint32_t window_start,
    std::uint32_t max_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t causal_start_position,
    std::uint32_t window_tokens)
{
    const std::uint32_t batch_row = blockIdx.z;
    if (cache_len_device != nullptr) cache_len = *cache_len_device;
    if (causal_start_position != 0xffffffffu) {
        cache_len = causal_start_position + batch_row + 1;
        window_start = window_tokens == 0 || cache_len <= window_tokens
            ? 0
            : cache_len - window_tokens;
    }
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t queries_per_kv = q_heads / kv_heads;
    const std::uint32_t query_tiles_per_kv = (queries_per_kv + 7) / 8;
    const std::uint32_t query_groups = kv_heads * query_tiles_per_kv;
    const std::uint32_t kv_head = group / query_tiles_per_kv;
    const std::uint32_t query_base =
        kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
    const std::uint32_t k_tile = blockIdx.y;
    const std::uint32_t context_tiles = (max_tokens + 63) / 64;
    scores += batch_row * q_heads * max_tokens;
    probability_tiles += batch_row * query_groups * context_tiles * 512;
    probability_scales += batch_row * query_groups * context_tiles * 8;
    if (k_tile >= (cache_len + 63) / 64) return;
    if (k_tile * 64 + 63 < window_start) return;
    const float amplification = infer_probability_amplification(cache_len - window_start);
    std::uint8_t* tile = probability_tiles + (group * context_tiles + k_tile) * 512;
    __shared__ std::uint8_t scale_codes[8][4];
    if (threadIdx.x < 32) {
        const int row = threadIdx.x / 4;
        const int kb = threadIdx.x & 3;
            float max_value = 0.0f;
            for (int offset = 0; offset < 16; ++offset) {
                const std::uint32_t token = k_tile * 64 + kb * 16 + offset;
                if (token >= window_start && token < cache_len) {
                    const std::uint32_t q_head = query_base + row;
                    if (q_head < (kv_head + 1) * queries_per_kv) {
                        max_value = fmaxf(max_value, scores[q_head * max_tokens + token]);
                    }
                }
            }
        scale_codes[row][kb] = max_value == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_value * amplification / 6.0f, __NV_SATFINITE, __NV_E4M3));
    }
    __syncthreads();
    for (int byte = threadIdx.x; byte < 512; byte += blockDim.x) {
        std::uint8_t packed = 0;
        for (int nibble = 0; nibble < 2; ++nibble) {
            const int index = byte * 2 + nibble;
            const int lane = index / 32;
            const int v = index & 31;
            const int t0 = lane & 3;
            const int t1 = lane >> 2;
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int v2 = (v >> 4) & 1;
            const int row = t1 + 8 * v1;
            const int col = t0 * 8 + v0 + 32 * v2;
            const std::uint32_t token = k_tile * 64 + col;
            float value = 0.0f;
            const std::uint32_t q_head = query_base + row;
            if (row < 8 && q_head < (kv_head + 1) * queries_per_kv &&
                token >= window_start && token < cache_len) {
                value = scores[q_head * max_tokens + token];
            }
            const float scale = row < 8 ? infer_e4m3_value(scale_codes[row][col / 16]) : 0.0f;
            packed |= infer_e2m1_code(scale == 0.0f ? 0.0f : value * amplification / scale)
                << (nibble * 4);
        }
        tile[byte] = packed;
    }
    const std::uint32_t tile_index = group * context_tiles + k_tile;
    if (threadIdx.x < 8) {
        const int row = threadIdx.x;
        probability_scales[tile_index * 8 + row] = infer_scale_word(scale_codes[row]);
    }
}

__global__ void infer_sm12x_kv_pv_kernel(
    const std::uint8_t* __restrict__ probability_tiles,
    const std::uint32_t* __restrict__ probability_scales,
    const std::uint8_t* __restrict__ value_values,
    const std::uint8_t* __restrict__ value_scales,
    const float* __restrict__ value_tail,
    float* __restrict__ output,
    std::uint32_t cache_len,
    const std::uint32_t* cache_len_device,
    std::uint32_t window_start,
    std::uint32_t max_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t causal_start_position,
    std::uint32_t window_tokens,
    std::uint32_t output_row_offset,
    float* __restrict__ partial_output,
    std::uint32_t pv_splits)
{
    const std::uint32_t batch_row = blockIdx.z / pv_splits;
    const std::uint32_t split = blockIdx.z % pv_splits;
    if (cache_len_device != nullptr) cache_len = *cache_len_device;
    if (causal_start_position != 0xffffffffu) {
        cache_len = causal_start_position + batch_row + 1;
        window_start = window_tokens == 0 || cache_len <= window_tokens
            ? 0
            : cache_len - window_tokens;
    }
    __shared__ __align__(16) std::uint8_t b_smem[512];
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t queries_per_kv = q_heads / kv_heads;
    const std::uint32_t query_tiles_per_kv = (queries_per_kv + 7) / 8;
    const std::uint32_t query_groups = kv_heads * query_tiles_per_kv;
    const std::uint32_t kv_head = group / query_tiles_per_kv;
    const std::uint32_t query_base =
        kv_head * queries_per_kv + (group % query_tiles_per_kv) * 8;
    const std::uint32_t dim_tile = blockIdx.y;
    const std::uint32_t context_tiles = (cache_len + 63) / 64;
    const std::uint32_t max_context_tiles = (max_tokens + 63) / 64;
    probability_tiles += batch_row * query_groups * max_context_tiles * 512;
    probability_scales += batch_row * query_groups * max_context_tiles * 8;
    float* destination = pv_splits == 1
        ? output + (output_row_offset + batch_row) * q_heads * head_dim
        : partial_output + (batch_row * pv_splits + split) * q_heads * head_dim;
    const std::uint32_t full_tokens = cache_len / 16 * 16;
    const std::uint32_t tail_len = cache_len & 15u;
    const std::uint32_t width = kv_heads * head_dim;
    const float probability_correction =
        infer_probability_amplification(cache_len - window_start);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    const std::uint32_t first_context_tile = window_start / 64;
    const std::uint32_t active_context_tiles = context_tiles - first_context_tile;
    const std::uint32_t context_tile_begin =
        first_context_tile + active_context_tiles * split / pv_splits;
    const std::uint32_t context_tile_end =
        first_context_tile + active_context_tiles * (split + 1) / pv_splits;
    for (std::uint32_t kt = context_tile_begin; kt < context_tile_end; ++kt) {
        const std::uint8_t* a_tile = probability_tiles + (group * max_context_tiles + kt) * 512;
        const std::uint32_t value_tile_index =
            (kv_head * (head_dim / 8) + dim_tile) * max_context_tiles + kt;
        const std::uint8_t* compact_tile = value_values + value_tile_index * 256;
        for (int index = threadIdx.x; index < 512; index += blockDim.x) {
            b_smem[index] = 0;
        }
        __syncthreads();

        const int lane = threadIdx.x;
        const int t0 = lane & 3;
        const int dim = lane >> 2;
        std::uint8_t b_scale_codes[4] = {};
        float tail_scales[4] = {};
        for (int kb = 0; kb < 4; ++kb) {
            const std::uint32_t block_start = kt * 64 + kb * 16;
            if (block_start + 16 <= full_tokens) {
                b_scale_codes[kb] = value_scales[(value_tile_index * 8 + dim) * 4 + kb];
            } else if (block_start == full_tokens && tail_len != 0) {
                float max_abs = 0.0f;
                for (std::uint32_t token = 0; token < tail_len; ++token) {
                    const float value = value_tail[token * width + kv_head * head_dim + dim_tile * 8 + dim];
                    if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
                }
                b_scale_codes[kb] = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
                tail_scales[kb] = infer_e4m3_value(b_scale_codes[kb]);
            }
        }
        for (int v = 0; v < 16; ++v) {
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int col = t0 * 8 + v0 + 32 * v1;
            const std::uint32_t token = kt * 64 + col;
            std::uint8_t code = 0;
            if (token < full_tokens) {
                code = infer_get_packed_nibble(compact_tile, dim * 64 + col);
            } else if (token < cache_len) {
                const float value = value_tail[(token - full_tokens) * width + kv_head * head_dim + dim_tile * 8 + dim];
                const float scale = tail_scales[col / 16];
                code = infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
            }
            infer_set_packed_nibble(b_smem + lane * 16, v, code);
        }
        __syncthreads();

        std::uint32_t a0;
        std::uint32_t a1;
        std::uint32_t a2;
        std::uint32_t a3;
        std::uint32_t b0;
        std::uint32_t b1;
        infer_load_native_m16n8k64(a_tile, b_smem, a0, a1, a2, a3, b0, b1);
        const std::uint32_t sfa = (lane & 3) == 0
            ? probability_scales[((group * max_context_tiles + kt) * 8) + (lane >> 2)]
            : 0;
        infer_mma_m16n8k64(
            a0, a1, a2, a3, b0, b1,
            sfa, infer_scale_word(b_scale_codes), d0, d1, d2, d3);
        __syncthreads();
    }

    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 2;
    const std::uint32_t q_head = query_base + row;
    if (q_head < (kv_head + 1) * queries_per_kv) {
        destination[q_head * head_dim + dim_tile * 8 + col] =
            d0 / probability_correction;
        destination[q_head * head_dim + dim_tile * 8 + col + 1] =
            d1 / probability_correction;
    }
}

__global__ void infer_sm12x_kv_pv_reduce_kernel(
    const float* __restrict__ partial_output,
    float* __restrict__ output,
    std::uint32_t pv_splits,
    std::uint32_t q_heads,
    std::uint32_t head_dim,
    std::uint32_t output_row_offset)
{
    const std::uint32_t batch_row = blockIdx.y;
    const std::uint32_t width = q_heads * head_dim;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) return;
    float sum = 0.0f;
    for (std::uint32_t split = 0; split < pv_splits; ++split) {
        sum += partial_output[(batch_row * pv_splits + split) * width + index];
    }
    output[(output_row_offset + batch_row) * width + index] = sum;
}

static cudaError_t infer_sm12x_kv_attention_impl(
    const float* query,
    const std::uint8_t* key_values,
    const std::uint8_t* key_scales,
    const float* key_tail,
    const std::uint8_t* value_values,
    const std::uint8_t* value_scales,
    const float* value_tail,
    std::uint8_t* query_tiles,
    std::uint32_t* query_scales,
    float* scores,
    std::uint8_t* probability_tiles,
    std::uint32_t* probability_scales,
    float* partial_output,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t window_start,
    std::uint32_t max_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t pv_splits,
    cudaStream_t stream)
{
    if (query == nullptr || key_values == nullptr || key_scales == nullptr || key_tail == nullptr ||
        value_values == nullptr || value_scales == nullptr || value_tail == nullptr ||
        query_tiles == nullptr || query_scales == nullptr || scores == nullptr ||
        probability_tiles == nullptr || probability_scales == nullptr || output == nullptr ||
        cache_len == 0 || cache_len > max_tokens || window_start >= cache_len ||
        q_heads == 0 || kv_heads == 0 ||
        (q_heads % kv_heads) != 0 || head_dim == 0 || (head_dim % 64) != 0 ||
        pv_splits == 0 || pv_splits > 32 ||
        (pv_splits > 1 && partial_output == nullptr)) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t head_k_tiles = head_dim / 64;
    const std::uint32_t token_tiles = (cache_len + 7) / 8;
    const std::uint32_t context_tiles = (cache_len + 63) / 64;
    const std::uint32_t query_groups = kv_heads * ((q_heads / kv_heads + 7) / 8);
    infer_sm12x_kv_quantize_query_kernel<<<dim3(query_groups, head_k_tiles, 1), 128, 0, stream>>>(
        query, query_tiles, query_scales, q_heads, kv_heads, head_dim, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_qk_kernel<<<dim3(query_groups, token_tiles, 1), 32, 0, stream>>>(
        query_tiles, query_scales, key_values, key_scales, key_tail, scores,
        cache_len, nullptr, window_start, max_tokens, q_heads, kv_heads, head_dim,
        0xffffffffu, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_softmax_kernel<<<q_heads, 256, 0, stream>>>(
        scores, cache_len, nullptr, window_start, max_tokens, q_heads, 0xffffffffu, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_quantize_probability_kernel<<<dim3(query_groups, context_tiles, 1), 128, 0, stream>>>(
        scores, probability_tiles, probability_scales, cache_len, nullptr, window_start, max_tokens,
        q_heads, kv_heads, 0xffffffffu, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_pv_kernel<<<dim3(query_groups, head_dim / 8, pv_splits), 32, 0, stream>>>(
        probability_tiles, probability_scales, value_values, value_scales, value_tail,
        output, cache_len, nullptr, window_start, max_tokens, q_heads, kv_heads, head_dim,
        0xffffffffu, 0, 0, partial_output, pv_splits);
    status = cudaGetLastError();
    if (status != cudaSuccess || pv_splits == 1) return status;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t output_values = q_heads * head_dim;
    infer_sm12x_kv_pv_reduce_kernel<<<
        dim3((output_values + kThreads - 1) / kThreads, 1, 1), kThreads, 0, stream>>>(
        partial_output, output, pv_splits, q_heads, head_dim, 0);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_kv_qk_on_stream(
    const float* query,
    const std::uint8_t* key_values,
    const std::uint8_t* key_scales,
    const float* key_tail,
    std::uint8_t* query_tiles,
    std::uint32_t* query_scales,
    float* scores,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (query == nullptr || key_values == nullptr || key_scales == nullptr ||
        key_tail == nullptr || query_tiles == nullptr || query_scales == nullptr ||
        scores == nullptr || cache_len == 0 || cache_len > max_tokens ||
        q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0 ||
        head_dim == 0 || (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t head_k_tiles = head_dim / 64;
    const std::uint32_t token_tiles = (cache_len + 7) / 8;
    const std::uint32_t query_groups = kv_heads * ((q_heads / kv_heads + 7) / 8);
    infer_sm12x_kv_quantize_query_kernel<<<
        dim3(query_groups, head_k_tiles, 1), 128, 0, stream>>>(
        query, query_tiles, query_scales, q_heads, kv_heads, head_dim, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_qk_kernel<<<dim3(query_groups, token_tiles, 1), 32, 0, stream>>>(
        query_tiles, query_scales, key_values, key_scales, key_tail, scores,
        cache_len, nullptr, 0, max_tokens, q_heads, kv_heads, head_dim,
        0xffffffffu, 0);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_kv_attention_on_stream(
    const float* query, const std::uint8_t* key_values, const std::uint8_t* key_scales,
    const float* key_tail, const std::uint8_t* value_values,
    const std::uint8_t* value_scales, const float* value_tail,
    std::uint8_t* query_tiles, std::uint32_t* query_scales, float* scores,
    std::uint8_t* probability_tiles, std::uint32_t* probability_scales,
    float* partial_output, float* output,
    std::uint32_t cache_len, std::uint32_t max_tokens, std::uint32_t q_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t pv_splits,
    cudaStream_t stream) {
    return infer_sm12x_kv_attention_impl(
        query, key_values, key_scales, key_tail, value_values, value_scales, value_tail,
        query_tiles, query_scales, scores, probability_tiles, probability_scales,
        partial_output, output, cache_len, 0, max_tokens, q_heads, kv_heads, head_dim,
        pv_splits, stream);
}

extern "C" cudaError_t infer_sm12x_kv_attention_window_on_stream(
    const float* query, const std::uint8_t* key_values, const std::uint8_t* key_scales,
    const float* key_tail, const std::uint8_t* value_values,
    const std::uint8_t* value_scales, const float* value_tail,
    std::uint8_t* query_tiles, std::uint32_t* query_scales, float* scores,
    std::uint8_t* probability_tiles, std::uint32_t* probability_scales,
    float* partial_output, float* output,
    std::uint32_t cache_len, std::uint32_t window_start, std::uint32_t max_tokens,
    std::uint32_t q_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t pv_splits, cudaStream_t stream) {
    return infer_sm12x_kv_attention_impl(
        query, key_values, key_scales, key_tail, value_values, value_scales, value_tail,
        query_tiles, query_scales, scores, probability_tiles, probability_scales,
        partial_output, output, cache_len, window_start, max_tokens, q_heads, kv_heads,
        head_dim, pv_splits, stream);
}

extern "C" cudaError_t infer_sm12x_kv_append_causal_attention_rows_on_stream(
    const float* query,
    const float* key,
    const float* value,
    std::uint8_t* key_values,
    std::uint8_t* key_scales,
    std::uint8_t* value_values,
    std::uint8_t* value_scales,
    float* key_tail,
    float* value_tail,
    std::uint8_t* query_tiles,
    std::uint32_t* query_scales,
    float* scores,
    std::uint8_t* probability_tiles,
    std::uint32_t* probability_scales,
    float* output,
    std::uint32_t input_row_offset,
    std::uint32_t start_position,
    std::uint32_t rows,
    std::uint32_t max_tokens,
    std::uint32_t q_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t window_tokens,
    std::uint32_t workspace_rows,
    cudaStream_t stream)
{
    if (query == nullptr || key == nullptr || value == nullptr || key_values == nullptr ||
        key_scales == nullptr || value_values == nullptr || value_scales == nullptr ||
        key_tail == nullptr || value_tail == nullptr || query_tiles == nullptr ||
        query_scales == nullptr || scores == nullptr || probability_tiles == nullptr ||
        probability_scales == nullptr || output == nullptr || rows == 0 ||
        start_position >= max_tokens || rows > max_tokens - start_position ||
        q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0 || workspace_rows == 0 ||
        head_dim == 0 || (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t kv_width = kv_heads * head_dim;
    const std::uint32_t head_k_tiles = head_dim / 64;
    const std::uint32_t query_groups = kv_heads * ((q_heads / kv_heads + 7) / 8);
    std::uint32_t processed = 0;
    while (processed < rows) {
        const std::uint32_t position = start_position + processed;
        const std::uint32_t until_tail_wrap = 16 - (position & 15u);
        const std::uint32_t batch_rows = min(min(rows - processed, workspace_rows), until_tail_wrap);
        const std::uint32_t input_row = input_row_offset + processed;
        infer_sm12x_kv_copy_tail_kernel<<<
            dim3((kv_width + 255) / 256, batch_rows, 1), 256, 0, stream>>>(
            key + input_row * kv_width, value + input_row * kv_width,
            key_tail, value_tail, position, kv_width);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_key_kernel<<<
            dim3(kv_heads, head_dim / 16, batch_rows), 1, 0, stream>>>(
            key_tail, key_values, key_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_finalize_value_kernel<<<
            dim3(kv_heads, head_dim / 8, batch_rows), 1, 0, stream>>>(
            value_tail, value_values, value_scales, position, max_tokens, kv_heads, head_dim);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;

        const std::uint32_t final_cache_len = position + batch_rows;
        const std::uint32_t token_tiles = (final_cache_len + 7) / 8;
        const std::uint32_t context_tiles = (final_cache_len + 63) / 64;
        infer_sm12x_kv_quantize_query_kernel<<<
            dim3(query_groups, head_k_tiles, batch_rows), 128, 0, stream>>>(
            query, query_tiles, query_scales, q_heads, kv_heads, head_dim, input_row);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_qk_kernel<<<dim3(query_groups, token_tiles, batch_rows), 32, 0, stream>>>(
            query_tiles, query_scales, key_values, key_scales, key_tail, scores,
            0, nullptr, 0, max_tokens, q_heads, kv_heads, head_dim, position, window_tokens);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_softmax_kernel<<<dim3(q_heads, batch_rows, 1), 256, 0, stream>>>(
            scores, 0, nullptr, 0, max_tokens, q_heads, position, window_tokens);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_quantize_probability_kernel<<<
            dim3(query_groups, context_tiles, batch_rows), 128, 0, stream>>>(
            scores, probability_tiles, probability_scales, 0, nullptr,
            0, max_tokens, q_heads, kv_heads, position, window_tokens);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        infer_sm12x_kv_pv_kernel<<<
            dim3(query_groups, head_dim / 8, batch_rows), 32, 0, stream>>>(
            probability_tiles, probability_scales, value_values, value_scales, value_tail,
            output, 0, nullptr, 0, max_tokens, q_heads, kv_heads, head_dim,
            position, window_tokens, input_row, nullptr, 1);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        processed += batch_rows;
    }
    return cudaSuccess;
}

extern "C" cudaError_t infer_sm12x_kv_attention_indexed_on_stream(
    const float* query,
    const std::uint8_t* key_values,
    const std::uint8_t* key_scales,
    const float* key_tail,
    const std::uint8_t* value_values,
    const std::uint8_t* value_scales,
    const float* value_tail,
    std::uint8_t* query_tiles,
    std::uint32_t* query_scales,
    float* scores,
    std::uint8_t* probability_tiles,
    std::uint32_t* probability_scales,
    float* partial_output,
    float* output,
    const std::uint32_t* cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t pv_splits,
    cudaStream_t stream)
{
    if (query == nullptr || key_values == nullptr || key_scales == nullptr || key_tail == nullptr ||
        value_values == nullptr || value_scales == nullptr || value_tail == nullptr ||
        query_tiles == nullptr || query_scales == nullptr || scores == nullptr ||
        probability_tiles == nullptr || probability_scales == nullptr ||
        partial_output == nullptr || output == nullptr ||
        cache_len == nullptr || max_tokens == 0 || kv_heads == 0 || head_dim == 0 ||
        (head_dim % 64) != 0 || pv_splits < 2 || pv_splits > 32) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t head_k_tiles = head_dim / 64;
    const std::uint32_t max_token_tiles = (max_tokens + 7) / 8;
    const std::uint32_t max_context_tiles = (max_tokens + 63) / 64;
    infer_sm12x_kv_quantize_query_kernel<<<dim3(kv_heads, head_k_tiles, 1), 128, 0, stream>>>(
        query, query_tiles, query_scales, kv_heads * 8, kv_heads, head_dim, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_qk_kernel<<<dim3(kv_heads, max_token_tiles, 1), 32, 0, stream>>>(
        query_tiles, query_scales, key_values, key_scales, key_tail, scores,
        0, cache_len, 0, max_tokens, kv_heads * 8, kv_heads, head_dim,
        0xffffffffu, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_softmax_kernel<<<kv_heads * 8, 256, 0, stream>>>(
        scores, 0, cache_len, 0, max_tokens, kv_heads * 8, 0xffffffffu, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_quantize_probability_kernel<<<dim3(kv_heads, max_context_tiles, 1), 128, 0, stream>>>(
        scores, probability_tiles, probability_scales, 0, cache_len, 0, max_tokens,
        kv_heads * 8, kv_heads, 0xffffffffu, 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_pv_kernel<<<dim3(kv_heads, head_dim / 8, pv_splits), 32, 0, stream>>>(
        probability_tiles, probability_scales, value_values, value_scales, value_tail,
        output, 0, cache_len, 0, max_tokens, kv_heads * 8, kv_heads, head_dim,
        0xffffffffu, 0, 0, partial_output, pv_splits);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t q_heads = kv_heads * 8;
    const std::uint32_t output_values = q_heads * head_dim;
    infer_sm12x_kv_pv_reduce_kernel<<<
        dim3((output_values + kThreads - 1) / kThreads, 1, 1), kThreads, 0, stream>>>(
        partial_output, output, pv_splits, q_heads, head_dim, 0);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_kv_pv_from_probabilities_on_stream(
    const float* probabilities,
    const std::uint8_t* value_values,
    const std::uint8_t* value_scales,
    const float* value_tail,
    std::uint8_t* probability_tiles,
    std::uint32_t* probability_scales,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (probabilities == nullptr || value_values == nullptr || value_scales == nullptr ||
        value_tail == nullptr || probability_tiles == nullptr ||
        probability_scales == nullptr || output == nullptr || cache_len == 0 ||
        cache_len > max_tokens || kv_heads == 0 || head_dim == 0 ||
        (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t context_tiles = (cache_len + 63) / 64;
    infer_sm12x_kv_quantize_probability_kernel<<<dim3(kv_heads, context_tiles, 1), 128, 0, stream>>>(
        probabilities, probability_tiles, probability_scales, cache_len, nullptr,
        0, max_tokens, kv_heads * 8, kv_heads, 0xffffffffu, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_pv_kernel<<<dim3(kv_heads, head_dim / 8, 1), 32, 0, stream>>>(
        probability_tiles, probability_scales, value_values, value_scales, value_tail,
        output, cache_len, nullptr, 0, max_tokens, kv_heads * 8, kv_heads, head_dim,
        0xffffffffu, 0, 0, nullptr, 1);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_kv_pv_from_probabilities_split_on_stream(
    const float* probabilities,
    const std::uint8_t* value_values,
    const std::uint8_t* value_scales,
    const float* value_tail,
    std::uint8_t* probability_tiles,
    std::uint32_t* probability_scales,
    float* partial_output,
    float* output,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    std::uint32_t pv_splits,
    cudaStream_t stream)
{
    if (probabilities == nullptr || value_values == nullptr || value_scales == nullptr ||
        value_tail == nullptr || probability_tiles == nullptr ||
        probability_scales == nullptr || partial_output == nullptr || output == nullptr ||
        cache_len == 0 || cache_len > max_tokens || kv_heads == 0 ||
        head_dim == 0 || (head_dim % 64) != 0 || pv_splits < 2 || pv_splits > 32) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t q_heads = kv_heads * 8;
    const std::uint32_t context_tiles = (cache_len + 63) / 64;
    infer_sm12x_kv_quantize_probability_kernel<<<
        dim3(kv_heads, context_tiles, 1), 128, 0, stream>>>(
        probabilities, probability_tiles, probability_scales, cache_len, nullptr,
        0, max_tokens, q_heads, kv_heads, 0xffffffffu, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_pv_kernel<<<
        dim3(kv_heads, head_dim / 8, pv_splits), 32, 0, stream>>>(
        probability_tiles, probability_scales, value_values, value_scales, value_tail,
        output, cache_len, nullptr, 0, max_tokens, q_heads, kv_heads, head_dim,
        0xffffffffu, 0, 0, partial_output, pv_splits);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    constexpr std::uint32_t kThreads = 256;
    const std::uint32_t output_values = q_heads * head_dim;
    infer_sm12x_kv_pv_reduce_kernel<<<
        dim3((output_values + kThreads - 1) / kThreads, 1, 1), kThreads, 0, stream>>>(
        partial_output, output, pv_splits, q_heads, head_dim, 0);
    return cudaGetLastError();
}

__global__ void infer_sm12x_moe_silu_quantize_slots_reference_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* const* __restrict__ gate_up_table,
    std::uint8_t* __restrict__ b_native_tiles,
    std::uint32_t* __restrict__ sfb,
    const float* __restrict__ input_scale_table,
    const float* __restrict__ gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t k_tiles,
    std::uint32_t groups)
{
    const std::uint32_t slot = blockIdx.x;
    const std::uint32_t kt = blockIdx.y;
    if (slot >= groups || kt >= k_tiles) return;

    std::uint8_t* tile = b_native_tiles + (slot * k_tiles + kt) * 512;
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        tile[idx] = 0;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const std::uint32_t expert = indices[slot];
    const float input_scale = input_scale_table[expert];
    if (input_scale <= 0.0f || !isfinite(input_scale)) return;
    const float gate_up_alpha = gate_up_alpha_table[expert];
    const float* gate_up = gate_up_table[slot];
    std::uint8_t codes[64];
    std::uint8_t scale_codes[4];

    for (int kb = 0; kb < 4; ++kb) {
        const std::uint32_t start = kt * 64 + kb * 16;
        float max_abs = 0.0f;
        for (int offset = 0; offset < 16; ++offset) {
            const std::uint32_t row = start + offset;
            if (row < rows) {
                const float gate_value = gate_up[row] * gate_up_alpha;
                const float up_value = gate_up[rows + row] * gate_up_alpha;
                const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
                const float value = (gate_value * sigmoid * up_value) / input_scale;
                if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
            }
        }
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const float scale = infer_e4m3_value(scale_code);
        scale_codes[kb] = scale_code;
        for (int offset = 0; offset < 16; ++offset) {
            const std::uint32_t row = start + offset;
            float scaled = 0.0f;
            if (row < rows && scale != 0.0f) {
                const float gate_value = gate_up[row] * gate_up_alpha;
                const float up_value = gate_up[rows + row] * gate_up_alpha;
                const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
                scaled = ((gate_value * sigmoid * up_value) / input_scale) / scale;
            }
            codes[kb * 16 + offset] = infer_e2m1_code(scaled);
        }
    }

    for (int lane = 0; lane < 32; ++lane) {
        const int t0 = lane & 3;
        for (int v = 0; v < 16; ++v) {
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int col = t0 * 8 + v0 + 32 * v1;
            infer_set_packed_nibble(tile, lane * 32 + v, codes[col]);
        }
    }
    sfb[slot * k_tiles + kt] = static_cast<std::uint32_t>(scale_codes[0]) |
        (static_cast<std::uint32_t>(scale_codes[1]) << 8) |
        (static_cast<std::uint32_t>(scale_codes[2]) << 16) |
        (static_cast<std::uint32_t>(scale_codes[3]) << 24);
}

extern "C" cudaError_t infer_sm12x_moe_silu_quantize_slots_reference_on_stream(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* b_native_tiles,
    std::uint32_t* sfb,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || gate_up_table == nullptr || b_native_tiles == nullptr || sfb == nullptr || input_scale_table == nullptr || gate_up_alpha_table == nullptr || rows == 0 || groups == 0 || (rows % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t k_tiles = rows / 64;
    infer_sm12x_moe_silu_quantize_slots_reference_kernel<<<dim3(groups, k_tiles, 1), 128, 0, stream>>>(indices, gate_up_table, b_native_tiles, sfb, input_scale_table, gate_up_alpha_table, rows, k_tiles, groups);
    return cudaGetLastError();
}

template <bool Bf16Input>
__global__ void infer_sm12x_moe_silu_quantize_slots_kernel(
    const std::uint32_t* __restrict__ indices,
    const std::uint32_t* __restrict__ sorted_routes,
    const std::uint32_t* __restrict__ sorted_experts,
    const float* const* __restrict__ gate_up_table,
    const std::uint16_t* __restrict__ gate_up_bf16,
    std::uint8_t* __restrict__ b_native_tiles,
    std::uint32_t* __restrict__ sfb,
    const float* __restrict__ input_scale_table,
    const float* __restrict__ gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t k_tiles,
    std::uint32_t groups)
{
    const std::uint32_t slot = blockIdx.x;
    const std::uint32_t kt = blockIdx.y;
    if (slot >= groups || kt >= k_tiles) return;

    __shared__ float values[64];
    __shared__ std::uint8_t codes[64];
    __shared__ std::uint8_t scale_codes[4];
    __shared__ float scales[4];

    std::uint8_t* tile = b_native_tiles + (slot * k_tiles + kt) * 512;
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        tile[idx] = 0;
    }

    const std::uint32_t route = sorted_routes == nullptr ? slot : sorted_routes[slot];
    const std::uint32_t expert =
        sorted_experts == nullptr ? indices[route] : sorted_experts[slot];
    const float input_scale = input_scale_table[expert];
    if (input_scale <= 0.0f || !isfinite(input_scale)) return;
    const float gate_up_alpha = gate_up_alpha_table[expert];
    const float* gate_up = Bf16Input ? nullptr : gate_up_table[route];
    const int scale_group = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    float value = 0.0f;
    if (lane < 16) {
        const std::uint32_t row = kt * 64 + scale_group * 16 + lane;
        if (row < rows) {
            const std::uint32_t base = route * rows * 2;
            const float gate_value = (Bf16Input
                    ? __bfloat162float(__ushort_as_bfloat16(gate_up_bf16[base + row]))
                    : gate_up[row]) * gate_up_alpha;
            const float up_value = (Bf16Input
                    ? __bfloat162float(__ushort_as_bfloat16(gate_up_bf16[base + rows + row]))
                    : gate_up[rows + row]) * gate_up_alpha;
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            value = (gate_value * sigmoid * up_value) / input_scale;
        }
        values[scale_group * 16 + lane] = value;
    }

    float max_abs = lane < 16 && isfinite(value) ? fabsf(value) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        scale_codes[scale_group] = scale_code;
        scales[scale_group] = infer_e4m3_value(scale_code);
    }
    __syncthreads();

    if (lane < 16) {
        const float scale = scales[scale_group];
        const float scaled = scale == 0.0f ? 0.0f : values[scale_group * 16 + lane] / scale;
        codes[scale_group * 16 + lane] = infer_e2m1_code(scaled);
    }
    __syncthreads();

    // Each packed byte contains two adjacent logical nibbles. Writing whole
    // bytes avoids the read-modify-write races of parallel nibble stores.
    for (int packed_idx = threadIdx.x; packed_idx < 256; packed_idx += blockDim.x) {
        const int output_lane = packed_idx >> 3;
        const int pair = packed_idx & 7;
        const int v = pair << 1;
        const int t0 = output_lane & 3;
        const int col0 = t0 * 8 + (v & 7) + 32 * ((v >> 3) & 1);
        const int next_v = v + 1;
        const int col1 = t0 * 8 + (next_v & 7) + 32 * ((next_v >> 3) & 1);
        tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            codes[col0] | (codes[col1] << 4));
    }
    if (threadIdx.x == 0) {
        sfb[slot * k_tiles + kt] = static_cast<std::uint32_t>(scale_codes[0]) |
            (static_cast<std::uint32_t>(scale_codes[1]) << 8) |
            (static_cast<std::uint32_t>(scale_codes[2]) << 16) |
            (static_cast<std::uint32_t>(scale_codes[3]) << 24);
    }
}

extern "C" cudaError_t infer_sm12x_moe_silu_quantize_slots_on_stream(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* b_native_tiles,
    std::uint32_t* sfb,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || gate_up_table == nullptr || b_native_tiles == nullptr || sfb == nullptr || input_scale_table == nullptr || gate_up_alpha_table == nullptr || rows == 0 || groups == 0 || (rows % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t k_tiles = rows / 64;
    infer_sm12x_moe_silu_quantize_slots_kernel<false><<<dim3(groups, k_tiles, 1), 128, 0, stream>>>(
        indices, nullptr, nullptr, gate_up_table, nullptr, b_native_tiles, sfb,
        input_scale_table, gate_up_alpha_table, rows, k_tiles, groups);
    return cudaGetLastError();
}

__global__ void infer_sm12x_moe_silu_quantize_slots_residual_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* const* __restrict__ gate_up_table,
    std::uint8_t* __restrict__ primary_tiles,
    std::uint32_t* __restrict__ primary_scales,
    std::uint8_t* __restrict__ residual_tiles,
    std::uint32_t* __restrict__ residual_scales,
    const float* __restrict__ gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t k_tiles,
    std::uint32_t groups,
    float swiglu_limit)
{
    const std::uint32_t slot = blockIdx.x;
    const std::uint32_t kt = blockIdx.y;
    if (slot >= groups || kt >= k_tiles) return;

    __shared__ float values[64];
    __shared__ float residuals[64];
    __shared__ std::uint8_t primary_codes[64];
    __shared__ std::uint8_t residual_codes[64];
    __shared__ std::uint8_t primary_scale_codes[4];
    __shared__ std::uint8_t residual_scale_codes[4];
    __shared__ float primary_scale_values[4];
    __shared__ float residual_scale_values[4];

    std::uint8_t* primary_tile = primary_tiles + (slot * k_tiles + kt) * 512;
    std::uint8_t* residual_tile = residual_tiles + (slot * k_tiles + kt) * 512;
    for (int index = threadIdx.x; index < 512; index += blockDim.x) {
        primary_tile[index] = 0;
        residual_tile[index] = 0;
    }

    const std::uint32_t expert = indices[slot];
    const float gate_up_alpha = gate_up_alpha_table[expert];
    const float* gate_up = gate_up_table[slot];
    const int scale_group = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    float value = 0.0f;
    if (lane < 16) {
        const std::uint32_t row = kt * 64 + scale_group * 16 + lane;
        if (row < rows) {
            float gate_value = gate_up[row] * gate_up_alpha;
            float up_value = gate_up[rows + row] * gate_up_alpha;
            if (swiglu_limit > 0.0f) {
                gate_value = fminf(gate_value, swiglu_limit);
                up_value = fminf(fmaxf(up_value, -swiglu_limit), swiglu_limit);
            }
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            value = gate_value * sigmoid * up_value;
        }
        values[scale_group * 16 + lane] = value;
    }

    float max_abs = lane < 16 && isfinite(value) ? fabsf(value) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        primary_scale_codes[scale_group] = scale_code;
        primary_scale_values[scale_group] = infer_e4m3_value(scale_code);
    }
    __syncthreads();

    float residual = 0.0f;
    if (lane < 16) {
        const int index = scale_group * 16 + lane;
        const float scale = primary_scale_values[scale_group];
        const std::uint8_t code = infer_e2m1_code(
            scale == 0.0f ? 0.0f : values[index] / scale);
        primary_codes[index] = code;
        residual = values[index] - infer_e2m1_value(code) * scale;
        residuals[index] = residual;
    }

    max_abs = lane < 16 && isfinite(residual) ? fabsf(residual) : 0.0f;
    for (int delta = 8; delta > 0; delta >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, delta));
    }
    if (lane == 0) {
        const std::uint8_t scale_code = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        residual_scale_codes[scale_group] = scale_code;
        residual_scale_values[scale_group] = infer_e4m3_value(scale_code);
    }
    __syncthreads();

    if (lane < 16) {
        const int index = scale_group * 16 + lane;
        const float scale = residual_scale_values[scale_group];
        residual_codes[index] = infer_e2m1_code(
            scale == 0.0f ? 0.0f : residuals[index] / scale);
    }
    __syncthreads();

    for (int packed_idx = threadIdx.x; packed_idx < 256; packed_idx += blockDim.x) {
        const int output_lane = packed_idx >> 3;
        const int pair = packed_idx & 7;
        const int v = pair << 1;
        const int t0 = output_lane & 3;
        const int col0 = t0 * 8 + (v & 7) + 32 * ((v >> 3) & 1);
        const int next_v = v + 1;
        const int col1 = t0 * 8 + (next_v & 7) + 32 * ((next_v >> 3) & 1);
        primary_tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            primary_codes[col0] | (primary_codes[col1] << 4));
        residual_tile[output_lane * 16 + pair] = static_cast<std::uint8_t>(
            residual_codes[col0] | (residual_codes[col1] << 4));
    }
    if (threadIdx.x == 0) {
        primary_scales[slot * k_tiles + kt] = infer_scale_word(primary_scale_codes);
        residual_scales[slot * k_tiles + kt] = infer_scale_word(residual_scale_codes);
    }
}

extern "C" cudaError_t infer_sm12x_moe_silu_quantize_slots_residual_on_stream(
    const std::uint32_t* indices,
    const float* const* gate_up_table,
    std::uint8_t* primary_tiles,
    std::uint32_t* primary_scales,
    std::uint8_t* residual_tiles,
    std::uint32_t* residual_scales,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    float swiglu_limit,
    cudaStream_t stream)
{
    if (indices == nullptr || gate_up_table == nullptr ||
        primary_tiles == nullptr || primary_scales == nullptr ||
        residual_tiles == nullptr || residual_scales == nullptr ||
        gate_up_alpha_table == nullptr || !isfinite(swiglu_limit) ||
        rows == 0 || groups == 0 || (rows % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t k_tiles = rows / 64;
    infer_sm12x_moe_silu_quantize_slots_residual_kernel<<<
        dim3(groups, k_tiles, 1), 128, 0, stream>>>(
        indices, gate_up_table, primary_tiles, primary_scales,
        residual_tiles, residual_scales, gate_up_alpha_table, rows, k_tiles,
        groups, swiglu_limit);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_moe_silu_quantize_bf16_slots_on_stream(
    const std::uint32_t* indices,
    const std::uint16_t* gate_up_bf16,
    std::uint8_t* b_native_tiles,
    std::uint32_t* sfb,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || gate_up_bf16 == nullptr || b_native_tiles == nullptr || sfb == nullptr || input_scale_table == nullptr || gate_up_alpha_table == nullptr || rows == 0 || groups == 0 || (rows % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t k_tiles = rows / 64;
    infer_sm12x_moe_silu_quantize_slots_kernel<true><<<dim3(groups, k_tiles, 1), 128, 0, stream>>>(
        indices, nullptr, nullptr, nullptr, gate_up_bf16, b_native_tiles, sfb,
        input_scale_table, gate_up_alpha_table, rows, k_tiles, groups);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_moe_silu_quantize_bf16_sorted_slots_on_stream(
    const std::uint32_t* indices,
    const std::uint32_t* sorted_routes,
    const std::uint32_t* sorted_experts,
    const std::uint16_t* gate_up_bf16,
    std::uint8_t* b_native_tiles,
    std::uint32_t* sfb,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || sorted_routes == nullptr || sorted_experts == nullptr ||
        gate_up_bf16 == nullptr || b_native_tiles == nullptr || sfb == nullptr ||
        input_scale_table == nullptr || gate_up_alpha_table == nullptr ||
        rows == 0 || groups == 0 || (rows % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t k_tiles = rows / 64;
    infer_sm12x_moe_silu_quantize_slots_kernel<true><<<
        dim3(groups, k_tiles, 1), 128, 0, stream>>>(
        indices, sorted_routes, sorted_experts, nullptr, gate_up_bf16,
        b_native_tiles, sfb, input_scale_table, gate_up_alpha_table,
        rows, k_tiles, groups);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_moe_silu_quantize_bf16_expert_sorted_slots_on_stream(
    const std::uint32_t* sorted_experts,
    const std::uint16_t* gate_up_bf16,
    std::uint8_t* b_native_tiles,
    std::uint32_t* sfb,
    const float* input_scale_table,
    const float* gate_up_alpha_table,
    std::uint32_t rows,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (sorted_experts == nullptr || gate_up_bf16 == nullptr ||
        b_native_tiles == nullptr || sfb == nullptr ||
        input_scale_table == nullptr || gate_up_alpha_table == nullptr ||
        rows == 0 || groups == 0 || (rows % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t k_tiles = rows / 64;
    infer_sm12x_moe_silu_quantize_slots_kernel<true><<<
        dim3(groups, k_tiles, 1), 128, 0, stream>>>(
        sorted_experts, nullptr, nullptr, nullptr, gate_up_bf16,
        b_native_tiles, sfb, input_scale_table, gate_up_alpha_table,
        rows, k_tiles, groups);
    return cudaGetLastError();
}

__global__ void infer_sm12x_indexed_gemv_kernel(
    const std::uint32_t* __restrict__ indices,
    const std::uint8_t* const* __restrict__ a_native_tiles_table,
    const std::uint32_t* const* __restrict__ a_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* __restrict__ b_native_tiles,
    const std::uint32_t* __restrict__ sfb,
    float* const* __restrict__ d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups)
{
    const std::uint32_t m_tile = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (m_tile >= m_tiles || group >= groups) return;
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) return;

    const std::uint8_t* a_native_tiles = a_native_tiles_table[expert];
    const std::uint32_t* sfa = a_scales_table[expert];
    float* out = d[group];
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::uint8_t* a_tile = a_native_tiles + (m_tile * k_tiles + k_tile) * 512;
        const std::uint8_t* b_tile = b_native_tiles + k_tile * 512;
        const std::uint32_t* a_regs = reinterpret_cast<const std::uint32_t*>(a_tile + threadIdx.x * 16);
        const std::uint32_t* b_regs = reinterpret_cast<const std::uint32_t*>(b_tile + threadIdx.x * 16);
        const std::uint32_t a0 = a_regs[0];
        const std::uint32_t a1 = a_regs[1];
        const std::uint32_t a2 = a_regs[2];
        const std::uint32_t a3 = a_regs[3];
        const std::uint32_t b0 = b_regs[0];
        const std::uint32_t b1 = b_regs[1];
        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[m_tile * k_tiles + k_tile]), "h"(bid), "h"(tid),
              "r"(sfb[k_tile]), "h"(bid), "h"(tid));
        d0 = nd0;
        d1 = nd1;
        d2 = nd2;
        d3 = nd3;
    }

    if ((threadIdx.x & 3) == 0) {
        const int row_base = threadIdx.x >> 2;
        const int out_base = static_cast<int>(m_tile) * 16;
        out[out_base + row_base] = d0;
        out[out_base + row_base + 8] = d2;
    }
}

extern "C" cudaError_t infer_sm12x_indexed_gemv_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_native_tiles_table,
    const std::uint32_t* const* a_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    float* const* d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || a_native_tiles_table == nullptr || a_scales_table == nullptr || b_native_tiles == nullptr || sfb == nullptr || d == nullptr || table_len == 0 || m_tiles == 0 || k_tiles == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(m_tiles, groups, 1);
    infer_sm12x_indexed_gemv_kernel<<<grid, 32, 0, stream>>>(indices, a_native_tiles_table, a_scales_table, table_len, b_native_tiles, sfb, d, m_tiles, k_tiles, groups);
    return cudaGetLastError();
}

__global__ void infer_sm12x_indexed_grouped_gemv_kernel(
    const std::uint32_t* __restrict__ indices,
    const std::uint8_t* const* __restrict__ a_native_tiles_table,
    const std::uint32_t* const* __restrict__ a_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* __restrict__ b_native_tiles,
    const std::uint32_t* __restrict__ sfb,
    float* const* __restrict__ d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups)
{
    const std::uint32_t m_tile = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (m_tile >= m_tiles || group >= groups) return;
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) return;

    const std::uint8_t* a_native_tiles = a_native_tiles_table[expert];
    const std::uint32_t* sfa = a_scales_table[expert];
    float* out = d[group];
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::uint8_t* a_tile = a_native_tiles + (m_tile * k_tiles + k_tile) * 512;
        const std::uint8_t* b_tile = b_native_tiles + (group * k_tiles + k_tile) * 512;
        const std::uint32_t* a_regs = reinterpret_cast<const std::uint32_t*>(a_tile + threadIdx.x * 16);
        const std::uint32_t* b_regs = reinterpret_cast<const std::uint32_t*>(b_tile + threadIdx.x * 16);
        const std::uint32_t a0 = a_regs[0];
        const std::uint32_t a1 = a_regs[1];
        const std::uint32_t a2 = a_regs[2];
        const std::uint32_t a3 = a_regs[3];
        const std::uint32_t b0 = b_regs[0];
        const std::uint32_t b1 = b_regs[1];
        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[m_tile * k_tiles + k_tile]), "h"(bid), "h"(tid),
              "r"(sfb[group * k_tiles + k_tile]), "h"(bid), "h"(tid));
        d0 = nd0;
        d1 = nd1;
        d2 = nd2;
        d3 = nd3;
    }

    if ((threadIdx.x & 3) == 0) {
        const int row_base = threadIdx.x >> 2;
        const int out_base = static_cast<int>(m_tile) * 16;
        out[out_base + row_base] = d0;
        out[out_base + row_base + 8] = d2;
    }
}

extern "C" cudaError_t infer_sm12x_indexed_grouped_gemv_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_native_tiles_table,
    const std::uint32_t* const* a_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    float* const* d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || a_native_tiles_table == nullptr || a_scales_table == nullptr || b_native_tiles == nullptr || sfb == nullptr || d == nullptr || table_len == 0 || m_tiles == 0 || k_tiles == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(m_tiles, groups, 1);
    infer_sm12x_indexed_grouped_gemv_kernel<<<grid, 32, 0, stream>>>(indices, a_native_tiles_table, a_scales_table, table_len, b_native_tiles, sfb, d, m_tiles, k_tiles, groups);
    return cudaGetLastError();
}

template <bool AddResidual>
__global__ void infer_sm12x_indexed_grouped_gemv_row_scales_kernel(
    const std::uint32_t* __restrict__ indices,
    const std::uint8_t* const* __restrict__ a_native_tiles_table,
    const std::uint32_t* const* __restrict__ a_row_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* __restrict__ b_native_tiles,
    const std::uint32_t* __restrict__ sfb,
    const std::uint8_t* __restrict__ residual_native_tiles,
    const std::uint32_t* __restrict__ residual_sfb,
    float* const* __restrict__ d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups)
{
    const std::uint32_t m_tile = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (m_tile >= m_tiles || group >= groups) return;
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) return;

    const std::uint8_t* a_native_tiles = a_native_tiles_table[expert];
    const std::uint32_t* row_scales = a_row_scales_table[expert];
    float* out = d[group];
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;
    const std::uint32_t scale_lane = threadIdx.x & 3;
    const std::uint32_t scale_row = (threadIdx.x >> 2) + (scale_lane == 1 ? 8 : 0);

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::uint8_t* a_tile = a_native_tiles + (m_tile * k_tiles + k_tile) * 512;
        const std::uint8_t* b_tile = b_native_tiles + (group * k_tiles + k_tile) * 512;
        const std::uint32_t* a_regs = reinterpret_cast<const std::uint32_t*>(a_tile + threadIdx.x * 16);
        const std::uint32_t* b_regs = reinterpret_cast<const std::uint32_t*>(b_tile + threadIdx.x * 16);
        const std::uint32_t a0 = a_regs[0];
        const std::uint32_t a1 = a_regs[1];
        const std::uint32_t a2 = a_regs[2];
        const std::uint32_t a3 = a_regs[3];
        const std::uint32_t b0 = b_regs[0];
        const std::uint32_t b1 = b_regs[1];
        const std::uint32_t sfa = scale_lane < 2
            ? row_scales[(m_tile * k_tiles + k_tile) * 16 + scale_row]
            : 0;
        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa), "h"(bid), "h"(tid),
              "r"(sfb[group * k_tiles + k_tile]), "h"(bid), "h"(tid));
        if constexpr (AddResidual) {
            const std::uint8_t* residual_tile =
                residual_native_tiles + (group * k_tiles + k_tile) * 512;
            const std::uint32_t* residual_regs =
                reinterpret_cast<const std::uint32_t*>(residual_tile + threadIdx.x * 16);
            const std::uint32_t residual_b0 = residual_regs[0];
            const std::uint32_t residual_b1 = residual_regs[1];
            float rd0 = 0.0f;
            float rd1 = 0.0f;
            float rd2 = 0.0f;
            float rd3 = 0.0f;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%10, %11, %12, %13},"
                "{%14},"
                "{%15, %16},"
                "{%17},"
                "{%18, %19};\n"
                : "=f"(rd0), "=f"(rd1), "=f"(rd2), "=f"(rd3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(residual_b0), "r"(residual_b1),
                  "f"(nd0), "f"(nd1), "f"(nd2), "f"(nd3),
                  "r"(sfa), "h"(bid), "h"(tid),
                  "r"(residual_sfb[group * k_tiles + k_tile]), "h"(bid), "h"(tid));
            d0 = rd0;
            d1 = rd1;
            d2 = rd2;
            d3 = rd3;
        } else {
            d0 = nd0;
            d1 = nd1;
            d2 = nd2;
            d3 = nd3;
        }
    }

    if ((threadIdx.x & 3) == 0) {
        const int row_base = threadIdx.x >> 2;
        const int out_base = static_cast<int>(m_tile) * 16;
        out[out_base + row_base] = d0;
        out[out_base + row_base + 8] = d2;
    }
}

extern "C" cudaError_t infer_sm12x_indexed_grouped_gemv_row_scales_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_native_tiles_table,
    const std::uint32_t* const* a_row_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    float* const* d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || a_native_tiles_table == nullptr ||
        a_row_scales_table == nullptr || b_native_tiles == nullptr ||
        sfb == nullptr || d == nullptr || table_len == 0 || m_tiles == 0 ||
        k_tiles == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(m_tiles, groups, 1);
    infer_sm12x_indexed_grouped_gemv_row_scales_kernel<false><<<grid, 32, 0, stream>>>(
        indices, a_native_tiles_table, a_row_scales_table, table_len,
        b_native_tiles, sfb, nullptr, nullptr, d, m_tiles, k_tiles, groups);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_indexed_grouped_gemv_row_scales_residual_on_stream(
    const std::uint32_t* indices,
    const std::uint8_t* const* a_native_tiles_table,
    const std::uint32_t* const* a_row_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    const std::uint8_t* residual_native_tiles,
    const std::uint32_t* residual_sfb,
    float* const* d,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t groups,
    cudaStream_t stream)
{
    if (indices == nullptr || a_native_tiles_table == nullptr ||
        a_row_scales_table == nullptr || b_native_tiles == nullptr ||
        sfb == nullptr || residual_native_tiles == nullptr ||
        residual_sfb == nullptr || d == nullptr || table_len == 0 ||
        m_tiles == 0 || k_tiles == 0 || groups == 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(m_tiles, groups, 1);
    infer_sm12x_indexed_grouped_gemv_row_scales_kernel<true><<<grid, 32, 0, stream>>>(
        indices, a_native_tiles_table, a_row_scales_table, table_len,
        b_native_tiles, sfb, residual_native_tiles, residual_sfb, d,
        m_tiles, k_tiles, groups);
    return cudaGetLastError();
}

template <int Terms>
__global__ void infer_sm12x_gemv_row_scales_kernel(
    const std::uint8_t* __restrict__ a_native_tiles,
    const std::uint32_t* __restrict__ a_row_scales,
    const std::uint8_t* __restrict__ b_native_tiles,
    const std::uint32_t* __restrict__ sfb,
    const std::uint8_t* __restrict__ residual_native_tiles,
    const std::uint32_t* __restrict__ residual_sfb,
    const std::uint8_t* __restrict__ residual2_native_tiles,
    const std::uint32_t* __restrict__ residual2_sfb,
    float* __restrict__ output,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t k_splits,
    float alpha)
{
    const std::uint32_t m_tile = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t split = blockIdx.z;
    if (m_tile >= m_tiles) return;
    b_native_tiles += row * k_tiles * 512;
    sfb += row * k_tiles;
    if constexpr (Terms >= 2) {
        residual_native_tiles += row * k_tiles * 512;
        residual_sfb += row * k_tiles;
    }
    if constexpr (Terms >= 3) {
        residual2_native_tiles += row * k_tiles * 512;
        residual2_sfb += row * k_tiles;
    }
    output += (row * k_splits + split) * m_tiles * 16;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t bid = 0;
    const std::uint16_t tid = 0;
    const std::uint32_t scale_lane = threadIdx.x & 3;
    const std::uint32_t scale_row = (threadIdx.x >> 2) + (scale_lane == 1 ? 8 : 0);

    const std::uint32_t k_begin = k_tiles * split / k_splits;
    const std::uint32_t k_end = k_tiles * (split + 1) / k_splits;
    for (std::uint32_t k_tile = k_begin; k_tile < k_end; ++k_tile) {
        const std::uint8_t* a_tile =
            a_native_tiles + (m_tile * k_tiles + k_tile) * 512;
        const std::uint8_t* b_tile = b_native_tiles + k_tile * 512;
        const std::uint32_t* a_regs =
            reinterpret_cast<const std::uint32_t*>(a_tile + threadIdx.x * 16);
        const std::uint32_t* b_regs =
            reinterpret_cast<const std::uint32_t*>(b_tile + threadIdx.x * 16);
        const std::uint32_t sfa = scale_lane < 2
            ? a_row_scales[(m_tile * k_tiles + k_tile) * 16 + scale_row]
            : 0;
        float nd0 = 0.0f;
        float nd1 = 0.0f;
        float nd2 = 0.0f;
        float nd3 = 0.0f;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(nd0), "=f"(nd1), "=f"(nd2), "=f"(nd3)
            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
              "r"(b_regs[0]), "r"(b_regs[1]),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa), "h"(bid), "h"(tid),
              "r"(sfb[k_tile]), "h"(bid), "h"(tid));
        if constexpr (Terms >= 2) {
            const std::uint8_t* residual_tile = residual_native_tiles + k_tile * 512;
            const std::uint32_t* residual_regs = reinterpret_cast<const std::uint32_t*>(
                residual_tile + threadIdx.x * 16);
            float rd0 = 0.0f;
            float rd1 = 0.0f;
            float rd2 = 0.0f;
            float rd3 = 0.0f;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%10, %11, %12, %13},"
                "{%14},"
                "{%15, %16},"
                "{%17},"
                "{%18, %19};\n"
                : "=f"(rd0), "=f"(rd1), "=f"(rd2), "=f"(rd3)
                : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                  "r"(residual_regs[0]), "r"(residual_regs[1]),
                  "f"(nd0), "f"(nd1), "f"(nd2), "f"(nd3),
                  "r"(sfa), "h"(bid), "h"(tid),
                  "r"(residual_sfb[k_tile]), "h"(bid), "h"(tid));
            if constexpr (Terms >= 3) {
                const std::uint8_t* residual2_tile = residual2_native_tiles + k_tile * 512;
                const std::uint32_t* residual2_regs = reinterpret_cast<const std::uint32_t*>(
                    residual2_tile + threadIdx.x * 16);
                float td0 = 0.0f;
                float td1 = 0.0f;
                float td2 = 0.0f;
                float td3 = 0.0f;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3},"
                    "{%4, %5, %6, %7},"
                    "{%8, %9},"
                    "{%10, %11, %12, %13},"
                    "{%14},"
                    "{%15, %16},"
                    "{%17},"
                    "{%18, %19};\n"
                    : "=f"(td0), "=f"(td1), "=f"(td2), "=f"(td3)
                    : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                      "r"(residual2_regs[0]), "r"(residual2_regs[1]),
                      "f"(rd0), "f"(rd1), "f"(rd2), "f"(rd3),
                      "r"(sfa), "h"(bid), "h"(tid),
                      "r"(residual2_sfb[k_tile]), "h"(bid), "h"(tid));
                d0 = td0;
                d1 = td1;
                d2 = td2;
                d3 = td3;
            } else {
                d0 = rd0;
                d1 = rd1;
                d2 = rd2;
                d3 = rd3;
            }
        } else {
            d0 = nd0;
            d1 = nd1;
            d2 = nd2;
            d3 = nd3;
        }
    }

    if ((threadIdx.x & 3) == 0) {
        const int row_base = threadIdx.x >> 2;
        const int out_base = static_cast<int>(m_tile) * 16;
        output[out_base + row_base] = d0 * alpha;
        output[out_base + row_base + 8] = d2 * alpha;
    }
}

extern "C" cudaError_t infer_sm12x_gemv_row_scales_residual_batch_on_stream(
    const std::uint8_t* a_native_tiles,
    const std::uint32_t* a_row_scales,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    const std::uint8_t* residual_native_tiles,
    const std::uint32_t* residual_sfb,
    float* output,
    std::uint32_t rows,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    float alpha,
    cudaStream_t stream)
{
    if (a_native_tiles == nullptr || a_row_scales == nullptr ||
        b_native_tiles == nullptr || sfb == nullptr ||
        residual_native_tiles == nullptr || residual_sfb == nullptr ||
        output == nullptr || rows == 0 || m_tiles == 0 || k_tiles == 0 ||
        !isfinite(alpha)) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_gemv_row_scales_kernel<2><<<dim3(m_tiles, rows), 32, 0, stream>>>(
        a_native_tiles, a_row_scales, b_native_tiles, sfb,
        residual_native_tiles, residual_sfb, nullptr, nullptr,
        output, m_tiles, k_tiles, 1, alpha);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sm12x_gemv_row_scales_residual2_batch_on_stream(
    const std::uint8_t* a_native_tiles,
    const std::uint32_t* a_row_scales,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    const std::uint8_t* residual_native_tiles,
    const std::uint32_t* residual_sfb,
    const std::uint8_t* residual2_native_tiles,
    const std::uint32_t* residual2_sfb,
    float* output,
    std::uint32_t rows,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    float alpha,
    cudaStream_t stream)
{
    if (a_native_tiles == nullptr || a_row_scales == nullptr ||
        b_native_tiles == nullptr || sfb == nullptr ||
        residual_native_tiles == nullptr || residual_sfb == nullptr ||
        residual2_native_tiles == nullptr || residual2_sfb == nullptr ||
        output == nullptr || rows == 0 || m_tiles == 0 || k_tiles == 0 ||
        !isfinite(alpha)) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_gemv_row_scales_kernel<3><<<dim3(m_tiles, rows), 32, 0, stream>>>(
        a_native_tiles, a_row_scales, b_native_tiles, sfb,
        residual_native_tiles, residual_sfb, residual2_native_tiles,
        residual2_sfb, output, m_tiles, k_tiles, 1, alpha);
    return cudaGetLastError();
}

__global__ void infer_sm12x_reduce_gemv_partials_kernel(
    const float* __restrict__ partials,
    float* __restrict__ output,
    std::uint32_t values,
    std::uint32_t splits)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = blockIdx.y;
    if (index >= values) return;
    float sum = 0.0f;
    for (std::uint32_t split = 0; split < splits; ++split) {
        sum += partials[(row * splits + split) * values + index];
    }
    output[row * values + index] = sum;
}

extern "C" cudaError_t infer_sm12x_gemv_row_scales_residual2_splitk_batch_on_stream(
    const std::uint8_t* a_native_tiles,
    const std::uint32_t* a_row_scales,
    const std::uint8_t* b_native_tiles,
    const std::uint32_t* sfb,
    const std::uint8_t* residual_native_tiles,
    const std::uint32_t* residual_sfb,
    const std::uint8_t* residual2_native_tiles,
    const std::uint32_t* residual2_sfb,
    float* partials,
    float* output,
    std::uint32_t rows,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    std::uint32_t k_splits,
    float alpha,
    cudaStream_t stream)
{
    if (a_native_tiles == nullptr || a_row_scales == nullptr ||
        b_native_tiles == nullptr || sfb == nullptr ||
        residual_native_tiles == nullptr || residual_sfb == nullptr ||
        residual2_native_tiles == nullptr || residual2_sfb == nullptr ||
        partials == nullptr || output == nullptr || rows == 0 ||
        m_tiles == 0 || k_tiles == 0 || k_splits < 2 ||
        k_splits > k_tiles || !isfinite(alpha)) {
        return cudaErrorInvalidValue;
    }
    infer_sm12x_gemv_row_scales_kernel<3><<<
        dim3(m_tiles, rows, k_splits), 32, 0, stream>>>(
        a_native_tiles, a_row_scales, b_native_tiles, sfb,
        residual_native_tiles, residual_sfb, residual2_native_tiles,
        residual2_sfb, partials, m_tiles, k_tiles, k_splits, alpha);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const std::uint32_t values = m_tiles * 16;
    infer_sm12x_reduce_gemv_partials_kernel<<<
        dim3((values + 255) / 256, rows), 256, 0, stream>>>(
        partials, output, values, k_splits);
    return cudaGetLastError();
}
