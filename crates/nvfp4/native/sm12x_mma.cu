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
    if (kt >= k_tiles) return;
    std::uint8_t* tile = b_native_tiles + kt * 512;
    for (int index = threadIdx.x; index < 512; index += blockDim.x) tile[index] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    std::uint8_t codes[64];
    std::uint8_t scale_codes[4];
    for (int block = 0; block < 4; ++block) {
        float max_abs = 0.0f;
        for (int offset = 0; offset < 16; ++offset) {
            max_abs = fmaxf(max_abs, fabsf(input[kt * 64 + block * 16 + offset]));
        }
        scale_codes[block] = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        const float scale = infer_e4m3_value(scale_codes[block]);
        for (int offset = 0; offset < 16; ++offset) {
            const float value = input[kt * 64 + block * 16 + offset];
            codes[block * 16 + offset] = infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale);
        }
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
    sfb[kt] = static_cast<std::uint32_t>(scale_codes[0])
        | (static_cast<std::uint32_t>(scale_codes[1]) << 8)
        | (static_cast<std::uint32_t>(scale_codes[2]) << 16)
        | (static_cast<std::uint32_t>(scale_codes[3]) << 24);
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

__global__ void infer_sm12x_kv_copy_tail_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ key_tail,
    float* __restrict__ value_tail,
    std::uint32_t position,
    std::uint32_t width)
{
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    const std::uint32_t destination = (position & 15u) * width + column;
    key_tail[destination] = key[column];
    value_tail[destination] = value[column];
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

__global__ void infer_sm12x_kv_quantize_query_kernel(
    const float* __restrict__ query,
    std::uint8_t* __restrict__ query_tiles,
    std::uint32_t* __restrict__ query_scales,
    std::uint32_t head_dim)
{
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t k_tile = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* tile = query_tiles + (group * (head_dim / 64) + k_tile) * 512;
    std::uint8_t scale_codes[4];
    float scales[4];
    for (int kb = 0; kb < 4; ++kb) {
        float max_abs = 0.0f;
        for (int row = 0; row < 8; ++row) {
            for (int offset = 0; offset < 16; ++offset) {
                const float value = query[(group * 8 + row) * head_dim + k_tile * 64 + kb * 16 + offset];
                if (isfinite(value)) max_abs = fmaxf(max_abs, fabsf(value));
            }
        }
        scale_codes[kb] = max_abs == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_abs / 6.0f, __NV_SATFINITE, __NV_E4M3));
        scales[kb] = infer_e4m3_value(scale_codes[kb]);
    }
    for (int lane = 0; lane < 32; ++lane) {
        const int t0 = lane & 3;
        const int t1 = lane >> 2;
        for (int v = 0; v < 32; ++v) {
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int v2 = (v >> 4) & 1;
            const int row = t1 + 8 * v1;
            const int col = t0 * 8 + v0 + 32 * v2;
            const int kb = col / 16;
            float value = 0.0f;
            if (row < 8) {
                value = query[(group * 8 + row) * head_dim + k_tile * 64 + col];
            }
            infer_set_packed_nibble(
                tile, lane * 32 + v,
                infer_e2m1_code(scales[kb] == 0.0f ? 0.0f : value / scales[kb]));
        }
    }
    query_scales[group * (head_dim / 64) + k_tile] =
        static_cast<std::uint32_t>(scale_codes[0])
        | (static_cast<std::uint32_t>(scale_codes[1]) << 8)
        | (static_cast<std::uint32_t>(scale_codes[2]) << 16)
        | (static_cast<std::uint32_t>(scale_codes[3]) << 24);
}

__global__ void infer_sm12x_kv_qk_kernel(
    const std::uint8_t* __restrict__ query_tiles,
    const std::uint32_t* __restrict__ query_scales,
    const std::uint8_t* __restrict__ key_values,
    const std::uint8_t* __restrict__ key_scales,
    const float* __restrict__ key_tail,
    float* __restrict__ scores,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    __shared__ __align__(16) std::uint8_t b_smem[512];
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token_tile = blockIdx.y;
    const std::uint32_t complete_tiles = cache_len / 8;
    const bool compact = token_tile < complete_tiles;
    const std::uint32_t tail_len = cache_len & 7u;
    const std::uint32_t head_k_tiles = head_dim / 64;
    const std::uint32_t max_token_tiles = (max_tokens + 7) / 8;
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
            tile = (group * max_token_tiles + token_tile) * head_k_tiles + kt;
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
                    const float value = key_tail[(tail_start + row) * width + group * head_dim + kt * 64 + kb * 16 + offset];
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
                const float value = key_tail[(tail_start + row) * width + group * head_dim + kt * 64 + col];
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
        const std::uint32_t b_scale_word = compact
            ? infer_scale_word(key_scales + (tile * 8 + row) * 4)
            : infer_scale_word(tail_scale_codes);
        infer_mma_m16n8k64(
            a0, a1, a2, a3, b0, b1,
            query_scales[group * head_k_tiles + kt], b_scale_word,
            d0, d1, d2, d3);
        __syncthreads();
    }

    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 2;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const std::uint32_t q_head = group * 8 + row;
    const std::uint32_t token0 = token_tile * 8 + col;
    if (token0 < cache_len) scores[q_head * max_tokens + token0] = d0 * scale;
    if (token0 + 1 < cache_len) scores[q_head * max_tokens + token0 + 1] = d1 * scale;
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
    float* scores, std::uint32_t cache_len, std::uint32_t max_tokens) {
    __shared__ float maxima[256];
    __shared__ float sums[256];
    InferOnlineSoftmaxState state = {-INFINITY, 0.0f};
    float* row = scores + blockIdx.x * max_tokens;
    for (std::uint32_t token = threadIdx.x; token < cache_len; token += blockDim.x) {
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
    for (std::uint32_t token = threadIdx.x; token < cache_len; token += blockDim.x) {
        row[token] = expf(row[token] - maximum) * inverse_sum;
    }
}

__global__ void infer_sm12x_kv_quantize_probability_kernel(
    const float* __restrict__ scores,
    std::uint8_t* __restrict__ probability_tiles,
    std::uint32_t* __restrict__ probability_scales,
    std::uint32_t cache_len,
    std::uint32_t max_tokens)
{
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t k_tile = blockIdx.y;
    if (threadIdx.x != 0) return;
    const std::uint32_t context_tiles = (max_tokens + 63) / 64;
    std::uint8_t* tile = probability_tiles + (group * context_tiles + k_tile) * 512;
    std::uint8_t scale_codes[4];
    float scales[4];
    for (int kb = 0; kb < 4; ++kb) {
        float max_value = 0.0f;
        for (int row = 0; row < 8; ++row) {
            for (int offset = 0; offset < 16; ++offset) {
                const std::uint32_t token = k_tile * 64 + kb * 16 + offset;
                if (token < cache_len) {
                    max_value = fmaxf(max_value, scores[(group * 8 + row) * max_tokens + token]);
                }
            }
        }
        scale_codes[kb] = max_value == 0.0f ? 0 : static_cast<std::uint8_t>(
            __nv_cvt_float_to_fp8(max_value / 6.0f, __NV_SATFINITE, __NV_E4M3));
        scales[kb] = infer_e4m3_value(scale_codes[kb]);
    }
    for (int lane = 0; lane < 32; ++lane) {
        const int t0 = lane & 3;
        const int t1 = lane >> 2;
        for (int v = 0; v < 32; ++v) {
            const int v0 = v & 7;
            const int v1 = (v >> 3) & 1;
            const int v2 = (v >> 4) & 1;
            const int row = t1 + 8 * v1;
            const int col = t0 * 8 + v0 + 32 * v2;
            const std::uint32_t token = k_tile * 64 + col;
            float value = 0.0f;
            if (row < 8 && token < cache_len) {
                value = scores[(group * 8 + row) * max_tokens + token];
            }
            const float scale = scales[col / 16];
            infer_set_packed_nibble(
                tile, lane * 32 + v,
                infer_e2m1_code(scale == 0.0f ? 0.0f : value / scale));
        }
    }
    probability_scales[group * context_tiles + k_tile] =
        static_cast<std::uint32_t>(scale_codes[0])
        | (static_cast<std::uint32_t>(scale_codes[1]) << 8)
        | (static_cast<std::uint32_t>(scale_codes[2]) << 16)
        | (static_cast<std::uint32_t>(scale_codes[3]) << 24);
}

__global__ void infer_sm12x_kv_pv_kernel(
    const std::uint8_t* __restrict__ probability_tiles,
    const std::uint32_t* __restrict__ probability_scales,
    const std::uint8_t* __restrict__ value_values,
    const std::uint8_t* __restrict__ value_scales,
    const float* __restrict__ value_tail,
    float* __restrict__ output,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim)
{
    __shared__ __align__(16) std::uint8_t b_smem[512];
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t dim_tile = blockIdx.y;
    const std::uint32_t context_tiles = (cache_len + 63) / 64;
    const std::uint32_t max_context_tiles = (max_tokens + 63) / 64;
    const std::uint32_t full_tokens = cache_len / 16 * 16;
    const std::uint32_t tail_len = cache_len & 15u;
    const std::uint32_t width = kv_heads * head_dim;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (std::uint32_t kt = 0; kt < context_tiles; ++kt) {
        const std::uint8_t* a_tile = probability_tiles + (group * max_context_tiles + kt) * 512;
        const std::uint32_t value_tile_index =
            (group * (head_dim / 8) + dim_tile) * max_context_tiles + kt;
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
                    const float value = value_tail[token * width + group * head_dim + dim_tile * 8 + dim];
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
                const float value = value_tail[(token - full_tokens) * width + group * head_dim + dim_tile * 8 + dim];
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
        infer_mma_m16n8k64(
            a0, a1, a2, a3, b0, b1,
            probability_scales[group * max_context_tiles + kt],
            infer_scale_word(b_scale_codes), d0, d1, d2, d3);
        __syncthreads();
    }

    const int row = threadIdx.x >> 2;
    const int col = (threadIdx.x & 3) * 2;
    const std::uint32_t q_head = group * 8 + row;
    output[q_head * head_dim + dim_tile * 8 + col] = d0;
    output[q_head * head_dim + dim_tile * 8 + col + 1] = d1;
}

extern "C" cudaError_t infer_sm12x_kv_attention_on_stream(
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
    float* output,
    std::uint32_t cache_len,
    std::uint32_t max_tokens,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    cudaStream_t stream)
{
    if (query == nullptr || key_values == nullptr || key_scales == nullptr || key_tail == nullptr ||
        value_values == nullptr || value_scales == nullptr || value_tail == nullptr ||
        query_tiles == nullptr || query_scales == nullptr || scores == nullptr ||
        probability_tiles == nullptr || probability_scales == nullptr || output == nullptr ||
        cache_len == 0 || cache_len > max_tokens || kv_heads == 0 || head_dim == 0 ||
        (head_dim % 64) != 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t head_k_tiles = head_dim / 64;
    const std::uint32_t token_tiles = (cache_len + 7) / 8;
    const std::uint32_t context_tiles = (cache_len + 63) / 64;
    infer_sm12x_kv_quantize_query_kernel<<<dim3(kv_heads, head_k_tiles, 1), 1, 0, stream>>>(
        query, query_tiles, query_scales, head_dim);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_qk_kernel<<<dim3(kv_heads, token_tiles, 1), 32, 0, stream>>>(
        query_tiles, query_scales, key_values, key_scales, key_tail, scores,
        cache_len, max_tokens, kv_heads, head_dim);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_softmax_kernel<<<kv_heads * 8, 256, 0, stream>>>(
        scores, cache_len, max_tokens);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_quantize_probability_kernel<<<dim3(kv_heads, context_tiles, 1), 1, 0, stream>>>(
        scores, probability_tiles, probability_scales, cache_len, max_tokens);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    infer_sm12x_kv_pv_kernel<<<dim3(kv_heads, head_dim / 8, 1), 32, 0, stream>>>(
        probability_tiles, probability_scales, value_values, value_scales, value_tail,
        output, cache_len, max_tokens, kv_heads, head_dim);
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

    const std::uint32_t expert = indices[slot];
    const float input_scale = input_scale_table[expert];
    if (input_scale <= 0.0f || !isfinite(input_scale)) return;
    const float gate_up_alpha = gate_up_alpha_table[expert];
    const float* gate_up = Bf16Input ? nullptr : gate_up_table[slot];
    const int scale_group = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    float value = 0.0f;
    if (lane < 16) {
        const std::uint32_t row = kt * 64 + scale_group * 16 + lane;
        if (row < rows) {
            const std::uint32_t base = slot * rows * 2;
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
    infer_sm12x_moe_silu_quantize_slots_kernel<false><<<dim3(groups, k_tiles, 1), 128, 0, stream>>>(indices, gate_up_table, nullptr, b_native_tiles, sfb, input_scale_table, gate_up_alpha_table, rows, k_tiles, groups);
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
    infer_sm12x_moe_silu_quantize_slots_kernel<true><<<dim3(groups, k_tiles, 1), 128, 0, stream>>>(indices, nullptr, gate_up_bf16, b_native_tiles, sfb, input_scale_table, gate_up_alpha_table, rows, k_tiles, groups);
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
