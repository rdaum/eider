#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

// Six scaled E4M3 residual components preserve low input bits while sharing
// one FP8 weight read across the component columns of a small batch.
__global__ void split_f32_fp8_parts(const float* input, uint8_t* parts, float* powers,
                                   uint32_t rows, uint32_t width) {
    __shared__ float maxima[256];
    const uint32_t row = blockIdx.x, lane = threadIdx.x;
    float maximum = 0.f;
    for (uint32_t k=lane; k<width; k+=256)
        maximum = fmaxf(maximum, fabsf(input[size_t(row)*width+k]));
    maxima[lane] = maximum;
    __syncthreads();
    for (uint32_t step=128; step; step>>=1) {
        if (lane<step) maxima[lane] = fmaxf(maxima[lane], maxima[lane+step]);
        __syncthreads();
    }
    int exponent = 0;
    frexpf(maxima[0], &exponent);
    const float power = ldexpf(1.f, max(-106, min(120, exponent-8)));
    if (!lane) powers[row] = power;
    for (uint32_t k=lane; k<width; k+=256) {
        float residual = input[size_t(row)*width+k], scale = power;
        #pragma unroll
        for (int part=0; part<6; ++part) {
            const __nv_fp8_e4m3 value(residual / scale);
            parts[(size_t(part)*rows+row)*width+k] = value.__x;
            residual = fmaf(-float(value), scale, residual);
            scale *= 0.0625f;
        }
    }
}

__global__ void sum_f32_fp8_parts(const float* parts, const float* powers, float* output,
                                 uint32_t rows, uint32_t width, float weight_scale) {
    const uint32_t i = blockIdx.x*blockDim.x+threadIdx.x, count=rows*width;
    if (i>=count) return;
    float sum=parts[5*size_t(count)+i];
    #pragma unroll
    for (int part=4; part>=0; --part) sum=fmaf(sum,0.0625f,parts[part*size_t(count)+i]);
    output[i]=sum*(powers[i/width]*weight_scale);
}

extern "C" cudaError_t infer_split_f32_fp8_parts_on_stream(
    const float* input, uint8_t* parts, float* powers, uint32_t rows, uint32_t width,
    cudaStream_t stream) {
    split_f32_fp8_parts<<<rows,256,0,stream>>>(input,parts,powers,rows,width);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_sum_f32_fp8_parts_on_stream(
    const float* parts, const float* powers, float* output, uint32_t rows,
    uint32_t width, float weight_scale, cudaStream_t stream) {
    sum_f32_fp8_parts<<<(rows*width+255)/256,256,0,stream>>>(
        parts,powers,output,rows,width,weight_scale);
    return cudaGetLastError();
}

template<unsigned Parts>
__global__ void lfm2_split_bf16(const float* input, __nv_bfloat16* parts,
                               uint32_t count, uint32_t stride) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= stride) return;
    float x = i < count ? input[i] : 0;
    __nv_bfloat16 high = __float2bfloat16_rn(x);
    parts[i] = high;
    if constexpr (Parts == 1) return;
    float remainder = x - __bfloat162float(high);
    __nv_bfloat16 middle = __float2bfloat16_rn(remainder);
    parts[i] = high;
    parts[stride + i] = middle;
    parts[2 * stride + i] = __float2bfloat16_rn(remainder - __bfloat162float(middle));
}

template<unsigned Parts>
__global__ void lfm2_sum_parts(const float* parts, float* output, uint32_t count, uint32_t stride,
    const float* channels, float scale, uint32_t width) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = parts[i];
    if constexpr (Parts == 3) value += parts[stride + i] + parts[2 * stride + i];
    output[i] = value * (channels ? channels[i % width] : scale);
}

extern "C" int infer_lfm2_split_bf16_on_stream(
    const float* input, __nv_bfloat16* parts, uint32_t count, uint32_t stride, uint32_t parts_count, cudaStream_t stream) {
    if (parts_count == 1) lfm2_split_bf16<1><<<(stride + 255) / 256, 256, 0, stream>>>(input, parts, count, stride);
    else lfm2_split_bf16<3><<<(stride + 255) / 256, 256, 0, stream>>>(input, parts, count, stride);
    return cudaGetLastError();
}

extern "C" int infer_lfm2_sum_parts_on_stream(
    const float* parts, float* output, uint32_t count, uint32_t stride,
    const float* channels, float scale, uint32_t width, uint32_t parts_count, cudaStream_t stream) {
    if (parts_count == 1) lfm2_sum_parts<1><<<(count + 255) / 256, 256, 0, stream>>>(parts, output, count, stride, channels, scale, width);
    else lfm2_sum_parts<3><<<(count + 255) / 256, 256, 0, stream>>>(parts, output, count, stride, channels, scale, width);
    return cudaGetLastError();
}

// State contains the previous L-1 B*x values, oldest first, per channel.
// LFM gates are linear: neither B nor C has a sigmoid/SiLU activation.
__global__ void lfm2_short_conv(const float* bcx, const float* weight,
                               float* state, float* output,
                               uint32_t hidden, uint32_t width) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    float bx = bcx[c] * bcx[2 * hidden + c];
    float sum = bx * weight[c * width + width - 1];
    for (uint32_t j = 0; j + 1 < width; ++j)
        sum = fmaf(state[c * (width - 1) + j], weight[c * width + j], sum);
    for (uint32_t j = 0; j + 2 < width; ++j)
        state[c * (width - 1) + j] = state[c * (width - 1) + j + 1];
    if (width > 1) state[c * (width - 1) + width - 2] = bx;
    output[c] = bcx[hidden + c] * sum;
}

// Only selection uses the correction bias. Normalization includes the LFM
// reference's 1e-6 denominator epsilon and precedes routed scaling.
__global__ void lfm2_router(const float* logits, const float* bias,
                           uint32_t* indices, float* weights,
                           uint32_t experts, uint32_t top_k, bool normalize,
                           float scale) {
    logits += size_t(blockIdx.x) * experts;
    indices += size_t(blockIdx.x) * top_k;
    weights += size_t(blockIdx.x) * top_k;
    __shared__ float probability[512];
    __shared__ float score[512];
    uint32_t e = threadIdx.x;
    if (e < experts) {
        probability[e] = 1.0f / (1.0f + expf(-logits[e]));
        score[e] = probability[e] + bias[e];
    }
    __syncthreads();
    if (e != 0) return;
    float sum = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < experts; ++i)
            if (score[i] > score[best]) best = i;
        indices[k] = best;
        weights[k] = probability[best];
        sum += weights[k];
        score[best] = -INFINITY;
    }
    for (uint32_t k = 0; k < top_k; ++k)
        weights[k] *= normalize ? scale / (sum + 1e-6f) : scale;
}

extern "C" int infer_lfm2_short_conv_on_stream(
    const float* bcx, const float* weight, float* state, float* output,
    uint32_t hidden, uint32_t width, cudaStream_t stream) {
    lfm2_short_conv<<<(hidden + 255) / 256, 256, 0, stream>>>(
        bcx, weight, state, output, hidden, width);
    return cudaGetLastError();
}

extern "C" int infer_lfm2_router_on_stream(
    const float* logits, const float* bias, uint32_t* indices, float* weights,
    uint32_t experts, uint32_t top_k, int normalize, float scale, cudaStream_t stream) {
    lfm2_router<<<1, 512, 0, stream>>>(logits, bias, indices, weights, experts,
                                   top_k, normalize != 0, scale);
    return cudaGetLastError();
}

// Each output reads only the previous chunk's history. Commit the new history
// in a separate launch so no output races with a history update.
__global__ void lfm2_short_conv_batch(const float* bcx, const float* weight,
                                     const float* state, float* output,
                                     uint32_t hidden, uint32_t width, uint32_t rows) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (c >= hidden || row >= rows) return;
    const float* current = bcx + size_t(row) * 3 * hidden;
    float sum = current[c] * current[2 * hidden + c] * weight[c * width + width - 1];
    for (uint32_t j = 0; j + 1 < width; ++j) {
        int64_t source = int64_t(row) + j - (width - 1);
        float bx;
        if (source < 0) bx = state[c * (width - 1) + row + j];
        else {
            const float* previous = bcx + size_t(source) * 3 * hidden;
            bx = previous[c] * previous[2 * hidden + c];
        }
        sum = fmaf(bx, weight[c * width + j], sum);
    }
    output[size_t(row) * hidden + c] = current[hidden + c] * sum;
}

__global__ void lfm2_conv_commit(const float* bcx, float* state,
                                uint32_t hidden, uint32_t width, uint32_t rows) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    for (uint32_t j = 0; j + 1 < width; ++j) {
        int64_t source = int64_t(rows) + j - (width - 1);
        if (source < 0) state[c * (width - 1) + j] = state[c * (width - 1) + rows + j];
        else {
            const float* previous = bcx + size_t(source) * 3 * hidden;
            state[c * (width - 1) + j] = previous[c] * previous[2 * hidden + c];
        }
    }
}

extern "C" int infer_lfm2_short_conv_batch_on_stream(
    const float* bcx, const float* weight, float* state, float* output,
    uint32_t hidden, uint32_t width, uint32_t rows, cudaStream_t stream) {
    lfm2_short_conv_batch<<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(
        bcx, weight, state, output, hidden, width, rows);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    lfm2_conv_commit<<<(hidden + 255) / 256, 256, 0, stream>>>(bcx, state, hidden, width, rows);
    return cudaGetLastError();
}

extern "C" int infer_lfm2_router_batch_on_stream(
    const float* logits, const float* bias, uint32_t* indices, float* weights,
    uint32_t experts, uint32_t top_k, uint32_t rows, int normalize, float scale,
    cudaStream_t stream) {
    lfm2_router<<<rows, 512, 0, stream>>>(logits, bias, indices, weights, experts,
                                       top_k, normalize != 0, scale);
    return cudaGetLastError();
}

__global__ void lfm2_decode_inputs(uint32_t token, uint32_t position,
                                 uint32_t* token_id, uint32_t* start, uint32_t* length) {
    *token_id = token;
    *start = position;
    *length = position + 1;
}

extern "C" int infer_lfm2_decode_inputs_on_stream(
    uint32_t token, uint32_t position, uint32_t* token_id, uint32_t* start,
    uint32_t* length, cudaStream_t stream) {
    lfm2_decode_inputs<<<1, 1, 0, stream>>>(token, position, token_id, start, length);
    return cudaGetLastError();
}

// Grouped expert GEMM keeps the checkpoint in row-major NVFP4. Its E2M1 times
// UE4M3 products are exactly representable in BF16. Three BF16 components carry
// each F32 activation through tensor cores without reducing its mantissa to BF16.
namespace {
__device__ __forceinline__ uint32_t lfm2_pair(float x, float y) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(x))
        | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(y))) << 16);
}
__device__ __forceinline__ uint16_t lfm2_scale_bf16(uint8_t code) {
    return code < 8 ? __bfloat16_as_ushort(__float2bfloat16_rn(code * 0x1p-9f))
        : uint16_t((unsigned(code) << 4) + 0x3c00u);
}
__device__ __forceinline__ uint32_t lfm2_weight_pair(uint8_t packed, uint16_t sf) {
    const unsigned selector = (packed & 7) | ((packed >> 4 & 7) << 8);
    const unsigned lo = __byte_perm(0xc0800000u, 0xc0804000u, selector) & 0x00ff00ffu;
    const unsigned hi = (__byte_perm(0x3f3f3f00u, 0x40404040u, selector) & 0x00ff00ffu) << 8;
    const unsigned sign = (unsigned(packed & 8) << 12) | (unsigned(packed & 128) << 24);
    const unsigned scales2 = unsigned(sf) | (unsigned(sf) << 16);
    unsigned result;
    asm("mul.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(lo | hi | sign), "r"(scales2));
    return result;
}
__device__ __forceinline__ void lfm2_weight_group(
    const uint8_t* weight, const uint8_t* scales, unsigned row, unsigned k,
    unsigned cols, unsigned lane, uint32_t& low, uint32_t& high) {
    const size_t offset = size_t(row) * (cols / 2) + k / 2;
    const uint2 bytes = *reinterpret_cast<const uint2*>(weight + offset);
    const uint8_t code = scales[size_t(row) * (cols / 16) + k / 16];
    const uint16_t sf = lfm2_scale_bf16(code);
    const unsigned shift = (lane & 3) * 8;
    low = lfm2_weight_pair(uint8_t(bytes.x >> shift), sf);
    high = lfm2_weight_pair(uint8_t(bytes.y >> shift), sf);
}
__device__ __forceinline__ void lfm2_mma(float* d, uint32_t a0, uint32_t a1,
                                       uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool Routed, unsigned Columns = 1, unsigned DenseFormat = 0, bool Bf16 = false>
__global__ void lfm2_gemm(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned experts, unsigned out_features, unsigned in_features,
    const float* dense_input, const uint16_t* dense_weight, float* dense_output, unsigned rows,
    const float* dense_channels, float dense_scale, const uint8_t* dense_block_scales) {
    __shared__ unsigned expert, begin, end;
    if (threadIdx.x == 0) {
      if constexpr (!Routed) {
        expert = 0; begin = blockIdx.y * 8; end = min(begin + 8, rows);
      } else {
        unsigned tile = blockIdx.y;
        expert = experts;
        for (unsigned e = 0; e < experts; ++e) {
            const unsigned first = offsets[e], last = offsets[e + 1];
            const unsigned tiles = (last - first + Columns * 8 - 1) / (Columns * 8);
            if (tile < tiles) {
                expert = e;
                begin = first + tile * Columns * 8;
                end = min(begin + Columns * 8, last);
                break;
            }
            tile -= tiles;
        }
      }
    }
    __syncthreads();
    if (expert == experts) return;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned row0 = blockIdx.x * 64 + warp * 16 + (lane >> 2);
    const unsigned row1 = row0 + 8;
    const unsigned pair = (lane & 3) * 2;
    const float* input[Columns] = {};
    #pragma unroll
    for (unsigned n = 0; n < Columns; ++n) {
        const unsigned input_index = begin + n * 8 + (lane >> 2);
        if (input_index < end) {
            if constexpr (Routed) input[n] = inputs[sorted_routes[input_index]];
            else input[n] = dense_input + static_cast<size_t>(input_index) * in_features;
        }
    }
    const uint8_t* weight = Routed ? weights[expert] : nullptr;
    const uint8_t* scale = Routed ? scales[expert] : nullptr;
    // Four high-component accumulators shorten the FP32 summation chains.
    float hi[Columns][4][4] = {}, mid[Columns][4] = {}, low[Columns][4] = {};
    for (unsigned base_k = 0; base_k < in_features; base_k += 64) {
      #pragma unroll
      for (unsigned part = 0; part < 4; ++part) {
        const unsigned k = base_k + part * 16;
        if (k >= in_features) continue;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if constexpr (Routed) {
            if (row0 < out_features) {
                lfm2_weight_group(weight, scale, row0, k, in_features, lane, a0, a2);
            }
            if (row1 < out_features) {
                lfm2_weight_group(weight, scale, row1, k, in_features, lane, a1, a3);
            }
        } else if constexpr (DenseFormat == 1) {
            if (row0 < out_features) {
                const auto* w = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(dense_weight)
                    + static_cast<size_t>(row0) * in_features + k + pair);
                const float2 lo = __half22float2(__nv_cvt_fp8x2_to_halfraw2(w[0], __NV_E4M3));
                const float2 hi = __half22float2(__nv_cvt_fp8x2_to_halfraw2(w[4], __NV_E4M3));
                a0 = lfm2_pair(lo.x, lo.y); a2 = lfm2_pair(hi.x, hi.y);
            }
            if (row1 < out_features) {
                const auto* w = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(dense_weight)
                    + static_cast<size_t>(row1) * in_features + k + pair);
                const float2 lo = __half22float2(__nv_cvt_fp8x2_to_halfraw2(w[0], __NV_E4M3));
                const float2 hi = __half22float2(__nv_cvt_fp8x2_to_halfraw2(w[4], __NV_E4M3));
                a1 = lfm2_pair(lo.x, lo.y); a3 = lfm2_pair(hi.x, hi.y);
            }
        } else if constexpr (DenseFormat == 2) {
            if (row0 < out_features) lfm2_weight_group(reinterpret_cast<const uint8_t*>(dense_weight),
                dense_block_scales, row0, k, in_features, lane, a0, a2);
            if (row1 < out_features) lfm2_weight_group(reinterpret_cast<const uint8_t*>(dense_weight),
                dense_block_scales, row1, k, in_features, lane, a1, a3);
        } else {
            if (row0 < out_features) {
                const auto* w = reinterpret_cast<const uint32_t*>(dense_weight
                    + static_cast<size_t>(row0) * in_features + k + pair);
                a0 = w[0]; a2 = w[4];
            }
            if (row1 < out_features) {
                const auto* w = reinterpret_cast<const uint32_t*>(dense_weight
                    + static_cast<size_t>(row1) * in_features + k + pair);
                a1 = w[0]; a3 = w[4];
            }
        }
        #pragma unroll
        for (unsigned n = 0; n < Columns; ++n) {
        float x[4] = {};
        if (input[n]) {
            const float2 lo = *reinterpret_cast<const float2*>(input[n] + k + pair);
            const float2 hi = *reinterpret_cast<const float2*>(input[n] + k + pair + 8);
            x[0] = lo.x; x[1] = lo.y; x[2] = hi.x; x[3] = hi.y;
        }
        float high[4], middle[4], tail[4];
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            high[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
            if constexpr (!Bf16) {
                const float residual = x[i] - high[i];
                middle[i] = __bfloat162float(__float2bfloat16_rn(residual));
                tail[i] = residual - middle[i];
            }
        }
        lfm2_mma(hi[n][part], a0, a1, a2, a3, lfm2_pair(high[0], high[1]), lfm2_pair(high[2], high[3]));
        if constexpr (!Bf16) {
            lfm2_mma(mid[n], a0, a1, a2, a3, lfm2_pair(middle[0], middle[1]), lfm2_pair(middle[2], middle[3]));
            lfm2_mma(low[n], a0, a1, a2, a3, lfm2_pair(tail[0], tail[1]), lfm2_pair(tail[2], tail[3]));
        }
        }
      }
    }
    const float global = Routed ? globals[expert] : dense_scale;
    #pragma unroll
    for (unsigned n = 0; n < Columns; ++n) {
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned route_index = begin + n * 8 + (lane & 3) * 2 + (i & 1);
        const unsigned row = i < 2 ? row0 : row1;
        if (route_index < end && row < out_features) {
            const float high = (hi[n][0][i] + hi[n][1][i]) + (hi[n][2][i] + hi[n][3][i]);
            const float scale = !Routed && dense_channels ? dense_channels[row] : global;
            const float result = (high + (mid[n][i] + low[n][i])) * scale;
            if constexpr (Routed) outputs[sorted_routes[route_index]][row] = result;
            else dense_output[static_cast<size_t>(route_index) * out_features + row] = result;
        }
    }
    }
}
} // namespace

extern "C" cudaError_t infer_lfm2_expert_gemm_on_stream(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned routes, unsigned experts, unsigned out_features, unsigned in_features,
    cudaStream_t stream) {
    if (!sorted_routes || !offsets || !inputs || !weights || !scales || !globals || !outputs
        || routes == 0 || experts == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0) return cudaErrorInvalidValue;
    const dim3 grid((out_features + 63) / 64, (routes + 15) / 16 + experts - 1);
    lfm2_gemm<true, 2><<<grid, 128, 0, stream>>>(sorted_routes, offsets, inputs,
        weights, scales, globals, outputs, experts, out_features, in_features, nullptr, nullptr, nullptr, 0, nullptr, 1.0f, nullptr);
    return cudaGetLastError();
}

__device__ __forceinline__ void lfm2_async16(void* destination, const void* source, bool valid) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(source), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void lfm2_async4(void* destination, const void* source, bool valid) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(address), "l"(source), "r"(valid ? 4 : 0) : "memory");
}
// Every finite E2M1-by-UE4M3 weight product is exactly representable in FP16.
__device__ __forceinline__ uint32_t lfm2_half_weight_pair(uint8_t packed, uint16_t sf) {
    unsigned pair;
    asm("{ .reg .b8 code; cvt.u8.u32 code,%1; cvt.rn.f16x2.e2m1x2 %0,code; }"
        : "=r"(pair) : "r"(unsigned(packed)));
    const float scale = __uint_as_float(unsigned(sf) << 16);
    const unsigned half = __half_as_ushort(__float2half_rn(scale));
    const unsigned halves = half | (half << 16);
    unsigned result;
    asm("mul.f16x2 %0,%1,%2;" : "=r"(result) : "r"(pair), "r"(halves));
    return result;
}
__device__ __forceinline__ void lfm2_half_mma(
    float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool Prepared, unsigned TileRows = 16>
__device__ __forceinline__ void lfm2_stage_expert(
    uint8_t* destination_weight, uint32_t* destination_scale,
    std::conditional_t<Prepared, uint16_t, float>* destination_input,
    const uint8_t* weight, const uint8_t* scale,
    const std::conditional_t<Prepared, uint16_t, float>* const* source,
    unsigned first_row, unsigned out_features, unsigned in_features, unsigned k) {
    unsigned row = threadIdx.x / 2, half = threadIdx.x & 1;
    bool valid = first_row + row < out_features;
    const uint8_t* w = weight + (valid ? size_t(first_row + row) * (in_features / 2) : 0) + k / 2 + half * 16;
    lfm2_async16(destination_weight + row * 48 + half * 16, w, valid);
    if (threadIdx.x < 64) {
        row = threadIdx.x;
        valid = first_row + row < out_features;
        const uint8_t* s = scale + (valid ? size_t(first_row + row) * (in_features / 16) : 0) + k / 16;
        lfm2_async4(destination_scale + row, s, valid);
    }
    if constexpr (Prepared) {
        for (unsigned i = threadIdx.x; i < 2 * TileRows * 8; i += 128) {
            const unsigned component = i / (TileRows * 8);
            row = (i / 8) % TileRows;
            const unsigned col = (i % 8) * 8;
            const uint16_t* x = source[row] ? source[row] : source[0];
            lfm2_async16(destination_input + (component * TileRows + row) * 72 + col,
                x + component * in_features + k + col, source[row] != nullptr);
        }
    } else {
        for (unsigned i = threadIdx.x; i < TileRows * 16; i += 128) {
            row = i / 16;
            const unsigned col = (i % 16) * 4;
            const float* x = source[row] ? source[row] : source[0];
            lfm2_async16(destination_input + row * 72 + col, x + k + col, source[row] != nullptr);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool Prepared, unsigned Columns = 2, bool Bf16 = false>
__global__ void lfm2_expert_gemm_pipeline(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, const uint16_t* prepared, const float* input_powers,
    const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned experts, unsigned out_features, unsigned in_features) {
    static_assert(!Prepared || Columns == 2);
    constexpr unsigned TileRows = Columns * 8;
    __shared__ unsigned expert, begin, end;
    __shared__ __align__(16) uint8_t packed[2][64][48];
    __shared__ uint32_t scale_tile[2][64];
    __shared__ __align__(16) std::conditional_t<Prepared, uint16_t, float>
        input_tile[2][Prepared ? 2 : 1][TileRows][72];
    __shared__ const std::conditional_t<Prepared, uint16_t, float>* source[TileRows];
    if (threadIdx.x == 0) {
        unsigned tile = blockIdx.y;
        expert = experts;
        for (unsigned e = 0; e < experts; ++e) {
            const unsigned first = offsets[e], last = offsets[e + 1];
            const unsigned tiles = (last - first + TileRows - 1) / TileRows;
            if (tile < tiles) {
                expert = e; begin = first + tile * TileRows; end = min(begin + TileRows, last);
                break;
            }
            tile -= tiles;
        }
    }
    __syncthreads();
    if (expert == experts) return;
    if (threadIdx.x < TileRows) {
        if (begin + threadIdx.x >= end) source[threadIdx.x] = nullptr;
        else if constexpr (Prepared) source[threadIdx.x] = prepared
            + size_t(sorted_routes[begin + threadIdx.x]) * 2 * in_features;
        else source[threadIdx.x] = inputs[sorted_routes[begin + threadIdx.x]];
    }
    __syncthreads();
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned row0 = warp * 16 + (lane >> 2), row1 = row0 + 8;
    const unsigned pair = (lane & 3) * 2, input_row = lane >> 2;
    const uint8_t* weight = weights[expert];
    const uint8_t* scale = scales[expert];
    lfm2_stage_expert<Prepared, TileRows>(&packed[0][0][0], &scale_tile[0][0], &input_tile[0][0][0][0],
        weight, scale, source, blockIdx.x * 64, out_features, in_features, 0);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float hi[Columns][4][4] = {}, mid[Columns][4] = {}, low[Columns][4] = {};
    // Copy the next compressed tile while tensor cores consume this stage.
    for (unsigned base = 0, stage = 0; base < in_features; base += 64, stage ^= 1) {
        if (base + 64 < in_features) lfm2_stage_expert<Prepared, TileRows>(
            &packed[stage ^ 1][0][0], &scale_tile[stage ^ 1][0], &input_tile[stage ^ 1][0][0][0],
            weight, scale, source, blockIdx.x * 64, out_features, in_features, base + 64);
        #pragma unroll
        for (unsigned p = 0; p < 4; ++p) {
            const uint2 w0 = *reinterpret_cast<const uint2*>(&packed[stage][row0][p * 8]);
            const uint2 w1 = *reinterpret_cast<const uint2*>(&packed[stage][row1][p * 8]);
            const uint16_t s0 = lfm2_scale_bf16(uint8_t(scale_tile[stage][row0] >> (p * 8)));
            const uint16_t s1 = lfm2_scale_bf16(uint8_t(scale_tile[stage][row1] >> (p * 8)));
            const unsigned shift = (lane & 3) * 8;
            const uint32_t a0 = Prepared ? lfm2_half_weight_pair(uint8_t(w0.x >> shift), s0)
                : lfm2_weight_pair(uint8_t(w0.x >> shift), s0);
            const uint32_t a1 = Prepared ? lfm2_half_weight_pair(uint8_t(w1.x >> shift), s1)
                : lfm2_weight_pair(uint8_t(w1.x >> shift), s1);
            const uint32_t a2 = Prepared ? lfm2_half_weight_pair(uint8_t(w0.y >> shift), s0)
                : lfm2_weight_pair(uint8_t(w0.y >> shift), s0);
            const uint32_t a3 = Prepared ? lfm2_half_weight_pair(uint8_t(w1.y >> shift), s1)
                : lfm2_weight_pair(uint8_t(w1.y >> shift), s1);
            if constexpr (Prepared) {
                uint32_t parts[2][4];
                #pragma unroll
                for (unsigned component = 0; component < 2; ++component) {
                    const uint16_t* ptr = &input_tile[stage][component]
                        [(lane & 7) + (lane / 16) * 8][p * 16 + ((lane / 8) & 1) * 8];
                    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(parts[component][0]), "=r"(parts[component][1]), "=r"(parts[component][2]), "=r"(parts[component][3]) : "r"(address));
                }
                #pragma unroll
                for (unsigned n = 0; n < Columns; ++n) {
                    lfm2_half_mma(hi[n][p], a0, a1, a2, a3, parts[0][2 * n], parts[0][2 * n + 1]);
                    lfm2_half_mma(mid[n], a0, a1, a2, a3, parts[1][2 * n], parts[1][2 * n + 1]);
                }
            } else {
                #pragma unroll
                for (unsigned n = 0; n < Columns; ++n) {
                    const float* x = &input_tile[stage][0][input_row + n * 8][p * 16 + pair];
                    const float2 v0 = *reinterpret_cast<const float2*>(x);
                    const float2 v1 = *reinterpret_cast<const float2*>(x + 8);
                    float v[4] = {v0.x, v0.y, v1.x, v1.y};
                    float high[4], middle[4], tail[4];
                    #pragma unroll
                    for (unsigned i = 0; i < 4; ++i) {
                        high[i] = __bfloat162float(__float2bfloat16_rn(v[i]));
                        if constexpr (!Bf16) {
                            const float residual = v[i] - high[i];
                            middle[i] = __bfloat162float(__float2bfloat16_rn(residual));
                            tail[i] = residual - middle[i];
                        }
                    }
                    lfm2_mma(hi[n][p], a0, a1, a2, a3, lfm2_pair(high[0], high[1]), lfm2_pair(high[2], high[3]));
                    if constexpr (!Bf16) {
                        lfm2_mma(mid[n], a0, a1, a2, a3, lfm2_pair(middle[0], middle[1]), lfm2_pair(middle[2], middle[3]));
                        lfm2_mma(low[n], a0, a1, a2, a3, lfm2_pair(tail[0], tail[1]), lfm2_pair(tail[2], tail[3]));
                    }
                }
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    const float global = globals[expert];
    #pragma unroll
    for (unsigned n = 0; n < Columns; ++n) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned route = begin + n * 8 + (lane & 3) * 2 + (i & 1);
            const unsigned row = blockIdx.x * 64 + (i < 2 ? row0 : row1);
            if (route < end && row < out_features) {
                const float high = (hi[n][0][i] + hi[n][1][i]) + (hi[n][2][i] + hi[n][3][i]);
                if constexpr (Prepared) outputs[sorted_routes[route]][row] =
                    ((high + mid[n][i]) * input_powers[sorted_routes[route]]) * global;
                else outputs[sorted_routes[route]][row] = (high + (mid[n][i] + low[n][i])) * global;
            }
        }
    }
}

extern "C" cudaError_t infer_lfm2_expert_gemm_pipelined_on_stream(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned routes, unsigned experts, unsigned out_features, unsigned in_features,
    cudaStream_t stream) {
    if (!sorted_routes || !offsets || !inputs || !weights || !scales || !globals || !outputs
        || routes == 0 || experts == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0) return cudaErrorInvalidValue;
    const dim3 grid((out_features + 63) / 64, (routes + 15) / 16 + experts - 1);
    if (in_features % 64 == 0) {
        if (experts >= 240 && routes >= 4096 && out_features <= 3072 && in_features <= 3072) {
            const dim3 wide_grid((out_features + 63) / 64, (routes + 31) / 32 + experts - 1);
            lfm2_expert_gemm_pipeline<false, 4><<<wide_grid, 128, 0, stream>>>(sorted_routes, offsets, inputs,
                nullptr, nullptr, weights, scales, globals, outputs, experts, out_features, in_features);
            return cudaGetLastError();
        }
        lfm2_expert_gemm_pipeline<false><<<grid, 128, 0, stream>>>(sorted_routes, offsets, inputs, nullptr, nullptr,
            weights, scales, globals, outputs, experts, out_features, in_features);
    } else {
        lfm2_gemm<true, 2><<<grid, 128, 0, stream>>>(sorted_routes, offsets, inputs,
            weights, scales, globals, outputs, experts, out_features, in_features, nullptr, nullptr, nullptr, 0, nullptr, 1.0f, nullptr);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t infer_lfm2_expert_gemm_bf16_on_stream(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned routes, unsigned experts, unsigned out_features, unsigned in_features,
    cudaStream_t stream) {
    if (!sorted_routes || !offsets || !inputs || !weights || !scales || !globals || !outputs
        || routes == 0 || experts == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0) return cudaErrorInvalidValue;
    const dim3 grid((out_features + 63) / 64, (routes + 15) / 16 + experts - 1);
    if (in_features % 64 == 0) {
        if (experts >= 240 && routes >= 4096 && out_features <= 3072 && in_features <= 3072) {
            const dim3 wide_grid((out_features + 63) / 64, (routes + 31) / 32 + experts - 1);
            lfm2_expert_gemm_pipeline<false, 4, true><<<wide_grid, 128, 0, stream>>>(sorted_routes, offsets, inputs,
                nullptr, nullptr, weights, scales, globals, outputs, experts, out_features, in_features);
            return cudaGetLastError();
        }
        lfm2_expert_gemm_pipeline<false, 2, true><<<grid, 128, 0, stream>>>(sorted_routes, offsets, inputs, nullptr, nullptr,
            weights, scales, globals, outputs, experts, out_features, in_features);
    } else {
        lfm2_gemm<true, 2, 0, true><<<grid, 128, 0, stream>>>(sorted_routes, offsets, inputs,
            weights, scales, globals, outputs, experts, out_features, in_features, nullptr, nullptr, nullptr, 0, nullptr, 1.0f, nullptr);
    }
    return cudaGetLastError();
}

// Power-of-two scaling keeps the leading component near 32 and its residual
// in FP16 range. Separate accumulators retain the low component through GEMM.
__global__ void lfm2_prepare_expert_inputs(
    const float* const* inputs, uint16_t* output, float* power, unsigned k) {
    const unsigned row = blockIdx.x;
    __shared__ float maxima[8];
    float maximum = 0;
    for (unsigned col = threadIdx.x; col < k; col += 256)
        maximum = fmaxf(maximum, fabsf(inputs[row][col]));
    for (unsigned offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = maximum;
    __syncthreads();
    maximum = 0;
    #pragma unroll
    for (unsigned warp = 0; warp < 8; ++warp) maximum = fmaxf(maximum, maxima[warp]);
    int exponent;
    frexpf(maximum, &exponent);
    const float scale = maximum > 0 ? ldexpf(1.0f, max(-126, exponent - 6)) : 1.0f;
    if (threadIdx.x == 0) power[row] = scale;
    for (unsigned col = threadIdx.x; col < k; col += 256) {
        const float x = inputs[row][col] / scale;
        const __half high = __float2half_rn(x);
        const __half low = __float2half_rn(x - __half2float(high));
        output[(size_t(row) * 2) * k + col] = __half_as_ushort(high);
        output[(size_t(row) * 2 + 1) * k + col] = __half_as_ushort(low);
    }
}

extern "C" cudaError_t infer_lfm2_expert_gemm_f16_parts_on_stream(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, uint16_t* prepared, float* input_powers,
    const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned routes, unsigned experts, unsigned out_features, unsigned in_features,
    cudaStream_t stream) {
    if (!sorted_routes || !offsets || !inputs || !prepared || !input_powers || !weights || !scales || !globals
        || !outputs || routes == 0 || experts == 0 || out_features == 0 || in_features == 0
        || in_features % 64 != 0) return cudaErrorInvalidValue;
    lfm2_prepare_expert_inputs<<<routes, 256, 0, stream>>>(
        inputs, prepared, input_powers, in_features);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const dim3 grid((out_features + 63) / 64, (routes + 15) / 16 + experts - 1);
    lfm2_expert_gemm_pipeline<true><<<grid, 128, 0, stream>>>(sorted_routes, offsets, nullptr,
        prepared, input_powers, weights, scales, globals, outputs, experts, out_features, in_features);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_lfm2_dense_gemm_on_stream(
    const float* input, const uint16_t* weight, float* output,
    unsigned rows, unsigned out_features, unsigned in_features, cudaStream_t stream) {
    if (!input || !weight || !output || rows == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0) return cudaErrorInvalidValue;
    lfm2_gemm<false><<<dim3((out_features + 63) / 64, (rows + 7) / 8), 128, 0, stream>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, out_features,
        in_features, input, weight, output, rows, nullptr, 1.0f, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_lfm2_dense_gemm_bf16_on_stream(
    const float* input, const uint16_t* weight, float* output,
    unsigned rows, unsigned out_features, unsigned in_features, cudaStream_t stream) {
    if (!input || !weight || !output || rows == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0) return cudaErrorInvalidValue;
    lfm2_gemm<false, 1, 0, true><<<dim3((out_features + 63) / 64, (rows + 7) / 8), 128, 0, stream>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, out_features,
        in_features, input, weight, output, rows, nullptr, 1.0f, nullptr);
    return cudaGetLastError();
}

// Independent KV partitions keep the head-64 decode work distributed over SMs.
__global__ void lfm2_decode_attention_parts(
    const float* query, const float* key, const float* value, const uint32_t* length,
    float* partial, unsigned q_heads, unsigned kv_heads, unsigned partitions) {
    constexpr unsigned Warps = 4;
    __shared__ float accum[Warps][64];
    __shared__ float maxima[Warps], sums[Warps];
    const unsigned head = blockIdx.x, part = blockIdx.y;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned count = *length;
    const unsigned span = (count + partitions - 1) / partitions;
    const unsigned start = part * span, end = min(start + span, count);
    const unsigned kv_head = head / (q_heads / kv_heads);
    const float q0 = query[head * 64 + lane], q1 = query[head * 64 + lane + 32];
    float maximum = -INFINITY, sum = 0, acc0 = 0, acc1 = 0;
    for (unsigned row = start + warp; row < end; row += Warps) {
        const size_t base = (static_cast<size_t>(row) * kv_heads + kv_head) * 64 + lane;
        float dot0 = q0 * key[base], dot1 = q1 * key[base + 32];
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            dot0 += __shfl_xor_sync(0xffffffffu, dot0, delta);
            dot1 += __shfl_xor_sync(0xffffffffu, dot1, delta);
        }
        const float score = (dot0 + dot1) * .125f;
        const float next_max = fmaxf(maximum, score);
        const float rescale = expf(maximum - next_max), weight = expf(score - next_max);
        sum = sum * rescale + weight;
        acc0 = fmaf(weight, value[base], acc0 * rescale);
        acc1 = fmaf(weight, value[base + 32], acc1 * rescale);
        maximum = next_max;
    }
    accum[warp][lane] = acc0; accum[warp][lane + 32] = acc1;
    if (lane == 0) { maxima[warp] = maximum; sums[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float top = -INFINITY;
        #pragma unroll
        for (unsigned w = 0; w < Warps; ++w) top = fmaxf(top, maxima[w]);
        float total = 0, out0 = 0, out1 = 0;
        #pragma unroll
        for (unsigned w = 0; w < Warps; ++w) {
            const float scale = sums[w] > 0 ? expf(maxima[w] - top) : 0;
            total += sums[w] * scale;
            out0 = fmaf(accum[w][lane], scale, out0);
            out1 = fmaf(accum[w][lane + 32], scale, out1);
        }
        const size_t base = (static_cast<size_t>(head) * partitions + part) * 66;
        partial[base + lane] = out0; partial[base + lane + 32] = out1;
        if (lane == 0) { partial[base + 64] = top; partial[base + 65] = total; }
    }
}
__global__ void lfm2_decode_attention_merge(const float* partial, float* output, unsigned partitions) {
    const unsigned head = blockIdx.x, lane = threadIdx.x;
    const float* base = partial + static_cast<size_t>(head) * partitions * 66;
    float maximum = -INFINITY;
    for (unsigned p = 0; p < partitions; ++p) maximum = fmaxf(maximum, base[p * 66 + 64]);
    float sum = 0, out0 = 0, out1 = 0;
    for (unsigned p = 0; p < partitions; ++p) {
        const float* v = base + p * 66;
        const float scale = v[65] > 0 ? expf(v[64] - maximum) : 0;
        sum += v[65] * scale;
        out0 = fmaf(v[lane], scale, out0); out1 = fmaf(v[lane + 32], scale, out1);
    }
    const float inv = sum > 0 ? 1.0f / sum : 0;
    output[head * 64 + lane] = out0 * inv; output[head * 64 + lane + 32] = out1 * inv;
}
extern "C" cudaError_t infer_lfm2_decode_attention_on_stream(
    const float* query, const float* key, const float* value, const uint32_t* length,
    float* partial, float* output, unsigned q_heads, unsigned kv_heads,
    unsigned partitions, cudaStream_t stream) {
    if (!query || !key || !value || !length || !partial || !output || q_heads == 0
        || kv_heads == 0 || q_heads % kv_heads || partitions == 0 || partitions > 64)
        return cudaErrorInvalidValue;
    lfm2_decode_attention_parts<<<dim3(q_heads, partitions), 128, 0, stream>>>(
        query, key, value, length, partial, q_heads, kv_heads, partitions);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    lfm2_decode_attention_merge<<<q_heads, 32, 0, stream>>>(partial, output, partitions);
    return cudaGetLastError();
}

// Each warp computes eight queries and reuses K/V loads across them. The block
// shares 32 K/V rows across 64 queries; keys are transposed to avoid bank conflicts.
__global__ void lfm2_prefill_attention_tiled(
    const float* query, const float* key, const float* value, float* output,
    unsigned tokens, unsigned start_position, unsigned q_heads, unsigned kv_heads) {
    constexpr unsigned Queries = 8, Warps = 8, Rows = Queries * Warps;
    __shared__ float keys[64][33], values[32][64];
    __shared__ float queries[Rows][64], weights[Rows][32];
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned head = blockIdx.y, first_query = blockIdx.x * Rows;
    const unsigned kv_head = head / (q_heads / kv_heads);
    const unsigned end = start_position + min(first_query + Rows, tokens);
    float maximum[Queries], sum[Queries] = {}, acc0[Queries] = {}, acc1[Queries] = {};
    #pragma unroll
    for (unsigned n = 0; n < Queries; ++n) {
        const unsigned row = first_query + warp + n * Warps;
        const size_t base = (size_t(row) * q_heads + head) * 64;
        queries[warp + n * Warps][lane] = row < tokens ? query[base + lane] : 0;
        queries[warp + n * Warps][lane + 32] = row < tokens ? query[base + lane + 32] : 0;
        maximum[n] = -INFINITY;
    }
    for (unsigned first_key = 0; first_key < end; first_key += 32) {
        for (unsigned i = threadIdx.x; i < 32 * 64; i += blockDim.x) {
            const unsigned token = first_key + i / 64, d = i % 64;
            const size_t base = (size_t(token) * kv_heads + kv_head) * 64 + d;
            keys[d][i / 64] = token < end ? key[base] : 0;
            values[i / 64][d] = token < end ? value[base] : 0;
        }
        __syncthreads();
        float dot[Queries] = {};
        #pragma unroll
        for (unsigned d = 0; d < 64; ++d) {
            const float k = keys[d][lane];
            #pragma unroll
            for (unsigned n = 0; n < Queries; ++n)
                dot[n] = fmaf(queries[warp + n * Warps][d], k, dot[n]);
        }
        #pragma unroll
        for (unsigned n = 0; n < Queries; ++n) {
            const unsigned row = first_query + warp + n * Warps;
            const bool valid = row < tokens && first_key + lane <= start_position + row;
            const float score = valid ? dot[n] * .125f : -INFINITY;
            float tile_max = score;
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, delta));
            const float next_max = fmaxf(maximum[n], tile_max);
            const float rescale = maximum[n] != -INFINITY ? expf(maximum[n] - next_max) : 0;
            const float weight = valid ? expf(score - next_max) : 0;
            weights[warp + n * Warps][lane] = weight;
            float tile_sum = weight;
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, delta);
            sum[n] = sum[n] * rescale + tile_sum;
            maximum[n] = next_max;
            acc0[n] *= rescale; acc1[n] *= rescale;
        }
        __syncwarp();
        #pragma unroll
        for (unsigned t = 0; t < 32; ++t) {
            const float v0 = values[t][lane], v1 = values[t][lane + 32];
            #pragma unroll
            for (unsigned n = 0; n < Queries; ++n) {
                const float weight = weights[warp + n * Warps][t];
                acc0[n] = fmaf(weight, v0, acc0[n]);
                acc1[n] = fmaf(weight, v1, acc1[n]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned n = 0; n < Queries; ++n) {
        const unsigned row = first_query + warp + n * Warps;
        const size_t base = (size_t(row) * q_heads + head) * 64;
        if (row < tokens) {
            output[base + lane] = acc0[n] / sum[n];
            output[base + lane + 32] = acc1[n] / sum[n];
        }
    }
}
extern "C" cudaError_t infer_lfm2_prefill_attention_on_stream(
    const float* query, const float* key, const float* value, float* output,
    unsigned tokens, unsigned start_position, unsigned q_heads, unsigned kv_heads, cudaStream_t stream) {
    lfm2_prefill_attention_tiled<<<dim3((tokens + 63) / 64, q_heads), 256, 0, stream>>>(
        query, key, value, output, tokens, start_position, q_heads, kv_heads);
    return cudaGetLastError();
}

// FP8 and NVFP4 stay compressed in device memory. Tensor-core operands decode
// exactly to BF16; three BF16 components preserve the F32 activation mantissa.
extern "C" cudaError_t infer_quantized_dense_gemm_f32_on_stream(
    const float* input, const uint8_t* weight, const uint8_t* block_scales,
    const float* channel_scales, float global_scale, float* output,
    unsigned rows, unsigned out_features, unsigned in_features, unsigned format,
    cudaStream_t stream) {
    if (!input || !weight || !output || rows == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0 || format < 1 || format > 2 || (format == 2 && !block_scales))
        return cudaErrorInvalidValue;
    const dim3 grid((out_features + 63) / 64, (rows + 7) / 8);
    if (format == 1) {
        lfm2_gemm<false, 1, 1><<<grid, 128, 0, stream>>>(nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, 1, out_features, in_features,
            input, reinterpret_cast<const uint16_t*>(weight), output, rows,
            channel_scales, global_scale, nullptr);
    } else {
        lfm2_gemm<false, 1, 2><<<grid, 128, 0, stream>>>(nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, 1, out_features, in_features,
            input, reinterpret_cast<const uint16_t*>(weight), output, rows,
            nullptr, global_scale, block_scales);
    }
    return cudaGetLastError();
}

// Small BF16 batches reuse each packed FP8 weight across every active row.
// Each row has the same reduction order in serial and speculative decode.
template <unsigned Rows>
__global__ void fp8_bf16_rows(const float* input, const uint8_t* weight,
    const float* channel_scales, float scale, float* output,
    unsigned rows, unsigned out, unsigned cols) {
    __shared__ float partials[Rows][8];
    const unsigned channel = blockIdx.x, lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    const auto* packed = reinterpret_cast<const uint32_t*>(weight + size_t(channel) * cols);
    float sums[Rows] = {};
    for (unsigned k = threadIdx.x; k < cols / 4; k += 256) {
        const uint32_t bits = packed[k];
        const float2 lo = __half22float2(__nv_cvt_fp8x2_to_halfraw2(bits & 0xffff, __NV_E4M3));
        const float2 hi = __half22float2(__nv_cvt_fp8x2_to_halfraw2(bits >> 16, __NV_E4M3));
        #pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            if (row >= rows) continue;
            const float4 x = reinterpret_cast<const float4*>(input + size_t(row) * cols)[k];
            sums[row] = fmaf(__bfloat162float(__float2bfloat16_rn(x.x)), lo.x, sums[row]);
            sums[row] = fmaf(__bfloat162float(__float2bfloat16_rn(x.y)), lo.y, sums[row]);
            sums[row] = fmaf(__bfloat162float(__float2bfloat16_rn(x.z)), hi.x, sums[row]);
            sums[row] = fmaf(__bfloat162float(__float2bfloat16_rn(x.w)), hi.y, sums[row]);
        }
    }
    #pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        for (unsigned delta = 16; delta; delta >>= 1)
            sums[row] += __shfl_down_sync(0xffffffff, sums[row], delta);
        if (lane == 0) partials[row][warp] = sums[row];
    }
    __syncthreads();
    if (warp == 0) {
        const float multiplier = channel_scales ? channel_scales[channel] : scale;
        #pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            float sum = lane < 8 ? partials[row][lane] : 0.f;
            for (unsigned delta = 16; delta; delta >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, delta);
            if (lane == 0 && row < rows) output[size_t(row) * out + channel] = sum * multiplier;
        }
    }
}

extern "C" cudaError_t infer_fp8_bf16_rows_on_stream(
    const float* input, const uint8_t* weight, const float* channel_scales,
    float scale, float* output, unsigned rows, unsigned out, unsigned cols,
    cudaStream_t stream) {
    if (!input || !weight || !output || rows == 0 || rows > 8 || out == 0 ||
        cols == 0 || cols % 4 || !std::isfinite(scale)) return cudaErrorInvalidValue;
    if (rows == 1) fp8_bf16_rows<1><<<out, 256, 0, stream>>>(
        input, weight, channel_scales, scale, output, rows, out, cols);
    else if (rows <= 4) fp8_bf16_rows<4><<<out, 256, 0, stream>>>(
        input, weight, channel_scales, scale, output, rows, out, cols);
    else fp8_bf16_rows<8><<<out, 256, 0, stream>>>(
        input, weight, channel_scales, scale, output, rows, out, cols);
    return cudaGetLastError();
}

// Expand one dense projection into reusable scratch. Global/channel scales
// remain F32 and are applied after GEMM, avoiding a second weight rounding.
__global__ void expand_dense_bf16(const uint8_t* weights, const uint8_t* scales,
    uint32_t* output, unsigned pairs) {
    const unsigned pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    if (scales) {
        output[pair] = lfm2_weight_pair(weights[pair], lfm2_scale_bf16(scales[pair / 8]));
    } else {
        const uint16_t packed = reinterpret_cast<const uint16_t*>(weights)[pair];
        const float2 v = __half22float2(__nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3));
        output[pair] = lfm2_pair(v.x, v.y);
    }
}
extern "C" cudaError_t infer_expand_dense_bf16_on_stream(
    const uint8_t* weights, const uint8_t* scales, uint16_t* output,
    unsigned elements, cudaStream_t stream) {
    if (!weights || !output || !elements || elements % 16) return cudaErrorInvalidValue;
    expand_dense_bf16<<<(elements / 2 + 255) / 256, 256, 0, stream>>>(
        weights, scales, reinterpret_cast<uint32_t*>(output), elements / 2);
    return cudaGetLastError();
}
