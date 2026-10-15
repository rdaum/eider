#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

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
__device__ __forceinline__ float lfm2_e2m1(uint8_t code) {
    const uint32_t mag = code & 7, exponent = mag >> 1, mantissa = mag & 1;
    const uint32_t bits = exponent == 0 ? mantissa * 0x3f000000u
        : ((exponent + 126u) << 23u) | (mantissa << 22u);
    return __uint_as_float(bits | (static_cast<uint32_t>(code & 8) << 28));
}
__device__ __forceinline__ float lfm2_e4m3(uint8_t code) {
    const uint32_t exponent = (code >> 3) & 15, mantissa = code & 7;
    return exponent == 0 ? mantissa * 0x1p-9f
        : __uint_as_float(((exponent + 120u) << 23u) | (mantissa << 20u));
}
__device__ __forceinline__ uint32_t lfm2_pair(float x, float y) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(x))
        | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(y))) << 16);
}
__device__ __forceinline__ uint32_t lfm2_weight_pair(
    const uint8_t* weight, const uint8_t* scales, unsigned row, unsigned k, unsigned cols) {
    const uint8_t packed = weight[static_cast<size_t>(row) * (cols / 2) + k / 2];
    const float scale = lfm2_e4m3(scales[static_cast<size_t>(row) * (cols / 16) + k / 16]);
    return lfm2_pair(lfm2_e2m1(packed & 15) * scale, lfm2_e2m1(packed >> 4) * scale);
}
__device__ __forceinline__ void lfm2_mma(float* d, uint32_t a0, uint32_t a1,
                                       uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool Routed>
__global__ void lfm2_gemm(
    const uint32_t* sorted_routes, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights,
    const uint8_t* const* scales, const float* globals, float* const* outputs,
    unsigned experts, unsigned out_features, unsigned in_features,
    const float* dense_input, const uint16_t* dense_weight, float* dense_output, unsigned rows) {
    __shared__ unsigned expert, begin, end;
    if (threadIdx.x == 0) {
      if constexpr (!Routed) {
        expert = 0; begin = blockIdx.y * 8; end = min(begin + 8, rows);
      } else {
        unsigned tile = blockIdx.y;
        expert = experts;
        for (unsigned e = 0; e < experts; ++e) {
            const unsigned first = offsets[e], last = offsets[e + 1];
            const unsigned tiles = (last - first + 7) / 8;
            if (tile < tiles) {
                expert = e;
                begin = first + tile * 8;
                end = min(begin + 8, last);
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
    const unsigned input_index = begin + (lane >> 2);
    const float* input = nullptr;
    if (input_index < end) {
        if constexpr (Routed) input = inputs[sorted_routes[input_index]];
        else input = dense_input + static_cast<size_t>(input_index) * in_features;
    }
    const uint8_t* weight = Routed ? weights[expert] : nullptr;
    const uint8_t* scale = Routed ? scales[expert] : nullptr;
    // Four high-component accumulators shorten the FP32 summation chains.
    float hi[4][4] = {}, mid[4] = {}, low[4] = {};
    for (unsigned base_k = 0; base_k < in_features; base_k += 64) {
      #pragma unroll
      for (unsigned part = 0; part < 4; ++part) {
        const unsigned k = base_k + part * 16;
        if (k >= in_features) continue;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if constexpr (Routed) {
            if (row0 < out_features) {
                a0 = lfm2_weight_pair(weight, scale, row0, k + pair, in_features);
                a2 = lfm2_weight_pair(weight, scale, row0, k + pair + 8, in_features);
            }
            if (row1 < out_features) {
                a1 = lfm2_weight_pair(weight, scale, row1, k + pair, in_features);
                a3 = lfm2_weight_pair(weight, scale, row1, k + pair + 8, in_features);
            }
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
        float x[4] = {};
        if (input) {
            x[0] = input[k + pair]; x[1] = input[k + pair + 1];
            x[2] = input[k + pair + 8]; x[3] = input[k + pair + 9];
        }
        float high[4], middle[4], tail[4];
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            high[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
            const float residual = x[i] - high[i];
            middle[i] = __bfloat162float(__float2bfloat16_rn(residual));
            tail[i] = residual - middle[i];
        }
        lfm2_mma(hi[part], a0, a1, a2, a3, lfm2_pair(high[0], high[1]), lfm2_pair(high[2], high[3]));
        lfm2_mma(mid, a0, a1, a2, a3, lfm2_pair(middle[0], middle[1]), lfm2_pair(middle[2], middle[3]));
        lfm2_mma(low, a0, a1, a2, a3, lfm2_pair(tail[0], tail[1]), lfm2_pair(tail[2], tail[3]));
      }
    }
    const float global = Routed ? globals[expert] : 1.0f;
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned route_index = begin + (lane & 3) * 2 + (i & 1);
        const unsigned row = i < 2 ? row0 : row1;
        if (route_index < end && row < out_features) {
            const float high = (hi[0][i] + hi[1][i]) + (hi[2][i] + hi[3][i]);
            const float result = (high + (mid[i] + low[i])) * global;
            if constexpr (Routed) outputs[sorted_routes[route_index]][row] = result;
            else dense_output[static_cast<size_t>(route_index) * out_features + row] = result;
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
    const dim3 grid((out_features + 63) / 64, (routes + 7) / 8 + experts - 1);
    lfm2_gemm<true><<<grid, 128, 0, stream>>>(sorted_routes, offsets, inputs,
        weights, scales, globals, outputs, experts, out_features, in_features, nullptr, nullptr, nullptr, 0);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_lfm2_dense_gemm_on_stream(
    const float* input, const uint16_t* weight, float* output,
    unsigned rows, unsigned out_features, unsigned in_features, cudaStream_t stream) {
    if (!input || !weight || !output || rows == 0 || out_features == 0 || in_features == 0
        || in_features % 16 != 0) return cudaErrorInvalidValue;
    lfm2_gemm<false><<<dim3((out_features + 63) / 64, (rows + 7) / 8), 128, 0, stream>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, out_features,
        in_features, input, weight, output, rows);
    return cudaGetLastError();
}
