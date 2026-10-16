#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
template<unsigned Format> constexpr unsigned block_elements = Format == 0 ? 32 : 256;
template<unsigned Format> constexpr unsigned block_bytes = Format == 0 ? 34 : 66;

__device__ __forceinline__ unsigned load_u32_unaligned(const uint8_t* p) {
    const auto* h = reinterpret_cast<const uint16_t*>(p);
    return unsigned(h[0]) | (unsigned(h[1]) << 16);
}
__device__ __forceinline__ float block_scale(const uint8_t* block) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(block)));
}
// The IQ2 integer includes its odd subgroup multiplier. Its largest magnitude
// is 43*31, which is exactly representable in FP16. The block scale stays F32.
template<unsigned Format>
__device__ __forceinline__ int integer_weight(const uint8_t* block, unsigned k,
                                             const uint64_t* grid) {
    if constexpr (Format == 0) return int(reinterpret_cast<const int8_t*>(block + 2)[k]);
    else {
        const uint8_t* group = block + 2 + (k / 32) * 8;
        const unsigned packed = load_u32_unaligned(group + 4);
        const unsigned part = (k % 32) / 8;
        const unsigned bits = (packed >> (7 * part)) & 127;
        const unsigned signs = bits | ((__popc(bits) & 1) << 7);
        const int value = int((__ldg(grid + group[part]) >> ((k & 7) * 8)) & 255);
        const int scaled = value * int(2 * (packed >> 28) + 1);
        return signs & (1u << (k & 7)) ? -scaled : scaled;
    }
}
template<unsigned Format>
__device__ __forceinline__ float weight_scale(const uint8_t* block) {
    return block_scale(block) * (Format == 0 ? 1.f : 0.125f);
}

template<unsigned Format>
__device__ __forceinline__ uint32_t integer_pair(const uint8_t* block, unsigned k, const uint64_t* grid) {
    int a, b;
    if constexpr (Format == 0) {
        const auto* values = reinterpret_cast<const int8_t*>(block + 2);
        a = values[k]; b = values[k + 1];
    } else {
        const uint8_t* group = block + 2 + (k / 32) * 8;
        const unsigned packed = load_u32_unaligned(group + 4);
        const unsigned part = (k % 32) / 8;
        const unsigned bits = (packed >> (7 * part)) & 127;
        const unsigned signs = (bits | ((__popc(bits) & 1) << 7)) >> (k & 7);
        const unsigned values = unsigned(__ldg(grid + group[part]) >> ((k & 7) * 8));
        const int multiplier = int(2 * (packed >> 28) + 1);
        a = int(values & 255) * multiplier;
        b = int((values >> 8) & 255) * multiplier;
        if (signs & 1) a = -a;
        if (signs & 2) b = -b;
    }
    return unsigned(__half_as_ushort(__int2half_rn(a)))
        | (unsigned(__half_as_ushort(__int2half_rn(b))) << 16);
}

template<unsigned Format, bool Bf16>
__device__ __forceinline__ uint32_t tensor_pair(const uint8_t* block, unsigned k, const uint64_t* grid) {
    const uint32_t pair = integer_pair<Format>(block, k, grid);
    if constexpr (!Bf16) return pair;
    const float scale = weight_scale<Format>(block);
    return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(__half2float(__ushort_as_half(pair)) * scale)))
        | (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(__half2float(__ushort_as_half(pair >> 16)) * scale))) << 16);
}

// Dense and routed decode retain compressed weights, F32 inputs and a fixed
// warp reduction. Four rows reuse each weight load and dequantization.
template<unsigned Format, bool Routed, unsigned Routes = 4>
__global__ void packed_matvec(
    const float* dense_input, const uint8_t* dense_weight, float* dense_output,
    const uint32_t* sorted, const uint32_t* sorted_experts, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights, float* const* outputs,
    const uint64_t* grid, unsigned rows, unsigned out, unsigned cols) {
    constexpr unsigned Warps = 8;
    const unsigned first = Routed ? blockIdx.y : blockIdx.y * Routes;
    const unsigned expert = Routed ? sorted_experts[first] : 0;
    if constexpr (Routed) if ((first - offsets[expert]) % Routes) return;
    const unsigned count = min(Routes, (Routed ? offsets[expert + 1] : rows) - first);
    const unsigned row = blockIdx.x * Warps + threadIdx.x / 32;
    if (row >= out) return;
    const unsigned lane = threadIdx.x & 31;
    const uint8_t* weight = (Routed ? weights[expert] : dense_weight)
        + size_t(row) * (cols / block_elements<Format>) * block_bytes<Format>;
    const float* x[Routes] = {};
    #pragma unroll
    for (unsigned r = 0; r < Routes; ++r) if (r < count)
        x[r] = Routed ? inputs[sorted[first + r]] : dense_input + size_t(first + r) * cols;
    float accum[Routes][4] = {};
    constexpr unsigned Values = Format == 0 ? 4 : 8;
    for (unsigned k = lane * Values; k < cols; k += 32 * Values) {
        const uint8_t* block = weight + (k / block_elements<Format>) * block_bytes<Format>;
        const unsigned local = k % block_elements<Format>;
        float w[Values];
        if constexpr (Format == 1) {
            const uint8_t* group = block + 2 + (local / 32) * 8;
            const unsigned packed = load_u32_unaligned(group + 4);
            const unsigned part = (local % 32) / 8;
            const unsigned bits = (packed >> (7 * part)) & 127;
            const unsigned signs = bits | ((__popc(bits) & 1) << 7);
            const uint64_t codes = __ldg(grid + group[part]);
            const float scale = block_scale(block) * (float(packed >> 28) + .5f) * .25f;
            #pragma unroll
            for (unsigned j = 0; j < Values; ++j) {
                const int value = int((codes >> (j * 8)) & 255);
                w[j] = scale * float(signs & (1u << j) ? -value : value);
            }
        } else {
            const float scale = weight_scale<Format>(block);
            #pragma unroll
            for (unsigned j = 0; j < Values; ++j) w[j] = float(integer_weight<Format>(block, local + j, grid)) * scale;
        }
        #pragma unroll
        for (unsigned r = 0; r < Routes; ++r) if (r < count) {
            if ((reinterpret_cast<uintptr_t>(x[r]) & 15) == 0) {
                #pragma unroll
                for (unsigned j = 0; j < Values; j += 4) {
                    const float4 v = reinterpret_cast<const float4*>(x[r] + k + j)[0];
                    accum[r][0] = fmaf(v.x, w[j], accum[r][0]);
                    accum[r][1] = fmaf(v.y, w[j + 1], accum[r][1]);
                    accum[r][2] = fmaf(v.z, w[j + 2], accum[r][2]);
                    accum[r][3] = fmaf(v.w, w[j + 3], accum[r][3]);
                }
            } else {
                #pragma unroll
                for (unsigned j = 0; j < Values; ++j) accum[r][j % 4] = fmaf(x[r][k + j], w[j], accum[r][j % 4]);
            }
        }
    }
    float sum[Routes];
    #pragma unroll
    for (unsigned r = 0; r < Routes; ++r) sum[r] = (accum[r][0] + accum[r][1]) + (accum[r][2] + accum[r][3]);

    #pragma unroll
    for (unsigned r = 0; r < Routes; ++r) if (r < count) {
        #pragma unroll
        for (unsigned step = 16; step; step >>= 1) sum[r] += __shfl_xor_sync(0xffffffff, sum[r], step);
        if (!lane) {
            if constexpr (Routed) outputs[sorted[first + r]][row] = sum[r];
            else dense_output[size_t(first + r) * out + row] = sum[r];
        }
    }
}

template<bool Routed, bool Bf16>
__global__ void prepare_inputs(const float* dense, const float* const* inputs,
                               uint16_t* prepared, float* powers, unsigned cols, bool relu_squared) {
    const unsigned row = blockIdx.x;
    const float* input = Routed ? inputs[row] : dense + size_t(row) * cols;
    if constexpr (Bf16) {
        for (unsigned k = threadIdx.x; k < cols; k += 256) {
            float value = input[k];
            if (relu_squared) { value = fmaxf(value, 0.f); value *= value; }
            prepared[size_t(row) * cols + k] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
        }
        return;
    }
    __shared__ float maxima[8];
    float maximum = 0;
    for (unsigned k = threadIdx.x; k < cols; k += 256) maximum = fmaxf(maximum, fabsf(input[k]));
    for (unsigned step = 16; step; step >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
    if (!(threadIdx.x & 31)) maxima[threadIdx.x / 32] = maximum;
    __syncthreads();
    maximum = 0;
    #pragma unroll
    for (unsigned warp = 0; warp < 8; ++warp) maximum = fmaxf(maximum, maxima[warp]);
    int exponent = 0;
    frexpf(maximum, &exponent);
    const float scale = maximum > 0 ? ldexpf(1.f, max(-126, exponent - 6)) : 1.f;
    if (!threadIdx.x) powers[row] = scale;
    for (unsigned k = threadIdx.x; k < cols; k += 256) {
        const float value = input[k] / scale;
        const __half high = __float2half_rn(value);
        const __half low = __float2half_rn(value - __half2float(high));
        const float tail = (value - __half2float(high)) - __half2float(low);
        prepared[(size_t(row) * 3) * cols + k] = __half_as_ushort(high);
        prepared[(size_t(row) * 3 + 1) * cols + k] = __half_as_ushort(low);
        prepared[(size_t(row) * 3 + 2) * cols + k] = __half_as_ushort(__float2half_rn(tail * 65536.f));
    }
}

template<bool Bf16>
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2,
                                   unsigned a3, unsigned b0, unsigned b1) {
    if constexpr (Bf16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Build route tiles once per projection, rather than scanning all experts in
// every matrix CTA. One warp scans 32 groups of up to 32 experts.
template<unsigned TileRows>
__global__ void prepare_tiles(const uint32_t* offsets, uint32_t* tiles,
                              unsigned experts, unsigned capacity) {
    __shared__ unsigned counts[1024], group_totals[32];
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const unsigned first = threadIdx.x < experts ? offsets[threadIdx.x] : 0;
    const unsigned last = threadIdx.x < experts ? offsets[threadIdx.x + 1] : 0;
    const unsigned count = (last - first + TileRows - 1) / TileRows;
    unsigned prefix = count;
    for (unsigned delta = 1; delta < 32; delta <<= 1) {
        const unsigned value = __shfl_up_sync(0xffffffff, prefix, delta);
        if (lane >= delta) prefix += value;
    }
    counts[threadIdx.x] = prefix - count;
    if (lane == 31) group_totals[warp] = prefix;
    __syncthreads();
    if (!warp) {
        unsigned total = group_totals[lane];
        for (unsigned delta = 1; delta < 32; delta <<= 1) {
            const unsigned value = __shfl_up_sync(0xffffffff, total, delta);
            if (lane >= delta) total += value;
        }
        group_totals[lane] = total;
    }
    __syncthreads();
    const unsigned start = counts[threadIdx.x] + (warp ? group_totals[warp - 1] : 0);
    for (unsigned t = 0; t < count; ++t) {
        const unsigned begin = first + t * TileRows;
        tiles[(start + t) * 3] = threadIdx.x;
        tiles[(start + t) * 3 + 1] = begin;
        tiles[(start + t) * 3 + 2] = min(begin + TileRows, last);
    }
    for (unsigned t = group_totals[31] + threadIdx.x; t < capacity; t += 1024)
        tiles[t * 3] = experts;
}

template<unsigned K, unsigned TileRows, unsigned Parts>
__device__ __forceinline__ void stage_inputs(uint16_t* destination,
    const uint16_t* input, const unsigned* ids, unsigned valid_rows,
    unsigned cols, unsigned base) {
    for (unsigned i = threadIdx.x; i < Parts * TileRows * (K / 8); i += blockDim.x) {
        const unsigned part = i / (TileRows * (K / 8));
        const unsigned row = (i / (K / 8)) % TileRows, k = (i % (K / 8)) * 8;
        const auto* source = input + (size_t(ids[row]) * Parts + part) * cols + base + k;
        const unsigned address = __cvta_generic_to_shared(destination + (part * TileRows + row) * (K + 8) + k);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(address), "l"(source), "r"(row < valid_rows ? 16 : 0) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Each CTA projects 8 to 64 activation rows through 64 or 128 output channels.
// Short dense batches split the reduction across one CTA per quantization block.
// Integer weights stay packed; the FP16 block scale is applied in F32 after the dot.
template<unsigned Format, bool Routed, unsigned Columns = 2, bool SplitK = false, bool Bf16 = false>
__global__ void packed_gemm(
    const uint32_t* sorted, const uint32_t* tiles, const uint8_t* const* weights,
    float* const* outputs, const uint8_t* dense_weight, float* dense_output,
    const uint16_t* input, const float* powers, const uint64_t* grid,
    unsigned rows, unsigned experts, unsigned out, unsigned cols) {
    constexpr unsigned K = block_elements<Format>, Step = K < 64 ? K : 64, Stride = Step + 8;
    constexpr unsigned TileOut = Bf16 && !SplitK ? 128 : 64;
    constexpr unsigned TileRows = Columns * 8, Parts = Bf16 ? 1 : 3;
    __shared__ unsigned expert, begin, end, ids[TileRows];
    __shared__ __align__(16) uint16_t w[TileOut][block_bytes<Format> / 2 + 1];
    __shared__ __align__(16) uint16_t x[2][Parts][TileRows][Stride];
    if (!threadIdx.x) {
        if constexpr (!Routed) { expert = 0; begin = blockIdx.y * TileRows; end = min(begin + TileRows, rows); }
        else { expert = tiles[blockIdx.y * 3]; begin = tiles[blockIdx.y * 3 + 1]; end = tiles[blockIdx.y * 3 + 2]; }
    }
    __syncthreads();
    if (expert == experts) return;
    if (threadIdx.x < TileRows) ids[threadIdx.x] = begin + threadIdx.x < end
        ? (Routed ? sorted[begin + threadIdx.x] : begin + threadIdx.x) : 0;
    __syncthreads();
    const uint8_t* weight = Routed ? weights[expert] : dense_weight;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const unsigned row0 = warp * 16 + lane / 4, row1 = row0 + 8;
    const unsigned pair = (lane & 3) * 2;
    float result[Columns][4] = {};
    for (unsigned base = SplitK ? blockIdx.z * K : 0;
         base < (SplitK ? (blockIdx.z + 1) * K : cols); base += K) {
        constexpr unsigned Words = block_bytes<Format> / 2;
        for (unsigned i = threadIdx.x; i < TileOut * Words; i += blockDim.x) {
            const unsigned row = i / Words, word = i % Words;
            const unsigned global_row = blockIdx.x * TileOut + row;
            const auto* block = reinterpret_cast<const uint16_t*>(weight
                + (size_t(global_row) * (cols / K) + base / K) * block_bytes<Format>);
            w[row][word] = global_row < out ? block[word] : 0;
        }
        stage_inputs<Step, TileRows, Parts>(&x[0][0][0][0], input, ids, end - begin, cols, base);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        float high[Columns][4][4] = {}, low[Columns][4] = {}, tail[Columns][4] = {};
        for (unsigned chunk = 0, stage = 0; chunk < K; chunk += Step, stage ^= 1) {
            if (chunk + Step < K) stage_inputs<Step, TileRows, Parts>(&x[stage ^ 1][0][0][0],
                input, ids, end - begin, cols, base + chunk + Step);
            #pragma unroll
            for (unsigned k = 0; k < Step; k += 16) {
                const auto* w0 = reinterpret_cast<const uint8_t*>(&w[row0][0]);
                const auto* w1 = reinterpret_cast<const uint8_t*>(&w[row1][0]);
                const unsigned a0 = tensor_pair<Format, Bf16>(w0, chunk + k + pair, grid);
                const unsigned a1 = tensor_pair<Format, Bf16>(w1, chunk + k + pair, grid);
                const unsigned a2 = tensor_pair<Format, Bf16>(w0, chunk + k + pair + 8, grid);
                const unsigned a3 = tensor_pair<Format, Bf16>(w1, chunk + k + pair + 8, grid);
                #pragma unroll
                for (unsigned n = 0; n < Columns; ++n) {
                    const auto* hi = reinterpret_cast<const uint32_t*>(&x[stage][0][lane / 4 + n * 8][k + pair]);
                    if constexpr (Bf16) {
                        mma<true>(result[n], a0, a1, a2, a3, hi[0], hi[4]);
                    } else {
                        const auto* lo = reinterpret_cast<const uint32_t*>(&x[stage][1][lane / 4 + n * 8][k + pair]);
                        mma<false>(high[n][(k / 16) % 4], a0, a1, a2, a3, hi[0], hi[4]);
                        mma<false>(low[n], a0, a1, a2, a3, lo[0], lo[4]);
                        const auto* last = reinterpret_cast<const uint32_t*>(&x[stage][2][lane / 4 + n * 8][k + pair]);
                        mma<false>(tail[n], a0, a1, a2, a3, last[0], last[4]);
                    }
                }
            }
            if (chunk + Step < K) {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
                __syncthreads();
            }
        }
        #pragma unroll
        for (unsigned n = 0; n < Columns; ++n) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                if constexpr (Bf16) continue;
                const float hi = (high[n][0][i] + high[n][1][i]) + (high[n][2][i] + high[n][3][i]);
                const float correction = fmaf(tail[n][i], 0x1p-16f, low[n][i]);
                result[n][i] = fmaf(hi + correction, weight_scale<Format>(reinterpret_cast<const uint8_t*>(&w[i < 2 ? row0 : row1][0])), result[n][i]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned n = 0; n < Columns; ++n) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned local = n * 8 + (lane & 3) * 2 + (i & 1);
            const unsigned row = blockIdx.x * TileOut + (i < 2 ? row0 : row1);
            if (begin + local < end && row < out) {
                const unsigned id = ids[local];
                const float value = Bf16 ? result[n][i] : result[n][i] * powers[id];
                if constexpr (Routed) outputs[id][row] = value;
                else dense_output[(size_t(SplitK ? blockIdx.z : 0) * rows + id) * out + row] = value;
            }
        }
    }
}

__global__ void sum_split_blocks(const float* partials, float* output, unsigned count, unsigned blocks) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = 0;
    for (unsigned block = 0; block < blocks; ++block) value += partials[size_t(block) * count + i];
    output[i] = value;
}

template<unsigned Format, bool Routed, bool Bf16>
cudaError_t project(
    const float* dense_input, const uint8_t* dense_weight, float* dense_output,
    const uint32_t* sorted, const uint32_t* sorted_experts, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights, float* const* outputs,
    const uint64_t* grid, uint16_t* prepared, float* powers, uint32_t* tile_map, float* partials,
    unsigned rows, unsigned experts, unsigned out, unsigned cols, bool relu_squared, cudaStream_t stream) {
    if (prepared) {
        prepare_inputs<Routed, Bf16><<<rows, 256, 0, stream>>>(dense_input, inputs, prepared, powers, cols, relu_squared);
        auto status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        if constexpr (Format == 1 && !Routed) {
            if (partials) {
                packed_gemm<Format, false, 1, true, Bf16><<<dim3((out + 63) / 64, (rows + 7) / 8, cols / 256), 128, 0, stream>>>(
                    nullptr, nullptr, nullptr, nullptr, dense_weight, partials,
                    prepared, powers, grid, rows, 1, out, cols);
                status = cudaGetLastError();
                if (status != cudaSuccess) return status;
                sum_split_blocks<<<(rows * out + 255) / 256, 256, 0, stream>>>(partials, dense_output, rows * out, cols / 256);
                return cudaGetLastError();
            }
        }
        constexpr unsigned Columns = Bf16 && Routed ? 8 : (Bf16 || (Format == 1 && !Routed) ? 4 : 2);
        constexpr unsigned TileRows = Columns * 8;
        const unsigned tiles = (rows + TileRows - 1) / TileRows + (Routed ? experts - 1 : 0);
        if constexpr (Routed) {
            prepare_tiles<TileRows><<<1, 1024, 0, stream>>>(offsets, tile_map, experts, tiles);
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
        }
        constexpr unsigned TileOut = Bf16 ? 128 : 64;
        packed_gemm<Format, Routed, Columns, false, Bf16><<<dim3((out + TileOut - 1) / TileOut, tiles), TileOut * 2, 0, stream>>>(
            sorted, tile_map, weights, outputs, dense_weight, dense_output,
            prepared, powers, grid, rows, experts, out, cols);
    } else {
        if constexpr (Routed) {
            if (rows <= 128) {
                packed_matvec<Format, Routed, 1><<<dim3((out + 7) / 8, rows), 256, 0, stream>>>(
                    dense_input, dense_weight, dense_output, sorted, sorted_experts, offsets,
                    inputs, weights, outputs, grid, rows, out, cols);
                return cudaGetLastError();
            }
        }
        packed_matvec<Format, Routed><<<dim3((out + 7) / 8, Routed ? rows : (rows + 3) / 4), 256, 0, stream>>>(
            dense_input, dense_weight, dense_output, sorted, sorted_experts, offsets,
            inputs, weights, outputs, grid, rows, out, cols);
    }
    return cudaGetLastError();
}
template<unsigned Format>
__global__ void expand_bf16(const uint8_t* packed, const uint64_t* grid, uint16_t* output, unsigned elements) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const uint8_t* block = packed + size_t(i / block_elements<Format>) * block_bytes<Format>;
    const float value = float(integer_weight<Format>(block, i % block_elements<Format>, grid)) * weight_scale<Format>(block);
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
} // namespace

extern "C" cudaError_t infer_ggml_dense_on_stream(
    const float* input, const uint8_t* weight, float* output, const uint64_t* grid,
    uint16_t* prepared, float* powers, uint32_t* tiles, float* partials, unsigned format, unsigned bf16, unsigned rows,
    unsigned out, unsigned cols, cudaStream_t stream) {
    if (format == 0 && bf16) return project<0, false, true>(input, weight, output, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, grid, prepared, powers, tiles, partials, rows, 1, out, cols, false, stream);
    if (format == 0) return project<0, false, false>(input, weight, output, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, grid, prepared, powers, tiles, partials, rows, 1, out, cols, false, stream);
    if (format == 1 && bf16) return project<1, false, true>(input, weight, output, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, grid, prepared, powers, tiles, partials, rows, 1, out, cols, false, stream);
    if (format == 1) return project<1, false, false>(input, weight, output, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, grid, prepared, powers, tiles, partials, rows, 1, out, cols, false, stream);
    return cudaErrorInvalidValue;
}
extern "C" cudaError_t infer_ggml_routed_on_stream(
    const uint32_t* sorted, const uint32_t* sorted_experts, const uint32_t* offsets,
    const float* const* inputs, const uint8_t* const* weights, float* const* outputs,
    const uint64_t* grid, uint16_t* prepared, float* powers, uint32_t* tiles, unsigned format, unsigned bf16, unsigned relu_squared,
    unsigned rows, unsigned experts, unsigned out, unsigned cols, cudaStream_t stream) {
    if (format == 0 && bf16) return project<0, true, true>(nullptr, nullptr, nullptr, sorted,
        sorted_experts, offsets, inputs, weights, outputs, grid, prepared, powers, tiles, nullptr, rows, experts, out, cols, relu_squared, stream);
    if (format == 0) return project<0, true, false>(nullptr, nullptr, nullptr, sorted,
        sorted_experts, offsets, inputs, weights, outputs, grid, prepared, powers, tiles, nullptr, rows, experts, out, cols, relu_squared, stream);
    if (format == 1 && bf16) return project<1, true, true>(nullptr, nullptr, nullptr, sorted,
        sorted_experts, offsets, inputs, weights, outputs, grid, prepared, powers, tiles, nullptr, rows, experts, out, cols, relu_squared, stream);
    if (format == 1) return project<1, true, false>(nullptr, nullptr, nullptr, sorted,
        sorted_experts, offsets, inputs, weights, outputs, grid, prepared, powers, tiles, nullptr, rows, experts, out, cols, relu_squared, stream);
    return cudaErrorInvalidValue;
}

extern "C" cudaError_t infer_ggml_expand_bf16_on_stream(
    const uint8_t* packed, const uint64_t* grid, uint16_t* output,
    unsigned elements, unsigned format, cudaStream_t stream) {
    if (format == 0) expand_bf16<0><<<(elements + 255) / 256, 256, 0, stream>>>(packed, grid, output, elements);
    else if (format == 1) expand_bf16<1><<<(elements + 255) / 256, 256, 0, stream>>>(packed, grid, output, elements);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}
