#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kChunk = 64;
constexpr int kTile = 16;
constexpr int kOutputSharedBytes =
    (32 * kDim + kChunk * kDim + kChunk * kDim + kDim * kDim) * sizeof(std::uint16_t);

using Bf16 = __nv_bfloat16;

__device__ __forceinline__ Bf16 load_bf16(const std::uint16_t* values, std::size_t index) {
    return reinterpret_cast<const Bf16*>(values)[index];
}

__device__ __forceinline__ void store_bf16(std::uint16_t* values,
                                           std::size_t index,
                                           float value) {
    reinterpret_cast<Bf16*>(values)[index] = __float2bfloat16_rn(value);
}

__device__ __forceinline__ void store_bf16_pair(std::uint16_t* values,
                                                std::size_t index,
                                                float first,
                                                float second) {
    reinterpret_cast<__nv_bfloat162*>(values + index)[0] =
        __floats2bfloat162_rn(first, second);
}

__device__ __forceinline__ float2 load_bf16_pair(const std::uint16_t* values,
                                                 std::size_t index) {
    return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(values + index)[0]);
}

__device__ __forceinline__ std::size_t vector_index(int token, int head, int feature) {
    return (static_cast<std::size_t>(token) * kHeads + head) * kDim + feature;
}

__device__ __forceinline__ std::size_t scalar_index(int token, int head) {
    return static_cast<std::size_t>(token) * kHeads + head;
}

__device__ __forceinline__ std::size_t triangle_index(int token, int head, int col) {
    return (static_cast<std::size_t>(token) * kHeads + head) * kChunk + col;
}

__device__ __forceinline__ void accumulator_coordinate(int item, int& row, int& col) {
    const int lane = threadIdx.x % 32;
    row = lane / 4 + ((item & 2) != 0 ? 8 : 0);
    col = (lane % 4) * 2 + (item & 1) + ((item & 4) != 0 ? 8 : 0);
}

__device__ __forceinline__ void chunk_bounds(int chunk,
                                             const std::int32_t* cu_seqlens,
                                             const std::int32_t* chunk_indices,
                                             int total_tokens,
                                             int& sequence,
                                             int& start,
                                             int& length) {
    sequence = chunk_indices[chunk * 2];
    const int local_chunk = chunk_indices[chunk * 2 + 1];
    const int sequence_start = cu_seqlens[sequence];
    const int sequence_end = min(cu_seqlens[sequence + 1], total_tokens);
    start = sequence_start + local_chunk * kChunk;
    length = max(0, min(kChunk, sequence_end - start));
}

__global__ void qwen36_gdn_cumsum_kernel(const std::uint16_t* gate,
                                         float* gate_cumsum,
                                         const std::int32_t* cu_seqlens,
                                         const std::int32_t* chunk_indices,
                                         int total_tokens) {
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    int sequence;
    int start;
    int length;
    chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens, sequence, start, length);
    const int token = threadIdx.x;
    const int lane = token % 32;
    const int warp = token / 32;
    float sum = token < length
        ? __bfloat162float(load_bf16(gate, scalar_index(start + token, head)))
        : 0.0f;
    for (int offset = 1; offset < 32; offset *= 2) {
        const float previous = __shfl_up_sync(0xffffffff, sum, offset);
        if (lane >= offset) sum += previous;
    }
    __shared__ float warp_totals[2];
    if (lane == 31) warp_totals[warp] = sum;
    __syncthreads();
    if (warp == 1) sum += warp_totals[0];
    if (token < length) {
        gate_cumsum[scalar_index(start + token, head)] = sum;
    }
}

__global__ void qwen36_gdn_kkt_kernel(const std::uint16_t* key,
                                      const std::uint16_t* beta,
                                      const float* gate_cumsum,
                                      float* a,
                                      const std::int32_t* cu_seqlens,
                                      const std::int32_t* chunk_indices,
                                      int total_tokens) {
    __shared__ Bf16 shared_key[kChunk * kDim];
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    int sequence;
    int start;
    int length;
    chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens, sequence, start, length);

    for (int index = threadIdx.x; index < kChunk * kDim; index += blockDim.x) {
        const int token = index / kDim;
        const int feature = index % kDim;
        shared_key[index] = token < length
            ? load_bf16(key, vector_index(start + token, head, feature))
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int tile_row = warp / 4;
    const int tile_col = warp % 4;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16, wmma::row_major> lhs;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::col_major> rhs;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int feature = 0; feature < kDim; feature += kTile) {
        wmma::load_matrix_sync(lhs, shared_key + tile_row * kTile * kDim + feature, kDim);
        wmma::load_matrix_sync(rhs, shared_key + tile_col * kTile * kDim + feature, kDim);
        wmma::mma_sync(accumulator, lhs, rhs, accumulator);
    }
    for (int item = 0; item < accumulator.num_elements; ++item) {
        int local_row;
        int local_col;
        accumulator_coordinate(item, local_row, local_col);
        const int row = tile_row * kTile + local_row;
        const int col = tile_col * kTile + local_col;
        if (row >= length) continue;
        float value = 0.0f;
        if (row == col) {
            value = __bfloat162float(load_bf16(beta, scalar_index(start + row, head)));
        } else if (col < row) {
            const float row_beta =
                __bfloat162float(load_bf16(beta, scalar_index(start + row, head)));
            const float decay = expf(
                gate_cumsum[scalar_index(start + row, head)] -
                gate_cumsum[scalar_index(start + col, head)]);
            value = row_beta * decay * accumulator.x[item];
        }
        a[triangle_index(start + row, head, col)] = value;
    }
}

__global__ void qwen36_gdn_solve_kernel(float* a,
                                        std::uint16_t* a_inverse,
                                        const std::int32_t* cu_seqlens,
                                        const std::int32_t* chunk_indices,
                                        int total_tokens) {
    __shared__ float transform[kChunk * kChunk];
    __shared__ float lower_row[kChunk];
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    int sequence;
    int start;
    int length;
    chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens, sequence, start, length);
    if (length < kChunk) {
        const int col = threadIdx.x;
        for (int row = 0; row < length; ++row) {
            if (col < kChunk) {
                lower_row[col] = a[triangle_index(start + row, head, col)];
            }
            __syncthreads();
            if (col < kChunk) {
                if (col <= row) {
                    float value = col == row ? lower_row[row] : 0.0f;
                    for (int inner = col; inner < row; ++inner) {
                        value -= lower_row[inner] * transform[inner * kChunk + col];
                    }
                    transform[row * kChunk + col] = value;
                } else {
                    transform[row * kChunk + col] = 0.0f;
                }
            }
            __syncthreads();
        }
        if (col < kChunk) {
            for (int row = 0; row < length; ++row) {
                store_bf16(
                    a_inverse,
                    triangle_index(start + row, head, col),
                    transform[row * kChunk + col]);
            }
        }
        return;
    }

    __shared__ float diagonal[64];
    __shared__ float product[16 * 16];
    const int thread = threadIdx.x;
    if (thread < kChunk) {
        diagonal[thread] = a[triangle_index(start + thread, head, thread)];
    }
    __syncthreads();

    const int warp = thread / 32;
    const int lane = thread % 32;
    if (warp < 4 && lane < 16) {
        const int block_start = warp * 16;
        for (int local_row = 0; local_row < 16; ++local_row) {
            const int row = block_start + local_row;
            const int col = block_start + lane;
            float inverse = lane == local_row ? 1.0f : 0.0f;
            if (lane < local_row) {
                for (int inner = lane; inner < local_row; ++inner) {
                    inverse -=
                        a[triangle_index(start + row, head, block_start + inner)] *
                        a[triangle_index(start + block_start + inner, head, col)];
                }
            }
            __syncwarp(0x0000ffff);
            a[triangle_index(start + row, head, col)] =
                lane <= local_row ? inverse : 0.0f;
            __syncwarp(0x0000ffff);
        }
    }
    __syncthreads();

    for (int index = thread; index < kChunk * kChunk; index += blockDim.x) {
        const int row = index / kChunk;
        const int col = index % kChunk;
        transform[index] = row / 16 == col / 16
            ? a[triangle_index(start + row, head, col)] * diagonal[col]
            : 0.0f;
    }
    __syncthreads();

    for (int block_row = 1; block_row < 4; ++block_row) {
        for (int block_col = 0; block_col < block_row; ++block_col) {
            const int local_row = thread / 16;
            const int local_col = thread % 16;
            const int row = block_row * 16 + local_row;
            const int col = block_col * 16 + local_col;
            float sum = 0.0f;
            for (int middle_block = block_col; middle_block < block_row; ++middle_block) {
                for (int inner = 0; inner < 16; ++inner) {
                    sum +=
                        a[triangle_index(
                            start + row,
                            head,
                            middle_block * 16 + inner)] *
                        transform[(middle_block * 16 + inner) * kChunk + col];
                }
            }
            product[thread] = sum;
            __syncthreads();
            float solved = 0.0f;
            for (int inner = 0; inner < 16; ++inner) {
                solved -=
                    a[triangle_index(
                        start + block_row * 16 + local_row,
                        head,
                        block_row * 16 + inner)] *
                    product[inner * 16 + local_col];
            }
            transform[row * kChunk + col] = solved;
            __syncthreads();
        }
    }

    for (int index = thread; index < kChunk * kChunk; index += blockDim.x) {
        const int row = index / kChunk;
        const int col = index % kChunk;
        store_bf16(
            a_inverse,
            triangle_index(start + row, head, col),
            transform[index]);
    }
}

__global__ void qwen36_gdn_wu_kernel(const std::uint16_t* key,
                                     const std::uint16_t* value,
                                     const std::uint16_t* a_inverse,
                                     const float* gate_cumsum,
                                     std::uint16_t* w,
                                     std::uint16_t* u,
                                     const std::int32_t* cu_seqlens,
                                     const std::int32_t* chunk_indices,
                                     int total_tokens) {
    __shared__ Bf16 scaled_key[kChunk * kDim];
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    int sequence;
    int start;
    int length;
    chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens, sequence, start, length);
    for (int index = threadIdx.x; index < kChunk * kDim; index += blockDim.x) {
        const int token = index / kDim;
        const int feature = index % kDim;
        scaled_key[index] = token < length
            ? __float2bfloat16_rn(
                  expf(gate_cumsum[scalar_index(start + token, head)]) *
                  __bfloat162float(load_bf16(key, vector_index(start + token, head, feature))))
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    if (length < kChunk) {
        for (int index = threadIdx.x; index < length * kDim; index += blockDim.x) {
            const int token = index / kDim;
            const int feature = index % kDim;
            float transformed_key = 0.0f;
            float transformed_value = 0.0f;
            for (int source = 0; source < length; ++source) {
                const float transform = __bfloat162float(load_bf16(
                    a_inverse,
                    triangle_index(start + token, head, source)));
                transformed_key += transform *
                    __bfloat162float(scaled_key[source * kDim + feature]);
                transformed_value += transform *
                    __bfloat162float(load_bf16(
                        value,
                        vector_index(start + source, head, feature)));
            }
            store_bf16(w, vector_index(start + token, head, feature), transformed_key);
            store_bf16(u, vector_index(start + token, head, feature), transformed_value);
        }
        return;
    }

    const int warp = threadIdx.x / 32;
    const int tile_col = warp % 8;
    for (int tile_row = warp / 8; tile_row < 4; tile_row += 2) {
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16, wmma::row_major>
            transform_fragment;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::row_major>
            input_fragment;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> w_accumulator;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> u_accumulator;

        wmma::fill_fragment(w_accumulator, 0.0f);
        wmma::fill_fragment(u_accumulator, 0.0f);
        for (int source = 0; source < kChunk; source += kTile) {
            wmma::load_matrix_sync(
                transform_fragment,
                reinterpret_cast<const Bf16*>(a_inverse) +
                    triangle_index(start + tile_row * kTile, head, source),
                kHeads * kChunk);
            wmma::load_matrix_sync(
                input_fragment,
                scaled_key + source * kDim + tile_col * kTile,
                kDim);
            wmma::mma_sync(
                w_accumulator, transform_fragment, input_fragment, w_accumulator);
            wmma::load_matrix_sync(
                input_fragment,
                reinterpret_cast<const Bf16*>(value) +
                    vector_index(start + source, head, tile_col * kTile),
                kHeads * kDim);
            wmma::mma_sync(
                u_accumulator, transform_fragment, input_fragment, u_accumulator);
        }
        for (int item = 0; item < w_accumulator.num_elements; item += 2) {
            int local_row;
            int local_col;
            accumulator_coordinate(item, local_row, local_col);
            const int row = tile_row * kTile + local_row;
            const int feature = tile_col * kTile + local_col;
            if (row < length) {
                store_bf16_pair(
                    w,
                    vector_index(start + row, head, feature),
                    w_accumulator.x[item],
                    w_accumulator.x[item + 1]);
            }
        }
        for (int item = 0; item < u_accumulator.num_elements; item += 2) {
            int local_row;
            int local_col;
            accumulator_coordinate(item, local_row, local_col);
            const int row = tile_row * kTile + local_row;
            const int feature = tile_col * kTile + local_col;
            if (row < length) {
                store_bf16_pair(
                    u,
                    vector_index(start + row, head, feature),
                    u_accumulator.x[item],
                    u_accumulator.x[item + 1]);
            }
        }
    }
}

__global__ void qwen36_gdn_h_kernel(const std::uint16_t* key,
                                    const std::uint16_t* u,
                                    const std::uint16_t* w,
                                    std::uint16_t* value_new,
                                    const float* gate_cumsum,
                                    std::uint16_t* h,
                                    float* state,
                                    const std::int32_t* cu_seqlens,
                                    const std::int64_t* chunk_offsets,
                                    int total_tokens) {
    constexpr int kValuePartition = 32;
    constexpr int kVectorElements = 8;
    constexpr int kMatrixElements = kChunk * kDim;
    constexpr int kPartitionStateElements = kValuePartition * kDim;
    constexpr int kPartitionValueElements = kChunk * kValuePartition;

    __shared__ Bf16 shared_matrix[kMatrixElements];
    __shared__ Bf16 shared_state[kPartitionStateElements];
    __shared__ Bf16 shared_value[kPartitionValueElements];
    __shared__ float shared_decay[kChunk];

    const int value_partition = blockIdx.x;
    const int sequence = blockIdx.y;
    const int head = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int first_value = value_partition * kValuePartition;
    const int sequence_start = cu_seqlens[sequence];
    const int sequence_end = min(cu_seqlens[sequence + 1], total_tokens);
    const int first_chunk = static_cast<int>(chunk_offsets[sequence]);
    const int end_chunk = static_cast<int>(chunk_offsets[sequence + 1]);
    float* head_state = state +
        (static_cast<std::size_t>(sequence) * kHeads + head) * kDim * kDim;

    for (int chunk = first_chunk; chunk < end_chunk; ++chunk) {
        const int start = sequence_start + (chunk - first_chunk) * kChunk;
        const int length = max(0, min(kChunk, sequence_end - start));
        std::uint16_t* chunk_h = h +
            (static_cast<std::size_t>(chunk) * kHeads + head) * kDim * kDim;

        for (int index = threadIdx.x;
             index < kPartitionStateElements;
             index += blockDim.x) {
            const int local_value = index / kDim;
            const int key_feature = index % kDim;
            const int state_index = (first_value + local_value) * kDim + key_feature;
            const Bf16 packed = __float2bfloat16_rn(head_state[state_index]);
            shared_state[index] = packed;
            reinterpret_cast<Bf16*>(chunk_h)[state_index] = packed;
        }
        for (int segment = threadIdx.x;
             segment < kMatrixElements / kVectorElements;
             segment += blockDim.x) {
            const int element = segment * kVectorElements;
            const int token = element / kDim;
            const int feature = element % kDim;
            uint4 packed = {};
            if (token < length) {
                packed = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const Bf16*>(w) +
                    vector_index(start + token, head, feature))[0];
            }
            reinterpret_cast<uint4*>(shared_matrix + element)[0] = packed;
        }
        for (int segment = threadIdx.x;
             segment < kPartitionValueElements / kVectorElements;
             segment += blockDim.x) {
            const int element = segment * kVectorElements;
            const int token = element / kValuePartition;
            const int local_value = element % kValuePartition;
            uint4 packed = {};
            if (token < length) {
                packed = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const Bf16*>(u) +
                    vector_index(start + token, head, first_value + local_value))[0];
            }
            reinterpret_cast<uint4*>(shared_value + element)[0] = packed;
        }
        __syncthreads();

        const int token_tile = warp;
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16, wmma::row_major>
            w_fragment;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::col_major>
            state_fragment;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
            correction_accumulators[2];
#pragma unroll
        for (int accumulator = 0; accumulator < 2; ++accumulator) {
            wmma::fill_fragment(correction_accumulators[accumulator], 0.0f);
        }
        for (int key_feature = 0; key_feature < kDim; key_feature += kTile) {
            wmma::load_matrix_sync(
                w_fragment,
                shared_matrix + token_tile * kTile * kDim + key_feature,
                kDim);
#pragma unroll
            for (int accumulator = 0; accumulator < 2; ++accumulator) {
                wmma::load_matrix_sync(
                    state_fragment,
                    shared_state + accumulator * kTile * kDim + key_feature,
                    kDim);
                wmma::mma_sync(
                    correction_accumulators[accumulator],
                    w_fragment,
                    state_fragment,
                    correction_accumulators[accumulator]);
            }
        }
#pragma unroll
        for (int accumulator = 0; accumulator < 2; ++accumulator) {
            for (int item = 0; item < correction_accumulators[accumulator].num_elements;
                 item += 2) {
                int local_row;
                int local_col;
                accumulator_coordinate(item, local_row, local_col);
                const int token = token_tile * kTile + local_row;
                const int local_value = accumulator * kTile + local_col;
                const int shared_index = token * kValuePartition + local_value;
                const float2 transformed = load_bf16_pair(
                    reinterpret_cast<const std::uint16_t*>(shared_value),
                    shared_index);
                const float first =
                    transformed.x - correction_accumulators[accumulator].x[item];
                const float second =
                    transformed.y - correction_accumulators[accumulator].x[item + 1];
                store_bf16_pair(
                    reinterpret_cast<std::uint16_t*>(shared_value),
                    shared_index,
                    first,
                    second);
                if (token < length) {
                    store_bf16_pair(
                        value_new,
                        vector_index(start + token, head, first_value + local_value),
                        first,
                        second);
                }
            }
        }
        __syncthreads();

        const float chunk_gate = gate_cumsum[scalar_index(start + length - 1, head)];
        if (threadIdx.x < kChunk) {
            shared_decay[threadIdx.x] = threadIdx.x < length
                ? expf(chunk_gate -
                       gate_cumsum[scalar_index(start + threadIdx.x, head)])
                : 0.0f;
        }
        __syncthreads();
        for (int pair = threadIdx.x; pair < kMatrixElements / 2; pair += blockDim.x) {
            const int element = pair * 2;
            const int token = element / kDim;
            const int feature = element % kDim;
            float2 values = {};
            if (token < length) {
                values = load_bf16_pair(
                    key,
                    vector_index(start + token, head, feature));
            }
            const float decay = shared_decay[token];
            reinterpret_cast<__nv_bfloat162*>(shared_matrix + element)[0] =
                __floats2bfloat162_rn(decay * values.x, decay * values.y);
        }
        __syncthreads();

        const int state_tile_row = warp / 2;
        const int first_state_tile_col = (warp % 2) * 4;
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16, wmma::col_major>
            value_fragment;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::row_major>
            key_fragment;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
            state_accumulators[4];
        const float chunk_decay = expf(chunk_gate);
#pragma unroll
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            const int tile_col = first_state_tile_col + accumulator;
            wmma::load_matrix_sync(
                state_accumulators[accumulator],
                head_state +
                    (first_value + state_tile_row * kTile) * kDim +
                    tile_col * kTile,
                kDim,
                wmma::mem_row_major);
            for (int item = 0; item < state_accumulators[accumulator].num_elements;
                 ++item) {
                state_accumulators[accumulator].x[item] *= chunk_decay;
            }
        }
        for (int token = 0; token < kChunk; token += kTile) {
            wmma::load_matrix_sync(
                value_fragment,
                shared_value + token * kValuePartition + state_tile_row * kTile,
                kValuePartition);
#pragma unroll
            for (int accumulator = 0; accumulator < 4; ++accumulator) {
                const int tile_col = first_state_tile_col + accumulator;
                wmma::load_matrix_sync(
                    key_fragment,
                    shared_matrix + token * kDim + tile_col * kTile,
                    kDim);
                wmma::mma_sync(
                    state_accumulators[accumulator],
                    value_fragment,
                    key_fragment,
                    state_accumulators[accumulator]);
            }
        }
#pragma unroll
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            const int tile_col = first_state_tile_col + accumulator;
            for (int item = 0; item < state_accumulators[accumulator].num_elements;
                 item += 2) {
                int local_row;
                int local_col;
                accumulator_coordinate(item, local_row, local_col);
                reinterpret_cast<float2*>(
                    head_state +
                    (first_value + state_tile_row * kTile + local_row) * kDim +
                    tile_col * kTile + local_col)[0] =
                    make_float2(
                        state_accumulators[accumulator].x[item],
                        state_accumulators[accumulator].x[item + 1]);
            }
        }
        __syncthreads();
    }
}
__global__ void qwen36_gdn_output_kernel(const std::uint16_t* query,
                                         const std::uint16_t* key,
                                         const std::uint16_t* value_new,
                                         const std::uint16_t* h,
                                         const float* gate_cumsum,
                                         std::uint16_t* output,
                                         const std::int32_t* cu_seqlens,
                                         const std::int32_t* chunk_indices,
                                         int total_tokens,
                                         float scale) {
    constexpr int kPartitionTokens = 32;
    constexpr int kVectorElements = 8;
    constexpr int kQueryElements = kPartitionTokens * kDim;
    constexpr int kKeyElements = kChunk * kDim;
    constexpr int kValueElements = kChunk * kDim;
    constexpr int kStateElements = kDim * kDim;

    extern __shared__ Bf16 shared[];
    Bf16* shared_query = shared;
    Bf16* shared_key = shared_query + kQueryElements;
    Bf16* shared_value = shared_key + kKeyElements;
    Bf16* shared_state = shared_value + kValueElements;

    const int token_partition = blockIdx.x;
    const int first_token = token_partition * kPartitionTokens;
    const int chunk = blockIdx.y;
    const int head = blockIdx.z;
    const int warp = threadIdx.x / 32;
    int sequence;
    int start;
    int length;
    chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens, sequence, start, length);
    const Bf16* chunk_h = reinterpret_cast<const Bf16*>(h) +
        (static_cast<std::size_t>(chunk) * kHeads + head) * kStateElements;

    for (int segment = threadIdx.x;
         segment < kQueryElements / kVectorElements;
         segment += blockDim.x) {
        const int element = segment * kVectorElements;
        const int local_token = element / kDim;
        const int feature = element % kDim;
        const int token = first_token + local_token;
        uint4 packed = {};
        if (token < length) {
            packed = reinterpret_cast<const uint4*>(
                reinterpret_cast<const Bf16*>(query) +
                vector_index(start + token, head, feature))[0];
        }
        reinterpret_cast<uint4*>(shared_query + element)[0] = packed;
    }
    for (int segment = threadIdx.x;
         segment < kKeyElements / kVectorElements;
         segment += blockDim.x) {
        const int element = segment * kVectorElements;
        const int token = element / kDim;
        const int feature = element % kDim;
        uint4 packed_key = {};
        uint4 packed_value = {};
        if (token < length) {
            packed_key = reinterpret_cast<const uint4*>(
                reinterpret_cast<const Bf16*>(key) +
                vector_index(start + token, head, feature))[0];
            packed_value = reinterpret_cast<const uint4*>(
                reinterpret_cast<const Bf16*>(value_new) +
                vector_index(start + token, head, feature))[0];
        }
        reinterpret_cast<uint4*>(shared_key + element)[0] = packed_key;
        reinterpret_cast<uint4*>(shared_value + element)[0] = packed_value;
    }
    for (int segment = threadIdx.x;
         segment < kStateElements / kVectorElements;
         segment += blockDim.x) {
        const int element = segment * kVectorElements;
        reinterpret_cast<uint4*>(shared_state + element)[0] =
            reinterpret_cast<const uint4*>(chunk_h + element)[0];
    }
    __syncthreads();

    const int attention_tile_row = warp / 2;
    const int first_attention_tile_col = (warp % 2) * 2;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16, wmma::row_major>
        query_fragment;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::col_major>
        key_fragment;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        attention_accumulators[2];
#pragma unroll
    for (int accumulator = 0; accumulator < 2; ++accumulator) {
        wmma::fill_fragment(attention_accumulators[accumulator], 0.0f);
    }
    for (int feature = 0; feature < kDim; feature += kTile) {
        wmma::load_matrix_sync(
            query_fragment,
            shared_query + attention_tile_row * kTile * kDim + feature,
            kDim);
#pragma unroll
        for (int accumulator = 0; accumulator < 2; ++accumulator) {
            const int tile_col = first_attention_tile_col + accumulator;
            wmma::load_matrix_sync(
                key_fragment,
                shared_key + tile_col * kTile * kDim + feature,
                kDim);
            wmma::mma_sync(
                attention_accumulators[accumulator],
                query_fragment,
                key_fragment,
                attention_accumulators[accumulator]);
        }
    }

    std::uint16_t* shared_attention = reinterpret_cast<std::uint16_t*>(shared_key);
#pragma unroll
    for (int accumulator = 0; accumulator < 2; ++accumulator) {
        const int tile_col = first_attention_tile_col + accumulator;
        for (int item = 0; item < attention_accumulators[accumulator].num_elements;
             item += 2) {
            int local_row;
            int local_col;
            accumulator_coordinate(item, local_row, local_col);
            const int row = first_token + attention_tile_row * kTile + local_row;
            const int col = tile_col * kTile + local_col;
            float first_attention = 0.0f;
            float second_attention = 0.0f;
            if (row < length && col <= row && col < length) {
                first_attention = attention_accumulators[accumulator].x[item] *
                    expf(gate_cumsum[scalar_index(start + row, head)] -
                         gate_cumsum[scalar_index(start + col, head)]);
            }
            if (row < length && col + 1 <= row && col + 1 < length) {
                second_attention = attention_accumulators[accumulator].x[item + 1] *
                    expf(gate_cumsum[scalar_index(start + row, head)] -
                         gate_cumsum[scalar_index(start + col + 1, head)]);
            }
            store_bf16_pair(
                shared_attention,
                (attention_tile_row * kTile + local_row) * kChunk + col,
                first_attention,
                second_attention);
        }
    }
    __syncthreads();

    float* query_decay = reinterpret_cast<float*>(shared_key + kPartitionTokens * kChunk);
    if (threadIdx.x < kPartitionTokens) {
        const int token = first_token + threadIdx.x;
        query_decay[threadIdx.x] = token < length
            ? expf(gate_cumsum[scalar_index(start + token, head)])
            : 0.0f;
    }
    __syncthreads();
    for (int segment = threadIdx.x;
         segment < kQueryElements / kVectorElements;
         segment += blockDim.x) {
        const int element = segment * kVectorElements;
        const int local_token = element / kDim;
        const float decay = query_decay[local_token];
#pragma unroll
        for (int offset = 0; offset < kVectorElements; offset += 2) {
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(
                    shared_query + element + offset)[0]);
            reinterpret_cast<__nv_bfloat162*>(
                shared_query + element + offset)[0] =
                __floats2bfloat162_rn(decay * values.x, decay * values.y);
        }
    }
    __syncthreads();

    const int output_tile_row = warp / 2;
    const int first_output_tile_col = (warp % 2) * 4;
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16, wmma::row_major>
        lhs_fragment;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::row_major>
        value_fragment;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16, wmma::col_major>
        state_fragment;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        output_accumulators[4];
#pragma unroll
    for (int accumulator = 0; accumulator < 4; ++accumulator) {
        wmma::fill_fragment(output_accumulators[accumulator], 0.0f);
    }
    for (int feature = 0; feature < kDim; feature += kTile) {
        wmma::load_matrix_sync(
            lhs_fragment,
            shared_query + output_tile_row * kTile * kDim + feature,
            kDim);
#pragma unroll
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            const int tile_col = first_output_tile_col + accumulator;
            wmma::load_matrix_sync(
                state_fragment,
                shared_state + tile_col * kTile * kDim + feature,
                kDim);
            wmma::mma_sync(
                output_accumulators[accumulator],
                lhs_fragment,
                state_fragment,
                output_accumulators[accumulator]);
        }
    }
    for (int source = 0; source < kChunk; source += kTile) {
        wmma::load_matrix_sync(
            lhs_fragment,
            reinterpret_cast<const Bf16*>(shared_attention) +
                output_tile_row * kTile * kChunk + source,
            kChunk);
#pragma unroll
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            const int tile_col = first_output_tile_col + accumulator;
            wmma::load_matrix_sync(
                value_fragment,
                shared_value + source * kDim + tile_col * kTile,
                kDim);
            wmma::mma_sync(
                output_accumulators[accumulator],
                lhs_fragment,
                value_fragment,
                output_accumulators[accumulator]);
        }
    }

#pragma unroll
    for (int accumulator = 0; accumulator < 4; ++accumulator) {
        const int tile_col = first_output_tile_col + accumulator;
        for (int item = 0; item < output_accumulators[accumulator].num_elements;
             item += 2) {
            int local_row;
            int local_col;
            accumulator_coordinate(item, local_row, local_col);
            const int row = first_token + output_tile_row * kTile + local_row;
            const int feature = tile_col * kTile + local_col;
            if (row < length) {
                store_bf16_pair(
                    output,
                    vector_index(start + row, head, feature),
                    scale * output_accumulators[accumulator].x[item],
                    scale * output_accumulators[accumulator].x[item + 1]);
            }
        }
    }
}

cudaError_t validate_common(const void* first,
                            const void* second,
                            const std::int32_t* cu_seqlens,
                            const std::int32_t* chunk_indices,
                            std::uint32_t total_tokens,
                            std::uint32_t chunk_count) {
    if (first == nullptr || second == nullptr || cu_seqlens == nullptr ||
        chunk_indices == nullptr || total_tokens == 0 || chunk_count == 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t infer_qwen36_gdn_chunk_cumsum_on_stream(
    const std::uint16_t* gate,
    float* gate_cumsum,
    const std::int32_t* cu_seqlens,
    const std::int32_t* chunk_indices,
    std::uint32_t total_tokens,
    std::uint32_t chunk_count,
    cudaStream_t stream) {
    const cudaError_t valid = validate_common(
        gate, gate_cumsum, cu_seqlens, chunk_indices, total_tokens, chunk_count);
    if (valid != cudaSuccess) return valid;
    qwen36_gdn_cumsum_kernel<<<dim3(chunk_count, kHeads), 64, 0, stream>>>(
        gate, gate_cumsum, cu_seqlens, chunk_indices, static_cast<int>(total_tokens));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen36_gdn_chunk_kkt_on_stream(
    const std::uint16_t* key,
    const std::uint16_t* beta,
    const float* gate_cumsum,
    float* a,
    const std::int32_t* cu_seqlens,
    const std::int32_t* chunk_indices,
    std::uint32_t total_tokens,
    std::uint32_t chunk_count,
    cudaStream_t stream) {
    const cudaError_t valid = validate_common(
        key, a, cu_seqlens, chunk_indices, total_tokens, chunk_count);
    if (valid != cudaSuccess || beta == nullptr || gate_cumsum == nullptr) {
        return cudaErrorInvalidValue;
    }
    qwen36_gdn_kkt_kernel<<<dim3(chunk_count, kHeads), 512, 0, stream>>>(
        key, beta, gate_cumsum, a, cu_seqlens, chunk_indices, static_cast<int>(total_tokens));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen36_gdn_chunk_solve_on_stream(
    float* a,
    std::uint16_t* a_inverse,
    const std::int32_t* cu_seqlens,
    const std::int32_t* chunk_indices,
    std::uint32_t total_tokens,
    std::uint32_t chunk_count,
    cudaStream_t stream) {
    const cudaError_t valid = validate_common(
        a, a_inverse, cu_seqlens, chunk_indices, total_tokens, chunk_count);
    if (valid != cudaSuccess) return valid;
    qwen36_gdn_solve_kernel<<<dim3(chunk_count, kHeads), 256, 0, stream>>>(
        a, a_inverse, cu_seqlens, chunk_indices, static_cast<int>(total_tokens));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen36_gdn_chunk_wu_on_stream(
    const std::uint16_t* key,
    const std::uint16_t* value,
    const std::uint16_t* a_inverse,
    const float* gate_cumsum,
    std::uint16_t* w,
    std::uint16_t* u,
    const std::int32_t* cu_seqlens,
    const std::int32_t* chunk_indices,
    std::uint32_t total_tokens,
    std::uint32_t chunk_count,
    cudaStream_t stream) {
    const cudaError_t valid = validate_common(
        key, w, cu_seqlens, chunk_indices, total_tokens, chunk_count);
    if (valid != cudaSuccess || value == nullptr || a_inverse == nullptr ||
        gate_cumsum == nullptr || u == nullptr) {
        return cudaErrorInvalidValue;
    }
    qwen36_gdn_wu_kernel<<<dim3(chunk_count, kHeads), 512, 0, stream>>>(
        key,
        value,
        a_inverse,
        gate_cumsum,
        w,
        u,
        cu_seqlens,
        chunk_indices,
        static_cast<int>(total_tokens));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen36_gdn_chunk_h_on_stream(
    const std::uint16_t* key,
    const std::uint16_t* u,
    const std::uint16_t* w,
    std::uint16_t* value_new,
    const float* gate_cumsum,
    std::uint16_t* h,
    float* state,
    const std::int32_t* cu_seqlens,
    const std::int64_t* chunk_offsets,
    std::uint32_t sequence_count,
    std::uint32_t total_tokens,
    cudaStream_t stream) {
    if (key == nullptr || u == nullptr || w == nullptr || value_new == nullptr ||
        gate_cumsum == nullptr || h == nullptr || state == nullptr || cu_seqlens == nullptr ||
        chunk_offsets == nullptr || sequence_count == 0 || total_tokens == 0) {
        return cudaErrorInvalidValue;
    }
    qwen36_gdn_h_kernel<<<dim3(4, sequence_count, kHeads), 128, 0, stream>>>(
        key,
        u,
        w,
        value_new,
        gate_cumsum,
        h,
        state,
        cu_seqlens,
        chunk_offsets,
        static_cast<int>(total_tokens));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen36_gdn_chunk_output_on_stream(
    const std::uint16_t* query,
    const std::uint16_t* key,
    const std::uint16_t* value_new,
    const std::uint16_t* h,
    const float* gate_cumsum,
    std::uint16_t* output,
    const std::int32_t* cu_seqlens,
    const std::int32_t* chunk_indices,
    std::uint32_t total_tokens,
    std::uint32_t chunk_count,
    float scale,
    cudaStream_t stream) {
    const cudaError_t valid = validate_common(
        query, output, cu_seqlens, chunk_indices, total_tokens, chunk_count);
    if (valid != cudaSuccess || key == nullptr || value_new == nullptr || h == nullptr ||
        gate_cumsum == nullptr) {
        return cudaErrorInvalidValue;
    }
    const cudaError_t attribute = cudaFuncSetAttribute(
        qwen36_gdn_output_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kOutputSharedBytes);
    if (attribute != cudaSuccess) return attribute;
    qwen36_gdn_output_kernel<<<
        dim3(2, chunk_count, kHeads), 128, kOutputSharedBytes, stream>>>(
        query,
        key,
        value_new,
        h,
        gate_cumsum,
        output,
        cu_seqlens,
        chunk_indices,
        static_cast<int>(total_tokens),
        scale);
    return cudaGetLastError();
}
