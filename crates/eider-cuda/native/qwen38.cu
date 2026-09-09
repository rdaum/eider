#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>

#include <cstddef>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    return value;
}

__device__ __forceinline__ float qwen38_rope_value(
    const float* values,
    std::uint32_t dim,
    std::uint32_t rotary_dim,
    std::uint32_t position,
    float theta) {
    if (dim >= rotary_dim) {
        return values[dim];
    }
    const std::uint32_t half = rotary_dim / 2;
    const std::uint32_t pair = dim % half;
    const float frequency = powf(
        theta, -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim));
    float sine;
    float cosine;
    sincosf(static_cast<float>(position) * frequency, &sine, &cosine);
    const float first = values[pair];
    const float second = values[pair + half];
    return dim < half ? first * cosine - second * sine
                      : second * cosine + first * sine;
}

__global__ void qwen38_qsa_prepare_query_kernel(
    const float* projection,
    const float* q_norm,
    float* query,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t input_row_offset,
    std::uint32_t start_position,
    float eps,
    float theta) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = blockIdx.y;
    const std::uint32_t dim = threadIdx.x;
    if (head >= heads || dim >= head_dim) {
        return;
    }
    const std::size_t projection_stride = static_cast<std::size_t>(heads + 1) * head_dim;
    const float* input = projection +
        static_cast<std::size_t>(input_row_offset + row) * projection_stride +
        static_cast<std::size_t>(head) * head_dim;
    query += static_cast<std::size_t>(row) * heads * head_dim;
    extern __shared__ float scratch[];
    const float value = input[dim];
    scratch[dim] = value * value;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            scratch[dim] += scratch[dim + stride];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
    scratch[dim] = value * inverse_rms * q_norm[dim];
    __syncthreads();
    query[static_cast<std::size_t>(head) * head_dim + dim] = qwen38_rope_value(
        scratch, dim, rotary_dim, start_position + row, theta);
}

__global__ void qwen38_qsa_append_keys_kernel(
    const float* projection,
    __nv_bfloat16* key_pool,
    std::uint32_t input_row_offset,
    std::uint32_t slot,
    std::uint32_t page_offset,
    std::uint32_t page_tokens,
    std::uint32_t heads,
    std::uint32_t head_dim) {
    const std::uint32_t dim = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = blockIdx.y;
    if (dim >= head_dim) {
        return;
    }
    const std::size_t projection_stride =
        static_cast<std::size_t>(heads + 1) * head_dim;
    const float* key = projection +
        static_cast<std::size_t>(input_row_offset + row) * projection_stride +
        static_cast<std::size_t>(heads) * head_dim;
    const std::size_t destination =
        (static_cast<std::size_t>(slot) * page_tokens + page_offset + row) * head_dim + dim;
    key_pool[destination] = __float2bfloat16_rn(key[dim]);
}

__global__ void qwen38_qsa_score_blocks_kernel(
    const float* query,
    const __nv_bfloat16* key_pool,
    const std::uint32_t* page_table,
    const float* k_norm,
    float* scores,
    std::uint32_t page_tokens,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rotary_dim,
    std::uint32_t start_cache_len,
    std::uint32_t rows,
    std::uint32_t max_blocks,
    float eps,
    float theta) {
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t dim = threadIdx.x;
    if (block >= (start_cache_len + rows - 1) / 4 || dim >= head_dim) {
        return;
    }
    const std::uint32_t token = block * 4;
    const std::uint32_t page_slot = page_table[token / page_tokens];
    const std::uint32_t page_offset = token % page_tokens;
    const std::size_t page_base =
        static_cast<std::size_t>(page_slot) * page_tokens * head_dim;
    float pooled = 0.0f;
    for (std::uint32_t row = 0; row < 4; ++row) {
        pooled += __bfloat162float(key_pool[
            page_base + static_cast<std::size_t>(page_offset + row) * head_dim + dim]);
    }
    pooled = __bfloat162float(__float2bfloat16_rn(pooled * 0.25f));
    extern __shared__ float scratch[];
    scratch[dim] = pooled;
    scratch[head_dim + dim] = pooled * pooled;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            scratch[head_dim + dim] += scratch[head_dim + dim + stride];
        }
        __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(scratch[head_dim] / static_cast<float>(head_dim) + eps);
    scratch[dim] = pooled * inverse_rms * k_norm[dim];
    __syncthreads();
    const float key = qwen38_rope_value(scratch, dim, rotary_dim, token, theta);
    for (std::uint32_t row_index = 0; row_index < rows; ++row_index) {
        if (block >= (start_cache_len + row_index) / 4) {
            continue;
        }
        const float* row_query =
            query + static_cast<std::size_t>(row_index) * heads * head_dim;
        float score = 0.0f;
        for (std::uint32_t head = 0; head < heads; ++head) {
            float dot = row_query[static_cast<std::size_t>(head) * head_dim + dim] * key;
            dot = block_sum(dot);
            if (threadIdx.x == 0) {
                score += fmaxf(dot, 0.0f);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scores[static_cast<std::size_t>(row_index) * max_blocks + block] =
                score * rsqrtf(static_cast<float>(head_dim));
        }
    }
}

template <int ItemsPerThread>
__global__ void qwen38_qsa_sort_blocks_kernel(
    const float* scores,
    std::uint8_t* selected_blocks,
    std::uint32_t start_cache_len,
    std::uint32_t max_blocks,
    std::uint32_t selected_budget_blocks) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t cache_len = start_cache_len + row;
    const std::uint32_t complete_blocks = cache_len / 4;
    const std::uint32_t selected_complete_blocks = min(complete_blocks, selected_budget_blocks);
    const std::uint32_t tail_tokens = cache_len % 4;
    scores += static_cast<std::size_t>(row) * max_blocks;
    selected_blocks += static_cast<std::size_t>(row) * max_blocks;
    using BlockSort = cub::BlockRadixSort<unsigned long long, 256, ItemsPerThread>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    unsigned long long keys[ItemsPerThread];
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t block = threadIdx.x * ItemsPerThread + item;
        if (block < complete_blocks) {
            const float score = isfinite(scores[block])
                ? fmaxf(scores[block], 0.0f)
                : 0.0f;
            keys[item] =
                static_cast<unsigned long long>(__float_as_uint(score)) << 32 |
                static_cast<unsigned long long>(~block);
        } else {
            keys[item] = 0;
        }
    }
    BlockSort(sort_storage).SortDescending(keys);
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const std::uint32_t rank = threadIdx.x * ItemsPerThread + item;
        if (rank < selected_complete_blocks) {
            selected_blocks[~static_cast<std::uint32_t>(keys[item])] = 1;
        }
    }
    if (tail_tokens != 0 && threadIdx.x == 0) {
        selected_blocks[complete_blocks] = 1;
    }
}

__global__ void qwen38_qsa_select_blocks_kernel(
    const float* scores,
    std::uint8_t* selected_blocks,
    std::uint32_t start_cache_len,
    std::uint32_t max_blocks,
    std::uint32_t selected_budget_blocks) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t cache_len = start_cache_len + row;
    const std::uint32_t complete_blocks = cache_len / 4;
    const std::uint32_t selected_complete_blocks = min(complete_blocks, selected_budget_blocks);
    const std::uint32_t tail_tokens = cache_len % 4;
    scores += static_cast<std::size_t>(row) * max_blocks;
    selected_blocks += static_cast<std::size_t>(row) * max_blocks;
    if (complete_blocks <= selected_complete_blocks) {
        for (std::uint32_t block = threadIdx.x; block < complete_blocks;
             block += blockDim.x) {
            selected_blocks[block] = 1;
        }
        if (tail_tokens != 0 && threadIdx.x == 0) {
            selected_blocks[complete_blocks] = 1;
        }
        return;
    }

    __shared__ std::uint32_t histogram[256];
    __shared__ std::uint32_t prefix;
    __shared__ std::uint32_t rank;
    __shared__ std::uint32_t count;
    __shared__ std::uint32_t tie_cutoff;
    if (threadIdx.x == 0) {
        prefix = 0;
        rank = selected_complete_blocks;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0;
        __syncthreads();
        const std::uint32_t higher_mask = shift == 24
            ? 0u
            : ~((1u << (shift + 8)) - 1u);
        for (std::uint32_t block = threadIdx.x; block < complete_blocks;
             block += blockDim.x) {
            const float score = isfinite(scores[block]) ? fmaxf(scores[block], 0.0f) : 0.0f;
            const std::uint32_t bits = __float_as_uint(score);
            if ((bits & higher_mask) == prefix) {
                atomicAdd(&histogram[(bits >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t higher = 0;
            for (int byte = 255; byte >= 0; --byte) {
                const std::uint32_t next = higher + histogram[byte];
                if (rank <= next) {
                    prefix |= static_cast<std::uint32_t>(byte) << shift;
                    rank -= higher;
                    break;
                }
                higher = next;
            }
        }
        __syncthreads();
    }

    std::uint32_t local_greater = 0;
    for (std::uint32_t block = threadIdx.x; block < complete_blocks;
         block += blockDim.x) {
        const float score = isfinite(scores[block]) ? fmaxf(scores[block], 0.0f) : 0.0f;
        local_greater += __float_as_uint(score) > prefix;
    }
    local_greater = static_cast<std::uint32_t>(
        block_sum(static_cast<float>(local_greater)));
    if (threadIdx.x == 0) {
        count = selected_complete_blocks - local_greater;
        std::uint32_t seen = 0;
        tie_cutoff = 0;
        for (std::uint32_t block = 0; block < complete_blocks; ++block) {
            const float score =
                isfinite(scores[block]) ? fmaxf(scores[block], 0.0f) : 0.0f;
            if (__float_as_uint(score) == prefix && ++seen == count) {
                tie_cutoff = block;
                break;
            }
        }
    }
    __syncthreads();
    for (std::uint32_t block = threadIdx.x; block < complete_blocks;
         block += blockDim.x) {
        const float score = isfinite(scores[block]) ? fmaxf(scores[block], 0.0f) : 0.0f;
        const std::uint32_t bits = __float_as_uint(score);
        const bool selected = bits > prefix || (bits == prefix && block <= tie_cutoff);
        selected_blocks[block] = selected ? 1 : 0;
    }
    if (tail_tokens != 0 && threadIdx.x == 0) {
        selected_blocks[complete_blocks] = 1;
    }
}

__global__ void qwen38_qsa_build_tile_mask_kernel(
    const std::uint8_t* selected_blocks,
    std::uint8_t* selected_tiles,
    std::uint32_t* selected_block_indices,
    std::uint32_t* selected_token_tiles,
    std::uint32_t* selected_context_tiles,
    std::uint32_t* selected_counts,
    std::uint32_t start_cache_len,
    std::uint32_t max_blocks,
    std::uint32_t max_tiles,
    std::uint32_t index_capacity) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t cache_len = start_cache_len + row;
    const std::uint32_t visible_blocks = (cache_len + 3) / 4;
    selected_blocks += static_cast<std::size_t>(row) * max_blocks;
    selected_tiles += static_cast<std::size_t>(row) * max_tiles;
    selected_block_indices += static_cast<std::size_t>(row) * index_capacity;
    selected_token_tiles += static_cast<std::size_t>(row) * index_capacity;
    selected_context_tiles += static_cast<std::size_t>(row) * index_capacity;
    std::uint32_t block_count = 0;
    for (std::uint32_t base = 0; base < visible_blocks; base += 32) {
        const std::uint32_t block = base + lane;
        const bool selected = block < visible_blocks && selected_blocks[block] != 0;
        const std::uint32_t ballot = __ballot_sync(0xffffffffu, selected);
        const std::uint32_t prefix = __popc(ballot & ((1u << lane) - 1u));
        if (selected && block_count + prefix < index_capacity) {
            selected_block_indices[block_count + prefix] = block;
        }
        block_count += __popc(ballot);
    }
    const std::uint32_t visible_token_tiles = (visible_blocks + 1) / 2;
    std::uint32_t token_tile_count = 0;
    for (std::uint32_t base = 0; base < visible_token_tiles; base += 32) {
        const std::uint32_t token_tile = base + lane;
        const std::uint32_t first_block = token_tile * 2;
        const bool selected = token_tile < visible_token_tiles &&
            (selected_blocks[first_block] != 0 ||
             (first_block + 1 < visible_blocks && selected_blocks[first_block + 1] != 0));
        const std::uint32_t ballot = __ballot_sync(0xffffffffu, selected);
        const std::uint32_t prefix = __popc(ballot & ((1u << lane) - 1u));
        if (selected && token_tile_count + prefix < index_capacity) {
            selected_token_tiles[token_tile_count + prefix] = token_tile;
        }
        token_tile_count += __popc(ballot);
    }
    const std::uint32_t visible_context_tiles = (visible_blocks + 15) / 16;
    std::uint32_t context_tile_count = 0;
    for (std::uint32_t base = 0; base < visible_context_tiles; base += 32) {
        const std::uint32_t context_tile = base + lane;
        bool selected = false;
        if (context_tile < visible_context_tiles) {
            const std::uint32_t first_block = context_tile * 16;
            for (std::uint32_t block = first_block;
                 block < min(first_block + 16, visible_blocks); ++block) {
                selected |= selected_blocks[block] != 0;
            }
        }
        const std::uint32_t ballot = __ballot_sync(0xffffffffu, selected);
        const std::uint32_t prefix = __popc(ballot & ((1u << lane) - 1u));
        if (selected) {
            selected_tiles[context_tile] = 1;
            if (context_tile_count + prefix < index_capacity) {
                selected_context_tiles[context_tile_count + prefix] = context_tile;
            }
        }
        context_tile_count += __popc(ballot);
    }
    if (lane == 0) {
        selected_counts[row * 3] = min(block_count, index_capacity);
        selected_counts[row * 3 + 1] = min(token_tile_count, index_capacity);
        selected_counts[row * 3 + 2] = min(context_tile_count, index_capacity);
    }
}

__global__ void qwen38_hc_norm_kernel(const float* input,
                                      const float* delta_weight,
                                      float* output,
                                      std::uint32_t hidden,
                                      std::uint32_t hc_count,
                                      float eps) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t branch = group % hc_count;
    const std::size_t offset = static_cast<std::size_t>(group) * hidden;
    float square_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = input[offset + col];
        square_sum += value * value;
    }
    square_sum = block_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const std::size_t weight_offset = static_cast<std::size_t>(branch) * hidden;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        output[offset + col] = input[offset + col] * inverse_rms
            * (1.0f + delta_weight[weight_offset + col]);
    }
}

__global__ void qwen38_hc_silu_scale_kernel(float* values,
                                            std::size_t count,
                                            float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < count) {
        const float value = values[index] * scale;
        values[index] = value / (1.0f + expf(-value));
    }
}

__global__ void qwen38_hc_collapse_kernel(const float* normed,
                                          const float* gate_logits,
                                          float* output,
                                          std::uint32_t hidden,
                                          std::uint32_t hc_count,
                                          std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::size_t token = index / hidden;
    const std::size_t col = index % hidden;
    const std::size_t token_offset = token * hc_count * hidden;
    float sum = 0.0f;
    for (std::uint32_t branch = 0; branch < hc_count; ++branch) {
        const std::size_t offset = token_offset
            + static_cast<std::size_t>(branch) * hidden + col;
        const float gate = 1.0f / (1.0f + expf(-gate_logits[offset]));
        sum += gate * normed[offset];
    }
    output[index] = sum / static_cast<float>(hc_count);
}

__global__ void qwen38_hc_combine_kernel(const float* residual,
                                         const float* block_output,
                                         const float* inject_logits,
                                         float* output,
                                         std::uint32_t hidden,
                                         std::uint32_t hc_count,
                                         std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::size_t stream_width = static_cast<std::size_t>(hc_count) * hidden;
    const std::size_t token = index / stream_width;
    const std::size_t within_token = index % stream_width;
    const std::size_t branch = within_token / hidden;
    const std::size_t col = within_token % hidden;
    const float logit = inject_logits[token * hc_count + branch]
        / static_cast<float>(hc_count);
    const float injection = 2.0f / (1.0f + expf(-logit));
    output[index] = residual[index]
        + injection * block_output[token * hidden + col];
}

__global__ void qwen38_repeat_streams_kernel(const float* input,
                                             float* output,
                                             std::uint32_t hidden,
                                             std::uint32_t hc_count,
                                             std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < count) {
        const std::size_t token = index / (static_cast<std::size_t>(hc_count) * hidden);
        output[index] = input[token * hidden + index % hidden];
    }
}

__global__ void qwen38_ple_gate_value_kernel(const float* key,
                                             const float* query,
                                             const float* value,
                                             float* gated,
                                             std::uint32_t hidden,
                                             std::uint32_t hc_count) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t token = group / hc_count;
    const std::uint32_t branch = group % hc_count;
    const std::size_t stream_offset = static_cast<std::size_t>(group) * hidden;
    float dot = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        dot += key[stream_offset + col] * query[stream_offset + col];
    }
    dot = block_sum(dot);
    __shared__ float gate;
    if (threadIdx.x == 0) {
        const float scaled = dot * rsqrtf(static_cast<float>(hidden));
        const float signed_root = scaled > 0.0f
            ? sqrtf(fmaxf(scaled, 1e-6f))
            : (scaled < 0.0f ? -sqrtf(fmaxf(-scaled, 1e-6f)) : 0.0f);
        gate = 1.0f / (1.0f + expf(-signed_root));
    }
    __syncthreads();
    const std::size_t value_offset = static_cast<std::size_t>(token) * hidden;
    for (std::uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        gated[stream_offset + col] = gate * value[value_offset + col];
    }
}

__global__ void qwen38_ple_conv_update_kernel(const float* normalized,
                                              const float* gated,
                                              const __nv_bfloat16* weight,
                                              float* state,
                                              float* output,
                                              std::uint32_t tokens,
                                              std::uint32_t channels,
                                              std::uint32_t kernel,
                                              std::uint32_t dilation,
                                              std::uint32_t history) {
    const std::uint32_t channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    const std::size_t state_offset = static_cast<std::size_t>(channel) * history;
    const std::size_t weight_offset = static_cast<std::size_t>(channel) * kernel;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        float conv = 0.0f;
        for (std::uint32_t tap = 0; tap < kernel; ++tap) {
            const std::uint32_t lag = (kernel - 1 - tap) * dilation;
            const std::uint32_t centre = history + token;
            const std::uint32_t source = centre - lag;
            const float x = source < history
                ? state[state_offset + source]
                : normalized[static_cast<std::size_t>(source - history) * channels + channel];
            conv += x * __bfloat162float(weight[weight_offset + tap]);
        }
        const float activated = conv / (1.0f + expf(-conv));
        const std::size_t output_offset = static_cast<std::size_t>(token) * channels + channel;
        output[output_offset] = gated[output_offset] + activated;
    }
    for (std::uint32_t position = 0; position < history; ++position) {
        const std::uint32_t source = tokens + position;
        state[state_offset + position] = source < history
            ? state[state_offset + source]
            : normalized[static_cast<std::size_t>(source - history) * channels + channel];
    }
}

}  // namespace

extern "C" cudaError_t infer_qwen38_hc_norm_f32_on_stream(
        const float* input,
        const float* delta_weight,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        float eps,
        cudaStream_t stream) {
    if (input == nullptr || delta_weight == nullptr || output == nullptr
        || tokens == 0 || hidden == 0 || hc_count == 0 || eps <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    qwen38_hc_norm_kernel<<<tokens * hc_count, threads, 0, stream>>>(
        input, delta_weight, output, hidden, hc_count, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_hc_silu_scale_f32_on_stream(
        float* values,
        std::size_t count,
        float scale,
        cudaStream_t stream) {
    if (values == nullptr || count == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_hc_silu_scale_kernel<<<blocks, threads, 0, stream>>>(values, count, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_hc_collapse_f32_on_stream(
        const float* normed,
        const float* gate_logits,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (normed == nullptr || gate_logits == nullptr || output == nullptr
        || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = static_cast<std::size_t>(tokens) * hidden;
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_hc_collapse_kernel<<<blocks, threads, 0, stream>>>(
        normed, gate_logits, output, hidden, hc_count, count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_hc_combine_f32_on_stream(
        const float* residual,
        const float* block_output,
        const float* inject_logits,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (residual == nullptr || block_output == nullptr || inject_logits == nullptr
        || output == nullptr || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = static_cast<std::size_t>(tokens) * hc_count * hidden;
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_hc_combine_kernel<<<blocks, threads, 0, stream>>>(
        residual, block_output, inject_logits, output, hidden, hc_count, count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_repeat_streams_f32_on_stream(
        const float* input,
        float* output,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (input == nullptr || output == nullptr || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t count = static_cast<std::size_t>(tokens) * hidden * hc_count;
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    qwen38_repeat_streams_kernel<<<blocks, threads, 0, stream>>>(
        input, output, hidden, hc_count, count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_ple_gate_value_f32_on_stream(
        const float* key,
        const float* query,
        const float* value,
        float* gated,
        std::uint32_t tokens,
        std::uint32_t hidden,
        std::uint32_t hc_count,
        cudaStream_t stream) {
    if (key == nullptr || query == nullptr || value == nullptr || gated == nullptr
        || tokens == 0 || hidden == 0 || hc_count == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    qwen38_ple_gate_value_kernel<<<tokens * hc_count, threads, 0, stream>>>(
        key, query, value, gated, hidden, hc_count);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_ple_conv_update_f32_on_stream(
        const float* normalized,
        const float* gated,
        const std::uint16_t* weight_bf16,
        float* state,
        float* output,
        std::uint32_t tokens,
        std::uint32_t channels,
        std::uint32_t kernel,
        std::uint32_t dilation,
        cudaStream_t stream) {
    if (normalized == nullptr || gated == nullptr || weight_bf16 == nullptr
        || state == nullptr || output == nullptr || tokens == 0 || channels == 0
        || kernel < 2 || dilation == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t history = (kernel - 1) * dilation;
    constexpr int threads = 256;
    const unsigned int blocks = (channels + threads - 1) / threads;
    qwen38_ple_conv_update_kernel<<<blocks, threads, 0, stream>>>(
        normalized,
        gated,
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
        state,
        output,
        tokens,
        channels,
        kernel,
        dilation,
        history);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_qsa_prepare_and_select_on_stream(
        const float* projection,
        const float* q_norm,
        const float* k_norm,
        std::uint16_t* key_pool_bf16,
        const std::uint32_t* page_table,
        float* query,
        float* scores,
        std::uint8_t* selected_blocks,
        std::uint8_t* selected_tiles,
        std::uint32_t* selected_block_indices,
        std::uint32_t* selected_token_tiles,
        std::uint32_t* selected_context_tiles,
        std::uint32_t* selected_counts,
        std::uint32_t slot,
        std::uint32_t page_offset,
        std::uint32_t cache_len,
        std::uint32_t max_tokens,
        std::uint32_t page_tokens,
        std::uint32_t page_slots,
        std::uint32_t heads,
        std::uint32_t head_dim,
        std::uint32_t rotary_dim,
        std::uint32_t compress_ratio,
        std::uint32_t budget,
        std::uint32_t index_capacity,
        float eps,
        float theta,
        cudaStream_t stream) {
    if (projection == nullptr || q_norm == nullptr || k_norm == nullptr ||
        key_pool_bf16 == nullptr || page_table == nullptr || query == nullptr ||
        scores == nullptr || selected_blocks == nullptr || selected_tiles == nullptr ||
        selected_block_indices == nullptr || selected_token_tiles == nullptr ||
        selected_context_tiles == nullptr || selected_counts == nullptr ||
        cache_len == 0 || cache_len > max_tokens || page_tokens == 0 ||
        page_slots == 0 || slot >= page_slots || page_offset >= page_tokens ||
        heads == 0 || head_dim == 0 || head_dim > 1024 ||
        (head_dim & (head_dim - 1)) != 0 || rotary_dim == 0 ||
        rotary_dim > head_dim || (rotary_dim % 2) != 0 ||
        compress_ratio != 4 || budget == 0 || (budget % compress_ratio) != 0 ||
        index_capacity < budget / compress_ratio + 1 ||
        eps <= 0.0f || !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t complete_blocks = cache_len / compress_ratio;
    const std::uint32_t max_blocks = (max_tokens + compress_ratio - 1) / compress_ratio;
    const std::uint32_t max_tiles = (max_tokens + 63) / 64;
    cudaError_t status = cudaMemsetAsync(selected_blocks, 0, max_blocks, stream);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(selected_tiles, 0, max_tiles, stream);
    if (status != cudaSuccess) return status;

    qwen38_qsa_prepare_query_kernel<<<
        dim3(heads, 1), head_dim, head_dim * sizeof(float), stream>>>(
        projection, q_norm, query, heads, head_dim, rotary_dim,
        0, cache_len - 1, eps, theta);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    constexpr std::uint32_t kAppendThreads = 128;
    qwen38_qsa_append_keys_kernel<<<
        (head_dim + kAppendThreads - 1) / kAppendThreads, kAppendThreads, 0, stream>>>(
        projection, reinterpret_cast<__nv_bfloat16*>(key_pool_bf16), 0, slot,
        page_offset, page_tokens, heads, head_dim);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (complete_blocks != 0) {
        qwen38_qsa_score_blocks_kernel<<<
            dim3(complete_blocks, 1), head_dim, 2 * head_dim * sizeof(float), stream>>>(
            query, reinterpret_cast<const __nv_bfloat16*>(key_pool_bf16),
            page_table, k_norm, scores, page_tokens, heads,
            head_dim, rotary_dim, cache_len, 1, max_blocks, eps, theta);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    constexpr std::uint32_t kSelectThreads = 256;
    const std::uint32_t selected_complete_blocks =
        min(complete_blocks, budget / compress_ratio);
    if (complete_blocks <= selected_complete_blocks) {
        qwen38_qsa_select_blocks_kernel<<<1, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, cache_len, max_blocks, budget / compress_ratio);
    } else if (complete_blocks <= 2048) {
        qwen38_qsa_sort_blocks_kernel<8><<<1, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, cache_len, max_blocks, budget / compress_ratio);
    } else if (complete_blocks <= 4096) {
        qwen38_qsa_sort_blocks_kernel<16><<<1, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, cache_len, max_blocks, budget / compress_ratio);
    } else {
        qwen38_qsa_select_blocks_kernel<<<1, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, cache_len, max_blocks, budget / compress_ratio);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    qwen38_qsa_build_tile_mask_kernel<<<1, 32, 0, stream>>>(
        selected_blocks, selected_tiles, selected_block_indices, selected_token_tiles,
        selected_context_tiles, selected_counts, cache_len, max_blocks, max_tiles,
        index_capacity);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_qsa_select_appended_rows_on_stream(
        const float* projection,
        const float* q_norm,
        const float* k_norm,
        const std::uint16_t* key_pool_bf16,
        const std::uint32_t* page_table,
        float* queries,
        float* scores,
        std::uint8_t* selected_blocks,
        std::uint8_t* selected_tiles,
        std::uint32_t* selected_block_indices,
        std::uint32_t* selected_token_tiles,
        std::uint32_t* selected_context_tiles,
        std::uint32_t* selected_counts,
        std::uint32_t input_row_offset,
        std::uint32_t start_cache_len,
        std::uint32_t rows,
        std::uint32_t max_tokens,
        std::uint32_t page_tokens,
        std::uint32_t heads,
        std::uint32_t head_dim,
        std::uint32_t rotary_dim,
        std::uint32_t compress_ratio,
        std::uint32_t budget,
        std::uint32_t index_capacity,
        float eps,
        float theta,
        cudaStream_t stream) {
    if (projection == nullptr || q_norm == nullptr || k_norm == nullptr ||
        key_pool_bf16 == nullptr || page_table == nullptr || queries == nullptr ||
        scores == nullptr || selected_blocks == nullptr || selected_tiles == nullptr ||
        selected_block_indices == nullptr || selected_token_tiles == nullptr ||
        selected_context_tiles == nullptr || selected_counts == nullptr ||
        start_cache_len == 0 || rows == 0 || start_cache_len > max_tokens ||
        rows - 1 > max_tokens - start_cache_len || page_tokens == 0 || heads == 0 ||
        head_dim == 0 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0 ||
        rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 ||
        compress_ratio != 4 || budget == 0 || (budget % compress_ratio) != 0 ||
        index_capacity < budget / compress_ratio + 1 ||
        eps <= 0.0f || !isfinite(theta) || theta <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t final_cache_len = start_cache_len + rows - 1;
    const std::uint32_t complete_blocks = final_cache_len / compress_ratio;
    const std::uint32_t max_blocks = (max_tokens + compress_ratio - 1) / compress_ratio;
    const std::uint32_t max_tiles = (max_tokens + 63) / 64;
    cudaError_t status = cudaMemsetAsync(
        selected_blocks, 0, static_cast<std::size_t>(rows) * max_blocks, stream);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(
        selected_tiles, 0, static_cast<std::size_t>(rows) * max_tiles, stream);
    if (status != cudaSuccess) return status;

    qwen38_qsa_prepare_query_kernel<<<
        dim3(heads, rows), head_dim, head_dim * sizeof(float), stream>>>(
        projection, q_norm, queries, heads, head_dim, rotary_dim,
        input_row_offset, start_cache_len - 1, eps, theta);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (complete_blocks != 0) {
        qwen38_qsa_score_blocks_kernel<<<
            complete_blocks, head_dim, 2 * head_dim * sizeof(float), stream>>>(
            queries, reinterpret_cast<const __nv_bfloat16*>(key_pool_bf16),
            page_table, k_norm, scores, page_tokens, heads, head_dim, rotary_dim,
            start_cache_len, rows, max_blocks, eps, theta);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    constexpr std::uint32_t kSelectThreads = 256;
    const std::uint32_t selected_complete_blocks =
        min(complete_blocks, budget / compress_ratio);
    if (complete_blocks <= selected_complete_blocks) {
        qwen38_qsa_select_blocks_kernel<<<rows, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, start_cache_len, max_blocks, budget / compress_ratio);
    } else if (complete_blocks <= 2048) {
        qwen38_qsa_sort_blocks_kernel<8><<<rows, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, start_cache_len, max_blocks, budget / compress_ratio);
    } else if (complete_blocks <= 4096) {
        qwen38_qsa_sort_blocks_kernel<16><<<rows, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, start_cache_len, max_blocks, budget / compress_ratio);
    } else {
        qwen38_qsa_select_blocks_kernel<<<rows, kSelectThreads, 0, stream>>>(
            scores, selected_blocks, start_cache_len, max_blocks, budget / compress_ratio);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    qwen38_qsa_build_tile_mask_kernel<<<rows, 32, 0, stream>>>(
        selected_blocks, selected_tiles, selected_block_indices, selected_token_tiles,
        selected_context_tiles, selected_counts, start_cache_len, max_blocks, max_tiles,
        index_capacity);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_qwen38_qsa_append_keys_on_stream(
        const float* projection,
        std::uint16_t* key_pool_bf16,
        std::uint32_t slot,
        std::uint32_t page_offset,
        std::uint32_t page_tokens,
        std::uint32_t page_slots,
        std::uint32_t heads,
        std::uint32_t head_dim,
        std::uint32_t input_row_offset,
        std::uint32_t rows,
        cudaStream_t stream) {
    if (projection == nullptr || key_pool_bf16 == nullptr || page_tokens == 0 ||
        page_slots == 0 || slot >= page_slots || page_offset >= page_tokens ||
        heads == 0 || head_dim == 0 || rows == 0 || rows > page_tokens - page_offset) {
        return cudaErrorInvalidValue;
    }
    constexpr std::uint32_t kThreads = 128;
    const dim3 grid((head_dim + kThreads - 1) / kThreads, rows);
    qwen38_qsa_append_keys_kernel<<<grid, kThreads, 0, stream>>>(
        projection, reinterpret_cast<__nv_bfloat16*>(key_pool_bf16),
        input_row_offset, slot, page_offset, page_tokens, heads, head_dim);
    return cudaGetLastError();
}
