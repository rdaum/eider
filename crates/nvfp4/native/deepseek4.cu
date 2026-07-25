#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kScaleBlock = 128;
constexpr std::uint32_t kHyperStreams = 4;
constexpr std::uint32_t kHyperMix = 24;

__device__ __forceinline__ float e4m3_value(std::uint8_t code) {
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

__device__ __forceinline__ float e8m0_value(std::uint8_t code) {
    if (code == 0xff) {
        return __uint_as_float(0x7fffffffU);
    }
    return ldexpf(1.0f, static_cast<int>(code) - 127);
}

__device__ __forceinline__ float warp_sum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[8];
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
    return warp_sum(value);
}

__device__ __forceinline__ float warp_max(float value) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 1));
    return value;
}

__device__ __forceinline__ float block_max(float value) {
    __shared__ float warp_maxima[8];
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        warp_maxima[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? warp_maxima[lane] : -__int_as_float(0x7f800000);
    return warp_max(value);
}

__global__ void block_fp8_linear_f32_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ weight,
    const std::uint8_t* __restrict__ scales,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t batch = blockIdx.y;
    if (row >= rows || batch >= batch_rows) {
        return;
    }

    const std::uint32_t scale_cols = cols / kScaleBlock;
    const std::uint32_t scale_row = row / kScaleBlock;
    const std::size_t input_base = static_cast<std::size_t>(batch) * cols;
    const std::size_t weight_base = static_cast<std::size_t>(row) * cols;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::uint8_t scale_code =
            scales[static_cast<std::size_t>(scale_row) * scale_cols + col / kScaleBlock];
        sum += input[input_base + col] * e4m3_value(weight[weight_base + col])
            * e8m0_value(scale_code);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        output[static_cast<std::size_t>(batch) * rows + row] = sum;
    }
}

__global__ void block_fp8_grouped_linear_f32_kernel(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ weight,
    const std::uint8_t* __restrict__ scales,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t groups,
    std::uint32_t rows_per_group,
    std::uint32_t cols_per_group) {
    const std::uint32_t flat_row = blockIdx.x;
    const std::uint32_t batch = blockIdx.y;
    const std::uint32_t total_rows = groups * rows_per_group;
    if (flat_row >= total_rows || batch >= batch_rows) {
        return;
    }
    const std::uint32_t group = flat_row / rows_per_group;
    const std::uint32_t scale_cols = cols_per_group / kScaleBlock;
    const std::uint32_t scale_row = flat_row / kScaleBlock;
    const std::size_t input_base =
        (static_cast<std::size_t>(batch) * groups + group) * cols_per_group;
    const std::size_t weight_base =
        static_cast<std::size_t>(flat_row) * cols_per_group;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols_per_group; col += blockDim.x) {
        const std::uint8_t scale_code =
            scales[static_cast<std::size_t>(scale_row) * scale_cols
                + col / kScaleBlock];
        sum += input[input_base + col] * e4m3_value(weight[weight_base + col])
            * e8m0_value(scale_code);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        output[static_cast<std::size_t>(batch) * total_rows + flat_row] = sum;
    }
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__global__ void hyper_prepare_f32_kernel(
    const float* __restrict__ streams,
    const float* __restrict__ fn,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ post,
    float* __restrict__ comb,
    float* __restrict__ collapsed,
    std::uint32_t batch_rows,
    std::uint32_t hidden,
    float rms_eps,
    float hc_eps,
    std::uint32_t sinkhorn_iters) {
    const std::uint32_t batch = blockIdx.x;
    if (batch >= batch_rows) {
        return;
    }
    const std::size_t flat = static_cast<std::size_t>(kHyperStreams) * hidden;
    const float* row = streams + static_cast<std::size_t>(batch) * flat;
    __shared__ float inverse_rms;
    __shared__ float mixed[kHyperMix];
    __shared__ float pre[kHyperStreams];
    __shared__ float post_shared[kHyperStreams];
    __shared__ float comb_shared[kHyperStreams * kHyperStreams];

    float square_sum = 0.0f;
    for (std::size_t index = threadIdx.x; index < flat; index += blockDim.x) {
        const float value = row[index];
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_sum(square_sum);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(flat) + rms_eps);
    }
    __syncthreads();

    for (std::uint32_t output = 0; output < kHyperMix; ++output) {
        float sum = 0.0f;
        const float* weight = fn + static_cast<std::size_t>(output) * flat;
        for (std::size_t index = threadIdx.x; index < flat; index += blockDim.x) {
            sum = fmaf(row[index] * inverse_rms, weight[index], sum);
        }
        sum = block_sum(sum);
        if (threadIdx.x == 0) {
            mixed[output] = sum;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (std::uint32_t stream = 0; stream < kHyperStreams; ++stream) {
            pre[stream] = sigmoid(mixed[stream] * scale[0] + base[stream]) + hc_eps;
            post_shared[stream] =
                2.0f * sigmoid(mixed[kHyperStreams + stream] * scale[1]
                    + base[kHyperStreams + stream]);
        }
        for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
            float max_logit = -__int_as_float(0x7f800000);
            for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
                const std::uint32_t index = source * kHyperStreams + target;
                const float logit =
                    mixed[2 * kHyperStreams + index] * scale[2]
                    + base[2 * kHyperStreams + index];
                comb_shared[index] = logit;
                max_logit = fmaxf(max_logit, logit);
            }
            float sum = 0.0f;
            for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
                const std::uint32_t index = source * kHyperStreams + target;
                const float value = expf(comb_shared[index] - max_logit);
                comb_shared[index] = value;
                sum += value;
            }
            for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
                comb_shared[source * kHyperStreams + target] =
                    comb_shared[source * kHyperStreams + target] / sum + hc_eps;
            }
        }
        for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
            float sum = hc_eps;
            for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
                sum += comb_shared[source * kHyperStreams + target];
            }
            for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
                comb_shared[source * kHyperStreams + target] /= sum;
            }
        }
        for (std::uint32_t iteration = 1; iteration < sinkhorn_iters; ++iteration) {
            for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
                float sum = hc_eps;
                for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
                    sum += comb_shared[source * kHyperStreams + target];
                }
                for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
                    comb_shared[source * kHyperStreams + target] /= sum;
                }
            }
            for (std::uint32_t target = 0; target < kHyperStreams; ++target) {
                float sum = hc_eps;
                for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
                    sum += comb_shared[source * kHyperStreams + target];
                }
                for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
                    comb_shared[source * kHyperStreams + target] /= sum;
                }
            }
        }
        for (std::uint32_t stream = 0; stream < kHyperStreams; ++stream) {
            post[static_cast<std::size_t>(batch) * kHyperStreams + stream] =
                post_shared[stream];
        }
        for (std::uint32_t index = 0; index < kHyperStreams * kHyperStreams; ++index) {
            comb[static_cast<std::size_t>(batch) * kHyperStreams * kHyperStreams + index] =
                comb_shared[index];
        }
    }
    __syncthreads();

    for (std::uint32_t feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        float value = 0.0f;
        for (std::uint32_t stream = 0; stream < kHyperStreams; ++stream) {
            value = fmaf(pre[stream], row[static_cast<std::size_t>(stream) * hidden + feature], value);
        }
        collapsed[static_cast<std::size_t>(batch) * hidden + feature] = value;
    }
}

__global__ void hyper_apply_f32_kernel(
    const float* __restrict__ streams,
    const float* __restrict__ sublayer,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t hidden) {
    const std::uint32_t batch = blockIdx.y;
    const std::uint32_t target = blockIdx.x;
    if (batch >= batch_rows || target >= kHyperStreams) {
        return;
    }
    const std::size_t stream_base =
        static_cast<std::size_t>(batch) * kHyperStreams * hidden;
    const float post_value = post[static_cast<std::size_t>(batch) * kHyperStreams + target];
    const float* comb_row = comb + static_cast<std::size_t>(batch) * kHyperStreams * kHyperStreams;
    for (std::uint32_t feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        float value =
            post_value * sublayer[static_cast<std::size_t>(batch) * hidden + feature];
        for (std::uint32_t source = 0; source < kHyperStreams; ++source) {
            value = fmaf(
                comb_row[source * kHyperStreams + target],
                streams[stream_base + static_cast<std::size_t>(source) * hidden + feature],
                value);
        }
        output[stream_base + static_cast<std::size_t>(target) * hidden + feature] = value;
    }
}

__global__ void hyper_head_f32_kernel(
    const float* __restrict__ streams,
    const float* __restrict__ fn,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t hidden,
    float rms_eps,
    float hc_eps) {
    const std::uint32_t batch = blockIdx.x;
    if (batch >= batch_rows) {
        return;
    }
    const std::size_t flat = static_cast<std::size_t>(kHyperStreams) * hidden;
    const float* row = streams + static_cast<std::size_t>(batch) * flat;
    __shared__ float inverse_rms;
    __shared__ float weights[kHyperStreams];
    float square_sum = 0.0f;
    for (std::size_t index = threadIdx.x; index < flat; index += blockDim.x) {
        const float value = row[index];
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_sum(square_sum);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(flat) + rms_eps);
    }
    __syncthreads();
    for (std::uint32_t output_index = 0; output_index < kHyperStreams; ++output_index) {
        float sum = 0.0f;
        const float* weight = fn + static_cast<std::size_t>(output_index) * flat;
        for (std::size_t index = threadIdx.x; index < flat; index += blockDim.x) {
            sum = fmaf(row[index] * inverse_rms, weight[index], sum);
        }
        sum = block_sum(sum);
        if (threadIdx.x == 0) {
            weights[output_index] =
                sigmoid(sum * scale[0] + base[output_index]) + hc_eps;
        }
        __syncthreads();
    }
    for (std::uint32_t feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        float value = 0.0f;
        for (std::uint32_t stream_index = 0; stream_index < kHyperStreams;
             ++stream_index) {
            value = fmaf(
                weights[stream_index],
                row[static_cast<std::size_t>(stream_index) * hidden + feature],
                value);
        }
        output[static_cast<std::size_t>(batch) * hidden + feature] = value;
    }
}

__global__ void rope_interleaved_trailing_f32_kernel(
    float* __restrict__ values,
    const float* __restrict__ inv_freq,
    const std::uint32_t* __restrict__ positions,
    std::uint32_t batch_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rope_dim,
    float direction) {
    const std::size_t pair_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pairs_per_row =
        static_cast<std::size_t>(heads) * (rope_dim / 2);
    const std::size_t total_pairs = static_cast<std::size_t>(batch_rows) * pairs_per_row;
    if (pair_index >= total_pairs) {
        return;
    }
    const std::uint32_t batch = pair_index / pairs_per_row;
    const std::size_t within_row = pair_index % pairs_per_row;
    const std::uint32_t head = within_row / (rope_dim / 2);
    const std::uint32_t pair = within_row % (rope_dim / 2);
    const float angle =
        static_cast<float>(positions[batch]) * inv_freq[pair] * direction;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);
    const std::size_t offset =
        (static_cast<std::size_t>(batch) * heads + head) * head_dim
        + (head_dim - rope_dim) + 2 * pair;
    const float even = values[offset];
    const float odd = values[offset + 1];
    values[offset] = even * cosine - odd * sine;
    values[offset + 1] = odd * cosine + even * sine;
}

__device__ __forceinline__ const float* attention_entry(
    const float* sliding,
    std::uint32_t sliding_length,
    std::uint32_t sliding_start,
    std::uint32_t sliding_capacity,
    const float* compressed,
    std::uint32_t compressed_length,
    const std::int32_t* selected,
    std::uint32_t selected_count,
    std::uint32_t entry,
    std::uint32_t head_dim) {
    if (entry < sliding_length) {
        const std::uint32_t slot = (sliding_start + entry) % sliding_capacity;
        return sliding + static_cast<std::size_t>(slot) * head_dim;
    }
    const std::uint32_t compressed_entry = entry - sliding_length;
    const std::int32_t index =
        selected_count == 0 ? static_cast<std::int32_t>(compressed_entry)
                            : selected[compressed_entry];
    if (index < 0 || static_cast<std::uint32_t>(index) >= compressed_length) {
        return nullptr;
    }
    return compressed + static_cast<std::size_t>(index) * head_dim;
}

__global__ void attention_f32_kernel(
    const float* __restrict__ query,
    const float* const* __restrict__ sliding_tables,
    const std::uint32_t* __restrict__ sliding_lengths,
    const std::uint32_t* __restrict__ sliding_starts,
    const float* const* __restrict__ compressed_tables,
    const std::uint32_t* __restrict__ compressed_lengths,
    const std::int32_t* __restrict__ selected_indices,
    const float* __restrict__ sinks,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t sliding_capacity,
    std::uint32_t selected_count,
    float scaling) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t batch = blockIdx.y;
    if (head >= heads || batch >= batch_rows) {
        return;
    }
    const float* q =
        query + (static_cast<std::size_t>(batch) * heads + head) * head_dim;
    const float* sliding = sliding_tables[batch];
    const std::uint32_t sliding_length = sliding_lengths[batch];
    const float* compressed = compressed_tables[batch];
    const std::uint32_t compressed_length = compressed_lengths[batch];
    const std::int32_t* selected =
        selected_count == 0
            ? nullptr
            : selected_indices + static_cast<std::size_t>(batch) * selected_count;
    const std::uint32_t compressed_entries =
        selected_count == 0 ? compressed_length : selected_count;
    const std::uint32_t entries = sliding_length + compressed_entries;
    __shared__ float logit;
    __shared__ float maximum;
    __shared__ float denominator;

    if (threadIdx.x == 0) {
        maximum = sinks[head];
    }
    __syncthreads();
    for (std::uint32_t entry = 0; entry < entries; ++entry) {
        const float* kv = attention_entry(
            sliding,
            sliding_length,
            sliding_starts[batch],
            sliding_capacity,
            compressed,
            compressed_length,
            selected,
            selected_count,
            entry,
            head_dim);
        float dot = 0.0f;
        if (kv != nullptr) {
            for (std::uint32_t feature = threadIdx.x; feature < head_dim;
                 feature += blockDim.x) {
                dot = fmaf(q[feature], kv[feature], dot);
            }
            dot = block_sum(dot) * scaling;
        } else {
            dot = -__int_as_float(0x7f800000);
        }
        if (threadIdx.x == 0) {
            maximum = fmaxf(maximum, dot);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        denominator = expf(sinks[head] - maximum);
    }
    __syncthreads();
    for (std::uint32_t entry = 0; entry < entries; ++entry) {
        const float* kv = attention_entry(
            sliding,
            sliding_length,
            sliding_starts[batch],
            sliding_capacity,
            compressed,
            compressed_length,
            selected,
            selected_count,
            entry,
            head_dim);
        float dot = 0.0f;
        if (kv != nullptr) {
            for (std::uint32_t feature = threadIdx.x; feature < head_dim;
                 feature += blockDim.x) {
                dot = fmaf(q[feature], kv[feature], dot);
            }
            dot = block_sum(dot) * scaling;
        } else {
            dot = -__int_as_float(0x7f800000);
        }
        if (threadIdx.x == 0) {
            logit = expf(dot - maximum);
            denominator += logit;
        }
        __syncthreads();
    }

    float* out = output + (static_cast<std::size_t>(batch) * heads + head) * head_dim;
    for (std::uint32_t feature = threadIdx.x; feature < head_dim; feature += blockDim.x) {
        out[feature] = 0.0f;
    }
    __syncthreads();
    for (std::uint32_t entry = 0; entry < entries; ++entry) {
        const float* kv = attention_entry(
            sliding,
            sliding_length,
            sliding_starts[batch],
            sliding_capacity,
            compressed,
            compressed_length,
            selected,
            selected_count,
            entry,
            head_dim);
        float dot = 0.0f;
        if (kv != nullptr) {
            for (std::uint32_t feature = threadIdx.x; feature < head_dim;
                 feature += blockDim.x) {
                dot = fmaf(q[feature], kv[feature], dot);
            }
            dot = block_sum(dot) * scaling;
        } else {
            dot = -__int_as_float(0x7f800000);
        }
        if (threadIdx.x == 0) {
            logit = expf(dot - maximum) / denominator;
        }
        __syncthreads();
        if (kv != nullptr) {
            for (std::uint32_t feature = threadIdx.x; feature < head_dim;
                 feature += blockDim.x) {
                out[feature] = fmaf(logit, kv[feature], out[feature]);
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ const float* causal_attention_entry(
    const float* sliding,
    std::uint32_t sliding_length,
    std::uint32_t sliding_start,
    std::uint32_t sliding_capacity,
    const float* current,
    std::uint32_t current_start,
    std::uint32_t query_offset,
    const float* compressed,
    std::uint32_t compressed_length,
    const std::int32_t* selected,
    std::uint32_t selected_count,
    std::uint32_t causal_compressed_length,
    std::uint32_t entry,
    std::uint32_t head_dim) {
    const std::uint32_t current_visible = query_offset + 1;
    const std::uint32_t window = sliding_capacity + 1;
    const std::uint32_t total_visible = sliding_length + current_visible;
    const std::uint32_t skipped =
        total_visible > window ? total_visible - window : 0;
    const std::uint32_t prior_skipped =
        skipped < sliding_length ? skipped : sliding_length;
    const std::uint32_t prior_visible = sliding_length - prior_skipped;
    const std::uint32_t current_skipped = skipped - prior_skipped;
    const std::uint32_t current_kept = current_visible - current_skipped;
    if (entry < prior_visible) {
        const std::uint32_t logical = prior_skipped + entry;
        const std::uint32_t slot =
            (sliding_start + logical) % sliding_capacity;
        return sliding + static_cast<std::size_t>(slot) * head_dim;
    }
    entry -= prior_visible;
    if (entry < current_kept) {
        return current
            + static_cast<std::size_t>(
                  current_start + current_skipped + entry)
                * head_dim;
    }
    const std::uint32_t compressed_entry = entry - current_kept;
    const std::int32_t index =
        selected_count == 0
            ? static_cast<std::int32_t>(compressed_entry)
            : selected[compressed_entry];
    if (index < 0
        || static_cast<std::uint32_t>(index) >= compressed_length
        || static_cast<std::uint32_t>(index) >= causal_compressed_length) {
        return nullptr;
    }
    return compressed + static_cast<std::size_t>(index) * head_dim;
}

__global__ void causal_attention_f32_kernel(
    const float* __restrict__ query,
    const float* const* __restrict__ sliding_tables,
    const std::uint32_t* __restrict__ sliding_lengths,
    const std::uint32_t* __restrict__ sliding_starts,
    const float* __restrict__ current_kv,
    const std::uint32_t* __restrict__ current_sequence_starts,
    const std::uint32_t* __restrict__ query_offsets,
    const std::uint32_t* __restrict__ positions,
    const float* const* __restrict__ compressed_tables,
    const std::uint32_t* __restrict__ compressed_lengths,
    const std::int32_t* __restrict__ selected_indices,
    const float* __restrict__ sinks,
    float* __restrict__ output,
    std::uint32_t batch_rows,
    std::uint32_t current_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t sliding_capacity,
    std::uint32_t compression_ratio,
    std::uint32_t selected_count,
    float scaling) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t batch = blockIdx.y;
    if (head >= heads || batch >= batch_rows) {
        return;
    }
    const std::uint32_t current_start = current_sequence_starts[batch];
    const std::uint32_t query_offset = query_offsets[batch];
    if (current_start >= current_rows
        || query_offset >= current_rows - current_start) {
        return;
    }
    const float* q =
        query + (static_cast<std::size_t>(batch) * heads + head) * head_dim;
    const float* sliding = sliding_tables[batch];
    const std::uint32_t sliding_length = sliding_lengths[batch];
    const float* compressed = compressed_tables[batch];
    const std::uint32_t compressed_length = compressed_lengths[batch];
    const std::int32_t* selected =
        selected_count == 0
            ? nullptr
            : selected_indices + static_cast<std::size_t>(batch) * selected_count;
    const std::uint32_t causal_compressed_length =
        compression_ratio == 0
            ? 0
            : min(
                  compressed_length,
                  (positions[batch] + 1) / compression_ratio);
    const std::uint32_t current_visible = query_offset + 1;
    const std::uint32_t window = sliding_capacity + 1;
    const std::uint32_t total_visible = sliding_length + current_visible;
    const std::uint32_t sliding_entries =
        total_visible < window ? total_visible : window;
    const std::uint32_t compressed_entries =
        selected_count == 0 ? causal_compressed_length : selected_count;
    const std::uint32_t entries = sliding_entries + compressed_entries;

    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale;
    __shared__ float entry_scale;
    if (threadIdx.x == 0) {
        maximum = sinks[head];
        denominator = 1.0f;
    }
    float accumulators[2] = {0.0f, 0.0f};
    __syncthreads();
    for (std::uint32_t entry = 0; entry < entries; ++entry) {
        const float* kv = causal_attention_entry(
            sliding,
            sliding_length,
            sliding_starts[batch],
            sliding_capacity,
            current_kv,
            current_start,
            query_offset,
            compressed,
            compressed_length,
            selected,
            selected_count,
            causal_compressed_length,
            entry,
            head_dim);
        float dot = 0.0f;
        if (kv != nullptr) {
            for (std::uint32_t feature = threadIdx.x; feature < head_dim;
                 feature += blockDim.x) {
                dot = fmaf(q[feature], kv[feature], dot);
            }
            dot = block_sum(dot) * scaling;
        } else {
            dot = -__int_as_float(0x7f800000);
        }
        if (threadIdx.x == 0) {
            const float next_maximum = fmaxf(maximum, dot);
            old_scale = expf(maximum - next_maximum);
            entry_scale = kv == nullptr ? 0.0f : expf(dot - next_maximum);
            denominator =
                denominator * old_scale + entry_scale;
            maximum = next_maximum;
        }
        __syncthreads();
        std::uint32_t accumulator_index = 0;
        for (std::uint32_t feature = threadIdx.x; feature < head_dim;
             feature += blockDim.x, ++accumulator_index) {
            const float value = kv == nullptr ? 0.0f : kv[feature];
            accumulators[accumulator_index] =
                accumulators[accumulator_index] * old_scale
                + entry_scale * value;
        }
        __syncthreads();
    }
    float* out =
        output + (static_cast<std::size_t>(batch) * heads + head) * head_dim;
    std::uint32_t accumulator_index = 0;
    for (std::uint32_t feature = threadIdx.x; feature < head_dim;
         feature += blockDim.x, ++accumulator_index) {
        out[feature] = accumulators[accumulator_index] / denominator;
    }
}

__global__ void indexer_topk_f32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ head_weights,
    const float* const* __restrict__ compressed_tables,
    const std::uint32_t* __restrict__ compressed_lengths,
    const std::uint32_t* __restrict__ positions,
    std::int32_t* __restrict__ selected_indices,
    std::uint32_t batch_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t compression_ratio,
    std::uint32_t top_k,
    float query_scale,
    float weights_scale) {
    const std::uint32_t batch = blockIdx.x;
    if (batch >= batch_rows) {
        return;
    }
    extern __shared__ unsigned char shared_storage[];
    float* best_scores = reinterpret_cast<float*>(shared_storage);
    std::int32_t* best_indices =
        reinterpret_cast<std::int32_t*>(best_scores + top_k);
    for (std::uint32_t slot = threadIdx.x; slot < top_k;
         slot += blockDim.x) {
        best_scores[slot] = -__int_as_float(0x7f800000);
        best_indices[slot] = -1;
    }
    __syncthreads();

    const float* compressed = compressed_tables[batch];
    const std::uint32_t causal_length = min(
        compressed_lengths[batch],
        (positions[batch] + 1) / compression_ratio);
    for (std::uint32_t entry = 0; entry < causal_length; ++entry) {
        float contribution = 0.0f;
        if (threadIdx.x < heads) {
            const float* q =
                query
                + (static_cast<std::size_t>(batch) * heads + threadIdx.x)
                    * head_dim;
            const float* key =
                compressed + static_cast<std::size_t>(entry) * head_dim;
            float dot = 0.0f;
            for (std::uint32_t feature = 0; feature < head_dim; ++feature) {
                dot = fmaf(q[feature], key[feature], dot);
            }
            contribution =
                head_weights[static_cast<std::size_t>(batch) * heads + threadIdx.x]
                * fmaxf(0.0f, dot * query_scale)
                * weights_scale;
        }
        const float score = block_sum(contribution);
        if (threadIdx.x == 0 && score > best_scores[top_k - 1]) {
            std::uint32_t insert = top_k - 1;
            while (insert > 0 && score > best_scores[insert - 1]) {
                best_scores[insert] = best_scores[insert - 1];
                best_indices[insert] = best_indices[insert - 1];
                --insert;
            }
            best_scores[insert] = score;
            best_indices[insert] = static_cast<std::int32_t>(entry);
        }
        __syncthreads();
    }
    for (std::uint32_t slot = threadIdx.x; slot < top_k;
         slot += blockDim.x) {
        selected_indices[
            static_cast<std::size_t>(batch) * top_k + slot] =
            best_indices[slot];
    }
}

__device__ __forceinline__ float sqrt_softplus(float value) {
    const float softplus =
        value > 20.0f ? value : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    return sqrtf(softplus);
}

__global__ void router_topk_f32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    std::uint32_t* __restrict__ indices,
    float* __restrict__ weights,
    std::uint32_t batch_rows,
    std::uint32_t experts,
    std::uint32_t top_k,
    float routed_scale) {
    const std::uint32_t batch = blockIdx.x;
    if (batch >= batch_rows) {
        return;
    }
    extern __shared__ float shared[];
    float* scores = shared;
    float* selection = shared + experts;
    for (std::uint32_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float score =
            sqrt_softplus(logits[static_cast<std::size_t>(batch) * experts + expert]);
        scores[expert] = score;
        selection[expert] = score + bias[expert];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 1.0e-20f;
        for (std::uint32_t route = 0; route < top_k; ++route) {
            std::uint32_t best = 0;
            float best_score = -__int_as_float(0x7f800000);
            for (std::uint32_t expert = 0; expert < experts; ++expert) {
                const float candidate = selection[expert];
                if (candidate > best_score) {
                    best_score = candidate;
                    best = expert;
                }
            }
            selection[best] = -__int_as_float(0x7f800000);
            indices[static_cast<std::size_t>(batch) * top_k + route] = best;
            const float weight = scores[best];
            weights[static_cast<std::size_t>(batch) * top_k + route] = weight;
            sum += weight;
        }
        for (std::uint32_t route = 0; route < top_k; ++route) {
            weights[static_cast<std::size_t>(batch) * top_k + route] =
                weights[static_cast<std::size_t>(batch) * top_k + route]
                * routed_scale / sum;
        }
    }
}

__global__ void router_hash_f32_kernel(
    const float* __restrict__ logits,
    const std::int64_t* __restrict__ token_to_expert,
    const std::uint32_t* __restrict__ token_ids,
    std::uint32_t* __restrict__ indices,
    float* __restrict__ weights,
    std::uint32_t batch_rows,
    std::uint32_t vocab,
    std::uint32_t experts,
    std::uint32_t top_k,
    float routed_scale) {
    const std::uint32_t batch = blockIdx.x;
    if (batch >= batch_rows || token_ids[batch] >= vocab) {
        return;
    }
    __shared__ float route_weights[32];
    const std::uint32_t token = token_ids[batch];
    if (threadIdx.x < top_k) {
        const std::int64_t expert =
            token_to_expert[static_cast<std::size_t>(token) * top_k + threadIdx.x];
        if (expert < 0 || expert >= experts) {
            indices[static_cast<std::size_t>(batch) * top_k + threadIdx.x] = experts;
            route_weights[threadIdx.x] = 0.0f;
        } else {
            indices[static_cast<std::size_t>(batch) * top_k + threadIdx.x] =
                static_cast<std::uint32_t>(expert);
            route_weights[threadIdx.x] = sqrt_softplus(
                logits[static_cast<std::size_t>(batch) * experts + expert]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 1.0e-20f;
        for (std::uint32_t route = 0; route < top_k; ++route) {
            sum += route_weights[route];
        }
        for (std::uint32_t route = 0; route < top_k; ++route) {
            weights[static_cast<std::size_t>(batch) * top_k + route] =
                route_weights[route] * routed_scale / sum;
        }
    }
}

__global__ void compress_windows_f32_kernel(
    const float* __restrict__ kv,
    const float* __restrict__ gate,
    const float* __restrict__ position_bias,
    const float* __restrict__ prior_kv,
    const float* __restrict__ prior_gate,
    float* __restrict__ output,
    std::uint32_t windows,
    std::uint32_t ratio,
    std::uint32_t compressed_width,
    bool overlapping,
    bool has_prior) {
    const std::uint32_t feature = blockIdx.x;
    const std::uint32_t window = blockIdx.y;
    if (feature >= compressed_width || window >= windows) {
        return;
    }
    const std::uint32_t projected_width =
        overlapping ? 2 * compressed_width : compressed_width;
    const std::uint32_t slots = overlapping ? 2 * ratio : ratio;
    const std::uint32_t slot = threadIdx.x;
    __shared__ float shared_maximum;
    __shared__ float shared_denominator;
    float kv_value = 0.0f;
    float gate_value = -__int_as_float(0x7f800000);
    if (slot < slots) {
        if (!overlapping || slot >= ratio) {
            const std::uint32_t current_slot = overlapping ? slot - ratio : slot;
            const std::size_t row =
                (static_cast<std::size_t>(window) * ratio + current_slot)
                * projected_width;
            const std::uint32_t component =
                overlapping ? compressed_width + feature : feature;
            kv_value = kv[row + component];
            gate_value =
                gate[row + component]
                + position_bias[static_cast<std::size_t>(current_slot) * projected_width
                    + component];
        } else if (window > 0) {
            const std::size_t row =
                (static_cast<std::size_t>(window - 1) * ratio + slot)
                * projected_width;
            kv_value = kv[row + feature];
            gate_value =
                gate[row + feature]
                + position_bias[static_cast<std::size_t>(slot) * projected_width + feature];
        } else if (has_prior) {
            kv_value = prior_kv[static_cast<std::size_t>(slot) * compressed_width + feature];
            gate_value =
                prior_gate[static_cast<std::size_t>(slot) * compressed_width + feature];
        }
    }
    const float maximum = block_max(gate_value);
    if (threadIdx.x == 0) {
        shared_maximum = maximum;
    }
    __syncthreads();
    const float exponential =
        slot < slots ? expf(gate_value - shared_maximum) : 0.0f;
    const float denominator = block_sum(exponential);
    if (threadIdx.x == 0) {
        shared_denominator = denominator;
    }
    __syncthreads();
    const float contribution =
        slot < slots ? kv_value * exponential / shared_denominator : 0.0f;
    const float compressed = block_sum(contribution);
    if (threadIdx.x == 0) {
        output[static_cast<std::size_t>(window) * compressed_width + feature] =
            compressed;
    }
}

__global__ void store_compression_overlap_f32_kernel(
    const float* __restrict__ kv,
    const float* __restrict__ gate,
    const float* __restrict__ position_bias,
    float* __restrict__ overlap_kv,
    float* __restrict__ overlap_gate,
    std::uint32_t window,
    std::uint32_t ratio,
    std::uint32_t compressed_width) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t values = ratio * compressed_width;
    if (index >= values) {
        return;
    }
    const std::uint32_t slot = index / compressed_width;
    const std::uint32_t feature = index % compressed_width;
    const std::uint32_t projected_width = 2 * compressed_width;
    const std::size_t source =
        (static_cast<std::size_t>(window) * ratio + slot) * projected_width
        + feature;
    overlap_kv[index] = kv[source];
    overlap_gate[index] =
        gate[source]
        + position_bias[
            static_cast<std::size_t>(slot) * projected_width + feature];
}

__global__ void arithmetic_positions_u32_kernel(
    std::uint32_t* __restrict__ positions,
    std::uint32_t len,
    std::uint32_t start,
    std::uint32_t stride) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        positions[index] = start + index * stride;
    }
}

__global__ void swiglu_pair_clamped_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t width,
    float limit) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t values = static_cast<std::size_t>(rows) * width;
    if (index >= values) {
        return;
    }
    const float gate_value = fminf(gate[index], limit);
    const float up_value = fmaxf(-limit, fminf(up[index], limit));
    output[index] = gate_value * sigmoid(gate_value) * up_value;
}

__global__ void routed_accumulate_f32_kernel(
    const float* __restrict__ route_output,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    std::uint32_t rows,
    std::uint32_t routes_per_row,
    std::uint32_t width) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t values = static_cast<std::size_t>(rows) * width;
    if (index >= values) {
        return;
    }
    const std::uint32_t row = index / width;
    const std::uint32_t feature = index % width;
    float value = 0.0f;
    for (std::uint32_t route = 0; route < routes_per_row; ++route) {
        const std::size_t route_index =
            static_cast<std::size_t>(row) * routes_per_row + route;
        value = fmaf(
            route_weights[route_index],
            route_output[route_index * width + feature],
            value);
    }
    output[index] = value;
}

}  // namespace

extern "C" cudaError_t infer_deepseek4_block_fp8_linear_f32_on_stream(
    const float* input,
    const std::uint8_t* weight,
    const std::uint8_t* scales,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t rows,
    std::uint32_t cols,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || scales == nullptr || output == nullptr
        || batch_rows == 0 || rows == 0 || cols == 0 || rows % kScaleBlock != 0
        || cols % kScaleBlock != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(rows, batch_rows);
    block_fp8_linear_f32_kernel<<<grid, kThreads, 0, stream>>>(
        input, weight, scales, output, batch_rows, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_block_fp8_grouped_linear_f32_on_stream(
    const float* input,
    const std::uint8_t* weight,
    const std::uint8_t* scales,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t groups,
    std::uint32_t rows_per_group,
    std::uint32_t cols_per_group,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || scales == nullptr || output == nullptr
        || batch_rows == 0 || groups == 0 || rows_per_group == 0
        || cols_per_group == 0 || rows_per_group % kScaleBlock != 0
        || cols_per_group % kScaleBlock != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(groups * rows_per_group, batch_rows);
    block_fp8_grouped_linear_f32_kernel<<<grid, kThreads, 0, stream>>>(
        input,
        weight,
        scales,
        output,
        batch_rows,
        groups,
        rows_per_group,
        cols_per_group);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_hyper_prepare_f32_on_stream(
    const float* streams,
    const float* fn,
    const float* base,
    const float* scale,
    float* post,
    float* comb,
    float* collapsed,
    std::uint32_t batch_rows,
    std::uint32_t hidden,
    float rms_eps,
    float hc_eps,
    std::uint32_t sinkhorn_iters,
    cudaStream_t stream) {
    if (streams == nullptr || fn == nullptr || base == nullptr || scale == nullptr
        || post == nullptr || comb == nullptr || collapsed == nullptr || batch_rows == 0
        || hidden == 0 || !isfinite(rms_eps) || rms_eps <= 0.0f || !isfinite(hc_eps)
        || hc_eps <= 0.0f || sinkhorn_iters == 0) {
        return cudaErrorInvalidValue;
    }
    hyper_prepare_f32_kernel<<<batch_rows, kThreads, 0, stream>>>(
        streams,
        fn,
        base,
        scale,
        post,
        comb,
        collapsed,
        batch_rows,
        hidden,
        rms_eps,
        hc_eps,
        sinkhorn_iters);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_hyper_apply_f32_on_stream(
    const float* streams,
    const float* sublayer,
    const float* post,
    const float* comb,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t hidden,
    cudaStream_t stream) {
    if (streams == nullptr || sublayer == nullptr || post == nullptr || comb == nullptr
        || output == nullptr || batch_rows == 0 || hidden == 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(kHyperStreams, batch_rows);
    hyper_apply_f32_kernel<<<grid, kThreads, 0, stream>>>(
        streams, sublayer, post, comb, output, batch_rows, hidden);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_hyper_head_f32_on_stream(
    const float* streams,
    const float* fn,
    const float* base,
    const float* scale,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t hidden,
    float rms_eps,
    float hc_eps,
    cudaStream_t stream) {
    if (streams == nullptr || fn == nullptr || base == nullptr || scale == nullptr
        || output == nullptr || batch_rows == 0 || hidden == 0 || !isfinite(rms_eps)
        || rms_eps <= 0.0f || !isfinite(hc_eps) || hc_eps <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    hyper_head_f32_kernel<<<batch_rows, kThreads, 0, stream>>>(
        streams, fn, base, scale, output, batch_rows, hidden, rms_eps, hc_eps);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_rope_interleaved_trailing_f32_on_stream(
    float* values,
    const float* inv_freq,
    const std::uint32_t* positions,
    std::uint32_t batch_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t rope_dim,
    float direction,
    cudaStream_t stream) {
    if (values == nullptr || inv_freq == nullptr || positions == nullptr || batch_rows == 0
        || heads == 0 || head_dim == 0 || rope_dim == 0 || rope_dim > head_dim
        || rope_dim % 2 != 0 || !isfinite(direction)
        || (direction != 1.0f && direction != -1.0f)) {
        return cudaErrorInvalidValue;
    }
    const std::size_t total_pairs =
        static_cast<std::size_t>(batch_rows) * heads * (rope_dim / 2);
    const std::uint32_t blocks =
        static_cast<std::uint32_t>((total_pairs + kThreads - 1) / kThreads);
    rope_interleaved_trailing_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        values, inv_freq, positions, batch_rows, heads, head_dim, rope_dim, direction);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_attention_f32_on_stream(
    const float* query,
    const float* const* sliding_tables,
    const std::uint32_t* sliding_lengths,
    const std::uint32_t* sliding_starts,
    const float* const* compressed_tables,
    const std::uint32_t* compressed_lengths,
    const std::int32_t* selected_indices,
    const float* sinks,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t sliding_capacity,
    std::uint32_t selected_count,
    float scaling,
    cudaStream_t stream) {
    if (query == nullptr || sliding_tables == nullptr || sliding_lengths == nullptr
        || sliding_starts == nullptr || compressed_tables == nullptr
        || compressed_lengths == nullptr || sinks == nullptr || output == nullptr
        || batch_rows == 0 || heads == 0 || head_dim == 0 || sliding_capacity == 0
        || (selected_count != 0 && selected_indices == nullptr) || !isfinite(scaling)
        || scaling <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(heads, batch_rows);
    attention_f32_kernel<<<grid, kThreads, 0, stream>>>(
        query,
        sliding_tables,
        sliding_lengths,
        sliding_starts,
        compressed_tables,
        compressed_lengths,
        selected_indices,
        sinks,
        output,
        batch_rows,
        heads,
        head_dim,
        sliding_capacity,
        selected_count,
        scaling);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_causal_attention_f32_on_stream(
    const float* query,
    const float* const* sliding_tables,
    const std::uint32_t* sliding_lengths,
    const std::uint32_t* sliding_starts,
    const float* current_kv,
    const std::uint32_t* current_sequence_starts,
    const std::uint32_t* query_offsets,
    const std::uint32_t* positions,
    const float* const* compressed_tables,
    const std::uint32_t* compressed_lengths,
    const std::int32_t* selected_indices,
    const float* sinks,
    float* output,
    std::uint32_t batch_rows,
    std::uint32_t current_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t sliding_capacity,
    std::uint32_t compression_ratio,
    std::uint32_t selected_count,
    float scaling,
    cudaStream_t stream) {
    if (query == nullptr || sliding_tables == nullptr
        || sliding_lengths == nullptr || sliding_starts == nullptr
        || current_kv == nullptr || current_sequence_starts == nullptr
        || query_offsets == nullptr || positions == nullptr
        || compressed_tables == nullptr || compressed_lengths == nullptr
        || sinks == nullptr || output == nullptr || batch_rows == 0
        || current_rows == 0 || heads == 0 || head_dim == 0
        || head_dim > 2 * kThreads || sliding_capacity == 0
        || (selected_count != 0 && selected_indices == nullptr)
        || !isfinite(scaling) || scaling <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(heads, batch_rows);
    causal_attention_f32_kernel<<<grid, kThreads, 0, stream>>>(
        query,
        sliding_tables,
        sliding_lengths,
        sliding_starts,
        current_kv,
        current_sequence_starts,
        query_offsets,
        positions,
        compressed_tables,
        compressed_lengths,
        selected_indices,
        sinks,
        output,
        batch_rows,
        current_rows,
        heads,
        head_dim,
        sliding_capacity,
        compression_ratio,
        selected_count,
        scaling);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_indexer_topk_f32_on_stream(
    const float* query,
    const float* head_weights,
    const float* const* compressed_tables,
    const std::uint32_t* compressed_lengths,
    const std::uint32_t* positions,
    std::int32_t* selected_indices,
    std::uint32_t batch_rows,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t compression_ratio,
    std::uint32_t top_k,
    cudaStream_t stream) {
    if (query == nullptr || head_weights == nullptr
        || compressed_tables == nullptr || compressed_lengths == nullptr
        || positions == nullptr || selected_indices == nullptr
        || batch_rows == 0 || heads == 0 || heads > kThreads
        || head_dim == 0 || compression_ratio == 0 || top_k == 0
        || top_k > 4096) {
        return cudaErrorInvalidValue;
    }
    const std::size_t shared_bytes =
        static_cast<std::size_t>(top_k)
        * (sizeof(float) + sizeof(std::int32_t));
    indexer_topk_f32_kernel<<<batch_rows, kThreads, shared_bytes, stream>>>(
        query,
        head_weights,
        compressed_tables,
        compressed_lengths,
        positions,
        selected_indices,
        batch_rows,
        heads,
        head_dim,
        compression_ratio,
        top_k,
        rsqrtf(static_cast<float>(head_dim)),
        rsqrtf(static_cast<float>(heads)));
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_router_topk_f32_on_stream(
    const float* logits,
    const float* bias,
    std::uint32_t* indices,
    float* weights,
    std::uint32_t batch_rows,
    std::uint32_t experts,
    std::uint32_t top_k,
    float routed_scale,
    cudaStream_t stream) {
    if (logits == nullptr || bias == nullptr || indices == nullptr || weights == nullptr
        || batch_rows == 0 || experts == 0 || top_k == 0 || top_k > experts
        || !isfinite(routed_scale) || routed_scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const std::size_t shared_bytes = static_cast<std::size_t>(2) * experts * sizeof(float);
    router_topk_f32_kernel<<<batch_rows, kThreads, shared_bytes, stream>>>(
        logits, bias, indices, weights, batch_rows, experts, top_k, routed_scale);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_router_hash_f32_on_stream(
    const float* logits,
    const std::int64_t* token_to_expert,
    const std::uint32_t* token_ids,
    std::uint32_t* indices,
    float* weights,
    std::uint32_t batch_rows,
    std::uint32_t vocab,
    std::uint32_t experts,
    std::uint32_t top_k,
    float routed_scale,
    cudaStream_t stream) {
    if (logits == nullptr || token_to_expert == nullptr || token_ids == nullptr
        || indices == nullptr || weights == nullptr || batch_rows == 0 || vocab == 0
        || experts == 0 || top_k == 0 || top_k > experts || top_k > 32
        || !isfinite(routed_scale) || routed_scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    router_hash_f32_kernel<<<batch_rows, 32, 0, stream>>>(
        logits,
        token_to_expert,
        token_ids,
        indices,
        weights,
        batch_rows,
        vocab,
        experts,
        top_k,
        routed_scale);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_compress_windows_f32_on_stream(
    const float* kv,
    const float* gate,
    const float* position_bias,
    const float* prior_kv,
    const float* prior_gate,
    float* output,
    std::uint32_t windows,
    std::uint32_t ratio,
    std::uint32_t compressed_width,
    bool overlapping,
    bool has_prior,
    cudaStream_t stream) {
    if (kv == nullptr || gate == nullptr || position_bias == nullptr || output == nullptr
        || windows == 0 || ratio == 0 || ratio > kThreads || compressed_width == 0
        || (has_prior && (prior_kv == nullptr || prior_gate == nullptr))) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(compressed_width, windows);
    compress_windows_f32_kernel<<<grid, kThreads, 0, stream>>>(
        kv,
        gate,
        position_bias,
        prior_kv,
        prior_gate,
        output,
        windows,
        ratio,
        compressed_width,
        overlapping,
        has_prior);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_store_compression_overlap_f32_on_stream(
    const float* kv,
    const float* gate,
    const float* position_bias,
    float* overlap_kv,
    float* overlap_gate,
    std::uint32_t window,
    std::uint32_t ratio,
    std::uint32_t compressed_width,
    cudaStream_t stream) {
    if (kv == nullptr || gate == nullptr || position_bias == nullptr
        || overlap_kv == nullptr || overlap_gate == nullptr || ratio == 0
        || compressed_width == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t values = ratio * compressed_width;
    const std::uint32_t blocks =
        (values + kThreads - 1) / kThreads;
    store_compression_overlap_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        kv,
        gate,
        position_bias,
        overlap_kv,
        overlap_gate,
        window,
        ratio,
        compressed_width);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_arithmetic_positions_u32_on_stream(
    std::uint32_t* positions,
    std::uint32_t len,
    std::uint32_t start,
    std::uint32_t stride,
    cudaStream_t stream) {
    if (positions == nullptr || len == 0 || stride == 0) {
        return cudaErrorInvalidValue;
    }
    const std::uint32_t blocks =
        (len + kThreads - 1) / kThreads;
    arithmetic_positions_u32_kernel<<<blocks, kThreads, 0, stream>>>(
        positions, len, start, stride);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_swiglu_pair_clamped_f32_on_stream(
    const float* gate,
    const float* up,
    float* output,
    std::uint32_t rows,
    std::uint32_t width,
    float limit,
    cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || rows == 0
        || width == 0 || !isfinite(limit) || limit <= 0.0f) {
        return cudaErrorInvalidValue;
    }
    const std::size_t values = static_cast<std::size_t>(rows) * width;
    const std::uint32_t blocks =
        static_cast<std::uint32_t>((values + kThreads - 1) / kThreads);
    swiglu_pair_clamped_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        gate, up, output, rows, width, limit);
    return cudaGetLastError();
}

extern "C" cudaError_t infer_deepseek4_routed_accumulate_f32_on_stream(
    const float* route_output,
    const float* route_weights,
    float* output,
    std::uint32_t rows,
    std::uint32_t routes_per_row,
    std::uint32_t width,
    cudaStream_t stream) {
    if (route_output == nullptr || route_weights == nullptr || output == nullptr
        || rows == 0 || routes_per_row == 0 || width == 0) {
        return cudaErrorInvalidValue;
    }
    const std::size_t values = static_cast<std::size_t>(rows) * width;
    const std::uint32_t blocks =
        static_cast<std::uint32_t>((values + kThreads - 1) / kThreads);
    routed_accumulate_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        route_output, route_weights, output, rows, routes_per_row, width);
    return cudaGetLastError();
}
