#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Compile the production Eider kernel itself. This keeps the compiler
// comparison tied to the current W4A16 implementation instead of a rewrite.
#include "../../crates/eider-cuda/native/sm121_w4a16.cu"

namespace {

constexpr int kBenchTileM = 16;
constexpr int kBenchTileK = 16;
constexpr int kWarps = 16;
constexpr int kWarmupLaunches = 10;
constexpr int kTimedLaunches = 100;
constexpr int kQwenHidden = 5120;
constexpr int kQwenIntermediate = 17408;
constexpr int kQwenGateUp = kQwenIntermediate * 2;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

struct Buffers {
    std::uint32_t* indices = nullptr;
    float* input = nullptr;
    std::uint8_t* weight = nullptr;
    std::uint8_t* scales = nullptr;
    float* global_scale = nullptr;
    std::uint16_t* output_bf16 = nullptr;
    float* output_f32 = nullptr;

    ~Buffers() {
        cudaFree(indices);
        cudaFree(input);
        cudaFree(weight);
        cudaFree(scales);
        cudaFree(global_scale);
        cudaFree(output_bf16);
        cudaFree(output_f32);
    }
};

__global__ __launch_bounds__(512) void fixed_dense_w4a16_kernel(
    const std::uint32_t* __restrict__ indices,
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ tiled_weight,
    const std::uint8_t* __restrict__ tiled_scales,
    const float* __restrict__ global_scales,
    std::uint16_t* __restrict__ output_bf16,
    float* __restrict__ output_f32,
    std::uint32_t out_features,
    std::uint32_t in_features) {
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t warp = threadIdx.x >> 5u;
    const std::uint32_t out_tile = blockIdx.x;
    const std::uint32_t expert = indices[0];
    const std::uint32_t k_tiles = in_features / kBenchTileK;
    const std::size_t expert_weight_stride =
        static_cast<std::size_t>(out_features) * in_features / 2;
    const std::size_t expert_scale_stride =
        static_cast<std::size_t>(out_features) * in_features / kBenchTileK;
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

    for (std::uint32_t k_tile = warp; k_tile < k_tiles; k_tile += kWarps) {
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

        const std::uint8_t* weight_row0 = tile_weight + row0 * (kBenchTileK / 2);
        const std::uint8_t* weight_row1 = tile_weight + row1 * (kBenchTileK / 2);
        const std::uint32_t a0 = dequant_pair(weight_row0[pair0 / 2], scale0);
        const std::uint32_t a1 = dequant_pair(weight_row0[pair1 / 2], scale0);
        const std::uint32_t a2 = dequant_pair(weight_row1[pair0 / 2], scale1);
        const std::uint32_t a3 = dequant_pair(weight_row1[pair1 / 2], scale1);

        std::uint32_t b0 = 0;
        std::uint32_t b1 = 0;
        if (lane < 4) {
            const float* input_tile = input + k_tile * kBenchTileK;
            b0 = pack_bf16_pair(input_tile[pair0], input_tile[pair0 + 1]);
            b1 = pack_bf16_pair(input_tile[pair1], input_tile[pair1 + 1]);
        }
        b0 = __shfl_sync(0xffffffffu, b0, lane & 3u);
        b1 = __shfl_sync(0xffffffffu, b1, lane & 3u);
        mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }

    __shared__ float partial[kWarps][kBenchTileM];
    if ((lane & 3u) == 0) {
        partial[warp][row0] = d0;
        partial[warp][row1] = d2;
    }
    __syncthreads();
    if (threadIdx.x < kBenchTileM) {
        float value = 0.0f;
#pragma unroll
        for (std::uint32_t partial_index = 0; partial_index < kWarps;
             ++partial_index) {
            value += partial[partial_index][threadIdx.x];
        }
        const std::size_t output_index =
            static_cast<std::size_t>(out_tile) * kBenchTileM + threadIdx.x;
        const __nv_bfloat16 output_value = __float2bfloat16_rn(value);
        output_bf16[output_index] = __bfloat16_as_ushort(output_value);
        output_f32[output_index] = __bfloat162float(output_value);
    }
}

void make_buffers(
    Buffers& buffers,
    int m,
    int k,
    float global_scale,
    const std::vector<std::uint8_t>* scales,
    int batch = 1) {
    const std::vector<std::uint32_t> indices(batch, 0);
    const std::vector<float> input(static_cast<std::size_t>(batch) * k, 1.0f);
    check(cudaMalloc(&buffers.indices, indices.size() * sizeof(std::uint32_t)), "cudaMalloc(W4A16 indices)");
    check(cudaMalloc(&buffers.input, input.size() * sizeof(float)), "cudaMalloc(W4A16 input)");
    check(cudaMalloc(&buffers.weight, static_cast<std::size_t>(m) * k / 2), "cudaMalloc(W4A16 weight)");
    check(cudaMalloc(&buffers.scales, static_cast<std::size_t>(m) * k / kBenchTileK), "cudaMalloc(W4A16 scales)");
    check(cudaMalloc(&buffers.global_scale, sizeof(float)), "cudaMalloc(W4A16 global scale)");
    check(cudaMalloc(&buffers.output_bf16, static_cast<std::size_t>(batch) * m * sizeof(std::uint16_t)), "cudaMalloc(W4A16 BF16 output)");
    check(cudaMalloc(&buffers.output_f32, static_cast<std::size_t>(batch) * m * sizeof(float)), "cudaMalloc(W4A16 F32 output)");
    check(cudaMemcpy(buffers.indices, indices.data(), indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy(W4A16 indices)");
    check(cudaMemcpy(buffers.input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(W4A16 input)");
    check(cudaMemset(buffers.weight, 0x22, static_cast<std::size_t>(m) * k / 2), "cudaMemset(W4A16 weight)");
    if (scales == nullptr) {
        check(cudaMemset(buffers.scales, 0x38, static_cast<std::size_t>(m) * k / kBenchTileK), "cudaMemset(W4A16 scales)");
    } else {
        check(cudaMemcpy(buffers.scales, scales->data(), scales->size(), cudaMemcpyHostToDevice), "cudaMemcpy(W4A16 scales)");
    }
    check(cudaMemcpy(buffers.global_scale, &global_scale, sizeof(global_scale), cudaMemcpyHostToDevice), "cudaMemcpy(W4A16 global scale)");
}

void launch(Buffers& buffers, int m, int k, int batch = 1) {
    check(
        infer_sm121_w4a16_gate_up_on_stream(
            buffers.indices,
            buffers.input,
            buffers.weight,
            buffers.scales,
            buffers.global_scale,
            buffers.output_bf16,
            buffers.output_f32,
            batch,
            1,
            m,
            k,
            nullptr),
        "infer_sm121_w4a16_gate_up_on_stream");
}

void launch_fixed(Buffers& buffers, int m, int k) {
    fixed_dense_w4a16_kernel<<<m / kBenchTileM, kWarps * 32>>>(
        buffers.indices,
        buffers.input,
        buffers.weight,
        buffers.scales,
        buffers.global_scale,
        buffers.output_bf16,
        buffers.output_f32,
        m,
        k);
    check(cudaGetLastError(), "fixed_dense_w4a16_kernel");
}

void check_output(Buffers& buffers, int values, float expected, const char* label) {
    std::vector<float> actual(values);
    check(cudaMemcpy(actual.data(), buffers.output_f32, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(W4A16 output)");
    for (std::size_t row = 0; row < actual.size(); ++row) {
        if (actual[row] != expected) {
            std::fprintf(stderr, "%s mismatch: row=%zu expected=%f actual=%f\n", label, row, expected, actual[row]);
            std::exit(EXIT_FAILURE);
        }
    }
}

void run_correctness_case() {
    constexpr int m = 32;
    constexpr int k = 32;
    std::vector<std::uint8_t> scales(m * k / kBenchTileK, 0x38);
    for (int out_tile = 0; out_tile < m / kBenchTileM; ++out_tile) {
        const int second_k_tile =
            (out_tile * (k / kBenchTileK) + 1) * kBenchTileM;
        std::fill_n(scales.begin() + second_k_tile, kBenchTileM, 0x40);
    }
    Buffers buffers;
    make_buffers(buffers, m, k, 0.5f, &scales);
    launch(buffers, m, k);
    check(cudaDeviceSynchronize(), "W4A16 correctness launch");
    check_output(buffers, m, 24.0f, "native W4A16 correctness");
    std::puts("native W4A16 preserves tiled scales, global scaling, and BF16 output rounding");
}

void run_batch_correctness_case() {
    constexpr int batch = 2;
    constexpr int m = 32;
    constexpr int k = 32;
    Buffers buffers;
    make_buffers(buffers, m, k, 1.0f, nullptr, batch);
    launch(buffers, m, k, batch);
    check(cudaDeviceSynchronize(), "W4A16 batch correctness launch");
    check_output(buffers, batch * m, static_cast<float>(k), "native W4A16 batch correctness");
    std::puts("native W4A16 eight-warp batch specialization passed");
}

void run_benchmark(const char* label, int m, int k, bool fixed) {
    Buffers buffers;
    make_buffers(buffers, m, k, 1.0f, nullptr);
    const auto launch_selected = [&]() {
        if (fixed) {
            launch_fixed(buffers, m, k);
        } else {
            launch(buffers, m, k);
        }
    };
    launch_selected();
    check(cudaDeviceSynchronize(), "W4A16 correctness launch");
    check_output(buffers, m, static_cast<float>(k), label);

    for (int iteration = 0; iteration < kWarmupLaunches; ++iteration) {
        launch_selected();
    }
    check(cudaDeviceSynchronize(), "W4A16 warmup");

    cudaEvent_t start{};
    cudaEvent_t end{};
    check(cudaEventCreate(&start), "cudaEventCreate(W4A16 start)");
    check(cudaEventCreate(&end), "cudaEventCreate(W4A16 end)");
    check(cudaEventRecord(start), "cudaEventRecord(W4A16 start)");
    for (int iteration = 0; iteration < kTimedLaunches; ++iteration) {
        launch_selected();
    }
    check(cudaEventRecord(end), "cudaEventRecord(W4A16 end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(W4A16 end)");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, end), "cudaEventElapsedTime(W4A16)");
    std::printf(
        "native%s %s W4A16 latency: %.3f us (%d launches)\n",
        fixed ? " fixed-16" : "",
        label,
        elapsed_ms * 1000.0 / static_cast<double>(kTimedLaunches),
        kTimedLaunches);
    check(cudaEventDestroy(start), "cudaEventDestroy(W4A16 start)");
    check(cudaEventDestroy(end), "cudaEventDestroy(W4A16 end)");
}

}  // namespace

int main() {
    check(cudaSetDevice(0), "cudaSetDevice");
    run_correctness_case();
    run_batch_correctness_case();
    run_benchmark("Qwen gate+up 34816x5120", kQwenGateUp, kQwenHidden, false);
    run_benchmark("Qwen down 5120x17408", kQwenHidden, kQwenIntermediate, false);
    run_benchmark("Qwen gate+up 34816x5120", kQwenGateUp, kQwenHidden, true);
    run_benchmark("Qwen down 5120x17408", kQwenHidden, kQwenIntermediate, true);
    return EXIT_SUCCESS;
}
