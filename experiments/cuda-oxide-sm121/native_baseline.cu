#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define EIDER_PRAGMA_IMPL(value) _Pragma(#value)
#define EIDER_PRAGMA(value) EIDER_PRAGMA_IMPL(value)

namespace {

constexpr int kLanes = 32;
constexpr int kWarmupLaunches = 100;
constexpr int kTimedLaunches = 10000;
constexpr int kTimedBatchLaunches = 1000;
constexpr int kGemvWarmupLaunches = 10;
constexpr int kGemvTimedLaunches = 100;
constexpr int kKLoopTiles = 64;
constexpr int kKLoopBlocks = 64;
constexpr int kQwenHidden = 5120;
constexpr int kQwenIntermediate = 17408;
constexpr int kQwenGateUp = kQwenIntermediate * 2;
constexpr std::uint32_t kPackedTwos = 0x04040404u;
constexpr std::uint32_t kUnitScaleWord = 0x38383838u;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

__global__ void native_nvfp4_mma_probe(
    std::uint32_t packed,
    float4* output) {
    const std::uint32_t a[4] = {packed, packed, packed, packed};
    const std::uint32_t b[2] = {packed, packed};
    const float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::uint16_t byte_id = 0;
    const std::uint16_t thread_id = 0;
    float d0;
    float d1;
    float d2;
    float d3;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "%14, {%15, %16},"
        "%17, {%15, %16};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(kUnitScaleWord), "h"(byte_id), "h"(thread_id),
          "r"(kUnitScaleWord));

    const int lane = threadIdx.x;
    output[blockIdx.x * blockDim.x + lane] = make_float4(d0, d1, d2, d3);
}

__global__ void native_nvfp4_mma_kloop(
    const uint4* a_tiles,
    const uint4* b_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t k_tiles,
    float4* output) {
    const int lane = threadIdx.x;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t byte_id = 0;
    const std::uint16_t thread_id = 0;

#if defined(NATIVE_UNROLL)
    EIDER_PRAGMA(unroll NATIVE_UNROLL)
#endif
    for (std::uint32_t tile = 0; tile < k_tiles; ++tile) {
        const uint4 a = a_tiles[tile * kLanes + lane];
        const uint4 b = b_tiles[tile * kLanes + lane];
        float n0;
        float n1;
        float n2;
        float n3;

        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "%14, {%15, %16},"
            "%17, {%15, %16};\n"
            : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
              "r"(b.x), "r"(b.y),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[tile]), "h"(byte_id), "h"(thread_id),
              "r"(sfb[tile]));
        d0 = n0;
        d1 = n1;
        d2 = n2;
        d3 = n3;
    }

    output[blockIdx.x * blockDim.x + lane] = make_float4(d0, d1, d2, d3);
}

__global__ void native_nvfp4_gemv(
    const uint4* a_tiles,
    const uint4* b_tiles,
    const std::uint32_t* sfa,
    const std::uint32_t* sfb,
    std::uint32_t m_tiles,
    std::uint32_t k_tiles,
    float* output) {
    const std::uint32_t m_tile = blockIdx.x;
    if (m_tile >= m_tiles) {
        return;
    }

    const int lane = threadIdx.x;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const std::uint16_t byte_id = 0;
    const std::uint16_t thread_id = 0;

    for (std::uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::uint32_t weight_tile = m_tile * k_tiles + k_tile;
        const uint4 a = a_tiles[weight_tile * kLanes + lane];
        const uint4 b = b_tiles[k_tile * kLanes + lane];
        float n0;
        float n1;
        float n2;
        float n3;

        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "%14, {%15, %16},"
            "%17, {%15, %16};\n"
            : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
              "r"(b.x), "r"(b.y),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa[weight_tile]), "h"(byte_id), "h"(thread_id),
              "r"(sfb[k_tile]));
        d0 = n0;
        d1 = n1;
        d2 = n2;
        d3 = n3;
    }

    if ((lane & 3) == 0) {
        const int row = lane >> 2;
        const int output_base = static_cast<int>(m_tile) * 16;
        output[output_base + row] = d0;
        output[output_base + row + 8] = d2;
    }
}

void run_native_gemv_benchmark(const char* label, int m, int k) {
    const int m_tiles = m / 16;
    const int k_tiles = k / 64;
    const std::size_t weight_lanes =
        static_cast<std::size_t>(m_tiles) * k_tiles * kLanes;
    const std::size_t vector_lanes = static_cast<std::size_t>(k_tiles) * kLanes;
    uint4* weight = nullptr;
    uint4* vector = nullptr;
    std::uint32_t* weight_scales = nullptr;
    std::uint32_t* vector_scales = nullptr;
    float* output = nullptr;
    check(cudaMalloc(&weight, weight_lanes * sizeof(uint4)), "cudaMalloc(GEMV weight)");
    check(cudaMalloc(&vector, vector_lanes * sizeof(uint4)), "cudaMalloc(GEMV vector)");
    check(
        cudaMalloc(
            &weight_scales,
            static_cast<std::size_t>(m_tiles) * k_tiles * sizeof(std::uint32_t)),
        "cudaMalloc(GEMV weight scales)");
    check(
        cudaMalloc(&vector_scales, k_tiles * sizeof(std::uint32_t)),
        "cudaMalloc(GEMV vector scales)");
    check(cudaMalloc(&output, m * sizeof(float)), "cudaMalloc(GEMV output)");
    check(cudaMemset(weight, 0x04, weight_lanes * sizeof(uint4)), "cudaMemset(GEMV weight)");
    check(cudaMemset(vector, 0x04, vector_lanes * sizeof(uint4)), "cudaMemset(GEMV vector)");

    std::vector<std::uint32_t> host_weight_scales(
        static_cast<std::size_t>(m_tiles) * k_tiles,
        kUnitScaleWord);
    std::vector<std::uint32_t> host_vector_scales(k_tiles, kUnitScaleWord);
    check(
        cudaMemcpy(
            weight_scales,
            host_weight_scales.data(),
            host_weight_scales.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(GEMV weight scales)");
    check(
        cudaMemcpy(
            vector_scales,
            host_vector_scales.data(),
            host_vector_scales.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(GEMV vector scales)");

    native_nvfp4_gemv<<<m_tiles, kLanes>>>(
        weight,
        vector,
        weight_scales,
        vector_scales,
        m_tiles,
        k_tiles,
        output);
    check(cudaDeviceSynchronize(), "GEMV correctness launch");
    std::vector<float> host_output(m);
    check(
        cudaMemcpy(
            host_output.data(),
            output,
            host_output.size() * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(GEMV output)");
    const float expected = 128.0f * static_cast<float>(k_tiles);
    for (std::size_t row = 0; row < host_output.size(); ++row) {
        if (host_output[row] != expected) {
            std::fprintf(
                stderr,
                "%s GEMV mismatch: row=%zu expected=%f actual=%f\n",
                label,
                row,
                expected,
                host_output[row]);
            std::exit(EXIT_FAILURE);
        }
    }

    for (int iteration = 0; iteration < kGemvWarmupLaunches; ++iteration) {
        native_nvfp4_gemv<<<m_tiles, kLanes>>>(
            weight,
            vector,
            weight_scales,
            vector_scales,
            m_tiles,
            k_tiles,
            output);
    }
    check(cudaDeviceSynchronize(), "GEMV warmup");
    cudaEvent_t start{};
    cudaEvent_t end{};
    check(cudaEventCreate(&start), "cudaEventCreate(GEMV start)");
    check(cudaEventCreate(&end), "cudaEventCreate(GEMV end)");
    check(cudaEventRecord(start), "cudaEventRecord(GEMV start)");
    for (int iteration = 0; iteration < kGemvTimedLaunches; ++iteration) {
        native_nvfp4_gemv<<<m_tiles, kLanes>>>(
            weight,
            vector,
            weight_scales,
            vector_scales,
            m_tiles,
            k_tiles,
            output);
    }
    check(cudaEventRecord(end), "cudaEventRecord(GEMV end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(GEMV end)");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, end), "cudaEventElapsedTime(GEMV)");
    std::printf(
        "native %s GEMV latency: %.3f us (%d launches)\n",
        label,
        elapsed_ms * 1000.0 / static_cast<double>(kGemvTimedLaunches),
        kGemvTimedLaunches);

    check(cudaEventDestroy(start), "cudaEventDestroy(GEMV start)");
    check(cudaEventDestroy(end), "cudaEventDestroy(GEMV end)");
    check(cudaFree(weight), "cudaFree(GEMV weight)");
    check(cudaFree(vector), "cudaFree(GEMV vector)");
    check(cudaFree(weight_scales), "cudaFree(GEMV weight scales)");
    check(cudaFree(vector_scales), "cudaFree(GEMV vector scales)");
    check(cudaFree(output), "cudaFree(GEMV output)");
}

}  // namespace

int main() {
    check(cudaSetDevice(0), "cudaSetDevice");

    float4* output = nullptr;
    check(cudaMalloc(&output, kKLoopBlocks * kLanes * sizeof(float4)), "cudaMalloc");

    for (int iteration = 0; iteration < kWarmupLaunches; ++iteration) {
        native_nvfp4_mma_probe<<<1, kLanes>>>(kPackedTwos, output);
    }
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t start{};
    cudaEvent_t end{};
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&end), "cudaEventCreate(end)");
    check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < kTimedLaunches; ++iteration) {
        native_nvfp4_mma_probe<<<1, kLanes>>>(kPackedTwos, output);
    }
    check(cudaEventRecord(end), "cudaEventRecord(end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");

    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, end), "cudaEventElapsedTime");

    std::array<float4, kLanes> host{};
    check(
        cudaMemcpy(host.data(), output, kLanes * sizeof(float4), cudaMemcpyDeviceToHost),
        "cudaMemcpy");
    for (int lane = 0; lane < kLanes; ++lane) {
        const float values[4] = {host[lane].x, host[lane].y, host[lane].z, host[lane].w};
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            if (values[accumulator] != 128.0f) {
                std::fprintf(
                    stderr,
                    "output mismatch: accumulator=%d lane=%d expected=128 actual=%f\n",
                    accumulator,
                    lane,
                    values[accumulator]);
                return EXIT_FAILURE;
            }
        }
    }

    std::printf(
        "native launch latency: %.3f us (%d launches)\n",
        elapsed_ms * 1000.0 / static_cast<double>(kTimedLaunches),
        kTimedLaunches);

    uint4* a_tiles = nullptr;
    uint4* b_tiles = nullptr;
    std::uint32_t* sfa = nullptr;
    std::uint32_t* sfb = nullptr;
    const std::size_t tile_lanes = kKLoopTiles * kLanes;
    check(cudaMalloc(&a_tiles, tile_lanes * sizeof(uint4)), "cudaMalloc(a_tiles)");
    check(cudaMalloc(&b_tiles, tile_lanes * sizeof(uint4)), "cudaMalloc(b_tiles)");
    check(cudaMalloc(&sfa, kKLoopTiles * sizeof(std::uint32_t)), "cudaMalloc(sfa)");
    check(cudaMalloc(&sfb, kKLoopTiles * sizeof(std::uint32_t)), "cudaMalloc(sfb)");
    check(cudaMemset(a_tiles, 0x04, tile_lanes * sizeof(uint4)), "cudaMemset(a_tiles)");
    check(cudaMemset(b_tiles, 0x04, tile_lanes * sizeof(uint4)), "cudaMemset(b_tiles)");
    std::array<std::uint32_t, kKLoopTiles> host_scales{};
    host_scales.fill(kUnitScaleWord);
    check(
        cudaMemcpy(
            sfa,
            host_scales.data(),
            host_scales.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(sfa)");
    check(
        cudaMemcpy(
            sfb,
            host_scales.data(),
            host_scales.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(sfb)");

    for (int iteration = 0; iteration < kWarmupLaunches; ++iteration) {
        native_nvfp4_mma_kloop<<<1, kLanes>>>(
            a_tiles, b_tiles, sfa, sfb, kKLoopTiles, output);
    }
    check(cudaDeviceSynchronize(), "K-loop warmup");
    check(cudaEventRecord(start), "cudaEventRecord(K-loop start)");
    for (int iteration = 0; iteration < kTimedLaunches; ++iteration) {
        native_nvfp4_mma_kloop<<<1, kLanes>>>(
            a_tiles, b_tiles, sfa, sfb, kKLoopTiles, output);
    }
    check(cudaEventRecord(end), "cudaEventRecord(K-loop end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(K-loop end)");
    check(cudaEventElapsedTime(&elapsed_ms, start, end), "cudaEventElapsedTime(K-loop)");

    check(
        cudaMemcpy(host.data(), output, kLanes * sizeof(float4), cudaMemcpyDeviceToHost),
        "cudaMemcpy(K-loop output)");
    const float expected_kloop = 128.0f * static_cast<float>(kKLoopTiles);
    for (int lane = 0; lane < kLanes; ++lane) {
        const float values[4] = {host[lane].x, host[lane].y, host[lane].z, host[lane].w};
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            if (values[accumulator] != expected_kloop) {
                std::fprintf(
                    stderr,
                    "K-loop mismatch: accumulator=%d lane=%d expected=%f actual=%f\n",
                    accumulator,
                    lane,
                    expected_kloop,
                    values[accumulator]);
                return EXIT_FAILURE;
            }
        }
    }

    std::printf(
        "native %d-tile K-loop latency: %.3f us (%d launches)\n",
        kKLoopTiles,
        elapsed_ms * 1000.0 / static_cast<double>(kTimedLaunches),
        kTimedLaunches);

    for (int iteration = 0; iteration < kWarmupLaunches; ++iteration) {
        native_nvfp4_mma_kloop<<<kKLoopBlocks, kLanes>>>(
            a_tiles, b_tiles, sfa, sfb, kKLoopTiles, output);
    }
    check(cudaDeviceSynchronize(), "batch K-loop warmup");
    check(cudaEventRecord(start), "cudaEventRecord(batch K-loop start)");
    for (int iteration = 0; iteration < kTimedBatchLaunches; ++iteration) {
        native_nvfp4_mma_kloop<<<kKLoopBlocks, kLanes>>>(
            a_tiles, b_tiles, sfa, sfb, kKLoopTiles, output);
    }
    check(cudaEventRecord(end), "cudaEventRecord(batch K-loop end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(batch K-loop end)");
    check(
        cudaEventElapsedTime(&elapsed_ms, start, end),
        "cudaEventElapsedTime(batch K-loop)");

    std::array<float4, kKLoopBlocks * kLanes> batch_host{};
    check(
        cudaMemcpy(
            batch_host.data(),
            output,
            batch_host.size() * sizeof(float4),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(batch K-loop output)");
    for (std::size_t index = 0; index < batch_host.size(); ++index) {
        const float values[4] = {
            batch_host[index].x,
            batch_host[index].y,
            batch_host[index].z,
            batch_host[index].w};
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
            if (values[accumulator] != expected_kloop) {
                std::fprintf(
                    stderr,
                    "batch K-loop mismatch: fragment=%zu accumulator=%d expected=%f actual=%f\n",
                    index,
                    accumulator,
                    expected_kloop,
                    values[accumulator]);
                return EXIT_FAILURE;
            }
        }
    }

    std::printf(
        "native %d-warp by %d-tile latency: %.3f us (%d launches)\n",
        kKLoopBlocks,
        kKLoopTiles,
        elapsed_ms * 1000.0 / static_cast<double>(kTimedBatchLaunches),
        kTimedBatchLaunches);

    check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check(cudaEventDestroy(end), "cudaEventDestroy(end)");
    check(cudaFree(a_tiles), "cudaFree(a_tiles)");
    check(cudaFree(b_tiles), "cudaFree(b_tiles)");
    check(cudaFree(sfa), "cudaFree(sfa)");
    check(cudaFree(sfb), "cudaFree(sfb)");
    check(cudaFree(output), "cudaFree");
    run_native_gemv_benchmark("Qwen gate+up 34816x5120", kQwenGateUp, kQwenHidden);
    run_native_gemv_benchmark("Qwen down 5120x17408", kQwenHidden, kQwenIntermediate);
    return EXIT_SUCCESS;
}
