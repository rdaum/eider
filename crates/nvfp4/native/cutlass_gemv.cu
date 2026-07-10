#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

__device__ __forceinline__ float infer_e2m1_value_lut(std::uint8_t nibble) {
    const float magnitude = (nibble & 0x7) == 0x0 ? 0.0f :
                            (nibble & 0x7) == 0x1 ? 0.5f :
                            (nibble & 0x7) == 0x2 ? 1.0f :
                            (nibble & 0x7) == 0x3 ? 1.5f :
                            (nibble & 0x7) == 0x4 ? 2.0f :
                            (nibble & 0x7) == 0x5 ? 3.0f :
                            (nibble & 0x7) == 0x6 ? 4.0f : 6.0f;
    return (nibble & 0x8) ? -magnitude : magnitude;
}

__device__ __forceinline__ float infer_e4m3_value_lut(std::uint8_t code) {
    if ((code & 0x80) == 0) {
        const std::uint8_t exp = (code >> 3) & 0x0f;
        const std::uint8_t mant = code & 0x07;
        if (exp == 0) {
            return static_cast<float>(mant) * 0.001953125f;
        }
        if (exp == 0x0f && mant == 0x07) {
            return NAN;
        }
        return __uint_as_float((static_cast<std::uint32_t>(exp) + 120u) << 23 |
                               static_cast<std::uint32_t>(mant) << 20);
    }
    const float sign = -1.0f;
    const std::uint8_t exp = (code >> 3) & 0x0f;
    const std::uint8_t mant = code & 0x07;
    if (exp == 0) {
        return sign * static_cast<float>(mant) * exp2f(-9.0f);
    }
    if (exp == 0x0f && mant == 0x07) {
        return NAN;
    }
    return sign * (1.0f + static_cast<float>(mant) / 8.0f) *
           exp2f(static_cast<float>(exp) - 7.0f);
}

template <int kVectorSize_, typename ThreadShape_, typename ElementCompute_, typename ElementAccumulator_, typename ElementC_, typename ElementD_, typename LayoutOutput_>
class SparkInferGemvF32Epilogue {
public:
    using ThreadShape = ThreadShape_;
    using ElementCompute = ElementCompute_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementC = ElementC_;
    using ElementD = ElementD_;
    using LayoutOutput = LayoutOutput_;
    using TensorRefD = cutlass::TensorRef<ElementD, LayoutOutput>;
    static constexpr int kVectorSize = kVectorSize_;
    static constexpr int kThreadsPerCol = ThreadShape::kM;
    static constexpr int kThreadsPerRow = ThreadShape::kN;
    static constexpr int kThreadCount = kThreadsPerCol * kThreadsPerRow;

    struct Params {
        TensorRefD tensor_d;
        ElementCompute alpha{1};
        ElementCompute beta{0};
        int64_t stride_d{0};
    };

    struct SharedStorage {
        int unused;
    };

    CUTLASS_HOST_DEVICE SparkInferGemvF32Epilogue(Params const& params, SharedStorage&) : params_(params) {}

    CUTLASS_DEVICE void operator()(ElementAccumulator frag_acc, ElementC frag_c, int batch_idx) {
        if (threadIdx.x == 0) {
            int row = blockIdx.x * blockDim.y + threadIdx.y;
            int offset = row + batch_idx * params_.stride_d;
            float value = static_cast<float>(params_.alpha) * static_cast<float>(frag_acc) +
                          static_cast<float>(params_.beta) * static_cast<float>(frag_c);
            params_.tensor_d.at({offset, 0}) = static_cast<ElementD>(value);
        }
    }

private:
    Params const& params_;
};

extern "C" int infer_cutlass_fp4_gemv_f32_supported(std::uint32_t m, std::uint32_t k) {
    return m > 0 && k > 0 && (k % 32) == 0;
}

extern "C" cudaError_t infer_cutlass_fp4_gemv_f32_on_stream(const std::uint8_t* a_values,
                                                                   const std::uint8_t* a_scales,
                                                                   const std::uint8_t* b_values,
                                                                   const std::uint8_t* b_scales,
                                                                   const float* c,
                                                                   float* d,
                                                                   std::uint32_t m,
                                                                   std::uint32_t k,
                                                                   float alpha,
                                                                   cudaStream_t stream) {
    if (a_values == nullptr || a_scales == nullptr || b_values == nullptr || b_scales == nullptr ||
        c == nullptr || d == nullptr || !infer_cutlass_fp4_gemv_f32_supported(m, k)) {
        return cudaErrorInvalidValue;
    }

    using ElementA = cutlass::float_e2m1_t;
    using ElementB = cutlass::float_e2m1_t;
    using ElementC = float;
    using ElementD = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using ElementAccumulatorMainloop = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    static constexpr int kVectorSize = 16;
    static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;
    using ThreadShape = cutlass::gemm::GemmShape<16, 8>;
    using EpilogueOp = SparkInferGemvF32Epilogue<kVectorSize, ThreadShape, ElementCompute, ElementAccumulator, ElementC, ElementD, LayoutC>;
    using Gemv = cutlass::gemm::device::GemvBlockScaled<
        cutlass::gemm::kernel::GemvBlockScaled<ElementA, LayoutA, ElementB, ElementD, ElementAccumulatorMainloop, EpilogueOp, kElementsPerAccess,
        0, 0,
        cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, 16>>;

    typename EpilogueOp::Params epilogue{
        cutlass::TensorRef<ElementD, LayoutC>(d, LayoutC::packed({static_cast<int>(m), 1})),
        alpha,
        0.0f,
        static_cast<int64_t>(m),
    };

    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord{static_cast<int>(m), static_cast<int>(k)},
        1,
        epilogue,
        cutlass::TensorRef<ElementA, LayoutA>(reinterpret_cast<ElementA*>(const_cast<std::uint8_t*>(a_values)), LayoutA::packed({static_cast<int>(m), static_cast<int>(k)})),
        reinterpret_cast<ElementB const*>(b_values),
        c,
        d,
        reinterpret_cast<cutlass::float_ue4m3_t const*>(a_scales),
        reinterpret_cast<cutlass::float_ue4m3_t const*>(b_scales),
        static_cast<int64_t>(k),
        static_cast<int64_t>(m) * static_cast<int64_t>(k),
        static_cast<int64_t>(k),
        static_cast<int64_t>(m),
        static_cast<int64_t>(m),
        0,
        0,
        0,
    };

    Gemv gemv_op;
    cutlass::Status status = gemv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }
    status = gemv_op.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }
    status = gemv_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaGetLastError();
    }
    return cudaGetLastError();
}

// ============================================================================
// Grouped FP4 blockscaled GEMV — SIMT dequantization path
// ============================================================================
// Dequantizes FP4 (E2M1) weights and scales to f32, then accumulates the
// dot product in f32. Each thread handles one output row (M dimension).
//
// Weight A is stored as packed E2M1 nibbles in column-major [K, M] order:
//   element A[k][m] at flat index m*K + k, packed at byte (m*K + k) / 2.
// SFA is raw ModelOpt [M, K/16] row-major UE4M3 scales.
// SFB is [K/16] (per-K-block UE4M3 scales, simple row-major).
// B (activation vector) is packed E2M1 nibbles: element B[k] at byte k / 2.
//
// Thread block: 128 threads, 1-D. Each thread computes one M row.
// Grid: (ceil(M/128), 1, groups).

namespace infer_grouped_fp4 {

static constexpr int kSFVecSize = 16;
static constexpr int kPackedElements = 2;
static constexpr int kThreadCount = 128;
static constexpr int kParallelKThreadCount = 64;
static constexpr int kRowsPerSmallKBlock = 4;
static constexpr int kThreadsPerSmallKRow = 32;

struct GroupedGemvPlan {
    std::uint32_t m = 0;
    std::uint32_t k = 0;
    std::uint32_t groups = 0;
};

} // namespace infer_grouped_fp4

__global__ void infer_grouped_fp4_gemv_f32_parallel_k_kernel(
    const std::uint8_t* const* __restrict__ a_values,
    const std::uint8_t* const* __restrict__ a_scales,
    const std::uint8_t* const* __restrict__ b_values,
    const std::uint8_t* const* __restrict__ b_scales,
    float* const* __restrict__ d,
    float alpha,
    float beta,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t groups)
{
    using namespace infer_grouped_fp4;

    const std::uint32_t row = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (row >= m || group >= groups) return;

    const std::uint8_t* ptr_A = a_values[group];
    const std::uint8_t* ptr_SFA = a_scales[group];
    const std::uint8_t* ptr_B = b_values[group];
    const std::uint8_t* ptr_SFB = b_scales[group];
    float* ptr_D = d[group];

    const int n_k_blocks = static_cast<int>(k) / kSFVecSize;
    const int64_t a_row_base = static_cast<int64_t>(row) * k;
    float partial = 0.0f;

    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        const int k_start = kb * kSFVecSize;
        const float sfa = infer_e4m3_value_lut(ptr_SFA[row * n_k_blocks + kb]);
        const float sfb = infer_e4m3_value_lut(ptr_SFB[kb]);
        const float scale = sfa * sfb;

        #pragma unroll
        for (int packed = 0; packed < kSFVecSize / kPackedElements; ++packed) {
            const int kk = k_start + packed * kPackedElements;
            const std::uint8_t a_byte = ptr_A[(a_row_base + kk) / kPackedElements];
            const std::uint8_t b_byte = ptr_B[kk / kPackedElements];
            partial += infer_e2m1_value_lut(a_byte & 0xF) *
                       infer_e2m1_value_lut(b_byte & 0xF) * scale;
            partial += infer_e2m1_value_lut((a_byte >> 4) & 0xF) *
                       infer_e2m1_value_lut((b_byte >> 4) & 0xF) * scale;
        }
    }

    __shared__ float scratch[kParallelKThreadCount];
    scratch[threadIdx.x] = partial;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ptr_D[row] = alpha * scratch[0] + beta * 0.0f;
    }
}

__global__ void infer_grouped_fp4_gemv_f32_indexed_a_parallel_k_kernel(
    const std::uint32_t* __restrict__ indices,
    const std::uint8_t* const* __restrict__ a_values_table,
    const std::uint8_t* const* __restrict__ a_scales_table,
    const std::uint8_t* __restrict__ b_values,
    const std::uint8_t* __restrict__ b_scales,
    float* const* __restrict__ d,
    float alpha,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t groups,
    std::uint32_t table_len)
{
    using namespace infer_grouped_fp4;

    const std::uint32_t row = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (row >= m || group >= groups) return;
    const std::uint32_t expert = indices[group];
    if (expert >= table_len) return;

    const std::uint8_t* ptr_A = a_values_table[expert];
    const std::uint8_t* ptr_SFA = a_scales_table[expert];
    float* ptr_D = d[group];

    const int n_k_blocks = static_cast<int>(k) / kSFVecSize;
    const int64_t a_row_base = static_cast<int64_t>(row) * k;
    float partial = 0.0f;

    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        const int k_start = kb * kSFVecSize;
        const float sfa = infer_e4m3_value_lut(ptr_SFA[row * n_k_blocks + kb]);
        const float sfb = infer_e4m3_value_lut(b_scales[kb]);
        const float scale = sfa * sfb;

        #pragma unroll
        for (int packed = 0; packed < kSFVecSize / kPackedElements; ++packed) {
            const int kk = k_start + packed * kPackedElements;
            const std::uint8_t a_byte = ptr_A[(a_row_base + kk) / kPackedElements];
            const std::uint8_t b_byte = b_values[kk / kPackedElements];
            partial += infer_e2m1_value_lut(a_byte & 0xF) *
                       infer_e2m1_value_lut(b_byte & 0xF) * scale;
            partial += infer_e2m1_value_lut((a_byte >> 4) & 0xF) *
                       infer_e2m1_value_lut((b_byte >> 4) & 0xF) * scale;
        }
    }

    __shared__ float scratch[kParallelKThreadCount];
    scratch[threadIdx.x] = partial;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ptr_D[row] = alpha * scratch[0];
    }
}

__global__ void infer_grouped_fp4_gemv_f32_contiguous_b_parallel_k_kernel(
    const std::uint8_t* const* __restrict__ a_values,
    const std::uint8_t* const* __restrict__ a_scales,
    const std::uint8_t* __restrict__ b_values,
    const std::uint8_t* __restrict__ b_scales,
    float* __restrict__ d,
    float alpha,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t groups)
{
    using namespace infer_grouped_fp4;

    const std::uint32_t row = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (row >= m || group >= groups) return;

    const std::uint8_t* ptr_A = a_values[group];
    const std::uint8_t* ptr_SFA = a_scales[group];
    const std::uint8_t* ptr_B = b_values + static_cast<std::uint64_t>(group) * (k / kPackedElements);
    const std::uint8_t* ptr_SFB = b_scales + static_cast<std::uint64_t>(group) * (k / kSFVecSize);
    float* ptr_D = d + static_cast<std::uint64_t>(group) * m;

    const int n_k_blocks = static_cast<int>(k) / kSFVecSize;
    const int64_t a_row_base = static_cast<int64_t>(row) * k;
    float partial = 0.0f;

    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        const int k_start = kb * kSFVecSize;
        const float sfa = infer_e4m3_value_lut(ptr_SFA[row * n_k_blocks + kb]);
        const float sfb = infer_e4m3_value_lut(ptr_SFB[kb]);
        const float scale = sfa * sfb;

        #pragma unroll
        for (int packed = 0; packed < kSFVecSize / kPackedElements; ++packed) {
            const int kk = k_start + packed * kPackedElements;
            const std::uint8_t a_byte = ptr_A[(a_row_base + kk) / kPackedElements];
            const std::uint8_t b_byte = ptr_B[kk / kPackedElements];
            partial += infer_e2m1_value_lut(a_byte & 0xF) *
                       infer_e2m1_value_lut(b_byte & 0xF) * scale;
            partial += infer_e2m1_value_lut((a_byte >> 4) & 0xF) *
                       infer_e2m1_value_lut((b_byte >> 4) & 0xF) * scale;
        }
    }

    __shared__ float scratch[kParallelKThreadCount];
    scratch[threadIdx.x] = partial;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ptr_D[row] = alpha * scratch[0];
    }
}

__global__ void infer_grouped_fp4_gemv_f32_small_k_kernel(
    const std::uint8_t* const* __restrict__ a_values,
    const std::uint8_t* const* __restrict__ a_scales,
    const std::uint8_t* const* __restrict__ b_values,
    const std::uint8_t* const* __restrict__ b_scales,
    float* const* __restrict__ d,
    float alpha,
    float beta,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t groups)
{
    using namespace infer_grouped_fp4;

    const std::uint32_t group = blockIdx.y;
    const int local_row = threadIdx.x / kThreadsPerSmallKRow;
    const int lane = threadIdx.x - local_row * kThreadsPerSmallKRow;
    const std::uint32_t row = blockIdx.x * kRowsPerSmallKBlock + local_row;
    if (row >= m || group >= groups) return;

    const std::uint8_t* ptr_A = a_values[group];
    const std::uint8_t* ptr_SFA = a_scales[group];
    const std::uint8_t* ptr_B = b_values[group];
    const std::uint8_t* ptr_SFB = b_scales[group];
    float* ptr_D = d[group];

    const int n_k_blocks = static_cast<int>(k) / kSFVecSize;
    const int64_t a_row_base = static_cast<int64_t>(row) * k;
    float partial = 0.0f;

    for (int kb = lane; kb < n_k_blocks; kb += kThreadsPerSmallKRow) {
        const int k_start = kb * kSFVecSize;
        const float sfa = infer_e4m3_value_lut(ptr_SFA[row * n_k_blocks + kb]);
        const float sfb = infer_e4m3_value_lut(ptr_SFB[kb]);
        const float scale = sfa * sfb;

        #pragma unroll
        for (int packed = 0; packed < kSFVecSize / kPackedElements; ++packed) {
            const int kk = k_start + packed * kPackedElements;
            const std::uint8_t a_byte = ptr_A[(a_row_base + kk) / kPackedElements];
            const std::uint8_t b_byte = ptr_B[kk / kPackedElements];
            partial += infer_e2m1_value_lut(a_byte & 0xF) *
                       infer_e2m1_value_lut(b_byte & 0xF) * scale;
            partial += infer_e2m1_value_lut((a_byte >> 4) & 0xF) *
                       infer_e2m1_value_lut((b_byte >> 4) & 0xF) * scale;
        }
    }

    __shared__ float scratch[kThreadCount];
    scratch[threadIdx.x] = partial;
    __syncthreads();

    const int row_base_thread = local_row * kThreadsPerSmallKRow;
    for (int offset = kThreadsPerSmallKRow / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            scratch[row_base_thread + lane] += scratch[row_base_thread + lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0) {
        ptr_D[row] = alpha * scratch[row_base_thread] + beta * 0.0f;
    }
}

__global__ void infer_grouped_fp4_gemv_f32_contiguous_b_small_k_kernel(
    const std::uint8_t* const* __restrict__ a_values,
    const std::uint8_t* const* __restrict__ a_scales,
    const std::uint8_t* __restrict__ b_values,
    const std::uint8_t* __restrict__ b_scales,
    float* __restrict__ d,
    float alpha,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t groups)
{
    using namespace infer_grouped_fp4;

    const std::uint32_t group = blockIdx.y;
    const int local_row = threadIdx.x / kThreadsPerSmallKRow;
    const int lane = threadIdx.x - local_row * kThreadsPerSmallKRow;
    const std::uint32_t row = blockIdx.x * kRowsPerSmallKBlock + local_row;
    if (row >= m || group >= groups) return;

    const std::uint8_t* ptr_A = a_values[group];
    const std::uint8_t* ptr_SFA = a_scales[group];
    const std::uint8_t* ptr_B = b_values + static_cast<std::uint64_t>(group) * (k / kPackedElements);
    const std::uint8_t* ptr_SFB = b_scales + static_cast<std::uint64_t>(group) * (k / kSFVecSize);
    float* ptr_D = d + static_cast<std::uint64_t>(group) * m;

    const int n_k_blocks = static_cast<int>(k) / kSFVecSize;
    const int64_t a_row_base = static_cast<int64_t>(row) * k;
    float partial = 0.0f;

    for (int kb = lane; kb < n_k_blocks; kb += kThreadsPerSmallKRow) {
        const int k_start = kb * kSFVecSize;
        const float sfa = infer_e4m3_value_lut(ptr_SFA[row * n_k_blocks + kb]);
        const float sfb = infer_e4m3_value_lut(ptr_SFB[kb]);
        const float scale = sfa * sfb;

        #pragma unroll
        for (int packed = 0; packed < kSFVecSize / kPackedElements; ++packed) {
            const int kk = k_start + packed * kPackedElements;
            const std::uint8_t a_byte = ptr_A[(a_row_base + kk) / kPackedElements];
            const std::uint8_t b_byte = ptr_B[kk / kPackedElements];
            partial += infer_e2m1_value_lut(a_byte & 0xF) *
                       infer_e2m1_value_lut(b_byte & 0xF) * scale;
            partial += infer_e2m1_value_lut((a_byte >> 4) & 0xF) *
                       infer_e2m1_value_lut((b_byte >> 4) & 0xF) * scale;
        }
    }

    __shared__ float scratch[kThreadCount];
    scratch[threadIdx.x] = partial;
    __syncthreads();

    const int row_base_thread = local_row * kThreadsPerSmallKRow;
    for (int offset = kThreadsPerSmallKRow / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            scratch[row_base_thread + lane] += scratch[row_base_thread + lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0) {
        ptr_D[row] = alpha * scratch[row_base_thread];
    }
}

__global__ void infer_grouped_fp4_gemv_f32_kernel(
    const std::uint8_t* const* __restrict__ a_values,
    const std::uint8_t* const* __restrict__ a_scales,
    const std::uint8_t* const* __restrict__ b_values,
    const std::uint8_t* const* __restrict__ b_scales,
    float* const* __restrict__ d,
    float alpha,
    float beta,
    std::uint32_t m,
    std::uint32_t k,
    std::uint32_t groups)
{
    using namespace infer_grouped_fp4;

    const std::uint32_t group = blockIdx.z;
    if (group >= groups) return;

    const int row = blockIdx.x * kThreadCount + threadIdx.x;
    if (row >= static_cast<int>(m)) return;

    const std::uint8_t* ptr_A = a_values[group];
    const std::uint8_t* ptr_SFA = a_scales[group];
    const std::uint8_t* ptr_B = b_values[group];
    const std::uint8_t* ptr_SFB = b_scales[group];
    float* ptr_D = d[group];

    const int n_k_blocks = static_cast<int>(k) / kSFVecSize;
    const int64_t a_row_base = static_cast<int64_t>(row) * k;

    float accum = 0.0f;

    for (int kb = 0; kb < n_k_blocks; ++kb) {
        const int k_start = kb * kSFVecSize;
        const float sfa = infer_e4m3_value_lut(ptr_SFA[row * n_k_blocks + kb]);
        const float sfb = infer_e4m3_value_lut(ptr_SFB[kb]);
        const float scale = sfa * sfb;

        #pragma unroll
        for (int ki = 0; ki < kSFVecSize; ++ki) {
            const int kk = k_start + ki;
            const int a_flat = static_cast<int>(a_row_base + kk);
            const std::uint8_t a_byte = ptr_A[a_flat / 2];
            const std::uint8_t a_nibble = (a_flat & 1)
                ? ((a_byte >> 4) & 0xF)
                : (a_byte & 0xF);

            const std::uint8_t b_byte = ptr_B[kk / 2];
            const std::uint8_t b_nibble = (kk & 1)
                ? ((b_byte >> 4) & 0xF)
                : (b_byte & 0xF);

            accum += infer_e2m1_value_lut(a_nibble) * infer_e2m1_value_lut(b_nibble) * scale;
        }
    }

    ptr_D[row] = alpha * accum + beta * 0.0f;
}

// ============================================================================
// C ABI wrappers
// ============================================================================

extern "C" int infer_cutlass_fp4_grouped_gemv_f32_supported(std::uint32_t m,
                                                                   std::uint32_t k,
                                                                   std::uint32_t groups) {
    return m > 0 && k > 0 && groups > 0 && (k % infer_grouped_fp4::kSFVecSize) == 0;
}

extern "C" void* infer_cutlass_fp4_grouped_gemv_f32_create(std::uint32_t m,
                                                                  std::uint32_t k,
                                                                  std::uint32_t groups) {
    if (!infer_cutlass_fp4_grouped_gemv_f32_supported(m, k, groups)) {
        return nullptr;
    }
    auto* plan = new infer_grouped_fp4::GroupedGemvPlan();
    plan->m = m;
    plan->k = k;
    plan->groups = groups;
    return plan;
}

extern "C" void infer_cutlass_fp4_grouped_gemv_f32_destroy(void* raw_plan) {
    delete reinterpret_cast<infer_grouped_fp4::GroupedGemvPlan*>(raw_plan);
}

extern "C" cudaError_t infer_cutlass_fp4_grouped_gemv_f32_on_stream(
    void* raw_plan,
    const std::uint8_t* const* a_values,
    const std::uint8_t* const* a_scales,
    const std::uint8_t* const* b_values,
    const std::uint8_t* const* b_scales,
    const float* const* c,
    float* const* d,
    float alpha,
    float beta,
    cudaStream_t stream) {
    using namespace infer_grouped_fp4;

    if (raw_plan == nullptr || a_values == nullptr || a_scales == nullptr ||
        b_values == nullptr || b_scales == nullptr || d == nullptr) {
        return cudaErrorInvalidValue;
    }
    (void)c;

    auto* plan = reinterpret_cast<GroupedGemvPlan*>(raw_plan);

    if (plan->k <= 512) {
        dim3 grid((plan->m + kRowsPerSmallKBlock - 1) / kRowsPerSmallKBlock, plan->groups, 1);
        dim3 block(kThreadCount, 1, 1);
        infer_grouped_fp4_gemv_f32_small_k_kernel<<<grid, block, 0, stream>>>(
            a_values,
            a_scales,
            b_values,
            b_scales,
            d,
            alpha,
            beta,
            plan->m,
            plan->k,
            plan->groups);
    } else if (plan->k >= 1024) {
        dim3 grid(plan->m, plan->groups, 1);
        dim3 block(kParallelKThreadCount, 1, 1);
        infer_grouped_fp4_gemv_f32_parallel_k_kernel<<<grid, block, 0, stream>>>(
            a_values,
            a_scales,
            b_values,
            b_scales,
            d,
            alpha,
            beta,
            plan->m,
            plan->k,
            plan->groups);
    } else {
        dim3 grid((plan->m + kThreadCount - 1) / kThreadCount, 1, plan->groups);
        dim3 block(kThreadCount, 1, 1);
        infer_grouped_fp4_gemv_f32_kernel<<<grid, block, 0, stream>>>(
            a_values,
            a_scales,
            b_values,
            b_scales,
            d,
            alpha,
            beta,
            plan->m,
            plan->k,
            plan->groups);
    }

    return cudaGetLastError();
}

extern "C" cudaError_t infer_cutlass_fp4_grouped_gemv_f32_indexed_a_on_stream(
    void* raw_plan,
    const std::uint32_t* indices,
    const std::uint8_t* const* a_values_table,
    const std::uint8_t* const* a_scales_table,
    std::uint32_t table_len,
    const std::uint8_t* b_values,
    const std::uint8_t* b_scales,
    float* const* d,
    float alpha,
    cudaStream_t stream) {
    using namespace infer_grouped_fp4;

    if (raw_plan == nullptr || indices == nullptr || a_values_table == nullptr ||
        a_scales_table == nullptr || b_values == nullptr || b_scales == nullptr || d == nullptr ||
        table_len == 0) {
        return cudaErrorInvalidValue;
    }

    auto* plan = reinterpret_cast<GroupedGemvPlan*>(raw_plan);
    if (plan->k < 1024) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(plan->m, plan->groups, 1);
    dim3 block(kParallelKThreadCount, 1, 1);
    infer_grouped_fp4_gemv_f32_indexed_a_parallel_k_kernel<<<grid, block, 0, stream>>>(
        indices,
        a_values_table,
        a_scales_table,
        b_values,
        b_scales,
        d,
        alpha,
        plan->m,
        plan->k,
        plan->groups,
        table_len);

    return cudaGetLastError();
}

extern "C" cudaError_t infer_cutlass_fp4_grouped_gemv_f32_contiguous_b_on_stream(
    void* raw_plan,
    const std::uint8_t* const* a_values,
    const std::uint8_t* const* a_scales,
    const std::uint8_t* b_values,
    const std::uint8_t* b_scales,
    float* d,
    float alpha,
    cudaStream_t stream) {
    using namespace infer_grouped_fp4;

    if (raw_plan == nullptr || a_values == nullptr || a_scales == nullptr ||
        b_values == nullptr || b_scales == nullptr || d == nullptr) {
        return cudaErrorInvalidValue;
    }

    auto* plan = reinterpret_cast<GroupedGemvPlan*>(raw_plan);
    if (plan->k <= 512) {
        dim3 grid((plan->m + kRowsPerSmallKBlock - 1) / kRowsPerSmallKBlock, plan->groups, 1);
        dim3 block(kThreadCount, 1, 1);
        infer_grouped_fp4_gemv_f32_contiguous_b_small_k_kernel<<<grid, block, 0, stream>>>(
            a_values,
            a_scales,
            b_values,
            b_scales,
            d,
            alpha,
            plan->m,
            plan->k,
            plan->groups);
    } else {
        dim3 grid(plan->m, plan->groups, 1);
        dim3 block(kParallelKThreadCount, 1, 1);
        infer_grouped_fp4_gemv_f32_contiguous_b_parallel_k_kernel<<<grid, block, 0, stream>>>(
            a_values,
            a_scales,
            b_values,
            b_scales,
            d,
            alpha,
            plan->m,
            plan->k,
            plan->groups);
    }

    return cudaGetLastError();
}
