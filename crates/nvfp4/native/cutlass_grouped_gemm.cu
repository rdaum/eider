#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace infer_grouped_fp4_gemm {

using namespace cute;

using UnderlyingProblemShape = Shape<int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<UnderlyingProblemShape>;
using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementSF = cutlass::float_ue4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using ClusterShape = Shape<_1, _1, _1>;
using MmaTileShape = Shape<_128, _128, _256>;

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutC*,
    kAlignmentC,
    ElementD,
    LayoutC*,
    kAlignmentD,
    EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA,
    LayoutA*,
    kAlignmentA,
    ElementB,
    LayoutB*,
    kAlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideC = typename GemmKernel::InternalStrideC;
using StrideD = typename GemmKernel::InternalStrideD;
using LayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
using BlockScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

struct Plan {
    int m;
    int max_n;
    int k;
    int groups;
    UnderlyingProblemShape* problem_sizes = nullptr;
    StrideA* stride_a = nullptr;
    StrideB* stride_b = nullptr;
    StrideC* stride_c = nullptr;
    StrideD* stride_d = nullptr;
    LayoutSFA* layout_sfa = nullptr;
    LayoutSFB* layout_sfb = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;

    ~Plan() {
        cudaFree(problem_sizes);
        cudaFree(stride_a);
        cudaFree(stride_b);
        cudaFree(stride_c);
        cudaFree(stride_d);
        cudaFree(layout_sfa);
        cudaFree(layout_sfb);
        cudaFree(workspace);
    }
};

__global__ void prepare_metadata_kernel(
    int m,
    int max_n,
    int k,
    int groups,
    std::uint32_t const* tokens_per_expert,
    UnderlyingProblemShape* problem_sizes,
    StrideA* stride_a,
    StrideB* stride_b,
    StrideC* stride_c,
    StrideD* stride_d,
    LayoutSFA* layout_sfa,
    LayoutSFB* layout_sfb) {
    const int group = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (group >= groups) return;
    const int n = min(static_cast<int>(tokens_per_expert[group]), max_n);
    problem_sizes[group] = {m, n, k};
    stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    stride_c[group] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    layout_sfa[group] = BlockScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n, k, 1));
    layout_sfb[group] = BlockScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m, n, k, 1));
}

Gemm::Arguments make_arguments(
    Plan const& plan,
    std::uint8_t const** a_values,
    std::uint8_t const** a_scales,
    std::uint8_t const** b_values,
    std::uint8_t const** b_scales,
    std::uint16_t** output,
    float** alpha) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion;
    fusion.alpha = 0.0f;
    fusion.alpha_ptr = nullptr;
    fusion.alpha_ptr_array = alpha;
    fusion.dAlpha = {_0{}, _0{}, 1};
    fusion.beta = 0.0f;
    fusion.beta_ptr = nullptr;
    fusion.beta_ptr_array = nullptr;
    fusion.dBeta = {_0{}, _0{}, 0};

    typename GemmKernel::TileSchedulerArguments scheduler;
    return Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {plan.groups, plan.problem_sizes, nullptr},
        {reinterpret_cast<ElementA::DataType const**>(a_values), plan.stride_a,
         reinterpret_cast<ElementB::DataType const**>(b_values), plan.stride_b,
         reinterpret_cast<ElementSF const**>(a_scales), plan.layout_sfa,
         reinterpret_cast<ElementSF const**>(b_scales), plan.layout_sfb},
        {fusion,
         reinterpret_cast<ElementC const**>(static_cast<void*>(output)), plan.stride_c,
         reinterpret_cast<ElementD**>(output), plan.stride_d},
        hw_info,
        scheduler};
}

cudaError_t run(
    Plan& plan,
    std::uint8_t const** a_values,
    std::uint8_t const** a_scales,
    std::uint8_t const** b_values,
    std::uint8_t const** b_scales,
    std::uint16_t** output,
    float** alpha,
    std::uint32_t const* tokens_per_expert,
    cudaStream_t stream) {
    constexpr int kThreads = 128;
    prepare_metadata_kernel<<<(plan.groups + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        plan.m, plan.max_n, plan.k, plan.groups, tokens_per_expert,
        plan.problem_sizes, plan.stride_a, plan.stride_b, plan.stride_c,
        plan.stride_d, plan.layout_sfa, plan.layout_sfb);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    auto arguments = make_arguments(
        plan, a_values, a_scales, b_values, b_scales, output, alpha);
    const std::size_t required = Gemm::get_workspace_size(arguments);
    if (required > plan.workspace_bytes) {
        if (plan.workspace != nullptr) {
            status = cudaFree(plan.workspace);
            if (status != cudaSuccess) return status;
            plan.workspace = nullptr;
            plan.workspace_bytes = 0;
        }
        if (required != 0) {
            status = cudaMalloc(&plan.workspace, required);
            if (status != cudaSuccess) return status;
            plan.workspace_bytes = required;
        }
    }
    if (Gemm::can_implement(arguments) != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    Gemm gemm;
    const cutlass::Status initialize_status = gemm.initialize(arguments, plan.workspace, stream);
    if (initialize_status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidConfiguration;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }
    return cudaGetLastError();
}

template <class T>
cudaError_t allocate_array(T** pointer, int count) {
    return cudaMalloc(reinterpret_cast<void**>(pointer), sizeof(T) * count);
}

}  // namespace infer_grouped_fp4_gemm

extern "C" int infer_cutlass_fp4_grouped_gemm_supported(
    std::uint32_t m,
    std::uint32_t max_n,
    std::uint32_t k,
    std::uint32_t groups) {
    return m > 0 && max_n > 0 && k > 0 && groups > 0 &&
           (m % 16) == 0 && (k % 64) == 0;
}

extern "C" void* infer_cutlass_fp4_grouped_gemm_create(
    std::uint32_t m,
    std::uint32_t max_n,
    std::uint32_t k,
    std::uint32_t groups) {
    if (!infer_cutlass_fp4_grouped_gemm_supported(m, max_n, k, groups)) return nullptr;
    using namespace infer_grouped_fp4_gemm;
    Plan* plan = new Plan{
        static_cast<int>(m), static_cast<int>(max_n), static_cast<int>(k),
        static_cast<int>(groups)};
    cudaError_t status = allocate_array(&plan->problem_sizes, plan->groups);
    if (status == cudaSuccess) status = allocate_array(&plan->stride_a, plan->groups);
    if (status == cudaSuccess) status = allocate_array(&plan->stride_b, plan->groups);
    if (status == cudaSuccess) status = allocate_array(&plan->stride_c, plan->groups);
    if (status == cudaSuccess) status = allocate_array(&plan->stride_d, plan->groups);
    if (status == cudaSuccess) status = allocate_array(&plan->layout_sfa, plan->groups);
    if (status == cudaSuccess) status = allocate_array(&plan->layout_sfb, plan->groups);
    if (status != cudaSuccess) {
        delete plan;
        return nullptr;
    }
    return plan;
}

extern "C" void infer_cutlass_fp4_grouped_gemm_destroy(void* raw_plan) {
    delete reinterpret_cast<infer_grouped_fp4_gemm::Plan*>(raw_plan);
}

extern "C" cudaError_t infer_cutlass_fp4_grouped_gemm_on_stream(
    void* raw_plan,
    std::uint8_t const** a_values,
    std::uint8_t const** a_scales,
    std::uint8_t const** b_values,
    std::uint8_t const** b_scales,
    std::uint16_t** output,
    float** alpha,
    std::uint32_t const* tokens_per_expert,
    cudaStream_t stream) {
    if (raw_plan == nullptr || a_values == nullptr || a_scales == nullptr ||
        b_values == nullptr || b_scales == nullptr || output == nullptr ||
        alpha == nullptr || tokens_per_expert == nullptr) {
        return cudaErrorInvalidValue;
    }
    return infer_grouped_fp4_gemm::run(
        *reinterpret_cast<infer_grouped_fp4_gemm::Plan*>(raw_plan),
        a_values, a_scales, b_values, b_scales, output, alpha,
        tokens_per_expert, stream);
}
