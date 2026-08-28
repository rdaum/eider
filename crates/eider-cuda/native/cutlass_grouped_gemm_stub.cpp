#include <cuda_runtime.h>

#include <cstdint>

extern "C" int infer_cutlass_fp4_grouped_gemm_supported(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    return 0;
}

extern "C" void* infer_cutlass_fp4_grouped_gemm_create(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    return nullptr;
}

extern "C" void infer_cutlass_fp4_grouped_gemm_destroy(void*) {}

extern "C" cudaError_t infer_cutlass_fp4_grouped_gemm_on_stream(
    void*, const std::uint8_t* const*, const std::uint8_t* const*,
    const std::uint8_t* const*, const std::uint8_t* const*, float* const*, float* const*,
    const std::uint32_t*, cudaStream_t) {
    return cudaErrorNotSupported;
}
