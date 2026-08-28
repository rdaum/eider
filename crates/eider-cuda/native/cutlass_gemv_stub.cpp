#include <cuda_runtime.h>

#include <cstdint>

extern "C" int infer_cutlass_fp4_gemv_f32_supported(std::uint32_t, std::uint32_t) {
    return 0;
}

extern "C" cudaError_t infer_cutlass_fp4_gemv_f32_on_stream(const std::uint8_t*,
                                                                   const std::uint8_t*,
                                                                   const std::uint8_t*,
                                                                   const std::uint8_t*,
                                                                   const float*,
                                                                   float*,
                                                                   std::uint32_t,
                                                                   std::uint32_t,
                                                                   float,
                                                                   cudaStream_t) {
    return cudaErrorNotSupported;
}

extern "C" int infer_cutlass_fp4_grouped_gemv_f32_supported(std::uint32_t,
                                                                   std::uint32_t,
                                                                   std::uint32_t) {
    return 0;
}

extern "C" void* infer_cutlass_fp4_grouped_gemv_f32_create(std::uint32_t,
                                                                  std::uint32_t,
                                                                  std::uint32_t) {
    return nullptr;
}

extern "C" void infer_cutlass_fp4_grouped_gemv_f32_destroy(void*) {}

extern "C" cudaError_t infer_cutlass_fp4_grouped_gemv_f32_on_stream(
    void*,
    const std::uint8_t* const*,
    const std::uint8_t* const*,
    const std::uint8_t* const*,
    const std::uint8_t* const*,
    const float* const*,
    float* const*,
    float,
    float,
    cudaStream_t) {
    return cudaErrorNotSupported;
}
