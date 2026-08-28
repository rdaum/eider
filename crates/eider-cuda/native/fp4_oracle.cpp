#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

extern "C" std::uint8_t infer_cuda_e2m1_rn(float value) {
    return static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest) & 0x0f);
}

extern "C" std::uint8_t infer_cuda_e4m3_satfinite(float value) {
    return static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}
