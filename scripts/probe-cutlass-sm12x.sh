#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEPS_DIR=${EIDER_DEPS_DIR:-"${REPO_ROOT}/.deps"}
CUTLASS_DIR=${CUTLASS_DIR:-"${DEPS_DIR}/cutlass"}
BUILD_DIR=${BUILD_DIR:-${CUTLASS_BUILD_DIR:-"${DEPS_DIR}/cutlass-build-sm121"}}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}

if [[ ! -d "${CUTLASS_DIR}" ]]; then
  mkdir -p "$(dirname -- "${CUTLASS_DIR}")"
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
fi

cmake -S "${CUTLASS_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCUTLASS_NVCC_ARCHS=121 \
  -DCUTLASS_ENABLE_TESTS=ON \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCMAKE_BUILD_TYPE=Release

"${CUDA_HOME}/bin/nvcc" \
  -std=c++17 \
  -O3 \
  --expt-relaxed-constexpr \
  -arch=sm_121 \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  -I"${CUDA_HOME}/targets/sbsa-linux/include" \
  "${SCRIPT_DIR}/cutlass_sm12x_nvfp4_compile_probe.cu" \
  -o "${BUILD_DIR}/spark_cutlass_sm12x_nvfp4_compile_probe"

"${BUILD_DIR}/spark_cutlass_sm12x_nvfp4_compile_probe"
