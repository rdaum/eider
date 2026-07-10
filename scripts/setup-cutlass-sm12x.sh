#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
DEPS_DIR=${EIDER_DEPS_DIR:-"${REPO_ROOT}/.deps"}
CUTLASS_DIR=${CUTLASS_DIR:-"${DEPS_DIR}/cutlass"}
CUTLASS_BUILD_DIR=${CUTLASS_BUILD_DIR:-"${DEPS_DIR}/cutlass-build-sm121"}
CUTLASS_REPO=${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}
CUTLASS_REF=${CUTLASS_REF:-main}
ENV_FILE=${ENV_FILE:-"${DEPS_DIR}/cutlass-sm12x.env"}

mkdir -p "${DEPS_DIR}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  printf 'missing nvcc at %s\n' "${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

if [[ ! -d "${CUTLASS_DIR}/.git" ]]; then
  rm -rf "${CUTLASS_DIR}"
  git clone --depth 1 --branch "${CUTLASS_REF}" "${CUTLASS_REPO}" "${CUTLASS_DIR}"
fi

cmake -S "${CUTLASS_DIR}" -B "${CUTLASS_BUILD_DIR}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCUTLASS_NVCC_ARCHS=121 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TOOLS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCMAKE_BUILD_TYPE=Release

cat > "${ENV_FILE}" <<EOF
export CUDA_HOME="${CUDA_HOME}"
export CUTLASS_DIR="${CUTLASS_DIR}"
export CUTLASS_BUILD_DIR="${CUTLASS_BUILD_DIR}"
EOF

printf 'CUTLASS configured for sm_121.\n'
printf 'To enable the CUTLASS decode path in this shell, run:\n'
printf '  source %q\n' "${ENV_FILE}"
