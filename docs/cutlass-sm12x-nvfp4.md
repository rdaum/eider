# SM12x CUTLASS NVFP4 probe notes

The Colfax SM12x NVFP4 article confirms the relevant native path for DGX Spark:

- Spark `sm_121` uses warp-level `mma.sync`, not SM100 `tcgen05` / TMEM.
- Native NVFP4 blockscaled MMA is fixed at `m16n8k64` with E2M1 operands,
  UE4M3 scale factors, and `scale_vec::4X` over the K=64 atom.
- CUTLASS/CuTe already encodes the SM12x scale-factor thread-value layouts for
  SFA/SFB. A handwritten kernel must exactly match those register layouts.

## Setup

Run this once to clone and configure CUTLASS for `sm_121` under the repo-local,
git-ignored `.deps/` directory:

```bash
scripts/setup-cutlass-sm12x.sh
source .deps/cutlass-sm12x.env
```

Then normal Cargo builds compile the CUTLASS decode GEMV object. Without the
environment file or a repo-local `.deps/cutlass` setup, the build compiles a stub
and decode falls back to cuBLASLt.

Override locations with `EIDER_DEPS_DIR`, `CUTLASS_DIR`,
`CUTLASS_BUILD_DIR`, and `CUDA_HOME`.

For CI or one-shot local builds, set `EIDER_AUTO_SETUP_CUTLASS=1` to let
`build.rs` run the setup script when CUTLASS is missing. Set
`EIDER_REQUIRE_CUTLASS=1` to fail instead of compiling the fallback stub.

## Local probe status

The repo does not vendor CUTLASS. The setup script clones NVIDIA CUTLASS and
configures it for `sm_121` with CUDA 13.0.

```bash
cmake -S .deps/cutlass -B .deps/cutlass-build-sm121 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCUTLASS_NVCC_ARCHS=121 \
  -DCUTLASS_ENABLE_TESTS=ON \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

The repo includes `scripts/cutlass_sm12x_nvfp4_compile_probe.cu`, a minimal
standalone compile probe using `cutlass::nv_float4_t<cutlass::float_e2m1_t>` and
UE4M3-backed SM120 blockscaled tensor op builders. `scripts/probe-cutlass-sm12x.sh`
configures CUTLASS for `sm_121`, compiles this probe, and runs the resulting
binary. This validates CUTLASS/CUDA SM121 compile support for the NVFP4 kernel
family, but not runtime GEMM performance.

## Integration implication

The row-major `M x K` view consumed by CUTLASS GEMV is memory-compatible with
the runtime's column-major `K x M` weight storage for `A^T * B`. The CUTLASS GEMV
scale addressing also matches the existing tiled UE4M3 scale layout for the
`N=1` decode path, so no load-time scale conversion is required for this kernel.

The crate now builds an optional CUTLASS-backed F32-output GEMV entrypoint when
`CUTLASS_DIR` and `CUTLASS_BUILD_DIR` point at a configured CUTLASS checkout. If
those headers are absent, a stub is archived and decode falls back to cuBLASLt.

## CUTLASS FP4 GEMV benchmark

CUTLASS's stock `examples/91_fp4_gemv/91_fp4_gemv.cu` was compiled directly for
`sm_121` because the generated example target omitted `tools/util/include` from
its include path. The tested kernel is a blockscaled FP4 GEMV specialized for
`N=1`, with FP4 output plus output scale factors.

```bash
/usr/local/cuda-13.0/bin/nvcc \
  /tmp/opencode/cutlass/examples/91_fp4_gemv/91_fp4_gemv.cu \
  -o /tmp/opencode/91_fp4_gemv_sm121 \
  -I/tmp/opencode/cutlass/include \
  -I/tmp/opencode/cutlass/tools/util/include \
  -I/tmp/opencode/cutlass/examples/common \
  -I/tmp/opencode/cutlass-build-sm121/include \
  -std=c++17 -O3 -DNDEBUG -DCUTLASS_VERSIONS_GENERATED \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
  --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 \
  -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 \
  -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-fno-strict-aliasing \
  --generate-code=arch=compute_121,code=sm_121
```

Measured with 200 iterations:

| Shape | Operation | CUTLASS GEMV | cuBLASLt FP4 TN |
| --- | --- | ---: | ---: |
| `M=6144,N=1,K=4096` | fused QKV | 0.0287 ms | 0.0568 ms |
| `M=24576,N=1,K=4096` | fused gate/up | 0.2562 ms | 0.2474 ms |
| `M=4096,N=1,K=12288` | FFN down | 0.0976 ms | 0.1035 ms |

The stock CUTLASS GEMV epilogue requires FP4 output and writes SFD output scales,
while the current inference path consumes F32 projection outputs for QK norm,
RoPE, SiLU, and residual work. `scripts/cutlass_sm12x_fp4_gemv_f32_bench.cu`
uses the same SM12x GEMV mainloop with a minimal F32-output epilogue.

Measured F32-output GEMV with 200 iterations:

| Shape | Operation | CUTLASS F32 GEMV | cuBLASLt FP4 TN |
| --- | --- | ---: | ---: |
| `M=6144,N=1,K=4096` | fused QKV | 0.0287 ms | 0.0599 ms |
| `M=24576,N=1,K=4096` | fused gate/up | 0.2372 ms | 0.2677 ms |
| `M=4096,N=1,K=12288` | FFN down | 0.0779 ms | 0.1275 ms |
| `M=151936,N=1,K=4096` | lm-head | 1.4477 ms | not measured |

Runtime integration result for Qwen3-8B NVFP4, 200 decode tokens, 3 repeats:

| Path | Median decode ms | Decode TPS |
| --- | ---: | ---: |
| cuBLASLt optimized baseline | 5247.5 | 38.11 |
| CUTLASS F32 GEMV decode | 4727.7 | 42.30 |
| vLLM reference | n/a | 38.6 |

## cuBLASLt decode GEMM counter probe

`fp4_cublaslt` now uses micromeasure diagnostic replay plus CUPTI/NVPerf range
profiling to report GPU utilization counters separately from normal CUDA-event
timing. On GB20B, the selected default metrics are:

- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`

Measured decode-shape counter medians:

| Shape | Operation | CUDA event / chunk | Memory peak | L2 peak | SM peak | Tensor active |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `M=6144,N=1,K=4096` | fused QKV | 4.551 ms / 80 ops | 21.17% | 21.17% | 22.13% | 23.78% |
| `M=24576,N=1,K=4096` | fused gate/up | 8.947 ms / 40 ops | 20.98% | 20.98% | 22.31% | 23.13% |
| `M=4096,N=1,K=12288` | FFN down | 4.152 ms / 40 ops | 21.06% | 21.06% | 20.80% | 22.31% |

All three cuBLASLt `N=1` decode shapes are low-utilization rather than clearly
memory- or tensor-core-saturated. This points away from bandwidth-only fixes for
the current short decode benchmark and toward decode-specific kernels, especially
a BF16/F32-output CUTLASS GEMV or a custom fused path for FFN/QKV shapes.
