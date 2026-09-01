# cuda-oxide SM121 experiment

This stand-alone experiment tests cuda-oxide against Eider's SM121 kernel requirements.
It covers the native NVFP4 instruction and the current W4A16 matvec design.

The experiment is outside the Eider workspace. It does not load model weights.
Each program allocates less than 128 MiB of device memory.

## Pinned inputs

- cuda-oxide commit: `97f8b2b7882f0c15ad9ce9b53abed5553920caa8`
- Rust toolchain: `nightly-2026-08-28`
- CUDA toolkit: 13.0
- CUDA target: `sm_121a`

Install `cargo-oxide` from the same commit:

```sh
cargo +nightly-2026-08-28 install \
    --git https://github.com/NVlabs/cuda-oxide \
    --rev 97f8b2b7882f0c15ad9ce9b53abed5553920caa8 \
    cargo-oxide
```

## Run the Rust probe

Inspect the generated PTX:

```sh
cargo oxide inspect --arch sm_121a
```

The PTX must contain this instruction:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

Build a CUBIN, launch the kernel, and compare the results:

```sh
cargo oxide run --materialize-cubin --arch sm_121a
```

The program checks two single-instruction inputs across all accumulator registers and lanes:

- A zero register image must produce zero.
- Eider's packed E2M1 fixture must produce 128.

The second input matches Eider's
`sm12x_mma_tile_frag_host_images_accumulate_k64` test.

The program also accumulates 64 native tiles in a K loop.
It checks one warp and a grid of 64 warps.

The final kernel matches the structure of `infer_sm12x_native_gemv_kernel`.
It uses one warp per M tile and writes the same 16-row output layout.

## Run the NVCC baselines

`native_baseline.cu` contains the equivalent CUDA C++ kernel.

```sh
mkdir -p target
/usr/local/cuda-13.0/bin/nvcc -O3 \
    --generate-code=arch=compute_121a,code=sm_121a \
    native_baseline.cu -o target/native-baseline
target/native-baseline
```

`w4a16_baseline.cu` compiles the production W4A16 kernel from Eider.
It also contains a dense 16-warp control kernel.

```sh
/usr/local/cuda-13.0/bin/nvcc -O3 \
    --generate-code=arch=compute_121a,code=sm_121a \
    w4a16_baseline.cu -o target/w4a16-baseline
target/w4a16-baseline
```

## Results

Hardware: NVIDIA GB10 in a DGX Spark. Date: 2026-08-31.

| Single-instruction property | cuda-oxide | NVCC |
|---|---:|---:|
| Typed launch latency | about 2.05 us | about 2.04 us |
| Registers per thread | 14 | 14 |
| Local, shared, and stack memory | 0 bytes | 0 bytes |
| Accumulator store | one 128-bit store | one 128-bit store |

Five cuda-oxide runs varied from 2.05 us to 2.09 us.
Five NVCC runs varied from 2.04 us to 2.05 us.
The measurement is launch-dominated, so the timing difference is noise.

Both compilers emit the `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instruction.
The aligned Rust output type also produces one `STG.E.128` instruction.

The safe Rust index retains one bounds predicate because the buffer length is a runtime value.
This predicate is the only material control-flow difference in the probe.

### K-loop result

The K-loop uses 64 native tiles and 64 MMA instructions per warp.
All input tiles use the packed E2M1 fixture and unit scales.

| K-loop property | cuda-oxide | NVCC |
|---|---:|---:|
| One-warp latency | about 5.1 us | about 8.2 us |
| 64-warp grid latency | about 6.1 us | about 8.2 us |
| Registers per thread | 80 | 40 |
| Local, shared, and stack memory | 0 bytes | 0 bytes |

The Rust kernel checks every output fragment against the expected value of 8192.
The native baseline performs the same check.

The generated Rust SASS keeps more tile data live before each MMA group.
It uses twice as many registers and runs faster in this small kernel.

Native unroll factors from 2 through 64 remained near 8.2 us.
An unroll factor of 1 increased the native latency to about 12.4 us.
The simple unroll setting does not explain the cuda-oxide result.

All blocks read the same small input fixture, so the K-loop is cache-resident.
The result is not a model-level throughput claim.

### Native GEMV result

The GEMV correctness case uses two M tiles and two K tiles.
It combines different scales and a zero weight tile.
The first M tile produces 384, and the second M tile produces 128.

The shape cases match the dense Qwen3.8 feed-forward matrix dimensions.
The gate and up weights use one combined output matrix.

| Qwen-shaped GEMV | cuda-oxide | NVCC |
|---|---:|---:|
| Gate+up `[34816, 5120]` | about 336 us | about 336 us |
| Down `[5120, 17408]` | about 155 us | about 164 us |
| Registers per thread | 80 | 40 |
| Local, shared, and stack memory | 0 bytes | 0 bytes |

Five repeated gate+up runs showed no material difference between the compilers.
The cuda-oxide down result remained approximately 5 percent faster.

The down shape has fewer M tiles and a longer K loop.
Its lower latency is consistent with the additional live tile data in the cuda-oxide schedule.

These are Qwen-shaped synthetic matrices, not calls from the Qwen3.8 runtime.
The current dense Qwen3.8 runtime quantizes activations and uses W4A4 for these projections.

### W4A16 result

The W4A16 kernel keeps E2M1 weights compressed in device memory.
It expands each weight tile to BF16 registers during the matvec.

The correctness case covers the tiled scale layout, the global scale, and BF16 output rounding.
A two-row case also covers the eight-warp batch kernel.

The self-contained routed benchmark later found an error in the BF16 MMA A fragment.
The kernel supplied its second and third fragment registers in the wrong order.
Uniform synthetic weights concealed this error.

The benchmark now uses varied E2M1 values and UE4M3 scales.
It compares direct F32, BF16, and graph outputs with the row-major W4A16 reference.
The benchmark covers top-1, top-8, and top-10 without a model checkpoint.

After the correction, fixed top-k kernels removed the single-row route division.

| Routed W4A16 gate/up `[1024, 2048]` | Generic top-k | Fixed top-k |
|---|---:|---:|
| Top-8 | about 39 us | about 37 us |
| Top-10 | about 48 us | about 46 us |

The fixed top-k kernels improve these cases by approximately 5 percent.
The top-1 specialization had no measurable effect, so the dense path remains generic.

The compiler comparison removes the routed-batch bookkeeping from both kernels.
Both dense kernels use a fixed 16-warp launch and perform the same stores.

| Dense W4A16 control | cuda-oxide | NVCC |
|---|---:|---:|
| Gate+up `[34816, 5120]` | about 480 us | about 480 us |
| Down `[5120, 17408]` | about 264 us | about 264 us |
| Registers per thread | 40 | 40 |
| Shared memory | 2 KiB | 2 KiB |
| Local and stack memory | 0 bytes | 0 bytes |

Both compilers produce spill-free kernels with the same resource use.
The corrected W4A16 comparison shows no measurable compiler advantage.

## Conclusion

cuda-oxide compiles and launches Eider's core SM121 NVFP4 instruction.
The single-instruction probe shows no launch or register cost relative to NVCC.

The complete native GEMV result confirms the K-loop observation.
The cuda-oxide schedule helps the long-K shape and has neutral performance on gate+up.

The W4A16 port also handles shared memory, warp shuffles, BF16 conversion, and BF16 MMA.
Its resource use matches NVCC, but its schedule does not improve performance.

The routed benchmark found a production improvement in the fixed top-k kernels.
This improvement does not require a new backend.

cuda-oxide still requires a pinned nightly Rust toolchain.
The current result does not justify an Eider runtime switch.
