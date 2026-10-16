# LFM2.5 performance on DGX Spark

This analysis uses the pinned ModelOpt NVFP4 checkpoint from [the model guide](lfm2.md).
It covers one active request on GB10, with F32 activations and KV storage.
Expert weights retain their original NVFP4 representation. Other matrices retain BF16 weights.

These profiles predate the scaled FP16 expert projection used by the decision path.
The [model guide](lfm2.md#implementation-boundary) describes that change and its request timings.

## Hardware ceilings

NVIDIA specifies 273 GB/s memory bandwidth, 128 GB unified memory, and 6,144 CUDA cores.
Its advertised 1 PFLOP figure requires FP4 and structured sparsity.
This checkpoint does not use structured sparsity.
These specifications come from [the DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

A CUDA device probe reported 48 SMs, 24 MiB L2, and 100 KiB shared memory per SM.
A 1 GiB streaming read measured 235.184 GB/s across 30 passes, after five warm-up passes.
The probe used 384 blocks of 256 threads, vector loads, and a reduction that consumed every loaded value.
Each timed pass took 4.566 ms. Other model processes were stopped.
This measured bandwidth is a useful comparison point, not a guarantee for inference kernels.

SM121 uses register-based `mma.sync` operations. It does not support the SM100 tensor-memory programming model.
The [Colfax SM12x implementation notes](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
describe these constraints and the NVFP4 scale layout.

NVIDIA's [RTX Blackwell architecture paper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), table 3,
gives a ratio of eight between dense FP4 and BF16 tensor throughput with F32 accumulation.
Applying that ratio to Spark's advertised sparse FP4 peak gives an estimated 62.5 TFLOP/s BF16 ceiling.
This is an architectural estimate, not a separately published GB10 BF16 specification.
The profiled projections use three BF16 products per F32 activation, so their arithmetic cost is roughly three times one BF16 multiplication.

## Decode traffic

Each token selects four experts in each of 38 MoE layers.
The following table counts matrix data once per token. Decimal GB are used throughout this analysis.

| Matrix group | Bytes per token |
| --- | ---: |
| Convolution projections, BF16 | 1,006,632,960 |
| Attention projections, BF16 | 209,715,200 |
| First two dense FFNs, BF16 | 289,406,976 |
| Routers, BF16 | 9,961,472 |
| Selected experts, NVFP4 plus block scales | 806,879,232 |
| Tied output projection, BF16 | 524,288,000 |
| Total | 2,846,883,840 |

The minimum KV read adds 40,960 bytes per context position per generated token.
This count assumes perfect reuse across query heads in each GQA group.
It excludes norm vectors, intermediate traffic, repeated cache reads, and scheduling costs.
The resulting ceilings are optimistic upper bounds for these formats.

| Context tokens | Minimum GB/token | At 273 GB/s | At 235.184 GB/s |
| --- | ---: | ---: | ---: |
| 0 | 2.847 | 95.9 tok/s | 82.6 tok/s |
| 4,864 | 3.046 | 89.6 tok/s | 77.2 tok/s |
| 14,984 | 3.461 | 78.9 tok/s | 68.0 tok/s |
| 32,768 | 4.189 | 65.2 tok/s | 56.1 tok/s |

An unchanged-format target near 60–65 tok/s at 15K context is therefore demanding but plausible.
Substantially higher single-request rates require fewer bytes, more memory bandwidth, or speculative execution.
Batching can improve aggregate throughput through weight reuse, but does not remove latency from one dependent sequence.

## Prefill traffic and arithmetic

A 256-token chunk can select all 64 experts in each MoE layer.
Perfect weight reuse gives approximately 14.95 GB of matrix traffic per chunk, including one output projection.
At 15K context, one complete KV read adds about 0.62 GB per chunk.
The measured-bandwidth ceiling is approximately 3,866 tok/s under these optimistic reuse assumptions.
Actual kernels reload tiles and materialize intermediate results.

The active linear projections contain 2.454 billion parameters, including the output head.
Excluding the output head, three-component tensor products require approximately 13.15 GFLOP per prompt token.
The estimated BF16 peak gives approximately 4,750 tok/s before attention, conversion, routing, and output projection costs.
Thus, several thousand prompt tokens per second is a useful optimization direction, not a claim that peak throughput is attainable.

Reproduce the traffic calculation from the local checkpoint:

```sh
python3 scripts/lfm2-roofline.py "$checkpoint" --bandwidth-gbs 235.184
```

## Baseline profile

The baseline is commit `4b310f2`. A cold 14,984-token prompt used 256-token chunks and generated 96 greedy tokens.
The profile included a repeated prompt to exercise graph replay and prefix restoration.
Nsight Systems traced CUDA graph nodes, so timings here include profiler overhead.

| Operation | Aggregate GPU time | Share |
| --- | ---: | ---: |
| Expert prefill projections | 11.438 s | 35.2% |
| Prefill attention | 9.381 s | 28.8% |
| Dense prefill projections | 6.009 s | 18.5% |
| Decode attention | 2.417 s | 7.4% |
| BF16 matrix-vector projections | 1.806 s | 5.6% |
| Expert decode gate/up | 0.605 s | 1.9% |
| Expert decode down | 0.289 s | 0.9% |

The profiled request achieved 523.4 tok/s prefill and 35.3 tok/s decode.
Decode attention alone took approximately 12.7 ms per generated token.
This profile identifies attention and projection kernels as the next costs to reduce.

## Kernel changes

All kernel comparisons include correctness checks before timing.
Projection checks compare every output against the F32 SIMT path.
Attention checks use an independent F64 calculation and causal boundary cases.

| Operation | Baseline | Current |
| --- | ---: | ---: |
| Decode attention, 14,984 context tokens | 1.25 ms | 0.28 ms |
| Prefill attention, 256 rows after 14,728 tokens | 30.4 ms | 5.00 ms |
| Prefill attention, 256 rows after 4,608 tokens | 9.45 ms | 1.60 ms |
| Dense convolution projection, 256 × 2,048 → 6,144 | 1.00 ms | 0.34 ms |

Partitioned decode distributes KV work across more CTAs, then merges F32 softmax statistics.
Prefill attention shares each KV tile across 64 queries. Each warp computes eight queries.
This reuses shared-memory loads without changing the arithmetic order within each query.
Increasing the tile size alone did not provide the same gain.

Dense projections split activations once into three BF16 planes and use persistent cuBLASLt plans.
One multiplication produces all three component products. A separate kernel adds them in F32.
Power-of-two row buckets bound the plan cache; padded rows contain zeros.
The two dense FFN down projections retain the original kernel's shorter F32 accumulation chains.
Replacing those long reductions with cuBLASLt exceeded the existing full-model logit tolerance.

The expert kernel retains row-major NVFP4 weights and uses exact BF16 products for dequantization.
A packed lookup covers E2M1 values and all finite, nonnegative UE4M3 scales.
Sixteen routed inputs share each weight tile. Asynchronous copies pipeline compressed weights,
scales, and F32 activations through two shared-memory stages.
The pipeline preserves the reference tensor kernel's arithmetic order and outputs.

Nsight Compute explained the earlier expert bottleneck: L1 load latency, rather than DRAM bandwidth.
The direct 16-route kernel reached only 28% of peak device-memory bandwidth.
Long-scoreboard stalls accounted for 62% of its warp stall cycles.
Its loads used about ten bytes of each 32-byte memory sector, on average.
The pipeline reduced a 4,096-route gate/up projection from about 4.5 to 2.8 ms, including route sorting.
Larger pipelined tiles of 32 and 64 routes regressed and were rejected.

Other rejected variants included synchronous expanded-weight staging, global activation-component
buffers, a separate tiled weight representation, and additional attention dot-product accumulators.
An alternate dense decode kernel won small benchmarks but did not improve a fixed-token model replay.
It was removed.

## Whole-request results

The [model guide](lfm2.md#request-timings) records the final request timings and reproduction command.
At 15K context, prefill improved from 525 to 1,729 tok/s and decode from 36.07 to 56.39 tok/s.
Cold time to first token fell from 28.59 to 8.71 seconds.
The 96 generated tokens matched the baseline, and prefix-restored repeats matched cold requests.

Decode reaches approximately 83% of the 68.0 tok/s bound derived from measured streaming bandwidth.
That bound excludes intermediate traffic and assumes perfect reuse, so it is not a practical latency target.
Prefill remains below the estimated arithmetic ceiling. Routed tiles, dequantization, component
conversion, normalization, and attention consume time beyond the counted matrix products.

These changes preserve F32 activations and KV storage. Native W4A4 tensor execution would introduce
activation quantization and require a separate quality comparison. Spark's sparse FP4 headline
cannot be used as the performance ceiling for the profiled three-component BF16 implementation.

## Remaining GPU costs

A final Nsight Systems run used the same 15K cold request and repeated request.
Aggregate kernel time fell from 32.54 to 11.66 seconds. These figures include profiler overhead.

| Operation | Baseline GPU time | Current GPU time |
| --- | ---: | ---: |
| Expert prefill projections | 11.438 s | 4.613 s |
| Prefill attention | 9.381 s | 1.289 s |
| Dense prefill projections, including component preparation | 6.009 s | 1.829 s |
| Decode attention, including partial merge | 2.417 s | 0.535 s |

Expert prefill remains the largest cost. Its 4.613 seconds account for 40% of
kernel time across the cold and repeated requests. Larger routed tiles lost
occupancy and regressed in the focused comparisons. Further gains should target
that kernel's conversion and data movement costs, with the same numerical checks.

## Numerical checks

The tiny fixture and independent 128-token full-checkpoint CPU references retain
an absolute logit tolerance of `2e-4`. The final 128-token test passes sequential,
batched, graph, interleaved, and restored-prefix execution. An initial cuBLAS
replacement for the long dense reductions failed at `0.000337`; that dispatch was rejected.

A separate 14,984-token comparison forces the same 16 continuation tokens through
both revisions and records all 128,000 vocabulary logits at 17 positions.
All 17 argmax choices match. Mean KL divergence is `7.17e-5` nats; the largest is
`0.001219` nats. Mean forced-token NLL changes by `-0.000757` nats.
Long-context logits are not bitwise equivalent: the maximum absolute difference is 1.163.
These measurements describe this prompt and continuation, not a general quality benchmark.

Temporary routing traces located the first changed expert set at layer 5, token 11,964.
The baseline fourth/fifth selection-score gap was `4.47e-8`.
Later layers contained more changed routes: 1,967 of 569,354 compared token/layer
selections differed. This is consistent with floating-point differences amplified
by discrete routing. Tracing did not change either revision's logits, and the
trace code was removed after comparison.

Generate distribution samples with `crates/infer/examples/lfm2-logits.rs`.
Its input JSON contains `text` and an optional `continuation` array of token IDs.
Use the same text and forced continuation for both revisions. Compare their files:

```sh
python3 scripts/lfm2-compare-logits.py baseline.json candidate.json
```

The example also reports prefill and decode durations. Set `timing_only` with a
fixed continuation to omit logit downloads during a decode timing comparison.
These diagnostic timings exclude HTTP, admission, prefix snapshots, and sampling.
Use the request benchmark for serving results.
