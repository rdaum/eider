# Ternary CUDA bakeoff

## Question

Ternary Bonsai stores four `{-1, 0, +1}` codes per byte with one FP32 scale
per 64 weights. Eider uses those packed weights directly for single-token
W2A8 decode. The serving default retains an exact BF16 expansion for prefill;
an explicit NVFP4 representation remains available for benchmarks and semantic
replay after the quality bakeoff below rejected it as the serving default.

The useful comparison is therefore not ternary versus BF16. It is:

- packed ternary W2A8 versus NVFP4 W4A16 for single-token decode;
- the current BF16 tensor-core prefill versus NVFP4 W4A4 prefill;
- the complete resident formats and end-to-end serving paths, including
  activation conversion.

`crates/nvfp4/benches/ternary_g64_w2a8.rs` contains the focused comparison at
the four production Bonsai projection shapes. It converts the same synthetic
group-scaled ternary values to NVFP4, checks both formats against their
references, includes activation conversion in the timed boundary, and reports
resident weight bytes.

## Existing CUDA techniques

### BitNet GPU W2A8

Microsoft's official [BitNet GPU kernel](https://github.com/microsoft/BitNet/tree/main/gpu)
uses the same broad arithmetic as Eider: two-bit weights are decoded into
packed INT8 lanes and consumed with `dp4a`. The important differences are the
layout and decode sequence:

- weights are prepared in 16-by-32 tiles rather than retained row-major;
- each 16-value word uses the interleave
  `[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]`;
- four INT8 lanes are decoded with `lop3.b32` followed by packed-byte
  subtraction, rather than shifts and per-code selection;
- the kernel is specialized by projection shape and K-reduction width.

The implementation is small enough to inspect directly in
[`bitnet_kernels.h`](https://github.com/microsoft/BitNet/blob/main/gpu/bitnet_kernels/bitnet_kernels.h).
Its A100 results are not transferable to GB10, but the offline tiling and
packed decode are directly relevant.

Bonsai's K64 scales differ from BitNet's scale convention, so the kernel is not
a drop-in replacement. The next ternary candidate should preserve Bonsai's
scales while adopting the tiled code layout, vector loads, packed decode, and
shape-specific sub-warp reduction.

### Redundant Segment Reduction

[RSR-core](https://github.com/UIC-InDeXLab/RSR-core), described in the
[RSR-core paper](https://arxiv.org/abs/2603.27462), preprocesses a ternary
matrix by grouping columns with identical patterns across small row blocks.
At decode time it aggregates each corresponding activation group once and
scatters the result through positive and negative row masks. Its retained CUDA
path uses sorted 16-bit permutation indices, removes all-zero groups, packs a
group's range and masks into one 64-bit word, and specializes the row-block
height.

This is a materially different decode algorithm rather than a faster unpack.
It may reduce arithmetic when patterns repeat, but it adds permutation and
metadata streams and is specific to matrix-vector execution. It needs a
Bonsai-shaped memory and latency measurement before integration; it does not
solve prefill.

### Lookup-table and multiplication-free kernels

The [bitnet.cpp paper](https://arxiv.org/abs/2502.11880) describes ternary
lookup-table and I2-with-scale kernels, but its main results target edge CPUs.
Likewise, [FairyFuse](https://arxiv.org/abs/2604.20913) reports that its
conditional add/subtract ternary technique transfers poorly to CUDA because
commodity NVIDIA GPUs lack the CPU instructions that make its packed masks
cheap. These approaches are useful warnings against assuming that removing
floating-point multiplication automatically helps GB10.

The current [Bonsai llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/22019)
also converges on a group-64 two-bit format, but does not yet provide a
production CUDA result for this exact checkpoint format.

## Current hypotheses

1. Packed ternary should retain a real bandwidth advantage during decode:
   2.5 effective bits per weight versus approximately 4.5 for NVFP4.
2. The current row-major Eider kernel leaves enough decode efficiency unused
   that NVFP4 may nevertheless win at some or all Bonsai shapes.
3. NVFP4 W4A4 should beat the persistent BF16 prefill path while cutting the
   prefill mirror from 16 to approximately 4.5 bits per weight.
4. If ternary wins decode but NVFP4 wins prefill, the likely deployment is a
   hybrid packed-ternary plus NVFP4 model. That is approximately seven bits per
   transformer weight before padding, far below the current 18.5-bit hybrid.
5. A BitNet-style tiled decode layout is the lowest-risk ternary candidate.
   RSR is a separate, higher-risk experiment after the direct kernel is tuned.

## Supporting Eider evidence

### Exact Bonsai layer bakeoff

The correctness-gated four-projection aggregate produced the following GB10
CUDA-event medians:

| Path | Median | Resident layer weights | Experiment |
| --- | ---: | ---: | --- |
| packed ternary W2A8 decode | 0.318208 ms | 60,293,120 bytes | `019fe6f2-8c5c-7870-b58a-b68fc4181dfb` |
| NVFP4 W4A16 decode | 0.786016 ms | 108,527,616 bytes | `019fe6f7-838f-7453-ab5d-1a39a2560b78` |
| direct ternary W2A8 prefill | 25.671008 ms | 60,293,120 bytes | `019fe6f4-38fd-7672-8c5b-94f21e973f41` |
| current BF16-mirror prefill | 2.804768 ms | 446,169,088 bytes | `019fe6f5-15ae-7c01-a168-7e9de996f043` |
| NVFP4 W4A4 prefill | 1.273952 ms | 108,527,616 bytes | `019fe6f9-3166-7ba3-84ff-b8a1486f2c49` |

Packed ternary decode is 2.4701 times faster than the current NVFP4 W4A16
control while retaining its 44.44% weight-byte advantage. NVFP4 W4A4 prefill
is 2.2016 times faster than the BF16 mirror and 20.1507 times faster than
direct ternary prefill. The focused result therefore supports packed ternary
decode plus an NVFP4 prefill mirror. That hybrid occupies 168,820,736 bytes per
transformer layer, 62.16% less than the current packed-ternary-plus-BF16
representation.

The synthetic ternary-to-NVFP4 weight conversion produced selected-row cosine
1.0 and NRMSE 0.03125. W4A4 outputs versus the same NVFP4 weights under W4A16
had cosine from 0.983309 to 0.991894 and NRMSE from 0.278340 to 0.280679. These
are kernel-level correctness and fidelity gates, not proof of full-model
quality.

### Serving trial

The trial serving model constructed each layer as packed group-64 ternary plus
a cuBLASLt-ready NVFP4 prefill weight. Per-sequence prefill workspaces retained
four shape-specific W4A4 plans and reusable packed activation matrices. Decode
continued to use the original two-bit weights and W2A8 kernel.

The real 8B GGUF loaded successfully, including a 256-token warm-up. A
deterministic 4,623-token prompt produced the correct requested summary and
measured 2,381.274 ms of prefill compute, or 1,941.398 token/s. The comparable
pre-change 4,603-token run measured 1,364.460 token/s, so the complete serving
path improved by 42.28% at that prompt length. This was a production-shaped
performance smoke check, not sufficient semantic evidence; the deterministic
replay below subsequently rejected the candidate. It is also not a substitute
for the durable Abenchting CUDA-event evidence.

Abenchting experiment `019fe70a-324c-70b1-b9c7-69e48a1527cc` compared the
production owner and execution API against the NVFP4 implementation.
Correctness passed. The paired baseline median was 1.292832 ms and the
candidate median was 1.291936 ms, a -0.069% change below the 2% practical
threshold and therefore correctly classified as inconclusive. Candidate
CUDA-event samples were `[1.293759987, 1.291935995, 1.283168003, 1.288735986,
1.346783996]` ms. The candidate reported the complete hybrid layer footprint
of 168,820,736 resident weight bytes.

### Pi semantic replay

Layer-level fidelity did not establish full-model semantic quality. A
standalone replay therefore rendered the same captured Pi Responses request
with the checkpoint's GGUF chat template and greedily decoded it through either
the exact BF16 expansion or the NVFP4 prefill representation. It retained every
selected token, the top five logits at every step, decoded text, and parsed tool
calls.

On the 4,395-token repository-orientation prompt, BF16 produced a coherent
description from the supplied project context. NVFP4 instead promoted the
tool-call marker from third to first place and attempted to pass the repository
directory to the file-reading tool. The first selected token differed and the
two outputs had no common token prefix. This is a deterministic semantic
regression, not sampling noise.

The original conversational boundary was then reconstructed on the captured
four-tool Pi harness by appending the model's recorded first answer and the user
instruction `read some of the files`. Both paths emitted the same 35-token call:
`read({"path":"/home/ryan/src/spark-infer","limit":2000})`. All selected token
IDs were identical even though selected logits differed by 1.602 on average
and as much as 5.191. BF16 therefore does not make Bonsai a reliable Pi agent,
but NVFP4 also introduces an additional failure and does not satisfy the
task-output acceptance gate.

The local replay measured approximately 1,530 token/s for BF16 and 1,781
token/s for NVFP4 on the 4,508-token reconstructed prompt. These single-run
wall-clock figures are diagnostic only; the durable Abenchting measurements
above remain the performance evidence. Serving defaults to BF16. NVFP4 remains
an explicit comparison mode rather than a production path.

### Related packed-Q2 control

A retained correctness-passing Eider experiment compared packed Q2 W2A16 with
NVFP4 W4A16 for six 4096-by-4096 routed matvecs. It is not the Bonsai kernel or
activation format, so it cannot decide this bakeoff, but it confirms that
low-bit packed decode can retain a measurable bandwidth advantage in Eider:

- Q2 experiment `019f9688-6dea-7a71-983b-ead4bf4debab`: CUDA-event samples
  `[0.222350407, 0.214547205, 0.197523201]` ms;
- NVFP4 experiment `019f9686-f106-7160-9238-d8a543302f07`: CUDA-event samples
  `[0.343905592, 0.340812802, 0.331471992]` ms;
- median Q2 latency was 37.05% lower, with 44.44% fewer nominal resident
  weight bytes.

That result supports the decode hypothesis only. It says nothing about Bonsai
quality, group-64 scaling, W2A8 activation cost, prefill, or the full model.

## Acceptance boundary

Retain a candidate only when all of the following hold:

- packed import and GPU output correctness pass;
- CUDA-event samples include activation conversion;
- all four Bonsai projection shapes are measured at M=1 and M=256;
- the layer-weighted result beats the current path, not merely one projection;
- model load and resident bytes are recorded;
- end-to-end serving preserves logits or task output and improves decode or
  prefill on a production prompt.

The trusted Abenchting `eider` project declares aggregate Bonsai workloads for
the two decode and three prefill paths. Per-projection benchmarks remain
available for diagnosing an aggregate result, while the aggregate workloads
are the durable format-selection evidence.
