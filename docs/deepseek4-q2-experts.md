# DeepSeek V4 Flash expert storage

Eider does not yet serve DeepSeek V4 Flash. The implemented boundary covers
the model's routed-expert weights: preparation, resident storage, one-token
expert execution, device-side usage accounting, and a bounded NVFP4 hot
overlay. Attention, routing, shared experts, residual mixing, tokenization, and
API integration remain separate work.

## Storage

The cold format uses four signed levels (`-3`, `-1`, `1`, `3`) packed into two
bits per weight. Each block of 64 input-channel weights shares one BF16 scale,
for `9/32` byte per weight. Cache format version 2 rejects the earlier
FP32-scale artifacts.

For the 43-layer, 256-expert checkpoint:

| Resident weights | Routed-expert payload |
| --- | ---: |
| Cold Q2 only | 72.5625 GiB |
| Q2 + 1 NVFP4 slot per layer | 73.1307 GiB |
| Q2 + 4 NVFP4 slots per layer | 74.8353 GiB |
| Q2 + 8 NVFP4 slots per layer | 77.1082 GiB |
| Q2 + 16 NVFP4 slots per layer | 81.6538 GiB |

The hot sizes include the per-row tensor scales needed to preserve independent
`w1` and `w3` scales. They exclude non-expert model weights, CUDA workspaces,
sequence state, and small pointer tables.

Preparation streams one expert at a time into self-describing per-layer
gate/up and down tables. A manifest is published only after every layer is
complete. Runtime routing remains on the device; usage counts are copied to
the host only at an explicit request or maintenance boundary.

`scripts/prepare-deepseek4-experts-streaming.sh` downloads one pinned expert
shard, prepares its layer, then discards that disposable staging copy before
advancing. During the same pass it rewrites each source shard with only the
attention, router, shared-expert, and normalisation tensors needed for serving.
The optional MTP block is omitted. Existing validated outputs are reused. A
real layer-0 conversion produced 1,811,939,400 Q2 artifact bytes and a
136.95 MiB thin serving shard from a 3.6 GiB source shard, and both validated
successfully after the source shard was removed.

Hotset refresh is observation-driven. It ranks cumulative routing counts,
loads the selected original NVFP4 experts from a bounded source cache into
fixed slots, and falls back to the complete Q2 table for every other expert.
A failed multi-matrix refresh clears both overlays so gate/up and down cannot
retain divergent mappings.

The source cache stores one self-validating NVFP4 record per selected
layer/expert pair and enforces a per-layer capacity. It is rebuilt from a JSON
hotset plan with:

```sh
DEEPSEEK4_HOT_CAPACITY=8 \
    scripts/prepare-deepseek4-hotset-streaming.sh hotset-plan.json
```

The plan maps layer numbers to expert indices, for example
`{"0":[4,17,93],"1":[8,42]}`. The preparer downloads one pinned layer shard,
extracts only those experts, deletes the disposable shard, and advances.
Layers omitted from the plan are cleared. At capacity eight, this cache is
bounded to about 4.55 GiB across all 43 layers.

## Evidence

The correctness-gated `ds4-q2-expert-layer-decode` benchmark includes gate/up,
clamped SwiGLU, down projection, and weighted accumulation for six routed
experts at the checkpoint's real shapes.

| Path | Median CUDA time | Experiment |
| --- | ---: | --- |
| Q2 with FP32 block scales | 0.3392 ms | `019f969b-b9f6-7b61-992c-71374496ae7f` |
| Q2 with BF16 block scales | 0.2915 ms | `019f96b6-4f8e-78f2-ba17-97f9c3c307bb` |
| BF16 Q2 with 3 of 6 routes hot | 0.3818 ms | `019f96b9-eda7-7473-b455-da5c8a0a7297` |
| Original NVFP4 | 0.4700 ms | `019f969e-4106-7fd1-882d-47f25a471e93` |

BF16 scales reduced the complete expert-layer median by 14.1% while saving
8.0625 GiB across the routed experts. The intended half-hot mixed path also
improved by 5.5% against its FP32-scale predecessor. This establishes storage
and component latency, not model quality; a full decode comparison remains the
acceptance gate before enabling Q2 by default.

## Deployment constraint

The NVIDIA checkpoint index reports 168,266,793,544 bytes (156.7 GiB). Keeping
that complete snapshot beside a 72.6 GiB Q2 artifact leaves effectively no
working space on the current filesystem.

Snapshot `e3cd60e7de98e9867116860d522499a728de1cf9` places each layer's routed
experts in exactly one shard: layers 0 through 42 map to shards 2 through 44.
That makes bounded shard-at-a-time conversion and hot-cache maintenance
practical. Removing the routed payload and optional MTP block from shards 1
through 45 gives a conservative thin-checkpoint upper bound of 8.253 GiB
(source headers included). Q2 experts plus that upper bound and eight hot
NVFP4 slots per layer total at most 85.36 GiB, leaving about 42.6 GiB of the
128 GiB unified-memory budget for CUDA workspaces, sequence state, and the
host.
