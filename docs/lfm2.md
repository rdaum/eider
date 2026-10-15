# LFM2.5-24B-A2B

Eider loads the [ModelOpt NVFP4 checkpoint](https://huggingface.co/infinity-ai-institute/LFM2.5-24B-A2B-NVFP4)
at revision `2d61270a276c320af204143d3d73e524f9ef44d2`.
The routed experts use NVFP4 weights. Attention, convolution, routers, dense
layers, embeddings, and the tied output projection retain BF16 weights.

The model has 40 layers: 30 gated short-convolution layers and 10 attention
layers. Its first two feed-forward blocks are dense. The remaining blocks
select four of 64 experts. Each request owns its convolution history and
attention history.

## Start the server

Download the pinned checkpoint:

```sh
hf download infinity-ai-institute/LFM2.5-24B-A2B-NVFP4 \
  --revision 2d61270a276c320af204143d3d73e524f9ef44d2
```

Start Eider:

```sh
EIDER_MODEL=lfm2.5-24b-a2b-nvfp4 scripts/run-eider --offline
```

Start Pi in a second terminal:

```sh
scripts/run-pi-eider-lfm2.sh
```

API requests use `eider-lfm2.5-24b-a2b`. Both `/v1/responses` and
`/v1/chat/completions` use the same LFM2 service. The checkpoint template always
starts a reasoning section. Tool calls use the LFM function-list protocol.
The parser accepts literal arguments and never evaluates generated code.

The catalogue defaults to 32,768 context tokens and prompt chunks of 256 tokens.
The checkpoint permits 131,072 tokens. Attention uses F32 KV storage, so longer
contexts increase memory use. Prefix snapshots retain both convolution state and attention history.
The cache retains chunk boundaries and the prompt before its final token.
Lookup restores the longest retained prefix that matches the request.
Restoration copies valid KV rows and convolution history into private request
storage. LRU eviction enforces `--retained-prefix-gib`; zero disables retention.

## Implementation boundary

Prefill groups routes by expert and multiplies eight rows per tensor-core tile.
The kernel reads the original NVFP4 weight and scale tables without expanding
the resident weights. It applies each projection's global scale before SiLU.
F32 activations are split into three BF16 components for tensor-core products.
Separate accumulators limit rounding error. Dense BF16 prefill projections use
the same arithmetic. Prefill scratch and route-sorting storage are reusable.

Head-64 attention divides KV rows among eight warps, then merges their softmax
sums. It uses F32 queries, keys, values, and accumulators.
Decode captures one CUDA graph per active sequence. Device inputs select the
next token, RoPE position, and valid KV length on each replay.
Graph lifetimes retain the model and sequence buffers they reference.
Speculative decoding is not implemented.

Model architecture and equations follow the
[official checkpoint](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
and [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py).

## Correctness checks

The small fixture and full-checkpoint reference tests pass with maximum
absolute logit error below `2e-4`. They cover sequential execution, batched
chunks, graph replay, and prefix restoration after the source request advances.
The fixture also covers cache eviction and disabled retention.
Offline catalogue startup, reasoning separation, and tool-call round trips
passed through both HTTP APIs. Live checks also covered concurrent requests,
streaming, stop sequences, output limits, and cancellation with state release.

Generate the small reference checkpoint and CPU logits:

```sh
python3 scripts/lfm2-reference-fixture.py /tmp/lfm2-reference
```

Run the CUDA reference test:

```sh
LFM2_REFERENCE_DIR=/tmp/lfm2-reference \
  cargo test -p eider-inference --lib lfm2_reference -- --ignored --test-threads=1
```

This test compares every token against independent CPU logits. It also checks
whole prompts, split prompts, and interleaved requests. The fixture includes
different global scales for every expert and projection.

For the full checkpoint, produce the reference outside its immutable snapshot:

```sh
checkpoint="$HOME/.cache/huggingface/hub/models--infinity-ai-institute--LFM2.5-24B-A2B-NVFP4/snapshots/2d61270a276c320af204143d3d73e524f9ef44d2"
python3 scripts/lfm2-checkpoint-reference.py "$checkpoint" \
  /tmp/lfm2-full-reference.json 124894 13 2 97 255 6 31 77
LFM2_REFERENCE_DIR="$checkpoint" LFM2_REFERENCE_JSON=/tmp/lfm2-full-reference.json \
  cargo test -p eider-inference --lib lfm2_reference -- --ignored --test-threads=1
```

The CPU reference dequantizes the same NVFP4 weights and uses F32 activations.
It checks execution accuracy, not quality against the original BF16 model.

Run the focused kernel and protocol tests:

```sh
cargo test -p eider-cuda --lib lfm2 -- --test-threads=1
cargo test -p eider-cuda --lib head64_attention -- --test-threads=1
cargo test -p eider-runtime --lib lfm2
```

The projection benchmark compares all outputs against the F32 SIMT path before
timing. The attention benchmark checks causal boundaries against an F64 reference.

```sh
cargo bench -p eider-cuda --bench lfm2_attention
cargo bench -p eider-cuda --bench lfm2_experts
```

## Request timings

Measured on GB10 with the pinned checkpoint, greedy sampling, one request at a
time, and an 8,192-token context limit. Each binary served one cold request
and one repeat. Model loading is excluded. The prompt has 4,864 tokens;
the completion limit is 96 tokens. Both versions use batched prefill, decode
graphs, and the prefix cache. Request timings exclude profiler instrumentation.

| Measurement | Before kernel changes | Current |
| --- | ---: | ---: |
| Prefill throughput | 172 tok/s | 684 tok/s |
| Decode throughput | 14.77 tok/s | 52.63 tok/s |
| Cold time to first token | 28.35 s | 7.14 s |
| Cold total request time | 34.79 s | 8.95 s |
| Repeated-prompt time to first token | 0.089 s | 0.065 s |
| Repeated-prompt total request time | 6.52 s | 1.87 s |

The repeated request restored 4,863 prompt tokens. All 96 generated tokens
matched the baseline on both requests. The completion limit truncates reasoning
in this timing workload. Separate API checks validate final answers, tool-call
round trips, streaming, output limits, and cancellation.

The original GPU trace showed serial attention and repeated expert matrix-vector
products as the main costs. At this context, the head-64 attention microbenchmark
improved from 2.88 to 0.22 ms for decode. Its final 256-row prefill chunk improved
from 82.6 to 9.46 ms. Grouped expert projections improved from about 7.4 to
1.5–1.7 ms per projection, including route sorting. These kernel measurements
use synthetic inputs; the request table measures the full runtime.

Run the same workload against a freshly started server:

```sh
python3 scripts/lfm2-request-bench.py --repetitions 220 \
  --output /tmp/lfm2-request-timing.json
```
