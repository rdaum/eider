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

The catalogue defaults to 32,768 context tokens and prompt chunks of 1,024 tokens.
The checkpoint permits 131,072 tokens. Attention uses F32 KV storage, so longer
contexts increase memory use. Prefix snapshots retain both convolution state and attention history.
The cache retains chunk boundaries and the prompt before its final token.
Lookup restores the longest retained prefix that matches the request.
Restoration copies valid KV rows and convolution history into private request
storage. LRU eviction enforces `--retained-prefix-gib`; zero disables retention.

## Decisions

`POST /v1/decisions` supports `noul`, `choice`, and `score` questions.
Each request can contain 1–64 questions, with up to 64 labels per question.
The service processes the shared state once and packs question suffixes into
one pass through each layer. It projects only 64 selected rows from the tied BF16 output head.
It produces no reasoning text or generated answer tokens.

Dense projections and routed experts process the packed rows together.
Convolution and attention process each segment with its own history and logical
positions. Questions see the shared prefix and their own earlier tokens.
All questions reuse one request-owned KV allocation, without prefix copies.
This works with `--max-active-sequences 1`.

If the token budget permits, the first pass includes the shared prefix.
Longer prefixes use chunked prefill. Complete suffixes then run in waves bounded
by `--decision-branch-capacity` and `--prefill-token-capacity`.
Suffixes larger than the token budget use isolated chunks and restore convolution
history between questions. Decision groups yield between passes for cancellation
and other requests.

The checkpoint template always opens `<think>`. The decision compiler closes
this empty section before the answer-label prompt. Returned probabilities cover
the supplied options. They remain uncalibrated unless an explicit, validated
calibration profile is loaded for this deployment.

Send a decision request:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eider-lfm2.5-24b-a2b",
    "state": "The parcel arrived yesterday. It contains a red book.",
    "questions": {
      "arrived": {
        "type": "noul",
        "instructions": "Has the parcel arrived?"
      },
      "colour": {
        "type": "choice",
        "instructions": "What colour is the book?",
        "criteria": {"red": null, "blue": null}
      }
    }
  }'
```

The full-checkpoint regression compares selected logits against the full head,
isolated questions, and reversed question order. It also covers queued and
active cancellation with one sequence slot.

On GB10, packed execution reduced three-question HTTP latency from 417 ms to
220 ms. Scaled FP16 expert components reduced it further to 193 ms. Each median
covers five runs after one warm-up, with 379 logical input tokens and 1,024-token
prefill capacity. The 64-question case fell from 5.65 seconds to 1.59 seconds,
then 1.39 seconds, in single runs.

Grouped and isolated questions produced identical answers and raw logits in
these checks. Packed execution differed from ordinary full-head execution by
at most `2.6e-5` logits, within the existing `2e-4` reference tolerance.
The new expert kernel changed sampled label logits by at most `1.4e-5`.
The packed three-question pass includes the prefix, so its server timing appears
under `branches`. These measurements cover one factual prompt, not a task-quality
evaluation.

```sh
checkpoint="$HOME/.cache/huggingface/hub/models--infinity-ai-institute--LFM2.5-24B-A2B-NVFP4/snapshots/2d61270a276c320af204143d3d73e524f9ef44d2"
LFM2_MODEL_DIR="$checkpoint" cargo test --release -p eider-inference --lib \
  lfm2_decisions_match_isolated_full_heads_and_cancel_cleanly -- \
  --ignored --test-threads=1 --nocapture
```

## Implementation boundary

Prefill groups routes by expert and multiplies sixteen rows per tensor-core tile.
The kernel reads the original NVFP4 weight and scale tables without expanding
the resident weights. It applies each projection's global scale before SiLU.
Each routed input uses a power-of-two scale and two FP16 components.
Matrix loads feed both components into separate tensor-core accumulators.
The weights remain packed NVFP4; their finite scaled products are exact in FP16.
The former three-component BF16 path remains the benchmark reference and K16-tail
fallback. Asynchronous copies overlap weight, scale, and activation loads with
computation. Most dense BF16 projections
use cached cuBLASLt plans with three activation components. The two long dense
FFN reductions retain the kernel with four high-component accumulators.
Prefill scratch and route-sorting storage are reusable. The FP16 component
buffers add about 32 MiB at 1,024-token capacity.

At 1,516 routes, the focused projection benchmark measured gate/up at 0.933 ms
versus 1.164 ms, and down at 0.919 ms versus 1.146 ms. These medians include
route sorting and activation preparation. The request improvement is smaller because
expert projections account for about 70% of the original decision GPU time.

The checkpoint also supplies calibrated FP4 activation scales. The runtime still
uses weight-only NVFP4. A calibrated W4A4 trial reached 149 ms on the same
three-question request, but failed the selected-logit consistency check between
packed and separate prefix/suffix execution. The maximum difference was 6.96.
That trial is excluded from serving; the existing accuracy tolerance is unchanged.

Head-64 prefill attention shares K/V tiles across 64 queries. Each warp computes
eight queries and reuses its loads. Decode partitions KV rows across GPU blocks,
then merges their softmax sums. Queries, keys, values, and accumulators remain F32.
Decode captures one CUDA graph per active sequence. Device inputs select the
next token, RoPE position, and valid KV length on each replay.
Single-token prefill uses this decode graph too.
Graph lifetimes retain the model and sequence buffers they reference.
Speculative decoding is not implemented.

Model architecture and equations follow the
[official checkpoint](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
and [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py).

## Correctness checks

The small fixture and 128-token full-checkpoint reference tests pass with
maximum absolute logit error below `2e-4`. They cover sequential execution, batched
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

Measured on GB10 with the pinned checkpoint, greedy sampling, and one request at
a time. Each fresh server handled one cold request and one repeat, with a
96-token completion limit. Model loading and profiler instrumentation are excluded.
The baseline is commit `4b310f2`, which already had batched prefill, graphs, and
prefix caching. Current runs use the catalogue's 1,024-token prompt chunks.

| Prompt | Baseline prefill | Current prefill | Baseline decode | Current decode |
| --- | ---: | ---: | ---: | ---: |
| 4,864 tokens | 684 tok/s | 1,908 tok/s | 52.63 tok/s | 61.38 tok/s |
| 14,984 tokens | 525 tok/s | 1,729 tok/s | 36.07 tok/s | 56.39 tok/s |

Cold time to first token fell from 7.14 to 2.56 seconds at 5K, and from 28.59 to
8.71 seconds at 15K. Repeated-prompt time to first token was 36 ms and 71 ms,
respectively. Each repeat restored all but the final prompt token.

The 15K run generated the same 96 tokens as the baseline. Cold and repeated
requests also matched. The completion limit truncates reasoning in this timing
workload. A separate long-context suffix request restored 14,336 tokens and
processed 656 new tokens at 1,255 tok/s. Its first token arrived in 574 ms, and
its completed answer correctly identified the kestrel and the eastward stream.

See [the performance analysis](lfm2-performance.md) for hardware ceilings,
rejected experiments, numerical checks, and remaining costs.

Run the workload against a freshly started server:

```sh
python3 scripts/lfm2-request-bench.py --repetitions 220 \
  --output /tmp/lfm2-request-timing.json
# Use --repetitions 680 for the 14,984-token prompt.
```
