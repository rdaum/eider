# LFM3-1.7B

Eider loads the [official Liquid AI NVFP4 checkpoint](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4)
at revision `b30025bbbbcbf62f6c110b40d51d36af6579f6a2`.
This checkpoint contains the chat model. The original `LFM3-1.7B-Base` checkpoint
contains the pretrained base model.

Feed-forward projections use native NVFP4 weights and W4A4 matrix multiplication.
Each projection uses its own checkpoint weight and activation scales.
Attention, convolution, embeddings, and norms retain their BF16 checkpoint weights.
Operator boundaries round to BF16, including residual additions and convolution products.
Accumulation uses F32. Scratch, convolution history, and KV storage use F32 containers
for these BF16 values.

The 24 layers contain 13 gated convolutions, six full-attention layers, and five
sliding-attention layers. Convolutions retain six previous values for their seven-tap filter.
Sliding attention includes the current token and up to 255 previous tokens.
Full attention uses partial RoPE with zero tail frequencies and pairs across the full head dimension.
Both attention types apply Q/K norms and per-head sigmoid gates.
Sandwich norms precede both residual additions.

## Start the server

```sh
scripts/run-eider-lfm3.sh
```

The launcher downloads the pinned checkpoint when the cache lacks its files.
For an offline start after the download, add `--offline`.
API requests use `eider-lfm3-1.7b`.

Start Pi in another terminal:

```sh
scripts/run-pi-eider-lfm3.sh
```

Pi defaults to Responses with low thinking. Set `PI_EIDER_PROVIDER=eider-chat`
to use Chat Completions, or `PI_EIDER_THINKING=off` to disable thinking.
Both launchers accept extra command-line arguments and work from any directory.

Both `/v1/responses` and `/v1/chat/completions` use the same inference service.
The service uses the checkpoint chat template and the shared reasoning/tool parser.

The catalogue defaults to 32,768 context tokens and prompt chunks of 256 tokens.
The checkpoint declares a maximum context of 262,144 tokens.
Each request owns its convolution history and attention cache.
Prefix snapshots retain both types of state, subject to `--retained-prefix-gib`.
The initial implementation allocates KV capacity for the request length, including sliding layers.
Decode uses the same execution path as a one-token prompt chunk.

The loader also accepts the original BF16 checkpoint through `--model-dir`.

## Decisions

`POST /v1/decisions` supports `noul`, `choice`, and `score` questions.
Each request reads the shared state once, then evaluates isolated question suffixes.
The readout projects only the 64 validated label tokens.
Decision sequences do not allocate or compute full-vocabulary logits.

```sh
curl -fsS http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eider-lfm3-1.7b",
    "state": "The parcel arrived yesterday. It contains a red book.",
    "questions": {
      "arrived": {"type": "noul", "instructions": "Has the parcel arrived?"},
      "colour": {
        "type": "choice",
        "instructions": "What colour is the book?",
        "criteria": {"red": null, "blue": null, "green": null}
      }
    }
  }'
```

The runtime follows Dory's decision execution pattern: combined prefix/suffix
passes, packed suffix waves, reusable sequence storage, and bounded CUDA graphs
for repeated shapes. Each branch shares the immutable KV prefix and starts from
the same convolution snapshot. BF16 projections and NVFP4 FFNs retain each
segment's reduction shape. This keeps cache hits and sibling branches from
changing decision logits through different NVFP4 rounding.

The LFM3 launcher defaults to one active sequence and 2,048 prefill tokens.
These settings retain the decision allocation between requests and fit larger waves.
Use `--decision-branch-capacity` to change the wave size; the server default is eight.
All launcher settings accept command-line overrides.
Prefix retention uses the configured cache budget. Set `--retained-prefix-gib 0`
to measure requests without retained-prefix reuse.

The checkpoint is the general chat model. Decision scores use the existing
label-normalization and calibration interface; they are not calibrated by default.

The GPU regressions compare compact and full readouts, packed and serial branches,
sibling independence, cancellation, retained prefixes, and graph invalidation.
Run them separately on an idle GPU:

```sh
LFM3_REFERENCE_DIR=/tmp/lfm3-reference cargo test -p eider-inference --lib \
  lfm3_compact_decisions -- --ignored
LFM3_MODEL=/path/to/checkpoint cargo test --release -p eider-inference --lib \
  lfm3_packed_decisions -- --ignored --nocapture
LFM3_DECISION_MODEL_DIR=/path/to/final-nvfp4 cargo test --release -p eider-inference --lib \
  lfm3_nvfp4_decisions_match_after_prefix_cache_hit -- --ignored
```

On GB10, the published checkpoint passed the scheduler comparison on 2026-10-27.
Warm request times with retained-prefix reuse disabled were:

| Questions | Serial suffixes | Packed suffixes with combined prefix |
| --- | ---: | ---: |
| 1 | 28.9 ms | 22.3 ms |
| 8 | 121.3 ms | 97.6 ms |
| 55 | 751.9 ms | 468.2 ms |

These are single warm measurements from the regression fixture used by
`scripts/questions.sh`, with eight branches per wave and 2,048 prefill tokens.
They measure the complete scheduler request, including prefix processing and readout.
They predate the 2026-10-28 fix that preserves NVFP4 segment shapes.

The live HTTP checks covered all three question types, 64-option answers,
64-question groups, validation errors, and chat after decisions.
Repeated three-question requests with a retained prefix took 31–33 ms over HTTP.

## Validation

The independent PyTorch reference follows the equations in
[Liquid's implementation](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4/blob/b30025bbbbcbf62f6c110b40d51d36af6579f6a2/modeling_lfm3.py).
It reproduces BF16 operator boundaries and simulates NVFP4 activation quantization.
It does not require Transformers custom-code execution.

Generate a small fixture, then run the GPU regression:

```sh
python3 scripts/lfm3-reference.py --nvfp4 --output /tmp/lfm3-reference
LFM3_REFERENCE_DIR=/tmp/lfm3-reference cargo test -p eider-inference --lib \
  lfm3_reference -- --ignored
```

Omit `--nvfp4` to validate BF16 projections.
The regression covers chunked prefill, decode, interleaved sequences, sliding-window
boundaries, invalid tokens, capacity limits, and prefix restoration.
The fixture uses 128-dimensional heads, a seven-tap convolution, and distinct projection scales.
Discrete dyadic BF16 weights limit differences from BLAS reduction order.
The quantizer regression compares GPU packed values and scales with the CPU encoder.
It covers BF16 inputs near FP4 rounding boundaries.

```sh
cargo test -p eider-cuda --lib activation_quantization_matches_host -- --ignored
cargo bench -p eider-cuda --bench lfm3_operators
```

For full-checkpoint logits, use `scripts/lfm3-reference.py --checkpoint PATH`
with `--output DIR` and `--tokens ID...`.
The `lfm3-logits` Rust example accepts `CHECKPOINT OUTPUT_JSON TOKEN...`.
It writes the final prompt logits, 32 greedy tokens, and elapsed prefill/decode times.

## GB10 validation, 2026-10-27

Both fixture formats passed the GPU regression. Interleaving and restored-state
comparisons were exact. Reference logits use BF16 tolerances: `atol=0.02`,
`rtol=0.01`, and RMS error below `0.01`.
The packed FP4 values and scales matched the CPU encoder exactly.
The shared quantizer uses precise division to preserve rounding at FP4 boundaries.

The pinned checkpoint passed live checks for both HTTP APIs and their streams,
reasoning output, tool calls, tool history, stop strings, concurrent requests,
prefix reuse, and cancellation. A 913-token prompt exercised multiple prefill
chunks and sliding-window boundaries.

One weather-tool request used 79 prompt tokens and produced 18 output tokens.
It measured 32.6 ms to the first token, 79.7 tokens/s during decode, and
73.2 tokens/s over the whole request. These are short smoke measurements.
The six-operator benchmark measured 28.7 microseconds for one row and
133.5 microseconds for 256 rows. It excludes matrix multiplication.

Full-checkpoint logits are not identical to the independent PyTorch reference.
For “What is the capital of France? Answer briefly.” with thinking disabled,
the final prompt distribution had KL divergence `0.00055` from the CPU reference.
Raw-logit RMS error was `0.826`, with cosine similarity `0.9815`.
Both selected “The”; Eider generated “The capital of France is Paris.”
This comparison does not establish broad model-quality parity.
