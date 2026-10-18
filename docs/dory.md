# Dory 1.0 4B NVFP4

Eider loads NVIDIA's [Dory NVFP4 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)
at revision `41f02335e1a1ad3289b8b32485e2265cbe97cfdf`.
The runtime uses the W4A16 weights without activation quantization.
Embeddings and the separate output head retain BF16 weights.

Dory uses nGPT L2 normalization, learned interpolation residuals, gated attention,
and input reinjection between loops. Its 72 physical sublayers contain 36 attention/MLP pairs.
Two loops execute 112 sublayers with shared weights and separate attention caches.
This matches the checkpoint's recommended `per_loop` configuration.
The runtime rejects `last_loop` cache semantics.

## Start

Start the server with the downloaded checkpoint:

```sh
scripts/run-eider-dory.sh
```

The default is two loops. For one loop, use:

```sh
scripts/run-eider-dory.sh --dory-loops 1
```

Start Pi in a second terminal:

```sh
scripts/run-pi-eider-dory.sh
```

The catalogue ID is `dory1.0-4b-nvfp4`.
API requests use `eider-dory1.0-4b`.
Both `/v1/responses` and `/v1/chat/completions` use the checkpoint template and shared output parser.
The launcher uses one active sequence and prompt chunks of 256 tokens.
The configured context limit is 262,144 tokens.

## Execution

Prefill uses batched BF16 matrix products with F32 accumulation.
A reusable workspace expands one NVFP4 projection at a time.
Decode reads packed NVFP4 weights directly and replays a CUDA graph.
Normalization and residual operations preserve the reference model's BF16 rounding boundaries.

KV storage is BF16. Each loop has independent causal attention history.
Request capacity includes the prompt and maximum output length.
At two loops, KV storage requires 224 KiB per reserved token, before workspace and prefix retention.
A full 256K allocation therefore requires 56 GiB of KV storage.
The configured limit does not imply a full-context quality or performance evaluation.

The prefix cache retains regular prompt endpoints and the prompt before its final token.
Sliding attention snapshots omit expired keys. Full attention snapshots retain their complete prefix.
`--retained-prefix-gib` bounds retained device memory. Zero disables retention.

## GB10 measurements

These single-request runs used a 568-token prompt, 128 generated tokens,
256-token chunks, and greedy decoding. Model loading is excluded.
Prefill includes initial matrix-plan construction. Decode includes graph capture.

| Loops | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| 1 | 1,458 | 52.1 |
| 2 | 1,017 | 35.4 |

The attention micromeasure uses 128 queries after a 1,024-token prefix.
Its median CUDA time fell from 2.283 ms to 1.315 ms after parallelization and BF16 KV storage.
The same 568-token generation case improved from 24.5 to 52.1 tok/s at one loop.
At two loops, decode improved from 16.2 to 35.4 tok/s.
These measurements do not establish long-context throughput or task accuracy.

## Decisions

`POST /v1/decisions` supports `noul`, `choice`, and `score` questions.
The runtime processes the shared prefix once, then processes each question suffix separately.
Questions reuse the same KV allocation and cannot see sibling suffixes.
The readout projects only the 64 label rows and applies their checkpoint logit scales.
It generates no reasoning or answer text.

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eider-dory1.0-4b",
    "state": "The parcel arrived yesterday. It contains a red book.",
    "questions": {
      "arrived": {"type": "noul", "instructions": "Has the parcel arrived?"},
      "colour": {
        "type": "choice",
        "instructions": "What colour is the book?",
        "criteria": {"red": null, "blue": null}
      }
    }
  }'
```

The two-question example took 323–325 ms in two HTTP runs at two loops.
Decision probabilities are uncalibrated. Changing the loop count changes the scoring model.
Suffixes run sequentially in this implementation. Packed multi-question execution remains future work.

## Validation

The independent PyTorch comparison uses the local checkpoint and equations from
[vLLM's Dory implementation](https://github.com/vllm-project/vllm/blob/55076950b6c155f384d6766d63563c3bc20a0e3f/vllm/model_executor/models/dory.py).
The final two-loop comparison has logit cosine similarity 0.99913 and the same top token
on a 27-token arithmetic prompt. This is numerical coverage, not a task-quality evaluation.
CUDA checks cover BF16 residual boundaries, causal attention, and sliding-window exclusion.
The checkpoint regression covers one and two loops, chunked prefill, decode graphs,
prefix restoration, selected-label readout, and isolated decision suffixes.

```sh
DORY_MODEL=/path/to/snapshot cargo test --release -p eider-inference --lib \
  dory_chunk_graph_and_prefix_agree -- --ignored
cargo bench -p eider-cuda --bench dory_attention
```

For a direct generation run and reference comparison:

```sh
cargo run --release -p eider-inference --example dory_generate -- \
  /path/to/snapshot 2 128 256 'What is 7 times 8? Answer briefly.' /tmp/dory-logits.json
python3 scripts/dory-checkpoint-reference.py /path/to/snapshot /tmp/dory-logits.json --loops 2
```
