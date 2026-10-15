# Nemotron 3.5 VL LiLiCorr

Eider loads the released companion and supports one to seven proposals per
cycle. LiLiCorr is opt-in. The target and companion use the same prefix cache
and preserve Mamba state through partial acceptance.

## Checkpoint contract

[NVIDIA's NVFP4 LiLiCorr checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr)
uses revision `7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2`. Its single safetensors
file is 1,317,952,596 bytes, about 1.23 GiB.

| Component | Released configuration |
| --- | --- |
| Architecture | `LiLiCorrDraftModel`; the `qwen3` model type alone is insufficient |
| Draft block | Eight positions: one anchor and seven proposals |
| Target features | Layers `[1, 22, 43, 64, 85]` from the 88-layer target |
| Feature projection | 20,480 inputs to 4,096 outputs |
| Backbone | Five layers; hidden 4,096; FFN 12,288 |
| Attention | 32 query heads, eight KV heads, head dimension 128 |
| Context | 8,192-position sliding window; noncausal draft attention |
| Position encoding | RoPE theta 1,000,000 |
| Dynamic convolution | Two taps, group size 16, around attention and MLP |
| Candidate head | Top eight candidates per slot; 56 candidate nodes |
| Correlator | Two layers, hidden/factor width 1,024, eight attention heads |
| Vocabulary | 131,072; mask token 990; its own NVFP4 LM head |

Tensor names start with `layers.`, `fc.`, `lilicorr.`, and `lm_head.`; they have
no `model.` prefix. The header contains 178 tensors. Attention weights and several
correlator projections remain BF16. FFNs, the feature projection, the LM head, and
some correlator linears use ModelOpt `W4A16_NVFP4`, with K16 scales. Loading this
recipe must preserve BF16 activations; it does not request W4A4 quantization.

## Correlator behaviour

The [vLLM implementation](https://github.com/vllm-project/vllm/blob/73a78e6f1f38e280986b81e0f2a9aa5e1ee6fe47/vllm/model_executor/models/lilicorr.py)
uses a DFlash backbone with grouped dynamic convolution. Its correlation head
consumes token embeddings, draft hidden states, and candidate log probabilities.

Each candidate gets five features: log probability, probability, distance from
the best log probability, fractional rank, and a top-rank indicator. A LayerNorm
and MLP transform those features. The head adds projected token and draft states,
plus learned slot and rank embeddings.

Two transformer layers attend over all 56 candidates. Learned per-head biases
encode relative slot distance and same-slot membership. These layers have no
persistent KV cache. The output produces normalised incoming and outgoing factor
vectors, conditioned on a projected target anchor. Dot products produce anchor
scores and adjacent-slot transition scores, scaled by eight.

Preserve full-vocabulary log-softmax values before top-eight selection. Ranking,
LayerNorm biases, the anchor term, and the slot-bias offset are trained semantics.
Shorter proposals must retain the original slot-bias offset and use the trained
prefix of the lattice.

## Eider integration

Download the companion into the normal Hugging Face cache:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr \
  --revision 7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
EIDER_LILICORR=1 scripts/run-eider-nemotron35.sh
```

The launcher resolves that pinned snapshot and enables seven proposals. Set
`LILICORR_DIR` to select a different local snapshot path. Explicit server arguments
still take precedence. `EIDER_SPECULATIVE_DRAFTS=0` disables proposals; values one
through seven select the proposal length. The direct server option is
`--lilicorr-dir DIR`. It defaults to seven proposals unless overridden.

`crates/infer/src/nemotron3/lilicorr/` validates all 178 consumed tensors and
implements the backbone, correlator, and sequential candidate walk. Target
features come from committed prefill, decode, or verification rows. Rejected
rows never enter the draft context. Prefix snapshots retain the draft ring
cache and anchor alongside target Mamba and attention state.

Greedy verification downloads only row argmax IDs. Other sampling policies use
full target logits and the realised sparse proposal probabilities. Acceptance
uses `min(1, p/q)`; rejection samples from `max(p-q, 0)`. Target probabilities
include temperature, top-k, top-p, presence penalties, and frequency penalties.
The proposal walk uses temperature over the learned conditional scores. It does
not add the unary draft logits to those scores.

Native W4A16 dense batches compute eight activation rows per MMA tile. Sliding
draft attention splits the context into 256-key partitions and merges their
softmax results. Both paths retain BF16 activation semantics.

Proposal projections, dynamic convolutions, and the correlation head use CUDA
graphs after their first execution. Target verification captures segments between
attention layers. Positions and attention remain explicit launches so graph
replay cannot retain a previous sequence position. All device work uses the
model stream. The scheduler retains the full verification shape and one tail
shape, which bounds recurrent snapshot storage.

Each live draft context uses about 320 MiB for its F32-backed, BF16-valued KV
window. Eight FP32 Mamba snapshot slots add 1.25 GiB per full verification
workspace. Prefix snapshots retain only the populated draft window. Companion
weights, projection scratch, and a shortened verification workspace add further
memory; these costs are separate from the target weights.

Nemotron prepares the checkpoint's NVFP4 dense weights once for batched
projections. The shared experts add about 3.3 GiB of prepared BF16 weights.
Single-token decode retains packed weights. Explicit BF16/FP8-to-NVFP4 storage
policies retain their compact representation and expand batch weights in scratch.

## Validation

`scripts/nemotron-lilicorr-reference.py` produces a CPU PyTorch reference from
local checkpoints. It can consume real target features saved by the ignored
`capture_lilicorr_reference_context` test. It does not download or alter weights.

The checkpoint comparison checks context projections, individual backbone layers,
the independent correlation head, full draft logits, candidate overlap, and exact
prefix replay. The full-target check compares 64 generated tokens with ordinary
decoding and exercises both seven-token and shortened proposals. The serving
check covers prefix reuse, seeded stochastic sampling with penalties, and queued
and active cancellation.

```sh
# Set these to local immutable HF snapshots.
export NEMOTRON_MODEL_DIR=/path/to/target
export LILICORR_DIR=/path/to/companion
cargo test --release -p eider-inference --lib \
  lilicorr_commits_match_serial_target -- --ignored
cargo test --release -p eider-inference --lib \
  lilicorr_serving_restores_prefix_and_preserves_request_policies -- --ignored
```

Each test loads the full target. Run them separately with the server stopped.
For the serving check, `LILICORR_PROMPT` selects a text prompt file and
`LILICORR_SERVING_REPORT` writes request timings and acceptance counts to JSON.
`LILICORR_TEST_DRAFTS` selects the proposal count for that check; it defaults to seven.

## GB10 measurements

The 2026-10-15 serving-engine replay used one code-review request with 11,298
prompt tokens and 128 output tokens. A repeated request restored 11,264 prompt
tokens. These are single-run measurements with the released NVFP4 target and
companion, without an HTTP adapter.

| Mode | Cold TTFT | Cached TTFT | Cached decode | Accepted drafts/cycle |
| --- | ---: | ---: | ---: | ---: |
| Initial target only, across both runs | 19.75–20.72 s | 547–549 ms | 15.12–15.25 tok/s | — |
| Initial three LiLiCorr proposals | 22.17 s | 602 ms | 14.93 tok/s | 2.10 |
| Initial seven LiLiCorr proposals | 22.57 s | 605 ms | 14.34 tok/s | 2.71 |
| Updated target only | 19.01 s | 530 ms | 15.12 tok/s | — |
| Updated seven LiLiCorr proposals | 19.23 s | 554 ms | 18.10 tok/s | 2.71 |

The updated run uses batched W4A16, split-context attention, and prepared target
dense weights. Cached request time fell from 9.46 seconds to 7.57 seconds.
Decode improved 26% over the initial seven-proposal path and 20% over the paired
target-only run. Both versions accepted 92 drafts across 34 cycles.
Prefix replay, seeded sampling, and cancellation passed. These throughput figures
use greedy decoding; they do not establish the same gain for stochastic requests.
LiLiCorr remains opt-in; the launcher selects seven proposals.

The focused `lilicorr_kernels` benchmark checks dense outputs against the prior
per-row kernel and attention against the serial implementation. It covers partial
MMA tiles, empty and short contexts, and a wrapped 8,192-token ring.
The candidate LM head fell from 11.74 ms to 1.46 ms; attention fell from 3.57 ms
to 1.24 ms per layer. Prepared shared projections preserve exact outputs and
measured 2.1–2.5 times faster than projection with repeated expansion.

```sh
cargo bench -p eider-cuda --bench lilicorr_kernels
cargo bench -p eider-cuda --bench nemotron35_dense -- 'prepared 8 rows'
```

The `nemotron35_experts` micromeasure also covers verification sizes of two,
four, and eight rows. At eight rows and 22 routes per row, median up/down times
were 1.55/1.50 ms for SIMT and 3.13/2.66 ms for the sorted prefill tensor path.
The comparison includes sorting and checks all outputs against SIMT first.
The runtime therefore retains SIMT for these small verification batches.

The model card reports coding acceptance lengths of 5.65 at temperature zero and
5.29 at temperature one with seven speculative tokens. These are reference
acceptance results, not Eider throughput measurements. They justify investigating
this path; whole-request measurements determine its net benefit on GB10.
The table above excludes the guaranteed next token from accepted drafts per cycle.
Adding that token gives 3.71 under vLLM's acceptance-length convention.

The algorithm is described in the [LiLiCorr paper](https://arxiv.org/abs/2608.20530).
The [llama.cpp conversion](https://github.com/ggml-org/llama.cpp/blob/e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3/conversion/nemotron.py)
and [draft graph](https://github.com/ggml-org/llama.cpp/blob/e64207bab6c2989a4ae5aa8b9e0cbe700e135ee3/src/models/dflash.cpp)
provide another implementation to compare against.
