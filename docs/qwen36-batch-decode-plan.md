# Qwen3.6 batched decode plan

## Decision

Build and measure a real batched decode API before designing the request
scheduler. The scheduler must consume a proven model primitive; it must not
define batch compatibility, state layout, or performance assumptions on the
model's behalf.

The first target is homogeneous greedy decode for a small fixed batch. Prompt
setup may remain sequential, but the decode tick must execute as one batched
model operation. A loop over `decode_one_token` is a correctness scaffold, not
a batch implementation and must not be benchmarked or presented as one.

Scheduler and OpenAI-compatible HTTP work begin only after the batch API has:

- a stable ownership and state contract;
- correctness against independent single-sequence decode;
- measured memory use;
- measured aggregate throughput and per-sequence latency; and
- at least one batch size greater than one that produces a repeatable aggregate
  throughput improvement.

## Current baseline

The current Qwen3.6 path is specialized for one token and one sequence:

```rust
pub fn decode_one_token(
    &self,
    state: &mut Qwen36DecodeState,
    token_id: u32,
) -> Result<Qwen36NextToken>;
```

`Qwen36DecodeState` owns one CUDA stream, one set of per-layer workspaces, ten
full-attention K/V caches, thirty Gated Delta Net recurrent states, token and
position scalars, LM-head scratch, and decode graphs captured against those
addresses.

The relevant model shape is:

| Property | Qwen3.6 value |
| --- | ---: |
| Layers | 40 |
| Linear-attention layers | 30 |
| Full-attention layers | 10 |
| Hidden width | 2,048 |
| KV heads | 2 |
| Attention head width | 256 |
| Routed experts | 256 |
| Experts per token | 8 |
| Expert intermediate width | 512 |
| Vocabulary | 248,320 |

The single-sequence path remains the reference implementation throughout this
work. It should not be removed or generalized prematurely.

## Initial batch contract

The first public experiment should be deliberately narrow:

- batch capacities `1`, `2`, `4`, and `8`;
- active rows are the dense prefix `0..batch_size`;
- all rows have the same cache length;
- all rows use greedy GPU top-1 output;
- every row has the same maximum context capacity;
- prompt setup runs per slot before timed decode; and
- decode state uses stable, batch-native allocations.

A starting API shape is:

```rust
pub struct Qwen36DecodeBatchState {
    max_batch: usize,
    max_tokens: usize,
    // Batch-native device state and workspaces.
}

impl Qwen36TextModel {
    pub fn new_decode_batch_state(
        &self,
        max_batch: usize,
        max_tokens: usize,
    ) -> Result<Qwen36DecodeBatchState>;

    pub fn prefill_batch_slot(
        &self,
        state: &mut Qwen36DecodeBatchState,
        slot: usize,
        prompt: &[u32],
    ) -> Result<Qwen36NextToken>;

    pub fn decode_batch_top1(
        &self,
        state: &mut Qwen36DecodeBatchState,
        token_ids: &[u32],
    ) -> Result<Vec<Qwen36NextToken>>;
}
```

The exact Rust surface may change as kernel constraints become concrete. The
important invariants are that state is batch-native, decode receives all rows
in one call, and batch size is explicit.

Do not start with `&mut [&mut Qwen36DecodeState]`. That shape preserves
independent allocations and captured graphs, encouraging pointer-table loops
instead of contiguous batched work. The eventual scheduler should own slots in
a batch-state pool rather than N unrelated decode states.

## State layout

Use the slot dimension explicitly in persistent and scratch storage:

| State | Proposed logical layout |
| --- | --- |
| Input token IDs | `[B]` |
| Positions/cache lengths | `[B]` |
| Hidden and residual rows | `[B, hidden]` |
| GDN convolution state | `[linear_layer, B, qkv_width, conv_history]` |
| GDN recurrent state | `[linear_layer, B, value_head, d, d]` |
| Full-attention K/V | `[full_layer, B, max_tokens, kv_width]` |
| Router output | `[B, experts]` |
| Routed slots | `[B, top_k]` |
| Routed gate/up | `[B, top_k, 2 * intermediate]` |
| Routed down inputs | `[B, top_k, intermediate]` |
| LM-head scratch/results | `[B, ...]` |

`[B, max_tokens, kv_width]` keeps each sequence's cache rows contiguous and is
compatible with a future slot/block table. Do not introduce a paged cache in
the first implementation; expose enough shape metadata that replacing the
backing layout later does not affect the decode call.

Add an exact `device_bytes()` or allocation report for the batch state. Batch
capacity and context capacity must be visible costs, not implicit multipliers.

## Measurement phase 0: establish the baseline

Before changing kernels, record the current single-sequence release baseline:

```sh
cargo run --release -p infer --bin qwen-bench -- \
    --model models/qwen3.6-35b-a3-nvfp4 \
    --prompt "Hello world, this is a benchmark." \
    --decode-tokens 200 \
    --warmup-repeats 1 \
    --repeats 3 \
    --temperature 0
```

Record both the normal end-to-end result and `--profile-decode` stage timings.
The profiled result is diagnostic and must not be compared directly with the
normal throughput result.

The baseline report should include:

- median and range of decode tokens/sec;
- median milliseconds per token;
- stage timing proportions;
- process/device allocation requested by model and decode state; and
- exact commit and environment.

## Measurement phase 1: kernel viability

Add focused micromeasures before integrating each batched kernel. Every bench
must run batch sizes `1`, `2`, `4`, and `8`, validate output before timing, and
report both tick latency and aggregate rows/sec.

### Linear-attention projections

Measure the real Qwen3.6 FP8 projection shapes with an `[B, hidden]` input.
Start with the QKV, Z, and output projections because their weight reuse is
shared by every sequence regardless of expert routing.

Evidence gate:

- confirm the implementation performs one batched matrix operation;
- compare batch-1 output against the current vector path;
- compare all batch rows against repeated CPU/reference projections; and
- retain the batched path only if aggregate rows/sec improves.

### Gated Delta Net kernels

Extend QKV preparation, convolution update, gate preparation, recurrent update,
gated RMSNorm, and output preparation with a slot dimension. State writes must
be disjoint by slot.

Evidence gate:

- run at least 100 recurrent decode steps;
- compare every slot against an independent single-sequence state;
- permute input rows and verify the same permutation in outputs; and
- include a reset/reuse test to catch state leakage between requests.

### Full attention

Add batch and per-slot position/cache-length inputs to the indexed attention
path. The first version may require equal cache lengths, but it must store
positions as a vector so ragged support does not require changing the public
decode contract.

Evidence gate:

- compare short and long cache lengths with the single-sequence path;
- verify K/V appends land in the correct slot and position;
- test different token data in every row; and
- include a cross-slot contamination test.

### Routed MoE

This is the least predictable batching stage because each token may select a
different set of eight experts.

Extend the route and workspace layout from `top_k` routed entries to
`B * top_k`. Benchmark at least two routing distributions:

1. maximum overlap, where rows select the same experts; and
2. low overlap, using routes from real prompts or deterministic disjoint sets.

The Marlin gate/up path, SiLU/quantization, SM12x down path, and weighted
accumulation must all consume the full routed batch. Do not claim a MoE batch
win from the maximum-overlap case alone.

Evidence gate:

- exact route/result correspondence for every row;
- correctness with repeated and distinct expert IDs;
- batch-1 parity with the current routed path;
- separate overlap and low-overlap performance results; and
- no per-token host dispatch inside the timed region.

### LM head

Add a batched greedy top-1 operation over `[B, hidden]`. It should reduce one
token and logit per row without materializing `[B, vocab]` logits.

Full-logit output for temperature/top-p sampling is a later mode. Keep greedy
and sampled batch contracts distinct until a device sampler exists.

## Integration phase 2: batch-native layer execution

Create `crates/infer/src/qwen3/qwen36_batch.rs` rather than adding another
large section to `qwen36.rs`.

Integrate in model order:

1. batched embedding gather;
2. per-layer input normalization;
3. linear-attention or full-attention batched step;
4. residual and FFN normalization;
5. batched router and routed MoE;
6. shared expert and final residual;
7. final normalization; and
8. batched LM-head top-1.

Keep eager execution until end-to-end correctness and timing are established.
CUDA graph capture is phase 3 because graphs constrain addresses, active batch
sizes, and launch structure. Capture one graph per proven fixed batch size only
after eager execution wins.

## Correctness plan

### Batch-1 equivalence

For the same model, prompt, token, and position:

- compare the chosen token and winning logit;
- compare hidden output after each layer within the established tolerance;
- compare full-attention K/V append rows;
- compare GDN convolution and recurrent states; and
- compare router indices and weights.

Batch 1 must remain within the current numerical contract before larger batches
are evaluated.

### N independent sequences

For `B = 2, 4, 8`, initialize B independent single-sequence states and one
batch state from the same prompts. Decode them for at least 100 tokens and
compare each step.

Cover:

- identical prompts;
- different prompts with equal tokenized length;
- different generated routes;
- EOS in one row while other rows continue, once active masks are added; and
- batch-state reset followed by a second unrelated request set.

### Failure and bounds

Test zero batch, batch greater than capacity, token outside vocabulary, context
overflow, duplicate slot assignment, invalid active rows, and mismatched input
lengths. Fail before launching CUDA work.

## End-to-end micromeasure

Add `crates/infer/benches/qwen36_decode_batch.rs` using `micromeasure`.

The timed region begins after model loading, prompt setup, cache allocation,
and correctness validation. Run enough decode ticks for stable CUDA-event and
wall-clock measurements.

Report:

| Metric | Meaning |
| --- | --- |
| Tick latency | Wall/CUDA time for one batch decode step |
| Aggregate tokens/sec | `B / tick_seconds` |
| Per-sequence tokens/sec | `1 / tick_seconds` |
| Speedup over B independent calls | Actual amortization from batching |
| State bytes | Persistent batch-state allocation |
| Peak process/device memory | End-to-end capacity cost |

The result table must include batch `1`, the current single-token API, and each
candidate larger batch. Compare normal eager execution separately from any
later graph-captured path.

## Scheduler handoff contract

Do not begin scheduler implementation until the batch work answers:

- supported and worthwhile batch sizes;
- whether cache lengths must be homogeneous;
- whether active rows must be dense;
- whether rows can finish independently;
- state allocation and reset cost;
- graph/address stability requirements;
- greedy versus sampled output modes; and
- the memory admission formula.

Write these answers into this document as measured results. They become the
scheduler's compatibility and admission rules.

The scheduler can then adopt the useful `tinfer` shape:

```text
Waiting -> Prefilling -> Decoding -> Finished
```

but it will group and admit requests according to the proven Eider batch
contract rather than assumptions inherited from the CPU implementation.

## Stop conditions

Stop or redirect the batch work when evidence shows any of the following:

- no batch size greater than one improves aggregate end-to-end throughput;
- improvements occur only in isolated kernels and disappear end to end;
- state memory makes the batch unusable within the GB10 unified-memory budget;
- low-overlap MoE routing erases the gain seen in synthetic overlap cases; or
- per-sequence latency exceeds the intended interactive service target without
  a compensating aggregate-throughput requirement.

In that case, retain the single-sequence decoder, build a serialized API worker,
and revisit batching only with a more specific kernel or serving workload.

## Deliverables

1. Single-sequence baseline report.
2. Focused batch micromeasures for projections, GDN, attention, MoE, and LM
   head.
3. `Qwen36DecodeBatchState` with allocation reporting and reset coverage.
4. End-to-end `decode_batch_top1` with batch-1 and N-sequence correctness tests.
5. `qwen36_decode_batch` micromeasure and result table.
6. Documented scheduler handoff contract based on measured behavior.
7. Only then, a scheduler and OpenAI-compatible serving plan.
