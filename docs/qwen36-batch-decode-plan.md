# Qwen3.6 batched decode

## Decision

Eider owns a model-level batched decode primitive before it owns a request
scheduler. The scheduler selects runnable sequences; the model API defines how
their persistent state is advanced and how shared execution storage is reused.

The API does not expose fixed slots, equal cache lengths, greedy-only output,
or a homogeneous context-capacity requirement. Those would turn current kernel
details into scheduler policy.

## Decode contract

Persistent sequence state and reusable batch execution storage are separate:

```rust
pub struct Qwen36DecodeRow<'a> {
    pub token_id: u32,
    pub state: &'a mut Qwen36SequenceState,
}

impl Qwen36TextModel {
    pub fn new_sequence_state(
        &self,
        max_tokens: usize,
    ) -> Result<Qwen36SequenceState>;

    pub fn new_decode_batch_workspace(
        &self,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36DecodeBatchWorkspace>;

    pub fn decode_batch<'w>(
        &self,
        workspace: &'w mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
    ) -> Result<Qwen36DecodedBatch<'w>>;
}
```

Each call has the following semantics:

- rows may be added, removed, or reordered between ticks;
- every row carries its own position and context capacity;
- a sequence capacity may be smaller than the workspace context capacity;
- output rows preserve the caller's row order;
- launch padding up to workspace capacity is private execution state;
- logits remain device-resident until the caller requests full host logits or
  batched GPU top-1; and
- sequence positions advance only for rows included in the call.

Sequence state and workspaces are tied to the model instance that created them.
The call rejects foreign state, foreign workspaces, invalid tokens, context
overflow, inconsistent full-attention cache state, empty batches, and batches
larger than workspace capacity before model execution.

## Ownership and layout

`Qwen36SequenceState` owns only state that persists for one request:

- the logical decode position and maximum context;
- thirty per-layer convolution and Gated Delta Net recurrent states; and
- ten per-layer compact SM12x FP4 K/V caches.

`Qwen36DecodeBatchWorkspace` owns storage shared by whichever rows a scheduler
selects for the current tick:

- token and position vectors;
- row-major hidden, residual, normalization, projection, and LM-head buffers;
- FP8 and NVFP4 activation scratch;
- GDN state-pointer tables assembled from the selected sequence rows;
- routed-MoE route, Marlin gate/up, SM12x down, and shared-expert storage; and
- one compact-attention scratch workspace reused serially across active rows.

Workspace creation performs all recurring host and device allocation. The
decode hot path reuses host token, position, and GDN pointer arrays; it does not
allocate a vector per tick or per layer.

## Execution path

One call executes the model in layer order:

1. batched BF16 embedding gather;
2. row-wise RMSNorm;
3. batched linear-attention projections and GDN state updates, or batched
   full-attention projections followed by per-sequence compact-cache append and
   attention;
4. row-wise residual and FFN normalization;
5. batched router top-k, Marlin routed gate/up, SM12x routed down, shared
   expert, and fused FFN finalization;
6. final RMSNorm and the batched LM head; and
7. either device logits, host logits for sampling, or GPU top-1 results.

The existing generation session remains on the graph-captured single-row path
until the batch path has throughput measurements and batched implementations
for every supported expert-storage variant. The scheduler can consume the new
contract directly for the fast NVFP4 model plan.

## Correctness evidence

Focused CUDA tests compare batched operations with independent rows for:

- BF16 projections;
- scalar-scaled FP8 W8A16 projections;
- NVFP4 W4A16 projections;
- full-attention Q/K normalization and gate splitting;
- MoE top-k routing;
- convolution and GDN recurrent updates; and
- compact K/V append and attention from offsets in larger dense buffers.

The model-level probe intentionally changes the schedule while comparing each
active sequence with an independently executed capacity-one batch:

```sh
cargo run --release -p infer --bin qwen36-batch-probe -- \
    models/qwen3.6-35b-a3-nvfp4 9707,3710 4 4
```

The four ticks cover two distinct inputs, a dropped sequence, ragged positions,
row reordering, and re-admission. All active logits match the independent
canonical path exactly, including after divergent state evolution and
re-admission.

The canonical batch kernels do not always produce bit-identical logits to the
older graph-captured single-row path because projection and MoE reduction
schedules differ. Current spot checks preserve top-1, but longer numerical and
generation-quality validation is still required before treating that as the
complete quality gate.

## Batch measurements

`qwen36_decode_batch` measures one complete scheduler-visible tick, including
GPU top-1 and synchronization. It validates every batched logit row against an
independent capacity-one decode before timing. The measured rows follow their
own greedy outputs, so routing is model-selected rather than synthetically
fixed.

On GB10 with the local `qwen3.6-35b-a3-nvfp4` checkpoint, 4,096-token sequence
capacity, a starting decode position of 128, and the worktree based on
`30e2543`:

The graph-captured production `decode_one_token()` path reaches 76.46 tokens/s
with a 12.871 ms median tick. The capacity-one batch API is not that baseline:
it is an eager execution path used to isolate the benefit of batching.

| Batch | Batched tick | Batched tokens/s | Canonical independent tick | Canonical independent tokens/s | Canonical speedup | Production speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.393 ms | 57.81 | 17.138 ms | 58.41 | 0.99x | 0.76x |
| 2 | 24.471 ms | 81.85 | 33.908 ms | 58.37 | 1.40x | 1.07x |
| 4 | 38.942 ms | 101.92 | 67.034 ms | 59.88 | 1.70x | 1.33x |
| 8 | 67.498 ms | 118.93 | 133.436 ms | 59.97 | 1.98x | 1.56x |

Exact owned device allocations were:

| Batch | Shared batch workspace | Request-owned sequence state |
| ---: | ---: | ---: |
| 1 | 4,424,732 bytes | 90,112,000 bytes |
| 2 | 8,345,396 bytes | 180,224,000 bytes |
| 4 | 16,252,260 bytes | 360,448,000 bytes |
| 8 | 32,065,988 bytes | 720,896,000 bytes |

Sequence state is 90,112,000 bytes per request at this context capacity. The
workspace grows by about 4 MB per batch row and is small beside persistent
request state. Batch 8 is therefore the throughput default, but its current
gain over optimized single-sequence production decode is 1.56x rather than the
1.98x canonical-batch amortization figure. Batch 4 remains a useful policy
point when a roughly 39 ms tick matters more than maximum aggregate throughput.

The batch implementation is currently eager. Recovering production-path graph
and launch efficiency is a later optimization because the scheduler contract
deliberately permits changing row membership, order, and positions.

The current workspace constructor supports the normal Marlin NVFP4 routed
gate/up, SM12x routed down, and NVFP4 shared-expert plan. Add true batched paths
for grouped or FP8 experts before moving mixed-storage Unsloth checkpoints off
the single-row decoder; do not implement that support as a host loop over
sequences.

## Scheduler handoff

The scheduler may rely on these semantic facts now:

- sequence state is independent of batch membership and row order;
- positions and cache lengths may be ragged;
- requests may leave and later rejoin a batch without moving state;
- workspace capacity is an execution limit, not a persistent slot layout; and
- greedy and sampled generation share the same decode operation.

The initial admission policy can use capacity 8, select available work without
waiting to fill the batch, and rotate runnable requests between ticks. This
captures the measured throughput gain without adding an artificial batching
delay or allowing a long-lived request to pin a workspace row.

The scheduler can then use the usual lifecycle:

```text
Waiting -> Prefilling -> Decoding -> Finished
```

without inheriting a fixed-slot or equal-length restriction from the first
kernel implementation.
