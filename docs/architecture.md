# Eider architecture

## Status

This document defines the target structure for Eider. It records a design
decision and the migration order; it does not claim the refactor is complete.

The first migration slice is complete. `eider-cuda` now rejects invalid device
representations, returns a loan for device-to-pinned-host readback, retains
captured-graph resources, and supplies typed device views. The DFlash2
projection uses those views. Physical SM12x page storage and Qwen sequence
state have moved out of `runtime` without a compatibility module. Qwen now
keeps persistent streams, workspaces, cache state, and retained DFlash state
in an inference-owned execution object. Qwen scheduler requests now retain
opaque sequence identities, while the execution object owns the live CUDA
sequences and GPU sampling-count buffers; batch leases restore that state on
every return path. The scheduler keeps request policy and queues.
Gemma 4 follows the same ownership rule: its chat service keeps scheduling and
output state, while the Gemma module retains live CUDA sequences behind opaque
identities and temporary batch leases.
Step-3.7 also retains its paged sequence and GPU sampling state behind opaque
identities in its model module. Its scheduler leases that state only for the
prefill, decode, and prefix-retention operations that use it.
Laguna uses the same handle-and-lease ownership boundary for its paged sequence
state across prefill, decode, prefix retention, cancellation, and completion.
BitNet retains its sequence state in a model-owned pool and leases it for each
prefill or decode operation; the service retains only request and output state.
Bonsai follows the same ownership rule for its sequence state and decode
workspace access.
Ling 3 retains its MLA and recurrent sequence state behind an opaque identity
and leases it for prefill and decode.
Nemotron 3 retains its hybrid attention and Mamba sequence state behind opaque
identities, with a batch lease for its batched prefill and speculative decode.
Flash Next retains each base sequence, native MTP state, speculative frontier,
and GPU sampling buffer as one execution-owned record behind an opaque identity.
DeepSeek V4 retains each target/MTP pair behind an opaque identity and leases
the pairs together when it builds ordinary, prefill, or speculative batches.
Muse Glimmer retains its sequence, including DFlash device state, behind an
opaque identity and leases it for prefill, prefix retention, and decode.
`eider-format` now owns GGUF indexing, GGML K-quant decoding, the
sharded safetensors index/cache, the versioned host-only NVFP4 artifact codec,
and ModelOpt checkpoint records and host layouts. `eider-cuda` owns the
explicit upload and cuBLASLt preparation of those records.
Flash Next now owns its QSA page backend, sequence state, and
retained-prefix cache configuration beside the model rather than under
`runtime`. Gemma 4 now owns its dense-model sequence state and physical page
table for the same reason. BitNet and Bonsai now keep their sequence state
with their model modules too. Laguna and Step-3.7 now keep their page tables,
append capabilities, and cache allocation rules with their models.
Muse Glimmer now keeps its snapshot-aware sequence state and cache allocation
rules beside its model implementation.
Ling 3 now keeps its MLA page pools, per-sequence recurrent state, graph
workspaces, and page-table updates beside its model implementation.
DeepSeek V4 now keeps compressed-attention page storage, MTP residual state,
and prefix snapshots beside its model implementation. Nemotron 3 now keeps its
hybrid attention and Mamba sequence state beside its model implementation.
The safetensors reader/index, NVFP4 artifact codec, and ModelOpt records have
moved into `eider-format`.
Flash Next now keeps its loaded model, QSA caches, persistent workspaces, and
GPU sampler in a model-owned execution state; its service retains request
policy and output state.
`eider-runtime` now owns request sampling and stop-sequence handling. Sampling
receives the backend's supported GPU top-k limit as a capability instead of
importing CUDA state.
It also owns sequence-cache retention policy; models provide the page geometry
when that policy selects a reusable prompt prefix.
Checkpoint chat rendering, structured output decoding, and Qwen XML tool
grammar now also live in `eider-runtime` with no CUDA dependency.
Generic scheduler limits, request generation policy, lifecycle events, and
API-facing chat request and usage records now also live in `eider-runtime`.
Checkpoint-derived generation defaults, including tokenizer EOS resolution and
chat-template defaults, now live there as well.
The model-specific schedulers remain in `eider-inference`; they own their
model state and invoke the runtime policy types directly.
The former `infer` package is now named `eider-inference`; its source directory
remains `crates/infer` during the migration.
Model loading now selects its CUDA device inside `eider-inference`; the API
actor has no CUDA resource import and receives inference-boundary errors.
The `eider-api` package has no direct CUDA dependency.
Bonsai GGUF chat-template parsing now also lives in `eider-inference`; the API
receives the resolved template and has no direct format dependency.
Both `eider-api` and `eider-runtime` forbid unsafe code at their crate
boundaries.
`eider-cuda` denies unchecked unsafe operations inside unsafe functions.
CUDA-owned Q2 and Q3 expert tables and NVFP4 paging slots now store opaque
`DeviceAddress` values in their device pointer tables instead of raw pointers.
The live Qwen3.6 and Laguna SM12x indexed-down plans now do the same for
expert tiles, scales, and route outputs. The legacy raw-table entry point
remains while focused benchmarks migrate.
Step-3.7 and Laguna SM12x gate/up quantization now also receive typed F32
activation address tables. `DeviceBuffer::address_at` bounds-checks their
element offsets while the CUDA boundary preserves the native table ABI.
Laguna's CUTLASS routed gate/up plan now uses its typed expert weight, scale,
and output tables directly; the remaining legacy raw-table caller is Qwen3.6.
Qwen3.6's FP8 routed-expert path likewise uses typed weight and channel-scale
address tables through its gate/up and down CUDA launches.
Ling 3's routed W4A16 workspace uses typed input, expert, output, and
weighted-accumulation tables for both one-token decode and batched execution.
Nemotron 3 uses the same typed routed W4A16 tables for resident expert slabs,
one-token decode, and flattened multi-row execution.
DeepSeek V4 attention metadata now stores typed addresses for page tables and
compressed-state tables, including explicit null entries for absent history.
Nemotron 3's paged F32 attention uses typed page-table addresses for both
single-sequence decode and ragged batch execution.
Its speculative verifier also uses typed previous-logit addresses.
Its Mamba convolution and SSM state tables use typed addresses through normal,
transactional, and rollback execution.
Its MTP cache tables use typed key and value addresses for ragged verification.
Qwen3.6 batch decode and prefill use typed convolution and recurrent-state
tables through ordinary, chunked, and speculative snapshot paths.
The persistent CUTLASS grouped-GEMM plan and its Qwen, Gemma, and Laguna
prefill callers also use opaque address tables for weights, activations,
outputs, and per-expert scales.
Model sources and focused benchmarks import host ModelOpt records directly
from `eider-format`; `eider-cuda` exports only CUDA preparation and execution
types.
`InferenceError` now preserves format failures separately from CUDA failures
at the actor-service boundary.

## Decision

Eider has five layers. `eider` is a package prefix, not an architectural
layer.

```text
eider-api
    |
eider-runtime
    |
eider-inference
   /              \\
eider-cuda      eider-format
```

`seqcache` remains a neutral dependency. It can serve both runtime and
inference. It must not depend on Eider, CUDA, or a model implementation.

The target packages are:

- `eider-api`
- `eider-runtime`
- `eider-inference`
- `eider-cuda`
- `eider-format`

The workspace can reach these package boundaries in several commits. It must
not keep permanent compatibility wrappers for the old structure.

## Design principles

### Make ownership visible

Every mutable CUDA resource has one visible owner. The owner includes streams,
scratch buffers, sequence state, cache reservations, staging buffers, and
captured graphs.

An asynchronous operation retains its resource loans until a completion
boundary. Rust lifetimes must represent this fact. A returned device result
must prevent reuse of its workspace and state until it completes.

### Keep abstractions costless

The type system can carry layout and borrowing information without runtime
work. Zero-sized layout markers, transparent scalar types, and generic model
engines compile away.

The hot prefill and decode paths must not add allocations, host-device copies,
reference-count operations, virtual calls, or implicit synchronisation because
of an abstraction.

Select a backend and a kernel plan during model construction. Do not select a
backend at every model layer or kernel call.

### Keep a small unsafe boundary

Only CUDA system bindings, kernel launches, and low-level storage system calls
need `unsafe`. API and runtime code must forbid unsafe code. Inference code
must also forbid it except for small, reviewed system modules.

No public API returns a raw CUDA pointer. No API above the CUDA layer accepts a
CUDA stream or a `DeviceBuffer`.

### Keep each representation distinct

Checkpoint layouts, host layouts, cuBLASLt layouts, native MMA layouts, and
kernel scratch layouts are different representations. A type must name its
representation.

An explicit preparation step converts one representation to another. The
runtime never reinterprets ModelOpt scales as cuBLASLt or native-MMA scales.

### Account for unified memory

GB10 device allocations consume the same 128 GB unified memory as host work.
Each persistent owner reports its bytes by category.

The categories are immutable weights, prepared layouts, sequence state, cache
pages, workspaces, graph resources, and host staging memory. A conversion or
overlay has an explicit owner and budget.

## Layer responsibilities

| Layer | Owns | Does not own |
| --- | --- | --- |
| API | HTTP, catalogue deployment, protocol adapters, telemetry mapping | CUDA selection, model buffers, checkpoint parsing |
| Runtime | admission, batching, request lifecycle, cancellation, prefix policy, output lifecycle, sampling policy | CUDA buffers, streams, model layers |
| Inference | loaded models, prepared weights, sequence state, workspaces, graph plans, attention and MoE composition | HTTP and request orchestration |
| CUDA | contexts, streams, events, allocations, typed device views, kernels, cuBLASLt, CUTLASS | checkpoint formats, scheduling, model-family policy |
| Format | safetensors, GGUF, ModelOpt host records, artifact codecs, disk records | CUDA buffers, streams, device uploads |

`seqcache` owns logical pages, reservations, prefix sharing, transactions,
accounting, and validation. A model inference backend owns physical page
storage, page tables, synchronisation, rollback, and reclamation.

The runtime decides when to reserve, retain, restore, or evict. The inference
backend performs the physical work. The neutral cache crate does not create a
CUDA stream or use a device allocation.

## CUDA contract

### Device data

`DeviceBuffer<T>` owns one device allocation. It only accepts a sealed
`DeviceRepr` type. `DeviceRepr` represents values that can safely exist in
device memory and in host readback.

Use transparent types for encoded scalar values. Examples include `Bf16`,
`Fp8E4M3`, `PackedE2M1`, and `Ue4m3`. Do not use `u8` or `u16` where the value
format matters.

Operations use borrowed views rather than owned buffers:

```rust
DeviceSlice<'a, T>
DeviceSliceMut<'a, T>
DeviceMatrix<'a, T, Layout>
DeviceMatrixMut<'a, T, Layout>
```

`Layout` is a zero-sized marker. The initial markers are `RowMajor`,
`ColumnMajor`, `ModelOptNvfp4`, `CublasLtVec16`, `Sm12xMma`, and `PagedKv`.

Views carry pointer, extent, stride, and layout information. Kernel calls
already need this data. The views add no allocation or device work.

### Streams and asynchronous work

All enqueues use an exclusive `CudaPass` or `&mut CudaStream`. An immutable
stream reference is only for observation after completion.

An execution guard owns a pass and retains all mutable resource loans. The
guard releases them only after an explicit completion operation. A failed
operation must also complete or safely cancel outstanding work before it
releases the resources.

Pinned host transfers return a pending loan. The loan prevents host access to
the pinned buffer until `wait` completes. This rule applies to both host-to-
device and device-to-host transfers.

Blocking operations say `blocking` in their names. Stream-ordered allocation
and initialisation take a pass explicitly. No general allocation helper calls
`cudaDeviceSynchronize` without saying so in its name and documentation.

### Graphs and events

Raw graph executables are private CUDA implementation details. A safe graph
plan owns its graph executable and every mutable allocation captured by that
graph. It also retains the lifetime of read-only model resources used by the
graph.

Event recording mutates event state. It takes an exclusive event reference and
an exclusive pass reference. Cross-stream dependencies use explicit events.

### Kernel API

The safe CUDA API follows this order:

1. input views
2. mutable output views
3. semantic scalar arguments or an argument structure
4. the mutable pass

For example:

```rust
rms_norm(input, weight, output, epsilon, &mut pass)?;
```

The safe API derives stable dimensions from views. It validates dynamic batch
dimensions at the call boundary. It prepares stable plans during model load.

The private FFI layer uses one canonical C declaration source. It groups
functions by primitive family: elementwise and normalisation, layout and copy,
attention and KV, routing and MoE, sampling, recurrent state, and
quantisation.

CUDA plans own device pointer tables and the allocations that those tables
reference. Inference code does not construct device pointer arrays with raw
pointer arithmetic.

## Inference contract

Inference owns all model-specific execution state. This includes prepared
weights, cache adapters, expert residency, mutable sequence state, streams,
workspaces, and graph plans.

The runtime uses typed sequence handles. It does not retain a mutable
CUDA-backed sequence object. A generational `SequenceId` detects a stale
handle. The inference engine resolves handles once while it builds a batch.

The stable runtime seam is batch execution, not individual tensor operations:

```rust
trait InferenceEngine {
    type SequenceId: Copy;
    type Decoded<'a>: DecodedBatch
    where
        Self: 'a;

    fn create_sequence(&mut self, limits: SequenceLimits)
        -> Result<Self::SequenceId>;

    fn decode<'a>(
        &'a mut self,
        rows: &[DecodeRow<Self::SequenceId>],
    ) -> Result<Self::Decoded<'a>>;
}
```

`Decoded<'a>` borrows the engine. It provides semantic operations such as
constraint masking, GPU sampling, compact diagnostic readback, and
speculative-state updates. It does not provide a raw stream, logits buffer, or
hidden-state buffer.

This interface keeps the runtime independent of the CUDA implementation. It
also keeps batch execution monomorphised. A model-family engine can use its
own fused kernels and prepared layouts without a generic tensor backend.

Model-specific cache types belong in inference. This includes
`Sm12xPageBackend` and composed prefill attention. `Sm12xKvPagePool` and raw
attention kernels belong in CUDA.

Expert residency is also inference state. A pure LRU planner can stay a small
CPU type, but it must not be combined with CUDA buffers, stream ordering, and
prepared expert records in one public type.

## Runtime contract

The runtime owns request policy and scheduling. It selects work, records
request lifecycle, manages cancellation, drives logical prefix-cache policy,
and turns sampled tokens into output events.

Sampling policy belongs in runtime. It includes temperature, top-k, top-p,
history penalties, the request RNG, grammar constraints, and finish rules.
Inference performs device sampling and owns device token-count state.

The runtime passes a sampling request to inference. Inference returns compact
sampled tokens. It does not return a vocabulary-sized logits buffer unless a
diagnostic path explicitly requests one.

The API can erase the concrete runtime service behind `dyn ActorService`. This
call occurs once per actor tick. It is outside the layer and kernel hot paths.

Do not add a universal backend trait for every tensor or CUDA operation. The
batch inference engine is the only required polymorphic seam.

## Format contract

`eider-format` reads and writes host data. It defines safetensors, GGUF,
ModelOpt records, cache artifacts, and disk-backed expert records.

Format types contain host bytes and metadata. They do not contain a
`DeviceBuffer` or a `CudaStream`. Inference explicitly prepares and uploads a
format record into a device plan.

For example, a ModelOpt NVFP4 record remains a host format type. A separate
inference preparation operation converts it into a cuBLASLt or native-MMA
weight. That operation is the only place that can change the layout.

## Error contract

Each layer has its own error type:

```text
CudaError   -> InferenceError -> RuntimeError -> API response
FormatError -^
```

CUDA errors name CUDA calls and device failures. Format errors name malformed
files and metadata. Runtime errors name invalid requests, admission limits,
and lifecycle errors.

Do not use a CUDA error variant for a request or format error. The API maps a
runtime error to protocol status and telemetry.

## Rust documentation and style

Each crate has crate documentation that states its owned concepts and forbidden
dependencies. Each public CUDA owner, view, and operation documents:

- memory space
- ownership and lifetime
- shape, stride, and layout
- stream ordering and synchronisation
- allocation behaviour
- graph-capture requirements
- thread and device affinity.

Every public unsafe item has a precise `# Safety` section. Every unsafe block
states its local safety invariant.

Use `#![deny(unsafe_op_in_unsafe_fn)]` in unsafe-capable crates. Use
`#![forbid(unsafe_code)]` in API and runtime crates. Enable
`#![deny(missing_docs)]` and broken intra-doc-link checks after each affected
public surface is complete.

## Migration plan

1. Correct the CUDA ownership contract in the existing crate. Add `DeviceRepr`,
   private raw pointers, pending host-transfer loans, explicit stream mutation,
   and graph-resource ownership.
2. Add typed scalar and layout views to one Qwen execution path. Remove the
   replaced API in the same change.
3. Move Qwen masking, sampling, speculative copies, and completion handling
   behind an inference execution guard.
4. Move model sequence state, KV implementations, expert residency, and page
   backends out of `runtime`.
5. Extract pure host checkpoint and artifact code into `eider-format`.
6. Extract the corrected device and FFI surface into `eider-cuda`.
7. Extract generic scheduling and serving into `eider-runtime`. Migrate model
   families one at a time.
8. Remove old re-exports, old module paths, and duplicate service adapters as
   each migration slice becomes complete.

Qwen is the reference migration path. It exercises batching, grammar masks,
GPU sampling, prefix caching, MTP, DFlash2, graphs, and multiple streams.

## Acceptance gates

Every migration slice must meet these conditions:

- The crate graph has no upward dependency from inference into runtime.
- API and runtime code contain no unsafe code or CUDA resource imports.
- Compile-fail tests reject workspace reuse during decoded work.
- Compile-fail tests reject pinned-buffer access during an asynchronous loan.
- Compile-fail tests reject incompatible device layouts.
- Focused CPU references retain numerical correctness for migrated kernels.
- Focused micromeasures report allocation, copy, and synchronisation changes.
- A hot prefill or decode path has no allocation and no hidden device-wide
  synchronisation.
- `cargo fmt --all`, relevant tests, Clippy, and `cargo doc --no-deps` pass.

Run an end-to-end model benchmark only at meaningful integration gates. Use
focused correctness tests and micromeasures for ordinary CUDA and structural
changes.

## Non-goals

This design does not create a generic tensor framework. It does not require a
common kernel API across future backends. It does not move CUDA code into
`seqcache`. It does not add compatibility layers for historical module names.

The design makes future backends possible at the batch inference boundary. It
does not prescribe a backend before there is a concrete requirement.
