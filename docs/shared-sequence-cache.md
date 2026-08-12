# Shared sequence and prefix cache

## Status and intent

This document records the design of the reusable
[`seqcache`](https://github.com/rdaum/seqcache) Rust crate. The implementation
now lives in its own repository, and Eider consumes an exact Git revision. It
is intended to be shared with CPU inference runtimes rather than duplicated in
each project.

The crate must unify two things which are currently separate:

1. ownership and allocation of live, token-proportional sequence state; and
2. indexing and retention of reusable prompt-prefix state.

The important consequence is that a prefix checkpoint references the same
immutable physical KV pages used by live sequences. It must not copy a valid
prefix into a second checkpoint allocation, and restoring a prefix must not copy
all of those KV values into a new full-capacity sequence allocation.

The shared crate is a control-plane and ownership component. CPU storage, CUDA
storage, attention kernels, recurrent-state representations, and stream or
executor integration remain in their respective runtimes.

## Why this is needed

Eider currently has a good ART-backed token-prefix index and compact
device-resident checkpoints, but its live KV cache still allocates every layer
for a sequence's full requested capacity. Checkpoint creation and restore copy
the valid prefix. Its 128-token blocks are units of prefix indexing, not shared
physical KV storage.

Tinfer similarly allocates token-proportional attention state for
`prompt_tokens + max_output_tokens` when a request is admitted. Its prefix cache
retains model-state clones and forks those clones into another request-sized
allocation on a hit. Ling has a compact checkpoint implementation, but the
generic Transformer and Qwen3.6 states can retain unused capacity. The tinfer
scheduler limits active sequence count but does not admit according to a
sequence-state byte budget.

These designs have four common defects:

- KV storage is reserved eagerly according to maximum request length.
- Prefix insertion duplicates KV values.
- Prefix restore duplicates KV values again.
- Sequence-count limits do not prove that admitted work will fit in memory.

The new crate must address all four. Merely moving the existing prefix index or
compact checkpoint type into another crate is insufficient.

## Goals

The crate must provide:

- fixed-size, incrementally allocated token pages;
- immutable sharing of complete prefix pages between live sequences and prefix
  entries;
- a private writable tail page per sequence, with copy-on-write when an
  unaligned branch is explicitly supported;
- a content-based longest-prefix index backed by `rart`;
- exact ownership and memory accounting for live sequences, retained prefixes,
  shared pages, reservations, and model-specific snapshots;
- byte-budgeted admission which cannot overcommit the configured managed-memory
  capacity;
- eviction which releases prefix ownership without invalidating pages still in
  use by live sequences;
- transactional operations: allocation, snapshot, or backend failures must not
  leave leaked references, lost cache entries, or false accounting;
- model-specific immutable snapshots for recurrent, convolutional, and other
  non-pageable state;
- backend-defined physical storage and synchronization;
- telemetry implemented with `fast-telemetry`, without hand-written shared
  atomic counters; and
- a deterministic, backend-independent conformance test suite.

The crate must work for conventional attention as well as hybrid models. It
must not assume that every layer has KV state or that all attention layers use
the same logical representation.

## Non-goals

The shared crate will not:

- own CPU attention kernels or CUDA kernels;
- depend on Eider's `nvfp4` crate or tinfer's `tinfer-kernels` crate;
- define K/V element types, quantization, or per-layer memory layouts;
- move model weights, expert paging, scheduler queues, sampling, or HTTP state;
- make CUDA buffers accessible to tinfer or CPU matrices accessible to Eider;
- provide cross-process or persistent prefix caching;
- permit prefix reuse across incompatible model instances, revisions, RoPE
  configurations, or cache geometries; or
- hide backend synchronization behind an implicit blocking operation.

## Repository and dependencies

The implementation lives in the standalone repository:

```text
seqcache/
    Cargo.toml
    src/
        lib.rs
        error.rs
        index.rs
        manager.rs
        metrics.rs
        reservation.rs
```

The names of internal modules may change as the implementation clarifies, but
the package must remain independently consumable. Its normal dependencies
should be limited to:

```toml
[dependencies]
fast-telemetry = "0.8"
rart = "0.11"
```

Do not add `tokio`, `tracing`, `serde`, `thiserror`, `parking_lot`, CUDA crates,
half-precision types, or a model runtime dependency. Implement `Display` and
`Error` directly for the small error enum. Standard-library collections and
ownership types are expected.

Tinfer currently uses `fast-telemetry` 0.7. It must be upgraded to the same
0.8 line before consuming this crate so that metric types and export behaviour
are not split across two library versions.

Repository consumers should pin an exact tested revision:

```toml
seqcache = { git = "https://github.com/rdaum/seqcache", rev = "<commit>" }
```

Once published, consumers may use a compatible crates.io release. A
branch-floating Git dependency is not an acceptable production dependency.

## Ownership and concurrency model

One cache manager is owned by one model scheduler or inference actor. All
metadata mutation is serialized by that owner. The core must use ordinary
non-atomic integers for page references, clocks, reservations, and byte counts.
It must not put the manager behind an internal `Arc`, `Mutex`, or `RwLock`, and
it must not add atomic fields for observation.

The manager may be `Send` when its backend and snapshot types are `Send`, so an
idle runtime can transfer ownership to another thread. It is not a concurrently
mutable service and should not be `Sync` by construction. Backends may use
their own synchronization where the hardware requires it, but that is outside
the shared metadata path.

Fast paths running on worker threads receive immutable page descriptions or
backend-native views prepared by the owning scheduler. They do not increment
page references and do not consult the prefix index. A decode step should enter
the cache manager only when it needs a new page, not once per token merely to
update metrics or length fields.

## Core terminology

`page`
: A fixed number of token positions in one sequence. A logical page represents
  the token-proportional cache storage needed by all relevant layers for those
  positions. The backend decides whether that is one allocation, per-layer
  slabs, multiple cache lanes, or another representation.

`page bundle`
: The backend-owned physical storage represented by one logical page ID. For a
  heterogeneous model it can include different storage for full attention,
  MLA, local attention, or other token-proportional state. Layers without
  token-proportional history consume no space in the bundle.

`tail page`
: The last, incompletely filled page of an active sequence. It is writable and
  privately owned.

`sealed page`
: A complete page whose KV contents and representation will never change. It
  can be referenced by multiple live sequences and prefix entries.

`prefix entry`
: A token-content key, a sequence position, references to sealed page bundles,
  and an immutable model-specific snapshot for non-pageable state.

`reservation`
: Capacity promised to an admitted sequence but not necessarily backed by a
  populated page yet. A strict reservation prevents a request from failing
  halfway through generation because other work consumed its future capacity.

`snapshot`
: Immutable model-specific state at a page-aligned position. Examples include
  Gated DeltaNet matrices, KDA convolution state, Mamba state, and LFM2 short
  convolution state. It does not contain token-proportional KV values managed
  as pages.

## Required invariants

The implementation must make the following invariants explicit and cover them
with tests:

1. A sealed page is immutable until it is reclaimed.
2. A writable page has exactly one active owner and no prefix owner.
3. A page is returned to its backend only after all active and prefix references
   are gone and the backend says reuse is safe.
4. A prefix entry contains only complete pages. Its position is therefore a
   non-zero multiple of the configured page-token count.
5. A sequence's logical position agrees with its complete pages and valid tail
   rows.
6. A sequence never consumes more physical pages plus outstanding reservations
   than admission granted it.
7. All memory arithmetic is checked. Accounting overflow is an error, not a
   saturating adjustment.
8. Page IDs detect stale handles. Reuse of a slab slot must change a generation
   or otherwise prevent an ABA-style stale ID from addressing new contents.
9. Prefix keys are content identities. Physical page IDs are never used as
   token-prefix keys because independently computed identical tokens can live
   in different physical pages.
10. A prefix hit is valid only within one compatible cache namespace. The
    simplest supported arrangement is one manager per loaded model instance and
    cache geometry.
11. Failed insertion, restore, allocation, page-table update, or snapshot
    restore leaves the manager in its prior valid state, apart from explicitly
    documented telemetry about the failure.
12. Dropping or cancelling a sequence releases its unused reservation and all
    active page references exactly once.

Use distinct opaque types for at least `PageId`, `SequenceId`, `PrefixEntryId`,
and token-block identity. Do not expose raw collection indices as stable IDs.

## Page geometry

Page geometry is fixed for the lifetime of a manager. Configuration must state:

- tokens per page;
- bytes per page bundle, or an exact backend query which is constant for that
  manager;
- maximum managed bytes or page slots;
- maximum bytes retained in non-pageable prefix snapshots;
- maximum prefix entries if an administrative entry bound is desired; and
- admission policy and any reserved emergency capacity.

The existing Eider prefix cache uses 128-token index blocks. Supporting 128 as
the initial page size gives a direct migration path, but the crate must not bake
128 into its public types. Smaller CUDA pages may reduce copy-on-write waste and
can fit paged-attention kernels better. Tinfer may make a different measured
choice. Prefix checkpoint alignment always follows the manager's configured
physical page size.

Do not quietly use one size for index blocks and another size for physical
pages in the first implementation. That creates partial ownership cases and
defeats zero-copy prefix insertion. If sub-page indexing is later justified, it
requires a separate design and measurements.

One logical page ID should normally identify the same token range across all
cache-bearing layers. A backend can implement that ID with per-layer slabs:

```text
layer 0 K pages: [page slot][page token][width]
layer 0 V pages: [page slot][page token][width]
layer 5 K pages: [page slot][head][page token][dim]
...
```

This keeps the core page table model-independent while allowing each layer to
retain its production layout.

## Prefix indexing

Use `rart::AdaptiveRadixTree` for longest-prefix lookup. The index should retain
the useful property of Eider's current design: fixed-width, big-endian token
block IDs form an ART key whose byte prefixes correspond exactly to token-page
prefixes.

Token block identities and physical page identities are separate:

```text
TokenBlockId = identity of a page-sized token slice
PageId       = ownership handle for backend KV storage
```

The token-block interner must be reference counted and garbage collected. The
current Eider `HashMap<Box<[u32]>, u32>` retains every unique query block ever
seen; the shared implementation must not have that unbounded metadata growth.

Use separate lookup and insertion flows:

- Lookup consults already interned token blocks without creating entries. At
  the first unknown block, no longer prefix can exist, so key construction can
  stop.
- Insertion interns missing blocks, increments their prefix-key references, and
  rolls those changes back if the insertion fails.
- Eviction decrements token-block references and removes unreferenced interner
  records.

The interner must handle ID exhaustion and must not reuse an ID while an ART key
can still contain it. Direct collision-free token comparison remains the final
authority; do not introduce a probabilistic hash-only key without storing and
verifying the original token block.

For a prompt of length `n`, the default reusable position is:

```text
floor((n - 1) / page_tokens) * page_tokens
```

This preserves the final prompt token for the decode transition, matching both
runtimes' scheduling semantics. The scheduler must stop a prefill chunk at that
position so the recurrent snapshot and sealed pages describe the same point in
the model state.

Longest-prefix lookup returns the greatest retained page-aligned prefix. It
must update LRU state only for the selected entry. A duplicate insertion must
refresh or retain the existing entry without evicting unrelated entries. In
particular, do not call a destructive capacity-preparation routine before
checking whether the exact key already exists.

## Page ownership and prefix reuse

A live sequence starts with no pages or with shared sealed pages restored from
a prefix entry. Appending tokens follows this lifecycle:

1. Consume rows in the sequence's private tail page if space remains.
2. Otherwise consume one previously admitted reservation and ask the backend
   for a page bundle.
3. Enqueue or perform K/V writes into that private page.
4. Commit the new logical position after the backend accepted the operation.
5. When the page becomes complete, seal it. Any precision conversion or layout
   finalization must be complete before the page is published for sharing.

Creating a prefix entry adds prefix references to existing sealed pages. It
does not allocate or copy KV storage. Restoring an aligned prefix adds active
references to those same pages and restores the separate immutable snapshot
into request-private recurrent state.

Normal cached prefixes are page aligned, so restore does not require copying a
tail. The core should nevertheless support an explicit branch operation from
an unaligned live sequence for future beam/speculative use. That operation must
copy only the partial tail into a new private page through the backend and share
all complete pages.

When an independently computed sequence attempts to insert a prefix key that is
already retained, keep the existing entry, update its recency, and leave the
new sequence's pages under that sequence's ownership. Deduplicating two already
materialized physical prefixes can be a later optimization; it is not required
for correctness.

## Admission and reservations

The manager must support strict byte-budgeted admission. Counting active
sequences is not sufficient.

An admission request provides at least:

- maximum logical sequence position (`prompt + maximum generated tokens`);
- prefix pages which will be shared on a hit;
- private fixed-state bytes for the active model-specific recurrent state;
- any backend-defined per-sequence page-table bytes; and
- optional policy metadata needed to queue the request fairly.

After prefix lookup, calculate the maximum number of new page bundles that the
request can require. Existing shared prefix pages are already resident and are
not charged again as physical bytes. Capacity for every additional page is
reserved before admission, even though the pages can be populated lazily.

This separation is important:

- reservation prevents mid-generation OOM;
- lazy population avoids clearing or initializing a full request-sized cache at
  admission; and
- shared prefix pages reduce the unique capacity a restored request reserves.

If a reservation does not fit, the manager may plan prefix evictions. Eviction
removes prefix references in LRU order. A page still referenced by an active
sequence does not become free merely because its prefix entry was evicted, so
the planner must count bytes that will actually become available rather than
the logical size of the evicted entry.

If sufficient capacity still cannot be guaranteed, admission returns a
non-fatal `WouldBlock` outcome. The scheduler keeps the request waiting. It must
not admit optimistically and hope that later decode allocations succeed.

Maintain separate exact values for:

- unique resident page bytes;
- free/reusable page capacity;
- promised but not yet populated page bytes;
- active private-state bytes;
- prefix snapshot bytes;
- page-table/backend metadata bytes when material; and
- reclaimable prefix-only page bytes.

Also expose a total managed-memory value suitable for server admission metrics.
Model weights and shared execution workspaces remain outside this manager, but
the runtime can subtract their known footprint when choosing the manager's hard
budget.

## Eviction and transactional mutation

Capacity changes must use a plan/commit pattern. A capacity check may identify
candidate prefix entries, but it must not destroy them before all fallible work
needed for the replacement has succeeded.

One acceptable shape is:

```rust
let plan = cache.plan_prefix_insert(key, snapshot_bytes)?;
let snapshot = model.create_snapshot(...)?;
cache.commit_prefix_insert(plan, snapshot)?;
```

The exact API may instead use a closure or guard whose `Drop` rolls back an
uncommitted operation. Required behaviour is more important than the type
names:

- duplicate detection occurs before eviction;
- failed snapshot creation does not empty useful cache state;
- failed backend allocation or page-table update releases tentative resources;
- prefix refcounts, token-block interner refs, reservations, and byte gauges are
  committed together; and
- no user-provided callback runs while internal metadata is half-mutated.

LRU order should be deterministic. The manager's plain `u64` logical clock must
have a defined overflow strategy; silently wrapping and then treating new
entries as oldest is incorrect. Renormalizing retained timestamps when the
clock approaches its limit is sufficient.

## Backend boundary

The shared crate owns logical state and calls a backend trait for physical
operations. The final trait should be driven by the two concrete adapters rather
than designed in isolation, but it will need capabilities equivalent to:

```rust
pub trait PageBackend {
    type Page;
    type Context;
    type Error;

    fn page_bytes(&self) -> usize;

    fn allocate_page(
        &mut self,
        context: &mut Self::Context,
    ) -> Result<Self::Page, Self::Error>;

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context,
    ) -> Result<Self::Page, Self::Error>;

    fn retire_page(
        &mut self,
        page: Self::Page,
        context: &mut Self::Context,
    ) -> Result<(), Self::Error>;
}
```

This is illustrative, not a requirement to preserve these exact method names.
The real API must also let the runtime obtain an immutable, ordered page table
for attention and a writable destination for append. It must not require a
virtual call or lock inside an attention loop.

The context makes synchronization explicit. Tinfer can use `()` or an executor
context. Eider can supply a CUDA stream and whatever device-page-table update
state it needs. The shared crate must not create a CUDA stream, synchronize a
stream, or assume the default stream.

Page retirement can be asynchronous. Once logical references reach zero, an
Eider backend may place the physical slot on a deferred-retirement list until a
CUDA event proves that prior users have completed. Such a page counts as
resident and unavailable until the backend reports it reusable. Tinfer can
normally recycle immediately after its cooperative kernel submission has
joined.

Avoid exposing a backend `Page` to arbitrary long-lived runtime ownership. The
manager should remain the source of truth for lifetime and return borrowed
views, stable IDs, or short-lived operation descriptors.

## Model-specific snapshots

The manager must be generic over an immutable snapshot value. A snapshot must
report its exact retained bytes. It need not implement `Clone`: Eider snapshots
can own device buffers and should be restored by borrowing them.

The restore operation therefore needs closure- or trait-based semantics similar
to:

```rust
cache.restore_longest(tokens, admission, |snapshot, position| {
    model.restore_private_state(snapshot, position)
})
```

If restoring the private state fails, page references and reservations acquired
for the tentative sequence must roll back.

The snapshot boundary is model specific:

- Qwen dense/MoE and ordinary GQA models may have an empty snapshot because all
  sequence history is in KV pages.
- Qwen3.6 stores Gated DeltaNet state while full-attention K/V lives in pages.
- Ling stores KDA convolution and recurrent matrix state while MLA history lives
  in pages.
- LFM2 stores short-convolution state while full-attention history lives in
  pages.
- Nemotron 3 stores Mamba and optional MTP state not represented by attention
  pages.
- Models with sliding, local, sparse, or compressed attention define page
  bundles containing exactly the token-proportional history their continuation
  requires.

Snapshot bytes count against the managed memory budget. Restoring normally
copies fixed-size recurrent data into mutable request-private state; it must not
copy the token-proportional attention pages.

Do not use `Arc` merely to share snapshots. The manager is single-owner and can
hold snapshots directly with plain metadata references. A borrowed snapshot is
sufficient during restore.

## Public API direction

The public API should expose operations, not internal collections. A likely
shape is:

```rust
pub struct SequenceCache<B, S> { /* private */ }
pub struct CacheConfig { /* page and byte limits */ }
pub struct AdmissionRequest { /* maximum position and private bytes */ }
pub struct SequenceId { /* opaque */ }
pub struct PrefixMatch { /* position and accounting, no owned pages */ }
pub enum AdmissionOutcome {
    Admitted(SequenceId),
    WouldBlock,
}

impl<B, S> SequenceCache<B, S>
where
    B: PageBackend,
    S: RetainedSnapshot,
{
    pub fn lookup_prefix(&mut self, tokens: &[u32]) -> Option<PrefixMatch>;
    pub fn admit(/* prefix, request, backend context */) -> Result<AdmissionOutcome, _>;
    pub fn reserve_append(/* sequence and rows */) -> Result<AppendTarget<'_>, _>;
    pub fn commit_append(/* target and rows */) -> Result<(), _>;
    pub fn retain_prefix(/* sequence, tokens, snapshot */) -> Result<_, _>;
    pub fn page_table(&self, sequence: SequenceId) -> Result<PageTableView<'_>, _>;
    pub fn finish(&mut self, sequence: SequenceId, /* context */) -> Result<(), _>;
    pub fn stats(&self) -> CacheStats;
}
```

This sketch deliberately does not settle borrow details. The implementation
agent should write the fake backend and one real Eider call site while shaping
the API; otherwise it is too easy to produce a theoretically generic trait that
cannot express CUDA stream ordering or tinfer's borrowed page slices.

Errors must distinguish malformed configuration, stale IDs, invariant failure,
backend failure, arithmetic overflow, and normal admission pressure.
`WouldBlock` is scheduling state, not an exceptional error.

## Attention-kernel contract

Zero-copy sharing requires attention kernels which consume pages. A cache
manager alone cannot provide the memory improvement if the runtime gathers all
pages into a contiguous request-sized buffer before every attention operation.

Each runtime needs backend-native operations equivalent to:

```text
append_kv(page_table, destination_position, K, V)
decode_attention(query, page_table, sequence_length)
prefill_attention(query_rows, page_table, start_position)
```

The shared crate supplies logical ordering and valid lengths. The backend
supplies addresses, layout, element type, and execution.

### Tinfer kernel requirements

Tinfer currently has both token-major and head-major contiguous cache views.
The CPU adapter must preserve the production layout chosen by each model; the
shared crate must not force all models through one generic matrix shape.

Decode attention should combine page contributions with an online softmax so it
does not gather a whole prefix. The numerical algorithm must carry the running
maximum, normalization sum, and weighted value accumulator across pages. SIMD
and cooperative executor plans should receive contiguous slices within each
page and must not consult manager metadata or increment shared counters in
their inner loops.

Prefill must handle a chunk which begins in an existing page and allocates one
or more new pages. It may use a backend workspace, but any temporary gather must
be bounded by the prefill chunk rather than maximum sequence length.

Tinfer currently converts some complete caches from F32 to F16 for decode.
With shared pages, storage-format conversion must finish before a page is
sealed or placed in the prefix cache. A backend may select an F16 page class,
convert pages as they become complete, or retain an intentional F32 mode. It
must never mutate a sealed page in place.

CPU allocation and first-touch policy remain in tinfer. On multi-node systems,
page slabs or bundles must respect the runtime's affinity and NUMA placement.
The shared crate must not spawn threads or choose CPUs.

### Eider kernel requirements

Eider should implement physical pages as stable slots in backend-owned CUDA
slabs. A logical page ID can address the same slot across per-layer K and V
slabs. Device allocations must remain stable for CUDA graph capture.

Paged attention needs a device-resident page table. For captured decode paths:

- the page-table buffer address and capacity-class workspace addresses remain
  stable;
- page IDs, valid lengths, and destination positions are updated in place;
- kernels bound reads by the logical sequence length;
- page-table updates are ordered on the request's explicit stream; and
- reclamation waits for all streams or captured launches which can still read a
  page.

Do not replace prefix-copy traffic with per-token host/device page-table
synchronization. Allocate page-table capacity at admission, update only when a
page boundary is crossed, and keep the decode graph's pointer arguments stable.

Eider's existing indexed-attention and device-resident length machinery is a
useful starting point, but the current indexed operation still addresses one
contiguous cache. It must be extended to translate logical token positions
through page IDs.

The CUDA backend may preallocate its physical slab to the configured pool size.
That still fixes the defect: allocation becomes globally bounded and pages are
shared rather than multiplied by every request's maximum length. Unified-memory
usage on GB10 must include slabs, recurrent snapshots, page tables, and deferred
retirements in the configured safety margin.

## Telemetry

The crate depends directly on `fast-telemetry` and owns cache-specific metric
types. Runtime exporters remain outside the crate. Eider and tinfer should
export these metrics alongside their existing server metrics.

Use native `Counter`, `Gauge`, and `Histogram` values. Do not reproduce the
current Eider pattern of process-wide `AtomicI64` values plus locks merely to
aggregate gauges. Exact cache state already exists as ordinary owner-thread
fields. Publish it directly through the manager's fast-telemetry handle and
return it through `CacheStats` for synchronous diagnostics.

Counters and histograms should include at least:

- prefix lookups, hits, misses, and restored tokens;
- prefix insertions, duplicate insertions, and evictions;
- admission successes and `WouldBlock` outcomes;
- pages allocated, recycled, sealed, copied-on-write, and retired;
- backend allocation and operation failures;
- lookup, insertion, eviction, admission, and restore latency; and
- bytes made reclaimable by eviction.

Gauges or exact exported state should include at least:

- active sequences;
- retained prefix entries;
- interned token blocks;
- resident, free, reserved, and deferred-retirement pages;
- unique resident page bytes;
- outstanding reservation bytes;
- active private-state bytes;
- retained snapshot bytes; and
- reclaimable prefix-only bytes.

Metric recording must occur at structural operations, normally page boundaries
or request lifecycle events. There must be no metric increment in the inner
per-head, per-row, or per-element attention loops. The crate must not expose
internal atomic counters for callers to poll.

Metrics should be owned per manager or by an explicitly supplied metrics handle
whose aggregation semantics are defined. A process-global singleton makes
tests interfere and makes multiple loaded models hard to account for. If the
fast-telemetry exporter requires a stable long-lived object, the runtime should
own that object and pass a reference when constructing the manager.

## Eider integration constraints

Eider has several model-specific schedulers and sequence-state types. The
shared cache must replace the common ownership pattern without flattening those
models into a false conventional-KV abstraction.

Integration must preserve:

- decode-first scheduling and bounded prefill;
- page-aligned prefill checkpoint boundaries;
- explicit non-default CUDA stream semantics;
- CUDA graph capture and stable workspace addresses;
- exact model-specific state for Qwen3.6, Step-3.7, Gemma 4, Laguna,
  Nemotron 3, and DeepSeek V4;
- sliding/local/global attention distinctions;
- MTP and recurrent-state checkpoint semantics; and
- existing request cancellation and lifecycle reporting.

The first Eider adapter should target one exercised model with conventional
full attention or Qwen3.6's full-attention layers. Do not begin by attempting to
generalize every model family. The shared crate API is acceptable only after
that adapter proves:

- stable device page tables;
- zero KV copy on prefix insertion and restore;
- correct stream-ordered append and reclamation; and
- logits matching the current contiguous path.

Temporary parity code is reasonable while the paged kernel is validated, but
the end state must remove the model's duplicate contiguous cache path rather
than retain permanent parallel implementations.

## Tinfer integration constraints

Tinfer will consume this same crate. Do not publish an API which assumes CUDA,
device bytes, asynchronous execution, or token-major F32 cache rows.

The tinfer adapter must fit its current model boundary:

- `ModelSequenceState` is opaque to the generic scheduler.
- Generic Transformer state includes attention KV and possibly short
  convolution state.
- Qwen3.6 combines full-attention history and Gated DeltaNet recurrent state.
- Ling combines MLA history with KDA convolution and recurrent matrix state.
- LFM2 combines full attention with short convolution.
- KV storage can be F32 or F16 and can be token-major or head-major.
- Prefix snapshots must preserve the exact model-specific recurrent state.
- The persistent cooperative executor owns worker placement; the cache crate
  must neither spawn workers nor synchronize them through shared atomics.
- Runtime admission currently permits up to 64 active sequences. Integration
  must make the shared manager's byte reservation authoritative even if the
  sequence-count limit remains as an additional fairness bound.

The tinfer scheduler should eventually replace `checkpoint() -> Self` and
`fork(capacity) -> Self` with separate page ownership and model-specific
snapshot operations. A restored sequence must reference cached pages directly,
not materialize a new `Matrix` for the cached prefix.

For Ling specifically, the KDA state remains a retained snapshot copied into
private mutable state on restore. The six MLA layers become pageable. Memory
accounting must include both the fixed KDA state and every unique MLA page, so a
large prompt cannot bypass admission merely because its token-proportional
portion is represented differently from a conventional KV cache.

Tinfer's integrated Transformers golden-logit test and model-specific smoke
tests remain correctness gates. F16 cache modes require parity coverage because
page sealing changes the point at which format conversion occurs.

## Suggested implementation phases

### Phase 1: backend-independent core

Create the backend-independent `seqcache` core with:

- checked configuration and typed IDs;
- a deterministic fake page backend;
- page reference and reservation accounting;
- active sequence creation, append reservation/commit, finish, and cancellation;
- sealed-page sharing and partial-tail copy-on-write;
- refcounted token-block interning;
- ART longest-prefix lookup;
- prefix insertion, duplicate handling, LRU eviction, and restore transactions;
- generic retained snapshots with exact byte accounting; and
- fast-telemetry metrics plus plain `CacheStats` snapshots.

This phase is not complete if it merely ports Eider's current
`runtime/prefix_cache.rs`. Tests must demonstrate that prefix entries and live
sequences reference the same fake physical pages and that restore performs no
full-prefix copy.

### Phase 2: Eider page backend and one model

Implement backend-owned CUDA slabs, stable host and device page tables,
stream-ordered page-table updates, deferred page reclamation, and paged append
and attention for one model. Integrate strict admission reservations into that
model's scheduler.

Measure and validate before expanding scope. The initial results must report:

- configured and observed pool bytes;
- admission behaviour under contention;
- prefix hit latency and tokens reused;
- bytes and time copied on insertion and restore, which should exclude KV;
- decode and prefill performance against the contiguous baseline; and
- page-table/kernel overhead at short and long contexts.

### Phase 3: Eider hybrid models

Move model-specific recurrent snapshots behind the shared manager and migrate
the other Eider schedulers. Add page-bundle support required by heterogeneous,
local, sliding, compressed, and speculative states only when a concrete model
adapter demands it.

Remove the old shared prefix-cache implementation after all production call
sites use the new crate.

### Phase 4: tinfer backend

Upgrade tinfer to `fast-telemetry` 0.8 and add the shared dependency. Implement
CPU page bundles and paged/segmented attention operations. Migrate the generic
Transformer first, then Qwen3.6 and Ling, validating recurrent snapshots and
both F32/F16 storage modes at each step.

Make byte reservations part of generic scheduler admission. Remove the
`checkpoint() -> Self`/`fork()` prefix mechanism and its eight-entry linear
cache once all runtime variants use the shared manager.

## Required core tests

The shared crate's tests must be deterministic and should need no runtime or
hardware. At minimum, cover:

1. Empty, short, exactly aligned, and multi-page prompt keys.
2. Longest-prefix selection among nested and divergent entries.
3. A lookup miss does not grow the token-block interner.
4. Eviction removes unreferenced token-block metadata.
5. Duplicate insertion does not evict another prefix.
6. Prefix insertion increments page references without copying or allocating
   pages.
7. Aligned restore shares every prefix page and allocates only future capacity
   as reservations.
8. Unaligned branch shares complete pages and copies exactly one tail page.
9. Evicting a prefix does not free a page still used by a live sequence.
10. Finishing the last active user makes an evicted page reclaimable.
11. Two prefix entries sharing early pages are accounted once physically.
12. Strict admission rejects or waits before overcommitting future pages.
13. Cancellation releases populated pages and unused reservations.
14. Backend allocation, copy, retire, snapshot-restore, and page-table update
    failures roll back cleanly.
15. Stale sequence and page handles are rejected after slot reuse.
16. Counter and gauge changes agree with `CacheStats` after every lifecycle
    operation.
17. Clock renormalization preserves LRU order.
18. Checked arithmetic rejects impossible geometry and capacity values.

A deterministic state-machine test should run long sequences of admit, append,
retain, restore, evict, cancel, and finish operations while recomputing all
reference counts and byte totals from first principles after each operation.
This can use a tiny in-test PRNG rather than adding a normal dependency.

## Runtime correctness and performance gates

Each backend needs tests outside the shared crate.

For Eider:

- compare paged append, prefill, and decode logits with the existing contiguous
  implementation;
- cover cache lengths around every page boundary;
- validate captured and uncaptured decode paths;
- validate multiple streams and deferred reclamation without a default-stream
  synchronization assumption;
- exercise repeated prefix restore while the original sequence is active and
  after it finishes; and
- run the focused model micromeasure and an end-to-end server workload.

For tinfer:

- compare paged attention with scalar/contiguous references for token-major and
  head-major F32/F16 layouts;
- cover page boundaries, GQA head mapping, and online-softmax combination;
- run `qwen3_transformers_golden` and relevant model graph tests;
- exercise Ling KDA plus MLA snapshot restore and Qwen3.6 hybrid restore;
- test multiple active requests sharing a long prefix; and
- demonstrate bounded memory under a workload which previously allocated every
  request's maximum context eagerly.

Performance acceptance is not simply “paged is faster”. The first target is
bounded memory and removal of full-prefix copying with an acceptable kernel
cost. Report separately:

- manager/index CPU time;
- page allocation or boundary-transition time;
- page-table update time;
- attention-kernel time;
- prefix insertion and restore time;
- resident and reserved bytes; and
- end-to-end prefill throughput, decode throughput, and TTFT.

Do not hide a gather, allocation, synchronization, or representation conversion
inside a broad end-to-end number.

## Completion criteria

The shared component is complete when:

- both runtimes can depend on the same neutral crate API;
- normal crate dependencies are only `rart` and `fast-telemetry`;
- a prefix entry and restored sequences share sealed physical KV pages;
- insertion and aligned restore copy zero KV bytes;
- live caches grow by pages rather than maximum request length;
- strict reservations make managed-memory overcommit impossible;
- hybrid recurrent state remains model-correct and exactly accounted;
- no cache metadata requires cross-thread atomic mutation;
- the token interner and ART metadata shrink on eviction;
- all failure paths are transactional;
- Eider's explicit stream and graph-capture constraints are preserved;
- tinfer's CPU layouts, executor ownership, and F32/F16 modes are preserved; and
- correctness and performance results are recorded for at least one Eider model
  and one tinfer model before the old implementations are removed.

The architectural test is simple: a 10,000-token prefix used by several
requests should occupy one set of sealed KV pages, not one live allocation plus
one checkpoint allocation plus another copied allocation for every restored
request. Only each request's private recurrent state, writable tail, and future
page reservation should be distinct.
