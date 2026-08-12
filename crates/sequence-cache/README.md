# sequence-cache

`sequence-cache` manages reusable model state for autoregressive inference. It
tracks which physical pages belong to each live sequence, shares cached prompt
prefixes between requests, reserves memory before accepting work, and makes
writes across page boundaries transactional. The application supplies the
actual CPU or accelerator storage.

When a model processes a prompt, it produces internal state needed by every
later token. For a transformer this is usually the attention key/value cache.
Keeping that state for a repeated prompt can avoid most of the next request's
prefill, but the state is large, mutable, and normally split into fixed-width
physical pages. Reusing it safely requires more than a map from tokens to
buffers. An inference runtime must also answer:

- Which physical pages are owned by a live sequence?
- When is a page immutable and safe to share?
- How much memory has already been promised to admitted requests?
- What happens when one model operation writes across several pages?
- How is an append rolled back after an allocation, kernel, or page-table
  failure?
- Which retained prefix should be evicted under pressure?
- How can a request branch from a shared prefix without copying all prior state?

This crate provides that ownership, transaction, indexing, and accounting
machinery. The embedding runtime provides the actual storage through a
`PageBackend` implementation.

It was designed for transformer KV caches, but it deliberately does not know
what a key or value tensor is. A page may bundle KV tensors from many layers,
compressed attention state, or another form of position-indexed model state.
Non-paged state such as recurrent tensors can be retained alongside a prefix in
a runtime-defined snapshot.

## Why use it?

A basic paged KV cache is straightforward when every sequence owns every page
and failure is fatal. Production serving adds harder requirements:

- **Prompt reuse.** Identical page-aligned token prefixes should share the same
  immutable physical pages rather than recompute or copy them.
- **Strict admission.** Accepting a request should guarantee enough page,
  private-state, and page-table capacity for its declared maximum length.
- **Independent compute and storage granularity.** A model should be able to
  prefill 2,048 tokens in one operation even when physical cache pages contain
  128 tokens.
- **Transactional mutation.** A multi-page append must either commit as one
  logical operation or restore the exact previous sequence and accounting
  state.
- **Safe divergence.** A continuation may share complete prefix pages while
  receiving a private copy of an incomplete tail.
- **Bounded reuse.** Retained prefixes need exact accounting and predictable
  least-recently-used eviction under memory pressure.
- **Backend independence.** CPU buffers, CUDA slabs, and other storage systems
  need different allocation and synchronization, but the ownership rules are
  the same.

`sequence-cache` centralises those rules so each model or runtime backend does
not need to reinvent them.

## What the crate owns

The crate owns the logical state machine:

- admission and future-capacity reservations;
- generational sequence and page handles;
- ordered logical page tables;
- exact multi-page append reservations;
- whole-reservation commit and abort;
- optional partial commit for speculative execution;
- page validity and sealed-page transitions;
- page-aligned prefix retention and longest-prefix lookup;
- shared-page reference counts and unaligned-tail copy-on-write;
- least-recently-used prefix eviction;
- immediate and deferred page retirement accounting;
- structural metrics and invariant validation.

The runtime-owned backend supplies:

- physical page allocation and recycling;
- the contents and layout of one page bundle;
- backend-native page tables;
- copying a partially valid tail page;
- rollback state for storage touched by an append;
- synchronization with model execution;
- physical retirement and asynchronous reclamation.

```text
request scheduler / model runtime
              │
              │ admit, reserve, commit, retain, finish
              ▼
       SequenceCache<B, S>
       ├── ownership and accounting
       ├── append transactions
       └── retained-prefix index
              │
              │ PageBackend
              ▼
       physical runtime storage
       ├── CPU buffers
       ├── CUDA page slabs
       └── model-specific state pages
```

## What it is not

`sequence-cache` is not:

- a KV tensor container;
- an attention kernel or PagedAttention implementation;
- a CUDA, CPU, or accelerator allocator;
- a model scheduler;
- a tokenizer or prompt cache;
- a concurrent cache protected by internal locks.

The combined `SequenceCache` and a KV-aware `PageBackend` form a reusable KV
cache. The standalone crate is the backend-independent sequence-state manager
inside that system.

## When it is a good fit

Use this crate when a runtime has fixed-width physical pages and needs one or
more of:

- reusable prompt prefixes;
- multiple live sequences sharing storage;
- exact admission control;
- transactional writes spanning page boundaries;
- speculative partial commit;
- model-specific retained state;
- storage implemented outside the cache manager.

It is probably unnecessary for a single-sequence program with a contiguous
cache, no prefix reuse, and no need to recover from failed execution.

## Adding it to a project

The crate currently lives in the Eider repository and has not yet been released
as an independent package. A project can use a checked-out revision directly:

```toml
[dependencies]
sequence-cache = { path = "../eider/crates/sequence-cache" }
```

For a Git dependency, point Cargo at the Eider repository and pin `rev` to the
exact commit the application has tested. Implement `PageBackend` for the
runtime's storage, choose a `RetainedSnapshot` type for any non-paged state, and
construct `SequenceCache<Backend, Snapshot>`. Use `()` when there is no
additional retained state.

### Paged CPU KV-cache example

[`examples/cpu_paged_kv.rs`](examples/cpu_paged_kv.rs) implements a complete
in-memory CPU backend. Each physical page owns token-major key and value
matrices for every attention layer, following the simple storage shape used by
CPU inference engines such as [tinfer](https://github.com/rdaum/tinfer). The
example demonstrates:

- backend-owned physical storage behind compact page handles;
- a model-sized write scattered directly across several cache pages;
- append-only transactions without unnecessary content snapshots;
- actual allocation reuse when pages are recycled;
- a causal-attention read through the enlarged page table before commit;
- retaining an aligned prompt prefix; and
- restoring the same physical pages for a later request.

Run it with:

```sh
cargo run -p sequence-cache --example cpu_paged_kv
```

The generated floating-point rows stand in for a model's key and value
projection output. The ownership, transaction, page-table, recycling, and read
paths are real. Cache accounting covers the managed KV payload; ordinary Rust
collection and allocator metadata remains process overhead.

### CUDA paged-state example

[`examples/cuda_paged_state.rs`](examples/cuda_paged_state.rs) implements a
small accelerator backend with stable device allocations and a device-resident
logical page table. It publishes the complete enlarged table, launches one CUDA
kernel which scatters rows directly across three physical pages, synchronizes
before commit, and reads the committed result back for validation.

The example uses the CUDA Driver API directly and does not depend on Eider's
CUDA crates. It requires Linux, an NVIDIA driver, and a CUDA-capable device; a
CUDA toolkit is not required because the embedded PTX is compiled by the
driver. Run it with:

```sh
cargo run -p sequence-cache --features cuda-example --example cuda_paged_state
```

For clarity, commit and retirement synchronize the stream before shortening a
page table or recycling storage. A production backend can replace those host
waits with event-based deferred retirement while preserving the same ownership
contract.

## Core model

### Pages and sequences

A sequence owns an ordered list of logical pages. Each backend page is a bundle:
it may identify one physical slot whose corresponding storage exists in every
attention layer.

Only the final page of a sequence may be partially valid. That tail must be
private and writable. Once a page becomes full it is sealed, immutable, and
eligible for sharing through a retained prefix.

`PageId` and `SequenceId` are generational handles. Reusing an internal arena
slot does not make an old handle valid again.

### Retained prefixes

A retained prefix consists of:

1. a page-aligned token prefix;
2. references to its sealed physical pages; and
3. a `RetainedSnapshot` containing any non-paged state needed to resume.

Token runs are interned one physical-page block at a time. An adaptive radix
tree indexes the resulting block identities and returns the longest retained
prefix. Lookup misses do not intern new blocks, so probing arbitrary prompts
does not grow the index.

Retaining a prefix does not copy its physical pages. A restored sequence takes
another reference to the same sealed pages. Prefix entries are evicted in
least-recently-used order when capacity is required; live sequences keep their
references, so eviction never invalidates active work.

The helper `cacheable_prefix_tokens(prompt_tokens)` returns the largest aligned
position strictly before the final prompt token. This matches runtimes which
retain a prompt immediately before computing its final token for decode.

### Copy-on-write branches

`SequenceCache::branch` creates a divergent continuation from a live sequence
whose position is not page-aligned. Complete pages remain shared. The backend
copies only the valid rows of the final partial page into a new private tail.

Aligned sequences should normally be represented by retained prefixes rather
than `branch`.

## Multi-page append transactions

Physical page size does not limit model execution size. A caller may reserve any
positive number of rows up to the sequence's admitted maximum position. One
reservation can cover:

1. unused rows in an existing private tail page;
2. zero or more newly allocated full pages; and
3. a final newly allocated partial page.

For example, appending 2,048 rows at position 64 with 128-token pages produces
17 ordered writable segments: 64 rows in the existing tail, fifteen full new
pages, and 64 rows in the final page. The model can still execute one 2,048-row
prefill operation and scatter its output directly into those segments.

```text
input rows       0..64       64..1984        1984..2048
                 │           │                │
physical pages   tail[64..]   15 full pages    final[0..64]
```

### Reservation lifecycle

`reserve_append` performs the control-plane preparation before model execution:

1. validates the requested span against the admitted maximum;
2. allocates every required page;
3. asks the backend to prepare an `AppendTransaction` journal;
4. publishes the complete logical page table, including new pages; and
5. returns an `AppendReservation` describing every writable segment.

Publishing the full table before execution matters for causal prefill: later
rows in one compute chunk may attend to KV rows written into earlier newly
allocated pages.

The runtime then borrows all covered pages through `with_append_pages` and
writes directly into physical storage. `with_append_reservations` provides the
same ordered view for a batch of sequences.

After execution, the caller must consume the reservation with exactly one of:

- `commit_append(reservation, committed_rows, context)`; or
- `abort_append(reservation, context)`.

Reservations carry a nonce. A stale, copied, or mismatched reservation cannot
commit a different pending append.

### Exact and partial commit

The common prefill path commits every reserved row. `committed_rows` may also be
a non-empty prefix of the reservation for speculative execution. A partial
commit:

- advances the sequence only by the accepted prefix;
- keeps pages containing accepted rows;
- releases pages used only by the unused suffix;
- restores the released pages to admission accounting; and
- leaves unused rows in the final kept page invalid and writable.

A zero-row result should use `abort_append`, not partial commit.

Backends whose physical layout can overwrite older tail lanes during a
speculative span must journal those contents in `prepare_append` and restore any
uncommitted lanes during partial commit or abort.

### Synchronization responsibility

The crate cannot observe device execution directly. The runtime and backend
must ensure model writes have completed before storage is shortened, restored,
or recycled. For an asynchronous accelerator backend, this normally means that
fallible stream or event synchronization happens before the backend publishes a
shorter committed table or makes released pages reusable.

Do not use a large contiguous temporary cache followed by a copy merely to fit
this API. The reservation exposes every destination page specifically so model
output can be written or scattered directly.

## Typical lifecycle

The following example assumes the application has implemented `PageBackend`
and, optionally, a model-specific `RetainedSnapshot`.

```rust,ignore
use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, CacheConfig, SequenceCache,
};

let config = CacheConfig {
    page_tokens: 128,
    max_managed_bytes: 64 << 30,
    max_snapshot_bytes: 1 << 30,
    max_prefix_entries: None,
    emergency_bytes: 64 << 20,
};
let mut cache = SequenceCache::new(config, backend)?;
let mut context = backend_context();

// Reuse the longest retained prefix of this prompt, if one exists.
let prefix = cache.lookup_prefix(&prompt_tokens);
let request = AdmissionRequest {
    max_position,
    private_state_bytes: model_private_state_bytes(max_position),
    page_table_bytes: backend_page_table_bytes(max_position),
    allow_emergency: false,
};
let sequence = match cache.admit(
    prefix,
    request,
    &mut context,
    |snapshot, position| restore_retained_state(snapshot, position),
)? {
    AdmissionOutcome::Admitted(sequence) => sequence,
    AdmissionOutcome::WouldBlock => return scheduler.retry_later(),
};

// Reserve one compute-sized prefill span, independent of physical page size.
let rows = 2_048;
let reservation = cache.reserve_append(sequence, rows, &mut context)?;

let execution = cache.with_append_pages(&reservation, |backend, pages| {
    model.prefill_into_pages(
        backend,
        pages.iter().map(|page| (page.page(), page.segment())),
        reservation.start_position(),
        rows,
    )
});

match execution {
    Ok(()) => cache.commit_append(reservation, rows, &mut context)?,
    Err(error) => {
        cache.abort_append(reservation, &mut context)?;
        return Err(error.into());
    }
}

// Prefixes can be retained only at a sealed page boundary.
if cache.page_table(sequence)?.position().is_multiple_of(config.page_tokens) {
    let snapshot = capture_retained_state()?;
    cache.retain_prefix(sequence, &prompt_tokens, snapshot, &mut context)?;
}

cache.finish(sequence, &mut context)?;

// Admission polls automatically; runtimes may also poll between requests.
cache.reclaim_deferred(&mut context)?;
```

Production code should also ensure that a model failure is synchronized before
calling `abort_append` if outstanding work could still access the reservation's
pages.

## Implementing `PageBackend`

`PageBackend` is the direct storage contract. Its associated types let each
runtime expose its native page handle, operation context, transaction journal,
and error type.

```rust,ignore
impl PageBackend for MyBackend {
    type Page = MyPhysicalPage;
    type Context<'a> = MyExecutionContext<'a>;
    type AppendTransaction = MyAppendJournal;
    type Error = MyBackendError;

    // ...
}
```

### Associated types

- `Page` identifies one physical page bundle. The manager treats it as opaque.
- `Context<'a>` carries operation-local runtime state, such as a CUDA stream and
  the active sequence's device page table.
- `AppendTransaction` stores backend-owned rollback state for one pending
  append.
- `Error` reports storage and synchronization failures.

### Geometry and allocation

- `page_bytes()` must return the exact bytes occupied by one page bundle.
- `page_capacity()` optionally reports a hard slot count for preallocated
  storage. Admission enforces it in addition to the byte budget.
- `allocate_page()` returns either one complete writable allocation or an
  error with no side effects.
- `rollback_page()` infallibly returns an allocation which was never published.
- `copy_partial_page()` creates a private copy containing exactly the valid
  prefix of an existing tail.

### Append preparation

`prepare_append()` receives ordered `BackendAppendPage` descriptions before the
new table is published. Each description includes:

- the physical page;
- its first writable row;
- the corresponding input-row offset;
- the number of rows; and
- whether the page existed before the reservation.

The returned transaction should snapshot any pre-existing storage that an
abort or partial commit might need to restore. Newly allocated pages do not
usually require content snapshots because they can simply be released.

### Commit and abort

`commit_append()` receives a `BackendAppendCommit` containing:

- the complete page table after commit;
- pages which became sealed;
- suffix-only pages to release;
- the number of committed input rows; and
- the new logical position.

Publishing the table, sealing pages, restoring unused tail contents, and
reclaiming released pages must be atomic from the manager's perspective. On
failure the reservation remains pending and can be retried or aborted.

`abort_append()` restores the prior table and position, restores backend-owned
contents from the transaction journal, and reclaims every newly reserved page.
On failure the entire reservation remains pending.

### Page-table updates

`update_page_table()` atomically publishes a complete logical page ordering and
position. It is used for admission, prefix restoration, branching, and initial
reservation publication. A failure must leave the previously published table
and position intact.

### Retirement

`retire_pages()` transfers a complete batch to the backend. It must either take
every page or return `RetireError` containing all pages unchanged and in their
original order.

If storage cannot be reused immediately, return the number of deferred pages in
`RetireOutcome`. Those pages remain charged to the cache until
`poll_reclaimed()` reports them reusable. Set `retirement_is_immediate()` only
when successful retirement makes slots immediately available during admission
planning.

### Backend checklist

Before treating a backend as production-ready, verify:

- allocation failure after several successful page allocations;
- transaction-preparation failure;
- failure while publishing the expanded reservation table;
- abort after successful multi-page execution;
- failure during abort and a successful retry;
- exact and partial commit across page boundaries;
- restoration of overwritten private-tail contents;
- synchronization before shortening a table or recycling storage;
- prefix restore and unaligned-tail copy-on-write;
- immediate and deferred retirement accounting;
- stale reservation rejection.

## Capacity and admission

Admission is strict rather than optimistic. For each active sequence the cache
accounts for:

- exact private-state bytes;
- exact page-table bytes; and
- every future page needed to reach `AdmissionRequest::max_position`.

Global accounting also includes:

- unique resident pages, counted once even when shared;
- pages awaiting deferred retirement;
- retained snapshot bytes; and
- pages held only by retained prefixes.

`AdmissionOutcome::WouldBlock` is normal scheduler pressure, not an error. The
cache may first evict retained prefixes, but it never evicts storage referenced
by a live sequence.

`emergency_bytes` reserves part of the managed budget for requests whose policy
sets `allow_emergency`. It does not create additional storage; it prevents
ordinary admissions from consuming the configured margin.

`CacheStats` exposes the exact current accounting snapshot. Backends with a
fixed physical pool should report `page_capacity()` so admission checks both
bytes and actual slots.

## Concurrency

`SequenceCache` serializes mutation through `&mut self` and is deliberately
`!Sync`. A serving runtime may own it in an actor or protect it with its own
scheduler lock. Backend implementations do not need internal locking for calls
made through one manager.

The operation context is explicit because storage synchronization is a runtime
concern. A CPU backend might carry only a mutable page table; an accelerator
backend might carry a stream, events, and a stable device table.

## Errors and recovery

Capacity pressure is returned as `AdmissionOutcome::WouldBlock`. Invalid
handles, positions, reservations, configuration, and backend failures use
`CacheError`.

The important recovery rule is that a failed append commit or abort does not
discard the pending reservation. Once the backend condition is corrected, the
caller can retry the operation or abort it. Allocation and initial publication
failures roll back unpublished pages before returning.

Calling `finish` while an append is pending is rejected. The caller must commit
or abort the reservation first.

## Metrics and inspection

`CacheMetrics` exports `fast-telemetry` counters, histograms, and gauges under
the `sequence_cache` prefix.

Counters cover prefix hits and misses, restored tokens, admissions, prefix
evictions, page allocation and recycling, sealing, copy-on-write, retirement,
and backend failures. Histograms cover lookup, insertion, eviction, admission,
and snapshot-restore latency.

Gauges mirror `CacheStats`, including active sequences, resident and reserved
pages, deferred retirements, snapshot bytes, page-table bytes, reclaimable
prefix-only bytes, and total managed bytes.

`SequenceCache::validate()` recomputes ownership, reference counts, page
validity, alignment, accounting, and metric gauges from first principles. It is
intended for tests and health checks rather than the inference hot path.

## Testing

Run the crate tests with:

```sh
cargo test -p sequence-cache
```

The conformance suite uses an in-memory backend with failure injection. It
covers multi-page reservation geometry, allocation and publication rollback,
commit and abort retry behaviour, partial commit, page sealing, prefix sharing,
copy-on-write, recycling, metrics, generational handle staleness, and a
deterministic state machine which validates invariants after every operation.

An external backend should reuse the same scenarios against its physical
storage and synchronization implementation. Passing the manager's in-memory
tests does not prove that a device backend obeys ordering or rollback rules.

## Benchmarks

The micromeasure suite isolates manager, index, page-table, and accounting
overhead with a storage-free backend:

```sh
cargo bench -p sequence-cache --bench sequence_cache
```

It includes one-page and sixteen-page append transactions, a 2,048-row append
starting in a partial 128-token page, exact and partial commit lifecycles,
retained-prefix lookup and restore, and copy-on-write branching.

These measurements intentionally exclude physical allocation, device
synchronization, model execution, and KV writes. Backend-specific projects
should add separate benchmarks for those costs.

## Current status

The crate is currently developed inside the Eider workspace and is being
prepared as a reusable public package. Its API is already consumed by several
different CUDA cache layouts and is being evaluated by an independent CPU
inference runtime. Until an independent release policy is established, pinning
an exact repository revision is safer than assuming semver stability.
