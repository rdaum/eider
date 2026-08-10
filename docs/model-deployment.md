# Supported model deployment

Status: implemented

## Goal

Simplify Eider deployment to choosing a supported model and starting the
server. Operators should not need to run `hf download`, choose a local directory
layout, prepare model-specific caches manually, or reproduce the defaults from
one of the repository launch scripts.

The intended deployment interface is:

```sh
eider-serve agents-a1
eider-serve step-3.7-flash
eider-serve gemma-4-26b-a4b-nvfp4
```

On the first start, Eider resolves the selected checkpoint, downloads absent
files into the Hugging Face cache, prepares any Eider-specific artifacts, and
then starts serving. Later starts reuse both caches and must work without
network access once the selected revision is complete.

This is deliberately a catalogue of models Eider supports, not a general
Transformers model loader. A Hugging Face repository being downloadable does
not imply that its architecture, quantization, tokenizer, chat protocol, or
weight layout is compatible with Eider.

## Current deployment friction

`eider-serve` currently accepts only a checkpoint directory. Deployment
therefore requires several external steps:

1. Discover the correct repository and download command.
2. Reserve enough disk space and place every shard in an expected directory.
3. Select the matching Qwen, Agents-A1, or Step launch script.
4. For Step, run `step37-experts prepare` before starting the server.
5. Preserve model-specific context, paging, and served-name arguments from the
   script.
6. Keep the checkpoint directory writable because prepared Qwen and Step
   weights currently live below its `.eider-cache` directory.

These steps make a deployment depend on repository knowledge that the binary
already has or should have. They also make immutable Hugging Face snapshots an
awkward fit for the current derived-cache layout.

## User interface

### Supported models

The positional argument becomes a stable model ID from Eider's built-in
catalogue:

```sh
eider-serve agents-a1
eider-serve step-3.7-flash --step-expert-capacity 192
eider-serve gemma-4-26b-a4b-nvfp4
eider-serve muse-glimmer-30b-nvfp4
```

The catalogue supplies the served model name and model-specific runtime
defaults. Existing runtime flags remain available as deliberate overrides.

`--offline` prohibits network access and produces an actionable error when the
pinned snapshot is incomplete:

```sh
eider-serve --offline agents-a1
```

### Local checkpoints

Development and checkpoint-conversion work retains an explicit local path:

```sh
eider-serve --model-dir ./models/agents-a1-nvfp4
eider-serve --model-dir ./models/muse-glimmer-30b-nvfp4 \
  --dflash-gguf ./dflash-kquant.gguf
```

`--model-dir` is mutually exclusive with the catalogue ID. Local checkpoints
go through the same architecture and weight-format validation, but Eider does
not claim that arbitrary local Transformers checkpoints are supported. A
missing local path is always an error; it never triggers a large download based
on a possible path typo.

### Fetching ahead of startup

Automatic fetch-on-start is the primary deployment path. A separate fetch
command is useful for image construction and offline installation, and shares
the exact resolver and preparation implementation:

```sh
eider model fetch agents-a1
eider model fetch step-3.7-flash --prepare
```

This command is not a prerequisite for serving. `--prepare` currently builds
the Step-3.7 expert records; Qwen preparation remains part of model startup.

## Supported model catalogue

Each Eider release contains a small, reviewed catalogue. A model entry includes:

- Stable Eider model ID.
- Hugging Face repository and immutable commit revision.
- Accepted checkpoint architecture and quantization metadata.
- Required tokenizer, configuration, index, and weight files.
- Runtime family: Qwen3.6-compatible or Step-3.7.
- Preparation strategy and prepared-format version.
- Default served model name, context limit, and paging settings.

The current catalogue includes Qwen3.6 35B-A3B, Agents-A1, Step-3.7-Flash,
Muse Glimmer 30B, Nemotron 3 Puzzle 75B-A9B, and both supported Gemma 4
26B-A4B weight formats:
the NVIDIA ModelOpt NVFP4 checkpoint (`gemma-4-26b-a4b-nvfp4`) and Google's
upstream BF16 instruction-tuned checkpoint (`gemma-4-26b-a4b-it`). Both Gemma
entries use the Gemma 4 text runtime; image, video, and audio inputs remain
outside Eider's text-serving interface.

Muse Glimmer uses the pinned Inferact ModelOpt NVFP4 target and the pinned
`dflash-kquant.gguf` companion from Meta's official GGUF repository. Catalogue
resolution fetches both immutable revisions; `--dflash-gguf` supplies the
companion for a local development checkpoint. Greedy requests use DFlash while
sampled requests retain target-only decoding. Recipient-framed reasoning and
ATEM function calls are translated into the same Responses and Chat
Completions events as other model families; image and video inputs are not
exposed.

Conceptually:

```rust
struct ModelSpec {
    id: &'static str,
    repository: &'static str,
    revision: &'static str,
    architectures: &'static [&'static str],
    weight_format: WeightFormat,
    runtime: RuntimeFamily,
    preparation: Preparation,
    defaults: ServingDefaults,
}
```

Revisions are commit hashes, not moving branch names. Updating a supported
checkpoint is an Eider change with normal validation and release notes. Startup
logs record the catalogue ID, repository, resolved revision, checkpoint path,
and derived-artifact path.

The catalogue controls selection before any large download. The Hugging Face
client is transport and cache infrastructure; it is not model discovery or a
compatibility mechanism.

## Resolution and startup flow

Starting a catalogue model follows one path:

1. Look up the model ID and its pinned revision.
2. Resolve the revision in the Hugging Face cache.
3. Fetch small metadata files first when the snapshot is absent.
4. Validate architecture, quantization, tokenizer metadata, and the weight
   index against the catalogue entry.
5. Determine the exact weight shards and expected download size.
6. Check available disk space and fetch only the required files.
7. Resolve the Eider artifact directory for this model revision.
8. Validate or prepare required SM12x and expert-paging artifacts.
9. Pass the immutable checkpoint directory and writable artifact directory to
   the existing runtime loader.
10. Start the API only after the model is ready.

Failures identify the stage, model ID, revision, required file, and relevant
cache path. A disk-space failure should report checkpoint bytes, estimated
prepared-artifact bytes, available bytes, and the filesystem that is short.

The server continues to validate checkpoint metadata even after catalogue
selection. A corrupted cache or repository whose pinned contents do not match
the expected architecture must fail before CUDA allocations begin.

### Hugging Face client

Use the async [`hf-hub`](https://github.com/huggingface/hf-hub) client directly
from the Tokio-based server. The resolver should use its repository and
snapshot APIs rather than reproduce Hub URLs, authentication, Xet transfers,
cache bookkeeping, or concurrent-download coordination.

For a catalogue model, the resolver:

- Creates a model-repository handle for the catalogue repository.
- Passes the pinned commit hash as the snapshot revision.
- Uses allow patterns derived from the catalogue and validated weight index so
  unrelated repository files are not fetched.
- Leaves `local_dir` unset so completed files remain in the shared Hugging Face
  content-addressed cache.
- Maps `--offline` to `local_files_only(true)`.
- Installs a `ProgressHandler` that translates download events into `tracing`
  events and deployment metrics.
- Uses the returned snapshot directory as the immutable checkpoint root.

Before the large snapshot download, Eider can use the repository listing and
file-metadata APIs to resolve the commit, verify required paths, collect file
sizes, and perform disk-space preflight. Hub errors such as missing revisions,
authentication requirements, forbidden repositories, and rate limiting remain
distinct deployment errors.

The initial implementation should depend on the asynchronous API only. The
server already owns a Tokio runtime, so enabling the crate's blocking client
would add another runtime without simplifying this path.

## Cache layout

### Hugging Face snapshots

Original repository files remain in the Hugging Face cache. Eider respects the
cache location selected by `hf-hub`: `HF_HUB_CACHE`, then the legacy
`HUGGINGFACE_HUB_CACHE`, then the `hub` directory below `HF_HOME`. When
`HF_HOME` is unset, the crate uses `$XDG_CACHE_HOME/huggingface` or
`~/.cache/huggingface`. Eider does not copy weight shards into its own
directory or invent a parallel snapshot layout.

Snapshots are treated as immutable and may be shared by Eider, Python tooling,
and other processes. Authentication uses the normal Hugging Face precedence:
an explicitly configured client token, `HF_TOKEN`, `HF_TOKEN_PATH`, or the
token file below `HF_HOME`. Eider does not copy tokens into its own
configuration or logs and honours `HF_HUB_DISABLE_IMPLICIT_TOKEN`.

### Eider artifacts

Prepared files are reconstructible and belong under the XDG cache directory:

```text
$XDG_CACHE_HOME/eider/
└── models/
    └── <repository>/
        └── <revision>/
            ├── qwen36-experts-v1/
            └── step37-experts-v1/
```

When `XDG_CACHE_HOME` is unset, the default is `~/.cache/eider`. Repository
components are encoded so they cannot introduce path traversal or ambiguous
names.

The artifact key contains at least:

- Repository identity and resolved revision for catalogue models.
- A deterministic checkpoint metadata fingerprint for local models.
- Prepared-format version.
- Hardware target when the prepared representation is architecture-specific.

The format version remains in the directory name so incompatible Eider
releases can rebuild without interpreting old bytes. Cache contents are never
used merely because a directory exists: existing headers, shapes, source
identity, and completion markers are validated first.

No persistent configuration is required for the initial design. If catalogue
overrides are introduced later, they belong below `$XDG_CONFIG_HOME/eider`.
Operational logs or durable process state would belong below
`$XDG_STATE_HOME/eider`, not in the artifact cache.

### Containers and services

A container deployment needs two writable cache locations, which may share one
mounted parent volume:

```sh
docker run \
  -e XDG_CACHE_HOME=/var/cache \
  -e HF_HOME=/var/cache/huggingface \
  -v eider-cache:/var/cache \
  ... \
  eider eider-serve agents-a1
```

The application image contains the Eider binary and CUDA dependencies, not the
model weights. An offline deployment can populate the same volume during image
installation or a separate fetch job, then start with `--offline`.

## Preparation and concurrency

Model preparation becomes part of model resolution rather than shell-script
orchestration. The catalogue declares whether a model needs no preparation,
Qwen SM12x expert repacking, or the Step disk-backed expert format.

The Hub client already serializes concurrent fetches of the same
content-addressed blob and publishes completed files from `.incomplete` paths.
Eider does not add a second lock around snapshot downloads. Its own locking is
limited to derived artifacts, where preparation must retain the current
resumable behaviour while supporting concurrent starts safely:

- Take a per-artifact advisory lock before writing.
- Recheck completeness after acquiring the lock.
- Write records and manifests to temporary paths in the destination
  filesystem.
- Atomically rename completed files.
- Preserve valid completed layers after interruption.
- Never write into the Hugging Face snapshot.
- Log layer progress and total prepared bytes through `tracing`.

One process may prepare while another waits, but two processes must not rebuild
or partially consume the same artifact. A process that cannot acquire the lock
reports what it is waiting for and continues once the active preparer commits a
valid result.

Step preparation can require roughly another 100 GiB beyond the downloaded
checkpoint, so its catalogue entry must provide a conservative artifact-size
estimate. Automatic startup is convenient only if it fails before downloading
or preparing into an obviously undersized filesystem.

## Component boundaries

The deployment resolver belongs above the inference runtime:

```text
catalogue model ID ─┐
                    ├─> model resolver ─> ResolvedModel ─> inference actor
explicit local path ┘

ResolvedModel
├── checkpoint_dir     immutable original files
├── artifact_dir       writable Eider-derived files
├── model identity     catalogue ID and revision, or local fingerprint
├── runtime family
└── serving defaults
```

The `infer` crate remains directory-oriented and network-free. Qwen and Step
loaders accept an explicit artifact root; their legacy entry points retain the
old local `.eider-cache` default for probes and benchmarks. The Eider server
binary owns catalogue lookup, Hugging Face access, offline policy, and
deployment-facing errors.

The API actor continues to validate and select the runtime from checkpoint
metadata. The catalogue may select expected behaviour, but it does not bypass
the existing format checks.

## Script migration

The model-specific server scripts currently encode useful defaults but should
not remain a second deployment interface. Once catalogue startup covers their
behaviour:

- Move Agents-A1, Qwen, and Step defaults into catalogue entries.
- Integrate Step preparation into the resolver.
- Update Pi launchers to depend only on the served catalogue model name.
- Replace server launch scripts in documentation with `eider-serve <model>`.
- Remove scripts once they no longer provide unique development functionality.

This leaves one documented production path while preserving focused probes and
benchmark scripts.

## Operational behaviour

Startup emits structured events for model resolution, metadata validation,
download progress, preparation progress, cache hits, and final resolved paths.
The existing metrics endpoint should add counters and gauges for:

- Snapshot resolution hits and misses.
- Downloaded bytes and download failures.
- Preparation cache hits and rebuilds.
- Preparation bytes written and elapsed time.
- Time spent waiting on download or preparation locks.

Downloads honour cancellation and reuse any blobs already complete in the
Hugging Face cache. The client uses incomplete temporary files and atomic cache
publication, but Eider does not promise byte-range resumption of an interrupted
individual file; with the current client that file may restart. Authentication
failures, gated repositories, offline cache misses, revision mismatches, and
insufficient disk space remain distinct errors rather than collapsing into
"model directory does not exist".

## Security and reproducibility

- Only compiled catalogue entries may trigger automatic repository downloads.
- Revisions are immutable commit hashes.
- Repository code is never downloaded or executed.
- `trust_remote_code` has no equivalent in Eider.
- Only the metadata and weight files required by the catalogue are fetched.
- Hugging Face credentials are read through its established environment or
  token configuration and are redacted from errors.
- Resolved revision and model identity are visible in logs and metrics.

Local directories remain an explicit trusted-operator input and cannot be
confused with catalogue names.

## Non-goals

- Loading arbitrary Hugging Face architectures or quantization formats.
- Inferring runtime compatibility from model tags or repository names.
- Following `main` automatically at server restart.
- Copying Hugging Face snapshots into an Eider-owned model store.
- Hiding the disk and startup cost of Step expert preparation.
- Making network access a dependency after a pinned snapshot and its artifacts
  are complete.

## Implementation sequence

1. Introduce `ResolvedModel` and pass an explicit artifact directory through
   Qwen and Step loaders. Move `.eider-cache` outputs to the XDG cache without
   changing their prepared formats. Implemented.
2. Add the built-in model catalogue and local-path validation behind one model
   resolver. Keep this phase offline and test it with fixture directories.
   Implemented.
3. Add `hf-hub` snapshot resolution, pinned filtered downloads, authentication,
   progress reporting, disk preflight, and offline mode. Implemented.
4. Integrate Step expert preparation and Qwen repacking into startup with
   cross-process locking and atomic completion. Implemented.
5. Move model defaults out of server scripts, update deployment documentation,
   and remove redundant launchers. Implemented.
6. Add the optional `eider model fetch` interface if container and offline
   deployment workflows still need a separate prefetch command. Implemented.

## Acceptance criteria

- A clean machine with sufficient disk space can run a supported model using
  one server command and normal Hugging Face authentication.
- Restarting with populated caches performs no network transfer or preparation.
- `--offline` starts from complete caches and fails clearly from incomplete
  ones.
- Checkpoint snapshots remain unmodified and can be mounted read-only.
- Derived artifacts live under the XDG cache root and are safely shareable
  across processes.
- The selected model, repository, immutable revision, and cache paths are
  observable at startup.
- Unsupported catalogue names and incompatible local checkpoints fail before
  large downloads or CUDA allocation.
- Qwen, Agents-A1, and Step retain their current validated runtime behaviour and
  model-specific defaults.
