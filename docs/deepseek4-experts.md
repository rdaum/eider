# DeepSeek V4 Flash expert storage

Eider serves the DeepSeek V4 Flash text model from a thin non-expert
checkpoint and an exact, disk-backed NVFP4 expert store. The complete
156.7 GiB source checkpoint does not need to remain resident.

## Runtime layout

Each routed expert is stored as one self-validating record containing its
original ModelOpt NVFP4 `w1`, `w3`, and `w2` matrices. The 43-layer,
256-expert store occupies about 145.13 GiB on disk.

Serving allocates a fixed number of exact NVFP4 slots per layer. Logical
expert IDs are resolved through an LRU cache, missing records are read into
pinned host staging, and uploads run on a dedicated non-blocking CUDA stream.
The inference stream waits on explicit events before using remapped slot IDs.
A cache miss cannot fall back to a lower-precision weight.

The default eight slots per layer retain about 4.54 GiB of expert weights.
Together with the roughly 8.25 GiB thin checkpoint and pinned upload staging,
this leaves most of the Spark's unified memory available for CUDA workspaces,
sequence state, and the prompt-prefix cache. Increase
`--deepseek-expert-capacity` only after accounting for both device slots and
the shared 128 GiB host/device pool.

The exact path is checked against independent PyTorch references for early
decoder layers, a late learned-router/compressed-attention layer, and the
final hyper-head, normalisation, and vocabulary projection. The retained Q3
implementation is an explicit experiment only: it passes a per-layer error
gate but accumulates enough error across the model to produce unusable text.

## Preparation

Preparation downloads one pinned source shard at a time, writes all exact
expert records for its layer, prepares the corresponding thin checkpoint
shard, validates both products, and removes the disposable source shard:

```sh
scripts/prepare-deepseek4-experts-streaming.sh
```

If an older Q3 artifact is present, the script removes each Q3 layer only
after its exact replacement validates. This permits migration without enough
free disk to retain both complete expert formats.

The optional MTP block is omitted from the thin text checkpoint.

## Serving

After preparation:

```sh
scripts/run-eider-deepseek4.sh
scripts/run-pi-eider-deepseek4.sh
```

The server defaults to eight resident expert slots per layer and a
32,768-token context. Override those limits with
`DEEPSEEK4_EXPERT_CAPACITY`, `EIDER_MAX_CONTEXT_TOKENS`, or the corresponding
`eider-serve` command-line arguments.
