# DeepSeek V4 Flash expert storage

Eider serves the DeepSeek V4 Flash text model from a thin non-expert
checkpoint, resident blockwise-Q3 routed experts, and an optional bounded
cache of original NVFP4 experts. This split keeps the complete model within
the GB10 unified-memory budget without retaining the 156.7 GiB source
checkpoint.

## Resident format

The cold expert format uses eight signed levels (`-7`, `-5`, `-3`, `-1`, `1`,
`3`, `5`, `7`) packed into three bits per weight. Each block of 128
input-channel weights shares one BF16 scale. The resulting storage is `25/64`
byte per weight, including scales.

For the 43-layer, 256-expert checkpoint:

| Resident weights | Routed-expert payload |
| --- | ---: |
| Cold Q3 only | 100.7813 GiB |
| Q3 + 1 original-NVFP4 slot per layer | 101.3495 GiB |
| Q3 + 4 original-NVFP4 slots per layer | 103.0541 GiB |
| Q3 + 8 original-NVFP4 slots per layer | 105.3269 GiB |
| Q3 + 16 original-NVFP4 slots per layer | 109.8725 GiB |

The hot sizes include the per-row tensor scales needed to preserve independent
`w1` and `w3` scales. They exclude the thin checkpoint, CUDA workspaces,
sequence state, and small pointer tables.

Q3 replaced the earlier Q2 serving experiment. On real layer-3 activations,
Q2 accumulated enough expert error to invalidate later learned routing. The
block-128 Q3 path passes a full learned-router-layer gate against an
independent pinned-checkpoint reference: relative L2 below `0.12` and cosine
similarity above `0.99`. Installing every routed original-NVFP4 expert for the
same layer retains the stricter architecture gate of relative L2 below `0.01`
and cosine similarity above `0.999`.

## Preparation

Preparation streams one expert at a time into self-describing per-layer
gate/up and down tables. A manifest is published only after all layers
validate:

```sh
scripts/prepare-deepseek4-experts-streaming.sh
```

The script downloads one pinned source shard, prepares its Q3 layer, rewrites
the shard with only the attention, router, shared-expert, and normalisation
tensors needed for serving, then removes the disposable staging copy. Existing
validated Q3 tables and thin shards are reused.

The optional MTP block is omitted. The resulting thin checkpoint is at most
8.253 GiB. Cold Q3, that thin checkpoint, and eight original-NVFP4 slots per
layer occupy at most about 113.58 GiB, leaving roughly 14.42 GiB of the 128 GiB
unified-memory budget for the host, CUDA workspaces, and sequence state.

## Hot experts

Runtime routing remains on the device. Usage counts are copied to the host
only at an explicit request or maintenance boundary. A hotset refresh ranks
cumulative observations, installs available original-NVFP4 experts into fixed
slots, and leaves every other expert on the complete Q3 path.

The source cache stores one self-validating NVFP4 record per selected
layer/expert pair and enforces a per-layer capacity. It is rebuilt from a JSON
hotset plan with:

```sh
DEEPSEEK4_HOT_CAPACITY=8 \
    scripts/prepare-deepseek4-hotset-streaming.sh hotset-plan.json
```

The plan maps layer numbers to expert indices, for example
`{"0":[4,17,93],"1":[8,42]}`. The preparer downloads one pinned layer shard,
extracts only those experts, removes the disposable shard, and advances.
Layers omitted from the plan are cleared.

A failed multi-matrix refresh clears both overlays so gate/up and down cannot
retain divergent mappings.

## Serving

After preparation:

```sh
scripts/run-eider-deepseek4.sh
scripts/run-pi-eider-deepseek4.sh
```

The server defaults to a 32,768-token context and accepts at most eight cached
original-NVFP4 experts per layer. The runtime sizes each overlay from the
cache's actual contents and retains one fallback slot for an empty layer.
Those defaults are deliberately conservative because device allocations and
host memory share the same 128 GiB physical pool on GB10.
