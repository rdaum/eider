"""Independent eager PyTorch check of the official packed checkpoint, one prompt."""

import argparse
import json
import mmap
import struct
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
device = "cuda"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("gguf", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
file = args.gguf.open("rb")


def read(fmt):
    return struct.unpack("<" + fmt, file.read(struct.calcsize("<" + fmt)))[0]


def string():
    return file.read(read("Q")).decode("utf-8")


def metadata(kind):
    if kind == 8:
        return string()
    if kind == 9:
        element, count = read("I"), read("Q")
        # Only architecture metadata is needed, not the embedded vocabulary.
        if count > 1024:
            for _ in range(count):
                metadata(element)
            return None
        return [metadata(element) for _ in range(count)]
    return read(
        {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }[kind]
    )


assert file.read(4) == b"GGUF" and read("I") == 3
ntensors, nmetadata = read("Q"), read("Q")
config = {}
for _ in range(nmetadata):
    key = string()
    config[key] = metadata(read("I"))
assert config["general.architecture"] == "flint"
tensors = {}
for _ in range(ntensors):
    name, rank = string(), read("I")
    dims = [read("Q") for _ in range(rank)]
    tensors[name] = {"dims": dims, "kind": read("I"), "offset": read("Q")}
alignment = config.get("general.alignment", 32)
data_offset = (file.tell() + alignment - 1) // alignment * alignment
data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
e2m1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)
u = np.arange(128)
exp = u >> 3
frac = u & 7
scale = np.where(exp == 0, frac * 2.0**-9, (1 + frac / 8) * 2.0 ** (exp - 7)).astype(
    np.float32
)


def raw(name, expert=None):
    t = tensors[name]
    shape = tuple(reversed(t["dims"]))
    offset = data_offset + t["offset"]
    n = np.prod(shape)
    if t["kind"] == 0:
        return np.frombuffer(data, "<f4", int(n), offset).copy().reshape(shape)
    if t["kind"] == 30:
        b = np.frombuffer(data, "<u2", int(n), offset).astype(np.uint32) << 16
        return b.view(np.float32).reshape(shape)
    assert t["kind"] == 40
    if expert is not None:
        shape = shape[1:]
        n = np.prod(shape)
        offset += int(expert * n // 64 * 36)
    b = np.frombuffer(data, np.uint8, int(n // 64 * 36), offset).reshape(-1, 36)
    packed = b[:, 4:].reshape(-1, 4, 8)
    nib = np.concatenate((packed & 15, packed >> 4), axis=-1)
    value = e2m1[nib] * scale[b[:, :4], None]
    global_scale = raw(name.replace(".weight", ".scale")).flatten()
    return value.reshape(shape), float(global_scale[0 if expert is None else expert])


def tensor(name):
    return torch.from_numpy(raw(name)).to(device)


def bf(x):
    return x.to(torch.bfloat16).float()


def linear(x, name, expert=None, round_weights=True, output_bf=True):
    value, glob = raw(name, expert)
    w = torch.from_numpy(value).to(device)
    if round_weights:
        w = bf(w * glob)
        y = F.linear(x, w)
    else:
        y = F.linear(x, w) * glob
    return bf(y) if output_bf else y


def norm(x, name):
    return bf(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * tensor(name))


def mlp(x, p, suffix="", expert=None):
    rounded = expert is None
    g = linear(x, p + ".ffn_gate" + suffix + ".weight", expert, rounded)
    u = linear(x, p + ".ffn_up" + suffix + ".weight", expert, rounded)
    return linear(
        bf(F.silu(g) * u), p + ".ffn_down" + suffix + ".weight", expert, rounded
    )


def rope(x):
    half = x.shape[-1] // 2
    freq = 10000.0 ** (-torch.arange(half, device=device).float() / half)
    theta = torch.arange(len(x), device=device)[:, None, None] * freq
    a, b = x.chunk(2, -1)
    return bf(
        torch.cat(
            (a * theta.cos() - b * theta.sin(), a * theta.sin() + b * theta.cos()), -1
        )
    )


# Same token sequence as the Rust parity example, truncated for an independent check.
tokens = [150003, 100, 131, 162, 193, 224, 150004]
x = tensor("token_embd.weight")[tokens]
begin = time.monotonic()
for layer in range(config["flint.block_count"]):
    p = f"blk.{layer}"
    z = norm(x, p + ".attn_norm.weight")
    q = norm(
        linear(z, p + ".attn_q.weight").reshape(-1, 32, 128), p + ".attn_q_norm.weight"
    )
    k = norm(
        linear(z, p + ".attn_k.weight").reshape(-1, 4, 128), p + ".attn_k_norm.weight"
    )
    v = linear(z, p + ".attn_v.weight").reshape(-1, 4, 128)
    local = config["flint.attention.sliding_window_pattern"][layer]
    if config["flint.attention.rope_pattern"][layer]:
        q, k = rope(q), rope(k)
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(8, dim=1)) / 128**0.5
    pos = torch.arange(len(x), device=device)
    mask = pos[None, :] > pos[:, None]
    if local:
        mask |= pos[None, :] <= pos[:, None] - config["flint.attention.sliding_window"]
    scores = scores.masked_fill(mask, -torch.inf)
    scores = torch.cat(
        (scores, tensor(p + ".attn_sinks.weight")[:, None, None].expand(-1, len(x), 1)),
        dim=-1,
    )
    probs = scores.softmax(-1)[..., : len(x)]
    a = bf(torch.einsum("hts,shd->thd", probs, v.repeat_interleave(8, dim=1))).reshape(
        -1, 4096
    )
    x = bf(x + linear(a, p + ".attn_output.weight"))
    z = norm(x, p + ".ffn_norm.weight")
    if layer < config["flint.leading_dense_block_count"]:
        y = mlp(z, p)
    else:
        shared = mlp(z, p, "_shexp")
        scores = bf(F.linear(z, tensor(p + ".ffn_gate_inp.weight"))).sigmoid()
        ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :6]
        w = scores.gather(1, ids)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(z)
        for row in range(len(z)):
            for slot in range(6):
                expert = int(ids[row, slot])
                value = mlp(z[row : row + 1], p, "_exps", expert)
                out[row] += w[row, slot] * value[0]
        y = bf(bf(out) + shared)
    x = bf(x + y)
    if layer % 10 == 0:
        print("layer", layer, "seconds", round(time.monotonic() - begin, 2), flush=True)
logits = (
    linear(
        norm(x[-1:], "output_norm.weight"),
        "output.weight",
        round_weights=False,
        output_bf=False,
    )
    .flatten()
    .cpu()
    .tolist()
)
args.output.write_text(json.dumps({"tokens": tokens, "logits": logits}))
print("top", int(np.argmax(logits)), "seconds", time.monotonic() - begin)
