"""Layer-at-a-time access to the official Qwen3.8-Flash-Next FP8 checkpoint (Qwen/Qwen3.8-Flash-Next-FP8), which does
not fit in memory as BF16 (~250 GB). Run with the mlx-lm PR #1788 environment (.venv-q38).

Checkpoint layout (config quantization_config): the 512 routed experts of every layer are FP8 E4M3 with one FP32
scale per 128 x 128 tile (`weight_scale_inv`, multiplied in), stored one expert per tensor
(`mlp.experts.<e>.{gate,up,down}_proj.weight`); the n-gram table is FP8 (128 shards + `weight_scale`); everything
else (attention / DeltaNet, shared expert, router, hyper-connections, PLE projections, embeddings, lm_head) is BF16.

Tensors are read straight from the safetensors files through numpy memmaps (no copy of the 173 GB), FP8 is decoded
with an exact 256-entry table, and layer i is returned in the module naming of the PR's DecoderLayer, with the PR's
`sanitize` conversions (norm weights stored as w - 1 get +1, conv1d (C, 1, K) -> (C, K, 1)) and the experts stacked
as SwitchGLU weights [E, out, in].
"""
import glob
import json
import os
import re
import struct
import subprocess

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen4_exp

REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
PREFIX = "model.language_model."
FOLD_ONE = qwen4_exp.Model._FOLD_ONE
_NP = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F32": np.float32, "F16": np.float16, "I64": np.int64,
       "I32": np.int32, "U8": np.uint8}


def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


class MemoryGuard:
    """Stops the job (cleanly, with a message) if swap use grows by more than `limit_gb` since the start: running out
    of memory on a 128 GB Mac makes the kernel watchdog restart the machine.
    Also caps MLX's buffer cache so that freed buffers are returned instead of hoarded."""

    def __init__(self, limit_gb=12.0, cache_gb=4.0):
        self.start, self.limit = swap_used_gb(), limit_gb
        mx.set_cache_limit(int(cache_gb * 2 ** 30))

    def check(self, where=""):
        grown = swap_used_gb() - self.start
        if grown > self.limit:
            raise SystemExit(f"MemoryGuard: swap grew by {grown:.1f} GB ({where}); stopping to protect the machine")


def snapshot():
    return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{REPO.replace('/', '--')}/snapshots/*"))[0]


def e4m3_table():
    """FP8 E4M3 (fn: no infinities, S.1111.111 = NaN) code -> value, exactly."""
    c = np.arange(256)
    s, e, m = c >> 7, (c >> 3) & 15, c & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(s == 1, -v, v).astype(np.float32)


_LUT = None


def fp8_decode(codes, scale=None, block=128):
    """uint8 codes [..., N, K] (numpy or mx) x per-tile scale [..., ceil(N/128), ceil(K/128)] -> float32 mx."""
    global _LUT
    if _LUT is None:
        _LUT = mx.array(e4m3_table())
    w = _LUT[mx.array(codes).astype(mx.uint32)]
    if scale is not None:
        s = mx.array(scale).astype(mx.float32)
        n, k = w.shape[-2:]
        s = mx.repeat(mx.repeat(s, block, axis=-2), block, axis=-1)[..., :n, :k]
        w = w * s
    return w


def to_f32(a, kind):
    """numpy array of a stored float dtype -> float32 numpy (BF16 bit patterns widened exactly)."""
    if kind == "BF16":
        return (np.asarray(a, np.uint32) << 16).view(np.float32)
    return np.asarray(a, np.float32)


class Checkpoint:
    def __init__(self, path=None):
        self.path = path or snapshot()
        self.config = json.load(open(os.path.join(self.path, "config.json")))
        self.map = json.load(open(os.path.join(self.path, "model.safetensors.index.json")))["weight_map"]
        self._hdr = {}

    def _header(self, fn):
        if fn not in self._hdr:
            with open(os.path.join(self.path, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._hdr[fn] = (json.loads(f.read(n)), 8 + n)
        return self._hdr[fn]

    def raw(self, name):
        """-> (numpy memmap of the stored dtype, safetensors dtype string)."""
        fn = self.map[name]
        h, base = self._header(fn)
        e = h[name]
        s, t = e["data_offsets"]
        shape = tuple(e["shape"])
        if t == s:
            return np.zeros(shape, _NP[e["dtype"]]), e["dtype"]
        return np.memmap(os.path.join(self.path, fn), dtype=_NP[e["dtype"]], mode="r", offset=base + s,
                         shape=shape), e["dtype"]

    def get(self, name, dtype=mx.bfloat16):
        """Any non-FP8 tensor as an mx array (BF16 kept bit-exact)."""
        a, kind = self.raw(name)
        if kind == "BF16":
            return mx.array(np.ascontiguousarray(a)).view(mx.bfloat16).astype(dtype)
        assert kind != "F8_E4M3", name
        return mx.array(np.ascontiguousarray(a))

    def keys(self, prefix):
        return sorted(k for k in self.map if k.startswith(prefix))

    def text_args(self, **override):
        tc = dict(self.config["text_config"])
        tc.update(override)
        return qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=tc).text

    def experts(self, i, proj, n_experts, dtype=mx.bfloat16):
        """Layer i's routed experts' `proj` weights, FP8-decoded and stacked: [E, out, in]."""
        p = f"{PREFIX}layers.{i}.mlp.experts."
        codes = np.stack([self.raw(f"{p}{e}.{proj}.weight")[0] for e in range(n_experts)])
        scale = np.stack([to_f32(*self.raw(f"{p}{e}.{proj}.weight_scale_inv")) for e in range(n_experts)])
        return fp8_decode(codes, scale).astype(dtype)

    def layer_weights(self, i, n_experts, skip=()):
        """Layer i's weights in DecoderLayer naming (no n-gram table: see NgramTable). Returns {name: mx array}.
        skip: module paths not to load (e.g. the matrices that are packed), "mlp.switch_mlp" for the experts."""
        p = f"{PREFIX}layers.{i}."
        out = {}
        for k in self.keys(p):
            rel = k[len(p):]
            if rel.startswith("mlp.experts.") or ".ngram_embedding." in rel:
                continue
            if any(rel.startswith(s + ".") for s in skip):
                continue
            if rel.startswith("ple.ple_embedding."):  # hash constants: rebuilt from the config, checked by NgramTable
                continue
            v = self.get(k)
            if rel.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
                v = v.transpose(0, 2, 1)
            if rel.endswith(FOLD_ONE):
                v = (1.0 + v.astype(mx.float32)).astype(v.dtype)
            out[rel] = v
        if "mlp.switch_mlp" not in skip:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                out[f"mlp.switch_mlp.{proj}.weight"] = self.experts(i, proj, n_experts)
        return out

    def top_weights(self):
        """Embeddings, final hyper-connection mixer (carries the final norm) and lm_head."""
        out = {"embed_tokens.weight": self.get(f"{PREFIX}embed_tokens.weight"), "lm_head.weight": self.get("lm_head.weight")}
        for k in self.keys(f"{PREFIX}hyper_connection_mixer."):
            v = self.get(k)
            if k.endswith(FOLD_ONE):
                v = (1.0 + v.astype(mx.float32)).astype(v.dtype)
            out["hyper_connection_mixer." + k.split("hyper_connection_mixer.")[1]] = v
        return out


class NgramTable:
    """Replacement for the PR's _ShardedEmbedding: gathers only the addressed rows of the FP8 n-gram table from the
    memory-mapped shards and decodes them (the table is 51B parameters; it is never loaded as a whole)."""

    def __init__(self, ck, layer, emb, memo=False):
        """memo: remember results by the bytes of the index array (several evaluation methods look up the same
        windows: the rows depend only on the token ids)."""
        self.memo = {} if memo else None
        p = f"{PREFIX}layers.{layer}.ple.ple_embedding."
        self.rows, self.dim, n = emb.ngram_embedding.rows, emb.ngram_embedding.dim, emb.ngram_embedding.n_shards
        self.shards = [ck.raw(f"{p}ngram_embedding.shard_{s}.weight") for s in range(n)]
        for s, (a, kind) in enumerate(self.shards):
            assert a.shape == (self.rows, self.dim) and kind in ("F8_E4M3", "BF16"), (s, a.shape, kind)
        sc = to_f32(*ck.raw(f"{p}ngram_embedding.weight_scale"))
        assert sc.size == 1, f"expected one per-tensor n-gram scale, got {sc.shape}"
        self.scale = float(sc.reshape(-1)[0])
        for name, mine in (("layer_multipliers", emb._mults), ("ngram_heads_vocab_sizes", emb._sizes),
                           ("ngram_heads_offsets", emb._offsets)):
            theirs = np.asarray(ck.raw(p + name)[0]).astype(np.int64)
            assert np.array_equal(theirs, np.array(mine)), f"n-gram hash constant {name} differs from the config's"

    def __call__(self, gid):
        if self.memo is not None:
            import hashlib
            key = hashlib.blake2b(np.array(gid).tobytes(), digest_size=16).hexdigest() + str(tuple(gid.shape))
            if key not in self.memo:
                self.memo[key] = self._lookup(gid)
                mx.eval(self.memo[key])
            return self.memo[key]
        return self._lookup(gid)

    def _lookup(self, gid):
        flat = np.array(gid).reshape(-1)
        shard, row = flat // self.rows, flat % self.rows
        codes = np.empty((flat.size, self.dim), np.uint8 if self.shards[0][1] == "F8_E4M3" else np.uint16)
        for s in np.unique(shard).tolist():
            sel = np.nonzero(shard == s)[0]
            codes[sel] = self.shards[s][0][row[sel]]
        if self.shards[0][1] == "BF16":
            v = mx.array(codes).view(mx.bfloat16).astype(mx.float32)
        else:
            v = fp8_decode(codes) * self.scale
        return v.reshape(*gid.shape, self.dim)
