"""Where the time goes on Qwen3-1.7B: a per-step map of prefill and batch-1 decode, for a packed format with and without rotated inputs.

Each kernel call (or group of MLX calls) is evaluated on its own and timed with the wall clock, so every step of _hidden_gemm_epi and of
DecodeQwen3 gets a time, summed over the layers. Those barriers add cost, so the map also reports the uninstrumented total of the same
configuration, timed in the same session, and the time of an empty evaluation ("sync_per_step"). Shares are comparable within a run; only the
uninstrumented totals are speeds. `read_mb` is the data one call of the step must read from weights and inputs, per forward pass, from the
array shapes and not from hardware counters.

Usage: python time_map.py [--lengths 512,2048,8192] [--reps 7] [--ctx 512,2048] [--configs fmt:e2m1/B8/M1..10,rot:e2m1/B8/M1..10] [--out results/27_timemap.csv]
Run under run_timed.sh: the GPU must otherwise be idle.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode as dec  # noqa: E402
import fused_qwen3 as fq  # noqa: E402
import sc_decode as G  # noqa: E402
import sc_kernels as S  # noqa: E402

REPO = "Qwen/Qwen3-1.7B-Base"
CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")


def mb(*arrays):
    return sum(a.nbytes for a in arrays) / 2 ** 20


class Timer:
    def __init__(self):
        self.ms, self.read = defaultdict(float), defaultdict(float)
        self.order = []

    def __call__(self, name, fn, reads=()):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        self.ms[name] += (time.perf_counter() - t0) * 1e3
        self.read[name] += mb(*reads)
        if name not in self.order:
            self.order.append(name)
        return out


def quant_rms(f, T, group, h, nw, Mp=None):
    if f.rot is not True:
        return T(f"{group}.rms_quant", lambda: S.rmsnorm_quant(h, nw, f.eps, Mp), (h, nw))
    xn = T(f"{group}.rms_norm", lambda: mx.fast.rms_norm(h, nw, f.args.rms_norm_eps), (h, nw))
    return T(f"{group}.rot_quant", lambda: S.rot_quant(xn, f.signs(xn.shape[1]), Mp), (xn,))


def quant_act(f, T, group, x, Mp=None, nt=256):
    if not (f.rot is True or (f.rot == "down" and group == "down")):
        return T(f"{group}.act_quant", lambda: S.act_quant(x, Mp, nt=nt), (x,))
    return T(f"{group}.rot_quant", lambda: S.rot_quant(x, f.signs(x.shape[1]), Mp), (x,))


def attention(f, T, at, c, q, k, v, L, mask):
    def rope():
        qq, kk, vv = q[None], k[None], v[None]
        qq = at.rope(qq, offset=c.offset)
        kk = at.rope(kk, offset=c.offset)
        kk, vv = c.update_and_fetch(kk, vv)
        return qq, kk, vv
    qq, kk, vv = T("attn.rope_cache", rope, (q, k, v))
    o = T("attn.sdpa", lambda: mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=at.scale, mask=mask), (qq, kk, vv))
    return T("attn.reshape", lambda: o.transpose(0, 2, 1, 3).reshape(L, -1), (o,))


def prefill_map(f, ids, T):
    S.set_format(f.fmt)
    a = f.args
    hq, hk, d = a.num_attention_heads, a.num_key_value_heads, a.head_dim
    cache = f.make_cache()
    h = T("embed", lambda: f.model.model.embed_tokens(ids)[0])
    L = h.shape[0]
    Mp = S.pad_rows(L)
    if Mp != L:
        h = T("embed", lambda: mx.pad(h, [(0, Mp - L), (0, 0)]))
    mask = "causal" if L > 1 else None
    tile = fq.tiles_for(Mp)
    xq, xs = quant_rms(f, T, "qkv", h, f.layers[0][0].input_layernorm.weight, Mp)
    nl = len(f.layers)
    for i, (layer, ws) in enumerate(f.layers):
        at = layer.self_attn

        def run(group, xq, xs, epi, res=None, prev=()):
            w = ws[group]
            wq = T(f"{group}.unpack", lambda: S.unpack_i8(w.codes, w.bcodes, w.N, w.K), (w.codes, w.bcodes))
            return T(f"{group}.gemm_epi", lambda: S.mm_epi(xq, wq, xs, w.wscale, epi, h.dtype, res=res, tile=tile[group]),
                     (xq, wq) + tuple(prev))

        y = run("qkv", xq, xs, 1)
        q, k, v = T("qkv.headnorm", lambda: S.qkv_headnorm_y(y, at.q_norm.weight, at.k_norm.weight, f.eps, L, hq, hk, d), (y,))
        o = attention(f, T, at, cache[i], q, k, v, L, mask)
        xq, xs = quant_act(f, T, "o", o, Mp)
        h = run("o", xq, xs, 2, res=h, prev=(h,))
        xq, xs = quant_rms(f, T, "gu", h, layer.post_attention_layernorm.weight)
        act = run("gu", xq, xs, 3)
        xq, xs = quant_act(f, T, "down", act, nt=512)
        h = run("down", xq, xs, 2, res=h, prev=(h,))
        if i + 1 < nl:
            xq, xs = quant_rms(f, T, "qkv", h, f.layers[i + 1][0].input_layernorm.weight)
    n = T("final_norm", lambda: f.model.model.norm(h[:L][None]), (h,))
    T("output_head", lambda: f.logits(n[:, -1:]), (n,))


def decode_map(f, d, tok, cache, T):
    S.set_format(f.fmt)
    a = f.args
    hq, hk, dd = a.num_attention_heads, a.num_key_value_heads, a.head_dim
    h = T("embed", lambda: f.model.model.embed_tokens(tok)[0])
    for i, (layer, ws) in enumerate(f.layers):
        at = layer.self_attn

        def gemv(group, xq, xs, n, epi, res=None):
            w = ws[group]
            return T(f"{group}.gemv", lambda: G.gemv(xq, w.codes, w.bcodes, xs, w.wscale, n, epi, **({"res": res} if res is not None else {})),
                     (xq, w.codes, w.bcodes))

        xq, xs = quant_rms(f, T, "qkv", h, layer.input_layernorm.weight, 1)
        y = gemv("qkv", xq, xs, (hq + 2 * hk) * dd, 1)
        q, k, v = T("qkv.headnorm", lambda: S.qkv_headnorm_y(y[None], at.q_norm.weight, at.k_norm.weight, f.eps, 1, hq, hk, dd), (y,))
        o = attention(f, T, at, cache[i], q, k, v, 1, None)
        xq, xs = quant_act(f, T, "o", o, 1)
        h = gemv("o", xq, xs, ws["o"].wscale.size, 2, res=h)[None]
        xq, xs = quant_rms(f, T, "gu", h, layer.post_attention_layernorm.weight, 1)
        act = gemv("gu", xq, xs, ws["gu"].wscale.size, 3)
        xq, xs = quant_act(f, T, "down", act[None], 1)
        h = gemv("down", xq, xs, ws["down"].wscale.size, 2, res=h)[None]
    n = T("final_norm", lambda: f.model.model.norm(h[None]), (h,))
    T("output_head", lambda: d.head(n), (n,))


def sync_ms(n=2000):
    x = mx.zeros((1,))
    mx.eval(x)
    t0 = time.perf_counter()
    for _ in range(n):
        x = x + 1
        mx.eval(x)
    return (time.perf_counter() - t0) * 1e3 / n


def real_prefill(f, ids):
    t0 = time.perf_counter()
    cache = f.make_cache()
    mx.eval(f.logits(f.hidden(ids, cache)[:, -1:]))
    return (time.perf_counter() - t0) * 1e3


def main():
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--configs", default="fmt:e2m1/B8/M1..10,rot:e2m1/B8/M1..10")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    fs = {c: fq.load_fused(REPO, CACHE, packed=c)[0] for c in a.configs.split(",")}
    mxm = load(os.path.join(CACHE, REPO.split("/")[-1] + "-mxfp4"))[0]
    rng = np.random.default_rng(0)
    rows = []
    sync = sync_ms()
    print(f"sync_per_step {sync:.3f} ms", flush=True)

    def add(cfg, what, L, name, ms, read, n):
        rows.append(dict(config=cfg, what=what, tokens=L, step=name, ms=f"{ms:.3f}", read_mb=f"{read:.2f}", passes=n, sync_per_step_ms=f"{sync:.3f}"))

    for L in [int(x) for x in a.lengths.split(",")]:
        ids = mx.array(rng.integers(0, 151936, (1, L)).astype(np.int32))
        for _ in range(2):  # warm-up
            for f in fs.values():
                real_prefill(f, ids)
        per = {c: [] for c in fs}
        reals = {c: [] for c in fs}
        mlx_t = []
        for _ in range(a.reps):
            for c, f in fs.items():
                T = Timer()
                prefill_map(f, ids, T)
                per[c].append(T)
                reals[c].append(real_prefill(f, ids))
            t0 = time.perf_counter()
            from mlx_lm.models.cache import make_prompt_cache
            mx.eval(mxm(ids, cache=make_prompt_cache(mxm))[:, -1:])
            mlx_t.append((time.perf_counter() - t0) * 1e3)
        add("mlx_mxfp4", "prefill", L, "UNINSTRUMENTED_TOTAL", float(np.median(mlx_t)), 0, a.reps)
        for c in fs:
            add(c, "prefill", L, "UNINSTRUMENTED_TOTAL", float(np.median(reals[c])), 0, a.reps)
            tot = float(np.median([sum(t.ms.values()) for t in per[c]]))
            add(c, "prefill", L, "INSTRUMENTED_TOTAL", tot, 0, a.reps)
            for name in per[c][0].order:
                add(c, "prefill", L, name, float(np.median([t.ms[name] for t in per[c]])), per[c][0].read[name], a.reps)
            print(f"prefill {L} {c}: real {np.median(reals[c]):.1f} ms, instrumented {tot:.1f} ms; MLX {np.median(mlx_t):.1f} ms", flush=True)
    # decode: teacher-free greedy is not needed, the token only selects an embedding
    for ctx in [int(x) for x in a.ctx.split(",")]:
        prompt = mx.array(rng.integers(0, 151936, (1, ctx)).astype(np.int32))
        for c, f in fs.items():
            S.set_format(f.fmt)
            d = dec.DecodeQwen3(f, head=mxm)
            tok = mx.array(rng.integers(0, 151936, (1, 1)).astype(np.int32))
            # uninstrumented step, then instrumented, each on its own cache
            cache = f.make_cache()
            mx.eval(f.hidden(prompt, cache))
            real = []
            for _ in range(a.steps):
                t0 = time.perf_counter()
                mx.eval(d.step(tok, cache))
                real.append((time.perf_counter() - t0) * 1e3)
            cache = f.make_cache()
            mx.eval(f.hidden(prompt, cache))
            ts = []
            for _ in range(a.steps):
                T = Timer()
                decode_map(f, d, tok, cache, T)
                ts.append(T)
            skip = a.steps // 4
            add(c, "decode", ctx, "UNINSTRUMENTED_TOTAL", float(np.median(real[skip:])), 0, a.steps - skip)
            tot = float(np.median([sum(t.ms.values()) for t in ts[skip:]]))
            add(c, "decode", ctx, "INSTRUMENTED_TOTAL", tot, 0, a.steps - skip)
            for name in ts[0].order:
                add(c, "decode", ctx, name, float(np.median([t.ms[name] for t in ts[skip:]])), ts[0].read[name], a.steps - skip)
            print(f"decode ctx={ctx} {c}: real {np.median(real[skip:]):.2f} ms, instrumented {tot:.2f} ms", flush=True)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.out)
    else:
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
