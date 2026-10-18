"""In-situ GEMV tile tuning for decode: tokens/s of the W4A8 GEMV decode (separate quantization kernels) with
several per-shape tile assignments, interleaved with MLX MXFP4 (ctx 512, 128 generated tokens, median of reps).
Usage: python tune_decode.py qwen3|olmoe   (run through run_timed.sh)
       python tune_decode.py qwen35 [config,config,...]   (Qwen3.5-35B-A3B; configs from CONFIGS_QWEN35)
For Qwen3.5 only the MoE half is mx.compile'd (traced with its tile set); the attention half reads GEMV_TILES at
every call, so each configuration's tile set is installed before each of its runs."""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_decode as bd  # noqa: E402
import sc_decode as G  # noqa: E402

CONFIGS = {
    "qwen3": {
        "default": {},
        "qkv_4x16x4": {(4096, 2048): (4, 16, 4)},
        "qkv_8x16x2": {(4096, 2048): (8, 16, 2)},
        "qkv_4x16x4_down_8x16x1": {(4096, 2048): (4, 16, 4), (2048, 6144): (8, 16, 1)},
        "all_4x16x4": {(4096, 2048): (4, 16, 4), (2048, 2048): (4, 16, 4), (12288, 2048): (4, 16, 4),
                       (2048, 6144): (4, 16, 4)},
        "all_4x32x2": {(4096, 2048): (4, 32, 2), (2048, 2048): (4, 32, 2), (12288, 2048): (4, 32, 2),
                       (2048, 6144): (4, 32, 2)},
    },
    "olmoe": {
        "default": {},
        "qkv_8x16x1": {(6144, 2048): (8, 16, 1)},
        "experts_4x32x2": {(2048, 2048): (4, 32, 2), (2048, 1024): (4, 32, 2)},
        "qkv_8x16x1_experts_4x32x2": {(6144, 2048): (8, 16, 1), (2048, 2048): (4, 32, 2), (2048, 1024): (4, 32, 2)},
        "all_4x16x4": {(6144, 2048): (4, 16, 4), (2048, 2048): (4, 16, 4), (2048, 1024): (4, 16, 4)},
    },
}


# Qwen3.5-35B-A3B shapes (N, K): qkvz (12288, 2048), out / o (2048, 4096), full-attention qkv (9216, 2048),
# experts gate|up (1024, 2048, 9 slots, EPI 3), experts down (2048, 512, 9 slots)
GU, DN, QKVZ, OUT, QKV = (1024, 2048), (2048, 512), (12288, 2048), (2048, 4096), (9216, 2048)
CONFIGS_QWEN35 = {
    "default": {},
    "gu_4x16x4": {GU: (4, 16, 4)}, "gu_8x16x2": {GU: (8, 16, 2)}, "gu_4x32x2": {GU: (4, 32, 2)},
    "gu_4x8x4": {GU: (4, 8, 4)}, "gu_4x16x1": {GU: (4, 16, 1)}, "gu_8x8x2": {GU: (8, 8, 2)},
    "dn_4x16x1": {DN: (4, 16, 1)}, "dn_8x16x1": {DN: (8, 16, 1)}, "dn_2x16x4": {DN: (2, 16, 4)},
    "dn_4x8x2": {DN: (4, 8, 2)}, "dn_4x8x4": {DN: (4, 8, 4)}, "dn_2x8x4": {DN: (2, 8, 4)},
    "dn_1x16x8": {DN: (1, 16, 8)},
    "qkvz_4x16x4": {QKVZ: (4, 16, 4)}, "qkvz_8x16x2": {QKVZ: (8, 16, 2)}, "qkvz_4x32x2": {QKVZ: (4, 32, 2)},
    "out_4x16x4": {OUT: (4, 16, 4)}, "out_4x32x2": {OUT: (4, 32, 2)}, "out_8x16x1": {OUT: (8, 16, 1)},
    "out_2x32x4": {OUT: (2, 32, 4)},
    "qkv_4x16x4": {QKV: (4, 16, 4)}, "qkv_8x16x2": {QKV: (8, 16, 2)},
}


def main_qwen35(names):
    import decode as dec
    import fused_qwen35 as fz
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    cache_dir = os.path.expanduser("~/.cache/mxfp4-int8-m5")
    mxm = load(os.path.join(cache_dir, fz.REPO.split("/")[-1] + "-mxfp4"))[0]
    f, _ = fz.load_fused_qwen35()
    f.two_pass = "auto"
    prompt = mx.array(np.random.default_rng(0).integers(0, 248320, (1, 512)).astype(np.int32))

    class Ours(bd.OursMethod):
        def cache(self):
            return self.f.make_cache()

    class MLX(bd.MLXMethod):
        def cache(self):
            return make_prompt_cache(self.model)

    cfgs = {n: CONFIGS_QWEN35[n] for n in names} if names else dict(CONFIGS_QWEN35)
    if isinstance(names, dict):
        cfgs = names
    meths, tiles = {"mlx_mxfp4": MLX(mxm)}, {"mlx_mxfp4": {}}
    for name, c in cfgs.items():
        G.GEMV_TILES.clear()
        G.GEMV_TILES.update(c)
        meths[name], tiles[name] = Ours(f, dec.DecodeQwen35(f, head=mxm)), dict(c)
        bd.generate(meths[name], prompt, 8)
    times = {k: [] for k in meths}
    keys = list(meths)
    for r in range(5):
        for k in keys[r % len(keys):] + keys[:r % len(keys)]:
            G.GEMV_TILES.clear()
            G.GEMV_TILES.update(tiles[k])
            times[k].append(bd.generate(meths[k], prompt, 128))
    G.GEMV_TILES.clear()
    ref = 128 / np.median(times["mlx_mxfp4"])
    out = {}
    for k, v in times.items():
        tps = 128 / np.median(v)
        out[k] = tps
        print(f"qwen35 {k:28s} {tps:7.1f} tok/s  {tps / ref:.3f}x MLX MXFP4  "
              f"(range {128 / max(v):.1f}-{128 / min(v):.1f})", flush=True)
    return out


def main(model):
    import decode as dec
    ms = bd.build(model, ["mlx_mxfp4", "ours_sepquant"], os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    vocab = 151936 if model == "qwen3" else 50304
    prompt = mx.array(np.random.default_rng(0).integers(0, vocab, (1, 512)).astype(np.int32))
    f, head = ms["ours_sepquant"].f, ms["mlx_mxfp4"].model
    Dec = dec.DecodeQwen3 if model == "qwen3" else dec.DecodeOlmoe
    meths = {"mlx_mxfp4": ms["mlx_mxfp4"]}
    cfgs = dict(CONFIGS[model]) if len(sys.argv) < 3 else {"default": {}}
    for name, c in cfgs.items():
        # the decode halves are mx.compile'd: a fresh decoder per tile set, traced (warm-up) while it is active
        G.GEMV_TILES.clear()
        G.GEMV_TILES.update(c)
        meths[name] = bd.OursMethod(f, Dec(f, head=head, fuse_prologue=False))
        bd.generate(meths[name], prompt, 8)
    if model == "qwen3" and len(sys.argv) >= 3:  # prologue variants (python tune_decode.py qwen3 prologue)
        for name, fp in (("act_in_gemv_prologue", "act"), ("all_in_gemv_prologue", True)):
            meths[name] = bd.OursMethod(f, Dec(f, head=head, fuse_prologue=fp))
            bd.generate(meths[name], prompt, 8)
    G.GEMV_TILES.clear()
    bd.generate(meths["mlx_mxfp4"], prompt, 8)
    times = {k: [] for k in meths}
    for r in range(5):
        for k, m in meths.items():
            times[k].append(bd.generate(m, prompt, 128))
    ref = 128 / np.median(times["mlx_mxfp4"])
    for k, v in times.items():
        tps = 128 / np.median(v)
        print(f"{model} {k:28s} {tps:7.1f} tok/s  {tps / ref:.3f}x MLX MXFP4", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "qwen35":
        main_qwen35(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    else:
        main(sys.argv[1])
