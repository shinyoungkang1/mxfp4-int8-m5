"""Packs Qwen3.8-Flash-Next layers (fused_qwen38.pack_qwen38) as soon as all of a layer's checkpoint files have been
downloaded (the download and the packing overlap). Usage (.venv-q38): python q38_pack_when_ready.py [format]"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fused_qwen38 as FQ  # noqa: E402
import q38_ckpt as C  # noqa: E402

if __name__ == "__main__":
    fmt = sys.argv[1] if len(sys.argv) > 1 else FQ.SC
    ck = C.Checkpoint()
    n = ck.text_args().num_hidden_layers
    out = FQ.pack_dir(fmt, os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    todo = [i for i in range(n) if not os.path.exists(os.path.join(out, f"layer_{i:02d}.safetensors"))]
    while todo:
        have = {f for f in os.listdir(ck.path) if f.endswith(".safetensors")}
        ready = [i for i in todo if {ck.map[k] for k in ck.keys(f"{C.PREFIX}layers.{i}.")} <= have]
        if ready:
            FQ.pack_qwen38(fmt, layers=ready)
            todo = [i for i in todo if i not in ready]
        else:
            time.sleep(30)
    print("all layers packed", flush=True)
