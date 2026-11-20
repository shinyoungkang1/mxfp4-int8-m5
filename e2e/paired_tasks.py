"""Paired comparison of two zero-shot runs from their per-example files (eval_tasks.py writes <out>.samples.json).

Both systems answered the same documents, so the difference between them is tested on the documents where they
disagree, which is far tighter than comparing two accuracies with their separate standard errors:
  per task    accuracy of A and B on the shared documents, B - A in points, the discordant counts (only A right /
              only B right), the exact McNemar p-value and a 95% interval of the difference (paired standard error);
  all tasks   the mean over tasks of B - A, with a paired bootstrap interval (documents resampled within each task,
              the same draw for both systems), with and without the tasks named by --without (default: lambada_openai).
One metric per task is used for both systems: acc_norm or acc, whichever has the higher mean over the two systems
(eval_tasks.py's summaries take the larger of the two for each system separately, which is not a paired quantity);
--metric acc or --metric acc_norm forces one (a task without it falls back to the other and says so).
The files must describe the same documents and prompts: doc_hash and prompt_hash are compared for every shared
doc_id and a mismatch stops the comparison.

Usage: python paired_tasks.py A.samples.json B.samples.json [--out diff.json] [--reps 10000] [--metric auto]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import provenance as prov  # noqa: E402

METRICS = ["acc_norm", "acc", "exact_match"]


def number(v):
    if isinstance(v, str):
        return {"True": 1.0, "False": 0.0}.get(v, None) if v in ("True", "False") else float(v)
    return float(v)


def read(path):
    with open(path) as f:
        d = json.load(f)
    if "samples" not in d or not d["samples"]:
        raise SystemExit(f"{path}: no per-example records (written by an eval_tasks.py without sample logging?)")
    return d


def mcnemar(b, c):
    """Exact two-sided McNemar p-value: b and c are the two discordant counts, Binomial(b + c, 1/2) under H0."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    return min(1.0, 2 * sum(math.exp(x) for x in logs))


def pair(task, ra, rb, metric):
    """-> (name of the metric used, d = B - A per shared document, A's and B's values) after the identity checks."""
    a, b = {r["doc_id"]: r for r in ra}, {r["doc_id"]: r for r in rb}
    if len(a) != len(ra) or len(b) != len(rb):
        raise SystemExit(f"{task}: a doc_id occurs twice")
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit(f"{task}: no document in common")
    for i in ids:
        for h in ("doc_hash", "prompt_hash"):
            if a[i].get(h) != b[i].get(h):
                raise SystemExit(f"{task} doc {i}: {h} differs between the two files (not the same document or prompt)")
    have = [m for m in METRICS if m in a[ids[0]] and m in b[ids[0]]]
    if not have:
        raise SystemExit(f"{task}: none of {METRICS} in both files")
    col = lambda rows, m: np.array([number(rows[i][m]) for i in ids])  # noqa: E731
    if metric == "auto":
        m = max(have, key=lambda m: col(a, m).mean() + col(b, m).mean())
    else:
        m = metric if metric in have else have[0]
    return m, col(b, m) - col(a, m), col(a, m), col(b, m), len(set(a) | set(b)) - len(ids)


def boot_means(d, reps, rng):
    """reps bootstrap means of d (documents resampled with replacement)."""
    n = len(d)
    if set(np.unique(d)) <= {-1.0, 0.0, 1.0}:  # the same distribution as index resampling, without the index matrix
        up, down = float(np.mean(d == 1)), float(np.mean(d == -1))
        cnt = rng.multinomial(n, [up, down, max(0.0, 1 - up - down)], size=reps)
        return (cnt[:, 0] - cnt[:, 1]) / n
    out = np.empty(reps)
    for r0 in range(0, reps, 256):
        idx = rng.integers(0, n, (min(256, reps - r0), n))
        out[r0:r0 + len(idx)] = d[idx].mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="auto", choices=["auto", "acc", "acc_norm"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--without", default="lambada_openai", help="tasks left out of the second mean, comma-separated")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    A, B = read(a.a), read(a.b)
    ca, cb = A.get("config", {}), B.get("config", {})
    name = lambda c: " ".join(str(c.get(k)) for k in ("model", "method", "format") if c.get(k))  # noqa: E731
    print(f"A: {name(ca)}   ({a.a})\nB: {name(cb)}   ({a.b})")
    for k in ("limit", "num_fewshot", "lm_eval"):
        if ca.get(k) != cb.get(k):
            print(f"note: {k} differs: A {ca.get(k)}, B {cb.get(k)}")
    tasks = [t for t in A["samples"] if t in B["samples"]]
    only = sorted(set(A["samples"]) ^ set(B["samples"]))
    if only:
        print("note: in one file only, not compared:", ", ".join(only))
    if not tasks:
        raise SystemExit("no task in common")
    rng = np.random.default_rng(0)
    rows, boots = [], {}
    print(f"{'task':16s} {'metric':8s} {'n':>6s} {'A':>7s} {'B':>7s} {'B-A':>7s} {'95% interval':>17s} {'only A':>7s} {'only B':>7s} {'McNemar p':>10s}")
    for t in tasks:
        m, d, va, vb, dropped = pair(t, A["samples"][t], B["samples"][t], a.metric)
        n = len(d)
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        only_a, only_b = int(np.sum(d < 0)), int(np.sum(d > 0))
        binary = set(np.unique(np.concatenate([va, vb]))) <= {0.0, 1.0}
        row = dict(task=t, metric=m, n=n, a=float(va.mean()), b=float(vb.mean()), diff=float(d.mean()), se=se,
                   lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se), only_a_right=only_a,
                   only_b_right=only_b, mcnemar_p=mcnemar(only_a, only_b) if binary else None, not_shared=dropped)
        rows.append(row)
        boots[t] = boot_means(d, a.reps, rng)
        p = f"{row['mcnemar_p']:10.4f}" if binary else f"{'-':>10s}"
        print(f"{t:16s} {m:8s} {n:6d} {100 * row['a']:7.2f} {100 * row['b']:7.2f} {100 * row['diff']:+7.2f} "
              f"[{100 * row['lo']:+6.2f}, {100 * row['hi']:+6.2f}] {only_a:7d} {only_b:7d} {p}"
              + (f"   ({dropped} documents in one file only)" if dropped else ""))
    means = []
    skip = [t for t in a.without.split(",") if t in tasks]
    for label, ts in [("all tasks", tasks)] + ([(f"without {', '.join(skip)}", [t for t in tasks if t not in skip])] if skip and len(skip) < len(tasks) else []):
        diff = float(np.mean([r["diff"] for r in rows if r["task"] in ts]))
        bm = np.mean([boots[t] for t in ts], axis=0)
        lo, hi = (float(x) for x in np.percentile(bm, [2.5, 97.5]))
        means.append(dict(tasks=ts, label=label, a=float(np.mean([r["a"] for r in rows if r["task"] in ts])),
                          b=float(np.mean([r["b"] for r in rows if r["task"] in ts])), diff=diff, lo=lo, hi=hi,
                          p_boot=float(min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0))))))
        print(f"mean of {len(ts)} task means ({label}): A {100 * means[-1]['a']:.2f}, B {100 * means[-1]['b']:.2f}, "
              f"B-A {100 * diff:+.2f} points, 95% paired bootstrap interval [{100 * lo:+.2f}, {100 * hi:+.2f}] ({a.reps} resamples)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"a": dict(file=os.path.basename(a.a), **{k: ca.get(k) for k in ("model", "method", "format", "limit")}),
                       "b": dict(file=os.path.basename(a.b), **{k: cb.get(k) for k in ("model", "method", "format", "limit")}),
                       "tasks": rows, "means": means, "reps": a.reps, "provenance": prov.stamp()}, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
