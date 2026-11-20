"""Provenance of a result file: which code, which software and which configuration produced it.

Every script that writes results calls one of
  stamp(**fields)             -> dict, embedded in a JSON result (under "provenance")
  sidecar(out, **fields)      appends the same dict as one JSON line to <out>.prov.jsonl (for CSVs, which are often
                              written in append mode by several runs: one line per run, the CSV itself stays plain)
  comment(**fields)           -> short "commit=...; started=..." text
  csv_head(state, **fields)   -> the first '#' line of a freshly written timing CSV: power state + comment()
The fixed fields: commit (git HEAD of the code tree, None outside a git checkout), dirty (code files that differ
from that commit; results and paper files are not counted), started (start of this process, local time with UTC
offset), written, argv, cwd, python / mlx / mlx_lm / macos / chip. Callers add what only they know:
format (weight format name), pack (file of the packed weights), head (output projection), model, method.

Import this module at the top of a script, so that `started` is the start of the process and not of the write.
"""
import datetime
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = ["e2e", "sim", "kernels", "bench", "tools", "setup.sh"]
T0 = datetime.datetime.now().astimezone()
_HOME = os.path.expanduser("~")
_fixed = None


def _run(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.rstrip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001  (no git, no sysctl: the stamp must never stop a run)
        return None


def _version(dist):
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:  # noqa: BLE001
        return None


def tilde(v):
    """Paths under the home directory are written with ~ (result files carry no user name)."""
    if isinstance(v, str):
        return v.replace(_HOME, "~")
    if isinstance(v, (list, tuple)):
        return [tilde(x) for x in v]
    if isinstance(v, dict):
        return {k: tilde(x) for k, x in v.items()}
    return v if isinstance(v, (int, float, bool, type(None))) else str(v)


def fixed():
    """The fields that do not change within a process (read once, at the first stamp)."""
    global _fixed
    if _fixed is None:
        commit = _run("git", "-C", ROOT, "rev-parse", "HEAD")
        dirty = None
        if commit:
            st = _run("git", "-C", ROOT, "status", "--porcelain", "--untracked-files=all", "--", *CODE)
            dirty = [ln[3:] for ln in (st or "").splitlines()]
        _fixed = dict(commit=commit, dirty=dirty, started=T0.isoformat(timespec="seconds"),
                      argv=tilde([os.path.basename(sys.argv[0])] + sys.argv[1:]), cwd=tilde(os.getcwd()),
                      python=platform.python_version(), mlx=_version("mlx"), mlx_lm=_version("mlx-lm"),
                      macos=platform.mac_ver()[0], chip=_run("sysctl", "-n", "machdep.cpu.brand_string"))
    return _fixed


def stamp(**fields):
    d = dict(fixed())
    d["written"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    d.update(tilde(fields))
    return d


def sidecar(out, fresh=False, **fields):
    """Append the stamp of this run to <out>.prov.jsonl (fresh: the file <out> was written anew, so the sidecar is
    too); returns the stamp."""
    d = stamp(out=os.path.basename(out), **fields)
    with open(out + ".prov.jsonl", "w" if fresh else "a") as f:
        f.write(json.dumps(d) + "\n")
    return d


def comment(**fields):
    f = fixed()
    c = (f["commit"] or "no-git")[:12] + ("+dirty" if f["dirty"] else "")
    parts = [f"commit={c}", f"started={f['started']}"] + [f"{k}={v}" for k, v in fields.items() if v not in (None, "")]
    return "; ".join(parts)


def csv_head(state="", **fields):
    return "# " + "; ".join(x for x in (state, comment(**fields)) if x) + "\n"


def _cli(argv):
    """python provenance.py                                        print the stamp of this checkout
    python provenance.py step <file.jsonl> --queue Q --step S --event start|end [--kind K] [--exit N] [--seconds N]
                         -- command...                            append one line for a queue step (e2e/lib_queue.sh)"""
    if not argv:
        print(json.dumps(stamp(), indent=1))
        return
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["step"])
    ap.add_argument("file")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--event", required=True, choices=["start", "end"])
    ap.add_argument("--kind", default="")
    ap.add_argument("--exit", type=int, default=None)
    ap.add_argument("--seconds", type=int, default=None)
    cut = argv.index("--") if "--" in argv else len(argv)  # the step's command follows "--", untouched
    a, cmd = ap.parse_args(argv[:cut]), argv[cut + 1:]
    d = stamp(queue=a.queue, step=a.step, event=a.event, kind=a.kind or None, command=cmd)
    for k in ("argv", "started"):  # those of this helper, not of the step
        d.pop(k)
    if a.event == "end":
        d.update(exit=a.exit, seconds=a.seconds)
    with open(a.file, "a") as f:
        f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    _cli(sys.argv[1:])
