"""End-to-end speed: h3metal against another implementation on the same prompt, size and seed.

Every run is a fresh process (as a user would run it), timed by wall clock.
Memory is the peak physical footprint the kernel records per process
(proc_pid_rusage, the figure Activity Monitor shows as "Memory"), sampled
over the whole process tree: an implementation that runs its stages in
child processes is charged for its largest one, and the sum over the tree
is recorded too.

Runs alternate between the implementations, so thermal drift hits all of
them alike, and the machine cools down between runs. Both implementations
cache prompt embeddings; --cold gives every run a prompt of its own, so the
text encoder always runs.

    python bench/speed.py --sizes 1024,2048,4096 --runs 2 \\
        --h3metal "h3metal generate --model DIR --min-area 1024" \\
        --other "h3-image-mlx generate --models DIR" --out results.json
"""

import argparse
import ctypes
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time

PROMPT = ("A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light.")

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
RUSAGE_INFO_V4 = 4


class _RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
        "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance",
        "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
        "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time", "logical_writes",
        "lifetime_max_phys_footprint", "instructions", "cycles", "billed_energy", "serviced_energy",
        "interval_max_phys_footprint", "runnable_time", "flags")]


def _peak_footprint(pid: int) -> int:
    info = _RusageV4()
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return 0
    return int(info.lifetime_max_phys_footprint)


def _tree(root: int) -> list[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = {}
    for line in out.split("\n"):
        if line.strip():
            p, pp = (int(v) for v in line.split())
            kids.setdefault(pp, []).append(p)
    todo, seen = [root], []
    while todo:
        p = todo.pop()
        seen.append(p)
        todo += kids.get(p, [])
    return seen


def run(cmd: list[str]) -> dict:
    peaks: dict[int, int] = {}
    tree_peak = [0]
    t = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            live = 0
            for pid in _tree(proc.pid):
                f = _peak_footprint(pid)
                if f:
                    peaks[pid] = max(peaks.get(pid, 0), f)
                    live += f
            tree_peak[0] = max(tree_peak[0], live)
            stop.wait(0.2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out, err = proc.communicate()
    wall = time.time() - t
    stop.set()
    th.join()
    return {"wall": wall, "footprint_gb": max(peaks.values(), default=0) / 1e9, "tree_gb": tree_peak[0] / 1e9,
            "ok": proc.returncode == 0, "stdout": out[-2000:], "stderr": err[-2000:] if proc.returncode else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--cold", action="store_true", help="a new prompt every run: the text encoder always runs")
    ap.add_argument("--h3metal", required=True, help="command prefix; -p/-s/--seed/-o are appended")
    ap.add_argument("--other", help="command prefix for the other implementation (h3-image-mlx syntax)")
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    tmp = tempfile.mkdtemp()
    n = 0
    for size in [int(s) for s in a.sizes.split(",")]:
        for r in range(a.runs):
            n += 1
            prompt = f"{a.prompt} Variation {time.time_ns() % 10**9}." if a.cold else a.prompt
            jobs = [("h3metal", shlex.split(a.h3metal) + ["-p", prompt, "-s", f"{size}x{size}", "--seed", str(a.seed),
                                                         "-o", os.path.join(tmp, f"h3metal_{size}_{r}.png")])]
            if a.other:
                jobs.append(("other", shlex.split(a.other) + ["--prompt", prompt, "--width", str(size), "--height",
                                                             str(size), "--seed", str(a.seed), "--output",
                                                             os.path.join(tmp, f"other_{size}_{r}.png")]))
            if n % 2 == 0:
                jobs.reverse()
            for name, cmd in jobs:
                time.sleep(a.cooldown)
                res = run(cmd)
                res.update({"impl": name, "size": size, "run": r, "cold": a.cold})
                results.append(res)
                print(f"{name:8s} {size}x{size} run {r}: {res['wall']:.1f} s, peak footprint "
                      f"{res['footprint_gb']:.2f} GB (process tree {res['tree_gb']:.2f} GB)"
                      f"{'' if res['ok'] else ' FAILED'}", flush=True)
                if not res["ok"]:
                    print(res["stderr"], file=sys.stderr)
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=1)
    print(f"images in {tmp}")


if __name__ == "__main__":
    main()
