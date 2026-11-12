"""End-to-end speed: h3metal against another implementation on the same prompt, size and seed.

Every run is a fresh process (as a user would run it), timed by wall clock.
Memory is the peak physical footprint the kernel records for the process
(/usr/bin/time -l, "peak memory footprint"; for an implementation that runs
its stages in child processes, the largest child's maximum resident set is
recorded as well).
Runs alternate between the implementations, so thermal drift hits all of
them alike, and the machine cools down between runs.

    python bench/speed.py --sizes 1024,2048,4096 --runs 2 \
        --h3metal "h3metal generate --model DIR --min-area 1024" \
        --other "h3-image-mlx generate --models DIR" --out results.json
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

PROMPT = ("A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light.")


def run(cmd: list[str]) -> dict:
    t = time.time()
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True)
    wall = time.time() - t
    rss = foot = 0
    for line in p.stderr.splitlines():
        if "maximum resident set size" in line:
            rss = int(line.split()[0])
        if "peak memory footprint" in line:
            foot = int(line.split()[0])
    return {"wall": wall, "rss_gb": rss / 1e9, "footprint_gb": foot / 1e9, "ok": p.returncode == 0,
            "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:] if p.returncode else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--h3metal", required=True, help="command prefix; -p/-s/--seed/-o are appended")
    ap.add_argument("--other", help="command prefix for the other implementation (h3-image-mlx syntax)")
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    tmp = tempfile.mkdtemp()
    for size in [int(s) for s in a.sizes.split(",")]:
        for r in range(a.runs):
            jobs = [("h3metal", shlex.split(a.h3metal) + ["-p", a.prompt, "-s", f"{size}x{size}", "--seed", str(a.seed),
                                                         "-o", os.path.join(tmp, f"h3metal_{size}_{r}.png")])]
            if a.other:
                jobs.append(("other", shlex.split(a.other) + ["--prompt", a.prompt, "--width", str(size), "--height",
                                                             str(size), "--seed", str(a.seed), "--output",
                                                             os.path.join(tmp, f"other_{size}_{r}.png")]))
            if r % 2:
                jobs.reverse()
            for name, cmd in jobs:
                time.sleep(a.cooldown)
                res = run(cmd)
                res.update({"impl": name, "size": size, "run": r})
                results.append(res)
                print(f"{name:8s} {size}x{size} run {r}: {res['wall']:.1f} s, peak footprint "
                      f"{res['footprint_gb']:.2f} GB, max RSS {res['rss_gb']:.2f} GB"
                      f"{'' if res['ok'] else ' FAILED'}", flush=True)
                if not res["ok"]:
                    print(res["stderr"], file=sys.stderr)
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=1)
    print(f"images in {tmp}")


if __name__ == "__main__":
    main()
