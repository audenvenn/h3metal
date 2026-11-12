"""Markdown tables from bench/speed.py results: median wall time and peak footprint per implementation and size.

    python bench/speed_table.py results.json [more.json ...] [--other-name h3-image-mlx]
"""

import argparse
import json
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--other-name", default="h3-image-mlx")
    a = ap.parse_args()
    runs = []
    for path in a.results:
        runs += json.load(open(path))
    groups = defaultdict(list)
    for r in runs:
        if r.get("ok", True):
            groups[(r.get("cold", False), r["size"], r["impl"])].append(r)
    for cold in sorted({k[0] for k in groups}):
        print(f"\n{'Every run a new prompt (text encoder included)' if cold else 'Repeated prompt (embeddings cached)'}\n")
        print(f"| Canvas | h3metal | {a.other_name} | Speed-up | h3metal peak footprint | {a.other_name} peak footprint |")
        print("|---|---:|---:|---:|---:|---:|")
        for size in sorted({k[1] for k in groups if k[0] == cold}):
            ours, other = groups.get((cold, size, "h3metal"), []), groups.get((cold, size, "other"), [])
            t_ours = statistics.median(r["wall"] for r in ours) if ours else None
            t_other = statistics.median(r["wall"] for r in other) if other else None
            f_ours = max(r["footprint_gb"] for r in ours) if ours else None
            f_other = max(r["footprint_gb"] for r in other) if other else None
            cell = lambda v, unit: f"{v:.1f} {unit}" if v is not None else "-"  # noqa: E731
            ratio = f"{t_other / t_ours:.2f}x" if t_ours and t_other else "-"
            n = f" ({len(ours)}/{len(other)} runs)"
            print(f"| {size} x {size}{n} | {cell(t_ours, 's')} | {cell(t_other, 's')} | {ratio} | "
                  f"{cell(f_ours, 'GB')} | {cell(f_other, 'GB')} |")


if __name__ == "__main__":
    main()
