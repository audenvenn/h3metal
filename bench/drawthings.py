"""h3metal against Draw Things' command-line tool (draw-things-cli, local inference) on FLUX.2 [klein] 4B and
Z-Image-Turbo: the same prompt, size, steps, guidance (CFG 1) and seed. Every run is a fresh process timed by
bench/speed.py's run; the implementations alternate, with a pause before every run, and one warm-up of each is
not recorded. h3metal's prompt cache is off, so both encode the prompt every run. Draw Things keeps its
recommended settings for each model otherwise.

Draw Things' models: draw-things-cli models ensure --models-dir DIR --model M --include-dependencies, for the four
files in CASES.

    python bench/drawthings.py --cli path/to/draw-things-cli --models-dir DIR --out drawthings.json
"""
import argparse
import json
import os
import shlex
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed import PROMPT, run  # noqa: E402

# model: (h3metal --model, steps, Draw Things' 8-bit S and 8-bit files)
CASES = {"FLUX.2 [klein] 4B": ("flux2", 4, ("flux_2_klein_4b_i8x.ckpt", "flux_2_klein_4b_q8p.ckpt")),
         "Z-Image-Turbo": ("zimage", 8, ("z_image_turbo_1.0_i8x.ckpt", "z_image_turbo_1.0_q8p.ckpt"))}

ap = argparse.ArgumentParser()
ap.add_argument("--cli", default="draw-things-cli")
ap.add_argument("--models-dir", required=True)
ap.add_argument("--h3metal", default=f"{sys.executable} -m h3metal generate")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--cooldown", type=float, default=45.0)
ap.add_argument("--images", default="drawthings-images")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.images, exist_ok=True)
os.environ["H3METAL_CACHE"] = "off"
results, n = [], 0
for model, (h3, steps, files) in CASES.items():
    variants = [("h3metal", shlex.split(a.h3metal) + ["--model", h3])] + [
        (f"Draw Things {f[:-5].rsplit('_', 1)[1]}",
         [a.cli, "generate", "--local", "--models-dir", a.models_dir, "--model", f]) for f in files]
    for r in range(-1, a.runs):
        n += 1
        for name, cmd in variants[n % len(variants):] + variants[:n % len(variants)]:
            out = os.path.join(a.images, f"{h3}-{name.split()[-1]}-{r}.png")
            if name == "h3metal":
                cmd = cmd + ["-p", PROMPT, "-s", f"{a.size}x{a.size}", "--seed", "42", "-o", out]
            else:
                cmd = cmd + ["--prompt", PROMPT, "--width", str(a.size), "--height", str(a.size), "--steps",
                             str(steps), "--cfg", "1", "--seed", "42", "--output", out]
            time.sleep(a.cooldown)
            res = dict(run(cmd), model=model, impl=name, run=r, steps=steps)
            print(f"{model:18s} {name:16s} {'warm-up' if r < 0 else f'run {r}'}: {res['wall']:.1f} s, peak footprint "
                  f"{res['footprint_gb']:.2f} GB{'' if res['ok'] else ' FAILED: ' + res['stderr'][-300:]}", flush=True)
            if r >= 0:
                results.append(res)
                json.dump(results, open(a.out, "w"), indent=1)
for model in CASES:
    for name in dict.fromkeys(x["impl"] for x in results if x["model"] == model):
        walls = [x["wall"] for x in results if x["model"] == model and x["impl"] == name and x["ok"]]
        print(f"{model:18s} {name:16s} mean {statistics.mean(walls):.1f} s over {len(walls)} runs" if walls else
              f"{model:18s} {name:16s} failed")
