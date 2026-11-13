"""What a LoRA adapter costs: the denoising time of the same request with and without it, warm (the prompt's
embeddings cached, the model loaded), interleaved runs.

    python bench/lora_speed.py MODEL_DIR LORA.safetensors [--runs 3] [--size 1024]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.cli import open_pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("lora")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--size", type=int, default=1024)
a = ap.parse_args()
pipe, name = open_pipeline(os.path.expanduser(a.model))
prompt = "A lighthouse on a rocky coast at dusk, photorealistic."
times = {"without": [], "with": []}
for i in range(a.runs + 1):  # the first round warms up (text encoder, mapping the weights, the adapter)
    for tag, loras in (("without", []), ("with", [(a.lora, 1.0)])):
        _, timing = pipe.generate(prompt, a.size, a.size, seed=i, loras=loras)
        if i:
            times[tag].append(timing.stages["denoise"])
        print(f"{tag:8s} {timing}", flush=True)
w0, w1 = np.median(times["without"]), np.median(times["with"])
print(f"{name} {a.size}x{a.size}, {os.path.basename(a.lora)}: denoising {w0:.2f} s without, {w1:.2f} s with "
      f"(+{100 * (w1 / w0 - 1):.1f}%), median of {a.runs}")
