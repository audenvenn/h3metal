"""The vision tower on the engine against the official module's features (see tests/ref/edit_parts.py)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal import edit  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.vision import VisionTower  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

r = np.load(os.environ.get("H3METAL_REF", "ref") + "/edit_ref.npz")
path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.h3metal/model/vision.h3m")


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


gpu = Gpu()
vt = VisionTower(gpu, WeightFile(gpu, path))
p, grid = edit.vision_patches(r["vis_img"])
vt(p, grid)  # compiles the kernels
t = time.time()
emb, ds = vt(p, grid)
dt = time.time() - t
errs = [rel(d, r[f"deepstack{i}"]) for i, d in enumerate(ds)]
print(f"vision tower {p.shape[0]} patches, {dt:.2f} s: embeds rel err {rel(emb, r['vis_embeds']):.4f}, "
      f"deepstack {' '.join(f'{e:.4f}' for e in errs)}")
assert rel(emb, r["vis_embeds"]) < 0.03 and max(errs) < 0.03
