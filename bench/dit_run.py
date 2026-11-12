"""Times a whole denoising run (6 steps x 50 blocks) of a transformer file at a given size, with memory counters:
python bench/dit_run.py DIT.h3m SIZE [SIZE...]"""
import os
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.weights import WeightFile, bf16_bits
from h3metal.dit import DiT, Layout, patchify, euler
from h3metal.rng import TorchRandn

def pageins():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pageins"):
            return int(line.split()[-1].rstrip("."))
    return 0

path = sys.argv[1]
gpu = Gpu()
T = 300
for size in [int(s) for s in sys.argv[2:]]:
    wf = WeightFile(gpu, path)
    t0 = time.time()
    m = DiT(gpu, wf)
    emb = (np.random.default_rng(1).standard_normal((T, 5120)) * 0.5).astype(np.float32)
    text = gpu.empty((384, 5120), "bf16"); text.numpy()[:T] = bf16_bits(emb)
    rows = m.encode_text(text, T)
    lat = size // 16
    m.prepare(Layout(T, np.ones(T, np.uint8), [], lat, lat))
    x = patchify(TorchRandn(0).normal((1, 24, 1, lat, lat)).reshape(24, lat, lat))
    p0 = pageins(); t1 = time.time(); steps = []
    for s in range(6):
        ts = time.time()
        v = m.step(s, rows, x)
        x = euler(x, v, s, m.sigmas)
        steps.append(time.time() - ts)
    t2 = time.time(); p1 = pageins()
    print(f"{os.path.basename(path)} {size}x{size}: load+text {t1 - t0:.1f}s, denoise {t2 - t1:.1f}s "
          f"(steps {' '.join(f'{s:.1f}' for s in steps)}), page-ins {(p1 - p0) * 16384 / 1e9:.1f} GB, finite={np.isfinite(x).all()}", flush=True)
    m.free(); wf.close()
