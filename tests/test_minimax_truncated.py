"""MiniMax-H3-Image's few-step transformer (convert --pdmd) on real weights, truncated: the token refiner, the first
block, the embeddings and the output layer with pDMD's LoRA merged, against float32 on the CPU
(tests/ref/minimax_truncated.py), at both steps of the 2-step grid, in each transformer format.

    python tests/test_minimax_truncated.py SRC LORA REF_DIR

SRC is the original checkpoint, LORA pDMD's file, REF_DIR what the reference side wrote.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.convert import convert_dit  # noqa: E402
from h3metal.dit import DiT, Layout  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile, bf16_bits  # noqa: E402

src, lora, ref_dir = sys.argv[1:4]
r = np.load(os.path.join(ref_dir, "ref.npz"))
gpu = Gpu()
T = r["txt"].shape[0]


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


with tempfile.TemporaryDirectory() as d:
    for name, fmts in (("int8", "q8,q8,q8,q8"), ("int8 attention, int4 MLP", "q8,q8,q4,q4")):
        path = os.path.join(d, "dit.h3m")
        convert_dit(src, path, fmts=dict(zip(("qkv", "out", "fc1", "fc2"), fmts.split(","))), layers=1,
                    flow={"shift": 12.0, "steps": [2]}, lora=[(lora, 1.0)], log=lambda *a: None)
        wf = WeightFile(gpu, path)
        dit = DiT(gpu, wf)
        sigmas, first = dit.schedules[2]
        assert np.array_equal(np.array(sigmas, np.float32), r["sigmas"]), (sigmas, r["sigmas"])
        text = gpu.empty((-(-T // 128) * 128, r["txt"].shape[1]), "bf16")
        text.numpy()[:T] = bf16_bits(r["txt"])
        rows = dit.encode_text(text, T)
        dit.prepare(Layout(T, np.ones(T, np.uint8), [], int(r["lat_hw"][0]), int(r["lat_hw"][1])))
        for s in range(2):
            v = dit.step(first + s, rows, r["latents"])
            print(f"{name}: step {s} (sigma {sigmas[s]:.4f}): rel err {rel(v, r[f'vel{s}']):.4f} against float32")
        dit.free()
        wf.close()
