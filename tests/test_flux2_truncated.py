"""A FLUX.2 [klein] checkpoint's real weights, truncated, against float32 references on the CPU (the reference side
is tests/ref/flux2_truncated.py): one double-stream and one single-stream block with every embedding and
modulation layer, and the text encoder's first four layers. For checkpoints whose float32 reference does not fit
the Mac (9B); run it on 4B as well, whose whole pipeline is checked against diffusers (bench/flux2_quality.py),
to calibrate the errors.

    python tests/test_flux2_truncated.py SRC REF_DIR

SRC is the original checkpoint (for the text encoder's weights), REF_DIR what the reference side wrote.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.flux2 import Flux2DiT, convert_transformer, variant_name  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.te import TextEncoder, convert_te  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

src, ref_dir = sys.argv[1], sys.argv[2]
r = np.load(os.path.join(ref_dir, "ref.npz"))
gpu = Gpu()


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


with tempfile.TemporaryDirectory() as d:
    dit_path = os.path.join(d, "dit.h3m")
    convert_transformer(ref_dir, dit_path, log=lambda *a: None)
    wf = WeightFile(gpu, dit_path)
    dit = Flux2DiT(gpu, wf)
    name = variant_name(dit.D)
    dit.prepare(r["txt"], r["i_ids"])
    o = dit.step(float(r["sigma"]), r["latents"])
    err_dit = rel(o, r["out"])
    print(f"{name}, one block of each kind, real weights: step output rel err {err_dit:.4f} against float32")
    dit.free()
    wf.close()

    te_path = os.path.join(d, "te.h3m")
    L = int(r["te_layers"])
    convert_te(os.path.join(src, "text_encoder"), te_path, fmt="q8", layers=L, prefix="model", log=lambda *a: None)
    wf = WeightFile(gpu, te_path)
    n = int(r["n_valid"])
    out = TextEncoder(gpu, wf).encode(r["ids"].tolist(), taps=(2, 3), n_valid=n)
    D = r["hs2"].shape[1]
    for i, (k, hs) in enumerate((("hidden state 2", r["hs2"]), ("hidden state 3", r["hs3"]))):
        got = out[:, i * D:(i + 1) * D]
        print(f"{name} text encoder, {k}: prompt rows rel err {rel(got[:n], hs[:n]):.4f}, padding rows "
              f"{rel(got[n:], hs[n:]):.4f}")
    wf.close()
