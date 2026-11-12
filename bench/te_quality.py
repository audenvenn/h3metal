"""Text-encoder accuracy against reference embeddings (tests/ref/reference.py --te-only).

    python bench/te_quality.py TE.h3m REF_DIR [--act f16|i8]

Prints, per reference file, the relative error of hidden_states[50] and its
cosine similarity per token (the mean and the worst token).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal.gpu import Gpu  # noqa: E402
from h3metal.te import TextEncoder  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("te")
    ap.add_argument("refs")
    ap.add_argument("--act", default="f16")
    a = ap.parse_args()
    gpu = Gpu()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.refs, "te_ref_*.npz"))):
        r = np.load(path)
        if "tags" in r and (r["tags"] == 0).any():
            continue  # edit presentations need the vision tower; see bench/quality.py
        wf = WeightFile(gpu, a.te)
        te = TextEncoder(gpu, wf, act=a.act)
        t = time.time()
        h = te.encode(r["ids"])
        dt = time.time() - t
        wf.close()
        ref = r["emb"]
        rel = np.linalg.norm(h - ref) / np.linalg.norm(ref)
        cos = (h * ref).sum(1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
        rows.append(rel)
        print(f"{os.path.basename(path)} ({len(r['ids'])} tokens): rel error {rel:.4f}, "
              f"token cosine mean {cos.mean():.5f} min {cos.min():.5f}  ({dt:.1f} s)", flush=True)
    print(f"{os.path.basename(a.te)} act={a.act}: mean rel error {np.mean(rows):.4f}")


if __name__ == "__main__":
    main()
