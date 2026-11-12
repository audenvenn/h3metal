"""An edit's text-encoder input and output against the official modules, real weights.

The presentation of the reference images (labels, vision blocks, multimodal
positions), the vision tower's features and the DeepStack additions all
feed the text encoder; this compares h3metal's hidden_states[50] with a
reference run of the official processor, vision tower and decoder layers
(tests/ref/reference.py --te-only --image).

    python tests/test_edit_te.py MODEL_DIR REF.npz PROMPT_FILE IMAGE
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal.pipeline import Pipeline, _Draws, edit_inputs  # noqa: E402
from h3metal.te import TextEncoder  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402


def main():
    model, ref_path, prompt_file, image = sys.argv[1:5]
    from PIL import Image
    prompt = open(prompt_file).read()
    # A model directory without a transformer is enough here.
    d = tempfile.mkdtemp()
    try:
        for f in ("te.h3m", "vae.h3m", "vision.h3m", "tokenizer.json"):
            os.symlink(os.path.join(model, f), os.path.join(d, f))
        open(os.path.join(d, "dit.h3m"), "w").close()
        pipe = Pipeline(d, log=lambda *a: None)
        ids, types, pos3, rows, feats, deep, _ = edit_inputs(pipe, prompt, [Image.open(image)], _Draws(0))
        wf = WeightFile(pipe.gpu, pipe.paths["te"])
        te = TextEncoder(pipe.gpu, wf)
        t = time.time()
        h = te.encode(ids, pos3=pos3, vision=feats, vision_rows=rows, deepstack=deep)
        dt = time.time() - t
        wf.close()
    finally:
        shutil.rmtree(d)
    r = np.load(ref_path)
    same_ids = len(ids) == len(r["ids"]) and np.array_equal(ids, r["ids"])
    ref = r["emb"].astype(np.float32)
    print(f"tokens: {len(ids)} (reference {len(r['ids'])}), identical: {same_ids}; vision rows {len(rows)}")
    if not same_ids:
        sys.exit(1)
    rel = np.linalg.norm(h - ref) / np.linalg.norm(ref)
    cos = (h * ref).sum(1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
    vis = np.zeros(len(ids), bool)
    vis[rows] = True
    rv = np.linalg.norm(h[vis] - ref[vis]) / np.linalg.norm(ref[vis])
    rt = np.linalg.norm(h[~vis] - ref[~vis]) / np.linalg.norm(ref[~vis])
    print(f"hidden_states[50] ({te.fmt}, {dt:.1f} s): rel error {rel:.4f} (vision rows {rv:.4f}, text rows {rt:.4f}); "
          f"token cosine mean {cos.mean():.5f} min {cos.min():.5f}")
    worst = np.argsort(cos)[:5]
    print("lowest cosines:", ", ".join(f"token {i} (id {int(ids[i])}, {'vision' if vis[i] else 'text'}, "
                                       f"|ref| {np.linalg.norm(ref[i]):.1f}): {cos[i]:.3f}" for i in worst))
    assert cos.mean() > 0.99, "the edit presentation does not match the reference"


if __name__ == "__main__":
    main()
