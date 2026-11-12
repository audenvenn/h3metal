"""An edit's inputs against the official processor (see tests/ref/edit_parts.py): the vision tower's patches and
the text encoder's multimodal positions. The vision tower itself: tests/test_vision.py; the VAE encoder:
tests/test_vae_encoder.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal import edit  # noqa: E402

r = np.load(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/edit_ref.npz")
p, grid = edit.vision_patches(r["vis_img"])
err = np.abs(p - r["pixel_values"]).max()
print("patches:", p.shape, "max abs err", err, "grid", grid, r["grid"])
assert err < 1e-5 and tuple(grid) == tuple(r["grid"][0])
tags = (r["ids"] == 151655).astype(np.int32)
pos = edit.edit_positions(tags, [tuple(g) for g in r["grid"]])
print("positions equal:", np.array_equal(pos, r["pos3"]))
assert np.array_equal(pos, r["pos3"])
