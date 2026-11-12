"""The whole Z-Image path, prompt to pixels (tokenizer and chat template, text encoder, noise, schedule,
transformer, VAE scaling, decoder, postprocessing), against diffusers' ZImagePipeline on small random models
(reference: tests/ref/mini_zimage_pipe.py).

    python tests/test_zimage_pipe.py REF_DIR
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.zimage import ZImagePipeline, convert_zimage, is_zimage_checkpoint, is_zimage_dir  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_zimage_pipe"
out = os.path.join(src, "h3")
assert is_zimage_checkpoint(src)
os.environ["H3METAL_CACHE"] = "off"
for te in ("f16", "q8"):
    convert_zimage(src, out, dit="q8", te=te, log=lambda *a: None)
    assert is_zimage_dir(out)
    r = np.load(os.path.join(src, "pipe_ref.npz"))
    pipe = ZImagePipeline(out, log=lambda *a: None)
    img, timing = pipe.generate(str(r["prompt"]), int(r["W"]), int(r["H"]), seed=int(r["seed"]))
    ref = np.rint(r["img"] * 255).astype(np.float64)
    mse = ((img.astype(np.float64) - ref) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
    print(f"text encoder {te}: {img.shape}, PSNR against diffusers {psnr:.1f} dB, mean abs {np.abs(img - ref).mean():.2f}"
          f" ({timing})")
    assert psnr > 30, psnr
print("ok")
