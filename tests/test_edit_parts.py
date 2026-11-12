"""Edit-only parts against the official modules (see tests/ref/edit_parts.py)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal import edit
from h3metal.st import SafeTensors

W = os.environ["H3METAL_WEIGHTS"]  # the official MiniMax-H3-Image directory
r = np.load(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/edit_ref.npz")
def rel(a, b): return np.linalg.norm(a - b) / np.linalg.norm(b)

p, grid = edit.vision_patches(r["vis_img"])
print("patches:", p.shape, "max abs err", np.abs(p - r["pixel_values"]).max(), "grid", grid, r["grid"])
st = SafeTensors(os.path.join(W, "text_encoder", "model-00014-of-00014.safetensors"))
vw = {k[len("model.visual."):]: st.get(k) for k in st.keys() if k.startswith("model.visual.")}
cfg = json.load(open(os.path.join(W, "text_encoder", "config.json")))["vision_config"]
t = time.time()
vt = edit.VisionTower(vw, cfg)
emb, ds = vt(p, grid)
print(f"vision tower {time.time() - t:.2f}s: embeds rel err {rel(emb, r['vis_embeds']):.4f}",
      " deepstack", [f"{rel(d, r[f'deepstack{i}']):.4f}" for i, d in enumerate(ds)])
tags = (r["ids"] == 151655).astype(np.int32)
pos = edit.edit_positions(tags, [tuple(g) for g in r["grid"]])
print("positions equal:", np.array_equal(pos, r["pos3"]))
vs = SafeTensors(os.path.join(W, "vae", "diffusion_pytorch_model.safetensors"))
ew = {k: vs.get(k) for k in vs.keys() if k.startswith(("encoder.", "quant_conv"))}
enc = edit.VAEEncoder(ew)
t = time.time()
mean, std = enc.encode(r["ref_img"])
print(f"vae encoder {time.time() - t:.2f}s: mean rel err {rel(mean, r['post_mean']):.5f} std rel err {rel(std, r['post_std']):.5f}")
