"""Reference VAE decode with the official module (torch + diffusers env).
    python tests/ref/vae_decode.py WEIGHTS_DIR OUT.npz [lat_h lat_w]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ["DIFFUSERS_SCRIPTS"])  # diffusers/scripts, for its weight conversion
import convert_minimax_h3_to_diffusers as conv
from safetensors import safe_open
from diffusers import AutoencoderKLMiniMaxH3Image
src, out = sys.argv[1], sys.argv[2]
lh, lw = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (32, 40)
cfg = json.load(open(os.path.join(src, "vae_diffusers", "config.json")))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
vae = AutoencoderKLMiniMaxH3Image(**cfg).float().eval()
sd = {}
with safe_open(os.path.join(src, "vae", "diffusion_pytorch_model.safetensors"), "pt") as f:
    for k in f.keys():
        for tk, t in conv.convert_video_vae_key(k, f.get_tensor(k), cfg):
            sd[tk] = t
missing, unexpected = vae.load_state_dict(sd, strict=False)
print("missing", missing[:5], "unexpected", unexpected[:5])
rng = np.random.default_rng(7)
lat = rng.standard_normal((24, lh, lw)).astype(np.float32)
mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1); std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1)
with torch.no_grad():
    img = vae.decode(torch.from_numpy(lat)[None] * std + mean, return_dict=False)[0]
np.savez(out, lat=lat, img=img[0].numpy())
print("saved", img.shape)
