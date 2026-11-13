"""Reference side of the real-weights check for a FLUX.2 [klein] checkpoint too large for a float32 reference on
the Mac (9B): its first double-stream and first single-stream block with every embedding and modulation layer,
and its text encoder's first layers, run in float32 on the CPU (torch, diffusers, transformers).

    python tests/ref/flux2_truncated.py SRC OUT_DIR CAPTION.npy

SRC is the original checkpoint (diffusers layout). CAPTION.npy holds real caption features [512][joint] for the
transformer's input (h3metal's text encoder on a prompt): a few channels of real ones are two orders of magnitude
above the rest, which random features would not exercise. OUT_DIR receives:
  transformer/   a config with one block of each kind, next to links to SRC's weight shards (diffusers reads only
                 the blocks the config has), for h3metal's converter
  ref.npz        latents [n][C] (unit normal: the first step's noise), the caption features, image positions,
                 sigma, the step's output (float32); the text encoder's ids and its hidden states 2 and 3 (float32;
                 transformers normalizes the last one, 4 here, which the pipeline's taps never are)
"""
import json
import os
import sys

import numpy as np
import torch

src, out, cap = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)

# 1. The truncated transformer: config with one block of each kind, the shards linked.
cfg = json.load(open(os.path.join(src, "transformer", "config.json")))
cfg.update(num_layers=1, num_single_layers=1)
json.dump(cfg, open(os.path.join(out, "transformer", "config.json"), "w"), indent=1)
for f in os.listdir(os.path.join(src, "transformer")):
    if f.endswith(".safetensors") or f.endswith(".index.json"):
        dst = os.path.join(out, "transformer", f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(src, "transformer", f), dst)

from diffusers import Flux2Transformer2DModel  # noqa: E402

model = Flux2Transformer2DModel.from_pretrained(os.path.join(out, "transformer"), torch_dtype=torch.float32).eval()
D = cfg["num_attention_heads"] * cfg["attention_head_dim"]
joint = cfg["joint_attention_dim"]
gh, gw, n_txt = 32, 32, 512  # 512 x 512 pixels: 1024 image rows
lat = rng.standard_normal((gh * gw, cfg["in_channels"])).astype(np.float32)
txt = np.load(cap).astype(np.float32)
assert txt.shape == (n_txt, joint), txt.shape
i_ids = np.stack(np.meshgrid(np.arange(1), np.arange(gh), np.arange(gw), np.arange(1), indexing="ij"),
                 -1).reshape(-1, 4)
t_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n_txt))[None]
sigma = 1.0
with torch.no_grad():
    o = model(hidden_states=torch.from_numpy(lat)[None], encoder_hidden_states=torch.from_numpy(txt)[None],
              timestep=torch.tensor([sigma]), img_ids=torch.from_numpy(i_ids)[None], txt_ids=t_ids, guidance=None,
              return_dict=False)[0][0].numpy()
rec = {"latents": lat, "txt": txt, "i_ids": i_ids, "sigma": np.float32(sigma), "out": o}
del model

# 2. The text encoder's first layers.
from transformers import AutoTokenizer, Qwen3ForCausalLM  # noqa: E402

TE_LAYERS = 4
tok = AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))
text = tok.apply_chat_template([{"role": "user", "content": "A lighthouse on a rocky coast at dusk, photorealistic."}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
te = Qwen3ForCausalLM.from_pretrained(os.path.join(src, "text_encoder"), torch_dtype=torch.float32,
                                      num_hidden_layers=TE_LAYERS).eval()
with torch.no_grad():
    hs = te(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True,
            use_cache=False).hidden_states
rec.update(ids=enc["input_ids"][0].numpy(), n_valid=int(enc["attention_mask"].sum()), te_layers=TE_LAYERS,
           hs2=hs[2][0].numpy(), hs3=hs[3][0].numpy())
np.savez(os.path.join(out, "ref.npz"), **rec)
print("saved", out, {k: getattr(v, "shape", v) for k, v in rec.items()})
