"""Reference side of the mini FLUX.2 test (runs in an environment with torch and diffusers).

Builds a small Flux2Transformer2DModel with random weights, saves it as a diffusers checkpoint
(OUT_DIR/transformer) and records, for klein's four-step schedule from fixed latents and caption features, the
transformer's output at every step on the reference trajectory, plus the schedule itself.

    python tests/ref/mini_flux2.py OUT_DIR
"""
import json
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
# The width (heads x 128) and the MLP's width (3x) must be multiples of 256 (h3metal rotates blocks of 256).
cfg = dict(patch_size=1, in_channels=128, num_layers=2, num_single_layers=2, attention_head_dim=128,
           num_attention_heads=2, joint_attention_dim=192, timestep_guidance_channels=256, mlp_ratio=3.0,
           axes_dims_rope=(32, 32, 32, 32), rope_theta=2000, eps=1e-6, guidance_embeds=False)
model = Flux2Transformer2DModel(**cfg).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.ndim == 1:
            v = 1.0 + 0.1 * torch.randn_like(p)
        elif "timestep_embedder" in name:
            v = torch.randn_like(p) * (1.0 / p.shape[1] ** 0.5)
        elif "modulation" in name or "norm_out" in name:
            # Modulation of order 1, as in a trained model: with small random weights the gates come out near
            # 0.03, every block adds almost nothing, and the test would not see the blocks at all.
            v = torch.randn_like(p) * (2.0 / p.shape[1] ** 0.5)
        elif "x_embedder" in name or "context_embedder" in name:
            v = torch.randn_like(p) * (0.2 / p.shape[1] ** 0.5)  # small inputs: the blocks make most of the output
        elif p.ndim == 2:
            v = torch.randn_like(p) * (1.0 / p.shape[1] ** 0.5)
        else:
            v = torch.randn_like(p) * 0.05
        p.copy_(v.to(torch.bfloat16).float())  # keep the values bf16-exact, like the released checkpoint
model.save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)

# Inputs: a 9 x 11 grid of image rows and 128 caption rows (klein pads captions to 512, a multiple of 128).
gh, gw, n_txt = 9, 11, 128
latents = torch.randn(1, gh * gw, cfg["in_channels"])
txt = torch.randn(1, n_txt, cfg["joint_attention_dim"])
t_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n_txt))[None]
i_ids = torch.cartesian_prod(torch.arange(1), torch.arange(gh), torch.arange(gw), torch.arange(1))[None]

sched = FlowMatchEulerDiscreteScheduler.from_config(json.load(open(sys.argv[2]))) if len(sys.argv) > 2 else \
    FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
                                    base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096,
                                    time_shift_type="exponential")
steps = 4
mu = compute_empirical_mu(image_seq_len=gh * gw, num_steps=steps)
sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
rec = {"latents": latents[0].numpy(), "txt": txt[0].numpy(), "i_ids": i_ids[0].numpy(),
       "timesteps": sched.timesteps.numpy(), "sigmas": sched.sigmas.numpy(), "mu": np.float64(mu)}
x = latents.clone()
with torch.no_grad():
    for i, t in enumerate(sched.timesteps):
        o = model(hidden_states=x, encoder_hidden_states=txt, timestep=t.expand(1) / 1000, img_ids=i_ids,
                  txt_ids=t_ids, guidance=None, return_dict=False)[0]
        rec[f"x{i}"] = x[0].numpy()
        rec[f"out{i}"] = o[0].numpy()
        x = sched.step(o, t, x, return_dict=False)[0]
rec["final"] = x[0].numpy()
with torch.no_grad():  # how much the blocks matter: the step-0 output with every block's gates zeroed
    for m in (model.double_stream_modulation_img, model.double_stream_modulation_txt, model.single_stream_modulation):
        m.linear.weight.view(-1, 3, cfg["num_attention_heads"] * 128)[:, 2] = 0
    o0 = model(hidden_states=latents, encoder_hidden_states=txt, timestep=sched.timesteps[:1] / 1000, img_ids=i_ids,
               txt_ids=t_ids, guidance=None, return_dict=False)[0]
    print("blocks' share of the step-0 output (rel L2 change without them):",
          float((o0[0] - torch.from_numpy(rec["out0"])).norm() / torch.from_numpy(rec["out0"]).norm()))
np.savez(os.path.join(out, "ref.npz"), **rec)
json.dump(cfg, open(os.path.join(out, "mini_config.json"), "w"), indent=1)
print("saved", out, "sigmas", [round(float(s), 4) for s in sched.sigmas], "mu", round(mu, 4))
