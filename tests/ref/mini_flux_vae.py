"""Reference side of the FLUX decoder test (runs in an environment with torch and diffusers).

A small AutoencoderKL with the FLUX layout (16 latent channels, four levels, mid-block attention) and random
weights, saved as a diffusers checkpoint (OUT_DIR/vae), and one decode of random latents.

    python tests/ref/mini_flux_vae.py OUT_DIR
"""
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL

out = sys.argv[1]
torch.manual_seed(0)
# Channel counts are multiples of 128, as h3metal's decoder requires (the product kernel's tile width).
vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16, block_out_channels=(128, 128, 256, 256),
                    down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                    layers_per_block=2, norm_num_groups=32, mid_block_add_attention=True, use_quant_conv=False,
                    use_post_quant_conv=False, scaling_factor=0.3611, shift_factor=0.1159).float().eval()
with torch.no_grad():
    for name, p in vae.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        elif p.ndim >= 2:
            fan_in = p[0].numel()
            p.copy_(torch.randn_like(p) * (1.0 / fan_in ** 0.5))
        else:
            p.copy_(0.05 * torch.randn_like(p))
vae.save_pretrained(os.path.join(out, "vae"), safe_serialization=True)
z = torch.randn(1, 16, 20, 28)  # latents as the transformer leaves them
with torch.no_grad():
    img = vae.decode(z / vae.config.scaling_factor + vae.config.shift_factor, return_dict=False)[0]
np.savez(os.path.join(out, "vae_ref.npz"), z=z[0].numpy(), img=img[0].permute(1, 2, 0).numpy())
print("saved", out, tuple(img.shape))
