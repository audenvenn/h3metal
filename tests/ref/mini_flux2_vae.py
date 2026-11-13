"""Reference side of the FLUX.2 decoder test (runs in an environment with torch and diffusers).

A small AutoencoderKLFlux2 (32 latent channels, the 1 x 1 convolution before the decoder, batch-norm statistics
for the 2 x 2 patches the transformer works on) with random weights, saved as a diffusers checkpoint
(OUT_DIR/vae), and one decode the way klein's pipeline does it: the transformer's packed latents denormalized
with the statistics, unpacked, decoded. tests/test_flux_vae.py reads the result.

    python tests/ref/mini_flux2_vae.py OUT_DIR
"""
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2

out = sys.argv[1]
torch.manual_seed(0)
# Channel counts are multiples of 128, as h3metal's decoder requires (the product kernel's tile width).
vae = AutoencoderKLFlux2(in_channels=3, out_channels=3, latent_channels=32, block_out_channels=(128, 128, 256, 256),
                         down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                         layers_per_block=2, norm_num_groups=32, mid_block_add_attention=True, use_quant_conv=True,
                         use_post_quant_conv=True, batch_norm_eps=1e-4, patch_size=(2, 2)).float().eval()
with torch.no_grad():
    for name, p in vae.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        elif p.ndim >= 2:
            fan_in = p[0].numel()
            p.copy_(torch.randn_like(p) * (1.0 / fan_in ** 0.5))
        else:
            p.copy_(0.05 * torch.randn_like(p))
    vae.bn.running_mean.copy_(0.3 * torch.randn(128))
    vae.bn.running_var.copy_(0.5 + torch.rand(128))
vae.save_pretrained(os.path.join(out, "vae"), safe_serialization=True)
z = torch.randn(1, 128, 10, 14)  # packed latents as the transformer leaves them
with torch.no_grad():
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    lat = z * std + vae.bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = lat.shape
    lat = lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
    img = vae.decode(lat, return_dict=False)[0]
    # The encoder (an edit's reference images): a random image, its latent mean (the pipeline's argmax).
    x = torch.rand(1, 3, 96, 128) * 2 - 1
    mean = vae.encode(x).latent_dist.mode()
np.savez(os.path.join(out, "vae_ref.npz"), z=z[0].numpy(), img=img[0].permute(1, 2, 0).numpy(),
         enc_img=x[0].permute(1, 2, 0).numpy(), enc_mean=mean[0].numpy())
print("saved", out, tuple(img.shape))
