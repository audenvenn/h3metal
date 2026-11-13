"""Reference side of the Z-Image pipeline test (runs in an environment with torch, diffusers and transformers).

Small random versions of the three models with matching sizes (transformer, Qwen3 text encoder with the real
vocabulary, FLUX-layout VAE), saved in Z-Image's directory layout next to the real tokenizer (OUT_DIR/tokenizer,
fetched beforehand), and one image from diffusers' ZImagePipeline: everything between the prompt and the pixels.

    python tests/ref/mini_zimage_pipe.py OUT_DIR
"""
import json
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

out = sys.argv[1]
torch.manual_seed(0)


def init(model, std2d=0.5):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                v = 1.0 + 0.1 * torch.randn_like(p)
            elif "pad_token" in name or "embed_tokens" in name:
                v = torch.randn_like(p)
            elif p.ndim >= 2:
                v = torch.randn_like(p) * (std2d / p[0].numel() ** 0.5)
            else:
                v = torch.randn_like(p) * 0.05
            p.copy_(v.to(torch.bfloat16).float())


te_cfg = Qwen3Config(vocab_size=151936, hidden_size=512, intermediate_size=1536, num_hidden_layers=4,
                     num_attention_heads=8, num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6,
                     rope_theta=1000000.0, max_position_embeddings=4096, tie_word_embeddings=True)
lm = Qwen3ForCausalLM(te_cfg).eval()
init(lm)
lm.save_pretrained(os.path.join(out, "text_encoder"), safe_serialization=True)
tr = ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=768, n_layers=2,
                              n_refiner_layers=2, n_heads=6, n_kv_heads=6, norm_eps=1e-5, qk_norm=True,
                              cap_feat_dim=512, rope_theta=256.0, t_scale=1000.0, axes_dims=[32, 48, 48],
                              axes_lens=[1536, 512, 512]).eval()
init(tr)
tr.save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)
vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=16, block_out_channels=(128, 128, 256, 256),
                    down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                    layers_per_block=2, norm_num_groups=32, mid_block_add_attention=True, use_quant_conv=False,
                    use_post_quant_conv=False, scaling_factor=0.3611, shift_factor=0.1159).eval()
init(vae, std2d=1.0)
vae.save_pretrained(os.path.join(out, "vae"), safe_serialization=True)
with open(os.path.join(out, "model_index.json"), "w") as f:
    json.dump({"_class_name": "ZImagePipeline"}, f)

pipe = ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0,
                                                                use_dynamic_shifting=False),
                      vae=vae, text_encoder=Qwen3Model.from_pretrained(os.path.join(out, "text_encoder")).eval(),
                      tokenizer=AutoTokenizer.from_pretrained(os.path.join(out, "tokenizer")), transformer=tr)
prompt = "A red fox sitting in fresh snow at sunrise, photorealistic."
H, W, seed = 192, 256, 7
with torch.no_grad():
    img = pipe(prompt=prompt, height=H, width=W, num_inference_steps=8, guidance_scale=0.0,
               generator=torch.Generator("cpu").manual_seed(seed), output_type="np").images[0]
np.savez(os.path.join(out, "pipe_ref.npz"), img=img, H=H, W=W, seed=seed, prompt=prompt)
print("saved", out, img.shape, float(img.mean()))
