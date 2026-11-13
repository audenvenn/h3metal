"""Reference side of the FLUX.2 text encoder test (runs in an environment with torch and transformers).

A small Qwen3ForCausalLM with random weights, saved like FLUX.2's text_encoder/ (keys under "model."), and what
klein's pipeline reads from it: a prompt padded to a fixed length with a padding mask, the hidden states after
layers 1, 2 and 3 stacked per token.

    python tests/ref/mini_flux2_te.py OUT_DIR
"""
import os
import sys

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

out = os.path.join(sys.argv[1], "text_encoder")
torch.manual_seed(1)
cfg = Qwen3Config(vocab_size=1024, hidden_size=512, intermediate_size=1536, num_hidden_layers=4,
                  num_attention_heads=8, num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6,
                  rope_theta=1000000.0, max_position_embeddings=4096, tie_word_embeddings=True)
model = Qwen3ForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * (0.5 / p.shape[-1] ** 0.5) if p.ndim == 2 else torch.randn_like(p) * 0.02)
    for p in model.parameters():
        p.copy_(p.to(torch.bfloat16).float())  # the checkpoint is bfloat16; the reference computes in float32
model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
# Reload instead of model.float(): .to(bfloat16) also rounded the rotary inv_freq buffer, and angles from a
# rounded inv_freq drift from the true ones in proportion to the position (0.2% x position radians).
model = Qwen3ForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()

n_valid, T, pad, taps = 37, 128, 7, (1, 2, 3)
ids = torch.full((1, T), pad, dtype=torch.long)
ids[0, :n_valid] = torch.randint(10, cfg.vocab_size, (n_valid,))
mask = torch.zeros(1, T, dtype=torch.long)
mask[0, :n_valid] = 1
with torch.no_grad():
    o = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
feats = torch.cat([o.hidden_states[k] for k in taps], dim=-1)[0]
np.savez(os.path.join(sys.argv[1], "te_ref.npz"), ids=ids[0].numpy(), n_valid=n_valid, taps=np.array(taps),
         feats=feats.to(torch.bfloat16).float().numpy())
print("saved", sys.argv[1], feats.shape)
