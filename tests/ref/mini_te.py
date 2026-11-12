"""Reference side of the mini text-encoder test: a small random Qwen3-VL text model (torch + transformers).
    python tests/ref/mini_te.py OUT_DIR"""
import json
import os
import sys
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
tc = dict(hidden_size=512, intermediate_size=1024, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
          head_dim=128, vocab_size=1000, rms_norm_eps=1e-6, max_position_embeddings=4096,
          rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_interleaved": True,
                           "mrope_section": [24, 20, 20]})
vc = dict(depth=2, hidden_size=128, intermediate_size=256, num_heads=2, out_hidden_size=512, patch_size=16,
          spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0, 1, 1], num_position_embeddings=64)
cfg = Qwen3VLConfig(text_config=tc, vision_config=vc, image_token_id=999, video_token_id=998,
                    vision_start_token_id=997, vision_end_token_id=996)
model = Qwen3VLForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if "norm" in n:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * (0.7 / p.shape[-1] ** 0.5 if p.ndim >= 2 else 0.02))
        p.copy_(p.to(torch.bfloat16).float())
sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items() if "language_model" in k}
save_file(sd, os.path.join(out, "model.safetensors"))
full = cfg.to_dict()
full["text_config"]["rope_theta"] = 5000000.0
full["text_config"]["rope_scaling"] = {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"}
json.dump(full, open(os.path.join(out, "config.json"), "w"), default=str)
rng = np.random.default_rng(3)
ids = torch.from_numpy(rng.integers(0, 990, size=(1, 77)))
with torch.no_grad():
    o = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
np.savez(os.path.join(out, "ref.npz"), ids=ids[0].numpy(), **{f"h{i}": o.hidden_states[i][0].numpy() for i in range(len(o.hidden_states))})
print("hidden states", len(o.hidden_states), o.hidden_states[2].shape)
