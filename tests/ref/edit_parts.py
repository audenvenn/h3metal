"""Reference for the edit-only parts, with the official modules on the real weights (torch env):
image resizing, the Qwen3-VL processor, the vision tower (+ DeepStack), the text encoder's multimodal
positions, and the VAE encoder's posterior.

    python tests/ref/edit_parts.py WEIGHTS_DIR IMAGE OUT.npz
"""
import json, os, sys
import numpy as np, torch
from PIL import Image
from safetensors import safe_open

sys.path.insert(0, os.environ["DIFFUSERS_SCRIPTS"])  # diffusers/scripts, for its weight conversion
import convert_minimax_h3_to_diffusers as conv  # noqa: E402
from diffusers import AutoencoderKLMiniMaxH3Image  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLConfig  # noqa: E402
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm  # noqa: E402

W, img_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {}
image = Image.open(img_path).convert("RGB")
proc = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
def resize(im, lo, hi):
    size = bound_short_edge(*im.size, lo, hi, 32)
    return im if im.size == size else proc.resize(im, height=size[1], width=size[0])
ref_img = resize(image, 1024, 2048)
vis_img = resize(image, 1024, 1024)
res["ref_img"] = np.array(ref_img)
res["vis_img"] = np.array(vis_img)

processor = AutoProcessor.from_pretrained(os.path.join(W, "processor"))
vision = processor.image_processor(images=[vis_img], return_tensors="pt")
res["pixel_values"] = vision["pixel_values"].numpy()
res["grid"] = vision["image_grid_thw"].numpy()

cfg = Qwen3VLConfig.from_pretrained(os.path.join(W, "text_encoder"))
vm = qm.Qwen3VLVisionModel._from_config(cfg.vision_config).float().eval()
sd = {}
with safe_open(os.path.join(W, "text_encoder", "model-00014-of-00014.safetensors"), "pt") as f:
    for k in f.keys():
        if k.startswith("model.visual."):
            sd[k[len("model.visual."):]] = f.get_tensor(k).float()
print(vm.load_state_dict(sd, strict=False))
with torch.no_grad():
    vo = vm(vision["pixel_values"].float(), grid_thw=vision["image_grid_thw"])
res["vis_embeds"] = vo.pooler_output.numpy()
for i, d in enumerate(vo.deepstack_features):
    res[f"deepstack{i}"] = d.numpy()

tok = AutoTokenizer.from_pretrained(os.path.join(W, "tokenizer"))
merge = processor.image_processor.merge_size ** 2
ids, tags = [], []
for i, grid in enumerate(vision["image_grid_thw"]):
    lab = tok(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"]
    vb = [tok.convert_tokens_to_ids("<|vision_start|>")] + [tok.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge) + [tok.convert_tokens_to_ids("<|vision_end|>")]
    ids += lab + vb
    tags += [1] * len(lab) + [0] * len(vb)
prompt = "Replace the background with a snowy mountain landscape at dusk."
pids = tok(prompt, add_special_tokens=False)["input_ids"]
ids += pids
tags += [1] * len(pids)
res["ids"], res["tags"] = np.array(ids), np.array(tags)
mm = torch.tensor(processor.create_mm_token_type_ids([ids]))
full = qm.Qwen3VLModel.__new__(qm.Qwen3VLModel)
torch.nn.Module.__init__(full)
full.config = cfg
pos, _ = qm.Qwen3VLModel.get_rope_index(full, torch.tensor([ids]), mm, image_grid_thw=vision["image_grid_thw"])
res["pos3"] = pos[:, 0].numpy()

vcfg = json.load(open(os.path.join(W, "vae_diffusers", "config.json")))
vcfg = {k: v for k, v in vcfg.items() if not k.startswith("_")}
vae = AutoencoderKLMiniMaxH3Image(**vcfg).float().eval()
vsd = {}
with safe_open(os.path.join(W, "vae", "diffusion_pytorch_model.safetensors"), "pt") as f:
    for k in f.keys():
        for tk, t in conv.convert_video_vae_key(k, f.get_tensor(k), vcfg):
            vsd[tk] = t
vae.load_state_dict(vsd, strict=False)
pix = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1)[None]
pix = pix[..., : pix.shape[-2] // 16 * 16, : pix.shape[-1] // 16 * 16]
pm = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1); ps = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
pix = (pix.float() / 255.0 - pm) / ps
with torch.no_grad():
    post = vae.encode(pix, return_dict=False)[0]
res["post_mean"] = post.mean[0].numpy()
res["post_std"] = post.std[0].numpy()
np.savez(out, **res)
print({k: v.shape for k, v in res.items()})
