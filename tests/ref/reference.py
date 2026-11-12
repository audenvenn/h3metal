"""Reference generation with the official modules, streamed layer by layer.

MiniMax-H3-Image needs ~135 GB in bfloat16, far more than this Mac has, so
this runs the official diffusers/transformers modules one decoder layer at a
time, loading each layer's weights from the original safetensors right
before it runs. The math is the released pipeline's (text encoder
hidden_states[50], packed sequence, six PDD steps, tiled VAE decode); only
the scheduling of weight loads differs. Used to validate h3metal and to
measure its image quality.

    python tests/ref/reference.py --weights DIR --prompt "..." --size 1024x1024 \
        --min-area 1024 --seed 0 --dtype bf16 --out ref.npz
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.environ["DIFFUSERS_SCRIPTS"])  # diffusers/scripts, for its weight conversion
import convert_minimax_h3_to_diffusers as conv  # noqa: E402
from diffusers import AutoencoderKLMiniMaxH3Image  # noqa: E402
from diffusers.models.transformers import transformer_minimax_h3_image as tm  # noqa: E402
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (  # noqa: E402
    build_image_packed_sequence, patchify_video_latents, get_image_canvas_size)
from transformers import AutoTokenizer, Qwen3VLConfig  # noqa: E402
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm  # noqa: E402


class Shards:
    def __init__(self, d):
        self.d = d
        idx = [f for f in os.listdir(d) if f.endswith(".index.json")]
        self.wm = json.load(open(os.path.join(d, idx[0])))["weight_map"] if idx else None
        self.single = None if idx else [f for f in os.listdir(d) if f.endswith(".safetensors")][0]
        self.open = {}

    def get(self, k):
        fn = self.wm[k] if self.wm else self.single
        if fn not in self.open:
            self.open[fn] = safe_open(os.path.join(self.d, fn), "pt")
        return self.open[fn].get_tensor(k)

    def keys_with(self, prefix):
        ks = self.wm.keys() if self.wm else []
        return [k for k in ks if k.startswith(prefix)]


def text_encoder(W, ids, dev, dtype, layers=50):
    cfg = Qwen3VLConfig.from_pretrained(os.path.join(W, "text_encoder"))
    tc = cfg.text_config
    tc._attn_implementation = "sdpa"
    sh = Shards(os.path.join(W, "text_encoder"))
    pre = "model.language_model"
    T = len(ids)
    emb = sh.get(f"{pre}.embed_tokens.weight")[torch.tensor(ids)]
    h = emb.to(dev, dtype)[None]
    rot = qm.Qwen3VLTextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(T, device=dev)[None].expand(3, 1, T)
    pe = rot(h, pos)
    mask = torch.full((T, T), float("-inf"), device=dev).triu(1)[None, None].to(dtype)
    layer = qm.Qwen3VLTextDecoderLayer(tc, 0).to(dev, dtype).eval()
    for i in range(layers):
        sd = {k[len(f"{pre}.layers.{i}."):]: sh.get(k) for k in sh.keys_with(f"{pre}.layers.{i}.")}
        layer.load_state_dict(sd)
        layer.to(dev, dtype)
        with torch.no_grad():
            h = layer(h, attention_mask=mask, position_ids=pos[0], position_embeddings=pe)
            if isinstance(h, tuple):
                h = h[0]
    return h[0].to(torch.bfloat16)  # the pipeline casts to the encoder's dtype


def dit_denoise(W, emb, lat_h, lat_w, noise, dev, dtype, log):
    cfgj = json.load(open(os.path.join(W, "transformer_diffusers", "config.json")))
    cfgj = {k: v for k, v in cfgj.items() if not k.startswith("_")}
    sh = Shards(os.path.join(W, "transformer"))
    ccfg = dict(cfgj)
    H, HD = cfgj["num_attention_heads"], cfgj["attention_head_dim"]
    convcfg = {"num_attention_heads": H, "attention_head_dim": HD, "ffn_dim": cfgj["ffn_dim"]}
    small = dict(cfgj, num_layers=0)
    model = tm.MiniMaxH3ImageTransformer2DModel(**small).eval()
    # Glue weights (everything but the 50 blocks), from the original keys.
    glue = {}
    for k in list(sh.wm.keys()):
        if k.startswith("blocks."):
            continue
        for tk, t in conv.convert_transformer_key(k, conv.reorder_interleaved_qkv(sh.get(k), H, HD)
                                                  if k.endswith(".attn.qkv_proj.weight") else sh.get(k), ccfg):
            glue[tk] = t
    glue.update(conv.convert_pdd_heads(W, cfgj))
    missing, unexpected = model.load_state_dict(glue, strict=False)
    assert not [m for m in missing if not m.startswith("transformer_blocks")], missing
    model.to(dev)
    for n, p in model.named_parameters():  # keep the checkpoint's mixed precision
        if not n.startswith(("proj_in", "time_embedder", "pdd_proj_out")):
            p.data = p.data.to(dtype)
    block = tm.MiniMaxH3ImageTransformerBlock(cfgj["hidden_size"], H, HD, cfgj["ffn_dim"], cfgj["time_embed_dim"],
                                              cfgj["norm_eps"], cfgj["qk_norm_eps"]).to(dev, dtype).eval()
    sig = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
    t0 = torch.tensor([1.0 - s for s in sig[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - t0, torch.zeros(1)])
    timesteps = 1.0 - sigmas[:-1]
    T = emb.shape[0]
    tags = torch.ones(T, dtype=torch.long)
    pos, token_tags, img_idx, txt_idx, n_ref = build_image_packed_sequence(tags, [], lat_h, lat_w, (1, 2, 2), 0)
    lat = patchify_video_latents(noise.reshape(1, 24, 1, lat_h, lat_w), (1, 2, 2)).float()
    vels = []
    for s in range(6):
        ts = time.time()
        row_t = torch.full((pos.shape[0],), float(timesteps[s]), dtype=torch.float32)
        uniq, tidx = torch.unique(row_t, sorted=True, return_inverse=True)
        with torch.no_grad():
            rotary = model.rope(pos.float().to(dev))
            image_embeds = model.proj_in(lat[None].to(dev).float())
            text_embeds = model.context_embedder(emb[None].to(dev, dtype))
            text_embeds = model.token_refiner(text_embeds)
            x = text_embeds.new_zeros((1, pos.shape[0], text_embeds.shape[-1]))
            x = x.index_copy(1, txt_idx.to(dev), text_embeds)
            x = x.index_copy(1, img_idx.to(dev), image_embeds.to(text_embeds.dtype))
            temb = model.time_embedder(model.time_proj(uniq.to(dev)).float())
            adaln_idx = (tidx * 3 + token_tags).to(dev)
            for l in range(cfgj["num_layers"]):
                sd = {}
                for k in sh.keys_with(f"blocks.{l}."):
                    t = sh.get(k)
                    if k.endswith(".attn.qkv_proj.weight"):
                        t = conv.reorder_interleaved_qkv(t, H, HD)
                    for tk, tt in conv.convert_transformer_key(k, t, ccfg):
                        sd[tk.replace(f"transformer_blocks.{l}.", "")] = tt
                block.load_state_dict(sd)
                block.to(dev, dtype)
                x = block(x, temb, adaln_idx, rotary)
            head = model.pdd_proj_out[s]
            y = model.norm_out(x, temb, tidx.to(dev)).float()
            v = head(y).index_select(1, img_idx.to(dev))[0].float().cpu()
        vels.append(v.numpy())
        lat = (lat.float() + (sigmas[s] - sigmas[s + 1]) * v).float()
        log(f"  step {s + 1}/6 {time.time() - ts:.0f}s")
    rows = lat.reshape(lat_h // 2, lat_w // 2, 24, 2, 2).permute(2, 0, 3, 1, 4)
    return rows.reshape(24, lat_h, lat_w), vels


def vae_decode(W, lat):
    cfg = json.load(open(os.path.join(W, "vae_diffusers", "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    vae = AutoencoderKLMiniMaxH3Image(**cfg).float().eval()
    sd = {}
    with safe_open(os.path.join(W, "vae", "diffusion_pytorch_model.safetensors"), "pt") as f:
        for k in f.keys():
            for tk, t in conv.convert_video_vae_key(k, f.get_tensor(k), cfg):
                sd[tk] = t
    vae.load_state_dict(sd, strict=False)
    mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1)
    std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1)
    with torch.no_grad():
        img = vae.decode(lat[None] * std + mean, return_dict=False)[0]
    pm = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    ps = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    img = (img * ps + pm).clamp(0, 1)
    return (img * 255).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--min-area", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--te-only", action="store_true")
    ap.add_argument("--emb", help="reuse prompt embeddings from this npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, dtype = torch.device(a.device), {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    log = lambda *m: print(*m, flush=True)
    w, h = (int(x) for x in a.size.split("x"))
    ch, cw = get_image_canvas_size(h, w, a.min_area, 32)
    tok = AutoTokenizer.from_pretrained(os.path.join(a.weights, "tokenizer"))
    ids = tok(a.prompt, add_special_tokens=False)["input_ids"][:4096]
    res = {"ids": np.array(ids), "canvas": np.array([ch, cw])}
    t = time.time()
    if a.emb:
        emb = torch.from_numpy(np.load(a.emb)["emb"]).to(torch.bfloat16)
    else:
        emb = text_encoder(a.weights, ids, dev, dtype).cpu()
        log(f"text encoder {time.time() - t:.0f}s")
    res["emb"] = emb.float().numpy()
    if not a.te_only:
        g = torch.Generator("cpu").manual_seed(a.seed)
        noise = torch.randn((1, 24, 1, ch // 16, cw // 16), generator=g, dtype=torch.float32)
        res["noise"] = noise.numpy()
        lat, vels = dit_denoise(a.weights, emb, ch // 16, cw // 16, noise, dev, dtype, log)
        res["latents"] = lat.numpy()
        for i, v in enumerate(vels):
            res[f"vel{i}"] = v
        res["image"] = vae_decode(a.weights, lat.float())
    np.savez(a.out, **res)
    log(f"saved {a.out} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
