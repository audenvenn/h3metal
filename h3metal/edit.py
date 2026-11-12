"""The parts only an edit needs: the Qwen3-VL vision tower (with its DeepStack
features), which runs once per reference image on MLX, and the multimodal
positions of the text encoder's input. The VAE encoder is in vae.py.
"""

from __future__ import annotations

import math

import numpy as np

PATCH, TPATCH, MERGE = 16, 2, 2
VIS_MEAN = VIS_STD = 0.5


# -- sizes and resizing (diffusers MiniMaxH3ImageEditResizeStep) -------------------------------------------------
def bound_short_edge(width: int, height: int, lo: int, hi: int, multiple: int = 32):
    short = min(width, height)
    if lo <= short <= hi:
        return width, height
    scale = (lo if short < lo else hi) / short
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def resize_image(img, lo: int, hi: int):
    """PIL image -> PIL image resized for one consumer (LANCZOS, like VaeImageProcessor.resize)."""
    from PIL import Image
    size = bound_short_edge(img.width, img.height, lo, hi)
    return img if img.size == size else img.resize(size, Image.LANCZOS)


# -- vision tower -------------------------------------------------------------------------------------------------
def vision_patches(img: np.ndarray):
    """uint8 [H][W][3] (sides multiples of 32) -> (patches [N][3*2*16*16], grid (1, h, w)), in the processor's
    spatial-merge block order."""
    H, W = img.shape[:2]
    x = (img.astype(np.float32) / 255.0 - VIS_MEAN) / VIS_STD          # [H][W][C]
    x = x.transpose(2, 0, 1)                                          # [C][H][W]
    x = np.stack([x, x], 0)                                           # temporal patch: the frame twice
    gh, gw = H // PATCH, W // PATCH
    x = x.reshape(TPATCH, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)                            # bh, bw, mh, mw, C, T, ph, pw
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * TPATCH * PATCH * PATCH)), (1, gh, gw)


def _axis_taps(index, size, side):
    src = index.astype(np.float32) * np.float32(side - 1) / np.float32(max(size - 1, 1))
    floor = np.floor(src)
    taps = np.clip(floor[:, None].astype(np.int64) + np.arange(2)[None], 0, side - 1)
    dist = np.abs(src[:, None] - floor[:, None] - np.arange(2)[None])
    return taps, np.clip(1 - dist, 0, None).astype(np.float32)


def pos_interp(gh: int, gw: int, side: int):
    """Bilinear (align_corners) resampling of the square learned position table to a gh x gw grid, in merge-block
    order -> (indices [N][4], weights [N][4])."""
    n = np.arange(gh * gw)
    bw = gw // MERGE
    in_col, in_row = n % MERGE, (n // MERGE) % MERGE
    bcol, brow = (n // (MERGE * MERGE)) % bw, n // (MERGE * MERGE * bw)
    row, col = brow * MERGE + in_row, bcol * MERGE + in_col
    ht, hw = _axis_taps(row, gh, side)
    wt, ww = _axis_taps(col, gw, side)
    idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
    wts = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
    return idx, wts


def vision_rope(gh: int, gw: int, head_dim: int, theta: float = 10000.0):
    hpos, wpos = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    shape = (gh // MERGE, MERGE, gw // MERGE, MERGE)
    hpos = hpos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    wpos = wpos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    spatial = head_dim // 2
    inv = (1.0 / theta ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial)).astype(np.float32)
    fh = hpos[:, None].astype(np.float32) * inv[None]
    fw = wpos[:, None].astype(np.float32) * inv[None]
    f = np.concatenate([fh, fw], -1)
    f = np.concatenate([f, f], -1)
    return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)


class VisionTower:
    """Qwen3-VL vision encoder. weights: name -> float array (the checkpoint's model.visual.* without prefix)."""

    def __init__(self, weights: dict, cfg: dict):
        import mlx.core as mx
        self.mx = mx
        self.cfg = cfg
        self.depth = cfg["depth"]
        self.heads = cfg["num_heads"]
        self.dim = cfg["hidden_size"]
        self.hd = self.dim // self.heads
        self.ds_index = cfg["deepstack_visual_indexes"]
        self.side = int(math.isqrt(cfg["num_position_embeddings"]))
        dt = mx.float16
        f32 = mx.float32
        w = {k: mx.array(v) for k, v in weights.items()}
        pe = w["patch_embed.proj.weight"]
        self.patch_w = pe.reshape(pe.shape[0], -1).T.astype(dt)
        self.patch_b = w["patch_embed.proj.bias"].astype(f32)
        self.pos_embed = w["pos_embed.weight"].astype(f32)
        self.blocks = []
        for i in range(self.depth):
            p = f"blocks.{i}."
            self.blocks.append({
                "n1w": w[p + "norm1.weight"].astype(f32), "n1b": w[p + "norm1.bias"].astype(f32),
                "n2w": w[p + "norm2.weight"].astype(f32), "n2b": w[p + "norm2.bias"].astype(f32),
                "qkv_w": w[p + "attn.qkv.weight"].T.astype(dt), "qkv_b": w[p + "attn.qkv.bias"].astype(f32),
                "proj_w": w[p + "attn.proj.weight"].T.astype(dt), "proj_b": w[p + "attn.proj.bias"].astype(f32),
                "fc1_w": w[p + "mlp.linear_fc1.weight"].T.astype(dt), "fc1_b": w[p + "mlp.linear_fc1.bias"].astype(f32),
                "fc2_w": w[p + "mlp.linear_fc2.weight"].T.astype(dt), "fc2_b": w[p + "mlp.linear_fc2.bias"].astype(f32),
            })
        def merger(p):
            return {"nw": w[p + "norm.weight"].astype(f32), "nb": w[p + "norm.bias"].astype(f32),
                    "fc1_w": w[p + "linear_fc1.weight"].T.astype(dt), "fc1_b": w[p + "linear_fc1.bias"].astype(f32),
                    "fc2_w": w[p + "linear_fc2.weight"].T.astype(dt), "fc2_b": w[p + "linear_fc2.bias"].astype(f32)}
        self.merger = merger("merger.")
        self.ds = [merger(f"deepstack_merger_list.{i}.") for i in range(len(self.ds_index))]
        mx.eval(self.__dict__.get("patch_w"))

    def _lin(self, x, w, b):
        mx = self.mx
        return (x.astype(mx.float16) @ w).astype(mx.float32) + b

    def _merge(self, x, m, post: bool):
        mx = self.mx
        n = x.shape[0] // (MERGE * MERGE)
        if post:
            x = mx.fast.layer_norm(x.reshape(n, -1), m["nw"], m["nb"], 1e-6)
        else:
            x = mx.fast.layer_norm(x, m["nw"], m["nb"], 1e-6).reshape(n, -1)
        x = self._lin(x, m["fc1_w"], m["fc1_b"])
        x = x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))  # exact GELU
        return self._lin(x, m["fc2_w"], m["fc2_b"])

    def __call__(self, patches: np.ndarray, grid):
        """patches [N][1536] (one image) -> (embeds [N/4][out], [deepstack features])."""
        mx = self.mx
        _, gh, gw = grid
        x = mx.array(patches)
        x = (x.astype(mx.float16) @ self.patch_w).astype(mx.float32) + self.patch_b
        idx, wts = pos_interp(gh, gw, self.side)
        pe = (self.pos_embed[mx.array(idx.reshape(-1))].reshape(-1, 4, self.dim) * mx.array(wts)[:, :, None]).sum(1)
        x = x + pe
        cos, sin = (mx.array(a) for a in vision_rope(gh, gw, self.hd))
        N = x.shape[0]
        H, hd = self.heads, self.hd
        feats = []
        for i, b in enumerate(self.blocks):
            h = mx.fast.layer_norm(x, b["n1w"], b["n1b"], 1e-6)
            qkv = self._lin(h, b["qkv_w"], b["qkv_b"]).reshape(N, 3, H, hd)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

            def rope(t):
                t1, t2 = t[..., : hd // 2], t[..., hd // 2:]
                rot = mx.concatenate([-t2, t1], -1)
                return t * cos[:, None, :] + rot * sin[:, None, :]
            q, k = rope(q), rope(k)
            q, k, v = (t.transpose(1, 0, 2)[None].astype(mx.float16) for t in (q, k, v))
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
            o = o[0].transpose(1, 0, 2).reshape(N, -1).astype(mx.float32)
            x = x + self._lin(o, b["proj_w"], b["proj_b"])
            h = mx.fast.layer_norm(x, b["n2w"], b["n2b"], 1e-6)
            h = self._lin(h, b["fc1_w"], b["fc1_b"])
            h = 0.5 * h * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h * h * h)))  # gelu tanh
            x = x + self._lin(h, b["fc2_w"], b["fc2_b"])
            if i in self.ds_index:
                feats.append(self._merge(x, self.ds[self.ds_index.index(i)], post=True))
            mx.eval(x)
        out = self._merge(x, self.merger, post=False)
        mx.eval(out, *feats)
        return np.array(out), [np.array(f) for f in feats]


# -- text encoder positions for an edit (Qwen3-VL get_rope_index) ----------------------------------------------
def edit_positions(tags_image: np.ndarray, grids) -> np.ndarray:
    """mm token types (1 = image pad, else 0) per token and the patch grids -> [3][T] positions."""
    out = []
    cur = 0
    T = len(tags_image)
    i = 0
    gi = iter(grids)
    while i < T:
        j = i
        while j < T and tags_image[j] == tags_image[i]:
            j += 1
        if tags_image[i] == 0:
            n = j - i
            out.append(np.tile(np.arange(n) + cur, (3, 1)))
            cur += n
        else:
            t, h, w = next(gi)
            lh, lw = h // MERGE, w // MERGE
            tt, hh, ww = np.meshgrid(np.arange(t), np.arange(lh) + cur, np.arange(lw) + cur, indexing="ij")
            p = np.stack([tt, hh, ww]).reshape(3, -1)
            p[0] += cur
            out.append(p)
            cur += max(h, w) // MERGE
        i = j
    return np.concatenate(out, 1)
