"""Inputs of an edit: reference images resized and cut into the vision tower's
patches, the tower's position tables, and the multimodal positions of the
text encoder's input. The vision tower itself is in vision.py, the VAE
encoder in vae.py.
"""

from __future__ import annotations


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
