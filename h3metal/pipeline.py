"""Text-to-image (and, with h3metal.edit, image editing) end to end.

The three stages run one after another and each maps only its own weight
file, so the text encoder, the transformer and the VAE are never resident at
the same time; on a 24 GB Mac the 33 B transformer gets the memory to itself.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import dit as ditm
from .gpu import Gpu
from .weights import WeightFile, bf16_bits, bf16_to_f32

MAX_TOKENS = 4096
DEFAULT_MIN_AREA = 2048 * 2048


@dataclass
class Timing:
    stages: dict = field(default_factory=dict)

    def add(self, name: str, seconds: float):
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def total(self) -> float:
        return sum(self.stages.values())

    def __str__(self):
        return ", ".join(f"{k} {v:.1f}s" for k, v in self.stages.items()) + f" (total {self.total():.1f}s)"


def noise(seed: int, shape) -> np.ndarray:
    """Standard normal noise: the draws of the official pipeline's torch CPU generator for this seed."""
    from .rng import TorchRandn
    return TorchRandn(seed).normal(shape)


class Pipeline:
    def __init__(self, model_dir: str, gpu: Gpu | None = None, log=print):
        model_dir = os.path.expanduser(model_dir)
        self.dir = model_dir
        self.gpu = gpu or Gpu()
        self.log = log
        self.paths = {k: os.path.join(model_dir, f"{k}.h3m") for k in ("te", "dit", "vae", "vision")}
        for k in ("te", "dit", "vae"):
            if not os.path.exists(self.paths[k]):
                raise FileNotFoundError(f"missing {self.paths[k]}: `h3metal download --model flux2-klein-4b` (or "
                                        "z-image-turbo) fetches a model ready to run; MiniMax-H3-Image takes "
                                        "`h3metal download` and `h3metal convert`; or point --model at a "
                                        "converted model")
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._te_cache: dict[str, np.ndarray] = {}

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids[:MAX_TOKENS]

    def _cache_path(self, key: str) -> str | None:
        root = os.environ.get("H3METAL_CACHE", os.path.expanduser("~/.h3metal/cache"))
        if root in ("", "0", "off"):
            return None
        import hashlib
        st = os.stat(self.paths["te"])
        h = hashlib.sha256(f"{key}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:32]
        return os.path.join(root, f"{h}.npy")

    def encode_prompt(self, prompt: str, timing: Timing) -> np.ndarray:
        """The text encoder's output for a prompt, from memory or the on-disk cache when it was seen before."""
        if prompt in self._te_cache:
            return self._te_cache[prompt]
        path = self._cache_path("t2i|" + prompt)
        if path and os.path.exists(path):
            emb = bf16_to_f32(np.load(path))
            self._te_cache = {prompt: emb}
            timing.add("text (cached)", 0.0)
            return emb
        from .te import TextEncoder
        self.release()
        t = time.time()
        wf = WeightFile(self.gpu, self.paths["te"])
        te = TextEncoder(self.gpu, wf)
        emb = te.encode(self.tokenize(prompt))
        del te
        wf.close()
        timing.add("text", time.time() - t)
        self._te_cache = {prompt: emb}
        if path:
            _store(path, bf16_bits(emb))
        return emb

    def _dit(self, timing: Timing):
        """The transformer, mapped once and kept for the following requests."""
        if getattr(self, "_model", None) is None:
            t = time.time()
            self._dit_wf = WeightFile(self.gpu, self.paths["dit"])
            self._model = ditm.DiT(self.gpu, self._dit_wf)
            timing.add("load", time.time() - t)
        return self._model

    def release(self):
        """Unmaps the transformer (before the text encoder needs the memory)."""
        if getattr(self, "_model", None) is not None:
            self._model.free()
            self._model = None
            self._dit_wf.close()
            self._dit_wf = None

    def denoise(self, emb: np.ndarray, lat_h: int, lat_w: int, seed: int, timing: Timing,
                refs=None, ref_rows=None, text_types=None, latents: np.ndarray | None = None,
                progress=None) -> np.ndarray:
        g = self.gpu
        model = self._dit(timing)
        t = time.time()
        T = emb.shape[0]
        key = (emb.shape, float(emb[0, :8].sum()), float(emb[-1, :8].sum()), refs is None)
        if getattr(self, "_rows_key", None) != key or text_types is not None:
            text = g.empty((ditm.ops.pad_rows(T, 128), emb.shape[1]), "bf16")
            text.numpy()[:T] = bf16_bits(emb)
            self._rows = model.encode_text(text, T)
            self._rows_key = key
        rows = self._rows
        refs = refs or []
        tt = np.ones(T, np.uint8) if text_types is None else text_types
        lay = ditm.Layout(T, tt, refs, lat_h, lat_w)
        model.prepare(lay)
        C = model.cfg["latents"]
        if latents is None:
            latents = noise(seed, (1, C, 1, lat_h, lat_w)).reshape(C, lat_h, lat_w)
        x = ditm.patchify(latents.astype(np.float32))
        n_ref = lay.n_ref_rows
        cond = ref_rows if ref_rows is not None else np.zeros((0, x.shape[1]), np.float32)
        timing.add("text rows", time.time() - t)
        for s in range(model.n_steps):
            t = time.time()
            v = model.step(s, rows, np.concatenate([cond, x]) if n_ref else x)
            x = ditm.euler(x, v[n_ref:], s, model.sigmas)
            timing.add("denoise", time.time() - t)
            if progress:
                progress(s + 1, model.n_steps)
        return ditm.unpatchify(x, C, lat_h, lat_w)

    def decode(self, lat: np.ndarray, timing: Timing) -> np.ndarray:
        from .vae import VAEDecoder
        t = time.time()
        wf = WeightFile(self.gpu, self.paths["vae"])
        dec = VAEDecoder(self.gpu, wf)
        img = dec.decode(lat)
        wf.close()
        timing.add("decode", time.time() - t)
        return img

    def edit(self, prompt: str, images, width: int = 2048, height: int = 2048, seed: int = 0,
             min_area: int = DEFAULT_MIN_AREA, progress=None):
        """Edit with reference images (PIL images), named <Picture 1>, <Picture 2>, ... in the prompt.
        -> (uint8 image [H][W][3], Timing)."""
        return generate_edit(self, prompt, images, width, height, seed, min_area, progress)

    def generate(self, prompt: str, width: int = 2048, height: int = 2048, seed: int = 0,
                 min_area: int = DEFAULT_MIN_AREA, progress=None, latents: np.ndarray | None = None):
        """-> (uint8 image [H][W][3], Timing). The prompt's embeddings and the transformer are kept for the next
        call, so further images of the same prompt cost the denoising and the decode only."""
        timing = Timing()
        ch, cw = ditm.canvas_size(height, width, min_area)
        emb = self.encode_prompt(prompt, timing)
        lat = self.denoise(emb, ch // 16, cw // 16, seed, timing, latents=latents, progress=progress)
        img = self.decode(lat, timing)
        out = to_image(img, width, height)
        return out, timing


def _store(path: str, arr: np.ndarray, keep: int = 32):
    """Writes a cache entry and keeps only the `keep` most recent ones."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
    entries = sorted((os.path.getmtime(os.path.join(d, f)), f) for f in os.listdir(d) if f.endswith(".npy"))
    for _, f in entries[:-keep]:
        os.remove(os.path.join(d, f))


def to_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """[3][H][W] in [0, 1] -> uint8 [height][width][3]: truncated to 8 bits, then LANCZOS-resized if needed."""
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    if u8.shape[0] != height or u8.shape[1] != width:
        from PIL import Image
        u8 = np.asarray(Image.fromarray(u8).resize((width, height), Image.LANCZOS))
    return u8


IMAGE_PAD, VISION_START, VISION_END = "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"
REF_T = 0.999


class _Draws:
    """The request's random draws, in the order the released pipeline takes them from its generator."""

    def __init__(self, seed: int):
        from .rng import TorchRandn
        self.g = TorchRandn(seed)

    def normal(self, shape) -> np.ndarray:
        return self.g.normal(shape)


def edit_inputs(pipe: "Pipeline", prompt: str, images, draws: _Draws):
    """References -> (token ids, row types, positions, vision rows, features, deepstack, reference latents)."""
    from PIL import ImageOps
    from . import edit
    from .weights import WeightFile
    pipe.release()
    imgs = [ImageOps.exif_transpose(im).convert("RGB") for im in images]
    for im in imgs:
        if min(im.size) < 32:
            raise ValueError(f"a reference image needs at least 32 pixels on each side, got {im.size}")
    vae_imgs = [edit.resize_image(im, 1024, 2048) for im in imgs]
    vis_imgs = [edit.resize_image(im, 1024, 1024) for im in imgs]
    from .vision import VisionTower
    wf = WeightFile(pipe.gpu, pipe.paths["vision"])
    vt = VisionTower(pipe.gpu, wf)
    feats, deep, grids = [], [[], [], []], []
    for im in vis_imgs:
        p, grid = edit.vision_patches(np.asarray(im))
        e, ds = vt(p, grid)
        feats.append(e)
        for i, d in enumerate(ds):
            deep[i].append(d)
        grids.append(grid)
    del vt
    wf.close()
    tok = pipe.tokenizer
    pad, vs, ve = (tok.token_to_id(t) for t in (IMAGE_PAD, VISION_START, VISION_END))
    ids, types = [], []
    for i, grid in enumerate(grids):
        label = tok.encode(f"Picture {i + 1}: ", add_special_tokens=False).ids[:MAX_TOKENS]
        n = grid[0] * grid[1] * grid[2] // 4
        block = [vs] + [pad] * n + [ve]
        ids += label + block
        types += [1] * len(label) + [0] * len(block)
    pids = tok.encode(prompt, add_special_tokens=False).ids[:MAX_TOKENS]
    ids += pids
    types += [1] * len(pids)
    ids = np.array(ids)
    mm = (ids == pad).astype(np.int32)
    pos3 = edit.edit_positions(mm, grids)
    rows = np.nonzero(mm)[0]
    # VAE posteriors: one sample per reference, first draws of the generator.
    from .vae import VAEEncoder
    wf = WeightFile(pipe.gpu, pipe.paths["vae"])
    enc = VAEEncoder(pipe.gpu, wf)
    lat_mean = np.array(wf.meta["latents_mean"], np.float32)[:, None, None]
    lat_std = np.array(wf.meta["latents_std"], np.float32)[:, None, None]
    ref_lats = []
    for im in vae_imgs:
        a = np.asarray(im)
        a = a[: a.shape[0] // 16 * 16, : a.shape[1] // 16 * 16]
        mean, std = enc.encode(a)
        z = (mean + std * draws.normal((1,) + mean.shape)[0]).astype(np.float16).astype(np.float32)
        z = z[:, : z.shape[1] // 2 * 2, : z.shape[2] // 2 * 2]
        ref_lats.append((z - lat_mean) / lat_std)
    del enc
    wf.close()
    return ids, np.array(types, np.uint8), pos3, rows, np.concatenate(feats), \
        [np.concatenate(d) for d in deep], ref_lats


def generate_edit(pipe: "Pipeline", prompt: str, images, width: int, height: int, seed: int = 0,
                  min_area: int = DEFAULT_MIN_AREA, progress=None):
    """Edit: -> (uint8 image, Timing). The prompt refers to the references as <Picture 1>, <Picture 2>, ..."""
    from .te import TextEncoder
    timing = Timing()
    draws = _Draws(seed)
    t = time.time()
    ids, types, pos3, rows, feats, deep, ref_lats = edit_inputs(pipe, prompt, images, draws)
    timing.add("references", time.time() - t)
    t = time.time()
    pipe.release()
    wf = WeightFile(pipe.gpu, pipe.paths["te"])
    emb = TextEncoder(pipe.gpu, wf).encode(ids, pos3=pos3, vision=feats, vision_rows=rows, deepstack=deep)
    wf.close()
    timing.add("text", time.time() - t)
    ch, cw = ditm.canvas_size(height, width, min_area)
    lat_h, lat_w = ch // 16, cw // 16
    C = ref_lats[0].shape[0]
    target = draws.normal((1, C, 1, lat_h, lat_w)).reshape(C, lat_h, lat_w)
    cond = []
    for z in ref_lats:
        n = draws.normal((1,) + z.shape)[0]
        tt = np.float32(REF_T)
        cond.append(ditm.patchify((tt * z + (np.float32(1) - tt) * n).astype(np.float32)))
    refs = [(z.shape[1], z.shape[2]) for z in ref_lats]
    lat = pipe.denoise(emb, lat_h, lat_w, seed, timing, refs=refs, ref_rows=np.concatenate(cond),
                       text_types=types, latents=target, progress=progress)
    img = pipe.decode(lat, timing)
    return to_image(img, width, height), timing
