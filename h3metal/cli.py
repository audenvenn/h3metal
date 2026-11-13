"""h3metal command line: generate, edit, convert, info."""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_MODEL = os.environ.get("H3METAL_MODEL", os.path.expanduser("~/.h3metal/model"))
ZIMAGE_MODEL = os.path.expanduser("~/.h3metal/zimage")  # converted Z-Image-Turbo, by default
MODEL_NAMES = {"zimage": ZIMAGE_MODEL, "z-image": ZIMAGE_MODEL, "z-image-turbo": ZIMAGE_MODEL,
               "minimax-h3": DEFAULT_MODEL, "h3": DEFAULT_MODEL}


def _model_dir(name: str) -> str:
    """--model: a converted model's directory, or one of the names in MODEL_NAMES."""
    return MODEL_NAMES.get(name.lower(), os.path.expanduser(name))


AREAS = {"4k": 4096 * 4096, "2k": 2048 * 2048, "1k": 1024 * 1024}


def size_for(ratio: str, area: int = AREAS["4k"]):
    """The official sizing: about `area` pixels at the ratio, both sides multiples of 32 (16:9 -> 5472x3072)."""
    import math
    x, y = (int(v) for v in ratio.split(":"))
    return round(math.sqrt(area * x / y) / 32) * 32, round(math.sqrt(area * y / x) / 32) * 32


def _size(s: str):
    """WIDTHxHEIGHT, or a ratio like 16:9 (the official ~16 MP size), optionally @2k or @1k for a smaller area."""
    s = s.lower()
    try:
        if ":" in s:
            ratio, _, area = s.partition("@")
            w, h = size_for(ratio, AREAS[area or "4k"])
        else:
            w, h = (int(x) for x in s.split("x"))
    except (ValueError, KeyError):
        raise argparse.ArgumentTypeError("size must look like 2048x2048, 16:9 or 16:9@2k")
    if w % 16 or h % 16 or not (256 <= w <= 8192 and 256 <= h <= 8192):
        raise argparse.ArgumentTypeError("width and height must be multiples of 16 (32 for MiniMax-H3-Image) "
                                         "between 256 and 8192")
    return w, h


def _open_image(path: str):
    """A reference image; formats Pillow cannot read (HEIC photos, for instance) go through macOS's sips."""
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "ref.png")
            r = subprocess.run(["sips", "-s", "format", "png", path, "--out", png], capture_output=True)
            if r.returncode or not os.path.exists(png):
                raise ValueError(f"cannot read the image {path}")
            im = Image.open(png)
            im.load()
            return im


def _save(img, path: str, meta: dict):
    """Saves the image; PNG files carry the generation parameters and say that the image is AI-generated."""
    from PIL import Image, PngImagePlugin
    im = Image.fromarray(img)
    if path.lower().endswith(".png"):
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "h3metal")
        info.add_text("Description", f"AI-generated image ({meta.get('model', 'MiniMax-H3-Image')})")
        for k, v in meta.items():
            info.add_text(k, str(v))
        im.save(path, pnginfo=info)
    else:
        im.save(path)


def _progress(step, total):
    sys.stderr.write(f"\r  denoising {step}/{total}")
    sys.stderr.flush()
    if step == total:
        sys.stderr.write("\n")


def cmd_generate(a):
    from .pipeline import Pipeline, generate_edit
    from .zimage import ZImagePipeline, is_zimage_dir
    t0 = time.time()
    a.model = _model_dir(a.model)
    if is_zimage_dir(a.model):
        if a.image:
            raise ValueError("Z-Image-Turbo makes images from text only; --image needs MiniMax-H3-Image")
        pipe = ZImagePipeline(a.model)
    else:
        pipe = Pipeline(a.model)
    model_name = getattr(pipe, "model_name", "MiniMax-H3-Image")
    _check_device(pipe.gpu.device)
    w, h = a.size or getattr(pipe, "default_size", (2048, 2048))
    if model_name == "MiniMax-H3-Image" and (w % 32 or h % 32):
        raise ValueError("MiniMax-H3-Image needs width and height that are multiples of 32")
    for i in range(a.count):
        seed = a.seed + i
        t1 = time.time()
        if a.image:
            refs = [_open_image(p) for p in a.image]
            img, timing = generate_edit(pipe, a.prompt, refs, w, h, seed=seed, min_area=a.min_area,
                                        progress=_progress)
        else:
            img, timing = pipe.generate(a.prompt, w, h, seed=seed, min_area=a.min_area, progress=_progress)
        out = a.output or f"h3metal_{int(time.time())}.png"
        if a.count > 1:
            stem, ext = os.path.splitext(out)
            out = f"{stem}_{seed}{ext or '.png'}"
        _save(img, out, {"prompt": a.prompt, "seed": seed, "size": f"{w}x{h}", "model": model_name,
                         "references": ", ".join(os.path.basename(p) for p in a.image or []) or "none"})
        print(f"{out}: {w}x{h}, seed {seed}; {timing}; wall {time.time() - (t0 if i == 0 else t1):.1f}s")


FORMAT_ALIASES = {"mix": "q8,q8,q4,q4"}  # int8 attention, int4 MLP


def auto_format(memory_bytes: int | None = None) -> str:
    """The transformer format for this Mac: one that stays resident in its memory (int8 needs ~32 GB; int8
    attention with int4 MLP fits 24 GB; int4 everywhere 16 GB). Streaming int8 from the SSD keeps up at
    2048 x 2048 but not at smaller sizes, where a step is too short to hide it."""
    if memory_bytes is None:
        import subprocess
        memory_bytes = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    gb = memory_bytes / 2**30
    return "q8" if gb > 24.5 else "mix" if gb > 16.5 else "q4"


def cmd_convert(a):
    from .convert import convert_dit
    from .zimage import convert_zimage, is_zimage_checkpoint
    if is_zimage_checkpoint(a.src):
        # Z-Image's 6 B transformer is 6.2 GB in int8, which fits every Mac h3metal runs on.
        dit = "q8" if a.dit == "auto" else FORMAT_ALIASES.get(a.dit, a.dit)
        out = _model_dir(a.out) if a.out else ZIMAGE_MODEL  # never MiniMax-H3's directory by default
        convert_zimage(a.src, out, dit=dit, te=a.te, parts=a.only.split(",") if a.only else None)
        print(f"next: h3metal generate --model {'zimage' if out == ZIMAGE_MODEL else out} -p \"...\"")
        return
    a.out = _model_dir(a.out) if a.out else DEFAULT_MODEL
    if a.dit == "auto":
        a.dit = auto_format()
        print(f"transformer format for this Mac: {a.dit} (choose another with --dit)")
    a.dit = FORMAT_ALIASES.get(a.dit, a.dit)
    from .te import convert_te, convert_vision
    from .vae import convert_vae
    os.makedirs(a.out, exist_ok=True)
    src = a.src
    parts = a.only.split(",") if a.only else ["vae", "vision", "te", "dit", "tokenizer"]
    if "tokenizer" in parts:
        import shutil
        shutil.copy(os.path.join(src, "tokenizer", "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
    if "vae" in parts:
        convert_vae(src, os.path.join(a.out, "vae.h3m"), fmt=a.vae)
    if "vision" in parts:
        convert_vision(os.path.join(src, "text_encoder"), os.path.join(a.out, "vision.h3m"))
    if "te" in parts:
        convert_te(os.path.join(src, "text_encoder"), os.path.join(a.out, "te.h3m"), fmt=a.te)
    if "dit" in parts:
        fmts = dict(zip(("qkv", "out", "fc1", "fc2"), a.dit.split(","))) if "," in a.dit else \
            {k: a.dit for k in ("qkv", "out", "fc1", "fc2")}
        convert_dit(src, os.path.join(a.out, "dit.h3m"), fmts=fmts)
    if a.gptq:
        from .calib import calibrate_model
        fmts = dict(zip(("qkv", "out", "fc1", "fc2"), a.dit.split(","))) if "," in a.dit else a.dit
        calibrate_model(src, a.out, fmts)


def cmd_download(a):
    from .download import download
    if a.dest is None:
        a.dest = os.path.expanduser("~/.h3metal/checkpoint" if a.model == "minimax-h3" else f"~/.h3metal/{a.model}")
    if not download(a.dest, a.source, a.jobs, model=a.model):
        sys.exit("some files did not download; run the command again to resume")
    print(f"next: h3metal convert {a.dest}")


def _check_device(device: str):
    from .gpu import has_matrix_units
    if not has_matrix_units(device):
        print(f"warning: h3metal is built for the GPU matrix units of M5- and M6-class chips; on {device} its "
              "kernels may fail to compile or run far slower.", file=sys.stderr)


def cmd_info(a):
    from .gpu import Gpu
    g = Gpu()
    print(f"device: {g.device}; recommended working set {g.working_set / 1e9:.1f} GB; "
          f"max buffer {g.max_buffer / 1e9:.1f} GB")
    _check_device(g.device)
    from .zimage import is_zimage_dir
    a.model = _model_dir(a.model)
    print(f"model: {'Z-Image-Turbo' if is_zimage_dir(a.model) else 'MiniMax-H3-Image'} ({a.model})")
    for k in ("te", "dit", "vae") + (() if is_zimage_dir(a.model) else ("vision",)):
        p = os.path.join(a.model, f"{k}.h3m")
        if os.path.exists(p):
            from .weights import read_header
            m = read_header(p)["meta"]
            print(f"{k:7s} {os.path.getsize(p) / 1e9:6.2f} GB  {m.get('fmts') or m.get('fmt', '')}")
        else:
            print(f"{k:7s} missing ({p})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h3metal", description="MiniMax-H3-Image (and Z-Image-Turbo) on Apple "
                                 "Silicon.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="text to image, or an edit with --image")
    g.add_argument("-p", "--prompt", required=True)
    g.add_argument("-s", "--size", type=_size, default=None,
                   help="WIDTHxHEIGHT, or an aspect ratio such as 16:9 for the official ~16 MP size "
                        "(16:9@2k for ~4 MP); default 2048x2048 (1024x1024 for Z-Image-Turbo)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-n", "--count", type=int, default=1, help="images to make, with seeds seed, seed+1, ...")
    g.add_argument("-o", "--output")
    g.add_argument("-i", "--image", action="append", help="reference image (repeat for several); makes it an edit")
    g.add_argument("--min-area", type=int, default=2048 * 2048,
                   help="smallest canvas the image is generated on (default 2048*2048, as released); "
                        "1024 generates at the requested size")
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help="a converted model's directory, or zimage / minimax-h3 for the default ones")
    g.set_defaults(func=cmd_generate)
    c = sub.add_parser("convert", help="convert the official weights into h3metal's format")
    c.add_argument("src", help="the MiniMax-H3-Image directory (transformer/, text_encoder/, vae/, ...), or a "
                   "Z-Image-Turbo one")
    c.add_argument("--out", default=None, help="default ~/.h3metal/model (~/.h3metal/zimage for Z-Image-Turbo)")
    c.add_argument("--dit", default="auto", help="auto (by this Mac's memory), q8, mix (int8 attention, int4 MLP), "
                   "q4, q4g, or one per matrix: qkv,out,fc1,fc2 (e.g. q8,q8,q4,q4)")
    c.add_argument("--te", default="q8", help="text encoder weights: q8, q4g or f16")
    c.add_argument("--vae", default="f16")
    c.add_argument("--only", help="comma-separated subset of vae,vision,te,dit,tokenizer")
    c.add_argument("--gptq", action="store_true",
                   help="choose the transformer's codes with GPTQ on calibration runs (about an hour on an M6)")
    c.set_defaults(func=cmd_convert)
    d = sub.add_parser("download", help="fetch the official weights h3metal converts from (about 125 GB; "
                       "33 GB for Z-Image-Turbo)")
    d.add_argument("dest", nargs="?", default=None,
                   help="default ~/.h3metal/checkpoint (~/.h3metal/z-image-turbo with --model z-image-turbo)")
    d.add_argument("--model", choices=["minimax-h3", "z-image-turbo"], default="minimax-h3")
    d.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    d.add_argument("-j", "--jobs", type=int, default=4)
    d.set_defaults(func=cmd_download)
    i = sub.add_parser("info", help="device and model files")
    i.add_argument("--model", default=DEFAULT_MODEL)
    i.set_defaults(func=cmd_info)
    a = ap.parse_args(argv)
    try:
        a.func(a)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # a one-line message; H3METAL_DEBUG=1 shows the traceback
        if os.environ.get("H3METAL_DEBUG"):
            raise
        sys.exit(f"h3metal: error: {e}")


if __name__ == "__main__":
    main()
