"""h3metal command line: generate, edit, convert, info."""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_MODEL = os.environ.get("H3METAL_MODEL", os.path.expanduser("~/.h3metal/model"))


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
    if w % 32 or h % 32 or not (256 <= w <= 8192 and 256 <= h <= 8192):
        raise argparse.ArgumentTypeError("width and height must be multiples of 32 between 256 and 8192")
    return w, h


def _progress(step, total):
    sys.stderr.write(f"\r  denoising {step}/{total}")
    sys.stderr.flush()
    if step == total:
        sys.stderr.write("\n")


def cmd_generate(a):
    from PIL import Image
    from .pipeline import Pipeline, generate_edit
    t0 = time.time()
    pipe = Pipeline(a.model)
    _check_device(pipe.gpu.device)
    w, h = a.size
    for i in range(a.count):
        seed = a.seed + i
        t1 = time.time()
        if a.image:
            refs = [Image.open(p) for p in a.image]
            img, timing = generate_edit(pipe, a.prompt, refs, w, h, seed=seed, min_area=a.min_area,
                                        progress=_progress)
        else:
            img, timing = pipe.generate(a.prompt, w, h, seed=seed, min_area=a.min_area, progress=_progress)
        out = a.output or f"h3metal_{int(time.time())}.png"
        if a.count > 1:
            stem, ext = os.path.splitext(out)
            out = f"{stem}_{seed}{ext or '.png'}"
        Image.fromarray(img).save(out)
        print(f"{out}: {w}x{h}, seed {seed}; {timing}; wall {time.time() - (t0 if i == 0 else t1):.1f}s")


FORMAT_ALIASES = {"mix": "q8,q8,q4,q4"}  # int8 attention, int4 MLP


def cmd_convert(a):
    from .convert import convert_dit
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
    if not download(a.dest, a.source, a.jobs):
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
    for k in ("te", "dit", "vae", "vision"):
        p = os.path.join(a.model, f"{k}.h3m")
        if os.path.exists(p):
            from .weights import read_header
            m = read_header(p)["meta"]
            print(f"{k:7s} {os.path.getsize(p) / 1e9:6.2f} GB  {m.get('fmts') or m.get('fmt', '')}")
        else:
            print(f"{k:7s} missing ({p})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h3metal", description="MiniMax-H3-Image on Apple Silicon.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="text to image, or an edit with --image")
    g.add_argument("-p", "--prompt", required=True)
    g.add_argument("-s", "--size", type=_size, default=(2048, 2048),
                   help="WIDTHxHEIGHT, or an aspect ratio such as 16:9 for the official ~16 MP size "
                        "(16:9@2k for ~4 MP); default 2048x2048")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-n", "--count", type=int, default=1, help="images to make, with seeds seed, seed+1, ...")
    g.add_argument("-o", "--output")
    g.add_argument("-i", "--image", action="append", help="reference image (repeat for several); makes it an edit")
    g.add_argument("--min-area", type=int, default=2048 * 2048,
                   help="smallest canvas the image is generated on (default 2048*2048, as released); "
                        "1024 generates at the requested size")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.set_defaults(func=cmd_generate)
    c = sub.add_parser("convert", help="convert the official weights into h3metal's format")
    c.add_argument("src", help="the MiniMax-H3-Image directory (transformer/, text_encoder/, vae/, ...)")
    c.add_argument("--out", default=DEFAULT_MODEL)
    c.add_argument("--dit", default="q8", help="q8, q4, q4g, mix (int8 attention, int4 MLP) or one per matrix: "
                   "qkv,out,fc1,fc2 (e.g. q8,q8,q4,q4)")
    c.add_argument("--te", default="q8", help="text encoder weights: q8, q4g or f16")
    c.add_argument("--vae", default="f16")
    c.add_argument("--only", help="comma-separated subset of vae,vision,te,dit,tokenizer")
    c.add_argument("--gptq", action="store_true",
                   help="choose the transformer's codes with GPTQ on calibration runs (about an hour on an M6)")
    c.set_defaults(func=cmd_convert)
    d = sub.add_parser("download", help="fetch the official weights h3metal converts from (about 125 GB)")
    d.add_argument("dest", nargs="?", default=os.path.expanduser("~/.h3metal/checkpoint"))
    d.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    d.add_argument("-j", "--jobs", type=int, default=4)
    d.set_defaults(func=cmd_download)
    i = sub.add_parser("info", help="device and model files")
    i.add_argument("--model", default=DEFAULT_MODEL)
    i.set_defaults(func=cmd_info)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
