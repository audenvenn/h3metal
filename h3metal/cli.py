"""h3metal command line: generate, edit, convert, info."""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_MODEL = os.environ.get("H3METAL_MODEL", os.path.expanduser("~/.h3metal/model"))


def _size(s: str):
    try:
        w, h = (int(x) for x in s.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 2048x2048")
    if w % 32 or h % 32 or not (512 <= w <= 8192 and 512 <= h <= 8192):
        raise argparse.ArgumentTypeError("width and height must be multiples of 32 between 512 and 8192")
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
    w, h = a.size
    if a.image:
        refs = [Image.open(p) for p in a.image]
        img, timing = generate_edit(pipe, a.prompt, refs, w, h, seed=a.seed, min_area=a.min_area,
                                    progress=_progress)
    else:
        img, timing = pipe.generate(a.prompt, w, h, seed=a.seed, min_area=a.min_area, progress=_progress)
    out = a.output or f"h3metal_{int(time.time())}.png"
    Image.fromarray(img).save(out)
    print(f"{out}: {w}x{h}, seed {a.seed}; {timing}; wall {time.time() - t0:.1f}s")


def cmd_convert(a):
    from .convert import convert_dit
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


def cmd_download(a):
    from .download import download
    if not download(a.dest, a.source, a.jobs):
        sys.exit("some files did not download; run the command again to resume")
    print(f"next: h3metal convert {a.dest}")


def cmd_info(a):
    from .gpu import Gpu
    g = Gpu()
    print(f"device: {g.device}; recommended working set {g.working_set / 1e9:.1f} GB; "
          f"max buffer {g.max_buffer / 1e9:.1f} GB")
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
    g.add_argument("-s", "--size", type=_size, default=(2048, 2048), help="WIDTHxHEIGHT (default 2048x2048)")
    g.add_argument("--seed", type=int, default=0)
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
    c.add_argument("--dit", default="q8", help="q8, q4, q4g or one per matrix: qkv,out,fc1,fc2 (e.g. q8,q8,q4,q4)")
    c.add_argument("--te", default="q8", help="text encoder weights: q8, q4g or f16")
    c.add_argument("--vae", default="f16")
    c.add_argument("--only", help="comma-separated subset of vae,vision,te,dit,tokenizer")
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
