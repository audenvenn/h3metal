"""h3metal command line: generate, edit, convert, info."""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_MODEL = os.environ.get("H3METAL_MODEL", os.path.expanduser("~/.h3metal/model"))
ZIMAGE_MODEL = os.path.expanduser("~/.h3metal/zimage")  # converted Z-Image-Turbo, by default
FLUX2_MODEL = os.path.expanduser("~/.h3metal/flux2-klein")  # converted FLUX.2 [klein] 4B, by default
FLUX2_9B_MODEL = os.path.expanduser("~/.h3metal/flux2-klein-9b-h3m")  # converted FLUX.2 [klein] 9B
MODEL_NAMES = {"zimage": ZIMAGE_MODEL, "z-image": ZIMAGE_MODEL, "z-image-turbo": ZIMAGE_MODEL,
               "flux2": FLUX2_MODEL, "flux2-klein": FLUX2_MODEL, "flux2-klein-4b": FLUX2_MODEL, "klein": FLUX2_MODEL,
               "flux2-9b": FLUX2_9B_MODEL, "flux2-klein-9b": FLUX2_9B_MODEL, "klein-9b": FLUX2_9B_MODEL,
               "minimax-h3": DEFAULT_MODEL, "h3": DEFAULT_MODEL}


def _has_model(d: str) -> bool:
    """A converted model is there: a transformer file (MiniMax-H3-Image may have only its few-step one)."""
    return any(os.path.exists(os.path.join(d, f)) for f in ("dit.h3m", "dit-pdmd.h3m"))


def _model_dir(name: str | None) -> str:
    """--model: a converted model's directory, or one of the names in MODEL_NAMES. Without it, the first converted
    of MiniMax-H3-Image (or H3METAL_MODEL if set), Z-Image-Turbo, FLUX.2 [klein] 4B and 9B: the model the user
    has."""
    if name:
        return MODEL_NAMES.get(name.lower(), os.path.expanduser(name))
    if "H3METAL_MODEL" in os.environ:
        return DEFAULT_MODEL
    for d in (DEFAULT_MODEL, ZIMAGE_MODEL, FLUX2_MODEL, FLUX2_9B_MODEL):
        if _has_model(d):
            return d
    return DEFAULT_MODEL


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


def open_pipeline(model_dir: str):
    """The pipeline for a converted model's directory -> (pipeline, model name)."""
    from .flux2 import Flux2Pipeline, is_flux2_dir
    from .pipeline import Pipeline
    from .zimage import ZImagePipeline, is_zimage_dir
    if is_zimage_dir(model_dir):
        pipe = ZImagePipeline(model_dir)
    elif is_flux2_dir(model_dir):
        pipe = Flux2Pipeline(model_dir)
    else:
        pipe = Pipeline(model_dir)
    return pipe, getattr(pipe, "model_name", "MiniMax-H3-Image")


def cmd_generate(a):
    from .pipeline import generate_edit
    t0 = time.time()
    a.model = _model_dir(a.model)
    pipe, model_name = open_pipeline(a.model)
    if a.image and model_name == "Z-Image-Turbo":
        raise ValueError("Z-Image-Turbo makes images from text only; --image needs FLUX.2 [klein] or "
                         "MiniMax-H3-Image")
    _check_device(pipe.gpu.device)
    flux2_edit = model_name.startswith("FLUX.2") and a.image
    from .lora import parse_spec
    loras = [parse_spec(s) for s in a.lora or []]
    if loras and model_name == "MiniMax-H3-Image":
        raise ValueError("LoRA adapters work with Z-Image-Turbo and FLUX.2 [klein]")
    steps = {"steps": a.steps} if a.steps else {}
    if steps and model_name != "MiniMax-H3-Image":
        raise ValueError(f"{model_name} runs the step count it was distilled for; --steps is for MiniMax-H3-Image")
    if steps:
        pipe.steps_path(a.steps)  # fails before the text encoder runs if no transformer here runs that many
    w, h = a.size or ((None, None) if flux2_edit else getattr(pipe, "default_size", (2048, 2048)))
    if model_name == "MiniMax-H3-Image" and (w % 32 or h % 32):
        raise ValueError("MiniMax-H3-Image needs width and height that are multiples of 32")
    for i in range(a.count):
        seed = a.seed + i
        t1 = time.time()
        if flux2_edit:  # the official pipeline: without -s the image takes the first reference's size
            refs = [_open_image(p) for p in a.image]
            img, timing = pipe.generate(a.prompt, w, h, seed=seed, images=refs, progress=_progress, loras=loras)
        elif a.image:
            refs = [_open_image(p) for p in a.image]
            img, timing = generate_edit(pipe, a.prompt, refs, w, h, seed=seed, min_area=a.min_area,
                                        progress=_progress, **steps)
        else:
            img, timing = pipe.generate(a.prompt, w, h, seed=seed, min_area=a.min_area, progress=_progress,
                                        **({"loras": loras} if loras else {}), **steps)
        out = a.output or f"h3metal_{int(time.time())}.png"
        if a.count > 1:
            stem, ext = os.path.splitext(out)
            out = f"{stem}_{seed}{ext or '.png'}"
        ih, iw = img.shape[:2]
        _save(img, out, {"prompt": a.prompt, "seed": seed, "size": f"{iw}x{ih}", "model": model_name,
                         "references": ", ".join(os.path.basename(p) for p in a.image or []) or "none",
                         "loras": ", ".join(f"{os.path.basename(p)}:{s:g}" for p, s in loras) or "none",
                         **({"steps": str(a.steps)} if steps else {})})
        print(f"{out}: {iw}x{ih}, seed {seed}; {timing}; wall {time.time() - (t0 if i == 0 else t1):.1f}s")


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
    from .flux2 import checkpoint_name, convert_flux2, is_flux2_checkpoint
    from .zimage import convert_zimage, is_zimage_checkpoint
    if is_flux2_checkpoint(a.src):
        dit = "q8" if a.dit == "auto" else FORMAT_ALIASES.get(a.dit, a.dit)  # 3.9 GB in int8
        nine = checkpoint_name(a.src).endswith("9B")
        default = FLUX2_9B_MODEL if nine else FLUX2_MODEL
        out = _model_dir(a.out) if a.out else default
        convert_flux2(a.src, out, dit=dit, te=a.te, parts=a.only.split(",") if a.only else None)
        alias = "flux2-9b" if nine else "flux2"
        print(f"next: h3metal generate --model {alias if out == default else out} -p \"...\"")
        return
    if is_zimage_checkpoint(a.src):
        # Z-Image's 6 B transformer is 6.2 GB in int8, which fits every Mac h3metal runs on.
        dit = "q8" if a.dit == "auto" else FORMAT_ALIASES.get(a.dit, a.dit)
        out = _model_dir(a.out) if a.out else ZIMAGE_MODEL  # never MiniMax-H3's directory by default
        convert_zimage(a.src, out, dit=dit, te=a.te, parts=a.only.split(",") if a.only else None)
        print(f"next: h3metal generate --model {'zimage' if out == ZIMAGE_MODEL else out} -p \"...\"")
        return
    a.out = _model_dir(a.out) if a.out else DEFAULT_MODEL
    if a.pdmd:
        return convert_pdmd(a)
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


def convert_pdmd(a):
    """MiniMax-H3-Image's few-step transformer, dit-pdmd.h3m: pDMD's 2-step LoRA merged into the weights, converted
    for the plain flow sampler at 2, 3 and 4 steps (shift 12), next to the six-step dit.h3m. The text encoder, VAE
    and vision files are shared; any that are missing are converted too."""
    from .convert import convert_dit
    from .download import PDMD, pdmd_lora
    if a.gptq:
        raise ValueError("--gptq calibrates the six-step transformer; it does not apply to --pdmd")
    if a.dit == "auto":
        a.dit = auto_format()
        print(f"transformer format for this Mac: {a.dit} (choose another with --dit)")
    a.dit = FORMAT_ALIASES.get(a.dit, a.dit)
    lora = pdmd_lora() if a.pdmd == "auto" else a.pdmd
    shared = [p for p, f in (("vae", "vae.h3m"), ("vision", "vision.h3m"), ("te", "te.h3m"),
                             ("tokenizer", "tokenizer.json")) if not os.path.exists(os.path.join(a.out, f))]
    if shared:
        a.only, a.pdmd = ",".join(shared), None
        print(f"converting the missing {a.only} as well")
        cmd_convert(a)
    fmts = dict(zip(("qkv", "out", "fc1", "fc2"), a.dit.split(","))) if "," in a.dit else \
        {k: a.dit for k in ("qkv", "out", "fc1", "fc2")}
    convert_dit(a.src, os.path.join(a.out, "dit-pdmd.h3m"), fmts=fmts, lora=[(lora, 1.0)],
                flow={"shift": PDMD["shift"], "steps": PDMD["steps"]})
    print("next: h3metal generate --steps 2 -p \"...\" (3 and 4 steps look closer to the six-step model)")


def cmd_download(a):
    from .download import CONVERTED, OTHER, NotPublished, download, download_converted, download_lora
    if a.lora:
        path = download_lora(a.lora, a.source)
        print(f"next: h3metal generate ... --lora {os.path.splitext(os.path.basename(path))[0]}")
        return
    alias = {"z-image-turbo": "zimage", "flux2-klein-4b": "flux2"}.get(a.model)
    if a.model in CONVERTED and not a.original:
        if a.source == "huggingface":
            dest = os.path.abspath(os.path.expanduser(a.dest)) if a.dest else \
                os.path.expanduser(f"~/.h3metal/{CONVERTED[a.model]['dest']}")
            try:
                ok = download_converted(dest, a.model, a.jobs)
            except NotPublished as e:  # then as before: the official checkpoint, to convert
                print(f"{e}; fetching the official checkpoint to convert instead")
                a.dest = None
            else:
                if not ok:
                    sys.exit("some files did not download; run the command again to resume")
                default = os.path.expanduser(f"~/.h3metal/{CONVERTED[a.model]['dest']}")
                print(f"next: h3metal generate --model {alias if dest == default else dest} -p \"...\"")
                return
        else:
            print(f"converted weights are on Hugging Face only; fetching the official checkpoint from {a.source} "
                  "to convert")
    terms = OTHER.get(a.model, {}).get("license")
    if terms and not a.accept_license:
        sys.exit(f"{a.model} is licensed under the {terms}. Read it, and pass --accept-license to download the "
                 "weights under it.")
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


def cmd_serve(a):
    from .serve import serve
    from .flux2 import flux2_name, is_flux2_dir
    from .zimage import is_zimage_dir
    names = {"MiniMax-H3-Image": DEFAULT_MODEL, "Z-Image-Turbo": ZIMAGE_MODEL, "FLUX.2 [klein] 4B": FLUX2_MODEL,
             "FLUX.2 [klein] 9B": FLUX2_9B_MODEL}
    models = {n: d for n, d in names.items() if _has_model(d)}
    default = None
    if a.model:
        d = _model_dir(a.model)
        default = "Z-Image-Turbo" if is_zimage_dir(d) else flux2_name(d) if is_flux2_dir(d) else "MiniMax-H3-Image"
        models[default] = d
    serve(models, default, host=a.host, port=a.port, open_browser=a.open, lora_dir=a.loras)


def cmd_info(a):
    from .gpu import Gpu
    g = Gpu()
    print(f"device: {g.device}; recommended working set {g.working_set / 1e9:.1f} GB; "
          f"max buffer {g.max_buffer / 1e9:.1f} GB")
    _check_device(g.device)
    from .flux2 import flux2_name, is_flux2_dir
    from .zimage import is_zimage_dir
    if a.model is None and not any(_has_model(d)
                                   for d in (DEFAULT_MODEL, ZIMAGE_MODEL, FLUX2_MODEL, FLUX2_9B_MODEL)):
        print("no model yet: `h3metal download --model flux2-klein-4b` (or z-image-turbo) fetches one ready to run")
        return
    a.model = _model_dir(a.model)
    other = is_zimage_dir(a.model) or is_flux2_dir(a.model)
    name = "Z-Image-Turbo" if is_zimage_dir(a.model) else flux2_name(a.model) if is_flux2_dir(a.model) else \
        "MiniMax-H3-Image"
    print(f"model: {name} ({a.model})")
    from .weights import read_header
    parts = ("te", "dit", "vae") + (() if other else ("dit-pdmd", "vision"))
    for k in parts:
        p = os.path.join(a.model, f"{k}.h3m")
        if os.path.exists(p):
            m = read_header(p)["meta"]
            steps = ""
            if k.startswith("dit") and not other:
                grids = sorted(int(n) for n in (m.get("schedules") or {str(len(m["timesteps"])): 0}))
                steps = f"  steps {', '.join(map(str, grids))}"
            print(f"{k:8s} {os.path.getsize(p) / 1e9:6.2f} GB  {m.get('fmts') or m.get('fmt', '')}{steps}")
        elif k == "dit-pdmd":
            print(f"{k:8s} not converted (2, 3 and 4 steps: `h3metal convert <checkpoint> --pdmd`)")
        elif not (k == "dit" and os.path.exists(os.path.join(a.model, "dit-pdmd.h3m"))):
            print(f"{k:8s} missing ({p})")


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="h3metal", description="Image generation on Apple Silicon's GPU matrix units: "
                                 "FLUX.2 [klein], Z-Image-Turbo and MiniMax-H3-Image.")
    ap.add_argument("--version", action="version", version=f"h3metal {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="text to image, or an edit with --image")
    g.add_argument("-p", "--prompt", required=True)
    g.add_argument("-s", "--size", type=_size, default=None,
                   help="WIDTHxHEIGHT, or an aspect ratio such as 16:9 for the official ~16 MP size "
                        "(16:9@2k for ~4 MP); default 1024x1024 for FLUX.2 [klein] and Z-Image-Turbo (an edit "
                        "with FLUX.2 [klein]: the first reference's size), 2048x2048 for MiniMax-H3-Image")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-n", "--count", type=int, default=1, help="images to make, with seeds seed, seed+1, ...")
    g.add_argument("-o", "--output")
    g.add_argument("-i", "--image", action="append", help="reference image (repeat for several); makes it an edit")
    g.add_argument("--lora", action="append", metavar="FILE[:STRENGTH]",
                   help="a LoRA adapter (.safetensors) for Z-Image-Turbo or FLUX.2 [klein], at a strength "
                        "(default 1); a bare name is looked up in ~/.h3metal/loras; repeat for several")
    g.add_argument("--steps", type=int, default=None,
                   help="MiniMax-H3-Image: 6 (default), or 2, 3 or 4 with the few-step transformer (convert --pdmd)")
    g.add_argument("--min-area", type=int, default=2048 * 2048,
                   help="smallest canvas the image is generated on (default 2048*2048, as released); "
                        "1024 generates at the requested size")
    g.add_argument("--model", default=None,
                   help="a converted model's directory, or flux2 / flux2-9b / zimage / minimax-h3 for the default ones "
                   "(default: the first converted of MiniMax-H3-Image, Z-Image-Turbo, FLUX.2 [klein])")
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
    c.add_argument("--pdmd", nargs="?", const="auto", metavar="LORA",
                   help="MiniMax-H3-Image: write the few-step transformer (2, 3 or 4 steps), dit-pdmd.h3m, with "
                        "pDMD's 2-step LoRA merged (downloaded unless LORA names its file)")
    c.add_argument("--gptq", action="store_true",
                   help="choose the transformer's codes with GPTQ on calibration runs (about an hour on an M6)")
    c.set_defaults(func=cmd_convert)
    d = sub.add_parser("download", help="fetch converted weights ready to run (Z-Image-Turbo 10.7 GB, FLUX.2 "
                       "[klein] 4B 8.3 GB), or the official weights h3metal converts from (MiniMax-H3-Image about "
                       "125 GB, FLUX.2 [klein] 9B 35 GB; --original for the others)")
    d.add_argument("dest", nargs="?", default=None,
                   help="default: ~/.h3metal/zimage or ~/.h3metal/flux2-klein for converted weights, "
                        "~/.h3metal/<model> for an official checkpoint (~/.h3metal/checkpoint for MiniMax-H3-Image)")
    d.add_argument("--model", choices=["minimax-h3", "z-image-turbo", "flux2-klein-4b", "flux2-klein-9b"],
                   default="minimax-h3")
    d.add_argument("--original", action="store_true",
                   help="z-image-turbo, flux2-klein-4b: the official checkpoint, to convert yourself, instead of "
                        "the converted weights")
    d.add_argument("--lora", metavar="OWNER/NAME[/FILE]",
                   help="fetch a LoRA adapter's .safetensors file into ~/.h3metal/loras instead (from --source)")
    d.add_argument("--accept-license", action="store_true",
                   help="accept a model's own license where it asks for that (FLUX.2 [klein] 9B: the FLUX "
                        "Non-Commercial License)")
    d.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface",
                   help="where official checkpoints come from; converted weights are on Hugging Face only")
    d.add_argument("-j", "--jobs", type=int, default=4)
    d.set_defaults(func=cmd_download)
    v = sub.add_parser("serve", help="a local web page for making images; the model stays loaded between them")
    v.add_argument("--model", default=None, help="the model selected at first (default: the first converted one)")
    v.add_argument("--host", default="127.0.0.1", help="the address to listen on (default: this Mac only)")
    v.add_argument("--port", type=int, default=7860)
    v.add_argument("--open", action="store_true", help="open the page in the browser")
    v.add_argument("--loras", default="~/.h3metal/loras", help="the LoRA files the page offers (default "
                   "~/.h3metal/loras)")
    v.set_defaults(func=cmd_serve)
    i = sub.add_parser("info", help="device and model files")
    i.add_argument("--model", default=None)
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
