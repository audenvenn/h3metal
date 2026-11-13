"""`h3metal serve`: a local web page for making images. The model stays loaded between images, so only the
first one pays for mapping the weights. One image at a time (the GPU); images go to ~/.h3metal/outputs with
the same PNG metadata as `h3metal generate`. Standard library only; it listens on 127.0.0.1 unless told
otherwise.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT_DIR = os.path.expanduser("~/.h3metal/outputs")
LORA_DIR = os.path.expanduser("~/.h3metal/loras")
MAX_BODY = 64 << 20
MAX_LORAS = 4


class Studio:
    def __init__(self, models: dict[str, str], default: str | None, lora_dir: str = LORA_DIR):
        self.models = models                    # name -> converted directory
        self.lora_dir = lora_dir                # the page offers the .safetensors files here
        self.default = default if default in models else next(iter(models), None)
        self.lock = threading.Lock()            # one image at a time
        self.pipe = self.pipe_name = None
        self.progress = {"busy": False, "stage": "", "step": 0, "steps": 0}
        os.makedirs(OUT_DIR, exist_ok=True)

    def _pipeline(self, name: str):
        from .cli import open_pipeline
        if self.pipe_name != name:
            if self.pipe is not None and hasattr(self.pipe, "release"):
                self.pipe.release()
            self.pipe = None
            self.progress["stage"] = f"loading {name}"
            self.pipe, _ = open_pipeline(self.models[name])
            self.pipe_name = name
        return self.pipe

    def lora_files(self) -> list[str]:
        if not os.path.isdir(self.lora_dir):
            return []
        return sorted(f for f in os.listdir(self.lora_dir) if f.endswith(".safetensors"))

    def _loras(self, req: dict, name: str) -> list:
        """The request's adapters [(path, strength)]: files of the LoRA directory only, by name."""
        items = req.get("loras") or []
        if not items:
            return []
        if name == "MiniMax-H3-Image":
            raise ValueError("LoRA adapters work with Z-Image-Turbo and FLUX.2 [klein]")
        if len(items) > MAX_LORAS:
            raise ValueError(f"at most {MAX_LORAS} LoRA adapters")
        files, out = set(self.lora_files()), []
        for it in items:
            f, s = str(it.get("name", "")), float(it.get("strength", 1.0))
            if f not in files:
                raise ValueError(f"no LoRA {f!r} in {self.lora_dir}")
            if not -4.0 <= s <= 4.0:
                raise ValueError("LoRA strength must be between -4 and 4")
            out.append((os.path.join(self.lora_dir, f), s))
        return out

    def generate(self, req: dict) -> dict:
        name = req.get("model") or self.default  # the request is checked before anything heavy is imported
        if name not in self.models:
            raise ValueError(f"unknown model {name!r}")
        prompt = str(req.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("the prompt is empty")
        seed = int(req.get("seed", 0))
        size = req.get("size")
        w, h = (int(size[0]), int(size[1])) if size else (None, None)
        images = req.get("images") or []
        if images and name == "Z-Image-Turbo":
            raise ValueError("Z-Image-Turbo makes images from text only")
        loras = self._loras(req, name)
        from .cli import _save
        from .pipeline import generate_edit
        refs = []
        if images:
            from PIL import Image
            refs = [Image.open(io.BytesIO(base64.b64decode(d.split(",", 1)[-1]))) for d in images]

        def progress(step, total):
            self.progress.update(stage="denoising", step=step, steps=total)
        t0 = time.time()
        self.progress.update(busy=True, stage="starting", step=0, steps=0)
        try:
            pipe = self._pipeline(name)
            self.progress["stage"] = "text"
            if name.startswith("FLUX.2"):
                img, timing = pipe.generate(prompt, w, h, seed=seed, images=refs or None, progress=progress,
                                            loras=loras)
            elif refs:  # MiniMax-H3-Image
                w, h = (w, h) if size else (2048, 2048)
                img, timing = generate_edit(pipe, prompt, refs, w, h, seed=seed, progress=progress)
            else:
                w, h = (w, h) if size else getattr(pipe, "default_size", (2048, 2048))
                img, timing = pipe.generate(prompt, w, h, seed=seed, progress=progress, loras=loras)
        finally:
            self.progress.update(busy=False, stage="", step=0, steps=0)
        ih, iw = img.shape[:2]
        out = os.path.join(OUT_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}_{seed}.png")
        _save(img, out, {"prompt": prompt, "seed": seed, "size": f"{iw}x{ih}", "model": name,
                         "references": str(len(refs)) if refs else "none", "loras": _lora_text(loras)})
        return {"url": "/outputs/" + os.path.basename(out), "size": [iw, ih], "seed": seed, "model": name,
                "timing": str(timing), "wall": round(time.time() - t0, 1)}

    def history(self, n: int = 60) -> list[dict]:
        files = sorted((f for f in os.listdir(OUT_DIR) if f.endswith(".png")), reverse=True)[:n]
        out = []
        if files:
            from PIL import Image
        for f in files:
            try:
                meta = Image.open(os.path.join(OUT_DIR, f)).text  # noqa: F821 (imported above when there are files)
            except Exception:
                continue
            out.append({"url": "/outputs/" + f, "prompt": meta.get("prompt", ""), "seed": meta.get("seed", ""),
                        "size": meta.get("size", ""), "model": meta.get("model", "")})
        return out


def _lora_text(loras) -> str:
    return ", ".join(f"{os.path.basename(p)}:{s:g}" for p, s in loras) or "none"


def make_handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: one line per image instead
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._json(200, {"models": list(studio.models), "default": studio.default,
                                 "loras": studio.lora_files(), "progress": studio.progress})
            elif self.path == "/history":
                self._json(200, studio.history())
            elif self.path.startswith("/outputs/"):
                name = os.path.basename(self.path[len("/outputs/"):])
                path = os.path.join(OUT_DIR, name)
                if not name.endswith(".png") or not os.path.isfile(path):
                    return self._send(404, b"not found", "text/plain")
                with open(path, "rb") as f:
                    self._send(200, f.read(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "not JSON"})
            if not studio.lock.acquire(blocking=False):
                return self._json(409, {"error": "busy with another image"})
            try:
                res = studio.generate(req)
                print(f"{res['url']}: {res['model']}, {res['size'][0]}x{res['size'][1]}, seed {res['seed']}; "
                      f"{res['timing']}", flush=True)
                self._json(200, res)
            except Exception as e:  # shown on the page
                self._json(400, {"error": str(e)})
            finally:
                studio.lock.release()
    return Handler


def serve(models: dict[str, str], default: str | None, host: str = "127.0.0.1", port: int = 7860,
          open_browser: bool = False, lora_dir: str = LORA_DIR):
    if not models:
        raise SystemExit("no converted model: `h3metal download --model z-image-turbo` (or flux2-klein-4b, "
                         "minimax-h3) and `h3metal convert` first")
    studio = Studio(models, default, os.path.expanduser(lora_dir))
    httpd = ThreadingHTTPServer((host, port), make_handler(studio))
    url = f"http://{host}:{port}/"
    print(f"h3metal studio at {url} ({', '.join(models)}); Ctrl-C stops it", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>h3metal studio</title>
<style>
:root { --bg: #f7f7f5; --panel: #ffffff; --ink: #1d1d1f; --muted: #6e6e73; --line: #e2e2e0; --accent: #2a62d6; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161617; --panel: #1f1f21; --ink: #f2f2f2; --muted: #9a9aa0; --line: #333336; --accent: #6b9bff; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 -apple-system, BlinkMacSystemFont, sans-serif; }
header { padding: 14px 20px; border-bottom: 1px solid var(--line); display: flex; gap: 12px; align-items: baseline; }
header b { font-size: 17px; } header span { color: var(--muted); font-size: 13px; }
main { display: grid; grid-template-columns: 340px 1fr; gap: 20px; padding: 20px; max-width: 1400px; margin: auto; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 4px; }
select, textarea, input { width: 100%; font: inherit; color: var(--ink); background: var(--bg);
  border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
textarea { min-height: 120px; resize: vertical; }
.row { display: flex; gap: 8px; } .row > * { flex: 1; }
button { font: inherit; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
#go { width: 100%; margin-top: 16px; background: var(--accent); color: #fff; font-weight: 600; }
#go:disabled { opacity: .55; cursor: default; }
.ghost { background: var(--bg); color: var(--ink); border: 1px solid var(--line); padding: 8px 10px; flex: 0 0 auto; }
#refs { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; } #refs img { width: 56px; height: 56px; object-fit: cover; border-radius: 6px; }
#status { color: var(--muted); font-size: 13px; min-height: 20px; margin-top: 10px; }
#bar { height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; margin-top: 6px; }
#bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .3s; }
#view { display: flex; justify-content: center; align-items: center; min-height: 420px; }
#view img { max-width: 100%; max-height: 72vh; border-radius: 8px; }
#meta { color: var(--muted); font-size: 13px; margin-top: 8px; text-align: center; overflow-wrap: anywhere; }
#gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; margin-top: 16px; }
#gallery img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; cursor: pointer; }
.note { color: var(--muted); font-size: 12px; margin-top: 12px; }
</style></head><body>
<header><b>h3metal studio</b><span>image generation on this Mac's GPU &middot; images are AI-generated</span></header>
<main>
<section class="panel">
  <label for="model">Model</label><select id="model"></select>
  <label for="prompt">Prompt</label><textarea id="prompt" placeholder="A red fox in fresh snow at sunrise"></textarea>
  <label>Size</label>
  <div class="row"><select id="size">
    <option value="1024x1024">1024 &times; 1024</option><option value="1536x1024">1536 &times; 1024</option>
    <option value="1024x1536">1024 &times; 1536</option><option value="2048x2048">2048 &times; 2048</option>
    <option value="auto">as the reference (edits)</option></select></div>
  <label for="seed">Seed</label>
  <div class="row"><input id="seed" type="number" value="0" min="0"><button class="ghost" id="rnd" title="random seed">&#x21bb;</button></div>
  <label for="lora">LoRA (files in ~/.h3metal/loras)</label>
  <div class="row"><select id="lora"><option value="">none</option></select>
    <input id="strength" type="number" value="1" step="0.1" min="-4" max="4" title="strength" style="flex: 0 0 76px"></div>
  <label id="reflabel" for="files">Reference images (edits)</label>
  <input id="files" type="file" accept="image/*" multiple><div id="refs"></div>
  <button id="go">Generate</button>
  <div id="bar"><i></i></div><div id="status"></div>
  <p class="note">The first image after choosing a model maps its weights; the next ones start at once.</p>
</section>
<section class="panel">
  <div id="view"><span class="note">Your image appears here.</span></div><div id="meta"></div>
  <div id="gallery"></div>
</section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let refs = [], polling = null;
async function state() { return (await fetch('/state')).json(); }
function refAllowed() { return !$('model').value.startsWith('Z-Image'); }
function syncRefs() { const ok = refAllowed(); $('files').disabled = !ok; $('reflabel').style.opacity = ok ? 1 : .5; }
async function init() {
  const s = await state();
  for (const m of s.models) { const o = document.createElement('option'); o.textContent = m; $('model').appendChild(o); }
  for (const f of s.loras || []) { const o = document.createElement('option'); o.textContent = f; $('lora').appendChild(o); }
  if (s.default) $('model').value = s.default;
  syncRefs(); loadHistory();
}
$('model').onchange = syncRefs;
$('rnd').onclick = () => { $('seed').value = Math.floor(Math.random() * 1e9); };
$('files').onchange = async (e) => {
  refs = []; $('refs').innerHTML = '';
  for (const f of e.target.files) {
    const url = await new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(f); });
    refs.push(url); const im = document.createElement('img'); im.src = url; $('refs').appendChild(im);
  }
};
function show(url, text) { $('view').innerHTML = ''; const im = document.createElement('img'); im.src = url; $('view').appendChild(im); $('meta').textContent = text; }
async function loadHistory() {
  const h = await (await fetch('/history')).json(); $('gallery').innerHTML = '';
  for (const it of h) {
    const im = document.createElement('img'); im.src = it.url; im.title = it.prompt; im.loading = 'lazy';
    im.onclick = () => show(it.url, `${it.model} · ${it.size} · seed ${it.seed} · ${it.prompt}`);
    $('gallery').appendChild(im);
  }
}
function poll() {
  polling = setInterval(async () => {
    const p = (await state()).progress;
    if (p.steps) { $('bar').firstChild.style.width = (100 * p.step / p.steps) + '%'; }
    $('status').textContent = p.stage ? (p.stage + (p.steps ? ` ${p.step}/${p.steps}` : '')) : '';
  }, 300);
}
$('go').onclick = async () => {
  const size = $('size').value;
  const body = { model: $('model').value, prompt: $('prompt').value, seed: +$('seed').value,
                 size: size === 'auto' ? null : size.split('x').map(Number), images: refAllowed() ? refs : [],
                 loras: $('lora').value ? [{ name: $('lora').value, strength: +$('strength').value }] : [] };
  $('go').disabled = true; $('bar').firstChild.style.width = '0'; $('status').textContent = 'starting'; poll();
  try {
    const r = await fetch('/generate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    show(j.url, `${j.model} · ${j.size[0]}x${j.size[1]} · seed ${j.seed} · ${j.wall} s (${j.timing})`);
    $('status').textContent = `done in ${j.wall} s`; loadHistory();
  } catch (e) { $('status').textContent = 'error: ' + e.message; }
  finally { clearInterval(polling); $('go').disabled = false; $('bar').firstChild.style.width = '0'; }
};
init();
</script></body></html>
"""
