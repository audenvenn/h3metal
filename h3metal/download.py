"""`h3metal download`: fetch converted weights, or the files of an official checkpoint that h3metal reads.

Converted weights (CONVERTED) are h3metal's own files, ready to run: 10.7 GB for Z-Image-Turbo instead of a
33 GB checkpoint and a conversion, 8.3 GB for FLUX.2 [klein] 4B instead of 15 GB. Each h3metal version fetches
them at a fixed tag of their repository, the layout it reads; a converter change that alters a layout gets a
new tag. They are the bytes `h3metal convert` makes with its default settings, and every file is checked
against its published SHA-256.

The official repository holds about 204 GB: the transformer twice (original
and diffusers layout), the VAE twice, and the whole 64-layer text encoder.
h3metal converts from the original layout and needs the text encoder's first
50 layers (MiniMax-H3 reads hidden_states[50]) and its vision tower, so this
fetches about 125 GB. Hugging Face (MiniMaxAI/MiniMax-H3-Image) or ModelScope
(MiniMax/MiniMax-H3-Image) can serve it; downloads resume, and files with a
published SHA-256 are checked. Standard library only.

By downloading the weights you accept the MiniMax-H3-Image Community License
Agreement (LICENSE in the downloaded directory), which among other terms
limits use to its Applicable Territory.

`h3metal download --lora owner/name` fetches a LoRA adapter's file into
~/.h3metal/loras instead. Adapters carry their own licenses.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPOS = {"huggingface": "MiniMaxAI/MiniMax-H3-Image", "modelscope": "MiniMax/MiniMax-H3-Image"}
# Other models h3metal runs: their repositories, all files needed.
OTHER = {"z-image-turbo": {"huggingface": "Tongyi-MAI/Z-Image-Turbo", "modelscope": "Tongyi-MAI/Z-Image-Turbo"},
         "flux2-klein-4b": {"huggingface": "black-forest-labs/FLUX.2-klein-4B",
                            "modelscope": "black-forest-labs/FLUX.2-klein-4B",
                            # the transformer again as one file in BFL's own layout, and sample images
                            "exclude": ["flux-2-klein-4b.safetensors", "*.jpg"]},
         "flux2-klein-9b": {"huggingface": "black-forest-labs/FLUX.2-klein-9B",
                            "modelscope": "black-forest-labs/FLUX.2-klein-9B",
                            "exclude": ["flux-2-klein-9b.safetensors", "*.jpg"],
                            # FLUX Non-Commercial License v2.1: `h3metal download` asks for --accept-license
                            "license": "FLUX Non-Commercial License v2.1 (non-commercial use only; see LICENSE.md "
                                       "in the downloaded directory)"}}
# Converted weights on Hugging Face: repository, the tag holding the layout this version reads, the directory
# they go to. (FLUX.2 [klein] 9B's license does not let h3metal publish converted copies.)
CONVERTED = {"z-image-turbo": {"repo": "audenvenn/h3metal-z-image-turbo", "revision": "h3m-1", "dest": "zimage"},
             "flux2-klein-4b": {"repo": "audenvenn/h3metal-flux2-klein-4b", "revision": "h3m-1",
                                "dest": "flux2-klein"}}
UA = {"User-Agent": "h3metal-download"}
N_LAYERS = 50

WANTED = [
    "LICENSE", "README.md", "model_index.json", "pdd_config.json", "pdd_fused_heads.safetensors",
    "transformer/*", "vae/*", "tokenizer/*", "processor/*", "text_encoder/config.json",
    "text_encoder/*.json", "text_encoder/*.txt", "examples/*",
]


# pDMD's 2-step LoRA for MiniMax-H3 (arXiv 2609.35768), which MiniMax-H3-Image takes as a one-frame video. The fp32
# file is the training precision (the bf16 one is a rounding of it). convert --pdmd merges it.
PDMD = {"repo": "pdmd2026/pdmd_2NFE_lora", "file": "lora_model_0_fp32.safetensors", "license": "Apache-2.0",
        "dir": os.path.expanduser("~/.h3metal/pdmd"), "shift": 12.0, "steps": [2, 3, 4],
        # the file h3metal was checked with (revision 3bc1bf125da7 of the repository)
        "size": 2767276568, "sha256": "f81e52f91271ca68f6a3587a491f4089fe536d627e81eb659f62d096f1d32b16"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def pdmd_lora(log=print) -> str:
    """pDMD's LoRA file: the local copy when it is the one h3metal was checked with (no network needed), otherwise
    downloaded, and refused if Hugging Face now serves a different file."""
    path = os.path.join(PDMD["dir"], PDMD["file"])
    if os.path.exists(path) and os.path.getsize(path) == PDMD["size"] and _sha256(path) == PDMD["sha256"]:
        return path
    log(f"pDMD's 2-step LoRA ({PDMD['license']}, {PDMD['repo']}):")
    path = download_lora(f"{PDMD['repo']}/{PDMD['file']}", dest=PDMD["dir"], log=log)
    if _sha256(path) != PDMD["sha256"]:
        raise SystemExit(f"{path} is not the pDMD release h3metal was checked with (its SHA-256 differs); "
                         "pass the file you want to merge as --pdmd FILE")
    return path


def _json(url: str, attempts: int = 5):
    """A JSON API answer. Connections that fail on the way (resets, TLS errors, timeouts), 429 and 5xx answers are
    retried with growing pauses; other HTTP errors (401, 404: the repository or revision does not exist) are not."""
    req = urllib.request.Request(url, headers=UA)
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or i == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == attempts - 1:
                raise SystemExit(f"cannot reach {url.split('/')[2]}: {getattr(e, 'reason', e)} "
                                 f"(tried {attempts} times; run the command again to resume)") from e
        time.sleep(2 ** (i + 1))


def list_files(source: str, repo: str | None = None, revision: str = "main"):
    """[(path, size, sha256 or None, url)] of the whole repository (Hugging Face: at that revision)."""
    repo = repo or REPOS[source]
    out = []
    if source == "modelscope":
        d = _json(f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true")
        if d.get("Code") != 200:
            raise SystemExit(f"ModelScope: cannot list {repo}: {d.get('Message')}")
        for f in d["Data"]["Files"]:
            if f.get("Type") != "tree":
                url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{f['Path']}"
                out.append((f["Path"], int(f["Size"]), f.get("Sha256") or None, url))
    else:
        d = _json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
        for s in d.get("siblings", []):
            sha = (s.get("lfs") or {}).get("sha256")
            out.append((s["rfilename"], int(s.get("size") or 0), sha,
                        f"https://huggingface.co/{repo}/resolve/{revision}/{s['rfilename']}"))
    return out


def needed(files, index: dict | None):
    """The subset h3metal reads: see WANTED, plus the text-encoder shards holding layers < 50 or the vision tower."""
    keep = [f for f in files if any(fnmatch.fnmatch(f[0], p) for p in WANTED)]
    if index:
        shards = set()
        for k, shard in index["weight_map"].items():
            layer = None
            if ".layers." in k and "language_model" in k:
                layer = int(k.split(".layers.")[1].split(".")[0])
            if "embed_tokens" in k or "visual" in k or (layer is not None and layer < N_LAYERS):
                shards.add(shard)
        keep += [f for f in files if f[0].startswith("text_encoder/") and os.path.basename(f[0]) in shards]
    return sorted(set(keep), key=lambda f: f[1])


def _fetch(item, dest: str) -> tuple[str, bool, str]:
    path, size, sha, url = item
    target = os.path.join(dest, path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) == size:
        return path, True, "present"
    part = target + ".part"
    for attempt in range(8):
        cmd = ["curl", "-sS", "-L", "--fail", "--connect-timeout", "20", "--speed-limit", "20000",
               "--speed-time", "60", "-C", "-", "-o", part, url]
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in url:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got == size:
            break
        if got > size:
            os.remove(part)
        time.sleep(2 + attempt * 3)
    else:
        return path, False, "incomplete"
    if sha:
        if _sha256(part) != sha:
            os.remove(part)
            return path, False, "sha256 mismatch"
    os.replace(part, target)
    return path, True, "ok"


def _fetch_all(todo, dest: str, jobs: int, log, what: str) -> bool:
    total = sum(f[1] for f in todo)
    log(f"{what} -> {dest}: {len(todo)} files, {total / 1e9:.1f} GB")
    t0, ok = time.time(), True
    with ThreadPoolExecutor(jobs) as ex:
        for path, good, msg in ex.map(lambda it: _fetch(it, dest), todo):
            log(f"  {'ok ' if good else 'ERR'} {path} ({msg})")
            ok &= good
    log(f"done in {time.time() - t0:.0f} s")
    return ok


class NotPublished(Exception):
    """The converted weights' repository or tag does not exist (yet)."""


def download_converted(dest: str, model: str, jobs: int = 4, log=print) -> bool:
    """h3metal's converted weights for model, at the tag this version reads. Raises NotPublished if Hugging Face
    has no such repository or tag; network errors propagate."""
    c = CONVERTED[model]
    try:
        listing = list_files("huggingface", c["repo"], c["revision"])
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):  # Hugging Face answers 401 for a repository that does not exist
            raise NotPublished(f"{c['repo']}@{c['revision']} is not published") from e
        raise
    files = [f for f in listing if not f[0].startswith(".")]
    if not any(f[0].endswith(".h3m") for f in files):
        raise SystemExit(f"{c['repo']}@{c['revision']} holds no weight files")
    return _fetch_all(sorted(files, key=lambda f: f[1]), dest, jobs, log, f"{c['repo']}@{c['revision']}")


def download(dest: str, source: str = "huggingface", jobs: int = 4, log=print, model: str = "minimax-h3") -> bool:
    """The official checkpoint's files that h3metal converts from."""
    repo = OTHER[model][source] if model in OTHER else REPOS[source]
    files = list_files(source, repo)
    if model in OTHER:
        skip = OTHER[model].get("exclude", [])
        todo = sorted((f for f in files if not any(fnmatch.fnmatch(f[0], p) for p in skip)), key=lambda f: f[1])
    else:
        idx_item = [f for f in files if f[0] == "text_encoder/model.safetensors.index.json"]
        index = None
        if idx_item:
            _fetch(idx_item[0], dest)
            with open(os.path.join(dest, idx_item[0][0])) as f:
                index = json.load(f)
        todo = needed(files, index)
    return _fetch_all(todo, dest, jobs, log, f"{repo} [{source}]")


def minimax_plan(source: str = "huggingface", dest: str | None = None):
    """MiniMax-H3-Image's checkpoint for converting while downloading: (the small files: configs, tokenizer, the
    VAE, the per-step heads; the text encoder's shards that h3metal reads, vision tower first, then by layer; the
    transformer's shards by block). dest receives the text encoder's index, which says what its shards hold."""
    files = list_files(source)
    idx_item = [f for f in files if f[0] == "text_encoder/model.safetensors.index.json"]
    if not idx_item:
        raise SystemExit("the checkpoint has no text-encoder index")
    if dest:
        _fetch(idx_item[0], dest)
        with open(os.path.join(dest, idx_item[0][0])) as f:
            index = json.load(f)
    else:
        index = None
    keep = needed(files, index)
    is_shard = lambda f: fnmatch.fnmatch(f[0], "*/model-*.safetensors")  # noqa: E731
    small = [f for f in keep if not is_shard(f)]
    visual = {s for k, s in (index or {}).get("weight_map", {}).items() if "visual" in k}
    te = sorted((f for f in keep if is_shard(f) and f[0].startswith("text_encoder/")),
                key=lambda f: (os.path.basename(f[0]) not in visual, f[0]))
    dit = sorted((f for f in keep if is_shard(f) and f[0].startswith("transformer/")), key=lambda f: f[0])
    return small, te, dit


LORA_DIR = os.path.expanduser("~/.h3metal/loras")


def download_lora(spec: str, source: str = "huggingface", dest: str = LORA_DIR, log=print) -> str:
    """A LoRA adapter from a model repository: "owner/name" (its only .safetensors file) or "owner/name/file".
    -> the local path, ~/.h3metal/loras/<file> (the generate command's --lora finds it by name)."""
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise SystemExit(f"--lora takes owner/name or owner/name/file.safetensors, not {spec!r}")
    repo, want = "/".join(parts[:2]), "/".join(parts[2:]) or None
    files = [f for f in list_files(source, repo) if f[0].endswith(".safetensors")]
    if want:
        files = [f for f in files if f[0] == want]
    elif len(files) > 1:
        names = ", ".join(f[0] for f in files)
        raise SystemExit(f"{repo} has several .safetensors files ({names}): name one, as {repo}/<file>")
    if not files:
        raise SystemExit(f"{repo}: no {want or '.safetensors file'} there")
    path, size, sha, url = files[0]
    name = os.path.basename(path)
    os.makedirs(dest, exist_ok=True)
    if not _fetch_all([(name, size, sha, url)], dest, 1, log, f"{repo} [{source}]"):
        raise SystemExit("the download did not complete; run the command again to resume")
    return os.path.join(dest, name)


if __name__ == "__main__":
    sys.exit(0 if download(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "huggingface") else 1)
