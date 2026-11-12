"""`h3metal download`: fetch the files of the official checkpoint that h3metal reads.

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
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPOS = {"huggingface": "MiniMaxAI/MiniMax-H3-Image", "modelscope": "MiniMax/MiniMax-H3-Image"}
# Other models h3metal runs: their repositories, all files needed.
OTHER = {"z-image-turbo": {"huggingface": "Tongyi-MAI/Z-Image-Turbo", "modelscope": "Tongyi-MAI/Z-Image-Turbo"}}
UA = {"User-Agent": "h3metal-download"}
N_LAYERS = 50

WANTED = [
    "LICENSE", "README.md", "model_index.json", "pdd_config.json", "pdd_fused_heads.safetensors",
    "transformer/*", "vae/*", "tokenizer/*", "processor/*", "text_encoder/config.json",
    "text_encoder/*.json", "text_encoder/*.txt", "examples/*",
]


def _json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def list_files(source: str, repo: str | None = None):
    """[(path, size, sha256 or None, url)] of the whole repository."""
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
        d = _json(f"https://huggingface.co/api/models/{repo}/revision/main?blobs=true")
        for s in d.get("siblings", []):
            sha = (s.get("lfs") or {}).get("sha256")
            out.append((s["rfilename"], int(s.get("size") or 0), sha,
                        f"https://huggingface.co/{repo}/resolve/main/{s['rfilename']}"))
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
        h = hashlib.sha256()
        with open(part, "rb") as f:
            while chunk := f.read(8 << 20):
                h.update(chunk)
        if h.hexdigest() != sha:
            os.remove(part)
            return path, False, "sha256 mismatch"
    os.replace(part, target)
    return path, True, "ok"


def download(dest: str, source: str = "huggingface", jobs: int = 4, log=print, model: str = "minimax-h3") -> bool:
    repo = OTHER[model][source] if model in OTHER else REPOS[source]
    files = list_files(source, repo)
    if model in OTHER:
        todo = sorted(files, key=lambda f: f[1])
    else:
        idx_item = [f for f in files if f[0] == "text_encoder/model.safetensors.index.json"]
        index = None
        if idx_item:
            _fetch(idx_item[0], dest)
            with open(os.path.join(dest, idx_item[0][0])) as f:
                index = json.load(f)
        todo = needed(files, index)
    total = sum(f[1] for f in todo)
    log(f"{repo} [{source}] -> {dest}: {len(todo)} files, {total / 1e9:.1f} GB")
    t0, ok = time.time(), True
    with ThreadPoolExecutor(jobs) as ex:
        for path, good, msg in ex.map(lambda it: _fetch(it, dest), todo):
            log(f"  {'ok ' if good else 'ERR'} {path} ({msg})")
            ok &= good
    log(f"done in {time.time() - t0:.0f} s")
    return ok


if __name__ == "__main__":
    sys.exit(0 if download(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "huggingface") else 1)
