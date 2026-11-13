"""LoRA adapters on h3metal's converted models.

An adapter changes a linear layer W of the original model to W + scale * B A, with A [r][in] and B [out][r].
h3metal applies it as exactly that, a low-rank term, instead of merging it into the stored weights: a merge
would round the int8 weights a second time and would have to be redone for every strength. On the GPU the term
runs inside the layer's own product (kernels/gemm.metal, LORA): the adapter's input side is a narrow product of
the same activations, and its output side a short second product on the matrix units, added to the result
before it is stored. Layers the CPU computes (the modulation of both models) take the term in float.

Converted layers are the original ones transformed: rows permuted for the rotary embedding, signs flipped,
products fused, inputs rotated (h3metal/quant.py). Each model describes those transforms once, as plans
(LinearPlan, below) that its converter and this module both use, so an adapter goes through exactly the
transforms its layer went through.

Files: the PEFT/diffusers layout (lora_A, lora_B), kohya's (lora_down, lora_up, alpha, module names with
underscores), and the original checkpoints' module names (ai-toolkit, ComfyUI, fal), with the prefixes those
tools add. A file with anything h3metal cannot apply exactly (DoRA, LoKr, LoHa, weight differences, layers
h3metal does not keep, the text encoder) is refused with a list of what it cannot apply, never applied in part.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .quant import rotate_in
from .st import SafeTensors


@dataclass
class Part:
    """Rows of a linear layer of the original model (diffusers names), as they appear in a converted layer."""
    module: str
    n: int                           # rows taken
    rows: np.ndarray | None = None   # which output rows, in order (None: all n of them)
    sign: np.ndarray | None = None   # a sign per taken row
    cols: tuple | None = None        # input columns [c0, c1) (None: all)

    def take(self, w: np.ndarray) -> np.ndarray:
        """The converted rows of a weight [out][in] of the module (or of an adapter's B [out][r], cols None)."""
        if self.cols is not None:
            w = w[:, self.cols[0]:self.cols[1]]
        if self.rows is not None:
            w = w[self.rows]
        if self.sign is not None:
            w = w * self.sign[:, None]
        return w

    def take_rows(self, b: np.ndarray) -> np.ndarray:
        """The converted rows of an adapter's B [out][r] (out: the module's rows from the first)."""
        b = b[self.rows] if self.rows is not None else b[:self.n]
        if self.sign is not None:
            b = b * self.sign[:, None]
        return b

    @property
    def out_rows(self) -> int:
        """Rows the module has at least (its output size, where the part takes all of them)."""
        return int(self.rows.max()) + 1 if self.rows is not None else self.n


@dataclass
class LinearPlan:
    """A converted linear layer: the parts' rows, concatenated in order, with the input rotated by s_in.

    kind: "gpu" (a stored product), "host" (a float matrix the CPU applies) or "table" (Z-Image's modulation,
    precomputed per step; the model applies those itself)."""
    kind: str
    parts: list
    k: int                            # the converted layer's input size
    s_in: np.ndarray | None = None
    rotate: bool = True

    def assemble(self, get) -> np.ndarray:
        """The converted weight, from get(module) -> the original weight [out][in] in float32."""
        cache = {}

        def w(m):
            if m not in cache:
                cache.clear()
                cache[m] = get(m)
            return cache[m]
        mats = [p.take(w(p.module)) for p in self.parts]
        mat = np.concatenate(mats) if len(mats) > 1 else mats[0]
        return rotate_in(mat, self.s_in) if self.rotate else mat


# -- reading files ------------------------------------------------------------------------------------------------

PREFIXES = ("base_model.model.", "diffusion_model.", "transformer.", "model.", "unet.")
UNSUPPORTED = (("dora_scale", "DoRA"), (".magnitude", "DoRA"), ("lokr_", "LoKr"), ("hada_", "LoHa"),
               (".diff_b", "bias differences"), (".diff", "weight differences"), ("lora_mid", "LoCon"),
               (".bias", "bias adapters"))
TEXT_ENCODER = ("lora_te", "text_encoder", "te1_", "te2_", "text_model")


@dataclass
class Layer:
    A: np.ndarray       # [r][in]
    B: np.ndarray       # [out][r]
    scale: float        # alpha / r, or 1 without alpha


class LoraError(ValueError):
    pass


def _split(key: str):
    """-> (module, which) with which in A, B, alpha; None for keys that are not adapter weights."""
    for suf, which in ((".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_down.weight", "A"),
                       (".lora_up.weight", "B"), (".lora.down.weight", "A"), (".lora.up.weight", "B"),
                       (".lora_A.default.weight", "A"), (".lora_B.default.weight", "B"), (".alpha", "alpha")):
        if key.endswith(suf):
            return key[:-len(suf)], which
    return None


def _strip(module: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if module.startswith(p):
                module, changed = module[len(p):], True
    return module


def read(path: str) -> dict:
    """An adapter file -> {module (as written, prefixes removed): Layer}."""
    if not os.path.isfile(path):
        raise LoraError(f"no LoRA file at {path}")
    st = SafeTensors(path)
    parts, bad = {}, []
    for key in st.keys():
        low = key.lower()
        if any(t in low for t in TEXT_ENCODER):
            bad.append(f"{key} (text encoder)")
            continue
        sp = _split(key)
        if sp is None:
            what = next((name for pat, name in UNSUPPORTED if pat in key), "an unknown kind of tensor")
            bad.append(f"{key} ({what})")
            continue
        module, which = sp
        parts.setdefault(_strip(module), {})[which] = key
    if bad:
        raise LoraError(f"{os.path.basename(path)}: h3metal applies plain LoRA layers (A and B matrices) only; "
                        f"it cannot apply {len(bad)} tensors exactly, for example " + "; ".join(bad[:4]))
    out = {}
    for module, d in parts.items():
        if "A" not in d or "B" not in d:
            raise LoraError(f"{os.path.basename(path)}: {module} has no {'B' if 'A' in d else 'A'} matrix")
        A = st.get(d["A"]).astype(np.float32)
        B = st.get(d["B"]).astype(np.float32)
        if A.ndim != 2 or B.ndim != 2:  # convolution-shaped LoRAs (r, in, 1, 1) are 1x1 products
            A, B = A.reshape(A.shape[0], -1), B.reshape(B.shape[0], -1)
        if B.shape[1] != A.shape[0]:
            raise LoraError(f"{os.path.basename(path)}: {module}: A {A.shape} and B {B.shape} disagree")
        scale = 1.0
        if "alpha" in d:
            scale = float(np.asarray(st.get(d["alpha"])).reshape(-1)[0]) / A.shape[0]
        out[module] = Layer(A, B, scale)
    if not out:
        raise LoraError(f"{os.path.basename(path)}: no LoRA layers in the file")
    return out


def shapes(path: str) -> dict | None:
    """The adapter's layers from the file's header alone: {module: (A's shape, B's shape)}, or None for a file
    read() would refuse. Cheap: no tensor is read."""
    try:
        st = SafeTensors(path)
    except Exception:
        return None
    parts = {}
    for key in st.keys():
        if any(t in key.lower() for t in TEXT_ENCODER):
            return None
        sp = _split(key)
        if sp is None:
            return None
        module, which = sp
        parts.setdefault(_strip(module), {})[which] = st.shape(key)
    out = {}
    for module, d in parts.items():
        if "A" not in d or "B" not in d:
            return None
        a, b = d["A"], d["B"]
        out[module] = ((a[0], int(np.prod(a[1:]))), (b[0], int(np.prod(b[1:]))))
    return out or None


def fits(layer_shapes: dict | None, plans: dict, aliases: dict) -> bool:
    """Whether terms() would take an adapter with these layer shapes (from shapes())."""
    if not layer_shapes:
        return False
    index = _index(plans)
    for module, (a, b) in layer_shapes.items():
        targets = resolve(module, index, aliases)
        if not targets:
            return False
        rows = 0
        for real, where in targets:
            n, k = _shape(index, plans, real)
            last = where + n if np.ndim(where) == 0 else int(np.max(where)) + 1
            rows = max(rows, last)
            if k != a[1]:
                return False
        if b[0] != rows or b[1] != a[0]:
            return False
    return True


# -- applying -----------------------------------------------------------------------------------------------------

@dataclass
class Term:
    """An adapter's term on one converted layer: rows += scale * B_e A_e x (in the layer's rotated input space)."""
    A: np.ndarray   # [r][K] (rotated, scaled by strength * scale)
    B: np.ndarray   # [N][r] (the layer's rows; zero elsewhere)


def _index(plans: dict) -> dict:
    """module -> [(plan name, row offset in the converted layer, Part)]."""
    index = {}
    for name, plan in plans.items():
        off = 0
        for p in plan.parts:
            index.setdefault(p.module, []).append((name, off, p))
            off += p.n
    return index


def _shape(index: dict, plans: dict, module: str) -> tuple:
    """(rows, input size) of a module of the original model, from the plans."""
    out = max(p.out_rows for _, _, p in index[module])
    k = max(p.cols[1] if p.cols else plans[name].k for name, _, p in index[module])
    return out, k


def resolve(module: str, index: dict, aliases: dict) -> list:
    """An adapter's module name -> [(module, where its rows are in the adapter's B)] of the plans' modules: the
    index of its first row, or an array with the B row of each of its rows (an alias that reorders rows)."""
    if module in index:
        return [(module, 0)]
    if module in aliases:
        return aliases[module]
    flat = module.replace(".", "_")
    if flat.startswith("lora_unet_"):
        flat = flat[len("lora_unet_"):]
    under = {m.replace(".", "_"): m for m in list(index) + list(aliases)}
    if flat in under and under[flat] != module:
        return resolve(under[flat], index, aliases)
    return []


def terms(adapters, plans: dict, aliases: dict, *, model: str) -> dict:
    """[(adapter layers from read(), strength)] -> {plan name: [Term]}, each layer's terms ordered by their first
    row (narrow rank ranges per column tile). One term per adapter layer and converted layer: an adapter on a
    fused original layer (an alias) stays one term, whatever number of modules it spans."""
    index = _index(plans)
    out = {}
    for layers, strength in adapters:
        missing = []
        for module, lay in layers.items():
            targets = resolve(module, index, aliases)
            if not targets:
                missing.append(module)
                continue
            def b_rows(real, where):
                n = _shape(index, plans, real)[0]
                return np.arange(where, where + n) if np.ndim(where) == 0 else np.asarray(where)
            targets = [(real, b_rows(real, where)) for real, where in targets]
            rows = max(int(r.max()) + 1 for _, r in targets)
            k = {_shape(index, plans, real)[1] for real, _ in targets}
            if lay.B.shape[0] != rows or k != {lay.A.shape[1]}:
                raise LoraError(f"{module}: the LoRA's layer is {lay.B.shape[0]} x {lay.A.shape[1]}, {model}'s is "
                                f"{rows} x {k.pop()}: a LoRA for another model?")
            groups = {}
            for real, rows_in_b in targets:
                for name, off, part in index[real]:
                    groups.setdefault(name, []).append((off, part, rows_in_b))
            for name, entries in groups.items():
                plan = plans[name]
                cols = {e[1].cols for e in entries}
                if len(cols) != 1:
                    raise LoraError(f"{module}: parts of {name} read different inputs")
                c = entries[0][1].cols
                A = lay.A if c is None else lay.A[:, c[0]:c[1]]
                A = (A * np.float32(strength * lay.scale)).astype(np.float32)
                if plan.rotate and plan.s_in is not None:
                    A = rotate_in(A, plan.s_in)
                Be = np.zeros((sum(p.n for p in plan.parts), lay.A.shape[0]), np.float32)
                for off, part, rows_in_b in entries:
                    Be[off:off + part.n] = part.take_rows(lay.B[rows_in_b])
                out.setdefault(name, []).append(Term(A, Be))
        if missing:
            raise LoraError(f"this LoRA has layers that {model} does not have or h3metal does not keep, for example "
                            + ", ".join(missing[:4]) + (f" ({len(missing)} in all)" if len(missing) > 4 else ""))
    for ts in out.values():
        ts.sort(key=lambda t: int(np.argmax(np.abs(t.B).sum(1) > 0)))
    return out


def stack(ts: list) -> tuple:
    """Terms of one layer -> one low-rank pair (A [R][K], B [N][R]) with the ranks side by side."""
    return np.concatenate([t.A for t in ts]), np.concatenate([t.B for t in ts], axis=1)


LORA_DIR = os.path.expanduser("~/.h3metal/loras")


def parse_spec(spec: str, lora_dir: str = LORA_DIR) -> tuple:
    """"path[:strength]" -> (path, strength). A name that is no file is looked up in ~/.h3metal/loras, with or
    without its .safetensors suffix (where `h3metal download --lora` puts adapters)."""
    m = re.match(r"^(.*?)(?::(-?[0-9.]+))?$", spec)
    path, s = os.path.expanduser(m.group(1)), m.group(2)
    if not os.path.exists(path) and os.sep not in path:
        for cand in (path, path + ".safetensors"):
            if os.path.isfile(os.path.join(lora_dir, cand)):
                path = os.path.join(lora_dir, cand)
                break
    return path, float(s) if s is not None else 1.0


class Loaded:
    """A pipeline's adapters: files read once (kept while in use), and the set on its transformer, so that a
    request with the adapters of the previous one costs nothing."""

    def __init__(self):
        self.files = {}      # path -> (key, layers)
        self.applied = None  # the set on the transformer: ((key, strength), ...)

    def read(self, specs) -> list:
        """[(path, strength)] -> [(key, layers, strength)]; files that changed on disk are read again."""
        out, keep = [], {}
        for path, strength in specs or ():
            path = os.path.abspath(os.path.expanduser(path))
            try:
                st = os.stat(path)
            except OSError:
                raise LoraError(f"no LoRA file at {path}") from None
            key = (path, st.st_mtime_ns, st.st_size)
            old = self.files.get(path)
            layers = old[1] if old and old[0] == key else read(path)
            keep[path] = (key, layers)
            out.append((key, layers, float(strength)))
        self.files = keep
        return out

    def apply(self, dit, adapters) -> bool:
        """Puts the adapters on the transformer unless they are on it already. -> whether it changed anything."""
        want = tuple((k, s) for k, _, s in adapters)
        if want == self.applied:
            return False
        try:
            dit.set_loras([(layers, s) for _, layers, s in adapters])
        except Exception:
            dit.set_loras([])
            self.applied = ()
            raise
        self.applied = want
        return True
