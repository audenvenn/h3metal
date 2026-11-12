"""The .h3m weight file: one file per model component, mapped read-only and
handed to the GPU without copying.

Layout:
    magic b"H3MW", u32 version, u64 header length, JSON header (UTF-8),
    zero padding up to DATA_START, tensor data.

Every tensor starts on a 256-byte boundary and never crosses a multiple of
CHUNK bytes, so the file can be wrapped as Metal buffers of CHUNK bytes each
(a buffer per chunk; Metal limits the size of one buffer). The header maps
names to entries:

    plain tensor:   {"offset", "shape", "dtype"}           dtype: f32 f16 bf16 i8 u8 i32
    linear layer:   {"kind": "linear", "fmt", "N", "K", "G", "w": {...}, "sw": {...}, "bias": {...}}
                    with the formats of kernels/gemm.metal (q8, q4, q4g, f16, bf16)

and carries a "meta" object (model config, conversion settings).
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

from .gpu import Gpu, Mapping, Tensor
from .ops import QWeight

MAGIC = b"H3MW"
VERSION = 1
DATA_START = 8 << 20      # header region reserved at the start of the file
CHUNK = 2 << 30           # Metal buffer per 2 GiB of file
ALIGN = 256

_ITEM = {"f32": 4, "f16": 2, "bf16": 2, "i8": 1, "u8": 1, "i32": 4, "u16": 2}


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns, round to nearest even."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Writer:
    """Writes tensors one by one, so a converter never holds a whole model in memory."""

    def __init__(self, path: str, meta: dict):
        self.path = path
        self.tmp = path + ".partial"
        self.f = open(self.tmp, "wb")
        self.f.truncate(DATA_START)
        self.pos = DATA_START
        self.entries: dict[str, dict] = {}
        self.meta = meta

    def _place(self, nbytes: int) -> int:
        pos = (self.pos + ALIGN - 1) // ALIGN * ALIGN
        if nbytes > CHUNK:
            raise ValueError(f"tensor of {nbytes} bytes exceeds the {CHUNK}-byte chunk")
        if pos // CHUNK != (pos + nbytes - 1) // CHUNK:
            pos = (pos // CHUNK + 1) * CHUNK
        return pos

    def _write(self, data: np.ndarray) -> dict:
        raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        pos = self._place(raw.nbytes)
        self.f.seek(pos)
        self.f.write(raw.tobytes() if raw.nbytes < (64 << 20) else memoryview(raw))
        self.pos = pos + raw.nbytes
        return {"offset": pos, "nbytes": int(raw.nbytes)}

    def tensor(self, name: str, data: np.ndarray, dtype: str):
        if name in self.entries:
            raise KeyError(f"duplicate tensor {name}")
        e = self._write(data)
        e.update({"shape": list(data.shape), "dtype": dtype})
        self.entries[name] = e

    def linear(self, name: str, fmt: str, N: int, K: int, w: np.ndarray, sw: np.ndarray | None,
               bias: np.ndarray | None = None, G: int = 128, extra: dict | None = None):
        e = {"kind": "linear", "fmt": fmt, "N": N, "K": K, "G": G, "w": self._write(w)}
        if sw is not None:
            e["sw"] = self._write(np.ascontiguousarray(sw, dtype=np.float32))
        if bias is not None:
            e["bias"] = self._write(np.ascontiguousarray(bias, dtype=np.float32))
        if extra:
            e.update(extra)
        self.entries[name] = e

    def close(self):
        header = json.dumps({"meta": self.meta, "tensors": self.entries}, separators=(",", ":")).encode()
        if 16 + len(header) > DATA_START:
            raise ValueError("header does not fit the reserved region")
        self.f.seek(0)
        self.f.write(MAGIC + struct.pack("<IQ", VERSION, len(header)) + header)
        end = (self.pos + 16383) // 16384 * 16384
        self.f.truncate(end)
        self.f.close()
        os.replace(self.tmp, self.path)


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not an h3metal weight file")
        version, n = struct.unpack("<IQ", f.read(12))
        if version != VERSION:
            raise ValueError(f"{path}: unsupported weight file version {version}")
        return json.loads(f.read(n))


class WeightFile:
    """A mapped .h3m file: tensors and linear layers as GPU views."""

    def __init__(self, gpu: Gpu, path: str):
        self.path = path
        hdr = read_header(path)
        self.meta = hdr["meta"]
        self.entries = hdr["tensors"]
        self.map = Mapping(gpu, path, CHUNK)

    def __contains__(self, name):
        return name in self.entries

    def tensor(self, name: str) -> Tensor:
        e = self.entries[name]
        return self.map.tensor(e["offset"], tuple(e["shape"]), e["dtype"])

    def host(self, name: str) -> np.ndarray:
        """The tensor as a read-only NumPy view of the mapping."""
        e = self.entries[name]
        dt = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16, "i8": np.int8, "u8": np.uint8,
              "i32": np.int32}[e["dtype"]]
        raw = self.map.host(e["offset"], e["nbytes"])
        return raw.view(dt).reshape(e["shape"])

    def linear(self, name: str) -> QWeight:
        e = self.entries[name]
        fmt, N, K = e["fmt"], e["N"], e["K"]
        wdt = {"q8": "i8", "q4": "u8", "q4g": "u8", "f16": "f16", "bf16": "bf16", "h8": "i8", "h4": "u8", "h4g": "u8"}[fmt]
        wshape = (N, K // 2) if fmt in ("q4", "q4g", "h4", "h4g") else (N, K)
        w = self.map.tensor(e["w"]["offset"], wshape, wdt)
        sw = self.map.tensor(e["sw"]["offset"], (e["sw"]["nbytes"] // 4,), "f32") if "sw" in e else None
        bias = self.map.tensor(e["bias"]["offset"], (N,), "f32") if "bias" in e else None
        return QWeight(fmt, w, sw, N, K, e.get("G", 128), bias)

    def prefetch(self, names=None):
        """Ask the kernel to read ahead the given entries (all of them by default)."""
        for name in (names or self.entries):
            e = self.entries[name]
            parts = [e] if "offset" in e else [e[k] for k in ("w", "sw", "bias") if k in e]
            for p in parts:
                self.map.advise(p["offset"], p["nbytes"], 1)

    def close(self):
        self.map.close()
