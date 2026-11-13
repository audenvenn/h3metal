"""Minimal safetensors reader: memory-mapped, bfloat16 widened to float32 on read.

NumPy has no bfloat16, and the official checkpoints are mostly bfloat16, so
this reads the format directly instead of going through a framework.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

_DT = {"F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2), "I8": (np.int8, 1),
       "U8": (np.uint8, 1), "I32": (np.int32, 4), "I64": (np.int64, 8), "F64": (np.float64, 8)}


class SafeTensors:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        self.meta = hdr.pop("__metadata__", {})
        self.info = hdr
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self):
        return self.info.keys()

    def dtype(self, name: str) -> str:
        return self.info[name]["dtype"]

    def shape(self, name: str) -> tuple:
        return tuple(self.info[name]["shape"])

    def raw(self, name: str) -> np.ndarray:
        """The stored values; bfloat16 as uint16 bit patterns."""
        e = self.info[name]
        dt, _ = _DT[e["dtype"]]
        a, b = e["data_offsets"]
        return np.frombuffer(self.mm, dtype=dt, count=(b - a) // np.dtype(dt).itemsize,
                             offset=self.base + a).reshape(e["shape"])

    def get(self, name: str) -> np.ndarray:
        """The tensor as float32 (floating types) or its stored integer type."""
        r = self.raw(name)
        if self.info[name]["dtype"] == "BF16":
            return (r.astype(np.uint32) << 16).view(np.float32)
        if r.dtype in (np.float16, np.float64):
            return r.astype(np.float32)
        return np.array(r)


class Sharded:
    """A directory of safetensors shards with a `*.index.json` weight map (or a single file)."""

    def __init__(self, directory: str):
        self.files: dict[str, SafeTensors] = {}
        self.where: dict[str, str] = {}
        idx = [f for f in os.listdir(directory) if f.endswith(".index.json")]
        if idx:
            with open(os.path.join(directory, idx[0])) as f:
                wm = json.load(f)["weight_map"]
            for k, v in wm.items():
                self.where[k] = v
        else:
            for fn in sorted(os.listdir(directory)):
                if fn.endswith(".safetensors"):
                    st = SafeTensors(os.path.join(directory, fn))
                    for k in st.keys():
                        self.where[k] = fn
        self.dir = directory

    def _file(self, name: str) -> SafeTensors:
        fn = self.where[name]
        if fn not in self.files:
            self.files[fn] = SafeTensors(os.path.join(self.dir, fn))
        return self.files[fn]

    def keys(self):
        return self.where.keys()

    def __contains__(self, name):
        return name in self.where

    def get(self, name: str) -> np.ndarray:
        return self._file(name).get(name)

    def raw(self, name: str) -> np.ndarray:
        return self._file(name).raw(name)

    def dtype(self, name: str) -> str:
        return self._file(name).dtype(name)

    def shape(self, name: str) -> tuple:
        return self._file(name).shape(name)


def save(path: str, tensors: dict):
    """Writes NumPy arrays as a safetensors file (float32, float16, int8, uint8, int32, int64, float64)."""
    names = {np.dtype(v): k for k, (v, _) in _DT.items() if k != "BF16"}
    hdr, off, blobs = {}, 0, []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        hdr[name] = {"dtype": names[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)
