"""Python side of the GPU runtime (src/capi.mm): buffers, file mappings, dispatches.

The model is written against this module. A `Tensor` is a view of a GPU
buffer (buffer handle, byte offset, shape, dtype); kernels are launched with
`Gpu.launch`, which records them into the open command buffer. Nothing runs
until `Gpu.wait()` (or `Gpu.commit()`).
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_library() -> str:
    candidates = [
        os.environ.get("H3METAL_LIB", ""),
        os.path.join(_HERE, "libh3metal.dylib"),
        os.path.join(_HERE, "..", "build", "cmake", "libh3metal.dylib"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("libh3metal.dylib not found; build it with cmake (see README)")


def _find_kernels() -> str:
    for c in (os.environ.get("H3METAL_KERNELS", ""), os.path.join(_HERE, "kernels"),
              os.path.join(_HERE, "..", "kernels")):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Metal kernel sources not found")


class _Arg(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_void_p), ("offset", ctypes.c_uint64)]


def _load():
    lib = ctypes.CDLL(_find_library())
    vp, u64, i32, cp = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p
    sig = {
        "h3_last_error": (cp, []),
        "h3_open": (vp, [cp]),
        "h3_close": (None, [vp]),
        "h3_device_name": (cp, [vp]),
        "h3_recommended_working_set": (u64, [vp]),
        "h3_max_buffer_length": (u64, [vp]),
        "h3_buffer": (vp, [vp, u64]),
        "h3_buffer_contents": (vp, [vp]),
        "h3_buffer_length": (u64, [vp]),
        "h3_buffer_release": (None, [vp]),
        "h3_map_open": (vp, [vp, cp, u64]),
        "h3_map_size": (u64, [vp]),
        "h3_map_chunks": (i32, [vp]),
        "h3_map_chunk": (vp, [vp, i32]),
        "h3_map_data": (vp, [vp]),
        "h3_map_advise": (i32, [vp, u64, u64, i32]),
        "h3_map_close": (None, [vp]),
        "h3_pipeline": (vp, [vp, cp, cp, cp]),
        "h3_pipeline_max_threads": (i32, [vp]),
        "h3_dispatch": (i32, [vp, vp, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                              ctypes.POINTER(_Arg), i32, vp, ctypes.c_uint32, i32, ctypes.c_uint32]),
        "h3_commit": (None, [vp]),
        "h3_wait": (ctypes.c_double, [vp]),
        "h3_set_profile": (None, [vp, i32]),
        "h3_profile_take": (i32, [vp, ctypes.POINTER(ctypes.c_double), i32]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args
    return lib


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


class GpuError(RuntimeError):
    pass


def _err() -> str:
    return lib().h3_last_error().decode()


_NP = {
    "f32": np.float32, "f16": np.float16, "bf16": np.uint16, "i8": np.int8, "u8": np.uint8,
    "i32": np.int32, "u32": np.uint32, "i16": np.int16, "u16": np.uint16,
}
_SIZE = {"f32": 4, "f16": 2, "bf16": 2, "i8": 1, "u8": 1, "i32": 4, "u32": 4, "i16": 2, "u16": 2, "q4": 0.5}


@dataclass
class Buffer:
    handle: int
    nbytes: int
    owner: object = None  # keeps a file mapping alive

    def __del__(self):
        if self.owner is None and self.handle and _lib is not None:
            _lib.h3_buffer_release(self.handle)
            self.handle = 0


@dataclass
class Tensor:
    """A typed view of (part of) a GPU buffer."""
    buf: Buffer
    offset: int
    shape: tuple
    dtype: str
    strides: tuple = field(default=None)  # in elements; None = contiguous

    @property
    def nbytes(self) -> int:
        return int(math.prod(self.shape) * _SIZE[self.dtype])

    def arg(self):
        return (self.buf.handle, self.offset)

    def numpy(self) -> np.ndarray:
        """A NumPy view of the data (valid after the GPU work that writes it has finished)."""
        ptr = lib().h3_buffer_contents(self.buf.handle)
        if not ptr:
            raise GpuError("buffer has no CPU-visible contents")
        if self.dtype == "q4":
            raw = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(ptr + self.offset))
            return raw
        n = int(math.prod(self.shape))
        arr = np.ctypeslib.as_array((ctypes.c_uint8 * (n * _SIZE[self.dtype])).from_address(ptr + self.offset))
        return arr.view(_NP[self.dtype]).reshape(self.shape)

    def view(self, shape, dtype=None, offset_elems=0):
        dtype = dtype or self.dtype
        return Tensor(self.buf, self.offset + int(offset_elems * _SIZE[self.dtype]), tuple(shape), dtype)

    def row(self, start: int, count: int):
        """Rows [start, start + count) of a 2-D (or higher) contiguous tensor."""
        per = int(math.prod(self.shape[1:]))
        return Tensor(self.buf, self.offset + int(start * per * _SIZE[self.dtype]), (count,) + tuple(self.shape[1:]),
                      self.dtype)


def has_matrix_units(device: str) -> bool:
    """Whether the GPU has the matrix units (neural accelerators) h3metal's products are built for: M5 and later."""
    import re
    m = re.search(r"\bApple M(\d+)", device)
    return bool(m) and int(m.group(1)) >= 5


class Kernel:
    """A compiled pipeline plus its launch geometry helpers."""

    def __init__(self, gpu: "Gpu", file: str, fn: str, macros: dict | None = None):
        ms = ";".join(f"{k}={v}" for k, v in sorted((macros or {}).items()))
        self.handle = lib().h3_pipeline(gpu.ctx, file.encode(), fn.encode(), ms.encode())
        if not self.handle:
            raise GpuError(_err())
        self.max_threads = lib().h3_pipeline_max_threads(self.handle)
        self.name = f"{file}:{fn}({ms})"


class Gpu:
    def __init__(self, kernel_dir: str | None = None):
        self.ctx = lib().h3_open((kernel_dir or _find_kernels()).encode())
        if not self.ctx:
            raise GpuError(_err())
        self.device = lib().h3_device_name(self.ctx).decode()
        self.working_set = lib().h3_recommended_working_set(self.ctx)
        self.max_buffer = lib().h3_max_buffer_length(self.ctx)
        self._kernels = {}
        self.names = []  # kernel names of the dispatches recorded in profile mode

    def kernel(self, file: str, fn: str, **macros) -> Kernel:
        key = (file, fn, tuple(sorted(macros.items())))
        k = self._kernels.get(key)
        if k is None:
            k = Kernel(self, file, fn, macros)
            self._kernels[key] = k
        return k

    # -- memory --------------------------------------------------------------------------------------------------
    def buffer(self, nbytes: int) -> Buffer:
        h = lib().h3_buffer(self.ctx, max(1, int(nbytes)))
        if not h:
            raise GpuError(_err())
        return Buffer(h, int(nbytes))

    def empty(self, shape, dtype="f32") -> Tensor:
        shape = tuple(int(s) for s in shape)
        nbytes = int(math.ceil(math.prod(shape) * _SIZE[dtype]))
        return Tensor(self.buffer(nbytes), 0, shape, dtype)

    def upload(self, arr: np.ndarray, dtype: str | None = None) -> Tensor:
        if dtype is None:
            dtype = {np.dtype(np.float32): "f32", np.dtype(np.float16): "f16", np.dtype(np.int8): "i8",
                     np.dtype(np.uint8): "u8", np.dtype(np.int32): "i32", np.dtype(np.uint32): "u32",
                     np.dtype(np.uint16): "bf16", np.dtype(np.int16): "i16"}[arr.dtype]
        t = self.empty(arr.shape, dtype)
        t.numpy()[...] = np.ascontiguousarray(arr).view(t.numpy().dtype).reshape(t.shape)
        return t

    # -- execution -----------------------------------------------------------------------------------------------
    def launch(self, k: Kernel, grid, threads, args, params: bytes = b"", params_index: int | None = None,
               tg_mem: int = 0):
        g = (ctypes.c_uint32 * 3)(*(list(grid) + [1, 1, 1])[:3])
        t = (ctypes.c_uint32 * 3)(*(list(threads) + [1, 1, 1])[:3])
        n = len(args)
        arr = (_Arg * max(1, n))()
        for i, a in enumerate(args):
            if a is None:
                arr[i].buf, arr[i].offset = None, 0
            elif isinstance(a, Tensor):
                arr[i].buf, arr[i].offset = a.buf.handle, a.offset
            else:  # (Buffer, offset)
                arr[i].buf, arr[i].offset = a[0].handle, a[1]
        pi = n if params_index is None else params_index
        rc = lib().h3_dispatch(self.ctx, k.handle, g, t, arr, n, params or None, len(params), pi, tg_mem)
        if rc:
            raise GpuError(_err())
        if self._profile:
            self.names.append(k.name)

    _profile = False

    def commit(self):
        lib().h3_commit(self.ctx)

    def wait(self) -> float:
        dt = lib().h3_wait(self.ctx)
        if dt < 0:
            raise GpuError(_err())
        return dt

    def profile(self, on: bool):
        lib().h3_set_profile(self.ctx, 1 if on else 0)
        self._profile = on
        self.names = []

    def profile_take(self):
        out = (ctypes.c_double * 100000)()
        n = lib().h3_profile_take(self.ctx, out, 100000)
        names, self.names = self.names[:n], self.names[n:]
        return list(zip(names, out[:n]))

    def close(self):
        if self.ctx:
            lib().h3_close(self.ctx)
            self.ctx = None


class Mapping:
    """A read-only file mapped into GPU buffers (see src/capi.h)."""

    def __init__(self, gpu: Gpu, path: str, chunk: int):
        self.handle = lib().h3_map_open(gpu.ctx, path.encode(), chunk)
        if not self.handle:
            raise GpuError(_err())
        self.size = lib().h3_map_size(self.handle)
        self.chunk = chunk
        n = lib().h3_map_chunks(self.handle)
        self.chunks = [Buffer(lib().h3_map_chunk(self.handle, i), min(chunk, self.size - i * chunk), owner=self)
                       for i in range(n)]
        self.data = lib().h3_map_data(self.handle)

    def tensor(self, offset: int, shape, dtype) -> Tensor:
        c = offset // self.chunk
        return Tensor(self.chunks[c], offset - c * self.chunk, tuple(shape), dtype)

    def host(self, offset: int, nbytes: int) -> np.ndarray:
        return np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(self.data + offset))

    def advise(self, offset: int, length: int, advice: int):
        lib().h3_map_advise(self.handle, offset, length, advice)

    def close(self):
        if self.handle:
            for c in self.chunks:
                c.handle = 0
            lib().h3_map_close(self.handle)
            self.handle = None


def pack(fmt: str, *vals) -> bytes:
    return struct.pack("<" + fmt, *vals)
