"""torch.randn on the CPU, reproduced with NumPy.

The official pipeline draws its noise with torch's CPU generator. Drawing the
same numbers makes h3metal's images directly comparable with the reference
for the same seed, without depending on torch. torch's CPU generator is
MT19937 seeded with the low 32 bits of the seed; float32 normals are filled
16 at a time: 16 uniforms u (24-bit), then Box-Muller on pairs (u[i], u[i+8]).
A tail of n % 16 elements is filled by redoing the last 16 of the tensor.
Sizes below 16 take a different path in torch and are not supported here.
"""

from __future__ import annotations

import numpy as np


class TorchRandn:
    def __init__(self, seed: int):
        bg = np.random.MT19937()
        bg._legacy_seeding(int(seed) & 0xFFFFFFFF)
        self.bg = bg

    def _uniform(self, n: int) -> np.ndarray:
        raw = self.bg.random_raw(n).astype(np.uint32)
        return ((raw & np.uint32((1 << 24) - 1)).astype(np.float32) * np.float32(2.0 ** -24))

    def normal(self, shape) -> np.ndarray:
        n = int(np.prod(shape))
        if n < 16:
            raise ValueError("torch draws fewer than 16 normals differently")
        out = self._uniform(n)
        full = n // 16 * 16

        def f32(x):
            return np.asarray(x, dtype=np.float64).astype(np.float32)

        def box_muller(chunk):  # chunk [k][16]; every operation rounded to float32 where torch rounds
            u1 = np.float32(1) - chunk[:, :8]
            u2 = chunk[:, 8:]
            radius = f32(np.sqrt(f32(np.float32(-2) * f32(np.log(u1.astype(np.float64))))))
            theta = f32(2.0 * np.pi * u2.astype(np.float64))
            c = f32(np.cos(theta.astype(np.float64)))
            s = f32(np.sin(theta.astype(np.float64)))
            return np.concatenate([radius * c, radius * s], axis=1).astype(np.float32)

        out[:full] = box_muller(out[:full].reshape(-1, 16)).reshape(-1)
        if n != full:
            tail = self._uniform(16)
            out[n - 16:] = tail
            out[n - 16:] = box_muller(out[n - 16:].reshape(1, 16)).reshape(-1)
        return out.reshape(shape)
