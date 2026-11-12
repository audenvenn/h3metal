"""Conversion numerics and file format, on the CPU only (runs in CI)."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal import quant  # noqa: E402
from h3metal.convert import schedule, timestep_embedding  # noqa: E402
from h3metal.dit import Layout, canvas_size, euler, patchify, unpatchify  # noqa: E402
from h3metal.edit import edit_positions, vision_patches  # noqa: E402
from h3metal.vae import split_tiles  # noqa: E402
from h3metal.weights import Writer, bf16_bits, bf16_to_f32, read_header  # noqa: E402


def test_rotation_preserves_products():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 512)).astype(np.float32)
    x = rng.standard_normal((8, 512)).astype(np.float32)
    s = quant.random_signs(512, 3)
    wr = quant.rotate_in(w, s)
    xr = quant.rotate_in(x, s)  # the activations are rotated the same way
    assert np.allclose(xr @ wr.T, x @ w.T, atol=1e-3)


def test_int4_pack_round_trip():
    rng = np.random.default_rng(1)
    q = rng.integers(-8, 8, size=(5, 64))
    assert np.array_equal(quant.unpack_int4(quant.pack_int4(q)), q)


def test_quantizers():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((256, 1024)).astype(np.float32)
    for fmt, tol in (("q8", 0.01), ("q4", 0.15), ("q4g", 0.12)):
        codes, scales = quant.quantize(fmt, w)
        back = quant.dequant(fmt, codes, scales, 256, 1024)
        err = np.linalg.norm(back - w) / np.linalg.norm(w)
        assert err < tol, (fmt, err)


def test_bf16():
    x = np.array([1.0, -2.5, 3.14159265, 1e-3, 65504.0], np.float32)
    back = bf16_to_f32(bf16_bits(x))
    assert np.all(np.abs(back - x) <= np.abs(x) * 2 ** -8)


def test_weight_file_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.h3m")
        w = Writer(path, {"kind": "test"})
        a = np.arange(1000, dtype=np.float32)
        w.tensor("a", a, "f32")
        codes, scales = quant.quantize("q8", np.ones((128, 256), np.float32))
        w.linear("lin", "q8", 128, 256, codes, scales)
        w.close()
        hdr = read_header(path)
        assert hdr["meta"]["kind"] == "test"
        e = hdr["tensors"]["a"]
        assert e["offset"] % 256 == 0
        data = np.memmap(path, dtype=np.uint8, mode="r")
        got = np.frombuffer(data, np.float32, count=1000, offset=e["offset"])
        assert np.array_equal(got, a)


def test_schedule():
    sig = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
    t, s = schedule(sig)
    assert len(t) == 6 and len(s) == 7 and s[-1] == 0.0 and t[0] == 0.0
    assert np.all(np.diff(s) < 0)
    assert np.allclose(t, 1 - s[:-1])


def test_timestep_embedding():
    e = timestep_embedding(np.array([0.0, 0.5], np.float32))
    assert e.shape == (2, 256)
    assert np.allclose(e[0, :128], 1.0) and np.allclose(e[0, 128:], 0.0)  # [cos, sin] at t = 0
    assert np.isclose(e[1, 0], np.cos(0.5)) and np.isclose(e[1, 128], np.sin(0.5))


def test_layout_and_patches():
    lay = Layout(5, np.ones(5, np.uint8), [(4, 6)], 8, 8)
    assert lay.n_ref_rows == 6 and lay.n_img_rows == 16 and lay.length == 27
    pos = lay.positions()
    assert np.array_equal(pos[:5, 0], np.arange(5)) and np.all(pos[5:11, 0] == 5) and np.all(pos[11:, 0] == 6)
    assert np.allclose(pos[11:, 1].reshape(4, 4)[:, 0], [0, 8, 16, 24])  # a square canvas spans [0, 32)
    rt = lay.row_types()
    assert np.all(rt[:5] == 1) and np.all(rt[5:11] == 2) and np.all(rt[11:] == 0)
    lat = np.random.default_rng(3).standard_normal((24, 8, 10)).astype(np.float32)
    assert np.array_equal(unpatchify(patchify(lat), 24, 8, 10), lat)
    x = np.ones((4, 96), np.float32)
    assert np.allclose(euler(x, x, 0, [1.0, 0.75, 0.0]), 1.25)
    assert canvas_size(1024, 1024) == (2048, 2048) and canvas_size(1024, 1024, 1024) == (1024, 1024)


def test_vae_tiles():
    starts, sizes, overlaps = split_tiles(1024)
    assert starts == [0, 192, 384, 576, 768] and overlaps == [64, 64, 64, 64]
    assert split_tiles(200) == ([0], [200], [])
    s, _, o = split_tiles(4096)
    assert len(s) == 21 and s[-1] + 256 == 4096


def test_edit_positions_and_patches():
    # 3 text tokens, a 2x2 merged image (4 x 4 patches), 2 text tokens
    mm = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0])
    pos = edit_positions(mm, [(1, 4, 4)])
    assert np.array_equal(pos[:, :3], np.tile(np.arange(3), (3, 1)))
    assert np.array_equal(pos[0, 3:7], [3, 3, 3, 3])
    assert np.array_equal(pos[1, 3:7], [3, 3, 4, 4]) and np.array_equal(pos[2, 3:7], [3, 4, 3, 4])
    assert np.array_equal(pos[:, 7:], np.tile([5, 6], (3, 1)))
    img = np.zeros((32, 64, 3), np.uint8)
    p, grid = vision_patches(img)
    assert grid == (1, 2, 4) and p.shape == (8, 3 * 2 * 16 * 16) and np.allclose(p, -1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
