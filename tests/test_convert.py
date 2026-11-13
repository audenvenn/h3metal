"""Conversion numerics and file format, on the CPU only (runs in CI)."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal import quant  # noqa: E402
from h3metal.convert import _lora_target, flow_sigmas, lora_deltas, schedule, timestep_embedding  # noqa: E402
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


def test_flow_grids():
    """The few-step grids, bit for bit those of diffusers' MiniMaxH3Scheduler (shift 12, set_timesteps(n + 1)),
    taken from tests/ref/mini_dit_flow.py."""
    want = {2: [0x3f800000, 0x3f6c4ec5, 0],
            3: [0x3f800000, 0x3f75c28f, 0x3f5b6db6, 0],
            4: [0x3f800000, 0x3f7914c2, 0x3f6c4ec5, 0x3f4ccccd, 0]}
    for n, bits in want.items():
        got = flow_sigmas(n, 12.0)
        assert got.dtype == np.float32 and got.view(np.uint32).tolist() == bits, (n, got)
    x = np.ones((3, 4), np.float32)
    v = np.full((3, 4), 0.5, np.float32)
    sig = flow_sigmas(2, 12.0)
    assert np.allclose(euler(x, v, 0, sig, "blend"), euler(x, v, 0, sig), rtol=0, atol=1e-6)
    assert np.allclose(euler(x, v, 1, sig, "blend"), x + sig[1] * v)  # the last step lands on x0


def test_pdmd_lora_mapping():
    """pDMD's LoRA (diffusers' module names) onto the checkpoint's tensors: QKV interleaved per head, the SwiGLU
    halves swapped; the checkpoint's own names unchanged."""
    from h3metal import st
    from h3metal.lora import LoraError
    assert _lora_target("transformer_blocks.7.attn.to_k") == ("blocks.7.attn.qkv_proj.weight", "k")
    assert _lora_target("token_refiner.refiner_blocks.1.ff.net.0.proj") == ("token_refiner.blocks.1.mlp.fc1.weight", "swap")
    assert _lora_target("transformer_blocks.0.attn.to_out.0") == ("blocks.0.attn.out_proj.weight", None)
    assert _lora_target("blocks.3.mlp.fc1") == ("blocks.3.mlp.fc1.weight", None)
    assert _lora_target("proj_out") == ("final_layer.video_out.weight", None)
    H, hd, D, F, r = 2, 3, 4, 5, 2
    cfg = {"num_attention_heads": H, "attention_head_dim": hd, "ffn_hidden_size": F}
    rng = np.random.default_rng(0)
    mats = {f"transformer_blocks.0.attn.to_{x}": (rng.standard_normal((r, D)), rng.standard_normal((H * hd, r)))
            for x in "qkv"}
    mats["transformer_blocks.0.ff.net.0.proj"] = (rng.standard_normal((r, D)), rng.standard_normal((2 * F, r)))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.safetensors")
        st.save(path, {f"transformer.{m}.lora_{w}.weight": np.ascontiguousarray(x, np.float32)
                       for m, (a, b) in mats.items() for w, x in (("A", a), ("B", b))})
        keys = ["blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight"]
        deltas = lora_deltas([(path, 0.5)], cfg, keys)
        qkv = np.zeros((3 * H * hd, D), np.float32)
        for add in deltas[keys[0]]:
            add(qkv)
        want = np.stack([0.5 * mats[f"transformer_blocks.0.attn.to_{x}"][1] @ mats[f"transformer_blocks.0.attn.to_{x}"][0]
                         for x in "qkv"]).reshape(3, H, hd, D).transpose(1, 0, 2, 3).reshape(3 * H * hd, D)
        assert np.allclose(qkv, want, atol=1e-5)
        fc1 = np.zeros((2 * F, D), np.float32)
        for add in deltas[keys[1]]:
            add(fc1)
        a, b = mats["transformer_blocks.0.ff.net.0.proj"]
        dv = 0.5 * b @ a  # diffusers: [value; gate]
        assert np.allclose(fc1, np.concatenate([dv[F:], dv[:F]]), atol=1e-5)
        try:
            lora_deltas([(path, 1.0)], cfg, keys[:1])
            raise AssertionError("a module outside the checkpoint must be refused")
        except LoraError as e:
            assert "not in MiniMax-H3-Image" in str(e)


def test_pdmd_lora_is_pinned():
    """convert --pdmd takes the local LoRA without the network when its SHA-256 is the pinned one, and fetches it
    otherwise."""
    import hashlib
    from h3metal import download
    with tempfile.TemporaryDirectory() as d:
        body = b"not really a LoRA"
        saved = dict(download.PDMD)
        calls = []
        orig = download.download_lora
        download.PDMD.update(dir=d, file="a.safetensors", size=len(body), sha256=hashlib.sha256(body).hexdigest())
        download.download_lora = lambda spec, dest, log: calls.append(spec) or os.path.join(dest, "a.safetensors")
        try:
            with open(os.path.join(d, "a.safetensors"), "wb") as f:
                f.write(body)
            assert download.pdmd_lora(log=lambda *a: None) == os.path.join(d, "a.safetensors") and not calls
            with open(os.path.join(d, "a.safetensors"), "wb") as f:
                f.write(b"something else entirely")
            try:
                download.pdmd_lora(log=lambda *a: None)
                raise AssertionError("a different file must be refused")
            except SystemExit as e:
                assert "SHA-256" in str(e) and calls, e
        finally:
            download.PDMD.clear()
            download.PDMD.update(saved)
            download.download_lora = orig


def test_api_retries():
    """The downloader's API calls ride out a few failed connections and server errors, but not a 404."""
    import io
    import ssl
    import urllib.error
    from h3metal import download
    calls = []

    def flaky(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.URLError(ssl.SSLEOFError(8, "EOF occurred in violation of protocol"))
        if len(calls) == 2:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return io.BytesIO(b'{"ok": 1}')

    orig_open, orig_sleep = download.urllib.request.urlopen, download.time.sleep
    download.urllib.request.urlopen, download.time.sleep = flaky, lambda s: None
    try:
        assert download._json("https://example.invalid/api") == {"ok": 1} and len(calls) == 3

        def missing(req, timeout):
            calls.append(req.full_url)
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
        download.urllib.request.urlopen = missing
        calls.clear()
        try:
            download._json("https://example.invalid/api")
            raise AssertionError("a 404 must not be retried")
        except urllib.error.HTTPError as e:
            assert e.code == 404 and len(calls) == 1
    finally:
        download.urllib.request.urlopen, download.time.sleep = orig_open, orig_sleep


def test_streamed_shards():
    """Converting while downloading: shards arrive in read order (file:// URLs through the real downloader), at most
    three are on disk at once, values are the shards' own, and a read from a deleted shard is refused."""
    import hashlib
    import json
    from h3metal import st
    from h3metal.stream import ShardStream, StreamedSharded
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dest:
        os.makedirs(os.path.join(src, "ckpt"))
        rng = np.random.default_rng(0)
        want, where, files = {}, {}, []
        for i in range(5):  # layer i's two tensors; layer 2 straddles shards 1 and 2
            for part in ("a", "b"):
                want[f"layers.{i}.{part}"] = rng.standard_normal((64, 32)).astype(np.float32)
        names = sorted(want, key=lambda k: (int(k.split(".")[1]), k))
        chunks = [names[:5], names[5:8], names[8:]]
        for j, keys in enumerate(chunks):
            fn = f"model-{j + 1:05d}-of-00003.safetensors"
            path = os.path.join(src, "ckpt", fn)
            st.save(path, {k: want[k] for k in keys})
            where.update({k: fn for k in keys})
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            files.append((f"ckpt/{fn}", os.path.getsize(path), sha, "file://" + path))
        os.makedirs(os.path.join(dest, "ckpt"))
        with open(os.path.join(dest, "ckpt", "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": where}, f)
        stream = ShardStream(files, dest, ahead=1, log=lambda *a: None)
        shards = StreamedSharded(os.path.join(dest, "ckpt"), stream, "ckpt/")
        for k in names:
            assert np.array_equal(shards.get(k), want[k]), k
        assert stream.peak_files <= 3, stream.peak_files
        try:
            shards.get(names[0])
            raise AssertionError("a read from a deleted shard must be refused")
        except RuntimeError as e:
            assert "read past" in str(e)
        shards.close()
        stream.close()
        assert not [f for f in os.listdir(os.path.join(dest, "ckpt")) if f.endswith(".safetensors")]


def _mini_minimax_checkpoint(d: str, shards: int = 3):
    """A random MiniMax-H3-Image transformer in the original checkpoint's layout, tiny, its blocks spread over shards
    in order (as the release's are), with the per-step heads and grid."""
    import json
    from h3metal import st
    D, H, hd, F, TE, TXT, L = 256, 2, 128, 512, 256, 256, 3
    inner = H * hd
    rng = np.random.default_rng(1)
    r = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32)  # noqa: E731
    one = lambda n: (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
    t = {"time_embedder.proj_in.weight": r(TE, 256), "time_embedder.proj_in.bias": r(TE),
         "time_embedder.proj_out.weight": r(TE, TE), "time_embedder.proj_out.bias": r(TE),
         "video_patch_proj.weight": r(D, 96), "video_patch_proj.bias": r(D),
         "condition_proj.weight": r(D, TXT), "condition_proj.bias": r(D),
         "final_layer.adaln_proj.linear.weight": r(2 * D, TE), "final_layer.adaln_proj.linear.bias": r(2 * D),
         "final_layer.norm.weight": one(D), "token_refiner.final_norm.weight": one(D),
         "final_layer.video_out.weight": r(96, D), "final_layer.video_out.bias": r(96)}
    blocks = [f"token_refiner.blocks.{i}" for i in range(2)] + [f"blocks.{l}" for l in range(L)]
    for b in blocks:
        t.update({f"{b}.attn.qkv_proj.weight": r(3 * inner, D), f"{b}.attn.out_proj.weight": r(D, inner),
                  f"{b}.attn.q_norm.weight": one(hd), f"{b}.attn.k_norm.weight": one(hd),
                  f"{b}.mlp.fc1.weight": r(2 * F, D), f"{b}.mlp.fc2.weight": r(D, F),
                  f"{b}.norm1.weight": one(D), f"{b}.norm2.weight": one(D)})
        if b.startswith("blocks."):
            t.update({f"{b}.adaln_proj.linear.weight": r(18 * D, TE) * 0.5, f"{b}.adaln_proj.linear.bias": r(18 * D)})
    os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
    order = [k for k in t if not k.startswith("blocks.")] + [k for b in blocks[2:] for k in t if k.startswith(b + ".")]
    per = -(-len(order) // shards)
    where = {}
    for j in range(shards):
        fn = f"model-{j + 1:05d}-of-{shards:05d}.safetensors"
        keys = order[j * per:(j + 1) * per]
        st.save(os.path.join(d, "transformer", fn), {k: t[k] for k in keys})
        where.update({k: fn for k in keys})
    with open(os.path.join(d, "transformer", "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": where}, f)
    with open(os.path.join(d, "transformer", "config.json"), "w") as f:
        json.dump({"hidden_size": D, "num_layers": L, "token_refiner_num_layers": 2, "num_attention_heads": H,
                   "attention_head_dim": hd, "ffn_hidden_size": F, "latents_dim": 24, "patch_size": [1, 2, 2],
                   "text_dim": TXT, "timestep_input_dim": 256, "time_embed_hidden_size": TE, "time_embed_dim": TE,
                   "rope_inv_freq_len": 16, "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5}, f)
    st.save(os.path.join(d, "pdd_fused_heads.safetensors"),
            {"video_out.weight": r(6, 96, D), "video_out.bias": r(6, 96)})
    with open(os.path.join(d, "pdd_config.json"), "w") as f:
        json.dump({"sigmas": [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0]}, f)
    return sorted(set(where.values()))


def test_streamed_conversion_is_the_same():
    """MiniMax-H3-Image's transformer converted while its shards stream in (and go) is byte for byte the one
    converted from the whole checkpoint, and no more than three shards were on disk at once."""
    import hashlib
    import shutil
    from h3metal.convert import convert_dit
    from h3metal.stream import ShardStream, StreamedSharded
    fmts = {"qkv": "q8", "out": "q8", "fc1": "q4", "fc2": "q4"}
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dest:
        names = _mini_minimax_checkpoint(src)
        convert_dit(src, os.path.join(src, "whole.h3m"), fmts=fmts, log=lambda *a: None)
        os.makedirs(os.path.join(dest, "transformer"))
        for f in ("transformer/config.json", "transformer/model.safetensors.index.json", "pdd_config.json",
                  "pdd_fused_heads.safetensors"):
            shutil.copy(os.path.join(src, f), os.path.join(dest, f))
        files = []
        for fn in names:
            path = os.path.join(src, "transformer", fn)
            files.append((f"transformer/{fn}", os.path.getsize(path), hashlib.sha256(open(path, "rb").read()).hexdigest(),
                          "file://" + path))
        stream = ShardStream(files, dest, ahead=1, log=lambda *a: None)
        shards = StreamedSharded(os.path.join(dest, "transformer"), stream, "transformer/")
        convert_dit(dest, os.path.join(dest, "streamed.h3m"), fmts=fmts, shards=shards, log=lambda *a: None)
        shards.close()
        stream.close()
        a = open(os.path.join(src, "whole.h3m"), "rb").read()
        b = open(os.path.join(dest, "streamed.h3m"), "rb").read()
        assert a == b, "the streamed conversion differs"
        assert stream.peak_files <= 3, stream.peak_files
        assert not [f for f in os.listdir(os.path.join(dest, "transformer")) if f.endswith(".safetensors")]


def test_two_transformers_in_one_pass():
    """convert_dits writes the six-step file and the few-step one (a LoRA merged, the flow grids) in one pass over
    streamed shards, each byte for byte what convert_dit writes alone, still with three shards on disk at most."""
    import hashlib
    import shutil
    from h3metal import st
    from h3metal.convert import convert_dit, convert_dits
    from h3metal.stream import ShardStream, StreamedSharded
    fmts = {"qkv": "q8", "out": "q8", "fc1": "q4", "fc2": "q4"}
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dest:
        names = _mini_minimax_checkpoint(src)
        rng = np.random.default_rng(2)
        lora = {}
        for m, (o, i) in {"transformer_blocks.1.attn.to_k": (256, 256), "transformer_blocks.2.ff.net.0.proj": (1024, 256),
                          "token_refiner.refiner_blocks.0.attn.to_out.0": (256, 256)}.items():
            lora[f"transformer.{m}.lora_A.weight"] = rng.standard_normal((4, i)).astype(np.float32) * 0.05
            lora[f"transformer.{m}.lora_B.weight"] = rng.standard_normal((o, 4)).astype(np.float32) * 0.05
        lpath = os.path.join(src, "lora.safetensors")
        st.save(lpath, lora)
        few = dict(fmts=fmts, flow={"shift": 12.0, "steps": [2, 3]}, lora=[(lpath, 1.0)])
        convert_dit(src, os.path.join(src, "six.h3m"), fmts=fmts, log=lambda *a: None)
        convert_dit(src, os.path.join(src, "few.h3m"), log=lambda *a: None, **few)
        os.makedirs(os.path.join(dest, "transformer"))
        for f in ("transformer/config.json", "transformer/model.safetensors.index.json", "pdd_config.json",
                  "pdd_fused_heads.safetensors"):
            shutil.copy(os.path.join(src, f), os.path.join(dest, f))
        files = [(f"transformer/{fn}", os.path.getsize(os.path.join(src, "transformer", fn)),
                  hashlib.sha256(open(os.path.join(src, "transformer", fn), "rb").read()).hexdigest(),
                  "file://" + os.path.join(src, "transformer", fn)) for fn in names]
        stream = ShardStream(files, dest, ahead=1, log=lambda *a: None)
        shards = StreamedSharded(os.path.join(dest, "transformer"), stream, "transformer/")
        convert_dits(dest, [dict(out=os.path.join(dest, "six.h3m"), fmts=fmts),
                            dict(out=os.path.join(dest, "few.h3m"), **few)], shards=shards, log=lambda *a: None)
        shards.close()
        stream.close()
        for f in ("six.h3m", "few.h3m"):
            assert open(os.path.join(src, f), "rb").read() == open(os.path.join(dest, f), "rb").read(), f
        assert stream.peak_files <= 3, stream.peak_files


def test_dit_files():
    """Which transformer file runs how many steps: the six-step one by default, the few-step one for 2-4."""
    from h3metal.pipeline import dit_files
    with tempfile.TemporaryDirectory() as d:
        Writer(os.path.join(d, "dit-pdmd.h3m"), {"kind": "dit", "timesteps": [0.0] * 9, "sigmas": [1.0, 0.9, 0.0],
               "schedules": {"2": {"sigmas": [1.0, 0.9, 0.0], "first": 0}, "3": {"sigmas": [], "first": 2},
                             "4": {"sigmas": [], "first": 5}}, "default_steps": 2}).close()
        files = dit_files(d)
        assert files[None].endswith("dit-pdmd.h3m") and sorted(k for k in files if k) == [2, 3, 4]
        Writer(os.path.join(d, "dit.h3m"), {"kind": "dit", "timesteps": [0.0] * 6, "sigmas": [0.0] * 7}).close()
        files = dit_files(d)
        assert files[None].endswith("dit.h3m") and files[6].endswith("dit.h3m") and files[2].endswith("dit-pdmd.h3m")


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


def test_torch_randn_replica():
    # First values of torch.randn((32,), generator=torch.Generator("cpu").manual_seed(0)), recorded from torch 2.15.
    from h3metal.rng import TorchRandn
    got = TorchRandn(0).normal((32,))
    want = np.array([-1.1258398, -1.1523602, -0.2505786, -0.4338788, 0.8487104, 0.6920092, -0.3160128, -2.1152196],
                    np.float32)
    assert np.allclose(got[:8], want, atol=1e-6), got[:8]


def test_cli_sizes_and_device_check():
    import argparse
    from h3metal.cli import Ratio, _size, resolve_size, size_for
    from h3metal.gpu import has_matrix_units
    assert size_for("16:9") == (5472, 3072) and size_for("1:1") == (4096, 4096)  # MiniMax's recommended sizes
    assert _size("2048x1536") == (2048, 1536) and _size("16:9@2k") == size_for("16:9", 2048 * 2048)
    # a ratio alone takes the model's native area: MiniMax's 16 MP, FLUX.2's and Z-Image's 1 MP
    assert isinstance(_size("16:9"), Ratio) and resolve_size(_size("16:9"), 4096 * 4096) == (5472, 3072)
    assert resolve_size(_size("16:9"), 1024 * 1024) == (1376, 768) and resolve_size((640, 480), 1) == (640, 480)
    assert _size("2000x1008") == (2000, 1008)  # multiples of 16 pass (Z-Image); MiniMax-H3 checks for 32 later
    for bad in ("2008x2008", "100x100", "16:9@8k", "big"):
        try:
            _size(bad)
            raise AssertionError(bad)
        except argparse.ArgumentTypeError:
            pass
    assert has_matrix_units("Apple M6") and has_matrix_units("Apple M5 Pro") and not has_matrix_units("Apple M4 Max")
    from h3metal.cli import DEFAULT_MODEL, FLUX2_9B_MODEL, FLUX2_MODEL, ZIMAGE_MODEL, _model_dir
    assert _model_dir("zimage") == ZIMAGE_MODEL and _model_dir("Z-Image-Turbo") == ZIMAGE_MODEL
    assert _model_dir("minimax-h3") == DEFAULT_MODEL and _model_dir("/some/dir") == "/some/dir"
    assert _model_dir("flux2") == FLUX2_MODEL and _model_dir("flux2-9b") == FLUX2_9B_MODEL
    # whichever is converted, in this order; MiniMax-H3's directory when none is
    assert _model_dir(None) in (DEFAULT_MODEL, ZIMAGE_MODEL, FLUX2_MODEL, FLUX2_9B_MODEL)
    from h3metal.flux2 import variant_name
    assert variant_name(3072) == "FLUX.2 [klein] 4B" and variant_name(4096) == "FLUX.2 [klein] 9B"
    import contextlib
    import io
    from h3metal import __version__, cli
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            cli.main(["--version"])
        except SystemExit as e:
            assert e.code == 0, e.code
    assert out.getvalue().strip() == f"h3metal {__version__}", out.getvalue()
    from h3metal.cli import auto_format
    assert [auto_format(g * 2**30) for g in (16, 24, 32, 64)] == ["q4", "mix", "q8", "q8"]


def test_download_choices():
    """What `h3metal download` fetches for each model and source (no network: the listings and the fetches are
    stand-ins): converted weights from Hugging Face at the pinned tag by default for Z-Image-Turbo and FLUX.2
    [klein] 4B, the official checkpoint with --original, from ModelScope, while the converted weights are not
    published, and for the other models; klein 9B only with its license accepted."""
    import urllib.error
    from h3metal import cli, download as dl
    real = dl._json, dl._fetch_all
    seen, urls = [], []
    official = ("model_index.json", "transformer/a.safetensors", "flux-2-klein-4b.safetensors",
                "flux-2-klein-9b.safetensors", "sample.jpg")

    def listing(url):
        urls.append(url)
        if "/api/models/audenvenn/h3metal-" in url:  # converted weights
            return {"siblings": [{"rfilename": "dit.h3m", "size": 3, "lfs": {"sha256": "ab"}},
                                 {"rfilename": ".gitattributes", "size": 1}]}
        if "modelscope" in url:
            return {"Code": 200, "Data": {"Files": [{"Path": n, "Size": 1, "Type": "blob"} for n in official]}}
        return {"siblings": [{"rfilename": n, "size": 1} for n in official]}
    try:
        dl._json = listing
        dl._fetch_all = lambda todo, dest, jobs, log, what: seen.append((what, dest, sorted(f[0] for f in todo))) or True
        home = os.path.expanduser("~/.h3metal")
        rev = dl.CONVERTED["flux2-klein-4b"]["revision"]
        cli.main(["download", "--model", "flux2-klein-4b"])
        assert seen[-1] == (f"audenvenn/h3metal-flux2-klein-4b@{rev}", f"{home}/flux2-klein", ["dit.h3m"]), seen[-1]
        assert urls[-1].endswith(f"/revision/{rev}?blobs=true"), urls[-1]
        cli.main(["download", "--model", "z-image-turbo"])
        assert seen[-1][:2] == (f"audenvenn/h3metal-z-image-turbo@{rev}", f"{home}/zimage"), seen[-1]
        cli.main(["download", "--model", "flux2-klein-4b", "--original"])  # no duplicate transformer, no samples
        assert seen[-1] == ("black-forest-labs/FLUX.2-klein-4B [huggingface]", f"{home}/flux2-klein-4b",
                            ["flux-2-klein-9b.safetensors", "model_index.json", "transformer/a.safetensors"]), seen[-1]
        cli.main(["download", "--model", "z-image-turbo", "--source", "modelscope"])
        assert seen[-1][:2] == ("Tongyi-MAI/Z-Image-Turbo [modelscope]", f"{home}/z-image-turbo"), seen[-1]
        n = len(seen)
        try:  # klein 9B's license is non-commercial: no download without accepting it
            cli.main(["download", "--model", "flux2-klein-9b"])
            raise AssertionError("the 9B download did not ask for the license")
        except SystemExit as e:
            assert "Non-Commercial" in str(e) and "--accept-license" in str(e), e
        assert len(seen) == n
        cli.main(["download", "--model", "flux2-klein-9b", "--source", "modelscope", "--accept-license"])
        assert seen[-1] == ("black-forest-labs/FLUX.2-klein-9B [modelscope]", f"{home}/flux2-klein-9b",
                            ["flux-2-klein-4b.safetensors", "model_index.json", "transformer/a.safetensors"]), seen[-1]

        def unpublished(url):  # Hugging Face answers 401 for a repository that does not exist
            if "/api/models/audenvenn/h3metal-" in url:
                raise urllib.error.HTTPError(url, 401, "Repository not found", None, None)
            return listing(url)
        dl._json = unpublished
        cli.main(["download", "--model", "z-image-turbo"])  # then the official checkpoint, as before
        assert seen[-1][:2] == ("Tongyi-MAI/Z-Image-Turbo [huggingface]", f"{home}/z-image-turbo"), seen[-1]
    finally:
        dl._json, dl._fetch_all = real


def test_lora_download_and_names():
    """`h3metal download --lora owner/name[/file]` (no network: stand-ins) and --lora NAME's lookup."""
    from h3metal import download as dl
    from h3metal.lora import parse_spec
    real = dl.list_files, dl._fetch_all
    seen = []
    with tempfile.TemporaryDirectory() as d:
        try:
            dl.list_files = lambda source, repo: [(f, 1, None, f"https://example/{repo}/{f}") for f in
                                                  {"a/one": ["README.md", "style.safetensors"],
                                                   "a/two": ["x.safetensors", "y.safetensors"]}[repo]]
            dl._fetch_all = lambda todo, dest, jobs, log, what: seen.append((todo[0][0], dest)) or True
            assert dl.download_lora("a/one", dest=d) == os.path.join(d, "style.safetensors")
            assert dl.download_lora("a/two/y.safetensors", dest=d) == os.path.join(d, "y.safetensors")
            for bad in ("a/two", "a/two/z.safetensors", "single"):
                try:
                    dl.download_lora(bad, dest=d)
                    raise AssertionError(bad)
                except SystemExit:
                    pass
            assert seen == [("style.safetensors", d), ("y.safetensors", d)], seen
        finally:
            dl.list_files, dl._fetch_all = real
        open(os.path.join(d, "pixel.safetensors"), "wb").close()
        assert parse_spec("pixel:0.7", d) == (os.path.join(d, "pixel.safetensors"), 0.7)
        assert parse_spec("pixel.safetensors", d) == (os.path.join(d, "pixel.safetensors"), 1.0)
        assert parse_spec("/x/missing.safetensors:-1", d) == ("/x/missing.safetensors", -1.0)


def test_serve_page_and_checks():
    """The web page's server without a GPU: the page, its state, no paths outside the outputs, input checks."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from h3metal import serve
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ld:
        serve.OUT_DIR = d
        open(os.path.join(ld, "style.safetensors"), "wb").close()
        open(os.path.join(ld, "notes.txt"), "wb").close()
        studio = serve.Studio({"Z-Image-Turbo": "/nonexistent", "MiniMax-H3-Image": "/nonexistent"}, None, ld)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(studio))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            assert b"h3metal studio" in urllib.request.urlopen(url + "/").read()
            st = json.loads(urllib.request.urlopen(url + "/state").read())
            assert st["models"] == ["Z-Image-Turbo", "MiniMax-H3-Image"] and st["default"] == "Z-Image-Turbo"
            assert st["loras"] == [{"name": "style.safetensors", "models": []}], st["loras"]  # fits no model
            assert st["steps"] == {}, st["steps"]  # no few-step transformer
            assert json.loads(urllib.request.urlopen(url + "/history").read()) == []
            for path in ("/outputs/..%2Fsecret.png", "/outputs/missing.png", "/nothing"):
                try:
                    urllib.request.urlopen(url + path)
                    raise AssertionError(path)
                except urllib.error.HTTPError as e:
                    assert e.code == 404, (path, e.code)
            lo = '"prompt": "x", "loras": '
            for body, msg in ((b'{"prompt": ""}', "empty"), (b'{"prompt": "x", "model": "nope"}', "unknown model"),
                              (b"not json", "not JSON"),
                              (('{' + lo + '[{"name": "../style.safetensors"}]}').encode(), "no LoRA"),
                              (('{' + lo + '[{"name": "notes.txt"}]}').encode(), "no LoRA"),
                              (('{' + lo + '[{"name": "style.safetensors", "strength": 9}]}').encode(), "between"),
                              (('{"model": "MiniMax-H3-Image", ' + lo + '[{"name": "style.safetensors"}]}').encode(),
                               "work with"),
                              (b'{"prompt": "x", "model": "MiniMax-H3-Image", "steps": 2}', "does not run 2 steps"),
                              (b'{"prompt": "x", "model": "Z-Image-Turbo", "steps": 4}', "does not run 4 steps")):
                try:
                    urllib.request.urlopen(urllib.request.Request(url + "/generate", data=body))
                    raise AssertionError(body)
                except urllib.error.HTTPError as e:
                    assert e.code == 400 and msg in json.loads(e.read())["error"], (body, e.code)
        finally:
            httpd.shutdown()


def test_serve_calls_each_pipeline_with_its_own_arguments():
    """The web page's server against the pipelines' real signatures (no GPU): every model, text to image and edits,
    with and without a LoRA or a step count, gets only arguments its pipeline takes."""
    import inspect
    from h3metal import pipeline, serve
    from h3metal.flux2 import Flux2Pipeline
    from h3metal.zimage import ZImagePipeline
    img = np.zeros((32, 32, 3), np.uint8)
    calls = []

    class Fake:
        def __init__(self, cls):
            self.sig = inspect.signature(cls.generate)

        def generate(self, *a, **k):
            self.sig.bind(self, *a, **k)  # TypeError for an argument the real pipeline does not take
            calls.append(k)
            return img, "timing"

    edit_sig = inspect.signature(pipeline.generate_edit)

    def fake_edit(*a, **k):
        edit_sig.bind(*a, **k)
        calls.append(k)
        return img, "timing"

    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ld:
        serve.OUT_DIR = d
        studio = serve.Studio({"MiniMax-H3-Image": "/nonexistent", "Z-Image-Turbo": "/nonexistent",
                               "FLUX.2 [klein] 4B": "/nonexistent"}, None, ld)
        studio.steps = {"MiniMax-H3-Image": [6, 2]}
        fakes = {"MiniMax-H3-Image": Fake(pipeline.Pipeline), "Z-Image-Turbo": Fake(ZImagePipeline),
                 "FLUX.2 [klein] 4B": Fake(Flux2Pipeline)}
        studio._pipeline = lambda name: fakes[name]
        studio._loras = lambda req, name: []
        orig = pipeline.generate_edit
        pipeline.generate_edit = fake_edit
        try:
            for req in ({"model": "MiniMax-H3-Image", "prompt": "x"},
                        {"model": "MiniMax-H3-Image", "prompt": "x", "steps": 2},
                        {"model": "Z-Image-Turbo", "prompt": "x"},
                        {"model": "FLUX.2 [klein] 4B", "prompt": "x"}):
                out = studio.generate(req)
                assert out["url"].startswith("/outputs/"), out
            assert calls[1].get("steps") == 2 and "steps" not in calls[0], calls
        finally:
            pipeline.generate_edit = orig


def test_lora_terms_match_the_converter():
    """A LoRA's terms on the converted layers equal the converter's own transform (h3metal/lora.py LinearPlan)
    applied to the dense update scale * B A: for every layout (diffusers names, kohya's, the original fused
    names) and both models, including rows that fused, permuted, sign-flipped or reordered layers move."""
    import numpy as np
    from h3metal import lora
    from h3metal import flux2, zimage
    from h3metal.st import save
    rng = np.random.default_rng(0)
    zc = {"dim": 512, "heads": 4, "head_dim": 128, "ffn": 1280, "layers": 1, "refiner_layers": 1}
    fc = {"dim": 256, "heads": 2, "head_dim": 128, "ffn": 768, "double_layers": 1, "single_layers": 1,
          "in_channels": 128, "joint_dim": 192, "t_channels": 256}
    cases = (("zimage", zimage.model_plans(zc, 3), {
        "layers.0.attention.to_q": None, "layers.0.attention.to_v": None, "layers.0.feed_forward.w3": None,
        "noise_refiner.0.feed_forward.w2": None, "context_refiner.0.attention.to_out.0": None,
        "layers.0.adaLN_modulation.0": None}, {"layers.0.attention.qkv": None}),
             ("flux2", flux2.model_plans(fc, 5), {
        "transformer_blocks.0.attn.add_k_proj": None, "transformer_blocks.0.ff.linear_in": None,
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj": None, "single_transformer_blocks.0.attn.to_out": None,
        "x_embedder": None, "double_stream_modulation_img.linear": None},
              {"double_blocks.0.txt_attn.qkv": None, "final_layer.adaLN_modulation.1": None}))
    with tempfile.TemporaryDirectory() as d:
        for model, (plans, aliases), direct, fused in cases:
            index = lora._index(plans)
            shapes = {m: lora._shape(index, plans, m) for m in index}
            dense = {}  # module -> the update it gets, scale * B A
            files = {"peft": {}, "kohya": {}}
            for m in direct:
                n_out, k = shapes[m]
                A, B = rng.standard_normal((4, k)), rng.standard_normal((n_out, 4))
                dense[m] = B @ A
                files["peft"][f"diffusion_model.{m}.lora_A.weight"] = A.astype(np.float32)
                files["peft"][f"diffusion_model.{m}.lora_B.weight"] = B.astype(np.float32)
                kk = "lora_unet_" + m.replace(".", "_")
                files["kohya"][f"{kk}.lora_down.weight"] = A.astype(np.float32)
                files["kohya"][f"{kk}.lora_up.weight"] = (B * 4).astype(np.float32)
                files["kohya"][f"{kk}.alpha"] = np.array(1.0, np.float32)  # alpha 1, rank 4: scale 1/4
            files["fused"] = dict(files["peft"])
            direct_only = dict(dense)
            for a in fused:
                parts = aliases[a]
                k = shapes[parts[0][0]][1]
                rows = []
                for m, where in parts:
                    n_out = shapes[m][0]
                    rows.append(np.arange(where, where + n_out) if np.ndim(where) == 0 else np.asarray(where))
                n_b = max(int(r.max()) + 1 for r in rows)
                A, B = rng.standard_normal((4, k)), rng.standard_normal((n_b, 4))
                for (m, _), r in zip(parts, rows):
                    dense[m] = dense.get(m, 0) + B[r] @ A
                files["fused"][f"base_model.model.{a}.lora_A.weight"] = A.astype(np.float32)
                files["fused"][f"base_model.model.{a}.lora_B.weight"] = B.astype(np.float32)
            other = cases[1][1] if model == "zimage" else cases[0][1]
            for layout in ("peft", "kohya", "fused"):
                path = os.path.join(d, f"{model}_{layout}.safetensors")
                save(path, files[layout])
                # the web page's check from the header alone: fits this model, not the other
                assert lora.fits(lora.shapes(path), plans, aliases) and not lora.fits(lora.shapes(path), *other)
                layers = lora.read(path)
                ts = lora.terms([(layers, 0.75)], plans, aliases, model=model)
                want = dense if layout == "fused" else direct_only
                for name, plan in plans.items():
                    if not any(p.module in want for p in plan.parts):
                        assert name not in ts, (layout, name)
                        continue
                    expect = 0.75 * plan.assemble(lambda m: np.asarray(
                        want.get(m, np.zeros(shapes[m])), np.float32))
                    A, B = lora.stack(ts[name])
                    got = B.astype(np.float64) @ A.astype(np.float64)
                    err = np.abs(got - expect).max() / np.abs(expect).max()
                    assert err < 1e-5, (model, layout, name, err)
    for bad in ({"diffusion_model.layers.0.attention.to_q.lora_A.weight": np.ones((4, 100), np.float32),
                 "diffusion_model.layers.0.attention.to_q.lora_B.weight": np.ones((512, 4), np.float32)},
                {"diffusion_model.layers.9.attention.to_q.lora_A.weight": np.ones((4, 512), np.float32),
                 "diffusion_model.layers.9.attention.to_q.lora_B.weight": np.ones((512, 4), np.float32)}):
        with tempfile.TemporaryDirectory() as d:
            save(os.path.join(d, "bad.safetensors"), bad)
            try:
                lora.terms([(lora.read(os.path.join(d, "bad.safetensors")), 1.0)], *cases[0][1], model="Z-Image")
            except lora.LoraError:
                continue
            raise AssertionError("a LoRA for another model was not refused")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
