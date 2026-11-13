"""Mini DiT converted for the flow sampler with a LoRA merged, against diffusers' MiniMax-H3 transformer with the LoRA
merged in float32 (tests/ref/mini_dit_flow.py): the step grids, every step's velocity, and the Euler update.

    python tests/test_dit_flow_mini.py [REF_DIR]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal import st  # noqa: E402
from h3metal.convert import convert_dit, flow_sigmas  # noqa: E402
from h3metal.dit import DiT, Layout, euler  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.lora import read  # noqa: E402
from h3metal.weights import WeightFile, bf16_bits  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_dit_flow"
r = np.load(os.path.join(src, "ref_flow.npz"))
shift = float(r["shift"])
GRIDS = (2, 3, 4)

for n in GRIDS:
    got = flow_sigmas(n, shift)
    assert np.array_equal(got, r[f"sigmas{n}"]), (n, got, r[f"sigmas{n}"])
print("grids: bit for bit those of diffusers' MiniMaxH3Scheduler")
for n in GRIDS:
    sig = r[f"sigmas{n}"]
    nxt = euler(r["latents"], r[f"vel{n}_0"], 0, sig, "blend")
    assert np.array_equal(nxt, r[f"next{n}"]), (n, np.abs(nxt - r[f"next{n}"]).max())
print("blend update: bit for bit the scheduler's step")

# The same adapter written in the checkpoint's own layout (fused per-head QKV, [gate; value] fc1), as ComfyUI's
# conversion has it: it must merge to the same matrices.
H, hd = 4, 128
native = {}
layers = read(os.path.join(src, "lora.safetensors"))
for module in {m.rsplit(".attn.", 1)[0] for m in layers if ".attn.to_q" in m}:
    dst = module.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.").replace("transformer_blocks.", "blocks.")
    q, k, v = (layers[f"{module}.attn.to_{x}"] for x in "qkv")
    r_ = q.A.shape[0]
    A = np.concatenate([q.A, k.A, v.A])
    B = np.zeros((3 * H * hd, 3 * r_), np.float32).reshape(H, 3, hd, 3 * r_)
    for j, lay in enumerate((q, k, v)):
        B[:, j, :, j * r_:(j + 1) * r_] = lay.B.reshape(H, hd, r_) * lay.scale
    native[f"{dst}.attn.qkv_proj.lora_A.weight"] = A
    native[f"{dst}.attn.qkv_proj.lora_B.weight"] = B.reshape(3 * H * hd, 3 * r_)
    for a, b, swap in ((".attn.to_out.0", ".attn.out_proj", False), (".ff.net.0.proj", ".mlp.fc1", True),
                       (".ff.net.2", ".mlp.fc2", False)):
        lay = layers[module + a]
        Bn = lay.B * lay.scale
        if swap:
            half = Bn.shape[0] // 2
            Bn = np.concatenate([Bn[half:], Bn[:half]])
        native[f"{dst}{b}.lora_A.weight"], native[f"{dst}{b}.lora_B.weight"] = lay.A, Bn
native_path = os.path.join(src, "lora_native.safetensors")
st.save(native_path, {k: np.ascontiguousarray(v, np.float32) for k, v in native.items()})

gpu = Gpu()
T = r["text"].shape[0]
text = gpu.empty((64, r["text"].shape[1]), "bf16")
text.numpy()[:T] = bf16_bits(r["text"])
lay = Layout(T, np.where(r["tags"] == 1, 1, 0).astype(np.uint8), [], int(r["lat_hw"][0]), int(r["lat_hw"][1]))


def velocities(lora, name):
    out = os.path.join(src, f"dit_flow_{name}.h3m")
    convert_dit(src, out, fmts={k: "q8" for k in ("qkv", "out", "fc1", "fc2")}, refiner_fmt="q8", log=lambda *a: None,
                flow={"shift": shift, "steps": list(GRIDS)}, lora=lora)
    wf = WeightFile(gpu, out)
    dit = DiT(gpu, wf)
    assert dit.update == "blend" and dit.default_steps == GRIDS[0], (dit.update, dit.default_steps)
    rows = dit.encode_text(text, T)
    dit.prepare(lay)
    vel = {}
    for n in GRIDS:
        sig, first = dit.schedules[n]
        assert np.array_equal(np.array(sig, np.float32), r[f"sigmas{n}"])
        for s in range(n):
            vel[n, s] = dit.step(first + s, rows, r["latents"])
    dit.free()
    wf.close()
    return vel


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


merged = velocities([(os.path.join(src, "lora.safetensors"), 1.0)], "lora")
worst = 0.0
for (n, s), v in merged.items():
    ref = r[f"vel{n}_{s}"]
    cos = float((v * ref).sum() / np.linalg.norm(v) / np.linalg.norm(ref))
    worst = max(worst, rel(v, ref))
    print(f"{n} steps, step {s}: rel L2 err {rel(v, ref):.4f}  cosine {cos:.6f}")
assert worst < 0.02, worst

plain = velocities(None, "plain")
miss = min(rel(plain[k], r[f"vel{k[0]}_{k[1]}"]) for k in plain)
print(f"without the LoRA: rel L2 err at least {miss:.3f} (the merge is what makes it match)")
assert miss > 10 * worst, (miss, worst)

nat = velocities([(native_path, 1.0)], "native")
same = max(rel(nat[k], merged[k]) for k in merged)
print(f"the adapter in the checkpoint's own layout: rel diff {same:.1e} from the diffusers layout")
assert same < 1e-3, same
print("ok")
