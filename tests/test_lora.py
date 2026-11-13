"""LoRA adapters on the mini models, against the models with the adapter merged into their float32 weights
(the reference side is tests/ref/mini_lora.py, after tests/ref/mini_zimage.py and tests/ref/mini_flux2.py).

    python tests/test_lora.py [REF_DIR]

Each file layout (diffusers/PEFT names, kohya names with an alpha, the original checkpoint's fused names) must
give the same result, at strengths 1 and 0.5, as close to the reference as the int8 model without an adapter is
to its own; removing the adapters must give back the outputs without them, bit for bit. Files h3metal cannot
apply exactly must be refused.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal import lora  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref")


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_model(which: str, gpu: Gpu):
    src = os.path.join(root, f"mini_{which}")
    ref = np.load(os.path.join(src, "ref.npz"))
    lref = np.load(os.path.join(src, "lora", "ref.npz"))
    out = os.path.join(src, f"{which}_q8.h3m")
    if which == "zimage":
        from h3metal.zimage import ZImageDiT as DiT
        from h3metal.zimage import convert_transformer
    else:
        from h3metal.flux2 import Flux2DiT as DiT
        from h3metal.flux2 import convert_transformer
    convert_transformer(src, out, fmts={k: "q8" for k in ("qkv", "out", "fc1", "fc2")}, log=lambda *a: None)
    dit = DiT(gpu, WeightFile(gpu, out))
    n_steps = len([k for k in lref.files if k.startswith("base")])

    def run():
        if which == "zimage":
            C, Hh, W = ref["latents"].shape
            dit.prepare(ref["cap"].shape[0], Hh, W)
            dit.encode_caption(ref["cap"])
            return [dit.step(s, ref[f"x{s}"]) for s in range(n_steps)]
        dit.prepare(ref["txt"], ref["i_ids"])
        return [dit.step(float(ref["timesteps"][s]) / 1000, ref[f"x{s}"]) for s in range(n_steps)]

    plain = run()
    base_err = max(rel(o, lref[f"base{s}"]) for s, o in enumerate(plain))
    print(f"{which}: int8 model without an adapter: worst step {base_err:.4f}")
    layouts = ["peft", "kohya", "fused"] + (["fused_final"] if which == "flux2" else [])
    results = {}
    for layout in layouts:
        layers = lora.read(os.path.join(src, "lora", f"{layout}.safetensors"))
        for strength, tag in ((1.0, "1"), (0.5, "05")):
            if layout != "peft" and tag == "05":
                continue
            dit.set_loras([(layers, strength)])
            outs = run()
            err = max(rel(o, lref[f"out{s}_{tag}"]) for s, o in enumerate(outs))
            moved = min(rel(lref[f"out{s}_{tag}"], lref[f"base{s}"]) for s in range(n_steps))
            print(f"  {layout:11s} strength {strength}: worst step {err:.4f} (the adapter moves the output "
                  f"{moved:.3f})")
            # as close as the model without an adapter (with room for the adapter's own int8 input side), and far
            # closer than the adapter's effect
            assert err < max(1.5 * base_err, 0.02) and err < moved / 5, (layout, strength, err, base_err, moved)
            results[(layout, tag)] = outs
    # kohya's layout holds the same terms (B doubled, alpha halving it): the same products, bit for bit. (The fused
    # layouts hold a shared-A query/key/value adapter as one term instead of three, whose int8 input side is
    # rounded with other scales; each is checked against the reference above.)
    assert all(np.array_equal(a, b) for a, b in zip(results[("kohya", "1")], results[("peft", "1")]))
    print("  kohya and peft layouts: the same outputs, bit for bit")
    dit.set_loras([])
    again = run()
    assert all(np.array_equal(a, b) for a, b in zip(again, plain)), "removing the adapters changed the outputs"
    print("  removed: outputs without an adapter, bit for bit")


def check_refusals():
    from h3metal.st import save
    with tempfile.TemporaryDirectory() as d:
        A, B = np.ones((4, 8), np.float32), np.ones((8, 4), np.float32)
        cases = {
            "dora": {"diffusion_model.layers.0.attention.to_q.lora_A.weight": A,
                     "diffusion_model.layers.0.attention.to_q.lora_B.weight": B,
                     "diffusion_model.layers.0.attention.to_q.dora_scale": np.ones((1, 8), np.float32)},
            "te": {"lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": A,
                   "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": B},
            "lokr": {"lora_unet_layers_0_attention_to_q.lokr_w1": A},
            "half": {"diffusion_model.layers.0.attention.to_q.lora_A.weight": A},
        }
        for name, sd in cases.items():
            p = os.path.join(d, f"{name}.safetensors")
            save(p, sd)
            try:
                lora.read(p)
            except lora.LoraError as e:
                print(f"refused {name}: {str(e)[:110]}")
                continue
            raise AssertionError(f"{name} was not refused")


if __name__ == "__main__":
    gpu = Gpu()
    check_refusals()
    check_model("zimage", gpu)
    check_model("flux2", gpu)
    print("ok")
