# Contributing to h3metal

Thanks for looking. The goal is narrow: run MiniMax-H3-Image on one Apple
Silicon Mac as fast as the hardware allows, at the image quality of the
official pipeline, and say exactly how fast and how close that is.
Contributions that serve that goal are welcome.

## Ground rules

- **English only.** Code, comments, documentation, commit messages, issues and
  pull requests are in English. CI checks the repository for this.
- **Measured claims.** A statement about speed, memory or quality comes with
  the machine, the commands and the numbers. Speed is measured back to back
  against the baseline, with nothing else using the GPU.
- **Quality is measured against the official pipeline.** A change to the
  numerics (kernels, formats, conversion) comes with `bench/quality.py`
  results against a reference run (`tests/ref/reference.py`).
- **One change per pull request**, and one change per measurement.

## Setting up

```sh
cmake -S . -B build/cmake -G Ninja && cmake --build build/cmake
python3 -m venv .venv && .venv/bin/pip install -e .
```

The Metal kernels in `kernels/` are compiled at run time, so editing a
kernel needs no rebuild; editing `src/` does.

## Tests

| What | Needs | Command |
|---|---|---|
| Conversion numerics, file format, layouts | nothing | `python tests/test_convert.py` |
| Matrix products, attention, row kernels against NumPy | GPU | `python tests/test_gemm.py`, `tests/test_attention.py`, `tests/test_ops.py` |
| Transformer and text encoder against the official modules, random weights | GPU; a torch + diffusers + transformers environment for the reference side | `tests/ref/mini_dit.py`, `tests/ref/mini_te.py`, then `tests/test_dit_mini.py`, `tests/test_te_mini.py` |
| VAE and the edit-only parts against the official modules, real weights | GPU, the official checkpoint, the reference environment | `tests/ref/vae_decode.py`, `tests/ref/edit_parts.py`, then `tests/test_vae.py`, `tests/test_vae_encoder.py`, `tests/test_edit_parts.py` |
| A converted text encoder against reference embeddings, without the GPU | the converted file, reference embeddings (`tests/ref/reference.py --te-only`) | `tests/check_te_cpu.py`; on the GPU, `bench/te_quality.py` |
| An edit's text-encoder output (vision tower, DeepStack, positions) against the official modules | GPU, a converted model, `tests/ref/reference.py --te-only --image` | `tests/test_edit_te.py` |
| Image quality against the official pipeline | GPU, checkpoint, reference environment | `tests/ref/reference.py`, then `bench/quality.py` |
| Z-Image-Turbo's parts against the official modules, random weights | GPU, the reference environment | `tests/ref/mini_zimage.py`, `mini_qwen3.py`, `mini_flux_vae.py`, `mini_zimage_pipe.py`, then `tests/test_zimage_mini.py`, `test_qwen3_te.py`, `test_flux_vae.py`, `test_zimage_pipe.py` |
| FLUX.2 [klein]'s parts against the official modules, random weights | GPU, the reference environment | `tests/ref/mini_flux2.py` and `mini_flux2_sim.py` (the error quantization alone costs), `mini_flux2_te.py`, `mini_flux2_vae.py`, then `tests/test_flux2_mini.py`, `test_flux2_te.py`, `test_flux_vae.py` |
| Z-Image-Turbo and FLUX.2 [klein] against the official pipelines, real weights | GPU, checkpoint, reference environment | `tests/ref/zimage_reference.py`, `flux2_reference.py`, then `bench/zimage_quality.py`, `bench/flux2_quality.py` |
| Speed | idle GPU, converted model | `bench/dit_layer_bench.py`, `bench/zimage_layer_bench.py`, `bench/flux2_step_bench.py`, `bench/gemm_bench.py`, `bench/attn_bench.py` |

The reference scripts import diffusers' weight conversion script; point
`DIFFUSERS_SCRIPTS` at the `scripts/` directory of a diffusers checkout.
`H3METAL_WEIGHTS` names the official checkpoint and `H3METAL_REF` a working
directory for reference outputs.

CI runs the first row, byte-compiles the Python sources, lints them with
`ruff check .` and checks the English-only rule. It cannot run the GPU tests, so say in the pull request
which of them you ran and on what machine.

## Workflow

1. For anything larger than a fix, open an issue first and describe what you
   want to change and why.
2. Branch from `main`. `main` is protected: changes arrive through pull
   requests, CI has to pass, and pull requests are squash-merged.
3. Write commit and pull-request titles in the imperative ("Add ...", "Fix
   ..."), 72 characters or fewer. In the body, say what changed and what was
   measured or verified.
4. Add an entry under "Unreleased" in `CHANGELOG.md` if users will notice.

## Releases

- Versions follow [Semantic Versioning](https://semver.org). The version lives
  in `pyproject.toml` and `h3metal/__init__.py`.
- A release is a tag `vX.Y.Z` on `main` with the matching `CHANGELOG.md`
  section as its notes.
- Every release carries a wheel with the compiled GPU runtime, so users need
  no compiler: `uv build --wheel -o dist` on an Apple Silicon Mac (the
  library targets macOS 26, CMakeLists.txt), checked by installing it into a
  fresh environment and making an image, then `gh release upload vX.Y.Z
  dist/h3metal-X.Y.Z-py3-none-macosx_26_0_arm64.whl`. The README's install
  line names that wheel.

## The model's license

MiniMax-H3-Image is licensed separately from this code, under the
MiniMax-H3-Image Community License Agreement, which limits where and how the
model and its derivatives may be used and distributed. Converted weight files
are derivatives of the model: anything that distributes them must follow that
license (see the README).
