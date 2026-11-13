# The engine, and adding a model

h3metal is two things: a small engine for diffusion transformers on the matrix units of M5/M6-class GPUs,
and the models built on it (MiniMax-H3-Image, Z-Image-Turbo). This page describes the engine's parts and the
steps that took Z-Image from nothing to a pipeline matching diffusers. Follow the same steps for another
model.

## What the engine provides

**Runtime**
- `src/` (C++/Objective-C++) and `h3metal/gpu.py` (ctypes). Kernels compile at run time from `kernels/`,
  specialized by macros and cached per macro set.
- Tensors are views into Metal buffers, and weight files are mapped without copies.
- `launch()` encodes a dispatch and `commit()` ends a command buffer. `wait()` blocks; it is the point where
  the host may read or write shared memory. `profile()` times each kernel on its own.

**Weights**
- `h3metal/weights.py`: the `.h3m` format, a JSON header and aligned tensors. Its `Writer` stores quantized
  linear layers (`linear()`) and plain tensors. `WeightFile` maps a file, and can read ahead, release and
  touch pages for transformers that stream from the SSD.
- `h3metal/st.py` reads safetensors, sharded or not. `h3metal/rng.py` reproduces torch's CPU `randn`, so
  noise matches the reference pipelines for a seed.

**Quantization**
- `h3metal/quant.py` rotates every quantized matrix's input dimension by a block-diagonal randomized
  Hadamard matrix (blocks of 256 channels) and quantizes it: int8 or int4 per row, int4 per group of 128.
- The kernel that produces a product's input applies the same rotation to the activations and quantizes them
  to int8 per row, so outlier channels are spread before quantization. Every quantized product's input
  dimension must therefore be a multiple of 256.
- `h3metal/calib.py` chooses int4 codes with GPTQ (its block runner is MiniMax-H3's).

**Kernels**

| File | Kernels |
|---|---|
| `gemm.metal` | products on the matrix units: int8 x int8, int8 x int4 (per row or per group), half, bf16, and half x int8/int4. Epilogues: store, or a gated residual update `x += gate[row type] * (A W^T + b)`. A long K can be split over accumulating dispatches |
| `attention.metal` | flash attention: half or int8 scores (per-row scales), grouped-query heads, causal masks, a batch of short sequences in one dispatch, the leaner softmax |
| `ops.metal` | `norm_mod_q8`: RMSNorm, modulation, Hadamard rotation and int8 quantization of a row in one pass, optionally first adding a normalized, gated product output (`POST`); `rot_q8`: rotation and quantization, optionally of value * silu(gate); `qk_norm_rope`: per-head RMSNorm, rotary embedding on the first `rot` channels (half-split pairs) and the split into Q, K, V, optionally as int8 with scales; LayerNorm, GroupNorm (+ SiLU), row softmax, GELU, im2col |
| `conv.metal` | 3x3 convolutions as nine accumulating products over views of an input stored with a one-pixel border (zeros or reflection), any batch, stride 1 or 2; helpers that build those bordered inputs |

**Text encoders**
- `h3metal/te.py` runs Qwen3-family decoders (Qwen3, and Qwen3-VL's text model with its multimodal rotary
  embedding). It stops at any layer and streams its weights one layer at a time.

**Decoders**
- `h3metal/flux_vae.py`: the FLUX autoencoder's decoder (16 latent channels), on `conv.metal`.
- `h3metal/vae.py`: MiniMax-H3's ViT decoder, and its CNN encoder for edits.

## Conventions that matter

- **Rows.** Activations have a row per token and buffers are padded to 128 rows. The product kernel runs
  whole tiles of rows and writes them all, so every buffer a product touches must have room for the padded
  rows.
- **Rotary embedding.** It pairs channel i with i + rot/2. A model that rotates adjacent pairs (Z-Image,
  FLUX) gets its query and key rows permuted per head at conversion, even channels first. Scores do not
  change, because queries and keys are permuted alike.
- **Modulation.** In a distilled model the timesteps are fixed, so the timestep-dependent scales, shifts and
  gates are computed at conversion, one table per step. The norm weights and the rotations' sign flips are
  folded into the same tables.
- **Tiles must be checked.** A product tile that looked fastest in a sweep once skipped part of the work. Run
  `tests/test_gemm.py` (`check_dit_tiles`) for every tile at the shapes the model uses, and sweep only
  candidates whose results match a float64 reference.

## Adding a model: the steps Z-Image took

1. **A small reference.** Build the model's official module (diffusers, transformers) with random weights.
   Keep every quantized dimension a multiple of 256 and the real head dimension. Save it in the original
   checkpoint layout and record the reference outputs: `tests/ref/mini_zimage.py`, `mini_qwen3.py`,
   `mini_flux_vae.py`.
2. **A converter.** Map checkpoint names to `.h3m` entries. Concatenate the fused matrices (q|k|v, value|gate),
   rotate and quantize them, fold the signs, and precompute the modulation tables. Reorder anything whose
   layout differs from the engine's: rotary pairs, patch feature order (`convert_transformer` in
   `h3metal/zimage.py`).
3. **A block.** Write the block as a sequence of engine ops on preallocated buffers. Z-Image's is about 25
   lines (`ZImageDiT._block`). Something the engine lacks becomes a kernel in `ops.metal` with a launcher in
   `ops.py`: Z-Image needed one, the normalized, gated residual update.
4. **Tests per part** against the small reference: `tests/test_zimage_mini.py`, `test_qwen3_te.py`,
   `test_flux_vae.py`.
5. **The pipeline and the CLI.** Cover tokenization (with the model's chat template, if any), noise, the
   schedule, the decoder's latent scaling and the postprocessing. Then test the whole path against the
   official pipeline on the small models with the real tokenizer (`tests/test_zimage_pipe.py`). Teach
   `convert`, `generate` and `info` to recognize the model, and write which model made an image into its
   PNG metadata.
6. **Speed.** Time one full-size block with random weights against the matrix units' floor
   (`bench/zimage_layer_bench.py`) before the real weights arrive. Then sweep checked tiles at the model's
   shapes.
7. **Real weights.** Compare part by part with a reference run of the official pipeline
   (`tests/ref/zimage_reference.py`, `bench/zimage_quality.py`). Compare end to end with the best existing
   Mac implementation (`bench/speed.py --other ...`).

## What FLUX.2 [klein] added

FLUX.2 [klein] 4B (`h3metal/flux2.py`) took the same steps. It needed two
engine changes: LayerNorm in `norm_mod_q8`, and text encoder taps with
padding rows that attend to the prompt only. Four lessons for the next model:

- **A small reference has to exercise the blocks.** With small random
  weights FLUX.2's gates came out near 0.03, every block added almost
  nothing, and the test passed at 0.01% while seeing almost none of the
  blocks. `tests/ref/mini_flux2.py` scales the modulation to order 1 and
  prints the blocks' share of the output (96%); check that share.
- **Attribute a small model's error before trusting a threshold.**
  `tests/ref/mini_flux2_sim.py` applies h3metal's rounding (rotated int8
  weights and activations, int8 queries and keys) inside the official module
  in float. The engine's error has to match what that rounding alone costs
  (`tests/test_flux2_mini.py` checks 1.1 times it), so a few-percent error
  is either explained or a bug.
- **A reference script can be wrong too.** Casting a transformers model to
  bfloat16 to save it, and back to float to run it, also rounds its rotary
  `inv_freq` buffer. The reference's angles then drift with the position.
  The text encoder test showed an error growing with the row index that no
  kernel explained, until the reference itself was compared with a reloaded
  model. Reload, never cast back.
- **When the end-to-end number disagrees, measure every part against
  float32 first.** FLUX.2's whole image matched diffusers to only 24 dB
  while its transformer matched to 38.5 dB. A float32 text encoder on the
  CPU showed why: the padding tokens' hidden states are fragile in any
  rounding, the official bfloat16 one included, and h3metal's are as close
  to float32 as the official ones (docs/flux2.md). Raising the precision
  would have cost 4.5 GB and changed nothing.

## What LoRA adapters added

Adapters (`h3metal/lora.py`, [docs/lora.md](lora.md)) needed one engine
change. A product can carry a low-rank term (`ops.LowRank`), which runs as a
narrow product of the same activations plus a short second product added in
the main product's epilogue. Three lessons:

- **Check a hardware property, then build on it.** The epilogue adds the
  low-rank product's destination registers to the main product's element by
  element. That is only right if an int8 product and a half-precision one of
  the same tile shape lay out their destinations the same way. Nothing
  documents that, so it was measured first on every tile shape in use,
  grouped and not. `tests/test_gemm.py` keeps checking it, so a GPU that
  differs fails a test instead of drawing wrong images.
- **Describe a transform once.** The converters permute, sign-flip, fuse,
  split and rotate the original layers. An adapter has to go through exactly
  the same transforms. Writing them a second time for the adapters would
  have made two copies to keep in step. Instead both read the same plans
  (`model_plans`), and a CPU test checks that an adapter's terms equal the
  plans' transform of its dense update. Rebuilding the converters on the
  plans changed no converted tensor, byte for byte.
- **Keep what a precomputed table was made from.** Z-Image's modulation
  tables fold norm weights and rotation signs into the AdaLN output. An
  adapter of the AdaLN projection moves that output, and the tables cannot
  be unfolded. The converter now keeps the output and the norm weights
  (16 MB) next to the tables.
