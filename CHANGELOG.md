# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.6.0] - 2026-11-13

### Added

- MiniMax-H3-Image in 2, 3 or 4 steps, with pDMD's 2-step LoRA (Apache-2.0).
  - Commands: `h3metal convert ~/.h3metal/checkpoint --pdmd` fetches the
    adapter (2.8 GB, checked against its SHA-256) and writes the few-step
    transformer, `dit-pdmd.h3m`, next to the six-step one. `h3metal generate
    --steps 2` (or 3, 4) uses it, and the web page offers the step counts.
  - Method: the adapter is merged into the weights in float32 before they
    are quantized, so it costs nothing at run time. The transformer's own
    output head, the grids of diffusers' MiniMaxH3Scheduler (shift 12) and
    its Euler update replace the six-step grid and its per-step heads.
  - Speed on an M6 Mac mini: 15.7 s at 2048 x 2048 in 2 steps instead of 41.7 s
    in 6, and 84.6 s instead of 240.1 s at 4096 x 4096.
  - Checks:
    - On a mini model against diffusers' MiniMax-H3 transformer with the
      adapter merged, the grids and the update are bit for bit the
      scheduler's, and every step is within 0.16%.
    - On real weights, truncated to one block, merging costs no precision
      against float32.
    - Details in docs/few-step.md.
- `bench/speed.py --variant NAME=COMMAND` compares h3metal configurations.

### Changed

- MiniMax-H3-Image's attention rotates queries and keys by a Walsh-Hadamard
  transform before their int8 rounding. Scores are unchanged in exact
  arithmetic, and on real weights the transformer's error against float32
  falls from 0.87% to 0.74%, at under 0.1% of a block's time
  (docs/design.md).
- `h3metal info` lists each MiniMax-H3-Image transformer file with the step
  counts it runs.

## [0.5.1] - 2026-11-13

### Added

- `h3metal download --model flux2-klein-4b` and `--model z-image-turbo`
  fetch h3metal's converted weights from Hugging Face, ready to run.
  - FLUX.2 [klein] 4B: 8.3 GB instead of a 15 GB checkpoint and a
    conversion.
  - Z-Image-Turbo: 10.7 GB instead of 33 GB.
  - The repositories are audenvenn/h3metal-flux2-klein-4b and
    audenvenn/h3metal-z-image-turbo, at tag h3m-1, the layout this version
    reads. They hold the bytes `h3metal convert` makes with its default
    settings, and every file is checked against its SHA-256.
  - `--original` fetches the official checkpoints as before. While a
    converted copy is not published, `download` falls back to them.

## [0.5.0] - 2026-11-13

### Added

- FLUX.2 [klein] 9B (Black Forest Labs, FLUX Non-Commercial License v2.1).
  - Commands: `h3metal download --model flux2-klein-9b --accept-license`,
    then `h3metal convert`, then `h3metal generate --model flux2-9b`, with
    edits and LoRA adapters.
  - Speed: a 1024 x 1024 image takes 14.1 s from a new prompt on an M6
    Mac mini, 2.3 times faster than mflux 0.24 on the same machine.
  - Fidelity: on real weights its transformer matches float32 to 0.11% and
    its text encoder to 0.7-1.0%, as close as 4B's. The check runs on one
    block of each kind and the first text-encoder layers
    (`tests/test_flux2_truncated.py`), because a float32 reference of the
    whole model does not fit the Mac.
- `h3metal download --lora OWNER/NAME[/FILE]` fetches a LoRA adapter into
  `~/.h3metal/loras`, where `--lora NAME` finds it by name.
- The web page offers, for the selected model, the LoRA files that fit it.
  It checks them from their headers.

## [0.4.0] - 2026-11-13

### Added

- LoRA adapters for Z-Image-Turbo and FLUX.2 [klein] (`--lora FILE[:STRENGTH]`,
  repeatable; the web page offers the files in `~/.h3metal/loras`; `loras=`
  in Python).
  - Layouts: diffusers/PEFT, kohya and the original checkpoints' names.
  - Each adapter is applied exactly, as a low-rank term inside each layer's
    product (`kernels/gemm.metal`, `LORA`): no second rounding of the int8
    weights, any strength, several adapters at once.
  - Files h3metal cannot apply exactly (DoRA, LoKr, text-encoder adapters,
    another model's) are refused, never applied in part. See docs/lora.md.
- Each release carries a wheel with the compiled GPU runtime: `pip install`
  of it needs no compiler. The README installs it.

### Changed

- Z-Image conversions keep the modulation's AdaLN output and norm weights
  (16 MB), which adapters of the modulation need. Earlier conversions still
  work, and adapters without modulation layers work on them.
- The converters of both models build their weights from shared plans, which
  the adapters use too. The converted tensors are unchanged, byte for byte.
- The GPU runtime targets macOS 26 explicitly, whatever macOS builds it.
- `h3metal --help` names the three models.

## [0.3.0] - 2026-11-13

### Added

- FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0), text to image and
  edits from reference images: `h3metal download --model flux2-klein-4b`,
  `h3metal convert`, `h3metal generate --model flux2 [-i REF ...]`. A 1024 x 1024 image
  takes 7.4 s from a new prompt on an M6 Mac mini, 2.3 times faster than
  mflux 0.24 with its own 8-bit weights. Its transformer and decoder
  reproduce diffusers' image to 38.5 dB from the same text embeddings
  (docs/flux2.md). An edit lands within 33.7 dB of diffusers' from the
  same inputs.
- The engine: LayerNorm in the fused norm-modulate-quantize kernel, text
  encoder taps (hidden states of several layers) and padded prompts whose
  padding attends to the prompt only, half-precision embedding and output
  layers on the matrix units, FLUX.2's decoder (batch-norm statistics on
  2 x 2 patches, a 1 x 1 convolution before the decoder) and the FLUX VAE's
  encoder for reference images.
- `h3metal serve`: a local web page for making images (models, prompt,
  size, seed, reference images, the images made so far). The model stays
  loaded between images. Standard library only, on 127.0.0.1 by default.

### Changed

- `generate` and `info` without `--model` use the first converted of
  MiniMax-H3-Image, Z-Image-Turbo and FLUX.2 [klein]: the model the user
  has. Before, they always looked for MiniMax-H3-Image.
- The README opens with the three models, and its quick start with FLUX.2
  [klein], the smallest download.

### Fixed

- The Qwen3 reference scripts (tests/ref/mini_qwen3.py, mini_flux2_te.py)
  computed their reference with a rotary inv_freq rounded to bfloat16: they
  cast the model to bfloat16 to save it and back to float to run it, and
  the cast rounds that buffer too, so the reference's angles drifted in
  proportion to the position. They reload the saved model instead. Z-Image's
  text encoder test now measures 0.43% instead of 0.51%.
- `ops.gemm` refuses a tile whose width does not divide the product's
  columns. The grid covers whole tiles, so such a tile left the last columns
  unwritten. That, not the matrix units, is why a 32 x 512 tile once gave
  wrong results on the 5376-column MLP output. The guard it replaces (no
  tiles wider than 256 columns) would have let a narrower tile do the same,
  for example a 256-column tile on 1152 columns. No tile in use was affected.

## [0.2.0] - 2026-11-13

### Added

- Z-Image-Turbo (Tongyi-MAI, Apache-2.0), text to image: `h3metal download --model z-image-turbo`,
  `h3metal convert ~/.h3metal/z-image-turbo`, then `h3metal generate --model zimage` (docs/zimage.md). Its
  transformer runs on the same int8 products and attention, its Qwen3-4B text encoder on h3metal/te.py. On
  an M6 Mac mini a 1024 x 1024 image takes 17.4 s from a new prompt, 2.1 times faster than mflux 0.24 with
  its own 8-bit weights, and the image is within 31.3 dB PSNR of diffusers' in bfloat16.
- Engine: 3x3 convolutions as matrix products without im2col (kernels/conv.metal: batches, strides, zero
  or reflected borders), the FLUX autoencoder's decoder (h3metal/flux_vae.py), a fused normalized-and-gated
  residual update, a row softmax, and GroupNorm statistics that read whole rows.

### Changed

- The VAE encoder of edits uses the new convolutions: a reference encodes in 0.73 s instead of 0.88 s at
  1024 x 1024 and 3.1 s instead of 3.8 s at 2048 x 2048, with the same result.
- Attention with int8 scores is about 10% faster from 2048 x 2048 up (5% at
  1024 x 1024): the softmax updates its running maximum only when a score
  exceeds it by more than 8 (log2 units), keeps per-lane row sums and applies
  the query scale in the exponent's fused multiply-add, and a threadgroup now
  holds ten SIMD-groups instead of sixteen. A transformer block runs at 80%
  of the matrix units' floor instead of 76%; a 4096 x 4096 image takes 236 s
  instead of 244 s.

### Fixed

- The same seed now gives the same image bit for bit: GroupNorm's statistics
  were summed with float atomics, whose order varied from run to run, so the
  VAE encoder (edits) normalized identical inputs slightly differently. They
  are now summed in a fixed order.
- The documentation blamed the softmax's vector work for what attention
  leaves on the table; measurements show it is the barrier after every block
  of keys (docs/design.md).

## [0.1.0] - 2026-11-13

### Added

- MiniMax-H3-Image on Apple Silicon: text-to-image and editing with up to
  16 reference images, from the official checkpoint converted with
  `h3metal convert`.
- Its own Metal engine for the GPU neural accelerators: int8 x int8 and
  int8 x int4 matrix products, flash attention, fused normalization,
  modulation, rotation and quantization.
- The transformer's AdaLN modulation precomputed for the six-step schedule,
  so its 13.4 B parameters never reach the device.
- Attention with the score product in int8 (per-row scales) and
  double-buffered keys and values; long products split over up to four
  accumulating dispatches.
- The transformer's weight format chosen from the Mac's memory
  (`--dit auto`): int8, int8 attention with int4 MLP, or int4.
- The text encoder streamed from the SSD one layer at a time (about half a
  gigabyte resident), and prompt embeddings cached on disk.
- Everything an edit needs on the engine too: the Qwen3-VL vision tower
  (0.6 s per reference) and the VAE encoder (convolutions as im2col and
  half-precision products). The package depends on NumPy, tokenizers and
  Pillow only.
- GPTQ calibration of the 4-bit weights on denoising trajectories, with a
  format per product (e.g. int8 attention, int4 MLP).
- `h3metal download`, `convert`, `generate` and `info`; a Python API
  (`h3metal.Pipeline`). References can be any image macOS opens, HEIC
  included; PNG output records the prompt, seed and size and says that the
  image is AI-generated.
- Tests against the official diffusers and transformers modules, a CPU
  check of a converted text encoder, and benchmarks of every kernel, of
  image quality and of end-to-end speed against another implementation.

[Unreleased]: https://github.com/audenvenn/h3metal/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/audenvenn/h3metal/releases/tag/v0.5.1
[0.5.0]: https://github.com/audenvenn/h3metal/releases/tag/v0.5.0
[0.4.0]: https://github.com/audenvenn/h3metal/releases/tag/v0.4.0
[0.3.0]: https://github.com/audenvenn/h3metal/releases/tag/v0.3.0
[0.2.0]: https://github.com/audenvenn/h3metal/releases/tag/v0.2.0
[0.1.0]: https://github.com/audenvenn/h3metal/releases/tag/v0.1.0
