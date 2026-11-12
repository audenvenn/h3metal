# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- Z-Image-Turbo (Tongyi-MAI, Apache-2.0), text to image: `h3metal download --model z-image-turbo`,
  `h3metal convert ~/.h3metal/z-image-turbo`, then `h3metal generate --model zimage` (docs/zimage.md). Its
  transformer runs on the same int8 products and attention, its Qwen3-4B text encoder on h3metal/te.py.
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

[Unreleased]: https://github.com/audenvenn/h3metal/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/audenvenn/h3metal/releases/tag/v0.1.0
