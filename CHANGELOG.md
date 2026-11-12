# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- MiniMax-H3-Image on Apple Silicon: text-to-image and editing with up to
  16 reference images, from the official checkpoint converted with
  `h3metal convert`.
- Its own Metal engine for the GPU neural accelerators: int8 x int8 and
  int8 x int4 matrix products, flash attention, fused normalization,
  modulation, rotation and quantization.
- The transformer's AdaLN modulation precomputed for the six-step schedule,
  so its 13.4 B parameters never reach the device.
- Attention with the score product in int8 (per-row scales), and the long
  MLP product split over two accumulating dispatches.
- The text encoder streamed from the SSD one layer at a time (about half a
  gigabyte resident), and prompt embeddings cached on disk.
- Everything an edit needs on the engine too: the Qwen3-VL vision tower
  (0.6 s per reference) and the VAE encoder (convolutions as im2col and
  half-precision products). The package depends on NumPy, tokenizers and
  Pillow only.
- GPTQ calibration of the 4-bit weights on denoising trajectories, with a
  format per product (e.g. int8 attention, int4 MLP).
- `h3metal download`, `convert`, `generate` and `info`; a Python API
  (`h3metal.Pipeline`).
- Tests against the official diffusers and transformers modules, a CPU
  check of a converted text encoder, and benchmarks of every kernel, of
  image quality and of end-to-end speed against another implementation.
