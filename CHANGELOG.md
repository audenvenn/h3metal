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
