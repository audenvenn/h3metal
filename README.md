# h3metal

[![CI](https://github.com/audenvenn/h3metal/actions/workflows/ci.yml/badge.svg)](https://github.com/audenvenn/h3metal/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), the
33 B-parameter text-to-image and image-editing model, on Apple Silicon, with
its own Metal engine for the matrix units of the M5/M6 GPU.

<!-- NUMBERS -->

## Quick start

Needs a Mac with Apple silicon and Metal 4 (macOS 26 or later), Python 3.10+
and the Xcode command line tools. The matrix units h3metal is built for are in
M5- and M6-class GPUs.

```sh
pip install "h3metal[edit] @ git+https://github.com/audenvenn/h3metal"

h3metal download                       # the official weights h3metal reads, ~125 GB
h3metal convert ~/.h3metal/checkpoint
h3metal generate -p "A red fox in fresh snow at sunrise" -s 2048x2048 -o fox.png
h3metal generate -p "Put <Picture 1> on a beach at dusk" -i photo.jpg -o edit.png
```

`h3metal download --source modelscope` fetches from ModelScope instead of
Hugging Face. Converting needs the downloaded files only once; afterwards
`~/.h3metal/model` holds what h3metal runs (24-40 GB, see
[formats](#weight-formats)).

## How

- **Precomputed modulation.** Every transformer block projects the timestep
  into six modulation vectors through a 2688 x 96,768 matrix: 13.4 B
  parameters, a third of the transformer. The checkpoint samples on a fixed
  six-step grid, so the converter evaluates every vector the model can use
  and stores the tables (116 MB). Those parameters never reach the GPU.
- **Integer matrix products on the GPU's matrix units.** Weights in int8 or
  int4, activations quantized to int8 per row by the kernel that produces
  them, products accumulated in int32 by the hardware (Metal 4 tensor
  operations): nothing is dequantized in software.
- **Hadamard rotation** of every product's input spreads outlier channels
  before rounding; the random signs are folded into the weights and the
  modulation tables, so the rotation costs three register butterflies and
  five lane shuffles per 256 values.
- **Flash attention** with the operands in registers and the output's
  rescale skipped whenever no row's maximum moved.
- **One command buffer per block** over memory-mapped weights: only the block
  that runs is pinned, so the 33 B transformer runs in 10 GB (int4) and an
  int8 model can be streamed on a 24 GB Mac.

[docs/design.md](docs/design.md) has the details.

## Weight formats

<!-- FORMATS -->

## Image quality

<!-- QUALITY -->

## Benchmarks

<!-- BENCH -->

## The model's license

h3metal's code is Apache-2.0. The model is not: MiniMax-H3-Image is licensed
under the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE),
which, among other terms, grants rights only within its Applicable
Territory (worldwide except the European Union, the United Kingdom, the
Republic of Korea and the United States of America), binds use to an
acceptable use policy, and asks that AI-generated images be disclosed as
such. You obtain the weights from MiniMax (`h3metal download` fetches them
from MiniMax's own repositories) and accept that license by doing so. Files
that `h3metal convert` writes are derivatives of the model and remain under
it.

## Acknowledgements

The reference implementations in
[diffusers](https://github.com/huggingface/diffusers) and
[transformers](https://github.com/huggingface/transformers) (Apache-2.0) are
what h3metal is tested against, and diffusers' conversion script documents
the original checkpoint's layout. The register layout of the matrix units'
SIMD-group operations is the one [MLX](https://github.com/ml-explore/mlx)'s
NAX kernels use.
