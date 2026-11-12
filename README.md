# h3metal

[![CI](https://github.com/audenvenn/h3metal/actions/workflows/ci.yml/badge.svg)](https://github.com/audenvenn/h3metal/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), the
33 B-parameter text-to-image and image-editing model, on Apple Silicon, with
its own Metal engine for the matrix units of the M5/M6 GPU.

<!-- HERO -->

On a Mac mini with an M6 (12-core GPU, 24 GB), a 2048 x 2048 image takes
TBD s and a 4096 x 4096 image TBD s, TBD and TBD times faster than
[h3-image-mlx](https://github.com/Anionex/h3-image-mlx) on the same machine,
at TBD dB PSNR against the official pipeline. Details in
[Benchmarks](#benchmarks) and [Image quality](#image-quality).

## Quick start

Needs a Mac with Apple silicon and Metal 4 (macOS 26 or later), Python 3.10+
and the Xcode command line tools (pip fetches CMake and Ninja to build the
GPU runtime). The matrix units h3metal is built for are in M5- and M6-class
GPUs; it is developed and measured on an M6.

```sh
pip install "h3metal[edit] @ git+https://github.com/audenvenn/h3metal"

h3metal download                       # the official checkpoint, ~125 GB
h3metal convert ~/.h3metal/checkpoint  # writes ~/.h3metal/model
h3metal generate -p "A red fox in fresh snow at sunrise" -o fox.png
h3metal generate -p "Put <Picture 1> on a beach at dusk" -i photo.jpg -o edit.png
```

`h3metal download --source modelscope` fetches from ModelScope instead of
Hugging Face. Converting reads the checkpoint once; afterwards
`~/.h3metal/model` holds everything h3metal runs (TBD GB with the default
formats, see [Weight formats](#weight-formats)) and the checkpoint can be
deleted.

## Usage

```sh
h3metal generate -p PROMPT [-s SIZE] [--seed N] [-n COUNT] [-o OUT.png] [-i REF.png ...]
```

- `-s` takes `WIDTHxHEIGHT` (multiples of 32), or an aspect ratio: `16:9` is
  MiniMax's recommended 16 MP size for that ratio (5472 x 3072), `16:9@2k`
  the same ratio at about 4 MP. The default is 2048 x 2048.
- Like the official pipeline, h3metal generates at 4 MP or more and resizes
  a smaller request back down afterwards, so `-s 1024x1024` costs as much as
  2048 x 2048; `--min-area 1024` generates at the requested size instead.
- `-n 4` makes four images with seeds `seed`, `seed + 1`, ...; the prompt is
  encoded once. Prompt embeddings are cached on disk (`~/.h3metal/cache`), so
  a repeated prompt skips the text encoder.
- `-i` makes it an edit. Repeat it for several references and name them
  `<Picture 1>`, `<Picture 2>`, ... in the prompt. References can be PNG, JPEG,
  WebP or anything else macOS opens, HEIC photos included.
- PNG output carries the prompt, seed and size, and says that the image is
  AI-generated, as the model's license asks.
- The very first run compiles the Metal kernels, which adds a few seconds;
  macOS caches them for later runs.

From Python:

```python
from h3metal import Pipeline
from PIL import Image

pipe = Pipeline("~/.h3metal/model")
image, timing = pipe.generate("A red fox in fresh snow at sunrise", 2048, 2048, seed=0)
Image.fromarray(image).save("fox.png")

edited, _ = pipe.edit("Put <Picture 1> on a beach at dusk", [Image.open("photo.jpg")], 2048, 2048)
```

Images come out as `uint8` arrays; `timing` says where the time went.

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
- **Flash attention** with the operands in registers, the score product in
  int8, and the output's rescale skipped whenever no row's maximum moved.
- **One command buffer per block** over memory-mapped weights: only the block
  that runs is pinned, so the transformer runs in TBD GB and the 26 GB text
  encoder streams from the SSD with a footprint of half a gigabyte.
- **GPTQ** chooses the 4-bit codes against real denoising trajectories.

[docs/design.md](docs/design.md) has the details, including what was tried
and did not pay.

## Weight formats

TBD

## Image quality

TBD

## Benchmarks

TBD

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Measured claims only: a change to
speed or numerics comes with the numbers and how they were taken.

## Acknowledgements

The reference implementations in
[diffusers](https://github.com/huggingface/diffusers) and
[transformers](https://github.com/huggingface/transformers) (Apache-2.0) are
what h3metal is tested against, and diffusers' conversion script documents
the original checkpoint's layout. The register layout of the matrix units'
SIMD-group operations is the one [MLX](https://github.com/ml-explore/mlx)'s
NAX kernels use. [h3-image-mlx](https://github.com/Anionex/h3-image-mlx)
showed MiniMax-H3-Image running on a Mac first and is the baseline h3metal
is measured against.
