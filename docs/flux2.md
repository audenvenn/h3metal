# FLUX.2 [klein] on h3metal

[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
is Black Forest Labs' 4 B rectified-flow transformer, distilled to four steps
and released under Apache-2.0. h3metal runs it from text to image and for
edits with one or more reference images.

```sh
h3metal download --model flux2-klein-4b   # h3metal's converted weights, ready to run: 8.3 GB
# or: h3metal download --model flux2-klein-4b --original (15 GB), then h3metal convert ~/.h3metal/flux2-klein-4b
h3metal generate --model flux2 -p "A lighthouse on a rocky coast at dusk" -s 1536x1024 -o lighthouse.png
```

## The model

- **Text encoder.** Qwen3-4B. The pipeline wraps the prompt in Qwen3's chat
  template (thinking off), pads it to 512 tokens and stacks the hidden
  states after layers 9, 18 and 27: 7,680 features per token. h3metal keeps
  the first 27 of the 36 layers, in int8 (3.5 GB).
- **Transformer.** Five double-stream blocks, in which the caption and the
  image have their own weights and share one attention, then twenty
  single-stream blocks, in which attention and the MLP read the same
  normalized rows and share one output projection. 24 heads of 128;
  LayerNorm without affine; a four-axis rotary embedding. The modulation is
  global: one set per stream for every double-stream block and one for every
  single-stream block, from the timestep alone. In int8: 3.9 GB.
- **Decoder.** FLUX.1's decoder with 32 latent channels, a 1 x 1
  convolution before it, and latents normalized with batch-norm statistics
  over their 2 x 2 patches.

## On the engine

h3metal/flux2.py lists what the conversion does; the main points:

- Every block matrix is rotated on its input side and quantized, with the
  rotations' sign flips folded into the modulation tables and into the
  matrices that produce the inputs. The modulation depends on the step's
  timestep, and the schedule's shift on the image's size, so the tables are
  computed on the CPU every step (a few milliseconds). A block's two
  streams share the sign flips of the attention values, since the attention
  mixes their rows.
- The rotary embedding rotates adjacent channel pairs. Permuting every
  head's query and key rows turns that into the engine's half-split
  rotation, as for Z-Image.
- A single-stream block's output projection is split into its attention
  and MLP parts. Their inputs are quantized separately, each row with its
  own scale, so they run as two products accumulating into the residual
  stream.
- The caption's 512 rows come first. 512 is a multiple of every product's
  row tile, so the double-stream blocks' per-stream products never touch
  the other stream's rows.
- The fused norm-modulate-quantize kernel gained LayerNorm: it subtracts
  the row's mean in a second pass over the registers, then takes the
  variance about it (no cancellation).
- The text encoder returns the tapped layers. Padding rows attend to the
  prompt's rows only, themselves excluded, as the reference's padding mask
  has it: one causal attention over the prompt rows and one over the
  padding rows with the prompt as keys.
- The embedding and output layers run as half-precision products on the
  matrix units (38 ms a step as float on the vector units, 0.8 ms now).

## Edits

A reference image is prepared as the official pipeline prepares it: scaled
down to 1024 x 1024 pixels of area if it is larger (Lanczos), then
center-cropped to multiples of 16. The FLUX.2 VAE's encoder (on the
decoder's convolutions, its quant_conv folded into its last one) makes its
latents, and their mean is the encoding. Packed in 2 x 2 patches and
normalized with the batch-norm statistics, the encodings join the image
rows after the generated image's, at their own place on the rotary
embedding's first axis (10 for the first reference, 20 for the second, and
so on). They take the same modulation as the generated rows, and only the
generated rows' output is kept. Without a size, the image takes the first
reference's.

Against diffusers' pipeline in bfloat16 with the same prompt, noise and reference (a 1024 x 1024 image,
"make it a bright sunny afternoon"): the prepared reference pixels are identical, the encoded reference rows
within 0.8%, and the edited image within 33.7 dB PSNR (SSIM 0.988). An edit stays closer to the reference run
than a new image does, since the reference image anchors it. A 1024 x 1024 edit from one 1024 x 1024
reference takes 13.0 s (the transformer then attends over 8,704 rows instead of 4,608).

## Fidelity

Against diffusers' `Flux2KleinPipeline` in bfloat16 on the Mac's GPU
(tests/ref/flux2_reference.py, bench/flux2_quality.py), with the same
prompt and noise, at 1024 x 1024:

| Part | Result |
|---|---|
| Chat template and the 512 tokens | identical |
| Transformer, one step from the reference's input and embeddings | 3.9% (4.4% before 0.6.1) |
| Transformer and decoder, fed the reference's text embeddings | PSNR 39.0 dB, SSIM 0.994 (38.5 dB, 0.995 before 0.6.1) |
| The whole pipeline | PSNR 23.9 dB, SSIM 0.932 (24.2 dB, 0.933 before 0.6.1) |

Since 0.6.1 the queries and keys are rotated before their int8 rounding (docs/design.md); the numbers in
parentheses are without it. The whole pipeline's PSNR follows the fragile padding rows described next more
than the transformer.

The gap between the last two rows is the prompt's padding. 482 of the 512
caption rows are padding tokens, and their hidden states are numerically
fragile: the official pipeline in bfloat16 moves them 12-13% away from the
same text encoder in float32 by layer 18. These rows still steer the image.
A float32 text encoder on the CPU makes the comparison; every image below
comes from h3metal's transformer and decoder (which reproduce diffusers' to
38.5 dB; measured before 0.6.1) and differs only in its text embeddings:

| Text embeddings | Prompt rows vs float32 | Padding rows vs float32 (layers 9 / 18 / 27) | Image vs the float32 embeddings' image |
|---|---:|---:|---|
| diffusers, bfloat16 | 0.8% | 2.6% / 13.1% / 12.2% | PSNR 27.3 dB, SSIM 0.958 |
| h3metal, int8 weights | 0.6% | 3.4% / 25.1% / 20.6% | PSNR 28.3 dB, SSIM 0.966 |

h3metal's prompt rows are closer to float32 than diffusers' bfloat16 ones,
and its image is as close to the float32 image as the official pipeline's.
The official pipeline and h3metal are two equally faithful roundings of the
same model, each 27-28 dB from float32, and so about 24 dB from each other.

The mini-model tests check the parts against random-weight models in float:
the transformer stays within 1.1 times the error its quantization alone
costs (simulated in float with diffusers, tests/ref/mini_flux2_sim.py), the
text encoder within 0.5% (prompt and padding rows alike), the decoder at
68.5 dB.

## Speed

On a Mac mini (M6, 24 GB), against mflux 0.24 on the same machine with its own 8-bit copy of the
model, four steps, every run a fresh process ([method](benchmarks.md#flux2-klein-4b-against-mflux)):

| Size | h3metal | mflux | Speed-up |
|---|---:|---:|---:|
| 1024 x 1024, new prompt every run | 7.4 s | 16.9 s | 2.3x |
| 1024 x 1024, prompt seen before | 6.2 s | 15.3 s | 2.5x |

A 1536 x 1024 image takes 10.5 s. One transformer step at 1024 x 1024 (4,608 rows with the caption) takes
1.24 s, 73% of what the matrix units need at their measured peaks (`bench/flux2_step_bench.py`): the
products take 63% of it, attention 26%, the row kernels 8%. Half-precision products for the embedding and
output layers (0.8 ms a step instead of 38 ms as float) and tiles swept at FLUX.2's shapes (`_ftile`) got it
from 69% to 73%. The next step is a product with two quantized inputs for the single-stream blocks' output
projection: it would accumulate the attention and MLP parts in one pass over the residual stream instead of
two.

## FLUX.2 [klein] 9B

[FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) is the larger klein, released
under the FLUX Non-Commercial License v2.1 (see the README for its terms). It has the same architecture and
the same pipeline as 4B, made wider:

- the transformer has 4,096 channels (32 heads) instead of 3,072;
- it has eight double-stream and 24 single-stream blocks;
- its text encoder is Qwen3-8B, read at the same taps (9, 18 and 27): 12,288 features per token.

```sh
h3metal download --model flux2-klein-9b --accept-license   # 35 GB (--source modelscope works too)
h3metal convert ~/.h3metal/flux2-klein-9b                  # int8, 17 GB, about two minutes, into ~/.h3metal/flux2-klein-9b-h3m
h3metal generate --model flux2-9b -p "A lighthouse on a rocky coast at dusk" -o lighthouse.png
```

Nothing in the engine changed for it: the converter and the transformer read their sizes from the config.

**Tiles.** Every tile `_ftile` picks at 9B's shapes is checked against float64 by `tests/test_gemm.py`. A sweep at
those shapes found the chosen tiles within 0-4% of the best ones, so the rules stay as swept for 4B.

**Fidelity.** A float32 reference of the whole model does not fit a 24 GB Mac (MPS refuses 16 GB of float32
transformer weights for 4B already). So the real weights are checked in two parts:

- the first double-stream and the first single-stream block, with every embedding and modulation layer;
- the text encoder's first four layers.

Both are compared with float32 on the CPU (`tests/ref/flux2_truncated.py`, `tests/test_flux2_truncated.py`).
The same check on 4B, whose whole pipeline is compared with diffusers above, calibrates the numbers:

| Relative error against float32 | 9B | 4B |
|---|---:|---:|
| transformer step, one block of each kind | 0.11% | 0.38% |
| text encoder, hidden state 2 (prompt rows / padding rows) | 0.69% / 0.91% | 0.83% / 1.27% |
| text encoder, hidden state 3 (prompt rows / padding rows) | 0.72% / 1.04% | 0.83% / 1.69% |

The checks feed the transformer h3metal's caption features for a real prompt. A few channels of real ones
are two orders of magnitude above the rest, which random features would not exercise. They compare
hidden state 3 rather than 4 for a reason: transformers normalizes the last hidden state it returns (4 in the
truncated model), and the pipeline's taps never are the last one.

**Speed.** On the Mac mini (M6, 24 GB), against mflux 0.24 with its own 8-bit copy, four steps, every run a
fresh process:

| Size | h3metal | mflux | Speed-up |
|---|---:|---:|---:|
| 1024 x 1024, new prompt every run | 14.1 s (14.1, 14.1) | 31.9 s (32.0, 31.7) | 2.3x |
| 1024 x 1024, prompt seen before (2 runs after a warm-up) | 12.4 s (12.8, 11.9) | 31.8 s (31.6, 31.9) | 2.6x |

mflux's copy was saved once with `mflux-save --model <checkpoint> --base-model flux2-klein-9b --quantize 8`.
The method is the 4B one ([benchmarks](benchmarks.md#flux2-klein-9b-against-mflux)): `bench/speed.py`,
fresh processes, alternating, 60 s pauses.

One transformer step at 1024 x 1024 takes 2.71 s, 71% of what the matrix units need at their measured peaks
(1.93 s). The products take 73% of the step, attention 21%, the row kernels 6%. In int8 the transformer is
10.2 GB and the text encoder's 27 layers 6.5 GB. They take turns in memory, as for 4B. The process footprint peaks at 2.3 GB, besides the 17 GB of weight files it maps. mflux's peaks at 13.9 GB.
