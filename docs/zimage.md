# Z-Image-Turbo on the engine

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Tongyi-MAI, Apache-2.0) is the second
model h3metal runs, and the first test of how general its engine is. It is a 6 B single-stream diffusion
transformer distilled to eight steps, conditioned on Qwen3-4B, and decoded by the FLUX autoencoder.

Most of it runs on the parts MiniMax-H3-Image already uses. What it needed beyond them is small: one fused
row kernel, a row permutation at conversion, a convolution kernel, and the decoder built from it.

## The transformer

Z-Image's blocks normalize their input (RMSNorm times 1 + the timestep's scale), run attention or a SwiGLU
MLP, normalize the result again (RMSNorm) and add it to the residual stream through a tanh gate. Two refiner
blocks run over the image rows alone and two over the caption rows alone. Then 30 blocks run over both, the
image rows first.

What runs as for MiniMax-H3:
- the int8 x int8 products with Hadamard-rotated, per-row int8 activations made by the kernels that produce
  them (`norm_mod_q8`, `rot_q8` with SwiGLU);
- attention with the int8 score product and the leaner softmax (docs/design.md#attention);
- the modulation precomputed for the eight-step schedule, so the AdaLN projections never reach the device;
- the rotations' sign flips folded into the tables and the producing matrices.

What is new:
- **The normalized, gated residual update.** `x += tanh(gate) * w * rmsnorm(y)` needs the whole row of a
  product's output, so it cannot be the product's epilogue. Each update instead runs in the same pass as the
  next norm: `norm_mod_q8` with `POST` reads the residual row and the product row, adds the update, writes the
  row back, and goes on to normalize, modulate, rotate and quantize it. Only the update before a change of row
  range (refiners to main blocks) runs on its own (`post_norm_add`).
- **Rotary embedding over three axes (caption index, row, column), on pairs of adjacent channels.** The
  engine's kernel rotates channel i with i + 64. Permuting every head's query and key rows at conversion (even
  channels first, then the odd ones) turns one convention into the other and leaves every score unchanged;
  the angle tables for the three axes are built on the host.
- **Tiles for products with 3,840 inputs.** A checked sweep picks 32 x 256 tiles (37 TOPS against 33 for the
  tiles MiniMax-H3's 5,376-input products use) and 64 x 128 tiles swept four at a time for the MLP
  up-projection (38-39 TOPS).
- **The patch layout.** Z-Image orders a patch's features (dy, dx, channel); the conversion reorders the
  embedding's columns and the output layer's rows to the engine's (channel, dy, dx).

One block with random weights at the real sizes on the M6 (`bench/zimage_layer_bench.py`) takes 57.8 ms at
1024 x 1024 (4,224 rows), 79% of the time the matrix units need at their measured peaks, and 362 ms at
2048 x 2048 (80%). On the released weights the eight steps take 15.0 s at 1024 x 1024 and a whole image
17.4 s from a new prompt (text encoder 0.5-1 s, decoder 0.8 s): 2.1 times faster than mflux
(docs/benchmarks.md).

Weights stay int8 by default: 6.2 GB, which fits every Mac h3metal runs on. Four-bit weights hurt this
model more than MiniMax-H3's: every block normalizes its output before adding it, so weight noise is not
damped by small outputs. On the small test model, 4-bit weights alone cost 8.5% (simulated in float with
diffusers), against 0.6% for int8.

## The text encoder

Qwen3-4B is h3metal/te.py unchanged. Z-Image reads `hidden_states[-2]`, the residual stream after its 35th of
36 layers, so the conversion keeps 35 layers. Qwen3-VL's multimodal rotary embedding with equal positions on
its three axes is plain rotary embedding. The prompt is wrapped in Qwen3's chat template (one user turn, the
assistant turn opened), as Z-Image's pipeline does.

## The FLUX decoder

A convolutional decoder:
- conv_in;
- a middle block: ResNet block, single-head attention over all latent pixels, ResNet block;
- four up blocks of three ResNet blocks each, the first three ending in a 2x nearest upsampling and a
  convolution;
- GroupNorm, SiLU, conv_out.

**Half-precision activations in the up blocks.** conv_in and the middle block run at the latents' size and
keep float activations. The up blocks have up to 64 times as many pixels (a float activation of 256 channels
is 4.3 GB at 2048 x 2048), so they store theirs in half precision, while GroupNorm, padding and the
convolutions' residual sum in float. The last convolution stores its 3 channels, not a padded tile of 128,
and the transformer frees its activations before decoding. h3metal's peak footprint at 2048 x 2048 fell
from 19.7 GB to 7.8 GB. The decoder became faster (less memory traffic) and its result did not change
against the bfloat16 reference (57.2 dB on its latents); against the float test model it is 69.5 dB, far
above an 8-bit image. GroupNorm's statistics are summed in a fixed order, without atomics, so the same seed
gives the same image bit for bit.

**Convolutions without im2col** (`kernels/conv.metal`). The input is stored with a one-pixel zero border.
For a fixed tap, the inputs of 64 consecutive output pixels of a row are 64 consecutive pixels of that
buffer: a plain matrix. So a 3x3 convolution is nine accumulating products whose left operands are views
into the input, with nothing copied. GroupNorm + SiLU writes straight into the bordered layout, and the
upsampling writes its bordered output in one pass. Against im2col plus products this takes the decoder from
1.24 s to 0.86 s at 1024 x 1024 and from 5.9 s to 4.6 s at 2048 x 2048 (random weights; 0.77 s and 4.0 s
with the half-precision up blocks). The GroupNorm
statistics kernel was rewritten along the way: 256-thread threadgroups reading whole rows, instead of one
thread per threadgroup.

**The middle block's attention** has one head of 512 channels. That is too wide for the flash-attention
kernel's registers, so its scores are materialized a chunk of query rows at a time: Q K^T, a row softmax,
P V, all as products on the matrix units. The value bias passes through the softmax (its rows sum to one),
so it is folded into the output projection's bias. At 1024 x 1024 it costs 0.05 s; at 2048 x 2048
(65,536 latent pixels) 1.3 s of the 4.6 s.

## Against the official pipeline, on the released weights

`tests/ref/zimage_reference.py` runs diffusers' `ZImagePipeline` on the Mac's GPU in bfloat16, as the model
card does (the prompt encoded first, so that the text encoder and the transformer need not share the GPU's
memory). `bench/zimage_quality.py` compares h3metal with it part by part, for "A red fox sitting in fresh snow
at sunrise, photorealistic, soft golden light.", seed 42, 1024 x 1024:

| Part | Against the bfloat16 reference |
|---|---|
| Text encoder rows (int8 weights, half activations) | 2.0% |
| Transformer, one step from the reference's input and caption rows (int8) | 3.9% |
| Decoder, on the reference's final latents | 57.2 dB |
| Whole image, same prompt and noise | 31.3 dB PSNR, SSIM 0.974 |

The reference itself computes in bfloat16, so part of each difference is its own rounding. The two images
(diffusers left, h3metal right):

![The same prompt and seed: diffusers in bfloat16 (left) and h3metal in int8 (right)](images/zimage_same_seed.jpg)

## Tests

Small random models, checked against the reference implementations:

| Part | Reference | Test | Result |
|---|---|---|---|
| Transformer, int8 | diffusers | `tests/test_zimage_mini.py` | 0.8% at every one of the eight steps |
| Text encoder | transformers, float32 | `tests/test_qwen3_te.py` | 0.5% half / 2.2% int8 |
| FLUX decoder | diffusers `AutoencoderKL` | `tests/test_flux_vae.py` | 72.5 dB |
| Whole pipeline, prompt to pixels | diffusers `ZImagePipeline` | `tests/test_zimage_pipe.py` | 56.8 dB |

The whole-pipeline test uses Z-Image's real tokenizer. It covers the chat template, the torch-identical noise,
the schedule, the VAE scaling and the postprocessing. `tests/test_gemm.py` checks every tile Z-Image uses at
its real shapes against float64.

The schedule is diffusers' `ZImagePipeline` with eight steps: sigmas linspace(1, 1/8, 8) shifted by 3, eight
transformer passes, the number the model card gives for the Turbo model.
