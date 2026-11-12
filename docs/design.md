# How h3metal runs MiniMax-H3-Image

MiniMax-H3-Image is a 33.1 B-parameter diffusion transformer (DiT) that
conditions on a 32 B Qwen3-VL text encoder and decodes with a ViT VAE. In
bfloat16 the three need about 135 GB. This document describes how h3metal
fits them into a Mac and where its speed comes from.

## The work

A request runs three stages, one after the other:

| Stage | What runs | Cost |
|---|---|---|
| Text encoder | 50 of Qwen3-VL's 64 decoder layers (hidden 5120, GQA 64/8, SwiGLU 25600) over the prompt tokens; for an edit, also the 27-layer vision tower over each reference | ~24 B parameters, read once |
| Transformer | 2 refiner blocks over the text rows, then 6 steps x 50 blocks (hidden 5376, 56 heads x 128, SwiGLU 14336) over one packed sequence of text rows, reference rows and image rows (one row per 32 x 32 pixels) | 6.25 PFLOP at 4096 x 4096 |
| VAE decoder | a 24-layer ViT (1024 wide) on overlapping 256-pixel tiles | ~0.1 PFLOP at 4096 x 4096 |

At 4096 x 4096 the sequence has 16,384 image rows; at 2048 x 2048, 4,096.
Almost all the time goes to the transformer's matrix products and attention.

## Precomputed modulation

Every block modulates its two norms with six vectors (shift, scale, gate for
attention and for the MLP) projected from the timestep embedding: a
2688 x 96,768 matrix per block, 13.4 B parameters in all, more than a third of
the transformer. The modulation depends only on the timestep and the row's
modality (image or text), and the checkpoint is distilled to a fixed grid of
six steps; reference images of an edit are held at t = 0.999. So the
converter evaluates every modulation vector the model can use (6 steps x 2
modalities + 1 reference timestep, for each block), with the reference's
own rounding, folds the norm weights in, and stores the tables (116 MB). The
13.4 B parameters never reach the device.

## Matrix products on the neural accelerators

M5- and M6-class GPUs have matrix units in every core, reached through Metal
4's tensor operations (`mpp::tensor_ops::matmul2d`). They multiply integer
operands natively: int8 x int8 and int8 x int4 accumulate into int32. On an
M6 (12-core GPU) the matrix units reach about 42 TOPS in int8 and 20 TFLOPS
in half precision when fed from registers (`tests` microbenchmarks); h3metal's
products reach 30-36 TOPS at the transformer's shapes.

Weights are stored as int8 (W8A8) or int4 (W4A8) with one scale per output
row. Activations are quantized to int8 per row, on the fly, by the kernel
that produces them (the norm, the attention output, the SwiGLU), so every
product runs entirely on the integer path: no weight is ever dequantized in
software. The product's epilogue applies the two scales and, for the
attention output and the MLP, the gated residual update directly to the
float residual stream.

### Rotation

Int8 activations lose precision when a few channels are much larger than the
rest. Every quantized product's input dimension is therefore rotated by a
block-diagonal randomized Hadamard matrix (blocks of 256 channels): the
weights once, at conversion, and the activations in the producing kernel,
where a 256-point transform costs three butterfly stages in registers and
five across SIMD lanes. The product is unchanged (the rotation is
orthonormal), while outliers are spread over their block before rounding.
The random sign flips are folded into the modulation tables and into the
rows of the matrix that produces each activation, so they cost nothing.

### GPTQ

The 4-bit weights can be chosen with GPTQ instead of rounding to nearest
(`h3metal/calib.py`): block by block, the inputs of each product are
collected over real denoising trajectories, and each matrix is quantized
against them.

## Attention

Full self-attention over the packed sequence, 56 heads of 128. The kernel is
a flash-attention variant built on the matrix units' SIMD-group operations:
16 query rows per SIMD-group, keys and values staged in threadgroup memory
per block of 32, scores and the running softmax kept in registers, and the
output's rescale skipped whenever no row's running maximum moved. It reaches
14-15 TFLOPS at the transformer's sequence lengths, against 9.8-15.5 for
MLX's fused attention on the same machine.

## Memory

Weight files (`.h3m`) are mapped read-only and handed to Metal without a copy,
in 1 GiB buffers. Metal pins every buffer a command buffer references, so the
engine submits one command buffer per transformer block: only the block that
runs is pinned, and the kernel pages the rest in and out as needed. With
4-bit weights the transformer takes 10 GB and stays resident; 8-bit weights
(19 GB) can be streamed, the next block read ahead while the current one runs.
The text encoder is mapped while it runs and released before the transformer
starts.

## Exactness

The engine reproduces the official pipeline's math: the packed sequence and
its rotary positions, the float32 round trips of the timesteps, the
modulation's bfloat16 rounding, the tiled VAE decode with its blending, the
8-bit truncation of the output. Its tests compare it with the official
diffusers and transformers modules: on random weights for the transformer and
the text encoder, on the real weights for the VAE and the edit-only parts.
