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

Long products run as several dispatches over slices of K, all accumulating
into the residual stream: exact, and each dispatch's working set fits the
caches better. The MLP's second product (K = 14,336) uses four slices around
2048 x 2048 and two otherwise, with 32 x 512 tiles; the attention output
(K = 7,168) two slices around 2048 x 2048. Tiles and splits come from
interleaved A/B runs on a warm GPU, which is what a long generation runs on:
a quick sweep on a cool GPU ranks them differently. At 2048 x 2048 a whole
block takes 139 ms: the four products about 73% (33 TOPS on average),
attention 22%, the norms, rotary embedding and rotations the remaining 5%.

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
output's rescale skipped whenever no row's running maximum moved. In half
precision it reaches 14-15 TFLOPS at the transformer's sequence lengths,
against 9.8-15.5 for MLX's fused attention on the same machine.

The transformer runs the score product in int8: the kernel that applies the
query/key norms and the rotary embedding also quantizes every query and key
row to int8 with its own scale, the scores accumulate in int32 at twice the
half-precision rate, and the two scales are applied to each block of scores
before the softmax. P V stays in half precision. A layer's attention takes
29.6 ms instead of 37.5 ms at 2048 x 2048 and 0.46 s instead of 0.56 s at
4096 x 4096, where attention is half of a block's time.

P V in int8 does not pay on this GPU. With V in int8 (one scale per row)
and each block of probabilities quantized per row against its largest
value, the result stays within 1% of the float product, but the extra work
per score (the V scales, the block maximum, the rounding, the scaling back
into the float output) costs more than the faster product saves: 32.2 ms
instead of 29.9 ms per layer at 2048 x 2048, 0.48 s instead of 0.45 s at
4096 x 4096. The attention is bound by vector work, not by the matrix
units. (The matrix units also reject unsigned 8-bit operands in registers,
which would have halved the probabilities' rounding error.)

Keys and values are double-buffered in threadgroup memory: each thread loads
its share of the next block into registers before the current block's
products and stores it afterwards, so the loads overlap the products and one
barrier per block remains instead of two (int8 keys leave room for two
buffers). With 16 SIMD-groups per threadgroup this is 2% faster than the
single-buffered kernel at 4096 x 4096. Ordering P V by output dimension
first (each pair of output fragments takes both of the block's value
fragments back to back) is another 4-5%: 27.3 ms per layer at
2048 x 2048, 0.38 s at 4096 x 4096. Also tried, and slower: 32 query
rows per SIMD-group so that every key and value fragment feeds two products
(register pressure: 2-5x slower), and accumulating each block's P V in half
precision (the matrix units multiply half x half -> half no faster than
half x half -> float: 18.7 against 19.2 TFLOPS).

## Memory

Weight files (`.h3m`) are mapped read-only and handed to Metal without a copy,
in 1 GiB buffers. Metal pins every buffer a command buffer references, so the
engine submits one command buffer per transformer block: only the block that
runs is pinned, and the kernel pages the rest in and out as needed. With
4-bit weights the transformer takes 10 GB, with int8 attention and int4 MLP
13.5 GB, and either stays resident on a 24 GB Mac. Int8 weights (19.3 GB) do
not fit there, and plain least-recently-used eviction would then read the
whole file again at every step, since every block is used once per step in
the same order. So the engine keeps as many blocks resident as Metal's
working-set estimate allows (less room for activations) and streams only the
rest: each streamed block is read two blocks ahead and marked for eviction
(`MADV_DONTNEED`) right after use, so the kernel drops those pages rather
than the resident ones.
The text encoder is mapped while it runs and released before the transformer
starts; it, too, commits one layer at a time, so its 26 GB of int8 weights
stream from the SSD in 4-5 seconds with a footprint of about half a
gigabyte. Its products take int8 weights with half-precision activations:
against the official module run in float32, its output (hidden_states[50])
is within 0.47% (relative L2 error; mean per-token cosine 0.99995). The
int8-activation path (`H3METAL_TE_ACT=i8`) is faster on long prompts and
within 1.4%. Prompt embeddings are cached on disk (`~/.h3metal/cache`), so a
repeated prompt skips the text encoder.

## VAE

The decoder is a 24-layer ViT applied to overlapping 256-pixel tiles: all
tiles of a size run as one batch, through the same matrix-product and
attention kernels as the transformer (a batch of short sequences in one
dispatch), and are blended exactly as the reference blends them. The
encoder, needed for the reference images of an edit, is a CNN: every
convolution runs as an im2col pass and a half-precision product on the
matrix units, with GroupNorm and SiLU fused into one pass. A 2048 x 2048
reference encodes in about 4 seconds.

## Edits

The vision tower of Qwen3-VL (27 layers) turns each reference image into
the features of its vision block in the text encoder's input, plus three
DeepStack feature sets that are added to those rows after the first three
decoder layers. It runs on the same engine, half-precision products with
float accumulation (the head dim of 72 padded to 96 for the attention
kernel), 0.6 s for a 1024 x 1024 reference. The
references then enter the transformer as their own rows, held at
t = 0.999, ahead of the image rows.

## Exactness

The engine reproduces the official pipeline's math: the packed sequence and
its rotary positions, the float32 round trips of the timesteps, the
modulation's bfloat16 rounding, the tiled VAE decode with its blending, the
8-bit truncation of the output. Its tests compare it with the official
diffusers and transformers modules: on random weights for the transformer and
the text encoder, on the real weights for the VAE and the edit-only parts.
