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
in half precision when fed from registers (`bench/hw/mma_rate.py`); h3metal's
products reach 31-38 TOPS at the transformer's shapes.

Weights are stored as int8 (W8A8) or int4 (W4A8) with one scale per output
row. Activations are quantized to int8 per row, on the fly, by the kernel
that produces them (the norm, the attention output, the SwiGLU), so every
product runs entirely on the integer path: no weight is ever dequantized in
software. The product's epilogue applies the two scales and, for the
attention output and the MLP, the gated residual update directly to the
float residual stream.

Long products run as several dispatches over slices of K, all accumulating
into the residual stream: exact, and each dispatch's working set fits the
caches better. The MLP's second product (K = 14,336) uses 32 x 256 tiles
with four slices from 2048 x 2048 up and two below; the attention output
(K = 7,168) two slices around 2048 x 2048. Tiles and splits come from
interleaved A/B runs on a warm GPU, which is what a long generation runs on:
a quick sweep on a cool GPU ranks them differently.

A sweep also has to check what it times. A 32 x 512 tile once came out
fastest for the int8 MLP down-projection because it did less work. The grid
covers whole tiles, and the projection's 5376 output columns are 10.5 tiles
of 512, so the last 256 columns were never written, and the residual stream
kept its old values there. The int8 transformer's images were grainy until a
comparison of the weight formats on real weights showed it. The matrix units
were never the problem: wherever 512 divides the columns, 512-column tiles
are exact, though no faster than the tiles in use (0.2-5% slower on both
models' int8 qkv and up-projections). `ops.gemm` now refuses any tile width
that does not divide the columns, and `tests/test_gemm.py` checks every tile
and split the transformers use, at their real shapes and sizes.

At 2048 x 2048 a whole int8 block takes 128 ms, 79% of the time the matrix
units would need at their measured peaks. The four products take 75% of the
block at 35.3 TOPS (84% of the int8 peak), attention 19%, and the norms,
rotary embedding and rotations the remaining 5%. With int4 weights the
products reach 31.5 TOPS, since unpacking the 4-bit codes costs time once
the products are compute-bound; int4 pays where memory is short, not in
speed. docs/benchmarks.md has the profiles.

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
before the softmax. P V stays in half precision. (Queries and keys are rotated
before that rounding; see below.) With the changes below, a
layer's attention takes 24.8 ms at 2048 x 2048 and 0.34 s at 4096 x 4096,
where attention is almost half of a block's time: 81% and 85% of the time
the matrix units need for the two products at their measured peaks
(37.5 ms and 0.56 s with half-precision scores and none of the changes).

Queries and keys are rotated before their int8 rounding (MiniMax-H3-Image).
On the real weights a few channels dominate every query and key row: in the
first block the rotary channels hold 89% of the keys' energy, and in the
worst head the largest channel is 11 times the median one. One int8 scale per
row then leaves the other channels coarse. The kernel multiplies queries and
keys by the same orthonormal matrix, a 128-point Walsh-Hadamard transform
with fixed random signs, which leaves every product q.k unchanged; afterwards
the largest channel is twice the median. That costs about 0.05 ms per block
at 2048 x 2048, under 0.1% of the block.

On the real weights, truncated to the token refiner, the first block and the
output layer (`tests/test_minimax_truncated.py`, base weights), the
transformer's error against float32 goes from 0.87% to 0.74% with int8
weights, and from 1.35% to 1.26% with the int4 MLP. With half-precision
scores it would be 0.46%. Two other fixes were simulated on the CPU, on the
first block's real queries, keys and values:
- removing the keys' mean over the tokens as well (SageAttention's
  smoothing) takes the attention output's error from 0.49% to 0.42%, but it
  needs a second pass;
- moving channel scale from keys to queries (SmoothQuant's migration) does
  not help.

Z-Image-Turbo and FLUX.2 [klein] keep the plain rounding until their
comparisons with diffusers are rerun.

P V in int8 does not pay on this GPU. With V in int8 (one scale per row)
and each block of probabilities quantized per row against its largest
value, the result stays within 1% of the float product, but the extra work
per score (the V scales, the block maximum, the rounding, the scaling back
into the float output) costs more than the faster product saves: 32.2 ms
instead of 29.9 ms per layer at 2048 x 2048, 0.48 s instead of 0.45 s at
4096 x 4096. (The matrix units also reject unsigned 8-bit operands in
registers, which would have halved the probabilities' rounding error.)

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

What holds the kernel back is the barrier at the end of every block, not
the products or the softmax. Timing variants that compute wrong results on
purpose shows it, at 2048 x 2048 with int8 scores: with the staging and its
barrier removed (every block reads the first block's keys and values), the
two products alone run at 97% of the matrix units' peaks and the whole
kernel at 88%; the barrier alone costs 8%, and the global loads and stores
cost nothing without the barrier but another 11% with it. A barrier makes
all SIMD-groups of a threadgroup wait for the slowest one's loads, and
lines them up so that they all start the latency-bound score product at
once (one int8 product of 16 x 32 x 16 needs dozens of others in flight to
keep the matrix units busy). Metal has no barrier that only some
SIMD-groups wait at, nor asynchronous copies into threadgroup memory.

Two changes recover part of it. The softmax does less vector work per
score (`VX`): each lane takes the maximum of its own scores first and
compares it with the running maximum; only when a score exceeds that by
more than 8 (in log2 units) do the four lanes of a row exchange their
maxima and the output get rescaled, so P V sees probabilities up to 2^8,
which half precision holds without loss of relative precision. The
query's scale moves into the exponent's fused multiply-add, and every lane
keeps its own partial row sums until the end. On its own this is 1-2%
faster, but it needs fewer registers, and with it ten SIMD-groups per
threadgroup (160 query rows) beat sixteen: 24.9 ms instead of 27.3 ms per
layer at 2048 x 2048 and 0.35 s instead of 0.39 s at 4096 x 4096 (10%),
5% at 1024 x 1024. Without the leaner softmax, ten SIMD-groups gain 5%.
Nine are slower than sixteen and eight, eleven and twelve are between, so
this is a matter of how the threadgroups pack onto the GPU's cores; it may
differ on other chips. The outputs are as accurate as before: on the
captured inputs of real transformer layers every variant lands within
1.66e-3 of float64 attention, the rounding of the bf16 output.

Also tried, and slower: each SIMD-group reading its keys and values
straight from device memory instead of threadgroup memory, which removes
the barriers (3-4x slower: the cache does not serve 16 SIMD-groups reading
the same rows); loading two blocks ahead; storing the next block right
after the score product instead of at the end of the block; and splitting
the score product into two or four independent accumulator chains
(registers again).

The same decomposition for the current kernel (int8 scores, the leaner
softmax, ten SIMD-groups), every variant timed in one session at 2048 x 2048
(4096 x 4096 in parentheses): the products and the softmax alone run at 95%
(96%) of the floor; a barrier after every block brings that to 92% (93%);
moving the keys and values under those barriers brings it to 82% (85%), the
kernel as it is. Without the barrier the moves cost nothing (as measured
above), so what is left is the barrier coupling every SIMD-group to the
slowest one. No hand-over without it does better on this GPU:

- When every SIMD-group stores a share of each block, none can start block
  b + 1 before the slowest has finished block b. With N buffers filled d
  blocks ahead, the SIMD-groups can drift min(d, N - d - 1) blocks apart:
  none with two buffers. Counters in threadgroup memory in place of the
  barrier keep that coupling and add the cost of polling them: 2.5x slower
  polling with relaxed loads and one acquire fence after, 5x with acquire
  loads.
- A third buffer would allow a block of drift, but at 32 keys a block three
  buffers need 40 KB of threadgroup memory, over the 32 KB limit.
- One extra SIMD-group that only moves blocks (a producer) lets the others
  drift a block apart with two buffers. But moving the blocks alone takes it
  as long as the whole kernel takes (24.8 against 24.5 ms at 2048 x 2048),
  and even when it moves nothing after the first block, the kernel is about
  as fast as with the barrier (24.3 against 24.5 ms; 0.35 against 0.34 s at
  4096 x 4096). With the moves it is 1.8x slower.

Every variant matched the barrier kernel bit for bit before it was timed.
Their release and acquire ordering costs nothing measurable: dropping the
fences changes the time by under 0.5%. So the barrier stays.

## Memory

Weight files (`.h3m`) are mapped read-only and handed to Metal without a copy,
in 1 GiB buffers. Metal pins every buffer a command buffer references, so the
engine submits one command buffer per transformer block: only the block that
runs is pinned, and the kernel pages the rest in and out as needed. With
4-bit weights the transformer's file is 10.8 GB, with int8 attention and
int4 MLP 14.9 GB, and either stays in memory on a 24 GB Mac. Int8 weights
(20.8 GB) do not fit there. Every block is used once per step, always in the
same order, so least-recently-used eviction reads most of the file again at
every step. The engine tries to keep as many blocks resident as Metal's
working-set estimate allows and to stream only the rest: each streamed block
is read two blocks ahead and marked for eviction (`MADV_DONTNEED`) after use.
Measured, that does not hold. The GPU's reads do not mark pages as used, so
the kernel evicts the resident blocks as readily as the streamed ones
(touching their pages from the CPU at every step did not help either), and
six steps at 1024 x 1024 still page in about 100 GB. At 2048 x 2048 and above
a step computes for longer than the SSD needs to read the whole file (it
reads 5.9 GB/s), so int8 costs little there; at 1024 x 1024 it is bound by
the SSD. That is why `h3metal convert` picks the format from the Mac's
memory (`--dit auto`): int8 above 24 GB, int8 attention with int4 MLP at
24 GB, int4 at 16 GB.

The text encoder is mapped while it runs and released before the transformer
starts; it, too, commits one layer at a time, so its 26 GB of int8 weights
stream from the SSD in about 5 seconds with a footprint of about half a
gigabyte. (Asking the kernel to read the next two layers ahead made it
slower, 4.9 s instead of 4.2 s for a 300-token prompt: the kernel's own
read-ahead of a sequential scan does better.) Its products take int8 weights
with half-precision activations: against the official module run in
float32, its output (hidden_states[50]) is within 0.37% (relative L2 error,
mean over nine prompts; mean per-token cosine 0.99996). The int8-activation
path (`H3METAL_TE_ACT=i8`) is faster on long prompts and within 1.24%.
Prompt embeddings are cached on disk (`~/.h3metal/cache`), so a repeated
prompt skips the text encoder.

## VAE

The decoder is a 24-layer ViT applied to overlapping 256-pixel tiles: all
tiles of a size run as one batch, through the same matrix-product and
attention kernels as the transformer (a batch of short sequences in one
dispatch), and are blended exactly as the reference blends them. The
encoder, needed for the reference images of an edit, is a CNN. Its 3x3
convolutions run as nine accumulating half-precision products per output
tile whose left operands are views into the input, stored with a one-pixel
border filled by reflection (kernels/conv.metal; a stride-2 convolution
reads every other pixel and uses only the bottom and right border, as the
reference pads it); GroupNorm and SiLU write straight into that layout. A
2048 x 2048 reference encodes in 3.1 s (3.8 s with im2col, which only the
three-channel first convolution still uses).

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
diffusers and transformers modules: on random weights for the transformer,
on the real weights for the text encoder, the vision tower and the VAE (the
figures are in the README's Image quality section).

Whole images have not been compared with the official pipeline's yet. In
bfloat16 the official pipeline needs about 135 GB. `tests/ref/reference.py`
runs it through the official modules with the transformer streamed one block
at a time. On the CPU that works but takes about an hour per 1024 x 1024
image. On the GPU of a 24 GB Mac it ran into swap within seconds, for a
reason not found yet. A comparison on a larger Mac, or on a GPU server, would
settle it.
