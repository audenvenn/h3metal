# Benchmarks

Everything here was measured on one machine, back to back, with nothing
else using the GPU. Commands and raw results are given so that every number
can be reproduced.

## Machine

| | |
|---|---|
| Computer | Mac mini (Mac18,5) |
| Chip | Apple M6: 12-core CPU (2 super, 4 performance, 6 efficiency cores), 12-core GPU with matrix units, Metal 4 |
| Memory | 24 GB (Metal's recommended working set: 19.1 GB) |
| System | macOS 27.2 |

## End to end, against h3-image-mlx

[h3-image-mlx](https://github.com/Anionex/h3-image-mlx) 0.1.0a3 (MLX
0.32.4) runs the Comfy-Org INT8 ConvRot weights through its W8A8 TensorOps
path; on a Mac with less than 32 GB its default `auto` mode streams the
transformer one block at a time.

Method (`bench/speed.py`):

- every run is a fresh process, as from the command line, timed by wall
  clock from launch to the PNG on disk;
- the two implementations alternate, with a 60 s pause before every run, so
  thermal drift affects both alike; one warm-up run of each per size is not
  recorded;
- memory is the peak physical footprint the kernel records for each process
  (the figure Activity Monitor shows), sampled over the whole process tree;
  h3-image-mlx runs each stage in a child process and is charged for the
  largest one;
- both implementations cache prompt embeddings, so a repeated prompt skips
  the text encoder; the tables give both the repeated-prompt time and the
  time with a new prompt every run;
- the same prompt, seed and canvas for both. h3metal is run with
  `--min-area 1024` so that it, like h3-image-mlx, generates at exactly the
  requested size (by default both MiniMax's pipeline and h3metal generate a
  1024 x 1024 request on a 2048 x 2048 canvas).

`bench_run`: `bench/speed.py --sizes 1024,2048 --runs 2 --warmup`, then
`--sizes 2048 --runs 1 --cold` and `--sizes 4096 --runs 1`, with h3metal's
default formats for a 24 GB Mac (int8 attention, int4 MLP) and the prompt
"A red fox sitting in fresh snow at sunrise, photorealistic, soft golden
light.", seed 42. Means of two runs (one at 4096 x 4096):

| Size | h3metal | h3-image-mlx | Speed-up | h3metal peak footprint | h3-image-mlx peak footprint |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024 | 10.1 s | 45.7 s | 4.5x | 0.6 GB | 2.5 GB |
| 2048 x 2048 | 40.6 s | 115.4 s | 2.8x | 2.6 GB | 2.9 GB |
| 2048 x 2048, new prompt | 46.5 s | 118.1 s | 2.5x | 2.6 GB | 2.9 GB |
| 4096 x 4096 | 236.4 s | 458.3 s | 1.9x | 5.4 GB | 6.9 GB |

Single runs: h3metal 11.0 and 9.2 s at 1024 x 1024, 40.6 and 40.6 s at
2048 x 2048; h3-image-mlx 45.7 and 45.7 s, 115.3 and 115.5 s.

The h3metal times at 2048 x 2048 (repeated prompt) and 4096 x 4096 are from
after the attention changes (2026-11-13), measured the same way against
0.1.0 itself, run by run in turn: 0.1.0 took 41.3 and 41.1 s at
2048 x 2048, as in the comparison above (40.8 and 41.5 s), and 243.9 s at
4096 x 4096 (244.2 s above). The other figures are from the 0.1.0
comparison (2026-11-12), where attention is a small part of the time.

With int8 weights throughout, like h3-image-mlx's, h3metal takes 44.9 and
45.0 s at 2048 x 2048 (2.6x). Those two runs were made on their own right
after the comparison, with the same warm-up and pauses.

Neither implementation holds the whole transformer in its own memory on
this Mac. h3metal maps its weight files, which macOS keeps in its file
cache, and a process's footprint does not count that cache; h3-image-mlx
loads the transformer one block at a time. The footprints are what each
process allocates beyond that.

At the same seed the two draw nearly the same picture (h3metal on the
left), the differences coming from their different quantizations:

![The same prompt and seed, h3metal and h3-image-mlx](images/same_seed.jpg)

## Z-Image-Turbo, against mflux

[mflux](https://github.com/filipstrand/mflux) 0.24.0 (MLX 0.32.4) with its own 8-bit copy of the
released model, saved once with `mflux-save --model <checkpoint> --base-model z-image-turbo --quantize 8`, so
that neither implementation quantizes at startup (h3metal runs the int8 files `h3metal convert` wrote).
Both run eight transformer passes (`--steps 8`). Same method as above (`bench/speed.py`: fresh processes,
alternating, 60 s pauses, the h3metal command `h3metal generate --model zimage`), prompt "A red fox sitting
in fresh snow at sunrise, photorealistic, soft golden light.", seed 42:

| Size | h3metal | mflux | Speed-up |
|---|---:|---:|---:|
| 1024 x 1024, prompt seen before (2 runs after a warm-up) | 16.2 s (16.5, 15.8) | 36.9 s (37.0, 36.7) | 2.3x |
| 1024 x 1024, new prompt every run (2 runs) | 17.4 s (17.3, 17.4) | 36.9 s (37.0, 36.8) | 2.1x |
| 2048 x 2048 (1 run) | 97.1 s | 217.4 s | 2.2x |

mflux encodes every prompt anew (it pads it to 512 tokens), so the new-prompt row compares like with like.
The 2048 x 2048 h3metal run reused the cached prompt, which saves it about a second.

The two draw different pictures for the same seed: mflux draws its noise with MLX's generator and shifts the
schedule by the image's size (mu 1.15 at 1024 x 1024), where diffusers and h3metal draw torch's noise and shift
by 3.

Memory does not compare like for like from the process footprints (h3metal 2.2 GB at 1024 x 1024 and 7.8 GB
at 2048 x 2048; mflux 10.4 and 15.5 GB). h3metal maps its weight files, which macOS keeps in its file cache
and a footprint does not count, while mflux reads its 10 GB of weights into its own memory. Counting the
6.1 GB transformer file that must stay resident while it denoises, h3metal needs about 8.4 GB at 1024 x 1024
against mflux's 10.4 GB. The resident set size does not help either: macOS leaves the GPU's buffers out of
it (h3metal 0.6 GB, mflux 5.3 GB). Run as its README shows, without quantization, mflux reads the released
float32 transformer into bfloat16 (about 20 GB with the text encoder). On this 24 GB Mac it filled 11 GB of
swap and was stopped.

Commands: `bench/speed.py --sizes 1024 --runs 2 --warmup`, then `--sizes 1024 --runs 2 --cold` and
`--sizes 2048 --runs 1`, each with `--h3metal "h3metal generate --model zimage"` and `--other
"mflux-generate-z-image-turbo --model <the 8-bit copy> --base-model z-image-turbo --steps 8"`; the single runs
are in the table.

## FLUX.2 [klein] 4B, against mflux

mflux 0.24.0 with its own 8-bit copy of the released model (`mflux-save --model <checkpoint> --base-model
flux2-klein-4b --quantize 8`). Both run four transformer passes (`--steps 4 --guidance 1.0` for mflux).
Same method as above (`bench/speed.py`: fresh processes, alternating, 60 s pauses, the h3metal command
`h3metal generate --model flux2`), the same prompt, seed 42:

| Size | h3metal | mflux | Speed-up |
|---|---:|---:|---:|
| 1024 x 1024, prompt seen before (2 runs after a warm-up) | 6.2 s (6.7, 5.8) | 15.3 s (15.6, 15.0) | 2.5x |
| 1024 x 1024, new prompt every run (2 runs) | 7.4 s (7.4, 7.4) | 16.9 s (17.7, 16.2) | 2.3x |

Both encode a new prompt's 512 padded tokens, so the second row compares like with like. A 1536 x 1024 image
takes h3metal 10.5 s. The process footprints are 2.2 GB for h3metal, which maps its 8.2 GB of weight files,
and 8.9 GB for mflux; as with Z-Image they do not compare memory like for like (above).

One transformer step at 1024 x 1024 (4,608 rows with the caption) takes 1.24 s, 73% of what the matrix units
need at their measured peaks (`bench/flux2_step_bench.py`). The products take 63% of the step, attention
26%, the row kernels (quantization, query and key norms, the modulated norms) 8%.

## Kernels

`bench/hw/mma_rate.py` measures the matrix units on register fragments,
the ceiling everything else is compared with:

| Operands | Rate |
|---|---:|
| int8 x int8 -> int32 | 42.0 TOPS |
| half x half -> float | 20.6 TFLOPS |
| half x half -> half | 20.3 TFLOPS |
| bfloat x bfloat -> float | 20.7 TFLOPS |

Half-precision accumulation is no faster than float accumulation, which is
why P V in attention keeps float accumulators.

`bench/gemm_bench.py`, `bench/attn_bench.py` and `bench/dit_layer_bench.py`
time the transformer's products, its attention and a whole block with
random int8 weights. Tile shapes and splits were chosen with interleaved
A/B runs on a warm GPU (a long generation runs warm, and a quick sweep on a
cool GPU ranks them differently).

Every candidate must also give correct results: a sweep once ranked a tile
first that skipped part of the work (see docs/design.md), and
`tests/test_gemm.py` now checks every tile the transformer uses at its real
shapes.

One transformer block, `FMT=q8 python bench/dit_layer_bench.py 2048` (and
4096): each run of the script reports the best of three block times and
one profiled run's kernel times, and repeated runs vary by a few percent. Products
are compared with the int8 peak; attention with the time its two products
would take at peak (scores in int8 at 42.0 TOPS, P V in half precision at
20.6 TFLOPS); the block with the sum of those, its floor.

| int8 weights | 2048 x 2048 (4,396 rows) | 4096 x 4096 (16,684 rows) |
|---|---:|---:|
| Attention input (Q, K, V) | 27.1 ms, 37.5 TOPS, 89% | 101.9 ms, 37.9 TOPS, 90% |
| Attention | 24.8 ms, 81% | 339.6 ms, 85% |
| Attention output | 10.2 ms, 33.1 TOPS, 79% | 37.9 ms, 34.0 TOPS, 81% |
| MLP up | 36.3 ms, 37.3 TOPS, 89% | 149.6 ms, 34.4 TOPS, 82% |
| MLP down | 22.3 ms, 30.4 TOPS, 72% | 83.8 ms, 30.7 TOPS, 73% |
| Norms, rotary embedding, rotations | 6.4 ms | 24.4 ms |
| Whole block | 128.1 ms, 79% of 100.7 ms | 735.8 ms, 81% of 594.9 ms |

Every figure is the median of three runs of the script (blocks of
127.9-128.3 ms and 729.2-743.3 ms; single kernels vary more, the MLP up
product at 4096 x 4096 from 148 to 155 ms). In 0.1.0, before the attention
changes in docs/design.md, the block took 132.5 ms (76%) and 780.4 ms
(76%), attention 28.2 ms (71%) and 395.5 ms (73%).

With int4 weights the products reach 31.9 TOPS (76%) at 2048 x 2048 and
31.8 TOPS (76%) at 4096 x 4096: unpacking the 4-bit codes costs time once
a product is compute-bound. A whole block takes 137.7 ms (73% of its floor)
and 776.3 ms (77%). The default format on a 24 GB Mac, int8 attention with
int4 MLP, sits in between.
