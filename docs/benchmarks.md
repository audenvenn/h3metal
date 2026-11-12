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
| 2048 x 2048 | 41.2 s | 115.4 s | 2.8x | 2.6 GB | 2.9 GB |
| 2048 x 2048, new prompt | 46.5 s | 118.1 s | 2.5x | 2.6 GB | 2.9 GB |
| 4096 x 4096 | 244.2 s | 458.3 s | 1.9x | 5.4 GB | 6.9 GB |

Single runs: h3metal 11.0 and 9.2 s at 1024 x 1024, 40.8 and 41.5 s at
2048 x 2048; h3-image-mlx 45.7 and 45.7 s, 115.3 and 115.5 s.

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
4096): the block time is the best of three runs, the kernels' times come
from one profiled run, and repeated runs vary by a few percent. Products
are compared with the int8 peak; attention with the time its two products
would take at peak (scores in int8 at 42.0 TOPS, P V in half precision at
20.6 TFLOPS); the block with the sum of those, its floor.

| int8 weights | 2048 x 2048 (4,396 rows) | 4096 x 4096 (16,684 rows) |
|---|---:|---:|
| Attention input (Q, K, V) | 27.4 ms, 37.1 TOPS, 88% | 106.2 ms, 36.3 TOPS, 86% |
| Attention | 28.2 ms, 71% | 395.5 ms, 73% |
| Attention output | 10.3 ms, 33.0 TOPS, 79% | 38.6 ms, 33.3 TOPS, 79% |
| MLP up | 37.3 ms, 36.3 TOPS, 86% | 140.6 ms, 36.6 TOPS, 87% |
| MLP down | 22.2 ms, 30.5 TOPS, 73% | 84.9 ms, 30.3 TOPS, 72% |
| Norms, rotary embedding, rotations | 6.4 ms | 26.1 ms |
| Whole block | 132.5 ms, 76% of 100.7 ms | 780.4 ms, 76% of 594.9 ms |

With int4 weights the products reach 31.5 TOPS (75%) at 2048 x 2048 and
31.8 TOPS (76%) at 4096 x 4096: unpacking the 4-bit codes costs time once
a product is compute-bound. A whole block takes 141.5 ms (71% of its floor)
and 816.1 ms (73%). The default format on a 24 GB Mac, int8 attention with
int4 MLP, sits in between.
