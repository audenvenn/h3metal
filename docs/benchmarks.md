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

TBD

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
| Attention input (Q, K, V) | 27.4 ms, 37.1 TOPS, 88% | 102.7 ms, 37.6 TOPS, 89% |
| Attention | 28.2 ms, 71% | 392.1 ms, 74% |
| Attention output | 10.3 ms, 33.0 TOPS, 79% | 41.7 ms, 30.9 TOPS, 73% |
| MLP up | 37.3 ms, 36.3 TOPS, 86% | 141.0 ms, 36.5 TOPS, 87% |
| MLP down | 22.2 ms, 30.5 TOPS, 73% | 84.7 ms, 30.4 TOPS, 72% |
| Norms, rotary embedding, rotations | 6.4 ms | 25.0 ms |
| Whole block | 132.5 ms, 76% of 100.7 ms | 783.1 ms, 76% of 594.9 ms |

With int4 weights the products reach 31.5 TOPS (75%) at 2048 x 2048 and
31.8 TOPS (76%) at 4096 x 4096: unpacking the 4-bit codes costs time once
a product is compute-bound. A whole block takes 141.5 ms (71% of its floor)
and 816.1 ms (73%). The default format on a 24 GB Mac, int8 attention with
int4 MLP, sits in between.
