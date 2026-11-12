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

`bench/gemm_bench.py`, `bench/attn_bench.py` and `bench/dit_layer_bench.py`
time the transformer's matrix products, its attention and a whole block
with random weights.

TBD
