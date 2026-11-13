# Images

`hero.jpg`: AI-generated with h3metal on a Mac mini (M6, 24 GB), int8
attention and int4 MLP weights, 3136 x 1344, seed 2, 48 s from a new
prompt; resized to 2000 pixels wide for the README. Prompt:

> A wide cinematic photograph of a dark blacksmith workshop at night. In the
> center, the word "h3metal" in bold lowercase letters, cast from glowing
> molten metal, bright orange-white at its edges and cooling to deep red,
> rests on a black steel anvil. Sparks fly upward and drift across the
> frame. On the left a forge glows with burning coals; on the right hammers
> and tongs hang on a soot-stained brick wall. Thin smoke catches the warm
> light. Shallow depth of field, high detail, dramatic contrast,
> photorealistic.

```sh
h3metal generate -p "<the prompt>" -s 21:9@2k --seed 1 -n 2 -o hero.png   # the second image
```

`same_seed.jpg`: AI-generated, "A red fox sitting in fresh snow at sunrise,
photorealistic, soft golden light.", seed 42, 2048 x 2048, by h3metal (left)
and h3-image-mlx 0.1.0a3 (right) during the benchmark in
docs/benchmarks.md; resized to 640 x 640 each.

`zimage_same_seed.jpg`: AI-generated with Z-Image-Turbo, "A red fox sitting
in fresh snow at sunrise, photorealistic, soft golden light.", seed 42,
1024 x 1024, by diffusers' ZImagePipeline in bfloat16 on the Mac's GPU
(left; tests/ref/zimage_reference.py) and by h3metal with int8 weights
(right; bench/zimage_quality.py); resized to 800 x 800 each.

MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License
Agreement, Copyright © 2026 MiniMax. All Rights Reserved. Z-Image-Turbo is
licensed under the Apache License, Version 2.0 (Tongyi-MAI).
