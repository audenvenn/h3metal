# Images

`pdmd_steps.jpg`: AI-generated with h3metal on a Mac mini (M6, 24 GB), int8 attention and int4 MLP weights,
2048 x 2048, seed 42; each picture resized to 554 x 554. Columns: the six-step model (`dit.h3m`), then the
few-step transformer (`dit-pdmd.h3m`, docs/few-step.md) at 4 and at 2 steps. Prompts, by row:

> A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light.

> A cozy corner bookshop on a rainy evening, warm light spilling from the windows onto the wet cobblestones, a
> hand-painted wooden sign above the door reading "THE QUIET PAGE", a bicycle leaning against the lamppost,
> photorealistic.

> A flat vector travel poster of a lighthouse on a cliff above a turquoise sea, seagulls, bold geometric shapes,
> the words "VISIT CAPE ARIA" in large white letters at the top.

```sh
h3metal generate -p "<the prompt>" --seed 42 [--steps 4 | --steps 2] -o out.png
```

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

`lora_outpaint.jpg`: AI-generated, both halves.

- **Left, the input:** h3metal's FLUX.2 [klein] 4B at 1024 x 1024, seed 2,
  from the prompt "A paper lantern festival over a river at night,
  photorealistic." It was shrunk to 576 x 576 and centered on a pure green
  1024 x 1024 canvas.
- **Right, the edit:** the same model with fal's outpainting adapter
  ([fal/flux-2-klein-4B-outpaint-lora](https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora),
  Apache-2.0), seed 42, 13 s on the Mac mini. The prompt is the adapter's,
  "Fill the green spaces according to the image".
- Both halves were resized to 800 x 800 for the page.

```sh
h3metal generate --model flux2 -p "Fill the green spaces according to the image" -i framed.png \
    --lora flux-outpaint-lora.safetensors --seed 42 -o outpainted.png
```
