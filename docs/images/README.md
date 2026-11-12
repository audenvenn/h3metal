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

MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License
Agreement, Copyright © 2026 MiniMax. All Rights Reserved.
