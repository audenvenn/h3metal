# LoRA adapters

h3metal applies LoRA adapters to Z-Image-Turbo and FLUX.2 [klein] 4B: style
adapters such as the many trained for Z-Image-Turbo, and FLUX.2 [klein]'s edit
adapters (fal's object removal, outpainting, zoom and others).

```sh
h3metal generate --model zimage -p "Pixel art style. A red fox in fresh snow" --lora pixel_art.safetensors
h3metal generate --model flux2 -p "Fill the green spaces according to the image" -i framed.png \
    --lora flux-outpaint-lora.safetensors:0.9
```

`--lora FILE[:STRENGTH]` takes a strength (default 1). Repeat it for several
adapters. `h3metal serve` offers the `.safetensors` files in
`~/.h3metal/loras`. From Python, pass `loras=[(path, strength)]` to the
pipeline's `generate`.

`h3metal download --lora OWNER/NAME` fetches an adapter's file from Hugging
Face (or ModelScope with `--source modelscope`) into `~/.h3metal/loras`.
Name the file as `OWNER/NAME/FILE` when the repository holds several. A
bare name is then enough for `--lora`:

```sh
h3metal download --lora AI-ModelScope/pixel_art_style_lora_z_image_turbo --source modelscope
h3metal generate --model zimage -p "Pixel art style. A lighthouse at dusk" --lora pixel_art_style_z_image_turbo
```

## What it reads

- **Layouts:** diffusers/PEFT (`lora_A`, `lora_B`, optional `alpha`), kohya
  (`lora_down`, `lora_up`, `alpha`, module names with underscores), and the
  original checkpoints' names. The last covers FLUX.2's `double_blocks`,
  `single_blocks`, `img_in`, ... and Z-Image's fused `attention.qkv`. The
  prefixes ai-toolkit, ComfyUI and PEFT add are removed (`diffusion_model.`,
  `base_model.model.`, `transformer.`). As in diffusers, a layer is scaled by
  alpha / rank when the file gives an alpha, by 1 otherwise.
- **The final layer's modulation:** FLUX.2's original final layer orders its
  modulation (shift, scale), and diffusers' `norm_out` orders it (scale,
  shift). An adapter of `final_layer.adaLN_modulation.1` therefore has its
  output rows swapped, as diffusers' checkpoint converter swaps the weights.
  diffusers' LoRA loader (0.43) maps that layer without the swap. On a mini
  model it then lands 15% from the adapter merged in float32.
  `tests/test_lora.py` checks h3metal's mapping (the `fused_final` layout).
- **Layers:**
  - Z-Image-Turbo: every linear layer of its blocks (attention, MLP and the
    AdaLN modulation, in the main and refiner blocks).
  - FLUX.2 [klein]: every linear layer, including the embedders, the
    timestep MLP, the modulations and the output layer.
- **Refused:** a file h3metal cannot apply exactly is refused, with a list of
  what it cannot apply. It is never applied in part. That covers:
  - DoRA, LoKr and LoHa;
  - weight differences;
  - text-encoder adapters;
  - layers h3metal does not keep (Z-Image's patch embedder and final layer,
    its timestep MLP);
  - a file whose shapes belong to another model.

## How

An adapter changes a layer `W` to `W + s B A`, with `A` of `r` rows and `B`
of `r` columns, where `s` is alpha / rank times the strength. h3metal applies
it as exactly that, a low-rank term. It does not merge it into the stored
weights, for two reasons:

- A merge rounds the int8 weights a second time, and the int8 rounding is
  most of h3metal's error budget.
- A merge has to be redone for every strength and every combination of
  adapters.

**On the GPU,** the term runs inside the layer's own product
(`kernels/gemm.metal`, `LORA`):

1. The adapter's input side `T = A x` is a narrow int8 product of the same
   activations the layer reads, `r` outputs wide.
2. The layer's product then adds `T B^T` to its result before storing it, as
   a second, short product on the matrix units with the same tile shape. Its
   destination registers line up with the main product's (checked on every
   tile shape in use by `tests/test_gemm.py`), so the term costs no extra
   pass over the layer's output.

Several adapters, or the query, key and value adapters of one fused layer,
sit side by side in the rank dimension. Each column tile reads only the rank
range whose `B` rows touch its columns: a query adapter costs nothing in the
key and value columns.

**On the CPU,** the layers it computes take the term in float. These are
FLUX.2's timestep MLP and modulations, and Z-Image's modulation, which
h3metal precomputes per step.

- Z-Image's modulation tables fold the norm weights and the input rotations'
  signs into the AdaLN output. To recompute them under an adapter, the
  converter keeps that output and the norm weights alongside (16 MB).
- A conversion made before h3metal 0.4 lacks them. An adapter with
  modulation layers then asks you to run `h3metal convert` again; adapters
  without them work on it as is.

**Transforms.** A converted layer is the original one transformed:

- rows permuted for the rotary embedding;
- signs flipped;
- products fused or split;
- inputs rotated.

Each model describes those transforms once, as plans (`model_plans` in
`h3metal/zimage.py` and `h3metal/flux2.py`). The converter builds the
weights from them and the adapters go through them. A test without a GPU
(`tests/test_convert.py`) checks, for every layout and both models, that an
adapter's terms equal the plans' transform of its dense update `s B A`.

## Fidelity

**Mini models** with random weights and a random adapter on every supported
layer (`tests/test_lora.py`, reference `tests/ref/mini_lora.py`). The
reference merges the adapter into the float32 weights. Errors are the worst
step's relative L2 error.

| | int8, no adapter | with the adapter, strength 1 | strength 0.5 | the adapter's effect |
|---|---:|---:|---:|---:|
| Z-Image | 0.88% | 0.92% | 0.88% | 24% |
| FLUX.2 [klein] | 2.7% | 3.0% | 2.8% | 104% |

- Every layout gives these numbers. Kohya's and diffusers' layouts give the
  same outputs bit for bit.
- Removing the adapters gives back the outputs without them, bit for bit.

**Real weights** against diffusers' `load_lora_weights`, on the Mac's GPU in
bfloat16:

**Z-Image-Turbo with a pixel-art adapter** (rank 32, attention, MLP and
modulation of every block). The test case:

- prompt "Pixel art style. A red fox sitting in fresh snow at sunrise, soft
  golden light.";
- 1024 x 1024, seed 42, the same noise on both sides.

| | whole image | first step |
|---|---:|---:|
| without the adapter | 23.2 dB PSNR, SSIM 0.936 | 5.7% |
| with it | 24.5 dB PSNR, SSIM 0.925 | 3.4% |

The adapter changes the picture completely (pixel art), and h3metal follows
diffusers with it as closely as without it.

**FLUX.2 [klein] with fal's outpainting adapter** (rank 16, every layer),
as an edit:

- the input is a 576 x 576 picture centered on a green 1024 x 1024 canvas;
- the prompt is the adapter's, "Fill the green spaces according to the
  image";
- h3metal is fed the reference's text embeddings, which separates the
  transformer, the adapter and the decoder from the text encoder
  (docs/flux2.md).

| | whole image | first step |
|---|---:|---:|
| without the adapter | 43.5 dB PSNR, SSIM 0.998 | 2.7% |
| with it | 30.8 dB PSNR, SSIM 0.971 | 4.6% |

Without the adapter, klein leaves most of the green border in place.
Copying a picture is easy to match, and with the adapter it paints the whole
border instead. The first step's error still grows, from 2.7% to 4.6%. A
float16 run of the first step was the yardstick (float32 does not fit this
Mac's GPU, and float16 has 8 times bfloat16's precision):

| against float16 | without the adapter | with it |
|---|---:|---:|
| diffusers in bfloat16 | 1.43% | 2.28% |
| h3metal (int8) | 2.59% | 4.00% |

Both grow by the same factor, about 1.6. The adapter makes this model more
sensitive to rounding, whatever does the rounding. h3metal's error stays at
the same ratio to bfloat16's with the adapter as without.

- The adapter moves the first step by 78% of its size in both
  implementations.
- h3metal's move agrees with diffusers' to a cosine of 0.997. The rest is
  the two runs' rounding errors, which are independent.

The edited image:

<p align="center"><img src="images/lora_outpaint.jpg" alt="Left: a night photo of a river lined with paper lanterns, small, in the middle of a bright green square. Right: the same scene extended to fill the whole square: more lanterns, the river bank with plants in the foreground, the night sky above" width="100%"/></p>
<p align="center"><em>AI-generated. Left: the input. Right: h3metal's FLUX.2 [klein] 4B with fal's outpainting adapter, 13 s on a Mac mini M6.</em></p>

## Speed

Denoising time of a 1024 x 1024 image with and without the adapter, on the
same request, warm, median of three interleaved runs (`bench/lora_speed.py`,
Mac mini M6):

| | adapter | without | with | cost |
|---|---|---:|---:|---:|
| Z-Image-Turbo, 8 steps | pixel art, rank 32, every block layer and the modulation | 16.99 s | 17.35 s | +2.1% |
| FLUX.2 [klein] 4B, 4 steps (an edit) | fal's outpainting, rank 16, every layer | 4.99 s | 5.14 s | +3.1% |

Putting an adapter on the transformer takes 0.2-0.5 s, once per set of
adapters and strengths. That covers reading the file, rotating and
quantizing its input sides, and recomputing Z-Image's tables.
