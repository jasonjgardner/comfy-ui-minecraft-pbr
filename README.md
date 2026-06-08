# ComfyUI Minecraft PBR

Custom nodes that pack PBR material maps into **labPBR 1.3** (Java Edition — Iris/OptiFine)
and **Bedrock RTX** resource-pack textures, entirely on the GPU.

These are *packer* nodes — they don't run any model. They're built to consume the
IMAGE outputs of the [ComfyUI-Chord](https://github.com/ubisoft/ComfyUI-Chord)
material-estimation node (`basecolor`, `normal`, `roughness`, `metalness`), but
accept those maps from any source.

## Nodes (category **Minecraft PBR**)

| Node | Purpose |
|------|---------|
| **Pack labPBR** | albedo + specular `_s` (smoothness/F0/porosity·SSS/emission) + normal `_n` (XY/AO/height) |
| **Pack Bedrock RTX** | albedo + DirectX `_normal` + `_mer` (R=metal, G=emissive, B=roughness); `_mers` when SSS is enabled |
| **Pack Metal Masks** | one region MASK per labPBR metal (iron/gold/aluminum/chrome/copper/lead/platinum/silver/custom) → a per-pixel metal-id map for the packer's `metal_mask` |
| **Save labPBR** | writes `name.png`, `name_s.png`, `name_n.png` (linear RGBA) |
| **Save Bedrock RTX** | writes `name.png`, `name_normal.png`, `name_mer.png` / `_mers.png` |
| **Derive AO + Height** | AO (normal divergence) + POM height (Frankot–Chellappa) → two MASKs |
| **Extract Emission** | luminance-threshold a basecolor → emission MASK |
| **Latent Variation** | one source LATENT → a batch of N variant LATENTs for randomized blocks, CTM variant sets, and weathering states (clean → cracked → mossy → weathered) |

`roughness` and `metalness` inputs accept **IMAGE or MASK**, so Chord's 1-channel
IMAGE outputs wire straight in. The `_s` and `_n` textures flow as 4-channel RGBA
IMAGE tensors; use the Save nodes to write valid alpha-bearing PNGs (the stock
SaveImage node drops alpha).

## Channel reference

**labPBR `_s`** (all linear): R = perceptual smoothness `1 − √roughness` · G = F0/metal id
(dielectric ≈ 10, metals 230–237, custom 255) · B = porosity (0–64) or SSS (65–255),
SSS wins per-pixel · A = emission (0/255 = none, 1–254 = level).

**labPBR `_n`**: R/G = normal X/Y in **DirectX (Y-down)** convention · B = ambient
occlusion · A = POM height (0.25 = 25% block depth … 1.0 = surface). The packer
flips an OpenGL Y-up input to DirectX by default (`flip_normal_y`).

**Bedrock `_mer`/`_mers`** (all linear): R = metalness · G = emissive · B = roughness ·
A (MERS) = SSS thickness (zeroed where metallic). The SSS alpha (MERS) is read only by
**Vibrant Visuals**; ray-traced (RTX) mode ignores it and uses plain `_mer`.

## Typical graph

```
ComfyUI-Chord ─ basecolor ┐
              ─ normal    ┤→ Pack labPBR → albedo/specular/normal → Save labPBR
              ─ roughness ┤
              ─ metalness ┘
```

The packer derives AO + height from the normal automatically; toggle
`compute_porosity` / `compute_sss` / `compute_emission` and pick a `hardcoded_metal`
as needed. Wire the Derive / Extract nodes in only when you want to preview or edit
those channels before packing.

### Multiple metals in one texture

For a surface mixing metal types (e.g. a copper face with a chrome border), paint a
region MASK per metal, feed them into **Pack Metal Masks**, and wire its `metal_mask`
output into **Pack labPBR**'s `metal_mask` input. Painted pixels get that metal's exact
labPBR id in the specular green channel (iron 230 … silver 237, custom 255), overriding
both `hardcoded_metal` and the metalness threshold; unpainted pixels fall back to normal
behaviour. Overlapping regions resolve last-wins (named metals over `custom`).

## Latent variations

Every node above works on IMAGE/MASK tensors; **Latent Variation** is the package's
one LATENT node. It takes a single source LATENT and emits a batched LATENT of `count`
variants for randomized blocks, Connected Textures (CTM) variant sets, and weathering
states (clean → cracked → mossy → weathered).

- **Model-free (default, no `model` wired):** seeded slerp perturbation of the source
  latent. `variation_strength` 0 returns the source; higher values diverge more (and
  decode noisier, since there is no denoise step).
- **Guided (`model` + `positive` wired):** an img2img re-denoise through ComfyUI's
  sampler, so output stays clean and `variation_strength` becomes the denoise amount.
  The same conditioning is applied across the batch; variants differ by `seed`. Run it
  twice with different prompts for, e.g., a mossy set vs a cracked set.

Inputs: `samples` (LATENT), `count`, `seed`, `variation_strength`, plus optional
`model` / `positive` / `negative` / `steps` / `cfg` / `sampler_name` / `scheduler`
for the guided path. Wire a `VAE Decode` after it to feed the batched output into the
Pack labPBR / Pack Bedrock RTX nodes.

## Install

Clone or symlink this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
No extra Python dependencies. Reference: <https://shaderlabs.org/wiki/LabPBR_Material_Standard>.
