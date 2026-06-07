# Latent Variation Node — Design Spec

**Date:** 2026-06-07
**Project:** comfy-ui-minecraft-pbr
**Status:** Approved design, pending implementation plan

## Summary

Add a ComfyUI node, `MinecraftLatentVariation` ("Latent Variation"), that takes a
single source latent and emits a **batch of N variant latents**. It serves
randomized blocks, Connected Textures (CTM) variant sets, and weathering states
(clean → cracked → mossy → weathered).

The node is **hybrid**: it works with zero model dependency (cheap, deterministic
latent perturbation) and becomes semantically controllable when a diffusion model
is wired in (prompt-guided img2img).

This is the first piece of "idea D — Variation & material conditioning" for the
package. The package is currently a pure PBR *packer* (no diffusion/latent
involvement); this node introduces the first latent-space workflow.

## Scope

- **In scope:** Variation of a single basecolor latent → batched latent output.
- **Out of scope (deliberate):** Coordinated multi-map PBR-set variation. The core
  is *architected* so the PBR-set version is a clean extension, not a rewrite (see
  Forward Compatibility). Per-variant prompt lists ("named states" in one batch)
  are also deferred — single shared conditioning for now.

## Interface

**Class:** `MinecraftLatentVariation`
**Display name:** "Latent Variation"
**Category:** "Minecraft PBR"
**Return:** `("LATENT",)` — batched `(N, C, H, W)`

### Inputs

**Required**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `samples` | `LATENT` | — | Source latent. If it is itself a batch, variation operates on index 0 (documented). |
| `count` | `INT` | 4 | Number of variants in the output batch. Min 1. |
| `seed` | `INT` | 0 | Base seed. Per-variant seed = `seed + i`; reproducible. |
| `variation_strength` | `FLOAT` | 0.3 | Range 0.0–1.0. |

**Optional (guided path)**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `model` | `MODEL` | — | Presence switches model-free → guided path. |
| `positive` | `CONDITIONING` | — | **Required if `model` is wired.** |
| `negative` | `CONDITIONING` | zeroed | Defaults to a zeroed conditioning derived from `positive`. |
| `steps` | `INT` | 20 | Sampler steps. Ignored in model-free path. |
| `cfg` | `FLOAT` | 7.0 | Ignored in model-free path. |
| `sampler_name` | combo | comfy default | From `comfy.samplers.KSampler.SAMPLERS`. Ignored model-free. |
| `scheduler` | combo | comfy default | From `comfy.samplers.KSampler.SCHEDULERS`. Ignored model-free. |

## Behavior — two paths

### Model-free (no `model` wired)

For each variant `i` in `0..count-1`:

1. `seed_i = seed + i`.
2. Generate a seeded random latent `z_i` matching the source shape.
3. **slerp** from the source toward `z_i` by `variation_strength`
   (`t=0` ≈ original, higher = more divergence).

Stack the `count` results into a batched latent `(N, C, H, W)`.

No sampler, no VAE, fully deterministic.

**Documented tradeoff:** without a model we cannot denoise, so pushing
`variation_strength` high yields noisier output on decode. Low strength stays close
to the source. This is the honest cost of the zero-dependency path.

### Guided (`model` wired)

For each variant `i`:

1. `seed_i = seed + i`.
2. img2img: renoise the source and run the sampler at `denoise = variation_strength`
   with `positive` / `negative` conditioning, `cfg`, `sampler_name`, `scheduler`,
   `steps`.

Output stays clean because it renoises + denoises. The same conditioning is applied
across the whole batch; variants differ by seed. (To get a *mossy* set vs a
*cracked* set, run the node twice with different prompts.)

## Architecture

Respects the existing package split: **pure-torch core in `labpbr/`, thin nodes in
`nodes/`.**

- **`labpbr/variation.py`** — pure-torch core, no ComfyUI imports:
  - `slerp_latents(source, noise, t)` → spherical interpolation between two latents.
  - `make_variation_batch(source, seed, count, strength, noise=None)` → builds the
    batched model-free variation set.
- **`nodes/variation_nodes.py`** — thin node wrapper. The guided path's sampler
  call (needs `comfy.sample` / ComfyUI runtime) lives here, **not** in the core.
- **Registration** — add to the package's `NODE_CLASS_MAPPINGS` and
  `NODE_DISPLAY_NAME_MAPPINGS`.

## Forward compatibility (PBR-set version)

`make_variation_batch` accepts an **optional pre-generated `noise` tensor**. The
future coordinated-PBR node will generate noise **once** and pass the *same* noise
to each map's call (basecolor / normal / roughness / metalness), so the maps
perturb in a correlated way (a mossy patch shifts color *and* roughness *and*
normal together). This optional `noise` parameter is the entire seam required for
idea-D scope #2 — no rewrite of the core.

## Error handling

- `model` wired but no `positive` conditioning → clear `ValueError`.
- `count < 1` → `ValueError`.
- `samples` missing the expected latent key / wrong shape → validation error.

## Testing

- **Core (`labpbr/variation.py`)** is pure torch → unit-testable headless:
  - slerp endpoints: `t=0` returns source, `t=1` returns the noise latent.
  - determinism: same `seed` → identical batch.
  - output shape is `(count, C, H, W)`.
  - passing an explicit `noise` tensor reproduces results (forward-compat seam).
- **Guided path** needs a model → integration/smoke level, or manual ComfyUI
  verification. Not part of the headless unit suite.

## Decisions log

1. Sub-thread: latent **variation** node (not the material-prompt front-end).
2. Mechanism: **hybrid** — model-free default + opt-in guided.
3. Scope: **basecolor now, architected for the PBR set later**.
4. Output: **batch of N** in one run, single `variation_strength` (no gradient ramp).
5. I/O: **LATENT in → LATENT out** (lean on stock VAE Encode/Decode).
6. Conditioning: **single shared** positive/negative across the batch.
