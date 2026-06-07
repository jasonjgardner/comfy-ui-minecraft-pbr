# Latent Variation Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `MinecraftLatentVariation` ComfyUI node that turns one source latent into a batch of N variant latents — model-free seeded slerp by default, prompt-guided img2img when a diffusion model is wired.

**Architecture:** Pure-torch variation math lives in `labpbr/variation.py` (no ComfyUI imports, unit-testable headless). The thin node in `nodes/variation_nodes.py` wraps it for the model-free path and calls ComfyUI's `common_ksampler` for the guided path. The core's `make_variation_batch` takes an optional pre-generated `noise` tensor — the single seam that lets a future coordinated-PBR node drive correlated variation across multiple maps without a rewrite.

**Tech Stack:** Python ≥3.10, PyTorch (provided by ComfyUI runtime), ComfyUI V3 node API (`comfy_api.latest.io`). Tests are standalone scripts (the repo convention — see `tests/test_smoke.py`), not pytest.

**Spec:** [docs/superpowers/specs/2026-06-07-latent-variation-node-design.md](../specs/2026-06-07-latent-variation-node-design.md)

---

## Environment Notes (read once)

- **Embedded Python (has torch):**
  `M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe`
- The repo is symlinked into ComfyUI at
  `…\ComfyUI\custom_nodes\comfy-ui-minecraft-pbr` → `m:\AI\comfy-ui-minecraft-pbr`,
  so `tests/test_load.py` validates the live working tree.
- `tests/test_smoke.py` and the new `tests/test_variation.py` are **pure-torch** —
  they insert the repo root on `sys.path` and import `labpbr` directly, so they run
  with any torch-enabled Python.
- `tests/test_load.py` needs the **ComfyUI runtime** — run it with the embedded
  Python from the ComfyUI root (see its docstring).
- Working branch is already `feat/latent-variations`.

## File Structure

- **Create** `labpbr/variation.py` — pure-torch core: `slerp_latents`, `make_variation_batch`, `_seeded_noise`.
- **Create** `tests/test_variation.py` — standalone smoke test for the core.
- **Modify** `labpbr/__init__.py` — export `slerp_latents`, `make_variation_batch`.
- **Create** `nodes/variation_nodes.py` — `MinecraftLatentVariation` V3 node.
- **Modify** `nodes/__init__.py` — import + register the node.
- **Modify** `tests/test_load.py:27` — bump node-count assertion `7` → `8`.
- **Modify** `README.md` — document the node (node table + a usage note).

---

## Task 1: Pure-torch variation core

**Files:**
- Create: `labpbr/variation.py`
- Test: `tests/test_variation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_variation.py`:

```python
"""Smoke test for the latent variation core (pure torch, no ComfyUI).

Run with ComfyUI's embedded Python (or any torch-enabled Python):
    python_embeded\\python.exe comfy-ui-minecraft-pbr\\tests\\test_variation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from labpbr.variation import make_variation_batch, slerp_latents  # noqa: E402

C, H, W = 4, 16, 16


def main():
    source = torch.randn(C, H, W, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(C, H, W, generator=torch.Generator().manual_seed(2))

    # endpoints
    assert torch.allclose(slerp_latents(source, noise, 0.0), source), "t=0 must return source"
    assert torch.allclose(slerp_latents(source, noise, 1.0), noise), "t=1 must return noise"
    print("  ok slerp endpoints")

    # batch shape
    batch = make_variation_batch(source, seed=42, count=4, strength=0.3)
    assert batch.shape == (4, C, H, W), f"shape {tuple(batch.shape)} != (4, {C}, {H}, {W})"
    print(f"  ok batch shape {tuple(batch.shape)}")

    # determinism
    batch2 = make_variation_batch(source, seed=42, count=4, strength=0.3)
    assert torch.equal(batch, batch2), "same seed must be deterministic"
    print("  ok deterministic")

    # variants actually differ
    assert not torch.equal(batch[0], batch[1]), "variants must differ from each other"
    print("  ok variants differ")

    # forward-compat seam: explicit noise overrides seeding
    pre = torch.randn(4, C, H, W, generator=torch.Generator().manual_seed(7))
    n1 = make_variation_batch(source, seed=0, count=4, strength=0.3, noise=pre)
    n2 = make_variation_batch(source, seed=999, count=4, strength=0.3, noise=pre)
    assert torch.equal(n1, n2), "explicit noise must make the result independent of seed"
    print("  ok explicit-noise seam")

    # batched source uses index 0
    b_from_batched = make_variation_batch(source.unsqueeze(0), seed=42, count=4, strength=0.3)
    assert torch.equal(b_from_batched, batch), "batched source should vary index 0"
    print("  ok batched-source uses index 0")

    print("\nALL VARIATION TESTS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
& "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" tests\test_variation.py
```
Expected: FAIL — `ModuleNotFoundError: No module named 'labpbr.variation'`.

- [ ] **Step 3: Write minimal implementation**

Create `labpbr/variation.py`:

```python
"""Latent variation: build a batch of variant latents from one source latent.

Pure-torch, no ComfyUI dependencies. This module is the *model-free* path: it
interpolates the source latent toward seeded random latents via spherical
interpolation (slerp). The guided (diffusion) path lives in the node layer
because it needs the ComfyUI sampler.

``make_variation_batch`` accepts an optional pre-generated ``noise`` tensor. That
is the seam a future coordinated multi-map (PBR-set) variation node uses: it
generates noise once and passes the *same* noise to each map so basecolor /
normal / roughness / metalness perturb in a correlated way.
"""

import torch


def slerp_latents(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latents.

    ``a`` and ``b`` share a shape (typically ``(C, H, W)``). ``t`` in ``[0, 1]``:
    ``t<=0`` returns ``a``, ``t>=1`` returns ``b``. Falls back to linear
    interpolation when the two latents are nearly collinear, which keeps the
    interpolation numerically stable (``sin(theta) -> 0``).
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b

    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = a_flat / (a_flat.norm() + 1e-12)
    b_norm = b_flat / (b_flat.norm() + 1e-12)
    dot = (a_norm * b_norm).sum().clamp(-1.0, 1.0)

    if dot.abs() > 0.9995:
        return torch.lerp(a, b, t)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    wa = torch.sin((1.0 - t) * theta) / sin_theta
    wb = torch.sin(t * theta) / sin_theta
    return wa * a + wb * b


def _seeded_noise(source: torch.Tensor, seed: int, count: int) -> torch.Tensor:
    """``(count, *source.shape)`` Gaussian noise; variant ``i`` is seeded ``seed + i``."""
    gen = torch.Generator(device="cpu")
    layers = []
    for i in range(count):
        gen.manual_seed(int(seed) + i)
        layers.append(torch.randn(source.shape, generator=gen, dtype=source.dtype))
    return torch.stack(layers, dim=0).to(source.device)


def make_variation_batch(
    source: torch.Tensor,
    seed: int,
    count: int,
    strength: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build ``count`` variant latents from a single source latent.

    ``source`` is ``(C, H, W)`` or ``(B, C, H, W)`` (batched -> index 0 is used).
    Returns a stacked ``(count, C, H, W)`` tensor where variant ``i`` is
    ``slerp(source, noise_i, strength)``.

    ``noise``: optional pre-generated ``(count, C, H, W)`` tensor used verbatim
    instead of seeding fresh noise -- the seam that lets a coordinated multi-map
    variation node share one noise set across maps.
    """
    if source.dim() == 4:
        source = source[0]
    if noise is None:
        noise = _seeded_noise(source, seed, count)
    variants = [slerp_latents(source, noise[i], strength) for i in range(count)]
    return torch.stack(variants, dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
& "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" tests\test_variation.py
```
Expected: PASS — ends with `ALL VARIATION TESTS PASSED`.

- [ ] **Step 5: Commit**

```
git add labpbr/variation.py tests/test_variation.py
git commit -m "feat: add pure-torch latent variation core (slerp batch)"
```

---

## Task 2: Export the core from the labpbr package

**Files:**
- Modify: `labpbr/__init__.py`

- [ ] **Step 1: Add the import block**

In `labpbr/__init__.py`, after the `from .pack_labpbr import ...` line (currently line 27), add:

```python
from .variation import make_variation_batch, slerp_latents
```

- [ ] **Step 2: Add to `__all__`**

In the `__all__` list in `labpbr/__init__.py`, add these two entries (anywhere before the closing `]`, e.g. after `"build_specular",`):

```python
    "make_variation_batch",
    "slerp_latents",
```

- [ ] **Step 3: Verify the package import resolves the new symbols**

Run:
```
& "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" -c "import sys; sys.path.insert(0, '.'); import labpbr; print(labpbr.make_variation_batch, labpbr.slerp_latents)"
```
Expected: prints two `<function …>` reprs, no ImportError.

- [ ] **Step 4: Commit**

```
git add labpbr/__init__.py
git commit -m "feat: export latent variation core from labpbr package"
```

---

## Task 3: The ComfyUI node + registration

**Files:**
- Create: `nodes/variation_nodes.py`
- Modify: `nodes/__init__.py`
- Modify: `tests/test_load.py:27`

- [ ] **Step 1: Update the load test to expect 8 nodes (failing first)**

In `tests/test_load.py`, change line 27 from:

```python
assert len(mappings) == 7, f"expected 7 nodes, got {len(mappings)}"
```

to:

```python
assert len(mappings) == 8, f"expected 8 nodes, got {len(mappings)}"
```

- [ ] **Step 2: Run the load test to verify it fails**

Run (from the ComfyUI root, per the test docstring):
```
& "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-ui-minecraft-pbr\tests\test_load.py"
```
Expected: FAIL — `AssertionError: expected 8 nodes, got 7` (node not registered yet).

- [ ] **Step 3: Create the node**

Create `nodes/variation_nodes.py`:

```python
"""Latent variation node: turn one source latent into a batch of N variants.

Model-free (no ``model`` wired): seeded slerp perturbation via the pure-torch
``labpbr.variation`` core. Guided (``model`` wired): img2img re-denoise through
ComfyUI's sampler at ``denoise = variation_strength``. Output is a batched LATENT,
one VAE Decode away from the Pack nodes.
"""

import comfy.samplers
import torch
from comfy_api.latest import io

from ..labpbr.variation import make_variation_batch

CATEGORY = "Minecraft PBR"


def _zero_conditioning(conditioning):
    """Zeroed copy of a CONDITIONING (mirrors ComfyUI's ConditioningZeroOut).

    Used as the default ``negative`` in the guided path when the caller leaves it
    unwired, so img2img has a valid (empty) negative prompt.
    """
    out = []
    for t in conditioning:
        d = t[1].copy()
        pooled = d.get("pooled_output", None)
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t[0]), d])
    return out


class MinecraftLatentVariation(io.ComfyNode):
    """Emit a batch of variant latents from a single source latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftLatentVariation",
            display_name="Minecraft PBR - Latent Variation",
            category=CATEGORY,
            description="Make N variants of a latent (randomized blocks / weathering). "
            "Model-free seeded slerp by default; prompt-guided img2img when a model is wired.",
            inputs=[
                io.Latent.Input("samples", tooltip="Source latent; if batched, index 0 is varied."),
                io.Int.Input("count", default=4, min=1, max=64, tooltip="Number of variants to emit."),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True, tooltip="Base seed; variant i uses seed + i.",
                ),
                io.Float.Input(
                    "variation_strength", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="0 = source, 1 = full divergence. Guided path maps this to denoise.",
                ),
                io.Model.Input("model", optional=True, tooltip="Wire a model to switch to prompt-guided img2img."),
                io.Conditioning.Input("positive", optional=True, tooltip="Required when a model is wired."),
                io.Conditioning.Input("negative", optional=True, tooltip="Defaults to zeroed conditioning."),
                io.Int.Input("steps", default=20, min=1, max=10000, tooltip="Sampler steps (guided path only)."),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, tooltip="CFG scale (guided path only)."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, tooltip="Sampler (guided path only)."),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, tooltip="Scheduler (guided path only)."),
            ],
            outputs=[io.Latent.Output(display_name="variants")],
        )

    @classmethod
    def execute(
        cls,
        samples,
        count,
        seed,
        variation_strength,
        model=None,
        positive=None,
        negative=None,
        steps=20,
        cfg=7.0,
        sampler_name=None,
        scheduler=None,
    ) -> io.NodeOutput:
        if count < 1:
            raise ValueError("count must be >= 1")

        source = samples["samples"]

        if model is None:
            batch = make_variation_batch(source, seed=seed, count=count, strength=variation_strength)
            return io.NodeOutput({"samples": batch})

        if positive is None:
            raise ValueError("positive conditioning is required when a model is wired")

        from nodes import common_ksampler  # lazy: avoid import cycles at node-load time

        tiled = source[:1].repeat(count, 1, 1, 1)
        neg = negative if negative is not None else _zero_conditioning(positive)
        out = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, neg, {"samples": tiled}, denoise=variation_strength,
        )[0]
        return io.NodeOutput(out)
```

- [ ] **Step 4: Register the node**

In `nodes/__init__.py`, add the import alongside the others (after the `from .save_nodes import ...` line):

```python
from .variation_nodes import MinecraftLatentVariation
```

and add this entry to the `NODE_CLASS_MAPPINGS` dict (e.g. after the `MinecraftExtractEmission` line):

```python
    "MinecraftLatentVariation": MinecraftLatentVariation,
```

- [ ] **Step 5: Run the load test to verify it passes**

Run:
```
& "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-ui-minecraft-pbr\tests\test_load.py"
```
Expected: PASS — lists 8 nodes including
`MinecraftLatentVariation   <Minecraft PBR - Latent Variation>  inputs=11 outputs=1`,
ending with `NODE PACKAGE LOADS CLEANLY`.

- [ ] **Step 6: Commit**

```
git add nodes/variation_nodes.py nodes/__init__.py tests/test_load.py
git commit -m "feat: add MinecraftLatentVariation node (model-free slerp + guided img2img)"
```

---

## Task 4: Document the node in the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Inspect the existing node table**

Run:
```
& "M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe" -c "print(open('README.md', encoding='utf-8').read())"
```
Find the node listing/table (the seven existing nodes) and the section that describes the IMAGE-centric flow.

- [ ] **Step 2: Add a row/entry for the new node**

Add an entry matching the existing table's column format. Use this content (adapt the markdown columns to the table already present):

```
**Latent Variation** — Turn one source latent into a batch of N variants for
randomized blocks, CTM variant sets, and weathering states. Model-free by default
(seeded slerp perturbation; `variation_strength` 0 = source, higher = more
divergence, and noisier on decode). Wire a `model` + `positive` conditioning to
switch to prompt-guided img2img (clean output; `variation_strength` becomes the
denoise amount). LATENT in -> batched LATENT out; add a `VAE Decode` to feed the
Pack nodes.
```

- [ ] **Step 3: Commit**

```
git add README.md
git commit -m "docs: document the Latent Variation node in the README"
```

---

## Manual Verification (guided path — cannot be unit-tested headless)

The guided path needs a real diffusion model, so verify it by hand in ComfyUI:

1. Restart ComfyUI (or reload custom nodes). Confirm **Minecraft PBR - Latent Variation** appears under the "Minecraft PBR" category.
2. **Model-free:** `Empty Latent Image` → `Latent Variation` (count=4, strength=0.3, no model) → `VAE Decode` → `Preview Image`. Expect 4 visibly different images; raising `variation_strength` increases divergence/noise.
3. **Determinism:** run twice with the same `seed` → identical previews.
4. **Guided:** `Load Checkpoint` (model + a CLIP-encoded `positive`) wired into the node; a `VAE Encode` of a real texture into `samples`; strength ≈ 0.4 → 4 prompt-flavored variants that stay structurally close to the source and are clean (not noisy).
5. **Error path:** wire `model` but leave `positive` unwired → execution raises a clear "positive conditioning is required when a model is wired" error.

---

## Self-Review (completed during planning)

- **Spec coverage:** hybrid model-free/guided (Tasks 1, 3) ✓; basecolor-only scope with forward-compat `noise` seam (Task 1 `make_variation_batch`) ✓; batch of N + single `variation_strength` (Task 1/3) ✓; LATENT in → LATENT out (Task 3 schema) ✓; single shared conditioning (Task 3 execute) ✓; pure-core/thin-node split (Tasks 1 vs 3) ✓; error handling for missing `positive` and `count < 1` (Task 3) ✓; tests for endpoints/determinism/shape/seam (Task 1) ✓.
- **Placeholder scan:** none — every code/command step is complete.
- **Type consistency:** `make_variation_batch(source, seed, count, strength, noise=None)` and `slerp_latents(a, b, t)` signatures match between definition (Task 1), export (Task 2), and call site (Task 3). Latent value shape (`dict["samples"]`, `(B,C,H,W)`) is consistent with the verified `io.Latent` contract and `common_ksampler`.
