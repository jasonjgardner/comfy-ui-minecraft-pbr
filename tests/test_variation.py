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

    # strength 0 is a pass-through: every variant equals the source
    passthrough = make_variation_batch(source, seed=42, count=3, strength=0.0)
    assert torch.equal(passthrough[0], source) and torch.equal(passthrough[2], source), \
        "strength=0 must return the source unchanged"
    print("  ok strength=0 pass-through")

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
