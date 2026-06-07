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
