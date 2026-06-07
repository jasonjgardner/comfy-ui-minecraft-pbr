"""Derived material channels: porosity, subsurface scattering, emission.

All tensors are batch-first ``(B, C, H, W)`` float in ``[0, 1]``.
"""

import torch
import torch.nn.functional as F

from .normal import _sobel, gaussian_blur


def _normalize(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)
    return torch.where(hi - lo > 0, (x - lo) / (hi - lo + 1e-8), x)


def calculate_porosity(
    ao: torch.Tensor,
    smoothness: torch.Tensor,
    f0: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Porosity intensity in ``[0, 1]`` (deep + rough + non-reflective).

    Returns the raw porosity *intensity*; the labPBR 0-64 blue-band encoding is
    applied later by :func:`porosity_to_labpbr` (so node-supplied overrides and
    computed values pass through the same encoding).
    """
    porosity = (1.0 - ao) * (1.0 - smoothness) * (1.0 - f0)
    if normalize:
        porosity = _normalize(porosity)
    return porosity


def porosity_to_labpbr(porosity: torch.Tensor) -> torch.Tensor:
    """Encode porosity intensity into the specular blue 0-64 band (linear)."""
    return torch.clamp(porosity, 0.0, 1.0) * (64.0 / 255.0)


def calculate_sss(
    normal: torch.Tensor,
    ao: torch.Tensor | None = None,
    curvature_weight: float = 0.7,
    ao_weight: float = 0.3,
    blur_radius: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    """SSS thickness in ``[0, 1]`` from normal curvature plus inverted AO.

    Thin features (edges, tips) scatter more; crevices add a soft glow.
    """
    nx = normal[:, 0:1] * 2.0 - 1.0
    ny = normal[:, 1:2] * 2.0 - 1.0
    nz = normal[:, 2:3] * 2.0 - 1.0
    sx, sy = _sobel(normal.device, normal.dtype)

    def grads(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = F.pad(c, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(p, sx), F.conv2d(p, sy)

    dxx, dxy = grads(nx)
    dyx, dyy = grads(ny)
    dzx, dzy = grads(nz)
    curvature = torch.sqrt(dxx**2 + dxy**2 + dyx**2 + dyy**2 + dzx**2 + dzy**2 + 1e-6)
    if normalize:
        curvature = _normalize(curvature)

    sss = curvature
    if ao is not None and ao_weight > 0:
        sss = curvature * curvature_weight + (1.0 - ao) * ao_weight
    sss = gaussian_blur(sss, blur_radius, sigma_div=3.0)
    if normalize:
        sss = _normalize(sss)
    return torch.clamp(sss, 0.0, 1.0)


def sss_to_labpbr(sss: torch.Tensor) -> torch.Tensor:
    """Encode SSS intensity into the specular blue 65-255 band."""
    return 65.0 / 255.0 + sss * (190.0 / 255.0)


def extract_emission(
    basecolor: torch.Tensor,
    threshold: float = 0.85,
    knee: float = 0.1,
    bloom_radius: int = 0,
    emission_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Emission intensity in ``[0, 1]`` from bright basecolor regions.

    A smoothstep knee avoids a hard luminance cutoff; an optional bloom blur and
    region mask refine the result.
    """
    from .color import luminance

    lum = luminance(basecolor)
    knee_low, knee_high = threshold - knee, threshold + knee
    if knee > 0:
        t = torch.clamp((lum - knee_low) / (knee_high - knee_low + 1e-6), 0.0, 1.0)
        emission = t * t * (3.0 - 2.0 * t)
    else:
        emission = (lum > threshold).to(basecolor.dtype)

    excess = torch.clamp((lum - knee_low) / (1.0 - knee_low + 1e-6), 0.0, 1.0)
    emission = emission * excess

    if bloom_radius > 0:
        emission = torch.maximum(emission, gaussian_blur(emission, bloom_radius, sigma_div=3.0))
    if emission_mask is not None:
        emission = emission * emission_mask
    return torch.clamp(emission, 0.0, 1.0)


def emission_to_labpbr(emission: torch.Tensor) -> torch.Tensor:
    """Encode emission into the specular alpha 0-254 band (255 also = none)."""
    return torch.clamp(emission, 0.0, 1.0) * (254.0 / 255.0)
