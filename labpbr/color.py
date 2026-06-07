"""Scalar channel conversions shared by labPBR and Bedrock packing.

All tensors are batch-first ``(B, C, H, W)`` float in ``[0, 1]``. LabPBR data
channels (smoothness, F0, porosity, SSS, emission, AO, height) are stored
*linearly* -- no gamma is applied anywhere in this package.
"""

import torch

from .metals import metal_f0_value

# Typical dielectric reflectance: F0 ~= 0.04 -> byte 10.
_DIELECTRIC_F0 = 10.0 / 255.0
# Rec. 709 luma coefficients.
_LUMA = (0.2126, 0.7152, 0.0722)


def roughness_to_smoothness(roughness: torch.Tensor) -> torch.Tensor:
    """Perceptual smoothness from linear roughness.

    LabPBR defines ``roughness = (1 - perceptualSmoothness)^2``; the inverse is
    ``perceptualSmoothness = 1 - sqrt(roughness)``.
    """
    return 1.0 - torch.sqrt(torch.clamp(roughness, 0.0, 1.0))


def metalness_to_f0(
    metalness: torch.Tensor,
    threshold: float = 0.5,
    hardcoded_metal: str = "none",
    metal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map metalness to the linear F0 / metal-id green channel.

    Pixels above ``threshold`` become metal (the ``hardcoded_metal`` byte, or 255
    custom when "none"); others become the dielectric byte. An optional
    ``metal_mask`` of normalised metal ids overrides per pixel where > 0.
    """
    metal_f0 = metal_f0_value(hardcoded_metal)
    is_metal = (metalness > threshold).to(metalness.dtype)
    f0 = torch.lerp(
        torch.full_like(metalness, _DIELECTRIC_F0),
        torch.full_like(metalness, metal_f0),
        is_metal,
    )

    if metal_mask is None:
        return f0

    mask = metal_mask
    if mask.shape != f0.shape and mask.dim() < f0.dim():
        mask = mask.unsqueeze(0)
    return torch.where(mask > 0.0, mask, f0)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance of an RGB tensor -> ``(B, 1, H, W)``."""
    r, g, b = _LUMA
    return r * rgb[:, 0:1] + g * rgb[:, 1:2] + b * rgb[:, 2:3]
