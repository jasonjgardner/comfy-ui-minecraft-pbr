"""Assemble Bedrock normal and MER/MERS textures.

Bedrock stores raw roughness and metalness (no labPBR encoding). MER is RGB
(R=metalness, G=emissive, B=roughness); when SSS is supplied the texture becomes
RGBA (MERS). The SSS alpha is read only by Vibrant Visuals -- ray-traced (RTX)
mode ignores it. Metalness and SSS are mutually exclusive per pixel (only
non-metals scatter), so SSS is zeroed above ``metal_threshold``.
"""

import torch

from .normal import to_directx


def build_bedrock_normal(
    normal: torch.Tensor,
    flip_y: bool = True,
    swap_xy: bool = False,
) -> torch.Tensor:
    """Bedrock DirectX normal RGB ``(B, 3, H, W)`` (no AO/height packing)."""
    return to_directx(normal, flip_y=flip_y, swap_xy=swap_xy)[:, 0:3]


def build_mer(
    metalness: torch.Tensor,
    roughness: torch.Tensor,
    emission: torch.Tensor | None = None,
    sss: torch.Tensor | None = None,
    metal_threshold: float = 0.5,
) -> torch.Tensor:
    """MER (R=metal, G=emissive, B=roughness) or MERS (+A=SSS) linear texture."""
    b, _, h, w = roughness.shape
    device, dtype = roughness.device, roughness.dtype

    if emission is None:
        emission = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
    channels = [metalness, emission, roughness]

    if sss is not None:
        is_metal = (metalness > metal_threshold).to(dtype)
        channels.append(sss * (1.0 - is_metal))

    return torch.cat(channels, dim=1)
