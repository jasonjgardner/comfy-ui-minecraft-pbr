"""Assemble labPBR 1.3 specular (_s) and normal (_n) textures.

Both outputs are RGBA ``(B, 4, H, W)``. A missing emission channel is encoded as
alpha 255 (labPBR's "no emission" sentinel; 0-254 are emission levels), and a
missing height as 1.0 ("surface"), so the packed textures stay labPBR-valid even
when those features are disabled.
"""

import torch

from .color import metalness_to_f0, roughness_to_smoothness
from .derive import porosity_to_labpbr, sss_to_labpbr
from .normal import to_directx


def build_specular(
    roughness: torch.Tensor,
    metalness: torch.Tensor,
    porosity: torch.Tensor | None = None,
    sss: torch.Tensor | None = None,
    emission: torch.Tensor | None = None,
    hardcoded_metal: str = "none",
    metal_mask: torch.Tensor | None = None,
    sss_threshold: float = 0.01,
) -> torch.Tensor:
    """labPBR specular RGBA: R=smoothness, G=F0/metal, B=porosity|SSS, A=emission.

    ``porosity`` and ``sss`` are raw intensities in ``[0, 1]``; the labPBR blue-band
    encoding (porosity 0-64, SSS 65-255) is applied here so they cannot collide.
    Where both are present, per-pixel SSS wins above ``sss_threshold``.
    """
    b, _, h, w = roughness.shape
    device, dtype = roughness.device, roughness.dtype

    smoothness = roughness_to_smoothness(roughness)
    f0 = metalness_to_f0(metalness, hardcoded_metal=hardcoded_metal, metal_mask=metal_mask)

    if sss is not None and porosity is not None:
        blue = torch.where(sss > sss_threshold, sss_to_labpbr(sss), porosity_to_labpbr(porosity))
    elif sss is not None:
        blue = sss_to_labpbr(sss)
    elif porosity is not None:
        blue = porosity_to_labpbr(porosity)
    else:
        blue = torch.zeros(b, 1, h, w, device=device, dtype=dtype)

    if emission is None:
        # labPBR: alpha 255 is the "no emission" sentinel (0-254 are emission levels).
        emission = torch.ones(b, 1, h, w, device=device, dtype=dtype)

    return torch.cat([smoothness, f0, blue, emission], dim=1)


def build_normal(
    normal: torch.Tensor,
    ao: torch.Tensor | None = None,
    height: torch.Tensor | None = None,
    flip_y: bool = False,
    swap_xy: bool = False,
) -> torch.Tensor:
    """labPBR normal RGBA: R=X, G=Y, B=ambient occlusion, A=height/displacement."""
    b, _, h, w = normal.shape
    device, dtype = normal.device, normal.dtype

    oriented = to_directx(normal, flip_y=flip_y, swap_xy=swap_xy)
    if ao is None:
        ao = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    if height is None:
        height = torch.ones(b, 1, h, w, device=device, dtype=dtype)

    return torch.cat([oriented[:, 0:1], oriented[:, 1:2], ao, height], dim=1)
