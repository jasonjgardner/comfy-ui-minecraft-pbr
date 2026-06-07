"""Composite per-metal region masks into one labPBR metal-id map.

Each labPBR hardcoded metal occupies a specific specular green byte (230-237, or
255 for custom). This builds a normalised id map (``byte / 255``) where painted
regions carry the exact metal id and unpainted pixels stay 0 -- i.e. "no
override", so ``metalness_to_f0`` falls back to its metalness/dielectric rule
there. Values are discrete (thresholded, never blended): interpolating between
two metal ids would produce a meaningless third id.
"""

import torch

from .metals import LABPBR_METALS

# Real metals only (drop "none"); list order is the overlap priority -- later
# entries paint over earlier ones, so specific named metals win over "custom".
METAL_MASK_PRESETS: list[str] = [name for name, value in LABPBR_METALS.items() if value is not None]


def build_metal_id_map(
    masks: dict[str, torch.Tensor | None],
    threshold: float = 0.5,
) -> torch.Tensor | None:
    """Combine ``{preset_name: (B, 1, H, W) | None}`` masks into a metal-id map.

    Returns ``(B, 1, H, W)`` in ``[0, 1]`` (metal byte / 255), or ``None`` when no
    mask is supplied. Pixels where a mask exceeds ``threshold`` take that metal's
    id; later presets override earlier ones on overlap.
    """
    present = [(name, masks[name]) for name in METAL_MASK_PRESETS if masks.get(name) is not None]
    if not present:
        return None

    result = torch.zeros_like(present[0][1])
    for name, mask in present:
        value = LABPBR_METALS[name] / 255.0
        result = torch.where(mask > threshold, torch.full_like(result, value), result)
    return result
