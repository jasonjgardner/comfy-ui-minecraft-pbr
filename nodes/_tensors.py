"""Adapters between ComfyUI tensor conventions and the labPBR math library.

ComfyUI IMAGE tensors are ``(B, H, W, C)`` float ``[0, 1]``; MASK tensors are
``(B, H, W)``. The ``labpbr`` package works in batch-first ``(B, C, H, W)``. These
helpers convert in both directions and write linear (no-gamma) PNGs.
"""

import os

import numpy as np
import torch
from PIL import Image

from ..labpbr.color import luminance


def image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE ``(B, H, W, C)`` -> ``(B, C, H, W)``."""
    return image.permute(0, 3, 1, 2).contiguous()


def bchw_to_image(tensor: torch.Tensor) -> torch.Tensor:
    """Batch-first ``(B, C, H, W)`` -> ComfyUI IMAGE ``(B, H, W, C)``."""
    return tensor.permute(0, 2, 3, 1).contiguous()


def scalar_to_bchw(value: torch.Tensor) -> torch.Tensor:
    """Coerce an IMAGE or MASK scalar map to a single-channel ``(B, 1, H, W)``.

    Accepts a 3-D MASK ``(B, H, W)``, a multi-channel IMAGE (collapsed to
    luminance), or an already single-channel IMAGE.
    """
    if value.dim() == 3:  # MASK (B, H, W)
        return value.unsqueeze(1)
    bchw = image_to_bchw(value)  # IMAGE (B, H, W, C) -> (B, C, H, W)
    if bchw.shape[1] == 1:
        return bchw
    return luminance(bchw)


def mask_to_bchw(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Optional MASK ``(B, H, W)`` (or single-channel IMAGE) -> ``(B, 1, H, W)``."""
    if mask is None:
        return None
    return scalar_to_bchw(mask)


def to_device(tensor: torch.Tensor | None, device) -> torch.Tensor | None:
    """Move an optional tensor onto ``device`` (ComfyUI may hand inputs from CPU
    and GPU nodes in the same call -- e.g. GPU images with CPU masks)."""
    if tensor is None:
        return None
    return tensor.to(device)


def tensor_to_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Single-channel ``(B, 1, H, W)`` -> ComfyUI MASK ``(B, H, W)``."""
    return tensor.squeeze(1).contiguous()


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` float -> ``(H, W, C)`` uint8 with linear scaling."""
    clamped = torch.clamp(tensor, 0.0, 1.0)
    return (clamped.permute(1, 2, 0).cpu().float().numpy() * 255.0 + 0.5).astype(np.uint8)


def _pil_mode(channels: int) -> str:
    return {1: "L", 3: "RGB", 4: "RGBA"}.get(channels, "RGB")


def save_texture(tensor: torch.Tensor, path: str) -> None:
    """Write one batch-first ``(1, C, H, W)`` or ``(C, H, W)`` texture as a PNG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = _to_uint8(tensor)
    image = Image.fromarray(array, mode=_pil_mode(array.shape[2]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
