"""Pure-torch labPBR / Bedrock RTX packing math (no ComfyUI dependencies).

Every public function operates on batch-first ``(B, C, H, W)`` float tensors in
``[0, 1]`` and stores all data channels linearly (no gamma).
"""

from .color import luminance, metalness_to_f0, roughness_to_smoothness
from .derive import (
    calculate_porosity,
    calculate_sss,
    emission_to_labpbr,
    extract_emission,
    porosity_to_labpbr,
    sss_to_labpbr,
)
from .metal_mask import METAL_MASK_PRESETS, build_metal_id_map
from .metals import LABPBR_METALS, METAL_NAMES, metal_f0_value
from .normal import (
    compute_divergence,
    derive_ao_and_height,
    frankot_chellappa,
    normal_to_ao,
    normal_to_height,
    to_directx,
)
from .pack_bedrock import build_bedrock_normal, build_mer
from .pack_labpbr import build_normal, build_specular

__all__ = [
    "luminance",
    "metalness_to_f0",
    "roughness_to_smoothness",
    "calculate_porosity",
    "calculate_sss",
    "emission_to_labpbr",
    "extract_emission",
    "porosity_to_labpbr",
    "sss_to_labpbr",
    "LABPBR_METALS",
    "METAL_NAMES",
    "metal_f0_value",
    "METAL_MASK_PRESETS",
    "build_metal_id_map",
    "compute_divergence",
    "derive_ao_and_height",
    "frankot_chellappa",
    "normal_to_ao",
    "normal_to_height",
    "to_directx",
    "build_bedrock_normal",
    "build_mer",
    "build_normal",
    "build_specular",
]
