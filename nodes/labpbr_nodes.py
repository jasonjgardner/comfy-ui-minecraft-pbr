"""labPBR 1.3 packer node (Java Edition / Iris / OptiFine)."""

import torch
from comfy_api.latest import io

from ..labpbr.color import metalness_to_f0, roughness_to_smoothness
from ..labpbr.derive import calculate_porosity, calculate_sss, emission_to_labpbr, extract_emission
from ..labpbr.metals import METAL_NAMES
from ..labpbr.normal import derive_ao_and_height
from ..labpbr.pack_labpbr import build_normal, build_specular
from ._tensors import bchw_to_image, image_to_bchw, mask_to_bchw, scalar_to_bchw, to_device

CATEGORY = "Minecraft PBR"


class MinecraftLabPBRPack(io.ComfyNode):
    """Pack Chord-style PBR maps into labPBR 1.3 albedo, specular (_s), normal (_n)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftLabPBRPack",
            display_name="Minecraft PBR - Pack labPBR",
            category=CATEGORY,
            description="Assemble albedo, specular (_s: smoothness/F0/porosity-SSS/emission) and "
            "normal (_n: XY/AO/height) RGBA textures for labPBR 1.3 shaders.",
            inputs=[
                io.Image.Input("basecolor"),
                io.Image.Input("normal", tooltip="Tangent-space normal (OpenGL Y-up)."),
                io.MultiType.Input("roughness", types=[io.Image, io.Mask]),
                io.MultiType.Input("metalness", types=[io.Image, io.Mask]),
                io.Combo.Input("hardcoded_metal", options=METAL_NAMES, default="none",
                               tooltip="Metal id for metallic pixels; 'none' uses custom (255)."),
                io.Boolean.Input("flip_normal_y", default=True,
                                 tooltip="labPBR _n stores DirectX Y-down; flip an OpenGL Y-up normal "
                                 "(default). Disable only if your input is already DirectX."),
                io.Boolean.Input("swap_normal_xy", default=False),
                io.Boolean.Input("derive_ao_height", default=True,
                                 tooltip="Derive AO + height from the normal when not supplied."),
                io.Boolean.Input("include_height", default=True, tooltip="Write derived height into _n alpha (POM)."),
                io.Boolean.Input("seamless", default=False),
                io.Boolean.Input("compute_porosity", default=False),
                io.Boolean.Input("compute_sss", default=False),
                io.Boolean.Input("compute_emission", default=False),
                io.Float.Input("emission_threshold", default=0.85, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("emission_knee", default=0.1, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("ao_strength", default=2.0, min=0.0, max=10.0, step=0.1, advanced=True),
                io.Int.Input("ao_blur", default=5, min=0, max=64, advanced=True),
                io.Float.Input("height_intensity", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True),
                io.Boolean.Input("height_invert", default=True, advanced=True),
                io.MultiType.Input("ao", types=[io.Image, io.Mask], optional=True,
                                   tooltip="Override derived AO."),
                io.MultiType.Input("height", types=[io.Image, io.Mask], optional=True,
                                   tooltip="Override derived height (e.g. Chord 'Normal to Height' IMAGE)."),
                io.MultiType.Input("emission", types=[io.Image, io.Mask], optional=True,
                                   tooltip="Override extracted emission."),
                io.MultiType.Input("porosity", types=[io.Image, io.Mask], optional=True),
                io.MultiType.Input("sss", types=[io.Image, io.Mask], optional=True),
                io.MultiType.Input("metal_mask", types=[io.Image, io.Mask], optional=True,
                                   tooltip="Per-pixel metal-id map (from Pack Metal Masks); "
                                   "overrides hardcoded_metal/metalness on painted pixels."),
                io.Mask.Input("height_mask", optional=True, tooltip="Where 1, flatten POM height."),
            ],
            outputs=[
                io.Image.Output(display_name="albedo"),
                io.Image.Output(display_name="specular"),
                io.Image.Output(display_name="normal"),
            ],
        )

    @classmethod
    def execute(
        cls,
        basecolor,
        normal,
        roughness,
        metalness,
        hardcoded_metal="none",
        flip_normal_y=True,
        swap_normal_xy=False,
        derive_ao_height=True,
        include_height=True,
        seamless=False,
        compute_porosity=False,
        compute_sss=False,
        compute_emission=False,
        emission_threshold=0.85,
        emission_knee=0.1,
        ao_strength=2.0,
        ao_blur=5,
        height_intensity=1.0,
        height_invert=True,
        ao=None,
        height=None,
        emission=None,
        porosity=None,
        sss=None,
        metal_mask=None,
        height_mask=None,
    ) -> io.NodeOutput:
        # ComfyUI may pass inputs from CPU and GPU nodes together; align to one device.
        base = image_to_bchw(basecolor)
        device = base.device
        norm = image_to_bchw(normal).to(device)
        rough = scalar_to_bchw(roughness).to(device)
        metal = scalar_to_bchw(metalness).to(device)

        ao_t = to_device(mask_to_bchw(ao), device)
        height_t = to_device(mask_to_bchw(height), device)
        emission_t = to_device(mask_to_bchw(emission), device)
        porosity_t = to_device(mask_to_bchw(porosity), device)
        sss_t = to_device(mask_to_bchw(sss), device)
        metal_mask_t = to_device(mask_to_bchw(metal_mask), device)

        if derive_ao_height and (ao_t is None or height_t is None):
            derived_ao, derived_height = derive_ao_and_height(
                norm,
                seamless=seamless,
                ao_strength=ao_strength,
                ao_blur=ao_blur,
                height_intensity=height_intensity,
                height_min=0.25,
                height_invert=height_invert,
                height_mask=to_device(mask_to_bchw(height_mask), device),
            )
            ao_t = ao_t if ao_t is not None else derived_ao
            if height_t is None and include_height:
                height_t = derived_height

        if compute_porosity and porosity_t is None:
            smoothness = roughness_to_smoothness(rough)
            f0 = metalness_to_f0(metal)
            ao_for_porosity = ao_t if ao_t is not None else torch.ones_like(rough)
            porosity_t = calculate_porosity(ao_for_porosity, smoothness, f0)

        if compute_sss and sss_t is None:
            sss_t = calculate_sss(norm, ao=ao_t)

        if compute_emission and emission_t is None:
            emission_t = extract_emission(base, threshold=emission_threshold, knee=emission_knee)

        emission_encoded = emission_to_labpbr(emission_t) if emission_t is not None else None

        specular = build_specular(
            rough, metal, porosity=porosity_t, sss=sss_t, emission=emission_encoded,
            hardcoded_metal=hardcoded_metal, metal_mask=metal_mask_t,
        )
        normal_tex = build_normal(
            norm, ao=ao_t, height=height_t, flip_y=flip_normal_y, swap_xy=swap_normal_xy,
        )

        return io.NodeOutput(
            bchw_to_image(base),
            bchw_to_image(specular),
            bchw_to_image(normal_tex),
        )
