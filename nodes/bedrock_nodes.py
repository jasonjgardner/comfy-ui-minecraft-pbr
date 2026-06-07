"""Bedrock RTX packer node (Minecraft Bedrock Edition ray tracing)."""

from comfy_api.latest import io

from ..labpbr.derive import calculate_sss, extract_emission
from ..labpbr.pack_bedrock import build_bedrock_normal, build_mer
from ._tensors import bchw_to_image, image_to_bchw, mask_to_bchw, scalar_to_bchw, to_device

CATEGORY = "Minecraft PBR"


class MinecraftBedrockPack(io.ComfyNode):
    """Pack PBR maps into Bedrock RTX albedo, DirectX normal, and MER/MERS textures."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftBedrockPack",
            display_name="Minecraft PBR - Pack Bedrock RTX",
            category=CATEGORY,
            description="Assemble albedo, DirectX normal (_normal) and MER "
            "(R=metal, G=emissive, B=roughness) textures for Minecraft Bedrock. "
            "Enabling SSS adds an A channel (MERS) read only by Vibrant Visuals; "
            "ray-traced (RTX) mode ignores the SSS channel.",
            inputs=[
                io.Image.Input("basecolor"),
                io.Image.Input("normal", tooltip="Tangent-space normal (OpenGL Y-up)."),
                io.MultiType.Input("roughness", types=[io.Image, io.Mask]),
                io.MultiType.Input("metalness", types=[io.Image, io.Mask]),
                io.Boolean.Input("flip_normal_y", default=True,
                                 tooltip="Bedrock expects DirectX Y-down; flip an OpenGL normal."),
                io.Boolean.Input("swap_normal_xy", default=False),
                io.Boolean.Input("compute_sss", default=False,
                                 tooltip="Derive SSS -> MERS (RGBA). Read only by Vibrant Visuals; "
                                 "ignored by ray-traced (RTX) mode, and never applied to metal pixels."),
                io.Boolean.Input("compute_emission", default=False),
                io.Float.Input("emission_threshold", default=0.85, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("emission_knee", default=0.1, min=0.0, max=1.0, step=0.01, advanced=True),
                io.MultiType.Input("emission", types=[io.Image, io.Mask], optional=True),
                io.MultiType.Input("sss", types=[io.Image, io.Mask], optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="albedo"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="mer"),
            ],
        )

    @classmethod
    def execute(
        cls,
        basecolor,
        normal,
        roughness,
        metalness,
        flip_normal_y=True,
        swap_normal_xy=False,
        compute_sss=False,
        compute_emission=False,
        emission_threshold=0.85,
        emission_knee=0.1,
        emission=None,
        sss=None,
    ) -> io.NodeOutput:
        # ComfyUI may pass inputs from CPU and GPU nodes together; align to one device.
        base = image_to_bchw(basecolor)
        device = base.device
        norm = image_to_bchw(normal).to(device)
        rough = scalar_to_bchw(roughness).to(device)
        metal = scalar_to_bchw(metalness).to(device)

        emission_t = to_device(mask_to_bchw(emission), device)
        sss_t = to_device(mask_to_bchw(sss), device)

        if compute_sss and sss_t is None:
            sss_t = calculate_sss(norm, ao=None)
        if compute_emission and emission_t is None:
            emission_t = extract_emission(base, threshold=emission_threshold, knee=emission_knee)

        normal_tex = build_bedrock_normal(norm, flip_y=flip_normal_y, swap_xy=swap_normal_xy)
        mer = build_mer(metal, rough, emission=emission_t, sss=sss_t)

        return io.NodeOutput(
            bchw_to_image(base),
            bchw_to_image(normal_tex),
            bchw_to_image(mer),
        )
