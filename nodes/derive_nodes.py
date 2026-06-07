"""Standalone derivation nodes: AO + height from a normal map, emission from albedo.

These expose the same math the packer can run internally, as separate nodes so a
graph can preview/edit AO, height, or emission before packing.
"""

from comfy_api.latest import io

from ..labpbr.derive import extract_emission
from ..labpbr.normal import derive_ao_and_height
from ._tensors import image_to_bchw, mask_to_bchw, tensor_to_mask, to_device

CATEGORY = "Minecraft PBR"


class MinecraftDeriveAOHeight(io.ComfyNode):
    """Derive ambient occlusion and a POM height map from a normal map."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftDeriveAOHeight",
            display_name="Minecraft PBR - Derive AO + Height",
            category=CATEGORY,
            description="Approximate AO (from normal divergence) and a height map "
            "(Frankot-Chellappa integration) from a normal map.",
            inputs=[
                io.Image.Input("normal", tooltip="Tangent-space normal map (OpenGL Y-up)."),
                io.Boolean.Input("seamless", default=False, tooltip="Treat the input as tileable."),
                io.Float.Input("ao_strength", default=2.0, min=0.0, max=10.0, step=0.1),
                io.Int.Input("ao_blur", default=5, min=0, max=64),
                io.Float.Input("height_intensity", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input(
                    "height_min", default=0.25, min=0.0, max=1.0, step=0.05,
                    tooltip="Deepest POM value; 0.25 = 25% block depth for Minecraft.",
                ),
                io.Boolean.Input("height_invert", default=True, tooltip="Flip POM direction so bumps rise."),
                io.Mask.Input("height_mask", optional=True, tooltip="Where 1, flatten height to height_min."),
            ],
            outputs=[
                io.Mask.Output(display_name="ao"),
                io.Mask.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        normal,
        seamless=False,
        ao_strength=2.0,
        ao_blur=5,
        height_intensity=1.0,
        height_min=0.25,
        height_invert=True,
        height_mask=None,
    ) -> io.NodeOutput:
        norm = image_to_bchw(normal)
        ao, height = derive_ao_and_height(
            norm,
            seamless=seamless,
            ao_strength=ao_strength,
            ao_blur=ao_blur,
            height_intensity=height_intensity,
            height_min=height_min,
            height_invert=height_invert,
            height_mask=to_device(mask_to_bchw(height_mask), norm.device),
        )
        return io.NodeOutput(tensor_to_mask(ao), tensor_to_mask(height))


class MinecraftExtractEmission(io.ComfyNode):
    """Extract an emission mask from bright regions of a basecolor map."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftExtractEmission",
            display_name="Minecraft PBR - Extract Emission",
            category=CATEGORY,
            description="Luminance-threshold the basecolor (with a soft knee) into an emission mask.",
            inputs=[
                io.Image.Input("basecolor"),
                io.Float.Input("threshold", default=0.85, min=0.0, max=1.0, step=0.01),
                io.Float.Input("knee", default=0.1, min=0.0, max=1.0, step=0.01),
                io.Int.Input("bloom_radius", default=0, min=0, max=64),
                io.Mask.Input("emission_mask", optional=True, tooltip="Restrict emission to this region."),
            ],
            outputs=[io.Mask.Output(display_name="emission")],
        )

    @classmethod
    def execute(cls, basecolor, threshold=0.85, knee=0.1, bloom_radius=0, emission_mask=None) -> io.NodeOutput:
        base = image_to_bchw(basecolor)
        emission = extract_emission(
            base,
            threshold=threshold,
            knee=knee,
            bloom_radius=bloom_radius,
            emission_mask=to_device(mask_to_bchw(emission_mask), base.device),
        )
        return io.NodeOutput(tensor_to_mask(emission))
