"""Per-pixel metal-id packing for multi-metal labPBR surfaces.

Paint a separate region mask for each labPBR hardcoded metal (e.g. a copper face
with a chrome border) and composite them into one metal-id map to feed the
``metal_mask`` input of the labPBR packer.
"""

from comfy_api.latest import io

from ..labpbr.metal_mask import METAL_MASK_PRESETS, build_metal_id_map
from ..labpbr.metals import LABPBR_METALS
from ._tensors import bchw_to_image, mask_to_bchw, to_device

CATEGORY = "Minecraft PBR"


class MinecraftMetalMask(io.ComfyNode):
    """Composite per-metal region masks into a labPBR metal-id map (metal_mask)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        metal_inputs = [
            io.MultiType.Input(
                name,
                types=[io.Image, io.Mask],
                optional=True,
                tooltip=f"Region that is {name} (labPBR specular-green id {LABPBR_METALS[name]}).",
            )
            for name in METAL_MASK_PRESETS
        ]
        return io.Schema(
            node_id="MinecraftMetalMask",
            display_name="Minecraft PBR - Pack Metal Masks",
            category=CATEGORY,
            description="Combine per-metal region masks into one metal-id map. Wire the IMAGE "
            "output into the labPBR packer's 'metal_mask' input; painted pixels override the "
            "F0/metal green channel with that metal's exact id. On overlap, later metals win. "
            "Output is an IMAGE (id replicated to RGB) so the discrete F0 ids survive intact.",
            inputs=[
                *metal_inputs,
                io.Float.Input(
                    "threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Mask value above which a pixel is assigned that metal id.",
                ),
            ],
            outputs=[io.Image.Output(display_name="metal_mask")],
        )

    @classmethod
    def execute(cls, threshold=0.5, **metal_masks) -> io.NodeOutput:
        # Align every supplied mask to the first one's device before compositing.
        coerced = {name: mask_to_bchw(metal_masks.get(name)) for name in METAL_MASK_PRESETS}
        device = next((t.device for t in coerced.values() if t is not None), None)
        coerced = {name: to_device(t, device) for name, t in coerced.items()}

        result = build_metal_id_map(coerced, threshold=threshold)
        if result is None:
            raise ValueError("Pack Metal Masks: connect at least one metal region mask.")
        # Replicate the id across RGB; an equal-channel gray IMAGE round-trips the exact
        # metal byte through the packer's luminance collapse (coefficients sum to 1.0).
        return io.NodeOutput(bchw_to_image(result.repeat(1, 3, 1, 1)))
