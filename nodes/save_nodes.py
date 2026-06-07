"""Output nodes that write labPBR / Bedrock texture sets as linear RGBA PNGs.

Stock SaveImage drops alpha and is RGB-only; the specular (_s) and normal (_n)
textures need a faithful 4-channel PNG, so these nodes write the files directly
with no gamma applied.
"""

import os

import folder_paths
from comfy_api.latest import io

from ._tensors import image_to_bchw, save_texture

CATEGORY = "Minecraft PBR"


def _write_set(filename_prefix: str, suffixed_images: list[tuple[str, object]]) -> list[dict]:
    """Write ``[(suffix, IMAGE), ...]`` sharing one counter; return UI preview entries.

    The base name (empty suffix) is the albedo/diffuse; others append e.g. ``_s``.
    """
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir
    )
    ui_images: list[dict] = []
    for suffix, image in suffixed_images:
        batch = image_to_bchw(image)
        for index in range(batch.shape[0]):
            name = f"{filename}_{counter + index:05}{suffix}.png"
            save_texture(batch[index], os.path.join(full_output_folder, name))
            if suffix == "":  # preview the albedo only
                ui_images.append({"filename": name, "subfolder": subfolder, "type": "output"})
    return ui_images


class MinecraftSaveLabPBR(io.ComfyNode):
    """Save a labPBR 1.3 set: ``name.png`` (albedo), ``name_s.png``, ``name_n.png``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftSaveLabPBR",
            display_name="Minecraft PBR - Save labPBR",
            category=CATEGORY,
            description="Write albedo, specular (_s) and normal (_n) PNGs for a labPBR resource pack.",
            inputs=[
                io.Image.Input("albedo"),
                io.Image.Input("specular"),
                io.Image.Input("normal"),
                io.String.Input("filename_prefix", default="minecraft_pbr/material"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, albedo, specular, normal, filename_prefix="minecraft_pbr/material") -> io.NodeOutput:
        ui_images = _write_set(
            filename_prefix,
            [("", albedo), ("_s", specular), ("_n", normal)],
        )
        return io.NodeOutput(ui={"images": ui_images})


class MinecraftSaveBedrock(io.ComfyNode):
    """Save a Bedrock RTX set: ``name.png``, ``name_normal.png``, ``name_mer.png`` (``_mers`` if RGBA)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MinecraftSaveBedrock",
            display_name="Minecraft PBR - Save Bedrock RTX",
            category=CATEGORY,
            description="Write albedo, DirectX normal and MER/MERS PNGs for a Bedrock RTX resource pack.",
            inputs=[
                io.Image.Input("albedo"),
                io.Image.Input("normal"),
                io.Image.Input("mer"),
                io.String.Input("filename_prefix", default="minecraft_pbr/material"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, albedo, normal, mer, filename_prefix="minecraft_pbr/material") -> io.NodeOutput:
        mer_suffix = "_mers" if mer.shape[-1] == 4 else "_mer"
        ui_images = _write_set(
            filename_prefix,
            [("", albedo), ("_normal", normal), (mer_suffix, mer)],
        )
        return io.NodeOutput(ui={"images": ui_images})
