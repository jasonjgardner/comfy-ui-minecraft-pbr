"""ComfyUI V3 nodes for labPBR / Bedrock RTX texture packing."""

from .bedrock_nodes import MinecraftBedrockPack
from .derive_nodes import MinecraftDeriveAOHeight, MinecraftExtractEmission
from .labpbr_nodes import MinecraftLabPBRPack
from .metal_nodes import MinecraftMetalMask
from .save_nodes import MinecraftSaveBedrock, MinecraftSaveLabPBR

NODE_CLASS_MAPPINGS = {
    "MinecraftLabPBRPack": MinecraftLabPBRPack,
    "MinecraftBedrockPack": MinecraftBedrockPack,
    "MinecraftMetalMask": MinecraftMetalMask,
    "MinecraftSaveLabPBR": MinecraftSaveLabPBR,
    "MinecraftSaveBedrock": MinecraftSaveBedrock,
    "MinecraftDeriveAOHeight": MinecraftDeriveAOHeight,
    "MinecraftExtractEmission": MinecraftExtractEmission,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
