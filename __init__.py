"""comfy-ui-minecraft-pbr: pack PBR maps into labPBR 1.3 / Bedrock RTX textures.

Designed to consume the IMAGE outputs of the ComfyUI-Chord material-estimation
node (basecolor, normal, roughness, metalness), but works with any matching maps.
"""

from .nodes import NODE_CLASS_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS"]
