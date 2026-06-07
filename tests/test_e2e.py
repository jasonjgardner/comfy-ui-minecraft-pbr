"""End-to-end test through node execute() with ComfyUI-shaped tensors.

Feeds BHWC IMAGE inputs and Chord-style 1-channel IMAGE roughness/metalness, then
checks output IMAGE shapes and that the save node writes RGBA PNGs.
"""

import importlib.util
import os
import sys

from PIL import Image

COMFY_ROOT = r"M:\AI\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
PKG_DIR = os.path.join(COMFY_ROOT, "custom_nodes", "comfy-ui-minecraft-pbr")

os.chdir(COMFY_ROOT)
sys.path.insert(0, COMFY_ROOT)

import folder_paths  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("comfy-ui-minecraft-pbr", os.path.join(PKG_DIR, "__init__.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["comfy-ui-minecraft-pbr"] = module
spec.loader.exec_module(module)
M = module.NODE_CLASS_MAPPINGS

B, H, W = 1, 40, 56
g = torch.Generator().manual_seed(1)
basecolor = torch.rand(B, H, W, 3, generator=g)        # IMAGE BHWC
normal = torch.rand(B, H, W, 3, generator=g) * 0.4 + 0.3
rough_img = torch.rand(B, H, W, 1, generator=g)        # Chord-style 1-channel IMAGE
metal_mask = torch.rand(B, H, W, generator=g)          # MASK (proves IMAGE-or-MASK)

print("Pack labPBR (all features on, mixed IMAGE/MASK scalars):")
out = M["MinecraftLabPBRPack"].execute(
    basecolor=basecolor, normal=normal, roughness=rough_img, metalness=metal_mask,
    hardcoded_metal="gold", compute_porosity=True, compute_sss=True, compute_emission=True,
)
albedo, specular, normal_n = out.args
assert albedo.shape == (B, H, W, 3), albedo.shape
assert specular.shape == (B, H, W, 4), specular.shape
assert normal_n.shape == (B, H, W, 4), normal_n.shape
print(f"  albedo {tuple(albedo.shape)}  _s {tuple(specular.shape)}  _n {tuple(normal_n.shape)}")

print("Pack Bedrock (SSS -> MERS):")
bout = M["MinecraftBedrockPack"].execute(
    basecolor=basecolor, normal=normal, roughness=rough_img, metalness=metal_mask, compute_sss=True,
)
b_albedo, b_normal, mer = bout.args
assert b_normal.shape == (B, H, W, 3), b_normal.shape
assert mer.shape == (B, H, W, 4), mer.shape  # MERS
print(f"  _normal {tuple(b_normal.shape)}  mer/mers {tuple(mer.shape)}")

print("Save labPBR:")
res = M["MinecraftSaveLabPBR"].execute(
    albedo=albedo, specular=specular, normal=normal_n, filename_prefix="mcpbr_e2e_test/mat",
)
out_dir = folder_paths.get_output_directory()
base = res.ui["images"][0]["filename"][:-len(".png")]  # strip extension; albedo has no suffix
sub = res.ui["images"][0]["subfolder"]
paths = {
    "albedo": os.path.join(out_dir, sub, base + ".png"),
    "_s": os.path.join(out_dir, sub, base + "_s.png"),
    "_n": os.path.join(out_dir, sub, base + "_n.png"),
}
for label, p in paths.items():
    assert os.path.exists(p), f"missing {label}: {p}"
    im = Image.open(p)
    expected = "RGB" if label == "albedo" else "RGBA"
    assert im.mode == expected, f"{label}: mode {im.mode} != {expected}"
    assert im.size == (W, H), f"{label}: size {im.size}"
    print(f"  wrote {label}: {im.mode} {im.size}  ({p})")
    im.close()

# cleanup
for p in paths.values():
    os.remove(p)
try:
    os.rmdir(os.path.join(out_dir, sub))
except OSError:
    pass

print("\nEND-TO-END TEST PASSED")
