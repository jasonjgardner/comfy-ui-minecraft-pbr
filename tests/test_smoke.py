"""Smoke test for the pure labPBR math library (no ComfyUI required).

Run with ComfyUI's embedded Python:
    python_embeded\\python.exe comfy-ui-minecraft-pbr\\tests\\test_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from labpbr import (  # noqa: E402
    build_bedrock_normal,
    build_mer,
    build_normal,
    build_specular,
    calculate_porosity,
    calculate_sss,
    derive_ao_and_height,
    emission_to_labpbr,
    extract_emission,
    metalness_to_f0,
    porosity_to_labpbr,
    roughness_to_smoothness,
)

B, H, W = 2, 64, 48  # non-square, batched -> catches shape bugs


def _maps():
    g = torch.Generator().manual_seed(0)
    basecolor = torch.rand(B, 3, H, W, generator=g)
    normal = torch.rand(B, 3, H, W, generator=g) * 0.4 + 0.3
    roughness = torch.rand(B, 1, H, W, generator=g)
    metalness = torch.rand(B, 1, H, W, generator=g)
    return basecolor, normal, roughness, metalness


def _check(name, t, channels):
    assert t.shape == (B, channels, H, W), f"{name}: {tuple(t.shape)} != {(B, channels, H, W)}"
    assert torch.isfinite(t).all(), f"{name}: non-finite values"
    assert t.min() >= 0.0 and t.max() <= 1.0, f"{name}: out of [0,1] -> [{t.min()}, {t.max()}]"
    print(f"  ok {name}: shape {tuple(t.shape)}, range [{t.min():.3f}, {t.max():.3f}]")


def main():
    basecolor, normal, roughness, metalness = _maps()

    print("smoothness / F0:")
    _check("smoothness", roughness_to_smoothness(roughness), 1)
    _check("f0(none)", metalness_to_f0(metalness), 1)
    _check("f0(iron)", metalness_to_f0(metalness, hardcoded_metal="iron"), 1)

    print("derive AO + height:")
    ao, height = derive_ao_and_height(normal)
    _check("ao", ao, 1)
    _check("height", height, 1)
    assert height.min() >= 0.25 - 1e-4, f"height below POM floor: {height.min()}"

    print("porosity / SSS / emission:")
    porosity = calculate_porosity(ao, roughness_to_smoothness(roughness), metalness_to_f0(metalness))
    _check("porosity", porosity, 1)  # raw intensity in [0, 1]
    _check("porosity(_s blue)", porosity_to_labpbr(porosity), 1)
    assert porosity_to_labpbr(porosity).max() <= 64.0 / 255.0 + 1e-4, "encoded porosity exceeds 0-64 band"
    sss = calculate_sss(normal, ao=ao)
    _check("sss", sss, 1)
    emission = extract_emission(basecolor, threshold=0.5)
    _check("emission", emission, 1)

    print("labPBR pack:")
    spec = build_specular(
        roughness, metalness, porosity=porosity, sss=sss,
        emission=emission_to_labpbr(emission), hardcoded_metal="gold",
    )
    _check("specular(_s)", spec, 4)
    _check("normal(_n)", build_normal(normal, ao=ao, height=height), 4)

    # Regression: a raw porosity override (no SSS) must stay in the 0-64 blue band,
    # never leak into the 65-255 SSS band.
    spec_porosity = build_specular(roughness, metalness, porosity=torch.ones(B, 1, H, W))
    assert spec_porosity[:, 2:3].max() <= 64.0 / 255.0 + 1e-4, "porosity override leaked into SSS band"
    print(f"  ok porosity override stays in 0-64 band (max {spec_porosity[:, 2:3].max():.4f})")

    print("Bedrock pack:")
    _check("bedrock normal", build_bedrock_normal(normal), 3)
    _check("mer", build_mer(metalness, roughness, emission=emission), 3)
    _check("mers", build_mer(metalness, roughness, emission=emission, sss=sss), 4)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
