"""LabPBR hardcoded metal table (specular green-channel encoding).

Reference: https://shaderlabs.org/wiki/LabPBR_Material_Standard

The green channel of the specular (_s) texture stores reflectance F0 linearly:
- 0-229   : dielectric F0 (we emit ~10 -> F0 ~= 0.04, typical insulator)
- 230-254 : predefined metals with shader-known F0 spectra
- 255     : "custom" metal, shader uses the albedo as F0
"""

# name -> green-channel byte value (None means "use the auto dielectric/custom rule")
LABPBR_METALS: dict[str, int | None] = {
    "none": None,
    "custom": 255,
    "iron": 230,
    "gold": 231,
    "aluminum": 232,
    "chrome": 233,
    "copper": 234,
    "lead": 235,
    "platinum": 236,
    "silver": 237,
}

# Ordered list for combo widgets.
METAL_NAMES: list[str] = list(LABPBR_METALS.keys())


def metal_f0_value(name: str) -> float:
    """Return the linear [0,1] F0 byte for a metal name's metallic regions.

    "none" and unknown names fall back to custom-metal behaviour (255 -> 1.0).
    """
    value = LABPBR_METALS.get(name)
    if value is None:
        return 1.0
    return value / 255.0
