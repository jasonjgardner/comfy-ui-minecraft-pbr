"""Normal-map operations: DirectX conversion, AO and height derivation.

All tensors are batch-first ``(B, C, H, W)`` float in ``[0, 1]`` unless noted.
Height is reconstructed with the Frankot-Chellappa FFT integration; AO is
approximated from the divergence of the tangent-space normal field.
"""

import torch
import torch.nn.functional as F


def _sobel(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
    return sx.view(1, 1, 3, 3) / 8.0, sy.view(1, 1, 3, 3) / 8.0


def gaussian_blur(x: torch.Tensor, radius: int, sigma_div: float = 2.0) -> torch.Tensor:
    """Separable reflect-padded Gaussian blur on a ``(B, 1, H, W)`` tensor."""
    if radius <= 0:
        return x
    size = radius * 2 + 1
    sigma = radius / sigma_div
    coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius
    k = torch.exp(-(coords**2) / (2 * sigma * sigma))
    k = k / k.sum()
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def to_directx(normal: torch.Tensor, flip_y: bool = True, swap_xy: bool = False) -> torch.Tensor:
    """Apply optional X/Y swap and Y flip (OpenGL Y-up -> DirectX Y-down)."""
    out = normal.clone()
    if swap_xy:
        out[:, 0], out[:, 1] = normal[:, 1], normal[:, 0]
    if flip_y:
        out[:, 1] = 1.0 - out[:, 1]
    return out


def _gradient(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tangent gradients in ``[-1, 1]`` from the packed XY channels."""
    nx = (normal[:, 0:1] - 0.5) * 2.0
    ny = (normal[:, 1:2] - 0.5) * 2.0
    return nx, ny


def _make_seamless(grad_x: torch.Tensor, grad_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror gradients into a 2x2 tile so FFT integration has no edge seam."""
    gx_h, gx_v, gx_hv = (torch.flip(grad_x, d) for d in ([3], [2], [2, 3]))
    gy_h, gy_v, gy_hv = (torch.flip(grad_y, d) for d in ([3], [2], [2, 3]))
    gx = torch.cat(
        [torch.cat([grad_x, -gx_h], 3), torch.cat([gx_v, -gx_hv], 3)], 2
    )
    gy = torch.cat(
        [torch.cat([grad_y, gy_h], 3), torch.cat([-gy_v, -gy_hv], 3)], 2
    )
    return gx, gy


def frankot_chellappa(
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    low_freq: float = 1.0,
    mid_freq: float = 1.0,
    high_freq: float = 1.0,
) -> torch.Tensor:
    """Integrate gradients to a height field in ``[0, 1]`` via FFT.

    Frequency-band weights let callers emphasise large-scale, medium, or fine
    detail independently.
    """
    b, _, h, w = grad_x.shape
    device, dtype = grad_x.device, grad_x.dtype

    rows = (torch.arange(h, device=device, dtype=dtype) - (h // 2 + 1)) / (h - h % 2)
    cols = (torch.arange(w, device=device, dtype=dtype) - (w // 2 + 1)) / (w - w % 2)
    v_grid, u_grid = torch.meshgrid(rows, cols, indexing="ij")
    u_grid = torch.fft.ifftshift(u_grid)
    v_grid = torch.fft.ifftshift(v_grid)

    radius = torch.sqrt(u_grid**2 + v_grid**2)
    radius = radius / (radius.max() + 1e-8)
    low_mask = 1.0 - torch.clamp((radius - 0.1) / 0.15, 0.0, 1.0)
    high_mask = torch.clamp((radius - 0.35) / 0.25, 0.0, 1.0)
    mid_mask = torch.clamp(1.0 - low_mask - high_mask, 0.0, 1.0)
    low_mask = low_mask * low_mask * (3.0 - 2.0 * low_mask)
    high_mask = high_mask * high_mask * (3.0 - 2.0 * high_mask)
    freq_weight = low_mask * low_freq + mid_mask * mid_freq + high_mask * high_freq

    gx_f = torch.fft.fft2(grad_x.squeeze(1))
    gy_f = torch.fft.fft2(grad_y.squeeze(1))
    numerator = (-1j * u_grid * gx_f) + (-1j * v_grid * gy_f)
    denominator = u_grid**2 + v_grid**2 + 1e-16
    z_f = numerator / denominator
    z_f[:, 0, 0] = 0.0
    z_f = z_f * freq_weight

    z = torch.real(torch.fft.ifft2(z_f)).unsqueeze(1)
    z_min = z.amin(dim=(2, 3), keepdim=True)
    z_max = z.amax(dim=(2, 3), keepdim=True)
    return (z - z_min) / (z_max - z_min + 1e-8)


def normal_to_height(
    normal: torch.Tensor,
    seamless: bool = False,
    low_freq: float = 1.0,
    mid_freq: float = 1.0,
    high_freq: float = 1.0,
    intensity: float = 1.0,
    min_height: float = 0.25,
    invert: bool = True,
    height_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Height map in ``[min_height, 1]`` for Minecraft POM (black = deepest)."""
    b, _, h, w = normal.shape
    grad_x, grad_y = _gradient(normal)
    if not seamless:
        grad_x, grad_y = _make_seamless(grad_x, grad_y)

    height = frankot_chellappa(-grad_x, grad_y, low_freq, mid_freq, high_freq)
    if not seamless:
        height = height[:, :, :h, :w]
    if invert:
        height = 1.0 - height

    height = min_height + height * (1.0 - min_height)
    if intensity < 1.0:
        midpoint = (min_height + 1.0) / 2.0
        height = midpoint + (height - midpoint) * intensity

    if height_mask is not None:
        height = height * (1.0 - height_mask) + min_height * height_mask
    return height


def compute_divergence(normal: torch.Tensor) -> torch.Tensor:
    """``dnx/dx + dny/dy`` -- positive in crevices, negative on bumps."""
    nx = (normal[:, 0:1] - 0.5) * 2.0
    ny = (normal[:, 1:2] - 0.5) * 2.0
    kx, ky = _sobel(normal.device, normal.dtype)
    return F.conv2d(nx, kx, padding=1) + F.conv2d(ny, ky, padding=1)


def normal_to_ao(normal: torch.Tensor, strength: float = 2.0, blur_radius: int = 5) -> torch.Tensor:
    """Approximate AO in ``[0, 1]`` (1 = unoccluded) from normal divergence."""
    div = gaussian_blur(compute_divergence(normal), blur_radius)
    return torch.clamp(1.0 - torch.clamp(div * strength, 0.0, 1.0), 0.0, 1.0)


def derive_ao_and_height(
    normal: torch.Tensor,
    seamless: bool = False,
    ao_strength: float = 2.0,
    ao_blur: int = 5,
    height_low_freq: float = 1.0,
    height_mid_freq: float = 1.0,
    height_high_freq: float = 1.0,
    height_intensity: float = 1.0,
    height_min: float = 0.25,
    height_invert: bool = True,
    height_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper returning ``(ao, height)`` from a single normal map."""
    ao = normal_to_ao(normal, strength=ao_strength, blur_radius=ao_blur)
    height = normal_to_height(
        normal,
        seamless=seamless,
        low_freq=height_low_freq,
        mid_freq=height_mid_freq,
        high_freq=height_high_freq,
        intensity=height_intensity,
        min_height=height_min,
        invert=height_invert,
        height_mask=height_mask,
    )
    return ao, height
