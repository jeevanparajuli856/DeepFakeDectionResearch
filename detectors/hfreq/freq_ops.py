import torch
import torch.fft


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    img: Tensor [3,H,W] in range [0,1]
    returns: Tensor [1,H,W] luminance (Y)
    """
    r, g, b = img[0:1], img[1:2], img[2:3]
    # BT.601 luma
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def fft_log_magnitude(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    y: Tensor [1,H,W] in [0,1]
    returns: Tensor [1,H,W] log magnitude spectrum (shifted)
    """
    # remove DC bias to stabilize
    y0 = y - y.mean(dim=(-2, -1), keepdim=True)

    F = torch.fft.fft2(y0.squeeze(0))  # [H,W] complex
    F = torch.fft.fftshift(F)          # center low-freq
    mag = torch.abs(F)                 # [H,W]
    logmag = torch.log(mag + eps)

    # back to [1,H,W]
    return logmag.unsqueeze(0)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    """
    Per-sample normalization to keep training stable.
    x: [1,H,W]
    """
    mean = x.mean()
    std = x.std().clamp(min=1e-6)
    return (x - mean) / std
