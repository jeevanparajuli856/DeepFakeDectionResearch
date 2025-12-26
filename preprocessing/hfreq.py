from PIL import Image
import torch
import torchvision.transforms as T

from detectors.hfreq.freq_ops import rgb_to_y, fft_log_magnitude, normalize_map


_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),  # [0,1], [3,H,W]
])


def preprocess_hfreq(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = _tf(img)              # [3,256,256]
    y = rgb_to_y(x)           # [1,256,256]
    f = fft_log_magnitude(y)  # [1,256,256]
    f = normalize_map(f)      # [1,256,256]
    return f
