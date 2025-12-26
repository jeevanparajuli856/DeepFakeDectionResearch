from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch


_dire_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0,1], [3,224,224]
])

_DIRE_MEAN = [0.485, 0.456, 0.406]
_DIRE_STD  = [0.229, 0.224, 0.225]


def preprocess_dire(path: str, aug_norm: bool = True) -> torch.Tensor:
    """
    Returns: Tensor [3,224,224]
    """
    img = Image.open(path).convert("RGB")
    x = _dire_trans(img)
    if aug_norm:
        x = TF.normalize(x, mean=_DIRE_MEAN, std=_DIRE_STD)
    return x
