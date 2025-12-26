import torch
from detectors.dire.load_model import load_dire_model


class DireDetector:
    """
    Wrapper around official DIRE classifier.
    Input:  [3,224,224] tensor (already normalized if aug_norm=True)
    Output: probability of being synthetic (float)
    """
    def __init__(self, model_path: str, arch: str = "resnet50", use_cpu: bool = False):
        self.model, self.device = load_dire_model(model_path, arch=arch, use_cpu=use_cpu)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> float:
        xb = x.unsqueeze(0).to(self.device)  # [1,3,224,224]
        prob = self.model(xb).sigmoid().item()
        return float(prob)
