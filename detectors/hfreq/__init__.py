import torch
import numpy as np
from detectors.hfreq.model import HFreqCNN


class HFreqDetector:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(model_path, map_location=device)
        self.model = HFreqCNN().to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def score(self, x: torch.Tensor) -> float:
        """
        x: [1,256,256] freq map from preprocess_hfreq
        returns: float score (logit or prob). We'll return logit for thresholding consistency.
        """
        with torch.no_grad():
            xb = x.unsqueeze(0).to(self.device)  # [B,1,H,W]
            logit = self.model(xb).item()
        return float(logit)
