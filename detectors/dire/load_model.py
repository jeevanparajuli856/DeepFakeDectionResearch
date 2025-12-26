import sys
from pathlib import Path
import torch

# Point Python to the official DIRE repo folder (detectors/dire_main)
DIRE_MAIN_DIR = Path(__file__).resolve().parents[1] / "dire_main"
if str(DIRE_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(DIRE_MAIN_DIR))

from utils.utils import get_network  # noqa: E402


def load_dire_model(model_path: str, arch: str = "resnet50", use_cpu: bool = False):
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    model = get_network(arch)
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model, device
