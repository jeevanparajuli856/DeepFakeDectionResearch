import re
from pathlib import Path


def resolve_image_path(path: str, use_jpeg: bool) -> str:
    """
    Resolve PNG master path to JPEG-90 path at runtime.
    Assumes identical relative structure under data/images_jpeg/.
    """
    if not use_jpeg:
        return path

    p = path.replace("data/images/", "data/images_jpeg/")
    p = re.sub(r"\.(png|jpg|jpeg)$", ".jpg", p, flags=re.IGNORECASE)
    return p


def assert_exists(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resolved image path not found: {path}")
