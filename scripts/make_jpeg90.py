from pathlib import Path
from PIL import Image
import os

SRC_ROOT = Path("data/images")
DST_ROOT = Path("data/images_jpeg")
QUALITY = 90

def main():
    for p in SRC_ROOT.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        rel = p.relative_to(SRC_ROOT)
        out = DST_ROOT / rel
        out = out.with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(p).convert("RGB")
        img.save(out, format="JPEG", quality=QUALITY, optimize=True)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()

# Run:

# python scripts/make_jpeg90.py

# Small but important fix you should make now

# Your example manifest paths show data/docs/... but your file structure is:

# data/images/doc/...


# So: make sure manifests use paths under data/images/... (and scorer will rewrite to data/images_jpeg/... automatically).

#conver the png to jpeg 90 image the cropping and other thing itself do by preprocessor.
