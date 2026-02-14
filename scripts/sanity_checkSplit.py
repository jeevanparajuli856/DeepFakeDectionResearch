import pandas as pd
from pathlib import Path

MANIFEST = "data/manifests/master_manifest_new.csv"

PNG_ROOT = Path("data/images")
JPEG_ROOT = Path("data/images_jpeg")

df = pd.read_csv(MANIFEST)

# ----------------------------
# 1) PNG path sanity check
# ----------------------------
bad_png = df[~df["path"].str.startswith(str(PNG_ROOT))]

if len(bad_png) > 0:
    print("❌ Invalid PNG paths detected:")
    print(bad_png[["image_id", "path"]].head())
else:
    print("✅ All PNG paths are valid.")

# ----------------------------
# 2) JPEG resolution check
# ----------------------------
def png_to_jpeg_path(png_path: str) -> Path:
    """
    Mimics scorer JPEG path resolution logic.
    """
    p = Path(png_path)
    rel = p.relative_to(PNG_ROOT)
    return JPEG_ROOT / rel.with_suffix(".jpg")

missing_jpeg = []

for _, row in df.iterrows():
    png_path = Path(row["path"])
    jpeg_path = png_to_jpeg_path(row["path"])

    if not jpeg_path.exists():
        missing_jpeg.append({
            "image_id": row["image_id"],
            "png_path": str(png_path),
            "expected_jpeg": str(jpeg_path),
        })

# ----------------------------
# 3) Report
# ----------------------------
if missing_jpeg:
    print(f"\n❌ Missing JPEG files: {len(missing_jpeg)}")
    print(pd.DataFrame(missing_jpeg).head())
else:
    print("✅ All JPEG paths resolve correctly.")
