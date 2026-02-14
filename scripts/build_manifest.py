from pathlib import Path
import pandas as pd

SRC_ROOT = Path("data/images")
OUT_MANIFEST = Path("data/manifests/master_manifest_new.csv")

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SCENARIOS = {"doc", "headshot", "scene"}


def parse_row(path: Path) -> dict:
    scenario = path.parent.name
    if scenario not in SCENARIOS:
        raise ValueError(f"Unexpected scenario folder: {scenario}")

    stem = path.stem

    if "_real_" in stem:
        label = 0
        generator_family = "real"
        prefix = f"{scenario}_"
        if not stem.startswith(prefix):
            raise ValueError(f"Unexpected real filename: {stem}")
        source = stem.split("_real_")[0][len(prefix):]
    elif "_nano_2_5_" in stem:
        label = 1
        generator_family = "nano25"
        source = "nano_2_5"
    elif "_nano_pro_" in stem:
        label = 1
        generator_family = "nanopro"
        source = "nano_pro"
    elif "_sd_" in stem:
        label = 1
        generator_family = "sd"
        source = "sd"
    else:
        raise ValueError(f"Unrecognized filename pattern: {stem}")

    rel_path = Path("data") / "images" / scenario / path.name

    return {
        "image_id": stem,
        "path": rel_path.as_posix(),
        "label": label,
        "scenario": scenario,
        "source": source,
        "generator_family": generator_family,
        "split": "UNASSIGNED",
    }


def main() -> None:
    rows = []
    for scenario_dir in sorted(SRC_ROOT.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if scenario_dir.name not in SCENARIOS:
            continue

        for path in sorted(scenario_dir.iterdir()):
            if path.is_dir():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append(parse_row(path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found under data/images")

    if df["image_id"].duplicated().any():
        dupes = df[df["image_id"].duplicated()]["image_id"].tolist()[:5]
        raise RuntimeError(f"Duplicate image_id values detected: {dupes}")

    df = df.sort_values("image_id").reset_index(drop=True)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MANIFEST, index=False)

    counts = df.groupby(["scenario", "generator_family"]).size().reset_index(name="count")
    print("Wrote manifest:", OUT_MANIFEST)
    print(counts)


if __name__ == "__main__":
    main()
