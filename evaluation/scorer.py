import pandas as pd
from tqdm import tqdm

from evaluation.path_resolver import resolve_image_path, assert_exists


def run_scoring(
    csv_path: str,
    detector_name: str,
    detector_fn,
    preprocess_fn,
    out_csv: str,
    use_jpeg: bool,
    logger,
    scenario:str | None = None,
):
    """
    detector_fn(image_tensor) -> float
    preprocess_fn(path) -> tensor
    """
    df = pd.read_csv(csv_path)
    if scenario is not None:
        df = df[df["scenario"] == scenario].copy()
        logger.info(f"Scenario filter applied: {scenario}")
    scores = []

    logger.info(f"Scoring {len(df)} samples from {csv_path}")
    logger.info(f"JPEG mode: {use_jpeg}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = resolve_image_path(row["path"], use_jpeg)
        assert_exists(img_path)

        try:
            img = preprocess_fn(img_path)
            score = detector_fn(img)
        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}")
            score = None

        scores.append(score)

    df["score"] = scores
    df.to_csv(out_csv, index=False)

    logger.info(f"Wrote scores to {out_csv}")
