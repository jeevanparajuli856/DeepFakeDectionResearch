import argparse
from pathlib import Path
import json

from utils.logger import setup_logger
from detectors.dire.dire_wrapper import DireDetector
from preprocessing.dire import preprocess_dire

from evaluation.scorer import run_scoring
from evaluation.calibrate import calibrate
from evaluation.metrics import compute_metrics


SCENARIOS = ["doc", "headshot", "scene"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to DIRE .pth checkpoint")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--use_cpu", action="store_true")

    ap.add_argument("--val_csv", default="data/manifests/val.csv")
    ap.add_argument("--test_seen_csv", default="data/manifests/test_seen.csv")
    ap.add_argument("--test_unseen_csv", default="data/manifests/test_unseen.csv")

    ap.add_argument("--out_root", default="outputs/dire")
    ap.add_argument("--fpr_targets", default="0.01,0.001")
    ap.add_argument("--jpeg", action="store_true", help="Run ONLY JPEG scoring (no calibration).")

    args = ap.parse_args()
    fpr_targets = [float(x) for x in args.fpr_targets.split(",")]

    out_root = Path(args.out_root)
    log_dir = out_root / "log"
    logger = setup_logger("dire_run", log_dir)

    logger.info("==== DIRE runner start ====")
    logger.info(f"model_path={args.model_path} arch={args.arch} cpu={args.use_cpu}")
    logger.info(f"fpr_targets={fpr_targets} jpeg_only={args.jpeg}")

    det = DireDetector(args.model_path, arch=args.arch, use_cpu=args.use_cpu)
    detector_fn = det.score

    # If jpeg-only: score JPEG test splits using existing calibration.json
    if args.jpeg:
        calib_path = out_root / "calibration.json"
        assert calib_path.exists(), f"Missing calibration: {calib_path}"

        scores_seen_j = out_root / "scores_test_seen_jpeg.csv"
        scores_unseen_j = out_root / "scores_test_unseen_jpeg.csv"

        run_scoring(args.test_seen_csv, "dire", detector_fn, preprocess_dire, str(scores_seen_j), True, logger)
        run_scoring(args.test_unseen_csv, "dire", detector_fn, preprocess_dire, str(scores_unseen_j), True, logger)

        metrics_seen_j = out_root / "metrics_test_seen_jpeg.json"
        metrics_unseen_j = out_root / "metrics_test_unseen_jpeg.json"

        compute_metrics(str(scores_seen_j), str(calib_path), fpr_targets, str(metrics_seen_j), logger)
        compute_metrics(str(scores_unseen_j), str(calib_path), fpr_targets, str(metrics_unseen_j), logger)

        logger.info("==== DIRE JPEG-only complete ====")
        return

    # Full PNG run
    scores_val = out_root / "scores_val.csv"
    run_scoring(args.val_csv, "dire", detector_fn, preprocess_dire, str(scores_val), False, logger)

    calib_path = out_root / "calibration.json"
    calibrate(str(scores_val), fpr_targets, str(calib_path), logger)

    scores_seen = out_root / "scores_test_seen.csv"
    scores_unseen = out_root / "scores_test_unseen.csv"

    run_scoring(args.test_seen_csv, "dire", detector_fn, preprocess_dire, str(scores_seen), False, logger)
    run_scoring(args.test_unseen_csv, "dire", detector_fn, preprocess_dire, str(scores_unseen), False, logger)

    metrics_seen = out_root / "metrics_test_seen.json"
    metrics_unseen = out_root / "metrics_test_unseen.json"

    compute_metrics(str(scores_seen), str(calib_path), fpr_targets, str(metrics_seen), logger)
    compute_metrics(str(scores_unseen), str(calib_path), fpr_targets, str(metrics_unseen), logger)

    # convenience summary
    with open(metrics_seen) as f:
        ms = json.load(f)
    with open(metrics_unseen) as f:
        mu = json.load(f)

    summary = {"seen_png": ms, "unseen_png": mu}
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("==== DIRE full PNG pipeline complete ====")
    logger.info("Next: `python runners/run_dire.py --model_path ... --jpeg` after JPEG-90 generation.")
    logger.info("==== done ====")


if __name__ == "__main__":
    main()

# val scoring (PNG)

# calibration (PNG val only)

# test_seen + test_unseen (PNG)

# then optional --jpeg mode: scores JPEG only using same calibration