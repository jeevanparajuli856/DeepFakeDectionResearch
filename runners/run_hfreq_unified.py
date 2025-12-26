import argparse
from pathlib import Path
import json

from utils.logger import setup_logger
from detectors.hfreq.hfreq_wrapper import HFreqDetector
from evaluation.scorer import run_scoring
from evaluation.calibrate import calibrate
from evaluation.metrics import compute_metrics
from trainers.hfreq_trainer import train_one_scenario
from preprocessing.hfreq import preprocess_hfreq


SCENARIOS = ["doc", "headshot", "scene"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_seen_csv", required=True)
    ap.add_argument("--test_unseen_csv", required=True)

    ap.add_argument("--out_root", default="outputs/hfreq_unified")
    ap.add_argument("--fpr_targets", default="0.01,0.001")
    ap.add_argument("--jpeg", action="store_true")

    args = ap.parse_args()
    fpr_targets = [float(x) for x in args.fpr_targets.split(",")]

    out_root = Path(args.out_root)
    log_dir = out_root / "log"
    logger = setup_logger("hfreq_unified", log_dir)

    logger.info("==== HFREQ unified training start ====")

    # -------------------------------------------------
    # 1) Train unified HFREQ model (ONCE)
    # -------------------------------------------------
    model_dir = out_root / "model"
    model_path = train_one_scenario(
        scenario="unified",
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        out_dir=model_dir,
        logger=logger,
        unified=True,
    )

    det = HFreqDetector(model_path)
    detector_fn = det.score

    # -------------------------------------------------
    # 2) Per-scenario evaluation + calibration
    # -------------------------------------------------
    for scenario in SCENARIOS:
        logger.info(f"==== Scenario: {scenario} ====")
        scen_dir = out_root / scenario
        scen_dir.mkdir(parents=True, exist_ok=True)

        # ---- VAL (PNG)
        scores_val = scen_dir / "scores_val.csv"
        run_scoring(
            args.val_csv,
            "hfreq_unified",
            detector_fn,
            preprocess_hfreq,
            str(scores_val),
            use_jpeg=False,
            logger=logger,
            scenario=scenario,
        )

        calib_path = scen_dir / "calibration.json"
        calibrate(str(scores_val), fpr_targets, str(calib_path), logger)

        # ---- TEST SEEN
        scores_seen = scen_dir / "scores_test_seen.csv"
        run_scoring(
            args.test_seen_csv,
            "hfreq_unified",
            detector_fn,
            preprocess_hfreq,
            str(scores_seen),
            False,
            logger,
            scenario=scenario,
        )

        # ---- TEST UNSEEN
        scores_unseen = scen_dir / "scores_test_unseen.csv"
        run_scoring(
            args.test_unseen_csv,
            "hfreq_unified",
            detector_fn,
            preprocess_hfreq,
            str(scores_unseen),
            False,
            logger,
            scenario=scenario,
        )

        metrics_seen = scen_dir / "metrics_test_seen.json"
        metrics_unseen = scen_dir / "metrics_test_unseen.json"

        compute_metrics(str(scores_seen), str(calib_path), fpr_targets, str(metrics_seen), logger)
        compute_metrics(str(scores_unseen), str(calib_path), fpr_targets, str(metrics_unseen), logger)

        with open(metrics_seen) as f:
            ms = json.load(f)
        with open(metrics_unseen) as f:
            mu = json.load(f)

        summary = {
            "scenario": scenario,
            "training": "unified",
            "seen": ms,
            "unseen": mu,
        }
        with open(scen_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    logger.info("==== HFREQ unified pipeline complete ====")


if __name__ == "__main__":
    main()