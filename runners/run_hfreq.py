import argparse
from pathlib import Path
import json
import pandas as pd

from utils.logger import setup_logger
from detectors.hfreq.trainer import train_one_scenario
from detectors.hfreq import HFreqDetector
from preprocessing.hfreq import preprocess_hfreq

from evaluation.scorer import run_scoring
from evaluation.calibrate import calibrate
from evaluation.metrics import compute_metrics, filter_unseen, build_unseen_eval_df


SCENARIOS = ["doc", "headshot", "scene"]
UNSEEN_GENERATORS = ["nano25", "nanopro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/manifests/train.csv")
    ap.add_argument("--val_csv", default="data/manifests/val.csv")
    ap.add_argument("--test_seen_csv", default="data/manifests/test_seen.csv")
    ap.add_argument("--test_unseen_csv", default="data/manifests/test_unseen.csv")

    ap.add_argument("--out_root", default="outputs/hfreq")
    ap.add_argument("--seed", type=int, default=825)
    ap.add_argument("--fpr_targets", default="0.01,0.001")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)

    ap.add_argument("--jpeg", action="store_true",
                    help="Run ONLY JPEG scoring (no training/calibration).")

    args = ap.parse_args()
    fpr_targets = [float(x) for x in args.fpr_targets.split(",")]

    out_root = Path(args.out_root)
    log_dir = out_root / "log"
    logger = setup_logger("hfreq_run", log_dir)

    logger.info("==== hfreq runner start ====")
    logger.info(f"jpeg_only={args.jpeg}")

    # ============================================================
    # JPEG-ONLY MODE (reuse existing model + calibration)
    # ============================================================
    if args.jpeg:
        for scenario in SCENARIOS:
            scen_dir = out_root / scenario
            model_path = scen_dir / f"model_{scenario}.pt"
            calib_path = scen_dir / "calibration.json"

            assert model_path.exists(), f"Missing model: {model_path}"
            assert calib_path.exists(), f"Missing calibration: {calib_path}"

            det = HFreqDetector(str(model_path))
            detector_fn = det.score

            # Score JPEG images
            scores_seen_j = scen_dir / "scores_test_seen_jpeg.csv"
            scores_unseen_j = scen_dir / "scores_test_unseen_jpeg.csv"

            run_scoring(
                csv_path=args.test_seen_csv,
                detector_name="hfreq",
                detector_fn=detector_fn,
                preprocess_fn=preprocess_hfreq,
                out_csv=str(scores_seen_j),
                use_jpeg=True,
                logger=logger,
                scenario=scenario,
            )

            run_scoring(
                csv_path=args.test_unseen_csv,
                detector_name="hfreq",
                detector_fn=detector_fn,
                preprocess_fn=preprocess_hfreq,
                out_csv=str(scores_unseen_j),
                use_jpeg=True,
                logger=logger,
                scenario=scenario,
            )

            # ---- Metrics (SEEN)
            metrics_seen_j = scen_dir / "metrics_test_seen_jpeg.json"
            compute_metrics(
                str(scores_seen_j),
                str(calib_path),
                fpr_targets,
                str(metrics_seen_j),
                logger,
            )

            # ---- Metrics (UNSEEN per generator)
            df_seen = pd.read_csv(scores_seen_j)
            df_unseen = pd.read_csv(scores_unseen_j)

            unseen_summary = {}

            for gen in UNSEEN_GENERATORS:
                sub_df = build_unseen_eval_df(df_seen, df_unseen, gen)
                if sub_df is None:
                    logger.warning(f"Skip unseen {gen}: missing data")
                    continue

                sub_csv = scen_dir / f"scores_test_unseen_{gen}_jpeg.csv"
                sub_df.to_csv(sub_csv, index=False)

                out_json = scen_dir / f"metrics_test_unseen_{gen}_jpeg.json"
                compute_metrics(
                    str(sub_csv),
                    str(calib_path),
                    fpr_targets,
                    str(out_json),
                    logger,
                )

                with open(out_json) as f:
                    unseen_summary[gen] = json.load(f)

            summary = {
                "scenario": scenario,
                "seen_jpeg": json.load(open(metrics_seen_j)),
                "unseen_jpeg": unseen_summary,
            }

            with open(scen_dir / "summary_jpeg.json", "w") as f:
                json.dump(summary, f, indent=2)

        logger.info("==== hfreq JPEG-only scoring complete ====")
        return

    # ============================================================
    # FULL PNG PIPELINE
    # ============================================================
    for scenario in SCENARIOS:
        scen_dir = out_root / scenario
        scen_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        model_path = train_one_scenario(
            scenario=scenario,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            out_dir=str(scen_dir),
            logger=logger,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

        det = HFreqDetector(model_path)
        detector_fn = det.score

        # 2) Validation scoring (PNG)
        scores_val = scen_dir / "scores_val.csv"
        run_scoring(
            csv_path=args.val_csv,
            detector_name="hfreq",
            detector_fn=detector_fn,
            preprocess_fn=preprocess_hfreq,
            out_csv=str(scores_val),
            use_jpeg=False,
            logger=logger,
            scenario=scenario,
        )

        # 3) Calibration
        calib_path = scen_dir / "calibration.json"
        calibrate(str(scores_val), fpr_targets, str(calib_path), logger)

        # 4) Test SEEN
        scores_seen = scen_dir / "scores_test_seen.csv"
        run_scoring(
            csv_path=args.test_seen_csv,
            detector_name="hfreq",
            detector_fn=detector_fn,
            preprocess_fn=preprocess_hfreq,
            out_csv=str(scores_seen),
            use_jpeg=False,
            logger=logger,
            scenario=scenario,
        )

        metrics_seen = scen_dir / "metrics_test_seen.json"
        compute_metrics(str(scores_seen), str(calib_path), fpr_targets, str(metrics_seen), logger)

        # 5) Test UNSEEN (per generator)
        scores_unseen = scen_dir / "scores_test_unseen.csv"
        run_scoring(
            csv_path=args.test_unseen_csv,
            detector_name="hfreq",
            detector_fn=detector_fn,
            preprocess_fn=preprocess_hfreq,
            out_csv=str(scores_unseen),
            use_jpeg=False,
            logger=logger,
            scenario=scenario,
        )

        

        df_seen = pd.read_csv(scores_seen)
        df_unseen = pd.read_csv(scores_unseen)
        unseen_summary = {}
        for gen in ["nano25", "nanopro"]:
            sub_df = build_unseen_eval_df(df_seen, df_unseen, gen)
            if sub_df is None:
                logger.warning(f"Skip unseen {gen}: missing data")
                continue

            sub_csv = scen_dir / f"scores_test_unseen_{gen}.csv"
            sub_df.to_csv(sub_csv, index=False)

            out_json = scen_dir / f"metrics_test_unseen_{gen}.json"
            compute_metrics(
                str(sub_csv),
                str(calib_path),
                fpr_targets,
                str(out_json),
                logger,
            )

            with open(out_json) as f:
                unseen_summary[gen] = json.load(f)

        # 6) Summary
        summary = {
            "scenario": scenario,
            "seen_png": json.load(open(metrics_seen)),
            "unseen_png": unseen_summary,
        }

        with open(scen_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[{scenario}] summary written")

    logger.info("==== hfreq full PNG pipeline complete ====")


if __name__ == "__main__":
    main()
