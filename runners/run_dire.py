import argparse
from pathlib import Path
import json
import pandas as pd

from utils.logger import setup_logger
from detectors.dire.dire_wrapper import DireDetector
from preprocessing.dire import preprocess_dire

from evaluation.scorer import run_scoring
from evaluation.calibrate import calibrate
from evaluation.metrics import compute_metrics, build_unseen_eval_df


SCENARIOS = ["doc", "headshot", "scene"]
UNSEEN_GENERATORS = ["nano25", "nanopro"]


def main():
    parser = argparse.ArgumentParser(description="Run DIRE benchmark (per-scenario)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--use_cpu", action="store_true")

    parser.add_argument("--val_csv", default="data/manifests/val.csv")
    parser.add_argument("--test_seen_csv", default="data/manifests/test_seen.csv")
    parser.add_argument("--test_unseen_csv", default="data/manifests/test_unseen.csv")

    parser.add_argument("--out_root", default="outputs/dire")
    parser.add_argument("--fpr_targets", default="0.01,0.001")
    parser.add_argument("--jpeg", action="store_true")

    args = parser.parse_args()
    fpr_targets = [float(x) for x in args.fpr_targets.split(",")]

    out_root = Path(args.out_root)
    log_dir = out_root / "log"
    logger = setup_logger("dire_run", log_dir)

    detector = DireDetector(
        model_path=args.model_path,
        arch=args.arch,
        use_cpu=args.use_cpu,
    )
    detector_fn = detector.score

    # ================= JPEG MODE =================
    if args.jpeg:
        for scenario in SCENARIOS:
            scen_dir = out_root / scenario
            calib_path = scen_dir / "calibration.json"

            scores_seen = scen_dir / "scores_test_seen_jpeg.csv"
            scores_unseen = scen_dir / "scores_test_unseen_jpeg.csv"

            run_scoring(
                args.test_seen_csv, "dire", detector_fn,
                preprocess_dire, str(scores_seen),
                use_jpeg=True, logger=logger, scenario=scenario
            )
            run_scoring(
                args.test_unseen_csv, "dire", detector_fn,
                preprocess_dire, str(scores_unseen),
                use_jpeg=True, logger=logger, scenario=scenario
            )

            metrics_seen = scen_dir / "metrics_test_seen_jpeg.json"
            compute_metrics(str(scores_seen), str(calib_path), fpr_targets, str(metrics_seen), logger)

            df_seen = pd.read_csv(scores_seen)
            df_unseen = pd.read_csv(scores_unseen)
            unseen_summary = {}

            for gen in UNSEEN_GENERATORS:
                sub_df = build_unseen_eval_df(df_seen, df_unseen, gen)
                if sub_df is None:
                    continue

                sub_csv = scen_dir / f"scores_test_unseen_{gen}_jpeg.csv"
                sub_df.to_csv(sub_csv, index=False)

                out_json = scen_dir / f"metrics_test_unseen_{gen}_jpeg.json"
                compute_metrics(str(sub_csv), str(calib_path), fpr_targets, str(out_json), logger)

                with open(out_json) as f:
                    unseen_summary[gen] = json.load(f)

            summary = {
                "scenario": scenario,
                "seen_jpeg": json.load(open(metrics_seen)),
                "unseen_jpeg": unseen_summary,
            }

            with open(scen_dir / "summary_jpeg.json", "w") as f:
                json.dump(summary, f, indent=2)

        return

    # ================= PNG MODE =================
    for scenario in SCENARIOS:
        scen_dir = out_root / scenario
        scen_dir.mkdir(parents=True, exist_ok=True)

        scores_val = scen_dir / "scores_val.csv"
        run_scoring(
            args.val_csv, "dire", detector_fn,
            preprocess_dire, str(scores_val),
            use_jpeg=False, logger=logger, scenario=scenario
        )

        calib_path = scen_dir / "calibration.json"
        calibrate(str(scores_val), fpr_targets, str(calib_path), logger)

        scores_seen = scen_dir / "scores_test_seen.csv"
        run_scoring(
            args.test_seen_csv, "dire", detector_fn,
            preprocess_dire, str(scores_seen),
            use_jpeg=False, logger=logger, scenario=scenario
        )

        metrics_seen = scen_dir / "metrics_test_seen.json"
        compute_metrics(str(scores_seen), str(calib_path), fpr_targets, str(metrics_seen), logger)

        scores_unseen = scen_dir / "scores_test_unseen.csv"
        run_scoring(
            args.test_unseen_csv, "dire", detector_fn,
            preprocess_dire, str(scores_unseen),
            use_jpeg=False, logger=logger, scenario=scenario
        )

        df_seen = pd.read_csv(scores_seen)
        df_unseen = pd.read_csv(scores_unseen)
        unseen_summary = {}

        for gen in UNSEEN_GENERATORS:
            sub_df = build_unseen_eval_df(df_seen, df_unseen, gen)
            if sub_df is None:
                continue

            sub_csv = scen_dir / f"scores_test_unseen_{gen}.csv"
            sub_df.to_csv(sub_csv, index=False)

            out_json = scen_dir / f"metrics_test_unseen_{gen}.json"
            compute_metrics(str(sub_csv), str(calib_path), fpr_targets, str(out_json), logger)

            with open(out_json) as f:
                unseen_summary[gen] = json.load(f)

        summary = {
            "scenario": scenario,
            "seen_png": json.load(open(metrics_seen)),
            "unseen_png": unseen_summary,
        }

        with open(scen_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
