# evaluation/plot_results.py

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# ============================================================
# 1. LOAD SUMMARY FILES
# ============================================================

def load_summaries(out_root="outputs", variant="png"):
    rows = []
    out_root = Path(out_root)

    summary_name = "summary.json" if variant == "png" else "summary_jpeg.json"
    seen_key = "seen_png" if variant == "png" else "seen_jpeg"
    unseen_key = "unseen_png" if variant == "png" else "unseen_jpeg"

    for detector in ["hfreq", "dire"]:
        det_root = out_root / detector
        if not det_root.exists():
            continue

        for scen_dir in det_root.iterdir():
            if not scen_dir.is_dir():
                continue

            summary_path = scen_dir / summary_name
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                s = json.load(f)

            scenario = s["scenario"]

            # Seen (SD)
            rows.append({
                "detector": detector,
                "scenario": scenario,
                "generator": "sd",
                **s[seen_key],
            })

            # Unseen (Nano)
            for gen, m in s[unseen_key].items():
                rows.append({
                    "detector": detector,
                    "scenario": scenario,
                    "generator": gen,
                    **m,
                })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 2. PLOTTING HELPERS
# ============================================================

def plot_tpr(df, fpr, save_dir=None, tag="png"):
    key = f"TPR@FPR={fpr}"

    for scenario in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scenario]

        plt.figure(figsize=(6, 4))
        for det in ["hfreq", "dire"]:
            d = sub[sub["detector"] == det].sort_values("generator")
            plt.plot(
                d["generator"],
                d[key],
                marker="o",
                linewidth=2,
                label=det.upper(),
            )

        plt.title(f"{scenario.upper()} — TPR @ FPR={fpr}")
        plt.xlabel("Generator")
        plt.ylabel("True Positive Rate")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_dir:
            out_path = Path(save_dir) / f"{scenario}_tpr_fpr{str(fpr).replace('.', '')}_{tag}.png"
            plt.savefig(out_path, dpi=300)
        else:
            plt.show()


def plot_auroc(df, save_dir=None, tag="png"):
    for scenario in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scenario]

        plt.figure(figsize=(6, 4))
        for det in ["hfreq", "dire"]:
            d = sub[sub["detector"] == det].sort_values("generator")
            plt.plot(
                d["generator"],
                d["auroc"],
                marker="o",
                linewidth=2,
                label=det.upper(),
            )

        plt.title(f"{scenario.upper()} — AUROC")
        plt.xlabel("Generator")
        plt.ylabel("AUROC")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_dir:
            out_path = Path(save_dir) / f"{scenario}_auroc_{tag}.png"
            plt.savefig(out_path, dpi=300)
        else:
            plt.show()


# ============================================================
# 3. MAIN ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Deepfake Benchmark Results")
    parser.add_argument("--out_root", default="outputs")
    parser.add_argument("--variant", choices=["png", "jpeg"], default="png")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    df = load_summaries(args.out_root, args.variant)

    print("Loaded rows:", len(df))
    print(df.head())

    # Main SOC/KYC plots
    plot_tpr(df, fpr=0.01, save_dir=args.save_dir, tag=args.variant)
    plot_tpr(df, fpr=0.001, save_dir=args.save_dir, tag=args.variant)

    # Ranking plot
    plot_auroc(df, save_dir=args.save_dir, tag=args.variant)


if __name__ == "__main__":
    main()
