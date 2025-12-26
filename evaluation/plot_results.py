# evaluation/plot_results.py

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# ============================================================
# 1. LOAD SUMMARY FILES
# ============================================================

def load_summaries(out_root="outputs"):
    rows = []
    out_root = Path(out_root)

    for detector in ["hfreq", "dire"]:
        det_root = out_root / detector
        if not det_root.exists():
            continue

        for scen_dir in det_root.iterdir():
            if not scen_dir.is_dir():
                continue

            summary_path = scen_dir / "summary.json"
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
                **s["seen_png"],
            })

            # Unseen (Nano)
            for gen, m in s["unseen_png"].items():
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

def plot_tpr(df, fpr):
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
        plt.show()


def plot_auroc(df):
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
        plt.show()


# ============================================================
# 3. MAIN ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Deepfake Benchmark Results")
    parser.add_argument("--out_root", default="outputs")
    args = parser.parse_args()

    df = load_summaries(args.out_root)

    print("Loaded rows:", len(df))
    print(df.head())

    # Main SOC/KYC plots
    plot_tpr(df, fpr=0.01)
    plot_tpr(df, fpr=0.001)

    # Ranking plot
    plot_auroc(df)
    # plt.savefig("outputs/figures/scene_tpr_fpr1.png", dpi=300)


if __name__ == "__main__":
    main()
