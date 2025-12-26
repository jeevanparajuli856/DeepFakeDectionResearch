import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def build_unseen_eval_df(df_seen, df_unseen, gen):
    """
    Build valid evaluation set for unseen generators:
    real from seen + fake from unseen.
    """
    real_df = df_seen[df_seen["label"] == 0]
    fake_df = df_unseen[df_unseen["generator_family"] == gen]

    if real_df.empty or fake_df.empty:
        return None

    return pd.concat([real_df, fake_df], ignore_index=True)


def filter_unseen(df, gen):
    real = df[df["label"] == 0]
    fake = df[df["generator_family"] == gen]
    return pd.concat([real, fake], ignore_index=True)

def compute_metrics(scores_csv, calib_json, fpr_targets, out_json, logger):
    df = pd.read_csv(scores_csv).dropna(subset=["score"])
    y = df["label"].values
    s = df["score"].values

    with open(calib_json) as f:
        calib = json.load(f)

    metrics = {
        "auroc": float(roc_auc_score(y, s))
    }

    for tgt in fpr_targets:
        thr = calib["thresholds"][str(tgt)]
        preds = (s >= thr).astype(int)

        fp = ((preds == 1) & (y == 0)).sum()
        tp = ((preds == 1) & (y == 1)).sum()
        fn = ((preds == 0) & (y == 1)).sum()

        fpr = fp / max((y == 0).sum(), 1)
        tpr = tp / max((tp + fn), 1)

        metrics[f"TPR@FPR={tgt}"] = float(tpr)
        metrics[f"FPR@thr={tgt}"] = float(fpr)

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {out_json}")
