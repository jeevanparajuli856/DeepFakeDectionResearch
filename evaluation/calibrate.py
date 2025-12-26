import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression


def calibrate(scores_csv, fpr_targets, out_json, logger):
    df = pd.read_csv(scores_csv)
    df = df.dropna(subset=["score"])

    y = df["label"].values
    s = df["score"].values

    fpr, tpr, thr = roc_curve(y, s)

    thresholds = {}
    for tgt in fpr_targets:
        idx = np.argmin(np.abs(fpr - tgt))
        thresholds[str(tgt)] = float(thr[idx])

    # logistic calibration (optional but useful)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(s.reshape(-1, 1), y)

    calib = {
        "thresholds": thresholds,
        "logistic": {
            "coef": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0]),
        },
    }

    with open(out_json, "w") as f:
        json.dump(calib, f, indent=2)

    logger.info(f"Saved calibration to {out_json}")
