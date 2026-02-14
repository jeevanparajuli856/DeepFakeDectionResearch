import pandas as pd
import numpy as np
from pathlib import Path
import logging

SEED = 825

# ---------------- CONFIG ----------------
INPUT_MANIFEST = "data/manifests/master_manifest_new.csv"
OUT_DIR = Path("data/manifests")

SCENARIOS = ["doc", "headshot", "scene"]

# counts PER SCENARIO
REAL_COUNTS = {
    "train": 300,
    "val": 100,
    "test_seen": 100,
}

SD_COUNTS = {
    "train": 300,
    "val": 100,
    "test_seen": 100,
}

NANO_COUNTS_BY_SCENARIO = {
    "doc": {"nano25": 334, "nanopro": 167},
    "headshot": {"nano25": 333, "nanopro": 166},
    "scene": {"nano25": 333, "nanopro": 167},
}
# ---------------------------------------


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def sample(df, n, rng):
    assert len(df) >= n, f"Not enough samples: need {n}, got {len(df)}"
    return df.sample(n=n, random_state=rng)


def main():
    setup_logger()
    rng = np.random.RandomState(SEED)

    df = pd.read_csv(INPUT_MANIFEST)
    df["split"] = "UNASSIGNED"

    for scenario in SCENARIOS:
        logging.info(f"Processing scenario: {scenario}")

        # ---------------- REAL ----------------
        real_df = df[
            (df.scenario == scenario) &
            (df.label == 0)
        ].copy()

        used_idx = set()

        for split, n in REAL_COUNTS.items():
            chosen = sample(real_df.drop(index=used_idx, errors="ignore"), n, rng)
            df.loc[chosen.index, "split"] = split
            used_idx.update(chosen.index)

        # ---------------- SD ----------------
        sd_df = df[
            (df.scenario == scenario) &
            (df.generator_family == "sd")
        ].copy()

        used_idx = set()

        for split, n in SD_COUNTS.items():
            chosen = sample(sd_df.drop(index=used_idx, errors="ignore"), n, rng)
            df.loc[chosen.index, "split"] = split
            used_idx.update(chosen.index)

        # ---------------- NANO (UNSEEN ONLY) ----------------
        for gen, n in NANO_COUNTS_BY_SCENARIO[scenario].items():
            nano_df = df[
                (df.scenario == scenario) &
                (df.generator_family == gen)
            ]
            chosen = sample(nano_df, n, rng)
            df.loc[chosen.index, "split"] = "test_unseen"

    # ---------------- SANITY CHECKS ----------------
    assert not (df[(df.generator_family.isin(["nano25", "nanopro"])) &
                    (df.split.isin(["train", "val", "test_seen"]))].any().any()), \
        "Nano leaked into seen splits!"

    assert not (df[(df.generator_family == "sd") &
                    (df.split == "test_unseen")].any().any()), \
        "SD leaked into unseen!"

    assert (df.split != "UNASSIGNED").all(), "Some samples were not assigned a split!"

    nano25_total = (df.generator_family == "nano25").sum()
    nanopro_total = (df.generator_family == "nanopro").sum()
    assert nano25_total == 1000, f"Expected 1000 nano25 samples, got {nano25_total}"
    assert nanopro_total == 500, f"Expected 500 nanopro samples, got {nanopro_total}"

    # ---------------- WRITE OUTPUTS ----------------
    for split in ["train", "val", "test_seen", "test_unseen"]:
        out_path = OUT_DIR / f"{split}.csv"
        df[df.split == split].to_csv(out_path, index=False)
        logging.info(f"Wrote {out_path}")

    logging.info("Dataset splitting complete and verified.")


if __name__ == "__main__":
    main()



# 🔍 Why this is reviewer-safe

# Explicit counts per scenario (no silent imbalance)

# Hard assertions (pipeline fails if rules are violated)

# No heuristic splitting

# Unseen ≠ seen by construction

# This is exactly how benchmarks like SIDBench enforce discipline.