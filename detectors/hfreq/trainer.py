import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from detectors.hfreq.model import HFreqCNN
from preprocessing.hfreq import preprocess_hfreq


class ManifestDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = preprocess_hfreq(row["path"])
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_scenario(
    scenario: str,
    train_csv: str,
    val_csv: str,
    out_dir: str,
    logger,
    seed: int = 825,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    device: str = None,
):
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load manifests
    tr = pd.read_csv(train_csv)
    va = pd.read_csv(val_csv)

    # filter scenario + real/sd only
    tr = tr[(tr.scenario == scenario) & (tr.generator_family.isin(["real", "sd"]))]
    va = va[(va.scenario == scenario) & (va.generator_family.isin(["real", "sd"]))]

    logger.info(f"[{scenario}] Train samples: {len(tr)} | Val samples: {len(va)}")

    train_loader = DataLoader(ManifestDataset(tr), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ManifestDataset(va), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = HFreqCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_path = out_dir / f"model_{scenario}.pt"
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / max(len(train_loader.dataset), 1)

        # ---- Validate AUROC
        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x).cpu().numpy()
                prob = 1 / (1 + np.exp(-logits))  # sigmoid
                ys.append(y.numpy())
                ss.append(prob)

        ys = np.concatenate(ys)
        ss = np.concatenate(ss)
        auc = roc_auc_score(ys, ss)

        logger.info(f"[{scenario}] epoch={epoch} loss={avg_loss:.4f} val_auroc={auc:.4f}")

        if auc > best_auc + 1e-4:
            best_auc = auc
            patience_left = patience
            torch.save({"model_state": model.state_dict()}, best_path)
            logger.info(f"[{scenario}] New best AUROC={best_auc:.4f} -> saved {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"[{scenario}] Early stopping.")
                break

    meta = {"scenario": scenario, "best_val_auroc": float(best_auc)}
    with open(out_dir / f"train_meta_{scenario}.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[{scenario}] Training complete. Best AUROC={best_auc:.4f}")
    return str(best_path)
