"""Train the first Approach 2 model: a PyTorch MLP."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dl_common import (  # noqa: E402
    FraudDataset,
    MLP,
    calculate_metrics,
    load_frozen_contract,
    predict_probabilities,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-processed-dir",
        type=Path,
        default=PROJECT_DIR.parent / "Approch-1" / "data" / "processed",
        help="Existing leakage-excluded processed contract directory.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    processed_dir = PROJECT_DIR / "results" / "processed"
    splits = load_frozen_contract(args.source_processed_dir, processed_dir)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        FraudDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    model = MLP(input_dim=train_x.shape[1]).to(device)
    positive_count = float(train_y.sum())
    negative_count = float(len(train_y) - positive_count)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    checkpoint_path = PROJECT_DIR / "models" / "mlp" / "mlp_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_pr_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(targets)

        val_probabilities, _ = predict_probabilities(model, val_x, device)
        val_metrics = calculate_metrics(val_y, val_probabilities)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_y),
            "val_loss": None,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "MLP",
                    "input_dim": int(train_x.shape[1]),
                    "architecture": [64, 32],
                    "dropout": 0.20,
                    "seed": args.seed,
                    "pos_weight": negative_count / positive_count,
                    "best_val_metrics": val_metrics,
                    "best_epoch": best_epoch,
                    "feature_contract": "Approch-1 frozen 29-feature leakage-excluded baseline",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    training_seconds = time.perf_counter() - start_time
    (log_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "model": "MLP",
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_pr_auc,
        "training_seconds": training_seconds,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint": str(checkpoint_path),
        "train_size": len(train_y),
        "val_size": len(val_y),
        "train_fraud": int(train_y.sum()),
        "val_fraud": int(val_y.sum()),
    }
    (log_dir / "mlp_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
