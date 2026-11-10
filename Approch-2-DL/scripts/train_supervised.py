"""Train Approach 2 supervised neural models: ResidualMLP and CNN1D.

Uses the same frozen 29-feature contract and training loop as train_mlp.py.
Checkpoints are saved to Approch-2-DL/models/<name>/best.pt.
Metrics are saved to Approch-2-DL/results/<name>_metrics.json.
"""
from __future__ import annotations

import argparse
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
    calculate_metrics,
    load_frozen_contract,
    predict_probabilities,
    set_seed,
)
from submission_models import ResidualMLP, CNN1D  # noqa: E402


MODEL_CLASSES = {"ResidualMLP": ResidualMLP, "CNN1D": CNN1D}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(MODEL_CLASSES),
        required=True,
        help="Which model to train.",
    )
    parser.add_argument(
        "--source-processed-dir",
        type=Path,
        default=PROJECT_DIR.parent / "Approch-1" / "data" / "processed",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name: str = args.model
    set_seed(args.seed)

    processed_dir = PROJECT_DIR / "results" / "processed"
    splits = load_frozen_contract(args.source_processed_dir, processed_dir)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    test_x, test_y = splits["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        FraudDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )

    ModelCls = MODEL_CLASSES[model_name]
    model = ModelCls(input_dim=train_x.shape[1]).to(device)
    positive_count = float(train_y.sum())
    negative_count = float(len(train_y) - positive_count)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    checkpoint_dir = PROJECT_DIR / "models" / model_name.lower()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name.lower()}_best.pt"

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
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(targets)

        val_probs, _ = predict_probabilities(model, val_x, device)
        val_metrics = calculate_metrics(val_y, val_probs)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  val_pr_auc={val_metrics['pr_auc']:.4f}  val_f1={val_metrics['f1']:.4f}")

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "input_dim": int(train_x.shape[1]),
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
            print(f"  Early stop at epoch {epoch} (patience {args.patience})")
            break

    training_seconds = time.perf_counter() - start_time
    print(f"\nTraining complete: best epoch {best_epoch}, val PR-AUC {best_pr_auc:.4f}, {training_seconds:.1f}s")

    # Evaluate best checkpoint on val + test
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eval_model = ModelCls(input_dim=int(checkpoint["input_dim"]))
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model.eval()

    metrics: dict = {}
    inference: dict = {}
    for split_name, (sx, sy) in [("val", (val_x, val_y)), ("test", (test_x, test_y))]:
        probs, elapsed_ms = predict_probabilities(eval_model, sx, torch.device("cpu"))
        m = calculate_metrics(sy, probs)
        m["inference_total_ms"] = elapsed_ms
        m["inference_ms_per_sample"] = elapsed_ms / len(sy)
        metrics[split_name] = m
        inference[split_name] = {"samples": len(sy), "total_ms": elapsed_ms}

    output = {
        "model": model_name,
        "primary_metric": "pr_auc",
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "feature_contract": "Approch-1 frozen 29-feature leakage-excluded baseline",
        "metrics": metrics,
        "inference": inference,
    }
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name.lower()}_metrics.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Metrics saved to {out_path}")
    print(f"Test  F1={metrics['test']['f1']:.4f}  PR-AUC={metrics['test']['pr_auc']:.4f}  ROC-AUC={metrics['test']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
