"""Train Approach 2 advanced DL models: TabTransformer and Autoencoder.

TabTransformer: supervised, trained with BCEWithLogitsLoss on fraud label.
Autoencoder: unsupervised reconstruction, threshold fitted on validation set.
  - Anomaly score = normalised reconstruction error (NOT a calibrated probability).
  - A ROC/PR curve is still computed using the error as a ranking score.
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
from dl_advanced_models import Autoencoder, TabTransformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["TabTransformer", "Autoencoder"],
        required=True,
    )
    parser.add_argument(
        "--source-processed-dir",
        type=Path,
        default=PROJECT_DIR.parent / "Approch-1" / "data" / "processed",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─── TabTransformer training ────────────────────────────────────────────────

def train_tab_transformer(args: argparse.Namespace, splits: dict) -> None:
    model_name = "TabTransformer"
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
    model = TabTransformer(input_dim=train_x.shape[1]).to(device)
    positive_count = float(train_y.sum())
    negative_count = float(len(train_y) - positive_count)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    checkpoint_dir = PROJECT_DIR / "models" / "tabtransformer"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "tabtransformer_best.pt"

    best_pr_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(targets)

        val_probs, _ = predict_probabilities(model, val_x, device)
        val_metrics = calculate_metrics(val_y, val_probs)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d}  val_pr_auc={val_metrics['pr_auc']:.4f}")

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
                    "best_val_metrics": val_metrics,
                    "best_epoch": best_epoch,
                    "feature_contract": "Approch-1 frozen 29-feature leakage-excluded baseline",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"  [{model_name}] Early stop at epoch {epoch}")
            break

    elapsed = time.perf_counter() - start_time
    print(f"\n[{model_name}] Best epoch {best_epoch}, val PR-AUC {best_pr_auc:.4f}, {elapsed:.1f}s")

    # Evaluate best
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eval_model = TabTransformer(input_dim=checkpoint["input_dim"])
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
    out_path = results_dir / "tabtransformer_metrics.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


# ─── Autoencoder training ────────────────────────────────────────────────────

def train_autoencoder(args: argparse.Namespace, splits: dict) -> None:
    model_name = "Autoencoder"
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    test_x, test_y = splits["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train only on legitimate (label=0) samples for a purer anomaly detector
    legit_mask = train_y == 0
    ae_train_x = train_x[legit_mask]
    ae_train_y = torch.zeros(len(ae_train_x), dtype=torch.float32)

    train_loader = DataLoader(
        FraudDataset(ae_train_x, ae_train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )

    model = Autoencoder(input_dim=train_x.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = PROJECT_DIR / "models" / "autoencoder"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "autoencoder_best.pt"

    best_val_loss = float("inf")
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, _ in train_loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model.encoder(features)
            x_hat = model.decoder(z)
            loss = loss_fn(x_hat, features)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(features)

        # Validation: measure reconstruction error on val set (all classes)
        model.eval()
        with torch.inference_mode():
            val_tensor = torch.from_numpy(val_x).to(device)
            val_errors = model.reconstruction_error(val_tensor).cpu()
        val_loss = float(val_errors.mean())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Autoencoder] Epoch {epoch:3d}  val_recon_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            threshold = model.fit_threshold(val_errors, percentile=95.0)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "input_dim": int(train_x.shape[1]),
                    "threshold": threshold,
                    "feature_contract": "Approch-1 frozen 29-feature leakage-excluded baseline",
                    "score_is_calibrated": False,
                    "score_interpretation": "normalised reconstruction error; higher = more anomalous",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"  [Autoencoder] Early stop at epoch {epoch}")
            break

    elapsed = time.perf_counter() - start_time
    print(f"\n[Autoencoder] Best val recon loss {best_val_loss:.6f}, {elapsed:.1f}s")

    # Evaluate: use reconstruction error as anomaly score, compute ranking metrics
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eval_model = Autoencoder(input_dim=checkpoint["input_dim"])
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model.threshold = checkpoint["threshold"]
    eval_model.eval()

    metrics: dict = {}
    inference: dict = {}
    for split_name, (sx, sy) in [("val", (val_x, val_y)), ("test", (test_x, test_y))]:
        import time as _time
        t0 = _time.perf_counter()
        with torch.inference_mode():
            errors = eval_model.reconstruction_error(torch.from_numpy(sx)).numpy()
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        # Normalise to [0,1] for a comparable score using threshold as midpoint
        t = eval_model.threshold
        # sigmoid-like: score near 0 for safe, near 1 for very anomalous
        norm_scores = 1.0 / (1.0 + np.exp(-(errors - t) * 5.0 / max(t, 1e-8)))
        m = calculate_metrics(sy, norm_scores)
        m["inference_total_ms"] = elapsed_ms
        m["inference_ms_per_sample"] = elapsed_ms / len(sy)
        m["threshold"] = float(eval_model.threshold)
        m["anomaly_score_note"] = "normalised reconstruction error, NOT a calibrated fraud probability"
        metrics[split_name] = m
        inference[split_name] = {"samples": len(sy), "total_ms": elapsed_ms}

    output = {
        "model": model_name,
        "primary_metric": "pr_auc",
        "checkpoint": str(checkpoint_path),
        "feature_contract": "Approch-1 frozen 29-feature leakage-excluded baseline",
        "score_is_calibrated": False,
        "score_interpretation": "reconstruction-error anomaly score; not a supervised fraud probability",
        "metrics": metrics,
        "inference": inference,
    }
    results_dir = PROJECT_DIR / "results"
    out_path = results_dir / "autoencoder_metrics.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Metrics: test F1={metrics['test']['f1']:.4f}  PR-AUC={metrics['test']['pr_auc']:.4f}")
    print(f"Saved -> {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    processed_dir = PROJECT_DIR / "results" / "processed"
    splits = load_frozen_contract(
        args.source_processed_dir, processed_dir
    )
    if args.model == "TabTransformer":
        train_tab_transformer(args, splits)
    else:
        train_autoencoder(args, splits)


if __name__ == "__main__":
    main()
