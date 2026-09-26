"""Evaluate the saved Approach 2 MLP and verify reproducible inference."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dl_common import MLP, calculate_metrics, load_frozen_contract, predict_probabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-processed-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "processed",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "models" / "mlp" / "mlp_best.pt",
    )
    return parser.parse_args()


def save_metric_table(metrics: dict[str, dict[str, object]], path: Path) -> None:
    rows = []
    for split, values in metrics.items():
        row = {"split": split}
        row.update({key: value for key, value in values.items() if key != "confusion_matrix"})
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    splits = load_frozen_contract(args.source_processed_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MLP(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = {}
    probabilities_by_split = {}
    inference = {}
    for split in ("val", "test"):
        features, targets = splits[split]
        probabilities, elapsed_ms = predict_probabilities(model, features, torch.device("cpu"))
        split_metrics = calculate_metrics(targets, probabilities)
        split_metrics["inference_total_ms"] = elapsed_ms
        split_metrics["inference_ms_per_sample"] = elapsed_ms / len(targets)
        metrics[split] = split_metrics
        probabilities_by_split[split] = (targets, probabilities)
        inference[split] = {"samples": len(targets), "total_ms": elapsed_ms}

    results_dir = PROJECT_DIR / "results"
    evaluation_dir = PROJECT_DIR / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": "MLP",
        "primary_metric": "pr_auc",
        "checkpoint": str(args.checkpoint),
        "best_epoch": checkpoint["best_epoch"],
        "feature_contract": checkpoint["feature_contract"],
        "metrics": metrics,
        "inference": inference,
        "methodological_notes": {
            "ClaimStatus": "Excluded as post-adjudication leakage.",
            "Cluster": "Retained for baseline comparability but may be a target proxy; ablation and temporal validation are required.",
            "ClaimAmount": "Retained; strong class separation may indicate an engineered shortcut.",
            "identifiers_and_codes": "ClaimID, PatientID, ProviderID, DiagnosisCode, ProcedureCode, and ProviderLocation are excluded because they are unique or near-unique in this dataset.",
        },
    }
    (results_dir / "mlp_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    save_metric_table(metrics, results_dir / "mlp_metrics.csv")

    targets, probabilities = probabilities_by_split["test"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(targets, probabilities >= 0.5, display_labels=["Legitimate", "Fraud"], cmap="Blues", ax=ax)
    ax.set_title("MLP Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(evaluation_dir / "mlp_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(targets, probabilities, ax=ax, name="MLP")
    ax.set_title("MLP Test ROC Curve")
    fig.tight_layout()
    fig.savefig(evaluation_dir / "mlp_roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(targets, probabilities, ax=ax, name="MLP")
    ax.set_title("MLP Test Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(evaluation_dir / "mlp_pr_curve.png", dpi=150)
    plt.close(fig)

    history_path = PROJECT_DIR / "logs" / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
        ax.plot(epochs, [row["val_pr_auc"] for row in history], label="Validation PR-AUC")
        ax.set_xlabel("Epoch")
        ax.set_title("MLP Training History")
        ax.legend()
        fig.tight_layout()
        fig.savefig(evaluation_dir / "mlp_training_history.png", dpi=150)
        plt.close(fig)

    # A single-row inference smoke check verifies that the checkpoint can serve a claim.
    sample_features = splits["test"][0][:1]
    sample_probability, sample_elapsed_ms = predict_probabilities(model, sample_features, torch.device("cpu"))
    smoke = {
        "input_shape": list(sample_features.shape),
        "probability_fraud": float(sample_probability[0]),
        "prediction": "Fraud" if sample_probability[0] >= 0.5 else "Legitimate",
        "inference_ms": sample_elapsed_ms,
    }
    (results_dir / "inference_smoke_test.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(json.dumps({"inference_smoke_test": smoke}, indent=2))


if __name__ == "__main__":
    main()
