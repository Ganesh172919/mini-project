"""Reusable data, model, and metric helpers for Approach 2."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import Dataset


EXPECTED_FEATURE_COUNT = 29
SPLIT_NAMES = ("train", "val", "test")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class FraudDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class MLP(nn.Module):
    def __init__(self, input_dim: int = EXPECTED_FEATURE_COUNT) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _validate_processed_directory(processed_dir: Path) -> None:
    missing = [
        name
        for name in [
            "feature_names.txt",
            *(f"X_{split}.npy" for split in SPLIT_NAMES),
            *(f"y_{split}.npy" for split in SPLIT_NAMES),
        ]
        if not (processed_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Processed contract is incomplete at {processed_dir}: {', '.join(missing)}"
        )


def load_frozen_contract(
    source_dir: Path,
    destination_dir: Path | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load the existing leakage-excluded split and optionally copy it locally."""
    source_dir = source_dir.resolve()
    _validate_processed_directory(source_dir)
    if destination_dir is not None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "feature_names.txt",
            *(f"X_{split}.npy" for split in SPLIT_NAMES),
            *(f"y_{split}.npy" for split in SPLIT_NAMES),
        ]:
            shutil.copy2(source_dir / name, destination_dir / name)
        source_dir = destination_dir.resolve()

    feature_names = (source_dir / "feature_names.txt").read_text(encoding="utf-8").splitlines()
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected 29 features, found {len(feature_names)}")

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLIT_NAMES:
        features = np.load(source_dir / f"X_{split}.npy")
        targets = np.load(source_dir / f"y_{split}.npy")
        if features.ndim != 2 or features.shape[1] != len(feature_names):
            raise ValueError(f"Invalid {split} feature shape: {features.shape}")
        if len(features) != len(targets):
            raise ValueError(f"Feature/target length mismatch in {split}")
        splits[split] = (features.astype(np.float32), targets.astype(np.float32))

    metadata = {
        "source_processed_dir": str(source_dir),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "split_sizes": {split: len(values[1]) for split, values in splits.items()},
        "fraud_counts": {split: int(values[1].sum()) for split, values in splits.items()},
        "leakage_excluded": ["ClaimStatus"],
        "contract": "Approch-1 frozen 29-feature leakage-excluded baseline",
    }
    if destination_dir is not None:
        (destination_dir / "contract_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    return splits


def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(targets.astype(np.int64), predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "threshold": 0.5,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start is not None and end is not None:
        start.record()
    else:
        import time
        wall_start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor)).cpu().numpy()
    if start is not None and end is not None:
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
    else:
        elapsed_ms = float((time.perf_counter() - wall_start) * 1000)
    return probabilities, elapsed_ms
