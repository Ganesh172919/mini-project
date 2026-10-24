"""Approach 2 additional DL architectures: TabTransformer and Autoencoder.

TabTransformer (DL-04): linear feature projection + multi-head self-attention + MLP head.
Autoencoder (DL-05): reconstruction-error anomaly detector.
  NOTE: Autoencoder scores are NOT calibrated fraud probabilities. A threshold
  is fitted on validation reconstruction errors; scores above threshold are
  flagged. The raw score returned is the normalised reconstruction error.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class TabTransformer(nn.Module):
    """Lightweight TabTransformer-inspired architecture for tabular fraud detection.

    All 29 input features are projected individually to an embedding dim, then
    processed by 2 multi-head self-attention layers, then flattened and passed
    through a 2-layer MLP classification head.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Per-feature linear projection → (batch, input_dim, embed_dim)
        self.feature_embed = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) → (batch, input_dim, 1) → embed → (batch, input_dim, embed_dim)
        embedded = self.feature_embed(x.unsqueeze(-1))
        attended = self.transformer(embedded)  # (batch, input_dim, embed_dim)
        return self.head(attended).squeeze(-1)


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection via reconstruction error.

    The encoder compresses input to a latent vector; the decoder reconstructs
    the input. During inference, the mean squared reconstruction error is used
    as an anomaly score. A decision threshold is fitted on validation data.

    IMPORTANT: The reconstruction error is NOT a calibrated fraud probability.
    Use score_is_calibrated=False and present as anomaly_score in the API.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.input_dim = input_dim
        # Fitted threshold — set via fit_threshold() on validation data
        self.threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample mean reconstruction error (scalar per sample)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return ((x - x_hat) ** 2).mean(dim=1)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — returns raw MSE reconstruction error."""
        return self.forward(x)

    def fit_threshold(self, errors: torch.Tensor, percentile: float = 95.0) -> float:
        """Set decision threshold as the p-th percentile of val reconstruction errors."""
        self.threshold = float(torch.quantile(errors, percentile / 100.0).item())
        return self.threshold

    def predict_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Return a pseudo-logit for BCEWithLogitsLoss training.

        The autoencoder is trained to minimise reconstruction error on ALL
        samples. We then scale the error to a logit for the loss calculation
        by mapping threshold → 0.
        """
        errors = self.forward(x)
        # Normalise: positive logit when error > threshold
        return (errors - self.threshold) * 10.0
