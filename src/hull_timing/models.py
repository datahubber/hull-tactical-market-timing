"""Small PyTorch up/down classifiers over a (batch, seq_len, n_features) window of past days.

All models return logits for P(forward_return > 0). The GRUs are bidirectional, as in the
Kaggle notebooks; that reads the window both ways but the window itself ends at day t, so it
does not see the future.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class GRUClassifier(nn.Module):
    """BiGRU encoder, then last-step pooling (`attention=False`) or additive attention pooling."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.3,
        attention: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        d = 2 * hidden
        self.attn = nn.Linear(d, 1) if attention else None
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)  # (B, L, 2H)
        if self.attn is None:
            z = h[:, -1]
        else:
            w = torch.softmax(self.attn(h), dim=1)  # (B, L, 1), weights over time steps
            z = (w * h).sum(dim=1)
        return self.head(z).squeeze(-1)


class MLPClassifier(nn.Module):
    """Feed-forward net on the flattened window."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features * seq_len
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)


NEURAL_MODELS = ("gru", "attn_gru", "mlp")


def build_model(
    kind: str,
    n_features: int,
    seq_len: int,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.3,
    mlp_hidden: Sequence[int] = (256, 128, 64),
) -> nn.Module:
    if kind == "gru":
        return GRUClassifier(n_features, hidden, layers, dropout, attention=False)
    if kind == "attn_gru":
        return GRUClassifier(n_features, hidden, layers, dropout, attention=True)
    if kind == "mlp":
        return MLPClassifier(n_features, seq_len, mlp_hidden, dropout)
    raise ValueError(f"unknown model kind {kind!r}; expected one of {NEURAL_MODELS}")
