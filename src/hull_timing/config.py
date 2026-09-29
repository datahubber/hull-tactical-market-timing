"""Run configuration: a flat dataclass, overridable from a YAML file of the same keys."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # features
    use_lagged_targets: bool = True
    lags: tuple[int, ...] = (1, 2, 3, 5)
    windows: tuple[int, ...] = (5, 20)
    max_missing: float = 0.3
    z_clip: float = 10.0
    # walk-forward split (rows are trading days)
    n_folds: int = 6
    test_size: int = 504
    val_size: int = 252
    cal_size: int = 252
    gap: int = 5
    # models
    models: tuple[str, ...] = ("logreg", "gru", "attn_gru", "mlp")
    seq_len: int = 20
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.3
    mlp_hidden: tuple[int, ...] = (256, 128, 64)
    logreg_c: float = 0.01
    # optimisation
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 6
    threads: int = 2
    # policy and evaluation
    percentiles: tuple[float, ...] = (25.0, 50.0, 75.0)
    leverages: tuple[float, ...] = (0.0, 0.6, 1.2, 1.8)
    tune_trials: int = 500
    null_draws: int = 200
    window_days: int = 126

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        known = {f.name: f for f in fields(cls)}
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        defaults = cls()
        kw = {}
        for k, v in d.items():
            kw[k] = tuple(v) if isinstance(getattr(defaults, k), tuple) else v
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
