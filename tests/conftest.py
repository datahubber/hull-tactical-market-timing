"""Tiny synthetic stand-in for train.csv: no network, no competition data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hull_timing.config import Config


def make_synthetic(n: int = 400, n_feat: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date_id": np.arange(n)})
    for j in range(n_feat):
        df[f"F{j}"] = rng.normal(size=n).cumsum() * 0.1
    df.loc[: n // 10, "F0"] = np.nan  # a late-starting feature, like the real data
    df["D1"] = rng.integers(0, 2, size=n)
    df["risk_free_rate"] = 1e-4 + 1e-5 * rng.normal(size=n)
    df["forward_returns"] = 3e-4 + 0.01 * rng.standard_t(5, size=n)
    df["market_forward_excess_returns"] = df["forward_returns"] - 3e-4
    return df


@pytest.fixture
def synthetic() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture
def tiny_cfg() -> Config:
    return Config(
        n_folds=2,
        test_size=60,
        val_size=40,
        cal_size=40,
        gap=2,
        seq_len=5,
        hidden=8,
        layers=1,
        mlp_hidden=(16,),
        seeds=(0,),
        batch_size=64,
        max_epochs=2,
        patience=2,
        threads=1,
        tune_trials=20,
        null_draws=10,
        window_days=30,
    )
