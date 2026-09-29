"""Leak-safe features for the Hull Tactical daily S&P 500 data.

Timing convention (competition data description): row `t` holds features available on day
`t` and the target `forward_returns[t]`, the return from holding the index from day t to
day t+1. That target is only known on day t+1, which is why the live test feed supplies it
as `lagged_forward_returns`. Every feature built here for row t therefore uses the three
target columns from rows <= t-1 only, and raw features from rows <= t only.

`tests/test_features.py` checks this by perturbing the future and asserting that past
feature rows do not change.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ID_COL = "date_id"
TARGET_COLS = ("forward_returns", "risk_free_rate", "market_forward_excess_returns")
RETURN_COLS = ("forward_returns", "market_forward_excess_returns")
# Columns that appear only in the competition's test feed; never used as raw features.
TEST_ONLY_COLS = (
    "is_scored",
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
)


def add_lagged_targets(
    df: pd.DataFrame, lags: Sequence[int] = (1, 2, 3, 5), windows: Sequence[int] = (5, 20)
) -> pd.DataFrame:
    """Append lags and trailing mean/std of past targets. All use shift >= 1.

    Empty `lags` and `windows` add nothing (the no-lag ablation)."""
    if min(lags, default=1) < 1:
        raise ValueError("lags must be >= 1; lag 0 is the label itself")
    new: dict[str, pd.Series] = {}
    if lags:
        new["lag1_risk_free_rate"] = df["risk_free_rate"].shift(1)
    for col in RETURN_COLS:
        past = df[col].shift(1)
        for k in lags:
            new[f"lag{k}_{col}"] = df[col].shift(k)
        for w in windows:
            roll = past.rolling(w, min_periods=w)
            new[f"mean{w}_{col}"] = roll.mean()
            new[f"std{w}_{col}"] = roll.std()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def candidate_columns(df: pd.DataFrame) -> list[str]:
    """Every column that may be used as a model input."""
    banned = {ID_COL, *TARGET_COLS, *TEST_ONLY_COLS}
    return sorted(c for c in df.columns if c not in banned)


def select_features(train: pd.DataFrame, columns: Sequence[str], max_missing: float) -> list[str]:
    """Keep columns that, *in the training rows*, are missing at most `max_missing` of the
    time and take more than one value. (In the real data one column is constant over the
    first fold's training rows; its std there is rounding error, below 1e-16, so its z-score
    on later rows exceeded 1e16.)"""
    cols = list(columns)
    missing = train[cols].isna().mean()
    varies = train[cols].nunique(dropna=True) > 1
    return sorted(missing.index[(missing <= max_missing) & varies])


class Standardizer:
    """Mean/std scaling fitted on training rows only; NaN (after ffill) becomes 0 = train mean.

    `clip` bounds |z| so that a regime the training rows never saw (|z| of 30-50 occurs in
    the real data) cannot push an unbounded model such as the MLP far off its training range."""

    def __init__(self, clip: float | None = 10.0) -> None:
        self.clip = clip

    def fit(self, x: np.ndarray) -> Standardizer:
        self.mean_ = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        self.std_ = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.std_
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)


def build_frame(
    raw: pd.DataFrame, lags: Sequence[int] = (1, 2, 3, 5), windows: Sequence[int] = (5, 20)
) -> pd.DataFrame:
    """Sort by date, add lagged-target features and forward-fill gaps (a causal fill)."""
    df = raw.sort_values(ID_COL).reset_index(drop=True)
    df = add_lagged_targets(df, lags, windows)
    cols = candidate_columns(df)
    df[cols] = df[cols].ffill()
    return df


def make_windows(x: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    """Return x[t-seq_len+1 : t+1] for every t in `ends`, shape (len(ends), seq_len, n_feat)."""
    ends = np.asarray(ends)
    if ends.size and ends.min() < seq_len - 1:
        raise ValueError("every window end must be >= seq_len - 1")
    view = sliding_window_view(x, seq_len, axis=0)  # (n - L + 1, n_feat, L)
    return np.ascontiguousarray(view[ends - seq_len + 1].transpose(0, 2, 1))
