"""Map an up-probability to a daily S&P 500 position in [0, 2].

The main policy is the 4-tier rule from the Kaggle notebook: thresholds at the 25th, 50th
and 75th percentiles of the model's probabilities on a calibration block, mapped to
0 / 0.6 / 1.2 / 1.8x. In this repo the calibration block is disjoint from both the data used
for early stopping and the test block; the notebook used its single validation split for
early stopping and for the percentiles.

`tune_tiers` searches thresholds and leverages to maximise the score on the rows it is then
scored on. It approximates how the notebooks' optimised parameters (a fixed 0.49 / 1.2x rule;
an Optuna-tuned 4-tier variant) were most likely chosen, so the in-sample optimism of that
procedure can be measured.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metric import MAX_POSITION, MIN_POSITION, adjusted_sharpe


def fit_percentile_thresholds(
    p_cal: np.ndarray, percentiles: Sequence[float] = (25, 50, 75)
) -> np.ndarray:
    """Thresholds = percentiles of calibration-block probabilities (labels are not used)."""
    return np.percentile(np.asarray(p_cal, dtype=float), list(percentiles))


def tier_positions(p: np.ndarray, thresholds: np.ndarray, leverages: Sequence[float]) -> np.ndarray:
    """Tier i is used when p >= thresholds[i-1]; tier 0 when p is below every threshold."""
    thr = np.sort(np.asarray(thresholds, dtype=float))
    lev = np.asarray(leverages, dtype=float)
    if len(lev) != len(thr) + 1:
        raise ValueError("need exactly one more leverage than thresholds")
    tier = np.searchsorted(thr, np.asarray(p, dtype=float), side="right")
    return np.clip(lev[tier], MIN_POSITION, MAX_POSITION)


def binary_positions(p: np.ndarray, threshold: float = 0.49, leverage: float = 1.2) -> np.ndarray:
    """The ensemble notebook's rule: `leverage` when p >= threshold, else flat."""
    return tier_positions(p, np.array([threshold]), [0.0, leverage])


def tune_tiers(
    p: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    n_trials: int = 500,
    n_tiers: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Random search of thresholds (as quantiles of p) and leverages that maximises the score
    on the same rows it is scored on. Returns (thresholds, leverages, in-sample score)."""
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] = (np.array([]), np.array([1.0]), -np.inf)
    for _ in range(n_trials):
        thr = np.quantile(p, np.sort(rng.uniform(0, 1, n_tiers - 1)))
        lev = np.sort(rng.uniform(MIN_POSITION, MAX_POSITION, n_tiers))
        pos = tier_positions(p, thr, lev)
        if np.ptp(pos) == 0 and pos[0] == 0:
            continue
        try:
            s = adjusted_sharpe(pos, forward_returns, risk_free_rate)
        except ValueError:
            continue
        if s > best[2]:
            best = (thr, lev, s)
    return best
