"""Score fold predictions with the competition metric, plus the controls used in the README.

Per fold:
* `buy_hold` at constant leverage (`1.0x`, `1.2x`): no model. The score is a Sharpe ratio, so
  constant leverage up to 1.2x barely changes it.
Per fold and model:
* `tiers_cal`: the main policy (percentile tiers fitted on the cal block), scored on test.
Ensemble-only rows:
* `binary_0.49`: the ensemble notebook's fixed rule (p >= 0.49 -> 1.2x, else 0).
* `tuned_insample`: thresholds and leverages searched to maximise the score on the cal
  block, scored on that same block. This approximates how the notebooks' docstring figures
  ("Adj. Sharpe 4.24 / 4.62") were most likely obtained.
* `tuned_oos`: the same tuned rule applied to the untouched test block.
Plus a null control: circularly shifted copies of the policy's own positions, which keep the
leverage mix and autocorrelation but break the alignment with returns.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import Config
from .metric import adjusted_sharpe
from .policy import binary_positions, fit_percentile_thresholds, tier_positions, tune_tiers
from .train import Fold


def shift_null_pvalue(
    pos: np.ndarray, fwd: np.ndarray, rf: np.ndarray, draws: int, seed: int, min_shift: int = 21
) -> float:
    """One-sided p-value: share of circular shifts that score >= the real alignment."""
    rng = np.random.default_rng(seed)
    actual = adjusted_sharpe(pos, fwd, rf)
    shifts = rng.integers(min_shift, len(pos) - min_shift, size=draws)
    null = np.array([adjusted_sharpe(np.roll(pos, s), fwd, rf) for s in shifts])
    return float((1 + np.sum(null >= actual)) / (1 + draws))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


BASELINE_LEVERAGES = (1.0, 1.2)


def constant_rows(fold: Fold, fwd: np.ndarray, rf: np.ndarray) -> list[dict]:
    """Buy-and-hold at each constant leverage in BASELINE_LEVERAGES, scored on the test block."""
    te = fold.test
    return [
        {
            "fold": fold.k,
            "model": "buy_hold",
            "policy": f"{c:.1f}x",
            "score": adjusted_sharpe(np.full(len(te), c), fwd[te], rf[te]),
            "mean_pos": c,
        }
        for c in BASELINE_LEVERAGES
    ]


def score_fold(
    fold: Fold,
    probs: dict[str, dict[str, np.ndarray]],
    fwd: np.ndarray,
    rf: np.ndarray,
    cfg: Config,
) -> tuple[list[dict], list[dict]]:
    """Returns (rows for folds.csv, rows for seeds.csv)."""
    te, cal = fold.test, fold.cal
    y_te = (fwd[te] > 0).astype(float)
    rows = constant_rows(fold, fwd, rf)
    seed_rows = []
    for name, p in probs.items():
        thr = fit_percentile_thresholds(p["cal"], cfg.percentiles)
        pos = tier_positions(p["test"], thr, cfg.leverages)
        row = {
            "fold": fold.k,
            "model": name,
            "policy": "tiers_cal",
            "score": adjusted_sharpe(pos, fwd[te], rf[te]),
            "mean_pos": float(pos.mean()),
            "auc": _auc(y_te, p["test"]),
        }
        if name == "ensemble":
            row["null_p"] = shift_null_pvalue(pos, fwd[te], rf[te], cfg.null_draws, seed=fold.k)
        rows.append(row)
        for i, (pc, pt) in enumerate(
            zip(p.get("cal_by_seed", []), p.get("test_by_seed", []), strict=True)
        ):
            ps = tier_positions(pt, fit_percentile_thresholds(pc, cfg.percentiles), cfg.leverages)
            seed_rows.append(
                {
                    "fold": fold.k,
                    "model": name,
                    "seed": cfg.seeds[i],
                    "score": adjusted_sharpe(ps, fwd[te], rf[te]),
                }
            )

    if "ensemble" in probs:
        p = probs["ensemble"]
        pos = binary_positions(p["test"])
        rows.append(
            {
                "fold": fold.k,
                "model": "ensemble",
                "policy": "binary_0.49",
                "score": adjusted_sharpe(pos, fwd[te], rf[te]),
                "mean_pos": float(pos.mean()),
            }
        )
        thr, lev, s_in = tune_tiers(p["cal"], fwd[cal], rf[cal], cfg.tune_trials, seed=fold.k)
        pos_cal = tier_positions(p["cal"], thr, lev)
        pos_te = tier_positions(p["test"], thr, lev)
        rows.append(
            {
                "fold": fold.k,
                "model": "ensemble",
                "policy": "tuned_insample",
                "score": s_in,
                "mean_pos": float(pos_cal.mean()),
            }
        )
        rows.append(
            {
                "fold": fold.k,
                "model": "ensemble",
                "policy": "tuned_oos",
                "score": adjusted_sharpe(pos_te, fwd[te], rf[te]),
                "mean_pos": float(pos_te.mean()),
            }
        )
        rows.append(
            {
                "fold": fold.k,
                "model": "buy_hold",
                "policy": "1.0x_on_cal",
                "score": adjusted_sharpe(np.ones(len(cal)), fwd[cal], rf[cal]),
                "mean_pos": 1.0,
            }
        )
    return rows, seed_rows


def window_scores(pos: np.ndarray, fwd: np.ndarray, rf: np.ndarray, window: int) -> np.ndarray:
    """Scores on consecutive non-overlapping windows (126 trading days is about six months)."""
    n = len(pos) // window
    return np.array(
        [
            adjusted_sharpe(
                pos[i * window : (i + 1) * window],
                fwd[i * window : (i + 1) * window],
                rf[i * window : (i + 1) * window],
            )
            for i in range(n)
        ]
    )
