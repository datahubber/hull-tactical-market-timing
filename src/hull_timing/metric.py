"""Volatility-adjusted Sharpe ratio used to score the Hull Tactical competition.

Follows the host's public scoring code (Kaggle metric `hull-competition-sharpe`), including its
cap of 1,000,000 and its zero-volatility errors, and matches it to 3e-13 on 300 random windows
of train.csv. A strategy holds `position` in [0, 2] of the S&P 500 each day and the rest in the
risk-free asset:

    strategy_return = rf * (1 - position) + position * forward_return

The score is the annualised Sharpe of the strategy's excess return (geometric mean excess
return / std of strategy returns * sqrt(252)), divided by two penalties:

* volatility: 1 + max(0, strategy_vol / market_vol - 1.2)
* return gap: 1 + max(0, 100 * 252 * (market_mean_excess - strategy_mean_excess))**2 / 100
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252
MIN_POSITION = 0.0
MAX_POSITION = 2.0


@dataclass(frozen=True)
class SharpeReport:
    score: float
    sharpe: float
    vol_penalty: float
    return_penalty: float
    strategy_vol: float  # annualised, in percent
    market_vol: float  # annualised, in percent


def _geometric_mean(excess: np.ndarray) -> float:
    # (prod(1 + r)) ** (1 / n) - 1, computed in log space to avoid overflow on long series.
    return float(np.expm1(np.log1p(excess).mean()))


def sharpe_report(
    position: np.ndarray, forward_returns: np.ndarray, risk_free_rate: np.ndarray
) -> SharpeReport:
    pos = np.asarray(position, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)
    rf = np.asarray(risk_free_rate, dtype=float)
    if not (pos.shape == fwd.shape == rf.shape) or pos.ndim != 1 or len(pos) < 2:
        raise ValueError("position, forward_returns, risk_free_rate must be equal-length 1-D")
    if pos.max() > MAX_POSITION or pos.min() < MIN_POSITION:
        raise ValueError(f"positions must lie in [{MIN_POSITION}, {MAX_POSITION}]")

    strat = rf * (1.0 - pos) + pos * fwd
    strat_std = float(np.std(strat, ddof=1))
    mkt_std = float(np.std(fwd, ddof=1))
    if strat_std == 0.0 or mkt_std == 0.0:
        raise ValueError("zero volatility: the score is undefined")

    strat_mean = _geometric_mean(strat - rf)
    mkt_mean = _geometric_mean(fwd - rf)
    ann = np.sqrt(TRADING_DAYS)
    sharpe = strat_mean / strat_std * ann
    strat_vol = strat_std * ann * 100
    mkt_vol = mkt_std * ann * 100

    vol_penalty = 1.0 + max(0.0, strat_vol / mkt_vol - 1.2)
    gap = max(0.0, (mkt_mean - strat_mean) * 100 * TRADING_DAYS)
    return_penalty = 1.0 + gap**2 / 100
    score = min(sharpe / (vol_penalty * return_penalty), 1_000_000.0)
    return SharpeReport(score, sharpe, vol_penalty, return_penalty, strat_vol, mkt_vol)


def adjusted_sharpe(
    position: np.ndarray, forward_returns: np.ndarray, risk_free_rate: np.ndarray
) -> float:
    """Competition score for a daily position series (higher is better)."""
    return sharpe_report(position, forward_returns, risk_free_rate).score
