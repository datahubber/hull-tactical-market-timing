"""Buy-and-hold on the competition's live (private-leaderboard) window, rebuilt from public data.

The live returns were never released, so this module reconstructs them and checks the
reconstruction against train.csv first:

* ``forward_returns`` = next-day return of SPY's dividend-adjusted close
  (buy at today's close, sell at tomorrow's close);
* ``risk_free_rate`` = 13-week T-bill yield (Yahoo ``^IRX``, percent) / 100 / 252, same day.

Scored rows are trading days after the final submission deadline (2025-12-15) up to the last
data update before the final rerun (2026-06-25). The host did not publish that last date, so
the report scores a range of cutoffs. Holding 1.0x every day triggers neither penalty, so the
score is the plain Sharpe ratio of the window.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .metric import sharpe_report

SUBMISSION_DEADLINE = date(2025, 12, 15)
FINAL_RERUN = date(2026, 6, 25)
# Leaderboard reruns during the forecasting phase and my ensemble's score at each one
# (notebook aa1fc409de, as listed in my Kaggle submission list).
RERUNS = {
    date(2026, 1, 21): 3.589,
    date(2026, 2, 19): 1.688,
    date(2026, 3, 23): -0.403,
    date(2026, 4, 20): 0.753,
    date(2026, 5, 21): 1.378,
    date(2026, 6, 25): 1.699,
}
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"


@dataclass(frozen=True)
class Daily:
    """One row per trading day d: return from d's close to the next close, and d's rate."""

    dates: list[date]
    forward_returns: np.ndarray
    risk_free_rate: np.ndarray


def parse_chart(payload: dict, field: str) -> dict[date, float]:
    """Yahoo chart JSON -> {trading date: value}; field is 'adjclose' or 'close'."""
    res = payload["chart"]["result"][0]
    ind = res["indicators"]
    values = ind["adjclose"][0]["adjclose"] if field == "adjclose" else ind["quote"][0][field]
    out = {}
    for ts, v in zip(res["timestamp"], values, strict=True):
        if v is not None:
            out[datetime.fromtimestamp(ts, tz=timezone.utc).date()] = float(v)
    return out


def fetch_chart(symbol: str, start: date, end: date, field: str) -> dict[date, float]:
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp())
    url = YAHOO.format(symbol=symbol.replace("^", "%5E"), p1=p1, p2=p2)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return parse_chart(json.load(r), field)


def build_daily(prices: dict[date, float], tbill_pct: dict[date, float]) -> Daily:
    """Forward return of each trading day from consecutive prices; rate = T-bill % / 100 / 252."""
    days = sorted(prices)
    px = np.array([prices[d] for d in days])
    fwd = px[1:] / px[:-1] - 1
    keep = [i for i, d in enumerate(days[:-1]) if d in tbill_pct]
    return Daily(
        dates=[days[i] for i in keep],
        forward_returns=fwd[keep],
        risk_free_rate=np.array([tbill_pct[days[i]] / 100 / 252 for i in keep]),
    )


def align_end(train_fwd: np.ndarray, daily: Daily, n: int = 60) -> int:
    """Index into daily.dates of the day matching the last train row (least mean abs error)."""
    tail = train_fwd[-n:]
    errs = [
        np.abs(daily.forward_returns[k - n + 1 : k + 1] - tail).mean()
        for k in range(n - 1, len(daily.dates))
    ]
    return int(np.argmin(errs)) + n - 1


def check_against_train(train: pd.DataFrame, daily: Daily, n: int = 500) -> dict:
    """Map the last n train rows to dates by exact return match and score the reconstruction."""
    fwd = train["forward_returns"].to_numpy()
    rf = train["risk_free_rate"].to_numpy()
    end = align_end(fwd, daily)
    matched, rf_err, j = 0, [], end
    for i in range(len(fwd) - 1, len(fwd) - 1 - n, -1):
        for k in range(j, max(j - 5, -1), -1):  # a missing day in either series shifts by one
            if abs(daily.forward_returns[k] - fwd[i]) < 1e-6:
                matched += 1
                rf_err.append(abs(daily.risk_free_rate[k] - rf[i]))
                j = k - 1
                break
        else:
            j -= 1
    last = slice(len(fwd) - 180, len(fwd))  # the training phase's public test set
    ones = np.ones(180)
    rebuilt = slice(end - 179, end + 1)
    return {
        "last_train_date": daily.dates[end],
        "rows_checked": n,
        "forward_returns_exact": matched / n,
        "risk_free_rate_exact": float(np.mean(np.array(rf_err) < 1e-9)),
        "public_bh_train": sharpe_report(ones, fwd[last], rf[last]).score,
        "public_bh_rebuilt": sharpe_report(
            ones, daily.forward_returns[rebuilt], daily.risk_free_rate[rebuilt]
        ).score,
    }


def buy_and_hold(daily: Daily, first: date, last: date) -> tuple[int, float]:
    """(rows, score) of 1.0x every day over trading days first..last inclusive."""
    idx = [i for i, d in enumerate(daily.dates) if first <= d <= last]
    if len(idx) < 2:
        raise ValueError(f"fewer than 2 trading days between {first} and {last}")
    f, r = daily.forward_returns[idx], daily.risk_free_rate[idx]
    return len(idx), sharpe_report(np.ones(len(idx)), f, r).score


def rerun_cutoff(daily: Daily, rerun: date) -> date:
    """Last row whose forward return was known the day before a rerun (2 trading days back)."""
    prior = [d for d in daily.dates if d < rerun]
    return prior[-2]


def first_scored_day(daily: Daily) -> date:
    return next(d for d in daily.dates if d > SUBMISSION_DEADLINE)


def report(train: pd.DataFrame, daily: Daily, cutoffs: list[date]) -> str:
    chk = check_against_train(train, daily)
    start = first_scored_day(daily)
    lines = [
        "# Buy-and-hold on the live window (reconstructed)",
        "",
        "Generated by `hull-timing live-baseline` from SPY adjusted closes and the 13-week "
        "T-bill yield (Yahoo `^IRX`).",
        "",
        "## Reconstruction check against train.csv",
        "",
        f"- Last train row matches {chk['last_train_date']}.",
        f"- forward_returns reproduced to 1e-6 on {chk['forward_returns_exact']:.0%} of the "
        f"last {chk['rows_checked']} rows; risk_free_rate reproduced to 1e-9 on "
        f"{chk['risk_free_rate_exact']:.0%} of the matched rows.",
        f"- Buy-and-hold on the 180-row public set: {chk['public_bh_train']:.3f} from train.csv, "
        f"{chk['public_bh_rebuilt']:.3f} from the reconstruction.",
        "",
        f"## Final window: from {start} to the cutoff",
        "",
        "| Last scored row | Rows | Buy-and-hold | My ensemble 1.699 as a multiple |",
        "|---|--:|--:|--:|",
    ]
    for c in cutoffs:
        n, s = buy_and_hold(daily, start, c)
        lines.append(f"| {c} | {n} | {s:.3f} | {RERUNS[FINAL_RERUN] / s:.1f}x |")
    lines += [
        "",
        "## Leaderboard reruns (rows up to two trading days before each rerun)",
        "",
        "| Rerun | Last scored row | Rows | Buy-and-hold | My ensemble |",
        "|---|---|--:|--:|--:|",
    ]
    for rerun, mine in RERUNS.items():
        c = rerun_cutoff(daily, rerun)
        n, s = buy_and_hold(daily, start, c)
        lines.append(f"| {rerun} | {c} | {n} | {s:.3f} | {mine:.3f} |")
    return "\n".join(lines) + "\n"


def run(data: Path, out: Path) -> str:
    train = pd.read_csv(data / "train.csv", usecols=["forward_returns", "risk_free_rate"])
    begin, end = date(2015, 1, 1), date(2026, 7, 15)
    daily = build_daily(
        fetch_chart("SPY", begin, end, "adjclose"), fetch_chart("^IRX", begin, end, "close")
    )
    june = [d for d in daily.dates if date(2026, 6, 10) <= d < FINAL_RERUN]
    text = report(train, daily, june)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
