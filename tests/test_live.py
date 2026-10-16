from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from hull_timing.live import (
    Daily,
    align_end,
    build_daily,
    buy_and_hold,
    check_against_train,
    first_scored_day,
    parse_chart,
    rerun_cutoff,
)
from hull_timing.metric import sharpe_report


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.fixture
def market():
    days = _weekdays(date(2025, 6, 2), 300)
    rng = np.random.default_rng(0)
    prices = dict(zip(days, 100 * np.cumprod(1 + 4e-4 + 0.01 * rng.normal(size=300)), strict=True))
    tbill = {d: 3.5 + 0.01 * i for i, d in enumerate(days)}
    return prices, tbill


def test_parse_chart_skips_missing_values():
    ts = [int(datetime(2026, 1, d, 14, 30, tzinfo=timezone.utc).timestamp()) for d in (5, 6, 7)]
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": ts,
                    "indicators": {
                        "quote": [{"close": [1.0, None, 3.0]}],
                        "adjclose": [{"adjclose": [10.0, 20.0, None]}],
                    },
                }
            ]
        }
    }
    assert parse_chart(payload, "close") == {date(2026, 1, 5): 1.0, date(2026, 1, 7): 3.0}
    assert parse_chart(payload, "adjclose") == {date(2026, 1, 5): 10.0, date(2026, 1, 6): 20.0}


def test_build_daily_uses_next_close_and_same_day_rate(market):
    prices, tbill = market
    daily = build_daily(prices, tbill)
    days = sorted(prices)
    assert daily.dates == days[:-1]  # the last day has no next close yet
    assert daily.forward_returns[0] == pytest.approx(prices[days[1]] / prices[days[0]] - 1)
    assert daily.risk_free_rate[3] == pytest.approx(tbill[days[3]] / 100 / 252)


def test_alignment_and_train_check_recover_a_planted_tail(market):
    prices, tbill = market
    daily = build_daily(prices, tbill)
    end = 250  # train.csv stops at this day
    train = pd.DataFrame(
        {
            "forward_returns": daily.forward_returns[: end + 1],
            "risk_free_rate": daily.risk_free_rate[: end + 1],
        }
    )
    assert align_end(train["forward_returns"].to_numpy(), daily) == end
    chk = check_against_train(train, daily, n=200)
    assert chk["last_train_date"] == daily.dates[end]
    assert chk["forward_returns_exact"] == 1.0 and chk["risk_free_rate_exact"] == 1.0
    assert chk["public_bh_rebuilt"] == pytest.approx(chk["public_bh_train"])


def test_buy_and_hold_is_the_plain_sharpe_of_the_window(market):
    daily = build_daily(*market)
    first, last = daily.dates[10], daily.dates[60]
    n, score = buy_and_hold(daily, first, last)
    sl = slice(10, 61)
    rep = sharpe_report(np.ones(51), daily.forward_returns[sl], daily.risk_free_rate[sl])
    assert n == 51 and score == pytest.approx(rep.sharpe)
    with pytest.raises(ValueError):
        buy_and_hold(daily, last, first)


def test_scored_rows_start_after_the_deadline_and_reruns_lag_two_days():
    days = _weekdays(date(2025, 12, 10), 30)
    daily = Daily(days, np.zeros(30), np.zeros(30))
    assert first_scored_day(daily) == date(2025, 12, 16)  # deadline Mon 2025-12-15
    # a rerun on Wednesday scores rows up to Monday: Tuesday's return needs Wednesday's close
    assert rerun_cutoff(daily, date(2026, 1, 7)) == date(2026, 1, 5)
