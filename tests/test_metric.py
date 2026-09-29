import numpy as np
import pytest

from hull_timing.metric import adjusted_sharpe, sharpe_report


def _reference(pos, fwd, rf):
    """Direct transcription of the scoring rule with a plain product (no log-space trick)."""
    strat = rf * (1 - pos) + pos * fwd
    n = len(pos)
    s_mean = np.prod(1 + strat - rf) ** (1 / n) - 1
    m_mean = np.prod(1 + fwd - rf) ** (1 / n) - 1
    s_std, m_std = strat.std(ddof=1), fwd.std(ddof=1)
    sharpe = s_mean / s_std * np.sqrt(252)
    vol_pen = 1 + max(0, (s_std / m_std) - 1.2)
    gap = max(0, (m_mean - s_mean) * 100 * 252)
    return sharpe / (vol_pen * (1 + gap**2 / 100))


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    fwd = 4e-4 + 0.01 * rng.normal(size=500)
    rf = 1e-4 + 1e-5 * rng.normal(size=500)
    return fwd, rf


def test_matches_reference_on_random_positions(returns):
    fwd, rf = returns
    rng = np.random.default_rng(1)
    for _ in range(20):
        pos = rng.uniform(0, 2, size=len(fwd))
        assert adjusted_sharpe(pos, fwd, rf) == pytest.approx(_reference(pos, fwd, rf), rel=1e-9)


def test_buy_and_hold_has_no_penalty(returns):
    fwd, rf = returns
    r = sharpe_report(np.ones_like(fwd), fwd, rf)
    assert r.vol_penalty == 1.0 and r.return_penalty == 1.0
    assert r.score == pytest.approx(r.sharpe)


def test_double_leverage_pays_vol_penalty(returns):
    fwd, rf = returns
    r = sharpe_report(np.full_like(fwd, 2.0), fwd, rf)
    assert r.vol_penalty > 1.7  # strategy vol is about 2x market vol, allowance is 1.2x


def test_flat_position_scores_zero(returns):
    fwd, rf = returns
    assert adjusted_sharpe(np.zeros_like(fwd), fwd, rf) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bad", [-0.1, 2.1])
def test_out_of_range_positions_raise(returns, bad):
    fwd, rf = returns
    pos = np.ones_like(fwd)
    pos[3] = bad
    with pytest.raises(ValueError):
        adjusted_sharpe(pos, fwd, rf)
