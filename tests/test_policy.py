import numpy as np
import pytest

from hull_timing.metric import adjusted_sharpe
from hull_timing.policy import (
    binary_positions,
    fit_percentile_thresholds,
    tier_positions,
    tune_tiers,
)

LEV = (0.0, 0.6, 1.2, 1.8)


def test_percentile_tiers_split_calibration_block_into_quarters():
    p = np.random.default_rng(0).uniform(0.4, 0.6, size=4000)
    pos = tier_positions(p, fit_percentile_thresholds(p), LEV)
    shares = [np.mean(pos == lev) for lev in LEV]
    np.testing.assert_allclose(shares, 0.25, atol=0.01)


def test_tier_boundaries_and_clipping():
    thr = np.array([0.4, 0.5, 0.6])
    pos = tier_positions(np.array([0.1, 0.4, 0.55, 0.6, 0.9]), thr, LEV)
    np.testing.assert_allclose(pos, [0.0, 0.6, 1.2, 1.8, 1.8])
    assert tier_positions(np.array([0.9]), np.array([0.5]), [0.0, 5.0])[0] == 2.0
    with pytest.raises(ValueError):
        tier_positions(np.array([0.5]), thr, (0.0, 1.0))


def test_binary_rule_from_ensemble_notebook():
    np.testing.assert_allclose(binary_positions(np.array([0.48, 0.49, 0.7])), [0.0, 1.2, 1.2])


def test_tuning_on_noise_is_optimistic_in_sample():
    """Tuning on pure noise wins in-sample and does not carry over to fresh noise."""
    rng = np.random.default_rng(3)
    n = 500
    ins, oos = [], []
    for _ in range(10):
        p, fwd = rng.uniform(size=n), 3e-4 + 0.01 * rng.normal(size=n)
        rf = np.full(n, 1e-4) + 1e-6 * rng.normal(size=n)
        thr, lev, score = tune_tiers(p, fwd, rf, n_trials=100, seed=0)
        p2, fwd2 = rng.uniform(size=n), 3e-4 + 0.01 * rng.normal(size=n)
        ins.append(score - adjusted_sharpe(np.ones(n), fwd, rf))
        oos.append(
            adjusted_sharpe(tier_positions(p2, thr, lev), fwd2, rf)
            - adjusted_sharpe(np.ones(n), fwd2, rf)
        )
    assert np.mean(ins) > 0.3
    assert np.mean(ins) > np.mean(oos) + 0.3
