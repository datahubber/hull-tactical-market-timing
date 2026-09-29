import numpy as np
import pandas as pd
import pytest

from hull_timing.evaluate import constant_rows, shift_null_pvalue, window_scores
from hull_timing.metric import adjusted_sharpe
from hull_timing.pipeline import summarize
from hull_timing.train import walk_forward_folds


def _returns(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return 3e-4 + 0.01 * rng.normal(size=n), np.full(n, 1e-4)


def test_shift_null_detects_a_perfect_signal():
    fwd, rf = _returns()
    oracle = np.where(fwd > 0, 1.2, 0.0)  # knows tomorrow's sign
    assert shift_null_pvalue(oracle, fwd, rf, draws=99, seed=0) == 0.01


def test_shift_null_is_not_small_for_a_random_policy():
    fwd, rf = _returns()
    pos = np.random.default_rng(1).choice([0.0, 0.6, 1.2, 1.8], size=len(fwd))
    p = shift_null_pvalue(pos, fwd, rf, draws=99, seed=0)
    assert 0.05 < p <= 1.0


def test_constant_leverage_baselines_score_the_test_block():
    fwd, rf = _returns()
    folds = walk_forward_folds(len(fwd), n_folds=2, test_size=100, val_size=50, cal_size=50, gap=2)
    fold = folds[1]
    rows = {r["policy"]: r for r in constant_rows(fold, fwd, rf)}
    assert set(rows) == {"1.0x", "1.2x"}
    te = fold.test
    assert rows["1.0x"]["score"] == adjusted_sharpe(np.ones(len(te)), fwd[te], rf[te])
    assert rows["1.2x"]["score"] == adjusted_sharpe(np.full(len(te), 1.2), fwd[te], rf[te])
    assert rows["1.2x"]["mean_pos"] == 1.2


def test_window_scores_use_non_overlapping_windows():
    fwd, rf = _returns(n=260)
    pos = np.ones_like(fwd)
    w = window_scores(pos, fwd, rf, window=100)
    assert w.shape == (2,)  # the last 60 rows do not fill a window
    assert w[1] == adjusted_sharpe(pos[100:200], fwd[100:200], rf[100:200])


def test_summary_pairs_each_policy_with_buy_and_hold_by_fold():
    bh = [0.5, 1.0, -0.2]
    model = [0.7, 0.9, 0.1]  # per-fold differences +0.2, -0.1, +0.3
    rows = [
        {"fold": k, "model": "buy_hold", "policy": "1.0x", "score": s} for k, s in enumerate(bh)
    ]
    rows += [
        {"fold": k, "model": "m", "policy": "tiers_cal", "score": s} for k, s in enumerate(model)
    ]
    rows += [{"fold": k, "model": "m", "policy": "tuned_insample", "score": 9.0} for k in range(3)]
    s = summarize(pd.DataFrame(rows)).set_index(["model", "policy"])
    r = s.loc[("m", "tiers_cal")]
    d = np.array(model) - np.array(bh)
    assert r["beats_bh"] == 2
    assert r["diff_bh_mean"] == pytest.approx(d.mean())
    assert r["diff_bh_t"] == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(3)))
    assert np.isnan(s.loc[("m", "tuned_insample"), "diff_bh_mean"])  # in-sample: no comparison
