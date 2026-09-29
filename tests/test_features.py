import numpy as np
import pandas as pd
import pytest

from hull_timing.features import (
    TARGET_COLS,
    Standardizer,
    build_frame,
    candidate_columns,
    make_windows,
    select_features,
)


def _perturb_future(df: pd.DataFrame, t: int, seed: int = 1) -> pd.DataFrame:
    """Change everything not yet known on day t: raw features after t, targets from t on."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    raw = [c for c in df.columns if c not in ("date_id", *TARGET_COLS)]
    for c in raw:
        n = len(df) - t - 1
        is_int = pd.api.types.is_integer_dtype(df[c])
        out.loc[t + 1 :, c] = rng.integers(0, 2, n) if is_int else rng.normal(size=n)
    out.loc[t:, list(TARGET_COLS)] = rng.normal(size=(len(df) - t, len(TARGET_COLS)))
    return out


@pytest.mark.parametrize("t", [30, 150, 398])
def test_no_lookahead(synthetic, t):
    a = build_frame(synthetic)
    b = build_frame(_perturb_future(synthetic, t))
    cols = candidate_columns(a)
    pd.testing.assert_frame_equal(a.loc[:t, cols], b.loc[:t, cols])


def test_lag_features_differ_once_target_is_known(synthetic):
    # Sanity check of the test above: the day-t target *does* reach row t+1.
    t = 150
    a = build_frame(synthetic)
    b = build_frame(_perturb_future(synthetic, t))
    assert a.loc[t + 1, "lag1_forward_returns"] != b.loc[t + 1, "lag1_forward_returns"]


def test_lag1_is_previous_day_target(synthetic):
    f = build_frame(synthetic)
    np.testing.assert_allclose(
        f["lag1_forward_returns"].to_numpy()[1:], synthetic["forward_returns"].to_numpy()[:-1]
    )


def test_targets_and_test_only_columns_are_never_inputs(synthetic):
    df = build_frame(synthetic)
    df["is_scored"] = 1
    df["lagged_forward_returns"] = 0.0
    cols = candidate_columns(df)
    assert not set(cols) & {"date_id", "is_scored", "lagged_forward_returns", *TARGET_COLS}


def test_lag_zero_is_rejected(synthetic):
    with pytest.raises(ValueError):
        build_frame(synthetic, lags=(0, 1))


def test_no_lag_ablation_adds_nothing(synthetic):
    f = build_frame(synthetic, lags=(), windows=())
    assert not [c for c in f.columns if c.startswith(("lag", "mean", "std"))]


def test_select_features_uses_train_rows_only(synthetic):
    df = build_frame(synthetic)
    df.loc[300:, "F1"] = np.nan  # missing only outside the training rows
    cols = select_features(df.iloc[:250], candidate_columns(df), max_missing=0.3)
    assert "F1" in cols


def test_select_features_drops_columns_constant_in_train_rows(synthetic):
    df = build_frame(synthetic)
    df.loc[:249, "F2"] = 0.25  # constant while training, varies afterwards
    assert "F2" not in select_features(df.iloc[:250], candidate_columns(df), max_missing=0.3)
    assert "F2" in select_features(df, candidate_columns(df), max_missing=0.3)


def test_standardizer_fits_on_given_rows():
    x = np.array([[1.0], [3.0], [np.nan], [100.0]])
    z = Standardizer(clip=None).fit(x[:2]).transform(x)
    np.testing.assert_allclose(z[:, 0], [-1.0, 1.0, 0.0, 98.0])


def test_standardizer_clips_and_survives_zero_std():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [100.0, 6.0]])
    z = Standardizer(clip=10.0).fit(x[:2]).transform(x)
    np.testing.assert_allclose(z[:, 0], [-1.0, 1.0, 10.0])
    np.testing.assert_allclose(z[:, 1], [0.0, 0.0, 1.0])  # std 0 -> scale 1, no inf


def test_windows_end_at_t_and_never_include_later_rows():
    x = np.arange(40, dtype=float).reshape(20, 2)
    w = make_windows(x, np.array([4, 10, 19]), seq_len=5)
    assert w.shape == (3, 5, 2)
    np.testing.assert_array_equal(w[1], x[6:11])
    assert w[1, -1, 0] == x[10, 0]
    with pytest.raises(ValueError):
        make_windows(x, np.array([3]), seq_len=5)
