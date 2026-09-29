import numpy as np
import pandas as pd
import pytest
import torch

from hull_timing.cli import main
from hull_timing.features import build_frame
from hull_timing.train import folds_from_config, run_fold, walk_forward_folds


def test_folds_are_ordered_disjoint_and_purged():
    folds = walk_forward_folds(2000, n_folds=4, test_size=100, val_size=80, cal_size=60, gap=3)
    for f in folds:
        blocks = [f.train, f.val, f.cal, f.test]
        for a, b in zip(blocks, blocks[1:], strict=False):
            assert b[0] - a[-1] == 4  # a gap of 3 rows between consecutive blocks
    assert folds[-1].test[-1] == 1999
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert nxt.test[0] == prev.test[-1] + 1  # test blocks tile the end of the series
        assert nxt.train[0] == prev.train[0] and len(nxt.train) == len(prev.train) + 100


def test_too_little_training_data_raises():
    with pytest.raises(ValueError):
        walk_forward_folds(300, n_folds=3, test_size=80, val_size=40, cal_size=40, gap=1)


def test_run_fold_outputs(synthetic, tiny_cfg):
    torch.set_num_threads(1)
    df = build_frame(synthetic)
    fold = folds_from_config(len(df), tiny_cfg)[0]
    res = run_fold(df, fold, tiny_cfg, log=lambda *_: None)
    assert set(res["probs"]) == {"logreg", "gru", "attn_gru", "mlp", "ensemble"}
    for p in res["probs"].values():
        assert p["test"].shape == (len(fold.test),)
        assert ((p["test"] > 0) & (p["test"] < 1)).all()


def test_test_block_predictions_do_not_use_later_rows(synthetic, tiny_cfg):
    """Perturbing rows after day t must leave every prediction up to day t unchanged."""
    torch.set_num_threads(1)
    df = build_frame(synthetic)
    fold = folds_from_config(len(df), tiny_cfg)[0]
    t = int(fold.test[20])
    later = synthetic.copy()
    later.loc[t + 1 :, ["F1", "F2", "D1"]] += 5.0
    later.loc[t:, "forward_returns"] = 0.05
    a = run_fold(df, fold, tiny_cfg, log=lambda *_: None)["probs"]
    b = run_fold(build_frame(later), fold, tiny_cfg, log=lambda *_: None)["probs"]
    for name in a:
        np.testing.assert_allclose(a[name]["test"][:21], b[name]["test"][:21], rtol=1e-6)
        assert not np.allclose(a[name]["test"][22:], b[name]["test"][22:])


def test_cli_end_to_end(tmp_path, synthetic, tiny_cfg):
    data = tmp_path / "data"
    data.mkdir()
    synthetic.to_csv(data / "train.csv", index=False)
    cfg_path = tmp_path / "tiny.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(tiny_cfg.to_dict()))
    main(["run", "--config", str(cfg_path), "--data", str(data), "--out", str(tmp_path / "res")])
    summary = pd.read_csv(tmp_path / "res" / "summary.csv")
    assert {"buy_hold", "logreg", "ensemble"} <= set(summary.model)
    assert {"1.0x", "1.2x"} <= set(summary[summary.model == "buy_hold"].policy)
    main(["plot", "--results", str(tmp_path / "res"), "--out", str(tmp_path / "fig.png")])
    assert (tmp_path / "fig.png").stat().st_size > 10_000
