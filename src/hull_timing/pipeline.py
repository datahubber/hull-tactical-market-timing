"""End-to-end walk-forward run: load data, train every fold, score, write result tables."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .evaluate import score_fold, window_scores
from .features import ID_COL, build_frame
from .policy import fit_percentile_thresholds, tier_positions
from .train import folds_from_config, run_fold


def load_train(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    return pd.read_csv(path)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean, std (ddof=1) and range of each (model, policy) across folds.

    Test-block policies are also compared with buy-and-hold on the same test block, paired by
    fold: `beats_bh` counts folds won, `diff_bh_*` is the per-fold score difference, and
    `diff_bh_t` = mean / (std / sqrt(n_folds)), a paired t statistic."""
    g = folds.groupby(["model", "policy"])["score"]
    out = g.agg(["mean", "std", "min", "max", "count"]).reset_index()
    bh = folds[(folds.model == "buy_hold") & (folds.policy == "1.0x")][["fold", "score"]]
    pairs = folds.merge(bh, on="fold", suffixes=("", "_bh"))
    pairs = pairs[~pairs.policy.isin(["1.0x", "1.0x_on_cal", "tuned_insample"])]
    keys = [pairs.model, pairs.policy]
    diff = (pairs.score - pairs.score_bh).groupby(keys)
    paired = pd.DataFrame(
        {
            "beats_bh": (pairs.score > pairs.score_bh).groupby(keys).sum(),
            "diff_bh_mean": diff.mean(),
            "diff_bh_std": diff.std(),
        }
    )
    paired["diff_bh_t"] = paired.diff_bh_mean / (paired.diff_bh_std / np.sqrt(diff.count()))
    return out.merge(paired.reset_index(), on=["model", "policy"], how="left")


def run(
    cfg: Config, data_dir: str | Path, out_dir: str | Path, log: Callable[[str], None] = print
) -> pd.DataFrame:
    torch.set_num_threads(cfg.threads)
    t_start = time.time()
    lags, windows = (cfg.lags, cfg.windows) if cfg.use_lagged_targets else ((), ())
    df = build_frame(load_train(data_dir), lags, windows)
    folds = folds_from_config(len(df), cfg)
    fwd = df["forward_returns"].to_numpy(dtype=float)
    rf = df["risk_free_rate"].to_numpy(dtype=float)
    dates = df[ID_COL].to_numpy()

    rows, seed_rows, fits, fold_meta = [], [], [], []
    oos_pos: dict[str, list[np.ndarray]] = {"buy_hold": []}
    for fold in folds:
        log(
            f"fold {fold.k}: train {dates[fold.train[0]]}-{dates[fold.train[-1]]}, "
            f"test {dates[fold.test[0]]}-{dates[fold.test[-1]]}"
        )
        res = run_fold(df, fold, cfg, log)
        r, s = score_fold(fold, res["probs"], fwd, rf, cfg)
        rows += r
        seed_rows += s
        fits += res["fits"]
        oos_pos["buy_hold"].append(np.ones(len(fold.test)))
        for name, p in res["probs"].items():
            thr = fit_percentile_thresholds(p["cal"], cfg.percentiles)
            oos_pos.setdefault(name, []).append(tier_positions(p["test"], thr, cfg.leverages))
        fold_meta.append(
            {
                "fold": fold.k,
                "n_features": res["n_features"],
                **{
                    f"{b}_date_ids": [
                        int(dates[getattr(fold, b)[0]]),
                        int(dates[getattr(fold, b)[-1]]),
                    ]
                    for b in ("train", "val", "cal", "test")
                },
            }
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds_df = pd.DataFrame(rows)
    folds_df.to_csv(out / "folds.csv", index=False, float_format="%.4f")
    pd.DataFrame(seed_rows).to_csv(out / "seeds.csv", index=False, float_format="%.4f")
    pd.DataFrame(fits).to_csv(out / "fits.csv", index=False, float_format="%.4f")
    summarize(folds_df).to_csv(out / "summary.csv", index=False, float_format="%.4f")

    test_idx = np.concatenate([f.test for f in folds])
    win = {
        name: window_scores(np.concatenate(p), fwd[test_idx], rf[test_idx], cfg.window_days)
        for name, p in oos_pos.items()
    }
    pd.DataFrame(win).to_csv(out / "windows.csv", index_label="window", float_format="%.4f")
    meta = {
        "config": cfg.to_dict(),
        "folds": fold_meta,
        "runtime_min": round((time.time() - t_start) / 60, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "device": "cpu",
        "threads": cfg.threads,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done in {meta['runtime_min']} min -> {out}")
    return folds_df
