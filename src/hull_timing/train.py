"""Walk-forward training: expanding-window folds with purged train | val | cal | test blocks.

For each fold (rows are trading days, oldest first):

    [ train ......... ] gap [ val ] gap [ cal ] gap [ test ]

* train: model fitting, feature selection and scaler statistics.
* val:   early stopping for the neural nets.
* cal:   percentile thresholds for the position policy (never seen by the optimiser).
* test:  scored once. Each later fold moves every block forward by `test_size` rows, and
         the train block keeps its start (an expanding window).

The label is 1-day ahead, so a gap of 1 row is already enough to stop label overlap; the
default gap of 5 rows adds a margin for autocorrelated features.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from .config import Config
from .features import Standardizer, candidate_columns, make_windows, select_features
from .models import NEURAL_MODELS, build_model


@dataclass(frozen=True)
class Fold:
    k: int
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray


class FoldResult(TypedDict):
    """`probs[model]` maps "cal" / "test" to seed-averaged probabilities on that block; neural
    models also carry "cal_by_seed" / "test_by_seed" arrays of shape (n_seeds, block_len)."""

    probs: dict[str, dict[str, np.ndarray]]
    fits: list[dict[str, str | float]]
    n_features: int


def walk_forward_folds(
    n_rows: int,
    n_folds: int,
    test_size: int,
    val_size: int,
    cal_size: int,
    gap: int,
    first_row: int = 0,
    min_train: int = 50,
) -> list[Fold]:
    """The last `n_folds * test_size` rows become consecutive test blocks."""
    folds = []
    first_test = n_rows - n_folds * test_size
    for k in range(n_folds):
        t0 = first_test + k * test_size
        cal0 = t0 - gap - cal_size
        val0 = cal0 - gap - val_size
        train_end = val0 - gap
        if train_end - first_row < min_train:
            raise ValueError(f"fold {k}: only {train_end - first_row} training rows")
        folds.append(
            Fold(
                k=k,
                train=np.arange(first_row, train_end),
                val=np.arange(val0, val0 + val_size),
                cal=np.arange(cal0, cal0 + cal_size),
                test=np.arange(t0, t0 + test_size),
            )
        )
    return folds


def folds_from_config(n_rows: int, cfg: Config) -> list[Fold]:
    return walk_forward_folds(
        n_rows,
        cfg.n_folds,
        cfg.test_size,
        cfg.val_size,
        cfg.cal_size,
        cfg.gap,
        first_row=cfg.seq_len - 1,
    )


def fit_neural(
    kind: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[nn.Module, dict[str, float]]:
    """AdamW + BCE-with-logits, early stopping on val loss; returns the best-epoch weights."""
    torch.manual_seed(seed)
    model = build_model(
        kind, x_tr.shape[-1], cfg.seq_len, cfg.hidden, cfg.layers, cfg.dropout, cfg.mlp_hidden
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(1, cfg.patience // 2)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv, yv = torch.from_numpy(x_va), torch.from_numpy(y_va)
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, bad = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        sched.step(val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, bad = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"val_loss": best_loss, "best_epoch": float(best_epoch)}


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    xs = torch.from_numpy(x)
    out = [torch.sigmoid(model(xs[i : i + batch])) for i in range(0, len(xs), batch)]
    return torch.cat(out).numpy().astype(float)


def labels(df: pd.DataFrame) -> np.ndarray:
    """1 when the next-day S&P 500 return is positive (the notebooks' target)."""
    return (df["forward_returns"].to_numpy() > 0).astype(np.float32)


def run_fold(
    df: pd.DataFrame, fold: Fold, cfg: Config, log: Callable[[str], None] = print
) -> FoldResult:
    """Fit every configured model on one fold. Returns probabilities on the cal and test blocks."""
    cols = select_features(df.iloc[fold.train], candidate_columns(df), cfg.max_missing)
    x = df[cols].to_numpy(dtype=float)
    z = Standardizer(cfg.z_clip).fit(x[fold.train]).transform(x)
    y = labels(df)
    probs: dict[str, dict[str, np.ndarray]] = {}
    fits: list[dict[str, str | float]] = []

    if "logreg" in cfg.models:
        clf = LogisticRegression(C=cfg.logreg_c, max_iter=5000).fit(z[fold.train], y[fold.train])
        probs["logreg"] = {b: clf.predict_proba(z[getattr(fold, b)])[:, 1] for b in ("cal", "test")}

    neural = [m for m in cfg.models if m in NEURAL_MODELS]
    if neural:
        win = {
            b: make_windows(z, getattr(fold, b), cfg.seq_len)
            for b in ("train", "val", "cal", "test")
        }
        for kind in neural:
            by_seed = {"cal": [], "test": []}
            for seed in cfg.seeds:
                t = time.time()
                model, info = fit_neural(
                    kind, win["train"], y[fold.train], win["val"], y[fold.val], cfg, seed
                )
                for b in by_seed:
                    by_seed[b].append(predict_proba(model, win[b]))
                fits.append(
                    {"fold": fold.k, "model": kind, "seed": seed, **info, "sec": time.time() - t}
                )
                log(
                    f"  fold {fold.k} {kind:8s} seed {seed}: val_loss {info['val_loss']:.4f} "
                    f"best_epoch {info['best_epoch']:.0f} ({time.time() - t:.0f}s)"
                )
            probs[kind] = {b: np.mean(v, axis=0) for b, v in by_seed.items()}
            probs[kind]["test_by_seed"] = np.stack(by_seed["test"])
            probs[kind]["cal_by_seed"] = np.stack(by_seed["cal"])
        if len(neural) > 1:
            probs["ensemble"] = {
                b: np.mean([probs[m][b] for m in neural], axis=0) for b in ("cal", "test")
            }
    return {"probs": probs, "fits": fits, "n_features": len(cols)}
