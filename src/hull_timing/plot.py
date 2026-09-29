"""The README figure: per-fold out-of-sample scores, and the optimism of in-sample tuning.

Left: score on each test block. Right: the tuned rule's score minus buy-and-hold's score on the
same block, for the cal block it was tuned on (hollow) and the next test block (filled), so the
change of market period between the two blocks cancels out."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2; gray marks the baseline
FONT = 10.0


def _style(ax: plt.Axes, title: str, n_folds: int) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=FONT + 1.5, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=FONT, length=0)
    ax.set_xticks(range(n_folds), [str(k + 1) for k in range(n_folds)])
    ax.set_xlabel("walk-forward fold (504-day test block)", color=MUTED, fontsize=FONT)


def _legend(ax: plt.Axes) -> None:
    ax.legend(
        frameon=False,
        fontsize=FONT,
        labelcolor=INK2,
        loc="upper left",
        ncols=1,
        bbox_to_anchor=(0, -0.14),
    )


def plot_results(results: str | Path, out: str | Path) -> None:
    results = Path(results)
    df = pd.read_csv(results / "folds.csv")
    n_folds = len(json.loads((results / "meta.json").read_text())["folds"])

    def series(model: str, policy: str) -> pd.Series:
        return df[(df.model == model) & (df.policy == policy)].set_index("fold")["score"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.0), facecolor=SURFACE, layout="constrained")
    _style(a, "Out-of-sample score on each test block", n_folds)
    a.set_ylabel("competition score (adjusted Sharpe)", color=MUTED, fontsize=FONT)
    for model, policy, label, color, marker, dx in [
        ("buy_hold", "1.0x", "buy-and-hold 1.0x", MUTED, "D", -0.15),
        ("buy_hold", "1.2x", "always 1.2x", ORANGE, "s", 0.0),
        ("ensemble", "binary_0.49", "submitted rule (p >= 0.49 -> 1.2x)", BLUE, "o", 0.15),
    ]:
        s = series(model, policy)
        a.plot(
            s.index + dx,
            s.values,
            color=color,
            linewidth=0,
            marker=marker,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{label}, mean {s.mean():.2f}",
            zorder=3,
        )
    _legend(a)

    _style(b, "Tuned rule minus buy-and-hold, same block", n_folds)
    b.set_ylabel("score difference to buy-and-hold", color=MUTED, fontsize=FONT)
    ins = series("ensemble", "tuned_insample") - series("buy_hold", "1.0x_on_cal")
    oos = series("ensemble", "tuned_oos") - series("buy_hold", "1.0x")
    for k in ins.index:
        b.plot([k, k], [ins[k], oos[k]], color=GRID, linewidth=2, zorder=1)
    for s, label, face in [
        (ins, "tuned and scored on the cal block", SURFACE),
        (oos, "same rule on the next test block", BLUE),
    ]:
        b.plot(
            s.index,
            s.values,
            "o",
            color=BLUE,
            markerfacecolor=face,
            markeredgecolor=BLUE if face == SURFACE else SURFACE,
            markeredgewidth=2 if face == SURFACE else 1.5,
            markersize=8,
            label=f"{label}, mean {s.mean():+.2f}",
            zorder=3,
        )
    _legend(b)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
