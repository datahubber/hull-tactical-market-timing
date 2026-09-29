"""Command line entry point: `hull-timing run | summary | plot`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import Config


def _summary(results: Path) -> None:
    """Rebuild summary.csv from folds.csv (no retraining) and print it."""
    from .pipeline import summarize

    s = summarize(pd.read_csv(results / "folds.csv"))
    s.to_csv(results / "summary.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 140, "display.max_rows", 100):
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hull-timing", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="walk-forward train + evaluate")
    run.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    run.add_argument("--data", type=Path, default=Path("data"), help="folder with train.csv")
    run.add_argument("--out", type=Path, default=Path("results"))

    summ = sub.add_parser("summary", help="rebuild results/summary.csv from folds.csv and print it")
    summ.add_argument("--results", type=Path, default=Path("results"))

    plot = sub.add_parser("plot", help="draw the results figure")
    plot.add_argument("--results", type=Path, default=Path("results"))
    plot.add_argument("--out", type=Path, default=Path("assets/walk_forward.png"))

    args = ap.parse_args(argv)
    if args.cmd == "run":
        from .pipeline import run as run_pipeline

        run_pipeline(Config.from_yaml(args.config), args.data, args.out)
        _summary(args.out)
    elif args.cmd == "summary":
        _summary(args.results)
    elif args.cmd == "plot":
        from .plot import plot_results

        plot_results(args.results, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
