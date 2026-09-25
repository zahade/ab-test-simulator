"""Command-line entry point.

    python -m abtest --conversion-lift 0.05 --revenue-mult -0.03 --seed 42
"""
from __future__ import annotations

import argparse

from .data import load_customer_table
from .report import ExperimentConfig, analyse_experiment, format_report, recommend
from .simulate import simulate_experiment


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Simulate and analyse an A/B test on UCI Online Retail.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--conversion-lift", type=float, default=0.05,
                    help="true absolute conversion lift to inject (0.05 = +5pp)")
    ap.add_argument("--revenue-mult", type=float, default=0.0,
                    help="true relative change in spend per converter (-0.03 = -3%%)")
    ap.add_argument("--treatment-share", type=float, default=0.5)
    ap.add_argument("--stratify", default=None, help="covariate to block-randomise on, e.g. pre_revenue")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    customers = load_customer_table(args.raw_dir)
    df = simulate_experiment(customers, args.conversion_lift, args.revenue_mult,
                             args.treatment_share, args.seed, args.stratify)
    res = analyse_experiment(df, ExperimentConfig(alpha=args.alpha))
    print(format_report(res, recommend(res)))


if __name__ == "__main__":
    main()
