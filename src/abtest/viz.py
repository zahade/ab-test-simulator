"""Plotting helpers used by the notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import TestResult

CONTROL, TREATMENT, ACCENT = "#6c757d", "#0d6efd", "#dc3545"


def plot_pvalue_hist(pvals: np.ndarray, alpha: float = 0.05, ax=None, title: str = "A/A test p-values"):
    ax = ax or plt.gca()
    ax.hist(pvals, bins=20, range=(0, 1), color=CONTROL, edgecolor="white")
    ax.axhline(len(pvals) / 20, color=ACCENT, ls="--", lw=1, label="uniform expectation")
    ax.axvline(alpha, color=TREATMENT, lw=1)
    ax.set(title=f"{title}  (FPR = {(pvals < alpha).mean():.1%})", xlabel="p-value", ylabel="count")
    ax.legend(frameon=False)
    return ax


def plot_power_curve(effects, theoretical, empirical=None, ax=None, xlabel="true effect",
                     target: float = 0.8, fmt=lambda x: f"{x:.0%}"):
    ax = ax or plt.gca()
    ax.plot(effects, theoretical, color=TREATMENT, lw=2, label="formula")
    if empirical is not None:
        ax.scatter(effects, empirical, color=ACCENT, zorder=3, label="simulation")
    ax.axhline(target, color=CONTROL, ls=":", lw=1)
    ax.set(xlabel=xlabel, ylabel="power", ylim=(0, 1.02), title="Power vs effect size")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x)))
    ax.legend(frameon=False)
    return ax


def plot_ci_forest(results: list[TestResult], relative: bool = True, ax=None):
    """Forest plot of effect estimates with CIs, one row per test."""
    ax = ax or plt.gca()
    labels = [f"{r.metric} · {r.test}" for r in results]
    for i, r in enumerate(results):
        base = r.control_value if relative and r.control_value else 1
        est, lo, hi = r.abs_diff / base, r.ci_low / base, r.ci_high / base
        color = TREATMENT if r.significant else CONTROL
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=color, capsize=4)
    ax.axvline(0, color=ACCENT, lw=1)
    ax.set_yticks(range(len(results)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="relative effect (treatment vs control)" if relative else "absolute effect",
           title="Effect estimates with 95% CIs")
    if relative:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
    return ax


def plot_group_distributions(df: pd.DataFrame, metric: str = "revenue", ax=None, log: bool = True):
    ax = ax or plt.gca()
    for g, c in [("control", CONTROL), ("treatment", TREATMENT)]:
        v = df.loc[df["group"] == g, metric]
        v = np.log1p(v) if log else v
        ax.hist(v, bins=40, alpha=0.55, color=c, label=g, density=True)
    ax.set(xlabel=f"log(1 + {metric})" if log else metric, ylabel="density",
           title=f"{metric} by group")
    ax.legend(frameon=False)
    return ax
