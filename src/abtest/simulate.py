"""Turn observational customer data into a simulated randomised experiment.

The Online Retail data contains no experiment, so we:
  1. randomly assign eligible customers to control / treatment,
  2. optionally inject a known ("ground truth") treatment effect into the treatment arm,
  3. analyse the result exactly as we would a real test.

Because the true effect is known, we can also validate the statistics themselves:
A/A tests should give ~alpha false positives, and the empirical detection rate
should match the power formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def assign_groups(df: pd.DataFrame, treatment_share: float = 0.5, seed: int | None = None,
                  stratify_col: str | None = None, n_strata: int = 5) -> pd.DataFrame:
    """Randomly assign units. With stratify_col, randomise within quantile bins of that
    column (block randomisation) to guarantee balance on a key covariate."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    if stratify_col is None:
        is_t = rng.random(len(out)) < treatment_share
    else:
        strata = pd.qcut(out[stratify_col].rank(method="first"), n_strata, labels=False)
        is_t = np.zeros(len(out), dtype=bool)
        for s in np.unique(strata):
            idx = np.flatnonzero(strata.to_numpy() == s)
            k = int(round(len(idx) * treatment_share))
            is_t[rng.choice(idx, k, replace=False)] = True
    out["group"] = np.where(is_t, "treatment", "control")
    return out


def inject_effect(df: pd.DataFrame, conversion_lift_abs: float = 0.0,
                  revenue_multiplier: float = 0.0, seed: int | None = None) -> pd.DataFrame:
    """Apply a known treatment effect to the treatment arm.

    conversion_lift_abs  absolute change in conversion probability (e.g. 0.05 = +5pp).
                         Positive: non-converters are converted with prob lift / (1 - p).
                         Newly converted customers get (revenue, orders) resampled from
                         existing treatment converters, so they look like real buyers.
                         Negative: converters are lost with prob |lift| / p.
    revenue_multiplier   relative change in spend among treatment converters
                         (e.g. -0.05 = discount cuts spend per buyer by 5%).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    t = (out["group"] == "treatment").to_numpy()
    conv = out["converted"].to_numpy().astype(bool)
    p = conv[t].mean()

    if conversion_lift_abs > 0:
        q = min(conversion_lift_abs / (1 - p), 1.0)
        flip = t & ~conv & (rng.random(len(out)) < q)
        donors = np.flatnonzero(t & conv)
        picks = rng.choice(donors, flip.sum(), replace=True)
        out.loc[flip, "revenue"] = out["revenue"].to_numpy()[picks]
        out.loc[flip, "orders"] = out["orders"].to_numpy()[picks]
    elif conversion_lift_abs < 0:
        q = min(-conversion_lift_abs / p, 1.0)
        lose = t & conv & (rng.random(len(out)) < q)
        out.loc[lose, ["revenue", "orders"]] = 0

    if revenue_multiplier:
        m = t & (out["orders"].to_numpy() > 0)
        out.loc[m, "revenue"] = out.loc[m, "revenue"] * (1 + revenue_multiplier)

    out["converted"] = (out["orders"] > 0).astype(int)
    out["aov"] = np.where(out["orders"] > 0, out["revenue"] / out["orders"].clip(lower=1), np.nan)
    return out


def simulate_experiment(customers: pd.DataFrame, conversion_lift_abs: float = 0.0,
                        revenue_multiplier: float = 0.0, treatment_share: float = 0.5,
                        seed: int | None = None, stratify_col: str | None = None) -> pd.DataFrame:
    """assign_groups + inject_effect in one call."""
    df = assign_groups(customers, treatment_share, seed, stratify_col)
    return inject_effect(df, conversion_lift_abs, revenue_multiplier, seed=None if seed is None else seed + 1)


# ------------------------------------------------------ validating the stats
def _p_prop(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pp = (a.sum() + b.sum()) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return 2 * stats.norm.sf(abs((b.mean() - a.mean()) / se)) if se > 0 else 1.0


def _p_welch(a: np.ndarray, b: np.ndarray) -> float:
    return stats.ttest_ind(b, a, equal_var=False).pvalue


def _p_cuped(a, b, xa, xb) -> float:
    x = np.concatenate([xa, xb]); y = np.concatenate([a, b])
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return _p_welch(a - theta * (xa - x.mean()), b - theta * (xb - x.mean()))


_TESTS = {"proportion": _p_prop, "welch": _p_welch}


def simulate_pvalues(customers: pd.DataFrame, metric: str, test: str = "welch",
                     n_sims: int = 1000, conversion_lift_abs: float = 0.0,
                     revenue_multiplier: float = 0.0, covariate: str | None = None,
                     seed: int = 0) -> np.ndarray:
    """Run n_sims independent simulated experiments and return their p-values.

    With no injected effect these are A/A tests: p-values should be ~Uniform(0,1) and
    the share below alpha is the false-positive rate. With an effect, the share below
    alpha is the empirical power. test: 'proportion', 'welch', or 'cuped' (needs covariate).
    """
    rng = np.random.default_rng(seed)
    pvals = np.empty(n_sims)
    no_effect = conversion_lift_abs == 0 and revenue_multiplier == 0
    y_all = customers[metric].to_numpy(dtype=float)
    x_all = customers[covariate].to_numpy(dtype=float) if covariate else None
    for i in range(n_sims):
        s = int(rng.integers(0, 2**31 - 1))
        if no_effect:  # fast path: just reshuffle labels
            t = np.random.default_rng(s).random(len(y_all)) < 0.5
            y = y_all
        else:
            df = simulate_experiment(customers, conversion_lift_abs, revenue_multiplier, seed=s)
            t = (df["group"] == "treatment").to_numpy()
            y = df[metric].to_numpy(dtype=float)
        a, b = y[~t], y[t]
        mask_a, mask_b = ~np.isnan(a), ~np.isnan(b)
        if test == "cuped":
            pvals[i] = _p_cuped(a[mask_a], b[mask_b], x_all[~t][mask_a], x_all[t][mask_b])
        else:
            pvals[i] = _TESTS[test](a[mask_a], b[mask_b])
    return pvals


def simulate_peeking(p: float, n_per_group: int, n_looks: int = 10, n_sims: int = 2000,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """Show why 'stop as soon as it's significant' inflates false positives.

    Simulates A/A tests on Bernoulli(p) streams, checking a z-test at n_looks evenly
    spaced interim points. Returns the false-positive rate with a single final look
    versus stopping at the first significant peek.
    """
    rng = np.random.default_rng(seed)
    a = rng.random((n_sims, n_per_group)) < p
    b = rng.random((n_sims, n_per_group)) < p
    looks = np.linspace(n_per_group / n_looks, n_per_group, n_looks).astype(int)
    ca, cb = a.cumsum(1)[:, looks - 1], b.cumsum(1)[:, looks - 1]
    pa, pb = ca / looks, cb / looks
    pp = (ca + cb) / (2 * looks)
    se = np.sqrt(pp * (1 - pp) * 2 / looks)
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = 2 * stats.norm.sf(np.abs((pb - pa) / se))
    pv = np.nan_to_num(pv, nan=1.0)
    sig = pv < alpha
    return {"looks": looks, "fpr_single_look": sig[:, -1].mean(),
            "fpr_with_peeking": sig.any(axis=1).mean(),
            "fpr_by_look": np.cumsum(sig, axis=1).astype(bool).mean(axis=0)}
