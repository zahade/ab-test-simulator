"""Statistical tests for A/B experiments.

Every test returns a TestResult with the same fields so results can be stacked into
one table and fed to the recommendation logic regardless of which test produced them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class TestResult:
    metric: str
    test: str
    control_value: float
    treatment_value: float
    abs_diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    statistic: float
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def _split(df: pd.DataFrame, metric: str, group_col: str = "group"):
    a = df.loc[df[group_col] == "control", metric].dropna().to_numpy(dtype=float)
    b = df.loc[df[group_col] == "treatment", metric].dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"Both groups need observations for '{metric}'.")
    return a, b


def _rel(diff: float, base: float) -> float:
    return diff / base if base != 0 else np.nan


# ---------------------------------------------------------------- proportions
def two_proportion_ztest(df: pd.DataFrame, metric: str = "converted", alpha: float = 0.05,
                         group_col: str = "group") -> TestResult:
    """Pooled two-proportion z-test (p-value) with an unpooled Wald CI for the difference."""
    a, b = _split(df, metric, group_col)
    n1, n2 = len(a), len(b)
    p1, p2 = a.mean(), b.mean()
    p_pool = (a.sum() + b.sum()) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))

    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    return TestResult(metric, "two-proportion z-test", p1, p2, diff, _rel(diff, p1),
                      diff - zc * se_diff, diff + zc * se_diff, p_value, z, alpha)


def chi_square_test(df: pd.DataFrame, metric: str = "converted", alpha: float = 0.05,
                    group_col: str = "group", yates: bool = False) -> TestResult:
    """Pearson chi-square test of independence on the 2x2 group x outcome table.

    Without Yates' correction this is mathematically identical to the pooled z-test
    (chi2 = z^2), which the notebook demonstrates as a sanity check.
    """
    table = pd.crosstab(df[group_col], df[metric]).reindex(index=["control", "treatment"])
    chi2, p_value, _, _ = stats.chi2_contingency(table.to_numpy(), correction=yates)
    z = two_proportion_ztest(df, metric, alpha, group_col)  # reuse CI + effect sizes
    return TestResult(metric, "chi-square" + (" (Yates)" if yates else ""), z.control_value,
                      z.treatment_value, z.abs_diff, z.rel_lift, z.ci_low, z.ci_high,
                      p_value, chi2, alpha)


# ---------------------------------------------------------------------- means
def welch_ttest(df: pd.DataFrame, metric: str = "revenue", alpha: float = 0.05,
                group_col: str = "group", label: str | None = None) -> TestResult:
    """Welch's t-test (no equal-variance assumption) with a Welch–Satterthwaite CI."""
    a, b = _split(df, metric, group_col)
    t, p_value = stats.ttest_ind(b, a, equal_var=False)
    v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(v1 + v2)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))
    tc = stats.t.ppf(1 - alpha / 2, dof)
    diff = b.mean() - a.mean()
    return TestResult(label or metric, "Welch t-test", a.mean(), b.mean(), diff,
                      _rel(diff, a.mean()), diff - tc * se, diff + tc * se, p_value, t, alpha)


def mann_whitney(df: pd.DataFrame, metric: str = "revenue", alpha: float = 0.05,
                 group_col: str = "group") -> TestResult:
    """Non-parametric rank test. Robust to outliers, but tests distribution shift,
    not the mean — so the CI reported is a bootstrap CI on the *median* difference."""
    a, b = _split(df, metric, group_col)
    u, p_value = stats.mannwhitneyu(b, a, alternative="two-sided")
    lo, hi = bootstrap_ci(a, b, stat=np.median, alpha=alpha, n_boot=2000)
    diff = np.median(b) - np.median(a)
    return TestResult(metric, "Mann-Whitney U (median)", np.median(a), np.median(b), diff,
                      _rel(diff, np.median(a)), lo, hi, p_value, u, alpha)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, stat=np.mean, alpha: float = 0.05,
                 n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for stat(b) - stat(a)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = stat(b[ib], axis=1) - stat(a[ia], axis=1)
    return tuple(np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))


def bootstrap_test(df: pd.DataFrame, metric: str = "revenue", alpha: float = 0.05,
                   group_col: str = "group", n_boot: int = 5000, seed: int = 0) -> TestResult:
    """Bootstrap difference in means. p-value from the bootstrap distribution of the
    difference re-centred under H0 (a shift-based null)."""
    a, b = _split(df, metric, group_col)
    rng = np.random.default_rng(seed)
    obs = b.mean() - a.mean()
    lo, hi = bootstrap_ci(a, b, np.mean, alpha, n_boot, seed)
    # null: shift both groups to the pooled mean, resample, compare
    pooled = np.concatenate([a, b]).mean()
    a0, b0 = a - a.mean() + pooled, b - b.mean() + pooled
    null = (b0[rng.integers(0, len(b0), (n_boot, len(b0)))].mean(1)
            - a0[rng.integers(0, len(a0), (n_boot, len(a0)))].mean(1))
    p_value = (np.abs(null) >= abs(obs)).mean()
    return TestResult(metric, "bootstrap (mean)", a.mean(), b.mean(), obs,
                      _rel(obs, a.mean()), lo, hi, max(p_value, 1 / n_boot), np.nan, alpha)


def winsorize(df: pd.DataFrame, metric: str, upper_q: float = 0.99,
              new_col: str | None = None) -> pd.DataFrame:
    """Cap a heavy-tailed metric at its pooled upper quantile. The cap is computed on both
    groups together so it can't differ by arm. Trades a little bias for a lot of variance."""
    cap = df[metric].quantile(upper_q)
    return df.assign(**{new_col or f"{metric}_w": df[metric].clip(upper=cap)})


# ---------------------------------------------------------------------- CUPED
def cuped_adjust(df: pd.DataFrame, metric: str, covariate: str) -> tuple[pd.Series, float]:
    """CUPED (Deng et al., 2013): Y_adj = Y - theta * (X - mean(X)).

    X is a pre-experiment covariate, so it's independent of assignment and the
    adjustment can't bias the treatment effect — it only removes variance explained
    by pre-existing differences between customers. Variance falls by a factor of
    (1 - corr(X, Y)^2).
    """
    x = df[covariate].to_numpy(dtype=float)
    y = df[metric].to_numpy(dtype=float)
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return pd.Series(y - theta * (x - x.mean()), index=df.index), theta


def cuped_ttest(df: pd.DataFrame, metric: str = "revenue", covariate: str = "pre_revenue",
                alpha: float = 0.05, group_col: str = "group") -> tuple[TestResult, dict]:
    adj, theta = cuped_adjust(df, metric, covariate)
    tmp = df[[group_col]].assign(_adj=adj)
    res = welch_ttest(tmp, "_adj", alpha, group_col, label=metric)
    res.test = "Welch t-test + CUPED"
    # report un-adjusted group means for readability; the diff/CI/p are from adjusted data
    a, b = _split(df, metric, group_col)
    res.control_value, res.treatment_value = a.mean(), b.mean()
    res.rel_lift = _rel(res.abs_diff, a.mean())
    info = {"theta": theta,
            "corr": np.corrcoef(df[covariate], df[metric])[0, 1],
            "variance_reduction": 1 - adj.var() / df[metric].var()}
    return res, info


# ------------------------------------------------------------ design checks
def srm_check(df: pd.DataFrame, expected_share: float = 0.5, group_col: str = "group",
              threshold: float = 0.001) -> dict:
    """Sample Ratio Mismatch: chi-square goodness-of-fit on group sizes.
    A tiny p-value means the randomisation/logging is broken and results shouldn't be trusted."""
    counts = df[group_col].value_counts().reindex(["control", "treatment"]).fillna(0)
    n = counts.sum()
    expected = [n * (1 - expected_share), n * expected_share]
    chi2, p = stats.chisquare(counts.to_numpy(), expected)
    return {"control_n": int(counts["control"]), "treatment_n": int(counts["treatment"]),
            "chi2": chi2, "p_value": p, "srm_detected": p < threshold}


def balance_table(df: pd.DataFrame, covariates: list[str], group_col: str = "group") -> pd.DataFrame:
    """Standardised mean differences on pre-period covariates. |SMD| < 0.1 is the usual bar."""
    rows = []
    for c in covariates:
        a, b = _split(df, c, group_col)
        smd = (b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"covariate": c, "control_mean": a.mean(), "treatment_mean": b.mean(),
                     "smd": smd, "balanced": abs(smd) < 0.1})
    return pd.DataFrame(rows)


def adjust_pvalues(results: list[TestResult], method: str = "holm") -> pd.DataFrame:
    """Multiple-comparison correction across several metrics tested in one experiment."""
    table = pd.DataFrame([r.to_dict() for r in results])
    reject, p_adj, _, _ = multipletests(table["p_value"], alpha=results[0].alpha, method=method)
    table[f"p_{method}"] = p_adj
    table[f"significant_{method}"] = reject
    return table
