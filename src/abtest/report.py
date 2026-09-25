"""Run the full analysis on an experiment table and turn it into a business decision."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import stats as st


@dataclass
class ExperimentConfig:
    name: str = "Win-back email with 10% discount"
    primary_metric: str = "converted"          # binary: did the customer buy in the window?
    revenue_metric: str = "revenue"            # secondary: revenue per eligible customer
    guardrail_metric: str = "aov"              # guardrail: spend per order must not collapse
    cuped_covariate: str = "pre_revenue"
    alpha: float = 0.05
    min_practical_lift: float = 0.02           # smallest conversion lift (abs) worth shipping
    guardrail_tolerance: float = -0.05         # max acceptable relative AOV drop
    balance_covariates: list[str] = field(
        default_factory=lambda: ["pre_revenue", "pre_orders", "pre_recency_days", "country_uk"])


def analyse_experiment(df: pd.DataFrame, cfg: ExperimentConfig = ExperimentConfig()) -> dict:
    a = cfg.alpha
    primary_z = st.two_proportion_ztest(df, cfg.primary_metric, a)
    primary_chi = st.chi_square_test(df, cfg.primary_metric, a)
    rev_welch = st.welch_ttest(df, cfg.revenue_metric, a)
    rev_cuped, cuped_info = st.cuped_ttest(df, cfg.revenue_metric, cfg.cuped_covariate, a)
    rev_mw = st.mann_whitney(df, cfg.revenue_metric, a)
    guard = st.welch_ttest(df, cfg.guardrail_metric, a)

    return {
        "config": cfg,
        "srm": st.srm_check(df),
        "balance": st.balance_table(df, cfg.balance_covariates),
        "primary": primary_z,
        "primary_chi2": primary_chi,
        "revenue": rev_cuped,
        "revenue_unadjusted": rev_welch,
        "revenue_mann_whitney": rev_mw,
        "cuped": cuped_info,
        "guardrail": guard,
        # One pre-registered primary metric is tested at full alpha. The supporting
        # metrics (secondary + guardrail) are Holm-corrected as a family.
        "corrected": st.adjust_pvalues([rev_cuped, guard], "holm"),
        "n_eligible": len(df),
    }


def _pp(x: float) -> str:
    """Absolute change in a rate, as percentage points."""
    return f"{x * 100:+.1f}pp"


def _gbp(x: float) -> str:
    return f"-£{abs(x):,.0f}" if x < 0 else f"£{x:,.0f}"


def recommend(res: dict) -> dict:
    """Decision rules, applied in order. Returns a verdict and a plain-English rationale."""
    cfg: ExperimentConfig = res["config"]
    p, rev, g = res["primary"], res["revenue"], res["guardrail"]
    corr = res["corrected"].set_index("metric")
    p_sig = p.significant
    rev_sig = bool(corr.loc[rev.metric, "significant_holm"])
    g_harm = bool(corr.loc[g.metric, "significant_holm"]) and g.rel_lift < cfg.guardrail_tolerance

    reasons = []
    if res["srm"]["srm_detected"]:
        verdict = "INVALID"
        reasons.append(f"Sample ratio mismatch (p={res['srm']['p_value']:.2g}). "
                       "Assignment or logging is broken; results can't be trusted.")
    elif p_sig and p.abs_diff > 0 and not g_harm:
        verdict = "SHIP"
        reasons.append(f"Conversion rose {_pp(p.abs_diff)} ({p.rel_lift:+.1%} relative), "
                       f"95% CI [{_pp(p.ci_low)}, {_pp(p.ci_high)}], significant at α={p.alpha}.")
        if p.ci_low < cfg.min_practical_lift:
            reasons.append(f"The CI's lower bound is below the {_pp(cfg.min_practical_lift)} practical "
                           "threshold, so the true lift may be small. Monitor after launch.")
    elif p_sig and p.abs_diff > 0 and g_harm:
        verdict = "SHIP WITH CAUTION" if rev_sig and rev.abs_diff > 0 else "DO NOT SHIP"
        reasons.append(f"Conversion improved {_pp(p.abs_diff)}, but AOV fell {g.rel_lift:.1%}, "
                       f"beyond the {cfg.guardrail_tolerance:.0%} guardrail.")
        reasons.append("Net revenue per customer still rose significantly." if verdict != "DO NOT SHIP"
                       else "Net revenue per customer did not improve significantly.")
    elif p_sig and p.abs_diff < 0:
        verdict = "DO NOT SHIP"
        reasons.append(f"Conversion fell {_pp(p.abs_diff)} (CI [{_pp(p.ci_low)}, {_pp(p.ci_high)}]).")
    elif p.ci_high < cfg.min_practical_lift:
        verdict = "DO NOT SHIP"
        reasons.append(f"No significant effect, and the CI upper bound ({_pp(p.ci_high)}) rules out "
                       f"a lift of {_pp(cfg.min_practical_lift)} or more. The change isn't worth it.")
    else:
        verdict = "INCONCLUSIVE"
        reasons.append(f"No significant effect (p={p.p_value:.3f}), but the CI [{_pp(p.ci_low)}, "
                       f"{_pp(p.ci_high)}] still includes a worthwhile lift. The test is underpowered: "
                       "extend it or re-run with a larger population.")

    impact = revenue_impact(res)
    return {"verdict": verdict, "reasons": reasons, "impact": impact}


def revenue_impact(res: dict, population: int | None = None) -> dict:
    """Project incremental revenue over one test window if rolled out to the whole population."""
    rev = res["revenue"]
    n = population or res["n_eligible"]
    return {"per_customer": rev.abs_diff, "ci_per_customer": (rev.ci_low, rev.ci_high),
            "total": rev.abs_diff * n, "ci_total": (rev.ci_low * n, rev.ci_high * n),
            "population": n}


def results_table(res: dict) -> pd.DataFrame:
    rows = [res[k].to_dict() for k in
            ["primary", "primary_chi2", "revenue_unadjusted", "revenue", "revenue_mann_whitney", "guardrail"]]
    cols = ["metric", "test", "control_value", "treatment_value", "abs_diff", "rel_lift",
            "ci_low", "ci_high", "p_value", "significant"]
    return pd.DataFrame(rows)[cols]


def format_report(res: dict, rec: dict) -> str:
    cfg: ExperimentConfig = res["config"]
    srm, imp = res["srm"], rec["impact"]
    lines = [
        f"=== {cfg.name} ===",
        f"Units: {srm['control_n']:,} control / {srm['treatment_n']:,} treatment "
        f"(SRM p={srm['p_value']:.3f})",
        f"Balance: {int(res['balance']['balanced'].sum())}/{len(res['balance'])} covariates |SMD| < 0.1",
        "",
        results_table(res).to_string(index=False, float_format=lambda x: f"{x:,.4f}"),
        "",
        f"CUPED: corr={res['cuped']['corr']:.2f}, variance reduction={res['cuped']['variance_reduction']:.0%}",
        f"Projected incremental revenue (per window, {imp['population']:,} customers): "
        f"{_gbp(imp['total'])} (95% CI {_gbp(imp['ci_total'][0])} to {_gbp(imp['ci_total'][1])})",
        "",
        f"VERDICT: {rec['verdict']}",
        *[f"  - {r}" for r in rec["reasons"]],
    ]
    return "\n".join(lines)
