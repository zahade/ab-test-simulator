import numpy as np
import pytest

from abtest import simulate as sim


def test_aa_false_positive_rate_near_alpha(customers):
    p = sim.simulate_pvalues(customers, "converted", "proportion", n_sims=1000)
    assert 0.03 < (p < 0.05).mean() < 0.07


def test_injected_conversion_lift_is_unbiased(customers):
    lifts = []
    for s in range(200):
        df = sim.simulate_experiment(customers, conversion_lift_abs=0.05, seed=s)
        g = df.groupby("group")["converted"].mean()
        lifts.append(g["treatment"] - g["control"])
    assert np.mean(lifts) == pytest.approx(0.05, abs=0.005)


def test_negative_lift_and_revenue_multiplier(customers):
    base = sim.assign_groups(customers, seed=0)
    out = sim.inject_effect(base, conversion_lift_abs=-0.1, revenue_multiplier=-0.5, seed=0)
    t = out.group == "treatment"
    assert out.loc[t, "converted"].mean() < base.loc[t, "converted"].mean()
    assert out.loc[t, "revenue"].sum() < base.loc[t, "revenue"].sum()
    assert out.loc[~t].equals(base.loc[~t])  # control untouched


def test_stratified_assignment_balances_strata(customers):
    df = sim.assign_groups(customers, seed=0, stratify_col="pre_revenue")
    assert abs((df.group == "treatment").mean() - 0.5) < 0.002


def test_peeking_inflates_false_positives():
    r = sim.simulate_peeking(p=0.5, n_per_group=1000, n_looks=10, n_sims=2000)
    assert r["fpr_single_look"] < 0.07
    assert r["fpr_with_peeking"] > 0.12
