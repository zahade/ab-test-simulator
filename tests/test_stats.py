import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from abtest import stats as st
from abtest.simulate import assign_groups


def test_chi_square_equals_z_squared(customers):
    df = assign_groups(customers, seed=1)
    z, chi = st.two_proportion_ztest(df), st.chi_square_test(df)
    assert chi.statistic == pytest.approx(z.statistic ** 2)
    assert chi.p_value == pytest.approx(z.p_value)


def test_welch_matches_scipy(customers):
    df = assign_groups(customers, seed=2)
    r = st.welch_ttest(df, "revenue")
    a = df.loc[df.group == "control", "revenue"]; b = df.loc[df.group == "treatment", "revenue"]
    assert r.p_value == pytest.approx(sps.ttest_ind(b, a, equal_var=False).pvalue)
    assert r.ci_low < r.abs_diff < r.ci_high


def test_proportion_ci_coverage():
    """~95% of CIs should contain the true difference."""
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(1000):
        df = pd.DataFrame({"group": ["control"] * 800 + ["treatment"] * 800,
                           "converted": np.r_[rng.random(800) < 0.5, rng.random(800) < 0.55].astype(int)})
        r = st.two_proportion_ztest(df)
        hits += r.ci_low <= 0.05 <= r.ci_high
    assert 0.93 < hits / 1000 < 0.97


def test_srm_detects_imbalance():
    df = pd.DataFrame({"group": ["control"] * 600 + ["treatment"] * 400})
    assert st.srm_check(df)["srm_detected"]
    ok = pd.DataFrame({"group": ["control"] * 505 + ["treatment"] * 495})
    assert not st.srm_check(ok)["srm_detected"]


def test_cuped_reduces_variance_without_moving_estimate(customers):
    df = assign_groups(customers, seed=3)
    raw = st.welch_ttest(df, "revenue")
    adj, info = st.cuped_ttest(df, "revenue", "pre_revenue")
    assert info["variance_reduction"] > 0.1
    assert (adj.ci_high - adj.ci_low) < (raw.ci_high - raw.ci_low)


def test_holm_is_more_conservative(customers):
    df = assign_groups(customers, seed=4)
    results = [st.two_proportion_ztest(df), st.welch_ttest(df, "revenue"), st.welch_ttest(df, "aov")]
    table = st.adjust_pvalues(results)
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all()


def test_balance_table_flags_imbalance():
    df = pd.DataFrame({"group": ["control"] * 100 + ["treatment"] * 100,
                       "x": np.r_[np.zeros(100), np.ones(100)] + np.random.default_rng(0).normal(0, 1, 200)})
    assert not st.balance_table(df, ["x"])["balanced"].iloc[0]


def test_winsorize_caps_and_shrinks_variance(customers):
    out = st.winsorize(customers, "revenue", 0.99)
    assert out["revenue_w"].max() == pytest.approx(customers["revenue"].quantile(0.99))
    assert out["revenue_w"].var() < customers["revenue"].var()
