import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower

from abtest import power


def test_sample_size_gives_target_power():
    n = power.sample_size_proportions(0.55, 0.05)
    assert power.power_proportions(0.55, 0.05, n) == pytest.approx(0.80, abs=0.005)


def test_mde_and_sample_size_are_inverse_for_means():
    n = power.sample_size_means(sd=100, mde_abs=10)
    assert power.mde_means(sd=100, n_control=n) == pytest.approx(10, rel=0.01)


def test_means_matches_statsmodels():
    ours = power.sample_size_means(sd=100, mde_abs=10)
    theirs = NormalIndPower().solve_power(effect_size=0.1, alpha=0.05, power=0.8, ratio=1)
    assert ours == int(np.ceil(theirs))


def test_unequal_split_needs_more_units():
    n_equal = power.sample_size_means(100, 10) * 2
    n_c = power.sample_size_means(100, 10, ratio=0.25)
    assert n_c * 1.25 > n_equal


def test_cuped_sd_shrinks():
    assert power.cuped_sd(100, 0.8) == pytest.approx(60)


def test_invalid_proportion_raises():
    with pytest.raises(ValueError):
        power.sample_size_proportions(0.98, 0.05)
