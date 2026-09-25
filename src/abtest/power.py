"""Sample size, minimum detectable effect (MDE) and power calculations.

Closed-form normal-approximation formulas, written out explicitly so the maths is
visible. tests/test_power.py cross-checks them against statsmodels.

Notation: n = per-group size of the control arm, ratio k = n_treatment / n_control.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _z(alpha: float, power: float, two_sided: bool = True) -> tuple[float, float]:
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    return za, stats.norm.ppf(power)


# -------------------------------------------------------------- proportions
def sample_size_proportions(p0: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8,
                            ratio: float = 1.0) -> int:
    """Control-arm n to detect p0 -> p0 + mde_abs (two-sided, unpooled variance)."""
    if not 0 < p0 < 1 or not 0 < p0 + mde_abs < 1:
        raise ValueError("Proportions must be in (0, 1).")
    p1 = p0 + mde_abs
    za, zb = _z(alpha, power)
    var = p0 * (1 - p0) + p1 * (1 - p1) / ratio
    return int(np.ceil((za + zb) ** 2 * var / mde_abs ** 2))


def mde_proportions(p0: float, n_control: int, alpha: float = 0.05, power: float = 0.8,
                    ratio: float = 1.0) -> float:
    """Smallest absolute lift detectable with n_control (uses baseline variance for both arms)."""
    za, zb = _z(alpha, power)
    return (za + zb) * np.sqrt(p0 * (1 - p0) * (1 / n_control + 1 / (n_control * ratio)))


def power_proportions(p0: float, mde_abs: float, n_control: int, alpha: float = 0.05,
                      ratio: float = 1.0) -> float:
    p1 = p0 + mde_abs
    za = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p0 * (1 - p0) / n_control + p1 * (1 - p1) / (n_control * ratio))
    return float(stats.norm.cdf(abs(mde_abs) / se - za))


# -------------------------------------------------------------------- means
def sample_size_means(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8,
                      ratio: float = 1.0) -> int:
    za, zb = _z(alpha, power)
    return int(np.ceil((za + zb) ** 2 * sd ** 2 * (1 + 1 / ratio) / mde_abs ** 2))


def mde_means(sd: float, n_control: int, alpha: float = 0.05, power: float = 0.8,
              ratio: float = 1.0) -> float:
    za, zb = _z(alpha, power)
    return (za + zb) * sd * np.sqrt(1 / n_control + 1 / (n_control * ratio))


def power_means(sd: float, mde_abs: float, n_control: int, alpha: float = 0.05,
                ratio: float = 1.0) -> float:
    za = stats.norm.ppf(1 - alpha / 2)
    se = sd * np.sqrt(1 / n_control + 1 / (n_control * ratio))
    return float(stats.norm.cdf(abs(mde_abs) / se - za))


def cuped_sd(sd: float, corr: float) -> float:
    """Effective SD after CUPED: variance shrinks by (1 - rho^2)."""
    return sd * np.sqrt(1 - corr ** 2)


def duration_days(n_total: int, eligible_per_day: float) -> int:
    """How long the test must run to enrol n_total units at a given daily inflow."""
    return int(np.ceil(n_total / eligible_per_day))
