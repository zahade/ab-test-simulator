"""Synthetic customer table with the same schema as data.build_customer_table,
so the test suite runs without downloading the dataset (e.g. in GitHub Actions)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_customers(n: int = 4000, p: float = 0.55, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pre_revenue = rng.lognormal(6.3, 1.2, n)
    pre_orders = rng.poisson(3, n) + 1
    converted = (rng.random(n) < p).astype(int)
    orders = np.where(converted == 1, rng.poisson(1, n) + 1, 0)
    # outcome revenue correlated with pre-period revenue, like the real data
    revenue = np.where(converted == 1, pre_revenue * rng.lognormal(-0.6, 0.6, n), 0.0)
    return pd.DataFrame({
        "CustomerID": np.arange(n), "pre_revenue": pre_revenue, "pre_orders": pre_orders,
        "pre_recency_days": rng.integers(0, 270, n), "country_uk": (rng.random(n) < 0.9).astype(int),
        "revenue": revenue, "orders": orders, "converted": converted,
        "aov": np.where(orders > 0, revenue / np.maximum(orders, 1), np.nan),
    })


@pytest.fixture
def customers():
    return make_customers()
