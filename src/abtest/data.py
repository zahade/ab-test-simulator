"""Load, clean and reshape the UCI Online Retail dataset into an experiment-ready table.

The raw data is transactional (one row per invoice line). A/B tests randomise *units*,
so we aggregate to one row per customer with:
  - pre-period covariates (used for balance checks and CUPED variance reduction)
  - experiment-window outcomes (the metrics the test is evaluated on)
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
import ssl
import urllib.error

import numpy as np
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
RAW_XLSX_NAME = "Online Retail.xlsx"
CACHE_NAME = "online_retail.parquet"


def download_raw(raw_dir: str | Path = "data/raw") -> Path:
    """Download the dataset from UCI if it isn't already on disk."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = raw_dir / RAW_XLSX_NAME
    if xlsx_path.exists():
        return xlsx_path

    print(f"Downloading dataset from {UCI_ZIP_URL} ...")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(UCI_ZIP_URL, context=ctx, timeout=120) as resp:
            payload = resp.read()
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not download the dataset ({e.reason}).\n"
            f"Download it manually from https://archive.ics.uci.edu/dataset/352/online+retail,\n"
            f"unzip it, and place '{RAW_XLSX_NAME}' in: {raw_dir.resolve()}"
        ) from None

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract(RAW_XLSX_NAME, raw_dir)
    return xlsx_path


def load_raw(raw_dir: str | Path = "data/raw", use_cache: bool = True) -> pd.DataFrame:
    """Load raw transactions. The xlsx takes ~30s to parse, so a parquet cache is kept."""
    raw_dir = Path(raw_dir)
    cache = raw_dir / CACHE_NAME
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    xlsx_path = download_raw(raw_dir)
    df = pd.read_excel(
        xlsx_path,
        dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str},
    )
    if use_cache:
        df.to_parquet(cache, index=False)
    return df


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove rows that can't be attributed to a customer purchase.

    Returns the cleaned frame and a log of how many rows each rule removed,
    so the notebook can show the cleaning funnel transparently.
    """
    log = {"raw_rows": len(df)}
    out = df.copy()

    out = out.dropna(subset=["CustomerID"])
    log["after_drop_missing_customer"] = len(out)

    out = out[~out["InvoiceNo"].astype(str).str.startswith("C")]
    log["after_drop_cancellations"] = len(out)

    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    log["after_drop_nonpositive_qty_price"] = len(out)

    out = out.drop_duplicates()
    log["after_drop_duplicates"] = len(out)

    out["CustomerID"] = out["CustomerID"].astype(int)
    out["Revenue"] = out["Quantity"] * out["UnitPrice"]
    return out.reset_index(drop=True), log


@dataclass(frozen=True)
class ExperimentWindow:
    """Splits the timeline into a pre-period (covariates) and a test window (outcomes)."""

    pre_start: str = "2010-12-01"
    test_start: str = "2011-09-01"
    test_end: str = "2011-12-01"  # exclusive; Dec 2011 is only 9 days so it's excluded

    @property
    def test_days(self) -> int:
        return (pd.Timestamp(self.test_end) - pd.Timestamp(self.test_start)).days


def build_customer_table(tx: pd.DataFrame, window: ExperimentWindow = ExperimentWindow()) -> pd.DataFrame:
    """One row per customer who purchased in the pre-period (the eligible population).

    Outcome columns (test window):
        converted      1 if the customer placed any order in the window, else 0
        revenue        total spend in the window (0 for non-converters)
        orders         number of distinct invoices in the window
        aov            average order value in the window (NaN for non-converters)
    Covariate columns (pre-period):
        pre_revenue, pre_orders, pre_recency_days, country_uk
    """
    ts = tx["InvoiceDate"]
    pre = tx[(ts >= window.pre_start) & (ts < window.test_start)]
    test = tx[(ts >= window.test_start) & (ts < window.test_end)]

    pre_agg = pre.groupby("CustomerID").agg(
        pre_revenue=("Revenue", "sum"),
        pre_orders=("InvoiceNo", "nunique"),
        last_purchase=("InvoiceDate", "max"),
        country=("Country", "first"),
    )
    pre_agg["pre_recency_days"] = (pd.Timestamp(window.test_start) - pre_agg["last_purchase"]).dt.days
    pre_agg["country_uk"] = (pre_agg["country"] == "United Kingdom").astype(int)
    pre_agg = pre_agg.drop(columns=["last_purchase", "country"])

    test_agg = test.groupby("CustomerID").agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
    )

    cust = pre_agg.join(test_agg, how="left")
    cust[["revenue", "orders"]] = cust[["revenue", "orders"]].fillna(0)
    cust["orders"] = cust["orders"].astype(int)
    cust["converted"] = (cust["orders"] > 0).astype(int)
    cust["aov"] = np.where(cust["orders"] > 0, cust["revenue"] / cust["orders"].clip(lower=1), np.nan)
    return cust.reset_index()


def load_customer_table(raw_dir: str | Path = "data/raw",
                        window: ExperimentWindow = ExperimentWindow()) -> pd.DataFrame:
    """Convenience: raw -> clean -> customer table in one call."""
    tx, _ = clean_transactions(load_raw(raw_dir))
    return build_customer_table(tx, window)
