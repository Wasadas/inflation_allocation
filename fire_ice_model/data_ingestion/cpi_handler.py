"""
data_ingestion/cpi_handler.py

This module fetches UK inflation data (CPI) and transforms it into the
core signals used in the Neville-style macro regime framework.

PREAMBLE 
Inflation is not just a number and it is a regime-defining force in financial
markets. Academic research (e.g. Ilmanen, 2011; Ang, 2014; and the broader
regime literature) shows that asset returns behave very differently depending
on whether inflation is:

    1) Low and stable
    2) High and rising
    3) High but falling
    4) Re-accelerating after a decline

The Neville framework formalises this by focusing not only on the LEVEL
of inflation, but also its MOMENTUM and ACCELERATION.

What we compute
---------------
From the raw CPI index we derive three key components:

1. YoY CPI (level)
   The year-on-year percentage change.
   This captures whether inflation is economically meaningful.
   A commonly used macro threshold (e.g., 5%) is used to flag “high inflation”.

2. Trend (1st derivative)
   A smoothed 12-month momentum measure.
   This tells us whether inflation pressure is building or easing.
   Markets are highly sensitive to changes in trend, not just levels.

3. Acceleration (2nd derivative)
   The 3-month change in trend.
   This is the “velocity” of inflation.
   It distinguishes:
       - High but decelerating inflation (Boom)
       - High and accelerating inflation (Fire)

This second derivative is particularly important. Research in macro
asset pricing shows that turning points in inflation momentum often
drive large cross-asset reallocations.

Data sources
------------
We rely only on free public data. NO BQL / Bloomberg or client data was used.

- FRED (St. Louis Fed): UK CPI All Items (GBRCPIALLMINMEI)
- ONS (UK Office for National Statistics): direct CPI index download
- Bank of England (optional context for policy regime)

Design philosophy
-----------------
The goal is robustness:
- No API keys required
- Automatic fallback between ONS and FRED
- Caching to ensure reproducibility
- Clean monthly time index only (no annual artifacts)

The output is a DataFrame suitable for:
- Regime classification
- Asset allocation backtests
- Inflation surprise modelling
- Macro risk overlays

OBLIGATORY DISCLAIMER: the module does not forecast inflation.
It classifies the inflation regime using observable data.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from io import StringIO
import requests
import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

REGIME_CFG = CONFIG["regime"]
DATA_CFG   = CONFIG["data"]
CACHE_DIR  = Path(DATA_CFG["cache_dir"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Public Interface
# ------------------------------------------------------------------

def get_uk_cpi(
    start: str = "1988-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        cpi_index    : raw CPI index level
        cpi_yoy      : year-on-year % change
        cpi_mom      : month-on-month % change
        trend        : 12m smoothed trend (Neville momentum signal)
        acceleration : 3m change in trend (2nd derivative)
        regime_flag  : True if cpi_yoy > threshold (Neville 5%)

    Parameters
    ----------
    start : str   ISO date string
    end   : str   ISO date string (default: today)
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_file = CACHE_DIR / "uk_cpi.parquet"

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Refresh if data is stale (last row > 35 days ago)
        last_date = pd.Timestamp(df.index[-1])
        if (pd.Timestamp.today() - last_date).days < 35:
            logger.info("Loaded CPI from cache (%s rows)", len(df))
            return df.loc[start:end]

    df = _fetch_ons_cpi(start, end)

    if df is None or df.empty:
        logger.warning("ONS fetch failed — trying FRED")
        df = _fetch_fred_cpi(start, end)

    if df is None or df.empty:
        logger.error("Both FRED and ONS CPI fetches failed. Check network/SSL, and verify the endpoints are reachable.")
        raise RuntimeError("Could not retrieve UK CPI from any source.")

    df = _compute_derivatives(df)
    df.to_parquet(cache_file)
    logger.info("CPI data fetched and cached (%s rows)", len(df))
    return df.loc[start:end]


def get_inflation_surprise(
    cpi_df: pd.DataFrame,
    survey_source: str = "cleveland_fed_approx",
) -> pd.Series:
    """
    Computes CPI Surprise = Actual YoY - Expected YoY.

    For UK, a clean consensus series is not freely available at monthly
    frequency. We approximate using:
      - 'cleveland_fed_approx': 3-month lagged CPI as naive forecast
      - 'ewm': exponentially-weighted moving average as adaptive forecast

    This is the variable Neville et al. use to define a "regime" as
    a persistent deviation from expectations, not just a level.
    """
    actual = cpi_df["cpi_yoy"]

    if survey_source == "cleveland_fed_approx":
        # Naive forecast: what last quarter's CPI suggested
        expected = actual.shift(3)
    elif survey_source == "ewm":
        expected = actual.ewm(span=12, adjust=False).mean().shift(1)
    else:
        raise ValueError(f"Unknown survey_source: {survey_source}")

    surprise = (actual - expected).rename("cpi_surprise")
    return surprise


# ------------------------------------------------------------------
# Private Helpers
# ------------------------------------------------------------------

def _fetch_fred_cpi(start: str, end: str) -> pd.DataFrame | None:
    """Fetch UK CPI from FRED via the public CSV endpoint (no pandas_datareader)."""
    try:
        series_id = DATA_CFG["fred_series"]["uk_cpi"]

        # Public FRED CSV endpoint (no API key required for many series)
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}"
            f"&cosd={start}"
            f"&coed={end}"
        )
        logger.info("Requesting FRED CSV: %s", url)
        resp = requests.get(url, timeout=30)
        logger.info("FRED response status: %s", resp.status_code)
        resp.raise_for_status()

        raw = pd.read_csv(StringIO(resp.text))

        # FRED CSV date column is typically 'DATE' or 'observation_date' depending on endpoint
        if "DATE" in raw.columns:
            date_col = "DATE"
        elif "observation_date" in raw.columns:
            date_col = "observation_date"
        else:
            # Fallback: assume first column is the date
            date_col = raw.columns[0]

        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.set_index(date_col).sort_index()

        # Some endpoints may upper/lower-case the series id; try exact, then case-insensitive match
        if series_id in raw.columns:
            series_col = series_id
        else:
            matches = [c for c in raw.columns if c.lower() == series_id.lower()]
            series_col = matches[0] if matches else None

        if series_col is None:
            raise KeyError(f"Series column '{series_id}' not found in FRED CSV columns: {list(raw.columns)}")

        s = pd.to_numeric(raw[series_col], errors="coerce")

        df = pd.DataFrame(index=s.index)
        df["cpi_index"] = s

        logger.info("Fetched %s rows from FRED (CSV) (%s)", len(df), series_id)
        return df

    except Exception:
        logger.exception("FRED fetch error")
        return None


def _fetch_ons_cpi(start: str, end: str) -> pd.DataFrame | None:
    """Fetch UK CPI (or CPIH) from ONS via the CSV generator endpoint.

    Why this approach:
      - ONS beta API dataset endpoints can change and often 404.
      - The ONS website provides a stable CSV generator for each time series page.

    By default, this pulls CPI index D7BT from dataset MM23:
      https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23

    Config (optional) in config.yaml under `data`:
      ons_series_id: "d7bt"   # time series code on ONS
      ons_dataset_id: "mm23"  # dataset code on ONS
    """
    try:
        series_id = str(DATA_CFG.get("ons_series_id", "d7bt")).lower()
        dataset_id = str(DATA_CFG.get("ons_dataset_id", "mm23")).lower()

        # ONS CSV generator used by the website "Download full time series as CSV"
        uri = f"/economy/inflationandpriceindices/timeseries/{series_id}/{dataset_id}"
        url = f"https://www.ons.gov.uk/generator?format=csv&uri={uri}"

        logger.info("Requesting ONS CSV: %s", url)
        resp = requests.get(url, timeout=30)
        logger.info("ONS response status: %s", resp.status_code)
        resp.raise_for_status()

        raw = pd.read_csv(StringIO(resp.text))

        # ------------------------------------------------------------------
        # ONS generator CSVs are quirky:
        # - Often the first column is called "Title" and contains BOTH metadata rows
        #   (e.g., "CDID", "PreUnit", "Source") and the actual time periods.
        # - The value column header is usually a long human-readable series name.
        # - Some downloads include non-monthly rows (e.g., just "2025" as an annual value).
        #
        # Goal:
        # - Extract a clean MONTHLY time series: index=Timestamp (first of month), column=cpi_index
        # - Drop metadata + non-monthly rows
        # ------------------------------------------------------------------

        # 1) Date column candidate
        date_col = None
        for c in ("Date", "date", "Period", "period", "Time", "time"):
            if c in raw.columns:
                date_col = c
                break
        if date_col is None:
            date_col = raw.columns[0]

        # 2) Value column candidate
        value_col = None
        for c in ("Value", "value", "Observation", "observation"):
            if c in raw.columns:
                value_col = c
                break
        if value_col is None:
            value_col = raw.columns[-1]

        df = raw[[date_col, value_col]].copy()

        # Clean strings + coerce numeric values
        df[date_col] = (
            df[date_col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        )
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        # Prefer strictly monthly rows when the date column is the typical "Title" field.
        # Monthly rows usually look like "YYYY MON" (e.g., "2026 JAN").
        # This avoids annual/quarterly rows like "2025" or metadata labels.
        monthly_mask = df[date_col].str.match(r"^\d{4}\s+[A-Za-z]{3}$", na=False)
        if monthly_mask.any():
            df = df.loc[monthly_mask].copy()
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format="%Y %b")
        else:
            # Flexible parsing (drops metadata rows automatically via coercion)
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

            # Second attempt (helps with some ONS outputs like "2026 JAN")
            missing = parsed_dates.isna() & df[date_col].notna()
            if missing.any():
                parsed_dates_2 = pd.to_datetime(df.loc[missing, date_col], errors="coerce", format="%Y %b")
                parsed_dates.loc[missing] = parsed_dates_2

        df["date"] = parsed_dates
        df = df.dropna(subset=["date", value_col])

        df = df.rename(columns={value_col: "cpi_index"})
        df = df.set_index("date").sort_index()

        # If ONS includes duplicate timestamps, keep the last occurrence
        df = df[~df.index.duplicated(keep="last")]

        # Return only the clean numeric series (avoid leaking the raw string column)
        df = df[["cpi_index"]]

        logger.info(
            "Fetched %s rows from ONS (CSV generator) (%s/%s)",
            len(df),
            series_id.upper(),
            dataset_id.upper(),
        )
        return df.loc[start:end]

    except Exception:
        logger.exception("ONS fetch error")
        return None


def _compute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with cpi_index, compute all derived columns.
    If cpi_yoy is already present (ONS source), skip the YoY calc.
    """
    # YoY % change (if not already present)
    if "cpi_yoy" not in df.columns or df["cpi_yoy"].isna().all():
        df["cpi_yoy"] = df["cpi_index"].pct_change(12) * 100

    # MoM % change
    if "cpi_index" in df.columns and df["cpi_index"].notna().any():
        df["cpi_mom"] = df["cpi_index"].pct_change(1) * 100
    else:
        df["cpi_mom"] = df["cpi_yoy"].diff()

    # ----------------------------------------------------------------
    # 1st derivative: 12-month momentum (is the trend rising or falling?)
    # Uses EWM for noise reduction — Neville's approach is rolling but
    # EWM is more responsive to recent regime changes.
    # ----------------------------------------------------------------
    smoothing = REGIME_CFG.get("smoothing", "ewm")
    span      = REGIME_CFG.get("ewm_span", 6)
    window    = REGIME_CFG.get("momentum_window", 12)

    if smoothing == "ewm":
        df["trend"] = df["cpi_yoy"].ewm(span=span, adjust=False).mean()
    else:
        df["trend"] = df["cpi_yoy"].rolling(window).mean()

    df["trend_direction"] = df["trend"].diff(1)  # positive = rising

    # ----------------------------------------------------------------
    # 2nd derivative: acceleration over last 3 months
    # Key for distinguishing "Fire" (rising trend) vs "Boom" (still high
    # but decelerating) — the Neville framework's most nuanced signal.
    # ----------------------------------------------------------------
    accel_window = REGIME_CFG.get("acceleration_window", 3)
    df["acceleration"] = df["trend"].diff(accel_window)

    # ----------------------------------------------------------------
    # Regime flag: is YoY above the Neville 5% threshold?
    # ----------------------------------------------------------------
    threshold = REGIME_CFG.get("inflation_threshold", 5.0)
    df["above_threshold"] = df["cpi_yoy"] > threshold

    return df


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    df = get_uk_cpi(start="2000-01-01")
    print(df.tail(24).to_string())
    print(f"\nCurrent regime flag: {'🔥 ABOVE threshold' if df['above_threshold'].iloc[-1] else '❄️  Below threshold'}")
    print(f"Latest YoY CPI:  {df['cpi_yoy'].iloc[-1]:.2f}%")
    print(f"Trend:           {df['trend'].iloc[-1]:.2f}%")
    print(f"Acceleration:    {df['acceleration'].iloc[-1]:+.3f}")