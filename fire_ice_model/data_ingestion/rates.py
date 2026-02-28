"""
data_ingestion/rates.py

Fetches UK risk-free rate (BoE bank rate, FRED BOERUKM) for use in
Sharpe and Sortino calculations. Using a zero risk-free rate in a
high-rate environment overstates the strategy's excess return over cash.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from io import StringIO
import requests

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_CFG = CONFIG["data"]
CACHE_DIR = Path(DATA_CFG["cache_dir"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_uk_risk_free_rate(
    start: str,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch BoE bank rate (FRED BOERUKM) and return as monthly decimal rate.

    Used for Sharpe and Sortino so reported ratios reflect excess return
    over cash (honest view of active alpha in a high-rate environment).

    Parameters
    ----------
    start : str   ISO date string
    end   : str   ISO date string (default: today)
    use_cache : bool   If True, use cached series when fresh

    Returns
    -------
    pd.Series   Index = month-end dates, values = monthly risk-free rate in decimal
                (e.g. 0.004 for 0.4% per month). Annual rate is converted via
                (1 + r_annual) ** (1/12) - 1.
    """
    from datetime import datetime
    end = end or datetime.today().strftime("%Y-%m-%d")
    series_id = (DATA_CFG.get("fred_series") or {}).get("uk_base_rate", "BOERUKM")

    cache_file = CACHE_DIR / "uk_rf.parquet"
    if use_cache and cache_file.exists():
        try:
            s = pd.read_parquet(cache_file)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            last = s.index.max()
            if (pd.Timestamp.today() - last).days < 35 and s.index.min() <= pd.Timestamp(start):
                logger.info("Loaded UK risk-free rate from cache (%s rows)", len(s))
                return s.loc[start:end]
        except Exception as e:
            logger.debug("Cache read failed: %s", e)

    try:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start}&coed={end}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
        date_col = "DATE" if "DATE" in raw.columns else raw.columns[0]
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.set_index(date_col).sort_index()
        col = series_id if series_id in raw.columns else [c for c in raw.columns if c != date_col][0]
        r_annual = pd.to_numeric(raw[col], errors="coerce").dropna()
        if r_annual.empty:
            logger.warning("No valid BoE rate data from FRED (%s)", series_id)
            return pd.Series(dtype=float)
        # FRED reports rate in % (e.g. 5.25); convert to decimal and resample to month-end
        r_annual = r_annual / 100.0
        r_annual = r_annual[~r_annual.index.duplicated(keep="last")]
        try:
            r_me = r_annual.resample("ME").last().ffill()
        except TypeError:
            r_me = r_annual.resample("M").last().ffill()
        r_monthly = (1 + r_me) ** (1 / 12) - 1
        r_monthly.to_frame("rf_monthly").to_parquet(cache_file)
        logger.info("Fetched and cached UK risk-free rate (%s rows)", len(r_monthly))
        return r_monthly.loc[start:end]
    except Exception as e:
        logger.warning("Could not fetch UK risk-free rate: %s. Sharpe/Sortino will use 0%% RF.", e)
        return pd.Series(dtype=float)
