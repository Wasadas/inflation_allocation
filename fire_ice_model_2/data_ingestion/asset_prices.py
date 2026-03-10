"""
data_ingestion/asset_prices.py

Fetches price data for the Fire & Ice asset universe (UK GBP, London-listed)
and computes monthly returns for backtesting and allocation.

Uses config.yaml → assets for the list of tickers; falls back to allocation
keys if assets is not defined. Caches raw prices to DATA_CFG["cache_dir"].

CTA proxy: Config specifies a primary ticker (e.g. DBMG.L). When no Bloomberg
data is available we officially use a synthetic series (DBMF + GBP/USD hedge)
under the same column name; see CTA_SYNTHETIC_DISCLAIMER.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_CFG = CONFIG["data"]
# Resolve cache_dir relative to project root (parent of fire_ice_model_2) so it works regardless of cwd
_package_root = CONFIG_PATH.resolve().parent
_project_root = _package_root.parent
_cache_dir_cfg = DATA_CFG["cache_dir"]
CACHE_DIR = (_project_root / _cache_dir_cfg).resolve() if not Path(_cache_dir_cfg).is_absolute() else Path(_cache_dir_cfg)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CTA/trend proxy: config lists a primary ticker (e.g. DBMG.L); when no Bloomberg data is available
# we use DBMF (USD) + GBP/USD to build a synthetic GBP-hedged series under the same column name.
CTA_FALLBACK_USD_TICKER = "DBMF"   # USD-denominated; we hedge with GBPUSD to produce GBP series
CTA_GBPUSD_TICKER = "GBPUSD=X"     # FX: USD per GBP (so price up = GBP stronger)

# Human-readable disclaimer when CTA proxy is synthetic (no BBG/live data).
CTA_SYNTHETIC_DISCLAIMER = (
    "No Bloomberg data available. CTA proxy uses synthetic DBMF + GBP/USD hedge (not live listed CTA)."
)


def _fetch_single_ticker_series(
    yf,
    ticker: str,
    start: str,
    end: str,
    resample_month_end: bool = True,
) -> pd.Series | None:
    """Download one ticker and return its close series, at month-end if asked."""
    try:
        fd = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if fd.empty or "Close" not in fd.columns:
            return None
        close = fd["Close"]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        if resample_month_end:
            try:
                close = close.resample("ME").last()
            except TypeError:
                close = close.resample("M").last()
        return close
    except Exception as e:
        logger.debug("Download %s failed: %s", ticker, e)
        return None


def _synthetic_gbp_hedged_series(
    usd_series: pd.Series,
    gbp_usd_series: pd.Series,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Convert USD-denominated series to synthetic GBP-hedged: price_GBP = price_USD / GBPUSD.
    Align to target_index (month-end) and forward/back fill for missing dates.
    """
    common = usd_series.index.intersection(gbp_usd_series.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    usd = usd_series.reindex(common).ffill().bfill()
    gbp = gbp_usd_series.reindex(common).ffill().bfill()
    gbp = gbp.replace(0, np.nan).ffill().bfill()
    synthetic = usd / gbp
    return synthetic.reindex(target_index).ffill().bfill()


def _get_cta_proxy_ticker() -> str | None:
    """Primary CTA ticker from config (assets.trend.cta_proxy)."""
    assets = CONFIG.get("assets") or {}
    trend = assets.get("trend") or {}
    if isinstance(trend, dict):
        return (trend.get("cta_proxy") or "").strip() or None
    return None


def _tickers_from_config() -> list[str]:
    """Collect the tickers we care about from config.assets or, as a fallback, from allocation."""
    tickers = []
    assets = CONFIG.get("assets") or {}
    for category in assets.values():
        if isinstance(category, dict):
            for v in category.values():
                if isinstance(v, str) and v.strip():
                    tickers.append(v.strip())
    if tickers:
        return list(dict.fromkeys(tickers))
    # Fallback: union of allocation keys
    for regime_weights in (CONFIG.get("allocation") or {}).values():
        if isinstance(regime_weights, dict):
            tickers.extend(regime_weights.keys())
    return list(dict.fromkeys(tickers))


def get_all_asset_prices(
    start: str = "2005-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for the configured asset universe.

    Returns a DataFrame with DatetimeIndex (month-end) and columns = ticker symbols.
    Prices are resampled to month-end for consistency with monthly backtests.
    """
    import yfinance as yf

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = _tickers_from_config()
    if not tickers:
        raise ValueError("No tickers found in config (assets or allocation).")

    cache_file = CACHE_DIR / "asset_prices.parquet"
    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Only use cache if it covers the requested range
        if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end) - pd.offsets.MonthEnd(0):
            logger.info("Loaded asset prices from cache (%s rows, %s tickers)", len(df), len(df.columns))
            cta_primary = _get_cta_proxy_ticker()
            cta_col = df.get(cta_primary) if cta_primary else None
            if cta_primary and cta_col is not None and not cta_col.empty and bool(cta_col.notna().any(axis=None)):
                logger.info("CTA proxy: live data for %s (config cta_proxy) retrieved in GBP.", cta_primary)
            return df.loc[start:end].copy()

    logger.info("Downloading prices for %s from %s to %s", tickers, start, end)
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("yfinance returned no data; check tickers and date range.")

    # Normalise to DataFrame: index=date, columns=tickers, values=close price
    if len(tickers) == 1:
        close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
        prices = pd.DataFrame(close.values, index=close.index, columns=[tickers[0]])
    elif isinstance(data.columns, pd.MultiIndex):
        # Default group_by: (Close, ISF.L), (Close, IGLT.L) -> take Close row
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", axis=1, level=1).copy()
        prices.columns = [c if isinstance(c, str) else c[-1] for c in prices.columns]
    else:
        prices = data[["Close"]].copy() if "Close" in data.columns else data.iloc[:, :1].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    # Resample to month-end for monthly backtest alignment
    try:
        prices = prices.resample("ME").last().dropna(how="all")
    except TypeError:
        prices = prices.resample("M").last().dropna(how="all")
    # Keep only tickers we got; preserve order
    prices = prices[[c for c in tickers if c in prices.columns]].copy()

    # CTA proxy: try primary ticker from config; if missing or insufficient history, use synthetic (DBMF + GBP/USD).
    cta_primary = _get_cta_proxy_ticker()
    min_cta_months = 12  # require at least 12 months of data to treat as "live" data
    cta_insufficient = (
        cta_primary
        and (
            cta_primary not in prices.columns
            or prices[cta_primary].isna().all()
            or prices[cta_primary].notna().sum() < min_cta_months
        )
    )
    if cta_insufficient:
        # Official synthetic path: DBMF (USD) + GBPUSD => synthetic GBP-hedged series (no BBG data)
        dbmf = _fetch_single_ticker_series(yf, CTA_FALLBACK_USD_TICKER, start, end)
        gbp_usd = _fetch_single_ticker_series(yf, CTA_GBPUSD_TICKER, start, end)
        if dbmf is not None and gbp_usd is not None:
            dbmf_ok = not dbmf.empty and bool(dbmf.notna().any(axis=None))
            gbp_ok = not gbp_usd.empty and bool(gbp_usd.notna().any(axis=None))
            if dbmf_ok and gbp_ok:
                synthetic = _synthetic_gbp_hedged_series(dbmf, gbp_usd, prices.index)
                if not synthetic.empty:
                    aligned = synthetic.reindex(prices.index).ffill().bfill()
                    prices[cta_primary] = np.asarray(aligned, dtype=float)
                    logger.warning("%s %s", CTA_SYNTHETIC_DISCLAIMER, "Filled under config cta_proxy label.")
                else:
                    logger.warning(
                        "%s Synthetic series could not be built; FIRE regime will have no CTA data.",
                        CTA_SYNTHETIC_DISCLAIMER,
                    )
            else:
                logger.warning(
                    "%s DBMF or GBPUSD could not be retrieved; FIRE regime will have no CTA data.",
                    CTA_SYNTHETIC_DISCLAIMER,
                )
        else:
            logger.warning(
                "%s DBMF or GBPUSD could not be retrieved; FIRE regime will have no CTA data.",
                CTA_SYNTHETIC_DISCLAIMER,
            )
    elif cta_primary and cta_primary in prices.columns and not prices[cta_primary].empty and bool(prices[cta_primary].notna().any(axis=None)):
        logger.info("CTA proxy: live data for %s (config cta_proxy) retrieved in GBP.", cta_primary)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_file)
    logger.info("Cached asset prices (%s rows)", len(prices))
    return prices.loc[start:end].copy()


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly returns from a price DataFrame.

    Parameters
    ----------
    prices : DataFrame with DatetimeIndex and ticker columns (e.g. from get_all_asset_prices)

    Returns
    -------
    DataFrame of simple monthly returns (same index and columns as prices, first row NaN).

    Data cleaning:
        To guard against obvious data errors (e.g. split/mapping issues that create
        ~-99% one-month moves in otherwise stable ETFs), any single-month return
        below -40% is treated as bad data. The corresponding price is set to NaN
        and forward-filled, and returns are recomputed. This caps the impact of
        spurious collapses while preserving genuine large drawdowns.
    """
    rets = prices.pct_change()

    # Identify extreme negative moves that are almost certainly data errors.
    bad_mask = rets < -0.40
    if bad_mask.any().any():
        logger.warning(
            "Capping %s extreme monthly price moves (< -40%%) as NaN and forward-filling before computing returns.",
            int(bad_mask.sum().sum()),
        )
        clean_prices = prices.copy()
        clean_prices[bad_mask] = np.nan
        clean_prices = clean_prices.ffill()
        rets = clean_prices.pct_change()

    return rets.dropna(how="all")
