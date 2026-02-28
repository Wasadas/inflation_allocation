"""
trend_following/momentum_signals.py

Implements Time-Series Momentum (TSMOM) — the "Holy Grail" asset in
Neville et al. (2021). The paper shows that trend-following CTAs
deliver ~25% annualised real returns during Inflationary Fire, with
an ~80% hit rate — outperforming all other asset classes.

Why it works in Fire regimes:
    High and rising inflation creates PERSISTENT price trends:
    - Commodities trend up
    - Nominal bonds trend down
    - Equities trend down (real earnings decay)
    A 12-month TSMOM signal systematically captures these extended
    directional moves, generating positive real returns even when
    traditional assets fail.

Implementation:
    For each asset, compute a signal based on the sign of the
    12-month return. Scale position size by inverse volatility.
    The aggregate of all signals is a synthetic CTA return stream.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

TF_CFG   = CONFIG["trend_following"]
CTA_MODE = TF_CFG.get("cta_mode", "synthetic")   # "etf" | "synthetic"


# ------------------------------------------------------------------
# Core Signal
# ------------------------------------------------------------------

def compute_tsmom_signals(
    returns: pd.DataFrame,
    lookback_months: int | None = None,
    vol_target: float | None = None,
    vol_window: int = 36,
) -> pd.DataFrame:
    """
    Time-Series Momentum signals for each asset.

    For each asset at each date:
        signal_t = sign(cumulative return over lookback period)
        position_t = signal_t × (vol_target / rolling_vol_t)

    Parameters
    ----------
    returns        : DataFrame of monthly returns (not real)
    lookback_months: Lookback for signal (default: 12 from config)
    vol_target     : Annualised vol target per position (default: from config)
    vol_window     : Rolling window for volatility estimation

    Returns
    -------
    DataFrame of TSMOM position sizes (same shape as returns)
    """
    lookback   = lookback_months or TF_CFG["lookback_months"]
    vol_tgt    = vol_target or TF_CFG["vol_target_per_signal"]

    # ---- 1. Raw momentum signal: sign of N-month cumulative return ----
    cum_ret   = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    raw_signal = np.sign(cum_ret)

    # ---- 2. Volatility scaling (annualised) ----
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(12)
    rolling_vol = rolling_vol.replace(0, np.nan)

    # ---- 3. Vol-scaled positions ----
    positions = raw_signal * (vol_tgt / rolling_vol)

    # ---- 4. Cap positions at ±1 (100% long or short per asset) ----
    positions = positions.clip(-1, 1)

    return positions


def compute_cta_proxy_returns(
    returns: pd.DataFrame,
    positions: pd.DataFrame | None = None,
    lookback_months: int | None = None,
) -> pd.Series:
    """
    Synthetic CTA return stream: the equal-weighted average of
    TSMOM positions applied to each asset's subsequent return.

    This is equivalent to running a diversified trend-following
    fund across all asset classes.

    Returns
    -------
    Series of monthly CTA proxy returns
    """
    if positions is None:
        positions = compute_tsmom_signals(returns, lookback_months)

    # Positions at t, applied to returns at t+1 (no look-ahead)
    pos_lagged = positions.shift(1)

    # Weight each position equally
    n_assets    = pos_lagged.notna().sum(axis=1).replace(0, np.nan)
    asset_rets  = (pos_lagged * returns).sum(axis=1)
    cta_returns = asset_rets / n_assets

    cta_returns.name = "CTA_PROXY"
    return cta_returns


def compute_trend_strength(
    returns: pd.DataFrame,
    lookback_months: int | None = None,
) -> pd.DataFrame:
    """
    Trend strength metric: the absolute value of the N-month
    cumulative return, normalised by volatility.

    High values indicate strong, persistent trends — the condition
    under which CTA strategies are most profitable.

    Returns
    -------
    DataFrame of trend strength scores (higher = stronger trend)
    """
    lookback = lookback_months or TF_CFG["lookback_months"]
    cum_ret  = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    vol      = returns.rolling(lookback).std() * np.sqrt(12)

    strength = (cum_ret.abs() / vol).replace([np.inf, -np.inf], np.nan)
    strength.columns = [f"{c}_trend_strength" for c in strength.columns]
    return strength


# ------------------------------------------------------------------
# Moving Average Crossover (simpler alternative signal)
# ------------------------------------------------------------------

def compute_ma_crossover(
    prices: pd.DataFrame,
    short_window: int = 3,
    long_window: int = 12,
) -> pd.DataFrame:
    """
    Moving Average Crossover signal: long when short MA > long MA.

    This is the simplest CTA signal and was used in early Neville
    robustness checks. TSMOM is preferred but this is included
    as a sanity-check / alternative.

    Returns
    -------
    DataFrame of signals: +1 (long), -1 (short), 0 (flat)
    """
    short_ma = prices.rolling(short_window).mean()
    long_ma  = prices.rolling(long_window).mean()
    signal   = np.sign(short_ma - long_ma)
    return signal.rename(columns=lambda c: f"{c}_ma_signal")


# ------------------------------------------------------------------
# Signal Diagnostics
# ------------------------------------------------------------------

def signal_stats(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    classified_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyse how signal performance varies by inflation regime.
    Returns mean position and hit rate by regime for each asset.
    """
    regime = classified_df["regime"].reindex(positions.index, method="ffill")
    pnl    = (positions.shift(1) * returns)

    rows = []
    for reg in regime.unique():
        mask = regime == reg
        for asset in positions.columns:
            pos_mean = positions.loc[mask, asset].mean()
            hit_rate = (pnl.loc[mask, asset] > 0).mean() * 100
            avg_ret  = pnl.loc[mask, asset].mean() * 1200  # annualised %

            rows.append({
                "asset":           asset,
                "regime":          reg,
                "mean_position":   round(pos_mean, 3),
                "hit_rate_%":      round(hit_rate, 1),
                "avg_ann_ret_%":   round(avg_ret, 2),
            })

    return pd.DataFrame(rows).sort_values(["regime", "avg_ann_ret_%"], ascending=[True, False])


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    from data_ingestion.asset_prices import get_all_asset_prices, get_returns
    from data_ingestion.cpi_handler import get_uk_cpi
    from regime_engine.classifier import RegimeClassifier

    print("Fetching data...")
    prices  = get_all_asset_prices(start="2005-01-01")
    rets    = get_returns(prices)
    cpi     = get_uk_cpi(start="2005-01-01")
    regimes = RegimeClassifier().classify(cpi)

    positions  = compute_tsmom_signals(rets)
    cta_stream = compute_cta_proxy_returns(rets, positions)

    print("\n--- CTA Proxy Returns (last 12 months) ---")
    print((cta_stream.tail(12) * 100).round(2).to_string())

    ann = (1 + cta_stream).prod() ** (12 / len(cta_stream)) - 1
    print(f"\nAnnualised CTA return: {ann*100:.2f}%")

    print("\n--- Signal Stats by Regime ---")
    stats = signal_stats(positions, rets, regimes)
    print(stats.to_string(index=False))