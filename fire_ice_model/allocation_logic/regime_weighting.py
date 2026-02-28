
"""
allocation_logic/regime_weighting.py

Converts the Neville regime classification into portfolio weights.

Two output modes — both read from config.yaml:

    use_regime_probability: false  →  HARD SWITCH
        Weights snap to the allocation block for the active regime.
        Fast and transparent, but causes churn near threshold.

    use_regime_probability: true   →  PROBABILITY BLEND
        Weights are a continuous blend of all four regime weight sets,
        weighted by their sigmoid probabilities from classifier.py:

            w_t = P(fire)*w_fire + P(boom)*w_boom
                + P(ice)*w_ice   + P(recovery)*w_recovery

        This eliminates discrete jumps at the threshold and naturally
        reduces turnover when the regime signal is ambiguous.

Risk Parity (optional, config: risk_parity.enabled):
    The blended or hard weights are used as a STARTING POINT for a
    risk parity optimiser that scales positions so each asset
    contributes equally to portfolio volatility.

CTA mode (config: trend_following.cta_mode):
    "etf"       → use assets.trend.cta_proxy ticker (WTMA.L)
    "synthetic" → CTA weight is allocated to the synthetic trend signal
                  built in trend_following/momentum_signals.py
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------
# Reads package config from fire_ice_model/config.yaml
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f) or {}

ALLOC_CFG  = CONFIG.get("allocation", {})
RP_CFG     = CONFIG.get("risk_parity", {})
REGIME_CFG = CONFIG.get("regime", {})
TF_CFG     = CONFIG.get("trend_following", {})

if not ALLOC_CFG:
    logger.warning(
        "No 'allocation' block found in %s. Regime weighting will fail until it is added.",
        CONFIG_PATH,
    )


# ------------------------------------------------------------------
# Allocation key → ticker mapping (config uses logical names; returns use tickers)
# ------------------------------------------------------------------

def _allocation_key_to_ticker_map() -> dict:
    """
    Build mapping from allocation config keys (logical names) to ticker symbols.
    Walks config["assets"] (equities, bonds, real_assets, trend). Keys not in
    the map are treated as tickers (e.g. allocation written with ISF.L).
    """
    out = {}
    assets = CONFIG.get("assets") or {}
    for category in assets.values():
        if isinstance(category, dict):
            for logical_name, ticker in category.items():
                if isinstance(ticker, str) and ticker.strip():
                    out[logical_name.strip()] = ticker.strip()
    return out


def _to_ticker_weights(raw_weights: dict) -> pd.Series:
    """
    Convert allocation weights (keys = logical names or tickers) to a
    Series indexed by ticker. If two allocation keys map to the same ticker,
    their weights are summed.
    """
    mapping = _allocation_key_to_ticker_map()
    by_ticker = {}
    for key, weight in raw_weights.items():
        if not isinstance(weight, (int, float)):
            continue
        ticker = mapping.get(key, key)
        by_ticker[ticker] = by_ticker.get(ticker, 0) + float(weight)
    return pd.Series(by_ticker, dtype=float).fillna(0)


def _align_weights_to_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """
    Restrict weights to assets that exist in returns and renormalize.
    Handles missing data (e.g. delisted ticker) so backtest never sees NaN.
    """
    if returns is None or returns.columns is None:
        return weights
    common = weights.index.intersection(returns.columns)
    w = weights.reindex(common).fillna(0)
    total = w.sum()
    if total <= 0:
        return w
    return (w / total).round(4)


# ------------------------------------------------------------------
# Public Interface
# ------------------------------------------------------------------

def get_target_weights(
    regime: str,
    returns: Optional[pd.DataFrame] = None,
    use_risk_parity: Optional[bool] = None,
    regime_probs: Optional[dict] = None,
) -> pd.Series:
    """
    Returns target portfolio weights for the current regime.

    Parameters
    ----------
    regime          : str   Active regime ("FIRE", "BOOM", "ICE", "RECOVERY").
                            Used directly in hard-switch mode; used as fallback
                            label in probability mode.
    returns         : DataFrame of historical returns (needed for risk parity).
    use_risk_parity : bool  Override config setting if needed.
    regime_probs    : dict  Keys: "prob_fire", "prob_boom", "prob_ice",
                            "prob_recovery". If provided (and config enables
                            use_regime_probability), weights are blended.

    Returns
    -------
    pd.Series  Asset weights summing to 1.0.
    """
    use_rp   = use_risk_parity if use_risk_parity is not None else RP_CFG["enabled"]
    use_prob = REGIME_CFG.get("use_regime_probability", False)

    if use_prob and regime_probs is not None:
        base_weights = _blend_weights(regime_probs)
    else:
        base_weights = _get_hard_weights(regime)

    if use_rp and returns is not None:
        weights = _apply_risk_parity(base_weights, returns)
    else:
        weights = base_weights

    weights = weights / weights.sum()
    # Align to return columns so only assets with data get weight (avoids NaN in backtest)
    if returns is not None:
        weights = _align_weights_to_returns(weights, returns)
    return weights.round(4)


def build_weight_history(
    classified_df: pd.DataFrame,
    returns: pd.DataFrame,
    use_risk_parity: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a full history of target weights at each rebalance date.

    Automatically uses probability blending if classifier produced
    prob_fire / prob_boom / prob_ice / prob_recovery columns.

    Parameters
    ----------
    classified_df : Output of RegimeClassifier.classify()
    returns       : Monthly asset returns (for covariance estimation)

    Returns
    -------
    Tuple of (DataFrame of weights indexed by date, stats dict with
    n_rebalance_dates, n_risk_parity_requested, n_risk_parity_skipped).
    """
    rp_months     = max(RP_CFG.get("lookback_vol_days", 60) // 21, 12)
    has_probs     = "prob_fire" in classified_df.columns
    use_prob      = REGIME_CFG.get("use_regime_probability", False) and has_probs

    weight_rows = []
    n_risk_parity_requested = 0
    n_risk_parity_skipped   = 0

    for date, row in classified_df.iterrows():
        regime = row.get("dominant_regime") or row.get("regime")
        if pd.isna(regime):
            continue

        # Regime probabilities (if available and enabled)
        probs = None
        if use_prob:
            probs = {
                "prob_fire":     float(row.get("prob_fire", 0)),
                "prob_boom":     float(row.get("prob_boom", 0)),
                "prob_ice":      float(row.get("prob_ice", 0)),
                "prob_recovery": float(row.get("prob_recovery", 0)),
            }

        # Historical returns for covariance (no look-ahead)
        hist = returns.loc[:date].tail(rp_months)

        base_weights = _blend_weights(probs) if use_prob and probs else _get_hard_weights(str(regime))
        active = [a for a in base_weights.index if base_weights[a] > 0 and a in hist.columns]

        if use_risk_parity and len(hist) >= 12 and len(active) >= 2:
            n_risk_parity_requested += 1
            w = get_target_weights(str(regime), hist, use_risk_parity=True, regime_probs=probs)
        else:
            if use_risk_parity and len(hist) >= 12 and len(active) < 2:
                n_risk_parity_skipped += 1
            w = get_target_weights(str(regime), use_risk_parity=False, regime_probs=probs)

        w = _align_weights_to_returns(w, returns)
        weight_rows.append({"date": date, **w.to_dict()})

    df = pd.DataFrame(weight_rows).set_index("date")
    stats = {
        "n_rebalance_dates": len(weight_rows),
        "n_risk_parity_requested": n_risk_parity_requested,
        "n_risk_parity_skipped": n_risk_parity_skipped,
    }
    return df.fillna(0), stats


def compute_rebalance_cost(
    prev_weights: pd.Series,
    new_weights: pd.Series,
    tc_bps: Optional[int] = None,
) -> dict:
    """
    Compute trades and estimated transaction cost for a rebalance.

    Returns
    -------
    dict with keys: trades, turnover_pct, est_cost_bps
    """
    tc = tc_bps if tc_bps is not None else CONFIG["backtest"]["transaction_cost_bps"]
    trades   = new_weights - prev_weights
    turnover = trades.abs().sum() / 2        # one-way
    cost_bps = turnover * tc * 2             # round-trip

    return {
        "trades":       trades.round(4),
        "turnover_pct": round(float(turnover) * 100, 2),
        "est_cost_bps": round(float(cost_bps), 1),
    }


# ------------------------------------------------------------------
# Probability blending
# ------------------------------------------------------------------

def _blend_weights(regime_probs: dict) -> pd.Series:
    """
    Compute probability-weighted blend of all four regime weight sets.

        w_t = P(fire)*w_fire + P(boom)*w_boom
            + P(ice)*w_ice   + P(recovery)*w_recovery

    Probabilities are re-normalised here in case of floating-point drift.
    """
    prob_map = {
        "prob_fire":     "fire",
        "prob_boom":     "disinflationary_boom",
        "prob_ice":      "ice",
        "prob_recovery": "reflationary_recovery",
    }

    total = sum(regime_probs.values())
    if total <= 0:
        # Degenerate case — fall back to equal blend
        regime_probs = {k: 0.25 for k in regime_probs}
        total = 1.0

    blended_by_ticker = {}
    for prob_key, alloc_key in prob_map.items():
        p = regime_probs.get(prob_key, 0) / total
        raw = ALLOC_CFG.get(alloc_key, {})
        w = _to_ticker_weights(raw)
        for ticker in w.index:
            blended_by_ticker[ticker] = blended_by_ticker.get(ticker, 0) + p * w[ticker]

    return pd.Series(blended_by_ticker, dtype=float).fillna(0)


# ------------------------------------------------------------------
# Hard weight lookup
# ------------------------------------------------------------------

def _get_hard_weights(regime: str) -> pd.Series:
    """Load the static config weights for a single regime. Returns Series indexed by ticker."""
    regime_map = {
        "FIRE":     "fire",
        "BOOM":     "disinflationary_boom",
        "ICE":      "ice",
        "RECOVERY": "reflationary_recovery",
    }
    key = regime_map.get(str(regime).upper())
    if key is None:
        raise ValueError(f"Unknown regime '{regime}'. Valid: {list(regime_map)}")

    raw = ALLOC_CFG.get(key)
    if not raw:
        raise ValueError(f"No weights found in config.yaml for key: {key}")

    return _to_ticker_weights(raw)


# ------------------------------------------------------------------
# Risk Parity (Cyclical Coordinate Descent)
# ------------------------------------------------------------------

def _apply_risk_parity(
    base_weights: pd.Series,
    returns: pd.DataFrame,
    n_iters: int = 200,
    tol: float = 1e-8,
) -> pd.Series:
    """
    Equal Risk Contribution optimiser.

    Only optimises over assets with non-zero base weight that also
    appear in the returns DataFrame.  Zero-weight assets (e.g., CTA
    in synthetic mode — handled separately by momentum_signals.py)
    are left at zero and the remaining weights are renormalised.
    """
    active = [
        a for a in base_weights.index
        if base_weights[a] > 0 and a in returns.columns
    ]

    if len(active) < 2:
        logger.warning("Risk parity: fewer than 2 active assets — returning base weights.")
        return base_weights

    cov = returns[active].cov() * 12        # annualised covariance
    w   = base_weights[active].values.astype(float)
    w   = w / w.sum()

    w_min = RP_CFG.get("min_weight", 0.02)
    w_max = RP_CFG.get("max_weight", 0.45)

    for _ in range(n_iters):
        w_prev = w.copy()
        sigma  = float(np.sqrt(w @ cov.values @ w))

        for i in range(len(w)):
            rc     = float(cov.values[i] @ w) * w[i] / sigma
            target = sigma / len(w)
            grad   = float(cov.values[i] @ w) / sigma
            step   = (rc - target) / (grad + 1e-10)
            w[i]   = float(np.clip(w[i] - step, w_min, w_max))

        w = w / w.sum()

        if np.max(np.abs(w - w_prev)) < tol:
            break

    # Reconstruct full weight Series
    full = base_weights.copy() * 0.0
    for i, asset in enumerate(active):
        full[asset] = w[i]

    return full


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, force=True)

    from data_ingestion.asset_prices import get_all_asset_prices, get_returns
    from data_ingestion.cpi_handler import get_uk_cpi
    from regime_engine.classifier import RegimeClassifier

    prices  = get_all_asset_prices(start="2010-01-01")
    rets    = get_returns(prices)
    cpi     = get_uk_cpi(start="2010-01-01")
    clf     = RegimeClassifier()
    regimes = clf.classify(cpi)

    current = clf.get_current_regime(regimes)
    print(f"\nCurrent regime: {current['emoji']} {current['regime'].value}")

    probs = {k: current.get(k, 0) for k in
             ("prob_fire", "prob_boom", "prob_ice", "prob_recovery")}

    print("\n--- Probability-blended weights ---")
    w_blend = get_target_weights(str(current["regime"]), rets,
                                  use_risk_parity=False, regime_probs=probs)
    print(w_blend.to_string())

    print("\n--- Risk-parity weights ---")
    w_rp = get_target_weights(str(current["regime"]), rets,
                               use_risk_parity=True, regime_probs=probs)
    print(w_rp.to_string())

    print("\n--- All hard-switch regime weights ---")
    for r in ("FIRE", "BOOM", "ICE", "RECOVERY"):
        w = get_target_weights(r, use_risk_parity=False)
        print(f"\n{r}:")
        for asset, wt in w.sort_values(ascending=False).items():
            print(f"  {asset:<12} {wt*100:5.1f}%")