"""
backtest/engine.py

Vectorised backtest engine for the Fire & Ice model.

Simulates monthly rebalancing, applies transaction costs, and
produces a full return history decomposed by regime.

Benchmark: 60/40 (ISF.L / IGLT.L)
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

BT_CFG = CONFIG["backtest"]
TC_BPS = BT_CFG["transaction_cost_bps"]


def _get_cta_proxy_ticker_from_config() -> str | None:
    """Config cta_proxy ticker (e.g. DBMG.L); used to attach disclaimer when that column is dropped."""
    trend = (CONFIG.get("assets") or {}).get("trend")
    if isinstance(trend, dict):
        t = (trend.get("cta_proxy") or "").strip()
        return t or None
    return None


def _log_cta_disclaimer_if_dropped(dropped: list) -> None:
    """If the CTA proxy column was dropped (all-NaN returns), log the official disclaimer."""
    try:
        from data_ingestion.asset_prices import CTA_SYNTHETIC_DISCLAIMER
    except ImportError:
        return
    cta = _get_cta_proxy_ticker_from_config()
    if cta and cta in dropped:
        logger.info("%s", CTA_SYNTHETIC_DISCLAIMER)


def _to_month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert DatetimeIndex to month-end so returns (month-end) and weights (first-of-month from CPI) align."""
    p = idx.to_period("M")
    try:
        return p.to_timestamp("ME")
    except (TypeError, ValueError):
        return p.to_timestamp(how="end")


# ------------------------------------------------------------------
# Backtest Engine
# ------------------------------------------------------------------

class BacktestEngine:
    """
    Simulates the Fire & Ice portfolio from a history of target weights.

    Parameters
    ----------
    returns      : DataFrame of monthly asset returns
    weights      : DataFrame of monthly target weights (from allocation_logic)
    classified_df: DataFrame with 'regime' column
    tc_bps       : Transaction cost per leg in basis points
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        classified_df: pd.DataFrame,
        tc_bps: int = TC_BPS,
    ):
        self.returns        = returns
        self.weights        = weights
        self.classified_df  = classified_df
        self.tc_bps         = tc_bps

    def run(self) -> pd.DataFrame:
        """
        Run the backtest. Returns a DataFrame with columns:
            portfolio_return   : gross monthly return
            net_return         : after transaction costs
            transaction_cost   : cost incurred this period
            regime             : Neville regime at each date
            nav                : cumulative NAV (start = 1.0)
        """
        # Drop all-NaN columns to avoid ValueError in downstream ops (e.g. nanargmax / idxmax)
        returns = self.returns.copy()
        weights = self.weights.copy()
        valid_cols = returns.columns[returns.notna().any(axis=0)].tolist()
        if len(valid_cols) < len(returns.columns):
            dropped = returns.columns.difference(valid_cols).tolist()
            logger.info("Dropping all-NaN return columns before optimization: %s", dropped)
            _log_cta_disclaimer_if_dropped(dropped)
            returns = returns[valid_cols]
            weights = weights.reindex(columns=valid_cols).fillna(0)

        # Regime column: prefer dominant_regime (probability mode) if present
        regime_col = "dominant_regime" if "dominant_regime" in self.classified_df.columns else "regime"
        regime_series = self.classified_df[regime_col]

        # Normalize all indices to month-end so returns (month-end) and weights/CPI (first-of-month) align
        ret_idx_me   = _to_month_end(pd.DatetimeIndex(returns.index))
        wt_idx_me    = _to_month_end(pd.DatetimeIndex(weights.index))
        reg_idx_me   = _to_month_end(pd.DatetimeIndex(regime_series.index))

        # Reindex returns to month-end (no-op if already month-end); weights/regime get month-end index
        returns_me   = returns.copy()
        returns_me.index = ret_idx_me
        weights_me   = weights.copy()
        weights_me.index = wt_idx_me
        regime_me    = regime_series.copy()
        regime_me.index = reg_idx_me

        common_idx   = returns_me.index.intersection(weights_me.index).intersection(regime_me.index).sort_values()
        if len(common_idx) == 0:
            raise ValueError(
                "No common dates between returns and weights; check that return and CPI/weight "
                "indices align (e.g. month-end vs first-of-month)."
            )

        rets   = returns_me.loc[common_idx]
        wts    = weights_me.reindex(common_idx, method="ffill").fillna(0)
        regime = regime_me.reindex(common_idx, method="ffill")

        results  = []
        prev_wts = pd.Series(0.0, index=wts.columns)

        for date in common_idx:
            w      = wts.loc[date]
            r      = rets.loc[date]

            # Align weights to return columns so missing assets (e.g. delisted WTMA.L) don't produce NaN
            common = w.index.intersection(r.index)
            w_aligned = w.reindex(common).fillna(0)
            total = w_aligned.sum()
            if total > 0:
                w_aligned = w_aligned / total
            else:
                w_aligned = pd.Series(0.0, index=common)

            # Portfolio return (dot product of weights and returns)
            gross  = float((w_aligned * r.reindex(common).fillna(0)).sum())

            # Transaction cost: turnover × tc_bps / 10000
            turnover = float((w - prev_wts).abs().sum()) / 2
            cost     = turnover * self.tc_bps / 10_000
            net      = gross - cost

            results.append({
                "date":               date,
                "portfolio_return":   gross,
                "net_return":         net,
                "transaction_cost":   cost,
                "regime":             regime.get(date, None),
            })
            prev_wts = w

        df = pd.DataFrame(results).set_index("date")
        df["nav"] = (1 + df["net_return"]).cumprod()
        return df

    def run_benchmark(
        self,
        equity_ticker: str = "ISF.L",
        bond_ticker: str = "IGLT.L",
        equity_weight: float = 0.60,
    ) -> pd.DataFrame:
        """
        Run a static 60/40 benchmark for comparison.
        """
        if equity_ticker not in self.returns.columns:
            logger.warning("Benchmark equity %s not in returns", equity_ticker)
            return pd.DataFrame()
        if bond_ticker not in self.returns.columns:
            logger.warning("Benchmark bond %s not in returns", bond_ticker)
            return pd.DataFrame()

        bond_weight = 1 - equity_weight
        bm = (
            self.returns[equity_ticker] * equity_weight
            + self.returns[bond_ticker] * bond_weight
        )
        bm = bm.dropna()

        return pd.DataFrame({
            "benchmark_return": bm,
            "benchmark_nav":    (1 + bm).cumprod(),
        })


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(results: pd.DataFrame, col: str = "net_return") -> dict:
    """
    Compute standard performance metrics for a return series.
    """
    rets = results[col].dropna()
    nav  = (1 + rets).cumprod()

    # Annualised return
    n_years  = len(rets) / 12
    ann_ret  = (nav.iloc[-1]) ** (1 / n_years) - 1

    # Annualised volatility
    ann_vol  = rets.std() * np.sqrt(12)

    # Sharpe (assume 0% risk-free for simplicity; use BoE rate for refinement)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    roll_max = nav.cummax()
    dd       = (nav - roll_max) / roll_max
    max_dd   = dd.min()

    # Calmar ratio
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Sortino (downside deviation)
    neg_rets  = rets[rets < 0]
    down_vol  = neg_rets.std() * np.sqrt(12)
    sortino   = ann_ret / down_vol if down_vol > 0 else np.nan

    return {
        "ann_return_%":    round(ann_ret * 100, 2),
        "ann_vol_%":       round(ann_vol * 100, 2),
        "sharpe":          round(sharpe, 3),
        "sortino":         round(sortino, 3),
        "calmar":          round(calmar, 3),
        "max_drawdown_%":  round(max_dd * 100, 2),
        "n_months":        len(rets),
    }


def compute_regime_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Break down performance metrics by Neville regime.
    """
    rows = []
    for regime in results["regime"].dropna().unique():
        mask  = results["regime"] == regime
        sub   = results.loc[mask]
        m     = compute_metrics(sub)
        m["regime"] = regime
        rows.append(m)

    return pd.DataFrame(rows).set_index("regime")


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
    from allocation_logic.regime_weighting import build_weight_history

    start_date = BT_CFG.get("start_date", "2005-01-01")
    clean_window_start = BT_CFG.get("clean_window_start")
    if clean_window_start:
        effective_start = max(
            pd.Timestamp(start_date),
            pd.Timestamp(clean_window_start),
        ).strftime("%Y-%m-%d")
    else:
        effective_start = start_date

    print("Running backtest pipeline...")
    prices  = get_all_asset_prices(start=effective_start)
    rets    = get_returns(prices)
    cpi     = get_uk_cpi(start=effective_start)
    regimes = RegimeClassifier().classify(cpi)
    weights, rp_stats = build_weight_history(regimes, rets)

    engine  = BacktestEngine(rets, weights, regimes)
    results = engine.run()
    bm      = engine.run_benchmark()

    print("\n--- Portfolio Metrics ---")
    print(compute_metrics(results))

    if not bm.empty:
        bm_metrics = compute_metrics(bm, col="benchmark_return")
        print("\n--- 60/40 Benchmark Metrics ---")
        print(bm_metrics)

    print("\n--- Performance by Regime ---")
    print(compute_regime_metrics(results).to_string())

    n_req = rp_stats.get("n_risk_parity_requested", 0)
    n_skip = rp_stats.get("n_risk_parity_skipped", 0)
    if n_req > 0:
        pct_applied = round((n_req - n_skip) / n_req * 100)
        pct_base = round(n_skip / n_req * 100)
        print(
            "\nOptimization Note: Risk-parity weighting applied to %s%% of periods; "
            "%s%% utilized strategic base weights due to initial data inception constraints."
            % (pct_applied, pct_base)
        )