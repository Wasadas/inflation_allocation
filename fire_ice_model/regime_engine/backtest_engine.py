"""
backtest_engine.py

Monthly backtest for the Fire & Ice portfolio.
Handles position updates, transaction costs and regime tagging so we
can compare the strategy to a simple 60/40 benchmark.

The timing convention is “no look‑ahead”: weights at month‑end T only
use CPI information that would have been available by then, and we
attribute r_T to the weights held at the start of the month.
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


def _verify_fire_cta_weight(results: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Log average CTA proxy weight during FIRE months to verify the 25% allocation is active."""
    cta_ticker = _get_cta_proxy_ticker_from_config()
    if not cta_ticker or cta_ticker not in weights.columns:
        return
    fire_mask = results["regime"] == "FIRE"
    if not fire_mask.any():
        return
    fire_dates = results.index[fire_mask]
    wt_idx_me = _to_month_end(pd.DatetimeIndex(weights.index))
    weights_me = weights.copy()
    weights_me.index = wt_idx_me
    w_fire = weights_me.reindex(fire_dates, method="ffill")[cta_ticker]
    avg_pct = float(w_fire.mean() * 100)
    n_fire = len(fire_dates)
    logger.info(
        "FIRE regime: CTA proxy '%s' average weight = %.1f%% over %s months.",
        cta_ticker, avg_pct, n_fire,
    )


def _inject_synthetic_cta_if_enabled(returns: pd.DataFrame) -> pd.DataFrame:
    """
    When config trend_following.cta_mode is "synthetic", compute the in-model
    TSMOM CTA return series and inject it into returns under the config cta_proxy
    ticker (e.g. DBMG.L). This bypasses the need for an external ticker and
    ensures the FIRE 25% CTA allocation is active.

    Returns
    -------
    DataFrame with synthetic CTA column added (or unchanged if cta_mode != "synthetic").
    """
    tf_cfg = CONFIG.get("trend_following") or {}
    if tf_cfg.get("cta_mode") != "synthetic":
        return returns

    cta_ticker = _get_cta_proxy_ticker_from_config()
    if not cta_ticker:
        return returns

    try:
        from trend_following.momentum import compute_cta_proxy_returns
    except ImportError:
        logger.warning("trend_following.momentum not found; skipping synthetic CTA injection.")
        return returns

    # Build CTA from the configured trend assets that exist in returns (excludes missing DBMG.L)
    trend_assets = tf_cfg.get("assets_to_trend") or []
    rets_for_cta = returns[[c for c in trend_assets if c in returns.columns]]
    if rets_for_cta.empty:
        rets_for_cta = returns

    cta_returns = compute_cta_proxy_returns(rets_for_cta)
    aligned = cta_returns.reindex(returns.index).ffill().bfill().fillna(0.0)
    out = returns.copy()
    out[cta_ticker] = aligned.values
    logger.info(
        "Injected synthetic CTA return series as '%s' (cta_mode=synthetic); "
        "FIRE regime 25%% allocation will use this stream.",
        cta_ticker,
    )
    return out


def _to_month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert DatetimeIndex to month-end so returns (month-end) and weights (first-of-month from CPI) align."""
    p = idx.to_period("M")
    try:
        return p.to_timestamp("ME")
    except (TypeError, ValueError):
        return p.to_timestamp(how="end")


def _verify_no_lookahead_bias(
    classified_df: pd.DataFrame,
    returns_index: pd.Index,
    cpi_index: pd.Index,
    cpi_lag_months: int = 1,
) -> None:
    """
    Verify that the regime signal does not use future information relative to asset returns.

    With cpi_lag_months applied in the classifier, the regime at portfolio date T is based
    on CPI reference month T - cpi_lag_months (released before or during T). So weights at
    month-end T use only data that was available by T. This check ensures:
    - The classified index has been shifted (first regime date >= first CPI date + lag).
    - We log the information-set rule for auditability.
    """
    if not cpi_lag_months or cpi_lag_months <= 0:
        logger.info("Information set: cpi_lag_months is 0; no publication lag applied.")
        return
    reg_min = pd.Timestamp(classified_df.index.min())
    cpi_min = pd.Timestamp(cpi_index.min())
    # After lag, regime at date T is from CPI reference T - lag; so first regime date
    # should be at least cpi_min + lag (first date when we could know the first CPI).
    expected_min = cpi_min + pd.DateOffset(months=cpi_lag_months)
    if reg_min < expected_min:
        logger.warning(
            "Look-ahead check: regime index starts at %s; expected >= %s (cpi_lag_months=%s). "
            "Verify that RegimeClassifier applies cpi_lag_months.",
            reg_min.date(), expected_min.date(), cpi_lag_months,
        )
    else:
        logger.info(
            "Information set: weights at month-end T use regime from CPI reference month T - %s "
            "(released by T). No look-ahead between regime signal and asset returns.",
            cpi_lag_months,
        )


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

        # -----------------------------------------------------------------------
        # Return vs weight timing (no look-ahead).
        # The return r_T for month T is the return realized *during* month T
        # (from end of T-1 to end of T). We can only earn that return with the
        # weights we had at the *start* of month T, i.e. the weights set at
        # end of T-1. So we must use prev_wts (weights from end of previous
        # period) to compute the portfolio return for the current period. Using
        # w_T for r_T would imply we knew the month-T return before choosing
        # weights, which is look-ahead bias. We therefore compute:
        #   gross_T = (weights_held_at_start_of_T) · r_T
        # where weights_held_at_start_of_T = prev_wts (updated at end of T-1).
        # On the first date we have no prior weights, so we use the first
        # period's weights for that one month to avoid a zero return; from
        # the second month onward we use prev_wts strictly.
        # -----------------------------------------------------------------------
        results  = []
        prev_wts = pd.Series(0.0, index=wts.columns)

        for date in common_idx:
            w      = wts.loc[date]   # Weights we rebalance TO at end of this month (for next period)
            r      = rets.loc[date]  # Return that was realized *during* this month

            # Weights that were held during this month (and thus earned r): use previous
            # period's weights to avoid look-ahead; fallback to w only for the first month.
            weights_held = prev_wts if prev_wts.sum() > 0 else w

            common = weights_held.index.intersection(r.index)
            w_held_aligned = weights_held.reindex(common).fillna(0)
            total = w_held_aligned.sum()
            if total > 0:
                w_held_aligned = w_held_aligned / total
            else:
                w_held_aligned = pd.Series(0.0, index=common)

            # Portfolio return = return earned with weights we held (no look-ahead)
            gross  = float((w_held_aligned * r.reindex(common).fillna(0)).sum())

            # Transaction cost: we pay cost for moving from prev_wts to w at end of this month
            turnover = float((w - prev_wts).abs().sum()) / 2
            cost     = turnover * self.tc_bps / 10_000
            net      = gross - cost

            # Normalise regime to canonical string so metrics never see "Regime.FIRE" or "Reg".
            _r = regime.get(date, None)
            if _r is None or (isinstance(_r, float) and np.isnan(_r)):
                _r_label = None
            elif hasattr(_r, "value"):
                _r_label = _r.value
            else:
                _s = str(_r).strip()
                _r_label = _s.split(".", 1)[1] if _s.startswith("Regime.") else (_s or None)
            results.append({
                "date":               date,
                "portfolio_return":   gross,
                "net_return":         net,
                "transaction_cost":   cost,
                "regime":             _r_label,
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

def compute_real_wealth(results: pd.DataFrame, cpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deflate nominal NAV by UK CPI so Real Wealth reflects purchasing power.

    Uses CPI reference month = backtest month (month-end T uses CPI index for month T).
    Normalises CPI to 1 at the first common date so real_nav starts at 1.
    No look-ahead: alignment uses reindex/ffill so each month uses last known CPI.

    Parameters
    ----------
    results : DataFrame   Backtest output with index = month-end dates, column "nav".
    cpi_df  : DataFrame   CPI from get_uk_cpi(); must have "cpi_index" (e.g. after _compute_derivatives).

    Returns
    -------
    DataFrame   Copy of results with columns "real_nav" and "real_return" added.
    """
    if "cpi_index" not in cpi_df.columns:
        return results
    # Align CPI to month-end: use CPI index for each month, reindex to results
    cpi_idx = cpi_df["cpi_index"].dropna()
    if cpi_idx.empty:
        return results
    try:
        cpi_me = cpi_idx.resample("ME").last().ffill()
    except TypeError:
        cpi_me = cpi_idx.resample("M").last().ffill()
    aligned = cpi_me.reindex(results.index).ffill().bfill()
    first_common = aligned.first_valid_index()
    if first_common is None:
        return results
    cpi_norm = aligned / aligned.loc[first_common]
    out = results.copy()
    out["real_nav"] = results["nav"] / cpi_norm
    out["real_return"] = out["real_nav"].pct_change()
    return out


def compute_metrics(
    results: pd.DataFrame,
    col: str = "net_return",
    risk_free_monthly: pd.Series | None = None,
) -> dict:
    """
    Compute standard performance metrics for a return series.

    If risk_free_monthly is provided (index = dates, value = monthly RF in decimal),
    Sharpe and Sortino use excess return over that rate; otherwise RF is treated as 0.
    """
    rets = results[col].dropna()
    nav  = (1 + rets).cumprod()

    # Annualised return
    n_years  = len(rets) / 12
    ann_ret  = (nav.iloc[-1]) ** (1 / n_years) - 1

    # Annualised volatility
    ann_vol  = rets.std() * np.sqrt(12)

    # Risk-free: align to return index and compute geometric total then annualise
    ann_rf = 0.0
    if risk_free_monthly is not None and not risk_free_monthly.empty:
        rf_aligned = risk_free_monthly.reindex(rets.index).ffill().bfill()
        common = rets.index.intersection(rf_aligned.dropna().index)
        if len(common) > 0:
            rf_period = rf_aligned.loc[common]
            tot_rf = (1 + rf_period).prod()
            n_per = len(rf_period)
            ann_rf = tot_rf ** (12 / n_per) - 1 if n_per else 0.0

    # Sharpe: excess return over RF
    excess = ann_ret - ann_rf
    sharpe = excess / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    roll_max = nav.cummax()
    dd       = (nav - roll_max) / roll_max
    max_dd   = dd.min()

    # Calmar ratio
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Sortino (downside deviation), excess over RF
    neg_rets  = rets[rets < 0]
    down_vol  = neg_rets.std() * np.sqrt(12)
    sortino   = excess / down_vol if down_vol > 0 else np.nan

    out = {
        "ann_return_%":    round(ann_ret * 100, 2),
        "ann_vol_%":       round(ann_vol * 100, 2),
        "sharpe":          round(sharpe, 3),
        "sortino":         round(sortino, 3),
        "calmar":          round(calmar, 3),
        "max_drawdown_%":  round(max_dd * 100, 2),
        "n_months":        len(rets),
    }
    if risk_free_monthly is not None and not risk_free_monthly.empty:
        out["ann_rf_%"] = round(ann_rf * 100, 2)
    return out


def compute_regime_metrics(
    results: pd.DataFrame,
    risk_free_monthly: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Break down performance metrics by Neville regime.
    If risk_free_monthly is provided, Sharpe/Sortino use excess over that rate per regime.
    """
    rows = []
    for regime in results["regime"].dropna().unique():
        mask  = results["regime"] == regime
        sub   = results.loc[mask]
        m     = compute_metrics(sub, risk_free_monthly=risk_free_monthly)
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
    from data_ingestion.rates import get_uk_risk_free_rate
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
    rets    = _inject_synthetic_cta_if_enabled(rets)
    cpi     = get_uk_cpi(start=effective_start)
    regimes = RegimeClassifier().classify(cpi)
    cpi_lag = (CONFIG.get("regime") or {}).get("cpi_lag_months", 1)
    _verify_no_lookahead_bias(regimes, rets.index, cpi.index, cpi_lag)
    weights, rp_stats = build_weight_history(regimes, rets)

    engine  = BacktestEngine(rets, weights, regimes)
    results = engine.run()
    _verify_fire_cta_weight(results, weights)
    bm      = engine.run_benchmark()

    rep_cfg = CONFIG.get("reporting") or {}
    if rep_cfg.get("real_wealth") and "cpi_index" in cpi.columns:
        results = compute_real_wealth(results, cpi)

    # Risk-free rate for Sharpe/Sortino (BoE base rate when enabled)
    rf_series = None
    if rep_cfg.get("use_risk_free_rate"):
        end_ts = results.index.max()
        end_str = end_ts.strftime("%Y-%m-%d") if hasattr(end_ts, "strftime") else str(end_ts)[:10]
        try:
            rf_series = get_uk_risk_free_rate(effective_start, end_str)
            if not rf_series.empty:
                rf_series = rf_series.reindex(results.index).ffill().bfill()
        except Exception as e:
            logger.warning("Risk-free rate unavailable, using 0%% for Sharpe/Sortino: %s", e)

    print("\n--- Portfolio Metrics ---")
    print(compute_metrics(results, risk_free_monthly=rf_series))

    if not bm.empty:
        bm_metrics = compute_metrics(bm, col="benchmark_return", risk_free_monthly=rf_series)
        print("\n--- 60/40 Benchmark Metrics ---")
        print(bm_metrics)

    print("\n--- Performance by Regime ---")
    print(compute_regime_metrics(results, risk_free_monthly=rf_series).to_string())

    if "real_return" in results.columns:
        print("\n--- Real wealth metrics ---")
        print(compute_metrics(results, col="real_return", risk_free_monthly=rf_series))

    # Transparency: how FIRE was adapted so the regime table is interpretable.
    print(
        "\n--- Regime adaptations (transparency) ---\n"
        "FIRE (High + Rising): Strategic allocation tilts heavily to commodities (CMOD) and CTA (DBMG)\n"
        "and keeps duration/equities at 2% so inflation hedges are not diluted. Risk-parity is\n"
        "bypassed when the dominant regime is FIRE, so raw base weights are used.\n"
        "ICE, BOOM, and RECOVERY use risk-parity where applicable."
    )

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

    # Save interactive Plotly charts: cumulative returns with regime shading + weight heatmap
    try:
        from analysis.visualizer import save_backtest_charts
        out_dir = Path(CONFIG.get("reporting", {}).get("output_dir", "reports"))
        html_path = save_backtest_charts(results, weights, bm, out_dir / "backtest_charts.html")
        print("\nCharts saved: %s" % html_path)
    except Exception as e:
        logger.warning("Could not save backtest charts: %s", e)