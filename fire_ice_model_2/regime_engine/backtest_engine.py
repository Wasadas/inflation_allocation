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

from .classifier import CANONICAL_REGIME_LABELS

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Project root (parent of fire_ice_model_2) for resolving relative config paths
PROJECT_ROOT = CONFIG_PATH.resolve().parent.parent

BT_CFG = CONFIG["backtest"]
TC_BPS = BT_CFG["transaction_cost_bps"]
DRIFT_THRESHOLD = float(BT_CFG.get("rebalance_drift_threshold", 0.0) or 0.0)


def _benchmark_tickers_from_config(available_columns: list) -> tuple[str | None, str | None]:
    """
    Resolve 60/40 benchmark tickers from config.
    Returns (equity_ticker, bond_ticker) from assets.equities and assets.bonds
    that exist in available_columns. Prefers ISF.L / IGLT.L when present.
    """
    assets_cfg = CONFIG.get("assets") or {}
    equities = list((assets_cfg.get("equities") or {}).values())
    bonds = list((assets_cfg.get("bonds") or {}).values())
    avail = set(available_columns)
    eq = next((t for t in equities if t in avail), None)
    bd = next((t for t in bonds if t in avail), None)
    return (eq, bd)


def _get_cta_proxy_ticker_from_config() -> str | None:
    """Config cta_proxy ticker (e.g. DBMG.L); used to attach disclaimer when that column is dropped."""
    trend = (CONFIG.get("assets") or {}).get("trend")
    if isinstance(trend, dict):
        t = (trend.get("cta_proxy") or "").strip()
        return t or None
    return None


def _log_cta_disclaimer_if_dropped(dropped: list) -> None:
    """If we have to drop the CTA proxy column, explain why in the logs."""
    try:
        from data_ingestion.asset_prices import CTA_SYNTHETIC_DISCLAIMER
    except ImportError:
        return
    cta = _get_cta_proxy_ticker_from_config()
    if cta and cta in dropped:
        logger.info("%s", CTA_SYNTHETIC_DISCLAIMER)


def _verify_fire_cta_weight(results: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Sanity check: how much CTA risk we actually carried in FIRE months."""
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
    Build a synthetic CTA series when config asks for it and wire it into the
    actual return matrix the backtest uses.

    Returns
    -------
    DataFrame with synthetic CTA column added (or unchanged if cta_mode != "synthetic").
    """
    # We keep the config check simple here and push the heavy lifting into the
    # trend_following module so the backtest stays easy to read.
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

    # Build CTA from the configured trend assets that exist in returns (excludes missing DBMF)
    trend_assets = tf_cfg.get("assets_to_trend") or []
    rets_for_cta = returns[[c for c in trend_assets if c in returns.columns]]
    if rets_for_cta.empty:
        rets_for_cta = returns

    cta_returns = compute_cta_proxy_returns(rets_for_cta)
    aligned = cta_returns.reindex(returns.index).ffill().bfill().fillna(0.0)

    out = returns.copy()
    if cta_ticker in out.columns:
        # Merge live DBMF (where available) with synthetic CTA: keep live data,
        # but fill historical NaN gaps with the synthetic series so the CTA
        # hedge is active in early FIRE months before DBMF inception.
        live = out[cta_ticker]
        mask = live.isna() & aligned.notna()
        filled_count = int(mask.sum())
        pre2021_filled = int(mask[mask.index < pd.Timestamp("2021-01-01")].sum())
        merged = live.where(~mask, aligned)
        out[cta_ticker] = merged.values
        logger.info(
            "CTA backfill: filled %s DBMF return gaps (including %s pre-2021) with synthetic CTA "
            "(cta_mode=synthetic); kept live data where available.",
            filled_count,
            pre2021_filled,
        )
    else:
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
            w      = wts.loc[date]   # Target weights we would rebalance TO at end of this month (for next period)
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

            # Portfolio-level volatility targeting (final layer, after all weights are set).
            target_vol = float(BT_CFG.get("target_vol") or 0.0)
            vol_scale_max = float(BT_CFG.get("vol_scale_max") or 1.0)
            if target_vol > 0:
                # Use a 36-month rolling window of returns up to the previous month.
                hist_rets = rets.loc[:date].iloc[:-1].tail(36)
                if not hist_rets.empty:
                    hist_common = hist_rets.columns.intersection(common)
                    cov = hist_rets[hist_common].cov() * 12.0  # annualised covariance
                    w_vec = w_held_aligned.reindex(hist_common).fillna(0).values
                    try:
                        sigma = float(np.sqrt(w_vec @ cov.values @ w_vec))
                    except ValueError:
                        sigma = 0.0
                else:
                    sigma = 0.0
                if sigma > 0:
                    scale = min(target_vol / sigma, vol_scale_max)
                else:
                    scale = 1.0
                w_held_scaled = w_held_aligned * scale
            else:
                w_held_scaled = w_held_aligned

            # Portfolio return = return earned with (possibly scaled) weights we held (no look-ahead)
            gross  = float((w_held_scaled * r.reindex(common).fillna(0)).sum())

            # Drift-based rebalancing: only trade when target weights have moved
            # meaningfully away from current holdings. Small changes inside the
            # drift band are ignored to reduce turnover and cost.
            if prev_wts.sum() == 0 or DRIFT_THRESHOLD <= 0:
                w_effective = w
            else:
                delta = w - prev_wts
                w_effective = prev_wts.copy()
                mask = delta.abs() >= DRIFT_THRESHOLD
                w_effective[mask] = w[mask]

            # Transaction cost: we pay cost for moving from prev_wts to w_effective
            turnover = float((w_effective - prev_wts).abs().sum()) / 2
            cost     = turnover * self.tc_bps / 10_000
            net      = gross - cost

            # Normalise regime to canonical string only; drop "Reg", "Regime.FIRE", etc.
            _r = regime.get(date, None)
            if _r is None or (isinstance(_r, float) and np.isnan(_r)):
                _r_label = None
            elif hasattr(_r, "value"):
                _r_label = _r.value
            else:
                _s = str(_r).strip()
                _r_label = _s.split(".", 1)[1] if _s.startswith("Regime.") else (_s or None)
            if _r_label not in CANONICAL_REGIME_LABELS:
                _r_label = None
            results.append({
                "date":               date,
                "portfolio_return":   gross,
                "net_return":         net,
                "transaction_cost":   cost,
                "regime":             _r_label,
            })
            prev_wts = w_effective

        df = pd.DataFrame(results).set_index("date")
        df["nav"] = (1 + df["net_return"]).cumprod()
        return df

    def run_benchmark(
        self,
        equity_ticker: str | None = None,
        bond_ticker: str | None = None,
        equity_weight: float = 0.60,
        index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Run a static 60/40-style benchmark for comparison.

        The benchmark is defined as 60% equity / 40% bond using tickers from:
        - hard-coded default: ISF.L / IGLT.L, or
        - config assets.equities / assets.bonds as a fallback (first available pair).

        If `index` is provided, the resulting benchmark series is reindexed to that
        index (typically results.index from the backtest) with ffill/bfill so that
        metrics and charts can compare like-for-like.
        """
        # Resolve tickers: prefer explicit args, then defaults, then config-based fallback.
        if equity_ticker is None:
            equity_ticker = "ISF.L"
        if bond_ticker is None:
            bond_ticker = "IGLT.L"

        if equity_ticker not in self.returns.columns or bond_ticker not in self.returns.columns:
            eq, bd = _benchmark_tickers_from_config(list(self.returns.columns))
            if eq is not None and bd is not None:
                equity_ticker, bond_ticker = eq, bd
                logger.info("Benchmark: using %s / %s (fallback from config assets)", equity_ticker, bond_ticker)
            else:
                if equity_ticker not in self.returns.columns:
                    logger.warning("Benchmark equity %s not in returns; no fallback available", equity_ticker)
                if bond_ticker not in self.returns.columns:
                    logger.warning("Benchmark bond %s not in returns; no fallback available", bond_ticker)
                return pd.DataFrame()

        bond_weight = 1 - equity_weight
        bm = (
            self.returns[equity_ticker] * equity_weight
            + self.returns[bond_ticker] * bond_weight
        )
        bm = bm.dropna()
        if bm.empty:
            logger.warning("Benchmark series for %s/%s is empty after dropna; no 60/40 benchmark available.", equity_ticker, bond_ticker)
            return pd.DataFrame()

        bm_df = pd.DataFrame({
            "benchmark_return": bm,
            "benchmark_nav":    (1 + bm).cumprod(),
        })

        # Optional alignment to a target index (e.g. results.index).
        if index is not None and len(index) > 0:
            tgt_idx = pd.DatetimeIndex(index)
            if tgt_idx.tz is not None:
                tgt_idx = tgt_idx.tz_localize(None)
            src_idx = pd.DatetimeIndex(bm_df.index)
            if src_idx.tz is not None:
                src_idx = src_idx.tz_localize(None)
            bm_df = bm_df.copy()
            bm_df.index = src_idx
            # Match by calendar day: results.index is often month-end 23:59:59, benchmark is 00:00:00.
            tgt_dates = tgt_idx.normalize()
            src_dates = src_idx.normalize()
            bm_by_date = bm_df.set_index(src_dates)
            aligned = bm_by_date.reindex(tgt_dates).ffill().bfill()
            aligned.index = tgt_idx
            if aligned["benchmark_nav"].notna().any():
                logger.info(
                    "Benchmark aligned to target index (%s -> %s, %d points).",
                    src_idx.min().date() if len(src_idx) else None,
                    tgt_idx.max().date() if len(tgt_idx) else None,
                    len(aligned),
                )
                bm_df = aligned
            else:
                logger.warning(
                    "Benchmark alignment to target index produced all-NaN NAV; dropping 60/40 series "
                    "(no overlapping history on requested index)."
                )
                return pd.DataFrame()

        return bm_df


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


def _normalise_regime_label(x) -> str | None:
    """Map to one of FIRE/BOOM/ICE/RECOVERY or None; never return 'Reg' or 'Regime.XXX'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if hasattr(x, "value"):
        s = x.value
    else:
        s = str(x).strip()
        if s.startswith("Regime."):
            s = s.split(".", 1)[1]
    return s if s in CANONICAL_REGIME_LABELS else None


def compute_regime_metrics(
    results: pd.DataFrame,
    risk_free_monthly: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Break down performance metrics by Neville regime.
    If risk_free_monthly is provided, Sharpe/Sortino use excess over that rate per regime.
    Only canonical labels (FIRE, BOOM, ICE, RECOVERY) are shown; non-canonical values are excluded.
    """
    reg_series = results["regime"].map(_normalise_regime_label)
    rows = []
    for regime in CANONICAL_REGIME_LABELS:
        mask = reg_series == regime
        if not mask.any():
            continue
        sub = results.loc[mask]
        m = compute_metrics(sub, risk_free_monthly=risk_free_monthly)
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
    # Compute benchmark directly on the backtest index (results.index) so metrics and charts compare like-for-like.
    bm      = engine.run_benchmark(index=results.index)
    if bm.empty:
        logger.warning(
            "Benchmark is empty (no 60/40 series) even after alignment to results.index. Returns columns: %s",
            list(rets.columns),
        )
        print("[Benchmark] Empty after run_benchmark(index=results.index). Returns columns:", list(rets.columns))

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
        "FIRE (High + Rising): Strategic allocation tilts heavily to commodities and CTA\n"
        "and keeps duration/equities at 2% so inflation hedges are not diluted.\n"
        "ICE, BOOM, and RECOVERY use risk-parity where applicable (with blending)."
    )

    # Detailed optimisation summary: method usage by regime.
    method_counts = rp_stats.get("method_counts") or {}
    if method_counts:
        print("\nWeight method by regime:")
        for regime_key, counts in method_counts.items():
            total = sum(counts.values())
            if total == 0:
                continue
            base_n = counts.get("base", 0)
            rp_n = counts.get("rp", 0)
            blend_n = counts.get("blend", 0)
            parts = []
            if base_n:
                parts.append(f"{base_n} base")
            if rp_n:
                parts.append(f"{rp_n} pure RP")
            if blend_n:
                parts.append(f"{blend_n} blend")
            summary = ", ".join(parts)
            print(f"  {regime_key:<8}: {total:4d} months — {summary}")

    # Save interactive Plotly charts: three panels (NAV, drawdown, regime stacked area)
    try:
        from analysis.visualizer import save_backtest_charts
        out_dir_cfg = CONFIG.get("reporting", {}).get("output_dir", "reports")
        out_dir = (PROJECT_ROOT / out_dir_cfg).resolve() if not Path(out_dir_cfg).is_absolute() else Path(out_dir_cfg)
        html_path = save_backtest_charts(results, weights, bm, out_dir / "backtest_charts.html", classified_df=regimes)
        print("\nCharts saved: %s" % html_path)
    except Exception as e:
        logger.warning("Could not save backtest charts: %s", e)