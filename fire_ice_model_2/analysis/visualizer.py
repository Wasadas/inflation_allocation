"""
analysis/visualizer.py

Plotly charts for the Fire & Ice backtest: three panels —
(1) cumulative NAV vs 60/40 with regime shading,
(2) drawdown comparison,
(3) regime classification over time as stacked area.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

logger = logging.getLogger(__name__)

# Regime colours: light backgrounds for shading, solid for stacked area (Panel 1 & 3)
REGIME_COLOURS = {
    "FIRE":      "rgba(255, 0, 0, 0.15)",
    "ICE":       "rgba(0, 100, 255, 0.15)",
    "RECOVERY":  "rgba(0, 180, 0, 0.15)",
    "BOOM":      "rgba(255, 200, 0, 0.2)",
}
REGIME_SOLID = {
    "FIRE":      "rgba(220, 80, 80, 0.85)",
    "ICE":       "rgba(70, 130, 220, 0.85)",
    "RECOVERY":  "rgba(60, 160, 60, 0.85)",
    "BOOM":      "rgba(240, 180, 0, 0.85)",
}

# Portfolio and benchmark line colours
NAV_PORTFOLIO = "rgb(30, 60, 120)"   # dark blue
NAV_BENCHMARK = "rgb(200, 0, 0)"     # red
DD_PORTFOLIO = "rgb(30, 60, 120)"    # dark blue
DD_BENCHMARK = "rgb(200, 0, 0)"      # red


def _normalise_regime(r) -> str:
    """Coerce regime to string (handles Regime enum from classifier)."""
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return "ICE"
    s = getattr(r, "value", str(r))
    return str(s).upper().strip()


def _regime_segments(series: pd.Series):
    """Yield (start_ts, end_ts, regime) for each contiguous regime block."""
    if series.empty:
        return
    prev = _normalise_regime(series.iloc[0])
    start = series.index[0]
    for ts in series.index[1:]:
        curr = _normalise_regime(series.loc[ts])
        if curr != prev:
            yield start, ts, prev
            start = ts
            prev = curr
    yield start, series.index[-1], prev


def _regime_probability_series(
    results: pd.DataFrame,
    classified_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns FIRE, ICE, BOOM, RECOVERY and index = results.index,
    values in [0, 1] summing to 1. Uses classified_df prob_* if present, else one-hot from results['regime'].
    """
    idx = results.index.sort_values()
    order = ["FIRE", "ICE", "BOOM", "RECOVERY"]

    # Preferred path: smooth probabilities from the classifier, when available
    if classified_df is not None:
        prob_cols = ["prob_fire", "prob_boom", "prob_ice", "prob_recovery"]
        if all(c in classified_df.columns for c in prob_cols):
            raw = classified_df[prob_cols]
            if not raw.empty:
                probs = raw.reindex(idx).ffill().bfill()
                # If everything is NaN or zero we fall back to one-hot below.
                if np.isfinite(probs.to_numpy()).any() and probs.to_numpy().sum() != 0:
                    probs = probs.fillna(0)
                    probs.columns = ["FIRE", "BOOM", "ICE", "RECOVERY"]
                    probs = probs[order]
                    row_sum = probs.sum(axis=1).replace(0, np.nan)
                    probs = probs.div(row_sum, axis=0).fillna(0)
                    return probs

    # Fallback: one-hot from backtest results['regime'] (always aligned to NAV)
    regime_series = results.loc[idx, "regime"].map(_normalise_regime)
    out = pd.DataFrame(0.0, index=idx, columns=order)
    for reg in order:
        out.loc[regime_series == reg, reg] = 1.0
    return out


def save_backtest_charts(
    results: pd.DataFrame,
    weights: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_path: str | Path,
    classified_df: pd.DataFrame | None = None,
) -> Path:
    """
    Build three panels and save:

    - Panel 1: Fire & Ice vs 60/40 benchmark (cumulative NAV from 1.0), regime shading, hline at 1.0.
    - Panel 2: Drawdown comparison (portfolio and benchmark as negative %), filled areas.
    - Panel 3: Regime classification over time (stacked area of regime probabilities).

    Saves lightweight HTML (Plotly from CDN) and optional PNG. Weight heatmap removed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Fire & Ice vs 60/40 Benchmark",
            "Drawdown: Fire & Ice vs 60/40",
            "Regime Classification Over Time",
        ),
        vertical_spacing=0.10,
        row_heights=[0.38, 0.32, 0.30],
    )

    # Use results.index as single source of truth; NAV is reindexed to it.
    res_idx = pd.DatetimeIndex(results.index)
    if res_idx.tz is not None:
        res_idx = res_idx.tz_localize(None)

    common = res_idx.sort_values()
    nav_aligned = pd.Series(results["nav"].values, index=res_idx).reindex(common).ffill().bfill()
    if nav_aligned.isna().all():
        nav_aligned = pd.Series(1.0, index=common[:1])
    start_nav = float(nav_aligned.iloc[0]) if nav_aligned.iloc[0] != 0 else 1.0
    nav_aligned = nav_aligned / start_nav

    bm_aligned = None
    if not benchmark.empty and "benchmark_nav" in benchmark.columns:
        # Expect benchmark.index to already match results.index; do only minimal safety alignment.
        bench_idx = pd.DatetimeIndex(benchmark.index)
        if bench_idx.tz is not None:
            bench_idx = bench_idx.tz_localize(None)
        bm_series = pd.Series(benchmark["benchmark_nav"].values, index=bench_idx)
        bm_reindexed = bm_series.reindex(common).ffill().bfill()
        if bm_reindexed.notna().any():
            first_valid = bm_reindexed.dropna().iloc[0]
            start_bm = float(first_valid) if first_valid != 0 else 1.0
            bm_aligned = bm_reindexed / start_bm
            logger.info("Chart: adding 60/40 benchmark trace with %d points", len(bm_aligned))
            print("[Chart] Adding 60/40 benchmark trace, points:", len(bm_aligned))
        else:
            logger.warning("Chart: benchmark reindexed to results index is all NaN; skipping 60/40 line")
            print("[Chart] Skipping 60/40: reindexed benchmark is all NaN")
    else:
        if benchmark.empty:
            logger.warning("Chart: benchmark DataFrame is empty; skipping 60/40 line")
            print("[Chart] Skipping 60/40: benchmark DataFrame is empty")
        elif "benchmark_nav" not in benchmark.columns:
            logger.warning("Chart: benchmark has no 'benchmark_nav' column; skipping 60/40 line")
            print("[Chart] Skipping 60/40: no 'benchmark_nav' column; columns:", list(benchmark.columns))

    regime_series = results["regime"]

    # ---- Panel 1: Cumulative NAV, regime shading, hline at 1.0 ----
    for start_ts, end_ts, reg in _regime_segments(regime_series):
        color = REGIME_COLOURS.get(reg, "rgba(200,200,200,0.1)")
        fig.add_vrect(
            x0=start_ts, x1=end_ts,
            y0=0, y1=1, yref="paper",
            row=1, col=1,
            fillcolor=color, line_width=0,
        )

    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(0,0,0,0.35)", line_width=1, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=nav_aligned.index,
            y=nav_aligned.values,
            name="Fire & Ice",
            line=dict(color=NAV_PORTFOLIO, width=2),
        ),
        row=1, col=1,
    )
    if bm_aligned is not None and bm_aligned.notna().any():
        fig.add_trace(
            go.Scatter(
                x=bm_aligned.index,
                y=bm_aligned.values,
                name="60/40 Benchmark",
                line=dict(color=NAV_BENCHMARK, width=2, dash="dash"),
                mode="lines",
                xaxis="x",
                yaxis="y",
            ),
            row=1, col=1,
        )

    # ---- Panel 2: Drawdown (negative %) ----
    roll_max_p = nav_aligned.cummax()
    dd_p = (nav_aligned - roll_max_p) / roll_max_p
    fig.add_trace(
        go.Scatter(
            x=dd_p.index,
            y=dd_p.values * 100,
            name="Fire & Ice (drawdown)",
            line=dict(color=DD_PORTFOLIO, width=2),
            fill="tozeroy",
            fillcolor="rgba(30, 60, 120, 0.25)",
        ),
        row=2, col=1,
    )
    if bm_aligned is not None and bm_aligned.notna().any():
        roll_max_b = bm_aligned.cummax()
        dd_b = (bm_aligned - roll_max_b) / roll_max_b
        fig.add_trace(
            go.Scatter(
                x=dd_b.index,
                y=dd_b.values * 100,
                name="60/40 Benchmark (drawdown)",
                line=dict(color=DD_BENCHMARK, width=1.8),
                fill="tozeroy",
                fillcolor="rgba(200, 0, 0, 0.20)",
            ),
            row=2, col=1,
        )

    # ---- Panel 3: Stacked area regime probabilities ----
    regime_probs = _regime_probability_series(results, classified_df)
    if not regime_probs.empty:
        order = ["FIRE", "ICE", "BOOM", "RECOVERY"]
        for i, reg in enumerate(order):
            if reg not in regime_probs.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=regime_probs.index,
                    y=regime_probs[reg].values,
                    name=reg,
                    stackgroup="regime",
                    line=dict(width=0.5, color=REGIME_SOLID.get(reg, "rgba(150,150,150,0.8)")),
                    fill="tonexty",
                    fillcolor=REGIME_SOLID.get(reg, "rgba(150,150,150,0.6)"),
                ),
                row=3, col=1,
            )

    # Layout: clean white background, light horizontal grid only
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        height=900,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Regime weight", range=[0, 1], row=3, col=1)

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    output_path.write_text(html_str, encoding="utf-8")

    png_path = output_path.with_suffix(".png")
    try:
        pio.write_image(fig, png_path, width=1200, height=900, scale=2)
    except Exception:
        pass

    return output_path
