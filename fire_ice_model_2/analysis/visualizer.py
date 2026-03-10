"""
analysis/visualizer.py

Plotly charts for the Fire & Ice backtest: cumulative returns with regime shading
and a weight heatmap so we can verify the CTA/commodity tilt in FIRE zones.

Uses the same regime colour convention throughout: FIRE=red, ICE=blue,
RECOVERY=green, BOOM=yellow.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Regime background colours (Red=FIRE, Blue=ICE, Green=RECOVERY, Yellow=BOOM)
REGIME_COLOURS = {
    "FIRE":      "rgba(255, 0, 0, 0.15)",
    "ICE":       "rgba(0, 100, 255, 0.15)",
    "RECOVERY":  "rgba(0, 180, 0, 0.15)",
    "BOOM":      "rgba(255, 200, 0, 0.2)",
}


def _normalise_regime(r) -> str:
    """Coerce regime to string (handles Regime enum from classifier)."""
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return "ICE"  # default for missing
    s = getattr(r, "value", str(r))
    return str(s).upper().strip()


def _regime_segments(series: pd.Series):
    """
    Yield (start_ts, end_ts, regime) for each contiguous regime block.
    Used to draw background rectangles without gaps.
    """
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


def plot_cumulative_returns_with_regimes(
    results: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> go.Figure:
    """
    Portfolio vs 60/40 benchmark with background shaded by dominant_regime.
    Red=FIRE, Blue=ICE, Green=RECOVERY, Yellow=BOOM.
    """
    idx = results.index.sort_values()
    nav = results.loc[idx, "nav"].dropna()
    if nav.empty:
        return go.Figure()

    # Align benchmark to portfolio dates (skip if benchmark missing)
    bm_nav = None
    if not benchmark.empty and "benchmark_nav" in benchmark.columns:
        bm_nav = benchmark.reindex(idx)["benchmark_nav"].ffill().bfill()
    regime_series = results.loc[idx, "regime"]

    fig = go.Figure()

    # Background rectangles for each regime segment
    for start_ts, end_ts, reg in _regime_segments(regime_series):
        color = REGIME_COLOURS.get(reg, "rgba(200,200,200,0.1)")
        fig.add_vrect(
            x0=start_ts,
            x1=end_ts,
            y0=0,
            y1=1,
            yref="paper",
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=nav.values,
            name="Portfolio",
            line=dict(color="navy", width=2),
        )
    )
    if bm_nav is not None and bm_nav.notna().any():
        fig.add_trace(
            go.Scatter(
                x=bm_nav.index,
                y=bm_nav.values,
                name="60/40 Benchmark",
                line=dict(color="gray", width=1.5, dash="dot"),
            )
        )
    if "real_nav" in results.columns:
        rn = results.loc[idx, "real_nav"].dropna()
        if not rn.empty:
            fig.add_trace(
                go.Scatter(
                    x=rn.index,
                    y=rn.values,
                    name="Real wealth",
                    line=dict(color="green", width=1.5, dash="dash"),
                )
            )

    fig.update_layout(
        title="Cumulative returns (NAV) with regime shading",
        xaxis_title="",
        yaxis_title="NAV (1 = 100%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=50, b=40),
        height=400,
    )
    fig.update_yaxes(tickformat=".2f")
    return fig


def plot_weight_heatmap(
    weights: pd.DataFrame,
    results: pd.DataFrame,
) -> go.Figure:
    """
    Heatmap of asset weights over time. Aligns weights to backtest dates (ffill)
    so we can verify the CTA/commodity tilt during FIRE (red) zones.
    """
    # Align weights to the same index as results (month-end backtest dates)
    common = results.index.intersection(weights.index).sort_values()
    if common.empty:
        w = weights.copy()
    else:
        w = weights.reindex(common, method="ffill").fillna(0)

    if w.empty or w.columns.empty:
        return go.Figure()

    # Drop the long flat section before the portfolio is actually invested
    total = w.sum(axis=1)
    if total.gt(0.01).any():
        first_live = total[total.gt(0.01)].index[0]
        w = w.loc[first_live:]

    # Zero-out tiny weights to avoid noisy background
    w = w.where(w >= 0.02, 0.0)

    # Put DBMF and GSG early so FIRE tilt is easy to spot
    priority = ["DBMF", "GSG", "IGLT.L", "INXG.L", "ISF.L", "VMID.L", "IHYG.L", "SGLN.L"]
    cols = [c for c in priority if c in w.columns]
    cols += [c for c in w.columns if c not in cols]
    w = w[cols]

    fig = go.Figure(
        data=go.Heatmap(
            x=w.index,
            y=w.columns,
            z=w.values.T,
            colorscale="Blues",
            zmin=0,
            zmax=0.25,
            hovertemplate="%{x|%b %Y}<br>%{y}: %{z:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Asset weights over time (verify CTA/CMOD tilt in FIRE zones)",
        xaxis_title="",
        yaxis_title="",
        height=400,
        margin=dict(t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def save_backtest_charts(
    results: pd.DataFrame,
    weights: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Build both charts and save a single interactive HTML file (two figures stacked).
    Creates the output directory if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Single figure with two subplots so one HTML has both
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Cumulative returns (NAV) with regime shading",
            "Asset weights over time (CTA/CMOD tilt in FIRE)",
        ),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    # ---- Top: cumulative returns with regime shading ----
    idx = results.index.sort_values()
    nav = results.loc[idx, "nav"].dropna()
    if not nav.empty:
        bm_nav = None
        if not benchmark.empty and "benchmark_nav" in benchmark.columns:
            bm_nav = benchmark.reindex(idx)["benchmark_nav"].ffill().bfill()
        regime_series = results.loc[idx, "regime"]

        # Add regime background rectangles to the first subplot only
        for start_ts, end_ts, reg in _regime_segments(regime_series):
            color = REGIME_COLOURS.get(reg, "rgba(200,200,200,0.1)")
            fig.add_vrect(
                x0=start_ts, x1=end_ts,
                y0=0, y1=1, yref="paper",
                row=1, col=1,
                fillcolor=color, line_width=0,
            )

        fig.add_trace(
            go.Scatter(x=nav.index, y=nav.values, name="Portfolio", line=dict(color="navy", width=2)),
            row=1, col=1,
        )
        if bm_nav is not None and bm_nav.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=bm_nav.index, y=bm_nav.values,
                    name="60/40 Benchmark",
                    line=dict(color="gray", width=1.5, dash="dot"),
                ),
                row=1, col=1,
            )
        if "real_nav" in results.columns:
            rn = results.loc[idx, "real_nav"].dropna()
            if not rn.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rn.index, y=rn.values,
                        name="Real wealth",
                        line=dict(color="green", width=1.5, dash="dash"),
                    ),
                    row=1, col=1,
                )

    # ---- Bottom: weight heatmap ----
    # Align weights to backtest dates (one row per month) so the heatmap matches the timeline above
    w = weights.reindex(results.index.sort_values(), method="ffill").fillna(0)
    if not w.empty and not w.columns.empty:
        # Focus on periods where the portfolio is actually invested
        total = w.sum(axis=1)
        if total.gt(0.01).any():
            first_live = total[total.gt(0.01)].index[0]
            w = w.loc[first_live:]

        # Suppress very small weights to reduce visual noise
        w = w.where(w >= 0.02, 0.0)

        priority = ["DBMG.L", "CMOD.L", "IGLT.L", "INXG.L", "ISF.L", "VMID.L", "IHYG.L", "SGLN.L"]
        cols = [c for c in priority if c in w.columns] + [c for c in w.columns if c not in priority]
        w = w[cols]
        fig.add_trace(
            go.Heatmap(
                x=w.index, y=w.columns, z=w.values.T,
                colorscale="Blues", zmin=0, zmax=0.25,
                hovertemplate="%{x|%b %Y}<br>%{y}: %{z:.1%}<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)
    fig.write_html(str(output_path))
    return output_path
