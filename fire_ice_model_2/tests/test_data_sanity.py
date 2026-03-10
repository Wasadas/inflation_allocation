import pandas as pd


def test_asset_price_sanity_ranges():
    """Basic sanity checks on cached asset prices: max drawdown and single-month moves."""
    prices = pd.read_parquet(".cache/parquet/asset_prices.parquet")
    rets = prices.pct_change().dropna(how="all")

    # Per-asset max drawdown should be within a loose but plausible band.
    for col in prices.columns:
        series = prices[col].dropna()
        if series.empty:
            continue
        nav = (series / series.iloc[0])
        dd = (nav - nav.cummax()) / nav.cummax()
        max_dd = float(dd.min())
        assert max_dd >= -0.99, f"{col}: max drawdown looks like a data error ({max_dd:.2%})"

    # Flag extreme single-month crashes; they should be very rare.
    extreme = (rets < -0.5).sum()
    # Allow at most a handful of such moves across the whole universe.
    assert int(extreme.sum()) < 10, f"Too many >50% monthly crashes detected: {int(extreme.sum())}"

