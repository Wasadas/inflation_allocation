import pandas as pd


def test_weight_sums_and_regime_coverage():
    """Basic invariants: weights sum to ~1 and regime coverage matches results length."""
    results = pd.read_parquet(".cache/parquet/backtest_results.parquet")
    weights = pd.read_parquet(".cache/parquet/backtest_weights.parquet")

    # Weight sums should be very close to 1.0 for all rows.
    row_sums = weights.sum(axis=1)
    assert (row_sums.between(0.99, 1.01)).all(), "Some weight rows do not sum close to 1.0"

    # Regime coverage: number of non-NaN regime entries should match number of non-NaN net_return rows.
    n_results = results["net_return"].dropna().shape[0]
    n_regimes = results["regime"].dropna().shape[0]
    assert n_regimes == n_results, f"Regime coverage {n_regimes} != results length {n_results}"

