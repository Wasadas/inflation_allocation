"""
Unit test: acceleration scaling in RegimeClassifier is rolling-window only.

If the classifier used full-sample std to scale acceleration, then changing
only the first month's acceleration would change the scale for all months
and thus all probabilities. With rolling-window scaling (window=24), only
months whose rolling window includes the first month are affected; months
after that window should have identical probabilities when we change only
the first month. This test asserts that later months are unchanged.
"""

import numpy as np
import pandas as pd
import pytest


def _minimal_cpi_df(n_months: int = 36, first_accel: float = 0.0) -> pd.DataFrame:
    """Build minimal CPI DataFrame with cpi_yoy and acceleration (as from get_uk_cpi)."""
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    cpi_yoy = np.ones(n_months) * 4.0  # below 5% so "low" in threshold mode
    acceleration = np.zeros(n_months)
    acceleration[0] = first_accel
    return pd.DataFrame(
        {"cpi_yoy": cpi_yoy, "acceleration": acceleration},
        index=dates,
    )


@pytest.fixture
def classifier():
    from fire_ice_model.regime_engine.classifier import RegimeClassifier
    return RegimeClassifier()


def test_acceleration_scaling_is_rolling_only(classifier):
    """
    Changing only the first month's acceleration must not change probabilities
    for months beyond the rolling window (24). With full-sample std it would.
    """
    scale_window = 24  # from config
    # Run A: first month acceleration = 0
    cpi_a = _minimal_cpi_df(n_months=36, first_accel=0.0)
    out_a = classifier.classify(cpi_a)
    # Run B: first month acceleration = 10 (large change)
    cpi_b = _minimal_cpi_df(n_months=36, first_accel=10.0)
    out_b = classifier.classify(cpi_b)

    prob_cols = ["prob_fire", "prob_boom", "prob_ice", "prob_recovery"]
    if not all(c in out_a.columns for c in prob_cols):
        pytest.skip("Classifier not in probability mode (use_regime_probability=False)")

    # Months after the rolling window: from row scale_window onward, probabilities
    # must be identical (only the first month's acceleration changed; rolling window
    # at these rows does not include that first month, so no change).
    if len(out_a) <= scale_window:
        pytest.skip("Not enough months to test rolling window")
    later_a = out_a.iloc[scale_window:]
    later_b = out_b.iloc[scale_window:]
    for c in prob_cols:
        np.testing.assert_array_almost_equal(
            later_a[c].values,
            later_b[c].values,
            err_msg=f"Rolling scaling: later months must be unchanged when only first month acceleration changes ({c})",
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from fire_ice_model.regime_engine.classifier import RegimeClassifier
    test_acceleration_scaling_is_rolling_only(RegimeClassifier())
    print("OK: acceleration scaling is rolling-only.")
