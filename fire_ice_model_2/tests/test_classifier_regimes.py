import numpy as np
import pandas as pd

from fire_ice_model_2.regime_engine.classifier import RegimeClassifier, Regime


def _cpi_row(level_high: bool, rising: bool) -> pd.DataFrame:
    """Build a single-row CPI DataFrame with the desired level/direction."""
    cpi_yoy = 6.0 if level_high else 2.0
    accel = 0.1 if rising else -0.1
    return pd.DataFrame(
        {"cpi_yoy": [cpi_yoy], "acceleration": [accel]},
        index=pd.to_datetime(["2020-01-31"]),
    )


def test_regime_classifier_quadrants_threshold_mode():
    """RegimeClassifier should map level × direction to the four Neville regimes."""
    clf = RegimeClassifier()
    # Force threshold mode to avoid z-score complications
    clf.signal_mode = "threshold"
    clf.hard_threshold = 5.0

    cases = {
        Regime.FIRE:     _cpi_row(level_high=True,  rising=True),
        Regime.BOOM:     _cpi_row(level_high=True,  rising=False),
        Regime.ICE:      _cpi_row(level_high=False, rising=False),
        Regime.RECOVERY: _cpi_row(level_high=False, rising=True),
    }

    for expected_regime, df in cases.items():
        out = clf.classify(df)
        got = out["regime"].iloc[0]
        assert str(got) == expected_regime.value

