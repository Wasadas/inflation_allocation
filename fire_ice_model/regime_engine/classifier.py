"""
regime_engine/classifier.py

The "Neville Switch" — classifies each month into one of four
inflation regimes based on Neville et al. (2021):

    FIRE      : High inflation + Rising trend
    BOOM      : High inflation + Falling trend
    ICE       : Low inflation  + Falling trend
    RECOVERY  : Low inflation  + Rising trend

Two signal modes (set in config.yaml → regime.signal_mode):

    "threshold" — Neville original: high = cpi_yoy > 5%
    "zscore"    — Bauer (2015): high = z_score > zscore_threshold
                  z_score = (cpi_yoy - rolling_mean) / rolling_std
                  This is time-aware: a 5% print in 2021 scored ~4 SD;
                  the same print in 1975 would score near zero.

Regime output mode (set in config.yaml → regime.use_regime_probability):

    false — Hard binary switch per regime (original Neville)
    true  — Soft sigmoid probabilities blended across all four regimes
            P(Fire) = sigmoid(k * (z - z_threshold))
            Weights in allocation_logic are then a continuous blend.
            This eliminates churn when CPI bounces near the threshold.
            Acceleration scaling is rolling-window only (no full-sample std)
            for walk-forward validity.

Hysteresis is applied on top of both modes to prevent whipsaws on
noisy monthly prints.

Information-set timing (cpi_lag_months):
    UK ONS publishes CPI with a lag (typically mid-month for the previous
    reference month). To avoid look-ahead bias, the classifier shifts the
    output so that the regime labeled at date T is the regime computed from
    CPI with reference month T - cpi_lag_months (i.e. the last CPI release
    that was available by end of month T). Weights at month-end T therefore
    use only information that was released by that date. See config regime.cpi_lag_months.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

REGIME_CFG = CONFIG["regime"]


# ------------------------------------------------------------------
# Regime Enum
# ------------------------------------------------------------------

class Regime(str, Enum):
    """
    The four Neville inflation regimes.
    Inherits from str for easy serialisation to parquet/CSV.
    """
    FIRE     = "FIRE"       # 🔥 High + Rising  — Commodities/CTA win
    BOOM     = "BOOM"       # 🌤️  High + Falling — Equities recover
    ICE      = "ICE"        # ❄️  Low + Falling  — Duration/Bonds win
    RECOVERY = "RECOVERY"   # 🌱 Low + Rising   — Cyclicals/Credit win

    @property
    def emoji(self) -> str:
        return {
            "FIRE":     "🔥",
            "BOOM":     "🌤️",
            "ICE":      "❄️",
            "RECOVERY": "🌱",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "FIRE":     "Inflationary Fire: High & Rising — bonds & equities both fail",
            "BOOM":     "Disinflationary Boom: High but Falling — equities stabilise",
            "ICE":      "Deflationary Ice: Low & Falling — duration is king",
            "RECOVERY": "Reflationary Recovery: Low but Rising — cyclicals lead",
        }[self.value]


# ------------------------------------------------------------------
# Regime Classifier
# ------------------------------------------------------------------

class RegimeClassifier:
    """
    Classifies each date in a CPI time-series into a Neville regime.

    Reads all parameters from config.yaml. Key settings:

        regime.signal_mode          : "threshold" or "zscore"
        regime.inflation_threshold  : hard level (used in threshold mode)
        regime.zscore_window        : rolling window for z-score normalisation
        regime.zscore_threshold     : SDs above mean = "high inflation"
        regime.use_regime_probability: True = sigmoid blend, False = hard switch
        regime.sigmoid_steepness    : k parameter controlling sigmoid sharpness
        regime.cpi_lag_months       : months to shift regime index (default 1) so that
                                      regime at T uses CPI released by T (no look-ahead).
        regime.hysteresis_months    : months raw signal must persist before confirmed switch (default 1).

    Parameters
    ----------
    hysteresis_months : int, optional
        If provided, overrides config. Otherwise regime.hysteresis_months is used (default 1).
        Setting to 1 avoids filtering out short-lived BOOM (disinflationary) transitions.
    """

    def __init__(self, hysteresis_months: Optional[int] = None):
        self.hysteresis_months   = (
            hysteresis_months if hysteresis_months is not None
            else REGIME_CFG.get("hysteresis_months", 1)
        )
        self.signal_mode         = REGIME_CFG.get("signal_mode", "threshold")
        self.hard_threshold      = REGIME_CFG.get("inflation_threshold", 5.0)
        self.zscore_window       = REGIME_CFG.get("zscore_window", 120)
        self.zscore_threshold    = REGIME_CFG.get("zscore_threshold", 1.5)
        self.use_probabilities   = REGIME_CFG.get("use_regime_probability", False)
        self.sigmoid_steepness   = REGIME_CFG.get("sigmoid_steepness", 3.0)
        self.cpi_lag_months      = REGIME_CFG.get("cpi_lag_months", 1)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def classify(self, cpi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        cpi_df : DataFrame from cpi_handler.get_uk_cpi()
                 Required columns: cpi_yoy, acceleration

        Returns
        -------
        DataFrame with original columns plus:

            Index: "Knowledge date" — when cpi_lag_months > 0, the index is shifted so that
            the regime at date T is based on CPI reference month T - cpi_lag_months (i.e.
            only data released by end of T is used). This enforces no look-ahead.

            z_score          : normalised inflation signal (zscore mode only)
            inflation_signal : the scalar used for level classification
                               (raw cpi_yoy in threshold mode, z_score in zscore mode)
            above_threshold  : bool — is inflation "high" by the active signal?
            rising           : bool — is trend accelerating?

            --- Hard switch output ---
            raw_regime       : Regime without hysteresis
            regime           : Confirmed regime (hysteresis applied)
            regime_changed   : True in the month of a confirmed transition

            --- Probability output (use_regime_probability=True) ---
            prob_fire        : P(Fire)     — sigmoid of level × direction
            prob_boom        : P(Boom)
            prob_ice         : P(Ice)
            prob_recovery    : P(Recovery)
            dominant_regime  : argmax of probabilities (for display)
        """
        required = ["cpi_yoy", "acceleration"]
        missing  = [c for c in required if c not in cpi_df.columns]
        if missing:
            raise ValueError(f"cpi_df missing required columns: {missing}")

        df = cpi_df.copy()

        # ---- 1. Compute the level signal ----
        df = self._compute_level_signal(df)

        # ---- 2. Direction signal (already in cpi_df as acceleration) ----
        df["rising"] = df["acceleration"] > 0

        # ---- 3. Hard regime classification + hysteresis ----
        df["raw_regime"] = df.apply(self._classify_row, axis=1)
        df["regime"]     = self._apply_hysteresis(df["raw_regime"])

        df["regime_changed"] = df["regime"] != df["regime"].shift(1)
        df.loc[df.index[0], "regime_changed"] = False

        # ---- 4. Sigmoid probabilities (optional) ----
        if self.use_probabilities:
            df = self._compute_probabilities(df)

        # ---- 5. Information-set timing: shift index by cpi_lag_months ----
        # Why we do this: UK ONS publishes CPI with a lag (typically around the middle of
        # the month for the *previous* calendar month). So at the end of January we have
        # just learned December's CPI; we have not yet seen January's CPI. The regime we
        # assign to "end of January" must therefore be based on December's CPI (the last
        # release we have), not January's. The input cpi_df has index = CPI reference
        # month (e.g. 2024-01-31 for January's data). We shift the index forward by
        # cpi_lag_months so that the row with reference month M is labeled at M + lag —
        # the first date when that regime is actually known. Then when the backtest
        # uses "regime at end of February", it gets the regime from January's CPI,
        # which was released in February. No look-ahead: weights at month-end T use
        # only CPI that was released by T.
        if self.cpi_lag_months and int(self.cpi_lag_months) > 0:
            lag = int(self.cpi_lag_months)
            df.index = df.index + pd.DateOffset(months=lag)
            logger.debug(
                "Regime index shifted by cpi_lag_months=%s so weights at date T use CPI released by T (no look-ahead).",
                lag,
            )

        return df

    # ------------------------------------------------------------------
    # Level signal
    # ------------------------------------------------------------------

    def _compute_level_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the scalar used to judge whether inflation is 'high'.

        threshold mode : inflation_signal = cpi_yoy
                         above_threshold  = cpi_yoy > hard_threshold
        zscore mode    : inflation_signal = z_score (rolling normalised)
                         above_threshold  = z_score > zscore_threshold
        """
        if self.signal_mode == "zscore":
            roll = df["cpi_yoy"].rolling(self.zscore_window, min_periods=24)
            df["z_score"] = (df["cpi_yoy"] - roll.mean()) / roll.std()
            df["inflation_signal"] = df["z_score"]
            df["above_threshold"]  = df["z_score"] > self.zscore_threshold
        else:
            # Original Neville hard threshold
            df["z_score"]          = np.nan
            df["inflation_signal"] = df["cpi_yoy"]
            df["above_threshold"]  = df["cpi_yoy"] > self.hard_threshold

        return df

    # ------------------------------------------------------------------
    # Hard classification
    # ------------------------------------------------------------------

    def _classify_row(self, row: pd.Series) -> Regime:
        """
        Four-quadrant Neville logic:
            Level × Direction → one of FIRE / BOOM / ICE / RECOVERY
        """
        above  = bool(row["above_threshold"])
        rising = bool(row["rising"])

        if above and rising:
            return Regime.FIRE
        elif above and not rising:
            return Regime.BOOM
        elif not above and not rising:
            return Regime.ICE
        else:
            return Regime.RECOVERY

    def _apply_hysteresis(self, raw: pd.Series) -> pd.Series:
        """
        Regime must persist for `hysteresis_months` consecutive months
        before it is confirmed. Prevents flipping on a single noisy print.

        Uses rolling mode: if there is a clear majority in the window,
        that regime is confirmed; otherwise carry forward the last confirmed.
        """
        h = self.hysteresis_months
        if h <= 0:
            return raw

        confirmed = pd.Series(index=raw.index, dtype=object)
        current   = raw.iloc[0]

        for i in range(len(raw)):
            window = raw.iloc[max(0, i - h + 1): i + 1]
            mode   = window.mode()
            if len(mode) == 1:
                current = mode.iloc[0]
            confirmed.iloc[i] = current

        return confirmed

    # ------------------------------------------------------------------
    # Sigmoid probabilities
    # ------------------------------------------------------------------

    def _compute_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the hard classification into continuous regime probabilities
        using a sigmoid function. This eliminates the discrete jump at the
        threshold boundary, so portfolio weights blend smoothly.

        Formulation
        -----------
        Let s = inflation_signal, θ = threshold, k = sigmoid_steepness.

            p_high  = sigmoid(k * (s - θ))    # prob inflation is "high"
            p_low   = 1 - p_high
            p_rise  = sigmoid(k * acceleration_scaled)  # prob trend is "rising"
            p_fall  = 1 - p_rise

        Then:
            P(Fire)     = p_high * p_rise
            P(Boom)     = p_high * p_fall
            P(Ice)      = p_low  * p_fall
            P(Recovery) = p_low  * p_rise

        The four probabilities sum to 1 by construction.

        Scaling of acceleration (no data leakage):
            We scale acceleration by a *rolling* standard deviation so that at each
            date we only use information available up to that date. Using the
            full-sample std would leak future volatility into the past and make
            the sigmoid sensitivity time-varying in a look-ahead way. The rolling
            window (config acceleration_scale_window, default 24 months) gives
            time-aware, stable normalization.
        """
        k         = self.sigmoid_steepness
        threshold = self.zscore_threshold if self.signal_mode == "zscore" else self.hard_threshold

        # Sigmoid of (signal - threshold): centred on the threshold
        p_high = 1 / (1 + np.exp(-k * (df["inflation_signal"] - threshold)))
        p_low  = 1 - p_high

        # Scale acceleration with rolling std so that we never use future data.
        # Walk-forward valid: at each date t only (t - window, t] is used; no full-sample std.
        # At date t we divide acceleration by the std over (t - window, t]; that way
        # the sigmoid sees a normalized "rising vs falling" signal without leaking
        # information from later periods. If the window has too few observations
        # we fall back to a small constant to avoid division by zero.
        scale_window = REGIME_CFG.get("acceleration_scale_window", 24)
        min_periods  = max(scale_window // 2, 6)
        accel_roll_std = df["acceleration"].rolling(window=scale_window, min_periods=min_periods).std()
        accel_scale = accel_roll_std.replace(0, np.nan).ffill().bfill()
        accel_scale = accel_scale.clip(lower=0.01)
        accel_scaled = df["acceleration"] / accel_scale
        p_rise = 1 / (1 + np.exp(-k * accel_scaled))
        p_fall = 1 - p_rise

        df["prob_fire"]     = (p_high * p_rise).round(4)
        df["prob_boom"]     = (p_high * p_fall).round(4)
        df["prob_ice"]      = (p_low  * p_fall).round(4)
        df["prob_recovery"] = (p_low  * p_rise).round(4)

        # Dominant regime for display (argmax of probabilities)
        prob_cols = ["prob_fire", "prob_boom", "prob_ice", "prob_recovery"]
        regime_map = {
            "prob_fire":     Regime.FIRE,
            "prob_boom":     Regime.BOOM,
            "prob_ice":      Regime.ICE,
            "prob_recovery": Regime.RECOVERY,
        }
        probs = df[prob_cols]
        valid = ~probs.isna().all(axis=1)
        dominant = pd.Series(index=df.index, dtype=object)
        dominant.loc[valid] = probs.loc[valid].idxmax(axis=1).map(regime_map)
        dominant = dominant.ffill().fillna(Regime.ICE)
        df["dominant_regime"] = dominant

        return df

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def get_regime_stats(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary statistics by regime:
            months, % of time, episode count, average episode duration.
        Uses 'dominant_regime' if probabilities are enabled, else 'regime'.
        """
        col = "dominant_regime" if "dominant_regime" in classified_df.columns else "regime"
        df  = classified_df.copy()
        df["episode"] = (df[col] != df[col].shift(1)).cumsum()

        rows = []
        for r in Regime:
            mask     = df[col] == r
            months   = int(mask.sum())
            pct      = months / len(df) * 100
            episodes = int(df.loc[mask, "episode"].nunique()) if months > 0 else 0
            avg_dur  = months / episodes if episodes > 0 else 0

            rows.append({
                "regime":          r.value,
                "emoji":           r.emoji,
                "months":          months,
                "pct_of_time":     round(pct, 1),
                "episodes":        episodes,
                "avg_duration_mo": round(avg_dur, 1),
            })

        return pd.DataFrame(rows).set_index("regime")

    def get_current_regime(self, classified_df: pd.DataFrame) -> dict:
        """Return the latest regime state with full context."""
        latest = classified_df.iloc[-1]

        # Prefer dominant_regime (probability mode) over hard regime
        if "dominant_regime" in classified_df.columns:
            regime = latest["dominant_regime"]
            col    = "dominant_regime"
        else:
            regime = latest["regime"]
            col    = "regime"

        # Pandas may coerce Enum values to plain strings (e.g. after mode(),
        # serialization, or mixed dtype columns). Coerce back to `Regime`
        # so `.emoji` / `.description` are always available.
        if not isinstance(regime, Regime):
            regime = Regime(str(regime))

        # Duration: count consecutive months of current regime from the end
        duration = 0
        for i in range(len(classified_df) - 1, -1, -1):
            if str(classified_df.iloc[i][col]) == regime.value:
                duration += 1
            else:
                break

        out = {
            "regime":          regime,
            "emoji":           regime.emoji,
            "description":     regime.description,
            "duration_months": duration,
            "cpi_yoy":         round(float(latest["cpi_yoy"]), 2),
            "acceleration":    round(float(latest["acceleration"]), 3),
            "signal_mode":     self.signal_mode,
        }

        # Z-score context
        if "z_score" in classified_df.columns and not pd.isna(latest.get("z_score")):
            out["z_score"] = round(float(latest["z_score"]), 2)

        # Probability context
        for col_p in ("prob_fire", "prob_boom", "prob_ice", "prob_recovery"):
            if col_p in classified_df.columns:
                out[col_p] = round(float(latest[col_p]), 3)

        # "Since" date
        # Compare against string value to be robust if the df column contains strings.
        regime_runs = classified_df[classified_df[col].astype(str) == regime.value]
        if len(regime_runs) >= duration:
            out["since"] = regime_runs.index[-duration].strftime("%b %Y")

        return out


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

def classify_regimes(cpi_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Module-level convenience wrapper around RegimeClassifier."""
    return RegimeClassifier(**kwargs).classify(cpi_df)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, force=True)

    from data_ingestion.cpi_handler import get_uk_cpi

    cpi    = get_uk_cpi(start="2000-01-01")
    clf    = RegimeClassifier()
    result = clf.classify(cpi)

    print(f"\nSignal mode      : {clf.signal_mode}")
    print(f"Probabilities    : {clf.use_probabilities}")

    cols = ["cpi_yoy", "inflation_signal", "acceleration", "raw_regime", "regime"]
    if clf.use_probabilities:
        cols += ["prob_fire", "prob_boom", "prob_ice", "prob_recovery", "dominant_regime"]
    print("\n--- Regime History (last 36 months) ---")
    print(result[cols].tail(36).to_string())

    print("\n--- Regime Statistics ---")
    print(clf.get_regime_stats(result).to_string())

    print("\n--- Current Regime ---")
    current = clf.get_current_regime(result)
    print(f"  {current['emoji']} {current['regime'].value}  ({current['description']})")
    print(f"  CPI YoY: {current['cpi_yoy']}%", end="")
    if "z_score" in current:
        print(f"  |  Z-score: {current['z_score']}", end="")
    print(f"  |  Acceleration: {current['acceleration']:+.3f}")
    print(f"  Duration: {current['duration_months']} months (since {current.get('since', '?')})")
    if clf.use_probabilities:
        print(f"\n  Regime probabilities:")
        for k, label in [("prob_fire","🔥 Fire"),("prob_boom","🌤️  Boom"),("prob_ice","❄️  Ice"),("prob_recovery","🌱 Recovery")]:
            bar = "█" * int(current.get(k, 0) * 30)
            print(f"    {label:<18} {current.get(k,0):.1%}  {bar}")