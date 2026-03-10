## Fire & Ice v2 — One-pager for reviewers

**Problem.** How should a GBP investor allocate across a small ETF universe when inflation moves between four macro regimes (FIRE, BOOM, ICE, RECOVERY)? The goal is to turn UK CPI releases into a simple, transparent regime signal and compare a regime-aware portfolio to a 60/40 benchmark.

**Approach.** The model classifies each month into one of four Neville-style regimes using UK CPI level and direction (z-scored or thresholded), applies regime-specific base weights, and optionally overlays risk parity and a CTA sleeve. The v2 pipeline lives under `fire_ice_model_2/` and is driven entirely from `config.yaml` so parameters (thresholds, windows, weights, CTA mode) are easy to inspect and change.

**Data and pipeline.** Monthly ETF prices (ISF.L, VMID.L, IGLT.L, INXG.L, IHYG.L, GSG, IGLN.L, DBMF) are pulled from `yfinance`; UK CPI comes from the ONS CSV generator; the BoE policy rate is pulled via the configured macro series. Raw data are cached as Parquet under `.cache/parquet/`, and the main CLI entry point is:

```bash
python -m fire_ice_model_2.regime_engine.backtest_engine
```

This runs the full backtest and writes an interactive Plotly dashboard to `reports/backtest_charts.html` with cumulative NAV vs 60/40 and a regime performance table.

**Headline results (v2, 2005–2026).**

| Sample          | Ann. return | Ann. vol | Sharpe | Max drawdown | n months |
|-----------------|-------------|----------|--------|--------------|----------|
| Full history    | 2.16%       | 6.54%    | 0.33   | -27.74%      | 235      |
| Ex-2008–2009    | 3.57%       | 5.94%    | 0.60   | -13.25%      | 211      |

Outside the 2008–2009 crisis window the portfolio delivers a Sharpe ratio around 0.6 with a max drawdown near -13%; the deeper -27% event is concentrated in a single 2008 FIRE episode where a commodity supercycle peak coincided with a systemic financial shock.

**Integrity and limitations.** The codebase includes small tests for regime classification, data sanity (max drawdowns and single-month crashes), and backtest invariants (weight sums and regime coverage). The design intentionally avoids grid-searching parameters or cherry-picking assets after viewing results; instead it starts from literature-backed logic and focuses on data quality (e.g. handling a gold split, filling DBMF’s pre-inception gap with a synthetic CTA stream). The main limitations are that regimes are detected from monthly CPI alone, the CTA sleeve is a simple TSMOM proxy, and the model is not tuned to perfectly hedge extreme crisis moves like 2008.

