# Inflation Allocation (Fire & Ice)

A UK-focused, inflation-regime tactical allocation model based on Neville et al. (2021). It classifies the macro regime from published UK CPI (level × direction) and applies regime-specific portfolio weights with optional risk parity.

## Goals
- Classify inflation regimes (FIRE / BOOM / ICE / RECOVERY) from UK CPI
- Backtest regime-conditional allocation vs a 60/40 benchmark
- Report nominal and real wealth metrics, with BoE risk-free rate in Sharpe/Sortino

## Structure
- `fire_ice_model/` — main package
  - `data_ingestion/` — CPI (ONS/FRED), asset prices, BoE rate
  - `regime_engine/` — classifier, backtest, metrics
  - `allocation_logic/` — regime weights, risk parity
  - `trend_following/` — synthetic CTA (TSMOM)
  - `analysis/` — Plotly charts (cumulative returns, weight heatmap)
  - `tests/` — unit tests
- `fire_ice_model/config.yaml` — regime, allocation, backtest, and data settings
- Cache and outputs: `.cache/parquet/` (data), `reports/` (HTML charts) — not committed

## Setup
```bash
cd "/path/to/Inflation allocation"
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run
From the project root with the venv activated:
```bash
PYTHONPATH=. python fire_ice_model/regime_engine/backtest_engine.py
```
Or use the venv Python explicitly:
```bash
PYTHONPATH=. .venv/bin/python fire_ice_model/regime_engine/backtest_engine.py
```
Charts are written to `reports/backtest_charts.html` (open in a browser).

## Tests
```bash
PYTHONPATH=. python fire_ice_model/tests/test_classifier_acceleration.py
```
With pytest installed:
```bash
PYTHONPATH=. pytest fire_ice_model/tests/ -v
```
