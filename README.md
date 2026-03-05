# Inflation Allocation (Fire & Ice) Research project analysing inflation regimes and their impact on cross-asset portfolio behaviour.

This project explores whether inflation regimes can help explain cross-asset portfolio behaviour. Inspired by Neville et al. (2021), the model classifies inflation environments using UK CPI level and direction and applies regime-specific allocation rules. The goal is not to optimise returns but to study how inflation dynamics affect asset class performance and portfolio risk.

## Discussion and limitations

In this simple implementation the Fire and Ice portfolio ends up trading less and hugging cash and bonds more than a classic 60/40, so it gives up some return and Sharpe in exchange for lower volatility. If I had more time I would tune the regime definitions and trend signals, and experiment with extra assets or risk controls, to see whether the idea survives in a richer but still realistic setup.
## Goals
Classify inflation regimes (FIRE / BOOM / ICE / RECOVERY) from UK CPI.
Backtest regime-conditional allocation vs a 60/40 benchmark.
Report nominal and real wealth metrics, with BoE risk-free rate in Sharpe/Sortino.

## Structure
`fire_ice_model/` — main package.
`data_ingestion/` — CPI (ONS/FRED), asset prices, BoE rate.
`regime_engine/` — classifier, backtest, metrics.
`allocation_logic/` — regime weights, risk parity.
`trend_following/` — synthetic CTA (TSMOM).
`analysis/` — Plotly charts (cumulative returns, weight heatmap).
`tests/` — unit tests.
`fire_ice_model/config.yaml` — regime, allocation, backtest, and data settings.
Cache and outputs live under `.cache/parquet/` for data and `reports/` for HTML charts; these are not committed.

## Results (sample backtest, 2005–2026)

| Metric | Fire & Ice | 60/40 Benchmark |
|--------|------------|-----------------|
| Ann. return | 0.9% | 4.3% |
| Ann. vol | 5.3% | 8.1% |
| Sharpe | 0.18 | 0.53 |
| Max drawdown | -19.2% | -14.4% |

Charts are written to `reports/backtest_charts.html` after running the backtest.
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
python -m fire_ice_model.regime_engine.backtest_engine
```
Charts are written to `reports/backtest_charts.html` (open in a browser).

## Tests
```bash
python fire_ice_model/tests/test_classifier_acceleration.py
```
<img width="826" height="717" alt="Inflation_allocation" src="https://github.com/user-attachments/assets/d8e87385-27b2-4e44-a24e-bb67c8b12603" />


