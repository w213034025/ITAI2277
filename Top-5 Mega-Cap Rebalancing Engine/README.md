# Top-5 Mega-Cap Rebalancing Engine (Educational)

> **This is educational software, not financial, investment, or trading
> advice.** Automated trading carries real risk of capital loss from bugs,
> slippage, bad data, regime shifts, transaction costs, and events no
> backtest captures. Run on a **paper trading account only**. Past
> optimization results do not predict future returns. If this bot will ever
> manage money on behalf of someone else, registration/compliance
> requirements apply in most jurisdictions — that is a legal question for a
> licensed professional, not something this code addresses.

## What this is

A systematic allocation + rebalancing engine for a fixed 5-stock universe
(NVDA, AAPL, GOOGL, MSFT, AMZN as of late Aug 2026). It periodically:

1. Pulls recent price history.
2. Computes long-only, bounded (5–40% per name) mean-variance ("Modern
   Portfolio Theory") target weights (max Sharpe by default) via
   PyPortfolioOpt.
3. Compares target weights to actual current holdings (read from the
   broker, never assumed).
4. Rebalances only when drift exceeds a threshold, on a weekly/monthly
   calendar, or both ("hybrid" mode).
5. Sizes and submits marketable-limit orders through Alpaca's paper API.

It is **not** a day-trading or high-frequency system — the fastest cadence
is a daily drift check, and the actual trading cadence is expected to be
weekly-to-monthly.

## Project layout

```
config.py            All tunable parameters (universe, caps, thresholds, costs)
data_feed.py         yfinance price retrieval + validation (supports an end_date param for historical “as of” runs — see robustness_check.py)
features.py          Shared feature engineering (volatility, momentum, drawdown, beta) — used by both optimizer context and forecaster.py
optimizer.py         PyPortfolioOpt mean-variance optimization; accepts an optional mu_override to substitute a non-default
expected-return estimate
forecaster.py        Pooled ridge/LightGBM return forecaster + walk-forward evaluation utilities
portfolio_state.py   Current weights, drift calc, trade sizing
broker_alpaca.py     Alpaca paper-trading adapter
rebalance_engine.py  Orchestrates one full cycle, logs to SQLite
scheduler.py         APScheduler process for calendar/threshold cadence
backtest.py          Simple historical backtest vs equal-weight benchmark; mu_source="historical"|"ml" and compare_strategies()
robustness_check.py  Runs compare_strategies() across 6 historical windows to test whether a single backtest result generalizes
main.py              CLI entry point
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a **paper trading** account at https://alpaca.markets, then set:

```bash
export ALPACA_API_KEY="your_paper_key"
export ALPACA_SECRET_KEY="your_paper_secret"
export ALPACA_PAPER=true   # keep this true
```

## Usage

```bash
# Dry run — computes everything, logs it, submits NO orders
python main.py check

# Force a rebalance regardless of drift (still dry-run by default)
python main.py check --force

# Actually submit orders to your PAPER account
python main.py check --live

# Historical backtest vs. an equal-weight benchmark
python main.py backtest

# Same backtest, PLUS the experimental ML-forecaster strategy run side by
# side for comparison (does NOT change what the live engine does)
python main.py backtest --compare-ml

# Test whether a --compare-ml result holds across 6 different historical
# windows, or was a favorable one-off (see "AI/ML forecaster" below)
python robustness_check.py

# Long-running process that checks drift daily and rebalances on the
# configured calendar/threshold policy
python main.py schedule
```

Every cycle is logged to `logs/rebalancer.sqlite3` (target weights, actual
weights, computed drift, trades attempted, whether they executed) so you
can audit exactly what the optimizer decided and why, after the fact.

## AI/ML forecaster — built, evaluated, not deployed

`forecaster.py` is a pooled ridge/LightGBM model predicting next-5-day
returns per ticker, meant as an optional replacement for the optimizer's
default historical-mean expected-return estimate. It went through four
independent, pre-registered checks (gate, hyperparameter tuning + re-gate,
a live backtest comparison, and a 6-window robustness check) — all four
reached the same conclusion: it does not reliably beat the current
production default. Full writeup in `eda.ipynb` §9–§12 and
`design-document.md` §7.

It is fully wired (`optimizer.py`'s `mu_override`, `backtest.py`'s
`mu_source="ml"`) but switched off:

```python
# config.py
USE_ML_FORECASTER: bool = False   # flip only after re-validating on current data
```

The live engine (`rebalance_engine.py`) never reads this flag — only
`backtest.py` and `robustness_check.py` do, so this is diagnostic-only
regardless of its setting.

## Before you even think about going live

- [ ] Run in dry-run and paper mode for at least several weeks across
      different market conditions.
- [ ] Review every logged cycle — does the optimizer's behavior make sense?
- [ ] Stress-test with bad/missing data (`data_feed.py` raises rather than
      silently guessing, but confirm your alerting actually reaches you).
- [ ] Understand and accept the tax and cost implications of your actual
      rebalancing frequency.
- [ ] If any third party's money is involved, talk to a securities
      attorney before doing anything else.
- [ ] Replace the free yfinance feed with a paid, SLA-backed data source
      if you plan to run this unattended.

## Key parameters to sanity-check (config.py)

| Parameter | Default | Notes |
|---|---|---|
| `MIN_WEIGHT` | 0.05 | Diversification floor — every held name gets at least this |
| `MAX_WEIGHT` | 0.40 | Single-name cap |
| `LOOKBACK_DAYS` | 756 (~3y) | Mean/covariance estimation window |
| `DRIFT_THRESHOLD` | 0.04 | Absolute weight drift that triggers a trade |
| `CASH_BUFFER` | 0.07 | Always-uninvested cushion |
| `EST_SLIPPAGE_BPS` | 5.0 | Assumed slippage for sizing/backtest costs |
| `USE_ML_FORECASTER` | False | Gates the experimental forecaster — see above |

## Known simplifications / things to harden before relying on this

- `yfinance` is free but unofficial and rate-limited; swap in Polygon,
  Tiingo, or Alpaca market data for production reliability.
- Sample mean historical returns (the optimizer's default expected-return
  estimator) are notoriously noisy over short windows — a learned
  alternative was built and evaluated (see "AI/ML forecaster" above) but
  did not clear the bar to replace it.
- The backtest is a simplified daily-close model; it doesn't model
  intraday liquidity, partial fills, or borrowing costs.
- No corporate-action handling beyond what `yfinance`'s adjusted close
  already accounts for — verify around large splits/spinoffs.
- Universe is hardcoded; add a periodic market-cap-ranking check if you
  want it to track "current top 5" automatically.

## Known simplifications / things to harden before relying on this

- `yfinance` is free but unofficial and rate-limited; swap in Polygon,
  Tiingo, or Alpaca market data for production reliability.
- Sample mean historical returns (the optimizer's default expected-return
  estimator) are notoriously noisy over short windows — consider CAPM-based
  or shrinkage expected-return estimators for a 5-asset universe.
- The backtest is a simplified daily-close model; it doesn't model
  intraday liquidity, partial fills, or borrowing costs.
- No corporate-action handling beyond what `yfinance`'s adjusted close
  already accounts for — verify around large splits/spinoffs.
- Universe is hardcoded; add a periodic market-cap-ranking check if you
  want it to track "current top 5" automatically.
