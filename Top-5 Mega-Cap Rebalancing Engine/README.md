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
2. Computes long-only, capped mean-variance ("Modern Portfolio Theory")
   target weights (max Sharpe by default) via PyPortfolioOpt.
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
data_feed.py          yfinance price retrieval + validation
optimizer.py           PyPortfolioOpt mean-variance optimization
portfolio_state.py    Current weights, drift calc, trade sizing
broker_alpaca.py      Alpaca paper-trading adapter
rebalance_engine.py    Orchestrates one full cycle, logs to SQLite
scheduler.py          APScheduler process for calendar/threshold cadence
backtest.py           Simple historical backtest vs equal-weight benchmark
main.py               CLI entry point
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

# Long-running process that checks drift daily and rebalances on the
# configured calendar/threshold policy
python main.py schedule
```

Every cycle is logged to `logs/rebalancer.sqlite3` (target weights, actual
weights, computed drift, trades attempted, whether they executed) so you
can audit exactly what the optimizer decided and why, after the fact.

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
| `MAX_WEIGHT` | 0.40 | Single-name cap |
| `LOOKBACK_DAYS` | 756 (~3y) | Mean/covariance estimation window |
| `DRIFT_THRESHOLD` | 0.04 | Absolute weight drift that triggers a trade |
| `CASH_BUFFER` | 0.07 | Always-uninvested cushion |
| `EST_SLIPPAGE_BPS` | 5.0 | Assumed slippage for sizing/backtest costs |

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
