"""
Central configuration for the rebalancing engine.

EDUCATIONAL USE ONLY. Not financial advice. Paper-trade only until you have
independently validated every assumption below.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List

# --- Universe -----------------------------------------------------------
# Fixed top-5 by market cap as of late Aug 2026. Re-verify periodically —
# rankings shift (see refresh_universe.py for a semi-automated check).
TICKERS: List[str] = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN"]

# --- Optimization constraints -------------------------------------------
MIN_WEIGHT: float = 0.05         # diversification floor — every held name gets >= 5%
                                  # (was 0.0, i.e. long-only-only: the optimizer could and
                                  # did zero out names entirely, e.g. MSFT/AMZN in a live
                                  # `main.py check` run). Feasibility requires
                                  # len(TICKERS) * MIN_WEIGHT <= 1.0 — with 5 names, 0.05
                                  # is well inside the safe range (max feasible is 0.20).
MAX_WEIGHT: float = 0.40         # single-name cap to avoid concentration
RISK_FREE_RATE: float = 0.045    # update to current T-bill / SOFR-based rate
LOOKBACK_DAYS: int = 756         # ~3 years of daily data for cov/return est.
OPTIMIZATION_OBJECTIVE: str = "max_sharpe"  # or "min_volatility", "efficient_return"
TARGET_RETURN: float = None      # required if objective == "efficient_return"

# --- Rebalancing policy ---------------------------------------------------
REBALANCE_MODE: str = "hybrid"     # "calendar" | "threshold" | "hybrid"
CALENDAR_CADENCE: str = "weekly"   # "weekly" | "monthly"
CALENDAR_WEEKDAY: str = "fri"      # used if CALENDAR_CADENCE == "weekly"
DRIFT_THRESHOLD: float = 0.04      # 4% absolute drift triggers a trade
CASH_BUFFER: float = 0.07          # keep 7% uninvested at all times

# --- Experimental: ML return forecaster (forecaster.py) -------------------
# Gate result as of the last live-data validation run (see eda.ipynb §9):
# ridge and LightGBM both beat the naive historical-mean estimator on
# information coefficient (ridge 0.023 / LightGBM 0.054 vs. naive -0.066),
# but neither beat it on hit rate (0.526 / 0.527 vs. 0.529) — the
# pre-registered gate required both. LightGBM additionally clipped 59% of
# its predictions at the annualization safety rail (vs. 0% for ridge) and
# is the less trustworthy of the two. Net: NOT promoted to production use.
#
# Default is OFF. Flip to True only for an explicitly experimental backtest
# run (see backtest.run_backtest(mu_source="ml") / compare_strategies()) —
# the live engine (rebalance_engine.py) does not read this flag.
USE_ML_FORECASTER: bool = os.getenv("USE_ML_FORECASTER", "false").lower() == "true"
#USE_ML_FORECASTER: bool = True
ML_FORECASTER_MODEL_TYPE: str = "ridge"   # ridge, not lightgbm — see forecaster.py docstring
ML_FORECASTER_HORIZON_DAYS: int = 5       # matches the weekly calendar cadence
ML_FORECASTER_RETRAIN_DAYS: int = 21      # ~monthly; backtest.py retrains only when the model is this stale

# --- Cost / execution assumptions ----------------------------------------
EST_COMMISSION_PER_TRADE: float = 0.0   # most US equity brokers are $0/trade now
EST_SLIPPAGE_BPS: float = 5.0           # assumed slippage in basis points
MIN_TRADE_NOTIONAL: float = 50.0        # skip trades smaller than this ($)
USE_LIMIT_ORDERS: bool = True
LIMIT_OFFSET_BPS: float = 10.0          # marketable limit offset from mid

# --- Broker / environment -------------------------------------------------
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# --- Logging / storage -----------------------------------------------------
LOG_DB_PATH: str = os.getenv("REBAL_LOG_DB", "logs/rebalancer.sqlite3")

# --- Alerts (optional) ------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")


@dataclass
class RunConfig:
    """Bundled config snapshot, useful for logging what parameters produced a given trade."""
    tickers: List[str] = field(default_factory=lambda: list(TICKERS))
    min_weight: float = MIN_WEIGHT
    max_weight: float = MAX_WEIGHT
    risk_free_rate: float = RISK_FREE_RATE
    lookback_days: int = LOOKBACK_DAYS
    objective: str = OPTIMIZATION_OBJECTIVE
    drift_threshold: float = DRIFT_THRESHOLD
    cash_buffer: float = CASH_BUFFER

    def as_dict(self) -> Dict:
        return self.__dict__.copy()
