"""
Minimal backtest: on each calendar check, re-optimizes weights on a rolling
lookback and trades ONLY if the current allocation has drifted more than
config.DRIFT_THRESHOLD from the last-set target — matching the live engine's
hybrid calendar + drift-gated policy, not a fixed trading schedule. Charges
an assumed cost per trade and keeps an explicit cash position (earning
config.RISK_FREE_RATE) between rebalances rather than letting it wash out
into the risky assets. This is a simplification (no intraday fills, no
partial liquidity constraints) — treat results as directional, not precise.

Fixed vs. the earlier version of this file (both bugs affected every
backtest number regardless of which expected-return estimator was used):
  1. Cash buffer washout — daily re-normalization was collapsing the
     CASH position back into the risky assets every single day, making
     the backtest look more fully-invested than the live engine ever is.
  2. Fixed-cadence trading — this file traded unconditionally every
     `rebalance_every_n_days`, ignoring DRIFT_THRESHOLD entirely, so the
     "hybrid" policy was actually pure calendar rebalancing.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from data_feed import get_price_history
from optimizer import compute_target_weights, apply_cash_buffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CASH = "CASH"
BENCHMARK = "SPY"


def _optimize_target(window: pd.DataFrame, fallback: pd.Series, today,
                      mu_override: Optional[pd.Series] = None) -> pd.Series:
    """Runs the optimizer + cash buffer and returns a weight Series indexed by
    TICKERS + CASH, summing to 1.0. Falls back to holding current weights
    (not silently dropping CASH) if optimization fails."""
    try:
        raw = compute_target_weights(
            window, config.MIN_WEIGHT, config.MAX_WEIGHT,
            config.RISK_FREE_RATE, config.OPTIMIZATION_OBJECTIVE,
            mu_override=mu_override,
        )
        target = apply_cash_buffer(raw, config.CASH_BUFFER)  # includes a "CASH" key
        return pd.Series({t: target.get(t, 0.0) for t in config.TICKERS + [CASH]})
    except Exception as e:
        logger.warning("Optimization failed on %s (%s) — holding weights.", today.date(), e)
        return fallback.copy()


class _MLMuProvider:
    """EXPERIMENTAL. Lazily (re)trains a ReturnForecaster on data strictly
    before `today` and returns its annualized mu prediction — walk-forward
    safe, same discipline as the EDA notebook's §9 gate. Retrains only when
    stale (config.ML_FORECASTER_RETRAIN_DAYS) rather than at every trade, to
    mirror exactly how the gate itself was evaluated.

    NOT used anywhere unless mu_source="ml" is explicitly passed to
    run_backtest() — see config.USE_ML_FORECASTER's docstring for why this
    defaults to unused.
    """

    def __init__(self):
        self._model = None
        self._last_train_date = None

    def mu_for(self, window: pd.DataFrame, bench_window: pd.Series, tickers, today) -> Optional[pd.Series]:
        from forecaster import ReturnForecaster
        import features as feat_mod

        stale = (self._model is None or self._last_train_date is None
                  or (today - self._last_train_date).days >= config.ML_FORECASTER_RETRAIN_DAYS)
        feats = feat_mod.build_features(window, bench_window, tickers)
        try:
            if stale:
                self._model = ReturnForecaster(
                    model_type=config.ML_FORECASTER_MODEL_TYPE,
                    horizon_days=config.ML_FORECASTER_HORIZON_DAYS,
                ).fit(feats, window, tickers)
                self._last_train_date = today
            return self._model.predict(feats.iloc[-1], tickers)
        except Exception as e:
            logger.warning("ML forecaster failed on %s (%s) — falling back to historical mu.", today.date(), e)
            return None


def run_backtest(start_lookback: int = config.LOOKBACK_DAYS,
                  rebalance_every_n_days: int = 5,
                  cost_bps: float = config.EST_SLIPPAGE_BPS,
                  drift_threshold: float = config.DRIFT_THRESHOLD,
                  mu_source: str = "historical",
                  end_date: Optional[object] = None) -> pd.DataFrame:
    """
    mu_source: "historical" (default, production behavior) or "ml"
    (EXPERIMENTAL — see config.USE_ML_FORECASTER's docstring and the EDA
    notebook's §9 gate before trusting "ml" results; the gate did not clear).

    end_date: if provided (a datetime), runs the backtest "as of" that date
    instead of today — this is what robustness_check.py uses to test whether
    a result (e.g. ml_mu_experimental beating historical_mu) holds across
    several historical windows, or was specific to one favorable period.
    """
    if mu_source not in ("historical", "ml"):
        raise ValueError(f"Unknown mu_source: {mu_source!r} (expected 'historical' or 'ml')")
    if mu_source == "ml" and not config.USE_ML_FORECASTER:
        logger.warning(
            "mu_source='ml' requested but config.USE_ML_FORECASTER is False — the "
            "validation gate has not cleared (see forecaster.py / eda.ipynb Sec 9). "
            "Running anyway since 'ml' was explicitly requested; treat this run as "
            "experimental only, not a production comparison."
        )

    fetch_tickers = config.TICKERS + ([BENCHMARK] if mu_source == "ml" else [])
    raw_prices = get_price_history(fetch_tickers, start_lookback + 500, end_date=end_date)
    prices = raw_prices[config.TICKERS]
    benchmark = raw_prices[BENCHMARK] if mu_source == "ml" else None
    returns = prices.pct_change().dropna()
    ml_provider = _MLMuProvider() if mu_source == "ml" else None
    daily_rf = config.RISK_FREE_RATE / 252  # cash earns the risk-free rate, not 0%

    # weights now explicitly includes CASH — this is what keeps the 7% buffer
    # from being silently renormalized away on non-rebalance days.
    weights = pd.Series(0.0, index=config.TICKERS + [CASH])
    weights[config.TICKERS] = 1.0 / len(config.TICKERS)  # start equal-weight, no cash, pre-first-optimization
    last_target = weights.copy()  # drift is measured against this, not a freshly recomputed target

    portfolio_value = [1.0]
    weight_history = []
    turnover_cost_total = 0.0
    n_checks = 0
    n_trades = 0

    for i in range(start_lookback, len(prices)):
        today = prices.index[i]
        day_return = (returns.loc[today] * weights[config.TICKERS]).sum() + weights[CASH] * daily_rf
        portfolio_value.append(portfolio_value[-1] * (1 + day_return))

        # drift the weights with price moves (assets) and the risk-free accrual (cash) —
        # this is a value reallocation, not a decision, so it happens every day regardless
        # of whether a rebalance check occurs below.
        grown = weights.copy()
        grown[config.TICKERS] = weights[config.TICKERS] * (1 + returns.loc[today])
        grown[CASH] = weights[CASH] * (1 + daily_rf)
        weights = grown / grown.sum()

        is_calendar_check = (i - start_lookback) % rebalance_every_n_days == 0
        if is_calendar_check:
            n_checks += 1
            # drift is measured on risky-asset weights only, against the last TRADED
            # target (not a freshly recomputed one) — matching "any position drifts
            # more than X% from target" in the live engine's policy.
            drift = (weights[config.TICKERS] - last_target[config.TICKERS]).abs().max()
            first_rebalance = (i == start_lookback)

            if first_rebalance or drift > drift_threshold:
                window = prices.iloc[i - start_lookback:i]
                mu_override = None
                if mu_source == "ml":
                    bench_window = benchmark.iloc[i - start_lookback:i]
                    mu_override = ml_provider.mu_for(window, bench_window, config.TICKERS, today)
                target = _optimize_target(window, fallback=weights, today=today, mu_override=mu_override)

                # turnover cost is computed over TICKERS only — moving into/out of CASH
                # is the mirror image of the asset trades already counted, not a second trade.
                turnover = (target[config.TICKERS] - weights[config.TICKERS]).abs().sum()
                turnover_cost_total += turnover * (cost_bps / 10_000)
                portfolio_value[-1] *= (1 - turnover * (cost_bps / 10_000))

                weights = target
                last_target = target.copy()
                n_trades += 1

        weight_history.append(weights.copy())

    result = pd.DataFrame(weight_history, index=prices.index[start_lookback:])
    result["portfolio_value"] = portfolio_value[1:]

    # named distinctly from `benchmark` (the SPY price series used above for ML
    # features, when mu_source="ml") — these are two different things and were
    # previously both called `benchmark`, which is an easy mistake to repeat.
    eq_weight_index = (1 + returns[config.TICKERS].mean(axis=1)).cumprod().reindex(result.index)
    result["equal_weight_benchmark"] = eq_weight_index / eq_weight_index.iloc[0]

    logger.info(
        "Backtest complete. Final value=%.3f | Equal-weight benchmark=%.3f | "
        "Cumulative turnover cost drag=%.4f | Calendar checks=%d | Trades executed=%d (%.0f%% of checks) | "
        "Mean cash weight=%.3f (target ~%.2f)",
        result["portfolio_value"].iloc[-1],
        result["equal_weight_benchmark"].iloc[-1],
        turnover_cost_total,
        n_checks, n_trades, 100 * n_trades / max(n_checks, 1),
        result[CASH].mean(), config.CASH_BUFFER,
    )
    return result


def compare_strategies(**kwargs) -> pd.DataFrame:
    """
    Runs the production strategy (historical-mean mu) and the EXPERIMENTAL
    ML-forecaster strategy on identical dates, costs, and drift/cash-buffer
    policy, alongside the equal-weight benchmark, for a direct side-by-side
    comparison.

    The 'ml_mu_experimental' column is exactly that — experimental. As of
    the last validation gate (eda.ipynb Sec 9), the ML forecaster beat the
    naive historical-mean estimator on information coefficient but not on
    hit rate, so it is not used in the live engine. This function exists so
    that claim is checkable by anyone reading this code, not just asserted.

    Any kwargs are passed through to both run_backtest() calls (e.g.
    start_lookback, rebalance_every_n_days, cost_bps, drift_threshold).
    """
    historical = run_backtest(mu_source="historical", **kwargs)
    ml = run_backtest(mu_source="ml", **kwargs)

    comparison = pd.DataFrame({
        "historical_mu": historical["portfolio_value"],
        "ml_mu_experimental": ml["portfolio_value"],
        "equal_weight_benchmark": historical["equal_weight_benchmark"],
    })

    final = comparison.iloc[-1]
    logger.info(
        "Strategy comparison complete. Final values — historical_mu=%.3f | "
        "ml_mu_experimental=%.3f (EXPERIMENTAL, gate not cleared) | equal_weight_benchmark=%.3f",
        final["historical_mu"], final["ml_mu_experimental"], final["equal_weight_benchmark"],
    )
    return comparison


if __name__ == "__main__":
    df = run_backtest()
    print(df[["portfolio_value", "equal_weight_benchmark"]].tail(10))
