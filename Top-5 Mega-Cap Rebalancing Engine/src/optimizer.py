"""
Mean-variance (Modern Portfolio Theory) optimization via PyPortfolioOpt.

Note on inputs: sample-mean expected returns are extremely noisy over short
lookbacks. Consider swapping `expected_returns.mean_historical_return` for
`expected_returns.capm_return` or a shrinkage estimator, and always sanity
check the resulting weights before trusting them.
"""
from __future__ import annotations
import logging
from typing import Dict

import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

logger = logging.getLogger(__name__)


def compute_target_weights(
    prices: pd.DataFrame,
    min_weight: float,
    max_weight: float,
    risk_free_rate: float,
    objective: str = "max_sharpe",
    target_return: float | None = None,
    mu_override: pd.Series | None = None,
) -> Dict[str, float]:
    """
    Returns {ticker: weight} summing to 1.0 (long-only, capped).

    mu_override: if provided, used as the expected-return vector instead of
    PyPortfolioOpt's historical sample mean. EXPERIMENTAL — see forecaster.py
    and the EDA notebook's §9 validation gate before relying on this. As of
    the last gate check (live validation data), the ML forecaster beat the
    naive historical-mean estimator on information coefficient (ridge: 0.023
    vs. -0.066) but NOT on hit rate (0.526 vs. 0.529) — the pre-registered
    bar required both. Nothing in this codebase sets mu_override by default;
    config.USE_ML_FORECASTER gates it and defaults to False.
    """
    if mu_override is not None:
        mu = mu_override.reindex(prices.columns)
        if mu.isna().any():
            raise ValueError(f"mu_override is missing values for: {mu[mu.isna()].index.tolist()}")
    else:
        mu = expected_returns.mean_historical_return(prices)
    # Ledoit-Wolf shrinkage covariance is far more stable than sample cov
    # for a handful of highly correlated mega-cap tech names.
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))

    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == "min_volatility":
        ef.min_volatility()
    elif objective == "efficient_return":
        if target_return is None:
            raise ValueError("target_return must be set for 'efficient_return' objective")
        ef.efficient_return(target_return=target_return)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

    logger.info(
        "Optimized weights=%s | expected_return=%.2f%% volatility=%.2f%% sharpe=%.2f",
        cleaned, perf[0] * 100, perf[1] * 100, perf[2],
    )
    return dict(cleaned)


def apply_cash_buffer(weights: Dict[str, float], cash_buffer: float) -> Dict[str, float]:
    """Scale down risky-asset weights to leave `cash_buffer` fraction uninvested."""
    scale = 1.0 - cash_buffer
    scaled = {t: w * scale for t, w in weights.items()}
    scaled["CASH"] = cash_buffer
    return scaled
