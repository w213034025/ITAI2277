"""
Shared feature engineering for the ML return forecaster (forecaster.py).

Mirrors the EDA notebook's feature set exactly (vol21, vol63, mom12m,
drawdown, beta_spy) so a model trained in the notebook and a model trained
here (backtest.py, and eventually the live engine) see identical inputs.
Requires a benchmark price series (SPY) for the beta feature.
"""
from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = ["vol21", "vol63", "mom12m", "drawdown", "beta_spy"]


def build_features(prices: pd.DataFrame, benchmark: pd.Series, tickers: List[str]) -> pd.DataFrame:
    """
    prices: wide adjusted-close frame, columns include every ticker in `tickers`.
    benchmark: adjusted-close Series for the benchmark (e.g. SPY), same index as prices.
    Returns a wide frame indexed by date, columns "{ticker}_{feature}" — the
    exact naming convention forecaster.FEATURE_COLUMNS expects.

    Early rows will be NaN until each rolling window has enough history
    (up to 126 days, for beta) — this is expected; forecaster.py's training
    step drops NaN rows, and a live/backtest caller should only read the
    most recent (fully warmed-up) row for prediction.
    """
    log_ret = np.log(prices[tickers] / prices[tickers].shift(1))
    bench_ret = np.log(benchmark / benchmark.shift(1))

    features = pd.DataFrame(index=prices.index)
    for t in tickers:
        features[f"{t}_vol21"] = log_ret[t].rolling(21).std() * np.sqrt(252)
        features[f"{t}_vol63"] = log_ret[t].rolling(63).std() * np.sqrt(252)
        features[f"{t}_mom12m"] = prices[t].pct_change(252)
        roll_max = prices[t].cummax()
        features[f"{t}_drawdown"] = prices[t] / roll_max - 1
        cov_m = log_ret[t].rolling(126).cov(bench_ret)
        var_m = bench_ret.rolling(126).var()
        features[f"{t}_beta_spy"] = cov_m / var_m
    return features
