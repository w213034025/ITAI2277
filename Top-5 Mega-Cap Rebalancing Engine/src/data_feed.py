"""
Price data retrieval. yfinance is free but unofficial/rate-limited — swap in
Polygon/Tiingo/Alpaca market data for production reliability. This module
isolates that decision behind get_price_history().
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFeedError(Exception):
    pass


def get_price_history(tickers: List[str], lookback_days: int, end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted close prices, columns=tickers, index=date.
    Raises DataFeedError if data is missing, stale, or has excessive gaps.

    end_date: if provided, fetches history ending at this date instead of now —
    lets a caller simulate "as of" an earlier point in time (e.g. a multi-window
    backtest robustness check). Defaults to None (today), unchanged from before.
    The live engine never passes this; only backtest.py's robustness check does.
    """
    end = end_date if end_date is not None else datetime.utcnow()
    start = end - timedelta(days=int(lookback_days * 1.6))  # pad for weekends/holidays

    raw = yf.download(
        tickers,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise DataFeedError("yfinance returned no data — check tickers/network.")

    # Normalize into a simple close-price wide frame regardless of yfinance's
    # single- vs multi-ticker column shape.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    closes = closes.sort_index().tail(lookback_days)

    _validate(closes, tickers, as_of=end)
    return closes


def _validate(prices: pd.DataFrame, tickers: List[str], as_of: Optional[datetime] = None) -> None:
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataFeedError(f"No price series returned for: {missing}")

    if prices.isna().sum().sum() > 0:
        na_frac = prices.isna().mean()
        bad = na_frac[na_frac > 0.02]
        if not bad.empty:
            raise DataFeedError(f"Excessive missing data (>2%) in: {bad.index.tolist()}")
        prices.ffill(inplace=True)

    # Staleness is relative to the requested as-of date, not real-world "now" —
    # otherwise every historical ("as of N months ago") fetch would falsely warn,
    # since its last row is always old relative to today by design.
    reference_now = as_of if as_of is not None else pd.Timestamp.utcnow().tz_localize(None)
    last_date = prices.index.max()
    staleness = (reference_now - last_date).days
    if staleness > 5:
        logger.warning("Latest price data is %s days old (last=%s) — market may be closed, "
                        "or feed may be stale.", staleness, last_date.date())


def get_latest_prices(tickers: List[str]) -> dict:
    """Fast path: current/last prices only, for drift checks and order sizing."""
    data = yf.download(tickers, period="1d", progress=False, group_by="ticker", threads=True)
    prices = {}
    for t in tickers:
        try:
            prices[t] = float(data[t]["Close"].dropna().iloc[-1]) if isinstance(data.columns, pd.MultiIndex) \
                else float(data["Close"].dropna().iloc[-1])
        except Exception as e:
            raise DataFeedError(f"Could not fetch latest price for {t}: {e}")
    return prices
