"""
Turns (current holdings, latest prices, target weights) into a concrete,
costed trade list. No trade is ever inferred from guesses — everything here
is a deterministic function of broker-reported positions and fresh prices.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import config


@dataclass
class Position:
    ticker: str
    qty: float
    market_value: float


@dataclass
class TradeInstruction:
    ticker: str
    side: str          # "buy" | "sell"
    qty: float
    est_price: float
    est_notional: float
    reason: str


def compute_current_weights(positions: List[Position], cash: float) -> Dict[str, float]:
    total = sum(p.market_value for p in positions) + cash
    if total <= 0:
        raise ValueError("Total portfolio value is zero or negative — cannot compute weights.")
    weights = {p.ticker: p.market_value / total for p in positions}
    weights["CASH"] = cash / total
    return weights


def compute_drift(current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
    tickers = set(current) | set(target)
    return {t: current.get(t, 0.0) - target.get(t, 0.0) for t in tickers}


def needs_rebalance(drift: Dict[str, float], threshold: float) -> bool:
    return any(abs(v) > threshold for k, v in drift.items() if k != "CASH")


def build_trade_list(
    positions: List[Position],
    cash: float,
    target_weights: Dict[str, float],
    latest_prices: Dict[str, float],
) -> List[TradeInstruction]:
    """
    Compute buy/sell instructions to move from current holdings to target
    weights, net of a minimum trade notional filter (avoids churning on
    dust-sized drift) and applies an assumed slippage buffer to sizing.
    """
    total_value = sum(p.market_value for p in positions) + cash
    current_by_ticker = {p.ticker: p for p in positions}
    trades: List[TradeInstruction] = []

    for ticker, target_w in target_weights.items():
        if ticker == "CASH":
            continue
        price = latest_prices.get(ticker)
        if price is None or price <= 0:
            raise ValueError(f"Missing/invalid latest price for {ticker}; refusing to trade blind.")

        target_value = target_w * total_value
        current_value = current_by_ticker[ticker].market_value if ticker in current_by_ticker else 0.0
        delta_value = target_value - current_value

        if abs(delta_value) < config.MIN_TRADE_NOTIONAL:
            continue

        slip = price * (config.EST_SLIPPAGE_BPS / 10_000)
        exec_price = price + slip if delta_value > 0 else price - slip
        qty = abs(delta_value) / exec_price

        trades.append(TradeInstruction(
            ticker=ticker,
            side="buy" if delta_value > 0 else "sell",
            qty=round(qty, 4),
            est_price=round(exec_price, 2),
            est_notional=round(abs(delta_value), 2),
            reason=f"drift-to-target ({'under' if delta_value>0 else 'over'}weight)",
        ))

    return trades
