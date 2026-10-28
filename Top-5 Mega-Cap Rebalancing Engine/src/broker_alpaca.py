"""
Thin adapter over alpaca-py so the rest of the engine never talks to the
broker SDK directly (makes it easy to swap brokers or mock in tests).

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY env vars. Defaults to paper
trading (config.ALPACA_PAPER=True) — do not flip this to False until the
system has been run and reviewed on paper for an extended period.
"""
from __future__ import annotations
import logging
from typing import List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config
from portfolio_state import Position, TradeInstruction

logger = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Create a paper "
                "trading account and export these before running live."
            )
        self.client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER,
        )
        if not config.ALPACA_PAPER:
            logger.warning("ALPACA_PAPER is False — this client will submit LIVE orders.")

    def get_positions(self) -> List[Position]:
        raw = self.client.get_all_positions()
        return [
            Position(ticker=p.symbol, qty=float(p.qty), market_value=float(p.market_value))
            for p in raw
        ]

    def get_cash(self) -> float:
        account = self.client.get_account()
        return float(account.cash)

    def submit_trades(self, trades: List[TradeInstruction], dry_run: bool = True) -> List[dict]:
        """
        Submits marketable limit orders. dry_run=True (default) logs the
        orders without sending them — the caller must explicitly opt in to
        live submission, on top of the paper/live account setting.
        """
        results = []
        for t in trades:
            limit_price = self._limit_price(t)
            payload = dict(
                symbol=t.ticker,
                qty=t.qty,
                side=OrderSide.BUY if t.side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            if dry_run:
                logger.info("[DRY RUN] Would submit order: %s", payload)
                results.append({"dry_run": True, **payload})
                continue

            order = self.client.submit_order(LimitOrderRequest(**payload))
            logger.info("Submitted order id=%s %s %s @ %.2f", order.id, t.side, t.ticker, limit_price)
            results.append({"order_id": order.id, **payload})
        return results

    @staticmethod
    def _limit_price(t: TradeInstruction) -> float:
        offset = t.est_price * (config.LIMIT_OFFSET_BPS / 10_000)
        return round(t.est_price + offset, 2) if t.side == "buy" else round(t.est_price - offset, 2)
