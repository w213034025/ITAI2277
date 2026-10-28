"""
Orchestration layer: fetch data -> optimize -> compare to current holdings
-> decide whether to trade -> size trades -> submit (or log, in dry-run).

Run this directly for a one-off check, or point scheduler.py at
`run_rebalance_cycle` for calendar/hybrid cadences.
"""
from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config
from data_feed import get_price_history, get_latest_prices
from optimizer import compute_target_weights, apply_cash_buffer
from portfolio_state import compute_current_weights, compute_drift, needs_rebalance, build_trade_list
from broker_alpaca import AlpacaBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _log_to_db(record: dict) -> None:
    Path(config.LOG_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.LOG_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rebalance_events (
            ts TEXT, target_weights TEXT, current_weights TEXT,
            drift TEXT, trades TEXT, executed INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO rebalance_events VALUES (?,?,?,?,?,?)",
        (
            record["ts"],
            json.dumps(record["target_weights"]),
            json.dumps(record["current_weights"]),
            json.dumps(record["drift"]),
            json.dumps(record["trades"]),
            int(record["executed"]),
        ),
    )
    conn.commit()
    conn.close()


def run_rebalance_cycle(dry_run: bool = True, force: bool = False) -> dict:
    """
    One full cycle. Returns a summary dict (also persisted to sqlite).
    dry_run=True: compute and log everything, submit no orders.
    force=True: skip the drift-threshold gate and rebalance regardless
                (used for calendar mode, or manual runs).
    """
    logger.info("Starting rebalance cycle (dry_run=%s, force=%s)", dry_run, force)

    prices_hist = get_price_history(config.TICKERS, config.LOOKBACK_DAYS)
    raw_weights = compute_target_weights(
        prices_hist,
        min_weight=config.MIN_WEIGHT,
        max_weight=config.MAX_WEIGHT,
        risk_free_rate=config.RISK_FREE_RATE,
        objective=config.OPTIMIZATION_OBJECTIVE,
        target_return=config.TARGET_RETURN,
    )
    target_weights = apply_cash_buffer(raw_weights, config.CASH_BUFFER)

    broker = AlpacaBroker()
    positions = broker.get_positions()
    cash = broker.get_cash()
    current_weights = compute_current_weights(positions, cash)

    drift = compute_drift(current_weights, target_weights)
    should_trade = force or needs_rebalance(drift, config.DRIFT_THRESHOLD)

    trades = []
    executed = False
    if should_trade:
        latest_prices = get_latest_prices(config.TICKERS)
        trade_instructions = build_trade_list(positions, cash, target_weights, latest_prices)
        if trade_instructions:
            results = broker.submit_trades(trade_instructions, dry_run=dry_run)
            trades = [r for r in results]
            executed = not dry_run
        else:
            logger.info("Drift exceeded threshold but all deltas were below MIN_TRADE_NOTIONAL.")
    else:
        logger.info("Drift within threshold (%.2f%%) — no trade this cycle.", config.DRIFT_THRESHOLD * 100)

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "target_weights": target_weights,
        "current_weights": current_weights,
        "drift": drift,
        "trades": trades,
        "executed": executed,
    }
    _log_to_db(summary)
    logger.info("Cycle complete. executed=%s trades=%d", executed, len(trades))
    return summary


if __name__ == "__main__":
    # Safe default: dry run, threshold-gated. Change flags explicitly to go live.
    run_rebalance_cycle(dry_run=True, force=False)
