"""
Multi-window robustness check for backtest.compare_strategies().

A single compare_strategies() run only tells you what happened over one
specific historical window — e.g. ml_mu_experimental beating historical_mu
once is NOT evidence the ML forecaster is actually better; with only ~6-7
trades per run, one favorable window is well within noise. This script runs
the comparison across several different "as of" end dates (via
backtest.run_backtest's end_date parameter, added specifically for this) and
reports whether any pattern holds consistently, or only showed up once.

This is diagnostic, not production code — it does not change USE_ML_FORECASTER
or wire anything into the live engine. It exists to answer one question
honestly: was the result you saw real, or a favorable draw?

Usage: python robustness_check.py
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta

import pandas as pd

import config
from backtest import compare_strategies
from data_feed import DataFeedError

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Spaced ~4 months apart so each window's trading period is meaningfully
# different, not just a near-duplicate of its neighbor. 0 = today (the run
# you already have). Adjust if you want finer/coarser spacing, but keep in
# mind each window still needs ~start_lookback+500 days of price history
# BEFORE its end date — going much further back than this risks running out
# of data (the live feed's history starts 2015-01-02).
END_DATE_OFFSETS_MONTHS = [0, 4, 8, 12, 16, 20]


def _months_ago(n: int) -> datetime:
    # Approximate (30-day months) — fine for this purpose; we don't need
    # calendar precision, just meaningfully different, well-spaced windows.
    return datetime.utcnow() - timedelta(days=30 * n)


def run_robustness_check(offsets_months=END_DATE_OFFSETS_MONTHS, **backtest_kwargs) -> pd.DataFrame:
    rows = []
    for months_back in offsets_months:
        end_date = None if months_back == 0 else _months_ago(months_back)
        label = "today" if months_back == 0 else f"{months_back}mo ago ({end_date.date()})"
        try:
            cmp = compare_strategies(end_date=end_date, **backtest_kwargs)
        except DataFeedError as e:
            logger.warning("Window '%s' skipped — insufficient/unavailable data: %s", label, e)
            continue
        except Exception as e:
            logger.warning("Window '%s' skipped — unexpected error: %s", label, e)
            continue

        final = cmp.iloc[-1]
        rows.append({
            "window": label,
            "historical_mu": final["historical_mu"],
            "ml_mu_experimental": final["ml_mu_experimental"],
            "equal_weight_benchmark": final["equal_weight_benchmark"],
            "ml_beats_historical": final["ml_mu_experimental"] > final["historical_mu"],
            "historical_beats_equal_weight": final["historical_mu"] > final["equal_weight_benchmark"],
        })

    if not rows:
        raise RuntimeError("Every window failed — check data access before trusting any single-window result.")

    results = pd.DataFrame(rows)

    n = len(results)
    ml_wins = results["ml_beats_historical"].sum()
    hist_wins_vs_eq = results["historical_beats_equal_weight"].sum()

    print(results.to_string(index=False))
    print()
    print(f"ML forecaster beat historical-mean in {ml_wins}/{n} windows.")
    print(f"Historical-mean (production default) beat equal-weight in {hist_wins_vs_eq}/{n} windows.")
    print()
    if ml_wins in (0, n):
        print(f"Consistent result across all {n} windows — worth taking seriously, "
              f"though {n} windows is still a small sample for a final conclusion.")
    else:
        print(f"MIXED result ({ml_wins}/{n}) — the single favorable run you saw earlier "
              f"was not representative of all tested windows. This is evidence AGAINST "
              f"treating that one result as a reason to trust the ML forecaster; it's "
              f"consistent with the §9/§11 gate's conclusion that it has not demonstrated "
              f"a reliable edge, not a reason to override that conclusion.")

    return results


if __name__ == "__main__":
    run_robustness_check()
