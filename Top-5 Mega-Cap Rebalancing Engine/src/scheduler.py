"""
Long-running scheduler process. Checks drift on a regular cadence and only
places trades when the configured policy says to. Intended to run on a
small always-on VPS, or be adapted into a cloud-function handler triggered
by EventBridge/cron.
"""
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from rebalance_engine import run_rebalance_cycle

logger = logging.getLogger(__name__)


def _job():
    try:
        force = config.REBALANCE_MODE == "calendar"
        run_rebalance_cycle(dry_run=True, force=force)  # flip dry_run=False only after review
    except Exception:
        logger.exception("Rebalance cycle failed — no trades were placed this run.")
        # Wire up config.TELEGRAM_* alerting here for production use.


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    scheduler = BlockingScheduler(timezone="America/New_York")

    if config.REBALANCE_MODE in ("calendar", "hybrid"):
        if config.CALENDAR_CADENCE == "weekly":
            trigger = CronTrigger(day_of_week=config.CALENDAR_WEEKDAY, hour=16, minute=5)
        else:  # monthly
            trigger = CronTrigger(day="1", hour=16, minute=5)
        scheduler.add_job(_job, trigger, name="calendar_rebalance")

    if config.REBALANCE_MODE in ("threshold", "hybrid"):
        # Check drift daily after close without forcing a trade unless it
        # exceeds the threshold (handled inside run_rebalance_cycle).
        scheduler.add_job(_job, CronTrigger(day_of_week="mon-fri", hour=16, minute=10),
                           name="threshold_check")

    logger.info("Scheduler starting in mode=%s cadence=%s", config.REBALANCE_MODE, config.CALENDAR_CADENCE)
    scheduler.start()


if __name__ == "__main__":
    main()
