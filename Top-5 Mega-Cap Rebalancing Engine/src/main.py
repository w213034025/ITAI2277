"""
CLI entry point.

Examples:
    python main.py check                # one dry-run cycle, no orders sent
    python main.py check --live          # submit real paper-trading orders
    python main.py backtest              # run the historical backtest
    python main.py backtest --compare-ml # ALSO run the experimental ML-forecaster
                                          # strategy side by side (gate not cleared —
                                          # see config.USE_ML_FORECASTER's docstring)
    python main.py schedule              # start the long-running scheduler

Safety: 'check' defaults to dry_run=True. You must pass --live to actually
submit orders, and config.ALPACA_PAPER must be True unless you have
deliberately configured a live account (not recommended without extensive
paper-trading history, cost review, and — if managing others' money —
appropriate regulatory registration).
"""
import argparse

import config


def main():
    parser = argparse.ArgumentParser(description="Top-5 mega-cap rebalancing engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Run one rebalance cycle")
    p_check.add_argument("--live", action="store_true", help="Actually submit orders (still paper unless configured live)")
    p_check.add_argument("--force", action="store_true", help="Ignore drift threshold and rebalance now")

    p_backtest = sub.add_parser("backtest", help="Run historical backtest")
    p_backtest.add_argument("--compare-ml", action="store_true",
                             help="Also run the EXPERIMENTAL ML-forecaster strategy "
                                  "side by side (gate not cleared; see config.py)")
    sub.add_parser("schedule", help="Start the long-running scheduler")

    args = parser.parse_args()

    if args.command == "check":
        from rebalance_engine import run_rebalance_cycle
        if args.live and config.ALPACA_PAPER:
            print("NOTE: ALPACA_PAPER=True — orders will go to your paper account, not a live market.")
        run_rebalance_cycle(dry_run=not args.live, force=args.force)

    elif args.command == "backtest":
        if args.compare_ml:
            from backtest import compare_strategies
            df = compare_strategies()
            print(df.tail(15))
        else:
            from backtest import run_backtest
            df = run_backtest()
            print(df[["portfolio_value", "equal_weight_benchmark"]].tail(15))

    elif args.command == "schedule":
        from scheduler import main as scheduler_main
        scheduler_main()


if __name__ == "__main__":
    main()
