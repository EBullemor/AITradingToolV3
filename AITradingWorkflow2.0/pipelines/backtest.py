#!/usr/bin/env python3
"""
Backtest Pipeline

Validates strategies against historical data using walk-forward validation.

Usage:
    python pipelines/backtest.py --strategy fx_carry_momentum
    python pipelines/backtest.py --all
    python pipelines/backtest.py --strategy btc_trend_vol --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy to backtest (e.g., fx_carry_momentum)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backtest all strategies"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Backtest Pipeline")
    logger.info(f"Strategy: {args.strategy or 'ALL'}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info("=" * 60)
    
    # TODO: Implement backtest logic
    # This will be built in V1 (Weeks 4-8)
    
    logger.warning("Backtest pipeline not yet implemented")
    logger.info("This will be built during V1 phase (Weeks 4-8)")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
