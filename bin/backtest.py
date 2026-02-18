"""
Backtesting Entrypoint
======================
Runs historical options backtest comparing Black-Scholes, Rough Bergomi, and Neural SDE.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from quant.backtesting import HistoricalBacktester


def main():
    bt = HistoricalBacktester()
    df = bt.run_real_backtest()
    if not df.empty:
        print("\nBacktest complete. Results saved to outputs/backtest_results.json")
    return df


if __name__ == "__main__":
    main()
