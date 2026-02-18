"""Walk-forward backtest entrypoint."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.walk_forward_backtest import WalkForwardBacktester


def main():
    wf = WalkForwardBacktester(train_window_days=60, test_window_days=5)
    results = wf.run(max_folds=5)
    return results


if __name__ == "__main__":
    main()
