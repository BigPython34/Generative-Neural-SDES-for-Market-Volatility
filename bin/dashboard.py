"""
Dashboard Entrypoint
====================
Generates an interactive HTML dashboard with calibration and backtest results.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.dashboard_v2 import generate_dashboard


def main():
    return generate_dashboard()


if __name__ == "__main__":
    main()
