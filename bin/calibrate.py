"""
Calibration Entrypoint
======================
Runs Bergomi parameter calibration from market data.
Estimates Hurst, eta, xi0 from SPX/VIX data; optionally calibrates to options smile.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from quant.advanced_calibration import HighFrequencyAnalyzer


def main():
    analyzer = HighFrequencyAnalyzer(data_dir="data")
    params = analyzer.run_full_analysis()
    print("\nCalibration complete. Results saved to outputs/advanced_calibration.json")
    return params


if __name__ == "__main__":
    main()
