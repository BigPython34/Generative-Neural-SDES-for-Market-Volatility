"""
Run Wasserstein-1 Calibration Comparison
========================================
Thin wrapper for Wasserstein calibration pipeline.
"""
from _bootstrap import bootstrap

ROOT = bootstrap()

from quant.calibration.wasserstein import run_wasserstein_calibration_pipeline

def main():
    run_wasserstein_calibration_pipeline()

if __name__ == "__main__":
    main()
