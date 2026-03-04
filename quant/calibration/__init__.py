"""
Calibration submodule.
Hurst estimation, forward variance, VIX term structure, and joint calibration
for rough volatility models.
"""

from quant.calibration.hurst import (
    estimate_hurst_variogram,
    estimate_hurst_dma,
    compute_realized_volatility,
)
from quant.calibration.bergomi_optimizer import BergomiOptimizer, calibrate_bergomi_to_smile
from quant.calibration.forward_variance import (
    ForwardVarianceCurve,
    bootstrap_forward_variance,
    bootstrap_from_surface,
    bootstrap_xi0_from_vix,
    extract_atm_term_structure,
)
from quant.calibration.vix_futures_loader import (
    VIXFuturesTerm,
    VIXFuturesData,
    VIXTermStructureSnapshot,
    CalibrationMarketData,
    load_vix_futures,
    load_vix_term_structure,
    get_vix_term_snapshot,
    load_vix_futures_continuous,
    load_vix_spot_history,
    assemble_calibration_data,
)
from quant.calibration.joint_calibrator import (
    JointCalibrator,
    JointCalibrationResult,
    ExtendedRoughBergomi,
)

__all__ = [
    # Hurst estimation
    "estimate_hurst_variogram",
    "estimate_hurst_dma",
    "compute_realized_volatility",
    # Bergomi optimizer (single-maturity)
    "BergomiOptimizer",
    "calibrate_bergomi_to_smile",
    # Forward variance curve
    "ForwardVarianceCurve",
    "bootstrap_forward_variance",
    "bootstrap_from_surface",
    "bootstrap_xi0_from_vix",
    "extract_atm_term_structure",
    # VIX data loaders
    "VIXFuturesTerm",
    "VIXFuturesData",
    "VIXTermStructureSnapshot",
    "CalibrationMarketData",
    "load_vix_futures",
    "load_vix_term_structure",
    "get_vix_term_snapshot",
    "load_vix_futures_continuous",
    "load_vix_spot_history",
    "assemble_calibration_data",
    # Joint calibration
    "JointCalibrator",
    "JointCalibrationResult",
    "ExtendedRoughBergomi",
]
