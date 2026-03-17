"""
Calibration submodule.
Forward variance, VIX term structure, and joint calibration
for rough volatility models.
"""

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
    get_vix_term_snapshot_from_futures,
    load_vix_futures_continuous,
    load_vix_spot_history,
    assemble_calibration_data,
)
from quant.calibration.joint_calibrator import (
    JointCalibrator,
    JointCalibrationResult,
    ExtendedRoughBergomi,
)
from quant.calibration.neural_sde_q_calibrator import (
    GirsanovDrift,
    NeuralSDEQModel,
    NeuralSDEQResult,
    train_neural_sde_q,
    QLossWeights,
)
from quant.calibration.market_targets import prepare_spx_slices

__all__ = [
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
    "get_vix_term_snapshot_from_futures",
    "load_vix_futures_continuous",
    "load_vix_spot_history",
    "assemble_calibration_data",
    # Joint calibration
    "JointCalibrator",
    "JointCalibrationResult",
    "ExtendedRoughBergomi",
    # Neural SDE Q-calibration (Girsanov)
    "GirsanovDrift",
    "NeuralSDEQModel",
    "NeuralSDEQResult",
    "train_neural_sde_q",
    "QLossWeights",
    "prepare_spx_slices",
]
