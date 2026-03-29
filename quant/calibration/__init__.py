"""
Calibration submodule.
Forward variance, VIX term structure, and joint calibration
for rough volatility models.
"""

__all__ = [
    "BergomiOptimizer",
    "calibrate_bergomi_to_smile",
    "ForwardVarianceCurve",
    "bootstrap_forward_variance",
    "bootstrap_from_surface",
    "bootstrap_xi0_from_vix",
    "extract_atm_term_structure",
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
    "JointCalibrator",
    "JointCalibrationResult",
    "ExtendedRoughBergomi",
    "GirsanovDrift",
    "NeuralSDEQModel",
    "NeuralSDEQResult",
    "train_neural_sde_q",
    "QLossWeights",
    "prepare_spx_slices",
]


def __getattr__(name):
    if name == "prepare_spx_slices":
        from quant.calibration.market_targets import prepare_spx_slices

        return prepare_spx_slices
    if name in {
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
    }:
        from quant.calibration import vix_futures_loader as m

        return getattr(m, name)
    if name in {
        "ForwardVarianceCurve",
        "bootstrap_forward_variance",
        "bootstrap_from_surface",
        "bootstrap_xi0_from_vix",
        "extract_atm_term_structure",
    }:
        from quant.calibration import forward_variance as m

        return getattr(m, name)
    if name in {"JointCalibrator", "JointCalibrationResult", "ExtendedRoughBergomi"}:
        from quant.calibration import joint_calibrator as m

        return getattr(m, name)
    if name in {"GirsanovDrift", "NeuralSDEQModel", "NeuralSDEQResult", "train_neural_sde_q", "QLossWeights"}:
        from quant.calibration import neural_sde_q_calibrator as m

        return getattr(m, name)
    if name in {"BergomiOptimizer", "calibrate_bergomi_to_smile"}:
        from quant.calibration import bergomi_optimizer as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
