"""Result dataclasses and summaries for calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointCalibrationResult:
    """Results of joint SPX-VIX calibration."""
    H: float
    eta: float
    rho: float
    xi0_maturities: np.ndarray
    xi0_values: np.ndarray

    total_loss: float
    spx_loss: float
    vix_loss: float
    martingale_loss: float

    model_vix_ts: dict
    market_vix_ts: dict
    model_ivs: dict
    spx_rmse_bps: float
    spx_shape_rmse_bps: float
    spx_atm_bias_bps: float

    n_mc_paths: int
    n_iterations: int
    elapsed_seconds: float
    grid_evaluated: int
    method: str

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  JOINT SPX-VIX CALIBRATION RESULTS")
        lines.append("=" * 65)
        lines.append(f"  Method:     {self.method}")
        lines.append(f"  MC paths:   {self.n_mc_paths:,}")
        lines.append(f"  Grid pts:   {self.grid_evaluated}")
        lines.append(f"  Time:       {self.elapsed_seconds:.1f}s")
        lines.append("")
        lines.append("  Calibrated Parameters:")
        lines.append(f"    H (Hurst)    = {self.H:.4f}  {'← ROUGH' if self.H < 0.25 else ''}")
        lines.append(f"    η (vol-vol)  = {self.eta:.3f}")
        lines.append(f"    ρ (correl)   = {self.rho:.3f}")
        lines.append(f"    ξ₀ pillars   = {len(self.xi0_maturities)}")
        lines.append("")
        lines.append("  Loss Decomposition:")
        lines.append(f"    L_total      = {self.total_loss:.6f}")
        lines.append(f"    L_SPX        = {self.spx_loss:.6f}  "
                     f"(IV RMSE = {self.spx_rmse_bps:.1f} bps)")
        lines.append(f"    L_VIX        = {self.vix_loss:.6f}")
        lines.append(f"    L_martingale = {self.martingale_loss:.6f}")
        lines.append("")
        lines.append("  SPX Smile Diagnostics:")
        lines.append(f"    Total RMSE    = {self.spx_rmse_bps:.0f} bps")
        lines.append(f"    ATM bias      = {self.spx_atm_bias_bps:+.0f} bps "
                     f"(SPX-VIX joint calibration tension)")
        lines.append(f"    Shape RMSE    = {self.spx_shape_rmse_bps:.0f} bps "
                     f"(smile shape only, de-meaned)")
        lines.append("")

        lines.append("  VIX Term Structure:")
        lines.append(f"    {'Tenor':>8} {'Market':>8} {'Model':>8} {'Diff':>8}")
        lines.append("    " + "─" * 36)
        for tau in sorted(set(list(self.model_vix_ts.keys()) + list(self.market_vix_ts.keys()))):
            mkt = self.market_vix_ts.get(tau, None)
            mdl = self.model_vix_ts.get(tau, None)
            if mkt is not None and mdl is not None:
                lines.append(f"    {tau:>6}d {mkt:>8.2f} {mdl:>8.2f} {mdl - mkt:>+8.2f}")
            elif mdl is not None:
                lines.append(f"    {tau:>6}d {'N/A':>8} {mdl:>8.2f}")

        lines.append("=" * 65)
        return "\n".join(lines)


@dataclass
class NeuralSDEQResult:
    """Results from Neural SDE Q-calibration."""
    rho: float
    H_bergomi: float
    eta_bergomi: float

    final_loss: float
    loss_components: dict
    loss_history: list

    n_girsanov_params: int
    n_epochs: int
    n_paths: int
    elapsed_seconds: float

    model_vix_ts: dict = field(default_factory=dict)
    market_vix_ts: dict = field(default_factory=dict)
    martingale_error: float = 0.0

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  NEURAL SDE Q-CALIBRATION RESULTS (Girsanov)")
        lines.append("=" * 65)
        lines.append(f"  Girsanov params:  {self.n_girsanov_params}")
        lines.append(f"  Epochs:           {self.n_epochs}")
        lines.append(f"  MC paths/epoch:   {self.n_paths:,}")
        lines.append(f"  Time:             {self.elapsed_seconds:.1f}s")
        lines.append("")
        lines.append("  Seed Parameters (from rBergomi):")
        lines.append(f"    H_Q   = {self.H_bergomi:.4f}")
        lines.append(f"    η     = {self.eta_bergomi:.3f}")
        lines.append(f"    ρ     = {self.rho:.3f}")
        lines.append("")
        lines.append("  Loss Components:")
        for k, v in self.loss_components.items():
            lines.append(f"    {k:>15} = {v:.6f}")
        lines.append("")

        if self.model_vix_ts and self.market_vix_ts:
            lines.append("  VIX Term Structure (Q-model):")
            lines.append(f"    {'Tenor':>8} {'Market':>8} {'Model':>8} {'Diff':>8}")
            lines.append("    " + "─" * 36)
            for tau in sorted(self.market_vix_ts.keys()):
                mkt = self.market_vix_ts.get(tau, None)
                mdl = self.model_vix_ts.get(tau, None)
                if mkt is not None and mdl is not None:
                    lines.append(f"    {tau:>6}d {mkt:>8.2f} {mdl:>8.2f} {mdl - mkt:>+8.2f}")

        lines.append(f"\n  Martingale error: {self.martingale_error:.6f}")
        lines.append("=" * 65)
        return "\n".join(lines)
