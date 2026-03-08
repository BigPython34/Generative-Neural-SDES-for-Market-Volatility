#!/usr/bin/env python
"""
Joint SPX-VIX Calibration — CLI Script
========================================
Calibrates the rough Bergomi model simultaneously to:
  1. SPX/SPY implied volatility surface (multi-maturity smile)
  2. VIX term structure (CBOE indices: VIX1D → VIX1Y)
  3. VIX futures term structure (CBOE contracts)

This produces a Q-measure calibration consistent across both markets.

Usage
-----
    python bin/joint_calibration.py                    # Full calibration
    python bin/joint_calibration.py --quick             # Reduced grid (faster)
    python bin/joint_calibration.py --mc-paths 20000   # More MC paths
    python bin/joint_calibration.py --data-only         # Just show market data
    python bin/joint_calibration.py --update-config     # Write results to params.yaml

Output
------
    outputs/joint_calibration.json    — Full calibration report
    outputs/plots/joint_calibration_vix_ts.png     — VIX term structure fit
    outputs/plots/joint_calibration_spx_smile.png  — SPX smile fit
    outputs/plots/joint_calibration_xi0.png        — Forward variance curve
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Joint SPX-VIX Calibration (rough Bergomi)"
    )
    p.add_argument('--quick', action='store_true',
                   help='Reduced grid for faster calibration')
    p.add_argument('--mc-paths', type=int, default=10000,
                   help='Number of MC paths (default: 10000)')
    p.add_argument('--data-only', action='store_true',
                   help='Only load and display market data, no calibration')
    p.add_argument('--update-config', action='store_true',
                   help='Write calibrated params to config/params.yaml')
    p.add_argument('--lambda-spx', type=float, default=1.0,
                   help='SPX smile loss weight')
    p.add_argument('--lambda-vix', type=float, default=2.0,
                   help='VIX term structure loss weight')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip plot generation')
    return p.parse_args()


def generate_plots(result, save_dir: Path):
    """Generate diagnostic plots for the calibration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: VIX Term Structure Fit ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    market_taus = sorted(result.market_vix_ts.keys())
    model_taus = sorted(result.model_vix_ts.keys())
    all_taus = sorted(set(market_taus) | set(model_taus))

    mkt_vals = [result.market_vix_ts.get(t, np.nan) for t in all_taus]
    mdl_vals = [result.model_vix_ts.get(t, np.nan) for t in all_taus]

    ax.plot(all_taus, mkt_vals, 'ko-', markersize=8, linewidth=2,
            label='Market VIX', zorder=5)
    ax.plot(all_taus, mdl_vals, 'rs--', markersize=8, linewidth=2,
            label='Model VIX', zorder=4)

    for i, tau in enumerate(all_taus):
        if not np.isnan(mkt_vals[i]) and not np.isnan(mdl_vals[i]):
            diff = mdl_vals[i] - mkt_vals[i]
            color = 'green' if abs(diff) < 1.0 else 'orange' if abs(diff) < 2.0 else 'red'
            ax.annotate(f'{diff:+.2f}', (tau, mdl_vals[i]),
                       textcoords="offset points", xytext=(8, 8),
                       fontsize=8, color=color)

    ax.set_xlabel("Tenor (days)", fontsize=12)
    ax.set_ylabel("VIX Level", fontsize=12)
    ax.set_title(
        f"VIX Term Structure — Model vs Market\n"
        f"H={result.H:.3f}, η={result.eta:.2f}, ρ={result.rho:.2f}",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Custom x-ticks
    tick_positions = [t for t in all_taus if t in [1, 9, 30, 90, 180, 365]]
    ax.set_xticks(tick_positions if tick_positions else all_taus)
    ax.set_xticklabels([f"{t}d" for t in (tick_positions if tick_positions else all_taus)])

    fig.tight_layout()
    fig.savefig(save_dir / "joint_calibration_vix_ts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir / 'joint_calibration_vix_ts.png'}")

    # ── Plot 2: SPX Smile Fit (first 4 maturities) ──
    mat_keys = sorted(result.model_ivs.keys())
    n_mats = min(4, len(mat_keys))
    if n_mats > 0:
        fig, axes = plt.subplots(1, n_mats, figsize=(4.5 * n_mats, 4), squeeze=False)

        for i, T_key in enumerate(mat_keys[:n_mats]):
            ax = axes[0, i]
            strikes, model_ivs, market_ivs = result.model_ivs[T_key]
            strikes = np.array(strikes)
            model_ivs = np.array(model_ivs)
            market_ivs = np.array(market_ivs)

            valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)

            ax.plot(strikes[valid], market_ivs[valid] * 100, 'ko', markersize=4,
                    label='Market', alpha=0.7)
            ax.plot(strikes[valid], model_ivs[valid] * 100, 'r-', linewidth=1.5,
                    label='Model', alpha=0.8)

            dte = int(round(T_key * 365))
            rmse = np.sqrt(np.mean((model_ivs[valid] - market_ivs[valid]) ** 2)) * 10000
            ax.set_title(f"T={dte}d (RMSE={rmse:.0f}bps)", fontsize=10)
            ax.set_xlabel("Strike", fontsize=9)
            if i == 0:
                ax.set_ylabel("Implied Vol (%)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"SPX Smile Fit — H={result.H:.3f}, η={result.eta:.2f}, ρ={result.rho:.2f}",
            fontsize=13, fontweight='bold'
        )
        fig.tight_layout()
        fig.savefig(save_dir / "joint_calibration_spx_smile.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_dir / 'joint_calibration_spx_smile.png'}")

    # ── Plot 3: Forward Variance Curve ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    mats_d = result.xi0_maturities * 365
    xi_vals = result.xi0_values
    sigma_implied = np.sqrt(xi_vals) * 100

    ax.step(mats_d, sigma_implied, where='post', color='steelblue',
            linewidth=2, label='ξ₀(t)  →  σ(t)')
    ax.scatter(mats_d, sigma_implied, color='steelblue', s=40, zorder=5)

    ax.set_xlabel("Maturity (days)", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title("Forward Variance Curve ξ₀(t)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "joint_calibration_xi0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir / 'joint_calibration_xi0.png'}")


def update_config(result):
    """Write calibrated parameters to config/params.yaml."""
    import yaml

    config_path = ROOT / "config" / "params.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['bergomi']['hurst'] = round(float(result.H), 4)
    config['bergomi']['eta'] = round(float(result.eta), 3)
    config['bergomi']['rho'] = round(float(result.rho), 3)
    config['bergomi']['xi0'] = round(float(result.xi0_values[0]), 6)

    # Also update neural_sde fractional backbone
    if 'neural_sde' in config and 'fractional' in config['neural_sde']:
        config['neural_sde']['fractional']['hurst_init'] = round(float(result.H), 4)
        config['neural_sde']['fractional']['eta_init'] = round(float(result.eta), 3)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    print(f"\n  Updated config/params.yaml")
    print(f"    bergomi.hurst = {result.H:.4f}")
    print(f"    bergomi.eta   = {result.eta:.3f}")
    print(f"    bergomi.rho   = {result.rho:.3f}")
    print(f"    bergomi.xi0   = {result.xi0_values[0]:.6f}")


def main():
    args = parse_args()

    print("=" * 65)
    print("  JOINT SPX-VIX CALIBRATION — Rough Bergomi Model")
    print("  Bayer, Friz & Gatheral (2016)")
    print("=" * 65)

    # ── Load market data ──
    print("\n  Loading market data...")
    from quant.calibration.vix_futures_loader import assemble_calibration_data
    market_data = assemble_calibration_data(verbose=True)

    if args.data_only:
        print("\n" + market_data.summary())
        return

    # ── Run calibration ──
    from quant.calibration.joint_calibrator import JointCalibrator

    n_paths = 3000 if args.quick else args.mc_paths
    calibrator = JointCalibrator(
        lambda_spx=args.lambda_spx,
        lambda_vix=args.lambda_vix,
        n_mc_paths=n_paths,
        verbose=True,
    )

    result = calibrator.calibrate(market_data, quick=args.quick)

    # ── Save report ──
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    report = {
        'calibrated_params': {
            'H': float(result.H),
            'eta': float(result.eta),
            'rho': float(result.rho),
            'xi0_maturities': result.xi0_maturities.tolist(),
            'xi0_values': result.xi0_values.tolist(),
        },
        'losses': {
            'total': float(result.total_loss),
            'spx': float(result.spx_loss),
            'vix': float(result.vix_loss),
            'martingale': float(result.martingale_loss),
            'spx_rmse_bps': float(result.spx_rmse_bps),
        },
        'vix_term_structure': {
            'model': {str(k): v for k, v in result.model_vix_ts.items()},
            'market': {str(k): v for k, v in result.market_vix_ts.items()},
        },
        'diagnostics': {
            'n_mc_paths': result.n_mc_paths,
            'n_iterations': result.n_iterations,
            'elapsed_seconds': result.elapsed_seconds,
            'grid_evaluated': result.grid_evaluated,
            'method': result.method,
        },
    }

    report_path = output_dir / "joint_calibration.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # ── Generate plots ──
    if not args.no_plots:
        print("\n  Generating diagnostic plots...")
        plot_dir = ROOT / "outputs" / "plots"
        generate_plots(result, plot_dir)

    # ── Update config (always propagate calibrated params) ──
    update_config(result)
    if args.update_config:
        print("  (--update-config flag acknowledged)")

    print(f"\n  Joint calibration complete in {result.elapsed_seconds:.1f}s")
    print(f"     H = {result.H:.4f}, η = {result.eta:.3f}, ρ = {result.rho:.3f}")
    print(f"     SPX RMSE = {result.spx_rmse_bps:.0f} bps")
    print(f"     VIX L2   = {result.vix_loss:.4f}")


if __name__ == "__main__":
    main()
