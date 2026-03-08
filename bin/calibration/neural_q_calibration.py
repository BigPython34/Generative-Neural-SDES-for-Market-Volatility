#!/usr/bin/env python
"""
Neural SDE Q-Calibration — CLI Script
=======================================
Trains a Girsanov drift correction λ_φ(v,t) on top of the P-measure
Neural SDE to produce Q-consistent dynamics that reprice SPX options
and VIX term structure.

Data Sources Used
-----------------
    - VIX Term Structure (CBOE): VIX1D → VIX1Y (6 tenors)
    - VIX Futures (CBOE): 6 contracts (11d → 225d)
    - SPY Options Surface: 2345 options across 20 maturities (1d → 179d)
    - VVIX: vol-of-vol prior for η
    - SOFR: risk-free rate

Usage
-----
    python bin/neural_q_calibration.py                    # Full training
    python bin/neural_q_calibration.py --quick             # Quick (50 epochs, 2048 paths)
    python bin/neural_q_calibration.py --epochs 500        # More epochs
    python bin/neural_q_calibration.py --H 0.03 --rho -0.95  # Custom seed params

Output
------
    outputs/neural_q_calibration.json       — Full report
    outputs/plots/neural_q_loss_curve.png   — Training loss curve
    outputs/plots/neural_q_vix_fit.png      — VIX term structure fit
    outputs/plots/neural_q_loss_components.png — Loss decomposition
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np

from quant.calibration.market_targets import prepare_spx_slices


def parse_args():
    p = argparse.ArgumentParser(
        description="Neural SDE Q-Calibration (Girsanov drift correction)"
    )
    p.add_argument('--quick', action='store_true',
                   help='Quick mode: 50 epochs, 2048 paths, reduced surface')
    p.add_argument('--epochs', type=int, default=200,
                   help='Number of training epochs (default: 200)')
    p.add_argument('--paths', type=int, default=4096,
                   help='MC paths per epoch (default: 4096)')
    p.add_argument('--lr', type=float, default=3e-3,
                   help='Learning rate (default: 3e-3)')
    p.add_argument('--H', type=float, default=None,
                   help='Hurst exponent seed (default: from rBergomi results)')
    p.add_argument('--eta', type=float, default=None,
                   help='Vol-of-vol seed (default: from rBergomi results)')
    p.add_argument('--rho', type=float, default=None,
                   help='Correlation seed (default: from rBergomi results)')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip plot generation')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')
    p.add_argument('--max-strikes', type=int, default=8,
                   help='Max strikes per maturity (default: 8)')
    p.add_argument('--max-maturities', type=int, default=None,
                   help='Max maturities to use (default: all)')
    return p.parse_args()


def load_rbbergomi_results() -> dict:
    """Load previous rBergomi joint calibration results if available."""
    result_path = ROOT / "outputs" / "joint_calibration.json"
    if result_path.exists():
        with open(result_path, 'r') as f:
            data = json.load(f)
        # Extract params from nested structure
        params = data.get('calibrated_params', {})
        return {
            'H': params.get('H', 0.03),
            'eta': params.get('eta', 1.0),
            'rho': params.get('rho', -0.90),
        }
    return {}


def generate_plots(result, save_dir: Path):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Training Loss Curve ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(1, len(result.loss_history) + 1)
    ax.semilogy(epochs, result.loss_history, 'b-', linewidth=1.5, alpha=0.7)

    # Smoothed curve
    if len(result.loss_history) > 20:
        window = min(20, len(result.loss_history) // 5)
        smoothed = np.convolve(result.loss_history,
                               np.ones(window) / window, mode='valid')
        ax.semilogy(range(window, len(result.loss_history) + 1),
                    smoothed, 'r-', linewidth=2, label='Smoothed')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Q-Loss', fontsize=12)
    ax.set_title('Neural SDE Q-Calibration — Training Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "neural_q_loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'neural_q_loss_curve.png'}")

    # ── Plot 2: VIX Term Structure Fit ──
    if result.model_vix_ts and result.market_vix_ts:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        taus = sorted(set(list(result.model_vix_ts.keys()) +
                          list(result.market_vix_ts.keys())))
        mkt_vals = [result.market_vix_ts.get(t, np.nan) for t in taus]
        mdl_vals = [result.model_vix_ts.get(t, np.nan) for t in taus]

        ax.plot(taus, mkt_vals, 'ko-', markersize=8, linewidth=2,
                label='Market VIX', zorder=5)
        ax.plot(taus, mdl_vals, 'rs--', markersize=8, linewidth=2,
                label='Neural SDE Q-model', zorder=4)

        ax.set_xlabel('Tenor (days)', fontsize=12)
        ax.set_ylabel('VIX Level', fontsize=12)
        ax.set_title('VIX Term Structure — Neural SDE Q vs Market', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_dir / "neural_q_vix_fit.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_dir / 'neural_q_vix_fit.png'}")

    # ── Plot 3: Loss Components Bar Chart ──
    components = {k: v for k, v in result.loss_components.items() if k != 'total'}
    if components:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        names = list(components.keys())
        vals = [components[n] for n in names]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor='white')
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Q-Loss Component Decomposition', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        fig.tight_layout()
        fig.savefig(save_dir / "neural_q_loss_components.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_dir / 'neural_q_loss_components.png'}")


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  NEURAL SDE Q-CALIBRATION PIPELINE")
    print("=" * 65)

    # ═════════════════════════════════════════════════════════════
    # Step 1: Load rBergomi seed parameters
    # ═════════════════════════════════════════════════════════════
    print("\n[Step 1] Loading seed parameters...")
    rbg = load_rbbergomi_results()

    H_seed = args.H if args.H is not None else rbg.get('H', 0.03)
    eta_seed = args.eta if args.eta is not None else rbg.get('eta', 1.0)
    rho_seed = args.rho if args.rho is not None else rbg.get('rho', -0.90)

    print(f"  Seed: H={H_seed:.4f}, η={eta_seed:.3f}, ρ={rho_seed:.3f}")
    if rbg:
        print(f"  Source: outputs/joint_calibration.json")
    else:
        print(f"  Source: defaults (no rBergomi results found)")

    # ═════════════════════════════════════════════════════════════
    # Step 2: Assemble ALL market data
    # ═════════════════════════════════════════════════════════════
    print("\n[Step 2] Assembling market data...")

    from quant.calibration.vix_futures_loader import (
        assemble_calibration_data, VIX_INDEX_TENORS,
    )

    market_data = assemble_calibration_data(verbose=True)

    # ── 2a: SPX spot + risk-free rate ──
    spot = market_data.spx_spot or 5500.0
    r = market_data.risk_free_rate or 0.0373
    vvix = market_data.vvix

    # ── 2b: VIX Term Structure (6 tenors: 1d → 1y) ──
    market_vix_ts = {}
    vts = market_data.vix_term_structure
    if vts is not None:
        for i, label in enumerate(vts.labels):
            tau_days = VIX_INDEX_TENORS[label]["tau_days"]
            market_vix_ts[tau_days] = float(vts.vix_levels[i])
    print(f"\n  VIX Index TS: {len(market_vix_ts)} tenors — "
          f"{sorted(market_vix_ts.keys())}d")

    # ── 2c: VIX Futures (6 contracts, longer tenors) ──
    vix_futures = {}
    vf = market_data.vix_futures_term
    if vf is not None:
        for i in range(vf.n_contracts):
            vix_futures[int(vf.days_to_exp[i])] = float(vf.prices[i])
    print(f"  VIX Futures:  {len(vix_futures)} contracts — "
          f"{sorted(vix_futures.keys())}d")

    # ── 2d: SPY Options Surface (2345 pts, 20 maturities) ──
    spx_slices = []
    if market_data.spx_surface is not None:
        max_strikes = args.max_strikes
        max_maturities = args.max_maturities

        # In quick mode, reduce surface to 6 maturities × 6 strikes
        if args.quick:
            max_strikes = min(max_strikes, 6)
            if max_maturities is None:
                max_maturities = 6

        spx_slices = prepare_spx_slices(
            market_data.spx_surface, spot,
            max_strikes=max_strikes,
            max_maturities=max_maturities,
        )
        n_opts = sum(len(s['strikes']) for s in spx_slices)
        print(f"  SPX Surface:  {len(spx_slices)} maturities, {n_opts} options")
        print(f"    DTEs: {[s['dte'] for s in spx_slices]}")
        print(f"    Strikes/mat: {max_strikes}")
    else:
        print("  SPX Surface:  none available")

    # ── 2e: VVIX + SOFR ──
    print(f"  VVIX: {vvix:.1f}" if vvix else "  VVIX: N/A")
    print(f"  SOFR: {r:.3%}")

    if not market_vix_ts:
        print("\n  ERROR: No VIX term structure available.")
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════
    # Step 3: Train
    # ═════════════════════════════════════════════════════════════
    print("\n[Step 3] Training Neural SDE Q-model...")

    from quant.calibration.neural_sde_q_calibrator import train_neural_sde_q

    n_epochs = args.epochs
    n_paths = args.paths
    if args.quick:
        n_epochs = min(n_epochs, 50)
        n_paths = min(n_paths, 2048)
        print(f"  Quick mode: {n_epochs} epochs, {n_paths:,} paths")

    model, result = train_neural_sde_q(
        market_vix_ts=market_vix_ts,
        spot=spot,
        r=r,
        spx_slices=spx_slices,
        vix_futures=vix_futures,
        vvix=vvix,
        H_bergomi=H_seed,
        eta_bergomi=eta_seed,
        rho=rho_seed,
        n_epochs=n_epochs,
        n_paths=n_paths,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )

    # ═════════════════════════════════════════════════════════════
    # Step 4: Save results
    # ═════════════════════════════════════════════════════════════
    print("\n[Step 4] Saving results...")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    report = {
        'method': 'neural_sde_q_girsanov',
        'seed_params': {
            'H': H_seed,
            'eta': eta_seed,
            'rho': rho_seed,
        },
        'market_data': {
            'spot': spot,
            'r': r,
            'vvix': vvix,
            'vix_index_tenors': len(market_vix_ts),
            'vix_futures_contracts': len(vix_futures),
            'spx_maturities': len(spx_slices),
            'spx_total_options': sum(len(s['strikes']) for s in spx_slices),
        },
        'training': {
            'n_epochs': result.n_epochs,
            'n_paths': result.n_paths,
            'n_girsanov_params': result.n_girsanov_params,
            'elapsed_seconds': round(result.elapsed_seconds, 1),
        },
        'loss_components': result.loss_components,
        'loss_history_last10': result.loss_history[-10:],
        'vix_fit': {
            str(k): {'market': result.market_vix_ts.get(k, None),
                      'model': result.model_vix_ts.get(k, None)}
            for k in sorted(set(list(result.market_vix_ts.keys()) +
                                list(result.model_vix_ts.keys())))
        },
        'martingale_error': result.martingale_error,
    }

    report_path = out_dir / "neural_q_calibration.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {report_path}")

    # Save Q-model weights to disk for use in backtesting/pricing
    from quant.calibration.neural_sde_q_calibrator import save_q_model
    save_q_model(model)

    # ═════════════════════════════════════════════════════════════
    # Step 5: Plots
    # ═════════════════════════════════════════════════════════════
    if not args.no_plots:
        print("\n[Step 5] Generating plots...")
        plot_dir = ROOT / "outputs" / "plots"
        generate_plots(result, plot_dir)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
