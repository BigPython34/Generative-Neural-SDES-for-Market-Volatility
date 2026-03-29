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
    - SPX Options Surface: 2345 options across 20 maturities (1d → 179d)
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

from pathlib import Path

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np

from quant.calibration.cli_support import (
    load_previous_joint_result,
    prepare_spx_targets,
    vix_futures_to_dict,
    vix_snapshot_to_dict,
)
from quant.calibration.plotting import plot_neural_q_calibration
from quant.calibration.reporting import build_neural_q_report, save_json_report


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


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  NEURAL SDE Q-CALIBRATION PIPELINE")
    print("=" * 65)

    # ═════════════════════════════════════════════════════════════
    # Step 1: Load rBergomi seed parameters
    # ═════════════════════════════════════════════════════════════
    print("\n[Step 1] Loading seed parameters...")
    rbg = load_previous_joint_result(ROOT)

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

    from quant.calibration.market_data_vix import assemble_calibration_data

    market_data = assemble_calibration_data(verbose=True)

    # ── 2a: SPX spot + risk-free rate ──
    spot = market_data.spx_spot or 5500.0
    r = market_data.risk_free_rate or 0.0373
    vvix = market_data.vvix

    # ── 2b: VIX Term Structure (6 tenors: 1d → 1y) ──
    market_vix_ts = {}
    market_vix_ts = vix_snapshot_to_dict(market_data.vix_term_structure)
    print(f"\n  VIX Index TS: {len(market_vix_ts)} tenors — "
          f"{sorted(market_vix_ts.keys())}d")

    # ── 2c: VIX Futures (6 contracts, longer tenors) ──
    vix_futures = vix_futures_to_dict(market_data.vix_futures_term)
    print(f"  VIX Futures:  {len(vix_futures)} contracts — "
          f"{sorted(vix_futures.keys())}d")

    # ── 2d: SPX Options Surface (2345 pts, 20 maturities) ──
    spx_slices = []
    if market_data.spx_surface is not None:
        spx_slices = prepare_spx_targets(
            market_data.spx_surface,
            spot,
            quick=args.quick,
            max_strikes=args.max_strikes,
            max_maturities=args.max_maturities,
        )
        n_opts = sum(len(s['strikes']) for s in spx_slices)
        print(f"  SPX Surface:  {len(spx_slices)} maturities, {n_opts} options")
        print(f"    DTEs: {[s['dte'] for s in spx_slices]}")
        print(f"    Strikes/mat: {min(args.max_strikes, 6) if args.quick else args.max_strikes}")
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

    from quant.calibration.neural_q import train_neural_sde_q

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

    report = build_neural_q_report(
        result=result,
        seed_params={'H': H_seed, 'eta': eta_seed, 'rho': rho_seed},
        market_summary={
            'spot': spot,
            'r': r,
            'vvix': vvix,
            'vix_index_tenors': len(market_vix_ts),
            'vix_futures_contracts': len(vix_futures),
            'spx_maturities': len(spx_slices),
            'spx_total_options': sum(len(s['strikes']) for s in spx_slices),
        },
        vix_futures=vix_futures,
    )

    report_path = save_json_report(out_dir / "neural_q_calibration.json", report)
    print(f"  Report: {report_path}")

    # Save Q-model weights to disk for use in backtesting/pricing
    from quant.calibration.neural_q import save_q_model
    save_q_model(model)

    # ═════════════════════════════════════════════════════════════
    # Step 5: Plots
    # ═════════════════════════════════════════════════════════════
    if not args.no_plots:
        print("\n[Step 5] Generating plots...")
        plot_dir = ROOT / "outputs" / "plots"
        plot_neural_q_calibration(result, plot_dir)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
