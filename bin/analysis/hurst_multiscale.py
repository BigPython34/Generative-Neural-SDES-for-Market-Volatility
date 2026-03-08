import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Multi-Scale Hurst Exponent Estimation
=======================================

Rigorous estimation of the Hurst exponent H of realised volatility
across multiple time scales (5s → daily) and multiple assets (SPX, SPY, CAC40).

Methodology (Gatheral, Jaisson & Rosenbaum 2018):
  1. Compute daily Realized Variance from intraday log-returns at each frequency Δ
  2. Apply TSRV correction for ultra-high-frequency data (Δ ≤ 1min)
  3. Estimate H via variogram, structure function (q=1,2), ratio, and DMA on log(RV)
  4. Bootstrap confidence intervals (block bootstrap, block size ∝ n^{1/3})
  5. Inverse-variance weighted consensus across scales and methods
  6. Multifractal diagnostic: test ζ(q) linearity

Usage:
    python bin/hurst_multiscale.py                     # Full analysis
    python bin/hurst_multiscale.py --quick              # Fewer bootstraps
    python bin/hurst_multiscale.py --update-config      # Write consensus H to params.yaml
    python bin/hurst_multiscale.py --cross-asset        # Include SPY & CAC40

Outputs:
    outputs/hurst_multiscale_report.json    — Structured results
    outputs/plots/hurst_*.png                — Diagnostic plots
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.analysis.hurst_estimation import (
    DEFAULT_CROSS_ASSETS,
    DEFAULT_SPX_PATHS,
    MultiScaleResult,
    format_results_table,
    hurst_variogram,
    multifractal_spectrum,
    run_multiscale_hurst,
    save_results_json,
    test_monofractality,
)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def _ensure_plot_dir() -> Path:
    d = Path("outputs/plots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_variogram_loglog(result: MultiScaleResult, save_dir: Path) -> None:
    """
    Plot log-log variograms for each frequency, overlaid.

    This is the core diagnostic: a straight line with slope 2H.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.viridis

    freq_labels = list(result.rv_series.keys())
    n_freq = len(freq_labels)
    colors = [cmap(i / max(n_freq - 1, 1)) for i in range(n_freq)]

    for i, freq in enumerate(freq_labels):
        log_rv = result.rv_series[freq]
        est = hurst_variogram(log_rv, max_lag=min(80, len(log_rv) // 4))

        lags = est.lags_used
        log_lags = np.log(lags.astype(float))
        log_var = est.log_moments

        # Plot data
        ax.plot(log_lags, log_var, 'o', color=colors[i], markersize=3, alpha=0.6)

        # Plot regression line
        x_fit = np.array([log_lags[0], log_lags[-1]])
        y_fit = est.slope * x_fit + est.intercept
        ax.plot(x_fit, y_fit, '-', color=colors[i], linewidth=2,
                label=f"{freq}: H={est.H:.3f} (R²={est.r_squared:.3f})")

    # Reference lines for H=0.1 and H=0.5
    x_ref = np.array([0, np.log(80)])
    ax.plot(x_ref, 2 * 0.1 * x_ref - 2, '--', color='red', alpha=0.4,
            linewidth=1, label="H=0.10 (rough)")
    ax.plot(x_ref, 2 * 0.5 * x_ref - 2, '--', color='gray', alpha=0.4,
            linewidth=1, label="H=0.50 (Brownian)")

    ax.set_xlabel("log(lag τ) [trading days]", fontsize=12)
    ax.set_ylabel("log m(2, τ)", fontsize=12)
    ax.set_title(f"Variogram log-log — {result.asset}\n"
                 f"Consensus H = {result.consensus_H:.4f} ± {result.consensus_std:.4f}",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "hurst_variogram_loglog.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'hurst_variogram_loglog.png'}")


def plot_hurst_across_scales(result: MultiScaleResult, save_dir: Path) -> None:
    """
    Plot H estimates ± CI as a function of sampling frequency.

    Key diagnostic: if H is constant across scales → monofractal rough vol.
    If H increases with Δ → smoothing / aggregation effect.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq_labels = list(result.rv_series.keys())
    # Convert freq labels to approximate hours for x-axis ordering
    freq_hours = {
        "5s": 5 / 3600, "1m": 1 / 60, "5m": 5 / 60, "15m": 15 / 60,
        "30m": 30 / 60, "1h": 1, "daily": 6.5,
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    method_colors = {
        "variogram": "#1f77b4",
        "structure_q1": "#ff7f0e",
        "structure_q2": "#2ca02c",
        "ratio": "#d62728",
        "dma": "#9467bd",
    }
    offsets = {"variogram": -0.05, "structure_q1": -0.025, "structure_q2": 0,
               "ratio": 0.025, "dma": 0.05}

    for method, estimates in result.estimates.items():
        x_pos = []
        h_vals = []
        ci_lo = []
        ci_hi = []

        for i, est in enumerate(estimates):
            if i >= len(freq_labels):
                break
            freq = freq_labels[i]
            fh = freq_hours.get(freq, 1)
            x_pos.append(np.log10(fh) + offsets.get(method, 0))
            h_vals.append(est.H_point)
            ci_lo.append(est.ci_lower)
            ci_hi.append(est.ci_upper)

        x_pos = np.array(x_pos)
        h_vals = np.array(h_vals)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)

        # Ensure non-negative error bar values
        err_lo = np.maximum(h_vals - ci_lo, 0)
        err_hi = np.maximum(ci_hi - h_vals, 0)

        color = method_colors.get(method, "black")
        ax.errorbar(x_pos, h_vals,
                     yerr=[err_lo, err_hi],
                     fmt='o', color=color, capsize=4, markersize=6,
                     label=method, linewidth=1.5)

    # Consensus band
    H_c = result.consensus_H
    H_s = result.consensus_std
    ax.axhspan(H_c - 2 * H_s, H_c + 2 * H_s, alpha=0.15, color='blue',
               label=f"Consensus: {H_c:.4f} ± {H_s:.4f}")
    ax.axhline(H_c, color='blue', linewidth=1.5, alpha=0.6)

    # Reference lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label="H=0.5 (BM)")
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.4, label="H=0.1 (rough)")

    # X-axis ticks
    tick_pos = [np.log10(freq_hours[f]) for f in freq_labels if f in freq_hours]
    tick_labels = [f for f in freq_labels if f in freq_hours]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=10)

    ax.set_xlabel("Sampling frequency Δ", fontsize=12)
    ax.set_ylabel("Hurst exponent H", fontsize=12)
    ax.set_title(f"Hurst Exponent Across Scales — {result.asset}", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 0.7)

    fig.tight_layout()
    fig.savefig(save_dir / "hurst_across_scales.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'hurst_across_scales.png'}")


def plot_multifractal_spectrum(result: MultiScaleResult, save_dir: Path) -> None:
    """
    Plot ζ(q) vs q for each frequency.

    A straight line ζ(q) = H·q indicates monofractal behavior.
    Curvature indicates multifractality.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    freq_labels = list(result.rv_series.keys())
    cmap = plt.cm.viridis
    n_freq = len(freq_labels)
    colors = [cmap(i / max(n_freq - 1, 1)) for i in range(n_freq)]

    q_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Left panel: ζ(q) curves
    ax = axes[0]
    for i, freq in enumerate(freq_labels):
        log_rv = result.rv_series[freq]
        max_lag = min(60, len(log_rv) // 4)
        q_vals, zeta_q = multifractal_spectrum(log_rv, q_values=q_values, max_lag=max_lag)
        ax.plot(q_vals, zeta_q, 'o-', color=colors[i], label=freq, linewidth=1.5)

    # Reference: monofractal with H = consensus
    H_c = result.consensus_H
    ax.plot(q_values, H_c * q_values, 'k--', linewidth=1.5, alpha=0.5,
            label=f"ζ(q)={H_c:.3f}·q (monofractal)")

    ax.set_xlabel("Moment order q", fontsize=12)
    ax.set_ylabel("Scaling exponent ζ(q)", fontsize=12)
    ax.set_title("Multifractal Spectrum", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right panel: H(q) = ζ(q)/q  (should be constant for monofractal)
    ax = axes[1]
    for i, freq in enumerate(freq_labels):
        log_rv = result.rv_series[freq]
        max_lag = min(60, len(log_rv) // 4)
        q_vals, zeta_q = multifractal_spectrum(log_rv, q_values=q_values, max_lag=max_lag)
        H_q = zeta_q / q_vals
        ax.plot(q_vals, H_q, 'o-', color=colors[i], label=freq, linewidth=1.5)

    ax.axhline(H_c, color='blue', linewidth=1.5, alpha=0.5, linestyle='--',
               label=f"Consensus H = {H_c:.3f}")
    ax.axhline(0.5, color='gray', linewidth=1, alpha=0.3, linestyle='--')

    ax.set_xlabel("Moment order q", fontsize=12)
    ax.set_ylabel("H(q) = ζ(q)/q", fontsize=12)
    ax.set_title("Scale-dependent Hurst (constant → monofractal)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "hurst_multifractal_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'hurst_multifractal_spectrum.png'}")


def plot_bootstrap_distributions(result: MultiScaleResult, save_dir: Path) -> None:
    """
    Plot bootstrap distributions of H for the variogram estimator at each scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variogram_ests = result.estimates.get("variogram", [])
    freq_labels = list(result.rv_series.keys())

    n = len(variogram_ests)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for i, (est, ax) in enumerate(zip(variogram_ests, axes)):
        freq = freq_labels[i] if i < len(freq_labels) else f"scale_{i}"

        if len(est.all_H) > 0:
            ax.hist(est.all_H, bins=40, density=True, alpha=0.7, color='steelblue',
                    edgecolor='white')
            ax.axvline(est.H_point, color='red', linewidth=2,
                       label=f"H = {est.H_point:.4f}")
            ax.axvline(est.ci_lower, color='orange', linewidth=1.5, linestyle='--',
                       label=f"CI = [{est.ci_lower:.3f}, {est.ci_upper:.3f}]")
            ax.axvline(est.ci_upper, color='orange', linewidth=1.5, linestyle='--')

        ax.set_title(f"{freq}", fontsize=11)
        ax.set_xlabel("H", fontsize=10)
        if i == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Bootstrap Distributions of H (variogram) — {result.asset}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "hurst_bootstrap_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir / 'hurst_bootstrap_distributions.png'}")


def plot_logrv_timeseries(result: MultiScaleResult, save_dir: Path) -> None:
    """
    Plot log(RV) time series at each frequency (visual check for stationarity).
    Color-coded by frequency, with mean/std and bars-per-day info.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    freq_labels = list(result.rv_series.keys())
    n = len(freq_labels)
    if n == 0:
        return

    freq_colors = {
        "5s": "#e41a1c", "5m": "#377eb8", "15m": "#4daf4a",
        "30m": "#984ea3", "1h": "#ff7f00", "daily": "#a65628",
    }

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, freq in enumerate(freq_labels):
        ax = axes[i]
        log_rv = result.rv_series[freq]
        dates = result.dates.get(freq, None)
        color = freq_colors.get(freq, 'steelblue')
        
        # Plot with dates if available, otherwise use index
        if dates is not None and len(dates) == len(log_rv):
            ax.plot(dates, log_rv, linewidth=0.4, color=color, alpha=0.8)
            # Add rolling mean with dates
            if len(log_rv) > 60:
                roll = pd.Series(log_rv).rolling(30, min_periods=1).mean().values
                ax.plot(dates, roll, linewidth=1.5, color='black', alpha=0.5, label='30-day MA')
            
            # Format x-axis with dates
            date_range = (dates[-1] - dates[0]).days
            if date_range > 3650:  # > 10 years
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_minor_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            elif date_range > 1825:  # 5-10 years
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            elif date_range > 730:  # 2-5 years
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            else:  # < 2 years
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
        else:
            ax.plot(log_rv, linewidth=0.4, color=color, alpha=0.8)
            if len(log_rv) > 60:
                roll = pd.Series(log_rv).rolling(30, min_periods=1).mean().values
                ax.plot(roll, linewidth=1.5, color='black', alpha=0.5, label='30-day MA')
        
        ax.set_ylabel("log(RV)", fontsize=10)
        rv_annualized = np.sqrt(np.exp(np.mean(log_rv)) * 252) * 100
        ax.set_title(
            f"{freq} — {len(log_rv)} days, "
            f"mean={np.mean(log_rv):.2f}, std={np.std(log_rv):.2f}, "
            f"annualized σ ≈ {rv_annualized:.1f}%",
            fontsize=10, color=color, fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        if len(log_rv) > 60:
            ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle(f"Daily log(RV) Time Series — {result.asset}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "hurst_logrv_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir / 'hurst_logrv_timeseries.png'}")


def plot_cross_asset_comparison(
    results: dict[str, MultiScaleResult],
    save_dir: Path,
) -> None:
    """
    Compare H estimates across multiple assets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x_pos = np.arange(len(results))
    asset_names = list(results.keys())
    consensus = [r.consensus_H for r in results.values()]
    stds = [r.consensus_std for r in results.values()]

    colors = ['steelblue', '#ff7f0e', '#2ca02c', '#d62728']

    bars = ax.bar(x_pos, consensus, yerr=[2 * s for s in stds],
                  capsize=8, color=colors[:len(results)], alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    # Annotate
    for i, (h, s) in enumerate(zip(consensus, stds)):
        ax.text(i, h + 2 * s + 0.02, f"H={h:.4f}\n±{s:.4f}",
                ha='center', fontsize=10, fontweight='bold')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label="H=0.5 (BM)")
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.4, label="H=0.1 (rough)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(asset_names, fontsize=12)
    ax.set_ylabel("Consensus Hurst H", fontsize=12)
    ax.set_title("Cross-Asset Hurst Comparison (universality of roughness)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.1, 0.65)

    fig.tight_layout()
    fig.savefig(save_dir / "hurst_cross_asset.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'hurst_cross_asset.png'}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Scale Hurst Exponent Estimation for Rough Volatility"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fewer bootstraps (100 vs 500) for faster iteration")
    parser.add_argument("--cross-asset", action="store_true",
                        help="Include SPY & CAC40 in addition to SPX")
    parser.add_argument("--update-config", action="store_true",
                        help="Write consensus H to config/params.yaml")
    parser.add_argument("--max-lag", type=int, default=80,
                        help="Maximum lag for variogram (default: 80)")
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="Number of bootstrap replications (default: 500 or 100 if --quick)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    n_bootstrap = args.n_bootstrap or (100 if args.quick else 500)

    print("=" * 80)
    print("   MULTI-SCALE HURST EXPONENT ESTIMATION")
    print("   Gatheral, Jaisson & Rosenbaum (2018): 'Volatility is Rough'")
    print("=" * 80)
    print(f"\n  Settings: max_lag={args.max_lag}, n_bootstrap={n_bootstrap}")
    print(f"  Cross-asset: {args.cross_asset}, Update config: {args.update_config}\n")

    t0 = time.time()
    all_results: dict[str, MultiScaleResult] = {}

    # ── 1. SPX Multi-Scale ──────────────────────────────────────
    print("━" * 70)
    print("  [1] SPX — Multi-scale analysis (5s → daily)")
    print("━" * 70)

    spx_result = run_multiscale_hurst(
        asset_paths=DEFAULT_SPX_PATHS,
        max_lag=args.max_lag,
        n_bootstrap=n_bootstrap,
        verbose=True,
    )
    all_results["SPX"] = spx_result
    print(format_results_table(spx_result))

    # ── 2. Multifractal diagnostic ──────────────────────────────
    print("━" * 70)
    print("  [2] MULTIFRACTAL DIAGNOSTIC")
    print("━" * 70)

    for freq, log_rv in spx_result.rv_series.items():
        H_mono, r2_mono, curv = test_monofractality(log_rv, max_lag=min(60, len(log_rv) // 4))
        mono_str = "MONOFRACTAL" if r2_mono > 0.99 else "MULTIFRACTAL FEATURES"
        print(f"  {freq:>6s}: H_mono={H_mono:.4f}, R²={r2_mono:.4f}, "
              f"curvature={curv:.6f}  → {mono_str}")

    # ── 3. Cross-asset analysis ─────────────────────────────────
    if args.cross_asset:
        for label, fpath in DEFAULT_CROSS_ASSETS.items():
            if not Path(fpath).exists():
                print(f"\n  [{label}] SKIP — file not found: {fpath}")
                continue

            asset_name = label.split("_")[0]
            freq = label.split("_")[1] if "_" in label else "5m"

            print(f"\n{'━' * 70}")
            print(f"  [{label}] — Cross-asset analysis")
            print(f"{'━' * 70}")

            cross_result = run_multiscale_hurst(
                asset_paths={freq: fpath},
                max_lag=args.max_lag,
                n_bootstrap=n_bootstrap,
                verbose=True,
            )
            all_results[asset_name] = cross_result
            print(format_results_table(cross_result))

    # ── 4. Plots ────────────────────────────────────────────────
    if not args.no_plots:
        print("━" * 70)
        print("  [3] GENERATING DIAGNOSTIC PLOTS")
        print("━" * 70)

        save_dir = _ensure_plot_dir()

        plot_variogram_loglog(spx_result, save_dir)
        plot_hurst_across_scales(spx_result, save_dir)
        plot_multifractal_spectrum(spx_result, save_dir)
        plot_bootstrap_distributions(spx_result, save_dir)
        plot_logrv_timeseries(spx_result, save_dir)

        if len(all_results) > 1:
            plot_cross_asset_comparison(all_results, save_dir)

    # ── 5. Save JSON report ─────────────────────────────────────
    report_path = Path("outputs/hurst_multiscale_report.json")
    save_results_json(spx_result, report_path)
    print(f"\n  Report saved: {report_path}")

    # Also save cross-asset results
    if len(all_results) > 1:
        cross_report = {
            "consensus_comparison": {
                name: {"H": float(r.consensus_H), "std": float(r.consensus_std)}
                for name, r in all_results.items()
            }
        }
        cross_path = Path("outputs/hurst_cross_asset_report.json")
        with open(cross_path, "w") as f:
            json.dump(cross_report, f, indent=2)
        print(f"  Cross-asset report: {cross_path}")

    # ── 6. Update config if requested ───────────────────────────
    if args.update_config:
        _update_config(spx_result.consensus_H)

    # ── 7. Summary ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  FINAL CONSENSUS:  H = {spx_result.consensus_H:+.4f} "
          f"± {spx_result.consensus_std:.4f}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}\n")

    return all_results


def _update_config(H_consensus: float) -> None:
    """Write the consensus H to config/params.yaml (bergomi.hurst + fractional.hurst_init)."""
    import yaml

    config_path = Path("config/params.yaml")
    if not config_path.exists():
        print("  config/params.yaml not found - skipping update")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update bergomi.hurst
    import re
    old_bergomi = re.search(r'(bergomi:\s*\n\s*hurst:\s*)([\d.]+)', content)
    if old_bergomi:
        old_val = old_bergomi.group(2)
        content = content.replace(
            old_bergomi.group(0),
            old_bergomi.group(1) + f"{H_consensus:.4f}",
        )
        print(f"  bergomi.hurst: {old_val} -> {H_consensus:.4f}")

    # Update fractional.hurst_init
    old_frac = re.search(r'(hurst_init:\s*)([\d.]+)', content)
    if old_frac:
        old_val = old_frac.group(2)
        content = content.replace(
            old_frac.group(0),
            old_frac.group(1) + f"{H_consensus:.4f}",
        )
        print(f"  fractional.hurst_init: {old_val} -> {H_consensus:.4f}")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  config/params.yaml updated with H = {H_consensus:.4f}")


if __name__ == "__main__":
    results = main()
