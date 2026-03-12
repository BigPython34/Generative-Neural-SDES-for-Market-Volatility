"""
Paper Results: Master Script
=============================
Runs all experiments and generates a comprehensive set of publication-ready
figures + JSON reports.

Experiments:
  1. Exotic pricing comparison (BS vs Bergomi vs Neural SDE)
  2. VRP P&L backtest (always_sell, historical, bergomi, regime_bergomi)
  3. Stress test comparison (Neural SDE vs Bergomi vs deterministic)
  4. Calibration summary (joint Bergomi + Q-model)
  5. Regime detection timeline (VVIX, VIX3M, SKEW, PCR signals)
  6. Deep hedging backtest (learned δ vs BS vs Bartlett under rough vol)

All outputs go to outputs/paper_results/

Usage:
  python bin/experiments/paper_results.py
  python bin/experiments/paper_results.py --skip-exotic --skip-stress
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = ROOT / "outputs" / "paper_results"


# =====================================================================
# Individual experiment runners
# =====================================================================

def run_exotic(n_paths: int = 10000):
    """Run exotic pricing comparison."""
    print("\n" + "=" * 70)
    print("  [1/5] EXOTIC OPTION PRICING COMPARISON")
    print("=" * 70)
    from bin.experiments.exotic_comparison import run_exotic_comparison
    return run_exotic_comparison(n_paths=n_paths, save_fig=True)


def run_backtest():
    """Run VRP backtest with all models."""
    print("\n" + "=" * 70)
    print("  [2/5] VRP P&L BACKTEST")
    print("=" * 70)

    from bin.backtesting.pnl_backtest import (
        _load_spx_daily, _load_vix_daily, _load_spx_intraday_5m,
        _load_regime_signals, run_vrp_backtest, print_report, save_results,
    )

    spx = _load_spx_daily()
    vix = _load_vix_daily()
    intra = _load_spx_intraday_5m()
    regime_signals = _load_regime_signals()

    print(f"  SPX: {len(spx)} rows, VIX: {len(vix)} rows")
    if regime_signals:
        print(f"  Regime signals: {', '.join(regime_signals.keys())}")

    models = ["always_sell", "historical", "bergomi", "regime_bergomi"]
    results = []

    for model in models:
        try:
            r = run_vrp_backtest(
                spx, vix, intra,
                model_name=model,
                horizon=21,
                start_date="2010-01-01",
                threshold_vol_pts=1.0,
                regime_signals=regime_signals,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {model} -> {e}")

    if results:
        print_report(results)
        save_results(results, OUT_DIR / "vrp_backtest.json")

    return results


def run_stress(n_paths: int = 5000):
    """Run stress test comparison."""
    print("\n" + "=" * 70)
    print("  [3/5] STRESS TEST COMPARISON")
    print("=" * 70)

    from bin.experiments.stress_test_comparison import main as stress_main
    # Import individual pieces instead
    from bin.experiments.stress_test_comparison import (
        generate_neural_sde_stress, generate_bergomi_stress,
        generate_deterministic_stress, print_report as stress_report,
        make_figures as stress_figures, _get_current_vix,
    )
    from utils.config import load_config
    from quant.risk.risk_engine import RiskEngine

    config = load_config()
    spot = config["pricing"]["spot"]
    r = config["pricing"]["risk_free_rate"]
    vix = _get_current_vix() / 100.0

    engine = RiskEngine(spot=spot, r=r)
    engine.add_position("call", strike=spot, T=0.25, quantity=-10, iv=vix)
    engine.add_position("put", strike=spot, T=0.25, quantity=-10, iv=vix)

    nsde_results = {}
    berg_results = {}
    try:
        nsde_results = generate_neural_sde_stress(engine, n_paths)
    except Exception as e:
        print(f"  Neural SDE stress failed: {e}")

    try:
        berg_results = generate_bergomi_stress(engine, n_paths)
    except Exception as e:
        print(f"  Bergomi stress failed: {e}")

    det_results = generate_deterministic_stress(engine)

    stress_report(nsde_results, berg_results, det_results)
    if nsde_results and berg_results:
        stress_figures(nsde_results, berg_results, det_results, OUT_DIR, engine)

    return {"neural_sde": nsde_results, "bergomi": berg_results,
            "deterministic": det_results}


def run_calibration_summary():
    """Summarize calibration status."""
    print("\n" + "=" * 70)
    print("  [4/5] CALIBRATION SUMMARY")
    print("=" * 70)

    from utils.config import load_config
    config = load_config()
    berg = config["bergomi"]

    summary = {
        "bergomi": {
            "hurst": berg["hurst"],
            "eta": berg["eta"],
            "rho": berg["rho"],
            "xi0": berg.get("xi0", 0.04),
        },
        "simulation": config["simulation"],
        "pricing": config["pricing"],
    }

    # Check if Q-model exists
    q_model_path = ROOT / "models" / "neural_sde_q_model.eqx"
    summary["q_model_available"] = q_model_path.exists()
    if q_model_path.exists():
        q_config_path = ROOT / "models" / "neural_sde_q_model_config.json"
        if q_config_path.exists():
            with open(q_config_path) as f:
                summary["q_model_config"] = json.load(f)

    # P-model
    p_model_path = ROOT / "models" / "neural_sde_best.eqx"
    summary["p_model_available"] = p_model_path.exists()

    # Joint calibration
    jcal_path = ROOT / "outputs" / "joint_calibration.json"
    if jcal_path.exists():
        with open(jcal_path) as f:
            summary["joint_calibration"] = json.load(f)

    print(f"  Bergomi: H={berg['hurst']}, η={berg['eta']}, ρ={berg['rho']}")
    print(f"  P-model: {'✓' if summary['p_model_available'] else '✗'}")
    print(f"  Q-model: {'✓' if summary['q_model_available'] else '✗'}")

    out = OUT_DIR / "calibration_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved -> {out.relative_to(ROOT)}")

    return summary


def run_regime_timeline():
    """Generate regime detection timeline from TradingView data."""
    print("\n" + "=" * 70)
    print("  [5/5] REGIME DETECTION TIMELINE")
    print("=" * 70)

    from quant.regimes.regime_detector import RegimeDetector

    detector = RegimeDetector()

    # Load all signals
    signal_files = {
        "VIX": ROOT / "data" / "trading_view" / "volatility" / "vix_daily.csv",
        "VVIX": ROOT / "data" / "trading_view" / "volatility" / "vvix_daily.csv",
        "VIX3M": ROOT / "data" / "trading_view" / "volatility" / "vix3m_daily.csv",
        "SKEW": ROOT / "data" / "trading_view" / "sentiment" / "skew_daily.csv",
        "PCR": ROOT / "data" / "trading_view" / "sentiment" / "pcspx_daily.csv",
    }

    signals = {}
    for name, path in signal_files.items():
        if path.exists():
            df = pd.read_csv(path)
            if "time" in df.columns:
                df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
                df = df.sort_values("date").drop_duplicates("date", keep="last")
            signals[name] = df

    if not signals:
        print("  No TradingView signal data found!")
        return None

    # Merge on common dates
    merged = None
    for name, df in signals.items():
        df_slim = df[["date", "close"]].rename(columns={"close": name})
        if merged is None:
            merged = df_slim
        else:
            merged = pd.merge(merged, df_slim, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"  Merged signal data: {len(merged)} common dates "
          f"[{merged['date'].min().date()} → {merged['date'].max().date()}]")

    # Detect regime at each date
    regimes = []
    for _, row in merged.iterrows():
        vix = row.get("VIX", np.nan)
        vvix = row.get("VVIX", np.nan)
        vix3m = row.get("VIX3M", np.nan)
        skew = row.get("SKEW", np.nan)
        pcr = row.get("PCR", np.nan)

        r = detector.detect_from_values(
            vix=vix if not np.isnan(vix) else None,
            vvix=vvix if not np.isnan(vvix) else None,
            vix3m=vix3m if not np.isnan(vix3m) else None,
            skew=skew if not np.isnan(skew) else None,
            pcr=pcr if not np.isnan(pcr) else None,
        )
        regimes.append(r["regime"])

    merged["regime"] = regimes

    # Regime distribution
    regime_counts = merged["regime"].value_counts()
    print("  Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count} days ({count/len(merged)*100:.1f}%)")

    # --- Figure: Regime timeline ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Color map for regimes
    regime_colors = {
        "calm": "#2ecc71",
        "normal": "#3498db",
        "stressed": "#e67e22",
        "crisis": "#e74c3c",
    }

    dates = merged["date"]

    # Panel 1: VIX with regime background
    ax = axes[0]
    ax.plot(dates, merged["VIX"], color="black", linewidth=0.5, alpha=0.8)
    for regime, color in regime_colors.items():
        mask = merged["regime"] == regime
        ax.fill_between(dates, 0, 100, where=mask, alpha=0.15, color=color,
                        label=regime.capitalize())
    ax.set_ylabel("VIX", fontsize=11)
    ax.set_title("Market Regime Detection (7-Signal Model)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=4)
    ax.set_ylim(5, 90)
    ax.grid(alpha=0.3)

    # Panel 2: VVIX
    if "VVIX" in merged.columns:
        ax = axes[1]
        ax.plot(dates, merged["VVIX"], color="#8e44ad", linewidth=0.5, alpha=0.8)
        ax.set_ylabel("VVIX", fontsize=11)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(140, color="red", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)

    # Panel 3: SKEW
    if "SKEW" in merged.columns:
        ax = axes[2]
        ax.plot(dates, merged["SKEW"], color="#c0392b", linewidth=0.5, alpha=0.8)
        ax.set_ylabel("SKEW", fontsize=11)
        ax.axhline(115, color="green", linestyle="--", alpha=0.3, label="Low threshold")
        ax.axhline(135, color="orange", linestyle="--", alpha=0.3, label="Medium")
        ax.axhline(150, color="red", linestyle="--", alpha=0.3, label="High")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 4: PCR
    if "PCR" in merged.columns:
        ax = axes[3]
        ax.plot(dates, merged["PCR"], color="#2980b9", linewidth=0.5, alpha=0.8)
        ax.set_ylabel("PCR SPX", fontsize=11)
        ax.axhline(0.7, color="green", linestyle="--", alpha=0.3)
        ax.axhline(1.0, color="orange", linestyle="--", alpha=0.3)
        ax.axhline(1.3, color="red", linestyle="--", alpha=0.3)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regime_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {OUT_DIR / 'regime_timeline.png'}")

    # Save regime data
    regime_out = {
        "n_dates": len(merged),
        "date_range": [str(merged["date"].min().date()), str(merged["date"].max().date())],
        "regime_distribution": regime_counts.to_dict(),
        "signals_available": list(signals.keys()),
    }
    out = OUT_DIR / "regime_timeline.json"
    with open(out, "w") as f:
        json.dump(regime_out, f, indent=2, default=str)
    print(f"  Saved -> {out.relative_to(ROOT)}")

    return regime_out


# =====================================================================
# Summary figure: backtest equity curves
# =====================================================================

def plot_backtest_equity_curves(results, output_dir):
    """Plot equity curves for all backtest models."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {
        "always_sell": "#a8a8a8",
        "historical": "#3498db",
        "bergomi": "#457b9d",
        "regime_bergomi": "#e63946",
        "neural": "#2ecc71",
        "regime_neural": "#f39c12",
    }

    for r in results:
        eq = r.equity_curve
        label = f"{r.model_name} (Sharpe={r.sharpe:.2f})"
        ax.plot(eq, color=colors.get(r.model_name, "black"),
                linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("Trade #", fontsize=12)
    ax.set_ylabel("Cumulative P&L (bps)", fontsize=12)
    ax.set_title("VRP Backtest Equity Curves (2010–2026)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "backtest_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_dir / 'backtest_equity_curves.png'}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper Results Master Script")
    parser.add_argument("--skip-exotic", action="store_true",
                        help="Skip exotic pricing comparison")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip VRP backtest")
    parser.add_argument("--skip-stress", action="store_true",
                        help="Skip stress test comparison")
    parser.add_argument("--skip-hedging", action="store_true",
                        help="Skip deep hedging backtest")
    parser.add_argument("--n-paths-exotic", type=int, default=10000)
    parser.add_argument("--n-paths-stress", type=int, default=5000)
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PAPER RESULTS — MASTER SCRIPT")
    print("  Neural SDE + Rough Bergomi: Full Experimental Suite")
    print("=" * 70)
    print(f"  Output directory: {OUT_DIR.relative_to(ROOT)}")
    print()

    all_results = {}

    # 1. Exotic pricing
    if not args.skip_exotic:
        try:
            exotic = run_exotic(n_paths=args.n_paths_exotic)
            all_results["exotic"] = "done"
        except Exception as e:
            print(f"  EXOTIC FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  [1/5] Exotic pricing: SKIPPED")

    # 2. VRP Backtest
    if not args.skip_backtest:
        try:
            bt_results = run_backtest()
            all_results["backtest"] = "done"
            plot_backtest_equity_curves(bt_results, OUT_DIR)
        except Exception as e:
            print(f"  BACKTEST FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  [2/5] VRP Backtest: SKIPPED")

    # 3. Stress test
    if not args.skip_stress:
        try:
            stress = run_stress(n_paths=args.n_paths_stress)
            all_results["stress"] = "done"
        except Exception as e:
            print(f"  STRESS FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  [3/5] Stress test: SKIPPED")

    # 4. Calibration summary
    try:
        cal = run_calibration_summary()
        all_results["calibration"] = "done"
    except Exception as e:
        print(f"  CALIBRATION FAILED: {e}")

    # 5. Regime timeline
    try:
        regime = run_regime_timeline()
        all_results["regime"] = "done"
    except Exception as e:
        print(f"  REGIME FAILED: {e}")
        import traceback; traceback.print_exc()

    # 6. Deep hedging backtest
    if not args.skip_hedging:
        try:
            print(f"\n{'='*70}")
            print("  [6/6] Deep Hedging Backtest")
            print(f"{'='*70}")
            from bin.experiments.deep_hedging_backtest import (
                generate_bergomi_paths, DeepHedger)
            from utils.config import load_config as _lc
            _cfg = _lc()
            _S0 = 5500.0; _T = 30/365; _n_steps = 63
            _r = _cfg.get("pricing", {}).get("risk_free_rate", 0.045)
            _xi0 = _cfg["bergomi"]["xi0"]; _iv = float(np.sqrt(_xi0))
            _rho = _cfg["bergomi"]["rho"]

            import jax as _jax
            _k1, _k2 = _jax.random.split(_jax.random.PRNGKey(42))
            _st_tr, _vt_tr = generate_bergomi_paths(_cfg, 8192, _T, _n_steps, _S0, key=_k1)
            _st_te, _vt_te = generate_bergomi_paths(_cfg, 2048, _T, _n_steps, _S0, key=_k2)

            _dh = DeepHedger(spot=_S0, strike=_S0, T=_T, r=_r, iv=_iv,
                             tc_bps=2.0, rho=_rho)
            _dh.train(_st_tr, _vt_tr, n_epochs=200, batch_size=2048, verbose=True)
            _res = _dh.evaluate(_st_te, _vt_te)

            # Save results
            _hedge_report = {
                strat: {k: v for k, v in _res[strat].items() if k != "pnl"}
                for strat in ["deep", "bs", "bartlett"]
            }
            _hedge_report["improvement"] = _res["summary"]
            with open(OUT_DIR / "deep_hedging_vanilla.json", "w") as f:
                json.dump(_hedge_report, f, indent=2)

            all_results["deep_hedging"] = "done"
            imp = _res["summary"]
            print(f"  CVaR₉₅ reduction vs BS: {imp['cvar95_reduction_vs_bs']:+.1f}%")
        except Exception as e:
            print(f"  DEEP HEDGING FAILED: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  [6/6] Deep hedging: SKIPPED")

    # Final summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for key, status in all_results.items():
        print(f"  {key:<20} {status}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Output: {OUT_DIR.relative_to(ROOT)}/")

    # List all generated files
    if OUT_DIR.exists():
        files = sorted(OUT_DIR.glob("*"))
        print(f"\n  Generated files ({len(files)}):")
        for f in files:
            size = f.stat().st_size
            if size > 1024:
                print(f"    {f.name:<40} {size/1024:.1f} KB")
            else:
                print(f"    {f.name:<40} {size} B")

    print()


if __name__ == "__main__":
    main()
