"""
Stress Test Comparison: Neural SDE vs Bergomi vs Deterministic
==============================================================
Demonstrates that Neural SDE generates richer tail dynamics
than parametric (Bergomi) or deterministic (BS shocks) stress tests.

Key insights to show:
  1. Neural SDE produces fatter tails (higher kurtosis) than Bergomi
  2. Neural SDE captures vol clustering and mean-reversion
  3. Distributional stress (Neural SDE) vs point-estimate stress (det.)
  4. Joint spot-vol dynamics differ significantly across models

Portfolio: Short straddle (ATM) — sensitive to tails, vol jumps, and
           spot-vol correlation. This is the canonical VRP trade.

Usage:
  python bin/experiments/stress_test_comparison.py
  python bin/experiments/stress_test_comparison.py --n-paths 10000
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import load_config
from quant.risk.risk_engine import RiskEngine


# =====================================================================
# Scenario generators
# =====================================================================

def _get_current_vix() -> float:
    """Get latest VIX from TradingView."""
    import pandas as pd
    for p in [
        ROOT / "data" / "market" / "volatility" / "vix_daily.csv",
        ROOT / "data" / "market" / "vix" / "vix_daily.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p)
            if "time" in df.columns:
                df["date"] = pd.to_datetime(df["time"], unit="s")
                df = df.sort_values("date")
            if "close" in df.columns:
                return float(df["close"].dropna().iloc[-1])
    return 18.0


def generate_neural_sde_stress(engine: RiskEngine, n_paths: int) -> dict:
    """Run Neural SDE-driven stress test (model generates crisis paths)."""
    from engine.generative_trainer import GenerativeTrainer
    yaml_config = load_config()
    sim_cfg = yaml_config["simulation"]
    trainer_config = {"n_steps": sim_cfg["n_steps"], "T": sim_cfg["T"]}
    trainer = GenerativeTrainer(trainer_config)
    # Data already loaded in __init__
    model = trainer.load_model()
    if model is None:
        raise RuntimeError("No trained P-model found")

    # neural_stress_test expects config with 'n_steps' and 'T' flat keys
    return engine.neural_stress_test(model, trainer_config, n_paths=n_paths, seed=42)


def generate_bergomi_stress(engine: RiskEngine, n_paths: int) -> dict:
    """Run Bergomi-based stress scenarios (parametric rough vol)."""
    from quant.models.bergomi import RoughBergomiModel

    config = load_config()
    vix = _get_current_vix() / 100.0

    scenarios = {
        "Panic (VIX~45%)": {"xi0": 0.2025},
        "Elevated (VIX~30%)": {"xi0": 0.09},
        "Low-vol regime shift": {"xi0": 0.01},
        "Current regime": {"xi0": vix ** 2},
    }

    results = {}
    for name, params in scenarios.items():
        berg_cfg = dict(config["bergomi"])
        berg_cfg["xi0"] = params["xi0"]
        berg_cfg["n_steps"] = config["simulation"]["n_steps"]
        berg_cfg["T"] = config["simulation"]["T"]
        berg_cfg["mu"] = config["pricing"]["risk_free_rate"]
        berg_model = RoughBergomiModel(berg_cfg)

        st, vt = berg_model.simulate_spot_vol_paths(n_paths, s0=engine.spot)
        st = np.array(st)
        vt = np.array(vt)

        s_terminal = st[:, -1]
        vol_terminal = np.sqrt(np.clip(vt[:, -1], 1e-8, None))

        pnl = engine._value_portfolio(s_terminal, vol_terminal)

        var_95 = float(-np.percentile(pnl, 5))
        var_99 = float(-np.percentile(pnl, 1))
        cvar_95 = float(-np.mean(pnl[pnl <= -var_95])) if (pnl <= -var_95).any() else var_95
        cvar_99 = float(-np.mean(pnl[pnl <= -var_99])) if (pnl <= -var_99).any() else var_99

        results[name] = {
            "v0_as_vix": float(np.sqrt(params["xi0"]) * 100),
            "n_paths": n_paths,
            "expected_pnl": float(np.mean(pnl)),
            "pnl_std": float(np.std(pnl)),
            "VaR_95": var_95,
            "VaR_99": var_99,
            "CVaR_95": cvar_95,
            "CVaR_99": cvar_99,
            "max_loss": float(-np.min(pnl)),
            "terminal_spot_mean": float(np.mean(s_terminal)),
            "terminal_spot_p5": float(np.percentile(s_terminal, 5)),
            "terminal_vol_mean": float(np.mean(vol_terminal)),
            "terminal_vol_p95": float(np.percentile(vol_terminal, 95)),
            "prob_spot_drop_gt_10pct": float(np.mean(s_terminal < engine.spot * 0.9)),
            "prob_vol_gt_40pct": float(np.mean(vol_terminal > 0.40)),
            "pnl_skew": float(_skewness(pnl)),
            "pnl_kurtosis": float(_kurtosis(pnl)),
            "_pnl_array": pnl,
        }

    return results


def generate_deterministic_stress(engine: RiskEngine) -> dict:
    """Classic deterministic stress test (BS shocks, no paths)."""
    return engine.stress_test()


# =====================================================================
# Statistics helpers
# =====================================================================

def _skewness(x):
    m = np.mean(x); s = np.std(x)
    return float(np.mean(((x - m) / s) ** 3)) if s > 1e-12 else 0.0

def _kurtosis(x):
    m = np.mean(x); s = np.std(x)
    return float(np.mean(((x - m) / s) ** 4) - 3) if s > 1e-12 else 0.0


# =====================================================================
# Figures
# =====================================================================

def make_figures(nsde_results: dict, berg_results: dict, det_results: dict,
                 output_dir: Path, engine: RiskEngine):
    """Generate publication-quality stress test comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colors
    c_nsde = "#e63946"
    c_berg = "#457b9d"
    c_det = "#2a9d8f"

    scenarios = [s for s in nsde_results if s in berg_results]

    # --- Figure 1: VaR/CVaR comparison across scenarios ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "VaR_99", "Value-at-Risk (99%)"),
        (axes[1], "CVaR_99", "Conditional VaR (99%)"),
    ]:
        x = np.arange(len(scenarios))
        nsde_vals = [nsde_results[s].get(metric, 0) for s in scenarios]
        berg_vals = [berg_results[s].get(metric, 0) for s in scenarios]

        w = 0.35
        ax.bar(x - w/2, nsde_vals, w, label="Neural SDE", color=c_nsde, alpha=0.85)
        ax.bar(x + w/2, berg_vals, w, label="Rough Bergomi", color=c_berg, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([s.split(" (")[0] for s in scenarios],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Loss ($)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Stress Test: Neural SDE vs Rough Bergomi", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "stress_var_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_dir / 'stress_var_comparison.png'}")

    # --- Figure 2: P&L distribution under Panic scenario ---
    fig, ax = plt.subplots(figsize=(10, 5))
    panic_key = [s for s in scenarios if "Panic" in s]
    if panic_key:
        pk = panic_key[0]
        nsde_pnl = nsde_results[pk].get("_pnl_array")
        berg_pnl = berg_results[pk].get("_pnl_array")

        if nsde_pnl is not None:
            ax.hist(nsde_pnl, bins=100, density=True, alpha=0.5, color=c_nsde,
                    label=f"Neural SDE (kurt={_kurtosis(nsde_pnl):.1f})")
        if berg_pnl is not None:
            ax.hist(berg_pnl, bins=100, density=True, alpha=0.5, color=c_berg,
                    label=f"Rough Bergomi (kurt={_kurtosis(berg_pnl):.1f})")

        # Mark VaR
        if nsde_pnl is not None:
            ax.axvline(-nsde_results[pk]["VaR_99"], color=c_nsde, linestyle="--",
                       label=f"Neural VaR99=${nsde_results[pk]['VaR_99']:.2f}")
        if berg_pnl is not None:
            ax.axvline(-berg_results[pk]["VaR_99"], color=c_berg, linestyle="--",
                       label=f"Bergomi VaR99=${berg_results[pk]['VaR_99']:.2f}")

        ax.set_xlabel("P&L ($)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"P&L Distribution under {pk}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "stress_pnl_dist_panic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_dir / 'stress_pnl_dist_panic.png'}")

    # --- Figure 3: Tail metrics (skew, kurtosis) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "pnl_skew", "P&L Skewness"),
        (axes[1], "pnl_kurtosis", "P&L Excess Kurtosis"),
    ]:
        x = np.arange(len(scenarios))
        nsde_vals = [nsde_results[s].get(metric, 0) for s in scenarios]
        berg_vals = [berg_results[s].get(metric, 0) for s in scenarios]

        w = 0.35
        ax.bar(x - w/2, nsde_vals, w, label="Neural SDE", color=c_nsde, alpha=0.85)
        ax.bar(x + w/2, berg_vals, w, label="Rough Bergomi", color=c_berg, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([s.split(" (")[0] for s in scenarios],
                           rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Tail Risk: Neural SDE vs Rough Bergomi", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "stress_tail_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_dir / 'stress_tail_metrics.png'}")

    # --- Figure 4: Deterministic vs Distributional comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    det_names = list(det_results.keys())
    det_pnls = [det_results[n]["pnl"] for n in det_names]

    ax.barh(range(len(det_names)), det_pnls, height=0.6, color=c_det, alpha=0.85,
            label="Deterministic (single point)")

    # Overlay Neural SDE scenario ranges as error bars
    for i, name in enumerate(det_names):
        match = [s for s in nsde_results
                 if any(w in s.lower() for w in name.lower().split()[:2])]
        if match:
            nr = nsde_results[match[0]]
            ax.errorbar(nr["expected_pnl"], i, xerr=nr["pnl_std"],
                        fmt="o", color=c_nsde, capsize=5, markersize=8,
                        label="Neural SDE (mean±σ)" if i == 0 else "")

    ax.set_yticks(range(len(det_names)))
    ax.set_yticklabels(det_names, fontsize=9)
    ax.set_xlabel("P&L ($)", fontsize=12)
    ax.set_title("Deterministic vs Distributional Stress Test", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stress_det_vs_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_dir / 'stress_det_vs_dist.png'}")

    # --- Figure 5: Prob of extreme events ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "prob_spot_drop_gt_10pct", "P(Spot drop > 10%)"),
        (axes[1], "prob_vol_gt_40pct", "P(Vol > 40%)"),
    ]:
        x = np.arange(len(scenarios))
        nsde_vals = [nsde_results[s].get(metric, 0) * 100 for s in scenarios]
        berg_vals = [berg_results[s].get(metric, 0) * 100 for s in scenarios]

        w = 0.35
        ax.bar(x - w/2, nsde_vals, w, label="Neural SDE", color=c_nsde, alpha=0.85)
        ax.bar(x + w/2, berg_vals, w, label="Rough Bergomi", color=c_berg, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([s.split(" (")[0] for s in scenarios],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Probability (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Extreme Event Probabilities", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "stress_extreme_probs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_dir / 'stress_extreme_probs.png'}")


# =====================================================================
# Report
# =====================================================================

def print_report(nsde_results: dict, berg_results: dict, det_results: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("  STRESS TEST COMPARISON")
    print("=" * 100)

    scenarios = [s for s in nsde_results if s in berg_results]

    header = f"{'Scenario':<25} {'Metric':<14} {'Neural SDE':>12} {'R.Bergomi':>12} {'Diff':>8}"
    print(header)
    print("-" * 100)

    for s in scenarios:
        nr = nsde_results[s]
        br = berg_results[s]
        first = True
        for metric in ["VaR_99", "CVaR_99", "max_loss", "pnl_skew", "pnl_kurtosis",
                        "prob_spot_drop_gt_10pct"]:
            nv = nr.get(metric, 0)
            bv = br.get(metric, 0)
            label = s if first else ""
            first = False
            if abs(bv) > 1e-8:
                diff = (nv - bv) / abs(bv) * 100
                print(f"{label:<25} {metric:<14} {nv:>12.3f} {bv:>12.3f} {diff:>+7.1f}%")
            else:
                print(f"{label:<25} {metric:<14} {nv:>12.3f} {bv:>12.3f} {'N/A':>8}")
        print()

    print("-" * 100)

    # Deterministic
    print("\n  DETERMINISTIC STRESS SCENARIOS (for reference)")
    print(f"  {'Scenario':<25} {'P&L':>10} {'Spot Shock':>12} {'Vol Shock':>12}")
    print("  " + "-" * 60)
    for name, res in det_results.items():
        print(f"  {name:<25} {res['pnl']:>10.2f} {res['spot_shocked']:>12.1f} "
              f"{res.get('vol_shock', 0):>+11.0%}")
    print()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stress Test Comparison")
    parser.add_argument("--n-paths", type=int, default=5000,
                        help="MC paths per scenario (default 5000)")
    args = parser.parse_args()

    t0 = time.time()

    config = load_config()
    spot = config["pricing"]["spot"]
    r = config["pricing"]["risk_free_rate"]
    vix = _get_current_vix() / 100.0

    print("=" * 70)
    print("  STRESS TEST COMPARISON")
    print("  Neural SDE vs Rough Bergomi vs Deterministic")
    print("=" * 70)
    print(f"  Spot={spot:.0f}, r={r:.4f}, VIX={vix*100:.1f}%, n_paths={args.n_paths:,}")

    # Build test portfolio: short ATM straddle (canonical VRP trade)
    engine = RiskEngine(spot=spot, r=r)
    engine.add_position("call", strike=spot, T=0.25, quantity=-10, iv=vix)
    engine.add_position("put", strike=spot, T=0.25, quantity=-10, iv=vix)
    # Delta hedge
    delta_call = 0.5  # approximate ATM
    delta_put = -0.5
    net_delta = -10 * delta_call + (-10 * delta_put)  # = 0 for ATM
    if abs(net_delta) > 0.1:
        engine.add_position("stock", quantity=-net_delta)

    print(f"  Portfolio: Short 10x ATM straddle (K={spot:.0f}, T=3m)")
    print()

    # 1. Neural SDE stress test
    print("  [1/3] Neural SDE stress scenarios ...")
    try:
        nsde_results = generate_neural_sde_stress(engine, args.n_paths)
        # Add _pnl_array and skew/kurt
        for name, res in nsde_results.items():
            if "pnl_skew" not in res:
                res["pnl_skew"] = 0.0
                res["pnl_kurtosis"] = 0.0
        print(f"        {len(nsde_results)} scenarios generated")
    except Exception as e:
        print(f"        FAILED: {e}")
        import traceback; traceback.print_exc()
        nsde_results = {}

    # 2. Bergomi stress test
    print("  [2/3] Rough Bergomi stress scenarios ...")
    try:
        berg_results = generate_bergomi_stress(engine, args.n_paths)
        print(f"        {len(berg_results)} scenarios generated")
    except Exception as e:
        print(f"        FAILED: {e}")
        import traceback; traceback.print_exc()
        berg_results = {}

    # 3. Deterministic stress test
    print("  [3/3] Deterministic stress scenarios ...")
    det_results = generate_deterministic_stress(engine)
    print(f"        {len(det_results)} scenarios")

    # Report
    print_report(nsde_results, berg_results, det_results)

    # Figures
    output_dir = ROOT / "outputs" / "paper_results"
    if nsde_results and berg_results:
        make_figures(nsde_results, berg_results, det_results, output_dir, engine)

    # Save JSON (exclude _pnl_array for serialization)
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "portfolio": "Short 10x ATM straddle",
        "spot": spot, "vix": vix * 100,
        "neural_sde": {k: {kk: vv for kk, vv in v.items() if kk != "_pnl_array"}
                       for k, v in nsde_results.items()},
        "bergomi": {k: {kk: vv for kk, vv in v.items() if kk != "_pnl_array"}
                    for k, v in berg_results.items()},
        "deterministic": det_results,
    }
    out_path = output_dir / "stress_test_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path.relative_to(ROOT)}")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
