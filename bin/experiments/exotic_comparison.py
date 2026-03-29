"""
Exotic Pricing Comparison: BS vs Bergomi vs Neural SDE
======================================================
Prices path-dependent exotics under three calibrated models and produces
a publication-quality comparison table + figures.

Key insight: cliquets, autocallables, and variance swaps are *highly sensitive*
to the volatility dynamics (forward skew, vol-of-vol, path-dependence).
Rough vol and Neural SDE should demonstrably outperform Black-Scholes here.

Products tested:
  1. Asian call (arithmetic)   - moderate path-dependence
  2. Cliquet (ratchet)         - extreme forward-skew sensitivity
  3. Autocallable (Phoenix)    - barrier + coupon, regime-dependent
  4. Variance swap             - pure vol exposure
  5. Volatility swap           - convexity adjustment test
  6. Down-and-out call         - tail risk / barrier sensitivity
  7. Lookback call             - full path-dependence

References:
  - Bayer, Friz & Gatheral (2016): Pricing under rough volatility
  - Jacquier, Martini & Muguruza (2018): On VIX futures in the rough Bergomi model
  - Buehler et al. (2019): Deep Hedging

Usage:
  python bin/experiments/exotic_comparison.py
  python bin/experiments/exotic_comparison.py --n-paths 20000 --save-fig
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sitecustomize  # noqa: F401

import argparse
import json
import time
import numpy as np

# ---------------------------------------------------------------------------
# Path generation helpers
# ---------------------------------------------------------------------------

def _generate_bs_paths(spot, vol, r, T, n_steps, n_paths, seed=42):
    """GBM spot paths."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_ret = (r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_s = np.cumsum(log_ret, axis=1)
    s0_col = np.ones((n_paths, 1))
    paths = spot * np.hstack([s0_col, np.exp(log_s)])
    return paths


def _generate_bergomi_paths(spot, r, T, n_steps, n_paths):
    """Rough Bergomi spot + vol paths."""
    from quant.models.bergomi import RoughBergomiModel
    from utils.config import load_config
    cfg = load_config()
    bergomi_cfg = cfg["bergomi"]
    params = {
        "hurst": bergomi_cfg["hurst"],
        "eta": bergomi_cfg["eta"],
        "rho": bergomi_cfg["rho"],
        "xi0": bergomi_cfg.get("xi0", 0.04),
        "n_steps": n_steps,
        "T": T,
        "mu": r,
    }
    model = RoughBergomiModel(params)
    st, vt = model.simulate_spot_vol_paths(n_paths, s0=spot)
    return np.array(st), np.array(vt)


def _generate_neural_sde_paths(spot, r, T, n_steps, n_paths, use_q=True):
    """
    Neural SDE paths.  Tries Q-model first (correct for pricing),
    falls back to P-model with leverage.
    """
    import jax
    import jax.numpy as jnp
    from utils.config import load_config
    cfg = load_config()
    rho = cfg["bergomi"]["rho"]

    # --- Try Q-model (Girsanov) ---
    if use_q:
        try:
            from quant.calibration.neural_q import load_q_model
            q_model = load_q_model()
            if q_model is not None:
                dt = T / n_steps
                v0 = cfg["bergomi"].get("xi0", 0.04)
                init_log_v = float(np.log(max(v0, 1e-6)))
                key = jax.random.PRNGKey(42)
                log_v_paths, spot_rel, _ = q_model.simulate(
                    init_log_v, n_steps, dt, key, n_paths
                )
                spot_paths = spot * np.array(spot_rel)
                s0_col = np.full((n_paths, 1), spot)
                spot_full = np.hstack([s0_col, spot_paths])
                var_paths = np.exp(np.array(log_v_paths))
                return spot_full, var_paths, "Q"
        except Exception:
            pass

    # --- Fallback: P-model with leverage effect ---
    try:
        from engine.generative_trainer import GenerativeTrainer
        trainer = GenerativeTrainer(cfg)
        trainer.load_data()
        model = trainer.load_model()
        if model is None:
            return None, None, None

        from quant.pricers.pricing import DeepPricingEngine
        engine = DeepPricingEngine(trainer, model)
        spot_paths, var_paths = engine.generate_market_paths(n_paths, s0=spot, seed=42)
        return np.array(spot_paths), np.array(var_paths), "P"
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, None


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_exotic_comparison(n_paths: int = 10000, save_fig: bool = True):
    from quant.pricers.exotic_pricer import ExoticPricer
    from utils.config import load_config

    cfg = load_config()
    spot = cfg["pricing"]["spot"]
    r_cfg = cfg["pricing"]["risk_free_rate"]

    # Use SOFR if available
    r = r_cfg
    try:
        from quant.loader.sofr_loader import get_sofr
        sofr = get_sofr()
        if sofr.is_available:
            r = sofr.get_rate()
    except Exception:
        pass

    T = 0.25   # 3 months
    n_steps = 63  # daily steps for 3 months

    print("=" * 70)
    print("  EXOTIC PRICING COMPARISON: BS vs BERGOMI vs NEURAL SDE")
    print("=" * 70)
    print(f"  Spot={spot:.0f}  r={r:.4f}  T={T:.2f}y  n_paths={n_paths}")
    print()

    # Estimate BS vol from VIX — TradingView first (36yr), then Yahoo
    vix_vol = 0.20
    try:
        import pandas as pd
        for vix_path in [
            ROOT / "data" / "market" / "volatility" / "vix_daily.csv",
            ROOT / "data" / "market" / "vix" / "vix_daily.csv",
        ]:
            if vix_path.exists():
                vix_df = pd.read_csv(vix_path)
                if "time" in vix_df.columns:
                    vix_df["date"] = pd.to_datetime(vix_df["time"], unit="s")
                    vix_df = vix_df.sort_values("date")
                if "close" in vix_df.columns:
                    vix_vol = float(vix_df["close"].dropna().iloc[-1]) / 100.0
                    break
    except Exception:
        pass
    print(f"  BS implied vol: {vix_vol*100:.1f}%")

    # Generate paths
    print("\n  Generating paths ...")
    t0 = time.time()

    bs_paths = _generate_bs_paths(spot, vix_vol, r, T, n_steps, n_paths)
    print(f"    BS:       {bs_paths.shape}  ({time.time()-t0:.1f}s)")

    t1 = time.time()
    berg_spot, berg_var = _generate_bergomi_paths(spot, r, T, n_steps, n_paths)
    print(f"    Bergomi:  {berg_spot.shape}  ({time.time()-t1:.1f}s)")

    t2 = time.time()
    nn_spot, nn_var, nn_measure = _generate_neural_sde_paths(spot, r, T, n_steps, n_paths)
    if nn_spot is not None:
        print(f"    NeuralSDE({nn_measure}): {nn_spot.shape}  ({time.time()-t2:.1f}s)")
    else:
        print(f"    NeuralSDE: FAILED - skipping")

    # Price all exotics
    pricer = ExoticPricer(spot=spot, r=r, T=T)
    strike = spot  # ATM

    products = {
        "Asian Call (arith)": lambda p: pricer.asian_call(p, strike, "arithmetic"),
        "Asian Call (geom)": lambda p: pricer.asian_call(p, strike, "geometric"),
        "Cliquet": lambda p: pricer.cliquet(p, local_cap=0.05, local_floor=-0.03, global_floor=0.0),
        "Autocallable": lambda p: pricer.autocallable(p, coupon_rate=0.08, autocall_barrier=1.0, ki_barrier=0.6),
        "Var Swap (K=0.04)": lambda p: pricer.variance_swap(p, strike_var=vix_vol**2),
        "Vol Swap (K=20%)": lambda p: pricer.volatility_swap(p, strike_vol=vix_vol),
        "D&O Call (B=90%)": lambda p: pricer.down_and_out_call(p, strike, strike * 0.90),
        "Lookback Call": lambda p: pricer.lookback_call(p, strike),
    }

    # Collect results
    results = {}
    models = {"BS": bs_paths, "Bergomi": berg_spot}
    if nn_spot is not None:
        models[f"NeuralSDE({nn_measure})"] = nn_spot

    print(f"\n  Pricing {len(products)} products x {len(models)} models ...")

    for prod_name, price_fn in products.items():
        results[prod_name] = {}
        for model_name, paths in models.items():
            try:
                res = price_fn(paths)
                # Extract the main price field
                if "price" in res:
                    price = res["price"]
                    se = res.get("std_error", 0)
                elif "price_pct" in res:
                    price = res["price_pct"]
                    se = res.get("std_error_pct", 0)
                else:
                    price = 0
                    se = 0
                results[prod_name][model_name] = {
                    "price": float(price),
                    "std_error": float(se),
                    "details": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                for k, v in res.items()
                                if k not in ("price", "std_error", "price_pct", "std_error_pct")},
                }
            except Exception as e:
                results[prod_name][model_name] = {"price": float("nan"), "std_error": 0,
                                                    "details": {"error": str(e)}}

    # Print table
    print("\n" + "=" * 90)
    print("  EXOTIC PRICING RESULTS")
    print("=" * 90)

    model_names = list(models.keys())
    header = f"{'Product':<22}"
    for m in model_names:
        header += f" {m:>18}"
    print(header)
    print("-" * 90)

    for prod_name, model_results in results.items():
        line = f"{prod_name:<22}"
        for m in model_names:
            r_m = model_results.get(m, {"price": float("nan")})
            p = r_m["price"]
            se = r_m.get("std_error", 0)
            if abs(p) < 100:
                line += f"  {p:>9.4f} +/-{se:.4f}"
            else:
                line += f"  {p:>9.2f} +/-{se:.2f}"
        print(line)

    print("-" * 90)

    # Relative differences (Bergomi vs BS, Neural vs BS)
    if len(model_names) >= 2:
        print("\n  Relative to BS (%):")
        for prod_name, model_results in results.items():
            bs_p = model_results.get("BS", {}).get("price", float("nan"))
            if np.isnan(bs_p) or abs(bs_p) < 1e-10:
                continue
            line = f"  {prod_name:<22}"
            for m in model_names[1:]:
                m_p = model_results.get(m, {}).get("price", float("nan"))
                if not np.isnan(m_p):
                    diff_pct = (m_p - bs_p) / abs(bs_p) * 100
                    line += f"  {m}: {diff_pct:>+7.2f}%"
            print(line)

    # Save results
    out_dir = ROOT / "outputs" / "paper_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exotic_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {"spot": spot, "r": r, "T": T, "n_paths": n_paths,
                        "bs_vol": vix_vol, "n_steps": n_steps},
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path.relative_to(ROOT)}")

    # Figures
    if save_fig:
        _plot_exotic_comparison(results, model_names, out_dir)
        _plot_path_samples(models, T, n_steps, out_dir)

    return results


def _plot_exotic_comparison(results, model_names, out_dir):
    """Bar chart comparing exotic prices across models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping figures")
        return

    products = list(results.keys())
    n_prod = len(products)
    n_mod = len(model_names)

    # Filter products with comparable price scales
    vanilla_prods = [p for p in products if "Autocallable" not in p]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(vanilla_prods))
    width = 0.8 / n_mod
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12"]

    for i, model in enumerate(model_names):
        prices = []
        for p in vanilla_prods:
            val = results[p].get(model, {}).get("price", 0)
            prices.append(float(val) if not np.isnan(val) else 0)
        ax.bar(x + i * width, prices, width, label=model, color=colors[i % len(colors)],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * (n_mod - 1) / 2)
    ax.set_xticklabels(vanilla_prods, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title("Exotic Option Pricing: BS vs Bergomi vs Neural SDE", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig_path = out_dir / "exotic_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {fig_path.relative_to(ROOT)}")

    # Relative difference chart
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(model_names[1:], 1):
        diffs = []
        for p in vanilla_prods:
            bs_p = results[p].get("BS", {}).get("price", float("nan"))
            m_p = results[p].get(model, {}).get("price", float("nan"))
            if np.isnan(bs_p) or np.isnan(m_p) or abs(bs_p) < 1e-10:
                diffs.append(0)
            else:
                diffs.append((m_p - bs_p) / abs(bs_p) * 100)
        ax2.bar(x + (i - 1) * width * 1.5, diffs, width * 1.2, label=f"{model} vs BS",
                color=colors[i % len(colors)], alpha=0.85, edgecolor="white")

    ax2.set_xticks(x + width * 0.5)
    ax2.set_xticklabels(vanilla_prods, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Relative difference vs BS (%)", fontsize=11)
    ax2.set_title("Model Impact on Exotic Pricing (% difference from BS)", fontsize=13, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig2_path = out_dir / "exotic_relative_diff.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Figure saved -> {fig2_path.relative_to(ROOT)}")


def _plot_path_samples(models_dict, T, n_steps, out_dir):
    """Plot sample paths for each model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_sample = 50
    t_grid = np.linspace(0, T, n_steps + 1)

    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4),
                              sharey=True)
    if len(models_dict) == 1:
        axes = [axes]

    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12"]
    for ax, (name, paths), color in zip(axes, models_dict.items(), colors):
        n_show = min(n_sample, paths.shape[0])
        for i in range(n_show):
            ax.plot(t_grid[:paths.shape[1]], paths[i, :] / paths[i, 0],
                    alpha=0.15, color=color, linewidth=0.5)
        # Mean path
        mean_path = np.mean(paths / paths[:, 0:1], axis=0)
        ax.plot(t_grid[:paths.shape[1]], mean_path, color="black", linewidth=2, label="mean")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("S(t) / S(0)")
    fig.suptitle("Sample Spot Paths by Model", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = out_dir / "path_samples.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {fig_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=10000)
    parser.add_argument("--save-fig", action="store_true", default=True)
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()
    run_exotic_comparison(n_paths=args.n_paths, save_fig=not args.no_fig)
