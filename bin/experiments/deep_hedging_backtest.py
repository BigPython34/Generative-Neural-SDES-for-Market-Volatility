"""
Deep Hedging Backtest
=====================
Compares deep hedging (learned policy) vs analytical deltas (BS, Bartlett)
under rough volatility dynamics (rBergomi).

Tests three derivative types:
  1. Vanilla call  — baseline (deep ≈ Bartlett expected)
  2. Asian call    — path-dependent (deep > BS expected)
  3. Lookback call — extreme path-dependence (deep >> BS expected)

Generates:
  • Training loss curves (per product)
  • P&L distribution histograms (deep vs BS vs Bartlett)
  • CVaR reduction bar chart
  • Summary JSON for paper_results/

References:
  Buehler, Gonon, Teichmann & Wood (2019). Deep Hedging. QF 19(8).
  Horvath, Teichmann & Žurič (2021). Deep Hedging under Rough Volatility.
  Carbonneau & Godin (2021). Equal Risk Pricing with Deep Hedging.

Usage:
    python bin/experiments/deep_hedging_backtest.py
    python bin/experiments/deep_hedging_backtest.py --epochs 200 --quick
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import argparse
from pathlib import Path

from utils.config import load_config
from quant.models.bergomi import RoughBergomiModel
from engine.deep_hedger import DeepHedger


# ═══════════════════════════════════════════════════════════════════
#  Path generation helpers
# ═══════════════════════════════════════════════════════════════════

def generate_bergomi_paths(cfg, n_paths, T, n_steps, S0, key=None):
    """
    Generate (spot, var) from rBergomi with spot-vol correlation.

    Returns
    -------
    spot : (n_paths, n_steps+1)  — absolute prices incl. S₀
    var  : (n_paths, n_steps+1)  — variance incl. V₀
    """
    params = {
        "hurst":  cfg["bergomi"]["hurst"],
        "eta":    cfg["bergomi"]["eta"],
        "rho":    cfg["bergomi"]["rho"],
        "xi0":    cfg["bergomi"]["xi0"],
        "n_steps": n_steps,
        "T":      T,
        "mu":     cfg.get("pricing", {}).get("risk_free_rate", 0.045),
    }
    model = RoughBergomiModel(params)
    st, vt = model.simulate_spot_vol_paths(n_paths, s0=S0, key=key)

    # Prepend V₀ = ξ₀ to make var shape (n_paths, n_steps+1)
    v0 = jnp.full((n_paths, 1), model.xi0)
    vt_full = jnp.concatenate([v0, vt], axis=1)

    return np.asarray(st), np.asarray(vt_full)


# ═══════════════════════════════════════════════════════════════════
#  Figure generation
# ═══════════════════════════════════════════════════════════════════

def plot_training_curves(all_histories, out_dir):
    """Training loss curves for each product."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, len(all_histories), figsize=(5 * len(all_histories), 4),
                             squeeze=False)
    for i, (name, hist) in enumerate(all_histories.items()):
        ax = axes[0, i]
        ax.plot(hist, linewidth=0.8, color="C0")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropic Risk")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Deep Hedger — Training Convergence", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "deep_hedge_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'deep_hedge_training.png'}")


def plot_pnl_distributions(all_results, out_dir):
    """P&L histograms for each product (3 strategies overlaid)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, (name, res) in enumerate(all_results.items()):
        ax = axes[0, i]
        bins = np.linspace(-5, 5, 80)
        ax.hist(res["bs"]["pnl"], bins=bins, alpha=0.4, label="BS", color="C1", density=True)
        ax.hist(res["bartlett"]["pnl"], bins=bins, alpha=0.4, label="Bartlett", color="C2", density=True)
        ax.hist(res["deep"]["pnl"], bins=bins, alpha=0.6, label="Deep", color="C0", density=True)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Hedging P&L")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Deep Hedging — P&L Distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "deep_hedge_pnl_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'deep_hedge_pnl_dist.png'}")


def plot_cvar_comparison(all_results, out_dir):
    """Bar chart: CVaR₉₅ across strategies and products."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    products = list(all_results.keys())
    strategies = ["bs", "bartlett", "deep"]
    labels = ["Black-Scholes", "Bartlett", "Deep Hedge"]
    colors = ["C1", "C2", "C0"]

    x = np.arange(len(products))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    for j, (strat, label, color) in enumerate(zip(strategies, labels, colors)):
        vals = [all_results[p][strat]["cvar_95"] for p in products]
        ax.bar(x + j * width, vals, width, label=label, color=color, alpha=0.8)

    ax.set_ylabel("CVaR₉₅ (lower = better hedge)")
    ax.set_title("Deep Hedging — Tail Risk Comparison", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(products, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "deep_hedge_cvar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'deep_hedge_cvar.png'}")


def plot_hedge_ratios(hedger, spot_paths, var_paths, out_dir, product_name="Vanilla Call"):
    """
    Compare hedge ratios on a single representative path:
    deep policy vs BS delta over time.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Pick one path (median terminal spot)
    S_T = spot_paths[:, -1]
    idx = int(np.argsort(S_T)[len(S_T) // 2])
    sp = spot_paths[idx]
    vp = var_paths[idx]

    n_steps = sp.shape[0] - 1
    dt = hedger.T / n_steps
    S0 = float(sp[0])

    deep_deltas, bs_deltas, times = [], [], []

    for step in range(n_steps):
        t = step * dt
        tau = hedger.T - t
        S = float(sp[step])
        V = float(vp[step])
        sigma = np.sqrt(max(V, 1e-8))

        # BS delta
        from engine.deep_hedger import _bs_delta_jax
        bd = float(_bs_delta_jax(
            jnp.float32(S), hedger.strike, jnp.float32(tau),
            hedger.r, jnp.float32(sigma)))
        if not hedger.is_call:
            bd -= 1.0
        bs_deltas.append(bd)

        # Deep delta
        dp = deep_deltas[-1] if deep_deltas else 0.0
        feats = jnp.array([
            S / S0,
            float(jnp.clip(jnp.log(max(V, 1e-10)), -7.0, 2.0)),
            float(jnp.clip(jnp.log(S / hedger.strike), -1.0, 1.0)),
            tau / hedger.T,
            dp,
            bd,
        ])
        dd = float(hedger.policy(feats))
        deep_deltas.append(dd)
        times.append(t * 365)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(times, sp[:-1], color="k", linewidth=0.8)
    ax1.set_ylabel("Spot")
    ax1.set_title(f"{product_name} — Hedge Ratio Comparison (1 path)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, bs_deltas, label="BS Δ", color="C1", linewidth=0.8, alpha=0.8)
    ax2.plot(times, deep_deltas, label="Deep δ", color="C0", linewidth=1.2)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Hedge Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "deep_hedge_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'deep_hedge_ratios.png'}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep Hedging Backtest")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--quick", action="store_true",
                        help="Fast run: fewer paths & epochs")
    parser.add_argument("--save-policy", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 100
        args.n_train = 8192
        args.n_test  = 2048

    cfg = load_config()
    out_dir = Path("outputs/paper_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Configuration ────────────────────────────────────────────
    S0      = 5500.0
    T       = 30 / 365           # 30-day maturity
    n_steps = 63                 # ~3 steps per trading day
    r       = cfg.get("pricing", {}).get("risk_free_rate", 0.045)
    xi0     = cfg["bergomi"]["xi0"]
    iv      = float(np.sqrt(xi0))   # ≈ 17.1%
    rho     = cfg["bergomi"]["rho"]
    tc_bps  = cfg.get("hedging", {}).get("transaction_cost_bps", 2.0)

    print("=" * 65)
    print("  DEEP HEDGING BACKTEST")
    print("  Buehler et al. (2019) + Horvath et al. (2021)")
    print("=" * 65)
    print(f"  Maturity: {T*365:.0f}d | Steps: {n_steps} | "
          f"IV: {iv:.1%} | TC: {tc_bps} bps")
    print(f"  Bergomi: H={cfg['bergomi']['hurst']:.3f}, "
          f"η={cfg['bergomi']['eta']:.3f}, ρ={rho:.3f}")
    print(f"  Train: {args.n_train} paths | Test: {args.n_test} paths | "
          f"Epochs: {args.epochs}")
    print()

    # ── Generate rBergomi paths ──────────────────────────────────
    print("Generating rBergomi spot-vol paths...")
    t0 = time.time()
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    st_train, vt_train = generate_bergomi_paths(
        cfg, args.n_train, T, n_steps, S0, key=k1)
    st_test, vt_test = generate_bergomi_paths(
        cfg, args.n_test, T, n_steps, S0, key=k2)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"(train {st_train.shape}, test {st_test.shape})")

    # ── Products to test ─────────────────────────────────────────
    products = [
        {
            "name": "Vanilla Call",
            "payoff_type": "vanilla",
            "payoff_kwargs": {},
        },
        {
            "name": "Asian Call",
            "payoff_type": "asian",
            "payoff_kwargs": {},
        },
        {
            "name": "Lookback Call",
            "payoff_type": "lookback",
            "payoff_kwargs": {},
        },
    ]

    all_results = {}
    all_histories = {}

    for product in products:
        pname = product["name"]
        print(f"\n{'─'*65}")
        print(f"  {pname}")
        print(f"{'─'*65}")

        hedger = DeepHedger(
            spot=S0, strike=S0, T=T, r=r, iv=iv,
            opt_type="call", tc_bps=tc_bps,
            risk_measure="entropic", risk_param=1.0,
            payoff_type=product["payoff_type"],
            payoff_kwargs=product["payoff_kwargs"],
            rho=rho,
        )
        print(f"  BS premium: {hedger.premium:.4f}")

        # Train
        t0 = time.time()
        history = hedger.train(
            st_train, vt_train,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
            verbose=True,
        )
        train_time = time.time() - t0
        print(f"  Training: {train_time:.1f}s")
        all_histories[pname] = history

        # Evaluate
        print("  Evaluating on test set...")
        result = hedger.evaluate(st_test, vt_test)
        all_results[pname] = result

        # Print comparison table
        print(f"\n  {'Strategy':<15} {'Mean P&L':>10} {'Std P&L':>10} "
              f"{'CVaR95':>10} {'Track Err':>10}")
        print(f"  {'-'*55}")
        for strat, label in [("bs", "BS Delta"), ("bartlett", "Bartlett"),
                              ("deep", "Deep Hedge")]:
            s = result[strat]
            print(f"  {label:<15} {s['mean_pnl']:>10.4f} {s['std_pnl']:>10.4f} "
                  f"{s['cvar_95']:>10.4f} {s['tracking_error']:>10.4f}")

        imp = result["summary"]
        print(f"\n  CVaR₉₅ reduction:  vs BS {imp['cvar95_reduction_vs_bs']:+.1f}%  "
              f"| vs Bartlett {imp['cvar95_reduction_vs_bart']:+.1f}%")
        print(f"  Std reduction:     vs BS {imp['std_reduction_vs_bs']:+.1f}%  "
              f"| vs Bartlett {imp['std_reduction_vs_bart']:+.1f}%")

        # Optionally save policy
        if args.save_policy:
            save_dir = f"models/deep_hedger/{product['payoff_type']}"
            hedger.save_policy(save_dir)
            print(f"  Policy saved → {save_dir}/")

    # ── Figures ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Generating figures...")

    plot_training_curves(all_histories, out_dir)
    plot_pnl_distributions(all_results, out_dir)
    plot_cvar_comparison(all_results, out_dir)

    # Hedge ratio plot for vanilla
    if "Vanilla Call" in all_results:
        hedger_vanilla = DeepHedger(
            spot=S0, strike=S0, T=T, r=r, iv=iv, rho=rho, tc_bps=tc_bps)
        hedger_vanilla.policy = all_results  # dummy — need the actual hedger
        # Reconstruct vanilla hedger for plot
        # (the hedger was overwritten in the loop, so we reload if saved)
        try:
            hedger_vanilla = DeepHedger.load("models/deep_hedger/vanilla")
            plot_hedge_ratios(hedger_vanilla, st_test, vt_test, out_dir)
        except Exception:
            pass  # skip if policy not saved

    # ── JSON report ──────────────────────────────────────────────
    report = {
        "experiment": "deep_hedging_backtest",
        "config": {
            "S0": S0, "T": T, "n_steps": n_steps, "r": r, "iv": iv,
            "rho": rho, "tc_bps": tc_bps,
            "n_train": args.n_train, "n_test": args.n_test,
            "n_epochs": args.epochs, "batch_size": args.batch_size,
            "bergomi": {k: cfg["bergomi"][k] for k in ["hurst", "eta", "rho", "xi0"]},
        },
        "products": {},
    }

    for pname, res in all_results.items():
        report["products"][pname] = {
            strat: {k: v for k, v in res[strat].items() if k != "pnl"}
            for strat in ["deep", "bs", "bartlett"]
        }
        report["products"][pname]["improvement"] = res["summary"]

    with open(out_dir / "deep_hedging_backtest.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results → {out_dir / 'deep_hedging_backtest.json'}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for pname, res in all_results.items():
        imp = res["summary"]
        print(f"  {pname:<20}  CVaR₉₅ vs BS: {imp['cvar95_reduction_vs_bs']:+.1f}%  "
              f"| vs Bartlett: {imp['cvar95_reduction_vs_bart']:+.1f}%")
    print()


if __name__ == "__main__":
    main()
