"""
Hedging Animation
=================
Generates animated visualisations of delta-hedging strategies
(Deep Hedge vs BS vs Bartlett) under rough volatility.

Four-panel dark-themed animation:
  ┌───────────────┬───────────────┐
  │  Spot Path    │  Variance     │
  ├───────────────┼───────────────┤
  │  Hedge Ratios │  Cumul. P&L   │
  └───────────────┴───────────────┘

Supports:
  • Model: rBergomi (default), GBM, or pre-loaded Neural SDE paths
  • Product: vanilla, asian, lookback
  • Output: MP4 (ffmpeg) or GIF (pillow)
  • Multiple paths overlaid with one highlighted "hero" path
  • Real-time P&L counter and strategy labels

Usage:
    python bin/experiments/hedging_animation.py
    python bin/experiments/hedging_animation.py --product asian --fps 30 --gif
    python bin/experiments/hedging_animation.py --model gbm --paths 50

Requirements:
    pip install matplotlib   (already installed)
    ffmpeg in PATH for MP4   (or use --gif for Pillow fallback)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import sitecustomize  # noqa: F401

import argparse
import time
from pathlib import Path
import numpy as np
from scipy.stats import norm
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.config import load_config
from quant.models.black_scholes import BlackScholes
from quant.models.bergomi import RoughBergomiModel
from engine.deep_hedger import (
    DeepHedger, _bs_delta_jax, _make_bs_delta,
    _make_bartlett_delta, _make_deep_delta,
)

# ═══════════════════════════════════════════════════════════════════
#  Colour palette (dark theme)
# ═══════════════════════════════════════════════════════════════════
BG         = "#0d1117"
PANEL_BG   = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR  = "#c9d1d9"
ACCENT     = "#58a6ff"      # blue — deep hedge
BS_COLOR   = "#f0883e"      # orange — BS
BART_COLOR = "#a371f7"      # purple — Bartlett
SPOT_COLOR = "#56d364"      # green — spot
VAR_COLOR  = "#ff7b72"      # red — variance
GHOST      = 0.08           # alpha for background paths
STRIKE_COL = "#8b949e"


def _apply_dark_theme(fig, axes):
    """Apply dark theme to all axes."""
    fig.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)


# ═══════════════════════════════════════════════════════════════════
#  Path generation
# ═══════════════════════════════════════════════════════════════════

def generate_paths(model_name, cfg, n_paths, T, n_steps, S0, key):
    """Generate (spot, var) paths.  spot: (n, n_steps+1), var: (n, n_steps+1)."""
    if model_name == "bergomi":
        params = {
            "hurst":  cfg["bergomi"]["hurst"],
            "eta":    cfg["bergomi"]["eta"],
            "rho":    cfg["bergomi"]["rho"],
            "xi0":    cfg["bergomi"]["xi0"],
            "n_steps": n_steps,
            "T":      T,
            "mu":     cfg.get("pricing", {}).get("risk_free_rate", 0.045),
        }
        m = RoughBergomiModel(params)
        st, vt = m.simulate_spot_vol_paths(n_paths, s0=S0, key=key)
        v0 = jnp.full((n_paths, 1), m.xi0)
        vt = jnp.concatenate([v0, vt], axis=1)
        return np.asarray(st), np.asarray(vt)
    else:  # GBM
        r = cfg.get("pricing", {}).get("risk_free_rate", 0.045)
        xi0 = cfg["bergomi"]["xi0"]
        sigma = float(np.sqrt(xi0))
        dt = T / n_steps
        Z = np.array(jax.random.normal(key, (n_paths, n_steps)))
        log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        spot = S0 * np.exp(np.cumsum(log_ret, axis=1))
        spot = np.concatenate([np.full((n_paths, 1), S0), spot], axis=1)
        var = np.full_like(spot, sigma**2)
        return spot, var


# ═══════════════════════════════════════════════════════════════════
#  Step-by-step hedging (NumPy — for animation)
# ═══════════════════════════════════════════════════════════════════

def compute_step_by_step(spot, var, strike, T, r, tc_bps, iv, rho,
                         deep_policy=None, is_call=True, payoff_type="vanilla"):
    """
    Compute per-step hedge ratios and cumulative P&L for 3 strategies.

    Returns dict of arrays, each (n_steps+1,).
    """
    n_steps = spot.shape[0] - 1
    dt = T / n_steps
    S0 = float(spot[0])
    tc = tc_bps / 10_000.0
    premium = float(BlackScholes.price(S0, strike, T, r, iv, "call" if is_call else "put"))

    strategies = {}
    for name in ["bs", "bartlett", "deep"]:
        strategies[name] = {
            "delta": np.zeros(n_steps + 1),
            "cash":  np.zeros(n_steps + 1),
            "pnl":   np.zeros(n_steps + 1),
        }
        strategies[name]["cash"][0] = premium

    for step in range(n_steps):
        S = float(spot[step])
        V = float(np.clip(var[step], 1e-8, None))
        tau = max(T - step * dt, 1e-8)
        sigma = float(np.sqrt(V))

        # BS delta
        d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        bs_d = float(norm.cdf(d1))
        if not is_call:
            bs_d -= 1.0

        # Bartlett delta
        d2 = d1 - sigma * np.sqrt(tau)
        vanna = -float(norm.pdf(d1)) * d2 / sigma
        bart_d = np.clip(bs_d + vanna * rho * sigma / max(S, 1e-4), -2.0, 2.0)

        # Deep delta
        if deep_policy is not None:
            dp = strategies["deep"]["delta"][step]
            feats = jnp.array([
                S / S0,
                float(jnp.clip(jnp.log(max(V, 1e-10)), -7.0, 2.0)),
                float(jnp.clip(jnp.log(S / strike), -1.0, 1.0)),
                tau / T,
                dp,
                bs_d,
            ])
            deep_d = float(deep_policy(feats))
        else:
            deep_d = bs_d  # fallback

        deltas = {"bs": bs_d, "bartlett": bart_d, "deep": deep_d}

        for name, new_d in deltas.items():
            s = strategies[name]
            old_d = s["delta"][step]
            trade = new_d - old_d
            c = s["cash"][step]
            c -= trade * S
            c -= abs(trade) * S * tc
            c *= np.exp(r * dt)
            s["delta"][step + 1] = new_d
            s["cash"][step + 1] = c

    # Terminal P&L
    S_T = float(spot[-1])
    if payoff_type == "vanilla":
        payoff = max(S_T - strike, 0) if is_call else max(strike - S_T, 0)
    elif payoff_type == "asian":
        avg = float(np.mean(spot[1:]))
        payoff = max(avg - strike, 0) if is_call else max(strike - avg, 0)
    elif payoff_type == "lookback":
        payoff = max(float(np.max(spot)) - strike, 0) if is_call else max(strike - float(np.min(spot)), 0)
    else:
        payoff = max(S_T - strike, 0)

    # Running P&L = cash + delta * S - payoff_so_far (we show mark-to-market)
    for name in strategies:
        s = strategies[name]
        for t in range(n_steps + 1):
            stock_val = s["delta"][t] * float(spot[t])
            # Approximate mark-to-market: cash + stock - BS option value at t
            tau_t = max(T - t * dt, 0)
            if tau_t > 0:
                opt_val = float(BlackScholes.price(
                    float(spot[t]), strike, tau_t, r, sigma, "call" if is_call else "put"))
            else:
                opt_val = payoff
            s["pnl"][t] = s["cash"][t] + stock_val - opt_val

    return strategies, premium, payoff


# ═══════════════════════════════════════════════════════════════════
#  Animation builder
# ═══════════════════════════════════════════════════════════════════

def build_animation(
    spot_paths, var_paths, strike, T, r, tc_bps, iv, rho,
    deep_policy=None, hero_idx=None, product="vanilla",
    fps=24, duration_sec=8, model_label="rBergomi",
):
    """
    Build a 4-panel matplotlib animation.

    Returns (fig, anim) — call anim.save(...) to write to file.
    """
    n_paths, total_len = spot_paths.shape
    n_steps = total_len - 1
    dt = T / n_steps
    days = np.arange(total_len) * dt * 365

    # Hero path (median terminal spot for a "normal" path)
    if hero_idx is None:
        hero_idx = int(np.argsort(spot_paths[:, -1])[n_paths // 2])
    hero_spot = spot_paths[hero_idx]
    hero_var  = var_paths[hero_idx]

    # Compute step-by-step for hero
    strats, premium, payoff = compute_step_by_step(
        hero_spot, hero_var, strike, T, r, tc_bps, iv, rho,
        deep_policy=deep_policy, payoff_type=product,
    )

    # ── Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_spot, ax_var, ax_delta, ax_pnl = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    _apply_dark_theme(fig, axes.ravel())
    fig.subplots_adjust(hspace=0.30, wspace=0.25, left=0.07, right=0.95,
                        top=0.90, bottom=0.08)

    # ── Static elements ──────────────────────────────────────────
    # Ghost paths (background)
    ghost_indices = np.random.choice(n_paths, min(40, n_paths), replace=False)
    for gi in ghost_indices:
        ax_spot.plot(days, spot_paths[gi], color=SPOT_COLOR, alpha=GHOST, linewidth=0.3)
        ax_var.plot(days, np.sqrt(np.clip(var_paths[gi], 1e-8, None)) * 100,
                    color=VAR_COLOR, alpha=GHOST, linewidth=0.3)

    # Strike line
    ax_spot.axhline(strike, color=STRIKE_COL, linestyle="--", linewidth=0.8,
                    alpha=0.5, label=f"K = {strike:.0f}")
    ax_pnl.axhline(0, color=STRIKE_COL, linestyle="--", linewidth=0.5, alpha=0.5)

    # Axis limits
    s_pad = float(np.std(spot_paths[:, -1])) * 0.3
    ax_spot.set_xlim(days[0], days[-1])
    ax_spot.set_ylim(hero_spot.min() - s_pad, hero_spot.max() + s_pad)
    ax_var.set_xlim(days[0], days[-1])
    vol_hero = np.sqrt(np.clip(hero_var, 1e-8, None)) * 100
    ax_var.set_ylim(0, vol_hero.max() * 1.5)
    ax_delta.set_xlim(days[0], days[-1])
    ax_delta.set_ylim(-0.2, 1.2)
    pnl_all = np.concatenate([strats[n]["pnl"] for n in strats])
    pnl_pad = max(abs(pnl_all.max()), abs(pnl_all.min())) * 1.3
    ax_pnl.set_xlim(days[0], days[-1])
    ax_pnl.set_ylim(-pnl_pad, pnl_pad)

    # Labels
    ax_spot.set_ylabel("Spot Price", fontsize=9)
    ax_var.set_ylabel("σ (%)", fontsize=9)
    ax_delta.set_ylabel("Hedge Ratio δ", fontsize=9)
    ax_delta.set_xlabel("Days", fontsize=9)
    ax_pnl.set_ylabel("Hedging P&L", fontsize=9)
    ax_pnl.set_xlabel("Days", fontsize=9)

    # Title
    title_text = fig.suptitle("", fontsize=14, color=TEXT_COLOR, fontweight="bold")

    # ── Animated lines ───────────────────────────────────────────
    # (No patheffects — too slow for frame-by-frame rendering)

    line_spot, = ax_spot.plot([], [], color=SPOT_COLOR, linewidth=2.0, solid_capstyle="round")
    dot_spot,  = ax_spot.plot([], [], "o", color=SPOT_COLOR, markersize=6, zorder=5)

    line_var,  = ax_var.plot([], [], color=VAR_COLOR, linewidth=1.8, solid_capstyle="round")
    dot_var,   = ax_var.plot([], [], "o", color=VAR_COLOR, markersize=5, zorder=5)

    line_deep, = ax_delta.plot([], [], color=ACCENT, linewidth=2.2,
                               label="Deep Hedge", solid_capstyle="round")
    line_bs,   = ax_delta.plot([], [], color=BS_COLOR, linewidth=1.4,
                               label="BS Δ", alpha=0.85, linestyle="-")
    line_bart, = ax_delta.plot([], [], color=BART_COLOR, linewidth=1.4,
                               label="Bartlett", alpha=0.85, linestyle="--")
    ax_delta.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG,
                    edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    pnl_deep, = ax_pnl.plot([], [], color=ACCENT, linewidth=2.2,
                             label="Deep", solid_capstyle="round")
    pnl_bs,   = ax_pnl.plot([], [], color=BS_COLOR, linewidth=1.4,
                             label="BS", alpha=0.85)
    pnl_bart, = ax_pnl.plot([], [], color=BART_COLOR, linewidth=1.4,
                             label="Bartlett", alpha=0.85, linestyle="--")
    ax_pnl.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG,
                  edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # P&L annotations
    ann_deep = ax_pnl.annotate("", xy=(0, 0), fontsize=9, color=ACCENT,
                                fontweight="bold")
    ann_bs   = ax_pnl.annotate("", xy=(0, 0), fontsize=9, color=BS_COLOR)
    ann_bart = ax_pnl.annotate("", xy=(0, 0), fontsize=9, color=BART_COLOR)

    # ── Animation frames ─────────────────────────────────────────
    total_frames = int(fps * duration_sec)
    # Map frames → data indices (smooth slow start, fast end)
    frame_to_idx = np.clip(
        np.round(np.linspace(0, n_steps, total_frames) ** 0.85
                 / n_steps ** 0.85 * n_steps).astype(int),
        0, n_steps,
    )

    def init():
        for l in [line_spot, dot_spot, line_var, dot_var,
                  line_deep, line_bs, line_bart,
                  pnl_deep, pnl_bs, pnl_bart]:
            l.set_data([], [])
        return []

    def update(frame):
        idx = frame_to_idx[frame]
        d = days[:idx + 1]

        # Spot
        line_spot.set_data(d, hero_spot[:idx + 1])
        dot_spot.set_data([days[idx]], [hero_spot[idx]])

        # Variance → vol %
        v = np.sqrt(np.clip(hero_var[:idx + 1], 1e-8, None)) * 100
        line_var.set_data(d, v)
        dot_var.set_data([days[idx]], [v[-1]])

        # Deltas
        line_deep.set_data(d, strats["deep"]["delta"][:idx + 1])
        line_bs.set_data(d, strats["bs"]["delta"][:idx + 1])
        line_bart.set_data(d, strats["bartlett"]["delta"][:idx + 1])

        # P&L
        pnl_deep.set_data(d, strats["deep"]["pnl"][:idx + 1])
        pnl_bs.set_data(d, strats["bs"]["pnl"][:idx + 1])
        pnl_bart.set_data(d, strats["bartlett"]["pnl"][:idx + 1])

        # Annotations (current P&L value)
        y_offset = pnl_pad * 0.05
        for ann, name, color in [(ann_deep, "deep", ACCENT),
                                  (ann_bs, "bs", BS_COLOR),
                                  (ann_bart, "bartlett", BART_COLOR)]:
            val = strats[name]["pnl"][idx]
            ann.set_text(f" {val:+.2f}")
            ann.xy = (days[idx], val)
            ann.set_position((days[idx] + 0.3, val + y_offset))
            y_offset -= pnl_pad * 0.08

        # Title
        pct = idx / n_steps * 100
        spot_val = hero_spot[idx]
        title_text.set_text(
            f"Deep Hedging under {model_label}  ·  "
            f"{product.capitalize()} Call  ·  "
            f"Day {days[idx]:.1f}/{days[-1]:.0f}  ·  "
            f"S = {spot_val:,.0f}"
        )

        return [line_spot, dot_spot, line_var, dot_var,
                line_deep, line_bs, line_bart,
                pnl_deep, pnl_bs, pnl_bart,
                ann_deep, ann_bs, ann_bart, title_text]

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, blit=False, interval=1000 / fps,
    )
    return fig, anim


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hedging Animation")
    parser.add_argument("--model", choices=["bergomi", "gbm"], default="bergomi",
                        help="Volatility model for path generation")
    parser.add_argument("--product", choices=["vanilla", "asian", "lookback"],
                        default="vanilla")
    parser.add_argument("--paths", type=int, default=50,
                        help="Number of ghost paths in background")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Animation duration in seconds")
    parser.add_argument("--gif", action="store_true",
                        help="Save as GIF instead of MP4")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs for deep hedger")
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--skip-training", action="store_true",
                        help="Use BS delta as 'deep' (no training)")
    parser.add_argument("--dpi", type=int, default=100,
                        help="DPI for output (lower = faster)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 15 paths, 4s, 10fps, 30 epochs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (auto-named if omitted)")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.paths = 15
        args.duration = 4.0
        args.fps = 10
        args.epochs = 30
        args.n_train = 4096
        args.dpi = 80

    cfg = load_config()
    out_dir = Path("outputs/paper_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    S0      = 5500.0
    strike  = 5500.0
    T       = 30 / 365
    n_steps = 63
    r       = cfg.get("pricing", {}).get("risk_free_rate", 0.045)
    xi0     = cfg["bergomi"]["xi0"]
    iv      = float(np.sqrt(xi0))
    rho     = cfg["bergomi"]["rho"]
    tc_bps  = cfg.get("hedging", {}).get("transaction_cost_bps", 2.0)

    model_label = "rBergomi" if args.model == "bergomi" else "GBM"

    print("=" * 60)
    print(f"  HEDGING ANIMATION  ·  {model_label}  ·  {args.product}")
    print("=" * 60)

    # ── Generate paths ───────────────────────────────────────────
    print(f"\n  Generating {args.paths} paths ({model_label})...")
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)
    spot_paths, var_paths = generate_paths(
        args.model, cfg, args.paths, T, n_steps, S0, k1)
    print(f"  Shape: {spot_paths.shape}")

    # ── Train deep hedger ────────────────────────────────────────
    deep_policy = None
    if not args.skip_training:
        print(f"\n  Training deep hedger ({args.epochs} epochs)...")
        st_train, vt_train = generate_paths(
            args.model, cfg, args.n_train, T, n_steps, S0, k2)

        dh = DeepHedger(
            spot=S0, strike=strike, T=T, r=r, iv=iv,
            opt_type="call", tc_bps=tc_bps,
            risk_measure="entropic", risk_param=1.0,
            payoff_type=args.product, rho=rho,
        )
        dh.train(st_train, vt_train, n_epochs=args.epochs,
                 batch_size=min(4096, args.n_train), verbose=True)
        deep_policy = dh.policy
        print("  Training complete.")
    else:
        print("  Skipping training (deep = BS fallback)")

    # ── Build animation ──────────────────────────────────────────
    print("\n  Building animation frames...")
    t0 = time.time()
    fig, anim = build_animation(
        spot_paths, var_paths, strike, T, r, tc_bps, iv, rho,
        deep_policy=deep_policy,
        product=args.product,
        fps=args.fps,
        duration_sec=args.duration,
        model_label=model_label,
    )

    # ── Save ─────────────────────────────────────────────────────
    if args.output:
        out_file = args.output
    else:
        ext = "gif" if args.gif else "mp4"
        out_file = str(out_dir / f"hedging_animation_{args.product}.{ext}")

    total_frames = int(args.fps * args.duration)
    print(f"  Rendering {total_frames} frames → {out_file}")

    # Progress callback
    _frame_count = [0]
    def _progress(current_frame, total):
        _frame_count[0] += 1
        if _frame_count[0] % 10 == 0 or _frame_count[0] == total:
            pct = _frame_count[0] / total * 100
            print(f"    frame {_frame_count[0]:>4d}/{total}  ({pct:.0f}%)", flush=True)

    if args.gif:
        writer = animation.PillowWriter(fps=args.fps)
    else:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000,
                                         extra_args=["-vcodec", "libx264"])
    try:
        anim.save(out_file, writer=writer, dpi=args.dpi,
                  progress_callback=_progress)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  ({out_file})")
    except Exception as e:
        # Fallback to GIF if ffmpeg not available
        if not args.gif:
            print(f"  MP4 failed ({e}), falling back to GIF...")
            out_file = str(out_dir / f"hedging_animation_{args.product}.gif")
            writer = animation.PillowWriter(fps=min(args.fps, 15))
            anim.save(out_file, writer=writer, dpi=args.dpi,
                      progress_callback=_progress)
            print(f"  Saved GIF → {out_file}")
        else:
            raise

    # Save a static "last frame" PNG by calling the last update manually
    png_file = str(out_dir / f"hedging_animation_{args.product}_final.png")
    fig.savefig(png_file, dpi=min(args.dpi + 50, 200), facecolor=BG)
    plt.close(fig)
    print(f"  Static frame → {png_file}")

    print("\n  Done! 🎬")


if __name__ == "__main__":
    main()
