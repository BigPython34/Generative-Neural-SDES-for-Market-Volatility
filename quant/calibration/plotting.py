"""Shared calibration plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _surface_from_model_ivs(model_ivs: dict, spot: float | None = None):
    import pandas as pd

    rows = []
    for T_key in sorted(model_ivs.keys()):
        strikes, model_vals, market_vals = model_ivs[T_key]
        dte = int(round(float(T_key) * 365))
        strikes = np.asarray(strikes, dtype=float)
        model_vals = np.asarray(model_vals, dtype=float)
        market_vals = np.asarray(market_vals, dtype=float)

        for k, m_iv, mk_iv in zip(strikes, model_vals, market_vals):
            rows.append(
                {
                    "dte": dte,
                    "strike": float(k),
                    "model_iv": float(m_iv) if not np.isnan(m_iv) else np.nan,
                    "market_iv": float(mk_iv) if not np.isnan(mk_iv) else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None

    if spot is None or spot <= 0:
        spot = float(np.nanmedian(df["strike"].values))
    df["moneyness"] = np.log(df["strike"] / spot)

    model_surface = df[["dte", "strike", "moneyness", "model_iv"]].rename(
        columns={"model_iv": "impliedVolatility"}
    )
    market_surface = df[["dte", "strike", "moneyness", "market_iv"]].rename(
        columns={"market_iv": "impliedVolatility"}
    )

    model_surface = model_surface[np.isfinite(model_surface["impliedVolatility"])]
    market_surface = market_surface[np.isfinite(market_surface["impliedVolatility"])]
    return model_surface, market_surface


def plot_joint_calibration(result, save_dir: Path, market_surface=None, spot: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from quant.calibration.options_support import VolatilitySurfaceVisualizer

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    market_taus = sorted(result.market_vix_ts.keys())
    model_taus = sorted(result.model_vix_ts.keys())
    all_taus = sorted(set(market_taus) | set(model_taus))
    mkt_vals = [result.market_vix_ts.get(t, np.nan) for t in all_taus]
    mdl_vals = [result.model_vix_ts.get(t, np.nan) for t in all_taus]

    ax.plot(all_taus, mkt_vals, "ko-", markersize=8, linewidth=2, label="Market VIX", zorder=5)
    ax.plot(all_taus, mdl_vals, "rs--", markersize=8, linewidth=2, label="Model VIX", zorder=4)

    for i, tau in enumerate(all_taus):
        if not np.isnan(mkt_vals[i]) and not np.isnan(mdl_vals[i]):
            diff = mdl_vals[i] - mkt_vals[i]
            color = "green" if abs(diff) < 1.0 else "orange" if abs(diff) < 2.0 else "red"
            ax.annotate(f"{diff:+.2f}", (tau, mdl_vals[i]), textcoords="offset points", xytext=(8, 8), fontsize=8, color=color)

    ax.set_xlabel("Tenor (days)", fontsize=12)
    ax.set_ylabel("VIX Level", fontsize=12)
    ax.set_title(f"VIX Term Structure — Model vs Market\nH={result.H:.3f}, η={result.eta:.2f}, ρ={result.rho:.2f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    tick_positions = [t for t in all_taus if t in [1, 9, 30, 90, 180, 365]]
    ax.set_xticks(tick_positions if tick_positions else all_taus)
    ax.set_xticklabels([f"{t}d" for t in (tick_positions if tick_positions else all_taus)])

    fig.tight_layout()
    fig.savefig(save_dir / "joint_calibration_vix_ts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    mat_keys = sorted(result.model_ivs.keys())
    n_mats = min(8, len(mat_keys))
    if len(mat_keys) > n_mats:
        idx = np.linspace(0, len(mat_keys) - 1, n_mats, dtype=int)
        selected_keys = [mat_keys[i] for i in idx]
    else:
        selected_keys = mat_keys

    if n_mats > 0:
        fig, axes = plt.subplots(1, n_mats, figsize=(4.5 * n_mats, 4), squeeze=False)
        for i, T_key in enumerate(selected_keys):
            ax = axes[0, i]
            strikes, model_ivs, market_ivs = result.model_ivs[T_key]
            strikes = np.array(strikes)
            model_ivs = np.array(model_ivs)
            market_ivs = np.array(market_ivs)
            valid_market = ~np.isnan(market_ivs)
            valid_model = ~np.isnan(model_ivs)
            valid_pair = valid_model & valid_market

            # Plot market smile always (even if model fails on some strikes)
            ax.plot(strikes[valid_market], market_ivs[valid_market] * 100, "ko", markersize=4, label="Market", alpha=0.7)

            # Plot model smile where available
            if valid_model.any():
                ax.plot(strikes[valid_model], model_ivs[valid_model] * 100, "r-", linewidth=1.5, label="Model", alpha=0.8)

            dte = int(round(T_key * 365))
            if valid_pair.any():
                rmse = np.sqrt(np.mean((model_ivs[valid_pair] - market_ivs[valid_pair]) ** 2)) * 10000
            else:
                rmse = float('nan')
            ax.set_title(f"T={dte}d (RMSE={rmse:.0f}bps)", fontsize=10)
            ax.set_xlabel("Strike", fontsize=9)
            if i == 0:
                ax.set_ylabel("Implied Vol (%)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Ensure the SPX strike axis extends to include the full market range
            strike_min = float(np.nanmin(strikes[valid_market]))
            strike_max = float(np.nanmax(strikes[valid_market]))
            ax.set_xlim(strike_min, max(strike_max, 900))

        fig.suptitle(f"SPX Smile Fit — H={result.H:.3f}, η={result.eta:.2f}, ρ={result.rho:.2f}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_dir / "joint_calibration_spx_smile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3D surfaces (market vs model)
    viz = VolatilitySurfaceVisualizer()

    model_surface, market_surface_from_result = _surface_from_model_ivs(result.model_ivs, spot=spot)

    if market_surface is not None and len(market_surface) > 0:
        full_mkt = market_surface.copy()
        if "moneyness" not in full_mkt.columns and spot is not None and spot > 0:
            full_mkt["moneyness"] = np.log(full_mkt["strike"] / spot)
        if all(c in full_mkt.columns for c in ["dte", "moneyness", "impliedVolatility"]):
            fig_mkt_3d = viz.plot_3d_surface(full_mkt, "SPX Market Implied Vol Surface (3D)")
            fig_mkt_3d.write_html(save_dir / "joint_calibration_market_surface_3d.html", include_plotlyjs="cdn")
    elif market_surface_from_result is not None and len(market_surface_from_result) > 0:
        fig_mkt_3d = viz.plot_3d_surface(market_surface_from_result, "SPX Market Surface (Calibration Points, 3D)")
        fig_mkt_3d.write_html(save_dir / "joint_calibration_market_surface_3d.html", include_plotlyjs="cdn")

    if model_surface is not None and len(model_surface) > 0:
        fig_model_3d = viz.plot_3d_surface(model_surface, "SPX Model Implied Vol Surface (3D)")
        fig_model_3d.write_html(save_dir / "joint_calibration_model_surface_3d.html", include_plotlyjs="cdn")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    mats_d = result.xi0_maturities * 365
    sigma_implied = np.sqrt(result.xi0_values) * 100
    ax.step(mats_d, sigma_implied, where="post", color="steelblue", linewidth=2, label="ξ₀(t)  →  σ(t)")
    ax.scatter(mats_d, sigma_implied, color="steelblue", s=40, zorder=5)
    ax.set_xlabel("Maturity (days)", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title("Forward Variance Curve ξ₀(t)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "joint_calibration_xi0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_neural_q_calibration(result, save_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(1, len(result.loss_history) + 1)
    ax.semilogy(epochs, result.loss_history, "b-", linewidth=1.5, alpha=0.7)
    added_smoothed = False
    if len(result.loss_history) > 20:
        window = min(20, len(result.loss_history) // 5)
        smoothed = np.convolve(result.loss_history, np.ones(window) / window, mode="valid")
        ax.semilogy(range(window, len(result.loss_history) + 1), smoothed, "r-", linewidth=2, label="Smoothed")
        added_smoothed = True
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Q-Loss", fontsize=12)
    ax.set_title("Neural SDE Q-Calibration — Training Loss", fontsize=14)
    if added_smoothed:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "neural_q_loss_curve.png", dpi=150)
    plt.close(fig)

    if result.model_vix_ts and result.market_vix_ts:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        taus = sorted(set(list(result.model_vix_ts.keys()) + list(result.market_vix_ts.keys())))
        mkt_vals = [result.market_vix_ts.get(t, np.nan) for t in taus]
        mdl_vals = [result.model_vix_ts.get(t, np.nan) for t in taus]
        ax.plot(taus, mkt_vals, "ko-", markersize=8, linewidth=2, label="Market VIX", zorder=5)
        ax.plot(taus, mdl_vals, "rs--", markersize=8, linewidth=2, label="Neural SDE Q-model", zorder=4)
        ax.set_xlabel("Tenor (days)", fontsize=12)
        ax.set_ylabel("VIX Level", fontsize=12)
        ax.set_title("VIX Term Structure — Neural SDE Q vs Market", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "neural_q_vix_fit.png", dpi=150)
        plt.close(fig)

    components = {k: v for k, v in result.loss_components.items() if k != "total"}
    if components:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        names = list(components.keys())
        vals = [components[n] for n in names]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white")
        ax.set_ylabel("Loss Value", fontsize=12)
        ax.set_title("Q-Loss Component Decomposition", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / "neural_q_loss_components.png", dpi=150)
        plt.close(fig)
