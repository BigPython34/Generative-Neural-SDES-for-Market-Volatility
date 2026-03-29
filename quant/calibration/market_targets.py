"""Shared market-target preparation helpers for calibration pipelines."""

from __future__ import annotations

from typing import Optional

import numpy as np


def prepare_spx_slices(
    surface_df,
    spot: float,
    max_strikes: int = 8,
    max_maturities: Optional[int] = None,
    min_dte: int = 5,
    max_dte: int = 200,
    moneyness_range: tuple[float, float] = (-0.25, 0.25),
    include_itm: bool = True,
    moneyness_basis: str = "spot",
    moneyness_tenor_widen_alpha: float = 0.0,
    dividend_yield: float = 0.0,
) -> list[dict]:
    """
    Prepare multi-maturity SPX option slices for calibration.

    Filters surface by moneyness/DTE/IV quality, subsamples strikes,
    computes ATM-centric vega weights.
    Output schema is compatible with both rBergomi and Neural-Q calibrators.
    """
    from scipy.stats import norm as sp_norm

    df = surface_df.copy()

    mcol = "moneyness_fwd" if moneyness_basis == "forward" and "moneyness_fwd" in df.columns else "moneyness"
    if mcol not in df.columns:
        mcol = "moneyness"
    df["moneyness_eff"] = df[mcol].astype(float)

    df = df[
        (df["dte"] >= min_dte)
        & (df["dte"] <= max_dte)
        & (df["impliedVolatility"] > 0.01)
        & (df["impliedVolatility"] < 2.0)
    ]

    if include_itm:
        df_selected = df
    else:
        df_selected = df[
            ((df["type"] == "call") & (df["moneyness_eff"] >= 0))
            | ((df["type"] == "put") & (df["moneyness_eff"] < 0))
        ]
        if len(df_selected) < 10:
            df_selected = df

    slices: list[dict] = []
    for dte, group in df_selected.groupby("dte"):
        if len(group) < 3:
            continue

        T = float(group["T"].iloc[0])
        widen = float(moneyness_tenor_widen_alpha) * np.sqrt(max(T, 1e-8))
        lo = float(moneyness_range[0]) - widen
        hi = float(moneyness_range[1]) + widen
        group = group[(group["moneyness_eff"] >= lo) & (group["moneyness_eff"] <= hi)]
        if len(group) < 3:
            continue

        if len(group) > max_strikes:
            group = group.sort_values("moneyness_eff")
            idx = np.linspace(0, len(group) - 1, max_strikes, dtype=int)
            group = group.iloc[idx]

        strikes = group["strike"].values.astype(float)
        market_ivs = group["impliedVolatility"].values.astype(float)
        option_types = group["type"].tolist()
        market_mids = group["mid"].values.astype(float) if "mid" in group.columns else np.full(len(group), np.nan)
        market_spread_pct = (
            group["spread_pct"].values.astype(float) if "spread_pct" in group.columns else np.full(len(group), np.nan)
        )

        fwd = float(spot) * np.exp((0.0 - float(dividend_yield)) * T)
        d1 = (np.log(fwd / strikes) + 0.5 * market_ivs**2 * T) / (
            market_ivs * np.sqrt(max(T, 1e-6))
        )
        vega = fwd * np.sqrt(max(T, 1e-6)) * sp_norm.pdf(d1)
        vega = np.maximum(vega, 1e-6)

        slices.append(
            {
                "T": T,
                "dte": int(dte),
                "strikes": strikes,
                "market_ivs": market_ivs,
                "option_types": option_types,
                "vega_weights": vega,
                "market_mids": market_mids,
                "market_spread_pct": market_spread_pct,
            }
        )

    slices.sort(key=lambda x: x["T"])

    if max_maturities is not None and len(slices) > max_maturities:
        idx = np.linspace(0, len(slices) - 1, max_maturities, dtype=int)
        slices = [slices[i] for i in idx]

    return slices
