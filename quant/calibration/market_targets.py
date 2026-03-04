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
) -> list[dict]:
    """
    Prepare multi-maturity SPX option slices for calibration.

    Filters OTM options, subsamples strikes, computes ATM-centric vega weights.
    Output schema is compatible with both rBergomi and Neural-Q calibrators.
    """
    from scipy.stats import norm as sp_norm

    df = surface_df.copy()

    df = df[
        (df["moneyness"] >= moneyness_range[0])
        & (df["moneyness"] <= moneyness_range[1])
        & (df["dte"] >= min_dte)
        & (df["dte"] <= max_dte)
        & (df["impliedVolatility"] > 0.01)
        & (df["impliedVolatility"] < 2.0)
    ]

    df_otm = df[
        ((df["type"] == "call") & (df["moneyness"] >= 0))
        | ((df["type"] == "put") & (df["moneyness"] < 0))
    ]
    if len(df_otm) < 10:
        df_otm = df

    slices: list[dict] = []
    for dte, group in df_otm.groupby("dte"):
        if len(group) < 3:
            continue

        if len(group) > max_strikes:
            group = group.sort_values("moneyness")
            idx = np.linspace(0, len(group) - 1, max_strikes, dtype=int)
            group = group.iloc[idx]

        T = float(group["T"].iloc[0])
        strikes = group["strike"].values.astype(float)
        market_ivs = group["impliedVolatility"].values.astype(float)
        option_types = group["type"].tolist()

        d1 = (np.log(spot / strikes) + 0.5 * market_ivs**2 * T) / (
            market_ivs * np.sqrt(max(T, 1e-6))
        )
        vega = spot * np.sqrt(max(T, 1e-6)) * sp_norm.pdf(d1)
        vega = np.maximum(vega, 1e-6)

        slices.append(
            {
                "T": T,
                "dte": int(dte),
                "strikes": strikes,
                "market_ivs": market_ivs,
                "option_types": option_types,
                "vega_weights": vega,
            }
        )

    slices.sort(key=lambda x: x["T"])

    if max_maturities is not None and len(slices) > max_maturities:
        idx = np.linspace(0, len(slices) - 1, max_maturities, dtype=int)
        slices = [slices[i] for i in idx]

    return slices
