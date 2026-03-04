"""
Forward Variance Curve  ξ₀(t)
==============================
Bootstrap a piecewise-constant forward variance curve from the ATM implied
volatility term structure of SPX/SPY options.

Mathematical foundation
-----------------------
In the rough Bergomi model the instantaneous variance is

    V_t = ξ₀(t) · ℰ(η Ŵᴴ)_t

where ξ₀(t) = E^Q[V_t] is the initial *forward variance curve*, fully
determined by today's option prices (Bayer, Friz & Gatheral 2016, §2.1).

From the ATM term structure {σ_ATM(Tᵢ)} we extract ξ₀ via:

    ∫₀ᵀ ξ₀(s) ds  =  σ²_ATM(T) · T          (Dupire / Bergomi)

So the piecewise-constant forward variance on [Tᵢ₋₁, Tᵢ) is

    ξ₀,ᵢ  =  [σ²_ATM(Tᵢ)·Tᵢ − σ²_ATM(Tᵢ₋₁)·Tᵢ₋₁] / (Tᵢ − Tᵢ₋₁)

This is exact: the total-variance identity holds for any local-vol model,
and the rBergomi model's ATM level is controlled by ξ₀(t) alone
(η, ρ, H only affect the smile shape around ATM).

Negative forward variances
--------------------------
If the market ATM term structure is not monotonically increasing in total
variance, the bootstrap can produce ξ₀,ᵢ < 0.  We handle this by:
  1. Smoothing the total variance curve with a monotone cubic spline
  2. Clamping ξ₀,ᵢ ≥ ε (with ε = 1e-6)

References
----------
[1] Bayer, Friz & Gatheral (2016). Pricing under rough volatility. QF.
[2] Bergomi (2005). Smile dynamics II. Risk.
[3] Gatheral (2006). The Volatility Surface, ch. 3.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ForwardVarianceCurve:
    """
    Piecewise-constant forward variance curve.

    Attributes
    ----------
    maturities : array of maturity pillars T₁ < T₂ < … < Tₙ  (years)
    xi_values  : array of forward variances ξ₀(t) on each interval
                 xi_values[i] is the constant value on [T_{i-1}, T_i)
                 with T_0 = 0.
    atm_ivs    : ATM implied vols at each pillar (for diagnostics)
    total_var  : ATM total variance σ²·T at each pillar
    """
    maturities: np.ndarray
    xi_values: np.ndarray
    atm_ivs: np.ndarray
    total_var: np.ndarray
    spot: float = 0.0
    timestamp: str = ""

    def __call__(self, t: float) -> float:
        """Evaluate ξ₀(t) at any time t ≥ 0."""
        if t <= 0:
            return float(self.xi_values[0])
        idx = np.searchsorted(self.maturities, t, side='right')
        idx = min(idx, len(self.xi_values) - 1)
        return float(self.xi_values[idx])

    def evaluate_array(self, t_array: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of ξ₀(t)."""
        indices = np.searchsorted(self.maturities, t_array, side='right')
        indices = np.clip(indices, 0, len(self.xi_values) - 1)
        return self.xi_values[indices]

    def integrated_variance(self, T: float) -> float:
        """Compute ∫₀ᵀ ξ₀(s) ds."""
        if T <= 0:
            return 0.0
        result = 0.0
        t_prev = 0.0
        for i, T_i in enumerate(self.maturities):
            if T_i >= T:
                result += self.xi_values[i] * (T - t_prev)
                return result
            result += self.xi_values[i] * (T_i - t_prev)
            t_prev = T_i
        # Beyond last pillar: extrapolate flat
        result += self.xi_values[-1] * (T - t_prev)
        return result

    def implied_vol_at(self, T: float) -> float:
        """Reconstruct ATM implied vol at maturity T from the curve."""
        iv2T = self.integrated_variance(T)
        if T <= 0 or iv2T <= 0:
            return 0.0
        return np.sqrt(iv2T / T)

    def summary(self) -> str:
        lines = ["Forward Variance Curve ξ₀(t)"]
        lines.append("─" * 55)
        lines.append(f"{'Maturity':>10} {'T(yrs)':>8} {'ξ₀':>10} {'σ_ATM':>8} {'σ²T':>10}")
        lines.append("─" * 55)
        for i in range(len(self.maturities)):
            T = self.maturities[i]
            dte = int(round(T * 365))
            lines.append(
                f"{dte:>8}d  {T:>8.4f} {self.xi_values[i]:>10.6f} "
                f"{self.atm_ivs[i]*100:>7.2f}% {self.total_var[i]:>10.6f}"
            )
        lines.append("─" * 55)
        return "\n".join(lines)


def extract_atm_term_structure(
    surface_df: pd.DataFrame,
    spot: float,
    moneyness_band: float = 0.03,
    min_dte: int = 3,
    max_dte: int = 365,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ATM implied volatilities from an options surface.

    For each maturity slice, select options within |moneyness| < band
    and take the volume/OI-weighted average IV.

    Parameters
    ----------
    surface_df : DataFrame with columns [strike, impliedVolatility, dte, T, moneyness, ...]
    spot : current spot price
    moneyness_band : half-width of ATM band (e.g. 0.03 = ±3%)
    min_dte, max_dte : maturity filters

    Returns
    -------
    (maturities, atm_ivs) : arrays of maturity (years) and ATM implied vol
    """
    df = surface_df.copy()
    df = df[(df['dte'] >= min_dte) & (df['dte'] <= max_dte)]
    df = df[df['impliedVolatility'] > 0.01]  # filter garbage IVs

    # Recompute moneyness if needed
    if 'moneyness' not in df.columns:
        df['moneyness'] = np.log(df['strike'] / spot)

    maturities = []
    atm_ivs = []

    for dte, group in df.groupby('dte'):
        atm = group[group['moneyness'].abs() < moneyness_band]
        if len(atm) < 1:
            # Widen the band
            atm = group.nsmallest(3, 'moneyness', keep='all')
            atm = atm[atm['moneyness'].abs() < 0.10]

        if len(atm) == 0:
            continue

        # Weight by openInterest + volume (liquidity proxy)
        oi = atm.get('openInterest', pd.Series(1, index=atm.index)).fillna(1).values.astype(float)
        vol = atm.get('volume', pd.Series(1, index=atm.index)).fillna(1).values.astype(float)
        weights = oi + vol + 1e-6
        weights = weights / weights.sum()

        atm_iv = np.average(atm['impliedVolatility'].values, weights=weights)
        T = float(atm['T'].iloc[0])

        maturities.append(T)
        atm_ivs.append(atm_iv)

    sort_idx = np.argsort(maturities)
    return np.array(maturities)[sort_idx], np.array(atm_ivs)[sort_idx]


def bootstrap_forward_variance(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
    spot: float = 0.0,
    min_xi: float = 1e-6,
    monotone_smooth: bool = True,
) -> ForwardVarianceCurve:
    """
    Bootstrap piecewise-constant ξ₀(t) from ATM term structure.

    ξ₀,ᵢ = [σ²(Tᵢ)·Tᵢ − σ²(Tᵢ₋₁)·Tᵢ₋₁] / (Tᵢ − Tᵢ₋₁)

    Parameters
    ----------
    maturities : sorted maturity array (years)
    atm_ivs : ATM implied vols at each maturity
    spot : spot price (for metadata)
    min_xi : floor for forward variance
    monotone_smooth : if True, enforce monotone total variance before bootstrap

    Returns
    -------
    ForwardVarianceCurve
    """
    assert len(maturities) == len(atm_ivs), "maturities and atm_ivs must have same length"
    assert len(maturities) >= 2, "Need at least 2 maturities for bootstrapping"

    T = np.array(maturities, dtype=float)
    iv = np.array(atm_ivs, dtype=float)
    total_var = iv ** 2 * T  # σ²(T) · T

    if monotone_smooth:
        # Enforce monotonicity of total variance (calendar arbitrage free)
        for i in range(1, len(total_var)):
            if total_var[i] < total_var[i - 1]:
                total_var[i] = total_var[i - 1] + 1e-8
        # Recompute consistent ATM IVs
        iv = np.sqrt(total_var / T)

    # Bootstrap forward variances
    n = len(T)
    xi = np.zeros(n)

    # First interval [0, T_0]
    xi[0] = total_var[0] / T[0]

    # Subsequent intervals [T_{i-1}, T_i]
    for i in range(1, n):
        dT = T[i] - T[i - 1]
        if dT < 1e-10:
            xi[i] = xi[i - 1]
        else:
            xi[i] = (total_var[i] - total_var[i - 1]) / dT

    # Floor negative forward variances
    xi = np.maximum(xi, min_xi)

    return ForwardVarianceCurve(
        maturities=T,
        xi_values=xi,
        atm_ivs=iv,
        total_var=total_var,
        spot=spot,
    )


def bootstrap_from_surface(
    surface_df: pd.DataFrame,
    spot: float,
    **kwargs,
) -> ForwardVarianceCurve:
    """
    End-to-end: options surface → forward variance curve.

    Parameters
    ----------
    surface_df : options DataFrame from cache
    spot : spot price
    **kwargs : passed to extract_atm_term_structure and bootstrap_forward_variance

    Returns
    -------
    ForwardVarianceCurve
    """
    atm_kwargs = {k: kwargs[k] for k in ['moneyness_band', 'min_dte', 'max_dte'] if k in kwargs}
    boot_kwargs = {k: kwargs[k] for k in ['min_xi', 'monotone_smooth'] if k in kwargs}

    T, ivs = extract_atm_term_structure(surface_df, spot, **atm_kwargs)

    if len(T) < 2:
        raise ValueError(
            f"Only {len(T)} ATM maturity pillars found. "
            f"Need ≥2 for bootstrapping. Check moneyness_band or data quality."
        )

    curve = bootstrap_forward_variance(T, ivs, spot=spot, **boot_kwargs)
    return curve


def bootstrap_xi0_from_vix(
    vix_targets: dict,
    min_xi: float = 1e-6,
) -> ForwardVarianceCurve:
    """
    Bootstrap ξ₀(t) from VIX term structure (CBOE indices).

    In the rBergomi model the VIX at tenor τ satisfies *exactly*:

        E^Q[ VIX²(τ) ] = (1/τ) ∫₀^τ ξ₀(s) ds

    because the exponential martingale ℰ(η Ŵᴴ) has unit expectation.
    This identity is model-parameter-independent (does not depend on
    H, η, or ρ), so ξ₀(t) is uniquely determined by the market VIX
    term structure.

    Inverting:

        ∫₀^τ ξ₀(s) ds = VIX²(τ) · τ  =: TV(τ)

    gives the piecewise-constant forward variance on [τᵢ₋₁, τᵢ):

        ξ₀,ᵢ = [ TV(τᵢ) − TV(τᵢ₋₁) ] / (τᵢ − τᵢ₋₁)

    This is the correct approach for joint SPX-VIX calibration
    (Rømer 2022, Bayer et al. 2016 §4):
      - ξ₀(t) is fixed from VIX (model-free)
      - (H, η, ρ) are then calibrated to SPX smile shape

    Parameters
    ----------
    vix_targets : {tau_days: vix_level}, e.g. {30: 22.66, 90: 23.24, ...}
    min_xi      : floor for forward variance

    Returns
    -------
    ForwardVarianceCurve
    """
    tenors_days = np.array(sorted(vix_targets.keys()), dtype=float)
    tenors_years = tenors_days / 365.0
    vix_levels = np.array([vix_targets[int(t)] for t in tenors_days])

    # VIX level → implied variance
    impl_var = (vix_levels / 100.0) ** 2

    # Total variance: TV(τ) = VIX²(τ) · τ
    total_var = impl_var * tenors_years

    # Enforce monotonicity (calendar-arbitrage-free)
    for i in range(1, len(total_var)):
        if total_var[i] < total_var[i - 1]:
            total_var[i] = total_var[i - 1] + 1e-8

    # Bootstrap forward variance
    n = len(tenors_years)
    xi = np.zeros(n)
    xi[0] = total_var[0] / tenors_years[0]  # = impl_var[0]

    for i in range(1, n):
        dT = tenors_years[i] - tenors_years[i - 1]
        if dT < 1e-10:
            xi[i] = xi[i - 1]
        else:
            xi[i] = (total_var[i] - total_var[i - 1]) / dT

    xi = np.maximum(xi, min_xi)

    # Recompute consistent "IVs" (actually VIX-implied vols)
    consistent_iv = np.sqrt(np.maximum(total_var / tenors_years, min_xi))

    return ForwardVarianceCurve(
        maturities=tenors_years,
        xi_values=xi,
        atm_ivs=consistent_iv,
        total_var=total_var,
        spot=0.0,
    )
