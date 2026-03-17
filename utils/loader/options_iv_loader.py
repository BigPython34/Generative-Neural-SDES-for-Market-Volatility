"""
Options-Based Instantaneous Variance Loader
============================================
Extracts instantaneous variance under Q from cached SPY options surfaces,
using the model-free VIX methodology (CBOE formula) applied per-snapshot.

This solves the fundamental issue with using VIX directly:
  - VIX = sqrt(E^Q[∫₀ᵀ V_t dt]) is a 30-day INTEGRATED variance
  - Training a diffusion on VIX level ≠ training on instantaneous V_t
  - Here we compute the VIX-like integral at multiple tenors, then
    differentiate w.r.t. tenor to recover an instantaneous variance proxy.

Method (Dupire-style from term structure):
  1. For each snapshot, compute model-free implied variance W(T) at several tenors
  2. W(T) = (2/T) Σ_i ΔK_i/K_i² · e^{rT} · OTM_price(K_i, T)   (CBOE VIX formula)
  3. Instantaneous forward variance: v(T) = ∂(T·W(T))/∂T
  4. This gives a time series of v(T) at chosen maturities

When not enough snapshots, falls back to VIX with a correction factor.

References:
  - CBOE VIX White Paper (2019)
  - Dupire (1994): Pricing with a smile
  - Gatheral (2006): The Volatility Surface, Ch. 2
"""

import numpy as np
import pandas as pd
import os
from utils.config import load_config


class OptionsVarianceLoader:
    """
    Extract instantaneous variance from cached options surfaces.
    Falls back to VIX-corrected loader when insufficient options data.
    """

    def __init__(self, config_path: str = "config/params.yaml"):
        self.config = load_config(config_path)
        self.cache_dir = self.config['data'].get('options_cache_dir', 'data/options_cache')
        self.stride_ratio = self.config['data'].get('stride_ratio', 0.5)

    def _load_snapshots(self) -> list:
        """Load and sort all options surface snapshots."""
        meta_path = os.path.join(self.cache_dir, 'index.json')
        snapshots = []

        # Scan for CSV files directly
        for f in sorted(os.listdir(self.cache_dir)):
            if f.endswith('.csv') and f.startswith('SPY_surface_'):
                fpath = os.path.join(self.cache_dir, f)
                try:
                    df = pd.read_csv(fpath)
                    if 'impliedVolatility' in df.columns and 'T' in df.columns:
                        snapshots.append({
                            'path': fpath,
                            'filename': f,
                            'df': df,
                        })
                except Exception:
                    continue

        return snapshots

    def _compute_model_free_variance(self, surface: pd.DataFrame,
                                      r: float = 0.045) -> dict:
        """
        Compute model-free implied variance W(T) at each available tenor
        using the CBOE VIX methodology.

        W(T) = (2/T) Σ_i [ΔK_i / K_i²] · e^{rT} · mid_i

        where sum is over OTM options (puts for K < F, calls for K >= F).

        Returns {T: W(T)} for each available maturity.
        """
        result = {}

        for T_val in sorted(surface['T'].unique()):
            if T_val < 0.005 or T_val > 1.0:  # skip <2d and >1y
                continue

            slice_df = surface[surface['T'] == T_val].copy()
            if len(slice_df) < 5:
                continue

            # Need mid prices
            if 'mid' in slice_df.columns:
                slice_df['price'] = slice_df['mid']
            elif 'bid' in slice_df.columns and 'ask' in slice_df.columns:
                slice_df['price'] = (slice_df['bid'] + slice_df['ask']) / 2
            else:
                continue

            slice_df = slice_df[slice_df['price'] > 0.01].copy()

            # Forward price: use put-call parity at ATM
            atm_mask = (slice_df['moneyness'].abs() < 0.02)
            if atm_mask.sum() == 0:
                continue

            # Use OTM options only
            otm = slice_df[
                ((slice_df['type'] == 'put') & (slice_df['moneyness'] <= 0)) |
                ((slice_df['type'] == 'call') & (slice_df['moneyness'] >= 0))
            ].copy()

            if len(otm) < 3:
                continue

            otm = otm.sort_values('strike')
            K = otm['strike'].values
            prices = otm['price'].values

            # ΔK
            dK = np.zeros_like(K)
            dK[0] = K[1] - K[0]
            dK[-1] = K[-1] - K[-2]
            dK[1:-1] = (K[2:] - K[:-2]) / 2

            # W(T) = (2/T) Σ dK/K² · e^{rT} · price
            W = (2.0 / T_val) * np.sum(dK / K ** 2 * np.exp(r * T_val) * prices)

            # Sanity: W should be a reasonable variance (vol 5%–200%)
            if 0.0025 < W < 4.0:
                result[T_val] = float(W)

        return result

    def _term_structure_to_instantaneous(self, term_structure: dict) -> dict:
        """
        Convert total implied variance W(T) to instantaneous forward variance.

        v(T) = ∂[T · W(T)] / ∂T

        Using finite differences on sorted maturities.
        """
        if len(term_structure) < 2:
            return term_structure  # Can't differentiate

        Ts = np.array(sorted(term_structure.keys()))
        TWs = np.array([T * term_structure[T] for T in Ts])

        # Forward-difference instantaneous variance
        inst = {}
        for i in range(len(Ts) - 1):
            T_mid = (Ts[i] + Ts[i + 1]) / 2
            v_fwd = (TWs[i + 1] - TWs[i]) / (Ts[i + 1] - Ts[i])
            if v_fwd > 0:  # Must be positive (calendar arbitrage free)
                inst[float(T_mid)] = float(v_fwd)

        return inst

    def get_variance_paths(self, segment_length: int = None,
                           target_tenor_days: int = 30) -> np.ndarray:
        """
        Build variance paths from options surfaces.

        For each snapshot, extracts instantaneous variance at target_tenor.
        Creates a time series of v(t) values, then segments into paths.

        Falls back to corrected VIX if insufficient options data.

        Args:
            segment_length: Path length for training
            target_tenor_days: Target tenor for variance extraction (default 30d)

        Returns:
            (n_paths, segment_length) array of variance values
        """
        if segment_length is None:
            segment_length = self.config['data']['segment_length']

        snapshots = self._load_snapshots()

        if len(snapshots) < segment_length:
            print(f"   [OPTIONS] Only {len(snapshots)} snapshots, need {segment_length}")
            print(f"   [OPTIONS] Falling back to VIX-corrected loader")
            return self._fallback_vix_corrected(segment_length)

        # Extract instantaneous variance from each snapshot
        r = self.config['pricing'].get('risk_free_rate', 0.045)
        variance_series = []
        timestamps = []

        for snap in snapshots:
            ts = self._compute_model_free_variance(snap['df'], r=r)
            inst = self._term_structure_to_instantaneous(ts)

            if not inst:
                # Fallback: use ATM IV²
                atm = snap['df'][snap['df']['moneyness'].abs() < 0.03]
                if len(atm) > 0:
                    iv = atm['impliedVolatility'].median()
                    variance_series.append(float(iv ** 2))
                continue

            # Find closest to target tenor
            target_T = target_tenor_days / 365.0
            closest_T = min(inst.keys(), key=lambda t: abs(t - target_T))
            variance_series.append(inst[closest_T])

        variance_series = np.array(variance_series)
        variance_series = variance_series[(variance_series > 0.001) & (variance_series < 5.0)]

        if len(variance_series) < segment_length:
            print(f"   [OPTIONS] Only {len(variance_series)} valid points after filtering")
            return self._fallback_vix_corrected(segment_length)

        # Segment into overlapping paths
        paths = []
        stride = max(1, int(segment_length * self.stride_ratio))
        for i in range(0, len(variance_series) - segment_length, stride):
            paths.append(variance_series[i:i + segment_length])

        if not paths:
            return self._fallback_vix_corrected(segment_length)

        paths = np.array(paths)
        np.random.shuffle(paths)
        print(f"   [OPTIONS] {paths.shape[0]} paths from {len(snapshots)} snapshots")
        return paths

    def _fallback_vix_corrected(self, segment_length: int) -> np.ndarray:
        """
        Fallback: use VIX data but apply a correction for the integrated→instantaneous bias.

        VIX² ≈ E^Q[∫₀ᵀ V_t dt] / T
        For short-term (T ~ 30d), V_t ≈ VIX² · (1 + correction_factor)
        where correction_factor accounts for the term structure slope.

        This is a first-order approximation; proper options-based extraction is preferred.
        """
        from data_loader import MarketDataLoader

        print("   [OPTIONS FALLBACK] Using VIX with instantaneous correction")
        loader = MarketDataLoader()
        paths = np.array(loader.get_realized_vol_paths(segment_length))

        # Apply correction: for H ~ 0.1, the ratio V_inst / VIX² ~ 1.0 + O(η²T^{2H})
        # This is small for T=30d, H=0.1 → correction ~ 3-5%
        # We scale to match the expected short-term forward variance
        correction = 1.03  # Conservative correction
        paths = paths * correction

        print(f"   [OPTIONS FALLBACK] Applied {(correction-1)*100:.0f}% VIX→instantaneous correction")
        return paths
