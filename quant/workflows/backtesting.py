import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Historical Backtesting Module
=============================
Backtests volatility models against REAL SPX options surfaces.
Uses cached market data from data/options_cache/ (Yahoo Finance snapshots).

Models compared:
  1. Black-Scholes (flat ATM vol)
  2. Rough Bergomi (parametric, fBM-driven)
  3. Neural SDE (data-driven, signature-conditioned)

Each model generates MC paths -> prices calls at each (K, T) -> inverts to IV
-> compares against market IV from the real surface.

Key fixes vs previous version:
  - Bergomi: fBM correlation properly scaled (no more jnp.std normalisation)
  - Neural SDE: evaluated at TRAINING dt and time-stretched to target maturity
  - xi0: calibrated from VIX futures term structure (not just ATM^2)
  - Plots: publication-quality, RMSE annotations, percent moneyness axis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.fetcher.options_cache import OptionsDataCache
from quant.models.bergomi import RoughBergomiModel
from quant.models.black_scholes import BlackScholes
from utils.config import load_config
from engine.generative_trainer import GenerativeTrainer

# Q-model support (Girsanov drift correction for risk-neutral pricing)
try:
    from quant.calibration.neural_q import load_q_model
    _HAS_Q_CALIBRATOR = True
except ImportError:
    _HAS_Q_CALIBRATOR = False


def _compute_smile_metrics(moneyness, model_ivs, market_ivs):
    """
    Compute advanced smile metrics beyond vanilla RMSE.

    All returned values are in vol points (%).

    Metrics (ref: Gatheral & Jacquier 2014, De Marco & Henry-Labordère 2015):
      - rmse: overall root-mean-square error
      - atm_bias: model ATM IV - market ATM IV (level error)
      - shape_rmse: RMSE after removing ATM bias (pure curvature/skew error)
      - wing_rmse: RMSE on wings only (|moneyness| > 5%)
      - max_err: worst-case absolute error across strikes
      - vega_rmse: vega-weighted RMSE (economically relevant)
    """
    m = np.asarray(moneyness)
    mod = np.asarray(model_ivs) * 100
    mkt = np.asarray(market_ivs) * 100

    diff = mod - mkt
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # ATM bias (at the closest-to-ATM point)
    atm_idx = int(np.argmin(np.abs(m)))
    atm_bias = float(diff[atm_idx])

    # Shape RMSE (after subtracting ATM level shift)
    shape_diff = diff - atm_bias
    shape_rmse = float(np.sqrt(np.mean(shape_diff ** 2)))

    # Wing RMSE (|moneyness| > 5% — where rough vol matters)
    wing_mask = np.abs(m) > 0.05
    wing_rmse = (float(np.sqrt(np.mean(diff[wing_mask] ** 2)))
                 if wing_mask.sum() >= 2 else np.nan)

    # Max absolute error
    max_err = float(np.max(np.abs(diff)))

    # Vega-weighted RMSE (Gaussian ATM weighting)
    vega_w = np.exp(-0.5 * (m / 0.10) ** 2)
    vega_w /= vega_w.sum()
    vega_rmse = float(np.sqrt(np.sum(vega_w * diff ** 2)))

    return {
        'rmse': rmse,
        'atm_bias': atm_bias,
        'shape_rmse': shape_rmse,
        'wing_rmse': wing_rmse,
        'max_err': max_err,
        'vega_rmse': vega_rmse,
    }


# ===================================================================
#  VIX Futures Term Structure  (for xi0 calibration)
# ===================================================================

def load_vix_term_structure() -> dict:
    """
    Load VIX futures term structure from CBOE data.
    Returns  {dte_days: last_known_close}  for interpolation.
    Prefers vix_futures_all.csv (source of truth), falls back to vix_*_front / vix_futures_*.
    """
    base = "data/cboe_vix_futures_full"
    ts = {}

    # 1) Try vix_futures_all.csv (canonical source)
    all_path = os.path.join(base, "vix_futures_all.csv")
    if os.path.exists(all_path):
        df = pd.read_csv(all_path)
        if "Futures" in df.columns:
            df["expiration_date"] = pd.to_datetime(
                df["Futures"].str.extract(r"\((.*?)\)")[0], format="%b %Y", errors="coerce"
            )
        df = df.dropna(subset=["expiration_date"])
        last_date = df["Trade Date"].max()
        last_group = df[df["Trade Date"] == last_date].sort_values("expiration_date")
        price_col = next((c for c in ["Close", "close", "Settle", "settle"]
                         if c in last_group.columns), None)
        if price_col:
            for i, dte in enumerate([30, 60, 90]):
                if i < len(last_group):
                    val = last_group.iloc[i][price_col]
                    if pd.notna(val) and val > 0:
                        ts[dte] = float(val)
        if ts:
            return ts

    # 2) Fallback: vix_1m_front / vix_futures_front_month etc.
    fallbacks = [
        (30, "vix_1m_front.csv"), (30, "vix_futures_front_month.csv"),
        (60, "vix_2m_front.csv"), (60, "vix_futures_2M.csv"),
        (90, "vix_3m_front.csv"), (90, "vix_futures_3M.csv"),
    ]
    for dte, fname in fallbacks:
        if dte in ts:
            continue
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        price_col = next((c for c in ["close", "Close", "settle", "Settle"]
                         if c in df.columns and df[c].replace(0, np.nan).dropna().shape[0] > 0), None)
        if price_col:
            vals = df[price_col].replace(0, np.nan).dropna()
            if len(vals) > 0:
                ts[dte] = float(vals.iloc[-1])

    return ts


def load_vix_futures_history() -> pd.DataFrame:
    """
    Load full VIX futures history for fair backtesting (no look-ahead).

    Returns a normalized dataframe with columns:
      - trade_date (datetime64[ns])
      - expiration_date (datetime64[ns])
      - dte_days (int)
      - price (float)
    """
    all_path = os.path.join("data", "cboe_vix_futures_full", "vix_futures_all.csv")
    if not os.path.exists(all_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(all_path)
    except Exception:
        return pd.DataFrame()

    if "Trade Date" not in df.columns:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["Trade Date"], errors="coerce")

    if "expiration_date" in df.columns:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    elif "Futures" in df.columns:
        df["expiration_date"] = pd.to_datetime(
            df["Futures"].astype(str).str.extract(r"\((.*?)\)")[0],
            format="%b %Y",
            errors="coerce",
        )
    else:
        return pd.DataFrame()

    price_col = next((c for c in ["Close", "close", "Settle", "settle"] if c in df.columns), None)
    if price_col is None:
        return pd.DataFrame()

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["dte_days"] = (df["expiration_date"] - df["trade_date"]).dt.days

    out = df[["trade_date", "expiration_date", "dte_days", "price"]].copy()
    out = out.dropna()
    out = out[(out["dte_days"] > 0) & (out["price"] > 0)]
    return out.sort_values(["trade_date", "expiration_date"]).reset_index(drop=True)


def xi0_from_term_structure(ts: dict, dte: int, spot_vix: float = None) -> float:
    """
    Interpolate forward variance xi0 from VIX futures.

    VIX futures ~ sqrt(E^Q[V_{0,T}]) for tenor T, so xi0 ~ (VIX_fut / 100)^2.
    """
    if not ts:
        v = spot_vix if spot_vix else 15.0
        return (v / 100.0) ** 2

    dtes = sorted(ts.keys())
    vix_vals = [ts[d] for d in dtes]

    if dte <= dtes[0]:
        v = spot_vix if spot_vix else vix_vals[0]
    elif dte >= dtes[-1]:
        v = vix_vals[-1]
    else:
        v = float(np.interp(dte, dtes, vix_vals))

    return (v / 100.0) ** 2


def xi0_from_history_at_date(hist: pd.DataFrame,
                             snapshot_dt,
                             dte: int,
                             spot_vix: float = None) -> float:
    """
    Fair xi0 estimate using only futures information available at snapshot date.
    """
    if hist is None or hist.empty:
        v = spot_vix if spot_vix else 15.0
        return (v / 100.0) ** 2

    snap = pd.to_datetime(snapshot_dt, errors="coerce")
    if pd.isna(snap):
        v = spot_vix if spot_vix else 15.0
        return (v / 100.0) ** 2

    available = hist[hist["trade_date"] <= snap]
    if available.empty:
        v = spot_vix if spot_vix else 15.0
        return (v / 100.0) ** 2

    last_date = available["trade_date"].max()
    curve = available[available["trade_date"] == last_date].sort_values("dte_days")
    if curve.empty:
        v = spot_vix if spot_vix else 15.0
        return (v / 100.0) ** 2

    x = curve["dte_days"].to_numpy(dtype=float)
    y = curve["price"].to_numpy(dtype=float)

    # Consolidate duplicate DTE rows (multiple contracts could map to same coarse expiry date)
    ux = np.unique(x)
    if ux.size != x.size:
        y2 = []
        for u in ux:
            y2.append(float(np.nanmean(y[x == u])))
        x = ux
        y = np.asarray(y2, dtype=float)

    if x.size == 0:
        v = spot_vix if spot_vix else 15.0
        return (v / 100.0) ** 2
    if x.size == 1:
        v = float(y[0])
        return (v / 100.0) ** 2

    v = float(np.interp(float(dte), x, y, left=y[0], right=y[-1]))
    return (v / 100.0) ** 2


# ===================================================================
#  Backtester
# ===================================================================

class HistoricalBacktester:
    """
    Backtests volatility models against REAL market options data.
    """

    def __init__(self, r: float = None, fair_mode: bool = True, smile_grid_points: int = 41):
        cfg = load_config()
        self.r = r if r is not None else self._resolve_rate(cfg)
        self.cfg = cfg
        self.cache = OptionsDataCache()
        self.backtest_results = None
        self.fair_mode = bool(fair_mode)
        self.smile_grid = np.linspace(-0.15, 0.15, int(smile_grid_points))

        # VIX futures term structure for xi0
        self.vix_ts = load_vix_term_structure()
        self.vix_futures_history = load_vix_futures_history() if self.fair_mode else pd.DataFrame()
        if self.vix_ts:
            print(f"   VIX futures term structure: "
                  + ", ".join(f"{d}d={v:.1f}" for d, v in sorted(self.vix_ts.items())))
        else:
            print("   WARNING: No VIX futures data -> using ATM vol for xi0")

        if self.fair_mode:
            if self.vix_futures_history.empty:
                print("   Fair mode: ON (historical futures unavailable, fallback to ATM/static xi0)")
            else:
                print(f"   Fair mode: ON ({len(self.vix_futures_history):,} VIX futures rows loaded)")
        else:
            print("   Fair mode: OFF (uses latest term structure for all snapshots)")

    @staticmethod
    def _resolve_rate(cfg) -> float:
        """Use real SOFR rate when available, fall back to config."""
        if cfg['pricing'].get('use_sofr', True):
            try:
                from quant.loader.sofr_loader import get_sofr
                sofr = get_sofr()
                if sofr.is_available:
                    return sofr.get_rate()
            except Exception:
                pass
        return cfg['pricing']['risk_free_rate']

    def _compute_xi0(self, snapshot_time, dte: int, atm_vol: float) -> float:
        """Compute xi0 with optional no-look-ahead logic."""
        if self.fair_mode:
            xi0 = xi0_from_history_at_date(
                self.vix_futures_history,
                snapshot_time,
                dte,
                spot_vix=float(atm_vol) * 100,
            )
            if np.isfinite(xi0) and xi0 > 0:
                return float(xi0)

        return float(xi0_from_term_structure(self.vix_ts, dte, spot_vix=atm_vol * 100))

    def _smile_to_grid(self, moneyness_vals, iv_vals) -> np.ndarray:
        """Interpolate one smile onto a fixed moneyness grid for robust plotting."""
        x = np.asarray(moneyness_vals, dtype=float)
        y = np.asarray(iv_vals, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 3:
            return np.full(self.smile_grid.shape, np.nan)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        ux = np.unique(x)
        if ux.size != x.size:
            y2 = []
            for u in ux:
                y2.append(float(np.nanmedian(y[x == u])))
            x = ux
            y = np.asarray(y2, dtype=float)

        if x.size < 3:
            return np.full(self.smile_grid.shape, np.nan)

        yi = np.interp(self.smile_grid, x, y)
        yi[self.smile_grid < x.min()] = np.nan
        yi[self.smile_grid > x.max()] = np.nan

        # Light robust smoothing for readability (visual only)
        ys = yi.copy()
        for i in range(1, len(yi) - 1):
            w = yi[i - 1:i + 2]
            if np.isfinite(w).sum() >= 2:
                ys[i] = np.nanmedian(w)
        return ys

    def _median_smile_by_dte(self, sub_df: pd.DataFrame, key: str) -> np.ndarray:
        curves = []
        for _, row in sub_df.iterrows():
            vals = row.get(key, [])
            if vals is None or len(vals) == 0:
                continue
            curves.append(self._smile_to_grid(row['moneyness'], vals))

        if not curves:
            return np.full(self.smile_grid.shape, np.nan)

        arr = np.vstack(curves)
        return np.nanmedian(arr, axis=0)

    # ---------------------------------------------------------------
    #  Smile extraction
    # ---------------------------------------------------------------
    def _filter_smile(self, surface: pd.DataFrame, target_dte: int,
                      moneyness_range: tuple = (-0.15, 0.15),
                      min_bid: float = 0.05) -> tuple:
        """Extract a clean OTM smile slice for a specific maturity."""
        available_dtes = sorted(surface['dte'].unique())
        closest_dte = min(available_dtes, key=lambda x: abs(x - target_dte))

        smile = surface[surface['dte'] == closest_dte].copy()

        # OTM filter
        otm_mask = (
            ((smile['type'] == 'call') & (smile['moneyness'] >= 0)) |
            ((smile['type'] == 'put') & (smile['moneyness'] <= 0))
        )
        smile = smile[otm_mask]

        # Quality filters
        smile = smile[
            (smile['bid'] >= min_bid) &
            (smile['impliedVolatility'] > 0.05) &
            (smile['impliedVolatility'] < 1.5) &
            (smile['moneyness'] >= moneyness_range[0]) &
            (smile['moneyness'] <= moneyness_range[1])
        ].copy()

        # Dedup by strike
        smile = smile.sort_values('volume', ascending=False, na_position='last')
        smile = smile.drop_duplicates('strike').sort_values('moneyness')

        return smile, closest_dte

    # ---------------------------------------------------------------
    #  MC pricing -> IV
    # ---------------------------------------------------------------
    def _price_with_mc(self, S_T: np.ndarray, strikes: np.ndarray,
                       spot: float, T: float, r: float) -> np.ndarray:
        """Price calls via MC terminal values, invert to IV."""
        ivs = []
        for K in strikes:
            payoff = np.maximum(S_T - K, 0)
            mc_price = np.exp(-r * T) * np.mean(payoff)
            iv = BlackScholes.implied_vol(mc_price, spot, K, T, r, 'call')
            ivs.append(iv if not np.isnan(iv) else 0.0)
        return np.array(ivs)

    # ---------------------------------------------------------------
    #  Neural SDE: evaluate at training dt, time-stretch
    # ---------------------------------------------------------------
    def _neural_sde_terminal_spot(self, neural_model, spot, atm_vol,
                                  T_target, n_mc, rho, rng_key):
        """
        Generate terminal spot prices from the Neural SDE.

        CRITICAL: The Neural SDE is evaluated at its TRAINING dt because
        the OU prior (kappa*(theta-x)*dt) and neural correction (f(sig)*dt)
        were learned at that exact time scale.

        For maturities > T_train:  chain multiple 120-step blocks, carrying
        the signature state across blocks (Chevyrev & Kormilitzin, 2016).
        For maturities < T_train:  use only the first portion.
        """
        import jax
        import jax.numpy as jnp

        sim_cfg = self.cfg['simulation']
        train_n = sim_cfg['n_steps']      # 120
        train_T = sim_cfg['T']            # 0.01832
        train_dt = train_T / train_n

        n_blocks = max(1, int(np.ceil(T_target / train_T)))
        total_steps = n_blocks * train_n

        rng_key, sk_vol, sk_spot = jax.random.split(rng_key, 3)
        dW_vol = jax.random.normal(sk_vol, (n_mc, total_steps)) * jnp.sqrt(train_dt)

        v0_arr = jnp.full(n_mc, atm_vol ** 2)

        if n_blocks == 1:
            var_paths = jax.vmap(
                neural_model.generate_variance_path, in_axes=(0, 0, None)
            )(v0_arr, dW_vol, train_dt)
        else:
            # Chain blocks while propagating signature state to preserve
            # the non-Markovian memory across block boundaries.
            d = 2
            sig_state = (
                jnp.zeros((n_mc, d)),
                jnp.zeros((n_mc, d ** 2)),
                jnp.zeros((n_mc, d ** 3)),
            )

            all_var = []
            v_init = v0_arr
            for b in range(n_blocks):
                dW_block = dW_vol[:, b * train_n:(b + 1) * train_n]

                def _gen_with_state(v0_i, dw_i, s1_i, s2_i, s3_i):
                    return neural_model.generate_variance_path_with_state(
                        v0_i, dw_i, train_dt, init_sig_state=(s1_i, s2_i, s3_i)
                    )

                var_block, new_sig = jax.vmap(
                    _gen_with_state, in_axes=(0, 0, 0, 0, 0)
                )(v_init, dW_block, sig_state[0], sig_state[1], sig_state[2])

                all_var.append(var_block)
                v_init = var_block[:, -1]
                sig_state = new_sig

            var_paths = jnp.concatenate(all_var, axis=1)

        var_paths = jnp.clip(var_paths, 1e-6, 5.0)

        n_price = int(round(T_target / train_dt))
        n_price = max(1, min(n_price, var_paths.shape[1]))
        var_pricing = var_paths[:, :n_price]

        # Use train_dt consistently for both vol and spot noise to avoid
        # correlation bias from mismatched scaling.
        dW_vol_price = dW_vol[:, :n_price]
        z_indep = jax.random.normal(sk_spot, (n_mc, n_price)) * jnp.sqrt(train_dt)
        dw_spot = rho * dW_vol_price + jnp.sqrt(1 - rho**2) * z_indep

        v0_col = jnp.full((n_mc, 1), float(atm_vol) ** 2)
        var_prev = jnp.concatenate([v0_col, var_pricing[:, :-1]], axis=1)

        vol_paths = jnp.sqrt(var_prev)
        log_ret = (self.r - 0.5 * var_prev) * train_dt + vol_paths * dw_spot
        S_T = float(spot) * jnp.exp(jnp.sum(log_ret, axis=1))

        return np.array(S_T)

    # ---------------------------------------------------------------
    #  Main backtest loop
    # ---------------------------------------------------------------
    def run_real_backtest(self, target_dtes: list = None,
                         n_mc: int = None,
                         date_range: tuple = None) -> pd.DataFrame:
        bt_cfg = self.cfg['backtesting']
        if target_dtes is None:
            target_dtes = [7, 17, 31, 45]
        if n_mc is None:
            n_mc = bt_cfg.get('n_mc_paths', 5000)

        moneyness_range = tuple(bt_cfg.get('moneyness_range', [-0.15, 0.15]))

        # Load snapshots
        snapshots_df = self.cache.list_snapshots("SPX")
        if snapshots_df.empty:
            print("ERROR: No cached SPX options data.")
            return pd.DataFrame()

        snapshots = snapshots_df.to_dict('records')

        # Optional date filter for walk-forward
        if date_range is not None:
            start_d, end_d = date_range
            def in_range(r):
                try:
                    dt = pd.to_datetime(r.get('datetime', r.get('timestamp', ''))).date()
                    return start_d <= dt <= end_d
                except Exception:
                    return False
            snapshots = [s for s in snapshots if in_range(s)]
            if not snapshots:
                print(f"No snapshots in date range {start_d} to {end_d}")
                return pd.DataFrame()

        print(f"\n{'='*70}")
        print(f"   REAL OPTIONS BACKTEST")
        print(f"   {len(snapshots)} snapshots x {len(target_dtes)} maturities  |  MC={n_mc}")
        print(f"{'='*70}")

        # Neural SDE — prefer Q-model (Girsanov) over P-model
        import jax
        import jax.numpy as jnp

        sim_cfg = self.cfg['simulation']

        # Try Q-model first (correct for pricing: risk-neutral measure)
        q_model = None
        has_q_model = False
        if _HAS_Q_CALIBRATOR:
            q_model = load_q_model()
            has_q_model = q_model is not None

        # Fallback: P-model (physical measure — biased for pricing)
        config = {'n_steps': sim_cfg['n_steps'], 'T': sim_cfg['T']}
        trainer = GenerativeTrainer(config)
        neural_model = trainer.load_model()
        has_neural_p = neural_model is not None

        has_neural = has_q_model or has_neural_p

        if has_q_model:
            print(f"   Neural SDE: Q-model (Girsanov drift, ρ={q_model.rho:.3f})")
            if has_neural_p:
                print(f"               P-model also available (fallback)")
        elif has_neural_p:
            train_dt = sim_cfg['T'] / sim_cfg['n_steps']
            print(f"   Neural SDE: P-model only (dt={train_dt:.6f} yr)")
            print(f"   ⚠ P-model pricing is biased — run neural_q_calibration for Q-model")
        else:
            print("   WARNING: No trained Neural SDE found.")

        # Bergomi params — priority: joint_calibration > calibration_report > config
        bergomi_cfg = dict(self.cfg['bergomi'])
        bergomi_source = 'config/params.yaml'

        calib_sources = [
            ('outputs/joint_calibration.json', 'calibrated_params',
             {'H': 'hurst', 'eta': 'eta', 'rho': 'rho'}),
            ('outputs/calibration_report.json', 'bergomi_params', None),
            ('outputs/advanced_calibration.json', 'bergomi_params', None),
        ]

        for calib_path, key_name, field_map in calib_sources:
            if not os.path.exists(calib_path):
                continue
            try:
                with open(calib_path, 'r') as f:
                    calib = json.load(f)
                if key_name in calib:
                    params = calib[key_name]
                    if field_map:
                        for src_k, dst_k in field_map.items():
                            if src_k in params:
                                bergomi_cfg[dst_k] = params[src_k]
                    else:
                        bergomi_cfg.update(params)
                    bergomi_source = calib_path
                    break
            except Exception:
                continue

        print(f"   Bergomi: H={bergomi_cfg['hurst']:.4f}, "
              f"η={bergomi_cfg['eta']:.3f}, ρ={bergomi_cfg['rho']:.3f}  "
              f"(from {bergomi_source})")

        rng_key = jax.random.PRNGKey(123)
        results = []

        for snap_idx, snapshot in enumerate(snapshots):
            spot = snapshot['spot']
            snap_time = snapshot.get('datetime', snapshot.get('timestamp', '?'))
            print(f"\n--- Snapshot {snap_idx+1}/{len(snapshots)}: "
                  f"Spot=${spot:.2f} ({snap_time}) ---")

            fpath = os.path.join(self.cache.cache_dir, snapshot['filename'])
            surface = (pd.read_parquet(fpath) if fpath.endswith('.parquet')
                       else pd.read_csv(fpath))

            for target_dte in target_dtes:
                smile, actual_dte = self._filter_smile(
                    surface, target_dte, moneyness_range)

                if len(smile) < 5:
                    print(f"   DTE={target_dte:3d}d: skipped ({len(smile)} opts)")
                    continue

                T = actual_dte / 365.0
                strikes = smile['strike'].values
                market_ivs = smile['impliedVolatility'].values
                moneyness = smile['moneyness'].values

                atm_idx = np.argmin(np.abs(moneyness))
                atm_vol = market_ivs[atm_idx]

                # xi0 from VIX futures term structure (fair mode avoids look-ahead)
                xi0 = self._compute_xi0(snap_time, actual_dte, atm_vol)

                # === 1. Black-Scholes: flat ATM vol ===
                bs_ivs = np.full_like(market_ivs, atm_vol)

                # === 2. Rough Bergomi ===
                n_steps_berg = max(60, actual_dte * 2)
                bergomi_params = {
                    'hurst': bergomi_cfg['hurst'],
                    'eta': bergomi_cfg['eta'],
                    'rho': bergomi_cfg['rho'],
                    'xi0': xi0,
                    'n_steps': n_steps_berg,
                    'T': T,
                    'mu': self.r
                }

                try:
                    bergomi = RoughBergomiModel(bergomi_params)
                    rng_b = jax.random.PRNGKey(snap_idx * 100 + actual_dte)
                    spot_paths, _ = bergomi.simulate_spot_vol_paths(
                        n_mc, s0=spot, mu=self.r, key=rng_b)
                    S_T_berg = np.array(spot_paths)[:, -1]
                    bergomi_ivs = self._price_with_mc(
                        S_T_berg, strikes, spot, T, self.r)
                except Exception as e:
                    print(f"   Bergomi error: {e}")
                    bergomi_ivs = bs_ivs.copy()

                # === 3. Neural SDE ===
                neural_measure = 'N/A'
                if has_q_model:
                    # Q-model (Girsanov): correct for pricing (risk-neutral)
                    try:
                        rng_key, sk = jax.random.split(rng_key)
                        init_log_v = float(np.log(max(atm_vol ** 2, 1e-10)))
                        n_q_steps = max(1, int(round(T * 252)))
                        dt_q = 1.0 / 252.0
                        _, spot_paths_q, _ = q_model.simulate(
                            init_log_v, n_q_steps, dt_q, sk, n_mc
                        )
                        S_T_neural = float(spot) * np.array(spot_paths_q[:, -1])
                        neural_ivs = self._price_with_mc(
                            S_T_neural, strikes, spot, T, self.r)
                        neural_measure = 'Q'
                    except Exception as e:
                        print(f"   Neural Q-model error: {e}")
                        neural_ivs = np.full_like(market_ivs, np.nan)
                elif has_neural_p:
                    # P-model fallback (biased for pricing)
                    try:
                        rng_key, sk = jax.random.split(rng_key)
                        S_T_neural = self._neural_sde_terminal_spot(
                            neural_model, spot, atm_vol, T, n_mc,
                            rho=bergomi_cfg['rho'], rng_key=sk)
                        neural_ivs = self._price_with_mc(
                            S_T_neural, strikes, spot, T, self.r)
                        neural_measure = 'P'
                    except Exception as e:
                        print(f"   Neural P-model error: {e}")
                        neural_ivs = np.full_like(market_ivs, np.nan)
                else:
                    neural_ivs = np.full_like(market_ivs, np.nan)

                # === Advanced Smile Metrics ===
                bs_metrics = _compute_smile_metrics(moneyness, bs_ivs, market_ivs)
                berg_metrics = _compute_smile_metrics(moneyness, bergomi_ivs, market_ivs)

                bs_rmse = bs_metrics['rmse']
                berg_rmse = berg_metrics['rmse']

                candidates = [('BS', bs_rmse), ('Bergomi', berg_rmse)]
                if not np.any(np.isnan(neural_ivs)):
                    neural_metrics = _compute_smile_metrics(
                        moneyness, neural_ivs, market_ivs)
                    neural_rmse = neural_metrics['rmse']
                    candidates.append(('Neural', neural_rmse))
                else:
                    neural_rmse = np.nan
                    neural_metrics = {k: np.nan for k in bs_metrics}

                best = min(candidates, key=lambda x: x[1])[0]

                results.append({
                    'snapshot': snap_idx + 1,
                    'spot': spot,
                    'dte': actual_dte,
                    'n_options': len(smile),
                    'atm_vol': atm_vol * 100,
                    'xi0': xi0,
                    'bs_rmse': bs_rmse,
                    'bergomi_rmse': berg_rmse,
                    'neural_sde_rmse': neural_rmse,
                    'neural_measure': neural_measure,
                    'best_model': best,
                    # Advanced metrics per model
                    'bs_atm_bias': bs_metrics['atm_bias'],
                    'bs_shape_rmse': bs_metrics['shape_rmse'],
                    'bs_wing_rmse': bs_metrics['wing_rmse'],
                    'berg_atm_bias': berg_metrics['atm_bias'],
                    'berg_shape_rmse': berg_metrics['shape_rmse'],
                    'berg_wing_rmse': berg_metrics['wing_rmse'],
                    'neural_atm_bias': neural_metrics.get('atm_bias', np.nan),
                    'neural_shape_rmse': neural_metrics.get('shape_rmse', np.nan),
                    'neural_wing_rmse': neural_metrics.get('wing_rmse', np.nan),
                    'neural_max_err': neural_metrics.get('max_err', np.nan),
                    'neural_vega_rmse': neural_metrics.get('vega_rmse', np.nan),
                    'berg_max_err': berg_metrics['max_err'],
                    'berg_vega_rmse': berg_metrics['vega_rmse'],
                    # Raw data for plots
                    'moneyness': moneyness.tolist(),
                    'market_iv': (market_ivs * 100).tolist(),
                    'bs_iv': (bs_ivs * 100).tolist(),
                    'bergomi_iv': (bergomi_ivs * 100).tolist(),
                    'neural_iv': ((neural_ivs * 100).tolist()
                                  if not np.any(np.isnan(neural_ivs)) else []),
                })

                n_str = f"{neural_rmse:.2f}%" if not np.isnan(neural_rmse) else "N/A"
                m_tag = f"({neural_measure})" if neural_measure != 'N/A' else ""
                print(f"   DTE={actual_dte:3d}d | {len(smile):2d} opts | "
                      f"ATM={atm_vol*100:.1f}% | xi0={xi0:.4f} | "
                      f"BS={bs_rmse:.2f}% | Berg={berg_rmse:.2f}% | "
                      f"Neural{m_tag}={n_str} | \u2192 {best}")

        self.backtest_results = pd.DataFrame(results)
        self._print_summary()
        return self.backtest_results

    # ---------------------------------------------------------------
    #  Summary
    # ---------------------------------------------------------------
    def _print_summary(self):
        df = self.backtest_results
        if df is None or len(df) == 0:
            return

        print(f"\n{'='*70}")
        print(f"   BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"\n   Scenarios: {len(df)}")

        # Neural SDE measure indicator
        if 'neural_measure' in df.columns:
            measures = df['neural_measure'].dropna().unique()
            q_tag = ' (Q-Girsanov)' if 'Q' in measures else ' (P-model ⚠)'
        else:
            q_tag = ''

        print(f"\n   Average RMSE (vol points):")
        print(f"      Black-Scholes : {df['bs_rmse'].mean():.2f}%")
        print(f"      Bergomi       : {df['bergomi_rmse'].mean():.2f}%")
        if df['neural_sde_rmse'].notna().any():
            print(f"      Neural SDE{q_tag} : {df['neural_sde_rmse'].mean():.2f}%")

        # Advanced metrics
        print(f"\n   ATM Bias (model - market, vol pts):")
        print(f"      BS: {df['bs_atm_bias'].mean():+.2f}% | "
              f"Bergomi: {df['berg_atm_bias'].mean():+.2f}%", end='')
        if df.get('neural_atm_bias') is not None and df['neural_atm_bias'].notna().any():
            print(f" | Neural: {df['neural_atm_bias'].mean():+.2f}%")
        else:
            print()

        print(f"\n   Shape RMSE (curvature/skew only, after ATM bias removal):")
        print(f"      BS: {df['bs_shape_rmse'].mean():.2f}% | "
              f"Bergomi: {df['berg_shape_rmse'].mean():.2f}%", end='')
        if df.get('neural_shape_rmse') is not None and df['neural_shape_rmse'].notna().any():
            print(f" | Neural: {df['neural_shape_rmse'].mean():.2f}%")
        else:
            print()

        wing_bs = df['bs_wing_rmse'].dropna()
        wing_berg = df['berg_wing_rmse'].dropna()
        if len(wing_bs) > 0:
            print(f"\n   Wing RMSE (|m| > 5% — rough vol advantage zone):")
            print(f"      BS: {wing_bs.mean():.2f}% | "
                  f"Bergomi: {wing_berg.mean():.2f}%", end='')
            if df.get('neural_wing_rmse') is not None and df['neural_wing_rmse'].notna().any():
                print(f" | Neural: {df['neural_wing_rmse'].dropna().mean():.2f}%")
            else:
                print()

        print(f"\n   RMSE by Maturity:")
        for dte in sorted(df['dte'].unique()):
            sub = df[df['dte'] == dte]
            n_str = (f"{sub['neural_sde_rmse'].mean():.2f}%"
                     if sub['neural_sde_rmse'].notna().any() else "N/A")
            print(f"      DTE={dte:3d}d: BS={sub['bs_rmse'].mean():.2f}% | "
                  f"Berg={sub['bergomi_rmse'].mean():.2f}% | Neural={n_str}")

        print(f"\n   Win Rate:")
        for model, count in df['best_model'].value_counts().items():
            print(f"      {model}: {count}/{len(df)} ({100*count/len(df):.0f}%)")

    # ---------------------------------------------------------------
    #  Visualisation
    # ---------------------------------------------------------------
    def plot_backtest_results(self) -> go.Figure:
        """Publication-quality backtest visualisation."""
        df = self.backtest_results
        if df is None or len(df) == 0:
            return None

        unique_dtes = sorted(df['dte'].unique())
        n_cols = len(unique_dtes)

        titles = [f'DTE = {d} days' for d in unique_dtes]
        titles += ['RMSE by Maturity', 'Win Rate'] + [''] * max(0, n_cols - 2)

        row2 = [{}] + [{"type": "pie"}] + [None] * max(0, n_cols - 2)

        fig = make_subplots(
            rows=2, cols=n_cols,
            subplot_titles=titles[:2 * n_cols],
            specs=[[{}] * n_cols, row2],
            vertical_spacing=0.18,
            horizontal_spacing=0.06
        )

        C = {'market': '#FFFFFF', 'BS': '#888888',
             'Bergomi': '#00E676', 'Neural': '#00BCD4'}

        # Row 1: median smiles on fixed moneyness grid (all snapshots)
        for ci, dte in enumerate(unique_dtes):
            sub = df[df['dte'] == dte]
            m = self.smile_grid * 100

            market_curve = self._median_smile_by_dte(sub, 'market_iv')
            bs_curve = self._median_smile_by_dte(sub, 'bs_iv')
            berg_curve = self._median_smile_by_dte(sub, 'bergomi_iv')
            neural_curve = self._median_smile_by_dte(sub, 'neural_iv')

            fig.add_trace(go.Scatter(
                x=m, y=market_curve, mode='lines',
                name='Market', line=dict(color=C['market'], width=2.5),
                legendgroup='mkt', showlegend=(ci == 0)
            ), row=1, col=ci + 1)

            fig.add_trace(go.Scatter(
                x=m, y=bs_curve, mode='lines',
                name='Black-Scholes',
                line=dict(color=C['BS'], dash='dash', width=1.5),
                legendgroup='bs', showlegend=(ci == 0)
            ), row=1, col=ci + 1)

            fig.add_trace(go.Scatter(
                x=m, y=berg_curve, mode='lines',
                name='Rough Bergomi',
                line=dict(color=C['Bergomi'], width=2),
                legendgroup='berg', showlegend=(ci == 0)
            ), row=1, col=ci + 1)

            if np.isfinite(neural_curve).any():
                fig.add_trace(go.Scatter(
                    x=m, y=neural_curve, mode='lines',
                    name='Neural SDE',
                    line=dict(color=C['Neural'], width=2),
                    legendgroup='neural', showlegend=(ci == 0)
                ), row=1, col=ci + 1)

            # RMSE annotation
            ann = (f"BS {sub['bs_rmse'].mean():.1f}%<br>"
                   f"Berg {sub['bergomi_rmse'].mean():.1f}%")
            if sub['neural_sde_rmse'].notna().any():
                ann += f"<br>Neural {sub['neural_sde_rmse'].mean():.1f}%"
            fig.add_annotation(
                text=ann, showarrow=False,
                xref=f'x{ci+1}', yref=f'y{ci+1}',
                x=0.02, y=0.98, xanchor='left', yanchor='top',
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0.6)', borderpad=4,
                row=1, col=ci + 1
            )

            fig.update_xaxes(title_text='Moneyness (%)',
                             row=1, col=ci + 1, tickfont=dict(size=10), range=[-15, 15])
            fig.update_yaxes(
                title_text='IV (%)' if ci == 0 else '',
                row=1, col=ci + 1, tickfont=dict(size=10))

        # Row 2 col 1: RMSE bars
        labels = [f"{d}d" for d in unique_dtes]
        bs_m = [df[df['dte'] == d]['bs_rmse'].mean() for d in unique_dtes]
        bg_m = [df[df['dte'] == d]['bergomi_rmse'].mean() for d in unique_dtes]

        fig.add_trace(go.Bar(
            x=labels, y=bs_m, name='BS', marker_color=C['BS'],
            showlegend=False, text=[f"{v:.1f}" for v in bs_m],
            textposition='outside', textfont=dict(size=9)
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=labels, y=bg_m, name='Bergomi', marker_color=C['Bergomi'],
            showlegend=False, text=[f"{v:.1f}" for v in bg_m],
            textposition='outside', textfont=dict(size=9)
        ), row=2, col=1)

        if df['neural_sde_rmse'].notna().any():
            nm = [df[df['dte'] == d]['neural_sde_rmse'].mean()
                  for d in unique_dtes]
            fig.add_trace(go.Bar(
                x=labels, y=nm, name='Neural', marker_color=C['Neural'],
                showlegend=False, text=[f"{v:.1f}" for v in nm],
                textposition='outside', textfont=dict(size=9)
            ), row=2, col=1)

        fig.update_yaxes(title_text='RMSE (vol pts %)', row=2, col=1)

        # Row 2 col 2: pie
        wc = df['best_model'].value_counts()
        fig.add_trace(go.Pie(
            labels=wc.index, values=wc.values,
            marker_colors=[C.get(m, '#FF9800') for m in wc.index],
            textinfo='label+percent', hole=0.45,
            textfont=dict(size=12)
        ), row=2, col=2)

        fig.update_layout(
            title=dict(text='<b>Backtest: Model IV vs Real SPX Options</b>',
                       font=dict(size=18)),
            template='plotly_dark',
            height=750, width=350 * n_cols,
            barmode='group', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='center', x=0.5, font=dict(size=11)),
            margin=dict(l=60, r=30, t=90, b=50)
        )

        return fig

    # ---------------------------------------------------------------
    #  Save
    # ---------------------------------------------------------------
    def save_results(self, filepath: str = None):
        if self.backtest_results is None or len(self.backtest_results) == 0:
            return
        if filepath is None:
            filepath = self.cfg.get('outputs', {}).get(
                'backtest', 'outputs/backtest_results.json')

        df = self.backtest_results
        cols = ['snapshot', 'spot', 'dte', 'n_options', 'atm_vol', 'xi0',
                'bs_rmse', 'bergomi_rmse', 'neural_sde_rmse', 'neural_measure',
                'best_model',
                'bs_atm_bias', 'bs_shape_rmse', 'bs_wing_rmse',
                'berg_atm_bias', 'berg_shape_rmse', 'berg_wing_rmse',
                'neural_atm_bias', 'neural_shape_rmse', 'neural_wing_rmse']
        cols = [c for c in cols if c in df.columns]

        # Load Bergomi source info
        bergomi_source = 'config'
        for src_path in ['outputs/joint_calibration.json',
                         'outputs/calibration_report.json',
                         'outputs/advanced_calibration.json']:
            if os.path.exists(src_path):
                bergomi_source = src_path
                break

        out = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Real SPX options (Yahoo Finance cache)',
            'fair_mode': self.fair_mode,
            'bergomi_params_source': bergomi_source,
            'vix_term_structure': {str(k): v for k, v in self.vix_ts.items()},
            'n_scenarios': len(df),
            'summary': {
                'bs_mean_rmse': float(df['bs_rmse'].mean()),
                'bergomi_mean_rmse': float(df['bergomi_rmse'].mean()),
                'neural_sde_mean_rmse': (
                    float(df['neural_sde_rmse'].mean())
                    if df['neural_sde_rmse'].notna().any() else None),
                'neural_measure': (
                    df['neural_measure'].mode().iloc[0]
                    if 'neural_measure' in df.columns
                    and df['neural_measure'].notna().any() else None),
            },
            'advanced_metrics': {
                'bs_atm_bias': float(df['bs_atm_bias'].mean())
                    if 'bs_atm_bias' in df.columns else None,
                'berg_atm_bias': float(df['berg_atm_bias'].mean())
                    if 'berg_atm_bias' in df.columns else None,
                'neural_atm_bias': (
                    float(df['neural_atm_bias'].mean())
                    if 'neural_atm_bias' in df.columns
                    and df['neural_atm_bias'].notna().any() else None),
                'bs_shape_rmse': float(df['bs_shape_rmse'].mean())
                    if 'bs_shape_rmse' in df.columns else None,
                'berg_shape_rmse': float(df['berg_shape_rmse'].mean())
                    if 'berg_shape_rmse' in df.columns else None,
                'neural_shape_rmse': (
                    float(df['neural_shape_rmse'].mean())
                    if 'neural_shape_rmse' in df.columns
                    and df['neural_shape_rmse'].notna().any() else None),
                'berg_wing_rmse': float(df['berg_wing_rmse'].dropna().mean())
                    if 'berg_wing_rmse' in df.columns else None,
                'neural_wing_rmse': (
                    float(df['neural_wing_rmse'].dropna().mean())
                    if 'neural_wing_rmse' in df.columns
                    and df['neural_wing_rmse'].notna().any() else None),
            },
            'win_rates': df['best_model'].value_counts().to_dict(),
            'by_maturity': {
                str(d): {
                    'bs_rmse': float(df[df['dte']==d]['bs_rmse'].mean()),
                    'bergomi_rmse': float(df[df['dte']==d]['bergomi_rmse'].mean()),
                }
                for d in sorted(df['dte'].unique())
            },
            'daily_results': df[cols].to_dict('records')
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n   Results saved to {filepath}")


# ===================================================================
#  Entry point
# ===================================================================

def run_backtest():
    print("=" * 70)
    print("   REAL OPTIONS BACKTEST: BS vs Bergomi vs Neural SDE")
    print("=" * 70)

    bt = HistoricalBacktester()
    results = bt.run_real_backtest(
        target_dtes=[7, 17, 31, 45],
    )

    if len(results) > 0:
        print("\nGenerating plots...")
        fig = bt.plot_backtest_results()
        if fig:
            os.makedirs("outputs/plots", exist_ok=True)
            fig.write_html("outputs/plots/backtest_smiles.html")
            print("   Saved: outputs/plots/backtest_smiles.html")
            try:
                fig.show()
            except Exception:
                pass  # browser rendering may fail in some environments
        bt.save_results()

    return results
