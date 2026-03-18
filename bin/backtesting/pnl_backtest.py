"""
Variance Risk Premium (VRP) P&L Backtest
=========================================
A *real* P&L backtest that trades the Variance Risk Premium:
  - Model predicts future Realized Volatility (RV)
  - Market prices Implied Volatility (VIX as proxy for 30-day ATM IV)
  - When model_RV < VIX  -> sell variance (short straddle, delta-hedged)
  - When model_RV > VIX  -> buy variance (long straddle, delta-hedged)
  - P&L realized over the next 21 trading days

Theory (Carr & Madan 1998):
  Hedged P&L of a short straddle ~ integral of 0.5 * Gamma * S^2 * (sigma_impl^2 - sigma_real^2) dt
  Simplification for ATM straddle:
    P&L ~ Vega * (IV - RV)   (in vol points)
    P&L ~ Gamma * S^2 * (IV^2 - RV^2) * T / 2   (in $)

What this tests:
  - Does the Neural SDE / Bergomi model predict future RV better than VIX?
  - Can the VRP be harvested systematically?
  - Compares: always-sell, model-conditional-sell, BS flat-vol strategies

Data needed (all available):
    - SPX daily prices  (data/market/equity_indices/spx_daily.csv)
    - SPX 5m intraday   (data/market/equity_indices/spx_5m.csv) for RV estimation
    - VIX daily          (data/market/volatility/vix_daily.csv)
  - SOFR rates         (data/rates/sofr_daily_nyfed.csv) optional

References:
  - Carr & Wu (2009): "Variance Risk Premiums"
  - Bollerslev, Tauchen & Zhou (2009): "Expected Stock Returns and Variance Risk Premia"

Usage:
  python bin/backtest/pnl_backtest.py
  python bin/backtest/pnl_backtest.py --start 2020-01-01 --end 2025-12-31
  python bin/backtest/pnl_backtest.py --model neural --horizon 21
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from utils.loader.realized_variance import (
    compute_forward_annualized_rv_from_prices,
    compute_rolling_annualized_rv_from_prices,
    compute_single_day_annualized_rv_from_intraday_df,
)


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def _load_spx_daily() -> pd.DataFrame:
    """Load SPX daily OHLCV with datetime index.
    Prefers TradingView (150+ years) over Yahoo."""
    paths = [
        ROOT / "data" / "trading_view" / "equity_indices" / "spx_daily.csv",
        ROOT / "data" / "market" / "equity_indices" / "spx_daily.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No SPX daily data found")

    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise KeyError(f"Cannot find date column in SPX daily. Columns: {list(df.columns)}")

    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def _load_vix_daily() -> pd.DataFrame:
    """Load VIX daily with datetime index.
    Prefers TradingView (36 years, 1990-present) over Yahoo."""
    paths = [
        ROOT / "data" / "trading_view" / "volatility" / "vix_daily.csv",
        ROOT / "data" / "market" / "volatility" / "vix_daily.csv",
    ]

    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No VIX daily data found")

    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def _load_spx_intraday_5m() -> Optional[pd.DataFrame]:
    """Load SPX 5-minute data for precise RV computation.
    Prefers TradingView SPX 5m (2yr) over Yahoo."""
    paths = [
        ROOT / "data" / "trading_view" / "equity_indices" / "spx_5m.csv",
        ROOT / "data" / "market" / "equity_indices" / "spx_5m.csv",
        ROOT / "data" / "trading_view" / "equity_etfs" / "spy_5m.csv",
    ]

    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
    return None


def _load_sofr() -> Optional[pd.DataFrame]:
    """Load SOFR rates if available."""
    p = ROOT / "data" / "rates" / "sofr_daily_nyfed.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "date" in df.columns and "rate" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            return df
    return None


def _load_tv_daily(subpath: str) -> Optional[pd.DataFrame]:
    """Load a TradingView daily CSV -> df with 'date' and 'close'."""
    p = ROOT / "data" / "trading_view" / subpath
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "time" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def _load_regime_signals() -> dict:
    """Load VVIX, VIX3M, SKEW, PCR daily from TradingView.
    Returns dict of {signal_name: DataFrame with date+close}."""
    signals = {}
    mapping = {
        "vvix": "volatility/vvix_daily.csv",
        "vix3m": "volatility/vix3m_daily.csv",
        "vix6m": "volatility/vix6m_daily.csv",
        "vix9d": "volatility/vix9d_daily.csv",
        "skew": "sentiment/skew_daily.csv",
        "pcr_spx": "sentiment/pcspx_daily.csv",
    }
    for name, subpath in mapping.items():
        df = _load_tv_daily(subpath)
        if df is not None:
            signals[name] = df
    return signals


# =====================================================================
# 3. MODEL PREDICTIONS
# =====================================================================

# -- Neural SDE model cache (load once, reuse) --
_NEURAL_MODEL_CACHE: dict = {}


def _get_neural_model():
    """Load the trained P-measure Neural SDE model (cached)."""
    if "model" in _NEURAL_MODEL_CACHE:
        return _NEURAL_MODEL_CACHE["model"], _NEURAL_MODEL_CACHE["config"]

    from engine.generative_trainer import GenerativeTrainer
    from utils.config import load_config
    import jax  # noqa: F811

    yaml_config = load_config()
    sim_cfg = yaml_config["simulation"]
    trainer_config = {"n_steps": sim_cfg["n_steps"], "T": sim_cfg["T"]}
    trainer = GenerativeTrainer(trainer_config)
    # Data loaded in __init__
    model = trainer.load_model()  # returns NeuralRoughSimulator
    if model is None:
        raise RuntimeError("No trained P-model found in models/")

    _NEURAL_MODEL_CACHE["model"] = model
    _NEURAL_MODEL_CACHE["config"] = trainer_config
    _NEURAL_MODEL_CACHE["trainer"] = trainer
    return model, trainer_config


def _predict_neural_sde_rv(date_idx: int, spx_daily: pd.DataFrame,
                            vix_level: float, horizon: int = 21) -> Optional[float]:
    """
    Use the trained Neural SDE P-model to predict future RV.

    Strategy:
      1. Initialize at current VIX level (VIX^2 -> variance -> log_v)
      2. Generate n_mc variance paths via model.generate_variance_path()
      3. Average realized variance over the horizon
      4. Return annualized variance

    Returns annualized variance prediction, or None on failure.
    """
    try:
        import jax
        import jax.numpy as jnp

        model, config = _get_neural_model()

        # Initialize at current VIX level (as vol proxy)
        v0 = (vix_level / 100.0) ** 2  # VIX -> variance
        log_v0 = float(np.log(max(v0, 1e-6)))

        n_mc = 2000
        n_steps = config["n_steps"]
        dt = config["T"] / n_steps

        # Vary seed per date to avoid identical predictions
        key = jax.random.PRNGKey(42 + date_idx)
        noise = jax.random.normal(key, (n_mc, n_steps)) * jnp.sqrt(dt)
        init = jnp.full(n_mc, log_v0)

        var_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
            init, noise, dt
        )
        # var_paths is in variance space (exp(log_v))
        # Average over paths and time steps
        mean_var = float(jnp.mean(var_paths))
        return max(mean_var, 1e-8)
    except Exception as e:
        return None


def _predict_bergomi_rv(vix_level: float, horizon_days: int = 21) -> Optional[float]:
    """
    Use calibrated Bergomi model to predict expected variance.
    Under Bergomi, E[V_T] depends on H, eta, xi0.
    """
    try:
        from utils.config import load_config
        bergomi_cfg = load_config()["bergomi"]
        H = bergomi_cfg["hurst"]
        eta = bergomi_cfg["eta"]
        xi0 = bergomi_cfg.get("xi0", (vix_level / 100.0)**2)

        T = horizon_days / 252.0
        # Under rough Bergomi: E[V_T] = xi0 * exp(-0.5 * eta^2 * T^(2H))
        # This is the forward variance in the flat forward var curve case
        expected_var = xi0 * np.exp(-0.5 * eta**2 * T**(2 * H))
        return float(expected_var)
    except Exception:
        return None


# =====================================================================
# 4. STRATEGY & P&L COMPUTATION
# =====================================================================

@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    entry_spot: float
    exit_spot: float
    vix_entry: float          # market IV (annualized vol, %)
    rv_predicted: float       # model-predicted RV (annualized vol, %)
    rv_realized: float        # actual RV over holding period (annualized vol, %)
    signal: str               # "sell_var" or "buy_var" or "flat"
    pnl_per_unit: float       # $ P&L per unit vega
    pnl_bps: float            # P&L in bps of spot
    model_name: str


@dataclass
class BacktestResult:
    model_name: str
    n_trades: int
    total_pnl: float
    mean_pnl: float
    std_pnl: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    mean_vix: float
    mean_rv: float
    mean_vrp: float           # VIX - RV (in vol points)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


def _compute_straddle_pnl(spot_entry: float, spot_exit: float,
                           iv_entry: float, rv_actual: float,
                           T: float, r: float, direction: int) -> float:
    """
    Compute hedged straddle P&L using Carr-Madan (1998) approximation.

    The delta-hedged short-variance P&L is:
      P&L_sell = 0.5 * Gamma_avg * S^2 * (IV^2 - RV^2) * T
    which is positive when IV > RV (the typical VRP harvest).

    Direction convention:
      direction = +1  ->  sell variance (short straddle, collect premium)
      direction = -1  ->  buy variance (long straddle, pay premium)

    ATM gamma ~ N'(0) / (S * sigma * sqrt(T))
    ATM vega  ~ 2 * S * sqrt(T) * N'(0)   (straddle = call + put)

    Returns P&L in $ terms (per 1 straddle notional unit).
    """
    from scipy.stats import norm as sp_norm

    iv = iv_entry / 100.0   # VIX is in %, convert to decimal
    if iv < 0.01:
        return 0.0

    # ATM straddle vega ~ 2 * S * sqrt(T) * N'(0)
    vega_unit = 2.0 * spot_entry * np.sqrt(T) * sp_norm.pdf(0)

    # Variance difference: positive when IV > RV -> profit for seller
    var_diff = iv**2 - rv_actual   # IV^2 - RV (annualized variances)

    # Carr-Madan: hedged PnL ~ 0.5 * Gamma * S^2 * var_diff * T
    # Using vega form: Gamma * S ~ Vega / (S * sigma * T), so
    # PnL ~ 0.5 * (Vega / sigma) * var_diff
    # direction > 0 => sell variance => profit from positive var_diff
    pnl_dollar = direction * 0.5 * vega_unit / iv * var_diff

    return pnl_dollar


def _detect_regime_at_date(row, merged, regime_signals: dict) -> dict:
    """Detect regime at a specific backtest date using pre-loaded signals."""
    try:
        from quant.regimes.regime_detector import RegimeDetector
        detector = RegimeDetector.__new__(RegimeDetector)
        detector.cfg = {"bergomi": {"hurst": 0.07, "eta": 1.9, "rho": -0.7}}

        date = row["date"]
        vix = row["vix_close"]

        # Look up signal values at this date
        vvix_val = _lookup_signal(regime_signals.get("vvix"), date)
        vix3m_val = _lookup_signal(regime_signals.get("vix3m"), date)
        skew_val = _lookup_signal(regime_signals.get("skew"), date)
        pcr_val = _lookup_signal(regime_signals.get("pcr_spx"), date)

        # RV from merged data
        rv_val = row.get("rv_rolling_21d", np.nan)

        return detector.detect_from_values(
            vix=vix, vvix=vvix_val, vix3m=vix3m_val,
            skew=skew_val, pcr=pcr_val, rv_annual=rv_val,
        )
    except Exception:
        return {"regime": "normal", "confidence": 0.0}


def _lookup_signal(df: Optional[pd.DataFrame], date) -> Optional[float]:
    """Find closest value in a signal DataFrame for a given date."""
    if df is None:
        return None
    mask = df["date"] <= date
    if not mask.any():
        return None
    return float(df.loc[mask.values, "close"].iloc[-1])


def run_vrp_backtest(
    spx_daily: pd.DataFrame,
    vix_daily: pd.DataFrame,
    spx_intraday: Optional[pd.DataFrame] = None,
    model_name: str = "bergomi",
    horizon: int = 21,
    start_date: str = None,
    end_date: str = None,
    threshold_vol_pts: float = 0.0,  # minimum VRP to trade (in vol pts)
    risk_free_rate: float = 0.045,
    regime_signals: Optional[dict] = None,
) -> BacktestResult:
    """
    Run the VRP backtest for a given model.

    Steps:
    1. Align SPX + VIX on common dates
    2. For each date (non-overlapping windows of `horizon` days):
       a. Read VIX (market IV proxy)
       b. Compute model's predicted RV
       c. If VRP = VIX - model_RV > threshold -> sell variance
       d. Compute realized P&L over next `horizon` days
    3. Aggregate results
    """
    # Merge SPX + VIX
    merged = pd.merge(
        spx_daily[["date", "close"]].rename(columns={"close": "spx_close"}),
        vix_daily[["date", "close"]].rename(columns={"close": "vix_close"}),
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    # Date range filtering
    if start_date:
        merged = merged[merged["date"] >= pd.Timestamp(start_date)]
    if end_date:
        merged = merged[merged["date"] <= pd.Timestamp(end_date)]
    merged = merged.reset_index(drop=True)

    if len(merged) < horizon + 5:
        raise ValueError(f"Not enough data: {len(merged)} rows for horizon={horizon}")

    # Compute historical rolling RV for "historical" model
    merged["rv_rolling_21d"] = compute_rolling_annualized_rv_from_prices(
        merged["spx_close"].values,
        window_days=21,
        trading_days_per_year=252.0,
    )

    # Pre-compute intraday RV if available
    intraday_rv_cache = {}
    if spx_intraday is not None:
        print("  Pre-computing intraday RV for available dates ...")
        for dt in merged["date"].unique():
            day_data = spx_intraday[spx_intraday["datetime"].dt.date == pd.Timestamp(dt).date()]
            rv = compute_single_day_annualized_rv_from_intraday_df(
                day_data,
                price_col="close",
                trading_days_per_year=252.0,
                min_bars=10,
            )
            if rv is not None:
                intraday_rv_cache[pd.Timestamp(dt).date()] = rv

    T = horizon / 252.0
    trades = []
    equity = [0.0]
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0

    print(f"\n  Running VRP backtest: model={model_name}, horizon={horizon}d")
    print(f"  Date range: {merged['date'].iloc[0].date()} -> {merged['date'].iloc[-1].date()}")
    print(f"  Total trading days: {len(merged)}")

    # Non-overlapping windows (step = horizon)
    idx = max(21, merged["rv_rolling_21d"].first_valid_index() or 21)
    n_trades = 0

    while idx + horizon < len(merged):
        row = merged.iloc[idx]
        spot = row["spx_close"]
        vix = row["vix_close"]
        signal = None  # will be set by model or trade decision block

        # Skip if VIX is suspicious
        if np.isnan(vix) or vix < 5 or vix > 100:
            idx += horizon
            continue

        # Forward realized variance (ground truth)
        fwd_rv = compute_forward_annualized_rv_from_prices(
            merged["spx_close"].values,
            start_idx=int(idx),
            horizon_days=int(horizon),
            trading_days_per_year=252.0,
        )
        if fwd_rv is None:
            break

        # Model-predicted RV
        if model_name == "always_sell":
            # Always sell variance — pure VRP harvest
            predicted_rv = 0.0  # dummy (always below VIX)
            signal = "sell_var"
        elif model_name == "historical":
            # Use trailing 21d RV as forecast
            predicted_rv = row.get("rv_rolling_21d", np.nan)
            if np.isnan(predicted_rv):
                idx += horizon
                continue
        elif model_name == "bergomi":
            predicted_rv = _predict_bergomi_rv(vix, horizon)
            if predicted_rv is None:
                idx += horizon
                continue
        elif model_name == "neural":
            predicted_rv = _predict_neural_sde_rv(idx, merged, vix, horizon)
            if predicted_rv is None:
                idx += horizon
                continue
        elif model_name == "regime_neural":
            # Regime-conditional: use RegimeDetector signals to filter,
            # Neural SDE for RV prediction
            predicted_rv = _predict_neural_sde_rv(idx, merged, vix, horizon)
            if predicted_rv is None:
                # Fallback to bergomi if neural fails
                predicted_rv = _predict_bergomi_rv(vix, horizon)
                if predicted_rv is None:
                    idx += horizon
                    continue
            # Regime filter: skip selling variance in stressed/crisis
            regime_info = _detect_regime_at_date(row, merged, regime_signals)
            if regime_info["regime"] in ("stressed", "crisis"):
                # In stress: buy variance (or flat)
                signal = "flat"
            # else: normal flow below
        elif model_name == "regime_bergomi":
            predicted_rv = _predict_bergomi_rv(vix, horizon)
            if predicted_rv is None:
                idx += horizon
                continue
            regime_info = _detect_regime_at_date(row, merged, regime_signals)
            if regime_info["regime"] in ("stressed", "crisis"):
                signal = "flat"
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Predicted vol (annualized)
        predicted_vol = np.sqrt(max(predicted_rv, 1e-8)) * 100  # in %
        iv_vol = vix  # VIX is already in vol % terms

        vrp = iv_vol - predicted_vol  # VRP in vol points

        # Trade decision (respect regime override if signal already set)
        if model_name == "always_sell":
            signal = "sell_var"
            direction = +1   # +1 = sell variance = short straddle
        elif signal == "flat":
            # Regime filter already decided to stay flat
            direction = 0
        elif vrp > threshold_vol_pts:
            signal = "sell_var"
            direction = +1
        elif vrp < -threshold_vol_pts:
            signal = "buy_var"
            direction = -1   # -1 = buy variance = long straddle
        else:
            signal = "flat"
            direction = 0

        # Compute P&L
        exit_row = merged.iloc[idx + horizon]
        exit_spot = exit_row["spx_close"]

        if direction != 0:
            pnl = _compute_straddle_pnl(spot, exit_spot, vix, fwd_rv, T,
                                          risk_free_rate, direction)
        else:
            pnl = 0.0

        pnl_bps = pnl / spot * 10000

        trade = TradeRecord(
            entry_date=str(row["date"].date()),
            exit_date=str(exit_row["date"].date()),
            entry_spot=float(spot),
            exit_spot=float(exit_spot),
            vix_entry=float(vix),
            rv_predicted=float(predicted_vol),
            rv_realized=float(np.sqrt(fwd_rv) * 100),  # vol %
            signal=signal,
            pnl_per_unit=float(pnl),
            pnl_bps=float(pnl_bps),
            model_name=model_name,
        )
        trades.append(trade)

        cum_pnl += pnl_bps
        equity.append(cum_pnl)
        peak = max(peak, cum_pnl)
        dd = peak - cum_pnl
        max_dd = max(max_dd, dd)

        n_trades += 1
        idx += horizon

    if n_trades == 0:
        print("  WARNING: no trades executed!")
        return BacktestResult(
            model_name=model_name, n_trades=0,
            total_pnl=0, mean_pnl=0, std_pnl=0, sharpe=0,
            max_drawdown=0, hit_rate=0, mean_vix=0, mean_rv=0, mean_vrp=0,
        )

    # Aggregate
    pnls = [t.pnl_bps for t in trades if t.signal != "flat"]
    pnls_arr = np.array(pnls) if pnls else np.array([0.0])
    hits = sum(1 for p in pnls if p > 0)
    vix_arr = [t.vix_entry for t in trades]
    rv_arr = [t.rv_realized for t in trades]

    mean_pnl = float(np.mean(pnls_arr))
    std_pnl = float(np.std(pnls_arr)) if len(pnls_arr) > 1 else 0.0
    sharpe = mean_pnl / std_pnl * np.sqrt(252 / horizon) if std_pnl > 0 else 0.0

    result = BacktestResult(
        model_name=model_name,
        n_trades=n_trades,
        total_pnl=float(cum_pnl),
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        sharpe=sharpe,
        max_drawdown=float(max_dd),
        hit_rate=float(hits / len(pnls)) if pnls else 0.0,
        mean_vix=float(np.mean(vix_arr)),
        mean_rv=float(np.mean(rv_arr)),
        mean_vrp=float(np.mean(vix_arr) - np.mean(rv_arr)),
        trades=[asdict(t) for t in trades],
        equity_curve=equity,
    )

    return result


# =====================================================================
# 5. REPORTING
# =====================================================================

def print_report(results: list[BacktestResult]):
    """Print comparison table across models."""
    print("\n" + "=" * 90)
    print("  VRP P&L BACKTEST RESULTS")
    print("=" * 90)

    header = (
        f"{'Model':<16} {'Trades':>6} {'Total(bps)':>11} {'Mean(bps)':>10} "
        f"{'Std(bps)':>9} {'Sharpe':>7} {'HitRate':>8} {'MaxDD':>8} {'VRP':>6}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        print(
            f"{r.model_name:<16} {r.n_trades:>6} {r.total_pnl:>11.1f} "
            f"{r.mean_pnl:>10.1f} {r.std_pnl:>9.1f} {r.sharpe:>7.2f} "
            f"{r.hit_rate:>7.1%} {r.max_drawdown:>8.1f} {r.mean_vrp:>6.1f}"
        )

    print("-" * 90)
    print("  VRP = mean(VIX) - mean(RV). Positive = variance sellers profit on average.")
    print("  Sharpe annualized. P&L in bps of spot per trade window.")
    print()


def save_results(results: list[BacktestResult], path: Path):
    """Save results to JSON."""
    out = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }
    for r in results:
        out["models"][r.model_name] = {
            "n_trades": r.n_trades,
            "total_pnl_bps": r.total_pnl,
            "mean_pnl_bps": r.mean_pnl,
            "std_pnl_bps": r.std_pnl,
            "sharpe": r.sharpe,
            "max_drawdown_bps": r.max_drawdown,
            "hit_rate": r.hit_rate,
            "mean_vix": r.mean_vix,
            "mean_rv": r.mean_rv,
            "mean_vrp": r.mean_vrp,
            "equity_curve": r.equity_curve,
            "n_sample_trades": len(r.trades[:10]),  # first 10 for inspection
            "sample_trades": r.trades[:10],
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved -> {path.relative_to(ROOT)}")


# =====================================================================
# 6. MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="VRP P&L Backtest")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=21,
                        help="Holding period in trading days (default 21)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Min VRP (vol pts) to trade (default 1.0)")
    parser.add_argument("--models", type=str,
                        default="always_sell,historical,bergomi,regime_bergomi",
                        help="Comma-separated list of models")
    parser.add_argument("--output", type=str, default="outputs/pnl_backtest.json",
                        help="Output JSON path")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("  VRP P&L BACKTEST")
    print("=" * 60)

    # Load data
    print("\n  Loading data ...")
    spx_daily = _load_spx_daily()
    vix_daily = _load_vix_daily()
    spx_intraday = _load_spx_intraday_5m()

    print(f"  SPX daily: {len(spx_daily)} rows "
          f"[{spx_daily['date'].min().date()} -> {spx_daily['date'].max().date()}]")
    print(f"  VIX daily: {len(vix_daily)} rows "
          f"[{vix_daily['date'].min().date()} -> {vix_daily['date'].max().date()}]")
    if spx_intraday is not None:
        print(f"  SPX 5m: {len(spx_intraday)} rows (for intraday RV)")
    else:
        print("  SPX 5m: not available (using daily close-to-close RV)")

    # Load regime signals from TradingView (VVIX, VIX3M, SKEW, PCR)
    regime_signals = _load_regime_signals()
    if regime_signals:
        loaded_names = list(regime_signals.keys())
        print(f"  Regime signals: {', '.join(loaded_names)}")

    # Run all models
    models = [m.strip() for m in args.models.split(",")]
    results = []

    for model in models:
        try:
            r = run_vrp_backtest(
                spx_daily=spx_daily,
                vix_daily=vix_daily,
                spx_intraday=spx_intraday,
                model_name=model,
                horizon=args.horizon,
                start_date=args.start,
                end_date=args.end,
                threshold_vol_pts=args.threshold,
                regime_signals=regime_signals,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR running model '{model}': {e}")
            import traceback
            traceback.print_exc()

    # Report
    if results:
        print_report(results)
        save_results(results, ROOT / args.output)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
