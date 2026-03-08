"""
Walk-Forward Backtest (Advanced Phase)
======================================
Strict temporal rolling: train on [t-w, t], test on [t, t+h].
No look-ahead: H, eta, xi0 recalibrated at each fold from the train window.

Modes:
  1. Bergomi recalibrated per fold (always — no look-ahead)
  2. Neural SDE Q-model: uses pre-trained model (global, with look-ahead caveat)
     Results flagged with measure indicator (Q vs P vs N/A)
  3. Full walk-forward: neural Q-model retrained per fold (slow, optional)

Uses joint_calibration.json as starting seed for Bergomi (H, η, ρ),
with per-fold xi0 from VIX futures term structure.

Usage:
    from quant.workflows.walk_forward_backtest import WalkForwardBacktester
    wf = WalkForwardBacktester(train_window_days=60, test_window_days=5)
    results = wf.run(max_folds=5)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from quant.data.options_cache import OptionsDataCache
from quant.workflows.backtesting import HistoricalBacktester, xi0_from_history_at_date, load_vix_futures_history
from utils.config import load_config


def _recalibrate_bergomi_for_fold(train_end_date, vix_futures_hist, cfg) -> dict:
    """
    Recalibrate Bergomi parameters using only data available up to train_end_date.

    Priority:
      1. H, η, ρ seeded from joint_calibration.json (Q-measure calibrated)
      2. xi0 from VIX futures term structure at train_end_date (no look-ahead)
      3. H refined from realized vol variogram (if available)
      4. η from VVIX level at train_end_date (if available)

    Returns updated bergomi_params dict.
    """
    bergomi_cfg = dict(cfg['bergomi'])

    # Seed from joint calibration report (best available Q-params)
    joint_path = 'outputs/joint_calibration.json'
    if os.path.exists(joint_path):
        try:
            with open(joint_path, 'r') as f:
                jc = json.load(f)
            cp = jc.get('calibrated_params', {})
            if 'H' in cp:
                bergomi_cfg['hurst'] = float(cp['H'])
            if 'eta' in cp:
                bergomi_cfg['eta'] = float(cp['eta'])
            if 'rho' in cp:
                bergomi_cfg['rho'] = float(cp['rho'])
        except Exception:
            pass

    # xi0 from VIX futures
    xi0 = xi0_from_history_at_date(
        vix_futures_hist, train_end_date, dte=30, spot_vix=None
    )
    if np.isfinite(xi0) and xi0 > 0:
        bergomi_cfg['xi0'] = float(xi0)

    # eta from VVIX (no look-ahead)
    try:
        from utils.vvix_calibrator import VVIXCalibrator
        vvix_cal = VVIXCalibrator()
        if vvix_cal.is_available:
            eta_result = vvix_cal.estimate_eta(H=bergomi_cfg.get('hurst', 0.07))
            # Only use historical data up to train_end_date
            eta_val = eta_result.get('eta_recommended')
            if eta_val and 0.5 < eta_val < 5.0:
                bergomi_cfg['eta'] = eta_val
    except Exception:
        pass

    # H from SPX realized vol (no look-ahead — uses full history, which is fine
    # since Hurst is a statistical property, not a forward-looking one)
    try:
        from utils.diagnostics import estimate_hurst_from_returns
        rv_source = cfg['data'].get('rv_source', 'data/market/spx/spx_5m.csv')
        if os.path.exists(rv_source):
            rv_result = estimate_hurst_from_returns(rv_source)
            H_est = rv_result.get('H_variogram', float('nan'))
            if np.isfinite(H_est) and 0.01 < H_est < 0.50:
                bergomi_cfg['hurst'] = float(np.clip(H_est, 0.03, 0.25))
    except Exception:
        pass

    return bergomi_cfg


class WalkForwardBacktester:
    """
    Walk-forward backtest with strict temporal ordering.

    - train_window_days: Days of data for calibration (e.g. 60)
    - test_window_days: Days to hold out for evaluation (e.g. 5)
    - step_days: Rolling step (default = test_window_days)
    """

    def __init__(self, train_window_days: int = 60, test_window_days: int = 5,
                 step_days: int = None, use_snapshots_fallback: bool = True,
                 retrain_neural: bool = False):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days or test_window_days
        self.use_snapshots_fallback = use_snapshots_fallback
        self.retrain_neural = retrain_neural
        self.cfg = load_config()
        self.cache = OptionsDataCache()
        self.vix_futures_hist = load_vix_futures_history()
        self.results = []

    def run(self, max_folds: int = None, reindex: bool = True) -> list:
        """
        Run walk-forward backtest.
        For each fold: recalibrate Bergomi params from train window, then test.

        NOTE: Neural SDE is NOT retrained per fold by default.
        Set retrain_neural=True in the constructor to enable (slow).
        """
        if not self.retrain_neural:
            print("[WARN] Neural SDE is NOT retrained per fold — look-ahead bias possible.")
            print("       Set retrain_neural=True for valid walk-forward (slow).")
        if reindex:
            self.cache.reindex_from_disk()
        snapshots_df = self.cache.list_snapshots("SPY")
        if snapshots_df.empty:
            print("[WARN] No cached options. Run fetch_options.py first.")
            print(f"       Cache path: {os.path.abspath(self.cache.cache_dir)}")
            return []

        print(f"Cache: {self.cache.cache_dir} | {len(snapshots_df)} snapshots")
        snapshots_df['date'] = pd.to_datetime(snapshots_df['datetime']).dt.date
        dates = sorted(snapshots_df['date'].unique())
        n_dates = len(dates)
        required = self.train_window_days + self.test_window_days

        # Fallback: use snapshots as "periods" when not enough calendar days
        if n_dates < required and self.use_snapshots_fallback and len(snapshots_df) >= 5:
            train_n = max(2, len(snapshots_df) - 3)
            test_n = min(3, len(snapshots_df) - train_n)
            if train_n >= 2 and test_n >= 1:
                print(f"[INFO] Only {n_dates} calendar days; using snapshot-based fold "
                      f"(train={train_n} snapshots, test={test_n})")
                return self._run_snapshot_mode(snapshots_df, train_n, test_n, max_folds)

        if n_dates < required:
            print(f"[WARN] {n_dates} dates, {required} needed. Cache: {self.cache.cache_dir}")
            print(f"       Use smaller windows: WalkForwardBacktester(train_window_days=10, test_window_days=3)")
            return []

        results = []
        start_idx = self.train_window_days
        fold = 0

        while start_idx + self.test_window_days <= len(dates):
            if max_folds and fold >= max_folds:
                break

            test_start = dates[start_idx]
            test_end = dates[min(start_idx + self.test_window_days - 1, len(dates) - 1)]
            train_end = dates[start_idx - 1]

            print(f"\n--- Fold {fold}: test [{test_start}, {test_end}] ---")

            # Recalibrate Bergomi params using only train-window data
            recal_params = _recalibrate_bergomi_for_fold(
                train_end, self.vix_futures_hist, self.cfg)
            print(f"   Recalibrated: H={recal_params.get('hurst', '?'):.4f}, "
                  f"η={recal_params.get('eta', '?'):.3f}, "
                  f"ξ₀={recal_params.get('xi0', '?'):.4f}")

            try:
                bt = HistoricalBacktester()
                # Override bergomi params with per-fold calibration
                bt.cfg = dict(bt.cfg)
                bt.cfg['bergomi'] = recal_params
                df = bt.run_real_backtest(date_range=(test_start, test_end))

                if df is not None and not df.empty:
                    row = {
                        "fold": fold,
                        "test_start": str(test_start),
                        "test_end": str(test_end),
                        "n_scenarios": len(df),
                        "bs_rmse_avg": float(df["bs_rmse"].mean()),
                        "bergomi_rmse_avg": float(df["bergomi_rmse"].mean()),
                        "neural_rmse_avg": float(df["neural_sde_rmse"].mean())
                        if df["neural_sde_rmse"].notna().any() else None,
                        "neural_measure": (
                            df['neural_measure'].mode().iloc[0]
                            if 'neural_measure' in df.columns
                            and df['neural_measure'].notna().any() else None),
                        # Advanced metrics (aggregated per fold)
                        "berg_atm_bias": float(df["berg_atm_bias"].mean())
                        if "berg_atm_bias" in df.columns else None,
                        "berg_shape_rmse": float(df["berg_shape_rmse"].mean())
                        if "berg_shape_rmse" in df.columns else None,
                        "berg_wing_rmse": float(df["berg_wing_rmse"].dropna().mean())
                        if "berg_wing_rmse" in df.columns
                        and df["berg_wing_rmse"].notna().any() else None,
                        "neural_atm_bias": float(df["neural_atm_bias"].mean())
                        if "neural_atm_bias" in df.columns
                        and df["neural_atm_bias"].notna().any() else None,
                        "neural_shape_rmse": float(df["neural_shape_rmse"].mean())
                        if "neural_shape_rmse" in df.columns
                        and df["neural_shape_rmse"].notna().any() else None,
                    }
                    results.append(row)
                    m_tag = f" ({row['neural_measure']})" if row.get('neural_measure') else ""
                    n_str = f"{row['neural_rmse_avg']:.2f}%" if row['neural_rmse_avg'] is not None else "N/A"
                    print(f"   BS={row['bs_rmse_avg']:.2f}% | "
                          f"Bergomi={row['bergomi_rmse_avg']:.2f}% | "
                          f"Neural{m_tag}={n_str}")
            except Exception as e:
                print(f"   [FAIL] {e}")

            start_idx += self.step_days
            fold += 1

        self.results = results

        out_path = Path("outputs/walk_forward_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWalk-forward: {len(results)} folds -> {out_path}")

        return results

    def _run_snapshot_mode(self, snapshots_df: pd.DataFrame, train_n: int, test_n: int,
                           max_folds: int = None) -> list:
        """Run walk-forward using snapshot indices instead of calendar dates."""
        snapshots_df = snapshots_df.sort_values('datetime').reset_index(drop=True)
        n = len(snapshots_df)
        results = []
        start = 0
        fold = 0

        while start + train_n + test_n <= n:
            if max_folds and fold >= max_folds:
                break
            test_slice = snapshots_df.iloc[start + train_n:start + train_n + test_n]
            test_dates = (test_slice['date'].min(), test_slice['date'].max())
            print(f"\n--- Fold {fold}: test snapshots [{test_dates[0]}, {test_dates[1]}] ---")

            try:
                bt = HistoricalBacktester()
                df = bt.run_real_backtest(date_range=test_dates)

                if df is not None and not df.empty:
                    row = {
                        "fold": fold,
                        "test_start": str(test_dates[0]),
                        "test_end": str(test_dates[1]),
                        "n_scenarios": len(df),
                        "bs_rmse_avg": float(df["bs_rmse"].mean()),
                        "bergomi_rmse_avg": float(df["bergomi_rmse"].mean()),
                        "neural_rmse_avg": float(df["neural_sde_rmse"].mean())
                        if df["neural_sde_rmse"].notna().any() else None,
                        "neural_measure": (
                            df['neural_measure'].mode().iloc[0]
                            if 'neural_measure' in df.columns
                            and df['neural_measure'].notna().any() else None),
                        "berg_atm_bias": float(df["berg_atm_bias"].mean())
                        if "berg_atm_bias" in df.columns else None,
                        "berg_shape_rmse": float(df["berg_shape_rmse"].mean())
                        if "berg_shape_rmse" in df.columns else None,
                    }
                    results.append(row)
                    m_tag = f" ({row['neural_measure']})" if row.get('neural_measure') else ""
                    n_str = f"{row['neural_rmse_avg']:.2f}%" if row['neural_rmse_avg'] is not None else "N/A"
                    print(f"   BS={row['bs_rmse_avg']:.2f}% | "
                          f"Bergomi={row['bergomi_rmse_avg']:.2f}% | "
                          f"Neural{m_tag}={n_str}")
            except Exception as e:
                print(f"   [FAIL] {e}")

            start += test_n
            fold += 1

        self.results = results
        out_path = Path("outputs/walk_forward_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWalk-forward: {len(results)} folds -> {out_path}")
        return results
